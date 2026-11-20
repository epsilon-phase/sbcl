;;;; Differential test of the elementwise loop vectorizer.

;;;; This software is part of the SBCL system. See the README file for
;;;; more information.
;;;;
;;;; While most of SBCL is derived from the CMU CL system, the test
;;;; files (like this one) were written from scratch after the fork
;;;; from CMU CL.
;;;;
;;;; This software is in the public domain and is provided with
;;;; absolutely no warranty. See the COPYING and CREDITS files for
;;;; more information.

;;; Every loop shape in this file is compiled twice -- once at (OPTIMIZE SPEED),
;;; where the vectorizer runs, and once at (OPTIMIZE (SPEED 0)), where it cannot
;;; (see the manual's "Automatic Vectorization") -- and the two are run over the
;;; same inputs at eleven lengths, comparing every element.  A shape that packs
;;; but computes a different answer fails here, which is how every
;;; wrong-compilation bug in this feature was found.
;;;
;;; The shapes and the bug they once exposed, so that a regression is easy to
;;; place:
;;;
;;;   two loads, one store        the integer store's value is a CAST, and the
;;;                               LOGAND that narrows it is a second wrapper
;;;   load plus scalar            a splat of a tagged negative fixnum needs
;;;                               masking to the lane width before the IMUL
;;;   index as the value          a ramp is 16-byte only, and its index arrives
;;;                               through a cast
;;;   negate                      a packed negate does not wrap, so unary minus
;;;                               on an unsigned lane must be refused
;;;   two stores                  several loops per function, one splice each
;;;   adjustable / (VECTOR T)     AREF inlines to data-vector access behind
;;;                               WITH-ARRAY-DATA
;;;
;;; The lengths straddle every hand-off: 0 runs nothing, 1-9 cross from the packed
;;; loop to the scalar tail, 15/16/17 straddle a second pair of vectors, and
;;; 1001/1007 are longer than any lane width so a lane shift cannot hide.

#-x86-64 (invoke-restart 'run-tests::skip-file)

;;;; run.lisp --- differential test harness for SBCL's automatic loop vectorizer.
;;;;
;;;; It compiles each loop shape twice -- once at (optimize speed), where the
;;;; vectorizer runs, and once at (optimize (speed 0) ...), where it cannot
;;;; (src/compiler/vectorize.lisp:1610-1615 gates the pass on (>= speed 2)) --
;;;; runs both over the same inputs and compares every element.  Each vectorized
;;;; function is run down both runtime dispatch paths by binding
;;;; SB-VM::*AVX2-AVAILABLE*, the flag the emitted code tests at run time
;;;; (src/compiler/x86-64/vec-256.lisp:737 sets SB-C::*VECTOR-WIDE-PREDICATE* to
;;;; SB-VM::AVX2-AVAILABLE-P; src/code/x86-64-vm.lisp:346 defines the flag).
;;;;
;;;; Load it with the built core:
;;;;   src/runtime/sbcl --core output/sbcl.core --noinform --no-sysinit \
;;;;       --no-userinit --disable-debugger --load run.lisp
;;;;
;;;; Exit status 0 means every check passed on both dispatch paths; 1 means a
;;;; check failed (the first failure is printed with its shape, element type,
;;;; operator, length, path, element index and both values); 2 means the values
;;;; all matched but nothing was vectorized, so the comparison proves nothing.

;;; Mnemonics that are evidence of a packed loop.  Taken from the repository's
;;; own tests rather than invented here:
;;;   tests/auto-vectorize.pure.lisp:101-102
;;;       (defparameter *vectorize-packed-mnemonics*
;;;         '("addps" "addpd" "mulps" "movups" "shufps"))
;;;   tests/auto-vectorize.pure.lisp:721 "psubq", :776 "paddq",
;;;       :796 '("paddd" "psubd" "movdqu"), :1167-1171 "psubb" "paddb" "pxor",
;;;       :1397 "xorps", :1399 "xorpd", :1438 "psubw", :1440 "paddw",
;;;       :2200 "pmullw", :2826 "pmulld"
;;;   benchmarks/auto-vectorize-kernels.lisp:184-284 ":packed" lists
;;;       "movdqa" "mulpd" "movupd" "pand" "por"
;;;
;;; "xorps" "xorpd" "pxor" "pand" "por" are named by the tests too, but they are
;;; left out here: scalar code emits `xorps xmm0, xmm0` merely to zero a
;;; register, so they do not distinguish packed code.  Measured: the (speed 0)
;;; reference for "c[i] := (coerce i 'single-float)" contains xorps while being
;;; plainly scalar.  Every loop shape here loads and stores, so the load/store
;;; and arithmetic names below still recognize a packed loop, and a match on
;;; them is evidence no scalar loop can produce.
;;;
;;; A mnemonic is searched as a substring, so a 128-bit name also matches its
;;; VEX/256-bit spelling ("addps" is inside "vaddps") -- the same trick the
;;; tests use at tests/auto-vectorize.pure.lisp:2836.
(defparameter *packed-mnemonics*
  '("addps" "addpd" "mulps" "mulpd" "movups" "movupd" "shufps"
    "paddd" "psubd" "paddq" "psubq" "paddw" "psubw" "paddb" "psubb"
    "pmulld" "pmullw" "movdqu" "movdqa"))

;;; The lengths every case runs over: 0 runs nothing, 1-9 cross the hand-off
;;; from the packed loop to the scalar tail, 15/16/17 straddle a second pair of
;;; vectors, and 1001/1007 are longer than any lane width so a lane shift cannot
;;; hide.  (Mirrors the reasoning of tests/auto-vectorize.pure.lisp:43-48.)
(defparameter *lengths* '(0 1 3 7 8 9 15 16 17 1001 1007))

(defparameter *float-types* '(single-float double-float))
(defparameter *integer-types* '((unsigned-byte 32) (unsigned-byte 16)
                                (unsigned-byte 8)))
(defparameter *element-types* (append *float-types* *integer-types*))
(defparameter *binary-ops* '(+ - *))

;;; Input patterns: positives, negatives, zeros, -0.0, fractions and values near
;;; the top of the integer range, so a lane read from the wrong offset or a
;;; wrapped lane shows up as a mismatch instead of an accidental match.
(defparameter *values*
  `((single-float . #(1.5 -0.0 0.0 3.0 -0.5 7.75 -1.0 0.125 2.0 -4.5))
    (double-float . #(1.5d0 -0.0d0 0.0d0 3.0d0 -0.5d0 7.75d0 -1.0d0 0.125d0
                       2.0d0 -4.5d0))
    ((unsigned-byte 32) . #(0 1 2 300000 4294967295 3000000000 7 65536 9 15))
    ((unsigned-byte 16) . #(0 1 2 300 65535 40000 128 7 9 15))
    ((unsigned-byte 8) . #(0 1 2 7 255 200 128 3 9 15))))

;;; A value the loop cannot be expected to leave behind, for the eight spare
;;; elements past N that a packed store running past the bound would hit.
(defparameter *guards*
  `((single-float . -12345.25f0) (double-float . -12345.25d0)
    ((unsigned-byte 32) . 7777777) ((unsigned-byte 16) . 7777)
    ((unsigned-byte 8) . 77)))

;;; The loop-invariant scalar each "load and a scalar" case reads.  The floats
;;; are not representable in binary, so a path that rounds differently from the
;;; scalar loop cannot cancel out (compare tests/auto-vectorize.pure.lisp:66-74).
(defparameter *scalars*
  `((single-float . 1.1f0) (double-float . 1.1d0)
    ((unsigned-byte 32) . 3) ((unsigned-byte 16) . 3) ((unsigned-byte 8) . 5)))

(defun lookup (table et)
  (cdr (assoc et table :test #'equal)))

(defun float-type-p (et)
  (member et *float-types* :test #'equal))

;;; COUNT elements of ET drawn from the value pattern, starting at OFFSET.
(defun element-values (et count &optional (offset 0))
  (let ((pattern (lookup *values* et)))
    (loop for i from 0 below count
          collect (aref pattern (mod (+ i offset) (length pattern))))))

(defun make-test-array (et count &optional (offset 0))
  (make-array count :element-type et
              :initial-contents (element-values et count offset)))

;;; ---- compiling -------------------------------------------------------

(defvar *avx2-flag* (find-symbol "*AVX2-AVAILABLE*" "SB-VM")
  "The special the emitted dispatch reads, found rather than named so that a
core without it still loads this harness.")

(defun compile-quietly (form)
  "Compile FORM, returning (values FUNCTION WARNINGS).  Compiler output is
captured so the table stays readable; warnings other than style warnings are
returned so that a broken declaration is visible instead of silent."
  (let ((warnings '())
        (text (make-string-output-stream)))
    (handler-bind ((warning (lambda (c)
                              (unless (typep c 'style-warning)
                                (push (princ-to-string c) warnings)))))
      (let ((*error-output* text)
            (sb-c::*auto-vectorize-notes* nil))
        (values (compile nil form) (nreverse warnings))))))

(defun disassembly-text (function)
  (string-downcase (with-output-to-string (s) (disassemble function :stream s))))

(defun mnemonics-in (text)
  (remove-if-not (lambda (m) (search m text)) *packed-mnemonics*))

(defparameter *shape-two-loads*
  (list :name :two-loads
        :label "c[i] := a[i] OP b[i]"
        :decl :simple :args '(a b c n) :arrays '(a b c) :outputs '(c)
        :body (lambda (et op)
                (declare (ignore et))
                `(dotimes (i n)
                   (setf (aref c i) (,op (aref a i) (aref b i)))))))

(defparameter *shape-load-scalar*
  (list :name :load-scalar
        :label "c[i] := a[i] OP k"
        :decl :simple :args '(a b c k n) :arrays '(a b c) :scalars '(k)
        :outputs '(c)
        :body (lambda (et op)
                (declare (ignore et))
                `(dotimes (i n)
                   (setf (aref c i) (,op (aref a i) k))))))

;;; (coerce i 'single-float) yields a SINGLE-FLOAT, so this shape is legal only
;;; where the destination element type is SINGLE-FLOAT; storing it into a
;;; DOUBLE-FLOAT array is a type error the compiler reports.
(defparameter *shape-index-value*
  (list :name :index-value
        :label "c[i] := (coerce i 'single-float)"
        :decl :simple :args '(a b c n) :arrays '(a b c) :outputs '(c)
        :ops nil :types '(single-float) :op-column "n/a"
        :body (lambda (et op)
                (declare (ignore et op))
                `(dotimes (i n)
                   (setf (aref c i) (coerce i 'single-float))))))

(defparameter *shape-negate*
  (list :name :negate
        :label "c[i] := -a[i]"
        :decl :simple :args '(a b c n) :arrays '(a b c) :outputs '(c)
        :ops nil :op-column "- (unary)"
        :body (lambda (et op)
                (declare (ignore et op))
                `(dotimes (i n)
                   (setf (aref c i) (- (aref a i)))))))

(defparameter *shape-two-stores*
  (list :name :two-stores
        :label "c[i] := a[i] OP b[i]; d[i] := -a[i]"
        :decl :simple :args '(a b c d n) :arrays '(a b c d) :outputs '(c d)
        :body (lambda (et op)
                (declare (ignore et))
                `(dotimes (i n)
                   (setf (aref c i) (,op (aref a i) (aref b i)))
                   (setf (aref d i) (- (aref a i)))))))

;;; Non-simple arrays: adjustable with a fill pointer, declared (VECTOR ...).
(defparameter *shape-adjustable*
  (list :name :adjustable
        :label "(vector single-float) adjustable: c[i] := a[i] OP b[i]"
        :decl :vector :args '(a b c n) :arrays '(a b c) :outputs '(c)
        :types '(single-float)
        :body (lambda (et op)
                (declare (ignore et))
                `(dotimes (i n)
                   (setf (aref c i) (,op (aref a i) (aref b i)))))))

(defparameter *shapes*
  (list *shape-two-loads* *shape-load-scalar* *shape-index-value*
        *shape-negate* *shape-two-stores* *shape-adjustable*))

;;; ---- policies --------------------------------------------------------

;;; Where the vectorizer runs: the pass is gated on (>= speed 2)
;;; (src/compiler/vectorize.lisp:1610-1615, :1618-1622).  Integer lanes wrap, so
;;; they also need (safety 0), which is where the pass says integer lanes are
;;; equivalent to the scalar loop (src/compiler/vectorize.lisp:56).
(defun fast-policy (et)
  (if (float-type-p et)
      '((speed 3))
      '((speed 3) (safety 0))))

;;; Where it cannot.  Floats are compared against (speed 0) (safety 3); integer
;;; lanes against (speed 0) (safety 1) as asked -- but a safety-1 store checks
;;; the element type, so an unsigned loop that wraps signals TYPE-ERROR there.
;;; Measured on this core: "sub (UNSIGNED-BYTE 16) speed0 safety1 => SIGNALS
;;; TYPE-ERROR: The value -4 is not of type (UNSIGNED-BYTE 16)", while the same
;;; loop at safety 0 gives (65532 65338 193 200).  When the safety-1 reference
;;; cannot run, the case is compared against (speed 0) (safety 0) instead --
;;; still unvectorized, since 0 < 2 -- and the REF column marks it with a star.
(defun reference-policies (et)
  (if (float-type-p et)
      '(("safety3" . ((speed 0) (safety 3))))
      '(("safety1" . ((speed 0) (safety 1)))
        ("safety0*" . ((speed 0) (safety 0))))))

;;; ---- inputs ----------------------------------------------------------

(defun buffer-value (et name i)
  "The initial value of element I of buffer NAME: a and b are inputs read at
different phases of the pattern, c and d start as the guard value."
  (cond ((eq name 'a)
         (let ((pattern (lookup *values* et)))
           (aref pattern (mod i (length pattern)))))
        ((eq name 'b)
         (let ((pattern (lookup *values* et)))
           (aref pattern (mod (+ i 3) (length pattern)))))
        (t (lookup *guards* et))))

(defun make-buffer (shape et name n)
  "A buffer of N + 8 elements, so that a packed store running past N is visible."
  (let* ((count (+ n 8))
         (contents (loop for i from 0 below count collect (buffer-value et name i))))
    (if (eq (getf shape :decl) :vector)
        (make-array count :element-type et :adjustable t :fill-pointer count
                    :initial-contents contents)
        (make-array count :element-type et :initial-contents contents))))

(defun make-inputs (case n)
  (let* ((shape (getf case :shape))
         (et (getf case :et))
         (buffers (mapcar (lambda (name) (cons name (make-buffer shape et name n)))
                          (getf shape :arrays)))
         (scalar (lookup *scalars* et))
         (args (loop for arg in (getf shape :args)
                     collect (cond ((eql arg 'n) n)
                                   ((eql arg 'k) scalar)
                                   (t (cdr (assoc arg buffers)))))))
    (values args buffers)))

;;; ---- one case -------------------------------------------------------

(define-condition harness-failure (error)
  ((message :initarg :message :reader failure-message))
  (:report failure-message))

(defun fail-case (control &rest args)
  (error 'harness-failure :message (apply #'format nil control args)))

(defun op-name (shape op)
  (or (and op (string-downcase (string op)))
      (getf shape :op-column "n/a")))

(defun array-declaration (shape et)
  (if (eq (getf shape :decl) :vector)
      `(vector ,et)
      `(simple-array ,et (*))))

(defun case-form (case policy)
  (let* ((shape (getf case :shape))
         (et (getf case :et))
         (op (getf case :op)))
    `(lambda ,(getf shape :args)
       (declare ,@(mapcar (lambda (v) `(type ,(array-declaration shape et) ,v))
                          (getf shape :arrays))
                ,@(when (getf shape :scalars)
                    `((type ,et ,@(getf shape :scalars))))
                (type (and fixnum (integer 0 #.array-dimension-limit)) n)
                (optimize ,@policy))
       ,(funcall (getf shape :body) et op))))

(defun reference-contents (case function)
  "Runs FUNCTION, the scalar reference, over every length, and returns one alist
per length: output array name to the full contents of that array."
  (let ((shape (getf case :shape)))
    (loop for n in *lengths*
          collect
          (multiple-value-bind (args buffers) (make-inputs case n)
            (apply function args)
            (loop for name in (getf shape :outputs)
                  collect (cons name
                                (loop for i from 0 below (+ n 8)
                                      collect (aref (cdr (assoc name buffers)) i))))))))

(defun compile-reference (case)
  "Returns (values FUNCTION NOTE CONTENTS): the scalar reference compiled at the
first candidate policy that runs it over every length without signalling."
  (let* ((last-error nil)
         (found
           (loop for (note . policy) in (reference-policies (getf case :et))
                 do (let ((function (compile-quietly (case-form case policy))))
                      (handler-case
                          (return (list function note
                                        (reference-contents case function)))
                        (error (e) (setf last-error e)))))))
    (if found
        (values (first found) (second found) (third found))
        (fail-case "no scalar reference ran for ~a, ~a, op ~a: ~a"
                   (getf (getf case :shape) :label) (getf case :et)
                   (getf case :op) last-error))))

;;; ---- the run ---------------------------------------------------------

(defvar *results* '())
(defvar *current-row* nil)
(defvar *failed* nil)

(defun all-cases ()
  "One case per shape, element type and operator the shape is legal for."
  (loop for shape in *shapes*
        append (loop for et in (or (getf shape :types) *element-types*)
                     append (let ((ops (getf shape :ops *binary-ops*)))
                              (if ops
                                  (loop for op in ops
                                        collect (list :shape shape :et et :op op))
                                  (list (list :shape shape :et et :op nil)))))))

(defun run-on-path (function wide args)
  "Calls FUNCTION with SB-VM::*AVX2-AVAILABLE* bound to WIDE, the flag the
emitted code tests before choosing its 256-bit or its 128-bit loop."
  (if *avx2-flag*
      (progv (list *avx2-flag*) (list (and wide t)) (apply function args))
      (apply function args)))

(defun check-path (case function path expected)
  (let* ((shape (getf case :shape))
         (et (getf case :et))
         (guard (lookup *guards* et)))
    (loop for n in *lengths*
          for expect in expected
          do (multiple-value-bind (args buffers) (make-inputs case n)
               (run-on-path function (eq path :wide) args)
               (dolist (name (getf shape :outputs))
                 (let ((array (cdr (assoc name buffers)))
                       (wanted (cdr (assoc name expect))))
                   (loop for i from 0 below n
                         unless (eql (aref array i) (nth i wanted))
                           do (fail-case "~a | ~a | op ~a | length ~d | ~a path: element ~d is ~s, expected ~s"
                                         (getf shape :label) et
                                         (op-name shape (getf case :op)) n path
                                         i (aref array i) (nth i wanted)))
                   (loop for i from n below (+ n 8)
                         unless (eql (aref array i) guard)
                           do (fail-case "~a | ~a | op ~a | length ~d | ~a path: element ~d is past the bound ~d but was written: ~s"
                                         (getf shape :label) et
                                         (op-name shape (getf case :op)) n path
                                         i n (aref array i)))))))))

(defun run-case (case)
  (let* ((shape (getf case :shape))
         (et (getf case :et))
         (op (getf case :op))
         (row (list :label (getf shape :label) :et et
                    :op (op-name shape op) :lengths (length *lengths*)
                    :ref "?" :packed nil :wide "?" :narrow "?")))
    (setf *current-row* row)
    (multiple-value-bind (fast warnings)
        (compile-quietly (case-form case (fast-policy et)))
      (dolist (warning warnings)
        (format t "~&; compile warning for ~a, ~a: ~a~%"
                (getf shape :label) et
                (subseq warning 0 (min 100 (length warning)))))
      (let* ((text (disassembly-text fast))
             (packed (mnemonics-in text)))
        (setf (getf row :packed) packed
              ;; the emitted dispatch reads the flag this harness binds; a case
              ;; whose code does not mention it cannot be steered down a path
              (getf row :dispatch) (and (search "avx2-available" text) t))
        (multiple-value-bind (reference note expected) (compile-reference case)
          (let ((reference-packed (mnemonics-in (disassembly-text reference))))
            (when reference-packed
              (fail-case "the (speed 0) reference for ~a, ~a, op ~a is itself packed (~{~a~^ ~})"
                         (getf shape :label) et (op-name shape op) reference-packed)))
          (setf (getf row :ref) note)
          ;; both bindings of the dispatch flag are exercised whatever the image
          ;; emitted; the WIDE column says n/a when there is no 256-bit loop to
          ;; choose, since then the flag picks the same code either way
          (check-path case fast :wide expected)
          (check-path case fast :narrow expected)
          ;; both bindings are always exercised.  WIDE says which loop the flag
          ;; can actually reach: "ok" when the code tests the flag and has a
          ;; 256-bit loop, "n/a" when there is no 256-bit loop at all, and
          ;; "fixed" when 256-bit code is there with no test of the flag, so
          ;; binding it t or nil runs the same instructions.
          (setf (getf row :wide)
                (cond ((not (search "ymm" text)) "n/a")
                      ((getf row :dispatch) "ok")
                      (t "fixed"))
                (getf row :narrow) "ok"))))
    (push row *results*)))

;;; ---- the table -------------------------------------------------------

;;; ---- the test ------------------------------------------------------------

(with-test (:name :vectorizer-differential)
  (let ((cases (all-cases)))
    (assert (>= (length cases) 50) ()
            "only ~d cases were generated; the shape list is broken" (length cases))
    (dolist (case cases)
      (run-case case))))
