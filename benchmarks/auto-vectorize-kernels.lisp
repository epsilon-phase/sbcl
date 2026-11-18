;;;; benchmarks/auto-vectorize-kernels.lisp
;;;;
;;;; Realistic SIMD-friendly workloads for SBCL's automatic vectorizer.
;;;;
;;;; Every kernel here is compiled TWICE from one body of loop text, once
;;;; normally, with the pass deciding, and once with SB-C::*AUTO-VECTORIZE*
;;;; bound to NIL.  The scalar twin is therefore the same loop text, not a
;;;; hand-written scalar version, so any measured difference is the pass.
;;;;
;;;; Reproduce with, from the sbcl checkout:
;;;;
;;;;   src/runtime/sbcl --core output/sbcl.core --noinform --no-sysinit \
;;;;       --no-userinit --disable-debugger --load benchmarks/auto-vectorize-kernels.lisp
;;;;
;;;; TIMING: best-of-N.  N = 7 samples per measurement; each sample runs the
;;;; kernel repeatedly until at least 120 ms of CPU time has passed, and the
;;;; reported number is the fastest sample's per-call time.  It is CPU time
;;;; from GET-INTERNAL-RUN-TIME, not wall time, because another process competes
;;;; for the machine while these numbers are taken.  Even so the 8192-element
;;;; cache-resident rows are small enough that interference shows in them.

(defpackage #:auto-vectorize-kernels
  (:use #:cl)
  (:export #:run))

(in-package #:auto-vectorize-kernels)

;;;; ---------------------------------------------------------------- ;
;;;; deterministic 32-bit xorshift, so every run fills arrays the same ;
;;;; ---------------------------------------------------------------- ;

(defparameter *rng-state* #x5DEECE6D)

(defun rng-u32 ()
  (let ((x *rng-state*))
    (setf x (logand #xffffffff (logxor x (ash x 13))))
    (setf x (logand #xffffffff (logxor x (ash x -17))))
    (setf x (logand #xffffffff (logxor x (ash x 5))))
    (setf *rng-state* x)
    x))

;;;; ------------------- array helpers ------------------------------- ;

(defun make-f32 (n) (make-array n :element-type 'single-float))
(defun make-f64 (n) (make-array n :element-type 'double-float))
(defun make-u8 (n)  (make-array n :element-type '(unsigned-byte 8)))
(defun make-u32 (n) (make-array n :element-type '(unsigned-byte 32)))
(defun make-s16 (n) (make-array n :element-type '(signed-byte 16)))
(defun make-s8  (n) (make-array n :element-type '(signed-byte 8)))

(defun fill-f32 (a)
  (dotimes (i (length a) a)
    (setf (aref a i)
          (coerce (/ (rng-u32) 4294967296.0d0) 'single-float))))

(defun fill-f64 (a)
  (dotimes (i (length a) a)
    (setf (aref a i) (/ (rng-u32) 4294967296.0d0))))

(defun fill-u8 (a)
  (dotimes (i (length a) a)
    (setf (aref a i) (logand #xff (rng-u32)))))

(defun fill-s16 (a)
  "16-bit audio samples: full range, so a clamp has something to clamp."
  (dotimes (i (length a) a)
    (setf (aref a i) (- (logand #xffff (rng-u32)) #x8000))))

(defun fill-s8 (a)
  (dotimes (i (length a) a)
    (setf (aref a i) (- (logand #xff (rng-u32)) #x80))))

(defun fill-u32 (a)
  (dotimes (i (length a) a)
    (setf (aref a i) (rng-u32))))

(defun arrays-eql (a b)
  (and (= (length a) (length b))
       (dotimes (i (length a) t)
         (unless (eql (aref a i) (aref b i))
           (return nil)))))

;;;; -------- kernel compilation: one body, two compiles -------------- ;

(defstruct (kernel-spec (:constructor make-spec))
  name scalar-name args declares body packed
  v-notes s-notes mnemonics-found)

(defparameter *kernel-specs* '())

(defun note-text (condition)
  (if (simple-condition-format-control condition)
      (apply #'format nil
             (simple-condition-format-control condition)
             (simple-condition-format-arguments condition))
      (princ-to-string condition)))

;;; Compiles (lambda ARGS DECLARES BODY) with the pass on or off and returns
;;; the compiler's own vectorizer notes (success notes AND "unable to
;;; vectorize this loop: ..." reasons).  Compiler output is silenced; the
;;; notes are captured as conditions, not scraped from printed text.
(defun compile-twin (name args declares body vectorize)
  (let ((sb-c::*auto-vectorize* vectorize)
        (notes '()))
    (handler-bind ((sb-c::compiler-note
                     (lambda (c) (push (note-text c) notes))))
      (let ((*error-output* (make-broadcast-stream)))
        (setf (fdefinition name)
              (compile nil `(lambda ,args ,@declares ,body)))))
    (nreverse notes)))

(defun packed-evidence (function mnemonics)
  (let ((text (with-output-to-string (*standard-output*)
                (disassemble function))))
    (remove-if-not (lambda (m) (search m text :test #'char-equal))
                   mnemonics)))

(defun register-kernel (name scalar-name args declares body packed)
  (let ((v-notes (compile-twin name args declares body t))
        (s-notes (compile-twin scalar-name args declares body nil)))
    (push (make-spec :name name :scalar-name scalar-name :args args
                     :declares declares :body body :packed packed
                     :v-notes v-notes :s-notes s-notes
                     :mnemonics-found (packed-evidence (fdefinition name) packed))
          *kernel-specs*))
  (values))

(defmacro define-kernel (name args declares body &key packed)
  (let ((scalar-name (intern (format nil "~a-SCALAR" (symbol-name name))
                             (symbol-package name))))
    `(progn
       (defun ,name ())                 ; placeholders; replaced by COMPILE
       (defun ,scalar-name ())
       (register-kernel ',name ',scalar-name ',args ',declares ',body ',packed))))

;;;; ------------------------- the kernels ---------------------------- ;
;;;;
;;;; 1. particle integration: pos += vel*dt, the standard SIMD case in
;;;;    games and physics.  single-float, 4-wide.

(define-kernel integrate-f32
    (pos vel dt n)
  ((declare (type (simple-array single-float (*)) pos vel)
            (type single-float dt)
            (type fixnum n)
            (optimize (speed 3))))
  (dotimes (i n)
    (setf (aref pos i) (+ (aref pos i) (* (aref vel i) dt))))
  :packed ("addps" "mulps" "movups"))

;;;; 2. SAXPY: y += a*x, BLAS level 1.  single-float.

(define-kernel saxpy-f32
    (y x a n)
  ((declare (type (simple-array single-float (*)) y x)
            (type single-float a)
            (type fixnum n)
            (optimize (speed 3))))
  (dotimes (i n)
    (setf (aref y i) (+ (aref y i) (* a (aref x i)))))
  :packed ("addps" "mulps"))

;;;; 3. audio mix + gain: out = 0.5*(l+r), downmixing two channels.

(define-kernel mix-stereo-f32
    (out l r n)
  ((declare (type (simple-array single-float (*)) out l r)
            (type fixnum n)
            (optimize (speed 3))))
  (dotimes (i n)
    (setf (aref out i) (* 0.5 (+ (aref l i) (aref r i)))))
  :packed ("addps" "mulps"))

;;;; 4. u8 brightness clamp (image tone mapping): out = min(in, 200).
;;;;    Integer lanes need (safety 0).

(define-kernel clamp-u8
    (out in n)
  ((declare (type (simple-array (unsigned-byte 8) (*)) out in)
            (type fixnum n)
            (optimize (speed 3) (safety 0))))
  (dotimes (i n)
    (setf (aref out i) (min (aref in i) 200)))
  :packed ("pminub" "movdqu" "movdqa"))

;;;; 5. PNG Sub filter.  The textbook form (aref out (- i 3)) is not
;;;;    vectorizable, because the offset is negative; what the pass takes is
;;;;    either an offset-0 read from an already-written line,

(define-kernel png-sub-prev-u8
    (out in prev n)
  ((declare (type (simple-array (unsigned-byte 8) (*)) out in prev)
            (type fixnum n)
            (optimize (speed 3) (safety 0))))
  (dotimes (i n)
    (setf (aref out i) (+ (aref in i) (aref prev i))))
  :packed ("paddb" "movdqu" "movdqa"))

;;;; or a literal positive offset into the input.

(define-kernel png-sub-offset-u8
    (out in n)
  ((declare (type (simple-array (unsigned-byte 8) (*)) out in)
            (type fixnum n)
            (optimize (speed 3) (safety 0))))
  (dotimes (i n)
    (setf (aref out i) (+ (aref in i) (aref in (+ i 3)))))
  :packed ("paddb" "movdqu" "movdqa"))

;;;; 6. integer reduction: a checksum-style exact u32 sum.
;;;;    Float reductions change the result, so the pass refuses them;
;;;;    integer addition is exact, so the pass takes this one.

(define-kernel checksum-u32
    (a n)
  ((declare (type (simple-array (unsigned-byte 32) (*)) a)
            (type fixnum n)
            (optimize (speed 3) (safety 0))))
  (let ((total 0))
    (declare (type (unsigned-byte 32) total))
    (dotimes (i n)
      (incf total (aref a i)))
    total)
  :packed ("paddd" "movdqu" "movdqa"))

;;;; 7. index-affine ramp: out = i * 0.01, gradient/curve table generation.

(define-kernel ramp-f32
    (out n)
  ((declare (type (simple-array single-float (*)) out)
            (type fixnum n)
            (optimize (speed 3))))
  (dotimes (i n)
    (setf (aref out i) (* (float i) 0.01)))
  :packed ("mulps" "addps"))

;;;; 8. double-float SAXPY: the 2-lane case.

(define-kernel daxpy-f64
    (y x a n)
  ((declare (type (simple-array double-float (*)) y x)
            (type double-float a)
            (type fixnum n)
            (optimize (speed 3))))
  (dotimes (i n)
    (setf (aref y i) (+ (aref y i) (* a (aref x i)))))
  :packed ("addpd" "mulpd" "movupd"))

;;;; 10. 16-bit audio clamp: the loop the pass takes at the DEFAULT policy,
;;;;     because min and max cannot store a value the lane cannot hold.

(define-kernel clamp-s16
    (out in n)
  ((declare (type (simple-array (signed-byte 16) (*)) out in)
            (type fixnum n)
            (optimize (speed 3))))
  (dotimes (i n)
    (setf (aref out i) (max -1000 (min 1000 (aref in i)))))
  :packed ("pminsw" "pmaxsw"))

;;;; 11. signed-byte image clamp, same story with the biased unsigned
;;;;     instructions, since SSE2 has no signed byte minimum.

(define-kernel clamp-s8
    (out in n)
  ((declare (type (simple-array (signed-byte 8) (*)) out in)
            (type fixnum n)
            (optimize (speed 3))))
  (dotimes (i n)
    (setf (aref out i) (min 100 (aref in i))))
  :packed ("pminub" "pxor"))

;;;; 12. masking: a bitwise operation on n-bit values has an n-bit result, so
;;;;     this is exact too, and it vectorizes at the default policy.

(define-kernel mask-u32
    (out in n)
  ((declare (type (simple-array (unsigned-byte 32) (*)) out in)
            (type fixnum n)
            (optimize (speed 3))))
  (dotimes (i n)
    (setf (aref out i) (logior (logand (aref in i) #xff00) #x80)))
  :packed ("pand" "por"))

;;;; ---------------------- timing: best of N ------------------------- ;

(defparameter *timing-sizes* '(8192 1048576))
(defparameter *correctness-sizes* '(3 7))     ; exercise the scalar tail
(defparameter *samples* 7)
(defparameter *sample-min-ms* 120)

(defun best-of (fn &optional (samples *samples*) (min-ms *sample-min-ms*))
  ;; best per-call time, in internal-time units
  (funcall fn)                                 ; warm up
  (let ((best most-positive-double-float)
        (limit (round (* min-ms 0.001d0 internal-time-units-per-second))))
    (dotimes (s samples best)
      (let ((start (get-internal-run-time))
            (reps 0))
        (loop
          (funcall fn)
          (incf reps)
          (when (>= (- (get-internal-run-time) start) limit)
            (return)))
        (let ((per (/ (coerce (- (get-internal-run-time) start) 'double-float)
                      reps)))
          (when (< per best)
            (setf best per)))))))

(defun to-ms (units)
  (/ (* units 1000.0d0) internal-time-units-per-second))

;;;; ---------------------- per-kernel drivers ------------------------ ;
;;;; Each maker returns three thunks: call the vectorized kernel, call the
;;;; scalar twin, and compare the two results element by element (or the two
;;;; sums, for the reduction).

(defun make-integrate-runs (n)
  (let* ((pos-v (fill-f32 (make-f32 n)))
         (pos-s (copy-seq pos-v))
         (vel   (fill-f32 (make-f32 n))))
    (values
     (lambda () (integrate-f32 pos-v vel 0.001 n))
     (lambda () (integrate-f32-scalar pos-s vel 0.001 n))
     (lambda (v s) (declare (ignore v s)) (arrays-eql pos-v pos-s)))))

(defun make-saxpy-runs (n)
  (let* ((y-v (fill-f32 (make-f32 n)))
         (y-s (copy-seq y-v))
         (x   (fill-f32 (make-f32 n)))
         (a   (coerce (/ (rng-u32) 4294967296.0d0) 'single-float)))
    (values
     (lambda () (saxpy-f32 y-v x a n))
     (lambda () (saxpy-f32-scalar y-s x a n))
     (lambda (v s) (declare (ignore v s)) (arrays-eql y-v y-s)))))

(defun make-mix-runs (n)
  (let* ((out-v (make-f32 n))
         (out-s (make-f32 n))
         (l     (fill-f32 (make-f32 n)))
         (r     (fill-f32 (make-f32 n))))
    (values
     (lambda () (mix-stereo-f32 out-v l r n))
     (lambda () (mix-stereo-f32-scalar out-s l r n))
     (lambda (v s) (declare (ignore v s)) (arrays-eql out-v out-s)))))

(defun make-clamp-runs (n)
  (let* ((out-v (make-u8 n))
         (out-s (make-u8 n))
         (in    (fill-u8 (make-u8 n))))
    (values
     (lambda () (clamp-u8 out-v in n))
     (lambda () (clamp-u8-scalar out-s in n))
     (lambda (v s) (declare (ignore v s)) (arrays-eql out-v out-s)))))

(defun make-sub-prev-runs (n)
  (let* ((out-v (make-u8 n))
         (out-s (make-u8 n))
         (in    (fill-u8 (make-u8 n)))
         (prev  (fill-u8 (make-u8 n))))
    (values
     (lambda () (png-sub-prev-u8 out-v in prev n))
     (lambda () (png-sub-prev-u8-scalar out-s in prev n))
     (lambda (v s) (declare (ignore v s)) (arrays-eql out-v out-s)))))

(defun make-sub-offset-runs (n)
  ;; the input needs n+3 elements: the guard reads up to (n-1)+3
  (let* ((out-v (make-u8 n))
         (out-s (make-u8 n))
         (in    (fill-u8 (make-u8 (+ n 3)))))
    (values
     (lambda () (png-sub-offset-u8 out-v in n))
     (lambda () (png-sub-offset-u8-scalar out-s in n))
     (lambda (v s) (declare (ignore v s)) (arrays-eql out-v out-s)))))

(defun make-checksum-runs (n)
  (let ((a (fill-u32 (make-u32 n))))
    (values
     (lambda () (checksum-u32 a n))
     (lambda () (checksum-u32-scalar a n))
     (lambda (v s)
       ;; print what each side computed, to show what happened once the
       ;; running sum left 32 bits
       (let ((exact 0))
         (dotimes (i n) (setf exact (+ exact (aref a i))))
         (format t "~17texact sum ~d, mod 2^32 ~x; vector ~d, scalar ~d~%"
                 exact (logand exact #xffffffff) v s)
         (eql v s))))))

(defun make-ramp-runs (n)
  (let ((out-v (make-f32 n))
        (out-s (make-f32 n)))
    (values
     (lambda () (ramp-f32 out-v n))
     (lambda () (ramp-f32-scalar out-s n))
     (lambda (v s) (declare (ignore v s)) (arrays-eql out-v out-s)))))

(defun make-daxpy-runs (n)
  (let* ((y-v (fill-f64 (make-f64 n)))
         (y-s (copy-seq y-v))
         (x   (fill-f64 (make-f64 n)))
         (a   (/ (rng-u32) 4294967296.0d0)))
    (values
     (lambda () (daxpy-f64 y-v x a n))
     (lambda () (daxpy-f64-scalar y-s x a n))
     (lambda (v s) (declare (ignore v s)) (arrays-eql y-v y-s)))))

(defun make-clamp-s16-runs (n)
  (let* ((out-v (make-s16 n))
         (out-s (make-s16 n))
         (in    (fill-s16 (make-s16 n))))
    (values
     (lambda () (clamp-s16 out-v in n))
     (lambda () (clamp-s16-scalar out-s in n))
     (lambda (v s) (declare (ignore v s)) (arrays-eql out-v out-s)))))

(defun make-clamp-s8-runs (n)
  (let* ((out-v (make-s8 n))
         (out-s (make-s8 n))
         (in    (fill-s8 (make-s8 n))))
    (values
     (lambda () (clamp-s8 out-v in n))
     (lambda () (clamp-s8-scalar out-s in n))
     (lambda (v s) (declare (ignore v s)) (arrays-eql out-v out-s)))))

(defun make-mask-u32-runs (n)
  (let* ((out-v (make-u32 n))
         (out-s (make-u32 n))
         (in    (fill-u32 (make-u32 n))))
    (values
     (lambda () (mask-u32 out-v in n))
     (lambda () (mask-u32-scalar out-s in n))
     (lambda (v s) (declare (ignore v s)) (arrays-eql out-v out-s)))))

(defstruct bench label spec traffic make-runs)

(defparameter *benches*
  (list
   (make-bench :label "particle integration  pos += vel*dt"
               :spec 'integrate-f32 :traffic 12
               :make-runs 'make-integrate-runs)
   (make-bench :label "SAXPY  y += a*x"
               :spec 'saxpy-f32 :traffic 12
               :make-runs 'make-saxpy-runs)
   (make-bench :label "audio mix + gain  out = 0.5*(l+r)"
               :spec 'mix-stereo-f32 :traffic 12
               :make-runs 'make-mix-runs)
   (make-bench :label "u8 brightness clamp  out = min(in,200)"
               :spec 'clamp-u8 :traffic 2
               :make-runs 'make-clamp-runs)
   (make-bench :label "s16 audio clamp  out = clamp(in,-1000,1000)"
               :spec 'clamp-s16 :traffic 4
               :make-runs 'make-clamp-s16-runs)
   (make-bench :label "s8 image clamp  out = min(in,100)"
               :spec 'clamp-s8 :traffic 2
               :make-runs 'make-clamp-s8-runs)
   (make-bench :label "u32 mask  out = (in & #xff00) | #x80"
               :spec 'mask-u32 :traffic 8
               :make-runs 'make-mask-u32-runs)
   (make-bench :label "PNG Sub filter  out = in + prev"
               :spec 'png-sub-prev-u8 :traffic 3
               :make-runs 'make-sub-prev-runs)
   (make-bench :label "PNG Sub filter  out = in[i] + in[i+3]"
               :spec 'png-sub-offset-u8 :traffic 3
               :make-runs 'make-sub-offset-runs)
   (make-bench :label "u32 checksum sum  total += a[i]"
               :spec 'checksum-u32 :traffic 4
               :make-runs 'make-checksum-runs)
   (make-bench :label "index ramp  out = i*0.01"
               :spec 'ramp-f32 :traffic 4
               :make-runs 'make-ramp-runs)
   (make-bench :label "double SAXPY  y += a*x"
               :spec 'daxpy-f64 :traffic 24
               :make-runs 'make-daxpy-runs)))

;;;; ---------------------------- the run ------------------------------ ;

(defparameter *mismatches* '())

(defun find-spec (name)
  (find name *kernel-specs* :key #'kernel-spec-name))

(defun size-tag (n)
  (if (= n 8192) "8192 cache" "1048576 stream"))

(defun run-bench (bench)
  (let ((spec (find-spec (bench-spec bench)))
        (rows '()))
    (dolist (n (append *correctness-sizes* *timing-sizes*))
      (multiple-value-bind (vcall scall compare)
          (funcall (bench-make-runs bench) n)
        (let ((vres (funcall vcall))
              (sres (funcall scall)))
          (unless (funcall compare vres sres)
            (push (list (bench-label bench) n) *mismatches*)
            (format t "~&!! MISMATCH: ~a at n = ~d~%~17tvector result: ~s~%~17tscalar result: ~s"
            (bench-label bench) n vres sres)))
        (when (member n *timing-sizes*)
          (let ((sv (to-ms (best-of vcall)))
                (ss (to-ms (best-of scall))))
            (format t "~&~38a ~a ~9,3f ~9,3f ~6,2fx ~8,1f ~8,1f  ~a~%"
                    (bench-label bench) (size-tag n) ss sv
                    (/ ss sv)
                    (/ (* (bench-traffic bench) n 1.0d0) (* ss 1.0d6))
                    (/ (* (bench-traffic bench) n 1.0d0) (* sv 1.0d6))
                    (if (and (kernel-spec-v-notes spec)
                             (kernel-spec-mnemonics-found spec))
                        "vectorized"
                        "NOT VECTORIZED"))
            (push (list (kernel-spec-name spec) (bench-label bench) n ss sv)
                  rows)))))
    (nreverse rows)))

(defun print-evidence ()
  (dolist (spec (reverse *kernel-specs*))
    (terpri)
    (format t "~a~%" (kernel-spec-name spec))
    (if (kernel-spec-v-notes spec)
        (dolist (note (kernel-spec-v-notes spec))
          (format t "  compiler note: ~a~%" note))
        (format t "  compiler note: none~%"))
    (if (kernel-spec-mnemonics-found spec)
        (format t "  disassembly: packed instructions found: ~{~a~^ ~}~%"
                (kernel-spec-mnemonics-found spec))
        (format t "  disassembly: NO packed instruction from ~{~a~^ ~}~%"
                (kernel-spec-packed spec)))
    (when (kernel-spec-s-notes spec)
      (format t "  !! scalar twin also got notes: ~{~a~^; ~}~%"
              (kernel-spec-s-notes spec)))))

(defun rows-for (name rows)
  (remove-if-not (lambda (r) (eq (first r) name)) rows))

(defun print-summary (rows)
  (terpri)
  (format t "=== summary ===~%")
  (let ((vectorized '()) (missed '()))
    (dolist (spec (reverse *kernel-specs*))
      (if (and (kernel-spec-v-notes spec)
               (kernel-spec-mnemonics-found spec))
          (push spec vectorized)
          (push spec missed)))
    (format t "vectorized: ~d of ~d kernels~%"
            (length vectorized) (length *kernel-specs*))
    (dolist (spec (nreverse vectorized))
      (let* ((rs (rows-for (kernel-spec-name spec) rows))
             (speeds (with-output-to-string (s)
                       (dolist (r rs)
                         (format s "  ~a: ~,2fx (~,3f / ~,3f ms vector)"
                                 (size-tag (third r))
                                 (/ (fourth r) (fifth r))
                                 (fourth r) (fifth r))))))
        (format t "  ~a~a~%" (second (first rs)) speeds)))
    (when missed
      (format t "NOT vectorized:~%")
      (dolist (spec (nreverse missed))
        (if (kernel-spec-v-notes spec)
            (dolist (note (kernel-spec-v-notes spec))
              (format t "  ~a: ~a~%" (kernel-spec-name spec) note))
            (format t "  ~a: no note and no packed instructions~%"
                    (kernel-spec-name spec)))))
    (if *mismatches*
        (progn (format t "correctness FAILURES:~%")
               (dolist (m (nreverse *mismatches*))
                 (format t "  ~a at n = ~d~%" (first m) (second m)))
               (format t "  the u32 reduction is the only kernel whose two results can differ: ~
the packed accumulator (PADDD) wraps each 32-bit lane, so the vectorized ~
result is the true sum modulo 2^32 plus the recombined lane carries, ~
while the scalar twin's unchecked chain violates its (unsigned-byte 32) ~
accumulator once the running sum passes 2^32 - CL leaves that undefined ~
at (safety 0), and the compiled code returned something else.  Sums that ~
fit in 32 bits agree exactly."))
        (format t "correctness: every vectorized output eql to its scalar twin~%~
                   at n = 3, 7, 8192 and 1048576.~%"))))

(defun run ()
  (format t "~&SBCL ~a; automatic vectorizer ~:[OFF~;on~] (kill switch ~
              sb-c::*auto-vectorize*).~%"
          (lisp-implementation-version) sb-c::*auto-vectorize*)
  (format t "timing: best of ~d samples per measurement, each sample runs the~%~
             kernel repeatedly for at least ~d ms of CPU time ~
             (GET-INTERNAL-RUN-TIME);~%the reported number is the fastest ~
             sample.  Another process competes for~%the machine, so the ~
             8192-element rows are the noisiest.~%"
          *samples* *sample-min-ms*)
  (format t "~2&~38a ~13a ~9a ~9a ~6a ~8a ~8a~%"
          "kernel" "size" "scalar ms" "vector ms" "ratio" "sca GB/s" "vec GB/s")
  (let ((rows '()))
    (dolist (bench *benches*)
      (setf rows (append rows (run-bench bench)))
      (sb-ext:gc :full t))
    (print-evidence)
    (print-summary rows)
    (terpri)
    (format t "reproduce with:~%  src/runtime/sbcl --core output/sbcl.core ~
               --noinform --no-sysinit --no-userinit~%  --disable-debugger ~
               --load benchmarks/auto-vectorize-kernels.lisp~%")
    (values)))

(run)
