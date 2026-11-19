
;;;; Tests of automatic vectorization on arm64, where the packed instructions are
;;;; NEON.

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

;;; tests/auto-vectorize.pure.lisp covers the pass itself against the x86-64
;;; instructions it was written for.  This file covers the same contract on the
;;; backend that emits NEON: the loops compute what the scalar loop computes,
;;; a length guard keeps a packed store inside the array, and the instruction
;;; the primitive table advertises is the one the code contains.

#-arm64 (invoke-restart 'run-tests::skip-file)

(deftype vectorize-index () `(integer 0 (,array-dimension-limit)))

;;; 0 runs no iterations, 1 through 9 cross the hand-off from the packed loop to
;;; the scalar tail (four elements at a time for singles, sixteen for bytes),
;;; 16 and 17 straddle a second vector, and 1000 is long enough that a lane
;;; shift cannot hide in a short loop.
(defparameter *vectorize-lengths* '(0 1 2 3 4 5 7 8 9 16 17 1000))

(defparameter *single-values*
  '(1.5f0 -2.25f0 0.0f0 3.0f0 -0.5f0 7.75f0 -1.0f0 0.125f0 2.0f0 -4.5f0))
(defparameter *double-values*
  '(1.5d0 -2.25d0 0.0d0 3.0d0 -0.5d0 7.75d0 -1.0d0 0.125d0 2.0d0 -4.5d0))
(defparameter *divisors*
  '(2.0f0 -4.0f0 0.5f0 8.0f0 -1.25f0 3.0f0 -2.0f0 1.5f0 0.25f0 -16.0f0))

(defparameter *u8-values* '(0 1 2 3 7 11 100 255 200 17))
(defparameter *s8-values* '(0 1 -2 3 -7 11 -100 127 -128 17))
(defparameter *u16-values* '(0 1 2 3 7 11 1000 65535 30000 17))
(defparameter *s16-values* '(0 1 -2 3 -7 11 -1000 32767 -32768 17))
(defparameter *u32-values* '(0 1 2 3 7 11 100 4294967295 1000 65535))
(defparameter *s32-values* '(0 1 -2 3 -7 11 -100 2147483647 -2147483648 65535))
(defparameter *u64-values* '(0 1 2 3 1099511627776 4294967296 1000 65535
                             18446744073709551615 17))
(defparameter *s64-values* '(0 1 -2 3 -1099511627776 4294967296 -1000 65535
                             9223372036854775807 -9223372036854775808))

(defparameter *integer-guard* 42)

;;; A value that the element type can hold, as the fill for the elements past the
;;; loop bound and as the expected value they keep.
(defun guard-value (element-type)
  (coerce (if (subtypep element-type 'integer) *integer-guard* -1) element-type))

;;; VALUE as it is stored in a lane of BITS bits.
(defun wrap (value bits)
  (let ((v (ldb (byte bits 0) value)))
    (if (logbitp (1- bits) v) (- v (ash 1 bits)) v)))

;;; VALUE as an array of ELEMENT-TYPE holds it: a packed lane and the scalar
;;; store wrap the same way.
(defun truncate-to (value element-type)
  (if (subtypep element-type 'integer)
      (if (eq (first element-type) 'signed-byte)
          (wrap value (second element-type))
          (ldb (byte (second element-type) 0) value))
      value))

;;;; Helpers.

(defun fill-from (values count &optional (offset 0))
  (loop for i from 0 below count
        collect (nth (mod (+ i offset) (length values)) values)))

(defun make-vector (element-type values count &optional (offset 0))
  (make-array count :element-type element-type
              :initial-contents (fill-from values count offset)))

(defun disassembly-text (function)
  (string-downcase
   (with-output-to-string (stream) (disassemble function :stream stream))))

;;; A NEON instruction names the lane count and width it works on, and nothing
;;; outside the vector unit does, so a lane suffix is evidence of a packed
;;; instruction whatever the mnemonic.
(defun packed-p (function)
  (let ((text (disassembly-text function)))
    (some (lambda (suffix) (search suffix text))
          '(".16b" ".8h" ".4s" ".2d"))))

(defun uses (function &rest mnemonics)
  "True when every mnemonic in MNEMONICS names an instruction of FUNCTION."
  (let ((text (disassembly-text function)))
    (every (lambda (mnemonic) (search mnemonic text)) mnemonics)))

(defun check-elementwise (function operation element-type values-a values-b context)
  (loop for n in *vectorize-lengths*
        for a = (make-vector element-type values-a n)
        for b = (make-vector element-type values-b n 3)
        ;; eight spare elements: a packed store past N would land in them
        for guard = (guard-value element-type)
        for c = (make-array (+ n 8) :element-type element-type
                              :initial-element guard)
        do (funcall function a b c n)
           (dotimes (i n)
             (let ((want (truncate-to (funcall operation (aref a i) (aref b i))
                                      element-type))
                   (got (aref c i)))
               (unless (equalp got want)
                 (error "~@<~A: element ~D of ~D is ~S, expected ~S.~:@>"
                        context i n got want))))
           (loop for i from n below (+ n 8)
                 do (unless (eql (aref c i) guard)
                      (error "~@<~A: element ~D is past the loop bound but was ~
                              written: ~S.~:@>"
                             context i (aref c i))))))

(defun check-floats (function operation element-type values-a values-b context)
  (loop for n in *vectorize-lengths*
        for a = (make-vector element-type values-a n)
        for b = (make-vector element-type values-b n 3)
        for guard = (guard-value element-type)
        for c = (make-array (+ n 8) :element-type element-type :initial-element guard)
        do (funcall function a b c n)
           (dotimes (i n)
             (let ((want (truncate-to (funcall operation (aref a i) (aref b i))
                                      element-type))
                   (got (aref c i)))
               (unless (= got want)
                 (error "~@<~A: element ~D of ~D is ~S, expected ~S.~:@>"
                        context i n got want))))
           ;; every spare element, not just the last, and also when N is zero
           (loop for i from n below (+ n 8)
                 do (unless (eql (aref c i) guard)
                      (error "~@<~A: element ~D is past the loop bound but was ~
                              written: ~S.~:@>"
                             context i (aref c i))))))

;;;; The loops.

(defun vec-single-add (a b c n)
  (declare (type (simple-array single-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i)))))

(defun vec-single-sub-mul-div (a b c n)
  (declare (type (simple-array single-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (/ (- (aref a i) (aref b i)) (* 2.0 (aref b i))))))

(defun vec-single-negate-scale (a c n)
  (declare (type (simple-array single-float (*)) a c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (- (* (aref a i) 1.1)))))

(defun vec-double-add (a b c n)
  (declare (type (simple-array double-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i)))))

(defun vec-double-negate (a b c n)
  (declare (type (simple-array double-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (declare (ignore b))
  (dotimes (i n) (setf (aref c i) (- (aref a i)))))

(defun vec-double-div (a b c n)
  (declare (type (simple-array double-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (/ (aref a i) (aref b i)))))

(defmacro define-int-loop (name element-type body)
  `(defun ,name (a b c n)
     (declare (type (simple-array ,element-type (*)) a b c)
              (type vectorize-index n)
              (optimize speed (safety 0)))
     (dotimes (i n) (setf (aref c i) ,body))))

(define-int-loop vec-u8-sub (unsigned-byte 8) (- (aref a i) (aref b i)))
(define-int-loop vec-s8-add (signed-byte 8) (+ (aref a i) (aref b i)))
(define-int-loop vec-u16-sub (unsigned-byte 16) (- (aref a i) (aref b i)))
(define-int-loop vec-s16-add (signed-byte 16) (+ (aref a i) (aref b i)))
(define-int-loop vec-u32-add (unsigned-byte 32) (+ (aref a i) (aref b i)))
(define-int-loop vec-s32-logand (signed-byte 32) (logand (aref a i) (aref b i)))
(define-int-loop vec-u64-logxor (unsigned-byte 64) (logxor (aref a i) (aref b i)))
(define-int-loop vec-s64-sub (signed-byte 64) (- (aref a i) (aref b i)))
(define-int-loop vec-u16-mul (unsigned-byte 16) (* (aref a i) (aref b i)))
(define-int-loop vec-s16-mul (signed-byte 16) (* (aref a i) (aref b i)))
(define-int-loop vec-u32-mul (unsigned-byte 32) (* (aref a i) (aref b i)))
(define-int-loop vec-u8-mul (unsigned-byte 8) (* (aref a i) (aref b i)))
(define-int-loop vec-s8-mul (signed-byte 8) (* (aref a i) (aref b i)))

;;; the bitwise operators are the same instruction at every lane width, but a
;;; wrong lane suffix or a narrow source would still pass a value test of one
;;; width, so each width with each signedness is exercised.
(define-int-loop vec-u8-logand (unsigned-byte 8) (logand (aref a i) (aref b i)))
(define-int-loop vec-u16-logand (unsigned-byte 16) (logand (aref a i) (aref b i)))
(define-int-loop vec-u64-logand (unsigned-byte 64) (logand (aref a i) (aref b i)))
(define-int-loop vec-u8-logior (unsigned-byte 8) (logior (aref a i) (aref b i)))
(define-int-loop vec-s8-logior (signed-byte 8) (logior (aref a i) (aref b i)))
(define-int-loop vec-u16-logior (unsigned-byte 16) (logior (aref a i) (aref b i)))
(define-int-loop vec-s16-logior (signed-byte 16) (logior (aref a i) (aref b i)))
(define-int-loop vec-u32-logior (unsigned-byte 32) (logior (aref a i) (aref b i)))
(define-int-loop vec-s32-logior (signed-byte 32) (logior (aref a i) (aref b i)))
(define-int-loop vec-u64-logior (unsigned-byte 64) (logior (aref a i) (aref b i)))
(define-int-loop vec-s64-logior (signed-byte 64) (logior (aref a i) (aref b i)))
(define-int-loop vec-u8-logxor (unsigned-byte 8) (logxor (aref a i) (aref b i)))
(define-int-loop vec-s8-logxor (signed-byte 8) (logxor (aref a i) (aref b i)))
(define-int-loop vec-u16-logxor (unsigned-byte 16) (logxor (aref a i) (aref b i)))
(define-int-loop vec-u32-logxor (unsigned-byte 32) (logxor (aref a i) (aref b i)))

;;; minimum and maximum: NEON has every lane width and both signednesses, so
;;; these vectorize at the default policy, without a SAFETY 0 declaration.
(define-int-loop vec-u8-clamp (unsigned-byte 8) (min (aref a i) 200))
(define-int-loop vec-u8-max (unsigned-byte 8) (max (aref a i) (aref b i)))
(define-int-loop vec-s8-min (signed-byte 8) (min (aref a i) (aref b i)))
(define-int-loop vec-u16-max (unsigned-byte 16) (max (aref a i) (aref b i)))
(define-int-loop vec-s16-clamp (signed-byte 16)
  (max -1000 (min 1000 (aref a i))))
(define-int-loop vec-s32-clamp (signed-byte 32)
  (max -1000000 (min 1000000 (aref a i))))
(define-int-loop vec-u32-min (unsigned-byte 32) (min (aref a i) 4000000000))

;;; A variable broadcast into lanes narrower than a word.
(defun vec-u8-add-variable (a c k n)
  (declare (type (simple-array (unsigned-byte 8) (*)) a c)
           (type (unsigned-byte 8) k) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) k))))

(defun vec-s32-add-variable (a c k n)
  (declare (type (simple-array (signed-byte 32) (*)) a c)
           (type (signed-byte 32) k) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) k))))

(defun vec-u64-add-variable (a c k n)
  (declare (type (simple-array (unsigned-byte 64) (*)) a c)
           (type (unsigned-byte 64) k) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) k))))

;;; A scalar declared FIXNUM reaches a 64-bit lane loop as an untagged word,
;;; whatever primitive type it was represented by on the way in: the move into
;;; the vop's untagged storage class is what strips the tag.  This function pins
;;; that, since an untagging inside the splat halves the constant instead.  A
;;; fixnum scalar matches only the signed 64-bit lanes - the front end classes it
;;; (signed-byte 64), which is not the (unsigned-byte 64) of an unsigned loop -
;;; so the unsigned side is covered by vec-u64-add-variable above.
(defun vec-s64-add-fixnum (a c k n)
  (declare (type (simple-array (signed-byte 64) (*)) a c)
           (type fixnum k) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) k))))


;;;; the index ramp

(defun vec-gradient (c n)
  (declare (type (simple-array single-float (*)) c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (coerce i 'single-float))))

(defun vec-gradient-double (c n)
  (declare (type (simple-array double-float (*)) c) (type vectorize-index n)
           (optimize speed))
  ;; (FLOAT I) would be a single-float conversion, which the pass does not accept
  ;; as an index; COERCE gives it a double-float ramp to build.
  (dotimes (i n) (setf (aref c i) (+ 0.5d0 (coerce i 'double-float)))))

;;;; reading a narrower source into float lanes

(defun vec-u8-to-single (in out n)
  (declare (type (simple-array (unsigned-byte 8) (*)) in)
           (type (simple-array single-float (*)) out)
           (type vectorize-index n) (optimize speed))
  (dotimes (i n) (setf (aref out i) (* 0.5 (aref in i)))))

(defun vec-s8-to-single (in out n)
  (declare (type (simple-array (signed-byte 8) (*)) in)
           (type (simple-array single-float (*)) out)
           (type vectorize-index n) (optimize speed))
  (dotimes (i n) (setf (aref out i) (+ 10.0 (aref in i)))))

(defun vec-u8-to-double (in out n)
  (declare (type (simple-array (unsigned-byte 8) (*)) in)
           (type (simple-array double-float (*)) out)
           (type vectorize-index n) (optimize speed))
  (dotimes (i n) (setf (aref out i) (* 0.5d0 (aref in i)))))

(defun vec-s8-to-double (in out n)
  (declare (type (simple-array (signed-byte 8) (*)) in)
           (type (simple-array double-float (*)) out)
           (type vectorize-index n) (optimize speed))
  (dotimes (i n) (setf (aref out i) (+ 0.25d0 (aref in i)))))

(defun vec-u16-to-double (in out n)
  (declare (type (simple-array (unsigned-byte 16) (*)) in)
           (type (simple-array double-float (*)) out)
           (type vectorize-index n) (optimize speed))
  (dotimes (i n) (setf (aref out i) (+ 1d0 (aref in i)))))

(defun vec-s16-to-double (in out n)
  (declare (type (simple-array (signed-byte 16) (*)) in)
           (type (simple-array double-float (*)) out)
           (type vectorize-index n) (optimize speed))
  (dotimes (i n) (setf (aref out i) (* 0.5d0 (aref in i)))))

(defun vec-single-to-double (in out n)
  (declare (type (simple-array single-float (*)) in)
           (type (simple-array double-float (*)) out)
           (type vectorize-index n) (optimize speed))
  (dotimes (i n) (setf (aref out i) (* 2d0 (aref in i)))))

;;;; a reduction, and a stencil that reads ahead of what it writes

(defun vec-u32-sum (a n total)
  (declare (type (simple-array (unsigned-byte 32) (*)) a)
           (type vectorize-index n) (type (unsigned-byte 32) total)
           (optimize speed (safety 0)))
  (dotimes (i n) (incf total (aref a i)))
  total)

(defun vec-three-tap (in out n)
  (declare (type (simple-array single-float (*)) in)
           (type (simple-array single-float (*)) out)
           (type vectorize-index n) (optimize speed))
  (dotimes (i n)
    (setf (aref out (+ i 2)) (+ (aref in i) (aref in (+ i 1))))))

;;;; Tests.

(test-util:with-test (:name :vectorize-arm64-float)
  (check-floats #'vec-single-add #'+ 'single-float
                *single-values* *single-values* 'vec-single-add)
  (check-floats #'vec-single-sub-mul-div
                (lambda (a b) (/ (- a b) (* 2.0 b))) 'single-float
                *single-values* *divisors* 'vec-single-sub-mul-div)
  (check-floats #'vec-double-add #'+ 'double-float
                *double-values* *double-values* 'vec-double-add)
  (check-floats #'vec-double-negate (lambda (a b) (declare (ignore b)) (- a))
                'double-float *double-values* *double-values* 'vec-double-negate)
  (assert (uses #'vec-double-negate "fneg" ".2d") () "double negation is not packed")
  (check-floats #'vec-double-div #'/ 'double-float
                *double-values* (mapcar (lambda (x) (coerce x 'double-float)) *divisors*)
                'vec-double-div)
  (loop for n in *vectorize-lengths*
        for a = (make-vector 'single-float *single-values* n)
        for c = (make-array (+ n 8) :element-type 'single-float :initial-element 0.0)
        do (vec-single-negate-scale a c n)
           (dotimes (i n)
             (unless (= (aref c i) (- (* (aref a i) 1.1)))
               (error "negate/scale is wrong at element ~D of ~D" i n)))
           (unless (= (aref c (+ n 7)) 0.0)
             (error "negate/scale ran past the loop bound")))
  (assert (uses #'vec-single-add "fadd" ".4s") () "single add is not packed")
  (assert (uses #'vec-double-add "fadd" ".2d") () "double add is not packed")
  (assert (uses #'vec-single-sub-mul-div "fsub" "fmul" "fdiv") ()
          "sub/mul/div are not all packed")
  (assert (uses #'vec-single-negate-scale "fneg") () "packed negation is missing"))

(test-util:with-test (:name :vectorize-arm64-float-aliasing)
  ;; the destination is one of the sources: the same values as the scalar loop
  (loop for n in '(4 5 9 16 1000)
        for a = (make-vector 'single-float *single-values* n)
        for b = (make-vector 'single-float *single-values* n 3)
        for want = (map 'vector (lambda (x y) (+ x y)) a b)
        do (let ((copy (copy-seq a)))
             (vec-single-add copy b copy n)
             (assert (equalp copy want) () "aliased destination is wrong"))
           (let ((copy (copy-seq b)))
             (vec-single-add a copy copy n)
             (assert (equalp copy want) () "aliased second source is wrong"))))

(test-util:with-test (:name :vectorize-arm64-length-guard)
  ;; a count past the end must still signal, not write past the array
  (let ((a (make-array 4 :element-type 'single-float :initial-element 1.0))
        (b (make-array 4 :element-type 'single-float :initial-element 1.0))
        (c (make-array 4 :element-type 'single-float :initial-element 0.0)))
    (assert-error (vec-single-add a b c 8))
    (assert-error (vec-single-add a b c 5))))

(test-util:with-test (:name :vectorize-arm64-integer)
  (check-elementwise #'vec-u8-sub #'- '(unsigned-byte 8)
                     *u8-values* *u8-values* 'vec-u8-sub)
  (check-elementwise #'vec-s8-add #'+ '(signed-byte 8)
                     *s8-values* *s8-values* 'vec-s8-add)
  (check-elementwise #'vec-u16-sub #'- '(unsigned-byte 16)
                     *u16-values* *u16-values* 'vec-u16-sub)
  (check-elementwise #'vec-s16-add #'+ '(signed-byte 16)
                     *s16-values* *s16-values* 'vec-s16-add)
  (check-elementwise #'vec-u32-add #'+ '(unsigned-byte 32)
                     *u32-values* *u32-values* 'vec-u32-add)
  (check-elementwise #'vec-s32-logand #'logand '(signed-byte 32)
                     *s32-values* *s32-values* 'vec-s32-logand)
  (check-elementwise #'vec-u64-logxor #'logxor '(unsigned-byte 64)
                     *u64-values* *u64-values* 'vec-u64-logxor)
  (check-elementwise #'vec-s64-sub #'- '(signed-byte 64)
                     *s64-values* *s64-values* 'vec-s64-sub)
  (assert (uses #'vec-u8-sub "sub" ".16b") () "byte lanes are not packed")
  (assert (uses #'vec-u16-sub "sub" ".8h") () "16-bit lanes are not packed")
  (assert (uses #'vec-u32-add "add" ".4s") () "32-bit lanes are not packed")
  (assert (uses #'vec-s64-sub "sub" ".2d") () "64-bit lanes are not packed")
  (assert (uses #'vec-s32-logand "and") () "packed LOGAND is missing")
  (assert (uses #'vec-u64-logxor "eor") () "packed LOGXOR is missing"))

(test-util:with-test (:name :vectorize-arm64-broadcast)
  (loop for k in '(0 1 200 255)
        do (loop for n in '(0 1 3 16 17 100)
                 for a = (make-vector '(unsigned-byte 8) *u8-values* n)
                 for c = (make-array n :element-type '(unsigned-byte 8) :initial-element 0)
                 do (vec-u8-add-variable a c k n)
                    (dotimes (i n)
                      (unless (= (aref c i) (mod (+ (aref a i) k) 256))
                        (error "u8 broadcast is wrong at ~D of ~D" i n)))))
  (assert (uses #'vec-u8-add-variable "dup") () "byte broadcast is not packed")
  (loop for k in '(0 -7 12345)
        do (loop for n in '(0 4 9 17 100)
                 for a = (make-vector '(signed-byte 32) *s32-values* n)
                 for c = (make-array n :element-type '(signed-byte 32) :initial-element 0)
                 do (vec-s32-add-variable a c k n)
                    (dotimes (i n)
                      (unless (= (aref c i) (wrap (+ (aref a i) k) 32))
                        (error "s32 broadcast is wrong at ~D of ~D" i n)))))
  (loop for n in '(0 2 3 17)
        for a = (make-array n :element-type '(unsigned-byte 64)
                              :initial-contents (fill-from *u64-values* n))
        for c = (make-array n :element-type '(unsigned-byte 64) :initial-element 0)
        do (vec-u64-add-variable a c 1099511627776 n)
           (dotimes (i n)
             (unless (= (aref c i) (ldb (byte 64 0) (+ (aref a i) 1099511627776)))
               (error "u64 broadcast is wrong at ~D of ~D" i n))))
  ;; the tag is stripped on the way in, so every bit the value has must survive
  (loop for k in (list (ash 1 40) (1- (ash 1 62)) 255 0)
        do (loop for n in '(0 2 3 17)
                 for a = (make-array n :element-type '(unsigned-byte 64)
                                     :initial-contents (fill-from *u64-values* n))
                 for c = (make-array n :element-type '(unsigned-byte 64)
                                     :initial-element 0)
                 do (vec-u64-add-variable a c k n)
                    (dotimes (i n)
                      (unless (= (aref c i) (ldb (byte 64 0) (+ (aref a i) k)))
                        (error "u64 broadcast of ~D is wrong at ~D of ~D" k i n)))))
  (assert (uses #'vec-u64-add-variable "dup") () "64-bit broadcast is not packed")
  (loop for k in '(-1 -7 12345 -1099511627776 1099511627776)
        do (loop for n in '(0 2 3 17)
                 for a = (make-array n :element-type '(signed-byte 64)
                                     :initial-contents (fill-from *s64-values* n))
                 for c = (make-array n :element-type '(signed-byte 64) :initial-element 0)
                 do (vec-s64-add-fixnum a c k n)
                    (dotimes (i n)
                      (unless (= (aref c i) (wrap (+ (aref a i) k) 64))
                        (error "s64 broadcast of a fixnum ~D is wrong at ~D of ~D"
                               k i n)))))
  (assert (uses #'vec-s64-add-fixnum "dup") () "64-bit broadcast of a fixnum is not packed")
)

(test-util:with-test (:name :vectorize-arm64-min-max)
  (check-elementwise #'vec-u8-clamp (lambda (a b) (declare (ignore b)) (min a 200))
                     '(unsigned-byte 8) *u8-values* *u8-values* 'vec-u8-clamp)
  (check-elementwise #'vec-s8-min #'min '(signed-byte 8)
                     *s8-values* *s8-values* 'vec-s8-min)
  (check-elementwise #'vec-u16-max #'max '(unsigned-byte 16)
                     *u16-values* *u16-values* 'vec-u16-max)
  (check-elementwise #'vec-u32-min (lambda (a b) (declare (ignore b)) (min a 4000000000))
                     '(unsigned-byte 32) *u32-values* *u32-values* 'vec-u32-min)
  (loop for n in *vectorize-lengths*
        for a = (make-vector '(signed-byte 16) *s16-values* n)
        for c = (make-array n :element-type '(signed-byte 16) :initial-element 0)
        do (vec-s16-clamp a a c n)
           (dotimes (i n)
             (unless (= (aref c i) (max -1000 (min 1000 (aref a i))))
               (error "s16 clamp is wrong at ~D of ~D" i n))))
  (loop for n in *vectorize-lengths*
        for a = (make-vector '(signed-byte 32) *s32-values* n)
        for c = (make-array n :element-type '(signed-byte 32) :initial-element 0)
        do (vec-s32-clamp a a c n)
           (dotimes (i n)
             (unless (= (aref c i) (max -1000000 (min 1000000 (aref a i))))
               (error "s32 clamp is wrong at ~D of ~D" i n))))
  (assert (uses #'vec-u8-clamp "umin") () "unsigned byte minimum is not packed")
  (assert (uses #'vec-s8-min "smin" ".16b") () "signed byte minimum is not packed")
  (assert (uses #'vec-u16-max "umax" ".8h") () "unsigned word maximum is not packed")
  (assert (uses #'vec-s16-clamp "smin" "smax") () "signed word clamp is not packed")
  (assert (uses #'vec-s32-clamp "smin" "smax" ".4s") () "32-bit clamp is not packed")
  (assert (uses #'vec-u32-min "umin" ".4s") () "unsigned 32-bit minimum is not packed")
  (check-elementwise #'vec-u8-max #'max '(unsigned-byte 8)
                     *u8-values* *u8-values* 'vec-u8-max)
  (assert (uses #'vec-u8-max "umax" ".16b") () "unsigned byte maximum is not packed"))

(test-util:with-test (:name :vectorize-arm64-multiply)
  (check-elementwise #'vec-u16-mul #'*
                     '(unsigned-byte 16) *u16-values* *u16-values* 'vec-u16-mul)
  (check-elementwise #'vec-s16-mul #'*
                     '(signed-byte 16) *s16-values* *s16-values* 'vec-s16-mul)
  (check-elementwise #'vec-u32-mul #'*
                     '(unsigned-byte 32) *u32-values* *u32-values* 'vec-u32-mul)
  (check-elementwise #'vec-u8-mul #'* '(unsigned-byte 8)
                     *u8-values* *u8-values* 'vec-u8-mul)
  (check-elementwise #'vec-s8-mul #'* '(signed-byte 8)
                     *s8-values* *s8-values* 'vec-s8-mul)
  (assert (uses #'vec-u16-mul "mul" ".8h") () "16-bit multiply is not packed")
  (assert (uses #'vec-u32-mul "mul" ".4s") () "32-bit multiply is not packed")
  (assert (uses #'vec-u8-mul "mul" ".16b") () "byte multiply is not packed")
  (assert (uses #'vec-s8-mul "mul" ".16b") () "signed byte multiply is not packed"))

(test-util:with-test (:name :vectorize-arm64-bitwise)
  (check-elementwise #'vec-u8-logand #'logand '(unsigned-byte 8)
                     *u8-values* *u8-values* 'vec-u8-logand)
  (check-elementwise #'vec-u16-logand #'logand '(unsigned-byte 16)
                     *u16-values* *u16-values* 'vec-u16-logand)
  (check-elementwise #'vec-u64-logand #'logand '(unsigned-byte 64)
                     *u64-values* *u64-values* 'vec-u64-logand)
  (check-elementwise #'vec-u8-logior #'logior '(unsigned-byte 8)
                     *u8-values* *u8-values* 'vec-u8-logior)
  (check-elementwise #'vec-s8-logior #'logior '(signed-byte 8)
                     *s8-values* *s8-values* 'vec-s8-logior)
  (check-elementwise #'vec-u16-logior #'logior '(unsigned-byte 16)
                     *u16-values* *u16-values* 'vec-u16-logior)
  (check-elementwise #'vec-s16-logior #'logior '(signed-byte 16)
                     *s16-values* *s16-values* 'vec-s16-logior)
  (check-elementwise #'vec-u32-logior #'logior '(unsigned-byte 32)
                     *u32-values* *u32-values* 'vec-u32-logior)
  (check-elementwise #'vec-s32-logior #'logior '(signed-byte 32)
                     *s32-values* *s32-values* 'vec-s32-logior)
  (check-elementwise #'vec-u64-logior #'logior '(unsigned-byte 64)
                     *u64-values* *u64-values* 'vec-u64-logior)
  (check-elementwise #'vec-s64-logior #'logior '(signed-byte 64)
                     *s64-values* *s64-values* 'vec-s64-logior)
  (check-elementwise #'vec-u8-logxor #'logxor '(unsigned-byte 8)
                     *u8-values* *u8-values* 'vec-u8-logxor)
  (check-elementwise #'vec-s8-logxor #'logxor '(signed-byte 8)
                     *s8-values* *s8-values* 'vec-s8-logxor)
  (check-elementwise #'vec-u16-logxor #'logxor '(unsigned-byte 16)
                     *u16-values* *u16-values* 'vec-u16-logxor)
  (check-elementwise #'vec-u32-logxor #'logxor '(unsigned-byte 32)
                     *u32-values* *u32-values* 'vec-u32-logxor)
  ;; NEON's AND, ORR and EOR work on bytes whatever the element type is, so the
  ;; byte-wise lane form is the one to look for here.
  (assert (uses #'vec-u8-logand "and" ".16b") () "byte AND is not packed")
  (assert (uses #'vec-u16-logand "and" ".16b") () "word AND is not packed")
  (assert (uses #'vec-u64-logand "and" ".16b") () "64-bit AND is not packed")
  (assert (uses #'vec-u8-logior "orr" ".16b") () "byte OR is not packed")
  (assert (uses #'vec-s8-logior "orr" ".16b") () "signed byte OR is not packed")
  (assert (uses #'vec-u16-logior "orr" ".16b") () "word OR is not packed")
  (assert (uses #'vec-s16-logior "orr" ".16b") () "signed word OR is not packed")
  (assert (uses #'vec-u32-logior "orr" ".16b") () "32-bit OR is not packed")
  (assert (uses #'vec-s32-logior "orr" ".16b") () "signed 32-bit OR is not packed")
  (assert (uses #'vec-u64-logior "orr" ".16b") () "64-bit OR is not packed")
  (assert (uses #'vec-s64-logior "orr" ".16b") () "signed 64-bit OR is not packed")
  (assert (uses #'vec-u8-logxor "eor" ".16b") () "byte XOR is not packed")
  (assert (uses #'vec-s8-logxor "eor" ".16b") () "signed byte XOR is not packed")
  (assert (uses #'vec-u16-logxor "eor" ".16b") () "word XOR is not packed")
  (assert (uses #'vec-u32-logxor "eor" ".16b") () "32-bit XOR is not packed"))

(test-util:with-test (:name :vectorize-arm64-index-ramp)
  (loop for n in *vectorize-lengths*
        for c = (make-array n :element-type 'single-float :initial-element -1.0)
        do (vec-gradient c n)
           (dotimes (i n)
             (unless (= (aref c i) (coerce i 'single-float))
               (error "gradient is wrong at ~D of ~D: ~S" i n (aref c i)))))
  (loop for n in *vectorize-lengths*
        for c = (make-array n :element-type 'double-float :initial-element -1d0)
        do (vec-gradient-double c n)
           (dotimes (i n)
             (unless (= (aref c i) (+ 0.5d0 i))
               (error "double gradient is wrong at ~D of ~D" i n))))
  (assert (uses #'vec-gradient "fadd" ".4s") () "the index ramp is not packed")
  (assert (uses #'vec-gradient-double "fadd" ".2d")
          () "the double index ramp is not packed"))

(test-util:with-test (:name :vectorize-arm64-widen)
  (loop for n in *vectorize-lengths*
        for in = (make-vector '(unsigned-byte 8) '(0 1 2 200 255 7 9 11) n)
        for out = (make-array n :element-type 'single-float :initial-element -1.0)
        do (vec-u8-to-single in out n)
           (dotimes (i n)
             (unless (= (aref out i) (* 0.5 (aref in i)))
               (error "u8 to single is wrong at ~D of ~D" i n))))
  (loop for n in *vectorize-lengths*
        for in = (make-vector '(signed-byte 8) '(0 -1 2 127 -128 7 9 11) n)
        for out = (make-array n :element-type 'single-float :initial-element -1.0)
        do (vec-s8-to-single in out n)
           (dotimes (i n)
             (unless (= (aref out i) (+ 10.0 (aref in i)))
               (error "s8 to single is wrong at ~D of ~D" i n))))
  (loop for n in *vectorize-lengths*
        for in = (make-vector '(unsigned-byte 16) '(0 1 2 65535 30000 7 9 11) n)
        for out = (make-array n :element-type 'double-float :initial-element -1d0)
        do (vec-u16-to-double in out n)
           (dotimes (i n)
             (unless (= (aref out i) (1+ (aref in i)))
               (error "u16 to double is wrong at ~D of ~D: ~S" i n (aref out i)))))
  (loop for n in *vectorize-lengths*
        for in = (make-vector '(signed-byte 16) '(0 -1 2 32767 -32768 7 9 11) n)
        for out = (make-array n :element-type 'double-float :initial-element -1d0)
        do (vec-s16-to-double in out n)
           (dotimes (i n)
             (unless (= (aref out i) (* 0.5d0 (aref in i)))
               (error "s16 to double is wrong at ~D of ~D: ~S" i n (aref out i)))))
  (loop for n in *vectorize-lengths*
        for in = (make-vector 'single-float '(0.0 1.5 -2.25 3.0 1.0e30 7.75 -1.0 0.125) n)
        for out = (make-array n :element-type 'double-float :initial-element -1d0)
        do (vec-single-to-double in out n)
           (dotimes (i n)
             (unless (= (aref out i) (* 2d0 (aref in i)))
               (error "single to double is wrong at ~D of ~D" i n))))
  (loop for n in *vectorize-lengths*
        for in = (make-vector '(unsigned-byte 8) '(0 1 2 200 255 7 9 11) n)
        for out = (make-array n :element-type 'double-float :initial-element -1d0)
        do (vec-u8-to-double in out n)
           (dotimes (i n)
             (unless (= (aref out i) (* 0.5d0 (aref in i)))
               (error "u8 to double is wrong at ~D of ~D: ~S" i n (aref out i)))))
  (loop for n in *vectorize-lengths*
        for in = (make-vector '(signed-byte 8) '(0 -1 2 127 -128 7 9 11) n)
        for out = (make-array n :element-type 'double-float :initial-element -1d0)
        do (vec-s8-to-double in out n)
           (dotimes (i n)
             (unless (= (aref out i) (+ 0.25d0 (aref in i)))
               (error "s8 to double is wrong at ~D of ~D: ~S" i n (aref out i)))))
  (assert (uses #'vec-u8-to-single "ushll") () "u8 to single is not a packed widen")
  (assert (uses #'vec-u8-to-double "ushll" ".2d") () "u8 to double is not a packed widen")
  (assert (uses #'vec-s8-to-double "sshll" ".2d") () "s8 to double is not a packed widen")
  (assert (uses #'vec-s8-to-single "sshll") () "s8 to single is not a packed widen")
  (assert (uses #'vec-u16-to-double "ushll") () "u16 to double is not a packed widen")
  (assert (uses #'vec-s16-to-double "sshll") () "s16 to double is not a packed widen")
  (assert (uses #'vec-single-to-double "fcvtl") () "single to double is not a packed widen"))

(test-util:with-test (:name :vectorize-arm64-reduction)
  (loop for n in '(0 1 3 4 5 16 17 100)
        for a = (make-array n :element-type '(unsigned-byte 32)
                              :initial-contents (fill-from *u32-values* n))
        for exact = (reduce #'+ a :initial-value 0)
        do (assert (= (vec-u32-sum a n 0) (mod exact (expt 2 32)))
                   () "the packed sum of ~D elements is wrong" n))
  (assert (uses #'vec-u32-sum "add" ".4s") () "the reduction is not packed"))

(test-util:with-test (:name :vectorize-arm64-stencil)
  (loop for n in '(0 1 2 3 5 8 17)
        for in = (make-vector 'single-float *single-values* (+ n 2))
        for out = (make-array (+ n 2) :element-type 'single-float :initial-element 0.0)
        do (vec-three-tap in out n)
           (dotimes (i n)
             (unless (= (aref out (+ i 2)) (+ (aref in i) (aref in (+ i 1))))
               (error "three-tap is wrong at ~D of ~D" i n))))
  (assert (packed-p #'vec-three-tap) () "a read ahead of the store was not packed"))

(test-util:with-test (:name :vectorize-arm64-undeclared-stays-scalar)
  (let ((f (compile nil '(lambda (a b c n)
                           (declare (optimize speed))
                           (dotimes (i n)
                             (setf (aref c i) (+ (aref a i) (aref b i))))))))
    (assert (not (packed-p f)) () "an undeclared loop was vectorized")))

(test-util:with-test (:name :vectorize-arm64-note)
  (let* ((stream (make-string-output-stream))
         (notes nil))
    (let ((*error-output* stream)
          (sb-c::*auto-vectorize-notes* t))
      (compile nil '(lambda (a c n)
                      (declare (type (simple-array single-float (*)) a c)
                               (type vectorize-index n)
                               (optimize speed))
                      (dotimes (i n) (setf (aref c i) (* 2.0 (aref a i)))))))
    (let ((text (get-output-stream-string stream)))
      (setf notes (search "vectorized loop over" text))
      (assert notes () "no vectorization note in:~%~A" text))))
