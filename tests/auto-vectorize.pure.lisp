;;;; Tests of automatic vectorization of elementwise loops over float arrays.

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

;;; These tests assert that a loop became instructions of a particular
;;; instruction set, so they mean something only on a backend that has the vops
;;; for it.  The pass itself declines everywhere else, so the loops are still
;;; scalar there and still compute the right values.

;;; A loop such as
;;;
;;;   (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))
;;;
;;; over arrays declared (SIMPLE-ARRAY SINGLE-FLOAT (*)) or
;;; (SIMPLE-ARRAY DOUBLE-FLOAT (*)) is rewritten into a packed SSE loop
;;; followed by a scalar tail, with a fallback branch that runs the original
;;; scalar loop when N is larger than an array.  The tests below pin down the
;;; observable behaviour of that rewrite: the same values as the scalar loop,
;;; aliased arguments, an index past the end still signalling, no writes past
;;; N, and the packed loop being there on x86-64.

;;; This file asserts the x86-64 instruction set; tests/auto-vectorize-arm64.pure.lisp
;;; covers the same contract on NEON, where the packed instructions are different.
#-x86-64 (invoke-restart 'run-tests::skip-file)

;;; INDEX, the type of a bound into a vector: (INTEGER 0 (ARRAY-DIMENSION-LIMIT)).
;;; It is spelled out rather than named because the name and the package that
;;; owns it, SB-INT:INDEX or SB-EXT:INDEX depending on the build, are not
;;; something a test should depend on.
(deftype vectorize-index () `(integer 0 (,array-dimension-limit)))

;;;; Inputs shared by every test.

;;; The lengths every correctness test runs over.  0 runs no iterations at
;;; all, 1 through 9 cross the hand-off from the packed loop to the scalar tail,
;;; 4 being a whole vector of single-floats and 8 of double-floats, 16 and 17
;;; straddle a second pair of vectors, and 1000 is long enough that a lane
;;; shift cannot hide in a short loop.
(defparameter *vectorize-lengths* '(0 1 2 3 4 5 7 8 9 16 17 1000))

;;; The input values are not all equal: positives, negatives,
;;; zeros and fractions, so that a lane read from the wrong offset shows up as
;;; a mismatch instead of an accidental match.
(defparameter *vectorize-single-values*
  '(1.5f0 -2.25f0 0.0f0 3.0f0 -0.5f0 7.75f0 -1.0f0 0.125f0 2.0f0 -4.5f0))

(defparameter *vectorize-double-values*
  '(1.5d0 -2.25d0 0.0d0 3.0d0 -0.5d0 7.75d0 -1.0d0 0.125d0 2.0d0 -4.5d0))

;;; The same numbers with the zero removed, so that division never traps.
(defparameter *vectorize-single-divisors*
  '(2.0f0 -4.0f0 0.5f0 8.0f0 -1.25f0 3.0f0 -2.0f0 1.5f0 0.25f0 -16.0f0))

(defparameter *vectorize-double-divisors*
  '(2.0d0 -4.0d0 0.5d0 8.0d0 -1.25d0 3.0d0 -2.0d0 1.5d0 0.25d0 -16.0d0))

;;; The constants the multiplying loops use.  Each is also written as a literal
;;; inside its loop, because the vectorizer supports numeric float constants
;;; but need not support reading a special variable.  Neither value is
;;; representable in binary, so a path that rounds differently from the scalar
;;; loop shows up rather than cancelling out.
(defparameter *vectorize-single-scale* 1.1f0)
(defparameter *vectorize-single-add-scale* 2.3f0)
(defparameter *vectorize-double-scale* 1.1d0)
(defparameter *vectorize-double-add-scale* 2.3d0)

;;; A packed SIMD mnemonic, any one of which is evidence that a loop was
;;; vectorized.  Which instruction is used is an implementation detail, so this
;;; is a list rather than a single expected mnemonic.
(defun vectorize-notes-for (form &key (notes t))
  "The notes compiling FORM produces that are about vectorization."
  (let ((stream (make-string-output-stream)))
    (let ((*error-output* stream)
          (sb-c::*auto-vectorize-notes* notes))
      (compile nil form))
    (let ((text (get-output-stream-string stream))
          (result '()))
      (loop with start = 0
            for newline = (position #\Newline text :start start)
            do (let ((line (subseq text start (if newline newline (length text)))))
                 ;; keep the pass's own lines: its notes and the reasons it gives
                 (when (some (lambda (marker) (search marker line :test #'char-equal))
                             '("vectoriz" "lanes" "shift the destination" "is read at"
                               "not known to be an array" "not available" "safety 0"))
                   (push line result)))
               (if newline (setf start (1+ newline)) (return)))
      (nreverse result))))

(defun vectorize-note-mentions (notes string)
  (some (lambda (line) (search string line :test #'char-equal)) notes))

(defparameter *vectorize-packed-mnemonics*
  '("addps" "addpd" "mulps" "movups" "shufps"))

;;;; Helpers.

(defun vectorize-fill-list (values count &optional (offset 0))
  "COUNT values drawn from the VALUES pattern, starting at OFFSET."
  (loop for i from 0 below count
        collect (nth (mod (+ i offset) (length values)) values)))

(defun vectorize-sentinel (element-type)
  "A value of ELEMENT-TYPE that the references below cannot produce."
  (coerce -999.25 element-type))

(defun vectorize-array (element-type values count &optional (offset 0))
  "An array of COUNT elements of ELEMENT-TYPE drawn from the VALUES pattern."
  (make-array count
              :element-type element-type
              :initial-contents (vectorize-fill-list values count offset)))

(defun vectorize-guarded-array (element-type values count &optional (offset 0)
                                                                   (guard 0))
  "Like VECTORIZE-ARRAY, but the last GUARD elements are the sentinel.  The
arrays passed to the loops under test are longer than N, so an element that a
packed store wrote past N is recognizable instead of invisible."
  (let ((array (vectorize-array element-type values count offset))
        (sentinel (vectorize-sentinel element-type)))
    (loop for i from (- count guard) below count
          do (setf (aref array i) sentinel))
    array))

(defun vectorize-sentinel-array (element-type count)
  (make-array count
              :element-type element-type
              :initial-element (vectorize-sentinel element-type)))

(defun vectorize-same-value-p (x y)
  "True when a value computed by a vectorized loop is the one the scalar loop
computes.  The vector path must agree exactly, not merely closely."
  (and (= x y)
       (or (zerop x)
           (= (float-sign x) (float-sign y)))))

(defun vectorize-check-values (actual expected count context)
  (dotimes (i count)
    (let ((got (aref actual i))
          (want (nth i expected)))
      (unless (vectorize-same-value-p got want)
        (error "~@<~A: element ~D of ~D is ~S, expected ~S.~:@>"
               context i count got want)))))

(defun vectorize-check-untouched (array element-type start count context)
  "The COUNT elements of ARRAY starting at START were never written."
  (let ((sentinel (vectorize-sentinel element-type)))
    (loop for i from start below (+ start count)
          do (unless (eql (aref array i) sentinel)
               (error "~@<~A: element ~D is past the loop bound but was written: ~S.~:@>"
                      context i (aref array i))))))

(defun vectorize-check-elementwise (vectorized operation element-type
                                    values-a values-b context)
  "Run VECTORIZED (a (A B C N) loop) for every test length with N elements and
compare against OPERATION applied to the same inputs as plain numbers, which
does not involve any array loop and so cannot itself be vectorized."
  (loop for n in *vectorize-lengths*
        for a-list = (vectorize-fill-list values-a n)
        for b-list = (vectorize-fill-list values-b n 3)
        for expected = (mapcar (lambda (x y)
                                 (coerce (funcall operation x y) element-type))
                               a-list b-list)
        for a = (vectorize-guarded-array element-type values-a (+ n 3) 0 3)
        for b = (vectorize-guarded-array element-type values-b (+ n 3) 3 3)
        for c = (vectorize-sentinel-array element-type (+ n 3))
        for description = (list context (list element-type n))
        do (funcall vectorized a b c n)
           (vectorize-check-values c expected n description)
           (vectorize-check-untouched c element-type n 3 description)))

(defun vectorize-check-unary (vectorized operation element-type values context)
  "Like VECTORIZE-CHECK-ELEMENTWISE, for a (A C N) loop."
  (loop for n in *vectorize-lengths*
        for a-list = (vectorize-fill-list values n)
        for expected = (mapcar (lambda (x)
                                 (coerce (funcall operation x) element-type))
                               a-list)
        for a = (vectorize-guarded-array element-type values (+ n 3) 0 3)
        for c = (vectorize-sentinel-array element-type (+ n 3))
        for description = (list context (list element-type n))
        do (funcall vectorized a c n)
           (vectorize-check-values c expected n description)
           (vectorize-check-untouched c element-type n 3 description)))

;;; Aliasing is only exercised with C = A + B: whether the packed loop reads a
;;; vector of sources before storing it is independent of the operation.
(defun vectorize-check-aliasing (vectorized element-type values-a values-b context)
  (loop for n in *vectorize-lengths*
        for a-list = (vectorize-fill-list values-a n)
        for b-list = (vectorize-fill-list values-b n 3)
        for sums = (mapcar (lambda (x y) (coerce (+ x y) element-type))
                           a-list b-list)
        for doubles = (mapcar (lambda (x) (coerce (+ x x) element-type)) a-list)
        for description = (list context (list element-type n))
        do
           ;; C is A: the destination array is also the first source.
           (let ((a (vectorize-guarded-array element-type values-a (+ n 3) 0 3))
                 (b (vectorize-guarded-array element-type values-b (+ n 3) 3 3)))
             (funcall vectorized a b a n)
             (vectorize-check-values a sums n (list description 'c-is-a))
             (vectorize-check-untouched a element-type n 3 (list description 'c-is-a)))
           ;; A is B: both sources are the same array.
           (let ((a (vectorize-guarded-array element-type values-a (+ n 3) 0 3))
                 (c (vectorize-sentinel-array element-type (+ n 3))))
             (funcall vectorized a a c n)
             (vectorize-check-values c doubles n (list description 'a-is-b))
             (vectorize-check-untouched c element-type n 3 (list description 'a-is-b)))))

;;;; The loops under test, in exactly the shape the vectorizer understands:
;;;; declared simple float arrays, an INDEX bound, an elementwise assignment.

;;;; single-float

(defun vectorize-single-add (a b c n)
  (declare (type (simple-array single-float (*)) a b c)
           (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (+ (aref a i) (aref b i)))))

(defun vectorize-single-sub (a b c n)
  (declare (type (simple-array single-float (*)) a b c)
           (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (- (aref a i) (aref b i)))))

(defun vectorize-single-mul (a b c n)
  (declare (type (simple-array single-float (*)) a b c)
           (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (* (aref a i) (aref b i)))))

(defun vectorize-single-div (a b c n)
  (declare (type (simple-array single-float (*)) a b c)
           (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (/ (aref a i) (aref b i)))))

(defun vectorize-single-negate (a c n)
  (declare (type (simple-array single-float (*)) a c)
           (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (- (aref a i)))))

(defun vectorize-single-scale (a c n)
  (declare (type (simple-array single-float (*)) a c)
           (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (* (aref a i) 1.1f0))))

(defun vectorize-single-add-scale (a b c n)
  (declare (type (simple-array single-float (*)) a b c)
           (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (* (+ (aref a i) (aref b i)) 2.3f0))))

;;; The same loop with no declarations at all: nothing tells the compiler
;;; that these are float arrays, so there is nothing to vectorize, and the
;;; result has to be right anyway.
(defun vectorize-single-add-undeclared (a b c n)
  (dotimes (i n)
    (setf (aref c i) (+ (aref a i) (aref b i)))))

;;;; double-float

(defun vectorize-double-add (a b c n)
  (declare (type (simple-array double-float (*)) a b c)
           (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (+ (aref a i) (aref b i)))))

(defun vectorize-double-sub (a b c n)
  (declare (type (simple-array double-float (*)) a b c)
           (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (- (aref a i) (aref b i)))))

(defun vectorize-double-mul (a b c n)
  (declare (type (simple-array double-float (*)) a b c)
           (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (* (aref a i) (aref b i)))))

(defun vectorize-double-div (a b c n)
  (declare (type (simple-array double-float (*)) a b c)
           (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (/ (aref a i) (aref b i)))))

(defun vectorize-double-negate (a c n)
  (declare (type (simple-array double-float (*)) a c)
           (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (- (aref a i)))))

(defun vectorize-double-scale (a c n)
  (declare (type (simple-array double-float (*)) a c)
           (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (* (aref a i) 1.1d0))))

(defun vectorize-double-add-scale (a b c n)
  (declare (type (simple-array double-float (*)) a b c)
           (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (* (+ (aref a i) (aref b i)) 2.3d0))))

;;; The same loop with no declarations at all: nothing tells the compiler
;;; that these are float arrays, so there is nothing to vectorize, and the
;;; result has to be right anyway.
(defun vectorize-double-add-undeclared (a b c n)
  (dotimes (i n)
    (setf (aref c i) (+ (aref a i) (aref b i)))))

;;;; Correctness: every operation, both element types, every test length.

(test-util:with-test (:name :vectorize-single-add :skipped-on (not :x86-64))
  (vectorize-check-elementwise #'vectorize-single-add #'+
                               'single-float
                               *vectorize-single-values*
                               *vectorize-single-values*
                               'vectorize-single-add))

(test-util:with-test (:name :vectorize-single-sub :skipped-on (not :x86-64))
  (vectorize-check-elementwise #'vectorize-single-sub #'-
                               'single-float
                               *vectorize-single-values*
                               *vectorize-single-values*
                               'vectorize-single-sub))

(test-util:with-test (:name :vectorize-single-mul :skipped-on (not :x86-64))
  (vectorize-check-elementwise #'vectorize-single-mul #'*
                               'single-float
                               *vectorize-single-values*
                               *vectorize-single-values*
                               'vectorize-single-mul))

(test-util:with-test (:name :vectorize-single-div :skipped-on (not :x86-64))
  (vectorize-check-elementwise #'vectorize-single-div #'/
                               'single-float
                               *vectorize-single-values*
                               *vectorize-single-divisors*
                               'vectorize-single-div))

(test-util:with-test (:name :vectorize-single-negate :skipped-on (not :x86-64))
  (vectorize-check-unary #'vectorize-single-negate #'-
                         'single-float
                         *vectorize-single-values*
                         'vectorize-single-negate))

;;; The loops below multiply by a literal; the references multiply by the
;;; matching special variable, whose value is that same literal.
(test-util:with-test (:name :vectorize-single-scale :skipped-on (not :x86-64))
  (vectorize-check-unary #'vectorize-single-scale
                         (lambda (x) (* x *vectorize-single-scale*))
                         'single-float
                         *vectorize-single-values*
                         'vectorize-single-scale))

(test-util:with-test (:name :vectorize-single-add-scale :skipped-on (not :x86-64))
  (vectorize-check-elementwise #'vectorize-single-add-scale
                               (lambda (x y) (* (+ x y) *vectorize-single-add-scale*))
                               'single-float
                               *vectorize-single-values*
                               *vectorize-single-values*
                               'vectorize-single-add-scale))

(test-util:with-test (:name :vectorize-double-add :skipped-on (not :x86-64))
  (vectorize-check-elementwise #'vectorize-double-add #'+
                               'double-float
                               *vectorize-double-values*
                               *vectorize-double-values*
                               'vectorize-double-add))

(test-util:with-test (:name :vectorize-double-sub :skipped-on (not :x86-64))
  (vectorize-check-elementwise #'vectorize-double-sub #'-
                               'double-float
                               *vectorize-double-values*
                               *vectorize-double-values*
                               'vectorize-double-sub))

(test-util:with-test (:name :vectorize-double-mul :skipped-on (not :x86-64))
  (vectorize-check-elementwise #'vectorize-double-mul #'*
                               'double-float
                               *vectorize-double-values*
                               *vectorize-double-values*
                               'vectorize-double-mul))

(test-util:with-test (:name :vectorize-double-div :skipped-on (not :x86-64))
  (vectorize-check-elementwise #'vectorize-double-div #'/
                               'double-float
                               *vectorize-double-values*
                               *vectorize-double-divisors*
                               'vectorize-double-div))

(test-util:with-test (:name :vectorize-double-negate :skipped-on (not :x86-64))
  (vectorize-check-unary #'vectorize-double-negate #'-
                         'double-float
                         *vectorize-double-values*
                         'vectorize-double-negate))

(test-util:with-test (:name :vectorize-double-scale :skipped-on (not :x86-64))
  (vectorize-check-unary #'vectorize-double-scale
                         (lambda (x) (* x *vectorize-double-scale*))
                         'double-float
                         *vectorize-double-values*
                         'vectorize-double-scale))

(test-util:with-test (:name :vectorize-double-add-scale :skipped-on (not :x86-64))
  (vectorize-check-elementwise #'vectorize-double-add-scale
                               (lambda (x y) (* (+ x y) *vectorize-double-add-scale*))
                               'double-float
                               *vectorize-double-values*
                               *vectorize-double-values*
                               'vectorize-double-add-scale))

;;;; Aliasing: C may be the same array as A, and A may be the same as B.

(test-util:with-test (:name :vectorize-single-aliasing :skipped-on (not :x86-64))
  (vectorize-check-aliasing #'vectorize-single-add 'single-float
                            *vectorize-single-values*
                            *vectorize-single-values*
                            'vectorize-single-aliasing))

(test-util:with-test (:name :vectorize-double-aliasing :skipped-on (not :x86-64))
  (vectorize-check-aliasing #'vectorize-double-add 'double-float
                            *vectorize-double-values*
                            *vectorize-double-values*
                            'vectorize-double-aliasing))

;;;; N larger than the arrays: the fallback scalar loop signals an index error
;;;; rather than reading or writing out of bounds.  SPEED alone does not turn
;;;; bounds checks off, because SAFETY is still 1, and that is what makes the error
;;;; observable.

(test-util:with-test (:name :vectorize-single-n-exceeds-length :skipped-on (not :x86-64))
  (loop for (size . n) in '((0 . 1) (1 . 2) (1 . 5) (3 . 4) (4 . 5)
                            (8 . 9) (8 . 17) (1000 . 1001))
        do (let ((a (vectorize-array 'single-float *vectorize-single-values* size))
                 (b (vectorize-array 'single-float *vectorize-single-divisors* size 3))
                 (c (vectorize-sentinel-array 'single-float size)))
             (assert-error (funcall #'vectorize-single-add a b c n)
                           sb-int:invalid-array-index-error))))

(test-util:with-test (:name :vectorize-double-n-exceeds-length :skipped-on (not :x86-64))
  (loop for (size . n) in '((0 . 1) (1 . 2) (1 . 5) (3 . 4) (4 . 5)
                            (8 . 9) (8 . 17) (1000 . 1001))
        do (let ((a (vectorize-array 'double-float *vectorize-double-values* size))
                 (b (vectorize-array 'double-float *vectorize-double-divisors* size 3))
                 (c (vectorize-sentinel-array 'double-float size)))
             (assert-error (funcall #'vectorize-double-add a b c n)
                           sb-int:invalid-array-index-error))))

;;;; The same loop without declarations cannot be vectorized, and must still
;;;; compute the right values.

(test-util:with-test (:name :vectorize-single-no-declarations :skipped-on (not :x86-64))
  (vectorize-check-elementwise #'vectorize-single-add-undeclared #'+
                               'single-float
                               *vectorize-single-values*
                               *vectorize-single-values*
                               'vectorize-single-add-undeclared))

(test-util:with-test (:name :vectorize-double-no-declarations :skipped-on (not :x86-64))
  (vectorize-check-elementwise #'vectorize-double-add-undeclared #'+
                               'double-float
                               *vectorize-double-values*
                               *vectorize-double-values*
                               'vectorize-double-add-undeclared))

;;;; The loop was actually vectorized.  Two functions that differ only in
;;;; whether the array types are declared: the declared one must contain a
;;;; packed SIMD mnemonic, the undeclared one must not, whatever instructions
;;;; the vectorizer settles on in detail.

(test-util:with-test (:name :vectorize-disassembly :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           ;; Mnemonic case varies between disassemblers.
           (string-downcase
            (with-output-to-string (stream)
              (disassemble function :stream stream))))
         (loop-function (element-type declaredp)
           (compile nil
                    `(lambda (a b c n)
                       ,@(when declaredp
                           `((declare (type (simple-array ,element-type (*)) a b c)
                                      (type vectorize-index n))))
                       (declare (optimize speed))
                       (dotimes (i n)
                         (setf (aref c i) (+ (aref a i) (aref b i))))))))
    (dolist (element-type '(single-float double-float))
      (let ((vectorized (disassembly (loop-function element-type t)))
            (scalar (disassembly (loop-function element-type nil))))
        (assert (some (lambda (mnemonic) (search mnemonic vectorized))
                      *vectorize-packed-mnemonics*)
                ()
                "~S loop was not vectorized: no packed mnemonic in~%~A"
                element-type vectorized)
        (dolist (mnemonic '("addps" "addpd"))
          (assert (not (search mnemonic scalar))
                  ()
                  "~S loop without declarations contains ~A~%~A"
                  element-type mnemonic scalar))))))

;;;; Integer element types.
;;;;
;;;; The same elementwise contract as the float tests, over the four integer
;;;; element types the vectorizer handles.  The values are chosen so that no
;;;; result leaves the element type's range: a packed lane wraps around, and the
;;;; scalar path's behaviour with an out-of-range store is outside what this suite
;;;; pins down.

(defparameter *vectorize-u32-values* '(0 1 2 3 7 11 100 255 1000 65535))
(defparameter *vectorize-s32-values* '(0 1 -2 3 -7 11 -100 255 -1000 65535 -65535))
(defparameter *vectorize-u64-values* '(0 1 2 3 1099511627776 4294967296 1000 65535))
(defparameter *vectorize-s64-values* '(0 1 -2 3 -1099511627776 4294967296 -1000 65535))

;;; A value no loop below can produce, in range for all four element types.
(defparameter *vectorize-integer-guard* 12345)

(defun vectorize-integer-array (element-type values count &optional (offset 0))
  (make-array count
              :element-type element-type
              :initial-contents (vectorize-fill-list values count offset)))

(defun vectorize-check-integer-elementwise (function operation element-type
                                            values-a values-b context)
  "Run FUNCTION over every test length and compare each element against
OPERATION applied to the same numbers, which involves no array loop."
  (loop for n in *vectorize-lengths*
        for a = (vectorize-integer-array element-type values-a n)
        for b = (vectorize-integer-array element-type values-b n 3)
        ;; eight spare elements: a packed store that ran past N would land in them
        for c = (make-array (+ n 8) :element-type element-type
                              :initial-element *vectorize-integer-guard*)
        do (funcall function a b c n)
           (dotimes (i n)
             (let ((want (funcall operation (aref a i) (aref b i)))
                   (got (aref c i)))
               (unless (= got want)
                 (error "~@<~A: element ~D of ~D is ~S, expected ~S.~:@>"
                        context i n got want))))
           (loop for i from n below (+ n 8)
                 do (unless (= (aref c i) *vectorize-integer-guard*)
                      (error "~@<~A: element ~D is past the loop bound but was written: ~S.~:@>"
                             context i (aref c i))))))

;;; The loops.  Declared, so they vectorize; the operations are elementwise.

(defun vectorize-u32-add (a b c n)
  (declare (type (simple-array (unsigned-byte 32) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i)))))

(defun vectorize-s32-add (a b c n)
  (declare (type (simple-array (signed-byte 32) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i)))))

(defun vectorize-u64-add (a b c n)
  (declare (type (simple-array (unsigned-byte 64) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i)))))

(defun vectorize-s64-add (a b c n)
  (declare (type (simple-array (signed-byte 64) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i)))))

(defun vectorize-s32-subtract (a b c n)
  (declare (type (simple-array (signed-byte 32) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (- (aref a i) (aref b i)))))

(defun vectorize-s64-subtract (a b c n)
  (declare (type (simple-array (signed-byte 64) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (- (aref a i) (aref b i)))))

(defun vectorize-s32-logand (a b c n)
  (declare (type (simple-array (signed-byte 32) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (logand (aref a i) (aref b i)))))

(defun vectorize-s32-logior (a b c n)
  (declare (type (simple-array (signed-byte 32) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (logior (aref a i) (aref b i)))))

(defun vectorize-u64-logxor (a b c n)
  (declare (type (simple-array (unsigned-byte 64) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (logxor (aref a i) (aref b i)))))

;;; A negative constant reaches the backend as a tagged fixnum, and a lane that
;;; is broadcast from a tagged value has to be recovered arithmetically, or -3
;;; comes out as a huge positive.
(defun vectorize-s32-add-negative-constant (a b c n)
  (declare (type (simple-array (signed-byte 32) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) -3))))

;;; A scalar the loop declares FIXNUM reaches the 64-bit lane splat as an
;;; untagged word, exactly like one declared (SIGNED-BYTE 64): the move into the
;;; vop's untagged storage class is what strips the tag.  Untagging it a second
;;; time inside the vop stores half the value.
(defun vectorize-s64-add-fixnum (a c k n)
  (declare (type (simple-array (signed-byte 64) (*)) a c) (type fixnum k)
           (type vectorize-index n) (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) k))))

(defun vectorize-s64-subtract-scalar (a b c n k)
  (declare (type (simple-array (signed-byte 64) (*)) a b c) (type vectorize-index n)
           (type (signed-byte 64) k) (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (- (aref a i) k))))

;;; A variable broadcast into lanes narrower than a word: the splat vop cannot
;;; tell a tagged operand from an untagged one, and shifted a value that had
;;; already arrived untagged, storing half of it.

(defun vectorize-u8-add-variable (a c k n)
  (declare (type (simple-array (unsigned-byte 8) (*)) a c) (type (unsigned-byte 8) k)
           (type vectorize-index n) (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) k))))

(defun vectorize-u16-add-variable (a c k n)
  (declare (type (simple-array (unsigned-byte 16) (*)) a c) (type (unsigned-byte 16) k)
           (type vectorize-index n) (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) k))))

(defun vectorize-u32-add-variable (a c k n)
  (declare (type (simple-array (unsigned-byte 32) (*)) a c) (type (unsigned-byte 32) k)
           (type vectorize-index n) (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) k))))

(defun vectorize-s32-add-variable (a c k n)
  (declare (type (simple-array (signed-byte 32) (*)) a c) (type (signed-byte 32) k)
           (type vectorize-index n) (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) k))))

(test-util:with-test (:name :vectorize-integer-add :skipped-on (not :x86-64))
  (vectorize-check-integer-elementwise #'vectorize-u32-add #'+ '(unsigned-byte 32)
                                       *vectorize-u32-values* *vectorize-u32-values*
                                       'vectorize-u32-add)
  (vectorize-check-integer-elementwise #'vectorize-s32-add #'+ '(signed-byte 32)
                                       *vectorize-s32-values* *vectorize-s32-values*
                                       'vectorize-s32-add)
  (vectorize-check-integer-elementwise #'vectorize-u64-add #'+ '(unsigned-byte 64)
                                       *vectorize-u64-values* *vectorize-u64-values*
                                       'vectorize-u64-add)
  (vectorize-check-integer-elementwise #'vectorize-s64-add #'+ '(signed-byte 64)
                                       *vectorize-s64-values* *vectorize-s64-values*
                                       'vectorize-s64-add))

(test-util:with-test (:name :vectorize-integer-subtract :skipped-on (not :x86-64))
  ;; the minuend pattern is the larger one, so nothing wraps
  (vectorize-check-integer-elementwise #'vectorize-s32-subtract #'-
                                       '(signed-byte 32)
                                       '(65535 1000 255 100 11 7 3 2 1 0)
                                       *vectorize-s32-values* 'vectorize-s32-subtract)
  (vectorize-check-integer-elementwise #'vectorize-s64-subtract #'-
                                       '(signed-byte 64)
                                       '(4294967296 1099511627776 65535 1000 1 0)
                                       *vectorize-s64-values* 'vectorize-s64-subtract))

(test-util:with-test (:name :vectorize-integer-bitwise :skipped-on (not :x86-64))
  (vectorize-check-integer-elementwise #'vectorize-s32-logand #'logand '(signed-byte 32)
                                       *vectorize-s32-values* *vectorize-s32-values*
                                       'vectorize-s32-logand)
  (vectorize-check-integer-elementwise #'vectorize-s32-logior #'logior '(signed-byte 32)
                                       '(0 1 2 4 8 16 32 64) '(0 3 5 9 17 33 65 129)
                                       'vectorize-s32-logior)
  (vectorize-check-integer-elementwise #'vectorize-u64-logxor #'logxor '(unsigned-byte 64)
                                       *vectorize-u64-values* '(1 2 3 4 5 6 7 8)
                                       'vectorize-u64-logxor))

(test-util:with-test (:name :vectorize-integer-negative-constant :skipped-on (not :x86-64))
  (vectorize-check-integer-elementwise #'vectorize-s32-add-negative-constant
                                       (lambda (x ignorable) (declare (ignore ignorable)) (+ x -3))
                                       '(signed-byte 32)
                                       *vectorize-s32-values* *vectorize-s32-values*
                                       'vectorize-s32-add-negative-constant))

(test-util:with-test (:name :vectorize-integer-scalar-variable :skipped-on (not :x86-64))
  (loop for n in *vectorize-lengths*
        for a = (vectorize-integer-array '(signed-byte 64) *vectorize-s64-values* n)
        for c = (make-array (+ n 4) :element-type '(signed-byte 64)
                              :initial-element *vectorize-integer-guard*)
        for k = -1099511627776
        do (vectorize-s64-subtract-scalar a a c n k)
           (dotimes (i n)
             (let ((want (- (aref a i) k)))
               (unless (= (aref c i) want)
                 (error "~@<element ~D of ~D is ~S, expected ~S.~:@>"
                        i n (aref c i) want)))))
  ;; a scalar operand is broadcast, so the loop must be packed
  (let ((text (string-downcase (with-output-to-string (stream)
                                 (disassemble #'vectorize-s64-subtract-scalar
                                              :stream stream)))))
    (assert (search "psubq" text) () "a loop with a scalar variable was not packed:~%~A" text)))

;;; The values are large enough that storing half of one cannot pass, for the
;;; unsigned types and for a negative value of a signed one.
(test-util:with-test (:name :vectorize-broadcast-variable :skipped-on (not :x86-64))
  (let ((cases '((vectorize-u8-add-variable (unsigned-byte 8) 200)
                 (vectorize-u16-add-variable (unsigned-byte 16) 60000)
                 (vectorize-u32-add-variable (unsigned-byte 32) 123456789)
                 (vectorize-s32-add-variable (signed-byte 32) -123456789))))
    (loop for (name element-type value) in cases
          do (loop for n in *vectorize-lengths*
                   for a = (vectorize-integer-array element-type '(0) n)
                   ;; a stored value is never 0, so a write past N shows up here
                   for c = (make-array (+ n 8) :element-type element-type
                                         :initial-element 0)
                   do (funcall name a c value n)
                      (dotimes (i n)
                        (unless (= (aref c i) value)
                          (error "~@<~A: element ~D of ~D is ~S, expected a broadcast of ~S.~:@>"
                                 name i n (aref c i) value)))
                      (loop for i from n below (+ n 8)
                            do (unless (zerop (aref c i))
                                 (error "~@<~A: element ~D is past the loop bound but was written: ~S.~:@>"
                                        name i (aref c i))))))
    ;; each loop broadcasts a variable, so each must have a splat in it
    (dolist (case cases)
      (let ((text (string-downcase (with-output-to-string (stream)
                                     (disassemble (first case) :stream stream)))))
        ;; the 128-bit splats shuffle a single lane into place; the 256-bit
        ;; ones broadcast it with one instruction
        (assert (some (lambda (mnemonic) (search mnemonic text))
                      '("pshufd" "pbroadcast" "punpcklqdq"))
                () "~A was not vectorized:~%~A" (first case) text)))))

(test-util:with-test (:name :vectorize-broadcast-fixnum-scalar :skipped-on (not :x86-64))
  (loop for k in '(-1 -7 12345 -1099511627776 1099511627776)
        do (loop for n in *vectorize-lengths*
                 for a = (vectorize-integer-array '(signed-byte 64)
                                                  *vectorize-s64-values* n)
                 for c = (make-array (+ n 8) :element-type '(signed-byte 64)
                                       :initial-element 0)
                 do (vectorize-s64-add-fixnum a c k n)
                    (dotimes (i n)
                      (let ((want (logand (+ (aref a i) k) (1- (ash 1 64)))))
                        (unless (= (aref c i) (if (logbitp 63 want) (- want (ash 1 64)) want))
                          (error "~@<broadcast of ~D: element ~D of ~D is ~S, expected ~S.~:@>"
                                 k i n (aref c i) (+ (aref a i) k)))))
                    (loop for i from n below (+ n 8)
                          do (unless (zerop (aref c i))
                               (error "~@<broadcast of ~D: element ~D is past the loop ~
                                       bound but was written: ~S.~:@>"
                                      k i (aref c i))))))
  ;; the scalar is broadcast, so the loop must be packed
  (let ((text (string-downcase (with-output-to-string (stream)
                                 (disassemble #'vectorize-s64-add-fixnum :stream stream)))))
    (assert (search "paddq" text) () "a fixnum scalar was not packed:~%~A" text)))

;;; A constant too wide for a fixnum has no splat: the loop must be left scalar,
;;; not rejected by the backend, and must compute what it always computed.
(test-util:with-test (:name :vectorize-integer-wide-constant :skipped-on (not :x86-64))
  (flet ((loop-with-wide-constant (a c n)
           (declare (type (simple-array (unsigned-byte 64) (*)) a c)
                    (type vectorize-index n) (optimize speed))
           (dotimes (i n) (setf (aref c i) (+ (aref a i) 18446744073709551615)))))
    (let* ((n 17)
           (a (make-array n :element-type '(unsigned-byte 64) :initial-element 0))
           (c (make-array n :element-type '(unsigned-byte 64) :initial-element 0)))
      (loop-with-wide-constant a c n)
      (dotimes (i n)
        (assert (= (aref c i) 18446744073709551615))))))

(test-util:with-test (:name :vectorize-integer-disassembly :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    (dolist (case (list (cons #'vectorize-s32-add '("paddd" "psubd" "movdqu"))
                        (cons #'vectorize-u64-add '("paddq" "movdqu"))
                        (cons #'vectorize-s32-logand '("pand" "movdqu"))))
      (let ((text (disassembly (car case))))
        (assert (some (lambda (mnemonic) (search mnemonic text)) (cdr case))
                ()
                "~S was not vectorized:~%~A" (car case) text)))))

;;;; LOOP forms.  The vectorizer recognizes
;;;;
;;;;   (loop for i [of-type type] [from 0] below|upto n do <elementwise store>)
;;;;
;;;; which must produce exactly what the DOTIMES version produces; loop shapes
;;;; it does not recognize must still compute the right thing, scalar.

(defun vectorize-single-add-dotimes (a b c n)
  (declare (type (simple-array single-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))
  c)

(defun vectorize-single-add-loop-below (a b c n)
  (declare (type (simple-array single-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (loop for i below n do (setf (aref c i) (+ (aref a i) (aref b i))))
  c)

(defun vectorize-single-add-loop-from-0 (a b c n)
  (declare (type (simple-array single-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (loop for i from 0 below n do (setf (aref c i) (+ (aref a i) (aref b i))))
  c)

(defun vectorize-single-add-loop-of-type (a b c n)
  (declare (type (simple-array single-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (loop for i of-type fixnum below n do (setf (aref c i) (+ (aref a i) (aref b i))))
  c)

(defun vectorize-single-add-loop-upto (a b c n)
  (declare (type (simple-array single-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (loop for i upto (1- n) do (setf (aref c i) (+ (aref a i) (aref b i))))
  c)

(test-util:with-test (:name :vectorize-loop-forms :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    (loop for n in *vectorize-lengths*
          for a = (vectorize-array 'single-float *vectorize-single-values* n)
          for b = (vectorize-array 'single-float *vectorize-single-values* n 3)
          for reference = (vectorize-array 'single-float *vectorize-single-values* n)
          do (funcall #'vectorize-single-add-dotimes a b reference n)
             (dolist (function (list #'vectorize-single-add-loop-below
                                     #'vectorize-single-add-loop-from-0
                                     #'vectorize-single-add-loop-of-type
                                     #'vectorize-single-add-loop-upto))
               (let ((c (vectorize-sentinel-array 'single-float n)))
                 (funcall function a b c n)
                 (dotimes (i n)
                   (unless (eql (aref c i) (aref reference i))
                     (error "~@<~S, length ~D: element ~D is ~S, expected ~S.~:@>"
                            function n i (aref c i) (aref reference i)))))))
    ;; and each of them is the packed loop
    (dolist (function (list #'vectorize-single-add-loop-below
                            #'vectorize-single-add-loop-from-0
                            #'vectorize-single-add-loop-of-type
                            #'vectorize-single-add-loop-upto))
      (let ((text (disassembly function)))
        (assert (some (lambda (mnemonic) (search mnemonic text))
                      *vectorize-packed-mnemonics*)
                () "~S was not vectorized:~%~A" function text)))))

;;; A loop body that writes more than one array.  The matcher that recognizes
;;; these is the portable one, so an arm64 build runs the same shapes
;;; (tests/auto-vectorize-arm64.pure.lisp); the two files have to agree on what
;;; the shapes mean, and this is the one that runs on the backend where the pass
;;; can be exercised.

(defun vectorize-multi-two-destinations (re im n)
  (declare (type (simple-array single-float (*)) re im) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref re i) (* 2.0 (aref re i)))
    (setf (aref im i) (+ 1.0 (aref im i))))
  (values))

;;; the second store reads the element the first one wrote, so the packed body
;;; has to keep the two stores in that order, as the scalar loop does
(defun vectorize-multi-in-place (a n)
  (declare (type (simple-array single-float (*)) a) (type vectorize-index n)
           (optimize speed))
  (let ((half (ash n -1)))
    (declare (type vectorize-index half))
    (dotimes (i half)
      (setf (aref a (+ i half)) (+ (aref a i) 1.0))
      (setf (aref a i) (* 2.0 (aref a (+ i half)))))
    a))

;;; a radix-2 butterfly: an index binding, intermediates that two stores share,
;;; and four stores across two arrays
(defun vectorize-multi-butterfly (re im wr wi n)
  (declare (type (simple-array single-float (*)) re im wr wi)
           (type vectorize-index n) (optimize speed))
  (let ((half (ash n -1)))
    (declare (type vectorize-index half))
    (dotimes (k half)
      (let* ((j (+ k half))
             (tr (- (* (aref re j) (aref wr k)) (* (aref im j) (aref wi k))))
             (ti (+ (* (aref re j) (aref wi k)) (* (aref im j) (aref wr k)))))
        (declare (type single-float tr ti))
        (setf (aref re j) (- (aref re k) tr))
        (setf (aref im j) (- (aref im k) ti))
        (setf (aref re k) (+ (aref re k) tr))
        (setf (aref im k) (+ (aref im k) ti))))
    (values)))

;;; one store, ahead of what the loop reads: no permutation argument makes this
;;; safe, so it stays scalar
(defun vectorize-multi-write-ahead (a n)
  (declare (type (simple-array single-float (*)) a) (type vectorize-index n)
           (optimize speed))
  (let ((half (ash n -1)))
    (declare (type vectorize-index half))
    (dotimes (i half)
      (setf (aref a (+ i half)) (+ (aref a i) 1.0)))
    a))

(test-util:with-test (:name :vectorize-multiple-stores :skipped-on (not :x86-64))
  ;; LABELS, not FLET: PACKED calls DISASSEMBLY
  (labels ((disassembly (function)
             (string-downcase (with-output-to-string (stream)
                                (disassemble function :stream stream))))
           (packed (function)
           (some (lambda (mnemonic) (search mnemonic (disassembly function)))
                 *vectorize-packed-mnemonics*)))
    ;; two destinations, neither one read
    (loop for n in *vectorize-lengths*
          for re = (vectorize-array 'single-float '(1.5 -2.25 0.5 3.75 -1.25) n)
          for im = (vectorize-array 'single-float '(7.5 -8.25 2.5 -3.0 0.25) n 2)
          for want-re = (map 'vector (lambda (x) (* 2.0 x)) re)
          for want-im = (map 'vector (lambda (x) (+ 1.0 x)) im)
          do (vectorize-multi-two-destinations re im n)
             (dotimes (i n)
               (unless (vectorize-same-value-p (aref re i) (aref want-re i))
                 (error "~@<two destinations: re ~D of ~D is ~S, expected ~S.~:@>"
                        i n (aref re i) (aref want-re i)))
               (unless (vectorize-same-value-p (aref im i) (aref want-im i))
                 (error "~@<two destinations: im ~D of ~D is ~S, expected ~S.~:@>"
                        i n (aref im i) (aref want-im i)))))
    (assert (packed #'vectorize-multi-two-destinations) ()
            "a loop with two destinations was not vectorized:~%~A"
            (disassembly #'vectorize-multi-two-destinations))
    ;; an in-place update where each store reads what the one before it wrote
    (loop for n in '(0 2 4 5 8 9 16 17 100)
          for half = (ash n -1)
          for a = (vectorize-array 'single-float '(1.5 -2.25 0.5 3.75 -1.25) n)
          do (let ((want (copy-seq a)))
               (dotimes (i half)
                 (let ((first (aref a i)))
                   (setf (aref want (+ i half)) (+ first 1.0)
                         (aref want i) (* 2.0 (+ first 1.0)))))
               (vectorize-multi-in-place a n)
               (dotimes (i n)
                 (unless (vectorize-same-value-p (aref a i) (aref want i))
                   (error "~@<in-place update: element ~D of ~D is ~S, expected ~S.~:@>"
                          i n (aref a i) (aref want i))))))
    (assert (packed #'vectorize-multi-in-place) ()
            "an in-place update was not vectorized:~%~A"
            (disassembly #'vectorize-multi-in-place))
    ;; the butterfly
    (loop for n in '(0 2 4 8 9 16 17 64 100)
          for half = (ash n -1)
          for re = (vectorize-array 'single-float '(1.5 -2.25 0.5 3.75 -1.25) n)
          for im = (vectorize-array 'single-float '(7.5 -8.25 2.5 -3.0 0.25) n 3)
          for wr = (vectorize-array 'single-float '(0.5 1.5 -0.75 2.25 1.0) half)
          for wi = (vectorize-array 'single-float '(1.25 -0.5 0.75 -1.5 2.0) half 1)
          do (let ((want-re (copy-seq re))
                   (want-im (copy-seq im)))
               (dotimes (k half)
                 (let* ((j (+ k half))
                        (tr (- (* (aref re j) (aref wr k)) (* (aref im j) (aref wi k))))
                        (ti (+ (* (aref re j) (aref wi k)) (* (aref im j) (aref wr k)))))
                   (setf (aref want-re j) (- (aref re k) tr)
                         (aref want-im j) (- (aref im k) ti)
                         (aref want-re k) (+ (aref re k) tr)
                         (aref want-im k) (+ (aref im k) ti))))
               (vectorize-multi-butterfly re im wr wi n)
               (dotimes (i n)
                 (unless (vectorize-same-value-p (aref re i) (aref want-re i))
                   (error "~@<butterfly: re ~D of ~D is ~S, expected ~S.~:@>"
                          i n (aref re i) (aref want-re i)))
                 (unless (vectorize-same-value-p (aref im i) (aref want-im i))
                   (error "~@<butterfly: im ~D of ~D is ~S, expected ~S.~:@>"
                          i n (aref im i) (aref want-im i))))))
    (assert (packed #'vectorize-multi-butterfly) ()
            "the butterfly was not vectorized:~%~A"
            (disassembly #'vectorize-multi-butterfly))
    ;; and the shape that cannot be packed
    (assert (not (packed #'vectorize-multi-write-ahead)) ()
            "a write ahead of every read was vectorized:~%~A"
            (disassembly #'vectorize-multi-write-ahead))))

;;; Loops that must be left alone, and must still be right.

(defun vectorize-single-add-loop-from-1 (a b c n)
  (declare (type (simple-array single-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (loop for i from 1 below n do (setf (aref c i) (+ (aref a i) (aref b i))))
  c)

(defun vectorize-single-add-loop-with-finally (a b c n)
  (declare (type (simple-array single-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (loop for i below n do (setf (aref c i) (+ (aref a i) (aref b i)))
        finally (return c)))

(defun vectorize-single-accumulate (a c n)
  "INCF into the destination, which the vectorizer does take; kept here because the
correctness check below still wants the values, and :vectorize-incf-store covers the
instruction side."
  (declare (type (simple-array single-float (*)) a c) (type vectorize-index n)
           (optimize speed))
  (loop for i below n do (incf (aref c i) (aref a i)))
  c)

(test-util:with-test (:name :vectorize-loop-not-recognized :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    (loop for n in *vectorize-lengths*
          for a = (vectorize-array 'single-float *vectorize-single-values* n)
          for b = (vectorize-array 'single-float *vectorize-single-values* n 3)
          do ;; a nonzero start leaves element 0 alone
             (let ((c (vectorize-sentinel-array 'single-float n)))
               (vectorize-single-add-loop-from-1 a b c n)
               (when (plusp n)
                 (unless (eql (aref c 0) (vectorize-sentinel 'single-float))
                   (error "~@<a loop from 1 wrote element 0.~:@>")))
               (loop for i from 1 below n
                     do (unless (eql (aref c i) (+ (aref a i) (aref b i)))
                          (error "~@<a loop from 1 is wrong at element ~D of ~D.~:@>" i n))))
             ;; a FINALLY clause is not one of the shapes recognized
             (let ((c (vectorize-sentinel-array 'single-float n)))
               (vectorize-single-add-loop-with-finally a b c n)
               (dotimes (i n)
                 (unless (eql (aref c i) (+ (aref a i) (aref b i)))
                   (error "~@<a loop with FINALLY is wrong at element ~D of ~D.~:@>" i n))))
             ;; accumulating into the destination is not elementwise assignment
             (let ((c (make-array n :element-type 'single-float :initial-element 1.0f0)))
               (vectorize-single-accumulate a c n)
               (dotimes (i n)
                 (unless (eql (aref c i) (+ 1.0f0 (aref a i)))
                   (error "~@<an accumulating loop is wrong at element ~D of ~D.~:@>" i n)))))
    (dolist (function (list #'vectorize-single-add-loop-from-1
                            #'vectorize-single-add-loop-with-finally))
      (let ((text (disassembly function)))
        (assert (notany (lambda (mnemonic) (search mnemonic text))
                        '("addps" "addpd"))
                () "~S was vectorized but should not have been:~%~A" function text)))))

;;;; The compiler says what it did.  A vectorized loop is reported by a note,
;;;; and a loop that is not vectorized is silent about it.

(defun vectorize-note-output (form)
  (with-output-to-string (stream)
    (let ((*error-output* stream))
      (compile nil form))))

(test-util:with-test (:name :vectorize-note :skipped-on (not :x86-64))
  (let ((output (vectorize-note-output
                 '(lambda (a b c n)
                    (declare (type (simple-array single-float (*)) a b c)
                             (type vectorize-index n) (optimize speed))
                    (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))))))
    (assert (search "vectorized loop over" output) () "No vectorization note in:~%~A" output))
  (let ((output (vectorize-note-output
                 '(lambda (a n)
                    (declare (type vector n) (optimize speed))
                    (dotimes (i n) (setf (aref a i) (aref a i)))))))
    (assert (not (search "vectorized loop over" output))
            () "A note was emitted for a loop that was not vectorized:~%~A" output)))

;;;; Byte lanes: sixteen at a time, and only at SAFETY 0.
;;;;
;;;; A packed byte store truncates where the scalar store signals a type error,
;;;; so the two agree only when the code has turned the check off.  These loops
;;;; therefore declare (safety 0); the last test checks that the same loop at the
;;;; default safety is left scalar instead.

(defun vectorize-u8-subtract (a b c n)
  (declare (type (simple-array (unsigned-byte 8) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (- (aref a i) (aref b i)))))

(defun vectorize-u8-add (a b c n)
  (declare (type (simple-array (unsigned-byte 8) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i)))))

(defun vectorize-u8-logxor (a b c n)
  (declare (type (simple-array (unsigned-byte 8) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (logxor (aref a i) (aref b i)))))

(defun vectorize-s8-subtract (a b c n)
  (declare (type (simple-array (signed-byte 8) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (- (aref a i) (aref b i)))))

;;; The same loop, checked: it must be left scalar.
(defun vectorize-u8-subtract-checked (a b c n)
  (declare (type (simple-array (unsigned-byte 8) (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (- (aref a i) (aref b i)))))

(defun vectorize-byte-guard (element-type)
  "A byte no loop below produces, in range for both byte element types."
  (if (equal element-type '(unsigned-byte 8)) 222 -34))

(defun vectorize-check-byte-elementwise (function operation element-type
                                         values-a values-b context)
  "Compare FUNCTION against OPERATION wrapped to the element type, over every
test length, and check that nothing was written past N."
  (loop with guard = (vectorize-byte-guard element-type)
        for n in *vectorize-lengths*
        for a = (vectorize-integer-array element-type values-a n)
        for b = (vectorize-integer-array element-type values-b n 3)
        for c = (make-array (+ n 16) :element-type element-type
                              :initial-element guard)
        do (funcall function a b c n)
           (dotimes (i n)
             (let ((want (logand (funcall operation (aref a i) (aref b i)) #xFF))
                   (got (aref c i)))
               (unless (= (logand got #xFF) want)
                 (error "~@<~A: element ~D of ~D is ~S, expected ~S.~:@>"
                        context i n got want))))
           (loop for i from n below (+ n 16)
                 do (unless (= (aref c i) guard)
                      (error "~@<~A: element ~D is past the loop bound but was written.~:@>"
                             context i)))))

(defparameter *vectorize-u8-values* '(0 1 2 3 7 11 100 255 128 64))
(defparameter *vectorize-s8-values* '(0 1 -2 3 -7 11 -100 127 -128 64))

(test-util:with-test (:name :vectorize-byte-subtract :skipped-on (not :x86-64))
  (vectorize-check-byte-elementwise #'vectorize-u8-subtract #'- '(unsigned-byte 8)
                                    *vectorize-u8-values* *vectorize-u8-values*
                                    'vectorize-u8-subtract))

(test-util:with-test (:name :vectorize-byte-add :skipped-on (not :x86-64))
  (vectorize-check-byte-elementwise #'vectorize-u8-add #'+ '(unsigned-byte 8)
                                    *vectorize-u8-values* *vectorize-u8-values*
                                    'vectorize-u8-add))

(test-util:with-test (:name :vectorize-byte-bitwise :skipped-on (not :x86-64))
  (vectorize-check-byte-elementwise #'vectorize-u8-logxor #'logxor '(unsigned-byte 8)
                                    *vectorize-u8-values* *vectorize-u8-values*
                                    'vectorize-u8-logxor))

(test-util:with-test (:name :vectorize-signed-byte-subtract :skipped-on (not :x86-64))
  (vectorize-check-byte-elementwise #'vectorize-s8-subtract #'- '(signed-byte 8)
                                    *vectorize-s8-values* *vectorize-s8-values*
                                    'vectorize-s8-subtract))

(test-util:with-test (:name :vectorize-byte-disassembly :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    (let ((text (disassembly #'vectorize-u8-subtract)))
      (assert (search "psubb" text) () "u8 subtract is not packed:~%~A" text)
      (assert (search "movdqu" text) () "u8 subtract does not move 16 bytes:~%~A" text))
    (assert (search "paddb" (disassembly #'vectorize-u8-add))
            () "u8 add is not packed")
    (assert (search "pxor" (disassembly #'vectorize-u8-logxor))
            () "u8 logxor is not packed")))

;;; At the default safety the scalar loop may signal, so the vectorizer must not
;;; take it; the values it computes are still right when nothing signals.
(test-util:with-test (:name :vectorize-byte-needs-safety-0 :skipped-on (not :x86-64))
  (let ((text (string-downcase (with-output-to-string (stream)
                                 (disassemble #'vectorize-u8-subtract-checked
                                              :stream stream)))))
    (assert (notany (lambda (mnemonic) (search mnemonic text))
                    '("psubb" "pxor" "paddb"))
            () "a checked u8 loop was vectorized:~%~A" text))
  (loop for n in *vectorize-lengths*
        for a = (vectorize-integer-array '(unsigned-byte 8) '(200 100 55 240) n)
        for b = (vectorize-integer-array '(unsigned-byte 8) '(0 1 2 3) n 3)
        for c = (make-array n :element-type '(unsigned-byte 8) :initial-element 0)
        do (vectorize-u8-subtract-checked a b c n)
           (dotimes (i n)
             (assert (= (aref c i) (- (aref a i) (aref b i)))))))

(defun vectorize-impure-count (a b c n count-function)
  (declare (type (simple-array single-float (*)) a b c) (optimize speed))
  (declare (ignore n))
  (dotimes (i (the fixnum (funcall count-function)))
    (setf (aref c i) (- (aref a i) (aref b i))))
  c)

;;;; Constant offsets in the index, which is what a PNG row filter needs.
;;;;
;;;;   filtered[i] = raw[i] - raw[i - bpp]        (Sub)
;;;;   filtered[i] = raw[i] - prior[i]            (Up)
;;;;
;;;; Sub, over the whole buffer, reads and writes at a constant offset from the
;;;; loop variable; Up is the two-array elementwise shape.  Both are byte arrays
;;;; in image code, so they declare (safety 0) as image code does.

(defun vectorize-offset-subtract (a b c n offset)
  (declare (type (simple-array single-float (*)) a b c) (type vectorize-index n offset)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c (+ i offset)) (- (aref a (+ i offset)) (aref b i))))
  c)

(defun vectorize-png-sub-filter (raw filtered n)
  "PNG filter type 1 (Sub) for three-byte pixels over a whole buffer.  The
element offset has to be a literal for the vectorizer to see it, which is what
a writer with a fixed color type ends up with."
  (declare (type (simple-array (unsigned-byte 8) (*)) raw filtered)
           (type vectorize-index n) (optimize speed (safety 0)))
  (dotimes (i (min 3 n))
    (setf (aref filtered i) (aref raw i)))
  (dotimes (i (- n 3))
    (setf (aref filtered (+ i 3))
          (- (aref raw (+ i 3)) (aref raw i))))
  filtered)

(defun vectorize-png-up-filter (raw prior filtered n)
  "PNG filter type 2 (Up): two arrays, elementwise."
  (declare (type (simple-array (unsigned-byte 8) (*)) raw prior filtered)
           (type vectorize-index n) (optimize speed (safety 0)))
  (dotimes (i n)
    (setf (aref filtered i) (- (aref raw i) (aref prior i))))
  filtered)

(test-util:with-test (:name :vectorize-offset-subtract :skipped-on (not :x86-64))
  (loop for n in *vectorize-lengths*
        for offset = 3
        for a = (vectorize-array 'single-float *vectorize-single-values* (+ n offset))
        for b = (vectorize-array 'single-float *vectorize-single-values* n 3)
        for c = (vectorize-sentinel-array 'single-float (+ n offset))
        do (vectorize-offset-subtract a b c n offset)
           (dotimes (i n)
             (let ((want (- (aref a (+ i offset)) (aref b i)))
                   (got (aref c (+ i offset))))
               (unless (vectorize-same-value-p got want)
                 (error "~@<offset loop: element ~D of ~D is ~S, expected ~S.~:@>"
                        i n got want))))
           ;; the elements before the offset are untouched
           (dotimes (i offset)
             (unless (eql (aref c i) (vectorize-sentinel 'single-float))
               (error "~@<offset loop wrote element ~D, before its offset.~:@>" i)))))

(test-util:with-test (:name :vectorize-png-sub-filter :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    (loop for n in '(0 1 3 4 17 31 64 65 1000)
          for bpp = 3
          for raw = (let ((v (make-array n :element-type '(unsigned-byte 8))))
                      (dotimes (i n v) (setf (aref v i) (mod (+ (* i 7) (* i i)) 256))))
          for a = (make-array n :element-type '(unsigned-byte 8) :initial-element 200)
          for b = (make-array n :element-type '(unsigned-byte 8) :initial-element 200)
          do (vectorize-png-sub-filter raw a n)
             (vectorize-png-up-filter raw raw b n)
             (dotimes (i n)
               (let ((sub (if (< i bpp) (aref raw i)
                              (logand (- (aref raw i) (aref raw (- i bpp))) #xFF))))
                 (unless (= (aref a i) sub)
                   (error "~@<Sub filter, length ~D: element ~D is ~S, expected ~S.~:@>"
                          n i (aref a i) sub))))
             (dotimes (i n)
               (unless (= (aref b i) 0)
                 (error "~@<Up filter of a buffer with itself, length ~D: element ~D is ~S.~:@>"
                        n i (aref b i)))))
    ;; both filters are the packed loop
    (let ((text (disassembly #'vectorize-png-sub-filter)))
      (assert (search "psubb" text) () "Sub filter is not packed:~%~A" text))
    (let ((text (disassembly #'vectorize-png-up-filter)))
      (assert (search "psubb" text) () "Up filter is not packed:~%~A" text))))

;;; A bound that could have side effects must stay with the scalar loop, and the
;;; loop must still compute correctly.
(test-util:with-test (:name :vectorize-impure-bound :skipped-on (not :x86-64))
  (let* ((calls 0)
         (function (lambda (a b c n) (vectorize-impure-count a b c n (lambda () (incf calls) n))))
         (a (vectorize-array 'single-float '(1.0f0 2.0f0 3.0f0) 8))
         (b (vectorize-array 'single-float '(0.5f0 1.0f0) 8 1))
         (c (make-array 8 :element-type 'single-float :initial-element 0.0f0)))
    (funcall function a b c 4)
    (dotimes (i 4)
      (assert (= (aref c i) (- (aref a i) (aref b i)))))))

;;;; A local (MAKE-ARRAY ... :ELEMENT-TYPE 'TYPE) is a declaration the source
;;;; already wrote, so the binding is declared with it.  An adjustable or displaced
;;;; array is not a simple array, so it proves nothing and must stay out.

(defun vectorize-made-locals (n)
  (declare (optimize speed))
  (let ((a (make-array n :element-type 'single-float :initial-element 1.5))
        (b (make-array n :element-type 'single-float :initial-element 2.25))
        (c (make-array n :element-type 'single-float)))
    (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))
    c))

(defun vectorize-made-locals-adjustable (n)
  (declare (optimize speed))
  (let ((a (make-array n :element-type 'single-float :adjustable t :initial-element 1.5))
        (b (make-array n :element-type 'single-float :adjustable t :initial-element 2.25))
        (c (make-array n :element-type 'single-float :adjustable t)))
    (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))
    c))

(test-util:with-test (:name :vectorize-made-locals :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    ;; the initializer proves (SIMPLE-ARRAY SINGLE-FLOAT (*)), so the binding is
    ;; declared with it and the loop is the packed one
    (let ((text (disassembly #'vectorize-made-locals)))
      (assert (search "addps" text)
              () "a loop over local MAKE-ARRAY results was not vectorized:~%~A" text))
    ;; an adjustable array is not simple, so nothing is proven and the loop stays
    ;; scalar, with a note that says why
    (let ((text (disassembly #'vectorize-made-locals-adjustable)))
      (assert (notany (lambda (mnemonic) (search mnemonic text)) '("addps" "addpd"))
              () "a loop over adjustable arrays was vectorized:~%~A" text))
    (let ((notes (vectorize-notes-for
                  '(lambda (n)
                     (declare (type fixnum n) (optimize speed))
                     (let ((a (make-array n :element-type 'single-float :adjustable t))
                           (b (make-array n :element-type 'single-float :adjustable t))
                           (c (make-array n :element-type 'single-float :adjustable t)))
                       (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))
                       c)))))
      (assert (vectorize-note-mentions notes "not known to be an array")
              () "no explanation for the adjustable local: ~s" notes)))
  (dolist (n '(0 1 2 3 4 7 16 17))
    (let ((result (vectorize-made-locals n)))
      (dotimes (i n)
        (assert (eql (aref result i) 3.75f0))))))

;;; VALUE wrapped into BITS and read back as a signed integer: what a store into a
;;; BITS-wide element type does.
(defun vectorize-sign-extend (value bits)
  (let ((wrapped (ldb (byte bits 0) value)))
    (if (logbitp (1- bits) wrapped) (- wrapped (ash 1 bits)) wrapped)))

;;;; Unary minus.  For floats this is a sign flip, which is not the same as
;;;; subtracting from zero: (- 0.0) is -0.0, and 0.0 - 0.0 is 0.0.

(defun vectorize-negate-single (a c n)
  (declare (type (simple-array single-float (*)) a c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (- (aref a i))))
  c)

(defun vectorize-negate-double (a c n)
  (declare (type (simple-array double-float (*)) a c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (- (aref a i))))
  c)

(defun vectorize-negate-integer (a c n)
  (declare (type (simple-array (signed-byte 32) (*)) a c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (- (aref a i))))
  c)

(test-util:with-test (:name :vectorize-unary-minus :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    ;; values include both zeroes: the sign of a zero must survive
    (loop for n in *vectorize-lengths*
          for a = (vectorize-array 'single-float '(0.0f0 -0.0f0 1.5f0 -2.5f0) n)
          for c = (make-array n :element-type 'single-float)
          do (vectorize-negate-single a c n)
             (dotimes (i n)
               (let ((want (- (aref a i))) (got (aref c i)))
                 (unless (eql got want)
                   (error "~@<negate: element ~D of ~D is ~S, expected ~S.~:@>"
                          i n got want)))))
    (loop for n in *vectorize-lengths*
          for a = (vectorize-array 'double-float '(0d0 -0d0 1.5d0 -2.5d0) n)
          for c = (make-array n :element-type 'double-float)
          do (vectorize-negate-double a c n)
             (dotimes (i n)
               (unless (eql (aref c i) (- (aref a i)))
                 (error "~@<double negate wrong at element ~D of ~D.~:@>" i n))))
    (loop for n in *vectorize-lengths*
          for a = (vectorize-integer-array '(signed-byte 32) '(0 1 -2 7 -100) n)
          for c = (make-array n :element-type '(signed-byte 32) :initial-element 0)
          do (vectorize-negate-integer a c n)
             (dotimes (i n)
               (unless (= (aref c i) (- (aref a i)))
                 (error "~@<integer negate wrong at element ~D of ~D.~:@>" i n))))
    (assert (search "xorps" (disassembly #'vectorize-negate-single))
            () "single-float negation is not a packed sign flip")
    (assert (search "xorpd" (disassembly #'vectorize-negate-double))
            () "double-float negation is not a packed sign flip")))

;;;; Sixteen-bit lanes, eight at a time.

(defun vectorize-u16-subtract (a b c n)
  (declare (type (simple-array (unsigned-byte 16) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (- (aref a i) (aref b i)))))

(defun vectorize-s16-add (a b c n)
  (declare (type (simple-array (signed-byte 16) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i)))))

(test-util:with-test (:name :vectorize-16-bit-lanes :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    (loop for n in *vectorize-lengths*
          for a = (vectorize-integer-array '(unsigned-byte 16) '(0 1 2 300 65535 4000) n)
          for b = (vectorize-integer-array '(unsigned-byte 16) '(0 1 2 300 3 4000) n 3)
          for c = (make-array (+ n 8) :element-type '(unsigned-byte 16)
                                :initial-element 4321)
          do (vectorize-u16-subtract a b c n)
             (dotimes (i n)
               (unless (= (aref c i) (logand (- (aref a i) (aref b i)) #xFFFF))
                 (error "~@<u16 subtract wrong at element ~D of ~D.~:@>" i n)))
             (loop for i from n below (+ n 8)
                   do (unless (= (aref c i) 4321)
                        (error "~@<u16 loop wrote element ~D past its bound.~:@>" i))))
    (loop for n in *vectorize-lengths*
          for a = (vectorize-integer-array '(signed-byte 16) '(0 1 -2 300 -32768 4000) n)
          for b = (vectorize-integer-array '(signed-byte 16) '(0 1 -2 300 3 4000) n 3)
          for c = (make-array n :element-type '(signed-byte 16) :initial-element 0)
          do (vectorize-s16-add a b c n)
             (dotimes (i n)
               (unless (= (aref c i) (vectorize-sign-extend (+ (aref a i) (aref b i)) 16))
                 (error "~@<s16 add wrong at element ~D of ~D: ~S.~:@>" i n (aref c i)))))
    (assert (search "psubw" (disassembly #'vectorize-u16-subtract))
            () "16-bit subtraction is not packed")
    (assert (search "paddw" (disassembly #'vectorize-s16-add))
            () "16-bit addition is not packed")))

;;;; A declaration of (VECTOR element-type) says the element type but not that the
;;;; data is inline, so the loop has to work for any vector of that type.

(defun vectorize-vector-declared (a b c n)
  (declare (type (vector single-float) a b c) (type vectorize-index n) (optimize speed))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))
  c)

(test-util:with-test (:name :vectorize-vector-declared :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    (assert (some (lambda (mnemonic) (search mnemonic (disassembly #'vectorize-vector-declared)))
                  *vectorize-packed-mnemonics*)
            () "a loop over (VECTOR SINGLE-FLOAT) was not vectorized")
    (macrolet ((case-values (make-a make-b context)
                 `(loop for n in '(0 1 2 3 5 8 17)
                        for a = (funcall ,make-a 'single-float *vectorize-single-values* n)
                        for b = (funcall ,make-b 'single-float *vectorize-single-values* n 3)
                        for c = (make-array n :element-type 'single-float :initial-element 0.0f0)
                        do (vectorize-vector-declared a b c n)
                           (dotimes (i n)
                             (unless (vectorize-same-value-p (aref c i)
                                                             (+ (aref a i) (aref b i)))
                               (error "~@<~A: element ~D of ~D is ~S, expected ~S.~:@>"
                                      ,context i n (aref c i) (+ (aref a i) (aref b i))))))))
      ;; simple vectors take the fast path
      (case-values #'vectorize-array #'vectorize-array "simple")
      ;; and adjustable ones still compute the right thing
      (case-values (lambda (element-type values count &optional (offset 0))
                     (make-array count :element-type element-type :adjustable t
                                       :initial-contents (vectorize-fill-list values count offset)))
                   (lambda (element-type values count &optional (offset 0))
                     (make-array count :element-type element-type :adjustable t
                                       :initial-contents (vectorize-fill-list values count offset)))
                   "adjustable"))))

;;;; An offset may be a variable.  It has to be known not to be negative before the
;;;; packed loop runs, so a negative one falls back to the scalar loop.

(defun vectorize-variable-offset (a b c n k)
  (declare (type (simple-array single-float (*)) a b c) (type vectorize-index n k)
           (optimize speed))
  (dotimes (i n)
    (setf (aref c i) (+ (aref a i) (aref b (+ i k)))))
  c)

(test-util:with-test (:name :vectorize-variable-offset :skipped-on (not :x86-64))
  (loop for k in '(0 1 3 8)
        do (loop for n in '(0 1 2 3 5 17)
                 for a = (vectorize-array 'single-float *vectorize-single-values* n)
                 for b = (vectorize-array 'single-float *vectorize-single-values* (+ n k 4))
                 for c = (make-array n :element-type 'single-float :initial-element 0.0f0)
                 do (vectorize-variable-offset a b c n k)
                    (dotimes (i n)
                      (unless (vectorize-same-value-p (aref c i) (+ (aref a i) (aref b (+ i k))))
                        (error "~@<k=~D: element ~D of ~D is ~S, expected ~S.~:@>"
                               k i n (aref c i) (+ (aref a i) (aref b (+ i k))))))))
  ;; a negative offset must not read out of bounds: the guard sends it to the
  ;; scalar loop, which reports the index error the loop always would
  (let ((a (vectorize-array 'single-float *vectorize-single-values* 4))
        (b (vectorize-array 'single-float *vectorize-single-values* 4))
        (c (make-array 4 :element-type 'single-float :initial-element 0.0f0)))
    (assert-error (vectorize-variable-offset a b c 4 -1) error)))

;;;; A negative scalar broadcast into narrow lanes.  Untagging a negative fixnum
;;;; sign-extends it, so a byte or word has to be masked to the lane width before it
;;;; is duplicated by a multiply; getting that wrong turns -3 into (-3 -4 -4 -4).

(defun vectorize-u8-add-negative (a c n)
  (declare (type (simple-array (unsigned-byte 8) (*)) a c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) -3))))

(defun vectorize-s8-add-negative (a c n)
  (declare (type (simple-array (signed-byte 8) (*)) a c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) -3))))

(defun vectorize-s16-add-negative (a c n)
  (declare (type (simple-array (signed-byte 16) (*)) a c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) -3))))

(defun vectorize-s16-subtract-literal (a c n)
  (declare (type (simple-array (signed-byte 16) (*)) a c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (- (aref a i) 300))))

(test-util:with-test (:name :vectorize-negative-narrow-splat :skipped-on (not :x86-64))
  (loop for n in *vectorize-lengths*
        for a = (vectorize-integer-array '(unsigned-byte 8) '(0 1 2 200 255 7) n)
        for c = (make-array n :element-type '(unsigned-byte 8) :initial-element 0)
        do (vectorize-u8-add-negative a c n)
           (dotimes (i n)
             (unless (= (aref c i) (logand (+ (aref a i) -3) #xFF))
               (error "~@<u8 + -3: element ~D of ~D is ~S, expected ~S.~:@>"
                      i n (aref c i) (logand (+ (aref a i) -3) #xFF)))))
  (loop for n in *vectorize-lengths*
        for a = (vectorize-integer-array '(signed-byte 8) '(0 1 2 -3 127 -128) n)
        for c = (make-array n :element-type '(signed-byte 8) :initial-element 0)
        do (vectorize-s8-add-negative a c n)
           (dotimes (i n)
             (unless (= (aref c i) (vectorize-sign-extend (+ (aref a i) -3) 8))
               (error "~@<s8 + -3: element ~D of ~D is ~S.~:@>" i n (aref c i)))))
  (loop for n in *vectorize-lengths*
        for a = (vectorize-integer-array '(signed-byte 16) '(0 1 2 -3 32767 -32768) n)
        for c = (make-array n :element-type '(signed-byte 16) :initial-element 0)
        do (vectorize-s16-add-negative a c n)
           (dotimes (i n)
             (unless (= (aref c i) (vectorize-sign-extend (+ (aref a i) -3) 16))
               (error "~@<s16 + -3: element ~D of ~D is ~S.~:@>" i n (aref c i)))))
  (loop for n in *vectorize-lengths*
        for a = (vectorize-integer-array '(signed-byte 16) '(0 1 299 300 -400 32767) n)
        for c = (make-array n :element-type '(signed-byte 16) :initial-element 0)
        do (vectorize-s16-subtract-literal a c n)
           (dotimes (i n)
             (unless (= (aref c i) (vectorize-sign-extend (- (aref a i) 300) 16))
               (error "~@<s16 - 300: element ~D of ~D is ~S.~:@>" i n (aref c i))))))

;;;; Byte-wise minimum and maximum (image clamping), unsigned only: SSE2 has
;;;; PMINUB/PMAXUB but the signed and word-width forms are SSE4.1.

(defun vectorize-u8-clamp (a c n)
  (declare (type (simple-array (unsigned-byte 8) (*)) a c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (min (aref a i) 200)))
  c)

(defun vectorize-u8-max-pair (a b c n)
  (declare (type (simple-array (unsigned-byte 8) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (max (aref a i) (aref b i))))
  c)

(defun vectorize-s8-min-pair (a b c n)
  (declare (type (simple-array (signed-byte 8) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (min (aref a i) (aref b i))))
  c)

(test-util:with-test (:name :vectorize-byte-min-max :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    (loop for n in *vectorize-lengths*
          for a = (vectorize-integer-array '(unsigned-byte 8) '(0 1 199 200 201 255 7 42) n)
          for c = (make-array n :element-type '(unsigned-byte 8) :initial-element 0)
          do (vectorize-u8-clamp a c n)
             (dotimes (i n)
               (unless (= (aref c i) (min (aref a i) 200))
                 (error "~@<u8 clamp: element ~D of ~D is ~S.~:@>" i n (aref c i)))))
    (loop for n in *vectorize-lengths*
          for a = (vectorize-integer-array '(unsigned-byte 8) '(0 1 199 200 255 7) n)
          for b = (vectorize-integer-array '(unsigned-byte 8) '(3 2 201 199 1 8) n 2)
          for c = (make-array n :element-type '(unsigned-byte 8) :initial-element 0)
          do (vectorize-u8-max-pair a b c n)
             (dotimes (i n)
               (unless (= (aref c i) (max (aref a i) (aref b i)))
                 (error "~@<u8 max: element ~D of ~D is ~S.~:@>" i n (aref c i))))
             ;; and the signed case still computes
             (vectorize-s8-min-pair (make-array n :element-type '(signed-byte 8) :initial-element -5)
                                    (make-array n :element-type '(signed-byte 8) :initial-element 3)
                                    (make-array n :element-type '(signed-byte 8) :initial-element 0)
                                    n))
    (assert (search "pminub" (disassembly #'vectorize-u8-clamp))
            () "byte minimum is not packed")
    (assert (search "pmaxub" (disassembly #'vectorize-u8-max-pair))
            () "byte maximum is not packed")
    ;; signed bytes have no packed minimum in SSE2, so that width biases the lanes
    ;; onto unsigned order, min's them with pminub and biases them back; the 256-bit
    ;; width has vpminsb and needs none of that.  Either way the result has to hold
    ;; for negatives, where unsigned order says otherwise
    (let ((text (disassembly #'vectorize-s8-min-pair)))
      (assert (or (search "pminsb" text) (search "pminub" text))
              () "signed byte minimum is not packed: ~a" text))
    (loop for n in *vectorize-lengths*
          for a = (vectorize-integer-array '(signed-byte 8) '(-128 -5 -1 0 3 127) n)
          for b = (vectorize-integer-array '(signed-byte 8) '(3 -6 1 -1 -128 0) n 2)
          for c = (make-array n :element-type '(signed-byte 8) :initial-element 0)
          do (vectorize-s8-min-pair a b c n)
             (dotimes (i n)
               (let ((want (min (aref a i) (aref b i))))
                 (unless (= (aref c i) want)
                   (error "~@<s8 min: element ~D of ~D is ~S, expected ~S~:@>"
                          i n (aref c i) want)))))))

;;;; (INCF place delta) and (DECF place delta) are stores too.

(defun vectorize-u8-accumulate (a c n)
  (declare (type (simple-array (unsigned-byte 8) (*)) a c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (incf (aref c i) (aref a i)))
  c)

(defun vectorize-single-incf-constant (c n)
  (declare (type (simple-array single-float (*)) c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (incf (aref c i) 2.0))
  c)

(defun vectorize-s32-decf (b c n)
  (declare (type (simple-array (signed-byte 32) (*)) b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (decf (aref c i) (aref b i)))
  c)

(test-util:with-test (:name :vectorize-incf-store :skipped-on (not :x86-64))
  (loop for n in *vectorize-lengths*
        for a = (vectorize-integer-array '(unsigned-byte 8) '(1 2 3 4 5) n)
        for c = (vectorize-integer-array '(unsigned-byte 8) '(10 20 30 40 50) n 1)
        for start = (copy-seq c)
        do (vectorize-u8-accumulate a c n)
           (dotimes (i n)
             (unless (= (aref c i) (logand (+ (aref start i) (aref a i)) #xFF))
               (error "~@<incf: element ~D of ~D is ~S.~:@>" i n (aref c i)))))
  (loop for n in *vectorize-lengths*
        for c = (make-array n :element-type 'single-float :initial-element 1.0f0)
        do (vectorize-single-incf-constant c n)
           (dotimes (i n)
             (unless (= (aref c i) 3.0f0)
               (error "~@<incf by constant: element ~D of ~D is ~S.~:@>" i n (aref c i)))))
  (loop for n in *vectorize-lengths*
        for b = (vectorize-integer-array '(signed-byte 32) '(1 2 3 4 5) n)
        for c = (vectorize-integer-array '(signed-byte 32) '(10 20 30 40 50) n)
        for start = (copy-seq c)
        do (vectorize-s32-decf b c n)
           (dotimes (i n)
             (unless (= (aref c i) (- (aref start i) (aref b i)))
               (error "~@<decf: element ~D of ~D is ~S.~:@>" i n (aref c i)))))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    (assert (search "paddb" (disassembly #'vectorize-u8-accumulate))
            () "an INCF loop was not vectorized")))

;;;; DO loops that count the same way a DOTIMES does.

(defun vectorize-do-add (a b c n)
  (declare (type (simple-array single-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (do ((i 0 (1+ i)))
      ((>= i n))
    (setf (aref c i) (+ (aref a i) (aref b i))))
  c)

(defun vectorize-do-add-plus (a b c n)
  (declare (type (simple-array single-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (do ((i 0 (+ i 1)))
      ((= i n))
    (setf (aref c i) (+ (aref a i) (aref b i))))
  c)

(defun vectorize-do-with-result (a b c n)
  (declare (type (simple-array single-float (*)) a b c) (type vectorize-index n)
           (optimize speed))
  (do ((i 0 (1+ i)))
      ((>= i n) c)
    (setf (aref c i) (+ (aref a i) (aref b i)))))

(test-util:with-test (:name :vectorize-do-loop :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    (loop for n in *vectorize-lengths*
          for a = (vectorize-array 'single-float *vectorize-single-values* n)
          for b = (vectorize-array 'single-float *vectorize-single-values* n 3)
          for c = (make-array n :element-type 'single-float :initial-element 0.0f0)
          do (dolist (function (list #'vectorize-do-add #'vectorize-do-add-plus))
               (let ((c (make-array n :element-type 'single-float :initial-element 0.0f0)))
                 (funcall function a b c n)
                 (dotimes (i n)
                   (unless (vectorize-same-value-p (aref c i) (+ (aref a i) (aref b i)))
                     (error "~@<~S, length ~D: element ~D is ~S.~:@>"
                            function n i (aref c i))))))
             ;; a DO with a result form is not one of the shapes taken, but computes
             (assert (eq (vectorize-do-with-result a b c n) c))
             (dotimes (i n)
               (unless (vectorize-same-value-p (aref c i) (+ (aref a i) (aref b i)))
                 (error "~@<a DO with a result form is wrong at element ~D of ~D.~:@>" i n))))
    (dolist (function (list #'vectorize-do-add #'vectorize-do-add-plus))
      (assert (some (lambda (mnemonic) (search mnemonic (disassembly function)))
                    *vectorize-packed-mnemonics*)
              () "~S was not vectorized" function))
    (assert (notany (lambda (mnemonic) (search mnemonic (disassembly #'vectorize-do-with-result)))
                    '("addps" "addpd"))
            () "a DO with a result form was vectorized")))

;;;; Reductions: a sum of a fixed-width integer array into a variable of that width.
;;;; Integer addition wraps at the declared width whatever the order, so adding the
;;;; lanes up at the end gives the same answer.  A float accumulator would not
;;;; (addition is not associative for floats), so it is refused.

(defun vectorize-u32-sum (a n total)
  (declare (type (simple-array (unsigned-byte 32) (*)) a) (type vectorize-index n)
           (type (unsigned-byte 32) total) (optimize speed (safety 0)))
  (dotimes (i n) (incf total (aref a i)))
  total)

(defun vectorize-s32-sum (a n total)
  (declare (type (simple-array (signed-byte 32) (*)) a) (type vectorize-index n)
           (type (signed-byte 32) total) (optimize speed (safety 0)))
  (dotimes (i n) (incf total (aref a i)))
  total)

(defun vectorize-single-sum (a n total)
  "A float sum: must stay scalar, and must still be right."
  (declare (type (simple-array single-float (*)) a) (type vectorize-index n)
           (type single-float total) (optimize speed))
  (dotimes (i n) (incf total (aref a i)))
  total)

(test-util:with-test (:name :vectorize-integer-reduction :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    (loop with total0 = 100
          for n in *vectorize-lengths*
          for a = (vectorize-integer-array '(unsigned-byte 32) '(1 2 3 4 5 6 7) n)
          do (let ((sum (vectorize-u32-sum a n total0)))
               (unless (= sum
                          (let ((expected total0))
                            (dotimes (i n expected) (setf expected (logand (+ expected (aref a i))
                                                                           #xFFFFFFFF)))))
                 (error "~@<u32 sum of ~D elements is ~S.~:@>" n sum))))
    (loop with total0 = -100
          for n in *vectorize-lengths*
          for a = (vectorize-integer-array '(signed-byte 32) '(1 -2 3 -4 5) n)
          do (let ((sum (vectorize-s32-sum a n total0)))
               (unless (= sum
                          (let ((expected total0))
                            (dotimes (i n expected)
                              (setf expected (vectorize-sign-extend (+ expected (aref a i)) 32)))))
                 (error "~@<s32 sum of ~D elements is ~S.~:@>" n sum))))
    (assert (search "paddd" (disassembly #'vectorize-u32-sum))
            () "an integer reduction was not vectorized")
    ;; A total that overflows the accumulator's declared width is undefined
    ;; behaviour in the scalar loop at (safety 0): it does not truncate, it just
    ;; produces a value outside the declared type.  So the packed sum, which is
    ;; computed modulo that width, agrees with the scalar one only up to
    ;; congruence, and this test records that boundary.
    (let* ((n 4000)
           (a (make-array n :element-type '(unsigned-byte 32) :initial-element #xFFFFFFFF))
           (total (vectorize-u32-sum a n 0))
           (exact (* n #xFFFFFFFF)))
      (assert (= (mod (- exact total) (expt 2 32)) 0)
              () "the packed sum of ~D elements is ~D, not congruent to ~D mod 2^32"
              n total exact)
      (when (< exact (expt 2 32))
        (assert (= total exact) () "a sum that fits was not exact")))
    ;; a float sum changes with the order of the additions, so it stays scalar
    (assert (notany (lambda (mnemonic) (search mnemonic (disassembly #'vectorize-single-sum)))
                    '("addps" "addpd"))
            () "a float reduction was vectorized")
    (loop for n in *vectorize-lengths*
          for a = (vectorize-array 'single-float '(1.0f0 2.0f0 3.0f0 4.0f0) n)
          do (let ((expected 0.5f0))
               (dotimes (i n) (setf expected (+ expected (aref a i))))
               (unless (eql (vectorize-single-sum a n 0.5f0) expected)
                 (error "~@<scalar float sum of ~D elements changed.~:@>" n))))))

;;;; A value that depends on the index: the loop index converted to the element
;;;; type becomes a lane vector, and a constant ramp supplies the lane offsets.

(defun vectorize-gradient (c n)
  (declare (type (simple-array single-float (*)) c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (* (float i) 0.5)))
  c)

(defun vectorize-gradient-coerce (c n)
  (declare (type (simple-array double-float (*)) c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (+ (coerce i 'double-float) 1d0)))
  c)

(defun vectorize-index-and-array (a c n)
  (declare (type (simple-array single-float (*)) a c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (+ (* (float i) 2.0) (aref a i))))
  c)

(defun vectorize-index-into-integers (c n)
  "An integer element type would need its own index conversion: left scalar."
  (declare (type (simple-array (unsigned-byte 32) (*)) c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) i))
  c)

(test-util:with-test (:name :vectorize-index-affine :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    (loop for n in *vectorize-lengths*
          for c = (make-array n :element-type 'single-float :initial-element 0.0f0)
          do (vectorize-gradient c n)
             (dotimes (i n)
               (unless (eql (aref c i) (coerce (* (float i) 0.5) 'single-float))
                 (error "~@<gradient: element ~D of ~D is ~S.~:@>" i n (aref c i)))))
    (loop for n in *vectorize-lengths*
          for c = (make-array n :element-type 'double-float :initial-element 0d0)
          do (vectorize-gradient-coerce c n)
             (dotimes (i n)
               (unless (= (aref c i) (+ (coerce i 'double-float) 1d0))
                 (error "~@<double gradient: element ~D of ~D is ~S.~:@>" i n (aref c i)))))
    (loop for n in *vectorize-lengths*
          for a = (vectorize-array 'single-float *vectorize-single-values* n)
          for c = (make-array n :element-type 'single-float :initial-element 0.0f0)
          do (vectorize-index-and-array a c n)
             (dotimes (i n)
               (unless (eql (aref c i) (+ (coerce (* (float i) 2.0) 'single-float) (aref a i)))
                 (error "~@<index + array: element ~D of ~D is ~S.~:@>" i n (aref c i)))))
    (let ((text (disassembly #'vectorize-gradient)))
      (assert (some (lambda (mnemonic) (search mnemonic text))
                    '("cvtsi2ss" "addps" "movaps"))
              () "the gradient loop was not vectorized:~%~A" text))
    (let ((text (disassembly #'vectorize-index-into-integers)))
      (assert (notany (lambda (mnemonic) (search mnemonic text)) '("paddd" "movdqu"))
              () "an integer index loop was vectorized:~%~A" text))))


;;;; Missed-optimization notes: a loop in the pass's shape that was not rewritten
;;;; says why, the way "unable to open-code X due to type uncertainty" does, so
;;;; the user can act on it.

(test-util:with-test (:name :vectorize-missed-declaration-note :skipped-on (not :x86-64))
  ;; B is undeclared: the note should name the declaration that would allow it
  (let ((notes (vectorize-notes-for
                '(lambda (a b c n)
                   (declare (type (simple-array single-float (*)) a c) (type fixnum n)
                            (optimize speed))
                   (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))))))
    (assert (vectorize-note-mentions notes "unable to vectorize") () "no note: ~s" notes)
    (assert (vectorize-note-mentions notes "simple-array SINGLE-FLOAT") ()
            "the note does not suggest a declaration: ~s" notes)
    (assert (vectorize-note-mentions notes "B") () "the note does not name B: ~s" notes))
  ;; and with no declarations at all it still says something
  (let ((notes (vectorize-notes-for
                '(lambda (a c n)
                   (declare (type fixnum n) (optimize speed))
                   (dotimes (i n) (setf (aref c i) (aref a i)))))))
    (assert (vectorize-note-mentions notes "not known to be an array") () "no note: ~s" notes)))

(test-util:with-test (:name :vectorize-missed-operator-note :skipped-on (not :x86-64))
  ;; nothing multiplies 64-bit lanes at any width, so that is what the note says
  (let ((notes (vectorize-notes-for
                '(lambda (a b c n)
                   (declare (type (simple-array (unsigned-byte 64) (*)) a b c)
                            (type fixnum n) (optimize speed))
                   (dotimes (i n) (setf (aref c i) (* (aref a i) (aref b i))))))))
    (assert (vectorize-note-mentions notes "not available for") () "no note: ~s" notes))
  ;; 32-bit multiply exists on sse4.1 and later, and where it does not the note is
  ;; the same one; where it does, the loop declines at default safety instead,
  ;; because the lanes would wrap where a checked loop signals
  (let ((notes (vectorize-notes-for
                '(lambda (a b c n)
                   (declare (type (simple-array (unsigned-byte 32) (*)) a b c)
                            (type fixnum n) (optimize speed))
                   (dotimes (i n) (setf (aref c i) (* (aref a i) (aref b i))))))))
    (if (member :sse4 sb-c:*backend-subfeatures*)
        (assert (vectorize-note-mentions notes "safety 0") () "no note: ~s" notes)
        (assert (vectorize-note-mentions notes "not available for") () "no note: ~s" notes)))
  ;; byte lanes outside (safety 0)
  (let ((notes (vectorize-notes-for
                '(lambda (a b c n)
                   (declare (type (simple-array (unsigned-byte 8) (*)) a b c)
                            (type fixnum n) (optimize speed))
                   (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))))))
    (assert (vectorize-note-mentions notes "safety 0") () "no note: ~s" notes)))

(test-util:with-test (:name :vectorize-no-note-when-quiet :skipped-on (not :x86-64))
  ;; a loop the pass does vectorize gets the success note and no complaint
  (let ((notes (vectorize-notes-for
                '(lambda (a b c n)
                   (declare (type (simple-array single-float (*)) a b c) (type fixnum n)
                            (optimize speed))
                   (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))))))
    (assert (not (vectorize-note-mentions notes "unable to vectorize"))
            () "a vectorized loop was reported as a missed optimization: ~s" notes))
  ;; a loop that is not the pass's shape at all is silent
  (let ((notes (vectorize-notes-for
                '(lambda (v n)
                   (declare (type vector v) (type fixnum n) (optimize speed))
                   (dotimes (i n) (print i)) v))))
    (assert (null notes) () "a loop of another shape was reported on: ~s" notes))
  ;; and *auto-vectorize-notes* turns all of it off
  (let ((notes (vectorize-notes-for
                '(lambda (a b c n)
                   (declare (type (simple-array single-float (*)) a c) (type fixnum n)
                            (optimize speed))
                   (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i)))))
                :notes nil)))
    (assert (null notes) () "notes were produced with *auto-vectorize-notes* nil: ~s" notes)))


;;;; Regressions for the review of the first six rounds.  Each of these was a real
;;;; bug: an integer loop that wrapped where the scalar loop signals, a guard that
;;;; dropped an offset, a subscript that put the loop index into the guard, a
;;;; reduction over data that is not inline, a shadowed name that kept its type,
;;;; and a COERCE spelling that never matched.

(defun vectorize-u32-add-unsafe (a b c n)
  (declare (type (simple-array (unsigned-byte 32) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))
  c)

(with-test (:name :vectorize-integer-lanes-need-safety-zero)
  (flet ((packed-p (function)
           (let ((text (string-downcase (with-output-to-string (stream)
                                          (disassemble function :stream stream)))))
             (some (lambda (mnemonic) (search mnemonic text)) '("paddd" "paddq" "paddw")))))
    ;; at the default safety a checked store signals where a packed lane wraps, so
    ;; the loop must stay scalar and say why
    (let ((notes (vectorize-notes-for
                  '(lambda (a b c n)
                     (declare (type (simple-array (unsigned-byte 32) (*)) a b c)
                              (type fixnum n) (optimize speed))
                     (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))))))
      (assert (vectorize-note-mentions notes "safety 0") () "no safety note: ~s" notes))
    ;; and at (safety 0) it is packed
    (assert (packed-p #'vectorize-u32-add-unsafe) () "u32 lanes were not packed at (safety 0)")))

(defun vectorize-alias-offset (c d n)
  "The destination is also a source, with an offset: the guard must cover the read."
  (declare (type (simple-array single-float (*)) c d) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (+ (aref c (+ i 3)) (aref d i))))
  c)

(with-test (:name :vectorize-alias-offset-guard)
  ;; length == n means the read at (+ i 3) leaves the array, so the loop must fall
  ;; back to the scalar one, which reports it, rather than reading past the end
  (loop for n in *vectorize-lengths*
        do (when (plusp n)            ; nothing is read from an empty range
             (let ((c (make-array n :element-type 'single-float :initial-element 1.0f0))
                   (d (make-array n :element-type 'single-float :initial-element 2.0f0)))
               (assert-error (vectorize-alias-offset c d n) error))))
  ;; with room for the offsets it is fine, and still packed
  (let* ((n 16)
         (c (make-array (+ n 3) :element-type 'single-float :initial-element 1.0f0))
         (d (make-array (+ n 3) :element-type 'single-float :initial-element 2.0f0))
         (before (copy-seq c)))
    (vectorize-alias-offset c d n)
    (dotimes (i n)
      (unless (eql (aref c i) (+ (aref before (+ i 3)) (aref d i)))
        (error "~@<alias offset: element ~D is ~S.~:@>" i (aref c i))))))

(defun vectorize-doubled-subscript (a c n)
  "(aref a (+ i i)): the offset mentions the index, so the loop must stay scalar."
  (declare (type (simple-array single-float (*)) a c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (aref a (+ i i))))
  c)

(with-test (:name :vectorize-index-referencing-offset)
  (loop for n in *vectorize-lengths*
        for a = (vectorize-array 'single-float *vectorize-single-values* (* 2 n))
        for c = (make-array n :element-type 'single-float :initial-element 0.0f0)
        do (vectorize-doubled-subscript a c n)
           (dotimes (i n)
             (unless (eql (aref c i) (aref a (* 2 i)))
               (error "~@<doubled subscript: element ~D of ~D is ~S.~:@>" i n (aref c i))))))

(defun vectorize-u32-sum-any (a n total)
  "A reduction whose source may not be simple."
  (declare (type (vector (unsigned-byte 32)) a) (type vectorize-index n)
           (type (unsigned-byte 32) total) (optimize speed (safety 0)))
  (dotimes (i n) (incf total (aref a i)))
  total)

(with-test (:name :vectorize-reduction-needs-inline-data)
  (let ((adjustable (make-array 8 :element-type '(unsigned-byte 32) :adjustable t))
        (simple (make-array 8 :element-type '(unsigned-byte 32))))
    (dotimes (i 8) (setf (aref adjustable i) (1+ i) (aref simple i) (1+ i)))
    (assert (= (vectorize-u32-sum-any adjustable 8 0) 36))
    (assert (= (vectorize-u32-sum-any simple 8 0) 36))))

(defun vectorize-shadowed-array (a n)
  "A rebind of a declared array name to a vector of T: not that array any more."
  (declare (type (simple-array single-float (*)) a) (type vectorize-index n)
           (optimize speed))
  (let ((a (make-array n :element-type t :initial-element nil)))
    (dotimes (i n) (setf (aref a i) 1.0f0))
    (values (aref a 0) (type-of (aref a 0)))))

(with-test (:name :vectorize-shadowed-binding)
  (multiple-value-bind (value type)
      (vectorize-shadowed-array (make-array 8 :element-type 'single-float :initial-element 0.0f0) 8)
    (assert (eql value 1.0f0) () "the shadowed vector holds ~S" value)
    (assert (eq type 'single-float) () "the shadowed vector holds a ~S" type)))

(defun vectorize-coerce-index (c n)
  (declare (type (simple-array single-float (*)) c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (coerce i 'single-float)))
  c)

(with-test (:name :vectorize-coerce-index)
  (loop for n in *vectorize-lengths*
        for c = (make-array n :element-type 'single-float :initial-element 0.0f0)
        do (vectorize-coerce-index c n)
           (dotimes (i n)
             (unless (eql (aref c i) (coerce i 'single-float))
               (error "~@<coerce: element ~D of ~D is ~S.~:@>" i n (aref c i)))))
  ;; the COERCE spelling is a matcher branch, so assert it takes it
  (let ((text (string-downcase (with-output-to-string (stream)
                                 (disassemble #'vectorize-coerce-index :stream stream)))))
    (assert (some (lambda (mnemonic) (search mnemonic text)) '("addps" "cvtsi2ss"))
            () "the COERCE spelling was not vectorized:~%~A" text)))


;;;; Stencils.  A read before the loop variable cannot be vectorized, because the
;;;; scalar loop it replaces would fault at index 0 anyway, so the pass declines it
;;;; and says what to write instead.  The pattern real code uses, a shifted
;;;; destination with non-negative reads, is vectorized.

(defun vectorize-stencil-shifted (in out n)
  "out[i+1] = in[i] + in[i+1] + in[i+2]: a three-tap window written the legal way."
  (declare (type (simple-array single-float (*)) in out) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n)
    (setf (aref out (+ i 1)) (+ (aref in i) (aref in (+ i 1)) (aref in (+ i 2)))))
  out)

(defun vectorize-stencil-backward (in out n)
  "Reads in[i-1]: declined, because it is wrong at i = 0 as written."
  (declare (type (simple-array single-float (*)) in out) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref out i) (- (aref in i) (aref in (- i 1)))))
  out)

(test-util:with-test (:name :vectorize-stencils :skipped-on (not :x86-64))
  (flet ((disassembly (function)
           (string-downcase (with-output-to-string (stream)
                              (disassemble function :stream stream)))))
    ;; the legal three-tap is packed
    (loop for n in *vectorize-lengths*
          for in = (vectorize-array 'single-float *vectorize-single-values* (+ n 2))
          for out = (make-array (+ n 1) :element-type 'single-float :initial-element 0.0f0)
          do (vectorize-stencil-shifted in out n)
             (dotimes (i n)
               (let ((want (+ (aref in i) (aref in (+ i 1)) (aref in (+ i 2)))))
                 (unless (vectorize-same-value-p (aref out (1+ i)) want)
                   (error "~@<stencil: element ~D of ~D is ~S, expected ~S.~:@>"
                          (1+ i) n (aref out (1+ i)) want)))))
    (assert (some (lambda (mnemonic) (search mnemonic (disassembly #'vectorize-stencil-shifted)))
                  *vectorize-packed-mnemonics*)
            () "the shifted three-tap was not vectorized")
    ;; the backward read is declined, computes what it always computed, and explains
    (assert (notany (lambda (mnemonic) (search mnemonic (disassembly #'vectorize-stencil-backward)))
                    '("addps" "subps"))
            () "a read before the loop variable was vectorized")
    (let ((notes (vectorize-notes-for
                  '(lambda (in out n)
                     (declare (type (simple-array single-float (*)) in out) (type fixnum n)
                              (optimize speed))
                     (dotimes (i n) (setf (aref out i) (- (aref in i) (aref in (- i 1)))))))))
      (assert (vectorize-note-mentions notes "shift the destination")
              () "no advice for the negative offset: ~s" notes))))

;;;; 16-bit lanes can multiply: PMULLW gives the low 16 bits of each product, which
;;;; is what storing into a 16-bit element gives, so the values agree with the
;;;; scalar loop at (safety 0).

(defun vectorize-u16-multiply (a b c n)
  (declare (type (simple-array (unsigned-byte 16) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (* (aref a i) (aref b i))))
  c)

(test-util:with-test (:name :vectorize-16-bit-multiply :skipped-on (not :x86-64))
  (loop for n in *vectorize-lengths*
        for a = (vectorize-integer-array '(unsigned-byte 16) '(3 300 1000 65535 7) n)
        for b = (vectorize-integer-array '(unsigned-byte 16) '(5 400 500 65535 11) n 2)
        for c = (make-array n :element-type '(unsigned-byte 16) :initial-element 0)
        do (vectorize-u16-multiply a b c n)
           (dotimes (i n)
             (let ((want (logand (* (aref a i) (aref b i)) #xFFFF)))
               (unless (= (aref c i) want)
                 (error "~@<u16 multiply: element ~D of ~D is ~S, expected ~S.~:@>"
                        i n (aref c i) want)))))
  (let ((text (string-downcase (with-output-to-string (stream)
                                 (disassemble #'vectorize-u16-multiply :stream stream)))))
    (assert (search "pmullw" text) () "u16 multiply is not packed:~%~A" text)))

;;;; Regression tests for the five findings of the second review.

;;; 1. UPTO includes its bound.  It is a synonym for BELOW in LOOP only in the
;;; sense that both take a bound; the last iteration differs, and the packed loop
;;; covering a shorter range is something a runtime guard cannot notice.
(defun vectorize-loop-upto-count (a c n)
  (declare (type (simple-array single-float (*)) a c) (type vectorize-index n)
           (optimize speed))
  (loop for i upto n do (setf (aref c i) (+ (aref a i) 1.0)))
  c)

(defun vectorize-loop-below-count (a c n)
  (declare (type (simple-array single-float (*)) a c) (type vectorize-index n)
           (optimize speed))
  (loop for i below n do (setf (aref c i) (+ (aref a i) 1.0)))
  c)

(with-test (:name :vectorize-loop-upto-includes-bound :skipped-on (not :x86-64))
  ;; the destination holds one element more than the loop's loop-variable bound
  (loop for n from 0 to 20
        do (let* ((size (+ n 2))
                  (a (make-array size :element-type 'single-float
                                      :initial-element 1.5))
                  (c (make-array size :element-type 'single-float
                                      :initial-element -1.0)))
             (vectorize-loop-upto-count a c n)
             (loop for i from 0 to n
                   do (assert (= (aref c i) 2.5) ()
                              "~@<UPTO dropped iteration ~D of 0..~D~:@>" i n))
             (let ((c2 (make-array size :element-type 'single-float
                                        :initial-element -1.0)))
               (vectorize-loop-below-count a c2 n)
               (loop for i from 0 to n
                     do (assert (if (< i n) (= (aref c2 i) 2.5) (= (aref c2 i) -1.0)) ()
                                "~@<BELOW ran iteration ~D of 0..~D~:@>" i n)))))
  (let ((notes (vectorize-notes-for
                '(lambda (a c n)
                   (declare (type (simple-array single-float (*)) a c)
                            (type vectorize-index n) (optimize speed))
                   (loop for i upto n do (setf (aref c i) (+ (aref a i) 1.0)))))))
    (assert (some (lambda (line) (search "vectorized" line :test #'char-equal)) notes)
            () "UPTO loops are no longer vectorized: ~A" notes)))

;;; 2. A destination written ahead of a read is not the same loop once the loop is
;;; packed: the scalar one reads back what it just stored.
(defun vectorize-alias-ahead (c n)
  (declare (type (simple-array (unsigned-byte 32) (*)) c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c (+ i 1)) (+ (aref c i) 1)))
  c)

(with-test (:name :vectorize-alias-ahead-declined :skipped-on (not :x86-64))
  (let ((notes (vectorize-notes-for
                '(lambda (c n)
                   (declare (type (simple-array (unsigned-byte 32) (*)) c)
                            (type vectorize-index n) (optimize speed (safety 0)))
                   (dotimes (i n) (setf (aref c (+ i 1)) (+ (aref c i) 1)))))))
    (assert (notany (lambda (line) (search "vectorized loop" line :test #'char-equal)) notes)
            () "A destination written ahead of its source was vectorized: ~A" notes))
  ;; and the values are whatever the scalar loop gives: a running total from 0
  (loop for n from 0 to 20
        do (let* ((c (make-array (1+ n) :element-type '(unsigned-byte 32)
                                       :initial-element 0))
                  (want (make-array (1+ n) :element-type '(unsigned-byte 32)
                                          :initial-element 0)))
             (dotimes (i n)
               (setf (aref want (+ i 1)) (1+ (aref want i))))
             (vectorize-alias-ahead c n)
             (assert (equalp c want) ()
                     "~@<aliasing ahead: got ~S, the scalar loop gives ~S~:@>"
                     (coerce c 'list) (coerce want 'list)))))

;;; 3. A policy declaration inside the body is not the policy the loop form is
;;; compiled under, so the loop must not be rewritten as if it were unchecked.
(with-test (:name :vectorize-body-optimize-declined :skipped-on (not :x86-64))
  (let ((notes (vectorize-notes-for
                '(lambda (a b c n)
                   (declare (type (simple-array (unsigned-byte 32) (*)) a b c)
                            (type vectorize-index n) (optimize (speed 3) (safety 0)))
                   (dotimes (i n)
                     (declare (optimize (safety 3)))
                     (setf (aref c i) (+ (aref a i) (aref b i))))))))
    (assert (notany (lambda (line) (search "vectorized loop" line :test #'char-equal)) notes)
            () "A body that declares (SAFETY 3) was vectorized unchecked: ~A" notes)))

;;; 4. A reduction over a variable offset must compile, and take the same elements
;;; as the scalar loop.
(defun vectorize-sum-variable-offset (a k n)
  (declare (type (simple-array (unsigned-byte 32) (*)) a) (type vectorize-index k n)
           (optimize speed (safety 0)))
  (let ((acc 0))
    (declare (type (unsigned-byte 32) acc))
    (dotimes (i n) (incf acc (aref a (+ i k))))
    acc))

(with-test (:name :vectorize-sum-variable-offset :skipped-on (not :x86-64))
  (loop for n in *vectorize-lengths*
        for k in '(0 1 2 3 0 1 2 3 0 1 2 3 0 1)
        do (let* ((size (+ n k))
                  (a (vectorize-integer-array '(unsigned-byte 32) '(1 2 3 4 5 6 7) size))
                  (want (let ((acc 0))
                          (dotimes (i n) (incf acc (aref a (+ i k))))
                          (logand acc #xFFFFFFFF))))
             (assert (= (vectorize-sum-variable-offset a k n) want) ()
                     "~@<variable-offset sum with k=~D, n=~D: got ~S, expected ~S~:@>"
                     k n (vectorize-sum-variable-offset a k n) want))))

;;; 5. The count is a form: it must be evaluated as often as DOTIMES evaluates it,
;;; which is once, even when it is a symbol macro over a side effect.
(defvar *vectorize-count-calls* 0)

(with-test (:name :vectorize-count-evaluated-once :skipped-on (not :x86-64))
  (let ((function
          (compile nil '(lambda (a c n)
                          (declare (type (simple-array single-float (*)) a c)
                                   (type vectorize-index n) (optimize speed))
                          (symbol-macrolet ((bound (progn (incf *vectorize-count-calls*) n)))
                            (dotimes (i bound)
                              (setf (aref c i) (+ (aref a i) 1.0))))
                          c))))
    (loop for n in *vectorize-lengths*
          do (let ((a (make-array n :element-type 'single-float :initial-element 1.0))
                   (c (make-array n :element-type 'single-float :initial-element 0.0)))
               (setf *vectorize-count-calls* 0)
               (funcall function a c n)
               (assert (= *vectorize-count-calls* 1) ()
                       "~@<the loop bound was evaluated ~D times for n=~D~:@>"
                       *vectorize-count-calls* n)
               (loop for i below n
                     do (assert (= (aref c i) 2.0) () "element ~D of ~D is ~S" i n (aref c i)))))))

;;;; Minimum and maximum on the narrower integer lanes, and the rule that lets an
;;;; exact integer expression vectorize without (safety 0).

(defun vectorize-s16-clamp (a c n)
  "Clamping 16-bit audio, at the default safety: min and max keep every value in
the lane, so nothing here can signal and the loop needs no (safety 0)."
  (declare (type (simple-array (signed-byte 16) (*)) a c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (max -1000 (min 1000 (aref a i)))))
  c)

(defun vectorize-u16-clamp (a c n)
  (declare (type (simple-array (unsigned-byte 16) (*)) a c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (max 1000 (aref a i))))
  c)

(defun vectorize-s8-clamp (a c n)
  (declare (type (simple-array (signed-byte 8) (*)) a c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (min 100 (aref a i))))
  c)

(with-test (:name :vectorize-narrow-integer-clamp :skipped-on (not :x86-64))
  ;; signed words, clamped by literal bounds on both sides
  (loop for n in *vectorize-lengths*
        for a = (vectorize-integer-array '(signed-byte 16) '(-32768 -1001 -3 0 1001 32767) n)
        for c = (make-array n :element-type '(signed-byte 16) :initial-element 0)
        do (vectorize-s16-clamp a c n)
           (dotimes (i n)
             (let ((want (max -1000 (min 1000 (aref a i)))))
               (unless (= (aref c i) want)
                 (error "~@<s16 clamp: element ~D of ~D is ~S, expected ~S~:@>"
                        i n (aref c i) want)))))
  ;; unsigned words, clamped only from below
  (loop for n in *vectorize-lengths*
        for b = (vectorize-integer-array '(unsigned-byte 16) '(0 1 1000 65535 42) n)
        for d = (make-array n :element-type '(unsigned-byte 16) :initial-element 0)
        do (vectorize-u16-clamp b d n)
           (dotimes (i n)
             (let ((want (max 1000 (aref b i))))
               (unless (= (aref d i) want)
                 (error "~@<u16 clamp: element ~D of ~D is ~S, expected ~S~:@>"
                        i n (aref d i) want)))))
  ;; signed bytes, clamped from above
  (loop for n in *vectorize-lengths*
        for e = (vectorize-integer-array '(signed-byte 8) '(-128 -1 0 1 100 127) n)
        for f = (make-array n :element-type '(signed-byte 8) :initial-element 0)
        do (vectorize-s8-clamp e f n)
           (dotimes (i n)
             (let ((want (min 100 (aref e i))))
               (unless (= (aref f i) want)
                 (error "~@<s8 clamp: element ~D of ~D is ~S, expected ~S~:@>"
                        i n (aref f i) want)))))
  ;; and the loops are packed, without any (safety 0) in sight
  (flet ((mnemonics (function)
           (let ((text (string-downcase (with-output-to-string (stream)
                                          (disassemble function :stream stream)))))
             (loop for mnemonic in '("pminsw" "pmaxsw" "psubusw" "pminub" "pmaxub" "paddw" "psubw"
                                     "pminsb" "pmaxsb" "pminuw" "pmaxuw")
                   when (search mnemonic text) collect mnemonic))))
    (assert (member "pminsw" (mnemonics #'vectorize-s16-clamp) :test #'string=) ()
            "s16 clamp is not packed: ~a" (mnemonics #'vectorize-s16-clamp))
    ;; unsigned words are a saturating difference at 128 bits and one vpmaxuw at 256
    (assert (some (lambda (m) (member m '("psubusw" "paddw" "psubw" "pmaxuw") :test #'string=))
                  (mnemonics #'vectorize-u16-clamp)) ()
            "u16 clamp is not packed: ~a" (mnemonics #'vectorize-u16-clamp))
    (assert (some (lambda (m) (member m '("pminub" "pminsb") :test #'string=))
                  (mnemonics #'vectorize-s8-clamp)) ()
            "s8 clamp is not packed: ~a" (mnemonics #'vectorize-s8-clamp))))

(with-test (:name :vectorize-inexact-integer-declined :skipped-on (not :x86-64))
  ;; arithmetic can wrap, so at the default safety this loop still declines, and
  ;; the note names the rule that would allow it
  (let ((notes (vectorize-notes-for
                '(lambda (a b c n)
                   (declare (type (simple-array (unsigned-byte 16) (*)) a b c)
                            (type vectorize-index n) (optimize speed))
                   (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))))))
    (assert (notany (lambda (line) (search "vectorized loop" line :test #'char-equal)) notes)
            () "an inexact u16 loop was vectorized at default safety: ~A" notes)
    (assert (some (lambda (line) (search "safety" line :test #'char-equal)) notes)
            () "the note does not name the (safety 0) rule: ~A" notes)))

;;;; Bitwise operations are exact too: an n-bit result always fits the lane, so a
;;;; masking loop vectorizes without (safety 0).

(defun vectorize-u8-mask (a c n)
  (declare (type (simple-array (unsigned-byte 8) (*)) a c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (logand (aref a i) #x7F)))
  c)

(defun vectorize-u32-mask (a c n)
  (declare (type (simple-array (unsigned-byte 32) (*)) a c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (logior (logand (aref a i) #xFF) #x80)))
  c)

(test-util:with-test (:name :vectorize-mask-at-any-safety :skipped-on (not :x86-64))
  (loop for n in *vectorize-lengths*
        for a = (vectorize-integer-array '(unsigned-byte 8) '(0 1 127 128 255 66) n)
        for c = (make-array n :element-type '(unsigned-byte 8) :initial-element 0)
        do (vectorize-u8-mask a c n)
           (dotimes (i n)
             (let ((want (logand (aref a i) #x7F)))
               (unless (= (aref c i) want)
                 (error "~@<u8 mask: element ~D of ~D is ~S, expected ~S~:@>"
                        i n (aref c i) want)))))
  (loop for n in *vectorize-lengths*
        for a = (vectorize-integer-array '(unsigned-byte 32) '(0 1 255 256 4294967295) n)
        for c = (make-array n :element-type '(unsigned-byte 32) :initial-element 0)
        do (vectorize-u32-mask a c n)
           (dotimes (i n)
             (let ((want (logior (logand (aref a i) #xFF) #x80)))
               (unless (= (aref c i) want)
                 (error "~@<u32 mask: element ~D of ~D is ~S, expected ~S~:@>"
                        i n (aref c i) want)))))
  ;; and packed, at the default policy
  (flet ((packed-p (function)
           (search "pand" (string-downcase (with-output-to-string (stream)
                                             (disassemble function :stream stream))))))
    (assert (packed-p #'vectorize-u8-mask) () "u8 mask is not packed")
    (assert (packed-p #'vectorize-u32-mask) () "u32 mask is not packed")))

;;;; Reading a narrow integer array into float lanes: an image buffer or 16-bit
;;;; audio consumed as floats.  The widen vop reads exactly the four (or two)
;;;; elements one float iteration consumes, so the float loop's guard covers it.

(defun vectorize-u8-to-single (in out n)
  (declare (type (simple-array (unsigned-byte 8) (*)) in)
           (type (simple-array single-float (*)) out) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref out i) (* 2.0 (aref in i))))
  out)

(defun vectorize-s8-to-single (in out n)
  (declare (type (simple-array (signed-byte 8) (*)) in)
           (type (simple-array single-float (*)) out) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref out i) (* 0.5 (aref in i))))
  out)

(defun vectorize-s16-to-double (in out n)
  (declare (type (simple-array (signed-byte 16) (*)) in)
           (type (simple-array double-float (*)) out) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref out i) (* 1.0d0 (aref in i))))
  out)

(test-util:with-test (:name :vectorize-narrow-source-widened :skipped-on (not :x86-64))
  (loop for n in *vectorize-lengths*
        for in = (vectorize-integer-array '(unsigned-byte 8) '(0 1 127 128 255 66) n)
        for out = (make-array n :element-type 'single-float :initial-element 0.0)
        do (vectorize-u8-to-single in out n)
           (dotimes (i n)
             (let ((want (* 2.0 (aref in i))))
               (unless (= (aref out i) want)
                 (error "~@<u8 to single: element ~D of ~D is ~S, expected ~S~:@>"
                        i n (aref out i) want)))))
  (loop for n in *vectorize-lengths*
        for in = (vectorize-integer-array '(signed-byte 8) '(-128 -5 0 5 127 66) n)
        for out = (make-array n :element-type 'single-float :initial-element 0.0)
        do (vectorize-s8-to-single in out n)
           (dotimes (i n)
             (let ((want (* 0.5 (aref in i))))
               (unless (= (aref out i) want)
                 (error "~@<s8 to single: element ~D of ~D is ~S, expected ~S~:@>"
                        i n (aref out i) want)))))
  (loop for n in *vectorize-lengths*
        for in = (vectorize-integer-array '(signed-byte 16) '(-32768 -3 0 3 32767) n)
        for out = (make-array n :element-type 'double-float :initial-element 0.0d0)
        do (vectorize-s16-to-double in out n)
           (dotimes (i n)
             (let ((want (* 1.0d0 (aref in i))))
               (unless (= (aref out i) want)
                 (error "~@<s16 to double: element ~D of ~D is ~S, expected ~S~:@>"
                        i n (aref out i) want)))))
  ;; and each is packed, with the widen instructions present
  (flet ((mnemonics (function)
           (let ((text (string-downcase (with-output-to-string (stream)
                                          (disassemble function :stream stream)))))
             (remove-if-not (lambda (m) (search m text))
                            '("punpcklbw" "punpcklwd" "psrld" "psrad" "cvtdq2ps"
                              "cvtdq2pd" "movd " "movq "
                              "pmovzxbd" "pmovsxbd" "pmovzxwd" "pmovsxwd")))))
    ;; the widen is the unpack sequence or, with SSE4.1, one narrowing load
    (assert (and (member "cvtdq2ps" (mnemonics #'vectorize-u8-to-single) :test #'string=)
                 (or (member "psrld" (mnemonics #'vectorize-u8-to-single) :test #'string=)
                     (member "pmovzxbd" (mnemonics #'vectorize-u8-to-single) :test #'string=)))
            () "u8 to single is not packed: ~a" (mnemonics #'vectorize-u8-to-single))
    (assert (and (member "cvtdq2ps" (mnemonics #'vectorize-s8-to-single) :test #'string=)
                 (or (member "psrad" (mnemonics #'vectorize-s8-to-single) :test #'string=)
                     (member "pmovsxbd" (mnemonics #'vectorize-s8-to-single) :test #'string=)))
            () "s8 to single is not packed: ~a" (mnemonics #'vectorize-s8-to-single))
    (assert (member "cvtdq2pd" (mnemonics #'vectorize-s16-to-double) :test #'string=) ()
            "s16 to double is not packed: ~a" (mnemonics #'vectorize-s16-to-double))))

;;; A 16-bit source is four words per packed iteration, so the read has to reach
;;; eight bytes; reading four left the upper two lanes reading whatever the movd
;;; left there, which is a zero.
(defun vectorize-u16-to-single (in out n)
  (declare (type (simple-array (unsigned-byte 16) (*)) in)
           (type (simple-array single-float (*)) out) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref out i) (* 1.0 (aref in i))))
  out)

(defun vectorize-s16-to-single (in out n)
  (declare (type (simple-array (signed-byte 16) (*)) in)
           (type (simple-array single-float (*)) out) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref out i) (* 1.0 (aref in i))))
  out)

(defun vectorize-u8-to-double (in out n)
  (declare (type (simple-array (unsigned-byte 8) (*)) in)
           (type (simple-array double-float (*)) out) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref out i) (* 1.0d0 (aref in i))))
  out)

(test-util:with-test (:name :vectorize-widen-wide-source :skipped-on (not :x86-64))
  (loop for n in *vectorize-lengths*
        for in = (vectorize-integer-array '(unsigned-byte 16) '(0 1 999 65535 12345) n)
        for out = (make-array n :element-type 'single-float :initial-element 0.0)
        do (vectorize-u16-to-single in out n)
           (dotimes (i n)
             (unless (= (aref out i) (coerce (aref in i) 'single-float))
               (error "~@<u16 to single: element ~D of ~D is ~S, expected ~S~:@>"
                      i n (aref out i) (coerce (aref in i) 'single-float)))))
  (loop for n in *vectorize-lengths*
        for in = (vectorize-integer-array '(signed-byte 16) '(-32768 -1 0 1 32767) n)
        for out = (make-array n :element-type 'single-float :initial-element 0.0)
        do (vectorize-s16-to-single in out n)
           (dotimes (i n)
             (unless (= (aref out i) (coerce (aref in i) 'single-float))
               (error "~@<s16 to single: element ~D of ~D is ~S, expected ~S~:@>"
                      i n (aref out i) (coerce (aref in i) 'single-float)))))
  (loop for n in *vectorize-lengths*
        for in = (vectorize-integer-array '(unsigned-byte 8) '(0 7 255 128 3) n)
        for out = (make-array n :element-type 'double-float :initial-element 0.0d0)
        do (vectorize-u8-to-double in out n)
           (dotimes (i n)
             (unless (= (aref out i) (coerce (aref in i) 'double-float))
               (error "~@<u8 to double: element ~D of ~D is ~S, expected ~S~:@>"
                      i n (aref out i) (coerce (aref in i) 'double-float))))))

;;; With SSE4.1 the four-lane widens are one narrowing load each; without it they
;;; are the unpack sequence.  Both must be present in the code that ran.
;;;; 256-bit lanes.
;;;;
;;;; Which width a loop takes is a property of the build, not of the program:
;;;; a build that enables :AVX2 publishes 256-bit primitives, the pass asks for
;;;; those first, and the same source then runs twice as many lanes.  Both
;;;; builds have to compute the same answer, so the value tests above run on
;;;; both; what these assert is the width that was actually chosen.

(defun vectorize-avx2-p ()
  (and (member :avx2 sb-c:*backend-subfeatures*) t))

(defun vectorize-disassembly-text (function)
  (string-downcase
   (with-output-to-string (stream) (disassemble function :stream stream))))

(defun vectorize-wide-p (function)
  "True when FUNCTION's disassembly uses a 256-bit register."
  (search "ymm" (vectorize-disassembly-text function)))

(defun vectorize-avx2-add-loop (element-type)
  (compile nil `(lambda (a b c n)
                  (declare (type (simple-array ,element-type (*)) a b c)
                           (type vectorize-index n)
                           (optimize speed))
                  (dotimes (i n)
                    (setf (aref c i) (+ (aref a i) (aref b i)))))))

(test-util:with-test (:name :vectorize-256-bit-lanes :skipped-on (not :x86-64))
  ;; float arithmetic and integer bitwise operations are exact at any safety, so
  ;; these loops vectorize whatever the policy is, and the only question left is
  ;; which width the build gives them.
  (flet ((add-loop (element-type)
           (compile nil `(lambda (a b c n)
                           (declare (type (simple-array ,element-type (*)) a b c)
                                    (type vectorize-index n)
                                    (optimize speed))
                           (dotimes (i n)
                             (setf (aref c i) (+ (aref a i) (aref b i)))))))
         (and-loop (element-type)
           (compile nil `(lambda (a b c n)
                           (declare (type (simple-array ,element-type (*)) a b c)
                                    (type vectorize-index n)
                                    (optimize speed))
                           (dotimes (i n)
                             (setf (aref c i) (logand (aref a i) (aref b i))))))))
    (dolist (function (list (add-loop 'single-float) (add-loop 'double-float)
                            (and-loop '(unsigned-byte 8)) (and-loop '(unsigned-byte 64))))
      (let ((text (vectorize-disassembly-text function))
            (wide (vectorize-wide-p function)))
        (if (vectorize-avx2-p)
            (assert wide () "loop did not take 256-bit lanes on an AVX2 build: ~A" text)
            (assert (not wide) () "loop took 256-bit lanes without AVX2: ~A" text))))))

;;; A byte or word source read into four double lanes has no 256-bit widen: the
;;; narrowing loads that would do it read eight bytes or sixteen, one element
;;; wider than the loop consumes, and on the last iteration that reads past the
;;; array.  So that loop keeps the 128-bit width, and the values still have to be
;;; right.  Reading a narrow source into single lanes does have one, and the pass
;;; must actually take it on a build that publishes it.
(test-util:with-test (:name :vectorize-256-bit-widen-falls-back :skipped-on (not :x86-64))
  (assert (not (vectorize-wide-p #'vectorize-u8-to-double)) ()
          "a byte-to-double widening loop took 256-bit lanes: ~A"
          (vectorize-disassembly-text #'vectorize-u8-to-double))
  (let ((values '(10 20 30 40 50 60 70 80 90 100)))
    (let ((a (make-array 10 :element-type '(unsigned-byte 8) :initial-contents values))
          (c (make-array 10 :element-type 'double-float :initial-element -1d0)))
      (vectorize-u8-to-double a c 10)
      (assert (equalp c (coerce values 'vector)) () "widened values: ~A" c))))

;;; The narrow-to-single widens at both widths: which one a build takes is the
;;; build's business, but a build that publishes the 256-bit primitive has to be
;;; the one that uses it, or the publication is dead weight nobody notices.
(test-util:with-test (:name :vectorize-256-bit-narrow-widen :skipped-on (not :x86-64))
  (flet ((mnemonics (function)
           (remove-if-not (lambda (m) (search m (vectorize-disassembly-text function)))
                          '("vpmovzxbd" "vpmovsxbd" "vpmovzxwd" "vpmovsxwd"
                            "pmovzxbd" "pmovsxbd" "pmovzxwd" "pmovsxwd" "cvtdq2ps"))))
    (dolist (function (list #'vectorize-u8-to-single #'vectorize-s8-to-single
                            #'vectorize-u16-to-single #'vectorize-s16-to-single))
      (assert (member "cvtdq2ps" (mnemonics function) :test #'string=) ()
              "narrow to single is not a packed widen: ~A"
              (vectorize-disassembly-text function))
      (if (vectorize-avx2-p)
          (assert (vectorize-wide-p function) ()
                  "a narrow-to-single widen stayed at 128 bits on an AVX2 build: ~A"
                  (vectorize-disassembly-text function))
          (assert (not (vectorize-wide-p function)) ()
                  "a narrow-to-single widen took 256-bit lanes without AVX2: ~A"
                  (vectorize-disassembly-text function))))
    (when (vectorize-avx2-p)
      (assert (member "vpmovzxbd" (mnemonics #'vectorize-u8-to-single) :test #'string=) ()
              "u8 to single is not the 256-bit narrowing load: ~A"
              (vectorize-disassembly-text #'vectorize-u8-to-single))
      (assert (member "vpmovsxwd" (mnemonics #'vectorize-s16-to-single) :test #'string=) ()
              "s16 to single is not the 256-bit narrowing load: ~A"
              (vectorize-disassembly-text #'vectorize-s16-to-single)))))

;;;; 32-bit clamping.  PMINSD, PMAXSD, PMINUD and PMAXUD are SSE4.1, VPMINSD and
;;;; friends are AVX2, and SSE2 has no packed 32-bit minimum at all - so a build
;;;; with neither instruction set declines the loop, as it always did.

(defun vectorize-clamp-s32 (a c n)
  (declare (type (simple-array (signed-byte 32) (*)) a c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (max -1000000 (min 1000000 (aref a i)))))
  c)

(defun vectorize-clamp-u32 (a c n)
  (declare (type (simple-array (unsigned-byte 32) (*)) a c) (type vectorize-index n)
           (optimize speed))
  (dotimes (i n) (setf (aref c i) (min (aref a i) 4000000000)))
  c)

(test-util:with-test (:name :vectorize-32-bit-clamp :skipped-on (not :x86-64))
  ;; min and max of values that fit the lane cannot overflow it, so these loops
  ;; vectorize at the default policy, unlike the multiply above
  (loop for n in *vectorize-lengths*
        for a = (vectorize-integer-array '(signed-byte 32)
                                         '(0 1 -1 999999 1000001 -2000000 2147483647 -2147483648) n)
        for c = (make-array n :element-type '(signed-byte 32) :initial-element 0)
        do (vectorize-clamp-s32 a c n)
           (dotimes (i n)
             (let ((want (max -1000000 (min 1000000 (aref a i)))))
               (unless (= (aref c i) want)
                 (error "~@<s32 clamp: element ~D of ~D is ~S, expected ~S~:@>"
                        i n (aref c i) want)))))
  (loop for n in *vectorize-lengths*
        for a = (vectorize-integer-array '(unsigned-byte 32)
                                         '(0 1 3999999999 4000000001 4294967295 7) n)
        for c = (make-array n :element-type '(unsigned-byte 32) :initial-element 0)
        do (vectorize-clamp-u32 a c n)
           (dotimes (i n)
             (let ((want (min (aref a i) 4000000000)))
               (unless (= (aref c i) want)
                 (error "~@<u32 clamp: element ~D of ~D is ~S, expected ~S~:@>"
                        i n (aref c i) want)))))
  (flet ((disassembly (function) (vectorize-disassembly-text function))
         (has (mnemonic function) (search mnemonic (vectorize-disassembly-text function))))
    (cond ((member :avx2 sb-c:*backend-subfeatures*)
           (assert (and (has "vpminud" #'vectorize-clamp-u32)
                        (has "vpminsd" #'vectorize-clamp-s32))
                   () "32-bit clamp did not take the 256-bit instructions: ~A"
                   (disassembly #'vectorize-clamp-s32))
           (dolist (function (list #'vectorize-clamp-u32 #'vectorize-clamp-s32))
             (assert (vectorize-wide-p function) ()
                     "32-bit clamp stayed at 128 bits on an AVX2 build: ~A"
                     (disassembly function))))
          ((member :sse4 sb-c:*backend-subfeatures*)
           (assert (and (has "pminud" #'vectorize-clamp-u32)
                        (has "pminsd" #'vectorize-clamp-s32))
                   () "32-bit clamp is not packed on an SSE4.1 build: ~A"
                   (disassembly #'vectorize-clamp-s32))))))

;;;; 32-bit multiplication.  PMULLD is SSE4.1 and VPMULLD is AVX2, so whether a
;;;; loop is packed at all is a property of the build: a build with neither
;;;; instruction declines it, and the values below are checked wherever it does not.

(defun vectorize-mul-u32 (a b c n)
  (declare (type (simple-array (unsigned-byte 32) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (* (aref a i) (aref b i))))
  c)

(defun vectorize-mul-s32 (a b c n)
  (declare (type (simple-array (signed-byte 32) (*)) a b c) (type vectorize-index n)
           (optimize speed (safety 0)))
  (dotimes (i n) (setf (aref c i) (* (aref a i) (aref b i))))
  c)

(test-util:with-test (:name :vectorize-32-bit-multiply :skipped-on (not :x86-64))
  ;; the lane holds the low 32 bits of the product, which is what the scalar loop
  ;; leaves once the store truncates - the same wrapping (safety 0) buys for
  ;; addition.  The low bits of a product do not depend on signedness, so the
  ;; unsigned expectation and the signed one are the same bits read differently.
  (flet ((low-32 (product)
           (logand product #xFFFFFFFF))
         (as-signed (bits)
           (if (logbitp 31 bits) (- bits #x100000000) bits)))
    (loop for n in *vectorize-lengths*
          for a = (vectorize-integer-array '(unsigned-byte 32)
                                           '(0 1 2 65536 4294967295 3 100000 7) n)
          for b = (vectorize-integer-array '(unsigned-byte 32)
                                           '(1 3 4294967295 65537 2 5 999 11) n 5)
          for c = (make-array n :element-type '(unsigned-byte 32) :initial-element 0)
          do (vectorize-mul-u32 a b c n)
             (dotimes (i n)
               (let ((want (low-32 (* (aref a i) (aref b i)))))
                 (unless (= (aref c i) want)
                   (error "~@<u32 multiply: element ~D of ~D is ~S, expected ~S~:@>"
                          i n (aref c i) want)))))
    (loop for n in *vectorize-lengths*
          for a = (vectorize-integer-array '(signed-byte 32)
                                           '(0 1 -1 2 -2 65536 -65536 2147483647) n)
          for b = (vectorize-integer-array '(signed-byte 32)
                                           '(1 -1 3 -3 65536 2 -2147483648 7) n 3)
          for c = (make-array n :element-type '(signed-byte 32) :initial-element 0)
          do (vectorize-mul-s32 a b c n)
             (dotimes (i n)
               (let ((want (as-signed (low-32 (* (aref a i) (aref b i))))))
                 (unless (= (aref c i) want)
                   (error "~@<s32 multiply: element ~D of ~D is ~S, expected ~S~:@>"
                          i n (aref c i) want))))))
  ;; packed when the build has an instruction for it, and the width picks itself:
  ;; vpmulld at 256 bits, pmulld at 128
  (flet ((disassembly (function) (vectorize-disassembly-text function)))
    (if (member :avx2 sb-c:*backend-subfeatures*)
        (progn
          (dolist (function (list #'vectorize-mul-u32 #'vectorize-mul-s32))
            (assert (search "vpmulld" (disassembly function)) ()
                    "32-bit multiply did not take the 256-bit instruction: ~A"
                    (disassembly function))
            (assert (vectorize-wide-p function) ()
                    "32-bit multiply stayed at 128 bits on an AVX2 build: ~A"
                    (disassembly function))))
        (when (member :sse4 sb-c:*backend-subfeatures*)
          (dolist (function (list #'vectorize-mul-u32 #'vectorize-mul-s32))
            (assert (search "pmulld" (disassembly function)) ()
                    "32-bit multiply is not packed on an SSE4.1 build: ~A"
                    (disassembly function)))))))

;;; Clamping is where the wider width earns its pay on image and audio data: a
;;; byte clamp is one VPMINUB per thirty-two elements.  The values are checked by
;;; the min/max tests above, which run at whichever width the build gives them;
;;; what these assert is the width, and that the instruction is the one the
;;; 256-bit table published rather than the 128-bit sequence.
(test-util:with-test (:name :vectorize-256-bit-min-max :skipped-on (not :x86-64))
  ;; "vpminub" contains "pminub", so one search covers both widths; where the
  ;; widths use different instructions, both are accepted
  (dolist (case (list (cons #'vectorize-u8-clamp '("pminub"))
                      (cons #'vectorize-u8-max-pair '("pmaxub"))
                      (cons #'vectorize-s8-min-pair '("pminsb" "pminub"))
                      (cons #'vectorize-u16-clamp '("pmaxuw" "psubusw"))
                      (cons #'vectorize-s16-clamp '("pminsw"))))
    (let ((function (car case))
          (mnemonics (cdr case)))
      (assert (some (lambda (m) (search m (vectorize-disassembly-text function))) mnemonics) ()
              "no ~{~a~^ or ~} in the min/max loop: ~A" mnemonics
              (vectorize-disassembly-text function))
      (if (vectorize-avx2-p)
          (assert (vectorize-wide-p function) ()
                  "a min/max loop stayed at 128 bits on an AVX2 build: ~A"
                  (vectorize-disassembly-text function))
          (assert (not (vectorize-wide-p function)) ()
                  "a min/max loop took 256-bit lanes without AVX2: ~A"
                  (vectorize-disassembly-text function))))))

;;; A single-float source read into double lanes is exact, since every
;;; single-float is a double-float.  The vop that does it once went missing while
;;; the widen section was regenerated: the defknown and the table entry outlived
;;; it, so the pass still emitted the call and the loop died at run time with an
;;; undefined function.  Compiling one of these loops is the check that catches
;;; that, and it is cheap.
(defun vectorize-single-to-double (in out n)
  (declare (type (simple-array single-float (*)) in)
           (type (simple-array double-float (*)) out)
           (type vectorize-index n) (optimize speed))
  (dotimes (i n) (setf (aref out i) (* 1.0d0 (aref in i))))
  out)

(test-util:with-test (:name :vectorize-single-to-double-widen :skipped-on (not :x86-64))
  (assert (search "cvtps2pd" (vectorize-disassembly-text #'vectorize-single-to-double)) ()
          "single to double is not a packed widen: ~A"
          (vectorize-disassembly-text #'vectorize-single-to-double))
  (if (vectorize-avx2-p)
      (assert (vectorize-wide-p #'vectorize-single-to-double) ()
              "single to double stayed at 128 bits on an AVX2 build: ~A"
              (vectorize-disassembly-text #'vectorize-single-to-double))
      (assert (not (vectorize-wide-p #'vectorize-single-to-double)) ()
              "single to double took 256-bit lanes without AVX2: ~A"
              (vectorize-disassembly-text #'vectorize-single-to-double)))
  (loop for n in *vectorize-lengths*
        for in = (vectorize-integer-array 'single-float '(-1.5 0.0 0.25 3.0 1.0e30) n)
        for out = (make-array n :element-type 'double-float :initial-element -1d0)
        do (vectorize-single-to-double in out n)
           (dotimes (i n)
             (unless (= (aref out i) (aref in i))
               (error "~@<single to double: element ~D of ~D is ~S, expected ~S~:@>"
                      i n (aref out i) (aref in i))))))

(test-util:with-test (:name :vectorize-widen-instruction-set :skipped-on (not :x86-64))
  (flet ((mnemonics (function)
           (let ((text (string-downcase (with-output-to-string (stream)
                                          (disassemble function :stream stream)))))
             (remove-if-not (lambda (m) (search m text))
                            '("pmovzxbd" "pmovsxbd" "pmovzxwd" "pmovsxwd"
                              "punpcklbw" "punpcklwd" "movq")))))
    (if (member :sse4 sb-c:*backend-subfeatures*)
        (progn
          (assert (member "pmovzxbd" (mnemonics #'vectorize-u8-to-single) :test #'string=) ()
                  "u8 to single is not the SSE4.1 load: ~a" (mnemonics #'vectorize-u8-to-single))
          (assert (member "pmovsxbd" (mnemonics #'vectorize-s8-to-single) :test #'string=) ()
                  "s8 to single is not the SSE4.1 load: ~a" (mnemonics #'vectorize-s8-to-single))
          (assert (member "pmovzxwd" (mnemonics #'vectorize-u16-to-single) :test #'string=) ()
                  "u16 to single is not the SSE4.1 load: ~a" (mnemonics #'vectorize-u16-to-single))
          (assert (member "pmovsxwd" (mnemonics #'vectorize-s16-to-single) :test #'string=) ()
                  "s16 to single is not the SSE4.1 load: ~a" (mnemonics #'vectorize-s16-to-single)))
        (assert (member "punpcklbw" (mnemonics #'vectorize-u8-to-single) :test #'string=) ()
                "u8 to single is not the unpack sequence: ~a" (mnemonics #'vectorize-u8-to-single)))))


;;; A loop over an array bound by LET does not see the array's type, because the
;;; leaf's type is only propagated to it after the pass has run.  A binding whose
;;; initializer proves the type gets a declaration added instead, which is a source
;;; rewrite like the loop expansion, so these tests pin down that the loops now
;;; vectorize and that the declaration is what the initializer already guaranteed.

(defun vectorize-let-destination (b n)
  (declare (type (simple-array single-float (*)) b) (type vectorize-index n)
           (optimize speed))
  (let ((a (make-array n :element-type 'single-float :initial-element 0.0)))
    (dotimes (i n) (setf (aref a i) (* 2.0 (aref b i))))
    a))

(defun vectorize-let-source (out n)
  (declare (type (simple-array single-float (*)) out) (type vectorize-index n)
           (optimize speed))
  (let ((a (make-array n :element-type 'single-float :initial-element 1.5)))
    (dotimes (i n) (setf (aref out i) (+ 1.0 (aref a i))))
    out))

(defun vectorize-let-the-binding (b out n)
  (declare (type (simple-array single-float (*)) b out) (type vectorize-index n)
           (optimize speed))
  (let ((a (the (simple-array single-float (*)) (copy-seq b))))
    (dotimes (i n) (setf (aref out i) (* 3.0 (aref a i))))
    out))

;;; an adjustable array is not simple, so its initializer proves nothing and the
;;; loop must stay scalar and still be right
(defun vectorize-let-adjustable (b n)
  (declare (type (simple-array single-float (*)) b) (type vectorize-index n)
           (optimize speed))
  (let ((a (make-array n :element-type 'single-float :adjustable t
                         :initial-element 0.0)))
    (dotimes (i n) (setf (aref a i) (* 2.0 (aref b i))))
    a))

(test-util:with-test (:name :vectorize-let-bound-arrays :skipped-on (not :x86-64))
  (let ((b (make-array 37 :element-type 'single-float)))
    (dotimes (i 37) (setf (aref b i) (coerce i 'single-float)))
    (let ((destination (vectorize-let-destination b 37)))
      (dotimes (i 37)
        (unless (= (aref destination i) (coerce (* 2 i) 'single-float))
          (error "~@<let-bound destination: element ~D is ~S, expected ~S~:@>"
                 i (aref destination i) (coerce (* 2 i) 'single-float)))))
    (let ((source (vectorize-let-source (make-array 37 :element-type 'single-float) 37)))
      (dotimes (i 37)
        (unless (= (aref source i) 2.5)
          (error "~@<let-bound source: element ~D is ~S, expected 2.5~:@>"
                 i (aref source i)))))
    (let ((out (make-array 37 :element-type 'single-float)))
      (vectorize-let-the-binding b out 37)
      (dotimes (i 37)
        (unless (= (aref out i) (coerce (* 3 i) 'single-float))
          (error "~@<let-bound the cast: element ~D is ~S, expected ~S~:@>"
                 i (aref out i) (coerce (* 3 i) 'single-float)))))
    (let ((adjustable (vectorize-let-adjustable b 37)))
      (dotimes (i 37)
        (unless (= (aref adjustable i) (coerce (* 2 i) 'single-float))
          (error "~@<let-bound adjustable: element ~D is ~S, expected ~S~:@>"
                 i (aref adjustable i) (coerce (* 2 i) 'single-float))))))
  (flet ((text (name)
           (string-downcase (with-output-to-string (stream)
                              (disassemble name :stream stream)))))
    (dolist (name '(vectorize-let-destination vectorize-let-source
                    vectorize-let-the-binding))
      (assert (some (lambda (mnemonic) (search mnemonic (text name)))
                    *vectorize-packed-mnemonics*)
              () "~A was not vectorized:~%~A" name (text name)))
    (assert (notany (lambda (mnemonic) (search mnemonic (text 'vectorize-let-adjustable)))
                    *vectorize-packed-mnemonics*)
            () "~A should have stayed scalar:~%~A"
            'vectorize-let-adjustable (text 'vectorize-let-adjustable))))
