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
;;; N, and - on x86-64 - the packed loop really being there.

;;; The vectorizer targets x86-64 and is part of the simd-pack support.
#-x86-64 (invoke-restart 'run-tests::skip-file)

;;; INDEX, the type of a bound into a vector: (INTEGER 0 (ARRAY-DIMENSION-LIMIT)).
;;; It is spelled out rather than named because the name and the package that
;;; owns it - SB-INT:INDEX or SB-EXT:INDEX depending on the build - are not
;;; something a test should depend on.
(deftype vectorize-index () `(integer 0 (,array-dimension-limit)))

;;;; Inputs shared by every test.

;;; The lengths every correctness test runs over.  0 runs no iterations at
;;; all, 1 through 9 cross the hand-off from the packed loop to the scalar tail
;;; (4 is a whole vector of single-floats, 8 of double-floats), 16 and 17
;;; straddle a second pair of vectors, and 1000 is long enough that a lane
;;; shift cannot hide in a short loop.
(defparameter *vectorize-lengths* '(0 1 2 3 4 5 7 8 9 16 17 1000))

;;; The input values deliberately are not all equal: positives, negatives,
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

(with-test (:name :vectorize-single-add)
  (vectorize-check-elementwise #'vectorize-single-add #'+
                               'single-float
                               *vectorize-single-values*
                               *vectorize-single-values*
                               'vectorize-single-add))

(with-test (:name :vectorize-single-sub)
  (vectorize-check-elementwise #'vectorize-single-sub #'-
                               'single-float
                               *vectorize-single-values*
                               *vectorize-single-values*
                               'vectorize-single-sub))

(with-test (:name :vectorize-single-mul)
  (vectorize-check-elementwise #'vectorize-single-mul #'*
                               'single-float
                               *vectorize-single-values*
                               *vectorize-single-values*
                               'vectorize-single-mul))

(with-test (:name :vectorize-single-div)
  (vectorize-check-elementwise #'vectorize-single-div #'/
                               'single-float
                               *vectorize-single-values*
                               *vectorize-single-divisors*
                               'vectorize-single-div))

(with-test (:name :vectorize-single-negate)
  (vectorize-check-unary #'vectorize-single-negate #'-
                         'single-float
                         *vectorize-single-values*
                         'vectorize-single-negate))

;;; The loops below multiply by a literal; the references multiply by the
;;; matching special variable, whose value is that same literal.
(with-test (:name :vectorize-single-scale)
  (vectorize-check-unary #'vectorize-single-scale
                         (lambda (x) (* x *vectorize-single-scale*))
                         'single-float
                         *vectorize-single-values*
                         'vectorize-single-scale))

(with-test (:name :vectorize-single-add-scale)
  (vectorize-check-elementwise #'vectorize-single-add-scale
                               (lambda (x y) (* (+ x y) *vectorize-single-add-scale*))
                               'single-float
                               *vectorize-single-values*
                               *vectorize-single-values*
                               'vectorize-single-add-scale))

(with-test (:name :vectorize-double-add)
  (vectorize-check-elementwise #'vectorize-double-add #'+
                               'double-float
                               *vectorize-double-values*
                               *vectorize-double-values*
                               'vectorize-double-add))

(with-test (:name :vectorize-double-sub)
  (vectorize-check-elementwise #'vectorize-double-sub #'-
                               'double-float
                               *vectorize-double-values*
                               *vectorize-double-values*
                               'vectorize-double-sub))

(with-test (:name :vectorize-double-mul)
  (vectorize-check-elementwise #'vectorize-double-mul #'*
                               'double-float
                               *vectorize-double-values*
                               *vectorize-double-values*
                               'vectorize-double-mul))

(with-test (:name :vectorize-double-div)
  (vectorize-check-elementwise #'vectorize-double-div #'/
                               'double-float
                               *vectorize-double-values*
                               *vectorize-double-divisors*
                               'vectorize-double-div))

(with-test (:name :vectorize-double-negate)
  (vectorize-check-unary #'vectorize-double-negate #'-
                         'double-float
                         *vectorize-double-values*
                         'vectorize-double-negate))

(with-test (:name :vectorize-double-scale)
  (vectorize-check-unary #'vectorize-double-scale
                         (lambda (x) (* x *vectorize-double-scale*))
                         'double-float
                         *vectorize-double-values*
                         'vectorize-double-scale))

(with-test (:name :vectorize-double-add-scale)
  (vectorize-check-elementwise #'vectorize-double-add-scale
                               (lambda (x y) (* (+ x y) *vectorize-double-add-scale*))
                               'double-float
                               *vectorize-double-values*
                               *vectorize-double-values*
                               'vectorize-double-add-scale))

;;;; Aliasing: C may be the same array as A, and A may be the same as B.

(with-test (:name :vectorize-single-aliasing)
  (vectorize-check-aliasing #'vectorize-single-add 'single-float
                            *vectorize-single-values*
                            *vectorize-single-values*
                            'vectorize-single-aliasing))

(with-test (:name :vectorize-double-aliasing)
  (vectorize-check-aliasing #'vectorize-double-add 'double-float
                            *vectorize-double-values*
                            *vectorize-double-values*
                            'vectorize-double-aliasing))

;;;; N larger than the arrays: the fallback scalar loop signals an index error
;;;; rather than reading or writing out of bounds.  SPEED alone does not turn
;;;; bounds checks off - SAFETY is still 1 - which is what makes the error
;;;; observable.

(with-test (:name :vectorize-single-n-exceeds-length)
  (loop for (size . n) in '((0 . 1) (1 . 2) (1 . 5) (3 . 4) (4 . 5)
                            (8 . 9) (8 . 17) (1000 . 1001))
        do (let ((a (vectorize-array 'single-float *vectorize-single-values* size))
                 (b (vectorize-array 'single-float *vectorize-single-divisors* size 3))
                 (c (vectorize-sentinel-array 'single-float size)))
             (assert-error (funcall #'vectorize-single-add a b c n)
                           sb-int:invalid-array-index-error))))

(with-test (:name :vectorize-double-n-exceeds-length)
  (loop for (size . n) in '((0 . 1) (1 . 2) (1 . 5) (3 . 4) (4 . 5)
                            (8 . 9) (8 . 17) (1000 . 1001))
        do (let ((a (vectorize-array 'double-float *vectorize-double-values* size))
                 (b (vectorize-array 'double-float *vectorize-double-divisors* size 3))
                 (c (vectorize-sentinel-array 'double-float size)))
             (assert-error (funcall #'vectorize-double-add a b c n)
                           sb-int:invalid-array-index-error))))

;;;; The same loop without declarations cannot be vectorized, and must still
;;;; compute the right values.

(with-test (:name :vectorize-single-no-declarations)
  (vectorize-check-elementwise #'vectorize-single-add-undeclared #'+
                               'single-float
                               *vectorize-single-values*
                               *vectorize-single-values*
                               'vectorize-single-add-undeclared))

(with-test (:name :vectorize-double-no-declarations)
  (vectorize-check-elementwise #'vectorize-double-add-undeclared #'+
                               'double-float
                               *vectorize-double-values*
                               *vectorize-double-values*
                               'vectorize-double-add-undeclared))

;;;; The loop was actually vectorized.  Two functions that differ only in
;;;; whether the array types are declared: the declared one must contain a
;;;; packed SIMD mnemonic, the undeclared one must not, whatever instructions
;;;; the vectorizer settles on in detail.

(with-test (:name :vectorize-disassembly)
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
