;;;; Timing of automatic vectorization: the same loop compiled with and without
;;;; the vectorizer.  Run as
;;;;
;;;;   src/runtime/sbcl --core output/sbcl.core --load benchmarks/auto-vectorize.lisp
;;;;
;;;; Sizes are chosen so that one is small enough to stay in a core's cache, where
;;;; the vector loop is limited by instruction throughput, and one is large enough
;;;; to stream from memory, where both loops are limited by bandwidth.
;;;;
;;;; This software is in the public domain and is provided with absolutely no
;;;; warranty. See the COPYING and CREDITS files for more information.

(defparameter *sizes* (list (* 8 1024) (* 1024 1024)))
;;; THE INTERNAL TIMER IS MILLISECOND-RESOLUTION, so time enough work per size
;;; that each measurement is tens of milliseconds long.
(defun element-size (element-type)
  "Bytes per element.  The two arrays are read and the third written, so the
traffic per element is twice this."
  (cond ((equal element-type 'single-float) 4)
        ((equal element-type 'double-float) 8)
        ((equal element-type '(unsigned-byte 16)) 2)
        ((equal element-type '(unsigned-byte 32)) 4)
        ((equal element-type '(unsigned-byte 64)) 8)
        (t (error "no size for ~a" element-type))))

(defun repetitions-for (size)
  (max 20 (ceiling 1000000000 size)))

(defmacro define-pair (name element-type op &key safety-zero)
  "Define NAME (vectorized) and NAME/SCALAR (the same loop, vectorizer off).

Integer lanes are only used at (safety 0) - a packed lane wraps where a checked
store would signal - so the integer pairs ask for it; float lanes do not care."
  (flet ((definition (name)
           `(defun ,name (a b c n)
              (declare (type (simple-array ,element-type (*)) a b c)
                       (type sb-int:index n)
                       (optimize speed ,@(when safety-zero '((safety 0)))))
              (dotimes (i n)
                (declare (type sb-int:index i))
                (setf (aref c i) (,op (aref a i) (aref b i))))
              c)))
    `(progn
       ,(definition name)
       (let ((sb-c::*auto-vectorize* nil))
         (eval ',(definition (intern (concatenate 'string (symbol-name name) "/SCALAR"))))))))

(define-pair add-single single-float +)
(define-pair add-double double-float +)
(define-pair mul-single single-float *)
(define-pair mul-double double-float *)

;;; Integer lanes: four 32-bit or two 64-bit at a time.
(define-pair add-u32 (unsigned-byte 32) + :safety-zero t)
(define-pair add-u64 (unsigned-byte 64) + :safety-zero t)
(define-pair add-u16 (unsigned-byte 16) + :safety-zero t)

(defun make-random-array (element-type n)
  (let ((v (make-array n :element-type element-type)))
    (dotimes (i n v)
      (setf (aref v i)
            (cond ((eq element-type 'single-float) (coerce (- (random 2.0) 1.0) 'single-float))
                  ((eq element-type 'double-float) (- (random 2d0) 1d0))
                  (t (random 1000)))))))

(defun time-loop (function element-type size)
  "SECONDS and GB/s for REPETITIONS-FOR calls of FUNCTION over SIZE elements."
  (let ((a (make-random-array element-type size))
        (b (make-random-array element-type size))
        (c (make-array size :element-type element-type)))
    (funcall function a b c size)
    (let ((start (get-internal-real-time))
          (repetitions (repetitions-for size))
          (elapsed 0))
      (dotimes (i repetitions)
        (funcall function a b c size)
        (let ((now (get-internal-real-time)))
          (incf elapsed (- now start))
          (setf start now)))
      (let* ((seconds (/ elapsed internal-time-units-per-second))
             (bytes (* 2 size repetitions (element-size element-type))))
        (values seconds (/ bytes seconds 1d9))))))

(dolist (size *sizes*)
  (format t "~&~a elements, ~a reps~%" size (repetitions-for size))
  (format t "~30a ~10a ~10a ~10a~%" "loop" "seconds" "GB/s" "speedup")
  (dolist (pair '((add-single single-float) (add-double double-float)
                  (mul-single single-float) (mul-double double-float)
                  (add-u32 (unsigned-byte 32)) (add-u64 (unsigned-byte 64))
                  (add-u16 (unsigned-byte 16))))
    (destructuring-bind (name element-type) pair
      (let ((scalar-name (intern (concatenate 'string (symbol-name name) "/SCALAR"))))
        (multiple-value-bind (scalar-seconds scalar-rate) (time-loop scalar-name element-type size)
          (multiple-value-bind (vector-seconds vector-rate) (time-loop name element-type size)
            (format t "~30a ~10,4f ~10,2f ~10,2f  (scalar ~,2f GB/s)~%"
                    (string-downcase name) vector-seconds vector-rate
                    (if (zerop vector-seconds) 0 (/ scalar-seconds vector-seconds))
                    scalar-rate)))))))
