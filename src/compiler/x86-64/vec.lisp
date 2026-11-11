;;;; Packed single/double float array access and arithmetic VOPs

;;;; This software is part of the SBCL system. See the README file for
;;;; more information.
;;;;
;;;; This software is derived from the CMUCL system, which was
;;;; written at Carnegie Mellon University and released into the
;;;; public domain. The software is in the public domain and is
;;;; provided with absolutely no warranty. See the COPYING and CREDITS
;;;; files for more information.

(in-package "SB-VM")

;;;; Function signatures.
;;;
;;; The arrays are spelled (SIMPLE-ARRAY <float> (*)) because
;;; SIMPLE-ARRAY-SINGLE-FLOAT and SIMPLE-ARRAY-DOUBLE-FLOAT are
;;; primitive type names (primtype.lisp), not Lisp type specifiers.
;;;
;;; A load is FLUSHABLE but deliberately neither MOVABLE nor FOLDABLE,
;;; and a store is neither: 128 bits at a time must not be reordered
;;; across a store into the same array, which is exactly what a loop
;;; that reads what it just wrote would do.

(defknown %vec-load-single ((simple-array single-float (*)) index)
    (simd-pack single-float)
  (flushable))
(defknown %vec-load-double ((simple-array double-float (*)) index)
    (simd-pack double-float)
  (flushable))
(defknown %vec-store-single
    ((simd-pack single-float) (simple-array single-float (*)) index)
    (values)
  ())
(defknown %vec-store-double
    ((simd-pack double-float) (simple-array double-float (*)) index)
    (values)
  ())

(defknown (%vec-add-single %vec-sub-single
           %vec-mul-single %vec-div-single)
    ((simd-pack single-float) (simd-pack single-float))
    (simd-pack single-float)
  (flushable movable))
(defknown (%vec-add-double %vec-sub-double
           %vec-mul-double %vec-div-double)
    ((simd-pack double-float) (simd-pack double-float))
    (simd-pack double-float)
  (flushable movable))

(defknown %vec-splat-single (single-float)
    (simd-pack single-float)
  (flushable movable))
(defknown %vec-splat-double (double-float)
    (simd-pack double-float)
  (flushable movable))

;;;; Loading and storing 128 bits at a time in a simple float array.
;;;
;;; The effective address is the one DEFINE-FULL-SETTER+ADDEND uses in
;;; array.lisp, with the element size in bytes in place of N-WORD-BYTES:
;;;
;;;   (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
;;;       vector index (index-scale <element-size> index))
;;;
;;; INDEX is restricted to ANY-REG and declared TAGGED-NUM so that
;;; INDEX-SCALE (macros.lisp) divides the element size by the fixnum
;;; tag bits; an IMMEDIATE index is not accepted, because the scale
;;; would then have to be the element size itself.

(macrolet ((define-vec-load (name translate element-type result-type
                                 element-size sc instruction)
             `(define-vop (,name)
                (:translate ,translate)
                (:args (vector :scs (descriptor-reg))
                       (index :scs (any-reg)))
                (:arg-types ,element-type tagged-num)
                (:results (result :scs (,sc)))
                (:result-types ,result-type)
                (:generator 3
                  (inst ,instruction result
                        (ea (- (* vector-data-offset n-word-bytes)
                               other-pointer-lowtag)
                            vector index
                            (index-scale ,element-size index))))))
           (define-vec-store (name translate result-type element-type
                                   element-size sc instruction)
             ;; VALUE is first in both the Lisp arglist and :ARGS.
             `(define-vop (,name)
                (:translate ,translate)
                (:args (value :scs (,sc))
                       (vector :scs (descriptor-reg))
                       (index :scs (any-reg)))
                (:arg-types ,result-type ,element-type tagged-num)
                (:generator 3
                  (inst ,instruction
                        (ea (- (* vector-data-offset n-word-bytes)
                               other-pointer-lowtag)
                            vector index
                            (index-scale ,element-size index))
                        value)))))
  (define-vec-load %vec-load-single %vec-load-single
    simple-array-single-float simd-pack-single 4 single-sse-reg movups)
  (define-vec-load %vec-load-double %vec-load-double
    simple-array-double-float simd-pack-double 8 double-sse-reg movupd)
  (define-vec-store %vec-store-single %vec-store-single
    simd-pack-single simple-array-single-float 4 single-sse-reg movups)
  (define-vec-store %vec-store-double %vec-store-double
    simd-pack-double simple-array-double-float 8 double-sse-reg movupd))

;;;; Packed arithmetic.
;;;
;;; These are two-operand instructions: the destination is also the
;;; first source. RESULT is initialized from A with an explicit MOVE,
;;; modelled on DEF-REAL-COMPLEX-OP in float.lisp, because :TARGET is
;;; only a packing preference (FIND-OK-TARGET-OFFSET in pack.lisp falls
;;; back on SELECT-LOCATION when the preference can't be honored, and
;;; the operand's SC restriction is satisfied by any XMM register), so
;;; relying on it would compute RESULT from whatever stale value it
;;; holds. The MOVE emits nothing when the packer does honor it.

(macrolet ((define-vec-arith (name translate type sc instruction)
             `(define-vop (,name)
                (:translate ,translate)
                (:args (a :scs (,sc) :target result)
                       (b :scs (,sc)))
                (:arg-types ,type ,type)
                (:results (result :scs (,sc)))
                (:result-types ,type)
                (:generator 2
                  (move result a)
                  (inst ,instruction result b)))))
  (define-vec-arith %vec-add-single %vec-add-single
    simd-pack-single single-sse-reg addps)
  (define-vec-arith %vec-sub-single %vec-sub-single
    simd-pack-single single-sse-reg subps)
  (define-vec-arith %vec-mul-single %vec-mul-single
    simd-pack-single single-sse-reg mulps)
  (define-vec-arith %vec-div-single %vec-div-single
    simd-pack-single single-sse-reg divps)
  (define-vec-arith %vec-add-double %vec-add-double
    simd-pack-double double-sse-reg addpd)
  (define-vec-arith %vec-sub-double %vec-sub-double
    simd-pack-double double-sse-reg subpd)
  (define-vec-arith %vec-mul-double %vec-mul-double
    simd-pack-double double-sse-reg mulpd)
  (define-vec-arith %vec-div-double %vec-div-double
    simd-pack-double double-sse-reg divpd))

;;;; Splatting one float across every lane.
;;;
;;; Modelled on %MAKE-SIMD-PACK-SINGLE/-DOUBLE (simd-pack.lisp): the arg
;;; is a scalar float SC targeted at the result, MOVEd once into it, and
;;; then the low lane alone is selected by the shuffle.

(define-vop (%vec-splat-single)
  (:translate %vec-splat-single)
  (:args (x :scs (single-reg) :target result))
  (:arg-types single-float)
  (:results (result :scs (single-sse-reg) :from (:argument 0)))
  (:result-types simd-pack-single)
  (:generator 2
    (move result x)
    (inst shufps result result #b00000000)))

(define-vop (%vec-splat-double)
  (:translate %vec-splat-double)
  (:args (x :scs (double-reg) :target result))
  (:arg-types double-float)
  (:results (result :scs (double-sse-reg) :from (:argument 0)))
  (:result-types simd-pack-double)
  (:generator 2
    (move result x)
    (inst unpcklpd result result)))

;;;; Hand the primitives to the loop vectorizer in SB-C, which cannot name this
;;;; package itself.  See the comment on SB-C::*VECTOR-PRIMITIVES*.

(macrolet ((publish (base element-type name)
             `(push (cons (list ,base ',element-type) ',name) primitives)))
  (let ((primitives nil))
    (publish "LOAD" single-float %vec-load-single)
    (publish "LOAD" double-float %vec-load-double)
    (publish "STORE" single-float %vec-store-single)
    (publish "STORE" double-float %vec-store-double)
    (publish "SPLAT" single-float %vec-splat-single)
    (publish "SPLAT" double-float %vec-splat-double)
    (publish "ADD" single-float %vec-add-single)
    (publish "ADD" double-float %vec-add-double)
    (publish "SUB" single-float %vec-sub-single)
    (publish "SUB" double-float %vec-sub-double)
    (publish "MUL" single-float %vec-mul-single)
    (publish "MUL" double-float %vec-mul-double)
    (publish "DIV" single-float %vec-div-single)
    (publish "DIV" double-float %vec-div-double)
    (setf sb-c::*vector-primitives* (nreverse primitives))))
