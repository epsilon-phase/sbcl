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

;;;; Integer lanes: 32 and 64 bits wide, signed and unsigned.
;;;;
;;;; The four shapes of the float sections above, over the packs
;;;; SIMD-PACK-UB32, SIMD-PACK-SB32, SIMD-PACK-UB64 and SIMD-PACK-SB64,
;;;; which all live in INT-SSE-REG.  As in the signatures at the top, the
;;;; array type is spelled (SIMPLE-ARRAY <element> (*)) where a Lisp type
;;;; specifier is wanted and by its primitive type name (vm-array.lisp) in
;;;; :ARG-TYPES.
;;;;
;;;; There is no integer MUL or DIV here, and that is not an omission to be
;;;; filled in later: PMULLD is SSE4.1 and PMULUDQ reads only the low half
;;;; of each 64-bit lane, so neither computes the element type the array
;;;; holds.  ADD, SUB and the three bitwise operations are baseline SSE2 at
;;;; both widths.

(defknown %vec-load-ub32 ((simple-array (unsigned-byte 32) (*)) index)
    (simd-pack (unsigned-byte 32))
  (flushable))
(defknown %vec-load-sb32 ((simple-array (signed-byte 32) (*)) index)
    (simd-pack (signed-byte 32))
  (flushable))
(defknown %vec-load-ub64 ((simple-array (unsigned-byte 64) (*)) index)
    (simd-pack (unsigned-byte 64))
  (flushable))
(defknown %vec-load-sb64 ((simple-array (signed-byte 64) (*)) index)
    (simd-pack (signed-byte 64))
  (flushable))
(defknown %vec-store-ub32
    ((simd-pack (unsigned-byte 32)) (simple-array (unsigned-byte 32) (*)) index)
    (values)
  ())
(defknown %vec-store-sb32
    ((simd-pack (signed-byte 32)) (simple-array (signed-byte 32) (*)) index)
    (values)
  ())
(defknown %vec-store-ub64
    ((simd-pack (unsigned-byte 64)) (simple-array (unsigned-byte 64) (*)) index)
    (values)
  ())
(defknown %vec-store-sb64
    ((simd-pack (signed-byte 64)) (simple-array (signed-byte 64) (*)) index)
    (values)
  ())

(defknown (%vec-add-ub32 %vec-sub-ub32
           %vec-and-ub32 %vec-or-ub32 %vec-xor-ub32)
    ((simd-pack (unsigned-byte 32)) (simd-pack (unsigned-byte 32)))
    (simd-pack (unsigned-byte 32))
  (flushable movable))
(defknown (%vec-add-sb32 %vec-sub-sb32
           %vec-and-sb32 %vec-or-sb32 %vec-xor-sb32)
    ((simd-pack (signed-byte 32)) (simd-pack (signed-byte 32)))
    (simd-pack (signed-byte 32))
  (flushable movable))
(defknown (%vec-add-ub64 %vec-sub-ub64
           %vec-and-ub64 %vec-or-ub64 %vec-xor-ub64)
    ((simd-pack (unsigned-byte 64)) (simd-pack (unsigned-byte 64)))
    (simd-pack (unsigned-byte 64))
  (flushable movable))
(defknown (%vec-add-sb64 %vec-sub-sb64
           %vec-and-sb64 %vec-or-sb64 %vec-xor-sb64)
    ((simd-pack (signed-byte 64)) (simd-pack (signed-byte 64)))
    (simd-pack (signed-byte 64))
  (flushable movable))

(defknown %vec-splat-ub32 ((unsigned-byte 32))
    (simd-pack (unsigned-byte 32))
  (flushable movable))
(defknown %vec-splat-sb32 ((signed-byte 32))
    (simd-pack (signed-byte 32))
  (flushable movable))
(defknown %vec-splat-ub64 ((unsigned-byte 64))
    (simd-pack (unsigned-byte 64))
  (flushable movable))
(defknown %vec-splat-sb64 ((signed-byte 64))
    (simd-pack (signed-byte 64))
  (flushable movable))

;;;; Loading and storing 128 bits at a time in a simple integer array.
;;;
;;; The effective address is the one the float section above uses, with the
;;; element size -- 4 or 8 bytes -- in place of N-WORD-BYTES, and INDEX is
;;; restricted and declared exactly as there.
;;;
;;; DEFINE-VEC-LOAD and DEFINE-VEC-STORE are lexically scoped to the float
;;; section above, so the same two shapes are spelled out here rather than
;;; the working float section being rewritten: MOVDQU needs no element type
;;; of its own, and only the element size and the array type vary.

(macrolet ((define-vec-load (name element-type result-type element-size)
             `(define-vop (,name)
                (:translate ,name)
                (:args (vector :scs (descriptor-reg))
                       (index :scs (any-reg)))
                (:arg-types ,element-type tagged-num)
                (:results (result :scs (int-sse-reg)))
                (:result-types ,result-type)
                (:generator 3
                  (inst movdqu result
                        (ea (- (* vector-data-offset n-word-bytes)
                               other-pointer-lowtag)
                            vector index
                            (index-scale ,element-size index))))))
           (define-vec-store (name element-type result-type element-size)
             ;; VALUE is first in both the Lisp arglist and :ARGS.
             `(define-vop (,name)
                (:translate ,name)
                (:args (value :scs (int-sse-reg))
                       (vector :scs (descriptor-reg))
                       (index :scs (any-reg)))
                (:arg-types ,result-type ,element-type tagged-num)
                (:generator 3
                  (inst movdqu
                        (ea (- (* vector-data-offset n-word-bytes)
                               other-pointer-lowtag)
                            vector index
                            (index-scale ,element-size index))
                        value)))))
  (define-vec-load %vec-load-ub32 simple-array-unsigned-byte-32 simd-pack-ub32 4)
  (define-vec-load %vec-load-sb32 simple-array-signed-byte-32 simd-pack-sb32 4)
  (define-vec-load %vec-load-ub64 simple-array-unsigned-byte-64 simd-pack-ub64 8)
  (define-vec-load %vec-load-sb64 simple-array-signed-byte-64 simd-pack-sb64 8)
  (define-vec-store %vec-store-ub32 simple-array-unsigned-byte-32 simd-pack-ub32 4)
  (define-vec-store %vec-store-sb32 simple-array-signed-byte-32 simd-pack-sb32 4)
  (define-vec-store %vec-store-ub64 simple-array-unsigned-byte-64 simd-pack-ub64 8)
  (define-vec-store %vec-store-sb64 simple-array-signed-byte-64 simd-pack-sb64 8))

;;;; Packed integer arithmetic and bitwise operations.
;;;
;;; Same reasoning as the float arithmetic above: the destination is also the
;;; first source, so RESULT is MOVEd from A first and :TARGET is only a
;;; packing preference.  The instructions wrap on overflow, which is what
;;; the (unsigned-byte N) and (signed-byte N) element types mean here, and
;;; PADDD/PSUBD/PADDQ/PSUBQ/PAND/POR/PXOR all exist at both widths.  The
;;; operands are packs rather than scalars, so the tagged-versus-untagged
;;; question of the splats below does not arise.

(macrolet ((define-vec-arith (name type instruction)
             `(define-vop (,name)
                (:translate ,name)
                (:args (a :scs (int-sse-reg) :target result)
                       (b :scs (int-sse-reg)))
                (:arg-types ,type ,type)
                (:results (result :scs (int-sse-reg)))
                (:result-types ,type)
                (:generator 2
                  (move result a)
                  (inst ,instruction result b)))))
  (define-vec-arith %vec-add-ub32 simd-pack-ub32 paddd)
  (define-vec-arith %vec-sub-ub32 simd-pack-ub32 psubd)
  (define-vec-arith %vec-and-ub32 simd-pack-ub32 pand)
  (define-vec-arith %vec-or-ub32 simd-pack-ub32 por)
  (define-vec-arith %vec-xor-ub32 simd-pack-ub32 pxor)
  (define-vec-arith %vec-add-sb32 simd-pack-sb32 paddd)
  (define-vec-arith %vec-sub-sb32 simd-pack-sb32 psubd)
  (define-vec-arith %vec-and-sb32 simd-pack-sb32 pand)
  (define-vec-arith %vec-or-sb32 simd-pack-sb32 por)
  (define-vec-arith %vec-xor-sb32 simd-pack-sb32 pxor)
  (define-vec-arith %vec-add-ub64 simd-pack-ub64 paddq)
  (define-vec-arith %vec-sub-ub64 simd-pack-ub64 psubq)
  (define-vec-arith %vec-and-ub64 simd-pack-ub64 pand)
  (define-vec-arith %vec-or-ub64 simd-pack-ub64 por)
  (define-vec-arith %vec-xor-ub64 simd-pack-ub64 pxor)
  (define-vec-arith %vec-add-sb64 simd-pack-sb64 paddq)
  (define-vec-arith %vec-sub-sb64 simd-pack-sb64 psubq)
  (define-vec-arith %vec-and-sb64 simd-pack-sb64 pand)
  (define-vec-arith %vec-or-sb64 simd-pack-sb64 por)
  (define-vec-arith %vec-xor-sb64 simd-pack-sb64 pxor))

;;;; Splatting one integer across every lane.
;;;
;;; The scalar arrives in a general-purpose register, not in the XMM register
;;; the result wants, and there is no move VOP between UNSIGNED-REG or
;;; SIGNED-REG and INT-SSE-REG (simd-pack.lisp moves between them with an
;;; explicit MOVD/MOVQ inside a generator, as %MAKE-SIMD-PACK-UB64 does), so
;;; the transfer is an instruction rather than a MOVE: MOVD for 32-bit lanes,
;;; MOVQ for 64-bit lanes.
;;;
;;; UNSIGNED-NUM and SIGNED-NUM are aliases over several primitive types
;;; (primtype.lisp), and on 64-bit x86-64 two of those members are the tagged
;;; fixnum types: (UNSIGNED-BYTE 32) is a subtype of POSITIVE-FIXNUM and
;;; (SIGNED-BYTE 32) of FIXNUM, so a scalar of either type arrives *tagged*,
;;; its value shifted left by N-FIXNUM-TAG-BITS.  Negative literals are
;;; splatted the same way, so dividing by two has to be an ARITHMETIC shift:
;;; SAR in a general-purpose register, before the transfer.  A logical shift
;;; would serve non-negative values only and looks right for exactly as long
;;; as one doesn't try it: -3 is tagged -6, whose low 32 bits are #xFFFFFFFA,
;;; and PSRLD by one gives #x7FFFFFFD (+2147483645) instead of #xFFFFFFFD
;;; (-3), because the bit shifted in is a zero rather than the sign.
;;;
;;; The primitive types are told apart at generation time the way
;;; MOVE-FROM-WORD/FIXNUM does it, from the operand's Lisp type: a ref whose
;;; type is a subtype of FIXNUM is tagged, anything else that can match is
;;; not.  32-bit lane elements are therefore always tagged here, while a
;;; 64-bit lane takes both -- (UNSIGNED-BYTE 64) and (SIGNED-BYTE 64) are not
;;; fixnums, but an integer literal in the same loop is -- so the test stays.
;;;
;;; MOVD and MOVQ with a general-purpose source zero the rest of the
;;; destination register (DEST[MAXVL-1:32] := 0, DEST[MAXVL-1:64] := 0), so
;;; what is left is the value alone in lane 0.  Neither broadcast depends on
;;; that, since PSHUFD rewrites all four lanes and PUNPCKLQDQ defines both
;;; halves, but it does mean nothing stale is ever left to be read.
;;;
;;; Both GPR classes are allowed for the scalar because the tagged primitive
;;; types are allocated in ANY-REG, SIGNED-REG or UNSIGNED-REG depending on
;;; the value's type, and MOVD/MOVQ only look at the bits.

;;; True when TN holds a tagged fixnum rather than an untagged integer.  The
;;; test is on the *primitive type* the register was allocated for, not on the
;;; Lisp type of the reference: a constant is materialized in whichever
;;; primitive type its lvar has, so an integer literal arrives untagged while a
;;; variable declared (SIGNED-BYTE 32) arrives tagged, even though both have a
;;; Lisp type that is a subtype of FIXNUM.  Guessing from the Lisp type puts the
;;; untagging shift in the wrong path and turns -3 into -2.
(defun tagged-fixnum-tn-p (tn)
  (let ((primitive-type (sb-c::tn-primitive-type tn)))
    (or (eq primitive-type (sb-c::primitive-type-or-lose 'fixnum))
        (eq primitive-type (sb-c::primitive-type-or-lose 'positive-fixnum)))))

(define-vop (%vec-splat-ub32)
  (:translate %vec-splat-ub32)
  (:args (x :scs (unsigned-reg signed-reg) :target result))
  (:arg-types unsigned-num)
  (:temporary (:sc unsigned-reg) word)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub32)
  (:generator 2
    (if (tagged-fixnum-tn-p x)
        (progn (move word x)
               (inst sar word n-fixnum-tag-bits)
               (inst movd result word))
        (inst movd result x))
    (inst pshufd result result #b00000000)))

(define-vop (%vec-splat-sb32)
  (:translate %vec-splat-sb32)
  (:args (x :scs (unsigned-reg signed-reg) :target result))
  (:arg-types signed-num)
  (:temporary (:sc unsigned-reg) word)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb32)
  (:generator 2
    (if (tagged-fixnum-tn-p x)
        (progn (move word x)
               (inst sar word n-fixnum-tag-bits)
               (inst movd result word))
        (inst movd result x))
    (inst pshufd result result #b00000000)))

(define-vop (%vec-splat-ub64)
  (:translate %vec-splat-ub64)
  (:args (x :scs (unsigned-reg signed-reg) :target result))
  (:arg-types unsigned-num)
  (:temporary (:sc unsigned-reg) word)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub64)
  (:generator 2
    (if (tagged-fixnum-tn-p x)
        (progn (move word x)
               (inst sar word n-fixnum-tag-bits)
               (inst movq result word))
        (inst movq result x))
    (inst punpcklqdq result result)))

(define-vop (%vec-splat-sb64)
  (:translate %vec-splat-sb64)
  (:args (x :scs (unsigned-reg signed-reg) :target result))
  (:arg-types signed-num)
  (:temporary (:sc unsigned-reg) word)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb64)
  (:generator 2
    (if (tagged-fixnum-tn-p x)
        (progn (move word x)
               (inst sar word n-fixnum-tag-bits)
               (inst movq result word))
        (inst movq result x))
    (inst punpcklqdq result result)))

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
    (publish "LOAD" (unsigned-byte 32) %vec-load-ub32)
    (publish "LOAD" (signed-byte 32) %vec-load-sb32)
    (publish "LOAD" (unsigned-byte 64) %vec-load-ub64)
    (publish "LOAD" (signed-byte 64) %vec-load-sb64)
    (publish "STORE" (unsigned-byte 32) %vec-store-ub32)
    (publish "STORE" (signed-byte 32) %vec-store-sb32)
    (publish "STORE" (unsigned-byte 64) %vec-store-ub64)
    (publish "STORE" (signed-byte 64) %vec-store-sb64)
    (publish "SPLAT" (unsigned-byte 32) %vec-splat-ub32)
    (publish "SPLAT" (signed-byte 32) %vec-splat-sb32)
    (publish "SPLAT" (unsigned-byte 64) %vec-splat-ub64)
    (publish "SPLAT" (signed-byte 64) %vec-splat-sb64)
    (publish "ADD" (unsigned-byte 32) %vec-add-ub32)
    (publish "ADD" (signed-byte 32) %vec-add-sb32)
    (publish "ADD" (unsigned-byte 64) %vec-add-ub64)
    (publish "ADD" (signed-byte 64) %vec-add-sb64)
    (publish "SUB" (unsigned-byte 32) %vec-sub-ub32)
    (publish "SUB" (signed-byte 32) %vec-sub-sb32)
    (publish "SUB" (unsigned-byte 64) %vec-sub-ub64)
    (publish "SUB" (signed-byte 64) %vec-sub-sb64)
    (publish "AND" (unsigned-byte 32) %vec-and-ub32)
    (publish "AND" (signed-byte 32) %vec-and-sb32)
    (publish "AND" (unsigned-byte 64) %vec-and-ub64)
    (publish "AND" (signed-byte 64) %vec-and-sb64)
    (publish "OR" (unsigned-byte 32) %vec-or-ub32)
    (publish "OR" (signed-byte 32) %vec-or-sb32)
    (publish "OR" (unsigned-byte 64) %vec-or-ub64)
    (publish "OR" (signed-byte 64) %vec-or-sb64)
    (publish "XOR" (unsigned-byte 32) %vec-xor-ub32)
    (publish "XOR" (signed-byte 32) %vec-xor-sb32)
    (publish "XOR" (unsigned-byte 64) %vec-xor-ub64)
    (publish "XOR" (signed-byte 64) %vec-xor-sb64)
    (setf sb-c::*vector-primitives* (nreverse primitives))))
