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
                (:args (a :scs (,sc))
                       (b :scs (,sc)))
                (:arg-types ,type ,type)
                (:temporary (:sc ,sc :from (:argument 0)) arith-tmp)
                (:results (result :scs (,sc)))
                (:result-types ,type)
                (:generator 2
                  (move arith-tmp a)
                  (inst ,instruction arith-tmp b)
                  (move result arith-tmp)))))
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
                (:args (a :scs (int-sse-reg))
                       (b :scs (int-sse-reg)))
                (:arg-types ,type ,type)
                (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
                (:results (result :scs (int-sse-reg)))
                (:result-types ,type)
                (:generator 2
                  (move arith-tmp a)
                  (inst ,instruction arith-tmp b)
                  (move result arith-tmp)))))
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

;;;; 8-bit lanes: sixteen at a time.
;;;;
;;;; PADDB/PSUBB are baseline SSE2, and a byte that wraps around is exactly what
;;;; a scalar loop does to an (unsigned-byte 8) array once the range check is
;;;; off.  That is the only case the vectorizer takes these for: the pass only
;;;; rewrites 8-bit element types when the code is compiled at SAFETY 0, because
;;;; at any higher safety the scalar store signals a type error where a packed
;;;; store would quietly truncate.
;;;;
;;;; One element is one byte and a tagged index counts two units per element, so
;;;; the index cannot be scaled: it is untagged into a scratch register and added
;;;; with a scale of 1.  (INDEX-SCALE 1 INDEX would ask the assembler for a
;;;; scale of 0, which does not exist.)
;;;;
;;;; No MUL, DIV, MIN or MAX here: PMULLW would compute the wrong width, and the
;;;; byte-wise minimum and maximum are SSE2's PMINUB/PMAXUB for unsigned only,
;;;; which is a separate feature with its own semantics.

(defknown %vec-load-ub8 ((simple-array (unsigned-byte 8) (*)) index)
    (simd-pack (unsigned-byte 8))
  (flushable))

(define-vop (%vec-load-ub8)
  (:translate %vec-load-ub8)
  (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
  (:arg-types simple-array-unsigned-byte-8 tagged-num)
  (:temporary (:sc any-reg) byte-index)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub8)
  (:generator 3
    (move byte-index index)
    (inst shr byte-index n-fixnum-tag-bits)
    (inst movdqu result
          (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
              vector byte-index 1))))

(defknown %vec-store-ub8 ((simd-pack (unsigned-byte 8)) (simple-array (unsigned-byte 8) (*)) index)
    (values)
  ())

(define-vop (%vec-store-ub8)
  (:translate %vec-store-ub8)
  (:args (value :scs (int-sse-reg))
         (vector :scs (descriptor-reg))
         (index :scs (any-reg)))
  (:arg-types simd-pack-ub8 simple-array-unsigned-byte-8 tagged-num)
  (:temporary (:sc any-reg) byte-index)
  (:generator 3
    (move byte-index index)
    (inst shr byte-index n-fixnum-tag-bits)
    (inst movdqu
          (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
              vector byte-index 1)
          value)))

(defknown %vec-splat-ub8 ((unsigned-byte 8)) (simd-pack (unsigned-byte 8)) (flushable movable))

;;; One byte in all sixteen lanes: strip the tag if there is one, mask the word
;;; down to the byte that is left -- the shift leaves a negative value
;;; sign-extended, and the multiplication below copies whatever is in the high
;;; half as well, turning -3 into #x...FCFCFCFD -- then copy the byte into all
;;; four bytes of a word by multiplying by #x01010101, and PSHUFD the word into
;;; all four dwords.
(define-vop (%vec-splat-ub8)
  (:translate %vec-splat-ub8)
  (:args (x :scs (unsigned-reg signed-reg)))
  (:arg-types unsigned-num)
  (:temporary (:sc unsigned-reg) word)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub8)
  (:generator 3
    (if (tagged-fixnum-tn-p x)
        (progn (move word x) (inst sar word n-fixnum-tag-bits))
        (move word x))
    (inst and word #xFF)
    (inst imul word word #x01010101)
    (inst movd result word)
    (inst pshufd result result #b00000000)))

(defknown %vec-add-ub8 ((simd-pack (unsigned-byte 8)) (simd-pack (unsigned-byte 8)))
    (simd-pack (unsigned-byte 8))
  (flushable movable))

(define-vop (%vec-add-ub8)
  (:translate %vec-add-ub8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-ub8 simd-pack-ub8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub8)
  (:generator 2
    (move arith-tmp a)
    (inst paddb arith-tmp b)
    (move result arith-tmp)))

(defknown %vec-sub-ub8 ((simd-pack (unsigned-byte 8)) (simd-pack (unsigned-byte 8)))
    (simd-pack (unsigned-byte 8))
  (flushable movable))

(define-vop (%vec-sub-ub8)
  (:translate %vec-sub-ub8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-ub8 simd-pack-ub8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub8)
  (:generator 2
    (move arith-tmp a)
    (inst psubb arith-tmp b)
    (move result arith-tmp)))

(defknown %vec-and-ub8 ((simd-pack (unsigned-byte 8)) (simd-pack (unsigned-byte 8)))
    (simd-pack (unsigned-byte 8))
  (flushable movable))

(define-vop (%vec-and-ub8)
  (:translate %vec-and-ub8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-ub8 simd-pack-ub8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub8)
  (:generator 2
    (move arith-tmp a)
    (inst pand arith-tmp b)
    (move result arith-tmp)))

(defknown %vec-or-ub8 ((simd-pack (unsigned-byte 8)) (simd-pack (unsigned-byte 8)))
    (simd-pack (unsigned-byte 8))
  (flushable movable))

(define-vop (%vec-or-ub8)
  (:translate %vec-or-ub8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-ub8 simd-pack-ub8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub8)
  (:generator 2
    (move arith-tmp a)
    (inst por arith-tmp b)
    (move result arith-tmp)))

(defknown %vec-xor-ub8 ((simd-pack (unsigned-byte 8)) (simd-pack (unsigned-byte 8)))
    (simd-pack (unsigned-byte 8))
  (flushable movable))

(define-vop (%vec-xor-ub8)
  (:translate %vec-xor-ub8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-ub8 simd-pack-ub8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub8)
  (:generator 2
    (move arith-tmp a)
    (inst pxor arith-tmp b)
    (move result arith-tmp)))

(defknown %vec-load-sb8 ((simple-array (signed-byte 8) (*)) index)
    (simd-pack (signed-byte 8))
  (flushable))

(define-vop (%vec-load-sb8)
  (:translate %vec-load-sb8)
  (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
  (:arg-types simple-array-signed-byte-8 tagged-num)
  (:temporary (:sc any-reg) byte-index)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb8)
  (:generator 3
    (move byte-index index)
    (inst shr byte-index n-fixnum-tag-bits)
    (inst movdqu result
          (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
              vector byte-index 1))))

(defknown %vec-store-sb8 ((simd-pack (signed-byte 8)) (simple-array (signed-byte 8) (*)) index)
    (values)
  ())

(define-vop (%vec-store-sb8)
  (:translate %vec-store-sb8)
  (:args (value :scs (int-sse-reg))
         (vector :scs (descriptor-reg))
         (index :scs (any-reg)))
  (:arg-types simd-pack-sb8 simple-array-signed-byte-8 tagged-num)
  (:temporary (:sc any-reg) byte-index)
  (:generator 3
    (move byte-index index)
    (inst shr byte-index n-fixnum-tag-bits)
    (inst movdqu
          (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
              vector byte-index 1)
          value)))

(defknown %vec-splat-sb8 ((signed-byte 8)) (simd-pack (signed-byte 8)) (flushable movable))

;;; One byte in all sixteen lanes: strip the tag if there is one, mask the word
;;; down to the byte that is left -- the shift leaves a negative value
;;; sign-extended, and the multiplication below copies whatever is in the high
;;; half as well, turning -3 into #x...FCFCFCFD -- then copy the byte into all
;;; four bytes of a word by multiplying by #x01010101, and PSHUFD the word into
;;; all four dwords.
(define-vop (%vec-splat-sb8)
  (:translate %vec-splat-sb8)
  (:args (x :scs (unsigned-reg signed-reg)))
  (:arg-types signed-num)
  (:temporary (:sc unsigned-reg) word)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb8)
  (:generator 3
    (if (tagged-fixnum-tn-p x)
        (progn (move word x) (inst sar word n-fixnum-tag-bits))
        (move word x))
    (inst and word #xFF)
    (inst imul word word #x01010101)
    (inst movd result word)
    (inst pshufd result result #b00000000)))

(defknown %vec-add-sb8 ((simd-pack (signed-byte 8)) (simd-pack (signed-byte 8)))
    (simd-pack (signed-byte 8))
  (flushable movable))

(define-vop (%vec-add-sb8)
  (:translate %vec-add-sb8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-sb8 simd-pack-sb8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb8)
  (:generator 2
    (move arith-tmp a)
    (inst paddb arith-tmp b)
    (move result arith-tmp)))

(defknown %vec-sub-sb8 ((simd-pack (signed-byte 8)) (simd-pack (signed-byte 8)))
    (simd-pack (signed-byte 8))
  (flushable movable))

(define-vop (%vec-sub-sb8)
  (:translate %vec-sub-sb8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-sb8 simd-pack-sb8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb8)
  (:generator 2
    (move arith-tmp a)
    (inst psubb arith-tmp b)
    (move result arith-tmp)))

(defknown %vec-and-sb8 ((simd-pack (signed-byte 8)) (simd-pack (signed-byte 8)))
    (simd-pack (signed-byte 8))
  (flushable movable))

(define-vop (%vec-and-sb8)
  (:translate %vec-and-sb8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-sb8 simd-pack-sb8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb8)
  (:generator 2
    (move arith-tmp a)
    (inst pand arith-tmp b)
    (move result arith-tmp)))

(defknown %vec-or-sb8 ((simd-pack (signed-byte 8)) (simd-pack (signed-byte 8)))
    (simd-pack (signed-byte 8))
  (flushable movable))

(define-vop (%vec-or-sb8)
  (:translate %vec-or-sb8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-sb8 simd-pack-sb8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb8)
  (:generator 2
    (move arith-tmp a)
    (inst por arith-tmp b)
    (move result arith-tmp)))

(defknown %vec-xor-sb8 ((simd-pack (signed-byte 8)) (simd-pack (signed-byte 8)))
    (simd-pack (signed-byte 8))
  (flushable movable))

(define-vop (%vec-xor-sb8)
  (:translate %vec-xor-sb8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-sb8 simd-pack-sb8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb8)
  (:generator 2
    (move arith-tmp a)
    (inst pxor arith-tmp b)
    (move result arith-tmp)))

;;;; 16-bit lanes: eight at a time.
;;;;
;;;; The four shapes of the integer sections above again, over SIMD-PACK-UB16
;;;; and SIMD-PACK-SB16, which live in INT-SSE-REG like the 32- and 64-bit
;;;; packs, with the arrays spelled SIMPLE-ARRAY-UNSIGNED-BYTE-16 and
;;;; SIMPLE-ARRAY-SIGNED-BYTE-16 (vm-array.lisp).
;;;;
;;;; Two bytes per element can be scaled: a tagged fixnum counts two units per
;;;; element, so (INDEX-SCALE 2 INDEX) is a scale of 1 and the index is used
;;;; exactly as it arrives.  That is what the 8-bit section above cannot do,
;;;; where one element is one byte, a scale of 1 is already the smallest the
;;;; assembler has, and the index therefore has to be untagged by hand.
;;;;
;;;; PADDW and PSUBW are baseline SSE2 and wrap at 16 bits, which is what an
;;;; (UNSIGNED-BYTE 16) or (SIGNED-BYTE 16) element means, and PAND/POR/PXOR do
;;;; not care about the lane width at all.  No MUL or DIV: the vectorizer asks
;;;; for those on float lanes only (VECTOR-OPERATOR-BASE in vectorize.lisp).

(defknown %vec-load-ub16 ((simple-array (unsigned-byte 16) (*)) index)
    (simd-pack (unsigned-byte 16))
  (flushable))
(defknown %vec-load-sb16 ((simple-array (signed-byte 16) (*)) index)
    (simd-pack (signed-byte 16))
  (flushable))
(defknown %vec-store-ub16
    ((simd-pack (unsigned-byte 16)) (simple-array (unsigned-byte 16) (*)) index)
    (values)
  ())
(defknown %vec-store-sb16
    ((simd-pack (signed-byte 16)) (simple-array (signed-byte 16) (*)) index)
    (values)
  ())

(defknown (%vec-add-ub16 %vec-sub-ub16
           %vec-and-ub16 %vec-or-ub16 %vec-xor-ub16)
    ((simd-pack (unsigned-byte 16)) (simd-pack (unsigned-byte 16)))
    (simd-pack (unsigned-byte 16))
  (flushable movable))
(defknown (%vec-add-sb16 %vec-sub-sb16
           %vec-and-sb16 %vec-or-sb16 %vec-xor-sb16)
    ((simd-pack (signed-byte 16)) (simd-pack (signed-byte 16)))
    (simd-pack (signed-byte 16))
  (flushable movable))

(defknown %vec-splat-ub16 ((unsigned-byte 16))
    (simd-pack (unsigned-byte 16))
  (flushable movable))
(defknown %vec-splat-sb16 ((signed-byte 16))
    (simd-pack (signed-byte 16))
  (flushable movable))

;;;; Loading and storing 128 bits at a time in a simple 16-bit array.
;;;
;;; The effective address is the one the float and 32/64-bit sections use,
;;; with the element size -- two bytes -- in place of N-WORD-BYTES.

(macrolet ((define-vec-load (name element-type result-type)
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
                            (index-scale 2 index))))))
           (define-vec-store (name element-type result-type)
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
                            (index-scale 2 index))
                        value)))))
  (define-vec-load %vec-load-ub16 simple-array-unsigned-byte-16 simd-pack-ub16)
  (define-vec-load %vec-load-sb16 simple-array-signed-byte-16 simd-pack-sb16)
  (define-vec-store %vec-store-ub16 simple-array-unsigned-byte-16 simd-pack-ub16)
  (define-vec-store %vec-store-sb16 simple-array-signed-byte-16 simd-pack-sb16))

;;;; Packed 16-bit arithmetic and bitwise operations.
;;;
;;; Same reasoning as the integer arithmetic above: the destination is also the
;;; first source, so RESULT is MOVEd from A first and :TARGET is only a packing
;;; preference.

(macrolet ((define-vec-arith (name type instruction)
             `(define-vop (,name)
                (:translate ,name)
                (:args (a :scs (int-sse-reg))
                       (b :scs (int-sse-reg)))
                (:arg-types ,type ,type)
                (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
                (:results (result :scs (int-sse-reg)))
                (:result-types ,type)
                (:generator 2
                  (move arith-tmp a)
                  (inst ,instruction arith-tmp b)
                  (move result arith-tmp)))))
  (define-vec-arith %vec-add-ub16 simd-pack-ub16 paddw)
  (define-vec-arith %vec-sub-ub16 simd-pack-ub16 psubw)
  (define-vec-arith %vec-and-ub16 simd-pack-ub16 pand)
  (define-vec-arith %vec-or-ub16 simd-pack-ub16 por)
  (define-vec-arith %vec-xor-ub16 simd-pack-ub16 pxor)
  (define-vec-arith %vec-add-sb16 simd-pack-sb16 paddw)
  (define-vec-arith %vec-sub-sb16 simd-pack-sb16 psubw)
  (define-vec-arith %vec-and-sb16 simd-pack-sb16 pand)
  (define-vec-arith %vec-or-sb16 simd-pack-sb16 por)
  (define-vec-arith %vec-xor-sb16 simd-pack-sb16 pxor))

;;;; Splatting one 16-bit integer across every lane.
;;;
;;; The scalar arrives in a general-purpose register, tagged unless the compiler
;;; materialized an untagged value, and there is no move VOP from there into
;;; INT-SSE-REG, so the transfer is an instruction rather than a MOVE, as in the
;;; splats above: strip the tag with an ARITHMETIC shift, mask the word down to
;;; the 16 bits that are left, copy the value into both halves of a 32-bit word
;;; by multiplying by #x00010001, MOVD that word into lane 0 -- which zeroes the
;;; rest of the register -- and PSHUFD the word into all four dwords, which is
;;; all eight 16-bit lanes.
;;;
;;; The AND is not decoration.  The shift leaves a negative value sign-extended
;;; to the full word, where the multiply copies whatever is in the high half as
;;; well: -3 in a 16-bit lane times #x00010001 is #x...FFFCFFFD, whose second
;;; half is -4.  Masking first makes the two halves identical for every value
;;; in the range, -32768 included.

(define-vop (%vec-splat-ub16)
  (:translate %vec-splat-ub16)
  (:args (x :scs (unsigned-reg signed-reg) :target result))
  (:arg-types unsigned-num)
  (:temporary (:sc unsigned-reg) word)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub16)
  (:generator 4
    (if (tagged-fixnum-tn-p x)
        (progn (move word x) (inst sar word n-fixnum-tag-bits))
        (move word x))
    (inst and word #xFFFF)
    (inst imul word word #x00010001)
    (inst movd result word)
    (inst pshufd result result #b00000000)))

(define-vop (%vec-splat-sb16)
  (:translate %vec-splat-sb16)
  (:args (x :scs (unsigned-reg signed-reg) :target result))
  (:arg-types signed-num)
  (:temporary (:sc unsigned-reg) word)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb16)
  (:generator 4
    (if (tagged-fixnum-tn-p x)
        (progn (move word x) (inst sar word n-fixnum-tag-bits))
        (move word x))
    (inst and word #xFFFF)
    (inst imul word word #x00010001)
    (inst movd result word)
    (inst pshufd result result #b00000000)))

;;;; Negating a packed float.
;;;
;;; A scalar (- x) on a float does not subtract: it flips the sign bit, with
;;; XORPS/XORPD against a constant mask (%NEGATE/SINGLE-FLOAT and
;;; %NEGATE/DOUBLE-FLOAT in float.lisp, which use
;;; (REGISTER-INLINE-CONSTANT :OWORD (ASH 1 31)) and (ASH 1 63) respectively).
;;; The packed version is that instruction with the mask in every lane, which
;;; makes it bit-identical to the scalar one on the cases a subtraction gets
;;; wrong: 0.0 comes out as -0.0 and -0.0 as 0.0, and a NaN keeps its payload
;;; with only the sign bit flipped.
;;;
;;; The constant is one 128-bit OWORD integer, because
;;; CANONICALIZE-INLINE-CONSTANT (insts.lisp) destructures the descriptor as
;;; (TYPE VALUE) and :OWORD takes a single integer.  The lanes are the sign bits
;;; of the pack: #x80000000 is the bit pattern of the single-float -0.0, which
;;; is the top bit of a 32-bit lane, so the first mask sets bits 31, 63, 95 and
;;; 127; and the double-float -0.0 -- (MAKE-DOUBLE-FLOAT #x80000000 0) -- is
;;; that dword in the high half of a 64-bit lane, so the second mask sets bit
;;; 63 and bit 127.

(defknown %vec-negate-single ((simd-pack single-float))
    (simd-pack single-float)
  (flushable movable))
(defknown %vec-negate-double ((simd-pack double-float))
    (simd-pack double-float)
  (flushable movable))

(define-vop (%vec-negate-single)
  (:translate %vec-negate-single)
  (:args (a :scs (single-sse-reg) :target result))
  (:arg-types simd-pack-single)
  (:results (result :scs (single-sse-reg)))
  (:result-types simd-pack-single)
  (:generator 2
    (move result a)
    (inst xorps result
          (register-inline-constant
           :oword (logior #x80000000
                          (ash #x80000000 32)
                          (ash #x80000000 64)
                          (ash #x80000000 96))))))

(define-vop (%vec-negate-double)
  (:translate %vec-negate-double)
  (:args (a :scs (double-sse-reg) :target result))
  (:arg-types simd-pack-double)
  (:results (result :scs (double-sse-reg)))
  (:result-types simd-pack-double)
  (:generator 2
    (move result a)
    (inst xorpd result
          (register-inline-constant
           :oword (logior (ash #x80000000 32)
                          (ash #x80000000 96))))))

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
    (publish "NEG" single-float %vec-negate-single)
    (publish "NEG" double-float %vec-negate-double)
    (publish "LOAD" (unsigned-byte 8) %vec-load-ub8)
    (publish "LOAD" (signed-byte 8) %vec-load-sb8)
    (publish "STORE" (unsigned-byte 8) %vec-store-ub8)
    (publish "STORE" (signed-byte 8) %vec-store-sb8)
    (publish "SPLAT" (unsigned-byte 8) %vec-splat-ub8)
    (publish "SPLAT" (signed-byte 8) %vec-splat-sb8)
    (publish "ADD" (unsigned-byte 8) %vec-add-ub8)
    (publish "ADD" (signed-byte 8) %vec-add-sb8)
    (publish "SUB" (unsigned-byte 8) %vec-sub-ub8)
    (publish "SUB" (signed-byte 8) %vec-sub-sb8)
    (publish "AND" (unsigned-byte 8) %vec-and-ub8)
    (publish "AND" (signed-byte 8) %vec-and-sb8)
    (publish "OR" (unsigned-byte 8) %vec-or-ub8)
    (publish "OR" (signed-byte 8) %vec-or-sb8)
    (publish "XOR" (unsigned-byte 8) %vec-xor-ub8)
    (publish "XOR" (signed-byte 8) %vec-xor-sb8)
    (publish "LOAD" (unsigned-byte 16) %vec-load-ub16)
    (publish "LOAD" (signed-byte 16) %vec-load-sb16)
    (publish "STORE" (unsigned-byte 16) %vec-store-ub16)
    (publish "STORE" (signed-byte 16) %vec-store-sb16)
    (publish "SPLAT" (unsigned-byte 16) %vec-splat-ub16)
    (publish "SPLAT" (signed-byte 16) %vec-splat-sb16)
    (publish "ADD" (unsigned-byte 16) %vec-add-ub16)
    (publish "ADD" (signed-byte 16) %vec-add-sb16)
    (publish "SUB" (unsigned-byte 16) %vec-sub-ub16)
    (publish "SUB" (signed-byte 16) %vec-sub-sb16)
    (publish "AND" (unsigned-byte 16) %vec-and-ub16)
    (publish "AND" (signed-byte 16) %vec-and-sb16)
    (publish "OR" (unsigned-byte 16) %vec-or-ub16)
    (publish "OR" (signed-byte 16) %vec-or-sb16)
    (publish "XOR" (unsigned-byte 16) %vec-xor-ub16)
    (publish "XOR" (signed-byte 16) %vec-xor-sb16)
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
