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

;;;; function signatures
;;;
;;; the arrays are spelled (simple-array <element> (*)) where a Lisp type
;;; specifier is wanted, and by their primitive type names in :arg-types.
;;;
;;; a load is flushable but neither movable nor foldable, and a store is neither
;;; of those either: 128 bits at a time must not be reordered across a store into
;;; the same array, because a loop that reads what it just wrote depends on that
;;; order.

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

(defknown %vec-widen-single-double ((simple-array single-float (*)) index)
    (simd-pack double-float)
  (flushable movable))

(defknown %vec-splat-single (single-float)
    (simd-pack single-float)
  (flushable movable))
(defknown %vec-splat-double (double-float)
    (simd-pack double-float)
  (flushable movable))

;;;; loading and storing 128 bits at a time in a simple float array
;;;
;;; the effective address is the one define-full-setter+addend uses in
;;; array.lisp, with the element size in place of n-word-bytes.  index is
;;; restricted to any-reg and declared tagged-num so that index-scale
;;; divides the element size by the tag bits; an immediate index is not
;;; accepted, since the scale would then have to be the element size.

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
;;; value is first in both the Lisp arglist and :args.
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

;;;; packed arithmetic
;;;
;;; the destination is also the first source, so result is moved from a
;;; explicitly: :target is only a packing preference, and relying on it
;;; would compute into whatever stale value the register holds.

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

;;;; splatting one float across every lane
;;;
;;; the arg is a scalar float sc targeted at the result and moved into it once,
;;; and the shuffle then selects the low lane alone.

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

;;;; integer lanes: 32 and 64 bits wide, signed and unsigned
;;;
;;; no integer mul or div here: pmulld is sse4.1 and pmuludq reads only the low
;;; half of each 64-bit lane.  add, sub and the three bitwise operations are
;;; baseline sse2 at both widths.

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

;;;; loading and storing 128 bits at a time in a simple integer array
;;;
;;; the same two shapes as the float section above, in a macrolet scoped to this
;;; section: movdqu needs no element type, and only the element size and the
;;; array type vary.

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

;;;; packed integer arithmetic and bitwise operations
;;;
;;; as above, result is moved from a explicitly.  the instructions wrap on
;;; overflow, which is what the (unsigned-byte n) and (signed-byte n)
;;; element types mean.

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

;;;; splatting one integer across every lane
;;;
;;; the scalar arrives in a general-purpose register and there is no move vop
;;; from there into int-sse-reg, so the transfer is an instruction: movd for
;;; 32-bit lanes, movq for 64-bit.
;;;
;;; the splat vops take an unsigned-num or signed-num operand, and every one of
;;; them takes it as it arrives: the argument's storage class is one of the
;;; untagged ones, so the move that put the value in the register is what
;;; stripped the tag, whatever primitive type the value came from.  untagging
;;; again in the 64-bit vops would halve a scalar the loop declared fixnum, which
;;; the pass offers to a 64-bit lane loop just as it does a (signed-byte 64) one;
;;; tests/auto-vectorize.pure.lisp pins both.

(define-vop (%vec-splat-ub32)
  (:translate %vec-splat-ub32)
  (:args (x :scs (unsigned-reg signed-reg) :target result))
  (:arg-types unsigned-num)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub32)
  (:generator 2
    (inst movd result x)
    (inst pshufd result result #b00000000)))

(define-vop (%vec-splat-sb32)
  (:translate %vec-splat-sb32)
  (:args (x :scs (unsigned-reg signed-reg) :target result))
  (:arg-types signed-num)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb32)
  (:generator 2
    (inst movd result x)
    (inst pshufd result result #b00000000)))

(define-vop (%vec-splat-ub64)
  (:translate %vec-splat-ub64)
  (:args (x :scs (unsigned-reg signed-reg) :target result))
  (:arg-types unsigned-num)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub64)
  (:generator 2
    (inst movq result x)
    (inst punpcklqdq result result)))

(define-vop (%vec-splat-sb64)
  (:translate %vec-splat-sb64)
  (:args (x :scs (unsigned-reg signed-reg) :target result))
  (:arg-types signed-num)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb64)
  (:generator 2
    (inst movq result x)
    (inst punpcklqdq result result)))

;;;; 8-bit lanes: sixteen at a time
;;;
;;; paddb/psubb are baseline sse2, and a byte that wraps around is what a
;;; scalar loop does to an (unsigned-byte 8) array once the range check is
;;; off, which is the only case the vectorizer takes these for.
;;;
;;; one element is one byte and a tagged index counts two units per element,
;;; so the index cannot be scaled: it is untagged and added with the smallest
;;; scale the assembler has.

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

;;; one byte in all sixteen lanes: mask down to the byte, copy it into all four
;;; bytes of a word with #x01010101, and pshufd the word.  the mask is required
;;; whatever the value's sign, since the multiplication copies the whole word and
;;; -3 would otherwise duplicate the sign extension, #x...FCFCFCFD.
(define-vop (%vec-splat-ub8)
  (:translate %vec-splat-ub8)
  (:args (x :scs (unsigned-reg signed-reg)))
  (:arg-types unsigned-num)
  (:temporary (:sc unsigned-reg) word)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub8)
  (:generator 3
    (move word x)
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

;;; as above, for (signed-byte 8).
(define-vop (%vec-splat-sb8)
  (:translate %vec-splat-sb8)
  (:args (x :scs (unsigned-reg signed-reg)))
  (:arg-types signed-num)
  (:temporary (:sc unsigned-reg) word)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb8)
  (:generator 3
    (move word x)
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

;;;; 16-bit lanes: eight at a time
;;;
;;; two bytes per element can be scaled: a tagged fixnum counts two units per
;;; element, so (index-scale 2 index) is a scale of 1 and the index is used
;;; exactly as it arrives.
;;;
;;; paddw and psubw wrap at 16 bits, which is what the element type means.

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

;;;; loading and storing 128 bits at a time in a simple 16-bit array
;;;
;;; the same effective address again, with the two-byte element size.

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

;;;; packed 16-bit arithmetic and bitwise operations
;;;
;;; as above: result is moved from a explicitly, :target is only a preference.

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

;;;; splatting one 16-bit integer across every lane
;;;
;;; mask to the 16 bits that are left, multiply by #x00010001 to get the value
;;; into both halves of a word, then movd and pshufd as for the bytes.
;;;
;;; the and is required: the multiplication copies the whole word, so without it
;;; a negative value's sign extension would leave the high half of
;;; -3 * #x00010001 as -4 rather than -3.

(define-vop (%vec-splat-ub16)
  (:translate %vec-splat-ub16)
  (:args (x :scs (unsigned-reg signed-reg) :target result))
  (:arg-types unsigned-num)
  (:temporary (:sc unsigned-reg) word)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub16)
  (:generator 4
    (move word x)
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
    (move word x)
    (inst and word #xFFFF)
    (inst imul word word #x00010001)
    (inst movd result word)
    (inst pshufd result result #b00000000)))

;;;; negating a packed float
;;;
;;; a scalar (- x) on a float flips the sign bit rather than subtracting
;;; (%negate/single-float in float.lisp), and the packed version is the same
;;; instruction with the mask in every lane: 0.0 becomes -0.0, -0.0 becomes
;;; 0.0, and a nan keeps its payload.
;;;
;;; the mask is one 128-bit integer, since canonicalize-inline-constant
;;; destructures an :oword descriptor as (type value).

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

;;; byte-wise minimum and maximum, unsigned only: pminub and pmaxub are sse2,
;;; the signed and word-width forms are sse4.1.

(defknown (%vec-min-ub8 %vec-max-ub8)
    ((simd-pack (unsigned-byte 8)) (simd-pack (unsigned-byte 8)))
    (simd-pack (unsigned-byte 8))
  (flushable movable))

(define-vop (%vec-min-ub8)
  (:translate %vec-min-ub8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-ub8 simd-pack-ub8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub8)
  (:generator 2
    (move arith-tmp a)
    (inst pminub arith-tmp b)
    (move result arith-tmp)))

(define-vop (%vec-max-ub8)
  (:translate %vec-max-ub8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-ub8 simd-pack-ub8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub8)
  (:generator 2
    (move arith-tmp a)
    (inst pmaxub arith-tmp b)
    (move result arith-tmp)))

;;;; a lane vector holding the loop index: [i, i+1, ... ]
;;;
;;; the scalar index is converted to float and broadcast, and a constant ramp
;;; supplies the lane offsets; the index arrives tagged, so it is untagged
;;; before the conversion.

(defknown (%vec-splat-index-single) (index) (simd-pack single-float) (flushable movable))
(defknown (%vec-splat-index-double) (index) (simd-pack double-float) (flushable movable))
(defknown (%vec-ramp-single) () (simd-pack single-float) (flushable movable))
(defknown (%vec-ramp-double) () (simd-pack double-float) (flushable movable))

(define-vop (%vec-splat-index-single)
  (:translate %vec-splat-index-single)
  (:args (index :scs (any-reg)))
  (:arg-types tagged-num)
  (:temporary (:sc signed-reg) word)
  (:results (result :scs (single-sse-reg)))
  (:result-types simd-pack-single)
  (:generator 3
    (move word index)
    (inst sar word n-fixnum-tag-bits)
    (inst cvtsi2ss result word)
    (inst shufps result result #b00000000)))

(define-vop (%vec-splat-index-double)
  (:translate %vec-splat-index-double)
  (:args (index :scs (any-reg)))
  (:arg-types tagged-num)
  (:temporary (:sc signed-reg) word)
  (:results (result :scs (double-sse-reg)))
  (:result-types simd-pack-double)
  (:generator 3
    (move word index)
    (inst sar word n-fixnum-tag-bits)
    (inst cvtsi2sd result word)
    (inst unpcklpd result result)))

;;; the lane offsets 0, 1, 2, 3 for single-floats and 0, 1 for doubles, as one
;;; 128-bit constant each: 1.0f0 is #x3F800000, 2.0f0 #x40000000, 3.0f0 #x40400000,
;;; 1.0d0 #x3FF0000000000000.
(define-vop (%vec-ramp-single)
  (:translate %vec-ramp-single)
  (:results (result :scs (single-sse-reg)))
  (:result-types simd-pack-single)
  (:generator 1
    (inst movaps result
          (register-inline-constant
           :oword (logior (ash #x3F800000 32) (ash #x40000000 64) (ash #x40400000 96))))))

(define-vop (%vec-ramp-double)
  (:translate %vec-ramp-double)
  (:results (result :scs (double-sse-reg)))
  (:result-types simd-pack-double)
  (:generator 1
    (inst movaps result
          (register-inline-constant :oword (ash #x3FF0000000000000 64)))))

;;; the width of a vector in every lane, which is how much the index pack advances:
;;; 4.0f0 is #x40800000 and 2.0d0 is #x4000000000000000.
(defknown (%vec-stride-single) () (simd-pack single-float) (flushable movable))
(defknown (%vec-stride-double) () (simd-pack double-float) (flushable movable))

(define-vop (%vec-stride-single)
  (:translate %vec-stride-single)
  (:results (result :scs (single-sse-reg)))
  (:result-types simd-pack-single)
  (:generator 1
    (inst movaps result
          (register-inline-constant
           :oword (logior #x40800000 (ash #x40800000 32)
                          (ash #x40800000 64) (ash #x40800000 96))))))

(define-vop (%vec-stride-double)
  (:translate %vec-stride-double)
  (:results (result :scs (double-sse-reg)))
  (:result-types simd-pack-double)
  (:generator 1
    (inst movaps result
          (register-inline-constant
           :oword (logior #x4000000000000000 (ash #x4000000000000000 64))))))

;;; pmullw gives the low 16 bits of each product, which is what the scalar loop's
;;; value is once it is stored in a 16-bit element.

(defknown (%vec-mul-ub16) ((simd-pack (unsigned-byte 16)) (simd-pack (unsigned-byte 16))) (simd-pack (unsigned-byte 16)) (flushable movable))

(define-vop (%vec-mul-ub16)
  (:translate %vec-mul-ub16)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-ub16 simd-pack-ub16)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub16)
  (:generator 3
    (move arith-tmp a)
    (inst pmullw arith-tmp b)
    (move result arith-tmp)))

(defknown (%vec-mul-sb16) ((simd-pack (signed-byte 16)) (simd-pack (signed-byte 16))) (simd-pack (signed-byte 16)) (flushable movable))

(define-vop (%vec-mul-sb16)
  (:translate %vec-mul-sb16)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-sb16 simd-pack-sb16)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb16)
  (:generator 3
    (move arith-tmp a)
    (inst pmullw arith-tmp b)
    (move result arith-tmp)))

;;; byte lanes have no multiply at any width, so the operands are unpacked into
;;; 16-bit lanes, multiplied with PMULLW, masked to the low byte and packed back.
;;; PACKUSWB saturates, so the mask before the pack is what makes the result wrap
;;; like the scalar loop does; the mask is made with PCMPEQD + PSRLW rather than a
;;; 128-bit constant, which register-inline-constant takes only as one integer.
(defknown (%vec-mul-ub8) ((simd-pack (unsigned-byte 8)) (simd-pack (unsigned-byte 8))) (simd-pack (unsigned-byte 8)) (flushable movable))

(define-vop (%vec-mul-ub8)
  (:translate %vec-mul-ub8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-ub8 simd-pack-ub8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) lo-a)
  (:temporary (:sc int-sse-reg :from (:argument 1)) lo-b)
  (:temporary (:sc int-sse-reg) hi-a)
  (:temporary (:sc int-sse-reg) hi-b)
  (:temporary (:sc int-sse-reg) zero)
  (:temporary (:sc int-sse-reg) mask)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub8)
  (:generator 8
    (inst pxor zero zero)
    (inst pcmpeqd mask mask)
    (inst psrlw mask 8)
    (move lo-a a)
    (inst punpcklbw lo-a zero)
    (move lo-b b)
    (inst punpcklbw lo-b zero)
    (move hi-a a)
    (inst punpckhbw hi-a zero)
    (move hi-b b)
    (inst punpckhbw hi-b zero)
    (inst pmullw lo-a lo-b)
    (inst pmullw hi-a hi-b)
    (inst pand lo-a mask)
    (inst pand hi-a mask)
    (inst packuswb lo-a hi-a)
    (move result lo-a)))

(defknown (%vec-mul-sb8) ((simd-pack (signed-byte 8)) (simd-pack (signed-byte 8))) (simd-pack (signed-byte 8)) (flushable movable))

(define-vop (%vec-mul-sb8)
  (:translate %vec-mul-sb8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-sb8 simd-pack-sb8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) lo-a)
  (:temporary (:sc int-sse-reg :from (:argument 1)) lo-b)
  (:temporary (:sc int-sse-reg) hi-a)
  (:temporary (:sc int-sse-reg) hi-b)
  (:temporary (:sc int-sse-reg) zero)
  (:temporary (:sc int-sse-reg) mask)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb8)
  (:generator 8
    (inst pxor zero zero)
    (inst pcmpeqd mask mask)
    (inst psrlw mask 8)
    (move lo-a a)
    (inst punpcklbw lo-a zero)
    (move lo-b b)
    (inst punpcklbw lo-b zero)
    (move hi-a a)
    (inst punpckhbw hi-a zero)
    (move hi-b b)
    (inst punpckhbw hi-b zero)
    (inst pmullw lo-a lo-b)
    (inst pmullw hi-a hi-b)
    (inst pand lo-a mask)
    (inst pand hi-a mask)
    (inst packuswb lo-a hi-a)
    (move result lo-a)))


(defknown (%vec-mul-ub8) ((simd-pack (unsigned-byte 8)) (simd-pack (unsigned-byte 8))) (simd-pack (unsigned-byte 8)) (flushable movable))

(define-vop (%vec-mul-ub8)
  (:translate %vec-mul-ub8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-ub8 simd-pack-ub8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) lo-a)
  (:temporary (:sc int-sse-reg :from (:argument 1)) lo-b)
  (:temporary (:sc int-sse-reg)) hi-a)
  (:temporary (:sc int-sse-reg)) hi-b)
  (:temporary (:sc int-sse-reg)) zero)
  (:temporary (:sc int-sse-reg)) mask)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub8)
  (:generator 8
    ;; the arguments are copied FIRST: operands are modelled as read at the start of
    ;; the vop, so a temporary written before the copy can be packed onto an
    ;; argument register and destroy it (cost a build and a wrong-values report
    ;; from the harness: element 8 was 253 instead of 9)
    (move lo-a a)
    (move lo-b b)
    (inst pxor zero zero)
    (inst pcmpeqd mask mask)
    (inst psrlw mask 8)
    (move hi-a lo-a)
    (inst punpckhbw hi-a zero)
    (inst punpcklbw lo-a zero)
    (move hi-b lo-b)
    (inst punpckhbw hi-b zero)
    (inst punpcklbw lo-b zero)
    (inst pmullw lo-a lo-b)
    (inst pmullw hi-a hi-b)
    ;; keep the low byte so the result WRAPS: PACKUSWB on its own saturates
    (inst pand lo-a mask)
    (inst pand hi-a mask)
    (inst packuswb lo-a hi-a)
    (move result lo-a)))

(defknown (%vec-mul-sb8) ((simd-pack (signed-byte 8)) (simd-pack (signed-byte 8))) (simd-pack (signed-byte 8)) (flushable movable))

(define-vop (%vec-mul-sb8)
  (:translate %vec-mul-sb8)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-sb8 simd-pack-sb8)
  (:temporary (:sc int-sse-reg :from (:argument 0)) lo-a)
  (:temporary (:sc int-sse-reg :from (:argument 1)) lo-b)
  (:temporary (:sc int-sse-reg)) hi-a)
  (:temporary (:sc int-sse-reg)) hi-b)
  (:temporary (:sc int-sse-reg)) zero)
  (:temporary (:sc int-sse-reg)) mask)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb8)
  (:generator 8
    ;; the arguments are copied FIRST: operands are modelled as read at the start of
    ;; the vop, so a temporary written before the copy can be packed onto an
    ;; argument register and destroy it (cost a build and a wrong-values report
    ;; from the harness: element 8 was 253 instead of 9)
    (move lo-a a)
    (move lo-b b)
    (inst pxor zero zero)
    (inst pcmpeqd mask mask)
    (inst psrlw mask 8)
    (move hi-a lo-a)
    (inst punpckhbw hi-a zero)
    (inst punpcklbw lo-a zero)
    (move hi-b lo-b)
    (inst punpckhbw hi-b zero)
    (inst punpcklbw lo-b zero)
    (inst pmullw lo-a lo-b)
    (inst pmullw hi-a hi-b)
    ;; keep the low byte so the result WRAPS: PACKUSWB on its own saturates
    (inst pand lo-a mask)
    (inst pand hi-a mask)
    (inst packuswb lo-a hi-a)
    (move result lo-a)))

;;; 32-bit multiplication is PMULLD, which is SSE4.1 rather than SSE2, so both the
;;; vop and its entry in the table are published only by a build that has the
;;; instruction: a table entry with no vop behind it is worse than no entry, since
;;; the pass finds it, emits the call, and the loop dies with an undefined function.
;;;
;;; one instruction serves both element types, because the low 32 bits of a product
;;; do not depend on the signedness of the operands - the lane holds the same bits
;;; either way, and only a load or store interprets them.

(defknown (%vec-mul-ub32) ((simd-pack (unsigned-byte 32)) (simd-pack (unsigned-byte 32))) (simd-pack (unsigned-byte 32)) (flushable movable))

(define-vop (%vec-mul-ub32)
  (:translate %vec-mul-ub32)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-ub32 simd-pack-ub32)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub32)
  (:generator 3
    ;; the arguments are read before the result is written, so the product is
    ;; computed into the temporary and moved to the result last
    (move arith-tmp a)
    (inst pmulld arith-tmp b)
    (move result arith-tmp)))

(defknown (%vec-mul-sb32) ((simd-pack (signed-byte 32)) (simd-pack (signed-byte 32))) (simd-pack (signed-byte 32)) (flushable movable))

(define-vop (%vec-mul-sb32)
  (:translate %vec-mul-sb32)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-sb32 simd-pack-sb32)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-sb32)
  (:generator 3
    (move arith-tmp a)
    (inst pmulld arith-tmp b)
    (move result arith-tmp)))

;;; minimum and maximum beyond the unsigned bytes above.  pminsw and pmaxsw are
;;; sse2, so signed words take one instruction each.  unsigned words have no sse2
;;; form: min(a,b) is a - max(a-b,0) and max(a,b) is a + max(b-a,0), and psubusw
;;; is that saturating difference.  signed bytes have no sse2 form either, but
;;; xor with #x80 maps them onto unsigned order, so the unbiased pminub does it.

(defknown (%vec-min-ub16 %vec-max-ub16)
    ((simd-pack (unsigned-byte 16)) (simd-pack (unsigned-byte 16)))
    (simd-pack (unsigned-byte 16))
  (flushable movable))

(defknown (%vec-min-sb16 %vec-max-sb16)
    ((simd-pack (signed-byte 16)) (simd-pack (signed-byte 16)))
    (simd-pack (signed-byte 16))
  (flushable movable))

(defknown (%vec-min-sb8 %vec-max-sb8)
    ((simd-pack (signed-byte 8)) (simd-pack (signed-byte 8)))
    (simd-pack (signed-byte 8))
  (flushable movable))

(define-vop (%vec-min-ub16)
  (:translate %vec-min-ub16)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-ub16 simd-pack-ub16)
  (:temporary (:sc int-sse-reg :from (:argument 0)) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub16)
  (:generator 3
    (inst psubusw arith-tmp b)
    (move result a)
    (inst psubw result arith-tmp)))

(define-vop (%vec-max-ub16)
  (:translate %vec-max-ub16)
  (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
  (:arg-types simd-pack-ub16 simd-pack-ub16)
  (:temporary (:sc int-sse-reg) arith-tmp)
  (:results (result :scs (int-sse-reg)))
  (:result-types simd-pack-ub16)
  (:generator 3
    ;; the temp is live across result's write, so it cannot share a register with
    ;; it, and it is written before a is read into result, so it cannot share a's
    (move arith-tmp b)
    (inst psubusw arith-tmp a)
    (move result a)
    (inst paddw result arith-tmp)))

(macrolet ((signed-word-min-max (operation instruction)
             `(define-vop (,(intern (format nil "%VEC-~a-SB16" operation)))
                (:translate ,(intern (format nil "%VEC-~a-SB16" operation)))
                (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
                (:arg-types simd-pack-sb16 simd-pack-sb16)
                (:temporary (:sc int-sse-reg) arith-tmp)
                (:results (result :scs (int-sse-reg)))
                (:result-types simd-pack-sb16)
                ;; b may be allocated to result's register, so it is read before
                ;; result is written: compute into the temp and move to result last
                (:generator 2
                  (move arith-tmp a)
                  (inst ,instruction arith-tmp b)
                  (move result arith-tmp)))))
  (signed-word-min-max "MIN" pminsw)
  (signed-word-min-max "MAX" pmaxsw))

(macrolet ((signed-byte-min-max (operation instruction)
             `(define-vop (,(intern (format nil "%VEC-~a-SB8" operation)))
                (:translate ,(intern (format nil "%VEC-~a-SB8" operation)))
                (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
                (:arg-types simd-pack-sb8 simd-pack-sb8)
                (:temporary (:sc int-sse-reg) biased-b)
                (:results (result :scs (int-sse-reg)))
                (:result-types simd-pack-sb8)
                (:generator 4
                  (move biased-b b)
                  (inst pxor biased-b (register-inline-constant :oword #x80808080808080808080808080808080))
                  (move result a)
                  (inst pxor result (register-inline-constant :oword #x80808080808080808080808080808080))
                  (inst ,instruction result biased-b)
                  (inst pxor result (register-inline-constant :oword #x80808080808080808080808080808080))))))
  (signed-byte-min-max "MIN" pminub)
  (signed-byte-min-max "MAX" pmaxub))

;;;; reading narrow integers into float lanes: a byte image or 16-bit audio widened
;;;; to the floats a float loop works in.
;;;
;;;; each reads exactly the elements one float iteration consumes (four for singles,
;;;; two for doubles), so the length guard the pass already emits for the float loop
;;;; covers the read.  Reading less would leave lanes unset, and reading more would
;;;; read past the end of the array on the last iteration.
;;;
;;;; the SSE2 sequence spreads each element over a dword with unpacks, keeps it and
;;;; drops the copies with a shift, then converts.  A byte source needs both unpacks;
;;;; a 16-bit source needs only the word one, since interleaving its bytes first
;;;; would scramble the value before the shift.
;;;
;;;; SSE4.1 has a narrowing load that does it in one instruction, so a backend with
;;;; it uses PMOVZXBD, PMOVSXBD, PMOVZXWD or PMOVSXWD for the four-lane loops, and
;;;; the template without a guard is the fallback; a template carries its own name
;;;; but translates the same function, which is how a guard selects between them.  The two-lane loops keep the SSE2
;;;; sequence, because no packed read reaches fewer than four bytes and the two bytes
;;;; of a byte source have to come through a register.

(progn
  (defknown %vec-widen-ub8-single ((simple-array (unsigned-byte 8) (*)) index) (simd-pack single-float)
    (flushable movable))
  (define-vop (%vec-widen-ub8-single-sse4)
    (:translate %vec-widen-ub8-single)
    (:guard (member :sse4 *backend-subfeatures*))
    (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
    (:arg-types simple-array-unsigned-byte-8 tagged-num)
    (:temporary (:sc any-reg) byte-index)
    (:results (result :scs (single-sse-reg)))
    (:result-types simd-pack-single)
    (:generator 4
      (move byte-index index)
      (inst shr byte-index n-fixnum-tag-bits)
      (inst pmovzxbd result (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                      vector byte-index 1))
      (inst cvtdq2ps result result)))
  (define-vop (%vec-widen-ub8-single)
    (:translate %vec-widen-ub8-single)
    (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
    (:arg-types simple-array-unsigned-byte-8 tagged-num)
    (:temporary (:sc any-reg) byte-index)
    (:results (result :scs (single-sse-reg)))
    (:result-types simd-pack-single)
    (:generator 4
      (move byte-index index)
      (inst shr byte-index n-fixnum-tag-bits)
      (inst movd result (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                      vector byte-index 1))
      (inst punpcklbw result result)
      (inst punpcklwd result result)
      (inst psrld result 24)
      (inst cvtdq2ps result result)))
)

(progn
  (defknown %vec-widen-sb8-single ((simple-array (signed-byte 8) (*)) index) (simd-pack single-float)
    (flushable movable))
  (define-vop (%vec-widen-sb8-single-sse4)
    (:translate %vec-widen-sb8-single)
    (:guard (member :sse4 *backend-subfeatures*))
    (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
    (:arg-types simple-array-signed-byte-8 tagged-num)
    (:temporary (:sc any-reg) byte-index)
    (:results (result :scs (single-sse-reg)))
    (:result-types simd-pack-single)
    (:generator 4
      (move byte-index index)
      (inst shr byte-index n-fixnum-tag-bits)
      (inst pmovsxbd result (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                      vector byte-index 1))
      (inst cvtdq2ps result result)))
  (define-vop (%vec-widen-sb8-single)
    (:translate %vec-widen-sb8-single)
    (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
    (:arg-types simple-array-signed-byte-8 tagged-num)
    (:temporary (:sc any-reg) byte-index)
    (:results (result :scs (single-sse-reg)))
    (:result-types simd-pack-single)
    (:generator 4
      (move byte-index index)
      (inst shr byte-index n-fixnum-tag-bits)
      (inst movd result (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                      vector byte-index 1))
      (inst punpcklbw result result)
      (inst punpcklwd result result)
      (inst psrad result 24)
      (inst cvtdq2ps result result)))
)

(progn
  (defknown %vec-widen-ub16-single ((simple-array (unsigned-byte 16) (*)) index) (simd-pack single-float)
    (flushable movable))
  (define-vop (%vec-widen-ub16-single-sse4)
    (:translate %vec-widen-ub16-single)
    (:guard (member :sse4 *backend-subfeatures*))
    (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
    (:arg-types simple-array-unsigned-byte-16 tagged-num)
    (:temporary (:sc any-reg) byte-index)
    (:results (result :scs (single-sse-reg)))
    (:result-types simd-pack-single)
    (:generator 4
      (move byte-index index)
      (inst shr byte-index n-fixnum-tag-bits)
      (inst pmovzxwd result (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                      vector byte-index 2))
      (inst cvtdq2ps result result)))
  (define-vop (%vec-widen-ub16-single)
    (:translate %vec-widen-ub16-single)
    (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
    (:arg-types simple-array-unsigned-byte-16 tagged-num)
    (:temporary (:sc any-reg) byte-index)
    (:results (result :scs (single-sse-reg)))
    (:result-types simd-pack-single)
    (:generator 4
      (move byte-index index)
      (inst shr byte-index n-fixnum-tag-bits)
      (inst movq result (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                      vector byte-index 2))
      (inst punpcklwd result result)
      (inst psrld result 16)
      (inst cvtdq2ps result result)))
)

(progn
  (defknown %vec-widen-sb16-single ((simple-array (signed-byte 16) (*)) index) (simd-pack single-float)
    (flushable movable))
  (define-vop (%vec-widen-sb16-single-sse4)
    (:translate %vec-widen-sb16-single)
    (:guard (member :sse4 *backend-subfeatures*))
    (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
    (:arg-types simple-array-signed-byte-16 tagged-num)
    (:temporary (:sc any-reg) byte-index)
    (:results (result :scs (single-sse-reg)))
    (:result-types simd-pack-single)
    (:generator 4
      (move byte-index index)
      (inst shr byte-index n-fixnum-tag-bits)
      (inst pmovsxwd result (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                      vector byte-index 2))
      (inst cvtdq2ps result result)))
  (define-vop (%vec-widen-sb16-single)
    (:translate %vec-widen-sb16-single)
    (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
    (:arg-types simple-array-signed-byte-16 tagged-num)
    (:temporary (:sc any-reg) byte-index)
    (:results (result :scs (single-sse-reg)))
    (:result-types simd-pack-single)
    (:generator 4
      (move byte-index index)
      (inst shr byte-index n-fixnum-tag-bits)
      (inst movq result (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                      vector byte-index 2))
      (inst punpcklwd result result)
      (inst psrad result 16)
      (inst cvtdq2ps result result)))
)

(progn
  (defknown %vec-widen-ub8-double ((simple-array (unsigned-byte 8) (*)) index) (simd-pack double-float)
    (flushable movable))
  (define-vop (%vec-widen-ub8-double)
    (:translate %vec-widen-ub8-double)
    (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
    (:arg-types simple-array-unsigned-byte-8 tagged-num)
    (:temporary (:sc any-reg) byte-index)
    (:temporary (:sc any-reg) word-tmp)
    (:results (result :scs (double-sse-reg)))
    (:result-types simd-pack-double)
    (:generator 4
      (move byte-index index)
      (inst shr byte-index n-fixnum-tag-bits)
      (inst movzx '(:word :dword) word-tmp (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                      vector byte-index 1))
      (inst movd result word-tmp)
      (inst punpcklbw result result)
      (inst punpcklwd result result)
      (inst psrld result 24)
      (inst cvtdq2pd result result)))
)

(progn
  (defknown %vec-widen-sb8-double ((simple-array (signed-byte 8) (*)) index) (simd-pack double-float)
    (flushable movable))
  (define-vop (%vec-widen-sb8-double)
    (:translate %vec-widen-sb8-double)
    (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
    (:arg-types simple-array-signed-byte-8 tagged-num)
    (:temporary (:sc any-reg) byte-index)
    (:temporary (:sc any-reg) word-tmp)
    (:results (result :scs (double-sse-reg)))
    (:result-types simd-pack-double)
    (:generator 4
      (move byte-index index)
      (inst shr byte-index n-fixnum-tag-bits)
      (inst movzx '(:word :dword) word-tmp (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                      vector byte-index 1))
      (inst movd result word-tmp)
      (inst punpcklbw result result)
      (inst punpcklwd result result)
      (inst psrad result 24)
      (inst cvtdq2pd result result)))
)

(progn
  (defknown %vec-widen-ub16-double ((simple-array (unsigned-byte 16) (*)) index) (simd-pack double-float)
    (flushable movable))
  (define-vop (%vec-widen-ub16-double)
    (:translate %vec-widen-ub16-double)
    (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
    (:arg-types simple-array-unsigned-byte-16 tagged-num)
    (:temporary (:sc any-reg) byte-index)
    (:results (result :scs (double-sse-reg)))
    (:result-types simd-pack-double)
    (:generator 4
      (move byte-index index)
      (inst shr byte-index n-fixnum-tag-bits)
      (inst movd result (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                      vector byte-index 2))
      (inst punpcklwd result result)
      (inst psrld result 16)
      (inst cvtdq2pd result result)))
)

(progn
  (defknown %vec-widen-sb16-double ((simple-array (signed-byte 16) (*)) index) (simd-pack double-float)
    (flushable movable))
  (define-vop (%vec-widen-sb16-double)
    (:translate %vec-widen-sb16-double)
    (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
    (:arg-types simple-array-signed-byte-16 tagged-num)
    (:temporary (:sc any-reg) byte-index)
    (:results (result :scs (double-sse-reg)))
    (:result-types simd-pack-double)
    (:generator 4
      (move byte-index index)
      (inst shr byte-index n-fixnum-tag-bits)
      (inst movd result (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                      vector byte-index 2))
      (inst punpcklwd result result)
      (inst psrad result 16)
      (inst cvtdq2pd result result)))
)

;;;; a single-float source read into double lanes: two floats are one MOVQ, and
;;;; CVTPS2PD takes them to two doubles with no rounding, since every single-float
;;;; is exactly a double-float.
;;;
;;;; this vop went missing while the widen section was regenerated, and nothing
;;;; failed until a loop needed it: the defknown above and the table entry below
;;;; had outlived it, which is what a missing vop looks like.  the pass finds the
;;;; entry, emits the call, and the loop dies at run time with an undefined
;;;; function, so tests/auto-vectorize.pure.lisp compiles one now.

(define-vop (%vec-widen-single-double)
  (:translate %vec-widen-single-double)
  (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
  (:arg-types simple-array-single-float tagged-num)
  (:temporary (:sc any-reg) byte-index)
  (:results (result :scs (double-sse-reg)))
  (:result-types simd-pack-double)
  (:generator 4
    (move byte-index index)
    (inst shr byte-index n-fixnum-tag-bits)
    (inst movq result (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                          vector byte-index 4))
    (inst cvtps2pd result result)))

;;; 32-bit minimum and maximum are PMINSD, PMAXSD, PMINUD and PMAXUD, which are
;;; SSE4.1, so like the 32-bit multiply they are published by the build that has
;;; the instructions rather than always.  A build without them declines the loop,
;;; which is what it did before: no 32-bit lane has a packed minimum in SSE2.

(defknown (%vec-min-sb32 %vec-max-sb32)
    ((simd-pack (signed-byte 32)) (simd-pack (signed-byte 32)))
    (simd-pack (signed-byte 32))
  (flushable movable))

(defknown (%vec-min-ub32 %vec-max-ub32)
    ((simd-pack (unsigned-byte 32)) (simd-pack (unsigned-byte 32)))
    (simd-pack (unsigned-byte 32))
  (flushable movable))

(macrolet ((define-vec-min-max-32 (name type instruction)
             `(define-vop (,name)
                (:translate ,name)
                (:args (a :scs (int-sse-reg)) (b :scs (int-sse-reg)))
                (:arg-types ,type ,type)
                (:temporary (:sc int-sse-reg) arith-tmp)
                (:results (result :scs (int-sse-reg)))
                (:result-types ,type)
                (:generator 2
                  (move arith-tmp a)
                  (inst ,instruction arith-tmp b)
                  (move result arith-tmp)))))
  (define-vec-min-max-32 %vec-min-sb32 simd-pack-sb32 pminsd)
  (define-vec-min-max-32 %vec-max-sb32 simd-pack-sb32 pmaxsd)
  (define-vec-min-max-32 %vec-min-ub32 simd-pack-ub32 pminud)
  (define-vec-min-max-32 %vec-max-ub32 simd-pack-ub32 pmaxud))

;;; hand the primitives to the loop vectorizer in SB-C, which cannot name this
;;; package itself; see the comment on SB-C::*VECTOR-PRIMITIVES*.

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
    (publish "MUL" (unsigned-byte 8) %vec-mul-ub8)
    (publish "MUL" (signed-byte 8) %vec-mul-sb8)
    (publish "MUL" (unsigned-byte 8) %vec-mul-ub8)
    (publish "MUL" (signed-byte 8) %vec-mul-sb8)
    (publish "DIV" single-float %vec-div-single)
    (publish "DIV" double-float %vec-div-double)
    (publish "NEG" single-float %vec-negate-single)
    (publish "NEG" double-float %vec-negate-double)
    (publish "INDEX" single-float %vec-splat-index-single)
    (publish "INDEX" double-float %vec-splat-index-double)
    (publish "RAMP" single-float %vec-ramp-single)
    (publish "RAMP" double-float %vec-ramp-double)
    (publish "STRIDE" single-float %vec-stride-single)
    (publish "STRIDE" double-float %vec-stride-double)
    (publish "MUL" (unsigned-byte 16) %vec-mul-ub16)
    (publish "MUL" (signed-byte 16) %vec-mul-sb16)
    (when (member :sse4 *backend-subfeatures*)
      (publish "MUL" (unsigned-byte 32) %vec-mul-ub32)
      (publish "MUL" (signed-byte 32) %vec-mul-sb32)
      (publish "MIN" (unsigned-byte 32) %vec-min-ub32)
      (publish "MAX" (unsigned-byte 32) %vec-max-ub32)
      (publish "MIN" (signed-byte 32) %vec-min-sb32)
      (publish "MAX" (signed-byte 32) %vec-max-sb32))
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
    (publish "WIDEN-SINGLE" double-float %vec-widen-single-double)
    (publish "WIDEN-UB8" single-float %vec-widen-ub8-single)
    (publish "WIDEN-UB8" double-float %vec-widen-ub8-double)
    (publish "WIDEN-SB8" single-float %vec-widen-sb8-single)
    (publish "WIDEN-SB8" double-float %vec-widen-sb8-double)
    (publish "WIDEN-UB16" single-float %vec-widen-ub16-single)
    (publish "WIDEN-UB16" double-float %vec-widen-ub16-double)
    (publish "WIDEN-SB16" single-float %vec-widen-sb16-single)
    (publish "WIDEN-SB16" double-float %vec-widen-sb16-double)
    (publish "MIN" (unsigned-byte 8) %vec-min-ub8)
    (publish "MIN" (unsigned-byte 16) %vec-min-ub16)
    (publish "MAX" (unsigned-byte 16) %vec-max-ub16)
    (publish "MIN" (signed-byte 16) %vec-min-sb16)
    (publish "MAX" (signed-byte 16) %vec-max-sb16)
    (publish "MIN" (signed-byte 8) %vec-min-sb8)
    (publish "MAX" (signed-byte 8) %vec-max-sb8)
    (publish "MAX" (unsigned-byte 8) %vec-max-ub8)
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
