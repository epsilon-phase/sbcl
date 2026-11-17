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
;;; a (unsigned-byte 32) or (signed-byte 32) scalar arrives *tagged*, so the tag
;;; is stripped with an arithmetic shift.  a logical shift looks right and is
;;; not: -3 is tagged -6, whose low 32 bits are #xFFFFFFFA, and shifting that
;;; logically by one gives 2147483645 rather than -3.

;;; true when tn holds a tagged fixnum.  the test is on the primitive type the
;;; register was allocated for, not on the Lisp type of the reference: an integer
;;; literal arrives untagged, a variable declared (signed-byte 32) arrives tagged,
;;; and guessing from the Lisp type splats -3 as -2.
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

;;; one byte in all sixteen lanes: strip the tag, then mask down to the byte,
;;; since the shift leaves a negative value sign-extended and the multiplication
;;; copies the high half as well, turning -3 into #x...FCFCFCFD.  Then copy the
;;; byte into all four bytes of a word with #x01010101 and pshufd the word.
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

;;; as above, for (signed-byte 8).
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
;;; strip the tag with an arithmetic shift, mask to the 16 bits that are left,
;;; multiply by #x00010001 to get the value into both halves of a word, then
;;; movd and pshufd as for the bytes.
;;;
;;; the and is required: sign extension leaves the high half of
;;; -3 * #x00010001 as -4 rather than -3.

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
