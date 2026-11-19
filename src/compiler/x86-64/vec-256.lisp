;;;; Packed 256-bit (AVX2) array access and arithmetic VOPs

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
;;; One name per element type and width, with the width in the name: a 256-bit
;;; pack is its own Lisp type, so a function returning one cannot share a name
;;; with a function returning a 128-bit one, and the defknown cannot describe
;;; both.  The loop vectorizer in SB-C picks the width first and then looks the
;;; primitives up in the table published at the end of this file.
;;;
;;; Nothing is published unless the build enables :AVX2, so a build without it
;;; keeps emitting the 128-bit loops and never reaches these VOPs.

(defknown %vec-load-single-256 ((simple-array single-float (*)) index)
    (simd-pack-256 single-float)
  (flushable))
(defknown %vec-load-double-256 ((simple-array double-float (*)) index)
    (simd-pack-256 double-float)
  (flushable))
(defknown %vec-store-single-256
    ((simd-pack-256 single-float) (simple-array single-float (*)) index)
    (values)
  ())
(defknown %vec-store-double-256
    ((simd-pack-256 double-float) (simple-array double-float (*)) index)
    (values)
  ())

(defknown (%vec-add-single-256 %vec-sub-single-256
           %vec-mul-single-256 %vec-div-single-256)
    ((simd-pack-256 single-float) (simd-pack-256 single-float))
    (simd-pack-256 single-float)
  (flushable movable))
(defknown (%vec-add-double-256 %vec-sub-double-256
           %vec-mul-double-256 %vec-div-double-256)
    ((simd-pack-256 double-float) (simd-pack-256 double-float))
    (simd-pack-256 double-float)
  (flushable movable))
(defknown %vec-negate-single-256 ((simd-pack-256 single-float))
    (simd-pack-256 single-float)
  (flushable movable))
(defknown %vec-negate-double-256 ((simd-pack-256 double-float))
    (simd-pack-256 double-float)
  (flushable movable))

(defknown %vec-splat-single-256 (single-float)
    (simd-pack-256 single-float)
  (flushable movable))
(defknown %vec-splat-double-256 (double-float)
    (simd-pack-256 double-float)
  (flushable movable))

;;;; loading and storing 256 bits at a time
;;;
;;; The effective address shape of the 128-bit section of vec.lisp: the element
;;; size in place of n-word-bytes, and a one-byte element untags the index by
;;; hand because index-scale cannot go below one.

(macrolet ((define-vec-load (name element-type result-type sc element-size instruction)
             `(define-vop (,name)
                (:translate ,name)
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
           (define-vec-store (name element-type result-type sc element-size instruction)
             `(define-vop (,name)
                (:translate ,name)
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
  (define-vec-load %vec-load-single-256 simple-array-single-float
    simd-pack-256-single single-avx2-reg 4 vmovups)
  (define-vec-load %vec-load-double-256 simple-array-double-float
    simd-pack-256-double double-avx2-reg 8 vmovupd)
  (define-vec-store %vec-store-single-256 simple-array-single-float
    simd-pack-256-single single-avx2-reg 4 vmovups)
  (define-vec-store %vec-store-double-256 simple-array-double-float
    simd-pack-256-double double-avx2-reg 8 vmovupd))

(defknown %vec-load-ub8-256 ((simple-array (unsigned-byte 8) (*)) index)
    (simd-pack-256 (unsigned-byte 8))
  (flushable))
(defknown %vec-load-sb8-256 ((simple-array (signed-byte 8) (*)) index)
    (simd-pack-256 (signed-byte 8))
  (flushable))
(defknown %vec-store-ub8-256
    ((simd-pack-256 (unsigned-byte 8)) (simple-array (unsigned-byte 8) (*)) index)
    (values)
  ())
(defknown %vec-store-sb8-256
    ((simd-pack-256 (signed-byte 8)) (simple-array (signed-byte 8) (*)) index)
    (values)
  ())

(macrolet ((define-byte-load (name element-type result-type)
             `(define-vop (,name)
                (:translate ,name)
                (:args (vector :scs (descriptor-reg))
                       (index :scs (any-reg)))
                (:arg-types ,element-type tagged-num)
                (:temporary (:sc any-reg) byte-index)
                (:results (result :scs (int-avx2-reg)))
                (:result-types ,result-type)
                (:generator 3
                  (move byte-index index)
                  (inst shr byte-index n-fixnum-tag-bits)
                  (inst vmovdqu result
                        (ea (- (* vector-data-offset n-word-bytes)
                               other-pointer-lowtag)
                            vector byte-index 1)))))
           (define-byte-store (name element-type result-type)
             `(define-vop (,name)
                (:translate ,name)
                (:args (value :scs (int-avx2-reg))
                       (vector :scs (descriptor-reg))
                       (index :scs (any-reg)))
                (:arg-types ,result-type ,element-type tagged-num)
                (:temporary (:sc any-reg) byte-index)
                (:generator 3
                  (move byte-index index)
                  (inst shr byte-index n-fixnum-tag-bits)
                  (inst vmovdqu
                        (ea (- (* vector-data-offset n-word-bytes)
                               other-pointer-lowtag)
                            vector byte-index 1)
                        value)))))
  (define-byte-load %vec-load-ub8-256 simple-array-unsigned-byte-8 simd-pack-256-ub8)
  (define-byte-load %vec-load-sb8-256 simple-array-signed-byte-8 simd-pack-256-sb8)
  (define-byte-store %vec-store-ub8-256 simple-array-unsigned-byte-8 simd-pack-256-ub8)
  (define-byte-store %vec-store-sb8-256 simple-array-signed-byte-8 simd-pack-256-sb8))

(defknown %vec-load-ub16-256 ((simple-array (unsigned-byte 16) (*)) index)
    (simd-pack-256 (unsigned-byte 16))
  (flushable))
(defknown %vec-load-sb16-256 ((simple-array (signed-byte 16) (*)) index)
    (simd-pack-256 (signed-byte 16))
  (flushable))
(defknown %vec-load-ub32-256 ((simple-array (unsigned-byte 32) (*)) index)
    (simd-pack-256 (unsigned-byte 32))
  (flushable))
(defknown %vec-load-sb32-256 ((simple-array (signed-byte 32) (*)) index)
    (simd-pack-256 (signed-byte 32))
  (flushable))
(defknown %vec-load-ub64-256 ((simple-array (unsigned-byte 64) (*)) index)
    (simd-pack-256 (unsigned-byte 64))
  (flushable))
(defknown %vec-load-sb64-256 ((simple-array (signed-byte 64) (*)) index)
    (simd-pack-256 (signed-byte 64))
  (flushable))
(defknown %vec-store-ub16-256
    ((simd-pack-256 (unsigned-byte 16)) (simple-array (unsigned-byte 16) (*)) index)
    (values)
  ())
(defknown %vec-store-sb16-256
    ((simd-pack-256 (signed-byte 16)) (simple-array (signed-byte 16) (*)) index)
    (values)
  ())
(defknown %vec-store-ub32-256
    ((simd-pack-256 (unsigned-byte 32)) (simple-array (unsigned-byte 32) (*)) index)
    (values)
  ())
(defknown %vec-store-sb32-256
    ((simd-pack-256 (signed-byte 32)) (simple-array (signed-byte 32) (*)) index)
    (values)
  ())
(defknown %vec-store-ub64-256
    ((simd-pack-256 (unsigned-byte 64)) (simple-array (unsigned-byte 64) (*)) index)
    (values)
  ())
(defknown %vec-store-sb64-256
    ((simd-pack-256 (signed-byte 64)) (simple-array (signed-byte 64) (*)) index)
    (values)
  ())

(macrolet ((define-vec-load (name element-type result-type element-size)
             `(define-vop (,name)
                (:translate ,name)
                (:args (vector :scs (descriptor-reg))
                       (index :scs (any-reg)))
                (:arg-types ,element-type tagged-num)
                (:results (result :scs (int-avx2-reg)))
                (:result-types ,result-type)
                (:generator 3
                  (inst vmovdqu result
                        (ea (- (* vector-data-offset n-word-bytes)
                               other-pointer-lowtag)
                            vector index
                            (index-scale ,element-size index))))))
           (define-vec-store (name element-type result-type element-size)
             `(define-vop (,name)
                (:translate ,name)
                (:args (value :scs (int-avx2-reg))
                       (vector :scs (descriptor-reg))
                       (index :scs (any-reg)))
                (:arg-types ,result-type ,element-type tagged-num)
                (:generator 3
                  (inst vmovdqu
                        (ea (- (* vector-data-offset n-word-bytes)
                               other-pointer-lowtag)
                            vector index
                            (index-scale ,element-size index))
                        value)))))
  (define-vec-load %vec-load-ub16-256 simple-array-unsigned-byte-16
    simd-pack-256-ub16 2)
  (define-vec-load %vec-load-sb16-256 simple-array-signed-byte-16
    simd-pack-256-sb16 2)
  (define-vec-load %vec-load-ub32-256 simple-array-unsigned-byte-32
    simd-pack-256-ub32 4)
  (define-vec-load %vec-load-sb32-256 simple-array-signed-byte-32
    simd-pack-256-sb32 4)
  (define-vec-load %vec-load-ub64-256 simple-array-unsigned-byte-64
    simd-pack-256-ub64 8)
  (define-vec-load %vec-load-sb64-256 simple-array-signed-byte-64
    simd-pack-256-sb64 8)
  (define-vec-store %vec-store-ub16-256 simple-array-unsigned-byte-16
    simd-pack-256-ub16 2)
  (define-vec-store %vec-store-sb16-256 simple-array-signed-byte-16
    simd-pack-256-sb16 2)
  (define-vec-store %vec-store-ub32-256 simple-array-unsigned-byte-32
    simd-pack-256-ub32 4)
  (define-vec-store %vec-store-sb32-256 simple-array-signed-byte-32
    simd-pack-256-sb32 4)
  (define-vec-store %vec-store-ub64-256 simple-array-unsigned-byte-64
    simd-pack-256-ub64 8)
  (define-vec-store %vec-store-sb64-256 simple-array-signed-byte-64
    simd-pack-256-sb64 8))

(defknown (%vec-add-ub8-256 %vec-sub-ub8-256 %vec-and-ub8-256 %vec-or-ub8-256 %vec-xor-ub8-256)
    ((simd-pack-256 (unsigned-byte 8)) (simd-pack-256 (unsigned-byte 8)))
    (simd-pack-256 (unsigned-byte 8))
  (flushable movable))
(defknown (%vec-add-sb8-256 %vec-sub-sb8-256 %vec-and-sb8-256 %vec-or-sb8-256 %vec-xor-sb8-256)
    ((simd-pack-256 (signed-byte 8)) (simd-pack-256 (signed-byte 8)))
    (simd-pack-256 (signed-byte 8))
  (flushable movable))
(defknown (%vec-add-ub16-256 %vec-sub-ub16-256 %vec-mul-ub16-256 %vec-and-ub16-256 %vec-or-ub16-256 %vec-xor-ub16-256)
    ((simd-pack-256 (unsigned-byte 16)) (simd-pack-256 (unsigned-byte 16)))
    (simd-pack-256 (unsigned-byte 16))
  (flushable movable))
(defknown (%vec-add-sb16-256 %vec-sub-sb16-256 %vec-mul-sb16-256 %vec-and-sb16-256 %vec-or-sb16-256 %vec-xor-sb16-256)
    ((simd-pack-256 (signed-byte 16)) (simd-pack-256 (signed-byte 16)))
    (simd-pack-256 (signed-byte 16))
  (flushable movable))
(defknown (%vec-add-ub32-256 %vec-sub-ub32-256 %vec-mul-ub32-256 %vec-and-ub32-256 %vec-or-ub32-256 %vec-xor-ub32-256)
    ((simd-pack-256 (unsigned-byte 32)) (simd-pack-256 (unsigned-byte 32)))
    (simd-pack-256 (unsigned-byte 32))
  (flushable movable))
(defknown (%vec-add-sb32-256 %vec-sub-sb32-256 %vec-mul-sb32-256 %vec-and-sb32-256 %vec-or-sb32-256 %vec-xor-sb32-256)
    ((simd-pack-256 (signed-byte 32)) (simd-pack-256 (signed-byte 32)))
    (simd-pack-256 (signed-byte 32))
  (flushable movable))
(defknown (%vec-add-ub64-256 %vec-sub-ub64-256 %vec-and-ub64-256 %vec-or-ub64-256 %vec-xor-ub64-256)
    ((simd-pack-256 (unsigned-byte 64)) (simd-pack-256 (unsigned-byte 64)))
    (simd-pack-256 (unsigned-byte 64))
  (flushable movable))
(defknown (%vec-add-sb64-256 %vec-sub-sb64-256 %vec-and-sb64-256 %vec-or-sb64-256 %vec-xor-sb64-256)
    ((simd-pack-256 (signed-byte 64)) (simd-pack-256 (signed-byte 64)))
    (simd-pack-256 (signed-byte 64))
  (flushable movable))

(macrolet ((define-vec-arith-256 (name type sc instruction)
             `(define-vop (,name)
                (:translate ,name)
                (:args (a :scs (,sc))
                       (b :scs (,sc)))
                (:arg-types ,type ,type)
                (:results (result :scs (,sc)))
                (:result-types ,type)
                (:generator 2
                  (inst ,instruction result a b)))))
  (define-vec-arith-256 %vec-add-single-256 simd-pack-256-single single-avx2-reg vaddps)
  (define-vec-arith-256 %vec-sub-single-256 simd-pack-256-single single-avx2-reg vsubps)
  (define-vec-arith-256 %vec-mul-single-256 simd-pack-256-single single-avx2-reg vmulps)
  (define-vec-arith-256 %vec-div-single-256 simd-pack-256-single single-avx2-reg vdivps)
  (define-vec-arith-256 %vec-add-double-256 simd-pack-256-double double-avx2-reg vaddpd)
  (define-vec-arith-256 %vec-sub-double-256 simd-pack-256-double double-avx2-reg vsubpd)
  (define-vec-arith-256 %vec-mul-double-256 simd-pack-256-double double-avx2-reg vmulpd)
  (define-vec-arith-256 %vec-div-double-256 simd-pack-256-double double-avx2-reg vdivpd))

;;;; packed arithmetic
;;;
;;; VEX encodes three operands, so the source registers are read before the
;;; result is written and no temporary is needed: the failure mode of the
;;; two-operand SSE forms, where a hoisted source could be allocated to the
;;; result register and the operation became a op a, cannot happen here.

(macrolet ((define-vec-arith-256 (name type instruction)
             `(define-vop (,name)
                (:translate ,name)
                (:args (a :scs (int-avx2-reg))
                       (b :scs (int-avx2-reg)))
                (:arg-types ,type ,type)
                (:results (result :scs (int-avx2-reg)))
                (:result-types ,type)
                (:generator 2
                  (inst ,instruction result a b)))))
  (define-vec-arith-256 %vec-add-ub8-256 simd-pack-256-ub8 vpaddb)
  (define-vec-arith-256 %vec-sub-ub8-256 simd-pack-256-ub8 vpsubb)
  (define-vec-arith-256 %vec-and-ub8-256 simd-pack-256-ub8 vpand)
  (define-vec-arith-256 %vec-or-ub8-256 simd-pack-256-ub8 vpor)
  (define-vec-arith-256 %vec-xor-ub8-256 simd-pack-256-ub8 vpxor)
  (define-vec-arith-256 %vec-add-sb8-256 simd-pack-256-sb8 vpaddb)
  (define-vec-arith-256 %vec-sub-sb8-256 simd-pack-256-sb8 vpsubb)
  (define-vec-arith-256 %vec-and-sb8-256 simd-pack-256-sb8 vpand)
  (define-vec-arith-256 %vec-or-sb8-256 simd-pack-256-sb8 vpor)
  (define-vec-arith-256 %vec-xor-sb8-256 simd-pack-256-sb8 vpxor)
  (define-vec-arith-256 %vec-add-ub16-256 simd-pack-256-ub16 vpaddw)
  (define-vec-arith-256 %vec-sub-ub16-256 simd-pack-256-ub16 vpsubw)
  (define-vec-arith-256 %vec-mul-ub16-256 simd-pack-256-ub16 vpmullw)
  (define-vec-arith-256 %vec-and-ub16-256 simd-pack-256-ub16 vpand)
  (define-vec-arith-256 %vec-or-ub16-256 simd-pack-256-ub16 vpor)
  (define-vec-arith-256 %vec-xor-ub16-256 simd-pack-256-ub16 vpxor)
  (define-vec-arith-256 %vec-add-sb16-256 simd-pack-256-sb16 vpaddw)
  (define-vec-arith-256 %vec-sub-sb16-256 simd-pack-256-sb16 vpsubw)
  (define-vec-arith-256 %vec-mul-sb16-256 simd-pack-256-sb16 vpmullw)
  (define-vec-arith-256 %vec-and-sb16-256 simd-pack-256-sb16 vpand)
  (define-vec-arith-256 %vec-or-sb16-256 simd-pack-256-sb16 vpor)
  (define-vec-arith-256 %vec-xor-sb16-256 simd-pack-256-sb16 vpxor)
  (define-vec-arith-256 %vec-add-ub32-256 simd-pack-256-ub32 vpaddd)
  (define-vec-arith-256 %vec-sub-ub32-256 simd-pack-256-ub32 vpsubd)
  (define-vec-arith-256 %vec-mul-ub32-256 simd-pack-256-ub32 vpmulld)
  (define-vec-arith-256 %vec-and-ub32-256 simd-pack-256-ub32 vpand)
  (define-vec-arith-256 %vec-or-ub32-256 simd-pack-256-ub32 vpor)
  (define-vec-arith-256 %vec-xor-ub32-256 simd-pack-256-ub32 vpxor)
  (define-vec-arith-256 %vec-add-sb32-256 simd-pack-256-sb32 vpaddd)
  (define-vec-arith-256 %vec-sub-sb32-256 simd-pack-256-sb32 vpsubd)
  (define-vec-arith-256 %vec-mul-sb32-256 simd-pack-256-sb32 vpmulld)
  (define-vec-arith-256 %vec-and-sb32-256 simd-pack-256-sb32 vpand)
  (define-vec-arith-256 %vec-or-sb32-256 simd-pack-256-sb32 vpor)
  (define-vec-arith-256 %vec-xor-sb32-256 simd-pack-256-sb32 vpxor)
  (define-vec-arith-256 %vec-add-ub64-256 simd-pack-256-ub64 vpaddq)
  (define-vec-arith-256 %vec-sub-ub64-256 simd-pack-256-ub64 vpsubq)
  (define-vec-arith-256 %vec-and-ub64-256 simd-pack-256-ub64 vpand)
  (define-vec-arith-256 %vec-or-ub64-256 simd-pack-256-ub64 vpor)
  (define-vec-arith-256 %vec-xor-ub64-256 simd-pack-256-ub64 vpxor)
  (define-vec-arith-256 %vec-add-sb64-256 simd-pack-256-sb64 vpaddq)
  (define-vec-arith-256 %vec-sub-sb64-256 simd-pack-256-sb64 vpsubq)
  (define-vec-arith-256 %vec-and-sb64-256 simd-pack-256-sb64 vpand)
  (define-vec-arith-256 %vec-or-sb64-256 simd-pack-256-sb64 vpor)
  (define-vec-arith-256 %vec-xor-sb64-256 simd-pack-256-sb64 vpxor))

;;;; negating a packed float: the sign bit set in every lane
;;;
;;; A 256-bit inline constant is one integer, as an :oword is one 128-bit
;;; integer.  The memory operand of a VEX float operation has to be aligned, and
;;; inline constants are aligned to their size class, so a :hword lands on 32.

(define-vop (%vec-negate-single-256)
  (:translate %vec-negate-single-256)
  (:args (a :scs (single-avx2-reg)))
  (:arg-types simd-pack-256-single)
  (:results (result :scs (single-avx2-reg)))
  (:result-types simd-pack-256-single)
  (:generator 2
    (inst vxorps result a
          (register-inline-constant :avx2 #x8000000080000000800000008000000080000000800000008000000080000000))))

(define-vop (%vec-negate-double-256)
  (:translate %vec-negate-double-256)
  (:args (a :scs (double-avx2-reg)))
  (:arg-types simd-pack-256-double)
  (:results (result :scs (double-avx2-reg)))
  (:result-types simd-pack-256-double)
  (:generator 2
    (inst vxorpd result a
          (register-inline-constant :avx2 #x8000000000000000800000000000000080000000000000008000000000000000))))

;;;; splatting one scalar across every lane
;;;
;;; vbroadcastss and vbroadcastsd read the low lane of an xmm.  The integer
;;; broadcasts read the low byte, word, dword or qword, so a narrow lane needs no
;;; mask once the tag is off: the instruction takes the part of the register it
;;; wants.  That is also why the byte splats here do not need the multiply
;;; replicate trick the 128-bit ones use.

(define-vop (%vec-splat-single-256)
  (:translate %vec-splat-single-256)
  (:args (x :scs (single-reg)))
  (:arg-types single-float)
  (:temporary (:sc single-sse-reg) low)
  (:results (result :scs (single-avx2-reg)))
  (:result-types simd-pack-256-single)
  (:generator 2
    (move low x)
    (inst vbroadcastss result low)))

(define-vop (%vec-splat-double-256)
  (:translate %vec-splat-double-256)
  (:args (x :scs (double-reg)))
  (:arg-types double-float)
  (:temporary (:sc double-sse-reg) low)
  (:results (result :scs (double-avx2-reg)))
  (:result-types simd-pack-256-double)
  (:generator 2
    (move low x)
    (inst vbroadcastsd result low)))

(defknown %vec-splat-ub8-256 ((unsigned-byte 8))
    (simd-pack-256 (unsigned-byte 8))
  (flushable movable))
(defknown %vec-splat-sb8-256 ((signed-byte 8))
    (simd-pack-256 (signed-byte 8))
  (flushable movable))
(defknown %vec-splat-ub16-256 ((unsigned-byte 16))
    (simd-pack-256 (unsigned-byte 16))
  (flushable movable))
(defknown %vec-splat-sb16-256 ((signed-byte 16))
    (simd-pack-256 (signed-byte 16))
  (flushable movable))
(defknown %vec-splat-ub32-256 ((unsigned-byte 32))
    (simd-pack-256 (unsigned-byte 32))
  (flushable movable))
(defknown %vec-splat-sb32-256 ((signed-byte 32))
    (simd-pack-256 (signed-byte 32))
  (flushable movable))
(defknown %vec-splat-ub64-256 ((unsigned-byte 64))
    (simd-pack-256 (unsigned-byte 64))
  (flushable movable))
(defknown %vec-splat-sb64-256 ((signed-byte 64))
    (simd-pack-256 (signed-byte 64))
  (flushable movable))
(macrolet ((define-vec-splat-narrow (name type result-type instruction)
             `(define-vop (,name)
                (:translate ,name)
                (:args (x :scs (unsigned-reg signed-reg)))
                (:arg-types ,type)
                (:temporary (:sc int-sse-reg) low)
                (:results (result :scs (int-avx2-reg)))
                (:result-types ,result-type)
                (:generator 3
                  ;; a lane narrower than a word arrives untagged: the backend
                  ;; strips the tag where the value is defined.  the 128-bit
                  ;; splats read the operand as it comes for the same reason.
                  (inst movd low x)
                  (inst ,instruction result low))))
           (define-vec-splat-word (name type result-type instruction)
             `(define-vop (,name)
                (:translate ,name)
                (:args (x :scs (unsigned-reg signed-reg)))
                (:arg-types ,type)
                (:temporary (:sc unsigned-reg) word)
                (:temporary (:sc int-sse-reg) low)
                (:results (result :scs (int-avx2-reg)))
                (:result-types ,result-type)
                (:generator 3
                  (if (tagged-fixnum-tn-p x)
                      (progn (move word x)
                             (inst sar word n-fixnum-tag-bits)
                             (inst movq low word))
                      (inst movq low x))
                  (inst ,instruction result low)))))
  (define-vec-splat-narrow %vec-splat-ub8-256 unsigned-num
    simd-pack-256-ub8 vpbroadcastb)
  (define-vec-splat-narrow %vec-splat-sb8-256 signed-num
    simd-pack-256-sb8 vpbroadcastb)
  (define-vec-splat-narrow %vec-splat-ub16-256 unsigned-num
    simd-pack-256-ub16 vpbroadcastw)
  (define-vec-splat-narrow %vec-splat-sb16-256 signed-num
    simd-pack-256-sb16 vpbroadcastw)
  (define-vec-splat-narrow %vec-splat-ub32-256 unsigned-num
    simd-pack-256-ub32 vpbroadcastd)
  (define-vec-splat-narrow %vec-splat-sb32-256 signed-num
    simd-pack-256-sb32 vpbroadcastd)
  (define-vec-splat-word %vec-splat-ub64-256 unsigned-num
    simd-pack-256-ub64 vpbroadcastq)
  (define-vec-splat-word %vec-splat-sb64-256 signed-num
    simd-pack-256-sb64 vpbroadcastq))

;;;; reading narrow elements into wider 256-bit lanes
;;;
;;; The 128-bit widens in vec.lisp read exactly the elements one iteration
;;; consumes, and these keep that rule at twice the width: a byte source feeds
;;; eight single lanes out of eight bytes, a word source eight lanes out of
;;; sixteen bytes, and a single-float source four double lanes out of sixteen.
;;; The instruction that widens is also the load, so no lane is left holding
;;; whatever was in the register.
;;;
;;; Only the single-float targets get a 256-bit widen here.  A byte or word
;;; source read into four double lanes would have to widen four elements, and
;;; the 256-bit narrowing loads read eight bytes or sixteen - one element wide
;;; of what the loop needs, which on the last iteration would read past the
;;; array.  Those loops keep the 128-bit width, where the read is exact.

(progn
  (defknown %vec-widen-single-double-256 ((simple-array single-float (*)) index)
      (simd-pack-256 double-float)
    (flushable movable))
  (define-vop (%vec-widen-single-double-256)
    (:translate %vec-widen-single-double-256)
    (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
    (:arg-types simple-array-single-float tagged-num)
    (:temporary (:sc any-reg) byte-index)
    (:results (result :scs (double-avx2-reg)))
    (:result-types simd-pack-256-double)
    (:generator 4
      (move byte-index index)
      (inst shr byte-index n-fixnum-tag-bits)
      (inst vcvtps2pd result (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                                 vector byte-index 4)))))

(macrolet ((define-vec-widen-narrow (name element-type arg-type result-spec
                                        pack-type reg-type load-instruction element-size)
             ;; a defknown takes a type specifier and a vop's :result-types takes a
             ;; primitive type name; (simd-pack-256 single-float) is only the first
             `(progn
                (defknown ,name (,arg-type index) ,result-spec
                  (flushable movable))
                (define-vop (,name)
                  (:translate ,name)
                  (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
                  (:arg-types ,element-type tagged-num)
                  (:temporary (:sc any-reg) byte-index)
                  (:results (result :scs (,reg-type)))
                  (:result-types ,pack-type)
                  (:generator 4
                    (move byte-index index)
                    (inst shr byte-index n-fixnum-tag-bits)
                    (inst ,load-instruction
                          result
                          (ea (- (* vector-data-offset n-word-bytes) other-pointer-lowtag)
                              vector byte-index ,element-size))
                    (inst vcvtdq2ps result result))))))
  (define-vec-widen-narrow %vec-widen-ub8-single-256 simple-array-unsigned-byte-8
    (simple-array (unsigned-byte 8) (*)) (simd-pack-256 single-float)
    simd-pack-256-single single-avx2-reg vpmovzxbd 1)
  (define-vec-widen-narrow %vec-widen-sb8-single-256 simple-array-signed-byte-8
    (simple-array (signed-byte 8) (*)) (simd-pack-256 single-float)
    simd-pack-256-single single-avx2-reg vpmovsxbd 1)
  (define-vec-widen-narrow %vec-widen-ub16-single-256 simple-array-unsigned-byte-16
    (simple-array (unsigned-byte 16) (*)) (simd-pack-256 single-float)
    simd-pack-256-single single-avx2-reg vpmovzxwd 2)
  (define-vec-widen-narrow %vec-widen-sb16-single-256 simple-array-signed-byte-16
    (simple-array (signed-byte 16) (*)) (simd-pack-256 single-float)
    simd-pack-256-single single-avx2-reg vpmovsxwd 2))

;;;; packed minimum and maximum
;;;
;;; Every signedness and width a 128-bit loop has here has a one-instruction AVX2
;;; form, where the 128-bit section needs the saturating-difference trick for
;;; unsigned words and a bias for signed bytes.  Nothing is exact beyond these
;;; four element types: 32-bit lanes keep the 128-bit width.

(defknown (%vec-min-ub32-256 %vec-max-ub32-256)
    ((simd-pack-256 (unsigned-byte 32)) (simd-pack-256 (unsigned-byte 32)))
    (simd-pack-256 (unsigned-byte 32))
  (flushable movable))
(defknown (%vec-min-sb32-256 %vec-max-sb32-256)
    ((simd-pack-256 (signed-byte 32)) (simd-pack-256 (signed-byte 32)))
    (simd-pack-256 (signed-byte 32))
  (flushable movable))
(defknown (%vec-min-ub8-256 %vec-max-ub8-256)
    ((simd-pack-256 (unsigned-byte 8)) (simd-pack-256 (unsigned-byte 8)))
    (simd-pack-256 (unsigned-byte 8))
  (flushable movable))
(defknown (%vec-min-sb8-256 %vec-max-sb8-256)
    ((simd-pack-256 (signed-byte 8)) (simd-pack-256 (signed-byte 8)))
    (simd-pack-256 (signed-byte 8))
  (flushable movable))
(defknown (%vec-min-ub16-256 %vec-max-ub16-256)
    ((simd-pack-256 (unsigned-byte 16)) (simd-pack-256 (unsigned-byte 16)))
    (simd-pack-256 (unsigned-byte 16))
  (flushable movable))
(defknown (%vec-min-sb16-256 %vec-max-sb16-256)
    ((simd-pack-256 (signed-byte 16)) (simd-pack-256 (signed-byte 16)))
    (simd-pack-256 (signed-byte 16))
  (flushable movable))

(macrolet ((define-vec-min-max-256 (name type instruction)
             `(define-vop (,name)
                (:translate ,name)
                (:args (a :scs (int-avx2-reg))
                       (b :scs (int-avx2-reg)))
                (:arg-types ,type ,type)
                (:results (result :scs (int-avx2-reg)))
                (:result-types ,type)
                (:generator 2
                  (inst ,instruction result a b)))))
  (define-vec-min-max-256 %vec-min-ub8-256 simd-pack-256-ub8 vpminub)
  (define-vec-min-max-256 %vec-max-ub8-256 simd-pack-256-ub8 vpmaxub)
  (define-vec-min-max-256 %vec-min-sb8-256 simd-pack-256-sb8 vpminsb)
  (define-vec-min-max-256 %vec-max-sb8-256 simd-pack-256-sb8 vpmaxsb)
  (define-vec-min-max-256 %vec-min-ub16-256 simd-pack-256-ub16 vpminuw)
  (define-vec-min-max-256 %vec-max-ub16-256 simd-pack-256-ub16 vpmaxuw)
  (define-vec-min-max-256 %vec-min-sb16-256 simd-pack-256-sb16 vpminsw)
  (define-vec-min-max-256 %vec-max-sb16-256 simd-pack-256-sb16 vpmaxsw)
  ;; 32-bit lanes have no packed minimum in any 128-bit instruction set below
  ;; sse4.1, and the 128-bit forms need it; here VPMINUD, VPMAXUD, VPMINSD and
  ;; VPMAXSD are AVX2, so they are always here
  (define-vec-min-max-256 %vec-min-ub32-256 simd-pack-256-ub32 vpminud)
  (define-vec-min-max-256 %vec-max-ub32-256 simd-pack-256-ub32 vpmaxud)
  (define-vec-min-max-256 %vec-min-sb32-256 simd-pack-256-sb32 vpminsd)
  (define-vec-min-max-256 %vec-max-sb32-256 simd-pack-256-sb32 vpmaxsd))

;;;; handing the primitives to SB-C
;;;
;;; Only when the build enables :AVX2.  The pass asks for a 256-bit primitive by
;;; trying this table first, so leaving it empty is what keeps a build without
;;; the subfeature on the 128-bit loops: no VOP here is reachable otherwise.

(macrolet ((publish (base element-type name)
             `(push (cons (list ,base ',element-type) ',name) primitives)))
  (let ((primitives nil))
    (when (member :avx2 *backend-subfeatures*)
      (publish "LOAD" single-float %vec-load-single-256)
      (publish "LOAD" double-float %vec-load-double-256)
      (publish "STORE" single-float %vec-store-single-256)
      (publish "STORE" double-float %vec-store-double-256)
      (publish "SPLAT" single-float %vec-splat-single-256)
      (publish "SPLAT" double-float %vec-splat-double-256)
      (publish "ADD" single-float %vec-add-single-256)
      (publish "SUB" single-float %vec-sub-single-256)
      (publish "MUL" single-float %vec-mul-single-256)
      (publish "DIV" single-float %vec-div-single-256)
      (publish "NEG" single-float %vec-negate-single-256)
      (publish "ADD" double-float %vec-add-double-256)
      (publish "SUB" double-float %vec-sub-double-256)
      (publish "MUL" double-float %vec-mul-double-256)
      (publish "DIV" double-float %vec-div-double-256)
      (publish "NEG" double-float %vec-negate-double-256)
      (publish "WIDEN-SINGLE" double-float %vec-widen-single-double-256)
      (publish "WIDEN-UB8" single-float %vec-widen-ub8-single-256)
      (publish "WIDEN-SB8" single-float %vec-widen-sb8-single-256)
      (publish "WIDEN-UB16" single-float %vec-widen-ub16-single-256)
      (publish "WIDEN-SB16" single-float %vec-widen-sb16-single-256)
      (publish "MIN" (unsigned-byte 8) %vec-min-ub8-256)
      (publish "MAX" (unsigned-byte 8) %vec-max-ub8-256)
      (publish "MIN" (signed-byte 8) %vec-min-sb8-256)
      (publish "MAX" (signed-byte 8) %vec-max-sb8-256)
      (publish "MIN" (unsigned-byte 16) %vec-min-ub16-256)
      (publish "MAX" (unsigned-byte 16) %vec-max-ub16-256)
      (publish "MIN" (signed-byte 16) %vec-min-sb16-256)
      (publish "MAX" (signed-byte 16) %vec-max-sb16-256)
      (publish "MIN" (unsigned-byte 32) %vec-min-ub32-256)
      (publish "MAX" (unsigned-byte 32) %vec-max-ub32-256)
      (publish "MIN" (signed-byte 32) %vec-min-sb32-256)
      (publish "MAX" (signed-byte 32) %vec-max-sb32-256)
      (publish "LOAD" (unsigned-byte 8) %vec-load-ub8-256)
      (publish "STORE" (unsigned-byte 8) %vec-store-ub8-256)
      (publish "SPLAT" (unsigned-byte 8) %vec-splat-ub8-256)
      (publish "LOAD" (signed-byte 8) %vec-load-sb8-256)
      (publish "STORE" (signed-byte 8) %vec-store-sb8-256)
      (publish "SPLAT" (signed-byte 8) %vec-splat-sb8-256)
      (publish "LOAD" (unsigned-byte 16) %vec-load-ub16-256)
      (publish "STORE" (unsigned-byte 16) %vec-store-ub16-256)
      (publish "SPLAT" (unsigned-byte 16) %vec-splat-ub16-256)
      (publish "LOAD" (signed-byte 16) %vec-load-sb16-256)
      (publish "STORE" (signed-byte 16) %vec-store-sb16-256)
      (publish "SPLAT" (signed-byte 16) %vec-splat-sb16-256)
      (publish "LOAD" (unsigned-byte 32) %vec-load-ub32-256)
      (publish "STORE" (unsigned-byte 32) %vec-store-ub32-256)
      (publish "SPLAT" (unsigned-byte 32) %vec-splat-ub32-256)
      (publish "LOAD" (signed-byte 32) %vec-load-sb32-256)
      (publish "STORE" (signed-byte 32) %vec-store-sb32-256)
      (publish "SPLAT" (signed-byte 32) %vec-splat-sb32-256)
      (publish "LOAD" (unsigned-byte 64) %vec-load-ub64-256)
      (publish "STORE" (unsigned-byte 64) %vec-store-ub64-256)
      (publish "SPLAT" (unsigned-byte 64) %vec-splat-ub64-256)
      (publish "LOAD" (signed-byte 64) %vec-load-sb64-256)
      (publish "STORE" (signed-byte 64) %vec-store-sb64-256)
      (publish "SPLAT" (signed-byte 64) %vec-splat-sb64-256)
      (publish "ADD" (unsigned-byte 8) %vec-add-ub8-256)
      (publish "SUB" (unsigned-byte 8) %vec-sub-ub8-256)
      (publish "AND" (unsigned-byte 8) %vec-and-ub8-256)
      (publish "OR" (unsigned-byte 8) %vec-or-ub8-256)
      (publish "XOR" (unsigned-byte 8) %vec-xor-ub8-256)
      (publish "ADD" (signed-byte 8) %vec-add-sb8-256)
      (publish "SUB" (signed-byte 8) %vec-sub-sb8-256)
      (publish "AND" (signed-byte 8) %vec-and-sb8-256)
      (publish "OR" (signed-byte 8) %vec-or-sb8-256)
      (publish "XOR" (signed-byte 8) %vec-xor-sb8-256)
      (publish "ADD" (unsigned-byte 16) %vec-add-ub16-256)
      (publish "SUB" (unsigned-byte 16) %vec-sub-ub16-256)
      (publish "MUL" (unsigned-byte 16) %vec-mul-ub16-256)
      (publish "AND" (unsigned-byte 16) %vec-and-ub16-256)
      (publish "OR" (unsigned-byte 16) %vec-or-ub16-256)
      (publish "XOR" (unsigned-byte 16) %vec-xor-ub16-256)
      (publish "ADD" (signed-byte 16) %vec-add-sb16-256)
      (publish "SUB" (signed-byte 16) %vec-sub-sb16-256)
      (publish "MUL" (signed-byte 16) %vec-mul-sb16-256)
      (publish "AND" (signed-byte 16) %vec-and-sb16-256)
      (publish "OR" (signed-byte 16) %vec-or-sb16-256)
      (publish "XOR" (signed-byte 16) %vec-xor-sb16-256)
      (publish "ADD" (unsigned-byte 32) %vec-add-ub32-256)
      (publish "SUB" (unsigned-byte 32) %vec-sub-ub32-256)
      (publish "MUL" (unsigned-byte 32) %vec-mul-ub32-256)
      (publish "AND" (unsigned-byte 32) %vec-and-ub32-256)
      (publish "OR" (unsigned-byte 32) %vec-or-ub32-256)
      (publish "XOR" (unsigned-byte 32) %vec-xor-ub32-256)
      (publish "ADD" (signed-byte 32) %vec-add-sb32-256)
      (publish "SUB" (signed-byte 32) %vec-sub-sb32-256)
      (publish "MUL" (signed-byte 32) %vec-mul-sb32-256)
      (publish "AND" (signed-byte 32) %vec-and-sb32-256)
      (publish "OR" (signed-byte 32) %vec-or-sb32-256)
      (publish "XOR" (signed-byte 32) %vec-xor-sb32-256)
      (publish "ADD" (unsigned-byte 64) %vec-add-ub64-256)
      (publish "SUB" (unsigned-byte 64) %vec-sub-ub64-256)
      (publish "AND" (unsigned-byte 64) %vec-and-ub64-256)
      (publish "OR" (unsigned-byte 64) %vec-or-ub64-256)
      (publish "XOR" (unsigned-byte 64) %vec-xor-ub64-256)
      (publish "ADD" (signed-byte 64) %vec-add-sb64-256)
      (publish "SUB" (signed-byte 64) %vec-sub-sb64-256)
      (publish "AND" (signed-byte 64) %vec-and-sb64-256)
      (publish "OR" (signed-byte 64) %vec-or-sb64-256)
      (publish "XOR" (signed-byte 64) %vec-xor-sb64-256)
      )
    (setf sb-c::*vector-primitives-256* (nreverse primitives))))
