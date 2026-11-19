;;;; Packed array access and arithmetic VOPs for the ARM64 vectorizer
;;;;
;;;; This software is part of the SBCL system. See the README file for
;;;; more information.
;;;;
;;;; This software is derived from the CMUCL system, which was
;;;; written at Carnegie Mellon University and released into the
;;;; public domain. The software is in the public domain and is
;;;; provided with absolutely no warranty. See the COPYING and CREDITS
;;;; files for more information.

(in-package "SB-VM")

;;;; what the loop vectorizer in SB-C needs
;;;
;;; the pass itself is backend independent: it recognizes an elementwise loop,
;;; rewrites it to run four single-float (or eight 16-bit) elements at a time,
;;; and emits calls to the primitives listed at the end of this file.  A backend
;;; that publishes no table gets no vectorization, and this is arm64's table.
;;;
;;; every primitive takes and returns a SIMD-PACK of one element type.  The
;;; loop vectorizer emits the calls without checking anything, so the element
;;; type in each name, its DEFKNOWN, its :ARG-TYPES and its :RESULT-TYPES all
;;; have to say the same thing.  The arrays are spelled (simple-array <element>
;;; (*)) where a Lisp type specifier is wanted, and by their primitive type
;;; names in :ARG-TYPES.
;;;
;;; a load is flushable but neither movable nor foldable, and a store is neither
;;; of those either, for the same reason as on x86-64: 128 bits at a time must
;;; not be reordered across a store into the same array, because a loop that
;;; reads what it just wrote depends on that order.

;;; NEON loads and stores address memory through a general-purpose register, and
;;; the base register of an array access holds the tag, so the byte offset is
;;; computed here: the index arrives tagged, one tag bit counts two units per
;;; element, and the data offset is small enough to be an immediate.
(defun emit-vec-byte-offset (offset index element-size)
  (aver (member element-size '(1 2 4 8 16)))
  (if (= element-size 1)
      (inst asr offset index n-fixnum-tag-bits)
      (inst lsl offset index (1- (integer-length (1- element-size)))))
  (inst add offset offset (- (* vector-data-offset n-word-bytes)
                            other-pointer-lowtag)))

;;; a DEFKNOWN is a claim about Lisp types, while a VOP's :ARG-TYPES name
;;; primitive types, and the two spellings differ: the pack of a primitive type is
;;; (SIMD-PACK <element>), and SIMD-PACK-UB8 itself names no Lisp type.
(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun vec-pack-ctype (name)
    (ecase name
      (simd-pack-single '(simd-pack single-float))
      (simd-pack-double '(simd-pack double-float))
      (simd-pack-ub8 '(simd-pack (unsigned-byte 8)))
      (simd-pack-ub16 '(simd-pack (unsigned-byte 16)))
      (simd-pack-ub32 '(simd-pack (unsigned-byte 32)))
      (simd-pack-ub64 '(simd-pack (unsigned-byte 64)))
      (simd-pack-sb8 '(simd-pack (signed-byte 8)))
      (simd-pack-sb16 '(simd-pack (signed-byte 16)))
      (simd-pack-sb32 '(simd-pack (signed-byte 32)))
      (simd-pack-sb64 '(simd-pack (signed-byte 64))))))

;;; the Lisp type of a simple array of one of those elements, for a DEFKNOWN; the
;;; :ARG-TYPES of a VOP name the primitive type instead.
(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun vec-array-ctype (name)
    (ecase name
      (simple-array-single-float '(simple-array single-float (*)))
      (simple-array-double-float '(simple-array double-float (*)))
      (simple-array-unsigned-byte-8 '(simple-array (unsigned-byte 8) (*)))
      (simple-array-signed-byte-8 '(simple-array (signed-byte 8) (*)))
      (simple-array-unsigned-byte-16 '(simple-array (unsigned-byte 16) (*)))
      (simple-array-signed-byte-16 '(simple-array (signed-byte 16) (*)))
      (simple-array-unsigned-byte-32 '(simple-array (unsigned-byte 32) (*)))
      (simple-array-signed-byte-32 '(simple-array (signed-byte 32) (*)))
      (simple-array-unsigned-byte-64 '(simple-array (unsigned-byte 64) (*)))
      (simple-array-signed-byte-64 '(simple-array (signed-byte 64) (*))))))

;;;; function signatures

(defknown (%vec-negate-single) ((simd-pack single-float))
    (simd-pack single-float)
  (flushable movable))
(defknown (%vec-negate-double) ((simd-pack double-float))
    (simd-pack double-float)
  (flushable movable))

(defknown (%vec-splat-single) (single-float)
    (simd-pack single-float)
  (flushable movable))
(defknown (%vec-splat-double) (double-float)
    (simd-pack double-float)
  (flushable movable))

(defknown (%vec-splat-index-single) (index) (simd-pack single-float) (flushable movable))
(defknown (%vec-splat-index-double) (index) (simd-pack double-float) (flushable movable))
(defknown (%vec-ramp-single) () (simd-pack single-float) (flushable movable))
(defknown (%vec-ramp-double) () (simd-pack double-float) (flushable movable))
(defknown (%vec-stride-single) () (simd-pack single-float) (flushable movable))
(defknown (%vec-stride-double) () (simd-pack double-float) (flushable movable))

;;;; loading and storing 128 bits at a time
;;;
;;; LDR and STR move a whole Q register when the destination is one, which is
;;; what an element slice of a simple array is: the guard the vectorizer emits
;;; covers a whole vector, so reading a vector at a time cannot leave the array.
;;;
;;; both address memory by a byte offset rather than by the element index: one
;;; byte per element cannot be scaled, a tagged index counts two units per
;;; element, and an iteration touches every array at the same index, so the two
;;; instructions that turn that index into an offset are worth spending once and
;;; sharing.  %VEC-OFFSET-* computes it, and the loop vectorizer emits it once
;;; per iteration for a backend that publishes it under the OFFSET base.

(macrolet ((define-vec-load (name array-type pack-type sc)
             `(progn
                (defknown (,name) (,(vec-array-ctype array-type) (unsigned-byte 64))
                    ,(vec-pack-ctype pack-type)
                  (flushable))
                (define-vop (,name)
                (:translate ,name)
                (:args (vector :scs (descriptor-reg))
                       (offset :scs (unsigned-reg)))
                (:arg-types ,array-type unsigned-byte-64)
                (:results (result :scs (,sc)))
                (:result-types ,pack-type)
                (:generator 2
                  (inst ldr result (@ vector offset))))))
           (define-vec-store (name pack-type array-type sc)
             `(progn
                (defknown (,name) (,(vec-pack-ctype pack-type) ,(vec-array-ctype array-type)
                                   (unsigned-byte 64))
                    (values)
                  ())
                (define-vop (,name)
                (:translate ,name)
                (:args (value :scs (,sc))
                       (vector :scs (descriptor-reg))
                       (offset :scs (unsigned-reg)))
                (:arg-types ,pack-type ,array-type unsigned-byte-64)
                (:generator 2
                  (inst str value (@ vector offset))))))
           (define-vec-offset (name element-size)
             `(progn
                (defknown (,name) (index) (unsigned-byte 64) (flushable movable))
                (define-vop (,name)
                  (:translate ,name)
                  (:args (index :scs (any-reg)))
                  ;; the pass passes the loop's index, or an index plus a
                  ;; displacement it has guarded to be non-negative, so a fixnum
                  ;; is all this may claim: claiming POSITIVE-FIXNUM would add a
                  ;; tag check to a variable subscript's offset on every iteration
                  (:arg-types tagged-num)
                  (:results (offset :scs (unsigned-reg)))
                  (:result-types unsigned-byte-64)
                  (:generator 2
                    (emit-vec-byte-offset offset index ,element-size))))))
  (define-vec-offset %vec-offset-single 4)
  (define-vec-offset %vec-offset-double 8)
  (define-vec-offset %vec-offset-ub8 1)
  (define-vec-offset %vec-offset-sb8 1)
  (define-vec-offset %vec-offset-ub16 2)
  (define-vec-offset %vec-offset-sb16 2)
  (define-vec-offset %vec-offset-ub32 4)
  (define-vec-offset %vec-offset-sb32 4)
  (define-vec-offset %vec-offset-ub64 8)
  (define-vec-offset %vec-offset-sb64 8)

  (define-vec-load %vec-load-single simple-array-single-float simd-pack-single
    single-neon-reg)
  (define-vec-load %vec-load-double simple-array-double-float simd-pack-double
    double-neon-reg)
  (define-vec-store %vec-store-single simd-pack-single simple-array-single-float
    single-neon-reg)
  (define-vec-store %vec-store-double simd-pack-double simple-array-double-float
    double-neon-reg)

  (define-vec-load %vec-load-ub8 simple-array-unsigned-byte-8 simd-pack-ub8
    int-neon-reg)
  (define-vec-load %vec-load-sb8 simple-array-signed-byte-8 simd-pack-sb8
    int-neon-reg)
  (define-vec-load %vec-load-ub16 simple-array-unsigned-byte-16 simd-pack-ub16
    int-neon-reg)
  (define-vec-load %vec-load-sb16 simple-array-signed-byte-16 simd-pack-sb16
    int-neon-reg)
  (define-vec-load %vec-load-ub32 simple-array-unsigned-byte-32 simd-pack-ub32
    int-neon-reg)
  (define-vec-load %vec-load-sb32 simple-array-signed-byte-32 simd-pack-sb32
    int-neon-reg)
  (define-vec-load %vec-load-ub64 simple-array-unsigned-byte-64 simd-pack-ub64
    int-neon-reg)
  (define-vec-load %vec-load-sb64 simple-array-signed-byte-64 simd-pack-sb64
    int-neon-reg)

  (define-vec-store %vec-store-ub8 simd-pack-ub8 simple-array-unsigned-byte-8
    int-neon-reg)
  (define-vec-store %vec-store-sb8 simd-pack-sb8 simple-array-signed-byte-8
    int-neon-reg)
  (define-vec-store %vec-store-ub16 simd-pack-ub16 simple-array-unsigned-byte-16
    int-neon-reg)
  (define-vec-store %vec-store-sb16 simd-pack-sb16 simple-array-signed-byte-16
    int-neon-reg)
  (define-vec-store %vec-store-ub32 simd-pack-ub32 simple-array-unsigned-byte-32
    int-neon-reg)
  (define-vec-store %vec-store-sb32 simd-pack-sb32 simple-array-signed-byte-32
    int-neon-reg)
  (define-vec-store %vec-store-ub64 simd-pack-ub64 simple-array-unsigned-byte-64
    int-neon-reg)
  (define-vec-store %vec-store-sb64 simd-pack-sb64 simple-array-signed-byte-64
    int-neon-reg))

;;;; packed arithmetic
;;;
;;; a three-operand instruction writes a register that is not one of its
;;; sources, and NEON reads both sources before writing the destination, so the
;;; result may be allocated over an argument and no temporary is needed.
;;; FADD/FSUB/FMUL/FDIV take their lane size in the suffix, and ADD/SUB of
;;; integer lanes wrap at the lane width, which is what the element type means.

(macrolet ((define-vec-float-arith (name type sc instruction size)
             `(progn
                (defknown (,name) (,(vec-pack-ctype type) ,(vec-pack-ctype type))
                    ,(vec-pack-ctype type)
                  (flushable movable))
                (define-vop (,name)
                (:translate ,name)
                (:args (a :scs (,sc))
                       (b :scs (,sc)))
                (:arg-types ,type ,type)
                (:results (result :scs (,sc)))
                (:result-types ,type)
                (:generator 2
                  (inst ,instruction result a b ,size))))))
  (define-vec-float-arith %vec-add-single simd-pack-single single-neon-reg fadd :4s)
  (define-vec-float-arith %vec-sub-single simd-pack-single single-neon-reg fsub :4s)
  (define-vec-float-arith %vec-mul-single simd-pack-single single-neon-reg fmul :4s)
  (define-vec-float-arith %vec-div-single simd-pack-single single-neon-reg fdiv :4s)
  (define-vec-float-arith %vec-add-double simd-pack-double double-neon-reg fadd :2d)
  (define-vec-float-arith %vec-sub-double simd-pack-double double-neon-reg fsub :2d)
  (define-vec-float-arith %vec-mul-double simd-pack-double double-neon-reg fmul :2d)
  (define-vec-float-arith %vec-div-double simd-pack-double double-neon-reg fdiv :2d))

;;; (- x) on a float flips the sign bit rather than subtracting, so that 0.0
;;; becomes -0.0 and a nan keeps its payload: FNEG is that instruction.
(define-vop (%vec-negate-single)
  (:translate %vec-negate-single)
  (:args (a :scs (single-neon-reg)))
  (:arg-types simd-pack-single)
  (:results (result :scs (single-neon-reg)))
  (:result-types simd-pack-single)
  (:generator 2
    (inst fneg result a :4s)))

(define-vop (%vec-negate-double)
  (:translate %vec-negate-double)
  (:args (a :scs (double-neon-reg)))
  (:arg-types simd-pack-double)
  (:results (result :scs (double-neon-reg)))
  (:result-types simd-pack-double)
  (:generator 2
    (inst fneg result a :2d)))

;;;; packed integer arithmetic and bitwise operations

(macrolet ((define-vec-int-arith (name type instruction size)
             `(progn
                (defknown (,name) (,(vec-pack-ctype type) ,(vec-pack-ctype type))
                    ,(vec-pack-ctype type)
                  (flushable movable))
                (define-vop (,name)
                (:translate ,name)
                (:args (a :scs (int-neon-reg))
                       (b :scs (int-neon-reg)))
                (:arg-types ,type ,type)
                (:results (result :scs (int-neon-reg)))
                (:result-types ,type)
                (:generator 2
                  (inst ,instruction result a b ,size))))))
  (define-vec-int-arith %vec-add-ub8 simd-pack-ub8 add :16b)
  (define-vec-int-arith %vec-add-sb8 simd-pack-sb8 add :16b)
  (define-vec-int-arith %vec-sub-ub8 simd-pack-ub8 sub :16b)
  (define-vec-int-arith %vec-sub-sb8 simd-pack-sb8 sub :16b)
  (define-vec-int-arith %vec-and-ub8 simd-pack-ub8 and :16b)
  (define-vec-int-arith %vec-and-sb8 simd-pack-sb8 and :16b)
  (define-vec-int-arith %vec-or-ub8 simd-pack-ub8 orr :16b)
  (define-vec-int-arith %vec-or-sb8 simd-pack-sb8 orr :16b)
  (define-vec-int-arith %vec-xor-ub8 simd-pack-ub8 eor :16b)
  (define-vec-int-arith %vec-xor-sb8 simd-pack-sb8 eor :16b)

  (define-vec-int-arith %vec-add-ub16 simd-pack-ub16 add :8h)
  (define-vec-int-arith %vec-add-sb16 simd-pack-sb16 add :8h)
  (define-vec-int-arith %vec-sub-ub16 simd-pack-ub16 sub :8h)
  (define-vec-int-arith %vec-sub-sb16 simd-pack-sb16 sub :8h)
  (define-vec-int-arith %vec-and-ub16 simd-pack-ub16 and :16b)
  (define-vec-int-arith %vec-and-sb16 simd-pack-sb16 and :16b)
  (define-vec-int-arith %vec-or-ub16 simd-pack-ub16 orr :16b)
  (define-vec-int-arith %vec-or-sb16 simd-pack-sb16 orr :16b)
  (define-vec-int-arith %vec-xor-ub16 simd-pack-ub16 eor :16b)
  (define-vec-int-arith %vec-xor-sb16 simd-pack-sb16 eor :16b)

  (define-vec-int-arith %vec-add-ub32 simd-pack-ub32 add :4s)
  (define-vec-int-arith %vec-add-sb32 simd-pack-sb32 add :4s)
  (define-vec-int-arith %vec-sub-ub32 simd-pack-ub32 sub :4s)
  (define-vec-int-arith %vec-sub-sb32 simd-pack-sb32 sub :4s)
  (define-vec-int-arith %vec-and-ub32 simd-pack-ub32 and :16b)
  (define-vec-int-arith %vec-and-sb32 simd-pack-sb32 and :16b)
  (define-vec-int-arith %vec-or-ub32 simd-pack-ub32 orr :16b)
  (define-vec-int-arith %vec-or-sb32 simd-pack-sb32 orr :16b)
  (define-vec-int-arith %vec-xor-ub32 simd-pack-ub32 eor :16b)
  (define-vec-int-arith %vec-xor-sb32 simd-pack-sb32 eor :16b)

  (define-vec-int-arith %vec-add-ub64 simd-pack-ub64 add :2d)
  (define-vec-int-arith %vec-add-sb64 simd-pack-sb64 add :2d)
  (define-vec-int-arith %vec-sub-ub64 simd-pack-ub64 sub :2d)
  (define-vec-int-arith %vec-sub-sb64 simd-pack-sb64 sub :2d)
  (define-vec-int-arith %vec-and-ub64 simd-pack-ub64 and :16b)
  (define-vec-int-arith %vec-and-sb64 simd-pack-sb64 and :16b)
  (define-vec-int-arith %vec-or-ub64 simd-pack-ub64 orr :16b)
  (define-vec-int-arith %vec-or-sb64 simd-pack-sb64 orr :16b)
  (define-vec-int-arith %vec-xor-ub64 simd-pack-ub64 eor :16b)
  (define-vec-int-arith %vec-xor-sb64 simd-pack-sb64 eor :16b))

;;; MUL of integer lanes gives the low half of each product, which is what the
;;; scalar loop's value is once it is stored in the element type.  There is no
;;; 64-bit form short of the multiply-long instructions, so a 64-bit multiply
;;; loop stays scalar: NEON multiplies only the narrower lanes.

(macrolet ((define-vec-mul (name type size)
             `(progn
                (defknown (,name) (,(vec-pack-ctype type) ,(vec-pack-ctype type))
                    ,(vec-pack-ctype type)
                  (flushable movable))
                (define-vop (,name)
                (:translate ,name)
                (:args (a :scs (int-neon-reg))
                       (b :scs (int-neon-reg)))
                (:arg-types ,type ,type)
                (:results (result :scs (int-neon-reg)))
                (:result-types ,type)
                (:generator 3
                  (inst mul result a b ,size))))))
  (define-vec-mul %vec-mul-ub8 simd-pack-ub8 :16b)
  (define-vec-mul %vec-mul-sb8 simd-pack-sb8 :16b)
  (define-vec-mul %vec-mul-ub16 simd-pack-ub16 :8h)
  (define-vec-mul %vec-mul-sb16 simd-pack-sb16 :8h)
  (define-vec-mul %vec-mul-ub32 simd-pack-ub32 :4s)
  (define-vec-mul %vec-mul-sb32 simd-pack-sb32 :4s))

;;; minimum and maximum.  Unlike x86-64's SSE2, every lane width and both
;;; signednesses have a packed form here, except 64-bit lanes.

(macrolet ((define-vec-min-max (name type instruction size)
             `(progn
                (defknown (,name) (,(vec-pack-ctype type) ,(vec-pack-ctype type))
                    ,(vec-pack-ctype type)
                  (flushable movable))
                (define-vop (,name)
                (:translate ,name)
                (:args (a :scs (int-neon-reg))
                       (b :scs (int-neon-reg)))
                (:arg-types ,type ,type)
                (:results (result :scs (int-neon-reg)))
                (:result-types ,type)
                (:generator 2
                  (inst ,instruction result a b ,size))))))
  (define-vec-min-max %vec-min-ub8 simd-pack-ub8 umin :16b)
  (define-vec-min-max %vec-max-ub8 simd-pack-ub8 umax :16b)
  (define-vec-min-max %vec-min-sb8 simd-pack-sb8 smin :16b)
  (define-vec-min-max %vec-max-sb8 simd-pack-sb8 smax :16b)
  (define-vec-min-max %vec-min-ub16 simd-pack-ub16 umin :8h)
  (define-vec-min-max %vec-max-ub16 simd-pack-ub16 umax :8h)
  (define-vec-min-max %vec-min-sb16 simd-pack-sb16 smin :8h)
  (define-vec-min-max %vec-max-sb16 simd-pack-sb16 smax :8h)
  (define-vec-min-max %vec-min-ub32 simd-pack-ub32 umin :4s)
  (define-vec-min-max %vec-max-ub32 simd-pack-ub32 umax :4s)
  (define-vec-min-max %vec-min-sb32 simd-pack-sb32 smin :4s)
  (define-vec-min-max %vec-max-sb32 simd-pack-sb32 smax :4s))

;;;; splatting one element across every lane

;;; a float arrives in an S or D register and DUP (element) reads lane zero of
;;; it, so one instruction broadcasts it.
(define-vop (%vec-splat-single)
  (:translate %vec-splat-single)
  (:args (x :scs (single-reg)))
  (:arg-types single-float)
  (:results (result :scs (single-neon-reg)))
  (:result-types simd-pack-single)
  (:generator 2
    (inst dup result x :4s 0)))

(define-vop (%vec-splat-double)
  (:translate %vec-splat-double)
  (:args (x :scs (double-reg)))
  (:arg-types double-float)
  (:results (result :scs (double-neon-reg)))
  (:result-types simd-pack-double)
  (:generator 2
    (inst dup result x :2d 0)))

;;; an integer arrives as an untagged word: the argument's storage class is one
;;; of the untagged ones, and the move into it is what strips the tag, whatever
;;; the primitive type the value came from says.  DUP (general) replicates the
;;; low lane of the register, which is the value truncated to the lane width -
;;; the same bits a scalar store into such an element would keep.
(macrolet ((define-vec-splat (name arg-type primitive-type result-type size)
             `(progn
                (defknown (,name) (,arg-type) ,(vec-pack-ctype result-type)
                    (flushable movable))
                (define-vop (,name)
                  (:translate ,name)
                  (:args (x :scs (unsigned-reg signed-reg)))
                  (:arg-types ,primitive-type)
                  (:results (result :scs (int-neon-reg)))
                  (:result-types ,result-type)
                  (:generator 2
                    (inst dup result x ,size))))))
  (define-vec-splat %vec-splat-ub8 (unsigned-byte 8) unsigned-num simd-pack-ub8 :16b)
  (define-vec-splat %vec-splat-sb8 (signed-byte 8) signed-num simd-pack-sb8 :16b)
  (define-vec-splat %vec-splat-ub16 (unsigned-byte 16) unsigned-num simd-pack-ub16 :8h)
  (define-vec-splat %vec-splat-sb16 (signed-byte 16) signed-num simd-pack-sb16 :8h)
  (define-vec-splat %vec-splat-ub32 (unsigned-byte 32) unsigned-num simd-pack-ub32 :4s)
  (define-vec-splat %vec-splat-sb32 (signed-byte 32) signed-num simd-pack-sb32 :4s)
  (define-vec-splat %vec-splat-ub64 (unsigned-byte 64) unsigned-num simd-pack-ub64 :2d)
  (define-vec-splat %vec-splat-sb64 (signed-byte 64) signed-num simd-pack-sb64 :2d))

;;;; a lane vector holding the loop index: [i, i+1, ... ]
;;;
;;; the scalar index is converted to float and broadcast, and a constant ramp
;;; supplies the lane offsets.  SCVTF takes a signed 32-bit integer per lane on
;;; the vector path, which no loop over a real array exceeds, but the scalar
;;; conversion is a 64-bit one and is exact, so it is used instead: an array
;;; this pass cannot address is not worth a wrong answer.
;;;
;;; 1.0f0 is #x3F800000, 2.0f0 #x40000000, 3.0f0 #x40400000, 1.0d0 is
;;; #x3FF0000000000000.

(define-vop (%vec-splat-index-single)
  (:translate %vec-splat-index-single)
  (:args (index :scs (any-reg)))
  (:arg-types tagged-num)
  (:temporary (:sc signed-reg) word)
  (:temporary (:sc double-reg) converted)
  (:temporary (:sc single-reg) narrowed)
  (:results (result :scs (single-neon-reg)))
  (:result-types simd-pack-single)
  (:generator 4
    (inst asr word index n-fixnum-tag-bits)
    (inst scvtf converted word)
    (inst fcvt narrowed converted)
    (inst dup result narrowed :4s 0)))

(define-vop (%vec-splat-index-double)
  (:translate %vec-splat-index-double)
  (:args (index :scs (any-reg)))
  (:arg-types tagged-num)
  (:temporary (:sc signed-reg) word)
  (:temporary (:sc double-reg) converted)
  (:results (result :scs (double-neon-reg)))
  (:result-types simd-pack-double)
  (:generator 3
    (inst asr word index n-fixnum-tag-bits)
    (inst scvtf converted word)
    (inst dup result converted :2d 0)))

(define-vop (%vec-ramp-single)
  (:translate %vec-ramp-single)
  (:results (result :scs (single-neon-reg)))
  (:result-types simd-pack-single)
  (:generator 1
    (load-inline-constant
     result :oword (logior (ash #x3F800000 32)
                           (ash #x40000000 64)
                           (ash #x40400000 96)))))

(define-vop (%vec-ramp-double)
  (:translate %vec-ramp-double)
  (:results (result :scs (double-neon-reg)))
  (:result-types simd-pack-double)
  (:generator 1
    (load-inline-constant
     result :oword (ash #x3FF0000000000000 64))))

;;; the width of a vector in every lane, which is how much the index pack
;;; advances: 4.0f0 is #x40800000 and 2.0d0 is #x4000000000000000.
(define-vop (%vec-stride-single)
  (:translate %vec-stride-single)
  (:results (result :scs (single-neon-reg)))
  (:result-types simd-pack-single)
  (:generator 1
    (load-inline-constant
     result :oword (logior #x40800000 (ash #x40800000 32)
                           (ash #x40800000 64) (ash #x40800000 96)))))

(define-vop (%vec-stride-double)
  (:translate %vec-stride-double)
  (:results (result :scs (double-neon-reg)))
  (:result-types simd-pack-double)
  (:generator 1
    (load-inline-constant
     result :oword (logior #x4000000000000000
                           (ash #x4000000000000000 64)))))

;;;; reading narrow integers into float lanes: a byte image or 16-bit audio
;;;;
;;; each reads exactly the elements one float iteration consumes (four for
;;; singles, two for doubles), so the length guard the pass already emits for
;;; the float loop covers the read.  The load is the narrow one - LDR B, H, S or
;;; D zeroes the rest of the register - and USHLL/SSHLL widen the lanes a step
;;; at a time; SCVTF then converts them, and a single-float source is a plain
;;; FCVTL, which is exact because every single-float is a double-float.

(progn
  (defknown %vec-widen-single-double ((simple-array single-float (*)) index)
      (simd-pack double-float)
    (flushable movable))
  (define-vop (%vec-widen-single-double)
    (:translate %vec-widen-single-double)
    (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
    (:arg-types simple-array-single-float tagged-num)
    (:temporary (:sc non-descriptor-reg) offset)
    (:results (result :scs (double-neon-reg)))
    (:result-types simd-pack-double)
    (:generator 4
      (emit-vec-byte-offset offset index 4)
      (inst ldr result (@ vector offset) :d)
      (inst fcvtl result result))))

(macrolet ((define-vec-widen (name primitive-type result-type element-size
                                 load-width long-size long-source flags)
             `(progn
                (defknown (,name) (,(vec-array-ctype primitive-type) index)
                    ,(vec-pack-ctype result-type)
                  (flushable movable))
                (define-vop (,name)
                  (:translate ,name)
                  (:args (vector :scs (descriptor-reg)) (index :scs (any-reg)))
                  (:arg-types ,primitive-type tagged-num)
                  (:temporary (:sc non-descriptor-reg) offset)
                  (:results (result :scs (,(if (eq result-type 'simd-pack-double)
                                               'double-neon-reg
                                               'single-neon-reg))))
                  (:result-types ,result-type)
                  (:generator 6
                    (emit-vec-byte-offset offset index ,element-size)
                    (inst ldr result (@ vector offset) ,load-width)
                    ,@(when long-size
                        `((inst ,(if flags 'sshll 'ushll)
                                result ,long-size result ,long-source)))
                    (inst ,(if flags 'sshll 'ushll) result :4s result :4h)
;;; SCVTF to two doubles reads 64-bit integer lanes, so two 32-bit lanes are
;;; widened one step further before the conversion; four singles convert from
;;; the 32-bit lanes directly.
                    ,@(when (eq result-type 'simd-pack-double)
                        `((inst ,(if flags 'sshll 'ushll) result :2d result :2s)))
                    (inst scvtf result result ,(if (eq result-type 'simd-pack-double)
                                                   :2d
                                                   :4s)))))))
  (define-vec-widen %vec-widen-ub8-single
    simple-array-unsigned-byte-8 simd-pack-single 1 :s :8h :8b nil)
  (define-vec-widen %vec-widen-sb8-single
    simple-array-signed-byte-8 simd-pack-single 1 :s :8h :8b t)
  (define-vec-widen %vec-widen-ub16-single
    simple-array-unsigned-byte-16 simd-pack-single 2 :d nil nil nil)
  (define-vec-widen %vec-widen-sb16-single
    simple-array-signed-byte-16 simd-pack-single 2 :d nil nil t)
  (define-vec-widen %vec-widen-ub8-double
    simple-array-unsigned-byte-8 simd-pack-double 1 :h :8h :8b nil)
  (define-vec-widen %vec-widen-sb8-double
    simple-array-signed-byte-8 simd-pack-double 1 :h :8h :8b t)
  (define-vec-widen %vec-widen-ub16-double
    simple-array-unsigned-byte-16 simd-pack-double 2 :s nil nil nil)
  (define-vec-widen %vec-widen-sb16-double
    simple-array-signed-byte-16 simd-pack-double 2 :s nil nil t))

;;;; hand the primitives to the loop vectorizer in SB-C, which cannot name this
;;; package itself; see the comment on SB-C::*VECTOR-PRIMITIVES*.  What is left
;;; out is what NEON does not have packed: a 64-bit multiply, and minimum or
;;; maximum of 64-bit lanes.

(macrolet ((publish (base element-type name)
             `(push (cons (list ,base ',element-type) ',name) primitives)))
  (let ((primitives nil))
    (publish "OFFSET" single-float %vec-offset-single)
    (publish "OFFSET" double-float %vec-offset-double)
    (publish "OFFSET" (unsigned-byte 8) %vec-offset-ub8)
    (publish "OFFSET" (signed-byte 8) %vec-offset-sb8)
    (publish "OFFSET" (unsigned-byte 16) %vec-offset-ub16)
    (publish "OFFSET" (signed-byte 16) %vec-offset-sb16)
    (publish "OFFSET" (unsigned-byte 32) %vec-offset-ub32)
    (publish "OFFSET" (signed-byte 32) %vec-offset-sb32)
    (publish "OFFSET" (unsigned-byte 64) %vec-offset-ub64)
    (publish "OFFSET" (signed-byte 64) %vec-offset-sb64)
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
    (publish "WIDEN-SINGLE" double-float %vec-widen-single-double)
    (publish "WIDEN-UB8" single-float %vec-widen-ub8-single)
    (publish "WIDEN-UB8" double-float %vec-widen-ub8-double)
    (publish "WIDEN-SB8" single-float %vec-widen-sb8-single)
    (publish "WIDEN-SB8" double-float %vec-widen-sb8-double)
    (publish "WIDEN-UB16" single-float %vec-widen-ub16-single)
    (publish "WIDEN-UB16" double-float %vec-widen-ub16-double)
    (publish "WIDEN-SB16" single-float %vec-widen-sb16-single)
    (publish "WIDEN-SB16" double-float %vec-widen-sb16-double)
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
    (publish "MUL" (unsigned-byte 8) %vec-mul-ub8)
    (publish "MUL" (signed-byte 8) %vec-mul-sb8)
    (publish "AND" (unsigned-byte 8) %vec-and-ub8)
    (publish "AND" (signed-byte 8) %vec-and-sb8)
    (publish "OR" (unsigned-byte 8) %vec-or-ub8)
    (publish "OR" (signed-byte 8) %vec-or-sb8)
    (publish "XOR" (unsigned-byte 8) %vec-xor-ub8)
    (publish "XOR" (signed-byte 8) %vec-xor-sb8)
    (publish "MIN" (unsigned-byte 8) %vec-min-ub8)
    (publish "MAX" (unsigned-byte 8) %vec-max-ub8)
    (publish "MIN" (signed-byte 8) %vec-min-sb8)
    (publish "MAX" (signed-byte 8) %vec-max-sb8)
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
    (publish "MUL" (unsigned-byte 16) %vec-mul-ub16)
    (publish "MUL" (signed-byte 16) %vec-mul-sb16)
    (publish "AND" (unsigned-byte 16) %vec-and-ub16)
    (publish "AND" (signed-byte 16) %vec-and-sb16)
    (publish "OR" (unsigned-byte 16) %vec-or-ub16)
    (publish "OR" (signed-byte 16) %vec-or-sb16)
    (publish "XOR" (unsigned-byte 16) %vec-xor-ub16)
    (publish "XOR" (signed-byte 16) %vec-xor-sb16)
    (publish "MIN" (unsigned-byte 16) %vec-min-ub16)
    (publish "MAX" (unsigned-byte 16) %vec-max-ub16)
    (publish "MIN" (signed-byte 16) %vec-min-sb16)
    (publish "MAX" (signed-byte 16) %vec-max-sb16)
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
    (publish "MUL" (unsigned-byte 32) %vec-mul-ub32)
    (publish "MUL" (signed-byte 32) %vec-mul-sb32)
    (publish "MIN" (unsigned-byte 32) %vec-min-ub32)
    (publish "MAX" (unsigned-byte 32) %vec-max-ub32)
    (publish "MIN" (signed-byte 32) %vec-min-sb32)
    (publish "MAX" (signed-byte 32) %vec-max-sb32)
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
