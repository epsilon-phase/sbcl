;;;; automatic vectorization of elementwise array loops
;;;;
;;;; This file rewrites the classic elementwise array loop
;;;;
;;;;   (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))
;;;;
;;;; into a loop over SIMD packs plus a scalar tail, keeping the original loop as
;;;; the fallback when an array is shorter than the count.  It runs inside IR1
;;;; conversion, before macroexpansion, on the DOTIMES, DO and LOOP forms it
;;;; recognizes by name, so a loop that a macro expands to is not rewritten.  The
;;;; types are the compiler's own: the ctype of each variable's leaf in the lexenv
;;;; the form is converted in.  What it produces is a replacement form, which IR1
;;;; converts like any other.
;;;;
;;;; This software is part of the SBCL system. See the README file for
;;;; more information.
;;;;
;;;; This software is derived from the CMUCL system, which was
;;;; written at Carnegie Mellon University and released into the
;;;; public domain. The software is in the public domain and is
;;;; provided with absolutely no warranty. See the COPYING and CREDITS
;;;; files for more information.

(in-package "SB-C")

;;; Set to NIL to compile without automatic vectorization.
(defvar *auto-vectorize* t)

;;; when true, a vectorized loop notes the element type and the number of lanes
(defvar *auto-vectorize-notes* t)

;;; the width the expansion in progress is emitting for, 16 or 32 bytes
(defvar *vector-lane-bytes* 16)
;;; the function a backend defines when the width should be chosen by the machine
;;; rather than by the build: NIL means the build fixed it, and the pass then emits
;;; only the widest loop the build published
(defvar *vector-wide-predicate* nil)

;;;; decoding types

(defun vector-float-element-type-p (element-type)
  (memq element-type '(single-float double-float)))

;;; how many bytes one element takes
(defun vector-element-size (element-type)
  (cond ((eq element-type 'single-float) 4)
        ((eq element-type 'double-float) 8)
        (t (ceiling (second element-type) 8))))

;;; how many elements the backend handles at a time.  a backend that published
;;; 256-bit primitives is asked for those first, so the width is what the
;;; primitives the loop needs came to, not a property of the element type.
(defun vector-width (element-type &optional (bytes *vector-lane-bytes*))
  (/ bytes (vector-element-size element-type)))

;;; integer lanes are only equivalent to the scalar loop at (safety 0), where the
;;; scalar store wraps too; the pass reads it from the lexenv when it matches a loop
;;; and carries it on the plan, so a nested locally is honoured.
;;; the primitive base name for operator over element-type, or nil where the backend
;;; has no such vector.  mul and div are float only, the bitwise ones integer only.
(defun vector-operator-base (operator element-type)
  (let ((floats (vector-float-element-type-p element-type)))
    (case operator
      (+ "ADD")
      (- "SUB")
      ;; which widths have a multiply is the primitive table's business: pmullw
      ;; covers 16-bit lanes at any build, pmulld and vpmulld need sse4.1 or avx2,
      ;; and nothing multiplies 64-bit lanes short of avx512
      (* "MUL")
      (/ (and floats "DIV"))
      (logand (and (not floats) "AND"))
      (logior (and (not floats) "OR"))
      (logxor (and (not floats) "XOR"))
      (:negate (and floats "NEG"))
      (:index (and floats "INDEX"))
      ;; minimum and maximum exist for the narrower integer lanes only, which the
      ;; primitive table decides: pminsw is sse2, the byte and unsigned-word forms
      ;; are reached with a bias or a saturating subtract
      (min (and (not floats) "MIN"))
      (max (and (not floats) "MAX")))))

;;; the element types the backend can move and compute a vector at a time, in the
;;; order the classifier below tries them.
(defparameter +vector-element-types+
  '(single-float double-float
    (signed-byte 8) (unsigned-byte 8)
    (signed-byte 16) (unsigned-byte 16)
    (signed-byte 32) (unsigned-byte 32)
    (signed-byte 64) (unsigned-byte 64)))

;;; the ctype of a variable as the compiler resolved it: declarations, the casts,
;;; let-bound make-array results and inferred types all arrive here.
(defun vector-variable-ctype (variable lexenv)
  (multiple-value-bind (var found) (lexenv-find variable vars :lexenv lexenv)
    (when (and found (lambda-var-p var))
      (lambda-var-type var))))

;;; what ctype says about a value this pass can use, as
;;;   (:array element-type . simple-p)  an array to load and store elements of
;;;   (:scalar . element-type)          a value of that type, to be broadcast
;;; or nil.  simple arrays are tried first: (vector t) and (array t (*)) say what
;;; the elements are but not that the data is inline.
(defun vector-ctype-info (ctype)
  (when (and ctype (neq ctype *universal-type*))
    (or (loop for element-type in +vector-element-types+
              when (csubtypep ctype (specifier-type `(simple-array ,element-type (*))))
                return (list* :array element-type t))
        (loop for element-type in +vector-element-types+
              when (csubtypep ctype (specifier-type `(vector ,element-type)))
                return (list* :array element-type nil))
        (loop for element-type in +vector-element-types+
              when (csubtypep ctype (specifier-type element-type))
                return (cons :scalar element-type)))))

;;; (VALUES KIND TYPE): KIND is :ARRAY with TYPE (ELEMENT-TYPE . SIMPLE-P), or
;;; :SCALAR with TYPE an element type, or NIL when the variable is not usable.
(defun vector-variable-info (variable lexenv)
  (let ((info (vector-ctype-info (vector-variable-ctype variable lexenv))))
    (if info
        (values (car info) (cdr info))
        (values nil nil))))

(defun vector-unwrap-the (form)
  (let ((type nil))
    (loop while (and (consp form) (eq (car form) 'the))
          do (setf type (second form) form (third form)))
    (values form type)))

;;; (values name element-type simple-p) for an array reference.  a the cast supplies
;;; the type itself, anything else takes it from the variable's own ctype.
(defun vector-array-type (form lexenv)
  (multiple-value-bind (inner spec) (vector-unwrap-the form)
    (let* ((ctype (if spec
                      (specifier-type spec)
                      (and (symbolp inner) (vector-variable-ctype inner lexenv))))
           (info (vector-ctype-info ctype)))
      (if (and info (eq (car info) :array))
          (values inner (second info) (cddr info))
          (values nil nil nil)))))

;;;; Recognizing a loop

;;; what a recognized loop turned into.  expression is a tree of
;;;   (:array name)     a W element slice read from a declared array
;;;   (:scalar sym)     a loop-invariant value, broadcast
;;;   (:const . number) a constant, broadcast
;;;   (:op operator subexpression ...) elementwise arithmetic
(defstruct (vector-loop (:conc-name vloop-))
  element-type index count stores sources expression intermediates body fallback
  accumulator unchecked-p)

;;; a bound the generated code can evaluate without side effects: a variable, a
;;; literal, or arithmetic on one of those.
(defun vector-pure-count-p (count)
  (cond ((symbolp count) t)
        ((typep count 'unsigned-byte) t)
        ((and (consp count) (or (eq (car count) '-) (eq (car count) '+)))
         (and (= (length count) 3)
              (symbolp (second count))
              (typep (third count) 'integer)))
        ((and (consp count) (memq (car count) '(1- 1+)))
         (and (= (length count) 2) (symbolp (second count))))
        (t nil)))

;;; an integer expression whose value fits the element type whatever the inputs, so
;;; that narrowing it into the array cannot signal at any safety and the loop does
;;; not need (safety 0) the way integer addition and multiplication do.  Loads,
;;; broadcasts and constants are exact; so is min and max; and so is every bitwise
;;; operation, since a bitwise operation on n-bit values has an n-bit result.
(defun vector-exact-expression-p (expression)
  (case (car expression)
    ((:array :scalar :const) t)
    (:op (and (memq (second expression) '(min max logand logior logxor))
              (every #'vector-exact-expression-p (cddr expression))))
    ((:negate :index) nil)))

;;; the offset of a subscript from the loop variable: (aref a i) is 0,
;;; (aref a (+ i 3)) and (aref a (+ 3 i)) are 3.  a variable offset can be negative
;;; at run time, so the generated guard checks it; a literal is checked here.
(defun vector-subscript-offset (subscript index)
  "The constant displacement of SUBSCRIPT from INDEX, and whether it is a literal.
(AREF A I) is 0, (AREF A (+ I 3)) is 3, (AREF A (- I 1)) is -1.  A negative literal
is refused by the callers, since the scalar loop would read before the array.  A
non-literal offset is used only when it is a variable: it ends up in a guard outside
the loop, so it may not mention the index."
  (cond ((eq subscript index) (values 0 t))
        ((and (consp subscript) (member (car subscript) '(+ -)))
         (let ((arguments (cdr subscript)))
           (when (and (= (length arguments) 2)
                      (eq (first arguments) index))
             (let ((other (second arguments)))
               (cond ((typep other 'integer)
                      (values (if (eq (car subscript) '-) (- other) other) t))
                     ((and (eq (car subscript) '+)
                           (symbolp other)
                           (not (eq other index)))
                      (values other nil)))))))
        ;; (+ 3 i) as well as (+ i 3)
        ((and (consp subscript) (eq (car subscript) '+)
              (= (length (cdr subscript)) 2)
              (typep (first (cdr subscript)) 'integer)
              (eq (second (cdr subscript)) index))
         (values (first (cdr subscript)) t))))

;;; Match an elementwise expression over N-element slices, or return NIL.
(defun vector-match-expression (form env index element-type &optional substitutions)
  (cond ((numberp form)
;;; an integer constant has to fit in a fixnum: anything wider reaches the backend
;;; as a bignum, which no splat accepts.
         (when (and (typep form element-type)
                    (or (vector-float-element-type-p element-type)
                        (typep form 'fixnum)))
           (list :const form)))
        ((symbolp form)
         ;; a LET* binding of the loop body stands for the expression it is bound
         ;; to: the stores below see that expression, or a name for it when they
         ;; share it and it is worth computing once.
         (or (cdr (assoc form substitutions))
             ;; the loop index shadows whatever the enclosing scope says about that
             ;; name, so it is never a broadcast operand
             (unless (eq form index)
               (multiple-value-bind (kind type) (vector-variable-info form env)
                 (when (and (eq kind :scalar) (equal type element-type))
                   (list :scalar form))))))
        ((atom form) nil)
        ((eq (car form) 'the)
         ;; only a THE that pins the element type exactly is usable
         (when (equal (second form) element-type)
           (vector-match-expression (third form) env index element-type substitutions)))
        ;; the loop index itself, converted to the element type: a value that
        ;; depends on the index, like a gradient or a ramp table
        ((and (eq (car form) 'float) (null (cddr form))
              (eq (second form) index)
              (equal element-type 'single-float))
         (list :index))
        ((and (eq (car form) 'coerce) (= (length form) 3) (eq (second form) index)
              (consp (third form)) (eq (car (third form)) 'quote)
              (equal (second (third form)) element-type)
              (vector-float-element-type-p element-type))
         (list :index))
        ((eq (car form) 'aref)
         (destructuring-bind (array . subscripts) (cdr form)
           (when (null (cdr subscripts))
             (multiple-value-bind (offset literal-p)
                 (vector-subscript-offset (car subscripts) index)
               (when offset
                 (multiple-value-bind (name type simple-p) (vector-array-type array env)
                   (cond ((equal type element-type)
                          (list :array name offset simple-p literal-p))
                         ;; a narrower integer source read into a float lane, which
                         ;; only the simple-array widen vops handle
                         ((and simple-p
                               (vector-widen-base type element-type)
                               (vector-primitive (vector-widen-base type element-type)
                                                 element-type))
                          (list :widen (list :array name offset simple-p literal-p)
                                type)))))))))
;;; (- x) on a float flips the sign bit, which is not (- 0 x): the two disagree on
;;; zeroes, so floats get their own packed negation and integers subtract from zero.
        ((and (eq (car form) '-) (null (cdr (cdr form))))
         (let ((subexpression (vector-match-expression (second form) env index element-type substitutions)))
           (when subexpression
             (if (vector-float-element-type-p element-type)
                 (list :negate subexpression)
                 (list :op '- (list :const 0) subexpression)))))
        ((memq (car form) '(+ - * / logand logior logxor min max))
         (let ((arguments (cdr form)))
           (when (and (cdr arguments) (vector-operator-base (car form) element-type))
             (let ((subexpressions
                     (mapcar (lambda (argument)
                               (vector-match-expression argument env index element-type
                                                        substitutions))
                             arguments)))
               (when (every #'identity subexpressions)
                 (cons :op (cons (car form) subexpressions)))))))
        (t nil)))

;;; Match (SETF (AREF name index) expression) writing an element type slice.
;;; (INCF place delta) and (DECF place delta) are the stores the matcher
;;; understands, with the destination on the right of the operator.
(defun vector-normalize-store (form)
  (if (and (consp form) (memq (car form) '(incf decf)))
      (let ((place (second form))
            (delta (if (cddr form) (third form) 1)))
        `(setf ,place (,(if (eq (car form) 'incf) '+ '-) ,place ,delta)))
      form))

(defun vector-match-element-store (form env index &optional substitutions element-type)
  (let ((form (vector-normalize-store form)))
    (when (and (consp form) (eq (car form) 'setf) (null (cdddr form)))
    (let ((place (second form)))
      (when (and (consp place) (eq (car place) 'aref))
        (destructuring-bind (array . subscripts) (cdr place)
          (when (null (cdr subscripts))
            (multiple-value-bind (offset literal-p)
                (vector-subscript-offset (car subscripts) index)
              (when offset
                (multiple-value-bind (name store-type simple-p) (vector-array-type array env)
                  (when (and name (or (null element-type) (equal store-type element-type)))
                    (let ((expression
                            (vector-match-expression (third form) env index store-type
                                                     substitutions)))
                        (when expression
                          (values name store-type expression offset
                                  simple-p literal-p))))))))))))))

;;; How many times SYMBOL appears in FORM.
(defun vector-count-symbol (form symbol)
  (cond ((eq form symbol) 1)
        ((consp form) (reduce #'+ (mapcar (lambda (part) (vector-count-symbol part symbol))
                                          form)))
        (t 0)))

;;; The bindings and the stores of a loop body, with the body flattened:
;;; (let* ((v init) ...) (setf ...) (setf ...))  is  bindings ((v init) ...) and
;;; the stores it surrounds.  A SETF of several places is split the way the
;;; scalar loop would perform it, left to right.
(defun vector-flatten-body (statements)
  (let ((bindings '()) (stores '()))
    (labels ((walk (forms)
               (dolist (form forms)
                 (let ((form (vector-normalize-store form)))
                   (cond ((and (consp form) (memq (car form) '(let let*)))
                          (let ((pairs (second form)))
                            (when (or (null pairs)
                                      (not (every (lambda (pair)
                                                    (and (consp pair) (= (length pair) 2)
                                                         (symbolp (first pair))))
                                                  pairs)))
                              (return-from vector-flatten-body nil))
                            (dolist (pair pairs) (push pair bindings))
                            ;; a LET* that declares the type of its own bindings is
                            ;; fine - those declarations are what the pass reads the
                            ;; types from - but one that changes what the body means
                            ;; is not
                            (multiple-value-bind (declarations statements)
                                (vector-split-declarations (cddr form))
                              (when (vector-special-declaration-p declarations)
                                (return-from vector-flatten-body nil))
                              (walk statements))))
                         ((and (consp form) (eq (car form) 'setf))
                          (when (oddp (length (cdr form)))
                            (return-from vector-flatten-body nil))
                          (loop for (place value) on (cdr form) by #'cddr
                                do (push `(setf ,place ,value) stores)))
                         (t (return-from vector-flatten-body nil)))))))
      (walk statements)
      (values (nreverse bindings) (nreverse stores)))))

;;; Match a loop body as element stores into arrays of one element type.  A body
;;; may write several of them - a butterfly writes four elements across a real
;;; and an imaginary vector, a complex multiply two - and may bind intermediates
;;; in a LET*; one that the stores share is bound once in the emitted loop rather
;;; than recomputed, since the expressions are trees.
;;; The form a LET* binding stands for, following the bindings before it: a body
;;; that writes (aref re j) with (j (+ k half)) is addressing k + HALF.
(defun vector-resolve-index-form (form index-substitutions)
  (if (symbolp form)
      (let ((entry (assoc form index-substitutions :test #'eq)))
        (if entry (vector-resolve-index-form (cdr entry) index-substitutions) form))
      form))

(defun vector-substitute-subscripts (form index-substitutions)
  "Replace a LET*-bound index with the arithmetic it stands for, in every AREF."
  (cond ((atom form) form)
        ((eq (car form) 'aref)
         (cons 'aref
               (cons (second form)
                     (mapcar (lambda (subscript)
                               (vector-resolve-index-form subscript index-substitutions))
                             (cddr form)))))
        (t (mapcar (lambda (subform)
                     (vector-substitute-subscripts subform index-substitutions))
                   form))))

;;; The element type of the array a store writes to, without matching its value:
;;; the store matcher needs the type before it can match the bindings the value
;;; is built from.
(defun vector-store-place-type (form env)
  (let ((form (vector-normalize-store form)))
    (when (and (consp form) (eq (car form) 'setf) (null (cdddr form))
               (consp (second form)) (eq (car (second form)) 'aref))
      (destructuring-bind (array . subscripts) (cdr (second form))
        (when (null (cdr subscripts))
          (nth-value 1 (vector-array-type array env)))))))

(defun vector-match-stores (statements env index)
  (multiple-value-bind (bindings stores) (vector-flatten-body statements)
    (when (and stores
               ;; every store must write a slice of one and the same element type
               (let ((type (vector-store-place-type (first stores) env)))
                 (and type
                      (every (lambda (store) (equal (vector-store-place-type store env) type))
                             stores)
                      type)))
     ;; a binding that is an index - (j (+ k half)) - is not a value the stores
     ;; compute; it is arithmetic the subscript matcher should see directly.
     (let ((index-substitutions '()) (value-bindings '()))
       (dolist (binding bindings)
         (let ((form (vector-resolve-index-form (second binding) index-substitutions)))
           (if (vector-subscript-offset form index)
               (push (cons (first binding) form) index-substitutions)
               (push binding value-bindings))))
       (setf stores (mapcar (lambda (store)
                              (vector-substitute-subscripts store index-substitutions))
                            stores))
       (setf bindings
             (mapcar (lambda (binding)
                       (list (first binding)
                             (vector-substitute-subscripts (second binding)
                                                           index-substitutions)))
                     (nreverse value-bindings)))
       (let ((element-type (vector-store-place-type (first stores) env))
             (substitutions '()) (intermediates '()) (matched '()))
         (flet ((uses (symbol)
                  (reduce #'+ (append (mapcar (lambda (store)
                                                (vector-count-symbol store symbol))
                                              stores)
                                      (mapcar (lambda (binding)
                                                (vector-count-symbol (second binding) symbol))
                                              bindings)))))
           (dolist (binding bindings)
             (let* ((var (first binding))
                    (tree (vector-match-expression (second binding) env index
                                                   element-type substitutions)))
               (unless tree (return-from vector-match-stores nil))
               (if (> (uses var) 1)
                   (progn (push (list var tree) intermediates)
                          (push (cons var (list :intermediate var)) substitutions))
                   (push (cons var tree) substitutions))))
           (dolist (store stores)
             (multiple-value-bind (store-name store-type expression offset
                                   simple-p literal-p)
                 (vector-match-element-store store env index substitutions element-type)
               (unless (and store-name (equal store-type element-type))
                 (return-from vector-match-stores nil))
               (push (list store-name offset simple-p literal-p expression) matched)))
           (values element-type (nreverse matched) (nreverse intermediates))))))))

;;; Split leading DECLARE forms from a body.
(defun vector-split-declarations (forms)
  (let ((declarations '()))
    (loop while (and (consp forms) (consp (car forms)) (eq (caar forms) 'declare))
          do (push (pop forms) declarations))
    (values (nreverse declarations) forms)))

(defun vector-special-declaration-p (declarations)
  "Whether a loop body's declarations forbid vectorizing it, or change what its
code means.  SPECIAL and TRULY-THE refer to bindings and types this pass would
have to reproduce; an OPTIMIZE declaration sets the policy of the scalar store,
which is not the policy the loop form itself is compiled under, so a body that
declares (SAFETY 3) must not be rewritten as if it were unchecked."
  (some (lambda (declaration)
          (some (lambda (spec) (memq (car spec) '(special sb-ext:truly-the optimize)))
                (cdr declaration)))
        declarations))

;;; Match (DOTIMES (index count) (setf (aref destination index) expression)).
(defun vector-match-dotimes (form env)
  (when (and (consp form) (eq (car form) 'dotimes)
             (consp (cdr form)) (consp (cadr form)))
    (destructuring-bind (spec . body) (cdr form)
      (when (and (consp spec) (null (cddr spec))) ; no result form: returns NIL
        (destructuring-bind (index count) spec
          (when (and (symbolp index)
                     (vector-pure-count-p count))
            (multiple-value-bind (declarations statements) (vector-split-declarations body)
              (when (not (vector-special-declaration-p declarations))
                (multiple-value-bind (element-type stores intermediates)
                    (vector-match-stores statements env index)
                  (when element-type
                    (make-vector-loop
                     :element-type element-type
                     :index index
                     :count count
                     :stores stores
                     :sources (vector-stores-sources stores intermediates)
                     :intermediates intermediates
                     :expression (fifth (first stores))
                     :body body
                     :fallback form :unchecked-p (policy env (= safety 0)))))))))))))

;;; the arrays read by an expression tree, as an alist of name to (offsets . simple-p)
;;; in order of first use: the guard has to cover every offset an array is read at.
;;; the primitive base that widens SOURCE-TYPE into TARGET-TYPE, or NIL when the
;;; backend has no such widen.  A byte or 16-bit array read by a float loop is the
;;; common case: an image buffer or 16-bit audio consumed as floats.
(defun vector-pack-type (element-type)
  (if (eql *vector-lane-bytes* 32)
      `(simd-pack-256 ,element-type)
      `(simd-pack ,element-type)))

(defun vector-widen-base (source-type target-type)
  ;; a single-float source read into double lanes is a precision increase with no
  ;; rounding and no range to prove, so it needs nothing but the convert
  (cond ((and (eq source-type 'single-float) (eq target-type 'double-float))
         "WIDEN-SINGLE")
        ((and (vector-float-element-type-p target-type)
              (member source-type
                      '((unsigned-byte 8) (signed-byte 8)
                        (unsigned-byte 16) (signed-byte 16))
                      :test #'equal))
         (format nil "WIDEN-~a~a"
                 (if (eq (car source-type) 'signed-byte) "SB" "UB")
                 (second source-type)))))

;;; Replace each (:INTERMEDIATE var) leaf by the expression VAR stands for, so that
;;; the walkers see the work the loop really does.  The emitted code keeps the
;;; marker, because there the value is computed once and reused.
(defun vector-inline-intermediates (expression intermediates)
  (when (eq (car expression) :intermediate)
    (return-from vector-inline-intermediates
      (vector-inline-intermediates
       (second (assoc (second expression) intermediates :test #'eq)) intermediates)))
  (case (car expression)
    (:op (cons :op (cons (second expression)
                         (mapcar (lambda (subexpression)
                                   (vector-inline-intermediates subexpression intermediates))
                                 (cddr expression)))))
    (:negate (list :negate (vector-inline-intermediates (second expression) intermediates)))
    (:widen (list :widen (vector-inline-intermediates (second expression) intermediates)
                  (third expression)))
    (t expression)))

;;; Every array every store of a body reads, with the offsets it reads them at.
(defun vector-stores-sources (stores &optional intermediates (accumulator '()))
  (dolist (store stores accumulator)
    (setf accumulator
          (vector-collect-arrays
           (if intermediates
               (vector-inline-intermediates (fifth store) intermediates)
               (fifth store))
           accumulator))))

(defun vector-collect-arrays (expression &optional (accumulator '()))
  (ecase (car expression)
    (:array (let* ((name (second expression))
                   (entry (assoc name accumulator))
                   (offset (third expression)))
              (if entry
                  (progn (pushnew offset (car (cdr entry)) :test #'equal) accumulator)
                  (append accumulator
                          (list (cons name
                                      (cons (list offset) (fourth expression))))))))
    (:op (dolist (subexpression (cddr expression) accumulator)
           (setf accumulator (vector-collect-arrays subexpression accumulator))))
    (:widen (vector-collect-arrays (second expression) accumulator))
    (:negate (vector-collect-arrays (second expression) accumulator))
    ((:scalar :const :index) accumulator)))

;;; The loop invariant parts of an expression tree, in order of first use.
(defun vector-collect-scalars (expression &optional (accumulator '()))
  (case (car expression)
    (:array accumulator)
    (:op (dolist (subexpression (cddr expression) accumulator)
           (setf accumulator (vector-collect-scalars subexpression accumulator))))
    (:widen (vector-collect-scalars (second expression) accumulator))
    (:negate (vector-collect-scalars (second expression) accumulator))
    (:index accumulator)
    ((:scalar :const) (adjoin expression accumulator :test #'equal))
    (t accumulator)))

;;; the number of elements a loop clause covers: below b is b, upto b is b+1, and
;;; (1+ (1- n)) is n, so that upto (1- n) is written as the count n.  The count
;;; reaches the generated guard and both loops.
(defun vector-loop-count (bound upto)
  (cond ((not upto) bound)
        ((and (consp bound)
              (or (and (eq (car bound) '1-) (null (cddr bound)))
                  (and (eq (car bound) '-) (null (cdddr bound)) (eql (third bound) 1))))
         ;; (1+ (1- N)) and (1+ (- N 1)) are N
         (second bound))
        (t `(1+ ,bound))))

;;; A LOOP clause keyword: the symbols in a LOOP form are the user's own, so
;;; they are matched by name rather than by identity.
(defun vector-loop-keyword (form name)
  (and (symbolp form) (string-equal (symbol-name form) name)))

;;; match the loop shapes that mean the same thing as a dotimes over an elementwise
;;; body:
;;;
;;;   (loop for index [of-type type] [from 0] below|upto count do body)
;;;
;;; anything else is left alone: a nonzero start, a to bound, several body forms,
;;; another clause.
(defun vector-match-loop (form env)
  (when (and (consp form) (eq (car form) 'loop))
    (let ((clauses (cdr form)))
      (when (and (or (vector-loop-keyword (car clauses) "FOR")
                     (vector-loop-keyword (car clauses) "AS"))
                 (symbolp (second clauses)))
        (let ((index (second clauses)))
          (setf clauses (cddr clauses))
          (when (vector-loop-keyword (car clauses) "OF-TYPE")
            (setf clauses (cddr clauses)))
          (when (vector-loop-keyword (car clauses) "FROM")
            (unless (equal (second clauses) 0)
              (return-from vector-match-loop nil))
            (setf clauses (cddr clauses)))
          (when (or (vector-loop-keyword (car clauses) "BELOW")
                    (vector-loop-keyword (car clauses) "UPTO"))
;;; below n stops at n-1 and upto n stops at n, so the count differs; a negative
;;; offset is known here only when it is a literal.
            (let* ((upto (and (vector-loop-keyword (car clauses) "UPTO") t))
                   (count (vector-loop-count (second clauses) upto)))
              (setf clauses (cddr clauses))
              (when (and (vector-loop-keyword (car clauses) "DO")
                         (null (cddr clauses))
                         (vector-pure-count-p count))
                (multiple-value-bind (element-type stores intermediates)
                    (vector-match-stores (list (second clauses)) env index)
                  (when element-type
                    (make-vector-loop
                     :element-type element-type
                     :index index
                     :count count
                     :stores stores
                     :sources (vector-stores-sources stores intermediates)
                     :intermediates intermediates
                     :expression (fifth (first stores))
                     :body (list (second clauses))
                     :fallback form :unchecked-p (policy env (= safety 0)))))))))))))

;;; (1+ index) or (+ index 1) in a DO loop's step.
(defun vector-increments-index-p (step index)
  (or (and (consp step) (eq (car step) '1+) (null (cddr step)) (eq (second step) index))
      (and (consp step) (eq (car step) '+)
           (let ((arguments (cdr step)))
             (and (= (length arguments) 2)
                  (member index arguments :test #'eq)
                  (eql (if (eq (first arguments) index)
                           (second arguments)
                           (first arguments))
                       1))))))

;;; Match the DO loops that mean the same thing as a DOTIMES over an elementwise
;;; body:
;;;
;;;   (do ((index 0 (1+ index))) ((>= index count)) body)
;;;
;;; Anything else is left alone: several bindings, a different step, result forms,
;;; another end test.
(defun vector-match-do (form env)
  (when (and (consp form) (memq (car form) '(do do*))
             (consp (cdr form)) (consp (cddr form)))
    (destructuring-bind (bindings end-clause . body) (cdr form)
      (when (and (consp bindings) (null (cdr bindings)) (consp (first bindings))
                 (consp end-clause) (null (cdr end-clause)))
        (destructuring-bind (index &optional (init nil init-p) (step nil step-p))
            (first bindings)
          (when (and (symbolp index)
                     init-p step-p
                     (eql init 0)
                     (vector-increments-index-p step index))
            (let ((test (first end-clause)))
              (when (and (consp test) (memq (car test) '(>= =)) (= (length test) 3)
                         (eq (second test) index))
                (let ((count (third test)))
                  (when (vector-pure-count-p count)
                    (multiple-value-bind (declarations statements)
                        (vector-split-declarations body)
                      (when (not (vector-special-declaration-p declarations))
                        (multiple-value-bind (element-type stores intermediates)
                            (vector-match-stores statements env index)
                          (when element-type
                            (make-vector-loop
                             :element-type element-type
                             :index index
                             :count count
                             :stores stores
                             :sources (vector-stores-sources stores intermediates)
                             :intermediates intermediates
                             :expression (fifth (first stores))
                             :body body
                             :fallback form :unchecked-p (policy env (= safety 0)))))))))))))))))

;;; The part of VALUE that is being added to the accumulator PLACE:
;;; (+ place expression) or (+ expression place), and nothing else.
(defun vector-reduction-increment (value place)
  (when (and (consp value) (eq (car value) '+) (= (length value) 3))
    (cond ((eq (second value) place) (third value))
          ((eq (third value) place) (second value)))))

;;; a sum of a fixed-width integer array into a variable of that same width:
;;;
;;;   (dotimes (i count) (incf accumulator expression))
;;;
;;; integer addition wraps at the declared width whatever the order of the adds, so
;;; accumulating into lanes and adding them at the end gives the same answer.
;;; floating point addition does not wrap, so a float accumulator is refused.
(defun vector-match-reduction (form env)
  (when (and (consp form) (eq (car form) 'dotimes)
             (consp (cdr form)) (consp (cadr form)))
    (destructuring-bind (spec . body) (cdr form)
      (when (and (consp spec) (null (cddr spec)))
        (destructuring-bind (index count) spec
          (when (and (symbolp index) (vector-pure-count-p count))
            (multiple-value-bind (declarations statements) (vector-split-declarations body)
              (when (and (null (cdr statements))
                         (not (vector-special-declaration-p declarations)))
                (let ((store (vector-normalize-store (first statements))))
                  (when (and (consp store) (eq (car store) 'setf) (null (cdddr store)))
                    (let ((place (second store)))
                      (when (and (symbolp place) (not (eq place index)))
                        (multiple-value-bind (kind type) (vector-variable-info place env)
                          (when (and (eq kind :scalar)
                                     (not (vector-float-element-type-p type)))
                            ;; only the part being added is vectorized; if the
                            ;; accumulator itself appears there, the loop is not a
                            ;; plain sum of the arrays
                            (let ((expression
                                    (vector-reduction-increment (third store) place)))
                              (when expression
                                (setf expression
                                      (vector-match-expression
                                       expression env index type)))
                              (when expression
                                (let ((plan (make-vector-loop
                                             :element-type type
                                             :index index
                                             :count count
                                             :accumulator place
                                             :sources (vector-collect-arrays expression)
                                             :expression expression
                                             :body statements
                                             :fallback form :unchecked-p (policy env (= safety 0)))))
                                  ;; the reduction expansion addresses the sources
                                  ;; directly, so it needs their data inline
                                  (when (vector-loop-all-simple-p plan)
                                    plan))))))))))))))))))

;;;; Generating the vectorized loop

;;; the backend primitives that move and combine one vector: an alist of
;;; ((base element-type) . name), where base is "LOAD", "STORE", "SPLAT", "ADD",
;;; "SUB", "MUL" or "DIV", filled in by the backend file that defines them so that
;;; this pass is inert on a build without them.  "OFFSET" is the one base that is
;;; optional within a backend that publishes the rest: a backend whose loads and
;;; stores address by the byte offset it computes publishes it, and one whose
;;; addressing mode takes the index, as x86-64's does, does not.  the backend file hands the names over
;;; rather than this one writing them, because this file cannot name the backend's
;;; package.  a string in the heap that is STRING= to a package name defeats string
;;; coalescing in SAVE-LISP-AND-DIE, which tests/hide-packages.test.sh enforces.
(defvar *vector-primitives* nil)

;;; the 256-bit primitives, published separately and only by a backend whose
;;; build enables them.  a 256-bit pack is its own Lisp type, so the functions
;;; cannot be the same names as the 128-bit ones and are indexed by width.
(defvar *vector-primitives-256* nil)

(defun vector-primitive (base element-type &optional (bytes *vector-lane-bytes*))
  (cdr (assoc (list base element-type)
              (if (eql bytes 32) *vector-primitives-256* *vector-primitives*)
              :test #'equal)))

;;; true when this element type has both a 256-bit load and store, which is what
;;; makes a wider loop worth attempting at all.
(defun vector-wide-available-p (element-type)
  (and *vector-primitives-256*
       (vector-primitive "LOAD" element-type 32)
       (vector-primitive "STORE" element-type 32)
       t))

;;; every primitive the expression needs at the current width, so that the wider
;;; attempt declines before emitting rather than halfway through.  the inner
;;; read of a widen is not a load: the widen vop reads it itself.
(defun vector-expression-primitives (expression element-type &optional (accumulator '()))
  (flet ((add (base) (adjoin base accumulator :test #'string=)))
    (case (car expression)
      (:array (add "LOAD"))
      (:negate (vector-expression-primitives (second expression) element-type (add "NEG")))
      (:widen (let ((base (vector-widen-base (third expression) element-type)))
                (if base (add base) (add "UNEXPRESSIBLE"))))
      (:op (let ((base (vector-operator-base (second expression) element-type)))
             (if base
                 (dolist (subexpression (cddr expression) (add base))
                   (setf accumulator
                         (vector-expression-primitives subexpression element-type
                                                       accumulator)))
                 (add "UNEXPRESSIBLE"))))
      (:index (add "STRIDE"))
      (t accumulator))))

;;; The operators an expression tree uses, in order of first use.
(defun vector-collect-operators (expression &optional (accumulator '()))
  (case (car expression)
    (:op (dolist (subexpression (cddr expression)
             (adjoin (second expression) accumulator))
           (setf accumulator (vector-collect-operators subexpression accumulator))))
    (:negate (adjoin :negate (vector-collect-operators (second expression) accumulator)))
    (:index (adjoin :index accumulator))
    (t accumulator)))

;;; Emit an expression tree over SIMD packs. SCALARS maps each loop invariant
;;; leaf to the variable holding its broadcast pack.
(defun vector-emit (expression element-type index scalars data-references
                    &optional offset intermediates)
  (ecase (car expression)
    (:index (cdr (assoc :index scalars)))
    (:negate `(,(vector-primitive "NEG" element-type)
               ,(vector-emit (second expression) element-type index scalars
                             data-references offset intermediates)))
    (:array (let* ((name (second expression))
                   (subscript (third expression))
                   (reference (or (cadr (assoc name data-references)) name)))
              `(,(vector-primitive "LOAD" element-type) ,reference
                ,(if offset
                     (vector-offset-form name subscript index data-references
                                         offset element-type)
                     (vector-subscript name subscript index data-references)))))
    (:widen (let ((inner (second expression))
                  (source-type (third expression)))
              ;; a widen reads a narrower element type than it writes, so its
              ;; accesses do not share the destination's offset
              `(,(vector-primitive (vector-widen-base source-type element-type) element-type)
                ,(or (cadr (assoc (second inner) data-references)) (second inner))
                ,(vector-subscript (second inner) (third inner) index data-references))))
    (:op (let ((operator (second expression)))
           (let ((name (vector-primitive (vector-operator-base operator element-type) element-type)))
             (reduce (lambda (left right) `(,name ,left ,right))
                     (mapcar (lambda (subexpression)
                               (vector-emit subexpression element-type index scalars
                                            data-references offset intermediates))
                             (cddr expression))))))
    ((:scalar :const)
     ;; splats are computed once, before the loop, and live in an SSE register
     (cdr (assoc expression scalars :test #'equal)))
    (:intermediate
     ;; a value the body binds in a LET* and more than one store uses
     (second expression))))

;;; an iteration reads everything it touches before writing anything, which matches
;;; the scalar loop only when the destination is written at or behind every place it
;;; is read: a destination written ahead of a read is read back by the scalar loop
;;; and not by this one.
(defun vector-alias-safe-p (destination-offset source-offsets)
  (every (lambda (source-offset)
           (or (eql destination-offset source-offset)
               (and (integerp destination-offset) (integerp source-offset)
                    (<= destination-offset source-offset))
               ;; a variable offset is guarded to be non-negative at run time, so a
               ;; destination at 0 is never ahead of it
               (and (eql destination-offset 0) (not (integerp source-offset)))))
         source-offsets))

;;; Whether a body that writes several arrays means, packed, what it means
;;; element by element.  An array the body never reads is only written, which is
;;; always safe.  An array it reads and writes is safe when every offset it writes
;;; is an offset it reads - the iteration permutes the elements it read, which is
;;; what a butterfly does - or when no write is ahead of any read, which is the
;;; rule a single store has always had.
(defun vector-stores-safe-p (stores sources)
  (let ((written '()))
    (dolist (store stores)
      (let ((entry (assoc (first store) written :test #'eq)))
        (if entry
            (pushnew (second store) (cdr entry) :test #'equal)
            (push (cons (first store) (list (second store))) written))))
    (flet ((read-offsets (name) (second (assoc name sources :test #'eq)))
           (write-offsets (name) (cdr (assoc name written :test #'eq))))
      (every (lambda (store)
               (let* ((name (first store))
                      (writes (write-offsets name))
                      (reads (read-offsets name)))
                 (cond ((null reads) t)
                       ;; every element the body writes to this array is one it
                       ;; read: the iteration permutes what it touched
                       ((and (subsetp writes reads :test #'equal)
                             (subsetp reads writes :test #'equal))
                        t)
                       (t (vector-alias-safe-p (second store) reads)))))
             stores))))

(defun vector-loop-arrays (loop)
  "The arrays a loop touches: the ones it writes first, then the ones it only
reads.  An array it both reads and writes - (setf (aref c i) (+ (aref c (+ i 3))
...)) - appears once, with every offset either side touches it at, so the length
guard covers the reads as well as the writes."
  (let ((arrays '()))
    (flet ((touch (name offsets simple-p)
             (let ((entry (assoc name arrays :test #'eq)))
               (if entry
                   (dolist (offset offsets)
                     (pushnew offset (second entry) :test #'equal))
                   (setf arrays (append arrays
                                        (list (cons name (cons offsets simple-p)))))))))
      (dolist (store (vloop-stores loop))
        (touch (first store) (list (second store)) (third store)))
      (dolist (source (vloop-sources loop))
        (touch (first source) (second source) (cddr source))))
    arrays))

;;; Whether every array a plan reads has its data inline.  A reduction has no
;;; destination, since its result is a variable, so only the sources matter.
(defun vector-loop-all-simple-p (loop)
  (every (lambda (entry) (cddr entry)) (vloop-sources loop)))

;;; The data vector and start index to use for each array that is not known to be
;;; simple, so that the loop body can address a plain simple array.
(defun vector-data-references (loop)
  (loop for (name offsets . simple-p) in (vector-loop-arrays loop)
        unless simple-p
          collect (cons name (cons (make-symbol "DATA") (make-symbol "START")))))

;;; The byte offset of an access, for a backend whose loads and stores address by
;;; one rather than by the element index.  A loop computes a single offset per
;;; iteration and shares it among the accesses that use the plain index, since an
;;; iteration reads and writes every array at that same one; an access at any
;;; other subscript - a stencil's (+ i 2) - computes its own, which is what each
;;; access cost before they could be shared.
(defun vector-offset-form (name subscript index data-references shared element-type)
  (let ((subscript (vector-subscript name subscript index data-references)))
    (if (and shared (equal subscript index))
        shared
        `(,(vector-primitive "OFFSET" element-type) ,subscript))))

;;; The index expression for element I plus OFFSET of the array NAME: when the
;;; array's data is not inline, the element is at START + I + OFFSET of the data
;;; vector.
(defun vector-subscript (name offset index data-references)
  (let ((start (cddr (assoc name data-references))))
    (cond ((null start) (if (eql offset 0) index `(+ ,index ,offset)))
          ((eql offset 0) `(+ ,index ,start))
          (t `(+ ,index ,start ,offset)))))

;;; Wrap FORMS in one WITH-ARRAY-DATA per array whose data is not inline, so the
;;; data vector and start index are found once per call instead of per element.
(defun vector-with-data-vectors (references form)
  (if (null references)
      form
      (destructuring-bind (name data . start) (first references)
        (let ((end (make-symbol "END")))
          `(sb-kernel:with-array-data ((,data ,name) (,start) (,end) :force-inline t)
             (declare (ignore ,end))
             ,(vector-with-data-vectors (rest references) form))))))

;;; Return the vectorized form for a matched loop, or NIL if this backend cannot
;;; express it (which is the normal answer on a build whose backend has not
;;; published its primitives, and for byte lanes outside SAFETY 0).
(defun attempt-expansion (expander loop bytes)
  (let ((*vector-lane-bytes* bytes))
    (funcall expander loop)))

;;; ---------------------------------------------------------------- probe
;;; A pass that will live after LOOP-ANALYZE (src/compiler/main.lisp) has a natural
;;; loop object for every loop in the component, with the blocks that make it up and
;;; with the bodies already macroexpanded, inline-expanded and let-converted.  This
;;; probe prints what such a loop is really made of, so that the analysis can be
;;; written against the IR1 that exists instead of the source forms it comes from.

(defvar *vector-node-probe* nil
  "When true, print the IR1 of every loop in each compiled component.")

(defun vector-probe-fun-name (node)
  (let ((use (lvar-uses (combination-fun node))))
    (typecase use
      (ref (let ((leaf (ref-leaf use)))
             (typecase leaf
               (functional (or (functional-debug-name leaf) "functional"))
               (global-var (leaf-source-name leaf))
               (t (type-of leaf)))))
      (t (type-of use)))))

(defun vector-probe-type (lvar)
  (if lvar
      (type-specifier (lvar-type lvar))
      'null))

(defun vector-probe-node (node)
  (typecase node
    (basic-combination
     (format nil "~a(~{~a~^, ~}) -> ~a"
             (vector-probe-fun-name node)
             (mapcar #'vector-probe-type (combination-args node))
             (vector-probe-type (node-lvar node))))
    (cset (format nil "set ~a = ~a"
                  (leaf-source-name (set-var node))
                  (vector-probe-type (set-value node))))
    (cif (format nil "if ~a" (vector-probe-type (if-test node))))
    (ref (let ((leaf (ref-leaf node)))
           (typecase leaf
             (functional (format nil "ref ~a" (or (functional-debug-name leaf) "functional")))
             (constant (format nil "ref constant ~s" (constant-value leaf)))
             (t (format nil "ref ~a" (leaf-source-name leaf))))))
    (bind (format nil "bind ~a" (or (functional-debug-name (bind-lambda node))
                                    "lambda")))
    (creturn (format nil "return"))
    (exit (format nil "exit"))
    (entry (format nil "entry"))
    (t (format nil "~a" (type-of node)))))

(defun vector-probe-block-count (loop)
  ;; LOOP-BLOCKS is the first block of the loop, the rest follow it through
  ;; BLOCK-LOOP-NEXT: it is a chain, not a list.
  (do ((block (loop-blocks loop) (block-loop-next block))
       (count 0 (1+ count)))
      ((null block) count)))

(defun vector-probe-loop (loop depth)
  (let ((indent (make-string depth :initial-element #\space)))
    (format *error-output* "~&;~a~a loop: depth ~a, ~a block~:p, head b~a, exits ~a~%"
            indent (if (eq (loop-kind loop) :outer) "outer" "natural")
            (loop-depth loop) (vector-probe-block-count loop)
            (block-number (loop-head loop))
            (mapcar #'block-number (loop-exits loop)))
    (do ((block (loop-blocks loop) (block-loop-next block)))
        ((null block))
      (format *error-output* "~&;~a  b~a -> ~a~%"
              indent (block-number block) (mapcar #'block-number (block-succ block)))
      (do-nodes (node lvar block)
        (format *error-output* "~&;~a    ~a~a~%"
                indent (vector-probe-node node)
                (if lvar
                    (format nil " ; value ~a" (vector-probe-type lvar))
                    ""))))
    (dolist (inferior (loop-inferiors loop))
      (vector-probe-loop inferior (+ depth 2)))))

(defun vector-node-probe-component (component)
  (when *vector-node-probe*
    (format *error-output* "~&; component ~a~%" (component-name component))
    (vector-probe-loop (component-outer-loop component) 2)
    (format *error-output* "~&;~%")))

;;; ------------------------------------------------- recognition, on IR1
;;; After LOOP-ANALYZE every natural loop is a CLOOP with its blocks, and by then the
;;; body has been macroexpanded, inline-expanded and let-converted: an inlined helper
;;; is the arithmetic it does, AREF is lowered, and every lvar carries a derived type.
;;; These read a loop into a plan, and print that plan back as source the existing
;;; emitter takes - with the array types as (THE ...) casts, since they came from
;;; LVAR-TYPE and there may be no declaration in the enclosing scope at all.
;;;
;;; (:element-type SINGLE-FLOAT :index I :limit N
;;;  :stores ((C :op "+" ((:load A 0 SINGLE-FLOAT) (:load B 0 SINGLE-FLOAT)))))
(defun vector-node-call-name (node)
  (let ((use (lvar-uses (combination-fun node))))
    (when (ref-p use)
      (let* ((leaf (ref-leaf use))
             (name (typecase leaf
                     (global-var (leaf-source-name leaf))
                     (functional (functional-debug-name leaf))
                     (t nil))))
        (when name (typecase name (string name) (symbol (symbol-name name))
                                 (t (prin1-to-string name))))))))
(defun vector-node-name-p (node name)
  (let ((n (and (basic-combination-p node) (vector-node-call-name node))))
    (and n (string= n name))))
(defun vector-node-leaf-var (lvar)
  (let ((use (lvar-uses lvar)))
    (when (and (ref-p use) (lambda-var-p (ref-leaf use)))
      (ref-leaf use))))
(defun vector-node-constant (lvar)
  (let ((use (lvar-uses lvar)))
    (when (ref-p use)
      (let ((leaf (ref-leaf use)))
        (when (constant-p leaf) (constant-value leaf))))))
(defun vector-node-blocks (loop)
  (do ((b (loop-blocks loop) (block-loop-next b)) (acc '() (cons b acc)))
      ((null b) (nreverse acc))))
(defun vector-node-nodes (block)
  (let ((acc '()) (ctran (block-start block)))
    (loop while ctran for node = (ctran-next ctran) while node
          do (push node acc) (setf ctran (node-next node)))
    (nreverse acc)))
(defun vector-node-index-p (lvar index-var depth)
  "True when LVAR carries the loop index: I itself, or a parameter of an inline
function called with it - ARET's lowering passes the index through one."
  (when (> depth 12) (return-from vector-node-index-p nil))
  (let ((leaf (vector-node-leaf-var lvar)))
    (cond ((and leaf (eq leaf index-var)) t)
          (leaf
           (let ((home (lambda-var-home leaf)))
             (when home
               (let ((call (handler-case (let-combination home) (error () nil))))
                 (when call
                   (let ((pos (position leaf (lambda-vars home))))
                     (and pos
                          (vector-node-index-p (nth pos (combination-args call))
                                      index-var (1+ depth)))))))))
          (t nil))))

(defun vector-node-var-init (var)
  "The value VAR is called with at its home lambda, when that is a constant."
  (let* ((home (lambda-var-home var))
         (call (and home (handler-case (let-combination home) (error () nil))))
         (pos (and call (position var (lambda-vars home)))))
    (and pos (vector-node-constant (nth pos (combination-args call))))))

(defun vector-node-tree (lvar depth index-var)
  (when (> depth 8) (return-from vector-node-tree nil))
  (let ((use (lvar-uses lvar)))
    (cond ((vector-node-constant lvar) (list :const (vector-node-constant lvar)))
          ((and (ref-p use) (lambda-var-p (ref-leaf use)))
           (list :var (leaf-source-name (ref-leaf use))))
          ((not (basic-combination-p use)) nil)
          ((vector-node-name-p use "DATA-VECTOR-REF-WITH-OFFSET")
           (let* ((args (combination-args use))
                  (array (vector-node-leaf-var (first args))))
             (and array
                  (eql (vector-node-constant (third args)) 0)
                  (vector-node-index-p (second args) index-var 0)
                  (list :load (leaf-source-name array)
                        (vector-node-constant (third args))
                        (type-specifier (lvar-type (node-lvar use)))))))
          (t (let ((op (vector-node-call-name use)))
               (when (member op '("+" "-" "*" "/") :test #'string=)
                 (let ((trees (mapcar (lambda (a) (vector-node-tree a (1+ depth) index-var))
                                      (combination-args use))))
                   (and (every #'identity trees) (list :op op trees)))))))))

;;; the whole loop, if it is one this pass understands
(defmacro vector-node-fail (reason) `(return-from vector-node-plan ,reason))
(defun vector-node-plan (loop)
  (unless (and (eq (loop-kind loop) :natural) (null (loop-inferiors loop)))
    (vector-node-fail :not-a-simple-loop))
  (let* ((blocks (vector-node-blocks loop))
         (head (loop-head loop)))
    (unless (eq (first blocks) head) (vector-node-fail :head-not-first))
    (unless (= (length blocks) 2) (vector-node-fail (list :block-count (length blocks))))
    (let* ((body (second blocks))
           (cif (find-if (lambda (n) (typep n 'cif)) (vector-node-nodes head)))
           (test (and cif (lvar-uses (if-test cif))))
           (args (and (basic-combination-p test) (combination-args test))))
      (unless cif (vector-node-fail :no-if-in-header))
      (unless (and (basic-combination-p test) (string= (or (vector-node-call-name test) "") "<"))
        (vector-node-fail (list :test-not-lessp (and (basic-combination-p test) (vector-node-call-name test)))))
      (let* ((index-lvar (find-if (lambda (a) (vector-node-leaf-var a)) args))
             (index-var (and index-lvar (vector-node-leaf-var index-lvar)))
             (limit-lvar (and index-lvar (find-if (lambda (a) (not (eq a index-lvar))) args)))
             (body-nodes (vector-node-nodes body))
             (increment (find-if (lambda (n) (and (typep n 'cset) (eq (set-var n) index-var)))
                                 body-nodes))
             (stores (remove-if-not (lambda (n) (vector-node-name-p n "DATA-VECTOR-SET-WITH-OFFSET"))
                                    body-nodes)))
        (unless index-var (vector-node-fail :no-index))
        (unless increment (vector-node-fail :no-increment))
        (unless stores (vector-node-fail :no-store))
        (let* ((step (lvar-uses (set-value increment)))
               (step-args (and (basic-combination-p step) (combination-args step))))
          (unless (and step-args (string= (or (vector-node-call-name step) "") "+"))
            (vector-node-fail (list :increment-not-plus (and (basic-combination-p step) (vector-node-call-name step)))))
          (unless (and (eq (vector-node-leaf-var (first step-args)) index-var)
                       (eql (vector-node-constant (second step-args)) 1))
            (vector-node-fail :increment-not-by-one)))
        (unless (every (lambda (ref) (memq (ctran-block (node-prev ref)) blocks))
                       (leaf-refs index-var))
          (vector-node-fail :index-used-outside))
        (let ((entries '()))
          (dolist (store stores)
            (let* ((sargs (combination-args store))
                   (array (vector-node-leaf-var (first sargs)))
                   (tree (vector-node-tree (fourth sargs) 0 index-var)))
              (unless array (vector-node-fail :store-array-not-a-variable))
              (unless (vector-node-index-p (second sargs) index-var 0)
                (vector-node-fail :store-index-not-the-induction-variable))
              (unless (eql (vector-node-constant (third sargs)) 0)
                (vector-node-fail (list :store-offset (vector-node-constant (third sargs)))))
              (unless tree (vector-node-fail :no-value-tree))
              (push (list (leaf-source-name array) (type-specifier (lvar-type (fourth sargs))) tree)
                    entries)))
          (unless (eql (vector-node-var-init index-var) 0)
            (vector-node-fail (list :index-does-not-start-at-zero (vector-node-var-init index-var))))
          (list :element-type (second (first entries))
                :index (leaf-source-name index-var)
                :limit (or (vector-node-constant limit-lvar)
                           (let ((leaf (vector-node-leaf-var limit-lvar)))
                             (and leaf (leaf-source-name leaf))))
                :stores (mapcar (lambda (e) (cons (first e) (third e))) (nreverse entries))))))))


;;; ---- from the plan back to source -----------------------------------------
(defun vector-node-print-tree (tree subst index)
  (case (car tree)
    (:const (second tree))
    (:var (second tree))
    (:load `(aref ,(or (cdr (assoc (second tree) subst :test #'equal))
                       (second tree))
                  ,index))
    (:op `(,(intern (second tree) "CL")
           ,@(mapcar (lambda (sub) (vector-node-print-tree sub subst index))
                     (third tree))))
    (t (error "unexpected tree ~s" tree))))

(defun vector-node-tree-arrays (tree acc)
  (case (car tree)
    (:load (adjoin (second tree) acc :test #'equal))
    (:op (reduce (lambda (acc2 sub) (vector-node-tree-arrays sub acc2)) (third tree)
                 :initial-value acc))
    (t acc)))

(defun vector-node-source-form (plan)
  (let* ((stores (getf plan :stores))
         (arrays (reduce (lambda (acc store) (vector-node-tree-arrays (cdr store) acc))
                         stores
                         :initial-value (remove-duplicates (mapcar #'car stores)
                                                           :test #'equal)))
         (index (make-symbol "VN-I"))
         (subst (mapcar (lambda (name)
                          (cons name (make-symbol (format nil "VN-~a" name))))
                        arrays))
         (type (getf plan :element-type)))
    `(let (,@(mapcar (lambda (pair)
                       `(,(cdr pair)
                         (the (simple-array ,type (*)) ,(car pair))))
                     subst))
       (dotimes (,index ,(getf plan :limit))
         ,@(mapcar (lambda (store)
                     `(setf (aref ,(cdr (assoc (car store) subst :test #'equal)) ,index)
                            ,(vector-node-print-tree (cdr store) subst index)))
                   stores)))))


;;; Put the replacement where the loop was: convert it into a fresh block between
;;; the loop's preheader and its exit, relink those two edges, and delete the two
;;; blocks the loop was made of.  WRAP-PREDICATE (ir1tran.lisp:1192) is the model
;;; for building a block here; there is no call to repoint, so this is the same
;;; surgery WITHOUT the lambda that the inline-expansion path uses.
(defvar *vector-node-splice* nil
  "When true, a loop the node analysis plans is replaced by what it emits.")

(defun vector-node-splice-loop (loop plan)
  (let* ((blocks (vector-node-blocks loop))
         (head (loop-head loop))
         (outside-in (remove-if (lambda (block) (memq block blocks)) (block-pred head)))
         (outside-out (remove-if (lambda (block) (memq block blocks)) (block-succ head))))
    ;; one way in and one way out: anything else is a shape that needs more care
    (unless (and (= (length outside-in) 1) (= (length outside-out) 1))
      (return-from vector-node-splice-loop nil))
    (let ((preheader (first outside-in))
          (exit (first outside-out))
          (start (make-ctran))
          (next (make-ctran)))
      (declare (ignorable exit))
      (let ((middle (ctran-starts-block start))
            (after (ctran-starts-block next)))
        (with-ir1-environment-from-node (first (vector-node-nodes head))
          (ir1-convert-progn-body start next nil (list (vector-node-source-form plan))))
        ;; the code runs after the preheader and before the loop's exit
        (link-blocks middle after)
        (link-blocks after exit)
        (setf (block-succ preheader) (list middle))
        (setf (block-pred middle) (list preheader))
        ;; the loop is gone: drop both blocks, back edge and all
        (dolist (block blocks)
          (setf (block-delete-p block) t))
        (dolist (block blocks)
          (delete-block block t))
        t))))

(defun vector-expand-loop (loop)
  (if (vloop-accumulator loop)
      (vector-expand-reduction-loop loop)
      (vector-expand-elementwise-loop loop)))

;;; A reduction keeps the accumulator in lanes and adds them to the variable at the
;;; end, so the two widths share nothing worth hoisting: each attempt is a whole
;;; expansion, and the machine picks between them when the loop runs.
(defun vector-expand-reduction-loop (loop)
  (let* ((element-type (vloop-element-type loop))
         (dispatch (and *vector-wide-predicate*
                        (vector-wide-available-p element-type)
                        t))
;;; both attempts are silent here: which note to print depends on what the pair
;;; of them produced, and only one note should be printed
         (wide (let ((*auto-vectorize-notes* nil))
                 (and (vector-wide-available-p element-type)
                      (vector-elementwise-attempt #'vector-expand-reduction loop 32))))
         (narrow (let ((*auto-vectorize-notes* nil))
                   (vector-elementwise-attempt #'vector-expand-reduction loop 16))))
    (cond ((not (or wide narrow)) nil)
;;; a wide attempt with no narrow one would be emitted with no test to stop it, and a
;;; reduction has nothing to fall back to: leave it scalar
          ((and wide dispatch (null narrow)) nil)
          (t (when *auto-vectorize-notes*
               (if (and wide narrow dispatch)
                   (compiler-notify "vectorized loop over ~a, ~a elements at a time where ~
                                     the machine has avx2 and ~a where it does not"
                                    element-type (vector-width element-type 32)
                                    (vector-width element-type 16))
                   (compiler-notify "vectorized loop over ~a, ~a elements at a time"
                                    element-type
                                    (vector-width element-type (if wide 32 16)))))
             (if (and wide narrow dispatch)
                 `(if (,*vector-wide-predicate*) ,wide ,narrow)
                 (or wide narrow))))))

(defun vector-elementwise-attempt (expander loop bytes)
  (let ((*vector-lane-bytes* bytes))
    (funcall expander loop)))

;;; What one width would emit for an elementwise loop, or NIL when this backend
;;; cannot express it at that width.  The pieces are separate because a build that
;;; hands over a predicate puts the loops for two widths in one function: only the
;;; packed loop and the lane bindings differ between them, while the length guard,
;;; the hand-off point and the scalar tail are the same code either way.
;;;
;;; (values :width N :guard (CONJUNCT ...) :pack-type SYMBOL :bindings (LET* ...)
;;;         :packed <do> :pick-end (- COUNT-VAR (MOD COUNT-VAR N)))
;;; COUNT-VAR, END and the data-vector names belong to the emitter, which shares
;;; them between the two plans: a fresh set of names would leave the packed loop
;;; reading a vector nothing bound.
(defun vector-elementwise-plan (loop bytes count-var end references)
  (let* ((*vector-lane-bytes* bytes)
         (element-type (vloop-element-type loop))
         (width (vector-width element-type))
         (index (vloop-index loop))
         (stores (vloop-stores loop))
         (intermediates (vloop-intermediates loop))
         ;; the analysis below sees what the loop really computes; the emitted
         ;; stores keep the markers, so a shared value is computed once
         (expressions (mapcar (lambda (store)
                                (vector-inline-intermediates (fifth store) intermediates))
                              stores))
         (scalars (mapcar (lambda (leaf) (cons leaf (make-symbol "SPLAT")))
                          (remove-duplicates (mapcan #'vector-collect-scalars expressions)
                                             :test #'equal)))
         (arrays (vector-loop-arrays loop))
;;; a read at (- i 1) leaves the array on the first packed iteration, and the scalar
;;; loop it replaces faults at i = 0 too; only a literal can be known negative here.
         (negative-literals
          (loop for entry in arrays
                thereis (some (lambda (offset) (and (typep offset 'integer) (minusp offset)))
                              (second entry))))
         (pack-type (vector-pack-type element-type))
         (load (vector-primitive "LOAD" element-type))
         (store (vector-primitive "STORE" element-type))
         (splat (and scalars (vector-primitive "SPLAT" element-type)))
         (operators (remove-duplicates (mapcan #'vector-collect-operators expressions)
                                       :test #'eq))
         (arithmetic (mapcar (lambda (operator)
                               (vector-primitive (vector-operator-base operator element-type)
                                                 element-type))
                             operators))
         (indexed (and (memq :index operators) t))
         (ramp (and indexed (vector-primitive "RAMP" element-type)))
         (stride (and indexed (vector-primitive "STRIDE" element-type)))
;;; a backend whose loads and stores address by a byte offset lets the iteration
;;; compute that offset once and share it; one that addresses by the index, as
;;; x86-64 does with its scaled address mode, leaves this NIL and each access
;;; builds its own.
         (offset (and (vector-primitive "OFFSET" element-type) (make-symbol "OFFSET"))))
    (when indexed
;;; the lane offsets, then a constant step per iteration: rebuilding the index pack
;;; each time costs four instructions and a dependency chain.
      (setf scalars (acons :index (make-symbol "INDEX-LANES") scalars))
      (setf scalars (acons :stride (make-symbol "STRIDE") scalars)))
    (when (and load store
               (every #'identity arithmetic)
;;; a widen or an operator the wider width has no instruction for has to be
;;; refused here, before anything is emitted, so that the 128-bit attempt runs
               (every (lambda (base) (vector-primitive base element-type))
                      (remove-duplicates
                       (mapcan (lambda (expression)
                                 (vector-expression-primitives expression element-type))
                               expressions)
                       :test #'string=))
               (if indexed (and ramp stride) t)
               (or splat (null (remove-if (lambda (scalar)
                                            (memq (car scalar) '(:index :stride)))
                                          scalars)))
               (or (vector-float-element-type-p element-type)
                   (vloop-unchecked-p loop)
                   (every #'vector-exact-expression-p expressions))
               (not negative-literals)
               (vector-stores-safe-p stores (vloop-sources loop)))
      (labels ((reference (name) (or (cadr (assoc name references)) name))
               ;; the pack a LET* binding stood for, or the name it was given
               (emit (expression offset)
                 (vector-emit expression element-type index scalars references
                              offset intermediates))
               (store-form (entry offset)
                 (let ((name (first entry)) (subscript-offset (second entry))
                       (expression (fifth entry)))
                   `(,store ,(emit expression offset)
                            ,(reference name)
                            ,(if offset
                                 (vector-offset-form name subscript-offset index
                                                     references offset element-type)
                                 (vector-subscript name subscript-offset index
                                                   references))))))
        (let ((guard
                (loop for entry in arrays
                      for name = (first entry)
                      for offsets = (second entry)
                      append (cons `(>= (length ,name) (+ ,count-var (max 0 ,@offsets)))
                                   (loop for offset in offsets
                                         unless (typep offset '(integer 0 *))
                                           collect `(>= ,offset 0)))))
              (bindings
                (mapcar (lambda (scalar)
                          (case (car scalar)
                            (:stride `(,(cdr scalar)
                                       (,(vector-primitive "STRIDE" element-type))))
                            (:index `(,(cdr scalar) (,ramp)))
                            (t `(,(cdr scalar) (,splat ,(second (car scalar)))))))
                        scalars))
              (packed
                `(do ((,index 0 (+ ,index ,width)))
                     ((>= ,index ,end))
                   (declare (type index ,index))
                   ,(let* ((stores-form
                             `(progn ,@(mapcar (lambda (entry)
                                                 (store-form entry offset))
                                               stores)))
                           ;; the intermediates a body binds and its stores share
                           ;; are computed once here, in the order it bound them
                           (stores-form
                             (if intermediates
                                 `(let* ,(loop for (var tree) in intermediates
                                               collect `(,var ,(emit tree offset)))
                                    (declare (type ,pack-type
                                                   ,@(mapcar #'first intermediates)))
                                    ,stores-form)
                                 stores-form)))
                      ;; a backend that addresses by a byte offset computes it
                      ;; once for the whole iteration
                      (if offset
                          `(let* ((,offset (,(vector-primitive "OFFSET" element-type)
                                            ,index)))
                             (declare (type (unsigned-byte 64) ,offset))
                             ,stores-form)
                          stores-form))
                   ,@(when indexed
                       ;; leave the pack holding the offsets for the next iteration
                       (let ((lanes (cdr (assoc :index scalars)))
                             (stride (cdr (assoc :stride scalars))))
                         `((setf ,lanes
                                 (,(vector-primitive "ADD" element-type)
                                  ,lanes ,stride))))))))
          (list :width width
                :element-type element-type
                :names (mapcar #'cdr scalars)
                :guard guard
                :pack-type pack-type
                :bindings bindings
                :packed packed
                :pick-end `(- ,count-var (mod ,count-var ,width))))))))

;;; Emit an elementwise loop from one or two plans: one packed loop, or both with
;;; the machine choosing at run time.  The guard, the hand-off point END and the
;;; scalar tail are written once for either shape.
(defun vector-emit-elementwise (loop count-var end references wide narrow dispatch)
  (let ((element-type (vloop-element-type loop))
        (index (vloop-index loop))
        (count (vloop-count loop))
        (ok (make-symbol "PACKED-RUNS"))
        (wide-p (make-symbol "WIDE")))
    (when *auto-vectorize-notes*
      (cond ((and dispatch narrow)
             (compiler-notify "vectorized loop over ~a, ~a elements at a time where ~
                               the machine has avx2 and ~a where it does not"
                              element-type (getf wide :width) (getf narrow :width)))
            (dispatch
;;; nothing narrower can express this loop, so what the machine cannot run it must
;;; not reach: the element loop does all the work instead
             (compiler-notify "vectorized loop over ~a, ~a elements at a time, where ~
                               the machine has avx2 and element by element where it ~
                               does not" element-type (getf wide :width)))
            (t
             (compiler-notify "vectorized loop over ~a, ~a elements at a time"
                              element-type (getf (or wide narrow) :width)))))
    (flet ((packed-branch (plan)
             `(let* ,(getf plan :bindings)
                (declare (type ,(getf plan :pack-type) ,@(getf plan :names)))
                ,(getf plan :packed)))
           (tail ()
             ;; the tail, element by element, exactly as written
             `(do ((,index ,end (1+ ,index)))
                  ((>= ,index ,count-var))
                (declare (type index ,index))
                ,@(vloop-body loop))))
      (let* ((guard (if dispatch (getf (or narrow wide) :guard)
                        (getf (or wide narrow) :guard)))
             (body
               (vector-with-data-vectors
                references
                (if dispatch
;;; the lane bindings are 256-bit instructions too, so the test has to cover them:
;;; a machine without avx2 must not reach them, not merely skip the loop
                    `(let* ((,ok (and (typep ,count-var 'index) ,@guard))
                            (,wide-p (and ,ok (,*vector-wide-predicate*)))
                            (,end (if ,wide-p
                                      ,(getf wide :pick-end)
                                      ,(if narrow `(if ,ok ,(getf narrow :pick-end) 0) 0))))
                       (declare (type index ,end))
                       (if ,wide-p
                           ,(packed-branch wide)
                           ,(when narrow `(when ,ok ,(packed-branch narrow))))
                       ,(tail))
                    `(let* ((,end (if (and (typep ,count-var 'index) ,@guard)
                                      ,(getf (or wide narrow) :pick-end)
                                      0)))
                       (declare (type index ,end))
                       ,(packed-branch (or wide narrow))
                       ,(tail))))))
;;; the guard chooses END instead of branching between two copies of the loop.  When
;;; it fails, END is 0, the packed loop does not run, and the tail does all the
;;; work: it is the original loop, written once.
        `(let ((,count-var ,count))
           ,body)))))

(defun vector-expand-elementwise-loop (loop)
  (let* ((element-type (vloop-element-type loop))
         (count-var (make-symbol "COUNT"))
         (end (make-symbol "VECTOR-END"))
         (references (vector-data-references loop))
         (dispatch (and *vector-wide-predicate*
                        (vector-wide-available-p element-type)
                        t))
         (wide (and (vector-wide-available-p element-type)
                    (vector-elementwise-plan loop 32 count-var end references)))
         (narrow (vector-elementwise-plan loop 16 count-var end references)))
    (cond ((and wide narrow dispatch)
           (vector-emit-elementwise loop count-var end references wide narrow t))
;;; only the 256-bit tables can express this loop, so it is still asked for at run
;;; time: a machine without avx2 runs the element loop instead of trapping
          ((and wide dispatch)
           (vector-emit-elementwise loop count-var end references wide nil t))
          (wide (vector-emit-elementwise loop count-var end references wide nil nil))
          (narrow (vector-emit-elementwise loop count-var end references nil narrow nil))
          (t nil))))

;;; expand a reduction: accumulate the elementwise expression in a vector, run it
;;; over the tail one element at a time, then add the accumulator's lanes to the
;;; variable.
(defun vector-expand-reduction (loop)
  (let* ((element-type (vloop-element-type loop))
         (width (vector-width element-type))
         (index (vloop-index loop))
         (count (vloop-count loop))
         (accumulator (vloop-accumulator loop))
         (expression (vloop-expression loop))
         (scalars (mapcar (lambda (leaf) (cons leaf (make-symbol "SPLAT")))
                          (vector-collect-scalars expression)))
         (arrays (vloop-sources loop))
         (pack-type (vector-pack-type element-type))
         (accumulating (make-symbol "ACCUM"))
         (lanes (make-symbol "LANES"))
         (end (make-symbol "VECTOR-END"))
         (lane (make-symbol "LANE"))
         (load (vector-primitive "LOAD" element-type))
         (store (vector-primitive "STORE" element-type))
         (splat (vector-primitive "SPLAT" element-type))
         (add (vector-primitive "ADD" element-type))
         (offset (and (vector-primitive "OFFSET" element-type) (make-symbol "OFFSET")))
         (arithmetic (mapcar (lambda (operator)
                               (vector-primitive (vector-operator-base operator element-type)
                                                 element-type))
                             (vector-collect-operators expression))))
    (when (and load store splat add (every #'identity arithmetic)
               (every (lambda (base) (vector-primitive base element-type))
                      (vector-expression-primitives expression element-type))
               (not (memq :index (vector-collect-operators expression)))
               (or (vector-float-element-type-p element-type)
                   (vloop-unchecked-p loop))
;;; no head peel: a reduction reads its sources from 0, so a literal negative offset
;;; is not taken, and a variable one is checked by the generated guard.
               (every (lambda (entry)
                        (every (lambda (offset)
                                 (or (not (integerp offset)) (>= offset 0)))
                               (second entry)))
                      arrays))
      (when *auto-vectorize-notes*
        (compiler-notify "~@<vectorized reduction over ~a, ~a elements at a time; ~
                          the sum is computed modulo the element type, ~
                          ~a~:>"
                         element-type width
                         (if (vector-float-element-type-p element-type)
                             "which reorders the additions"
                             (concatenate
                              'string
                              "so a total that overflows the element type differs from "
                              "the scalar loop, which does not truncate it"))))
      `(if (and (typep ,count 'index)
                ,@(loop for entry in arrays
                        for name = (first entry)
                        for offsets = (second entry)
                        append (cons `(>= (length ,name) (+ ,count (max 0 ,@offsets)))
                                     (loop for offset in offsets
                                           unless (typep offset '(integer 0 *))
                                             collect `(>= ,offset 0)))))
           (let* (,@(mapcar (lambda (scalar)
                              `(,(cdr scalar) (,splat ,(second (car scalar)))))
                            scalars)
                  (,accumulating (,splat 0))
                  (,lanes (make-array ,width :element-type ',element-type))
                  (,end (- ,count (mod ,count ,width))))
             (declare (type index ,end)
                      (type ,pack-type ,accumulating)
                      ,@(mapcar (lambda (scalar) `(type ,pack-type ,(cdr scalar))) scalars)
                      (dynamic-extent ,lanes))
             (do ((,index 0 (+ ,index ,width)))
                 ((>= ,index ,end))
               (declare (type index ,index))
               ,(if offset
                    `(let ((,offset (,(vector-primitive "OFFSET" element-type)
                                     ,index)))
                       (declare (type (unsigned-byte 64) ,offset))
                       (setf ,accumulating
                             (,add ,accumulating
                                   ,(vector-emit expression element-type index scalars
                                                 nil offset))))
                    `(setf ,accumulating
                           (,add ,accumulating
                                 ,(vector-emit expression element-type index scalars
                                               nil)))))
             ;; the tail, one element at a time, exactly as written
             (do ((,index ,end (1+ ,index)))
                 ((>= ,index ,count))
               (declare (type index ,index))
               ,@(vloop-body loop))
             ;; and the lanes into the variable
             (,store ,accumulating ,lanes
                     ,(if offset
                          `(,(vector-primitive "OFFSET" element-type) 0)
                          0))
             (dotimes (,lane ,width)
               (incf ,accumulator (aref ,lanes ,lane)))
             nil)
           ,(vloop-fallback loop)))))

;;;; missed-optimization notes
;;;
;;; the pass declines silently and the most common reason is a type declaration the
;;; user did not write, which is the one thing they can fix, so the pass explains a
;;; loop with a shape it would otherwise rewrite.

;;; The index, count and body of a loop in one of the shapes the pass takes, with
;;; no regard for types.
(defun vector-shape-loop (form)
  (cond ((and (consp form) (eq (car form) 'dotimes)
              (consp (second form)) (null (cddr (second form))))
         (values (first (second form)) (second (second form)) (cddr form)))
        ((and (consp form) (memq (car form) '(do do*)))
         (let ((bindings (second form))
               (end-clause (third form)))
           (when (and (consp bindings) (null (cdr bindings))
                      (consp end-clause) (null (cdr end-clause))
                      (consp (first bindings)))
             (destructuring-bind (index &optional (init nil init-p) (step nil step-p))
                 (first bindings)
               (declare (ignore init))
               (let ((test (first end-clause)))
                 (when (and (symbolp index) init-p step-p
                            (vector-increments-index-p step index)
                            (consp test) (= (length test) 3))
                   (values index (third test) (cdddr form))))))))
        ((and (consp form) (eq (car form) 'loop))
         (vector-shape-loop-clauses (cdr form)))))

;;; (loop for index [of-type ...] [from 0] below/upto/below count do body)
(defun vector-shape-loop-clauses (clauses)
  (let ((index nil) (count nil) (body '()))
    (loop while clauses
          do (case (car clauses)
               (for (push (second clauses) index) (setf clauses (cddr clauses)))
               (below (push (second clauses) count) (setf clauses (cddr clauses)))
               (upto (push (second clauses) count) (setf clauses (cddr clauses)))
               (do (push (second clauses) body) (setf clauses (cddr clauses)))
               (of-type (setf clauses (cddr clauses)))
               (from (setf clauses (cddr clauses)))
               (t (return-from vector-shape-loop-clauses nil))))
    (when (and index count (= (length index) 1) (= (length count) 1) (= (length body) 1))
      (values (first index) (first count) body))))

;;; Is EXPRESSION built only from the operators, array references and constants the
;;; pass understands?  Returns (values T arrays variables) or NIL.  The arrays and
;;; variables are returned rather than accumulated in place, since rebinding a
;;; parameter would not be visible to the caller.
(defun vector-shape-expression (expression index)
  (cond ((numberp expression) (values t '() '()))
        ((symbolp expression)
         (if (keywordp expression)
             (values nil nil nil)
             (values t '() (list expression))))
        ((and (consp expression) (eq (car expression) 'aref) (= (length expression) 3)
              (symbolp (second expression)))
         (if (vector-subscript-offset (third expression) index)
             (values t (list (second expression)) '())
             (values nil nil nil)))
        ((and (consp expression) (eq (car expression) 'float) (null (cddr expression))
              (eq (second expression) index))
         (values t '() '()))
        ((and (consp expression) (eq (car expression) 'coerce) (= (length expression) 3)
              (eq (second expression) index))
         (values t '() '()))
        ((and (consp expression) (eq (car expression) '-) (null (cddr expression)))
         (vector-shape-expression (second expression) index))
        ((and (consp expression)
              (memq (car expression) '(+ - * / logand logior logxor min max)))
         (let ((arrays '()) (scalars '()))
           (dolist (argument (cdr expression))
             (multiple-value-bind (ok argument-arrays argument-scalars)
                 (vector-shape-expression argument index)
               (unless ok
                 (return-from vector-shape-expression (values nil nil nil)))
               (setf arrays (append argument-arrays arrays)
                     scalars (append argument-scalars scalars))))
           (values t arrays scalars)))
        (t (values nil nil nil))))

;;; The reasons FORM, a loop in this pass's shape, was not rewritten.
(defun vector-missed-diagnosis (form env)
  (multiple-value-bind (index count body) (vector-shape-loop form)
    (unless (and index count (= (length body) 1))
      (return-from vector-missed-diagnosis nil))
    (multiple-value-bind (declarations statements) (vector-split-declarations body)
      (declare (ignore declarations))
      (unless (= (length statements) 1)
        (return-from vector-missed-diagnosis nil))
      (let ((store (vector-normalize-store (first statements))))
        (unless (and (consp store) (eq (car store) 'setf) (null (cdddr store)))
          (return-from vector-missed-diagnosis nil))
        (let ((place (second store)))
          (unless (and (consp place) (eq (car place) 'aref) (= (length place) 3)
                       (eq (third place) index) (symbolp (second place)))
            (return-from vector-missed-diagnosis nil))
          (multiple-value-bind (shape arrays-in-expression)
              (vector-shape-expression (third store) index)
            (unless shape
              (return-from vector-missed-diagnosis nil))
            (vector-diagnose-types (adjoin (second place) arrays-in-expression) env
                                   (third store) (policy env (= safety 0)) index)))))))

;;; The operator symbols an elementwise expression is built from.
(defun vector-shape-operators (expression)
  (if (consp expression)
      (cond ((memq (car expression) '(aref float coerce)) '())
            ((eq (car expression) '-)
             (cons '- (vector-shape-operators (second expression))))
            (t (cons (car expression)
                     (mapcan #'vector-shape-operators (cdr expression)))))
      '()))

;;; A subscript that reads before the loop variable cannot be vectorized, and the
;;; scalar loop it replaces would fault at index 0 anyway.  Say so, because the
;;; fix is a rewrite the user can do.
(defun vector-negative-offset-reason (form index)
  (labels ((walk (form)
             (cond ((not (consp form)) nil)
                   ((and (eq (car form) 'aref) (= (length form) 3)
                         (consp (third form)))
                    (multiple-value-bind (offset literal)
                        (vector-subscript-offset (third form) index)
                      (cond ((and literal (minusp offset))
                             (format nil "~s is read at ~s, before the loop variable; ~
                                          loop over the interior and shift the destination ~
                                          instead, as in (setf (aref out (+ i 1)) ...)"
                                     (second form) (third form)))
                            (t (walk (third form))))))
                   (t (some #'walk (cdr form))))))
    (walk form)))

;;; What stops ARRAYS and EXPRESSION from being vectorized, in the voice of a
;;; compiler note?  Two things can: an array with no usable declaration, or lanes
;;; the element type does not have, either an operator without a primitive or byte
;;; lanes outside (safety 0).
(defun vector-diagnose-types (arrays lexenv expression unchecked-p index)
  (let ((element-type nil)
        (reads '())
        (unregistered '())
        (reasons '()))
;;; the destination comes first: it is the element type the loop would run at, and
;;; every array after it is read, possibly at a narrower width than the lanes hold
    (dolist (name arrays)
      (multiple-value-bind (kind type) (vector-variable-info name lexenv)
        (cond ((null kind)
               (push name unregistered))
              ((eq kind :array)
;;; TYPE is dotted: the element type in the CAR, the simple-array flag in the CDR
               (if element-type
                   (push (list name (car type) (cdr type)) reads)
                   (setf element-type (car type)))))))
    (dolist (name (nreverse unregistered))
      (push (if element-type
                (format nil "~s is not known to be an array; a (declare (type (simple-array ~a (*)) ~s)) ~
                             would allow this loop to be vectorized"
                        name element-type name)
                (format nil "~s is not known to be an array; declaring its element type would ~
                             allow this loop to be vectorized"
                        name))
            reasons))
    (let ((reason (vector-negative-offset-reason expression index)))
      (when reason (push reason reasons)))
    (when element-type
;;; the operator having a base is not the whole answer: the base names a primitive,
;;; and whether this build has it is the table's business.  pmulld needs sse4.1 where
;;; vpmulld needs avx2, so a loop can be available at one width and not the other, and
;;; 64-bit lanes have no multiply at either.
      ;; LABELS, not FLET: widen-reasons asks available-p which primitives this
      ;; build has
      (labels ((available-p (base)
                 (or (vector-primitive base element-type 16)
                     (vector-primitive base element-type 32)))
;;; and no packed load reads an array of another element type into these lanes
;;; unless the backend has the widening one: nothing reads a 32-bit or 64-bit
;;; integer array into float lanes at any width, nothing reads a double-float array
;;; into single-float ones, and the widening loads take a simple array only.
             (widen-reasons ()
               (loop for (name source-type simple-p) in (nreverse reads)
                     unless (equal source-type element-type)
                       collect (let ((base (vector-widen-base source-type element-type)))
                                 (cond ((null base)
                                        (format nil "~s holds ~a, and no packed load reads ~a into ~a lanes"
                                                name source-type source-type element-type))
                                       ((not (available-p base))
                                        (format nil "~s holds ~a, and this build has no packed load ~
                                                     that reads ~a into ~a lanes"
                                                name source-type source-type element-type))
                                       ((not simple-p)
                                        (format nil "~s holds ~a and is not a simple array, which the ~
                                                     packed load into ~a lanes does not take"
                                                name source-type element-type)))))))
        (dolist (operator (remove-duplicates (vector-shape-operators expression) :test #'eq))
          (let ((base (vector-operator-base operator element-type)))
            (unless (and base (available-p base))
              (push (format nil "~a is not available for ~a lanes" operator element-type)
                    reasons))))
        (dolist (reason (widen-reasons))
          (when reason (push reason reasons))))
      (unless (or (vector-float-element-type-p element-type) unchecked-p
                  (vector-exact-expression-p expression))
        (push (format nil "~a lanes wrap where a checked loop would signal, so they are only ~
                           used at (safety 0) or in a loop whose value is built from loads, ~
                           broadcasts, MIN, MAX and the bitwise operations"
                      element-type)
              reasons)))
    (nreverse reasons)))

(defun vector-note-missed-optimization (form env)
  (when *auto-vectorize-notes*
    (dolist (reason (vector-missed-diagnosis form env))
      (compiler-notify "~@<unable to vectorize this loop: ~2I~_~A~:>" reason))))

;;;; Bindings that prove their own type
;;;;
;;;; A loop over an array bound by LET does not see the array's type: the leaf's
;;;; ctype is the universal type until IR1 propagates the initializer's type to it,
;;;; and the pass runs before that.  Two initializers say what they are without any
;;;; propagation, though, so the pass declares them where IR1 can see it: a
;;;; MAKE-ARRAY with an :ELEMENT-TYPE and no adjustment, fill pointer or
;;;; displacement is a simple array of that element type, and a THE cast says what
;;;; it says.  Adding the declaration is a source rewrite like any other here, and
;;;; the declaration is what the initializer already guarantees.

;;; the element type a MAKE-ARRAY call pins, or NIL.  Anything that would make the
;;; result non-simple or adjustable disqualifies it, since only the declaration of a
;;; simple array is what the loop then relies on.
(defun vector-make-array-element-type (form)
  (when (and (consp form) (eq (car form) 'make-array) (cddr form))
    (let ((element-type nil)
          (simple-p t)
          (arguments (cddr form)))
      (loop while arguments
            for key = (pop arguments)
            do (let ((value (and arguments (pop arguments))))
                 (case key
                   (:element-type
                    (when (and (consp value) (eq (car value) 'quote) (= 2 (length value)))
                      (setf element-type (second value))))
                   ((:adjustable :fill-pointer :displaced-to :displaced-index-offset)
                    (when value (setf simple-p nil))))))
      (when (and simple-p element-type
                 (member element-type +vector-element-types+ :test #'equal))
        element-type))))

;;; the type a binding's initializer proves, as a specifier to declare, or NIL.
(defun vector-binding-type (form)
  (cond ((vector-make-array-element-type form)
         `(simple-array ,(vector-make-array-element-type form) (*)))
        ((and (consp form) (eq (car form) 'the) (= 3 (length form)))
         ;; the cast is what the user asserted, and declaring the binding with it is
         ;; the same assertion applied to the variable
         (let ((info (ignore-errors (vector-ctype-info (specifier-type (second form))))))
           (when (and info (eq (car info) :array))
             (second form))))))

;;; whether any form in SOURCE is a loop the pass would consider, so that
;;; declarations are only added to code where they can matter
(defun vector-source-has-loop-p (source)
  ;; the walk has to survive dotted lists: a declare specifier such as
  ;; (flags symbols values ignore . binding-decls) is one, and SOME would reject it
  (labels ((walk (form)
             (cond ((atom form) nil)
                   ((memq (car form) '(dotimes do do* loop)) t)
                   ((eq (car form) 'quote) nil)
                   (t (or (walk (car form)) (walk (cdr form)))))))
    (walk source)))

;;;; Entry point

;;; a LET or LET* whose bindings prove array types, and whose body has a loop: the
;;; same form with those types declared, so that the loop is vectorized.  Adding the
;;; declarations is all this does.
(defun vector-let-form (form lexenv)
;;; (locally (declare ...) (dotimes ...)) arrives as a LET with no bindings: the
;;; declarations still apply to whatever the loop becomes, so rewrite it in place
  (when (and (consp form)
             (memq (car form) '(let let*))
             (null (second form))
             (vector-standard-loop-form-p form lexenv))
    (let ((body (copy-list (cddr form))))
      (loop for tail on body
            for rewritten = (vectorize-loop-form (car tail) lexenv)
            when rewritten
              do (return `(,(car form) nil ,@(ldiff body tail) ,rewritten ,@(cdr tail)))
            finally (return nil))))
  (when (and (consp form)
             (memq (car form) '(let let*))
             (listp (second form))
             (vector-standard-loop-form-p form lexenv)
             (vector-source-has-loop-p `(progn ,@(cddr form))))
    (let ((declarations '()))
      (dolist (binding (second form))
        (when (and (consp binding) (symbolp (car binding))
                   (cdr binding) (null (cddr binding)))
          (let ((type (vector-binding-type (second binding))))
            (when type
              (push `(type ,type ,(car binding)) declarations)))))
      (when declarations
        (let ((body (copy-list (cddr form)))
              (head '()))
          ;; declarations keep their place at the head of the body
          (loop while (and body (consp (car body)) (eq (caar body) 'declare))
                do (push (pop body) head))
          `(,(car form) ,(second form)
            ,@(nreverse head)
            (declare ,@(nreverse declarations))
            ,@body))))))

;;; the replacement for form, a loop this pass understands, or nil.  called from
;;; IR1-CONVERT for each form as it is converted, which is before macroexpansion: it
;;; sees what the user wrote, so a loop that a macro expands to is not rewritten.
(defun vector-standard-loop-form-p (form lexenv)
  (not (nth-value 1 (lexenv-find (car form) funs :lexenv lexenv))))

;;; the manual documents this pass as a speed 2 optimization: the code is emitted
;;; twice, so do not spend it on code compiled for space.
;;; the replacement for FORM, or NIL, and whether converting it should turn the pass
;;; off entirely.  A loop becomes the vectorized loop, whose expansion is opaque; a
;;; LET whose bindings prove array types becomes itself with those types declared,
;;; and only that rewrite lets its body reach the pass.
(defun vectorize-form (form lexenv)
  (when (and *auto-vectorize*
             (policy lexenv (>= speed 2))
             (consp form))
    (if (memq (car form) '(let let*))
        (values (vector-let-form form lexenv) nil)
        (values (vectorize-loop-form form lexenv) t))))

(defun vectorize-loop-form (form lexenv)
  (when (and *auto-vectorize*
             (policy lexenv (>= speed 2))
             (consp form)
             (memq (car form) '(dotimes do do* loop))
             (vector-standard-loop-form-p form lexenv))
    (let* ((plan (case (car form)
                   (loop (vector-match-loop form lexenv))
                   ((do do*) (vector-match-do form lexenv))
                   (t (or (vector-match-dotimes form lexenv)
                          (vector-match-reduction form lexenv)))))
;;; a plan can still be declined while expanding, over lanes that need (safety 0) or
;;; a primitive the backend does not have, and that is the case worth explaining.
           (replacement (and plan (vector-expand-loop plan))))
      (if replacement
          replacement
          (progn (vector-note-missed-optimization form lexenv) nil)))))

;;; converting the form this pass returns must not re-enter the pass, or the pass
;;; would rewrite its own fallback branch, the original loop.
(defvar *vectorize-inhibited* nil)
(defvar *vectorize-inhibited-forms* nil)
