;;;; Automatic vectorization of elementwise loops.
;;;;
;;;; SBCL has no loop vectorizer: SIMD is available either through hand-written
;;;; SB-SIMD intrinsics or not at all. This pass adds the missing half: it
;;;; recognizes the classic elementwise array loop
;;;;
;;;;   (dotimes (i n) (setf (aref c i) (+ (aref a i) (aref b i))))
;;;;
;;;; and rewrites it into a W-wide loop over register-resident SIMD packs plus a
;;;; scalar tail loop, where W is 4 for single-float and 2 for double-float
;;;; (SSE2, which is baseline for x86-64, so no runtime CPU dispatch is needed).
;;;;
;;;; The pass runs inside IR1 conversion, on each form as it is converted, so what
;;;; it sees is macroexpanded code: a loop written inline and one produced by a
;;;; macro arrive identically.  The types it uses are the compiler's own - the ctype
;;;; of each variable's leaf, in the lexenv the form is being converted in - so a
;;;; declaration, a THE cast, a let-bound MAKE-ARRAY result and an inferred type all
;;;; count, and shadowed names and nested LOCALLY scopes are handled by the compiler
;;;; rather than by this pass.  What it does with them is still small and local: it
;;;; returns a replacement form, which IR1 then converts like any other, and every
;;;; rewrite keeps the original loop as the fallback branch when the arrays are too
;;;; short to cover N.
;;;;
;;;; The elements each iteration touches are read before any of them is written,
;;;; so the rewrite is valid even when the destination aliases a source.

(in-package "SB-C")

;;; Set to NIL to compile without automatic vectorization.
(defvar *auto-vectorize* t)

;;; When true, a vectorized loop produces a compiler note naming the element
;;; type and how many elements are handled at a time.
(defvar *auto-vectorize-notes* t)

;;;; Deciding what a variable is declared to be

;;; The element types this backend can move and compute a vector at a time.
(defun vector-supported-element-type-p (element-type)
  (member element-type '(single-float double-float
                         (signed-byte 8) (unsigned-byte 8)
                         (signed-byte 16) (unsigned-byte 16)
                         (signed-byte 32) (unsigned-byte 32)
                         (signed-byte 64) (unsigned-byte 64))
          :test #'equal))

(defun vector-float-element-type-p (element-type)
  (memq element-type '(single-float double-float)))

;;; The element type of a declared simple array, or NIL.
;;; How many elements the backend handles at a time: 128 bits worth.
(defun vector-width (element-type)
  (cond ((eq element-type 'single-float) 4)
        ((eq element-type 'double-float) 2)
        ((eql (second element-type) 8) 16)
        ((eql (second element-type) 16) 8)
        ((eql (second element-type) 32) 4)
        (t 2)))

;;; Integer lanes only carry the meaning of the scalar loop when the scalar loop is
;;; not checking the element type: above (safety 0), storing a value that does not
;;; fit signals a type error, where a packed store wraps.  Read from the lexenv when
;;; a loop is matched, and carried on the plan, so a nested LOCALLY is honoured.
;;; The primitive base name for OPERATOR over ELEMENT-TYPE, or NIL when this
;;; pair is not something the backend can do a vector at a time.  Multiplication
;;; and division are float only (the packed integer forms are narrower than the
;;; element type); the bitwise operations are integer only.
(defun vector-operator-base (operator element-type)
  (let ((floats (vector-float-element-type-p element-type)))
    (case operator
      (+ "ADD")
      (- "SUB")
      (* (and floats "MUL"))
      (/ (and floats "DIV"))
      (logand (and (not floats) "AND"))
      (logior (and (not floats) "OR"))
      (logxor (and (not floats) "XOR"))
      (:negate (and floats "NEG"))
      (:index (and floats "INDEX"))
      ;; SSE2 has byte-wise minimum and maximum for unsigned lanes only
      (min (and (equal element-type '(unsigned-byte 8)) "MIN"))
      (max (and (equal element-type '(unsigned-byte 8)) "MAX")))))

;;; Extend ENV (an alist of variable name to declared type spec) by the
;;; declarations found at the head of FORMS.
;;; The element types the backend can move and compute a vector at a time, in the
;;; order the classifier below tries them.
(defparameter +vector-element-types+
  '(single-float double-float
    (signed-byte 8) (unsigned-byte 8)
    (signed-byte 16) (unsigned-byte 16)
    (signed-byte 32) (unsigned-byte 32)
    (signed-byte 64) (unsigned-byte 64)))

;;; The ctype of a variable at the point where a form is being converted, or NIL.
;;; This is the compiler's own answer - declarations, THE casts, let-bound
;;; MAKE-ARRAY results and inferred types all arrive here already resolved - so
;;; the vectorizer does not have to model types itself.
(defun vector-variable-ctype (variable lexenv)
  (multiple-value-bind (var found) (lexenv-find variable vars :lexenv lexenv)
    (when (and found (lambda-var-p var))
      (lambda-var-type var))))

;;; What CTYPE says about a value this pass can use, as
;;;   (:ARRAY element-type . simple-p)  an array to load and store elements of
;;;   (:SCALAR . element-type)          a value of that type, to be broadcast
;;; or NIL for anything else.  Simple arrays are tried first, because a
;;; declaration of (VECTOR T) or (ARRAY T (*)) says what the elements are but not
;;; that the data is inline; those get their data vector once with WITH-ARRAY-DATA.
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

;;; Strip any (THE type form) wrappers of FORM, returning (VALUES form type).
(defun vector-unwrap-the (form)
  (let ((type nil))
    (loop while (and (consp form) (eq (car form) 'the))
          do (setf type (second form) form (third form)))
    (values form type)))

;;; (VALUES NAME ELEMENT-TYPE SIMPLE-P) for an array reference, or NILs.  A THE
;;; cast is answered from the cast, anything else from the variable's own ctype.
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

;;; What a recognized loop turned into. EXPRESSION is a tree of
;;;   (:ARRAY name)  a W element slice read from a declared array
;;;   (:SCALAR sym)  a float variable, broadcast
;;;   (:CONST . number) a float constant, broadcast
;;;   (:OP operator subexpression ...) elementwise float arithmetic
(defstruct (vector-loop (:conc-name vloop-))
  element-type index count destination destination-offset destination-simple-p
  destination-literal-p sources expression body fallback accumulator unchecked-p)

;;; A loop bound the generated code may evaluate more than once: a variable, a
;;; literal, or arithmetic on one of those, which has no side effects.  A bound
;;; that could do something visible stays with the scalar loop.
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

;;; The declared type implied by a local binding of the form
;;; (MAKE-ARRAY ... :ELEMENT-TYPE 'TYPE), or NIL.  This is not inference: the
;;; element type is written in the source, and the result is a simple array as
;;; long as none of ADJUSTABLE, FILL-POINTER or DISPLACED-TO is given, so it can
;;; be treated exactly like a declaration of that type.
;;; The offset of a subscript from the loop variable: (aref a i) is 0,
;;; (aref a (+ i 3)) and (aref a (+ 3 i)) are 3.  A negative offset would index
;;; before the loop variable's range, so it is not taken.
;;; (VALUES OFFSET LITERAL-P), or NIL if SUBSCRIPT is not the loop variable plus
;;; something.  The something may be a literal, or a variable; a variable can be
;;; negative at run time, so the generated guard checks it, which a literal does
;;; not need.
(defun vector-subscript-offset (subscript index)
  (cond ((eq subscript index) (values 0 t))
        ((and (consp subscript) (eq (car subscript) '+))
         (let ((arguments (cdr subscript)))
           (when (and (= (length arguments) 2)
                      (member index arguments :test #'eq))
             (let ((other (if (eq (first arguments) index)
                              (second arguments)
                              (first arguments))))
                ;; the offset ends up in a guard evaluated outside the loop, so it
                ;; may not mention the loop index
                (cond ((typep other '(integer 0 *)) (values other t))
                      ((and (symbolp other) (not (eq other index)))
                       (values other nil)))))))))

;;; Match an elementwise expression over N-element slices, or return NIL.
(defun vector-match-expression (form env index element-type)
  (cond ((numberp form)
         ;; An integer constant has to fit in a fixnum: anything wider reaches the
         ;; backend as a bignum, which no splat accepts, so the loop is better
         ;; left scalar than rejected later.
         (when (and (typep form element-type)
                    (or (vector-float-element-type-p element-type)
                        (typep form 'fixnum)))
           (list :const form)))
        ((symbolp form)
         ;; the loop index shadows whatever the enclosing scope says about that
         ;; name, so it is never a broadcast operand
         (unless (eq form index)
           (multiple-value-bind (kind type) (vector-variable-info form env)
             (when (and (eq kind :scalar) (equal type element-type))
               (list :scalar form)))))
        ((atom form) nil)
        ((eq (car form) 'the)
         ;; only a THE that pins the element type exactly is usable
         (when (equal (second form) element-type)
           (vector-match-expression (third form) env index element-type)))
        ;; the loop index itself, converted to the element type: a value that
        ;; depends on where it is, like a gradient or a ramp table
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
                   (when (equal type element-type)
                     (list :array name offset simple-p literal-p))))))))
        ;; (- x) is a sign flip for floats, which is not the same as (- 0 x):
        ;; the two disagree on zeroes, so floats get their own packed negation
        ;; and integers subtract from zero, which is exactly what they do.
        ((and (eq (car form) '-) (null (cdr (cdr form))))
         (let ((subexpression (vector-match-expression (second form) env index element-type)))
           (when subexpression
             (if (vector-float-element-type-p element-type)
                 (list :negate subexpression)
                 (list :op '- (list :const 0) subexpression)))))
        ((memq (car form) '(+ - * / logand logior logxor min max))
         (let ((arguments (cdr form)))
           (when (and (cdr arguments) (vector-operator-base (car form) element-type))
             (let ((subexpressions
                     (mapcar (lambda (argument)
                               (vector-match-expression argument env index element-type))
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

(defun vector-match-element-store (form env index)
  (let ((form (vector-normalize-store form)))
    (when (and (consp form) (eq (car form) 'setf) (null (cdddr form)))
    (let ((place (second form)))
      (when (and (consp place) (eq (car place) 'aref))
        (destructuring-bind (array . subscripts) (cdr place)
          (when (null (cdr subscripts))
            (multiple-value-bind (offset literal-p)
                (vector-subscript-offset (car subscripts) index)
              (when offset
                (multiple-value-bind (name element-type simple-p) (vector-array-type array env)
                  (when name
                    (let ((expression
                            (vector-match-expression (third form) env index element-type)))
                        (when expression
                          (values name element-type expression offset
                                  simple-p literal-p))))))))))))))

;;; Split leading DECLARE forms from a body.
(defun vector-split-declarations (forms)
  (let ((declarations '()))
    (loop while (and (consp forms) (consp (car forms)) (eq (caar forms) 'declare))
          do (push (pop forms) declarations))
    (values (nreverse declarations) forms)))

(defun vector-special-declaration-p (declarations)
  (some (lambda (declaration)
          (some (lambda (spec) (memq (car spec) '(special sb-ext:truly-the)))
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
              (when (and (null (cdr statements)) (not (vector-special-declaration-p declarations)))
                (multiple-value-bind (destination element-type expression offset simple-p literal-p)
                    (vector-match-element-store (car statements) env index)
                  (when destination
                    (make-vector-loop
                     :element-type element-type
                     :index index
                     :count count
                     :destination destination
                     :destination-offset offset
                     :destination-simple-p simple-p
                     :destination-literal-p literal-p
                     :sources (vector-collect-arrays expression)
                     :expression expression
                     :body (car statements)
                     :fallback form :unchecked-p (policy env (= safety 0)))))))))))))

;;; The arrays read by an expression tree, as an alist of name to
;;; (OFFSETS . SIMPLE-P), in order of first use, where OFFSETS is every offset the
;;; array is read at.  The guard has to cover the largest of them, and it has to
;;; check any that is not a literal, since a variable offset can be negative.
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
    (:negate (vector-collect-arrays (second expression) accumulator))
    ((:scalar :const :index) accumulator)))

;;; The loop invariant parts of an expression tree, in order of first use.
(defun vector-collect-scalars (expression &optional (accumulator '()))
  (case (car expression)
    (:array accumulator)
    (:op (dolist (subexpression (cddr expression) accumulator)
           (setf accumulator (vector-collect-scalars subexpression accumulator))))
    (:negate (vector-collect-scalars (second expression) accumulator))
    (:index accumulator)
    ((:scalar :const) (adjoin expression accumulator :test #'equal))
    (t accumulator)))

;;; (1- n) and (- n 1) are the same bound as N one step down.
(defun vector-normalize-bound (count)
  (if (and (consp count)
           (or (and (eq (car count) '1-) (null (cddr count)))
               (and (eq (car count) '-) (eql (third count) 1) (null (cdddr count)))))
      (second count)
      count))

;;; A LOOP clause keyword: the symbols in a LOOP form are the user's own, so
;;; they are matched by name rather than by identity.
(defun vector-loop-keyword (form name)
  (and (symbolp form) (string-equal (symbol-name form) name)))

;;; Match the LOOP shapes that mean the same thing as a DOTIMES over an
;;; elementwise body:
;;;
;;;   (loop for index [of-type type] [from 0] below|upto count do body)
;;;
;;; BELOW and UPTO are synonyms in LOOP.  A bound written as (1- n) or (- n 1) is
;;; the same as BELOW N, so it normalizes to that.  Anything else - a nonzero
;;; start, a TO bound, several body forms, other clauses - is left alone.
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
            (let ((count (vector-normalize-bound (second clauses))))
              (setf clauses (cddr clauses))
              (when (and (vector-loop-keyword (car clauses) "DO")
                         (null (cddr clauses))
                         (vector-pure-count-p count))
                (multiple-value-bind (destination element-type expression offset simple-p literal-p)
                    (vector-match-element-store (second clauses) env index)
                  (when destination
                    (make-vector-loop
                     :element-type element-type
                     :index index
                     :count count
                     :destination destination
                     :destination-offset offset
                     :destination-simple-p simple-p
                     :destination-literal-p literal-p
                     :sources (vector-collect-arrays expression)
                     :expression expression
                     :body (second clauses)
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
;;; Anything else - several bindings, a different step, result forms, another
;;; end test - is left alone.
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
                      (when (and (null (cdr statements))
                                 (not (vector-special-declaration-p declarations)))
                        (multiple-value-bind (destination element-type expression
                                              offset simple-p literal-p)
                            (vector-match-element-store (first statements) env index)
                          (when destination
                            (make-vector-loop
                             :element-type element-type
                             :index index
                             :count count
                             :destination destination
                             :destination-offset offset
                             :destination-simple-p simple-p
                             :destination-literal-p literal-p
                             :sources (vector-collect-arrays expression)
                             :expression expression
                             :body (first statements)
                             :fallback form :unchecked-p (policy env (= safety 0)))))))))))))))))

;;; The part of VALUE that is being added to the accumulator PLACE:
;;; (+ place expression) or (+ expression place), and nothing else.
(defun vector-reduction-increment (value place)
  (when (and (consp value) (eq (car value) '+) (= (length value) 3))
    (cond ((eq (second value) place) (third value))
          ((eq (third value) place) (second value)))))

;;; A sum of a fixed-width integer array into a variable of that same width:
;;;
;;;   (dotimes (i count) (incf accumulator expression))
;;;
;;; Integer addition wraps at the declared width whatever the order of the adds, so
;;; accumulating into lanes and adding the lanes up at the end gives the same
;;; answer.  Floating point is not like that, so a float accumulator is refused.
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
                                             :body (first statements)
                                             :fallback form :unchecked-p (policy env (= safety 0)))))
                                  ;; the reduction expansion addresses the sources
                                  ;; directly, so it needs their data inline
                                  (when (vector-loop-all-simple-p plan)
                                    plan))))))))))))))))))

;;;; Generating the vectorized loop

;;; The backend primitives that move and combine one vector of floats: an alist
;;; of ((BASE ELEMENT-TYPE) . NAME), where BASE is "LOAD", "STORE", "SPLAT",
;;; "ADD", "SUB", "MUL" or "DIV". It is filled in by the backend file that
;;; defines them, src/compiler/x86-64/vec.lisp, and is NIL until then, which is
;;; what keeps this pass inert on a build that has no such primitives.
;;;
;;; The names are handed over rather than written here because this file cannot
;;; name the backend's package: a string anywhere in the heap that is STRING= to
;;; a package name defeats string coalescing in SAVE-LISP-AND-DIE, which
;;; tests/hide-packages.test.sh enforces.
(defvar *vector-primitives* nil)

(defun vector-primitive (base element-type)
  (cdr (assoc (list base element-type) *vector-primitives* :test #'equal)))

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
(defun vector-emit (expression element-type index scalars data-references)
  (ecase (car expression)
    (:index (cdr (assoc :index scalars)))
    (:negate `(,(vector-primitive "NEG" element-type)
               ,(vector-emit (second expression) element-type index scalars
                             data-references)))
    (:array (let* ((name (second expression))
                   (offset (third expression))
                   (reference (or (cadr (assoc name data-references)) name)))
              `(,(vector-primitive "LOAD" element-type) ,reference
                ,(vector-subscript name offset index data-references))))
    (:op (let ((operator (second expression)))
           (let ((name (vector-primitive (vector-operator-base operator element-type) element-type)))
             (reduce (lambda (left right) `(,name ,left ,right))
                     (mapcar (lambda (subexpression)
                               (vector-emit subexpression element-type index scalars
                                            data-references))
                             (cddr expression))))))
    ((:scalar :const)
     ;; splats are computed once, before the loop, and live in an SSE register
     (cdr (assoc expression scalars :test #'equal)))))

;;; The arrays a loop touches, as an alist of name to (OFFSETS . SIMPLE-P),
;;; destination first.
(defun vector-loop-arrays (loop)
  "The arrays a loop touches: the destination first, then the sources.  When the
destination is also a source - (setf (aref c i) (+ (aref c (+ i 3)) ...)) - its
offsets are merged into the destination's entry rather than dropped, so the length
guard still covers the reads."
  (let* ((destination (vloop-destination loop))
         (same (find destination (vloop-sources loop) :key #'car :test #'eq))
         (entry (cons (append (list (vloop-destination-offset loop))
                              (and same (second same)))
                      (vloop-destination-simple-p loop))))
    (cons (cons destination entry)
          (remove destination (vloop-sources loop) :key #'car :test #'eq))))

;;; Whether every array a plan reads has its data inline.  A reduction has no
;;; destination - its result is a variable - so only the sources matter.
(defun vector-loop-all-simple-p (loop)
  (every (lambda (entry) (cddr entry)) (vloop-sources loop)))

;;; The data vector and start index to use for each array that is not known to be
;;; simple, so that the loop body can address a plain simple array.
(defun vector-data-references (loop)
  (loop for (name offsets . simple-p) in (vector-loop-arrays loop)
        unless simple-p
          collect (cons name (cons (make-symbol "DATA") (make-symbol "START")))))

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
(defun vector-expand-loop (loop)
  (if (vloop-accumulator loop)
      (vector-expand-reduction loop)
      (vector-expand-elementwise-loop loop)))

(defun vector-expand-elementwise-loop (loop)
  (let* ((element-type (vloop-element-type loop))
         (width (vector-width element-type))
         (index (vloop-index loop))
         (count (vloop-count loop))
         (destination-offset (vloop-destination-offset loop))
         (end (make-symbol "VECTOR-END"))
         (expression (vloop-expression loop))
         (scalars (mapcar (lambda (leaf) (cons leaf (make-symbol "SPLAT")))
                          (vector-collect-scalars expression)))
         (arrays (vector-loop-arrays loop))
         (references (vector-data-references loop))
         (pack-type `(simd-pack ,element-type))
         (load (vector-primitive "LOAD" element-type))
         (store (vector-primitive "STORE" element-type))
         (splat (and scalars (vector-primitive "SPLAT" element-type)))
         (operators (vector-collect-operators expression))
         (arithmetic (mapcar (lambda (operator)
                               (vector-primitive (vector-operator-base operator element-type)
                                                 element-type))
                             operators))
         (indexed (and (memq :index operators) t))
         (ramp (and indexed (vector-primitive "RAMP" element-type)))
         (stride (and indexed (vector-primitive "STRIDE" element-type))))
    (when indexed
      ;; the lane offsets for the first iteration, then a constant step per
      ;; iteration: converting and broadcasting the index each time costs four
      ;; instructions and a dependency chain, where keeping the pack costs one add
      (setf scalars (acons :index (make-symbol "INDEX-LANES") scalars))
      (setf scalars (acons :stride (make-symbol "STRIDE") scalars)))
    (when (and load store
               (every #'identity arithmetic)
               (if indexed (and ramp stride) t)
               (or splat (null (remove-if (lambda (scalar)
                                            (memq (car scalar) '(:index :stride)))
                                          scalars)))
               (or (vector-float-element-type-p element-type)
                   (vloop-unchecked-p loop)))
      (when *auto-vectorize-notes*
        (compiler-notify "vectorized loop over ~a, ~a elements at a time"
                         element-type width))
      (flet ((reference (name) (or (cadr (assoc name references)) name)))
        (let ((guard
                (loop for entry in arrays
                      for name = (first entry)
                      for offsets = (second entry)
                      append (cons `(>= (length ,name) (+ ,count (max ,@offsets)))
                                   (loop for offset in offsets
                                         unless (typep offset '(integer 0 *))
                                           collect `(>= ,offset 0)))))
              (body
                (vector-with-data-vectors
                 references
                 `(progn
                    ;; the vector body
                    (do ((,index 0 (+ ,index ,width)))
                        ((>= ,index ,end))
                      (declare (type index ,index))
                      (,store ,(vector-emit expression element-type index scalars references)
                              ,(reference (vloop-destination loop))
                              ,(vector-subscript (vloop-destination loop)
                                                 destination-offset
                                                 index references))
                      ,@(when indexed
                          ;; leave the pack holding the offsets for the next
                          ;; iteration
                          (let ((lanes (cdr (assoc :index scalars)))
                                (stride (cdr (assoc :stride scalars))))
                            `((setf ,lanes
                                    (,(vector-primitive "ADD" element-type)
                                     ,lanes ,stride))))))
                    ;; the tail, element by element, exactly as written
                    (do ((,index ,end (1+ ,index)))
                        ((>= ,index ,count))
                      (declare (type index ,index))
                      ,(vloop-body loop))))))
          ;; The guard chooses END rather than branching between two copies of the
          ;; loop: when it fails, END is 0, the vector loop does not run, and the
          ;; tail loop - which is the original loop, written once - does all the
          ;; work.  So the loop is emitted twice in total (packed and scalar)
          ;; instead of three times, which matters because the code is what makes
          ;; compiling loop-heavy code slower.
          `(let* (,@(mapcar (lambda (scalar)
                             (case (car scalar)
                               (:stride `(,(cdr scalar)
                                          (,(vector-primitive "STRIDE" element-type))))
                               (:index `(,(cdr scalar) (,ramp)))
                               (t `(,(cdr scalar)
                                    (,splat ,(second (car scalar)))))))
                           scalars)
                  (,end (if (and (typep ,count 'index) ,@guard)
                            (- ,count (mod ,count ,width))
                            0)))
             (declare (type index ,end)
                      ,@(mapcar (lambda (scalar)
                                  `(type ,pack-type ,(cdr scalar)))
                                scalars))
             ,body))))))

;;; Expand a reduction: add the elementwise expression into a vector accumulator,
;;; run the same expression over the tail one element at a time, then add the
;;; accumulator's lanes to the variable.  The lanes are stored to a stack-allocated
;;; array and added with ordinary arithmetic, which keeps the result exact and needs
;;; no horizontal-add instruction.
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
         (pack-type `(simd-pack ,element-type))
         (accumulating (make-symbol "ACCUM"))
         (lanes (make-symbol "LANES"))
         (end (make-symbol "VECTOR-END"))
         (lane (make-symbol "LANE"))
         (load (vector-primitive "LOAD" element-type))
         (store (vector-primitive "STORE" element-type))
         (splat (vector-primitive "SPLAT" element-type))
         (add (vector-primitive "ADD" element-type))
         (arithmetic (mapcar (lambda (operator)
                               (vector-primitive (vector-operator-base operator element-type)
                                                 element-type))
                             (vector-collect-operators expression))))
    (when (and load store splat add (every #'identity arithmetic)
               (not (memq :index (vector-collect-operators expression)))
               (or (vector-float-element-type-p element-type)
                   (vloop-unchecked-p loop)))
      (when *auto-vectorize-notes*
        (compiler-notify "~@<vectorized reduction over ~a, ~a elements at a time; ~
                          the sum is computed modulo the element type, ~
                          ~a~:>"
                         element-type width
                         (if (vector-float-element-type-p element-type)
                             "which reorders the additions"
                             "so a total that would overflow it differs from the scalar ~
                              loop, where such an overflow was already undefined")))
      `(if (and (typep ,count 'index)
                ,@(loop for entry in arrays
                        for name = (first entry)
                        for offsets = (second entry)
                        append (cons `(>= (length ,name) (+ ,count (max ,@offsets)))
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
               (setf ,accumulating
                     (,add ,accumulating
                           ,(vector-emit expression element-type index scalars nil))))
             ;; the tail, one element at a time, exactly as written
             (do ((,index ,end (1+ ,index)))
                 ((>= ,index ,count))
               (declare (type index ,index))
               ,(vloop-body loop))
             ;; and the lanes into the variable
             (,store ,accumulating ,lanes 0)
             (dotimes (,lane ,width)
               (incf ,accumulator (aref ,lanes ,lane)))
             nil)
           ,(vloop-fallback loop)))))

;;;; Missed-optimization notes.
;;;;
;;;; The pass declines silently, and the most common reason is a type declaration
;;;; the user did not write.  That is the one thing they can fix, so a loop whose
;;;; SHAPE is the one this pass rewrites - same control form, same single store,
;;;; same elementwise expression - that was nevertheless not rewritten gets a note
;;;; saying what was missing.  This is the same service SBCL does when a vop's
;;;; translation fails: "unable to open-code X due to type uncertainty".
;;;;
;;;; Notes are only issued for a near miss, never for a loop that was simply not
;;;; this pass's business, so ordinary code that happens to contain loops stays
;;;; quiet.

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
        ((and (consp expression) (memq (car expression) '(+ * / logand logior logxor min max)))
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
                                   (third store) (policy env (= safety 0)))))))))

;;; The operator symbols an elementwise expression is built from.
(defun vector-shape-operators (expression)
  (if (consp expression)
      (cond ((memq (car expression) '(aref float coerce)) '())
            ((eq (car expression) '-)
             (cons '- (vector-shape-operators (second expression))))
            (t (cons (car expression)
                     (mapcan #'vector-shape-operators (cdr expression)))))
      '()))

;;; What stops ARRAYS and EXPRESSION from being vectorized, in the voice of a
;;; compiler note?  Two things can: an array with no usable declaration, or lanes
;;; the element type does not have - an operator without a primitive, or byte lanes
;;; outside (safety 0).
(defun vector-diagnose-types (arrays lexenv expression unchecked-p)
  (let ((element-type nil)
        (unregistered '())
        (reasons '()))
    (dolist (name arrays)
      (multiple-value-bind (kind type) (vector-variable-info name lexenv)
        (cond ((and (eq kind :array) (null element-type))
               (setf element-type (car type)))
              ((null kind)
               (push name unregistered)))))
    (dolist (name (nreverse unregistered))
      (push (if element-type
                (format nil "~s is not known to be an array; a (declare (type (simple-array ~a (*)) ~s)) ~
                             would allow this loop to be vectorized"
                        name element-type name)
                (format nil "~s is not known to be an array; declaring its element type would ~
                             allow this loop to be vectorized"
                        name))
            reasons))
    (when element-type
      (dolist (operator (remove-duplicates (vector-shape-operators expression) :test #'eq))
        (unless (vector-operator-base operator element-type)
          (push (format nil "~a is not available for ~a lanes" operator element-type)
                reasons)))
      (unless (or (vector-float-element-type-p element-type) unchecked-p)
        (push (format nil "~a lanes wrap where a checked loop would signal, so they are ~
                           only used when compiled at (safety 0)"
                      element-type)
              reasons)))
    (nreverse reasons)))

;;; Report them, once per loop.
(defun vector-note-missed-optimization (form env)
  (when *auto-vectorize-notes*
    (dolist (reason (vector-missed-diagnosis form env))
      (compiler-notify "~@<unable to vectorize this loop: ~2I~_~A~:>" reason))))

;;;; Entry point

;;; The replacement for FORM, a loop this pass understands, or NIL.  Called from
;;; IR1-CONVERT for each form as it is converted, so FORM is macroexpanded code - a
;;; loop written inline and one generated by a macro arrive here identically - and
;;; LEXENV is the environment it is being converted in, which is what makes the type
;;; lookups right for shadowed names, (THE ...) casts and a nested LOCALLY's
;;; OPTIMIZE declarations.
;;; A control form is syntax only when its name is not bound in the lexenv: a
;;; MACROLET or FLET of DOTIMES, DO or LOOP means something else by the form, and the
;;; compiler is the one that knows.  (Redefining one of them globally needs a package
;;; lock violation, so it is not a case worth paying for here.)  A snapshot of the
;;; standard macro functions would not work anyway: this file is loaded by the host
;;; compiler during the build, so the functions it captured would be the host's.
(defun vector-standard-loop-form-p (form lexenv)
  (not (nth-value 1 (lexenv-find (car form) funs :lexenv lexenv))))

(defun vectorize-loop-form (form lexenv)
  (when (and *auto-vectorize*
             (consp form)
             (memq (car form) '(dotimes do do* loop))
             (vector-standard-loop-form-p form lexenv))
    (let* ((plan (case (car form)
                   (loop (vector-match-loop form lexenv))
                   ((do do*) (vector-match-do form lexenv))
                   (t (or (vector-match-dotimes form lexenv)
                          (vector-match-reduction form lexenv)))))
           ;; a plan can still be declined while expanding - lanes that need
           ;; (safety 0), a primitive the backend does not have - and that is
           ;; exactly the case worth explaining
           (replacement (and plan (vector-expand-loop plan))))
      (if replacement
          replacement
          (progn (vector-note-missed-optimization form lexenv) nil)))))

;;; Converting the form this pass returns must not re-enter the pass, or the
;;; fallback branch - the original loop - would be rewritten too.  IR1-CONVERT binds
;;; this while, and only while, it converts a replacement.
(defvar *vectorize-inhibited* nil)
