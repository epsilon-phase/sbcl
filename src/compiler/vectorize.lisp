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
;;;; The pass runs on the *source* of a lambda body before IR1 conversion, so
;;;; what it knows are DECLARATIONS, not inferred types: an array takes part only
;;;; if it is declared (SIMPLE-ARRAY SINGLE-FLOAT (*)) or the DOUBLE-FLOAT
;;;; equivalent. In exchange the pass is small and cannot break code it does not
;;;; understand: every form it does not recognize is returned untouched, and every
;;;; rewrite keeps the original loop as the fallback branch when the arrays are
;;;; too short to cover N.
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
(defun vector-element-type (spec)
  (when (and (consp spec) (eq (car spec) 'simple-array))
    (let ((element-type (second spec)))
      (when (vector-supported-element-type-p element-type)
        element-type))))

;;; How many elements the backend handles at a time: 128 bits worth.
(defun vector-width (element-type)
  (cond ((eq element-type 'single-float) 4)
        ((eq element-type 'double-float) 2)
        ((eql (second element-type) 8) 16)
        ((eql (second element-type) 16) 8)
        ((eql (second element-type) 32) 4)
        (t 2)))

;;; Byte lanes only carry the meaning of the scalar loop when the scalar loop is
;;; not checking the element type: at any safety above 0, storing a byte that
;;; does not fit signals a type error, where a packed store truncates.  Set by
;;; VECTORIZE-LAMBDA-BODY from the policy and read when expanding.
(defvar *vectorize-byte-lanes* nil)

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
      ;; SSE2 has byte-wise minimum and maximum for unsigned lanes only
      (min (and (equal element-type '(unsigned-byte 8)) "MIN"))
      (max (and (equal element-type '(unsigned-byte 8)) "MAX")))))

;;; Extend ENV (an alist of variable name to declared type spec) by the
;;; declarations found at the head of FORMS.
(defun vector-extend-env (forms env)
  (dolist (form forms env)
    (when (and (consp form) (eq (car form) 'declare))
      (dolist (spec (cdr form))
        (case (car spec)
          (type
           (destructuring-bind (type-spec . variables) (cdr spec)
             (dolist (variable variables)
               (push (cons variable type-spec) env))))
          ;; the standard abbreviations may be used without TYPE
          ((single-float double-float fixnum index)
           (dolist (variable (cdr spec))
             (push (cons variable (car spec)) env))))))))

;;; (VALUES KIND ELEMENT-TYPE) for a variable, KIND being :SCALAR when it is
;;; declared to be a float itself and :ARRAY when it is a declared float array.
(defun vector-variable-info (variable env)
  (let ((spec (cdr (assq variable env))))
    (cond ((null spec) (values nil nil))
          ((vector-supported-element-type-p spec) (values :scalar spec))
          (t (let ((element-type (vector-array-element-type spec)))
               (if element-type
                   (values :array (cons element-type (vector-simple-spec-p spec)))
                   (values nil nil)))))))

;;; Strip any (THE type form) wrappers of FORM, returning (VALUES form type).
(defun vector-unwrap-the (form)
  (let ((type nil))
    (loop while (and (consp form) (eq (car form) 'the))
          do (setf type (second form) form (third form)))
    (values form type)))

;;; (VALUES ARRAY ELEMENT-TYPE) if FORM names a declared float array, following
;;; (THE (SIMPLE-ARRAY SINGLE-FLOAT (*)) name) as well as plain declarations.
;;; (VALUES NAME ELEMENT-TYPE SIMPLE-P) for an array reference, or NILs.  A
;;; declaration of (VECTOR element-type) or (ARRAY element-type (*)) tells us the
;;; element type but not whether the data is inline, so SIMPLE-P is NIL there and
;;; the generated code fetches the data vector once with WITH-ARRAY-DATA.
(defun vector-array-type (form env)
  (multiple-value-bind (inner spec) (vector-unwrap-the form)
    (let ((element-type (and spec (vector-array-element-type spec))))
      (cond (element-type (values inner element-type (vector-simple-spec-p spec)))
            ((and (symbolp inner) (null spec))
             (multiple-value-bind (kind type) (vector-variable-info inner env)
               (if (eq kind :array)
                   (values inner (car type) (cdr type))
                   (values nil nil nil))))
            (t (values nil nil nil))))))

;;; The element type of a declared array type, whether or not it is simple.
(defun vector-array-element-type (spec)
  (when (consp spec)
    (cond ((eq (car spec) 'simple-array) (vector-element-type spec))
          ((eq (car spec) 'vector)
           (and (vector-supported-element-type-p (second spec)) (second spec)))
          ((eq (car spec) 'array)
           (and (vector-supported-element-type-p (second spec)) (second spec))))))

(defun vector-simple-spec-p (spec)
  (and (consp spec) (eq (car spec) 'simple-array)))

;;;; Recognizing a loop

;;; What a recognized loop turned into. EXPRESSION is a tree of
;;;   (:ARRAY name)  a W element slice read from a declared array
;;;   (:SCALAR sym)  a float variable, broadcast
;;;   (:CONST . number) a float constant, broadcast
;;;   (:OP operator subexpression ...) elementwise float arithmetic
(defstruct (vector-loop (:conc-name vloop-))
  element-type index count destination destination-offset destination-simple-p
  destination-literal-p sources expression body fallback accumulator)

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
(defun vector-make-array-type (form)
  (when (and (consp form) (eq (car form) 'make-array))
    (let ((element-type nil)
          (displaced-p nil))
      ;; (MAKE-ARRAY dimensions :element-type ...): the keywords start at the
      ;; second argument
      (loop for (key value) on (cddr form) by #'cddr
            do (case key
                 (:element-type (setf element-type value))
                 ((:adjustable :fill-pointer :displaced-to) (setf displaced-p t))))
      (when (and element-type (not displaced-p))
        (let ((spec (if (and (consp element-type) (eq (car element-type) 'quote))
                        (second element-type)
                        element-type)))
          (let ((element (cond ((member spec '(single-float double-float) :test #'equal) spec)
                               ((and (consp spec)
                                     (memq (car spec) '(signed-byte unsigned-byte)))
                                spec)
                               (t nil))))
            (when (and element (vector-supported-element-type-p element))
              `(simple-array ,element (*)))))))))

;;; Extend ENV by the bindings of a LET/LET* that are MAKE-ARRAY forms with a
;;; literal element type.
(defun vector-extend-env-from-bindings (bindings env)
  (dolist (binding bindings env)
    (when (and (consp binding) (symbolp (first binding)))
      (let ((type (and (cdr binding) (vector-make-array-type (second binding)))))
        (when type (push (cons (first binding) type) env))))))

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
               (cond ((and (typep other '(integer 0 *)) ) (values other t))
                     ((symbolp other) (values other nil)))))))
        (t nil)))

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
         (multiple-value-bind (kind type) (vector-variable-info form env)
           (when (and (eq kind :scalar) (equal type element-type))
             (list :scalar form))))
        ((atom form) nil)
        ((eq (car form) 'the)
         ;; only a THE that pins the element type exactly is usable
         (when (equal (second form) element-type)
           (vector-match-expression (third form) env index element-type)))
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
  (when (and (consp form) (eq (car form) 'dotimes))
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
                     :fallback form)))))))))))

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
    ((:scalar :const) accumulator)))

;;; The loop invariant parts of an expression tree, in order of first use.
(defun vector-collect-scalars (expression &optional (accumulator '()))
  (case (car expression)
    (:array accumulator)
    (:op (dolist (subexpression (cddr expression) accumulator)
           (setf accumulator (vector-collect-scalars subexpression accumulator))))
    (:negate (vector-collect-scalars (second expression) accumulator))
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
                     :fallback form)))))))))))

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
  (when (and (consp form) (memq (car form) '(do do*)))
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
                             :fallback form)))))))))))))))

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
  (when (and (consp form) (eq (car form) 'dotimes))
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
                                      (vector-match-expression expression env index type)))
                              (when expression
                                (make-vector-loop
                                 :element-type type
                                 :index index
                                 :count count
                                 :accumulator place
                                 :sources (vector-collect-arrays expression)
                                 :expression expression
                                 :body (first statements)
                                 :fallback form)))))))))))))))))

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

(defun vector-arithmetic-base (operator)
  (ecase operator (+ "ADD") (- "SUB") (* "MUL") (/ "DIV")))

;;; The operators an expression tree uses, in order of first use.
(defun vector-collect-operators (expression &optional (accumulator '()))
  (case (car expression)
    (:op (dolist (subexpression (cddr expression)
             (adjoin (second expression) accumulator))
           (setf accumulator (vector-collect-operators subexpression accumulator))))
    (:negate (adjoin :negate (vector-collect-operators (second expression) accumulator)))
    (t accumulator)))

;;; Emit an expression tree over SIMD packs. SCALARS maps each loop invariant
;;; leaf to the variable holding its broadcast pack.
(defun vector-emit (expression element-type index scalars data-references)
  (ecase (car expression)
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
  (let ((destination (cons (list (vloop-destination-offset loop))
                           (vloop-destination-simple-p loop))))
    (cons (cons (vloop-destination loop) destination)
          (remove (vloop-destination loop) (vloop-sources loop)
                  :key #'car :test #'eq))))

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
         (arithmetic (mapcar (lambda (operator)
                               (vector-primitive (vector-operator-base operator element-type)
                                                 element-type))
                             (vector-collect-operators expression))))
    (when (and load store
               (every #'identity arithmetic)
               (or splat (null scalars))
               (or (/= width 16) *vectorize-byte-lanes*))
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
                                                 index references)))
                    ;; the tail, element by element, exactly as written
                    (do ((,index ,end (1+ ,index)))
                        ((>= ,index ,count))
                      (declare (type index ,index))
                      ,(vloop-body loop))))))
          `(if (and (typep ,count 'index) ,@guard)
               (let* (,@(mapcar (lambda (scalar)
                                  `(,(cdr scalar) (,splat ,(second (car scalar)))))
                                scalars)
                      (,end (- ,count (mod ,count ,width))))
                 (declare (type index ,end)
                          ,@(mapcar (lambda (scalar)
                                      `(type ,pack-type ,(cdr scalar)))
                                    scalars))
                 ,body)
               ,(vloop-fallback loop)))))))

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
               (or (/= width 16) *vectorize-byte-lanes*))
      (when *auto-vectorize-notes*
        (compiler-notify "vectorized reduction over ~a, ~a elements at a time"
                         element-type width))
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

;;;; Walking a body

(defun vector-vectorize-body (forms env)
  (multiple-value-bind (declarations statements) (vector-split-declarations forms)
    (let ((env (vector-extend-env declarations env)))
      (append declarations
              (mapcar (lambda (form) (vector-vectorize-form form env)) statements)))))

;;; Rewrite the loops we understand in FORM, leaving everything else alone.
;;; Only the constructs listed here are descended into; anything else is opaque.
(defun vector-vectorize-form (form env)
  (if (atom form)
      form
      (case (car form)
        ((quote declare) form)
        (function (if (and (consp (second form)) (eq (car (second form)) 'lambda))
                      `(function ,(vector-vectorize-form (second form) env))
                      form))
        (dolist `(dolist ,(second form) ,@(vector-vectorize-body (cddr form) env)))
        ((dotimes loop do do*)
         (let ((plan (case (car form)
                       (loop (vector-match-loop form env))
                       ((do do*) (vector-match-do form env))
                       (t (or (vector-match-dotimes form env)
                              (vector-match-reduction form env))))))
           (or (and plan (vector-expand-loop plan)) form)))
        (lambda `(lambda ,(second form) ,@(vector-vectorize-body (cddr form) env)))
        ((let let*)
         (multiple-value-bind (declarations statements) (vector-split-declarations (cddr form))
           `(,(car form)
             ,(mapcar (lambda (binding)
                        (if (consp binding)
                            (list (car binding) (vector-vectorize-form (second binding) env))
                            binding))
                      (second form))
             ,@declarations
             ,@(vector-vectorize-body
                statements
                (vector-extend-env
                 declarations
                 (vector-extend-env-from-bindings (second form) env))))))
        (locally
         (multiple-value-bind (declarations statements) (vector-split-declarations (cdr form))
           `(locally ,@declarations
                     ,@(vector-vectorize-body statements (vector-extend-env declarations env)))))
        (progn (cons 'progn (vector-vectorize-body (cdr form) env)))
        (the `(the ,(second form) ,(vector-vectorize-form (third form) env)))
        ((flet labels)
         `(,(car form)
           ,(mapcar (lambda (definition)
                      `(,(car definition) ,(second definition)
                        ,@(vector-vectorize-body (cddr definition) env)))
                    (second form))
           ,@(vector-vectorize-body (cddr form) env)))
        (multiple-value-bind
            (destructuring-bind (variables value-form . body) (cdr form)
              `(multiple-value-bind ,variables ,(vector-vectorize-form value-form env)
                 ,@(vector-vectorize-body body env))))
        ((when unless and or prog1 prog2) (cons (car form) (vector-vectorize-body (cdr form) env)))
        (if (destructuring-bind (test then &optional else) (cdr form)
              `(if ,(vector-vectorize-form test env)
                   ,(vector-vectorize-form then env)
                   ,@(when else (list (vector-vectorize-form else env))))))
        (cond `(cond ,@(mapcar (lambda (clause)
                                 (cons (vector-vectorize-form (car clause) env)
                                       (vector-vectorize-body (cdr clause) env)))
                               (cdr form))))
        ((block catch return-from)
         (if (eq (car form) 'return-from)
             `(return-from ,(second form) ,(vector-vectorize-form (third form) env))
             `(,(car form) ,(second form) ,@(vector-vectorize-body (cddr form) env))))
        (unwind-protect
         `(unwind-protect ,(vector-vectorize-form (second form) env)
                           ,@(vector-vectorize-body (cddr form) env)))
        (t form))))

;;;; Entry point

;;; The declared types of a lambda's parameters, as an alist for the walker.
;;; IR1 has already applied the body's declarations to the vars by the time the
;;; body is handed to us, so this is where (DECLARE (TYPE (SIMPLE-ARRAY
;;; SINGLE-FLOAT (*)) a)) shows up for a function's arguments.
(defun vector-parameter-environment (vars)
  (let ((env '()))
    (dolist (var vars env)
      (let ((name (leaf-%source-name var)))
        (when (symbolp name)
          ;; An FTYPE declaration does not set the var's type; the compiler
          ;; records it in the leaf's defined type instead, so both are read.
          (dolist (ctype (list (lambda-var-type var) (leaf-defined-type var)))
            (unless (or (null ctype) (eq ctype *universal-type*))
              (let ((spec (type-specifier ctype)))
                ;; a variable of a supported scalar type is a broadcast operand,
                ;; and a declaration of (VECTOR T) or (ARRAY T (*)) counts as a
                ;; usable array: its element type is known and the data vector is
                ;; found at run time
                (when (or (vector-supported-element-type-p spec)
                          (vector-array-element-type spec))
                  (push (cons name spec) env))))))))))

;;; Called on the body of every lambda before it is converted to IR1. Returns a
;;; body that is either unchanged or has its elementwise loops vectorized.
(defun vectorize-lambda-body (body lexenv &optional local-policy vars)
  (if (and *auto-vectorize*
           (member :x86-64 *features*)
           (policy (or local-policy lexenv) (>= speed 2)))
      (let ((*vectorize-byte-lanes* (policy (or local-policy lexenv) (= safety 0))))
        (vector-vectorize-body body (vector-parameter-environment vars)))
      body))
