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
      (logxor (and (not floats) "XOR")))))

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
          (t (let ((element-type (vector-element-type spec)))
               (if element-type (values :array element-type) (values nil nil)))))))

;;; Strip any (THE type form) wrappers of FORM, returning (VALUES form type).
(defun vector-unwrap-the (form)
  (let ((type nil))
    (loop while (and (consp form) (eq (car form) 'the))
          do (setf type (second form) form (third form)))
    (values form type)))

;;; (VALUES ARRAY ELEMENT-TYPE) if FORM names a declared float array, following
;;; (THE (SIMPLE-ARRAY SINGLE-FLOAT (*)) name) as well as plain declarations.
(defun vector-array-type (form env)
  (multiple-value-bind (inner spec) (vector-unwrap-the form)
    (let ((element-type (and spec (vector-element-type spec))))
      (cond (element-type (values inner element-type))
            ((and (symbolp inner) (null spec))
             (multiple-value-bind (kind type) (vector-variable-info inner env)
               (if (eq kind :array) (values inner type) (values nil nil))))
            (t (values nil nil))))))

;;;; Recognizing a loop

;;; What a recognized loop turned into. EXPRESSION is a tree of
;;;   (:ARRAY name)  a W element slice read from a declared array
;;;   (:SCALAR sym)  a float variable, broadcast
;;;   (:CONST . number) a float constant, broadcast
;;;   (:OP operator subexpression ...) elementwise float arithmetic
(defstruct (vector-loop (:conc-name vloop-))
  element-type index count destination destination-offset sources expression body fallback)

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
(defun vector-subscript-offset (subscript index)
  (cond ((eq subscript index) 0)
        ((and (consp subscript) (eq (car subscript) '+))
         (let ((arguments (cdr subscript)))
           (when (and (= (length arguments) 2)
                      (member index arguments :test #'eq))
             (let ((other (if (eq (first arguments) index)
                              (second arguments)
                              (first arguments))))
               (when (typep other '(integer 0 *))
                 other)))))
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
             (let ((offset (vector-subscript-offset (car subscripts) index)))
               (when offset
                 (multiple-value-bind (name type) (vector-array-type array env)
                   (when (equal type element-type)
                     (list :array name offset))))))))
        ((memq (car form) '(+ - * / logand logior logxor))
         ;; unary minus is not handled: (- x) differs from (- 0 x) on zeroes
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
(defun vector-match-element-store (form env index)
  (when (and (consp form) (eq (car form) 'setf) (null (cdddr form)))
    (let ((place (second form)))
      (when (and (consp place) (eq (car place) 'aref))
        (destructuring-bind (array . subscripts) (cdr place)
          (when (null (cdr subscripts))
            (let ((offset (vector-subscript-offset (car subscripts) index)))
              (when offset
                (multiple-value-bind (name element-type) (vector-array-type array env)
                  (when name
                    (let ((expression
                            (vector-match-expression (third form) env index element-type)))
                      (when expression
                        (values name element-type expression offset)))))))))))))

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
                (multiple-value-bind (destination element-type expression offset)
                    (vector-match-element-store (car statements) env index)
                  (when destination
                    (make-vector-loop
                     :element-type element-type
                     :index index
                     :count count
                     :destination destination
                     :destination-offset offset
                     :sources (vector-collect-arrays expression)
                     :expression expression
                     :body (car statements)
                     :fallback form)))))))))))

;;; The arrays read by an expression tree, as an alist of name to the largest
;;; constant offset the array is read at, in order of first use.
(defun vector-collect-arrays (expression &optional (accumulator '()))
  (ecase (car expression)
    (:array (let ((known (assoc (second expression) accumulator)))
              (if known
                  (progn (setf (cdr known) (max (cdr known) (third expression))) accumulator)
                  (append accumulator
                          (list (cons (second expression) (third expression)))))))
    (:op (dolist (subexpression (cddr expression) accumulator)
           (setf accumulator (vector-collect-arrays subexpression accumulator))))
    ((:scalar :const) accumulator)))

;;; The loop invariant parts of an expression tree, in order of first use.
(defun vector-collect-scalars (expression &optional (accumulator '()))
  (ecase (car expression)
    (:array accumulator)
    (:op (dolist (subexpression (cddr expression) accumulator)
           (setf accumulator (vector-collect-scalars subexpression accumulator))))
    ((:scalar :const) (adjoin expression accumulator :test #'equal))))

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
                (multiple-value-bind (destination element-type expression offset)
                    (vector-match-element-store (second clauses) env index)
                  (when destination
                    (make-vector-loop
                     :element-type element-type
                     :index index
                     :count count
                     :destination destination
                     :destination-offset offset
                     :sources (vector-collect-arrays expression)
                     :expression expression
                     :body (second clauses)
                     :fallback form)))))))))))

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
  (if (eq (car expression) :op)
      (dolist (subexpression (cddr expression)
               (adjoin (second expression) accumulator))
        (setf accumulator (vector-collect-operators subexpression accumulator)))
      accumulator))

;;; Emit an expression tree over SIMD packs. SCALARS maps each loop invariant
;;; leaf to the variable holding its broadcast pack.
(defun vector-emit (expression element-type index scalars)
  (ecase (car expression)
    (:array (if (zerop (third expression))
                `(,(vector-primitive "LOAD" element-type) ,(second expression) ,index)
                `(,(vector-primitive "LOAD" element-type) ,(second expression)
                  (+ ,index ,(third expression)))))
    (:op (let ((operator (second expression)))
           (let ((name (vector-primitive (vector-operator-base operator element-type) element-type)))
             (reduce (lambda (left right) `(,name ,left ,right))
                     (mapcar (lambda (subexpression)
                               (vector-emit subexpression element-type index scalars))
                             (cddr expression))))))
    ((:scalar :const)
     ;; splats are computed once, before the loop, and live in an SSE register
     (cdr (assoc expression scalars :test #'equal)))))

;;; The arrays a loop touches, as an alist of name to the largest constant
;;; offset it is used at, destination first.
(defun vector-loop-arrays (loop)
  (cons (cons (vloop-destination loop) (vloop-destination-offset loop))
        (remove (vloop-destination loop) (vloop-sources loop)
                :key #'car :test #'eq)))

;;; Return the vectorized form for a matched loop, or NIL if this backend cannot
;;; express it (which is the normal answer on a build whose backend has not
;;; published its primitives, and for byte lanes outside SAFETY 0).
(defun vector-expand-loop (loop)
  (let* ((element-type (vloop-element-type loop))
         (width (vector-width element-type))
         (index (vloop-index loop))
         (count (vloop-count loop))
         (destination-offset (vloop-destination-offset loop))
         (end (make-symbol "VECTOR-END"))
         (expression (vloop-expression loop))
         (scalars (mapcar (lambda (leaf) (cons leaf (make-symbol "SPLAT")))
                          (vector-collect-scalars expression)))
         (pack-type `(simd-pack ,element-type))
         (load (vector-primitive "LOAD" element-type))
         (store (vector-primitive "STORE" element-type))
         (splat (and scalars (vector-primitive "SPLAT" element-type)))
         (arithmetic (mapcar (lambda (operator)
                               (vector-primitive (vector-operator-base operator element-type)
                                                 element-type))
                             (vector-collect-operators expression))))
    (when (and load store (every #'identity arithmetic) (or splat (null scalars))
               (or (/= width 16) *vectorize-byte-lanes*))
      (when *auto-vectorize-notes*
        (compiler-notify "vectorized loop over ~a, ~a elements at a time"
                         element-type width))
      `(if (and (typep ,count 'index)
                ;; every array has to cover the last element the vector body
                ;; touches, which is (n - 1) plus that array's offset
                ,@(mapcar (lambda (entry)
                            `(>= (length ,(car entry)) (+ ,count ,(cdr entry))))
                          (vector-loop-arrays loop)))
           (let* (,@(mapcar (lambda (scalar)
                              `(,(cdr scalar) (,splat ,(second (car scalar)))))
                            scalars)
                  (,end (- ,count (mod ,count ,width))))
             (declare (type index ,end)
                      ,@(mapcar (lambda (scalar) `(type ,pack-type ,(cdr scalar))) scalars))
             ;; the vector body
             (do ((,index 0 (+ ,index ,width)))
                 ((>= ,index ,end))
               (declare (type index ,index))
               (,store ,(vector-emit expression element-type index scalars)
                       ,(vloop-destination loop)
                       ,(if (zerop destination-offset)
                            index
                            `(+ ,index ,destination-offset))))
             ;; the tail, element by element, exactly as written
             (do ((,index ,end (1+ ,index)))
                 ((>= ,index ,count))
               (declare (type index ,index))
               ,(vloop-body loop)))
           ;; not the shape we want: the loop the user wrote
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
        ((dotimes loop)
         (let ((plan (if (eq (car form) 'dotimes)
                         (vector-match-dotimes form env)
                         (vector-match-loop form env))))
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
      (let ((name (leaf-%source-name var))
            (type (lambda-var-type var)))
        (when (and (symbolp name) (not (eq type *universal-type*)))
          (push (cons name (type-specifier type)) env))))))

;;; Called on the body of every lambda before it is converted to IR1. Returns a
;;; body that is either unchanged or has its elementwise loops vectorized.
(defun vectorize-lambda-body (body lexenv &optional local-policy vars)
  (if (and *auto-vectorize*
           (member :x86-64 *features*)
           (policy (or local-policy lexenv) (>= speed 2)))
      (let ((*vectorize-byte-lanes* (policy (or local-policy lexenv) (= safety 0))))
        (vector-vectorize-body body (vector-parameter-environment vars)))
      body))
