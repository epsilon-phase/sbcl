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

;;;; Deciding what a variable is declared to be

;;; The float element type of a declared simple array, or NIL.
(defun vector-element-type (spec)
  (when (and (consp spec) (eq (car spec) 'simple-array))
    (let ((element-type (second spec)))
      (when (memq element-type '(single-float double-float))
        element-type))))

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
          ((memq spec '(single-float double-float)) (values :scalar spec))
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
  element-type index count destination sources expression body)

;;; Match an elementwise expression over N-element slices, or return NIL.
(defun vector-match-expression (form env index element-type)
  (cond ((numberp form)
         (when (typep form element-type) (list :const form)))
        ((symbolp form)
         (multiple-value-bind (kind type) (vector-variable-info form env)
           (when (and (eq kind :scalar) (eq type element-type))
             (list :scalar form))))
        ((atom form) nil)
        ((eq (car form) 'the)
         ;; only a THE that pins the element type exactly is usable
         (when (eq (second form) element-type)
           (vector-match-expression (third form) env index element-type)))
        ((eq (car form) 'aref)
         (destructuring-bind (array . subscripts) (cdr form)
           (when (and (null (cdr subscripts)) (eq (car subscripts) index))
             (multiple-value-bind (name type) (vector-array-type array env)
               (when (eq type element-type)
                 (list :array name))))))
        ((memq (car form) '(+ - * /))
         ;; unary minus is not handled: (- x) differs from (- 0 x) on zeroes
         (let ((arguments (cdr form)))
           (when (cdr arguments)
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
          (when (and (null (cdr subscripts)) (eq (car subscripts) index))
            (multiple-value-bind (name element-type) (vector-array-type array env)
              (when name
                (let ((expression (vector-match-expression (third form) env index element-type)))
                  (when expression
                    (values name element-type expression)))))))))))

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
                     (or (symbolp count) (typep count 'unsigned-byte)))
            (multiple-value-bind (declarations statements) (vector-split-declarations body)
              (when (and (null (cdr statements)) (not (vector-special-declaration-p declarations)))
                (multiple-value-bind (destination element-type expression)
                    (vector-match-element-store (car statements) env index)
                  (when destination
                    (make-vector-loop
                     :element-type element-type
                     :index index
                     :count count
                     :destination destination
                     :sources (vector-collect-arrays expression)
                     :expression expression
                     :body (car statements))))))))))))

;;; The arrays read by an expression tree, in order of first use.
(defun vector-collect-arrays (expression &optional (accumulator '()))
  (ecase (car expression)
    (:array (adjoin (second expression) accumulator))
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
    (:array `(,(vector-primitive "LOAD" element-type) ,(second expression) ,index))
    (:op (let ((operator (second expression)))
           (let ((name (vector-primitive (vector-arithmetic-base operator) element-type)))
             (reduce (lambda (left right) `(,name ,left ,right))
                     (mapcar (lambda (subexpression)
                               (vector-emit subexpression element-type index scalars))
                             (cddr expression))))))
    ((:scalar :const)
     ;; splats are computed once, before the loop, and live in an SSE register
     (cdr (assoc expression scalars :test #'equal)))))

;;; Return the vectorized form for a matched loop, or NIL if this backend cannot
;;; express it (which is the normal answer while the target compiler is still
;;; being built, before the backend primitives exist).
(defun vector-expand-loop (loop)
  (let* ((element-type (vloop-element-type loop))
         (width (if (eq element-type 'single-float) 4 2))
         (index (vloop-index loop))
         (count (vloop-count loop))
         (end (make-symbol "VECTOR-END"))
         (expression (vloop-expression loop))
         (scalars (mapcar (lambda (leaf) (cons leaf (make-symbol "SPLAT")))
                          (vector-collect-scalars expression)))
         (pack-type `(simd-pack ,element-type))
         (load (vector-primitive "LOAD" element-type))
         (store (vector-primitive "STORE" element-type))
         (splat (and scalars (vector-primitive "SPLAT" element-type)))
         (arithmetic (mapcar (lambda (operator)
                               (vector-primitive (vector-arithmetic-base operator) element-type))
                             (vector-collect-operators expression))))
    (when (and load store (every #'identity arithmetic) (or splat (null scalars)))
      `(if (and (typep ,count 'index)
                ,@(mapcar (lambda (array) `(>= (length ,array) ,count))
                          (cons (vloop-destination loop) (vloop-sources loop))))
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
                       ,index))
             ;; the tail, element by element, exactly as written
             (do ((,index ,end (1+ ,index)))
                 ((>= ,index ,count))
               (declare (type index ,index))
               ,(vloop-body loop)))
           ;; not the shape we want: the loop the user wrote
           (dotimes (,index ,count) ,(vloop-body loop))))))

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
        (dotimes (let ((loop (vector-match-dotimes form env)))
                   (or (and loop (vector-expand-loop loop)) form)))
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
             ,@(vector-vectorize-body statements (vector-extend-env declarations env)))))
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
      (vector-vectorize-body body (vector-parameter-environment vars))
      body))
