;;;; vectorizing elementwise loops at the IR1 level
;;;;
;;;; The second of the two passes in this feature, and the only one that needs IR1:
;;;; it recognizes an elementwise loop after LOOP-ANALYZE has run, so the body has
;;;; been macroexpanded, inline-expanded and let-converted, and arithmetic that
;;;; exists only as a call in the source -- a DECLAIM INLINE helper -- is the
;;;; arithmetic it sees.  It reads the loop into a plan, prints the plan back as a
;;;; DOTIMES, and lets the source-form pass (src/compiler/vectorize.lisp) rewrite
;;;; that: this file adds a recognizer, not an emitter, and the only thing it takes
;;;; from that file is the primitive table the plan checks the element type against.
;;;;
;;;; Where it keeps its state, and why:
;;;;
;;;; - the loops are found here, by VECTOR-NODE-FIND-LOOPS, and not by LOOP-ANALYZE.
;;;;   The pass runs inside the IR1 fixpoint, in the middle of optimization, and an
;;;;   analysis that writes into every block of the component leaves state the next
;;;;   round has to trust and cannot revalidate: FIND-DOMINATORS calls DFO-AS-NEEDED,
;;;;   and FIND-DFO deletes every block it cannot reach, including the BIND block of
;;;;   a functional still queued on COMPONENT-NEW-FUNCTIONALS that local call
;;;;   analysis has not made reachable yet (dfo.lisp:51-69).  A predecessor walk
;;;;   from each retreating edge needs none of that, and it writes nothing back.
;;;;
;;;; - the replacement is converted into a block of its own, and into the loop's
;;;;   exit start ctran, which is how WRAP-PREDICATE converts into an existing block
;;;;   (ir1tran.lisp:1200-1221).  The loop's own blocks are then unlinked and left
;;;;   for the next round's DFO sweep: DELETE-LAMBDA deliberately leaves a deleted
;;;;   lambda's blocks queued (ir1util.lisp:2191-2194), so hand deletion here would
;;;;   leave the ENTRY bookkeeping inconsistent.  The complete record of the
;;;;   attempts that got this wrong, and of the two bugs that the switch-on self
;;;;   build found afterwards, is doc/internals-notes/vectorizer-ir1-pass.txt.
;;;;
;;;; - the pass never runs in the cross-compiler (*VECTOR-NODE-CROSS-COMPILING*).
;;;;   Its output is an optimization of the compiler's own code when it runs there,
;;;;   and make-host-2 loads every host fasl -- the backend's primitive table
;;;;   included -- before it compiles the first target file, so it is not inert:
;;;;   vectorizing src/compiler/perfect-hash.lisp made the emitted AVX2 dispatch a
;;;;   direct call to SB-VM::AVX2-AVAILABLE-P in a file that precedes the one which
;;;;   proclaims it INLINE, which make-host-2 treats as fatal.

(in-package "SB-C")

;;; LVAR-TYPE caches its answer in the LVAR itself (node.lisp:322-327), and this
;;; pass runs between optimization rounds: a type it cached would be an answer the
;;; pipeline did not ask for, and the next round would read that back instead of
;;; the refined one.  %LVAR-DERIVED-TYPE is the same computation without the cache.
(defun vector-node-lvar-type (lvar)
  "LVAR's type, without caching it in the lvar as LVAR-TYPE would."
  (single-value-type (%lvar-derived-type lvar)))

;;; Set to NIL to compile without this pass.  The source-form pass in vectorize.lisp
;;; has a switch of its own, *AUTO-VECTORIZE*.
(defvar *vector-node-splice* t
  "When true, replace each loop the node analysis plans with what it emits.")

;;; How many loops have been replaced in each component, so that a component whose
;;; replacement the source-form pass declines does not go round the fixpoint for
;;; ever: the printed DOTIMES is itself a natural loop of the same shape.  A weak
;;; table, so a finished component needs no bookkeeping to be forgotten -- and so
;;; that nothing keeps a dead component alive.
(defvar *vector-node-spliced* (make-hash-table :test 'eq :weakness :key))


;;; ---- finding the loops, without asking the pipeline to
;;; LOOP-ANALYZE would give the pass a CLOOP per loop, and FIND-DOMINATORS the
;;; dominator tree it needs.  Both write into every block of the component, and a
;;; pass that runs between optimization rounds cannot hand the component back in
;;; the state the pipeline gave it: what it writes is either stale by the next
;;; round or a snapshot the next round's readers trust.  So the pass finds its own
;;; loops.  What it needs is only which blocks make up the loop, and for a natural
;;; loop that is the retreating edge's head plus every block that reaches the
;;; edge's source without passing through the head -- a predecessor walk over
;;; BLOCK-PRED, which needs no dominators at all.  Nothing is written back: these
;;; loop objects are the pass's own.
(defstruct (vector-node-loop (:conc-name vnloop-))
  ;; the block the retreating edges go to: the loop's test
  head
  ;; every block of the loop, HEAD first
  blocks
  ;; the loops inside this one; the plan refuses a loop that has any
  inferiors)

;;; BLOCK-NUMBER is assigned in reverse DFO (dfo.lisp:62-67), so a retreating edge
;;; goes from a smaller number to a larger one -- the test LOOP-ANALYZE applies
;;; (loop.lisp:84-99).
(defun vector-node-retreating-p (pred block)
  (let ((from (block-number pred)) (to (block-number block)))
    (and from to (<= from to))))

(defun vector-node-natural-loop (head source)
  "HEAD first, then every block that reaches SOURCE without passing through HEAD."
  (let ((seen (list head))
        (work (list source)))
    (loop while work
          do (let ((block (pop work)))
               (unless (memq block seen)
                 (push block seen)
                 (dolist (pred (block-pred block))
                   (unless (memq pred seen) (push pred work))))))
    (cons head (remove head (nreverse seen)))))

(defun vector-node-find-loops (component)
  "Every natural loop of COMPONENT, innermost first, as VECTOR-NODE-LOOPs."
  (let ((by-head (make-hash-table :test 'eq))
        (loops '()))
    (do-blocks (block component)
      (dolist (pred (block-pred block))
        (when (vector-node-retreating-p pred block)
          ;; one loop per head: two retreating edges into the same test are one
          ;; loop, as LOOP-ANALYZE's NOTE-LOOP-HEAD has it
          (let ((loop (gethash block by-head)))
            (unless loop
              (setf loop (make-vector-node-loop :head block))
              (setf (gethash block by-head) loop)
              (push loop loops))
            (let ((blocks (vector-node-natural-loop block pred)))
              (setf (vnloop-blocks loop)
                    (if (vnloop-blocks loop)
                        (union (vnloop-blocks loop) blocks)
                        blocks)))))))
    (dolist (loop loops)
      (setf (vnloop-inferiors loop)
            (remove-if-not (lambda (inner)
                             (and (neq inner loop)
                                  (< (length (vnloop-blocks inner))
                                     (length (vnloop-blocks loop)))
                                  (subsetp (vnloop-blocks inner)
                                           (vnloop-blocks loop))))
                           loops)))
    ;; innermost first, so a loop is looked at after everything it contains
    (sort loops #'< :key (lambda (loop) (length (vnloop-blocks loop))))))


(defvar *vector-node-cross-compiling* #+sb-xc-host t #-sb-xc-host nil
  "True in the compiler that will not produce the final code.

The pass only runs where its output is the program: never in the cross-compiler.
It CAN run there -- make-host-2 loads every host fasl up front, so the backend's
primitive tables are published before the first target file is compiled -- and
splicing there is what broke a switch-on self-build: it vectorized
PHASH-HEXEIGHT's (SETF (AREF KEY-A #:KEY) (LOGAND V 7)) DOTIMES in
src/compiler/perfect-hash.lisp (build-order 165), the source pass rewrote the
DOTIMES this pass printed, and the wide/narrow dispatch that rewrite emits is
(IF (SB-VM::AVX2-AVAILABLE-P) ...) -- a direct call, compiled into a target file
that precedes src/code/x86-64-vm.lisp (515), the file that proclaims
AVX2-AVAILABLE-P INLINE.  make-host-2 treats a warning as fatal:

  ; caught WARNING: Proclaiming SB-VM::AVX2-AVAILABLE-P to be INLINE, but 1 call
  ;   to it was previously compiled

which aborts the build, and the abort then looks like anything but that: what
comes out is a genesis header mismatch or a warm-init failure in an unrelated
file.  Nothing is lost by
declining here: a cross-compiled target file that is vectorized is a build-time
optimization of the compiler's own code, not of the program being built.")

(defun vector-node-component-pass (component)
  (when (and *vector-node-splice*
             (not *vector-node-cross-compiling*)
             ;; several loops per function: the cap is what stops the pass from
             ;; chasing its own output if the emitter ever declines the form we
             ;; spliced in, which is the only way a new loop can appear here
             (< (gethash component *vector-node-spliced* 0) 8)
             ;; the block numbers the loop search walks by come from FIND-DFO, so
             ;; the DFO has to be current: COMPONENT-REANALYZE set means it is not,
             ;; and DFO-AS-NEEDED here would sweep -- deleting every block the DFO
             ;; cannot reach, including the BIND block of a functional still
             ;; queued on COMPONENT-NEW-FUNCTIONALS that LOCALL-ANALYZE-COMPONENT
             ;; has not made reachable yet (dfo.lisp:51-69).  IR1-OPTIMIZE-PHASE-1
             ;; leaves the DFO current whenever it returns without pending work
             ;; (main.lisp:509-529), so declining costs the pass nothing: the next
             ;; round looks again.
             (not (component-reanalyze component)))
    ;; The loops are found here, privately (VECTOR-NODE-FIND-LOOPS), rather than by
    ;; LOOP-ANALYZE: that would write into every block, and this runs inside the
    ;; IR1 fixpoint -- the round after a splice deletes the exits the replacement's
    ;; TAGBODY brings, in a component whose constraint state and environment are
    ;; the live ones, so the analysis has to be the pass's own business.
    (dolist (loop (vector-node-find-loops component))
      ;; the same policy gate the source pass uses (see the manual's "Automatic
      ;; Vectorization"): at low speed the lvars carry no useful types and the
      ;; emitted casts become nonsense like (SIMPLE-ARRAY (INTEGER 0 0) (*))
      (when (and (block-last (vnloop-head loop))
                 (policy (block-last (vnloop-head loop)) (>= speed 2)))
        ;; One splice at a time: it invalidates every block list the search just
        ;; built, and the next round of IR1-PHASES starts over.
        ;;
        ;; A plan attempt that signals declines.  Scope: planning only, never the
        ;; splice.
        (let ((plan (handler-case (vector-node-plan loop) (error () nil))))
          ;; VECTOR-NODE-PLAN returns a plan plist or a failure reason, and reasons
          ;; are keywords as often as lists
          (when (and (consp plan) (getf plan :element-type))
            (when (vector-node-splice-loop loop plan)
              (incf (gethash component *vector-node-spliced* 0))
              (return-from vector-node-component-pass t))))))
    nil))


;;; ------------------------------------------------- recognition, on IR1
;;; After the IR1 fixpoint the body has been macroexpanded, inline-expanded and
;;; let-converted: an inlined helper is the arithmetic it does, AREF is lowered, and
;;; every lvar carries a derived type.  These read a loop into a plan, and print
;;; that plan back as source the source-form pass takes -- with the array types as
;;; (THE ...) casts, since they came from LVAR-TYPE and there may be no declaration
;;; in the enclosing scope at all.
;;;
;;; (:element-type SINGLE-FLOAT :index I :limit N
;;;  :stores ((C :op "+" ((:load A 0 SINGLE-FLOAT) (:load B 0 SINGLE-FLOAT)))))

(defun vector-node-call-name (node)
  ;; callers pass whatever they found in a block, and an INLINE-expanded AREF on a
  ;; non-simple array is not a COMBINATION at all: not recognising it is the right
  ;; answer, signalling is not
  (when (basic-combination-p node)
    (let ((use (lvar-uses (combination-fun node))))
      (when (ref-p use)
        (let* ((leaf (ref-leaf use))
               (name (typecase leaf
                       (global-var (leaf-source-name leaf))
                       (functional (functional-debug-name leaf))
                       (t nil))))
          (when name
            (typecase name
              (string name)
              (symbol (symbol-name name))
              (t (prin1-to-string name)))))))))
(defun vector-node-name-p (node name)
  (let ((n (and (basic-combination-p node) (vector-node-call-name node))))
    (and n (string= n name))))
(defun vector-node-leaf-var (lvar)
  ;; every reader here is total: a shape the pass does not understand must come
  ;; back as NIL, never as an error out of the compiler
  (let ((use (and (lvar-p lvar) (lvar-uses lvar))))
    (when (and (ref-p use) (lambda-var-p (ref-leaf use)))
      (ref-leaf use))))
(defun vector-node-constant (lvar)
  ;; a constant can arrive through a cast, like every other value here
  (let ((use (and (lvar-p lvar) (lvar-uses lvar))))
    (cond ((cast-p use) (vector-node-constant (cast-value use)))
          ((ref-p use)
           (let ((leaf (ref-leaf use)))
             (when (constant-p leaf) (constant-value leaf)))))))
(defun vector-node-blocks (loop)
  (vnloop-blocks loop))
(defun vector-node-nodes (block)
  (let ((acc '()) (ctran (block-start block)))
    (loop while ctran for node = (ctran-next ctran) while node
          do (push node acc) (setf ctran (node-next node)))
    (nreverse acc)))
(defun vector-node-index-p (lvar index-var depth)
  "True when LVAR carries the loop index: I itself, or a parameter of an inline
function called with it - ARET's lowering passes the index through one."
  ;; the index reaches an access through a CAST after inlining (the array's index
  ;; type is not the loop variable's), so look through it
  (let ((use (and (lvar-p lvar) (lvar-uses lvar))))
    (when (cast-p use)
      (return-from vector-node-index-p
        (vector-node-index-p (cast-value use) index-var depth))))
  (when (> depth 12) (return-from vector-node-index-p nil))
  (let ((leaf (vector-node-leaf-var lvar)))
    (cond ((and leaf (eq leaf index-var)) t)
          (leaf
           (let ((home (lambda-var-home leaf)))
             (when home
               ;; a WITH-ARRAY-DATA store gives the home lambda an MV-COMBINATION,
               ;; which COMBINATION-ARGS refuses
               (let ((call (handler-case (let-combination home) (error () nil))))
                 (unless (basic-combination-p call) (return-from vector-node-index-p nil))
                 (when call
                   (let ((pos (position leaf (lambda-vars home))))
                     (and pos
                          (vector-node-index-p (nth pos (combination-args call))
                                      index-var (1+ depth)))))))))
          (t nil))))

(defun vector-node-var-init (var)
  "The value VAR is called with at its home lambda, when that is a constant."
  (let* ((home (lambda-var-home var))
         (call (and home (handler-case (let-combination home) (error () nil)))))
    (unless (basic-combination-p call) (return-from vector-node-var-init nil))
    (let* ((pos (position var (lambda-vars home)))
           (value (and pos (vector-node-constant (nth pos (combination-args call))))))
      (unless value
        )
      value)))

(defun vector-node-loop-invariant-p (var blocks)
  "True when VAR is bound outside the loop being planned, and so survives the splice.

A leaf that is bound INSIDE the loop is not a loop invariant: the splice deletes
the loop's blocks, binding and all, and the replacement would refer to a variable
that no longer exists.  SBCL's AREF and float transforms bind such temporaries
freely -- (SETF (AREF OUT I) (* 2.0 (AREF IN I))) for a (UNSIGNED-BYTE 32) IN is
(+ X X) by the time the loop is analysed, with X bound by the transform's LET
inside the loop -- which printed (+ SB-C::X SB-C::X) into the replacement and
compiled an undefined variable."
  (let* ((home (lambda-var-home var))
         (bind (and home (typep home 'clambda) (lambda-bind home))))
    (or (null bind)
        (not (memq (node-block bind) blocks)))))

(defun vector-node-lane-mask (element-type)
  "The value a LANE of ELEMENT-TYPE keeps, when the store's narrowing is a LOGAND.

The compiler wraps a value the store has to narrow in (LOGAND VALUE MASK) for an
unsigned lane and in a CAST for a signed one, so the mask is implied by the element
type.  It is also the only LOGAND this reader looks through: a MASK the program wrote
itself -- (LOGAND (AREF A I) 3) into a byte array -- is a different operation and is
not the identity, so it has to keep its place in the tree and be refused."
  (and (consp element-type)
       (eq (first element-type) 'unsigned-byte)
       (1- (ash 1 (second element-type)))))

(defun vector-node-tree (lvar depth index-var blocks element-type)
  (unless (lvar-p lvar) (return-from vector-node-tree nil))
  (when (> depth 8) (return-from vector-node-tree nil))
  (let ((use (lvar-uses lvar)))
    (cond ((vector-node-constant lvar) (list :const (vector-node-constant lvar)))
          ((and (ref-p use) (lambda-var-p (ref-leaf use)))
           ;; only a variable the loop does not bind: see VECTOR-NODE-LOOP-INVARIANT-P
           (let ((var (ref-leaf use)))
             (when (vector-node-loop-invariant-p var blocks)
               (list :var (leaf-source-name var)))))
          ;; an integer store narrows its value through a CAST node (the type
          ;; check's knowledge that the sum needs the lane width); the emitter
          ;; re-derives that from the array's element type
          ((cast-p use)
           (vector-node-tree (cast-value use) (1+ depth) index-var blocks element-type))
          ((not (basic-combination-p use))

           nil)
          ((vector-node-name-p use "DATA-VECTOR-REF-WITH-OFFSET")
           (let* ((args (combination-args use))
                  (array (vector-node-leaf-var (first args))))             (and array
                  (eql (vector-node-constant (third args)) 0)
                  (vector-node-index-p (second args) index-var 0)
                  (list :load (leaf-source-name array)
                        (vector-node-constant (third args))
                        (type-specifier (vector-node-lvar-type (node-lvar use)))
                        ;; the leaf itself: a leaf is a form to IR1-CONVERT
                        ;; (ir1tran.lisp:632-636), a name is not
                        array))))
          ;; an integer store narrows its value: (+ ub32 ub32) derives (unsigned-byte
          ;; 33), so the VALUE is wrapped in LOGAND with the lane mask.  The mask is
          ;; implied by the array's element type when the form is printed, so the
          ;; reader looks through it
          ((and (vector-node-name-p use "LOGAND")
                (= (length (combination-args use)) 2)
                (eql (vector-node-constant (second (combination-args use)))
                     (vector-node-lane-mask element-type)))
           (vector-node-tree (first (combination-args use)) (1+ depth) index-var blocks
                             element-type))
          (t (let ((op (vector-node-call-name use)))
               ;; the bitwise operators are elementwise at every width and are
               ;; exact (an n-bit result of an n-bit operation), so they are the
               ;; one integer family the source pass vectorizes at any safety; here
               ;; they still take the (safety 0) gate, and the primitive table
               ;; decides which widths have them
               (when (member op '("+" "-" "*" "/" "%NEGATE"
                                  "LOGAND" "LOGIOR" "LOGXOR")
                             :test #'string=)
                 (let ((trees (mapcar (lambda (a)
                                      (vector-node-tree a (1+ depth) index-var blocks
                                                        element-type))
                                    (combination-args use))))
                   (and (every #'identity trees)
                        ;; the operator is the SYMBOL, which is what VECTOR-OPERATOR-BASE
                        ;; and the emitter's tables are keyed by; printing it interns
                        ;; nothing, and the IR1 path needs no per-call-site interning
                        (list :op (if (string= op "%NEGATE") '- (intern op "CL")) trees)))))))))

;;; the whole loop, if it is one this pass understands
;;; VECTOR-NODE-PLAN answers with a plan plist or with a reason; the reason is what
;;; names the rule that refused a shape when a probe is printed, and it is how every
;;; rule in this pass was found.
(defmacro vector-node-fail (reason)
  `(return-from vector-node-plan ,reason))
(defun vector-node-redundant-call-p (node blocks)
  "True when NODE is a call whose value is unused and whose callee is bound inside
BLOCKS.  TRANSFORM-CALL leaves such a node behind for every source-transformed
AREF: the transform's body is converted in its place and the original combination
stays in the ctran chain with no lvar, which FLUSHABLE-COMBINATION-P does not
answer for (it only answers for :KNOWN calls).  Its body is accounted for node by
node anyway, so the call itself adds nothing and may go with the blocks."
  (and (basic-combination-p node)
       (null (node-lvar node))
       (eq (basic-combination-kind node) :local)
       (let* ((use (lvar-uses (combination-fun node)))
              (fun (and (ref-p use) (ref-leaf use)))
              (bind (and (typep fun 'clambda) (lambda-bind fun))))
         (and bind (memq (node-block bind) blocks)))))

(defun vector-node-mark-consumed (node table)
  "Mark NODE and everything its arguments are computed from in TABLE.

The splice deletes the loop's blocks, so a node the plan did not account for is
deleted with them: a side effect inside the loop would simply vanish, with no
warning and correct values for the arrays.  Walking backwards from the test, the
increment and the stores gives the set of nodes the replacement reproduces."
  (when (and node (not (gethash node table)))
    (setf (gethash node table) t)
    (when (basic-combination-p node)
      (dolist (lvar (combination-args node))
        (let ((use (and (lvar-p lvar) (lvar-uses lvar))))
          (cond ((consp use)
                 (mapc (lambda (u) (vector-node-mark-consumed u table)) use))
                (use (vector-node-mark-consumed use table)))))))
  table)

(defun vector-node-plan (loop)
  ;; a loop with a loop inside it is not elementwise: the inner one is the shape,
  ;; and the outer one is looked at after the round that follows its splice
  (when (vnloop-inferiors loop)
    (vector-node-fail :has-inner-loop))
  (let* ((blocks (vector-node-blocks loop))
         (head (vnloop-head loop)))
    (unless (eq (first blocks) head) (vector-node-fail :head-not-first))
    ;; three blocks are as ordinary as two: after inlining, DOTIMES gives the
    ;; increment a block of its own (test / body / increment), so the shape check
    ;; below looks at every block rather than assuming the body is the second
    (unless (<= (length blocks) 3) (vector-node-fail (list :block-count (length blocks))))
    ;; what the pass cannot carry through the splice: a closure, a return or a
    ;; dispatch inside the loop.  ENTRY and BIND are NOT in this list -- an
    ;; ordinary DOTIMES has both (its BLOCK and TAGBODY) and the sweep retires them
    ;; with the region; refusing them refused the shape the pass exists for
    (dolist (node (mapcan #'vector-node-nodes blocks))
      (when (typep node '(or enclose creturn jump-table))
        (vector-node-fail (list :structural-node (type-of node)))))
    (let* ((cif (find-if (lambda (n) (typep n 'cif)) (vector-node-nodes head)))
           (test (and cif (lvar-uses (if-test cif))))
           (args (and (basic-combination-p test) (combination-args test))))
      (unless cif (vector-node-fail :no-if-in-header))
      (unless (and (basic-combination-p test) (string= (or (vector-node-call-name test) "") "<"))
        (vector-node-fail (list :test-not-lessp (and (basic-combination-p test) (vector-node-call-name test)))))
      (let* ((step-node nil)
             (index-lvar (find-if (lambda (a) (vector-node-leaf-var a)) args))
             (index-var (and index-lvar (vector-node-leaf-var index-lvar)))
             (limit-lvar (and index-lvar (find-if (lambda (a) (not (eq a index-lvar))) args)))
             (body-nodes (mapcan #'vector-node-nodes blocks))
             (increment (find-if (lambda (n) (and (typep n 'cset) (eq (set-var n) index-var)))
                                 body-nodes))
             (stores (remove-if-not (lambda (n) (vector-node-name-p n "DATA-VECTOR-SET-WITH-OFFSET"))
                                    body-nodes)))
        ;; the loop must be exactly: test, index increment, and stores.  Anything
        ;; else -- an accumulator, a conditional inside the body, hidden state -- is
        ;; not elementwise, and reverse-engineering the rest of the loop is what
        ;; splices this into code it cannot represent (the harness's LOOP COLLECT
        ;; compiled with it on was the first casualty)
        (unless (= 1 (count-if (lambda (n) (typep n 'cset))
                               (mapcan #'vector-node-nodes blocks)))
          (vector-node-fail :more-than-the-index-is-set))
        (unless index-var (vector-node-fail :no-index))
        (unless increment (vector-node-fail :no-increment))
        (unless stores (vector-node-fail :no-store))
        ;; a loop that already holds vector primitives is vector code: it came
        ;; from this pass or from the source-form emitter, and rewriting it again
        ;; is how the self-build ran out of heap
        (when (find-if (lambda (node)
                         (and (basic-combination-p node)
                              (let ((name (vector-node-call-name node)))
                                (and name (>= (length name) 4)
                                     (string= name "%VEC" :end1 4 :end2 4)))))
                       (mapcan #'vector-node-nodes blocks))
          (vector-node-fail :already-vector-code))
        ;; A non-simple array is out of scope HERE, and that is a scope decision
        ;; rather than a gap: its AREF inlines to data-vector access behind
        ;; WITH-ARRAY-DATA, which the source-form pass already handles (it hoists the
        ;; data vector and the start index itself, see VECTOR-WITH-DATA-VECTORS).
        ;; This pass exists for loops the source pass CANNOT see -- arithmetic that
        ;; only exists after inlining -- so a shape the source pass covers is left to
        ;; it rather than duplicated here.  Same for a ramp built from the index.
        (unless (every (lambda (store)
                         (let ((ty (vector-node-lvar-type (first (combination-args store)))))
                           (and (array-type-p ty)
                                (not (array-type-complexp ty)))))
                       stores)
          (vector-node-fail :not-a-simple-array))
        (let ((step (lvar-uses (set-value increment))))
          (setf step-node step)
          ;; the increment's value can arrive through a cast, as every other value
          ;; in this pass can; the cast chain is walked before the shape is read
          (loop repeat 4
                while (cast-p step)
                do (setf step (lvar-uses (cast-value step))))
          (setf step-node step)
          (let ((step-args (and (basic-combination-p step) (combination-args step)))
                (name (and (basic-combination-p step) (vector-node-call-name step))))
            ;; (1+ i) is as common as (+ i 1), and DOTIMES expands to either
            (unless (and step-args (member name '("+" "1+") :test #'string=))
              (vector-node-fail (list :increment-not-plus name)))
            (unless (and (eq (vector-node-leaf-var (first step-args)) index-var)
                         (or (and (string= (or name "") "1+") (= (length step-args) 1))
                             (eql (vector-node-constant (second step-args)) 1)))
              (vector-node-fail :increment-not-by-one))))
        (unless (every (lambda (ref) (memq (ctran-block (node-prev ref)) blocks))
                       (leaf-refs index-var))
          (vector-node-fail :index-used-outside))
        (let ((entries '()))
          (dolist (store stores)
            (let* ((sargs (combination-args store))
                   (array (vector-node-leaf-var (first sargs)))
                   ;; the element type comes from the ARRAY, not from the stored
                   ;; value: the value's lvar can carry the type of a cast or a
                   ;; constant (this printed (SIMPLE-ARRAY (INTEGER 0 0) (*)) and
                   ;; failed at run time).  It is read before the tree because the
                   ;; tree reader needs it: the only LOGAND it may look through is
                   ;; the narrowing to this lane's own width.
                   (element-type
                     (let ((ty (vector-node-lvar-type (first sargs))))
                       (and (array-type-p ty)
                            (not (array-type-complexp ty))
                            (type-specifier (array-type-element-type ty)))))
                   (tree (vector-node-tree (fourth sargs) 0 index-var blocks element-type)))
              (unless array (vector-node-fail :store-array-not-a-variable))
              (unless (vector-node-index-p (second sargs) index-var 0)
                (vector-node-fail :store-index-not-the-induction-variable))
              (unless (eql (vector-node-constant (third sargs)) 0)
                (vector-node-fail (list :store-offset (vector-node-constant (third sargs)))))
              (unless tree (vector-node-fail :no-value-tree))
              ;; -x on an unsigned lane has to wrap, and the emitter's negate does
              ;; not: the scalar reference and the packed loop disagree (the harness
              ;; catches it as a wrong value, which is how this was found)
              ;; -x on an unsigned lane has to wrap; the emitter's negate does not,
              ;; so the packed loop and the scalar reference disagree on the value
              ;; (the harness found this the moment casts were looked through)
              (when (and (consp tree) (eq (first tree) :op) (string= (second tree) "-")
                         (= (length (third tree)) 1)
                         (consp element-type)
                         (eq (first element-type) 'unsigned-byte))
                (vector-node-fail :unary-minus-on-unsigned))
              (unless element-type (vector-node-fail :no-element-type))
              (push (list (leaf-source-name array)
                          element-type
                          tree
                          ;; the array's own leaf: the IR1 path indexes the variable
                          ;; the recogniser read, not a name
                          array)
                    entries)))
          ;; everything else in the loop's blocks has to be accounted for, or the
          ;; splice deletes it
          (let ((consumed (make-hash-table :test 'eq)))
            (dolist (store stores)
              (vector-node-mark-consumed store consumed))
            (mapc (lambda (n) (vector-node-mark-consumed n consumed))
                  (list test step-node increment
                        ;; the limit can be computed inside the loop (a hoisted
                        ;; VECTOR-LENGTH, say) and the emitted DOTIMES names it
                        (and (lvar-p limit-lvar) (lvar-uses limit-lvar))))
            ;; only nodes that do work matter: a combination or a set.  The
            ;; structural nodes (block starts, exits) go with the blocks
            ;; a node goes with the blocks unless deleting it would be observable:
            ;; it has an effect, or its value is used outside the loop.  Asking
            ;; "would deleting this change behaviour" is the question; asking
            ;; "is it in the emitted form" enumerates the compiler's pure
            ;; operations one refusal at a time
            (dolist (node (mapcan #'vector-node-nodes blocks))
              (unless (or (gethash node consumed)
                          (not (basic-combination-p node))
                          (vector-node-redundant-call-p node blocks)
                          ;; the value is consumed inside the loop: the stores
                          ;; reproduce it
                          (let ((res (node-lvar node)))
                            (and res
                                 (let ((dest (lvar-dest res)))
                                   (and dest (memq (node-block dest) blocks)))))
                          ;; a transform or an inline expansion, which is compiler
                          ;; machinery rather than a call that can have an effect
                          (let ((use (lvar-uses (combination-fun node))))
                            (and (ref-p use) (functional-p (ref-leaf use))))
                          ;; the bounds check an AREF brings: the loop the emitter
                          ;; produces keeps it (the scalar tail still signals), so
                          ;; dropping this one does not lose the check
                          (let ((name (and (basic-combination-p node)
                                           (vector-node-call-name node))))
                            (and name (search "%CHECK-BOUND" name)))
                          (flushable-combination-p node))
                (vector-node-fail (list :endangers (type-of node))))))
          (unless (eql (vector-node-var-init index-var) 0)
            (vector-node-fail (list :index-does-not-start-at-zero (vector-node-var-init index-var))))
          ;; the element type has to be one this backend can load and store a
          ;; pack of, and every store has to agree: the emitter reads ONE element
          ;; type for the whole loop (VLOOP holds one), and the splice takes it
          ;; from the first store.  Without this the pass plans a loop over
          ;; SIMPLE-VECTOR -- whose element type is T -- prints a DOTIMES the
          ;; source pass then declines to rewrite, and has replaced a loop with
          ;; itself: two such loops in MAKE-WEAK-VECTOR were the first thing a
          ;; switch-on self-build spliced.
          (let ((*vector-lane-bytes* 16)
                (element-type (second (first entries))))
            (unless (every (lambda (entry) (equal (second entry) element-type)) entries)
              (vector-node-fail (list :mixed-element-types
                                      (mapcar #'second entries))))
            (unless (and (vector-primitive "LOAD" element-type)
                         (vector-primitive "STORE" element-type))
              (vector-node-fail (list :no-lane-for element-type))))
          (list :element-type (second (first entries))
                ;; An integer lane wraps where a checked loop would signal: the
                ;; packed store masks, the scalar store it replaces signals.  The
                ;; source pass states the same rule twice
                ;; (VECTOR-EXACT-EXPRESSION-P for a tree that cannot leave the
                ;; lane, VLOOP-UNCHECKED-P for the rest); this pass takes only the
                ;; strict half -- integer lanes at (safety 0) -- and carries it on
                ;; the plan, where VECTOR-NODE-IR1-FAMILY-P reads it.
                :unchecked-p (policy (block-last head) (= safety 0))
                :index (leaf-source-name index-var)
                ;; the leaves the IR1 path emits: the index variable, the limit (a
                ;; leaf, or the constant it is), and every array the plan touches
                :index-var index-var
                :limit-leaf (or (vector-node-leaf-var limit-lvar)
                                (vector-node-constant limit-lvar))
                :array-vars (let ((acc '()))
                              (flet ((add (name leaf)
                                       (when leaf
                                         (pushnew (cons name leaf) acc
                                                  :test (lambda (a b) (equal (car a) (car b)))))))
                                (dolist (entry entries)
                                  (add (first entry) (fourth entry)))
                                (labels ((walk (tree)
                                           (case (car tree)
                                             (:load (add (second tree) (fifth tree)))
                                             (:op (mapc #'walk (third tree)))
                                             (t nil))))
                                  (dolist (entry entries) (walk (third entry))))
                                acc))
                ;; the limit has to be a name or a literal: an expression would
                ;; be re-evaluated by the emitted DOTIMES, and it was printing as
                ;; NIL, i.e. (THE INTEGER NIL)
                :limit (or (vector-node-constant limit-lvar)
                           (let ((leaf (vector-node-leaf-var limit-lvar)))
                             (and leaf (leaf-source-name leaf)))
                           (vector-node-fail :limit-not-expressible))
                :stores (mapcar (lambda (e) (cons (first e) (third e))) (nreverse entries))))))))


;;; ---- from the plan back to source -----------------------------------------
(defun vector-node-print-tree (tree subst index)
  (case (car tree)
    (:const (second tree))
    (:var (second tree))
    (:load `(aref ,(or (cdr (assoc (second tree) subst :test #'equal))
                       (second tree))
                  ,index))
    (:op `(,(second tree)
           ,@(mapcar (lambda (sub) (vector-node-print-tree sub subst index))
                     (third tree))))
    (t (error "unexpected tree ~s" tree))))

;;; Every array the plan touches, as a THE cast that names its element type: a
;;; load's type is the one the recogniser read from its lvar, and a store's is the
;;; loop's, which is the same by construction.  Carrying the casts rather than
;;; relying on declarations is what lets the printed form work in a scope that
;;; declares none of them.
(defun vector-node-array-casts (plan)
  (let ((accum '()))
    (labels ((add (name type)
               (unless (assoc name accum :test #'equal)
                 (push (cons name `(the (simple-array ,type (*)) ,name)) accum)))
             (walk (tree)
               (case (car tree)
                 (:load (add (second tree) (fourth tree)))
                 (:op (mapc #'walk (third tree)))
                 (t nil))))
      (dolist (store (getf plan :stores))
        (add (car store) (getf plan :element-type))
        (walk (cdr store)))
      (nreverse accum))))

(defun vector-node-guard-form (plan)
  "What the packed loop needs of the arrays and the count, as a test.

  It is the guard the source-form emitter builds (VECTOR-ELEMENTWISE-PLAN), with one
  simplification: this pass only plans a store at offset 0, so the length has to
  cover COUNT and no more.  There is no run-time alias test to make -- the emitter's
  alias rule (VECTOR-STORES-SAFE-P) is decided when the loop is planned, from the
  offsets, not from the arrays."
  (let ((count (getf plan :limit)))
    `(and (typep ,count 'index)
          ,@(mapcar (lambda (cast) `(>= (length ,(cdr cast)) ,count))
                    (vector-node-array-casts plan)))))

(defun vector-node-packed-form (plan)
  "The packed loop over the loop's aligned prefix, as the source-form pass's shape.

  The DOTIMES is still printed source in this shape of the pass, so that the
  source-form pass can turn it into packs: that is the layering this work removes,
  rung by rung.  What is no longer printed is the loop itself.  The index it hands
  back is the loop's OWN index variable, set to the first element the packed loop
  does not do, so the loop the pass found does the rest -- exactly the emitter's
  packed-loop-then-scalar-tail, with the original loop as the tail.

  The arrays carry their element types as THE casts: the recogniser read them from
  LVAR-TYPE and the loop's scope may declare none of them."
  (let* ((stores (getf plan :stores))          ; (array-name . value-tree) pairs
         (type (getf plan :element-type))
         (index (getf plan :index))
         (packed (make-symbol "VN-I"))
         (count (getf plan :limit))
         (subst (vector-node-array-casts plan)))
    `(progn
       ;; hand the loop the first element it is to do, before the packed loop runs:
       ;; the packed loop can only do the aligned prefix
       (setq ,index (- ,count (mod ,count ,(vector-width type))))
       (dotimes (,packed ,index)
         ,@(mapcar (lambda (store)
                     `(setf (aref ,(cdr (assoc (car store) subst :test #'equal)) ,packed)
                            ,(vector-node-print-tree (cdr store) subst packed)))
                   stores)))))


;;; Put the packed loop in front of the loop, and leave the loop there.
;;;
;;; Nothing is deleted and nothing is moved: a guard block is inserted on the edge
;;; from the loop's preheader, and the packed-prefix code hangs off its true edge.
;;; The loop the pass found is what runs the elements the packed loop does not -- it
;;; is the scalar tail, entered with its own index variable already advanced to the
;;; first element the packed loop left.  That is the shape the source-form emitter
;;; builds (guard, packed loop, scalar tail) with no printed source and no block
;;; surgery beyond the one insertion.
;;;
;;; The technique is INSERT-CLEANUP-CODE's (ir1util.lisp:36-55), which is SBCL's one
;;; deliberate example of building blocks and nodes in a live component, and the
;;; invariants are the same ones, for the same reasons:
;;;
;;; - the new block is made with MAKE-CTRAN + CTRAN-STARTS-BLOCK, and it is spliced
;;;   into the flow with CHANGE-BLOCK-SUCCESSOR, which also repoints the IF that
;;;   jumps to the loop when there is one;
;;; - the form is converted into a ctran the new block owns, and delivers into the
;;;   loop's own start ctran, so the loop's BLOCK-LAST and its ENTRY nodes are left
;;;   exactly as the pipeline left them;
;;; - WITH-IR1-ENVIRONMENT-FROM-NODE supplies the component, the lexenv and the
;;;   source path the conversion needs, so the new code is converted in the loop's
;;;   own scope and inherits its bindings and cleanups;
;;; - COMPONENT-REANALYZE is set, because the block chain, the DFO and the block
;;;   numbers are all stale afterwards.
;;;
;;; The pass itself never calls FIND-DFO or DELETE-BLOCK: the next round of
;;; IR1-PHASES, after local call analysis has run, is what refreshes the DFO and
;;; retires anything the insertion made unreachable.
;;; ---- the packed prefix as IR1 ---------------------------------------------
;;; What is emitted here is not printed source.  Every operand is a leaf the
;;; recogniser already holds -- the array variables the DATA-VECTOR-REF/SET nodes
;;; address, and the loop's own index variable -- and IR1-CONVERT turns a leaf into a
;;; REF (ir1tran.lisp:632-636), which is what an inline expansion relies on.  So
;;; IR1-CONVERT is still called, but on a form of VOP-call forms over leaves: no
;;; symbol of the user's program is looked up, and the name-based bug that printing
;;; produced (a transform's temporary taken for a loop invariant) cannot happen.
;;;
;;; The loop's own index variable is the packed counter.  It is already bound and
;;; already zero on entry, its step is the one CSET built here, and when the packed
;;; loop stops it holds the first index the packed loop did not do -- which is where
;;; the scalar tail, the loop the pass found, has to start.  No new binding.
;;;
;;; The ctran discipline, which is where the first attempt went wrong, is the one
;;; INSERT-CLEANUP-CODE uses (ir1util.lisp:36-55): a conversion is given a ctran to
;;; start at and a fresh one to continue at, and the blocks are linked by the
;;; conversions themselves.  Three blocks, each created and then filled by the code
;;; that follows it, never an empty landing pad:
;;;
;;;   ENTRY   the test, from the preheader; the back edge comes here too
;;;   JOIN    the CIF: consequent BODY, alternative the loop's own head
;;;   BODY    the VOP stores, then the CSET back to ENTRY
;;;
;;; The test is asked as (IF TEST T NIL) rather than converted bare, because MIN and
;;; multi-argument <= expand to conditionals: the IF translator then guarantees a
;;; join block, linked from both arms, for the CIF to live in.  TEST re-reads the
;;; count and the array lengths every iteration, which is what lets it be a form over
;;; leaves and nothing else; hoisting them into bound variables is a later rung.

(defparameter +vector-node-ir1-operators+ '(+ - * logand logior logxor)
  "The operators the IR1 path emits, as a plan tree carries them: the symbol.

Division is not here: it takes float lanes only, and no width has a packed integer
divide.  Which of these a width can do is the primitive table's answer
(VECTOR-PRIMITIVE), never this list's: no byte multiply at some widths, no 64-bit
multiply at any, and the 32-bit multiply and minimum only where the build has the
instructions.")

(defun vector-node-float-lane-p (element-type)
  (member element-type '(single-float double-float) :test #'equal))

(defun vector-node-ir1-family-p (element-type operator arity unchecked-p)
  "True when this element type, operator and arity are a family the IR1 path emits.

Rung C widens this one family at a time, and what it does not cover keeps the printed
path: a missing primitive is a decline here, never a VOP that is not there.  Arity 1
is a sign flip (the recogniser turns %NEGATE into a one-operand -), which is a XOR on
the sign bit and not a subtraction of zero, and only the float lanes have it.

UNCHECKED-P is the plan's (safety 0), and an integer lane is only reached there: the
packed store masks to the lane width where the scalar store it replaces would signal a
type error, so the two loops agree at (safety 0) and not above it.  The element type
is compared with EQUAL because an integer element type is a LIST -- (UNSIGNED-BYTE 32)
-- and two equal lists are never EQ."
  (and (or (vector-node-float-lane-p element-type) unchecked-p)
       (case arity
         (1 (and (vector-node-float-lane-p element-type) (eq operator '-)))
         (2 (member operator +vector-node-ir1-operators+ :test #'eq)))))

(defun vector-node-lane-p (element-type)
  "True when this backend publishes a pack of ELEMENT-TYPE it can move."
  (and (vector-primitive "LOAD" element-type)
       (vector-primitive "STORE" element-type)))

(defun vector-node-tree-primitives-p (tree element-type)
  "True when every primitive TREE needs exists at the width this build published.

This half of the gate must not be left to the emitter.  A store whose operator has no
primitive at that width -- a 64-bit multiply, a 32-bit one on a build without SSE4.1 --
has to be refused before the first block exists, because refusing after that leaves an
unreachable block with no nodes, and FIND-DFO cannot delete one: it reaches
NOTE-BLOCK-DELETION, which asks the block for its BLOCK-LAST.  Measured, by letting
(UNSIGNED-BYTE 64) multiply through the gate and having VECTOR-NODE-STORE-FORM answer
NIL instead: the compiler died with 'The value NIL is not of type SB-C::NODE'."
  (and (vector-node-lane-p element-type)
       (let* ((operator (second tree))
              (arity (length (third tree)))
              ;; a one-operand - is a sign flip, not a subtraction of zero
              (base (if (= arity 1) "NEG" (vector-operator-base operator element-type))))
         (and base (vector-primitive base element-type)))))

(defun vector-node-packed-ir1-p (plan)
  "The shapes this emits as IR1: an elementwise store of (OP (LOAD A I) (LOAD B I)).

Everything else keeps the printed path, so a mistake here is one family wide.  Every
refusal lives HERE, and that is a structural rule rather than tidiness: once the
emission has made a block it cannot decline, and the gate is the last moment at which
the loop can still be left to the printed path."
  (let ((type (getf plan :element-type)))
    (and (lambda-var-p (getf plan :index-var))
         (getf plan :limit-leaf)
         ;; the room test names each array once, so no two names may be one variable
         (= (length (getf plan :array-vars))
            (length (remove-duplicates (mapcar #'cdr (getf plan :array-vars))
                                       :test #'eq)))
         (vector-node-lane-p type)
         (every (lambda (store)
                  (let ((tree (cdr store)))
                    (and (consp tree)
                         (eq (car tree) :op)
                         (member (length (third tree)) '(1 2))
                         (vector-node-ir1-family-p type (second tree)
                                                   (length (third tree))
                                                   (getf plan :unchecked-p))
                         (every (lambda (sub)
                                  (and (eq (car sub) :load)
                                       (equal (fourth sub) type)
                                       (fifth sub)))
                                (third tree))
                         (vector-node-tree-primitives-p tree type))))
                (getf plan :stores)))))

(defun vector-node-store-form (plan entry index)
  "(%VEC-STORE-SINGLE (%VEC-ADD-SINGLE ...) ARRAY INDEX), with leaves for operands.

Total for the shapes VECTOR-NODE-PACKED-IR1-P accepts -- the gate has already asked
for every primitive this reads -- which is what lets the emission never decline."
  (let* ((type (getf plan :element-type))
         (node (cdr entry))
         (array (cdr (assoc (car entry) (getf plan :array-vars) :test #'equal))))
    (aver array)
    (flet ((operand (sub)
             (let ((leaf (cdr (assoc (second sub) (getf plan :array-vars)
                                     :test #'equal))))
               (aver leaf)
               `(,(vector-primitive "LOAD" type) ,leaf ,index))))
      (let* ((operands (mapcar #'operand (third node)))
             ;; one operand is a sign flip, which has its own primitive and is not a
             ;; subtraction of zero (VECTOR-OPERATOR-BASE :NEGATE)
             (base (if (= (length operands) 1)
                       "NEG"
                       (vector-operator-base (second node) type))))
        (aver (and base (vector-primitive base type)))
        `(,(vector-primitive "STORE" type)
          (,(vector-primitive base type) ,@operands)
          ,array ,index)))))

(defun vector-node-room-test (plan index width)
  "The packed loop's test: the index against the limit, and the room in the arrays.

One (IF ... T NIL) over leaves -- the loop's own index variable, the limit, and the
arrays -- so that the blocks the conversion makes are the same for every element type
and every number of stores: the arms are constants, so each of them ends in a REF, and
the only other blocks are the ones a CIF lives in.

The room in the arrays is asked in the test rather than folded into the bound with
(MIN LIMIT (LENGTH A) (LENGTH B) ...), because that shape puts MIN's own value
selection in the test: whether it expands to a conditional depends on the element type
(the lengths and the limit are (INTEGER 0 N) for an integer array) and it costs a MIN
per iteration either way.  This way (<= (+ INDEX WIDTH) LIMIT) is what stops the packed
loop, so the index it hands the scalar tail is the first element the tail has to do,
and every array is at least LIMIT long, so neither a load nor a store can leave one.

A limit that is not an index -- negative, or wide enough to be a bignum -- fails the
first length check and sends the whole loop to the tail, which is the original loop
entered at index 0."
  (let ((limit (getf plan :limit-leaf))
        (arrays (remove-duplicates (mapcar #'cdr (getf plan :array-vars)) :test #'eq)))
    `(and (<= (+ ,index ,width) ,limit)
          ,@(mapcar (lambda (array) `(>= (length ,array) ,limit)) arrays))))

(defun vector-node-emit-packed-ir1 (plan preheader head)
  "Build the packed prefix as blocks, nodes and one CSET, ahead of the loop.

The three blocks are created and filled in the order they are written: BODY by the
stores and the CSET, ENTRY by the test, JOIN by the CIF.  Nothing is created as an
empty landing pad, which is what the first attempt got wrong, and nothing here can
decline -- VECTOR-NODE-PACKED-IR1-P has already answered, and a decline after a block
exists in the component's chain would leave one with no nodes that FIND-DFO cannot
delete.  THE VALVE at the end names that condition instead of letting it through."
  (let* ((type (getf plan :element-type))
         (index (getf plan :index-var))
         (limit (getf plan :limit-leaf))
         (width (vector-width type))
         (stores (getf plan :stores))
         (component *current-component*)
         ;; every block CTRAN-STARTS-BLOCK makes is spliced in front of this one
         (tail (component-last-block component))
         (anchor (block-prev tail)))
    (aver (or (lambda-var-p limit) (integerp limit)))
    (aver (consp stores))
    (let* ((start (make-ctran))
           (entry (ctran-starts-block start))       ; filled by the test, below
           (body-start (make-ctran))
           (body (ctran-starts-block body-start)))  ; filled by the stores, below
      ;; BODY: the stores, each converted after the one before it, then the CSET
      (let ((cursor body-start))
        (dolist (store stores)
          (let ((next (make-ctran)))
            (ir1-convert cursor next nil (vector-node-store-form plan store index))
            (setf cursor next)))
        ;; the increment: INDEX := INDEX + WIDTH, then back to the test.  SETQ-VAR
        ;; (ir1-translators.lisp:1450-1486) is the model for the CSET.
        (let* ((next (make-ctran))
               (value-lvar (make-lvar))
               (set (make-set index value-lvar)))
          (ir1-convert cursor next value-lvar `(+ ,index ,width))
          (setf (lvar-dest value-lvar) set)
          (push set (basic-var-sets index))
          (link-node-to-previous-ctran set next)
          (use-continuation set (block-start entry) nil)))
      ;; ENTRY: the test.  JOIN is the block the IF translator's join lands in, and
      ;; it is where the CIF goes.
      (let* ((test-ctran (make-ctran))
             (test-lvar (make-lvar))
             (test (make-if test-lvar body head)))
        (setf (lvar-dest test-lvar) test)
        (ir1-convert start test-ctran test-lvar
                     `(if ,(vector-node-room-test plan index width) t nil))
        (let ((join (ctran-starts-block test-ctran)))
          (link-node-to-previous-ctran test test-ctran)
          (setf (block-last join) test)
          (link-blocks join body)
          (link-blocks join head)
          ;; THE VALVE.  Every block this made has to end in a node: the next DFO
          ;; sweep deletes the unreachable ones, and deleting one asks
          ;; NOTE-BLOCK-DELETION for its BLOCK-LAST, which is NIL here if the
          ;; conversion left a block it never filled.  An assertion and not a
          ;; decline, because a decline is impossible at this point: the blocks are
          ;; already in the component's chain, and an empty one is precisely what
          ;; the sweep cannot delete.  The gate is what keeps it from firing.
          (do ((block (block-prev tail) (block-prev block)))
              ((eq block anchor))
            (aver (block-last block)))))
      ;; the loop is entered through ENTRY now, not straight from the preheader
      (change-block-successor preheader head entry)
      t)))

(defun vector-node-splice-loop (loop plan)
  (let* ((blocks (vector-node-blocks loop))
         (head (vnloop-head loop))
         ;; the loop is entered from one place: anything else would need the guard
         ;; on every edge, and the guard has side effects on the index variable
         (outside-in (remove-if (lambda (block) (memq block blocks)) (block-pred head))))
    (unless (and (= (length outside-in) 1)
                 (block-start head))
      (return-from vector-node-splice-loop nil))
    (let* ((preheader (first outside-in))
           (component (block-component head))
           (head-home (block-home-lambda-or-null head)))
      ;; Both edges stay inside the loop's home lambda: a cross-lambda edge is an
      ;; NLX target that environment-analyze has not established yet
      ;; (envanal.lisp:200-225 runs later).  BLOCK-HOME-LAMBDA on an empty block
      ;; would read (block-last block) = nil (ir1util.lisp:1437-1439), hence
      ;; BLOCK-HOME-LAMBDA-OR-NULL (ir1util.lisp:1405-1433).
      (unless (and head-home
                   (or (null (block-last preheader))   ; the component head
                       (eq (block-home-lambda-or-null preheader) head-home)))
        (return-from vector-node-splice-loop nil))
      ;; The anchor is the loop's own ENTRY node: the mess-up of the cleanup in the
      ;; lexenv of the head's test.  The block/tagbody translators convert that
      ;; entry BEFORE adding the cleanup (ir1-translators.lisp:199-208 and
      ;; 294-314), so its lexenv is the scope the loop runs in -- every binding and
      ;; every enclosing cleanup still in effect.  Anchoring on a preheader node
      ;; would drop the loop-construct cleanups.
      (let ((anchor (or (let ((cleanup (and (block-last head)
                                            (lexenv-cleanup (node-lexenv (block-last head))))))
                          (and cleanup (cleanup-mess-up cleanup)))
                        (block-last preheader)
                        (lambda-bind head-home))))
        (with-ir1-environment-from-node anchor
          (let ((done
                  (if (vector-node-packed-ir1-p plan)
                      ;; blocks, nodes and one CSET, with the operands as leaves: no
                      ;; printed source, and the source-form pass is not involved
                      (vector-node-emit-packed-ir1 plan preheader head)
                      ;; the packed prefix as the source-form pass's own shape,
                      ;; converted into a guard block; that pass has to be on while
                      ;; it is converted or the DOTIMES stays scalar
                      (let* ((start (make-ctran))
                             (guard (ctran-starts-block start)))
                        (let ((*auto-vectorize* t))
                          (ir1-convert start (block-start head) nil
                                       `(when ,(vector-node-guard-form plan)
                                          ,(vector-node-packed-form plan))))
                        (change-block-successor preheader head guard)
                        t))))
            (when done
              ;; the block chain, the DFO and the block numbers are stale
              (setf (component-reanalyze component) t)
              ;; either path leaves functionals waiting for local call analysis (a
              ;; converted DOTIMES, or the LETs a VOP call does not have); the next
              ;; round runs it (main.lisp:503-508) and then optimizes what was added
              (locall-analyze-component component))
            done))))))
