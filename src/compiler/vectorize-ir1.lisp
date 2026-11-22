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

(defun vector-node-tree (lvar depth index-var blocks)
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
          ((cast-p use) (vector-node-tree (cast-value use) (1+ depth) index-var blocks))
          ((not (basic-combination-p use))

           nil)
          ((vector-node-name-p use "DATA-VECTOR-REF-WITH-OFFSET")
           (let* ((args (combination-args use))
                  (array (vector-node-leaf-var (first args))))             (and array
                  (eql (vector-node-constant (third args)) 0)
                  (vector-node-index-p (second args) index-var 0)
                  (list :load (leaf-source-name array)
                        (vector-node-constant (third args))
                        (type-specifier (vector-node-lvar-type (node-lvar use)))))))
          ;; an integer store narrows its value: (+ ub32 ub32) derives (unsigned-byte
          ;; 33), so the VALUE is wrapped in LOGAND with the lane mask.  The mask is
          ;; implied by the array's element type when the form is printed, so the
          ;; reader looks through it
          ((and (vector-node-name-p use "LOGAND")
                (= (length (combination-args use)) 2)
                (vector-node-constant (second (combination-args use))))
           (vector-node-tree (first (combination-args use)) (1+ depth) index-var blocks))
          (t (let ((op (vector-node-call-name use)))
               (when (member op '("+" "-" "*" "/" "%NEGATE") :test #'string=)
                 (let ((trees (mapcar (lambda (a) (vector-node-tree a (1+ depth) index-var blocks))
                                      (combination-args use))))
                   (and (every #'identity trees)
                        (list :op (if (string= op "%NEGATE") "-" op) trees)))))))))

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
                   (tree (vector-node-tree (fourth sargs) 0 index-var blocks)))
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
                         (let ((et (type-specifier (vector-node-lvar-type (fourth sargs)))))
                           (and (consp et) (string= (symbol-name (first et))
                                                    "UNSIGNED-BYTE"))))
                (vector-node-fail :unary-minus-on-unsigned))
              ;; the element type comes from the ARRAY, not from the stored value:
              ;; the value's lvar can carry the type of a cast or a constant (this
              ;; printed (SIMPLE-ARRAY (INTEGER 0 0) (*)) and failed at run time)
              (push (list (leaf-source-name array)
                          (type-specifier
                           (array-type-element-type (vector-node-lvar-type (first sargs))))
                          tree)
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
                :index (leaf-source-name index-var)
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
    (:op `(,(intern (second tree) "CL")
           ,@(mapcar (lambda (sub) (vector-node-print-tree sub subst index))
                     (third tree))))
    (t (error "unexpected tree ~s" tree))))

(defun vector-node-source-form (plan)
  "The loop the plan describes, as the DOTIMES the source-form pass rewrites.

  It is the loop and nothing else.  The arrays carry their element types as THE
  casts, because the recogniser read them from LVAR-TYPE and the scope the loop
  sits in may declare none of them; the index and the count are bound by the
  DOTIMES, so nothing in the form refers to a binding of the loop it replaces.

  An earlier version wrapped this in a fresh local function and called it, to
  hand IR1 a top-down conversion.  A call to a local function is exactly what
  IR1-CONVERT-LOCAL-COMBINATION refuses once the callee has been LET converted
  (ir1tran.lisp:1255-1262), and that is what aborted a switch-on self-build in
  src/code/weak.lisp.  The wrapper is not needed: the loop is converted into a
  block of its own (VECTOR-NODE-SPLICE-LOOP), which is the shape IR1 expects,
  and the note in doc/internals-notes records what the wrapper was for."
  (let* ((stores (getf plan :stores))          ; (array-name . value-tree) pairs
         (type (getf plan :element-type))
         (array-type `(simple-array ,type (*)))
         (index (make-symbol "VN-I"))
         ;; every reference to an array says what it holds: a load's value type and
         ;; the store's element type both come from here, not from the scope
         (subst (mapcar (lambda (store)
                          (cons (car store) `(the ,array-type ,(car store))))
                        stores)))
    `(dotimes (,index ,(getf plan :limit))
       ,@(mapcar (lambda (store)
                   `(setf (aref ,(cdr (assoc (car store) subst :test #'equal)) ,index)
                          ,(vector-node-print-tree (cdr store) subst index)))
                 stores))))


;;; Put the replacement where the loop was: convert it into a fresh block between
;;; the loop's preheader and its exit, relink those two edges, and delete the two
;;; blocks the loop was made of.  WRAP-PREDICATE (ir1tran.lisp:1192) is the model
;;; for building a block here; there is no call to repoint, so this is the same
;;; surgery WITHOUT the lambda that the inline-expansion path uses.
(defun vector-node-splice-loop (loop plan)
  (let* ((blocks (vector-node-blocks loop))
         (head (vnloop-head loop))
         (outside-in (remove-if (lambda (block) (memq block blocks)) (block-pred head)))
         (outside-out (remove-if (lambda (block) (memq block blocks)) (block-succ head))))
    ;; one way in and one way out: anything else needs more care than this
    (unless (and (= (length outside-in) 1) (= (length outside-out) 1))
      (return-from vector-node-splice-loop nil))
    (let* ((preheader (first outside-in))
           (exit (first outside-out))
           (component (block-component head))
           (head-home (block-home-lambda-or-null head))
           (exit-ctran (and exit (block-start exit)))
           (exit-home (and exit-ctran (block-home-lambda-or-null exit))))
      ;; EXIT must be a real block we can deliver into, and both edges of the
      ;; loop must stay inside the loop's home lambda: a cross-lambda edge is
      ;; an NLX target that environment-analyze has not established yet
      ;; (envanal.lisp:200-225 runs later).  BLOCK-HOME-LAMBDA on an empty
      ;; block would read (block-last block) = nil (ir1util.lisp:1437-1439),
      ;; hence BLOCK-HOME-LAMBDA-OR-NULL (ir1util.lisp:1405-1433).
      ;; the replacement is converted INTO EXIT's start ctran, so if that ctran has
      ;; been deleted (a merge upstream of the loop retired it) RETURN-FROM would
      ;; throw LOCALL-ALREADY-LET-CONVERTED (ir1-translators.lisp:231).  Decline.
      (when (ctran-deleted-p exit-ctran)
        (return-from vector-node-splice-loop nil))
      (unless (and head-home exit-ctran
                   (eq exit-home head-home)
                   (or (null (block-last preheader))   ; the component head
                       (eq (block-home-lambda-or-null preheader) head-home)))
        (return-from vector-node-splice-loop nil))
      ;; The anchor is the loop's own ENTRY node: the mess-up of the cleanup in
      ;; the lexenv of the head's test.  The block/tagbody translators convert
      ;; that entry BEFORE adding the cleanup (ir1-translators.lisp:199-208 and
      ;; 294-314), so its lexenv is the scope the loop runs in -- every binding
      ;; and every enclosing cleanup still in effect, minus the loop's own
      ;; cleanups, which the splice deletes.  Anchoring on a preheader node
      ;; instead would drop the loop-construct cleanups; anchoring inside the
      ;; loop would inherit cleanups of code that is about to be deleted
      ;; (merge-lets stores this lexenv as the call's LAMBDA-CALL-LEXENV,
      ;; locall.lisp:1081, and node-enclosing-cleanup walks it, ir1util.lisp:23).
      ;;
      ;; The lexenv is the anchor's, unchanged: it is the scope the loop runs in,
      ;; and the ENTRY nodes the emitted DOTIMES creates must be registered
      ;; against the same lambda its blocks resolve to.  Both come from this one
      ;; lexenv -- registration from (LEXENV-LAMBDA *LEXENV*)
      ;; (ir1-translators.lisp:201, 296, 298), home from (NODE-HOME-LAMBDA ...)
      ;; (ir1util.lisp:1377-1389) -- so a merge that retires that lambda fixes
      ;; both at once, which is what MERGE-LETS does (locall.lisp:1130-1140).
      (let ((anchor (or (let ((cleanup (and (block-last head)
                                            (lexenv-cleanup (node-lexenv (block-last head))))))
                          (and cleanup (cleanup-mess-up cleanup)))
                        (block-last preheader)
                        (lambda-bind head-home))))
        (with-ir1-environment-from-node anchor
          ;; Build the replacement's block by hand and convert the form into it.
          ;; WRAP-PREDICATE (ir1tran.lisp:1200-1221) is the model: MAKE-CTRAN,
          ;; CTRAN-STARTS-BLOCK, IR1-CONVERT, then LINK-BLOCKS.  Nothing is
          ;; converted into a live block: the form starts at a ctran this block
          ;; owns and continues into EXIT's start ctran, which is where any
          ;; conversion continuing into an existing block delivers.  That leaves
          ;; EXIT's own start ctran and BLOCK-LAST untouched, which is the whole
          ;; reason the replacement gets a block of its own: JOIN-SUCCESSOR-IF-
          ;; POSSIBLE and the LET merges of the next round both read the block
          ;; they are joining, and a conversion that put its nodes in the middle
          ;; of EXIT would leave them reading code that is about to be deleted.
          (let* ((start (make-ctran))
                 (middle (ctran-starts-block start)))
            ;; the emitted DOTIMES is what the source-form pass rewrites into the
            ;; packed loop, so that pass has to be on while it is converted: this
            ;; one recognizes a loop and emits one, it does not emit packs.
            (let ((*auto-vectorize* t))
              (ir1-convert start exit-ctran nil (vector-node-source-form plan)))
            ;; rewire the entry and leave the deletion to the pipeline.  Do not
            ;; call DELETE-BLOCK on the loop's blocks by hand: DELETE-LAMBDA
            ;; deliberately leaves a deleted lambda's blocks queued
            ;; (ir1util.lisp:2191-2194), so hand deletion leaves ENTRY/EXIT
            ;; bookkeeping inconsistent.  Do not flag the blocks first either --
            ;; DELETE-BLOCK-LAZILY skips a flagged block (ir1util.lisp:2402-2406)
            ;; and MARK-FOR-DELETION stops walking predecessors at one.
            (unlink-blocks preheader head)
            (link-blocks preheader middle)
            ;; Do NOT sweep here.  The replacement's own functionals (the LETs a
            ;; DOTIMES expands to) are queued in COMPONENT-NEW-FUNCTIONALS and only
            ;; become reachable once LOCALL-ANALYZE-COMPONENT runs, which happens
            ;; after this: a sweep now can delete the replacement as unreachable.
            ;; Flagging the component makes the next round's dfo-as-needed
            ;; (dfo.lisp:51-69 -> ir1util.lisp:2826) sweep the old region
            ;; transitively, by which time locall has run.
            (setf (component-reanalyze component) t)
            ;; Put the replacement through the fixpoint the rest of the component
            ;; already ran: locall-analyze-component drains
            ;; COMPONENT-NEW-FUNCTIONALS and LET converts what the DOTIMES
            ;; expanded to (main.lisp:503-506; locall.lisp:1411, 1044-1136), and
            ;; the next ir1-optimize round deletes the exits that creates
            ;; with MAYBE-DELETE-EXIT (ir1opt.lisp:516, 1239).  Re-running
            ;; IR1-PHASES here would build a fresh *CONSTRAINT-UNIVERSE* while the
            ;; component's lvars still index the old one (main.lisp:549-552).
            (locall-analyze-component component)
            ;; DFO and block numbers are stale; dfo-as-needed (main.lisp:764)
            ;; re-runs find-dfo on this flag (same setf as insert-cleanup-code,
            ;; ir1util.lisp:37, and node-ends-block, ir1util.lisp:2106).
            t))))))

