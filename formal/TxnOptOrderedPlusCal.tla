----------------------- MODULE TxnOptOrderedPlusCal -----------------------
EXTENDS Integers, Sequences

CONSTANT CandidateCount

CandidateOrder == [index \in 1..CandidateCount |-> index]
CandidateIndices == 1..Len(CandidateOrder)
TxnPhases == {
    "PREPARED", "RESERVED", "EVALUATING", "VALIDATED", "COMMITTED",
    "ABORTED", "INTERRUPTED"
}

(* --algorithm TxnOptAtomicBatch {
variables
    pending = CandidateIndices,
    completed = {},
    committedTrace = <<>>,
    cacheVisible = {},
    terminated = FALSE,
    txnPhase = "PREPARED";

define {
    AtomicPublication ==
        \/ committedTrace = <<>>
        \/ committedTrace = CandidateOrder
    CacheRefinesTrace ==
        cacheVisible = {committedTrace[index] : index \in DOMAIN committedTrace}
    PhaseTypeOK == txnPhase \in TxnPhases
    PrivateCompletionOnly ==
        txnPhase # "COMMITTED" => /\ committedTrace = <<>> /\ cacheVisible = {}
    CommittedPublishesWholeBatch ==
        txnPhase = "COMMITTED" =>
            /\ committedTrace = CandidateOrder
            /\ cacheVisible = CandidateIndices
}

process (Worker \in CandidateIndices) {
AwaitEvaluation:
    await terminated \/ txnPhase = "EVALUATING";
Complete:
    if (~terminated) {
        pending := pending \ {self};
        completed := completed \cup {self};
    };
DoneWorker:
    skip;
}

process (Owner = 0) {
Reserve:
    if (~terminated) {
        txnPhase := "RESERVED";
    };
StartEvaluation:
    if (~terminated) {
        txnPhase := "EVALUATING";
    };
AwaitBatch:
    await terminated \/ pending = {};
Validate:
    if (~terminated) {
        txnPhase := "VALIDATED";
    };
Commit:
    if (~terminated) {
        committedTrace := CandidateOrder;
        cacheVisible := CandidateIndices;
        txnPhase := "COMMITTED";
    };
DoneOwner:
    skip;
}

process (Failure = -1) {
ChooseFailure:
    either {
        if (txnPhase # "COMMITTED") {
            terminated := TRUE;
            pending := {};
            completed := {};
            either {
                txnPhase := "ABORTED";
            } or {
                txnPhase := "INTERRUPTED";
            };
        };
    } or {
        skip;
    };
DoneFailure:
    skip;
}
} *)
\* BEGIN TRANSLATION (chksum(pcal) = "17f11d5e" /\ chksum(tla) = "bb29cb5c")
VARIABLES pending, completed, committedTrace, cacheVisible, terminated, 
          txnPhase, pc

(* define statement *)
AtomicPublication ==
    \/ committedTrace = <<>>
    \/ committedTrace = CandidateOrder
CacheRefinesTrace ==
    cacheVisible = {committedTrace[index] : index \in DOMAIN committedTrace}
PhaseTypeOK == txnPhase \in TxnPhases
PrivateCompletionOnly ==
    txnPhase # "COMMITTED" => /\ committedTrace = <<>> /\ cacheVisible = {}
CommittedPublishesWholeBatch ==
    txnPhase = "COMMITTED" =>
        /\ committedTrace = CandidateOrder
        /\ cacheVisible = CandidateIndices


vars == << pending, completed, committedTrace, cacheVisible, terminated, 
           txnPhase, pc >>

ProcSet == (CandidateIndices) \cup {0} \cup {-1}

Init == (* Global variables *)
        /\ pending = CandidateIndices
        /\ completed = {}
        /\ committedTrace = <<>>
        /\ cacheVisible = {}
        /\ terminated = FALSE
        /\ txnPhase = "PREPARED"
        /\ pc = [self \in ProcSet |-> CASE self \in CandidateIndices -> "AwaitEvaluation"
                                        [] self = 0 -> "Reserve"
                                        [] self = -1 -> "ChooseFailure"]

AwaitEvaluation(self) == /\ pc[self] = "AwaitEvaluation"
                         /\ terminated \/ txnPhase = "EVALUATING"
                         /\ pc' = [pc EXCEPT ![self] = "Complete"]
                         /\ UNCHANGED << pending, completed, committedTrace, 
                                         cacheVisible, terminated, txnPhase >>

Complete(self) == /\ pc[self] = "Complete"
                  /\ IF ~terminated
                        THEN /\ pending' = pending \ {self}
                             /\ completed' = (completed \cup {self})
                        ELSE /\ TRUE
                             /\ UNCHANGED << pending, completed >>
                  /\ pc' = [pc EXCEPT ![self] = "DoneWorker"]
                  /\ UNCHANGED << committedTrace, cacheVisible, terminated, 
                                  txnPhase >>

DoneWorker(self) == /\ pc[self] = "DoneWorker"
                    /\ TRUE
                    /\ pc' = [pc EXCEPT ![self] = "Done"]
                    /\ UNCHANGED << pending, completed, committedTrace, 
                                    cacheVisible, terminated, txnPhase >>

Worker(self) == AwaitEvaluation(self) \/ Complete(self) \/ DoneWorker(self)

Reserve == /\ pc[0] = "Reserve"
           /\ IF ~terminated
                 THEN /\ txnPhase' = "RESERVED"
                 ELSE /\ TRUE
                      /\ UNCHANGED txnPhase
           /\ pc' = [pc EXCEPT ![0] = "StartEvaluation"]
           /\ UNCHANGED << pending, completed, committedTrace, cacheVisible, 
                           terminated >>

StartEvaluation == /\ pc[0] = "StartEvaluation"
                   /\ IF ~terminated
                         THEN /\ txnPhase' = "EVALUATING"
                         ELSE /\ TRUE
                              /\ UNCHANGED txnPhase
                   /\ pc' = [pc EXCEPT ![0] = "AwaitBatch"]
                   /\ UNCHANGED << pending, completed, committedTrace, 
                                   cacheVisible, terminated >>

AwaitBatch == /\ pc[0] = "AwaitBatch"
              /\ terminated \/ pending = {}
              /\ pc' = [pc EXCEPT ![0] = "Validate"]
              /\ UNCHANGED << pending, completed, committedTrace, cacheVisible, 
                              terminated, txnPhase >>

Validate == /\ pc[0] = "Validate"
            /\ IF ~terminated
                  THEN /\ txnPhase' = "VALIDATED"
                  ELSE /\ TRUE
                       /\ UNCHANGED txnPhase
            /\ pc' = [pc EXCEPT ![0] = "Commit"]
            /\ UNCHANGED << pending, completed, committedTrace, cacheVisible, 
                            terminated >>

Commit == /\ pc[0] = "Commit"
          /\ IF ~terminated
                THEN /\ committedTrace' = CandidateOrder
                     /\ cacheVisible' = CandidateIndices
                     /\ txnPhase' = "COMMITTED"
                ELSE /\ TRUE
                     /\ UNCHANGED << committedTrace, cacheVisible, txnPhase >>
          /\ pc' = [pc EXCEPT ![0] = "DoneOwner"]
          /\ UNCHANGED << pending, completed, terminated >>

DoneOwner == /\ pc[0] = "DoneOwner"
             /\ TRUE
             /\ pc' = [pc EXCEPT ![0] = "Done"]
             /\ UNCHANGED << pending, completed, committedTrace, cacheVisible, 
                             terminated, txnPhase >>

Owner == Reserve \/ StartEvaluation \/ AwaitBatch \/ Validate \/ Commit
            \/ DoneOwner

ChooseFailure == /\ pc[-1] = "ChooseFailure"
                 /\ \/ /\ IF txnPhase # "COMMITTED"
                             THEN /\ terminated' = TRUE
                                  /\ pending' = {}
                                  /\ completed' = {}
                                  /\ \/ /\ txnPhase' = "ABORTED"
                                     \/ /\ txnPhase' = "INTERRUPTED"
                             ELSE /\ TRUE
                                  /\ UNCHANGED << pending, completed, 
                                                  terminated, txnPhase >>
                    \/ /\ TRUE
                       /\ UNCHANGED <<pending, completed, terminated, txnPhase>>
                 /\ pc' = [pc EXCEPT ![-1] = "DoneFailure"]
                 /\ UNCHANGED << committedTrace, cacheVisible >>

DoneFailure == /\ pc[-1] = "DoneFailure"
               /\ TRUE
               /\ pc' = [pc EXCEPT ![-1] = "Done"]
               /\ UNCHANGED << pending, completed, committedTrace, 
                               cacheVisible, terminated, txnPhase >>

Failure == ChooseFailure \/ DoneFailure

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == Owner \/ Failure
           \/ (\E self \in CandidateIndices: Worker(self))
           \/ Terminating

Spec == Init /\ [][Next]_vars

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION

=============================================================================
