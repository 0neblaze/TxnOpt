----------------------- MODULE TxnOptOrderedPlusCal -----------------------
EXTENDS Integers, Sequences

CONSTANT CandidateCount

CandidateOrder == [index \in 1..CandidateCount |-> index]
TxnPhases == {
    "PREPARED", "RESERVED", "EVALUATING", "VALIDATED", "COMMITTED",
    "ABORTED", "INTERRUPTED"
}

(* --algorithm TxnOptOrdered {
variables
    pending = 1..Len(CandidateOrder),
    completed = {},
    nextCommit = 1,
    committedTrace = <<>>,
    cacheVisible = {},
    terminated = FALSE,
    txnPhase = [candidate \in 1..Len(CandidateOrder) |-> "PREPARED"];

define {
    CanonicalPrefix ==
        committedTrace = SubSeq(CandidateOrder, 1, Len(committedTrace))
    CacheRefinesTrace ==
        cacheVisible = {committedTrace[index] : index \in DOMAIN committedTrace}
    PhaseTypeOK == txnPhase \in [1..Len(CandidateOrder) -> TxnPhases]
    CommittedPhaseMatchesTrace ==
        {candidate \in 1..Len(CandidateOrder) :
            txnPhase[candidate] = "COMMITTED"} = cacheVisible
    ValidatedMatchesCompleted ==
        ~terminated =>
            ({candidate \in 1..Len(CandidateOrder) :
                txnPhase[candidate] = "VALIDATED"} = completed)
}

process (Worker \in 1..Len(CandidateOrder)) {
Reserve:
    await ~terminated /\ self \in pending;
    txnPhase[self] := "RESERVED";
Evaluate:
    await ~terminated;
    txnPhase[self] := "EVALUATING";
Complete:
    await ~terminated;
    pending := pending \ {self};
    completed := completed \cup {self};
    txnPhase[self] := "VALIDATED";
DoneWorker:
    skip;
}

process (Committer = 0) {
CommitLoop:
    while (~terminated /\ nextCommit <= Len(CandidateOrder)) {
AwaitNext:
        await nextCommit \in completed;
CommitNext:
        await ~terminated;
        completed := completed \ {nextCommit};
        committedTrace := Append(committedTrace, CandidateOrder[nextCommit]);
        cacheVisible := cacheVisible \cup {CandidateOrder[nextCommit]};
        txnPhase[nextCommit] := "COMMITTED";
        nextCommit := nextCommit + 1;
    };
DoneCommitter:
    skip;
}

process (Failure = -1) {
ChooseFailure:
    either {
        terminated := TRUE;
        pending := {};
        completed := {};
        txnPhase := [candidate \in 1..Len(CandidateOrder) |->
            IF txnPhase[candidate] = "COMMITTED" THEN "COMMITTED"
            ELSE IF txnPhase[candidate] = "EVALUATING" THEN "INTERRUPTED"
            ELSE "ABORTED"];
    } or {
        skip;
    };
DoneFailure:
    skip;
}
} *)
\* BEGIN TRANSLATION (chksum(pcal) = "538b992b" /\ chksum(tla) = "a123ddd6")
VARIABLES pending, completed, nextCommit, committedTrace, cacheVisible,
          terminated, txnPhase, pc

(* define statement *)
CanonicalPrefix ==
    committedTrace = SubSeq(CandidateOrder, 1, Len(committedTrace))
CacheRefinesTrace ==
    cacheVisible = {committedTrace[index] : index \in DOMAIN committedTrace}
PhaseTypeOK == txnPhase \in [1..Len(CandidateOrder) -> TxnPhases]
CommittedPhaseMatchesTrace ==
    {candidate \in 1..Len(CandidateOrder) :
        txnPhase[candidate] = "COMMITTED"} = cacheVisible
ValidatedMatchesCompleted ==
    ~terminated =>
        ({candidate \in 1..Len(CandidateOrder) :
            txnPhase[candidate] = "VALIDATED"} = completed)


vars == << pending, completed, nextCommit, committedTrace, cacheVisible,
           terminated, txnPhase, pc >>

ProcSet == (1..Len(CandidateOrder)) \cup {0} \cup {-1}

Init == (* Global variables *)
        /\ pending = 1..Len(CandidateOrder)
        /\ completed = {}
        /\ nextCommit = 1
        /\ committedTrace = <<>>
        /\ cacheVisible = {}
        /\ terminated = FALSE
        /\ txnPhase = [candidate \in 1..Len(CandidateOrder) |-> "PREPARED"]
        /\ pc = [self \in ProcSet |-> CASE self \in 1..Len(CandidateOrder) -> "Reserve"
                                        [] self = 0 -> "CommitLoop"
                                        [] self = -1 -> "ChooseFailure"]

Reserve(self) == /\ pc[self] = "Reserve"
                 /\ ~terminated /\ self \in pending
                 /\ txnPhase' = [txnPhase EXCEPT ![self] = "RESERVED"]
                 /\ pc' = [pc EXCEPT ![self] = "Evaluate"]
                 /\ UNCHANGED << pending, completed, nextCommit,
                                 committedTrace, cacheVisible, terminated >>

Evaluate(self) == /\ pc[self] = "Evaluate"
                  /\ ~terminated
                  /\ txnPhase' = [txnPhase EXCEPT ![self] = "EVALUATING"]
                  /\ pc' = [pc EXCEPT ![self] = "Complete"]
                  /\ UNCHANGED << pending, completed, nextCommit,
                                  committedTrace, cacheVisible, terminated >>

Complete(self) == /\ pc[self] = "Complete"
                  /\ ~terminated
                  /\ pending' = pending \ {self}
                  /\ completed' = (completed \cup {self})
                  /\ txnPhase' = [txnPhase EXCEPT ![self] = "VALIDATED"]
                  /\ pc' = [pc EXCEPT ![self] = "DoneWorker"]
                  /\ UNCHANGED << nextCommit, committedTrace, cacheVisible,
                                  terminated >>

DoneWorker(self) == /\ pc[self] = "DoneWorker"
                    /\ TRUE
                    /\ pc' = [pc EXCEPT ![self] = "Done"]
                    /\ UNCHANGED << pending, completed, nextCommit,
                                    committedTrace, cacheVisible, terminated,
                                    txnPhase >>

Worker(self) == Reserve(self) \/ Evaluate(self) \/ Complete(self)
                   \/ DoneWorker(self)

CommitLoop == /\ pc[0] = "CommitLoop"
              /\ IF ~terminated /\ nextCommit <= Len(CandidateOrder)
                    THEN /\ pc' = [pc EXCEPT ![0] = "AwaitNext"]
                    ELSE /\ pc' = [pc EXCEPT ![0] = "DoneCommitter"]
              /\ UNCHANGED << pending, completed, nextCommit, committedTrace,
                              cacheVisible, terminated, txnPhase >>

AwaitNext == /\ pc[0] = "AwaitNext"
             /\ nextCommit \in completed
             /\ pc' = [pc EXCEPT ![0] = "CommitNext"]
             /\ UNCHANGED << pending, completed, nextCommit, committedTrace,
                             cacheVisible, terminated, txnPhase >>

CommitNext == /\ pc[0] = "CommitNext"
              /\ ~terminated
              /\ completed' = completed \ {nextCommit}
              /\ committedTrace' = Append(committedTrace, CandidateOrder[nextCommit])
              /\ cacheVisible' = (cacheVisible \cup {CandidateOrder[nextCommit]})
              /\ txnPhase' = [txnPhase EXCEPT ![nextCommit] = "COMMITTED"]
              /\ nextCommit' = nextCommit + 1
              /\ pc' = [pc EXCEPT ![0] = "CommitLoop"]
              /\ UNCHANGED << pending, terminated >>

DoneCommitter == /\ pc[0] = "DoneCommitter"
                 /\ TRUE
                 /\ pc' = [pc EXCEPT ![0] = "Done"]
                 /\ UNCHANGED << pending, completed, nextCommit,
                                 committedTrace, cacheVisible, terminated,
                                 txnPhase >>

Committer == CommitLoop \/ AwaitNext \/ CommitNext \/ DoneCommitter

ChooseFailure == /\ pc[-1] = "ChooseFailure"
                 /\ \/ /\ terminated' = TRUE
                       /\ pending' = {}
                       /\ completed' = {}
                       /\ txnPhase' =         [candidate \in 1..Len(CandidateOrder) |->
                                      IF txnPhase[candidate] = "COMMITTED" THEN "COMMITTED"
                                      ELSE IF txnPhase[candidate] = "EVALUATING" THEN "INTERRUPTED"
                                      ELSE "ABORTED"]
                    \/ /\ TRUE
                       /\ UNCHANGED <<pending, completed, terminated, txnPhase>>
                 /\ pc' = [pc EXCEPT ![-1] = "DoneFailure"]
                 /\ UNCHANGED << nextCommit, committedTrace, cacheVisible >>

DoneFailure == /\ pc[-1] = "DoneFailure"
               /\ TRUE
               /\ pc' = [pc EXCEPT ![-1] = "Done"]
               /\ UNCHANGED << pending, completed, nextCommit, committedTrace,
                               cacheVisible, terminated, txnPhase >>

Failure == ChooseFailure \/ DoneFailure

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == Committer \/ Failure
           \/ (\E self \in 1..Len(CandidateOrder): Worker(self))
           \/ Terminating

Spec == Init /\ [][Next]_vars

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION

=============================================================================
