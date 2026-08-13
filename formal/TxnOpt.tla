------------------------------ MODULE TxnOpt ------------------------------
EXTENDS Integers, Sequences, FiniteSets

CONSTANT CandidateCount

CandidateOrder == [index \in 1..CandidateCount |-> index]
CandidateIndices == 1..Len(CandidateOrder)
SeqToSet(sequence) == {sequence[index] : index \in DOMAIN sequence}
TxnPhases == {
    "PREPARED", "RESERVED", "EVALUATING", "VALIDATED", "COMMITTED",
    "ABORTED", "INTERRUPTED"
}

VARIABLES pending, completed, nextCommit, committedTrace, cacheVisible, terminated,
          txnPhase

vars == <<pending, completed, nextCommit, committedTrace, cacheVisible, terminated,
          txnPhase>>

Init ==
    /\ CandidateCount > 0
    /\ pending = CandidateIndices
    /\ completed = {}
    /\ nextCommit = 1
    /\ committedTrace = <<>>
    /\ cacheVisible = {}
    /\ terminated = FALSE
    /\ txnPhase = [candidate \in CandidateIndices |-> "PREPARED"]

Reserve(candidate) ==
    /\ ~terminated
    /\ candidate \in pending
    /\ txnPhase[candidate] = "PREPARED"
    /\ txnPhase' = [txnPhase EXCEPT ![candidate] = "RESERVED"]
    /\ UNCHANGED <<pending, completed, nextCommit, committedTrace,
                    cacheVisible, terminated>>

StartEvaluation(candidate) ==
    /\ ~terminated
    /\ candidate \in pending
    /\ txnPhase[candidate] = "RESERVED"
    /\ txnPhase' = [txnPhase EXCEPT ![candidate] = "EVALUATING"]
    /\ UNCHANGED <<pending, completed, nextCommit, committedTrace,
                    cacheVisible, terminated>>

Complete(candidate) ==
    /\ ~terminated
    /\ candidate \in pending
    /\ txnPhase[candidate] = "EVALUATING"
    /\ pending' = pending \ {candidate}
    /\ completed' = completed \cup {candidate}
    /\ txnPhase' = [txnPhase EXCEPT ![candidate] = "VALIDATED"]
    /\ UNCHANGED <<nextCommit, committedTrace, cacheVisible, terminated>>

Commit ==
    /\ ~terminated
    /\ nextCommit \in completed
    /\ completed' = completed \ {nextCommit}
    /\ committedTrace' = Append(committedTrace, CandidateOrder[nextCommit])
    /\ cacheVisible' = cacheVisible \cup {CandidateOrder[nextCommit]}
    /\ txnPhase' = [txnPhase EXCEPT ![nextCommit] = "COMMITTED"]
    /\ nextCommit' = nextCommit + 1
    /\ UNCHANGED <<pending, terminated>>

Terminate ==
    /\ ~terminated
    /\ terminated' = TRUE
    /\ pending' = {}
    /\ completed' = {}
    /\ txnPhase' = [candidate \in CandidateIndices |->
           IF txnPhase[candidate] = "COMMITTED" THEN "COMMITTED"
           ELSE IF txnPhase[candidate] = "EVALUATING" THEN "INTERRUPTED"
           ELSE "ABORTED"]
    /\ UNCHANGED <<nextCommit, committedTrace, cacheVisible>>

Next ==
    (\E candidate \in pending : Reserve(candidate))
    \/ (\E candidate \in pending : StartEvaluation(candidate))
    \/ (\E candidate \in pending : Complete(candidate))
    \/ Commit
    \/ Terminate

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pending \subseteq CandidateIndices
    /\ completed \subseteq CandidateIndices
    /\ pending \cap completed = {}
    /\ nextCommit \in 1..(Len(CandidateOrder) + 1)
    /\ committedTrace \in Seq(SeqToSet(CandidateOrder))
    /\ cacheVisible \subseteq SeqToSet(CandidateOrder)
    /\ terminated \in BOOLEAN
    /\ txnPhase \in [CandidateIndices -> TxnPhases]

CanonicalCommitPrefix ==
    committedTrace = SubSeq(CandidateOrder, 1, Len(committedTrace))

CacheOnlyAfterCommit == cacheVisible = SeqToSet(committedTrace)

CommittedPhaseMatchesTrace ==
    {candidate \in CandidateIndices : txnPhase[candidate] = "COMMITTED"}
        = SeqToSet(committedTrace)

ValidatedMatchesCompleted ==
    ~terminated =>
        ({candidate \in CandidateIndices : txnPhase[candidate] = "VALIDATED"}
            = completed)

LastCompletePrefixOnFailure == terminated => CanonicalCommitPrefix

=============================================================================
