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

VARIABLES pending, completed, committedTrace, cacheVisible, terminated, txnPhase

vars == <<pending, completed, committedTrace, cacheVisible, terminated, txnPhase>>

Init ==
    /\ CandidateCount > 0
    /\ pending = CandidateIndices
    /\ completed = {}
    /\ committedTrace = <<>>
    /\ cacheVisible = {}
    /\ terminated = FALSE
    /\ txnPhase = "PREPARED"

Reserve ==
    /\ ~terminated
    /\ txnPhase = "PREPARED"
    /\ txnPhase' = "RESERVED"
    /\ UNCHANGED <<pending, completed, committedTrace, cacheVisible, terminated>>

StartEvaluation ==
    /\ ~terminated
    /\ txnPhase = "RESERVED"
    /\ txnPhase' = "EVALUATING"
    /\ UNCHANGED <<pending, completed, committedTrace, cacheVisible, terminated>>

Complete(candidate) ==
    /\ ~terminated
    /\ txnPhase = "EVALUATING"
    /\ candidate \in pending
    /\ pending' = pending \ {candidate}
    /\ completed' = completed \cup {candidate}
    /\ UNCHANGED <<committedTrace, cacheVisible, terminated, txnPhase>>

Validate ==
    /\ ~terminated
    /\ txnPhase = "EVALUATING"
    /\ pending = {}
    /\ txnPhase' = "VALIDATED"
    /\ UNCHANGED <<pending, completed, committedTrace, cacheVisible, terminated>>

Commit ==
    /\ ~terminated
    /\ txnPhase = "VALIDATED"
    /\ committedTrace' = CandidateOrder
    /\ cacheVisible' = CandidateIndices
    /\ txnPhase' = "COMMITTED"
    /\ UNCHANGED <<pending, completed, terminated>>

Terminate ==
    /\ ~terminated
    /\ txnPhase # "COMMITTED"
    /\ terminated' = TRUE
    /\ pending' = {}
    /\ completed' = {}
    /\ txnPhase' \in {"ABORTED", "INTERRUPTED"}
    /\ UNCHANGED <<committedTrace, cacheVisible>>

Next ==
    Reserve
    \/ StartEvaluation
    \/ (\E candidate \in pending : Complete(candidate))
    \/ Validate
    \/ Commit
    \/ Terminate

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ pending \subseteq CandidateIndices
    /\ completed \subseteq CandidateIndices
    /\ pending \cap completed = {}
    /\ committedTrace \in Seq(CandidateIndices)
    /\ cacheVisible \subseteq CandidateIndices
    /\ terminated \in BOOLEAN
    /\ txnPhase \in TxnPhases

CanonicalAtomicPublication ==
    \/ committedTrace = <<>>
    \/ committedTrace = CandidateOrder

CacheOnlyAfterCommit == cacheVisible = SeqToSet(committedTrace)

PrivateCompletionOnly ==
    txnPhase # "COMMITTED" => /\ committedTrace = <<>> /\ cacheVisible = {}

CommittedPublishesWholeBatch ==
    txnPhase = "COMMITTED" =>
        /\ committedTrace = CandidateOrder
        /\ cacheVisible = CandidateIndices

LastCompletePrefixOnFailure ==
    terminated => /\ committedTrace = <<>> /\ cacheVisible = {}

=============================================================================
