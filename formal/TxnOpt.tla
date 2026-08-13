------------------------------ MODULE TxnOpt ------------------------------
EXTENDS Naturals, Sequences

CONSTANT WorkBudget

VARIABLES phase, startedWork, committed, cacheVisible, semanticEvents

vars == <<phase, startedWork, committed, cacheVisible, semanticEvents>>

Phases == {
    "PREPARED", "RESERVED", "EVALUATING", "VALIDATED", "COMMITTED",
    "ABORTED", "INTERRUPTED"
}

Init ==
    /\ phase = "PREPARED"
    /\ startedWork = 0
    /\ committed = FALSE
    /\ cacheVisible = FALSE
    /\ semanticEvents = <<>>

Reserve ==
    /\ phase = "PREPARED"
    /\ startedWork < WorkBudget
    /\ phase' = "RESERVED"
    /\ UNCHANGED <<startedWork, committed, cacheVisible, semanticEvents>>

BeginEvaluation ==
    /\ phase = "RESERVED"
    /\ startedWork < WorkBudget
    /\ phase' = "EVALUATING"
    /\ startedWork' = startedWork + 1
    /\ UNCHANGED <<committed, cacheVisible, semanticEvents>>

Validate ==
    /\ phase = "EVALUATING"
    /\ phase' = "VALIDATED"
    /\ UNCHANGED <<startedWork, committed, cacheVisible, semanticEvents>>

Commit ==
    /\ phase = "VALIDATED"
    /\ phase' = "COMMITTED"
    /\ committed' = TRUE
    /\ cacheVisible' = TRUE
    /\ semanticEvents' = Append(semanticEvents, "COMMITTED")
    /\ UNCHANGED startedWork

Abort ==
    /\ phase \in {"PREPARED", "RESERVED", "EVALUATING", "VALIDATED"}
    /\ phase' = "ABORTED"
    /\ UNCHANGED <<startedWork, committed, cacheVisible, semanticEvents>>

Interrupt ==
    /\ phase \in {"PREPARED", "RESERVED", "EVALUATING", "VALIDATED"}
    /\ phase' = "INTERRUPTED"
    /\ UNCHANGED <<startedWork, committed, cacheVisible, semanticEvents>>

Next == Reserve \/ BeginEvaluation \/ Validate \/ Commit \/ Abort \/ Interrupt

Spec == Init /\ [][Next]_vars

TypeOK ==
    /\ phase \in Phases
    /\ startedWork \in 0..WorkBudget
    /\ committed \in BOOLEAN
    /\ cacheVisible \in BOOLEAN
    /\ semanticEvents \in Seq({"COMMITTED"})

WorkChargedAtStart ==
    phase \in {"EVALUATING", "VALIDATED", "COMMITTED"} => startedWork > 0

CacheOnlyAfterCommit == cacheVisible => committed

FailureHasNoVisibleWrite ==
    phase \in {"ABORTED", "INTERRUPTED"} => ~cacheVisible

=============================================================================
