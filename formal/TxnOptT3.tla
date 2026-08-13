------------------------------- MODULE TxnOptT3 -------------------------------
EXTENDS TLC

(*
Finite quotient of the scheduler-relative adversarial T3 witness class.
successorStarted represents a non-empty finite set of work started from
pre-resolution information whose every member is observationally inequivalent
to the realized canonical successor.  At predecessor acceptance, the scheduler
is observationally equivalent to one of three outcomes: it waited,
discarded/recomputed the non-equivalent work, or used it.
*)

VARIABLES phase, successorStarted, outcome

vars == <<phase, successorStarted, outcome>>

Init ==
    /\ phase = "PENDING"
    /\ successorStarted = FALSE
    /\ outcome = "none"

StartSuccessor ==
    /\ phase = "PENDING"
    /\ ~successorStarted
    /\ successorStarted' = TRUE
    /\ UNCHANGED <<phase, outcome>>

ResolveWait ==
    /\ phase = "PENDING"
    /\ ~successorStarted
    /\ phase' = "DONE"
    /\ outcome' = "wait"
    /\ UNCHANGED successorStarted

ResolveDiscard ==
    /\ phase = "PENDING"
    /\ successorStarted
    /\ phase' = "DONE"
    /\ outcome' = "discard"
    /\ UNCHANGED successorStarted

ResolveUse ==
    /\ phase = "PENDING"
    /\ successorStarted
    /\ phase' = "DONE"
    /\ outcome' = "use"
    /\ UNCHANGED successorStarted

Resolve == ResolveWait \/ ResolveDiscard \/ ResolveUse
Next == StartSuccessor \/ Resolve

Spec == Init /\ [][Next]_vars /\ WF_vars(Resolve)

SI == outcome # "use"
ZW == outcome # "discard"
AP == successorStarted

TypeOK ==
    /\ phase \in {"PENDING", "DONE"}
    /\ successorStarted \in BOOLEAN
    /\ outcome \in {"none", "wait", "discard", "use"}

NoTripleAtTermination == phase = "DONE" => ~(SI /\ ZW /\ AP)
Termination == <>(phase = "DONE")

=============================================================================
