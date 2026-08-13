# Draft refinement mapping for `txnopt-contract-v1`

Status: T1/T2 implementation mapping; independent review pending.

| Formal variable or action | Python contract | Planned native receipt |
| --- | --- | --- |
| `phase` | `CandidateTxn.phase` | transaction phase code |
| `pending`/`completed` | private evaluation futures/results | physical task receipts |
| `nextCommit` | canonical resolved-result index | canonical resolution-order array |
| `committedTrace` | committed semantic event prefix | semantic counters/order arrays |
| `cacheVisible` | cache transaction publication | future native cache generation |
| `terminated` | `RunResult.termination_reason` | typed phase/interrupt receipt |
| `Complete(candidate)` | barrier/ordered physical completion | physical task receipt only |
| `Commit` | runtime-owned state and cache publication | validated round receipt applied by Python runtime |
| `Terminate` | deadline, budget, worker, validation, snapshot, or write failure | `ABORTED`/`INTERRUPTED` receipt |

`PythonTxnRuntime` owns the abstract state and cache generation. The native
module returns a prepared, validated result; it never publishes Python state or
cache. This preserves one commit owner while allowing one packed native call per
round. The native receipt refines the evaluation transition; reservation,
validation, commit, rollback, budget settlement, and cache publication refine
through the Python runtime. Independent review is still required before this
mapping is accepted.
