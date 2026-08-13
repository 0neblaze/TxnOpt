# Draft refinement mapping for `txnopt-contract-v1`

Status: structural draft; not an accepted proof.

| Formal variable or action | Python contract | Planned native receipt |
| --- | --- | --- |
| `phase` | `CandidateTxn.phase` | transaction phase code |
| `startedWork` | budget ledger reservation/settlement | started request counter |
| `committed` | runtime-owned state replacement | committed transaction bit |
| `cacheVisible` | cache transaction publication | cache commit generation |
| `semanticEvents` | committed semantic trace records | semantic event SoA |
| `Reserve` | runtime budget reservation | round reservation receipt |
| `BeginEvaluation` | ordered oracle batch launch | started-work receipt |
| `Validate` | `Oracle.validate` for every result | validation status array |
| `Commit` | sole `TxnRuntime` state/cache commit | typed commit receipt |
| `Abort` | rollback on validation, snapshot, or write failure | abort receipt |
| `Interrupt` | deadline, insufficient budget, late result, worker failure | interrupt receipt |

The mapping is incomplete until the Python reference runtime and
`txnopt-native-round-v1` exist. In particular, T1 needs canonical candidate and
random-tape identities, and T2 needs an explicit prior committed-state digest.
