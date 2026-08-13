# TxnOpt formal model

`TxnOpt.tla` is the first executable state-machine skeleton for
`txnopt-contract-v1`; `TxnOpt.cfg` defines a bounded TLC model. The current
model checks transaction type safety, work-at-start accounting, and that failed
or interrupted transactions cannot expose cache writes.

This is **not** a completed proof of T1-T4. Completion requires:

1. a multi-candidate random-tape model with arbitrary physical completion
   permutations for T1;
2. an explicit last-commit-prefix abstraction for deadline and worker failure
   in T2;
3. a PlusCal counterexample construction for T3;
4. parameterized accounting and a non-vacuity check against measured
   `W`, `Qmax`, `Cmax`, `P`, and remaining budget for T4;
5. an independently reviewed refinement mapping to Python and native receipts.

See `refinement-mapping.md` for the current implementation correspondence.
