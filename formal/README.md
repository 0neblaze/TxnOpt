# TxnOpt formal model

`TxnOpt.tla` models arbitrary physical completion order, canonical commit order,
atomic cache visibility, and failure termination for `txnopt-contract-v1`.
`TxnOpt.cfg` is the bounded four-candidate TLC model.
`TxnOptOrderedPlusCal.tla` contains the process-level PlusCal source for worker,
committer, and failure interleavings. `TxnOptT3.tla` checks the finite
wait/discard/use quotient and weak-fair termination used by the T3 meta-theorem.
`proofs.md` gives the generalized inductive T1/T2 argument, the quantified T3
proof, and the T4 algebraic and measured-cost bounds. `model-check-receipt.json`
binds all model inputs, the generated PlusCal translation, the exact
`tla2tools.jar`, and the observed bounded state counts.

Re-run the receipt with:

```bash
python tools/verify_txnopt_formal.py \
  --java /path/to/java \
  --tla2tools /path/to/tla2tools.jar
```

The bounded T1/T2 safety checks are complete for four candidates, and the T3
quotient checks both its no-triple invariant and termination under weak fairness.
Review Attempts01 and 03 remain immutable `NEEDS_WORK` history. Correction
Attempt04 and independent Review Attempt05 pass the scheduler-relative
conditional T3 theorem, independently recomputed T4 unit/Cmax bounds, and the
single-owner aggregate refinement replay; they do not relabel earlier reviews
or turn the five-state quotient into a general-domain mechanized theorem.
`txnopt-physical-trace-v1` now binds each observed run to a conservative elapsed
transaction `Cmax`; Level 2 still requires independent review and full-scope
measurements before any positive claim.

Build10 adds a private prepared native-round receipt without changing the
aggregate semantic trace. Its exact external review scope is frozen in
`reviews/txnopt_native_round_refinement_review_request_attempt06.json`. That
file is a request, not a review: until a separate reviewer returns a new signed
attempt with zero critical and major findings, Build10 remains
`REVIEW_PENDING_BUILD10` and Pre-cloud Attempt02 remains blocked.
