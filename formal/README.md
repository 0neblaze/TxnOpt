# TxnOpt formal model

`TxnOpt.tla` models arbitrary physical completion order, canonical commit order,
atomic cache visibility, and failure termination for `txnopt-contract-v1`.
`TxnOpt.cfg` is the bounded four-candidate TLC model.
`TxnOptOrderedPlusCal.tla` contains the process-level PlusCal source for worker,
committer, and failure interleavings. `proofs.md` gives the generalized
inductive T1/T2 argument, the T3 counterexample, and the T4 algebraic bound.
`model-check-receipt.json` binds both model inputs, the generated PlusCal
translation, the exact `tla2tools.jar`, and the observed bounded state counts.

Re-run the receipt with:

```bash
python tools/verify_txnopt_formal.py \
  --java /path/to/java \
  --tla2tools /path/to/tla2tools.jar
```

The bounded TLA+/PlusCal checks are complete for four candidates. The formal
package is not independently accepted until a reviewer signs the refinement
mapping and T1--T4 proof package. Measured `Qmax` and `Cmax` are also required
before T4 may support a positive performance claim.
