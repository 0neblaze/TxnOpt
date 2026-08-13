# Refinement mapping for `txnopt-contract-v1`

Status: aggregate-transaction mapping corrected; ready for independent review.

The checked model and `PythonTxnRuntime` use the same publication granularity:
one candidate batch is evaluated privately and becomes visible in one atomic
round commit. Candidate completion is not a public commit.

| Formal variable or action | Python implementation | Native round receipt |
| --- | --- | --- |
| `txnPhase` | one aggregate `CandidateTxn.phase` | aggregate transaction phase |
| `pending`/`completed` | ghost view of private futures/results | private task receipts |
| `committedTrace` | committed round semantic prefix | no direct publication |
| `cacheVisible` | committed `InMemoryCacheStore` generation | no direct publication |
| `Reserve` | `BudgetLedger.reserve` plus cache snapshot | prepared work receipt |
| `Complete(candidate)` | a private future/result completion | physical task receipt only |
| `Validate` | ordered result reconstruction and Oracle validation | validated typed arrays |
| `Commit` | one runtime-owned state/cache publication | receipt applied by Python owner |
| `Terminate` | abort/interruption before publication | typed failure/interrupt status |

### Abstraction relation

At each visible Python step:

- formal `txnPhase` equals the aggregate `CandidateTxn.phase`;
- formal `pending` and `completed` are ghost sets derived from task receipts and
  future terminal state; neither is semantic output;
- formal `committedTrace` is the sequence of fully committed rounds represented
  by the runtime semantic stream;
- formal `cacheVisible` is the solve-local cache after the last successful
  `cache.commit`;
- staged cache entries and evaluated arrays have no abstract visible
  counterpart before commit.

Each physical candidate completion is therefore a stuttering step. Ordered
result reconstruction and validation refine `Validate`. The Python runtime then
applies one macro-step that publishes the selected state and commits all staged
cache entries; this refines formal `Commit`. Any deadline or fault before that
macro-step refines `Terminate` and leaves the previous committed prefix intact.

The native module is an adapter, not a second state owner. It may return private
typed arrays and physical task receipts, but it cannot publish Python state,
semantic trace, budget, or cache. For an internally parallel native batch, the
ghost completion sets are reconstructed from its receipt; the abstraction
relation still exposes only the final Python commit or abort.

The prior per-candidate publication model did not refine the implementation and
is superseded by this aggregate model. The new TLC receipt binds the aggregate
TLA+/PlusCal inputs, while acceptance of this mapping remains an independent
review action.
