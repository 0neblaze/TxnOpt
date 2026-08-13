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

The executable independent mapping is
`txnopt_evidence.refinement.replay_aggregate_refinement`. It treats
PREPARED/RESERVED/EVALUATING/VALIDATED as stuttering steps, permits visible
state and cache-generation changes only on one COMMITTED event, and requires
ABORTED/INTERRUPTED events to expose neither. The raw-only reviewer invokes this
module in its separate process and binds `aggregate_refinement_replay=PASS` to
the signed review; malformed visible intermediate state or non-atomic cache
generation is rejected.

The executable relation also enforces the single-owner part of the formal
state: at most one transaction may be non-terminal, every phase repeats the
same unique ordered candidate-key tuple and visible snapshot, and its length
must equal the immediately preceding `candidate_screening.admitted_count`.
Each whole-batch commit appends a domain-separated digest of that ordered tuple
to the replay receipt. Sequential rounds therefore refine sequential
instantiations of the one-transaction TLA+ model; interleaved owners, changed
candidate sets, missing admitted transactions, and reused private state are
rejected before a review can pass. The v1 semantic event set is closed during
replay: only screening, transaction, and round-termination events may appear
between the run boundaries, and no non-transaction event may carry a visible
state digest or cache generation.

A backend contract failure that leaves cache publication unknowable is retained
as an `INTERRUPTED` failure event with `cache_outcome="unknown"`. It is not a
refinement witness: the replay receipt sets `prefix_safety_proven=false`, and a
successful independent review must reject it. This preserves failure evidence
without falsely claiming either commit or rollback.
