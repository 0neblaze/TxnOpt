# Research roadmap

The current public roadmap is
[`txnopt-level1-to-level3-roadmap.md`](txnopt-level1-to-level3-roadmap.md).
It governs the in-place transition from the frozen EVRPTW implementation to
TxnOpt. Level 1, Level 2, and Level 3 are mandatory sequential gates; a later
level cannot be opened by relabelling a failed earlier result.

The unpublished strategy appendix is maintained locally as
`docs/roadmap/txnopt-research-strategy.local.md` and is intentionally ignored.
The earlier local EVRPTW roadmap remains ignored historical planning input.

## Current status

- The legacy implementation is frozen at tag `stage052-legacy-freeze-v1`.
- Stage 5.2 performance-calibration Attempt16 is retained and closed as a
  known unique failure; it is not a successful benchmark result.
- TxnOpt Level 1 implementation is in progress on `codex/txnopt-level1`.
- Clean Build Attempt09 and its Attempt17/Attempt18 pre-cloud chain remain
  immutable prior evidence. Clean Build10 now binds the active native prepared
  transaction/reservation/cache-delta/trace path. Its additive refinement note
  remains pending independent review, and a new calibration, formal plan, and
  pre-cloud receipt are required.
- The corrected aggregate T1/T2 and conditional T3/T4 package passes its current
  independent formal review. Legacy compatibility Attempt01 resolves the old
  timeout while retaining the immutable historical source-receipt mismatch.
- Cloud procurement and the formal Level 1 matrix are blocked while the active
  successor identity is rebuilt. The prior 32-core estimate was about 1.20
  hours for Build09's provisional 1200-work/3-second budgets; it must not be
  relabelled as a forecast for the new native source.
- Level 2 and Level 3 remain required future gates and are not implemented.
- No push, remote rename, public package/archive release, or submission is
  authorized by this roadmap.

Roadmap status is sequencing and governance metadata, never implementation or
readiness evidence. Only independently replayed artifacts may support a result
claim.
