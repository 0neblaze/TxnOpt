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
- Clean Build Attempt07 passes local build, sanitizer, resource, and replay
  gates. Local calibration Attempt14 is complete but not Level 1 evidence:
  RCPSP passes the representative speedup threshold and EVRPTW does not.
- The corrected aggregate T1/T2 model and T3/T4 proof package are ready for a
  new independent review; Attempt01 remains immutable `NEEDS_WORK` evidence.
- Cloud procurement and the formal Level 1 matrix remain blocked. The current
  32-core estimate is only for provisional 1200-work/3-second budgets.
- Level 2 and Level 3 remain required future gates and are not implemented.
- No push, remote rename, public package/archive release, or submission is
  authorized by this roadmap.

Roadmap status is sequencing and governance metadata, never implementation or
readiness evidence. Only independently replayed artifacts may support a result
claim.
