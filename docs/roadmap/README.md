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
- Build14/Attempt24/Attempt25 and Build18/Attempt28/Attempt29 remain immutable
  predecessor evidence. Build19/Attempt30/Attempt31/Pre-cloud09 is also
  retained, but Independent Review13 rejected its account-ready claim with
  three major findings: stale operator guidance and an SDK exception boundary
  that could expose temporary CAM-role credentials.
- Build20 with formal plan Attempt32, representative calibration Attempt33,
  and producer-side Pre-cloud10 is retained, but Independent Review14 rejected
  it because CVM credential construction/retrieval remained outside the safe
  SDK exception boundary. The closed successor is Build21 with formal plan
  Attempt34 and representative calibration Attempt35. Pre-cloud11 and
  Independent Review15 must report zero findings before the repository may reach
  `READY_FOR_TENCENT_ACCOUNT_INPUT_NOT_AUTHORIZED`.
- The corrected aggregate T1/T2 and conditional T3/T4 package remains bounded
  formal evidence. The 2,880-run formal matrix, holdout, Tencent API access,
  purchase, instance creation, and COS use remain unstarted and unauthorized.
- Level 2 and Level 3 remain required future gates and are not implemented.
- No push, remote rename, public package/archive release, or submission is
  authorized by this roadmap.

Roadmap status is sequencing and governance metadata, never implementation or
readiness evidence. Only independently replayed artifacts may support a result
claim.
