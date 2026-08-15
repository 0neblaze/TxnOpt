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
  SDK exception boundary. Build21 with formal plan Attempt34, representative
  calibration Attempt35, deployment Attempt04, and Pre-cloud11 is also retained
  as rejected evidence: Independent Review15 found that the safe outer error
  still retained the credential-bearing SDK exception through `__context__`.
  Build22 is retained as a failed build because its first offline invocation
  bound an empty UV cache. Build23, Attempt38, Attempt39, Deployment05 and
  Pre-cloud12 are retained as rejected by Independent Review16 for two major
  findings. Build24, Attempt40, Attempt41, Deployment06, and Pre-cloud13 then
  completed their local producer gates, but Independent Review17 retained one
  major active-document cutover finding. Review17 remains immutable; only a
  zero-finding Independent Review18 may reach
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
