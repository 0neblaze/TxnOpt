# TxnOpt experiment index

| Protocol | Status | Scope | Raw evidence | Independent review |
| --- | --- | --- | --- | --- |
| `txnopt-level1-protocol-v1` | build complete, experiment not started | 12 EVRPTW + 24 RCPSP, 10 seeds | build manifest only | pending |
| Level 2 | gated | unopened | none | none |
| Level 3 | gated | unopened | none | none |

`level1-protocol.json` and its SHA-256 sidecar are the immutable case-set
identity for new Level 1 runs. Creating the protocol does not authorize cloud
purchase, benchmark launch, holdout access, or a readiness claim.

The prior internal build identity is recorded in
`manifests/txnopt_level1_build_attempt02.json`. Its status is deliberately
`BUILD_AND_LOCAL_RESOURCE_GATES_COMPLETE_NOT_LEVEL1_READY`: build and soak
success are not experiment readiness. Attempt01 remains immutable.

The current clean internal producer is
`manifests/txnopt_level1_build_attempt03.json`. It adds exact producer-identity
binding for campaign configs and raw manifests, plus the corrected EVRPTW
initialization and post-screen admission boundary. Its status remains
`BUILD_AND_LOCAL_RESOURCE_GATES_COMPLETE_NOT_LEVEL1_READY`; the recorded local
calibration points are exploratory and do not satisfy the two-domain speedup or
confidence-interval gates.

Build Attempt04 supersedes Attempt03 for new runs after local calibration
Attempt11 exposed an RCPSP deadline-classification defect. Attempt04 maps
deadline-limited CP-SAT `UNKNOWN`/`FEASIBLE` outcomes to transactional timeout,
so the runtime returns the last committed prefix. The build gates pass, but the
Level 1 experiment and performance gates remain pending.

Build Attempt05 supersedes Attempt04 for new runs. It adds a bounded solve-local
LRU for repeated EVRPTW safe route screening and exports cache statistics only
through the physical trace, so the semantic digest remains independent of this
performance optimization. Its build, sanitizer, and 100,000-round resource
gates pass; Build05-bound calibration and all Level 1 experiment gates remain
pending.

The first 100,000-round resource soak exposed an unbounded task-receipt queue
and failed before its original harness could write a receipt. The exact error
and that evidence limitation are retained in
`manifests/txnopt_resource_soak_attempt01_failure.json`; a passing rerun must
use a new attempt identity.

Attempt02 drains physical task receipts at each native round boundary. Its
100,000-round clean-source soak passed with zero thread/FD delta, 8 KiB RSS
growth, and zero fallback.
