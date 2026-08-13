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

Local calibration Attempt14 is the first complete Build05-bound representative
sample: 96/96 raw runs passed 96/96 independent replays. Fixed-work semantic
and objective parity are exact. RCPSP reaches 2.12x representative four-worker
geometric-mean speedup, but EVRPTW reaches only 1.24x because the 5-customer
case does not amortize worker scheduling. The formal cloud matrix is therefore
blocked even though the provisional 32-core runtime estimate fits the ten-day
window. This calibration is not Level 1 evidence and does not open a holdout.

Build Attempt06 scoped default pytest discovery to the active TxnOpt suite and
retained independent formal
review Attempt01. It passes 102 tests and the same build, sanitizer, resource,
legacy, and historical-path gates. Attempt14 remains bound to Build05 and is
not relabelled as Build06 evidence. Build06 is not Level 1 ready.

Build Attempt07 is the current clean internal artifact. It binds the corrected
aggregate TLA+/PlusCal model, T3 conditional meta-theorem, runtime-audited T4
unit bounds, and 111 active tests including property-generated cache, budget,
duplicate-key, late-result, and transaction-window cases. Build, wheel,
sanitizer, 100,000-round resource, formal replay, legacy, and protected-path
gates pass. The correction is only `READY_FOR_INDEPENDENT_REVIEW`; measured
finite `Cmax`, the EVRPTW speedup gate, and the Level 1 matrix remain open.

The first 100,000-round resource soak exposed an unbounded task-receipt queue
and failed before its original harness could write a receipt. The exact error
and that evidence limitation are retained in
`manifests/txnopt_resource_soak_attempt01_failure.json`; a passing rerun must
use a new attempt identity.

Attempt02 drains physical task receipts at each native round boundary. Its
100,000-round clean-source soak passed with zero thread/FD delta, 8 KiB RSS
growth, and zero fallback.
