# TxnOpt experiment index

| Protocol | Status | Scope | Raw evidence | Independent review |
| --- | --- | --- | --- | --- |
| `txnopt-level1-protocol-v1` | Build11 local build/semantic/performance/bounded-exhaustive-fault/runtime gates complete; pre-cloud Attempt05 blocked; purchase and formal matrix not authorized | 12 EVRPTW + 24 RCPSP, 10 seeds | Build11 Attempt22 calibration and fault Attempt03 retained; Attempt23 raw root absent/unexecuted | Build11 evidence-lifecycle and formal-successor review pending |
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

Build Attempt07 binds the corrected
aggregate TLA+/PlusCal model, T3 conditional meta-theorem, runtime-audited T4
unit bounds, and 111 active tests including property-generated cache, budget,
duplicate-key, late-result, and transaction-window cases. Build, wheel,
sanitizer, 100,000-round resource, formal replay, legacy, and protected-path
gates pass. The correction is only `READY_FOR_INDEPENDENT_REVIEW`; measured
finite `Cmax`, the EVRPTW speedup gate, and the Level 1 matrix remain open.

Build Attempt08 was the clean internal artifact for its source. It adds a bounded,
solve-local cache for independently reconstructed EVRPTW route reports. The
cache rejects cross-instance reuse, keeps solution-level customer coverage
outside the cache, and changes only physical statistics. Its 117 tests, wheel,
sanitizers, formal replay, legacy verifier, protected-path check, and
100,000-round soak pass. An unbound 12-case developer diagnostic measured only
1.01x EVRPTW four-worker geometric-mean fixed-work speedup, so it does not
replace Attempt14 and the cloud matrix remains blocked.

Build08-bound local calibration Attempt16 supersedes that unbound diagnostic:
96/96 raw runs passed 96/96 independent replays with zero fallback and exact
fixed-work semantic/objective parity. RCPSP reaches 2.11x representative
four-worker geometric-mean speedup; EVRPTW reaches only 1.19x. Consequently
Attempt16 is retained as `PERFORMANCE_GATE_NOT_READY`, and neither cloud rental
nor the Level 1 matrix is authorized.

Independent formal successor review Attempt03 accepts the corrected atomic
window and aggregate model granularity, but remains `NEEDS_WORK`: T3 is still a
textual meta-theorem, generic contract-error paths lack uniform waste receipts,
the refinement relation is not mechanically checked, and finite measured
`Cmax` is absent. The Level 1 formal gate and Level 2 positive entry stay
closed.

Correction Attempt04 and independent Review Attempt05 preserve Attempt03 and
close those formal findings for the current package. T3 is accepted only as a
scheduler-relative conditional theorem; the reviewer independently recomputes
T4 unit and measured per-run Cmax bounds; aggregate replay enforces one active
transaction, ordered candidate-key identity, and atomic state/cache visibility.
The formal package has zero unresolved critical findings, but Level 1 remains
not ready because the complete legacy compatibility run timed out and no clean
build-bound successor calibration or confidence interval exists. A dirty
developer probe reached 1.33x representative EVRPTW speedup and is retained only
as a diagnostic, not as evidence or cloud authorization.

Build Attempt09 is the first clean successor that binds those runtime,
refinement, reviewer, and adaptive native-scheduling changes to an internal
`txnopt 0.1.0a1` wheel. Wheel-installed tests, formal replay, legacy-freeze
verification, ASan/UBSan, TSan, and the 100,000-round resource soak pass. It is
still `NOT_LEVEL1_READY`: local calibration Attempt17 now binds 96/96 raw runs
and independent replays to Build09, with zero fallback and exact fixed-work
semantic/objective parity. Representative EVRPTW and RCPSP four-worker
geometric-mean speedups are 1.35x and 2.13x, so both local performance gates
pass. Complete compatibility, full-scope Cmax/CI evidence, and the
preregistered Level 1 matrix remain open. A Build09-bound conservative
32-physical-core estimate predicts about 4,321 seconds for the matrix, but no
cloud purchase is authorized.

Legacy compatibility Attempt01 resolves the earlier 1,200-second timeout. The
exact frozen wheel hash was reconstructed from the freeze tag without payload
changes, legacy package/CLI/native ABI smoke checks pass, and 61/62 isolated
test files exit zero. The sole retained failing assertion is the declared
four-file mismatch against an older immutable source-disposition receipt; all
other 1,935 collected outcomes pass or skip as specified. The compatibility
decision is complete without rewriting historical provenance.

Formal campaign Attempt18 is a new Build09-bound identity, not the Attempt17
calibration plan. Its 2,880 configs and config tree are frozen under
`/home/oneblaze/txnopt-plans/level1-formal-plan-attempt18`; its dedicated raw
root does not exist. `level1-analysis-protocol-v1.json` fixes the paired
fixed-work speedup, 20,000-resample 95% bootstrap interval, one-worker overhead,
semantic/objective parity, and measured-Cmax analysis before execution. The
orchestration preflight passes structurally but cannot execute without a
separate signed purchase authorization. Full-scope CI and Cmax are completion
gates produced after the matrix, not circular preconditions for procurement.
The signed Build09 local decision is
`manifests/txnopt_level1_precloud_gate_attempt01.json`; it records
`READY_FOR_SEPARATE_PROCUREMENT_AUTHORIZATION` while keeping purchase,
execution, holdout access, Level 1 readiness, and public release false. The
post-Build09 native prepared-transaction change supersedes that procurement
gate for the active source; the receipt itself remains immutable.

Build Attempt10 is the current clean internal artifact. It binds the generic
prepared transaction in `cpp/txnopt_core`, complete-work reservation, prepared
cache-delta receipt, one-shot physical-trace drain, and independent native-round
receipt replay to source revision `3e18b99`. Its installed-wheel tests (174),
five property suites (600 generated examples), Ruff, strict mypy, wheel RECORD,
legacy verifier, ASan/UBSan, TSan, protected-history comparison, and 100,000-round
resource soak pass with zero fallback. The signed Attempt04/05 formal package is
retained only for its prior source: the additive Build10 refinement note is
`REVIEW_PENDING_BUILD10`. Build10-bound local calibration Attempt20 now retains
96/96 raw runs and 96/96 independent reviews with exact fixed-work semantic and
objective parity. Representative EVRPTW and RCPSP four-worker geometric-mean
speedups are 1.32x and 2.12x, while maximum one-worker overhead is 0.46% and
1.84%. A conservative 32-physical-core estimate predicts about 4,243 seconds
for the matrix. These are representative calibration results only: no
full-scope confidence interval exists, the successor review remains pending,
and purchase, execution, holdout access, and Level 1 readiness are false.

Formal plan Attempt21 is the fresh post-calibration Build10 identity. It fixes
2,880 configs under config tree
`7ecceafc45289f16fa3a5ec7d455fff22b914a766c50f113bde8221da4760d2a`
and preregisters the exact paired bootstrap and T4/Cmax gates. Its structural
preflight is `PASS_NOT_AUTHORIZED_TO_EXECUTE`; the raw root is absent. Pre-cloud
Attempt02 deliberately records `BLOCKED_FORMAL_SUCCESSOR_REVIEW_PENDING`, so it
authorizes neither procurement nor execution.

Build Attempt11 is the current clean internal producer. It adds the
hash-chained `txnopt-evidence-lifecycle-v1` contract without changing the
semantic or physical trace protocols: new raw/failure bundles are v2 and bind
`PLANNED -> RUNNING -> SEALED`; independent replay appends `REVIEWED` using
the exact raw-manifest digest. Build10 v1 bundles remain readable and
unchanged. The Build11 wheel passes 186 installed tests, five property suites
(600 generated examples), Ruff, strict mypy, wheel RECORD verification,
legacy verification, ASan/UBSan, TSan, protected-history comparison, and a
100,000-round resource soak with zero fallback.

Build11 local calibration Attempt22 retains 96/96 raw v2 bundles and 96/96
independent v2 reviews. Every lifecycle, raw/review hash, producer identity,
positive observed Cmax, fixed-work semantic digest, and objective replays
exactly. Representative EVRPTW and RCPSP four-worker geometric-mean speedups
are 1.35x and 2.12x; maximum one-worker overhead is 8.08% and 0.70%. These are
representative pre-cloud results only; EVRPTW has a 0.91x worst pair and no
full-scope confidence interval has been computed.

Formal plan Attempt23 is the fresh unexecuted Build11 identity. Its 2,880
configs are fixed by tree
`4a828f134d971172cd6ebdef15d45e14b8315be9dc7cf51e96ea00856d27e901`,
the preregistered analysis remains unchanged, its raw root and atomic claim are
absent, and clean preflight is `PASS_NOT_AUTHORIZED_TO_EXECUTE`. A conservative
32-physical-core estimate is about 4,300 seconds. Local fault Attempt01 is an
immutable orchestration failure that started no tests; Attempt02 passes 24
fault categories and 25 exact Build11-wheel test cases with zero failures,
errors, or fallback. Attempt03 strengthens that scope with all six completion
orders for three candidates across barrier and ordered execution, every
candidate failure position, and both runtime deadline checkpoints: 50
auditor-owned microstate cases plus the 25 retained producer cases all pass.
Pre-cloud Attempt05 therefore records the complete local
build, semantic, representative performance, fault/prefix, and runtime gates,
but remains
`BLOCKED_BUILD11_INDEPENDENT_REVIEW_PENDING`; it authorizes neither procurement
nor execution.

The first 100,000-round resource soak exposed an unbounded task-receipt queue
and failed before its original harness could write a receipt. The exact error
and that evidence limitation are retained in
`manifests/txnopt_resource_soak_attempt01_failure.json`; a passing rerun must
use a new attempt identity.

Attempt02 drains physical task receipts at each native round boundary. Its
100,000-round clean-source soak passed with zero thread/FD delta, 8 KiB RSS
growth, and zero fallback.
