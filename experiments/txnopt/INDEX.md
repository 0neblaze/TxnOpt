# TxnOpt experiment index

| Protocol | Status | Scope | Raw evidence | Independent review |
| --- | --- | --- | --- | --- |
| `txnopt-level1-protocol-v2` | Build19–Build21 and failed Reviews13–15 are immutable predecessors; Build22 is a retained build failure; Build23 is the clean producer for Attempts38/39 and Pre-cloud12 | 12 EVRPTW + 24 RCPSP, 10 seeds; minimum 64 physical cores and provider-reported 128 GB | Attempt35 retains 96/96 rejected-predecessor raw/review bundles; Attempt34 remains unexecuted; successor evidence must use Attempts38/39 | Review16 is the next authoritative adjudication; zero findings are required before account input |
| `txnopt-level1-protocol-v1` | Immutable Build14 / Attempt24 predecessor | 12 EVRPTW + 24 RCPSP, 10 seeds | Attempt25 retains 96/96 representative raw/review bundles; Attempt24 remains unexecuted | Attempt11 and pre-cloud Attempt07 remain immutable prior decisions and do not authorize Build18 |
| Level 2 | gated | unopened | none | none |
| Level 3 | gated | unopened | none | none |

`level1-protocol-v2.json` and its SHA-256 sidecar are the active case-set and
Tencent-resource identity for new Level 1 runs. The unversioned
`level1-protocol.json` is the immutable protocol-v1 predecessor. Creating or
materializing either protocol does not authorize cloud purchase, benchmark
launch, holdout access, or a readiness claim.

The retained Build19 Tencent chain uses `level1-protocol-v2.json`. External
formal plan Attempt30 binds 2,880 configs,
2,880 pre-run expected identities, a 64-physical-core topology, provider-
reported memory of at least 128 GB, no default region, and an absent formal raw
root. Local calibration Attempt31 binds 96 raw bundles and 96 independent
reviews; all 12 fixed-work semantic/objective groups match, all exact-work Cmax
and native receipt checks pass, and the 51-token capacity estimate is about
2,059 seconds. The estimate is representative local evidence, not live 64-core
scaling proof. The offline Build19 Tencent deployment Attempt02 passes fresh
installation and fixture checks without containing credentials or a region.
Pre-cloud Attempt08 recorded an account-ready candidate but Independent Review
Attempt12 subsequently found four major gaps, so Attempt08 cannot authorize or
support a final account-ready claim. Build19 fixed those four findings, but
Review13 found three further major gaps and rejected Pre-cloud09. Build20 with
Attempt32/33, offline deployment Attempt03, and producer-side Pre-cloud10 is
also retained: Review14 found one remaining CVM credential-acquisition exception
leak. Attempt33
retains 96 raw bundles and 96 independent reviews, all 12 fixed-work parity
groups match, and its 51-token capacity estimate is about 2,045 seconds.
Attempt32 remains unexecuted with no launch claim or raw root. Build21 passes
363 installed-wheel tests, five fixed-seed property suites, both sanitizer
smokes, the 100,000-round soak, and live TLC. Attempt34 binds 2,880 configs and
2,880 expected identities and remains unexecuted; Attempt35 retains 96 raw
bundles and 96 independent reviews, all 12 fixed-work parity groups match, and
its 51-token estimate is about 2,024 seconds. Deployment Attempt04 passes fresh
offline installation and replay, while the local 12-core host is correctly
rejected. Pre-cloud11 is the producer-side candidate, but Review15 rejects it
because the safe outer SDK error retains the credential-bearing original via
`__context__`. Build22 then failed before producing a wheel because its offline
build bound an empty UV cache. Build23 closes that finding by detaching the
public CVM/COS error from both `__cause__` and `__context__`; its 368
installed-wheel tests, five property suites, sanitizers, 100,000-round soak,
and live TLC pass. The remaining successor chain is Attempt38/39,
Deployment05, Pre-cloud12, and Review16. None of these
artifacts authorizes an API call, purchase, instance, COS bucket, formal matrix,
holdout, Level 1 claim, or release.

Build13 is retained as a complete but rejected internal artifact. Independent
Review Attempt10 found one major gap: interrupted or aborted native exact work
could omit its native receipt. Build14 supersedes it for new planning, requires
native receipt work ledgers to cover every committed, aborted, interrupted, or
T4-positive work path, and rejects missing, zero-work, and multi-stream
under-covered receipts. Build14's wheel-installed 287-test suite, five property
suites, Ruff, strict mypy, wheel RECORD, legacy verifier, ASan/UBSan, TSan,
protected-history comparison, and 100,000-round resource soak pass. Independent
Review Attempt11 reports zero critical, major, or minor findings while keeping
Level 1, purchase, cloud, formal execution, publication, and Level 2 closed.

External formal plan-v2 Attempt24 binds Build14, 2,880 canonical configs, and
2,880 pre-run expected identities through separate config and identity tree
digests. Independent full-plan review found no findings. Its preflight is
`PASS_NOT_AUTHORIZED_TO_EXECUTE`; the raw root, atomic launch claim, and target
processes are absent.

Build14 local calibration Attempt25 retains 96/96 anchored v3 raw bundles and
96/96 independent reviews with zero fallback, positive measured Cmax, and exact
fixed-work semantic/objective parity. The representative four-worker geometric
means are 1.3741x for EVRPTW and 2.1841x for RCPSP; maximum one-worker overheads
are 5.69% and 1.86%. Independent audit reports zero findings. The 32-physical-
core estimate is about 4,092 seconds, but it remains provisional local evidence.
Pre-cloud Attempt07 is `READY_FOR_SEPARATE_PROCUREMENT_AUTHORIZATION_NOT_AUTHORIZED`:
no purchase, formal execution, holdout access, Level 1 seal, push, or release has
been authorized.

The prior internal build identity is recorded in
`manifests/txnopt_level1_build_attempt02.json`. Its status is deliberately
`BUILD_AND_LOCAL_RESOURCE_GATES_COMPLETE_NOT_LEVEL1_READY`: build and soak
success are not experiment readiness. Attempt01 remains immutable.

Historical Build Attempt03 was the clean internal producer for its source:
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

Historical Build Attempt10 was the clean internal artifact for its source. It binds the generic
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

Historical Build Attempt11 was the clean internal producer for its source. It adds the
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

Build11 review-packet verification Attempt08 binds the unchanged Attempt07
request, all 16 source/artifact inputs, the installed Build11 wheel, the sealed
10-input formal receipt, producer/reviewer import independence, and 46/46
lifecycle/refinement tests. Its machine status is
`MACHINE_VERIFIED_EXTERNAL_DECISION_PENDING`: it does not supply a reviewer
identity or decision. Attempt08 did not unblock the then-current pre-cloud
Attempt05; successor Attempt06 remains blocked for the same external decision.

Independent review Attempt09 resolves that pending decision as `NEEDS_WORK`
with zero critical, three major, and zero minor findings. The v2 reviewer could
accept a fully re-signed producer identity, detached run label/input config, or
self-rebound result metadata because the bundle sidecar was treated as a trust
root. Build11 and Attempt23 remain immutable and unpromoted. The successor must
use a new clean build plus a pre-run expected identity that binds the exact
config bytes, producer source/tree/wheel/native identity, domain, execution
mode, and result contract before independent replay.

Build12 freezes the first anchored-v3 correction at revision `028d5e7`, but it
is retained as failed build evidence. Its wheel and isolated installation pass;
adversarial review then finds that a native run could delete every native-round
receipt and satisfy the empty receipt sequence vacuously. The build stops before
full installed-wheel, sanitizer, resource, or successor-review gates. Its exact
failure receipt is
`manifests/txnopt_level1_build_attempt12_failure.json`; no Build12 artifact may
authorize calibration, procurement, or execution.

The successor campaign contract is `txnopt-level1-campaign-plan-v2`. It binds
one pre-run expected-identity digest per config plus an ordered identity-tree
digest. Formal preflight, raw execution, and independent review consume those
frozen files; they may not reconstruct trust anchors from a completed raw
bundle. Launch, execution, per-run review, and final review receipts carry the
same binding. Plan v1 remains a non-formal compatibility reader only.

Static gate Attempt01 binds the unchanged Build11 implementation and installed
wheel to 24/24 producer-owned contract, dependency, distribution, and CLI
tests. It records entrypoint coverage 1, zero core import cycles, zero reverse
dependencies, zero active `evrptw` imports or new `stage05.2` schemas, no Level
1 full-native fast path, and zero protected-history changes. Pre-cloud Attempt06
adds this exact local evidence while retaining the external-review blocker and
all purchase/execution boundaries.

Completion audit Attempt01 maps every Level 1 predicate to its exact evidence.
It records all local implementation and pre-cloud gates complete, but keeps
Level 1 incomplete: full-scope confidence intervals and raw replay do not exist
because Attempt23 has not started, and Attempt09 proves that zero unresolved
Build11 findings is false. The only current implementation action is the
anchored evidence-identity correction followed by a new clean build and a new
independent review; procurement and formal execution remain unauthorized.

The first 100,000-round resource soak exposed an unbounded task-receipt queue
and failed before its original harness could write a receipt. The exact error
and that evidence limitation are retained in
`manifests/txnopt_resource_soak_attempt01_failure.json`; a passing rerun must
use a new attempt identity.

Attempt02 drains physical task receipts at each native round boundary. Its
100,000-round clean-source soak passed with zero thread/FD delta, 8 KiB RSS
growth, and zero fallback.
