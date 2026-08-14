# TxnOpt Level 1 to Level 3 roadmap

Status: **Level 1 in progress**

Authority: this tracked document

Private appendix: `txnopt-research-strategy.local.md` (ignored)

Legacy input: `D:/UserData/Downloads/PLAN.md` (non-authoritative)

This roadmap governs an in-place transition of the existing Git history into
TxnOpt. It does not authorize a push, GitHub rename, PyPI or Zenodo release,
cloud purchase, public alpha, or manuscript submission.

## 1. Non-negotiable boundaries

1. Level 1, Level 2, and Level 3 are sequential mandatory phases. They may not
   be deleted, marked optional, or bypassed by changing a run label.
2. Historical run labels, Stage paths, manifests, schemas, hashes, lifecycle
   states, and the old C++ ABI are immutable. They are not moved, renamed,
   rewritten, or recomputed for visual consistency.
3. New TxnOpt evidence binds a new source, wheel, native binary, environment,
   protocol, and attempt identity. It never promotes incomplete Stage 5.2
   evidence.
4. A failed gate produces a new immutable failure attempt. Raw evidence is not
   overwritten or deleted.
5. Runtime fallback is forbidden. Unsupported modes and unavailable native
   functionality fail closed.

## 2. Current verified baseline

- Legacy source freeze commit: `3b0cf371759f3465c7264b85894d090004f3cf43`.
- Legacy freeze tag: `stage052-legacy-freeze-v1`.
- The label deliberately says `freeze`, not `correctness`: the historical
  source-disposition receipt still binds an older source snapshot and exposes
  four expected mismatches with the active legacy implementation.
- Stage 5.2 Attempt16 is a retained, closed, unique failure. Its raw failure
  remains unchanged; a signed adjudication identifies missing terminal
  process-tree I/O accounting for short-lived workers.
- The broad novelty claim "problem-independent speculative parallel search that
  preserves a serial decision sequence" is covered by prior work. TxnOpt must
  therefore be evaluated as an auditable runtime/refinement contribution with
  deadline/failure prefix safety, cross-domain adapters, and non-vacuous waste
  bounds, not as the invention of speculative simulated annealing or ordered
  transactions.
- Clean Build Attempt08 passes Ruff, strict mypy, 117 TxnOpt tests, wheel/CLI
  verification, ASan/UBSan, TSan, historical-path preservation, and a
  100,000-round resource soak with zero fallback.
- Build08 adds a bounded solve-local EVRPTW route-report cache with exact
  cached/uncached differential tests, LRU eviction, solution-level coverage
  replay, and fail-closed cross-instance ownership.
- Build08-bound local calibration Attempt16 completes 96/96 raw runs and 96/96
  independent replays with zero fallback and exact fixed-work semantic and
  objective parity. RCPSP reaches 2.11x representative four-worker geometric
  mean; EVRPTW reaches only 1.19x, so the 1.30x gate remains failed and cloud
  rental remains unauthorized.
- Build09-bound local calibration Attempt17 supersedes Attempt16 for the
  Build09 producer without rewriting it. All 96 raw runs pass independent replay with
  zero fallback and exact fixed-work semantic/objective parity. Representative
  four-worker geometric-mean speedup is 1.35x for EVRPTW and 2.13x for RCPSP;
  maximum one-worker overhead is 2.68%. These local gates pass, but Attempt17
  is not full Level 1 evidence and does not authorize cloud procurement.
- Clean Build10 freezes the post-Build09 source that deepens `cpp/txnopt_core` with a prepared native
  phase machine, complete-work reservation, atomic cache delta, and semantic
  phase receipt. Its build, installed-wheel, sanitizer, protected-history, and
  100,000-round resource gates pass. Build09, Attempt17, and formal-plan
  Attempt18 remain immutable evidence but are not successor evidence for this
  implementation; Build10's additive refinement review remains pending.
- Clean Build11 freezes the current producer and adds the hash-chained
  `txnopt-evidence-lifecycle-v1` contract. New v2 raw/failure bundles bind
  `PLANNED -> RUNNING -> SEALED`; independent replay appends `REVIEWED` while
  Build10 v1 bundles remain readable and unchanged. Build11 passes its wheel,
  Ruff, strict mypy, property, sanitizer, protected-history, and 100,000-round
  resource gates with zero fallback.
- Build11-bound local calibration Attempt22 completes 96/96 raw runs and 96/96
  independent reviews with exact lifecycle, semantic, objective, and positive
  measured-Cmax replay. Representative four-worker geometric-mean speedups are
  1.35x for EVRPTW and 2.12x for RCPSP; maximum one-worker overheads are 8.08%
  and 0.70%.
- Build11 local fault Attempt01 is retained as an orchestration failure that
  started no tests. Corrected Attempt02 executes 24 fault categories as 25
  exact producer-test cases against the installed Build11 wheel. Completion
  order, deadline/worker prefix safety, budget/cache/snapshot atomicity, late
  rollback, unknown commit outcome, and T4 waste gates pass with zero fallback.
- Attempt03 retains Attempt02 and adds 50 auditor-owned bounded-exhaustive
  microstate cases against the same installed Build11 wheel: all six
  three-candidate completion orders across barrier and ordered execution, all
  three worker-failure positions crossed with those orders, and both runtime
  deadline checkpoints. All 75 combined cases pass; this is bounded runtime
  evidence, not a general unbounded proof.
- Review-packet verification Attempt08 makes the Build11 successor request
  executable without self-approving it. It verifies the unchanged 16 bound
  inputs, installed Build11 wheel, sealed 10-input formal receipt,
  producer/reviewer import independence, and 46/46 lifecycle/refinement tests.
  Its status remains `MACHINE_VERIFIED_EXTERNAL_DECISION_PENDING`; a separately
  identified external reviewer must still issue the allowed PASS or
  `NEEDS_WORK` decision.
- Static gate Attempt01 binds the unchanged Build11 implementation and wheel to
  24/24 producer-owned contract, dependency, distribution, and CLI tests. It
  records entrypoint coverage 1, zero core import cycles, zero reverse
  dependencies, zero active `evrptw` imports or Stage 5.2 schemas, no Level 1
  full-native fast path, and zero protected-history changes.
- Build05-bound local calibration Attempt14 completes 96/96 raw runs and 96/96
  independent replays. Fixed-work semantic/objective parity is 100%. Its
  representative RCPSP four-worker geometric-mean speedup is 2.12x, while
  EVRPTW is 1.24x and therefore below the 1.30x Level 1 gate.
- The 32-physical-core estimate for the provisional 1200-work/3-second matrix
  is about 4,387 seconds (1.22 hours), but this does not authorize purchase:
  the EVRPTW performance gate and final budget freeze remain open.
- The Build09-bound successor estimate uses Attempt17 p95 observations and
  predicts about 4,321 seconds (1.20 hours) on 32 physical cores at 80%
  scheduler efficiency. The ten-day runtime gate passes provisionally, while
  procurement remains separately unauthorized.
- Legacy compatibility Attempt01 replaces the previous undiagnosed timeout with
  a complete decision. The exact frozen wheel hash is reconstructed from the
  freeze tag with unchanged payloads; package, CLI, and native ABI smoke checks
  pass. File-isolated testing completes in under five minutes. Its sole retained
  failure is the four-file mismatch against the older immutable
  source-disposition receipt already declared by the freeze manifest; the
  receipt remains unchanged rather than being falsified.
- Independent T3/T4 review Attempt01 reports `NEEDS_WORK`. TLC continues to
  support bounded safety only. A successor correction now uses the complete
  uncommitted atomic transaction as T4's `W`, emits receipt-bound unit audits,
  formalizes the conditional T3 meta-theorem, and aligns the checked model with
  aggregate runtime publication. It remains `READY_FOR_INDEPENDENT_REVIEW`, not
  accepted evidence; measured finite `Cmax` is still required.
- Independent successor review Attempt03 remains immutable `NEEDS_WORK`.
  Correction Attempt04 and independent Review Attempt05 close its formal
  findings: T3 is an explicitly scheduler-relative conditional theorem, every
  active transaction error has fail-fast rollback/audit evidence, the raw-only
  reviewer independently recomputes T4 and measured per-run `Cmax`, and the
  aggregate refinement replay rejects interleaved owners, candidate-set drift,
  hidden state publication, and invalid round termination. This passes the
  current formal package only; it does not prove a general-domain T3 theorem,
  complete legacy compatibility, or Level 1 readiness.
- A dirty development probe of the adaptive native scheduling policy reached
  1.33x representative EVRPTW four-worker geometric-mean fixed-work speedup
  with exact digest parity. It is diagnostic only: no clean producer identity,
  confidence interval, or independent performance review exists yet.

## 3. Target repository and dependency structure

```text
src/txnopt/                         generic transactional runtime
src/txnopt_cases/evrptw/            EVRPTW adapter
src/txnopt_cases/rcpsp/             RCPSP adapter
src/txnopt_evidence/                runner, reviewer, artifact, lifecycle
src/txnopt_legacy/                  read-only historical/provenance readers
cpp/txnopt_core/                    generic transaction/budget/cache/concurrency
cpp/txnopt_cases/evrptw/            EVRPTW native screening/exact kernels
formal/                             TLA+/PlusCal and refinement mapping
experiments/txnopt/                 new protocols and result index
legacy/                             frozen wheel/environment/ABI evidence index
```

The import graph is fixed:

```text
txnopt_cases -> txnopt
txnopt_evidence -> txnopt + txnopt_cases
txnopt_evidence historical replay -> txnopt_legacy
txnopt -> standard library only at the public contract boundary
```

`txnopt` must not import cases, evidence, or legacy. New core code must not
import Stage-numbered modules. Compatibility belongs in explicit adapters and
is removed from the active wheel before the Level 1 gate closes.

## 4. Product identity and public surface

The Level 1 internal artifact identity is:

- distribution: `txnopt`;
- version: `0.1.0a1`;
- CMake project: `txnopt_core`;
- native module: `txnopt._native`;
- protocols: `txnopt-contract-v1`, `txnopt-native-round-v1`,
  `txnopt-semantic-trace-v1`, and `txnopt-physical-trace-v1`;
- Public CLI: `txnopt run`, `txnopt verify`, `txnopt plan`, `txnopt preflight`,
  `txnopt review`, `txnopt archive inventory`, and the explicit
  `txnopt cloud tencent ...` namespace. Per-run replay is an internal
  fresh-process reviewer entrypoint rather than a public command.

The root package exports exactly five names:

1. `TxnRuntime`: sole owner of live state, budget, cache transactions, and
   commit/rollback.
2. `SearchKernel`: pure deterministic candidate proposal and decision logic;
   it cannot mutate runtime state.
3. `Oracle`: stable key, determinism declaration, safe screening, ordered batch
   evaluation, independent validation, and objective construction.
4. `RunConfig`: seed, workers, execution mode, fixed-work or deadline budget,
   speculation window, and trace policy.
5. `RunResult`: last committed state, objective, termination reason, semantic
   digest, physical artifact reference, and provenance.

Candidate transactions, budget ledger, cache transaction, random tape, and
semantic/physical event types are versioned internal contracts and are not
re-exported from the root package.

## 5. Runtime semantics

The candidate transaction state machine is:

```text
PREPARED -> RESERVED -> EVALUATING -> VALIDATED -> COMMITTED
      \          \             \             \-> ABORTED / INTERRUPTED
```

Only `TxnRuntime` may publish committed state or cache writes. Work is charged
when it starts. Each solve packs its immutable instance context once. Each
round crosses the Python/native seam once with contiguous SoA buffers and
returns one typed receipt. Late results, insufficient budget, deadline,
snapshot invalidation, worker failure, validation failure, and write failure
abort the transaction without acceptance or cache visibility.

Level 1 implements three canonical execution modes:

- serial one-worker reference;
- deterministic barrier execution;
- ordered transactional execution.

Unordered asynchronous execution is only a counterexample generator. A
multi-round full-native fast path is outside Level 1 and may enter Level 2 only
after proving equivalence to the canonical round stream.

## 6. Level 1: method-complete internal package (weeks 1-8)

### Week 1: freeze and direction lock

- [x] Inspect writers, processes, leases, source authority, and worktree.
- [x] Independently review, adjudicate, fully retain, close, and release the
  Attempt16 coordination lease without restarting old Formal work.
- [x] Run the legacy validation matrix on one exact implementation snapshot.
- [x] Create the honest `stage052-legacy-freeze-v1` boundary.
- [x] Run three focused novelty-retrieval rounds and stop broad retrieval after
  direct overlap is established.
- [x] Pivot the research claim away from inventing serial-equivalent speculative
  annealing or ordered transactions.

### Week 2: structure and naming

- [x] Establish the generic, case, evidence, and legacy Python namespaces and
  both C++ source roots. This is a structural boundary, not an active solver.
- [x] Migrate the EVRPTW model, parser, objective, validator, and exact-charging
  oracle behind `txnopt_cases.evrptw` without duplicating the formal
  vehicle-first objective.
- [x] Migrate the active EVRPTW neighborhood surface after its incremental,
  measurement, and native protocols have been separated from the old module.
- [x] Split producer-only runners from independent raw-only reviewers in
  `txnopt_evidence` and eliminate mutual imports.
- [x] Add an append-only, hash-chained evidence lifecycle. New producers seal
  `PLANNED -> RUNNING -> SEALED`, independent replay appends `REVIEWED`, and
  legacy v1 bundles remain readable without retroactive mutation.
- [x] Freeze the old repository instructions under tracked legacy governance,
  then replace the root instructions with current TxnOpt rules.
- [x] Switch distribution, wheel contents, CMake project, native module, CLI,
  documentation, CI, tests, tools, and experiment entrypoints together. A
  partially renamed wheel is not a Level 1 artifact.

### Weeks 3-4: Python runtime and EVRPTW adapter

- [x] Implement a single live `TxnRuntime` state owner around a deterministic
  random tape and canonical candidate order.
- [x] Centralize budget reservation/settlement, cache staging, commit, rollback,
  and trace emission in the runtime.
- [x] Implement serial, barrier, and ordered transaction modes against the same
  contract.
- [x] Keep EVRPTW objective and validation exclusively in the case adapter and
  verify them differentially against the frozen implementation.

### Weeks 4-5: new native round ABI

- [x] Freeze every `stage05.2-*` ABI and introduce a separate
  `txnopt-native-round-v1` source identity.
- [x] Move the active prepared transaction, reservation, cache-delta, and trace
  path into the deep `cpp/txnopt_core` round module. It reuses the frozen
  concurrency and exact-kernel headers behind a new module identity while
  keeping solve-wide budget and publishable cache ownership in Python
  `TxnRuntime`.
- [x] Keep EVRPTW packing, safe screening, and exact charging in the
  case-native adapter.
- [x] Require one contiguous SoA request and one typed receipt per round, with
  identical fixed-work semantic digests for Python and native execution.
- [x] Do not implement a Level 1 multi-round full-native path.

### Weeks 5-6: RCPSP and formal model

- [x] Represent RCPSP state as a precedence-feasible activity order plus mode
  vector.
- [x] Implement safe precedence/resource lower-bound screening, block
  remove/reinsert, and mode change.
- [x] Use fixed-seed single-thread CP-SAT exact repair and reconstruct every
  schedule in an independent validator; the objective is feasible makespan.
- [x] Complete bounded T1/T2 proof and model checking.
- [x] Produce independently reviewable T3/T4 proofs. Close the positive Level 2
  entry if T3 fails or T4 is vacuous at measured parameters.

The formal obligations are:

- T1: fixed-work observational equivalence for any physical completion order;
- T2: deadline or worker failure returns only the last complete commit prefix;
- T3: schedule independence, zero speculative waste, and fully asynchronous
  progress cannot all be unconditional for arbitrary state-dependent search;
- T4: discarded cost is at most
  `min(Bremaining, W * Qmax) * Cmax`, while post-boundary in-flight cost is at
  most `min(P, W * Qmax) * Cmax`.

### Weeks 7-8: bounded cloud window

The fixed Level 1 matrix is 12 EVRPTW and 24 RCPSP instances, ten seeds each:

- pilot: 8 EVRPTW + 16 RCPSP;
- validation: 4 EVRPTW + 8 RCPSP;
- formal axes: serial 1-worker, TxnOpt 1/4-worker, barrier 4-worker;
- exploratory only: 8-worker representative cases;
- both fixed-work and fixed-time;
- fault injection concentrated on exhaustive microstates and representative
  instances;
- no Level 2 or Level 3 holdout access.

Cloud procurement is blocked until local semantic, fault, build, and runtime
estimation gates pass. The formal environment is exclusive Linux with at least
32 physical cores and 128 GB RAM. The run window is at most 14 consecutive
days; the predicted formal matrix must fit in ten days with four days reserved
for failed reruns. Sixty-four cores are used only if total cost is lower.

Current gate status: **Build11 local build, semantic, representative
performance, fault/prefix, and runtime-estimate gates are complete; the Build11
review packet is machine-verified, but the external independent
evidence-lifecycle/formal-successor decision remains pending; purchase and
execution are unauthorized**. Attempt22 is the current local calibration,
Attempt03 is the current local fault receipt, static Attempt01 closes the local
identity/import/naming predicates, and review-packet Attempt08 is machine-only
evidence. Formal plan Attempt23 freezes 2,880 Build11-bound configs
with a fresh absent raw root and passes structural preflight. Pre-cloud Attempt06
supersedes Attempt05 and remains
`BLOCKED_BUILD11_INDEPENDENT_REVIEW_PENDING`; no procurement or execution
authority is implied, and full-scope confidence intervals remain post-run
Level 1 completion evidence.

Completion audit Attempt01 classifies all 16 Level 1 predicates. Static and
local runtime predicates pass; representative semantic, quality, overhead,
speedup, replay, fallback, and waste results remain explicitly scoped to local
evidence. The two completion predicates that cannot yet be proven are the
full-scope 95% confidence-interval bounds and zero unresolved Build11 findings.
They require, respectively, the unstarted formal matrix and a separately
identified external review decision. No further local implementation gap is
known, but this is not a Level 1 readiness claim.

The prior formal matrix is frozen separately as external plan Attempt18: 2,880 unique
configs, config tree
`c609613e00928708bbb69e5d22be417e99489cdf921d2f677759412ab16b8929`,
raw root `/home/oneblaze/txnopt-results/level1-formal-build09-attempt18`, and
plan SHA-256
`85e6e6471fe4480c8a05e9887e6166e5d934ea557d53ce0a33fe098529947c8f`.
The raw root is absent. Attempt18 remains unexecuted and is not the active-source
formal identity after the native transaction change.
`experiments/txnopt/level1-analysis-protocol-v1.json`
preregisters the paired fixed-work geometric mean, deterministic 20,000-sample
percentile bootstrap, 95% confidence interval, one-worker overhead, exact
semantic/objective parity, and measured `Cmax` gates before any formal run.

Full-scope confidence intervals, measured full-scope `Cmax`, and the final
performance predicates are outputs of the formal matrix and therefore remain
post-procurement Level 1 completion gates; requiring them before procurement
would be circular. Actual execution remains fail-closed until a clean successor
plan exists and the user supplies a separate signed authorization bound to that
exact plan, preregistered analysis digest, config tree, build manifest and
wheel, fresh raw root, exclusive-Linux resource contract, and 14-day maximum
window. The campaign runner then creates an atomic launch claim, rejects an
active target process, verifies the provisioned host and exact installed
payload, and terminates the complete process group on timeout. It produces raw
evidence only and records no readiness decision. A separate process and tool
must replay all raw manifests, independently recompute every present T4 waste
bound, and reconcile every prefix/refinement receipt before any aggregate gate
or internal-seal decision is issued.

### Level 1 gate

Every predicate must be true:

```text
authority_clean
entrypoint_coverage == 1
core_import_cycles == 0
txnopt_reverse_dependencies == 0
fixed_work_semantic_digest_match == 1
deadline_fault_prefix_safety == 1
validator_objective_cache_budget_raw_replay == 1
fallback_count == 0
historical_path_and_byte_mutations == 0
single_worker_overhead <= 0.15
evrptw_4_worker_geomean_speedup >= 1.30
rcpsp_4_worker_geomean_speedup >= 1.30
both_95ci_lower_bounds > 1.05
fixed_work_quality_regressions == 0
measured_waste <= T4_bound
unresolved_critical_findings == 0
```

The output is an internal `txnopt 0.1.0a1` wheel, source archive, formal model,
raw evidence, and independent review. It is not published.

## 7. Level 2: submission-line complete (cumulative weeks 10-12)

Level 2 is mandatory after Level 1 and cannot start early. Freeze
`txnopt-contract-v1` and a Level 2 preregistration. Use an independent 30
EVRPTW + 60 RCPSP instance set, 20 seeds, formal 1/2/4/8-worker axes, and an
exploratory 16-worker axis. Complete major ablations, a fully independent
rerun, a public alpha candidate, reproduction instructions, and a preprint
draft.

Publication claims require all of the following:

- 100% raw-to-table traceability;
- every formal review is `READY`;
- the independent rerun passes;
- zero legacy imports on the active path;
- single-worker overhead at most 10%;
- both-domain 4-worker geometric mean speedup at least 1.5x with 95% CI lower
  bound above 1.2x;
- no unexplained systematic negative scaling at 8 workers;
- independent two-person review of T1-T4;
- no incomplete or `NOT_READY` Stage 5.2 result described as complete.

Failure preserves Level 1 but blocks Level 3 and all confirmatory claims.

## 8. Level 3: full protocol coverage (cumulative months 5-7)

Level 3 starts only after Level 2 passes and implementation, parameters, and
statistics are frozen. It executes the full matrix: 60 Homberger-1000 EVRPTW,
120 PSPLIB J120, 30 seeds, five controls, 1/2/4/8/16 workers, fixed-work,
fixed-time, fault injection, and scaling.

It additionally requires:

- 100% state-transition, terminal-failure, schema
  v1/v2/legacy/TxnOpt, recovery, retention, and compaction coverage;
- Python/native abstract-trajectory equivalence;
- complete historical replay;
- zero identity, objective, or trace mismatch;
- one-time final holdout access with no retuning-and-reuse;
- source, container, TLA+, per-run data, independent validators, DOI archive,
  and submission package.

Any semantic or key-parameter change after Level 2 increments the contract
version, downgrades the old Level 2 result to exploratory, and requires a new
uncontaminated confirmation set. Push, GitHub rename, Zenodo, and submission
remain separately authorized external actions.

## 9. Required verification

- Exhaust completion orders, deadlines, and failure points on microstates.
- Property-test duplicate keys, cache conflicts, insufficient budget, late
  results, and invalid snapshots.
- Differentially test Python and C++ round transactions.
- Replay frozen EVRPTW objective/validator cases.
- Rebuild RCPSP schedules in an independent validator.
- Enforce producer-only raw output and reviewer-only independent reconstruction.
- Record exact exit codes for Release, Ruff, strict mypy, pytest, ASan, TSan,
  and long resource-leak tests.
- Permit `evrptw` only in `txnopt_cases.evrptw`, legacy, historical evidence,
  literature, and data licences at the final naming gate.

This document records required work and verified status. An unchecked item is
not implemented, and a checked implementation item is not evidence-ready until
its independent gate passes.
