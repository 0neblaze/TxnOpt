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
- CLI: `txnopt run`, `txnopt verify`, `txnopt replay`, `txnopt env`, and
  `txnopt legacy verify`.

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
- [ ] Migrate EVRPTW model, parser, objective, validator, charging, and
  neighborhoods behind `txnopt_cases.evrptw` without duplicating the formal
  vehicle-first objective.
- [ ] Split producer-only runners from independent raw-only reviewers in
  `txnopt_evidence` and eliminate mutual imports.
- [x] Freeze the old repository instructions under tracked legacy governance,
  then replace the root instructions with current TxnOpt rules.
- [ ] Switch distribution, wheel contents, CMake project, native module, CLI,
  documentation, CI, tests, tools, and experiment entrypoints together. A
  partially renamed wheel is not a Level 1 artifact.

### Weeks 3-4: Python runtime and EVRPTW adapter

- [ ] Implement a single live `TxnRuntime` state owner around a deterministic
  random tape and canonical candidate order.
- [ ] Centralize budget reservation/settlement, cache staging, commit, rollback,
  and trace emission in the runtime.
- [ ] Implement serial, barrier, and ordered transaction modes against the same
  contract.
- [ ] Keep EVRPTW objective and validation exclusively in the case adapter and
  verify them differentially against the frozen implementation.

### Weeks 4-5: new native round ABI

- [ ] Freeze every `stage05.2-*` ABI and introduce a separate
  `txnopt-native-round-v1` source identity.
- [ ] Move only the active generic transaction, budget, cache, concurrency, and
  trace path into `cpp/txnopt_core`.
- [ ] Keep EVRPTW packing, screening, and exact charging in the case-native
  adapter.
- [ ] Require one contiguous SoA request and one typed receipt per round, with
  identical fixed-work semantic digests for Python and native execution.
- [ ] Do not implement a Level 1 multi-round full-native path.

### Weeks 5-6: RCPSP and formal model

- [ ] Represent RCPSP state as a precedence-feasible activity order plus mode
  vector.
- [ ] Implement safe precedence/resource lower-bound screening, block
  remove/reinsert, and mode change.
- [ ] Use fixed-seed single-thread CP-SAT exact repair and reconstruct every
  schedule in an independent validator; the objective is feasible makespan.
- [ ] Complete T1/T2 proof and model checking. Produce independently reviewable
  T3/T4 proofs. Close the positive Level 2 entry if T3 fails or T4 is vacuous at
  measured parameters.

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
