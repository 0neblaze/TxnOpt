# TxnOpt repository instructions

These instructions apply to every path in this repository. The frozen
pre-TxnOpt policy is preserved byte-for-byte at
`legacy/governance/AGENTS.stage052.md` and by tag
`stage052-legacy-freeze-v1`. Do not copy its Stage-specific defaults into new
TxnOpt code.

## Repository identity and authority

- This repository is being transformed in place into **TxnOpt**. Do not create
  a replacement repository or discard Git history.
- The authoritative implementation checkout is the WSL ext4 path
  `/home/oneblaze/work/Reproducible-EVRPTW`. A Windows checkout or old worktree
  is not authoritative by proximity.
- The active roadmap is
  `docs/roadmap/txnopt-level1-to-level3-roadmap.md`. The ignored private
  appendix is `docs/roadmap/txnopt-research-strategy.local.md`.
- Level 1, Level 2, and Level 3 are mandatory sequential gates. Planned work is
  not implemented work, and implemented work is not evidence-ready work.
- Pushes, GitHub repository renames, PyPI/Zenodo releases, cloud purchases,
  holdout access, public alphas, and submissions require separate explicit
  authorization.

## Legacy freeze and evidence boundary

- Commit `3b0cf371759f3465c7264b85894d090004f3cf43` and tag
  `stage052-legacy-freeze-v1` are the legacy implementation boundary.
- The tag is a **freeze**, not a correctness claim. Historical source
  disposition records bind an older snapshot and must remain unchanged.
- Stage 5.2 performance-calibration Attempt16 is a closed, retained unique
  failure. Never describe it as a successful benchmark, revive it, reuse its
  label, or overwrite its raw/review evidence.
- Historical run labels, Stage paths, manifests, registries, lifecycle states,
  schemas, hashes, source maps, and old C++/Python ABI exports are immutable.
  Do not move, rename, regenerate, relabel, or promote them.
- `experiments/baselines/stage00/` is immutable. Large raw evidence remains in
  ignored or governed external storage; tracked summaries must come from an
  independent raw replay.
- Any code, config, ABI, runtime, resource, prerequisite, or evidence change
  requires a new TxnOpt source identity and attempt label. A failed attempt is
  append-only evidence.

## Target package and dependency graph

The target structure is:

```text
src/txnopt/                         generic transactional runtime
src/txnopt_cases/evrptw/            EVRPTW adapter
src/txnopt_cases/rcpsp/             RCPSP adapter
src/txnopt_evidence/                producers, reviewers, artifacts, lifecycle
src/txnopt_legacy/                  read-only historical/provenance readers
cpp/txnopt_core/                    generic native transaction runtime
cpp/txnopt_cases/evrptw/            EVRPTW native kernels
formal/                             TLA+/PlusCal and refinement mapping
experiments/txnopt/                 new protocol and evidence indexes
legacy/                             frozen identity and governance metadata
```

The dependency direction is fixed:

```text
txnopt_cases -> txnopt
txnopt_evidence -> txnopt + txnopt_cases
txnopt_evidence historical replay -> txnopt_legacy
txnopt -> never cases/evidence/legacy
```

- `txnopt` contracts use the standard library only. Core code must not import
  `evrptw`, Stage-numbered modules, case adapters, evidence, or legacy.
- Case adapters own domain semantics and may depend only inward on `txnopt`.
- Runners produce raw artifacts only. Reviewers run independently and rebuild
  state from raw events; producer and reviewer modules must not mutually import.
- New evidence uses the hash-chained `txnopt-evidence-lifecycle-v1` contract.
  Producers may advance only through `PLANNED -> RUNNING -> SEALED`; an
  independent reviewer appends `REVIEWED`. Legacy raw/review schemas remain
  readable but never acquire retroactive lifecycle events or rewritten hashes.
- `txnopt_legacy` is read-only. New active execution may not depend on it.
- Break cycles by extracting immutable identities, DTOs, or ports. Do not hide
  an architectural cycle behind a local import.

## Public API and internal contracts

The root `txnopt` package exports exactly:

- `TxnRuntime`
- `SearchKernel`
- `Oracle`
- `RunConfig`
- `RunResult`

Do not root-export candidate transactions, budget ledgers, cache transactions,
random tapes, semantic events, physical events, native handles, or Stage
compatibility types. These are versioned internal contracts.

The Level 1 identities are:

- distribution `txnopt`, version `0.1.0a1`;
- CMake project `txnopt_core`;
- extension `txnopt._native`;
- `txnopt-contract-v1`;
- `txnopt-native-round-v1`;
- `txnopt-semantic-trace-v1`;
- `txnopt-physical-trace-v1`.

A partially renamed distribution is not a Level 1 artifact. During migration,
old build metadata may remain only until the coordinated wheel/CMake/CLI
cutover is ready; never claim the transitional wheel as `txnopt 0.1.0a1`.

## Runtime ownership and state machine

`TxnRuntime` is the sole live owner of current/global-best state, random-tape
position, budget, cache generation, commit/rollback, and trace publication.
`SearchKernel` is pure proposal/decision logic. `Oracle` owns stable keys,
determinism declarations, safe screening, ordered expensive evaluation,
independent validation, and domain objective construction.

The candidate transaction state machine is:

```text
PREPARED -> RESERVED -> EVALUATING -> VALIDATED -> COMMITTED
      \          \             \             \-> ABORTED / INTERRUPTED
```

- Work is charged when it starts.
- Cache writes remain staged until a validated canonical commit.
- Each solve packs immutable instance context once.
- Each round crosses the Python/native seam once using contiguous SoA buffers
  and one typed receipt.
- Late results, insufficient budget, deadline, stale snapshot, worker failure,
  validation failure, and cache-write failure roll back the transaction.
- No hidden serial, scalar, Python, or old-ABI fallback is permitted.
- Level 1 canonical modes are serial, deterministic barrier, and ordered
  transaction execution. Unordered async is counterexample-only.
- Level 1 must not add a multi-round full-native fast path.

## Domain policies

### EVRPTW

- The objective remains the lexicographic tuple
  `(vehicle_count, total_distance, total_charging_time, charging_count)`.
- Vehicle count has absolute priority, including under simulated annealing.
- Objective construction, station-visit counting, comparison, validation,
  charging, and neighborhood semantics have one implementation in
  `txnopt_cases.evrptw`; callers must not duplicate them.
- Until differential gates pass, the new adapter must not silently activate the
  frozen `evrptw` solver or ABI.

### RCPSP

- State is a precedence-feasible activity order plus mode vector.
- The adapter owns precedence/resource lower-bound screening, block
  remove/reinsert, mode changes, single-thread fixed-seed CP-SAT exact repair,
  and makespan objective construction.
- An independent validator reconstructs every schedule. No EVRPTW field may
  leak into the generic runtime.

## Formal and trace policy

- T1-T4 are obligations, not claims. Keep model/checker outputs and proof status
  explicit. A draft TLA+ model is not a proof.
- Semantic trace contains only deterministic replay fields and participates in
  the semantic digest.
- Physical trace contains timestamps, queues, worker/thread identity, resource
  use, and completion order. It never participates in semantic equivalence.
- The Python reference and native round receipt must refine to the same formal
  state transition. Fixed-work semantic digests must match exactly.
- Close the positive Level 2 route if T3 is false, T4 is vacuous at measured
  parameters, or a direct source covers the complete narrowed contribution.

## Testing and evidence gates

- Use microstate exhaustion for completion order, deadline, and failure points.
- Property-test duplicate keys, cache conflicts, budget exhaustion, late
  results, stale snapshots, and commit/write failure.
- Differentially test Python and C++ one-round transactions.
- Differentially replay EVRPTW objective/validator and independently rebuild
  RCPSP schedules.
- Enforce import direction and root-export tests.
- Record exact exit codes for pytest, Ruff, strict mypy, Release builds,
  Python/native differential runs, ASan, TSan, and resource-leak tests.
- Formal evidence requires one exact clean source commit, wheel, native binary,
  environment, config, instance set, and reviewer identity. Never combine gates
  from different snapshots.
- Before modifying or renting compute for a formal matrix, verify there is no
  active writer, process, lease, socket, or shared-memory owner for the target.
- Cloud work is blocked until local build, semantic, fault, and runtime
  estimation gates pass. The Level 1 cloud window is at most 14 consecutive
  days and must be predicted to finish in ten days with four days of rerun
  margin.
- Full-scope confidence intervals and measured full-scope Cmax are outputs of
  the Level 1 formal matrix and are completion gates, not circular
  pre-procurement gates. Procurement still requires separate explicit user
  authorization after the local pre-cloud gates pass.
- The formal campaign runner requires a signed authorization bound to one exact
  plan, analysis digest, config tree, producer build manifest and wheel, absent raw
  root, and 14-day exclusive-Linux resource contract. It atomically claims the
  attempt before raw writes and terminates complete process groups on timeout.
  It produces raw evidence only. Independent review runs in separate processes
  from those raw manifests, rechecks host/producer/orchestration identities and
  per-run prefix/T4 receipts, and is the only layer allowed to aggregate
  performance or issue an internal-seal decision.

## Naming gate

At Level 1 completion, active imports, wheel contents, build targets, and CLI
must use TxnOpt. `evrptw` may then occur only in `txnopt_cases.evrptw`, frozen
legacy material, historical evidence, literature, and data licences. There may
be no active `import evrptw`, old CLI wrapper, new `stage05.2` schema, or old
native fallback in the TxnOpt artifact.

Until that coordinated cutover is complete, preserve the old source paths and
ABI rather than performing a visual mass rename that breaks historical
identity. Mark transitional status honestly in the roadmap and do not publish a
transitional wheel.
