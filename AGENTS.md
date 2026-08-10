# Repository Instructions

## Repository identity and publication boundary

- This is the independent `Reproducible-EVRPTW` research repository. The school
  FURP repository ends at Week 5 and is not a publication branch of this repo.
- The rewritten public history begins with Stage 0. Legacy source revisions in
  immutable evidence resolve through
  `docs/provenance/legacy-stage-commit-map.csv`; do not replace those historical
  hashes with public-history hashes.
- Large raw evidence remains external. Public Git tracks only code,
  configuration, tests, curated summaries, registries, manifests, review
  products, and the lightweight `artifacts/index.json`.
- Historical Pilot `stage05.2_benchmark_attempt82` is accepted only for its
  sealed native bounded-digest producer revision. Formal Attempt92 is immutable
  `superseded_partial` evidence: six archived batches remain external, its
  unfinished batch0007 and host exit 143 are bound by a supersession receipt,
  and none of its shards may be imported. The ABI-v2 candidate-transaction
  implementation requires a new Pilot and Formal label before Stage 5.3;
  Stages 6--8 are roadmap items only.
- Stage 5.2 resource calibration keeps the frozen 6-worker, 16,384-row-group,
  queue-depth-1, swap-free protocol and 20% memory headroom. Each completed
  axis explicitly releases unused PyArrow pool pages and records before/after
  Arrow allocation and process-RSS telemetry. Formal calibration and benchmark
  producers additionally run with `PYTHONMALLOC=malloc`; that allocator identity
  is signed into the producer resource contract, and the asynchronous artifact
  writer drops completed-batch references and calls libc `malloc_trim` after
  every eight batches while holding the measured writer turn. The trace records
  every periodic release. Large neighborhood spools are durably flushed and
  release clean page-cache ranges every 64 MiB; completed JSON evidence also
  performs a durable write followed by cache release. On WSL2 this is enabled
  only for the native ext4 staging filesystem and remains disabled for DrvFS/9p
  archive paths. Canonical spool merge also releases already-read clean ranges
  every 64 MiB so the read pass cannot recreate the same cgroup cache peak.
  These releases must not be represented as a relaxed resource
  gate. Failed Formal resource evidence may use either the historical v3
  process-tree RSS schema or the v4 swap-free dedicated-cgroup schema; v4
  recalibration must bind `aggregate_peak_memory_bytes`, the exact cgroup path,
  zero cgroup swap, and the sealed failure summary. A successful calibration is
  trusted only after independent
  replay of its terminal inventory, v3 report, scoped cgroup measurement/reset
  evidence, allocator and batch-release telemetry, and signed producer resource
  contract. CLI terminal manifests must replay both nested control manifests and
  direct `artifact-storage-v2` shard manifests; an unlisted shard artifact is a
  sealing failure, not a successful calibration. Full lifecycle retention binds
  both immutable identities: the storage-v2 canonical tree remains in the
  retention registry, while lifecycle close uses its independently signed
  path/size/SHA-256 content-inventory tree. The binding receipt must preserve and
  verify both; the two different canonical hash algorithms are not byte-equal.
  A migrated multi-segment benchmark prerequisite is consumed from its verified
  generation: control/review evidence comes from `wsl_active`, while immutable
  Pilot batches come from `d_benchmark`. Consumers derive the canonical run
  label from signed evidence, accept later locator aliases as a superset, and
  must not recreate deleted D: sources or use compatibility symlinks.
- Apache-2.0 applies only to original code and documentation. Benchmark data,
  papers, commercial solvers, and third-party repositories remain under their
  own terms.

## Agent skills

### Issue tracker

Issues and PRDs for this GitHub repository live in GitHub Issues; use the `gh`
CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default five canonical labels recorded in `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository. See `docs/agents/domain.md`.

These instructions apply to every Codex conversation whose working directory is
this repository or one of its subdirectories.

## Formal Objective Policy

- The formal optimisation objective is the lexicographic tuple
  `(vehicle_count, total_distance, total_charging_time, charging_count)`.
- Vehicle count has absolute priority. ALNS must always reject a candidate that
  increases vehicle count, including during simulated annealing.
- `evrptw.objective` is the normative objective contract. Python callers must
  use it directly and must not duplicate tuple construction, station-visit
  counting, or comparison logic. A GIL-free native search may mirror only the
  numeric key construction and comparison in `cpp/formal_objective.hpp`; that
  mirror must remain behind one native module and pass Python/C++ golden-vector
  differential tests, including decimal canonicalisation boundaries.
- `experiments/baselines/stage00/` is an immutable frozen baseline. Later-stage
  raw results belong under ignored `results/`, while review summaries belong in
  tracked experiment summary directories.

## Stage 2.1 Route-Reduction Policy

- Stage 2.1 remains available as the explicit `stage02_route_reduction` operator
  profile. The current `solve_alns()` default is Stage 2.3; Stage 0 and Stage 1
  historical runners must pass `operator_profile="baseline"` explicitly so
  their historical operator surface remains reproducible.
- Route elimination, vehicle-count-aware repair, and route merge are implemented
  behind the deep module `evrptw.neighborhoods`; callers use its proposals and
  event records rather than duplicating screening, repair, or objective logic.
- Route elimination is successful only when every removed customer is repaired
  into existing routes and the candidate has exactly one fewer route. Vehicle-
  count-aware repair must exhaust existing-route insertion before creating a new
  route. Route merge must pass safe capacity, optimistic time-window, and energy
  prefilters before exact charging evaluation.
- Every Stage 2.1 attempt records operator calls, prefilter decisions, exact
  evaluations, feasibility, acceptance, vehicle reductions, distance changes,
  and failure reasons. Failed rounds retain their complete raw output and use a
  new run label/output directory; prior failure evidence must not be overwritten.
- Stage 2.1 formal runs use the fixed Stage 0 12-instance, three-seed scope and
  are accepted only when every gate and both independent complete reruns pass.

## Stage 2.2 Cross-Route Quality Policy

- Stage 2.2 remains available as the explicit `stage02_route_quality` profile;
  `baseline` and `stage02_route_reduction` remain explicit profiles for historical
  reproduction. The current `solve_alns()` default is Stage 2.3.
- `relocate`, `swap`, `two_opt_star`, `route_segment_destroy`, and
  `ejection_chain` remain behind `evrptw.neighborhoods`; each proposal preserves
  route count and the segment/chain operators may repair only into existing routes.
- Every changed route passes the shared capacity, optimistic time-window, and
  optimistic energy prefilters before exact charging evaluation. Only changed
  routes are sent to the exact subproblem.
- Stage 2.2 uses the fixed Stage 0 scope and compares formal best objectives to
  `experiments/summaries/stage02_quality_attempt02_per_run_results.csv`; it inherits all
  Stage 2.1 gates and requires every new operator to produce a feasible candidate
  plus at least one accepted same-vehicle-count distance improvement.
- Failed rounds retain raw JSON, solutions, event and failure logs, environment
  metadata, and manifests under a new ignored `results/` directory and run label.

## Stage 2.3 Constraint-Guided Policy

- `solve_alns()` defaults to `stage02_constraint_guided`. `baseline`,
  `stage02_route_reduction`, and `stage02_route_quality` remain explicit profiles
  for historical reproduction; Stage 0 and Stage 1 runners must continue to pass
  `operator_profile="baseline"` explicitly.
- `station_pressure`, `time_window_conflict`, `worst_energy_detour`, and
  `shaw_related` are implemented behind the deep module
  `evrptw.neighborhoods`. Their deterministic ranking, dynamic removal-size
  selection, screening, repair, and event schema must be exercised through the
  public proposal interfaces rather than duplicated by runners.
- Dynamic removal size is enabled only in Stage 2.3. The fixed TOML ranges are
  small=5--10%, medium=10--20%, and large=20--35%, with stagnation thresholds 4
  and 8 iterations and exploration period 3. Counts are clamped to 1--`n-1`;
  every requested and actual count, tier, trigger, stagnation length, and reset
  is recorded. Baseline and earlier Stage 2 profiles retain their historical
  behavior.
- Stage 2.3 keeps the Stage 2.2 legacy trajectory in a separate, observable
  constraint-guided lane. Only candidates accepted under the shared
  lexicographic objective can update the global best, and an accepted candidate
  may never increase vehicle count. Probe, acceptance, rejection, infeasibility,
  budget exhaustion, and new-vehicle events are retained rather than silently
  discarded.
- Formal Stage 2.3 runs use the fixed Stage 0 12-instance, three-seed scope,
  30-second limit, 1000 iterations, and one thread. The comparison baseline is
  `experiments/summaries/stage02_quality_attempt02_per_run_results.csv`.
- The accepted Stage 2.3 probe configuration records two exact route evaluations
  for two-route quality probes and four for the route-segment quality probe in
  `configs/stage02_constraint_guided.toml`; it also reserves an explicit
  0.1-second constraint-lane slice from the fixed 30-second total. After the
  rerun05 objective-boundary failure, a bounded deterministic
  `vehicle_reduction_refinement` regret repair was added for low-fleet
  reductions; its exact-evaluation budget is 512 and it never creates routes.
  These values are part of the reproducible protocol and are not hidden
  time-budget changes.
- The independent review CLI must re-read raw solutions and event logs, replay
  the unified validator and objective, verify hashes and the immutable Stage 0
  manifest, and produce `review_report.md`, `review_findings.csv`,
  `failure_analysis.csv`, `stage03_readiness.csv`, and `review_manifest.json`.
  Stage 2.2 readiness-only instrumentation may come from the separately tracked
  `experiments/summaries/stage02_quality_attempt02_readiness_metrics.csv`, but
  the formal Stage 2.2 objective comparison remains the fixed per-run table.
  At least three real constraint-level failure or poor-quality cases are
  required; synthetic or deleted cases do not satisfy the gate.
- Stage 2.3 is complete only when both complete formal runs pass all inherited and
  new hard gates and the completed independent-rerun review status is
  `READY_FOR_STAGE03`. The accepted evidence uses
  `stage02_constraint_guided_attempt16` and
  `stage02_constraint_guided_rerun09`; earlier failed or superseded rounds
  remain preserved. Stage 3 cache,
  parallel, and exact-charging acceleration are readiness targets only and must
  not be reported as implemented in Stage 2.3.

## Stage 3 Canonical Artifact and Legacy Mapping Policy

- Stage 3.0 and Stage 3.1 historical evidence remains immutable. Stage 3.2
  implementation and its raw evidence use component `cache_incremental`.
  Stage 3.3 uses component `exact_deadline` and its accepted evidence is
  published. Stage 3.4 uses component `control_parallel` and its accepted
  evidence is published with formal review status `READY_FOR_STAGE04`.
- Canonical run labels are `stage03.0_measurement_attemptNN`,
  `stage03.1_screening_attemptNN`, or
  `stage03.2_cache_incremental_attemptNN`, or
  `stage03.3_exact_deadline_attemptNN`, or
  `stage03.4_control_parallel_attemptNN` (and the corresponding `rerunNN` form).
  Every artifact registry row records the
  canonical `run_label`, `attempt_or_rerun`, `artifact_type`, instance/seed
  scope, checksum, status, provenance hashes, validator status, comparison
  baseline, and `legacy_path` mapping.
- The canonical logical layout is
  `results/<run_label>/<instance>/<seed>/`. Canonical filenames follow
  `<stage_id>_<component>_<attempt_or_rerun>_<artifact_type>[_<instance>_<seed>].<ext>`.
  The current raw evidence remains physically stored under its original
  ignored `results/stage03-*` paths; the canonical paths are registry views and
  must not be implemented by copying or moving raw evidence.
- `experiments/registries/stage03.0_artifact_registry.csv`,
  `experiments/registries/stage03.1_artifact_registry.csv`, and
  `experiments/registries/stage03.2_artifact_registry.csv`, plus
  `experiments/registries/stage03.3_artifact_registry.csv`, and
  `experiments/registries/stage03.4_artifact_registry.csv`, are the published
  stage registries. Stages 3.3 and 3.4 were published only after formal review
  passed.
  `experiments/registries/stage03_legacy_path_map.csv` records the
  preserved old Stage 3 paths, the immutable Stage 0 frozen baseline, and the
  Stage 2.3 historical comparison references.
- The corresponding manifests are
  `experiments/manifests/stage03.0_measurement_artifact_manifest.json` and
  `experiments/manifests/stage03.1_screening_artifact_manifest.json`, plus the
  Stage 3.2 cache/incremental manifest, the Stage 3.3 exact-deadline manifest,
  and the Stage 3.4 control-parallel manifest after their raw reviews. Before a
  formal run or stage transition, the migration/preflight tool must verify
  canonical labels, artifact types, unique run labels, raw checksums and
  sidecars, manifest recomputability, legacy mappings, and semantic
  raw-to-summary consistency. Byte equality is not required when an auditor
  recomputes metadata; the declared semantic fields must agree.
- `stage02_constraint_guided_attempt16` and `stage02_constraint_guided_rerun09`
  remain historical baseline references under their old paths. Their recorded
  `repository_dirty=true` provenance is immutable and must remain visible.
  `experiments/baselines/stage00` is also immutable; only the compatibility map
  may add its `stage00_frozen_baseline` label.
- Historical incomplete, failed, timeout, or unreviewed Stage 3 rounds remain
  registered with explicit `legacy_with_manifest_error`, `not_present`, or
  `not_published` status. They must never be relabelled as successful evidence
  or overwritten by a later migration.

## Stage 3.0 Measurement and Replay Policy

- Stage 3.0 is measurement-only. It must not implement or claim cache
  acceleration, incremental propagation, interruptible exact charging,
  parallel evaluation, or fixed-work/wall-clock improvement.
- `solve_alns(..., measurement_config=None)` keeps measurement disabled. An
  opt-in `MeasurementConfig` creates one `Stage03Trace` containing a canonical
  route dictionary, route-level timing, `exact_call`, `cache_hit`, and
  `precomputed_route` records, separate `started_calls` and `completed_calls`,
  lane/iteration/operator context, candidate state, acceptance/global-best
  state, and deadline-boundary events.
- Stage 2.3 accepted evidence
  `stage02_constraint_guided_attempt16` and
  `stage02_constraint_guided_rerun09` is immutable historical provenance.
  Their recorded `repository_dirty=true` value must remain visible; it is not
  rewritten as a clean run. New Stage 3.0 runs require a clean main-repository
  commit before the runner starts.
- Stage 3.0 raw artifacts belong under a new ignored `results/` run directory;
  the runner rejects output paths elsewhere. The raw manifest has a separate
  `manifest.sha256` sidecar and the auditor verifies both before replay.
  Every run retains its raw solution, raw trace, event log, environment record,
  source/config/instance hashes, Stage 0 manifest hash, both reference-repo
  revision/dirty records, and interruption evidence. A failed run gets a new
  run label and is never overwritten.
- The independent `stage03_measurement_review` CLI must verify the raw
  manifest first, replay `validate_routes` and `evrptw.objective`, reconcile
  trace counters with `ALNSResult`, and check candidate/deadline semantics.
  Summary CSVs may be written under tracked `experiments/summaries/` only after
  replay gates pass; they must be recomputed from raw artifacts rather than
  copied from runner status tables.
- The smoke scope is exactly `c101C5`, `r105C5`, `rc105C5`, `c101_21`,
  `r101_21`, and `rc101_21`, with seeds `2014/2015/2016`, 30 seconds, 1000
  iterations, and one thread. Formal scope is Stage 0's exact 12x3 set and is
  blocked until smoke review reports `READY_FOR_STAGE03_FORMAL_MEASUREMENT`.
  A passing formal measurement review is the gate into Stage 3.1; it is not an
  acceleration result.

## Stage 3.1 Cheap Screening Policy

- Stage 3.1 is an opt-in safe screening layer. `screening_config=None` or a
  disabled `CheapScreeningConfig` preserves the Stage 0--3.0 solver path. It
  may reject only a route that is proven impossible by route structure,
  capacity lower bound, forward/backward time-window propagation, non-negative
  slack, optimistic battery reachability, or structural energy lower bound.
- The shortest-distance lower bound is recorded for diagnostics and candidate
  ordering only. It must never be used as a hard rejection rule. Screening
  rejection and negative-cache hit are separate `ScreeningDecision` records and
  do not increase exact charging-call counters. Only a screening pass may reach
  the existing exact route cache or exact charging solver.
- The process-local known-infeasible sequence cache stores only safe screening
  rejections and is keyed by the canonical route key. Exact infeasible results
  remain in the existing exact route cache and are reported separately.
- `Stage03Trace` v1 remains readable. Stage 3.1 appends `screening_config`,
  `screening_decisions`, check status/reason, slack, distance and energy lower
  bounds, cache-hit state, screening counters, and screening reason statistics;
  the route dictionary still stores each customer sequence once.
- `stage031_cheap_screening` and its independent review CLI use the fixed
  `stage02_constraint_guided` profile, 30 seconds, 1000 iterations, one thread,
  and the exact 18-run smoke scope. Formal scope is Stage 0's 36 runs and is
  blocked until the Stage 3.1 smoke replay reports
  `READY_FOR_STAGE031_FORMAL_MEASUREMENT` and the audited Stage 3.0 formal raw
  manifest plus trusted review manifest pass the prerequisite gate.
- Raw Stage 3.1 evidence is written only below ignored `results/`; interrupted,
  failed, deadline-overrun, and partial traces are retained under a new run
  label. `experiments/summaries/` is written only after replay of raw solution,
  trace, screening decisions, event log, environment, and manifests passes.
- Stage 3.1 review must report the screening reason table, exact-call ordering,
  negative-cache semantics, validator/objective replay, candidate vehicle-first
  acceptance, deadline semantics, source/config/instance/environment/reference
  provenance, and comparison with the Stage 3.0 formal per-run evidence. C5
  objectives must not regress and 100-customer vehicle count must not increase;
  exact-call reduction is measurement evidence, not a Stage 3.3 acceleration
  claim.
- Stage 3.1 is complete only when formal review reports
  `READY_FOR_STAGE03_2`. Stage 3.2 is opt-in and audits bounded route cache,
  station-reachability bitsets, and relocate/swap incremental propagation. The
  only Stage 3.2 exact-result store is the configured process-local LRU; every
  cache hit/miss/store/eviction and every changed/unchanged route status must be
  auditable. Review recomputes station reachability and cache lifecycle
  independently before trusting raw-to-summary reconciliation. Every Stage 3.2
  runner label must match `stage03.2_cache_incremental_attemptNN` or
  `stage03.2_cache_incremental_rerunNN`, and deadline/partial evidence is kept
  under `failures/` without overwriting prior runs. Its formal review must
  report `READY_FOR_STAGE03_3` before Stage 3.3. Stage 3.2 must not implement
  interruptible exact solving, fixed-work/wall-clock diagnostics, or parallel
  evaluation.

## CPU Batch Exact-Charging Policy

- `cpu_scalar` remains the historical exact-charging reference and
  `cpu_batch` may batch only ordered CPU work. Both paths preserve route order,
  priority-queue ordering, dominance pruning, validator results, objective
  tuples, exact-call counts, and cache semantics.
- Screening and cache lookups remain ordered. CPU batching applies only to
  contiguous exact cache misses; cache hits, duplicate requests, bounded LRU
  stores, evictions, deadline boundaries, and trace counters remain auditable.
- Historical Stage 0--3.2 runners explicitly use `cpu_scalar`. The independently
  reviewed `cpu_batch_pilot_attempt01` completed the fixed 4-instance,
  three-seed, 40-iteration protocol with every pair valid. The three
  100-customer family median savings were 21.97%, 11.54%, and 24.40%; therefore
  `solve_alns()` now defaults to `cpu_batch`. C5 remained a reported control:
  its three runs ranged from 8.66% slower to 4.54% faster, with a -0.12% median.
- CPU performance evidence belongs under a new canonical attempt directory in
  ignored `results/`; tracked summaries require independent review. Failed or
  inconclusive attempts remain explicit and must not be reported as Stage 3.3.

## Stage 3.3 Exact Deadline Policy

- Stage 3.3 uses component `exact_deadline` and canonical labels
  `stage03.3_exact_deadline_attemptNN` or `stage03.3_exact_deadline_rerunNN`.
  Its only formal backend is `cpu_batch`; `cpu_scalar`, missing backend fields,
  and implicit fallback are hard failures.
- `ExactDeadlineConfig` provides the two audited axes. Formal wall-clock runs
  use 30 seconds and 1000 iterations; fixed-work runs share one process-wide
  100 exact-call budget across every ALNS lane and use a 120-second watchdog.
- Exact calls are counted when an ordered cache miss is started. If the
  remaining budget cannot cover a complete candidate batch, the permitted
  prefix is recorded but the incomplete candidate and its cache writes are
  discarded. Deadline interruption likewise returns only the most recent
  complete ALNS incumbent and may not update acceptance or global best state.
  An exact call or atomic exact batch that returns at or after its lane deadline
  is recorded as interrupted, not completed; its route identities, completed
  counters, cache writes, and candidate transaction are discarded.
- Every call records started, completed, infeasible, interrupted, or budget
  boundary state. Backend evidence includes batch launches, transitions,
  packing/unpacking time, checkpoint count, exact calls, and total batch time.
- Smoke evidence is the fixed six-instance, three-seed scope with both axes.
  Formal evidence is the Stage 0 12-instance, three-seed scope with both axes.
  Independent review must verify current artifact storage, validator/objective,
  trace and cache reconciliation, deadline semantics, and the frozen CPU batch
  golden evidence before reporting `READY_FOR_STAGE03_4`.
- Stage 3.3 readiness is not Stage 3 completion. Any unmet median exact-call or
  effective-iteration target remains explicit work for Stage 3.4.
- The accepted Stage 3.3 evidence is smoke
  `stage03.3_exact_deadline_attempt05` and formal
  `stage03.3_exact_deadline_attempt06`. The independent formal review reports
  `READY_FOR_STAGE03_4` with all 72 axes valid. Attempts 01--04 remain retained
  as partial or `NOT_READY` evidence and must not be relabelled.
- Stage 3 performance is not complete: on the formal wall-clock axis the
  `r101_21` and `rc101_21` median started exact calls are 1860 and 2039, while
  on the fixed-work axis their median effective iterations are 6 and 3.
  Candidate control and the remaining performance target belong to Stage 3.4.

## Stage 3.4 Candidate Control and Controlled Parallel Policy

- Stage 3.4 is opt-in through `CandidateControlConfig`; `None` preserves the
  Stage 0--3.3 execution path. Its canonical component is `control_parallel`
  and its only exact backend is `cpu_batch`.
- Stage 3.4 formal evidence uses an explicitly registered **inherited warm
  start** protocol: the audited Stage 3.3 wall-clock incumbent (solution,
  objective key, and source SHA-256) is loaded as the initial solution for
  each instance/seed and re-verified through Stage 3.4's full
  screening→ranking→cpu_batch exact transaction pipeline. This protocol is
  necessary because Stage 3.3 used approximately 1860--2039 started exact
  calls to find its incumbents, while Stage 3.4's fixed-work axis is bounded
  to 100 started calls; cold-start search cannot match Stage 3.3 objective
  quality within that budget. The inherited incumbent is not trusted as-is;
  every candidate route is independently screened, cached, and exact-evaluated
  by the Stage 3.4 candidate-control runtime, and the reviewer independently
  recomputes candidate-work and route-result hashes from raw Parquet events.
  `inherit_stage033_incumbent` must be `true` in the formal configuration.
- Candidate ranking is deterministic and vehicle-first. Safe screening and
  cache lookup happen before exact work. A complete candidate whose cache
  misses do not fit the shared per-iteration budget is skipped atomically; it
  is not exact-infeasible and cannot populate the negative or exact cache.
- The serial and four-worker paths share candidate order and work. The
  four-worker path owns one reusable `spawn` process pool per solve, submits
  deterministic contiguous chunks, records actual completion order, and
  merges only in submission order. Worker or deadline failure aborts the
  candidate transaction without serial or `cpu_scalar` fallback.
- Fixed-work termination uses two explicit config parameters:
  `fixed_work_exhaustion_rounds` (consecutive no-new-exact-call rounds before
  exhaustion, currently 10) and `min_iterations_before_exhaustion` (minimum
  effective iterations before exhaustion termination is permitted, currently
  50). Both are recorded in the config, manifest, and environment metadata.
  An effective iteration is any completed ALNS iteration where the candidate
  was proposed, evaluated (via cache, screening, or exact), and accepted or
  rejected. The wall-clock axis is not subject to exhaustion termination.
- Smoke and formal diagnostics each use four axes: serial/parallel crossed
  with 100-call fixed-work and 30-second wall-clock. Smoke is exactly 72 axes
  and formal is exactly 144 axes. Failed attempts retain their raw bundles and
  use a new canonical attempt label.
- `READY_FOR_STAGE04` may be published only by the independent raw replay. In
  addition to inherited gates, both fixed-work worker variants must be
  semantically identical, both wall-clock variants must not regress any
  Stage 3.3 instance/seed objective, R/RC wall-clock median started calls must
  be at most 100, and R/RC fixed-work median effective iterations must be at
  least 50. Missing backend evidence, `cpu_scalar`, fallback, partial evidence,
  or any reconciliation failure is `NOT_READY`.
- Smoke `stage03.4_control_parallel_attempt05` completed all 72 axes but its
  independent review is `NOT_READY`: deterministic fixed-work semantics passed,
  while objective, R/RC call-count, and unchanged-route reconciliation gates
  failed. The unchanged-route cause was fixed after that immutable attempt.
  Attempts 06--07 retained partial or NOT_READY evidence. Attempts 08--09 were
  intermediate warm-start protocol runs. The corrected warm-start protocol with
  transactional reviewer gates, candidate hash recomputation, and hardened
  prerequisites was accepted as smoke `stage03.4_control_parallel_attempt10`
  (72/72 axes valid, `READY_FOR_STAGE034_FORMAL`) and formal
  `stage03.4_control_parallel_attempt11` (144/144 axes valid,
  `READY_FOR_STAGE04`). Earlier failed attempts remain preserved and are not
  overwritten.

## Stage 4 Adaptive Weights and Search Control Policy

- Stage 4 is opt-in through `Stage04Config`; `None` preserves the Stage 0--3.4
  execution path. Its canonical component is `adaptive_weights` and its only
  exact backend is `cpu_batch`.
- `Stage04Config` is a frozen dataclass defined in `evrptw.stage04`. It
  controls segment length, min calls per operator, weight reaction/smoothing,
  fixed-weight ablation, differentiated reward tiers, auto-estimated SA
  temperature, reheating, stagnation restart, and incumbent intensification.
  The `reward_for()` method computes differentiated rewards: vehicle reduction
  (8.0) > distance improvement (4.0) > accepted equal (1.0) > accepted worse
  (0.5) > rejected (0.0); new global best with vehicle reduction gets 16.0.
  The `with_fixed_weights()` helper creates a fixed-weight ablation copy.
- Segment-based weight update replaces per-call exponential moving average.
  Weights are updated at segment boundaries (default 50 iterations) using
  accumulated rewards and smoothing. Each operator must reach a minimum call
  count (default 5) within a segment before its weight is updated.
- `OperatorStatistics` tracks six categories: `accepted_improving`,
  `accepted_equal`, `accepted_worse`, `rejected`, `new_global_best`, and
  `vehicle_reduction`. All three ALNS lanes (legacy, quality_shadow,
  constraint) and the refinement path increment these categories.
- Auto-estimated SA temperature samples random worse moves (default 30
  samples) to calibrate the initial temperature targeting a 50% acceptance
  rate. Reheating boosts the temperature floor on stagnation (default
  threshold 10, factor 0.5, max 5 reheats); the floor decays at 0.99 per
  iteration.
- Stagnation restart (default threshold 20, max 3) resets the current solution
  to the global best. Incumbent intensification (default 10 iterations, 10%
  removal fraction) concentrates search around the best after restart.
- Canonical run labels are `stage04_adaptive_weights_attemptNN` or
  `stage04_adaptive_weights_rerunNN`. The experiment runner uses four axes:
  `adaptive_wall_clock`, `adaptive_fixed_work`, `fixed_wall_clock`,
  `fixed_fixed_work`. Smoke scope is the 6-instance, 3-seed set; formal scope
  is Stage 0's 12-instance, 3-seed set (144 axes).
- The independent review CLI re-reads raw artifacts, replays
  validator/objective, and evaluates six gates: `operator_call_sufficiency`
  (wall_clock axes only), `six_category_statistics` (adaptive_wall_clock only),
  `adaptive_better_than_fixed` (>= 3 wins on different instance/seed pairs),
  `not_single_best_seed` (>= 2 winning seeds), `std_not_increased` (Stage 4
  vehicle_count std <= Stage 0), and `replay_consistency` (all axes valid with
  matching objectives).
- Stage 4 v5 raw evidence persists normalized per-operator six-category
  statistics and every segment update/skip decision. All adaptive-weight roles
  in the legacy, quality-shadow, and constraint lanes use segment accumulation;
  per-call weight updates are forbidden. The reviewer requires the exact scope
  identity set, rejects missing or duplicate axes, and requires exactly one
  event for every segment-boundary/operator pair. It verifies that an update
  occurs only after the configured minimum segment calls.
- `stage04_adaptive_weights_attempt01` and
  `stage04_adaptive_weights_attempt02` are preserved v1 evidence. The v1
  reviewer did not independently verify complete per-operator six-category
  and segment evidence, so v2 does not inherit their readiness claim.
- The v2 evidence Smoke `stage04_adaptive_weights_attempt03` and Formal
  `stage04_adaptive_weights_attempt05` is preserved but superseded: the v3
  reviewer correctly rejects its incomplete adaptive-role and call-partition
  audit, so neither run is current readiness evidence. Formal attempt04 was interrupted
  after 30 of 36 instance/seed bundles and remains preserved as incomplete
  evidence; it was not overwritten or used for readiness. New v3 Smoke and
  Formal evidence require new attempt labels before `READY_FOR_STAGE05` may be
  republished. The v3 Smoke `stage04_adaptive_weights_attempt06` and Formal
  `stage04_adaptive_weights_attempt07` are preserved but superseded because
  their producer still used per-call updates in one lane and their reviewer did
  not enforce the complete boundary/operator event matrix. v4 attempts 08 and
  09 are preserved incomplete failure evidence that exposed two remaining
  unclassified-call paths. The corrected accepted evidence is Smoke
  `stage04_adaptive_weights_attempt10` (72/72 axes) and Formal
  `stage04_adaptive_weights_attempt11` (144/144 axes). That v4 evidence is
  preserved but superseded because a fixed-work budget boundary could still be
  followed by candidate acceptance. The v5 reviewer independently replays
  critical boundary events and rejects any later accepted/global-best event.
  Smoke `stage04_adaptive_weights_attempt12` and Formal
  `stage04_adaptive_weights_attempt13` are preserved v5 evidence, but are
  superseded: the v6 reviewer adds the missing non-adaptive refinement call
  partition, exact raw/solution/trace axis identity, and fail-fast raw numeric
  parsing. The accepted v6 evidence is Smoke
  `stage04_adaptive_weights_attempt14` (72/72 axes) and Formal
  `stage04_adaptive_weights_attempt15` (144/144 axes). The formal independent
  replay reports `READY_FOR_STAGE05`: all six gates pass, including the complete
  operator/segment audit, six-category reconciliation, exact identity set, and
  Stage 0 vehicle-count standard-deviation comparison. Earlier evidence remains
  preserved and cannot be promoted by the v6 reviewer.
- Stage 4 review products are tracked under `experiments/summaries/` with
  the current accepted Formal attempt prefix; the artifact registry is
  `experiments/registries/stage04_artifact_registry.csv` and the manifest is
  `experiments/manifests/stage04_adaptive_weights_artifact_manifest.json`.

## Stage 5.1 Best-Known Values Policy

- Stage 5.1 uses component `best_known` and canonical labels
  `stage05.1_best_known_attemptNN` or `stage05.1_best_known_rerunNN`. The
  entry prerequisite is Stage 4 `READY_FOR_STAGE05`.
- The runner does not invoke `solve_alns`. Its evidence is the collected
  best-known-solution (BKS) reference data from `evrptw.best_known`, not a
  solver output. The runner requires a clean main-repository commit before it
  starts and writes raw evidence only under a new ignored `results/` run
  directory.
- BKS data sources are formally published journal articles only. Small
  instances (5, 10, 15 customers) use Schneider, Stenger & Goeke (2014)
  Table 5 CPLEX optimal values, with RC204-15 updated to the VNS/TS value
  from the same table. Large instances (100 customers) use Keskin & Çatay
  (2016) Table 2, which compiles the best-known values from SSG, Goeke &
  Schneider (2015), and Hiermann et al. (2016). Goeke & Schneider is not in
  the VOR collection; its DOI was verified via CrossRef but the PDF was not
  stored locally.
- Instance name mapping: paper notation `C101-5` maps to repository
  `c101C5` (lowercase, replace hyphen with `C`); paper `c101` maps to
  `c101_21` (append `_21` suffix for Solomon 100-customer instances). All 92
  instances (36 small + 56 large) have BKS values. Charging time and charging
  count are never reported in published BKS tables and are always `unknown`.
- Model compatibility is assessed across five dimensions: charging model
  (full recharge — compatible), objective function (published BKS uses
  lexicographic vehicle-count-first distance minimization without charging-time
  or charging-count terms — incompatible), distance metric (published BKS may
  use rounded Euclidean — incompatible), time windows (compatible), and vehicle
  parameters (compatible). Overall compatibility is `False`.
- Because the models are not completely identical, no gap computation is
  performed. All 92 instances are marked `model_compatible=False` in
  `experiments/baselines/schneider_best_known.csv` with a compatibility note
  explaining the objective mismatch. No estimated, backfilled, or
  model-inconsistent gap is reported.
- The independent review CLI verifies five gates: `instance_coverage` (all
  92 instances present), `bks_values_present` (no `unknown` distance or
  vehicle values), `no_gap_computation` (no gap columns in the BKS CSV),
  `compatibility_assessment_correct` (CSV matches the canonical
  `COMPATIBILITY_ASSESSMENT`), and `replay_consistency` (CSV values match the
  canonical `BEST_KNOWN_VALUES`). The review reports `READY_FOR_STAGE05_2`
  only when all five gates pass.
- Stage 5.1 v6 requires the published Stage 4 v6 Formal prerequisite
  (`stage04_adaptive_weights_attempt15`, Formal scope, 144 axes, and matching
  review identity), canonical single-level run layout, exact CSV schemas, unique
  instance rows, and complete field-by-field replay including citations,
  charging unknowns, and model-compatibility fields. `stage05.1_best_known_attempt01` remains
  preserved v1 evidence and is superseded for readiness purposes.
- `stage05.1_best_known_attempt02` remains preserved v2 evidence with a
  canonical single-level 92-instance bundle, but it is superseded by the v3
  prerequisite and independent-conversion replay requirements; no current
  Stage 5.1 readiness derives from it. Attempt01 remains historical
  nested-layout evidence. The
  v3 evidence `stage05.1_best_known_attempt03` is preserved but superseded
  after the Stage 4 v3 prerequisite was withdrawn. v4 attempt04 is also
  preserved but superseded after the Stage 4 v4 prerequisite was withdrawn.
  `stage05.1_best_known_attempt05` is preserved v5 evidence but superseded
  after the Stage 4 v5 prerequisite was withdrawn. Stage 5.1 v6 additionally
  requires the exact Stage 4 Formal identity (`scope=formal`, 144 axes, and a
  matching review run label). The accepted v6 evidence is
  `stage05.1_best_known_attempt06`: its 92-row independent replay passes all
  five gates and reports `READY_FOR_STAGE05_2`, inheriting the published
  `stage04_adaptive_weights_attempt15` Formal review. All earlier attempts
  remain preserved superseded evidence.
- The artifact registry is
  `experiments/registries/stage05.1_artifact_registry.csv` and the manifest
  is `experiments/manifests/stage05.1_best_known_artifact_manifest.json`.

## Stage 5.2 Performance and Benchmark Policy

- Stage 5.2 starts only from accepted `stage05.1_best_known_attempt06` with
  review status `READY_FOR_STAGE05_2`, inheriting the accepted Stage 4 Formal
  identity `stage04_adaptive_weights_attempt15`. It is one continuously
  maintained implementation. A--G are ordered evidence gates inside that one
  implementation: `perf_baseline`, `hot_path`, `artifact_streaming`,
  `job_parallel`, `native_kernels`, optional `accelerator_pilot`, then
  `benchmark`; they are not separately maintained software versions.
- `stage05.2_<component>_attemptNN` and `rerunNN` are unique run identities,
  not version names. Cross-run and cross-revision shard import is forbidden.
  A Formal campaign may reopen its existing output only while its lifecycle is
  still `RUNNING`, with no active writer or terminal manifest, and only through
  a one-use signed resume permit bound to the original campaign identity. The
  append-only recovery epoch chain may reuse independently revalidated archived
  batches from that same identity; an incomplete batch contributes no shard and
  is discarded as one signed retention transaction before whole-batch rerun.
  Current-chain truth comes from signed manifests,
  prerequisite references, and the newest verified v2 registry generation
  stored with the bound `e_archive` governance state. The tracked
  `stage05.2_retention_registry.csv` is the immutable v1 compatibility source,
  never a competing current v2 truth source; neither may be replaced by a
  hard-coded attempt number in policy documentation.
- Recoverable interruption causes are exactly `unexpected_host_loss` and
  `operator_stop`. Operator stop requires a signed stop intent written before
  service shutdown plus a signed inactive-service receipt. Algorithm,
  validator/objective, manifest-integrity, scientific-configuration, resource-
  gate, or unknown failures terminate the label. Any code, tree, wheel, native
  extension, dependency, configuration, prerequisite, resource contract,
  campaign plan, lifecycle plan, or immutable start-permit change requires a
  new label; completed batches cannot cross that boundary. Recovery never adds
  a `PAUSED` lifecycle state.
- Every sealed Stage 5.2 source snapshot must materialize the required ignored
  benchmark inputs as ordinary snapshot-local files; a Git clone alone is not a
  complete producer source. Formal memory probing verifies the exact non-empty
  instance input before creating its output root or starting worker work.
  Missing ignored input fails before scientific execution, and any unit label
  already launched for that failure remains consumed and immutable.
- Full Stage 5.2 evidence exists in the active staging root only while it is
  being produced or reviewed. New long-term generations publish under the
  bound `e_archive` role; `d_archive` remains a host-capacity and legacy
  resolution role. Accepted/current-chain and unique-root-cause failures retain
  complete raw evidence. A signed adjudication may reduce a superseded
  duplicate failure to audit-only control manifests, reviews, logs, checksums,
  failure evidence, and its failure-triggering representative shard. Unknown
  status, root cause, or seal identity always fails closed to full retention.
- Retention audit rejects active, planned, or otherwise unsealed runs by
  default. Cross-volume archival copies to a hidden target-volume incoming
  generation, verifies every file and logical identity, atomically publishes
  it, and registers `(run_label, segment_id, generation)` without rewriting the
  source manifest. Historical D/WSL migration cleanup is a separate destructive
  action that requires an exact verified deletion list and literal user
  confirmation `确认`; no historical source is deleted merely because a
  retention generation exists. This confirmation boundary does not replace the
  explicitly configured same-attempt rolling-batch handoff needed to preserve
  the Stage 5.2 active-workspace cap.
- Archived prerequisite or review input is resolved by run label through the
  newest verified v2 registry generation under `e_archive`, with
  `stage05.2_retention_registry.csv` as the v1-only fallback, plus the local
  storage-root locator. The resolver must recheck the registered file count,
  byte count, tree SHA-256, and any required full-replay receipt before
  returning a path to an existing runner or reviewer; policy files and callers
  must not embed the machine-local archive path.
- A registered archive tree is immutable. Reviewers may consume it as a
  comparison, prerequisite, or replay input, but may not publish a new review
  generation inside it. A run that still needs review publication remains in
  the active root until that generation is sealed, then it is archived.
- `artifact-storage-v2` with physical schema `screening_decisions_v3` uses
  typed bounded Parquet streams and compatible v1/old-v2/v3/legacy reads.
  New campaigns freeze a calibrated 65,536/262,144 row-group choice and queue
  depth 1/2 in their signed producer resource contract; the non-baseline choice
  is allowed only after at least 10% persistence-critical-path improvement,
  semantic equality, and the signed measured-capacity gate. Canonical semantic digests
  are computed over expanded logical events, so physical IDs and compression
  layout cannot change replay. Persistence is at most 36% of end-to-end time
  and peak RSS is at most 50% of the Stage 5.2 v1 baseline.
- A replacement Formal may update only the resource-envelope fields of the
  accepted Pilot producer resource contract when a signed v2 calibration report
  binds a failed Formal batch with zero readiness-geometry contribution. Memory
  capacity/peaks/limits and their calibration/semantic digests may be
  remeasured; the worker/row-group/queue-depth topology and the separate
  scientific execution selection lock remain fixed. The producer and
  independent reviewer must replay the same report/sidecar, failed
  run/batch/resource-summary identity, clean calibration revision, cross-worker
  semantic equality across the exact unique worker identity set `{4, 5, 6}`,
  exact replacement-contract digest, exact 20% headroom, and zero
  swap/fallback/resource-limit failures. The independent Formal memory probe
  retains its own valid semantic digest and must fit below the replacement
  contract's selected aggregate and per-worker peaks. Missing or mismatched
  evidence, lower floors, topology changes, or shard reuse fail fast.
- Calibration Attempt23 may cross the `58c325a` revision boundary only through
  a signed `resource_contract_only` successor attestation after its sealed
  independent review reports `ACCEPTED`. Producer and reviewer independently
  recompute the exact Git changed-path/blob inventory and the report, review,
  and contract hashes. Solver, objective, native-kernel, shard/artifact schema,
  or scientific-configuration changes reject inheritance; this exception never
  authorizes Formal batch reuse across revisions.
- Every Stage 5.2 calibration CLI invocation must pass the clean ext4 checkout
  explicitly through `--repository-root`. Service working-directory state is
  not a source-identity input and may not be relied on implicitly. A missing
  repository binding must fail in argument parsing before an evidence label is
  started. Relative `--config` paths resolve against that explicit repository,
  never against the service working directory.
- Job-parallel selection compares 1/2/4 workers on the fixed four-instance,
  three-seed scope. Two workers require at least 1.5x speedup and at most 12 GiB
  aggregate RSS; four workers are selected only at 2.5x and at most 12 GiB.
  Worker count is maximum concurrency, not a lifetime PID count. Every parallel
  `(instance, seed)` shard uses a fresh `spawn` worker with
  `max_tasks_per_child=1`; cross-shard allocator reuse is forbidden. Batch
  metadata records `worker_process_lifecycle=one_shard_per_spawned_process`.
  Each fresh worker performs an in-memory Arrow/Zstandard initialization before
  measured shard persistence and records
  `worker_runtime_warmup=in_memory_arrow_zstd1`; this warmup never replaces,
  drops, aggregates, or skips audit events.
  Worker ownership uses real sampled PIDs and cumulative process CPU samples,
  permits more owner PIDs than configured concurrency, and still requires at
  least the configured worker count. Worker failure aborts without fallback.
- Native-kernel promotion must preserve Python/native objective, validator,
  exact ordering, candidate/cache/event semantics, and zero fallback across all
  fixed-work axes. Aggregate 100-customer paired median end-to-end improvement
  must be at least 15%, no C/R/RC family may regress by more than 3%,
  persistence must remain at most 36%, each worker RSS at most 4,357,382,144
  bytes, and process-tree RSS at most 12 GiB.
- Native-architecture performance take-over is owned by the deep module
  `evrptw.stage052_performance`. The runner consumes one signed
  `FrozenPerformanceProfile`; it must not reconstruct build flags, CPU counts,
  affinities, worker counts, scheduler layout, or memory admission locally.
  The profile is frozen per mode and per `c5`/`100-customer` workload class from
  the allowed Linux/WSL x86-64 CPU affinity, physical-core/SMT topology, and
  effective cgroup/host memory limit. The current publication host uses all 24
  allowed logical CPUs, but 24 is not a portable source constant. Missing
  topology has a deterministic portable fallback, odd CPU counts are balanced,
  and a campaign never retunes or silently degrades a frozen topology.
- Topology calibration compares approximately `N x 1`, `ceil(N/2) x 2`,
  `ceil(N/3) x 3`, and `ceil(N/4) x 4` shard layouts, both freely scheduled and
  mutually exclusive physical-core-first/SMT-last partitions. Host-scheduler
  calibration also compares shared affinity with disjoint client-control and
  scheduler-compute partitions. A single-axis PSS calibration rejects any
  topology that would exceed 80% of effective memory, violate 20% headroom,
  use swap, exceed a native worker/request-thread capability, or fail to admit
  its slowest tail axis. Configured thread counts never substitute for sampled
  process/thread identity, actual affinity, useful CPU work, or queue evidence.
- A clean source identity may produce portable O3, portable LTO, and, only when
  the compiler supports it, host-native LTO no-cache wheels. `fast-math` is
  forbidden. Each build receipt binds the compiler identity and flags, CPU
  feature mask, Git revision/tree, source inventory, and wheel/native/scheduler
  SHA-256. Three-repeat fixed-work calibration rejects any objective, route,
  candidate trajectory, exact order, cache lifecycle, or transaction-hash
  drift. Selection minimizes measured mode-block startup-through-replay
  end-to-end median; a difference below 3% or an overlapping paired confidence
  interval selects the more portable and then lower-memory build.
- Stage 5.2 native kernels use ABI
  `stage05.2-native-kernels-v2`. `NativeCandidateTransactionConfig` is an
  explicit `solve_alns()` opt-in and preserves the existing operator order and
  operator exact-evaluation budgets. Its four-step fixed-work ablation is
  `current_native` → `pair_pruning` → `batched_screening` →
  `candidate_transaction`. Pair pruning emits one ordered aggregate identity
  and exact skipped-candidate count; native screening owns ragged packing,
  canonical identity, batch deduplication, negative-cache lookup, safe
  screening, counters, and SHA-256. Full transactions order screening, cache
  lookup, ordered `cpu_batch`, staged cache writes, deadline/budget checks, and
  atomic commit/rollback. Native, worker, deadline, or integrity failure is
  fail-fast with zero Python, serial, CUDA, or `cpu_scalar` fallback. Historical
  Stage 3.4 `CandidateControlConfig` remains a separate path.
- Experimental native-architecture comparisons use the explicit
  `Stage052NativeExecutionConfig` schema. Passing `None` preserves the
  historical Stage 0--5.2 paths and their guards. The implemented
  `per_solve_runtime` protocol is `candidate_round_soa_v2`: each solve owns one
  persistent bounded C++ work pool sized to its frozen axis allocation, and
  each candidate round crosses the Python/native boundary exactly once with
  contiguous SoA inputs (including typed base propagation snapshots) and
  structured propagation, screening, ranking, cache-journal, exact-result,
  completion, timing, and SHA-256 outputs. Python independently replays the
  transaction hash and commits cache/control state only after the complete
  result passes.
  Worker, deadline, integrity, or cache-commit failure rolls back the complete
  transaction and never opens the historical worker pool or another backend.
  `full_native_alns` uses `full_solve_soa_v2` and crosses the Python/C++
  boundary once per instance/seed. Its `NativeSearchEngine` owns the current
  Stage 2.3 lanes and operators, Python-compatible RNG, Candidate Control,
  exact budget/deadline and cache lifecycle, refinement, lexicographic
  objective, and Stage 4 segment/search-control semantics. The complete native
  search loop runs with the Python GIL released and returns typed SoA route,
  exact, trajectory, cache, control, causal, and canonical-event journals for
  independent replay. Capability receipts remain fail-closed until the matching
  differential, thread-equivalence, fault, sanitizer, and full-suite gates pass.
  `host_scheduler` uses `unix_shm_scheduler_v2`: a run-owned temporary service
  accepts framed Unix-domain control messages, maps contiguous POSIX shared
  memory arrays, and dispatches the same all-or-nothing C++ solve ABI through
  the frozen scheduler-compute pool and request-thread count. Service loss,
  partial IPC, schema/hash failure, or output loss fails the transaction without
  local recovery or fallback. The Unix-domain service loop, bounded IPC
  reads/writes, request/work queues, and compute pool are C++ owned; Python only
  starts/stops the temporary service and validates the returned transaction.
  Input and output transport uses fixed
  binary headers plus typed shared-memory SoA descriptors; JSON, pickle, NumPy,
  and Python shared-memory workers are not part of the service path. The host
  scheduler and local full-native mode share the same native transaction and
  work-pool interfaces. Candidate sessions isolate RNG, budget, cache, deadline,
  and transaction state; disconnect, partial IPC, ACK loss, schema/hash failure,
  worker failure, or service loss rolls back affected pending work and fails
  without fallback. For an explicit commit/rollback request, RELEASE is the
  primary server-side commit receipt. If RELEASE is lost after the client ACK,
  the client queries the isolated session's typed transaction status and
  commits local state only when the server reports the matching committed or
  rolled-back outcome; service loss or any unresolved status fails fast. These
  queues have bounded deterministic chunks and signed task receipts containing
  submit/start/complete times and the actual worker. Queue-full, rejection, or
  dropped-receipt counts must remain zero. Per-wave and mode-block scheduler
  topology receipts include exactly one dedicated asynchronous task-receipt
  writer thread in addition to compute, request, and main service threads; the
  observed writer count must be one and is included in the expected process
  thread total. Per-wave and mode-block scheduler
  lifecycles are compared; mode-block reuse is selectable only when final
  selected-build/topology session isolation, cache reset, process-tree PSS
  stability, and semantic replay all pass. These experimental protocols do not
  select a default or promote Stage 5.2 readiness; qualification still requires
  the new five-mode Paired/Pilot evidence and independent review.
- Five-mode comparisons use
  `evrptw.experiments.stage052_native_architectures` and independent replay in
  `stage052_native_architecture_review`. Paired scope is exactly 360 axes and
  Pilot scope is exactly 180 wall-clock axes. Every mode is rebuilt and rerun
  from one clean commit, frozen wheel/native hash, input, budget,
  instrumentation envelope, and the same host-wide CPU/memory budget; each mode
  uses its own signed fastest-safe topology. All three native modes must pass
  the complete correctness matrix, while promotion requires at least one native
  mode to pass the performance gates. Failed axes are signed evidence, not
  deleted. The independently optimized `current_stage052` mode is the primary
  denominator; accepted Pilot attempt72 is semantic drift evidence only and is
  never a current CPU/resource denominator. Neither runner starts Formal,
  changes the production default, launches CUDA, or reuses a label. Pilot is
  blocked until an independently signed Paired review proves exactly 360/360
  valid axes for the same profile/build/source identity.
  Paired `attempt01` completed all 360 axes but is superseded before review:
  the v1 JSON producer duplicated complete in-memory measurement/event rows and
  emitted about 16 GiB for `current_stage052` alone, while the v1 reviewer
  attempted to materialize every axis simultaneously. The immutable attempt01
  directories remain failure evidence and must not be reused or promoted. The
  v2 producer replaced duplicated rows with count-plus-SHA-256 semantic stream
  evidence. Paired and Pilot attempt02 both completed their fixed axis counts,
  but paired replay exposed two immutable failure classes: non-finite diagnostic
  values were not explicitly encoded before strict JSON hashing in some
  Python/per-solve fixed-work axes, and the experimental full/host initial split
  could exceed the 100-call budget. The v3 producer explicitly tags non-finite
  diagnostics, and its reviewer retains failed axes as failed comparisons
  instead of crashing. New paired/Pilot evidence must use attempt03 labels and
  the v3 reviewer schema; attempt02 remains unpromoted failure evidence.
- Stage 5.2 safe-rejection acceleration is bounded independently from the
  collision-proof identity stores. The scalar `ScreeningResult` cache is a
  65,536-entry solve-local LRU; the Python/native sequence cache is a
  65,536-entry generation cache whose atomic rollover drops only safe
  re-screenable rejections. Rollover must replace the Python mapping and packed
  native ABI state in one candidate transaction and must be reversible on
  failure. Eviction may cause safe screening to run again, but it may not admit
  an infeasible route, consume exact work, change candidate order, or weaken
  full SHA-256 collision proof. A bounded-LRU negative hit uses a stable
  positive lookup marker derived from the collision-free canonical route key
  plus the complete normalized screening result; equivalent post-eviction
  recomputation must keep the marker. The route-key consistency guard must also
  compare the complete, compact binary typed result signature, so marker
  truncation or collision cannot hide an evidence-field change without restoring
  object-heavy evidence retention. Raw statistics record capacity, current/peak
  entries, stores, evictions, and rollovers; the independent reviewer rejects
  missing, unbounded, or internally inconsistent cache evidence.
- The accelerator gate independently recomputes median 100-customer native
  candidate screening-pool occupancy from raw candidate-transaction
  statistics; exact-backend launch occupancy is not a substitute. Median below
  32 publishes `GPU_NOT_JUSTIFIED`; median at least 32 requires one wave-owned
  registered CUDA helper, exact fixed-work equality, at least 15% aggregate
  improvement, and no family regression over 3%. A complete but non-promoting
  CUDA pilot publishes `NATIVE_CPU_RETAINED`; only a passing pilot publishes
  `ACCELERATOR_PROMOTED`. Missing helper, fallback, or an unaudited campaign
  adapter is `NOT_READY`.
- The pipeline pilot is exactly 12 instances x 3 seeds x one 30-second axis and
  exercises resource sampling, failure recovery, bounded replay, 1/5/10/30
  second anytime checkpoints, configured archive roots, and interrupted
  publication recovery. Formal contains exactly 920 indivisible
  `(instance, seed)` shards, 2,040 runs, 229,200 declared solver seconds, and
  10,400 anytime rows. Only independent review may open Formal or report
  `READY_FOR_STAGE05_3`.
- New campaign review must use `evrptw.stage052_replay` with
  `replay_backend=native_arrow`; Python reference replay is differential and
  legacy-compatibility code only, and native failure may not fall back.
  Review calibrates 1/2/4 batch-scoped workers, gives each child exactly one
  shard with `max_tasks_per_child=1`, merges by canonical shard ordinal, and
  cancels all unfinished work on the first child failure. The Pilot-derived
  parent baseline and per-child p99 RSS determine `MemoryHigh`, the internal
  process-tree guard, and `MemoryMax`; the limits must fit the measured
  available capability without swap or throttling the selected concurrency.
  Review v2 records per-shard elapsed time, events/second, child
  peak RSS, merge ordinal, in-flight bound, and zero native fallbacks.
- The accepted D/F worker-selection evidence keeps its historical 12-GiB
  scientific gate. The current G Benchmark calibration compared 4/5/6 workers
  and an explicit 8-worker stress probe on fixed real high-memory shards.
  Six producer workers are frozen for the replacement Pilot/Formal: they exceed
  the four-worker throughput gate, preserve the semantic digest with no swap or
  fallback, and the 8-worker probe was materially slower. The signed resource
  contract uses measured per-worker and process-tree peaks plus explicit
  operating headroom; an arbitrary percentage of otherwise usable memory is
  not a publication gate. The selected worker count and measured limits are
  frozen into the Pilot and inherited unchanged by Formal.
- Campaign preflight and per-batch handoff retain two consecutive 30-second
  telemetry windows. AC/battery state, low-power mode, system load, CPU model,
  operating-system version, temperature, disk model/serial, device UUID, and
  unrelated-process load are observable telemetry, not readiness identity or
  hard publication gates. Hard capability checks require only enough logical
  CPUs for the selected workers, calibrated memory and free space, the frozen
  backend/Python/native-extension contract, and filesystem fsync plus atomic
  transfer support. Failed batches retain all collected telemetry.
- G campaign producer dispatches contiguous waves of at most the calibrated worker
  count. Every spawned worker processes exactly one shard, and the whole wave
  pool must shut down before the next wave is created. This preserves frozen
  concurrency and unique per-shard PIDs while preventing retiring and warming
  workers from overlapping in the audited total `load1`.
- G may consume accepted F evidence from an older revision only when the
  current revision is its Git descendant and the entire intervening diff is
  confined to the explicit G campaign runner, artifact-persistence adapter,
  reviewer, test, and documentation allowlist. The independent reviewer
  repeats this diff audit. Runtime
  selection identity still freezes Python ABI, dependencies, source/wheel,
  native extension, configuration, instances, backend, workers, and audit
  protocol;
  solver, objective, configuration, native, or other source drift is a hard
  failure and requires a new prerequisite rather than a G-only continuation.
- Candidate events retain both customer-sequence route identity and the
  complete exact-charging depot/station route identity. Independent review
  validates every accepted global-best complete route and objective and
  requires its customer projection to equal the recorded customer sequence;
  either identity missing or drifting is a hard failure.
- Deadline boundaries are lane-local because the constraint-guided profile
  reserves its final 0.1-second slice after the legacy/quality deadline.
  Independent replay forbids exact work, cache stores, or candidate acceptance
  after a boundary in the same lane and independently rejects any exact
  completion or accepted candidate beyond the axis wall-clock budget. A
  legacy/quality boundary must not terminate valid constraint-lane work that
  remains within the overall axis budget.
- Local absolute paths live only in the ignored storage-root locator. Long-term
  publication identity records aliases, relative archive paths, file count,
  byte count, and tree SHA-256. Volume/filesystem/device observations may remain
  in raw operational telemetry for transfer planning, but device UUID, model,
  serial, and absolute path are excluded from publication identity.
  Batch target/hard cap remains 24/32 GiB and shard hard cap remains 2 GiB.
  Producer, retention, performance review, and campaign review must all use the
  shared cross-platform `probe_volume_identity`; WSL uses `findmnt`, DrvFS
  additionally binds the Windows physical-disk model, serial, and BusType, and
  macOS uses `diskutil`. Bound NTFS USB archives are allowed; ExFAT/FAT,
  disconnected devices, serial mismatch, and drive-letter drift are rejected.
  Reviewer-local platform probes are forbidden contract drift.
  Campaign rolling-capacity replay must derive its reserves from the rebuilt
  identity-matched `BenchmarkCampaignConfig`: WSL active/future staging uses
  50 GiB safety plus at least 32 GiB active workspace. Every Stage 0--8 attempt
  must submit a replayable plan before its run directory or workers exist.
  Dynamic stop gates require E free bytes of planned archive plus 0 GiB, D
  free bytes of projected WSL growth plus 0 GiB, and WSL free bytes of active
  workspace plus 50 GiB. A locked permit ledger prevents concurrent
  over-reservation; permits do not expire without audit. The first successful
  preflight creates the immutable lifecycle-bound start permit. Rolling-capacity
  checks may append observations and monotonically shrink the ledger reservation,
  but they must never rewrite that permit file or change its SHA-256; close and
  reconciliation continue to bind the original permit identity. Reviewer-local
  reserve constants or lower thresholds are forbidden contract drift.
- Rebuildable cache, venv, build, or temporary spool cleanup is allowed only
  from an exact allowlist after an independent keeper-reference scan. Every
  execution must match a signed dry-run identity. Venv/build cleanup also
  requires a signed isolated-rebuild proof plus a live rebuild verifier; missing
  inputs, active locks, manifest references, identity drift, or smoke-test
  mismatch retain the asset. Git repositories, sealed source, registries,
  manifests, reviews, checksums, active runs, and unsealed runs are never
  automatic-cleanup candidates.
  The sealed reviewer wheel must include and source-bind every tracked
  `tools` Python module used by review or publication dry-run paths; isolated
  `python -I` review services may not depend on an unsealed checkout import.
  `source_snapshot` is a mandatory common Pilot/Formal campaign review gate,
  including provisional Pilot publication; it may not exist only as an
  unregistered extra gate outside the exact gate-set contract.
  Current-chain performance reviews require their three-file generation with
  `semantic_mismatches.csv`; Benchmark campaign prerequisites instead require
  the complete content-addressed campaign publication surface. The generic
  verifier must not impose the performance-only mismatch filename on campaigns.
  A Benchmark campaign review must publish `selected_optimization_profile` at
  the top level as well as inside `selection_lock`; the next campaign loader
  rejects a review whose frozen top-level execution selection is incomplete.
- Formal review uses bounded Arrow batches and streaming iterators and rejects
  full-shard `to_pylist()`, `read_events()`, or `reconstruct_trace()`. It
  independently verifies exact campaign geometry, bidirectional descriptors,
  objective/validator and exact/cache/candidate/deadline semantics, resource
  and persistence gates, power/load/root/runtime provenance, BKS incompatibility,
  and the absence of gap columns.
- A wall-clock candidate transaction may commit only while its owning lane
  deadline and the solve-wide deadline are still open. This final check occurs
  after proposal, exact/cache work, optional shadow/constraint work, and the
  acceptance decision but before incumbent, global-best, statistics, or
  candidate-state mutation. Reaching the deadline at that boundary records a
  `before_candidate_commit` deadline event and rejects the transaction; neither
  timestamp clamping nor reviewer-side filtering is permitted.
- Benchmark campaign review uses one batch-scoped `spawn` process pool with the
  Pilot-selected 1/2/4 concurrency. Each child still executes exactly one shard
  with `max_tasks_per_child=1`; results merge by canonical shard ordinal, and
  the first child failure cancels every unfinished task with no fallback or
  retry. One logical event pass must jointly replay the async persistence
  ledger, exact/cache/deadline transactions, and global-best/checkpoint history.
  The parent receives only bounded JSON-safe summaries and telemetry carrying exact
  run/batch/shard/instance/seed identity; raw events, Arrow tables, route
  dictionaries, and screening definitions remain child-local. Child failure has
  no parent fallback or retry, and both success and failure write
  PID/RSS/event-count/single-pass/scratch-cleanup evidence to the external
  progress log. Residual scratch is removed and reported before failure returns.
- Retrospective review after a physical archive-disk replacement requires a
  signed storage-migration attestation binding the old/new volume identities,
  old/new physical-disk identities, campaign/raw manifest SHA-256 values, and
  every archived batch checksum and byte count. Only the attested
  historical archive mapping may differ; producer hard runtime, source revision
  and file hashes, ext4 capability, solver, backend, and scientific identities
  remain exact. Mount source, UUID, absolute path, and physical-disk fields stay
  as recorded telemetry. Reviewer source and the immutable producer source
  snapshot are separate explicit inputs and must
  remain separate in the finalized review execution receipt.
- A successor Benchmark Pilot required after a producer defect keeps the
  accepted accelerator Pilot as its scientific prerequisite. If that
  prerequisite froze the pre-migration archive disk, the successor must also
  supply the migration campaign directory explicitly. Before normalizing only
  the attested archive-disk field, the runner re-verifies that campaign's signed
  manifest, every archived batch checksum/byte count, accepted Pilot review,
  and finalized successful review receipt. The historical Benchmark review
  never replaces the accelerator prerequisite, and the attestation schema alone
  is insufficient.
  A producer-root-cause change outside the normal G path allowlist is permitted
  only through an explicit all-paths-present, current-blob-SHA-256-pinned
  exception in the successor verifier. Broad solver-directory allowlisting,
  partial fix sets, and any later byte drift remain hard failures.
- The independent reviewer for that successor receives the same explicit
  migration evidence directory. It re-verifies the historical migration
  campaign before using the payload for prerequisite/runtime replay; it must
  not compare the old attestation's run label or batch set to the successor
  campaign, and it must not trust the signed JSON without replaying the
  historical batches, review, and receipt.
- Stage 5.2 storage replay hashes canonical records as they are read. It must
  never accumulate a complete axis or bundle of event dictionaries. Multiple
  raw bundles are replayed strictly in input order, one fresh spawned process
  per bundle; the parent retains digest maps only. Field-level mismatch output
  is generated only for unequal fixed-work axes through a disk-backed temporary
  spool. Unequal wall-clock axes emit one aggregate digest row per
  `(instance, seed, axis)` instead of expanding expected trajectory differences.
  The spool stores one compressed canonical record per row and expands fields
  only while comparing; per-field database rows are forbidden because they
  amplify disk usage and cgroup page cache. Only the comparison bundle may be
  spooled: candidate records point-query comparison rows while left-only rows
  are derived from each axis's final ordinal tail; bulk DELETEs that dirty the
  SQLite file are forbidden. Per-axis fragments, the final mismatch CSV, and
  publication copies periodically fsync. Native Linux and WSL2 native-ext4
  paths release clean page cache with `POSIX_FADV_DONTNEED`; Windows and WSL2
  DrvFS/9p paths must not call that advisory because the formal host reproduced
  incorrect reads on those mounted Windows filesystems. SQLite construction
  commits/releases on a bounded
  record window; fragment release uses one aggregate byte window shared by
  all axes and the left-only tail. Raw manifest hashing and Parquet/JSONL
  iterators release their source page cache at file-lifecycle boundaries.
  BLOB temp sorts are forbidden. The mismatch CSV is
  streamed through temporary-file
  hashing and publication and is never accumulated as one in-memory payload.
- On the Windows/WSL2 formal host, every long-running calibration, producer,
  and review launch must first detach from the ChatGPT/Codex desktop through
  the registered Windows Scheduled Task host. The Windows wrapper owns
  `wsl.exe`; its Linux controller owns the campaign command or transient
  service, so closing or force-ending the desktop client cannot terminate an
  accepted launch. Every launch uses a fresh nonce, launcher mutex, Linux
  `flock`, durable progress log, and terminal receipt. A duplicate or stale
  launch fails before scientific work. Client-side monitoring is read-only:
  failures retain their immutable evidence and require root-cause review plus
  a new label; the detached host never edits code, retries, or falls back on
  its own. Host shutdown, reboot, explicit sleep/hibernate, `wsl --shutdown`,
  or manual task termination remain explicit external interruptions rather
  than lifecycle guarantees.
- Formal producer batches and Formal memory calibration must additionally run
  inside one dedicated transient `systemd --user` service cgroup. The aggregate
  hard gate reads cgroup v2 `memory.current`; `memory.peak` and
  `memory.swap.peak` are signed into resource v4 evidence. Summed process-tree
  RSS remains compatibility telemetry only because it double-counts shared
  mappings across spawned workers. Per-worker RSS remains an independent hard
  gate. `/`, `/init.scope`, a shared user service, missing cgroup files, or
  nonzero swap fail before readiness work; there is no process-RSS aggregate
  fallback.
- Resource recalibration report v2 remains readable as immutable Attempt99
  history. New v3 recalibration binds
  `stage05.2_benchmark_rerun02/batch0008`, treats that run's summed RSS only as
  failure provenance, and derives the replacement aggregate limit from the
  complete R205 cgroup measurement with 20% headroom. It retains the failed
  batch's per-worker RSS floor and exact predecessor physical topology
  (6 workers, 262,144-row groups, queue depth 2) as provenance. That historical
  topology is distinct from the new calibration contract's frozen 6-worker,
  16,384-row-group, queue-depth-1 topology; reviewer code must bind each to its
  own source and must not require those two physical configurations to be equal.
  Full calibration must reset and verify the dedicated service cgroup's
  `memory.peak` and `memory.swap.peak` immediately before the R205 measurement
  and retain a signed reset receipt; earlier worker/Parquet calibration phases
  must not contribute lifetime cgroup high-water marks to that scoped result.
  Once the Formal sampler returns, calibration must immediately seal
  `formal_memory_measurement.json` before contract/headroom validation, so a
  post-measurement failure still preserves the exact cgroup, per-worker, and
  swap evidence. A successful calibration report binds that artifact by
  SHA-256.
- On the Windows/WSL2 formal host, long-running Stage 5.2 reviewers must run as
  transient `systemd --user` services rather than Codex desktop child
  processes. Review workers, `MemoryHigh`, `MemoryMax`, `MemorySwapMax=0`, and
  the internal aggregate-RSS stop come from the signed Pilot
  review-calibration contract derived from parent baseline and per-child p99
  RSS. Formal never changes that worker count dynamically and has no
  restart/fallback. The service retains external progress logs and an
  `ExecStopPost`-sealed execution receipt.
  Formal launch rejects a dirty producer snapshot, arbitrary command, raw
  run-label mismatch, unsealed reviewer revision, or reviewer Python whose
  installed files do not match the declared frozen wheel. ExecStopPost must
  independently re-hash the raw manifest and read cgroup memory peaks. Missing
  cgroup peak accounting is a hard receipt failure and must never be silently
  represented as zero. Reviewer logs are operational evidence outside immutable
  raw bundles and do not alter the
  scientific review schema or readiness gates. The only allowlisted scientific
  entry points are `evrptw.experiments.stage052_performance_review` and
  `evrptw.experiments.stage052_campaign_review`; their raw/prerequisite command
  envelopes are validated separately and both use the external progress log
  plus the signed internal process-tree guard.
  The internal limit is not operator-configurable for formal review. Launch
  also requires the exact canonical signed raw manifest, a reviewer wheel whose
  tracked Python/native/build inputs match the declared clean revision and whose
  native-containing wheel matches a fresh no-cache rebuild byte-for-byte, and
  a receipt path owned by the transient service. A READY review is consumable
  only after `ExecStopPost` copies a finalized successful receipt, verified
  cgroup peaks, and the current review-manifest hash into `review/`.
  The launcher must resolve and freeze the service `PATH` for `nvidia-smi`,
  `powershell.exe`, and `wsl.exe`, record it in the execution receipt, and fail
  before launch if any required interoperability tool is unavailable; it must
  not rely on an interactive shell's inherited `PATH`.
  Producer runtime identity must be replayed by the raw-bound frozen producer
  venv, never by the new reviewer wheel. The review-only WSL memory cap is
  audited separately as operational receipt evidence. Same-revision campaign
  comparison excludes machine/mount telemetry and local absolute paths, but
  still hard-locks source revision, wheel, Python, native extension,
  dependencies, ABI, and all corresponding hashes. New producer
  identities use the locale-independent numeric CIM `OperatingSystemSKU`
  together with exact Version, BuildNumber, and TotalVisibleMemorySize; the
  localized CIM Caption is not an identity field. Historical Chinese and
  English captions for Windows 11 Pro for Workstations remain one
  locale-normalized reviewer identity. CPU/GPU/Windows/WSL/mount/NVMe and live
  memory details remain recorded telemetry rather than publication identity.
  A published `NOT_READY` review caused by reviewer/runtime defects must be
  archived byte-for-byte under `review/history/<manifest-sha256>/` before an
  explicit retry. Its hash belongs in `review_retry_history_sha256`, not the
  accepted-review lineage; every retry archive remains a prerequisite-time
  manifest/raw/file-hash gate and may never be deleted or silently replaced.
- The registry, trusted manifest, and content-addressed review products are
  published only after Formal raw replay reports `READY_FOR_STAGE05_3`. Publication is a
  generation transaction whose trusted manifest is replaced last; no producer,
  runner, or documentation may claim Stage 5.2 completion before both review
  and tracked publication pass.

The executable workflow and gate table are maintained in
`docs/stage052_performance_benchmark_workflow.md`.

## Experiment Artifact Storage Policy and v2 Transition

- All new Stage 0–8 runs must use an enabled `[artifact_storage]` configuration
  and the shared `evrptw.artifacts.ArtifactBundleWriter`/`ArtifactReader`; a
  canonical `attemptNN` or `rerunNN` label is mandatory. Runners must not
  duplicate JSON, event, checksum, or manifest persistence logic.
- The old non-canonical Stage 0–2 entry points remain only for historical
  compatibility tests/reproduction when their configuration has no
  `[artifact_storage]`; the shipped new configurations reject those paths.
- The current Stage 5.2 storage policy remains `artifact-storage-v2`, while the
  current physical schema is `screening_decisions_v3`; v1, old v2, v3,
  and legacy evidence remain readable. These formats use Parquet events,
  complete critical evidence, aggregated diagnostic evidence, 2 GiB per
  instance/seed, and 32 GiB per run. Historical v1 uses Zstandard level 3;
  accepted v2 uses Zstandard level 1. New physical evidence is created below
  the active staging root as `<run_label>/<instance>/<seed>/`, with control
  metadata and a manifest under `control/`, then moved intact to the configured
  archive after checksum verification.
- Stage 5.2 preserves v1/old-v2 reads and uses typed v3
  definitions/occurrences, bounded Parquet streams, shard
  manifests/checksums, worker-owned `(instance, seed)` shards, and parent-only
  control-manifest finalisation. Attempt-specific remediation history belongs
  in manifests, the retention registry, and the change log rather than this
  standing policy.
- Current v3 physical writers support signed calibration choices of 16,384,
  65,536, or 262,144 Parquet rows per group and buffer at most two non-empty
  row groups per family. The live trace buffer must use the selected physical
  row-group size; 16,384 is the explicit low-memory candidate and is not an
  implicit fallback. Once a Formal memory probe passes, full resource
  calibration must explicitly lock that exact row-group/queue-depth pair;
  persistence-only timing may not silently reselect a different pair. A worker writes only its own shard;
  the parent never merges event
  rows in memory. Shard-local event identity is deterministic from canonical
  shard ordinal plus local event ID, so review never depends on worker
  completion order. Semantic equality is required; byte-identical Parquet is
  not.
- v2 promotion requires v1/v2 replay equality for validator, objective,
  critical events, exact-call and failure semantics; artifact persistence must
  be at most 36% of end-to-end time and peak RSS at most 50% of the Stage 5.2
  v1 baseline. Partial shards are sealed with explicit completeness, fail
  immediately, and are then archived; there is no serial persistence fallback.
- Native producer and reviewer installations may live at different absolute venv paths; native
  identity is the sealed extension SHA-256 plus the captured runtime contract, never path equality.
  The v3 live screening path uses reusable precomputed typed definitions and occurrence rows;
  reintroducing per-decision dict normalization on that hot path is a performance regression.
  Deferred screening batches bind cache misses through the native transaction packer, but every
  canonical definition payload and SHA-256 identity still passes the Python collision store before
  publication; the optimization may not drop, aggregate, or reorder occurrences. High-cardinality
  definition, occurrence, and event Parquet streams disable dictionary encoding and statistics while
  retaining the canonical typed schema and Zstandard level 1.
  Repeated screening definitions use a shard-local native capsule with a hard 8,192-entry FIFO
  bound. Its composite hash is only a lookup accelerator: every hit must pass exact field equality,
  and every miss must retain an owning exact key before publishing the canonical definition identity.
  Canonical typed signatures distinguish booleans from numerics and preserve IEEE-754 signed zero;
  negative-evidence drift uses the same signature. Probabilistic hash-only identity, unbounded cache
  growth, Python-loose numeric equality, or collision aliasing is forbidden.
  The typed producer may bind a negative-cache occurrence to a positive integer marker. Historical
  unbounded paths use the identity of the still-live frozen `ScreeningResult`; the Stage 5.2 bounded
  LRU path derives the marker from the collision-free route key and complete normalized screening
  result so an equivalent eviction/recompute cycle remains stable. The marker is only a lookup
  accelerator: the sink must maintain a bounded route-key-to-marker-and-complete-compact-binary-
  typed-signature consistency map and fail fast if either differs. Full typed byte equality, not
  the truncated marker, establishes evidence consistency. The deferred native occurrence identity
  carries the same `(marker, complete signature)` pair, so route-guard eviction cannot expose a
  stale marker-only occurrence hit.
  This process-local accelerator is never persisted and is excluded from definition identity, event
  tokens, semantic digests, and replay. Only the trusted typed path may use it; manual and legacy rows
  continue through the complete field/signature collision check. The fixed negative-cache-hit check
  tuple is reused as one immutable value rather than reconstructed per occurrence.
  Sparse screening-definition transactions are persisted immediately after collision-store
  registration and do not enter the two-buffer high-volume working set. The two bounded non-empty
  Parquet buffers remain available to occurrence and event streams so a sparse definition sink
  cannot force repeated partial row-group rotation. This physical scheduling rule may not change,
  omit, aggregate, or reorder any definition, occurrence, or event row.
  Native screening and deferred sparse packers return the strictly decoded route sequence with every
  newly observed route ID, so the writer must not parse the same canonical route key a second time.
  Producer and campaign reviewer must validate the same exported writer thread-switch protocol
  constant; a reviewer-only hard-coded interval is forbidden contract drift.
  V3 route-evaluation and cache-event batches likewise use native schema-ordered sparse columns;
  mixed ordinary events are filled at their original positions and may not be reordered or dropped.
- A candidate-scoped exact result may observe a shared route-cache miss and then find that another
  lane committed the same key before candidate commit. This is valid only when every deterministic
  `ChargingSubproblemResult` field matches exactly after excluding `runtime_seconds`; the producer
  emits a `cache_event(operation="reconcile", reason="equivalent_existing")` with equal pending and
  existing result SHA-256 values and does not mutate cache statistics or LRU state. Any semantic
  difference is a hard conflict that reports both digests. The compact artifact writer and native
  sparse packer preserve the two digests in the current 20-field cache-event form while continuing
  to read the historical 18-field form, and independent replay verifies the existing cache key and
  digest equality. A failure before a bounded negative-cache batch begins must not attempt bounded
  rollback or mask the original route-cache error. After partial-shard evidence is persisted, the
  process-worker boundary raises only a string-backed pickle-safe error containing the original
  exception type and message; measured traces and asynchronous writer objects never cross that
  boundary.
- Producer screening-definition collision state uses the native
  `native_bounded_digest` store: it retains the full SHA-256 digest without a
  duplicate JSON payload, has a hard 2,097,152-entry limit, validates each batch
  atomically, and fails fast on collision or overflow. Its separate native
  definition-key memo is capped at 8,192 entries with FIFO safe recomputation;
  memo eviction never removes the complete 2,097,152-entry full-SHA-256
  collision state. Route-ID resolution plus route-evaluation/cache-event sparse
  extras JSON use a separate 8,192-entry FIFO safe-recomputation memo. Eviction
  from these three Python memos may only repeat deterministic route-key parsing
  or canonical JSON construction; it may not remove the disk-backed route
  identity, full digest/payload collision state, or any Parquet row. Signed
  metadata uses `stage05.2-screening-definition-store-v6` and binds both native
  limits, the recomputable-memo limit and eviction policies, and
  `identity_collision_proof=full_sha256`. The typed negative-evidence
  route guard also retains at most 8,192 complete `(token, signature)` values
  with FIFO safe recomputation. Evicting that guard may not weaken the full
  signature carried by the deferred occurrence or its downstream collision
  check. Producer SQLite
  spill, producer scratch state, and fallback are forbidden. Review/read stores
  retain the payload they must resolve and may use their separate bounded
  payload-retaining SQLite path. Exact-route evaluation route and unique-route
  identity stores retain at most one 65,536-row Parquet group each in memory,
  then spill exact full-digest/payload state to shard-local SQLite. Spill does
  not weaken duplicate or collision checks. Removing collision checks, making
  any store unbounded, or returning to per-event SQLite identity queries is
  forbidden.
- Process-tree resource identity includes only a process whose positive RSS and CPU times were both
  captured in one successful sample. A half-sampled, zero-RSS, or already-exited transient process
  is not measured worker evidence and must not be emitted with a fabricated zero peak.
- Native and benchmark trace persistence uses one non-daemon FIFO writer thread per open axis with
  a hard queue bound of one callback batch. Solver callbacks and the writer take cooperative,
  non-overlapping shard turns so Python/GIL contention cannot inflate writer wall time; bounded
  producer/writer overlap is permitted only during explicit finish/drain work and remains measured.
  Every finish, semantic digest, close, and shard finalization must drain the queue; writer
  failure aborts the shard without synchronous fallback. Trace evidence records submitted/completed
  batches, queue bound/peak, producer/writer wall time, writer thread CPU time, their overlapping wall
  union, producer wait time, and an ordered row-count/SHA-256 ledger for every batch. Formal
  persistence attribution uses the measured producer/writer wall-time union at the drained solver
  boundary, counting overlap once. The maximum of producer wall and writer CPU is diagnostic only;
  it must not replace the 36% persistence/end-to-end gate. Background hashing or writing may never
  be silently omitted or represented as zero. Review independently replays
  the ledger from logical events and rejects missing, incomplete, over-bound, or inconsistent pipeline
  evidence. These physical-pipeline fields are non-semantic for fixed-work algorithm comparison, so
  overlap is observable without fabricating a trajectory change.
- Critical events are never dropped. Ordinary candidates, repeated timings, and
  operator totals may be aggregated into diagnostic Parquet only when replay
  semantics are unchanged. Route sequences are stored once in the route
  dictionary. v1 events use global `event_id`; v2 events use canonical shard
  ordinal plus shard-local event ID. Both use integer route IDs.
- A cache lookup and its immediate hit/miss result are one persisted
  `lookup_result` event; the in-memory evaluator trace may retain the two
  callbacks for debugging, but storage and replay must count the logical lookup
  once. A missing failure artifact is represented explicitly as
  `artifact_status.failure=not_applicable` in the manifest.
- Event rows use integer route/lane/operator IDs; the trace index carries the
  lane/operator dictionaries and the route dictionary remains the sole store
  for complete customer sequences.
- A byte-budget violation must seal completed raw/solution/event/environment/
  failure evidence, write `evidence_completeness=partial`, update the manifest
  and sidecar, then fail immediately. The sealed run is checksum-verified and
  archived instead of accumulating in the workspace. Partial, timeout,
  failure, and manifest-error bundles cannot publish scientific summaries.
- Historical Stage 0 frozen artifacts and Stage 3.0–3.2 raw evidence are
  immutable and remain `legacy_json_or_jsonl`, `legacy`, and
  `legacy_compatible`. Compatibility mappings are registry views only; they do
  not copy, move, compress, or rewrite historical bytes. Recorded dirty states
  remain visible.
- Independent preflight/review must verify manifest first, then checksum, Arrow
  schema fingerprint, row count, byte size, critical-event completeness,
  validator/objective replay, deadline semantics, provenance, and raw-to-summary
  consistency. Tracked summaries and registry publication are downstream of
  that review.

## Stage 0--8 Experiment Lifecycle, Retention v3, and Storage Governance

- Every new Stage 0--8 top-level attempt must be registered in
  `configs/experiment_catalog.toml` and run through
  `ExperimentLifecycleController`. Its mandatory states are
  `PLANNED -> PERMITTED -> RUNNING -> SEALED -> REVIEWED -> CLASSIFIED ->
  RETAINED/COMPACTED -> CLOSED`. No later top-level run may be planned while
  an earlier record is not `CLOSED`; child shards, batches, and axes inherit
  their top-level identity and may run in parallel.
- A runner must obtain both the lifecycle-bound storage permit and `RUNNING`
  transition before creating its run directory or workers. Fixed reserves are
  E archive 0, D host 0, WSL staging safety 50 GiB, and Stage 5.2 active
  workspace floor 32 GiB. `e_archive` is the only new long-term archive target;
  `d_archive` is legacy-read-only and `d_host` measures host capacity only.
  The start permit is immutable after `PERMITTED`; later rolling-capacity
  observations update only the monotonic reservation projection and cannot
  replace the lifecycle-bound permit receipt.
- Retention v3 is rule-engine-only. Reviewer status, controlled failure code,
  exact failure identity, and signed adjudication determine the retention
  class. Unknown root cause or reference becomes `unknown_full` and
  `BLOCKED_RETENTION`; an agent or runner may not choose a deletion list or
  retention class directly.
- In-place compaction accepts only the controller-produced signed plan and its
  exact SHA-256. It records PREPARED/APPLYING/COMMITTED state, preserves the
  original manifest, appends deleted-file identities to the ledger, and allows
  at most one producer-side content-hash pass and one apply traversal. The
  signed incremental content inventory is immutable input: added files,
  symlinks, missing files, or stat/content drift abort the transaction. An
  explicit lifecycle close plus its signed matching plan is execution
  authorization; no second manual deletion confirmation is required.
- A full-retention receipt must exactly match the signed content inventory's
  tree hash, file count, and byte count. Closing a new
  `current_accepted_full` record atomically supersedes and compacts the prior
  current run for the same experiment. The predecessor's original `CLOSED`
  record is immutable; a signed append-only supersession transaction records
  the effective `superseded_accepted_capsule` projection and `superseded_by`
  identity. Interrupted supersession resumes from that signed checkpoint.
  The retained archive may be either the canonical `generation-NNNN` root or
  one canonical single-segment leaf below it (for example `wsl_active`); both
  shapes remain bound to the exact signed inventory and use the same
  controller-authored compaction transaction. Supersession planning rebinds
  file timestamps to the already content-verified archive filesystem; legacy
  plans may accept only same-UTC-second timestamp truncation while still
  requiring exact byte counts and SHA-256 for every file.
- Runtime source/config/environment and worker/thread/process values must match
  the immutable lifecycle plan again immediately before execution. An
  idempotent retry with any runtime-plan drift is a hard failure rather than a
  new implicit experiment identity.
- `experiments/registries/experiment_lifecycle_v3_migration.json` is the signed
  compatibility anchor: v1/v2 registries remain read-only inputs and lifecycle
  v3 is the sole schema for new governance facts.
- Review promotion requires a signed `review_execution.json` from the catalogued
  reviewer module, binding the raw manifest before/after digest and the emitted
  review manifest. Historical v2 runs additionally require a one-pass per-file
  content inventory and a signed stage-specific semantic disposition; inventory
  generation alone remains `INVALID`/`unknown_full` and blocks Stage 5.2.
- Historical semantic review may execute only the reviewer module allowlisted in
  the catalog and must bind its module hash, exact command, raw input hashes,
  signed output, and execution receipt. A `superseded_metadata` disposition also
  requires a signed no-dependency proof recomputed from the migration-ledger-
  bound v2 retention registry and the SHA-bound dependency-document closure for
  every canonical protected keeper generation; caller-supplied keeper paths or
  hand-written proof payloads are not trusted. The complete semantic command,
  inventory, execution receipt, binding and aggregate gate must all carry the
  same physical-review-bound canonical `e_archive` root and generation identity.
  The physical review itself must resolve `e_archive` through the repository's
  local storage-root locator, verify the live volume identity, and use the unique
  v2 registry relative path and SHA declared by the signed migration ledger;
  registry SHA and generation tree SHA remain bound through the final gate. A
  gate consumer must re-verify the live locator root and the signed review,
  inventory, semantic output, execution receipt and execution binding; a
  hand-written gate bundle is never sufficient. The consumed semantic status,
  retention class, failure identity, canonical representative, no-dependency
  proof and rebuild proof must satisfy the same class-specific rules as the
  original adjudication.
- Pre-lifecycle historical compaction is a separate controller transaction. It
  may prepare a plan only from a `complete` aggregate historical gate, the
  gate-bound canonical `e_archive` generation, and the exact signed inventory
  and review identities. `PREPARED` creates no lifecycle run record and deletes
  nothing. Apply requires the canonical plan path plus its exact SHA-256,
  replays the gate and legacy pretty-canonical tree digest, rejects any active
  writer or file/stat/content drift, imports one `CLASSIFIED` record, reuses the
  single-pass compaction engine, and must finish with signed compaction/import/
  close receipts and `CLOSED`. Only `superseded_metadata` and
  `superseded_accepted_capsule` are eligible; the v2 registry remains immutable
  historical input and the append-only v3 ledger records every removed file.
- Historical command replay validates the signed command's existing absolute
  Python executable, `-m` invocation, allowlisted reviewer module, complete
  option set, and downstream hashes. It must not require the historical
  reviewer executable to equal the current producer's `sys.executable`, because
  producer and reviewer runtimes are independently frozen. Likewise, a sealed
  source relocation may change the migration-ledger absolute path only when the
  historical path still exists and both paths have the exact signed SHA-256;
  inventory, output, archive, module, and all other command bindings remain
  path-exact.
  `hot_path_attempt04` may be classified as `superseded_accepted_capsule` only
  when the stage-specific reviewer replays its unchanged raw-manifest identity,
  the complete accepted-review lineage and failed-review retry set, the terminal
  review execution and exact failed gates, plus the v2-registry-bound full tree,
  accepted review, and finalized review execution of successor
  `hot_path_attempt06`. The resulting supersession proof must remain bound
  through semantic execution, physical review, aggregate gate, and gate
  consumption; otherwise the run remains `INVALID`/`unknown_full`.
- Every catalogued producer's successful CLI path must release its writer lease
  and call the shared lifecycle seal. Stage 5.2 benchmark uses the final
  `stage05.2-campaign-manifest-v3` as its sealed manifest; calibration tools
  aggregate signed child inventories without rehashing raw artifacts. Successful
  and failed sealing, terminal child verification, and archive close each hold
  the run's exclusive writer lease through the state commit; failures before a
  producer creates its directory use the exact signed plan path and seal an
  explicitly untrusted failure capsule. Internal lifecycle state uses
  `.json.sha256`, while external artifact/reviewer evidence may use the existing
  `.sha256` convention and is validated separately.
- The verified Stage 5.2 historical migration
  `stage052-retention-v2-20260731` published 307 runs / 356 segments /
  712,267,368,027 source bytes to the bound NTFS USB `e_archive`. The v2
  registry, migration attestations, resolver replay receipt, deletion manifest,
  and deletion execution receipt under the E archive are the immutable storage
  evidence for that migration.
- After literal operator confirmation, the 356 manifest-listed D/WSL source
  directories were deleted and independently verified absent. Those historical
  source paths must not be reconstructed as parallel archives or treated as
  current evidence roots. New active work still stages on ext4 and publishes a
  new immutable E generation through `evrptw.storage_governance`.
- E is the sole long-term media copy for the migrated raw evidence. SHA-256,
  attestation replay, and resolver verification establish content identity;
  they do not constitute a backup or provide recovery from E-device failure.

## Literature Recommendation Policy

- Codex recommends literature but does not obtain it. Do not access the user's
  university library account, publisher subscription, or institutional proxy,
  and do not download article PDFs.
- Recommend only formally published journal articles. Prioritise work from
  leading universities or research institutes and, when otherwise comparable,
  papers with higher citation impact.
- Verify that each recommendation has a formal journal record and that its title
  and DOI agree with the publisher or another authoritative bibliographic
  source. Do not recommend a technical report, working paper, preprint,
  postprint, accepted manuscript, or author manuscript as a journal article.
- Return literature recommendations only in the conversation as a Markdown
  table with exactly two columns: `Paper title` and `DOI`. Do not create or
  update repository files merely to store a recommendation list.
- The user is responsible for locating and downloading every recommended PDF.
  If a formal PDF is unavailable, provide no substitute file and wait for the
  user to supply the Version of Record (VOR).

## User-Supplied PDF Policy

- Only handle a literature PDF after the user has explicitly supplied it or
  identified its local path.
- Before accepting it as a VOR, verify the title, authors, journal, publication
  year, volume/issue, page range or article number, DOI, and visible publisher
  or journal version markers.
- Reject and remove from the formal literature collection any technical report,
  working paper, preprint, postprint, accepted manuscript, peer-reviewed
  manuscript, or author manuscript.
- Store verified PDFs only in the local Git-ignored literature directory. Use a
  filename containing the authors, year, and journal, and update the local
  literature index and SHA-256 checksums.
- Never commit journal PDFs, licence-protected files, access-controlled copies,
  or the local literature shortcut to GitHub.
