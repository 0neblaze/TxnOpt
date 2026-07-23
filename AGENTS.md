# Repository Instructions

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
- All objective construction and comparison must use `evrptw.objective`; callers
  must not duplicate tuple construction, station-visit counting, or comparison
  logic.
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
  not version names. A failed or interrupted run consumes its label and shards
  may never be imported into another run, but bulky evidence need not remain in
  the repository workspace. Current-chain truth comes from signed manifests,
  prerequisite references, and `stage05.2_retention_registry.csv`, never from a
  hard-coded attempt number in policy documentation.
- Full Stage 5.2 evidence exists in the active staging root only while it is
  being produced or reviewed. Complete, partial, failed, `NOT_READY`, and
  superseded runs are checksum-verified and archived under the configured
  `d_archive` alias at `stage05.2/history/<run_label>/`; the workspace retains
  only the signed retention inventory, lightweight registry, change log, and
  independently published summaries. Archive failure or identity mismatch
  retains the source and fails immediately.
- Retention audit rejects active, planned, unknown, or otherwise unsealed runs
  by default. The one-time pre-redesign historical override must bind the
  expected directory count and byte count. Same-volume archival uses atomic
  rename; cross-volume archival copies to a hidden target-volume temporary
  directory, verifies the full identity, atomically publishes it on that
  volume, and only then removes the source. Registry updates merge by immutable
  run identity and may never replace unrelated historical rows.
- Archived prerequisite or review input is resolved by run label through
  `stage05.2_retention_registry.csv` plus the local storage-root locator. The
  resolver must recheck the registered file count, byte count, and tree SHA-256
  before returning a path to an existing runner or reviewer; policy files and
  callers must not embed the machine-local archive path.
- A registered archive tree is immutable. Reviewers may consume it as a
  comparison, prerequisite, or replay input, but may not publish a new review
  generation inside it. A run that still needs review publication remains in
  the active root until that generation is sealed, then it is archived.
- `artifact-storage-v2` with physical schema `screening_decisions_v3` uses
  typed bounded Parquet streams, 65,536-row groups, at most two non-empty buffer
  groups, and compatible v1/old-v2/v3/legacy reads. Canonical semantic digests
  are computed over expanded logical events, so physical IDs and compression
  layout cannot change replay. Persistence is at most 36% of end-to-end time
  and peak RSS is at most 50% of the Stage 5.2 v1 baseline.
- Job-parallel selection compares 1/2/4 workers on the fixed four-instance,
  three-seed scope. Two workers require at least 1.5x speedup and at most 12 GiB
  aggregate RSS; four workers are selected only at 2.5x and at most 12 GiB.
  Worker ownership uses real PIDs and cumulative process CPU samples; worker
  failure aborts without fallback.
- Native-kernel promotion must preserve Python/native objective, validator,
  exact ordering, candidate/cache/event semantics, and zero fallback across all
  fixed-work axes. Aggregate 100-customer paired median end-to-end improvement
  must be at least 15%, no C/R/RC family may regress by more than 3%,
  persistence must remain at most 36%, each worker RSS at most 4,357,382,144
  bytes, and process-tree RSS at most 12 GiB.
- The accelerator gate independently recomputes 100-customer batch occupancy.
  Median below 32 publishes `GPU_NOT_JUSTIFIED`; median at least 32 requires the
  registered helper, exact fixed-work equality, at least 15% aggregate
  improvement, and no family regression over 3%. Missing helper, fallback, or
  an unaudited campaign adapter is `NOT_READY`.
- The pipeline pilot is exactly 12 instances x 3 seeds x one 30-second axis and
  exercises resource sampling, failure recovery, bounded replay, 1/5/10/30
  second anytime checkpoints, configured archive roots, and interrupted
  publication recovery. Formal contains exactly 920 indivisible
  `(instance, seed)` shards, 2,040 runs, 229,200 declared solver seconds, and
  10,400 anytime rows. Only independent review may open Formal or report
  `READY_FOR_STAGE05_3`.
- Local absolute paths live only in the ignored storage-root locator. Tracked
  evidence records aliases, relative archive paths, volume identities, run
  status, source revision, prerequisite identities, byte count, and checksum.
  Batch target/hard cap remains 24/32 GiB and shard hard cap remains 2 GiB.
- Formal review uses bounded Arrow batches and streaming iterators and rejects
  full-shard `to_pylist()`, `read_events()`, or `reconstruct_trace()`. It
  independently verifies exact campaign geometry, bidirectional descriptors,
  objective/validator and exact/cache/candidate/deadline semantics, resource
  and persistence gates, power/load/root/runtime provenance, BKS incompatibility,
  and the absence of gap columns.
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
  publication copies periodically fsync and release clean page cache with
  `POSIX_FADV_DONTNEED`. SQLite construction commits/releases on a bounded
  record window; fragment release uses one aggregate byte window shared by
  all axes and the left-only tail. Raw manifest hashing and Parquet/JSONL
  iterators release their source page cache at file-lifecycle boundaries.
  BLOB temp sorts are forbidden. The mismatch CSV is
  streamed through temporary-file
  hashing and publication and is never accumulated as one in-memory payload.
- On the Windows/WSL2 formal host, long-running Stage 5.2 reviewers must run as
  transient `systemd --user` services rather than Codex desktop child
  processes. The service uses `MemoryHigh=5G`, `MemoryMax=6G`,
  `MemorySwapMax=2G`, no restart/fallback, an internal 5.5-GiB aggregate-RSS
  stop, external progress logs, and an `ExecStopPost`-sealed execution receipt.
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
  plus the 5.5-GiB internal process-tree guard.
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
  audited separately as operational receipt evidence; only that live memory
  field may differ from the historical producer identity. The exact Chinese and
  English CIM captions for Windows 11 Pro for Workstations are one locale-normalized
  edition identity; every other Windows caption, CPU, GPU,
  Windows, WSL, mount, NVMe, wheel, Python, native, and dependency identities
  remain exact hard gates.
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
- Current v3 physical writers use 65,536-row Parquet row groups and buffer at
  most two non-empty row groups per family. A worker writes only its own shard;
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
  Native screening and deferred sparse packers return the strictly decoded route sequence with every
  newly observed route ID, so the writer must not parse the same canonical route key a second time.
  Producer and campaign reviewer must validate the same exported writer thread-switch protocol
  constant; a reviewer-only hard-coded interval is forbidden contract drift.
  V3 route-evaluation and cache-event batches likewise use native schema-ordered sparse columns;
  mixed ordinary events are filled at their original positions and may not be reordered or dropped.
- Producer screening-definition collision state stores the full SHA-256 digest without retaining a
  duplicate JSON payload; review/read stores retain the payload they must resolve. Exact-route
  evaluation identities likewise remain in a bounded in-memory full-digest/payload store and spill
  to shard-local SQLite only after the declared hard limit. Removing collision checks, making either
  store unbounded, or returning to per-event SQLite identity queries is forbidden.
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
