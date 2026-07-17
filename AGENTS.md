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
  Smoke `stage04_adaptive_weights_attempt12` passed 72/72 axes under v5; its
  original ordering-sensitive review is preserved and the corrected independent
  `review_v2` is the accepted smoke review. Formal
  `stage04_adaptive_weights_attempt13` passed 144/144 axes and all six gates
  under the same v5 reviewer, publishing `READY_FOR_STAGE05`.
- Stage 4 review products are tracked under `experiments/summaries/` with
  `stage04_adaptive_weights_attempt13_` prefix; the artifact registry is
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
- Stage 5.1 v5 requires the published Stage 4 v5 prerequisite, canonical
  single-level run layout, exact CSV schemas, unique instance rows, and
  complete field-by-field replay including citations, charging unknowns, and
  model-compatibility fields. `stage05.1_best_known_attempt01` remains
  preserved v1 evidence and is superseded for readiness purposes.
- `stage05.1_best_known_attempt02` remains preserved v2 evidence with a
  canonical single-level 92-instance bundle, but it is superseded by the v3
  prerequisite and independent-conversion replay requirements. No current
  Stage 5.1 readiness may be published until a new attempt passes after Stage 4
  v3 is republished. Attempt01 remains historical nested-layout evidence. The
  v3 evidence `stage05.1_best_known_attempt03` is preserved but superseded
  after the Stage 4 v3 prerequisite was withdrawn. v4 attempt04 is also
  preserved but superseded after the Stage 4 v4 prerequisite was withdrawn.
  `stage05.1_best_known_attempt05` uses the published v5 Stage 4 Formal
  prerequisite and independently replays exactly 92 canonical rows, passing
  all five gates and publishing `READY_FOR_STAGE05_2`.
- The artifact registry is
  `experiments/registries/stage05.1_artifact_registry.csv` and the manifest
  is `experiments/manifests/stage05.1_best_known_artifact_manifest.json`.

## Experiment Artifact Storage v2

- All new Stage 0–8 runs must use an enabled `[artifact_storage]` configuration
  and the shared `evrptw.artifacts.ArtifactBundleWriter`/`ArtifactReader`; a
  canonical `attemptNN` or `rerunNN` label is mandatory. Runners must not
  duplicate JSON, event, checksum, or manifest persistence logic.
- The old non-canonical Stage 0–2 entry points remain only for historical
  compatibility tests/reproduction when their configuration has no
  `[artifact_storage]`; the shipped new configurations reject those paths.
- The current policy is `artifact-storage-v1`: Parquet events with Zstandard
  level 3, complete critical evidence, aggregated diagnostic evidence, 2 GiB
  per instance/seed, and 32 GiB per run. New physical evidence belongs under
  `results/<run_label>/<instance>/<seed>/` with control metadata and a manifest
  under `control/`.
- Critical events are never dropped. Ordinary candidates, repeated timings, and
  operator totals may be aggregated into diagnostic Parquet only when replay
  semantics are unchanged. Route sequences are stored once in the route
  dictionary; events use global `event_id` and integer route IDs.
- A cache lookup and its immediate hit/miss result are one persisted
  `lookup_result` event; the in-memory evaluator trace may retain the two
  callbacks for debugging, but storage and replay must count the logical lookup
  once. A missing failure artifact is represented explicitly as
  `artifact_status.failure=not_applicable` in the manifest.
- Event rows use integer route/lane/operator IDs; the trace index carries the
  lane/operator dictionaries and the route dictionary remains the sole store
  for complete customer sequences.
- A byte-budget violation must retain completed raw/solution/event/environment/
  failure evidence, write `evidence_completeness=partial`, update the manifest
  and sidecar, then fail immediately. Partial, timeout, failure, and manifest
  error bundles cannot publish summaries.
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
