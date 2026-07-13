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

- This repository currently has completed evidence only for `stage03.0` with
  component `measurement` and `stage03.1` with component `screening`.
  `stage03.2`, `stage03.3`, and `stage03.4` are planned stages only; do not
  create placeholder results, registries, or readiness claims for them.
- Canonical run labels are `stage03.0_measurement_attemptNN` or
  `stage03.1_screening_attemptNN`. Every artifact registry row records the
  canonical `run_label`, `attempt_or_rerun`, `artifact_type`, instance/seed
  scope, checksum, status, provenance hashes, validator status, comparison
  baseline, and `legacy_path` mapping.
- The canonical logical layout is
  `results/<run_label>/<instance>/<seed>/`. Canonical filenames follow
  `<stage_id>_<component>_<attempt_or_rerun>_<artifact_type>[_<instance>_<seed>].<ext>`.
  The current raw evidence remains physically stored under its original
  ignored `results/stage03-*` paths; the canonical paths are registry views and
  must not be implemented by copying or moving raw evidence.
- `experiments/registries/stage03.0_artifact_registry.csv` and
  `experiments/registries/stage03.1_artifact_registry.csv` are the stage
  registries. `experiments/registries/stage03_legacy_path_map.csv` records the
  preserved old Stage 3 paths, the immutable Stage 0 frozen baseline, and the
  Stage 2.3 historical comparison references.
- The corresponding manifests are
  `experiments/manifests/stage03.0_measurement_artifact_manifest.json` and
  `experiments/manifests/stage03.1_screening_artifact_manifest.json`. Before a
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
  `READY_FOR_STAGE03_2`. No Stage 3.2 cache, incremental propagation,
  interruptible exact solver, parallel evaluation, or fixed-work/wall-clock
  acceleration may be implemented under this policy.

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
