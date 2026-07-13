# stage02_constraint_guided_attempt13

Final review status: **PENDING_INDEPENDENT_RERUN**

The review re-read raw solution JSON and raw operator events, then replayed the unified validator.

## Findings

- `run_coverage_unique`: **pass** — observed=36 rows, 36 unique keys; expected=36 complete unique (instance, seed) keys.
- `raw_solution_validator`: **pass** — observed=36/36; expected=36/36 raw solutions pass unified validator.
- `raw_objective_recomputation`: **pass** — observed=36/36; expected=validator objective equals solver objective for every raw solution.
- `raw_solution_artifacts_complete`: **pass** — observed=36/36; expected=raw JSON and solution JSON for every run.
- `operator_event_artifacts_complete`: **pass** — observed=event_csv=1, failure_csv=1, event_rows=231314, failure_rows=196037; expected=one operator event CSV and one failure event CSV.
- `environment_and_manifest_complete`: **pass** — observed=environment=True, manifest=1, manifest_integrity=True; expected=environment JSON and checksum-valid run manifest.
- `source_config_instance_hash_consistency`: **pass** — observed=consistent; expected=source/config/instance/environment hashes agree.
- `stage00_manifest_unchanged`: **pass** — observed=b226b97e0e67288aaaf85726ad855df71cb81406685c57c8e8c40cd8996aa0da; expected=current Stage 0 manifest SHA-256.
- `constraint_level_failure_diagnosis`: **pass** — observed=20; expected=at least 3 real raw failure or poor-quality cases.
- `constraint_event_replay`: **pass** — observed={'operators': {'station_pressure': {'called': True, 'feasible': True, 'accepted': True}, 'time_window_conflict': {'called': True, 'feasible': True, 'accepted': True}, 'worst_energy_detour': {'called': True, 'feasible': True, 'accepted': True}, 'shaw_related': {'called': True, 'feasible': True, 'accepted': True}}, 'accepted_vehicle_increase': False, 'observed_tiers': ['large', 'medium', 'small'], 'focused_actual_max': 20, 'stagnation_escalations': 17898, 'global_best_resets': 144, 'invalid_dynamic_counts': 0}; expected=raw event CSV independently satisfies Stage 2.3 event gates.
- `stage02_2_metric_supplement_provenance`: **pass** — observed={'coverage_ok': True, 'protocol_ok': True, 'protocol': {'algorithm': ['ALNS_STAGE02_ROUTE_QUALITY'], 'operator_profile': ['stage02_route_quality'], 'run_label': ['stage02_quality_instrumented_attempt01'], 'time_limit_seconds': ['30.0'], 'max_iterations': ['1000'], 'threads': ['1'], 'objective_schema': ['vehicles,distance,charging_time,charging_count']}, 'source_hashes': ['0686b3b522be957d2a13abff56a8cd9e788ff4923dbce3238ab86c0ff70982ee'], 'repository_revisions': ['6870b3125e9170a47246ec7333dbe382b7a0d854'], 'instance_hashes_ok': True, 'objectives_match_fixed_baseline': True, 'recorded_metrics_and_feasibility_ok': True}; expected=the fixed Stage 2.2 metric supplement records the formal protocol, source/instance hashes, feasible objectives, and complete coverage.
- `stage03_readiness_completeness`: **pass** — observed=36/36 rows; expected=one complete Stage 2.2 versus Stage 2.3 row per run with recorded metrics.
- `runner_gate_report_complete`: **pass** — observed=present; expected=exactly one runner gate report is present.
- `independent_complete_rerun_replay`: **pending** — observed={'status': 'first_complete_run_pending_independent_rerun', 'current_run_keys': 36}; expected=repeatability evidence and both complete runs are independently verified.
- `runner_hard_gates`: **pending** — observed={'coverage_ok': True, 'validator_ok': True, 'objective_recomputation_ok': True, 'objective_comparison_ok': True, 'objective_observed': {'c101C5': 'equal', 'r105C5': 'equal', 'rc105C5': 'equal', 'c104C10': 'equal', 'r103C10': 'equal', 'rc102C10': 'equal', 'c106C15': 'equal', 'r105C15': 'equal', 'rc103C15': 'equal', 'c101_21': 'equal', 'r101_21': 'equal', 'rc101_21': 'equal'}, 'focused_ok': True, 'focused_observed': {'r101_21': {'mean_vehicle_ok': True, 'distance_guard_ok': True, 'vehicle_stability_ok': True}, 'rc101_21': {'mean_vehicle_ok': True, 'distance_guard_ok': True, 'vehicle_stability_ok': True}}, 'route_elimination_ok': True, 'route_merge_ok': True, 'quality_coverage': {'relocate': True, 'swap': True, 'two_opt_star': True, 'route_segment_destroy': True, 'ejection_chain': True}, 'quality_same_vehicle_distance_improvement': True, 'constraint_event_replay_ok': True, 'stage00_ok': True, 'independent_complete_rerun_ok': True}; expected=independent replay of all Stage 2.3 hard gates passes.

## Failure analysis coverage

Retained real failure/poor-quality cases: 20.
Constraint categories are derived from raw event reasons; no synthetic cases are added.

## Stage 3 entry targets (not claimed as completed in Stage 2.3)

- 100-customer R/RC median exact charging calls ≤ 100.
- 100-customer R/RC median effective iterations ≥ 50.
- Validator feasibility remains 100%.
- The formal objective does not regress under acceleration.
