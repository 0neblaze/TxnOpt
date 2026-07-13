# stage02_constraint_guided_rerun04

Final review status: **READY_FOR_STAGE03**

The review re-read raw solution JSON and raw operator events, then replayed the unified validator.

## Findings

- `run_coverage_unique`: **pass** — observed=36 rows, 36 unique keys; expected=36 complete unique (instance, seed) keys.
- `raw_solution_validator`: **pass** — observed=36/36; expected=36/36 raw solutions pass unified validator.
- `raw_objective_recomputation`: **pass** — observed=36/36; expected=validator objective equals solver objective for every raw solution.
- `raw_solution_artifacts_complete`: **pass** — observed=36/36; expected=raw JSON and solution JSON for every run.
- `operator_event_artifacts_complete`: **pass** — observed=event_csv=1, failure_csv=1, event_rows=203091, failure_rows=167835; expected=one operator event CSV and one failure event CSV.
- `environment_and_manifest_complete`: **pass** — observed=environment=True, manifest=1, manifest_integrity=True; expected=environment JSON and checksum-valid run manifest.
- `source_config_instance_hash_consistency`: **pass** — observed=consistent; expected=source/config/instance/environment hashes agree.
- `stage00_manifest_unchanged`: **pass** — observed=b226b97e0e67288aaaf85726ad855df71cb81406685c57c8e8c40cd8996aa0da; expected=current Stage 0 manifest SHA-256.
- `constraint_level_failure_diagnosis`: **pass** — observed=20; expected=at least 3 real raw failure or poor-quality cases.
- `constraint_event_replay`: **pass** — observed={'operators': {'station_pressure': {'called': True, 'feasible': True, 'accepted': True}, 'time_window_conflict': {'called': True, 'feasible': True, 'accepted': True}, 'worst_energy_detour': {'called': True, 'feasible': True, 'accepted': True}, 'shaw_related': {'called': True, 'feasible': True, 'accepted': True}}, 'accepted_vehicle_increase': False, 'observed_tiers': ['large', 'medium', 'small'], 'focused_actual_max': 20, 'stagnation_escalations': 17864, 'global_best_resets': 146, 'invalid_dynamic_counts': 0}; expected=raw event CSV independently satisfies Stage 2.3 event gates.
- `stage03_readiness_completeness`: **pass** — observed=36/36 rows; expected=one complete Stage 2.2 versus Stage 2.3 row per run with recorded metrics.
- `runner_gate_report_complete`: **pass** — observed=present; expected=exactly one runner gate report is present.
- `independent_complete_rerun_replay`: **pass** — observed={'status': 'pass', 'first_raw_revalidated': True, 'first_gate_pass': True, 'second_gate_pass': True, 'configuration_match': True, 'first_event_replay': True, 'first_event_observed': {'operators': {'station_pressure': {'called': True, 'feasible': True, 'accepted': True}, 'time_window_conflict': {'called': True, 'feasible': True, 'accepted': True}, 'worst_energy_detour': {'called': True, 'feasible': True, 'accepted': True}, 'shaw_related': {'called': True, 'feasible': True, 'accepted': True}}, 'accepted_vehicle_increase': False, 'observed_tiers': ['large', 'medium', 'small'], 'focused_actual_max': 20, 'stagnation_escalations': 17864, 'global_best_resets': 146, 'invalid_dynamic_counts': 0}, 'first_hard_replay': True, 'first_hard_observed': {'coverage_ok': True, 'validator_ok': True, 'objective_recomputation_ok': True, 'objective_comparison_ok': True, 'objective_observed': {'c101C5': 'equal', 'r105C5': 'equal', 'rc105C5': 'equal', 'c104C10': 'equal', 'r103C10': 'equal', 'rc102C10': 'equal', 'c106C15': 'equal', 'r105C15': 'equal', 'rc103C15': 'equal', 'c101_21': 'equal', 'r101_21': 'equal', 'rc101_21': 'equal'}, 'focused_ok': True, 'focused_observed': {'r101_21': {'mean_vehicle_ok': True, 'distance_guard_ok': True, 'vehicle_stability_ok': True}, 'rc101_21': {'mean_vehicle_ok': True, 'distance_guard_ok': True, 'vehicle_stability_ok': True}}, 'route_elimination_ok': True, 'route_merge_ok': True, 'quality_coverage': {'relocate': True, 'swap': True, 'two_opt_star': True, 'route_segment_destroy': True, 'ejection_chain': True}, 'quality_same_vehicle_distance_improvement': True, 'constraint_event_replay_ok': True, 'stage00_ok': True, 'independent_complete_rerun_ok': True}, 'second_event_replay': True, 'second_hard_replay': True, 'first_artifacts_complete': True, 'second_artifacts_complete': True, 'first_hashes_consistent': True, 'second_hashes_consistent': True, 'first_manifest_integrity': True, 'second_manifest_integrity': True}; expected=repeatability evidence and both complete runs are independently verified.
- `runner_hard_gates`: **pass** — observed={'coverage_ok': True, 'validator_ok': True, 'objective_recomputation_ok': True, 'objective_comparison_ok': True, 'objective_observed': {'c101C5': 'equal', 'r105C5': 'equal', 'rc105C5': 'equal', 'c104C10': 'equal', 'r103C10': 'equal', 'rc102C10': 'equal', 'c106C15': 'equal', 'r105C15': 'equal', 'rc103C15': 'equal', 'c101_21': 'equal', 'r101_21': 'equal', 'rc101_21': 'equal'}, 'focused_ok': True, 'focused_observed': {'r101_21': {'mean_vehicle_ok': True, 'distance_guard_ok': True, 'vehicle_stability_ok': True}, 'rc101_21': {'mean_vehicle_ok': True, 'distance_guard_ok': True, 'vehicle_stability_ok': True}}, 'route_elimination_ok': True, 'route_merge_ok': True, 'quality_coverage': {'relocate': True, 'swap': True, 'two_opt_star': True, 'route_segment_destroy': True, 'ejection_chain': True}, 'quality_same_vehicle_distance_improvement': True, 'constraint_event_replay_ok': True, 'stage00_ok': True, 'independent_complete_rerun_ok': True}; expected=independent replay of all Stage 2.3 hard gates passes.

## Failure analysis coverage

Retained real failure/poor-quality cases: 20.
Constraint categories are derived from raw event reasons; no synthetic cases are added.

## Stage 3 entry targets (not claimed as completed in Stage 2.3)

- 100-customer R/RC median exact charging calls ≤ 100.
- 100-customer R/RC median effective iterations ≥ 50.
- Validator feasibility remains 100%.
- The formal objective does not regress under acceleration.
