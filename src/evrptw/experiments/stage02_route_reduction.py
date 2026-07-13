from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

from evrptw.alns import ALNSResult, solve_alns
from evrptw.environment import collect_environment
from evrptw.experiments.stage00_baseline import (
    load_config as load_stage00_config,
)
from evrptw.experiments.stage00_baseline import (
    verify_results,
)
from evrptw.models import Instance
from evrptw.neighborhoods import OperatorProfile, VehicleOperatorConfig
from evrptw.objective import (
    ObjectiveComparison,
    SolutionObjective,
    compare_objectives,
    count_charging_visits,
)
from evrptw.parser import parse_schneider
from evrptw.validation import SolutionReport, validate_routes

SCHEMA_VERSION = "1"
ROUTE_REDUCTION_ALGORITHM = "ALNS_STAGE02_ROUTE_REDUCTION"
ROUTE_QUALITY_ALGORITHM = "ALNS_STAGE02_ROUTE_QUALITY"
CONSTRAINT_GUIDED_ALGORITHM = "ALNS_STAGE02_CONSTRAINT_GUIDED"
# Kept as the historical public constant for Stage 2.1 callers.
ALGORITHM = ROUTE_REDUCTION_ALGORITHM
FOCUSED_100_INSTANCES = ("c101_21", "r101_21", "rc101_21")
FOCUSED_R_RC_INSTANCES = ("r101_21", "rc101_21")
FORMAL_INSTANCES = (
    "c101C5",
    "r105C5",
    "rc105C5",
    "c104C10",
    "r103C10",
    "rc102C10",
    "c106C15",
    "r105C15",
    "rc103C15",
    "c101_21",
    "r101_21",
    "rc101_21",
)
FORMAL_SEEDS = (2014, 2015, 2016)
FAILURE_EVENT_STATUSES = {
    "failed",
    "prefilter_rejected",
    "exact_infeasible",
    "budget_exhausted",
    "not_applicable",
    "time_limit",
}

PER_RUN_FIELDS = (
    "schema_version",
    "run_label",
    "experiment_id",
    "instance",
    "algorithm",
    "operator_profile",
    "seed",
    "repository_revision",
    "repository_dirty",
    "algorithm_source_sha256",
    "instance_sha256",
    "start_utc",
    "end_utc",
    "time_limit_seconds",
    "max_iterations",
    "threads",
    "objective_schema",
    "objective_key",
    "primary_vehicle_count",
    "secondary_total_distance",
    "tertiary_total_charging_time",
    "quaternary_charging_count",
    "total_energy",
    "total_charged_energy",
    "runtime_seconds",
    "first_feasible_time",
    "best_time",
    "iterations",
    "accepted_moves",
    "improving_moves",
    "rejected_moves",
    "charging_subproblem_calls",
    "cache_hits",
    "cache_misses",
    "unique_route_evaluations",
    "effective_iterations",
    "removal_tier_counts",
    "maximum_stagnation",
    "charging_subproblem_average_seconds",
    "charging_labels_generated",
    "charging_labels_pruned",
    "coverage_violations",
    "route_structure_violations",
    "capacity_violations",
    "time_window_violations",
    "energy_violations",
    "objective_violations",
    "other_violations",
    "total_violation_count",
    "neighborhood_events",
    "operator_failure_events",
    "operator_statistics_json",
    "constraint_operator_statistics_json",
    "status",
    "feasible",
    "failure_reason",
    "raw_log_path",
    "solution_path",
)

SUMMARY_FIELDS = (
    "instance",
    "algorithm",
    "runs",
    "feasible_runs",
    "feasibility_rate",
    "best_objective_key",
    "best_vehicle_count",
    "best_total_distance",
    "best_total_charging_time",
    "best_charging_count",
    "mean_vehicle_count",
    "median_vehicle_count",
    "worst_vehicle_count",
    "standard_deviation_vehicle_count",
    "mean_total_distance",
    "median_total_distance",
    "worst_total_distance",
    "standard_deviation_total_distance",
    "mean_total_charging_time",
    "median_total_charging_time",
    "worst_total_charging_time",
    "standard_deviation_total_charging_time",
    "mean_charging_count",
    "median_charging_count",
    "worst_charging_count",
    "standard_deviation_charging_count",
    "mean_runtime_seconds",
    "median_runtime_seconds",
    "worst_runtime_seconds",
    "standard_deviation_runtime_seconds",
    "mean_iterations",
    "median_iterations",
    "worst_iterations",
    "standard_deviation_iterations",
)

OPERATOR_SUMMARY_FIELDS = (
    "instance",
    "operator",
    "runs",
    "calls",
    "feasible_repairs",
    "accepted",
    "rejected",
    "improved",
    "best",
    "vehicle_reductions",
    "distance_improvements",
    "prefilter_passed",
    "prefilter_rejected",
    "new_routes_created",
    "exact_route_evaluations",
    "failure_reasons_json",
)

OPERATOR_FAILURE_FIELDS = (
    "run_label",
    "instance",
    "seed",
    "iteration",
    "operator",
    "status",
    "reason",
    "route_indices",
    "affected_route_indices",
    "removed_customers",
    "candidate_customer_sequence",
    "candidate_route_sequences",
    "candidate_vehicle_delta",
    "prefilter_passed",
    "new_routes_created",
    "exact_route_evaluations",
    "selection_rank",
    "chain_depth",
    "segment_length",
    "track",
    "constraint_category",
    "removal_tier",
    "removal_size_requested",
    "removal_size_actual",
    "stagnation_iterations",
    "removal_trigger",
    "reset_observed",
    "ranking_score",
)

OPERATOR_EVENT_FIELDS = (
    "run_label",
    "instance",
    "seed",
    "iteration",
    "operator",
    "status",
    "reason",
    "accepted",
    "vehicle_reduction",
    "distance_improvement",
    "candidate_objective_key",
    "route_indices",
    "affected_route_indices",
    "removed_customers",
    "candidate_customer_sequence",
    "candidate_route_sequences",
    "candidate_vehicle_delta",
    "candidate_feasible",
    "prefilter_passed",
    "new_routes_created",
    "exact_route_evaluations",
    "selection_rank",
    "chain_depth",
    "segment_length",
    "track",
    "constraint_category",
    "removal_tier",
    "removal_size_requested",
    "removal_size_actual",
    "stagnation_iterations",
    "removal_trigger",
    "reset_observed",
    "ranking_score",
)

COMPARISON_FIELDS = (
    "instance",
    "baseline",
    "candidate",
    "metric",
    "statistic",
    "baseline_value",
    "candidate_value",
    "absolute_delta",
    "relative_delta",
    "classification",
    "gate_status",
    "reason",
)

GATE_FIELDS = ("gate", "status", "observed", "expected", "evidence")
REPEATABILITY_FIELDS = (
    "first_run",
    "second_run",
    "status",
    "first_gate_status",
    "second_gate_status",
    "first_run_keys",
    "second_run_keys",
    "configuration_match",
    "details",
)

OBJECTIVE_SCHEMA = "vehicles,distance,charging_time,charging_count"
_LOWER_IS_BETTER = {
    "vehicle_count",
    "total_distance",
    "total_charging_time",
    "charging_count",
    "runtime_seconds",
}


@dataclass(frozen=True, slots=True)
class Stage02Config:
    schema_version: str
    experiment_id: str
    algorithm: str
    operator_profile: OperatorProfile
    baseline_dir: Path
    stage01_per_run: Path
    comparison_per_run: Path
    comparison_label: str
    comparison_gate: str
    benchmark_dir: Path
    instances: tuple[str, ...]
    seeds: tuple[int, ...]
    time_limit_seconds: float
    max_iterations: int
    threads: int
    vehicle_operator_config: VehicleOperatorConfig


def load_config(path: Path) -> Stage02Config:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    try:
        stage = payload["stage02"]
        benchmark = payload["benchmark"]
        run = payload["run"]
        operators = payload["vehicle_operators"]
        config = Stage02Config(
            schema_version=str(stage["schema_version"]),
            experiment_id=str(stage["experiment_id"]),
            algorithm=str(stage["algorithm"]),
            operator_profile=OperatorProfile(
                str(stage.get("operator_profile", OperatorProfile.STAGE02_ROUTE_REDUCTION.value))
            ),
            baseline_dir=Path(str(stage["baseline_dir"])),
            stage01_per_run=Path(
                str(stage.get("stage01_per_run") or stage["comparison_per_run"])
            ),
            comparison_per_run=Path(
                str(stage.get("comparison_per_run") or stage["stage01_per_run"])
            ),
            comparison_label=str(stage.get("comparison_label", "stage01")),
            comparison_gate=str(stage.get("comparison_gate", "stage01_best_objective")),
            benchmark_dir=Path(str(benchmark["directory"])),
            instances=tuple(str(value) for value in benchmark["instances"]),
            seeds=tuple(int(value) for value in run["seeds"]),
            time_limit_seconds=float(run["time_limit_seconds"]),
            max_iterations=int(run["max_iterations"]),
            threads=int(run["threads"]),
            vehicle_operator_config=VehicleOperatorConfig(
                max_route_elimination_attempts=int(operators["max_route_elimination_attempts"]),
                route_elimination_exact_evaluation_budget=int(
                    operators["route_elimination_exact_evaluation_budget"]
                ),
                route_merge_exact_evaluation_budget=int(
                    operators["route_merge_exact_evaluation_budget"]
                ),
                vehicle_repair_exact_evaluation_budget=int(
                    operators["vehicle_repair_exact_evaluation_budget"]
                ),
                vehicle_reduction_refinement_exact_evaluation_budget=int(
                    operators.get(
                        "vehicle_reduction_refinement_exact_evaluation_budget",
                        VehicleOperatorConfig().vehicle_reduction_refinement_exact_evaluation_budget,
                    )
                ),
                relocate_exact_evaluation_budget=int(
                    operators.get(
                        "relocate_exact_evaluation_budget",
                        VehicleOperatorConfig().relocate_exact_evaluation_budget,
                    )
                ),
                swap_exact_evaluation_budget=int(
                    operators.get(
                        "swap_exact_evaluation_budget",
                        VehicleOperatorConfig().swap_exact_evaluation_budget,
                    )
                ),
                two_opt_star_exact_evaluation_budget=int(
                    operators.get(
                        "two_opt_star_exact_evaluation_budget",
                        VehicleOperatorConfig().two_opt_star_exact_evaluation_budget,
                    )
                ),
                route_segment_exact_evaluation_budget=int(
                    operators.get(
                        "route_segment_exact_evaluation_budget",
                        VehicleOperatorConfig().route_segment_exact_evaluation_budget,
                    )
                ),
                ejection_chain_exact_evaluation_budget=int(
                    operators.get(
                        "ejection_chain_exact_evaluation_budget",
                        VehicleOperatorConfig().ejection_chain_exact_evaluation_budget,
                    )
                ),
                quality_probe_exact_evaluation_budget=int(
                    operators.get(
                        "quality_probe_exact_evaluation_budget",
                        VehicleOperatorConfig().quality_probe_exact_evaluation_budget,
                    )
                ),
                quality_route_segment_probe_exact_evaluation_budget=int(
                    operators.get(
                        "quality_route_segment_probe_exact_evaluation_budget",
                        VehicleOperatorConfig().quality_route_segment_probe_exact_evaluation_budget,
                    )
                ),
                constraint_lane_time_budget_seconds=float(
                    operators.get(
                        "constraint_lane_time_budget_seconds",
                        VehicleOperatorConfig().constraint_lane_time_budget_seconds,
                    )
                ),
                route_segment_min_length=int(
                    operators.get(
                        "route_segment_min_length",
                        VehicleOperatorConfig().route_segment_min_length,
                    )
                ),
                route_segment_max_length=int(
                    operators.get(
                        "route_segment_max_length",
                        VehicleOperatorConfig().route_segment_max_length,
                    )
                ),
                ejection_chain_max_depth=int(
                    operators.get(
                        "ejection_chain_max_depth",
                        VehicleOperatorConfig().ejection_chain_max_depth,
                    )
                ),
                ejection_chain_beam_width=int(
                    operators.get(
                        "ejection_chain_beam_width",
                        VehicleOperatorConfig().ejection_chain_beam_width,
                    )
                ),
                constraint_probe_exact_evaluation_budget=int(
                    operators.get(
                        "constraint_probe_exact_evaluation_budget",
                        VehicleOperatorConfig().constraint_probe_exact_evaluation_budget,
                    )
                ),
                station_pressure_exact_evaluation_budget=int(
                    operators.get(
                        "station_pressure_exact_evaluation_budget",
                        VehicleOperatorConfig().station_pressure_exact_evaluation_budget,
                    )
                ),
                time_window_conflict_exact_evaluation_budget=int(
                    operators.get(
                        "time_window_conflict_exact_evaluation_budget",
                        VehicleOperatorConfig().time_window_conflict_exact_evaluation_budget,
                    )
                ),
                worst_energy_detour_exact_evaluation_budget=int(
                    operators.get(
                        "worst_energy_detour_exact_evaluation_budget",
                        VehicleOperatorConfig().worst_energy_detour_exact_evaluation_budget,
                    )
                ),
                shaw_related_exact_evaluation_budget=int(
                    operators.get(
                        "shaw_related_exact_evaluation_budget",
                        VehicleOperatorConfig().shaw_related_exact_evaluation_budget,
                    )
                ),
                small_removal_min_fraction=float(
                    operators.get(
                        "small_removal_min_fraction",
                        VehicleOperatorConfig().small_removal_min_fraction,
                    )
                ),
                small_removal_max_fraction=float(
                    operators.get(
                        "small_removal_max_fraction",
                        VehicleOperatorConfig().small_removal_max_fraction,
                    )
                ),
                medium_removal_min_fraction=float(
                    operators.get(
                        "medium_removal_min_fraction",
                        VehicleOperatorConfig().medium_removal_min_fraction,
                    )
                ),
                medium_removal_max_fraction=float(
                    operators.get(
                        "medium_removal_max_fraction",
                        VehicleOperatorConfig().medium_removal_max_fraction,
                    )
                ),
                large_removal_min_fraction=float(
                    operators.get(
                        "large_removal_min_fraction",
                        VehicleOperatorConfig().large_removal_min_fraction,
                    )
                ),
                large_removal_max_fraction=float(
                    operators.get(
                        "large_removal_max_fraction",
                        VehicleOperatorConfig().large_removal_max_fraction,
                    )
                ),
                medium_stagnation_threshold=int(
                    operators.get(
                        "medium_stagnation_threshold",
                        VehicleOperatorConfig().medium_stagnation_threshold,
                    )
                ),
                large_stagnation_threshold=int(
                    operators.get(
                        "large_stagnation_threshold",
                        VehicleOperatorConfig().large_stagnation_threshold,
                    )
                ),
                exploration_period=int(
                    operators.get(
                        "exploration_period",
                        VehicleOperatorConfig().exploration_period,
                    )
                ),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid Stage 2 configuration: {error}") from error
    _validate_config(config)
    return config


def run_stage02(
    *,
    config_path: Path,
    output_dir: Path,
    summary_dir: Path,
    run_label: str = "stage02",
    repeat_of: Path | None = None,
) -> dict[str, Path]:
    config = load_config(config_path)
    _validate_run_label(run_label)
    root = _repository_root()
    baseline_dir = _resolve(root, config.baseline_dir)
    stage01_path = _resolve(root, config.stage01_per_run)
    comparison_path = _resolve(root, config.comparison_per_run)
    benchmark_dir = _resolve(root, config.benchmark_dir)
    stage00_config_path = root / "configs" / "stage00_baseline.toml"
    stage00_config = load_stage00_config(stage00_config_path)
    if benchmark_dir.resolve() != stage00_config.benchmark_dir.resolve():
        raise RuntimeError("Stage 2 benchmark directory differs from Stage 0")
    if stage00_config.instances != config.instances or stage00_config.seeds != config.seeds:
        raise RuntimeError("Stage 2 scope differs from the canonical Stage 0 scope")
    verify_results(stage00_config, baseline_dir, require_manifest=True)
    stage1_baseline = _load_stage1_baseline(stage01_path, config)
    comparison_baseline = _load_comparison_baseline(comparison_path, config)
    stage00_baseline = _load_stage00_baseline(baseline_dir, benchmark_dir, config)
    baseline_manifest_sha256 = _sha256(baseline_dir / "manifest.json")

    if output_dir.exists():
        raise FileExistsError(f"Stage 2.1 output directory already exists: {output_dir}")
    tracked_names = _tracked_names(run_label, config.comparison_label)
    existing = [summary_dir / name for name in tracked_names if (summary_dir / name).exists()]
    if existing:
        raise FileExistsError(f"tracked Stage 2.1 summaries already exist: {existing}")

    output_dir.mkdir(parents=True)
    raw_dir = output_dir / "raw"
    solution_dir = output_dir / "solutions"
    raw_dir.mkdir()
    solution_dir.mkdir()
    per_run_path = output_dir / f"{run_label}_per_run_results.csv"
    failure_path = output_dir / f"{run_label}_failure_cases.csv"
    _write_csv(per_run_path, PER_RUN_FIELDS, [])
    _write_csv(failure_path, PER_RUN_FIELDS, [])

    source_hashes = _source_hashes(root)
    algorithm_hash = _combined_hash(source_hashes)
    rows: list[dict[str, Any]] = []
    operator_records: list[tuple[str, int, dict[str, dict[str, object]]]] = []
    contextual_events: list[dict[str, object]] = []

    for instance_name in config.instances:
        instance_path = benchmark_dir / f"{instance_name}.txt"
        if not instance_path.is_file():
            raise FileNotFoundError(f"benchmark instance is missing: {instance_path}")
        instance_hash = _sha256(instance_path)
        instance = parse_schneider(instance_path)
        for seed in config.seeds:
            try:
                started = datetime.now(UTC)
                result = solve_alns(
                    instance,
                    seed=seed,
                    max_iterations=config.max_iterations,
                    time_limit_seconds=config.time_limit_seconds,
                    operator_profile=config.operator_profile,
                    vehicle_operator_config=config.vehicle_operator_config,
                )
                ended = datetime.now(UTC)
                row, events = _record_run(
                    config=config,
                    run_label=run_label,
                    instance=instance,
                    seed=seed,
                    result=result,
                    started=started,
                    ended=ended,
                    repository_revision=_git_revision(root),
                    repository_dirty=_git_dirty(root),
                    algorithm_hash=algorithm_hash,
                    instance_hash=instance_hash,
                    output_dir=output_dir,
                    raw_dir=raw_dir,
                    solution_dir=solution_dir,
                )
                rows.append(row)
                operator_records.append((instance.name, seed, result.neighborhood_statistics))
                contextual_events.extend(
                    {
                        "run_label": run_label,
                        "instance": instance.name,
                        "seed": seed,
                        **event,
                    }
                    for event in events
                )
                _write_csv(per_run_path, PER_RUN_FIELDS, rows)
                _write_csv(
                    failure_path,
                    PER_RUN_FIELDS,
                    [item for item in rows if not item["feasible"]],
                )
            except BaseException as error:
                _write_interrupted_run_artifacts(
                    root=root,
                    config=config,
                    config_path=config_path,
                    output_dir=output_dir,
                    run_label=run_label,
                    raw_dir=raw_dir,
                    solution_dir=solution_dir,
                    per_run_path=per_run_path,
                    failure_path=failure_path,
                    source_hashes=source_hashes,
                    algorithm_hash=algorithm_hash,
                    baseline_manifest_sha256=baseline_manifest_sha256,
                    baseline_dir=baseline_dir,
                    benchmark_dir=benchmark_dir,
                    rows=rows,
                    operator_records=operator_records,
                    contextual_events=contextual_events,
                    error=error,
                )
                raise

    summary_rows = _summarize(rows)
    operator_summary_rows = _summarize_operators(operator_records)
    operator_failure_rows = _operator_failure_rows(contextual_events)
    comparison_rows = _compare_baseline(
        comparison_baseline,
        rows,
        baseline_label=config.comparison_label,
        candidate_label=config.operator_profile.value,
    )
    focused_rows = [row for row in comparison_rows if row["instance"] in FOCUSED_100_INSTANCES]
    gate_rows = _evaluate_gates(
        config=config,
        rows=rows,
        stage00_baseline=stage00_baseline,
        stage1_baseline=stage1_baseline,
        comparison_baseline=comparison_baseline,
        contextual_events=contextual_events,
        baseline_manifest_sha256=baseline_manifest_sha256,
        baseline_dir=baseline_dir,
    )
    repeatability_rows, repeatability_ok = _repeatability_check(
        repeat_of=repeat_of,
        run_label=run_label,
        config=config,
        current_rows=rows,
        current_gate_rows=gate_rows,
    )
    if repeat_of is not None:
        gate_rows.append(
            _gate(
                "independent_complete_rerun",
                repeatability_ok,
                "pass" if repeatability_ok else "fail",
                "first and second complete runs pass every hard gate",
                "repeatability comparison artifact",
            )
        )

    summary_path = output_dir / f"{run_label}_summary_results.csv"
    operator_summary_path = output_dir / f"{run_label}_operator_summary.csv"
    operator_event_path = output_dir / f"{run_label}_operator_events.csv"
    operator_failure_path = output_dir / f"{run_label}_operator_failure_events.csv"
    repeatability_path = output_dir / f"{run_label}_repeatability.csv"
    comparison_path = output_dir / f"{run_label}_{config.comparison_label}_comparison.csv"
    focused_path = output_dir / f"{run_label}_100_customer_comparison.csv"
    gate_path = output_dir / f"{run_label}_gate_report.csv"
    environment_path = output_dir / f"{run_label}_environment.json"
    parameters_path = output_dir / f"{run_label}_parameters.toml"
    _write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    _write_csv(operator_summary_path, OPERATOR_SUMMARY_FIELDS, operator_summary_rows)
    _write_csv(operator_event_path, OPERATOR_EVENT_FIELDS, _operator_event_rows(contextual_events))
    _write_csv(operator_failure_path, OPERATOR_FAILURE_FIELDS, operator_failure_rows)
    _write_csv(repeatability_path, REPEATABILITY_FIELDS, repeatability_rows)
    _write_csv(comparison_path, COMPARISON_FIELDS, comparison_rows)
    _write_csv(focused_path, COMPARISON_FIELDS, focused_rows)
    _write_csv(gate_path, GATE_FIELDS, gate_rows)
    shutil.copy2(config_path, parameters_path)
    _write_json(
        environment_path,
        _environment_record(
            root=root,
            config=config,
            config_path=config_path,
            run_label=run_label,
            source_hashes=source_hashes,
            algorithm_hash=algorithm_hash,
            baseline_manifest_sha256=baseline_manifest_sha256,
            baseline_dir=baseline_dir,
            benchmark_dir=benchmark_dir,
        ),
    )
    manifest_path = output_dir / f"{run_label}_manifest.json"
    _write_manifest(output_dir, manifest_path)

    copies = {
        "per_run": (per_run_path, summary_dir / f"{run_label}_per_run_results.csv"),
        "summary": (summary_path, summary_dir / f"{run_label}_summary_results.csv"),
        "failures": (failure_path, summary_dir / f"{run_label}_failure_cases.csv"),
        "operator_summary": (
            operator_summary_path,
            summary_dir / f"{run_label}_operator_summary.csv",
        ),
        "operator_events": (
            operator_event_path,
            summary_dir / f"{run_label}_operator_events.csv",
        ),
        "operator_failure_events": (
            operator_failure_path,
            summary_dir / f"{run_label}_operator_failure_events.csv",
        ),
        "repeatability": (
            repeatability_path,
            summary_dir / f"{run_label}_repeatability.csv",
        ),
        "comparison": (
            comparison_path,
            summary_dir / f"{run_label}_{config.comparison_label}_comparison.csv",
        ),
        "100_customer_comparison": (
            focused_path,
            summary_dir / f"{run_label}_100_customer_comparison.csv",
        ),
        "gate_report": (gate_path, summary_dir / f"{run_label}_gate_report.csv"),
        "environment": (environment_path, summary_dir / f"{run_label}_environment.json"),
        "parameters": (parameters_path, summary_dir / f"{run_label}_parameters.toml"),
    }
    summary_dir.mkdir(parents=True, exist_ok=True)
    for source, destination in copies.values():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"tracked Stage 2 summary already exists: {destination}")
        destination.write_bytes(source.read_bytes())

    if any(row["status"] != "pass" for row in gate_rows):
        failed = "; ".join(row["gate"] for row in gate_rows if row["status"] != "pass")
        raise RuntimeError(f"Stage 2 acceptance gates failed: {failed}")

    return {
        **{name: destination for name, (_, destination) in copies.items()},
        "raw_dir": raw_dir,
        "solution_dir": solution_dir,
        "manifest": manifest_path,
    }


def _write_interrupted_run_artifacts(
    *,
    root: Path,
    config: Stage02Config,
    config_path: Path,
    output_dir: Path,
    run_label: str,
    raw_dir: Path,
    solution_dir: Path,
    per_run_path: Path,
    failure_path: Path,
    source_hashes: dict[str, str],
    algorithm_hash: str,
    baseline_manifest_sha256: str,
    baseline_dir: Path,
    benchmark_dir: Path,
    rows: list[dict[str, Any]],
    operator_records: list[tuple[str, int, dict[str, dict[str, object]]]],
    contextual_events: list[dict[str, object]],
    error: BaseException,
) -> None:
    """Persist a complete partial-run evidence bundle before propagating errors."""

    summary_path = output_dir / f"{run_label}_summary_results.csv"
    operator_summary_path = output_dir / f"{run_label}_operator_summary.csv"
    operator_event_path = output_dir / f"{run_label}_operator_events.csv"
    operator_failure_path = output_dir / f"{run_label}_operator_failure_events.csv"
    repeatability_path = output_dir / f"{run_label}_repeatability.csv"
    comparison_path = output_dir / f"{run_label}_{config.comparison_label}_comparison.csv"
    focused_path = output_dir / f"{run_label}_100_customer_comparison.csv"
    gate_path = output_dir / f"{run_label}_gate_report.csv"
    environment_path = output_dir / f"{run_label}_environment.json"
    parameters_path = output_dir / f"{run_label}_parameters.toml"
    interruption_path = output_dir / f"{run_label}_interruption.json"

    _write_csv(per_run_path, PER_RUN_FIELDS, rows)
    _write_csv(
        failure_path,
        PER_RUN_FIELDS,
        [row for row in rows if not row.get("feasible")],
    )
    _write_csv(summary_path, SUMMARY_FIELDS, _summarize(rows))
    _write_csv(
        operator_summary_path,
        OPERATOR_SUMMARY_FIELDS,
        _summarize_operators(operator_records),
    )
    _write_csv(
        operator_event_path,
        OPERATOR_EVENT_FIELDS,
        _operator_event_rows(contextual_events),
    )
    _write_csv(
        operator_failure_path,
        OPERATOR_FAILURE_FIELDS,
        _operator_failure_rows(contextual_events),
    )
    _write_csv(
        repeatability_path,
        REPEATABILITY_FIELDS,
        [
            {
                "first_run": "",
                "second_run": "",
                "status": "interrupted",
                "first_gate_status": "",
                "second_gate_status": "",
                "first_run_keys": "",
                "second_run_keys": f"{len(rows)} partial rows",
                "configuration_match": "",
                "details": repr(error),
            }
        ],
    )
    _write_csv(comparison_path, COMPARISON_FIELDS, [])
    _write_csv(focused_path, COMPARISON_FIELDS, [])
    _write_csv(
        gate_path,
        GATE_FIELDS,
        [
            _gate(
                "run_interrupted",
                False,
                {"completed_runs": len(rows), "error": repr(error)},
                "all formal runs complete",
                "interruption evidence bundle",
            )
        ],
    )
    shutil.copy2(config_path, parameters_path)
    _write_json(
        environment_path,
        _environment_record(
            root=root,
            config=config,
            config_path=config_path,
            run_label=run_label,
            source_hashes=source_hashes,
            algorithm_hash=algorithm_hash,
            baseline_manifest_sha256=baseline_manifest_sha256,
            baseline_dir=baseline_dir,
            benchmark_dir=benchmark_dir,
        ),
    )
    _write_json(
        interruption_path,
        {
            "schema_version": config.schema_version,
            "run_label": run_label,
            "error_type": type(error).__name__,
            "error": repr(error),
            "completed_run_count": len(rows),
            "raw_directory": str(raw_dir),
            "solution_directory": str(solution_dir),
        },
    )
    _write_manifest(output_dir, output_dir / f"{run_label}_manifest.json")


def _record_run(
    *,
    config: Stage02Config,
    run_label: str,
    instance: Instance,
    seed: int,
    result: ALNSResult,
    started: datetime,
    ended: datetime,
    repository_revision: str,
    repository_dirty: bool,
    algorithm_hash: str,
    instance_hash: str,
    output_dir: Path,
    raw_dir: Path,
    solution_dir: Path,
) -> tuple[dict[str, Any], tuple[dict[str, object], ...]]:
    routes = [list(route) for route in result.routes]
    report = validate_routes(instance, routes, claimed_objective=result.objective_value)
    validation_objective, validation_failure = _validated_objective(
        instance, report, result.objective
    )
    identifier = f"{instance.name}-alns_{config.operator_profile.value}-{seed}"
    raw_path = raw_dir / f"{identifier}.json"
    solution_path = solution_dir / f"{identifier}.json"
    events = tuple(result.neighborhood_events)
    failure_events = tuple(
        event for event in events if str(event.get("status")) in FAILURE_EVENT_STATUSES
    )
    row: dict[str, Any] = {field: "" for field in PER_RUN_FIELDS}
    row.update(
        {
            "schema_version": config.schema_version,
            "run_label": run_label,
            "experiment_id": identifier,
            "instance": instance.name,
            "algorithm": config.algorithm,
            "operator_profile": result.operator_profile,
            "seed": seed,
            "repository_revision": repository_revision,
            "repository_dirty": repository_dirty,
            "algorithm_source_sha256": algorithm_hash,
            "instance_sha256": instance_hash,
            "start_utc": started.isoformat(),
            "end_utc": ended.isoformat(),
            "time_limit_seconds": config.time_limit_seconds,
            "max_iterations": config.max_iterations,
            "threads": config.threads,
            "objective_schema": OBJECTIVE_SCHEMA,
            "objective_key": _render_key(validation_objective),
            "primary_vehicle_count": (
                validation_objective.vehicle_count if validation_objective else ""
            ),
            "secondary_total_distance": (
                validation_objective.total_distance if validation_objective else ""
            ),
            "tertiary_total_charging_time": (
                validation_objective.total_charging_time if validation_objective else ""
            ),
            "quaternary_charging_count": (
                validation_objective.charging_count if validation_objective else ""
            ),
            "total_energy": report.total_energy,
            "total_charged_energy": report.total_charged_energy,
            "runtime_seconds": result.runtime_seconds,
            "first_feasible_time": result.first_feasible_time,
            "best_time": result.best_time,
            "iterations": result.iterations,
            "accepted_moves": result.accepted_moves,
            "improving_moves": result.improving_moves,
            "rejected_moves": result.rejected_moves,
            "charging_subproblem_calls": result.charging_subproblem_calls,
            "cache_hits": result.cache_hits,
            "cache_misses": result.cache_misses,
            "unique_route_evaluations": result.unique_route_evaluations,
            "effective_iterations": result.effective_iterations,
            "removal_tier_counts": json.dumps(
                result.removal_tier_counts, sort_keys=True, separators=(",", ":")
            ),
            "maximum_stagnation": result.maximum_stagnation,
            "charging_subproblem_average_seconds": (
                result.charging_subproblem_time / result.charging_subproblem_calls
                if result.charging_subproblem_calls
                else 0.0
            ),
            "charging_labels_generated": result.charging_labels_generated,
            "charging_labels_pruned": result.charging_labels_pruned,
            **_violation_counts(report),
            "neighborhood_events": len(events),
            "operator_failure_events": len(failure_events),
            "operator_statistics_json": json.dumps(
                result.neighborhood_statistics, sort_keys=True, separators=(",", ":")
            ),
            "constraint_operator_statistics_json": json.dumps(
                result.constraint_operator_statistics,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "status": "feasible" if validation_objective is not None else "invalid",
            "feasible": validation_objective is not None,
            "failure_reason": _join_failures(result.failure_reason, validation_failure),
            "raw_log_path": str(raw_path.relative_to(output_dir)),
            "solution_path": str(solution_path.relative_to(output_dir)),
        }
    )
    _write_json(
        raw_path,
        {
            "record": row,
            "solver_result": asdict(result),
            "neighborhood_events": events,
        },
    )
    _write_json(
        solution_path,
        {
            "instance": instance.name,
            "seed": seed,
            "algorithm": config.algorithm,
            "operator_profile": result.operator_profile,
            "routes": routes,
        },
    )
    return row, events


def _validated_objective(
    instance: Instance,
    report: SolutionReport,
    claimed: SolutionObjective | None,
) -> tuple[SolutionObjective | None, str]:
    if not report.feasible:
        violations = (*report.violations, *(v for route in report.routes for v in route.violations))
        return None, "unified validation failed: " + "; ".join(violations)
    objective = SolutionObjective.from_report(instance, report)
    if claimed is None or compare_objectives(objective, claimed) is not ObjectiveComparison.EQUAL:
        return None, "solver objective differs from unified validation"
    return objective, ""


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["instance"]), []).append(row)
    output: list[dict[str, Any]] = []
    for instance, group in sorted(grouped.items()):
        feasible = [row for row in group if bool(row["feasible"])]
        objectives = [_objective_from_row(row) for row in feasible]
        best = min(objectives, key=lambda objective: objective.key) if objectives else None
        output.append(
            {
                "instance": instance,
                "algorithm": str(group[0]["algorithm"]),
                "runs": len(group),
                "feasible_runs": len(feasible),
                "feasibility_rate": len(feasible) / len(group),
                "best_objective_key": _render_key(best),
                "best_vehicle_count": _objective_value(best, "vehicle_count"),
                "best_total_distance": _objective_value(best, "total_distance"),
                "best_total_charging_time": _objective_value(best, "total_charging_time"),
                "best_charging_count": _objective_value(best, "charging_count"),
                **_metric_summary(objectives, "vehicle_count"),
                **_metric_summary(objectives, "total_distance"),
                **_metric_summary(objectives, "total_charging_time"),
                **_metric_summary(objectives, "charging_count"),
                **_numeric_summary(group, "runtime_seconds"),
                **_numeric_summary(group, "iterations"),
            }
        )
    return output


def _summarize_operators(
    records: list[tuple[str, int, dict[str, dict[str, object]]]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for instance, _, statistics_by_operator in records:
        for operator, statistics_for_run in statistics_by_operator.items():
            grouped.setdefault((instance, operator), []).append(statistics_for_run)
    output: list[dict[str, Any]] = []
    for (instance, operator), group in sorted(grouped.items()):
        row: dict[str, Any] = {field: "" for field in OPERATOR_SUMMARY_FIELDS}
        row.update(
            {
                "instance": instance,
                "operator": operator,
                "runs": len(group),
            }
        )
        for field in OPERATOR_SUMMARY_FIELDS[3:-1]:
            row[field] = sum(_as_int(item.get(field, 0)) for item in group)
        failures: dict[str, int] = {}
        for item in group:
            raw = item.get("failure_reasons", {})
            if isinstance(raw, dict):
                for reason, count in raw.items():
                    failures[str(reason)] = failures.get(str(reason), 0) + _as_int(count)
        row["failure_reasons_json"] = json.dumps(failures, sort_keys=True, separators=(",", ":"))
        output.append(row)
    return output


def _compare_baseline(
    baseline: dict[tuple[str, int], SolutionObjective],
    rows: list[dict[str, Any]],
    *,
    baseline_label: str,
    candidate_label: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["instance"]), []).append(row)
    output: list[dict[str, Any]] = []
    for instance, group in sorted(grouped.items()):
        candidate_objectives = [_objective_from_row(row) for row in group if bool(row["feasible"])]
        baseline_objectives = [
            baseline[(instance, int(row["seed"]))] for row in group
        ]
        if not candidate_objectives:
            output.append(
                _comparison_row(
                    instance,
                    baseline_label,
                    candidate_label,
                    "objective",
                    "best",
                    _render_key(min(baseline_objectives, key=lambda value: value.key)),
                    "",
                    "",
                    "",
                    "invalid",
                    "fail",
                    f"no validator-feasible {candidate_label} run",
                )
            )
            continue
        candidate_best = min(candidate_objectives, key=lambda objective: objective.key)
        baseline_best = min(baseline_objectives, key=lambda objective: objective.key)
        comparison = compare_objectives(candidate_best, baseline_best)
        output.append(
            _comparison_row(
                instance,
                baseline_label,
                candidate_label,
                "objective",
                "best",
                _render_key(baseline_best),
                _render_key(candidate_best),
                "",
                "",
                comparison.value,
                "pass" if comparison is not ObjectiveComparison.WORSE else "fail",
                "vehicle-first objective comparison",
            )
        )
        for metric in (
            "vehicle_count",
            "total_distance",
            "total_charging_time",
            "charging_count",
        ):
            baseline_values = [float(getattr(item, metric)) for item in baseline_objectives]
            candidate_values = [float(getattr(item, metric)) for item in candidate_objectives]
            output.append(
                _comparison_row(
                    instance,
                    baseline_label,
                    candidate_label,
                    metric,
                    "mean",
                    statistics.fmean(baseline_values),
                    statistics.fmean(candidate_values),
                    statistics.fmean(candidate_values) - statistics.fmean(baseline_values),
                    _relative_delta(
                        statistics.fmean(baseline_values), statistics.fmean(candidate_values)
                    ),
                    _classify_metric(
                        metric,
                        statistics.fmean(baseline_values),
                        statistics.fmean(candidate_values),
                    ),
                    "pass",
                    "descriptive comparison",
                )
            )
    return output


def _evaluate_gates(
    *,
    config: Stage02Config,
    rows: list[dict[str, Any]],
    stage00_baseline: dict[tuple[str, int], SolutionObjective],
    stage1_baseline: dict[tuple[str, int], SolutionObjective],
    comparison_baseline: dict[tuple[str, int], SolutionObjective],
    contextual_events: list[dict[str, object]],
    baseline_manifest_sha256: str,
    baseline_dir: Path,
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    expected_keys = {(instance, seed) for instance in config.instances for seed in config.seeds}
    actual_keys = [(str(row["instance"]), int(row["seed"])) for row in rows]
    duplicate_keys = len(actual_keys) != len(set(actual_keys))
    coverage_ok = set(actual_keys) == expected_keys and not duplicate_keys
    gates.append(
        _gate(
            "run_coverage",
            coverage_ok,
            f"{len(actual_keys)} rows, {len(set(actual_keys))} unique keys",
            f"exactly {len(expected_keys)} unique (instance, seed) keys",
            "stage02 per-run results",
        )
    )
    feasible_ok = len(rows) == len(expected_keys) and all(bool(row["feasible"]) for row in rows)
    gates.append(
        _gate(
            "validator_feasibility",
            feasible_ok,
            _count_feasible(rows),
            f"{len(expected_keys)}/{len(expected_keys)}",
            "per-run validator result",
        )
    )

    comparison_by_instance = _group_objectives(comparison_baseline)
    candidate_by_instance = _group_objectives(
        {
            (str(row["instance"]), int(row["seed"])): _objective_from_row(row)
            for row in rows
            if bool(row["feasible"])
        }
    )
    objective_ok = True
    objective_observed: list[str] = []
    for instance in config.instances:
        baseline_best = min(comparison_by_instance[instance], key=lambda value: value.key)
        candidate_values = candidate_by_instance.get(instance, [])
        if len(candidate_values) != len(config.seeds):
            objective_ok = False
            objective_observed.append(
                f"{instance}:missing-feasible-runs-{len(candidate_values)}/{len(config.seeds)}"
            )
            continue
        candidate_best = min(candidate_values, key=lambda value: value.key)
        comparison = compare_objectives(candidate_best, baseline_best)
        objective_observed.append(f"{instance}:{comparison.value}")
        objective_ok = objective_ok and comparison is not ObjectiveComparison.WORSE
    gates.append(
        _gate(
            config.comparison_gate,
            objective_ok,
            "; ".join(objective_observed),
            "every instance is better or equal",
            f"{config.comparison_label} formal objective comparison",
        )
    )

    candidate_rows_by_instance = {
        instance: [row for row in rows if row["instance"] == instance]
        for instance in FOCUSED_R_RC_INSTANCES
    }
    for instance in FOCUSED_R_RC_INSTANCES:
        candidate_by_seed = {
            int(row["seed"]): _objective_from_row(row)
            for row in candidate_rows_by_instance[instance]
            if bool(row["feasible"])
        }
        if set(candidate_by_seed) != set(config.seeds):
            observed = f"{len(candidate_by_seed)}/{len(config.seeds)} feasible seeds"
            gates.extend(
                (
                    _gate(
                        f"{instance}_mean_vehicle_reduction",
                        False,
                        observed,
                        "all focused seeds feasible and mean <= Stage 0 - 1",
                        "per-run validator result",
                    ),
                    _gate(
                        f"{instance}_distance_guard",
                        False,
                        observed,
                        "all focused seeds feasible and mean <= Stage 1 × 1.10",
                        "per-run validator result",
                    ),
                    _gate(
                        f"{instance}_vehicle_seed_stability",
                        False,
                        observed,
                        "all focused seeds feasible and std <= Stage 0",
                        "per-run validator result",
                    ),
                )
            )
            continue
        stage00_values = [
            float(stage00_baseline[(instance, seed)].vehicle_count) for seed in config.seeds
        ]
        candidate_vehicle_values = [
            float(candidate_by_seed[seed].vehicle_count) for seed in config.seeds
        ]
        stage00_mean = statistics.fmean(stage00_values)
        candidate_mean = statistics.fmean(candidate_vehicle_values)
        gates.append(
            _gate(
                f"{instance}_mean_vehicle_reduction",
                candidate_mean <= stage00_mean - 1.0,
                candidate_mean,
                stage00_mean - 1.0,
                "Stage 0 vehicle-count mean",
            )
        )
        stage1_values = [
            float(stage1_baseline[(instance, seed)].total_distance) for seed in config.seeds
        ]
        candidate_distances = [
            float(candidate_by_seed[seed].total_distance) for seed in config.seeds
        ]
        stage1_mean_distance = statistics.fmean(stage1_values)
        candidate_mean_distance = statistics.fmean(candidate_distances)
        gates.append(
            _gate(
                f"{instance}_distance_guard",
                candidate_mean_distance <= stage1_mean_distance * 1.10,
                candidate_mean_distance,
                stage1_mean_distance * 1.10,
                "Stage 1 mean distance × 1.10",
            )
        )
        stage00_std = statistics.pstdev(stage00_values)
        candidate_std = statistics.pstdev(candidate_vehicle_values)
        gates.append(
            _gate(
                f"{instance}_vehicle_seed_stability",
                candidate_std <= stage00_std + 1e-9,
                candidate_std,
                stage00_std,
                "population standard deviation over three seeds",
            )
        )

    elimination_instances = {
        str(event["instance"])
        for event in contextual_events
        if event.get("operator") == "route_elimination"
        and event.get("status") == "candidate_proposed"
        and event.get("reason") == "route_eliminated"
    }
    merge_successes = [
        event
        for event in contextual_events
        if event.get("operator") == "route_merge"
        and event.get("status") == "candidate_proposed"
        and event.get("reason") == "route_merged"
    ]
    required_elimination_instances = min(2, len(config.instances))
    gates.append(
        _gate(
            "route_elimination_multiple_instances",
            len(elimination_instances) >= required_elimination_instances,
            sorted(elimination_instances),
            f"at least {required_elimination_instances} distinct instances",
            "route elimination event log",
        )
    )
    gates.append(
        _gate(
            "route_merge_vehicle_reduction",
            bool(merge_successes),
            len(merge_successes),
            "at least 1 candidate",
            "route merge event log",
        )
    )
    if config.operator_profile in (
        OperatorProfile.STAGE02_ROUTE_QUALITY,
        OperatorProfile.STAGE02_CONSTRAINT_GUIDED,
    ):
        quality_operators = (
            "relocate",
            "swap",
            "two_opt_star",
            "route_segment_destroy",
            "ejection_chain",
        )
        for operator in quality_operators:
            operator_events = [
                event for event in contextual_events if event.get("operator") == operator
            ]
            feasible_candidates = [
                event
                for event in operator_events
                if event.get("candidate_feasible") is True
                and event.get("status") in {"feasible_candidate", "candidate_proposed"}
            ]
            gates.append(
                _gate(
                    f"{operator}_candidate_coverage",
                    bool(operator_events) and bool(feasible_candidates),
                    {
                        "calls": len(operator_events),
                        "feasible_candidates": len(feasible_candidates),
                    },
                    "at least one call and one feasible candidate",
                    "Stage 2.2 operator event log",
                )
            )
        accepted_distance_improvements = [
            event
            for event in contextual_events
            if event.get("operator") in quality_operators
            and event.get("status") in {"feasible_candidate", "candidate_proposed"}
            and event.get("candidate_feasible") is True
            and event.get("accepted") is True
            and event.get("distance_improvement") is True
            and event.get("vehicle_reduction") is False
        ]
        gates.append(
            _gate(
                "route_quality_accepted_same_vehicle_distance_improvement",
                bool(accepted_distance_improvements),
                len(accepted_distance_improvements),
                "at least 1 accepted same-vehicle-count distance improvement",
                "Stage 2.2 operator event log",
            )
        )
    if config.operator_profile is OperatorProfile.STAGE02_CONSTRAINT_GUIDED:
        constraint_events = [
            event
            for event in contextual_events
            if event.get("track") == "constraint_lane"
            and event.get("operator") in {
                "station_pressure",
                "time_window_conflict",
                "worst_energy_detour",
                "shaw_related",
            }
        ]
        for operator in (
            "station_pressure",
            "time_window_conflict",
            "worst_energy_detour",
            "shaw_related",
        ):
            operator_events = [
                event for event in constraint_events if event.get("operator") == operator
            ]
            feasible_candidates = [
                event
                for event in operator_events
                if event.get("candidate_feasible") is True
                and event.get("status") in {"feasible_candidate", "candidate_proposed"}
            ]
            accepted_candidates = [
                event for event in feasible_candidates if event.get("accepted") is True
            ]
            gates.append(
                _gate(
                    f"{operator}_called_feasible_accepted",
                    bool(operator_events)
                    and bool(feasible_candidates)
                    and bool(accepted_candidates),
                    {
                        "calls": len(operator_events),
                        "feasible_candidates": len(feasible_candidates),
                        "accepted_candidates": len(accepted_candidates),
                    },
                    "at least one call, feasible candidate, and accepted candidate",
                    "Stage 2.3 constraint-lane event log",
                )
            )
        accepted_vehicle_increases = [
            event
            for event in constraint_events
            if event.get("accepted") is True
            and event.get("candidate_feasible") is True
            and event.get("candidate_vehicle_delta") not in (None, "")
            and _event_int(event.get("candidate_vehicle_delta")) > 0
        ]
        gates.append(
            _gate(
                "constraint_lane_no_accepted_vehicle_increase",
                not accepted_vehicle_increases,
                len(accepted_vehicle_increases),
                "zero accepted constraint-lane candidates with increased vehicle count",
                "Stage 2.3 constraint-lane event log",
            )
        )
        observed_tiers = {
            str(event.get("removal_tier"))
            for event in constraint_events
            if _event_int(event.get("removal_size_actual")) > 0
        }
        gates.append(
            _gate(
                "dynamic_removal_all_tiers",
                {"small", "medium", "large"} <= observed_tiers,
                sorted(observed_tiers),
                "small, medium, and large all occur with actual removals",
                "Stage 2.3 constraint-lane event log",
            )
        )
        focused_actual_counts = [
            _event_int(event.get("removal_size_actual"))
            for event in constraint_events
            if event.get("instance") in FOCUSED_100_INSTANCES
        ]
        gates.append(
            _gate(
                "100_customer_actual_removal_above_three",
                any(count > 3 for count in focused_actual_counts),
                max(focused_actual_counts, default=0),
                "at least one focused 100-customer actual removal > 3",
                "Stage 2.3 constraint-lane event log",
            )
        )
        stagnation_escalations = [
            event
            for event in constraint_events
            if "stagnation" in str(event.get("removal_trigger", ""))
            and str(event.get("removal_tier")) in {"medium", "large"}
        ]
        gates.append(
            _gate(
                "stagnation_tier_escalation",
                bool(stagnation_escalations),
                len(stagnation_escalations),
                "at least one medium/large tier triggered by stagnation",
                "Stage 2.3 constraint-lane event log",
            )
        )
        reset_events = [event for event in constraint_events if event.get("reset_observed") is True]
        gates.append(
            _gate(
                "global_best_reset_observed",
                bool(reset_events),
                len(reset_events),
                "at least one dynamic removal event records a global-best reset",
                "Stage 2.3 constraint-lane event log",
            )
        )

        def _valid_dynamic_count(event: dict[str, object]) -> bool:
            requested = _event_int(event.get("removal_size_requested"))
            actual = _event_int(event.get("removal_size_actual"))
            if requested == 0 and actual == 0:
                return True
            instance_name = str(event.get("instance"))
            customer_count = _customer_count_from_event(
                instance_name,
                config.benchmark_dir,
            )
            tier_name = str(event.get("removal_tier"))
            fraction_bounds = {
                "small": (
                    config.vehicle_operator_config.small_removal_min_fraction,
                    config.vehicle_operator_config.small_removal_max_fraction,
                ),
                "medium": (
                    config.vehicle_operator_config.medium_removal_min_fraction,
                    config.vehicle_operator_config.medium_removal_max_fraction,
                ),
                "large": (
                    config.vehicle_operator_config.large_removal_min_fraction,
                    config.vehicle_operator_config.large_removal_max_fraction,
                ),
            }
            if tier_name not in fraction_bounds or customer_count <= 1:
                return False
            minimum, maximum = fraction_bounds[tier_name]
            lower = max(1, min(customer_count - 1, math.ceil(customer_count * minimum)))
            upper = max(
                lower,
                min(customer_count - 1, math.floor(customer_count * maximum)),
            )
            if (
                actual == 0
                and str(event.get("status")) == "time_limit"
                and str(event.get("removal_trigger", "")).startswith("probe_not_started:")
            ):
                return lower <= requested <= upper
            return lower <= requested <= upper and lower <= actual <= upper

        invalid_dynamic_events = [
            event for event in constraint_events if not _valid_dynamic_count(event)
        ]
        gates.append(
            _gate(
                "dynamic_removal_counts_within_config",
                not invalid_dynamic_events,
                len(invalid_dynamic_events),
                "every requested and actual removal count is inside its tier range",
                "Stage 2.3 constraint-lane event log and TOML",
            )
        )
    manifest_unchanged = _sha256(baseline_dir / "manifest.json") == baseline_manifest_sha256
    gates.append(
        _gate(
            "stage00_immutable",
            manifest_unchanged,
            _sha256(baseline_dir / "manifest.json"),
            baseline_manifest_sha256,
            "Stage 0 manifest SHA-256",
        )
    )
    return gates


def _load_stage1_baseline(
    path: Path,
    config: Stage02Config,
) -> dict[tuple[str, int], SolutionObjective]:
    rows = _read_csv(path)
    filtered = [row for row in rows if row.get("algorithm") == "ALNS_EXACT_CHARGING"]
    expected = {(instance, seed) for instance in config.instances for seed in config.seeds}
    actual = [(str(row["instance"]), int(row["seed"])) for row in filtered]
    if set(actual) != expected or len(actual) != len(set(actual)):
        raise RuntimeError("Stage 1 baseline coverage does not match Stage 2.1 configuration")
    if any(row.get("feasible") != "True" for row in filtered):
        raise RuntimeError("Stage 1 baseline contains an infeasible ALNS run")
    return {
        (str(row["instance"]), int(row["seed"])): SolutionObjective(
            int(row["primary_vehicle_count"]),
            float(row["secondary_total_distance"]),
            float(row["tertiary_total_charging_time"]),
            int(row["quaternary_charging_count"]),
        )
        for row in filtered
    }


def _load_comparison_baseline(
    path: Path,
    config: Stage02Config,
) -> dict[tuple[str, int], SolutionObjective]:
    if config.comparison_label == "stage01":
        return _load_stage1_baseline(path, config)
    rows = _read_csv(path)
    expected_algorithm = (
        ROUTE_QUALITY_ALGORITHM
        if config.comparison_label == "stage02_2"
        else ROUTE_REDUCTION_ALGORITHM
    )
    filtered = [row for row in rows if row.get("algorithm") == expected_algorithm]
    expected = {(instance, seed) for instance in config.instances for seed in config.seeds}
    actual = [(str(row["instance"]), int(row["seed"])) for row in filtered]
    if set(actual) != expected or len(actual) != len(set(actual)):
        raise RuntimeError(
            f"{config.comparison_label} baseline coverage does not match Stage 2 scope"
        )
    if any(row.get("feasible") != "True" for row in filtered):
        raise RuntimeError(f"{config.comparison_label} baseline contains an infeasible run")
    return {
        (str(row["instance"]), int(row["seed"])): SolutionObjective(
            int(row["primary_vehicle_count"]),
            float(row["secondary_total_distance"]),
            float(row["tertiary_total_charging_time"]),
            int(row["quaternary_charging_count"]),
        )
        for row in filtered
    }


def _load_stage00_baseline(
    baseline_dir: Path,
    benchmark_dir: Path,
    config: Stage02Config,
) -> dict[tuple[str, int], SolutionObjective]:
    rows = _read_csv(baseline_dir / "per_run_results.csv")
    output: dict[tuple[str, int], SolutionObjective] = {}
    for row in rows:
        instance_name = str(row["instance"])
        if instance_name not in config.instances:
            continue
        payload = json.loads((baseline_dir / str(row["solution_path"])).read_text(encoding="utf-8"))
        instance = parse_schneider(benchmark_dir / f"{instance_name}.txt")
        routes = payload["routes"]
        output[(instance_name, int(row["seed"]))] = SolutionObjective(
            int(row["vehicle_count"]),
            float(row["total_distance"]),
            float(row["total_charging_time"]),
            count_charging_visits(instance, routes),
        )
    expected = {(instance, seed) for instance in config.instances for seed in config.seeds}
    if set(output) != expected:
        raise RuntimeError("Stage 0 baseline coverage does not match Stage 2.1 configuration")
    return output


def _objective_from_row(row: dict[str, Any]) -> SolutionObjective:
    return SolutionObjective(
        int(row["primary_vehicle_count"]),
        float(row["secondary_total_distance"]),
        float(row["tertiary_total_charging_time"]),
        int(row["quaternary_charging_count"]),
    )


def _group_objectives(
    objectives: dict[tuple[str, int], SolutionObjective],
) -> dict[str, list[SolutionObjective]]:
    grouped: dict[str, list[SolutionObjective]] = {}
    for (instance, _), objective in objectives.items():
        grouped.setdefault(instance, []).append(objective)
    return grouped


def _metric_summary(objectives: list[SolutionObjective], metric: str) -> dict[str, Any]:
    if not objectives:
        return {
            f"mean_{metric}": "",
            f"median_{metric}": "",
            f"worst_{metric}": "",
            f"standard_deviation_{metric}": "",
        }
    values = [float(getattr(objective, metric)) for objective in objectives]
    return {
        f"mean_{metric}": statistics.fmean(values),
        f"median_{metric}": statistics.median(values),
        f"worst_{metric}": max(values),
        f"standard_deviation_{metric}": statistics.pstdev(values),
    }


def _numeric_summary(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows]
    return {
        f"mean_{field}": statistics.fmean(values),
        f"median_{field}": statistics.median(values),
        f"worst_{field}": max(values),
        f"standard_deviation_{field}": statistics.pstdev(values),
    }


def _objective_value(objective: SolutionObjective | None, field: str) -> float | int | str:
    return getattr(objective, field) if objective is not None else ""


def _comparison_row(
    instance: str,
    baseline: str,
    candidate: str,
    metric: str,
    statistic: str,
    baseline_value: Any,
    candidate_value: Any,
    absolute_delta: Any,
    relative_delta: Any,
    classification: str,
    gate_status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "instance": instance,
        "baseline": baseline,
        "candidate": candidate,
        "metric": metric,
        "statistic": statistic,
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "absolute_delta": absolute_delta,
        "relative_delta": relative_delta,
        "classification": classification,
        "gate_status": gate_status,
        "reason": reason,
    }


def _classify_metric(metric: str, baseline: float, candidate: float) -> str:
    if math.isclose(baseline, candidate, rel_tol=0.0, abs_tol=1e-9):
        return "unchanged"
    if metric in _LOWER_IS_BETTER:
        return "improvement" if candidate < baseline else "regression"
    return "improvement" if candidate > baseline else "regression"


def _relative_delta(baseline: float, candidate: float) -> float:
    return (candidate - baseline) / max(abs(baseline), 1e-9)


def _gate(gate: str, passed: bool, observed: Any, expected: Any, evidence: str) -> dict[str, Any]:
    return {
        "gate": gate,
        "status": "pass" if passed else "fail",
        "observed": observed,
        "expected": expected,
        "evidence": evidence,
    }


def _operator_failure_rows(events: list[dict[str, object]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for event in events:
        if event.get("status") not in FAILURE_EVENT_STATUSES:
            continue
        output.append(
            {
                "run_label": event.get("run_label", ""),
                "instance": event.get("instance", ""),
                "seed": event.get("seed", ""),
                "iteration": event.get("iteration", ""),
                "operator": event.get("operator", ""),
                "status": event.get("status", ""),
                "reason": event.get("reason", ""),
                "route_indices": json.dumps(event.get("route_indices", ())),
                "affected_route_indices": json.dumps(
                    event.get("affected_route_indices", ())
                ),
                "removed_customers": json.dumps(event.get("removed_customers", ())),
                "candidate_customer_sequence": json.dumps(
                    event.get("candidate_customer_sequence", ())
                ),
                "candidate_route_sequences": json.dumps(
                    event.get("candidate_route_sequences", ())
                ),
                "candidate_vehicle_delta": event.get("candidate_vehicle_delta", ""),
                "prefilter_passed": event.get("prefilter_passed", False),
                "new_routes_created": event.get("new_routes_created", 0),
                "exact_route_evaluations": event.get("exact_route_evaluations", 0),
                "selection_rank": event.get("selection_rank", 0),
                "chain_depth": event.get("chain_depth", 0),
                "segment_length": event.get("segment_length", 0),
                "track": event.get("track", "legacy"),
                "constraint_category": event.get("constraint_category", ""),
                "removal_tier": event.get("removal_tier", ""),
                "removal_size_requested": event.get("removal_size_requested", 0),
                "removal_size_actual": event.get("removal_size_actual", 0),
                "stagnation_iterations": event.get("stagnation_iterations", 0),
                "removal_trigger": event.get("removal_trigger", ""),
                "reset_observed": event.get("reset_observed", False),
                "ranking_score": event.get("ranking_score", 0.0),
            }
        )
    return output


def _repeatability_check(
    *,
    repeat_of: Path | None,
    run_label: str,
    config: Stage02Config,
    current_rows: list[dict[str, Any]],
    current_gate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    if repeat_of is None:
        return (
            [
                {
                    "first_run": "",
                    "second_run": run_label,
                    "status": "pending",
                    "first_gate_status": "",
                    "second_gate_status": "pass"
                    if all(row["status"] == "pass" for row in current_gate_rows)
                    else "fail",
                    "first_run_keys": "",
                    "second_run_keys": str(len(current_rows)),
                    "configuration_match": "pending",
                    "details": "run a second complete experiment with --repeat-of",
                }
            ],
            False,
        )

    gate_path, per_run_path = _resolve_repeat_artifacts(repeat_of)
    if gate_path is None or per_run_path is None:
        return (
            [
                {
                    "first_run": str(repeat_of),
                    "second_run": run_label,
                    "status": "fail",
                    "first_gate_status": "missing",
                    "second_gate_status": "pass"
                    if all(row["status"] == "pass" for row in current_gate_rows)
                    else "fail",
                    "first_run_keys": "missing",
                    "second_run_keys": str(len(current_rows)),
                    "configuration_match": "false",
                    "details": "could not resolve first-run gate and per-run artifacts",
                }
            ],
            False,
        )

    first_gate_rows = _read_csv(gate_path)
    first_rows = _read_csv(per_run_path)
    first_gate_status = (
        "pass"
        if first_gate_rows and all(row.get("status") == "pass" for row in first_gate_rows)
        else "fail"
    )
    second_gate_status = (
        "pass"
        if current_gate_rows and all(row["status"] == "pass" for row in current_gate_rows)
        else "fail"
    )
    expected_keys = {(instance, seed) for instance in config.instances for seed in config.seeds}
    first_keys = _per_run_keys(first_rows)
    second_keys = _per_run_keys(current_rows)
    keys_match = first_keys == expected_keys and second_keys == expected_keys
    configuration_match = _same_run_configuration(first_rows, config)
    passed = first_gate_status == "pass" and second_gate_status == "pass" and keys_match
    passed = passed and configuration_match
    details = "; ".join(
        (
            f"first_gate={first_gate_status}",
            f"second_gate={second_gate_status}",
            f"first_keys={len(first_keys)}/{len(expected_keys)}",
            f"second_keys={len(second_keys)}/{len(expected_keys)}",
            f"configuration_match={configuration_match}",
        )
    )
    return (
        [
            {
                "first_run": str(gate_path.parent),
                "second_run": run_label,
                "status": "pass" if passed else "fail",
                "first_gate_status": first_gate_status,
                "second_gate_status": second_gate_status,
                "first_run_keys": f"{len(first_keys)}/{len(expected_keys)}",
                "second_run_keys": f"{len(second_keys)}/{len(expected_keys)}",
                "configuration_match": configuration_match,
                "details": details,
            }
        ],
        passed,
    )


def _resolve_repeat_artifacts(repeat_of: Path) -> tuple[Path | None, Path | None]:
    if repeat_of.is_file():
        gate_path = repeat_of
        if not gate_path.name.endswith("_gate_report.csv"):
            return None, None
        per_run_path = gate_path.with_name(
            gate_path.name.replace("_gate_report.csv", "_per_run_results.csv")
        )
        return (gate_path, per_run_path) if per_run_path.is_file() else (None, None)
    if not repeat_of.is_dir():
        return None, None
    gate_paths = sorted(repeat_of.glob("*_gate_report.csv"))
    per_run_paths = sorted(repeat_of.glob("*_per_run_results.csv"))
    if len(gate_paths) != 1 or len(per_run_paths) != 1:
        return None, None
    return gate_paths[0], per_run_paths[0]


def _per_run_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(str(row["instance"]), int(row["seed"])) for row in rows}


def _same_run_configuration(rows: list[dict[str, str]], config: Stage02Config) -> bool:
    if len(rows) != len(config.instances) * len(config.seeds):
        return False
    for row in rows:
        if (
            row.get("algorithm") != config.algorithm
            or row.get("operator_profile") != config.operator_profile.value
            or float(row.get("time_limit_seconds", "nan")) != config.time_limit_seconds
            or int(row.get("max_iterations", "-1")) != config.max_iterations
            or int(row.get("threads", "-1")) != config.threads
        ):
            return False
    return True


def _operator_event_rows(events: list[dict[str, object]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for event in events:
        output.append(
            {
                "run_label": event.get("run_label", ""),
                "instance": event.get("instance", ""),
                "seed": event.get("seed", ""),
                "iteration": event.get("iteration", ""),
                "operator": event.get("operator", ""),
                "status": event.get("status", ""),
                "reason": event.get("reason", ""),
                "accepted": event.get("accepted", False),
                "vehicle_reduction": event.get("vehicle_reduction", False),
                "distance_improvement": event.get("distance_improvement", False),
                "candidate_objective_key": json.dumps(
                    event.get("candidate_objective_key", ())
                ),
                "route_indices": json.dumps(event.get("route_indices", ())),
                "affected_route_indices": json.dumps(
                    event.get("affected_route_indices", ())
                ),
                "removed_customers": json.dumps(event.get("removed_customers", ())),
                "candidate_customer_sequence": json.dumps(
                    event.get("candidate_customer_sequence", ())
                ),
                "candidate_route_sequences": json.dumps(
                    event.get("candidate_route_sequences", ())
                ),
                "candidate_vehicle_delta": event.get("candidate_vehicle_delta", ""),
                "candidate_feasible": event.get("candidate_feasible", False),
                "prefilter_passed": event.get("prefilter_passed", False),
                "new_routes_created": event.get("new_routes_created", 0),
                "exact_route_evaluations": event.get("exact_route_evaluations", 0),
                "selection_rank": event.get("selection_rank", 0),
                "chain_depth": event.get("chain_depth", 0),
                "segment_length": event.get("segment_length", 0),
                "track": event.get("track", "legacy"),
                "constraint_category": event.get("constraint_category", ""),
                "removal_tier": event.get("removal_tier", ""),
                "removal_size_requested": event.get("removal_size_requested", 0),
                "removal_size_actual": event.get("removal_size_actual", 0),
                "stagnation_iterations": event.get("stagnation_iterations", 0),
                "removal_trigger": event.get("removal_trigger", ""),
                "reset_observed": event.get("reset_observed", False),
                "ranking_score": event.get("ranking_score", 0.0),
            }
        )
    return output


def _violation_counts(report: SolutionReport) -> dict[str, int]:
    violations = [*report.violations, *(v for route in report.routes for v in route.violations)]
    counts = {
        "coverage_violations": 0,
        "route_structure_violations": 0,
        "capacity_violations": 0,
        "time_window_violations": 0,
        "energy_violations": 0,
        "objective_violations": 0,
        "other_violations": 0,
    }
    for violation in violations:
        lower = violation.lower()
        if "unvisited" in lower or "more than once" in lower:
            key = "coverage_violations"
        elif "route" in lower or "depot" in lower or "unknown nodes" in lower:
            key = "route_structure_violations"
        elif "load" in lower or "capacity" in lower:
            key = "capacity_violations"
        elif "arrival" in lower or "due date" in lower:
            key = "time_window_violations"
        elif "battery" in lower or "energy" in lower or "charging" in lower:
            key = "energy_violations"
        elif "claimed objective" in lower:
            key = "objective_violations"
        else:
            key = "other_violations"
        counts[key] += 1
    counts["total_violation_count"] = len(violations)
    return counts


def _environment_record(
    *,
    root: Path,
    config: Stage02Config,
    config_path: Path,
    run_label: str,
    source_hashes: dict[str, str],
    algorithm_hash: str,
    baseline_manifest_sha256: str,
    baseline_dir: Path,
    benchmark_dir: Path,
) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "experiment_id": config.experiment_id,
        "run_label": run_label,
        "algorithm": config.algorithm,
        "operator_profile": config.operator_profile.value,
        "repository_revision": _git_revision(root),
        "repository_dirty": _git_dirty(root),
        "algorithm_source_sha256": algorithm_hash,
        "algorithm_source_files": source_hashes,
        "configuration_sha256": _sha256(config_path),
        "baseline_manifest_sha256": baseline_manifest_sha256,
        "reference_repository_revision": _reference_revision(root),
        "benchmark_directory": str(benchmark_dir),
        "baseline_directory": str(baseline_dir),
        "captured_environment": collect_environment(),
    }


def _source_hashes(root: Path) -> dict[str, str]:
    paths = (
        Path("src/evrptw/alns.py"),
        Path("src/evrptw/neighborhoods.py"),
        Path("src/evrptw/objective.py"),
        Path("src/evrptw/charging.py"),
        Path("src/evrptw/validation.py"),
        Path("src/evrptw/experiments/stage02_route_reduction.py"),
        Path("src/evrptw/experiments/stage02_route_quality.py"),
        Path("src/evrptw/experiments/stage02_constraint_guided.py"),
    )
    return {str(path): _sha256(root / path) for path in paths}


def _write_manifest(directory: Path, manifest_path: Path) -> None:
    files = {
        str(path.relative_to(directory)): _sha256(path)
        for path in directory.rglob("*")
        if path.is_file() and path != manifest_path
    }
    _write_json(manifest_path, {"schema_version": SCHEMA_VERSION, "files": files})


def _tracked_names(run_label: str, comparison_label: str) -> tuple[str, ...]:
    return tuple(
        f"{run_label}_{suffix}"
        for suffix in (
            "per_run_results.csv",
            "summary_results.csv",
            "failure_cases.csv",
            "operator_summary.csv",
            "operator_events.csv",
            "operator_failure_events.csv",
            "repeatability.csv",
            f"{comparison_label}_comparison.csv",
            "100_customer_comparison.csv",
            "gate_report.csv",
            "environment.json",
            "parameters.toml",
        )
    )


def _count_feasible(rows: list[dict[str, Any]]) -> str:
    return f"{sum(bool(row['feasible']) for row in rows)}/{len(rows)}"


@cache
def _customer_count_from_event(instance_name: str, benchmark_dir: Path) -> int:
    path = _resolve(_repository_root(), benchmark_dir) / f"{instance_name}.txt"
    return len(parse_schneider(path).customers)


def _join_failures(*reasons: str) -> str:
    return "; ".join(reason for reason in reasons if reason)


def _as_int(value: object) -> int:
    if isinstance(value, (bool, int, float, str)):
        return int(value)
    raise TypeError(f"expected numeric operator statistic, got {type(value).__name__}")


def _event_int(value: object) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, (bool, int, float, str)):
        return int(value)
    raise TypeError(f"expected numeric event value, got {type(value).__name__}")


def _render_key(objective: SolutionObjective | None) -> str:
    return json.dumps(objective.key if objective is not None else (), separators=(",", ":"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _combined_hash(values: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path, value in sorted(values.items()):
        digest.update(path.encode())
        digest.update(value.encode())
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _git_dirty(root: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )


def _reference_revision(root: Path) -> str | None:
    reference = root / "reference" / "VRP-EVRP-Project-Hub"
    if not (reference / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(reference), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _validate_run_label(run_label: str) -> None:
    if not run_label or run_label in {".", ".."} or "/" in run_label or "\\" in run_label:
        raise ValueError("run_label must be a non-empty single path segment")


def _validate_config(config: Stage02Config) -> None:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported Stage 2 schema version: {config.schema_version}")
    expected_algorithms = {
        OperatorProfile.STAGE02_ROUTE_REDUCTION: ROUTE_REDUCTION_ALGORITHM,
        OperatorProfile.STAGE02_ROUTE_QUALITY: ROUTE_QUALITY_ALGORITHM,
        OperatorProfile.STAGE02_CONSTRAINT_GUIDED: CONSTRAINT_GUIDED_ALGORITHM,
    }
    if config.operator_profile not in expected_algorithms:
        raise ValueError(
            "Stage 2 runner requires a route-reduction, route-quality, "
            "or constraint-guided profile"
        )
    if config.algorithm != expected_algorithms[config.operator_profile]:
        raise ValueError(
            f"{config.operator_profile.value} only supports "
            f"{expected_algorithms[config.operator_profile]}"
        )
    if config.instances != FORMAL_INSTANCES:
        raise ValueError(
            "Stage 2 formal scope must exactly match Stage 0 instances: "
            f"{FORMAL_INSTANCES}"
        )
    if config.seeds != FORMAL_SEEDS:
        raise ValueError(
            "Stage 2 formal scope must exactly match Stage 0 seeds: " f"{FORMAL_SEEDS}"
        )
    if config.time_limit_seconds <= 0 or config.max_iterations <= 0:
        raise ValueError("time limit and maximum iterations must be positive")
    if config.threads != 1:
        raise ValueError("Stage 2 requires exactly one thread")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 2 route experiments")
    parser.add_argument("--config", type=Path, default=Path("configs/stage02_route_reduction.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage02"))
    parser.add_argument("--summary-dir", type=Path, default=Path("experiments/summaries"))
    parser.add_argument("--run-label", default="stage02")
    parser.add_argument(
        "--repeat-of",
        type=Path,
        help="first complete Stage 2 output directory or gate report to compare",
    )
    arguments = parser.parse_args()
    outputs = run_stage02(
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        summary_dir=arguments.summary_dir,
        run_label=arguments.run_label,
        repeat_of=arguments.repeat_of,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
