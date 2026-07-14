from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evrptw.alns import solve_alns
from evrptw.artifacts import (
    ArtifactBundleWriter,
    ArtifactStorageConfig,
    artifact_context_from_run_label,
    canonical_run_label_is_valid,
    require_current_storage_config,
    write_solver_result_bundle,
)
from evrptw.environment import collect_environment
from evrptw.models import Instance
from evrptw.parser import parse_schneider
from evrptw.validation import SolutionReport, validate_routes

SCHEMA_VERSION = "1"
ALGORITHM = "ALNS_EXACT_CHARGING"
FLOAT_TOLERANCE = 1e-9

PER_RUN_FIELDS = (
    "schema_version",
    "experiment_id",
    "instance",
    "seed",
    "algorithm",
    "repository_revision",
    "repository_dirty",
    "algorithm_source_sha256",
    "instance_sha256",
    "start_utc",
    "end_utc",
    "time_limit_seconds",
    "max_iterations",
    "threads",
    "vehicle_count",
    "total_distance",
    "total_energy",
    "total_charged_energy",
    "total_charging_time",
    "runtime_seconds",
    "iterations",
    "accepted_moves",
    "improving_moves",
    "rejected_moves",
    "charging_subproblem_calls",
    "charging_subproblem_average_seconds",
    "coverage_violations",
    "route_structure_violations",
    "capacity_violations",
    "time_window_violations",
    "energy_violations",
    "objective_violations",
    "other_violations",
    "total_violation_count",
    "status",
    "feasible",
    "failure_reason",
    "raw_log_path",
    "solution_path",
)

SUMMARY_FIELDS = (
    "instance",
    "metric",
    "runs",
    "best",
    "mean",
    "median",
    "worst",
    "standard_deviation",
)

COMPARISON_FIELDS = (
    "instance",
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

_LOWER_IS_BETTER = {
    "vehicle_count",
    "total_distance",
    "total_energy",
    "total_charged_energy",
    "total_charging_time",
    "runtime_seconds",
}
_HIGHER_IS_BETTER = {"iterations", "feasibility_rate"}
_COMPARISON_METRICS = _LOWER_IS_BETTER | _HIGHER_IS_BETTER
_SOLUTION_METRICS = {
    "vehicle_count",
    "total_distance",
    "total_energy",
    "total_charged_energy",
    "total_charging_time",
}
_SUMMARY_METRICS = (
    "feasibility_rate",
    "vehicle_count",
    "total_distance",
    "total_energy",
    "total_charged_energy",
    "total_charging_time",
    "runtime_seconds",
    "iterations",
    "accepted_moves",
    "improving_moves",
    "rejected_moves",
    "charging_subproblem_calls",
    "charging_subproblem_average_seconds",
)
_CORE_ALGORITHM_FILES = (
    Path("src/evrptw/alns.py"),
    Path("src/evrptw/charging.py"),
    Path("src/evrptw/validation.py"),
)


@dataclass(frozen=True, slots=True)
class Stage00Config:
    schema_version: str
    baseline_id: str
    algorithm: str
    benchmark_dir: Path
    instances: tuple[str, ...]
    seeds: tuple[int, ...]
    time_limit_seconds: float
    max_iterations: int
    threads: int
    command: str
    artifact_storage: ArtifactStorageConfig | None = None


def load_config(path: Path) -> Stage00Config:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    try:
        stage = payload["stage00"]
        benchmark = payload["benchmark"]
        run = payload["run"]
        config = Stage00Config(
            schema_version=str(stage["schema_version"]),
            baseline_id=str(stage["baseline_id"]),
            algorithm=str(stage["algorithm"]),
            benchmark_dir=Path(benchmark["directory"]),
            instances=tuple(str(value) for value in benchmark["instances"]),
            seeds=tuple(int(value) for value in run["seeds"]),
            time_limit_seconds=float(run["time_limit_seconds"]),
            max_iterations=int(run["max_iterations"]),
            threads=int(run["threads"]),
            command=str(run["command"]),
            artifact_storage=(
                ArtifactStorageConfig(**dict(payload["artifact_storage"]))
                if "artifact_storage" in payload
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid Stage 0 configuration: {error}") from error
    _validate_config(config)
    return config


def run_stage00(
    config: Stage00Config,
    config_path: Path,
    output_dir: Path,
    *,
    baseline_dir: Path | None = None,
) -> dict[str, Path]:
    if config.artifact_storage is not None and not _is_current_storage_run(output_dir):
        raise ValueError(
            "a configured new Stage 0 run must use a canonical attemptNN/rerunNN "
            "directory; non-canonical output is reserved for legacy compatibility"
        )
    if _is_current_storage_run(output_dir):
        return _run_stage00_current(
            config,
            config_path,
            output_dir,
            baseline_dir=baseline_dir,
        )
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if baseline_dir is not None and baseline_dir.exists():
        raise FileExistsError(f"baseline directory already exists: {baseline_dir}")

    output_dir.mkdir(parents=True)
    raw_dir = output_dir / "raw"
    solution_dir = output_dir / "solutions"
    raw_dir.mkdir()
    solution_dir.mkdir()

    root = _repository_root()
    revision, dirty = _git_state(root)
    source_hashes = {
        str(path): _sha256(root / path) for path in _CORE_ALGORITHM_FILES
    }
    algorithm_hash = _combined_hash(source_hashes)
    instance_hashes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for instance_name in config.instances:
        instance_path = config.benchmark_dir / f"{instance_name}.txt"
        if not instance_path.is_file():
            raise FileNotFoundError(f"benchmark instance is missing: {instance_path}")
        instance_hashes[instance_name] = _sha256(instance_path)
        instance = parse_schneider(instance_path)
        for seed in config.seeds:
            rows.append(
                _run_once(
                    config,
                    instance,
                    seed,
                    revision,
                    dirty,
                    algorithm_hash,
                    instance_hashes[instance_name],
                    output_dir,
                    raw_dir,
                    solution_dir,
                )
            )

    environment = {
        "schema_version": config.schema_version,
        "baseline_id": config.baseline_id,
        "algorithm": config.algorithm,
        "repository_revision": revision,
        "repository_dirty": dirty,
        "algorithm_source_sha256": algorithm_hash,
        "algorithm_source_files": source_hashes,
        "instance_sha256": instance_hashes,
        "configuration_sha256": _sha256(config_path),
        "reference_repository_revision": _reference_revision(root),
        "physical_memory_bytes": _physical_memory_bytes(),
        "captured_environment": collect_environment(),
    }
    outputs = _write_result_set(output_dir, config_path, rows, environment)
    if baseline_dir is not None:
        _export_curated_baseline(output_dir, baseline_dir, config_path, rows, environment)
        outputs["baseline_dir"] = baseline_dir
    return outputs


def _is_current_storage_run(output_dir: Path) -> bool:
    return canonical_run_label_is_valid(output_dir.name)


def _run_stage00_current(
    config: Stage00Config,
    config_path: Path,
    output_dir: Path,
    *,
    baseline_dir: Path | None = None,
) -> dict[str, Path]:
    root = _repository_root()
    storage = require_current_storage_config(output_dir.name, config.artifact_storage)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if baseline_dir is not None and baseline_dir.exists():
        raise FileExistsError(f"immutable baseline directory already exists: {baseline_dir}")
    context = artifact_context_from_run_label(output_dir.name)
    writer = ArtifactBundleWriter(output_dir, context, storage)
    revision, dirty = _git_state(root)
    source_hashes = {str(path): _sha256(root / path) for path in _CORE_ALGORITHM_FILES}
    algorithm_hash = _combined_hash(source_hashes)
    metadata = {
        "schema_version": config.schema_version,
        "baseline_id": config.baseline_id,
        "algorithm": config.algorithm,
        "run_label": output_dir.name,
        "scope": "formal",
        "expected_instances": list(config.instances),
        "expected_seeds": list(config.seeds),
        "benchmark_directory": str(config.benchmark_dir),
        "repository_revision": revision,
        "repository_dirty": dirty,
        "algorithm_source_sha256": algorithm_hash,
        "algorithm_source_files": source_hashes,
        "configuration_sha256": _sha256(config_path),
        "reference_repository_revision": _reference_revision(root),
        "captured_environment": collect_environment(),
    }
    writer.write_control(metadata=metadata, configuration_path=config_path)
    index_path = output_dir / "control" / f"{output_dir.name}_per_run_results.csv"
    _write_csv(index_path, PER_RUN_FIELDS, [])
    rows: list[dict[str, Any]] = []
    for instance_name in config.instances:
        instance_path = config.benchmark_dir / f"{instance_name}.txt"
        instance_hash = _sha256(instance_path)
        instance = parse_schneider(instance_path)
        for seed in config.seeds:
            started = datetime.now(UTC)
            result = solve_alns(
                instance,
                seed=seed,
                max_iterations=config.max_iterations,
                time_limit_seconds=config.time_limit_seconds,
                operator_profile="baseline",
            )
            ended = datetime.now(UTC)
            routes = [list(route) for route in result.routes]
            report = validate_routes(instance, routes, claimed_objective=result.objective_value)
            experiment_id = f"{instance.name}-{config.algorithm.lower()}-{seed}"
            row: dict[str, Any] = {
                "schema_version": config.schema_version,
                "experiment_id": experiment_id,
                "instance": instance.name,
                "seed": seed,
                "algorithm": config.algorithm,
                "repository_revision": revision,
                "repository_dirty": dirty,
                "algorithm_source_sha256": algorithm_hash,
                "instance_sha256": instance_hash,
                "start_utc": started.isoformat(),
                "end_utc": ended.isoformat(),
                "time_limit_seconds": config.time_limit_seconds,
                "max_iterations": config.max_iterations,
                "threads": config.threads,
                "vehicle_count": report.vehicle_count,
                "total_distance": report.total_distance,
                "total_energy": report.total_energy,
                "total_charged_energy": report.total_charged_energy,
                "total_charging_time": report.total_charging_time,
                "runtime_seconds": result.runtime_seconds,
                "iterations": result.iterations,
                "accepted_moves": result.accepted_moves,
                "improving_moves": result.improving_moves,
                "rejected_moves": result.rejected_moves,
                "charging_subproblem_calls": result.charging_subproblem_calls,
                "charging_subproblem_average_seconds": (
                    result.charging_subproblem_time / result.charging_subproblem_calls
                    if result.charging_subproblem_calls
                    else 0.0
                ),
                **_violation_counts(report),
                "status": "feasible" if report.feasible else "invalid",
                "feasible": report.feasible,
                "failure_reason": result.failure_reason or _violations(report),
            }
            environment = {
                **metadata,
                "instance": instance.name,
                "seed": seed,
                "instance_sha256": instance_hash,
                "run_start_utc": started.isoformat(),
                "run_end_utc": ended.isoformat(),
            }
            refs = write_solver_result_bundle(
                writer,
                instance=instance,
                seed=seed,
                result=result,
                raw_record=row,
                environment_payload=environment,
            )
            row["raw_log_path"] = refs["raw"]
            row["solution_path"] = refs["solution"]
            rows.append(row)
            _write_csv(index_path, PER_RUN_FIELDS, rows)
    writer.record_existing_file(
        index_path,
        artifact_type="per_run_results",
        retention_class="control",
        storage_format="csv_control",
        row_count=len(rows),
    )
    bundle = writer.finalize()
    outputs: dict[str, Path] = {
        "output_dir": output_dir,
        "per_run": index_path,
        "manifest": bundle.manifest_path,
    }
    if baseline_dir is not None:
        baseline_environment = {
            **metadata,
            "baseline_view_of": str(output_dir),
            "physical_memory_bytes": _physical_memory_bytes(),
        }
        _export_current_baseline_view(
            output_dir,
            baseline_dir,
            config_path,
            rows,
            baseline_environment,
        )
        outputs["baseline_dir"] = baseline_dir
    return outputs


def verify_results(
    config: Stage00Config,
    results_dir: Path,
    *,
    require_manifest: bool = False,
) -> int:
    if require_manifest:
        _verify_manifest(results_dir)
    rows = _read_csv(results_dir / "per_run_results.csv")
    _verify_run_keys(config, rows)
    failures: list[str] = []
    for row in rows:
        solution_path = results_dir / row["solution_path"]
        if not solution_path.is_file():
            failures.append(f"missing solution: {solution_path}")
            continue
        payload = json.loads(solution_path.read_text(encoding="utf-8"))
        instance_name = row["instance"]
        instance = parse_schneider(config.benchmark_dir / f"{instance_name}.txt")
        routes = [list(route) for route in payload.get("routes", [])]
        report = validate_routes(
            instance,
            routes,
            claimed_objective=_float(row, "total_distance"),
        )
        failures.extend(_verify_row_against_report(row, report))

    expected_summary = _summarize(rows)
    actual_summary = _read_csv(results_dir / "summary_results.csv")
    if _normalized_rows(actual_summary, SUMMARY_FIELDS) != _normalized_rows(
        expected_summary, SUMMARY_FIELDS
    ):
        failures.append("summary_results.csv cannot be recomputed from per_run_results.csv")

    expected_failures = [row for row in rows if not _bool(row["feasible"])]
    actual_failures = _read_csv(results_dir / "failure_cases.csv")
    if _normalized_rows(actual_failures, PER_RUN_FIELDS) != _normalized_rows(
        expected_failures, PER_RUN_FIELDS
    ):
        failures.append("failure_cases.csv does not retain every failed run")
    if failures:
        raise ValueError("Stage 0 verification failed: " + " | ".join(failures))
    return len(rows)


def compare_results(
    config: Stage00Config,
    baseline_dir: Path,
    candidate_dir: Path,
    report_path: Path,
) -> list[dict[str, Any]]:
    baseline_summary = _summary_index(_read_csv(baseline_dir / "summary_results.csv"))
    candidate_summary = _summary_index(_read_csv(candidate_dir / "summary_results.csv"))
    output: list[dict[str, Any]] = []

    comparison_keys = {
        key
        for key in set(baseline_summary) | set(candidate_summary)
        if key[1] in _COMPARISON_METRICS
    }
    for key in sorted(comparison_keys):
        instance, metric = key
        baseline = baseline_summary.get(key)
        candidate = candidate_summary.get(key)
        if baseline is None or candidate is None:
            output.append(
                _comparison_row(
                    instance,
                    metric,
                    "record",
                    "" if baseline is None else "present",
                    "" if candidate is None else "present",
                    "",
                    "",
                    "regression",
                    "fail",
                    "expected summary metric is missing",
                )
            )
            continue
        for statistic in ("best", "mean", "median", "worst", "standard_deviation"):
            baseline_value = baseline[statistic]
            candidate_value = candidate[statistic]
            if baseline_value == "" and candidate_value == "":
                continue
            if baseline_value == "" or candidate_value == "":
                output.append(
                    _comparison_row(
                        instance,
                        metric,
                        statistic,
                        baseline_value,
                        candidate_value,
                        "",
                        "",
                        "regression",
                        "fail" if metric == "feasibility_rate" else "pass",
                        "candidate statistic is unavailable",
                    )
                )
                continue
            baseline_number = float(baseline_value)
            candidate_number = float(candidate_value)
            delta = candidate_number - baseline_number
            relative = delta / max(abs(baseline_number), FLOAT_TOLERANCE)
            classification = _classify(metric, statistic, baseline_number, candidate_number)
            gate = (
                "fail"
                if metric == "feasibility_rate" and classification == "regression"
                else "pass"
            )
            reason = _comparison_reason(metric, statistic, classification)
            output.append(
                _comparison_row(
                    instance,
                    metric,
                    statistic,
                    baseline_value,
                    candidate_value,
                    delta,
                    relative,
                    classification,
                    gate,
                    reason,
                )
            )

    baseline_rows = _read_csv(baseline_dir / "per_run_results.csv")
    candidate_rows = _read_csv(candidate_dir / "per_run_results.csv")
    output.extend(
        _comparison_gate_rows(
            config,
            baseline_rows,
            candidate_rows,
            baseline_dir,
            candidate_dir,
            list(baseline_summary.values()),
            list(candidate_summary.values()),
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(report_path, COMPARISON_FIELDS, output)
    return output


def _validate_config(config: Stage00Config) -> None:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Stage 0 schema version {config.schema_version}; expected {SCHEMA_VERSION}"
        )
    if config.algorithm != ALGORITHM:
        raise ValueError(f"Stage 0 only supports {ALGORITHM}")
    if not config.instances or len(set(config.instances)) != len(config.instances):
        raise ValueError("instances must be non-empty and unique")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be non-empty and unique")
    if config.time_limit_seconds <= 0 or config.max_iterations <= 0:
        raise ValueError("time limit and maximum iterations must be positive")
    if config.threads != 1:
        raise ValueError("Stage 0 requires exactly one thread")


def _run_once(
    config: Stage00Config,
    instance: Instance,
    seed: int,
    revision: str,
    dirty: bool,
    algorithm_hash: str,
    instance_hash: str,
    output_dir: Path,
    raw_dir: Path,
    solution_dir: Path,
) -> dict[str, Any]:
    start = datetime.now(UTC)
    result = solve_alns(
        instance,
        seed=seed,
        max_iterations=config.max_iterations,
        time_limit_seconds=config.time_limit_seconds,
        operator_profile="baseline",
    )
    end = datetime.now(UTC)
    routes = [list(route) for route in result.routes]
    report = validate_routes(instance, routes, claimed_objective=result.objective_value)
    violation_counts = _violation_counts(report)
    experiment_id = f"{instance.name}-{config.algorithm.lower()}-{seed}"
    raw_path = raw_dir / f"{experiment_id}.json"
    solution_path = solution_dir / f"{experiment_id}.json"
    failure_reason = result.failure_reason or _violations(report)
    row: dict[str, Any] = {
        "schema_version": config.schema_version,
        "experiment_id": experiment_id,
        "instance": instance.name,
        "seed": seed,
        "algorithm": config.algorithm,
        "repository_revision": revision,
        "repository_dirty": dirty,
        "algorithm_source_sha256": algorithm_hash,
        "instance_sha256": instance_hash,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "time_limit_seconds": config.time_limit_seconds,
        "max_iterations": config.max_iterations,
        "threads": config.threads,
        "vehicle_count": report.vehicle_count,
        "total_distance": report.total_distance,
        "total_energy": report.total_energy,
        "total_charged_energy": report.total_charged_energy,
        "total_charging_time": report.total_charging_time,
        "runtime_seconds": result.runtime_seconds,
        "iterations": result.iterations,
        "accepted_moves": result.accepted_moves,
        "improving_moves": result.improving_moves,
        "rejected_moves": result.rejected_moves,
        "charging_subproblem_calls": result.charging_subproblem_calls,
        "charging_subproblem_average_seconds": (
            result.charging_subproblem_time / result.charging_subproblem_calls
            if result.charging_subproblem_calls
            else 0.0
        ),
        **violation_counts,
        "status": "feasible" if report.feasible else "invalid",
        "feasible": report.feasible,
        "failure_reason": failure_reason,
        "raw_log_path": str(raw_path.relative_to(output_dir)),
        "solution_path": str(solution_path.relative_to(output_dir)),
    }
    _write_json(solution_path, {"instance": instance.name, "seed": seed, "routes": routes})
    _write_json(raw_path, {"record": row, "solver_result": asdict(result)})
    return row


def _write_result_set(
    directory: Path,
    config_path: Path,
    rows: list[dict[str, Any]],
    environment: dict[str, Any],
) -> dict[str, Path]:
    per_run = directory / "per_run_results.csv"
    summary = directory / "summary_results.csv"
    failures = directory / "failure_cases.csv"
    comparison_template = directory / "comparison_template.csv"
    environment_path = directory / "environment.json"
    parameters = directory / "parameters.toml"
    _write_csv(per_run, PER_RUN_FIELDS, rows)
    _write_csv(summary, SUMMARY_FIELDS, _summarize(rows))
    _write_csv(failures, PER_RUN_FIELDS, [row for row in rows if not _bool(row["feasible"])])
    _write_csv(comparison_template, COMPARISON_FIELDS, [])
    _write_json(environment_path, environment)
    shutil.copy2(config_path, parameters)
    _write_manifest(directory)
    return {
        "per_run": per_run,
        "summary": summary,
        "failures": failures,
        "comparison_template": comparison_template,
        "environment": environment_path,
        "parameters": parameters,
        "manifest": directory / "manifest.json",
    }


def _export_curated_baseline(
    output_dir: Path,
    baseline_dir: Path,
    config_path: Path,
    rows: list[dict[str, Any]],
    environment: dict[str, Any],
) -> None:
    baseline_dir.mkdir(parents=True)
    shutil.copytree(output_dir / "solutions", baseline_dir / "solutions")
    curated_rows = [dict(row) for row in rows]
    for row in curated_rows:
        row["raw_log_path"] = str((output_dir / str(row["raw_log_path"])).resolve())
    _write_result_set(baseline_dir, config_path, curated_rows, environment)


def _export_current_baseline_view(
    output_dir: Path,
    baseline_dir: Path,
    config_path: Path,
    rows: list[dict[str, Any]],
    environment: dict[str, Any],
) -> None:
    """Create a compatibility baseline view after current evidence is complete.

    The current Parquet/JSON bundle remains the source of truth.  The view
    contains only the legacy solution index and summary files required by
    existing Stage 1/2 consumers; each raw path points back to the immutable
    current bundle instead of copying raw evidence into the frozen directory.
    """

    baseline_dir.mkdir(parents=True)
    solution_dir = baseline_dir / "solutions"
    solution_dir.mkdir()
    curated_rows: list[dict[str, Any]] = []
    for row in rows:
        current_solution = output_dir / str(row["solution_path"])
        if not current_solution.is_file():
            raise FileNotFoundError(f"current solution artifact is missing: {current_solution}")
        solution_name = f"{row['experiment_id']}.json"
        destination = solution_dir / solution_name
        shutil.copy2(current_solution, destination)
        curated_row = dict(row)
        curated_row["solution_path"] = str(destination.relative_to(baseline_dir))
        curated_row["raw_log_path"] = str(
            (output_dir / str(row["raw_log_path"])).resolve()
        )
        curated_rows.append(curated_row)
    _write_result_set(baseline_dir, config_path, curated_rows, environment)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_instance: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_instance.setdefault(row["instance"], []).append(row)
    summary: list[dict[str, Any]] = []
    for instance, group in sorted(by_instance.items()):
        feasible_flags = [1.0 if _bool(row["feasible"]) else 0.0 for row in group]
        summary.append(
            {
                "instance": instance,
                "metric": "feasibility_rate",
                "runs": len(group),
                "best": "",
                "mean": statistics.fmean(feasible_flags),
                "median": "",
                "worst": "",
                "standard_deviation": "",
            }
        )
        for metric in _SUMMARY_METRICS[1:]:
            selected = (
                [row for row in group if _bool(row["feasible"])]
                if metric in _SOLUTION_METRICS
                else group
            )
            values = [float(row[metric]) for row in selected if row[metric] != ""]
            summary.append(summarize_metric_values(instance, metric, values))
    return summary


def summarize_metric_values(
    instance: str, metric: str, values: list[float]
) -> dict[str, Any]:
    if not values:
        return {
            "instance": instance,
            "metric": metric,
            "runs": 0,
            "best": "",
            "mean": "",
            "median": "",
            "worst": "",
            "standard_deviation": "",
        }
    higher_is_better = metric in _HIGHER_IS_BETTER
    return {
        "instance": instance,
        "metric": metric,
        "runs": len(values),
        "best": max(values) if higher_is_better else min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "worst": min(values) if higher_is_better else max(values),
        "standard_deviation": statistics.pstdev(values),
    }


def _verify_run_keys(config: Stage00Config, rows: list[dict[str, str]]) -> None:
    expected = {(instance, str(seed)) for instance in config.instances for seed in config.seeds}
    actual_list = [(row.get("instance", ""), row.get("seed", "")) for row in rows]
    actual = set(actual_list)
    duplicates = sorted(key for key in actual if actual_list.count(key) > 1)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if duplicates:
        raise ValueError(f"duplicate run records: {duplicates}")
    if missing:
        raise ValueError(f"missing run records: {missing}")
    if unexpected:
        raise ValueError(f"unexpected run records: {unexpected}")


def _verify_row_against_report(row: dict[str, str], report: SolutionReport) -> list[str]:
    identifier = row["experiment_id"]
    failures: list[str] = []
    if not report.feasible:
        failures.append(f"{identifier} failed unified validator: {_violations(report)}")
    if not _bool(row["feasible"]):
        failures.append(f"{identifier} is not recorded as feasible")
    expected_values = {
        "vehicle_count": float(report.vehicle_count),
        "total_distance": report.total_distance,
        "total_energy": report.total_energy,
        "total_charged_energy": report.total_charged_energy,
        "total_charging_time": report.total_charging_time,
    }
    for field, expected in expected_values.items():
        actual = _float(row, field)
        if not math.isclose(actual, expected, rel_tol=FLOAT_TOLERANCE, abs_tol=FLOAT_TOLERANCE):
            failures.append(
                f"{identifier} {field} mismatch: recorded={actual}, recomputed={expected}"
            )
    counts = _violation_counts(report)
    for field, expected in counts.items():
        if int(row[field]) != expected:
            failures.append(f"{identifier} {field} mismatch")
    return failures


def _violation_counts(report: SolutionReport) -> dict[str, int]:
    counts = {
        "coverage_violations": 0,
        "route_structure_violations": 0,
        "capacity_violations": 0,
        "time_window_violations": 0,
        "energy_violations": 0,
        "objective_violations": 0,
        "other_violations": 0,
    }
    violations = [*report.violations]
    violations.extend(value for route in report.routes for value in route.violations)
    for violation in violations:
        if violation.startswith(("unvisited customers", "customers visited more than once")):
            counts["coverage_violations"] += 1
        elif violation.startswith(("route must", "depot may", "unknown nodes")):
            counts["route_structure_violations"] += 1
        elif "exceeds capacity" in violation:
            counts["capacity_violations"] += 1
        elif "exceeds due date" in violation:
            counts["time_window_violations"] += 1
        elif "battery depleted" in violation:
            counts["energy_violations"] += 1
        elif violation.startswith("claimed objective"):
            counts["objective_violations"] += 1
        else:
            counts["other_violations"] += 1
    counts["total_violation_count"] = len(violations)
    return counts


def _classify(metric: str, statistic: str, baseline: float, candidate: float) -> str:
    if math.isclose(
        baseline,
        candidate,
        rel_tol=FLOAT_TOLERANCE,
        abs_tol=FLOAT_TOLERANCE,
    ):
        return "unchanged"
    lower_is_better = statistic == "standard_deviation" or metric in _LOWER_IS_BETTER
    if metric in _HIGHER_IS_BETTER and statistic != "standard_deviation":
        lower_is_better = False
    improved = candidate < baseline if lower_is_better else candidate > baseline
    return "improvement" if improved else "regression"


def _comparison_reason(metric: str, statistic: str, classification: str) -> str:
    if classification == "unchanged":
        return f"{metric} {statistic} is equal within tolerance {FLOAT_TOLERANCE:g}"
    direction = (
        "lower"
        if statistic == "standard_deviation" or metric in _LOWER_IS_BETTER
        else "higher"
    )
    if classification == "improvement":
        return f"candidate improved because {direction} {metric} {statistic} is better"
    return f"candidate regressed because {direction} {metric} {statistic} is better"


def _comparison_gate_rows(
    config: Stage00Config,
    baseline_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    baseline_dir: Path,
    candidate_dir: Path,
    baseline_summary: list[dict[str, str]],
    candidate_summary: list[dict[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    expected = {(instance, str(seed)) for instance in config.instances for seed in config.seeds}
    output.extend(_run_key_gate_rows("baseline", expected, baseline_rows))
    output.extend(_run_key_gate_rows("candidate", expected, candidate_rows))
    output.extend(_summary_integrity_gate_rows("baseline", baseline_rows, baseline_summary))
    output.extend(_summary_integrity_gate_rows("candidate", candidate_rows, candidate_summary))
    output.extend(_solution_validation_gate_rows(config, candidate_rows, candidate_dir))
    output.extend(_manifest_gate_rows("baseline", baseline_dir))
    output.extend(_manifest_gate_rows("candidate", candidate_dir))

    invalid = [row for row in candidate_rows if not _bool(row["feasible"])]
    failure_rows = _read_csv(candidate_dir / "failure_cases.csv")
    retained = {row["experiment_id"] for row in failure_rows}
    for row in invalid:
        if row["experiment_id"] not in retained:
            output.append(
                _comparison_row(
                    row["instance"],
                    "failure_record_retention",
                    f"seed_{row['seed']}",
                    "required",
                    "missing",
                    "",
                    "",
                    "regression",
                    "fail",
                    "candidate failure record was deleted",
                )
            )
    return output


def _run_key_gate_rows(
    source: str,
    expected: set[tuple[str, str]],
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    keys = [(row.get("instance", ""), row.get("seed", "")) for row in rows]
    actual = set(keys)
    issues: list[tuple[str, str, str]] = []
    issues.extend((instance, seed, "missing") for instance, seed in sorted(expected - actual))
    issues.extend((instance, seed, "unexpected") for instance, seed in sorted(actual - expected))
    issues.extend(
        (instance, seed, "duplicate")
        for instance, seed in sorted(key for key in actual if keys.count(key) > 1)
    )
    return [
        _comparison_row(
            instance,
            "run_coverage",
            f"{source}_seed_{seed}",
            "exactly_one",
            issue,
            "",
            "",
            "regression",
            "fail",
            f"{source} instance/seed record is {issue}",
        )
        for instance, seed, issue in issues
    ]


def _summary_integrity_gate_rows(
    source: str,
    per_run_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if _normalized_rows(_summarize(per_run_rows), SUMMARY_FIELDS) == _normalized_rows(
        summary_rows, SUMMARY_FIELDS
    ):
        return []
    return [
        _comparison_row(
            "*",
            "summary_integrity",
            source,
            "recomputable",
            "mismatch",
            "",
            "",
            "regression",
            "fail",
            f"{source} summary cannot be recomputed from per-run records",
        )
    ]


def _solution_validation_gate_rows(
    config: Stage00Config,
    rows: list[dict[str, str]],
    directory: Path,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        reason = ""
        try:
            solution_path = directory / row["solution_path"]
            payload = json.loads(solution_path.read_text(encoding="utf-8"))
            instance = parse_schneider(config.benchmark_dir / f"{row['instance']}.txt")
            report = validate_routes(
                instance,
                [list(route) for route in payload.get("routes", [])],
                claimed_objective=_float(row, "total_distance"),
            )
            reason = " | ".join(_verify_row_against_report(row, report))
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            reason = str(error)
        if reason:
            output.append(
                _comparison_row(
                    row.get("instance", "*"),
                    "validator_feasibility",
                    f"seed_{row.get('seed', '')}",
                    "validator_feasible",
                    "invalid",
                    "",
                    "",
                    "regression",
                    "fail",
                    reason,
                )
            )
    return output


def _manifest_gate_rows(source: str, directory: Path) -> list[dict[str, Any]]:
    try:
        _verify_manifest(directory)
    except ValueError as error:
        return [
            _comparison_row(
                "*",
                "manifest_integrity",
                source,
                "valid",
                "invalid",
                "",
                "",
                "regression",
                "fail",
                str(error),
            )
        ]
    return []


def _comparison_row(
    instance: str,
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


def _summary_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    output: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["instance"], row["metric"])
        if key in output:
            raise ValueError(f"duplicate summary record: {key}")
        output[key] = row
    return output


def _write_manifest(directory: Path) -> None:
    files = {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    _write_json(
        directory / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "hash_algorithm": "sha256",
            "files": files,
        },
    )


def _verify_manifest(directory: Path) -> None:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("manifest schema_version is invalid")
    if payload.get("hash_algorithm") != "sha256":
        raise ValueError("manifest hash_algorithm is invalid")
    expected = payload.get("files", {})
    actual = {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    if expected != actual:
        changed = sorted(
            path for path in set(expected) | set(actual) if expected.get(path) != actual.get(path)
        )
        raise ValueError(f"manifest checksum mismatch: {changed}")
    _verify_tracked_manifest_unchanged(manifest_path)


def _verify_tracked_manifest_unchanged(manifest_path: Path) -> None:
    root = _repository_root()
    try:
        relative = manifest_path.resolve().relative_to(root)
    except ValueError:
        return
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode == 1:
        return
    if tracked.returncode != 0:
        raise RuntimeError(
            "failed to inspect manifest tracking state: " + tracked.stderr.strip()
        )
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(relative)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if unchanged.returncode == 1:
        raise ValueError("tracked manifest differs from the committed Git trust anchor")
    if unchanged.returncode != 0:
        raise RuntimeError("failed to verify manifest trust anchor: " + unchanged.stderr.strip())


def _violations(report: SolutionReport) -> str:
    values = [*report.violations]
    values.extend(violation for route in report.routes for violation in route.violations)
    return " | ".join(values)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_state(root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return revision, bool(dirty)


def _reference_revision(root: Path) -> str | None:
    reference = root / "reference" / "VRP-EVRP-Project-Hub"
    if not (reference / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=reference,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _physical_memory_bytes() -> int:
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = os.sysconf("SC_PHYS_PAGES")
    return int(page_size * page_count)


def _combined_hash(values: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(f"{name}\0{value}\n".encode())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"expected a boolean value, got {value!r}")
    return normalized == "true"


def _float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"required CSV is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalized_rows(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> list[tuple[str, ...]]:
    return [tuple(str(row.get(field, "")) for field in fields) for row in rows]


def _write_csv(
    path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze and compare the Stage 0 ALNS baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the frozen Stage 0 experiment")
    run_parser.add_argument("--config", type=Path, default=Path("configs/stage00_baseline.toml"))
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--baseline-dir", type=Path)

    verify_parser = subparsers.add_parser("verify", help="verify Stage 0 artifacts")
    verify_parser.add_argument(
        "--config", type=Path, default=Path("configs/stage00_baseline.toml")
    )
    verify_parser.add_argument("--results-dir", type=Path, required=True)
    verify_parser.add_argument("--require-manifest", action="store_true")

    compare_parser = subparsers.add_parser("compare", help="compare candidate against Stage 0")
    compare_parser.add_argument(
        "--config", type=Path, default=Path("configs/stage00_baseline.toml")
    )
    compare_parser.add_argument("--baseline-dir", type=Path, required=True)
    compare_parser.add_argument("--candidate-dir", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path, required=True)

    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.command == "run":
        outputs = run_stage00(
            config,
            arguments.config,
            arguments.output_dir,
            baseline_dir=arguments.baseline_dir,
        )
        for name, path in outputs.items():
            print(f"{name}: {path}")
    elif arguments.command == "verify":
        count = verify_results(
            config,
            arguments.results_dir,
            require_manifest=arguments.require_manifest,
        )
        print(f"verified run records: {count}")
    else:
        rows = compare_results(
            config,
            arguments.baseline_dir,
            arguments.candidate_dir,
            arguments.report,
        )
        failed = sum(row["gate_status"] == "fail" for row in rows)
        print(f"comparison rows: {len(rows)}; failed gates: {failed}")
        if failed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
