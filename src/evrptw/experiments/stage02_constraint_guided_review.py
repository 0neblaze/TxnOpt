from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tomllib
from pathlib import Path
from typing import Any, cast

from evrptw.experiments.stage02_route_reduction import (
    FOCUSED_R_RC_INSTANCES,
    FORMAL_INSTANCES,
    FORMAL_SEEDS,
    _combined_hash,
    _load_stage00_baseline,
    _load_stage1_baseline,
    _read_csv,
    _repository_root,
    _sha256,
    _source_hashes,
    load_config,
)
from evrptw.objective import ObjectiveComparison, SolutionObjective, compare_objectives
from evrptw.parser import parse_schneider
from evrptw.validation import SolutionReport, validate_routes

REVIEW_SCHEMA_VERSION = "1"
REVIEW_FINDING_FIELDS = (
    "finding",
    "status",
    "observed",
    "expected",
    "evidence",
)
FAILURE_ANALYSIS_FIELDS = (
    "instance",
    "seed",
    "operator",
    "objective",
    "feasible",
    "first_failure_location",
    "constraint_trigger",
    "root_cause_classification",
    "next_step",
)
READINESS_FIELDS = (
    "instance",
    "seed",
    "stage02_2_objective",
    "stage02_3_objective",
    "stage02_2_feasible",
    "stage02_3_feasible",
    "stage02_2_iterations",
    "stage02_3_iterations",
    "stage02_2_effective_iterations",
    "stage02_3_effective_iterations",
    "stage02_2_exact_charging_calls",
    "stage02_3_exact_charging_calls",
    "stage02_2_cache_hits",
    "stage02_2_cache_misses",
    "stage02_2_unique_route_evaluations",
    "stage02_3_cache_hits",
    "stage02_3_cache_misses",
    "stage02_3_unique_route_evaluations",
    "stage02_2_runtime_seconds",
    "stage02_3_runtime_seconds",
    "stage02_2_removal_tier_counts",
    "stage02_3_removal_tier_counts",
    "stage02_2_failure_events",
    "stage02_3_failure_events",
    "stage02_2_accepted_moves",
    "stage02_3_accepted_moves",
    "stage02_2_constraint_statistics",
    "stage02_3_constraint_statistics",
    "objective_comparison",
    "readiness_status",
)
STAGE02_2_METRIC_FIELDS = (
    "effective_iterations",
    "cache_hits",
    "cache_misses",
    "unique_route_evaluations",
    "removal_tier_counts",
    "constraint_operator_statistics_json",
)
READINESS_BASELINE_METRIC_FIELDS = (
    "stage02_2_effective_iterations",
    "stage02_2_cache_hits",
    "stage02_2_cache_misses",
    "stage02_2_unique_route_evaluations",
    "stage02_2_removal_tier_counts",
    "stage02_2_constraint_statistics",
)


def review_run(
    *,
    run_dir: Path,
    comparison_dir: Path,
    review_label: str,
) -> dict[str, Path]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Stage 2.3 run directory is missing: {run_dir}")
    rows = _load_single_per_run(run_dir)
    candidate_records = _load_candidate_records(run_dir, rows)
    operator_events, failure_events = _load_event_logs(run_dir)
    comparison_rows = _load_comparison_rows(comparison_dir)
    root = _repository_root()
    comparison_metric_rows = _load_comparison_metric_rows(root, comparison_dir)
    environment = _load_environment(run_dir)
    provenance = _review_provenance(root, environment)
    configuration_ok, configuration_observed = _formal_run_configuration(
        root, run_dir, rows, environment
    )
    time_budget_ok, time_budget_observed = _time_budget_exemption(
        run_dir, rows, operator_events
    )

    findings: list[dict[str, Any]] = []
    expected_keys = {(instance, seed) for instance in FORMAL_INSTANCES for seed in FORMAL_SEEDS}
    actual_keys = {(str(row.get("instance")), int(row.get("seed", "-1"))) for row in rows}
    unique = len(actual_keys) == len(rows)
    findings.append(
        _finding(
            "run_coverage_unique",
            actual_keys == expected_keys and unique,
            f"{len(rows)} rows, {len(actual_keys)} unique keys",
            f"{len(expected_keys)} complete unique (instance, seed) keys",
            "candidate per-run CSV",
        )
    )

    validator_results = _revalidate_records(root, candidate_records)
    findings.append(
        _finding(
            "raw_solution_validator",
            all(item["report"].feasible for item in validator_results),
            (
                f"{sum(item['report'].feasible for item in validator_results)}"
                f"/{len(validator_results)}"
            ),
            "36/36 raw solutions pass unified validator",
            "raw solution JSON and validation replay",
        )
    )
    objective_consistent = all(
        item["objective_consistent"] for item in validator_results
    )
    findings.append(
        _finding(
            "raw_objective_recomputation",
            objective_consistent,
            (
                f"{sum(item['objective_consistent'] for item in validator_results)}"
                f"/{len(validator_results)}"
            ),
            "validator objective equals solver objective for every raw solution",
            "raw solver_result objective and validator recomputation",
        )
    )

    artifact_findings = _artifact_findings(
        run_dir,
        rows,
        candidate_records,
        environment,
        operator_events,
        failure_events,
    )
    findings.extend(artifact_findings)
    findings.append(
        _finding(
            "stage00_manifest_unchanged",
            _stage00_manifest_unchanged(root, environment),
            environment.get("baseline_manifest_sha256", ""),
            "current Stage 0 manifest SHA-256",
            "Stage 0 manifest and run environment",
        )
    )
    findings.append(
        _finding(
            "reproducibility_provenance_complete",
            bool(provenance["complete"]),
            provenance,
            "dependency lockfiles, native extension hash, package versions, and Python metadata",
            "current repository files and captured environment metadata",
        )
    )
    findings.append(
        _finding(
            "formal_run_configuration",
            configuration_ok,
            configuration_observed,
            (
                "algorithm/profile/time limit/max iterations/threads match the "
                "recorded formal protocol"
            ),
            "run parameters TOML, per-run rows, and environment JSON",
        )
    )
    findings.append(
        _finding(
            "time_budget_numerical_reproducibility_exemption",
            time_budget_ok,
            time_budget_observed,
            (
                "declared 30-second budget and event-backed cooperative-cutoff "
                "exemption for any overrun"
            ),
            "per-run runtime, timeout events, and environment provenance",
        )
    )

    failure_rows = _failure_analysis_rows(rows, candidate_records, operator_events)
    findings.append(
        _finding(
            "constraint_level_failure_diagnosis",
            len(failure_rows) >= 3,
            len(failure_rows),
            "at least 3 real raw failure or poor-quality cases",
            "raw operator event logs",
        )
    )
    event_replay_ok, event_replay_observed = _replay_constraint_events(
        root, run_dir, operator_events
    )
    findings.append(
        _finding(
            "constraint_event_replay",
            event_replay_ok,
            event_replay_observed,
            "raw event CSV independently satisfies Stage 2.3 event gates",
            "raw operator event CSV and parameters TOML",
        )
    )
    readiness_rows = _readiness_rows(rows, comparison_rows, comparison_metric_rows)
    metric_integrity_ok, metric_integrity_observed = _comparison_metric_integrity(
        root,
        comparison_rows,
        comparison_metric_rows,
    )
    findings.append(
        _finding(
            "stage02_2_metric_supplement_provenance",
            metric_integrity_ok,
            metric_integrity_observed,
            (
                "the fixed Stage 2.2 metric supplement records the formal protocol, "
                "source/instance hashes, feasible objectives, and complete coverage"
            ),
            "tracked Stage 2.2 readiness metric supplement",
        )
    )
    findings.append(
        _finding(
            "stage03_readiness_completeness",
            len(readiness_rows) == len(expected_keys)
            and all(row["readiness_status"] not in {"missing", "missing_baseline_metrics"}
                    for row in readiness_rows)
            and all(
                row[field] not in {"", "not_recorded_in_stage02_2_summary"}
                for row in readiness_rows
                for field in READINESS_BASELINE_METRIC_FIELDS
            ),
            f"{len(readiness_rows)}/{len(expected_keys)} rows",
            "one complete Stage 2.2 versus Stage 2.3 row per run with recorded metrics",
            "stage03_readiness.csv and Stage 2.2 metric supplement",
        )
    )
    gate_report_present = _gate_report_present(run_dir)
    findings.append(
        _finding(
            "runner_gate_report_complete",
            gate_report_present,
            "present" if gate_report_present else "missing or duplicated",
            "exactly one runner gate report is present",
            "candidate gate report artifact",
        )
    )
    independent_replay_ok, independent_replay_observed = _replay_independent_rerun(
        root,
        run_dir,
        rows,
        comparison_dir,
    )
    findings.append(
        _finding(
            "independent_complete_rerun_replay",
            independent_replay_ok,
            independent_replay_observed,
            "repeatability evidence and both complete runs are independently verified",
            "repeatability CSV, first-run raw solutions, and gate reports",
        )
    )
    candidate_gate_status, gate_replay_observed = _replay_runner_hard_gates(
        root,
        run_dir,
        rows,
        validator_results,
        comparison_rows,
        operator_events,
        event_replay_ok,
        independent_replay_ok,
    )
    findings.append(
        _finding(
            "runner_hard_gates",
            candidate_gate_status and gate_report_present,
            gate_replay_observed,
            "independent replay of all Stage 2.3 hard gates passes",
            "raw solutions, validator results, baseline files, and raw event CSV",
        )
    )
    independent_pending = (
        independent_replay_observed.get("status")
        == "first_complete_run_pending_independent_rerun"
    )
    if independent_pending:
        for finding in findings:
            if finding["finding"] in {
                "independent_complete_rerun_replay",
                "runner_hard_gates",
            }:
                finding["status"] = "pending"

    ready = all(row["status"] == "pass" for row in findings)
    review_status = (
        "PENDING_INDEPENDENT_RERUN"
        if independent_pending and not ready
        else "READY_FOR_STAGE03"
        if ready
        else "NOT_READY_FOR_STAGE03"
    )
    paths = _write_review_artifacts(
        run_dir=run_dir,
        review_label=review_label,
        findings=findings,
        failure_rows=failure_rows,
        readiness_rows=readiness_rows,
        environment=environment,
        ready=ready,
        review_status=review_status,
        review_metadata={
            "formal_run_configuration": configuration_observed,
            "time_budget_exemption": time_budget_observed,
        },
    )
    blocking_findings = _blocking_review_findings(findings, independent_pending)
    if blocking_findings or (not ready and not independent_pending):
        failed = ", ".join(blocking_findings)
        raise RuntimeError(f"Stage 2.3 independent review failed: {failed}")
    return paths


def _load_single_per_run(run_dir: Path) -> list[dict[str, str]]:
    paths = sorted(run_dir.glob("*_per_run_results.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one per-run CSV in {run_dir}, found {paths}")
    return _read_csv(paths[0])


def _load_candidate_records(
    run_dir: Path,
    rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        raw_path = run_dir / str(row.get("raw_log_path", ""))
        solution_path = run_dir / str(row.get("solution_path", ""))
        if not raw_path.is_file() or not solution_path.is_file():
            records.append({"row": row, "raw": {}, "solution": {}, "missing": True})
            continue
        records.append(
            {
                "row": row,
                "raw": json.loads(raw_path.read_text(encoding="utf-8")),
                "solution": json.loads(solution_path.read_text(encoding="utf-8")),
                "raw_path": raw_path,
                "solution_path": solution_path,
                "missing": False,
            }
        )
    return records


def _load_event_logs(
    run_dir: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    event_paths = sorted(run_dir.glob("*_operator_events.csv"))
    failure_paths = sorted(run_dir.glob("*_operator_failure_events.csv"))
    events = _read_csv(event_paths[0]) if len(event_paths) == 1 else []
    failures = _read_csv(failure_paths[0]) if len(failure_paths) == 1 else []
    return events, failures


def _load_environment(run_dir: Path) -> dict[str, Any]:
    paths = sorted(run_dir.glob("*_environment.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one environment JSON in {run_dir}, found {paths}")
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Stage 2.3 environment artifact must be a JSON object")
    return payload


def _revalidate_records(root: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        row = record["row"]
        if record.get("missing"):
            output.append(
                {
                    "report": _empty_report(),
                    "objective_consistent": False,
                    "objective": None,
                }
            )
            continue
        instance_path = root / "data" / "schneider" / f"{row['instance']}.txt"
        instance = parse_schneider(instance_path)
        raw_solution = record["solution"]
        routes = raw_solution.get("routes", [])
        if not isinstance(routes, list):
            raise TypeError(f"raw solution routes must be a list: {record['solution_path']}")
        route_lists = [list(route) for route in routes]
        solver_objective = _solver_objective(record["raw"].get("solver_result", {}))
        report = validate_routes(
            instance,
            route_lists,
            claimed_objective=(solver_objective.total_distance if solver_objective else None),
        )
        recomputed = SolutionObjective.from_report(instance, report) if report.feasible else None
        consistent = (
            solver_objective is not None
            and recomputed is not None
            and compare_objectives(solver_objective, recomputed) is ObjectiveComparison.EQUAL
        )
        output.append(
            {
                "report": report,
                "objective_consistent": consistent,
                "objective": recomputed,
            }
        )
    return output


def _solver_objective(payload: object) -> SolutionObjective | None:
    if not isinstance(payload, dict):
        return None
    objective = payload.get("objective")
    if not isinstance(objective, dict):
        return None
    try:
        return SolutionObjective(
            int(objective["vehicle_count"]),
            float(objective["total_distance"]),
            float(objective["total_charging_time"]),
            int(objective["charging_count"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _empty_report() -> SolutionReport:
    return SolutionReport(0, 0.0, 0.0, 0.0, 0.0, (), ("missing raw solution",))


def _artifact_findings(
    run_dir: Path,
    rows: list[dict[str, str]],
    records: list[dict[str, Any]],
    environment: dict[str, Any],
    operator_events: list[dict[str, str]],
    failure_events: list[dict[str, str]],
) -> list[dict[str, Any]]:
    raw_complete = bool(records) and all(not record.get("missing") for record in records)
    event_paths = sorted(run_dir.glob("*_operator_events.csv"))
    failure_paths = sorted(run_dir.glob("*_operator_failure_events.csv"))
    environment_ok = bool(environment)
    manifest_paths = _run_manifest_paths(run_dir)
    manifest_integrity = _run_manifest_integrity(run_dir)
    return [
        _finding(
            "raw_solution_artifacts_complete",
            raw_complete and len(rows) == len(records),
            f"{sum(not record.get('missing') for record in records)}/{len(records)}",
            "raw JSON and solution JSON for every run",
            "candidate raw/ and solutions/ directories",
        ),
        _finding(
            "operator_event_artifacts_complete",
            len(event_paths) == 1
            and len(failure_paths) == 1
            and bool(operator_events)
            and bool(failure_events),
            (
                f"event_csv={len(event_paths)}, failure_csv={len(failure_paths)}, "
                f"event_rows={len(operator_events)}, failure_rows={len(failure_events)}"
            ),
            "one operator event CSV and one failure event CSV",
            "candidate run directory",
        ),
        _finding(
            "environment_and_manifest_complete",
            environment_ok and len(manifest_paths) == 1 and manifest_integrity,
            (
                f"environment={environment_ok}, manifest={len(manifest_paths)}, "
                f"manifest_integrity={manifest_integrity}"
            ),
            "environment JSON and checksum-valid run manifest",
            "candidate run directory",
        ),
        _finding(
            "source_config_instance_hash_consistency",
            _hashes_consistent(
                _repository_root(), run_dir, rows, records, environment
            ),
            (
                "consistent"
                if _hashes_consistent(
                    _repository_root(), run_dir, rows, records, environment
                )
                else "inconsistent"
            ),
            "source/config/instance/environment hashes agree",
            "per-run rows, raw records, and environment JSON",
        ),
    ]


def _hashes_consistent(
    root: Path,
    run_dir: Path,
    rows: list[dict[str, str]],
    records: list[dict[str, Any]],
    environment: dict[str, Any],
) -> bool:
    source_values = {row.get("algorithm_source_sha256", "") for row in rows}
    instance_hashes: dict[str, set[str]] = {}
    for row in rows:
        instance_hashes.setdefault(str(row.get("instance")), set()).add(
            row.get("instance_sha256", "")
        )
    environment_source = str(environment.get("algorithm_source_sha256", ""))
    environment_sources = environment.get("algorithm_source_files", {})
    current_sources = _source_hashes(root)
    source_files_ok = environment_sources == current_sources
    source_ok = len(source_values) == 1 and bool(next(iter(source_values), ""))
    source_ok = source_ok and environment_source == next(iter(source_values), "")
    source_ok = source_ok and environment_source == _combined_hash(current_sources)
    instance_ok = bool(instance_hashes) and all(
        len(values) == 1 and bool(next(iter(values), ""))
        for values in instance_hashes.values()
    )
    instance_ok = instance_ok and all(
        _sha256(root / "data" / "schneider" / f"{instance}.txt")
        == next(iter(values))
        for instance, values in instance_hashes.items()
        if len(values) == 1 and next(iter(values), "")
    )
    parameter_paths = sorted(run_dir.glob("*_parameters.toml"))
    config_hash = str(environment.get("configuration_sha256", ""))
    config_ok = len(parameter_paths) == 1 and bool(config_hash)
    current_config = root / "configs" / "stage02_constraint_guided.toml"
    config_ok = config_ok and _sha256(parameter_paths[0]) == config_hash
    config_ok = config_ok and current_config.is_file()
    config_ok = config_ok and _sha256(current_config) == config_hash
    raw_row_ok = all(
        record.get("raw", {}).get("record", {}).get("algorithm_source_sha256")
        == record["row"].get("algorithm_source_sha256")
        for record in records
        if not record.get("missing")
    )
    return (
        source_ok
        and source_files_ok
        and instance_ok
        and config_ok
        and raw_row_ok
        and isinstance(environment_sources, dict)
    )


def _formal_run_configuration(
    root: Path,
    run_dir: Path,
    rows: list[dict[str, str]],
    environment: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    parameter_paths = sorted(run_dir.glob("*_parameters.toml"))
    expected = {
        "algorithm": "ALNS_STAGE02_CONSTRAINT_GUIDED",
        "operator_profile": "stage02_constraint_guided",
        "time_limit_seconds": 30.0,
        "max_iterations": 1000,
        "threads": 1,
        "objective_schema": "vehicles,distance,charging_time,charging_count",
    }
    if len(parameter_paths) != 1:
        return False, {"reason": "missing_or_duplicated_parameters_toml"}
    try:
        payload = tomllib.loads(parameter_paths[0].read_text(encoding="utf-8"))
        stage_payload = payload.get("stage02", payload)
        run_payload = payload["run"]
        observed = {
            "algorithm": stage_payload["algorithm"],
            "operator_profile": stage_payload["operator_profile"],
            "time_limit_seconds": float(run_payload["time_limit_seconds"]),
            "max_iterations": int(run_payload["max_iterations"]),
            "threads": int(run_payload["threads"]),
            "objective_schema": "vehicles,distance,charging_time,charging_count",
            "row_field_mismatches": [],
            "environment_mismatches": [],
        }
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        return False, {"reason": "parameters_toml_parse_failed", "error": str(error)}

    row_mismatches: list[dict[str, Any]] = []
    for row in rows:
        checks = {
            "algorithm": row.get("algorithm") == observed["algorithm"],
            "operator_profile": row.get("operator_profile") == observed["operator_profile"],
            "time_limit_seconds": _float_equal(
                row.get("time_limit_seconds"), observed["time_limit_seconds"]
            ),
            "max_iterations": _as_int(row.get("max_iterations"))
            == observed["max_iterations"],
            "threads": _as_int(row.get("threads")) == observed["threads"],
            "objective_schema": row.get("objective_schema") == observed["objective_schema"],
        }
        if not all(checks.values()):
            row_mismatches.append(
                {"instance": row.get("instance"), "seed": row.get("seed"), "checks": checks}
            )
    environment_mismatches = [
        field
        for field in ("algorithm", "operator_profile")
        if environment.get(field) != observed[field]
    ]
    observed["row_field_mismatches"] = row_mismatches
    observed["environment_mismatches"] = environment_mismatches
    passed = observed == {
        **expected,
        "row_field_mismatches": [],
        "environment_mismatches": [],
    }
    # Keep the current repository/config path in the audit evidence without
    # treating it as a substitute for the recorded parameter snapshot.
    observed["current_config_path"] = str(root / "configs" / "stage02_constraint_guided.toml")
    return passed, observed


def _time_budget_exemption(
    run_dir: Path,
    rows: list[dict[str, str]],
    operator_events: list[dict[str, str]],
) -> tuple[bool, dict[str, Any]]:
    parameter_paths = sorted(run_dir.glob("*_parameters.toml"))
    if len(parameter_paths) != 1:
        return False, {"reason": "missing_or_duplicated_parameters_toml"}
    try:
        payload = tomllib.loads(parameter_paths[0].read_text(encoding="utf-8"))
        run_payload = payload["run"]
        time_limit = float(run_payload["time_limit_seconds"])
    except (OSError, KeyError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        return False, {"reason": "parameters_toml_parse_failed", "error": str(error)}

    event_keys_with_timeout = {
        (str(event.get("instance")), str(event.get("seed")))
        for event in operator_events
        if event.get("status") == "time_limit"
    }
    overruns: list[dict[str, Any]] = []
    for row in rows:
        runtime = _as_float(row.get("runtime_seconds"))
        if runtime <= time_limit + 1e-9:
            continue
        key = (str(row.get("instance")), str(row.get("seed")))
        overruns.append(
            {
                "instance": key[0],
                "seed": key[1],
                "runtime_seconds": runtime,
                "overrun_seconds": runtime - time_limit,
                "iterations": _as_int(row.get("iterations")),
                "exact_charging_calls": _as_int(row.get("charging_subproblem_calls")),
                "timeout_event_recorded": key in event_keys_with_timeout,
            }
        )
    max_overrun = max(
        (float(item["overrun_seconds"]) for item in overruns),
        default=0.0,
    )
    return all(bool(item["timeout_event_recorded"]) for item in overruns), {
        "declared_time_limit_seconds": time_limit,
        "overrun_run_count": len(overruns),
        "max_overrun_seconds": max_overrun,
        "overrun_runs": overruns,
        "exemption_reason": (
            "cooperative exact charging calls are not interruptible; each overrun is "
            "retained with completed iterations, exact calls, timeout event, and provenance"
            if overruns
            else "no run exceeded the declared time limit"
        ),
    }


def _float_equal(value: object, expected: float) -> bool:
    try:
        return abs(float(str(value)) - expected) <= 1e-9
    except (TypeError, ValueError):
        return False


def _as_float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return float("nan")


def _review_provenance(
    root: Path,
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Capture dependency and native-runtime hashes during independent review."""

    files: dict[str, str] = {}
    for relative_path in ("pyproject.toml", "uv.lock"):
        path = root / relative_path
        if path.is_file():
            files[relative_path] = _sha256(path)

    captured = environment.get("captured_environment", {})
    if not isinstance(captured, dict):
        captured = {}
    native_value = captured.get("native_extension", "")
    native_path = Path(str(native_value))
    if native_path.is_file():
        files["native_extension"] = _sha256(native_path)
    packages = captured.get("packages", {})
    python_metadata = captured.get("python", {})
    review_cli = root / "src" / "evrptw" / "experiments" / "stage02_constraint_guided_review.py"
    if review_cli.is_file():
        files["review_cli"] = _sha256(review_cli)
    complete = (
        all(
            bool(files.get(name))
            for name in ("pyproject.toml", "uv.lock", "native_extension", "review_cli")
        )
        and isinstance(packages, dict)
        and bool(packages)
        and isinstance(python_metadata, dict)
        and bool(python_metadata.get("version"))
        and bool(python_metadata.get("executable"))
    )
    return {
        "complete": complete,
        "files": files,
        "package_versions": packages,
        "python": python_metadata,
    }


def _stage00_manifest_unchanged(root: Path, environment: dict[str, Any]) -> bool:
    manifest = root / "experiments" / "baselines" / "stage00" / "manifest.json"
    if not manifest.is_file() or _sha256(manifest) != environment.get(
        "baseline_manifest_sha256"
    ):
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        expected_files = payload["files"]
        if not isinstance(expected_files, dict):
            return False
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False
    baseline_dir = manifest.parent
    actual_files = {
        str(path.relative_to(baseline_dir))
        for path in baseline_dir.rglob("*")
        if path.is_file() and path != manifest
    }
    return actual_files == set(expected_files) and all(
        _sha256(baseline_dir / relative_path) == expected_hash
        for relative_path, expected_hash in expected_files.items()
    )


def _failure_analysis_rows(
    rows: list[dict[str, str]],
    records: list[dict[str, Any]],
    operator_events: list[dict[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    records_by_key = {
        (str(record["row"].get("instance")), str(record["row"].get("seed"))): record
        for record in records
        if not record.get("missing")
    }
    failure_statuses = {
        "failed",
        "prefilter_rejected",
        "exact_infeasible",
        "budget_exhausted",
        "time_limit",
    }
    for event in operator_events:
        status = str(event.get("status", ""))
        reason = str(event.get("reason", ""))
        if status not in failure_statuses:
            continue
        record = records_by_key.get((str(event.get("instance")), str(event.get("seed"))))
        if record is None:
            continue
        row = record["row"]
        objective = _solver_objective(record["raw"].get("solver_result", {}))
        category, root_cause, next_step = _diagnose_event(status, reason)
        key = (
            str(row["instance"]),
            str(row["seed"]),
            str(event.get("operator")),
            category,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "instance": row["instance"],
                "seed": row["seed"],
                "operator": event.get("operator", ""),
                "objective": _render_objective(objective),
                "feasible": row.get("feasible", "False"),
                "first_failure_location": (
                    f"iteration={event.get('iteration', '')};status={status};reason={reason}"
                ),
                "constraint_trigger": category,
                "root_cause_classification": root_cause,
                "next_step": next_step,
            }
        )
        if len(output) >= 20:
            return output
    return output


def _diagnose_event(status: str, reason: str) -> tuple[str, str, str]:
    lower = f"{status} {reason}".lower()
    if "capacity" in lower or "load" in lower:
        return "capacity", "operator_logic", "review insertion ordering and capacity prefilter"
    if "time" in lower or "window" in lower or "due" in lower:
        return (
            "time_window",
            "operator_logic",
            "review exact arrival replay and time-window ranking",
        )
    if "energy" in lower or "battery" in lower or "charging" in lower:
        return (
            "energy",
            "operator_logic",
            "review optimistic energy screen and exact charging candidate",
        )
    if "budget" in lower or status == "time_limit":
        return (
            "computational_limit",
            "computational_limit",
            "record budget and assess bounded candidate generation",
        )
    if status == "exact_infeasible":
        return (
            "exact_infeasible",
            "operator_logic",
            "compare prefilter assumptions with exact charging failure",
        )
    return (
        "operator_failure",
        "operator_logic",
        "inspect the operator event trace and add a focused regression test",
    )


def _readiness_rows(
    candidate_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, str]],
    comparison_metric_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    comparison_by_key = {
        (str(row.get("instance")), int(row.get("seed", "-1"))): row
        for row in comparison_rows
    }
    comparison_metrics_by_key = {
        (str(row.get("instance")), int(row.get("seed", "-1"))): row
        for row in comparison_metric_rows
    }
    output: list[dict[str, Any]] = []
    for row in candidate_rows:
        key = (str(row.get("instance")), int(row.get("seed", "-1")))
        baseline = comparison_by_key.get(key)
        metric_baseline = comparison_metrics_by_key.get(key)
        merged_baseline = dict(baseline or {})
        merged_baseline.update(metric_baseline or {})
        candidate_objective = _objective_from_row(row)
        baseline_objective = _objective_from_row(baseline) if baseline is not None else None
        comparison = (
            compare_objectives(candidate_objective, baseline_objective).value
            if candidate_objective is not None and baseline_objective is not None
            else "missing"
        )
        output.append(
            {
                "instance": row.get("instance", ""),
                "seed": row.get("seed", ""),
                "stage02_2_objective": _render_objective(baseline_objective),
                "stage02_3_objective": _render_objective(candidate_objective),
                "stage02_2_feasible": baseline.get("feasible", "") if baseline else "",
                "stage02_3_feasible": row.get("feasible", ""),
                "stage02_2_iterations": baseline.get("iterations", "") if baseline else "",
                "stage02_3_iterations": row.get("iterations", ""),
                "stage02_2_effective_iterations": _baseline_metric(
                    merged_baseline, "effective_iterations"
                ),
                "stage02_3_effective_iterations": row.get("effective_iterations", ""),
                "stage02_2_exact_charging_calls": (
                    baseline.get("charging_subproblem_calls", "") if baseline else ""
                ),
                "stage02_3_exact_charging_calls": row.get("charging_subproblem_calls", ""),
                "stage02_2_cache_hits": _baseline_metric(merged_baseline, "cache_hits"),
                "stage02_2_cache_misses": _baseline_metric(
                    merged_baseline, "cache_misses"
                ),
                "stage02_2_unique_route_evaluations": _baseline_metric(
                    merged_baseline, "unique_route_evaluations"
                ),
                "stage02_3_cache_hits": row.get("cache_hits", ""),
                "stage02_3_cache_misses": row.get("cache_misses", ""),
                "stage02_3_unique_route_evaluations": row.get(
                    "unique_route_evaluations", ""
                ),
                "stage02_2_runtime_seconds": baseline.get("runtime_seconds", "")
                if baseline
                else "",
                "stage02_3_runtime_seconds": row.get("runtime_seconds", ""),
                "stage02_2_removal_tier_counts": _baseline_metric(
                    merged_baseline, "removal_tier_counts"
                ),
                "stage02_3_removal_tier_counts": row.get("removal_tier_counts", ""),
                "stage02_2_failure_events": baseline.get("operator_failure_events", "")
                if baseline
                else "",
                "stage02_3_failure_events": row.get("operator_failure_events", ""),
                "stage02_2_accepted_moves": baseline.get("accepted_moves", "")
                if baseline
                else "",
                "stage02_3_accepted_moves": row.get("accepted_moves", ""),
                "stage02_2_constraint_statistics": _baseline_metric(
                    merged_baseline, "constraint_operator_statistics_json"
                ),
                "stage02_3_constraint_statistics": row.get(
                    "constraint_operator_statistics_json", ""
                ),
                "objective_comparison": comparison,
                "readiness_status": _readiness_status(
                    baseline, row, candidate_objective, comparison
                ),
            }
        )
    return output


def _baseline_metric(baseline: dict[str, str] | None, field: str) -> str:
    if baseline is None:
        return "missing_baseline"
    return baseline.get(field, "not_recorded_in_stage02_2_summary")


def _readiness_status(
    baseline: dict[str, str] | None,
    candidate: dict[str, str],
    candidate_objective: SolutionObjective | None,
    comparison: str,
) -> str:
    if baseline is None or candidate_objective is None:
        return "missing"
    if baseline.get("feasible") not in {"True", "true", True}:
        return "baseline_infeasible"
    if candidate.get("feasible") not in {"True", "true", True}:
        return "candidate_infeasible"
    if comparison == ObjectiveComparison.WORSE.value:
        return "objective_regression"
    return "ready"


def _load_comparison_rows(comparison_dir: Path) -> list[dict[str, str]]:
    root = _repository_root()
    candidates: list[Path] = []
    if comparison_dir.is_file():
        candidates.append(comparison_dir)
    elif comparison_dir.is_dir():
        candidates.extend(sorted(comparison_dir.glob("*_per_run_results.csv")))
    if not candidates:
        candidates.extend(
            sorted(
                (root / "experiments" / "summaries").glob(
                    "stage02_quality_attempt02_per_run_results.csv"
                )
            )
        )
    if len(candidates) != 1:
        raise RuntimeError(
            "could not resolve a unique Stage 2.2 comparison per-run CSV: "
            f"{candidates}"
        )
    rows = _read_csv(candidates[0])
    expected = {(instance, seed) for instance in FORMAL_INSTANCES for seed in FORMAL_SEEDS}
    actual = {(str(row.get("instance")), int(row.get("seed", "-1"))) for row in rows}
    if actual != expected or len(actual) != len(rows):
        raise RuntimeError("Stage 2.2 comparison coverage is incomplete or duplicated")
    return rows


def _load_comparison_metric_rows(
    root: Path,
    comparison_dir: Path,
) -> list[dict[str, str]]:
    """Load recorded Stage 2.2 instrumentation without changing its objective baseline."""

    required = set(STAGE02_2_METRIC_FIELDS)
    candidates: list[Path] = []
    if comparison_dir.is_file():
        candidates.append(comparison_dir)
    elif comparison_dir.is_dir():
        candidates.extend(sorted(comparison_dir.glob("*_per_run_results.csv")))
    metric_candidates: list[Path] = []
    for path in candidates:
        rows = _read_csv(path)
        if rows and required <= set(rows[0]):
            metric_candidates.append(path)
    candidates = metric_candidates
    supplement = root / "experiments" / "summaries" / (
        "stage02_quality_attempt02_readiness_metrics.csv"
    )
    if supplement.is_file() and supplement not in candidates:
        candidates.append(supplement)
    if len(candidates) != 1:
        return []
    rows = _read_csv(candidates[0])
    expected = {(instance, seed) for instance in FORMAL_INSTANCES for seed in FORMAL_SEEDS}
    actual = {(str(row.get("instance")), int(row.get("seed", "-1"))) for row in rows}
    if actual != expected or len(actual) != len(rows):
        return []
    return rows


def _comparison_metric_integrity(
    root: Path,
    comparison_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
) -> tuple[bool, dict[str, object]]:
    """Verify provenance of the tracked Stage 2.2 instrumentation supplement."""

    expected_keys = {
        (instance, seed) for instance in FORMAL_INSTANCES for seed in FORMAL_SEEDS
    }

    def row_key(row: dict[str, str]) -> tuple[str, int] | None:
        try:
            return str(row["instance"]), int(row["seed"])
        except (KeyError, TypeError, ValueError):
            return None

    metric_keys = [row_key(row) for row in metric_rows]
    comparison_keys = [row_key(row) for row in comparison_rows]
    metric_key_set = {key for key in metric_keys if key is not None}
    comparison_key_set = {key for key in comparison_keys if key is not None}
    coverage_ok = (
        len(metric_rows) == len(expected_keys)
        and len(metric_key_set) == len(metric_rows)
        and metric_key_set == expected_keys
        and comparison_key_set == expected_keys
        and len(comparison_keys) == len(comparison_key_set)
    )

    protocol_expectations = {
        "algorithm": "ALNS_STAGE02_ROUTE_QUALITY",
        "operator_profile": "stage02_route_quality",
        "run_label": "stage02_quality_instrumented_attempt01",
        "time_limit_seconds": "30.0",
        "max_iterations": "1000",
        "threads": "1",
        "objective_schema": "vehicles,distance,charging_time,charging_count",
    }
    protocol_observed: dict[str, list[str]] = {}
    protocol_ok = True
    for field, expected in protocol_expectations.items():
        values = sorted({str(row.get(field, "")) for row in metric_rows})
        protocol_observed[field] = values
        protocol_ok = protocol_ok and values == [expected]

    source_values = {str(row.get("algorithm_source_sha256", "")) for row in metric_rows}
    revision_values = {str(row.get("repository_revision", "")) for row in metric_rows}
    source_ok = len(source_values) == 1 and bool(next(iter(source_values), ""))
    revision_ok = len(revision_values) == 1 and bool(next(iter(revision_values), ""))

    instance_hash_ok = True
    for row in metric_rows:
        instance = str(row.get("instance", ""))
        recorded_hash = str(row.get("instance_sha256", ""))
        instance_path = root / "data" / "schneider" / f"{instance}.txt"
        if not instance_path.is_file() or not recorded_hash:
            instance_hash_ok = False
            continue
        instance_hash_ok = instance_hash_ok and _sha256(instance_path) == recorded_hash

    metric_by_key = {
        key: row
        for row in metric_rows
        if (key := row_key(row)) is not None
    }
    comparison_by_key = {
        key: row
        for row in comparison_rows
        if (key := row_key(row)) is not None
    }
    objective_matches = True
    for key in expected_keys:
        metric_objective = _objective_from_row(metric_by_key.get(key))
        comparison_objective = _objective_from_row(comparison_by_key.get(key))
        objective_matches = objective_matches and (
            metric_objective is not None
            and comparison_objective is not None
            and compare_objectives(metric_objective, comparison_objective)
            == ObjectiveComparison.EQUAL
        )

    recorded_metrics_ok = all(
        all(str(row.get(field, "")) not in {"", "not_recorded_in_stage02_2_summary"}
            for field in STAGE02_2_METRIC_FIELDS)
        and str(row.get("status", "")) == "feasible"
        and str(row.get("feasible", "")).lower() == "true"
        and _valid_json_object(row.get("constraint_operator_statistics_json", ""))
        for row in metric_rows
    )
    passed = (
        coverage_ok
        and protocol_ok
        and source_ok
        and revision_ok
        and instance_hash_ok
        and objective_matches
        and recorded_metrics_ok
    )
    return passed, {
        "coverage_ok": coverage_ok,
        "protocol_ok": protocol_ok,
        "protocol": protocol_observed,
        "source_hashes": sorted(source_values),
        "repository_revisions": sorted(revision_values),
        "instance_hashes_ok": instance_hash_ok,
        "objectives_match_fixed_baseline": objective_matches,
        "recorded_metrics_and_feasibility_ok": recorded_metrics_ok,
    }


def _valid_json_object(value: object) -> bool:
    try:
        return isinstance(json.loads(str(value)), dict)
    except (TypeError, json.JSONDecodeError):
        return False


def _objective_from_row(row: dict[str, str] | None) -> SolutionObjective | None:
    if row is None or row.get("feasible") not in {"True", "true", True}:
        return None
    try:
        return SolutionObjective(
            int(row["primary_vehicle_count"]),
            float(row["secondary_total_distance"]),
            float(row["tertiary_total_charging_time"]),
            int(row["quaternary_charging_count"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _gate_report_present(run_dir: Path) -> bool:
    paths = sorted(run_dir.glob("*_gate_report.csv"))
    return len(paths) == 1 and bool(_read_csv(paths[0]))


def _run_manifest_paths(run_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_dir.glob("*_manifest.json"))
        if path.name != "review_manifest.json"
    ]


def _run_manifest_integrity(run_dir: Path) -> bool:
    paths = _run_manifest_paths(run_dir)
    if len(paths) != 1:
        return False
    try:
        payload = json.loads(paths[0].read_text(encoding="utf-8"))
        expected_files = payload["files"]
        if not isinstance(expected_files, dict):
            return False
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    review_artifacts = {
        "review_report.md",
        "review_findings.csv",
        "failure_analysis.csv",
        "stage03_readiness.csv",
        "review_manifest.json",
    }
    actual_files = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
        and path != paths[0]
        and path.name not in review_artifacts
    }
    if actual_files != set(expected_files):
        return False
    return all(
        _sha256(run_dir / relative_path) == expected_hash
        for relative_path, expected_hash in expected_files.items()
    )


def _replay_independent_rerun(
    root: Path,
    run_dir: Path,
    current_rows: list[dict[str, str]],
    comparison_dir: Path,
) -> tuple[bool, dict[str, object]]:
    repeatability_paths = sorted(run_dir.glob("*_repeatability.csv"))
    if len(repeatability_paths) != 1:
        return False, {"reason": "missing_or_duplicated_repeatability_csv"}
    repeatability_rows = _read_csv(repeatability_paths[0])
    if len(repeatability_rows) != 1:
        return False, {"reason": "repeatability_csv_must_have_one_row"}
    repeatability = repeatability_rows[0]
    if repeatability.get("status") == "pending":
        return True, {
            "status": "first_complete_run_pending_independent_rerun",
            "current_run_keys": len(current_rows),
        }
    if repeatability.get("status") != "pass":
        return False, {
            "status": repeatability.get("status", "missing"),
            "reason": "repeatability_status_not_pass",
        }
    first_run_value = str(repeatability.get("first_run", ""))
    first_run = Path(first_run_value)
    if not first_run.is_absolute():
        first_run = root / first_run
    if not first_run.is_dir():
        return False, {"reason": "first_run_directory_missing", "first_run": str(first_run)}
    try:
        first_rows = _load_single_per_run(first_run)
        first_records = _load_candidate_records(first_run, first_rows)
        first_results = _revalidate_records(root, first_records)
        first_operator_events, first_failure_events = _load_event_logs(first_run)
        first_environment = _load_environment(first_run)
        comparison_rows = _load_comparison_rows(comparison_dir)
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return False, {"reason": "first_run_revalidation_failed", "error": str(error)}
    first_keys = {
        (str(row.get("instance")), int(row.get("seed", "-1"))) for row in first_rows
    }
    expected_keys = {(instance, seed) for instance in FORMAL_INSTANCES for seed in FORMAL_SEEDS}
    first_raw_ok = (
        len(first_rows) == len(expected_keys)
        and len(first_keys) == len(first_rows)
        and first_keys == expected_keys
        and len(first_results) == len(expected_keys)
        and all(item["report"].feasible and item["objective_consistent"] for item in first_results)
    )
    current_gate_paths = sorted(run_dir.glob("*_gate_report.csv"))
    first_gate_paths = sorted(first_run.glob("*_gate_report.csv"))
    first_gate_rows = _read_csv(first_gate_paths[0]) if len(first_gate_paths) == 1 else []
    current_gate_rows = (
        _read_csv(current_gate_paths[0]) if len(current_gate_paths) == 1 else []
    )
    first_gate_ok = bool(first_gate_rows) and all(
        row.get("status") == "pass" for row in first_gate_rows
    )
    current_gate_ok = bool(current_gate_rows) and all(
        row.get("status") == "pass" for row in current_gate_rows
    )
    first_event_replay_ok, first_event_observed = _replay_constraint_events(
        root, first_run, first_operator_events
    )
    first_provenance = _review_provenance(root, first_environment)
    first_configuration_ok, first_configuration_observed = _formal_run_configuration(
        root, first_run, first_rows, first_environment
    )
    first_time_budget_ok, first_time_budget_observed = _time_budget_exemption(
        first_run, first_rows, first_operator_events
    )
    first_hard_ok, first_hard_observed = _replay_runner_hard_gates(
        root,
        first_run,
        first_rows,
        first_results,
        comparison_rows,
        first_operator_events,
        first_event_replay_ok,
        True,
    )
    current_records = _load_candidate_records(run_dir, current_rows)
    current_results = _revalidate_records(root, current_records)
    current_operator_events, current_failure_events = _load_event_logs(run_dir)
    current_event_replay_ok, _ = _replay_constraint_events(
        root, run_dir, current_operator_events
    )
    current_hard_ok, _ = _replay_runner_hard_gates(
        root,
        run_dir,
        current_rows,
        current_results,
        comparison_rows,
        current_operator_events,
        current_event_replay_ok,
        True,
    )
    first_artifacts_ok = (
        len(sorted(first_run.glob("*_operator_events.csv"))) == 1
        and len(sorted(first_run.glob("*_operator_failure_events.csv"))) == 1
        and bool(first_operator_events)
        and bool(first_failure_events)
        and len(_failure_analysis_rows(first_rows, first_records, first_operator_events)) >= 3
    )
    current_artifacts_ok = (
        len(sorted(run_dir.glob("*_operator_events.csv"))) == 1
        and len(sorted(run_dir.glob("*_operator_failure_events.csv"))) == 1
        and bool(current_operator_events)
        and bool(current_failure_events)
    )
    first_hash_ok = _hashes_consistent(
        root, first_run, first_rows, first_records, first_environment
    )
    current_environment = _load_environment(run_dir)
    current_hash_ok = _hashes_consistent(
        root, run_dir, current_rows, current_records, current_environment
    )
    first_manifest_ok = _run_manifest_integrity(first_run)
    current_manifest_ok = _run_manifest_integrity(run_dir)
    configuration_ok = (
        _as_bool(repeatability.get("configuration_match"))
        and repeatability.get("first_run_keys") == "36/36"
        and repeatability.get("second_run_keys") == "36/36"
        and repeatability.get("first_gate_status") == "pass"
        and repeatability.get("second_gate_status") == "pass"
    )
    passed = (
        first_raw_ok
        and first_gate_ok
        and current_gate_ok
        and configuration_ok
        and first_event_replay_ok
        and first_provenance["complete"]
        and first_configuration_ok
        and first_time_budget_ok
        and first_hard_ok
        and current_event_replay_ok
        and current_hard_ok
        and first_artifacts_ok
        and current_artifacts_ok
        and first_hash_ok
        and current_hash_ok
        and first_manifest_ok
        and current_manifest_ok
    )
    return passed, {
        "status": repeatability.get("status"),
        "first_raw_revalidated": first_raw_ok,
        "first_gate_pass": first_gate_ok,
        "second_gate_pass": current_gate_ok,
        "configuration_match": configuration_ok,
        "first_event_replay": first_event_replay_ok,
        "first_event_observed": first_event_observed,
        "first_provenance_complete": first_provenance["complete"],
        "first_provenance": first_provenance,
        "first_formal_configuration": first_configuration_ok,
        "first_formal_configuration_observed": first_configuration_observed,
        "first_time_budget_exemption": first_time_budget_ok,
        "first_time_budget_observed": first_time_budget_observed,
        "first_hard_replay": first_hard_ok,
        "first_hard_observed": first_hard_observed,
        "second_event_replay": current_event_replay_ok,
        "second_hard_replay": current_hard_ok,
        "first_artifacts_complete": first_artifacts_ok,
        "second_artifacts_complete": current_artifacts_ok,
        "first_hashes_consistent": first_hash_ok,
        "second_hashes_consistent": current_hash_ok,
        "first_manifest_integrity": first_manifest_ok,
        "second_manifest_integrity": current_manifest_ok,
    }


def _replay_runner_hard_gates(
    root: Path,
    run_dir: Path,
    rows: list[dict[str, str]],
    validator_results: list[dict[str, Any]],
    comparison_rows: list[dict[str, str]],
    operator_events: list[dict[str, str]],
    constraint_replay_ok: bool,
    independent_replay_ok: bool,
) -> tuple[bool, dict[str, object]]:
    """Recompute runner hard gates from raw evidence instead of gate CSV status."""

    expected_keys = {(instance, seed) for instance in FORMAL_INSTANCES for seed in FORMAL_SEEDS}
    actual_keys = [(str(row.get("instance")), int(row.get("seed", "-1"))) for row in rows]
    coverage_ok = len(actual_keys) == len(set(actual_keys)) and set(actual_keys) == expected_keys
    validator_ok = len(validator_results) == len(expected_keys) and all(
        item["report"].feasible for item in validator_results
    )
    objective_ok = validator_ok and all(
        item["objective_consistent"] for item in validator_results
    )
    candidate_by_key = {
        (str(row.get("instance")), int(row.get("seed", "-1"))): item["objective"]
        for row, item in zip(rows, validator_results, strict=True)
        if item["report"].feasible and item["objective"] is not None
    }
    comparison_by_key = {
        (str(row.get("instance")), int(row.get("seed", "-1"))): _objective_from_row(row)
        for row in comparison_rows
    }
    objective_comparison_ok = objective_ok and all(
        candidate_by_key.get((instance, seed)) is not None
        and comparison_by_key.get((instance, seed)) is not None
        for instance, seed in expected_keys
    )
    objective_observed: dict[str, str] = {}
    for instance in FORMAL_INSTANCES:
        candidate_values: list[SolutionObjective] = [
            cast(SolutionObjective, candidate_by_key[(instance, seed)])
            for seed in FORMAL_SEEDS
            if (instance, seed) in candidate_by_key
            and candidate_by_key[(instance, seed)] is not None
        ]
        baseline_values: list[SolutionObjective] = [
            cast(SolutionObjective, comparison_by_key[(instance, seed)])
            for seed in FORMAL_SEEDS
            if (instance, seed) in comparison_by_key
            and comparison_by_key[(instance, seed)] is not None
        ]
        if len(candidate_values) != len(FORMAL_SEEDS) or len(baseline_values) != len(FORMAL_SEEDS):
            objective_comparison_ok = False
            objective_observed[instance] = "missing"
            continue
        candidate_best = min(candidate_values, key=lambda value: value.key)
        baseline_best = min(baseline_values, key=lambda value: value.key)
        comparison = compare_objectives(candidate_best, baseline_best)
        objective_observed[instance] = comparison.value
        objective_comparison_ok = objective_comparison_ok and (
            comparison is not ObjectiveComparison.WORSE
        )

    focused_ok = True
    focused_observed: dict[str, object] = {}
    try:
        config = load_config(root / "configs" / "stage02_constraint_guided.toml")
        stage00 = _load_stage00_baseline(
            root / config.baseline_dir,
            root / config.benchmark_dir,
            config,
        )
        stage1 = _load_stage1_baseline(root / config.stage01_per_run, config)
        for instance in FOCUSED_R_RC_INSTANCES:
            candidate_values = [candidate_by_key[(instance, seed)] for seed in FORMAL_SEEDS]
            stage00_values = [stage00[(instance, seed)] for seed in FORMAL_SEEDS]
            stage1_values = [stage1[(instance, seed)] for seed in FORMAL_SEEDS]
            candidate_vehicles = [float(value.vehicle_count) for value in candidate_values]
            stage00_vehicles = [float(value.vehicle_count) for value in stage00_values]
            candidate_distances = [float(value.total_distance) for value in candidate_values]
            stage1_distances = [float(value.total_distance) for value in stage1_values]
            mean_vehicle_ok = statistics.fmean(candidate_vehicles) <= statistics.fmean(
                stage00_vehicles
            ) - 1.0
            distance_ok = statistics.fmean(candidate_distances) <= statistics.fmean(
                stage1_distances
            ) * 1.10
            stability_ok = statistics.pstdev(candidate_vehicles) <= statistics.pstdev(
                stage00_vehicles
            ) + 1e-9
            focused_ok = focused_ok and mean_vehicle_ok and distance_ok and stability_ok
            focused_observed[instance] = {
                "mean_vehicle_ok": mean_vehicle_ok,
                "distance_guard_ok": distance_ok,
                "vehicle_stability_ok": stability_ok,
            }
    except (FileNotFoundError, KeyError, TypeError, ValueError, ZeroDivisionError):
        focused_ok = False
        focused_observed = {
            instance: "baseline_replay_failed" for instance in FOCUSED_R_RC_INSTANCES
        }

    elimination_instances = {
        str(event.get("instance"))
        for event in operator_events
        if event.get("operator") == "route_elimination"
        and event.get("status") == "candidate_proposed"
        and event.get("reason") == "route_eliminated"
        and _as_bool(event.get("candidate_feasible"))
        and _as_int(event.get("candidate_vehicle_delta")) == -1
    }
    route_elimination_ok = len(elimination_instances) >= 2
    route_merge_ok = any(
        event.get("operator") == "route_merge"
        and event.get("status") == "candidate_proposed"
        and event.get("reason") == "route_merged"
        and _as_bool(event.get("candidate_feasible"))
        and _as_int(event.get("candidate_vehicle_delta")) < 0
        for event in operator_events
    )
    quality_operators = (
        "relocate",
        "swap",
        "two_opt_star",
        "route_segment_destroy",
        "ejection_chain",
    )
    quality_coverage: dict[str, bool] = {}
    for operator in quality_operators:
        events = [event for event in operator_events if event.get("operator") == operator]
        quality_coverage[operator] = bool(events) and any(
            _as_bool(event.get("candidate_feasible"))
            and str(event.get("status")) in {"feasible_candidate", "candidate_proposed"}
            and _event_has_affected_routes(event)
            for event in events
        )
    quality_improvement_ok = any(
        event.get("operator") in quality_operators
        and _as_bool(event.get("candidate_feasible"))
        and _as_bool(event.get("accepted"))
        and _as_bool(event.get("distance_improvement"))
        and _event_has_affected_routes(event)
        and not _as_bool(event.get("vehicle_reduction"))
        for event in operator_events
    )
    stage00_ok = _stage00_manifest_unchanged(
        root,
        _load_environment(run_dir),
    )
    passed = (
        coverage_ok
        and validator_ok
        and objective_ok
        and objective_comparison_ok
        and focused_ok
        and route_elimination_ok
        and route_merge_ok
        and all(quality_coverage.values())
        and quality_improvement_ok
        and constraint_replay_ok
        and stage00_ok
        and independent_replay_ok
    )
    return passed, {
        "coverage_ok": coverage_ok,
        "validator_ok": validator_ok,
        "objective_recomputation_ok": objective_ok,
        "objective_comparison_ok": objective_comparison_ok,
        "objective_observed": objective_observed,
        "focused_ok": focused_ok,
        "focused_observed": focused_observed,
        "route_elimination_ok": route_elimination_ok,
        "route_merge_ok": route_merge_ok,
        "quality_coverage": quality_coverage,
        "quality_same_vehicle_distance_improvement": quality_improvement_ok,
        "constraint_event_replay_ok": constraint_replay_ok,
        "stage00_ok": stage00_ok,
        "independent_complete_rerun_ok": independent_replay_ok,
    }


def _replay_constraint_events(
    root: Path,
    run_dir: Path,
    events: list[dict[str, str]],
) -> tuple[bool, dict[str, object]]:
    constraint_events = [
        event
        for event in events
        if event.get("track") == "constraint_lane"
        and event.get("operator")
        in {"station_pressure", "time_window_conflict", "worst_energy_detour", "shaw_related"}
    ]
    operators = {
        str(event.get("operator"))
        for event in constraint_events
        if event.get("operator")
    }
    operator_status = {
        operator: {
            "called": any(event.get("operator") == operator for event in constraint_events),
            "feasible": any(
                event.get("operator") == operator
                and _as_bool(event.get("candidate_feasible"))
                for event in constraint_events
            ),
            "accepted": any(
                event.get("operator") == operator
                and _accepted_feasible_constraint_candidate(event)
                for event in constraint_events
            ),
        }
        for operator in (
            "station_pressure",
            "time_window_conflict",
            "worst_energy_detour",
            "shaw_related",
        )
    }
    accepted_vehicle_increase = any(
        _as_bool(event.get("accepted"))
        and (_as_int(event.get("candidate_vehicle_delta")) or 0) > 0
        for event in constraint_events
    )
    actual_events = [
        event for event in constraint_events if _as_int(event.get("removal_size_actual")) > 0
    ]
    observed_tiers = {
        str(event.get("removal_tier")) for event in actual_events
    }
    focused_actual_max = max(
        (
            _as_int(event.get("removal_size_actual"))
            for event in actual_events
            if event.get("instance") in {"c101_21", "r101_21", "rc101_21"}
        ),
        default=0,
    )
    escalation_count = sum(
        "stagnation" in str(event.get("removal_trigger", ""))
        and str(event.get("removal_tier")) in {"medium", "large"}
        for event in constraint_events
    )
    reset_count = sum(_as_bool(event.get("reset_observed")) for event in constraint_events)
    counts_ok, invalid_count = _dynamic_event_counts_ok(root, run_dir, constraint_events)
    passed = (
        operators
        == {"station_pressure", "time_window_conflict", "worst_energy_detour", "shaw_related"}
        and all(all(status.values()) for status in operator_status.values())
        and not accepted_vehicle_increase
        and {"small", "medium", "large"} <= observed_tiers
        and focused_actual_max > 3
        and escalation_count > 0
        and reset_count > 0
        and counts_ok
    )
    return passed, {
        "operators": operator_status,
        "accepted_vehicle_increase": accepted_vehicle_increase,
        "observed_tiers": sorted(observed_tiers),
        "focused_actual_max": focused_actual_max,
        "stagnation_escalations": escalation_count,
        "global_best_resets": reset_count,
        "invalid_dynamic_counts": invalid_count,
    }


def _dynamic_event_counts_ok(
    root: Path,
    run_dir: Path,
    events: list[dict[str, str]],
) -> tuple[bool, int]:
    parameter_paths = sorted(run_dir.glob("*_parameters.toml"))
    if len(parameter_paths) != 1:
        return False, len(events)
    try:
        payload = tomllib.loads(parameter_paths[0].read_text(encoding="utf-8"))
        operators = payload["vehicle_operators"]
        if not isinstance(operators, dict):
            return False, len(events)
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        return False, len(events)

    bounds = {
        "small": (
            float(operators["small_removal_min_fraction"]),
            float(operators["small_removal_max_fraction"]),
        ),
        "medium": (
            float(operators["medium_removal_min_fraction"]),
            float(operators["medium_removal_max_fraction"]),
        ),
        "large": (
            float(operators["large_removal_min_fraction"]),
            float(operators["large_removal_max_fraction"]),
        ),
    }
    invalid = 0
    customer_counts: dict[str, int] = {}
    for event in events:
        requested = _as_int(event.get("removal_size_requested"))
        actual = _as_int(event.get("removal_size_actual"))
        if (
            requested == 0
            and actual == 0
            and not str(event.get("removal_tier", ""))
            and str(event.get("status")) == "time_limit"
            and str(event.get("removal_trigger", "")).startswith(
                "probe_not_started:"
            )
        ):
            continue
        instance_name = str(event.get("instance"))
        if instance_name not in customer_counts:
            customer_path = root / "data" / "schneider" / f"{instance_name}.txt"
            try:
                customer_counts[instance_name] = len(parse_schneider(customer_path).customers)
            except (OSError, ValueError):
                customer_counts[instance_name] = 0
        customer_count = customer_counts[instance_name]
        tier = str(event.get("removal_tier"))
        if tier not in bounds or customer_count <= 1:
            invalid += 1
            continue
        minimum, maximum = bounds[tier]
        lower = max(1, min(customer_count - 1, math.ceil(customer_count * minimum)))
        upper = max(
            lower,
            min(customer_count - 1, math.floor(customer_count * maximum)),
        )
        not_started = (
            actual == 0
            and event.get("status") == "time_limit"
            and str(event.get("removal_trigger", "")).startswith("probe_not_started:")
        )
        if not (lower <= requested <= upper) or (
            not not_started and not (lower <= actual <= upper)
        ):
            invalid += 1
    return invalid == 0, invalid


def _as_bool(value: object) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def _event_has_affected_routes(event: dict[str, str]) -> bool:
    value = str(event.get("affected_route_indices", "")).strip()
    return value not in {"", "()", "[]", "None"}


def _accepted_feasible_constraint_candidate(event: dict[str, str]) -> bool:
    return (
        _as_bool(event.get("accepted"))
        and _as_bool(event.get("candidate_feasible"))
        and str(event.get("status")) in {"candidate_proposed", "feasible_candidate"}
    )


def _as_int(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _finding(
    finding: str,
    passed: bool,
    observed: object,
    expected: object,
    evidence: str,
) -> dict[str, Any]:
    return {
        "finding": finding,
        "status": "pass" if passed else "fail",
        "observed": observed,
        "expected": expected,
        "evidence": evidence,
    }


def _blocking_review_findings(
    findings: list[dict[str, Any]],
    independent_pending: bool,
) -> list[str]:
    allowed_pending = {
        "independent_complete_rerun_replay",
        "runner_hard_gates",
    }
    if independent_pending:
        return [
            str(row["finding"])
            for row in findings
            if row["status"] == "fail"
        ]
    return [
        str(row["finding"])
        for row in findings
        if row["status"] != "pass"
        or (row["status"] == "pending" and str(row["finding"]) not in allowed_pending)
    ]


def _render_objective(objective: SolutionObjective | None) -> str:
    return json.dumps(objective.key if objective is not None else (), separators=(",", ":"))


def _write_review_artifacts(
    *,
    run_dir: Path,
    review_label: str,
    findings: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    readiness_rows: list[dict[str, Any]],
    environment: dict[str, Any],
    ready: bool,
    review_status: str | None = None,
    review_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    report_path = run_dir / "review_report.md"
    findings_path = run_dir / "review_findings.csv"
    failure_path = run_dir / "failure_analysis.csv"
    readiness_path = run_dir / "stage03_readiness.csv"
    manifest_path = run_dir / "review_manifest.json"
    _write_csv(findings_path, REVIEW_FINDING_FIELDS, findings)
    _write_csv(failure_path, FAILURE_ANALYSIS_FIELDS, failure_rows)
    _write_csv(readiness_path, READINESS_FIELDS, readiness_rows)
    status = review_status or ("READY_FOR_STAGE03" if ready else "NOT_READY_FOR_STAGE03")
    report_lines = [
        f"# {review_label}",
        "",
        f"Final review status: **{status}**",
        "",
        "The review re-read raw solution JSON and raw operator events, then "
        "replayed the unified validator.",
        "",
        "## Findings",
        "",
    ]
    report_lines.extend(
        (
            f"- `{row['finding']}`: **{row['status']}** — "
            f"observed={row['observed']}; expected={row['expected']}."
        )
        for row in findings
    )
    report_lines.extend(
        (
            "",
            "## Failure analysis coverage",
            "",
            f"Retained real failure/poor-quality cases: {len(failure_rows)}.",
            "Constraint categories are derived from raw event reasons; "
            "no synthetic cases are added.",
            "",
            "## Stage 3 entry targets (not claimed as completed in Stage 2.3)",
            "",
            "- 100-customer R/RC median exact charging calls ≤ 100.",
            "- 100-customer R/RC median effective iterations ≥ 50.",
            "- Validator feasibility remains 100%.",
            "- The formal objective does not regress under acceleration.",
        )
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    files = {
        path.name: _sha256(path)
        for path in (report_path, findings_path, failure_path, readiness_path)
    }
    manifest_payload: dict[str, Any] = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_label": review_label,
        "status": status,
        "run_directory": str(run_dir),
        "candidate_environment": environment,
        "review_provenance": _review_provenance(
            _repository_root(), environment
        ),
        "review_metadata": review_metadata or {},
        "finding_count": len(findings),
        "failure_case_count": len(failure_rows),
        "readiness_row_count": len(readiness_rows),
        "files": files,
    }
    manifest_payload["manifest_payload_sha256"] = _payload_sha256(manifest_payload)
    _write_json(manifest_path, manifest_payload)
    return {
        "review_report": report_path,
        "review_findings": findings_path,
        "failure_analysis": failure_path,
        "stage03_readiness": readiness_path,
        "review_manifest": manifest_path,
    }


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _payload_sha256(payload: object) -> str:
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently review Stage 2.3 artifacts")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--review-label", required=True)
    arguments = parser.parse_args()
    outputs = review_run(
        run_dir=arguments.run_dir,
        comparison_dir=arguments.comparison_dir,
        review_label=arguments.review_label,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
