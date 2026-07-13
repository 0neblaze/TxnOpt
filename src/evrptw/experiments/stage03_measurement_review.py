from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evrptw.experiments.stage03_measurement import (
    OBJECTIVE_SCHEMA,
    RAW_PER_RUN_FIELDS,
    _assert_results_path,
    _combined_hash,
    _read_csv,
    _repository_root,
    _sha256,
    _source_hashes,
    _write_csv,
    _write_json,
)
from evrptw.measurement import Stage03Trace
from evrptw.objective import (
    ObjectiveComparison,
    SolutionObjective,
    compare_objectives,
)
from evrptw.parser import parse_schneider
from evrptw.validation import validate_routes

REVIEW_SCHEMA_VERSION = "stage03-review-v1"
READY_FOR_FORMAL = "READY_FOR_STAGE03_FORMAL_MEASUREMENT"
READY_FOR_STAGE31 = "READY_FOR_STAGE03_ACCELERATION"
FINDING_FIELDS = ("finding", "status", "observed", "expected", "evidence")
TRACE_FIELDS = (
    "run_label",
    "instance",
    "seed",
    "started_calls",
    "completed_calls",
    "exact_calls",
    "cache_hits",
    "precomputed_routes",
    "deadline_events",
    "reconciliation_status",
    "reconciliation_checks",
)
DEADLINE_FIELDS = (
    "run_label",
    "instance",
    "seed",
    "started_calls",
    "completed_calls",
    "deadline_events",
    "execution_errors",
    "accepted_time_limit_candidates",
    "classification",
)
BASELINE_FIELDS = (
    "instance",
    "seed",
    "candidate_objective_key",
    "attempt16_objective_key",
    "rerun09_objective_key",
    "vehicle_count_delta_vs_attempt16",
    "classification",
    "gate_status",
    "reason",
)
SUMMARY_FIELDS = (
    "instance",
    "runs",
    "feasible_runs",
    "feasibility_rate",
    "best_objective_key",
    "best_vehicle_count",
    "best_total_distance",
    "best_total_charging_time",
    "best_charging_count",
    "mean_runtime_seconds",
    "median_runtime_seconds",
    "mean_trace_exact_calls",
    "median_trace_exact_calls",
    "mean_trace_cache_hits",
    "median_trace_cache_hits",
    "mean_trace_precomputed_routes",
    "median_trace_precomputed_routes",
    "mean_effective_iterations",
    "median_effective_iterations",
)


@dataclass(frozen=True, slots=True)
class _AuditedRun:
    key: tuple[str, int]
    raw_row: dict[str, str]
    raw_payload: dict[str, Any]
    solution_payload: dict[str, Any]
    environment_payload: dict[str, Any]
    trace: Stage03Trace
    solver_result: SimpleNamespace | None
    instance: Any
    objective: SolutionObjective | None
    validator_feasible: bool
    objective_replay_ok: bool
    validator_violations: tuple[str, ...]
    reconciliation: dict[str, object]
    candidate_state_ok: bool
    event_log_ok: bool
    deadline_row: dict[str, object]


def review_run(
    *,
    run_dir: Path,
    summary_dir: Path | None = None,
    review_label: str | None = None,
) -> dict[str, Path]:
    """Replay a Stage 3.0 output directory and publish only passing summaries."""

    root = _repository_root()
    run_dir = run_dir if run_dir.is_absolute() else root / run_dir
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Stage 3.0 run directory is missing: {run_dir}")
    _assert_results_path(root, run_dir, "run_dir")
    _verify_manifest(run_dir)
    metadata = _read_json(run_dir / "run_metadata.json")
    scope = str(metadata.get("scope", ""))
    expected_instances = tuple(str(value) for value in metadata.get("expected_instances", ()))
    expected_seeds = tuple(int(str(value)) for value in metadata.get("expected_seeds", ()))
    expected_keys = {(instance, seed) for instance in expected_instances for seed in expected_seeds}
    label = review_label or str(metadata.get("run_label", run_dir.name))
    raw_rows = _read_csv(run_dir / "raw_per_run_results.csv")
    findings: list[dict[str, Any]] = []
    findings.append(
        _finding(
            "manifest_integrity",
            True,
            "verified before raw replay",
            "all listed files exist and match SHA-256",
            "manifest.json",
        )
    )
    actual_keys = [(str(row.get("instance", "")), int(row.get("seed", "-1"))) for row in raw_rows]
    duplicate_keys = sorted(key for key, count in Counter(actual_keys).items() if count > 1)
    coverage_ok = set(actual_keys) == expected_keys and not duplicate_keys
    findings.append(
        _finding(
            "run_coverage",
            coverage_ok,
            {"rows": len(actual_keys), "duplicates": duplicate_keys},
            f"exactly {len(expected_keys)} unique instance/seed keys",
            "raw_per_run_results.csv and run_metadata.json",
        )
    )

    rows_by_key = {key: row for key, row in zip(actual_keys, raw_rows, strict=False)}
    audited: list[_AuditedRun] = []
    for key in sorted(expected_keys):
        row = rows_by_key.get(key)
        if row is None:
            continue
        audited.append(_audit_one_run(run_dir, metadata, row))

    validator_ok = bool(audited) and len(audited) == len(expected_keys) and all(
        item.validator_feasible
        and item.objective is not None
        and item.objective_replay_ok
        for item in audited
    )
    replayed_feasible_count = sum(
        item.validator_feasible and item.objective is not None and item.objective_replay_ok
        for item in audited
    )
    findings.append(
        _finding(
            "validator_and_objective_replay",
            validator_ok,
            f"{replayed_feasible_count}/{len(expected_keys)} feasible objective replays",
            f"{len(expected_keys)}/{len(expected_keys)}",
            "raw solutions replayed through validate_routes and evrptw.objective",
        )
    )
    reconciliation_ok = bool(audited) and all(
        item.reconciliation.get("status") == "pass"
        and item.event_log_ok
        for item in audited
    )
    findings.append(
        _finding(
            "trace_reconciliation",
            reconciliation_ok,
            [item.reconciliation.get("status") for item in audited],
            "every raw trace reconciles with ALNSResult and event log",
            "raw trace, raw event log, and solver_result",
        )
    )
    candidate_state_ok = bool(audited) and all(item.candidate_state_ok for item in audited)
    findings.append(
        _finding(
            "candidate_state_vehicle_first",
            candidate_state_ok,
            sum(item.candidate_state_ok for item in audited),
            len(expected_keys),
            "raw candidate_state and neighborhood events",
        )
    )
    deadline_ok = bool(audited) and all(
        _int_value(item.deadline_row["accepted_time_limit_candidates"]) == 0
        for item in audited
    )
    findings.append(
        _finding(
            "deadline_boundary_semantics",
            deadline_ok,
            [item.deadline_row for item in audited],
            "started/completed calls are separate and no time-limit candidate is accepted",
            "raw trace deadline report",
        )
    )

    provenance_ok, provenance_reason = _provenance_ok(root, run_dir, metadata)
    findings.append(
        _finding(
            "provenance_and_stage00_freeze",
            provenance_ok,
            provenance_reason,
            "clean main revision, source/config hashes, both reference states, "
            "unchanged Stage 0 manifest",
            "run_metadata.json, parameters.toml, and Stage 0 manifest",
        )
    )
    run_provenance = [
        _per_run_provenance(root, run_dir, metadata, item.raw_row)
        for item in audited
    ]
    findings.append(
        _finding(
            "per_run_environment_and_instance_provenance",
            bool(audited) and len(run_provenance) == len(expected_keys)
            and all(passed for passed, _ in run_provenance),
            [reason for _, reason in run_provenance],
            "each raw environment, instance, source/config, and reference hash matches",
            "environments/<run>.json, raw_per_run_results.csv, and benchmark files",
        )
    )
    historical_provenance = metadata.get("historical_stage02_baselines", {})
    historical_dirty_ok = all(
        isinstance(value, dict) and value.get("repository_dirty") is True
        for value in historical_provenance.values()
    ) and set(historical_provenance) == {"attempt16", "rerun09"}
    findings.append(
        _finding(
            "historical_stage02_dirty_state_preserved",
            historical_dirty_ok,
            historical_provenance,
            "attempt16 and rerun09 retain their recorded repository_dirty=true",
            "run_metadata.json and historical Stage 2.3 environment records",
        )
    )
    baseline_rows = _load_baseline_rows(root, metadata)
    baseline_comparison = _compare_baselines(audited, baseline_rows)
    c5_rows = [
        row
        for row in baseline_comparison
        if row["instance"] in {"c101C5", "r105C5", "rc105C5"}
    ]
    focused_rows = [
        row
        for row in baseline_comparison
        if row["instance"] in {"c101_21", "r101_21", "rc101_21"}
    ]
    c5_ok = len(c5_rows) == 9 and all(row["gate_status"] == "pass" for row in c5_rows)
    focused_ok = len(focused_rows) == 9 and all(
        row["gate_status"] == "pass" for row in focused_rows
    )
    findings.append(
        _finding(
            "c5_objective_baseline",
            c5_ok,
            c5_rows,
            "C5 objective key equals the Stage 2.3 attempt16 baseline",
            "Stage 2.3 attempt16 per-run results",
        )
    )
    findings.append(
        _finding(
            "focused_100_customer_vehicle_guard",
            focused_ok,
            focused_rows,
            "100-customer runs are feasible and vehicle count is not worse than attempt16",
            "Stage 2.3 attempt16 per-run results",
        )
    )

    ready = all(row["status"] == "pass" for row in findings)
    if scope == "smoke":
        status = READY_FOR_FORMAL if ready else "NOT_READY_FOR_STAGE03_FORMAL_MEASUREMENT"
    elif scope == "formal":
        status = READY_FOR_STAGE31 if ready else "NOT_READY_FOR_STAGE03_1"
    else:
        raise ValueError(f"unsupported Stage 3.0 scope in metadata: {scope}")
    readiness_rows = [
        {
            "gate": row["finding"],
            "status": row["status"],
            "observed": row["observed"],
            "expected": row["expected"],
            "evidence": row["evidence"],
        }
        for row in findings
    ]
    readiness_rows.append(
        _finding(
            "overall_status",
            ready,
            status,
            READY_FOR_FORMAL if scope == "smoke" else READY_FOR_STAGE31,
            "all Stage 3.0 acceptance gates",
        )
    )
    recomputed_rows = [_recomputed_row(item, metadata) for item in audited]
    summary_rows = _summarize(recomputed_rows)
    trace_rows = [_trace_row(item, label) for item in audited]
    deadline_rows = [item.deadline_row | {"run_label": label} for item in audited]
    report_lines = _report_lines(label, scope, status, findings, len(audited))

    review_dir = run_dir / "review"
    review_dir.mkdir(exist_ok=True)
    output_paths = {
        "review_report": review_dir / "review_report.md",
        "review_findings": review_dir / "review_findings.csv",
        "stage03_readiness": review_dir / "stage03_readiness.csv",
        "trace_reconciliation": review_dir / "trace_reconciliation.csv",
        "deadline_report": review_dir / "deadline_report.csv",
        "baseline_comparison": review_dir / "baseline_comparison.csv",
        "recomputed_per_run": review_dir / "recomputed_per_run_results.csv",
        "summary_results": review_dir / "summary_results.csv",
        "review_manifest": review_dir / "review_manifest.json",
    }
    output_paths["review_report"].write_text(report_lines, encoding="utf-8")
    _write_csv(output_paths["review_findings"], FINDING_FIELDS, findings)
    _write_csv(output_paths["stage03_readiness"], FINDING_FIELDS, readiness_rows)
    _write_csv(output_paths["trace_reconciliation"], TRACE_FIELDS, trace_rows)
    _write_csv(output_paths["deadline_report"], DEADLINE_FIELDS, deadline_rows)
    _write_csv(output_paths["baseline_comparison"], BASELINE_FIELDS, baseline_comparison)
    _write_csv(output_paths["recomputed_per_run"], RAW_PER_RUN_FIELDS, recomputed_rows)
    _write_csv(output_paths["summary_results"], SUMMARY_FIELDS, summary_rows)
    review_manifest = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_label": label,
        "scope": scope,
        "status": status,
        "run_directory": str(run_dir),
        "finding_count": len(findings),
        "files": {
            path.name: _sha256(path)
            for path in output_paths.values()
            if path != output_paths["review_manifest"]
        },
    }
    review_manifest["manifest_payload_sha256"] = _payload_sha256(review_manifest)
    _write_json(output_paths["review_manifest"], review_manifest)

    if ready and summary_dir is not None:
        _publish_summaries(summary_dir, label, output_paths)
        output_paths["published_summary_dir"] = summary_dir
    return output_paths


def _audit_one_run(
    run_dir: Path,
    metadata: dict[str, Any],
    row: dict[str, str],
) -> _AuditedRun:
    instance_name = str(row["instance"])
    seed = int(str(row["seed"]))
    raw_payload = _read_json(run_dir / row["raw_path"])
    solution_payload = _read_json(run_dir / row["solution_path"])
    environment_payload = _read_json(run_dir / row["environment_path"])
    trace = Stage03Trace.from_dict(_read_json(run_dir / row["trace_path"]))
    event_log = _read_jsonl(run_dir / row["event_path"])
    solver_payload = raw_payload.get("solver_result")
    solver_result = SimpleNamespace(**solver_payload) if isinstance(solver_payload, dict) else None
    benchmark_dir = Path(str(metadata["benchmark_directory"]))
    instance_path = benchmark_dir / f"{instance_name}.txt"
    instance = parse_schneider(instance_path)
    routes = [[str(value) for value in route] for route in solution_payload.get("routes", [])]
    report = validate_routes(instance, routes)
    objective = SolutionObjective.from_report(instance, report) if report.feasible else None
    solution_objective = _objective_from_key(solution_payload.get("objective_key", ()))
    solver_objective = solver_payload.get("objective") if isinstance(solver_payload, dict) else None
    solver_solution_objective = (
        SolutionObjective(
            int(solver_objective["vehicle_count"]),
            float(solver_objective["total_distance"]),
            float(solver_objective["total_charging_time"]),
            int(solver_objective["charging_count"]),
        )
        if isinstance(solver_objective, dict)
        else None
    )
    objective_replay_ok = (
        objective is not None
        and solution_objective is not None
        and solver_solution_objective is not None
        and compare_objectives(solution_objective, objective) is ObjectiveComparison.EQUAL
        and compare_objectives(solver_solution_objective, objective)
        is ObjectiveComparison.EQUAL
    )
    candidate_state_ok = _candidate_state_ok(trace, solver_result)
    event_log_ok = _event_log_ok(event_log, trace, solver_result)
    reconciliation: dict[str, object]
    if solver_result is not None:
        reconciliation = trace.reconcile(solver_result)
    else:
        reconciliation = {"status": "not_available", "checks": {}}
    route_violations = tuple(
        item for route in report.routes for item in route.violations
    )
    deadline_row = _deadline_row(trace, event_log, instance_name, seed)
    return _AuditedRun(
        key=(instance_name, seed),
        raw_row=row,
        raw_payload=raw_payload,
        solution_payload=solution_payload,
        environment_payload=environment_payload,
        trace=trace,
        solver_result=solver_result,
        instance=instance,
        objective=objective,
        validator_feasible=report.feasible,
        objective_replay_ok=objective_replay_ok,
        validator_violations=(*report.violations, *route_violations),
        reconciliation=reconciliation,
        candidate_state_ok=candidate_state_ok,
        event_log_ok=event_log_ok,
        deadline_row=deadline_row,
    )


def _candidate_state_ok(
    trace: Stage03Trace,
    solver_result: SimpleNamespace | None = None,
) -> bool:
    candidate_states = [
        event
        for event in trace.events
        if event.get("event_type") == "candidate_state"
    ]
    for event in trace.events:
        if event.get("event_type") != "candidate_state":
            continue
        current_keys = _string_tuple(event.get("current_route_keys", ()))
        candidate_keys = _string_tuple(event.get("candidate_route_keys", ()))
        if any(key not in trace.route_dictionary for key in (*current_keys, *candidate_keys)):
            return False
        if event.get("accepted") is True:
            if not candidate_keys or not event.get("candidate_objective_key"):
                return False
            if bool(event.get("candidate_feasible")) is not True:
                return False
            if _int_value(event.get("candidate_vehicle_count", 0)) > _int_value(
                event.get("current_vehicle_count", 0)
            ):
                return False
            if event.get("status") == "time_limit":
                return False
    if solver_result is None:
        return True
    legacy_states = [event for event in candidate_states if event.get("lane") == "legacy"]
    return (
        len(legacy_states) == _int_value(getattr(solver_result, "effective_iterations", 0))
        and sum(event.get("accepted") is True for event in legacy_states)
        == _int_value(getattr(solver_result, "accepted_moves", 0))
        and sum(event.get("accepted") is not True for event in legacy_states)
        == _int_value(getattr(solver_result, "rejected_moves", 0))
    )


def _event_log_ok(
    event_log: list[dict[str, Any]],
    trace: Stage03Trace,
    solver_result: SimpleNamespace | None,
) -> bool:
    trace_events = [
        event.get("payload", {})
        for event in event_log
        if event.get("record_type") == "trace_event"
    ]
    neighborhood_events = [
        event.get("payload", {})
        for event in event_log
        if event.get("record_type") == "neighborhood_event"
    ]
    if len(trace_events) != len(trace.events):
        return False
    if any(
        _canonical_json(left) != _canonical_json(right)
        for left, right in zip(trace_events, trace.events, strict=True)
    ):
        return False
    if solver_result is None:
        return not neighborhood_events
    expected_neighborhood = list(solver_result.neighborhood_events)
    if len(neighborhood_events) != len(expected_neighborhood):
        return False
    for event in neighborhood_events:
        if event.get("accepted") is True:
            if event.get("candidate_feasible") is not True:
                return False
            if _int_value(event.get("candidate_vehicle_delta", 0)) > 0:
                return False
    return all(
        _canonical_json(left) == _canonical_json(right)
        for left, right in zip(neighborhood_events, expected_neighborhood, strict=True)
    )


def _deadline_row(
    trace: Stage03Trace,
    event_log: list[dict[str, Any]],
    instance: str,
    seed: int,
) -> dict[str, object]:
    accepted_time_limit = sum(
        event.get("event_type") == "candidate_state"
        and event.get("status") == "time_limit"
        and event.get("accepted") is True
        for event in trace.events
    )
    execution_errors = sum(event.get("event_type") == "execution_error" for event in trace.events)
    return {
        "run_label": "",
        "instance": instance,
        "seed": seed,
        "started_calls": trace.started_calls,
        "completed_calls": trace.completed_calls,
        "deadline_events": trace.deadline_events,
        "execution_errors": execution_errors,
        "accepted_time_limit_candidates": accepted_time_limit,
        "classification": (
            "cooperative_deadline_observed"
            if trace.deadline_events
            else "no_deadline_boundary_observed"
        ),
        "event_log_rows": len(event_log),
    }


def _provenance_ok(root: Path, run_dir: Path, metadata: dict[str, Any]) -> tuple[bool, str]:
    if metadata.get("repository_dirty") is not False:
        return False, f"repository_dirty={metadata.get('repository_dirty')}"
    try:
        current_sources = _source_hashes(root)
    except FileNotFoundError as error:
        return False, str(error)
    if current_sources != metadata.get("algorithm_source_files"):
        return False, "current source hashes differ from the run metadata"
    if _combined_hash(current_sources) != metadata.get("algorithm_source_sha256"):
        return False, "combined source hash differs from the run metadata"
    parameters = run_dir / "parameters.toml"
    config_hash = _sha256(parameters) if parameters.is_file() else ""
    if config_hash != metadata.get("configuration_sha256"):
        return False, "parameters.toml hash differs from the run metadata"
    baseline_manifest = Path(str(metadata["baseline_directory"])) / "manifest.json"
    if _sha256(baseline_manifest) != metadata.get("stage00_manifest_sha256"):
        return False, "Stage 0 manifest changed after the run"
    expected_hashes = {
        "stage00_configuration_sha256": root / "configs" / "stage00_baseline.toml",
        "stage02_configuration_sha256": root / "configs" / "stage02_constraint_guided.toml",
        "stage02_attempt16_per_run_sha256": Path(
            str(metadata["stage02_attempt16_per_run"])
        ),
        "stage02_rerun09_per_run_sha256": Path(str(metadata["stage02_rerun09_per_run"])),
    }
    for metadata_key, path in expected_hashes.items():
        if not path.is_absolute():
            path = root / path
        if _sha256(path) != metadata.get(metadata_key):
            return False, f"{metadata_key} changed after the run"
    return True, "source/config/Stage 0/reference provenance matches raw metadata"


def _per_run_provenance(
    root: Path,
    run_dir: Path,
    metadata: dict[str, Any],
    row: dict[str, str],
) -> tuple[bool, str]:
    try:
        environment_path = run_dir / row["environment_path"]
        environment = _read_json(environment_path)
        benchmark_dir = Path(str(metadata["benchmark_directory"]))
        if not benchmark_dir.is_absolute():
            benchmark_dir = root / benchmark_dir
        instance_path = benchmark_dir / f"{row['instance']}.txt"
        expected_reference = metadata.get("reference_repositories", {})
        checks = {
            "environment_hash": _payload_sha256(environment) == row["environment_sha256"],
            "instance_hash": _sha256(instance_path) == row["instance_sha256"]
            == environment.get("instance_sha256"),
            "repository_revision": environment.get("repository_revision")
            == metadata.get("repository_revision")
            == row["repository_revision"],
            "repository_dirty": environment.get("repository_dirty") is False
            and row["repository_dirty"] == "False",
            "source_hash": environment.get("algorithm_source_sha256")
            == metadata.get("algorithm_source_sha256")
            == row["algorithm_source_sha256"],
            "configuration_hash": environment.get("configuration_sha256")
            == metadata.get("configuration_sha256")
            == row["configuration_sha256"],
            "stage00_manifest_hash": environment.get("stage00_manifest_sha256")
            == metadata.get("stage00_manifest_sha256")
            == row["stage00_manifest_sha256"],
            "reference_repositories": environment.get("reference_repositories")
            == expected_reference,
        }
    except (KeyError, FileNotFoundError, TypeError, ValueError) as error:
        return False, f"provenance exception: {error}"
    return all(checks.values()), json.dumps(checks, sort_keys=True)


def _load_baseline_rows(
    root: Path,
    metadata: dict[str, Any],
) -> dict[str, dict[tuple[str, int], tuple[int, float, float, int]]]:
    output: dict[str, dict[tuple[str, int], tuple[int, float, float, int]]] = {}
    for label, raw_path in (
        ("attempt16", metadata.get("stage02_attempt16_per_run")),
        ("rerun09", metadata.get("stage02_rerun09_per_run")),
    ):
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = root / path
        rows = _read_csv(path)
        output[label] = {}
        for row in rows:
            key = (str(row["instance"]), int(row["seed"]))
            objective_key = json.loads(row["objective_key"])
            output[label][key] = (
                int(objective_key[0]),
                float(objective_key[1]),
                float(objective_key[2]),
                int(objective_key[3]),
            )
    return output


def _compare_baselines(
    audited: list[_AuditedRun],
    baselines: dict[str, dict[tuple[str, int], tuple[int, float, float, int]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in audited:
        candidate_key = item.objective.key if item.objective is not None else ()
        attempt16 = baselines.get("attempt16", {}).get(item.key, ())
        rerun09 = baselines.get("rerun09", {}).get(item.key, ())
        candidate_objective = item.objective
        attempt16_objective = _objective_from_key(attempt16)
        rerun09_objective = _objective_from_key(rerun09)
        if candidate_objective is None or attempt16_objective is None or rerun09_objective is None:
            classification = "missing_candidate_or_baseline"
            gate = "fail"
            vehicle_delta: object = ""
        else:
            vehicle_delta = (
                candidate_objective.vehicle_count - attempt16_objective.vehicle_count
            )
            if item.key[0] in {"c101C5", "r105C5", "rc105C5"}:
                gate = (
                    "pass"
                    if compare_objectives(candidate_objective, attempt16_objective)
                    is ObjectiveComparison.EQUAL
                    and compare_objectives(candidate_objective, rerun09_objective)
                    is ObjectiveComparison.EQUAL
                    else "fail"
                )
                classification = "unchanged" if gate == "pass" else "regression"
                reason = "C5 objective key must match both historical Stage 2.3 baselines"
            elif item.key[0] in {"c101_21", "r101_21", "rc101_21"}:
                gate = (
                    "pass"
                    if item.validator_feasible
                    and candidate_objective.vehicle_count <= attempt16_objective.vehicle_count
                    and candidate_objective.vehicle_count <= rerun09_objective.vehicle_count
                    else "fail"
                )
                classification = "vehicle_guard_pass" if gate == "pass" else "vehicle_regression"
                reason = (
                    "100-customer vehicle count guard against both historical baselines; "
                    "wall-clock is time-budget variation"
                )
            else:
                gate = "pass"
                classification = "time_budget_variation"
                reason = "Stage 3.0 records measurement only; no acceleration claim"
        output.append(
            {
                "instance": item.key[0],
                "seed": item.key[1],
                "candidate_objective_key": json.dumps(candidate_key),
                "attempt16_objective_key": json.dumps(attempt16),
                "rerun09_objective_key": json.dumps(rerun09),
                "vehicle_count_delta_vs_attempt16": vehicle_delta,
                "classification": classification,
                "gate_status": gate,
                "reason": (
                    reason
                    if candidate_objective is not None
                    and attempt16_objective is not None
                    and rerun09_objective is not None
                    else "raw baseline coverage is missing"
                ),
            }
        )
    return output


def _recomputed_row(
    item: _AuditedRun,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    raw = item.raw_row
    solver = item.solver_result
    environment = item.environment_payload
    reference_repositories = environment.get("reference_repositories", {})
    instance_path = Path(str(metadata["benchmark_directory"])) / f"{item.key[0]}.txt"
    solver_feasible = bool(getattr(solver, "feasible", False)) if solver is not None else False
    solver_metrics = {
        field: getattr(solver, field, "") if solver is not None else ""
        for field in (
            "runtime_seconds",
            "first_feasible_time",
            "best_time",
            "iterations",
            "effective_iterations",
            "accepted_moves",
            "improving_moves",
            "rejected_moves",
            "charging_subproblem_calls",
            "total_energy",
            "total_charged_energy",
        )
    }
    result: dict[str, Any] = {field: "" for field in RAW_PER_RUN_FIELDS}
    result.update(
        {
            "schema_version": metadata["schema_version"],
            "run_label": metadata["run_label"],
            "experiment_id": raw.get("experiment_id", ""),
            "scope": metadata["scope"],
            "instance": item.key[0],
            "seed": item.key[1],
            "algorithm": metadata["algorithm"],
            "operator_profile": metadata["operator_profile"],
            "repository_revision": environment.get("repository_revision", ""),
            "repository_dirty": environment.get("repository_dirty", ""),
            "algorithm_source_sha256": environment.get("algorithm_source_sha256", ""),
            "configuration_sha256": environment.get("configuration_sha256", ""),
            "instance_sha256": _sha256(instance_path),
            "stage00_manifest_sha256": environment.get("stage00_manifest_sha256", ""),
            "environment_sha256": _payload_sha256(environment),
            "reference_vrp_evrp_hub_revision": reference_repositories.get(
                "VRP-EVRP-Project-Hub", {}
            ).get("revision", ""),
            "reference_vrp_evrp_hub_dirty": reference_repositories.get(
                "VRP-EVRP-Project-Hub", {}
            ).get("dirty", ""),
            "reference_py_ga_vrptw_revision": reference_repositories.get(
                "py-ga-VRPTW", {}
            ).get("revision", ""),
            "reference_py_ga_vrptw_dirty": reference_repositories.get(
                "py-ga-VRPTW", {}
            ).get("dirty", ""),
            "start_utc": environment.get("run_start_utc", ""),
            "end_utc": environment.get("run_end_utc", ""),
            "time_limit_seconds": metadata["time_limit_seconds"],
            "max_iterations": metadata["max_iterations"],
            "threads": metadata["threads"],
            "objective_schema": OBJECTIVE_SCHEMA,
            "objective_key": json.dumps(item.objective.key if item.objective else ()),
            "vehicle_count": item.objective.vehicle_count if item.objective else "",
            "total_distance": item.objective.total_distance if item.objective else "",
            "total_charging_time": (
                item.objective.total_charging_time if item.objective else ""
            ),
            "charging_count": item.objective.charging_count if item.objective else "",
            "feasible": (
                item.validator_feasible
                and item.objective is not None
                and item.objective_replay_ok
                and solver_feasible
            ),
            "status": (
                "feasible"
                if item.validator_feasible and item.objective_replay_ok and solver_feasible
                else "invalid"
            ),
            "trace_started_calls": item.trace.started_calls,
            "trace_completed_calls": item.trace.completed_calls,
            "trace_exact_calls": item.trace.exact_calls,
            "trace_cache_hits": item.trace.cache_hits,
            "trace_precomputed_routes": item.trace.precomputed_routes,
            "trace_route_evaluations": len(item.trace.route_evaluations),
            "trace_deadline_events": item.trace.deadline_events,
            "trace_reconciliation_status": item.reconciliation.get("status", "not_available"),
            **solver_metrics,
            "peak_tracemalloc_bytes": environment.get("peak_tracemalloc_bytes", ""),
            "peak_rss_bytes": environment.get("peak_rss_bytes", ""),
            "raw_path": raw.get("raw_path", ""),
            "solution_path": raw.get("solution_path", ""),
            "trace_path": raw.get("trace_path", ""),
            "event_path": raw.get("event_path", ""),
            "environment_path": raw.get("environment_path", ""),
            "failure_reason": (
                "; ".join(item.validator_violations)
                or str(getattr(solver, "failure_reason", ""))
            ),
        }
    )
    if solver is None and not result["failure_reason"]:
        result["failure_reason"] = "solver result unavailable"
    return result


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["instance"])].append(row)
    output: list[dict[str, Any]] = []
    for instance, group in sorted(grouped.items()):
        feasible = [row for row in group if row.get("feasible") in {True, "True"}]
        objectives = [json.loads(str(row["objective_key"])) for row in feasible]
        runtimes = [
            float(row["runtime_seconds"])
            for row in group
            if row.get("runtime_seconds", "") != ""
        ]
        exact_calls = [int(row["trace_exact_calls"]) for row in group]
        cache_hits = [int(row["trace_cache_hits"]) for row in group]
        precomputed = [int(row["trace_precomputed_routes"]) for row in group]
        iterations = [
            int(row["effective_iterations"])
            for row in group
            if row.get("effective_iterations", "") != ""
        ]
        best_objective = _best_objective(objectives)
        best = best_objective.key if best_objective is not None else ()
        output.append(
            {
                "instance": instance,
                "runs": len(group),
                "feasible_runs": len(feasible),
                "feasibility_rate": len(feasible) / len(group) if group else 0.0,
                "best_objective_key": json.dumps(best),
                "best_vehicle_count": best[0] if best else "",
                "best_total_distance": best[1] if best else "",
                "best_total_charging_time": best[2] if best else "",
                "best_charging_count": best[3] if best else "",
                "mean_runtime_seconds": _mean(runtimes),
                "median_runtime_seconds": _median(runtimes),
                "mean_trace_exact_calls": _mean(exact_calls),
                "median_trace_exact_calls": _median(exact_calls),
                "mean_trace_cache_hits": _mean(cache_hits),
                "median_trace_cache_hits": _median(cache_hits),
                "mean_trace_precomputed_routes": _mean(precomputed),
                "median_trace_precomputed_routes": _median(precomputed),
                "mean_effective_iterations": _mean(iterations),
                "median_effective_iterations": _median(iterations),
            }
        )
    return output


def _trace_row(item: _AuditedRun, label: str) -> dict[str, Any]:
    return {
        "run_label": label,
        "instance": item.key[0],
        "seed": item.key[1],
        "started_calls": item.trace.started_calls,
        "completed_calls": item.trace.completed_calls,
        "exact_calls": item.trace.exact_calls,
        "cache_hits": item.trace.cache_hits,
        "precomputed_routes": item.trace.precomputed_routes,
        "deadline_events": item.trace.deadline_events,
        "reconciliation_status": item.reconciliation.get("status", "not_available"),
        "reconciliation_checks": json.dumps(item.reconciliation.get("checks", {}), sort_keys=True),
    }


def _report_lines(
    label: str,
    scope: str,
    status: str,
    findings: list[dict[str, Any]],
    run_count: int,
) -> str:
    lines = [
        f"# {label}",
        "",
        f"Scope: `{scope}`; raw runs replayed: `{run_count}`.",
        "",
        f"Final review status: **{status}**",
        "",
        "This review trusts the raw solution, raw trace, raw event log, environment records, "
        "and manifest. "
        "Tracked summaries are recomputed only after the raw replay gates pass.",
        "",
        "## Findings",
        "",
    ]
    lines.extend(
        f"- `{row['finding']}`: **{row['status']}** — "
        f"observed={row['observed']}; expected={row['expected']}."
        for row in findings
    )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "Stage 3.0 measures exact charging calls and replayability only. "
            "It does not claim cache acceleration, incremental propagation, "
            "interruptible exact solving, "
            "parallel evaluation, or fixed-work/wall-clock improvement.",
        )
    )
    return "\n".join(lines) + "\n"


def _publish_summaries(
    summary_dir: Path,
    label: str,
    output_paths: dict[str, Path],
) -> None:
    root = _repository_root()
    summaries_root = (root / "experiments" / "summaries").resolve()
    candidate = summary_dir.resolve()
    try:
        candidate.relative_to(summaries_root)
    except ValueError as error:
        raise ValueError(
            "Stage 3.0 summary_dir must be below experiments/summaries/: "
            f"{candidate}"
        ) from error
    summary_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "review_report": f"{label}_review_report.md",
        "review_findings": f"{label}_review_findings.csv",
        "stage03_readiness": f"{label}_stage03_readiness.csv",
        "trace_reconciliation": f"{label}_trace_reconciliation.csv",
        "deadline_report": f"{label}_deadline_report.csv",
        "baseline_comparison": f"{label}_baseline_comparison.csv",
        "recomputed_per_run": f"{label}_per_run_results.csv",
        "summary_results": f"{label}_summary_results.csv",
        "review_manifest": f"{label}_review_manifest.json",
    }
    destinations = [summary_dir / name for name in names.values()]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"tracked Stage 3.0 summaries already exist: {existing}")
    for key, name in names.items():
        shutil.copy2(output_paths[key], summary_dir / name)


def _verify_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage 3.0 manifest is missing: {manifest_path}")
    sidecar_path = run_dir / "manifest.sha256"
    if not sidecar_path.is_file():
        raise ValueError(f"Stage 3.0 manifest sidecar is missing: {sidecar_path}")
    observed_manifest_hash = sidecar_path.read_text(encoding="utf-8").strip()
    if observed_manifest_hash != _sha256(manifest_path):
        raise ValueError(
            "Stage 3.0 manifest sidecar hash mismatch: "
            f"expected {observed_manifest_hash}, observed {_sha256(manifest_path)}"
        )
    payload = _read_json(manifest_path)
    if payload.get("schema_version") != "1":
        raise ValueError(
            f"unsupported Stage 3.0 manifest schema: {payload.get('schema_version')}"
        )
    expected = dict(payload.get("files", {}))
    observed: dict[str, str] = {}
    for relative, expected_hash in expected.items():
        path = run_dir / relative
        if not path.is_file():
            raise ValueError(f"Stage 3.0 manifest lists a missing file: {path}")
        observed[relative] = _sha256(path)
        if observed[relative] != expected_hash:
            raise ValueError(
                f"Stage 3.0 manifest hash mismatch for {relative}: "
                f"expected {expected_hash}, observed {observed[relative]}"
            )
    actual_files = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
        and path != manifest_path
        and path != sidecar_path
        and "review" not in path.relative_to(run_dir).parts
    }
    if actual_files != set(expected):
        raise ValueError(
            "Stage 3.0 manifest coverage mismatch: "
            f"missing={sorted(set(expected) - actual_files)} "
            f"unexpected={sorted(actual_files - set(expected))}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


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
        "observed": _render(observed),
        "expected": _render(expected),
        "evidence": evidence,
    }


def _render(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)


def _objective_from_key(value: object) -> SolutionObjective | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return SolutionObjective(
            int(value[0]),
            float(value[1]),
            float(value[2]),
            int(value[3]),
        )
    except (TypeError, ValueError):
        return None


def _best_objective(values: list[object]) -> SolutionObjective | None:
    best: SolutionObjective | None = None
    for value in values:
        candidate = _objective_from_key(value)
        if candidate is None:
            continue
        if best is None or compare_objectives(candidate, best) is ObjectiveComparison.BETTER:
            best = candidate
    return best


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _int_value(value: object) -> int:
    if value is None or value == "":
        return 0
    if isinstance(value, (bool, int, float, str)):
        return int(value)
    raise TypeError(f"expected an integer-like value, got {type(value).__name__}")


def _mean(values: list[int] | list[float]) -> float | str:
    return sum(values) / len(values) if values else ""


def _median(values: list[int] | list[float]) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay-audit Stage 3.0 raw evidence")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--review-label")
    arguments = parser.parse_args()
    outputs = review_run(
        run_dir=arguments.run_dir,
        summary_dir=arguments.summary_dir,
        review_label=arguments.review_label,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    status = _read_json(outputs["review_manifest"])["status"]
    return 0 if status in {READY_FOR_FORMAL, READY_FOR_STAGE31} else 1


if __name__ == "__main__":
    raise SystemExit(main())
