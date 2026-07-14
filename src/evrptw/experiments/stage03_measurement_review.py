from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from evrptw.cache_incremental import (
    RouteCacheKey,
    build_route_propagation_snapshot,
)
from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES, FORMAL_SEEDS
from evrptw.experiments.stage03_measurement import (
    OBJECTIVE_SCHEMA,
    RAW_PER_RUN_FIELDS,
    SMOKE_INSTANCES,
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
READY_FOR_STAGE031_FORMAL = "READY_FOR_STAGE031_FORMAL_MEASUREMENT"
READY_FOR_STAGE032 = "READY_FOR_STAGE03_2"
READY_FOR_STAGE032_FORMAL = "READY_FOR_STAGE032_FORMAL_MEASUREMENT"
READY_FOR_STAGE033 = "READY_FOR_STAGE03_3"
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
    "screening_calls",
    "screening_passes",
    "screening_rejections",
    "screening_cache_hits",
    "screening_exact_call_blocked",
    "screening_reason_counts",
    "cache_incremental_counts",
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
    "mean_screening_calls",
    "median_screening_calls",
    "mean_screening_rejections",
    "median_screening_rejections",
    "mean_screening_cache_hits",
    "median_screening_cache_hits",
    "mean_effective_iterations",
    "median_effective_iterations",
)
SCREENING_REASON_FIELDS = ("run_label", "instance", "seed", "reason", "count")
STAGE03_COMPARISON_FIELDS = (
    "instance",
    "seed",
    "candidate_objective_key",
    "stage03_objective_key",
    "candidate_feasible",
    "stage03_feasible",
    "candidate_vehicle_count",
    "stage03_vehicle_count",
    "candidate_exact_calls",
    "stage03_exact_calls",
    "exact_call_delta",
    "classification",
    "gate_status",
    "reason",
)


@dataclass(frozen=True, slots=True)
class _TraceSummary:
    started_calls: int
    completed_calls: int
    exact_calls: int
    cache_hits: int
    precomputed_routes: int
    deadline_events: int
    route_evaluation_count: int
    screening_counts: dict[str, object]
    cache_incremental_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class _AuditedRun:
    key: tuple[str, int]
    raw_row: dict[str, str]
    raw_payload: dict[str, Any]
    solution_payload: dict[str, Any]
    environment_payload: dict[str, Any]
    trace_summary: _TraceSummary
    solver_result: SimpleNamespace | None
    instance: Any
    objective: SolutionObjective | None
    validator_feasible: bool
    objective_replay_ok: bool
    validator_violations: tuple[str, ...]
    reconciliation: dict[str, object]
    candidate_state_ok: bool
    event_log_ok: bool
    screening_ok: bool
    cache_incremental_ok: bool
    screening_reason_counts: dict[str, int]
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
    screening_payload = metadata.get("screening_config")
    screening_enabled = isinstance(screening_payload, dict) and bool(
        screening_payload.get("enabled", False)
    )
    cache_payload = metadata.get("cache_incremental_config")
    cache_incremental_enabled = isinstance(cache_payload, dict) and bool(
        cache_payload.get("enabled", False)
    )
    expected_instances = (
        tuple(SMOKE_INSTANCES) if scope == "smoke" else tuple(FORMAL_INSTANCES)
    )
    expected_seeds = tuple(FORMAL_SEEDS)
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
    screening_ok = (not screening_enabled) or bool(audited) and all(
        item.screening_ok for item in audited
    )
    if screening_enabled:
        findings.append(
            _finding(
                "cheap_screening_reconciliation",
                screening_ok,
                [
                    {
                        "instance": item.key[0],
                        "seed": item.key[1],
                        "screening": item.trace_summary.screening_counts,
                    }
                    for item in audited
                ],
                "every exact/cache route evaluation follows a screening pass; "
                "screening rejection/cache hit blocks exact charging",
                "raw trace screening_decisions and route_evaluations",
            )
        )

    cache_incremental_ok = (not cache_incremental_enabled) or (
        bool(audited)
        and all(
            item.cache_incremental_ok
            for item in audited
        )
    )
    if cache_incremental_enabled:
        findings.append(
            _finding(
                "cache_incremental_reconciliation",
                cache_incremental_ok,
                [
                    {
                        "instance": item.key[0],
                        "seed": item.key[1],
                        "cache_incremental": item.trace_summary.cache_incremental_counts,
                    }
                    for item in audited
                ],
                "cache key/lifecycle, screening order, changed-route propagation, "
                "station bitsets, evictions, and ALNSResult counters reconcile",
                "raw trace, raw event log, solver_result, and independent recomputation",
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

    stage03_comparison: list[dict[str, Any]] = []
    if screening_enabled:
        stage03_baselines = _load_stage03_formal_rows(root, metadata)
        stage03_comparison = _compare_stage03_formal(audited, stage03_baselines)
        findings.append(
            _finding(
                "stage03_formal_baseline_comparison",
                bool(stage03_comparison)
                and all(row["gate_status"] == "pass" for row in stage03_comparison),
                stage03_comparison,
                "C5 objective/feasibility is not worse and 100-customer vehicle count "
                "does not increase against Stage 3.0 formal evidence",
                "Stage 3.0 formal per-run results and raw Stage 3.1 trace",
            )
        )

    ready = all(row["status"] == "pass" for row in findings)
    if cache_incremental_enabled and scope == "smoke":
        status = (
            READY_FOR_STAGE032_FORMAL
            if ready
            else "NOT_READY_FOR_STAGE032_FORMAL_MEASUREMENT"
        )
    elif cache_incremental_enabled and scope == "formal":
        status = READY_FOR_STAGE033 if ready else "NOT_READY_FOR_STAGE03_3"
    elif screening_enabled and scope == "smoke":
        status = (
            READY_FOR_STAGE031_FORMAL
            if ready
            else "NOT_READY_FOR_STAGE031_FORMAL_MEASUREMENT"
        )
    elif screening_enabled and scope == "formal":
        status = READY_FOR_STAGE032 if ready else "NOT_READY_FOR_STAGE03_2"
    elif scope == "smoke":
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
            (
                READY_FOR_STAGE032_FORMAL
                if cache_incremental_enabled and scope == "smoke"
                else READY_FOR_STAGE033
                if cache_incremental_enabled
                else READY_FOR_STAGE031_FORMAL
                if screening_enabled and scope == "smoke"
                else READY_FOR_STAGE032
                if screening_enabled
                else READY_FOR_FORMAL
                if scope == "smoke"
                else READY_FOR_STAGE31
            ),
            "all Stage 3.2 acceptance gates"
            if cache_incremental_enabled
            else "all Stage 3.1 acceptance gates"
            if screening_enabled
            else "all Stage 3.0 acceptance gates",
        )
    )
    recomputed_rows = [_recomputed_row(item, metadata) for item in audited]
    summary_rows = _summarize(recomputed_rows)
    trace_rows = [_trace_row(item, label) for item in audited]
    deadline_rows = [item.deadline_row | {"run_label": label} for item in audited]
    screening_rows = _screening_reason_rows(audited, label) if screening_enabled else []
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
    if screening_enabled:
        output_paths["screening_reason_statistics"] = (
            review_dir / "screening_reason_statistics.csv"
        )
        output_paths["stage03_formal_comparison"] = (
            review_dir / "stage03_formal_comparison.csv"
        )
    if cache_incremental_enabled:
        output_paths["cache_incremental_statistics"] = (
            review_dir / "cache_incremental_statistics.csv"
        )
    output_paths["review_report"].write_text(report_lines, encoding="utf-8")
    _write_csv(output_paths["review_findings"], FINDING_FIELDS, findings)
    _write_csv(output_paths["stage03_readiness"], FINDING_FIELDS, readiness_rows)
    _write_csv(output_paths["trace_reconciliation"], TRACE_FIELDS, trace_rows)
    _write_csv(output_paths["deadline_report"], DEADLINE_FIELDS, deadline_rows)
    _write_csv(output_paths["baseline_comparison"], BASELINE_FIELDS, baseline_comparison)
    _write_csv(output_paths["recomputed_per_run"], RAW_PER_RUN_FIELDS, recomputed_rows)
    _write_csv(output_paths["summary_results"], SUMMARY_FIELDS, summary_rows)
    if screening_enabled:
        _write_csv(
            output_paths["screening_reason_statistics"],
            SCREENING_REASON_FIELDS,
            screening_rows,
        )
        _write_csv(
            output_paths["stage03_formal_comparison"],
            STAGE03_COMPARISON_FIELDS,
            stage03_comparison,
        )
    if cache_incremental_enabled:
        _write_csv(
            output_paths["cache_incremental_statistics"],
            ("run_label", "instance", "seed", "metric", "value"),
            _cache_statistics_rows(audited, label),
        )
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
    screening_ok = _screening_trace_ok(trace)
    cache_incremental_ok = (
        _cache_incremental_trace_ok(trace, instance, solver_result)
        if trace.cache_incremental_config is not None
        and bool(getattr(trace.cache_incremental_config, "enabled", False))
        else False
    )
    screening_reason_counts: dict[str, int] = {}
    for decision in trace.screening_decisions:
        reason = decision.reason or "screening_pass"
        screening_reason_counts[reason] = screening_reason_counts.get(reason, 0) + 1
    event_log_ok, event_log_rows = _event_log_ok_stream(
        run_dir / row["event_path"],
        trace,
        solver_result,
    )
    reconciliation: dict[str, object]
    if solver_result is not None:
        reconciliation = trace.reconcile(solver_result)
    else:
        reconciliation = {"status": "not_available", "checks": {}}
    route_violations = tuple(
        item for route in report.routes for item in route.violations
    )
    deadline_row = _deadline_row(trace, event_log_rows, instance_name, seed)
    trace_summary = _trace_summary(trace)
    solver_result_payload = solver_result
    if solver_result_payload is not None:
        solver_result_payload.neighborhood_events = []
    return _AuditedRun(
        key=(instance_name, seed),
        raw_row=row,
        raw_payload={},
        solution_payload={},
        environment_payload=environment_payload,
        trace_summary=trace_summary,
        solver_result=solver_result_payload,
        instance=instance,
        objective=objective,
        validator_feasible=report.feasible,
        objective_replay_ok=objective_replay_ok,
        validator_violations=(*report.violations, *route_violations),
        reconciliation=reconciliation,
        candidate_state_ok=candidate_state_ok,
        event_log_ok=event_log_ok,
        screening_ok=screening_ok,
        cache_incremental_ok=cache_incremental_ok,
        screening_reason_counts=screening_reason_counts,
        deadline_row=deadline_row,
    )


def _screening_trace_ok(trace: Stage03Trace) -> bool:
    if trace.screening_config is None or not trace.screening_config.enabled:
        return False
    allowed_statuses = {"pass", "rejected", "negative_cache_hit"}
    decisions_by_key: defaultdict[tuple[str, str], list[Any]] = defaultdict(list)
    decisions_by_route: defaultdict[str, list[Any]] = defaultdict(list)
    evaluations_by_key: defaultdict[tuple[str, str], list[Any]] = defaultdict(list)
    evaluations_by_route: defaultdict[str, list[Any]] = defaultdict(list)
    for decision in trace.screening_decisions:
        if decision.route_key not in trace.route_dictionary:
            return False
        if decision.status not in allowed_statuses:
            return False
        if decision.status == "pass":
            if decision.exact_call_blocked or decision.negative_cache_hit:
                return False
            if any(check.status == "fail" for check in decision.checks):
                return False
        elif decision.status == "rejected":
            if not decision.exact_call_blocked or decision.negative_cache_hit:
                return False
            if not decision.first_failed_check or not any(
                check.check == decision.first_failed_check and check.status == "fail"
                for check in decision.checks
            ):
                return False
        else:
            if not decision.exact_call_blocked or not decision.negative_cache_hit:
                return False
            if not any(
                check.check == "negative_sequence_cache" and check.status == "hit"
                for check in decision.checks
            ):
                return False
            if not any(
                prior.status == "rejected"
                and not prior.negative_cache_hit
                and prior.completed_at <= decision.started_at + 1e-9
                for prior in decisions_by_route[decision.route_key]
            ):
                return False
        if not decision.checks:
            return False
        decisions_by_key[(decision.lane, decision.route_key)].append(decision)
        decisions_by_route[decision.route_key].append(decision)
        if decision.status != "negative_cache_hit" and decision.negative_cache_hit:
            return False

    for evaluation in trace.route_evaluations:
        if evaluation.kind not in {"exact_call", "cache_hit"}:
            continue
        evaluations_by_key[(evaluation.lane, evaluation.route_key)].append(evaluation)
        evaluations_by_route[evaluation.route_key].append(evaluation)
        matching_pass = any(
            decision.status == "pass"
            and decision.completed_at <= evaluation.started_at + 1e-9
            for decision in decisions_by_key[(evaluation.lane, evaluation.route_key)]
        )
        if not matching_pass:
            return False

    for decision in trace.screening_decisions:
        if not decision.exact_call_blocked:
            continue
        later_exact = any(
            evaluation.started_at >= decision.completed_at - 1e-9
            for evaluation in evaluations_by_route[decision.route_key]
        )
        if later_exact:
            return False
    return True


def _independent_station_reachability(instance: Any) -> dict[str, object]:
    """Recompute optimistic depot/station bitsets without the production index."""

    safe_nodes = (instance.depot, *instance.stations)
    safe_index = {node.name: index for index, node in enumerate(safe_nodes)}
    capacity = instance.vehicle.battery_capacity
    rate = instance.vehicle.consumption_rate

    def direct(left: Any, right: Any) -> bool:
        return bool(left.distance_to(right) * rate <= capacity + 1e-9)

    safe_bitsets: dict[str, int] = {}
    for origin in safe_nodes:
        mask = 0
        frontier = [origin]
        visited = {origin.name}
        while frontier:
            current = frontier.pop()
            for destination in safe_nodes:
                if destination.name in visited or not direct(current, destination):
                    continue
                visited.add(destination.name)
                mask |= 1 << safe_index[destination.name]
                frontier.append(destination)
        mask |= 1 << safe_index[origin.name]
        safe_bitsets[origin.name] = mask

    origin_bitsets: dict[str, int] = {}
    for origin in instance.nodes:
        if origin.name in safe_bitsets:
            origin_bitsets[origin.name] = safe_bitsets[origin.name]
            continue
        mask = sum(
            1 << index
            for index, destination in enumerate(safe_nodes)
            if direct(origin, destination)
        )
        changed = True
        while changed:
            changed = False
            for index, destination in enumerate(safe_nodes):
                if mask & (1 << index):
                    expanded = mask | safe_bitsets[destination.name]
                    if expanded != mask:
                        mask = expanded
                        changed = True
        origin_bitsets[origin.name] = mask
    return {
        "safe_nodes": [node.name for node in safe_nodes],
        "bitsets": dict(sorted(safe_bitsets.items())),
        "origin_bitsets": dict(sorted(origin_bitsets.items())),
    }


def _independent_energy_reachable(instance: Any, origin_name: str, destination_name: str) -> bool:
    """Check the optimistic recharge frontier independently of the bitset index."""

    by_name = instance.by_name
    origin = by_name[origin_name]
    destination = by_name[destination_name]
    capacity = instance.vehicle.battery_capacity
    rate = instance.vehicle.consumption_rate
    safe_nodes = (instance.depot, *instance.stations)
    safe_names = {node.name for node in safe_nodes}
    frontier = [origin]
    visited = {origin.name}
    while frontier:
        current = frontier.pop()
        if current.distance_to(destination) * rate <= capacity + 1e-9:
            return True
        if current.name != origin.name and current.name not in safe_names:
            continue
        for safe_node in safe_nodes:
            if safe_node.name in visited:
                continue
            if current.distance_to(safe_node) * rate <= capacity + 1e-9:
                visited.add(safe_node.name)
                frontier.append(safe_node)
    return False


def _cache_result_fingerprint(evaluation: Any) -> tuple[object, ...]:
    return (
        evaluation.feasible,
        evaluation.failure_reason,
        evaluation.labels_generated,
        evaluation.labels_expanded,
        evaluation.labels_pruned,
    )


def _cache_incremental_trace_ok(
    trace: Stage03Trace,
    instance: Any,
    solver_result: SimpleNamespace | None,
) -> bool:
    config = trace.cache_incremental_config
    if config is None or not bool(getattr(config, "enabled", False)):
        return False
    if solver_result is None:
        return False
    expected = getattr(solver_result, "cache_incremental_statistics", {})
    if not isinstance(expected, dict):
        return False
    observed = trace.cache_incremental_counts
    for field_name in (
        "cache_lookups",
        "cache_hits",
        "cache_misses",
        "cache_stores",
        "cache_evictions",
        "cache_oversize_not_cached",
        "incremental_propagations",
        "incremental_fallbacks",
    ):
        if int(observed.get(field_name, 0)) != int(expected.get(field_name, 0)):
            return False

    # The route dictionary is the only source of customer sequences used by
    # replay.  Recompute every digest from the four-part cache key rather than
    # trusting a summary field.
    for evaluation in trace.route_evaluations:
        sequence = trace.route_dictionary.get(evaluation.route_key)
        if sequence is None or not evaluation.cache_key_digest:
            return False
        key = RouteCacheKey(
            str(config.instance_hash),
            sequence,
            str(config.charging_configuration_version),
            str(config.objective_schema_version),
        )
        if evaluation.cache_key_digest != key.digest:
            return False
        if evaluation.kind == "precomputed_route":
            if evaluation.route_change_status != "unchanged":
                return False
        elif evaluation.route_change_status == "unchanged":
            return False

    cache_events = [
        event for event in trace.events if event.get("event_type") == "cache_event"
    ]
    allowed_operations = {
        "lookup",
        "hit",
        "miss",
        "store",
        "evict",
        "oversize_not_cached",
    }
    if any(str(event.get("operation")) not in allowed_operations for event in cache_events):
        return False
    for event in cache_events:
        route_key = str(event.get("route_key", ""))
        sequence = trace.route_dictionary.get(route_key)
        if sequence is None:
            return False
        key = RouteCacheKey(
            str(config.instance_hash),
            sequence,
            str(config.charging_configuration_version),
            str(config.objective_schema_version),
        )
        if str(event.get("cache_key_digest", "")) != key.digest:
            return False
    if len([event for event in cache_events if event.get("operation") == "lookup"]) != int(
        observed["cache_lookups"]
    ):
        return False

    # Replay the bounded LRU state independently.  This verifies that a hit
    # has a preceding store, that an infeasible result is not silently changed
    # between exact and hit, and that an exact re-evaluation is possible only
    # after eviction or an oversize-not-cached decision.
    exact_by_digest: defaultdict[str, list[Any]] = defaultdict(list)
    hit_by_digest: defaultdict[str, list[Any]] = defaultdict(list)
    for evaluation in trace.route_evaluations:
        if evaluation.kind == "exact_call" and evaluation.exact_completed:
            exact_by_digest[evaluation.cache_key_digest].append(evaluation)
        elif evaluation.kind == "cache_hit":
            hit_by_digest[evaluation.cache_key_digest].append(evaluation)
    exact_cursor: defaultdict[str, int] = defaultdict(int)
    cache_state: OrderedDict[str, tuple[Any, int]] = OrderedDict()
    replayed_hits: defaultdict[str, list[tuple[object, ...]]] = defaultdict(list)
    for event in cache_events:
        operation = str(event.get("operation"))
        digest = str(event.get("cache_key_digest", ""))
        current_bytes = sum(entry[1] for entry in cache_state.values())
        if operation == "lookup":
            if _int_value(event.get("current_entries", -1)) != len(cache_state):
                return False
            if _int_value(event.get("current_bytes", -1)) != current_bytes:
                return False
            if digest in cache_state:
                cache_state.move_to_end(digest)
        elif operation == "miss":
            if digest in cache_state:
                return False
        elif operation == "hit":
            if digest not in cache_state:
                return False
            replayed_hits[digest].append(_cache_result_fingerprint(cache_state[digest][0]))
            cache_state.move_to_end(digest)
        elif operation == "evict":
            if digest not in cache_state:
                return False
            del cache_state[digest]
        elif operation == "oversize_not_cached":
            if _int_value(event.get("entry_bytes", 0)) <= config.max_memory_bytes:
                return False
            if digest in cache_state:
                return False
        elif operation == "store":
            if digest in cache_state:
                return False
            candidates = exact_by_digest[digest]
            cursor = exact_cursor[digest]
            if cursor >= len(candidates):
                return False
            exact_cursor[digest] += 1
            entry_bytes = _int_value(event.get("entry_bytes", 0))
            if entry_bytes <= 0 or entry_bytes > config.max_memory_bytes:
                return False
            cache_state[digest] = (candidates[cursor], entry_bytes)
            if _int_value(event.get("current_entries", -1)) != len(cache_state):
                return False
            expected_bytes = sum(entry[1] for entry in cache_state.values())
            if _int_value(event.get("current_bytes", -1)) != expected_bytes:
                return False
    if sum(exact_cursor.values()) != sum(
        1 for event in cache_events if event.get("operation") == "store"
    ):
        return False
    observed_hit_fingerprints = {
        digest: [_cache_result_fingerprint(item) for item in values]
        for digest, values in hit_by_digest.items()
    }
    if dict(replayed_hits) != observed_hit_fingerprints:
        return False
    expected_entries = int(expected.get("entries_current", len(cache_state)))
    if len(cache_state) != expected_entries:
        return False

    screening_by_lane_route: defaultdict[tuple[str, str], list[Any]] = defaultdict(list)
    for decision in trace.screening_decisions:
        screening_by_lane_route[(decision.lane, decision.route_key)].append(decision)
    for evaluation in trace.route_evaluations:
        if evaluation.kind not in {"exact_call", "cache_hit"}:
            continue
        prior_pass = any(
            decision.status == "pass"
            and decision.completed_at <= evaluation.started_at + 1e-9
            for decision in screening_by_lane_route[(evaluation.lane, evaluation.route_key)]
        )
        if not prior_pass:
            return False

    expected_reachability = expected.get("station_reachability", {})
    if not isinstance(expected_reachability, dict):
        return False
    independent_reachability = _independent_station_reachability(instance)
    if (
        expected_reachability.get("safe_nodes")
        != independent_reachability.get("safe_nodes")
        or expected_reachability.get("bitsets")
        != independent_reachability.get("bitsets")
        or (
            "origin_bitsets" in expected_reachability
            and expected_reachability.get("origin_bitsets")
            != independent_reachability.get("origin_bitsets")
        )
    ):
        return False
    for decision in trace.screening_decisions:
        energy_check = next(
            (
                check
                for check in decision.checks
                if check.check == "single_segment_battery_reachability"
            ),
            None,
        )
        if energy_check is None:
            continue
        sequence = trace.route_dictionary.get(decision.route_key)
        if sequence is None:
            return False
        chain = (instance.depot.name, *sequence, instance.depot.name)
        expected_energy = all(
            _independent_energy_reachable(instance, left, right)
            for left, right in zip(chain, chain[1:], strict=False)
        )
        if (energy_check.status == "pass") != expected_energy:
            return False
    for record in trace.incremental_propagations:
        base_sequence = trace.route_dictionary.get(str(record.get("base_route_key", "")))
        candidate_sequence = trace.route_dictionary.get(
            str(record.get("candidate_route_key", ""))
        )
        if base_sequence is None or candidate_sequence is None:
            return False
        full = build_route_propagation_snapshot(instance, candidate_sequence)
        candidate_is_structurally_valid = (
            all(name in instance.by_name for name in candidate_sequence)
            and len(set(candidate_sequence)) == len(candidate_sequence)
        )
        expected_status = "incremental" if candidate_is_structurally_valid else "fallback"
        if str(record.get("status")) != expected_status:
            return False
        for field_name in (
            "distance_lower_bound",
            "min_time_window_slack",
            "finish_time",
        ):
            observed_value = float(cast(Any, record.get(field_name, 0.0)))
            expected_value = float(
                getattr(full, {
                    "distance_lower_bound": "total_distance",
                    "min_time_window_slack": "min_time_window_slack",
                    "finish_time": "finish_time",
                }[field_name])
            )
            if abs(observed_value - expected_value) > 1e-7:
                return False
        if (
            abs(
                float(cast(Any, record.get("distance_lower_bound", 0.0)))
                - full.total_distance
            )
            > 1e-7
            or abs(
                float(cast(Any, record.get("min_time_window_slack", 0.0)))
                - full.min_time_window_slack
            )
            > 1e-7
            or abs(float(cast(Any, record.get("finish_time", 0.0))) - full.finish_time)
            > 1e-7
        ):
            return False
    return True


def _trace_summary(trace: Stage03Trace) -> _TraceSummary:
    return _TraceSummary(
        started_calls=trace.started_calls,
        completed_calls=trace.completed_calls,
        exact_calls=trace.exact_calls,
        cache_hits=trace.cache_hits,
        precomputed_routes=trace.precomputed_routes,
        deadline_events=trace.deadline_events,
        route_evaluation_count=len(trace.route_evaluations),
        screening_counts=dict(trace.screening_counts),
        cache_incremental_counts=dict(trace.cache_incremental_counts),
    )


def _cache_statistics_rows(
    audited: list[_AuditedRun],
    label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in audited:
        values = item.trace_summary.cache_incremental_counts
        rows.extend(
            {
                "run_label": label,
                "instance": item.key[0],
                "seed": item.key[1],
                "metric": metric,
                "value": value,
            }
            for metric, value in sorted(values.items())
        )
    return rows


def _screening_reason_rows(
    audited: list[_AuditedRun],
    label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in audited:
        counts: dict[str, int] = {}
        for reason, count in item.screening_reason_counts.items():
            counts[reason] = count
        rows.extend(
            {
                "run_label": label,
                "instance": item.key[0],
                "seed": item.key[1],
                "reason": reason,
                "count": count,
            }
            for reason, count in sorted(counts.items())
        )
    return rows


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


def _event_log_ok_stream(
    path: Path,
    trace: Stage03Trace,
    solver_result: SimpleNamespace | None,
) -> tuple[bool, int]:
    """Replay the event log without materialising millions of JSON objects."""

    trace_index = 0
    screening_index = 0
    neighborhood_index = 0
    row_count = 0
    expected_neighborhood = (
        getattr(solver_result, "neighborhood_events", ())
        if solver_result is not None
        else ()
    )
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row_count += 1
                record = json.loads(line)
                record_type = record.get("record_type")
                payload = record.get("payload", {})
                if record_type == "trace_event":
                    if trace_index >= len(trace.events):
                        return False, row_count
                    if _canonical_json(payload) != _canonical_json(trace.events[trace_index]):
                        return False, row_count
                    trace_index += 1
                elif record_type == "screening_decision":
                    if screening_index >= len(trace.screening_decisions):
                        return False, row_count
                    if _canonical_json(payload) != _canonical_json(
                        asdict(trace.screening_decisions[screening_index])
                    ):
                        return False, row_count
                    screening_index += 1
                elif record_type == "neighborhood_event":
                    if neighborhood_index >= len(expected_neighborhood):
                        return False, row_count
                    expected = expected_neighborhood[neighborhood_index]
                    if _canonical_json(payload) != _canonical_json(expected):
                        return False, row_count
                    if payload.get("accepted") is True:
                        if payload.get("candidate_feasible") is not True:
                            return False, row_count
                        if _int_value(payload.get("candidate_vehicle_delta", 0)) > 0:
                            return False, row_count
                    neighborhood_index += 1
                else:
                    return False, row_count
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False, row_count
    return (
        trace_index == len(trace.events)
        and screening_index == len(trace.screening_decisions)
        and neighborhood_index == len(expected_neighborhood),
        row_count,
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
    screening_events = [
        event.get("payload", {})
        for event in event_log
        if event.get("record_type") == "screening_decision"
    ]
    if len(trace_events) != len(trace.events):
        return False
    if any(
        _canonical_json(left) != _canonical_json(right)
        for left, right in zip(trace_events, trace.events, strict=True)
    ):
        return False
    if screening_events:
        if len(screening_events) != len(trace.screening_decisions):
            return False
        if any(
            _canonical_json(payload) != _canonical_json(asdict(decision))
            for payload, decision in zip(
                screening_events, trace.screening_decisions, strict=True
            )
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
    event_log_rows: int,
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
        "event_log_rows": event_log_rows,
    }


def _provenance_ok(root: Path, run_dir: Path, metadata: dict[str, Any]) -> tuple[bool, str]:
    if metadata.get("repository_dirty") is not False:
        return False, f"repository_dirty={metadata.get('repository_dirty')}"
    screening_payload = metadata.get("screening_config")
    screening_enabled = isinstance(screening_payload, dict) and bool(
        screening_payload.get("enabled", False)
    )
    cache_payload = metadata.get("cache_incremental_config")
    cache_incremental_enabled = isinstance(cache_payload, dict) and bool(
        cache_payload.get("enabled", False)
    )
    try:
        current_sources = _source_hashes(
            root,
            include_stage031=screening_enabled,
            include_stage032=cache_incremental_enabled,
        )
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


def _load_stage03_formal_rows(
    root: Path,
    metadata: dict[str, Any],
) -> dict[tuple[str, int], dict[str, object]]:
    cache_payload = metadata.get("cache_incremental_config")
    cache_incremental_enabled = isinstance(cache_payload, dict) and bool(
        cache_payload.get("enabled", False)
    )
    raw_path_value = (
        metadata.get("stage031_formal_per_run")
        if cache_incremental_enabled
        else metadata.get("stage03_formal_per_run")
    )
    if raw_path_value in {None, ""}:
        return {}
    path = Path(str(raw_path_value))
    if not path.is_absolute():
        path = root / path
    expected_path_hash = str(
        metadata.get(
            "stage031_formal_per_run_sha256"
            if cache_incremental_enabled
            else "stage03_formal_per_run_sha256",
            "",
        )
    )
    if not path.is_file() or not expected_path_hash:
        raise RuntimeError("Stage 3.0 formal per-run baseline is missing")
    if _sha256(path) != expected_path_hash:
        raise RuntimeError("Stage 3.0 formal per-run baseline hash mismatch")
    if cache_incremental_enabled:
        _verify_trusted_stage031_review_manifest(root, metadata)
    else:
        _verify_trusted_stage03_review_manifest(root, metadata)
    rows: dict[tuple[str, int], dict[str, object]] = {}
    for row in _read_csv(path):
        objective = json.loads(str(row.get("objective_key", "[]")))
        rows[(str(row["instance"]), int(row["seed"]))] = {
            "objective_key": tuple(objective),
            "feasible": str(row.get("feasible", "False")) == "True",
            "vehicle_count": _int_value(row.get("vehicle_count", "")),
            "exact_calls": _int_value(row.get("trace_exact_calls", "")),
        }
    return rows


def _verify_trusted_stage03_review_manifest(
    root: Path,
    metadata: dict[str, Any],
) -> None:
    raw_path_value = metadata.get("stage03_formal_review_manifest")
    expected_hash = str(metadata.get("stage03_formal_review_manifest_sha256", ""))
    if raw_path_value in {None, ""} or not expected_hash:
        raise RuntimeError("trusted Stage 3.0 formal review manifest is missing")
    review_manifest = Path(str(raw_path_value))
    if not review_manifest.is_absolute():
        review_manifest = root / review_manifest
    if not review_manifest.is_file():
        raise RuntimeError("trusted Stage 3.0 formal review manifest is missing")
    if _sha256(review_manifest) != expected_hash:
        raise RuntimeError("trusted Stage 3.0 formal review manifest hash mismatch")
    payload = _read_json(review_manifest)
    recorded_hash = str(payload.get("manifest_payload_sha256", ""))
    payload_without_hash = dict(payload)
    payload_without_hash.pop("manifest_payload_sha256", None)
    if not recorded_hash or recorded_hash != _payload_sha256(payload_without_hash):
        raise RuntimeError("trusted Stage 3.0 formal review manifest payload mismatch")
    if payload.get("scope") != "formal" or payload.get("status") not in {
        "READY_FOR_STAGE03_ACCELERATION",
        "READY_FOR_STAGE03_1",
        "READY_FOR_STAGE31",
    }:
        raise RuntimeError("trusted Stage 3.0 formal review is not ready")
    review_label = str(payload.get("review_label", ""))
    for name, expected_file_hash in dict(payload.get("files", {})).items():
        artifact = review_manifest.parent / str(name)
        if not artifact.is_file() and review_label:
            artifact = review_manifest.parent / f"{review_label}_{name}"
        if (
            not artifact.is_file()
            and name == "recomputed_per_run_results.csv"
            and review_label
        ):
            artifact = review_manifest.parent / f"{review_label}_per_run_results.csv"
        if not artifact.is_file() or _sha256(artifact) != str(expected_file_hash):
            raise RuntimeError(f"trusted Stage 3.0 review artifact hash mismatch: {name}")


def _verify_trusted_stage031_review_manifest(
    root: Path,
    metadata: dict[str, Any],
) -> None:
    raw_path_value = metadata.get("stage031_formal_review_manifest")
    expected_hash = str(metadata.get("stage031_formal_review_manifest_sha256", ""))
    if raw_path_value in {None, ""} or not expected_hash:
        raise RuntimeError("trusted Stage 3.1 formal review manifest is missing")
    review_manifest = Path(str(raw_path_value))
    if not review_manifest.is_absolute():
        review_manifest = root / review_manifest
    if not review_manifest.is_file() or _sha256(review_manifest) != expected_hash:
        raise RuntimeError("trusted Stage 3.1 formal review manifest hash mismatch")
    payload = _read_json(review_manifest)
    recorded_hash = str(payload.get("manifest_payload_sha256", ""))
    payload_without_hash = dict(payload)
    payload_without_hash.pop("manifest_payload_sha256", None)
    if not recorded_hash or recorded_hash != _payload_sha256(payload_without_hash):
        raise RuntimeError("trusted Stage 3.1 review manifest payload mismatch")
    if payload.get("scope") != "formal" or payload.get("status") != READY_FOR_STAGE032:
        raise RuntimeError("trusted Stage 3.1 formal review is not READY_FOR_STAGE03_2")
    review_label = str(payload.get("review_label", ""))
    for name, expected_file_hash in dict(payload.get("files", {})).items():
        artifact = review_manifest.parent / str(name)
        if not artifact.is_file() and review_label:
            artifact = review_manifest.parent / f"{review_label}_{name}"
        if (
            not artifact.is_file()
            and name == "recomputed_per_run_results.csv"
            and review_label
        ):
            artifact = review_manifest.parent / f"{review_label}_per_run_results.csv"
        if not artifact.is_file() or _sha256(artifact) != str(expected_file_hash):
            raise RuntimeError(f"trusted Stage 3.1 review artifact hash mismatch: {name}")


def _compare_stage03_formal(
    audited: list[_AuditedRun],
    baselines: dict[tuple[str, int], dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in audited:
        baseline = baselines.get(item.key)
        candidate_key = item.objective.key if item.objective is not None else ()
        candidate_feasible = item.validator_feasible and item.objective is not None
        candidate_vehicle = item.objective.vehicle_count if item.objective is not None else ""
        candidate_exact = item.trace_summary.exact_calls
        if baseline is None:
            rows.append(
                {
                    "instance": item.key[0],
                    "seed": item.key[1],
                    "candidate_objective_key": json.dumps(candidate_key),
                    "stage03_objective_key": "[]",
                    "candidate_feasible": candidate_feasible,
                    "stage03_feasible": False,
                    "candidate_vehicle_count": candidate_vehicle,
                    "stage03_vehicle_count": "",
                    "candidate_exact_calls": candidate_exact,
                    "stage03_exact_calls": "",
                    "exact_call_delta": "",
                    "classification": "missing_stage03_baseline",
                    "gate_status": "fail",
                    "reason": "Stage 3.0 formal per-run baseline is missing",
                }
            )
            continue
        baseline_key = _objective_from_key(baseline["objective_key"])
        stage03_feasible = bool(baseline["feasible"])
        stage03_vehicle = int(cast(Any, baseline["vehicle_count"]))
        stage03_exact = int(cast(Any, baseline["exact_calls"]))
        c5 = item.key[0] in {"c101C5", "r105C5", "rc105C5"}
        focused = item.key[0] in {"c101_21", "r101_21", "rc101_21"}
        if baseline_key is None or not stage03_feasible or not candidate_feasible:
            gate = "fail"
            classification = "infeasible_or_missing_objective"
            reason = "both Stage 3.0 and Stage 3.1 formal runs must replay feasible"
        elif c5:
            if item.objective is None:
                raise RuntimeError("candidate feasibility/objective state diverged")
            gate = (
                "pass"
                if compare_objectives(item.objective, baseline_key)
                is not ObjectiveComparison.WORSE
                else "fail"
            )
            classification = "objective_not_worse" if gate == "pass" else "objective_regression"
            reason = "C5 objective key must not be worse than Stage 3.0 formal evidence"
        elif focused:
            if item.objective is None:
                raise RuntimeError("candidate feasibility/objective state diverged")
            gate = "pass" if item.objective.vehicle_count <= stage03_vehicle else "fail"
            classification = "vehicle_guard_pass" if gate == "pass" else "vehicle_regression"
            reason = "100-customer vehicle count must not increase; time is budget variation"
        else:
            gate = "pass"
            classification = "time_budget_variation"
            reason = "non-gated wall-clock/objective variation is recorded, not acceleration"
        rows.append(
            {
                "instance": item.key[0],
                "seed": item.key[1],
                "candidate_objective_key": json.dumps(candidate_key),
                "stage03_objective_key": json.dumps(baseline["objective_key"]),
                "candidate_feasible": candidate_feasible,
                "stage03_feasible": stage03_feasible,
                "candidate_vehicle_count": candidate_vehicle,
                "stage03_vehicle_count": stage03_vehicle,
                "candidate_exact_calls": candidate_exact,
                "stage03_exact_calls": stage03_exact,
                "exact_call_delta": candidate_exact - stage03_exact,
                "classification": classification,
                "gate_status": gate,
                "reason": reason,
            }
        )
    return rows


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
            "trace_started_calls": item.trace_summary.started_calls,
            "trace_completed_calls": item.trace_summary.completed_calls,
            "trace_exact_calls": item.trace_summary.exact_calls,
            "trace_cache_hits": item.trace_summary.cache_hits,
            "trace_precomputed_routes": item.trace_summary.precomputed_routes,
            "trace_route_evaluations": item.trace_summary.route_evaluation_count,
            "trace_deadline_events": item.trace_summary.deadline_events,
            "trace_screening_calls": item.trace_summary.screening_counts["screening_calls"],
            "trace_screening_passes": item.trace_summary.screening_counts["screening_passes"],
            "trace_screening_rejections": item.trace_summary.screening_counts[
                "screening_rejections"
            ],
            "trace_screening_cache_hits": item.trace_summary.screening_counts[
                "screening_cache_hits"
            ],
            "trace_screening_exact_call_blocked": item.trace_summary.screening_counts[
                "screening_exact_call_blocked"
            ],
            "trace_screening_reason_counts": json.dumps(
                item.trace_summary.screening_counts["screening_reason_counts"], sort_keys=True
            ),
            "trace_cache_incremental_counts": json.dumps(
                item.trace_summary.cache_incremental_counts, sort_keys=True
            ),
            "trace_incremental_propagations": item.trace_summary.cache_incremental_counts[
                "incremental_propagations"
            ],
            "trace_incremental_fallbacks": item.trace_summary.cache_incremental_counts[
                "incremental_fallbacks"
            ],
            "trace_reconciliation_status": item.reconciliation.get("status", "not_available"),
            **solver_metrics,
            "peak_tracemalloc_bytes": environment.get("peak_tracemalloc_bytes", ""),
            "peak_rss_bytes": environment.get("peak_rss_bytes", ""),
            "raw_path": raw.get("raw_path", ""),
            "solution_path": raw.get("solution_path", ""),
            "trace_path": raw.get("trace_path", ""),
            "event_path": raw.get("event_path", ""),
            "environment_path": raw.get("environment_path", ""),
            "failure_path": raw.get("failure_path", ""),
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
        screening_calls = [
            int(row.get("trace_screening_calls", 0) or 0) for row in group
        ]
        screening_rejections = [
            int(row.get("trace_screening_rejections", 0) or 0) for row in group
        ]
        screening_cache_hits = [
            int(row.get("trace_screening_cache_hits", 0) or 0) for row in group
        ]
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
                "mean_screening_calls": _mean(screening_calls),
                "median_screening_calls": _median(screening_calls),
                "mean_screening_rejections": _mean(screening_rejections),
                "median_screening_rejections": _median(screening_rejections),
                "mean_screening_cache_hits": _mean(screening_cache_hits),
                "median_screening_cache_hits": _median(screening_cache_hits),
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
        "started_calls": item.trace_summary.started_calls,
        "completed_calls": item.trace_summary.completed_calls,
        "exact_calls": item.trace_summary.exact_calls,
        "cache_hits": item.trace_summary.cache_hits,
        "precomputed_routes": item.trace_summary.precomputed_routes,
        "deadline_events": item.trace_summary.deadline_events,
        "screening_calls": item.trace_summary.screening_counts["screening_calls"],
        "screening_passes": item.trace_summary.screening_counts["screening_passes"],
        "screening_rejections": item.trace_summary.screening_counts["screening_rejections"],
        "screening_cache_hits": item.trace_summary.screening_counts["screening_cache_hits"],
        "screening_exact_call_blocked": item.trace_summary.screening_counts[
            "screening_exact_call_blocked"
        ],
        "screening_reason_counts": json.dumps(
            item.trace_summary.screening_counts["screening_reason_counts"], sort_keys=True
        ),
        "cache_incremental_counts": json.dumps(
            item.trace_summary.cache_incremental_counts, sort_keys=True
        ),
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
    screening = any(row["finding"] == "cheap_screening_reconciliation" for row in findings)
    cache_incremental = any(
        row["finding"] == "cache_incremental_reconciliation" for row in findings
    )
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
            (
                "Stage 3.2 audits bounded route caching and incremental propagation. "
                "Cache evidence is not a Stage 3.3 fixed-work/wall-clock acceleration claim."
                if cache_incremental
                else "Stage 3.1 measures safe cheap screening and exact-call reduction only. "
                "It does not claim complete route caching, incremental propagation, "
                "interruptible exact solving, parallel evaluation, or fixed-work/wall-clock "
                "acceleration."
                if screening
                else "Stage 3.0 measures exact charging calls and replayability only. "
                "It does not claim cache acceleration, incremental propagation, "
                "interruptible exact solving, parallel evaluation, or fixed-work/wall-clock "
                "improvement."
            ),
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
    if "screening_reason_statistics" in output_paths:
        names["screening_reason_statistics"] = f"{label}_screening_reason_statistics.csv"
    if "stage03_formal_comparison" in output_paths:
        names["stage03_formal_comparison"] = f"{label}_stage03_formal_comparison.csv"
    if "cache_incremental_statistics" in output_paths:
        names["cache_incremental_statistics"] = (
            f"{label}_cache_incremental_statistics.csv"
        )
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
    parser = argparse.ArgumentParser(description="Replay-audit Stage 3 raw evidence")
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
    return 0 if status in {
        READY_FOR_FORMAL,
        READY_FOR_STAGE31,
        READY_FOR_STAGE031_FORMAL,
        READY_FOR_STAGE032,
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
