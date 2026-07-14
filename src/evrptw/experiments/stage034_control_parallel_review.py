"""Independent replay and Stage 4 readiness gates for Stage 3.4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evrptw.artifacts import ArtifactReader, verify_manifest
from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES, FORMAL_SEEDS
from evrptw.experiments.stage03_measurement import SMOKE_INSTANCES
from evrptw.experiments.stage033_exact_deadline_review import (
    _artifact_groups,
    _event_axis,
    _verify_frozen_cpu_batch_evidence,
)
from evrptw.experiments.stage034_control_parallel import DIAGNOSTIC_AXES
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.validation import validate_routes

READY_FOR_STAGE034_FORMAL = "READY_FOR_STAGE034_FORMAL"
READY_FOR_STAGE04 = "READY_FOR_STAGE04"
FOCUSED_R_RC = ("r101_21", "rc101_21")


def evaluate_stage034_gate(
    rows: Sequence[Mapping[str, object]],
    *,
    scope: str,
    prerequisites_valid: bool,
) -> dict[str, object]:
    if scope not in {"smoke", "formal"}:
        raise ValueError("Stage 3.4 scope must be smoke or formal")
    instances = SMOKE_INSTANCES if scope == "smoke" else FORMAL_INSTANCES
    expected = {
        (instance, seed, axis)
        for instance in instances
        for seed in FORMAL_SEEDS
        for axis in DIAGNOSTIC_AXES
    }
    observed: set[tuple[str, int, str]] = set()
    rows_valid = True
    quality_gate = True
    semantics_gate = True
    for row in rows:
        key = (
            str(row.get("instance")),
            _as_int(row.get("seed")),
            str(row.get("axis")),
        )
        if key in observed:
            rows_valid = False
        observed.add(key)
        rows_valid = rows_valid and all(
            (
                row.get("backend") == "cpu_batch",
                bool(row.get("valid")),
                0 <= _as_int(row.get("completed_calls"))
                <= _as_int(row.get("started_calls")),
                _as_int(row.get("unchanged_exact_calls")) == 0,
                bool(row.get("exact_reconciliation_valid")),
            )
        )
        if key[2].endswith("wall_clock"):
            quality_gate = quality_gate and bool(row.get("objective_not_worse"))
        semantics_gate = semantics_gate and all(
            (
                bool(row.get("candidate_reconciliation_valid")),
                bool(row.get("parallel_ordering_valid")),
                bool(row.get("paired_semantics_valid")),
            )
        )
    complete = observed == expected and len(rows) == len(expected)
    performance_gate = True
    for instance in FOCUSED_R_RC:
        for worker in ("serial", "parallel"):
            wall_values = [
                _as_int(row.get("started_calls"))
                for row in rows
                if row.get("instance") == instance
                and row.get("axis") == f"{worker}_wall_clock"
            ]
            fixed_values = [
                _as_int(row.get("effective_iterations"))
                for row in rows
                if row.get("instance") == instance
                and row.get("axis") == f"{worker}_fixed_exact_calls"
            ]
            performance_gate = performance_gate and all(
                (
                    len(wall_values) == len(FORMAL_SEEDS),
                    len(fixed_values) == len(FORMAL_SEEDS),
                    bool(wall_values) and statistics.median(wall_values) <= 100,
                    bool(fixed_values) and statistics.median(fixed_values) >= 50,
                )
            )
    ready = all(
        (
            complete,
            rows_valid,
            quality_gate,
            semantics_gate,
            performance_gate,
            prerequisites_valid,
        )
    )
    return {
        "schema_version": "stage034-review-v1",
        "scope": scope,
        "status": (
            READY_FOR_STAGE034_FORMAL
            if ready and scope == "smoke"
            else READY_FOR_STAGE04
            if ready
            else "NOT_READY"
        ),
        "scope_complete": complete,
        "all_rows_valid": rows_valid,
        "quality_gate": quality_gate,
        "semantics_gate": semantics_gate,
        "performance_gate": performance_gate,
        "prerequisites_valid": prerequisites_valid,
        "expected_rows": len(expected),
        "observed_rows": len(rows),
    }


def review_stage034(
    *,
    run_dir: Path,
    scope: str,
    benchmark_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    manifest = verify_manifest(run_dir)
    if manifest.get("stage_id") != "stage03.4" or manifest.get("component") != "control_parallel":
        raise RuntimeError("artifact bundle is not Stage 3.4 control_parallel evidence")
    if manifest.get("status") != "complete" or manifest.get("evidence_completeness") != "complete":
        raise RuntimeError("partial Stage 3.4 evidence cannot become ready")
    reader = ArtifactReader(run_dir)
    metadata_ref = next(
        item
        for item in manifest["artifacts"]
        if item.get("artifact_type") == "manifest_metadata"
    )
    control = reader.read_json(str(metadata_ref["relative_path"]))
    if not all(
        (
            control.get("scope") == scope,
            control.get("exact_backend") == "cpu_batch",
            control.get("repository_dirty") is False,
            control.get("diagnostic_axes") == list(DIAGNOSTIC_AXES),
            control.get("worker_counts") == [1, 4],
            isinstance(control.get("candidate_control"), Mapping),
            isinstance(control.get("candidate_control_grid"), Mapping),
        )
    ):
        raise RuntimeError("Stage 3.4 control provenance is invalid")
    baseline = _load_stage033_baseline(run_dir.parents[1])
    rows: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    ordering_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    objective_rows: list[dict[str, object]] = []
    screening_rows: list[dict[str, object]] = []
    cache_rows: list[dict[str, object]] = []
    backend_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    deadline_rows: list[dict[str, object]] = []
    candidate_configuration = _object(
        control.get("candidate_control"), "candidate-control configuration"
    )
    max_round_budget = _as_int(
        candidate_configuration.get("max_exact_calls_per_round")
    )
    grouped = _artifact_groups(manifest)
    for (instance_name, seed), artifacts in sorted(grouped.items()):
        raw = reader.read_json(artifacts["raw"])
        solution = reader.read_json(artifacts["solution"])
        events = reader.read_events(artifacts["events"])
        environment = reader.read_json(artifacts["environment"])
        route_dictionary = reader.read_parquet(artifacts["route_dictionary"])
        instance = parse_schneider(benchmark_dir / f"{instance_name}.txt")
        raw_axes = _object(raw.get("axes"), "raw axes")
        solution_axes = _object(solution.get("axes"), "solution axes")
        route_ids = {int(item["route_id"]) for item in route_dictionary}
        route_references_valid = all(
            event.get(field) is None or int(event[field]) in route_ids
            for event in events
            for field in ("route_id", "base_route_id", "candidate_route_id")
        )
        axis_rows: dict[str, dict[str, object]] = {}
        for axis in DIAGNOSTIC_AXES:
            raw_axis = _object(raw_axes.get(axis), f"{axis} raw")
            solution_axis = _object(solution_axes.get(axis), f"{axis} solution")
            routes = _routes(solution_axis.get("routes"))
            validation = validate_routes(instance, routes)
            objective = (
                SolutionObjective.from_report(instance, validation)
                if validation.feasible
                else None
            )
            objective_key = solution_axis.get("objective_key")
            objective_matches = (
                objective is not None
                and isinstance(objective_key, list)
                and tuple(objective_key) == objective.key
                and raw_axis.get("objective_key") == objective_key
            )
            axis_events = [event for event in events if _event_axis(event) == axis]
            decoded_events = [_decoded_event(event) for event in axis_events]
            unchanged_exact = sum(
                event.get("record_type") == "route_evaluation"
                and bool(event.get("exact_started"))
                and event.get("route_change_status") == "unchanged"
                for event in axis_events
            )
            event_started = sum(
                event.get("record_type") == "route_evaluation"
                and bool(event.get("exact_started"))
                for event in axis_events
            )
            event_completed = sum(
                event.get("record_type") == "route_evaluation"
                and bool(event.get("exact_completed"))
                for event in axis_events
            )
            exact_reconciliation_valid = (
                event_started == _as_int(raw_axis.get("started_calls"))
                and event_completed == _as_int(raw_axis.get("completed_calls"))
            )
            candidate_valid, candidate_details = _candidate_control_valid(
                axis_events,
                max_round_budget=max_round_budget,
            )
            ordering_valid, ordering_details = _parallel_ordering_valid(axis_events, axis)
            baseline_key = baseline.get((instance_name, seed))
            objective_not_worse = (
                True
                if not axis.endswith("wall_clock")
                else objective is not None
                and baseline_key is not None
                and objective.key <= baseline_key
            )
            backend = str(raw_axis.get("backend"))
            valid = all(
                (
                    validation.feasible,
                    objective_matches,
                    route_references_valid,
                    backend == "cpu_batch",
                    bool(raw_axis.get("valid")),
                    candidate_valid,
                    ordering_valid,
                    exact_reconciliation_valid,
                    unchanged_exact == 0,
                    environment.get("repository_dirty") is False,
                    environment.get("exact_backend") == "cpu_batch",
                )
            )
            row = {
                "instance": instance_name,
                "seed": seed,
                "axis": axis,
                "backend": backend,
                "valid": valid,
                "objective_key": json.dumps(objective_key, separators=(",", ":")),
                "objective_not_worse": objective_not_worse,
                "started_calls": _as_int(raw_axis.get("started_calls")),
                "completed_calls": _as_int(raw_axis.get("completed_calls")),
                "effective_iterations": _as_int(raw_axis.get("effective_iterations")),
                "unchanged_exact_calls": unchanged_exact,
                "candidate_reconciliation_valid": candidate_valid,
                "exact_reconciliation_valid": exact_reconciliation_valid,
                "parallel_ordering_valid": ordering_valid,
                "paired_semantics_valid": False,
                "candidate_work_hash": str(raw_axis.get("candidate_work_hash") or ""),
                "route_result_hash": str(raw_axis.get("route_result_hash") or ""),
                "acceptance_trajectory_hash": _trajectory_hash(axis_events),
                "termination_reason": str(raw_axis.get("termination_reason") or ""),
            }
            rows.append(row)
            axis_rows[axis] = row
            control_rows.append(
                {
                    "instance": instance_name,
                    "seed": seed,
                    "axis": axis,
                    **candidate_details,
                }
            )
            ordering_rows.append(
                {
                    "instance": instance_name,
                    "seed": seed,
                    "axis": axis,
                    **ordering_details,
                }
            )
            objective_rows.append(
                {
                    "instance": instance_name,
                    "seed": seed,
                    "axis": axis,
                    "objective_key": row["objective_key"],
                    "stage033_wall_clock_objective": json.dumps(
                        baseline_key, separators=(",", ":")
                    ) if baseline_key is not None else "",
                    "not_worse": objective_not_worse,
                }
            )
            identity = {"instance": instance_name, "seed": seed, "axis": axis}
            screening_counts = Counter(
                (str(event.get("status") or ""), str(event.get("reason") or ""))
                for event in decoded_events
                if event.get("record_type") == "screening_decision"
            )
            for event in decoded_events:
                if event.get("status") != "prefilter_rejected_aggregate":
                    continue
                key = (str(event.get("status")), str(event.get("reason") or ""))
                screening_counts[key] += _as_int(event.get("aggregate_count") or 1)
            screening_rows.extend(
                {
                    **identity,
                    "status": status,
                    "reason": reason,
                    "count": count,
                }
                for (status, reason), count in sorted(screening_counts.items())
            )
            cache_counts = Counter(
                (
                    str(event.get("operation") or ""),
                    str(event.get("lookup_result") or ""),
                )
                for event in decoded_events
                if event.get("record_type") == "cache_event"
            )
            cache_rows.extend(
                {
                    **identity,
                    "operation": operation,
                    "lookup_result": lookup_result,
                    "count": count,
                }
                for (operation, lookup_result), count in sorted(cache_counts.items())
            )
            backend_metrics = _object(
                raw_axis.get("exact_deadline_statistics") or {},
                "exact deadline statistics",
            )
            backend_rows.append(
                {
                    **identity,
                    "backend": backend,
                    **{
                        str(key): value
                        for key, value in backend_metrics.items()
                        if isinstance(value, (str, int, float, bool)) or value is None
                    },
                }
            )
            runtime_seconds = float(raw_axis.get("runtime_seconds") or 0.0)
            profile_rows.append(
                {
                    **identity,
                    "runtime_seconds": runtime_seconds,
                    "effective_iterations": row["effective_iterations"],
                    "effective_iterations_per_second": (
                        _as_int(row["effective_iterations"]) / runtime_seconds
                        if runtime_seconds > 0.0
                        else 0.0
                    ),
                    "started_calls": row["started_calls"],
                    "peak_rss_bytes": _as_int(environment.get("peak_rss_bytes")),
                }
            )
            deadline_counts = Counter(
                str(event.get("boundary") or event.get("status") or "")
                for event in decoded_events
                if event.get("record_type") == "deadline_boundary"
                or event.get("event_type") == "deadline_boundary"
            )
            deadline_rows.extend(
                {**identity, "boundary": boundary, "count": count}
                for boundary, count in sorted(deadline_counts.items())
            )
        for suffix in ("fixed_exact_calls", "wall_clock"):
            serial = axis_rows[f"serial_{suffix}"]
            parallel = axis_rows[f"parallel_{suffix}"]
            paired = all(
                serial[field] == parallel[field]
                for field in (
                    "objective_key",
                    "candidate_work_hash",
                    "route_result_hash",
                    "started_calls",
                    "completed_calls",
                    "effective_iterations",
                    "acceptance_trajectory_hash",
                )
            ) if suffix == "fixed_exact_calls" else True
            serial["paired_semantics_valid"] = paired
            parallel["paired_semantics_valid"] = paired
        findings.extend(
            {
                "instance": instance_name,
                "seed": seed,
                "axis": axis,
                "status": "pass" if row["valid"] else "fail",
                "reason": "independent solution/objective/candidate/parallel replay",
            }
            for axis, row in axis_rows.items()
        )
    prerequisites = _prerequisites_valid(run_dir.parents[1])
    gate = evaluate_stage034_gate(
        rows,
        scope=scope,
        prerequisites_valid=prerequisites,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_run_results": output_dir / "per_run_results.csv",
        "objective_comparison": output_dir / "objective_comparison.csv",
        "candidate_control_reconciliation": output_dir / "candidate_control_reconciliation.csv",
        "parallel_ordering": output_dir / "parallel_ordering.csv",
        "fixed_work_wall_clock": output_dir / "fixed_work_wall_clock.csv",
        "screening_reason_statistics": output_dir / "screening_reason_statistics.csv",
        "cache_statistics": output_dir / "cache_statistics.csv",
        "backend_metrics": output_dir / "backend_metrics.csv",
        "performance_profile": output_dir / "performance_profile.csv",
        "deadline_report": output_dir / "deadline_report.csv",
        "review_findings": output_dir / "review_findings.csv",
        "stage04_readiness": output_dir / "stage04_readiness.csv",
        "review_report": output_dir / "review_report.md",
        "review_manifest": output_dir / "review_manifest.json",
    }
    _write_csv(paths["per_run_results"], rows)
    _write_csv(paths["objective_comparison"], objective_rows)
    _write_csv(paths["candidate_control_reconciliation"], control_rows)
    _write_csv(paths["parallel_ordering"], ordering_rows)
    paired_rows = _paired_rows(rows)
    _write_csv(paths["fixed_work_wall_clock"], paired_rows)
    _write_csv(paths["screening_reason_statistics"], screening_rows)
    _write_csv(paths["cache_statistics"], cache_rows)
    _write_csv(paths["backend_metrics"], backend_rows)
    _write_csv(paths["performance_profile"], profile_rows)
    _write_csv(paths["deadline_report"], deadline_rows)
    _write_csv(paths["review_findings"], findings)
    _write_csv(
        paths["stage04_readiness"],
        [{"gate": key, "value": value} for key, value in gate.items()],
    )
    paths["review_report"].write_text(
        "# Stage 3.4 control/parallel independent review\n\n"
        f"- Status: `{gate['status']}`\n"
        f"- Complete axes: `{gate['observed_rows']}/{gate['expected_rows']}`\n"
        f"- Strict objective gate: `{'PASS' if gate['quality_gate'] else 'FAIL'}`\n"
        f"- R/RC performance gate: `{'PASS' if gate['performance_gate'] else 'FAIL'}`\n"
        f"- Deterministic semantics gate: `{'PASS' if gate['semantics_gate'] else 'FAIL'}`\n",
        encoding="utf-8",
    )
    file_hashes = {
        path.name: _sha256(path)
        for name, path in paths.items()
        if name != "review_manifest"
    }
    paths["review_manifest"].write_text(
        json.dumps(
            {
                **gate,
                "run_label": manifest["run_label"],
                "run_directory": str(run_dir),
                "files": file_hashes,
                "reviewer_sha256": _sha256(Path(__file__)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def _candidate_control_valid(
    events: Sequence[Mapping[str, object]],
    *,
    max_round_budget: int,
) -> tuple[bool, dict[str, object]]:
    decoded = [_decoded_event(event) for event in events]
    decisions = [
        event
        for event in decoded
        if event.get("event_type")
        in {"candidate_control_decision", "candidate_control_decision_aggregate"}
    ]
    budgets = [
        event
        for event in decoded
        if event.get("event_type") == "candidate_control_budget"
    ]
    per_iteration: dict[int, int] = {}
    for event in budgets:
        iteration = event.get("iteration")
        if iteration is None:
            continue
        key = _as_int(iteration)
        per_iteration[key] = per_iteration.get(key, 0) + _as_int(event.get("granted"))
    max_granted = max(per_iteration.values(), default=0)
    valid = bool(decisions) and max_granted <= max_round_budget and all(
        event.get("status") in {"selected", "not_selected"} for event in decisions
    )
    return valid, {
        "valid": valid,
        "decisions": len(decisions),
        "selected": sum(
            _as_int(event.get("aggregate_count") or 1)
            for event in decisions
            if event.get("status") == "selected"
        ),
        "not_selected": sum(
            _as_int(event.get("aggregate_count") or 1)
            for event in decisions
            if event.get("status") == "not_selected"
        ),
        "budget_events": len(budgets),
        "maximum_granted_per_iteration": max_granted,
    }


def _parallel_ordering_valid(
    events: Sequence[Mapping[str, object]],
    axis: str,
) -> tuple[bool, dict[str, object]]:
    batches = [
        _decoded_event(event)
        for event in events
        if _decoded_event(event).get("event_type") == "parallel_batch"
    ]
    expects_parallel = axis.startswith("parallel_")
    relevant = [
        event for event in batches
        if event.get("status") == ("parallel_complete" if expects_parallel else "serial_complete")
    ]
    valid = bool(relevant) and all(
        event.get("merge_order") == event.get("submission_order")
        and _as_int(event.get("worker_count", 0)) == (4 if expects_parallel else 1)
        for event in relevant
    )
    return valid, {
        "valid": valid,
        "batch_events": len(relevant),
        "worker_count": 4 if expects_parallel else 1,
    }


def _decoded_event(event: Mapping[str, object]) -> dict[str, object]:
    output = dict(event)
    extras = event.get("extras_json")
    if isinstance(extras, str) and extras:
        decoded = json.loads(extras)
        if isinstance(decoded, dict):
            output.update(decoded)
    return output


def _trajectory_hash(events: Sequence[Mapping[str, object]]) -> str:
    trajectory = [
        {
            key: event.get(key)
            for key in (
                "lane_id",
                "iteration",
                "operator_id",
                "status",
                "accepted",
                "global_best",
                "current_vehicle_count",
                "candidate_vehicle_count",
                "current_route_ids",
                "candidate_route_ids",
            )
        }
        for event in events
        if event.get("record_type") == "candidate_state"
    ]
    payload = json.dumps(
        trajectory,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_stage033_baseline(
    root: Path,
) -> dict[tuple[str, int], tuple[int, float, float, int]]:
    path = root / (
        "experiments/summaries/"
        "stage03.3_exact_deadline_attempt06_review/per_run_results.csv"
    )
    output: dict[tuple[str, int], tuple[int, float, float, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["axis"] != "wall_clock":
                continue
            key = json.loads(row["objective_key"])
            output[(row["instance"], _as_int(row["seed"]))] = (
                int(key[0]), float(key[1]), float(key[2]), int(key[3])
            )
    return output


def _prerequisites_valid(root: Path) -> bool:
    stage033 = root / (
        "experiments/summaries/"
        "stage03.3_exact_deadline_attempt06_review/review_manifest.json"
    )
    if not stage033.is_file():
        return False
    payload = json.loads(stage033.read_text(encoding="utf-8"))
    return (
        payload.get("status") == "READY_FOR_STAGE03_4"
        and _verify_frozen_cpu_batch_evidence(root)
    )


def _paired_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["instance"]), _as_int(row["seed"])), {})[
            str(row["axis"])
        ] = row
    return [
        {
            "instance": instance,
            "seed": seed,
            **{
                f"{axis}_{field}": axes[axis].get(field)
                for axis in DIAGNOSTIC_AXES
                for field in ("started_calls", "effective_iterations", "objective_key")
            },
        }
        for (instance, seed), axes in sorted(grouped.items())
    ]


def _routes(value: object) -> list[list[str]]:
    if not isinstance(value, list):
        raise RuntimeError("solution routes must be a list")
    return [[str(name) for name in route] for route in value if isinstance(route, list)]


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be an object")
    return value


def _as_int(value: object) -> int:
    try:
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (str, bytes, bytearray, int, float)):
            return int(value)
        raise TypeError(type(value).__name__)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"expected integer, got {value!r}") from error


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Review Stage 3.4 evidence")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/schneider"))
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    outputs = review_stage034(
        run_dir=arguments.run_dir.resolve(),
        scope=arguments.scope,
        benchmark_dir=arguments.benchmark_dir.resolve(),
        output_dir=arguments.output_dir.resolve(),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
