"""Independent gates for Stage 3.3 exact-deadline evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evrptw.artifacts import ArtifactReader, verify_manifest
from evrptw.cache_incremental import canonical_instance_hash
from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES, FORMAL_SEEDS
from evrptw.experiments.stage03_measurement import SMOKE_INSTANCES, _reference_repositories
from evrptw.experiments.stage033_exact_deadline import (
    DIAGNOSTIC_AXES,
    _source_sha256,
)
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.validation import validate_routes

READY_FOR_STAGE033_FORMAL = "READY_FOR_STAGE033_FORMAL"
READY_FOR_STAGE034 = "READY_FOR_STAGE03_4"


def evaluate_stage033_gate(
    rows: Sequence[Mapping[str, object]],
    *,
    scope: str,
) -> dict[str, object]:
    """Apply the complete-scope backend and exact-call accounting gate."""

    instances = SMOKE_INSTANCES if scope == "smoke" else FORMAL_INSTANCES
    if scope not in {"smoke", "formal"}:
        raise ValueError("Stage 3.3 scope must be smoke or formal")
    expected = {
        (instance, seed, axis)
        for instance in instances
        for seed in FORMAL_SEEDS
        for axis in DIAGNOSTIC_AXES
    }
    observed: set[tuple[str, int, str]] = set()
    valid = True
    for row in rows:
        key = (
            str(row.get("instance")),
            _as_int(row.get("seed")),
            str(row.get("axis")),
        )
        if key in observed:
            valid = False
        observed.add(key)
        started = _as_int(row.get("started_calls"))
        completed = _as_int(row.get("completed_calls"))
        axis = key[2]
        budget = row.get("exact_call_budget")
        valid = valid and all(
            (
                row.get("backend") == "cpu_batch",
                bool(row.get("valid")),
                0 <= completed <= started,
                axis in DIAGNOSTIC_AXES,
                (budget is None if axis == "wall_clock" else _as_int(budget) == 100),
                (started <= 100 if axis == "fixed_exact_calls" else True),
            )
        )
    complete = observed == expected and len(rows) == len(expected)
    ready = complete and valid
    return {
        "schema_version": "stage033-review-v1",
        "scope": scope,
        "status": (
            READY_FOR_STAGE033_FORMAL
            if ready and scope == "smoke"
            else READY_FOR_STAGE034
            if ready
            else "NOT_READY"
        ),
        "scope_complete": complete,
        "all_rows_valid": valid,
        "expected_rows": len(expected),
        "observed_rows": len(rows),
    }


def review_stage033(
    *,
    run_dir: Path,
    scope: str,
    benchmark_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Independently replay a complete Stage 3.3 artifact bundle."""

    manifest = verify_manifest(run_dir)
    if manifest.get("stage_id") != "stage03.3" or manifest.get("component") != "exact_deadline":
        raise RuntimeError("artifact bundle is not Stage 3.3 exact_deadline evidence")
    if manifest.get("status") != "complete" or manifest.get("evidence_completeness") != "complete":
        raise RuntimeError("partial Stage 3.3 evidence cannot be reviewed as ready")
    run_label = str(manifest["run_label"])
    reader = ArtifactReader(run_dir)
    metadata_reference = next(
        reference
        for reference in manifest.get("artifacts", [])
        if reference.get("artifact_type") == "manifest_metadata"
    )
    control = reader.read_json(str(metadata_reference["relative_path"]))
    config_reference = next(
        reference
        for reference in manifest.get("artifacts", [])
        if reference.get("artifact_type") == "config"
    )
    root = run_dir.parents[1]
    recorded_references = control.get("reference_repositories")
    if not all(
        (
            control.get("scope") == scope,
            control.get("exact_backend") == "cpu_batch",
            control.get("repository_dirty") is False,
            control.get("diagnostic_axes") == list(DIAGNOSTIC_AXES),
            control.get("configuration_sha256")
            == _sha256(run_dir / str(config_reference["relative_path"])),
            control.get("source_sha256") == _source_sha256(root),
            control.get("stage00_manifest_sha256")
            == _sha256(root / "experiments/baselines/stage00/manifest.json"),
            recorded_references == _reference_repositories(root),
        )
    ):
        raise RuntimeError("Stage 3.3 control provenance or backend is invalid")
    grouped = _artifact_groups(manifest)
    rows: list[dict[str, object]] = []
    deadline_rows: list[dict[str, object]] = []
    backend_rows: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    for key, artifacts in sorted(grouped.items()):
        instance_name, seed = key
        raw = reader.read_json(artifacts["raw"])
        solution = reader.read_json(artifacts["solution"])
        trace = reader.trace_index(artifacts["trace"])
        environment = reader.read_json(artifacts["environment"])
        events = reader.read_events(artifacts["events"])
        route_dictionary = reader.read_parquet(artifacts["route_dictionary"])
        if raw.get("instance") != instance_name or int(raw.get("seed", 0)) != seed:
            raise RuntimeError(f"raw identity mismatch: {instance_name}/{seed}")
        instance = parse_schneider(benchmark_dir / f"{instance_name}.txt")
        instance_hash = canonical_instance_hash(instance)
        raw_axes = _object(raw.get("axes"), "raw axes")
        solution_axes = _object(solution.get("axes"), "solution axes")
        trace_axes = _object(trace.get("axes"), "trace axes")
        route_ids = {int(row["route_id"]) for row in route_dictionary}
        route_references_valid = all(
            event.get(field) is None or int(event[field]) in route_ids
            for event in events
            for field in ("route_id", "base_route_id", "candidate_route_id")
        )
        provenance_valid = all(
            (
                environment.get("repository_dirty") is False,
                environment.get("repository_revision")
                == control.get("repository_revision"),
                environment.get("source_sha256") == control.get("source_sha256"),
                environment.get("configuration_sha256")
                == control.get("configuration_sha256"),
                environment.get("exact_backend") == "cpu_batch",
                environment.get("stage00_manifest_sha256")
                == control.get("stage00_manifest_sha256"),
                environment.get("reference_repositories") == recorded_references,
                int(raw.get("peak_rss_bytes") or 0) > 0,
            )
        )
        for axis in DIAGNOSTIC_AXES:
            raw_axis = _object(raw_axes.get(axis), f"{axis} raw evidence")
            solution_axis = _object(solution_axes.get(axis), f"{axis} solution")
            trace_axis = _object(trace_axes.get(axis), f"{axis} trace")
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
            axis_events = [
                event
                for event in events
                if _event_axis(event) == axis
            ]
            exact_events = [
                event
                for event in axis_events
                if event.get("record_type") == "route_evaluation"
                and event.get("kind") == "exact_call"
            ]
            started = sum(bool(event.get("exact_started")) for event in exact_events)
            completed = sum(bool(event.get("exact_completed")) for event in exact_events)
            interrupted = started - completed
            infeasible = sum(
                event.get("status") == "completed_infeasible"
                for event in exact_events
            )
            boundary_count = sum(
                event.get("event_type") == "exact_budget_boundary"
                for event in axis_events
            )
            backend = str(raw_axis.get("backend"))
            metrics = _object(raw_axis.get("backend_metrics"), "backend metrics")
            metric_fields = {
                "batch_launches",
                "transitions",
                "packing_seconds",
                "unpacking_seconds",
                "exact_calls",
                "total_seconds",
            }
            metrics_complete = metric_fields.issubset(metrics)
            metrics_reconcile = all(
                (
                    int(metrics.get("started_calls", -1)) == started,
                    int(metrics.get("completed_calls", -1)) == completed,
                    int(metrics.get("interrupted_calls", -1)) == interrupted,
                )
            )
            budget = (
                int(
                    _object(
                        raw_axis.get("exact_deadline_statistics"),
                        "deadline statistics",
                    ).get("exact_call_budget")
                    or 0
                )
                if axis == "fixed_exact_calls"
                else None
            )
            counters_match = all(
                (
                    started == int(raw_axis.get("started_calls", -1)),
                    completed == int(raw_axis.get("completed_calls", -1)),
                    interrupted == int(raw_axis.get("interrupted_calls", -1)),
                )
            )
            trace_summary = _object(trace_axis.get("summary"), "trace summary")
            trace_matches = all(
                (
                    int(trace_summary.get("started_calls", -1)) == started,
                    int(trace_summary.get("completed_calls", -1)) == completed,
                    str(trace_axis.get("trace_schema_version")) == "stage03-trace-v3",
                    trace_axis.get("instance_hash") == instance_hash,
                    _object(
                        raw_axis.get("trace_reconciliation"),
                        "trace reconciliation",
                    ).get("status")
                    == "pass",
                )
            )
            acceptance_valid = _acceptance_semantics_valid(axis_events)
            cache_lifecycle_valid = _cache_lifecycle_valid(axis_events)
            no_post_boundary_store = _no_cache_store_after_boundary(axis_events)
            candidate_rollback_valid = _candidate_rollbacks_are_transactional(axis_events)
            deadline_context_valid = _deadline_context_is_transactional(axis_events)
            fixed_protocol_valid = (
                axis != "fixed_exact_calls"
                or (
                    started <= 100
                    and (
                        (
                            started == 100
                            and raw_axis.get("termination_reason")
                            == "exact_call_budget_exhausted"
                        )
                        or (
                            started < 100
                            and raw_axis.get("termination_reason") == "iteration_limit"
                        )
                    )
                )
            )
            valid = all(
                (
                    bool(raw_axis.get("valid")),
                    validation.feasible,
                    objective_matches,
                    backend == "cpu_batch",
                    counters_match,
                    metrics_complete,
                    metrics_reconcile,
                    trace_matches,
                    provenance_valid,
                    route_references_valid,
                    acceptance_valid,
                    cache_lifecycle_valid,
                    no_post_boundary_store,
                    candidate_rollback_valid,
                    deadline_context_valid,
                    fixed_protocol_valid,
                    started <= 100 if axis == "fixed_exact_calls" else True,
                )
            )
            rows.append(
                {
                    "instance": instance_name,
                    "seed": seed,
                    "axis": axis,
                    "backend": backend,
                    "valid": valid,
                    "objective_key": json.dumps(objective_key, separators=(",", ":")),
                    "vehicle_count": objective.vehicle_count if objective is not None else "",
                    "runtime_seconds": raw_axis.get("runtime_seconds"),
                    "iterations": raw_axis.get("iterations"),
                    "effective_iterations": raw_axis.get("effective_iterations"),
                    "started_calls": started,
                    "completed_calls": completed,
                    "interrupted_calls": interrupted,
                    "infeasible_calls": infeasible,
                    "exact_call_budget": budget,
                    "budget_exhaustions": raw_axis.get("budget_exhaustions"),
                    "termination_reason": raw_axis.get("termination_reason"),
                    "trace_reconciliation_status": _object(
                        raw_axis.get("trace_reconciliation"), "trace reconciliation"
                    ).get("status"),
                    "provenance_valid": provenance_valid,
                    "route_references_valid": route_references_valid,
                    "candidate_rollback_valid": candidate_rollback_valid,
                    "cache_lifecycle_valid": cache_lifecycle_valid,
                    "deadline_context_valid": deadline_context_valid,
                }
            )
            deadline_rows.append(
                {
                    "instance": instance_name,
                    "seed": seed,
                    "axis": axis,
                    "started_calls": started,
                    "completed_calls": completed,
                    "interrupted_calls": interrupted,
                    "infeasible_calls": infeasible,
                    "budget_boundary_events": boundary_count,
                    "no_post_boundary_cache_store": no_post_boundary_store,
                    "acceptance_semantics_valid": acceptance_valid,
                    "candidate_rollback_valid": candidate_rollback_valid,
                    "cache_lifecycle_valid": cache_lifecycle_valid,
                    "deadline_context_valid": deadline_context_valid,
                }
            )
            backend_rows.append(
                {
                    "instance": instance_name,
                    "seed": seed,
                    "axis": axis,
                    "backend": backend,
                    **{field: metrics.get(field) for field in sorted(metric_fields)},
                    "started_calls": metrics.get("started_calls"),
                    "completed_calls": metrics.get("completed_calls"),
                    "interrupted_calls": metrics.get("interrupted_calls"),
                }
            )
            findings.append(
                {
                    "instance": instance_name,
                    "seed": seed,
                    "axis": axis,
                    "status": "pass" if valid else "fail",
                    "reason": (
                        "independent validator, objective, trace, backend and deadline replay"
                    ),
                }
            )
    golden = _verify_frozen_cpu_batch_evidence(run_dir.parents[1])
    gate = evaluate_stage033_gate(rows, scope=scope)
    if not golden:
        gate["status"] = "NOT_READY"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_run_results": output_dir / "per_run_results.csv",
        "fixed_work_wall_clock": output_dir / "fixed_work_wall_clock.csv",
        "deadline_report": output_dir / "deadline_report.csv",
        "backend_metrics": output_dir / "backend_metrics.csv",
        "review_findings": output_dir / "review_findings.csv",
        "review_report": output_dir / "review_report.md",
        "review_manifest": output_dir / "review_manifest.json",
    }
    _write_csv(paths["per_run_results"], rows)
    _write_csv(paths["fixed_work_wall_clock"], _paired_rows(rows))
    _write_csv(paths["deadline_report"], deadline_rows)
    _write_csv(paths["backend_metrics"], backend_rows)
    _write_csv(paths["review_findings"], findings)
    paths["review_report"].write_text(
        "# Stage 3.3 exact deadline independent review\n\n"
        f"- Status: `{gate['status']}`\n"
        f"- Scope: `{scope}`\n"
        f"- Complete paired rows: `{gate['observed_rows']}/{gate['expected_rows']}`\n"
        f"- Frozen CPU batch golden evidence: `{'PASS' if golden else 'FAIL'}`\n"
        "- Stage 3 overall performance claim: `not made; remaining targets move to Stage 3.4`\n",
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
                "run_label": run_label,
                "run_directory": str(run_dir),
                "frozen_cpu_batch_golden": golden,
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


def _artifact_groups(manifest: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, str]]:
    groups: dict[tuple[str, int], dict[str, str]] = {}
    for reference in manifest.get("artifacts", []):
        if not isinstance(reference, Mapping):
            continue
        artifact_type = str(reference.get("artifact_type"))
        artifact_subtype = str(reference.get("artifact_subtype") or "")
        required = {
            "raw",
            "solution",
            "trace",
            "environment",
            "route_dictionary",
            "events",
        }
        if artifact_type not in required:
            continue
        if artifact_type == "events" and artifact_subtype != "critical":
            continue
        relative = Path(str(reference["relative_path"]))
        if len(relative.parts) < 3:
            continue
        key = (relative.parts[0], int(relative.parts[1]))
        groups.setdefault(key, {})[artifact_type] = relative.as_posix()
    for key, artifacts in groups.items():
        if set(artifacts) != required:
            raise RuntimeError(f"incomplete Stage 3.3 artifact group: {key}")
    return groups


def _event_axis(event: Mapping[str, object]) -> str:
    extras = json.loads(str(event.get("extras_json") or "{}"))
    return str(extras.get("diagnostic_axis", ""))


def _acceptance_semantics_valid(events: Sequence[Mapping[str, object]]) -> bool:
    vehicle_first_valid = all(
        not (
            bool(event.get("accepted"))
            and event.get("candidate_vehicle_count") is not None
            and event.get("current_vehicle_count") is not None
            and _as_int(event["candidate_vehicle_count"])
            > _as_int(event["current_vehicle_count"])
        )
        for event in events
    )
    global_best_keys: list[tuple[int, float, float, int]] = []
    for event in events:
        extras = json.loads(str(event.get("extras_json") or "{}"))
        if bool(event.get("accepted")):
            candidate_key = extras.get("candidate_objective_key")
            if event.get("candidate_feasible") is not True or not (
                isinstance(candidate_key, list) and len(candidate_key) == 4
            ):
                return False
        if not bool(event.get("global_best")):
            continue
        key = extras.get("candidate_objective_key")
        if not isinstance(key, list) or len(key) != 4:
            return False
        objective_key = (int(key[0]), float(key[1]), float(key[2]), int(key[3]))
        if not global_best_keys or objective_key != global_best_keys[-1]:
            global_best_keys.append(objective_key)
    global_best_monotone = all(
        current < previous
        for previous, current in zip(global_best_keys, global_best_keys[1:], strict=False)
    )
    return vehicle_first_valid and global_best_monotone


def _cache_lifecycle_valid(events: Sequence[Mapping[str, object]]) -> bool:
    stored: set[str] = set()
    for event in events:
        if event.get("record_type") != "cache_event":
            continue
        digest = str(event.get("cache_key_digest") or "")
        operation = str(event.get("operation") or "")
        extras = json.loads(str(event.get("extras_json") or "{}"))
        if operation == "store":
            if not digest:
                return False
            stored.add(digest)
        elif operation == "evict":
            if digest not in stored:
                return False
            stored.remove(digest)
        elif operation == "lookup_result" and extras.get("lookup_result") == "hit":
            if digest not in stored:
                return False
    return True


def _no_cache_store_after_boundary(events: Sequence[Mapping[str, object]]) -> bool:
    boundary_ids = [
        _as_int(event.get("event_id"))
        for event in events
        if event.get("event_type") == "exact_budget_boundary"
    ]
    if not boundary_ids:
        return True
    boundary = min(boundary_ids)
    return not any(
        _as_int(event.get("event_id")) > boundary
        and (
            (
                event.get("record_type") == "cache_event"
                and event.get("operation") == "store"
            )
            or (
                event.get("record_type") == "candidate_state"
                and (
                    bool(event.get("accepted"))
                    or bool(event.get("global_best"))
                )
            )
        )
        for event in events
    )


def _candidate_rollbacks_are_transactional(
    events: Sequence[Mapping[str, object]],
) -> bool:
    for rollback in events:
        if rollback.get("event_type") != "candidate_cache_rollback":
            continue
        context = tuple(
            rollback.get(field) for field in ("lane_id", "iteration")
        )
        for event in events:
            if tuple(
                event.get(field) for field in ("lane_id", "iteration")
            ) != context:
                continue
            if (
                event.get("operation") == "store"
                or event.get("event_type") == "candidate_cache_commit"
                or bool(event.get("accepted"))
                or bool(event.get("global_best"))
            ):
                return False
    return True


def _deadline_context_is_transactional(
    events: Sequence[Mapping[str, object]],
) -> bool:
    for boundary in events:
        if boundary.get("event_type") != "deadline_boundary":
            continue
        boundary_id = _as_int(boundary.get("event_id"))
        context = tuple(
            boundary.get(field) for field in ("lane_id", "iteration")
        )
        for event in events:
            if _as_int(event.get("event_id")) <= boundary_id:
                continue
            if tuple(
                event.get(field) for field in ("lane_id", "iteration")
            ) != context:
                continue
            if (
                event.get("operation") == "store"
                or bool(event.get("accepted"))
                or bool(event.get("global_best"))
            ):
                return False
    return True


def _verify_frozen_cpu_batch_evidence(root: Path) -> bool:
    summary = root / "experiments" / "summaries" / "cpu_batch_pilot_attempt01_per_run.csv"
    report = root / "experiments" / "summaries" / "cpu_batch_pilot_attempt01_review.md"
    raw_dir = root / "results" / "cpu_batch_pilot_attempt01"
    raw_manifest = raw_dir / "raw_manifest.json"
    sidecar = raw_dir / "raw_manifest.sha256"
    replay_gate = raw_dir / "replay_gate.json"
    paired_gate = raw_dir / "paired_gate.json"
    paired_results = raw_dir / "paired_results.json"
    replay_results = raw_dir / "replay_results.json"
    required = (
        summary,
        report,
        raw_manifest,
        sidecar,
        replay_gate,
        paired_gate,
        paired_results,
        replay_results,
    )
    if not all(path.is_file() for path in required):
        return False
    if sidecar.read_text(encoding="utf-8").strip().split()[0] != _sha256(raw_manifest):
        return False
    manifest_payload = json.loads(raw_manifest.read_text(encoding="utf-8"))
    for relative, metadata in manifest_payload.get("files", {}).items():
        path = raw_dir / relative
        if (
            not path.is_file()
            or int(metadata.get("bytes", -1)) != path.stat().st_size
            or metadata.get("sha256") != _sha256(path)
        ):
            return False
    replay = json.loads(replay_gate.read_text(encoding="utf-8"))
    paired = json.loads(paired_gate.read_text(encoding="utf-8"))
    paired_rows = json.loads(paired_results.read_text(encoding="utf-8")).get("rows", [])
    replay_rows = json.loads(replay_results.read_text(encoding="utf-8")).get("rows", [])
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary_by_key = {
        (str(row["instance"]), int(row["seed"])): row for row in rows
    }
    paired_semantics_valid = len(paired_rows) == 12 and all(
        _golden_pair_valid(row, summary_by_key) for row in paired_rows
    )
    replay_semantics_valid = len(replay_rows) == 24 and all(
        bool(row.get("correctness"))
        and int(row.get("request_count", -1))
        == int(_object(row.get("metrics"), "golden metrics").get("exact_calls", -2))
        for row in replay_rows
    )
    return (
        len(rows) == 12
        and all(str(row.get("valid", "")).lower() == "true" for row in rows)
        and all(int(row.get("fixed_iterations", 0)) == 40 for row in rows)
        and replay.get("status") == "PASS_REPLAY"
        and replay.get("protocol_complete") is True
        and replay.get("correctness") is True
        and paired.get("status") == "ADOPT_CPU_BATCH_DEFAULT"
        and paired.get("all_pairs_valid") is True
        and paired.get("protocol_complete") is True
        and paired_semantics_valid
        and replay_semantics_valid
        and "ADOPT_CPU_BATCH_DEFAULT" in report.read_text(encoding="utf-8")
    )


def _golden_pair_valid(
    row: Mapping[str, object],
    summary_by_key: Mapping[tuple[str, int], Mapping[str, str]],
) -> bool:
    scalar = _object(row.get("scalar"), "golden scalar pair")
    batch = _object(row.get("batch"), "golden batch pair")
    key = (str(row.get("instance")), _as_int(row.get("seed")))
    summary = summary_by_key.get(key)
    if summary is None:
        return False
    semantic_fields = (
        "candidate_work_hash",
        "route_result_hash",
        "objective_key",
        "exact_calls",
        "effective_iterations",
        "validator_feasible",
        "trace_reconciliation_status",
    )
    return all(scalar.get(field) == batch.get(field) for field in semantic_fields) and all(
        (
            batch.get("validator_feasible") is True,
            batch.get("trace_reconciliation_status") == "pass",
            _as_int(row.get("fixed_iterations")) == 40,
            abs(float(summary["batch_runtime_seconds"]) - float(str(row["batch_runtime_seconds"])))
            <= 1e-12,
            abs(
                float(summary["scalar_runtime_seconds"])
                - float(str(row["scalar_runtime_seconds"]))
            )
            <= 1e-12,
            str(summary["valid"]).lower() == "true",
        )
    )


def _paired_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["instance"]), _as_int(row["seed"])), {})[
            str(row["axis"])
        ] = row
    output: list[dict[str, object]] = []
    for (instance, seed), axes in sorted(grouped.items()):
        fixed = axes["fixed_exact_calls"]
        wall = axes["wall_clock"]
        output.append(
            {
                "instance": instance,
                "seed": seed,
                "fixed_runtime_seconds": fixed["runtime_seconds"],
                "wall_runtime_seconds": wall["runtime_seconds"],
                "fixed_effective_iterations": fixed["effective_iterations"],
                "wall_effective_iterations": wall["effective_iterations"],
                "fixed_started_calls": fixed["started_calls"],
                "wall_started_calls": wall["started_calls"],
            }
        )
    return output


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (int, str, bytes, bytearray)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    raise ValueError(f"expected integer-compatible value, got {type(value).__name__}")


def _routes(value: object) -> list[list[str]]:
    if not isinstance(value, list) or not all(isinstance(route, list) for route in value):
        raise ValueError("solution routes must be a list of lists")
    return [[str(node) for node in route] for route in value]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Review Stage 3.3 exact-deadline evidence")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/schneider"))
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    outputs = review_stage033(
        run_dir=arguments.run_dir.resolve(),
        scope=arguments.scope,
        benchmark_dir=arguments.benchmark_dir.resolve(),
        output_dir=arguments.output_dir.resolve(),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    status = json.loads(outputs["review_manifest"].read_text(encoding="utf-8"))["status"]
    return 0 if status in {READY_FOR_STAGE033_FORMAL, READY_FOR_STAGE034} else 1


if __name__ == "__main__":
    raise SystemExit(main())
