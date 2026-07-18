"""Independent raw replay for Stage 5.2 performance evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evrptw.artifacts import (
    DIAGNOSTIC_SCHEMA,
    EVENTS_SCHEMA,
    ROUTE_DICTIONARY_SCHEMA,
    SCREENING_CHECKS_SCHEMA,
    V2_SCREENING_DECISIONS_SCHEMA,
    V2_SCREENING_DECISIONS_SCHEMA_V1,
    ArtifactIntegrityError,
    ArtifactReader,
    expand_v2_screening_decision,
)
from evrptw.best_known import BEST_KNOWN_VALUES
from evrptw.experiments.stage052_performance import (
    axes_for_scope,
    validate_stage052_run_label,
)
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.stage052 import (
    ArtifactStorageObservation,
    PerformanceObservation,
    Stage052Component,
    decide_accelerator,
    evaluate_artifact_storage_promotion,
    evaluate_promotion,
    select_worker_count,
)
from evrptw.stage052_evidence import verify_stage052_prerequisite
from evrptw.validation import validate_routes

STAGE052_REVIEW_SCHEMA_VERSION = "stage05.2-review-v1"
NOT_READY = "NOT_READY"
_CANONICAL_CUSTOMER_COUNTS = {
    record.instance: record.customer_count for record in BEST_KNOWN_VALUES
}
_NEXT_STATUS = {
    Stage052Component.PERF_BASELINE: "READY_FOR_STAGE052_HOT_PATH",
    Stage052Component.HOT_PATH: "READY_FOR_STAGE052_ARTIFACT_STREAMING",
    Stage052Component.ARTIFACT_STREAMING: "READY_FOR_STAGE052_JOB_PARALLEL",
    Stage052Component.JOB_PARALLEL: "READY_FOR_STAGE052_NATIVE_KERNELS",
    Stage052Component.NATIVE_KERNELS: "READY_FOR_STAGE052_ACCELERATOR_DECISION",
    Stage052Component.ACCELERATOR_PILOT: "READY_FOR_STAGE052_BENCHMARK",
    Stage052Component.BENCHMARK: "READY_FOR_STAGE05_3",
}
_C_PREREQUISITE_STATUS = {
    "perf_baseline": "READY_FOR_STAGE052_HOT_PATH",
    "hot_path": "READY_FOR_STAGE052_ARTIFACT_STREAMING",
}


def _iter_stage052_event_rows(
    reader: ArtifactReader,
    *,
    events_ref: Mapping[str, Any],
    compact_screening_ref: Mapping[str, Any] | None,
    event_columns: Sequence[str],
) -> Iterable[dict[str, object]]:
    def ordinary_rows() -> Iterable[dict[str, object]]:
        for batch in reader.iter_parquet_batches(
            str(events_ref["relative_path"]),
            schema=EVENTS_SCHEMA,
            columns=event_columns,
        ):
            yield from batch.to_pylist()

    if compact_screening_ref is None:
        yield from ordinary_rows()
        return

    def compact_rows() -> Iterable[dict[str, object]]:
        relative = str(compact_screening_ref["relative_path"])
        compact_schema = reader.parquet_schema(relative)
        if not any(
            compact_schema.equals(schema)
            for schema in (
                V2_SCREENING_DECISIONS_SCHEMA,
                V2_SCREENING_DECISIONS_SCHEMA_V1,
            )
        ):
            raise ArtifactIntegrityError("unsupported compact screening schema")
        definitions: dict[int, dict[str, object]] = {}
        for batch in reader.iter_parquet_batches(
            relative,
            schema=compact_schema,
        ):
            for row in batch.to_pylist():
                expanded = expand_v2_screening_decision(row, definitions=definitions)
                yield {key: value for key, value in expanded.items() if key in event_columns}

    yield from heapq.merge(
        ordinary_rows(),
        compact_rows(),
        key=lambda row: _strict_int(row.get("event_id"), "event_id"),
    )


def _iter_stage052_check_rows(
    reader: ArtifactReader,
    *,
    checks_ref: Mapping[str, Any],
    compact_screening_ref: Mapping[str, Any] | None,
) -> Iterable[dict[str, object]]:
    def ordinary_rows() -> Iterable[dict[str, object]]:
        for batch in reader.iter_parquet_batches(
            str(checks_ref["relative_path"]),
            schema=SCREENING_CHECKS_SCHEMA,
        ):
            yield from batch.to_pylist()

    if compact_screening_ref is None:
        yield from ordinary_rows()
        return

    def compact_rows() -> Iterable[dict[str, object]]:
        relative = str(compact_screening_ref["relative_path"])
        compact_schema = reader.parquet_schema(relative)
        if not any(
            compact_schema.equals(schema)
            for schema in (
                V2_SCREENING_DECISIONS_SCHEMA,
                V2_SCREENING_DECISIONS_SCHEMA_V1,
            )
        ):
            raise ArtifactIntegrityError("unsupported compact screening schema")
        definitions: dict[int, dict[str, object]] = {}
        for batch in reader.iter_parquet_batches(
            relative,
            schema=compact_schema,
        ):
            for row in batch.to_pylist():
                expanded = expand_v2_screening_decision(row, definitions=definitions)
                checks = expanded.get("embedded_checks")
                if not isinstance(checks, list):
                    continue
                for index, check in enumerate(checks):
                    if not isinstance(check, Mapping):
                        raise ArtifactIntegrityError("compact screening check must be an object")
                    yield {
                        "decision_event_id": row.get("event_id"),
                        "decision_id": row.get("decision_id"),
                        "check_index": index,
                        "check": str(check.get("check", "")),
                        "status": str(check.get("status", "")),
                        "value_bool": (
                            check.get("value") if isinstance(check.get("value"), bool) else None
                        ),
                        "value_float": (
                            float(check["value"])
                            if isinstance(check.get("value"), (int, float))
                            else None
                        ),
                        "value_text": (
                            check.get("value") if isinstance(check.get("value"), str) else None
                        ),
                        "reason": str(check.get("reason", "")),
                    }

    yield from heapq.merge(
        ordinary_rows(),
        compact_rows(),
        key=lambda row: (
            _strict_int(row.get("decision_event_id"), "decision_event_id"),
            _strict_int(row.get("check_index"), "check_index"),
        ),
    )


def verify_stage052_review_prerequisite(
    raw_dir: Path,
    *,
    expected_component: str,
    expected_status: str,
) -> None:
    """Verify one accepted Stage 5.2 independent-review identity and its files."""

    manifest_path = raw_dir / "review" / "review_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read prerequisite review manifest: {manifest_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("prerequisite review manifest must be an object")
    expected_identity = {
        "schema_version": STAGE052_REVIEW_SCHEMA_VERSION,
        "run_label": raw_dir.name,
        "component": expected_component,
        "scope": "performance",
        "status": expected_status,
    }
    for field, expected in expected_identity.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"prerequisite review {field} mismatch: "
                f"expected={expected} observed={payload.get(field)}"
            )
    gates = payload.get("gates")
    if (
        not isinstance(gates, Mapping)
        or not gates
        or any(
            not isinstance(gate, Mapping) or gate.get("passed") is not True
            for gate in gates.values()
        )
    ):
        raise ValueError("prerequisite review contains a failed or invalid gate")
    files = payload.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "review_findings.csv",
        "review_report.md",
    }:
        raise ValueError("prerequisite review file identity mismatch")
    for name, expected_checksum in files.items():
        path = raw_dir / "review" / str(name)
        if not path.is_file() or _sha256(path) != str(expected_checksum):
            raise ValueError(f"prerequisite review checksum mismatch: {name}")


def replay_stage052_storage_semantics(
    raw_dir: Path,
) -> dict[tuple[str, int, str], str]:
    """Recompute storage-comparison digests from raw JSON and streamed events."""

    reader = ArtifactReader(raw_dir)
    artifacts = [item for item in reader.manifest.get("artifacts", []) if isinstance(item, Mapping)]
    by_directory: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    for item in artifacts:
        relative = str(item.get("relative_path", ""))
        directory = str(Path(relative).parent)
        key = (str(item.get("artifact_type", "")), str(item.get("artifact_subtype", "")))
        by_directory.setdefault(directory, {})[key] = item

    output: dict[tuple[str, int, str], str] = {}
    for directory, items in sorted(by_directory.items()):
        raw_ref = items.get(("raw", ""))
        solution_ref = items.get(("solution", ""))
        trace_ref = items.get(("trace", ""))
        route_ref = items.get(("route_dictionary", "canonical_routes"))
        events_ref = items.get(("events", "critical"))
        compact_screening_ref = items.get(("events", "screening_decisions_v2"))
        checks_ref = items.get(("events", "screening_checks"))
        diagnostic_ref = items.get(("diagnostic", "aggregated"))
        required = (
            raw_ref,
            solution_ref,
            trace_ref,
            route_ref,
            events_ref,
            checks_ref,
            diagnostic_ref,
        )
        if all(item is None for item in required):
            continue
        if any(item is None for item in required):
            raise ArtifactIntegrityError(f"storage replay artifacts are incomplete in {directory}")
        assert raw_ref is not None
        assert solution_ref is not None
        assert trace_ref is not None
        assert route_ref is not None
        assert events_ref is not None
        assert checks_ref is not None
        assert diagnostic_ref is not None
        raw = reader.read_json(str(raw_ref["relative_path"]))
        solution = reader.read_json(str(solution_ref["relative_path"]))
        trace = reader.read_json(str(trace_ref["relative_path"]))
        instance = str(raw.get("instance", solution.get("instance", "")))
        seed = _strict_int(raw.get("seed", solution.get("seed")), "seed")
        raw_axes = raw.get("axes")
        solution_axes = solution.get("axes")
        if not isinstance(raw_axes, Mapping) or not isinstance(solution_axes, Mapping):
            raise ArtifactIntegrityError(f"storage replay axes are missing in {directory}")
        if set(map(str, raw_axes)) != set(map(str, solution_axes)):
            raise ArtifactIntegrityError(f"raw/solution axis identity mismatch in {directory}")
        lane_dictionary = _trace_dictionary(trace, "lane_dictionary", directory)
        operator_dictionary = _trace_dictionary(trace, "operator_dictionary", directory)
        route_dictionary: dict[int, dict[str, object]] = {}
        for batch in reader.iter_parquet_batches(
            str(route_ref["relative_path"]),
            schema=ROUTE_DICTIONARY_SCHEMA,
        ):
            for row in batch.to_pylist():
                route_id = _strict_int(row.pop("route_id", None), "route_id")
                if route_id in route_dictionary:
                    raise ArtifactIntegrityError(
                        f"duplicate route dictionary ID in {directory}: {route_id}"
                    )
                route_dictionary[route_id] = dict(row)
        hashers: dict[str, Any] = {}
        for axis, raw_axis in raw_axes.items():
            solution_axis = solution_axes.get(axis)
            if not isinstance(raw_axis, Mapping) or not isinstance(solution_axis, Mapping):
                raise ArtifactIntegrityError(
                    f"storage replay axis is invalid: {instance}/{seed}/{axis}"
                )
            base = {
                "objective_key": list(solution_axis.get("objective_key", [])),
                "routes": solution_axis.get("routes", []),
                "started_calls": _strict_int(raw_axis.get("started_calls"), "started_calls"),
                "completed_calls": _strict_int(raw_axis.get("completed_calls"), "completed_calls"),
            }
            axis_hasher = hashlib.sha256()
            axis_hasher.update(_canonical_json_bytes(base) + b"\n")
            hashers[str(axis)] = axis_hasher
        event_ordinals = {axis: 0 for axis in hashers}
        event_ranges: dict[str, tuple[int, int]] = {}
        event_columns = tuple(
            name
            for name in EVENTS_SCHEMA.names
            if name
            not in {
                "timestamp_seconds",
                "started_at",
                "completed_at",
                "duration_seconds",
            }
        )
        previous_event_id = 0
        for row in _iter_stage052_event_rows(
            reader,
            events_ref=events_ref,
            compact_screening_ref=compact_screening_ref,
            event_columns=event_columns,
        ):
            event_id = _strict_int(row.get("event_id"), "event_id")
            if event_id <= previous_event_id:
                raise ArtifactIntegrityError(f"event IDs are duplicate or unordered in {directory}")
            previous_event_id = event_id
            extras_raw = row.get("extras_json")
            if extras_raw is not None and not isinstance(extras_raw, str):
                raise ArtifactIntegrityError(f"event extras_json must be text in {directory}")
            try:
                extras = json.loads(extras_raw) if extras_raw else {}
            except json.JSONDecodeError as error:
                raise ArtifactIntegrityError(f"invalid event extras_json in {directory}") from error
            if not isinstance(extras, Mapping):
                raise ArtifactIntegrityError(f"event extras_json must be an object in {directory}")
            canonical_extras = dict(extras)
            if row.get("event_type") == "cache_event":
                for volatile_byte_field in (
                    "current_bytes",
                    "entry_bytes",
                    "lookup_current_bytes",
                ):
                    canonical_extras.pop(volatile_byte_field, None)
            axis = str(extras.get("benchmark_axis", ""))
            event_hasher = hashers.get(axis)
            if event_hasher is None:
                raise ArtifactIntegrityError(
                    f"event has unknown benchmark axis in {directory}: {axis}"
                )
            row.pop("event_id", None)
            lane_id = _strict_int(row.pop("lane_id", None), "lane_id")
            operator_id = _strict_int(row.pop("operator_id", None), "operator_id")
            row.pop("extras_json", None)
            _replace_event_route_ids(
                row,
                route_dictionary=route_dictionary,
                directory=directory,
            )
            event_ordinals[axis] += 1
            low, high = event_ranges.get(axis, (event_id, event_id))
            event_ranges[axis] = (min(low, event_id), max(high, event_id))
            event_payload = {
                "record": "event",
                "axis_event_ordinal": event_ordinals[axis],
                "lane": lane_dictionary.get(lane_id),
                "operator": operator_dictionary.get(operator_id),
                **row,
                "extras": canonical_extras,
            }
            if event_payload["lane"] is None or event_payload["operator"] is None:
                raise ArtifactIntegrityError(f"event dictionary identity is missing in {directory}")
            event_hasher.update(_canonical_json_bytes(event_payload) + b"\n")
        for row in _iter_stage052_check_rows(
            reader,
            checks_ref=checks_ref,
            compact_screening_ref=compact_screening_ref,
        ):
            event_id = _strict_int(row.get("decision_event_id"), "decision_event_id")
            matching_axes = [
                axis for axis, (low, high) in event_ranges.items() if low <= event_id <= high
            ]
            if len(matching_axes) != 1:
                raise ArtifactIntegrityError(
                    f"screening check event identity is ambiguous in {directory}"
                )
            axis = matching_axes[0]
            payload = dict(row)
            payload["decision_event_id"] = event_id - event_ranges[axis][0] + 1
            hashers[axis].update(
                _canonical_json_bytes({"record": "screening_check", **payload}) + b"\n"
            )
        for batch in reader.iter_parquet_batches(
            str(diagnostic_ref["relative_path"]),
            schema=DIAGNOSTIC_SCHEMA,
        ):
            for row in batch.to_pylist():
                lane = str(row.get("lane", ""))
                axis = lane.split(":", 1)[0]
                diagnostic_hasher = hashers.get(axis)
                if diagnostic_hasher is None:
                    raise ArtifactIntegrityError(
                        f"diagnostic row has unknown benchmark axis in {directory}: {axis}"
                    )
                payload = dict(row)
                payload["run_label"] = "<canonical-run-label>"
                diagnostic_hasher.update(
                    _canonical_json_bytes({"record": "diagnostic", **payload}) + b"\n"
                )
        for axis, axis_hasher in hashers.items():
            identity = (instance, seed, axis)
            if identity in output:
                raise ArtifactIntegrityError(f"duplicate storage replay identity: {identity}")
            output[identity] = axis_hasher.hexdigest()
    if not output:
        raise ArtifactIntegrityError("storage replay evidence is empty")
    return output


def validate_per_run_scope(
    rows: Sequence[Mapping[str, object]],
    *,
    instances: Sequence[str],
    seeds: Sequence[int],
    axes: Sequence[str],
) -> tuple[bool, str]:
    expected = {(instance, seed, axis) for instance in instances for seed in seeds for axis in axes}
    observed: list[tuple[str, int, str]] = []
    failures: list[str] = []
    for row in rows:
        try:
            instance_name = str(row["instance"])
            identity = (
                instance_name,
                _strict_int(row["seed"], "seed"),
                str(row["axis"]),
            )
            customer_count = _strict_int(row["customer_count"], "customer_count")
        except (KeyError, TypeError, ValueError) as error:
            failures.append(str(error))
            continue
        observed.append(identity)
        expected_customer_count = _CANONICAL_CUSTOMER_COUNTS.get(instance_name)
        if expected_customer_count != customer_count:
            failures.append(
                f"customer_count mismatch for {identity}: "
                f"expected={expected_customer_count} observed={customer_count}"
            )
        if not _strict_bool(row.get("validator_passed")):
            failures.append(f"validator failed for {identity}")
        if str(row.get("failure_status", "")):
            failures.append(f"failure status is present for {identity}")
    if len(set(observed)) != len(observed):
        failures.append("duplicate per-run axis identity")
    observed_set = set(observed)
    if observed_set != expected:
        failures.append(
            f"scope identity mismatch: missing={len(expected - observed_set)} "
            f"extra={len(observed_set - expected)}"
        )
    return not failures, "; ".join(failures) if failures else "exact scope passed"


def review_stage052(
    *,
    raw_dir: Path,
    benchmark_dir: Path,
    component: Stage052Component | str,
    scope: str,
    comparison_dirs: Sequence[Path] = (),
    prerequisite_dir: Path | None = None,
) -> dict[str, Path]:
    selected = Stage052Component(component)
    validate_stage052_run_label(raw_dir.name, selected)
    reader = ArtifactReader(raw_dir)
    manifest = reader.manifest
    if manifest.get("evidence_completeness") != "complete":
        raise ArtifactIntegrityError("partial Stage 5.2 evidence cannot be reviewed")
    per_run_ref = _one_artifact(reader, "per_run_results")
    rows = _read_csv(raw_dir / str(per_run_ref["relative_path"]))
    metadata_ref = _one_artifact(reader, "manifest_metadata")
    metadata = reader.read_json(str(metadata_ref["relative_path"]))
    instances = tuple(str(value) for value in metadata["instances"])
    seeds = tuple(_strict_int(value, "seed") for value in metadata["seeds"])
    if scope == "formal":
        scope_passed, scope_detail = _validate_formal_scope(rows, instances, seeds)
    else:
        customer_count = None if scope == "performance" else 5
        axes = tuple(axis.name for axis in axes_for_scope(scope, customer_count=customer_count))
        scope_passed, scope_detail = validate_per_run_scope(
            rows, instances=instances, seeds=seeds, axes=axes
        )
    replay_passed, replay_detail = _replay_solutions(reader, benchmark_dir=benchmark_dir)
    gates: dict[str, dict[str, object]] = {
        "exact_scope": {"passed": scope_passed, "detail": scope_detail},
        "replay_consistency": {"passed": replay_passed, "detail": replay_detail},
        "optimization_profile": _optimization_profile_gate(
            selected, metadata.get("optimization_profile")
        ),
        "persistence_attribution": {
            "passed": metadata.get("persistence_attribution") == "critical_event_rows",
            "detail": str(metadata.get("persistence_attribution")),
        },
    }
    prerequisite_contract = {
        Stage052Component.JOB_PARALLEL: (
            "artifact_streaming",
            "READY_FOR_STAGE052_JOB_PARALLEL",
        ),
        Stage052Component.NATIVE_KERNELS: (
            "job_parallel",
            "READY_FOR_STAGE052_NATIVE_KERNELS",
        ),
        Stage052Component.ACCELERATOR_PILOT: (
            "native_kernels",
            "READY_FOR_STAGE052_ACCELERATOR_DECISION",
        ),
    }.get(selected)
    if prerequisite_contract is not None:
        if prerequisite_dir is None:
            gates["component_prerequisite"] = {
                "passed": False,
                "detail": f"{selected.value} requires --prerequisite-dir",
            }
        else:
            try:
                prerequisite_identity = verify_stage052_prerequisite(
                    prerequisite_dir,
                    expected_component=prerequisite_contract[0],
                    expected_status=prerequisite_contract[1],
                    expected_run_label=(
                        "stage05.2_artifact_streaming_attempt04"
                        if selected is Stage052Component.JOB_PARALLEL
                        else None
                    ),
                )
            except (ArtifactIntegrityError, ValueError) as error:
                gates["component_prerequisite"] = {
                    "passed": False,
                    "detail": str(error),
                }
            else:
                gates["component_prerequisite"] = {
                    "passed": True,
                    "detail": prerequisite_identity.run_label,
                }
    gates.update(
        _component_gates(
            selected,
            rows,
            raw_dir=raw_dir,
            comparison_dirs=comparison_dirs,
            prerequisite_dir=prerequisite_dir,
        )
    )
    passed = all(bool(gate["passed"]) for gate in gates.values())
    status = _NEXT_STATUS[selected] if passed else NOT_READY
    review_dir = raw_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    findings_path = review_dir / "review_findings.csv"
    with findings_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("gate", "passed", "detail"))
        writer.writeheader()
        for gate, result in gates.items():
            writer.writerow({"gate": gate, **result})
    report_path = review_dir / "review_report.md"
    report_path.write_text(
        "\n".join(
            [
                f"# Stage 5.2 Review — {raw_dir.name}",
                "",
                f"**Status: {status}**",
                "",
                *[
                    f"- {name}: {'PASS' if result['passed'] else 'FAIL'} — {result['detail']}"
                    for name, result in gates.items()
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    review_manifest_path = review_dir / "review_manifest.json"
    review_manifest: dict[str, object] = {
                "schema_version": STAGE052_REVIEW_SCHEMA_VERSION,
                "run_label": raw_dir.name,
                "component": selected.value,
                "scope": scope,
                "status": status,
                "gates": gates,
                "files": {
                    findings_path.name: _sha256(findings_path),
                    report_path.name: _sha256(report_path),
                },
            }
    worker_gate = gates.get("worker_selection", {})
    if selected is Stage052Component.JOB_PARALLEL and worker_gate.get("passed") is True:
        review_manifest["selected_workers"] = worker_gate.get("selected_workers")
        review_manifest["selected_run_label"] = worker_gate.get("selected_run_label")
        review_manifest["input_runs"] = worker_gate.get("input_runs")
        review_manifest["resource_metrics"] = worker_gate.get("resource_metrics")
    review_manifest_path.write_text(
        json.dumps(review_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "review_report": report_path,
        "review_findings": findings_path,
        "review_manifest": review_manifest_path,
    }


def _validate_formal_scope(
    rows: Sequence[Mapping[str, object]],
    instances: Sequence[str],
    seeds: Sequence[int],
) -> tuple[bool, str]:
    expected = {
        (instance, seed, axis.name)
        for instance in instances
        for seed in seeds
        for axis in axes_for_scope("formal", customer_count=_CANONICAL_CUSTOMER_COUNTS[instance])
    }
    observed = {
        (str(row["instance"]), _strict_int(row["seed"], "seed"), str(row["axis"])) for row in rows
    }
    if len(rows) != len(observed):
        return False, "duplicate formal identity"
    count_failures = [
        str(row.get("instance"))
        for row in rows
        if _CANONICAL_CUSTOMER_COUNTS.get(str(row.get("instance")))
        != _strict_int(row.get("customer_count"), "customer_count")
    ]
    if count_failures:
        return False, f"customer_count mismatch for {len(count_failures)} formal axes"
    if observed != expected:
        return False, f"formal scope mismatch: expected={len(expected)} observed={len(observed)}"
    if len(expected) != 2040:
        return False, f"formal contract did not produce 2040 axes: {len(expected)}"
    return True, "exact 2040-run formal identity passed"


def _replay_solutions(reader: ArtifactReader, *, benchmark_dir: Path) -> tuple[bool, str]:
    failures: list[str] = []
    for item in reader.manifest.get("artifacts", []):
        if not isinstance(item, Mapping) or item.get("artifact_type") != "solution":
            continue
        payload = reader.read_json(str(item["relative_path"]))
        instance_name = str(payload["instance"])
        instance = parse_schneider(benchmark_dir / f"{instance_name}.txt")
        axes = payload.get("axes")
        if not isinstance(axes, Mapping):
            failures.append(f"{instance_name}: solution axes missing")
            continue
        for axis, raw_axis in axes.items():
            if not isinstance(raw_axis, Mapping):
                failures.append(f"{instance_name}/{axis}: invalid solution axis")
                continue
            routes = raw_axis.get("routes")
            if not isinstance(routes, list):
                failures.append(f"{instance_name}/{axis}: routes missing")
                continue
            report = validate_routes(instance, [list(map(str, route)) for route in routes])
            if not report.feasible:
                failures.append(f"{instance_name}/{axis}: validator failed")
                continue
            replayed = list(SolutionObjective.from_report(instance, report).key)
            recorded = list(raw_axis.get("objective_key", []))
            if replayed != recorded:
                failures.append(f"{instance_name}/{axis}: objective mismatch")
    return (
        not failures,
        "; ".join(failures[:20]) if failures else "validator/objective replay passed",
    )


def _component_gates(
    component: Stage052Component,
    rows: Sequence[Mapping[str, object]],
    *,
    raw_dir: Path,
    comparison_dirs: Sequence[Path],
    prerequisite_dir: Path | None,
) -> dict[str, dict[str, object]]:
    if component is Stage052Component.PERF_BASELINE:
        passed = _axis_semantics_equal(rows, "fixed_work_control", "fixed_work")
        return {
            "instrumentation_semantics": {
                "passed": passed,
                "detail": "fixed-work instrumentation semantic equality"
                if passed
                else "instrumentation changed fixed-work semantics",
            }
        }
    if component in {
        Stage052Component.HOT_PATH,
        Stage052Component.NATIVE_KERNELS,
    }:
        if len(comparison_dirs) != 1:
            return {
                "performance_promotion": {
                    "passed": False,
                    "detail": "one predecessor required",
                }
            }
        previous = _observations(_load_per_run(comparison_dirs[0]), axis="fixed_work")
        candidate = _observations(rows, axis="fixed_work")
        decision = evaluate_promotion(previous, candidate)
        return {
            "performance_promotion": {
                "passed": decision.passed,
                "detail": decision.detail,
            }
        }
    if component is Stage052Component.ARTIFACT_STREAMING:
        if len(comparison_dirs) != 2:
            return {
                "storage_prerequisites": {
                    "passed": False,
                    "detail": "exactly two comparison directories are required",
                }
            }
        by_component: dict[str, list[dict[str, str]]] = {}
        path_by_component: dict[str, Path] = {}
        for path in comparison_dirs:
            group = _load_per_run(path)
            components = {str(row["component"]) for row in group}
            if len(components) != 1:
                return {
                    "storage_prerequisites": {
                        "passed": False,
                        "detail": "comparison contains mixed component identities",
                    }
                }
            comparison_component = next(iter(components))
            expected_status = _C_PREREQUISITE_STATUS.get(comparison_component)
            if expected_status is None:
                return {
                    "storage_prerequisites": {
                        "passed": False,
                        "detail": f"unexpected comparison component: {comparison_component}",
                    }
                }
            try:
                verify_stage052_review_prerequisite(
                    path,
                    expected_component=comparison_component,
                    expected_status=expected_status,
                )
            except ValueError as error:
                return {
                    "storage_prerequisites": {
                        "passed": False,
                        "detail": str(error),
                    }
                }
            if comparison_component in by_component:
                return {
                    "storage_prerequisites": {
                        "passed": False,
                        "detail": f"duplicate comparison component: {comparison_component}",
                    }
                }
            by_component[comparison_component] = group
            path_by_component[comparison_component] = path
        if set(by_component) != {"perf_baseline", "hot_path"}:
            return {
                "storage_prerequisites": {
                    "passed": False,
                    "detail": "perf_baseline and hot_path comparisons are required",
                }
            }
        storage_decision = evaluate_artifact_storage_promotion(
            _storage_observations(by_component["perf_baseline"]),
            _storage_observations(
                by_component["hot_path"],
                semantic_digests=replay_stage052_storage_semantics(path_by_component["hot_path"]),
            ),
            _storage_observations(
                rows,
                semantic_digests=replay_stage052_storage_semantics(raw_dir),
            ),
        )
        return {
            "v1_v2_replay_equality": {
                "passed": storage_decision.replay_equality_passed,
                "detail": storage_decision.replay_detail,
            },
            "persistence_ratio": {
                "passed": storage_decision.persistence_passed,
                "detail": storage_decision.persistence_detail,
            },
            "peak_rss": {
                "passed": storage_decision.rss_passed,
                "detail": storage_decision.rss_detail,
            },
        }
    if component is Stage052Component.JOB_PARALLEL:
        if len(comparison_dirs) != 2:
            return {
                "worker_selection": {
                    "passed": False,
                    "detail": "1/2/4-worker evidence required",
                }
            }
        evidence_dirs = [*comparison_dirs, raw_dir]
        all_rows: list[Sequence[Mapping[str, object]]] = [
            *(_load_per_run(path) for path in comparison_dirs),
            list(rows),
        ]
        times: dict[int, float] = {}
        rss: dict[int, float] = {}
        run_by_worker: dict[int, str] = {}
        revisions: set[str] = set()
        configurations: set[str] = set()
        for evidence_dir, worker_rows in zip(evidence_dirs, all_rows, strict=True):
            workers = {_strict_int(row["worker_count"], "worker_count") for row in worker_rows}
            if len(workers) != 1:
                return {"worker_selection": {"passed": False, "detail": "mixed worker count"}}
            worker = next(iter(workers))
            if worker in run_by_worker:
                return {
                    "worker_selection": {
                        "passed": False,
                        "detail": f"duplicate worker evidence: {worker}",
                    }
                }
            resource = _load_resource_summary(evidence_dir)
            metadata = _load_metadata(evidence_dir)
            if _strict_int(resource.get("sample_count"), "sample_count") < 2:
                return {
                    "worker_selection": {
                        "passed": False,
                        "detail": f"insufficient resource samples for {evidence_dir.name}",
                    }
                }
            times[worker] = _strict_float(resource.get("run_wall_seconds"))
            rss[worker] = _strict_float(resource.get("aggregate_peak_rss_bytes")) / 2**30
            run_by_worker[worker] = evidence_dir.name
            revisions.add(str(metadata.get("repository_revision", "")))
            configurations.add(str(metadata.get("configuration_sha256", "")))
        if set(run_by_worker) != {1, 2, 4}:
            return {
                "worker_selection": {
                    "passed": False,
                    "detail": "worker evidence must contain exactly 1, 2, and 4 workers",
                }
            }
        identity_passed = len(revisions) == 1 and len(configurations) == 1
        if not identity_passed:
            return {
                "worker_identity": {
                    "passed": False,
                    "detail": "worker evidence mixes repository revisions or configurations",
                }
            }
        replay_maps = [replay_stage052_storage_semantics(path) for path in evidence_dirs]
        if prerequisite_dir is not None:
            replay_maps.insert(0, replay_stage052_storage_semantics(prerequisite_dir))
        fixed_identities = {
            identity for identity in replay_maps[0] if identity[2].startswith("fixed_work")
        }
        semantics_passed = bool(fixed_identities) and all(
            {identity for identity in replay if identity[2].startswith("fixed_work")}
            == fixed_identities
            and all(replay[identity] == replay_maps[0][identity] for identity in fixed_identities)
            for replay in replay_maps[1:]
        )
        if not semantics_passed:
            return {
                "worker_semantics": {
                    "passed": False,
                    "detail": "1/2/4-worker fixed-work semantic replay mismatch",
                }
            }
        try:
            selected = select_worker_count(times, rss)
        except ValueError as error:
            return {"worker_selection": {"passed": False, "detail": str(error)}}
        return {
            "worker_identity": {
                "passed": True,
                "detail": "worker revisions and configurations match",
            },
            "worker_semantics": {
                "passed": True,
                "detail": "C04 and 1/2/4-worker fixed-work replay equality passed",
            },
            "worker_selection": {
                "passed": True,
                "detail": f"selected_workers={selected}",
                "selected_workers": selected,
                "selected_run_label": run_by_worker[selected],
                "input_runs": [run_by_worker[worker] for worker in (1, 2, 4)],
                "resource_metrics": {
                    str(worker): {
                        "run_wall_seconds": times[worker],
                        "aggregate_peak_rss_gib": rss[worker],
                        "speedup": times[1] / times[worker],
                    }
                    for worker in (1, 2, 4)
                },
            },
        }
    if component is Stage052Component.ACCELERATOR_PILOT:
        occupancy = sorted(_strict_float(row["median_batch_occupancy"]) for row in rows)
        median = occupancy[len(occupancy) // 2] if occupancy else 0.0
        accelerator_decision = decide_accelerator(median_batch_occupancy=median)
        return {
            "accelerator_decision": {
                "passed": True,
                "detail": accelerator_decision.value,
            }
        }
    return {}


def _load_resource_summary(raw_dir: Path) -> dict[str, object]:
    reader = ArtifactReader(raw_dir)
    reference = _one_artifact(reader, "resource_summary")
    return reader.read_json(str(reference["relative_path"]))


def _load_metadata(raw_dir: Path) -> dict[str, object]:
    reader = ArtifactReader(raw_dir)
    reference = _one_artifact(reader, "manifest_metadata")
    return reader.read_json(str(reference["relative_path"]))


def _optimization_profile_gate(component: Stage052Component, observed: object) -> dict[str, object]:
    expected = (
        "none"
        if component is Stage052Component.PERF_BASELINE
        else "python"
        if component
        in {
            Stage052Component.HOT_PATH,
            Stage052Component.ARTIFACT_STREAMING,
            Stage052Component.JOB_PARALLEL,
        }
        else "native"
    )
    passed = observed == expected
    return {
        "passed": passed,
        "detail": f"expected={expected} observed={observed}",
    }


def _axis_semantics_equal(rows: Sequence[Mapping[str, object]], left: str, right: str) -> bool:
    by_identity = {
        (str(row["instance"]), _strict_int(row["seed"], "seed"), str(row["axis"])): str(
            row["semantic_digest"]
        )
        for row in rows
    }
    identities = {(key[0], key[1]) for key in by_identity if key[2] == left}
    return bool(identities) and all(
        by_identity.get((instance, seed, left)) == by_identity.get((instance, seed, right))
        for instance, seed in identities
    )


def _observations(
    rows: Sequence[Mapping[str, object]], *, axis: str
) -> list[PerformanceObservation]:
    return [
        PerformanceObservation(
            instance=str(row["instance"]),
            seed=_strict_int(row["seed"], "seed"),
            customer_count=_strict_int(row["customer_count"], "customer_count"),
            end_to_end_seconds=_strict_float(row["end_to_end_seconds"]),
            semantic_digest=str(row["semantic_digest"]),
        )
        for row in rows
        if row.get("axis") == axis
    ]


def _storage_observations(
    rows: Sequence[Mapping[str, object]],
    *,
    semantic_digests: Mapping[tuple[str, int, str], str] | None = None,
) -> list[ArtifactStorageObservation]:
    output: list[ArtifactStorageObservation] = []
    observed_identities: set[tuple[str, int, str]] = set()
    for row in rows:
        identity = (
            str(row["instance"]),
            _strict_int(row["seed"], "seed"),
            str(row["axis"]),
        )
        observed_identities.add(identity)
        output.append(
            ArtifactStorageObservation(
                instance=identity[0],
                seed=identity[1],
                axis=identity[2],
                storage_policy_version=str(row["storage_policy_version"]),
                semantic_digest=(
                    semantic_digests[identity]
                    if semantic_digests is not None
                    else str(row["semantic_digest"])
                ),
                artifact_persistence_seconds=_strict_float(row["artifact_persistence_seconds"]),
                end_to_end_seconds=_strict_float(row["end_to_end_seconds"]),
                peak_rss_bytes=_strict_int(row["peak_rss_bytes"], "peak_rss_bytes"),
            )
        )
    if semantic_digests is not None and observed_identities != semantic_digests.keys():
        raise ArtifactIntegrityError("storage replay identity does not match per-run rows")
    return output


def _load_per_run(raw_dir: Path) -> list[dict[str, str]]:
    reader = ArtifactReader(raw_dir)
    item = _one_artifact(reader, "per_run_results")
    return _read_csv(raw_dir / str(item["relative_path"]))


def _one_artifact(reader: ArtifactReader, artifact_type: str) -> Mapping[str, Any]:
    matches = [
        item
        for item in reader.manifest.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("artifact_type") == artifact_type
    ]
    if len(matches) != 1:
        raise ArtifactIntegrityError(f"expected one {artifact_type} artifact")
    return matches[0]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strict_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise TypeError(f"{field} must be an integer")


def _strict_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("numeric value cannot be boolean")
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError("numeric value is invalid")


def _strict_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return False


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _trace_dictionary(trace: Mapping[str, object], field: str, directory: str) -> dict[int, str]:
    raw = trace.get(field)
    if not isinstance(raw, Mapping):
        raise ArtifactIntegrityError(f"trace {field} is missing in {directory}")
    output: dict[int, str] = {}
    for raw_key, raw_value in raw.items():
        key = _strict_int(raw_key, field)
        if key in output:
            raise ArtifactIntegrityError(f"trace {field} contains duplicate IDs")
        output[key] = str(raw_value)
    return output


def _replace_event_route_ids(
    event: dict[str, object],
    *,
    route_dictionary: Mapping[int, Mapping[str, object]],
    directory: str,
) -> None:
    for field in ("route_id", "base_route_id", "candidate_route_id"):
        raw_value = event.get(field)
        if raw_value is None:
            continue
        route_id = _strict_int(raw_value, field)
        route = route_dictionary.get(route_id)
        if route is None:
            raise ArtifactIntegrityError(
                f"event references unknown route ID in {directory}: {route_id}"
            )
        event[field] = dict(route)
    for field in ("route_ids", "current_route_ids", "candidate_route_ids"):
        raw_values = event.get(field)
        if raw_values is None:
            continue
        if not isinstance(raw_values, (list, tuple)):
            raise ArtifactIntegrityError(
                f"event route ID collection is invalid in {directory}: {field}"
            )
        routes: list[dict[str, object]] = []
        for raw_value in raw_values:
            route_id = _strict_int(raw_value, field)
            route = route_dictionary.get(route_id)
            if route is None:
                raise ArtifactIntegrityError(
                    f"event references unknown route ID in {directory}: {route_id}"
                )
            routes.append(dict(route))
        event[field] = routes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Review Stage 5.2 raw evidence")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/schneider"))
    parser.add_argument(
        "--component",
        choices=tuple(component.value for component in Stage052Component),
        required=True,
    )
    parser.add_argument("--scope", choices=("performance", "pilot", "formal"), required=True)
    parser.add_argument("--comparison-dir", type=Path, action="append", default=[])
    parser.add_argument("--prerequisite-dir", type=Path)
    arguments = parser.parse_args()
    outputs = review_stage052(
        raw_dir=arguments.raw_dir,
        benchmark_dir=arguments.benchmark_dir,
        component=arguments.component,
        scope=arguments.scope,
        comparison_dirs=arguments.comparison_dir,
        prerequisite_dir=arguments.prerequisite_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
