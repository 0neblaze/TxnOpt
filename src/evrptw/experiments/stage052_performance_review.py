"""Independent raw replay for Stage 5.2 performance evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import json
import math
import os
import shutil
import statistics
import uuid
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from evrptw import _core as native_core
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
from evrptw.environment import collect_environment
from evrptw.experiments.stage052_performance import (
    PERFORMANCE_INSTANCES,
    PERFORMANCE_SEEDS,
    axes_for_scope,
    load_stage052_config,
    validate_stage052_run_label,
)
from evrptw.native_kernels import NativeKernelConfig
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
from evrptw.stage052_evidence import (
    JobParallelSelectionIdentity,
    Stage052PrerequisiteIdentity,
    abort_process_executor,
    validate_worker_ownership,
    verify_job_parallel_selection,
    verify_stage052_prerequisite,
    verify_stage052_review_files,
)
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
_PERFORMANCE_ENVIRONMENT_VARIABLES = {
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_PLACES",
    "OMP_PROC_BIND",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "VECLIB_MAXIMUM_THREADS",
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
    try:
        verify_stage052_review_files(raw_dir, payload)
    except ArtifactIntegrityError as error:
        raise ValueError(str(error)) from error
    raw_manifest_path = ArtifactReader(raw_dir).result.manifest_path
    if payload.get("raw_manifest_sha256") != _sha256(raw_manifest_path):
        raise ValueError("prerequisite review is stale for the current raw manifest")


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


def replay_stage052_storage_semantics_many(
    raw_dirs: Sequence[Path],
    *,
    max_workers: int = 4,
) -> list[dict[tuple[str, int, str], str]]:
    """Replay independent bundles concurrently without changing per-bundle ordering."""

    if not raw_dirs:
        raise ValueError("at least one Stage 5.2 replay directory is required")
    if max_workers <= 0:
        raise ValueError("replay max_workers must be positive")
    if len(raw_dirs) == 1:
        return [replay_stage052_storage_semantics(raw_dirs[0])]
    worker_count = min(max_workers, len(raw_dirs))
    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_context("spawn"),
    )
    futures: dict[Any, int] = {}
    results: list[dict[tuple[str, int, str], str] | None] = [None] * len(raw_dirs)
    try:
        futures = {
            executor.submit(replay_stage052_storage_semantics, raw_dir): index
            for index, raw_dir in enumerate(raw_dirs)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    except BaseException as error:
        for future in futures:
            future.cancel()
        try:
            abort_process_executor(executor)
        except BaseException as abort_error:
            raise RuntimeError(
                f"replay failure {type(error).__name__}: {error}; "
                f"process-pool abort failure {type(abort_error).__name__}: {abort_error}"
            ) from error
        raise
    executor.shutdown(wait=True)
    if any(result is None for result in results):
        raise RuntimeError("parallel Stage 5.2 replay returned an incomplete result set")
    return [result for result in results if result is not None]


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


def _render_review_findings(gates: Mapping[str, Mapping[str, object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=("gate", "passed", "detail"))
    writer.writeheader()
    for gate, result in gates.items():
        writer.writerow(
            {
                "gate": gate,
                "passed": result.get("passed"),
                "detail": result.get("detail"),
            }
        )
    return output.getvalue().encode("utf-8")


def _render_review_report(
    *,
    run_label: str,
    status: str,
    gates: Mapping[str, Mapping[str, object]],
) -> bytes:
    return "\n".join(
        [
            f"# Stage 5.2 Review — {run_label}",
            "",
            f"**Status: {status}**",
            "",
            *[
                f"- {name}: {'PASS' if result['passed'] else 'FAIL'} — {result['detail']}"
                for name, result in gates.items()
            ],
            "",
        ]
    ).encode("utf-8")


def _write_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_review_generation(
    *,
    review_dir: Path,
    findings: bytes,
    report: bytes,
    manifest: Mapping[str, object],
) -> dict[str, Path]:
    """Publish immutable review files behind one atomic manifest pointer."""

    review_dir.mkdir(parents=True, exist_ok=True)
    generation_id = hashlib.sha256(findings + b"\0" + report).hexdigest()
    generations_dir = review_dir / "generations"
    generations_dir.mkdir(exist_ok=True)
    generation_dir = generations_dir / generation_id
    findings_path = generation_dir / "review_findings.csv"
    report_path = generation_dir / "review_report.md"
    temporary_generation = generations_dir / f".{generation_id}.{uuid.uuid4().hex}.tmp"
    if generation_dir.exists():
        if (
            not generation_dir.is_dir()
            or not findings_path.is_file()
            or findings_path.read_bytes() != findings
            or not report_path.is_file()
            or report_path.read_bytes() != report
            or {path.name for path in generation_dir.iterdir()}
            != {"review_findings.csv", "review_report.md"}
        ):
            raise ArtifactIntegrityError("review generation identity collision")
    else:
        temporary_generation.mkdir()
        try:
            _write_fsync(temporary_generation / findings_path.name, findings)
            _write_fsync(temporary_generation / report_path.name, report)
            _fsync_directory(temporary_generation)
            os.replace(temporary_generation, generation_dir)
            _fsync_directory(generations_dir)
        finally:
            if temporary_generation.exists():
                shutil.rmtree(temporary_generation)

    manifest_payload = dict(manifest)
    manifest_payload["files"] = {
        findings_path.relative_to(review_dir).as_posix(): hashlib.sha256(findings).hexdigest(),
        report_path.relative_to(review_dir).as_posix(): hashlib.sha256(report).hexdigest(),
    }
    manifest_bytes = (json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path = review_dir / "review_manifest.json"
    temporary_manifest = review_dir / f".review_manifest.{uuid.uuid4().hex}.tmp"
    try:
        _write_fsync(temporary_manifest, manifest_bytes)
        os.replace(temporary_manifest, manifest_path)
        _fsync_directory(review_dir)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return {
        "review_report": report_path,
        "review_findings": findings_path,
        "review_manifest": manifest_path,
    }


def _archive_prior_review_generation(
    raw_dir: Path,
    *,
    manifest_path: Path,
    manifest_payload: Mapping[str, object],
) -> str:
    """Archive the accepted prior review before its atomic pointer is superseded."""

    prior_sha256 = _sha256(manifest_path)
    verified_files = verify_stage052_review_files(raw_dir, manifest_payload)
    archived_payloads = {
        "review_manifest.json": manifest_path.read_bytes(),
        **{relative: path.read_bytes() for relative, path in verified_files.items()},
    }
    history_dir = raw_dir / "review" / "history"
    history_dir.mkdir(exist_ok=True)
    archive_dir = history_dir / prior_sha256
    if archive_dir.exists():
        if (
            not archive_dir.is_dir()
            or {
                path.relative_to(archive_dir).as_posix()
                for path in archive_dir.rglob("*")
                if path.is_file()
            }
            != set(archived_payloads)
            or any(
                (archive_dir / name).read_bytes() != payload
                for name, payload in archived_payloads.items()
            )
        ):
            raise ArtifactIntegrityError("prior review archive identity collision")
        return prior_sha256
    temporary = history_dir / f".{prior_sha256}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        for name, payload in archived_payloads.items():
            (temporary / name).parent.mkdir(parents=True, exist_ok=True)
            _write_fsync(temporary / name, payload)
        _fsync_directory(temporary)
        os.replace(temporary, archive_dir)
        _fsync_directory(history_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return prior_sha256


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
    review_lineage = _prior_review_manifest_hashes(raw_dir)
    reader = ArtifactReader(raw_dir)
    manifest = reader.manifest
    if manifest.get("evidence_completeness") != "complete":
        raise ArtifactIntegrityError("partial Stage 5.2 evidence cannot be reviewed")
    if selected is Stage052Component.ACCELERATOR_PILOT and any(
        isinstance(item, dict) and item.get("artifact_type") == "accelerator_decision"
        for item in manifest.get("artifacts", [])
    ):
        return _review_accelerator_decision_only(
            raw_dir=raw_dir,
            reader=reader,
            benchmark_dir=benchmark_dir,
            scope=scope,
            prerequisite_dir=prerequisite_dir,
        )
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
    prerequisite_identity: Stage052PrerequisiteIdentity | None = None
    job_parallel_selection: JobParallelSelectionIdentity | None = None
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
                bound_prerequisite = metadata.get("component_prerequisite")
                binding_passed = _prerequisite_binding_matches(
                    bound_prerequisite,
                    prerequisite_identity,
                    prerequisite_dir,
                )
                gates["component_prerequisite"] = {
                    "passed": binding_passed,
                    "detail": (
                        prerequisite_identity.run_label
                        if binding_passed
                        else "producer metadata is not bound to the reviewed prerequisite"
                    ),
                }
                if selected is Stage052Component.NATIVE_KERNELS:
                    try:
                        job_parallel_selection = verify_job_parallel_selection(
                            prerequisite_dir, prerequisite_identity
                        )
                    except (ArtifactIntegrityError, ValueError) as error:
                        gates["job_parallel_selection"] = {
                            "passed": False,
                            "detail": str(error),
                        }
                    else:
                        selected_worker_rows = {
                            _strict_int(row["worker_count"], "worker_count") for row in rows
                        }
                        binding_passed = metadata.get(
                            "job_parallel_selection"
                        ) == job_parallel_selection.to_dict() and selected_worker_rows == {
                            job_parallel_selection.selected_workers
                        }
                        gates["job_parallel_selection"] = {
                            "passed": binding_passed,
                            "detail": (
                                f"selected_workers={job_parallel_selection.selected_workers}; "
                                f"selected_run={job_parallel_selection.selected_run_label}"
                                if binding_passed
                                else "native producer is not bound to D selected workers/run"
                            ),
                        }
    gates.update(
        _component_gates(
            selected,
            rows,
            raw_dir=raw_dir,
            comparison_dirs=comparison_dirs,
            prerequisite_dir=prerequisite_dir,
            prerequisite_identity=prerequisite_identity,
            job_parallel_selection=job_parallel_selection,
            benchmark_dir=benchmark_dir,
        )
    )
    passed = all(bool(gate["passed"]) for gate in gates.values())
    status = _NEXT_STATUS[selected] if passed else NOT_READY
    if selected is Stage052Component.JOB_PARALLEL and not passed:
        worker_gate = gates.get("worker_selection")
        if isinstance(worker_gate, dict) and worker_gate.get("passed") is True:
            gates["worker_selection"] = {
                "passed": True,
                "detail": (
                    "worker thresholds passed; selection withheld because review is NOT_READY"
                ),
            }
    review_dir = raw_dir / "review"
    review_manifest: dict[str, object] = {
        "schema_version": STAGE052_REVIEW_SCHEMA_VERSION,
        "run_label": raw_dir.name,
        "component": selected.value,
        "scope": scope,
        "status": status,
        "raw_manifest_sha256": _sha256(reader.result.manifest_path),
        "review_manifest_lineage_sha256": review_lineage,
        "gates": gates,
    }
    worker_gate = gates.get("worker_selection", {})
    if passed and selected is Stage052Component.JOB_PARALLEL and worker_gate.get("passed") is True:
        review_manifest["selected_workers"] = worker_gate.get("selected_workers")
        review_manifest["selected_run_label"] = worker_gate.get("selected_run_label")
        review_manifest["input_runs"] = worker_gate.get("input_runs")
        review_manifest["input_raw_manifest_sha256"] = worker_gate.get("input_raw_manifest_sha256")
        review_manifest["resource_metrics"] = worker_gate.get("resource_metrics")
    if (
        passed
        and selected is Stage052Component.NATIVE_KERNELS
        and job_parallel_selection is not None
    ):
        review_manifest["selected_workers"] = job_parallel_selection.selected_workers
        review_manifest["performance_predecessor"] = job_parallel_selection.selected_run_label
        review_manifest["native_configuration"] = NativeKernelConfig().to_dict()
    return _publish_review_generation(
        review_dir=review_dir,
        findings=_render_review_findings(gates),
        report=_render_review_report(run_label=raw_dir.name, status=status, gates=gates),
        manifest=review_manifest,
    )


def _review_accelerator_decision_only(
    *,
    raw_dir: Path,
    reader: ArtifactReader,
    benchmark_dir: Path,
    scope: str,
    prerequisite_dir: Path | None,
) -> dict[str, Path]:
    """Independently recompute a below-threshold F decision without GPU rows."""

    del benchmark_dir
    review_lineage = _prior_review_manifest_hashes(raw_dir)
    gates: dict[str, dict[str, object]] = {}
    artifacts = [item for item in reader.manifest.get("artifacts", []) if isinstance(item, dict)]
    artifact_types = [str(item.get("artifact_type", "")) for item in artifacts]
    allowed = {"manifest_metadata", "config", "accelerator_decision"}
    mutually_exclusive = (
        set(artifact_types) == allowed
        and artifact_types.count("manifest_metadata") == 1
        and artifact_types.count("config") == 1
        and artifact_types.count("accelerator_decision") == 1
    )
    gates["mutually_exclusive_schema"] = {
        "passed": mutually_exclusive,
        "detail": (
            "decision-only artifact set contains no solver/GPU rows"
            if mutually_exclusive
            else f"decision-only artifact set is invalid: {artifact_types}"
        ),
    }
    metadata = reader.read_json(str(_one_artifact(reader, "manifest_metadata")["relative_path"]))
    metadata_passed = (
        scope == "performance"
        and metadata.get("scope") == "performance"
        and metadata.get("component") == Stage052Component.ACCELERATOR_PILOT.value
        and metadata.get("optimization_profile") == "native"
        and metadata.get("native_kernel_config") == NativeKernelConfig().to_dict()
        and metadata.get("accelerator_decision_mode") == "decision_only"
    )
    gates["decision_metadata"] = {
        "passed": metadata_passed,
        "detail": "decision-only native metadata passed" if metadata_passed else "invalid metadata",
    }
    prerequisite_identity: Stage052PrerequisiteIdentity | None = None
    if prerequisite_dir is None:
        gates["component_prerequisite"] = {
            "passed": False,
            "detail": "accelerator decision requires accepted E prerequisite",
        }
    else:
        try:
            prerequisite_identity = verify_stage052_prerequisite(
                prerequisite_dir,
                expected_component=Stage052Component.NATIVE_KERNELS.value,
                expected_status="READY_FOR_STAGE052_ACCELERATOR_DECISION",
            )
        except (ArtifactIntegrityError, ValueError) as error:
            gates["component_prerequisite"] = {"passed": False, "detail": str(error)}
        else:
            binding_passed = _prerequisite_binding_matches(
                metadata.get("component_prerequisite"),
                prerequisite_identity,
                prerequisite_dir,
            )
            gates["component_prerequisite"] = {
                "passed": binding_passed,
                "detail": (
                    prerequisite_identity.run_label
                    if binding_passed
                    else "decision metadata is not bound to accepted E"
                ),
            }
    decision = reader.read_json(str(_one_artifact(reader, "accelerator_decision")["relative_path"]))
    expected_keys = {
        "schema_version",
        "decision_mode",
        "decision",
        "threshold",
        "median_batch_occupancy",
        "input_count",
        "inputs",
        "native_prerequisite",
        "gpu_rows_present",
        "fallback_used",
    }
    schema_passed = (
        set(decision) == expected_keys
        and decision.get("schema_version") == "stage05.2-accelerator-decision-v1"
        and decision.get("decision_mode") == "decision_only"
        and decision.get("decision") == "GPU_NOT_JUSTIFIED"
        and decision.get("threshold") == 32.0
        and decision.get("input_count") == 9
        and decision.get("gpu_rows_present") is False
        and decision.get("fallback_used") is False
        and prerequisite_identity is not None
        and decision.get("native_prerequisite") == prerequisite_identity.to_dict()
    )
    gates["decision_schema"] = {
        "passed": schema_passed,
        "detail": "exclusive decision-only schema passed" if schema_passed else "invalid schema",
    }
    recomputed_inputs: list[dict[str, object]] = []
    recomputed_median = math.inf
    if prerequisite_identity is not None and prerequisite_dir is not None:
        try:
            recomputed_inputs, recomputed_median = _recompute_native_occupancies(prerequisite_dir)
        except (ArtifactIntegrityError, KeyError, TypeError, ValueError) as error:
            gates["occupancy_recomputation"] = {
                "passed": False,
                "detail": str(error),
            }
        else:
            observed_median = _strict_float(decision.get("median_batch_occupancy"))
            occupancy_passed = (
                decision.get("inputs") == recomputed_inputs
                and math.isclose(observed_median, recomputed_median, rel_tol=0.0, abs_tol=1e-12)
                and recomputed_median < 32.0
            )
            gates["occupancy_recomputation"] = {
                "passed": occupancy_passed,
                "detail": (
                    f"median={recomputed_median:.12g} < 32"
                    if occupancy_passed
                    else "decision inputs/median do not match accepted E evidence"
                ),
            }
    else:
        gates["occupancy_recomputation"] = {
            "passed": False,
            "detail": "accepted E evidence is unavailable",
        }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    status = _NEXT_STATUS[Stage052Component.ACCELERATOR_PILOT] if passed else NOT_READY
    review_dir = raw_dir / "review"
    manifest = {
        "schema_version": STAGE052_REVIEW_SCHEMA_VERSION,
        "run_label": raw_dir.name,
        "component": Stage052Component.ACCELERATOR_PILOT.value,
        "scope": scope,
        "status": status,
        "raw_manifest_sha256": _sha256(reader.result.manifest_path),
        "review_manifest_lineage_sha256": review_lineage,
        "accelerator_decision": "GPU_NOT_JUSTIFIED" if passed else "NOT_READY",
        "selected_backend": "native_cpu" if passed else None,
        "gates": gates,
    }
    return _publish_review_generation(
        review_dir=review_dir,
        findings=_render_review_findings(gates),
        report=_render_review_report(run_label=raw_dir.name, status=status, gates=gates),
        manifest=manifest,
    )


def _recompute_native_occupancies(raw_dir: Path) -> tuple[list[dict[str, object]], float]:
    reader = ArtifactReader(raw_dir)
    expected = {
        (instance, seed)
        for instance in ("c101_21", "r101_21", "rc101_21")
        for seed in PERFORMANCE_SEEDS
    }
    expected_all = {
        (instance, seed) for instance in PERFORMANCE_INSTANCES for seed in PERFORMANCE_SEEDS
    }
    values: dict[tuple[str, int], float] = {}
    observed_all: set[tuple[str, int]] = set()
    references = [
        item
        for item in reader.manifest.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("artifact_type") == "raw"
    ]
    for reference in references:
        raw = reader.read_json(str(reference.get("relative_path", "")))
        if not isinstance(raw, Mapping):
            raise ArtifactIntegrityError("E occupancy raw payload must be an object")
        identity = (str(raw.get("instance", "")), _strict_int(raw.get("seed"), "seed"))
        if identity in observed_all:
            raise ArtifactIntegrityError(f"duplicate E raw occupancy identity: {identity}")
        observed_all.add(identity)
        if (
            raw.get("component") != Stage052Component.NATIVE_KERNELS.value
            or raw.get("scope") != "performance"
        ):
            raise ArtifactIntegrityError(f"invalid E raw occupancy source: {identity}")
        if identity not in expected:
            continue
        axes = raw.get("axes")
        fixed = axes.get("fixed_work") if isinstance(axes, Mapping) else None
        if (
            not isinstance(fixed, Mapping)
            or fixed.get("validator_passed") is not True
            or fixed.get("valid") is not True
        ):
            raise ArtifactIntegrityError(f"invalid E fixed-work raw axis: {identity}")
        backend = fixed.get("backend_metrics")
        if not isinstance(backend, Mapping):
            raise ArtifactIntegrityError(f"missing E raw backend metrics: {identity}")
        exact_calls = _strict_int(backend.get("exact_calls"), "exact_calls")
        batch_launches = _strict_int(backend.get("batch_launches"), "batch_launches")
        raw_occupancies = backend.get("launch_occupancies")
        if not isinstance(raw_occupancies, list):
            raise ArtifactIntegrityError(f"missing E raw launch occupancies: {identity}")
        occupancies = [_strict_int(value, "launch_occupancy") for value in raw_occupancies]
        if (
            exact_calls <= 0
            or batch_launches <= 0
            or any(value <= 0 for value in occupancies)
            or len(occupancies) != batch_launches
            or sum(occupancies) != exact_calls
        ):
            raise ArtifactIntegrityError(f"invalid E raw occupancy counters: {identity}")
        values[identity] = statistics.median(occupancies)
    if observed_all != expected_all:
        raise ArtifactIntegrityError("accepted E raw shard scope is not exactly 12 bundles")
    if set(values) != expected:
        raise ArtifactIntegrityError("accepted E occupancy scope is not exactly 9 values")
    inputs = [
        {
            "instance": instance,
            "seed": seed,
            "axis": "fixed_work",
            "median_batch_occupancy": values[(instance, seed)],
        }
        for instance, seed in sorted(values)
    ]
    return inputs, statistics.median(values.values())


def _validate_native_shard_timing_order(
    shard: tuple[str, int],
    shard_timings: Sequence[Mapping[str, object]],
) -> None:
    declared_order = tuple(axis.name for axis in axes_for_scope("performance"))
    by_axis = {str(timing.get("axis", "")): timing for timing in shard_timings}
    if len(by_axis) != len(shard_timings) or not set(by_axis).issubset(declared_order):
        raise ArtifactIntegrityError(f"native shard timing axis identity is invalid: {shard}")
    ordered = [by_axis[name] for name in declared_order if name in by_axis]
    for previous, current in zip(ordered, ordered[1:], strict=False):
        previous_completed = _strict_int(previous.get("axis_completed_ns"), "axis_completed_ns")
        current_started = _strict_int(current.get("axis_started_ns"), "axis_started_ns")
        if previous_completed > current_started:
            raise ArtifactIntegrityError(f"native shard timing axes overlap: {shard}")
    finalization_starts = {
        _strict_int(timing.get("finalize_started_ns"), "finalize_started_ns") for timing in ordered
    }
    if len(finalization_starts) != 1 or (
        ordered
        and max(
            _strict_int(timing.get("axis_completed_ns"), "axis_completed_ns") for timing in ordered
        )
        > next(iter(finalization_starts))
    ):
        raise ArtifactIntegrityError(f"native shard finalization precedes axis completion: {shard}")


def _audit_native_execution(
    raw_dir: Path,
    rows: Sequence[Mapping[str, object]],
) -> tuple[bool, str]:
    """Cross-check native counters against raw backend and trace-derived work."""

    try:
        reader = ArtifactReader(raw_dir)
        manifest = reader.manifest
        if (
            manifest.get("run_label") != raw_dir.name
            or manifest.get("component") != Stage052Component.NATIVE_KERNELS.value
            or manifest.get("status") != "complete"
            or manifest.get("evidence_completeness") != "complete"
            or manifest.get("storage_policy_version") != "artifact-storage-v2"
            or manifest.get("artifact_status") != {"failure": "not_applicable"}
        ):
            raise ArtifactIntegrityError("native parent manifest is not complete and canonical")
        if any(
            isinstance(item, Mapping) and item.get("artifact_type") == "failure"
            for item in manifest.get("artifacts", [])
        ):
            raise ArtifactIntegrityError("native evidence contains a shard failure artifact")
        row_by_axis: dict[tuple[str, int, str], Mapping[str, object]] = {}
        for row in rows:
            identity = (
                str(row["instance"]),
                _strict_int(row["seed"], "seed"),
                str(row["axis"]),
            )
            if identity in row_by_axis:
                raise ArtifactIntegrityError(f"duplicate native per-run identity: {identity}")
            row_by_axis[identity] = row

        payloads: dict[str, dict[tuple[str, int], Mapping[str, object]]] = {
            "raw": {},
            "solution": {},
            "trace": {},
        }
        for artifact_type, by_shard in payloads.items():
            references = [
                item
                for item in reader.manifest.get("artifacts", [])
                if isinstance(item, Mapping) and item.get("artifact_type") == artifact_type
            ]
            for reference in references:
                relative = str(reference.get("relative_path", ""))
                parts = Path(relative).parts
                shard_identity = (parts[0], _strict_int(parts[1], "seed"))
                if shard_identity in by_shard:
                    raise ArtifactIntegrityError(
                        f"duplicate native {artifact_type} shard: {shard_identity}"
                    )
                payload = reader.read_json(relative)
                if not isinstance(payload, Mapping):
                    raise ArtifactIntegrityError(
                        f"native {artifact_type} payload is not an object: {shard_identity}"
                    )
                by_shard[shard_identity] = payload
        expected_shards = {(identity[0], identity[1]) for identity in row_by_axis}
        canonical_ordinals = {
            identity: ordinal
            for ordinal, identity in enumerate(
                (instance, seed) for instance in PERFORMANCE_INSTANCES for seed in PERFORMANCE_SEEDS
            )
        }
        if any(set(by_shard) != expected_shards for by_shard in payloads.values()):
            raise ArtifactIntegrityError("native raw/trace shard scope mismatch")

        timing_reference = _one_artifact(reader, "timing_evidence")
        timing_payload = reader.read_json(str(timing_reference["relative_path"]))
        raw_timing_rows = timing_payload.get("rows")
        if (
            timing_payload.get("schema_version") != "stage05.2-timing-evidence-v1"
            or timing_payload.get("run_label") != raw_dir.name
            or timing_payload.get("component") != Stage052Component.NATIVE_KERNELS.value
            or not isinstance(raw_timing_rows, list)
        ):
            raise ArtifactIntegrityError("native timing evidence identity is invalid")
        timing_by_axis: dict[tuple[str, int, str], Mapping[str, object]] = {}
        for timing in raw_timing_rows:
            if not isinstance(timing, Mapping):
                raise ArtifactIntegrityError("native timing evidence row is invalid")
            timing_identity = (
                str(timing.get("instance", "")),
                _strict_int(timing.get("seed"), "seed"),
                str(timing.get("axis", "")),
            )
            if timing_identity in timing_by_axis:
                raise ArtifactIntegrityError(f"duplicate native timing identity: {timing_identity}")
            timing_by_axis[timing_identity] = timing
        if set(timing_by_axis) != set(row_by_axis):
            raise ArtifactIntegrityError("native timing/per-run scope mismatch")
        for shard in expected_shards:
            shard_timings = [
                timing for identity, timing in timing_by_axis.items() if identity[:2] == shard
            ]
            _validate_native_shard_timing_order(shard, shard_timings)
            totals = {
                (
                    _strict_int(timing.get("finalize_started_ns"), "finalize_started_ns"),
                    _strict_int(timing.get("finalize_completed_ns"), "finalize_completed_ns"),
                    _strict_int(timing.get("total_event_count"), "total_event_count"),
                    _strict_int(timing.get("axis_count"), "axis_count"),
                )
                for timing in shard_timings
            }
            if len(totals) != 1:
                raise ArtifactIntegrityError(f"native shard timing totals disagree: {shard}")
            _start, _completed, total_events, axis_count = next(iter(totals))
            if axis_count != len(shard_timings) or total_events != sum(
                _strict_int(timing.get("axis_event_count"), "axis_event_count")
                for timing in shard_timings
            ):
                raise ArtifactIntegrityError(f"native shard timing allocation is invalid: {shard}")

        audited: set[tuple[str, int, str]] = set()
        for shard in sorted(expected_shards):
            raw_payload = payloads["raw"][shard]
            solution_payload = payloads["solution"][shard]
            trace_payload = payloads["trace"][shard]
            event_identity = trace_payload.get("event_identity")
            if (
                raw_payload.get("run_label") != raw_dir.name
                or raw_payload.get("component") != Stage052Component.NATIVE_KERNELS.value
                or raw_payload.get("scope") != "performance"
                or raw_payload.get("instance") != shard[0]
                or raw_payload.get("seed") != shard[1]
                or solution_payload.get("instance") != shard[0]
                or solution_payload.get("seed") != shard[1]
                or not isinstance(event_identity, Mapping)
                or event_identity
                != {
                    "shard_ordinal": canonical_ordinals.get(shard),
                    "local_field": "event_id",
                }
            ):
                raise ArtifactIntegrityError(
                    f"native shard producer or trace event identity mismatch: {shard}"
                )
            raw_axes = raw_payload.get("axes")
            solution_axes = solution_payload.get("axes")
            trace_axes = trace_payload.get("axes")
            if not all(
                isinstance(value, Mapping) for value in (raw_axes, solution_axes, trace_axes)
            ):
                raise ArtifactIntegrityError(f"native raw/solution/trace axes are missing: {shard}")
            assert isinstance(raw_axes, Mapping)
            assert isinstance(solution_axes, Mapping)
            assert isinstance(trace_axes, Mapping)
            if set(raw_axes) != set(solution_axes) or set(raw_axes) != set(trace_axes):
                raise ArtifactIntegrityError(f"native raw/solution/trace axis mismatch: {shard}")
            for axis, raw_axis in raw_axes.items():
                axis_identity = (shard[0], shard[1], str(axis))
                current_row = row_by_axis.get(axis_identity)
                solution_axis = solution_axes.get(axis)
                trace_axis = trace_axes.get(axis)
                if (
                    current_row is None
                    or not isinstance(raw_axis, Mapping)
                    or not isinstance(solution_axis, Mapping)
                    or not isinstance(trace_axis, Mapping)
                ):
                    raise ArtifactIntegrityError(
                        f"native axis evidence is incomplete: {axis_identity}"
                    )
                reconciliation = raw_axis.get("trace_reconciliation")
                checks = (
                    reconciliation.get("checks") if isinstance(reconciliation, Mapping) else None
                )
                if (
                    raw_axis.get("valid") is not True
                    or raw_axis.get("validator_passed") is not True
                    or solution_axis.get("feasible") is not True
                    or raw_axis.get("objective_key") != solution_axis.get("objective_key")
                    or not isinstance(reconciliation, Mapping)
                    or reconciliation.get("status") != "pass"
                    or not isinstance(checks, Mapping)
                    or not checks
                    or any(value is not True for value in checks.values())
                ):
                    raise ArtifactIntegrityError(
                        f"native validity/trace reconciliation failed: {axis_identity}"
                    )
                backend = raw_axis.get("backend_metrics")
                result_summary = trace_axis.get("result_summary")
                if not isinstance(backend, Mapping) or not isinstance(result_summary, Mapping):
                    raise ArtifactIntegrityError(
                        f"native backend/result summary is missing: {axis_identity}"
                    )
                screening = result_summary.get("screening_statistics")
                incremental = result_summary.get("cache_incremental_statistics")
                if not isinstance(screening, Mapping) or not isinstance(incremental, Mapping):
                    raise ArtifactIntegrityError(
                        f"native screening/incremental summary is missing: {axis_identity}"
                    )

                exact_invocations = _strict_int(
                    backend.get("native_invocations"), "native_invocations"
                )
                exact_calls = _strict_int(backend.get("exact_calls"), "exact_calls")
                work_batches = _strict_int(backend.get("work_batches"), "work_batches")
                batch_launches = _strict_int(backend.get("batch_launches"), "batch_launches")
                raw_occupancies = backend.get("launch_occupancies")
                if not isinstance(raw_occupancies, list):
                    raise ArtifactIntegrityError(
                        f"native launch occupancies are missing: {axis_identity}"
                    )
                launch_occupancies = [
                    _strict_int(value, "launch_occupancy") for value in raw_occupancies
                ]
                exact_fallbacks = _strict_int(backend.get("native_fallbacks"), "native_fallbacks")
                exact_seconds = _strict_float(backend.get("native_kernel_seconds"))
                if (
                    exact_invocations <= 0
                    or exact_calls <= 0
                    or exact_invocations != work_batches
                    or exact_invocations != batch_launches
                    or len(launch_occupancies) != batch_launches
                    or any(value <= 0 for value in launch_occupancies)
                    or sum(launch_occupancies) != exact_calls
                    or exact_fallbacks != 0
                    or exact_seconds <= 0.0
                ):
                    raise ArtifactIntegrityError(
                        f"native exact counters do not reconcile: {axis_identity}"
                    )

                screening_calls = _strict_int(screening.get("screening_calls"), "screening_calls")
                screening_cache_hits = _strict_int(
                    screening.get("screening_cache_hits"), "screening_cache_hits"
                )
                screen_invocations = _strict_int(
                    screening.get("native_screening_invocations"),
                    "native_screening_invocations",
                )
                screen_seconds = _strict_float(screening.get("native_screening_seconds"))
                expected_screen_invocations = screening_calls - screening_cache_hits
                if (
                    screen_invocations <= 0
                    or screen_invocations != expected_screen_invocations
                    or screen_seconds <= 0.0
                ):
                    raise ArtifactIntegrityError(
                        f"native screening counters do not reconcile: {axis_identity}"
                    )

                propagation_invocations = _strict_int(
                    screening.get("native_propagation_invocations"),
                    "native_propagation_invocations",
                )
                propagation_seconds = _strict_float(screening.get("native_propagation_seconds"))
                expected_propagation_invocations = _strict_int(
                    incremental.get("incremental_propagations"),
                    "incremental_propagations",
                ) + _strict_int(
                    incremental.get("incremental_fallbacks"),
                    "incremental_fallbacks",
                )
                protocol_fallbacks = _strict_int(
                    screening.get("native_protocol_fallbacks"),
                    "native_protocol_fallbacks",
                )
                if (
                    propagation_invocations <= 0
                    or propagation_invocations != expected_propagation_invocations
                    or propagation_seconds <= 0.0
                    or protocol_fallbacks != 0
                ):
                    raise ArtifactIntegrityError(
                        f"native propagation counters do not reconcile: {axis_identity}"
                    )

                integer_bindings = {
                    "native_invocations": exact_invocations,
                    "native_fallbacks": exact_fallbacks,
                    "native_screening_invocations": screen_invocations,
                    "native_propagation_invocations": propagation_invocations,
                    "native_protocol_fallbacks": protocol_fallbacks,
                    "batch_launches": work_batches,
                    "exact_started_calls": exact_calls,
                    "exact_completed_calls": _strict_int(
                        backend.get("completed_calls"), "completed_calls"
                    ),
                }
                raw_trace_bindings = {
                    "started_calls": result_summary.get("exact_started_calls"),
                    "completed_calls": result_summary.get("exact_completed_calls"),
                    "effective_iterations": result_summary.get("effective_iterations"),
                    "termination_reason": result_summary.get("termination_reason"),
                }
                if any(
                    raw_axis.get(field) != expected
                    for field, expected in raw_trace_bindings.items()
                ):
                    raise ArtifactIntegrityError(
                        f"native raw/trace summary mismatch: {axis_identity}"
                    )
                float_bindings = {
                    "native_kernel_seconds": exact_seconds,
                    "native_screening_seconds": screen_seconds,
                    "native_propagation_seconds": propagation_seconds,
                    "median_batch_occupancy": statistics.median(launch_occupancies),
                }
                timing = timing_by_axis[axis_identity]
                solver_started_ns = _strict_int(
                    timing.get("solver_started_ns"), "solver_started_ns"
                )
                solver_completed_ns = _strict_int(
                    timing.get("solver_completed_ns"), "solver_completed_ns"
                )
                axis_started_ns = _strict_int(timing.get("axis_started_ns"), "axis_started_ns")
                axis_completed_ns = _strict_int(
                    timing.get("axis_completed_ns"), "axis_completed_ns"
                )
                finalize_started_ns = _strict_int(
                    timing.get("finalize_started_ns"), "finalize_started_ns"
                )
                finalize_completed_ns = _strict_int(
                    timing.get("finalize_completed_ns"), "finalize_completed_ns"
                )
                axis_event_count = _strict_int(timing.get("axis_event_count"), "axis_event_count")
                total_event_count = _strict_int(
                    timing.get("total_event_count"), "total_event_count"
                )
                axis_count = _strict_int(timing.get("axis_count"), "axis_count")
                if (
                    axis_started_ns != solver_started_ns
                    or solver_completed_ns <= solver_started_ns
                    or axis_completed_ns < solver_completed_ns
                    or finalize_completed_ns <= finalize_started_ns
                    or axis_event_count < 0
                    or total_event_count < 0
                    or axis_count <= 0
                ):
                    raise ArtifactIntegrityError(
                        f"native monotonic timing interval is invalid: {axis_identity}"
                    )
                solver_seconds = (solver_completed_ns - solver_started_ns) / 1_000_000_000
                post_solver_seconds = (axis_completed_ns - solver_completed_ns) / 1_000_000_000
                finalization_seconds = (finalize_completed_ns - finalize_started_ns) / 1_000_000_000
                finalization_share = (
                    finalization_seconds * axis_event_count / total_event_count
                    if total_event_count
                    else finalization_seconds / axis_count
                )
                persistence_seconds = post_solver_seconds + finalization_share
                float_bindings.update(
                    {
                        "solver_seconds": solver_seconds,
                        "artifact_persistence_seconds": persistence_seconds,
                        "end_to_end_seconds": (
                            (axis_completed_ns - axis_started_ns) / 1_000_000_000
                            + finalization_share
                        ),
                    }
                )
                if any(
                    _strict_int(current_row.get(field), field) != expected
                    for field, expected in integer_bindings.items()
                ) or any(
                    not math.isclose(
                        _strict_float(current_row.get(field)),
                        expected,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    for field, expected in float_bindings.items()
                ):
                    raise ArtifactIntegrityError(
                        f"native per-run/raw/trace binding mismatch: {axis_identity}"
                    )
                audited.add(axis_identity)
        if audited != set(row_by_axis):
            raise ArtifactIntegrityError("native execution audit did not cover every axis")
    except (ArtifactIntegrityError, IndexError, KeyError, TypeError, ValueError) as error:
        return False, str(error)
    return (
        True,
        f"{len(audited)} axes reconcile per-run, raw backend, and trace-derived native work",
    )


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
    prerequisite_identity: Stage052PrerequisiteIdentity | None,
    job_parallel_selection: JobParallelSelectionIdentity | None,
    benchmark_dir: Path,
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
    if component in {Stage052Component.HOT_PATH, Stage052Component.NATIVE_KERNELS}:
        if len(comparison_dirs) != 1:
            return {
                "performance_promotion": {
                    "passed": False,
                    "detail": "one predecessor required",
                }
            }
        if component is Stage052Component.NATIVE_KERNELS:
            if job_parallel_selection is None:
                return {
                    "native_configuration": {
                        "passed": False,
                        "detail": "accepted D worker selection is required",
                    }
                }
            comparison = comparison_dirs[0].resolve()
            expected_comparison = (
                prerequisite_dir.resolve().parent / job_parallel_selection.selected_run_label
                if prerequisite_dir is not None
                else None
            )
            if (
                prerequisite_dir is None
                or expected_comparison is None
                or comparison != expected_comparison
            ):
                return {
                    "native_predecessor": {
                        "passed": False,
                        "detail": (
                            "comparison path is not the exact reviewed D selected prerequisite"
                        ),
                    }
                }
            selected_index = job_parallel_selection.input_runs.index(
                job_parallel_selection.selected_run_label
            )
            if (
                _sha256(ArtifactReader(comparison).result.manifest_path)
                != (job_parallel_selection.input_raw_manifest_sha256[selected_index])
            ):
                return {
                    "native_predecessor": {
                        "passed": False,
                        "detail": "selected D raw manifest is not bound to its selection review",
                    }
                }
            metadata = _load_metadata(raw_dir)
            if metadata.get("native_kernel_config") != NativeKernelConfig().to_dict():
                return {
                    "native_configuration": {
                        "passed": False,
                        "detail": "complete opt-in native kernel configuration is required",
                    }
                }
            if prerequisite_identity is None:
                return {
                    "native_producer_contract": {
                        "passed": False,
                        "detail": "reviewed D prerequisite identity is missing",
                    }
                }
            producer_passed, producer_detail = _validate_native_evidence_contract(
                raw_dir,
                metadata=metadata,
                prerequisite=prerequisite_identity,
                selection=job_parallel_selection,
                benchmark_dir=benchmark_dir,
            )
            if not producer_passed:
                return {
                    "native_producer_contract": {
                        "passed": False,
                        "detail": producer_detail,
                    }
                }
            observable_native, native_detail = _audit_native_execution(raw_dir, rows)
            if not observable_native:
                return {
                    "native_execution": {
                        "passed": False,
                        "detail": native_detail,
                    }
                }
            replay_maps = replay_stage052_storage_semantics_many((comparison, raw_dir))
            fixed_identities = {
                identity for identity in replay_maps[0] if identity[2].startswith("fixed_work")
            }
            replay_equal = (
                len(fixed_identities) == 24
                and {
                    identity for identity in replay_maps[1] if identity[2].startswith("fixed_work")
                }
                == fixed_identities
                and all(
                    replay_maps[0][identity] == replay_maps[1][identity]
                    for identity in fixed_identities
                )
            )
            if not replay_equal:
                return {
                    "native_fixed_work_differential": {
                        "passed": False,
                        "detail": "24 fixed-work axes do not replay identically",
                    }
                }
        previous = _observations(_load_per_run(comparison_dirs[0]), axis="fixed_work")
        candidate = _observations(rows, axis="fixed_work")
        decision = evaluate_promotion(previous, candidate)
        gates = {
            "performance_promotion": {
                "passed": decision.passed,
                "detail": (
                    f"{decision.detail}; aggregate={decision.aggregate_median_saving:.6f}; "
                    f"families={dict(decision.family_median_savings)}"
                ),
            }
        }
        if component is Stage052Component.NATIVE_KERNELS:
            gates["native_fixed_work_differential"] = {
                "passed": True,
                "detail": "D selected run and native candidate replay equally on 24 axes",
            }
            gates["native_configuration"] = {
                "passed": True,
                "detail": NativeKernelConfig().abi_version,
            }
            gates["native_execution"] = {
                "passed": True,
                "detail": native_detail,
            }
            gates["native_producer_contract"] = {
                "passed": True,
                "detail": producer_detail,
            }
        return gates
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
        provenance_signatures: set[str] = set()
        raw_manifest_hashes: dict[str, str] = {}
        owners_by_worker: dict[int, tuple[int, ...]] = {}
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
            contract_passed, contract_detail = validate_job_parallel_evidence_contract(
                evidence_dir,
                expected_workers=worker,
                expected_prerequisite=prerequisite_identity,
                benchmark_dir=benchmark_dir,
            )
            if not contract_passed:
                return {
                    "comparison_completeness": {
                        "passed": False,
                        "detail": f"{evidence_dir.name}: {contract_detail}",
                    }
                }
            ownership_passed, ownership_detail, owners = validate_worker_ownership(
                resource,
                _load_shard_manifests(evidence_dir),
                expected_workers=worker,
                expected_run_label=evidence_dir.name,
                expected_component=Stage052Component.JOB_PARALLEL.value,
            )
            if not ownership_passed:
                return {
                    "worker_ownership": {
                        "passed": False,
                        "detail": f"{evidence_dir.name}: {ownership_detail}",
                    }
                }
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
            raw_manifest_hashes[evidence_dir.name] = _sha256(
                ArtifactReader(evidence_dir).result.manifest_path
            )
            owners_by_worker[worker] = owners
            revisions.add(str(metadata.get("repository_revision", "")))
            configurations.add(str(metadata.get("configuration_sha256", "")))
            provenance_signatures.add(_performance_provenance_signature(metadata))
        if set(run_by_worker) != {1, 2, 4}:
            return {
                "worker_selection": {
                    "passed": False,
                    "detail": "worker evidence must contain exactly 1, 2, and 4 workers",
                }
            }
        identity_passed = (
            len(revisions) == 1 and len(configurations) == 1 and len(provenance_signatures) == 1
        )
        if not identity_passed:
            return {
                "worker_identity": {
                    "passed": False,
                    "detail": (
                        "worker evidence mixes repository revisions, configurations, "
                        "or invariant performance provenance"
                    ),
                }
            }
        replay_dirs = (
            [prerequisite_dir, *evidence_dirs] if prerequisite_dir is not None else evidence_dirs
        )
        replay_maps = replay_stage052_storage_semantics_many(replay_dirs)
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
            "comparison_completeness": {
                "passed": True,
                "detail": "all 1/2/4-worker bundles contain the exact complete 36-axis scope",
            },
            "worker_identity": {
                "passed": True,
                "detail": "worker revisions, configurations, and provenance match",
            },
            "worker_semantics": {
                "passed": True,
                "detail": "C04 and 1/2/4-worker fixed-work replay equality passed",
            },
            "worker_ownership": {
                "passed": True,
                "detail": "all shards are bound to sampled executor PIDs",
                "owners": {str(worker): list(owners_by_worker[worker]) for worker in (1, 2, 4)},
            },
            "worker_selection": {
                "passed": True,
                "detail": f"selected_workers={selected}",
                "selected_workers": selected,
                "selected_run_label": run_by_worker[selected],
                "input_runs": [run_by_worker[worker] for worker in (1, 2, 4)],
                "input_raw_manifest_sha256": {
                    run_by_worker[worker]: raw_manifest_hashes[run_by_worker[worker]]
                    for worker in (1, 2, 4)
                },
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


def _load_shard_manifests(raw_dir: Path) -> list[dict[str, object]]:
    reader = ArtifactReader(raw_dir)
    references = [
        item
        for item in reader.manifest.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("artifact_type") == "shard_manifest"
    ]
    if not references:
        raise ArtifactIntegrityError("worker evidence has no shard manifests")
    return [reader.read_json(str(reference["relative_path"])) for reference in references]


def _validate_native_evidence_contract(
    raw_dir: Path,
    *,
    metadata: Mapping[str, object],
    prerequisite: Stage052PrerequisiteIdentity,
    selection: JobParallelSelectionIdentity,
    benchmark_dir: Path,
) -> tuple[bool, str]:
    """Validate E control, configuration, provenance, resources, and ownership."""

    try:
        reader = ArtifactReader(raw_dir)
        config_reference = _one_artifact(reader, "config")
        resource = _load_resource_summary(raw_dir)
        shard_manifests = _load_shard_manifests(raw_dir)
        config = load_stage052_config(raw_dir / str(config_reference["relative_path"]))
    except (ArtifactIntegrityError, KeyError, OSError, TypeError, ValueError) as error:
        return False, str(error)
    manifest = reader.manifest
    if (
        manifest.get("run_label") != raw_dir.name
        or manifest.get("component") != Stage052Component.NATIVE_KERNELS.value
        or manifest.get("status") != "complete"
        or manifest.get("evidence_completeness") != "complete"
        or manifest.get("storage_policy_version") != "artifact-storage-v2"
        or manifest.get("artifact_status") != {"failure": "not_applicable"}
    ):
        return False, "native parent manifest identity or completeness is invalid"
    expected_metadata = {
        "run_label": raw_dir.name,
        "component": Stage052Component.NATIVE_KERNELS.value,
        "scope": "performance",
        "instances": list(PERFORMANCE_INSTANCES),
        "seeds": list(PERFORMANCE_SEEDS),
        "worker_count": selection.selected_workers,
        "storage_policy_version": "artifact-storage-v2",
        "backend": "cpu_batch",
        "optimization_profile": "native",
        "native_kernel_config": NativeKernelConfig().to_dict(),
        "persistence_attribution": "critical_event_rows",
        "repository_dirty": False,
        "component_prerequisite": prerequisite.to_dict(),
        "job_parallel_selection": selection.to_dict(),
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            return False, f"native metadata {field} mismatch"
    revision = metadata.get("repository_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        return False, "native repository revision is invalid"
    if metadata.get("configuration_sha256") != config_reference.get("checksum"):
        return False, "native configuration checksum is not bound to the config artifact"
    if (
        config.native_kernels != NativeKernelConfig()
        or config.v2_storage.storage_policy_version != "artifact-storage-v2"
        or config.max_iterations != 1000
        or config.batch_size != 128
    ):
        return False, "native configuration contract is invalid"
    provenance_passed, provenance_detail = _validate_performance_provenance(
        metadata, benchmark_dir=benchmark_dir
    )
    if not provenance_passed:
        return False, provenance_detail
    shards_passed, shards_detail = _validate_native_shard_manifest_scope(
        shard_manifests,
        raw_dir=raw_dir,
        run_label=raw_dir.name,
        parent_artifacts=[
            item for item in manifest.get("artifacts", []) if isinstance(item, Mapping)
        ],
    )
    if not shards_passed:
        return False, shards_detail
    ownership_passed, ownership_detail, _owners = validate_worker_ownership(
        resource,
        shard_manifests,
        expected_workers=selection.selected_workers,
        expected_run_label=raw_dir.name,
        expected_component=Stage052Component.NATIVE_KERNELS.value,
    )
    if not ownership_passed:
        return False, ownership_detail
    return True, "native config, provenance, runtime, resources, and shard ownership passed"


def _validate_native_shard_manifest_scope(
    shard_manifests: Sequence[Mapping[str, object]],
    *,
    raw_dir: Path,
    run_label: str,
    parent_artifacts: Sequence[Mapping[str, object]],
) -> tuple[bool, str]:
    expected_artifact_schema = {
        ("route_dictionary", "canonical_routes"),
        ("events", "critical"),
        ("events", "screening_checks"),
        ("events", "screening_decisions_v2"),
        ("diagnostic", "aggregated"),
        ("raw", ""),
        ("solution", ""),
        ("environment", ""),
        ("trace", ""),
    }
    expected_shards = {
        (instance, seed) for instance in PERFORMANCE_INSTANCES for seed in PERFORMANCE_SEEDS
    }
    expected_ordinal_by_identity = {
        identity: ordinal
        for ordinal, identity in enumerate(
            (instance, seed) for instance in PERFORMANCE_INSTANCES for seed in PERFORMANCE_SEEDS
        )
    }
    parent_by_path: dict[str, Mapping[str, object]] = {}
    for artifact in parent_artifacts:
        relative = str(artifact.get("relative_path", ""))
        if not relative or relative in parent_by_path:
            return False, "native parent manifest contains an invalid or duplicate path"
        parent_by_path[relative] = artifact
    observed_shards: set[tuple[str, int]] = set()
    ordinals: set[int] = set()
    shard_artifact_paths: set[str] = set()
    expected_control_paths: set[str] = set()
    if len(shard_manifests) != len(expected_shards):
        return False, "native shard manifest count is incomplete"
    for shard in shard_manifests:
        try:
            identity = (str(shard["instance"]), _strict_int(shard["seed"], "seed"))
            ordinal = _strict_int(shard["shard_ordinal"], "shard_ordinal")
        except (KeyError, TypeError, ValueError) as error:
            return False, str(error)
        artifacts = shard.get("artifacts")
        if (
            identity not in expected_shards
            or identity in observed_shards
            or ordinal in ordinals
            or ordinal != expected_ordinal_by_identity.get(identity)
            or shard.get("schema_version") != "artifact-storage-v2"
            or shard.get("run_label") != run_label
            or shard.get("evidence_completeness") != "complete"
            or shard.get("storage_policy_version") != "artifact-storage-v2"
            or shard.get("event_identity") != "shard_ordinal+shard_local_event_id"
            or not isinstance(artifacts, list)
            or any(
                isinstance(item, Mapping) and item.get("artifact_type") == "failure"
                for item in artifacts
            )
        ):
            return False, "native shard identity, ordinal, completeness, or failure is invalid"
        artifact_keys: set[tuple[str, str]] = set()
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                return False, "native shard artifact reference is invalid"
            key = (
                str(artifact.get("artifact_type", "")),
                str(artifact.get("artifact_subtype", "")),
            )
            relative = str(artifact.get("relative_path", ""))
            parts = Path(relative).parts
            if (
                key in artifact_keys
                or relative in shard_artifact_paths
                or len(parts) < 3
                or parts[:2] != (identity[0], str(identity[1]))
                or artifact.get("evidence_completeness") != "complete"
                or artifact.get("storage_policy_version") != "artifact-storage-v2"
            ):
                return False, "native shard artifact schema, path, or completeness is invalid"
            parent = parent_by_path.get(relative)
            if parent is None or dict(parent) != dict(artifact):
                return False, "native shard artifact is not bound to the parent manifest"
            artifact_keys.add(key)
            shard_artifact_paths.add(relative)
        if artifact_keys != expected_artifact_schema:
            return False, "native shard artifact schema is incomplete or contains extras"
        shard_directory = Path(identity[0]) / str(identity[1])
        manifest_relative = (
            shard_directory / f"{run_label}_shard_manifest_{identity[0]}_{identity[1]}.json"
        ).as_posix()
        sidecar_relative = Path(manifest_relative).with_suffix(".sha256").as_posix()
        manifest_reference = parent_by_path.get(manifest_relative)
        sidecar_reference = parent_by_path.get(sidecar_relative)
        if (
            manifest_reference is None
            or manifest_reference.get("artifact_type") != "shard_manifest"
            or str(manifest_reference.get("artifact_subtype", "")) != ""
            or sidecar_reference is None
            or sidecar_reference.get("artifact_type") != "shard_manifest_sidecar"
            or str(sidecar_reference.get("artifact_subtype", "")) != ""
        ):
            return False, "native shard manifest or sidecar canonical parent reference is missing"
        manifest_path = raw_dir / manifest_relative
        sidecar_path = raw_dir / sidecar_relative
        try:
            manifest_bytes = manifest_path.read_bytes()
            on_disk_manifest = json.loads(manifest_bytes)
            sidecar_text = sidecar_path.read_text(encoding="utf-8").strip()
            sidecar_bytes = sidecar_path.read_bytes()
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            return False, f"native shard manifest or sidecar cannot be read: {error}"
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
        if (
            not isinstance(on_disk_manifest, Mapping)
            or dict(on_disk_manifest) != dict(shard)
            or manifest_reference.get("checksum") != manifest_sha256
            or manifest_reference.get("byte_size") != len(manifest_bytes)
            or sidecar_text != manifest_sha256
            or sidecar_reference.get("checksum") != sidecar_sha256
            or sidecar_reference.get("byte_size") != len(sidecar_bytes)
            or manifest_reference.get("evidence_completeness") != "complete"
            or sidecar_reference.get("evidence_completeness") != "complete"
            or manifest_reference.get("storage_policy_version") != "artifact-storage-v2"
            or sidecar_reference.get("storage_policy_version") != "artifact-storage-v2"
        ):
            return False, "native shard manifest or sidecar binding is invalid"
        expected_control_paths.update({manifest_relative, sidecar_relative})
        observed_shards.add(identity)
        ordinals.add(ordinal)
    if observed_shards != expected_shards or ordinals != set(range(len(expected_shards))):
        return False, "native shard identities or ordinals are incomplete"
    parent_shard_paths: set[str] = set()
    for relative, artifact in parent_by_path.items():
        parts = Path(relative).parts
        if len(parts) < 3:
            continue
        try:
            identity = (parts[0], _strict_int(parts[1], "seed"))
        except (TypeError, ValueError):
            continue
        if identity in expected_shards and artifact.get("artifact_type") not in {
            "shard_manifest",
            "shard_manifest_sidecar",
        }:
            parent_shard_paths.add(relative)
    if parent_shard_paths != shard_artifact_paths:
        return False, "native parent/shard artifact path sets are not bidirectionally equal"
    observed_control_paths = {
        relative
        for relative, artifact in parent_by_path.items()
        if artifact.get("artifact_type") in {"shard_manifest", "shard_manifest_sidecar"}
    }
    if observed_control_paths != expected_control_paths:
        return False, "native shard manifest control paths are not exactly canonical"
    return True, "exact 12 native shard manifests and ordinals passed"


def validate_job_parallel_evidence_contract(
    raw_dir: Path,
    *,
    expected_workers: int,
    expected_prerequisite: Stage052PrerequisiteIdentity | None,
    benchmark_dir: Path,
) -> tuple[bool, str]:
    """Validate one complete D comparison bundle before it can affect speedup."""

    if expected_prerequisite is None:
        return False, "reviewed C04 prerequisite identity is missing"
    try:
        reader = ArtifactReader(raw_dir)
        rows = _load_per_run(raw_dir)
        metadata = _load_metadata(raw_dir)
        shard_manifests = _load_shard_manifests(raw_dir)
    except (ArtifactIntegrityError, OSError, ValueError, TypeError) as error:
        return False, str(error)
    manifest = reader.manifest
    if (
        manifest.get("evidence_completeness") != "complete"
        or manifest.get("status") != "complete"
        or manifest.get("artifact_status") != {"failure": "not_applicable"}
        or manifest.get("component") != Stage052Component.JOB_PARALLEL.value
        or manifest.get("run_label") != raw_dir.name
        or manifest.get("storage_policy_version") != "artifact-storage-v2"
    ):
        return False, "parent manifest identity, status, or completeness is invalid"
    expected_metadata = {
        "run_label": raw_dir.name,
        "component": Stage052Component.JOB_PARALLEL.value,
        "scope": "performance",
        "instances": list(PERFORMANCE_INSTANCES),
        "seeds": list(PERFORMANCE_SEEDS),
        "worker_count": expected_workers,
        "storage_policy_version": "artifact-storage-v2",
        "backend": "cpu_batch",
        "repository_dirty": False,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            return (
                False,
                f"metadata {field} mismatch: expected={expected} observed={metadata.get(field)}",
            )
    prerequisite_dir = raw_dir.parent / expected_prerequisite.run_label
    if not _prerequisite_binding_matches(
        metadata.get("component_prerequisite"),
        expected_prerequisite,
        prerequisite_dir,
    ):
        return False, "job-parallel metadata is not bound to the reviewed C prerequisite"
    config_reference = _one_artifact(reader, "config")
    if metadata.get("configuration_sha256") != config_reference.get("checksum"):
        return False, "metadata configuration hash is not bound to the config artifact"
    axes = tuple(axis.name for axis in axes_for_scope("performance"))
    scope_passed, scope_detail = validate_per_run_scope(
        rows,
        instances=PERFORMANCE_INSTANCES,
        seeds=PERFORMANCE_SEEDS,
        axes=axes,
    )
    if not scope_passed:
        return False, scope_detail
    replay_passed, replay_detail = _replay_solutions(reader, benchmark_dir=benchmark_dir)
    if not replay_passed:
        return False, replay_detail
    for row in rows:
        expected_row_fields = {
            "component": Stage052Component.JOB_PARALLEL.value,
            "backend": "cpu_batch",
            "worker_count": str(expected_workers),
            "storage_policy_version": "artifact-storage-v2",
        }
        for field, expected in expected_row_fields.items():
            if str(row.get(field, "")) != expected:
                return False, f"per-run {field} mismatch"
    expected_shards = {
        (instance, seed) for instance in PERFORMANCE_INSTANCES for seed in PERFORMANCE_SEEDS
    }
    expected_ordinal_by_identity = {
        identity: ordinal
        for ordinal, identity in enumerate(
            (instance, seed) for instance in PERFORMANCE_INSTANCES for seed in PERFORMANCE_SEEDS
        )
    }
    observed_shards: set[tuple[str, int]] = set()
    ordinals: set[int] = set()
    for shard in shard_manifests:
        try:
            identity = (str(shard["instance"]), _strict_int(shard["seed"], "seed"))
            ordinal = _strict_int(shard["shard_ordinal"], "shard_ordinal")
        except (KeyError, TypeError, ValueError) as error:
            return False, str(error)
        if (
            identity in observed_shards
            or ordinal in ordinals
            or ordinal != expected_ordinal_by_identity.get(identity)
        ):
            return False, "duplicate shard identity or ordinal"
        observed_shards.add(identity)
        ordinals.add(ordinal)
        artifacts = shard.get("artifacts")
        if not isinstance(artifacts, list):
            return False, "shard artifacts are invalid"
        if (
            shard.get("run_label") != raw_dir.name
            or shard.get("schema_version") != "artifact-storage-v2"
            or shard.get("evidence_completeness") != "complete"
            or shard.get("storage_policy_version") != "artifact-storage-v2"
            or shard.get("event_identity") != "shard_ordinal+shard_local_event_id"
            or any(
                isinstance(item, Mapping) and item.get("artifact_type") == "failure"
                for item in artifacts
            )
        ):
            return False, "shard identity, completeness, storage, or failure status is invalid"
    if observed_shards != expected_shards or ordinals != set(range(len(expected_shards))):
        return False, "missing or extra shard identity/ordinal"
    axis_artifacts_passed, axis_artifacts_detail = _validate_job_parallel_axis_artifacts(
        reader,
        raw_dir=raw_dir,
        expected_workers=expected_workers,
        expected_shards=expected_shards,
        expected_axes=set(axes),
    )
    if not axis_artifacts_passed:
        return False, axis_artifacts_detail
    provenance_passed, provenance_detail = _validate_performance_provenance(
        metadata, benchmark_dir=benchmark_dir
    )
    if not provenance_passed:
        return False, provenance_detail
    return True, "complete canonical D evidence contract passed"


def _validate_job_parallel_axis_artifacts(
    reader: ArtifactReader,
    *,
    raw_dir: Path,
    expected_workers: int,
    expected_shards: set[tuple[str, int]],
    expected_axes: set[str],
) -> tuple[bool, str]:
    payloads: dict[str, dict[tuple[str, int], Mapping[str, object]]] = {
        artifact_type: {} for artifact_type in ("raw", "solution", "trace")
    }
    canonical_ordinals = {
        identity: ordinal
        for ordinal, identity in enumerate(
            (instance, seed) for instance in PERFORMANCE_INSTANCES for seed in PERFORMANCE_SEEDS
        )
    }
    for artifact_type, by_identity in payloads.items():
        references = [
            item
            for item in reader.manifest.get("artifacts", [])
            if isinstance(item, Mapping) and item.get("artifact_type") == artifact_type
        ]
        if len(references) != len(expected_shards):
            return False, f"expected {len(expected_shards)} complete {artifact_type} artifacts"
        for reference in references:
            relative = str(reference.get("relative_path", ""))
            parts = Path(relative).parts
            try:
                identity = (parts[0], _strict_int(parts[1], "seed"))
            except (IndexError, TypeError, ValueError) as error:
                return False, f"invalid {artifact_type} artifact identity: {error}"
            if (
                identity not in expected_shards
                or identity in by_identity
                or reference.get("evidence_completeness") != "complete"
            ):
                return False, f"duplicate, extra, or partial {artifact_type} artifact"
            payload = reader.read_json(relative)
            if not isinstance(payload, Mapping):
                return False, f"{artifact_type} payload must be an object"
            by_identity[identity] = payload
        if set(by_identity) != expected_shards:
            return False, f"missing {artifact_type} shard identity"

    for identity in sorted(expected_shards):
        raw = payloads["raw"][identity]
        solution = payloads["solution"][identity]
        trace = payloads["trace"][identity]
        event_identity = trace.get("event_identity")
        raw_identity = {
            "run_label": raw_dir.name,
            "component": Stage052Component.JOB_PARALLEL.value,
            "scope": "performance",
            "instance": identity[0],
            "seed": identity[1],
            "worker_count": expected_workers,
        }
        if any(raw.get(field) != expected for field, expected in raw_identity.items()):
            return False, f"raw producer identity mismatch for {identity}"
        if solution.get("instance") != identity[0] or solution.get("seed") != identity[1]:
            return False, f"solution identity mismatch for {identity}"
        if event_identity != {
            "shard_ordinal": canonical_ordinals.get(identity),
            "local_field": "event_id",
        }:
            return False, f"trace event identity mismatch for {identity}"
        raw_axes = raw.get("axes")
        solution_axes = solution.get("axes")
        trace_axes = trace.get("axes")
        if not all(isinstance(value, Mapping) for value in (raw_axes, solution_axes, trace_axes)):
            return False, f"raw/solution/trace axes are invalid for {identity}"
        assert isinstance(raw_axes, Mapping)
        assert isinstance(solution_axes, Mapping)
        assert isinstance(trace_axes, Mapping)
        if any(set(value) != expected_axes for value in (raw_axes, solution_axes, trace_axes)):
            return False, f"raw/solution/trace axis identity mismatch for {identity}"
        for axis in sorted(expected_axes):
            raw_axis = raw_axes[axis]
            solution_axis = solution_axes[axis]
            trace_axis = trace_axes[axis]
            if not all(
                isinstance(value, Mapping) for value in (raw_axis, solution_axis, trace_axis)
            ):
                return False, f"invalid axis payload for {identity}/{axis}"
            assert isinstance(raw_axis, Mapping)
            assert isinstance(solution_axis, Mapping)
            assert isinstance(trace_axis, Mapping)
            reconciliation = raw_axis.get("trace_reconciliation")
            if not isinstance(reconciliation, Mapping):
                return False, f"trace reconciliation missing for {identity}/{axis}"
            checks = reconciliation.get("checks")
            if (
                raw_axis.get("valid") is not True
                or raw_axis.get("validator_passed") is not True
                or solution_axis.get("feasible") is not True
                or reconciliation.get("status") != "pass"
                or not isinstance(checks, Mapping)
                or not checks
                or any(value is not True for value in checks.values())
                or raw_axis.get("objective_key") != solution_axis.get("objective_key")
            ):
                return False, f"raw validity/reconciliation failed for {identity}/{axis}"
            result_summary = trace_axis.get("result_summary")
            if not isinstance(result_summary, Mapping):
                return False, f"trace result summary missing for {identity}/{axis}"
            raw_to_trace = {
                "started_calls": "exact_started_calls",
                "completed_calls": "exact_completed_calls",
                "effective_iterations": "effective_iterations",
                "termination_reason": "termination_reason",
            }
            if any(
                raw_axis.get(raw_field) != result_summary.get(trace_field)
                for raw_field, trace_field in raw_to_trace.items()
            ):
                return False, f"raw/trace reconciliation mismatch for {identity}/{axis}"
    return True, "exact raw/solution/trace 36-axis identity and reconciliation passed"


def _validate_captured_runtime_signature(
    *,
    environment: Mapping[str, object],
    runtime_signature: Mapping[str, object],
    optimization_profile: object,
) -> tuple[bool, str]:
    python = environment.get("python")
    system = environment.get("system")
    packages = environment.get("packages")
    native_extension = environment.get("native_extension")
    native_sha256 = runtime_signature.get("native_extension_sha256")
    if (
        not isinstance(python, Mapping)
        or not isinstance(system, Mapping)
        or not isinstance(packages, Mapping)
        or not isinstance(native_extension, str)
        or not native_extension
        or not isinstance(native_sha256, str)
        or len(native_sha256) != 64
        or any(character not in "0123456789abcdef" for character in native_sha256)
    ):
        return False, "runtime environment identity is invalid"
    expected_runtime_signature = {
        "python": {
            "version": python.get("version"),
            "implementation": python.get("implementation"),
        },
        "system": dict(system),
        "packages": dict(sorted((str(key), value) for key, value in packages.items())),
        "native_extension_sha256": native_sha256,
    }
    if dict(runtime_signature) != expected_runtime_signature:
        return False, "runtime signature does not match the captured environment"
    if optimization_profile == "native":
        current_native_path = Path(str(native_core.__file__)).resolve()
        if (
            Path(native_extension).resolve() != current_native_path
            or not current_native_path.is_file()
            or native_sha256 != _sha256(current_native_path)
        ):
            return False, "native runtime signature does not match the reviewer extension"
    elif optimization_profile != "python":
        return False, "performance runtime optimization profile is invalid"
    return True, "captured runtime signature passed"


def _validate_performance_provenance(
    metadata: Mapping[str, object],
    *,
    benchmark_dir: Path,
) -> tuple[bool, str]:
    provenance = metadata.get("performance_provenance")
    if not isinstance(provenance, Mapping):
        return False, "performance provenance is missing"
    if provenance.get("schema_version") != "stage05.2-performance-provenance-v1":
        return False, "performance provenance schema is invalid"
    instance_hashes = provenance.get("instance_sha256")
    if not isinstance(instance_hashes, Mapping) or set(instance_hashes) != set(
        PERFORMANCE_INSTANCES
    ):
        return False, "instance hash identity is incomplete"
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in instance_hashes.values()
    ):
        return False, "instance hash value is invalid"
    expected_hashes = {
        instance: _sha256(benchmark_dir / f"{instance}.txt") for instance in PERFORMANCE_INSTANCES
    }
    if dict(instance_hashes) != expected_hashes:
        return False, "instance hashes do not match the reviewed benchmark inputs"
    required = {
        "warm_start",
        "operator_surface",
        "fixed_work_contract",
        "worker_affinity",
        "environment_variables",
        "background_load",
        "power_mode",
        "runtime_signature",
        "failure_policy",
        "fallback_allowed",
    }
    if not required.issubset(provenance):
        return False, "performance provenance fields are incomplete"
    if provenance.get("warm_start") != {"enabled": False, "source": None}:
        return False, "D warm-start contract is invalid"
    fixed_work = provenance.get("fixed_work_contract")
    if fixed_work != {
        "exact_call_budget": 100,
        "watchdog_seconds": 120.0,
        "max_iterations": 1000,
        "batch_size": 128,
        "backend": "cpu_batch",
    }:
        return False, "fixed-work contract is invalid"
    operator_surface = provenance.get("operator_surface")
    repository_root = benchmark_dir.resolve().parents[1]
    expected_operator_hashes = {
        "stage02_config_sha256": _sha256(
            repository_root / "configs" / "stage02_constraint_guided.toml"
        ),
        "stage04_config_sha256": _sha256(repository_root / "configs" / "stage04_weights.toml"),
    }
    if (
        not isinstance(operator_surface, Mapping)
        or operator_surface.get("operator_profile") != "stage02_constraint_guided"
        or any(
            not isinstance(operator_surface.get(field), str)
            or operator_surface.get(field) != expected
            for field, expected in expected_operator_hashes.items()
        )
    ):
        return False, "operator surface provenance is invalid"
    affinity = provenance.get("worker_affinity")
    if not isinstance(affinity, Mapping) or not isinstance(affinity.get("supported"), bool):
        return False, "worker affinity provenance is invalid"
    cpu_ids = affinity.get("cpu_ids")
    if (
        not isinstance(cpu_ids, list)
        or any(not isinstance(value, int) or isinstance(value, bool) for value in cpu_ids)
        or cpu_ids != sorted(set(cpu_ids))
    ):
        return False, "worker affinity CPU identity is invalid"
    environment_variables = provenance.get("environment_variables")
    if (
        not isinstance(environment_variables, Mapping)
        or set(environment_variables) != _PERFORMANCE_ENVIRONMENT_VARIABLES
        or any(
            value is not None and not isinstance(value, str)
            for value in environment_variables.values()
        )
    ):
        return False, "performance environment variables are invalid"
    background = provenance.get("background_load")
    if not isinstance(background, Mapping):
        return False, "background-load provenance is invalid"
    load_average = background.get("load_average")
    statuses = background.get("process_status_counts")
    if (
        not isinstance(load_average, list)
        or len(load_average) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in load_average
        )
        or not isinstance(statuses, Mapping)
        or not statuses
        or any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for key, value in statuses.items()
        )
    ):
        return False, "background-load provenance is invalid"
    power_mode = provenance.get("power_mode")
    if (
        not isinstance(power_mode, Mapping)
        or power_mode.get("available") is not True
        or power_mode.get("source") not in {"AC Power", "Battery Power"}
        or power_mode.get("low_power_mode") not in {0, 1}
    ):
        return False, "power-mode provenance is invalid"
    environment = metadata.get("environment")
    runtime_signature = provenance.get("runtime_signature")
    if not isinstance(environment, Mapping) or not isinstance(runtime_signature, Mapping):
        return False, "runtime provenance is invalid"
    runtime_passed, runtime_detail = _validate_captured_runtime_signature(
        environment=environment,
        runtime_signature=runtime_signature,
        optimization_profile=metadata.get("optimization_profile"),
    )
    if not runtime_passed:
        return False, runtime_detail
    python = environment.get("python")
    system = environment.get("system")
    packages = environment.get("packages")
    if (
        not isinstance(python, Mapping)
        or not isinstance(system, Mapping)
        or not isinstance(packages, Mapping)
    ):
        return False, "runtime environment identity is invalid"
    current_environment = collect_environment()
    current_python = current_environment.get("python")
    current_system = current_environment.get("system")
    current_packages = current_environment.get("packages")
    if not all(
        isinstance(value, Mapping) for value in (current_python, current_system, current_packages)
    ):
        return False, "reviewer runtime environment is incomplete"
    assert isinstance(current_python, Mapping)
    assert isinstance(current_system, Mapping)
    assert isinstance(current_packages, Mapping)
    recorded_python = {
        "version": python.get("version"),
        "implementation": python.get("implementation"),
    }
    reviewed_python = {
        "version": current_python.get("version"),
        "implementation": current_python.get("implementation"),
    }
    if (
        not all(isinstance(value, str) and value for value in recorded_python.values())
        or recorded_python != reviewed_python
    ):
        return False, "Python runtime identity does not match the reviewer"
    required_system_fields = {
        "platform",
        "machine",
        "processor",
        "cpu_count",
        "gpu_used",
    }
    if (
        not required_system_fields.issubset(system)
        or dict(system) != dict(current_system)
        or not isinstance(system.get("platform"), str)
        or not str(system.get("platform"))
        or not isinstance(system.get("machine"), str)
        or not str(system.get("machine"))
        or not isinstance(system.get("cpu_count"), int)
        or isinstance(system.get("cpu_count"), bool)
        or int(system.get("cpu_count", 0)) <= 0
    ):
        return False, "system runtime identity does not match the reviewer"
    if not packages or dict(packages) != dict(current_packages):
        return False, "package runtime identity does not match the reviewer"
    if provenance.get("failure_policy") != "abort_all_workers_without_fallback":
        return False, "worker failure policy is invalid"
    if provenance.get("fallback_allowed") is not False:
        return False, "fallback must be disabled"
    return True, "performance provenance passed"


def _performance_provenance_signature(metadata: Mapping[str, object]) -> str:
    provenance = metadata.get("performance_provenance")
    if not isinstance(provenance, Mapping):
        return "<missing>"
    invariant_fields = (
        "schema_version",
        "instance_sha256",
        "warm_start",
        "operator_surface",
        "fixed_work_contract",
        "worker_affinity",
        "environment_variables",
        "power_mode",
        "runtime_signature",
        "failure_policy",
        "fallback_allowed",
    )
    return hashlib.sha256(
        _canonical_json_bytes({field: provenance.get(field) for field in invariant_fields})
    ).hexdigest()


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
        try:
            number = float(value)
        except ValueError as error:
            raise TypeError("numeric value must be parseable") from error
        if not math.isfinite(number):
            raise ValueError("numeric value must be finite")
        return number
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


def _prior_review_manifest_hashes(raw_dir: Path) -> list[str]:
    """Preserve accepted review identity while publishing a stronger re-review."""

    path = raw_dir / "review" / "review_manifest.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError("cannot read prior Stage 5.2 review manifest") from error
    raw_lineage = payload.get("review_manifest_lineage_sha256")
    if raw_lineage is not None and raw_lineage != []:
        raise ArtifactIntegrityError(
            "prior Stage 5.2 review already has lineage and cannot be overwritten"
        )
    try:
        component = Stage052Component(str(payload.get("component", "")))
    except ValueError as error:
        raise ArtifactIntegrityError("prior Stage 5.2 review component is invalid") from error
    expected_identity = {
        "schema_version": STAGE052_REVIEW_SCHEMA_VERSION,
        "run_label": raw_dir.name,
        "component": component.value,
        "scope": "performance",
        "status": _NEXT_STATUS[component],
    }
    if any(payload.get(field) != expected for field, expected in expected_identity.items()):
        raise ArtifactIntegrityError("only an accepted prior Stage 5.2 review may be superseded")
    gates = payload.get("gates")
    if (
        not isinstance(gates, Mapping)
        or not gates
        or any(
            not isinstance(gate, Mapping) or gate.get("passed") is not True
            for gate in gates.values()
        )
    ):
        raise ArtifactIntegrityError("prior Stage 5.2 review gates are not accepted")
    verify_stage052_review_files(raw_dir, payload)
    current_raw_sha256 = _sha256(ArtifactReader(raw_dir).result.manifest_path)
    prior_raw_sha256 = payload.get("raw_manifest_sha256")
    if prior_raw_sha256 is not None and prior_raw_sha256 != current_raw_sha256:
        raise ArtifactIntegrityError("prior Stage 5.2 review is stale for the current raw manifest")
    return [
        _archive_prior_review_generation(
            raw_dir,
            manifest_path=path,
            manifest_payload=payload,
        )
    ]


def _review_lineage_archive_matches(
    prerequisite_dir: Path,
    current: Stage052PrerequisiteIdentity,
    prior_sha256: str,
) -> bool:
    if len(prior_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in prior_sha256
    ):
        return False
    current_manifest_path = prerequisite_dir / "review" / "review_manifest.json"
    archive_dir = prerequisite_dir / "review" / "history" / prior_sha256
    archive_manifest_path = archive_dir / "review_manifest.json"
    try:
        current_payload = json.loads(current_manifest_path.read_text(encoding="utf-8"))
        archive_bytes = archive_manifest_path.read_bytes()
        archive_payload = json.loads(archive_bytes)
        verify_stage052_review_files(prerequisite_dir, current_payload)
    except (OSError, json.JSONDecodeError):
        return False
    except ArtifactIntegrityError:
        return False
    expected_current_identity = {
        "schema_version": STAGE052_REVIEW_SCHEMA_VERSION,
        "run_label": current.run_label,
        "component": current.component,
        "scope": "performance",
        "status": current.status,
    }
    current_gates = current_payload.get("gates") if isinstance(current_payload, Mapping) else None
    if (
        not isinstance(current_payload, Mapping)
        or any(
            current_payload.get(field) != expected
            for field, expected in expected_current_identity.items()
        )
        or _sha256(current_manifest_path) != current.review_manifest_sha256
        or current_payload.get("raw_manifest_sha256") != current.raw_manifest_sha256
        or current_payload.get("review_manifest_lineage_sha256") != [prior_sha256]
        or not isinstance(current_gates, Mapping)
        or not current_gates
        or any(
            not isinstance(gate, Mapping) or gate.get("passed") is not True
            for gate in current_gates.values()
        )
        or not isinstance(archive_payload, Mapping)
        or hashlib.sha256(archive_bytes).hexdigest() != prior_sha256
        or not archive_dir.is_dir()
    ):
        return False
    expected_archive_identity = {
        "schema_version": STAGE052_REVIEW_SCHEMA_VERSION,
        "run_label": current.run_label,
        "component": current.component,
        "scope": "performance",
        "status": current.status,
    }
    if any(
        archive_payload.get(field) != expected
        for field, expected in expected_archive_identity.items()
    ):
        return False
    archived_raw_sha256 = archive_payload.get("raw_manifest_sha256")
    archived_lineage = archive_payload.get("review_manifest_lineage_sha256")
    if (archived_lineage is not None and archived_lineage != []) or (
        archived_raw_sha256 is not None and archived_raw_sha256 != current.raw_manifest_sha256
    ):
        return False
    gates = archive_payload.get("gates")
    files = archive_payload.get("files")
    if (
        not isinstance(gates, Mapping)
        or not gates
        or any(
            not isinstance(gate, Mapping) or gate.get("passed") is not True
            for gate in gates.values()
        )
        or not isinstance(files, Mapping)
        or len(files) != 2
    ):
        return False
    archived_review_paths = tuple(Path(str(name)) for name in files)
    legacy_files = {path.as_posix() for path in archived_review_paths} == {
        "review_findings.csv",
        "review_report.md",
    }
    generation_files = (
        {path.name for path in archived_review_paths} == {"review_findings.csv", "review_report.md"}
        and all(
            len(path.parts) == 3
            and path.parts[0] == "generations"
            and len(path.parts[1]) == 64
            and all(character in "0123456789abcdef" for character in path.parts[1])
            for path in archived_review_paths
        )
        and len({path.parts[1] for path in archived_review_paths}) == 1
    )
    if not legacy_files and not generation_files:
        return False
    relative_files = {"review_manifest.json", *(str(name) for name in files)}
    observed_files = {
        path.relative_to(archive_dir).as_posix()
        for path in archive_dir.rglob("*")
        if path.is_file()
    }
    if observed_files != relative_files:
        return False
    for name, checksum in files.items():
        relative = Path(str(name))
        archived_file = (archive_dir / relative).resolve()
        if (
            archive_dir.resolve() not in archived_file.parents
            or not archived_file.is_file()
            or _sha256(archived_file) != str(checksum)
        ):
            return False
    return True


def _prerequisite_binding_matches(
    bound: object,
    current: Stage052PrerequisiteIdentity,
    prerequisite_dir: Path,
) -> bool:
    """Accept an immutable producer's prior review hash only through signed lineage."""

    expected = current.to_dict()
    if bound == expected:
        return True
    if not isinstance(bound, Mapping) or set(bound) != set(expected):
        return False
    if any(
        bound.get(field) != value
        for field, value in expected.items()
        if field != "review_manifest_sha256"
    ):
        return False
    try:
        review = json.loads(
            (prerequisite_dir / "review" / "review_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    prior_sha256 = bound.get("review_manifest_sha256")
    lineage = review.get("review_manifest_lineage_sha256")
    return (
        isinstance(prior_sha256, str)
        and lineage == [prior_sha256]
        and _review_lineage_archive_matches(prerequisite_dir, current, prior_sha256)
    )


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
