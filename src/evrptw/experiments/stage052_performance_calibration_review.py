"""Independently replay and qualify a Stage 5.2 performance calibration.

The reviewer trusts neither the producer's selected profile nor its aggregate
semantic projections.  It verifies every signed child axis, streams the
canonical journals, replays the unified validator/objective, reruns the
deterministic selector, and publishes a signed qualification receipt only when
the derived profile is byte-for-byte equivalent at the canonical JSON level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Final, cast

from evrptw.candidate_control import stable_candidate_payload_hash
from evrptw.experiments.stage052_native_architecture_review import (
    _replay_canonical_journal,
    _replay_measurement_evidence,
    _replay_raw_native_control_journal,
)
from evrptw.experiments.stage052_native_architectures import (
    AXIS_PERSISTENCE_RECEIPT_SCHEMA_VERSION,
    workload_class_for_instance,
)
from evrptw.experiments.stage052_performance_calibration import (
    CALIBRATION_INPUT_DIRECTORY,
    CALIBRATION_INPUT_STORAGE_ALIAS,
    CALIBRATION_RECEIPT_SCHEMA_VERSION,
    OBSERVATION_PRODUCER_SCHEMA_VERSION,
    AxisObservation,
    CalibrationError,
    calibrate_performance_profile,
    load_fixed_work_observations,
    load_wheel_receipts,
)
from evrptw.experiments.stage052_telemetry_overhead import (
    load_telemetry_overhead_receipt,
)
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.stage052_atomic import publish_no_replace
from evrptw.stage052_performance import (
    ExecutionTopology,
    FrozenPerformanceProfile,
    RuntimeResourceSummaryV2,
    TelemetryOverheadReceipt,
    require_clean_repository_root,
)
from evrptw.validation import validate_routes

CALIBRATION_REVIEW_SCHEMA_VERSION: Final = (
    "stage05.2-native-architecture-performance-calibration-review-v2"
)
CALIBRATION_REVIEW_QUALIFICATION: Final = "QUALIFIED_FOR_ATTEMPT08"
CALIBRATION_REVIEW_FAILURE_SCHEMA_VERSION: Final = (
    "stage05.2-native-architecture-performance-calibration-review-failure-v1"
)
_SHA256_RE: Final = frozenset("0123456789abcdef")


class CalibrationReviewError(RuntimeError):
    """The calibration cannot be independently qualified."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise CalibrationReviewError(f"cannot hash review input: {path}") from error
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_SHA256_RE)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CalibrationReviewError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise CalibrationReviewError(f"{field} must be an array")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CalibrationReviewError(f"{field} must be non-empty text")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationReviewError(f"{field} must be a non-negative integer")
    return value


def _number(value: object, field: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise CalibrationReviewError(f"{field} must be a finite {qualifier} number")
    return float(value)


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CalibrationReviewError("review value is not canonical JSON") from error


_REPRESENTATIVE_TRAJECTORY_FIELDS = (
    "semantic_trajectory",
    "trajectory",
    "stage04_events",
    "candidate_transaction_events",
)
_REPRESENTATIVE_TRAJECTORY_MAX_ROWS = 200_000
_REPRESENTATIVE_TRAJECTORY_MAX_BYTES = 64 * 1024 * 1024


def _representative_row_evidence(rows: Sequence[object]) -> tuple[dict[str, object], int]:
    digest = hashlib.sha256(b"stage05.2-row-evidence-v1\0")
    payload_bytes = 0
    for row in rows:
        encoded = _canonical(row)
        payload_bytes += len(encoded)
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return {"count": len(rows), "sha256": digest.hexdigest()}, payload_bytes


def _thaw_json(value: object) -> object:
    """Project frozen containers back to their exact JSON representation."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise CalibrationReviewError("review value is not representable as JSON")


def _json_equivalent(left: object, right: object) -> bool:
    return _canonical(_thaw_json(left)) == _canonical(_thaw_json(right))


def _load_signed_json(path: Path, field: str) -> tuple[dict[str, object], str]:
    resolved = path.resolve()
    sidecar = Path(f"{resolved}.sha256")
    try:
        data = resolved.read_bytes()
        declared = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise CalibrationReviewError(f"cannot read signed {field}: {resolved}") from error
    observed = _sha256_bytes(data)
    if declared != observed:
        raise CalibrationReviewError(f"{field} SHA-256 sidecar mismatch")
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationReviewError(f"{field} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise CalibrationReviewError(f"{field} root is not an object")
    return cast(dict[str, object], payload), observed


def _atomic_signed_json(path: Path, payload: Mapping[str, object]) -> str:
    sidecar = Path(f"{path}.sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"calibration review output already exists: {path}")
    data = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    digest = _sha256_bytes(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    path_published = False
    sidecar_published = False
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        with temporary_sidecar.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(digest + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_no_replace(temporary_sidecar, sidecar)
        sidecar_published = True
        publish_no_replace(temporary, path)
        path_published = True
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        if not path_published and sidecar_published:
            sidecar.unlink(missing_ok=True)
        raise
    return digest


def _inventory_paths(
    receipt: Mapping[str, object],
    *,
    run_dir: Path,
) -> tuple[tuple[Path, ...], tuple[Path, ...], Path, Path | None]:
    inventory = _mapping(receipt.get("signed_input_inventory"), "signed_input_inventory")
    storage_roots = _mapping(
        receipt.get("input_storage_roots"),
        "input_storage_roots",
    )
    if storage_roots != {CALIBRATION_INPUT_STORAGE_ALIAS: CALIBRATION_INPUT_DIRECTORY}:
        raise CalibrationReviewError("calibration input storage roots are invalid")
    input_root = (run_dir / CALIBRATION_INPUT_DIRECTORY).resolve()
    try:
        input_root.relative_to(run_dir.resolve())
    except ValueError as error:  # pragma: no cover - constant child path
        raise CalibrationReviewError("calibration input root escapes its run") from error
    if not input_root.is_dir():
        raise CalibrationReviewError("calibration input bundle is unavailable")

    def path_for(entry: Mapping[str, object], field: str) -> Path:
        alias = _text(entry.get("storage_alias"), f"{field}.storage_alias")
        if alias != CALIBRATION_INPUT_STORAGE_ALIAS:
            raise CalibrationReviewError(f"{field} storage alias is invalid")
        relative = _text(entry.get("relative_path"), f"{field}.relative_path")
        portable = PurePosixPath(relative)
        if portable.is_absolute() or ".." in portable.parts or "\\" in relative:
            raise CalibrationReviewError(f"{field} portable path is invalid")
        resolved = (input_root / Path(*portable.parts)).resolve()
        try:
            resolved.relative_to(input_root)
        except ValueError as error:  # pragma: no cover - guarded by PurePosixPath
            raise CalibrationReviewError(f"{field} escapes its input bundle") from error
        return resolved

    def verify_signed_input(path: Path, expected: object, field: str) -> None:
        if not _is_sha256(expected) or _sha256_file(path) != expected:
            raise CalibrationReviewError(f"{field} hash mismatch")
        sidecar = Path(f"{path}.sha256")
        try:
            declared = sidecar.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as error:
            raise CalibrationReviewError(f"{field} sidecar is unavailable") from error
        if declared != expected:
            raise CalibrationReviewError(f"{field} sidecar mismatch")

    def paths(field: str) -> tuple[Path, ...]:
        result: list[Path] = []
        for index, raw in enumerate(_array(inventory.get(field), field)):
            entry = _mapping(raw, f"{field}[{index}]")
            path = path_for(entry, f"{field}[{index}]")
            verify_signed_input(path, entry.get("sha256"), f"{field}[{index}]")
            result.append(path)
        if not result or len(result) != len(set(result)):
            raise CalibrationReviewError(f"{field} paths must be non-empty and unique")
        return tuple(result)

    telemetry = _mapping(inventory.get("telemetry_overhead"), "telemetry_overhead")
    telemetry_path = path_for(telemetry, "telemetry_overhead")
    expected_telemetry = telemetry.get("sha256")
    verify_signed_input(telemetry_path, expected_telemetry, "telemetry overhead input")
    raw_host = inventory.get("host_envelope")
    host_path: Path | None = None
    if raw_host is not None:
        host_entry = _mapping(raw_host, "host_envelope")
        host_path = path_for(host_entry, "host_envelope")
        expected_host = host_entry.get("sha256")
        verify_signed_input(host_path, expected_host, "host envelope input")
    return (
        paths("wheel_receipts"),
        paths("fixed_work_observations"),
        telemetry_path,
        host_path,
    )


def _validate_manifest(run_dir: Path, receipt_path: Path, receipt_sha256: str) -> str:
    manifest_path = run_dir / "manifest.json"
    manifest, digest = _load_signed_json(manifest_path, "calibration manifest")
    if manifest.get("status") != "complete" or manifest.get("evidence_completeness") != "complete":
        raise CalibrationReviewError("calibration manifest is not terminal complete evidence")
    inventory = _array(manifest.get("artifact_inventory"), "manifest artifact_inventory")
    matches = [
        entry
        for entry in inventory
        if isinstance(entry, Mapping)
        and entry.get("relative_path") == receipt_path.relative_to(run_dir).as_posix()
        and entry.get("sha256") == receipt_sha256
    ]
    if len(matches) != 1:
        raise CalibrationReviewError("calibration manifest does not bind its receipt")
    return digest


def _raw_projection(payload: Mapping[str, object]) -> dict[str, object]:
    measurement = _mapping(payload.get("measurement_evidence"), "measurement_evidence")
    journal = _mapping(payload.get("canonical_semantic_journal"), "semantic journal")
    hashes = (
        payload.get("candidate_work_hash"),
        payload.get("route_result_hash"),
        journal.get("sha256"),
    )
    if any(not _is_sha256(value) for value in hashes):
        raise CalibrationReviewError("raw transaction hashes are invalid")
    projection = {
        "objective": payload.get("objective"),
        "routes": payload.get("routes"),
        "candidate_trajectory": payload.get("semantic_trajectory"),
        "exact_order": measurement.get("exact_route_order"),
        "cache_lifecycle": measurement.get("cache_lifecycle"),
        "transaction_hashes": list(hashes),
    }
    _canonical(projection)
    return projection


def _routes(payload: Mapping[str, object]) -> list[list[str]]:
    result: list[list[str]] = []
    for index, raw in enumerate(_array(payload.get("routes"), "raw routes")):
        if not isinstance(raw, list) or not all(isinstance(node, str) for node in raw):
            raise CalibrationReviewError(f"raw route {index} is invalid")
        result.append(cast(list[str], raw))
    return result


def _load_reconciled_raw_axis(path: Path) -> dict[str, object]:
    """Load one signed axis and independently apply its signed persistence receipt."""

    payload, _digest = _load_signed_json(path, "raw calibration axis")
    receipt_path = path.with_suffix(path.suffix + ".persistence")
    receipt, _receipt_digest = _load_signed_json(
        receipt_path,
        "raw calibration axis persistence receipt",
    )
    descriptor = payload.get("persistence_receipt")
    if (
        not isinstance(descriptor, Mapping)
        or descriptor
        != {
            "schema_version": AXIS_PERSISTENCE_RECEIPT_SCHEMA_VERSION,
            "path": receipt_path.name,
            "sidecar_path": receipt_path.name + ".sha256",
        }
        or set(receipt)
        != {
            "schema_version",
            "axis_path",
            "axis_sha256",
            "axis_status",
            "persistence_seconds",
            "end_to_end_seconds",
            "primary_artifact_bytes",
            "persistence_receipt_bytes",
            "artifact_bytes",
            "persistence_breakdown",
            "timing_scope",
        }
        or receipt.get("schema_version") != AXIS_PERSISTENCE_RECEIPT_SCHEMA_VERSION
        or receipt.get("axis_path") != path.name
        or receipt.get("axis_sha256") != _sha256_file(path)
        or receipt.get("axis_status") != payload.get("status")
        or not isinstance(receipt.get("persistence_breakdown"), Mapping)
    ):
        raise CalibrationReviewError("raw axis persistence receipt is invalid")
    for field in ("persistence_seconds", "end_to_end_seconds"):
        _number(receipt.get(field), f"raw axis {field}")
    primary_bytes = _integer(receipt.get("primary_artifact_bytes"), "primary_artifact_bytes")
    receipt_bytes = _integer(
        receipt.get("persistence_receipt_bytes"),
        "persistence_receipt_bytes",
    )
    artifact_bytes = _integer(receipt.get("artifact_bytes"), "artifact_bytes")
    if min(primary_bytes, receipt_bytes) <= 0 or artifact_bytes != primary_bytes + receipt_bytes:
        raise CalibrationReviewError("raw axis persistence byte accounting diverged")
    payload = dict(payload)
    payload["persistence_seconds"] = receipt["persistence_seconds"]
    payload["producer_pre_receipt_seconds"] = receipt["end_to_end_seconds"]
    payload["artifact_bytes"] = artifact_bytes
    payload["persistence_breakdown"] = receipt["persistence_breakdown"]
    return payload


def _review_raw_axis(
    path: Path,
    *,
    benchmark_dir: Path,
    build_identity: Mapping[str, object],
) -> tuple[dict[str, object], str, str, str]:
    payload = _load_reconciled_raw_axis(path)
    if payload.get("status") != "completed" or payload.get("axis") != "fixed_work":
        raise CalibrationReviewError(f"raw axis is not completed fixed-work evidence: {path}")
    for raw_field, identity_field in (
        ("revision", "git_revision"),
        ("wheel_sha256", "wheel_sha256"),
        ("native_sha256", "native_sha256"),
        ("scheduler_sha256", "scheduler_sha256"),
    ):
        if payload.get(raw_field) != build_identity.get(identity_field):
            raise CalibrationReviewError(f"raw axis {raw_field} differs from build identity")
    if payload.get("validator_passed") is not True or payload.get("fallback_count") != 0:
        raise CalibrationReviewError("raw axis validator/fallback gate failed")
    if _integer(payload.get("exact_started_calls"), "exact_started_calls") > 100:
        raise CalibrationReviewError("raw fixed-work axis exceeded its exact-call budget")
    instance_name = _text(payload.get("instance"), "raw instance")
    instance = parse_schneider(benchmark_dir / f"{instance_name}.txt")
    report = validate_routes(instance, _routes(payload))
    if not report.feasible:
        raise CalibrationReviewError("raw axis unified validator replay failed")
    objective = SolutionObjective.from_report(instance, report)
    if list(objective.key) != payload.get("objective"):
        raise CalibrationReviewError("raw axis objective replay mismatch")
    if payload.get("mode") == "full_native_alns":
        control_error = _replay_raw_native_control_journal(
            payload,
            axis_path=path,
            instance=instance,
        )
        if control_error is not None:
            raise CalibrationReviewError(control_error)
    journal_error = _replay_canonical_journal(payload, axis_path=path, instance=instance)
    if journal_error is not None:
        raise CalibrationReviewError(journal_error)
    measurement_error = _replay_measurement_evidence(payload, axis_path=path)
    if measurement_error is not None:
        raise CalibrationReviewError(measurement_error)
    mode = _text(payload.get("mode"), "raw mode")
    workload = workload_class_for_instance(instance_name)
    topology = _mapping(payload.get("topology"), "raw topology")
    key = _text(topology.get("performance_topology_key"), "performance_topology_key")
    prefix = f"calibration:{mode}:{workload}:"
    if not key.startswith(prefix) or len(key) == len(prefix):
        raise CalibrationReviewError("raw calibration topology key is invalid")
    return _raw_projection(payload), mode, workload, key.removeprefix(prefix)


def _resource_percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def _validate_resource_thread_tree(process_tree: Mapping[str, object]) -> None:
    thread_tree = _mapping(process_tree.get("thread_tree"), "resource thread_tree")
    rows = _array(process_tree.get("thread_metrics"), "resource thread_metrics")
    if (
        process_tree.get("thread_tree_status") != "available"
        or thread_tree.get("status") != "available"
        or thread_tree.get("sample_missed_processes") != 0
        or thread_tree.get("unresolved_thread_observation_count") != 0
        or thread_tree.get("unresolved_thread_ids") != []
        or thread_tree.get("observed_thread_identities") != len(rows)
        or not rows
    ):
        raise CalibrationReviewError("resource thread-tree evidence is incomplete")
    counter_names = {
        "voluntary_context_switches",
        "involuntary_context_switches",
        "minor_faults",
        "major_faults",
        "schedstat_runtime_ns",
        "schedstat_runqueue_delay_ns",
        "schedstat_timeslices",
        "cpu_migrations",
    }
    totals = {name: 0 for name in counter_names}
    user_seconds = 0.0
    system_seconds = 0.0
    affinity_union: set[int] = set()
    affinity_intersection: set[int] | None = None
    identities: set[tuple[int, float, int, int]] = set()
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"resource thread row {index}")
        pid = _integer(row.get("pid"), "resource thread pid")
        tid = _integer(row.get("tid"), "resource thread tid")
        start_ticks = _integer(row.get("thread_start_time_ticks"), "thread start ticks")
        create_time = _number(row.get("process_create_time"), "thread process create time")
        sample_count = _integer(row.get("sample_count"), "thread sample count")
        if min(pid, tid, sample_count) == 0 or create_time <= 0.0:
            raise CalibrationReviewError("resource thread identity is invalid")
        identity = (pid, create_time, tid, start_ticks)
        if identity in identities:
            raise CalibrationReviewError("resource thread identity is duplicated")
        identities.add(identity)
        if row.get("cpu_baseline_source") not in {"monitor_start", "thread_start"}:
            raise CalibrationReviewError("resource thread CPU baseline is invalid")
        user_seconds += _number(row.get("user_cpu_seconds"), "thread user CPU")
        system_seconds += _number(row.get("system_cpu_seconds"), "thread system CPU")
        counters = _mapping(row.get("counters"), "resource thread counters")
        if set(counters) != counter_names:
            raise CalibrationReviewError("resource thread counter schema is invalid")
        for name in counter_names:
            totals[name] += _integer(counters.get(name), f"resource thread {name}")
        affinity = row.get("last_affinity")
        if (
            not isinstance(affinity, list)
            or not affinity
            or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in affinity)
            or affinity != sorted(set(affinity))
        ):
            raise CalibrationReviewError("resource thread affinity is invalid")
        selected = set(affinity)
        affinity_union.update(selected)
        affinity_intersection = (
            selected
            if affinity_intersection is None
            else affinity_intersection.intersection(selected)
        )
    schedstat = {
        "runtime_ns": totals["schedstat_runtime_ns"],
        "runqueue_delay_ns": totals["schedstat_runqueue_delay_ns"],
        "timeslices": totals["schedstat_timeslices"],
    }
    expected_context_switches = (
        totals["voluntary_context_switches"] + totals["involuntary_context_switches"]
    )
    if (
        dict(_mapping(thread_tree.get("counters"), "thread aggregate counters")) != totals
        or thread_tree.get("affinity_union") != sorted(affinity_union)
        or thread_tree.get("affinity_intersection") != sorted(affinity_intersection or set())
        or process_tree.get("thread_affinity_union") != sorted(affinity_union)
        or process_tree.get("thread_affinity_intersection")
        != sorted(affinity_intersection or set())
        or dict(_mapping(process_tree.get("thread_tree_schedstat"), "thread schedstat"))
        != schedstat
        or process_tree.get("thread_tree_context_switches") != expected_context_switches
        or process_tree.get("thread_tree_cpu_migrations") != totals["cpu_migrations"]
        or process_tree.get("thread_tree_minor_faults") != totals["minor_faults"]
        or process_tree.get("thread_tree_major_faults") != totals["major_faults"]
        or not math.isclose(
            _number(thread_tree.get("user_cpu_seconds"), "thread aggregate user CPU"),
            user_seconds,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            _number(thread_tree.get("system_cpu_seconds"), "thread aggregate system CPU"),
            system_seconds,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise CalibrationReviewError("resource thread-tree aggregates do not replay")


def _resource_queue_totals(
    payloads: Sequence[Mapping[str, object]],
    scheduler_statistics: Mapping[str, object] | None,
) -> tuple[float, int, int, int, int]:
    wait_seconds = 0.0
    depth_peak = 0
    pending_peak = 0
    full_count = 0
    rejected_count = 0
    if scheduler_statistics is not None:
        for name in ("request_queue", "work_queue"):
            queue = _mapping(scheduler_statistics.get(name), f"scheduler {name}")
            if queue.get("pending") != 0 or (name == "work_queue" and queue.get("active") != 0):
                raise CalibrationReviewError("scheduler resource queue did not drain")
            wait_seconds += _number(queue.get("total_wait_seconds"), f"{name} wait")
            peak = _integer(queue.get("peak_pending"), f"{name} peak pending")
            depth_peak = max(depth_peak, peak)
            pending_peak = max(pending_peak, peak)
            full_count += _integer(queue.get("queue_full_count"), f"{name} queue full")
            rejected_count += _integer(queue.get("rejected_count"), f"{name} rejected")
    expected_pool_modes = {"per_solve_runtime", "full_native_alns", "host_scheduler"}
    observed_pools = 0
    for payload in payloads:
        mode = payload.get("mode")
        pool: Mapping[str, object] | None = None
        candidate = payload.get("candidate_transaction_statistics")
        if isinstance(candidate, Mapping):
            raw_pool = candidate.get("native_candidate_work_pool")
            if isinstance(raw_pool, Mapping) and raw_pool.get("enabled") is True:
                pool = raw_pool
        native = payload.get("native_execution_statistics")
        if isinstance(native, Mapping) and isinstance(native.get("work_pool_statistics"), Mapping):
            pool = cast(Mapping[str, object], native["work_pool_statistics"])
        if pool is None:
            if mode in expected_pool_modes:
                raise CalibrationReviewError("native calibration axis lacks work-pool telemetry")
            continue
        observed_pools += 1
        if pool.get("pending_tasks") != 0 or pool.get("active_tasks") != 0:
            raise CalibrationReviewError("native calibration work pool did not drain")
        wait_seconds += _number(pool.get("total_wait_seconds"), "work-pool wait")
        peak = _integer(pool.get("peak_pending_tasks"), "work-pool peak pending")
        depth_peak = max(depth_peak, peak)
        pending_peak = max(pending_peak, peak)
        full_count += _integer(pool.get("queue_full_count"), "work-pool queue full")
        rejected_count += _integer(pool.get("rejected_count"), "work-pool rejected")
        if _integer(pool.get("task_receipt_dropped_count"), "task receipt dropped") != 0:
            raise CalibrationReviewError("native calibration task receipts were dropped")
    if (
        payloads
        and payloads[0].get("mode") in expected_pool_modes
        and observed_pools != len(payloads)
    ):
        raise CalibrationReviewError("native work-pool resource matrix is incomplete")
    if full_count != 0 or rejected_count != 0:
        raise CalibrationReviewError("resource queue hard gate failed")
    return wait_seconds, depth_peak, pending_peak, full_count, rejected_count


def _derive_resource_summary(
    evidence: Mapping[str, object],
    *,
    run_root: Path,
    inventory_roles: Mapping[str, str],
    scheduler_statistics: Mapping[str, object] | None,
) -> tuple[RuntimeResourceSummaryV2, int, tuple[str, ...]]:
    if evidence.get("schema_version") != "stage05.2-calibration-resource-evidence-v1":
        raise CalibrationReviewError("raw resource evidence schema is invalid")
    if (
        evidence.get("worker_process_lifecycle") != "one_shard_per_spawned_process"
        or evidence.get("worker_multiprocessing_start_method") != "spawn"
        or evidence.get("worker_max_tasks_per_child") != 1
    ):
        raise CalibrationReviewError("raw resource worker lifecycle is invalid")
    try:
        topology = ExecutionTopology.from_dict(
            _mapping(evidence.get("topology"), "resource topology")
        )
    except ValueError as error:
        raise CalibrationReviewError("raw resource topology is invalid") from error
    process_tree = _mapping(evidence.get("process_tree"), "resource process_tree")
    _validate_resource_thread_tree(process_tree)
    elapsed = _number(evidence.get("elapsed_seconds"), "resource elapsed", positive=True)
    replay_seconds = _number(evidence.get("independent_replay_seconds"), "resource replay")
    compute_limit = len(topology.cpu_ids)
    cpu_seconds = _number(process_tree.get("process_tree_cpu_seconds"), "process-tree CPU")
    if (
        process_tree.get("compute_thread_limit") != compute_limit
        or process_tree.get("cpu_normalized_within_limit") is not True
        or cpu_seconds > elapsed * compute_limit * 1.000001
        or process_tree.get("actual_affinity_union") != list(topology.cpu_ids)
    ):
        raise CalibrationReviewError("raw resource CPU/affinity budget is invalid")
    before = _mapping(evidence.get("cgroup_before"), "resource cgroup_before")
    after = _mapping(evidence.get("cgroup_after"), "resource cgroup_after")
    if (
        before.get("status") != "available"
        or after.get("status") != "available"
        or before.get("cgroup_path") != after.get("cgroup_path")
        or before.get("memory_swap_current_bytes") != 0
        or after.get("memory_swap_current_bytes") != 0
    ):
        raise CalibrationReviewError("raw resource cgroup/swap identity is invalid")
    before_io = _mapping(before.get("io"), "resource cgroup before I/O")
    after_io = _mapping(after.get("io"), "resource cgroup after I/O")
    reported_io = _mapping(evidence.get("cgroup_io"), "resource cgroup I/O")
    io_fields = {
        "read_bytes",
        "write_bytes",
        "read_operations",
        "write_operations",
        "discard_bytes",
        "discard_operations",
    }
    if set(before_io) != io_fields or set(after_io) != io_fields or set(reported_io) != io_fields:
        raise CalibrationReviewError("raw resource cgroup I/O schema is invalid")
    for name in io_fields:
        before_value = _integer(before_io.get(name), f"cgroup before {name}")
        after_value = _integer(after_io.get(name), f"cgroup after {name}")
        if after_value < before_value or reported_io.get(name) != after_value - before_value:
            raise CalibrationReviewError("raw resource cgroup I/O does not replay")
    for event_name in ("oom", "oom_kill"):
        before_events = _mapping(before.get("memory_events"), "cgroup before memory events")
        after_events = _mapping(after.get("memory_events"), "cgroup after memory events")
        first = _integer(before_events.get(event_name), f"cgroup before {event_name}")
        last = _integer(after_events.get(event_name), f"cgroup after {event_name}")
        if last != first:
            raise CalibrationReviewError("raw resource cgroup OOM gate failed")
    raw_paths = evidence.get("axis_relative_paths")
    if (
        not isinstance(raw_paths, list)
        or not raw_paths
        or any(not isinstance(item, str) or not item for item in raw_paths)
        or len(set(raw_paths)) != len(raw_paths)
    ):
        raise CalibrationReviewError("raw resource axis inventory is invalid")
    payloads: list[dict[str, object]] = []
    normalized_paths: list[str] = []
    for raw_relative in cast(list[str], raw_paths):
        relative = PurePosixPath(raw_relative)
        if relative.is_absolute() or ".." in relative.parts or "\\" in raw_relative:
            raise CalibrationReviewError("raw resource axis path is invalid")
        path = (run_root / Path(*relative.parts)).resolve()
        try:
            path.relative_to(run_root)
        except ValueError as error:
            raise CalibrationReviewError("raw resource axis escapes calibration run") from error
        if inventory_roles.get(raw_relative) is None:
            raise CalibrationReviewError("raw resource axis is not in the signed inventory")
        payloads.append(_load_reconciled_raw_axis(path))
        normalized_paths.append(raw_relative)
    raw_axis_timings = _array(evidence.get("axis_timings"), "resource axis timings")
    axis_timings: dict[str, tuple[float, float, float]] = {}
    for index, raw_timing in enumerate(raw_axis_timings):
        timing = _mapping(raw_timing, f"resource axis timings[{index}]")
        if set(timing) != {
            "relative_path",
            "producer_parent_terminal_seconds",
            "independent_replay_seconds",
            "end_to_end_seconds",
        }:
            raise CalibrationReviewError("resource axis timing schema is invalid")
        relative_path = _text(timing.get("relative_path"), "resource axis timing path")
        producer_terminal = _number(
            timing.get("producer_parent_terminal_seconds"),
            "resource parent terminal timing",
            positive=True,
        )
        axis_replay = _number(
            timing.get("independent_replay_seconds"),
            "resource axis replay timing",
        )
        axis_end_to_end = _number(
            timing.get("end_to_end_seconds"),
            "resource axis end-to-end timing",
            positive=True,
        )
        if relative_path in axis_timings or not math.isclose(
            axis_end_to_end,
            producer_terminal + axis_replay,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise CalibrationReviewError("resource axis timing does not reconcile")
        axis_timings[relative_path] = (
            producer_terminal,
            axis_replay,
            axis_end_to_end,
        )
    if set(axis_timings) != set(normalized_paths):
        raise CalibrationReviewError("resource axis timing inventory is incomplete")
    worker_times: list[float] = []
    for relative_path, payload in zip(normalized_paths, payloads, strict=True):
        producer_terminal, _axis_replay, axis_end_to_end = axis_timings[relative_path]
        producer_pre_receipt = _number(
            payload.get("producer_pre_receipt_seconds"),
            "resource producer pre-receipt timing",
        )
        if producer_terminal + 1e-9 < producer_pre_receipt:
            raise CalibrationReviewError("parent terminal timing omits producer publication")
        worker_times.append(axis_end_to_end)
    if sum(value[1] for value in axis_timings.values()) > replay_seconds + 1e-6:
        raise CalibrationReviewError("resource aggregate replay timing is incomplete")
    queue_wait, queue_depth, pending_peak, queue_full, rejected = _resource_queue_totals(
        payloads,
        scheduler_statistics,
    )
    schedstat = _mapping(process_tree.get("thread_tree_schedstat"), "resource schedstat")
    scheduler_pid = evidence.get("scheduler_process_id")
    scheduler_pss = 0
    if scheduler_pid is not None:
        scheduler_pid_value = _integer(scheduler_pid, "resource scheduler PID")
        values = [
            row.get("maximum_pss_bytes")
            for row in _array(process_tree.get("process_metrics"), "resource process metrics")
            if isinstance(row, Mapping) and row.get("pid") == scheduler_pid_value
        ]
        if len(values) != 1:
            raise CalibrationReviewError("resource scheduler PSS identity is ambiguous")
        scheduler_pss = _integer(values[0], "resource scheduler PSS")
    summary = RuntimeResourceSummaryV2(
        elapsed_seconds=elapsed,
        effective_cores=cpu_seconds / elapsed,
        cpu_utilization_fraction=min(1.0, cpu_seconds / (elapsed * compute_limit)),
        user_cpu_seconds=_number(
            process_tree.get("process_tree_user_cpu_seconds"), "resource user CPU"
        ),
        system_cpu_seconds=_number(
            process_tree.get("process_tree_system_cpu_seconds"), "resource system CPU"
        ),
        run_queue_wait_seconds=_integer(schedstat.get("runqueue_delay_ns"), "runqueue delay")
        / 1_000_000_000.0,
        context_switches=_integer(
            process_tree.get("thread_tree_context_switches"), "resource context switches"
        ),
        cpu_migrations=_integer(
            process_tree.get("thread_tree_cpu_migrations"), "resource CPU migrations"
        ),
        minor_faults=_integer(
            process_tree.get("thread_tree_minor_faults"), "resource minor faults"
        ),
        major_faults=_integer(
            process_tree.get("thread_tree_major_faults"), "resource major faults"
        ),
        rss_bytes=_integer(process_tree.get("peak_aggregate_rss_bytes"), "resource peak RSS"),
        pss_bytes=_integer(process_tree.get("peak_aggregate_pss_bytes"), "resource peak PSS"),
        cgroup_memory_current_bytes=_integer(
            after.get("memory_current_bytes"), "resource cgroup current memory"
        ),
        cgroup_memory_peak_bytes=_integer(
            after.get("memory_peak_bytes"), "resource cgroup peak memory"
        ),
        io_read_bytes=_integer(reported_io.get("read_bytes"), "resource read bytes"),
        io_write_bytes=_integer(reported_io.get("write_bytes"), "resource write bytes"),
        queue_wait_seconds=queue_wait,
        queue_depth_peak=queue_depth,
        pending_tasks_peak=pending_peak,
        queue_full_count=queue_full,
        rejected_count=rejected,
        worker_p95_seconds=_resource_percentile(worker_times, 0.95),
        worker_max_seconds=max(worker_times),
        worker_min_seconds=min(worker_times),
        startup_seconds=max(
            _number(payload.get("startup_seconds"), "resource axis startup") for payload in payloads
        ),
        solver_seconds=max(
            _number(payload.get("solver_seconds"), "resource axis solver") for payload in payloads
        ),
        persistence_seconds=max(
            _number(payload.get("persistence_seconds"), "resource axis persistence")
            for payload in payloads
        ),
        replay_seconds=replay_seconds,
        p50_end_to_end_seconds=_resource_percentile(worker_times, 0.50),
        p95_end_to_end_seconds=_resource_percentile(worker_times, 0.95),
        p99_end_to_end_seconds=_resource_percentile(worker_times, 0.99),
        max_end_to_end_seconds=max(worker_times),
    )
    return summary, scheduler_pss, tuple(normalized_paths)


def _validate_resource_evidence(
    provenance: Mapping[str, object],
    *,
    run_root: Path | None = None,
    inventory_roles: Mapping[str, str] | None = None,
    replayed: dict[tuple[str, str, str, str], dict[str, object]] | None = None,
) -> int:
    resources = _array(provenance.get("resource_summaries"), "resource_summaries")
    if not resources:
        raise CalibrationReviewError("resource summary inventory is empty")
    for index, raw in enumerate(resources):
        row = _mapping(raw, f"resource_summaries[{index}]")
        try:
            summary = RuntimeResourceSummaryV2.from_dict(
                _mapping(row.get("resource_summary"), "resource_summary")
            )
        except ValueError as error:
            raise CalibrationReviewError("runtime resource summary is invalid") from error
        if (
            summary.cpu_utilization_fraction > 1.0
            or summary.queue_full_count != 0
            or summary.rejected_count != 0
            or summary.pss_bytes <= 0
            or summary.swap_in_bytes != 0
            or summary.swap_out_bytes != 0
        ):
            raise CalibrationReviewError("runtime CPU/memory/queue resource gate failed")
        if run_root is not None:
            if inventory_roles is None:
                raise CalibrationReviewError("resource inventory roles are unavailable")
            scheduler_raw = row.get("scheduler_statistics")
            scheduler_statistics = (
                None
                if scheduler_raw is None
                else _mapping(scheduler_raw, "resource scheduler statistics")
            )
            raw_resource = _mapping(row.get("raw_resource_statistics"), "raw_resource_statistics")
            derived, scheduler_pss, paths = _derive_resource_summary(
                raw_resource,
                run_root=run_root,
                inventory_roles=inventory_roles,
                scheduler_statistics=scheduler_statistics,
            )
            if derived.to_dict() != summary.to_dict():
                raise CalibrationReviewError(
                    "runtime resource summary does not independently replay"
                )
            role = _text(row.get("role"), "resource role")
            if any(inventory_roles[path] != role for path in paths):
                raise CalibrationReviewError("resource row does not bind its raw-axis role")
            if replayed is not None:
                key = (
                    _text(row.get("mode"), "resource mode"),
                    _text(row.get("workload_class"), "resource workload"),
                    _text(row.get("topology_id"), "resource topology ID"),
                    role,
                )
                if key in replayed:
                    raise CalibrationReviewError("resource identity is duplicated")
                replayed[key] = {
                    "summary": derived,
                    "scheduler_pss_bytes": scheduler_pss,
                    "topology": raw_resource["topology"],
                    "axis_relative_paths": paths,
                    "scheduler_statistics": scheduler_statistics,
                }
        if row.get("role") == "admitted-mode-block":
            diagnostics = _mapping(
                row.get("parallel_diagnostics"),
                "parallel_diagnostics",
            )
            expected_fields = {
                "schema_version",
                "shard_count",
                "isolated_axis_end_to_end_seconds",
                "sequential_projected_seconds",
                "mode_block_end_to_end_seconds",
                "speedup",
                "parallel_efficiency",
                "axes_per_hour",
                "cpu_seconds_per_axis",
                "axes_per_cpu_second",
                "worker_p95_over_min",
                "worker_max_over_min",
            }
            if (
                set(diagnostics) != expected_fields
                or diagnostics.get("schema_version") != "stage05.2-topology-parallel-diagnostics-v1"
            ):
                raise CalibrationReviewError("parallel diagnostics schema is invalid")
            shards = _integer(diagnostics.get("shard_count"), "parallel shard_count")
            if shards == 0:
                raise CalibrationReviewError("parallel shard_count must be positive")
            isolated = _number(
                diagnostics.get("isolated_axis_end_to_end_seconds"),
                "isolated axis E2E",
                positive=True,
            )
            sequential = _number(
                diagnostics.get("sequential_projected_seconds"),
                "sequential projected E2E",
                positive=True,
            )
            block = _number(
                diagnostics.get("mode_block_end_to_end_seconds"),
                "mode-block E2E",
                positive=True,
            )
            speedup = _number(diagnostics.get("speedup"), "parallel speedup", positive=True)
            efficiency = _number(
                diagnostics.get("parallel_efficiency"),
                "parallel efficiency",
                positive=True,
            )
            axes_per_hour = _number(
                diagnostics.get("axes_per_hour"),
                "axes per hour",
                positive=True,
            )
            cpu_seconds = summary.user_cpu_seconds + summary.system_cpu_seconds
            cpu_seconds_per_axis = _number(
                diagnostics.get("cpu_seconds_per_axis"),
                "CPU seconds per axis",
            )
            axes_per_cpu_second = _number(
                diagnostics.get("axes_per_cpu_second"),
                "axes per CPU second",
                positive=True,
            )
            expected_block = summary.elapsed_seconds + summary.replay_seconds
            comparisons = (
                (sequential, isolated * shards),
                (block, expected_block),
                (speedup, sequential / block),
                (efficiency, speedup / shards),
                (axes_per_hour, 3600.0 * shards / block),
                (cpu_seconds_per_axis, cpu_seconds / shards),
                (axes_per_cpu_second, shards / max(cpu_seconds, 1e-12)),
            )
            if any(
                not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9)
                for observed, expected in comparisons
            ):
                raise CalibrationReviewError("parallel diagnostics do not replay")
            worker_min = summary.worker_min_seconds
            for field_name, numerator in (
                ("worker_p95_over_min", summary.worker_p95_seconds),
                ("worker_max_over_min", summary.worker_max_seconds),
            ):
                raw_ratio = diagnostics.get(field_name)
                if worker_min <= 0.0:
                    if raw_ratio != "unavailable":
                        raise CalibrationReviewError("worker imbalance availability is invalid")
                elif not math.isclose(
                    _number(raw_ratio, field_name, positive=True),
                    numerator / worker_min,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise CalibrationReviewError("worker imbalance diagnostics do not replay")
    return len(resources)


def _validate_observation_resource_bindings(
    observation_payload: Mapping[str, object],
    rows: Sequence[AxisObservation],
    resources: Mapping[tuple[str, str, str, str], Mapping[str, object]],
) -> None:
    for row in rows:
        identity = (row.mode, row.workload_class, row.topology_id)
        isolated = resources.get((*identity, "isolated-memory-probe"))
        admitted = resources.get((*identity, "admitted-mode-block"))
        if isolated is None or admitted is None:
            raise CalibrationReviewError("observation lacks isolated/admitted resource evidence")
        isolated_summary = isolated.get("summary")
        admitted_summary = admitted.get("summary")
        if not isinstance(isolated_summary, RuntimeResourceSummaryV2) or not isinstance(
            admitted_summary, RuntimeResourceSummaryV2
        ):
            raise CalibrationReviewError("observation resource replay is unavailable")
        if (
            isolated.get("topology") != row.topology.to_dict()
            or admitted.get("topology") != row.topology.to_dict()
            or row.producer_end_to_end_seconds != admitted_summary.elapsed_seconds
            or row.independent_replay_seconds != admitted_summary.replay_seconds
            or row.end_to_end_seconds
            != admitted_summary.elapsed_seconds + admitted_summary.replay_seconds
            or row.pss_bytes != isolated_summary.pss_bytes
            or row.scheduler_pss_bytes != isolated.get("scheduler_pss_bytes")
        ):
            raise CalibrationReviewError("observation timing/PSS differs from raw resources")

    build_profile = _text(observation_payload.get("build_profile"), "observation build profile")
    lifecycle_root = _mapping(
        observation_payload.get("host_scheduler_lifecycle_evidence"),
        "host scheduler lifecycle evidence",
    )
    for workload, raw_receipts in lifecycle_root.items():
        if not isinstance(raw_receipts, list):
            raise CalibrationReviewError("host scheduler lifecycle matrix is invalid")
        for index, raw_receipt in enumerate(raw_receipts):
            receipt = _mapping(raw_receipt, f"host scheduler lifecycle {workload}[{index}]")
            topology_id = _text(receipt.get("topology_id"), "lifecycle topology ID")
            try:
                topology = ExecutionTopology.from_dict(
                    _mapping(receipt.get("topology"), "lifecycle topology")
                )
            except ValueError as error:
                raise CalibrationReviewError("lifecycle topology is invalid") from error
            if topology.workload_class != workload or receipt.get("build_profile") != build_profile:
                raise CalibrationReviewError("lifecycle build/workload identity differs")
            per_wave_rows = [
                resources.get(
                    (
                        "host_scheduler",
                        workload,
                        topology_id,
                        f"scheduler-per-wave-{wave}",
                    )
                )
                for wave in (1, 2)
            ]
            mode_block = resources.get(
                ("host_scheduler", workload, topology_id, "scheduler-mode-block")
            )
            if mode_block is None or any(row is None for row in per_wave_rows):
                raise CalibrationReviewError("lifecycle resource rows are incomplete")
            selected_per_wave = cast(list[Mapping[str, object]], per_wave_rows)
            per_wave_summaries = [
                cast(RuntimeResourceSummaryV2, item["summary"]) for item in selected_per_wave
            ]
            mode_summary = cast(RuntimeResourceSummaryV2, mode_block["summary"])
            per_wave_seconds = sum(
                summary.elapsed_seconds + summary.replay_seconds for summary in per_wave_summaries
            )
            mode_seconds = mode_summary.elapsed_seconds + mode_summary.replay_seconds
            per_wave_scheduler_pss = max(
                _integer(item.get("scheduler_pss_bytes"), "per-wave scheduler PSS")
                for item in selected_per_wave
            )
            mode_scheduler_pss = _integer(
                mode_block.get("scheduler_pss_bytes"), "mode-block scheduler PSS"
            )
            per_wave_process_pss = max(summary.pss_bytes for summary in per_wave_summaries)
            mode_process_pss = mode_summary.pss_bytes
            rss_stable = (
                per_wave_process_pss > 0
                and mode_process_pss > 0
                and mode_process_pss <= math.ceil(per_wave_process_pss * 1.20)
            )
            if (
                receipt.get("wave_count") != 2
                or receipt.get("session_isolation_passed") is not True
                or receipt.get("cache_reset_passed") is not True
                or receipt.get("semantic_replay_passed") is not True
                or receipt.get("rss_stability_passed") is not rss_stable
                or receipt.get("mode_block_faster") is not (mode_seconds < per_wave_seconds)
                or receipt.get("per_wave_end_to_end_seconds") != per_wave_seconds
                or receipt.get("mode_block_end_to_end_seconds") != mode_seconds
                or receipt.get("per_wave_scheduler_pss_peak_bytes") != per_wave_scheduler_pss
                or receipt.get("mode_block_scheduler_pss_peak_bytes") != mode_scheduler_pss
                or receipt.get("per_wave_process_tree_pss_peak_bytes") != per_wave_process_pss
                or receipt.get("mode_block_process_tree_pss_peak_bytes") != mode_process_pss
            ):
                raise CalibrationReviewError(
                    "host scheduler lifecycle does not independently replay"
                )


def _validate_scheduler_task_receipt_evidence(
    provenance: Mapping[str, object],
    *,
    run_root: Path,
) -> int:
    raw_inventory = _array(
        provenance.get("scheduler_task_receipt_inventory"),
        "scheduler_task_receipt_inventory",
    )
    indexed: dict[str, tuple[Path, Path, Mapping[str, object]]] = {}
    for index, raw in enumerate(raw_inventory):
        entry = _mapping(raw, f"scheduler_task_receipt_inventory[{index}]")
        if entry.get("storage_alias") != "stage052-performance-calibration-run":
            raise CalibrationReviewError("scheduler task-receipt storage alias is invalid")
        relative = PurePosixPath(_text(entry.get("relative_path"), "task receipt path"))
        relative_sidecar = PurePosixPath(
            _text(entry.get("relative_sidecar_path"), "task receipt sidecar path")
        )
        if any(
            value.is_absolute() or ".." in value.parts for value in (relative, relative_sidecar)
        ):
            raise CalibrationReviewError("scheduler task-receipt portable path is invalid")
        path = (run_root / Path(*relative.parts)).resolve()
        sidecar = (run_root / Path(*relative_sidecar.parts)).resolve()
        try:
            path.relative_to(run_root)
            sidecar.relative_to(run_root)
        except ValueError as error:
            raise CalibrationReviewError(
                "scheduler task-receipt escapes calibration run"
            ) from error
        if (
            path.name in indexed
            or entry.get("sha256") != _sha256_file(path)
            or entry.get("sidecar_sha256") != _sha256_file(sidecar)
            or sidecar.name != path.name + ".sha256"
        ):
            raise CalibrationReviewError("scheduler task-receipt inventory does not reconcile")
        indexed[path.name] = (path, sidecar, entry)
    resources = _array(provenance.get("resource_summaries"), "resource_summaries")
    observed: set[str] = set()
    from evrptw.experiments.stage052_native_architecture_review import (
        _review_scheduler_runtime_statistics,
    )

    for raw in resources:
        resource = _mapping(raw, "resource summary")
        statistics = resource.get("scheduler_statistics")
        if statistics is None:
            continue
        scheduler_statistics = _mapping(statistics, "scheduler_statistics")
        descriptor = _mapping(
            scheduler_statistics.get("task_receipts"),
            "scheduler task_receipts",
        )
        filename = _text(descriptor.get("path"), "scheduler task receipt filename")
        selected = indexed.get(filename)
        if selected is None:
            raise CalibrationReviewError("scheduler task-receipt file is not inventoried")
        worker_threads = _integer(
            scheduler_statistics.get("worker_threads"), "scheduler worker_threads"
        )
        request_threads = _integer(
            scheduler_statistics.get("request_threads"), "scheduler request_threads"
        )
        try:
            _review_scheduler_runtime_statistics(
                scheduler_statistics,
                worker_threads=worker_threads,
                request_threads=request_threads,
                evidence_root=selected[0].parent,
            )
        except RuntimeError as error:
            raise CalibrationReviewError(
                "scheduler task-receipt independent replay failed"
            ) from error
        observed.add(filename)
    if observed != set(indexed):
        raise CalibrationReviewError("scheduler task-receipt inventory has orphan files")
    return len(observed)


def _validate_memory_rejections(
    observation_payload: Mapping[str, object],
) -> int:
    rows = _array(observation_payload.get("memory_rejections"), "memory_rejections")
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"memory_rejections[{index}]")
        try:
            topology = ExecutionTopology.from_dict(
                _mapping(row.get("topology"), "memory rejection topology")
            )
        except ValueError as error:
            raise CalibrationReviewError("memory rejection topology is invalid") from error
        isolated = _integer(row.get("isolated_pss_bytes"), "isolated_pss_bytes")
        scheduler = _integer(row.get("scheduler_pss_bytes"), "scheduler_pss_bytes")
        projected = _integer(
            row.get("projected_concurrent_pss_bytes"),
            "projected_concurrent_pss_bytes",
        )
        effective_limit = _integer(
            row.get("effective_memory_limit_bytes"),
            "effective_memory_limit_bytes",
        )
        admission_available = _integer(
            row.get("admission_available_bytes"),
            "admission_available_bytes",
        )
        admission_required = _integer(
            row.get("admission_required_bytes"),
            "admission_required_bytes",
        )
        admission_headroom = _integer(
            row.get("admission_headroom_bytes"),
            "admission_headroom_bytes",
        )
        swap_used = _integer(row.get("swap_used_bytes"), "swap_used_bytes")
        swap_current = _integer(row.get("swap_current_bytes"), "swap_current_bytes")
        expected = scheduler + (isolated - scheduler) * topology.shard_count
        over_eighty = projected > math.floor(effective_limit * 0.80)
        below_headroom = admission_available < admission_required + admission_headroom
        swap_failure = swap_used > 0 or swap_current > 0
        if (
            scheduler > isolated
            or projected != expected
            or admission_required != projected
            or admission_available != effective_limit
            or admission_headroom != math.ceil(projected * 0.20)
            or not (over_eighty or below_headroom or swap_failure)
        ):
            raise CalibrationReviewError("memory rejection does not independently reproduce")
    return len(rows)


def _axis_projection(row: AxisObservation) -> dict[str, object]:
    return {
        "objective": row.objective,
        "routes": row.routes,
        "candidate_trajectory": row.candidate_trajectory,
        "exact_order": row.exact_order,
        "cache_lifecycle": row.cache_lifecycle,
        "transaction_hashes": list(row.transaction_hashes),
    }


def _replay_observation_children(
    observation_paths: Sequence[Path],
    observations: Sequence[AxisObservation],
    *,
    benchmark_dir: Path,
) -> tuple[int, int, int]:
    rows_by_source: dict[Path, list[AxisObservation]] = defaultdict(list)
    for row in observations:
        if row.source_observation_path is None:
            raise CalibrationReviewError("observation source identity is missing")
        rows_by_source[row.source_observation_path.resolve()].append(row)
    raw_count = 0
    resource_count = 0
    rejection_count = 0
    global_projection: dict[tuple[str, int], bytes] = {}
    for observation_path in observation_paths:
        payload, _digest = _load_signed_json(observation_path, "fixed-work observation")
        provenance = _mapping(payload.get("producer_provenance"), "producer_provenance")
        if provenance.get("schema_version") != OBSERVATION_PRODUCER_SCHEMA_VERSION:
            raise CalibrationReviewError("observation producer schema is unsupported")
        rejection_count += _validate_memory_rejections(payload)
        build_identity = _mapping(payload.get("build_identity"), "observation build_identity")
        inventory = _array(provenance.get("raw_axis_inventory"), "raw_axis_inventory")
        parent_run_label = _text(provenance.get("parent_run_label"), "parent_run_label")
        run_roots = [
            parent
            for parent in observation_path.resolve().parents
            if parent.name == parent_run_label
        ]
        if len(run_roots) != 1:
            raise CalibrationReviewError("observation calibration-run alias is ambiguous")
        run_root = run_roots[0]
        _validate_scheduler_task_receipt_evidence(provenance, run_root=run_root)
        admitted: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
        inventory_roles: dict[str, str] = {}
        for index, raw in enumerate(inventory):
            entry = _mapping(raw, f"raw_axis_inventory[{index}]")
            _review_axis_supporting_artifacts(entry, run_root=run_root)
            if entry.get("storage_alias") != "stage052-performance-calibration-run":
                raise CalibrationReviewError("raw axis storage alias is invalid")
            relative_path = PurePosixPath(
                _text(entry.get("relative_path"), "raw axis relative_path")
            )
            relative_sidecar = PurePosixPath(
                _text(
                    entry.get("relative_sidecar_path"),
                    "raw axis relative_sidecar_path",
                )
            )
            if any(
                value.is_absolute() or ".." in value.parts
                for value in (relative_path, relative_sidecar)
            ):
                raise CalibrationReviewError("raw axis portable path is invalid")
            path = (run_root / Path(*relative_path.parts)).resolve()
            sidecar = (run_root / Path(*relative_sidecar.parts)).resolve()
            try:
                path.relative_to(run_root)
                sidecar.relative_to(run_root)
            except ValueError as error:
                raise CalibrationReviewError("raw axis escapes calibration run") from error
            expected = entry.get("sha256")
            if not _is_sha256(expected) or _sha256_file(path) != expected:
                raise CalibrationReviewError("raw axis inventory hash mismatch")
            sidecar_expected = entry.get("sidecar_sha256")
            if not _is_sha256(sidecar_expected) or _sha256_file(sidecar) != sidecar_expected:
                raise CalibrationReviewError("raw axis sidecar inventory hash mismatch")
            role = _text(entry.get("role"), "raw axis role")
            relative_key = relative_path.as_posix()
            if relative_key in inventory_roles:
                raise CalibrationReviewError("raw axis inventory path is duplicated")
            inventory_roles[relative_key] = role
            projection, mode, workload, topology_id = _review_raw_axis(
                path,
                benchmark_dir=benchmark_dir,
                build_identity=build_identity,
            )
            raw_count += 1
            projection_bytes = _canonical(projection)
            instance = _text(
                _mapping(json.loads(path.read_text(encoding="utf-8")), "raw payload").get(
                    "instance"
                ),
                "raw instance",
            )
            repeat = _integer(payload.get("repeat"), "observation repeat")
            semantic_key = (instance, repeat)
            prior = global_projection.setdefault(semantic_key, projection_bytes)
            if prior != projection_bytes:
                raise CalibrationReviewError("fixed-work raw semantics differ across modes/builds")
            if entry.get("role") == "admitted-mode-block":
                admitted[(mode, workload, topology_id)].append(projection)
        resource_replays: dict[tuple[str, str, str, str], dict[str, object]] = {}
        resource_count += _validate_resource_evidence(
            provenance,
            run_root=run_root,
            inventory_roles=inventory_roles,
            replayed=resource_replays,
        )
        expected_rows = rows_by_source.get(observation_path.resolve(), [])
        if not expected_rows:
            raise CalibrationReviewError("signed observation produced no parsed axes")
        _validate_observation_resource_bindings(payload, expected_rows, resource_replays)
        if set(admitted) != {
            (row.mode, row.workload_class, row.topology_id) for row in expected_rows
        }:
            raise CalibrationReviewError("admitted raw topology matrix differs from observation")
        for row in expected_rows:
            projections = admitted[(row.mode, row.workload_class, row.topology_id)]
            if len(projections) != row.topology.shard_count:
                raise CalibrationReviewError("admitted raw shard count differs from topology")
            expected_projection = _axis_projection(row)
            if any(value != expected_projection for value in projections):
                raise CalibrationReviewError("observation projection differs from raw axes")
    return raw_count, resource_count, rejection_count


def _review_axis_supporting_artifacts(
    entry: Mapping[str, object],
    *,
    run_root: Path,
) -> None:
    expected_fields = {
        "storage_alias",
        "relative_path",
        "sha256",
        "relative_sidecar_path",
        "sidecar_sha256",
        "role",
        "supporting_artifacts",
    }
    if set(entry) != expected_fields:
        raise CalibrationReviewError("raw axis inventory fields are invalid")
    supporting = _array(
        entry.get("supporting_artifacts"),
        "raw axis supporting artifacts",
    )
    if not supporting:
        raise CalibrationReviewError("raw axis supporting-artifact inventory is empty")
    observed_paths: set[Path] = set()
    observed_roles: set[str] = set()
    for index, raw_support in enumerate(supporting):
        support = _mapping(raw_support, f"raw axis supporting artifact {index}")
        if set(support) != {"relative_path", "sha256", "role"}:
            raise CalibrationReviewError("raw axis supporting-artifact fields are invalid")
        relative = PurePosixPath(_text(support.get("relative_path"), "supporting artifact path"))
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise CalibrationReviewError("raw axis supporting-artifact path is invalid")
        path = (run_root / Path(*relative.parts)).resolve()
        try:
            path.relative_to(run_root)
        except ValueError as error:
            raise CalibrationReviewError(
                "raw axis supporting artifact escapes calibration run"
            ) from error
        expected_sha256 = support.get("sha256")
        role = _text(support.get("role"), "supporting artifact role")
        if (
            path in observed_paths
            or not _is_sha256(expected_sha256)
            or path.is_symlink()
            or not path.is_file()
            or _sha256_file(path) != expected_sha256
        ):
            raise CalibrationReviewError("raw axis supporting-artifact hash differs")
        observed_paths.add(path)
        observed_roles.add(role)
    if not {
        "axis-persistence-receipt",
        "axis-persistence-receipt-sidecar",
    }.issubset(observed_roles):
        raise CalibrationReviewError("raw axis persistence receipt is not fully inventoried")


def _replay_telemetry_children(
    overhead: TelemetryOverheadReceipt,
    *,
    run_root: Path,
    benchmark_dir: Path,
    build_identity: Mapping[str, object],
) -> int:
    overhead.require_representative_fixed_work()
    run_root = run_root.resolve()
    evidence = overhead.workload_evidence
    warm = evidence.get("warm_sample_evidence")
    pairs = evidence.get("paired_sample_evidence")
    if not isinstance(warm, tuple) or not isinstance(pairs, tuple):
        raise CalibrationReviewError("telemetry raw sample matrix is missing")
    surface_fields = {
        "semantic_telemetry",
        "physical_telemetry",
        "persistence",
        "independent_replay",
    }
    base_evidence = {
        key: value
        for key, value in evidence.items()
        if key
        not in {
            "warm_sample_evidence",
            "paired_sample_evidence",
            "unmonitored_telemetry_surface",
            "monitored_telemetry_surface",
            *surface_fields,
        }
    }
    expected_fingerprint: str | None = None
    replayed_axes = 0

    def resolve_relative(value: object, field: str) -> Path:
        relative = PurePosixPath(_text(value, field))
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise CalibrationReviewError(f"{field} is not a portable relative path")
        resolved = (run_root / Path(*relative.parts)).resolve()
        try:
            resolved.relative_to(run_root)
        except ValueError as error:
            raise CalibrationReviewError(f"{field} escapes the calibration run") from error
        return resolved

    def replay_sample(
        sample: object,
        *,
        enabled: bool,
        sample_index: int,
        expected_seconds: float | None,
    ) -> None:
        nonlocal expected_fingerprint, replayed_axes
        row = _mapping(sample, "telemetry sample evidence")
        expected_workload_evidence = {
            **base_evidence,
            **{field: enabled for field in surface_fields},
        }
        if (
            row.get("enabled") is not enabled
            or row.get("sample_index") != sample_index
            or row.get("workload_evidence") != expected_workload_evidence
        ):
            raise CalibrationReviewError("telemetry sample identity differs")
        elapsed = _number(row.get("elapsed_seconds"), "telemetry sample elapsed", positive=True)
        if expected_seconds is not None and elapsed != expected_seconds:
            raise CalibrationReviewError("telemetry paired timing does not replay")
        fingerprint = _text(row.get("fingerprint"), "telemetry sample fingerprint")
        if not _is_sha256(fingerprint):
            raise CalibrationReviewError("telemetry sample fingerprint is invalid")
        if expected_fingerprint is None:
            expected_fingerprint = fingerprint
        elif fingerprint != expected_fingerprint:
            raise CalibrationReviewError("telemetry on/off semantic fingerprint diverged")
        resources = _mapping(row.get("resource_summary"), "telemetry sample resources")
        if resources.get("sample_storage_alias") != "stage052-performance-calibration-run":
            raise CalibrationReviewError("telemetry sample storage alias is invalid")
        sample_path = resolve_relative(
            resources.get("sample_relative_path"),
            "telemetry sample path",
        )
        sample_sidecar = resolve_relative(
            resources.get("sample_sidecar_relative_path"),
            "telemetry sample sidecar path",
        )
        sample_payload, sample_sha256 = _load_signed_json(
            sample_path,
            "representative telemetry sample",
        )
        if (
            resources.get("sample_sha256") != sample_sha256
            or resources.get("sample_sidecar_sha256") != _sha256_file(sample_sidecar)
            or sample_sidecar != Path(f"{sample_path}.sha256")
            or sample_payload.get("schema_version")
            != "stage05.2-representative-telemetry-sample-v3"
            or sample_payload.get("enabled") is not enabled
            or sample_payload.get("fingerprint") != fingerprint
            or not _json_equivalent(
                sample_payload.get("workload_evidence"),
                expected_workload_evidence,
            )
            or not _json_equivalent(
                sample_payload.get("raw_axis_inventory"),
                resources.get("raw_axis_inventory"),
            )
        ):
            raise CalibrationReviewError("telemetry sample receipt does not reconcile")
        child_elapsed = _number(
            sample_payload.get("elapsed_seconds"),
            "telemetry child elapsed",
            positive=True,
        )
        if resources.get("child_elapsed_seconds") != child_elapsed or elapsed < child_elapsed:
            raise CalibrationReviewError("telemetry parent/child timing is invalid")
        raw_inventory = _array(sample_payload.get("raw_axis_inventory"), "telemetry raw axes")
        expected_axis_count = 1 if enabled else 0
        if len(raw_inventory) != expected_axis_count:
            raise CalibrationReviewError("telemetry on/off raw axis surface is invalid")
        fingerprint_payload = _mapping(
            sample_payload.get("fingerprint_payload"),
            "telemetry fingerprint payload",
        )
        if set(fingerprint_payload) != {
            "objective",
            "routes",
            "candidate_work_hash",
            "route_result_hash",
            "effective_iterations",
            "termination_reason",
            "accepted_moves",
            "rejected_moves",
            "exact_started_calls",
            "exact_completed_calls",
            "exact_interrupted_calls",
            *_REPRESENTATIVE_TRAJECTORY_FIELDS,
        }:
            raise CalibrationReviewError("telemetry fingerprint surface is incomplete")
        replayed_fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if replayed_fingerprint != fingerprint:
            raise CalibrationReviewError("telemetry minimal fingerprint does not replay")
        if not enabled:
            minimal = _mapping(
                sample_payload.get("minimal_replay_receipt"),
                "telemetry-off minimal replay receipt",
            )
            if (
                set(minimal)
                != {
                    "schema_version",
                    "instance",
                    "seed",
                    "routes",
                    "objective",
                    "candidate_work_events",
                    "route_result_events",
                    "candidate_work_hash",
                    "route_result_hash",
                    "trajectory_rows",
                }
                or minimal.get("schema_version") != "stage05.2-telemetry-off-minimal-replay-v1"
                or minimal.get("instance") != base_evidence.get("instance")
                or minimal.get("seed") != base_evidence.get("seed")
            ):
                raise CalibrationReviewError("telemetry-off minimal receipt identity differs")
            instance_name = _text(minimal.get("instance"), "telemetry-off instance")
            instance = parse_schneider(benchmark_dir / f"{instance_name}.txt")
            routes = _routes(minimal)
            report = validate_routes(instance, routes)
            if not report.feasible:
                raise CalibrationReviewError("telemetry-off validator replay failed")
            objective = SolutionObjective.from_report(instance, report)
            if list(objective.key) != minimal.get("objective"):
                raise CalibrationReviewError("telemetry-off objective replay failed")
            candidate_work = _array(
                minimal.get("candidate_work_events"),
                "telemetry-off candidate work",
            )
            route_results = _array(
                minimal.get("route_result_events"),
                "telemetry-off route results",
            )
            if (
                not all(isinstance(row, Mapping) for row in candidate_work)
                or not all(isinstance(row, Mapping) for row in route_results)
                or stable_candidate_payload_hash(tuple(candidate_work))
                != minimal.get("candidate_work_hash")
                or stable_candidate_payload_hash(tuple(route_results))
                != minimal.get("route_result_hash")
                or any(
                    minimal.get(field) != fingerprint_payload.get(field)
                    for field in (
                        "routes",
                        "objective",
                        "candidate_work_hash",
                        "route_result_hash",
                    )
                )
            ):
                raise CalibrationReviewError("telemetry-off transaction hashes do not replay")
            trajectory_rows = _mapping(
                minimal.get("trajectory_rows"),
                "telemetry-off trajectory rows",
            )
            if set(trajectory_rows) != set(_REPRESENTATIVE_TRAJECTORY_FIELDS):
                raise CalibrationReviewError("telemetry-off trajectory surface is incomplete")
            total_rows = 0
            total_bytes = 0
            for field in _REPRESENTATIVE_TRAJECTORY_FIELDS:
                rows = _array(
                    trajectory_rows.get(field),
                    f"telemetry-off {field} rows",
                )
                receipt, payload_bytes = _representative_row_evidence(rows)
                total_rows += len(rows)
                total_bytes += payload_bytes
                expected_receipt: Mapping[str, object]
                if field == "semantic_trajectory":
                    expected_receipt = {
                        "schema_version": "stage05.2-external-semantic-trajectory-v1",
                        "source": "canonical_semantic_journal:operator",
                        **receipt,
                    }
                else:
                    expected_receipt = receipt
                if fingerprint_payload.get(field) != expected_receipt:
                    raise CalibrationReviewError(f"telemetry-off {field} digest does not replay")
            if (
                total_rows > _REPRESENTATIVE_TRAJECTORY_MAX_ROWS
                or total_bytes > _REPRESENTATIVE_TRAJECTORY_MAX_BYTES
            ):
                raise CalibrationReviewError("telemetry-off trajectory exceeds its bounded receipt")
            return
        if sample_payload.get("minimal_replay_receipt") is not None:
            raise CalibrationReviewError("telemetry-on sample carries an off-only receipt")
        item = _mapping(raw_inventory[0], "telemetry raw axis inventory")
        _review_axis_supporting_artifacts(item, run_root=run_root)
        if item.get("storage_alias") != "stage052-performance-calibration-run":
            raise CalibrationReviewError("telemetry raw axis storage alias is invalid")
        axis_path = resolve_relative(item.get("relative_path"), "telemetry raw axis path")
        axis_sidecar = resolve_relative(
            item.get("relative_sidecar_path"),
            "telemetry raw axis sidecar path",
        )
        if (
            item.get("sha256") != _sha256_file(axis_path)
            or item.get("sidecar_sha256") != _sha256_file(axis_sidecar)
            or axis_sidecar != Path(f"{axis_path}.sha256")
        ):
            raise CalibrationReviewError("telemetry raw axis inventory hash differs")
        _review_raw_axis(
            axis_path,
            benchmark_dir=benchmark_dir,
            build_identity=build_identity,
        )
        axis_payload, _axis_digest = _load_signed_json(axis_path, "telemetry raw axis")
        if any(axis_payload.get(field) != value for field, value in fingerprint_payload.items()):
            raise CalibrationReviewError("telemetry raw axis fingerprint does not replay")
        replayed_axes += 1

    replay_sample(warm[0], enabled=False, sample_index=-2, expected_seconds=None)
    replay_sample(warm[1], enabled=True, sample_index=-1, expected_seconds=None)
    for index, pair in enumerate(pairs):
        pair_row = _mapping(pair, f"telemetry pair {index}")
        if (
            pair_row.get("pair_index") != index
            or pair_row.get("order") != overhead.pair_orders[index]
        ):
            raise CalibrationReviewError("telemetry pair order does not replay")
        order = overhead.pair_orders[index]
        replay_sample(
            pair_row.get("unmonitored"),
            enabled=False,
            sample_index=index * 2 if order == "off-on" else index * 2 + 1,
            expected_seconds=overhead.unmonitored_seconds[index],
        )
        replay_sample(
            pair_row.get("monitored"),
            enabled=True,
            sample_index=index * 2 + 1 if order == "off-on" else index * 2,
            expected_seconds=overhead.monitored_seconds[index],
        )
    if (
        expected_fingerprint is None
        or hashlib.sha256(expected_fingerprint.encode("ascii")).hexdigest()
        != overhead.workload_output_sha256
    ):
        raise CalibrationReviewError("telemetry workload digest does not replay")
    overhead.require_passed()
    return replayed_axes


def derive_calibration_review(
    *,
    calibration_run_dir: Path,
    benchmark_dir: Path,
) -> dict[str, object]:
    """Independently derive a qualification receipt without publishing it."""

    started = time.perf_counter()
    run_dir = calibration_run_dir.resolve()
    receipt_path = run_dir / "calibration_receipt.json"
    receipt, receipt_sha256 = _load_signed_json(receipt_path, "calibration receipt")
    if (
        receipt.get("schema_version") != CALIBRATION_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "calibrated"
        or receipt.get("formal_started") is not False
        or receipt.get("cuda_started") is not False
        or receipt.get("attempt08_started") is not False
    ):
        raise CalibrationReviewError("calibration receipt status/schema is invalid")
    run_label = _text(receipt.get("run_label"), "calibration run_label")
    if run_dir.name != run_label:
        raise CalibrationReviewError("calibration run directory identity mismatch")
    manifest_sha256 = _validate_manifest(run_dir, receipt_path, receipt_sha256)
    profile_name = _text(receipt.get("profile_path"), "profile_path")
    profile_path = (run_dir / profile_name).resolve()
    if profile_path.parent != run_dir:
        raise CalibrationReviewError("calibration profile escapes its run directory")
    profile_payload, profile_sha256 = _load_signed_json(profile_path, "frozen profile")
    try:
        profile = FrozenPerformanceProfile.from_dict(profile_payload)
    except ValueError as error:
        raise CalibrationReviewError("frozen performance profile is invalid") from error
    if (
        receipt.get("profile_sha256") != profile_sha256
        or receipt.get("profile_canonical_sha256") != profile.canonical_sha256
    ):
        raise CalibrationReviewError("calibration receipt does not bind its profile")
    wheel_paths, observation_paths, overhead_path, host_envelope_path = _inventory_paths(
        receipt,
        run_dir=run_dir,
    )
    if host_envelope_path is not None:
        host_payload, _host_digest = _load_signed_json(
            host_envelope_path,
            "host performance envelope",
        )
        if host_payload != profile.host.to_dict():
            raise CalibrationReviewError("signed host envelope differs from frozen profile")
    try:
        wheels = load_wheel_receipts(wheel_paths)
        observations = load_fixed_work_observations(observation_paths)
        overhead = load_telemetry_overhead_receipt(overhead_path)
        derived = calibrate_performance_profile(
            run_label=run_label,
            wheel_receipts=wheels,
            observations=observations,
            host=profile.host,
            telemetry_overhead=overhead,
            telemetry_overhead_path=overhead_path,
            host_envelope_path=host_envelope_path,
        )
    except (CalibrationError, OSError, ValueError) as error:
        raise CalibrationReviewError("independent calibration derivation failed") from error
    if derived.profile.to_dict() != profile.to_dict():
        raise CalibrationReviewError("independently derived performance profile differs")
    if (
        receipt.get("selected_build_profile") != derived.selected_build_profile
        or receipt.get("selected_topology_ids") != dict(derived.selected_topology_ids)
        or receipt.get("rejected_builds") != dict(sorted(derived.rejected_builds.items()))
        or receipt.get("mode_block_e2e_medians_seconds") != derived.mode_block_medians
        or receipt.get("signed_input_inventory") != dict(derived.signed_input_inventory)
    ):
        raise CalibrationReviewError("calibration aggregate receipt does not independently replay")
    portable_builds = [wheel for wheel in wheels if wheel.build_profile == "portable-o3"]
    if len(portable_builds) != 1:
        raise CalibrationReviewError("telemetry portable-o3 build identity is ambiguous")
    telemetry_identity = portable_builds[0].artifact_identity
    telemetry_raw_count = _replay_telemetry_children(
        overhead,
        run_root=run_dir,
        benchmark_dir=benchmark_dir.resolve(),
        build_identity=telemetry_identity.to_dict(),
    )
    raw_count, resource_count, rejection_count = _replay_observation_children(
        observation_paths,
        observations,
        benchmark_dir=benchmark_dir.resolve(),
    )
    identity = profile.selected_build.artifact_identity
    if identity is None:
        raise CalibrationReviewError("selected build identity is unavailable")
    source_path = Path(__file__).resolve()
    return {
        "schema_version": CALIBRATION_REVIEW_SCHEMA_VERSION,
        "status": "qualified",
        "qualification": CALIBRATION_REVIEW_QUALIFICATION,
        "calibration_run_label": run_label,
        "storage_alias": "stage052-performance-calibration-run",
        "calibration_receipt_relative_path": receipt_path.relative_to(run_dir).as_posix(),
        "calibration_receipt_sha256": receipt_sha256,
        "calibration_manifest_sha256": manifest_sha256,
        "profile_relative_path": profile_path.relative_to(run_dir).as_posix(),
        "profile_sha256": profile_sha256,
        "profile_canonical_sha256": profile.canonical_sha256,
        "rederived_profile_canonical_sha256": derived.profile.canonical_sha256,
        "repository_revision": identity.git_revision,
        "git_tree": identity.git_tree,
        "source_manifest_sha256": identity.source_manifest_sha256,
        "selected_build_profile": derived.selected_build_profile,
        "wheel_sha256": identity.wheel_sha256,
        "native_sha256": identity.native_sha256,
        "scheduler_sha256": identity.scheduler_sha256,
        "observation_count": len(observations),
        "raw_axis_replay_count": raw_count,
        "telemetry_raw_axis_replay_count": telemetry_raw_count,
        "resource_summary_replay_count": resource_count,
        "memory_rejection_replay_count": rejection_count,
        "fixed_work_semantics_identical": True,
        "validator_objective_replay_passed": True,
        "queue_full_count": 0,
        "rejected_count": 0,
        "swap_used": False,
        "selector_recomputation_passed": True,
        "reviewer_source_sha256": _sha256_file(source_path),
        "review_seconds": time.perf_counter() - started,
        "formal_started": False,
        "cuda_started": False,
        "attempt08_started": False,
    }


def review_calibration(
    *,
    calibration_run_dir: Path,
    benchmark_dir: Path,
    output_path: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    payload = derive_calibration_review(
        calibration_run_dir=calibration_run_dir,
        benchmark_dir=benchmark_dir,
    )
    destination = (
        calibration_run_dir.resolve() / "review" / "calibration_review_receipt.json"
        if output_path is None
        else output_path.resolve()
    )
    _atomic_signed_json(destination, payload)
    return destination, payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-run-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        required=True,
        help="Explicit clean ext4 Git worktree used for source identity",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    require_clean_repository_root(arguments.repository_root)
    destination = (
        arguments.calibration_run_dir.resolve() / "review" / "calibration_review_receipt.json"
        if arguments.output is None
        else arguments.output.resolve()
    )
    try:
        path, _payload = review_calibration(
            calibration_run_dir=arguments.calibration_run_dir,
            benchmark_dir=arguments.benchmark_dir,
            output_path=destination,
        )
    except BaseException as error:
        failure = destination.with_name(destination.name + ".failed.json")
        if not failure.exists() and not Path(f"{failure}.sha256").exists():
            _atomic_signed_json(
                failure,
                {
                    "schema_version": CALIBRATION_REVIEW_FAILURE_SCHEMA_VERSION,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "formal_started": False,
                    "cuda_started": False,
                    "attempt08_started": False,
                },
            )
        raise
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CALIBRATION_REVIEW_FAILURE_SCHEMA_VERSION",
    "CALIBRATION_REVIEW_QUALIFICATION",
    "CALIBRATION_REVIEW_SCHEMA_VERSION",
    "CalibrationReviewError",
    "derive_calibration_review",
    "main",
    "review_calibration",
)
