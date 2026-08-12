"""Produce signed fixed-work observations for Stage 5.2 performance calibration.

The producer executes the real five-mode solver surface from one attested wheel.
It first measures one isolated axis for every topology, rejects unsafe projected
concurrency, and only then runs a complete mode block.  Raw solver artifacts are
retained and aggregated into the signed observation inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import re
import statistics
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, cast

from evrptw import _core as native_core
from evrptw.alns import ALNSResult
from evrptw.candidate_control import stable_candidate_payload_hash
from evrptw.experiments.stage052_native_architectures import (
    AXIS_PERSISTENCE_RECEIPT_SCHEMA_VERSION,
    ArchitectureAxisTask,
    ArchitectureMode,
    _assign_performance_topology,
    _iter_semantic_candidate_trajectory,
    _row_evidence,
    _run_mode,
    _runtime_cgroup_snapshot,
    _runtime_io_accounting,
    _scheduler_runtime_summary,
    load_warm_start_bundle,
    workload_class_for_instance,
)
from evrptw.experiments.stage052_performance_calibration import (
    MEMORY_ADMISSION_EVIDENCE_SCHEMA_VERSION,
    OBSERVATION_PRODUCER_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    WORKLOAD_CLASSES,
    WheelReceipt,
    load_wheel_receipt,
)
from evrptw.experiments.stage052_telemetry_overhead import (
    TELEMETRY_SAMPLE_SCHEMA_VERSION,
)
from evrptw.native_scheduler import NativeHostScheduler
from evrptw.repository import repository_root
from evrptw.runtime_envelope import (
    DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS,
    ProcessTreeMonitor,
)
from evrptw.stage052_atomic import publish_no_replace
from evrptw.stage052_continuity_lease import require_owned
from evrptw.stage052_performance import (
    ExecutionTopology,
    HostPerformanceEnvelope,
    MemoryAdmission,
    RuntimeResourceSummaryV2,
    check_memory_admission,
    detect_host_performance,
    execution_topology_id,
    generate_mode_topology_candidates,
    require_clean_repository_root,
)
from evrptw.stage052_semantic_journal import evidence_json_value

FAILURE_SCHEMA_VERSION: Final = "stage05.2-performance-observation-failure-v1"
RESOURCE_EVIDENCE_SCHEMA_VERSION: Final = "stage05.2-calibration-resource-evidence-v3"
PREVIOUS_RESOURCE_EVIDENCE_SCHEMA_VERSION: Final = (
    "stage05.2-calibration-resource-evidence-v2"
)
LEGACY_RESOURCE_EVIDENCE_SCHEMA_VERSION: Final = (
    "stage05.2-calibration-resource-evidence-v1"
)
RESOURCE_SUMMARY_ACCOUNTING_SOURCE: Final = "process_tree_proc"
CALIBRATION_EXACT_CALLS: Final = 20
CALIBRATION_MAX_ITERATIONS: Final = 200
CALIBRATION_WATCHDOG_SECONDS: Final = 30.0
CALIBRATION_BATCH_SIZE: Final = 128
REPRESENTATIVE_TRAJECTORY_MAX_ROWS: Final = 200_000
REPRESENTATIVE_TRAJECTORY_MAX_BYTES: Final = 64 * 1024 * 1024
ISOLATED_MEMORY_PROBE_SAMPLE_INTERVAL_SECONDS: Final = (
    DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS
)
FIXED_WORK_BUDGET: Final = {
    "axis": "fixed_work",
    # Calibration is intentionally short.  The production 100-call protocol is
    # re-run only after the build/topology profile has been frozen, avoiding a
    # roughly five-fold multiplication of raw evidence for candidates that
    # cannot be selected.
    "exact_calls": CALIBRATION_EXACT_CALLS,
    "iterations": CALIBRATION_MAX_ITERATIONS,
    "watchdog_seconds": CALIBRATION_WATCHDOG_SECONDS,
    "batch_size": CALIBRATION_BATCH_SIZE,
}
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_ROW_EVIDENCE_SHA256: Final = hashlib.sha256(
    b"stage05.2-row-evidence-v1\0"
).hexdigest()


class PerformanceObservationError(RuntimeError):
    """The observation producer could not create trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class BlockExecution:
    """One isolated probe or admitted concurrent mode-block receipt."""

    elapsed_seconds: float
    independent_replay_seconds: float
    payloads: tuple[Mapping[str, object], ...]
    resource_statistics: Mapping[str, object]
    resource_summary: RuntimeResourceSummaryV2
    raw_inventory: tuple[Mapping[str, object], ...]
    scheduler_statistics: Mapping[str, object] | None
    scheduler_process_id: int | None
    scheduler_pss_bytes: int
    scheduler_task_receipt_inventory: tuple[Mapping[str, object], ...] = ()
    memory_admission: Mapping[str, object] | None = None

    @property
    def end_to_end_seconds(self) -> float:
        return self.elapsed_seconds + self.independent_replay_seconds


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PerformanceObservationError(f"cannot hash observation artifact: {path}") from error
    return digest.hexdigest()


def _scheduler_task_receipt_inventory(
    *,
    statistics: Mapping[str, object],
    receipt_path: Path,
    storage_root: Path,
    role: str,
) -> dict[str, object]:
    descriptor = statistics.get("task_receipts")
    if not isinstance(descriptor, Mapping):
        raise PerformanceObservationError("scheduler task-receipt descriptor is missing")
    sidecar = receipt_path.with_suffix(receipt_path.suffix + ".sha256")
    resolved_root = storage_root.resolve()
    resolved_path = receipt_path.resolve()
    resolved_sidecar = sidecar.resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root).as_posix()
        relative_sidecar = resolved_sidecar.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise PerformanceObservationError(
            "scheduler task-receipt evidence escapes its storage root"
        ) from error
    if (
        descriptor.get("path") != receipt_path.name
        or descriptor.get("sidecar_path") != sidecar.name
        or descriptor.get("sha256") != _sha256_file(receipt_path)
        or descriptor.get("sidecar_sha256") != _sha256_file(sidecar)
    ):
        raise PerformanceObservationError("scheduler task-receipt evidence does not reconcile")
    return {
        "storage_alias": "stage052-performance-calibration-run",
        "relative_path": relative_path,
        "sha256": descriptor["sha256"],
        "relative_sidecar_path": relative_sidecar,
        "sidecar_sha256": descriptor["sidecar_sha256"],
        "role": role,
    }


def _load_signed_host_envelope(path: Path) -> tuple[HostPerformanceEnvelope, str]:
    sidecar = Path(f"{path}.sha256")
    try:
        data = path.read_bytes()
        declared = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise PerformanceObservationError("cannot read signed host envelope") from error
    observed = hashlib.sha256(data).hexdigest()
    if declared != observed:
        raise PerformanceObservationError("host envelope SHA-256 mismatch")
    try:
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("host envelope root must be an object")
        envelope = HostPerformanceEnvelope.from_dict(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PerformanceObservationError("host envelope is invalid") from error
    return envelope, observed


def _require_parent_permit(parent_run_dir: Path, permit_path: Path) -> str:
    parent = parent_run_dir.resolve()
    if not parent.is_dir() or parent.name == "":
        raise PerformanceObservationError("calibration parent run directory is missing")
    sidecar = Path(f"{permit_path}.sha256")
    try:
        data = permit_path.read_bytes()
        declared = sidecar.read_text(encoding="ascii").strip().split()[0]
        payload = json.loads(data)
    except (OSError, UnicodeError, json.JSONDecodeError, IndexError) as error:
        raise PerformanceObservationError("calibration start permit is invalid") from error
    observed = hashlib.sha256(data).hexdigest()
    if (
        declared != observed
        or not isinstance(payload, dict)
        or payload.get("run_label") != parent.name
        or payload.get("status") != "reserved"
    ):
        raise PerformanceObservationError("calibration start permit identity mismatch")
    return observed


def _require_live_host_identity(
    frozen: HostPerformanceEnvelope,
) -> HostPerformanceEnvelope:
    observed = detect_host_performance()
    stable_fields = (
        "platform_name",
        "architecture",
        "allowed_cpu_ids",
        "physical_core_groups",
        "memory_total_bytes",
        "memory_limit_bytes",
        "swap_limit_bytes",
        "cpu_features",
    )
    if any(getattr(observed, field) != getattr(frozen, field) for field in stable_fields):
        raise PerformanceObservationError("live host identity differs from calibration envelope")
    # MemAvailable and cgroup current usage are live resource snapshots, not
    # stable host identity.  A child necessarily consumes some memory while it
    # starts, so requiring its effective headroom to be at least the parent's
    # frozen value makes every real calibration race its own startup.  The
    # signed envelope remains the fixed admission denominator; isolated PSS
    # calibration and the 20% headroom gate reject unsafe topologies later.
    if observed.swap_used_bytes != 0 or (
        observed.swap_current_bytes is not None and observed.swap_current_bytes != 0
    ):
        raise PerformanceObservationError("live calibration host already uses swap")
    return observed


def _current_process_pss_bytes(
    path: Path = Path("/proc/self/smaps_rollup"),
) -> int:
    """Read the current producer PSS without treating it as future demand."""

    try:
        rows = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise PerformanceObservationError("current process PSS is unavailable") from error
    for row in rows:
        fields = row.split()
        if len(fields) == 3 and fields[0] == "Pss:" and fields[2] == "kB":
            try:
                pss_bytes = int(fields[1]) * 1024
            except ValueError as error:
                raise PerformanceObservationError(
                    "current process PSS is invalid"
                ) from error
            if pss_bytes <= 0:
                raise PerformanceObservationError("current process PSS is not positive")
            return pss_bytes
    raise PerformanceObservationError("current process PSS is absent")


def _producer_peak_pss_bytes(statistics: Mapping[str, object]) -> int:
    root_pid = statistics.get("root_process_id")
    metrics = statistics.get("process_metrics")
    if (
        isinstance(root_pid, bool)
        or not isinstance(root_pid, int)
        or not isinstance(metrics, list)
    ):
        raise PerformanceObservationError("producer PSS evidence is unavailable")
    matching = [
        item
        for item in metrics
        if isinstance(item, Mapping) and item.get("pid") == root_pid
    ]
    if len(matching) != 1:
        raise PerformanceObservationError("producer PSS identity is ambiguous")
    value = matching[0].get("maximum_pss_bytes")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PerformanceObservationError("producer peak PSS is unavailable")
    return value


def _process_tree_resource_statistics(
    resource_evidence: Mapping[str, object],
) -> Mapping[str, object]:
    """Return the monitored tree from one signed block-resource envelope."""

    if resource_evidence.get("schema_version") not in {
        LEGACY_RESOURCE_EVIDENCE_SCHEMA_VERSION,
        PREVIOUS_RESOURCE_EVIDENCE_SCHEMA_VERSION,
        RESOURCE_EVIDENCE_SCHEMA_VERSION,
    }:
        raise PerformanceObservationError("resource evidence schema is invalid")
    process_tree = resource_evidence.get("process_tree")
    if not isinstance(process_tree, Mapping):
        raise PerformanceObservationError("resource process-tree evidence is unavailable")
    return cast(Mapping[str, object], process_tree)


def _worker_descendant_peak_pss_bytes(
    statistics: Mapping[str, object],
    *,
    scheduler_process_id: int | None,
) -> int:
    """Validate and return the isolated sample's simultaneous descendant PSS peak."""

    root_pid = statistics.get("root_process_id")
    additional_root_pids = statistics.get("additional_root_pids")
    metrics = statistics.get("process_metrics")
    receipt = statistics.get("worker_descendant_pss_peak")
    if (
        isinstance(root_pid, bool)
        or not isinstance(root_pid, int)
        or not isinstance(additional_root_pids, list)
        or not isinstance(metrics, list)
        or not isinstance(receipt, Mapping)
    ):
        raise PerformanceObservationError("worker descendant PSS evidence is unavailable")
    if any(
        isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
        for pid in additional_root_pids
    ):
        raise PerformanceObservationError("additional process-root identity is invalid")
    excluded_pids = {root_pid, *cast(list[int], additional_root_pids)}
    if scheduler_process_id is not None:
        if isinstance(scheduler_process_id, bool) or scheduler_process_id <= 0:
            raise PerformanceObservationError("scheduler process identity is invalid")
        if scheduler_process_id not in excluded_pids:
            raise PerformanceObservationError("scheduler is not a monitored process root")
    metric_pss: dict[tuple[int, object], int] = {}
    for item in metrics:
        if not isinstance(item, Mapping):
            raise PerformanceObservationError("worker process PSS row is invalid")
        pid = item.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise PerformanceObservationError("worker process identity is invalid")
        create_time = item.get("create_time")
        if (
            isinstance(create_time, bool)
            or not isinstance(create_time, int | float)
            or not math.isfinite(float(create_time))
        ):
            raise PerformanceObservationError("worker process create time is invalid")
        identity = (pid, create_time)
        if identity in metric_pss:
            raise PerformanceObservationError("worker process PSS identity is duplicated")
        value = item.get("maximum_pss_bytes")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PerformanceObservationError("process peak PSS is unavailable")
        metric_pss[identity] = value
    if receipt.get("status") != "available":
        raise PerformanceObservationError("worker descendant PSS peak is unavailable")
    peak = receipt.get("peak_bytes")
    sample_index = receipt.get("sample_index")
    process_rows = receipt.get("processes")
    sample_count = statistics.get("sample_count")
    complete_sample_count = statistics.get(
        "worker_descendant_pss_complete_sample_count"
    )
    incomplete_sample_count = statistics.get(
        "worker_descendant_pss_incomplete_sample_count"
    )
    if (
        isinstance(peak, bool)
        or not isinstance(peak, int)
        or peak <= 0
        or isinstance(sample_index, bool)
        or not isinstance(sample_index, int)
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or isinstance(complete_sample_count, bool)
        or not isinstance(complete_sample_count, int)
        or complete_sample_count <= 0
        or isinstance(incomplete_sample_count, bool)
        or not isinstance(incomplete_sample_count, int)
        or incomplete_sample_count != 0
        or not 0 <= sample_index < sample_count
        or not isinstance(process_rows, list)
        or not process_rows
        or statistics.get("peak_worker_descendant_pss_bytes") != peak
    ):
        raise PerformanceObservationError("worker descendant PSS peak receipt is invalid")
    total = 0
    observed_identities: set[tuple[int, object]] = set()
    for item in process_rows:
        if not isinstance(item, Mapping) or set(item) != {
            "pid",
            "create_time",
            "pss_bytes",
        }:
            raise PerformanceObservationError("worker descendant PSS process row is invalid")
        pid = item.get("pid")
        create_time = item.get("create_time")
        pss_bytes = item.get("pss_bytes")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or pid in excluded_pids
            or isinstance(create_time, bool)
            or not isinstance(create_time, int | float)
            or not math.isfinite(float(create_time))
            or isinstance(pss_bytes, bool)
            or not isinstance(pss_bytes, int)
            or pss_bytes <= 0
        ):
            raise PerformanceObservationError("worker descendant PSS process value is invalid")
        identity = (pid, create_time)
        if identity in observed_identities:
            raise PerformanceObservationError("worker descendant PSS process is duplicated")
        maximum = metric_pss.get(identity)
        if maximum is None or pss_bytes > maximum:
            raise PerformanceObservationError("worker descendant PSS exceeds process maximum")
        observed_identities.add(identity)
        total += pss_bytes
    if total != peak:
        raise PerformanceObservationError("worker descendant PSS peak does not reconcile")
    return peak


def _atomic_signed_json(path: Path, payload: Mapping[str, object]) -> str:
    if path.exists() or Path(f"{path}.sha256").exists():
        raise FileExistsError(f"observation output already exists: {path}")
    try:
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
    except (TypeError, ValueError) as error:
        raise PerformanceObservationError("observation output is not finite JSON") from error
    digest = hashlib.sha256(data).hexdigest()
    sidecar = Path(f"{path}.sha256")
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


def _signed_axis(
    path: Path,
    *,
    role: str,
    storage_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    sidecar = Path(f"{path}.sha256")
    try:
        data = path.read_bytes()
        declared = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise PerformanceObservationError(f"cannot read signed raw axis: {path}") from error
    observed = hashlib.sha256(data).hexdigest()
    if declared != observed:
        raise PerformanceObservationError(f"raw axis SHA-256 mismatch: {path}")
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PerformanceObservationError(f"raw axis is not valid JSON: {path}") from error
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        detail = payload.get("error") if isinstance(payload, dict) else None
        raise PerformanceObservationError(f"raw axis failed: {path}: {detail}")
    descriptor = payload.get("persistence_receipt")
    receipt_path = path.with_suffix(path.suffix + ".persistence")
    receipt_sidecar = receipt_path.with_suffix(receipt_path.suffix + ".sha256")
    if (
        not isinstance(descriptor, Mapping)
        or descriptor
        != {
            "schema_version": AXIS_PERSISTENCE_RECEIPT_SCHEMA_VERSION,
            "path": receipt_path.name,
            "sidecar_path": receipt_sidecar.name,
        }
        or receipt_path.is_symlink()
        or receipt_sidecar.is_symlink()
    ):
        raise PerformanceObservationError("raw axis persistence descriptor is invalid")
    try:
        receipt_data = receipt_path.read_bytes()
        receipt_declared = receipt_sidecar.read_text(encoding="ascii").strip()
        receipt = json.loads(receipt_data)
    except (OSError, UnicodeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PerformanceObservationError("raw axis persistence receipt is unreadable") from error
    receipt_sha256 = hashlib.sha256(receipt_data).hexdigest()
    if (
        receipt_declared != receipt_sha256
        or not isinstance(receipt, dict)
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
        or receipt.get("axis_sha256") != observed
        or receipt.get("axis_status") != "completed"
        or not isinstance(receipt.get("persistence_breakdown"), Mapping)
    ):
        raise PerformanceObservationError("raw axis persistence receipt is invalid")
    for field in ("persistence_seconds", "end_to_end_seconds"):
        _finite_seconds(receipt.get(field), f"axis receipt {field}")
    for field in (
        "primary_artifact_bytes",
        "persistence_receipt_bytes",
        "artifact_bytes",
    ):
        _nonnegative_int(receipt.get(field), f"axis receipt {field}")
    if receipt["artifact_bytes"] != (
        receipt["primary_artifact_bytes"] + receipt["persistence_receipt_bytes"]
    ):
        raise PerformanceObservationError("raw axis persistence bytes do not reconcile")
    payload = dict(payload)
    # The producer value stops before its own terminal receipt is published.
    # Keep it under the explicit pre-receipt name until independent replay adds
    # the complete axis end-to-end value.
    payload.pop("end_to_end_seconds", None)
    payload["persistence_seconds"] = receipt["persistence_seconds"]
    payload["producer_pre_receipt_seconds"] = receipt["end_to_end_seconds"]
    payload["artifact_bytes"] = receipt["artifact_bytes"]
    payload["persistence_breakdown"] = receipt["persistence_breakdown"]
    resolved_root = storage_root.resolve()
    try:
        relative_path = path.resolve().relative_to(resolved_root).as_posix()
        relative_sidecar = sidecar.resolve().relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise PerformanceObservationError("raw axis escapes its calibration run") from error
    supporting_artifacts: list[dict[str, object]] = []
    supporting_paths: set[Path] = set()

    def add_support(candidate: Path, artifact_role: str) -> None:
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise PerformanceObservationError(
                "raw axis supporting artifact escapes its calibration run"
            ) from error
        if candidate.is_symlink() or not resolved.is_file() or resolved in supporting_paths:
            raise PerformanceObservationError(
                "raw axis supporting artifact is missing, linked, or duplicated"
            )
        supporting_paths.add(resolved)
        supporting_artifacts.append(
            {
                "relative_path": relative,
                "sha256": _sha256_file(resolved),
                "role": artifact_role,
            }
        )

    add_support(receipt_path, "axis-persistence-receipt")
    add_support(receipt_sidecar, "axis-persistence-receipt-sidecar")
    semantic = payload.get("canonical_semantic_journal")
    if isinstance(semantic, Mapping) and isinstance(semantic.get("path"), str):
        bundle = path.parent / cast(str, semantic["path"])
        if bundle.is_symlink() or not bundle.is_dir():
            raise PerformanceObservationError("semantic journal bundle is unavailable")
        bundle_files = sorted(candidate for candidate in bundle.rglob("*") if candidate.is_file())
        if not bundle_files:
            raise PerformanceObservationError("semantic journal bundle is empty")
        for candidate in bundle_files:
            add_support(candidate, "semantic-journal-bundle-file")
    mode = payload.get("mode")
    work_pool: object = None
    if mode == ArchitectureMode.PER_SOLVE_RUNTIME.value:
        candidate_statistics = payload.get("candidate_transaction_statistics")
        if isinstance(candidate_statistics, Mapping):
            work_pool = candidate_statistics.get("native_candidate_work_pool")
    elif mode == ArchitectureMode.FULL_NATIVE_ALNS.value:
        native_statistics = payload.get("native_execution_statistics")
        if isinstance(native_statistics, Mapping):
            work_pool = native_statistics.get("work_pool_statistics")
    if isinstance(work_pool, Mapping):
        task_descriptor = work_pool.get("task_receipts")
        if (
            isinstance(task_descriptor, Mapping)
            and task_descriptor.get("schema_version") == "stage05.2-native-work-task-receipts-v3"
        ):
            task_name = task_descriptor.get("path")
            sidecar_name = task_descriptor.get("sidecar_path")
            if not isinstance(task_name, str) or not isinstance(sidecar_name, str):
                raise PerformanceObservationError("local native work-task descriptor is incomplete")
            add_support(path.parent / task_name, "local-native-work-task-stream")
            add_support(
                path.parent / sidecar_name,
                "local-native-work-task-stream-sidecar",
            )
    supporting_artifacts.sort(key=lambda item: cast(str, item["relative_path"]))
    return cast(dict[str, object], payload), {
        "storage_alias": "stage052-performance-calibration-run",
        "relative_path": relative_path,
        "sha256": observed,
        "relative_sidecar_path": relative_sidecar,
        "sidecar_sha256": _sha256_file(sidecar),
        "role": role,
        "supporting_artifacts": supporting_artifacts,
    }


def _finite_seconds(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise PerformanceObservationError(f"{field} must be finite and non-negative")
    return float(value)


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PerformanceObservationError(f"{field} must be a non-negative integer")
    return value


def _required_resource_int(statistics_payload: Mapping[str, object], field: str) -> int:
    value = statistics_payload.get(field)
    return _nonnegative_int(value, f"resource {field}")


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def _scheduler_queue_receipt(
    scheduler_statistics: Mapping[str, object] | None,
) -> tuple[float, int, int, int, int]:
    if scheduler_statistics is None:
        return (0.0, 0, 0, 0, 0)
    total_wait = 0.0
    peak_depth = 0
    pending_peak = 0
    queue_full = 0
    rejected = 0
    for name in ("request_queue", "work_queue"):
        raw = scheduler_statistics.get(name)
        if not isinstance(raw, Mapping):
            raise PerformanceObservationError("scheduler queue statistics are incomplete")
        total_wait += _finite_seconds(raw.get("total_wait_seconds"), f"{name} wait")
        peak = _nonnegative_int(raw.get("peak_pending"), f"{name} peak_pending")
        peak_depth = max(peak_depth, peak)
        pending_peak = max(pending_peak, peak)
        queue_full += _nonnegative_int(raw.get("queue_full_count"), f"{name} queue_full")
        rejected += _nonnegative_int(raw.get("rejected_count"), f"{name} rejected")
        if raw.get("pending") != 0 or (name == "work_queue" and raw.get("active") != 0):
            raise PerformanceObservationError("scheduler queue did not drain")
    if queue_full or rejected:
        raise PerformanceObservationError("scheduler queue overflow/rejection gate failed")
    return total_wait, peak_depth, pending_peak, queue_full, rejected


def _local_work_pool_receipt(
    payloads: Sequence[Mapping[str, object]],
) -> tuple[float, int, int, int, int]:
    total_wait = 0.0
    peak_depth = 0
    queue_full = 0
    rejected = 0
    expected_modes = {"per_solve_runtime", "full_native_alns", "host_scheduler"}
    observed_receipts = 0
    for payload in payloads:
        mode = payload.get("mode")
        raw: object = None
        candidate_statistics = payload.get("candidate_transaction_statistics")
        if isinstance(candidate_statistics, Mapping):
            candidate_pool = candidate_statistics.get("native_candidate_work_pool")
            if isinstance(candidate_pool, Mapping) and candidate_pool.get("enabled") is True:
                raw = candidate_pool
        native_statistics = payload.get("native_execution_statistics")
        if isinstance(native_statistics, Mapping) and isinstance(
            native_statistics.get("work_pool_statistics"), Mapping
        ):
            raw = native_statistics["work_pool_statistics"]
        if raw is None:
            if mode in expected_modes:
                raise PerformanceObservationError(
                    f"{mode} raw axis lacks bounded work-pool telemetry"
                )
            continue
        pool = cast(Mapping[str, object], raw)
        observed_receipts += 1
        wait = _finite_seconds(pool.get("total_wait_seconds"), "work-pool total wait")
        pending = _nonnegative_int(pool.get("peak_pending_tasks"), "peak_pending_tasks")
        full = _nonnegative_int(pool.get("queue_full_count"), "queue_full_count")
        pool_rejected = _nonnegative_int(pool.get("rejected_count"), "rejected_count")
        dropped = _nonnegative_int(
            pool.get("task_receipt_dropped_count"),
            "task_receipt_dropped_count",
        )
        if pool.get("pending_tasks") != 0 or pool.get("active_tasks") != 0:
            raise PerformanceObservationError("local work pool did not drain")
        if full or pool_rejected or dropped:
            raise PerformanceObservationError("local work-pool queue gate failed")
        total_wait += wait
        peak_depth = max(peak_depth, pending)
        queue_full += full
        rejected += pool_rejected
    first_mode = payloads[0].get("mode") if payloads else None
    if first_mode in expected_modes and observed_receipts != len(payloads):
        raise PerformanceObservationError("bounded work-pool telemetry matrix is incomplete")
    return total_wait, peak_depth, peak_depth, queue_full, rejected


def _resource_summary(
    *,
    elapsed_seconds: float,
    independent_replay_seconds: float,
    statistics_payload: Mapping[str, object],
    payloads: Sequence[Mapping[str, object]],
    topology: ExecutionTopology,
    cgroup_after: Mapping[str, object],
    io_accounting: Mapping[str, object],
    scheduler_statistics: Mapping[str, object] | None,
) -> RuntimeResourceSummaryV2:
    cpu_seconds = _finite_seconds(
        statistics_payload.get("process_tree_cpu_seconds"),
        "process_tree_cpu_seconds",
    )
    user_seconds = _finite_seconds(
        statistics_payload.get("process_tree_user_cpu_seconds"),
        "process_tree_user_cpu_seconds",
    )
    system_seconds = _finite_seconds(
        statistics_payload.get("process_tree_system_cpu_seconds"),
        "process_tree_system_cpu_seconds",
    )
    thread_tree = statistics_payload.get("thread_tree")
    if (
        statistics_payload.get("thread_tree_status") != "available"
        or not isinstance(thread_tree, Mapping)
        or thread_tree.get("status") not in {None, "available"}
    ):
        raise PerformanceObservationError("thread-tree diagnostic evidence is unavailable")
    schedstat = statistics_payload.get("process_tree_schedstat")
    if not isinstance(schedstat, Mapping):
        raise PerformanceObservationError("process-tree schedstat is unavailable")
    runqueue_ns = _nonnegative_int(schedstat.get("runqueue_delay_ns"), "runqueue_delay_ns")
    worker_times = [
        _finite_seconds(payload.get("end_to_end_seconds"), "axis end_to_end_seconds")
        for payload in payloads
    ]
    queue_wait, queue_depth, pending_peak, queue_full, rejected = _scheduler_queue_receipt(
        scheduler_statistics
    )
    local_wait, local_depth, local_pending, local_full, local_rejected = _local_work_pool_receipt(
        payloads
    )
    queue_wait += local_wait
    queue_depth = max(queue_depth, local_depth)
    pending_peak = max(pending_peak, local_pending)
    queue_full += local_full
    rejected += local_rejected
    cgroup_current = _required_resource_int(cgroup_after, "memory_current_bytes")
    cgroup_peak = _required_resource_int(cgroup_after, "memory_peak_bytes")
    return RuntimeResourceSummaryV2(
        elapsed_seconds=elapsed_seconds,
        effective_cores=cpu_seconds / max(elapsed_seconds, 1e-12),
        cpu_utilization_fraction=min(
            1.0,
            cpu_seconds / max(elapsed_seconds * len(topology.cpu_ids), 1e-12),
        ),
        user_cpu_seconds=user_seconds,
        system_cpu_seconds=system_seconds,
        run_queue_wait_seconds=runqueue_ns / 1_000_000_000.0,
        context_switches=_required_resource_int(
            statistics_payload,
            "process_tree_context_switches",
        ),
        cpu_migrations=_required_resource_int(
            statistics_payload,
            "process_tree_cpu_migrations",
        ),
        minor_faults=_required_resource_int(
            statistics_payload,
            "process_tree_minor_faults",
        ),
        major_faults=_required_resource_int(
            statistics_payload,
            "process_tree_major_faults",
        ),
        rss_bytes=_required_resource_int(statistics_payload, "peak_aggregate_rss_bytes"),
        pss_bytes=_required_resource_int(statistics_payload, "peak_aggregate_pss_bytes"),
        cgroup_memory_current_bytes=cgroup_current,
        cgroup_memory_peak_bytes=cgroup_peak,
        io_read_bytes=_nonnegative_int(io_accounting.get("read_bytes"), "I/O read_bytes"),
        io_write_bytes=_nonnegative_int(io_accounting.get("write_bytes"), "I/O write_bytes"),
        queue_wait_seconds=queue_wait,
        queue_depth_peak=queue_depth,
        pending_tasks_peak=pending_peak,
        queue_full_count=queue_full,
        rejected_count=rejected,
        worker_p95_seconds=_percentile(worker_times, 0.95),
        worker_max_seconds=max(worker_times),
        worker_min_seconds=min(worker_times),
        startup_seconds=max(
            _finite_seconds(payload.get("startup_seconds"), "axis startup_seconds")
            for payload in payloads
        ),
        solver_seconds=max(
            _finite_seconds(payload.get("solver_seconds"), "axis solver_seconds")
            for payload in payloads
        ),
        persistence_seconds=max(
            _finite_seconds(payload.get("persistence_seconds"), "axis persistence_seconds")
            for payload in payloads
        ),
        replay_seconds=independent_replay_seconds,
        p50_end_to_end_seconds=statistics.median(worker_times),
        p95_end_to_end_seconds=_percentile(worker_times, 0.95),
        p99_end_to_end_seconds=_percentile(worker_times, 0.99),
        max_end_to_end_seconds=max(worker_times),
    )


def _io_accounting_evidence(
    cgroup_before: Mapping[str, object],
    cgroup_after: Mapping[str, object],
    process_tree: Mapping[str, object],
) -> dict[str, object]:
    """Select one explicit, independently replayable I/O accounting provider."""

    try:
        return _runtime_io_accounting(cgroup_before, cgroup_after, process_tree)
    except RuntimeError as error:
        raise PerformanceObservationError(
            "calibration process-tree I/O accounting is unavailable"
        ) from error


def _scheduler_pss(
    resource_statistics: Mapping[str, object],
    scheduler_process_id: int | None,
) -> int:
    if scheduler_process_id is None:
        return 0
    metrics = resource_statistics.get("process_metrics")
    if not isinstance(metrics, list):
        raise PerformanceObservationError("scheduler process PSS evidence is unavailable")
    values = [
        entry.get("maximum_pss_bytes")
        for entry in metrics
        if isinstance(entry, Mapping) and entry.get("pid") == scheduler_process_id
    ]
    if len(values) != 1:
        raise PerformanceObservationError("scheduler process PSS identity is ambiguous")
    return _nonnegative_int(values[0], "scheduler maximum_pss_bytes")


def _task(
    *,
    receipt: WheelReceipt,
    repeat: int,
    instance_name: str,
    seed: int,
    benchmark_dir: Path,
    output_root: Path,
    warm_start: tuple[tuple[tuple[str, ...], ...], dict[str, object]],
    mode: ArchitectureMode,
    topology: ExecutionTopology,
    topology_id: str,
    role: str,
    shard_index: int,
    scheduler_socket_path: str,
) -> ArchitectureAxisTask:
    label = (
        "stage05.2_performance_calibration_raw_"
        f"{receipt.build_profile}_{repeat}_{mode.value}_{instance_name}_"
        f"{topology_id}_{role}_shard{shard_index:03d}"
    )
    base = ArchitectureAxisTask(
        scope="performance_calibration",
        repeat=repeat,
        axis="fixed_work",
        instance_name=instance_name,
        seed=seed,
        benchmark_dir=benchmark_dir,
        output_root=output_root,
        run_labels={item.value: label for item in ArchitectureMode},
        scheduler_socket_path=scheduler_socket_path,
        wheel_sha256=receipt.artifact_identity.wheel_sha256,
        native_sha256=receipt.artifact_identity.native_sha256,
        scheduler_sha256=receipt.artifact_identity.scheduler_sha256,
        revision=receipt.artifact_identity.git_revision,
        initial_customer_sequences=warm_start[0],
        initial_solution_provenance=dict(warm_start[1]),
        fixed_work_exact_calls=CALIBRATION_EXACT_CALLS,
        fixed_work_max_iterations=CALIBRATION_MAX_ITERATIONS,
        fixed_work_watchdog_seconds=CALIBRATION_WATCHDOG_SECONDS,
        exact_batch_size=CALIBRATION_BATCH_SIZE,
    )
    calibration_identity = hashlib.sha256(
        json.dumps(
            {
                "build_profile": receipt.build_profile,
                "repeat": repeat,
                "mode": mode.value,
                "topology": topology.to_dict(),
                "fixed_work_budget": dict(FIXED_WORK_BUDGET),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return _assign_performance_topology(
        base,
        topology,
        shard_index=shard_index,
        profile_sha256=calibration_identity,
        topology_key=f"calibration:{mode.value}:{topology.workload_class}:{topology_id}",
    )


def execute_mode_block(
    *,
    receipt: WheelReceipt,
    repeat: int,
    mode: ArchitectureMode,
    instance_name: str,
    seed: int,
    benchmark_dir: Path,
    output_root: Path,
    warm_start: tuple[tuple[tuple[str, ...], ...], dict[str, object]],
    topology: ExecutionTopology,
    topology_id: str,
    role: str,
    shard_count: int,
    archive_root: Path | None = None,
    wave_count: int = 1,
    frozen_host: HostPerformanceEnvelope | None = None,
    isolated_pss_bytes: int | None = None,
    scheduler_pss_bytes: int | None = None,
    producer_pss_bytes: int | None = None,
    worker_descendant_pss_bytes: int | None = None,
) -> BlockExecution:
    """Execute one real solver block under a candidate topology."""

    if shard_count not in {1, topology.shard_count}:
        raise ValueError("calibration block must be isolated or full-topology")
    if isinstance(wave_count, bool) or wave_count not in {1, 2}:
        raise ValueError("calibration block wave_count must be 1 or 2")
    guard_values = (
        frozen_host,
        isolated_pss_bytes,
        scheduler_pss_bytes,
        producer_pss_bytes,
        worker_descendant_pss_bytes,
    )
    if any(value is None for value in guard_values) != all(
        value is None for value in guard_values
    ):
        raise ValueError("calibration live-memory guard is incomplete")
    memory_admission_evidence: Mapping[str, object] | None = None
    if frozen_host is not None:
        live_host = _require_live_host_identity(frozen_host)
        admission, _projected, memory_admission_evidence = _memory_admission_evidence(
            frozen_host=frozen_host,
            live_host=live_host,
            topology=topology,
            isolated_pss_bytes=cast(int, isolated_pss_bytes),
            scheduler_pss_bytes=cast(int, scheduler_pss_bytes),
            producer_pss_bytes=cast(int, producer_pss_bytes),
            worker_descendant_pss_bytes=cast(int, worker_descendant_pss_bytes),
            resident_pss_bytes=_current_process_pss_bytes(),
        )
        if not admission.passed:
            raise PerformanceObservationError(
                f"live memory admission failed before {role}: {admission.reason}"
            )
    socket_hash = hashlib.sha256(
        f"{os.getpid()}:{receipt.build_profile}:{repeat}:{mode.value}:{topology_id}:{role}".encode()
    ).hexdigest()[:16]
    scheduler_socket = Path("/tmp") / f"evrptw-s52-cal-{socket_hash}.sock"
    if scheduler_socket.exists():
        raise FileExistsError(f"calibration scheduler socket exists: {scheduler_socket}")
    cgroup_before = _runtime_cgroup_snapshot()
    if cgroup_before.get("status") != "available":
        raise PerformanceObservationError("calibration cgroup accounting is unavailable")
    scheduler_statistics: Mapping[str, object] | None = None
    scheduler_pid: int | None = None
    scheduler_task_receipt_path: Path | None = None
    paths: list[Path] = []
    parent_terminal_seconds: dict[Path, float] = {}
    started = time.perf_counter()
    sample_interval_seconds = (
        ISOLATED_MEMORY_PROBE_SAMPLE_INTERVAL_SECONDS
        if role == "isolated-memory-probe"
        else DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS
    )
    with ProcessTreeMonitor(
        sample_interval_seconds=sample_interval_seconds
    ) as monitor:
        scheduler: NativeHostScheduler | None = None
        try:
            if mode is ArchitectureMode.HOST_SCHEDULER:
                scheduler_task_receipt_path = (
                    output_root / f"scheduler-task-receipts-{socket_hash}.jsonl"
                )
                scheduler = NativeHostScheduler(
                    scheduler_socket,
                    worker_threads=len(topology.scheduler_cpu_ids),
                    request_threads=topology.request_threads,
                    cpu_affinity=topology.scheduler_cpu_ids,
                    task_receipt_path=scheduler_task_receipt_path,
                )
                scheduler.start()
                scheduler_pid = scheduler.process_id
                monitor.register_additional_root(scheduler_pid)
            with ProcessPoolExecutor(
                max_workers=shard_count,
                mp_context=multiprocessing.get_context("spawn"),
                max_tasks_per_child=1,
            ) as executor:
                for wave_index in range(wave_count):
                    wave_role = role if wave_count == 1 else f"{role}-wave{wave_index + 1}"
                    tasks = tuple(
                        _task(
                            receipt=receipt,
                            repeat=repeat,
                            instance_name=instance_name,
                            seed=seed,
                            benchmark_dir=benchmark_dir,
                            output_root=output_root,
                            warm_start=warm_start,
                            mode=mode,
                            topology=topology,
                            topology_id=topology_id,
                            role=wave_role,
                            shard_index=index,
                            scheduler_socket_path=str(scheduler_socket),
                        )
                        for index in range(shard_count)
                    )
                    futures: dict[Future[str], float] = {}
                    for task in tasks:
                        submitted = time.perf_counter()
                        future = executor.submit(
                            _run_mode,
                            replace(task, scheduler_process_id=scheduler_pid),
                            mode,
                        )
                        futures[future] = submitted
                    for future in as_completed(futures):
                        path = Path(future.result())
                        paths.append(path)
                        parent_terminal_seconds[path] = time.perf_counter() - futures[future]
        finally:
            if scheduler is not None:
                scheduler.close()
                scheduler_statistics = _scheduler_runtime_summary(scheduler.runtime_statistics)
    elapsed = time.perf_counter() - started
    resource_statistics = monitor.statistics(
        elapsed_seconds=elapsed,
        compute_thread_limit=len(topology.cpu_ids),
    )
    cgroup_after = _runtime_cgroup_snapshot()
    if cgroup_after.get("status") != "available" or cgroup_after.get(
        "cgroup_path"
    ) != cgroup_before.get("cgroup_path"):
        raise PerformanceObservationError("calibration cgroup identity changed")
    for snapshot in (cgroup_before, cgroup_after):
        if snapshot.get("memory_swap_current_bytes") != 0:
            raise PerformanceObservationError("calibration used swap")
    raw_io = _io_accounting_evidence(cgroup_before, cgroup_after, resource_statistics)
    payloads: list[Mapping[str, object]] = []
    inventory: list[Mapping[str, object]] = []
    axis_timings: list[dict[str, object]] = []
    replay_started = time.perf_counter()
    from evrptw.experiments.stage052_performance_calibration_review import (
        _review_raw_axis,
    )

    for path in sorted(paths):
        payload, item = _signed_axis(
            path,
            role=role,
            storage_root=output_root if archive_root is None else archive_root,
        )
        axis_replay_started = time.perf_counter()
        projection, replay_mode, replay_workload, replay_topology_id = _review_raw_axis(
            path,
            benchmark_dir=benchmark_dir,
            build_identity=receipt.artifact_identity.to_dict(),
        )
        if (
            projection != _semantic_projection(payload)
            or replay_mode != mode.value
            or replay_workload != topology.workload_class
            or replay_topology_id != topology_id
        ):
            raise PerformanceObservationError("calibration independent replay projection diverged")
        axis_replay_seconds = time.perf_counter() - axis_replay_started
        terminal_seconds = parent_terminal_seconds[path]
        axis_end_to_end_seconds = terminal_seconds + axis_replay_seconds
        enriched_payload = dict(payload)
        enriched_payload["producer_parent_terminal_seconds"] = terminal_seconds
        enriched_payload["independent_replay_seconds"] = axis_replay_seconds
        enriched_payload["end_to_end_seconds"] = axis_end_to_end_seconds
        payloads.append(enriched_payload)
        inventory.append(item)
        axis_timings.append(
            {
                "relative_path": item["relative_path"],
                "producer_parent_terminal_seconds": terminal_seconds,
                "independent_replay_seconds": axis_replay_seconds,
                "end_to_end_seconds": axis_end_to_end_seconds,
            }
        )
    if len(payloads) != shard_count * wave_count:
        raise PerformanceObservationError("calibration mode block lost a shard result")
    scheduler_pss = _scheduler_pss(resource_statistics, scheduler_pid)
    scheduler_inventory: tuple[Mapping[str, object], ...] = ()
    if scheduler_statistics is not None:
        if scheduler_task_receipt_path is None:
            raise PerformanceObservationError("scheduler task-receipt target is missing")
        scheduler_inventory = (
            _scheduler_task_receipt_inventory(
                statistics=scheduler_statistics,
                receipt_path=scheduler_task_receipt_path,
                storage_root=output_root if archive_root is None else archive_root,
                role=role,
            ),
        )
    independent_replay_seconds = time.perf_counter() - replay_started
    summary = _resource_summary(
        elapsed_seconds=elapsed,
        independent_replay_seconds=independent_replay_seconds,
        statistics_payload=resource_statistics,
        payloads=payloads,
        topology=topology,
        cgroup_after=cgroup_after,
        io_accounting=raw_io,
        scheduler_statistics=scheduler_statistics,
    )
    raw_resource_evidence = {
        "schema_version": RESOURCE_EVIDENCE_SCHEMA_VERSION,
        "resource_summary_accounting_source": RESOURCE_SUMMARY_ACCOUNTING_SOURCE,
        "topology": topology.to_dict(),
        "elapsed_seconds": elapsed,
        "independent_replay_seconds": independent_replay_seconds,
        "process_tree": dict(resource_statistics),
        "cgroup_before": dict(cgroup_before),
        "cgroup_after": dict(cgroup_after),
        "io_accounting": dict(raw_io),
        "scheduler_process_id": scheduler_pid,
        "worker_process_lifecycle": "one_shard_per_spawned_process",
        "worker_multiprocessing_start_method": "spawn",
        "worker_max_tasks_per_child": 1,
        "axis_relative_paths": [item["relative_path"] for item in inventory],
        "axis_timings": axis_timings,
    }
    return BlockExecution(
        elapsed,
        independent_replay_seconds,
        tuple(payloads),
        raw_resource_evidence,
        summary,
        tuple(inventory),
        scheduler_statistics,
        scheduler_pid,
        scheduler_pss,
        scheduler_inventory,
        memory_admission_evidence,
    )


def _semantic_projection(payload: Mapping[str, object]) -> dict[str, object]:
    measurement = payload.get("measurement_evidence")
    journal = payload.get("canonical_semantic_journal")
    if not isinstance(measurement, Mapping) or not isinstance(journal, Mapping):
        raise PerformanceObservationError("raw axis semantic evidence is incomplete")
    transaction_hashes = (
        payload.get("candidate_work_hash"),
        payload.get("route_result_hash"),
        journal.get("sha256"),
    )
    hashes_are_complete = all(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None
        for value in transaction_hashes
    )
    candidate_events = payload.get("candidate_transaction_events")
    candidate_control = payload.get("candidate_control_statistics")
    candidate_statistics = payload.get("candidate_transaction_statistics")
    current_stage052_unavailable = (
        payload.get("mode") == ArchitectureMode.CURRENT_STAGE052.value
        and transaction_hashes[:2] == ("", "")
        and isinstance(transaction_hashes[2], str)
        and _SHA256_RE.fullmatch(transaction_hashes[2]) is not None
        and isinstance(candidate_events, Mapping)
        and set(candidate_events) == {"count", "sha256"}
        and candidate_events.get("count") == 0
        and candidate_events.get("sha256") == _EMPTY_ROW_EVIDENCE_SHA256
        and isinstance(candidate_control, Mapping)
        and not candidate_control
        and isinstance(candidate_statistics, Mapping)
        and candidate_statistics.get("native_candidate_transactions") == 0
    )
    if not hashes_are_complete and not current_stage052_unavailable:
        raise PerformanceObservationError("raw axis transaction hashes are invalid")
    return {
        "objective": payload.get("objective"),
        "routes": payload.get("routes"),
        "candidate_trajectory": payload.get("semantic_trajectory"),
        "exact_order": measurement.get("exact_route_order"),
        "cache_lifecycle": measurement.get("cache_lifecycle"),
        "transaction_hashes": list(transaction_hashes),
    }


def _replay_identical(payloads: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not payloads:
        raise PerformanceObservationError("calibration block has no raw payload")
    first = _semantic_projection(payloads[0])
    if any(_semantic_projection(payload) != first for payload in payloads[1:]):
        raise PerformanceObservationError("calibration shard semantic replay diverged")
    return first


def _scheduler_lifecycle_clean(
    statistics: Mapping[str, object] | None,
    *,
    expected_clients: int,
) -> bool:
    if statistics is None:
        return False
    peak_clients = statistics.get("peak_distinct_client_pids")
    if (
        isinstance(peak_clients, bool)
        or not isinstance(peak_clients, int)
        or peak_clients < expected_clients
    ):
        return False
    for queue_name in ("request_queue", "work_queue"):
        queue = statistics.get(queue_name)
        if not isinstance(queue, Mapping):
            return False
        for field in ("pending", "queue_full_count", "rejected_count"):
            if queue.get(field) != 0:
                return False
        if queue_name == "work_queue" and queue.get("active") != 0:
            return False
    return True


def compare_host_scheduler_lifecycles(
    *,
    receipt: WheelReceipt,
    repeat: int,
    instance_name: str,
    seed: int,
    benchmark_dir: Path,
    output_root: Path,
    warm_start: tuple[tuple[tuple[str, ...], ...], dict[str, object]],
    topology: ExecutionTopology,
    topology_id: str,
    archive_root: Path,
    frozen_host: HostPerformanceEnvelope | None = None,
    isolated_pss_bytes: int | None = None,
    scheduler_pss_bytes: int | None = None,
    producer_pss_bytes: int | None = None,
    worker_descendant_pss_bytes: int | None = None,
) -> tuple[dict[str, object], tuple[BlockExecution, ...]]:
    """Compare two scheduler-per-wave runs with one scheduler reused for two waves."""

    def per_wave() -> tuple[BlockExecution, BlockExecution]:
        return tuple(
            execute_mode_block(
                receipt=receipt,
                repeat=repeat,
                mode=ArchitectureMode.HOST_SCHEDULER,
                instance_name=instance_name,
                seed=seed,
                benchmark_dir=benchmark_dir,
                output_root=output_root,
                warm_start=warm_start,
                topology=topology,
                topology_id=topology_id,
                role=f"scheduler-per-wave-{wave_index + 1}",
                shard_count=topology.shard_count,
                archive_root=archive_root,
                frozen_host=frozen_host,
                isolated_pss_bytes=isolated_pss_bytes,
                scheduler_pss_bytes=scheduler_pss_bytes,
                producer_pss_bytes=producer_pss_bytes,
                worker_descendant_pss_bytes=worker_descendant_pss_bytes,
            )
            for wave_index in range(2)
        )  # type: ignore[return-value]

    def mode_block() -> BlockExecution:
        return execute_mode_block(
            receipt=receipt,
            repeat=repeat,
            mode=ArchitectureMode.HOST_SCHEDULER,
            instance_name=instance_name,
            seed=seed,
            benchmark_dir=benchmark_dir,
            output_root=output_root,
            warm_start=warm_start,
            topology=topology,
            topology_id=topology_id,
            role="scheduler-mode-block",
            shard_count=topology.shard_count,
            archive_root=archive_root,
            wave_count=2,
            frozen_host=frozen_host,
            isolated_pss_bytes=isolated_pss_bytes,
            scheduler_pss_bytes=scheduler_pss_bytes,
            producer_pss_bytes=producer_pss_bytes,
            worker_descendant_pss_bytes=worker_descendant_pss_bytes,
        )

    if repeat % 2 == 0:
        per_wave_blocks = per_wave()
        reused_block = mode_block()
    else:
        reused_block = mode_block()
        per_wave_blocks = per_wave()
    blocks = (*per_wave_blocks, reused_block)
    payloads = tuple(payload for block in blocks for payload in block.payloads)
    projections = tuple(_semantic_projection(payload) for payload in payloads)
    semantic_replay_passed = bool(projections) and all(
        projection == projections[0] for projection in projections[1:]
    )
    cache_lifecycles = tuple(projection["cache_lifecycle"] for projection in projections)
    cache_reset_passed = bool(cache_lifecycles) and all(
        lifecycle == cache_lifecycles[0] for lifecycle in cache_lifecycles[1:]
    )
    session_isolation_passed = all(
        _scheduler_lifecycle_clean(
            block.scheduler_statistics,
            expected_clients=topology.shard_count,
        )
        for block in blocks
    )
    per_wave_seconds = sum(block.end_to_end_seconds for block in per_wave_blocks)
    mode_block_seconds = reused_block.end_to_end_seconds
    per_wave_pss = max(block.scheduler_pss_bytes for block in per_wave_blocks)
    mode_block_pss = reused_block.scheduler_pss_bytes
    per_wave_process_tree_pss = max(block.resource_summary.pss_bytes for block in per_wave_blocks)
    mode_block_process_tree_pss = reused_block.resource_summary.pss_bytes
    rss_stability_passed = (
        per_wave_process_tree_pss > 0
        and mode_block_process_tree_pss > 0
        and mode_block_process_tree_pss <= math.ceil(per_wave_process_tree_pss * 1.20)
    )
    return (
        {
            "build_profile": receipt.build_profile,
            "topology_id": topology_id,
            "topology": topology.to_dict(),
            "session_isolation_passed": session_isolation_passed,
            "cache_reset_passed": cache_reset_passed,
            "rss_stability_passed": rss_stability_passed,
            "semantic_replay_passed": semantic_replay_passed,
            "mode_block_faster": mode_block_seconds < per_wave_seconds,
            "per_wave_end_to_end_seconds": per_wave_seconds,
            "mode_block_end_to_end_seconds": mode_block_seconds,
            "per_wave_scheduler_pss_peak_bytes": per_wave_pss,
            "mode_block_scheduler_pss_peak_bytes": mode_block_pss,
            "per_wave_process_tree_pss_peak_bytes": per_wave_process_tree_pss,
            "mode_block_process_tree_pss_peak_bytes": mode_block_process_tree_pss,
            "wave_count": 2,
        },
        blocks,
    )


def _memory_admission(
    frozen_host: HostPerformanceEnvelope,
    *,
    topology: ExecutionTopology,
    isolated_pss_bytes: int,
    scheduler_pss_bytes: int,
    producer_pss_bytes: int,
    worker_descendant_pss_bytes: int,
    live_host: HostPerformanceEnvelope | None = None,
    resident_pss_bytes: int = 0,
) -> tuple[MemoryAdmission, int]:
    if (
        isinstance(isolated_pss_bytes, bool)
        or not isinstance(isolated_pss_bytes, int)
        or isolated_pss_bytes <= 0
        or isinstance(scheduler_pss_bytes, bool)
        or not isinstance(scheduler_pss_bytes, int)
        or scheduler_pss_bytes < 0
        or isinstance(producer_pss_bytes, bool)
        or not isinstance(producer_pss_bytes, int)
        or producer_pss_bytes <= 0
        or isinstance(worker_descendant_pss_bytes, bool)
        or not isinstance(worker_descendant_pss_bytes, int)
        or worker_descendant_pss_bytes <= 0
    ):
        raise PerformanceObservationError("isolated PSS components are invalid")
    if (
        isinstance(resident_pss_bytes, bool)
        or not isinstance(resident_pss_bytes, int)
        or resident_pss_bytes < 0
    ):
        raise ValueError("resident_pss_bytes must be a non-negative integer")
    incremental_process_pss_bytes = (
        scheduler_pss_bytes + worker_descendant_pss_bytes * topology.shard_count
    )
    projected = producer_pss_bytes + incremental_process_pss_bytes
    if live_host is None:
        admission = check_memory_admission(
            frozen_host,
            projected,
            headroom_fraction=0.20,
        )
        effective_available = frozen_host.effective_memory_limit_bytes
        incremental_required = projected
    else:
        effective_available = min(
            frozen_host.effective_memory_limit_bytes,
            live_host.effective_memory_limit_bytes,
        )
        incremental_required = incremental_process_pss_bytes
        headroom = math.ceil(incremental_required * 0.20)
        swap_failure = live_host.swap_used_bytes > 0 or (
            live_host.swap_current_bytes is not None
            and live_host.swap_current_bytes > 0
        )
        passed = (
            not swap_failure
            and effective_available >= incremental_required + headroom
            and incremental_required <= math.floor(effective_available * 0.80)
        )
        reason = (
            "live swap is non-zero"
            if swap_failure
            else (
                "incremental projected PSS exceeds 80% of live effective memory"
                if incremental_required > math.floor(effective_available * 0.80)
                else (
                    "live effective memory is below incremental PSS plus headroom"
                    if effective_available < incremental_required + headroom
                    else "live memory and swap admission passed"
                )
            )
        )
        admission = MemoryAdmission(
            passed,
            effective_available,
            incremental_required,
            headroom,
            live_host.swap_total_bytes,
            live_host.swap_used_bytes,
            reason,
            0 if live_host.swap_current_bytes is None else live_host.swap_current_bytes,
        )
    if incremental_required > math.floor(effective_available * 0.80):
        admission = MemoryAdmission(
            False,
            admission.available_bytes,
            admission.required_bytes,
            admission.headroom_bytes,
            admission.swap_total_bytes,
            admission.swap_used_bytes,
            (
                "projected PSS exceeds 80% of effective memory"
                if live_host is None
                else "incremental projected PSS exceeds 80% of effective memory"
            ),
            admission.swap_current_bytes,
        )
    return admission, projected


def _memory_admission_evidence(
    *,
    frozen_host: HostPerformanceEnvelope,
    live_host: HostPerformanceEnvelope,
    topology: ExecutionTopology,
    isolated_pss_bytes: int,
    scheduler_pss_bytes: int,
    producer_pss_bytes: int,
    worker_descendant_pss_bytes: int,
    resident_pss_bytes: int,
) -> tuple[MemoryAdmission, int, dict[str, object]]:
    admission, projected = _memory_admission(
        frozen_host,
        topology=topology,
        isolated_pss_bytes=isolated_pss_bytes,
        scheduler_pss_bytes=scheduler_pss_bytes,
        producer_pss_bytes=producer_pss_bytes,
        worker_descendant_pss_bytes=worker_descendant_pss_bytes,
        live_host=live_host,
        resident_pss_bytes=resident_pss_bytes,
    )
    return (
        admission,
        projected,
        {
            "schema_version": MEMORY_ADMISSION_EVIDENCE_SCHEMA_VERSION,
            "frozen_effective_memory_limit_bytes": (
                frozen_host.effective_memory_limit_bytes
            ),
            "live_host": live_host.to_dict(),
            "resident_pss_bytes": resident_pss_bytes,
            "isolated_producer_pss_bytes": producer_pss_bytes,
            "isolated_worker_descendant_pss_bytes": worker_descendant_pss_bytes,
            "projected_concurrent_pss_bytes": projected,
            "incremental_required_bytes": admission.required_bytes,
            "admission": admission.to_dict(),
        },
    )


def _parallel_diagnostics(
    *,
    isolated: BlockExecution,
    concurrent: BlockExecution,
    topology: ExecutionTopology,
) -> dict[str, object]:
    """Derive auditable throughput and parallel-efficiency metrics."""

    shard_count = topology.shard_count
    sequential_projected_seconds = isolated.end_to_end_seconds * shard_count
    speedup = sequential_projected_seconds / concurrent.end_to_end_seconds
    parallel_efficiency = speedup / shard_count
    cpu_seconds = (
        concurrent.resource_summary.user_cpu_seconds
        + concurrent.resource_summary.system_cpu_seconds
    )
    worker_min = concurrent.resource_summary.worker_min_seconds
    return {
        "schema_version": "stage05.2-topology-parallel-diagnostics-v1",
        "shard_count": shard_count,
        "isolated_axis_end_to_end_seconds": isolated.end_to_end_seconds,
        "sequential_projected_seconds": sequential_projected_seconds,
        "mode_block_end_to_end_seconds": concurrent.end_to_end_seconds,
        "speedup": speedup,
        "parallel_efficiency": parallel_efficiency,
        "axes_per_hour": 3600.0 * shard_count / concurrent.end_to_end_seconds,
        "cpu_seconds_per_axis": cpu_seconds / shard_count,
        "axes_per_cpu_second": shard_count / max(cpu_seconds, 1e-12),
        "worker_p95_over_min": (
            concurrent.resource_summary.worker_p95_seconds / worker_min
            if worker_min > 0.0
            else "unavailable"
        ),
        "worker_max_over_min": (
            concurrent.resource_summary.worker_max_seconds / worker_min
            if worker_min > 0.0
            else "unavailable"
        ),
    }


def _axis_observation(
    *,
    block: BlockExecution,
    topology: ExecutionTopology,
    topology_id: str,
    mode: ArchitectureMode,
    workload_class: str,
    instance_name: str,
    seed: int,
    isolated_pss_bytes: int,
    scheduler_pss_bytes: int,
    producer_pss_bytes: int,
    worker_descendant_pss_bytes: int,
) -> dict[str, object]:
    semantic = _replay_identical(block.payloads)
    return {
        "mode": mode.value,
        "workload_class": workload_class,
        "instance": instance_name,
        "seed": seed,
        "topology_id": topology_id,
        "topology": topology.to_dict(),
        "producer_end_to_end_seconds": block.elapsed_seconds,
        "independent_replay_seconds": block.independent_replay_seconds,
        "end_to_end_seconds": block.end_to_end_seconds,
        "pss_bytes": isolated_pss_bytes,
        "scheduler_pss_bytes": scheduler_pss_bytes,
        "producer_pss_bytes": producer_pss_bytes,
        "worker_descendant_pss_bytes": worker_descendant_pss_bytes,
        "objective": semantic["objective"],
        "routes": semantic["routes"],
        "candidate_trajectory": semantic["candidate_trajectory"],
        "exact_order": semantic["exact_order"],
        "cache_lifecycle": semantic["cache_lifecycle"],
        "transaction_hashes": semantic["transaction_hashes"],
        "confidence_interval": None,
    }


def _validate_runtime_identity(receipt: WheelReceipt, repository: Path) -> None:
    identity = receipt.artifact_identity
    expected = {
        "git_revision": identity.git_revision,
        "git_tree": identity.git_tree,
        "source_manifest_sha256": identity.source_manifest_sha256,
        "performance_profile": receipt.build_profile,
        "compiler_id": receipt.compiler_id,
        "compiler_version": identity.compiler_version,
    }
    observed = {
        "git_revision": native_core.__build_git_revision__,
        "git_tree": native_core.__build_git_tree__,
        "source_manifest_sha256": native_core.__build_source_manifest_sha256__,
        "performance_profile": native_core.__build_performance_profile__,
        "compiler_id": native_core.__build_compiler_id__,
        "compiler_version": native_core.__build_compiler_version__,
    }
    if observed != expected:
        raise PerformanceObservationError("running native wheel differs from wheel receipt")
    if Path(native_core.__file__).resolve() != receipt.native_path.resolve():
        raise PerformanceObservationError(
            "running native extension path differs from wheel receipt"
        )
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if revision != identity.git_revision or dirty:
        raise PerformanceObservationError("performance observation requires its clean build commit")


def produce_fixed_work_observation(
    *,
    wheel_receipt_path: Path,
    warm_start_bundle_path: Path,
    benchmark_dir: Path,
    output_path: Path,
    raw_output_root: Path,
    repeat: int,
    seed: int,
    c5_instance: str,
    customer100_instance: str,
    continuity_lease_token: str,
    parent_run_dir: Path,
    start_permit_path: Path,
    host_envelope_path: Path,
    host: HostPerformanceEnvelope | None = None,
    repository_root_path: Path | None = None,
) -> dict[str, object]:
    """Run one build/repeat calibration envelope and publish signed evidence."""

    if repeat not in range(3):
        raise ValueError("performance observation repeat must be 0, 1, or 2")
    if output_path.exists() or raw_output_root.exists():
        raise FileExistsError("performance observation namespace already exists")
    parent = parent_run_dir.resolve()
    for candidate in (output_path.resolve(), raw_output_root.resolve()):
        try:
            candidate.relative_to(parent)
        except ValueError as error:
            raise PerformanceObservationError(
                "performance observation output escapes its governed calibration run"
            ) from error
    permit_sha256 = _require_parent_permit(parent, start_permit_path.resolve())
    root = repository_root() if repository_root_path is None else repository_root_path
    require_owned(
        root,
        token=continuity_lease_token,
        allowed_phases=frozenset({"validation", "performance-calibration"}),
    )
    receipt = load_wheel_receipt(wheel_receipt_path)
    _validate_runtime_identity(receipt, root)
    signed_host, host_envelope_sha256 = _load_signed_host_envelope(host_envelope_path.resolve())
    detected_host = signed_host if host is None else host
    if detected_host.to_dict() != signed_host.to_dict():
        raise PerformanceObservationError("supplied host differs from signed host envelope")
    _require_live_host_identity(detected_host)
    if detected_host.platform_name != "linux" or detected_host.architecture != "x86_64":
        raise PerformanceObservationError("performance observations support Linux x86-64 only")
    if detected_host.swap_used_bytes != 0 or (
        detected_host.swap_current_bytes is not None and detected_host.swap_current_bytes != 0
    ):
        raise PerformanceObservationError("performance observation host already uses swap")
    os.sched_setaffinity(0, detected_host.allowed_cpu_ids)
    os.environ.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    warm_starts = load_warm_start_bundle(
        warm_start_bundle_path,
        benchmark_dir=benchmark_dir,
    )
    instances = {"c5": c5_instance, "100-customer": customer100_instance}
    for workload, instance_name in instances.items():
        if workload_class_for_instance(instance_name) != workload:
            raise PerformanceObservationError("calibration instance workload class is invalid")
        if (instance_name, seed) not in warm_starts:
            raise PerformanceObservationError("calibration warm-start identity is missing")

    axes: list[dict[str, object]] = []
    memory_rejections: list[dict[str, object]] = []
    inventory: list[Mapping[str, object]] = []
    scheduler_inventory: list[Mapping[str, object]] = []
    resources: list[dict[str, object]] = []
    host_topology_candidates: dict[
        str,
        list[tuple[float, str, ExecutionTopology, int, int, int, int]],
    ] = {workload: [] for workload in WORKLOAD_CLASSES}
    raw_output_root.mkdir(parents=True)
    for mode in ArchitectureMode:
        for workload_class in WORKLOAD_CLASSES:
            instance_name = instances[workload_class]
            successful = 0
            for topology in generate_mode_topology_candidates(
                detected_host,
                mode=mode.value,
                workload_class=workload_class,
            ):
                topology_id = execution_topology_id(topology)
                probe = execute_mode_block(
                    receipt=receipt,
                    repeat=repeat,
                    mode=mode,
                    instance_name=instance_name,
                    seed=seed,
                    benchmark_dir=benchmark_dir,
                    output_root=raw_output_root,
                    warm_start=warm_starts[(instance_name, seed)],
                    topology=topology,
                    topology_id=topology_id,
                    role="isolated-memory-probe",
                    shard_count=1,
                    archive_root=parent,
                )
                inventory.extend(probe.raw_inventory)
                scheduler_inventory.extend(probe.scheduler_task_receipt_inventory)
                probe_process_tree = _process_tree_resource_statistics(
                    probe.resource_statistics
                )
                producer_pss = _producer_peak_pss_bytes(probe_process_tree)
                worker_descendant_pss = _worker_descendant_peak_pss_bytes(
                    probe_process_tree,
                    scheduler_process_id=probe.scheduler_process_id,
                )
                live_host = _require_live_host_identity(detected_host)
                admission, projected, admission_evidence = _memory_admission_evidence(
                    frozen_host=detected_host,
                    live_host=live_host,
                    topology=topology,
                    isolated_pss_bytes=probe.resource_summary.pss_bytes,
                    scheduler_pss_bytes=probe.scheduler_pss_bytes,
                    producer_pss_bytes=producer_pss,
                    worker_descendant_pss_bytes=worker_descendant_pss,
                    resident_pss_bytes=_current_process_pss_bytes(),
                )
                resources.append(
                    {
                        "mode": mode.value,
                        "workload_class": workload_class,
                        "topology_id": topology_id,
                        "role": "isolated-memory-probe",
                        "resource_summary": probe.resource_summary.to_dict(),
                        "raw_resource_statistics": dict(probe.resource_statistics),
                        "memory_admission": admission_evidence,
                    }
                )
                if not admission.passed:
                    memory_rejections.append(
                        {
                            "mode": mode.value,
                            "workload_class": workload_class,
                            "topology_id": topology_id,
                            "topology": topology.to_dict(),
                            "isolated_pss_bytes": probe.resource_summary.pss_bytes,
                            "scheduler_pss_bytes": probe.scheduler_pss_bytes,
                            "producer_pss_bytes": producer_pss,
                            "worker_descendant_pss_bytes": worker_descendant_pss,
                            "projected_concurrent_pss_bytes": projected,
                            "effective_memory_limit_bytes": admission.available_bytes,
                            "admission_available_bytes": admission.available_bytes,
                            "admission_required_bytes": admission.required_bytes,
                            "admission_headroom_bytes": admission.headroom_bytes,
                            "swap_used_bytes": admission.swap_used_bytes,
                            "swap_current_bytes": admission.swap_current_bytes,
                            "reason": admission.reason,
                            "memory_admission": admission_evidence,
                        }
                    )
                    continue
                block = execute_mode_block(
                    receipt=receipt,
                    repeat=repeat,
                    mode=mode,
                    instance_name=instance_name,
                    seed=seed,
                    benchmark_dir=benchmark_dir,
                    output_root=raw_output_root,
                    warm_start=warm_starts[(instance_name, seed)],
                    topology=topology,
                    topology_id=topology_id,
                    role="admitted-mode-block",
                    shard_count=topology.shard_count,
                    archive_root=parent,
                    frozen_host=detected_host,
                    isolated_pss_bytes=probe.resource_summary.pss_bytes,
                    scheduler_pss_bytes=probe.scheduler_pss_bytes,
                    producer_pss_bytes=producer_pss,
                    worker_descendant_pss_bytes=worker_descendant_pss,
                )
                if block.memory_admission is None:
                    raise PerformanceObservationError(
                        "admitted block lacks live memory evidence"
                    )
                inventory.extend(block.raw_inventory)
                scheduler_inventory.extend(block.scheduler_task_receipt_inventory)
                resources.append(
                    {
                        "mode": mode.value,
                        "workload_class": workload_class,
                        "topology_id": topology_id,
                        "role": "admitted-mode-block",
                        "resource_summary": block.resource_summary.to_dict(),
                        "parallel_diagnostics": _parallel_diagnostics(
                            isolated=probe,
                            concurrent=block,
                            topology=topology,
                        ),
                        "raw_resource_statistics": dict(block.resource_statistics),
                        "memory_admission": dict(block.memory_admission),
                        "scheduler_statistics": (
                            None
                            if block.scheduler_statistics is None
                            else dict(block.scheduler_statistics)
                        ),
                    }
                )
                axes.append(
                    _axis_observation(
                        block=block,
                        topology=topology,
                        topology_id=topology_id,
                        mode=mode,
                        workload_class=workload_class,
                        instance_name=instance_name,
                        seed=seed,
                        isolated_pss_bytes=probe.resource_summary.pss_bytes,
                        scheduler_pss_bytes=probe.scheduler_pss_bytes,
                        producer_pss_bytes=producer_pss,
                        worker_descendant_pss_bytes=worker_descendant_pss,
                    )
                )
                successful += 1
                if mode is ArchitectureMode.HOST_SCHEDULER:
                    host_topology_candidates[workload_class].append(
                        (
                            block.end_to_end_seconds,
                            topology_id,
                            topology,
                            probe.resource_summary.pss_bytes,
                            probe.scheduler_pss_bytes,
                            producer_pss,
                            worker_descendant_pss,
                        )
                    )
            if successful == 0:
                raise PerformanceObservationError(
                    f"no memory-safe topology remains for {mode.value}:{workload_class}"
                )

    lifecycle: dict[str, list[dict[str, object]]] = {}
    for workload_class in WORKLOAD_CLASSES:
        candidates = host_topology_candidates[workload_class]
        if not candidates:
            raise PerformanceObservationError("host scheduler lifecycle evidence is missing")
        instance_name = instances[workload_class]
        lifecycle[workload_class] = []
        for (
            _elapsed,
            topology_id,
            topology,
            isolated_pss,
            scheduler_pss,
            producer_pss,
            worker_descendant_pss,
        ) in sorted(
            candidates,
            key=lambda item: item[1],
        ):
            lifecycle_receipt, lifecycle_blocks = compare_host_scheduler_lifecycles(
                receipt=receipt,
                repeat=repeat,
                instance_name=instance_name,
                seed=seed,
                benchmark_dir=benchmark_dir,
                output_root=raw_output_root,
                warm_start=warm_starts[(instance_name, seed)],
                topology=topology,
                topology_id=topology_id,
                archive_root=parent,
                frozen_host=detected_host,
                isolated_pss_bytes=isolated_pss,
                scheduler_pss_bytes=scheduler_pss,
                producer_pss_bytes=producer_pss,
                worker_descendant_pss_bytes=worker_descendant_pss,
            )
            lifecycle[workload_class].append(lifecycle_receipt)
            for index, block in enumerate(lifecycle_blocks):
                if block.memory_admission is None:
                    raise PerformanceObservationError(
                        "scheduler lifecycle block lacks live memory evidence"
                    )
                inventory.extend(block.raw_inventory)
                scheduler_inventory.extend(block.scheduler_task_receipt_inventory)
                resources.append(
                    {
                        "mode": ArchitectureMode.HOST_SCHEDULER.value,
                        "workload_class": workload_class,
                        "topology_id": topology_id,
                        "role": (
                            "scheduler-mode-block"
                            if index == len(lifecycle_blocks) - 1
                            else f"scheduler-per-wave-{index + 1}"
                        ),
                        "resource_summary": block.resource_summary.to_dict(),
                        "raw_resource_statistics": dict(block.resource_statistics),
                        "memory_admission": dict(block.memory_admission),
                        "scheduler_statistics": (
                            None
                            if block.scheduler_statistics is None
                            else dict(block.scheduler_statistics)
                        ),
                    }
                )
    source_path = Path(__file__).resolve()
    warm_start_sha256 = _sha256_file(warm_start_bundle_path)
    warm_identity = {
        "storage_alias": "stage052-performance-calibration-inputs",
        "relative_path": f"warm-start-bundles/{warm_start_sha256}.json",
        "bundle_sha256": warm_start_sha256,
        "instances": instances,
        "seed": seed,
    }
    try:
        permit_relative = start_permit_path.resolve().relative_to(parent).as_posix()
        host_envelope_relative = host_envelope_path.resolve().relative_to(parent).as_posix()
    except ValueError as error:
        raise PerformanceObservationError(
            "governed permit/host envelope escapes the calibration run"
        ) from error
    payload: dict[str, object] = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "build_profile": receipt.build_profile,
        "repeat": repeat,
        "warm_start": warm_identity,
        "axes": axes,
        "host_scheduler_lifecycle_evidence": lifecycle,
        "build_identity": receipt.artifact_identity.to_dict(),
        "fixed_work_budget": dict(FIXED_WORK_BUDGET),
        "memory_rejections": memory_rejections,
        "producer_provenance": {
            "schema_version": OBSERVATION_PRODUCER_SCHEMA_VERSION,
            "repository_revision": receipt.artifact_identity.git_revision,
            "producer_source_sha256": _sha256_file(source_path),
            "parent_run_label": parent.name,
            "storage_alias": "stage052-performance-calibration-run",
            "start_permit_relative_path": permit_relative,
            "start_permit_sha256": permit_sha256,
            "host_envelope_relative_path": host_envelope_relative,
            "host_envelope_sha256": host_envelope_sha256,
            "raw_axis_inventory": list(inventory),
            "scheduler_task_receipt_inventory": list(scheduler_inventory),
            "resource_summaries": resources,
        },
    }
    _atomic_signed_json(output_path, payload)
    return payload


_REPRESENTATIVE_FINGERPRINT_FIELDS = (
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
    "semantic_trajectory",
    "trajectory",
    "stage04_events",
    "candidate_transaction_events",
)
_REPRESENTATIVE_TRAJECTORY_FIELDS = (
    "semantic_trajectory",
    "trajectory",
    "stage04_events",
    "candidate_transaction_events",
)


def _representative_fingerprint_payload_from_axis(
    payload: Mapping[str, object],
) -> dict[str, object]:
    fingerprint = {
        field: payload.get(field)
        for field in _REPRESENTATIVE_FINGERPRINT_FIELDS
        if field not in _REPRESENTATIVE_TRAJECTORY_FIELDS
    }
    for field in _REPRESENTATIVE_TRAJECTORY_FIELDS:
        raw = payload.get(field)
        expected_fields = (
            {"schema_version", "source", "count", "sha256"}
            if field == "semantic_trajectory"
            else {"count", "sha256"}
        )
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise PerformanceObservationError(f"representative {field} receipt is incomplete")
        fingerprint[field] = dict(raw)
    return fingerprint


def _representative_trajectory_rows(
    result: ALNSResult,
) -> dict[str, tuple[object, ...]]:
    sources: dict[str, Iterable[Mapping[str, object]]] = {
        "semantic_trajectory": _iter_semantic_candidate_trajectory(result),
        "trajectory": result.neighborhood_events,
        "stage04_events": result.stage04_event_log,
        "candidate_transaction_events": result.candidate_transaction_events,
    }
    retained: dict[str, tuple[object, ...]] = {}
    total_rows = 0
    total_bytes = 0
    for field, rows in sources.items():
        values: list[object] = []
        for row in rows:
            safe = evidence_json_value(dict(row))
            encoded = json.dumps(
                safe,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            total_rows += 1
            total_bytes += len(encoded)
            if (
                total_rows > REPRESENTATIVE_TRAJECTORY_MAX_ROWS
                or total_bytes > REPRESENTATIVE_TRAJECTORY_MAX_BYTES
            ):
                raise PerformanceObservationError(
                    "representative telemetry trajectory exceeds its bounded receipt"
                )
            values.append(safe)
        retained[field] = tuple(values)
    return retained


def _representative_fingerprint_payload_from_result(
    result: ALNSResult,
    trajectory_rows: Mapping[str, Sequence[object]],
) -> dict[str, object]:
    if result.objective is None:
        raise PerformanceObservationError("representative result has no objective")
    if set(trajectory_rows) != set(_REPRESENTATIVE_TRAJECTORY_FIELDS):
        raise PerformanceObservationError("representative trajectory surface is incomplete")
    fingerprint: dict[str, object] = {
        "objective": list(result.objective.key),
        "routes": [list(route) for route in result.routes],
        "candidate_work_hash": result.candidate_work_hash,
        "route_result_hash": result.route_result_hash,
        "effective_iterations": result.effective_iterations,
        "termination_reason": result.termination_reason,
        "accepted_moves": result.accepted_moves,
        "rejected_moves": result.rejected_moves,
        "exact_started_calls": result.exact_started_calls,
        "exact_completed_calls": result.exact_completed_calls,
        "exact_interrupted_calls": result.exact_interrupted_calls,
    }
    semantic_receipt = _row_evidence(trajectory_rows["semantic_trajectory"])
    fingerprint["semantic_trajectory"] = {
        "schema_version": "stage05.2-external-semantic-trajectory-v1",
        "source": "canonical_semantic_journal:operator",
        **semantic_receipt,
    }
    for field in _REPRESENTATIVE_TRAJECTORY_FIELDS[1:]:
        fingerprint[field] = _row_evidence(trajectory_rows[field])
    return fingerprint


def _representative_fingerprint(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _candidate_payload_receipt_matches(
    rows: Sequence[Mapping[str, object]],
    declared_hash: object,
) -> bool:
    """Validate an optional Candidate Control receipt without inventing telemetry.

    ``current_stage052`` does not construct a Candidate Control runtime.  Its
    result therefore uses the historical unavailable representation: no rows
    and an empty hash.  The telemetry-off control must preserve that absence;
    hashing the empty tuple would fabricate a receipt that the solver did not
    produce and would make the unmonitored control impossible to run.
    """

    if declared_hash == "":
        return not rows
    return (
        isinstance(declared_hash, str)
        and _SHA256_RE.fullmatch(declared_hash) is not None
        and stable_candidate_payload_hash(rows) == declared_hash
    )


def produce_representative_telemetry_sample(
    *,
    enabled: bool,
    sample_index: int,
    wheel_receipt_path: Path,
    warm_start_bundle_path: Path,
    benchmark_dir: Path,
    output_path: Path,
    raw_output_root: Path,
    seed: int,
    instance_name: str,
    continuity_lease_token: str,
    parent_run_dir: Path,
    start_permit_path: Path,
    host_envelope_path: Path,
    resource_sample_interval_seconds: float = (
        DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS
    ),
    repository_root_path: Path | None = None,
) -> dict[str, object]:
    """Run one representative current-stage fixed-work axis with telemetry on/off."""

    if (
        isinstance(sample_index, bool)
        or sample_index < -2
        or (sample_index == -2 and enabled)
        or (sample_index == -1 and not enabled)
    ):
        raise ValueError("representative telemetry sample index is invalid")
    if (
        isinstance(resource_sample_interval_seconds, bool)
        or not math.isfinite(resource_sample_interval_seconds)
        or resource_sample_interval_seconds <= 0.0
    ):
        raise ValueError("representative telemetry sample interval is invalid")
    if output_path.exists() or raw_output_root.exists():
        raise FileExistsError("representative telemetry sample namespace already exists")
    parent = parent_run_dir.resolve()
    for candidate in (output_path.resolve(), raw_output_root.resolve()):
        try:
            candidate.relative_to(parent)
        except ValueError as error:
            raise PerformanceObservationError(
                "representative telemetry output escapes its calibration run"
            ) from error
    _require_parent_permit(parent, start_permit_path.resolve())
    root = repository_root() if repository_root_path is None else repository_root_path
    require_owned(
        root,
        token=continuity_lease_token,
        allowed_phases=frozenset({"validation", "performance-calibration"}),
    )
    receipt = load_wheel_receipt(wheel_receipt_path)
    _validate_runtime_identity(receipt, root)
    host, _host_sha256 = _load_signed_host_envelope(host_envelope_path.resolve())
    _require_live_host_identity(host)
    if host.swap_used_bytes != 0 or (
        host.swap_current_bytes is not None and host.swap_current_bytes != 0
    ):
        raise PerformanceObservationError("representative telemetry host already uses swap")
    os.sched_setaffinity(0, host.allowed_cpu_ids)
    os.environ.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    warm_starts = load_warm_start_bundle(warm_start_bundle_path, benchmark_dir=benchmark_dir)
    if workload_class_for_instance(instance_name) != "c5":
        raise PerformanceObservationError("representative telemetry instance must be C5")
    warm_start = warm_starts.get((instance_name, seed))
    if warm_start is None:
        raise PerformanceObservationError("representative telemetry warm start is missing")
    candidates = generate_mode_topology_candidates(
        host,
        mode=ArchitectureMode.CURRENT_STAGE052.value,
        workload_class="c5",
    )
    target_shards = math.ceil(len(host.allowed_cpu_ids) / 4)
    topology = min(
        candidates,
        key=lambda item: (
            abs(item.shard_count - target_shards),
            item.affinity_policy != "physical_core_first",
            execution_topology_id(item),
        ),
    )
    topology_id = execution_topology_id(topology)
    task = _task(
        receipt=receipt,
        repeat=0,
        instance_name=instance_name,
        seed=seed,
        benchmark_dir=benchmark_dir,
        output_root=raw_output_root,
        warm_start=(warm_start[0], dict(warm_start[1])),
        mode=ArchitectureMode.CURRENT_STAGE052,
        topology=topology,
        topology_id=topology_id,
        role=f"telemetry-{'on' if enabled else 'off'}",
        shard_index=0,
        scheduler_socket_path="",
    )
    started = time.perf_counter()
    raw_inventory: list[Mapping[str, object]] = []
    replay_seconds = 0.0
    axis_path = Path(
        _run_mode(
            task,
            ArchitectureMode.CURRENT_STAGE052,
            resource_telemetry_enabled=enabled,
            resource_telemetry_sample_interval_seconds=(
                resource_sample_interval_seconds
            ),
        )
    )
    payload, inventory = _signed_axis(
        axis_path,
        role=(
            "representative-resource-telemetry-on"
            if enabled
            else "representative-resource-telemetry-off"
        ),
        storage_root=parent,
    )
    from evrptw.experiments.stage052_native_architecture_review import (
        ReviewRecord,
        _replay_record,
    )

    replay_started = time.perf_counter()
    replay = _replay_record(
        ReviewRecord(axis_path, payload),
        benchmark_dir,
        expected_resource_telemetry=enabled,
    )
    replay_seconds = time.perf_counter() - replay_started
    if replay.get("valid") is not True or replay.get("semantics_complete") is not True:
        raise PerformanceObservationError("representative telemetry replay failed")
    journal = payload.get("canonical_semantic_journal")
    physical = journal.get("physical_telemetry") if isinstance(journal, Mapping) else None
    measurement = payload.get("measurement_evidence")
    persistence = payload.get("persistence_breakdown")
    if not (
        isinstance(measurement, Mapping)
        and measurement.get("present") is True
        and isinstance(physical, Mapping)
        and isinstance(persistence, Mapping)
    ):
        raise PerformanceObservationError("representative telemetry surface is incomplete")
    resource_summary = payload.get("topology")
    raw_inventory.append(inventory)
    measured_elapsed_seconds = time.perf_counter() - started
    fingerprint_payload = _representative_fingerprint_payload_from_axis(payload)
    fingerprint = _representative_fingerprint(fingerprint_payload)
    evidence = {
        "kind": "representative-fixed-work-axis",
        "mode": ArchitectureMode.CURRENT_STAGE052.value,
        "axis": "fixed_work",
        "instance": instance_name,
        "seed": seed,
        "exact_calls": CALIBRATION_EXACT_CALLS,
        "iterations": CALIBRATION_MAX_ITERATIONS,
        "batch_size": CALIBRATION_BATCH_SIZE,
        "semantic_telemetry": True,
        "physical_telemetry": True,
        "persistence": True,
        "independent_replay": True,
        "resource_telemetry": enabled,
        "minimal_validator_replay": True,
        "fingerprints_identical": True,
    }
    if not isinstance(resource_summary, Mapping):
        raise PerformanceObservationError("representative telemetry resource summary is missing")
    sample = {
        "schema_version": TELEMETRY_SAMPLE_SCHEMA_VERSION,
        "enabled": enabled,
        "sample_index": sample_index,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "elapsed_seconds": measured_elapsed_seconds,
        "replay_seconds": replay_seconds,
        "resource_summary": dict(resource_summary),
        "workload_evidence": evidence,
        "raw_axis_inventory": raw_inventory,
    }
    _atomic_signed_json(output_path, sample)
    return sample


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-receipt", type=Path, required=True)
    parser.add_argument("--warm-start-bundle", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output-root", type=Path, required=True)
    parser.add_argument("--repeat", type=int, choices=range(3), required=True)
    parser.add_argument("--seed", type=int, default=2014)
    parser.add_argument("--c5-instance", default="c101C5")
    parser.add_argument("--customer100-instance", default="c101_21")
    parser.add_argument("--continuity-lease-token", required=True)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--start-permit", type=Path, required=True)
    parser.add_argument("--host-envelope", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        required=True,
        help="Explicit clean ext4 Git worktree used for source identity",
    )
    parser.add_argument("--telemetry-sample", choices=("on", "off"))
    parser.add_argument("--telemetry-sample-index", type=int)
    parser.add_argument(
        "--resource-sample-interval-seconds",
        type=float,
        default=DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS,
    )
    arguments = parser.parse_args(argv)
    resolved_repository = require_clean_repository_root(arguments.repository_root)
    try:
        if arguments.telemetry_sample is None:
            produce_fixed_work_observation(
                wheel_receipt_path=arguments.wheel_receipt,
                warm_start_bundle_path=arguments.warm_start_bundle,
                benchmark_dir=arguments.benchmark_dir,
                output_path=arguments.output,
                raw_output_root=arguments.raw_output_root,
                repeat=arguments.repeat,
                seed=arguments.seed,
                c5_instance=arguments.c5_instance,
                customer100_instance=arguments.customer100_instance,
                continuity_lease_token=arguments.continuity_lease_token,
                parent_run_dir=arguments.parent_run_dir,
                start_permit_path=arguments.start_permit,
                host_envelope_path=arguments.host_envelope,
                repository_root_path=resolved_repository,
            )
        else:
            if arguments.telemetry_sample_index is None:
                raise ValueError("--telemetry-sample-index is required for telemetry samples")
            produce_representative_telemetry_sample(
                enabled=arguments.telemetry_sample == "on",
                sample_index=arguments.telemetry_sample_index,
                wheel_receipt_path=arguments.wheel_receipt,
                warm_start_bundle_path=arguments.warm_start_bundle,
                benchmark_dir=arguments.benchmark_dir,
                output_path=arguments.output,
                raw_output_root=arguments.raw_output_root,
                seed=arguments.seed,
                instance_name=arguments.c5_instance,
                continuity_lease_token=arguments.continuity_lease_token,
                parent_run_dir=arguments.parent_run_dir,
                start_permit_path=arguments.start_permit,
                host_envelope_path=arguments.host_envelope,
                resource_sample_interval_seconds=(
                    arguments.resource_sample_interval_seconds
                ),
                repository_root_path=resolved_repository,
            )
    except BaseException as error:
        failure_path = arguments.output.with_name(arguments.output.name + ".failed.json")
        if not failure_path.exists() and not Path(f"{failure_path}.sha256").exists():
            _atomic_signed_json(
                failure_path,
                {
                    "schema_version": FAILURE_SCHEMA_VERSION,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "raw_output_root": str(arguments.raw_output_root.resolve()),
                    "formal_started": False,
                    "cuda_started": False,
                    "attempt08_started": False,
                },
            )
        raise
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "BlockExecution",
    "FAILURE_SCHEMA_VERSION",
    "FIXED_WORK_BUDGET",
    "PerformanceObservationError",
    "TELEMETRY_SAMPLE_SCHEMA_VERSION",
    "execute_mode_block",
    "main",
    "produce_fixed_work_observation",
    "produce_representative_telemetry_sample",
)
