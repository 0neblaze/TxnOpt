"""Public Stage 5.2 prerequisite and process-resource evidence seams."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]

from evrptw.artifacts import ArtifactIntegrityError, ArtifactReader

STAGE052_REVIEW_SCHEMA_VERSION = "stage05.2-review-v1"
STAGE052_RESOURCE_SCHEMA_VERSION = "stage05.2-run-resource-v2"
STAGE052_RESOURCE_MEASUREMENT_SCOPE = (
    "task_scheduling_through_parent_control_preparation"
)
_PERFORMANCE_ENVIRONMENT_VARIABLES = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_PLACES",
    "OMP_PROC_BIND",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "VECLIB_MAXIMUM_THREADS",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class Stage052PrerequisiteIdentity:
    run_label: str
    component: str
    status: str
    repository_revision: str
    configuration_sha256: str
    raw_manifest_sha256: str
    review_manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def verify_stage052_prerequisite(
    raw_dir: Path,
    *,
    expected_component: str,
    expected_status: str,
    expected_run_label: str | None = None,
) -> Stage052PrerequisiteIdentity:
    """Verify an accepted Stage 5.2 producer and independent review bundle."""

    raw_dir = raw_dir.resolve()
    if expected_run_label is not None and raw_dir.name != expected_run_label:
        raise ArtifactIntegrityError(
            f"prerequisite run label mismatch: expected={expected_run_label} "
            f"observed={raw_dir.name}"
        )
    review_manifest_path = raw_dir / "review" / "review_manifest.json"
    try:
        review = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(
            f"cannot read prerequisite review manifest: {review_manifest_path}"
        ) from error
    expected_review = {
        "schema_version": STAGE052_REVIEW_SCHEMA_VERSION,
        "run_label": raw_dir.name,
        "component": expected_component,
        "scope": "performance",
        "status": expected_status,
    }
    for field, expected in expected_review.items():
        if review.get(field) != expected:
            raise ArtifactIntegrityError(
                f"prerequisite review {field} mismatch: "
                f"expected={expected} observed={review.get(field)}"
            )
    gates = review.get("gates")
    if (
        not isinstance(gates, dict)
        or not gates
        or any(
            not isinstance(gate, dict) or gate.get("passed") is not True
            for gate in gates.values()
        )
    ):
        raise ArtifactIntegrityError("prerequisite review contains a failed or invalid gate")
    files = review.get("files")
    if not isinstance(files, dict) or set(files) != {
        "review_findings.csv",
        "review_report.md",
    }:
        raise ArtifactIntegrityError("prerequisite review file identity mismatch")
    for name, checksum in files.items():
        path = raw_dir / "review" / str(name)
        if not path.is_file() or _sha256(path) != str(checksum):
            raise ArtifactIntegrityError(f"prerequisite review checksum mismatch: {name}")

    reader = ArtifactReader(raw_dir)
    if reader.manifest.get("evidence_completeness") != "complete":
        raise ArtifactIntegrityError("prerequisite raw evidence is partial")
    metadata_items = [
        item
        for item in reader.manifest.get("artifacts", [])
        if isinstance(item, dict) and item.get("artifact_type") == "manifest_metadata"
    ]
    config_items = [
        item
        for item in reader.manifest.get("artifacts", [])
        if isinstance(item, dict) and item.get("artifact_type") == "config"
    ]
    if len(metadata_items) != 1 or len(config_items) != 1:
        raise ArtifactIntegrityError("prerequisite control artifacts are incomplete")
    metadata = reader.read_json(str(metadata_items[0]["relative_path"]))
    if (
        metadata.get("run_label") != raw_dir.name
        or metadata.get("component") != expected_component
        or metadata.get("scope") != "performance"
        or metadata.get("repository_dirty") is not False
    ):
        raise ArtifactIntegrityError("prerequisite producer identity mismatch")
    config_checksum = str(config_items[0].get("checksum", ""))
    if metadata.get("configuration_sha256") != config_checksum:
        raise ArtifactIntegrityError("prerequisite configuration checksum mismatch")
    revision = str(metadata.get("repository_revision", ""))
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ArtifactIntegrityError("prerequisite repository revision is invalid")
    manifest_path = raw_dir / "control" / f"{raw_dir.name}_manifest.json"
    return Stage052PrerequisiteIdentity(
        run_label=raw_dir.name,
        component=expected_component,
        status=expected_status,
        repository_revision=revision,
        configuration_sha256=config_checksum,
        raw_manifest_sha256=_sha256(manifest_path),
        review_manifest_sha256=_sha256(review_manifest_path),
    )


@dataclass(frozen=True, slots=True)
class RunResourceSummary:
    schema_version: str
    run_label: str
    component: str
    configured_worker_count: int
    measurement_scope: str
    run_wall_seconds: float
    sample_interval_seconds: float
    parent_pid: int
    descendant_pids: tuple[int, ...]
    aggregate_peak_rss_bytes: int
    mean_active_cores: float
    peak_active_cores: float
    sample_count: int
    status: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["descendant_pids"] = list(self.descendant_pids)
        return payload


class ProcessTreeResourceSampler:
    """Sample simultaneous RSS and CPU usage for one process tree."""

    def __init__(
        self,
        *,
        run_label: str,
        component: str,
        configured_worker_count: int,
        interval_seconds: float = 0.05,
        parent_pid: int | None = None,
    ) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("resource sample interval must be positive")
        if configured_worker_count not in {1, 2, 4}:
            raise ValueError("configured worker count must be 1, 2, or 4")
        self.run_label = run_label
        self.component = component
        self.configured_worker_count = configured_worker_count
        self.interval_seconds = interval_seconds
        self.parent_pid = parent_pid or os.getpid()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._peak_rss = 0
        self._active_core_samples: list[float] = []
        self._descendant_pids: set[int] = set()
        self._sample_count = 0
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("resource sampler is already started")
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="stage052-resource-sampler")
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> RunResourceSummary:
        if self._thread is None:
            raise RuntimeError("resource sampler was not started")
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 4.0))
        if self._thread.is_alive():
            raise RuntimeError("resource sampler did not stop")
        if self._error is not None:
            raise RuntimeError(f"resource sampling failed: {self._error}") from self._error
        wall = time.perf_counter() - self._started
        return RunResourceSummary(
            schema_version=STAGE052_RESOURCE_SCHEMA_VERSION,
            run_label=self.run_label,
            component=self.component,
            configured_worker_count=self.configured_worker_count,
            measurement_scope=STAGE052_RESOURCE_MEASUREMENT_SCOPE,
            run_wall_seconds=wall,
            sample_interval_seconds=self.interval_seconds,
            parent_pid=self.parent_pid,
            descendant_pids=tuple(sorted(self._descendant_pids)),
            aggregate_peak_rss_bytes=self._peak_rss,
            mean_active_cores=(
                sum(self._active_core_samples) / len(self._active_core_samples)
                if self._active_core_samples
                else 0.0
            ),
            peak_active_cores=max(self._active_core_samples, default=0.0),
            sample_count=self._sample_count,
            status="complete",
        )

    def _run(self) -> None:
        previous_wall: float | None = None
        previous_cpu: float | None = None
        try:
            while not self._stop.is_set():
                processes = self._processes()
                now = time.perf_counter()
                rss = 0
                cpu = 0.0
                for process in processes:
                    try:
                        with process.oneshot():
                            rss += int(process.memory_info().rss)
                            times = process.cpu_times()
                            cpu += float(times.user + times.system)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        continue
                self._peak_rss = max(self._peak_rss, rss)
                self._sample_count += 1
                if previous_wall is not None and previous_cpu is not None and now > previous_wall:
                    self._active_core_samples.append(
                        max(0.0, (cpu - previous_cpu) / (now - previous_wall))
                    )
                previous_wall = now
                previous_cpu = cpu
                self._stop.wait(self.interval_seconds)
        except BaseException as error:
            self._error = error

    def _processes(self) -> list[Any]:
        parent = psutil.Process(self.parent_pid)
        children = parent.children(recursive=True)
        self._descendant_pids.update(process.pid for process in children)
        return [parent, *children]


def validate_worker_ownership(
    resource_summary: dict[str, object],
    shard_manifests: list[dict[str, object]],
    *,
    expected_workers: int,
    expected_run_label: str,
    expected_component: str,
) -> tuple[bool, str, tuple[int, ...]]:
    """Validate actual shard-owner PIDs against the sampled process tree."""

    if resource_summary.get("schema_version") != STAGE052_RESOURCE_SCHEMA_VERSION:
        return False, "unsupported resource summary schema", ()
    resource_identity = (
        resource_summary.get("run_label"),
        resource_summary.get("component"),
        resource_summary.get("configured_worker_count"),
        resource_summary.get("measurement_scope"),
        resource_summary.get("status"),
    )
    expected_identity = (
        expected_run_label,
        expected_component,
        expected_workers,
        STAGE052_RESOURCE_MEASUREMENT_SCOPE,
        "complete",
    )
    if resource_identity != expected_identity:
        return (
            False,
            f"resource identity mismatch: expected={expected_identity} "
            f"observed={resource_identity}",
            (),
        )
    numeric_values = {
        "run_wall_seconds": resource_summary.get("run_wall_seconds"),
        "sample_interval_seconds": resource_summary.get("sample_interval_seconds"),
        "aggregate_peak_rss_bytes": resource_summary.get("aggregate_peak_rss_bytes"),
        "mean_active_cores": resource_summary.get("mean_active_cores"),
        "peak_active_cores": resource_summary.get("peak_active_cores"),
    }
    normalized_values: dict[str, float] = {}
    for field, value in numeric_values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            return False, f"resource {field} is invalid", ()
        normalized_values[field] = float(value)
    if (
        normalized_values["run_wall_seconds"] <= 0.0
        or normalized_values["sample_interval_seconds"] != 0.05
        or normalized_values["aggregate_peak_rss_bytes"] <= 0.0
        or normalized_values["peak_active_cores"]
        < normalized_values["mean_active_cores"]
    ):
        return False, "resource timing, RSS, or active-core values are invalid", ()
    sample_count = resource_summary.get("sample_count")
    if (
        not isinstance(sample_count, int)
        or isinstance(sample_count, bool)
        or sample_count < 2
    ):
        return False, "resource sample_count is invalid", ()
    parent_pid = resource_summary.get("parent_pid")
    descendants = resource_summary.get("descendant_pids")
    if (
        not isinstance(parent_pid, int)
        or isinstance(parent_pid, bool)
        or parent_pid <= 0
    ):
        return False, "resource parent_pid is invalid", ()
    if not isinstance(descendants, list) or any(
        not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
        for pid in descendants
    ):
        return False, "resource descendant_pids are invalid", ()
    if len(descendants) != len(set(descendants)) or parent_pid in descendants:
        return False, "resource process identities are duplicate", ()
    allowed_pids = {parent_pid} if expected_workers == 1 else set(descendants)
    owners: set[int] = set()
    ordinals: set[int] = set()
    for manifest in shard_manifests:
        if (
            manifest.get("run_label") != expected_run_label
            or manifest.get("evidence_completeness") != "complete"
        ):
            return False, "shard identity or completeness is invalid", ()
        ordinal = manifest.get("shard_ordinal")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal in ordinals:
            return False, "shard ordinal is invalid or duplicate", ()
        ordinals.add(ordinal)
        match = re.fullmatch(r"pid-(\d+)", str(manifest.get("worker_identity", "")))
        if match is None:
            return False, "shard worker_identity is not an actual PID", ()
        pid = int(match.group(1))
        if pid not in allowed_pids:
            return False, f"shard owner PID {pid} was not observed in the process tree", ()
        owners.add(pid)
    if len(owners) != expected_workers:
        return (
            False,
            f"expected {expected_workers} actual shard workers, observed {len(owners)}",
            tuple(sorted(owners)),
        )
    return True, "actual shard worker ownership passed", tuple(sorted(owners))


def abort_process_executor(executor: Any) -> None:
    """Terminate every live executor process and verify that none survived."""

    processes = getattr(executor, "_processes", None)
    if not isinstance(processes, Mapping) or not processes:
        executor.shutdown(wait=False, cancel_futures=True)
        raise RuntimeError("executor process identities are unavailable during abort")
    workers = tuple(processes.values())
    termination_errors: list[str] = []
    for process in workers:
        try:
            process.terminate()
        except BaseException as error:
            termination_errors.append(f"pid={getattr(process, 'pid', '?')}: {error}")
    for process in workers:
        try:
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
            if process.is_alive():
                termination_errors.append(
                    f"pid={getattr(process, 'pid', '?')}: survived terminate and kill"
                )
        except BaseException as error:
            termination_errors.append(f"pid={getattr(process, 'pid', '?')}: {error}")
    executor.shutdown(wait=not termination_errors, cancel_futures=True)
    if termination_errors:
        raise RuntimeError("; ".join(termination_errors))


def collect_performance_provenance(
    *,
    instance_paths: Mapping[str, Path],
    stage02_config_path: Path,
    stage04_config_path: Path,
    max_iterations: int,
    batch_size: int,
    runtime_environment: Mapping[str, object],
) -> dict[str, object]:
    """Capture non-secret inputs required to compare Stage 5.2 performance runs."""

    affinity_getter = getattr(os, "sched_getaffinity", None)
    affinity: dict[str, object]
    if affinity_getter is None:
        affinity = {"supported": False, "cpu_ids": []}
    else:
        affinity = {"supported": True, "cpu_ids": sorted(affinity_getter(0))}
    process_statuses: dict[str, int] = {}
    for process in psutil.process_iter(attrs=("status",)):
        try:
            status = str(process.info.get("status", "unknown"))
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        process_statuses[status] = process_statuses.get(status, 0) + 1
    return {
        "schema_version": "stage05.2-performance-provenance-v1",
        "instance_sha256": {
            name: _sha256(path) for name, path in sorted(instance_paths.items())
        },
        "warm_start": {"enabled": False, "source": None},
        "operator_surface": {
            "operator_profile": "stage02_constraint_guided",
            "stage02_config_sha256": _sha256(stage02_config_path),
            "stage04_config_sha256": _sha256(stage04_config_path),
        },
        "fixed_work_contract": {
            "exact_call_budget": 100,
            "watchdog_seconds": 120.0,
            "max_iterations": max_iterations,
            "batch_size": batch_size,
            "backend": "cpu_batch",
        },
        "worker_affinity": affinity,
        "environment_variables": {
            name: os.environ.get(name) for name in _PERFORMANCE_ENVIRONMENT_VARIABLES
        },
        "background_load": {
            "load_average": list(os.getloadavg()),
            "process_status_counts": dict(sorted(process_statuses.items())),
        },
        "power_mode": _collect_power_mode(),
        "runtime_signature": _runtime_signature(runtime_environment),
        "failure_policy": "abort_all_workers_without_fallback",
        "fallback_allowed": False,
    }


def _runtime_signature(environment: Mapping[str, object]) -> dict[str, object]:
    python = environment.get("python")
    system = environment.get("system")
    packages = environment.get("packages")
    native_extension = environment.get("native_extension")
    if (
        not isinstance(python, Mapping)
        or not isinstance(system, Mapping)
        or not isinstance(packages, Mapping)
        or not isinstance(native_extension, str)
    ):
        raise RuntimeError("runtime environment is incomplete")
    native_path = Path(native_extension)
    if not native_path.is_file():
        raise RuntimeError(f"native extension is unavailable: {native_extension}")
    return {
        "python": {
            "version": python.get("version"),
            "implementation": python.get("implementation"),
        },
        "system": dict(system),
        "packages": dict(sorted((str(key), value) for key, value in packages.items())),
        "native_extension_sha256": _sha256(native_path),
    }


def _collect_power_mode() -> dict[str, object]:
    if os.uname().sysname != "Darwin":
        return {"available": False, "source": "not_macos", "low_power_mode": None}
    source_output = _run_command(("pmset", "-g", "batt"))
    profile_output = _run_command(("pmset", "-g", "custom"))
    source = "unknown"
    source_match = re.search(r"Now drawing from '([^']+)'", source_output)
    if source_match is not None:
        source = source_match.group(1)
    if source not in {"AC Power", "Battery Power"}:
        raise RuntimeError(f"cannot determine active macOS power source: {source!r}")
    active_section = "AC Power" if "AC" in source else "Battery Power"
    low_power_mode: int | None = None
    current_section = ""
    for line in profile_output.splitlines():
        if line and not line[0].isspace():
            current_section = line.rstrip(":")
        match = re.match(r"\s*lowpowermode\s+(\d+)\s*$", line)
        if match is not None and active_section in current_section:
            low_power_mode = int(match.group(1))
            break
    if low_power_mode not in {0, 1}:
        raise RuntimeError("cannot determine active macOS low-power mode")
    return {
        "available": bool(source_output and profile_output),
        "source": source,
        "low_power_mode": low_power_mode,
    }


def _run_command(arguments: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            arguments,
            capture_output=True,
            check=False,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"performance provenance command failed: {arguments}") from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"performance provenance command returned {completed.returncode}: {arguments}"
        )
    output = completed.stdout.strip()
    if not output:
        raise RuntimeError(f"performance provenance command returned no output: {arguments}")
    return output
