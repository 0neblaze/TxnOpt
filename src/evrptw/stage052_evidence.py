"""Public Stage 5.2 prerequisite and process-resource evidence seams."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]

from evrptw.artifacts import (
    ArtifactIntegrityError,
    ArtifactReader,
    atomic_write_signed_json,
    signed_sidecar_matches,
)
from evrptw.repository import repository_root
from evrptw.stage052 import Stage052PrerequisiteRequirement
from evrptw.stage052_platform import read_windows_wsl_power_status

STAGE052_REVIEW_SCHEMA_VERSION = "stage05.2-review-v1"
STAGE052_RESOURCE_SCHEMA_VERSION = "stage05.2-run-resource-v3"
STAGE052_LEGACY_RESOURCE_SCHEMA_VERSION = "stage05.2-run-resource-v2"
STAGE052_RUNTIME_IDENTITY_SCHEMA_VERSION = "stage05.2-runtime-identity-v2"
STAGE052_CAMPAIGN_LOCK_SCHEMA_VERSION = "stage05.2-campaign-lock-v1"
STAGE052_STORAGE_ROOT_BINDING_SCHEMA_VERSION = "stage05.2-storage-root-binding-v1"
STAGE052_PERSISTENCE_ATTRIBUTION_SCHEMA_VERSION = (
    "stage05.2-persistence-attribution-v1"
)
STAGE052_RESOURCE_MEASUREMENT_SCOPE = "task_scheduling_through_parent_control_preparation"
STAGE052_PERSISTENCE_EXCLUSIONS = (
    "archive_transfer",
    "persistence_attribution_envelope_self_observation",
    "campaign_success_commit_after_gate",
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


def _finite_non_negative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class PersistenceInterval:
    """One serial active-write interval measured by a monotonic clock."""

    label: str
    started_ns: int
    completed_ns: int

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("persistence interval label is required")
        if (
            isinstance(self.started_ns, bool)
            or not isinstance(self.started_ns, int)
            or isinstance(self.completed_ns, bool)
            or not isinstance(self.completed_ns, int)
            or self.started_ns < 0
            or self.completed_ns < self.started_ns
        ):
            raise ValueError("persistence interval monotonic bounds are invalid")

    @property
    def seconds(self) -> float:
        return (self.completed_ns - self.started_ns) / 1_000_000_000

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "started_ns": self.started_ns,
            "completed_ns": self.completed_ns,
            "duration_seconds": self.seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PersistenceInterval:
        if set(payload) != {
            "label",
            "started_ns",
            "completed_ns",
            "duration_seconds",
        }:
            raise ValueError("persistence interval schema is invalid")
        label = payload.get("label")
        started = payload.get("started_ns")
        completed = payload.get("completed_ns")
        if (
            not isinstance(label, str)
            or isinstance(started, bool)
            or not isinstance(started, int)
            or isinstance(completed, bool)
            or not isinstance(completed, int)
        ):
            raise ValueError("persistence interval fields are invalid")
        interval = cls(label=label, started_ns=started, completed_ns=completed)
        duration = _finite_non_negative(payload.get("duration_seconds"), "duration_seconds")
        if not math.isclose(duration, interval.seconds, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("persistence interval duration does not match its bounds")
        return interval


@dataclass(frozen=True, slots=True)
class Stage052PersistenceAttribution:
    """Signed boundary between solver work and every primary active write.

    The primary artifact manifest is sealed before this attribution envelope is
    written, so the envelope can bind the completed manifest without a
    self-referential timing cycle.  Only the envelope itself and archive transfer
    are excluded from promotion timing.
    """

    run_label: str
    component: str
    scope: str
    subject_id: str
    primary_manifest_relative_path: str
    primary_manifest_sha256: str
    solver_seconds: float
    shard_persistence_seconds: float
    control_intervals: tuple[PersistenceInterval, ...]
    excluded_operations: tuple[str, ...] = STAGE052_PERSISTENCE_EXCLUSIONS

    def __post_init__(self) -> None:
        if not all((self.run_label, self.component, self.scope, self.subject_id)):
            raise ValueError("persistence attribution identity is incomplete")
        if (
            not self.primary_manifest_relative_path
            or re.fullmatch(r"[0-9a-f]{64}", self.primary_manifest_sha256) is None
        ):
            raise ValueError("persistence attribution manifest binding is invalid")
        solver = _finite_non_negative(self.solver_seconds, "solver_seconds")
        shard = _finite_non_negative(
            self.shard_persistence_seconds,
            "shard_persistence_seconds",
        )
        object.__setattr__(self, "solver_seconds", solver)
        object.__setattr__(self, "shard_persistence_seconds", shard)
        if self.excluded_operations != STAGE052_PERSISTENCE_EXCLUSIONS:
            raise ValueError("persistence attribution exclusions are not canonical")
        if len({interval.label for interval in self.control_intervals}) != len(
            self.control_intervals
        ):
            raise ValueError("persistence interval labels must be unique")
        previous_completed: int | None = None
        for interval in self.control_intervals:
            if previous_completed is not None and interval.started_ns < previous_completed:
                raise ValueError("persistence control intervals overlap or are unordered")
            previous_completed = interval.completed_ns
        if self.solver_seconds + self.total_persistence_seconds <= 0.0:
            raise ValueError("persistence attribution denominator must be positive")

    @property
    def control_persistence_seconds(self) -> float:
        return sum(interval.seconds for interval in self.control_intervals)

    @property
    def total_persistence_seconds(self) -> float:
        return self.shard_persistence_seconds + self.control_persistence_seconds

    @property
    def persistence_ratio(self) -> float:
        return self.total_persistence_seconds / (
            self.solver_seconds + self.total_persistence_seconds
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": STAGE052_PERSISTENCE_ATTRIBUTION_SCHEMA_VERSION,
            "run_label": self.run_label,
            "component": self.component,
            "scope": self.scope,
            "subject_id": self.subject_id,
            "primary_manifest_relative_path": self.primary_manifest_relative_path,
            "primary_manifest_sha256": self.primary_manifest_sha256,
            "solver_seconds": self.solver_seconds,
            "shard_persistence_seconds": self.shard_persistence_seconds,
            "control_intervals": [interval.to_dict() for interval in self.control_intervals],
            "control_persistence_seconds": self.control_persistence_seconds,
            "total_persistence_seconds": self.total_persistence_seconds,
            "persistence_ratio": self.persistence_ratio,
            "maximum_persistence_ratio": 0.30,
            "excluded_operations": list(self.excluded_operations),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Stage052PersistenceAttribution:
        expected = {
            "schema_version",
            "run_label",
            "component",
            "scope",
            "subject_id",
            "primary_manifest_relative_path",
            "primary_manifest_sha256",
            "solver_seconds",
            "shard_persistence_seconds",
            "control_intervals",
            "control_persistence_seconds",
            "total_persistence_seconds",
            "persistence_ratio",
            "maximum_persistence_ratio",
            "excluded_operations",
        }
        if set(payload) != expected:
            raise ValueError("persistence attribution schema is invalid")
        if payload.get("schema_version") != STAGE052_PERSISTENCE_ATTRIBUTION_SCHEMA_VERSION:
            raise ValueError("persistence attribution version is invalid")
        strings = {
            field: payload.get(field)
            for field in (
                "run_label",
                "component",
                "scope",
                "subject_id",
                "primary_manifest_relative_path",
                "primary_manifest_sha256",
            )
        }
        if any(not isinstance(value, str) for value in strings.values()):
            raise ValueError("persistence attribution identity fields are invalid")
        raw_intervals = payload.get("control_intervals")
        raw_exclusions = payload.get("excluded_operations")
        if not isinstance(raw_intervals, list) or not isinstance(raw_exclusions, list):
            raise ValueError("persistence attribution collections are invalid")
        intervals = tuple(
            PersistenceInterval.from_dict(interval)
            for interval in raw_intervals
            if isinstance(interval, Mapping)
        )
        if len(intervals) != len(raw_intervals) or any(
            not isinstance(value, str) for value in raw_exclusions
        ):
            raise ValueError("persistence attribution collection members are invalid")
        attribution = cls(
            run_label=str(strings["run_label"]),
            component=str(strings["component"]),
            scope=str(strings["scope"]),
            subject_id=str(strings["subject_id"]),
            primary_manifest_relative_path=str(strings["primary_manifest_relative_path"]),
            primary_manifest_sha256=str(strings["primary_manifest_sha256"]),
            solver_seconds=_finite_non_negative(payload.get("solver_seconds"), "solver_seconds"),
            shard_persistence_seconds=_finite_non_negative(
                payload.get("shard_persistence_seconds"),
                "shard_persistence_seconds",
            ),
            control_intervals=intervals,
            excluded_operations=tuple(raw_exclusions),
        )
        observed = (
            _finite_non_negative(
                payload.get("control_persistence_seconds"),
                "control_persistence_seconds",
            ),
            _finite_non_negative(
                payload.get("total_persistence_seconds"),
                "total_persistence_seconds",
            ),
            _finite_non_negative(payload.get("persistence_ratio"), "persistence_ratio"),
            _finite_non_negative(
                payload.get("maximum_persistence_ratio"),
                "maximum_persistence_ratio",
            ),
        )
        expected_values = (
            attribution.control_persistence_seconds,
            attribution.total_persistence_seconds,
            attribution.persistence_ratio,
            0.30,
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(observed, expected_values, strict=True)
        ):
            raise ValueError("persistence attribution derived totals are invalid")
        return attribution


@dataclass(frozen=True, slots=True)
class BatchPersistenceEnvelope:
    """Final non-self-referential timing seal for one archived batch state."""

    run_label: str
    batch_id: str
    base_attribution_sha256: str
    verified_manifest_sha256: str
    archived_manifest_sha256: str
    solver_seconds: float
    base_persistence_seconds: float
    state_intervals: tuple[PersistenceInterval, ...]

    def __post_init__(self) -> None:
        if not self.run_label or re.fullmatch(r"batch[0-9]{4}", self.batch_id) is None:
            raise ValueError("batch persistence envelope identity is invalid")
        if any(
            re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in (
                self.base_attribution_sha256,
                self.verified_manifest_sha256,
                self.archived_manifest_sha256,
            )
        ):
            raise ValueError("batch persistence envelope hashes are invalid")
        object.__setattr__(
            self,
            "solver_seconds",
            _finite_non_negative(self.solver_seconds, "solver_seconds"),
        )
        object.__setattr__(
            self,
            "base_persistence_seconds",
            _finite_non_negative(
                self.base_persistence_seconds,
                "base_persistence_seconds",
            ),
        )
        if tuple(interval.label for interval in self.state_intervals) != (
            "verified_batch_manifest_write",
            "archived_batch_manifest_write",
        ):
            raise ValueError("batch persistence state intervals are incomplete")
        if self.state_intervals[1].started_ns < self.state_intervals[0].completed_ns:
            raise ValueError("batch persistence state intervals overlap")
        if self.solver_seconds + self.total_persistence_seconds <= 0.0:
            raise ValueError("batch persistence envelope denominator must be positive")

    @property
    def state_persistence_seconds(self) -> float:
        return sum(interval.seconds for interval in self.state_intervals)

    @property
    def total_persistence_seconds(self) -> float:
        return self.base_persistence_seconds + self.state_persistence_seconds

    @property
    def persistence_ratio(self) -> float:
        return self.total_persistence_seconds / (
            self.solver_seconds + self.total_persistence_seconds
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "stage05.2-batch-persistence-envelope-v1",
            "run_label": self.run_label,
            "batch_id": self.batch_id,
            "base_attribution_sha256": self.base_attribution_sha256,
            "verified_manifest_sha256": self.verified_manifest_sha256,
            "archived_manifest_sha256": self.archived_manifest_sha256,
            "solver_seconds": self.solver_seconds,
            "base_persistence_seconds": self.base_persistence_seconds,
            "state_intervals": [interval.to_dict() for interval in self.state_intervals],
            "state_persistence_seconds": self.state_persistence_seconds,
            "total_persistence_seconds": self.total_persistence_seconds,
            "persistence_ratio": self.persistence_ratio,
            "maximum_persistence_ratio": 0.30,
            "excluded_operations": list(STAGE052_PERSISTENCE_EXCLUSIONS),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BatchPersistenceEnvelope:
        expected = {
            "schema_version",
            "run_label",
            "batch_id",
            "base_attribution_sha256",
            "verified_manifest_sha256",
            "archived_manifest_sha256",
            "solver_seconds",
            "base_persistence_seconds",
            "state_intervals",
            "state_persistence_seconds",
            "total_persistence_seconds",
            "persistence_ratio",
            "maximum_persistence_ratio",
            "excluded_operations",
        }
        if set(payload) != expected or payload.get("schema_version") != (
            "stage05.2-batch-persistence-envelope-v1"
        ):
            raise ValueError("batch persistence envelope schema is invalid")
        raw_intervals = payload.get("state_intervals")
        if not isinstance(raw_intervals, list) or len(raw_intervals) != 2:
            raise ValueError("batch persistence envelope intervals are invalid")
        intervals = tuple(
            PersistenceInterval.from_dict(value)
            for value in raw_intervals
            if isinstance(value, Mapping)
        )
        if len(intervals) != 2:
            raise ValueError("batch persistence envelope interval member is invalid")
        string_fields = (
            "run_label",
            "batch_id",
            "base_attribution_sha256",
            "verified_manifest_sha256",
            "archived_manifest_sha256",
        )
        if any(not isinstance(payload.get(field), str) for field in string_fields):
            raise ValueError("batch persistence envelope identity fields are invalid")
        envelope = cls(
            run_label=str(payload["run_label"]),
            batch_id=str(payload["batch_id"]),
            base_attribution_sha256=str(payload["base_attribution_sha256"]),
            verified_manifest_sha256=str(payload["verified_manifest_sha256"]),
            archived_manifest_sha256=str(payload["archived_manifest_sha256"]),
            solver_seconds=_finite_non_negative(payload.get("solver_seconds"), "solver_seconds"),
            base_persistence_seconds=_finite_non_negative(
                payload.get("base_persistence_seconds"),
                "base_persistence_seconds",
            ),
            state_intervals=intervals,
        )
        observed = (
            _finite_non_negative(
                payload.get("state_persistence_seconds"),
                "state_persistence_seconds",
            ),
            _finite_non_negative(
                payload.get("total_persistence_seconds"),
                "total_persistence_seconds",
            ),
            _finite_non_negative(payload.get("persistence_ratio"), "persistence_ratio"),
            _finite_non_negative(
                payload.get("maximum_persistence_ratio"),
                "maximum_persistence_ratio",
            ),
        )
        expected_values = (
            envelope.state_persistence_seconds,
            envelope.total_persistence_seconds,
            envelope.persistence_ratio,
            0.30,
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(observed, expected_values, strict=True)
        ) or payload.get("excluded_operations") != list(STAGE052_PERSISTENCE_EXCLUSIONS):
            raise ValueError("batch persistence envelope totals/exclusions are invalid")
        return envelope


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        # macOS stores extended attributes as AppleDouble ``._*`` entries on
        # ExFAT.  A copied ``._foo.dist-info`` directory matches Python's
        # distribution-name pattern but contains binary Finder metadata, not
        # package metadata.  Reject that namespace explicitly; decoding errors
        # from any real distribution still fail fast.
        distribution_path = getattr(distribution, "_path", None)
        if distribution_path is not None and Path(distribution_path).name.startswith("._"):
            continue
        raw_name = distribution.metadata.get("Name")
        if raw_name:
            versions[str(raw_name).lower()] = distribution.version
    return dict(sorted(versions.items()))


def _distribution_is_editable() -> bool:
    try:
        distribution = importlib.metadata.distribution("evrptw-reproduction")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("evrptw-reproduction is not installed") from error
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return False
    try:
        payload = json.loads(direct_url)
    except json.JSONDecodeError as error:
        raise RuntimeError("installed distribution direct_url.json is invalid") from error
    directory = payload.get("dir_info") if isinstance(payload, Mapping) else None
    return isinstance(directory, Mapping) and directory.get("editable") is True


def _installed_distribution_digest() -> str:
    try:
        distribution = importlib.metadata.distribution("evrptw-reproduction")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("evrptw-reproduction is not installed") from error
    files = distribution.files
    if files is None:
        raise RuntimeError("installed distribution has no file inventory")
    digest = hashlib.sha256()
    observed = 0
    for relative in sorted(files, key=str):
        path = Path(str(distribution.locate_file(relative)))
        if not path.is_file():
            continue
        digest.update(str(relative).encode("utf-8") + b"\0")
        digest.update(_sha256(path).encode("ascii") + b"\n")
        observed += 1
    if observed == 0:
        raise RuntimeError("installed distribution file inventory is empty")
    return digest.hexdigest()


def _runtime_identity_payload(
    *,
    wheel_path: Path,
    repository_revision: str,
) -> tuple[dict[str, object], dict[str, object]]:
    from evrptw import _core as native_core

    if re.fullmatch(r"[0-9a-f]{40}", repository_revision) is None:
        raise ValueError("repository_revision must be a full Git SHA-1")
    resolved_wheel = wheel_path.resolve()
    if not resolved_wheel.is_file():
        raise FileNotFoundError(resolved_wheel)
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Stage 5.2 evidence requires Python 3.13")
    editable = _distribution_is_editable()
    if editable:
        raise RuntimeError("Stage 5.2 evidence requires a non-editable wheel installation")
    python_executable = Path(sys.executable).resolve()
    native_extension = Path(str(native_core.__file__)).resolve()
    if not python_executable.is_file() or not native_extension.is_file():
        raise RuntimeError("Python or native extension runtime file is unavailable")
    dependencies = _dependency_versions()
    dependency_bytes = json.dumps(
        dependencies,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    source_root = repository_root().resolve()
    source_mount = _findmnt_identity(source_root)
    if source_mount.get("filesystem") != "ext4":
        raise RuntimeError("Stage 5.2 source repository must be on WSL2 ext4")
    tracked: dict[str, object] = {
        "schema_version": STAGE052_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "repository_revision": repository_revision,
        "wheel_filename": resolved_wheel.name,
        "wheel_sha256": _sha256(resolved_wheel),
        "python_version": sys.version.split()[0],
        "python_executable_sha256": _sha256(python_executable),
        "native_extension_sha256": _sha256(native_extension),
        "dependency_versions": dependencies,
        "dependency_manifest_sha256": hashlib.sha256(dependency_bytes).hexdigest(),
        "installed_distribution_sha256": _installed_distribution_digest(),
        "installed_editable": editable,
        "machine_identity": _stage052_machine_identity(),
        "source_repository_mount": source_mount,
    }
    local = {
        **tracked,
        "wheel_path": str(resolved_wheel),
        "python_executable": str(python_executable),
        "native_extension": str(native_extension),
        "source_repository_root": str(source_root),
    }
    return tracked, local


def verify_stage052_source_snapshot(root: Path) -> dict[str, object]:
    """Require a clean ext4 Git snapshot whose entire source tree is read-only."""

    resolved = root.resolve()
    mount = _findmnt_identity(resolved)
    if mount.get("filesystem") != "ext4":
        raise RuntimeError("Stage 5.2 source snapshot must be on WSL2 ext4")
    status = subprocess.run(
        ("git", "-C", str(resolved), "status", "--porcelain", "--untracked-files=no"),
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    ).stdout
    if status.strip():
        raise RuntimeError("Stage 5.2 source snapshot must be a clean Git checkout")
    revision = subprocess.run(
        ("git", "-C", str(resolved), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    ).stdout.strip()
    tracked_output = subprocess.run(
        ("git", "-C", str(resolved), "ls-files", "-z"),
        check=True,
        capture_output=True,
        timeout=10.0,
    ).stdout
    tracked_paths = [
        resolved / os.fsdecode(raw)
        for raw in tracked_output.split(b"\0")
        if raw
    ]
    if not tracked_paths:
        raise RuntimeError("Stage 5.2 source snapshot has no tracked files")
    tracked_relative = {
        path.relative_to(resolved).as_posix() for path in tracked_paths
    }
    allowed_local_files = {
        "configs/stage052_campaign_lock.local.json",
        "configs/stage052_campaign_lock.local.sha256",
        "configs/stage052_runtime_identity.local.json",
        "configs/stage052_storage_roots.local.toml",
    }
    allowed_untracked: dict[str, str] = {}
    writable: list[str] = []
    for path in tracked_paths:
        if not path.is_file():
            raise RuntimeError(f"tracked Stage 5.2 source file is unavailable: {path}")
    source_paths = [resolved, *resolved.rglob("*")]
    for path in source_paths:
        relative = path.relative_to(resolved)
        if relative.parts and relative.parts[0] == ".git":
            continue
        relative_text = relative.as_posix()
        if path.is_symlink():
            raise RuntimeError(f"Stage 5.2 source snapshot contains a symlink: {relative_text}")
        if path.lstat().st_mode & 0o222:
            writable.append(relative_text or ".")
        if not path.is_file() or relative_text in tracked_relative:
            continue
        allowed = relative_text in allowed_local_files or (
            len(relative.parts) == 3
            and relative.parts[:2] == ("data", "schneider")
        )
        if not allowed:
            raise RuntimeError(
                "Stage 5.2 source snapshot contains an unregistered untracked file: "
                f"{relative_text}"
            )
        allowed_untracked[relative_text] = _sha256(path)
    if writable:
        raise RuntimeError(
            "Stage 5.2 source snapshot contains writable tracked paths: "
            + ", ".join(sorted(writable)[:10])
        )
    return {
        "repository_revision": revision,
        "mount": mount,
        "tracked_file_count": len(tracked_paths),
        "allowed_untracked_sha256": dict(sorted(allowed_untracked.items())),
        "read_only": True,
    }


def _stage052_machine_identity() -> dict[str, object]:
    """Collect the stable Windows/WSL2 hardware and mount identity."""

    system = platform.system()
    release = platform.release()
    base: dict[str, object] = {
        "host_system": system,
        "linux_kernel": release,
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_bytes": int(psutil.virtual_memory().total),
    }
    if system != "Linux" or "microsoft" not in release.casefold():
        return {**base, "execution_environment": system.casefold()}
    windows = json.loads(
        _run_command(
            (
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_OperatingSystem | Select-Object "
                "Caption,Version,BuildNumber,TotalVisibleMemorySize) | "
                "ConvertTo-Json -Compress",
            )
        )
    )
    cpu = json.loads(
        _run_command(
            (
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object "
                "Name,NumberOfCores,NumberOfLogicalProcessors) | ConvertTo-Json -Compress",
            )
        )
    )
    gpu_line = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        )
    ).splitlines()
    if len(gpu_line) != 1 or len(gpu_line[0].split(",")) != 3:
        raise RuntimeError("Stage 5.2 requires exactly one auditable NVIDIA GPU")
    gpu_name, driver, capability = (part.strip() for part in gpu_line[0].split(","))
    wsl_version = _run_command(("wsl.exe", "--version")).replace("\x00", "")
    ext4 = _findmnt_identity(Path(sys.executable))
    d_archive = _findmnt_identity(Path("/mnt/d"))
    if ext4.get("filesystem") != "ext4" or d_archive.get("filesystem") != "9p":
        raise RuntimeError("Stage 5.2 runtime must use ext4 execution and D-drive 9p archive")
    d_disk = json.loads(
        _run_command(
            (
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$p=Get-Partition -DriveLetter D; $d=$p|Get-Disk; "
                "[pscustomobject]@{FriendlyName=$d.FriendlyName;SerialNumber=$d.SerialNumber;"
                "BusType=[string]$d.BusType;Number=$d.Number}|ConvertTo-Json -Compress",
            )
        )
    )
    if not isinstance(d_disk, Mapping) or str(d_disk.get("BusType")) != "NVMe":
        raise RuntimeError("Stage 5.2 D archive must be backed by NVMe")
    return {
        **base,
        "execution_environment": "windows11_wsl2",
        "windows": windows,
        "wsl_version": wsl_version,
        "cpu": cpu,
        "nvidia_gpu": {
            "name": gpu_name,
            "driver_version": driver,
            "cuda_capability": capability,
        },
        "ext4_mount": ext4,
        "d_archive_mount": d_archive,
        "d_archive_disk": dict(d_disk),
    }


def _findmnt_identity(path: Path) -> dict[str, object]:
    payload = json.loads(
        _run_command(
            (
                "findmnt",
                "--json",
                "--target",
                str(path),
                "--output",
                "SOURCE,FSTYPE,UUID,TARGET",
            )
        )
    )
    filesystems = payload.get("filesystems") if isinstance(payload, Mapping) else None
    if not isinstance(filesystems, list) or len(filesystems) != 1:
        raise RuntimeError(f"findmnt identity is invalid for {path}")
    item = filesystems[0]
    if not isinstance(item, Mapping):
        raise RuntimeError(f"findmnt identity is invalid for {path}")
    return {
        "source": str(item.get("source", "")),
        "filesystem": str(item.get("fstype", "")),
        "uuid": str(item.get("uuid", "")),
        "target": str(item.get("target", "")),
    }


def create_stage052_runtime_identity(
    *,
    output_path: Path,
    wheel_path: Path,
    repository_revision: str,
) -> dict[str, object]:
    """Freeze one local non-editable Python/native/wheel runtime identity."""

    tracked, local = _runtime_identity_payload(
        wheel_path=wheel_path,
        repository_revision=repository_revision,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(local, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return tracked


def verify_stage052_runtime_identity(
    path: Path,
    *,
    expected_repository_revision: str,
) -> dict[str, object]:
    """Verify the current process exactly matches a frozen local wheel runtime."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read Stage 5.2 runtime identity: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Stage 5.2 runtime identity must be an object")
    wheel_path = payload.get("wheel_path")
    if not isinstance(wheel_path, str) or not wheel_path:
        raise RuntimeError("Stage 5.2 runtime identity wheel_path is missing")
    tracked, current_local = _runtime_identity_payload(
        wheel_path=Path(wheel_path),
        repository_revision=expected_repository_revision,
    )
    for field, expected in current_local.items():
        observed = payload.get(field)
        if observed != expected:
            label = field.replace("_sha256", " hash")
            raise RuntimeError(
                f"Stage 5.2 runtime identity {label} mismatch: "
                f"expected={expected!r} observed={observed!r}"
            )
    if set(payload) != set(current_local):
        raise RuntimeError("Stage 5.2 runtime identity contains unknown or missing fields")
    return tracked


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    _update_digest_from_path(digest, path)
    return digest.hexdigest()


def _update_digest_from_path(digest: Any, path: Path) -> None:
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)


def stage052_storage_root_binding(
    *,
    alias: str,
    volume: Mapping[str, object],
) -> dict[str, object]:
    """Build the path-free staging-volume identity embedded in raw evidence."""

    if re.fullmatch(r"[a-z][a-z0-9_]*", alias) is None:
        raise ValueError("Stage 5.2 staging root alias is invalid")
    if set(volume) != {"device_uuid", "filesystem"}:
        raise ValueError("Stage 5.2 staging root volume fields are invalid")
    device_uuid = volume.get("device_uuid")
    filesystem = volume.get("filesystem")
    if (
        not isinstance(device_uuid, str)
        or not device_uuid
        or not isinstance(filesystem, str)
        or not filesystem
    ):
        raise ValueError("Stage 5.2 staging root volume identity is invalid")
    return {
        "schema_version": STAGE052_STORAGE_ROOT_BINDING_SCHEMA_VERSION,
        "alias": alias,
        "volume": {
            "device_uuid": device_uuid,
            "filesystem": filesystem,
        },
    }


def verify_stage052_storage_root_binding(
    metadata: Mapping[str, object],
    *,
    locator_path: Path,
    expected_alias: str,
) -> dict[str, object]:
    """Verify a path-free raw binding against the ignored local root locator."""

    from evrptw.stage052_campaign import StorageRootLocator

    observed = metadata.get("staging_root")
    if not isinstance(observed, Mapping) or set(observed) != {
        "schema_version",
        "alias",
        "volume",
    }:
        raise ArtifactIntegrityError("Stage 5.2 staging root binding is missing or invalid")
    if observed.get("schema_version") != STAGE052_STORAGE_ROOT_BINDING_SCHEMA_VERSION:
        raise ArtifactIntegrityError("Stage 5.2 staging root binding schema is invalid")
    alias = observed.get("alias")
    volume = observed.get("volume")
    if alias != expected_alias or not isinstance(volume, Mapping):
        raise ArtifactIntegrityError("Stage 5.2 staging root alias or volume is invalid")
    try:
        normalized = stage052_storage_root_binding(alias=expected_alias, volume=volume)
        locator = StorageRootLocator.from_toml(locator_path)
        configured = locator.resolve(expected_alias)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            "Stage 5.2 staging root local binding cannot be verified"
        ) from error
    if normalized != dict(observed) or normalized["volume"] != configured.volume.to_dict():
        raise ArtifactIntegrityError(
            "Stage 5.2 staging root identity does not match the local locator"
        )
    return normalized


def verify_stage052_review_files(
    raw_dir: Path,
    review: Mapping[str, object],
) -> dict[str, Path]:
    """Verify legacy or immutable-generation review file references."""

    files = review.get("files")
    if not isinstance(files, Mapping):
        raise ArtifactIntegrityError("prerequisite review file identity mismatch")
    relative_paths = tuple(str(value) for value in files)
    campaign_review = review.get("schema_version") == "stage05.2-campaign-review-v1"
    expected_campaign_names = {
        "per_run_results.csv",
        "family_summary.csv",
        "budget_summary.csv",
        "anytime_summary.csv",
        "resource_summary.csv",
        "persistence_summary.csv",
        "performance_gates.csv",
        "gpu_decision.json",
        "failure_analysis.csv",
        "review_findings.csv",
        "review_report.md",
    }
    campaign_publication_files: Mapping[str, object] | None = None
    if campaign_review:
        expected_campaign_files = {
            "per_run_results": "per_run_results.csv",
            "family_summary": "family_summary.csv",
            "budget_summary": "budget_summary.csv",
            "anytime_summary": "anytime_summary.csv",
            "resource_summary": "resource_summary.csv",
            "persistence_summary": "persistence_summary.csv",
            "performance_gates": "performance_gates.csv",
            "gpu_decision": "gpu_decision.json",
            "failure_analysis": "failure_analysis.csv",
            "review_findings": "review_findings.csv",
            "review_report": "review_report.md",
        }
        parsed = [Path(value) for value in relative_paths]
        if (
            len(parsed) != len(expected_campaign_names)
            or {path.name for path in parsed} != expected_campaign_names
            or any(len(path.parts) != 3 or path.parts[0] != "generations" for path in parsed)
            or len({path.parts[1] for path in parsed}) != 1
        ):
            raise ArtifactIntegrityError("campaign review file identity mismatch")
        generation = parsed[0].parts[1]
        if re.fullmatch(r"[0-9a-f]{64}", generation) is None:
            raise ArtifactIntegrityError("campaign review generation identity is invalid")
        publication_files = review.get("publication_files")
        if not isinstance(publication_files, Mapping) or set(publication_files) != set(
            expected_campaign_files
        ):
            raise ArtifactIntegrityError("campaign review publication file set is invalid")
        published: dict[str, str] = {}
        for key, item in publication_files.items():
            if not isinstance(item, Mapping) or set(item) != {"relative_path", "sha256"}:
                raise ArtifactIntegrityError("campaign review publication file is invalid")
            relative_path = item.get("relative_path")
            checksum = item.get("sha256")
            if not isinstance(relative_path, str) or not isinstance(checksum, str):
                raise ArtifactIntegrityError("campaign review publication identity is invalid")
            if Path(relative_path).name != expected_campaign_files[str(key)]:
                raise ArtifactIntegrityError(
                    "campaign review publication key/filename mapping is invalid"
                )
            published[relative_path] = checksum
        if published != {str(path): str(checksum) for path, checksum in files.items()}:
            raise ArtifactIntegrityError("campaign review files/publication_files differ")
        campaign_publication_files = publication_files
    else:
        if len(files) not in {2, 3}:
            raise ArtifactIntegrityError("prerequisite review file identity mismatch")
        legacy = set(relative_paths) == {"review_findings.csv", "review_report.md"}
        parsed = [] if legacy else [Path(value) for value in relative_paths]
    if not campaign_review and parsed:
        observed_names = {path.name for path in parsed}
        if (
            observed_names
            not in (
                {"review_findings.csv", "review_report.md"},
                {"review_findings.csv", "review_report.md", "semantic_mismatches.csv"},
            )
            or any(len(path.parts) != 3 or path.parts[0] != "generations" for path in parsed)
            or len({path.parts[1] for path in parsed}) != 1
        ):
            raise ArtifactIntegrityError("prerequisite review file identity mismatch")
        generation = parsed[0].parts[1]
        if len(generation) != 64 or any(
            character not in "0123456789abcdef" for character in generation
        ):
            raise ArtifactIntegrityError("prerequisite review generation identity is invalid")
    review_dir = (raw_dir / "review").resolve()
    verified: dict[str, Path] = {}
    for relative, checksum in files.items():
        relative_text = str(relative)
        path = (review_dir / relative_text).resolve()
        if review_dir not in path.parents or not path.is_file() or _sha256(path) != str(checksum):
            raise ArtifactIntegrityError(f"prerequisite review checksum mismatch: {relative_text}")
        verified[relative_text] = path
    if campaign_publication_files is not None:
        digest = hashlib.sha256()
        for key in sorted(campaign_publication_files):
            item = campaign_publication_files[key]
            assert isinstance(item, Mapping)
            relative_path = str(item["relative_path"])
            digest.update(key.encode("utf-8") + b"\0")
            _update_digest_from_path(digest, verified[relative_path])
        generation = next(iter(Path(path).parts[1] for path in verified))
        if digest.hexdigest() != generation:
            raise ArtifactIntegrityError("campaign review generation digest does not replay")
    elif parsed:
        by_name = {path.name: verified[path.as_posix()] for path in parsed}
        generation = parsed[0].parts[1]
        digest = hashlib.sha256()
        _update_digest_from_path(digest, by_name["review_findings.csv"])
        digest.update(b"\0")
        _update_digest_from_path(digest, by_name["review_report.md"])
        if "semantic_mismatches.csv" in by_name:
            digest.update(b"\0")
            _update_digest_from_path(digest, by_name["semantic_mismatches.csv"])
        generation_digest = digest.hexdigest()
        if generation_digest != generation:
            raise ArtifactIntegrityError("prerequisite review generation digest does not replay")
    return verified


_CAMPAIGN_COMMON_GATES = frozenset(
    {
        "accepted_prerequisite",
        "batch_shard_artifact_replay",
        "bks_model_compatibility",
        "campaign_geometry",
        "campaign_identity",
        "campaign_planning_replay",
        "event_evidence",
        "instance_input_hashes",
        "persistence_ratio",
        "power_load",
        "resource_limits",
        "runtime_provenance",
        "storage_root_roles",
        "storage_roots",
        "unique_shard_identity",
    }
)
CAMPAIGN_PILOT_GATES = _CAMPAIGN_COMMON_GATES | {
    "pilot_campaign_drills",
    "publication_dry_run",
}
CAMPAIGN_FORMAL_GATES = _CAMPAIGN_COMMON_GATES | {
    "pilot_archive_root_coverage",
    "rolling_capacity_replay",
}


def verify_stage052_campaign_gate_set(
    review: Mapping[str, object],
    *,
    scope: str,
) -> None:
    """Require the exact independent campaign gate surface for READY evidence."""

    expected = CAMPAIGN_PILOT_GATES if scope == "pilot" else CAMPAIGN_FORMAL_GATES
    if scope not in {"pilot", "formal"}:
        raise ArtifactIntegrityError("campaign review gate scope is invalid")
    gates = review.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(expected):
        raise ArtifactIntegrityError("campaign review mandatory gate set is incomplete")
    if any(
        not isinstance(gate, Mapping) or gate.get("passed") is not True
        for gate in gates.values()
    ):
        raise ArtifactIntegrityError("campaign review contains a failed or invalid gate")


@dataclass(frozen=True, slots=True)
class Stage052PrerequisiteIdentity:
    run_label: str
    component: str
    status: str
    repository_revision: str
    configuration_sha256: str
    raw_manifest_sha256: str
    review_manifest_sha256: str
    scope: str = "performance"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def stage052_campaign_lock_entry(
    raw_dir: Path,
    identity: Stage052PrerequisiteIdentity,
) -> dict[str, object]:
    """Build the exact immutable predecessor identity stored in the campaign lock."""

    reader = ArtifactReader(raw_dir.resolve())
    metadata_items = [
        item
        for item in reader.manifest.get("artifacts", ())
        if isinstance(item, Mapping) and item.get("artifact_type") == "manifest_metadata"
    ]
    if len(metadata_items) != 1:
        raise ArtifactIntegrityError("campaign-lock predecessor metadata is incomplete")
    metadata = reader.read_json(str(metadata_items[0].get("relative_path", "")))
    runtime = metadata.get("runtime_identity")
    if not isinstance(runtime, Mapping):
        raise ArtifactIntegrityError("campaign-lock predecessor runtime identity is missing")
    runtime_bytes = json.dumps(
        dict(runtime), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        **identity.to_dict(),
        "runtime_identity_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
    }


def verify_stage052_campaign_lock(
    lock_path: Path,
    *,
    raw_dir: Path,
    identity: Stage052PrerequisiteIdentity,
) -> None:
    """Require one prerequisite to match an exact signed campaign-lock entry."""

    sidecar = lock_path.with_suffix(".sha256")
    if not signed_sidecar_matches(lock_path, sidecar):
        raise ArtifactIntegrityError("Stage 5.2 campaign lock or sidecar is invalid")
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError("Stage 5.2 campaign lock is unreadable") from error
    entries = payload.get("entries") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != STAGE052_CAMPAIGN_LOCK_SCHEMA_VERSION
        or not isinstance(entries, list)
    ):
        raise ArtifactIntegrityError("Stage 5.2 campaign lock schema is invalid")
    expected = stage052_campaign_lock_entry(raw_dir, identity)
    matches = [entry for entry in entries if isinstance(entry, Mapping) and dict(entry) == expected]
    if len(matches) != 1:
        raise ArtifactIntegrityError(
            f"Stage 5.2 campaign lock does not bind exact prerequisite {identity.run_label}"
        )


def upsert_stage052_campaign_lock(
    lock_path: Path,
    *,
    raw_dir: Path,
    identity: Stage052PrerequisiteIdentity,
) -> tuple[Path, Path]:
    """Publish an accepted reviewed identity for the next current-chain producer."""

    entries: list[dict[str, object]] = []
    if lock_path.exists():
        sidecar = lock_path.with_suffix(".sha256")
        if not signed_sidecar_matches(lock_path, sidecar):
            raise ArtifactIntegrityError("cannot update an invalid Stage 5.2 campaign lock")
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != STAGE052_CAMPAIGN_LOCK_SCHEMA_VERSION
            or not isinstance(payload.get("entries"), list)
        ):
            raise ArtifactIntegrityError("cannot update a campaign lock with invalid schema")
        entries = [dict(entry) for entry in payload["entries"] if isinstance(entry, Mapping)]
    entry = stage052_campaign_lock_entry(raw_dir, identity)
    entries = [item for item in entries if item.get("run_label") != identity.run_label]
    entries.append(entry)
    entries.sort(key=lambda item: str(item.get("run_label", "")))
    return atomic_write_signed_json(
        lock_path,
        {
            "schema_version": STAGE052_CAMPAIGN_LOCK_SCHEMA_VERSION,
            "entries": entries,
        },
    )


@dataclass(frozen=True, slots=True)
class JobParallelSelectionIdentity:
    """Reviewed D selection consumed by every Stage 5.2 E producer."""

    selected_workers: int
    selected_run_label: str
    input_runs: tuple[str, str, str]
    run_wall_seconds: tuple[float, float, float]
    aggregate_peak_rss_gib: tuple[float, float, float]
    speedups: tuple[float, float, float]
    input_raw_manifest_sha256: tuple[str, str, str]
    review_manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_workers": self.selected_workers,
            "selected_run_label": self.selected_run_label,
            "input_runs": list(self.input_runs),
            "resource_metrics": {
                str(worker): {
                    "run_wall_seconds": self.run_wall_seconds[index],
                    "aggregate_peak_rss_gib": self.aggregate_peak_rss_gib[index],
                    "speedup": self.speedups[index],
                }
                for index, worker in enumerate((1, 2, 4))
            },
            "input_raw_manifest_sha256": {
                run_label: self.input_raw_manifest_sha256[index]
                for index, run_label in enumerate(self.input_runs)
            },
            "review_manifest_sha256": self.review_manifest_sha256,
        }


def verify_job_parallel_selection(
    raw_dir: Path,
    prerequisite: Stage052PrerequisiteIdentity,
) -> JobParallelSelectionIdentity:
    """Recompute the accepted 1/2/4-worker choice from the signed D review."""

    if (
        prerequisite.run_label != raw_dir.resolve().name
        or prerequisite.component != "job_parallel"
        or prerequisite.scope != "performance"
        or prerequisite.status != "READY_FOR_STAGE052_NATIVE_KERNELS"
    ):
        raise ArtifactIntegrityError("job-parallel selection prerequisite identity mismatch")
    review_path = raw_dir.resolve() / "review" / "review_manifest.json"
    if _sha256(review_path) != prerequisite.review_manifest_sha256:
        raise ArtifactIntegrityError("job-parallel selection review checksum mismatch")
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError("cannot read job-parallel selection review") from error
    selected = review.get("selected_workers")
    selected_run = review.get("selected_run_label")
    input_runs = review.get("input_runs")
    metrics = review.get("resource_metrics")
    raw_manifest_hashes = review.get("input_raw_manifest_sha256")
    if isinstance(selected, bool) or selected not in {2, 4}:
        raise ArtifactIntegrityError("job-parallel selected_workers must be 2 or 4")
    if (
        not isinstance(input_runs, list)
        or len(input_runs) != 3
        or not all(isinstance(item, str) for item in input_runs)
        or len(set(input_runs)) != 3
    ):
        raise ArtifactIntegrityError("job-parallel input_runs must contain three unique labels")
    pattern = re.compile(r"^stage05\.2_job_parallel_(?:attempt|rerun)[0-9]{2}$")
    if any(pattern.fullmatch(item) is None for item in input_runs):
        raise ArtifactIntegrityError("job-parallel input run label is not canonical")
    if not isinstance(metrics, dict) or set(metrics) != {"1", "2", "4"}:
        raise ArtifactIntegrityError("job-parallel resource metrics must contain 1/2/4 workers")
    if (
        not isinstance(raw_manifest_hashes, dict)
        or set(raw_manifest_hashes) != set(input_runs)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in raw_manifest_hashes.values()
        )
    ):
        raise ArtifactIntegrityError("job-parallel input raw manifest hashes are invalid")
    for run_label in input_runs:
        selected_dir = raw_dir.parent / run_label
        selected_reader = ArtifactReader(selected_dir)
        if _sha256(selected_reader.result.manifest_path) != raw_manifest_hashes[run_label]:
            raise ArtifactIntegrityError(
                f"job-parallel selection is stale for input raw manifest: {run_label}"
            )
    times: list[float] = []
    rss: list[float] = []
    speedups: list[float] = []
    for worker in (1, 2, 4):
        item = metrics[str(worker)]
        if not isinstance(item, dict):
            raise ArtifactIntegrityError("job-parallel resource metric must be an object")
        try:
            values = tuple(
                float(item[field])
                for field in (
                    "run_wall_seconds",
                    "aggregate_peak_rss_gib",
                    "speedup",
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactIntegrityError("invalid job-parallel resource metric") from error
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ArtifactIntegrityError("job-parallel resource metric must be finite and positive")
        times.append(values[0])
        rss.append(values[1])
        speedups.append(values[2])
    from evrptw.stage052 import select_worker_count

    recomputed = select_worker_count(
        dict(zip((1, 2, 4), times, strict=True)),
        dict(zip((1, 2, 4), rss, strict=True)),
    )
    expected_selected_run = input_runs[(1, 2, 4).index(recomputed)]
    if selected != recomputed or selected_run != expected_selected_run:
        raise ArtifactIntegrityError("job-parallel top-level selection does not recompute")
    gate = review.get("gates", {}).get("worker_selection")
    if not isinstance(gate, dict) or any(
        gate.get(field) != review.get(field)
        for field in (
            "selected_workers",
            "selected_run_label",
            "input_runs",
            "resource_metrics",
            "input_raw_manifest_sha256",
        )
    ):
        raise ArtifactIntegrityError("job-parallel selection gate/top-level mismatch")
    return JobParallelSelectionIdentity(
        selected_workers=recomputed,
        selected_run_label=expected_selected_run,
        input_runs=tuple(input_runs),
        run_wall_seconds=tuple(times),  # type: ignore[arg-type]
        aggregate_peak_rss_gib=tuple(rss),  # type: ignore[arg-type]
        speedups=tuple(speedups),  # type: ignore[arg-type]
        input_raw_manifest_sha256=tuple(
            str(raw_manifest_hashes[run_label]) for run_label in input_runs
        ),  # type: ignore[arg-type]
        review_manifest_sha256=prerequisite.review_manifest_sha256,
    )


def verify_stage052_prerequisite(
    raw_dir: Path,
    *,
    expected_component: str,
    expected_status: str | None = None,
    allowed_statuses: Sequence[str] = (),
    expected_run_label: str | None = None,
    expected_scope: str = "performance",
    require_passed_review: bool = True,
) -> Stage052PrerequisiteIdentity:
    """Verify an accepted Stage 5.2 producer and independent review bundle."""

    raw_dir = raw_dir.resolve()
    if expected_status is not None and allowed_statuses:
        raise ValueError("use expected_status or allowed_statuses, not both")
    statuses = (expected_status,) if expected_status is not None else tuple(allowed_statuses)
    if not statuses:
        raise ValueError("at least one prerequisite review status is required")
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
    expected_schema = (
        "stage05.2-campaign-review-v1"
        if expected_component == "benchmark" and expected_scope in {"pilot", "formal"}
        else STAGE052_REVIEW_SCHEMA_VERSION
    )
    expected_review = {
        "schema_version": expected_schema,
        "run_label": raw_dir.name,
        "component": expected_component,
        "scope": expected_scope,
    }
    for field, expected in expected_review.items():
        if review.get(field) != expected:
            raise ArtifactIntegrityError(
                f"prerequisite review {field} mismatch: "
                f"expected={expected} observed={review.get(field)}"
            )
    observed_status = str(review.get("status", ""))
    if observed_status not in statuses:
        raise ArtifactIntegrityError(
            "prerequisite review status mismatch: "
            f"expected one of {list(statuses)} observed={observed_status}"
        )
    gates = review.get("gates")
    if (
        not isinstance(gates, dict)
        or not gates
        or (
            require_passed_review
            and any(
                not isinstance(gate, dict) or gate.get("passed") is not True
                for gate in gates.values()
            )
        )
    ):
        raise ArtifactIntegrityError("prerequisite review contains an invalid gate set")
    verify_stage052_review_files(raw_dir, review)

    reader = ArtifactReader(raw_dir)
    if reader.manifest.get("evidence_completeness") != "complete":
        raise ArtifactIntegrityError("prerequisite raw evidence is partial")
    current_raw_manifest_sha256 = _sha256(reader.result.manifest_path)
    if review.get("raw_manifest_sha256") != current_raw_manifest_sha256:
        raise ArtifactIntegrityError("prerequisite review is stale for the current raw manifest")
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
        or metadata.get("scope") != expected_scope
        or metadata.get("repository_dirty") is not False
    ):
        raise ArtifactIntegrityError("prerequisite producer identity mismatch")
    if metadata.get("persistence_attribution") == "primary_active_writes_v1":
        attribution_path = (
            raw_dir / "control" / f"{raw_dir.name}_persistence_attribution.json"
        )
        attribution_sidecar = attribution_path.with_suffix(".sha256")
        if (
            not attribution_path.is_file()
            or not attribution_sidecar.is_file()
            or review.get("persistence_attribution_sha256")
            != _sha256(attribution_path)
            or review.get("persistence_attribution_sidecar_sha256")
            != _sha256(attribution_sidecar)
            or not signed_sidecar_matches(attribution_path, attribution_sidecar)
        ):
            raise ArtifactIntegrityError(
                "prerequisite review is stale for the persistence attribution envelope"
            )
    config_checksum = str(config_items[0].get("checksum", ""))
    if metadata.get("configuration_sha256") != config_checksum:
        raise ArtifactIntegrityError("prerequisite configuration checksum mismatch")
    revision = str(metadata.get("repository_revision", ""))
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ArtifactIntegrityError("prerequisite repository revision is invalid")
    return Stage052PrerequisiteIdentity(
        run_label=raw_dir.name,
        component=expected_component,
        status=observed_status,
        repository_revision=revision,
        configuration_sha256=config_checksum,
        raw_manifest_sha256=current_raw_manifest_sha256,
        review_manifest_sha256=_sha256(review_manifest_path),
        scope=expected_scope,
    )


def verify_stage052_evidence_input(
    raw_dir: Path,
    requirement: Stage052PrerequisiteRequirement,
) -> Stage052PrerequisiteIdentity:
    """Verify one named contract input, including signed NOT_READY remediation input."""

    identity = verify_stage052_prerequisite(
        raw_dir,
        expected_component=requirement.component.value,
        allowed_statuses=requirement.allowed_statuses,
        expected_run_label=requirement.exact_run_label,
        expected_scope=requirement.scope,
        require_passed_review=requirement.requires_passed_review,
    )
    if not requirement.requires_current_chain_identity:
        return identity

    review_path = raw_dir.resolve() / "review" / "review_manifest.json"
    try:
        review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError("current-chain review manifest is unreadable") from error
    if not isinstance(review_payload, Mapping):
        raise ArtifactIntegrityError("current-chain review manifest is invalid")
    current_review_files = verify_stage052_review_files(raw_dir.resolve(), review_payload)
    if any(
        len(Path(relative).parts) != 3
        or Path(relative).parts[0] != "generations"
        for relative in current_review_files
    ):
        raise ArtifactIntegrityError(
            "current-chain prerequisite requires an immutable review generation"
        )
    if {path.name for path in current_review_files.values()} != {
        "review_findings.csv",
        "review_report.md",
        "semantic_mismatches.csv",
    }:
        raise ArtifactIntegrityError(
            "current-chain prerequisite requires semantic_mismatches.csv"
        )
    reader = ArtifactReader(raw_dir.resolve())
    metadata_items = [
        item
        for item in reader.manifest.get("artifacts", ())
        if isinstance(item, Mapping) and item.get("artifact_type") == "manifest_metadata"
    ]
    if len(metadata_items) != 1:
        raise ArtifactIntegrityError("current-chain prerequisite metadata is incomplete")
    metadata = reader.read_json(str(metadata_items[0].get("relative_path", "")))
    expected_fields = {"backend": "cpu_batch"}
    if requirement.component.value in {"perf_baseline", "hot_path"}:
        expected_fields.update(
            {
                "storage_policy_version": "artifact-storage-v1",
                "screening_schema_version": "screening_decisions_v1",
            }
        )
    else:
        expected_fields.update(
            {
                "storage_policy_version": "artifact-storage-v2",
                "screening_schema_version": "screening_decisions_v3",
            }
        )
    if any(metadata.get(field) != expected for field, expected in expected_fields.items()):
        raise ArtifactIntegrityError(
            "prerequisite is historical evidence, not the current v3/cpu_batch chain"
        )
    observed_runtime = metadata.get("runtime_identity")
    if not isinstance(observed_runtime, Mapping):
        raise ArtifactIntegrityError("current-chain prerequisite has no frozen runtime identity")
    root = repository_root()
    runtime_path = root / (
        "configs/stage052_runtime_identity.local.json"
    )
    try:
        verified_runtime = verify_stage052_runtime_identity(
            runtime_path,
            expected_repository_revision=identity.repository_revision,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            "current-chain prerequisite runtime identity cannot be verified"
        ) from error
    if dict(observed_runtime) != verified_runtime:
        raise ArtifactIntegrityError(
            "current-chain prerequisite runtime differs from the active frozen wheel"
        )
    verify_stage052_storage_root_binding(
        metadata,
        locator_path=root / "configs/stage052_storage_roots.local.toml",
        expected_alias="wsl_staging",
    )
    return identity


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
    process_peak_rss_bytes: tuple[tuple[int, int], ...]
    mean_active_cores: float
    peak_active_cores: float
    load1_min: float
    load1_mean: float
    load1_max: float
    load1_sample_count: int
    sample_count: int
    status: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["descendant_pids"] = list(self.descendant_pids)
        payload["process_peak_rss_bytes"] = {
            str(pid): peak for pid, peak in self.process_peak_rss_bytes
        }
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
        self._process_peak_rss: dict[int, int] = {}
        self._load1_samples: list[float] = []
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
            process_peak_rss_bytes=tuple(sorted(self._process_peak_rss.items())),
            mean_active_cores=(
                sum(self._active_core_samples) / len(self._active_core_samples)
                if self._active_core_samples
                else 0.0
            ),
            peak_active_cores=max(self._active_core_samples, default=0.0),
            load1_min=min(self._load1_samples, default=0.0),
            load1_mean=(
                sum(self._load1_samples) / len(self._load1_samples) if self._load1_samples else 0.0
            ),
            load1_max=max(self._load1_samples, default=0.0),
            load1_sample_count=len(self._load1_samples),
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
                            process_rss = int(process.memory_info().rss)
                            times = process.cpu_times()
                            process_cpu = float(times.user + times.system)
                            rss += process_rss
                            cpu += process_cpu
                            self._process_peak_rss[process.pid] = max(
                                self._process_peak_rss.get(process.pid, 0),
                                process_rss,
                            )
                            if process.pid != self.parent_pid:
                                self._descendant_pids.add(process.pid)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        continue
                self._peak_rss = max(self._peak_rss, rss)
                self._sample_count += 1
                self._load1_samples.append(float(os.getloadavg()[0]))
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

    schema_version = resource_summary.get("schema_version")
    if schema_version not in {
        STAGE052_LEGACY_RESOURCE_SCHEMA_VERSION,
        STAGE052_RESOURCE_SCHEMA_VERSION,
    }:
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
        or normalized_values["peak_active_cores"] < normalized_values["mean_active_cores"]
    ):
        return False, "resource timing, RSS, or active-core values are invalid", ()
    sample_count = resource_summary.get("sample_count")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 2:
        return False, "resource sample_count is invalid", ()
    parent_pid = resource_summary.get("parent_pid")
    descendants = resource_summary.get("descendant_pids")
    if not isinstance(parent_pid, int) or isinstance(parent_pid, bool) or parent_pid <= 0:
        return False, "resource parent_pid is invalid", ()
    if not isinstance(descendants, list) or any(
        not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 for pid in descendants
    ):
        return False, "resource descendant_pids are invalid", ()
    if len(descendants) != len(set(descendants)) or parent_pid in descendants:
        return False, "resource process identities are duplicate", ()
    if schema_version == STAGE052_RESOURCE_SCHEMA_VERSION:
        process_peaks = resource_summary.get("process_peak_rss_bytes")
        if not isinstance(process_peaks, Mapping):
            return False, "resource process_peak_rss_bytes is invalid", ()
        try:
            normalized_peaks = {int(str(pid)): int(peak) for pid, peak in process_peaks.items()}
        except (TypeError, ValueError):
            return False, "resource process_peak_rss_bytes is invalid", ()
        if set(normalized_peaks) != {parent_pid, *descendants} or any(
            peak <= 0 for peak in normalized_peaks.values()
        ):
            return False, "resource process peak identities or values are invalid", ()
        load_values = tuple(
            resource_summary.get(field) for field in ("load1_min", "load1_mean", "load1_max")
        )
        normalized_load_values: list[float] = []
        for value in load_values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                return False, "resource load1 summary is invalid", ()
            normalized_load_values.append(float(value))
        if not (
            normalized_load_values[0] <= normalized_load_values[1] <= normalized_load_values[2]
        ):
            return False, "resource load1 summary is invalid", ()
        if resource_summary.get("load1_sample_count") != sample_count:
            return False, "resource load1 sample count does not reconcile", ()
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
        "instance_sha256": {name: _sha256(path) for name, path in sorted(instance_paths.items())},
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
    if os.uname().sysname == "Linux" and "microsoft" in os.uname().release.casefold():
        status = read_windows_wsl_power_status()
        return {
            "available": True,
            "source": "AC Power" if status.ac_online else "Battery Power",
            "low_power_mode": status.battery_saver,
            "battery_life_percent": status.battery_life_percent,
            "battery_flag": status.battery_flag,
            "active_power_scheme": status.active_power_scheme,
        }
    if os.uname().sysname != "Darwin":
        raise RuntimeError("Stage 5.2 performance provenance requires WSL2 or macOS replay")
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
    command = arguments
    decode_as_utf16 = Path(arguments[0]).name.casefold() == "wsl.exe"
    if Path(arguments[0]).name.casefold() == "powershell.exe":
        if len(arguments) < 2:
            raise RuntimeError("PowerShell command is missing its script argument")
        prefix = (
            "$utf8=[System.Text.UTF8Encoding]::new($false);"
            "[Console]::OutputEncoding=$utf8;$OutputEncoding=$utf8;"
        )
        command = (*arguments[:-1], prefix + arguments[-1])
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"performance provenance command failed: {arguments}") from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"performance provenance command returned {completed.returncode}: {arguments}"
        )
    try:
        output = completed.stdout.decode(
            "utf-16-le" if decode_as_utf16 else "utf-8"
        ).lstrip("\ufeff").strip()
    except UnicodeDecodeError as error:
        encoding = "UTF-16LE" if decode_as_utf16 else "UTF-8"
        raise RuntimeError(
            f"performance provenance command output is not valid {encoding}: {arguments}"
        ) from error
    if not output:
        raise RuntimeError(f"performance provenance command returned no output: {arguments}")
    return output
