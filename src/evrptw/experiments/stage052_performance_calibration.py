"""Governed Stage 5.2 performance calibration and profile freezing.

The CLI obtains the top-level lifecycle permit before running telemetry A/B and
fixed-work child observations from strict no-cache wheels.  The pure selector
also remains available for tests and independent replay.  Neither path starts
Formal, CUDA, or attempt08, and neither changes host-wide settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path, PurePosixPath
from typing import Final, cast

from evrptw.experiments.stage052_telemetry_overhead import (
    TELEMETRY_SAMPLE_SCHEMA_VERSION,
    TelemetryWorkloadSample,
    load_telemetry_overhead_receipt,
    measure_representative_telemetry_overhead,
    write_telemetry_overhead_receipt,
)
from evrptw.stage052_atomic import publish_no_replace
from evrptw.stage052_continuity_lease import require_owned
from evrptw.stage052_performance import (
    BuildArtifactIdentity,
    BuildCandidate,
    ExecutionTopology,
    FrozenPerformanceProfile,
    HostPerformanceEnvelope,
    MemoryAdmission,
    TelemetryOverheadReceipt,
    check_memory_admission,
    detect_host_performance,
    freeze_performance_profile,
    require_clean_repository_root,
)
from evrptw.storage_governance import (
    StartPermit,
    preflight_cli_attempt,
    seal_cli_attempt,
    seal_failed_cli_attempt,
)

CALIBRATION_SCHEMA_VERSION: Final = "stage05.2-native-architecture-performance-calibration-v1"
WHEEL_RECEIPT_SCHEMA_VERSION: Final = "stage05.2-native-architecture-wheel-receipt-v2"
OBSERVATION_SCHEMA_VERSION: Final = "stage05.2-native-architecture-fixed-work-observation-v3"
OBSERVATION_PRODUCER_SCHEMA_VERSION: Final = "stage05.2-native-architecture-observation-producer-v3"
CALIBRATION_RECEIPT_SCHEMA_VERSION: Final = "stage05.2-native-architecture-calibration-receipt-v1"
REQUIRED_BUILD_PROFILES: Final = ("portable-o3", "portable-lto")
OPTIONAL_BUILD_PROFILES: Final = ("host-native-lto",)
SUPPORTED_BUILD_PROFILES: Final = REQUIRED_BUILD_PROFILES + OPTIONAL_BUILD_PROFILES
MODE_NAMES: Final = (
    "current_stage052",
    "python_candidate_control",
    "per_solve_runtime",
    "full_native_alns",
    "host_scheduler",
)
WORKLOAD_CLASSES: Final = ("c5", "100-customer")
REPEAT_COUNT: Final = 3
CALIBRATION_INPUT_STORAGE_ALIAS: Final = "stage052-performance-calibration-inputs"
CALIBRATION_INPUT_DIRECTORY: Final = "calibration_inputs"
PERFORMANCE_BUILD_STORAGE_ALIAS: Final = "stage052-performance-build"
_GIT_SHA1_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_RUN_LABEL_RE: Final = re.compile(
    r"^stage05\.2_native_architecture_performance_calibration_(?:attempt|rerun)[0-9]{2}$"
)
_LIFECYCLE_SAFETY_FIELDS: Final = frozenset(
    {
        "session_isolation_passed",
        "cache_reset_passed",
        "rss_stability_passed",
        "semantic_replay_passed",
    }
)
_LIFECYCLE_BOOLEAN_FIELDS: Final = _LIFECYCLE_SAFETY_FIELDS | {"mode_block_faster"}
_LIFECYCLE_SECONDS_FIELDS: Final = {
    "per_wave_end_to_end_seconds",
    "mode_block_end_to_end_seconds",
}
_LIFECYCLE_BYTES_FIELDS: Final = {
    "per_wave_scheduler_pss_peak_bytes",
    "mode_block_scheduler_pss_peak_bytes",
    "per_wave_process_tree_pss_peak_bytes",
    "mode_block_process_tree_pss_peak_bytes",
}
_LIFECYCLE_FIELDS: Final = (
    _LIFECYCLE_BOOLEAN_FIELDS
    | _LIFECYCLE_SECONDS_FIELDS
    | _LIFECYCLE_BYTES_FIELDS
    | {"wave_count", "build_profile", "topology_id", "topology"}
)


class CalibrationError(RuntimeError):
    """Input evidence or a selection gate is invalid."""


def _strict(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise CalibrationError(
            f"{name} fields mismatch (missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)})"
        )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CalibrationError(f"{name} must be a JSON object")
    raw = cast(Mapping[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise CalibrationError(f"{name} keys must be strings")
    result = {cast(str, key): item for key, item in raw.items()}
    _canonical(result, name=name)
    return result


def _canonical(value: object, *, name: str = "JSON value") -> bytes:
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"{name} is not finite canonical JSON") from error


def _digest(value: object, *, name: str = "JSON value") -> str:
    return hashlib.sha256(_canonical(value, name=name)).hexdigest()


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CalibrationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _git_sha1(value: object, name: str) -> str:
    if not isinstance(value, str) or _GIT_SHA1_RE.fullmatch(value) is None:
        raise CalibrationError(f"{name} must be a lowercase 40-character Git SHA-1")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationError(f"{name} must be a non-empty string")
    return value


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CalibrationError(f"{name} must be a positive integer")
    return value


def _nonnegative(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationError(f"{name} must be a non-negative integer")
    return value


def _seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CalibrationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise CalibrationError(f"{name} must be finite and positive")
    return result


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise CalibrationError(f"{name} must be an array of non-empty strings")
    return tuple(value)


def _json_file(path: Path) -> tuple[dict[str, object], str]:
    sidecar = Path(f"{path}.sha256")
    try:
        encoded = path.read_bytes()
        declared = sidecar.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationError(f"cannot read signed JSON: {path}") from error
    observed = hashlib.sha256(encoded).hexdigest()
    if _SHA256_RE.fullmatch(declared) is None or declared != observed:
        raise CalibrationError(f"signed JSON SHA-256 mismatch: {path}")
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationError(f"cannot read strict JSON: {path}") from error
    if not isinstance(payload, dict):
        raise CalibrationError(f"JSON root must be an object: {path}")
    return cast(dict[str, object], payload), observed


def _file_hash(path: Path, name: str) -> str:
    if not path.is_file():
        raise CalibrationError(f"{name} is not a regular file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CalibrationError(f"cannot hash {name}: {path}") from error
    return digest.hexdigest()


def _relative_path(raw: object, receipt: Path, name: str) -> Path:
    path = Path(_text(raw, name))
    return (path if path.is_absolute() else receipt.parent / path).resolve()


def _portable_artifact_path(raw: object, receipt: Path, name: str) -> Path:
    value = _text(raw, name)
    portable = PurePosixPath(value)
    if portable.is_absolute() or ".." in portable.parts or "\\" in value:
        raise CalibrationError(f"{name} must be a portable receipt-relative path")
    resolved = (receipt.parent / Path(*portable.parts)).resolve()
    try:
        resolved.relative_to(receipt.parent.resolve())
    except ValueError as error:  # pragma: no cover - guarded by PurePosixPath
        raise CalibrationError(f"{name} escapes its build receipt") from error
    return resolved


def _artifact_relative_path(path: Path, receipt: Path, name: str) -> str:
    try:
        relative = path.resolve().relative_to(receipt.parent.resolve()).as_posix()
    except ValueError as error:
        raise CalibrationError(f"{name} is outside its portable build receipt root") from error
    if not relative or ".." in PurePosixPath(relative).parts:
        raise CalibrationError(f"{name} is not a portable receipt-relative path")
    return relative


def _flags(profile: str, value: Sequence[str]) -> tuple[str, ...]:
    flags = tuple(value)
    lower = tuple(item.casefold() for item in flags)
    if any("fast-math" in item or "ffast-math" in item for item in lower):
        raise CalibrationError("fast-math is forbidden")
    required = {
        "portable-o3": ("-o3",),
        "portable-lto": ("-o3", "-flto"),
        "host-native-lto": ("-o3", "-flto", "-march=native"),
    }[profile]
    missing = [item for item in required if item not in lower]
    if missing:
        raise CalibrationError(f"{profile} is missing required flags {missing}")
    if profile != "host-native-lto" and any(
        item.startswith(("-march=", "-mcpu=", "-mtune=")) for item in lower
    ):
        raise CalibrationError(f"{profile} contains host-specific compiler flags")
    if profile == "portable-o3" and "-flto" in lower:
        raise CalibrationError("portable-o3 cannot use LTO")
    return flags


def _lifecycle(value: object, *, allow_empty: bool = True) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise CalibrationError("host_scheduler_lifecycle_evidence must be an object")
    raw = dict(cast(Mapping[str, object], value))
    if not raw and allow_empty:
        return {}
    if set(raw) != set(WORKLOAD_CLASSES):
        raise CalibrationError("lifecycle evidence must contain c5 and 100-customer")
    result: dict[str, object] = {}
    for workload in WORKLOAD_CLASSES:
        raw_values = raw[workload]
        if not isinstance(raw_values, list) or not raw_values:
            raise CalibrationError(f"lifecycle evidence {workload} must be a non-empty array")
        normalized: list[dict[str, object]] = []
        for index, raw_evidence in enumerate(raw_values):
            evidence = _mapping(
                raw_evidence,
                f"lifecycle evidence {workload}[{index}]",
            )
            _strict(
                evidence,
                set(_LIFECYCLE_FIELDS),
                f"lifecycle evidence {workload}[{index}]",
            )
            if any(not isinstance(evidence[item], bool) for item in _LIFECYCLE_BOOLEAN_FIELDS):
                raise CalibrationError(f"lifecycle evidence {workload} booleans are invalid")
            seconds = {
                item: _seconds(evidence[item], f"lifecycle evidence {workload} {item}")
                for item in _LIFECYCLE_SECONDS_FIELDS
            }
            byte_counts = {
                item: _positive(evidence[item], f"lifecycle evidence {workload} {item}")
                for item in _LIFECYCLE_BYTES_FIELDS
            }
            wave_count = _nonnegative(evidence["wave_count"], "lifecycle wave_count")
            if wave_count != 2:
                raise CalibrationError(
                    "scheduler lifecycle comparison must contain exactly two waves"
                )
            mode_block_faster = (
                seconds["mode_block_end_to_end_seconds"] < seconds["per_wave_end_to_end_seconds"]
            )
            if evidence["mode_block_faster"] is not mode_block_faster:
                raise CalibrationError("scheduler lifecycle timing verdict diverged")
            build_profile = _text(evidence["build_profile"], "lifecycle build_profile")
            if build_profile not in SUPPORTED_BUILD_PROFILES:
                raise CalibrationError("lifecycle build profile is unsupported")
            topology_id = _text(evidence["topology_id"], "lifecycle topology_id")
            try:
                topology = ExecutionTopology.from_dict(
                    _mapping(evidence["topology"], "lifecycle topology")
                )
            except ValueError as error:
                raise CalibrationError("lifecycle topology is invalid") from error
            if topology.workload_class != workload:
                raise CalibrationError("lifecycle topology workload differs")
            normalized.append(
                {
                    **{item: bool(evidence[item]) for item in sorted(_LIFECYCLE_BOOLEAN_FIELDS)},
                    **seconds,
                    **byte_counts,
                    "wave_count": wave_count,
                    "build_profile": build_profile,
                    "topology_id": topology_id,
                    "topology": topology.to_dict(),
                }
            )
        if len({cast(str, item["topology_id"]) for item in normalized}) != len(normalized):
            raise CalibrationError("lifecycle topology evidence is duplicated")
        result[workload] = normalized
    return result


def _observation_build_identity(value: Mapping[str, object]) -> dict[str, object]:
    """Validate an optional row identity; an omitted identity is legacy-safe."""

    if not isinstance(value, Mapping):
        raise CalibrationError("observation build_identity must be an object")
    raw = dict(value)
    if not raw:
        return {}
    expected = {
        "schema_version",
        "git_revision",
        "git_tree",
        "source_manifest_sha256",
        "wheel_sha256",
        "native_sha256",
        "scheduler_sha256",
        "compiler_version",
        "flags",
        "cpu_feature_mask",
    }
    _strict(raw, expected, "observation build_identity")
    if not isinstance(raw["schema_version"], str):
        raise CalibrationError("observation build_identity schema_version must be a string")
    _git_sha1(raw["git_revision"], "observation build_identity git_revision")
    _git_sha1(raw["git_tree"], "observation build_identity git_tree")
    _sha256(raw["source_manifest_sha256"], "observation source_manifest_sha256")
    _sha256(raw["wheel_sha256"], "observation wheel_sha256")
    _sha256(raw["native_sha256"], "observation native_sha256")
    _sha256(raw["scheduler_sha256"], "observation scheduler_sha256")
    _text(raw["compiler_version"], "observation compiler_version")
    _flags_from_identity = _strings(raw["flags"], "observation flags")
    if any(
        "fast-math" in item.casefold() or "ffast-math" in item.casefold()
        for item in _flags_from_identity
    ):
        raise CalibrationError("fast-math is forbidden in observation build identity")
    _strings(raw["cpu_feature_mask"], "observation cpu_feature_mask")
    return raw


@dataclass(frozen=True, slots=True)
class WheelReceipt:
    """One no-cache wheel/native/scheduler receipt."""

    build_profile: str
    artifact_identity: BuildArtifactIdentity
    wheel_path: Path
    native_path: Path
    scheduler_path: Path
    compiler_id: str = ""
    no_cache_build: bool = True
    source_dirty: bool = False
    development_override: bool = False
    source_receipt_path: Path | None = None
    source_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.build_profile not in SUPPORTED_BUILD_PROFILES:
            raise CalibrationError(f"unsupported build profile {self.build_profile}")
        if not self.no_cache_build:
            raise CalibrationError("receipt is not a no-cache build")
        if self.source_dirty or self.development_override:
            raise CalibrationError("receipt is not a clean commit build")
        _text(self.compiler_id, "compiler_id")
        _flags(self.build_profile, self.artifact_identity.flags)
        if self.source_receipt_sha256 is not None and self.source_receipt_path is None:
            raise CalibrationError("wheel receipt source hash requires its receipt path")
        if self.source_receipt_sha256 is not None:
            _sha256(self.source_receipt_sha256, "source_receipt_sha256")

    def to_dict(self) -> dict[str, object]:
        if self.source_receipt_path is None:
            raise CalibrationError("portable wheel receipt requires its destination path")
        return {
            "schema_version": WHEEL_RECEIPT_SCHEMA_VERSION,
            "build_profile": self.build_profile,
            "no_cache_build": self.no_cache_build,
            "source_dirty": self.source_dirty,
            "development_override": self.development_override,
            "storage_alias": PERFORMANCE_BUILD_STORAGE_ALIAS,
            "wheel_relative_path": _artifact_relative_path(
                self.wheel_path, self.source_receipt_path, "wheel_path"
            ),
            "native_relative_path": _artifact_relative_path(
                self.native_path, self.source_receipt_path, "native_path"
            ),
            "scheduler_relative_path": _artifact_relative_path(
                self.scheduler_path, self.source_receipt_path, "scheduler_path"
            ),
            "compiler_id": self.compiler_id,
            "artifact_identity": self.artifact_identity.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        source_path: Path,
        source_sha256: str | None = None,
    ) -> WheelReceipt:
        expected = {
            "schema_version",
            "build_profile",
            "no_cache_build",
            "source_dirty",
            "development_override",
            "storage_alias",
            "wheel_relative_path",
            "native_relative_path",
            "scheduler_relative_path",
            "artifact_identity",
            "compiler_id",
        }
        _strict(payload, expected, "wheel_receipt")
        if payload["schema_version"] != WHEEL_RECEIPT_SCHEMA_VERSION:
            raise CalibrationError("unsupported wheel receipt schema")
        if payload["storage_alias"] != PERFORMANCE_BUILD_STORAGE_ALIAS:
            raise CalibrationError("wheel receipt storage alias is invalid")
        for name in ("no_cache_build", "source_dirty", "development_override"):
            if not isinstance(payload[name], bool):
                raise CalibrationError(f"{name} must be boolean")
        try:
            identity = BuildArtifactIdentity.from_dict(
                _mapping(payload["artifact_identity"], "artifact_identity")
            )
        except ValueError as error:
            raise CalibrationError("invalid wheel artifact identity") from error
        return cls(
            build_profile=_text(payload["build_profile"], "build_profile"),
            artifact_identity=identity,
            wheel_path=_portable_artifact_path(
                payload["wheel_relative_path"], source_path, "wheel_relative_path"
            ),
            native_path=_portable_artifact_path(
                payload["native_relative_path"], source_path, "native_relative_path"
            ),
            scheduler_path=_portable_artifact_path(
                payload["scheduler_relative_path"], source_path, "scheduler_relative_path"
            ),
            compiler_id=_text(payload["compiler_id"], "compiler_id"),
            no_cache_build=cast(bool, payload["no_cache_build"]),
            source_dirty=cast(bool, payload["source_dirty"]),
            development_override=cast(bool, payload["development_override"]),
            source_receipt_path=source_path.resolve() if source_sha256 is not None else None,
            source_receipt_sha256=source_sha256,
        )

    def verify_files(self) -> None:
        for path, expected, name in (
            (self.wheel_path, self.artifact_identity.wheel_sha256, "wheel"),
            (self.native_path, self.artifact_identity.native_sha256, "native"),
            (self.scheduler_path, self.artifact_identity.scheduler_sha256, "scheduler"),
        ):
            observed = _file_hash(path, name)
            if observed != expected:
                raise CalibrationError(
                    f"{name} SHA-256 mismatch for {self.build_profile}: {observed} != {expected}"
                )


def load_wheel_receipt(path: Path) -> WheelReceipt:
    """Load and attest one signed wheel receipt without imposing a build matrix."""

    resolved = path.resolve()
    payload, digest = _json_file(resolved)
    receipt = WheelReceipt.from_dict(
        payload,
        source_path=resolved,
        source_sha256=digest,
    )
    receipt.verify_files()
    return receipt


def load_wheel_receipts(paths: Sequence[Path]) -> tuple[WheelReceipt, ...]:
    resolved = [path.resolve() for path in paths]
    if not paths or len(resolved) != len(set(resolved)):
        raise CalibrationError("wheel receipt paths must be non-empty and unique")
    receipts = tuple(load_wheel_receipt(path) for path in resolved)
    profiles = {item.build_profile for item in receipts}
    if not set(REQUIRED_BUILD_PROFILES).issubset(profiles):
        raise CalibrationError("portable-o3 and portable-lto wheel receipts are required")
    if not profiles.issubset(SUPPORTED_BUILD_PROFILES) or len(receipts) != len(profiles):
        raise CalibrationError("wheel receipt build profile set is invalid")
    for receipt in receipts:
        receipt.verify_files()
    identity = {
        (
            item.artifact_identity.git_revision,
            item.artifact_identity.git_tree,
            item.artifact_identity.source_manifest_sha256,
        )
        for item in receipts
    }
    if len(identity) != 1:
        raise CalibrationError("wheel receipts are not bound to one clean Git tree")
    return tuple(sorted(receipts, key=lambda item: item.build_profile))


@dataclass(frozen=True, slots=True)
class AxisObservation:
    build_profile: str
    repeat: int
    mode: str
    workload_class: str
    instance: str
    seed: int
    topology_id: str
    topology: ExecutionTopology
    warm_start: object
    producer_end_to_end_seconds: float
    independent_replay_seconds: float
    end_to_end_seconds: float
    # PSS of one isolated client axis. The selector projects this value over
    # the frozen shard count before memory admission.
    pss_bytes: int
    # Shared scheduler PSS included in pss_bytes for host-scheduler axes.
    # It is counted once rather than once per client.
    scheduler_pss_bytes: int
    objective: object
    routes: object
    candidate_trajectory: object
    exact_order: object
    cache_lifecycle: object
    transaction_hashes: tuple[str, ...]
    confidence_interval: tuple[float, float] | None = None
    lifecycle_evidence: Mapping[str, object] = field(default_factory=dict)
    build_identity: Mapping[str, object] = field(default_factory=dict)
    fixed_work_budget: object = field(default_factory=dict)
    source_observation_path: Path | None = None
    source_observation_sha256: str | None = None
    semantic_digest: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.build_profile not in SUPPORTED_BUILD_PROFILES:
            raise CalibrationError(f"unsupported observation build profile {self.build_profile}")
        if self.repeat not in range(REPEAT_COUNT):
            raise CalibrationError("repeat must be 0, 1, or 2")
        if self.mode not in MODE_NAMES or self.workload_class not in WORKLOAD_CLASSES:
            raise CalibrationError("unsupported mode or workload class")
        _text(self.instance, "instance")
        _nonnegative(self.seed, "seed")
        _text(self.topology_id, "topology_id")
        producer_seconds = _seconds(
            self.producer_end_to_end_seconds,
            "producer_end_to_end_seconds",
        )
        replay_seconds = _seconds(
            self.independent_replay_seconds,
            "independent_replay_seconds",
        )
        end_to_end_seconds = _seconds(self.end_to_end_seconds, "end_to_end_seconds")
        if not math.isclose(
            producer_seconds + replay_seconds,
            end_to_end_seconds,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise CalibrationError("axis end-to-end timing does not reconcile")
        _positive(self.pss_bytes, "pss_bytes")
        _nonnegative(self.scheduler_pss_bytes, "scheduler_pss_bytes")
        if self.scheduler_pss_bytes > self.pss_bytes:
            raise CalibrationError("scheduler_pss_bytes exceeds isolated axis PSS")
        if self.mode != "host_scheduler" and self.scheduler_pss_bytes != 0:
            raise CalibrationError("non-host axis cannot report scheduler PSS")
        for name, value in (
            ("warm_start", self.warm_start),
            ("objective", self.objective),
            ("routes", self.routes),
            ("candidate_trajectory", self.candidate_trajectory),
            ("exact_order", self.exact_order),
            ("cache_lifecycle", self.cache_lifecycle),
            ("fixed_work_budget", self.fixed_work_budget),
        ):
            _canonical(value, name=name)
        if not self.transaction_hashes:
            raise CalibrationError("transaction_hashes must not be empty")
        for index, item in enumerate(self.transaction_hashes):
            _sha256(item, f"transaction_hashes[{index}]")
        if self.confidence_interval is not None:
            low, high = self.confidence_interval
            if not math.isfinite(low) or not math.isfinite(high) or low <= 0 or high < low:
                raise CalibrationError("confidence_interval is invalid")
        lifecycle = _lifecycle(self.lifecycle_evidence)
        object.__setattr__(self, "lifecycle_evidence", lifecycle)
        identity = _observation_build_identity(self.build_identity)
        object.__setattr__(self, "build_identity", identity)
        if (self.source_observation_path is None) != (self.source_observation_sha256 is None):
            raise CalibrationError("observation source path/hash must be provided together")
        if self.source_observation_sha256 is not None:
            _sha256(self.source_observation_sha256, "source_observation_sha256")
        semantic = {
            "mode": self.mode,
            "workload_class": self.workload_class,
            "instance": self.instance,
            "seed": self.seed,
            "topology_id": self.topology_id,
            "objective": self.objective,
            "routes": self.routes,
            "candidate_trajectory": self.candidate_trajectory,
            "exact_order": self.exact_order,
            "cache_lifecycle": self.cache_lifecycle,
            "transaction_hashes": list(self.transaction_hashes),
        }
        object.__setattr__(self, "semantic_digest", _digest(semantic, name="semantic payload"))

    @property
    def axis_key(self) -> tuple[str, str, str, int, str]:
        return (self.mode, self.workload_class, self.instance, self.seed, self.topology_id)

    @property
    def base_axis_key(self) -> tuple[str, str, str, int]:
        return (self.mode, self.workload_class, self.instance, self.seed)

    @property
    def mode_block(self) -> str:
        return f"{self.mode}:{self.workload_class}"

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "workload_class": self.workload_class,
            "instance": self.instance,
            "seed": self.seed,
            "topology_id": self.topology_id,
            "topology": self.topology.to_dict(),
            "warm_start": self.warm_start,
            "producer_end_to_end_seconds": self.producer_end_to_end_seconds,
            "independent_replay_seconds": self.independent_replay_seconds,
            "end_to_end_seconds": self.end_to_end_seconds,
            "pss_bytes": self.pss_bytes,
            "scheduler_pss_bytes": self.scheduler_pss_bytes,
            "objective": self.objective,
            "routes": self.routes,
            "candidate_trajectory": self.candidate_trajectory,
            "exact_order": self.exact_order,
            "cache_lifecycle": self.cache_lifecycle,
            "transaction_hashes": list(self.transaction_hashes),
            "confidence_interval": None
            if self.confidence_interval is None
            else list(self.confidence_interval),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        build_profile: str,
        repeat: int,
        warm_start: object,
        lifecycle_evidence: Mapping[str, object] | None = None,
        build_identity: Mapping[str, object] | None = None,
        fixed_work_budget: object | None = None,
        source_observation_path: Path | None = None,
        source_observation_sha256: str | None = None,
    ) -> AxisObservation:
        expected = {
            "mode",
            "workload_class",
            "instance",
            "seed",
            "topology_id",
            "topology",
            "producer_end_to_end_seconds",
            "independent_replay_seconds",
            "end_to_end_seconds",
            "pss_bytes",
            "scheduler_pss_bytes",
            "objective",
            "routes",
            "candidate_trajectory",
            "exact_order",
            "cache_lifecycle",
            "transaction_hashes",
            "confidence_interval",
        }
        _strict(payload, expected, "axis_observation")
        if not isinstance(payload["mode"], str) or not isinstance(payload["workload_class"], str):
            raise CalibrationError("axis mode/workload must be strings")
        if not isinstance(payload["instance"], str):
            raise CalibrationError("axis instance must be a string")
        try:
            topology = ExecutionTopology.from_dict(_mapping(payload["topology"], "topology"))
        except ValueError as error:
            raise CalibrationError("invalid execution topology") from error
        if topology.workload_class != payload["workload_class"]:
            raise CalibrationError("topology workload class differs from axis")
        transactions = payload["transaction_hashes"]
        if not isinstance(transactions, list):
            raise CalibrationError("transaction_hashes must be an array")
        interval_raw = payload["confidence_interval"]
        interval: tuple[float, float] | None
        if interval_raw is None:
            interval = None
        elif (
            isinstance(interval_raw, list)
            and len(interval_raw) == 2
            and all(
                not isinstance(item, bool) and isinstance(item, int | float)
                for item in interval_raw
            )
        ):
            interval = (float(interval_raw[0]), float(interval_raw[1]))
        else:
            raise CalibrationError("confidence_interval must be null or [low, high]")
        return cls(
            build_profile=build_profile,
            repeat=repeat,
            mode=payload["mode"],
            workload_class=payload["workload_class"],
            instance=payload["instance"],
            seed=_nonnegative(payload["seed"], "seed"),
            topology_id=_text(payload["topology_id"], "topology_id"),
            topology=topology,
            warm_start=warm_start,
            producer_end_to_end_seconds=_seconds(
                payload["producer_end_to_end_seconds"],
                "producer_end_to_end_seconds",
            ),
            independent_replay_seconds=_seconds(
                payload["independent_replay_seconds"],
                "independent_replay_seconds",
            ),
            end_to_end_seconds=_seconds(payload["end_to_end_seconds"], "end_to_end_seconds"),
            pss_bytes=_positive(payload["pss_bytes"], "pss_bytes"),
            scheduler_pss_bytes=_nonnegative(
                payload["scheduler_pss_bytes"],
                "scheduler_pss_bytes",
            ),
            objective=payload["objective"],
            routes=payload["routes"],
            candidate_trajectory=payload["candidate_trajectory"],
            exact_order=payload["exact_order"],
            cache_lifecycle=payload["cache_lifecycle"],
            transaction_hashes=tuple(
                _sha256(item, f"transaction_hashes[{index}]")
                for index, item in enumerate(transactions)
            ),
            confidence_interval=interval,
            lifecycle_evidence={} if lifecycle_evidence is None else lifecycle_evidence,
            build_identity={} if build_identity is None else build_identity,
            fixed_work_budget={} if fixed_work_budget is None else fixed_work_budget,
            source_observation_path=source_observation_path,
            source_observation_sha256=source_observation_sha256,
        )


def _verified_observation_provenance(
    value: object,
    *,
    observation_path: Path,
) -> dict[str, object]:
    provenance = _mapping(value, "observation producer provenance")
    _strict(
        provenance,
        {
            "schema_version",
            "repository_revision",
            "producer_source_sha256",
            "parent_run_label",
            "storage_alias",
            "start_permit_relative_path",
            "start_permit_sha256",
            "host_envelope_relative_path",
            "host_envelope_sha256",
            "raw_axis_inventory",
            "scheduler_task_receipt_inventory",
            "resource_summaries",
        },
        "observation producer provenance",
    )
    if provenance["schema_version"] != OBSERVATION_PRODUCER_SCHEMA_VERSION:
        raise CalibrationError("unsupported observation producer provenance schema")
    _git_sha1(provenance["repository_revision"], "observation repository_revision")
    _sha256(provenance["producer_source_sha256"], "observation producer_source_sha256")
    parent_run_label = _text(provenance["parent_run_label"], "observation parent_run_label")
    if _RUN_LABEL_RE.fullmatch(parent_run_label) is None:
        raise CalibrationError("observation parent run label is not canonical")
    if provenance["storage_alias"] != "stage052-performance-calibration-run":
        raise CalibrationError("observation storage alias is invalid")
    run_roots = [
        parent for parent in observation_path.resolve().parents if parent.name == parent_run_label
    ]
    if len(run_roots) != 1:
        raise CalibrationError("observation calibration-run alias is ambiguous")
    run_root = run_roots[0]

    def run_reference(value: object, field: str) -> Path:
        raw = _text(value, field)
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts or "\\" in raw:
            raise CalibrationError(f"{field} is not a portable relative path")
        resolved = (run_root / Path(*relative.parts)).resolve()
        try:
            resolved.relative_to(run_root)
        except ValueError as error:
            raise CalibrationError(f"{field} escapes the calibration run") from error
        return resolved

    provenance_sources: dict[str, Path] = {}
    for prefix in ("start_permit", "host_envelope"):
        source = run_reference(
            provenance[f"{prefix}_relative_path"],
            f"observation {prefix} relative path",
        )
        expected_source = _sha256(
            provenance[f"{prefix}_sha256"],
            f"observation {prefix} sha256",
        )
        if _file_hash(source, f"observation {prefix}") != expected_source:
            raise CalibrationError(f"observation {prefix} SHA-256 mismatch")
        sidecar = Path(f"{source}.sha256")
        try:
            declared = sidecar.read_text(encoding="ascii").strip().split()[0]
        except (OSError, UnicodeError, IndexError) as error:
            raise CalibrationError(f"observation {prefix} sidecar is invalid") from error
        if declared != expected_source:
            raise CalibrationError(f"observation {prefix} sidecar does not bind its artifact")
        provenance_sources[prefix] = source
    try:
        permit_payload = json.loads(provenance_sources["start_permit"].read_bytes())
        host_payload = json.loads(provenance_sources["host_envelope"].read_bytes())
        if (
            not isinstance(permit_payload, dict)
            or permit_payload.get("run_label") != parent_run_label
            or permit_payload.get("status") != "reserved"
        ):
            raise ValueError("start permit identity mismatch")
        HostPerformanceEnvelope.from_dict(_mapping(host_payload, "observation host envelope"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CalibrationError("observation lifecycle/host provenance is invalid") from error
    inventory = provenance["raw_axis_inventory"]
    if not isinstance(inventory, list) or not inventory:
        raise CalibrationError("observation raw_axis_inventory must be non-empty")
    for index, raw_entry in enumerate(inventory):
        entry = _mapping(raw_entry, f"raw_axis_inventory[{index}]")
        _strict(
            entry,
            {
                "storage_alias",
                "relative_path",
                "sha256",
                "relative_sidecar_path",
                "sidecar_sha256",
                "role",
                "supporting_artifacts",
            },
            f"raw_axis_inventory[{index}]",
        )
        if entry["storage_alias"] != "stage052-performance-calibration-run":
            raise CalibrationError("raw axis storage alias is invalid")
        path = run_reference(entry["relative_path"], "raw axis relative path")
        sidecar = run_reference(
            entry["relative_sidecar_path"],
            "raw axis sidecar relative path",
        )
        expected = _sha256(entry["sha256"], "raw axis sha256")
        sidecar_expected = _sha256(entry["sidecar_sha256"], "raw axis sidecar sha256")
        if _file_hash(path, "raw axis artifact") != expected:
            raise CalibrationError("raw axis artifact SHA-256 mismatch")
        if _file_hash(sidecar, "raw axis sidecar") != sidecar_expected:
            raise CalibrationError("raw axis sidecar file SHA-256 mismatch")
        try:
            declared = sidecar.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as error:
            raise CalibrationError("raw axis sidecar cannot be read") from error
        if declared != expected:
            raise CalibrationError("raw axis sidecar does not bind its artifact")
        _text(entry["role"], "raw axis role")
        supporting = entry["supporting_artifacts"]
        if not isinstance(supporting, list) or not supporting:
            raise CalibrationError("raw axis supporting-artifact inventory is empty")
        support_paths: set[Path] = set()
        support_roles: set[str] = set()
        for support_index, raw_support in enumerate(supporting):
            support = _mapping(
                raw_support,
                f"raw_axis_inventory[{index}].supporting_artifacts[{support_index}]",
            )
            _strict(
                support,
                {"relative_path", "sha256", "role"},
                "raw axis supporting artifact",
            )
            support_path = run_reference(
                support["relative_path"],
                "raw axis supporting artifact path",
            )
            support_sha = _sha256(
                support["sha256"],
                "raw axis supporting artifact sha256",
            )
            support_role = _text(support["role"], "raw axis supporting artifact role")
            if (
                support_path in support_paths
                or _file_hash(support_path, "raw axis supporting artifact") != support_sha
            ):
                raise CalibrationError("raw axis supporting-artifact inventory differs")
            support_paths.add(support_path)
            support_roles.add(support_role)
        if not {
            "axis-persistence-receipt",
            "axis-persistence-receipt-sidecar",
        }.issubset(support_roles):
            raise CalibrationError("raw axis persistence receipt is not fully inventoried")
    scheduler_inventory = provenance["scheduler_task_receipt_inventory"]
    if not isinstance(scheduler_inventory, list):
        raise CalibrationError("scheduler_task_receipt_inventory must be an array")
    for index, raw_entry in enumerate(scheduler_inventory):
        entry = _mapping(raw_entry, f"scheduler_task_receipt_inventory[{index}]")
        _strict(
            entry,
            {
                "storage_alias",
                "relative_path",
                "sha256",
                "relative_sidecar_path",
                "sidecar_sha256",
                "role",
            },
            f"scheduler_task_receipt_inventory[{index}]",
        )
        if entry["storage_alias"] != "stage052-performance-calibration-run":
            raise CalibrationError("scheduler task-receipt storage alias is invalid")
        path = run_reference(entry["relative_path"], "scheduler task-receipt path")
        sidecar = run_reference(
            entry["relative_sidecar_path"],
            "scheduler task-receipt sidecar path",
        )
        expected = _sha256(entry["sha256"], "scheduler task-receipt sha256")
        sidecar_expected = _sha256(entry["sidecar_sha256"], "scheduler task-receipt sidecar sha256")
        if (
            _file_hash(path, "scheduler task-receipt") != expected
            or _file_hash(sidecar, "scheduler task-receipt sidecar") != sidecar_expected
            or sidecar.read_text(encoding="ascii").strip() != expected
        ):
            raise CalibrationError("scheduler task-receipt inventory does not reconcile")
        _text(entry["role"], "scheduler task-receipt role")
    resources = provenance["resource_summaries"]
    if not isinstance(resources, list) or not resources:
        raise CalibrationError("observation resource_summaries must be non-empty")
    _canonical(resources, name="observation resource_summaries")
    return dict(provenance)


def _observation_payload(
    payload: Mapping[str, object],
    path: Path,
    source_sha256: str,
) -> tuple[AxisObservation, ...]:
    if payload.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise CalibrationError(f"unsupported observation schema: {path}")
    if "observations" in payload:
        _strict(
            payload,
            {
                "schema_version",
                "observations",
                "producer_provenance",
                "memory_rejections",
            },
            "observation_batch",
        )
        _verified_observation_provenance(payload["producer_provenance"], observation_path=path)
        if not isinstance(payload["memory_rejections"], list):
            raise CalibrationError("observation memory_rejections must be an array")
        _canonical(payload["memory_rejections"], name="observation memory_rejections")
        rows = payload["observations"]
        if not isinstance(rows, list) or not rows:
            raise CalibrationError("observation_batch.observations must be non-empty")
        result: list[AxisObservation] = []
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"observation_batch[{index}]")
            allowed_row = {
                "build_profile",
                "repeat",
                "warm_start",
                "axis",
                "host_scheduler_lifecycle_evidence",
                "build_identity",
                "fixed_work_budget",
            }
            _strict(row, allowed_row, f"observation_batch[{index}]")
            result.append(
                AxisObservation.from_dict(
                    _mapping(row["axis"], f"observation_batch[{index}].axis"),
                    build_profile=_text(row["build_profile"], "build_profile"),
                    repeat=_nonnegative(row["repeat"], "repeat"),
                    warm_start=row["warm_start"],
                    lifecycle_evidence=_lifecycle(row["host_scheduler_lifecycle_evidence"]),
                    build_identity=_observation_build_identity(
                        cast(
                            Mapping[str, object],
                            row["build_identity"],
                        )
                    ),
                    fixed_work_budget=row["fixed_work_budget"],
                    source_observation_path=path.resolve(),
                    source_observation_sha256=source_sha256,
                )
            )
        return tuple(result)
    _strict(
        payload,
        {
            "schema_version",
            "build_profile",
            "repeat",
            "warm_start",
            "axes",
            "host_scheduler_lifecycle_evidence",
            "build_identity",
            "fixed_work_budget",
            "producer_provenance",
            "memory_rejections",
        },
        "observation_envelope",
    )
    lifecycle = _lifecycle(payload["host_scheduler_lifecycle_evidence"])
    _verified_observation_provenance(payload["producer_provenance"], observation_path=path)
    if not isinstance(payload["memory_rejections"], list):
        raise CalibrationError("observation memory_rejections must be an array")
    _canonical(payload["memory_rejections"], name="observation memory_rejections")
    axes = payload["axes"]
    if not isinstance(axes, list) or not axes:
        raise CalibrationError("observation_envelope.axes must be non-empty")
    return tuple(
        AxisObservation.from_dict(
            _mapping(raw, f"observation_envelope.axes[{index}]"),
            build_profile=_text(payload["build_profile"], "build_profile"),
            repeat=_nonnegative(payload["repeat"], "repeat"),
            warm_start=payload["warm_start"],
            lifecycle_evidence=lifecycle,
            build_identity=_observation_build_identity(
                cast(Mapping[str, object], payload["build_identity"])
            ),
            fixed_work_budget=payload["fixed_work_budget"],
            source_observation_path=path.resolve(),
            source_observation_sha256=source_sha256,
        )
        for index, raw in enumerate(axes)
    )


def load_fixed_work_observations(paths: Sequence[Path]) -> tuple[AxisObservation, ...]:
    if not paths:
        raise CalibrationError("at least one observation file is required")
    result: list[AxisObservation] = []
    seen_paths: set[Path] = set()
    for raw in paths:
        path = raw.resolve()
        if path in seen_paths:
            raise CalibrationError(f"duplicate observation path: {path}")
        seen_paths.add(path)
        payload, digest = _json_file(path)
        result.extend(_observation_payload(payload, path, digest))
    seen: set[tuple[str, int, tuple[str, str, str, int, str]]] = set()
    for row in result:
        key = (row.build_profile, row.repeat, row.axis_key)
        if key in seen:
            raise CalibrationError(f"duplicate observation identity: {key!r}")
        seen.add(key)
    return tuple(sorted(result, key=lambda row: (row.build_profile, row.axis_key, row.repeat)))


def _candidate(receipt: WheelReceipt) -> BuildCandidate:
    profile = receipt.build_profile
    identity = receipt.artifact_identity
    return BuildCandidate(
        name=profile,
        flags=identity.flags,
        portable=profile != "host-native-lto",
        lto=profile.endswith("lto"),
        host_native=profile == "host-native-lto",
        compiler=receipt.compiler_id,
        cpu_features=identity.cpu_feature_mask,
        artifact_identity=identity,
    )


def _validate_identities(receipts: Sequence[WheelReceipt], rows: Sequence[AxisObservation]) -> None:
    commit_keys = {
        (
            item.artifact_identity.git_revision,
            item.artifact_identity.git_tree,
            item.artifact_identity.source_manifest_sha256,
        )
        for item in receipts
    }
    if len(commit_keys) != 1:
        raise CalibrationError("wheel receipts are not bound to one clean Git tree")
    by_build = {item.build_profile: item for item in receipts}
    observed_builds: set[str] = set()
    for row in rows:
        receipt = by_build.get(row.build_profile)
        if receipt is None:
            raise CalibrationError(
                f"observation references an unreceipted build: {row.build_profile}"
            )
        observed_builds.add(row.build_profile)
        identity = receipt.artifact_identity
        observed_identity = row.build_identity
        if observed_identity:
            expected_identity = identity.to_dict()
            if dict(observed_identity) != expected_identity:
                raise CalibrationError(
                    f"observation compiler/flags/CPU identity differs for {row.build_profile}"
                )
        if identity.git_revision not in {key[0] for key in commit_keys}:
            raise CalibrationError("observation build revision is not the clean commit")
    if observed_builds != set(by_build):
        raise CalibrationError(
            f"missing observations for builds: {sorted(set(by_build) - observed_builds)}"
        )


def _validate_matrix(
    rows: Sequence[AxisObservation],
) -> tuple[dict[str, set[tuple[str, str, str, int, str]]], dict[str, str]]:
    grouped: dict[str, dict[tuple[str, str, str, int, str], list[AxisObservation]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row.build_profile][row.axis_key].append(row)
    axis_sets = {build: set(item) for build, item in grouped.items()}
    if not axis_sets:
        raise CalibrationError("observation matrix is empty")
    expected = next(iter(axis_sets.values()))
    for build, axes in axis_sets.items():
        if axes != expected:
            raise CalibrationError(f"build {build} has a different topology/axis identity set")
        for axis in axes:
            entries = grouped[build][axis]
            if len(entries) != REPEAT_COUNT or {item.repeat for item in entries} != set(
                range(REPEAT_COUNT)
            ):
                raise CalibrationError(f"{build} axis {axis!r} must have exactly three repeats")
            if len({_digest(item.warm_start, name="warm_start") for item in entries}) != 1:
                raise CalibrationError(f"warm start changed across repeats for {axis!r}")
    warm_by_identity: dict[tuple[int, tuple[str, str, str, int, str]], str] = {}
    for row in rows:
        key = (row.repeat, row.axis_key)
        value = _digest(row.warm_start, name="warm_start")
        previous = warm_by_identity.setdefault(key, value)
        if previous != value:
            raise CalibrationError(f"warm start differs across builds for {key!r}")
    topology_by_identity: dict[tuple[int, tuple[str, str, str, int, str]], dict[str, object]] = {}
    for row in rows:
        key = (row.repeat, row.axis_key)
        topology_value = row.topology.to_dict()
        previous_topology = topology_by_identity.setdefault(key, topology_value)
        if previous_topology != topology_value:
            raise CalibrationError(f"topology identity differs across builds for {key!r}")
    budget_by_identity: dict[tuple[int, tuple[str, str, str, int, str]], str] = {}
    for row in rows:
        key = (row.repeat, row.axis_key)
        budget_digest = _digest(row.fixed_work_budget, name="fixed_work_budget")
        previous_budget = budget_by_identity.setdefault(key, budget_digest)
        if previous_budget != budget_digest:
            raise CalibrationError(f"fixed-work budget differs across builds for {key!r}")
    rejected: dict[str, str] = {}
    for build, axis_map in grouped.items():
        for axis in axis_map:
            if len({item.semantic_digest for item in grouped[build][axis]}) != 1:
                rejected.setdefault(
                    build,
                    f"semantic digest changed across repeats at {axis!r}",
                )
    viable_builds = [build for build in sorted(axis_sets) if build not in rejected]
    if not viable_builds:
        raise CalibrationError("every build changed semantic digest across repeats")
    # Prefer portable O3 as the deterministic semantic reference whenever it
    # remains viable.  This keeps comparisons fail-closed and reproducible.
    baseline = "portable-o3" if "portable-o3" in viable_builds else viable_builds[0]
    base = {
        (item.axis_key, item.repeat): item.semantic_digest
        for item in rows
        if item.build_profile == baseline
    }
    for build in sorted(axis_sets):
        if build in rejected:
            continue
        for row in rows:
            if (
                row.build_profile == build
                and row.semantic_digest != base[(row.axis_key, row.repeat)]
            ):
                rejected.setdefault(
                    build, f"semantic digest mismatch at {row.axis_key!r}, repeat {row.repeat}"
                )
    return axis_sets, rejected


def _stats(
    rows: Iterable[AxisObservation],
    host: HostPerformanceEnvelope,
) -> dict[str, dict[str, tuple[float, int, str, tuple[float, ...]]]]:
    grouped: dict[str, dict[str, dict[str, list[AxisObservation]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        grouped[row.build_profile][row.mode_block][row.topology_id].append(row)
    result: dict[str, dict[str, tuple[float, int, str, tuple[float, ...]]]] = {}
    for build, blocks in grouped.items():
        result[build] = {}
        for block, topologies in blocks.items():
            candidates: list[tuple[float, int, str, tuple[float, ...]]] = []
            for topology_id, values in topologies.items():
                topology = values[0].topology
                if any(item.topology.to_dict() != topology.to_dict() for item in values):
                    raise CalibrationError("build-selection topology identity changed")
                peak = max(item.pss_bytes for item in values)
                scheduler_peak = max(item.scheduler_pss_bytes for item in values)
                projected = scheduler_peak + (peak - scheduler_peak) * topology.shard_count
                admission = check_memory_admission(host, projected, headroom_fraction=0.20)
                if not admission.passed or projected > math.floor(
                    host.effective_memory_limit_bytes * 0.80
                ):
                    continue
                repeat_times = tuple(
                    sum(item.end_to_end_seconds for item in values if item.repeat == repeat)
                    for repeat in range(REPEAT_COUNT)
                )
                if any(value <= 0.0 for value in repeat_times):
                    raise CalibrationError("build-selection repeat timing matrix is incomplete")
                candidates.append(
                    (
                        statistics.median(repeat_times),
                        projected,
                        topology_id,
                        repeat_times,
                    )
                )
            if candidates:
                result[build][block] = min(
                    candidates,
                    key=lambda item: (item[0], item[1], item[2]),
                )
    return result


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise CalibrationError("cannot compute an empty confidence interval")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_bootstrap_median_ci(values: Sequence[float]) -> tuple[float, float]:
    """Exact deterministic percentile bootstrap over one paired repeat vector."""

    sample = tuple(values)
    if len(sample) != REPEAT_COUNT or any(not math.isfinite(value) for value in sample):
        raise CalibrationError("paired CI requires exactly three finite repeat values")
    medians = [
        statistics.median(sample[index] for index in indices)
        for indices in product(range(len(sample)), repeat=len(sample))
    ]
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def _select_build(
    candidates: Mapping[str, BuildCandidate],
    rows: Sequence[AxisObservation],
    rejected: Mapping[str, str],
    host: HostPerformanceEnvelope,
) -> tuple[BuildCandidate, dict[str, dict[str, float]], str, dict[str, object]]:
    stats = _stats(
        (item for item in rows if item.build_profile not in rejected),
        host,
    )
    blocks = {item.mode_block for item in rows}
    stats = {build: values for build, values in stats.items() if set(values) == blocks}
    if not stats:
        raise CalibrationError(
            "no semantically valid complete build with a memory-safe topology remains"
        )
    aggregate_times = {
        build: tuple(
            sum(value[3][repeat] for value in blocks.values()) for repeat in range(REPEAT_COUNT)
        )
        for build, blocks in stats.items()
    }
    scores = {build: statistics.median(values) for build, values in aggregate_times.items()}
    memories = {build: max(item[1] for item in values.values()) for build, values in stats.items()}
    fastest = min(scores, key=lambda build: (scores[build], build))
    paired_relative = {
        build: tuple(
            aggregate_times[build][repeat] / aggregate_times[fastest][repeat] - 1.0
            for repeat in range(REPEAT_COUNT)
        )
        for build in stats
    }
    paired_cis = {
        build: _paired_bootstrap_median_ci(values) for build, values in paired_relative.items()
    }

    eligible = [
        build
        for build in stats
        if scores[build] <= scores[fastest] * 1.03
        or paired_cis[build][0] <= 0.0 <= paired_cis[build][1]
    ]
    selected_name = min(
        eligible,
        key=lambda build: (
            not candidates[build].portable,
            candidates[build].host_native,
            memories[build],
            scores[build],
            build,
        ),
    )
    medians = {
        build: {block: value[0] for block, value in values.items()}
        for build, values in stats.items()
    }
    selection_statistics: dict[str, object] = {
        "method": "paired-exact-bootstrap-median-aggregate-e2e-v1",
        "repeat_count": REPEAT_COUNT,
        "aggregate_end_to_end_seconds_by_build": {
            build: list(values) for build, values in sorted(aggregate_times.items())
        },
        "aggregate_median_end_to_end_seconds_by_build": dict(sorted(scores.items())),
        "paired_relative_ci_vs_fastest": {
            build: list(interval) for build, interval in sorted(paired_cis.items())
        },
        "fastest_build": fastest,
        "eligible_builds": sorted(eligible),
    }
    reason = (
        f"best-safe-topology mode-block median E2E selected {selected_name}; fastest={fastest}; "
        f"eligible={','.join(sorted(eligible))}; "
        "portable/lower-memory tie-break within 3% or paired CI containing zero"
    )
    return candidates[selected_name], medians, reason, selection_statistics


def _select_topologies(
    rows: Sequence[AxisObservation],
    selected_build: str,
    host: HostPerformanceEnvelope,
) -> tuple[dict[str, ExecutionTopology], dict[str, int], dict[str, object], dict[str, str]]:
    chosen_rows = [item for item in rows if item.build_profile == selected_build]
    grouped: dict[str, dict[str, list[AxisObservation]]] = defaultdict(lambda: defaultdict(list))
    for row in chosen_rows:
        grouped[row.mode_block][row.topology_id].append(row)
    expected_blocks = {f"{mode}:{workload}" for mode in MODE_NAMES for workload in WORKLOAD_CLASSES}
    if set(grouped) != expected_blocks:
        raise CalibrationError("observations do not cover all five modes and workloads")
    selected: dict[str, ExecutionTopology] = {}
    required: dict[str, int] = {}
    details: dict[str, object] = {}
    rejected: dict[str, str] = {}
    for block in sorted(grouped):
        expected_axes = {item.base_axis_key for item in grouped[block][next(iter(grouped[block]))]}
        candidates: list[tuple[float, int, str, ExecutionTopology]] = []
        candidate_detail: dict[str, object] = {}
        for topology_id, values in sorted(grouped[block].items()):
            topology = values[0].topology
            if {item.base_axis_key for item in values} != expected_axes:
                rejected[f"{block}:{topology_id}"] = "topology axis matrix is incomplete"
                continue
            if any(item.topology.to_dict() != topology.to_dict() for item in values):
                rejected[f"{block}:{topology_id}"] = "topology identity changed"
                continue
            if set(topology.cpu_ids) != set(host.allowed_cpu_ids):
                rejected[f"{block}:{topology_id}"] = "topology does not consume all allowed CPUs"
                continue
            mode_name = block.split(":", 1)[0]
            if mode_name == "host_scheduler" and not topology.scheduler_cpu_ids:
                rejected[f"{block}:{topology_id}"] = (
                    "host_scheduler topology must reserve scheduler CPUs"
                )
                continue
            if mode_name != "host_scheduler" and topology.scheduler_cpu_ids:
                rejected[f"{block}:{topology_id}"] = (
                    "non-host-scheduler topology must not reserve scheduler CPUs"
                )
                continue
            peak = max(item.pss_bytes for item in values)
            scheduler_peak = max(item.scheduler_pss_bytes for item in values)
            client_peak = peak - scheduler_peak
            projected_peak = scheduler_peak + client_peak * topology.shard_count
            admission = check_memory_admission(
                host,
                projected_peak,
                headroom_fraction=0.20,
            )
            if host.swap_used_bytes != 0 or (
                host.swap_current_bytes is not None and host.swap_current_bytes != 0
            ):
                admission = MemoryAdmission(
                    False,
                    admission.available_bytes,
                    admission.required_bytes,
                    admission.headroom_bytes,
                    admission.swap_total_bytes,
                    admission.swap_used_bytes,
                    "actual swap is non-zero",
                    admission.swap_current_bytes,
                )
            if projected_peak > math.floor(host.effective_memory_limit_bytes * 0.80):
                admission = MemoryAdmission(
                    False,
                    admission.available_bytes,
                    admission.required_bytes,
                    admission.headroom_bytes,
                    admission.swap_total_bytes,
                    admission.swap_used_bytes,
                    "PSS exceeds 80% of effective memory",
                    admission.swap_current_bytes,
                )
            median = statistics.median(item.end_to_end_seconds for item in values)
            candidate_detail[topology_id] = {
                "pss_peak_bytes": peak,
                "scheduler_pss_peak_bytes": scheduler_peak,
                "projected_concurrent_pss_bytes": projected_peak,
                "projection_shard_count": topology.shard_count,
                "median_end_to_end_seconds": median,
                "memory_admission": {
                    "passed": admission.passed,
                    "available_bytes": admission.available_bytes,
                    "required_bytes": admission.required_bytes,
                    "headroom_bytes": admission.headroom_bytes,
                    "reason": admission.reason,
                    "swap_used_bytes": admission.swap_used_bytes,
                    "swap_current_bytes": admission.swap_current_bytes,
                },
            }
            if admission.passed:
                candidates.append((median, projected_peak, topology_id, topology))
        if not candidates:
            raise CalibrationError(f"no memory-safe topology remains for {block}")
        median, peak, topology_id, topology = min(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        selected[block] = topology
        required[block] = peak
        details[block] = {
            "selected_topology_id": topology_id,
            "required_bytes": peak,
            "median_end_to_end_seconds": median,
            "effective_memory_limit_bytes": host.effective_memory_limit_bytes,
            "candidate_topologies": candidate_detail,
        }
    return selected, required, details, rejected


def _lifecycle_contract(
    rows: Sequence[AxisObservation],
    *,
    selected_build: str,
    selected_topology_ids: Mapping[str, str],
    selected_topologies: Mapping[str, ExecutionTopology],
) -> tuple[dict[str, str], dict[str, object]]:
    by_workload: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen_observations: set[tuple[str, int]] = set()
    for row in rows:
        if row.build_profile != selected_build or not row.lifecycle_evidence:
            continue
        observation_identity = (row.build_profile, row.repeat)
        if observation_identity in seen_observations:
            continue
        seen_observations.add(observation_identity)
        for workload in WORKLOAD_CLASSES:
            raw_values = row.lifecycle_evidence.get(workload)
            if not isinstance(raw_values, list):
                raise CalibrationError("scheduler lifecycle evidence array is missing")
            topology_key = f"host_scheduler:{workload}"
            expected_topology_id = selected_topology_ids[topology_key]
            expected_topology = selected_topologies[topology_key]
            matching = [
                value
                for value in raw_values
                if isinstance(value, Mapping)
                and value.get("build_profile") == selected_build
                and value.get("topology_id") == expected_topology_id
            ]
            if len(matching) != 1:
                raise CalibrationError(
                    "scheduler lifecycle evidence is not bound to the selected build/topology"
                )
            selected_receipt = dict(cast(Mapping[str, object], matching[0]))
            if selected_receipt.get("topology") != expected_topology.to_dict():
                raise CalibrationError("scheduler lifecycle topology identity differs")
            by_workload[workload].append(selected_receipt)
    choices: dict[str, str] = {}
    evidence: dict[str, object] = {}
    for workload in WORKLOAD_CLASSES:
        values = by_workload.get(workload, [])
        if len(values) != REPEAT_COUNT:
            raise CalibrationError(
                "selected scheduler lifecycle requires exactly three repeat receipts"
            )
        safety = {
            field: all(value.get(field) is True for value in values)
            for field in _LIFECYCLE_SAFETY_FIELDS
        }
        per_wave_seconds = [
            _seconds(value["per_wave_end_to_end_seconds"], "per-wave lifecycle time")
            for value in values
        ]
        mode_block_seconds = [
            _seconds(value["mode_block_end_to_end_seconds"], "mode-block lifecycle time")
            for value in values
        ]
        per_wave_pss = [
            _positive(value["per_wave_scheduler_pss_peak_bytes"], "per-wave scheduler PSS")
            for value in values
        ]
        mode_block_pss = [
            _positive(
                value["mode_block_scheduler_pss_peak_bytes"],
                "mode-block scheduler PSS",
            )
            for value in values
        ]
        per_wave_process_tree_pss = [
            _positive(
                value["per_wave_process_tree_pss_peak_bytes"],
                "per-wave process-tree PSS",
            )
            for value in values
        ]
        mode_block_process_tree_pss = [
            _positive(
                value["mode_block_process_tree_pss_peak_bytes"],
                "mode-block process-tree PSS",
            )
            for value in values
        ]
        per_wave_median = statistics.median(per_wave_seconds)
        mode_block_median = statistics.median(mode_block_seconds)
        faster = mode_block_median < per_wave_median
        qualified = all(safety.values()) and faster
        evidence[workload] = {
            **dict(sorted(safety.items())),
            "selected_build_profile": selected_build,
            "selected_topology_id": selected_topology_ids[f"host_scheduler:{workload}"],
            "selected_topology": selected_topologies[f"host_scheduler:{workload}"].to_dict(),
            "mode_block_faster": faster,
            "per_wave_end_to_end_seconds_median": per_wave_median,
            "mode_block_end_to_end_seconds_median": mode_block_median,
            "per_wave_scheduler_pss_peak_bytes_max": max(per_wave_pss),
            "mode_block_scheduler_pss_peak_bytes_max": max(mode_block_pss),
            "per_wave_process_tree_pss_peak_bytes_max": max(per_wave_process_tree_pss),
            "mode_block_process_tree_pss_peak_bytes_max": max(mode_block_process_tree_pss),
            "sample_count": len(values),
            "wave_count": 2,
            "qualified": qualified,
        }
        choices[workload] = "mode-block" if qualified else "per-wave"
    return choices, evidence


def _signed_input_inventory(
    *,
    wheel_receipts: Sequence[WheelReceipt],
    observations: Sequence[AxisObservation],
    telemetry_overhead: TelemetryOverheadReceipt,
    telemetry_overhead_path: Path,
    host: HostPerformanceEnvelope,
    host_envelope_path: Path | None,
) -> tuple[dict[str, object], dict[str, str]]:
    wheel_entries: dict[Path, tuple[str, str]] = {}
    for receipt in wheel_receipts:
        if receipt.source_receipt_path is None or receipt.source_receipt_sha256 is None:
            raise CalibrationError("wheel receipt lacks signed source provenance")
        wheel_entries[receipt.source_receipt_path] = (
            receipt.source_receipt_sha256,
            receipt.build_profile,
        )
    observation_entries: dict[Path, tuple[str, str, int, str]] = {}
    for row in observations:
        if row.source_observation_path is None or row.source_observation_sha256 is None:
            raise CalibrationError("fixed-work observation lacks signed source provenance")
        source_path = row.source_observation_path.resolve()
        payload, _digest_value = _json_file(source_path)
        provenance = _mapping(payload.get("producer_provenance"), "producer_provenance")
        parent_run_label = _text(provenance.get("parent_run_label"), "parent_run_label")
        run_roots = [parent for parent in source_path.parents if parent.name == parent_run_label]
        if len(run_roots) != 1:
            raise CalibrationError("observation calibration-run alias is ambiguous")
        source_relative = source_path.relative_to(run_roots[0]).as_posix()
        portable_relative = f"fixed-work-observations/{parent_run_label}/{source_relative}"
        existing = observation_entries.get(source_path)
        identity = (
            row.source_observation_sha256,
            row.build_profile,
            row.repeat,
            portable_relative,
        )
        if existing is not None and existing != identity:
            raise CalibrationError("one observation file carries inconsistent portable identity")
        observation_entries[source_path] = identity
    try:
        reloaded_telemetry = load_telemetry_overhead_receipt(telemetry_overhead_path.resolve())
    except (OSError, ValueError) as error:
        raise CalibrationError("telemetry overhead source receipt is invalid") from error
    if reloaded_telemetry != telemetry_overhead:
        raise CalibrationError("telemetry overhead receipt changed after loading")
    telemetry_digest = _file_hash(
        telemetry_overhead_path.resolve(),
        "telemetry overhead receipt",
    )
    wheel_inventory = [
        {
            "storage_alias": CALIBRATION_INPUT_STORAGE_ALIAS,
            "relative_path": f"wheel-receipts/{profile}/{digest}.json",
            "sha256": digest,
        }
        for _path, (digest, profile) in sorted(wheel_entries.items(), key=lambda item: item[1][1])
    ]
    observation_inventory = [
        {
            "storage_alias": CALIBRATION_INPUT_STORAGE_ALIAS,
            "relative_path": portable_relative,
            "sha256": digest,
        }
        for _path, (digest, profile, repeat, portable_relative) in sorted(
            observation_entries.items(),
            key=lambda item: (item[1][1], item[1][2], item[1][0]),
        )
    ]
    telemetry_entry = {
        "storage_alias": CALIBRATION_INPUT_STORAGE_ALIAS,
        "relative_path": f"telemetry-overhead/{telemetry_digest}.json",
        "sha256": telemetry_digest,
    }
    inventory: dict[str, object] = {
        "wheel_receipts": wheel_inventory,
        "fixed_work_observations": observation_inventory,
        "telemetry_overhead": {
            **telemetry_entry,
        },
    }
    operational_locations: dict[str, str] = {}
    for path, (digest, profile) in wheel_entries.items():
        key = f"{CALIBRATION_INPUT_STORAGE_ALIAS}:wheel-receipts/{profile}/{digest}.json"
        operational_locations[key] = str(path.resolve())
    for path, (_digest_value, _profile, _repeat, portable_relative) in observation_entries.items():
        key = f"{CALIBRATION_INPUT_STORAGE_ALIAS}:{portable_relative}"
        operational_locations[key] = str(path.resolve())
    operational_locations[
        f"{CALIBRATION_INPUT_STORAGE_ALIAS}:{telemetry_entry['relative_path']}"
    ] = str(telemetry_overhead_path.resolve())
    if host_envelope_path is not None:
        host_path = host_envelope_path.resolve()
        host_payload, host_digest = _json_file(host_path)
        try:
            reloaded_host = HostPerformanceEnvelope.from_dict(host_payload)
        except ValueError as error:
            raise CalibrationError("signed host performance envelope is invalid") from error
        if reloaded_host.to_dict() != host.to_dict():
            raise CalibrationError("signed host performance envelope changed")
        host_entry = {
            "storage_alias": CALIBRATION_INPUT_STORAGE_ALIAS,
            "relative_path": f"host-envelope/{host_digest}.json",
            "sha256": host_digest,
        }
        inventory["host_envelope"] = host_entry
        operational_locations[
            f"{CALIBRATION_INPUT_STORAGE_ALIAS}:{host_entry['relative_path']}"
        ] = str(host_path)
    return inventory, operational_locations


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    run_label: str
    profile: FrozenPerformanceProfile
    wheel_receipts: tuple[WheelReceipt, ...]
    observations: tuple[AxisObservation, ...]
    rejected_builds: Mapping[str, str]
    mode_block_medians: Mapping[str, Mapping[str, float]]
    selected_topology_ids: Mapping[str, str]
    telemetry_overhead: TelemetryOverheadReceipt
    signed_input_inventory: Mapping[str, object]
    operational_input_locations: Mapping[str, str]

    @property
    def selected_build_profile(self) -> str:
        return self.profile.selected_build.name


def calibrate_performance_profile(
    *,
    run_label: str,
    wheel_receipts: Sequence[WheelReceipt],
    observations: Sequence[AxisObservation],
    host: HostPerformanceEnvelope,
    telemetry_overhead: TelemetryOverheadReceipt,
    telemetry_overhead_path: Path,
    host_envelope_path: Path | None = None,
) -> CalibrationResult:
    if _RUN_LABEL_RE.fullmatch(run_label) is None:
        raise CalibrationError(f"invalid calibration run label: {run_label}")
    receipt_profiles = {item.build_profile for item in wheel_receipts}
    if (
        not set(REQUIRED_BUILD_PROFILES).issubset(receipt_profiles)
        or not receipt_profiles.issubset(SUPPORTED_BUILD_PROFILES)
        or len(receipt_profiles) != len(wheel_receipts)
    ):
        raise CalibrationError("calibration build profile set is invalid")
    if host.platform_name != "linux" or host.architecture != "x86_64":
        raise CalibrationError("calibration supports Linux/WSL x86-64 only")
    try:
        telemetry_overhead.require_passed()
        telemetry_overhead.require_representative_fixed_work()
    except ValueError as error:
        raise CalibrationError("telemetry overhead gate failed") from error
    signed_input_inventory, operational_input_locations = _signed_input_inventory(
        wheel_receipts=wheel_receipts,
        observations=observations,
        telemetry_overhead=telemetry_overhead,
        telemetry_overhead_path=telemetry_overhead_path,
        host=host,
        host_envelope_path=host_envelope_path,
    )
    _validate_identities(wheel_receipts, observations)
    _, rejected = _validate_matrix(observations)
    candidates = {item.build_profile: _candidate(item) for item in wheel_receipts}
    selected, medians, reason, build_selection_statistics = _select_build(
        candidates,
        observations,
        rejected,
        host,
    )
    topologies, required, pss, rejected_topologies = _select_topologies(
        observations, selected.name, host
    )
    keys = {f"{mode}:{workload}" for mode in MODE_NAMES for workload in WORKLOAD_CLASSES}
    if set(topologies) != keys or set(required) != keys:
        raise CalibrationError("frozen topology matrix is incomplete")
    selected_ids = {
        block: cast(str, cast(Mapping[str, object], value)["selected_topology_id"])
        for block, value in pss.items()
    }
    lifecycle_choices, lifecycle_evidence = _lifecycle_contract(
        observations,
        selected_build=selected.name,
        selected_topology_ids=selected_ids,
        selected_topologies=topologies,
    )
    semantic_by_build: dict[str, dict[str, str]] = defaultdict(dict)
    for row in observations:
        if row.repeat == 0:
            key = f"{row.mode}:{row.workload_class}:{row.instance}:{row.seed}:{row.topology_id}"
            semantic_by_build[row.build_profile][key] = row.semantic_digest
    semantic = semantic_by_build.get(selected.name, {})
    calibration: dict[str, object] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "memory_required_bytes_by_topology": dict(sorted(required.items())),
        "pss_calibration": pss,
        "mode_block_medians_seconds": medians,
        "mode_block_e2e_medians_seconds": medians,
        "build_selection_statistics": build_selection_statistics,
        "semantic_digests": dict(sorted(semantic.items())),
        "semantic_digests_by_build": {
            build: dict(sorted(values.items()))
            for build, values in sorted(semantic_by_build.items())
        },
        "rejected_builds": dict(sorted(rejected.items())),
        "rejected_topologies": dict(sorted(rejected_topologies.items())),
        "repeat_count": REPEAT_COUNT,
        "fixed_work_budget": observations[0].fixed_work_budget,
        "fixed_work_only": True,
        "formal_started": False,
        "cuda_started": False,
        "attempt08_started": False,
        "host_scheduler_lifecycle_by_workload": lifecycle_choices,
        "host_scheduler_lifecycle_evidence": lifecycle_evidence,
        "telemetry_overhead": telemetry_overhead.to_dict(),
        "signed_input_inventory": signed_input_inventory,
    }
    profile = freeze_performance_profile(
        host, selected, topologies, calibration=calibration, selection_reason=reason
    )
    return CalibrationResult(
        run_label,
        profile,
        tuple(wheel_receipts),
        tuple(observations),
        dict(rejected),
        medians,
        selected_ids,
        telemetry_overhead,
        signed_input_inventory,
        operational_input_locations,
    )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> str:
    if path.exists():
        raise FileExistsError(f"calibration output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    sidecar = Path(f"{path}.sha256")
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary_sidecar.exists() or sidecar.exists():
        raise FileExistsError(f"calibration output temporary/sidecar exists: {path}")
    path_published = False
    sidecar_published = False
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary_sidecar.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(digest + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        publish_no_replace(temporary_sidecar, sidecar)
        sidecar_published = True
        publish_no_replace(temporary, path)
        path_published = True
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        if not path_published and sidecar_published:
            sidecar.unlink(missing_ok=True)
    return digest


def _portable_child(root: Path, relative_path: object, name: str) -> Path:
    raw = _text(relative_path, name)
    portable = PurePosixPath(raw)
    if portable.is_absolute() or ".." in portable.parts or "\\" in raw:
        raise CalibrationError(f"{name} must be a portable relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / Path(*portable.parts)).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:  # pragma: no cover - guarded by PurePosixPath
        raise CalibrationError(f"{name} escapes its storage root") from error
    return resolved


def _materialize_file(
    source: Path,
    destination: Path,
    expected_sha256: str,
    *,
    signed_json: bool,
) -> None:
    expected = _sha256(expected_sha256, "materialized file sha256")
    if _file_hash(source, "calibration input") != expected:
        raise CalibrationError("calibration input changed before materialization")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _file_hash(destination, "materialized calibration input") != expected:
            raise CalibrationError("materialized calibration input path collision")
    else:
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        if temporary.exists():
            raise FileExistsError(f"calibration input temporary exists: {temporary}")
        try:
            digest = hashlib.sha256()
            with source.open("rb") as reader, temporary.open("xb") as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    digest.update(chunk)
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            if digest.hexdigest() != expected:
                raise CalibrationError("calibration input changed during materialization")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    if signed_json:
        sidecar = Path(f"{destination}.sha256")
        if sidecar.exists():
            try:
                declared = sidecar.read_text(encoding="ascii").strip()
            except (OSError, UnicodeError) as error:
                raise CalibrationError("materialized input sidecar is unreadable") from error
            if declared != expected:
                raise CalibrationError("materialized input sidecar path collision")
        else:
            with sidecar.open("x", encoding="ascii", newline="\n") as handle:
                handle.write(expected + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def _observation_bundle_context(
    observation_path: Path,
) -> tuple[Path, Mapping[str, object]]:
    payload, _digest_value = _json_file(observation_path)
    provenance = _mapping(payload.get("producer_provenance"), "producer_provenance")
    parent_run_label = _text(provenance.get("parent_run_label"), "parent_run_label")
    roots = [
        parent for parent in observation_path.resolve().parents if parent.name == parent_run_label
    ]
    if len(roots) != 1:
        raise CalibrationError("observation calibration-run alias is ambiguous")
    return roots[0], provenance


def _materialize_observation_support(source: Path, destination: Path) -> None:
    source_root, provenance = _observation_bundle_context(source)
    destination_root, _destination_provenance = _observation_bundle_context(destination)

    for prefix in ("start_permit", "host_envelope"):
        relative = provenance[f"{prefix}_relative_path"]
        expected = _sha256(provenance[f"{prefix}_sha256"], f"{prefix}_sha256")
        _materialize_file(
            _portable_child(source_root, relative, f"{prefix}_relative_path"),
            _portable_child(destination_root, relative, f"{prefix}_relative_path"),
            expected,
            signed_json=True,
        )

    inventory = provenance.get("raw_axis_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise CalibrationError("observation raw-axis inventory is empty")
    for index, raw_entry in enumerate(inventory):
        entry = _mapping(raw_entry, f"raw_axis_inventory[{index}]")
        raw_source = _portable_child(
            source_root, entry.get("relative_path"), "raw axis relative_path"
        )
        raw_destination = _portable_child(
            destination_root, entry.get("relative_path"), "raw axis relative_path"
        )
        raw_sha = _sha256(entry.get("sha256"), "raw axis sha256")
        _materialize_file(raw_source, raw_destination, raw_sha, signed_json=False)
        sidecar_source = _portable_child(
            source_root,
            entry.get("relative_sidecar_path"),
            "raw axis relative_sidecar_path",
        )
        sidecar_destination = _portable_child(
            destination_root,
            entry.get("relative_sidecar_path"),
            "raw axis relative_sidecar_path",
        )
        sidecar_sha = _sha256(entry.get("sidecar_sha256"), "raw axis sidecar sha256")
        _materialize_file(
            sidecar_source,
            sidecar_destination,
            sidecar_sha,
            signed_json=False,
        )
        try:
            declared = sidecar_destination.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError) as error:
            raise CalibrationError("materialized raw sidecar is unreadable") from error
        if declared != raw_sha:
            raise CalibrationError("materialized raw sidecar does not bind its artifact")
        supporting = entry.get("supporting_artifacts")
        if not isinstance(supporting, list) or not supporting:
            raise CalibrationError("observation supporting-artifact inventory is empty")
        for support_index, raw_support in enumerate(supporting):
            support = _mapping(
                raw_support,
                f"raw_axis_inventory[{index}].supporting_artifacts[{support_index}]",
            )
            support_source = _portable_child(
                source_root,
                support.get("relative_path"),
                "raw axis supporting artifact path",
            )
            support_destination = _portable_child(
                destination_root,
                support.get("relative_path"),
                "raw axis supporting artifact path",
            )
            support_sha = _sha256(
                support.get("sha256"),
                "raw axis supporting artifact sha256",
            )
            _text(support.get("role"), "raw axis supporting artifact role")
            _materialize_file(
                support_source,
                support_destination,
                support_sha,
                signed_json=False,
            )
    scheduler_inventory = provenance.get("scheduler_task_receipt_inventory")
    if not isinstance(scheduler_inventory, list):
        raise CalibrationError("observation scheduler task-receipt inventory is invalid")
    for index, raw_entry in enumerate(scheduler_inventory):
        entry = _mapping(raw_entry, f"scheduler_task_receipt_inventory[{index}]")
        raw_source = _portable_child(
            source_root,
            entry.get("relative_path"),
            "scheduler task-receipt relative_path",
        )
        raw_destination = _portable_child(
            destination_root,
            entry.get("relative_path"),
            "scheduler task-receipt relative_path",
        )
        raw_sha = _sha256(entry.get("sha256"), "scheduler task-receipt sha256")
        _materialize_file(raw_source, raw_destination, raw_sha, signed_json=False)
        sidecar_source = _portable_child(
            source_root,
            entry.get("relative_sidecar_path"),
            "scheduler task-receipt sidecar relative_path",
        )
        sidecar_destination = _portable_child(
            destination_root,
            entry.get("relative_sidecar_path"),
            "scheduler task-receipt sidecar relative_path",
        )
        sidecar_sha = _sha256(entry.get("sidecar_sha256"), "scheduler task-receipt sidecar sha256")
        _materialize_file(
            sidecar_source,
            sidecar_destination,
            sidecar_sha,
            signed_json=False,
        )
        if sidecar_destination.read_text(encoding="ascii").strip() != raw_sha:
            raise CalibrationError(
                "materialized scheduler task-receipt sidecar does not bind its artifact"
            )


def _materialize_signed_inputs(result: CalibrationResult, run_dir: Path) -> Path:
    input_root = (run_dir / CALIBRATION_INPUT_DIRECTORY).resolve()
    if input_root.exists():
        raise FileExistsError(f"calibration input bundle already exists: {input_root}")
    input_root.mkdir(parents=True)
    locations = dict(result.operational_input_locations)
    inventory = _mapping(result.signed_input_inventory, "signed_input_inventory")
    wheel_by_source = {
        receipt.source_receipt_path.resolve(): receipt
        for receipt in result.wheel_receipts
        if receipt.source_receipt_path is not None
    }

    def materialize_entry(raw_entry: object, field: str) -> tuple[Path, Path]:
        entry = _mapping(raw_entry, field)
        alias = _text(entry.get("storage_alias"), f"{field}.storage_alias")
        if alias != CALIBRATION_INPUT_STORAGE_ALIAS:
            raise CalibrationError(f"{field} storage alias is invalid")
        relative = _text(entry.get("relative_path"), f"{field}.relative_path")
        expected = _sha256(entry.get("sha256"), f"{field}.sha256")
        key = f"{alias}:{relative}"
        source_raw = locations.get(key)
        if source_raw is None:
            raise CalibrationError(f"{field} has no operational source")
        source = Path(source_raw).resolve()
        destination = _portable_child(input_root, relative, f"{field}.relative_path")
        _materialize_file(source, destination, expected, signed_json=True)
        return source, destination

    wheel_entries = inventory.get("wheel_receipts")
    if not isinstance(wheel_entries, list):
        raise CalibrationError("wheel receipt inventory is invalid")
    for index, raw_entry in enumerate(wheel_entries):
        source, destination = materialize_entry(raw_entry, f"wheel_receipts[{index}]")
        receipt = wheel_by_source.get(source)
        if receipt is None or receipt.source_receipt_path is None:
            raise CalibrationError("wheel receipt source does not match loaded evidence")
        source_root = receipt.source_receipt_path.parent.resolve()
        for artifact_source, expected, name in (
            (receipt.wheel_path, receipt.artifact_identity.wheel_sha256, "wheel"),
            (receipt.native_path, receipt.artifact_identity.native_sha256, "native"),
            (receipt.scheduler_path, receipt.artifact_identity.scheduler_sha256, "scheduler"),
        ):
            try:
                relative_artifact = artifact_source.resolve().relative_to(source_root).as_posix()
            except ValueError as error:
                raise CalibrationError(f"{name} escapes its portable build bundle") from error
            artifact_destination = _portable_child(
                destination.parent, relative_artifact, f"{name}_relative_path"
            )
            _materialize_file(
                artifact_source,
                artifact_destination,
                expected,
                signed_json=False,
            )

    observation_entries = inventory.get("fixed_work_observations")
    if not isinstance(observation_entries, list):
        raise CalibrationError("fixed-work observation inventory is invalid")
    for index, raw_entry in enumerate(observation_entries):
        source, destination = materialize_entry(raw_entry, f"fixed_work_observations[{index}]")
        _materialize_observation_support(source, destination)

    materialize_entry(inventory.get("telemetry_overhead"), "telemetry_overhead")
    if inventory.get("host_envelope") is not None:
        materialize_entry(inventory["host_envelope"], "host_envelope")
    return input_root


def write_calibration_bundle(
    result: CalibrationResult,
    *,
    output_root: Path,
    allow_existing_run_dir: bool = False,
) -> dict[str, Path]:
    run_dir = (output_root / result.run_label).resolve()
    if run_dir.exists() and not allow_existing_run_dir:
        raise FileExistsError(f"calibration run label already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=allow_existing_run_dir)
    input_root = _materialize_signed_inputs(result, run_dir)
    profile_path = run_dir / "frozen_performance_profile.json"
    profile_sha = _atomic_json(profile_path, result.profile.to_dict())
    calibration_payload = _mapping(
        result.profile.to_dict()["calibration"],
        "frozen profile calibration",
    )
    receipt_path = run_dir / "calibration_receipt.json"
    _atomic_json(
        receipt_path,
        {
            "schema_version": CALIBRATION_RECEIPT_SCHEMA_VERSION,
            "calibration_schema_version": CALIBRATION_SCHEMA_VERSION,
            "run_label": result.run_label,
            "status": "calibrated",
            "formal_started": False,
            "cuda_started": False,
            "attempt08_started": False,
            "profile_path": profile_path.name,
            "profile_sha256": profile_sha,
            "profile_canonical_sha256": result.profile.canonical_sha256,
            "selection_reason": result.profile.selection_reason,
            "host": result.profile.host.to_dict(),
            "selected_build_profile": result.selected_build_profile,
            "selected_topology_ids": dict(sorted(result.selected_topology_ids.items())),
            "fixed_work_budget": calibration_payload["fixed_work_budget"],
            "memory_required_bytes_by_topology": calibration_payload[
                "memory_required_bytes_by_topology"
            ],
            "pss_calibration": calibration_payload["pss_calibration"],
            "semantic_digests": calibration_payload["semantic_digests"],
            "semantic_digests_by_build": calibration_payload["semantic_digests_by_build"],
            "host_scheduler_lifecycle_by_workload": calibration_payload[
                "host_scheduler_lifecycle_by_workload"
            ],
            "host_scheduler_lifecycle_evidence": calibration_payload[
                "host_scheduler_lifecycle_evidence"
            ],
            "telemetry_overhead": result.telemetry_overhead.to_dict(),
            "signed_input_inventory": dict(result.signed_input_inventory),
            "input_storage_roots": {
                CALIBRATION_INPUT_STORAGE_ALIAS: input_root.relative_to(run_dir).as_posix()
            },
            "mode_block_medians_seconds": result.mode_block_medians,
            "mode_block_e2e_medians_seconds": result.mode_block_medians,
            "build_selection_statistics": calibration_payload["build_selection_statistics"],
            "rejected_builds": dict(sorted(result.rejected_builds.items())),
            "wheel_receipts": [
                {
                    "build_profile": item.build_profile,
                    "artifact_identity": item.artifact_identity.to_dict(),
                }
                for item in result.wheel_receipts
            ],
            "observation_count": len(result.observations),
            "repeat_count": REPEAT_COUNT,
            "semantic_digest_count": len({item.semantic_digest for item in result.observations}),
        },
    )
    return {
        "run_dir": run_dir,
        "profile": profile_path,
        "profile_sidecar": Path(f"{profile_path}.sha256"),
        "receipt": receipt_path,
        "receipt_sidecar": Path(f"{receipt_path}.sha256"),
    }


def _terminal_manifest(
    *,
    output_root: Path,
    run_label: str,
    status: str,
    receipt_path: Path,
    failure: BaseException | None = None,
) -> Path:
    if status not in {"complete", "failed"}:
        raise ValueError("calibration terminal manifest status is invalid")
    run_dir = (output_root / run_label).resolve()
    receipt_sidecar = Path(f"{receipt_path}.sha256")
    path = run_dir / "manifest.json"
    _atomic_json(
        path,
        {
            "schema_version": "stage05.2-performance-calibration-manifest-v1",
            "run_label": run_label,
            "status": status,
            "evidence_completeness": "complete" if status == "complete" else "partial",
            "artifact_inventory": [
                {
                    "relative_path": receipt_path.relative_to(run_dir).as_posix(),
                    "sha256": _file_hash(receipt_path, "calibration receipt"),
                    "sidecar_relative_path": receipt_sidecar.relative_to(run_dir).as_posix(),
                    "sidecar_sha256": _file_hash(
                        receipt_sidecar,
                        "calibration receipt sidecar",
                    ),
                }
            ],
            "failure_type": None if failure is None else type(failure).__name__,
            "failure": None if failure is None else str(failure),
            "formal_started": False,
            "cuda_started": False,
            "attempt08_started": False,
        },
    )
    return path


def run_calibration_cli(
    *,
    wheel_receipt_paths: Sequence[Path],
    observation_paths: Sequence[Path],
    output_root: Path,
    run_label: str,
    host_envelope_path: Path | None = None,
    telemetry_overhead_receipt_path: Path,
    start_permit: StartPermit,
    allow_existing_run_dir: bool = False,
) -> dict[str, Path]:
    if start_permit.run_label != run_label or not start_permit.permit_path.is_file():
        raise CalibrationError("calibration lacks a lifecycle-bound start permit")
    receipts = load_wheel_receipts(wheel_receipt_paths)
    observations = load_fixed_work_observations(observation_paths)
    telemetry_overhead = load_telemetry_overhead_receipt(telemetry_overhead_receipt_path.resolve())
    if host_envelope_path is None:
        host = detect_host_performance()
    else:
        try:
            host_payload, _host_digest = _json_file(host_envelope_path)
            host = HostPerformanceEnvelope.from_dict(host_payload)
        except ValueError as error:
            raise CalibrationError("invalid host performance envelope") from error
    outputs = write_calibration_bundle(
        calibrate_performance_profile(
            run_label=run_label,
            wheel_receipts=receipts,
            observations=observations,
            host=host,
            telemetry_overhead=telemetry_overhead,
            telemetry_overhead_path=telemetry_overhead_receipt_path,
            host_envelope_path=host_envelope_path,
        ),
        output_root=output_root,
        allow_existing_run_dir=allow_existing_run_dir,
    )
    manifest_path = _terminal_manifest(
        output_root=output_root,
        run_label=run_label,
        status="complete",
        receipt_path=outputs["receipt"],
    )
    outputs["manifest"] = manifest_path
    outputs["manifest_sidecar"] = Path(f"{manifest_path}.sha256")
    return outputs


def _representative_telemetry_runner(
    *,
    repository_root_path: Path,
    receipt: WheelReceipt,
    run_dir: Path,
    host_envelope_path: Path,
    start_permit: StartPermit,
    warm_start_bundle_path: Path,
    benchmark_dir: Path,
    continuity_lease_token: str,
) -> Callable[[bool, int], TelemetryWorkloadSample]:
    """Build the attested child callback used by the complete telemetry A/B."""

    if receipt.source_receipt_path is None:
        raise CalibrationError("telemetry build receipt source path is unavailable")
    root = repository_root_path.resolve(strict=True)
    installed_root = receipt.native_path.parent.parent.resolve(strict=True)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join((str(installed_root), str(root))),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    def run(enabled: bool, sample_index: int) -> TelemetryWorkloadSample:
        label = (
            "warm-on"
            if sample_index == -1
            else (
                "warm-off"
                if sample_index == -2
                else f"sample-{sample_index:02d}-{'on' if enabled else 'off'}"
            )
        )
        sample_path = run_dir / "telemetry_samples" / f"{label}.json"
        raw_root = run_dir / "telemetry_raw" / label
        command = (
            sys.executable,
            "-m",
            "evrptw.experiments.stage052_performance_observation",
            "--wheel-receipt",
            str(receipt.source_receipt_path),
            "--warm-start-bundle",
            str(warm_start_bundle_path.resolve()),
            "--benchmark-dir",
            str(benchmark_dir.resolve()),
            "--output",
            str(sample_path),
            "--raw-output-root",
            str(raw_root),
            "--repeat",
            "0",
            "--continuity-lease-token",
            continuity_lease_token,
            "--parent-run-dir",
            str(run_dir),
            "--start-permit",
            str(start_permit.permit_path.resolve()),
            "--host-envelope",
            str(host_envelope_path),
            "--repository-root",
            str(root),
            "--telemetry-sample",
            "on" if enabled else "off",
        )
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not sample_path.is_file():
            raise CalibrationError(
                "representative telemetry child failed: " + completed.stderr.strip()[-2000:]
            )
        payload, digest = _json_file(sample_path)
        if (
            payload.get("schema_version") != TELEMETRY_SAMPLE_SCHEMA_VERSION
            or payload.get("enabled") is not enabled
        ):
            raise CalibrationError("representative telemetry sample identity mismatch")
        fingerprint = payload.get("fingerprint")
        summary = payload.get("resource_summary")
        evidence = payload.get("workload_evidence")
        if (
            not isinstance(fingerprint, str)
            or _SHA256_RE.fullmatch(fingerprint) is None
            or not isinstance(summary, Mapping)
            or not isinstance(evidence, Mapping)
        ):
            raise CalibrationError("representative telemetry sample is malformed")
        relative_path = sample_path.relative_to(run_dir).as_posix()
        sample_sidecar = Path(f"{sample_path}.sha256")
        return TelemetryWorkloadSample(
            fingerprint.encode("ascii"),
            {
                **dict(summary),
                "sample_storage_alias": "stage052-performance-calibration-run",
                "sample_relative_path": relative_path,
                "sample_sha256": digest,
                "sample_sidecar_relative_path": sample_sidecar.relative_to(run_dir).as_posix(),
                "sample_sidecar_sha256": _file_hash(
                    sample_sidecar,
                    "representative telemetry sample sidecar",
                ),
                "child_elapsed_seconds": payload.get("elapsed_seconds"),
                "replay_seconds": payload.get("replay_seconds"),
                "raw_axis_inventory": payload.get("raw_axis_inventory"),
            },
            dict(evidence),
        )

    return run


def _produce_observation_children(
    *,
    repository_root_path: Path,
    receipts: Sequence[WheelReceipt],
    run_dir: Path,
    host_envelope_path: Path,
    start_permit: StartPermit,
    warm_start_bundle_path: Path,
    benchmark_dir: Path,
    continuity_lease_token: str,
) -> tuple[Path, ...]:
    root = repository_root_path.resolve(strict=True)
    outputs: list[Path] = []
    for receipt in receipts:
        if receipt.source_receipt_path is None:
            raise CalibrationError("wheel receipt source path is unavailable")
        installed_root = receipt.native_path.parent.parent.resolve(strict=True)
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join((str(installed_root), str(root))),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        for repeat in range(REPEAT_COUNT):
            observation_path = (
                run_dir / "observations" / receipt.build_profile / f"repeat{repeat + 1}.json"
            )
            raw_root = run_dir / "raw" / receipt.build_profile / f"repeat{repeat + 1}"
            process_receipt_path = (
                run_dir
                / "observation_processes"
                / receipt.build_profile
                / f"repeat{repeat + 1}.json"
            )
            command = (
                sys.executable,
                "-m",
                "evrptw.experiments.stage052_performance_observation",
                "--wheel-receipt",
                str(receipt.source_receipt_path),
                "--warm-start-bundle",
                str(warm_start_bundle_path.resolve()),
                "--benchmark-dir",
                str(benchmark_dir.resolve()),
                "--output",
                str(observation_path),
                "--raw-output-root",
                str(raw_root),
                "--repeat",
                str(repeat),
                "--continuity-lease-token",
                continuity_lease_token,
                "--parent-run-dir",
                str(run_dir),
                "--start-permit",
                str(start_permit.permit_path.resolve()),
                "--host-envelope",
                str(host_envelope_path),
                "--repository-root",
                str(root),
            )
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            recorded_command = list(command)
            token_index = recorded_command.index("--continuity-lease-token") + 1
            recorded_command[token_index] = "<redacted>"
            _atomic_json(
                process_receipt_path,
                {
                    "schema_version": "stage05.2-performance-observation-process-v1",
                    "run_label": run_dir.name,
                    "build_profile": receipt.build_profile,
                    "repeat": repeat,
                    "command": recorded_command,
                    "installed_root": str(installed_root),
                    "returncode": completed.returncode,
                    "elapsed_seconds": time.perf_counter() - started,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "observation_path": str(observation_path),
                    "observation_sha256": (
                        _file_hash(observation_path, "fixed-work observation")
                        if completed.returncode == 0 and observation_path.is_file()
                        else None
                    ),
                },
            )
            if completed.returncode != 0 or not observation_path.is_file():
                raise CalibrationError(
                    f"fixed-work observation child failed for "
                    f"{receipt.build_profile} repeat {repeat + 1}"
                )
            outputs.append(observation_path)
    return tuple(outputs)


def calibration_resource_request(host: HostPerformanceEnvelope) -> dict[str, int]:
    """Return the portable lifecycle reservation for the detected CPU set."""

    cpu_count = len(host.allowed_cpu_ids)
    if cpu_count <= 0 or cpu_count > 1024:
        raise CalibrationError("detected CPU set is outside the Linux CPU_SET contract")
    return {
        "workers": cpu_count,
        # One full compute pool, one request/control lane per possible client,
        # plus the coordinator.  BLAS/OpenMP pools remain frozen at one.
        "threads": 2 * cpu_count + 1,
        # The N x 1 candidate can pair each client with one spawn worker; keep
        # bounded coordinator/service headroom without assuming this host has 24 CPUs.
        "processes": 2 * cpu_count + 8,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run and independently seal Stage 5.2 performance calibration children"
    )
    parser.add_argument("--wheel-receipt", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--warm-start-bundle", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--continuity-lease-token", required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        required=True,
        help="Explicit clean ext4 Git worktree used for source identity",
    )
    parser.add_argument("--telemetry-repeat-count", type=int, default=5)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage052_performance_calibration.toml"),
    )
    arguments = parser.parse_args(argv)
    resolved_repository = require_clean_repository_root(arguments.repository_root)
    resolved_config = (
        arguments.config.resolve(strict=True)
        if arguments.config.is_absolute()
        else (resolved_repository / arguments.config).resolve(strict=True)
    )
    output_dir = arguments.output_root / arguments.run_label
    host = detect_host_performance()
    resource_request = calibration_resource_request(host)
    start_permit = preflight_cli_attempt(
        config_path=resolved_config,
        output_dir=output_dir,
        run_label=arguments.run_label,
        workers=resource_request["workers"],
        threads=resource_request["threads"],
        processes=resource_request["processes"],
    )
    try:
        require_owned(
            resolved_repository,
            token=arguments.continuity_lease_token,
            allowed_phases=frozenset({"performance-calibration"}),
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        receipts = load_wheel_receipts(arguments.wheel_receipt)
        host_envelope_path = output_dir / "host_performance_envelope.json"
        _atomic_json(host_envelope_path, host.to_dict())
        telemetry_path = output_dir / "telemetry_overhead.json"
        telemetry_receipt = next(
            (receipt for receipt in receipts if receipt.build_profile == "portable-o3"),
            None,
        )
        if telemetry_receipt is None:
            raise CalibrationError("portable-o3 is required for representative telemetry A/B")
        telemetry = measure_representative_telemetry_overhead(
            run_sample=_representative_telemetry_runner(
                repository_root_path=resolved_repository,
                receipt=telemetry_receipt,
                run_dir=output_dir,
                host_envelope_path=host_envelope_path,
                start_permit=start_permit,
                warm_start_bundle_path=arguments.warm_start_bundle,
                benchmark_dir=arguments.benchmark_dir,
                continuity_lease_token=arguments.continuity_lease_token,
            ),
            repeat_count=arguments.telemetry_repeat_count,
        )
        write_telemetry_overhead_receipt(telemetry_path, telemetry)
        telemetry.require_passed()
        observation_paths = _produce_observation_children(
            repository_root_path=resolved_repository,
            receipts=receipts,
            run_dir=output_dir,
            host_envelope_path=host_envelope_path,
            start_permit=start_permit,
            warm_start_bundle_path=arguments.warm_start_bundle,
            benchmark_dir=arguments.benchmark_dir,
            continuity_lease_token=arguments.continuity_lease_token,
        )
        outputs = run_calibration_cli(
            wheel_receipt_paths=arguments.wheel_receipt,
            observation_paths=observation_paths,
            output_root=arguments.output_root,
            run_label=arguments.run_label,
            host_envelope_path=host_envelope_path,
            telemetry_overhead_receipt_path=telemetry_path,
            start_permit=start_permit,
            allow_existing_run_dir=True,
        )
        seal_cli_attempt(
            config_path=resolved_config,
            output_dir=output_dir,
            run_label=arguments.run_label,
            manifest_path=outputs["manifest"],
        )
    except BaseException as error:
        seal_failed_cli_attempt(
            config_path=resolved_config,
            output_dir=output_dir,
            run_label=arguments.run_label,
            error=error,
        )
        raise
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Small aliases keep the selector easy to embed in an audited preflight while
# retaining the explicit names used by the CLI implementation.
load_observations = load_fixed_work_observations
calibrate = calibrate_performance_profile


__all__ = (
    "AxisObservation",
    "CALIBRATION_INPUT_DIRECTORY",
    "CALIBRATION_INPUT_STORAGE_ALIAS",
    "CALIBRATION_RECEIPT_SCHEMA_VERSION",
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationError",
    "CalibrationResult",
    "MODE_NAMES",
    "OBSERVATION_PRODUCER_SCHEMA_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "OPTIONAL_BUILD_PROFILES",
    "PERFORMANCE_BUILD_STORAGE_ALIAS",
    "REPEAT_COUNT",
    "REQUIRED_BUILD_PROFILES",
    "SUPPORTED_BUILD_PROFILES",
    "WheelReceipt",
    "WHEEL_RECEIPT_SCHEMA_VERSION",
    "WORKLOAD_CLASSES",
    "calibrate_performance_profile",
    "calibrate",
    "calibration_resource_request",
    "load_fixed_work_observations",
    "load_wheel_receipt",
    "load_observations",
    "load_wheel_receipts",
    "main",
    "run_calibration_cli",
    "write_calibration_bundle",
)
