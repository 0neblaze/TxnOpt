"""Shared, auditable experiment-artifact storage.

The public interface in this module is deliberately small.  Experiment
runners provide a run context and records; this module owns the physical
layout, Arrow schemas, compression, checksums, manifest, and byte-budget
semantics.  Historical JSON/JSONL bundles are read through the compatibility
helpers but are never rewritten by this module.
"""

from __future__ import annotations

import contextlib
import hashlib
import heapq
import json
import os
import re
import shutil
import sqlite3
import tempfile
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, cast, overload

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

ARTIFACT_STORAGE_SCHEMA_VERSION = "artifact-storage-v1"
ARTIFACT_STORAGE_V2 = "artifact-storage-v2"
SUPPORTED_STORAGE_POLICIES = frozenset({ARTIFACT_STORAGE_SCHEMA_VERSION, ARTIFACT_STORAGE_V2})
SCREENING_DECISIONS_V1 = "screening_decisions_v1"
SCREENING_DECISIONS_V2 = "screening_decisions_v2"
SCREENING_DECISIONS_V3 = "screening_decisions_v3"
SUPPORTED_SCREENING_SCHEMAS = frozenset(
    {SCREENING_DECISIONS_V1, SCREENING_DECISIONS_V2, SCREENING_DECISIONS_V3}
)
V2_PARQUET_ROW_GROUP_SIZE = 65_536
ROUTE_IDENTITY_HOT_CACHE_ENTRIES = V2_PARQUET_ROW_GROUP_SIZE
ROUTE_IDENTITY_MEMORY_ENTRIES = 262_144
ROUTE_ID_RESOLUTION_CACHE_ENTRIES = V2_PARQUET_ROW_GROUP_SIZE
SCREENING_DEFINITION_HOT_CACHE_ENTRIES = 524_288
ROUTE_IDENTITY_COUNTER_NAMESPACES = 16
MAX_SCREENING_CHECKS_PER_DECISION = 8
# Keep live definition/occurrence transactions below one 65,536-row Parquet
# group while amortising collision checks and typed-column appends.
LIVE_SCREENING_TRANSACTION_ROWS = 8_192
UNIQUE_ROUTE_IDENTITY_SEMANTICS = frozenset(
    {"legacy_started", "completed_shared", "completed_lane"}
)
CURRENT_STORAGE_FORMAT = "parquet_or_json_control"
LEGACY_STORAGE_FORMAT = "legacy_json_or_jsonl"
DEFAULT_COMPRESSION = "zstd"
DEFAULT_COMPRESSION_LEVEL = 3
DEFAULT_PER_INSTANCE_SEED_MAX_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_PER_RUN_MAX_BYTES = 32 * 1024 * 1024 * 1024

_RUN_LABEL_RE = re.compile(
    r"^stage(?:[0-9]{2}|[0-9]{2}\.[0-9])_[a-z0-9_]+_(?:attempt|rerun)[0-9]{2}$"
)
_RUN_CONTEXT_RE = re.compile(
    r"^(?P<stage>stage(?:[0-9]{2}|[0-9]{2}\.[0-9]))_"
    r"(?P<component>[a-z0-9_]+)_(?:attempt|rerun)[0-9]{2}$"
)


class ArtifactStorageError(RuntimeError):
    """Base error for storage-policy violations."""


class ArtifactBudgetExceeded(ArtifactStorageError):
    """Raised after partial evidence is retained but a configured limit is hit."""


class ArtifactIntegrityError(ValueError):
    """Raised when a manifest, checksum, or schema record is invalid."""


def signed_sidecar_matches(path: Path, sidecar: Path | None = None) -> bool:
    """Accept a stable digest or a two-generation crash-transition sidecar."""

    sidecar_path = path.with_suffix(".sha256") if sidecar is None else sidecar
    if not path.is_file() or not sidecar_path.is_file():
        return False
    try:
        digests = tuple(
            line.strip()
            for line in sidecar_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, UnicodeDecodeError):
        return False
    if (
        not 1 <= len(digests) <= 2
        or len(set(digests)) != len(digests)
        or any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in digests
        )
    ):
        return False
    return _sha256(path) in digests


def atomic_write_signed_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    _replace: Callable[[Path, Path], None] = os.replace,
) -> tuple[Path, Path]:
    """Crash-consistently replace one JSON object and its SHA-256 sidecar."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    sidecar = path.with_suffix(".sha256")
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    final_sidecar = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.final.tmp")
    if temporary.exists() or temporary_sidecar.exists() or final_sidecar.exists():
        raise FileExistsError("artifact atomic-write temporary file already exists")
    previous_payload: bytes | None = None
    previous_sidecar: bytes | None = None
    if path.exists() != sidecar.exists():
        raise ArtifactIntegrityError(
            "signed JSON payload and sidecar must either both exist or both be absent"
        )
    if path.is_file() and sidecar.is_file():
        candidate_payload = path.read_bytes()
        candidate_sidecar = sidecar.read_bytes()
        if not signed_sidecar_matches(path, sidecar):
            raise ArtifactIntegrityError("existing signed JSON generation is invalid")
        previous_payload = candidate_payload
        previous_sidecar = candidate_sidecar
    previous_digest = (
        hashlib.sha256(previous_payload).hexdigest()
        if previous_payload is not None
        else None
    )
    transition_digests = tuple(
        dict.fromkeys(
            item for item in (previous_digest, digest) if item is not None
        )
    )
    payload_replaced = False
    sidecar_replaced = False
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary_sidecar.open("x", encoding="utf-8") as handle:
            handle.write("".join(f"{item}\n" for item in transition_digests))
            handle.flush()
            os.fsync(handle.fileno())
        _replace(temporary_sidecar, sidecar)
        sidecar_replaced = True
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _replace(temporary, path)
        payload_replaced = True
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        with final_sidecar.open("x", encoding="utf-8") as handle:
            handle.write(digest + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace(final_sidecar, sidecar)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        if payload_replaced or sidecar_replaced:
            if previous_payload is None or previous_sidecar is None:
                path.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
            else:
                rollback_payload = path.with_name(f".{path.name}.{os.getpid()}.rollback")
                rollback_sidecar = sidecar.with_name(
                    f".{sidecar.name}.{os.getpid()}.rollback"
                )
                try:
                    with rollback_payload.open("xb") as handle:
                        handle.write(previous_payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    with rollback_sidecar.open("xb") as handle:
                        handle.write(previous_sidecar)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(rollback_payload, path)
                    os.replace(rollback_sidecar, sidecar)
                    descriptor = os.open(path.parent, os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                finally:
                    rollback_payload.unlink(missing_ok=True)
                    rollback_sidecar.unlink(missing_ok=True)
        raise
    finally:
        if temporary.exists():
            temporary.unlink()
        if temporary_sidecar.exists():
            temporary_sidecar.unlink()
        if final_sidecar.exists():
            final_sidecar.unlink()
    return path, sidecar


@dataclass(frozen=True, slots=True)
class ArtifactStorageConfig:
    """Immutable storage policy used by all new experiment bundles."""

    enabled: bool = True
    storage_policy_version: str = ARTIFACT_STORAGE_SCHEMA_VERSION
    event_format: str = "parquet"
    compression: str = DEFAULT_COMPRESSION
    compression_level: int = DEFAULT_COMPRESSION_LEVEL
    critical_evidence: str = "full"
    diagnostic_evidence: str = "aggregate"
    screening_schema_version: str = ""
    per_instance_seed_max_bytes: int = DEFAULT_PER_INSTANCE_SEED_MAX_BYTES
    per_run_max_bytes: int = DEFAULT_PER_RUN_MAX_BYTES

    def __post_init__(self) -> None:
        if not self.screening_schema_version:
            object.__setattr__(
                self,
                "screening_schema_version",
                (
                    SCREENING_DECISIONS_V2
                    if self.storage_policy_version == ARTIFACT_STORAGE_V2
                    else SCREENING_DECISIONS_V1
                ),
            )
        if self.storage_policy_version not in SUPPORTED_STORAGE_POLICIES:
            raise ValueError(
                "unsupported artifact storage policy: "
                f"{self.storage_policy_version}; expected one of "
                f"{sorted(SUPPORTED_STORAGE_POLICIES)}"
            )
        if self.event_format != "parquet":
            raise ValueError("new experiment evidence must use parquet event storage")
        if self.compression != DEFAULT_COMPRESSION:
            raise ValueError("artifact storage currently supports only zstd compression")
        if not 1 <= self.compression_level <= 22:
            raise ValueError("zstd compression level must be between 1 and 22")
        if self.critical_evidence != "full":
            raise ValueError("critical evidence must use the full retention policy")
        if self.diagnostic_evidence != "aggregate":
            raise ValueError("diagnostic evidence must use the aggregate retention policy")
        if self.screening_schema_version not in SUPPORTED_SCREENING_SCHEMAS:
            raise ValueError(
                "unsupported physical screening schema: "
                f"{self.screening_schema_version}; expected one of "
                f"{sorted(SUPPORTED_SCREENING_SCHEMAS)}"
            )
        expected_screening_schemas = (
            {SCREENING_DECISIONS_V2, SCREENING_DECISIONS_V3}
            if self.storage_policy_version == ARTIFACT_STORAGE_V2
            else {SCREENING_DECISIONS_V1, SCREENING_DECISIONS_V2}
        )
        if self.screening_schema_version not in expected_screening_schemas:
            raise ValueError(
                f"{self.storage_policy_version} does not support {self.screening_schema_version}"
            )
        if self.per_instance_seed_max_bytes <= 0 or self.per_run_max_bytes <= 0:
            raise ValueError("artifact byte budgets must be positive")
        if self.per_instance_seed_max_bytes > self.per_run_max_bytes:
            raise ValueError("per-instance budget cannot exceed the run budget")

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "storage_policy_version": self.storage_policy_version,
            "event_format": self.event_format,
            "compression": self.compression,
            "compression_level": self.compression_level,
            "critical_evidence": self.critical_evidence,
            "diagnostic_evidence": self.diagnostic_evidence,
            "screening_schema_version": self.screening_schema_version,
            "per_instance_seed_max_bytes": self.per_instance_seed_max_bytes,
            "per_run_max_bytes": self.per_run_max_bytes,
        }


@dataclass(frozen=True, slots=True)
class ArtifactRunContext:
    stage_id: str
    component: str
    run_label: str

    def __post_init__(self) -> None:
        if not _RUN_LABEL_RE.fullmatch(self.run_label):
            raise ValueError(f"non-canonical run label: {self.run_label}")
        if not self.stage_id or not self.component:
            raise ValueError("stage_id and component are required")


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    relative_path: str
    artifact_type: str
    retention_class: str
    storage_format: str
    compression: str
    evidence_completeness: str
    checksum: str
    byte_size: int
    row_count: int | None = None
    schema_fingerprint: str = ""
    artifact_subtype: str = ""
    storage_policy_version: str = ARTIFACT_STORAGE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "artifact_type": self.artifact_type,
            "artifact_subtype": self.artifact_subtype,
            "retention_class": self.retention_class,
            "storage_format": self.storage_format,
            "compression": self.compression,
            "evidence_completeness": self.evidence_completeness,
            "checksum": self.checksum,
            "byte_size": self.byte_size,
            "row_count": self.row_count,
            "schema_fingerprint": self.schema_fingerprint,
            "storage_policy_version": self.storage_policy_version,
        }


@dataclass(frozen=True, slots=True)
class ArtifactBundleResult:
    run_dir: Path
    manifest_path: Path
    manifest_sidecar_path: Path
    artifacts: tuple[ArtifactReference, ...]
    evidence_completeness: str


@dataclass(frozen=True, slots=True)
class ArtifactReadResult:
    """Verified view of one artifact bundle."""

    run_dir: Path
    manifest_path: Path
    storage_format: str
    manifest: Mapping[str, Any]


def canonical_run_label_is_valid(run_label: str) -> bool:
    return _RUN_LABEL_RE.fullmatch(run_label) is not None


def artifact_context_from_run_label(run_label: str) -> ArtifactRunContext:
    match = _RUN_CONTEXT_RE.fullmatch(run_label)
    if match is None:
        raise ValueError(f"non-canonical run label: {run_label}")
    return ArtifactRunContext(
        stage_id=match.group("stage"),
        component=match.group("component"),
        run_label=run_label,
    )


def require_current_storage_config(
    run_label: str,
    config: ArtifactStorageConfig | None,
) -> ArtifactStorageConfig:
    if not canonical_run_label_is_valid(run_label):
        raise ValueError("new experiment runs must use a canonical attemptNN/rerunNN run label")
    if config is None or not config.enabled:
        raise ValueError("new experiment runs require an enabled [artifact_storage] configuration")
    return config


EVENTS_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.int64(), nullable=False),
        pa.field("record_type", pa.string(), nullable=False),
        pa.field("event_type", pa.string()),
        pa.field("timestamp_seconds", pa.float64()),
        pa.field("started_at", pa.float64()),
        pa.field("completed_at", pa.float64()),
        pa.field("duration_seconds", pa.float64()),
        pa.field("lane_id", pa.int32(), nullable=False),
        pa.field("iteration", pa.int64()),
        pa.field("operator_id", pa.int32(), nullable=False),
        pa.field("route_id", pa.int64()),
        pa.field("route_ids", pa.list_(pa.int64())),
        pa.field("current_route_ids", pa.list_(pa.int64())),
        pa.field("candidate_route_ids", pa.list_(pa.int64())),
        pa.field("base_route_id", pa.int64()),
        pa.field("candidate_route_id", pa.int64()),
        pa.field("status", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("operation", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("failure_reason", pa.string()),
        pa.field("feasible", pa.bool_()),
        pa.field("exact_started", pa.bool_()),
        pa.field("exact_completed", pa.bool_()),
        pa.field("candidate_feasible", pa.bool_()),
        pa.field("accepted", pa.bool_()),
        pa.field("global_best", pa.bool_()),
        pa.field("current_vehicle_count", pa.int64()),
        pa.field("candidate_vehicle_count", pa.int64()),
        pa.field("candidate_vehicle_delta", pa.int64()),
        pa.field("cache_key_digest", pa.string()),
        pa.field("evaluation_id", pa.int64()),
        pa.field("decision_id", pa.int64()),
        pa.field("route_change_status", pa.string()),
        pa.field("propagation_status", pa.string()),
        pa.field("extras_json", pa.string()),
    ]
)

ROUTE_DICTIONARY_SCHEMA = pa.schema(
    [
        pa.field("route_id", pa.int64(), nullable=False),
        pa.field("canonical_route_key", pa.string(), nullable=False),
        pa.field("route_digest", pa.string(), nullable=False),
        pa.field("customer_sequence", pa.list_(pa.string()), nullable=False),
    ]
)

SCREENING_CHECKS_SCHEMA = pa.schema(
    [
        pa.field("decision_event_id", pa.int64(), nullable=False),
        pa.field("decision_id", pa.int64()),
        pa.field("check_index", pa.int64(), nullable=False),
        pa.field("check", pa.string(), nullable=False),
        pa.field("status", pa.string()),
        pa.field("value_bool", pa.bool_()),
        pa.field("value_float", pa.float64()),
        pa.field("value_text", pa.string()),
        pa.field("reason", pa.string()),
    ]
)

V2_SCREENING_DECISIONS_SCHEMA_V1 = pa.schema(
    [
        pa.field("event_id", pa.int64(), nullable=False),
        pa.field("timestamp_seconds", pa.float64()),
        pa.field("started_at", pa.float64()),
        pa.field("completed_at", pa.float64()),
        pa.field("duration_seconds", pa.float64()),
        pa.field("lane_id", pa.int32(), nullable=False),
        pa.field("iteration", pa.int64()),
        pa.field("operator_id", pa.int32(), nullable=False),
        pa.field("route_id", pa.int64()),
        pa.field("status", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("decision_id", pa.int64()),
        pa.field("benchmark_axis", pa.string(), nullable=False),
        pa.field("demand", pa.float64()),
        pa.field("distance_increment_lower_bound", pa.float64()),
        pa.field("distance_lower_bound", pa.float64()),
        pa.field("exact_call_blocked", pa.bool_()),
        pa.field("first_failed_check", pa.string()),
        pa.field("min_time_window_slack", pa.float64()),
        pa.field("negative_cache_hit", pa.bool_()),
        pa.field("single_segment_reachable", pa.bool_()),
        pa.field("structural_energy_lower_bound", pa.float64()),
        pa.field(
            "checks",
            pa.list_(
                pa.struct(
                    [
                        pa.field("check", pa.string(), nullable=False),
                        pa.field("status", pa.string()),
                        pa.field("value_bool", pa.bool_()),
                        pa.field("value_float", pa.float64()),
                        pa.field("value_text", pa.string()),
                        pa.field("reason", pa.string()),
                    ]
                )
            ),
        ),
    ]
)

V2_SCREENING_DECISIONS_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.int64(), nullable=False),
        pa.field("definition_id", pa.int64(), nullable=False),
        pa.field("definition_json", pa.binary()),
        pa.field("started_at", pa.float64()),
        pa.field("completed_at", pa.float64()),
        pa.field("iteration", pa.int64()),
        pa.field("decision_id", pa.int64()),
    ]
)

V3_SCREENING_DEFINITIONS_SCHEMA = pa.schema(
    [
        pa.field("definition_id", pa.int64(), nullable=False),
        pa.field("lane_id", pa.int32(), nullable=False),
        pa.field("operator_id", pa.int32(), nullable=False),
        pa.field("route_id", pa.int64()),
        pa.field("status", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("benchmark_axis", pa.string(), nullable=False),
        pa.field("demand", pa.float64()),
        pa.field("distance_increment_lower_bound", pa.float64()),
        pa.field("distance_lower_bound", pa.float64()),
        pa.field("exact_call_blocked", pa.bool_()),
        pa.field("first_failed_check", pa.string()),
        pa.field("min_time_window_slack", pa.float64()),
        pa.field("negative_cache_hit", pa.bool_()),
        pa.field("single_segment_reachable", pa.bool_()),
        pa.field("structural_energy_lower_bound", pa.float64()),
        pa.field(
            "checks",
            pa.list_(
                pa.struct(
                    [
                        pa.field("check", pa.string(), nullable=False),
                        pa.field("status", pa.string()),
                        pa.field("value_bool", pa.bool_()),
                        pa.field("value_float", pa.float64()),
                        pa.field("value_text", pa.string()),
                        pa.field("reason", pa.string()),
                    ]
                )
            ),
            nullable=False,
        ),
    ]
)

V3_SCREENING_OCCURRENCES_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.int64(), nullable=False),
        pa.field("definition_id", pa.int64(), nullable=False),
        pa.field("started_at", pa.float64()),
        pa.field("completed_at", pa.float64()),
        pa.field("iteration", pa.int64()),
        pa.field("decision_id", pa.int64()),
    ]
)

DIAGNOSTIC_SCHEMA = pa.schema(
    [
        pa.field("run_label", pa.string(), nullable=False),
        pa.field("instance", pa.string()),
        pa.field("seed", pa.int64()),
        pa.field("lane", pa.string()),
        pa.field("iteration", pa.int64()),
        pa.field("operator", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("count", pa.int64()),
        pa.field("sum_value", pa.float64()),
        pa.field("min_value", pa.float64()),
        pa.field("max_value", pa.float64()),
    ]
)

ANYTIME_CHECKPOINT_SCHEMA = pa.schema(
    [
        pa.field("instance", pa.string(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("axis_budget_seconds", pa.int64(), nullable=False),
        pa.field("checkpoint_seconds", pa.int64(), nullable=False),
        pa.field(
            "objective_key",
            pa.struct(
                [
                    pa.field("vehicle_count", pa.int64(), nullable=False),
                    pa.field("total_distance", pa.float64(), nullable=False),
                    pa.field("total_charging_time", pa.float64(), nullable=False),
                    pa.field("charging_count", pa.int64(), nullable=False),
                ]
            ),
            nullable=False,
        ),
        pa.field("source", pa.string(), nullable=False),
        pa.field("incumbent_completed_at_seconds", pa.float64(), nullable=False),
        pa.field("incumbent_iteration", pa.int64()),
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _json_read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"cannot read artifact JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError(f"artifact JSON must contain an object: {path}")
    return payload


def _canonical_filename(
    context: ArtifactRunContext,
    artifact_type: str,
    instance: str | None = None,
    seed: int | None = None,
    extension: str = "json",
) -> str:
    suffix = "" if instance is None else f"_{instance}_{seed}"
    return f"{context.run_label}_{artifact_type}{suffix}.{extension}"


def _safe_artifact_path(run_dir: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactIntegrityError(
            f"artifact path must remain inside the run directory: {relative_path}"
        )
    return run_dir / relative


def _is_appledouble_path(path: Path) -> bool:
    """Identify macOS AppleDouble metadata created by non-APFS volumes."""

    return any(part.startswith("._") for part in path.parts)


def _schema_fingerprint(schema: pa.Schema) -> str:
    # Arrow may rename the child field of a list from ``item`` to
    # ``element`` while round-tripping through Parquet.  Fingerprint the
    # logical field contract rather than that non-semantic display detail.
    fields = [
        {
            "name": field.name,
            "type": str(field.type).replace("element", "item"),
            "nullable": field.nullable,
        }
        for field in schema
    ]
    return _payload_sha256(fields)


def artifact_schema_fingerprint(schema: pa.Schema) -> str:
    """Return the canonical logical fingerprint used by artifact descriptors."""

    return _schema_fingerprint(schema)


def _as_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _as_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _route_id(value: object, route_ids: Mapping[str, int]) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        return route_ids.get(value)
    return _as_int(value)


def _route_ids(value: object, route_ids: Mapping[str, int]) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    output: list[int] = []
    for item in value:
        route_id = _route_id(item, route_ids)
        if route_id is not None:
            output.append(route_id)
    return output


def _event_timestamp(event: Mapping[str, object]) -> float | None:
    for key in ("timestamp_seconds", "started_at", "timestamp"):
        value = event.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _event_type(event: Mapping[str, object]) -> str:
    value = event.get("event_type", event.get("record_type", "event"))
    return str(value)


_EVENT_KNOWN_FIELDS = frozenset(
    {
        "event_id",
        "record_type",
        "event_type",
        "timestamp_seconds",
        "started_at",
        "completed_at",
        "duration_seconds",
        "lane",
        "iteration",
        "operator",
        "route_key",
        "route_keys",
        "base_route_key",
        "candidate_route_key",
        "route_id",
        "route_ids",
        "current_route_keys",
        "candidate_route_keys",
        "current_route_ids",
        "candidate_route_ids",
        "base_route_id",
        "candidate_route_id",
        "status",
        "kind",
        "operation",
        "reason",
        "failure_reason",
        "feasible",
        "exact_started",
        "exact_completed",
        "candidate_feasible",
        "accepted",
        "global_best",
        "current_vehicle_count",
        "candidate_vehicle_count",
        "candidate_vehicle_delta",
        "cache_key_digest",
        "evaluation_id",
        "decision_id",
        "route_change_status",
        "propagation_status",
        "checks",
        "event_index",
        "record_class",
    }
)

_NEIGHBORHOOD_EXTRA_FIELDS = (
    "aggregate_count",
    "benchmark_axis",
    "candidate_objective_key",
    "candidate_pool_hash",
    "chain_depth",
    "constraint_category",
    "distance_improvement",
    "exact_route_evaluations",
    "new_routes_created",
    "prefilter_passed",
    "ranking_score",
    "removal_size_actual",
    "removal_size_requested",
    "removal_tier",
    "removal_trigger",
    "reset_observed",
    "segment_length",
    "selection_rank",
    "stagnation_iterations",
    "track",
    "vehicle_reduction",
)
_NEIGHBORHOOD_EXCLUDED_COLLECTION_FIELDS = frozenset(
    {
        "affected_route_indices",
        "candidate_customer_sequence",
        "candidate_route_sequences",
        "removed_customers",
        "route_indices",
    }
)
_MISSING_NEIGHBORHOOD_EXTRA = object()
_NEIGHBORHOOD_EXTRAS_CACHE_ENTRIES = 65_536
_NEIGHBORHOOD_TEMPLATE_FIELDS = (
    "record_type",
    "timestamp_seconds",
    "started_at",
    "completed_at",
    "duration_seconds",
    "lane",
    "operator",
    "status",
    "kind",
    "operation",
    "reason",
    "failure_reason",
    "feasible",
    "exact_started",
    "exact_completed",
    "candidate_feasible",
    "accepted",
    "global_best",
    "current_vehicle_count",
    "candidate_vehicle_count",
    "candidate_vehicle_delta",
    "cache_key_digest",
    "evaluation_id",
    "decision_id",
    "route_change_status",
    "propagation_status",
)
_NEIGHBORHOOD_TEMPLATE_CACHE_ENTRIES = 65_536
_NEIGHBORHOOD_CORE_FIELDS = frozenset(
    {
        "event_id",
        "record_type",
        "event_type",
        "timestamp_seconds",
        "started_at",
        "completed_at",
        "duration_seconds",
        "lane",
        "iteration",
        "operator",
        "status",
        "kind",
        "operation",
        "reason",
        "failure_reason",
        "feasible",
        "exact_started",
        "exact_completed",
        "candidate_feasible",
        "accepted",
        "global_best",
        "current_vehicle_count",
        "candidate_vehicle_count",
        "candidate_vehicle_delta",
        "cache_key_digest",
        "evaluation_id",
        "decision_id",
        "route_change_status",
        "propagation_status",
    }
)
_NEIGHBORHOOD_FAST_FIELDS = (
    _NEIGHBORHOOD_CORE_FIELDS
    | frozenset(_NEIGHBORHOOD_EXTRA_FIELDS)
    | _NEIGHBORHOOD_EXCLUDED_COLLECTION_FIELDS
)
_SPARSE_EVENT_COMMON_FIELDS = frozenset(
    {
        "benchmark_axis",
        "cache_key_digest",
        "completed_at",
        "duration_seconds",
        "event_type",
        "iteration",
        "lane",
        "operator",
        "record_type",
        "route_key",
        "started_at",
        "status",
        "timestamp_seconds",
    }
)
_CACHE_EVENT_EXTRA_FIELDS = (
    "benchmark_axis",
    "current_bytes",
    "current_entries",
    "entry_bytes",
    "lookup_current_bytes",
    "lookup_current_entries",
    "lookup_result",
)
_CACHE_EVENT_FAST_FIELDS = _SPARSE_EVENT_COMMON_FIELDS | frozenset(
    {
        "operation",
        *_CACHE_EVENT_EXTRA_FIELDS,
    }
)
_ROUTE_EVALUATION_EXTRA_FIELDS = (
    "benchmark_axis",
    "deadline_boundary",
    "labels_expanded",
    "labels_generated",
    "labels_pruned",
)
_ROUTE_EVALUATION_FAST_FIELDS = _SPARSE_EVENT_COMMON_FIELDS | frozenset(
    {
        "evaluation_id",
        "exact_completed",
        "exact_started",
        "failure_reason",
        "feasible",
        "kind",
        "route_change_status",
        *_ROUTE_EVALUATION_EXTRA_FIELDS,
    }
)


def _normalise_neighborhood_event_values(
    event: Mapping[str, object],
    *,
    event_id: int,
    lane_ids: Mapping[str, int],
    operator_ids: Mapping[str, int],
    extras_cache: dict[tuple[object, ...], str] | None = None,
    row_cache: dict[tuple[object, ...], tuple[object, ...]] | None = None,
) -> tuple[object, ...]:
    get = event.get
    extra_values = tuple(
        get(key, _MISSING_NEIGHBORHOOD_EXTRA) for key in _NEIGHBORHOOD_EXTRA_FIELDS
    )
    row_cache_key = (
        tuple(get(key, _MISSING_NEIGHBORHOOD_EXTRA) for key in _NEIGHBORHOOD_TEMPLATE_FIELDS),
        extra_values,
    )
    cached_row: tuple[object, ...] | None = None
    if row_cache is not None:
        with contextlib.suppress(TypeError):
            cached_row = row_cache.get(row_cache_key)
    if cached_row is not None:
        return (
            event_id,
            *cached_row[1:8],
            _as_int(get("iteration")),
            *cached_row[9:],
        )
    status = str(get("status", ""))
    extras_json: str | None = None
    if extras_cache is not None:
        try:
            extras_json = extras_cache.get(extra_values)
        except TypeError:
            extras_json = None
    if extras_json is None:
        extras = {
            key: value
            for key, value in zip(_NEIGHBORHOOD_EXTRA_FIELDS, extra_values, strict=True)
            if value is not _MISSING_NEIGHBORHOOD_EXTRA
        }
        extras_json = _json_text(extras) if extras else ""
        if extras_cache is not None:
            with contextlib.suppress(TypeError):
                extras_cache[extra_values] = extras_json
            if len(extras_cache) > _NEIGHBORHOOD_EXTRAS_CACHE_ENTRIES:
                extras_cache.pop(next(iter(extras_cache)))
    row: tuple[object, ...] = (
        event_id,
        str(get("record_type", "neighborhood_event")),
        "neighborhood_event",
        _event_timestamp(event),
        _as_float(get("started_at")),
        _as_float(get("completed_at")),
        _as_float(get("duration_seconds")),
        lane_ids[str(get("lane", ""))],
        _as_int(get("iteration")),
        operator_ids[str(get("operator", ""))],
        None,
        [],
        [],
        [],
        None,
        None,
        status,
        str(get("kind", "")),
        str(get("operation", "")),
        str(get("reason", "")),
        str(get("failure_reason", "")),
        _as_bool(get("feasible")),
        _as_bool(get("exact_started")),
        _as_bool(get("exact_completed")),
        _as_bool(get("candidate_feasible")),
        _as_bool(get("accepted")),
        _as_bool(get("global_best")),
        _as_int(get("current_vehicle_count")),
        _as_int(get("candidate_vehicle_count")),
        _as_int(get("candidate_vehicle_delta")),
        str(get("cache_key_digest", "")),
        _as_int(get("evaluation_id")),
        _as_int(get("decision_id")),
        str(get("route_change_status", "")),
        str(get("propagation_status", status)),
        extras_json,
    )
    if row_cache is not None:
        with contextlib.suppress(TypeError):
            row_cache[row_cache_key] = row
        if len(row_cache) > _NEIGHBORHOOD_TEMPLATE_CACHE_ENTRIES:
            row_cache.pop(next(iter(row_cache)))
    return row


def _normalise_sparse_route_event_values(
    event: Mapping[str, object],
    *,
    event_id: int,
    route_ids: Mapping[str, int],
    lane_ids: Mapping[str, int],
    operator_ids: Mapping[str, int],
    extra_fields: Sequence[str],
) -> tuple[object, ...]:
    get = event.get
    event_type = str(get("event_type", ""))
    status = str(get("status", ""))
    extras = {key: get(key) for key in extra_fields if key in event}
    return (
        event_id,
        str(get("record_type", event_type)),
        event_type,
        _event_timestamp(event),
        _as_float(get("started_at")),
        _as_float(get("completed_at")),
        _as_float(get("duration_seconds")),
        lane_ids[str(get("lane", ""))],
        _as_int(get("iteration")),
        operator_ids[str(get("operator", ""))],
        _route_id(get("route_key"), route_ids),
        [],
        [],
        [],
        None,
        None,
        status,
        str(get("kind", "")),
        str(get("operation", "")),
        str(get("reason", "")),
        str(get("failure_reason", "")),
        _as_bool(get("feasible")),
        _as_bool(get("exact_started")),
        _as_bool(get("exact_completed")),
        None,
        None,
        None,
        None,
        None,
        None,
        str(get("cache_key_digest", "")),
        _as_int(get("evaluation_id")),
        None,
        str(get("route_change_status", "")),
        str(get("propagation_status", status)),
        _json_text(extras) if extras else "",
    )


def _normalise_event_values(
    event: Mapping[str, object],
    *,
    event_id: int,
    route_ids: Mapping[str, int],
    lane_ids: Mapping[str, int],
    operator_ids: Mapping[str, int],
    neighborhood_extras_cache: dict[tuple[object, ...], str] | None = None,
    neighborhood_row_cache: dict[
        tuple[object, ...], tuple[object, ...]
    ] | None = None,
) -> tuple[object, ...]:
    if (
        event.get("event_type", event.get("record_type")) == "neighborhood_event"
        and event.keys() <= _NEIGHBORHOOD_FAST_FIELDS
    ):
        return _normalise_neighborhood_event_values(
            event,
            event_id=event_id,
            lane_ids=lane_ids,
            operator_ids=operator_ids,
            extras_cache=neighborhood_extras_cache,
            row_cache=neighborhood_row_cache,
        )
    if (
        event.get("event_type") == "cache_event"
        and event.keys() <= _CACHE_EVENT_FAST_FIELDS
    ):
        return _normalise_sparse_route_event_values(
            event,
            event_id=event_id,
            route_ids=route_ids,
            lane_ids=lane_ids,
            operator_ids=operator_ids,
            extra_fields=_CACHE_EVENT_EXTRA_FIELDS,
        )
    if (
        event.get("event_type") == "route_evaluation"
        and event.keys() <= _ROUTE_EVALUATION_FAST_FIELDS
    ):
        return _normalise_sparse_route_event_values(
            event,
            event_id=event_id,
            route_ids=route_ids,
            lane_ids=lane_ids,
            operator_ids=operator_ids,
            extra_fields=_ROUTE_EVALUATION_EXTRA_FIELDS,
        )
    return _normalise_general_event_values(
        event,
        event_id=event_id,
        route_ids=route_ids,
        lane_ids=lane_ids,
        operator_ids=operator_ids,
    )


def _normalise_general_event_values(
    event: Mapping[str, object],
    *,
    event_id: int,
    route_ids: Mapping[str, int],
    lane_ids: Mapping[str, int],
    operator_ids: Mapping[str, int],
) -> tuple[object, ...]:
    get = event.get
    if get("event_type") == "screening_decision":
        row = _normalise_screening_event(
            event,
            event_id=event_id,
            route_ids=route_ids,
            lane_ids=lane_ids,
            operator_ids=operator_ids,
        )
        return tuple(row.get(name) for name in EVENTS_SCHEMA.names)
    route_id = _route_id(get("route_key", get("route_id")), route_ids)
    base_route_id = _route_id(get("base_route_key", get("base_route_id")), route_ids)
    candidate_route_id = _route_id(
        get("candidate_route_key", get("candidate_route_id")), route_ids
    )
    current_route_ids = _route_ids(
        get("current_route_keys", get("current_route_ids")), route_ids
    )
    candidate_route_ids = _route_ids(
        get("candidate_route_keys", get("candidate_route_ids")), route_ids
    )
    extras = {
        str(key): value
        for key, value in event.items()
        if key not in _EVENT_KNOWN_FIELDS
        and (
            not isinstance(value, (list, tuple, dict))
            or key
            in {
                "current_objective_key",
                "candidate_objective_key",
                "source_objective_key",
                "objective_key",
                "submission_order",
                "completion_order",
                "merge_order",
                "chunk_sizes",
                "completed_indices",
            }
        )
    }
    if get("record_class"):
        extras["record_class"] = event["record_class"]
    event_type = _event_type(event)
    return (
        event_id,
        str(get("record_type", event_type)),
        event_type,
        _event_timestamp(event),
        _as_float(get("started_at")),
        _as_float(get("completed_at")),
        _as_float(get("duration_seconds")),
        lane_ids[str(get("lane", ""))],
        _as_int(get("iteration")),
        operator_ids[str(get("operator", ""))],
        route_id,
        _route_ids(get("route_keys", get("route_ids")), route_ids),
        current_route_ids,
        candidate_route_ids,
        base_route_id,
        candidate_route_id,
        str(get("status", "")),
        str(get("kind", "")),
        str(get("operation", "")),
        str(get("reason", "")),
        str(get("failure_reason", "")),
        _as_bool(get("feasible")),
        _as_bool(get("exact_started")),
        _as_bool(get("exact_completed")),
        _as_bool(get("candidate_feasible")),
        _as_bool(get("accepted")),
        _as_bool(get("global_best")),
        _as_int(get("current_vehicle_count")),
        _as_int(get("candidate_vehicle_count")),
        _as_int(get("candidate_vehicle_delta")),
        str(get("cache_key_digest", "")),
        _as_int(get("evaluation_id")),
        _as_int(get("decision_id")),
        str(get("route_change_status", "")),
        str(get("propagation_status", get("status", ""))),
        _json_text(extras) if extras else "",
    )


def _normalise_event(
    event: Mapping[str, object],
    *,
    event_id: int,
    route_ids: Mapping[str, int],
    lane_ids: Mapping[str, int],
    operator_ids: Mapping[str, int],
) -> dict[str, object]:
    values = _normalise_event_values(
        event,
        event_id=event_id,
        route_ids=route_ids,
        lane_ids=lane_ids,
        operator_ids=operator_ids,
    )
    return dict(zip(EVENTS_SCHEMA.names, values, strict=True))


_SCREENING_EVENT_TEMPLATE: dict[str, object] = {
    "route_ids": [],
    "current_route_ids": [],
    "candidate_route_ids": [],
    "base_route_id": None,
    "candidate_route_id": None,
    "kind": "",
    "operation": "",
    "failure_reason": "",
    "feasible": None,
    "exact_started": None,
    "exact_completed": None,
    "candidate_feasible": None,
    "accepted": None,
    "global_best": None,
    "current_vehicle_count": None,
    "candidate_vehicle_count": None,
    "candidate_vehicle_delta": None,
    "cache_key_digest": "",
    "evaluation_id": None,
    "route_change_status": "",
}


def _normalise_screening_event(
    event: Mapping[str, object],
    *,
    event_id: int,
    route_ids: Mapping[str, int],
    lane_ids: Mapping[str, int],
    operator_ids: Mapping[str, int],
) -> dict[str, object]:
    extras = {
        key: event.get(key)
        for key in (
            "benchmark_axis",
            "demand",
            "distance_increment_lower_bound",
            "distance_lower_bound",
            "exact_call_blocked",
            "first_failed_check",
            "min_time_window_slack",
            "negative_cache_hit",
            "single_segment_reachable",
            "structural_energy_lower_bound",
        )
        if key in event
    }
    row = _SCREENING_EVENT_TEMPLATE.copy()
    row.update(
        {
            "event_id": event_id,
            "record_type": "screening_decision",
            "event_type": "screening_decision",
            "timestamp_seconds": _as_float(event.get("started_at")),
            "started_at": _as_float(event.get("started_at")),
            "completed_at": _as_float(event.get("completed_at")),
            "duration_seconds": _as_float(event.get("duration_seconds")),
            "lane_id": lane_ids[str(event.get("lane", ""))],
            "iteration": _as_int(event.get("iteration")),
            "operator_id": operator_ids[str(event.get("operator", ""))],
            "route_id": _route_id(event.get("route_key"), route_ids),
            "status": str(event.get("status", "")),
            "reason": str(event.get("reason", "")),
            "decision_id": _as_int(event.get("decision_id")),
            "propagation_status": str(event.get("status", "")),
            "extras_json": _json_text(extras) if extras else "",
        }
    )
    return row


def _compact_screening_check_key(check: Mapping[str, object]) -> tuple[object, ...]:
    value = check.get("value")
    already_compact = any(
        field in check for field in ("value_bool", "value_float", "value_text")
    )
    return (
        str(check.get("check", "")),
        str(check.get("status", "")),
        _as_bool(check.get("value_bool"))
        if already_compact
        else value
        if isinstance(value, bool)
        else None,
        _as_float(check.get("value_float"))
        if already_compact
        else float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None,
        str(check["value_text"])
        if already_compact and check.get("value_text") is not None
        else value
        if isinstance(value, str)
        else None,
        str(check.get("reason", "")),
    )


def _normalise_compact_screening_checks(
    checks: Sequence[object],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        name, status, value_bool, value_float, value_text, reason = (
            _compact_screening_check_key(check)
        )
        output.append(
            {
                "check": name,
                "status": status,
                "value_bool": value_bool,
                "value_float": value_float,
                "value_text": value_text,
                "reason": reason,
            }
        )
    return output


@dataclass(frozen=True, slots=True)
class _PrecomputedScreeningDefinition:
    """Trusted typed cache-key tail produced by the live measurement bridge."""

    tail: tuple[object, ...]
    cache_hash: int = dataclass_field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_hash", hash(self.tail))


@dataclass(frozen=True, slots=True)
class _PendingScreeningDefinition:
    definition_id: int
    encoded: bytes
    payload: dict[str, object]
    row: tuple[object, ...]


type _BufferedScreeningDecision = tuple[
    int,
    str,
    str,
    int | None,
    str,
    float,
    float,
    _PrecomputedScreeningDefinition,
]


def _screening_definition_cache_key(
    payload: Mapping[str, object],
    *,
    lane_id: int,
    operator_id: int,
    route_id: int | None,
) -> tuple[object, ...]:
    precomputed = payload.get("_precomputed_screening_definition")
    if isinstance(precomputed, _PrecomputedScreeningDefinition):
        return (lane_id, operator_id, route_id, *precomputed.tail)
    raw_checks = payload.get("checks")
    checks = raw_checks if isinstance(raw_checks, (list, tuple)) else ()
    if len(checks) > MAX_SCREENING_CHECKS_PER_DECISION:
        raise ArtifactIntegrityError(
            "one screening decision exceeds the fixed eight-check domain"
        )
    return (
        lane_id,
        operator_id,
        route_id,
        str(payload.get("status") or ""),
        str(payload.get("reason") or ""),
        str(payload.get("benchmark_axis") or ""),
        _as_float(payload.get("demand")),
        _as_float(payload.get("distance_increment_lower_bound")),
        _as_float(payload.get("distance_lower_bound")),
        _as_bool(payload.get("exact_call_blocked")),
        str(payload.get("first_failed_check") or ""),
        _as_float(payload.get("min_time_window_slack")),
        _as_bool(payload.get("negative_cache_hit")),
        _as_bool(payload.get("single_segment_reachable")),
        _as_float(payload.get("structural_energy_lower_bound")),
        tuple(
            _compact_screening_check_key(check)
            for check in checks
            if isinstance(check, Mapping)
        ),
    )


def _screening_definition_from_cache_key(
    definition_key: tuple[object, ...],
) -> dict[str, object]:
    raw_checks = definition_key[15]
    if not isinstance(raw_checks, tuple):
        raise ArtifactIntegrityError("screening definition cache key checks are invalid")
    checks: list[dict[str, object]] = []
    for raw_check in raw_checks:
        if not isinstance(raw_check, tuple) or len(raw_check) != 6:
            raise ArtifactIntegrityError("screening definition cache check is invalid")
        name, status, value_bool, value_float, value_text, reason = raw_check
        checks.append(
            {
                "check": name,
                "status": status,
                "value_bool": value_bool,
                "value_float": value_float,
                "value_text": value_text,
                "reason": reason,
            }
        )
    return {
        "lane_id": definition_key[0],
        "operator_id": definition_key[1],
        "route_id": definition_key[2],
        "status": definition_key[3],
        "reason": definition_key[4],
        "benchmark_axis": definition_key[5],
        "demand": definition_key[6],
        "distance_increment_lower_bound": definition_key[7],
        "distance_lower_bound": definition_key[8],
        "exact_call_blocked": definition_key[9],
        "first_failed_check": definition_key[10],
        "min_time_window_slack": definition_key[11],
        "negative_cache_hit": definition_key[12],
        "single_segment_reachable": definition_key[13],
        "structural_energy_lower_bound": definition_key[14],
        "checks": checks,
    }


def _normalise_screening_definition(
    payload: Mapping[str, object],
    *,
    lane_id: int | None = None,
    operator_id: int | None = None,
    route_id: int | None = None,
) -> dict[str, object]:
    raw_checks = payload.get("checks")
    checks = raw_checks if isinstance(raw_checks, (list, tuple)) else ()
    if len(checks) > MAX_SCREENING_CHECKS_PER_DECISION:
        raise ArtifactIntegrityError(
            "one screening decision exceeds the fixed eight-check domain"
        )
    resolved_lane_id = lane_id if lane_id is not None else _as_int(payload.get("lane_id"))
    resolved_operator_id = (
        operator_id if operator_id is not None else _as_int(payload.get("operator_id"))
    )
    if resolved_lane_id is None or resolved_operator_id is None:
        raise ArtifactIntegrityError("screening definition requires lane and operator IDs")
    return {
        "lane_id": resolved_lane_id,
        "operator_id": resolved_operator_id,
        "route_id": route_id if route_id is not None else _as_int(payload.get("route_id")),
        "status": str(payload.get("status") or ""),
        "reason": str(payload.get("reason") or ""),
        "benchmark_axis": str(payload.get("benchmark_axis") or ""),
        "demand": _as_float(payload.get("demand")),
        "distance_increment_lower_bound": _as_float(payload.get("distance_increment_lower_bound")),
        "distance_lower_bound": _as_float(payload.get("distance_lower_bound")),
        "exact_call_blocked": _as_bool(payload.get("exact_call_blocked")),
        "first_failed_check": str(payload.get("first_failed_check") or ""),
        "min_time_window_slack": _as_float(payload.get("min_time_window_slack")),
        "negative_cache_hit": _as_bool(payload.get("negative_cache_hit")),
        "single_segment_reachable": _as_bool(payload.get("single_segment_reachable")),
        "structural_energy_lower_bound": _as_float(payload.get("structural_energy_lower_bound")),
        "checks": _normalise_compact_screening_checks(checks),
    }


def _screening_definition_identity(definition: Mapping[str, object]) -> tuple[int, bytes, str]:
    definition_json = orjson.dumps(definition, option=orjson.OPT_SORT_KEYS)
    digest = hashlib.sha256(definition_json).digest()
    definition_id = int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
    return definition_id, definition_json, digest.hex()


class _BoundedScreeningDefinitionStore:
    """Exact bounded definition identity store with fail-fast disk spill."""

    def __init__(self, *, cache_entries: int, scratch_root: Path | None = None) -> None:
        if cache_entries <= 0:
            raise ValueError("definition cache_entries must be positive")
        if scratch_root is not None:
            scratch_root.mkdir(parents=True, exist_ok=True)
        self._scratch_root = scratch_root
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="evrptw-screening-definitions-",
            dir=scratch_root,
        )
        self._connection: sqlite3.Connection | None = None
        self._encoded_memory: dict[int, bytes] = {}
        self._encoded_memory_entries = SCREENING_DEFINITION_HOT_CACHE_ENTRIES
        self._cache_entries = cache_entries
        self._cache: OrderedDict[int, dict[str, object]] = OrderedDict()
        self._closed = False

    def __enter__(self) -> _BoundedScreeningDefinitionStore:
        return self

    def __exit__(
        self,
        _exception_type: object,
        _exception: object,
        _traceback: object,
    ) -> None:
        self.close()

    def register(
        self,
        definition_id: int,
        definition: Mapping[str, object],
        *,
        allow_identical_existing: bool,
        encoded: bytes | None = None,
    ) -> bool:
        if encoded is None:
            expected_id, encoded, _ = _screening_definition_identity(definition)
        else:
            digest = hashlib.sha256(encoded).digest()
            expected_id = int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
        if expected_id != definition_id:
            raise ArtifactIntegrityError("screening definition hash mismatch")
        if self._connection is None:
            existing_payload = self._encoded_memory.get(definition_id)
            if existing_payload is not None:
                if existing_payload != encoded:
                    raise ArtifactIntegrityError("screening definition ID collision")
                if not allow_identical_existing:
                    raise ArtifactIntegrityError("duplicate screening definition ID")
                self._remember(definition_id, dict(definition))
                return False
            if len(self._encoded_memory) < self._encoded_memory_entries:
                self._encoded_memory[definition_id] = encoded
                self._remember(definition_id, dict(definition))
                return True
            self._spill_encoded_memory()
        connection = self._require_connection()
        inserted = (
            connection.execute(
                "INSERT OR IGNORE INTO definitions(definition_id, payload) VALUES (?, ?)",
                (definition_id, encoded),
            ).rowcount
            == 1
        )
        if not inserted:
            existing = connection.execute(
                "SELECT payload FROM definitions WHERE definition_id = ?",
                (definition_id,),
            ).fetchone()
            if existing is None:
                raise ArtifactIntegrityError("screening definition insert was not observable")
            existing_payload = bytes(existing[0])
            if existing_payload != encoded:
                raise ArtifactIntegrityError("screening definition ID collision")
            if not allow_identical_existing:
                raise ArtifactIntegrityError("duplicate screening definition ID")
        self._remember(definition_id, dict(definition))
        return inserted

    def register_many(
        self,
        definitions: Sequence[_PendingScreeningDefinition],
    ) -> frozenset[int]:
        """Register one bounded transaction with batched collision checks."""

        unique: dict[int, _PendingScreeningDefinition] = {}
        for definition in definitions:
            previous = unique.get(definition.definition_id)
            if previous is not None and previous.encoded != definition.encoded:
                raise ArtifactIntegrityError("screening definition ID collision")
            unique[definition.definition_id] = definition
        if self._connection is None:
            existing_memory = {
                definition_id: self._encoded_memory[definition_id]
                for definition_id in unique
                if definition_id in self._encoded_memory
            }
            for definition_id, payload in existing_memory.items():
                if payload != unique[definition_id].encoded:
                    raise ArtifactIntegrityError("screening definition ID collision")
            inserted_memory = tuple(
                definition
                for definition_id, definition in unique.items()
                if definition_id not in existing_memory
            )
            if (
                len(self._encoded_memory) + len(inserted_memory)
                <= self._encoded_memory_entries
            ):
                self._encoded_memory.update(
                    {
                        definition.definition_id: definition.encoded
                        for definition in inserted_memory
                    }
                )
                return frozenset(
                    definition.definition_id for definition in inserted_memory
                )
            self._spill_encoded_memory()
        connection = self._require_connection()
        existing: dict[int, bytes] = {}
        identifiers = tuple(unique)
        for offset in range(0, len(identifiers), 900):
            chunk = identifiers[offset : offset + 900]
            placeholders = ",".join("?" for _ in chunk)
            for definition_id, payload in connection.execute(
                f"SELECT definition_id, payload FROM definitions "  # noqa: S608
                f"WHERE definition_id IN ({placeholders})",
                chunk,
            ):
                existing[int(definition_id)] = bytes(payload)
        for definition_id, payload in existing.items():
            if payload != unique[definition_id].encoded:
                raise ArtifactIntegrityError("screening definition ID collision")
        inserted = tuple(
            definition
            for definition_id, definition in unique.items()
            if definition_id not in existing
        )
        try:
            connection.executemany(
                "INSERT INTO definitions(definition_id, payload) VALUES (?, ?)",
                (
                    (definition.definition_id, definition.encoded)
                    for definition in inserted
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ArtifactIntegrityError("screening definition batch insert failed") from error
        return frozenset(definition.definition_id for definition in inserted)

    def resolve(self, definition_id: int) -> dict[str, object]:
        cached = self._cache.get(definition_id)
        if cached is not None:
            self._cache.move_to_end(definition_id)
            return cached
        encoded = self._encoded_memory.get(definition_id)
        if encoded is not None:
            stored_payload = encoded
        else:
            connection = self._connection
            if connection is None:
                raise ArtifactIntegrityError(
                    "screening row references an unknown definition"
                )
            stored = connection.execute(
            "SELECT payload FROM definitions WHERE definition_id = ?",
            (definition_id,),
            ).fetchone()
            if stored is None:
                raise ArtifactIntegrityError(
                    "screening row references an unknown definition"
                )
            stored_payload = bytes(stored[0])
        try:
            decoded = orjson.loads(stored_payload)
        except orjson.JSONDecodeError as error:
            raise ArtifactIntegrityError("stored screening definition is invalid") from error
        if not isinstance(decoded, dict):
            raise ArtifactIntegrityError("stored screening definition must be an object")
        definition = cast(dict[str, object], decoded)
        self._remember(definition_id, definition)
        return definition

    def close(self) -> None:
        if self._closed:
            return
        if self._connection is not None:
            self._connection.close()
        self._temporary_directory.cleanup()
        self._encoded_memory.clear()
        self._cache.clear()
        self._closed = True

    def _remember(self, definition_id: int, definition: dict[str, object]) -> None:
        self._cache[definition_id] = definition
        self._cache.move_to_end(definition_id)
        if len(self._cache) > self._cache_entries:
            self._cache.popitem(last=False)

    def _spill_encoded_memory(self) -> None:
        if self._connection is not None:
            return
        database_path = Path(self._temporary_directory.name) / "definitions.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA cache_size=-2048")
            connection.execute(
                "CREATE TABLE definitions "
                "(definition_id INTEGER PRIMARY KEY, payload BLOB NOT NULL)"
            )
            connection.executemany(
                "INSERT INTO definitions(definition_id, payload) VALUES (?, ?)",
                self._encoded_memory.items(),
            )
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        self._encoded_memory.clear()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("screening definition store has not spilled to disk")
        return self._connection


def _json_text(payload: Mapping[str, object]) -> str:
    return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def _screening_definition_id_from_row(row: Mapping[str, object]) -> int:
    raw_definition_id = row.get("definition_id")
    if isinstance(raw_definition_id, bool) or not isinstance(raw_definition_id, int):
        raise ArtifactIntegrityError("screening definition_id must be an integer")
    return raw_definition_id


def _decode_screening_definition_json(value: object) -> dict[str, object]:
    if not isinstance(value, (bytes, bytearray, memoryview, str)):
        raise ArtifactIntegrityError("screening definition_json must be binary or text")
    try:
        definition = orjson.loads(value)
    except orjson.JSONDecodeError as error:
        raise ArtifactIntegrityError("screening definition_json is invalid") from error
    if not isinstance(definition, dict):
        raise ArtifactIntegrityError("screening definition must be an object")
    return cast(dict[str, object], definition)


def expand_v2_screening_decision(
    row: Mapping[str, object],
    *,
    definitions: dict[int, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Reconstruct one logical event from legacy or definition-encoded v2 storage."""

    expanded_row: Mapping[str, object] = row
    if "definition_id" in row:
        if definitions is None:
            raise ArtifactIntegrityError(
                "definition-encoded screening rows require definition state"
            )
        raw_definition_id = _screening_definition_id_from_row(row)
        definition_json = row.get("definition_json")
        if definition_json:
            definition = _decode_screening_definition_json(definition_json)
            canonical = orjson.dumps(definition, option=orjson.OPT_SORT_KEYS)
            expected_id = (
                int.from_bytes(hashlib.sha256(canonical).digest()[:8], "big")
                & 0x7FFF_FFFF_FFFF_FFFF
            )
            if expected_id != raw_definition_id:
                raise ArtifactIntegrityError("screening definition hash mismatch")
            previous = definitions.get(raw_definition_id)
            if previous is not None and previous != definition:
                raise ArtifactIntegrityError("screening definition ID collision")
            definitions[raw_definition_id] = definition
        resolved_definition = definitions.get(raw_definition_id)
        if resolved_definition is None:
            raise ArtifactIntegrityError("screening row references an unknown definition")
        expanded_row = _combine_screening_definition_occurrence(resolved_definition, row)

    return _expand_screening_decision_row(expanded_row)


def _combine_screening_definition_occurrence(
    definition: Mapping[str, object],
    occurrence: Mapping[str, object],
) -> dict[str, object]:
    combined = {**definition, **dict(occurrence)}
    started_at = combined.get("started_at")
    completed_at = combined.get("completed_at")
    if isinstance(started_at, (int, float)) and isinstance(completed_at, (int, float)):
        combined["timestamp_seconds"] = started_at
        combined["duration_seconds"] = completed_at - started_at
    return combined


def _expand_screening_decision_row(
    expanded_row: Mapping[str, object],
) -> dict[str, object]:
    raw_checks = expanded_row.get("checks")
    checks_json = expanded_row.get("checks_json")
    if checks_json:
        if not isinstance(checks_json, (bytes, bytearray, memoryview, str)):
            raise ArtifactIntegrityError("compact screening checks_json must be binary or text")
        try:
            raw_checks = orjson.loads(checks_json)
        except orjson.JSONDecodeError as error:
            raise ArtifactIntegrityError("compact screening checks_json is invalid") from error
    compact_checks = raw_checks if isinstance(raw_checks, (list, tuple)) else ()
    if len(compact_checks) > MAX_SCREENING_CHECKS_PER_DECISION:
        raise ArtifactIntegrityError(
            "one screening decision exceeds the fixed eight-check domain"
        )
    extras = {
        key: expanded_row.get(key)
        for key in (
            "benchmark_axis",
            "demand",
            "distance_increment_lower_bound",
            "distance_lower_bound",
            "exact_call_blocked",
            "first_failed_check",
            "min_time_window_slack",
            "negative_cache_hit",
            "single_segment_reachable",
            "structural_energy_lower_bound",
        )
    }
    output = _SCREENING_EVENT_TEMPLATE.copy()
    output.update(
        {
            "event_id": expanded_row.get("event_id"),
            "record_type": "screening_decision",
            "event_type": "screening_decision",
            "timestamp_seconds": expanded_row.get("timestamp_seconds"),
            "started_at": expanded_row.get("started_at"),
            "completed_at": expanded_row.get("completed_at"),
            "duration_seconds": expanded_row.get("duration_seconds"),
            "lane_id": expanded_row.get("lane_id"),
            "iteration": expanded_row.get("iteration"),
            "operator_id": expanded_row.get("operator_id"),
            "route_id": expanded_row.get("route_id"),
            "status": expanded_row.get("status"),
            "reason": expanded_row.get("reason"),
            "decision_id": expanded_row.get("decision_id"),
            "propagation_status": expanded_row.get("status"),
            "extras_json": _json_text(extras),
            "embedded_checks": [
                {
                    "check": str(check.get("check", "")),
                    "status": str(check.get("status") or ""),
                    "value": _check_value(check),
                    "reason": str(check.get("reason") or ""),
                }
                for check in compact_checks
                if isinstance(check, Mapping)
            ],
        }
    )
    return output


def register_v3_screening_definition(
    row: Mapping[str, object],
    *,
    definitions: dict[int, dict[str, object]],
) -> None:
    """Validate and register one typed ``screening_decisions_v3`` definition."""

    raw_definition_id = row.get("definition_id")
    if isinstance(raw_definition_id, bool) or not isinstance(raw_definition_id, int):
        raise ArtifactIntegrityError("screening definition_id must be an integer")
    definition = _normalise_screening_definition(row)
    expected_id, _, _ = _screening_definition_identity(definition)
    if expected_id != raw_definition_id:
        raise ArtifactIntegrityError("screening definition hash mismatch")
    previous = definitions.get(raw_definition_id)
    if previous is not None:
        raise ArtifactIntegrityError("duplicate screening definition ID")
    definitions[raw_definition_id] = definition


def expand_v3_screening_decision(
    row: Mapping[str, object],
    *,
    definitions: Mapping[int, dict[str, object]],
) -> dict[str, object]:
    """Expand one v3 occurrence using an independently loaded definition table."""

    raw_definition_id = row.get("definition_id")
    if isinstance(raw_definition_id, bool) or not isinstance(raw_definition_id, int):
        raise ArtifactIntegrityError("screening definition_id must be an integer")
    definition = definitions.get(raw_definition_id)
    if definition is None:
        raise ArtifactIntegrityError("screening row references an unknown definition")
    return _expand_screening_decision_row(_combine_screening_definition_occurrence(definition, row))


def _validate_event_routes(event: Mapping[str, object], route_ids: Mapping[str, int]) -> None:
    for key in (
        "route_key",
        "base_route_key",
        "candidate_route_key",
    ):
        value = event.get(key)
        if value is not None and str(value) and str(value) not in route_ids:
            raise ArtifactIntegrityError(f"event refers to an unregistered route key: {value}")
    for key in (
        "route_keys",
        "current_route_keys",
        "candidate_route_keys",
    ):
        values = event.get(key)
        if isinstance(values, (list, tuple)) and any(
            str(value) and str(value) not in route_ids for value in values
        ):
            raise ArtifactIntegrityError(f"event refers to an unregistered route list: {key}")


def _coalesce_cache_lookup_events(
    events: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Collapse a cache lookup plus its immediate hit/miss result.

    The solver trace keeps the two operational callbacks because that is useful
    while debugging the evaluator.  The persisted evidence contract records
    the logical lookup result once, so an audit cannot accidentally count the
    same lookup twice.  Older event streams remain readable because events
    without ``lookup_result`` retain their legacy operation names.
    """

    output: list[dict[str, object]] = []
    index = 0
    while index < len(events):
        current = dict(events[index])
        if (
            current.get("event_type") == "cache_event"
            and current.get("operation") == "lookup"
            and index + 1 < len(events)
        ):
            following = dict(events[index + 1])
            same_lookup = all(
                current.get(field) == following.get(field)
                for field in ("route_key", "cache_key_digest", "lane", "iteration", "operator")
            )
            if (
                same_lookup
                and following.get("event_type") == "cache_event"
                and following.get("operation") in {"hit", "miss"}
            ):
                combined = dict(following)
                combined["operation"] = "lookup_result"
                combined["lookup_result"] = following["operation"]
                combined["lookup_current_entries"] = current.get("current_entries")
                combined["lookup_current_bytes"] = current.get("current_bytes")
                output.append(combined)
                index += 2
                continue
        output.append(current)
        index += 1
    return output


def _normalise_check(
    check: Mapping[str, object], *, event_id: int, decision_id: int | None, index: int
) -> dict[str, object]:
    value = check.get("value")
    return {
        "decision_event_id": event_id,
        "decision_id": decision_id,
        "check_index": index,
        "check": str(check.get("check", "")),
        "status": str(check.get("status", "")),
        "value_bool": value if isinstance(value, bool) else None,
        "value_float": float(value) if isinstance(value, (int, float)) else None,
        "value_text": value if isinstance(value, str) else None,
        "reason": str(check.get("reason", "")),
    }


def _write_parquet(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    schema: pa.Schema,
    config: ArtifactStorageConfig,
) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [pa.array((row.get(field.name) for row in rows), type=field.type) for field in schema]
    table = pa.Table.from_arrays(columns, schema=schema)
    pq.write_table(
        table,
        path,
        compression=config.compression,
        compression_level=config.compression_level,
        use_dictionary=True,
        write_statistics=True,
        row_group_size=(
            V2_PARQUET_ROW_GROUP_SIZE
            if config.storage_policy_version == ARTIFACT_STORAGE_V2
            else None
        ),
    )
    return table.num_rows, _schema_fingerprint(table.schema)


def _normalise_anytime_checkpoint(row: Mapping[str, object]) -> dict[str, object]:
    objective = row.get("objective_key")
    if not isinstance(objective, (list, tuple)) or len(objective) != 4:
        raise ArtifactIntegrityError(
            "anytime checkpoint objective_key must contain four components"
        )
    return {
        "instance": row.get("instance"),
        "seed": row.get("seed"),
        "axis_budget_seconds": row.get("axis_budget_seconds"),
        "checkpoint_seconds": row.get("checkpoint_seconds"),
        "objective_key": {
            "vehicle_count": objective[0],
            "total_distance": objective[1],
            "total_charging_time": objective[2],
            "charging_count": objective[3],
        },
        "source": row.get("source"),
        "incumbent_completed_at_seconds": row.get("incumbent_completed_at_seconds"),
        "incumbent_iteration": row.get("incumbent_iteration"),
    }


class _StreamingParquetSink:
    """Bounded typed-column row-group writer for artifact-storage-v2 shards."""

    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        config: ArtifactStorageConfig,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.schema = schema
        self.config = config
        self._columns: tuple[list[object], ...] = tuple([] for _ in schema)
        self._buffered_row_count = 0
        self.row_count = 0
        high_volume_compact = schema.equals(V2_SCREENING_DECISIONS_SCHEMA) or schema.equals(
            V3_SCREENING_OCCURRENCES_SCHEMA
        )
        self._writer = pq.ParquetWriter(
            path,
            schema,
            compression=config.compression,
            compression_level=config.compression_level,
            use_dictionary=not high_volume_compact,
            write_statistics=not high_volume_compact,
        )

    def append(self, row: Mapping[str, object]) -> None:
        for field, column in zip(self.schema, self._columns, strict=True):
            column.append(row.get(field.name))
        self._buffered_row_count += 1
        if self._buffered_row_count >= V2_PARQUET_ROW_GROUP_SIZE:
            self.flush()

    def append_values(self, values: Sequence[object]) -> None:
        """Append one schema-ordered row without constructing a row mapping."""

        if len(values) != len(self._columns):
            raise ArtifactIntegrityError(
                f"typed row width does not match sink schema for {self.path}"
            )
        for column, value in zip(self._columns, values, strict=True):
            column.append(value)
        self._buffered_row_count += 1
        if self._buffered_row_count >= V2_PARQUET_ROW_GROUP_SIZE:
            self.flush()

    def append_value_rows(
        self,
        rows: Sequence[Sequence[object]],
        *,
        trusted_width: bool = False,
    ) -> None:
        """Append schema-ordered rows by extending typed columns per transaction."""

        if not rows:
            return
        width = len(self._columns)
        invalid_width = (
            len(rows[0]) != width
            if trusted_width
            else any(len(row) != width for row in rows)
        )
        if invalid_width:
            raise ArtifactIntegrityError(
                f"typed row width does not match sink schema for {self.path}"
            )
        offset = 0
        while offset < len(rows):
            available = V2_PARQUET_ROW_GROUP_SIZE - self._buffered_row_count
            chunk = rows[offset : offset + available]
            for column, values in zip(self._columns, zip(*chunk, strict=True), strict=True):
                column.extend(values)
            appended = len(chunk)
            self._buffered_row_count += appended
            offset += appended
            if self._buffered_row_count == V2_PARQUET_ROW_GROUP_SIZE:
                self.flush()

    def append_batch(self, batch: pa.RecordBatch) -> None:
        """Append one schema-identical Arrow batch without row materialization."""

        if not batch.schema.equals(self.schema, check_metadata=False):
            raise ArtifactIntegrityError(
                f"Arrow batch schema does not match sink schema for {self.path}"
            )
        self.flush()
        offset = 0
        while offset < batch.num_rows:
            current = batch.slice(offset, V2_PARQUET_ROW_GROUP_SIZE)
            self._writer.write_batch(current, row_group_size=V2_PARQUET_ROW_GROUP_SIZE)
            self.row_count += current.num_rows
            offset += current.num_rows

    @property
    def buffered_row_count(self) -> int:
        return self._buffered_row_count

    def flush(self) -> None:
        if self._buffered_row_count == 0:
            return
        arrays = [
            pa.array(column, type=field.type)
            for field, column in zip(self.schema, self._columns, strict=True)
        ]
        table = pa.Table.from_arrays(arrays, schema=self.schema)
        self._writer.write_table(table, row_group_size=V2_PARQUET_ROW_GROUP_SIZE)
        self.row_count += table.num_rows
        for column in self._columns:
            column.clear()
        self._buffered_row_count = 0

    def close(self) -> tuple[int, str]:
        errors: list[BaseException] = []
        try:
            self.flush()
        except BaseException as error:
            errors.append(error)
        try:
            self._writer.close()
        except BaseException as error:
            errors.append(error)
        if errors:
            raise BaseExceptionGroup(f"failed to close Parquet sink {self.path}", errors)
        return self.row_count, _schema_fingerprint(self.schema)


@overload
def _iter_coalesced_cache_lookup_events(
    events: Iterable[Mapping[str, object]],
) -> Iterable[dict[str, object]]: ...


@overload
def _iter_coalesced_cache_lookup_events(
    events: Iterable[Mapping[str, object] | _BufferedScreeningDecision],
) -> Iterable[dict[str, object] | _BufferedScreeningDecision]: ...


def _iter_coalesced_cache_lookup_events(
    events: Iterable[Mapping[str, object] | _BufferedScreeningDecision],
) -> Iterable[dict[str, object] | _BufferedScreeningDecision]:
    """Streaming equivalent of :func:`_coalesce_cache_lookup_events`."""

    pending: Mapping[str, object] | None = None
    for raw_event in events:
        if isinstance(raw_event, tuple):
            if pending is not None:
                yield dict(pending) if not isinstance(pending, dict) else pending
                pending = None
            yield raw_event
            continue
        current = raw_event
        if pending is None:
            if current.get("event_type") == "cache_event" and current.get("operation") == "lookup":
                pending = current
            else:
                yield dict(current) if not isinstance(current, dict) else current
            continue
        same_lookup = all(
            pending.get(field) == current.get(field)
            for field in ("route_key", "cache_key_digest", "lane", "iteration", "operator")
        )
        if (
            pending.get("event_type") == "cache_event"
            and pending.get("operation") == "lookup"
            and same_lookup
            and current.get("event_type") == "cache_event"
            and current.get("operation") in {"hit", "miss"}
        ):
            combined = dict(current)
            combined["operation"] = "lookup_result"
            combined["lookup_result"] = current["operation"]
            combined["lookup_current_entries"] = pending.get("current_entries")
            combined["lookup_current_bytes"] = pending.get("current_bytes")
            yield combined
            pending = None
        else:
            yield dict(pending) if not isinstance(pending, dict) else pending
            if current.get("event_type") == "cache_event" and current.get("operation") == "lookup":
                pending = current
            else:
                yield dict(current) if not isinstance(current, dict) else current
                pending = None
    if pending is not None:
        yield dict(pending) if not isinstance(pending, dict) else pending


def _stable_dictionary_id(value: str) -> int:
    """Return a deterministic positive int32 ID independent of completion order."""

    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def _invert_trace_dictionary(
    payload: Mapping[str, object],
    kind: str,
) -> dict[str, int]:
    output: dict[str, int] = {}
    seen_ids: set[int] = set()
    for raw_id, raw_value in payload.items():
        if not isinstance(raw_value, str) or not raw_value:
            raise ArtifactIntegrityError(f"{kind} dictionary value is invalid")
        try:
            dictionary_id = int(raw_id)
        except ValueError as error:
            raise ArtifactIntegrityError(f"{kind} dictionary ID is invalid") from error
        if (
            dictionary_id <= 0
            or raw_id != str(dictionary_id)
            or dictionary_id in seen_ids
            or raw_value in output
        ):
            raise ArtifactIntegrityError(f"{kind} dictionary is duplicate or invalid")
        seen_ids.add(dictionary_id)
        output[raw_value] = dictionary_id
    return output


def _stable_route_id(route_key: str) -> int:
    """Return a deterministic positive int63 ID for high-cardinality routes."""

    digest = hashlib.sha256(f"route:{route_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def compact_trace_payload(
    payload: Mapping[str, object],
    *,
    route_dictionary_ref: str,
    events_ref: str,
    screening_checks_ref: str,
    diagnostic_ref: str,
    lane_dictionary: Mapping[str, int] | None = None,
    operator_dictionary: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Return the JSON trace index without duplicating columnar evidence.

    Stage 3.0--3.2 historically serialised the same event collections into a
    trace JSON and an event JSONL.  The v2 storage contract keeps only the
    scalar trace counters/configuration in JSON and points the reader at the
    canonical Arrow artifacts.  The function deliberately removes known large
    collections instead of silently serialising them under a new key.
    """

    compact = dict(payload)
    for key in (
        "route_dictionary",
        "route_evaluations",
        "events",
        "screening_decisions",
        "incremental_propagations",
        "operator_calls",
        "candidate_states",
        "deadline_events",
    ):
        compact.pop(key, None)
    compact["trace_storage_version"] = "stage03-trace-index-v2"
    compact["route_dictionary_ref"] = route_dictionary_ref
    compact["events_ref"] = events_ref
    compact["screening_checks_ref"] = screening_checks_ref
    compact["diagnostic_ref"] = diagnostic_ref
    compact["lane_dictionary"] = {
        str(index): value for value, index in (lane_dictionary or {}).items()
    }
    compact["operator_dictionary"] = {
        str(index): value for value, index in (operator_dictionary or {}).items()
    }
    return compact


def canonical_artifact_paths(
    context: ArtifactRunContext, instance: str, seed: int
) -> dict[str, str]:
    directory = f"{instance}/{seed}"
    return {
        artifact_type: (
            f"{directory}/{_canonical_filename(context, artifact_type, instance, seed, extension)}"
        )
        for artifact_type, extension in (
            ("raw", "json"),
            ("solution", "json"),
            ("trace", "json"),
            ("events", "parquet"),
            ("route_dictionary", "parquet"),
            ("screening_checks", "parquet"),
            ("screening_decisions", "parquet"),
            ("screening_definitions", "parquet"),
            ("screening_occurrences", "parquet"),
            ("diagnostic", "parquet"),
            ("environment", "json"),
            ("failure", "json"),
        )
    }


def _event_route_keys(event: Mapping[str, object]) -> Iterable[str]:
    for field in (
        "route_key",
        "base_route_key",
        "candidate_route_key",
    ):
        value = event.get(field)
        if isinstance(value, str) and value:
            yield value
    for field in (
        "route_keys",
        "current_route_keys",
        "candidate_route_keys",
    ):
        values = event.get(field)
        if isinstance(values, (list, tuple)):
            yield from (value for value in values if isinstance(value, str) and value)


def _route_sequence_from_key(route_key: str) -> tuple[str, ...]:
    prefix = "route:"
    if not route_key.startswith(prefix):
        raise ArtifactIntegrityError(
            f"cannot reconstruct route dictionary entry from key: {route_key}"
        )
    encoded = route_key[len(prefix) :]
    if not encoded:
        return ()
    sequence: list[str] = []
    for token in encoded.split("|"):
        length_text, separator, customer = token.partition(":")
        if not separator or not length_text.isdigit() or int(length_text) != len(customer):
            raise ArtifactIntegrityError(
                f"invalid canonical route key in event stream: {route_key}"
            )
        sequence.append(customer)
    return tuple(sequence)


def write_solver_result_bundle(
    writer: ArtifactBundleWriter,
    *,
    instance: Any,
    seed: int,
    result: object,
    raw_record: Mapping[str, object],
    environment_payload: Mapping[str, object],
    error: BaseException | None = None,
) -> dict[str, str]:
    """Persist a generic ALNS result for future Stage 0--8 adapters.

    The solver-specific runner still owns validation and gate calculations.  It
    passes the already computed row here; this function owns only the physical
    artifact representation and never writes a tracked summary.
    """

    from dataclasses import asdict, is_dataclass

    instance_name = str(instance.name)
    routes = [list(route) for route in getattr(result, "routes", ())]
    result_feasible = getattr(result, "feasible", None)
    if result_feasible is None:
        result_feasible = getattr(result, "objective", None) is not None
    context_paths = canonical_artifact_paths(writer.context, instance_name, seed)
    record = {**dict(raw_record)}
    record.update(
        {
            "raw_path": context_paths["raw"],
            "solution_path": context_paths["solution"],
            "trace_path": context_paths["trace"],
            "event_path": context_paths["events"],
            "environment_path": context_paths["environment"],
        }
    )
    if error is not None or not bool(result_feasible):
        record["failure_path"] = context_paths["failure"]
    solver_payload: object
    if is_dataclass(result):
        solver_payload = asdict(cast(Any, result))
    elif isinstance(result, Mapping):
        solver_payload = dict(result)
    else:
        solver_payload = {"repr": repr(result)}
    if isinstance(solver_payload, dict):
        solver_payload["measurement_trace"] = None
        solver_payload["neighborhood_events"] = None
    trace = getattr(result, "measurement_trace", None)
    if trace is not None and hasattr(trace, "to_dict"):
        trace_payload = trace.to_dict()
        route_dictionary = dict(getattr(trace, "route_dictionary", {}))
    else:
        from evrptw.measurement import canonical_route_key

        route_dictionary = {canonical_route_key(tuple(route)): tuple(route) for route in routes}
        trace_payload = {
            "trace_schema_version": "solver-result-trace-v1",
            "config": {
                "enabled": False,
                "schema_version": "stage03-trace-v1",
                "record_route_dictionary": True,
                "record_operator_events": False,
                "record_candidate_states": False,
            },
            "summary": {},
        }
    events = [
        dict(event)
        for event in getattr(result, "neighborhood_events", ())
        if isinstance(event, Mapping)
    ]
    for event in events:
        for route_key in _event_route_keys(event):
            route_dictionary.setdefault(
                route_key,
                _route_sequence_from_key(route_key),
            )
    failure_payload = None
    if error is not None or not bool(result_feasible):
        failure_payload = {
            "run_label": writer.context.run_label,
            "instance": instance_name,
            "seed": seed,
            "error_type": type(error).__name__ if error is not None else "",
            "failure_reason": (
                str(error) if error is not None else str(getattr(result, "failure_reason", ""))
            ),
            "evidence_completeness": "complete",
        }
    return writer.write_instance_seed(
        instance=instance_name,
        seed=seed,
        raw_payload={"record": record, "solver_result": solver_payload},
        solution_payload={
            "instance": instance_name,
            "seed": seed,
            "routes": routes,
            "objective_key": list(getattr(getattr(result, "objective", None), "key", ())),
        },
        trace_payload=trace_payload,
        environment_payload=dict(environment_payload),
        route_dictionary=route_dictionary,
        critical_events=events,
        diagnostic_rows=aggregate_diagnostic_events(
            events,
            run_label=writer.context.run_label,
            instance=instance_name,
            seed=seed,
        ),
        failure_payload=failure_payload,
    )


class ArtifactBundleWriter:
    """Deep writer for one canonical run bundle."""

    def __init__(
        self,
        run_dir: Path,
        context: ArtifactRunContext,
        config: ArtifactStorageConfig | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.context = context
        self.config = config or ArtifactStorageConfig()
        if not self.config.enabled:
            raise ValueError("ArtifactBundleWriter requires an enabled storage policy")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts: list[ArtifactReference] = []
        self._instance_bytes = 0
        self._current_instance: tuple[str, int] | None = None
        self._next_event_id = 1
        self._manifest_path: Path | None = None

    @property
    def artifacts(self) -> tuple[ArtifactReference, ...]:
        return tuple(self._artifacts)

    def write_control(
        self,
        *,
        metadata: Mapping[str, object],
        configuration_path: Path | None = None,
    ) -> None:
        control = self.run_dir / "control"
        control.mkdir(parents=True, exist_ok=True)
        metadata_path = control / _canonical_filename(self.context, "run_metadata")
        _json_write(metadata_path, {**dict(metadata), "storage_policy": self.config.to_dict()})
        self._record_file(
            metadata_path,
            artifact_type="manifest_metadata",
            retention_class="control",
            storage_format="json_control",
            compression="none",
        )
        if configuration_path is not None:
            config_path = control / _canonical_filename(self.context, "config", extension="toml")
            shutil.copy2(configuration_path, config_path)
            self._record_file(
                config_path,
                artifact_type="config",
                retention_class="control",
                storage_format="toml_control",
                compression="none",
            )

    def record_existing_file(
        self,
        path: Path,
        *,
        artifact_type: str,
        retention_class: str = "control",
        storage_format: str = "csv_control",
        compression: str = "none",
        artifact_subtype: str = "",
        row_count: int | None = None,
        schema_fingerprint: str = "",
    ) -> None:
        """Register a writer-owned control file created by a caller.

        This is intentionally a registration-only seam.  Runners may create a
        small control index such as ``raw_per_run_results.csv``, but checksum
        and budget accounting still remain owned by this module.
        """

        if not path.is_file():
            raise FileNotFoundError(path)
        current_instance = self._current_instance
        self._current_instance = None
        try:
            try:
                self._record_file(
                    path,
                    artifact_type=artifact_type,
                    artifact_subtype=artifact_subtype,
                    retention_class=retention_class,
                    storage_format=storage_format,
                    compression=compression,
                    row_count=row_count,
                    schema_fingerprint=schema_fingerprint,
                )
            except ArtifactBudgetExceeded:
                # A control index is part of the evidence boundary too.  If
                # registering it exceeds a configured budget, preserve all
                # files already written, publish a partial manifest, and
                # re-raise instead of leaving an unverifiable bundle.
                self._register_partial_control_files()
                self._write_manifest(
                    status="partial",
                    evidence_completeness="partial",
                )
                raise
        finally:
            self._current_instance = current_instance

    def adopt_v2_shards(
        self,
        *,
        expected_identities: Sequence[tuple[str, int]],
        require_complete: bool = True,
    ) -> None:
        """Adopt worker-owned shards into a parent manifest using metadata only."""

        if self.config.storage_policy_version != ARTIFACT_STORAGE_V2:
            raise ValueError("only artifact-storage-v2 can adopt worker shards")
        expected = set(expected_identities)
        if len(expected) != len(expected_identities):
            raise ValueError("expected shard identities contain duplicates")
        discovered: list[tuple[int, str, int, Path, dict[str, Any]]] = []
        for manifest_path in self.run_dir.glob("*/*/*_shard_manifest_*.json"):
            if _is_appledouble_path(manifest_path.relative_to(self.run_dir)):
                continue
            payload = _json_read(manifest_path)
            try:
                identity = (str(payload["instance"]), int(payload["seed"]))
                ordinal = int(payload["shard_ordinal"])
            except (KeyError, TypeError, ValueError) as error:
                raise ArtifactIntegrityError(
                    f"invalid v2 shard identity: {manifest_path}"
                ) from error
            sidecar = manifest_path.with_suffix(".sha256")
            if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != _sha256(
                manifest_path
            ):
                raise ArtifactIntegrityError(f"v2 shard manifest sidecar mismatch: {manifest_path}")
            if payload.get("storage_policy_version") != ARTIFACT_STORAGE_V2:
                raise ArtifactIntegrityError("worker shard does not use artifact-storage-v2")
            if require_complete and payload.get("evidence_completeness") != "complete":
                raise ArtifactIntegrityError(f"partial worker shard cannot be adopted: {identity}")
            discovered.append((ordinal, identity[0], identity[1], manifest_path, payload))
        identities = {(instance, seed) for _, instance, seed, _, _ in discovered}
        if identities != expected or len(discovered) != len(expected):
            raise ArtifactIntegrityError(
                f"worker shard identity mismatch: expected={sorted(expected)} "
                f"observed={sorted(identities)}"
            )
        ordinals = [item[0] for item in discovered]
        if len(set(ordinals)) != len(ordinals):
            raise ArtifactIntegrityError("worker shard ordinals are not unique")
        current_instance = self._current_instance
        self._current_instance = None
        try:
            for _, _, _, manifest_path, payload in sorted(discovered):
                artifacts = payload.get("artifacts")
                if not isinstance(artifacts, list):
                    raise ArtifactIntegrityError("v2 shard manifest has no artifacts list")
                for item in artifacts:
                    if not isinstance(item, Mapping):
                        raise ArtifactIntegrityError("v2 shard artifact entry is invalid")
                    path = _safe_artifact_path(self.run_dir, str(item["relative_path"]))
                    if _sha256(path) != str(item["checksum"]):
                        raise ArtifactIntegrityError(f"v2 shard artifact checksum mismatch: {path}")
                    self._record_file(
                        path,
                        artifact_type=str(item["artifact_type"]),
                        artifact_subtype=str(item.get("artifact_subtype", "")),
                        retention_class=str(item["retention_class"]),
                        storage_format=str(item["storage_format"]),
                        compression=str(item["compression"]),
                        row_count=(
                            int(item["row_count"]) if item.get("row_count") is not None else None
                        ),
                        schema_fingerprint=str(item.get("schema_fingerprint", "")),
                    )
                self._record_file(
                    manifest_path,
                    artifact_type="shard_manifest",
                    retention_class="control",
                    storage_format="json_control",
                    compression="none",
                )
                self._record_file(
                    manifest_path.with_suffix(".sha256"),
                    artifact_type="shard_manifest_sidecar",
                    retention_class="control",
                    storage_format="sha256_control",
                    compression="none",
                )
        finally:
            self._current_instance = current_instance

    def write_instance_seed(
        self,
        *,
        instance: str,
        seed: int,
        raw_payload: Mapping[str, object],
        solution_payload: Mapping[str, object],
        trace_payload: Mapping[str, object],
        environment_payload: Mapping[str, object],
        route_dictionary: Mapping[str, Sequence[str]],
        critical_events: Iterable[Mapping[str, object]],
        diagnostic_rows: Iterable[Mapping[str, object]] = (),
        failure_payload: Mapping[str, object] | None = None,
        shard_ordinal: int | None = None,
        worker_identity: str | None = None,
    ) -> dict[str, str]:
        if self.config.storage_policy_version == ARTIFACT_STORAGE_V2:
            if shard_ordinal is None or shard_ordinal < 0:
                raise ValueError("artifact-storage-v2 requires a non-negative shard_ordinal")
            if not worker_identity:
                raise ValueError("artifact-storage-v2 requires worker_identity")
            return self._write_instance_seed_v2(
                instance=instance,
                seed=seed,
                shard_ordinal=shard_ordinal,
                worker_identity=worker_identity,
                raw_payload=raw_payload,
                solution_payload=solution_payload,
                trace_payload=trace_payload,
                environment_payload=environment_payload,
                route_dictionary=route_dictionary,
                critical_events=critical_events,
                diagnostic_rows=diagnostic_rows,
                failure_payload=failure_payload,
            )
        elif shard_ordinal is not None or worker_identity is not None:
            raise ValueError("shard identity is available only for artifact-storage-v2")
        self._current_instance = (instance, seed)
        self._instance_bytes = 0
        directory = self.run_dir / instance / str(seed)
        directory.mkdir(parents=True, exist_ok=True)
        route_rows, route_ids = self._route_rows(route_dictionary)
        event_input = _coalesce_cache_lookup_events(list(critical_events))
        lane_ids = {
            value: index
            for index, value in enumerate(
                sorted({str(event.get("lane", "")) for event in event_input})
            )
        }
        operator_ids = {
            value: index
            for index, value in enumerate(
                sorted({str(event.get("operator", "")) for event in event_input})
            )
        }
        event_rows: list[dict[str, object]] = []
        screening_rows: list[dict[str, object]] = []
        for event in event_input:
            event_id = self._next_event_id
            self._next_event_id += 1
            _validate_event_routes(event, route_ids)
            event_rows.append(
                _normalise_event(
                    event,
                    event_id=event_id,
                    route_ids=route_ids,
                    lane_ids=lane_ids,
                    operator_ids=operator_ids,
                )
            )
            checks = event.get("checks")
            if isinstance(checks, (list, tuple)):
                decision_id = _as_int(event.get("decision_id"))
                screening_rows.extend(
                    _normalise_check(
                        check,
                        event_id=event_id,
                        decision_id=decision_id,
                        index=index,
                    )
                    for index, check in enumerate(checks)
                    if isinstance(check, Mapping)
                )
        diagnostic = [self._normalise_diagnostic(row, instance, seed) for row in diagnostic_rows]
        paths: dict[str, str] = {}
        try:
            paths["raw"] = self._write_json_artifact(
                directory,
                "raw",
                instance,
                seed,
                raw_payload,
            )
            paths["solution"] = self._write_json_artifact(
                directory,
                "solution",
                instance,
                seed,
                solution_payload,
            )
            route_relative = str(
                (
                    directory
                    / _canonical_filename(
                        self.context, "route_dictionary", instance, seed, "parquet"
                    )
                ).relative_to(self.run_dir)
            )
            events_relative = str(
                (
                    directory
                    / _canonical_filename(self.context, "events", instance, seed, "parquet")
                ).relative_to(self.run_dir)
            )
            checks_relative = str(
                (
                    directory
                    / _canonical_filename(
                        self.context, "screening_checks", instance, seed, "parquet"
                    )
                ).relative_to(self.run_dir)
            )
            diagnostic_relative = str(
                (
                    directory
                    / _canonical_filename(self.context, "diagnostic", instance, seed, "parquet")
                ).relative_to(self.run_dir)
            )
            paths["environment"] = self._write_json_artifact(
                directory,
                "environment",
                instance,
                seed,
                environment_payload,
            )
            route_path = directory / _canonical_filename(
                self.context, "route_dictionary", instance, seed, "parquet"
            )
            route_count, route_schema = _write_parquet(
                route_path, route_rows, ROUTE_DICTIONARY_SCHEMA, self.config
            )
            self._record_file(
                route_path,
                artifact_type="route_dictionary",
                artifact_subtype="canonical_routes",
                retention_class="critical",
                storage_format="parquet",
                compression=self.config.compression,
                row_count=route_count,
                schema_fingerprint=route_schema,
            )
            paths["route_dictionary"] = str(route_path.relative_to(self.run_dir))
            event_path = directory / _canonical_filename(
                self.context, "events", instance, seed, "parquet"
            )
            event_count, event_schema = _write_parquet(
                event_path, event_rows, EVENTS_SCHEMA, self.config
            )
            self._record_file(
                event_path,
                artifact_type="events",
                artifact_subtype="critical",
                retention_class="critical",
                storage_format="parquet",
                compression=self.config.compression,
                row_count=event_count,
                schema_fingerprint=event_schema,
            )
            paths["events"] = str(event_path.relative_to(self.run_dir))
            checks_path = directory / _canonical_filename(
                self.context, "screening_checks", instance, seed, "parquet"
            )
            check_count, check_schema = _write_parquet(
                checks_path, screening_rows, SCREENING_CHECKS_SCHEMA, self.config
            )
            self._record_file(
                checks_path,
                artifact_type="events",
                artifact_subtype="screening_checks",
                retention_class="critical",
                storage_format="parquet",
                compression=self.config.compression,
                row_count=check_count,
                schema_fingerprint=check_schema,
            )
            paths["screening_checks"] = str(checks_path.relative_to(self.run_dir))
            diagnostic_path = directory / _canonical_filename(
                self.context, "diagnostic", instance, seed, "parquet"
            )
            diagnostic_count, diagnostic_schema = _write_parquet(
                diagnostic_path, diagnostic, DIAGNOSTIC_SCHEMA, self.config
            )
            self._record_file(
                diagnostic_path,
                artifact_type="diagnostic",
                artifact_subtype="aggregated",
                retention_class="diagnostic",
                storage_format="parquet",
                compression=self.config.compression,
                row_count=diagnostic_count,
                schema_fingerprint=diagnostic_schema,
            )
            paths["diagnostic"] = str(diagnostic_path.relative_to(self.run_dir))
            trace_index = compact_trace_payload(
                trace_payload,
                route_dictionary_ref=route_relative,
                events_ref=events_relative,
                screening_checks_ref=checks_relative,
                diagnostic_ref=diagnostic_relative,
                lane_dictionary=lane_ids,
                operator_dictionary=operator_ids,
            )
            trace_index["schema_fingerprints"] = {
                "route_dictionary": route_schema,
                "events": event_schema,
                "screening_checks": check_schema,
                "diagnostic": diagnostic_schema,
            }
            paths["trace"] = self._write_json_artifact(
                directory,
                "trace",
                instance,
                seed,
                trace_index,
            )
            if failure_payload is not None:
                paths["failure"] = self._write_json_artifact(
                    directory,
                    "failure",
                    instance,
                    seed,
                    failure_payload,
                    retention_class="critical",
                )
        except ArtifactBudgetExceeded:
            if failure_payload is None:
                failure_payload = {
                    "schema_version": ARTIFACT_STORAGE_SCHEMA_VERSION,
                    "run_label": self.context.run_label,
                    "instance": instance,
                    "seed": seed,
                    "status": "partial",
                    "evidence_completeness": "partial",
                    "failure_reason": "artifact byte budget exceeded",
                }
            failure_path = directory / _canonical_filename(self.context, "failure", instance, seed)
            _json_write(failure_path, failure_payload)
            self._record_file(
                failure_path,
                artifact_type="failure",
                retention_class="critical",
                storage_format="json_control",
                compression="none",
                enforce_budget=False,
            )
            self._register_partial_control_files()
            self._write_manifest(status="partial", evidence_completeness="partial")
            raise
        return paths

    def open_v2_shard(
        self,
        *,
        instance: str,
        seed: int,
        shard_ordinal: int,
        worker_identity: str,
    ) -> ArtifactV2ShardSession:
        """Open one bounded worker-owned v2 shard for incremental appends."""

        if self.config.storage_policy_version != ARTIFACT_STORAGE_V2:
            raise ValueError("open_v2_shard requires artifact-storage-v2")
        if shard_ordinal < 0:
            raise ValueError("artifact-storage-v2 requires a non-negative shard_ordinal")
        if not worker_identity:
            raise ValueError("artifact-storage-v2 requires worker_identity")
        return ArtifactV2ShardSession(
            self,
            instance=instance,
            seed=seed,
            shard_ordinal=shard_ordinal,
            worker_identity=worker_identity,
        )

    def write_v2_failure_shard(
        self,
        *,
        instance: str,
        seed: int,
        shard_ordinal: int,
        worker_identity: str,
        error: BaseException | str,
    ) -> None:
        """Publish a partial shard without overwriting worker fragments."""

        if self.config.storage_policy_version != ARTIFACT_STORAGE_V2:
            raise ValueError("failure shards require artifact-storage-v2")
        self._current_instance = (instance, seed)
        self._instance_bytes = 0
        artifact_start = len(self._artifacts)
        directory = self.run_dir / instance / str(seed)
        directory.mkdir(parents=True, exist_ok=True)
        failure_path = directory / _canonical_filename(self.context, "failure", instance, seed)
        manifest_path = directory / _canonical_filename(
            self.context, "shard_manifest", instance, seed
        )
        sidecar_path = manifest_path.with_suffix(".sha256")
        for path, kind in (
            (failure_path, "failure"),
            (manifest_path, "manifest"),
            (sidecar_path, "manifest_sidecar"),
        ):
            self._archive_v2_control_fragment(
                path,
                instance=instance,
                seed=seed,
                kind=kind,
            )
        self._register_v2_partial_fragments(directory)
        _json_write(
            failure_path,
            {
                "schema_version": ARTIFACT_STORAGE_V2,
                "run_label": self.context.run_label,
                "instance": instance,
                "seed": seed,
                "status": "partial",
                "evidence_completeness": "partial",
                "failure_reason": str(error),
            },
        )
        self._record_file(
            failure_path,
            artifact_type="failure",
            retention_class="critical",
            storage_format="json_control",
            compression="none",
            enforce_budget=False,
        )
        self._write_v2_shard_manifest(
            directory=directory,
            instance=instance,
            seed=seed,
            shard_ordinal=shard_ordinal,
            worker_identity=worker_identity,
            artifact_start=artifact_start,
            evidence_completeness="partial",
        )

    def _archive_v2_control_fragment(
        self,
        path: Path,
        *,
        instance: str,
        seed: int,
        kind: str,
    ) -> None:
        """Preserve an incomplete canonical control file before recovery."""

        if not path.is_file():
            return
        relative = path.relative_to(self.run_dir).as_posix()
        removed_bytes = sum(
            item.byte_size for item in self._artifacts if item.relative_path == relative
        )
        self._artifacts = [item for item in self._artifacts if item.relative_path != relative]
        self._instance_bytes = max(0, self._instance_bytes - removed_bytes)
        digest = _sha256(path)
        fragment = path.parent / (
            f"{self.context.run_label}_partial_fragment_{kind}_{instance}_{seed}_{digest[:12]}.bin"
        )
        suffix = 1
        while fragment.exists():
            fragment = path.parent / (
                f"{self.context.run_label}_partial_fragment_{kind}_"
                f"{instance}_{seed}_{digest[:12]}_{suffix}.bin"
            )
            suffix += 1
        path.replace(fragment)
        self._record_partial_fragment(fragment)

    def _register_v2_partial_fragments(self, directory: Path) -> None:
        registered = {item.relative_path for item in self._artifacts}
        for fragment in sorted(directory.iterdir()):
            if not fragment.is_file() or _is_appledouble_path(fragment.relative_to(self.run_dir)):
                continue
            relative = fragment.relative_to(self.run_dir).as_posix()
            if relative in registered:
                continue
            self._record_partial_fragment(fragment)
            registered.add(relative)

    def _record_partial_fragment(self, fragment: Path) -> None:
        self._record_file(
            fragment,
            artifact_type="partial_shard_fragment",
            artifact_subtype=fragment.name,
            retention_class="critical",
            storage_format="binary_partial",
            compression="unknown",
            enforce_budget=False,
        )

    def _write_instance_seed_v2(
        self,
        *,
        instance: str,
        seed: int,
        shard_ordinal: int,
        worker_identity: str,
        raw_payload: Mapping[str, object],
        solution_payload: Mapping[str, object],
        trace_payload: Mapping[str, object],
        environment_payload: Mapping[str, object],
        route_dictionary: Mapping[str, Sequence[str]],
        critical_events: Iterable[Mapping[str, object]],
        diagnostic_rows: Iterable[Mapping[str, object]],
        failure_payload: Mapping[str, object] | None,
    ) -> dict[str, str]:
        """Write one bounded, worker-owned artifact-storage-v2 shard."""

        if self.config.screening_schema_version == SCREENING_DECISIONS_V3:
            shard = self.open_v2_shard(
                instance=instance,
                seed=seed,
                shard_ordinal=shard_ordinal,
                worker_identity=worker_identity,
            )
            shard.append(
                route_dictionary=route_dictionary,
                critical_events=critical_events,
                diagnostic_rows=diagnostic_rows,
            )
            return shard.finalize(
                raw_payload=raw_payload,
                solution_payload=solution_payload,
                trace_payload=trace_payload,
                environment_payload=environment_payload,
                failure_payload=failure_payload,
            )

        self._current_instance = (instance, seed)
        self._instance_bytes = 0
        self._next_event_id = 1
        shard_artifact_start = len(self._artifacts)
        directory = self.run_dir / instance / str(seed)
        directory.mkdir(parents=True, exist_ok=True)
        route_rows, route_ids = self._route_rows(route_dictionary)
        lane_ids: dict[str, int] = {}
        operator_ids: dict[str, int] = {}
        paths = canonical_artifact_paths(self.context, instance, seed)
        event_path = self.run_dir / paths["events"]
        checks_path = self.run_dir / paths["screening_checks"]
        diagnostic_path = self.run_dir / paths["diagnostic"]
        event_sink = _StreamingParquetSink(event_path, EVENTS_SCHEMA, self.config)
        checks_sink = _StreamingParquetSink(checks_path, SCREENING_CHECKS_SCHEMA, self.config)
        diagnostic_sink = _StreamingParquetSink(diagnostic_path, DIAGNOSTIC_SCHEMA, self.config)
        try:
            paths["raw"] = self._write_json_artifact(directory, "raw", instance, seed, raw_payload)
            paths["solution"] = self._write_json_artifact(
                directory, "solution", instance, seed, solution_payload
            )
            paths["environment"] = self._write_json_artifact(
                directory, "environment", instance, seed, environment_payload
            )
            route_path = self.run_dir / paths["route_dictionary"]
            route_count, route_schema = _write_parquet(
                route_path, route_rows, ROUTE_DICTIONARY_SCHEMA, self.config
            )
            self._record_file(
                route_path,
                artifact_type="route_dictionary",
                artifact_subtype="canonical_routes",
                retention_class="critical",
                storage_format="parquet",
                compression=self.config.compression,
                row_count=route_count,
                schema_fingerprint=route_schema,
            )
            for event in _iter_coalesced_cache_lookup_events(critical_events):
                lane = str(event.get("lane", ""))
                operator = str(event.get("operator", ""))
                lane_ids.setdefault(lane, _stable_dictionary_id(f"lane:{lane}"))
                operator_ids.setdefault(operator, _stable_dictionary_id(f"operator:{operator}"))
                _validate_event_routes(event, route_ids)
                event_id = self._next_event_id
                self._next_event_id += 1
                event_sink.append(
                    _normalise_event(
                        event,
                        event_id=event_id,
                        route_ids=route_ids,
                        lane_ids=lane_ids,
                        operator_ids=operator_ids,
                    )
                )
                checks = event.get("checks")
                if isinstance(checks, (list, tuple)):
                    decision_id = _as_int(event.get("decision_id"))
                    for index, check in enumerate(checks):
                        if isinstance(check, Mapping):
                            checks_sink.append(
                                _normalise_check(
                                    check,
                                    event_id=event_id,
                                    decision_id=decision_id,
                                    index=index,
                                )
                            )
            for row in diagnostic_rows:
                diagnostic_sink.append(self._normalise_diagnostic(row, instance, seed))
            event_count, event_schema = event_sink.close()
            check_count, check_schema = checks_sink.close()
            diagnostic_count, diagnostic_schema = diagnostic_sink.close()
            for path, artifact_type, subtype, retention, count, fingerprint in (
                (event_path, "events", "critical", "critical", event_count, event_schema),
                (
                    checks_path,
                    "events",
                    "screening_checks",
                    "critical",
                    check_count,
                    check_schema,
                ),
                (
                    diagnostic_path,
                    "diagnostic",
                    "aggregated",
                    "diagnostic",
                    diagnostic_count,
                    diagnostic_schema,
                ),
            ):
                self._record_file(
                    path,
                    artifact_type=artifact_type,
                    artifact_subtype=subtype,
                    retention_class=retention,
                    storage_format="parquet",
                    compression=self.config.compression,
                    row_count=count,
                    schema_fingerprint=fingerprint,
                )
            trace_index = compact_trace_payload(
                trace_payload,
                route_dictionary_ref=paths["route_dictionary"],
                events_ref=paths["events"],
                screening_checks_ref=paths["screening_checks"],
                diagnostic_ref=paths["diagnostic"],
                lane_dictionary=lane_ids,
                operator_dictionary=operator_ids,
            )
            trace_index["event_identity"] = {
                "shard_ordinal": shard_ordinal,
                "local_field": "event_id",
            }
            trace_index["schema_fingerprints"] = {
                "route_dictionary": route_schema,
                "events": event_schema,
                "screening_checks": check_schema,
                "diagnostic": diagnostic_schema,
            }
            paths["trace"] = self._write_json_artifact(
                directory, "trace", instance, seed, trace_index
            )
            if failure_payload is not None:
                paths["failure"] = self._write_json_artifact(
                    directory,
                    "failure",
                    instance,
                    seed,
                    failure_payload,
                    retention_class="critical",
                )
            self._write_v2_shard_manifest(
                directory=directory,
                instance=instance,
                seed=seed,
                shard_ordinal=shard_ordinal,
                worker_identity=worker_identity,
                artifact_start=shard_artifact_start,
                evidence_completeness="complete",
            )
            paths["shard_manifest"] = str(
                (
                    directory / _canonical_filename(self.context, "shard_manifest", instance, seed)
                ).relative_to(self.run_dir)
            )
            return paths
        except BaseException as error:
            for sink in (event_sink, checks_sink, diagnostic_sink):
                with contextlib.suppress(BaseException):
                    sink.close()
            failure = {
                "schema_version": ARTIFACT_STORAGE_V2,
                "run_label": self.context.run_label,
                "instance": instance,
                "seed": seed,
                "status": "partial",
                "evidence_completeness": "partial",
                "failure_reason": str(error),
            }
            failure_path = directory / _canonical_filename(self.context, "failure", instance, seed)
            _json_write(failure_path, failure)
            self._record_file(
                failure_path,
                artifact_type="failure",
                retention_class="critical",
                storage_format="json_control",
                compression="none",
                enforce_budget=False,
            )
            self._write_v2_shard_manifest(
                directory=directory,
                instance=instance,
                seed=seed,
                shard_ordinal=shard_ordinal,
                worker_identity=worker_identity,
                artifact_start=shard_artifact_start,
                evidence_completeness="partial",
            )
            raise

    def _write_v2_shard_manifest(
        self,
        *,
        directory: Path,
        instance: str,
        seed: int,
        shard_ordinal: int,
        worker_identity: str,
        artifact_start: int,
        evidence_completeness: str,
    ) -> None:
        shard_artifacts = tuple(self._artifacts[artifact_start:])
        manifest_path = directory / _canonical_filename(
            self.context, "shard_manifest", instance, seed
        )
        _json_write(
            manifest_path,
            {
                "schema_version": ARTIFACT_STORAGE_V2,
                "storage_policy_version": ARTIFACT_STORAGE_V2,
                "run_label": self.context.run_label,
                "instance": instance,
                "seed": seed,
                "shard_ordinal": shard_ordinal,
                "worker_identity": worker_identity,
                "event_identity": "shard_ordinal+shard_local_event_id",
                "evidence_completeness": evidence_completeness,
                "artifacts": [artifact.to_dict() for artifact in shard_artifacts],
            },
        )
        self._record_file(
            manifest_path,
            artifact_type="shard_manifest",
            retention_class="control",
            storage_format="json_control",
            compression="none",
            enforce_budget=evidence_completeness == "complete",
        )
        sidecar_path = manifest_path.with_suffix(".sha256")
        sidecar_path.write_text(_sha256(manifest_path) + "\n", encoding="utf-8")
        self._record_file(
            sidecar_path,
            artifact_type="shard_manifest_sidecar",
            retention_class="control",
            storage_format="sha256_control",
            compression="none",
            enforce_budget=evidence_completeness == "complete",
        )

    def finalize(
        self,
        *,
        status: str = "complete",
        evidence_completeness: str = "complete",
    ) -> ArtifactBundleResult:
        if evidence_completeness not in {"complete", "partial", "legacy_unknown"}:
            raise ValueError(f"unsupported evidence completeness: {evidence_completeness}")
        manifest_path, sidecar_path = self._write_manifest(
            status=status,
            evidence_completeness=evidence_completeness,
        )
        return ArtifactBundleResult(
            self.run_dir,
            manifest_path,
            sidecar_path,
            self.artifacts,
            evidence_completeness,
        )

    def _write_manifest(
        self,
        *,
        status: str,
        evidence_completeness: str,
    ) -> tuple[Path, Path]:
        control = self.run_dir / "control"
        control.mkdir(parents=True, exist_ok=True)
        manifest_path = control / _canonical_filename(self.context, "manifest")
        payload = {
            "schema_version": self.config.storage_policy_version,
            "storage_policy_version": self.config.storage_policy_version,
            "storage_format": CURRENT_STORAGE_FORMAT,
            "policy_compliance": "current",
            "run_label": self.context.run_label,
            "stage_id": self.context.stage_id,
            "component": self.context.component,
            "status": status,
            "evidence_completeness": evidence_completeness,
            "storage_policy": self.config.to_dict(),
            "artifact_status": {
                "failure": (
                    "present"
                    if any(artifact.artifact_type == "failure" for artifact in self._artifacts)
                    else "not_applicable"
                )
            },
            "artifacts": [artifact.to_dict() for artifact in self._artifacts],
        }
        _json_write(manifest_path, payload)
        sidecar_path = control / (
            _canonical_filename(self.context, "manifest", extension="json")[:-5] + ".sha256"
        )
        sidecar_path.write_text(_sha256(manifest_path) + "\n", encoding="utf-8")
        self._manifest_path = manifest_path
        return manifest_path, sidecar_path

    def _route_rows(
        self, route_dictionary: Mapping[str, Sequence[str]]
    ) -> tuple[list[dict[str, object]], dict[str, int]]:
        route_ids = {key: index for index, key in enumerate(sorted(route_dictionary), start=1)}
        rows = [
            {
                "route_id": route_id,
                "canonical_route_key": key,
                "route_digest": _payload_sha256(list(route_dictionary[key])),
                "customer_sequence": list(route_dictionary[key]),
            }
            for key, route_id in sorted(route_ids.items(), key=lambda item: item[1])
        ]
        return rows, route_ids

    def _normalise_diagnostic(
        self, row: Mapping[str, object], instance: str, seed: int
    ) -> dict[str, object]:
        return {
            "run_label": self.context.run_label,
            "instance": row.get("instance", instance),
            "seed": _as_int(row.get("seed", seed)),
            "lane": str(row.get("lane", "")),
            "iteration": _as_int(row.get("iteration")),
            "operator": str(row.get("operator", "")),
            "reason": str(row.get("reason", "")),
            "metric": str(row.get("metric", "count")),
            "count": _as_int(row.get("count")),
            "sum_value": _as_float(row.get("sum_value")),
            "min_value": _as_float(row.get("min_value")),
            "max_value": _as_float(row.get("max_value")),
        }

    def _write_json_artifact(
        self,
        directory: Path,
        artifact_type: str,
        instance: str,
        seed: int,
        payload: Mapping[str, object],
        *,
        retention_class: str = "critical",
    ) -> str:
        path = directory / _canonical_filename(self.context, artifact_type, instance, seed)
        _json_write(path, payload)
        artifact_subtype = (
            "compact_index_v1"
            if artifact_type == "trace" and self.context.component == "benchmark"
            else ""
        )
        self._record_file(
            path,
            artifact_type=artifact_type,
            artifact_subtype=artifact_subtype,
            retention_class=retention_class,
            storage_format="json_control" if artifact_type == "environment" else "json",
            compression="none",
        )
        return str(path.relative_to(self.run_dir))

    def _record_file(
        self,
        path: Path,
        *,
        artifact_type: str,
        retention_class: str,
        storage_format: str,
        compression: str,
        artifact_subtype: str = "",
        row_count: int | None = None,
        schema_fingerprint: str = "",
        enforce_budget: bool = True,
    ) -> None:
        relative = path.relative_to(self.run_dir).as_posix()
        byte_size = path.stat().st_size
        self._artifacts = [item for item in self._artifacts if item.relative_path != relative]
        reference = ArtifactReference(
            relative,
            artifact_type,
            retention_class,
            storage_format,
            compression,
            "complete",
            _sha256(path),
            byte_size,
            row_count,
            schema_fingerprint,
            artifact_subtype,
            self.config.storage_policy_version,
        )
        self._artifacts.append(reference)
        if self._current_instance is not None:
            self._instance_bytes += byte_size
        if enforce_budget:
            if (
                self._current_instance is not None
                and self._instance_bytes > self.config.per_instance_seed_max_bytes
            ):
                raise ArtifactBudgetExceeded(
                    f"instance/seed artifact budget exceeded for {self._current_instance}: "
                    f"{self._instance_bytes} > {self.config.per_instance_seed_max_bytes}"
                )
            total = sum(item.byte_size for item in self._artifacts)
            if total > self.config.per_run_max_bytes:
                raise ArtifactBudgetExceeded(
                    f"run artifact budget exceeded: {total} > {self.config.per_run_max_bytes}"
                )

    def _register_partial_control_files(self) -> None:
        """Make a partial manifest self-contained after a budget failure."""

        control = self.run_dir / "control"
        if not control.is_dir():
            return
        registered = {item.relative_path for item in self._artifacts}
        for path in sorted(control.iterdir()):
            if not path.is_file():
                continue
            relative = path.relative_to(self.run_dir).as_posix()
            if relative in registered or path.name.endswith("_manifest.json"):
                continue
            if "raw_per_run_results" in path.name:
                artifact_type = "raw_per_run_results"
            elif "per_run_results" in path.name:
                artifact_type = "per_run_results"
            else:
                continue
            self._record_file(
                path,
                artifact_type=artifact_type,
                retention_class="control",
                storage_format="csv_control",
                compression="none",
                enforce_budget=False,
            )
            registered.add(relative)


def build_stage03_critical_events(
    trace: object,
    neighborhood_events: Iterable[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    """Convert the legacy Stage03Trace collections into one critical stream.

    New bundles have one event stream.  This adapter intentionally accepts the
    v1/v2 object shape so existing solver code and historical readers remain
    compatible while the persisted representation stops duplicating records.
    """

    return list(iter_stage03_critical_events(trace, neighborhood_events))


def iter_stage03_critical_events(
    trace: object,
    neighborhood_events: Iterable[Mapping[str, object]] = (),
) -> Iterable[dict[str, object]]:
    """Yield the critical stream in timestamp order without a full copy/sort."""

    from dataclasses import asdict, fields

    def route_evaluations() -> Iterable[dict[str, object]]:
        for record in getattr(trace, "route_evaluations", ()):
            payload = dict(record.__dict__) if hasattr(record, "__dict__") else {}
            if not payload:
                payload = asdict(record)
            payload["record_type"] = "route_evaluation"
            yield payload

    def trace_events() -> Iterable[dict[str, object]]:
        for event in getattr(trace, "events", ()):
            yield {
                **dict(event),
                "record_type": str(event.get("event_type", "event")),
            }

    def screening_decisions() -> Iterable[dict[str, object]]:
        field_names: tuple[str, ...] | None = None
        for decision in getattr(trace, "screening_decisions", ()):
            if field_names is None:
                field_names = tuple(field.name for field in fields(decision))
            payload = {name: getattr(decision, name) for name in field_names}
            payload["checks"] = [
                {
                    "check": check.check,
                    "status": check.status,
                    "value": check.value,
                    "reason": check.reason,
                }
                for check in decision.checks
            ]
            payload["record_type"] = "screening_decision"
            payload["event_type"] = "screening_decision"
            yield payload

    def propagations() -> Iterable[dict[str, object]]:
        for propagation in getattr(trace, "incremental_propagations", ()):
            yield {
                **dict(propagation),
                "record_type": "incremental_propagation",
            }

    def neighborhood() -> Iterable[dict[str, object]]:
        for event in neighborhood_events:
            yield {**dict(event), "record_type": "neighborhood_event"}

    def keyed(
        stream: Iterable[dict[str, object]], rank: int
    ) -> Iterable[tuple[bool, float, int, int, dict[str, object]]]:
        for index, payload in enumerate(stream):
            timestamp = _event_timestamp(payload)
            yield (
                timestamp is None,
                timestamp or 0.0,
                rank,
                index,
                payload,
            )

    ordinary = iter(
        heapq.merge(
            keyed(route_evaluations(), 0),
            keyed(trace_events(), 1),
            keyed(propagations(), 3),
            keyed(neighborhood(), 4),
        )
    )
    screening = iter(keyed(screening_decisions(), 2))
    ordinary_item = next(ordinary, None)
    screening_item = next(screening, None)
    while ordinary_item is not None and screening_item is not None:
        if ordinary_item[:4] <= screening_item[:4]:
            yield ordinary_item[4]
            ordinary_item = next(ordinary, None)
        else:
            yield screening_item[4]
            screening_item = next(screening, None)
    while ordinary_item is not None:
        yield ordinary_item[4]
        ordinary_item = next(ordinary, None)
    while screening_item is not None:
        yield screening_item[4]
        screening_item = next(screening, None)


def find_manifest(run_dir: Path) -> Path:
    """Find a v2 control manifest, falling back to the legacy root manifest."""

    candidates = sorted(
        path
        for path in (run_dir / "control").glob("*_manifest.json")
        if not _is_appledouble_path(path.relative_to(run_dir))
    )
    if candidates:
        return candidates[0]
    legacy = run_dir / "manifest.json"
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(f"artifact manifest is missing under {run_dir}")


def verify_manifest(run_dir: Path) -> dict[str, Any]:
    """Verify a current bundle without accepting summary status as evidence."""

    manifest_path = find_manifest(run_dir)
    manifest = _json_read(manifest_path)
    if manifest.get("storage_policy_version") not in SUPPORTED_STORAGE_POLICIES:
        raise ArtifactIntegrityError(
            f"unsupported current artifact manifest: {manifest.get('storage_policy_version')}"
        )
    if manifest.get("storage_format") != CURRENT_STORAGE_FORMAT:
        raise ArtifactIntegrityError("current artifact manifest storage format is invalid")
    if manifest.get("policy_compliance") != "current":
        raise ArtifactIntegrityError("current artifact manifest policy compliance is invalid")
    policy_payload = manifest.get("storage_policy")
    if not isinstance(policy_payload, Mapping):
        raise ArtifactIntegrityError("current artifact manifest lacks storage policy")
    try:
        policy = ArtifactStorageConfig(**dict(policy_payload))
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            "current artifact manifest storage policy is invalid"
        ) from error
    canonical_policy = policy.to_dict()
    legacy_policy = dict(canonical_policy)
    legacy_policy.pop("screening_schema_version")
    if dict(policy_payload) not in (canonical_policy, legacy_policy):
        raise ArtifactIntegrityError("current artifact manifest storage policy is not canonical")
    artifact_status = manifest.get("artifact_status")
    if not isinstance(artifact_status, Mapping):
        raise ArtifactIntegrityError("current artifact manifest lacks artifact status")
    sidecar = manifest_path.with_suffix(".sha256")
    if not sidecar.is_file():
        sidecar = manifest_path.with_name(manifest_path.stem + "_sha256.txt")
    if not sidecar.is_file():
        sidecar = manifest_path.parent / "manifest.sha256"
    if not sidecar.is_file():
        raise ArtifactIntegrityError(f"artifact manifest sidecar is missing: {sidecar}")
    if sidecar.read_text(encoding="utf-8").strip() != _sha256(manifest_path):
        raise ArtifactIntegrityError("artifact manifest sidecar hash mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ArtifactIntegrityError("current artifact manifest has no artifacts list")
    listed_paths: set[str] = set()
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise ArtifactIntegrityError("artifact manifest entry must be an object")
        if item.get("storage_policy_version") != manifest.get("storage_policy_version"):
            raise ArtifactIntegrityError(
                "artifact storage policy version does not match the manifest"
            )
        relative = str(item["relative_path"])
        listed_paths.add(relative)
        path = _safe_artifact_path(run_dir, relative)
        if not path.is_file():
            raise ArtifactIntegrityError(f"artifact listed by manifest is missing: {relative}")
        observed = _sha256(path)
        if observed != str(item["checksum"]):
            raise ArtifactIntegrityError(f"artifact checksum mismatch: {relative}")
        if path.stat().st_size != int(item["byte_size"]):
            raise ArtifactIntegrityError(f"artifact byte size mismatch: {relative}")
        if str(item.get("storage_format", "")).startswith("parquet"):
            parquet = pq.ParquetFile(path)
            actual_rows = parquet.metadata.num_rows
            if item.get("row_count") is not None and actual_rows != int(item["row_count"]):
                raise ArtifactIntegrityError(f"artifact row count mismatch: {relative}")
            expected_schema = str(item.get("schema_fingerprint", ""))
            if expected_schema and _schema_fingerprint(parquet.schema_arrow) != expected_schema:
                raise ArtifactIntegrityError(f"artifact schema fingerprint mismatch: {relative}")
    post_manifest_envelope_paths = _verify_post_manifest_persistence_envelopes(
        run_dir,
        run_label=str(manifest.get("run_label", "")),
    )
    for path in run_dir.rglob("*"):
        if not path.is_file() or path in {manifest_path, sidecar}:
            continue
        relative = path.relative_to(run_dir).as_posix()
        if _is_appledouble_path(path.relative_to(run_dir)):
            continue
        if relative == "review" or relative.startswith("review/"):
            continue
        if relative in post_manifest_envelope_paths:
            continue
        if relative not in listed_paths:
            raise ArtifactIntegrityError(f"unlisted artifact is present: {relative}")
    has_failure = any(
        isinstance(item, Mapping) and item.get("artifact_type") == "failure" for item in artifacts
    )
    expected_failure_status = "present" if has_failure else "not_applicable"
    if artifact_status.get("failure") != expected_failure_status:
        raise ArtifactIntegrityError("failure artifact status does not match manifest contents")
    return manifest


def _verify_post_manifest_persistence_envelopes(
    run_dir: Path,
    *,
    run_label: str,
) -> set[str]:
    """Verify the narrow signed envelope written after the primary manifest.

    It cannot be listed by the primary manifest without creating a self-reference.
    No other post-manifest file is exempted from the unlisted-artifact check.
    """

    if not run_label.startswith("stage05.2_"):
        return set()
    control = run_dir / "control"
    batch_match = re.fullmatch(r"batch[0-9]{4}", run_dir.name)
    is_batch_root = batch_match is not None and run_dir.parent.name == run_label
    expected_subject = run_dir.name if is_batch_root else None
    expected_stem = (
        f"{run_label}_{expected_subject}_persistence_attribution"
        if expected_subject is not None
        else f"{run_label}_persistence_attribution"
    )
    candidates = (
        [
            path
            for path in control.glob(f"{run_label}*_persistence_attribution.*")
            if path.is_file()
        ]
        if control.is_dir()
        else []
    )
    allowed_names = {f"{expected_stem}.json", f"{expected_stem}.sha256"}
    if any(path.name not in allowed_names for path in candidates):
        raise ArtifactIntegrityError(
            "post-manifest persistence attribution does not match the directory role"
        )
    stems = {path.name.rsplit(".", 1)[0] for path in candidates}
    verified: set[str] = set()
    campaign_attribution: Mapping[str, object] | None = None
    for stem in stems:
        payload_path = control / f"{stem}.json"
        sidecar_path = control / f"{stem}.sha256"
        if not payload_path.is_file() or not sidecar_path.is_file():
            raise ArtifactIntegrityError(
                "post-manifest persistence attribution must include JSON and sidecar"
            )
        if not signed_sidecar_matches(payload_path, sidecar_path):
            raise ArtifactIntegrityError("persistence attribution sidecar hash mismatch")
        try:
            attribution = json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError("persistence attribution JSON is invalid") from error
        if not isinstance(attribution, Mapping) or attribution.get("run_label") != run_label:
            raise ArtifactIntegrityError("persistence attribution identity is invalid")
        if expected_subject is not None and attribution.get("subject_id") != expected_subject:
            raise ArtifactIntegrityError("persistence attribution batch identity is invalid")
        if expected_subject is None:
            campaign_attribution = attribution
        verified.update(
            {
                payload_path.relative_to(run_dir).as_posix(),
                sidecar_path.relative_to(run_dir).as_posix(),
            }
        )
    batch_payload = run_dir / "batch_persistence_envelope.json"
    batch_sidecar = run_dir / "batch_persistence_envelope.sha256"
    batch_envelope_payload: Mapping[str, object] | None = None
    if batch_payload.exists() or batch_sidecar.exists():
        if expected_subject is None:
            raise ArtifactIntegrityError(
                "batch persistence envelope is only valid at a canonical batch root"
            )
        if not batch_payload.is_file() or not batch_sidecar.is_file():
            raise ArtifactIntegrityError(
                "batch persistence envelope must include JSON and sidecar"
            )
        if not signed_sidecar_matches(batch_payload, batch_sidecar):
            raise ArtifactIntegrityError("batch persistence envelope sidecar hash mismatch")
        try:
            decoded_payload = json.loads(batch_payload.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError("batch persistence envelope JSON is invalid") from error
        if (
            not isinstance(decoded_payload, Mapping)
            or decoded_payload.get("schema_version")
            != "stage05.2-batch-persistence-envelope-v1"
            or decoded_payload.get("run_label") != run_label
            or decoded_payload.get("batch_id") != expected_subject
        ):
            raise ArtifactIntegrityError("batch persistence envelope identity is invalid")
        batch_envelope_payload = decoded_payload
        verified.update(
            {
                batch_payload.relative_to(run_dir).as_posix(),
                batch_sidecar.relative_to(run_dir).as_posix(),
            }
        )
    batch_manifest = run_dir / "batch_manifest.json"
    batch_manifest_sidecar = run_dir / "batch_manifest.sha256"
    if batch_envelope_payload is not None and not (
        batch_manifest.is_file() and batch_manifest_sidecar.is_file()
    ):
        raise ArtifactIntegrityError(
            "batch persistence envelope requires the archived batch manifest"
        )
    if batch_manifest.exists() or batch_manifest_sidecar.exists():
        if expected_subject is None:
            raise ArtifactIntegrityError(
                "batch manifest envelope is only valid at a canonical batch root"
            )
        if not batch_manifest.is_file() or not batch_manifest_sidecar.is_file():
            raise ArtifactIntegrityError("batch manifest must include JSON and sidecar")
        manifest_sha = _sha256(batch_manifest)
        if not signed_sidecar_matches(batch_manifest, batch_manifest_sidecar):
            raise ArtifactIntegrityError("batch manifest sidecar hash mismatch")
        try:
            batch_state = json.loads(batch_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError("batch manifest JSON is invalid") from error
        expected_fields = {
            "schema_version",
            "run_label",
            "batch_id",
            "status",
            "root_alias",
            "archive_root_alias",
            "logical_path",
            "volume_identity",
            "shard_ids",
            "estimated_bytes",
            "checksum_sha256",
            "actual_bytes",
            "row_count",
            "physical_schema",
            "resource_summary_sha256",
            "persistence_attribution_sha256",
            "control_persistence_seconds",
            "persistence_ratio",
            "shard_manifest_sha256_by_id",
            "shard_actual_bytes_by_id",
            "transfer_mode",
            "archive_transfer_seconds",
            "failure_reason",
        }
        root_alias = batch_state.get("root_alias") if isinstance(batch_state, Mapping) else None
        volume = batch_state.get("volume_identity") if isinstance(batch_state, Mapping) else None
        status = batch_state.get("status") if isinstance(batch_state, Mapping) else None
        transfer_mode = (
            batch_state.get("transfer_mode") if isinstance(batch_state, Mapping) else None
        )
        archive_seconds = (
            batch_state.get("archive_transfer_seconds")
            if isinstance(batch_state, Mapping)
            else None
        )
        if (
            not isinstance(batch_state, Mapping)
            or set(batch_state) != expected_fields
            or batch_state.get("schema_version") != "stage05.2-batch-manifest-v1"
            or batch_state.get("run_label") != run_label
            or batch_state.get("batch_id") != expected_subject
            or status not in {"verified", "archived"}
            or batch_state.get("logical_path") != f"{run_label}/{expected_subject}"
            or not isinstance(root_alias, str)
            or not isinstance(volume, Mapping)
            or set(volume) != {"device_uuid", "filesystem"}
            or not all(isinstance(value, str) and value for value in volume.values())
            or batch_state.get("failure_reason") is not None
            or (
                status == "verified"
                and (transfer_mode is not None or archive_seconds is not None)
            )
            or (
                status == "archived"
                and (
                    root_alias != batch_state.get("archive_root_alias")
                    or transfer_mode
                    not in {"same_volume_atomic_rename", "cross_volume_verified_copy"}
                    or isinstance(archive_seconds, bool)
                    or not isinstance(archive_seconds, int | float)
                    or archive_seconds < 0
                )
            )
            or (batch_envelope_payload is not None and status != "archived")
        ):
            raise ArtifactIntegrityError("batch manifest identity/status/root is invalid")
        if (
            batch_envelope_payload is not None
            and batch_envelope_payload.get("archived_manifest_sha256") != manifest_sha
        ):
            raise ArtifactIntegrityError(
                "batch persistence envelope does not bind the archived batch manifest"
            )
        verified.update(
            {
                batch_manifest.relative_to(run_dir).as_posix(),
                batch_manifest_sidecar.relative_to(run_dir).as_posix(),
            }
        )
    campaign_manifest = run_dir / "campaign_manifest.json"
    campaign_sidecar = run_dir / "campaign_manifest.sha256"
    if campaign_manifest.exists() or campaign_sidecar.exists():
        if expected_subject is not None or not run_label.startswith("stage05.2_benchmark_"):
            raise ArtifactIntegrityError(
                "campaign manifest envelope is only valid at a benchmark campaign root"
            )
        if not campaign_manifest.is_file() or not campaign_sidecar.is_file():
            raise ArtifactIntegrityError("campaign manifest must include JSON and sidecar")
        if not signed_sidecar_matches(campaign_manifest, campaign_sidecar):
            raise ArtifactIntegrityError("campaign manifest sidecar hash mismatch")
        try:
            campaign_state = json.loads(campaign_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError("campaign manifest JSON is invalid") from error
        if (
            not isinstance(campaign_state, Mapping)
            or campaign_state.get("schema_version")
            != "stage05.2-campaign-manifest-v1"
            or campaign_state.get("run_label") != run_label
            or campaign_state.get("status") not in {"planned", "complete", "failed"}
        ):
            raise ArtifactIntegrityError("campaign manifest identity/status is invalid")
        if campaign_state.get("status") == "complete":
            primary_manifest_path = find_manifest(run_dir)
            persistence_ratio = (
                campaign_attribution.get("persistence_ratio")
                if campaign_attribution is not None
                else None
            )
            if (
                campaign_attribution is None
                or campaign_attribution.get("subject_id") != "campaign"
                or campaign_attribution.get("primary_manifest_relative_path")
                != primary_manifest_path.relative_to(run_dir).as_posix()
                or campaign_attribution.get("primary_manifest_sha256")
                != _sha256(primary_manifest_path)
                or isinstance(persistence_ratio, bool)
                or not isinstance(persistence_ratio, int | float)
                or not 0.0 <= float(persistence_ratio) <= 0.30
            ):
                raise ArtifactIntegrityError(
                    "complete campaign is not committed after a passing attribution gate"
                )
        verified.update(
            {
                campaign_manifest.relative_to(run_dir).as_posix(),
                campaign_sidecar.relative_to(run_dir).as_posix(),
            }
        )
    return verified


def _verify_legacy_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    sidecar_path = run_dir / "manifest.sha256"
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise ArtifactIntegrityError(f"legacy manifest or sidecar is missing under {run_dir}")
    if sidecar_path.read_text(encoding="utf-8").strip() != _sha256(manifest_path):
        raise ArtifactIntegrityError("legacy manifest sidecar hash mismatch")
    manifest = _json_read(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ArtifactIntegrityError("legacy artifact manifest has no files mapping")
    for relative, expected in files.items():
        path = run_dir / str(relative)
        if not path.is_file() or _sha256(path) != str(expected):
            raise ArtifactIntegrityError(f"legacy artifact checksum mismatch: {relative}")
    return manifest


class _RegisteredStableRouteIds(Mapping[str, int]):
    """Content-derived route IDs backed only by registered ID digests."""

    def __init__(self, registered: Mapping[int, str]) -> None:
        self._registered = registered

    def __getitem__(self, key: str) -> int:
        route_id = _stable_route_id(key)
        if route_id not in self._registered:
            raise KeyError(key)
        return route_id

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return len(self._registered)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and _stable_route_id(key) in self._registered


class _DiskBackedRouteKeyStore(Mapping[int, str]):
    """Bounded-memory, exact route-ID dictionary for streamed replay."""

    def __init__(
        self,
        *,
        scratch_root: Path | None,
        cache_entries: int = ROUTE_IDENTITY_HOT_CACHE_ENTRIES,
    ) -> None:
        if cache_entries <= 0:
            raise ValueError("route key cache_entries must be positive")
        if scratch_root is not None:
            scratch_root.mkdir(parents=True, exist_ok=True)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="evrptw-route-key-replay-",
            dir=scratch_root,
        )
        database_path = Path(self._temporary_directory.name) / "routes.sqlite3"
        self._connection = sqlite3.connect(database_path)
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._connection.execute("PRAGMA temp_store=FILE")
        self._connection.execute("PRAGMA cache_size=-2048")
        self._connection.execute(
            "CREATE TABLE routes ("
            "route_id INTEGER PRIMARY KEY, canonical_route_key TEXT NOT NULL UNIQUE"
            ")"
        )
        self._cache_entries = cache_entries
        self._cache: OrderedDict[int, str] = OrderedDict()
        self._route_count = 0
        self._closed = False

    def register(
        self,
        row: Mapping[str, object],
        *,
        require_canonical_key: bool,
    ) -> None:
        raw_route_id = row.get("route_id")
        if (
            isinstance(raw_route_id, bool)
            or not isinstance(raw_route_id, int)
            or raw_route_id <= 0
        ):
            raise ArtifactIntegrityError("route dictionary ID must be a positive integer")
        route_key = row.get("canonical_route_key")
        raw_sequence = row.get("customer_sequence")
        route_digest = row.get("route_digest")
        if not isinstance(route_key, str) or not route_key:
            raise ArtifactIntegrityError("route dictionary key must be non-empty text")
        if not isinstance(raw_sequence, list) or any(
            not isinstance(customer, str) for customer in raw_sequence
        ):
            raise ArtifactIntegrityError("route dictionary sequence must be a string list")
        sequence = tuple(raw_sequence)
        if require_canonical_key and _route_sequence_from_key(route_key) != sequence:
            raise ArtifactIntegrityError("route dictionary key/sequence mismatch")
        if route_digest != _payload_sha256(list(sequence)):
            raise ArtifactIntegrityError("route dictionary digest mismatch")
        try:
            self._connection.execute(
                "INSERT INTO routes(route_id, canonical_route_key) VALUES (?, ?)",
                (raw_route_id, route_key),
            )
        except sqlite3.IntegrityError as error:
            raise ArtifactIntegrityError(
                "route dictionary IDs and keys must be unique"
            ) from error
        self._route_count += 1
        self._remember(raw_route_id, route_key)

    def close(self) -> None:
        if self._closed:
            return
        self._connection.close()
        self._temporary_directory.cleanup()
        self._cache.clear()
        self._closed = True

    def __getitem__(self, route_id: int) -> str:
        cached = self._cache.get(route_id)
        if cached is not None:
            self._cache.move_to_end(route_id)
            return cached
        stored = self._connection.execute(
            "SELECT canonical_route_key FROM routes WHERE route_id = ?",
            (route_id,),
        ).fetchone()
        if stored is None:
            raise KeyError(route_id)
        route_key = str(stored[0])
        self._remember(route_id, route_key)
        return route_key

    def __iter__(self) -> Iterator[int]:
        return iter(())

    def __len__(self) -> int:
        return self._route_count

    def _remember(self, route_id: int, route_key: str) -> None:
        self._cache[route_id] = route_key
        self._cache.move_to_end(route_id)
        if len(self._cache) > self._cache_entries:
            self._cache.popitem(last=False)


class _DiskBackedRouteIdentityStore(Mapping[int, str]):
    """Exact route/call identity state with a fixed-size in-process hot set.

    Stage 5.2 may observe more unique routes than one Parquet row group.  The
    collision state therefore lives in SQLite on the shard's own volume.  The
    database is runtime scratch only and is removed before the shard manifest
    is sealed.
    """

    def __init__(
        self,
        *,
        scratch_root: Path,
        cache_entries: int = ROUTE_IDENTITY_HOT_CACHE_ENTRIES,
        memory_entries: int = ROUTE_IDENTITY_MEMORY_ENTRIES,
    ) -> None:
        if cache_entries <= 0:
            raise ValueError("route identity cache_entries must be positive")
        if memory_entries <= 0:
            raise ValueError("route identity memory_entries must be positive")
        scratch_root.mkdir(parents=True, exist_ok=True)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="evrptw-route-identities-",
            dir=scratch_root,
        )
        self._database_path = Path(self._temporary_directory.name) / "identities.sqlite3"
        self._connection = sqlite3.connect(self._database_path)
        self._connection.execute("PRAGMA journal_mode=OFF")
        self._connection.execute("PRAGMA synchronous=OFF")
        self._connection.execute("PRAGMA temp_store=MEMORY")
        self._connection.execute("PRAGMA cache_size=-2048")
        self._connection.execute(
            "CREATE TABLE routes (route_id INTEGER PRIMARY KEY, digest TEXT NOT NULL)"
        )
        self._connection.execute(
            "CREATE TABLE unique_identities ("
            "axis TEXT NOT NULL, semantics TEXT NOT NULL, identity_digest BLOB NOT NULL, "
            "payload BLOB NOT NULL, PRIMARY KEY(axis, semantics, identity_digest)) WITHOUT ROWID"
        )
        self._cache_entries = cache_entries
        self._memory_entries = memory_entries
        self._route_memory: dict[int, str] = {}
        self._routes_spilled = False
        self._route_cache: OrderedDict[int, str] = OrderedDict()
        self._unique_counts: Counter[tuple[str, str]] = Counter()
        self._route_count = 0
        self._closed = False

    @property
    def scratch_path(self) -> Path:
        return Path(self._temporary_directory.name)

    @property
    def hot_entries(self) -> int:
        return len(self._route_memory) + len(self._route_cache) + len(self._unique_counts)

    @property
    def hot_entry_limit(self) -> int:
        return (
            self._memory_entries
            + self._cache_entries
            + ROUTE_IDENTITY_COUNTER_NAMESPACES
        )

    def register_route(self, route_id: int, route_digest: str) -> bool:
        if not self._routes_spilled:
            previous = self._route_memory.get(route_id)
            if previous is not None:
                if previous != route_digest:
                    raise ArtifactIntegrityError(
                        f"stable route ID collision for route ID {route_id}"
                    )
                return False
            if len(self._route_memory) < self._memory_entries:
                self._route_memory[route_id] = route_digest
                self._route_count += 1
                return True
            self._spill_route_memory()
        try:
            previous = self[route_id]
        except KeyError:
            previous = None
        if previous is not None:
            if previous != route_digest:
                raise ArtifactIntegrityError(
                    f"stable route ID collision for route ID {route_id}"
                )
            return False
        self._connection.execute(
            "INSERT INTO routes(route_id, digest) VALUES (?, ?)",
            (route_id, route_digest),
        )
        self._route_count += 1
        self._remember_route(route_id, route_digest)
        return True

    def register_unique_identity(
        self,
        *,
        axis: str,
        semantics: str,
        identity: tuple[str, ...],
    ) -> bool:
        if semantics not in UNIQUE_ROUTE_IDENTITY_SEMANTICS:
            raise ValueError(f"unsupported unique route identity semantics: {semantics}")
        encoded = orjson.dumps(identity)
        digest = hashlib.sha256(encoded).digest()
        existing = self._connection.execute(
            "SELECT payload FROM unique_identities "
            "WHERE axis = ? AND semantics = ? AND identity_digest = ?",
            (axis, semantics, digest),
        ).fetchone()
        if existing is not None:
            if bytes(existing[0]) != encoded:
                raise ArtifactIntegrityError("unique route identity SHA-256 collision")
            return False
        namespace = (axis, semantics)
        if (
            namespace not in self._unique_counts
            and len(self._unique_counts) >= ROUTE_IDENTITY_COUNTER_NAMESPACES
        ):
            raise ArtifactIntegrityError("unique route identity counter namespace bound exceeded")
        self._connection.execute(
            "INSERT INTO unique_identities(axis, semantics, identity_digest, payload) "
            "VALUES (?, ?, ?, ?)",
            (axis, semantics, digest, encoded),
        )
        self._unique_counts[namespace] += 1
        return True

    def unique_identity_count(self, *, axis: str, semantics: str) -> int:
        return int(self._unique_counts[(axis, semantics)])

    def close(self) -> None:
        if self._closed:
            return
        self._connection.close()
        self._temporary_directory.cleanup()
        self._route_memory.clear()
        self._route_cache.clear()
        self._closed = True

    def __getitem__(self, route_id: int) -> str:
        in_memory = self._route_memory.get(route_id)
        if in_memory is not None:
            return in_memory
        cached = self._route_cache.get(route_id)
        if cached is not None:
            self._route_cache.move_to_end(route_id)
            return cached
        stored = self._connection.execute(
            "SELECT digest FROM routes WHERE route_id = ?", (route_id,)
        ).fetchone()
        if stored is None:
            raise KeyError(route_id)
        digest = str(stored[0])
        self._remember_route(route_id, digest)
        return digest

    def __iter__(self) -> Iterator[int]:
        return iter(())

    def __len__(self) -> int:
        return self._route_count

    def __contains__(self, route_id: object) -> bool:
        if isinstance(route_id, bool) or not isinstance(route_id, int):
            return False
        try:
            self[route_id]
        except KeyError:
            return False
        return True

    def _remember_route(self, route_id: int, digest: str) -> None:
        self._route_cache[route_id] = digest
        self._route_cache.move_to_end(route_id)
        if len(self._route_cache) > self._cache_entries:
            self._route_cache.popitem(last=False)

    def _spill_route_memory(self) -> None:
        if self._routes_spilled:
            return
        self._connection.executemany(
            "INSERT INTO routes(route_id, digest) VALUES (?, ?)",
            self._route_memory.items(),
        )
        self._route_memory.clear()
        self._routes_spilled = True


class ArtifactV2ShardSession:
    """Incremental ``open -> append -> flush -> finalize/abort`` v2 shard.

    The session owns all Parquet writers for one ``(instance, seed)`` shard.
    Route IDs are content-derived so an axis can be released immediately after
    it is appended; only ID-to-digest collision state survives across axes.
    """

    def __init__(
        self,
        owner: ArtifactBundleWriter,
        *,
        instance: str,
        seed: int,
        shard_ordinal: int,
        worker_identity: str,
    ) -> None:
        self._owner = owner
        self.instance = instance
        self.seed = seed
        self.shard_ordinal = shard_ordinal
        self.worker_identity = worker_identity
        owner._current_instance = (instance, seed)
        owner._instance_bytes = 0
        owner._next_event_id = 1
        self._artifact_start = len(owner._artifacts)
        self._directory = owner.run_dir / instance / str(seed)
        self._directory.mkdir(parents=True, exist_ok=True)
        self.paths = canonical_artifact_paths(owner.context, instance, seed)
        self._route_sink = _StreamingParquetSink(
            owner.run_dir / self.paths["route_dictionary"],
            ROUTE_DICTIONARY_SCHEMA,
            owner.config,
        )
        self._event_sink = _StreamingParquetSink(
            owner.run_dir / self.paths["events"], EVENTS_SCHEMA, owner.config
        )
        self._checks_sink = _StreamingParquetSink(
            owner.run_dir / self.paths["screening_checks"],
            SCREENING_CHECKS_SCHEMA,
            owner.config,
        )
        self._screening_sink: _StreamingParquetSink | None = None
        self._screening_definitions_sink: _StreamingParquetSink | None = None
        self._screening_occurrences_sink: _StreamingParquetSink | None = None
        if owner.config.screening_schema_version == SCREENING_DECISIONS_V3:
            self._screening_definitions_sink = _StreamingParquetSink(
                owner.run_dir / self.paths["screening_definitions"],
                V3_SCREENING_DEFINITIONS_SCHEMA,
                owner.config,
            )
            self._screening_occurrences_sink = _StreamingParquetSink(
                owner.run_dir / self.paths["screening_occurrences"],
                V3_SCREENING_OCCURRENCES_SCHEMA,
                owner.config,
            )
        else:
            self._screening_sink = _StreamingParquetSink(
                owner.run_dir / self.paths["screening_decisions"],
                V2_SCREENING_DECISIONS_SCHEMA,
                owner.config,
            )
        self._diagnostic_sink = _StreamingParquetSink(
            owner.run_dir / self.paths["diagnostic"],
            DIAGNOSTIC_SCHEMA,
            owner.config,
        )
        self._route_digests = _DiskBackedRouteIdentityStore(scratch_root=self._directory)
        self._route_ids = _RegisteredStableRouteIds(self._route_digests)
        self._resolved_route_ids: OrderedDict[str, int] = OrderedDict()
        self._lane_ids: dict[str, int] = {}
        self._operator_ids: dict[str, int] = {}
        self._screening_definition_cache: dict[
            tuple[object, ...], tuple[object, int, bytes, str]
        ] = {}
        self._neighborhood_extras_cache: dict[tuple[object, ...], str] = {}
        self._neighborhood_row_cache: dict[
            tuple[object, ...], tuple[object, ...]
        ] = {}
        self._screening_definition_store: _BoundedScreeningDefinitionStore | None = None
        self._pending_route_rows: list[tuple[object, ...]] = []
        self._pending_check_rows: list[tuple[object, ...]] = []
        self._active_sinks: list[_StreamingParquetSink] = []
        self._max_buffered_groups_observed = 0
        self._max_pending_screening_transaction_rows_observed = 0
        self._state = "open"

    @property
    def screening_schema_version(self) -> str:
        return self._owner.config.screening_schema_version

    def append(
        self,
        *,
        route_dictionary: Mapping[str, Sequence[str]],
        critical_events: Iterable[Mapping[str, object] | _BufferedScreeningDecision],
        diagnostic_rows: Iterable[Mapping[str, object]] = (),
        cache_lookups_coalesced: bool = False,
    ) -> int:
        """Append one logical axis and return its persisted event-row count."""

        self._require_open()
        for key, raw_sequence in route_dictionary.items():
            self._register_route(key, tuple(raw_sequence))

        count = 0
        pending_event_rows: list[tuple[object, ...]] = []
        pending_screening_definitions: list[_PendingScreeningDefinition] = []
        pending_screening_occurrences: list[tuple[object, ...]] = []

        def flush_screening_transaction() -> None:
            nonlocal pending_screening_definitions, pending_screening_occurrences
            if self._screening_definitions_sink is None:
                return
            if self._screening_occurrences_sink is None:
                raise RuntimeError("screening occurrence sink is unavailable")
            candidates = pending_screening_definitions
            occurrences = pending_screening_occurrences
            pending_screening_definitions = []
            pending_screening_occurrences = []
            if candidates:
                if self._screening_definition_store is None:
                    raise RuntimeError("screening definition store is unavailable")
                inserted_ids = self._screening_definition_store.register_many(candidates)
                definitions = tuple(
                    candidate.row
                    for candidate in candidates
                    if candidate.definition_id in inserted_ids
                )
                self._append_buffered_value_rows(
                    self._screening_definitions_sink,
                    definitions,
                )
            self._append_buffered_value_rows(
                self._screening_occurrences_sink,
                occurrences,
                trusted_width=True,
            )

        def flush_event_transaction() -> None:
            rows = tuple(pending_event_rows)
            pending_event_rows.clear()
            self._append_buffered_value_rows(self._event_sink, rows)

        next_event_id = self._owner._next_event_id
        events = (
            critical_events
            if cache_lookups_coalesced
            else _iter_coalesced_cache_lookup_events(critical_events)
        )
        for event in events:
            if isinstance(event, tuple):
                if self._screening_definitions_sink is None:
                    raise RuntimeError("buffered screening decisions require v3 storage")
                event_id = next_event_id
                next_event_id += 1
                compact_key = (event[2], event[4], event[1], id(event[7]))
                cached_entry = self._screening_definition_cache.get(compact_key)
                if cached_entry is None:
                    occurrence, definition = self._buffered_screening_decision_row(
                        event,
                        event_id=event_id,
                        cache_key=compact_key,
                    )
                else:
                    occurrence = (
                        event_id,
                        cached_entry[1],
                        event[5],
                        event[6],
                        event[3],
                        event[0],
                    )
                    definition = None
                pending_screening_occurrences.append(occurrence)
                if definition is not None:
                    pending_screening_definitions.append(definition)
                pending_count = len(pending_screening_occurrences)
                if pending_count > self._max_pending_screening_transaction_rows_observed:
                    self._max_pending_screening_transaction_rows_observed = pending_count
                if pending_count >= LIVE_SCREENING_TRANSACTION_ROWS:
                    flush_screening_transaction()
                count += 1
                continue
            event_get = event.get
            event_type = event_get("event_type")
            screening_event = event_type == "screening_decision"
            if screening_event:
                route_key = str(event_get("route_key", ""))
                screening_route_id = self._resolve_route_id(route_key) if route_key else None
            else:
                event_route_ids: dict[str, int] = {}
                for route_key in _event_route_keys(event):
                    if route_key:
                        route_id = self._resolve_route_id(route_key)
                        if route_id not in self._route_digests:
                            self._register_route(
                                route_key,
                                _route_sequence_from_key(route_key),
                                route_id=route_id,
                                validate_key=False,
                            )
                        event_route_ids[route_key] = route_id
            lane = str(event_get("lane", ""))
            operator = str(event_get("operator", ""))
            if lane not in self._lane_ids:
                self._lane_ids[lane] = _stable_dictionary_id(f"lane:{lane}")
            if operator not in self._operator_ids:
                self._operator_ids[operator] = _stable_dictionary_id(f"operator:{operator}")
            if not screening_event:
                _validate_event_routes(event, event_route_ids)
            if (
                event_type == "route_evaluation"
                and event_get("kind") == "exact_call"
            ):
                self._register_unique_route_evaluation_identity(event)
            event_id = next_event_id
            next_event_id += 1
            screening_definition: _PendingScreeningDefinition | None = None
            if screening_event:
                normalized_event, screening_definition = self._screening_decision_row(
                    event,
                    event_id=event_id,
                    route_key=route_key,
                    route_id=screening_route_id,
                    lane_id=self._lane_ids[lane],
                    operator_id=self._operator_ids[operator],
                )
            else:
                normalized_event = _normalise_event_values(
                    event,
                    event_id=event_id,
                    route_ids=event_route_ids,
                    lane_ids=self._lane_ids,
                    operator_ids=self._operator_ids,
                    neighborhood_extras_cache=self._neighborhood_extras_cache,
                    neighborhood_row_cache=self._neighborhood_row_cache,
                )
            if screening_event:
                screening_sink = (
                    self._screening_occurrences_sink
                    if self._screening_occurrences_sink is not None
                    else self._screening_sink
                )
                if screening_sink is None:
                    raise RuntimeError("screening occurrence sink is unavailable")
                if self._screening_definitions_sink is not None:
                    if not isinstance(normalized_event, tuple):
                        raise RuntimeError("v3 screening occurrence must be a typed row")
                    pending_screening_occurrences.append(normalized_event)
                    self._max_pending_screening_transaction_rows_observed = max(
                        self._max_pending_screening_transaction_rows_observed,
                        len(pending_screening_occurrences),
                    )
                else:
                    if not isinstance(normalized_event, Mapping):
                        raise RuntimeError("v2 screening occurrence must be a row mapping")
                    self._append_buffered(screening_sink, normalized_event)
                if screening_definition is not None:
                    pending_screening_definitions.append(screening_definition)
                if (
                    len(pending_screening_occurrences)
                    >= LIVE_SCREENING_TRANSACTION_ROWS
                ):
                    flush_screening_transaction()
            else:
                if not isinstance(normalized_event, tuple):
                    raise RuntimeError("critical event must be a typed row")
                pending_event_rows.append(normalized_event)
                if len(pending_event_rows) >= V2_PARQUET_ROW_GROUP_SIZE:
                    flush_event_transaction()
            checks = event_get("checks")
            if not screening_event and isinstance(checks, (list, tuple)):
                decision_id = _as_int(event_get("decision_id"))
                for index, check in enumerate(checks):
                    if isinstance(check, Mapping):
                        normalized_check = _normalise_check(
                            check,
                            event_id=event_id,
                            decision_id=decision_id,
                            index=index,
                        )
                        self._pending_check_rows.append(
                            tuple(
                                normalized_check.get(name)
                                for name in SCREENING_CHECKS_SCHEMA.names
                            )
                        )
                        if len(self._pending_check_rows) >= V2_PARQUET_ROW_GROUP_SIZE:
                            self._flush_pending_check_rows()
            count += 1
        self._owner._next_event_id = next_event_id
        flush_screening_transaction()
        flush_event_transaction()
        for row in diagnostic_rows:
            self._append_buffered(
                self._diagnostic_sink,
                self._owner._normalise_diagnostic(row, self.instance, self.seed),
            )
        return count

    def _buffered_screening_decision_row(
        self,
        event: _BufferedScreeningDecision,
        event_id: int,
        cache_key: tuple[object, ...],
    ) -> tuple[tuple[object, ...], _PendingScreeningDefinition | None]:
        precomputed = event[7]
        lane = event[2]
        operator = event[4]
        if lane not in self._lane_ids:
            self._lane_ids[lane] = _stable_dictionary_id(f"lane:{lane}")
        if operator not in self._operator_ids:
            self._operator_ids[operator] = _stable_dictionary_id(f"operator:{operator}")
        route_id = self._resolve_route_id(event[1])
        lane_id = self._lane_ids[lane]
        operator_id = self._operator_ids[operator]
        definition_key = (lane_id, operator_id, route_id, *precomputed.tail)
        if route_id not in self._route_digests:
            sequence = _route_sequence_from_key(event[1])
            self._register_route(
                event[1],
                sequence,
                route_id=route_id,
                validate_key=False,
            )
        definition = _screening_definition_from_cache_key(definition_key)
        definition_id, definition_json, definition_digest = (
            _screening_definition_identity(definition)
        )
        self._screening_definition_cache[cache_key] = (
            precomputed,
            definition_id,
            definition_json,
            definition_digest,
        )
        if len(self._screening_definition_cache) > SCREENING_DEFINITION_HOT_CACHE_ENTRIES:
            self._screening_definition_cache.pop(next(iter(self._screening_definition_cache)))
        if self._screening_definition_store is None:
            self._screening_definition_store = _BoundedScreeningDefinitionStore(
                cache_entries=1,
                scratch_root=self._directory,
            )
        pending = _PendingScreeningDefinition(
            definition_id=definition_id,
            encoded=definition_json,
            payload=definition,
            row=(
                definition_id,
                *(definition[name] for name in V3_SCREENING_DEFINITIONS_SCHEMA.names[1:]),
            ),
        )
        return (
            (event_id, definition_id, event[5], event[6], event[3], event[0]),
            pending,
        )

    def append_transcoded_v2_batches(
        self,
        *,
        route_batches: Iterable[pa.RecordBatch],
        critical_event_batches: Iterable[pa.RecordBatch],
        screening_check_batches: Iterable[pa.RecordBatch],
        screening_decision_batches: Iterable[pa.RecordBatch],
        diagnostic_batches: Iterable[pa.RecordBatch],
        lane_dictionary: Mapping[str, object],
        operator_dictionary: Mapping[str, object],
    ) -> None:
        """Transcode a verified old-v2 shard into v3 using Arrow batches.

        Old-v2 already stores non-screening families in the current typed
        schemas.  Those columns therefore remain columnar.  Only the sparse
        first-occurrence definition JSON is decoded; the high-volume
        occurrence stream is selected and written without dict-per-row work.
        """

        self._require_open()
        if (
            self._screening_definitions_sink is None
            or self._screening_occurrences_sink is None
            or self._screening_sink is not None
        ):
            raise RuntimeError("old-v2 Arrow transcoding requires screening_decisions_v3")
        self._lane_ids = _invert_trace_dictionary(lane_dictionary, "lane")
        self._operator_ids = _invert_trace_dictionary(operator_dictionary, "operator")
        for batch in route_batches:
            self._route_sink.append_batch(batch)
        for batch in critical_event_batches:
            self._event_sink.append_batch(batch)
        for batch in screening_check_batches:
            self._checks_sink.append_batch(batch)
        if self._screening_definition_store is None:
            self._screening_definition_store = _BoundedScreeningDefinitionStore(
                cache_entries=1,
                scratch_root=self._directory,
            )
        occurrence_fields = [field.name for field in V3_SCREENING_OCCURRENCES_SCHEMA]
        for batch in screening_decision_batches:
            if not batch.schema.equals(V2_SCREENING_DECISIONS_SCHEMA, check_metadata=False):
                raise ArtifactIntegrityError("old-v2 screening decision schema is invalid")
            definition_ids = batch.column(
                batch.schema.get_field_index("definition_id")
            ).to_pylist()
            definition_payloads = batch.column(
                batch.schema.get_field_index("definition_json")
            ).to_pylist()
            for definition_id, encoded in zip(
                definition_ids,
                definition_payloads,
                strict=True,
            ):
                if encoded is None:
                    continue
                if isinstance(definition_id, bool) or not isinstance(definition_id, int):
                    raise ArtifactIntegrityError("old-v2 definition ID is invalid")
                try:
                    definition = orjson.loads(encoded)
                except orjson.JSONDecodeError as error:
                    raise ArtifactIntegrityError(
                        "old-v2 screening definition JSON is invalid"
                    ) from error
                if not isinstance(definition, dict):
                    raise ArtifactIntegrityError("old-v2 screening definition is not an object")
                first = self._screening_definition_store.register(
                    definition_id,
                    definition,
                    allow_identical_existing=True,
                )
                if first:
                    self._append_buffered(
                        self._screening_definitions_sink,
                        {"definition_id": definition_id, **definition},
                    )
            self._screening_occurrences_sink.append_batch(batch.select(occurrence_fields))
        for batch in diagnostic_batches:
            self._diagnostic_sink.append_batch(batch)

    def flush(self) -> None:
        """Flush all currently buffered rows without closing the shard."""

        self._require_open()
        self._flush_pending_auxiliary_rows()
        for sink in self._sinks:
            sink.flush()
        self._active_sinks.clear()

    def finalize(
        self,
        *,
        raw_payload: Mapping[str, object],
        solution_payload: Mapping[str, object],
        trace_payload: Mapping[str, object],
        environment_payload: Mapping[str, object],
        failure_payload: Mapping[str, object] | None = None,
        anytime_checkpoints: Sequence[Mapping[str, object]] = (),
    ) -> dict[str, str]:
        """Close the shard, publish checksums, and write its complete manifest."""

        self._require_open()
        try:
            metadata = self._close_and_record_parquet()
            self.paths["raw"] = self._owner._write_json_artifact(
                self._directory, "raw", self.instance, self.seed, raw_payload
            )
            self.paths["solution"] = self._owner._write_json_artifact(
                self._directory, "solution", self.instance, self.seed, solution_payload
            )
            self.paths["environment"] = self._owner._write_json_artifact(
                self._directory,
                "environment",
                self.instance,
                self.seed,
                environment_payload,
            )
            if anytime_checkpoints:
                if self._owner.context.component != "benchmark":
                    raise ArtifactIntegrityError(
                        "anytime checkpoint artifacts are reserved for benchmark shards"
                    )
                if len(anytime_checkpoints) > 16:
                    raise ArtifactIntegrityError(
                        "one benchmark shard may contain at most 16 anytime checkpoints"
                    )
                checkpoint_path = self._directory / _canonical_filename(
                    self._owner.context,
                    "anytime_checkpoints",
                    self.instance,
                    self.seed,
                    "parquet",
                )
                checkpoint_count, checkpoint_schema = _write_parquet(
                    checkpoint_path,
                    tuple(_normalise_anytime_checkpoint(row) for row in anytime_checkpoints),
                    ANYTIME_CHECKPOINT_SCHEMA,
                    self._owner.config,
                )
                self._owner._record_file(
                    checkpoint_path,
                    artifact_type="anytime_checkpoints",
                    artifact_subtype="checkpoint_v1",
                    retention_class="critical",
                    storage_format="parquet",
                    compression=self._owner.config.compression,
                    row_count=checkpoint_count,
                    schema_fingerprint=checkpoint_schema,
                )
                self.paths["anytime_checkpoints"] = str(
                    checkpoint_path.relative_to(self._owner.run_dir)
                )
            trace_index = compact_trace_payload(
                trace_payload,
                route_dictionary_ref=self.paths["route_dictionary"],
                events_ref=self.paths["events"],
                screening_checks_ref=self.paths["screening_checks"],
                diagnostic_ref=self.paths["diagnostic"],
                lane_dictionary=self._lane_ids,
                operator_dictionary=self._operator_ids,
            )
            trace_index["event_identity"] = {
                "shard_ordinal": self.shard_ordinal,
                "local_field": "event_id",
            }
            trace_index["schema_fingerprints"] = metadata
            trace_index["screening_schema_version"] = self._owner.config.screening_schema_version
            if self._owner.config.screening_schema_version == SCREENING_DECISIONS_V3:
                trace_index["screening_definitions_ref"] = self.paths["screening_definitions"]
                trace_index["screening_occurrences_ref"] = self.paths["screening_occurrences"]
            else:
                trace_index["screening_decisions_ref"] = self.paths["screening_decisions"]
            self.paths["trace"] = self._owner._write_json_artifact(
                self._directory, "trace", self.instance, self.seed, trace_index
            )
            if failure_payload is not None:
                self.paths["failure"] = self._owner._write_json_artifact(
                    self._directory,
                    "failure",
                    self.instance,
                    self.seed,
                    failure_payload,
                    retention_class="critical",
                )
            self._owner._write_v2_shard_manifest(
                directory=self._directory,
                instance=self.instance,
                seed=self.seed,
                shard_ordinal=self.shard_ordinal,
                worker_identity=self.worker_identity,
                artifact_start=self._artifact_start,
                evidence_completeness="complete",
            )
            self.paths["shard_manifest"] = str(
                (
                    self._directory
                    / _canonical_filename(
                        self._owner.context,
                        "shard_manifest",
                        self.instance,
                        self.seed,
                    )
                ).relative_to(self._owner.run_dir)
            )
            self._state = "finalized"
            return dict(self.paths)
        except BaseException as error:
            self.abort(error)
            raise

    def abort(self, error: BaseException | str) -> None:
        """Close partial files and publish explicit failure evidence."""

        if self._state == "finalized":
            raise RuntimeError("cannot abort a finalized artifact v2 shard")
        if self._state == "aborted":
            return
        cleanup_errors: list[str] = []
        if self._state == "open":
            try:
                self._close_and_record_parquet()
            except BaseException as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        failure_path = self._directory / _canonical_filename(
            self._owner.context, "failure", self.instance, self.seed
        )
        manifest_path = self._directory / _canonical_filename(
            self._owner.context, "shard_manifest", self.instance, self.seed
        )
        for path, kind in (
            (failure_path, "failure"),
            (manifest_path, "manifest"),
            (manifest_path.with_suffix(".sha256"), "manifest_sidecar"),
        ):
            self._owner._archive_v2_control_fragment(
                path,
                instance=self.instance,
                seed=self.seed,
                kind=kind,
            )
        self._owner._register_v2_partial_fragments(self._directory)
        _json_write(
            failure_path,
            {
                "schema_version": ARTIFACT_STORAGE_V2,
                "run_label": self._owner.context.run_label,
                "instance": self.instance,
                "seed": self.seed,
                "status": "partial",
                "evidence_completeness": "partial",
                "failure_reason": str(error),
                "cleanup_failures": cleanup_errors,
            },
        )
        self._owner._record_file(
            failure_path,
            artifact_type="failure",
            retention_class="critical",
            storage_format="json_control",
            compression="none",
            enforce_budget=False,
        )
        self._owner._write_v2_shard_manifest(
            directory=self._directory,
            instance=self.instance,
            seed=self.seed,
            shard_ordinal=self.shard_ordinal,
            worker_identity=self.worker_identity,
            artifact_start=self._artifact_start,
            evidence_completeness="partial",
        )
        self._state = "aborted"

    @property
    def _sinks(self) -> tuple[_StreamingParquetSink, ...]:
        return tuple(
            sink
            for sink in (
                self._route_sink,
                self._event_sink,
                self._checks_sink,
                self._screening_sink,
                self._screening_definitions_sink,
                self._screening_occurrences_sink,
                self._diagnostic_sink,
            )
            if sink is not None
        )

    @property
    def max_buffered_groups_observed(self) -> int:
        """Maximum simultaneously non-empty row-group buffers for this shard."""

        return self._max_buffered_groups_observed

    @property
    def max_pending_screening_transaction_rows_observed(self) -> int:
        """Maximum bounded definition/occurrence transaction staged in Python."""

        return self._max_pending_screening_transaction_rows_observed

    @property
    def scratch_directory(self) -> Path:
        """Measured-volume directory for bounded live-stream scratch files."""

        return self._directory

    @property
    def unique_route_hot_entries(self) -> int:
        return self._route_digests.hot_entries

    @property
    def unique_route_hot_entry_limit(self) -> int:
        return self._route_digests.hot_entry_limit

    @property
    def unique_route_scratch_path(self) -> Path:
        return self._route_digests.scratch_path

    def unique_route_identity_count(self, semantics: str, axis: str) -> int:
        return self._route_digests.unique_identity_count(
            axis=axis,
            semantics=semantics,
        )

    def _append_buffered(self, sink: _StreamingParquetSink, row: Mapping[str, object]) -> None:
        if sink not in self._active_sinks and len(self._active_sinks) >= 2:
            victim = self._active_sinks.pop(0)
            victim.flush()
        sink.append(row)
        if sink.buffered_row_count:
            if sink not in self._active_sinks:
                self._active_sinks.append(sink)
        elif sink in self._active_sinks:
            self._active_sinks.remove(sink)
        self._max_buffered_groups_observed = max(
            self._max_buffered_groups_observed,
            len(self._active_sinks),
        )

    def _append_buffered_value_rows(
        self,
        sink: _StreamingParquetSink,
        rows: Sequence[Sequence[object]],
        *,
        trusted_width: bool = False,
    ) -> None:
        if not rows:
            return
        if sink not in self._active_sinks and len(self._active_sinks) >= 2:
            victim = self._active_sinks.pop(0)
            victim.flush()
        sink.append_value_rows(rows, trusted_width=trusted_width)
        if sink.buffered_row_count:
            if sink not in self._active_sinks:
                self._active_sinks.append(sink)
        elif sink in self._active_sinks:
            self._active_sinks.remove(sink)
        self._max_buffered_groups_observed = max(
            self._max_buffered_groups_observed,
            len(self._active_sinks),
        )

    def _flush_sink(self, sink: _StreamingParquetSink) -> None:
        sink.flush()
        if sink in self._active_sinks:
            self._active_sinks.remove(sink)

    def _screening_decision_row(
        self,
        event: Mapping[str, object],
        *,
        event_id: int,
        route_key: str,
        route_id: int | None,
        lane_id: int,
        operator_id: int,
    ) -> tuple[
        tuple[object, ...] | dict[str, object],
        _PendingScreeningDefinition | None,
    ]:
        event_get = event.get
        precomputed = (
            event_get("_precomputed_screening_definition")
            if self._screening_definitions_sink is not None
            else None
        )
        cache_key: tuple[object, ...]
        definition_key: tuple[object, ...]
        if isinstance(precomputed, _PrecomputedScreeningDefinition):
            definition_key = (lane_id, operator_id, route_id, *precomputed.tail)
            compact_key = (lane_id, operator_id, route_id, precomputed.cache_hash)
            cache_key = compact_key
            cached_entry = self._screening_definition_cache.get(compact_key)
            if cached_entry is not None and cached_entry[0] != precomputed.tail:
                cache_key = definition_key
                cached_entry = self._screening_definition_cache.get(definition_key)
        else:
            definition_key = _screening_definition_cache_key(
                event,
                lane_id=lane_id,
                operator_id=operator_id,
                route_id=route_id,
            )
            cache_key = definition_key
            cached_entry = self._screening_definition_cache.get(definition_key)
        cached = cached_entry[1:] if cached_entry is not None else None
        first_occurrence = False
        definition: dict[str, object] | None = None
        if cached is None:
            if route_id is not None and route_id not in self._route_digests:
                self._register_route(route_key, _route_sequence_from_key(route_key))
            definition = (
                _screening_definition_from_cache_key(definition_key)
                if isinstance(precomputed, _PrecomputedScreeningDefinition)
                else _normalise_screening_definition(
                    event,
                    lane_id=lane_id,
                    operator_id=operator_id,
                    route_id=route_id,
                )
            )
            definition_id, definition_json, definition_digest = _screening_definition_identity(
                definition
            )
            cached = (
                definition_id,
                definition_json,
                definition_digest,
            )
            self._screening_definition_cache[cache_key] = (
                precomputed.tail
                if isinstance(precomputed, _PrecomputedScreeningDefinition)
                else None,
                *cached,
            )
            if (
                len(self._screening_definition_cache)
                > SCREENING_DEFINITION_HOT_CACHE_ENTRIES
            ):
                self._screening_definition_cache.pop(next(iter(self._screening_definition_cache)))
            if self._screening_definition_store is None:
                self._screening_definition_store = _BoundedScreeningDefinitionStore(
                    cache_entries=1,
                    scratch_root=self._directory,
                )
            if self._screening_sink is not None:
                first_occurrence = self._screening_definition_store.register(
                    definition_id,
                    definition,
                    allow_identical_existing=True,
                    encoded=definition_json,
                )
        definition_id, definition_json, _ = cached
        definition_row = (
            _PendingScreeningDefinition(
                definition_id=definition_id,
                encoded=definition_json,
                payload=definition,
                row=(
                    definition_id,
                    *(definition[name] for name in V3_SCREENING_DEFINITIONS_SCHEMA.names[1:]),
                ),
            )
            if definition is not None
            and self._screening_definitions_sink is not None
            else None
        )
        started_at = event_get("started_at")
        if self._screening_sink is None:
            return (
                (
                    event_id,
                    definition_id,
                    started_at,
                    event_get("completed_at"),
                    event_get("iteration"),
                    event_get("decision_id"),
                ),
                definition_row,
            )
        return (
            {
                "event_id": event_id,
                "definition_id": definition_id,
                **(
                    {"definition_json": definition_json if first_occurrence else None}
                    if self._screening_sink is not None
                    else {}
                ),
                "started_at": started_at,
                "completed_at": event_get("completed_at"),
                "iteration": event_get("iteration"),
                "decision_id": event_get("decision_id"),
            },
            definition_row,
        )

    def _require_open(self) -> None:
        if self._state != "open":
            raise RuntimeError(f"artifact v2 shard session is not open: {self._state}")

    def _register_unique_route_evaluation_identity(
        self,
        event: Mapping[str, object],
    ) -> None:
        if (
            event.get("event_type") != "route_evaluation"
            or event.get("kind") != "exact_call"
        ):
            return
        axis = str(event.get("benchmark_axis", ""))
        route_key = str(event.get("route_key", ""))
        lane = str(event.get("lane", ""))
        axis_lane_prefix = f"{axis}:"
        if axis and lane.startswith(axis_lane_prefix):
            lane = lane[len(axis_lane_prefix) :]
        if event.get("exact_started") is True:
            self._route_digests.register_unique_identity(
                axis=axis,
                semantics="legacy_started",
                identity=(lane, route_key),
            )
        if event.get("exact_completed") is True:
            self._route_digests.register_unique_identity(
                axis=axis,
                semantics="completed_shared",
                identity=(route_key,),
            )
            self._route_digests.register_unique_identity(
                axis=axis,
                semantics="completed_lane",
                identity=("legacy" if lane == "initialization" else lane, route_key),
            )

    def _register_route(
        self,
        key: str,
        sequence: tuple[str, ...],
        *,
        route_id: int | None = None,
        validate_key: bool = True,
    ) -> None:
        if validate_key and key.startswith("route:") and _route_sequence_from_key(key) != sequence:
            raise ArtifactIntegrityError(
                f"route dictionary key does not match customer sequence: {key}"
            )
        resolved_route_id = self._resolve_route_id(key) if route_id is None else route_id
        route_digest = _payload_sha256(list(sequence))
        if not self._route_digests.register_route(resolved_route_id, route_digest):
            return
        self._pending_route_rows.append(
            (
                resolved_route_id,
                key,
                route_digest,
                list(sequence),
            )
        )
        if len(self._pending_route_rows) >= V2_PARQUET_ROW_GROUP_SIZE:
            self._flush_pending_route_rows()

    def _resolve_route_id(self, key: str) -> int:
        cached = self._resolved_route_ids.get(key)
        if cached is not None:
            return cached
        route_id = _stable_route_id(key)
        self._resolved_route_ids[key] = route_id
        if len(self._resolved_route_ids) > ROUTE_ID_RESOLUTION_CACHE_ENTRIES:
            self._resolved_route_ids.popitem(last=False)
        return route_id

    def _flush_pending_route_rows(self) -> None:
        rows = tuple(self._pending_route_rows)
        self._pending_route_rows.clear()
        self._append_buffered_value_rows(self._route_sink, rows)

    def _flush_pending_check_rows(self) -> None:
        rows = tuple(self._pending_check_rows)
        self._pending_check_rows.clear()
        self._append_buffered_value_rows(self._checks_sink, rows)

    def _flush_pending_auxiliary_rows(self) -> None:
        self._flush_pending_route_rows()
        self._flush_pending_check_rows()

    def _close_and_record_parquet(self) -> dict[str, str]:
        self._require_open()
        self._flush_pending_auxiliary_rows()
        descriptors = [
            (
                self._route_sink,
                "route_dictionary",
                "canonical_routes",
                "critical",
            ),
            (self._event_sink, "events", "critical", "critical"),
            (
                self._checks_sink,
                "events",
                "screening_checks",
                "critical",
            ),
            (
                self._diagnostic_sink,
                "diagnostic",
                "aggregated",
                "diagnostic",
            ),
        ]
        if self._screening_sink is not None:
            descriptors.append(
                (
                    self._screening_sink,
                    "events",
                    SCREENING_DECISIONS_V2,
                    "critical",
                )
            )
        else:
            if self._screening_definitions_sink is None or self._screening_occurrences_sink is None:
                raise RuntimeError("screening_decisions_v3 sinks are incomplete")
            descriptors.extend(
                (
                    (
                        self._screening_definitions_sink,
                        "events",
                        "screening_definitions_v3",
                        "critical",
                    ),
                    (
                        self._screening_occurrences_sink,
                        "events",
                        "screening_occurrences_v3",
                        "critical",
                    ),
                )
            )
        closed: list[tuple[_StreamingParquetSink, str, str, str, int, str]] = []
        errors: list[BaseException] = []
        for sink, artifact_type, subtype, retention in descriptors:
            try:
                count, fingerprint = sink.close()
                closed.append(
                    (
                        sink,
                        artifact_type,
                        subtype,
                        retention,
                        count,
                        fingerprint,
                    )
                )
            except BaseException as error:
                errors.append(error)
        if self._screening_definition_store is not None:
            try:
                self._screening_definition_store.close()
            except BaseException as error:
                errors.append(error)
        try:
            self._route_digests.close()
        except BaseException as error:
            errors.append(error)
        self._active_sinks.clear()
        self._state = "parquet_closed"
        fingerprints: dict[str, str] = {}
        fingerprint_keys = {
            "canonical_routes": "route_dictionary",
            "critical": "events",
            "screening_checks": "screening_checks",
            SCREENING_DECISIONS_V2: "screening_decisions",
            "screening_definitions_v3": "screening_definitions",
            "screening_occurrences_v3": "screening_occurrences",
            "aggregated": "diagnostic",
        }
        for sink, artifact_type, subtype, retention, count, fingerprint in closed:
            try:
                self._owner._record_file(
                    sink.path,
                    artifact_type=artifact_type,
                    artifact_subtype=subtype,
                    retention_class=retention,
                    storage_format="parquet",
                    compression=self._owner.config.compression,
                    row_count=count,
                    schema_fingerprint=fingerprint,
                )
                fingerprints[fingerprint_keys[subtype]] = fingerprint
            except BaseException as error:
                errors.append(error)
        if errors:
            raise BaseExceptionGroup("failed to close or register artifact v2 shard sinks", errors)
        return fingerprints


class ArtifactReader:
    """Read and verify both current Parquet bundles and legacy JSON bundles."""

    def __init__(self, run_dir: Path, *, verify: bool = True) -> None:
        self.run_dir = run_dir
        self._result = self._load(verify=verify)

    @property
    def result(self) -> ArtifactReadResult:
        return self._result

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self._result.manifest

    @property
    def storage_format(self) -> str:
        return self._result.storage_format

    @property
    def is_current(self) -> bool:
        return self.storage_format == CURRENT_STORAGE_FORMAT

    def _load(self, *, verify: bool) -> ArtifactReadResult:
        manifest_path = find_manifest(self.run_dir)
        current = manifest_path.parent.name == "control"
        if current:
            manifest = verify_manifest(self.run_dir) if verify else _json_read(manifest_path)
            storage_format = CURRENT_STORAGE_FORMAT
        else:
            manifest = (
                _verify_legacy_manifest(self.run_dir) if verify else _json_read(manifest_path)
            )
            storage_format = LEGACY_STORAGE_FORMAT
        return ArtifactReadResult(self.run_dir, manifest_path, storage_format, manifest)

    def read_json(self, relative_path: str | Path) -> dict[str, Any]:
        return _json_read(_safe_artifact_path(self.run_dir, Path(relative_path).as_posix()))

    def read_parquet(
        self,
        relative_path: str | Path,
        *,
        schema: pa.Schema | None = None,
    ) -> list[dict[str, Any]]:
        path = _safe_artifact_path(self.run_dir, Path(relative_path).as_posix())
        if not path.is_file() or path.suffix != ".parquet":
            raise ArtifactIntegrityError(f"Parquet artifact is missing: {path}")
        table = pq.read_table(path)
        if schema is not None and _schema_fingerprint(table.schema) != _schema_fingerprint(schema):
            raise ArtifactIntegrityError(f"Parquet schema mismatch: {path}")
        return [dict(row) for row in table.to_pylist()]

    def parquet_schema(self, relative_path: str | Path) -> pa.Schema:
        path = _safe_artifact_path(self.run_dir, Path(relative_path).as_posix())
        if not path.is_file() or path.suffix != ".parquet":
            raise ArtifactIntegrityError(f"Parquet artifact is missing: {path}")
        return pq.ParquetFile(path).schema_arrow

    def iter_parquet_batches(
        self,
        relative_path: str | Path,
        *,
        schema: pa.Schema | None = None,
        batch_size: int = V2_PARQUET_ROW_GROUP_SIZE,
        columns: Sequence[str] | None = None,
    ) -> Iterable[pa.RecordBatch]:
        """Yield verified Parquet batches without materialising the full table."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        path = _safe_artifact_path(self.run_dir, Path(relative_path).as_posix())
        if not path.is_file() or path.suffix != ".parquet":
            raise ArtifactIntegrityError(f"Parquet artifact is missing: {path}")
        parquet = pq.ParquetFile(path)
        if schema is not None and _schema_fingerprint(parquet.schema_arrow) != _schema_fingerprint(
            schema
        ):
            raise ArtifactIntegrityError(f"Parquet schema mismatch: {path}")
        selected_columns = tuple(columns) if columns is not None else None
        if selected_columns is not None:
            unknown = set(selected_columns) - set(parquet.schema_arrow.names)
            if unknown:
                raise ArtifactIntegrityError(
                    f"Parquet projection contains unknown columns: {sorted(unknown)}"
                )
        yield from parquet.iter_batches(
            batch_size=batch_size,
            columns=selected_columns,
        )

    def iter_parquet_rows(
        self,
        relative_path: str | Path,
        *,
        schema: pa.Schema | None = None,
        batch_size: int = V2_PARQUET_ROW_GROUP_SIZE,
        columns: Sequence[str] | None = None,
    ) -> Iterable[dict[str, object]]:
        """Yield row mappings from bounded Arrow batches without ``to_pylist``."""

        for batch in self.iter_parquet_batches(
            relative_path,
            schema=schema,
            batch_size=batch_size,
            columns=columns,
        ):
            names = batch.schema.names
            arrays = tuple(batch.column(index) for index in range(batch.num_columns))
            for row_index in range(batch.num_rows):
                yield {
                    name: array[row_index].as_py()
                    for name, array in zip(names, arrays, strict=True)
                }

    def iter_events(
        self,
        relative_path: str | Path,
        *,
        batch_size: int = V2_PARQUET_ROW_GROUP_SIZE,
        scratch_root: Path | None = None,
    ) -> Iterable[dict[str, object]]:
        """Stream writer-acceptable logical events across v1, v2, v3, and legacy.

        Ordinary event rows and physical screening streams are individually
        ordered by ``event_id`` and merged without materialising the event
        collection. Definition state is disk-backed and bounded; callers that
        bind active writes to a staging volume can provide ``scratch_root``.
        Omitting it preserves the historical system-temporary-directory
        behaviour.
        """

        path = _safe_artifact_path(self.run_dir, Path(relative_path).as_posix())
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ArtifactIntegrityError(f"legacy event row is not an object: {path}")
                    payload = item.get("payload")
                    logical = payload if isinstance(payload, dict) else item
                    yield dict(logical)
            return
        if path.suffix != ".parquet":
            raise ArtifactIntegrityError(f"unsupported event artifact: {path}")

        trace_path = path.with_name(path.name.replace("_events_", "_trace_", 1)).with_suffix(
            ".json"
        )
        if not trace_path.is_file():
            raise ArtifactIntegrityError(f"event trace index is missing: {trace_path}")
        trace_index = self.read_json(trace_path.relative_to(self.run_dir))
        lane_dictionary = {
            int(key): str(value)
            for key, value in dict(trace_index.get("lane_dictionary", {})).items()
        }
        operator_dictionary = {
            int(key): str(value)
            for key, value in dict(trace_index.get("operator_dictionary", {})).items()
        }
        route_dictionary_ref = trace_index.get("route_dictionary_ref")
        if not isinstance(route_dictionary_ref, str) or not route_dictionary_ref:
            raise ArtifactIntegrityError("event trace index lacks a route dictionary reference")
        route_by_id = _DiskBackedRouteKeyStore(scratch_root=scratch_root)
        try:
            for route_row in self.iter_parquet_rows(
                route_dictionary_ref,
                schema=ROUTE_DICTIONARY_SCHEMA,
                batch_size=batch_size,
            ):
                route_by_id.register(
                    route_row,
                    require_canonical_key=(
                        trace_index.get("screening_schema_version")
                        == SCREENING_DECISIONS_V3
                    ),
                )

            checks_path = path.with_name(
                path.name.replace("_events_", "_screening_checks_", 1)
            )
            ordinary_rows = self._iter_events_with_checks(
                path.relative_to(self.run_dir),
                checks_path=(
                    checks_path.relative_to(self.run_dir)
                    if checks_path.is_file()
                    else None
                ),
                batch_size=batch_size,
            )
            screening_rows = self._iter_physical_screening_events(
                path,
                batch_size=batch_size,
                scratch_root=scratch_root,
            )
            merged = heapq.merge(
                ordinary_rows,
                screening_rows,
                key=lambda row: _required_event_id(row),
            )
            previous_event_id = 0
            for physical_row in merged:
                event_id = _required_event_id(physical_row)
                if event_id <= previous_event_id:
                    raise ArtifactIntegrityError(
                        "event IDs must be unique and strictly increasing"
                    )
                previous_event_id = event_id
                yield _writer_event_mapping(
                    physical_row,
                    route_by_id=route_by_id,
                    lane_dictionary=lane_dictionary,
                    operator_dictionary=operator_dictionary,
                )
        finally:
            route_by_id.close()

    def _iter_events_with_checks(
        self,
        relative_path: str | Path,
        *,
        checks_path: str | Path | None,
        batch_size: int,
    ) -> Iterable[dict[str, object]]:
        rows = self.iter_parquet_rows(
            relative_path,
            schema=EVENTS_SCHEMA,
            batch_size=batch_size,
        )
        if checks_path is None:
            yield from rows
            return
        check_groups = iter(
            _group_screening_checks(
                self.iter_parquet_rows(
                    checks_path,
                    schema=SCREENING_CHECKS_SCHEMA,
                    batch_size=batch_size,
                )
            )
        )
        current_group = next(check_groups, None)
        for row in rows:
            event_id = _required_event_id(row)
            if current_group is not None and current_group[0] < event_id:
                raise ArtifactIntegrityError("screening check refers to an unknown event")
            if current_group is not None and current_group[0] == event_id:
                row["embedded_checks"] = current_group[1]
                current_group = next(check_groups, None)
            yield row
        if current_group is not None:
            raise ArtifactIntegrityError("screening check refers to an unknown event")

    def _iter_physical_screening_events(
        self,
        events_path: Path,
        *,
        batch_size: int,
        scratch_root: Path | None,
    ) -> Iterable[dict[str, object]]:
        compact_path = events_path.with_name(
            events_path.name.replace("_events_", "_screening_decisions_", 1)
        )
        definitions_path = events_path.with_name(
            events_path.name.replace("_events_", "_screening_definitions_", 1)
        )
        occurrences_path = events_path.with_name(
            events_path.name.replace("_events_", "_screening_occurrences_", 1)
        )
        v3_present = definitions_path.is_file() or occurrences_path.is_file()
        if v3_present and compact_path.is_file():
            raise ArtifactIntegrityError("ambiguous compact screening artifacts")
        if v3_present:
            if not definitions_path.is_file() or not occurrences_path.is_file():
                raise ArtifactIntegrityError(
                    "screening_decisions_v3 requires definitions and occurrences"
                )
            with _BoundedScreeningDefinitionStore(
                cache_entries=min(batch_size, V2_PARQUET_ROW_GROUP_SIZE),
                scratch_root=scratch_root,
            ) as definitions:
                for row in self.iter_parquet_rows(
                    definitions_path.relative_to(self.run_dir),
                    schema=V3_SCREENING_DEFINITIONS_SCHEMA,
                    batch_size=batch_size,
                ):
                    raw_definition_id = row.get("definition_id")
                    if isinstance(raw_definition_id, bool) or not isinstance(
                        raw_definition_id, int
                    ):
                        raise ArtifactIntegrityError("screening definition_id must be an integer")
                    definitions.register(
                        raw_definition_id,
                        _normalise_screening_definition(row),
                        allow_identical_existing=False,
                    )
                for row in self.iter_parquet_rows(
                    occurrences_path.relative_to(self.run_dir),
                    schema=V3_SCREENING_OCCURRENCES_SCHEMA,
                    batch_size=batch_size,
                ):
                    definition_id = _screening_definition_id_from_row(row)
                    yield _expand_screening_decision_row(
                        _combine_screening_definition_occurrence(
                            definitions.resolve(definition_id),
                            row,
                        )
                    )
            return
        if not compact_path.is_file():
            return
        compact_schema = self.parquet_schema(compact_path.relative_to(self.run_dir))
        if not any(
            compact_schema.equals(schema)
            for schema in (V2_SCREENING_DECISIONS_SCHEMA, V2_SCREENING_DECISIONS_SCHEMA_V1)
        ):
            raise ArtifactIntegrityError(f"unsupported compact screening schema: {compact_path}")
        compact_rows = self.iter_parquet_rows(
            compact_path.relative_to(self.run_dir),
            schema=compact_schema,
            batch_size=batch_size,
        )
        if compact_schema.equals(V2_SCREENING_DECISIONS_SCHEMA_V1):
            for row in compact_rows:
                yield _expand_screening_decision_row(row)
            return
        with _BoundedScreeningDefinitionStore(
            cache_entries=min(batch_size, V2_PARQUET_ROW_GROUP_SIZE),
            scratch_root=scratch_root,
        ) as definitions_v2:
            for row in compact_rows:
                definition_id = _screening_definition_id_from_row(row)
                definition_json = row.get("definition_json")
                if definition_json:
                    definition = _decode_screening_definition_json(definition_json)
                    definitions_v2.register(
                        definition_id,
                        definition,
                        allow_identical_existing=True,
                    )
                definition = definitions_v2.resolve(definition_id)
                yield _expand_screening_decision_row(
                    _combine_screening_definition_occurrence(definition, row)
                )

    def read_events(self, relative_path: str | Path) -> list[dict[str, Any]]:
        path = _safe_artifact_path(self.run_dir, Path(relative_path).as_posix())
        if path.suffix == ".parquet":
            parquet_rows = self.read_parquet(path.relative_to(self.run_dir), schema=EVENTS_SCHEMA)
            compact_path = path.with_name(path.name.replace("_events_", "_screening_decisions_", 1))
            definitions_path = path.with_name(
                path.name.replace("_events_", "_screening_definitions_", 1)
            )
            occurrences_path = path.with_name(
                path.name.replace("_events_", "_screening_occurrences_", 1)
            )
            if definitions_path.is_file() or occurrences_path.is_file():
                if not definitions_path.is_file() or not occurrences_path.is_file():
                    raise ArtifactIntegrityError(
                        "screening_decisions_v3 requires definitions and occurrences"
                    )
                definitions_v3: dict[int, dict[str, object]] = {}
                for row in self.read_parquet(
                    definitions_path.relative_to(self.run_dir),
                    schema=V3_SCREENING_DEFINITIONS_SCHEMA,
                ):
                    register_v3_screening_definition(row, definitions=definitions_v3)
                occurrence_rows = self.read_parquet(
                    occurrences_path.relative_to(self.run_dir),
                    schema=V3_SCREENING_OCCURRENCES_SCHEMA,
                )
                parquet_rows.extend(
                    expand_v3_screening_decision(row, definitions=definitions_v3)
                    for row in occurrence_rows
                )
                parquet_rows.sort(key=lambda row: int(row["event_id"]))
            elif compact_path.is_file():
                compact_schema = self.parquet_schema(compact_path.relative_to(self.run_dir))
                if not any(
                    compact_schema.equals(schema)
                    for schema in (
                        V2_SCREENING_DECISIONS_SCHEMA,
                        V2_SCREENING_DECISIONS_SCHEMA_V1,
                    )
                ):
                    raise ArtifactIntegrityError(
                        f"unsupported compact screening schema: {compact_path}"
                    )
                compact_rows = self.read_parquet(
                    compact_path.relative_to(self.run_dir),
                    schema=compact_schema,
                )
                definitions_v2: dict[int, dict[str, object]] = {}
                parquet_rows.extend(
                    expand_v2_screening_decision(row, definitions=definitions_v2)
                    for row in compact_rows
                )
                parquet_rows.sort(key=lambda row: int(row["event_id"]))
            return parquet_rows
        if path.suffix == ".jsonl":
            rows: list[dict[str, Any]] = []
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ArtifactIntegrityError(f"legacy event row is not an object: {path}")
                    payload = item.get("payload")
                    rows.append(payload if isinstance(payload, dict) else item)
            return rows
        raise ArtifactIntegrityError(f"unsupported event artifact: {path}")

    def trace_index(self, relative_path: str | Path) -> dict[str, Any]:
        payload = self.read_json(relative_path)
        if self.is_current and payload.get("trace_storage_version") != "stage03-trace-index-v2":
            raise ArtifactIntegrityError("current trace is not a v2 trace index")
        return payload

    def artifact_reference(self, relative_path: str | Path) -> Mapping[str, Any]:
        relative = Path(relative_path).as_posix()
        for item in self.manifest.get("artifacts", []):
            if isinstance(item, Mapping) and item.get("relative_path") == relative:
                return item
        raise ArtifactIntegrityError(f"artifact is not listed in manifest: {relative}")

    def reconstruct_trace(self, relative_path: str | Path) -> dict[str, Any]:
        """Rebuild a Stage03Trace-compatible payload from current columns."""

        index = self.trace_index(relative_path)
        if not self.is_current:
            return index
        expected_fingerprints = index.get("schema_fingerprints", {})
        required_fingerprint_names = {
            "route_dictionary",
            "events",
            "screening_checks",
            "diagnostic",
        }
        compact_screening_ref = index.get("screening_decisions_ref")
        screening_definitions_ref = index.get("screening_definitions_ref")
        screening_occurrences_ref = index.get("screening_occurrences_ref")
        if compact_screening_ref is not None:
            required_fingerprint_names.add("screening_decisions")
        if screening_definitions_ref is not None or screening_occurrences_ref is not None:
            if screening_definitions_ref is None or screening_occurrences_ref is None:
                raise ArtifactIntegrityError(
                    "trace v3 screening references require definitions and occurrences"
                )
            required_fingerprint_names.update({"screening_definitions", "screening_occurrences"})
        if not isinstance(
            expected_fingerprints, Mapping
        ) or not required_fingerprint_names.issubset(expected_fingerprints):
            raise ArtifactIntegrityError("trace schema_fingerprints must be an object")
        for name, reference in (
            ("route_dictionary", index["route_dictionary_ref"]),
            ("events", index["events_ref"]),
            ("screening_checks", index["screening_checks_ref"]),
            ("diagnostic", index["diagnostic_ref"]),
            ("screening_decisions", compact_screening_ref),
            ("screening_definitions", screening_definitions_ref),
            ("screening_occurrences", screening_occurrences_ref),
        ):
            if reference is None:
                continue
            actual_fingerprint = _schema_fingerprint(
                pq.ParquetFile(_safe_artifact_path(self.run_dir, str(reference))).schema_arrow
            )
            if str(expected_fingerprints.get(name, "")) != actual_fingerprint:
                raise ArtifactIntegrityError(f"trace schema fingerprint is invalid for {name}")
        route_rows = self.read_parquet(
            str(index["route_dictionary_ref"]), schema=ROUTE_DICTIONARY_SCHEMA
        )
        route_by_id = {int(row["route_id"]): str(row["canonical_route_key"]) for row in route_rows}
        route_dictionary = {
            str(row["canonical_route_key"]): tuple(str(value) for value in row["customer_sequence"])
            for row in route_rows
        }
        lane_dictionary = {
            int(key): str(value) for key, value in dict(index.get("lane_dictionary", {})).items()
        }
        operator_dictionary = {
            int(key): str(value)
            for key, value in dict(index.get("operator_dictionary", {})).items()
        }
        checks = self.read_parquet(
            str(index["screening_checks_ref"]), schema=SCREENING_CHECKS_SCHEMA
        )
        checks_by_event: dict[int, list[dict[str, object]]] = {}
        for row in checks:
            checks_by_event.setdefault(int(row["decision_event_id"]), []).append(
                {
                    "check": str(row["check"]),
                    "status": str(row.get("status") or ""),
                    "value": _check_value(row),
                    "reason": str(row.get("reason") or ""),
                }
            )
        route_evaluations: list[dict[str, object]] = []
        trace_events: list[dict[str, object]] = []
        screening_decisions: list[dict[str, object]] = []
        incremental_propagations: list[dict[str, object]] = []
        neighborhood_events: list[dict[str, object]] = []
        for row in sorted(
            self.read_events(str(index["events_ref"])),
            key=lambda item: int(item.get("event_id", 0)),
        ):
            record_type = str(row.get("record_type") or row.get("event_type") or "event")
            extras = _decode_extras(row.get("extras_json"))
            payload = _event_payload(
                row,
                route_by_id,
                extras,
                lane_dictionary=lane_dictionary,
                operator_dictionary=operator_dictionary,
            )
            payload.pop("embedded_checks", None)
            if record_type == "route_evaluation":
                route_evaluations.append(
                    {
                        **payload,
                        "evaluation_id": int(row.get("evaluation_id") or 0),
                        "route_key": _route_key(route_by_id, row.get("route_id")),
                        "completed_at": row.get("completed_at"),
                        "duration_seconds": float(row.get("duration_seconds") or 0.0),
                        "exact_started": bool(row.get("exact_started")),
                        "exact_completed": bool(row.get("exact_completed")),
                        "feasible": row.get("feasible"),
                        "failure_reason": str(row.get("failure_reason") or ""),
                        "kind": str(row.get("kind") or ""),
                        "route_change_status": str(row.get("route_change_status") or "unknown"),
                    }
                )
            elif record_type == "screening_decision":
                embedded_checks = row.get("embedded_checks")
                screening_decisions.append(
                    {
                        **payload,
                        "decision_id": int(row.get("decision_id") or 0),
                        "route_key": _route_key(route_by_id, row.get("route_id")),
                        "checks": (
                            embedded_checks
                            if isinstance(embedded_checks, list)
                            else checks_by_event.get(int(row["event_id"]), [])
                        ),
                        "demand": float(extras.get("demand", 0.0)),
                        "min_time_window_slack": float(extras.get("min_time_window_slack", 0.0)),
                        "distance_lower_bound": float(extras.get("distance_lower_bound", 0.0)),
                        "distance_increment_lower_bound": extras.get(
                            "distance_increment_lower_bound"
                        ),
                        "single_segment_reachable": bool(
                            extras.get("single_segment_reachable", False)
                        ),
                        "structural_energy_lower_bound": float(
                            extras.get("structural_energy_lower_bound", 0.0)
                        ),
                        "negative_cache_hit": bool(extras.get("negative_cache_hit", False)),
                        "exact_call_blocked": bool(extras.get("exact_call_blocked", False)),
                        "started_at": float(row.get("started_at") or 0.0),
                        "completed_at": float(row.get("completed_at") or 0.0),
                        "duration_seconds": float(row.get("duration_seconds") or 0.0),
                    }
                )
            elif record_type == "incremental_propagation":
                incremental_propagations.append(payload)
            elif record_type == "neighborhood_event":
                neighborhood_events.append(payload)
            else:
                trace_events.append(payload)
        rebuilt = dict(index)
        rebuilt["route_dictionary"] = route_dictionary
        rebuilt["route_evaluations"] = route_evaluations
        rebuilt["screening_decisions"] = screening_decisions
        rebuilt["incremental_propagations"] = incremental_propagations
        rebuilt["events"] = trace_events
        rebuilt["neighborhood_events"] = neighborhood_events
        return rebuilt


def _required_event_id(row: Mapping[str, object]) -> int:
    value = row.get("event_id")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactIntegrityError("event_id must be a positive integer")
    return value


def _group_screening_checks(
    rows: Iterable[Mapping[str, object]],
) -> Iterable[tuple[int, list[dict[str, object]]]]:
    current_event_id: int | None = None
    current_checks: list[dict[str, object]] = []
    expected_index = 0
    for row in rows:
        raw_event_id = row.get("decision_event_id")
        if isinstance(raw_event_id, bool) or not isinstance(raw_event_id, int):
            raise ArtifactIntegrityError("screening check event ID must be an integer")
        if current_event_id is None or raw_event_id != current_event_id:
            if current_event_id is not None:
                if raw_event_id < current_event_id:
                    raise ArtifactIntegrityError("screening checks are not event ordered")
                yield current_event_id, current_checks
            current_event_id = raw_event_id
            current_checks = []
            expected_index = 0
        raw_index = row.get("check_index")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ArtifactIntegrityError("screening check_index must be an integer")
        if raw_index != expected_index:
            raise ArtifactIntegrityError("screening check indexes must be contiguous")
        if raw_index >= 8:
            raise ArtifactIntegrityError(
                "one screening decision exceeds the fixed eight-check domain"
            )
        expected_index += 1
        current_checks.append(
            {
                "check": str(row.get("check") or ""),
                "status": str(row.get("status") or ""),
                "value": _check_value(row),
                "reason": str(row.get("reason") or ""),
            }
        )
    if current_event_id is not None:
        yield current_event_id, current_checks


def _writer_event_mapping(
    row: Mapping[str, object],
    *,
    route_by_id: Mapping[int, str],
    lane_dictionary: Mapping[int, str],
    operator_dictionary: Mapping[int, str],
) -> dict[str, object]:
    """Convert a physical event row into a mapping accepted by the writer."""

    output = _event_payload(
        row,
        route_by_id,
        _decode_extras(row.get("extras_json")),
        lane_dictionary=lane_dictionary,
        operator_dictionary=operator_dictionary,
    )
    output["event_id"] = _required_event_id(row)
    output["record_type"] = str(row.get("record_type") or row.get("event_type") or "event")
    output.pop("embedded_checks", None)
    output.pop("definition_id", None)
    output.pop("definition_json", None)
    for field_name in (
        "route_id",
        "route_ids",
        "current_route_ids",
        "candidate_route_ids",
        "base_route_id",
        "candidate_route_id",
    ):
        value = row.get(field_name)
        if value is not None:
            output[field_name] = value
    raw_checks = row.get("embedded_checks")
    if isinstance(raw_checks, list):
        output["checks"] = [dict(check) for check in raw_checks if isinstance(check, Mapping)]
    return output


def _check_value(row: Mapping[str, Any]) -> object:
    if row.get("value_bool") is not None:
        return row["value_bool"]
    if row.get("value_float") is not None:
        return row["value_float"]
    return row.get("value_text")


def _decode_extras(value: object) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ArtifactIntegrityError("event extras_json is invalid") from error
    if not isinstance(decoded, dict):
        raise ArtifactIntegrityError("event extras_json must contain an object")
    return decoded


def _route_key(route_by_id: Mapping[int, str], value: object) -> str:
    try:
        return route_by_id[int(str(value))]
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError(f"event refers to an unknown route id: {value}") from error


def _event_payload(
    row: Mapping[str, Any],
    route_by_id: Mapping[int, str],
    extras: Mapping[str, Any],
    *,
    lane_dictionary: Mapping[int, str],
    operator_dictionary: Mapping[int, str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "event_id",
            "record_type",
            "extras_json",
            "lane_id",
            "operator_id",
            "route_id",
            "route_ids",
            "current_route_ids",
            "candidate_route_ids",
            "base_route_id",
            "candidate_route_id",
        }
        and value is not None
        and value != ""
    }
    payload.update(extras)
    if row.get("lane_id") is not None:
        payload["lane"] = _dictionary_value(lane_dictionary, row["lane_id"], "lane")
    if row.get("operator_id") is not None:
        payload["operator"] = _dictionary_value(operator_dictionary, row["operator_id"], "operator")
    if row.get("route_id") is not None:
        payload["route_key"] = _route_key(route_by_id, row["route_id"])
    if row.get("route_ids"):
        payload["route_keys"] = [_route_key(route_by_id, value) for value in row["route_ids"]]
    if row.get("current_route_ids"):
        payload["current_route_keys"] = [
            _route_key(route_by_id, value) for value in row["current_route_ids"]
        ]
    if row.get("candidate_route_ids"):
        payload["candidate_route_keys"] = [
            _route_key(route_by_id, value) for value in row["candidate_route_ids"]
        ]
    for field_name, route_field in (
        ("base_route_key", "base_route_id"),
        ("candidate_route_key", "candidate_route_id"),
    ):
        if row.get(route_field) is not None:
            payload[field_name] = _route_key(route_by_id, row[route_field])
    return payload


def _dictionary_value(dictionary: Mapping[int, str], value: object, field_name: str) -> str:
    try:
        return dictionary[int(str(value))]
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"event refers to an unknown {field_name} dictionary id: {value}"
        ) from error


def storage_format_for_run(run_dir: Path) -> str:
    """Return ``current`` or ``legacy`` without rewriting either format."""

    try:
        manifest = _json_read(find_manifest(run_dir))
    except (FileNotFoundError, ArtifactIntegrityError):
        return "legacy"
    return (
        "current"
        if manifest.get("storage_policy_version") in SUPPORTED_STORAGE_POLICIES
        else "legacy"
    )


def aggregate_diagnostic_events(
    events: Iterable[Mapping[str, object]], *, run_label: str, instance: str, seed: int
) -> list[dict[str, object]]:
    """Aggregate ordinary events while preserving critical event rows."""

    counts: Counter[tuple[str, str, str, str]] = Counter()
    for event in events:
        event_type = str(event.get("event_type", event.get("record_type", "event")))
        if event_type in {
            "execution_error",
            "deadline_boundary",
            "cache_event",
            "screening_decision",
            "route_evaluation",
            "incremental_propagation",
        }:
            continue
        key = (
            str(event.get("lane", "")),
            str(event.get("operator", "")),
            str(event.get("reason", event.get("status", ""))),
            event_type,
        )
        counts[key] += 1
    return [
        {
            "run_label": run_label,
            "instance": instance,
            "seed": seed,
            "lane": lane,
            "iteration": None,
            "operator": operator,
            "reason": reason,
            "metric": event_type,
            "count": count,
            "sum_value": None,
            "min_value": None,
            "max_value": None,
        }
        for (lane, operator, reason, event_type), count in sorted(counts.items())
    ]
