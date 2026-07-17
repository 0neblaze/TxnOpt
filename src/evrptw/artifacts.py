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
import json
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

ARTIFACT_STORAGE_SCHEMA_VERSION = "artifact-storage-v1"
ARTIFACT_STORAGE_V2 = "artifact-storage-v2"
SUPPORTED_STORAGE_POLICIES = frozenset(
    {ARTIFACT_STORAGE_SCHEMA_VERSION, ARTIFACT_STORAGE_V2}
)
V2_PARQUET_ROW_GROUP_SIZE = 65_536
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
    per_instance_seed_max_bytes: int = DEFAULT_PER_INSTANCE_SEED_MAX_BYTES
    per_run_max_bytes: int = DEFAULT_PER_RUN_MAX_BYTES

    def __post_init__(self) -> None:
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
        raise ValueError(
            "new experiment runs must use a canonical attemptNN/rerunNN run label"
        )
    if config is None or not config.enabled:
        raise ValueError(
            "new experiment runs require an enabled [artifact_storage] configuration"
        )
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


def _as_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
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
        value = _as_float(event.get(key))
        if value is not None:
            return value
    return None


def _event_type(event: Mapping[str, object]) -> str:
    value = event.get("event_type", event.get("record_type", "event"))
    return str(value)


def _normalise_event(
    event: Mapping[str, object],
    *,
    event_id: int,
    route_ids: Mapping[str, int],
    lane_ids: Mapping[str, int],
    operator_ids: Mapping[str, int],
) -> dict[str, object]:
    known = {
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
    route_id = _route_id(event.get("route_id", event.get("route_key")), route_ids)
    base_route_id = _route_id(
        event.get("base_route_id", event.get("base_route_key")), route_ids
    )
    candidate_route_id = _route_id(
        event.get("candidate_route_id", event.get("candidate_route_key")), route_ids
    )
    current_route_ids = _route_ids(
        event.get("current_route_ids", event.get("current_route_keys")), route_ids
    )
    candidate_route_ids = _route_ids(
        event.get("candidate_route_ids", event.get("candidate_route_keys")), route_ids
    )
    extras = {
        str(key): value
        for key, value in event.items()
        if key not in known
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
    if event.get("record_class"):
        extras["record_class"] = event["record_class"]
    return {
        "event_id": event_id,
        "record_type": str(event.get("record_type", _event_type(event))),
        "event_type": _event_type(event),
        "timestamp_seconds": _event_timestamp(event),
        "started_at": _as_float(event.get("started_at")),
        "completed_at": _as_float(event.get("completed_at")),
        "duration_seconds": _as_float(event.get("duration_seconds")),
        "lane_id": lane_ids[str(event.get("lane", ""))],
        "iteration": _as_int(event.get("iteration")),
        "operator_id": operator_ids[str(event.get("operator", ""))],
        "route_id": route_id,
        "route_ids": _route_ids(event.get("route_ids", event.get("route_keys")), route_ids),
        "current_route_ids": current_route_ids,
        "candidate_route_ids": candidate_route_ids,
        "base_route_id": base_route_id,
        "candidate_route_id": candidate_route_id,
        "status": str(event.get("status", "")),
        "kind": str(event.get("kind", "")),
        "operation": str(event.get("operation", "")),
        "reason": str(event.get("reason", "")),
        "failure_reason": str(event.get("failure_reason", "")),
        "feasible": _as_bool(event.get("feasible")),
        "exact_started": _as_bool(event.get("exact_started")),
        "exact_completed": _as_bool(event.get("exact_completed")),
        "candidate_feasible": _as_bool(event.get("candidate_feasible")),
        "accepted": _as_bool(event.get("accepted")),
        "global_best": _as_bool(event.get("global_best")),
        "current_vehicle_count": _as_int(event.get("current_vehicle_count")),
        "candidate_vehicle_count": _as_int(event.get("candidate_vehicle_count")),
        "candidate_vehicle_delta": _as_int(event.get("candidate_vehicle_delta")),
        "cache_key_digest": str(event.get("cache_key_digest", "")),
        "evaluation_id": _as_int(event.get("evaluation_id")),
        "decision_id": _as_int(event.get("decision_id")),
        "route_change_status": str(event.get("route_change_status", "")),
        "propagation_status": str(
            event.get("propagation_status", event.get("status", ""))
        ),
        "extras_json": json.dumps(extras, sort_keys=True, separators=(",", ":"))
        if extras
        else "",
    }


def _validate_event_routes(
    event: Mapping[str, object], route_ids: Mapping[str, int]
) -> None:
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
    table = pa.Table.from_pylist([dict(row) for row in rows], schema=schema)
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


class _StreamingParquetSink:
    """Bounded row-group writer used by artifact-storage-v2 shards."""

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
        self.rows: list[dict[str, object]] = []
        self.row_count = 0
        self._writer = pq.ParquetWriter(
            path,
            schema,
            compression=config.compression,
            compression_level=config.compression_level,
            use_dictionary=True,
            write_statistics=True,
        )

    def append(self, row: Mapping[str, object]) -> None:
        self.rows.append(dict(row))
        if len(self.rows) >= V2_PARQUET_ROW_GROUP_SIZE:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        self._writer.write_table(table, row_group_size=V2_PARQUET_ROW_GROUP_SIZE)
        self.row_count += table.num_rows
        self.rows.clear()

    def close(self) -> tuple[int, str]:
        self.flush()
        self._writer.close()
        return self.row_count, _schema_fingerprint(self.schema)


def _iter_coalesced_cache_lookup_events(
    events: Iterable[Mapping[str, object]],
) -> Iterable[dict[str, object]]:
    """Streaming equivalent of :func:`_coalesce_cache_lookup_events`."""

    pending: dict[str, object] | None = None
    for raw_event in events:
        current = dict(raw_event)
        if pending is None:
            pending = current
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
            yield pending
            pending = current
    if pending is not None:
        yield pending


def _stable_dictionary_id(value: str) -> int:
    """Return a deterministic positive int32 ID independent of completion order."""

    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


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
            f"{directory}/"
            f"{_canonical_filename(context, artifact_type, instance, seed, extension)}"
        )
        for artifact_type, extension in (
            ("raw", "json"),
            ("solution", "json"),
            ("trace", "json"),
            ("events", "parquet"),
            ("route_dictionary", "parquet"),
            ("screening_checks", "parquet"),
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

        route_dictionary = {
            canonical_route_key(tuple(route)): tuple(route) for route in routes
        }
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
                str(error)
                if error is not None
                else str(getattr(result, "failure_reason", ""))
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
            "objective_key": list(
                getattr(getattr(result, "objective", None), "key", ())
            ),
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
        metadata_path = control / _canonical_filename(
            self.context, "run_metadata"
        )
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
                raise ArtifactIntegrityError(
                    f"v2 shard manifest sidecar mismatch: {manifest_path}"
                )
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
                        raise ArtifactIntegrityError(
                            f"v2 shard artifact checksum mismatch: {path}"
                        )
                    self._record_file(
                        path,
                        artifact_type=str(item["artifact_type"]),
                        artifact_subtype=str(item.get("artifact_subtype", "")),
                        retention_class=str(item["retention_class"]),
                        storage_format=str(item["storage_format"]),
                        compression=str(item["compression"]),
                        row_count=(
                            int(item["row_count"])
                            if item.get("row_count") is not None
                            else None
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
                (directory / _canonical_filename(
                    self.context, "route_dictionary", instance, seed, "parquet"
                )).relative_to(self.run_dir)
            )
            events_relative = str(
                (directory / _canonical_filename(
                    self.context, "events", instance, seed, "parquet"
                )).relative_to(self.run_dir)
            )
            checks_relative = str(
                (directory / _canonical_filename(
                    self.context, "screening_checks", instance, seed, "parquet"
                )).relative_to(self.run_dir)
            )
            diagnostic_relative = str(
                (directory / _canonical_filename(
                    self.context, "diagnostic", instance, seed, "parquet"
                )).relative_to(self.run_dir)
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
            failure_path = directory / _canonical_filename(
                self.context, "failure", instance, seed
            )
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
        checks_sink = _StreamingParquetSink(
            checks_path, SCREENING_CHECKS_SCHEMA, self.config
        )
        diagnostic_sink = _StreamingParquetSink(
            diagnostic_path, DIAGNOSTIC_SCHEMA, self.config
        )
        try:
            paths["raw"] = self._write_json_artifact(
                directory, "raw", instance, seed, raw_payload
            )
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
                operator_ids.setdefault(
                    operator, _stable_dictionary_id(f"operator:{operator}")
                )
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
                diagnostic_sink.append(
                    self._normalise_diagnostic(row, instance, seed)
                )
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
                    directory
                    / _canonical_filename(
                        self.context, "shard_manifest", instance, seed
                    )
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
            failure_path = directory / _canonical_filename(
                self.context, "failure", instance, seed
            )
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
            _canonical_filename(self.context, "manifest", extension="json")[:-5]
            + ".sha256"
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
        self._record_file(
            path,
            artifact_type=artifact_type,
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
        self._artifacts = [
            item for item in self._artifacts if item.relative_path != relative
        ]
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

    rows: list[dict[str, object]] = []
    route_evaluations = getattr(trace, "route_evaluations", ())
    for record in route_evaluations:
        payload = dict(record.__dict__) if hasattr(record, "__dict__") else {}
        if not payload:
            from dataclasses import asdict

            payload = asdict(record)
        payload["record_type"] = "route_evaluation"
        rows.append(payload)
    for event in getattr(trace, "events", ()):
        rows.append({**dict(event), "record_type": str(event.get("event_type", "event"))})
    from dataclasses import asdict

    for decision in getattr(trace, "screening_decisions", ()):
        payload = asdict(decision)
        payload["record_type"] = "screening_decision"
        payload["event_type"] = "screening_decision"
        rows.append(payload)
    for propagation in getattr(trace, "incremental_propagations", ()):
        rows.append({**dict(propagation), "record_type": "incremental_propagation"})
    rows.extend(
        {**dict(event), "record_type": "neighborhood_event"}
        for event in neighborhood_events
    )
    ordered = sorted(
        enumerate(rows),
        key=lambda item: (
            _event_timestamp(item[1]) is None,
            _event_timestamp(item[1]) or 0.0,
            item[0],
        ),
    )
    return [item[1] for item in ordered]


def find_manifest(run_dir: Path) -> Path:
    """Find a v2 control manifest, falling back to the legacy root manifest."""

    candidates = sorted((run_dir / "control").glob("*_manifest.json"))
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
    if policy.to_dict() != dict(policy_payload):
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
    for path in run_dir.rglob("*"):
        if not path.is_file() or path in {manifest_path, sidecar}:
            continue
        relative = path.relative_to(run_dir).as_posix()
        if relative == "review" or relative.startswith("review/"):
            continue
        if relative not in listed_paths:
            raise ArtifactIntegrityError(f"unlisted artifact is present: {relative}")
    has_failure = any(
        isinstance(item, Mapping) and item.get("artifact_type") == "failure"
        for item in artifacts
    )
    expected_failure_status = "present" if has_failure else "not_applicable"
    if artifact_status.get("failure") != expected_failure_status:
        raise ArtifactIntegrityError(
            "failure artifact status does not match manifest contents"
        )
    return manifest


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
                _verify_legacy_manifest(self.run_dir)
                if verify
                else _json_read(manifest_path)
            )
            storage_format = LEGACY_STORAGE_FORMAT
        return ArtifactReadResult(self.run_dir, manifest_path, storage_format, manifest)

    def read_json(self, relative_path: str | Path) -> dict[str, Any]:
        return _json_read(
            _safe_artifact_path(self.run_dir, Path(relative_path).as_posix())
        )

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

    def iter_parquet_batches(
        self,
        relative_path: str | Path,
        *,
        schema: pa.Schema | None = None,
        batch_size: int = V2_PARQUET_ROW_GROUP_SIZE,
    ) -> Iterable[pa.RecordBatch]:
        """Yield verified Parquet batches without materialising the full table."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        path = _safe_artifact_path(self.run_dir, Path(relative_path).as_posix())
        if not path.is_file() or path.suffix != ".parquet":
            raise ArtifactIntegrityError(f"Parquet artifact is missing: {path}")
        parquet = pq.ParquetFile(path)
        if schema is not None and _schema_fingerprint(
            parquet.schema_arrow
        ) != _schema_fingerprint(schema):
            raise ArtifactIntegrityError(f"Parquet schema mismatch: {path}")
        yield from parquet.iter_batches(batch_size=batch_size)

    def read_events(self, relative_path: str | Path) -> list[dict[str, Any]]:
        path = _safe_artifact_path(self.run_dir, Path(relative_path).as_posix())
        if path.suffix == ".parquet":
            return self.read_parquet(path.relative_to(self.run_dir), schema=EVENTS_SCHEMA)
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
        if (
            not isinstance(expected_fingerprints, Mapping)
            or not required_fingerprint_names.issubset(expected_fingerprints)
        ):
            raise ArtifactIntegrityError("trace schema_fingerprints must be an object")
        for name, reference in (
            ("route_dictionary", index["route_dictionary_ref"]),
            ("events", index["events_ref"]),
            ("screening_checks", index["screening_checks_ref"]),
            ("diagnostic", index["diagnostic_ref"]),
        ):
            actual_fingerprint = _schema_fingerprint(
                pq.ParquetFile(
                    _safe_artifact_path(self.run_dir, str(reference))
                ).schema_arrow
            )
            if str(expected_fingerprints.get(name, "")) != actual_fingerprint:
                raise ArtifactIntegrityError(
                    f"trace schema fingerprint is invalid for {name}"
                )
        route_rows = self.read_parquet(
            str(index["route_dictionary_ref"]), schema=ROUTE_DICTIONARY_SCHEMA
        )
        route_by_id = {
            int(row["route_id"]): str(row["canonical_route_key"]) for row in route_rows
        }
        route_dictionary = {
            str(row["canonical_route_key"]): tuple(
                str(value) for value in row["customer_sequence"]
            )
            for row in route_rows
        }
        lane_dictionary = {
            int(key): str(value)
            for key, value in dict(index.get("lane_dictionary", {})).items()
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
                screening_decisions.append(
                    {
                        **payload,
                        "decision_id": int(row.get("decision_id") or 0),
                        "route_key": _route_key(route_by_id, row.get("route_id")),
                        "checks": checks_by_event.get(int(row["event_id"]), []),
                        "demand": float(extras.get("demand", 0.0)),
                        "min_time_window_slack": float(
                            extras.get("min_time_window_slack", 0.0)
                        ),
                        "distance_lower_bound": float(
                            extras.get("distance_lower_bound", 0.0)
                        ),
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
        payload["lane"] = _dictionary_value(
            lane_dictionary, row["lane_id"], "lane"
        )
    if row.get("operator_id") is not None:
        payload["operator"] = _dictionary_value(
            operator_dictionary, row["operator_id"], "operator"
        )
    if row.get("route_id") is not None:
        payload["route_key"] = _route_key(route_by_id, row["route_id"])
    if row.get("route_ids"):
        payload["route_keys"] = [
            _route_key(route_by_id, value) for value in row["route_ids"]
        ]
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


def _dictionary_value(
    dictionary: Mapping[int, str], value: object, field_name: str
) -> str:
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
