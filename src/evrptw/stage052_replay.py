"""Deterministic Stage 5.2 shard replay adapters.

``python_reference`` preserves a readable differential oracle. ``native_arrow``
passes Arrow columns to the native state machine in bounded RecordBatch units;
it never falls back to Python after selection.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.compute as pc

from evrptw import _core
from evrptw.artifacts import (
    EVENTS_SCHEMA,
    ROUTE_DICTIONARY_SCHEMA,
    V3_SCREENING_DEFINITIONS_SCHEMA,
    V3_SCREENING_OCCURRENCES_SCHEMA,
    ArtifactIntegrityError,
    ArtifactReader,
)

type ReplayBackend = Literal["python_reference", "native_arrow"]
type AxisCounts = tuple[tuple[str, int, int], ...]
type EventTypeCounts = tuple[tuple[str, int], ...]
type PersistenceLedger = Mapping[str, tuple[tuple[int, str], ...]]

MAX_REPLAY_SCREENING_DEFINITIONS = 2_000_000
MAX_REPLAY_ROUTE_IDENTITIES = 2_000_000

_REQUIRED_COLUMNS = frozenset(
    {
        "event_id",
        "benchmark_axis",
        "lane",
        "event_type",
        "timestamp_seconds",
        "started_at",
        "completed_at",
        "iteration",
        "evaluation_id",
        "cache_key_digest",
        "operation",
        "lookup_result",
        "status",
        "reason",
        "route_key",
        "decision_id",
        "kind",
        "operator",
        "exact_started",
        "exact_completed",
        "feasible",
        "accepted",
        "global_best",
        "candidate_vehicle_delta",
        "native_fallback",
        "failure_reason",
    }
)
REPLAY_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.int64(), nullable=False),
        pa.field("benchmark_axis", pa.string()),
        pa.field("lane", pa.string()),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("timestamp_seconds", pa.float64()),
        pa.field("started_at", pa.float64()),
        pa.field("completed_at", pa.float64()),
        pa.field("iteration", pa.int64()),
        pa.field("evaluation_id", pa.int64()),
        pa.field("cache_key_digest", pa.string()),
        pa.field("operation", pa.string()),
        pa.field("lookup_result", pa.string()),
        pa.field("status", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("route_key", pa.string()),
        pa.field("decision_id", pa.int64()),
        pa.field("kind", pa.string()),
        pa.field("operator", pa.string()),
        pa.field("exact_started", pa.bool_()),
        pa.field("exact_completed", pa.bool_()),
        pa.field("feasible", pa.bool_()),
        pa.field("accepted", pa.bool_()),
        pa.field("global_best", pa.bool_()),
        pa.field("candidate_vehicle_delta", pa.int64()),
        pa.field("native_fallback", pa.bool_()),
        pa.field("failure_reason", pa.string()),
    ]
)


class ReplayIntegrityError(RuntimeError):
    """Raised at the first invalid logical event."""


@dataclass(frozen=True, slots=True)
class VerifiedShardBundle:
    """A previously authenticated shard exposed as bounded Arrow batches."""

    shard_id: str
    axis_budgets: Mapping[str, int]
    source_schema_version: str
    event_batches: tuple[pa.RecordBatch, ...] = ()
    persistence_ledgers: PersistenceLedger = field(default_factory=dict)
    event_batch_factory: Callable[[], Iterable[pa.RecordBatch]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    sparse_global_best_factory: Callable[[], Iterable[Mapping[str, object]]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if (
            not self.shard_id
            or not self.axis_budgets
            or (not self.event_batches and self.event_batch_factory is None)
            or (self.event_batches and self.event_batch_factory is not None)
        ):
            raise ValueError(
                "verified shard identity, axes, and exactly one event batch source are required"
            )
        if self.source_schema_version not in {
            "screening_decisions_v3",
            "screening_decisions_v2",
            "legacy",
        }:
            raise ValueError("unsupported verified shard source schema")
        axes = dict(self.axis_budgets)
        if any(
            not axis
            or isinstance(budget, bool)
            or not isinstance(budget, int)
            or budget <= 0
            for axis, budget in axes.items()
        ):
            raise ValueError("axis budgets must be positive integers")
        for batch in self.event_batches:
            if not isinstance(batch, pa.RecordBatch):
                raise TypeError("event_batches must contain Arrow RecordBatch values")
            missing = _REQUIRED_COLUMNS - set(batch.schema.names)
            if missing:
                raise ValueError(f"event batch lacks replay columns: {sorted(missing)}")
        ledgers = {
            axis: tuple(entries) for axis, entries in self.persistence_ledgers.items()
        }
        if ledgers and set(ledgers) != set(axes):
            raise ValueError("persistence ledger axes do not match the shard axes")
        for axis, entries in ledgers.items():
            if not entries or any(
                isinstance(row_count, bool)
                or not isinstance(row_count, int)
                or row_count <= 0
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for row_count, digest in entries
            ):
                raise ValueError(f"persistence ledger is invalid on {axis}")
        object.__setattr__(self, "axis_budgets", MappingProxyType(axes))
        object.__setattr__(self, "persistence_ledgers", MappingProxyType(ledgers))

    def iter_event_batches(self) -> Iterable[pa.RecordBatch]:
        batches = (
            self.event_batch_factory()
            if self.event_batch_factory is not None
            else iter(self.event_batches)
        )
        for batch in batches:
            if not isinstance(batch, pa.RecordBatch):
                raise TypeError("event batch source yielded a non-RecordBatch value")
            missing = _REQUIRED_COLUMNS - set(batch.schema.names)
            if missing:
                raise ValueError(f"event batch lacks replay columns: {sorted(missing)}")
            yield batch

    def iter_sparse_global_bests(self) -> Iterable[Mapping[str, object]]:
        if self.sparse_global_best_factory is not None:
            yield from self.sparse_global_best_factory()
            return
        for batch in self.iter_event_batches():
            mask = pc.and_(
                pc.and_(
                    pc.equal(batch.column("event_type"), "candidate_state"),
                    pc.equal(pc.fill_null(batch.column("accepted"), False), True),
                ),
                pc.equal(pc.fill_null(batch.column("global_best"), False), True),
            )
            for row in batch.filter(mask).to_pylist():
                yield cast(Mapping[str, object], row)


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    """Small deterministic result returned by either replay adapter."""

    backend: ReplayBackend
    shard_id: str
    event_count: int
    first_event_id: int
    last_event_id: int
    exact_started: int
    exact_completed: int
    accepted_candidates: int
    global_bests: int
    native_fallback_count: int
    deadline_axes: tuple[str, ...]
    axis_exact_counts: AxisCounts
    event_type_counts: EventTypeCounts
    failure_reason_counts: EventTypeCounts
    event_token_sha256: str


def _as_array(value: pa.Array | pa.ChunkedArray) -> pa.Array:
    return value.combine_chunks() if isinstance(value, pa.ChunkedArray) else value


def _nulls(data_type: pa.DataType, length: int) -> pa.Array:
    return pa.nulls(length, type=data_type)


def _constant(value: object, data_type: pa.DataType, length: int) -> pa.Array:
    return pa.array([value], type=data_type).take(pa.array(np.zeros(length, dtype=np.int64)))


def _mapped_values(
    identifiers: pa.Array,
    mapping: Mapping[int, str],
    *,
    field_name: str,
) -> pa.Array:
    keys = pa.array(tuple(mapping), type=identifiers.type)
    positions = pc.index_in(identifiers, value_set=keys)
    unknown = pc.and_(pc.invert(pc.is_null(identifiers)), pc.is_null(positions))
    if bool(pc.any(unknown).as_py()):
        raise ArtifactIntegrityError(f"native replay encountered an unknown {field_name}")
    values = pa.array(tuple(mapping.values()), type=pa.string())
    return _as_array(pc.take(values, positions))


def _table_from_projected_batches(
    reader: ArtifactReader,
    relative_path: str,
    *,
    schema: pa.Schema,
    columns: tuple[str, ...],
    batch_size: int,
    maximum_rows: int,
) -> pa.Table:
    batches: list[pa.RecordBatch] = []
    row_count = 0
    for batch in reader.iter_parquet_batches(
        relative_path,
        schema=schema,
        batch_size=batch_size,
        columns=columns,
    ):
        row_count += batch.num_rows
        if row_count > maximum_rows:
            raise ArtifactIntegrityError(
                f"native replay bounded-state limit exceeded for {relative_path}"
            )
        batches.append(batch)
    if not batches:
        projected = pa.schema([schema.field(name) for name in columns])
        return pa.Table.from_batches([], schema=projected)
    return pa.Table.from_batches(batches).combine_chunks()


def _ordinary_replay_batch(
    batch: pa.RecordBatch,
    *,
    lane_dictionary: Mapping[int, str],
    operator_dictionary: Mapping[int, str],
    route_dictionary: Mapping[int, str],
) -> pa.RecordBatch:
    length = batch.num_rows
    lane = _mapped_values(
        batch.column("lane_id"),
        lane_dictionary,
        field_name="lane_id",
    )
    lane_axis_mapping = {
        identifier: value.partition(":")[0] for identifier, value in lane_dictionary.items()
    }
    benchmark_axis = _mapped_values(
        batch.column("lane_id"),
        lane_axis_mapping,
        field_name="lane_id",
    )
    operator = _mapped_values(
        batch.column("operator_id"),
        operator_dictionary,
        field_name="operator_id",
    )
    route_key = _mapped_values(
        batch.column("route_id"),
        route_dictionary,
        field_name="route_id",
    )
    raw_event_type = batch.column("event_type")
    event_type = _as_array(
        pc.if_else(pc.is_null(raw_event_type), batch.column("record_type"), raw_event_type)
    )
    extras = pc.fill_null(batch.column("extras_json"), "")
    extracted = pc.extract_regex(
        extras,
        r'"lookup_result"\s*:\s*"(?P<lookup_result>[^"]*)"',
    )
    assert isinstance(extracted, pa.StructArray)
    lookup_result = extracted.field("lookup_result")
    native_fallback = _as_array(
        pc.match_substring_regex(
            extras,
            r'"native_(?:protocol_)?fallbacks?"\s*:\s*(?:true|[1-9][0-9]*)',
        )
    )
    by_name: dict[str, pa.Array] = {
        "event_id": batch.column("event_id"),
        "benchmark_axis": benchmark_axis,
        "lane": lane,
        "event_type": event_type,
        "timestamp_seconds": batch.column("timestamp_seconds"),
        "started_at": batch.column("started_at"),
        "completed_at": batch.column("completed_at"),
        "iteration": batch.column("iteration"),
        "evaluation_id": batch.column("evaluation_id"),
        "cache_key_digest": batch.column("cache_key_digest"),
        "operation": batch.column("operation"),
        "lookup_result": lookup_result,
        "status": batch.column("status"),
        "reason": batch.column("reason"),
        "route_key": route_key,
        "decision_id": batch.column("decision_id"),
        "kind": batch.column("kind"),
        "operator": operator,
        "exact_started": batch.column("exact_started"),
        "exact_completed": batch.column("exact_completed"),
        "feasible": batch.column("feasible"),
        "accepted": batch.column("accepted"),
        "global_best": batch.column("global_best"),
        "candidate_vehicle_delta": batch.column("candidate_vehicle_delta"),
        "native_fallback": native_fallback,
        "failure_reason": batch.column("failure_reason"),
    }
    if any(len(value) != length for value in by_name.values()):
        raise ArtifactIntegrityError("native ordinary replay projection width mismatch")
    return pa.record_batch([by_name[name] for name in REPLAY_SCHEMA.names], schema=REPLAY_SCHEMA)


def _screening_replay_batch(
    batch: pa.RecordBatch,
    *,
    definitions: pa.Table,
    lane_dictionary: Mapping[int, str],
    operator_dictionary: Mapping[int, str],
    route_dictionary: Mapping[int, str],
) -> pa.RecordBatch:
    length = batch.num_rows
    definition_ids = _as_array(definitions["definition_id"])
    positions = pc.index_in(batch.column("definition_id"), value_set=definition_ids)
    if bool(pc.any(pc.is_null(positions)).as_py()):
        raise ArtifactIntegrityError("screening occurrence refers to an unknown definition")

    def take_definition(name: str) -> pa.Array:
        return _as_array(pc.take(definitions[name], positions))

    lane_ids = take_definition("lane_id")
    operator_ids = take_definition("operator_id")
    route_ids = take_definition("route_id")
    lane = _mapped_values(lane_ids, lane_dictionary, field_name="screening lane_id")
    operator = _mapped_values(
        operator_ids,
        operator_dictionary,
        field_name="screening operator_id",
    )
    route_key = _mapped_values(
        route_ids,
        route_dictionary,
        field_name="screening route_id",
    )
    by_name: dict[str, pa.Array] = {
        "event_id": batch.column("event_id"),
        "benchmark_axis": take_definition("benchmark_axis"),
        "lane": lane,
        "event_type": _constant("screening_decision", pa.string(), length),
        "timestamp_seconds": _nulls(pa.float64(), length),
        "started_at": batch.column("started_at"),
        "completed_at": batch.column("completed_at"),
        "iteration": batch.column("iteration"),
        "evaluation_id": _nulls(pa.int64(), length),
        "cache_key_digest": _nulls(pa.string(), length),
        "operation": _nulls(pa.string(), length),
        "lookup_result": _nulls(pa.string(), length),
        "status": take_definition("status"),
        "reason": _as_array(
            pc.if_else(
                pc.equal(pc.fill_null(take_definition("reason"), ""), ""),
                pa.scalar(None, type=pa.string()),
                take_definition("reason"),
            )
        ),
        "route_key": route_key,
        "decision_id": batch.column("decision_id"),
        "kind": _nulls(pa.string(), length),
        "operator": operator,
        "exact_started": _nulls(pa.bool_(), length),
        "exact_completed": _nulls(pa.bool_(), length),
        "feasible": _nulls(pa.bool_(), length),
        "accepted": _nulls(pa.bool_(), length),
        "global_best": _nulls(pa.bool_(), length),
        "candidate_vehicle_delta": _nulls(pa.int64(), length),
        "native_fallback": _constant(False, pa.bool_(), length),
        "failure_reason": _nulls(pa.string(), length),
    }
    return pa.record_batch([by_name[name] for name in REPLAY_SCHEMA.names], schema=REPLAY_SCHEMA)


def _batch_prefix_length(batch: pa.RecordBatch, maximum_event_id: int) -> int:
    event_ids = np.asarray(batch.column("event_id"), dtype=np.int64)
    return int(np.searchsorted(event_ids, maximum_event_id, side="right"))


def _next_nonempty(iterator: Iterator[pa.RecordBatch]) -> pa.RecordBatch | None:
    for batch in iterator:
        if batch.num_rows:
            return batch
    return None


def _merge_replay_batches(
    ordinary: Iterable[pa.RecordBatch],
    screening: Iterable[pa.RecordBatch],
) -> Iterable[pa.RecordBatch]:
    ordinary_iterator = iter(ordinary)
    screening_iterator = iter(screening)
    ordinary_batch = _next_nonempty(ordinary_iterator)
    screening_batch = _next_nonempty(screening_iterator)
    while ordinary_batch is not None and screening_batch is not None:
        ordinary_last = int(ordinary_batch.column("event_id")[-1].as_py())
        screening_last = int(screening_batch.column("event_id")[-1].as_py())
        boundary = min(ordinary_last, screening_last)
        ordinary_count = _batch_prefix_length(ordinary_batch, boundary)
        screening_count = _batch_prefix_length(screening_batch, boundary)
        pieces = [
            batch.slice(0, count)
            for batch, count in (
                (ordinary_batch, ordinary_count),
                (screening_batch, screening_count),
            )
            if count
        ]
        combined = pa.Table.from_batches(pieces)
        order = pc.sort_indices(combined, sort_keys=[("event_id", "ascending")])
        sorted_table = pc.take(combined, order)
        assert isinstance(sorted_table, pa.Table)
        yield from sorted_table.combine_chunks().to_batches()
        ordinary_batch = (
            ordinary_batch.slice(ordinary_count)
            if ordinary_count < ordinary_batch.num_rows
            else _next_nonempty(ordinary_iterator)
        )
        screening_batch = (
            screening_batch.slice(screening_count)
            if screening_count < screening_batch.num_rows
            else _next_nonempty(screening_iterator)
        )
    if ordinary_batch is not None:
        yield ordinary_batch
        yield from ordinary_iterator
    if screening_batch is not None:
        yield screening_batch
        yield from screening_iterator


def verified_artifact_shard_bundle(
    *,
    reader: ArtifactReader,
    shard_id: str,
    event_relative_path: str,
    axis_budgets: Mapping[str, int],
    batch_size: int = 65_536,
) -> VerifiedShardBundle:
    """Create a restartable, bounded native Arrow source from verified v3 artifacts."""

    event_path = Path(event_relative_path)
    trace_relative_path = event_path.with_name(
        event_path.name.replace("_events_", "_trace_", 1)
    ).with_suffix(".json")
    trace = reader.read_json(trace_relative_path)
    if trace.get("screening_schema_version") != "screening_decisions_v3":
        raise ArtifactIntegrityError("native Arrow replay requires screening_decisions_v3")
    references = {
        name: trace.get(name)
        for name in (
            "route_dictionary_ref",
            "screening_definitions_ref",
            "screening_occurrences_ref",
        )
    }
    if any(not isinstance(value, str) or not value for value in references.values()):
        raise ArtifactIntegrityError("native Arrow replay trace references are incomplete")
    lane_dictionary = {
        int(key): str(value) for key, value in dict(trace.get("lane_dictionary", {})).items()
    }
    operator_dictionary = {
        int(key): str(value)
        for key, value in dict(trace.get("operator_dictionary", {})).items()
    }
    if not lane_dictionary or not operator_dictionary:
        raise ArtifactIntegrityError("native Arrow replay dictionaries are incomplete")
    raw_axes = trace.get("axes")
    if not isinstance(raw_axes, Mapping):
        raise ArtifactIntegrityError("native Arrow replay trace axes are missing")
    persistence_ledgers: dict[str, tuple[tuple[int, str], ...]] = {}
    for axis, raw_axis in raw_axes.items():
        if not isinstance(raw_axis, Mapping):
            raise ArtifactIntegrityError("native Arrow replay trace axis is invalid")
        pipeline = raw_axis.get("persistence_pipeline")
        ledger = pipeline.get("batch_ledger") if isinstance(pipeline, Mapping) else None
        if not isinstance(ledger, list):
            raise ArtifactIntegrityError("native Arrow replay persistence ledger is missing")
        entries: list[tuple[int, str]] = []
        for raw_entry in ledger:
            if not isinstance(raw_entry, Mapping):
                raise ArtifactIntegrityError("native Arrow replay ledger entry is invalid")
            row_count = raw_entry.get("row_count")
            digest = raw_entry.get("event_token_sha256")
            if (
                isinstance(row_count, bool)
                or not isinstance(row_count, int)
                or row_count <= 0
                or not isinstance(digest, str)
            ):
                raise ArtifactIntegrityError("native Arrow replay ledger fields are invalid")
            entries.append((row_count, digest))
        persistence_ledgers[str(axis)] = tuple(entries)

    def load_route_dictionary() -> dict[int, str]:
        route_table = _table_from_projected_batches(
            reader,
            cast(str, references["route_dictionary_ref"]),
            schema=ROUTE_DICTIONARY_SCHEMA,
            columns=("route_id", "canonical_route_key"),
            batch_size=batch_size,
            maximum_rows=MAX_REPLAY_ROUTE_IDENTITIES,
        )
        return {
            int(identifier): str(route_key)
            for identifier, route_key in zip(
                route_table["route_id"].to_pylist(),
                route_table["canonical_route_key"].to_pylist(),
                strict=True,
            )
        }

    def batches() -> Iterable[pa.RecordBatch]:
        route_dictionary = load_route_dictionary()
        definitions = _table_from_projected_batches(
            reader,
            cast(str, references["screening_definitions_ref"]),
            schema=V3_SCREENING_DEFINITIONS_SCHEMA,
            columns=(
                "definition_id",
                "lane_id",
                "operator_id",
                "route_id",
                "status",
                "reason",
                "benchmark_axis",
            ),
            batch_size=batch_size,
            maximum_rows=MAX_REPLAY_SCREENING_DEFINITIONS,
        )
        ordinary = (
            _ordinary_replay_batch(
                batch,
                lane_dictionary=lane_dictionary,
                operator_dictionary=operator_dictionary,
                route_dictionary=route_dictionary,
            )
            for batch in reader.iter_parquet_batches(
                event_relative_path,
                schema=EVENTS_SCHEMA,
                batch_size=batch_size,
            )
        )
        screening = (
            _screening_replay_batch(
                batch,
                definitions=definitions,
                lane_dictionary=lane_dictionary,
                operator_dictionary=operator_dictionary,
                route_dictionary=route_dictionary,
            )
            for batch in reader.iter_parquet_batches(
                cast(str, references["screening_occurrences_ref"]),
                schema=V3_SCREENING_OCCURRENCES_SCHEMA,
                batch_size=batch_size,
            )
        )
        yield from _merge_replay_batches(ordinary, screening)

    def sparse_global_bests() -> Iterable[Mapping[str, object]]:
        route_dictionary = load_route_dictionary()
        columns = (
            "event_id",
            "event_type",
            "timestamp_seconds",
            "lane_id",
            "iteration",
            "candidate_route_ids",
            "accepted",
            "global_best",
            "extras_json",
        )
        for batch in reader.iter_parquet_batches(
            event_relative_path,
            schema=EVENTS_SCHEMA,
            batch_size=batch_size,
            columns=columns,
        ):
            mask = pc.and_(
                pc.and_(
                    pc.equal(batch.column("event_type"), "candidate_state"),
                    pc.equal(pc.fill_null(batch.column("accepted"), False), True),
                ),
                pc.equal(pc.fill_null(batch.column("global_best"), False), True),
            )
            filtered = batch.filter(mask)
            for row in filtered.to_pylist():
                extras_raw = row.get("extras_json")
                try:
                    extras = orjson.loads(extras_raw) if isinstance(extras_raw, str) else {}
                except orjson.JSONDecodeError as error:
                    raise ArtifactIntegrityError(
                        "global-best extras_json is invalid"
                    ) from error
                if not isinstance(extras, dict):
                    raise ArtifactIntegrityError("global-best extras_json is not an object")
                raw_route_ids = row.get("candidate_route_ids")
                if not isinstance(raw_route_ids, list):
                    raise ArtifactIntegrityError(
                        "global-best event lacks candidate route IDs"
                    )
                try:
                    candidate_route_keys = [
                        route_dictionary[int(identifier)] for identifier in raw_route_ids
                    ]
                    lane = lane_dictionary[int(row["lane_id"])]
                except (KeyError, TypeError, ValueError) as error:
                    raise ArtifactIntegrityError(
                        "global-best event has an unknown dictionary ID"
                    ) from error
                yield {
                    "event_id": row.get("event_id"),
                    "event_type": "candidate_state",
                    "benchmark_axis": extras.get(
                        "benchmark_axis",
                        lane.partition(":")[0],
                    ),
                    "lane": lane,
                    "timestamp_seconds": row.get("timestamp_seconds"),
                    "iteration": row.get("iteration"),
                    "accepted": True,
                    "global_best": True,
                    "candidate_route_keys": candidate_route_keys,
                    "candidate_full_route_keys": extras.get(
                        "candidate_full_route_keys"
                    ),
                    "candidate_objective_key": extras.get("candidate_objective_key"),
                }

    return VerifiedShardBundle(
        shard_id=shard_id,
        axis_budgets=axis_budgets,
        source_schema_version="screening_decisions_v3",
        persistence_ledgers=persistence_ledgers,
        event_batch_factory=batches,
        sparse_global_best_factory=sparse_global_bests,
    )


@dataclass(slots=True)
class _AxisState:
    budget_seconds: int
    last_evaluation_id: int = 0
    exact_started: int = 0
    exact_completed: int = 0
    accepted_candidates: int = 0
    global_bests: int = 0
    deadline_lanes: set[str] | None = None
    cache_keys: set[str] | None = None
    cache_misses: set[str] | None = None
    completed_exact_keys: set[str] | None = None

    def __post_init__(self) -> None:
        self.deadline_lanes = set()
        self.cache_keys = set()
        self.cache_misses = set()
        self.completed_exact_keys = set()


@dataclass(slots=True)
class _LedgerState:
    entries: tuple[tuple[int, str], ...]
    index: int = 0
    seen: int = 0
    hasher: object = field(default_factory=hashlib.sha256)


class _PythonReplayState:
    def __init__(
        self,
        axis_budgets: Mapping[str, int],
        persistence_ledgers: PersistenceLedger,
    ) -> None:
        self.states = {axis: _AxisState(budget) for axis, budget in axis_budgets.items()}
        self.ledgers = {
            axis: _LedgerState(entries) for axis, entries in persistence_ledgers.items()
        }
        self.previous_event_id = 0
        self.first_event_id = 0
        self.event_count = 0
        self.native_fallback_count = 0
        self.event_type_counts: Counter[str] = Counter()
        self.failure_reason_counts: Counter[str] = Counter()
        self.hasher = hashlib.sha256()

    def consume(self, event: Mapping[str, object]) -> None:
        raw_event_id = event.get("event_id")
        if (
            isinstance(raw_event_id, bool)
            or not isinstance(raw_event_id, int)
            or raw_event_id <= self.previous_event_id
        ):
            raise ReplayIntegrityError("event IDs are not strictly increasing")
        if self.first_event_id == 0:
            self.first_event_id = raw_event_id
        self.previous_event_id = raw_event_id
        self.event_count += 1

        axis = str(event.get("benchmark_axis") or str(event.get("lane", "")).partition(":")[0])
        state = self.states.get(axis)
        if state is None:
            raise ReplayIntegrityError(f"event refers to an unknown benchmark axis: {axis}")
        lane = str(event.get("lane", "")) or axis
        event_type = str(event.get("event_type", ""))
        self.event_type_counts[event_type] += 1
        token = orjson.dumps(_event_token(event)) + b"\n"
        self.hasher.update(token)
        ledger = self.ledgers.get(axis)
        if ledger is not None:
            if ledger.index >= len(ledger.entries):
                raise ReplayIntegrityError(
                    f"persistence ledger ended before the event stream: {axis}"
                )
            hasher = ledger.hasher
            if not hasattr(hasher, "update") or not hasattr(hasher, "hexdigest"):
                raise ReplayIntegrityError("persistence ledger replay state is invalid")
            hasher.update(token)
            ledger.seen += 1
            expected_rows, expected_digest = ledger.entries[ledger.index]
            if ledger.seen == expected_rows:
                if hasher.hexdigest() != expected_digest:
                    raise ReplayIntegrityError(
                        f"persistence batch digest does not replay: {axis}/{ledger.index}"
                    )
                ledger.index += 1
                ledger.seen = 0
                ledger.hasher = hashlib.sha256()
            elif ledger.seen > expected_rows:
                raise ReplayIntegrityError(
                    f"persistence batch row count overflow: {axis}/{ledger.index}"
                )

        failure_reason = str(event.get("failure_reason") or "")
        fallback = event.get("native_fallback") is True or (
            event_type == "execution_error" and "fallback" in failure_reason.casefold()
        )
        if fallback:
            self.native_fallback_count += 1
            raise ReplayIntegrityError(f"native/protocol fallback observed on {axis}")
        if failure_reason:
            self.failure_reason_counts[failure_reason] += 1

        assert state.deadline_lanes is not None
        assert state.cache_keys is not None
        assert state.cache_misses is not None
        assert state.completed_exact_keys is not None
        if event_type == "deadline_boundary":
            timestamp = _finite_number(event.get("timestamp_seconds"))
            if timestamp is None or not math.isclose(
                timestamp,
                state.budget_seconds,
                rel_tol=0.0,
                abs_tol=1.0,
            ):
                raise ReplayIntegrityError(f"deadline boundary timestamp mismatch on {axis}")
            state.deadline_lanes.add(lane)

        if event_type == "route_evaluation":
            evaluation_id = event.get("evaluation_id")
            if (
                isinstance(evaluation_id, bool)
                or not isinstance(evaluation_id, int)
                or evaluation_id != state.last_evaluation_id + 1
            ):
                raise ReplayIntegrityError(f"route evaluation ordering mismatch on {axis}")
            state.last_evaluation_id = evaluation_id
            if event.get("exact_started") is True:
                if lane in state.deadline_lanes:
                    raise ReplayIntegrityError(f"exact work started after deadline on {axis}")
                state.exact_started += 1
                started_at = _finite_number(event.get("started_at"))
                if started_at is None or started_at < 0.0:
                    raise ReplayIntegrityError(f"invalid exact start time on {axis}")
                if event.get("exact_completed") is True:
                    state.exact_completed += 1
                    completed_at = _finite_number(event.get("completed_at"))
                    if (
                        completed_at is None
                        or completed_at < started_at
                        or completed_at > state.budget_seconds
                    ):
                        raise ReplayIntegrityError(f"exact completion crosses deadline on {axis}")
                    digest = str(event.get("cache_key_digest") or "")
                    if not digest or digest not in state.cache_misses:
                        raise ReplayIntegrityError(
                            f"exact completion lacks a preceding cache miss on {axis}"
                        )
                    state.cache_misses.discard(digest)
                    state.completed_exact_keys.add(digest)
                if event.get("status") == "interrupted_deadline":
                    state.deadline_lanes.add(lane)

        if event_type == "cache_event":
            operation = str(event.get("operation") or "")
            digest = str(event.get("cache_key_digest") or "")
            if operation == "store":
                if lane in state.deadline_lanes:
                    raise ReplayIntegrityError(f"cache store observed after deadline on {axis}")
                if not digest:
                    raise ReplayIntegrityError(f"cache store lacks key on {axis}")
                if digest not in state.completed_exact_keys:
                    raise ReplayIntegrityError(f"cache store precedes exact completion on {axis}")
                state.completed_exact_keys.discard(digest)
                state.cache_keys.add(digest)
            elif operation == "evict":
                if digest not in state.cache_keys:
                    raise ReplayIntegrityError(f"cache eviction refers to an absent key on {axis}")
                state.cache_keys.discard(digest)
            elif operation == "lookup_result":
                lookup_result = event.get("lookup_result")
                if lookup_result == "hit" and digest not in state.cache_keys:
                    raise ReplayIntegrityError(f"cache hit precedes store on {axis}")
                if lookup_result == "miss":
                    if not digest:
                        raise ReplayIntegrityError(f"cache miss lacks key on {axis}")
                    state.cache_misses.add(digest)

        if event_type == "candidate_state" and event.get("accepted") is True:
            if lane in state.deadline_lanes:
                raise ReplayIntegrityError(f"candidate accepted after deadline on {axis}")
            vehicle_delta = event.get("candidate_vehicle_delta")
            if (
                isinstance(vehicle_delta, bool)
                or not isinstance(vehicle_delta, int)
                or vehicle_delta > 0
            ):
                raise ReplayIntegrityError(
                    f"accepted candidate violates vehicle-first policy on {axis}"
                )
            state.accepted_candidates += 1
            if event.get("global_best") is True:
                state.global_bests += 1

    def summary(self, *, backend: ReplayBackend, shard_id: str) -> ReplaySummary:
        for axis, ledger in self.ledgers.items():
            if ledger.index != len(ledger.entries) or ledger.seen != 0:
                raise ReplayIntegrityError(
                    f"persistence ledger does not cover the complete event stream: {axis}"
                )
        return ReplaySummary(
            backend=backend,
            shard_id=shard_id,
            event_count=self.event_count,
            first_event_id=self.first_event_id,
            last_event_id=self.previous_event_id,
            exact_started=sum(state.exact_started for state in self.states.values()),
            exact_completed=sum(state.exact_completed for state in self.states.values()),
            accepted_candidates=sum(
                state.accepted_candidates for state in self.states.values()
            ),
            global_bests=sum(state.global_bests for state in self.states.values()),
            native_fallback_count=self.native_fallback_count,
            deadline_axes=tuple(
                axis
                for axis, state in sorted(self.states.items())
                if state.deadline_lanes
            ),
            axis_exact_counts=tuple(
                (axis, state.exact_started, state.exact_completed)
                for axis, state in sorted(self.states.items())
            ),
            event_type_counts=tuple(sorted(self.event_type_counts.items())),
            failure_reason_counts=tuple(sorted(self.failure_reason_counts.items())),
            event_token_sha256=self.hasher.hexdigest(),
        )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _event_token(event: Mapping[str, object]) -> tuple[object, ...]:
    event_type = str(event.get("event_type", ""))
    raw_reason = event.get("reason")
    reason = "" if event_type == "screening_decision" and raw_reason is None else raw_reason or None
    return (
        event_type,
        event.get("benchmark_axis"),
        event.get("lane"),
        event.get("iteration"),
        event.get("operator"),
        event.get("route_key"),
        event.get("decision_id"),
        event.get("kind") or None,
        event.get("operation") or None,
        event.get("status") or None,
        reason,
        event.get("exact_started"),
        event.get("exact_completed"),
        event.get("feasible"),
    )


def _python_reference_replay(bundle: VerifiedShardBundle) -> ReplaySummary:
    state = _PythonReplayState(bundle.axis_budgets, bundle.persistence_ledgers)
    for batch in bundle.iter_event_batches():
        for row in batch.to_pylist():
            state.consume(cast(Mapping[str, object], row))
    return state.summary(backend="python_reference", shard_id=bundle.shard_id)


def _validity(array: pa.Array) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
    return np.asarray(pc.invert(array.is_null()), dtype=np.uint8)


def _encoded_column(
    array: pa.Array,
) -> tuple[object, np.ndarray[tuple[int], np.dtype[np.uint8]], object]:
    valid = _validity(array)
    if pa.types.is_string(array.type):
        encoded = pc.dictionary_encode(array)
        assert isinstance(encoded, pa.DictionaryArray)
        indices = np.asarray(pc.fill_null(encoded.indices, -1), dtype=np.int32)
        dictionary = tuple(str(value) for value in encoded.dictionary.to_pylist())
        return indices, valid, dictionary
    if pa.types.is_boolean(array.type):
        values = np.asarray(pc.cast(pc.fill_null(array, False), pa.uint8()), dtype=np.uint8)
        return values, valid, ()
    if pa.types.is_integer(array.type):
        values = np.asarray(pc.fill_null(array, -1), dtype=np.int64)
        return values, valid, ()
    if pa.types.is_floating(array.type):
        values = np.asarray(pc.fill_null(array, math.nan), dtype=np.float64)
        return values, valid, ()
    raise TypeError(f"unsupported native replay column type: {array.type}")


def _encode_arrow_batch(batch: pa.RecordBatch) -> dict[str, object]:
    """Expose Arrow columns to C++ without constructing logical row dictionaries."""

    return {
        name: _encoded_column(batch.column(batch.schema.get_field_index(name)))
        for name in sorted(_REQUIRED_COLUMNS)
    }


def _native_replay(bundle: VerifiedShardBundle) -> ReplaySummary:
    try:
        state = _core.Stage052ReplayState(
            dict(bundle.axis_budgets),
            dict(bundle.persistence_ledgers),
        )
        for batch in bundle.iter_event_batches():
            state.consume(_encode_arrow_batch(batch))
        raw = state.finish()
    except ValueError as error:
        raise ReplayIntegrityError(str(error)) from error

    def as_integer(value: object, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ReplayIntegrityError(f"native replay {field_name} is not an integer")
        return value

    def integer(field_name: str) -> int:
        return as_integer(raw.get(field_name), field_name)

    def rows(field_name: str, width: int) -> tuple[tuple[object, ...], ...]:
        value = raw.get(field_name)
        if not isinstance(value, list | tuple):
            raise ReplayIntegrityError(f"native replay {field_name} is not an array")
        normalized: list[tuple[object, ...]] = []
        for item in value:
            if not isinstance(item, list | tuple) or len(item) != width:
                raise ReplayIntegrityError(
                    f"native replay {field_name} contains an invalid row"
                )
            normalized.append(tuple(item))
        return tuple(normalized)

    raw_deadline_axes = raw.get("deadline_axes")
    if not isinstance(raw_deadline_axes, list | tuple):
        raise ReplayIntegrityError("native replay deadline_axes is not an array")
    return ReplaySummary(
        backend="native_arrow",
        shard_id=bundle.shard_id,
        event_count=integer("event_count"),
        first_event_id=integer("first_event_id"),
        last_event_id=integer("last_event_id"),
        exact_started=integer("exact_started"),
        exact_completed=integer("exact_completed"),
        accepted_candidates=integer("accepted_candidates"),
        global_bests=integer("global_bests"),
        native_fallback_count=integer("native_fallback_count"),
        deadline_axes=tuple(str(axis) for axis in raw_deadline_axes),
        axis_exact_counts=tuple(
            (
                str(axis),
                as_integer(started, "axis exact started"),
                as_integer(completed, "axis exact completed"),
            )
            for axis, started, completed in rows("axis_exact_counts", 3)
        ),
        event_type_counts=tuple(
            (str(key), as_integer(value, "event type count"))
            for key, value in rows("event_type_counts", 2)
        ),
        failure_reason_counts=tuple(
            (str(key), as_integer(value, "failure reason count"))
            for key, value in rows("failure_reason_counts", 2)
        ),
        event_token_sha256=str(raw["event_token_sha256"]),
    )


def replay_verified_shard(
    bundle: VerifiedShardBundle,
    *,
    backend: ReplayBackend,
) -> ReplaySummary:
    """Replay one verified shard through the explicitly selected adapter."""

    if backend == "python_reference":
        return _python_reference_replay(bundle)
    if backend == "native_arrow":
        return _native_replay(bundle)
    raise ValueError(f"unsupported Stage 5.2 replay backend: {backend}")
