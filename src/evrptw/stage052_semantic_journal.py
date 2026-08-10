"""Bounded, signed Stage 5.2 semantic evidence journals.

The module owns the persistence seam for the complete runtime semantic trace.
Callers provide one completed ALNS result and receive a JSON-serializable
descriptor; reviewers consume the descriptor through the verified iterator.
Neither caller needs to materialize a second event collection.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, cast

import numpy as np

from evrptw.stage052_atomic import publish_no_replace
from evrptw.stage052_event_spool import (
    STAGE052_ATTEMPT_AUDIT_MAGIC,
    STAGE052_ATTEMPT_AUDIT_SCHEMA_VERSION,
    decode_stage052_attempt_payload,
    iter_stage052_attempt_audit,
)
from evrptw.stage052_physical_telemetry import PhysicalTelemetryWriter
from evrptw.stage052_writer_pipeline import (
    BatchWriteResult,
    BoundedFifoWriter,
    validate_pipeline_receipt,
)

SCHEMA_VERSION = "stage05.2-semantic-journal-v3"
_PIPELINE_BATCH_ROWS = 256
_NATIVE_CONTROL_RAW_DOMAIN = b"stage05.2-native-control-raw-v1\0"
SEMANTIC_STREAM_NAMES = (
    "candidate_state",
    "operator",
    "stage04",
    "candidate_transaction",
    "exact_work",
    "exact_result",
    "cache",
    "screening",
    "deadline",
    "termination",
    "native_failure",
)

# Physical scheduling observations are preserved by ``PhysicalTelemetryWriter``
# before canonicalization.  They must never enter the fixed-work semantic hash:
# worker timing and completion order may legitimately differ between equivalent
# architectures and builds.
PARALLEL_BATCH_NON_CANONICAL_FIELDS = frozenset(
    {
        "worker_count",
        "worker_protocol",
        "submission_order",
        "completion_order",
        "merge_order",
        "semantic_completion_order",
        "physical_task_receipts",
        "physical_observation",
        "chunk_sizes",
        "completed_indices",
        "batch_ordinal",
    }
)
SEMANTIC_TRAJECTORY_IMPLEMENTATION_STATUSES = frozenset(
    {
        "pair_prefilter_rejected_aggregate",
        "prefilter_rejected_aggregate",
    }
)


class _Objective(Protocol):
    @property
    def key(self) -> tuple[object, ...]: ...


class _Trace(Protocol):
    @property
    def runtime_semantic_events(self) -> Sequence[dict[str, object]]: ...

    @property
    def runtime_semantic_pipeline_receipts(self) -> tuple[dict[str, object], ...]: ...

    @property
    def runtime_semantic_pipeline_evidence(
        self,
    ) -> tuple[tuple[dict[str, object], Path], ...]: ...

    def seal_runtime_semantic_storage(self) -> dict[str, object] | None: ...


class SemanticJournalResult(Protocol):
    @property
    def measurement_trace(self) -> _Trace | None: ...

    @property
    def effective_iterations(self) -> int: ...

    @property
    def candidate_control_statistics(self) -> Mapping[str, object]: ...

    @property
    def objective(self) -> _Objective | None: ...

    @property
    def termination_reason(self) -> str: ...

    @property
    def iterations(self) -> int: ...

    @property
    def exact_started_calls(self) -> int: ...

    @property
    def exact_completed_calls(self) -> int: ...

    @property
    def exact_interrupted_calls(self) -> int: ...

    @property
    def native_control_journal_events(self) -> Sequence[Mapping[str, object]]: ...

    @property
    def native_execution_statistics(self) -> Mapping[str, object]: ...


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


class _TimedBinarySink:
    """Measure and hash compressed bytes as the streaming compressor writes."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._digest = hashlib.sha256()
        self.write_seconds = 0.0
        self.hash_seconds = 0.0

    def write(self, data: bytes) -> int:
        started = time.perf_counter()
        written = self._stream.write(data)
        self.write_seconds += time.perf_counter() - started
        if written != len(data):
            raise OSError("semantic journal compressed write was incomplete")
        started = time.perf_counter()
        self._digest.update(data)
        self.hash_seconds += time.perf_counter() - started
        return written

    def flush(self) -> None:
        self._stream.flush()

    def tell(self) -> int:
        return self._stream.tell()

    def fileno(self) -> int:
        return self._stream.fileno()

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


_PERSISTENCE_TIMING_FIELDS = (
    "event_serialization_seconds",
    "compression_seconds",
    "write_seconds",
    "fsync_seconds",
    "hash_seconds",
    "atomic_publish_seconds",
    "pipeline_union_seconds",
    "runtime_spool_union_seconds",
    "publication_seconds",
    "formal_attributed_seconds",
    "total_seconds",
)


def _validate_persistence_receipt(raw: object) -> None:
    if not isinstance(raw, dict) or set(raw) != set(_PERSISTENCE_TIMING_FIELDS):
        raise ValueError("semantic journal persistence receipt is invalid")
    for field_name in _PERSISTENCE_TIMING_FIELDS:
        value = raw[field_name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"semantic journal {field_name} is invalid")
    # Runtime-spool work occurs inside the measured solver interval.  It is
    # retained as a diagnostic subphase but must not be added a second time to
    # the post-solve persistence interval.
    attributed = float(raw["pipeline_union_seconds"]) + float(raw["publication_seconds"])
    if not math.isclose(
        attributed,
        float(raw["formal_attributed_seconds"]),
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ValueError("semantic journal persistence timings do not reconcile")


def evidence_json_value(value: object) -> object:
    """Convert NumPy and non-finite values into canonical JSON evidence."""

    if isinstance(value, np.generic):
        return evidence_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"nonfinite_float": "nan"}
        return {"nonfinite_float": "positive_inf" if value > 0.0 else "negative_inf"}
    if isinstance(value, dict):
        return {str(key): evidence_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [evidence_json_value(item) for item in value]
    return value


def canonical_trace_event(event: dict[str, object]) -> dict[str, object]:
    """Remove implementation timing while retaining replayable semantics."""

    canonical = {
        key: value
        for key, value in event.items()
        if key not in {"timestamp_seconds", "duration_seconds"}
    }
    if canonical.get("event_type") == "candidate_cache_transaction":
        return {}
    if canonical.get("event_type") in {
        "deadline_boundary",
        "exact_budget_boundary",
        "candidate_control_boundary",
    }:
        if canonical.get("termination_boundary") is not True:
            return {}
        canonical.pop("termination_boundary", None)
    if canonical.get("event_type") == "candidate_screening_aggregate":
        canonical = {
            key: value
            for key, value in canonical.items()
            if key
            in {
                "event_type",
                "physical_event_type",
                "status",
                "lane",
                "iteration",
                "operator",
                "calls",
                "passes",
                "rejections",
                "cache_hits",
                "exact_call_blocked",
                "reason_counts",
                "candidate_pool_hash",
                "screening_pool_hash",
                "screening_integrity_evidence",
                "semantic_event_id",
            }
        }
    if canonical.get("event_type") == "screening_decision":
        raw_checks = canonical.get("checks", [])
        if isinstance(raw_checks, list):
            normalized_checks: list[dict[str, object]] = []
            boolean_checks = {
                "route_structure",
                "single_segment_battery_reachability",
            }
            for raw_check in raw_checks:
                if not isinstance(raw_check, dict):
                    continue
                check_name = str(raw_check.get("check", ""))
                check_value = raw_check.get("value")
                normalized_checks.append(
                    {
                        "check": check_name,
                        "status": raw_check.get("status"),
                        "value": (
                            bool(check_value) if check_name in boolean_checks else check_value
                        ),
                        "reason": raw_check.get("reason", ""),
                    }
                )
            canonical["checks"] = normalized_checks
        canonical = {
            key: value
            for key, value in canonical.items()
            if key
            in {
                "event_type",
                "route_key",
                "lane",
                "iteration",
                "operator",
                "status",
                "first_failed_check",
                "reason",
                "checks",
                "demand",
                "min_time_window_slack",
                "distance_lower_bound",
                "distance_increment_lower_bound",
                "single_segment_reachable",
                "structural_energy_lower_bound",
                "negative_cache_hit",
                "exact_call_blocked",
                "semantic_event_id",
            }
        }
    if canonical.get("event_type") == "candidate_control_budget":
        canonical.pop("accounting", None)
        context = canonical.get("context")
        if isinstance(context, str) and context.endswith(":native_candidate_round"):
            canonical["context"] = (
                context.removesuffix(":native_candidate_round") + ":candidate_pool"
            )
    if canonical.get("event_type") == "exact_batch_started":
        canonical.pop("transaction_sha256", None)
    if canonical.get("event_type") == "candidate_plan_decision":
        canonical.pop("batch_ordinal", None)
        canonical.pop("transaction_status_code", None)
        canonical.pop("native_transaction_id", None)
    if canonical.get("event_type") == "cache_event":
        operation = canonical.get("operation")
        if operation == "lookup":
            return {}
        if operation in {"hit", "candidate_pending_hit", "miss"}:
            canonical = {
                key: value
                for key, value in canonical.items()
                if key
                in {
                    "event_type",
                    "route_key",
                    "cache_key_digest",
                    "lane",
                    "iteration",
                    "operator",
                    "current_entries",
                    "current_bytes",
                    "semantic_event_id",
                }
            }
            canonical["event_type"] = "cache_lookup_result"
            canonical["status"] = "miss" if operation == "miss" else "hit"
            canonical["cache_scope"] = (
                "candidate_pending" if operation == "candidate_pending_hit" else "committed"
            )
        elif operation in {"store", "evict", "reconcile", "oversize_not_cached"}:
            canonical = {
                key: value
                for key, value in canonical.items()
                if key
                in {
                    "event_type",
                    "operation",
                    "route_key",
                    "cache_key_digest",
                    "lane",
                    "iteration",
                    "operator",
                    "native_transaction_id",
                    "reason",
                    "entry_bytes",
                    "current_entries",
                    "current_bytes",
                    "pending_result_digest",
                    "existing_result_digest",
                    "semantic_event_id",
                }
            }
            canonical["event_type"] = "cache_lifecycle"
            canonical["status"] = operation
        else:
            raise ValueError(f"unsupported runtime cache operation: {operation!r}")
    if canonical.get("event_type") == "parallel_batch":
        canonical["event_type"] = "candidate_batch_complete"
        canonical["status"] = "complete"
        for key in PARALLEL_BATCH_NON_CANONICAL_FIELDS:
            canonical.pop(key, None)
        sequences = canonical.get("customer_sequences")
        if not isinstance(sequences, list | tuple):
            raise ValueError("candidate batch requires customer_sequences")
        canonical["result_count"] = len(sequences)
    if canonical.get("event_type") == "candidate_state":
        route_keys = canonical.get("candidate_route_keys")
        if not isinstance(route_keys, list | tuple) or not all(
            isinstance(key, str) for key in route_keys
        ):
            raise ValueError("candidate_state requires candidate_route_keys as a string array")
        full_route_keys = canonical.get("candidate_full_route_keys", [])
        if not isinstance(full_route_keys, list | tuple) or not all(
            isinstance(key, str) for key in full_route_keys
        ):
            raise ValueError("candidate_full_route_keys must be a string array when present")
        canonical = {
            key: canonical[key]
            for key in (
                "event_type",
                "lane",
                "iteration",
                "operator",
                "candidate_objective_key",
                "candidate_feasible",
                "accepted",
                "status",
                "reason",
                "semantic_event_id",
            )
            if key in canonical
        }
        canonical["candidate_route_keys"] = list(route_keys)
        canonical["candidate_full_route_keys"] = list(full_route_keys)
        identity = {
            "candidate_route_keys": list(route_keys),
            "candidate_full_route_keys": list(full_route_keys),
        }
        canonical["candidate_id"] = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    return canonical


def _canonical_runtime_event(
    raw_event: Mapping[str, object],
    *,
    stream_counts: dict[str, int],
    semantic_event_id: int,
) -> tuple[dict[str, object] | None, int]:
    """Project one raw runtime event onto the canonical semantic stream."""

    stream_name = raw_event.get("semantic_stream")
    if not isinstance(stream_name, str) or stream_name not in stream_counts:
        raise ValueError("runtime semantic event names an unknown stream")
    source_runtime_event_id = raw_event.get("runtime_causal_event_id")
    if (
        isinstance(source_runtime_event_id, bool)
        or not isinstance(source_runtime_event_id, int)
        or source_runtime_event_id != raw_event.get("semantic_event_id")
    ):
        raise ValueError("runtime semantic event lost its causal source identity")
    native_telemetry = {
        key: value for key, value in raw_event.items() if key.startswith("runtime_native_")
    }
    event = {
        key: value
        for key, value in raw_event.items()
        if key
        not in {
            "semantic_stream",
            "runtime_causal_event_id",
            "runtime_native_event_id",
            "runtime_native_stream_code",
            "runtime_native_event_code",
            "runtime_native_lane_id",
            "runtime_native_operator_id",
            "runtime_native_iteration",
            "runtime_native_transaction_id",
            "runtime_native_subject_id",
            "runtime_native_status_code",
            "runtime_native_flags",
        }
    }
    normalized = evidence_json_value(canonical_trace_event(event))
    if not isinstance(normalized, dict):
        raise AssertionError("canonical event normalization lost object identity")
    if not normalized:
        return None, semantic_event_id
    if native_telemetry:
        normalized["native_telemetry"] = evidence_json_value(native_telemetry)
    runtime_event_id = normalized.pop("semantic_event_id", None)
    if isinstance(runtime_event_id, bool) or not isinstance(runtime_event_id, int):
        raise ValueError("runtime semantic event lost its event ID")
    semantic_event_id += 1
    canonical = {
        **normalized,
        "runtime_event_id": runtime_event_id,
        "semantic_event_id": semantic_event_id,
        "stream_ordinal": stream_counts[stream_name],
        "semantic_stream": stream_name,
        "semantic_sequence": semantic_event_id - 1,
    }
    stream_counts[stream_name] += 1
    return canonical, semantic_event_id


def _canonical_events(
    result: SemanticJournalResult,
    *,
    physical_writer: PhysicalTelemetryWriter | None = None,
    physical_observer: Callable[[Mapping[str, object]], None] | None = None,
) -> Iterator[dict[str, object]]:
    trace = result.measurement_trace
    if trace is None:
        raise ValueError("semantic journal requires a measurement trace")
    runtime_events = trace.runtime_semantic_events
    if physical_writer is not None and physical_observer is not None:
        raise ValueError("physical telemetry has multiple owners")
    stream_counts = {name: 0 for name in SEMANTIC_STREAM_NAMES}
    semantic_event_id = 0
    termination: dict[str, object] | None = None
    for runtime_event_ordinal, raw_event in enumerate(runtime_events, start=1):
        raw_runtime_event_id = raw_event.get("semantic_event_id")
        if (
            isinstance(raw_runtime_event_id, bool)
            or not isinstance(raw_runtime_event_id, int)
            or raw_runtime_event_id != runtime_event_ordinal
        ):
            raise ValueError("runtime semantic trace IDs are not contiguous")
        if physical_writer is not None:
            physical_writer.observe(raw_event)
        if physical_observer is not None:
            physical_observer(raw_event)
        canonical, semantic_event_id = _canonical_runtime_event(
            raw_event,
            stream_counts=stream_counts,
            semantic_event_id=semantic_event_id,
        )
        if canonical is None:
            continue
        stream_name = raw_event.get("semantic_stream")
        if stream_name == "termination":
            termination = canonical
        yield canonical

    required_nonempty = {
        "stage04",
        "exact_work",
        "exact_result",
        "cache",
        "screening",
        "termination",
    }
    if result.effective_iterations > 0:
        required_nonempty.update({"candidate_state", "operator"})
    if result.candidate_control_statistics.get("enabled") is True:
        required_nonempty.add("candidate_transaction")
    missing = sorted(name for name in required_nonempty if not stream_counts[name])
    if missing:
        raise ValueError("runtime semantic journal is incomplete: " + ", ".join(missing))
    if result.objective is None:
        raise ValueError("runtime semantic termination requires an objective")
    if (
        termination is None
        or stream_counts["termination"] != 1
        or termination.get("event_type") != "termination"
        or termination.get("status") != result.termination_reason
        or termination.get("iterations") != result.iterations
        or termination.get("effective_iterations") != result.effective_iterations
        or termination.get("exact_started_calls") != result.exact_started_calls
        or termination.get("exact_completed_calls") != result.exact_completed_calls
        or termination.get("exact_interrupted_calls") != result.exact_interrupted_calls
        or termination.get("objective_key") != evidence_json_value(list(result.objective.key))
        or termination.get("semantic_event_id") != semantic_event_id
    ):
        raise ValueError("runtime semantic termination does not reconcile")
    if result.termination_reason != "iteration_limit" and not stream_counts["deadline"]:
        raise ValueError("runtime semantic deadline boundary is missing")


def semantic_bundle_path(
    axis_path: Path,
    descriptor: Mapping[str, object] | None = None,
) -> Path:
    """Resolve the same-directory evidence bundle without path traversal."""

    expected = axis_path.with_suffix(".semantic.bundle")
    if descriptor is None:
        return expected
    raw_name = descriptor.get("path")
    if (
        not isinstance(raw_name, str)
        or Path(raw_name).name != raw_name
        or raw_name != expected.name
    ):
        raise ValueError("semantic bundle must remain in the axis same directory")
    bundle = axis_path.parent / raw_name
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("semantic bundle is not a regular directory")
    return bundle


@dataclass(frozen=True, slots=True)
class _SemanticBatchPayload:
    encoded_events: bytes
    physical_events: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class _RuntimeSpoolEvidence:
    receipt: Mapping[str, object]
    audit_path: Path


class _SemanticJournalWriter:
    """Writer-thread owner for semantic, physical, ledger, hash, and fsync I/O."""

    def __init__(
        self,
        directory: Path,
        runtime_evidence: Sequence[_RuntimeSpoolEvidence],
        native_control_events: Sequence[Mapping[str, object]],
        native_control_source_sha256: str | None,
    ) -> None:
        self.journal_path = directory / "events.jsonl.gz"
        self.sidecar_path = directory / "events.sha256"
        self.ledger_path = directory / "pipeline-ledger.jsonl"
        self.ledger_sidecar_path = directory / "pipeline-ledger.sha256"
        self.runtime_path = directory / "runtime-spool-pipelines.jsonl.gz"
        self.runtime_sidecar_path = directory / "runtime-spool-pipelines.sha256"
        self.native_control_path = directory / "native-control-events.jsonl.gz"
        self.native_control_sidecar_path = directory / "native-control-events.sha256"
        self._directory = directory
        self._runtime_evidence = tuple(runtime_evidence)
        self._native_control_events = tuple(dict(event) for event in native_control_events)
        self._native_control_source_sha256 = native_control_source_sha256
        self._raw: BinaryIO | None = None
        self._ledger_raw: BinaryIO | None = None
        self._journal_sink: _TimedBinarySink | None = None
        self._ledger_sink: _TimedBinarySink | None = None
        self._runtime_raw: BinaryIO | None = None
        self._runtime_sink: _TimedBinarySink | None = None
        self._runtime_compressed: gzip.GzipFile | None = None
        self._native_control_raw: BinaryIO | None = None
        self._native_control_sink: _TimedBinarySink | None = None
        self._native_control_compressed: gzip.GzipFile | None = None
        self._native_control_rows_sha256: str | None = None
        self._compressed: gzip.GzipFile | None = None
        self._physical = PhysicalTelemetryWriter(directory)
        self._physical_started = False
        self._physical_descriptor: dict[str, object] | None = None
        self._compressor_call_seconds = 0.0
        self._batch_hash_seconds = 0.0
        self._audit_write_seconds = 0.0
        self._audit_hash_seconds = 0.0
        self._sidecar_write_seconds = 0.0
        self._fsync_seconds = 0.0
        self._finalized = False

    def _ensure_started(self) -> None:
        if self._raw is not None:
            return
        self._raw = self.journal_path.open("xb", buffering=1024 * 1024)
        self._ledger_raw = self.ledger_path.open("xb", buffering=1024 * 1024)
        self._journal_sink = _TimedBinarySink(self._raw)
        self._ledger_sink = _TimedBinarySink(self._ledger_raw)
        started = time.perf_counter()
        self._compressed = gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=6,
            fileobj=cast(BinaryIO, self._journal_sink),
            mtime=0,
        )
        self._compressor_call_seconds += time.perf_counter() - started
        self._physical.__enter__()
        self._physical_started = True
        self._runtime_raw = self.runtime_path.open("xb", buffering=1024 * 1024)
        self._runtime_sink = _TimedBinarySink(self._runtime_raw)
        started = time.perf_counter()
        self._runtime_compressed = gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=1,
            fileobj=cast(BinaryIO, self._runtime_sink),
            mtime=0,
        )
        self._compressor_call_seconds += time.perf_counter() - started
        self._write_runtime_receipts()
        if self._native_control_events:
            if (
                not isinstance(self._native_control_source_sha256, str)
                or len(self._native_control_source_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self._native_control_source_sha256
                )
            ):
                raise RuntimeError("native control source digest is invalid")
            self._native_control_raw = self.native_control_path.open("xb", buffering=1024 * 1024)
            self._native_control_sink = _TimedBinarySink(self._native_control_raw)
            started = time.perf_counter()
            self._native_control_compressed = gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=1,
                fileobj=cast(BinaryIO, self._native_control_sink),
                mtime=0,
            )
            self._compressor_call_seconds += time.perf_counter() - started
            rows_digest = hashlib.sha256(_NATIVE_CONTROL_RAW_DOMAIN)
            for event in self._native_control_events:
                encoded = _canonical_bytes(event) + b"\n"
                rows_digest.update(encoded)
                started = time.perf_counter()
                self._native_control_compressed.write(encoded)
                self._compressor_call_seconds += time.perf_counter() - started
            self._native_control_rows_sha256 = rows_digest.hexdigest()
        elif self._native_control_source_sha256 is not None:
            raise RuntimeError("native control digest exists without raw control rows")

    def _write_runtime_receipts(self) -> None:
        assert self._runtime_compressed is not None
        for pipeline_ordinal, evidence in enumerate(self._runtime_evidence):
            receipt = evidence.receipt
            attempted = receipt.get("attempted_batch_ledger")
            if not isinstance(attempted, list):
                raise RuntimeError("runtime spool attempted ledger is invalid")
            audit = self._persist_runtime_attempt_audit(
                pipeline_ordinal,
                evidence.audit_path,
                attempted,
            )
            summary = {
                key: value
                for key, value in receipt.items()
                if key not in {"batch_ledger", "attempted_batch_ledger"}
            }
            header = (
                _canonical_bytes(
                    {
                        "record_type": "pipeline",
                        "pipeline_ordinal": pipeline_ordinal,
                        "lifecycle_status": (
                            "terminal"
                            if pipeline_ordinal == len(self._runtime_evidence) - 1
                            else "superseded"
                        ),
                        "summary": summary,
                        "attempt_audit": audit,
                    }
                )
                + b"\n"
            )
            started = time.perf_counter()
            self._runtime_compressed.write(header)
            self._compressor_call_seconds += time.perf_counter() - started
            for row in attempted:
                line = (
                    _canonical_bytes(
                        {
                            "record_type": "attempt",
                            "pipeline_ordinal": pipeline_ordinal,
                            "row": row,
                        }
                    )
                    + b"\n"
                )
                started = time.perf_counter()
                self._runtime_compressed.write(line)
                self._compressor_call_seconds += time.perf_counter() - started

    def _persist_runtime_attempt_audit(
        self,
        pipeline_ordinal: int,
        source_path: Path,
        attempted: Sequence[object],
    ) -> dict[str, object]:
        if source_path.is_symlink() or not source_path.is_file():
            raise RuntimeError("runtime spool attempted audit source is invalid")
        target = self._directory / (
            f"runtime-spool-pipeline-{pipeline_ordinal:04d}.attempts.frames"
        )
        sidecar = target.with_suffix(target.suffix + ".sha256")
        digest = hashlib.sha256()
        observed_frames = 0
        logical_rows = 0
        with target.open("xb", buffering=1024 * 1024) as output:
            started = time.perf_counter()
            output.write(STAGE052_ATTEMPT_AUDIT_MAGIC)
            self._audit_write_seconds += time.perf_counter() - started
            started = time.perf_counter()
            digest.update(STAGE052_ATTEMPT_AUDIT_MAGIC)
            self._audit_hash_seconds += time.perf_counter() - started
            for frame in iter_stage052_attempt_audit(source_path):
                if observed_frames >= len(attempted):
                    raise RuntimeError("runtime spool attempted audit has extra frames")
                raw_row = attempted[observed_frames]
                if not isinstance(raw_row, Mapping):
                    raise RuntimeError("runtime spool attempted ledger row is invalid")
                decoded_rows = decode_stage052_attempt_payload(
                    frame.decoded_payload,
                    row_count=frame.row_count,
                )
                started = time.perf_counter()
                payload_digest = hashlib.sha256(frame.decoded_payload).hexdigest()
                self._audit_hash_seconds += time.perf_counter() - started
                if (
                    raw_row.get("attempt_ordinal") != frame.attempt_ordinal
                    or raw_row.get("batch_ordinal") != frame.batch_ordinal
                    or raw_row.get("row_count") != frame.row_count
                    or raw_row.get("payload_bytes") != frame.payload_bytes
                    or raw_row.get("sha256") != payload_digest
                ):
                    raise RuntimeError("runtime spool attempted audit does not match its ledger")
                for chunk in (frame.raw_header, frame.compressed_payload):
                    started = time.perf_counter()
                    output.write(chunk)
                    self._audit_write_seconds += time.perf_counter() - started
                    started = time.perf_counter()
                    digest.update(chunk)
                    self._audit_hash_seconds += time.perf_counter() - started
                observed_frames += 1
                logical_rows += len(decoded_rows)
            started = time.perf_counter()
            output.flush()
            self._audit_write_seconds += time.perf_counter() - started
            started = time.perf_counter()
            os.fsync(output.fileno())
            self._fsync_seconds += time.perf_counter() - started
        if observed_frames != len(attempted):
            raise RuntimeError("runtime spool attempted audit is incomplete")
        self._write_sidecar(sidecar, digest.hexdigest())
        return {
            "schema_version": STAGE052_ATTEMPT_AUDIT_SCHEMA_VERSION,
            "path": target.name,
            "sidecar_path": sidecar.name,
            "sha256": digest.hexdigest(),
            "bytes": target.stat().st_size,
            "sidecar_bytes": sidecar.stat().st_size,
            "frame_count": observed_frames,
            "logical_row_count": logical_rows,
        }

    def consume(
        self,
        ordinal: int,
        row_count: int,
        raw_payload: object,
    ) -> BatchWriteResult:
        if not isinstance(raw_payload, _SemanticBatchPayload) or not raw_payload.encoded_events:
            raise RuntimeError("semantic journal writer batch is invalid")
        self._ensure_started()
        assert self._compressed is not None
        assert self._ledger_sink is not None
        assert self._runtime_raw is not None
        assert self._runtime_sink is not None
        assert self._runtime_compressed is not None
        for physical_event in raw_payload.physical_events:
            self._physical.observe(physical_event)
        started = time.perf_counter()
        digest = hashlib.sha256(raw_payload.encoded_events).hexdigest()
        self._batch_hash_seconds += time.perf_counter() - started
        started = time.perf_counter()
        self._compressed.write(raw_payload.encoded_events)
        self._compressor_call_seconds += time.perf_counter() - started
        ledger_row = (
            _canonical_bytes(
                {
                    "batch_ordinal": ordinal,
                    "row_count": row_count,
                    "payload_bytes": len(raw_payload.encoded_events),
                    "sha256": digest,
                }
            )
            + b"\n"
        )
        self._ledger_sink.write(ledger_row)
        return BatchWriteResult(
            row_count=row_count,
            payload_bytes=len(raw_payload.encoded_events),
            sha256=digest,
        )

    def _write_sidecar(self, path: Path, digest: str) -> None:
        with path.open("x", encoding="ascii") as stream:
            started = time.perf_counter()
            stream.write(digest + "\n")
            stream.flush()
            self._sidecar_write_seconds += time.perf_counter() - started
            started = time.perf_counter()
            os.fsync(stream.fileno())
            self._fsync_seconds += time.perf_counter() - started

    def finalize(self) -> None:
        self._ensure_started()
        assert self._compressed is not None
        assert self._raw is not None
        assert self._ledger_raw is not None
        assert self._journal_sink is not None
        assert self._ledger_sink is not None
        assert self._runtime_raw is not None
        assert self._runtime_sink is not None
        assert self._runtime_compressed is not None
        started = time.perf_counter()
        self._compressed.close()
        self._compressor_call_seconds += time.perf_counter() - started
        self._compressed = None
        started = time.perf_counter()
        self._runtime_compressed.close()
        self._compressor_call_seconds += time.perf_counter() - started
        self._runtime_compressed = None
        if self._native_control_compressed is not None:
            started = time.perf_counter()
            self._native_control_compressed.close()
            self._compressor_call_seconds += time.perf_counter() - started
            self._native_control_compressed = None
        try:
            self._physical_descriptor = self._physical.finish()
        finally:
            if self._physical_started:
                self._physical.__exit__(None, None, None)
                self._physical_started = False
        streams = [self._raw, self._ledger_raw, self._runtime_raw]
        if self._native_control_raw is not None:
            streams.append(self._native_control_raw)
        for stream in streams:
            started = time.perf_counter()
            stream.flush()
            self._sidecar_write_seconds += time.perf_counter() - started
            started = time.perf_counter()
            os.fsync(stream.fileno())
            self._fsync_seconds += time.perf_counter() - started
        self._write_sidecar(self.sidecar_path, self._journal_sink.hexdigest())
        self._write_sidecar(
            self.ledger_sidecar_path,
            self._ledger_sink.hexdigest(),
        )
        self._write_sidecar(
            self.runtime_sidecar_path,
            self._runtime_sink.hexdigest(),
        )
        if self._native_control_sink is not None:
            self._write_sidecar(
                self.native_control_sidecar_path,
                self._native_control_sink.hexdigest(),
            )
        directory_fd = os.open(self._directory, os.O_RDONLY)
        try:
            started = time.perf_counter()
            os.fsync(directory_fd)
            self._fsync_seconds += time.perf_counter() - started
        finally:
            os.close(directory_fd)
        self._raw.close()
        self._ledger_raw.close()
        self._runtime_raw.close()
        if self._native_control_raw is not None:
            self._native_control_raw.close()
        self._raw = None
        self._ledger_raw = None
        self._runtime_raw = None
        self._native_control_raw = None
        self._finalized = True

    def abort(self) -> None:
        """Close temporary handles after a producer or writer failure."""

        if self._compressed is not None:
            with suppress(Exception):
                self._compressed.close()
            self._compressed = None
        if self._runtime_compressed is not None:
            with suppress(Exception):
                self._runtime_compressed.close()
            self._runtime_compressed = None
        if self._native_control_compressed is not None:
            with suppress(Exception):
                self._native_control_compressed.close()
            self._native_control_compressed = None
        if self._physical_started:
            self._physical.__exit__(None, None, None)
            self._physical_started = False
        for stream in (
            self._raw,
            self._ledger_raw,
            self._runtime_raw,
            self._native_control_raw,
        ):
            if stream is not None and not stream.closed:
                stream.close()
        self._raw = None
        self._ledger_raw = None
        self._runtime_raw = None
        self._native_control_raw = None

    def metadata(self) -> dict[str, object]:
        if (
            not self._finalized
            or self._journal_sink is None
            or self._ledger_sink is None
            or self._runtime_sink is None
            or self._physical_descriptor is None
        ):
            raise RuntimeError("semantic journal writer is not finalized")
        physical_persistence = self._physical_descriptor.get("persistence")
        if not isinstance(physical_persistence, dict):
            raise RuntimeError("physical telemetry persistence receipt is missing")
        compression_seconds = max(
            0.0,
            self._compressor_call_seconds
            - self._journal_sink.write_seconds
            - self._journal_sink.hash_seconds
            - self._runtime_sink.write_seconds
            - self._runtime_sink.hash_seconds,
        )
        native_control_descriptor: dict[str, object] | None = None
        native_control_write_seconds = 0.0
        native_control_hash_seconds = 0.0
        if self._native_control_events:
            if (
                self._native_control_sink is None
                or self._native_control_rows_sha256 is None
                or self._native_control_source_sha256 is None
            ):
                raise RuntimeError("native control persistence is incomplete")
            native_control_write_seconds = self._native_control_sink.write_seconds
            native_control_hash_seconds = self._native_control_sink.hash_seconds
            compression_seconds = max(
                0.0,
                compression_seconds - native_control_write_seconds - native_control_hash_seconds,
            )
            native_control_descriptor = {
                "schema_version": "stage05.2-native-control-raw-v1",
                "path": self.native_control_path.name,
                "sidecar_path": self.native_control_sidecar_path.name,
                "compressed_sha256": self._native_control_sink.hexdigest(),
                "rows_sha256": self._native_control_rows_sha256,
                "native_source_sha256": self._native_control_source_sha256,
                "event_count": len(self._native_control_events),
                "compressed_bytes": self.native_control_path.stat().st_size,
                "sidecar_bytes": self.native_control_sidecar_path.stat().st_size,
            }
        return {
            "journal_sha256": self._journal_sink.hexdigest(),
            "ledger_sha256": self._ledger_sink.hexdigest(),
            "compressed_bytes": self.journal_path.stat().st_size,
            "ledger_bytes": self.ledger_path.stat().st_size,
            "sidecar_bytes": self.sidecar_path.stat().st_size,
            "ledger_sidecar_bytes": self.ledger_sidecar_path.stat().st_size,
            "runtime_sha256": self._runtime_sink.hexdigest(),
            "runtime_compressed_bytes": self.runtime_path.stat().st_size,
            "runtime_sidecar_bytes": self.runtime_sidecar_path.stat().st_size,
            "runtime_pipeline_count": len(self._runtime_evidence),
            "native_control_events": native_control_descriptor,
            "physical_telemetry": self._physical_descriptor,
            "compression_seconds": compression_seconds,
            "write_seconds": (
                self._journal_sink.write_seconds
                + self._ledger_sink.write_seconds
                + self._runtime_sink.write_seconds
                + native_control_write_seconds
                + self._sidecar_write_seconds
                + self._audit_write_seconds
                + float(physical_persistence["write_seconds"])
            ),
            "hash_seconds": (
                self._journal_sink.hash_seconds
                + self._ledger_sink.hash_seconds
                + self._runtime_sink.hash_seconds
                + native_control_hash_seconds
                + self._batch_hash_seconds
                + self._audit_hash_seconds
                + float(physical_persistence["hash_seconds"])
            ),
            "fsync_seconds": self._fsync_seconds + float(physical_persistence["fsync_seconds"]),
        }


def _write_semantic_journal(
    axis_path: Path,
    result: SemanticJournalResult,
) -> dict[str, object]:
    """Atomically publish semantic and physical evidence through one rename."""

    total_started = time.perf_counter()
    trace = result.measurement_trace
    if trace is None:
        raise ValueError("semantic journal requires a measurement trace")
    seal_runtime = getattr(trace, "seal_runtime_semantic_storage", None)
    if callable(seal_runtime):
        seal_runtime()
    runtime_evidence_raw = getattr(trace, "runtime_semantic_pipeline_evidence", ())
    if not isinstance(runtime_evidence_raw, tuple | list):
        raise ValueError("runtime semantic pipeline evidence is invalid")
    runtime_evidence: list[_RuntimeSpoolEvidence] = []
    runtime_spool_union_seconds = 0.0
    observed_audit_paths: set[Path] = set()
    for raw_evidence in runtime_evidence_raw:
        if (
            not isinstance(raw_evidence, tuple | list)
            or len(raw_evidence) != 2
            or not isinstance(raw_evidence[0], Mapping)
            or not isinstance(raw_evidence[1], Path)
        ):
            raise ValueError("runtime semantic pipeline evidence row is invalid")
        raw_receipt, audit_path = raw_evidence
        validate_pipeline_receipt(raw_receipt, require_finalized=True)
        receipt = dict(raw_receipt)
        if (
            audit_path in observed_audit_paths
            or audit_path.is_symlink()
            or not audit_path.is_file()
        ):
            raise ValueError("runtime semantic attempted audit identity is invalid")
        observed_audit_paths.add(audit_path)
        runtime_evidence.append(_RuntimeSpoolEvidence(receipt=receipt, audit_path=audit_path))
        runtime_spool_union_seconds += float(receipt["producer_writer_wall_union_seconds"])
    compatibility_receipts = getattr(trace, "runtime_semantic_pipeline_receipts", ())
    if compatibility_receipts and len(compatibility_receipts) != len(runtime_evidence):
        raise ValueError("runtime semantic receipts are missing attempted audits")
    native_control_events_raw = getattr(result, "native_control_journal_events", ())
    if not isinstance(native_control_events_raw, tuple | list) or not all(
        isinstance(event, Mapping) for event in native_control_events_raw
    ):
        raise ValueError("native control journal events are invalid")
    native_control_events = tuple(
        cast(Mapping[str, object], event) for event in native_control_events_raw
    )
    native_statistics = getattr(result, "native_execution_statistics", {})
    if not isinstance(native_statistics, Mapping):
        raise ValueError("native execution statistics are invalid")
    native_control_source = native_statistics.get("control_journal_sha256")
    if not native_control_events:
        native_control_source = None

    bundle_path = semantic_bundle_path(axis_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.exists():
        raise FileExistsError(f"semantic evidence bundle already exists: {bundle_path}")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{bundle_path.name}.tmp-",
            dir=bundle_path.parent,
        )
    )
    event_count = 0
    uncompressed_bytes = 0
    stream_counts = {name: 0 for name in SEMANTIC_STREAM_NAMES}
    descriptor: dict[str, object] | None = None
    event_serialization_seconds = 0.0
    atomic_publish_seconds = 0.0
    publication_seconds = 0.0
    published = False
    writer: _SemanticJournalWriter | None = None
    pipeline: BoundedFifoWriter | None = None
    pipeline_finish_attempted = False
    try:
        writer = _SemanticJournalWriter(
            temporary,
            runtime_evidence,
            native_control_events,
            cast(str | None, native_control_source),
        )
        pipeline = BoundedFifoWriter(
            name="s52-semantic-write",
            batch_row_capacity=_PIPELINE_BATCH_ROWS,
            consumer=writer.consume,
            finalizer=writer.finalize,
        )
        physical_batch: list[Mapping[str, object]] = []

        def observe_physical(raw_event: Mapping[str, object]) -> None:
            if raw_event.get("event_type") == "parallel_batch":
                physical_batch.append(raw_event)

        events = _canonical_events(result, physical_observer=observe_physical)
        exhausted = False
        while not exhausted:
            pipeline.begin_producer_turn()
            producer_started_ns = time.perf_counter_ns()
            failed = True
            encoded_batch = bytearray()
            batch_rows = 0
            try:
                while batch_rows < _PIPELINE_BATCH_ROWS:
                    try:
                        event = next(events)
                    except StopIteration:
                        exhausted = True
                        break
                    started = time.perf_counter()
                    encoded = _canonical_bytes(event) + b"\n"
                    event_serialization_seconds += time.perf_counter() - started
                    encoded_batch.extend(encoded)
                    uncompressed_bytes += len(encoded)
                    event_count += 1
                    batch_rows += 1
                    stream_name = event["semantic_stream"]
                    if not isinstance(stream_name, str):
                        raise AssertionError("validated stream name changed type")
                    stream_counts[stream_name] += 1
                if batch_rows:
                    pipeline.submit(
                        _SemanticBatchPayload(
                            encoded_events=bytes(encoded_batch),
                            physical_events=tuple(physical_batch),
                        ),
                        row_count=batch_rows,
                    )
                    physical_batch.clear()
                elif physical_batch:
                    raise RuntimeError("physical telemetry lost its semantic batch")
                failed = False
            finally:
                try:
                    pipeline.record_producer_interval(
                        producer_started_ns,
                        time.perf_counter_ns(),
                    )
                finally:
                    pipeline.end_producer_turn(force_release=failed or exhausted)

        pipeline_finish_attempted = True
        raw_pipeline_receipt = pipeline.finish()
        active_ledger = validate_pipeline_receipt(
            raw_pipeline_receipt,
            require_finalized=True,
        )
        if (
            raw_pipeline_receipt["discarded_batches"] != 0
            or len(active_ledger) != raw_pipeline_receipt["completed_batches"]
        ):
            raise RuntimeError("semantic journal writer unexpectedly rewound a batch")
        writer_metadata = writer.metadata()
        ledger_descriptor = {
            "path": writer.ledger_path.name,
            "sidecar_path": writer.ledger_sidecar_path.name,
            "sha256": writer_metadata["ledger_sha256"],
            "bytes": writer_metadata["ledger_bytes"],
            "sidecar_bytes": writer_metadata["ledger_sidecar_bytes"],
            "entry_count": len(active_ledger),
        }
        compact_pipeline_receipt = {
            key: value
            for key, value in raw_pipeline_receipt.items()
            if key not in {"batch_ledger", "attempted_batch_ledger"}
        }
        compact_pipeline_receipt["ledger"] = ledger_descriptor

        bundle_bytes = sum(path.stat().st_size for path in temporary.iterdir() if path.is_file())
        publication_started = time.perf_counter()
        started = time.perf_counter()
        publish_no_replace(temporary, bundle_path)
        atomic_publish_seconds += time.perf_counter() - started
        published = True
        parent_fd = os.open(bundle_path.parent, os.O_RDONLY)
        try:
            started = time.perf_counter()
            os.fsync(parent_fd)
            parent_fsync_seconds = time.perf_counter() - started
        finally:
            os.close(parent_fd)
        publication_seconds = time.perf_counter() - publication_started
        raw_pipeline_union = raw_pipeline_receipt["producer_writer_wall_union_seconds"]
        if isinstance(raw_pipeline_union, bool) or not isinstance(raw_pipeline_union, int | float):
            raise RuntimeError("semantic journal pipeline union is invalid")
        pipeline_union_seconds = float(raw_pipeline_union)
        raw_writer_fsync = writer_metadata["fsync_seconds"]
        if isinstance(raw_writer_fsync, bool) or not isinstance(raw_writer_fsync, int | float):
            raise RuntimeError("semantic journal writer fsync timing is invalid")
        formal_attributed_seconds = pipeline_union_seconds + publication_seconds
        descriptor = {
            "schema_version": SCHEMA_VERSION,
            "encoding": "bundle/gzip-jsonl+typed-soa+pipeline-ledger",
            "path": bundle_path.name,
            "event_path": writer.journal_path.name,
            "sidecar_path": writer.sidecar_path.name,
            "sha256": writer_metadata["journal_sha256"],
            "compressed_bytes": writer_metadata["compressed_bytes"],
            "uncompressed_bytes": uncompressed_bytes,
            "sidecar_bytes": writer_metadata["sidecar_bytes"],
            "bundle_bytes": bundle_bytes,
            "event_count": event_count,
            "stream_counts": stream_counts,
            "physical_telemetry": writer_metadata["physical_telemetry"],
            "pipeline": compact_pipeline_receipt,
            "runtime_spool_pipelines": {
                "path": writer.runtime_path.name,
                "sidecar_path": writer.runtime_sidecar_path.name,
                "sha256": writer_metadata["runtime_sha256"],
                "compressed_bytes": writer_metadata["runtime_compressed_bytes"],
                "sidecar_bytes": writer_metadata["runtime_sidecar_bytes"],
                "pipeline_count": writer_metadata["runtime_pipeline_count"],
            },
            "native_control_events": writer_metadata["native_control_events"],
            "persistence": {
                "event_serialization_seconds": event_serialization_seconds,
                "compression_seconds": writer_metadata["compression_seconds"],
                "write_seconds": writer_metadata["write_seconds"],
                "fsync_seconds": float(raw_writer_fsync) + parent_fsync_seconds,
                "hash_seconds": writer_metadata["hash_seconds"],
                "atomic_publish_seconds": atomic_publish_seconds,
                "pipeline_union_seconds": pipeline_union_seconds,
                "runtime_spool_union_seconds": runtime_spool_union_seconds,
                "publication_seconds": publication_seconds,
                "formal_attributed_seconds": formal_attributed_seconds,
                "total_seconds": time.perf_counter() - total_started,
            },
        }
    except BaseException as primary_error:
        cleanup_errors: list[BaseException] = []
        if pipeline is not None and not pipeline_finish_attempted:
            try:
                pipeline.finish()
            except BaseException as error:
                cleanup_errors.append(error)
        if writer is not None:
            try:
                writer.abort()
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            if temporary.exists():
                shutil.rmtree(temporary)
            if published and bundle_path.exists():
                shutil.rmtree(bundle_path)
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "semantic journal persistence and cleanup failed",
                [primary_error, *cleanup_errors],
            ) from None
        raise
    assert descriptor is not None
    return descriptor


def write_semantic_journal(
    axis_path: Path,
    result: SemanticJournalResult,
) -> dict[str, object]:
    """Publish one journal and release every temporary runtime audit on failure."""

    trace = result.measurement_trace
    bundle_path = semantic_bundle_path(axis_path)
    if bundle_path.exists() or bundle_path.is_symlink():
        raise FileExistsError(f"semantic evidence bundle already exists: {bundle_path}")
    try:
        return _write_semantic_journal(axis_path, result)
    except BaseException as primary_error:
        release_runtime = getattr(trace, "release_runtime_semantic_storage", None)
        if callable(release_runtime):
            try:
                release_runtime()
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "semantic journal persistence and runtime cleanup failed",
                    [primary_error, cleanup_error],
                ) from None
        raise


def iter_canonical_semantic_events(
    result: SemanticJournalResult,
) -> Iterator[dict[str, object]]:
    """Validate and stream canonical events without publishing an artifact."""

    yield from _canonical_events(result)


def _iter_verified_pipeline_ledger(
    bundle: Path,
    raw_receipt: object,
) -> Iterator[dict[str, object]]:
    if not isinstance(raw_receipt, Mapping):
        raise ValueError("semantic journal pipeline receipt is invalid")
    expected = {
        "schema_version",
        "storage_model",
        "writer_thread_count",
        "writer_daemon",
        "finalized",
        "batch_row_capacity",
        "queue_bound_batches",
        "peak_queued_batches",
        "attempted_batches",
        "submitted_batches",
        "completed_batches",
        "discarded_batches",
        "discarded_rows",
        "rewind_operations",
        "producer_wall_seconds",
        "writer_wall_seconds",
        "writer_cpu_seconds",
        "producer_wait_seconds",
        "producer_writer_wall_union_seconds",
        "ledger",
    }
    if (
        set(raw_receipt) != expected
        or raw_receipt.get("schema_version") != "stage05.2-bounded-fifo-writer-v1"
        or raw_receipt.get("storage_model") != "one_batch_bounded_non_daemon_fifo"
        or raw_receipt.get("writer_thread_count") != 1
        or raw_receipt.get("writer_daemon") is not False
        or raw_receipt.get("finalized") is not True
        or raw_receipt.get("queue_bound_batches") != 1
    ):
        raise ValueError("semantic journal pipeline identity is invalid")

    def integer(field: str) -> int:
        value = raw_receipt.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"semantic journal pipeline {field} is invalid")
        return value

    capacity = integer("batch_row_capacity")
    peak = integer("peak_queued_batches")
    attempted = integer("attempted_batches")
    submitted = integer("submitted_batches")
    completed = integer("completed_batches")
    discarded = integer("discarded_batches")
    discarded_rows = integer("discarded_rows")
    rewinds = integer("rewind_operations")
    if (
        capacity != _PIPELINE_BATCH_ROWS
        or peak != int(attempted > 0)
        or attempted != submitted
        or submitted != completed
        or discarded != 0
        or discarded_rows != 0
        or rewinds != 0
    ):
        raise ValueError("semantic journal pipeline counters are invalid")
    for field in (
        "producer_wall_seconds",
        "writer_wall_seconds",
        "writer_cpu_seconds",
        "producer_wait_seconds",
        "producer_writer_wall_union_seconds",
    ):
        value = raw_receipt.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("semantic journal pipeline timing is invalid")
    if not math.isclose(
        float(raw_receipt["producer_writer_wall_union_seconds"]),
        float(raw_receipt["producer_wall_seconds"]) + float(raw_receipt["writer_wall_seconds"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("semantic journal pipeline union is invalid")
    raw_ledger = raw_receipt.get("ledger")
    if not isinstance(raw_ledger, Mapping) or set(raw_ledger) != {
        "path",
        "sidecar_path",
        "sha256",
        "bytes",
        "sidecar_bytes",
        "entry_count",
    }:
        raise ValueError("semantic journal pipeline ledger descriptor is invalid")
    if (
        raw_ledger.get("path") != "pipeline-ledger.jsonl"
        or raw_ledger.get("sidecar_path") != "pipeline-ledger.sha256"
        or raw_ledger.get("entry_count") != completed
    ):
        raise ValueError("semantic journal pipeline ledger identity is invalid")
    expected_digest = raw_ledger.get("sha256")
    expected_bytes = raw_ledger.get("bytes")
    expected_sidecar_bytes = raw_ledger.get("sidecar_bytes")
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
        or expected_sidecar_bytes != 65
    ):
        raise ValueError("semantic journal pipeline ledger extent is invalid")
    ledger_path = bundle / "pipeline-ledger.jsonl"
    sidecar_path = bundle / "pipeline-ledger.sha256"
    if (
        ledger_path.is_symlink()
        or sidecar_path.is_symlink()
        or not ledger_path.is_file()
        or not sidecar_path.is_file()
        or sidecar_path.read_text(encoding="ascii").strip() != expected_digest
    ):
        raise ValueError("semantic journal pipeline ledger files are invalid")
    digest = hashlib.sha256()
    observed_bytes = 0
    observed_count = 0
    with ledger_path.open("rb") as source:
        for line in source:
            digest.update(line)
            observed_bytes += len(line)
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("semantic journal pipeline ledger is invalid JSONL") from error
            if not isinstance(row, dict) or set(row) != {
                "batch_ordinal",
                "row_count",
                "payload_bytes",
                "sha256",
            }:
                raise RuntimeError("semantic journal pipeline ledger row is invalid")
            row_count = row.get("row_count")
            payload_bytes = row.get("payload_bytes")
            row_digest = row.get("sha256")
            if (
                row.get("batch_ordinal") != observed_count
                or isinstance(row_count, bool)
                or not isinstance(row_count, int)
                or not 0 < row_count <= capacity
                or isinstance(payload_bytes, bool)
                or not isinstance(payload_bytes, int)
                or payload_bytes <= 0
                or not isinstance(row_digest, str)
                or len(row_digest) != 64
                or any(character not in "0123456789abcdef" for character in row_digest)
            ):
                raise RuntimeError("semantic journal pipeline ledger values are invalid")
            observed_count += 1
            yield row
    if (
        observed_count != completed
        or observed_bytes != expected_bytes
        or digest.hexdigest() != expected_digest
    ):
        raise RuntimeError("semantic journal pipeline ledger does not reconcile")


def _verify_runtime_attempt_audit(
    bundle: Path,
    raw_descriptor: object,
    attempted: Sequence[Mapping[str, object]],
    *,
    pipeline_ordinal: int,
    terminal: bool,
) -> tuple[str | None, int]:
    expected_name = f"runtime-spool-pipeline-{pipeline_ordinal:04d}.attempts.frames"
    if not isinstance(raw_descriptor, Mapping) or set(raw_descriptor) != {
        "schema_version",
        "path",
        "sidecar_path",
        "sha256",
        "bytes",
        "sidecar_bytes",
        "frame_count",
        "logical_row_count",
    }:
        raise ValueError("runtime spool attempted audit descriptor is invalid")
    expected_digest = raw_descriptor.get("sha256")
    expected_bytes = raw_descriptor.get("bytes")
    frame_count = raw_descriptor.get("frame_count")
    logical_row_count = raw_descriptor.get("logical_row_count")
    if (
        raw_descriptor.get("schema_version") != STAGE052_ATTEMPT_AUDIT_SCHEMA_VERSION
        or raw_descriptor.get("path") != expected_name
        or raw_descriptor.get("sidecar_path") != expected_name + ".sha256"
        or raw_descriptor.get("sidecar_bytes") != 65
        or not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < len(STAGE052_ATTEMPT_AUDIT_MAGIC)
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count != len(attempted)
        or isinstance(logical_row_count, bool)
        or not isinstance(logical_row_count, int)
        or logical_row_count < 0
    ):
        raise ValueError("runtime spool attempted audit extent is invalid")
    path = bundle / expected_name
    sidecar = bundle / (expected_name + ".sha256")
    if (
        path.is_symlink()
        or sidecar.is_symlink()
        or not path.is_file()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="ascii").strip() != expected_digest
    ):
        raise ValueError("runtime spool attempted audit files are invalid")

    file_digest = hashlib.sha256(STAGE052_ATTEMPT_AUDIT_MAGIC)
    observed_bytes = len(STAGE052_ATTEMPT_AUDIT_MAGIC)
    observed_rows = 0
    observed_frames = 0
    raw_event_id = 0
    semantic_event_id = 0
    semantic_digest = hashlib.sha256()
    stream_counts = {name: 0 for name in SEMANTIC_STREAM_NAMES}
    for frame in iter_stage052_attempt_audit(path):
        if observed_frames >= len(attempted):
            raise RuntimeError("runtime spool attempted audit has extra frames")
        row = attempted[observed_frames]
        payload_digest = hashlib.sha256(frame.decoded_payload).hexdigest()
        if (
            row.get("attempt_ordinal") != frame.attempt_ordinal
            or row.get("batch_ordinal") != frame.batch_ordinal
            or row.get("row_count") != frame.row_count
            or row.get("payload_bytes") != frame.payload_bytes
            or row.get("sha256") != payload_digest
        ):
            raise RuntimeError("runtime spool attempted audit does not match its ledger")
        decoded_rows = decode_stage052_attempt_payload(
            frame.decoded_payload,
            row_count=frame.row_count,
        )
        if terminal and row.get("status") == "retained":
            for envelope_stream, raw_event in decoded_rows:
                raw_event_id += 1
                if (
                    raw_event.get("semantic_stream") != envelope_stream
                    or raw_event.get("semantic_event_id") != raw_event_id
                    or raw_event.get("runtime_causal_event_id") != raw_event_id
                ):
                    raise RuntimeError("terminal runtime spool event identity is invalid")
                canonical, semantic_event_id = _canonical_runtime_event(
                    raw_event,
                    stream_counts=stream_counts,
                    semantic_event_id=semantic_event_id,
                )
                if canonical is not None:
                    semantic_digest.update(_canonical_bytes(canonical) + b"\n")
        file_digest.update(frame.raw_header)
        file_digest.update(frame.compressed_payload)
        observed_bytes += len(frame.raw_header) + len(frame.compressed_payload)
        observed_rows += len(decoded_rows)
        observed_frames += 1
    if (
        observed_frames != frame_count
        or observed_rows != logical_row_count
        or observed_bytes != expected_bytes
        or file_digest.hexdigest() != expected_digest
    ):
        raise RuntimeError("runtime spool attempted audit does not reconcile")
    return (semantic_digest.hexdigest() if terminal else None), semantic_event_id


def _validate_runtime_spool_pipelines(
    bundle: Path,
    raw_descriptor: object,
) -> tuple[float, str | None, int]:
    if not isinstance(raw_descriptor, Mapping) or set(raw_descriptor) != {
        "path",
        "sidecar_path",
        "sha256",
        "compressed_bytes",
        "sidecar_bytes",
        "pipeline_count",
    }:
        raise ValueError("runtime spool pipeline descriptor is invalid")
    if (
        raw_descriptor.get("path") != "runtime-spool-pipelines.jsonl.gz"
        or raw_descriptor.get("sidecar_path") != "runtime-spool-pipelines.sha256"
        or raw_descriptor.get("sidecar_bytes") != 65
    ):
        raise ValueError("runtime spool pipeline identity is invalid")
    expected_digest = raw_descriptor.get("sha256")
    expected_bytes = raw_descriptor.get("compressed_bytes")
    pipeline_count = raw_descriptor.get("pipeline_count")
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or isinstance(pipeline_count, bool)
        or not isinstance(pipeline_count, int)
        or pipeline_count < 0
    ):
        raise ValueError("runtime spool pipeline extent is invalid")
    path = bundle / "runtime-spool-pipelines.jsonl.gz"
    sidecar = bundle / "runtime-spool-pipelines.sha256"
    if (
        path.is_symlink()
        or sidecar.is_symlink()
        or not path.is_file()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="ascii").strip() != expected_digest
    ):
        raise ValueError("runtime spool pipeline files are invalid")
    digest = hashlib.sha256()
    observed_bytes = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            observed_bytes += len(chunk)
    if observed_bytes != expected_bytes or digest.hexdigest() != expected_digest:
        raise RuntimeError("runtime spool pipeline SHA-256 mismatch")

    observed_pipelines = 0
    union_seconds = 0.0
    current_summary: dict[str, object] | None = None
    current_attempts: list[dict[str, object]] = []
    current_lifecycle_status: str | None = None
    current_audit: object = None
    terminal_semantic_digest: str | None = None
    terminal_semantic_count = 0

    def finish_current() -> None:
        nonlocal current_summary, current_attempts, current_lifecycle_status
        nonlocal current_audit, observed_pipelines, union_seconds
        nonlocal terminal_semantic_digest, terminal_semantic_count
        if current_summary is None:
            return
        active = [
            {
                "batch_ordinal": row["batch_ordinal"],
                "row_count": row["row_count"],
                "payload_bytes": row["payload_bytes"],
                "sha256": row["sha256"],
            }
            for row in current_attempts
            if row.get("status") == "retained"
        ]
        receipt = {
            **current_summary,
            "batch_ledger": active,
            "attempted_batch_ledger": current_attempts,
        }
        validate_pipeline_receipt(receipt, require_finalized=True)
        expected_lifecycle = (
            "terminal" if observed_pipelines == pipeline_count - 1 else "superseded"
        )
        if current_lifecycle_status != expected_lifecycle:
            raise RuntimeError("runtime spool lifecycle order is invalid")
        audit_digest, audit_count = _verify_runtime_attempt_audit(
            bundle,
            current_audit,
            current_attempts,
            pipeline_ordinal=observed_pipelines,
            terminal=expected_lifecycle == "terminal",
        )
        if expected_lifecycle == "terminal":
            if terminal_semantic_digest is not None or audit_digest is None:
                raise RuntimeError("runtime spool terminal audit is ambiguous")
            terminal_semantic_digest = audit_digest
            terminal_semantic_count = audit_count
        raw_union = receipt.get("producer_writer_wall_union_seconds")
        if isinstance(raw_union, bool) or not isinstance(raw_union, int | float):
            raise ValueError("runtime spool pipeline union is invalid")
        union_seconds += float(raw_union)
        observed_pipelines += 1
        current_summary = None
        current_attempts = []
        current_lifecycle_status = None
        current_audit = None

    with gzip.open(path, "rb") as compressed_source:
        for line in compressed_source:
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("runtime spool pipeline ledger is invalid JSONL") from error
            if not isinstance(record, dict):
                raise RuntimeError("runtime spool pipeline record is invalid")
            record_type = record.get("record_type")
            ordinal = record.get("pipeline_ordinal")
            if record_type == "pipeline":
                finish_current()
                if ordinal != observed_pipelines or set(record) != {
                    "record_type",
                    "pipeline_ordinal",
                    "lifecycle_status",
                    "summary",
                    "attempt_audit",
                }:
                    raise RuntimeError("runtime spool pipeline order is invalid")
                summary = record.get("summary")
                if not isinstance(summary, dict):
                    raise RuntimeError("runtime spool pipeline summary is invalid")
                current_summary = summary
                current_lifecycle_status = record.get("lifecycle_status")
                current_audit = record.get("attempt_audit")
            elif record_type == "attempt":
                if (
                    current_summary is None
                    or ordinal != observed_pipelines
                    or set(record) != {"record_type", "pipeline_ordinal", "row"}
                    or not isinstance(record.get("row"), dict)
                ):
                    raise RuntimeError("runtime spool attempt order is invalid")
                current_attempts.append(record["row"])
            else:
                raise RuntimeError("runtime spool pipeline record type is invalid")
    finish_current()
    if observed_pipelines != pipeline_count:
        raise RuntimeError("runtime spool pipeline count mismatch")
    if pipeline_count > 0 and terminal_semantic_digest is None:
        raise RuntimeError("runtime spool terminal audit is missing")
    return union_seconds, terminal_semantic_digest, terminal_semantic_count


def iter_verified_semantic_journal(
    axis_path: Path,
    descriptor: Mapping[str, object],
) -> Iterator[dict[str, object]]:
    """Verify and stream one journal without materializing its event set."""

    if descriptor.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("semantic journal schema is unsupported")
    if descriptor.get("encoding") != "bundle/gzip-jsonl+typed-soa+pipeline-ledger":
        raise ValueError("semantic journal encoding is unsupported")
    _validate_persistence_receipt(descriptor.get("persistence"))
    bundle = semantic_bundle_path(axis_path, descriptor)
    ledger = iter(_iter_verified_pipeline_ledger(bundle, descriptor.get("pipeline")))
    (
        runtime_spool_union_seconds,
        terminal_semantic_digest,
        terminal_semantic_count,
    ) = _validate_runtime_spool_pipelines(
        bundle,
        descriptor.get("runtime_spool_pipelines"),
    )
    persistence = descriptor.get("persistence")
    if not isinstance(persistence, Mapping) or not math.isclose(
        runtime_spool_union_seconds,
        float(persistence["runtime_spool_union_seconds"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("runtime spool pipeline attribution does not reconcile")
    if descriptor.get("event_path") != "events.jsonl.gz":
        raise ValueError("semantic journal event path is invalid")
    if descriptor.get("sidecar_path") != "events.sha256":
        raise ValueError("semantic journal sidecar path is invalid")
    journal_path = bundle / "events.jsonl.gz"
    sidecar = bundle / "events.sha256"
    if (
        journal_path.is_symlink()
        or sidecar.is_symlink()
        or not journal_path.is_file()
        or not sidecar.is_file()
    ):
        raise ValueError("semantic journal bundle files are invalid")
    expected_sha256 = descriptor.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("semantic journal SHA-256 descriptor is invalid")
    if sidecar.read_text(encoding="ascii").strip() != expected_sha256:
        raise RuntimeError("semantic journal SHA-256 sidecar mismatch")
    digest = hashlib.sha256()
    compressed_bytes = 0
    with journal_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            compressed_bytes += len(chunk)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError("semantic journal SHA-256 mismatch")
    if descriptor.get("compressed_bytes") != compressed_bytes:
        raise RuntimeError("semantic journal compressed byte count mismatch")
    observed_bundle_bytes = sum(path.stat().st_size for path in bundle.iterdir() if path.is_file())
    if descriptor.get("bundle_bytes") != observed_bundle_bytes:
        raise RuntimeError("semantic journal bundle byte count mismatch")

    expected_count = descriptor.get("event_count")
    expected_uncompressed = descriptor.get("uncompressed_bytes")
    expected_stream_counts = descriptor.get("stream_counts")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise ValueError("semantic journal event count is invalid")
    if isinstance(expected_uncompressed, bool) or not isinstance(expected_uncompressed, int):
        raise ValueError("semantic journal uncompressed byte count is invalid")
    if not isinstance(expected_stream_counts, dict) or set(expected_stream_counts) != set(
        SEMANTIC_STREAM_NAMES
    ):
        raise ValueError("semantic journal stream counts are invalid")

    observed_stream_counts = {name: 0 for name in SEMANTIC_STREAM_NAMES}
    observed_count = 0
    observed_uncompressed = 0
    previous_runtime_event_id = 0
    final_stream_name: str | None = None
    current_batch = next(ledger, None)
    current_batch_rows = 0
    current_batch_bytes = 0
    current_batch_digest = hashlib.sha256()
    observed_semantic_digest = hashlib.sha256()
    with gzip.open(journal_path, "rb") as source:
        for line in source:
            if current_batch is None:
                raise RuntimeError("semantic journal has events beyond its batch ledger")
            current_batch_rows += 1
            current_batch_bytes += len(line)
            current_batch_digest.update(line)
            observed_semantic_digest.update(line)
            observed_uncompressed += len(line)
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("semantic journal contains invalid JSONL") from error
            if not isinstance(event, dict):
                raise RuntimeError("semantic journal event is not an object")
            stream_name = event.get("semantic_stream")
            if not isinstance(stream_name, str) or stream_name not in observed_stream_counts:
                raise RuntimeError("semantic journal event names an unknown stream")
            if event.get("semantic_sequence") != observed_count:
                raise RuntimeError("semantic journal sequence is not contiguous")
            if event.get("semantic_event_id") != observed_count + 1:
                raise RuntimeError("semantic journal event IDs are not contiguous")
            runtime_event_id = event.get("runtime_event_id")
            if (
                isinstance(runtime_event_id, bool)
                or not isinstance(runtime_event_id, int)
                or runtime_event_id <= previous_runtime_event_id
            ):
                raise RuntimeError("semantic journal runtime event IDs are not ordered")
            if event.get("stream_ordinal") != observed_stream_counts[stream_name]:
                raise RuntimeError("semantic journal stream ordinal is not contiguous")
            previous_runtime_event_id = runtime_event_id
            final_stream_name = stream_name
            observed_stream_counts[stream_name] += 1
            observed_count += 1
            if current_batch_rows == current_batch["row_count"]:
                if (
                    current_batch_bytes != current_batch["payload_bytes"]
                    or current_batch_digest.hexdigest() != current_batch["sha256"]
                ):
                    raise RuntimeError("semantic journal batch ledger digest mismatch")
                current_batch = next(ledger, None)
                current_batch_rows = 0
                current_batch_bytes = 0
                current_batch_digest = hashlib.sha256()
            yield event
    if current_batch is not None or current_batch_rows != 0:
        raise RuntimeError("semantic journal batch ledger is incomplete")
    if observed_count != expected_count:
        raise RuntimeError("semantic journal event count mismatch")
    if observed_uncompressed != expected_uncompressed:
        raise RuntimeError("semantic journal uncompressed byte count mismatch")
    if observed_stream_counts != expected_stream_counts:
        raise RuntimeError("semantic journal stream counts mismatch")
    if observed_stream_counts["termination"] != 1 or final_stream_name != "termination":
        raise RuntimeError("semantic journal termination is not final and unique")
    if terminal_semantic_digest is not None and (
        terminal_semantic_count != observed_count
        or terminal_semantic_digest != observed_semantic_digest.hexdigest()
    ):
        raise RuntimeError("terminal runtime spool does not match the semantic journal")


def iter_verified_native_control_events(
    axis_path: Path,
    journal_descriptor: Mapping[str, object],
) -> Iterator[dict[str, object]]:
    """Verify and stream the raw decoded full-native control journal."""

    raw_descriptor = journal_descriptor.get("native_control_events")
    if not isinstance(raw_descriptor, Mapping) or set(raw_descriptor) != {
        "schema_version",
        "path",
        "sidecar_path",
        "compressed_sha256",
        "rows_sha256",
        "native_source_sha256",
        "event_count",
        "compressed_bytes",
        "sidecar_bytes",
    }:
        raise ValueError("native control journal descriptor is missing or invalid")
    if (
        raw_descriptor.get("schema_version") != "stage05.2-native-control-raw-v1"
        or raw_descriptor.get("path") != "native-control-events.jsonl.gz"
        or raw_descriptor.get("sidecar_path") != "native-control-events.sha256"
        or raw_descriptor.get("sidecar_bytes") != 65
    ):
        raise ValueError("native control journal identity is invalid")
    expected_compressed = raw_descriptor.get("compressed_sha256")
    expected_rows = raw_descriptor.get("rows_sha256")
    expected_source = raw_descriptor.get("native_source_sha256")
    expected_count = raw_descriptor.get("event_count")
    expected_bytes = raw_descriptor.get("compressed_bytes")
    for digest in (expected_compressed, expected_rows, expected_source):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("native control journal digest is invalid")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
    ):
        raise ValueError("native control journal extent is invalid")
    bundle = semantic_bundle_path(axis_path, journal_descriptor)
    path = bundle / "native-control-events.jsonl.gz"
    sidecar = bundle / "native-control-events.sha256"
    if (
        path.is_symlink()
        or sidecar.is_symlink()
        or not path.is_file()
        or not sidecar.is_file()
        or sidecar.read_text(encoding="ascii").strip() != expected_compressed
        or path.stat().st_size != expected_bytes
    ):
        raise ValueError("native control journal files are invalid")
    compressed_digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            compressed_digest.update(chunk)
    if compressed_digest.hexdigest() != expected_compressed:
        raise RuntimeError("native control compressed SHA-256 mismatch")
    rows_digest = hashlib.sha256(_NATIVE_CONTROL_RAW_DOMAIN)
    observed_count = 0
    with gzip.open(path, "rb") as source:
        for line in source:
            rows_digest.update(line)
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("native control journal is invalid JSONL") from error
            if not isinstance(event, dict):
                raise RuntimeError("native control journal row is invalid")
            observed_count += 1
            yield event
    if observed_count != expected_count or rows_digest.hexdigest() != expected_rows:
        raise RuntimeError("native control row digest or count mismatch")


__all__ = [
    "SCHEMA_VERSION",
    "SEMANTIC_STREAM_NAMES",
    "canonical_trace_event",
    "evidence_json_value",
    "iter_canonical_semantic_events",
    "iter_verified_native_control_events",
    "iter_verified_semantic_journal",
    "semantic_bundle_path",
    "write_semantic_journal",
]
