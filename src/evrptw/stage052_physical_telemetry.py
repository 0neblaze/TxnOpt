"""Typed physical scheduling telemetry for Stage 5.2 evidence bundles.

Physical worker completion order is intentionally excluded from canonical
semantic hashes because it varies with scheduling. This module preserves that
order in a separately signed typed SoA receipt for independent audit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import BinaryIO

from evrptw.measurement import canonical_route_key

SCHEMA_VERSION = "stage05.2-physical-telemetry-v2"
_DOMAIN = b"stage05.2-physical-batch-receipt-v2"
_GLOBAL_DOMAIN = b"stage05.2-physical-global-receipt-v2"
_I64 = struct.Struct("<q")
_FILES = {
    "batch_offsets": ("batch_offsets.i64le", "int64-le", 8),
    "route_offsets": ("route_offsets.i64le", "int64-le", 8),
    "task_offsets": ("task_offsets.i64le", "int64-le", 8),
    "runtime_event_ids": ("runtime_event_ids.i64le", "int64-le", 8),
    "transaction_ids": ("transaction_ids.i64le", "int64-le", 8),
    "iterations": ("iterations.i64le", "int64-le", 8),
    "context_sha256": ("context_sha256.u8", "uint8", 1),
    "route_key_sha256": ("route_key_sha256.u8", "uint8", 1),
    "submission_order": ("submission_order.i64le", "int64-le", 8),
    "completion_order": ("completion_order.i64le", "int64-le", 8),
    "merge_order": ("merge_order.i64le", "int64-le", 8),
    "semantic_completion_order": ("semantic_completion_order.i64le", "int64-le", 8),
    "observation_codes": ("observation_codes.i64le", "int64-le", 8),
    "task_worker_ids": ("task_worker_ids.i64le", "int64-le", 8),
    "task_first_indices": ("task_first_indices.i64le", "int64-le", 8),
    "task_last_indices": ("task_last_indices.i64le", "int64-le", 8),
    "task_submitted_ns": ("task_submitted_ns.i64le", "int64-le", 8),
    "task_started_ns": ("task_started_ns.i64le", "int64-le", 8),
    "task_completed_ns": ("task_completed_ns.i64le", "int64-le", 8),
    "batch_receipt_sha256": ("batch_receipt_sha256.u8", "uint8", 1),
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def validate_native_work_task_receipt_stream(
    descriptor: object,
    *,
    expected_path: Path,
    completed_tasks: int,
    worker_threads: int,
) -> dict[str, object]:
    """Validate, sign and return one bounded native work-task stream.

    This is the producer-side boundary shared by local and host native pools.
    The campaign reviewer deliberately reimplements the replay independently.
    """

    fields = {
        "schema_version",
        "path",
        "sha256",
        "bytes",
        "count",
        "storage_model",
        "receipt_batch_capacity",
        "queue_bound_batches",
        "peak_queued_batches",
        "submitted_batches",
        "completed_batches",
        "dropped_count",
        "producer_wait_seconds",
        "writer_wall_seconds",
        "writer_cpu_seconds",
        "serialization_seconds",
        "write_seconds",
        "file_fsync_seconds",
        "atomic_publish_seconds",
        "parent_fsync_seconds",
    }
    if not isinstance(descriptor, dict) or set(descriptor) != fields:
        raise RuntimeError("native work-task receipt descriptor is invalid")

    def nonnegative_integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"native work-task receipt {name} is invalid")
        return value

    path = expected_path.resolve()
    if (
        descriptor.get("schema_version") != "stage05.2-native-work-task-receipts-v3"
        or descriptor.get("storage_model") != "bounded_async_fifo_stream"
        or descriptor.get("path") != path.name
        or path.is_symlink()
        or not path.is_file()
        or isinstance(completed_tasks, bool)
        or not isinstance(completed_tasks, int)
        or completed_tasks < 0
        or isinstance(worker_threads, bool)
        or not isinstance(worker_threads, int)
        or worker_threads <= 0
    ):
        raise RuntimeError("native work-task receipt identity is invalid")
    expected_bytes = nonnegative_integer(descriptor.get("bytes"), "bytes")
    expected_count = nonnegative_integer(descriptor.get("count"), "count")
    batch_capacity = nonnegative_integer(descriptor.get("receipt_batch_capacity"), "batch capacity")
    queue_bound = nonnegative_integer(descriptor.get("queue_bound_batches"), "queue bound")
    peak_queued = nonnegative_integer(descriptor.get("peak_queued_batches"), "peak queue")
    submitted_batches = nonnegative_integer(
        descriptor.get("submitted_batches"), "submitted batches"
    )
    completed_batches = nonnegative_integer(
        descriptor.get("completed_batches"), "completed batches"
    )
    dropped = nonnegative_integer(descriptor.get("dropped_count"), "dropped count")
    for name in (
        "producer_wait_seconds",
        "writer_wall_seconds",
        "writer_cpu_seconds",
        "serialization_seconds",
        "write_seconds",
        "file_fsync_seconds",
        "atomic_publish_seconds",
        "parent_fsync_seconds",
    ):
        value = descriptor.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise RuntimeError("native work-task receipt timing is invalid")
    expected_sha256 = descriptor.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or expected_bytes != path.stat().st_size
        or expected_count != completed_tasks
        or batch_capacity != 4_096
        or queue_bound != 1
        or peak_queued > queue_bound
        or submitted_batches != completed_batches
        or (expected_count == 0) != (completed_batches == 0)
        or dropped != 0
    ):
        raise RuntimeError("native work-task receipt counters are invalid")
    digest = hashlib.sha256()
    batch_digest = hashlib.sha256()
    sequences: set[int] = set()
    worker_indices: set[int] = set()
    observed_count = 0
    observed_batches = 0
    batch_count = 0
    batch_first: int | None = None
    batch_last: int | None = None
    trailer_seen = False
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError("native work-task receipt JSONL is invalid") from error
            if not isinstance(row, dict):
                raise RuntimeError("native work-task receipt row is invalid")
            kind = row.get("kind")
            if kind == "trailer":
                expected_trailer = {
                    "kind": "trailer",
                    "schema_version": "stage05.2-native-work-task-receipts-v3",
                    "storage_model": "bounded_async_fifo_stream",
                    "receipt_batch_capacity": batch_capacity,
                    "queue_bound_batches": queue_bound,
                    "submitted_batches": submitted_batches,
                    "completed_batches": completed_batches,
                    "task_receipt_dropped_count": dropped,
                    "completed_tasks": completed_tasks,
                    "receipt_count": expected_count,
                }
                if (
                    trailer_seen
                    or batch_count != 0
                    or observed_count != expected_count
                    or observed_batches != completed_batches
                    or row != expected_trailer
                ):
                    raise RuntimeError("native work-task receipt trailer is invalid")
                trailer_seen = True
                continue
            if trailer_seen:
                raise RuntimeError("native work-task receipt follows its trailer")
            if kind == "batch":
                if (
                    set(row)
                    != {
                        "kind",
                        "batch_ordinal",
                        "row_count",
                        "first_task_sequence",
                        "last_task_sequence",
                        "sha256",
                    }
                    or row.get("batch_ordinal") != observed_batches
                    or row.get("row_count") != batch_count
                    or row.get("first_task_sequence") != batch_first
                    or row.get("last_task_sequence") != batch_last
                    or row.get("sha256") != batch_digest.hexdigest()
                    or not 0 < batch_count <= batch_capacity
                ):
                    raise RuntimeError("native work-task batch ledger is invalid")
                observed_batches += 1
                batch_digest = hashlib.sha256()
                batch_count = 0
                batch_first = None
                batch_last = None
                continue
            if kind != "task" or set(row) != {
                "kind",
                "task_sequence",
                "worker_index",
                "first_index",
                "last_index",
                "submitted_nanoseconds",
                "started_nanoseconds",
                "completed_nanoseconds",
            }:
                raise RuntimeError("native work-task receipt fields are invalid")
            batch_digest.update(line)
            sequence = nonnegative_integer(row["task_sequence"], "task sequence")
            worker = nonnegative_integer(row["worker_index"], "worker index")
            first = nonnegative_integer(row["first_index"], "first index")
            last = nonnegative_integer(row["last_index"], "last index")
            submitted = nonnegative_integer(row["submitted_nanoseconds"], "submitted")
            started = nonnegative_integer(row["started_nanoseconds"], "started")
            completed = nonnegative_integer(row["completed_nanoseconds"], "completed")
            if (
                sequence in sequences
                or worker >= worker_threads
                or first >= last
                or not submitted <= started <= completed
            ):
                raise RuntimeError("native work-task receipt ordering is invalid")
            if batch_first is None:
                batch_first = sequence
            batch_last = sequence
            batch_count += 1
            observed_count += 1
            sequences.add(sequence)
            worker_indices.add(worker)
    if (
        digest.hexdigest() != expected_sha256
        or observed_count != expected_count
        or observed_batches != completed_batches
        or batch_count != 0
        or not trailer_seen
    ):
        raise RuntimeError("native work-task receipt digest/count differs")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError(f"native work-task receipt sidecar exists: {sidecar}")
    sidecar_bytes = (expected_sha256 + "\n").encode("ascii")
    with sidecar.open("xb") as sidecar_stream:
        sidecar_stream.write(sidecar_bytes)
        sidecar_stream.flush()
        os.fsync(sidecar_stream.fileno())
    parent_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return {
        **descriptor,
        "sidecar_path": sidecar.name,
        "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "validated_worker_indices": sorted(worker_indices),
    }


def _int(value: object, field: str, *, nullable: bool = False) -> int:
    if value is None and nullable:
        return -1
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"physical telemetry {field} must be an integer")
    if not -(1 << 63) <= value < (1 << 63):
        raise ValueError(f"physical telemetry {field} exceeds int64")
    return value


def _order(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"physical telemetry {field} must be an integer array")
    return tuple(_int(item, field) for item in value)


def _sequences(value: object) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list | tuple):
        raise ValueError("physical telemetry customer_sequences must be an array")
    output: list[tuple[str, ...]] = []
    for raw in value:
        if not isinstance(raw, list | tuple) or not all(
            isinstance(customer, str) for customer in raw
        ):
            raise ValueError("physical telemetry contains an invalid customer sequence")
        output.append(tuple(raw))
    return tuple(output)


def _physical_task_rows(value: object) -> tuple[tuple[int, ...], ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError("physical telemetry task receipts must be an array")
    output: list[tuple[int, ...]] = []
    for raw in value:
        if not isinstance(raw, list | tuple) or len(raw) != 7:
            raise ValueError("physical telemetry task receipt row is invalid")
        output.append(tuple(_int(item, "physical_task_receipts") for item in raw))
    return tuple(output)


def _batch_receipt(
    *,
    runtime_event_id: int,
    transaction_id: int,
    iteration: int,
    context_sha256: bytes,
    route_key_sha256: Sequence[bytes],
    submission_order: Sequence[int],
    completion_order: Sequence[int],
    merge_order: Sequence[int],
    semantic_completion_order: Sequence[int],
    observation_code: int,
    task_receipts: Sequence[Sequence[int]],
) -> bytes:
    digest = hashlib.sha256()
    digest.update(_DOMAIN)
    digest.update(_I64.pack(runtime_event_id))
    digest.update(_I64.pack(transaction_id))
    digest.update(_I64.pack(iteration))
    digest.update(context_sha256)
    digest.update(_I64.pack(len(route_key_sha256)))
    for route_digest in route_key_sha256:
        digest.update(route_digest)
    for values in (
        submission_order,
        completion_order,
        merge_order,
        semantic_completion_order,
    ):
        digest.update(_I64.pack(len(values)))
        for value in values:
            digest.update(_I64.pack(value))
    digest.update(_I64.pack(observation_code))
    digest.update(_I64.pack(len(task_receipts)))
    for row in task_receipts:
        for value in row:
            digest.update(_I64.pack(value))
    return digest.digest()


class PhysicalTelemetryWriter:
    """Append raw parallel-batch order into fixed typed SoA files."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._streams: dict[str, BinaryIO] = {}
        self._digests = {name: hashlib.sha256() for name in _FILES}
        self._sizes = {name: 0 for name in _FILES}
        self._batch_count = 0
        self._work_item_count = 0
        self._route_count = 0
        self._task_count = 0
        self._global = hashlib.sha256(_GLOBAL_DOMAIN)
        self._started = 0.0
        self._write_seconds = 0.0
        self._hash_seconds = 0.0
        self._fsync_seconds = 0.0

    def __enter__(self) -> PhysicalTelemetryWriter:
        self._started = time.perf_counter()
        for name, (filename, _dtype, _itemsize) in _FILES.items():
            self._streams[name] = (self._directory / filename).open("xb")
        self._append("batch_offsets", _I64.pack(0))
        self._append("route_offsets", _I64.pack(0))
        self._append("task_offsets", _I64.pack(0))
        return self

    def __exit__(self, *_: object) -> None:
        for stream in self._streams.values():
            stream.close()

    def _append(self, name: str, data: bytes) -> None:
        started = time.perf_counter()
        self._streams[name].write(data)
        self._write_seconds += time.perf_counter() - started
        started = time.perf_counter()
        self._digests[name].update(data)
        self._hash_seconds += time.perf_counter() - started
        self._sizes[name] += len(data)

    def observe(self, raw_event: Mapping[str, object]) -> None:
        if raw_event.get("event_type") != "parallel_batch":
            return
        submission = _order(raw_event.get("submission_order"), "submission_order")
        completion = _order(raw_event.get("completion_order"), "completion_order")
        merge = _order(raw_event.get("merge_order"), "merge_order")
        semantic_completion = _order(
            raw_event.get("semantic_completion_order", merge),
            "semantic_completion_order",
        )
        task_receipts = _physical_task_rows(raw_event.get("physical_task_receipts"))
        observation = raw_event.get("physical_observation")
        if observation is not None and not isinstance(observation, str):
            raise ValueError("physical telemetry observation mode is invalid")
        observation_code = {
            None: 0,
            "unavailable": 0,
            "native_worker_observed": 1,
            "precomputed_input": 2,
            "native_caller_observed": 3,
        }.get(observation)
        if observation_code is None:
            raise ValueError("physical telemetry observation mode is invalid")
        sequences = _sequences(raw_event.get("customer_sequences"))
        if (
            len(set(submission)) != len(submission)
            or merge != submission
            or sorted(completion) != sorted(submission)
            or sorted(semantic_completion) != sorted(submission)
        ):
            raise ValueError(
                "physical telemetry order is not a complete batch permutation: "
                f"submission={submission!r} completion={completion!r} merge={merge!r}"
            )
        if not (len(submission) == len(completion) == len(merge)):
            raise ValueError("physical telemetry batch arrays have inconsistent lengths")
        if len(semantic_completion) != len(submission):
            raise ValueError("physical telemetry semantic completion extent is invalid")
        if bool(task_receipts) != (observation_code != 0):
            raise ValueError("physical telemetry task observation is incomplete")
        expected_first = 0
        for task_ordinal, row in enumerate(task_receipts):
            if (
                row[0] != task_ordinal
                or row[1] < -1
                or row[2] != expected_first
                or not row[2] < row[3] <= len(submission)
                or not 0 <= row[4] <= row[5] <= row[6]
                or (observation_code == 1 and row[1] < 0)
                or (
                    observation_code == 2 and not (row[1] == -1 and row[4] == row[5] == row[6] == 0)
                )
                or (observation_code == 3 and row[1] != -1)
            ):
                raise ValueError("physical telemetry task receipt failed reconciliation")
            expected_first = row[3]
        if task_receipts and expected_first != len(submission):
            raise ValueError("physical telemetry task receipt route coverage is incomplete")
        chunk_sizes = raw_event.get("chunk_sizes")
        if chunk_sizes is not None:
            chunks = _order(chunk_sizes, "chunk_sizes")
            if len(chunks) != len(submission) or sum(chunks) != len(sequences):
                raise ValueError("physical telemetry chunks do not cover exact routes")
        runtime_event_id = _int(raw_event.get("semantic_event_id"), "runtime_event_id")
        transaction_id = _int(
            raw_event.get("runtime_native_transaction_id"),
            "transaction_id",
            nullable=True,
        )
        iteration = _int(raw_event.get("iteration"), "iteration", nullable=True)
        context = {
            "lane": raw_event.get("lane"),
            "iteration": None if iteration == -1 else iteration,
            "operator": raw_event.get("operator"),
        }
        if not isinstance(context["lane"], str) or not isinstance(context["operator"], str):
            raise ValueError("physical telemetry context is incomplete")
        context_digest = hashlib.sha256(_canonical_bytes(context)).digest()
        route_digests = tuple(
            hashlib.sha256(canonical_route_key(list(sequence)).encode("utf-8")).digest()
            for sequence in sequences
        )
        receipt = _batch_receipt(
            runtime_event_id=runtime_event_id,
            transaction_id=transaction_id,
            iteration=iteration,
            context_sha256=context_digest,
            route_key_sha256=route_digests,
            submission_order=submission,
            completion_order=completion,
            merge_order=merge,
            semantic_completion_order=semantic_completion,
            observation_code=observation_code,
            task_receipts=task_receipts,
        )
        for name, value in (
            ("runtime_event_ids", runtime_event_id),
            ("transaction_ids", transaction_id),
            ("iterations", iteration),
        ):
            self._append(name, _I64.pack(value))
        self._append("context_sha256", context_digest)
        for route_digest in route_digests:
            self._append("route_key_sha256", route_digest)
        for name, values in (
            ("submission_order", submission),
            ("completion_order", completion),
            ("merge_order", merge),
            ("semantic_completion_order", semantic_completion),
        ):
            for value in values:
                self._append(name, _I64.pack(value))
        self._append("observation_codes", _I64.pack(observation_code))
        for row in task_receipts:
            for name, value in zip(
                (
                    "task_worker_ids",
                    "task_first_indices",
                    "task_last_indices",
                    "task_submitted_ns",
                    "task_started_ns",
                    "task_completed_ns",
                ),
                row[1:],
                strict=True,
            ):
                self._append(name, _I64.pack(value))
        self._append("batch_receipt_sha256", receipt)
        self._work_item_count += len(submission)
        self._route_count += len(sequences)
        self._task_count += len(task_receipts)
        self._batch_count += 1
        self._append("batch_offsets", _I64.pack(self._work_item_count))
        self._append("route_offsets", _I64.pack(self._route_count))
        self._append("task_offsets", _I64.pack(self._task_count))
        started = time.perf_counter()
        self._global.update(receipt)
        self._hash_seconds += time.perf_counter() - started

    def finish(self) -> dict[str, object]:
        for stream in self._streams.values():
            started = time.perf_counter()
            stream.flush()
            self._write_seconds += time.perf_counter() - started
            started = time.perf_counter()
            os.fsync(stream.fileno())
            self._fsync_seconds += time.perf_counter() - started
        shapes = {
            "batch_offsets": [self._batch_count + 1],
            "route_offsets": [self._batch_count + 1],
            "task_offsets": [self._batch_count + 1],
            "runtime_event_ids": [self._batch_count],
            "transaction_ids": [self._batch_count],
            "iterations": [self._batch_count],
            "context_sha256": [self._batch_count, 32],
            "route_key_sha256": [self._route_count, 32],
            "submission_order": [self._work_item_count],
            "completion_order": [self._work_item_count],
            "merge_order": [self._work_item_count],
            "semantic_completion_order": [self._work_item_count],
            "observation_codes": [self._batch_count],
            "task_worker_ids": [self._task_count],
            "task_first_indices": [self._task_count],
            "task_last_indices": [self._task_count],
            "task_submitted_ns": [self._task_count],
            "task_started_ns": [self._task_count],
            "task_completed_ns": [self._task_count],
            "batch_receipt_sha256": [self._batch_count, 32],
        }
        files = {
            name: {
                "path": filename,
                "dtype": dtype,
                "shape": shapes[name],
                "bytes": self._sizes[name],
                "sha256": self._digests[name].hexdigest(),
            }
            for name, (filename, dtype, _itemsize) in _FILES.items()
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "batch_count": self._batch_count,
            "work_item_count": self._work_item_count,
            "route_count": self._route_count,
            "task_count": self._task_count,
            "global_receipt_sha256": self._global.hexdigest(),
            "files": files,
            "persistence": {
                "write_seconds": self._write_seconds,
                "hash_seconds": self._hash_seconds,
                "fsync_seconds": self._fsync_seconds,
                "total_seconds": time.perf_counter() - self._started,
            },
        }


def _verified_file(bundle: Path, name: str, descriptor: Mapping[str, object]) -> Path:
    filename, dtype, itemsize = _FILES[name]
    if descriptor.get("path") != filename or descriptor.get("dtype") != dtype:
        raise ValueError(f"physical telemetry {name} descriptor is invalid")
    shape = descriptor.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape
        )
    ):
        raise ValueError(f"physical telemetry {name} shape is invalid")
    expected_bytes = itemsize
    for value in shape:
        expected_bytes *= value
    if descriptor.get("bytes") != expected_bytes:
        raise ValueError(f"physical telemetry {name} byte count is invalid")
    path = bundle / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"physical telemetry {name} is not a regular file")
    digest = hashlib.sha256()
    observed = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            observed += len(chunk)
    if observed != expected_bytes or digest.hexdigest() != descriptor.get("sha256"):
        raise RuntimeError(f"physical telemetry {name} integrity mismatch")
    return path


def _validate_persistence_receipt(raw: object) -> None:
    if not isinstance(raw, dict) or set(raw) != {
        "write_seconds",
        "hash_seconds",
        "fsync_seconds",
        "total_seconds",
    }:
        raise ValueError("physical telemetry persistence receipt is invalid")
    for field_name, value in raw.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"physical telemetry persistence {field_name} is invalid")
    component_total = sum(
        float(raw[field_name]) for field_name in ("write_seconds", "hash_seconds", "fsync_seconds")
    )
    if component_total > float(raw["total_seconds"]) + 1e-6:
        raise ValueError("physical telemetry persistence timings do not reconcile")


def _read_exact(source: BinaryIO, size: int, field: str) -> bytes:
    value = source.read(size)
    if len(value) != size:
        raise RuntimeError(f"physical telemetry {field} ended early")
    return value


def iter_verified_physical_telemetry(
    bundle: Path,
    descriptor: Mapping[str, object],
) -> Iterator[dict[str, object]]:
    """Verify all typed files and yield independently checked batches."""

    if descriptor.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("physical telemetry schema is unsupported")
    _validate_persistence_receipt(descriptor.get("persistence"))
    batch_count = _int(descriptor.get("batch_count"), "batch_count")
    work_item_count = _int(descriptor.get("work_item_count"), "work_item_count")
    route_count = _int(descriptor.get("route_count"), "route_count")
    task_count = _int(descriptor.get("task_count"), "task_count")
    if batch_count < 0 or work_item_count < 0 or route_count < 0 or task_count < 0:
        raise ValueError("physical telemetry counts cannot be negative")
    raw_files = descriptor.get("files")
    if not isinstance(raw_files, dict) or set(raw_files) != set(_FILES):
        raise ValueError("physical telemetry file set is incomplete")
    paths: dict[str, Path] = {}
    for name in _FILES:
        raw = raw_files[name]
        if not isinstance(raw, dict):
            raise ValueError(f"physical telemetry {name} descriptor is invalid")
        paths[name] = _verified_file(bundle, name, raw)
    expected_shapes = {
        "batch_offsets": [batch_count + 1],
        "route_offsets": [batch_count + 1],
        "task_offsets": [batch_count + 1],
        "runtime_event_ids": [batch_count],
        "transaction_ids": [batch_count],
        "iterations": [batch_count],
        "context_sha256": [batch_count, 32],
        "route_key_sha256": [route_count, 32],
        "submission_order": [work_item_count],
        "completion_order": [work_item_count],
        "merge_order": [work_item_count],
        "semantic_completion_order": [work_item_count],
        "observation_codes": [batch_count],
        "task_worker_ids": [task_count],
        "task_first_indices": [task_count],
        "task_last_indices": [task_count],
        "task_submitted_ns": [task_count],
        "task_started_ns": [task_count],
        "task_completed_ns": [task_count],
        "batch_receipt_sha256": [batch_count, 32],
    }
    for name, shape in expected_shapes.items():
        raw = raw_files[name]
        assert isinstance(raw, dict)
        if raw.get("shape") != shape:
            raise ValueError(f"physical telemetry {name} count/shape mismatch")
    streams = {name: path.open("rb") for name, path in paths.items()}
    global_digest = hashlib.sha256(_GLOBAL_DOMAIN)
    try:
        first_offset = _I64.unpack(_read_exact(streams["batch_offsets"], 8, "batch_offsets"))[0]
        if first_offset != 0:
            raise RuntimeError("physical telemetry first batch offset is not zero")
        first_route_offset = _I64.unpack(_read_exact(streams["route_offsets"], 8, "route_offsets"))[
            0
        ]
        if first_route_offset != 0:
            raise RuntimeError("physical telemetry first route offset is not zero")
        first_task_offset = _I64.unpack(_read_exact(streams["task_offsets"], 8, "task_offsets"))[0]
        if first_task_offset != 0:
            raise RuntimeError("physical telemetry first task offset is not zero")
        previous_offset = 0
        previous_route_offset = 0
        previous_task_offset = 0
        for batch in range(batch_count):
            next_offset = _I64.unpack(_read_exact(streams["batch_offsets"], 8, "batch_offsets"))[0]
            count = next_offset - previous_offset
            if count < 0 or next_offset > work_item_count:
                raise RuntimeError("physical telemetry batch offsets are invalid")
            next_route_offset = _I64.unpack(
                _read_exact(streams["route_offsets"], 8, "route_offsets")
            )[0]
            route_count_in_batch = next_route_offset - previous_route_offset
            if route_count_in_batch < 0 or next_route_offset > route_count:
                raise RuntimeError("physical telemetry route offsets are invalid")
            next_task_offset = _I64.unpack(_read_exact(streams["task_offsets"], 8, "task_offsets"))[
                0
            ]
            task_count_in_batch = next_task_offset - previous_task_offset
            if task_count_in_batch < 0 or next_task_offset > task_count:
                raise RuntimeError("physical telemetry task offsets are invalid")
            scalars = {
                name: _I64.unpack(_read_exact(streams[name], 8, name))[0]
                for name in (
                    "runtime_event_ids",
                    "transaction_ids",
                    "iterations",
                    "observation_codes",
                )
            }
            context_digest = _read_exact(streams["context_sha256"], 32, "context_sha256")
            route_digests = tuple(
                _read_exact(streams["route_key_sha256"], 32, "route_key_sha256")
                for _ in range(route_count_in_batch)
            )
            orders: dict[str, tuple[int, ...]] = {}
            for name in (
                "submission_order",
                "completion_order",
                "merge_order",
                "semantic_completion_order",
            ):
                orders[name] = tuple(
                    _I64.unpack(_read_exact(streams[name], 8, name))[0] for _ in range(count)
                )
            if (
                len(set(orders["submission_order"])) != count
                or orders["merge_order"] != orders["submission_order"]
                or sorted(orders["completion_order"]) != sorted(orders["submission_order"])
                or sorted(orders["semantic_completion_order"]) != sorted(orders["submission_order"])
            ):
                raise RuntimeError("physical telemetry order permutation is invalid")
            task_receipts = tuple(
                (
                    task,
                    *(
                        _I64.unpack(_read_exact(streams[name], 8, name))[0]
                        for name in (
                            "task_worker_ids",
                            "task_first_indices",
                            "task_last_indices",
                            "task_submitted_ns",
                            "task_started_ns",
                            "task_completed_ns",
                        )
                    ),
                )
                for task in range(task_count_in_batch)
            )
            observation_code = scalars["observation_codes"]
            if observation_code not in {0, 1, 2, 3} or bool(task_receipts) != (
                observation_code != 0
            ):
                raise RuntimeError("physical telemetry task observation is incomplete")
            expected_first = 0
            for row in task_receipts:
                if (
                    row[2] != expected_first
                    or not row[2] < row[3] <= count
                    or not 0 <= row[4] <= row[5] <= row[6]
                    or (observation_code == 1 and row[1] < 0)
                    or (
                        observation_code == 2
                        and not (row[1] == -1 and row[4] == row[5] == row[6] == 0)
                    )
                    or (observation_code == 3 and row[1] != -1)
                ):
                    raise RuntimeError("physical telemetry task receipt is invalid")
                expected_first = row[3]
            if task_receipts and expected_first != count:
                raise RuntimeError("physical telemetry task route coverage is incomplete")
            receipt = _read_exact(streams["batch_receipt_sha256"], 32, "batch_receipt_sha256")
            replayed = _batch_receipt(
                runtime_event_id=scalars["runtime_event_ids"],
                transaction_id=scalars["transaction_ids"],
                iteration=scalars["iterations"],
                context_sha256=context_digest,
                route_key_sha256=route_digests,
                submission_order=orders["submission_order"],
                completion_order=orders["completion_order"],
                merge_order=orders["merge_order"],
                semantic_completion_order=orders["semantic_completion_order"],
                observation_code=observation_code,
                task_receipts=task_receipts,
            )
            if replayed != receipt:
                raise RuntimeError("physical telemetry batch receipt mismatch")
            global_digest.update(receipt)
            previous_offset = next_offset
            previous_route_offset = next_route_offset
            previous_task_offset = next_task_offset
            yield {
                "batch_ordinal": batch,
                "runtime_event_id": scalars["runtime_event_ids"],
                "transaction_id": scalars["transaction_ids"],
                "iteration": scalars["iterations"],
                "context_sha256": context_digest.hex(),
                "route_key_sha256": tuple(value.hex() for value in route_digests),
                **orders,
                "observation_code": observation_code,
                "physical_task_receipts": task_receipts,
                "receipt_sha256": receipt.hex(),
            }
        if previous_offset != work_item_count:
            raise RuntimeError("physical telemetry final batch offset is invalid")
        if previous_route_offset != route_count:
            raise RuntimeError("physical telemetry final route offset is invalid")
        if previous_task_offset != task_count:
            raise RuntimeError("physical telemetry final task offset is invalid")
        for name, stream in streams.items():
            if stream.read(1):
                raise RuntimeError(f"physical telemetry {name} contains trailing bytes")
    finally:
        for stream in streams.values():
            stream.close()
    if global_digest.hexdigest() != descriptor.get("global_receipt_sha256"):
        raise RuntimeError("physical telemetry global receipt mismatch")


__all__ = [
    "PhysicalTelemetryWriter",
    "SCHEMA_VERSION",
    "iter_verified_physical_telemetry",
]
