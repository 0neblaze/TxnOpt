"""Bounded-memory compressed ownership store for Stage 5.2 event streams."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import time
import zlib
from array import array
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import overload

from evrptw.stage052_writer_pipeline import BatchWriteResult, BoundedFifoWriter

_NONFINITE_KEY = "__stage052_spool_nonfinite_float__"
_FRAME = struct.Struct("<Q")
STAGE052_ATTEMPT_AUDIT_SCHEMA_VERSION = "stage05.2-event-spool-attempt-audit-v1"
STAGE052_ATTEMPT_AUDIT_MAGIC = b"S52AUD1\n"
STAGE052_ATTEMPT_AUDIT_FRAME = struct.Struct("<QQQQQ")
STAGE052_ATTEMPT_AUDIT_MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
_BLOCK_ROWS = 256


@dataclass(frozen=True, slots=True)
class Stage052AttemptAuditFrame:
    """One independently replayable physical attempt from a runtime spool."""

    attempt_ordinal: int
    batch_ordinal: int
    row_count: int
    payload_bytes: int
    raw_header: bytes
    compressed_payload: bytes
    decoded_payload: bytes


def iter_stage052_attempt_audit(path: Path) -> Iterator[Stage052AttemptAuditFrame]:
    """Validate and stream an immutable attempted-batch audit file."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("Stage 5.2 attempted audit is not a regular file")
    with path.open("rb") as source:
        if source.read(len(STAGE052_ATTEMPT_AUDIT_MAGIC)) != STAGE052_ATTEMPT_AUDIT_MAGIC:
            raise RuntimeError("Stage 5.2 attempted audit magic is invalid")
        expected_attempt = 0
        while True:
            header = source.read(STAGE052_ATTEMPT_AUDIT_FRAME.size)
            if not header:
                break
            if len(header) != STAGE052_ATTEMPT_AUDIT_FRAME.size:
                raise RuntimeError("Stage 5.2 attempted audit header is truncated")
            (
                attempt_ordinal,
                batch_ordinal,
                row_count,
                payload_bytes,
                compressed_bytes,
            ) = STAGE052_ATTEMPT_AUDIT_FRAME.unpack(header)
            if (
                attempt_ordinal != expected_attempt
                or not 0 < row_count <= _BLOCK_ROWS
                or not 0 < payload_bytes <= STAGE052_ATTEMPT_AUDIT_MAX_PAYLOAD_BYTES
                or not 0 < compressed_bytes <= STAGE052_ATTEMPT_AUDIT_MAX_PAYLOAD_BYTES
            ):
                raise RuntimeError("Stage 5.2 attempted audit extent is invalid")
            compressed = source.read(compressed_bytes)
            if len(compressed) != compressed_bytes:
                raise RuntimeError("Stage 5.2 attempted audit payload is truncated")
            decompressor = zlib.decompressobj()
            try:
                decoded = decompressor.decompress(compressed, payload_bytes + 1)
            except zlib.error as error:
                raise RuntimeError("Stage 5.2 attempted audit payload is corrupt") from error
            if (
                len(decoded) != payload_bytes
                or not decompressor.eof
                or decompressor.unused_data
                or decompressor.unconsumed_tail
            ):
                raise RuntimeError("Stage 5.2 attempted audit payload does not reconcile")
            yield Stage052AttemptAuditFrame(
                attempt_ordinal=attempt_ordinal,
                batch_ordinal=batch_ordinal,
                row_count=row_count,
                payload_bytes=payload_bytes,
                raw_header=header,
                compressed_payload=compressed,
                decoded_payload=decoded,
            )
            expected_attempt += 1


def decode_stage052_attempt_payload(
    payload: bytes,
    *,
    row_count: int,
) -> tuple[tuple[str, dict[str, object]], ...]:
    """Decode one audited batch into the logical runtime event envelopes."""

    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Stage 5.2 attempted audit is invalid JSON") from error
    if not isinstance(raw, list) or len(raw) != row_count:
        raise RuntimeError("Stage 5.2 attempted audit row count does not reconcile")
    decoded_rows: list[tuple[str, dict[str, object]]] = []
    for envelope in raw:
        if not isinstance(envelope, dict):
            raise RuntimeError("Stage 5.2 attempted audit envelope is invalid")
        decoded_rows.append(Stage052EventSpool._decode_envelope(envelope))
    return tuple(decoded_rows)


def _encode_value(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return {
            _NONFINITE_KEY: (
                "nan" if math.isnan(value) else "positive_inf" if value > 0.0 else "negative_inf"
            )
        }
    if isinstance(value, dict):
        return {str(key): _encode_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_encode_value(item) for item in value]
    return value


def _decode_value(value: object) -> object:
    if isinstance(value, dict):
        if set(value) == {_NONFINITE_KEY}:
            code = value[_NONFINITE_KEY]
            if code == "nan":
                return float("nan")
            if code == "positive_inf":
                return float("inf")
            if code == "negative_inf":
                return float("-inf")
            raise RuntimeError("Stage 5.2 event spool scalar tag is invalid")
        return {str(key): _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


class Stage052EventSpool(Sequence[tuple[str, dict[str, object]]]):
    """Single-writer replayable sequence backed by compressed binary frames."""

    def __init__(self, *, prefix: str = "stage052-events-") -> None:
        descriptor, raw_path = tempfile.mkstemp(prefix=prefix, suffix=".frames")
        audit_descriptor, audit_raw_path = tempfile.mkstemp(
            prefix=f"{prefix}audit-",
            suffix=".frames",
        )
        self._path = Path(raw_path)
        self._audit_path = Path(audit_raw_path)
        # The writer thread owns every persistent write.  Unbuffered mode keeps
        # producer-side reads/drains from implicitly flushing writer-owned data.
        self._handle = os.fdopen(descriptor, "w+b", buffering=0)
        self._audit_handle = os.fdopen(audit_descriptor, "w+b", buffering=0)
        self._audit_detached = False
        self._audit_count = 0
        self._block_offsets = array("Q")
        self._block_starts = array("Q")
        self._pending: list[dict[str, object]] = []
        self._count = 0
        self._closed = False
        self._sealed = False
        self._pipeline_receipt: dict[str, object] | None = None
        self._pipeline = BoundedFifoWriter(
            name="s52-event-write",
            batch_row_capacity=_BLOCK_ROWS,
            consumer=self._write_batch,
            finalizer=self._finalize_writer,
        )

    @property
    def path(self) -> Path:
        return self._path

    def append(self, value: tuple[str, Mapping[str, object]]) -> None:
        if self._closed or self._sealed:
            raise RuntimeError("Stage 5.2 event spool is closed")
        self._pipeline.begin_producer_turn()
        producer_started = time.perf_counter_ns()
        failed = True
        try:
            self._append(value)
            failed = False
        finally:
            try:
                self._pipeline.record_producer_interval(
                    producer_started,
                    time.perf_counter_ns(),
                )
            finally:
                self._pipeline.end_producer_turn(force_release=failed)

    def _append(self, value: tuple[str, Mapping[str, object]]) -> None:
        if self._closed or self._sealed:
            raise RuntimeError("Stage 5.2 event spool is closed")
        stream, event = value
        if not isinstance(stream, str) or not stream or not isinstance(event, Mapping):
            raise ValueError("Stage 5.2 spooled event is invalid")
        self._pending.append({"stream": stream, "event": _encode_value(dict(event))})
        self._count += 1
        if len(self._pending) >= _BLOCK_ROWS:
            self._flush_pending()

    def extend(
        self,
        values: Iterable[tuple[str, Mapping[str, object]]],
    ) -> None:
        for value in values:
            self.append(value)

    def __len__(self) -> int:
        return self._count

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        if self._sealed:
            raise RuntimeError("Stage 5.2 event spool is sealed")
        pending = self._pending
        self._pending = []
        block_start = self._count - len(pending)
        self._pipeline.submit((block_start, pending), row_count=len(pending))

    def _write_batch(
        self,
        _ordinal: int,
        row_count: int,
        raw_payload: object,
    ) -> BatchWriteResult:
        if (
            not isinstance(raw_payload, tuple)
            or len(raw_payload) != 2
            or isinstance(raw_payload[0], bool)
            or not isinstance(raw_payload[0], int)
            or not isinstance(raw_payload[1], list)
            or len(raw_payload[1]) != row_count
        ):
            raise RuntimeError("Stage 5.2 event spool writer payload is invalid")
        block_start = raw_payload[0]
        pending = raw_payload[1]
        encoded = json.dumps(
            pending,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        compressed = zlib.compress(encoded, level=1)
        self._block_offsets.append(self._handle.tell())
        self._block_starts.append(block_start)
        self._handle.write(_FRAME.pack(len(compressed)))
        self._handle.write(compressed)
        if self._audit_handle.tell() == 0:
            self._audit_handle.write(STAGE052_ATTEMPT_AUDIT_MAGIC)
        self._audit_handle.write(
            STAGE052_ATTEMPT_AUDIT_FRAME.pack(
                self._audit_count,
                _ordinal,
                row_count,
                len(encoded),
                len(compressed),
            )
        )
        self._audit_handle.write(compressed)
        self._audit_count += 1
        return BatchWriteResult(
            row_count=row_count,
            payload_bytes=len(encoded),
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    def _finalize_writer(self) -> None:
        if self._audit_handle.tell() == 0:
            self._audit_handle.write(STAGE052_ATTEMPT_AUDIT_MAGIC)
        for stream in (self._handle, self._audit_handle):
            stream.flush()
            os.fsync(stream.fileno())

    def _drain(self) -> None:
        if not self._sealed:
            self._pipeline.begin_producer_turn()
            producer_started = time.perf_counter_ns()
            try:
                self._flush_pending()
            finally:
                try:
                    self._pipeline.record_producer_interval(
                        producer_started,
                        time.perf_counter_ns(),
                    )
                finally:
                    self._pipeline.end_producer_turn(force_release=True)
            self._pipeline.drain()

    def pipeline_statistics(self) -> dict[str, object]:
        """Drain and report the live one-batch writer without sealing it."""

        if self._closed:
            raise RuntimeError("Stage 5.2 event spool is closed")
        if self._pipeline_receipt is not None:
            return dict(self._pipeline_receipt)
        self._drain()
        return self._pipeline.snapshot()

    def seal(self) -> dict[str, object]:
        """Finish the writer while retaining the replayable temporary frames."""

        if self._closed:
            raise RuntimeError("Stage 5.2 event spool is closed")
        if self._pipeline_receipt is not None:
            return dict(self._pipeline_receipt)
        self._pipeline.begin_producer_turn()
        producer_started = time.perf_counter_ns()
        try:
            self._flush_pending()
        finally:
            try:
                self._pipeline.record_producer_interval(
                    producer_started,
                    time.perf_counter_ns(),
                )
            finally:
                self._pipeline.end_producer_turn(force_release=True)
        self._pipeline_receipt = self._pipeline.finish()
        self._sealed = True
        return dict(self._pipeline_receipt)

    def detach_pipeline_evidence(self) -> tuple[dict[str, object], Path]:
        """Transfer the immutable attempted-frame audit after writer shutdown."""

        receipt = self.seal()
        if self._audit_count != receipt.get("attempted_batches"):
            raise RuntimeError("Stage 5.2 event spool attempted audit is incomplete")
        if not self._audit_detached:
            self._audit_handle.close()
            self._audit_detached = True
        if self._audit_path.is_symlink() or not self._audit_path.is_file():
            raise RuntimeError("Stage 5.2 event spool attempted audit is missing")
        if self._audit_path.stat().st_size < len(STAGE052_ATTEMPT_AUDIT_MAGIC):
            raise RuntimeError("Stage 5.2 event spool attempted audit is incomplete")
        return receipt, self._audit_path

    def _read_block(self, block_index: int) -> list[dict[str, object]]:
        self._drain()
        return self._read_block_drained(block_index)

    def _read_block_drained(self, block_index: int) -> list[dict[str, object]]:
        with self._path.open("rb") as reader:
            reader.seek(self._block_offsets[block_index])
            header = reader.read(_FRAME.size)
            if len(header) != _FRAME.size:
                raise RuntimeError("Stage 5.2 event spool frame header is corrupt")
            size = _FRAME.unpack(header)[0]
            compressed = reader.read(size)
        if len(compressed) != size:
            raise RuntimeError("Stage 5.2 event spool frame is truncated")
        try:
            raw = zlib.decompress(compressed)
        except zlib.error as error:
            raise RuntimeError("Stage 5.2 event spool frame is corrupt") from error
        try:
            block = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Stage 5.2 event spool block is corrupt") from error
        if not isinstance(block, list) or not all(isinstance(envelope, dict) for envelope in block):
            raise RuntimeError("Stage 5.2 event spool block is invalid")
        return block

    @staticmethod
    def _decode_envelope(envelope: Mapping[str, object]) -> tuple[str, dict[str, object]]:
        stream = envelope.get("stream")
        event = envelope.get("event")
        if not isinstance(stream, str) or not stream or not isinstance(event, dict):
            raise RuntimeError("Stage 5.2 event spool row is invalid")
        decoded = _decode_value(event)
        if not isinstance(decoded, dict):
            raise RuntimeError("Stage 5.2 event spool decoded row is invalid")
        return stream, decoded

    def _read_at(self, index: int) -> tuple[str, dict[str, object]]:
        if self._closed:
            raise RuntimeError("Stage 5.2 event spool is closed")
        resolved = index + len(self) if index < 0 else index
        if resolved < 0 or resolved >= len(self):
            raise IndexError("Stage 5.2 event spool index out of range")
        self._drain()
        block_index = bisect_right(self._block_starts, resolved) - 1
        if block_index < 0:
            raise RuntimeError("Stage 5.2 event spool block index is invalid")
        block = self._read_block_drained(block_index)
        ordinal = resolved - self._block_starts[block_index]
        if ordinal >= len(block):
            raise RuntimeError("Stage 5.2 event spool row offset is invalid")
        return self._decode_envelope(block[ordinal])

    @overload
    def __getitem__(self, index: int) -> tuple[str, dict[str, object]]: ...

    @overload
    def __getitem__(self, index: slice) -> list[tuple[str, dict[str, object]]]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> tuple[str, dict[str, object]] | list[tuple[str, dict[str, object]]]:
        if isinstance(index, slice):
            return [self._read_at(item) for item in range(*index.indices(len(self)))]
        return self._read_at(index)

    def __iter__(self) -> Iterator[tuple[str, dict[str, object]]]:
        if self._closed:
            raise RuntimeError("Stage 5.2 event spool is closed")
        self._drain()
        for block_index in range(len(self._block_offsets)):
            for envelope in self._read_block_drained(block_index):
                yield self._decode_envelope(envelope)

    def truncate(self, count: int) -> None:
        if isinstance(count, bool) or not 0 <= count <= len(self):
            raise ValueError("Stage 5.2 event spool checkpoint is invalid")
        if self._sealed:
            raise RuntimeError("Stage 5.2 event spool is sealed")
        if count == len(self):
            return
        self._drain()
        self._pipeline.begin_producer_turn()
        producer_started = time.perf_counter_ns()
        retained_batches = 0
        failed = True
        try:
            retained_batches = self._truncate_drained(count)
            failed = False
        finally:
            try:
                self._pipeline.record_producer_interval(
                    producer_started,
                    time.perf_counter_ns(),
                )
            finally:
                self._pipeline.end_producer_turn(force_release=True)
        if not failed:
            self._pipeline.rewind(retained_batches)

    def _truncate_drained(self, count: int) -> int:
        if count == 0:
            self._handle.truncate(0)
            self._handle.seek(0)
            self._block_offsets = array("Q")
            self._block_starts = array("Q")
            self._count = 0
            return 0
        block_index = bisect_right(self._block_starts, count) - 1
        if block_index < 0:
            raise RuntimeError("Stage 5.2 event spool truncate block is invalid")
        if self._block_starts[block_index] == count:
            offset = self._block_offsets[block_index]
            self._handle.truncate(offset)
            self._handle.seek(offset)
            del self._block_offsets[block_index:]
            del self._block_starts[block_index:]
            self._count = count
            return block_index
        block = self._read_block_drained(block_index)
        keep = count - self._block_starts[block_index]
        offset = self._block_offsets[block_index]
        self._handle.truncate(offset)
        self._handle.seek(offset)
        del self._block_offsets[block_index:]
        del self._block_starts[block_index:]
        self._pending = block[:keep]
        self._count = count
        return block_index

    def close(self) -> None:
        if self._closed:
            return
        failure: BaseException | None = None
        try:
            if self._pipeline_receipt is None:
                try:
                    self.seal()
                except BaseException as error:
                    failure = error
        finally:
            self._closed = True
            try:
                self._handle.close()
            finally:
                self._path.unlink(missing_ok=True)
                if not self._audit_handle.closed:
                    self._audit_handle.close()
                if not self._audit_detached:
                    self._audit_path.unlink(missing_ok=True)
        if failure is not None:
            raise failure

    def __enter__(self) -> Stage052EventSpool:
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


__all__ = [
    "STAGE052_ATTEMPT_AUDIT_FRAME",
    "STAGE052_ATTEMPT_AUDIT_MAGIC",
    "STAGE052_ATTEMPT_AUDIT_MAX_PAYLOAD_BYTES",
    "STAGE052_ATTEMPT_AUDIT_SCHEMA_VERSION",
    "Stage052AttemptAuditFrame",
    "Stage052EventSpool",
    "decode_stage052_attempt_payload",
    "iter_stage052_attempt_audit",
]
