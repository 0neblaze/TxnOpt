"""One-batch bounded FIFO writer used by Stage 5.2 trace persistence."""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BatchWriteResult:
    """Writer-owned digest receipt for one completed logical batch."""

    row_count: int
    payload_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.row_count, bool)
            or not isinstance(self.row_count, int)
            or self.row_count <= 0
            or isinstance(self.payload_bytes, bool)
            or not isinstance(self.payload_bytes, int)
            or self.payload_bytes <= 0
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("bounded writer batch receipt is invalid")


@dataclass(frozen=True, slots=True)
class _QueuedBatch:
    ordinal: int
    attempt_ordinal: int
    row_count: int
    payload: object


class BoundedFifoWriter:
    """Run one non-daemon writer with a hard one-batch queue bound.

    The caller owns logical batching and records producer intervals. The writer
    owns serialization/I/O through ``consumer`` and final draining through
    ``finalizer``. Any writer exception is re-raised by the current producer or
    drain operation; synchronous fallback is intentionally unavailable.
    """

    queue_bound_batches = 1

    def __init__(
        self,
        *,
        name: str,
        batch_row_capacity: int,
        consumer: Callable[[int, int, object], BatchWriteResult],
        finalizer: Callable[[], None],
    ) -> None:
        if (
            not name
            or isinstance(batch_row_capacity, bool)
            or not isinstance(batch_row_capacity, int)
            or batch_row_capacity <= 0
        ):
            raise ValueError("bounded writer configuration is invalid")
        self._batch_row_capacity = batch_row_capacity
        self._consumer = consumer
        self._finalizer = finalizer
        self._queue: queue.Queue[_QueuedBatch | None] = queue.Queue(
            maxsize=self.queue_bound_batches
        )
        self._condition = threading.Condition()
        self._write_turn = threading.Lock()
        self._producer_turn_thread_id: int | None = None
        self._writer_error: BaseException | None = None
        self._attempted_batches = 0
        self._submitted_batches = 0
        self._completed_batches = 0
        self._discarded_batches = 0
        self._discarded_rows = 0
        self._rewind_operations = 0
        self._peak_queued_batches = 0
        self._producer_wait_seconds = 0.0
        self._producer_wall_seconds = 0.0
        self._writer_wall_seconds = 0.0
        self._writer_cpu_seconds = 0.0
        self._union_seconds = 0.0
        self._ledger: list[dict[str, object]] = []
        self._attempted_ledger: list[dict[str, object]] = []
        self._finished_receipt: dict[str, object] | None = None
        self._stopping = False
        self._thread = threading.Thread(
            target=self._writer_loop,
            name=name,
            daemon=False,
        )
        self._thread.start()

    @property
    def batch_row_capacity(self) -> int:
        return self._batch_row_capacity

    @property
    def writer_alive(self) -> bool:
        return self._thread.is_alive()

    def begin_producer_turn(self) -> None:
        """Own the shard turn before executing one solver callback."""

        thread_id = threading.get_ident()
        wait_started = time.perf_counter_ns()
        with self._condition:
            if self._producer_turn_thread_id is not None:
                if self._producer_turn_thread_id != thread_id:
                    raise RuntimeError("bounded FIFO producer thread changed")
                return
        while True:
            with self._condition:
                self._raise_writer_error_locked()
                if self._stopping:
                    raise RuntimeError("bounded FIFO writer is stopping")
                outstanding = self._submitted_batches != self._completed_batches
                if outstanding:
                    self._condition.wait(timeout=0.05)
                    continue
            self._write_turn.acquire()
            with self._condition:
                self._raise_writer_error_locked()
                if self._stopping:
                    self._write_turn.release()
                    raise RuntimeError("bounded FIFO writer is stopping")
                if self._submitted_batches == self._completed_batches:
                    self._producer_turn_thread_id = thread_id
                    self._producer_wait_seconds += (
                        time.perf_counter_ns() - wait_started
                    ) / 1_000_000_000.0
                    return
            self._write_turn.release()

    def _release_producer_turn(self, *, require_owner: bool) -> None:
        owner = self._producer_turn_thread_id
        if owner is None:
            return
        if owner != threading.get_ident():
            if require_owner:
                raise RuntimeError("bounded FIFO producer turn belongs to another thread")
            return
        self._producer_turn_thread_id = None
        self._write_turn.release()

    def end_producer_turn(self, *, force_release: bool = False) -> None:
        """Finish one callback and hand a submitted batch to the writer."""

        with self._condition:
            if self._producer_turn_thread_id != threading.get_ident():
                raise RuntimeError("bounded FIFO producer does not own the shard turn")
            outstanding = self._submitted_batches != self._completed_batches
        if force_release or outstanding:
            self._release_producer_turn(require_owner=True)

    def record_producer_interval(self, started_ns: int, completed_ns: int) -> None:
        """Record one producer callback/finish interval for union accounting."""

        if (
            isinstance(started_ns, bool)
            or not isinstance(started_ns, int)
            or isinstance(completed_ns, bool)
            or not isinstance(completed_ns, int)
            or min(started_ns, completed_ns) < 0
            or completed_ns < started_ns
        ):
            raise ValueError("bounded writer producer interval is invalid")
        with self._condition:
            if self._producer_turn_thread_id != threading.get_ident():
                raise RuntimeError("bounded writer producer interval requires the shard turn")
            elapsed = (completed_ns - started_ns) / 1_000_000_000.0
            self._producer_wall_seconds += elapsed
            # Producer and writer work are serialized by ``_write_turn``.  This
            # O(1) accounting therefore remains exact without retaining an
            # unbounded interval list or assuming cross-thread completion order.
            self._union_seconds += elapsed

    def _raise_writer_error_locked(self) -> None:
        if self._writer_error is not None:
            raise self._writer_error

    def submit(self, payload: object, *, row_count: int) -> None:
        """Submit exactly one bounded logical batch, waiting without fallback."""

        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or not 0 < row_count <= self._batch_row_capacity
        ):
            raise ValueError("bounded writer row count is invalid")
        with self._condition:
            self._raise_writer_error_locked()
            if self._stopping:
                raise RuntimeError("bounded FIFO writer is stopping")
            if self._producer_turn_thread_id != threading.get_ident():
                raise RuntimeError("bounded FIFO submit requires the producer turn")
            if self._submitted_batches != self._completed_batches:
                raise RuntimeError("one producer turn may submit only one callback batch")
            ordinal = self._submitted_batches
        queued = _QueuedBatch(ordinal, self._attempted_batches, row_count, payload)
        wait_started = time.perf_counter_ns()
        while True:
            with self._condition:
                self._raise_writer_error_locked()
                try:
                    self._queue.put_nowait(queued)
                except queue.Full:
                    self._condition.wait(timeout=0.05)
                    continue
                # Successful ``put`` is the linearization point.  The writer
                # cannot reconcile completion before this condition is released.
                self._attempted_batches += 1
                self._submitted_batches += 1
                self._peak_queued_batches = 1
                break
        waited = (time.perf_counter_ns() - wait_started) / 1_000_000_000.0
        with self._condition:
            self._producer_wait_seconds += waited

    def drain(self) -> None:
        """Wait until every submitted batch is complete and rethrow failures."""

        self._release_producer_turn(require_owner=True)
        wait_started = time.perf_counter_ns()
        with self._condition:
            while self._completed_batches < self._submitted_batches and self._writer_error is None:
                self._condition.wait(timeout=0.05)
            self._producer_wait_seconds += (time.perf_counter_ns() - wait_started) / 1_000_000_000.0
            self._raise_writer_error_locked()

    def rewind(self, retained_batches: int) -> None:
        """Discard drained tail batches after an atomic trace rollback.

        Writer time remains attributed, while explicit discarded counters make
        the physical work observable.  Retained ordinals are then reusable so
        the terminal ledger exactly replays the surviving logical frames.
        """

        if (
            isinstance(retained_batches, bool)
            or not isinstance(retained_batches, int)
            or retained_batches < 0
        ):
            raise ValueError("bounded FIFO retained batch count is invalid")
        self.drain()
        with self._write_turn, self._condition:
            self._raise_writer_error_locked()
            if retained_batches > self._completed_batches:
                raise ValueError("bounded FIFO rewind exceeds completed batches")
            discarded = self._completed_batches - retained_batches
            if discarded == 0:
                return
            discarded_rows = 0
            for row in self._ledger[retained_batches:]:
                row_count = row.get("row_count")
                if isinstance(row_count, bool) or not isinstance(row_count, int):
                    raise RuntimeError("bounded FIFO ledger row count is invalid")
                discarded_rows += row_count
            rewind_ordinal = self._rewind_operations + 1
            marked = 0
            for attempt in self._attempted_ledger:
                batch_ordinal = attempt.get("batch_ordinal")
                if (
                    attempt.get("status") == "retained"
                    and isinstance(batch_ordinal, int)
                    and not isinstance(batch_ordinal, bool)
                    and batch_ordinal >= retained_batches
                ):
                    attempt["status"] = "discarded"
                    attempt["rewind_ordinal"] = rewind_ordinal
                    marked += 1
            if marked != discarded:
                raise RuntimeError("bounded FIFO attempted ledger rewind is incomplete")
            del self._ledger[retained_batches:]
            self._discarded_batches += discarded
            self._discarded_rows += discarded_rows
            self._rewind_operations += 1
            self._submitted_batches = retained_batches
            self._completed_batches = retained_batches

    def snapshot(self) -> dict[str, object]:
        """Return a drained non-terminal receipt for diagnostics/tests."""

        self.drain()
        with self._condition:
            return self._receipt_locked(finalized=False)

    def finish(self) -> dict[str, object]:
        """Drain, finalize in the writer, join, and return a terminal receipt."""

        with self._condition:
            if self._finished_receipt is not None:
                return dict(self._finished_receipt)
            self._stopping = True
        try:
            self.drain()
            self._queue.put(None)
            self._thread.join()
            with self._condition:
                self._raise_writer_error_locked()
                if self._completed_batches != self._submitted_batches:
                    raise RuntimeError("bounded FIFO writer did not drain")
                self._finished_receipt = self._receipt_locked(finalized=True)
                return dict(self._finished_receipt)
        except BaseException:
            if self._thread.is_alive():
                with suppress(queue.Full):
                    self._queue.put_nowait(None)
                self._thread.join(timeout=5.0)
            raise

    def _receipt_locked(self, *, finalized: bool) -> dict[str, object]:
        return {
            "schema_version": "stage05.2-bounded-fifo-writer-v1",
            "storage_model": "one_batch_bounded_non_daemon_fifo",
            "writer_thread_count": 1,
            "writer_daemon": self._thread.daemon,
            "finalized": finalized,
            "batch_row_capacity": self._batch_row_capacity,
            "queue_bound_batches": self.queue_bound_batches,
            "peak_queued_batches": self._peak_queued_batches,
            "attempted_batches": self._attempted_batches,
            "submitted_batches": self._submitted_batches,
            "completed_batches": self._completed_batches,
            "discarded_batches": self._discarded_batches,
            "discarded_rows": self._discarded_rows,
            "rewind_operations": self._rewind_operations,
            "producer_wall_seconds": self._producer_wall_seconds,
            "writer_wall_seconds": self._writer_wall_seconds,
            "writer_cpu_seconds": self._writer_cpu_seconds,
            "producer_wait_seconds": self._producer_wait_seconds,
            "producer_writer_wall_union_seconds": self._union_seconds,
            "batch_ledger": [dict(row) for row in self._ledger],
            "attempted_batch_ledger": [dict(row) for row in self._attempted_ledger],
        }

    def _writer_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    self._queue.task_done()
                    break
                with self._write_turn:
                    wall_started = time.perf_counter_ns()
                    cpu_started = time.thread_time_ns()
                    try:
                        result = self._consumer(item.ordinal, item.row_count, item.payload)
                        if result.row_count != item.row_count:
                            raise RuntimeError("bounded writer consumer changed its row count")
                    finally:
                        cpu_completed = time.thread_time_ns()
                        wall_completed = time.perf_counter_ns()
                with self._condition:
                    elapsed = (wall_completed - wall_started) / 1_000_000_000.0
                    self._writer_wall_seconds += elapsed
                    self._writer_cpu_seconds += (cpu_completed - cpu_started) / 1_000_000_000.0
                    self._union_seconds += elapsed
                    active_row = {
                        "batch_ordinal": item.ordinal,
                        "row_count": result.row_count,
                        "payload_bytes": result.payload_bytes,
                        "sha256": result.sha256,
                    }
                    self._ledger.append(active_row)
                    self._attempted_ledger.append(
                        {
                            "attempt_ordinal": item.attempt_ordinal,
                            **active_row,
                            "status": "retained",
                            "rewind_ordinal": None,
                        }
                    )
                    self._completed_batches += 1
                    self._condition.notify_all()
                self._queue.task_done()
            with self._write_turn:
                wall_started = time.perf_counter_ns()
                cpu_started = time.thread_time_ns()
                try:
                    self._finalizer()
                finally:
                    cpu_completed = time.thread_time_ns()
                    wall_completed = time.perf_counter_ns()
            with self._condition:
                elapsed = (wall_completed - wall_started) / 1_000_000_000.0
                self._writer_wall_seconds += elapsed
                self._writer_cpu_seconds += (cpu_completed - cpu_started) / 1_000_000_000.0
                self._union_seconds += elapsed
                self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._writer_error = error
                self._condition.notify_all()
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()


def validate_pipeline_receipt(
    receipt: object,
    *,
    require_finalized: bool,
) -> tuple[Mapping[str, object], ...]:
    """Validate generic counters/timings and return the ordered batch ledger."""

    if not isinstance(receipt, Mapping):
        raise ValueError("bounded writer receipt must be an object")
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
        "batch_ledger",
        "attempted_batch_ledger",
    }
    if (
        set(receipt) != expected
        or receipt.get("schema_version") != "stage05.2-bounded-fifo-writer-v1"
        or receipt.get("storage_model") != "one_batch_bounded_non_daemon_fifo"
        or receipt.get("writer_thread_count") != 1
        or receipt.get("writer_daemon") is not False
        or receipt.get("finalized") is not require_finalized
        or receipt.get("queue_bound_batches") != 1
    ):
        raise ValueError("bounded writer receipt identity is invalid")

    def integer(field: str) -> int:
        value = receipt.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"bounded writer {field} is invalid")
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
        capacity == 0
        or peak > 1
        or peak != int(attempted > 0)
        or submitted != completed
        or attempted != completed + discarded
        or (discarded == 0) != (rewinds == 0)
        or (discarded == 0) != (discarded_rows == 0)
        or rewinds > discarded
    ):
        raise ValueError("bounded writer counters are invalid")
    for field in (
        "producer_wall_seconds",
        "writer_wall_seconds",
        "writer_cpu_seconds",
        "producer_wait_seconds",
        "producer_writer_wall_union_seconds",
    ):
        value = receipt.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("bounded writer timing is invalid")
    union = float(receipt["producer_writer_wall_union_seconds"])
    serialized_total = float(receipt["producer_wall_seconds"]) + float(
        receipt["writer_wall_seconds"]
    )
    if not math.isclose(union, serialized_total, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("bounded writer wall-time union is invalid")
    raw_ledger = receipt.get("batch_ledger")
    if not isinstance(raw_ledger, list) or len(raw_ledger) != completed:
        raise ValueError("bounded writer batch ledger is incomplete")
    ledger: list[Mapping[str, object]] = []
    for ordinal, raw in enumerate(raw_ledger):
        if not isinstance(raw, Mapping) or set(raw) != {
            "batch_ordinal",
            "row_count",
            "payload_bytes",
            "sha256",
        }:
            raise ValueError("bounded writer batch ledger row is invalid")
        if raw.get("batch_ordinal") != ordinal:
            raise ValueError("bounded writer batch ledger order is invalid")
        row_count = raw.get("row_count")
        payload_bytes = raw.get("payload_bytes")
        digest = raw.get("sha256")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or not 0 < row_count <= capacity
            or isinstance(payload_bytes, bool)
            or not isinstance(payload_bytes, int)
            or payload_bytes <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("bounded writer batch ledger values are invalid")
        ledger.append(raw)
    raw_attempted = receipt.get("attempted_batch_ledger")
    if not isinstance(raw_attempted, list) or len(raw_attempted) != attempted:
        raise ValueError("bounded writer attempted batch ledger is incomplete")
    retained_attempts: list[dict[str, object]] = []
    observed_discarded = 0
    observed_discarded_rows = 0
    observed_rewinds: set[int] = set()
    for attempt_ordinal, raw in enumerate(raw_attempted):
        if not isinstance(raw, Mapping) or set(raw) != {
            "attempt_ordinal",
            "batch_ordinal",
            "row_count",
            "payload_bytes",
            "sha256",
            "status",
            "rewind_ordinal",
        }:
            raise ValueError("bounded writer attempted ledger row is invalid")
        batch_ordinal = raw.get("batch_ordinal")
        row_count = raw.get("row_count")
        payload_bytes = raw.get("payload_bytes")
        digest = raw.get("sha256")
        status = raw.get("status")
        rewind_ordinal = raw.get("rewind_ordinal")
        if (
            raw.get("attempt_ordinal") != attempt_ordinal
            or isinstance(batch_ordinal, bool)
            or not isinstance(batch_ordinal, int)
            or batch_ordinal < 0
            or isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or not 0 < row_count <= capacity
            or isinstance(payload_bytes, bool)
            or not isinstance(payload_bytes, int)
            or payload_bytes <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or status not in {"retained", "discarded"}
        ):
            raise ValueError("bounded writer attempted ledger values are invalid")
        core = {
            "batch_ordinal": batch_ordinal,
            "row_count": row_count,
            "payload_bytes": payload_bytes,
            "sha256": digest,
        }
        if status == "retained":
            if rewind_ordinal is not None:
                raise ValueError("retained bounded writer attempt names a rewind")
            retained_attempts.append(core)
        else:
            if (
                isinstance(rewind_ordinal, bool)
                or not isinstance(rewind_ordinal, int)
                or not 1 <= rewind_ordinal <= rewinds
            ):
                raise ValueError("discarded bounded writer attempt lacks its rewind")
            observed_discarded += 1
            observed_discarded_rows += row_count
            observed_rewinds.add(rewind_ordinal)
    if (
        retained_attempts != [dict(row) for row in ledger]
        or observed_discarded != discarded
        or observed_discarded_rows != discarded_rows
        or observed_rewinds != set(range(1, rewinds + 1))
    ):
        raise ValueError("bounded writer attempted and active ledgers do not reconcile")
    return tuple(ledger)
