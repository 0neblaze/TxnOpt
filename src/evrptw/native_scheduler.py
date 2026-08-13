"""Host-wide Stage 5.2 scheduler control and shared-memory transport."""

from __future__ import annotations

import json
import math
import os
import secrets
import socket
import struct
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy.typing as npt
import psutil  # type: ignore[import-untyped]

from evrptw.runtime_envelope import TERMINAL_PROCESS_IO_RECEIPT_SCHEMA_VERSION
from evrptw.stage052_physical_telemetry import (
    validate_native_work_task_receipt_stream,
)

_READY_TIMEOUT_SECONDS = 10.0
_NUMERIC_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


_CONTROL_FRAME = struct.Struct("<QIIQQ128s64s192s")
_KERNEL_MAGIC = 0x4556525054574B33
_KERNEL_PROTOCOL_VERSION = 3
_SHUTDOWN_MESSAGE = 6
_RELEASED_MESSAGE = 4


@dataclass(slots=True)
class NativeHostScheduler:
    """One temporary scheduler with a shared compute pool and request workers."""

    socket_path: Path
    worker_threads: int = 24
    enable_fault_injection: bool = False
    production_fault: str | None = None
    request_threads: int = 6
    cpu_affinity: tuple[int, ...] | None = None
    task_receipt_path: Path | None = None
    _process: subprocess.Popen[bytes] | None = None
    _run_nonce: str | None = None
    _runtime_statistics: dict[str, object] | None = None
    _task_receipt_output_path: Path | None = None
    _task_started_monotonic: float | None = None
    _process_create_time: float | None = None
    _process_start_time_ticks: int | None = None
    _terminal_process_io_receipt: dict[str, object] | None = None

    @staticmethod
    def _process_start_ticks(pid: int) -> int:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="strict")
        _comm, suffix = raw.rsplit(")", 1)
        fields = suffix.split()
        if len(fields) <= 19:
            raise RuntimeError("native scheduler process start identity is unavailable")
        ticks = int(fields[19])
        if ticks < 0:
            raise RuntimeError("native scheduler process start identity is invalid")
        return ticks

    def _validated_cpu_affinity(self) -> tuple[int, ...] | None:
        affinity = self.cpu_affinity
        if affinity is None:
            return None
        if not isinstance(affinity, tuple) or not affinity:
            raise ValueError("native scheduler cpu_affinity must be a non-empty tuple")
        if any(isinstance(cpu, bool) or not isinstance(cpu, int) for cpu in affinity):
            raise ValueError("native scheduler cpu_affinity must contain integers")
        if any(cpu < 0 for cpu in affinity):
            raise ValueError("native scheduler cpu_affinity cannot contain negatives")
        if len(set(affinity)) != len(affinity):
            raise ValueError("native scheduler cpu_affinity cannot contain duplicates")
        return tuple(sorted(affinity))

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("host scheduler is already started")
        if (
            not isinstance(self.worker_threads, int)
            or isinstance(self.worker_threads, bool)
            or not 1 <= self.worker_threads <= 1024
        ):
            raise ValueError("native scheduler worker_threads must be in [1, 1024]")
        if (
            not isinstance(self.request_threads, int)
            or isinstance(self.request_threads, bool)
            or not 1 <= self.request_threads <= 64
        ):
            raise ValueError("native scheduler request_threads must be in [1, 64]")
        cpu_affinity = self._validated_cpu_affinity()
        if self.production_fault not in {
            None,
            "pause_before_execute",
            "initial_state_path_offset_oob",
            "exact_path_offset_oob",
            "deadline_text_screen_failure",
            "candidate_execute_output_failure",
            "candidate_commit_release_loss",
            "candidate_commit_before_apply_crash",
            "screen_response_before_local_apply_crash",
        }:
            raise ValueError("native scheduler production fault is invalid")
        if self.production_fault is not None and not self.enable_fault_injection:
            raise ValueError("native scheduler production fault requires fault injection")
        from evrptw import _core as native_core

        executable = Path(native_core.__file__).with_name("_native_host_scheduler")
        if not executable.is_file():
            raise RuntimeError("pure C++ host scheduler executable is missing")
        scheduler_environment = dict(os.environ)
        scheduler_environment.update(_NUMERIC_THREAD_ENVIRONMENT)
        run_nonce = secrets.token_hex(8)
        task_receipt_path = (
            self.task_receipt_path.resolve()
            if self.task_receipt_path is not None
            else self.socket_path.with_name(
                f"{self.socket_path.name}.{run_nonce}.task-receipts.jsonl"
            ).resolve()
        )
        if task_receipt_path.exists() or task_receipt_path.is_symlink():
            raise FileExistsError(
                f"native scheduler task-receipt target exists: {task_receipt_path}"
            )
        if not task_receipt_path.parent.is_dir():
            raise FileNotFoundError(
                f"native scheduler task-receipt parent is missing: {task_receipt_path.parent}"
            )
        command = [
            str(executable),
            str(self.socket_path),
            str(self.worker_threads),
            str(self.request_threads),
            run_nonce,
            f"--task-receipts={task_receipt_path}",
        ]
        if cpu_affinity is not None:
            command.append("--cpu-affinity=" + ",".join(str(cpu) for cpu in cpu_affinity))
        if self.enable_fault_injection:
            command.append("--enable-fault-injection")
        if self.production_fault is not None:
            command.append(f"--production-fault={self.production_fault}")
        task_started_monotonic = time.monotonic()
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=scheduler_environment,
        )
        self._process = process
        self._run_nonce = run_nonce
        self._runtime_statistics = None
        self._terminal_process_io_receipt = None
        self._task_receipt_output_path = task_receipt_path
        try:
            self._task_started_monotonic = task_started_monotonic
            self._process_create_time = float(psutil.Process(process.pid).create_time())
            self._process_start_time_ticks = self._process_start_ticks(process.pid)
        except (OSError, ValueError, psutil.Error) as error:
            self.close(force=True)
            raise RuntimeError("native scheduler process identity is unavailable") from error
        deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stderr = (process.stderr.read() if process.stderr is not None else b"").decode(
                    "utf-8", errors="replace"
                )
                self._process = None
                self.socket_path.unlink(missing_ok=True)
                self._cleanup_owned_segments(process.pid, run_nonce)
                self._run_nonce = None
                raise RuntimeError(f"host scheduler exited before becoming ready: {stderr.strip()}")
            if self.socket_path.exists():
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                        probe.connect(str(self.socket_path))
                except ConnectionRefusedError:
                    time.sleep(0.01)
                    continue
                expected_threads = self.worker_threads + self.request_threads + 2
                actual_affinity = self.observed_cpu_affinity()
                if (
                    self.observed_thread_count() == expected_threads
                    and self.observed_task_receipt_writer_thread_count() == 1
                    and (cpu_affinity is None or actual_affinity == cpu_affinity)
                ):
                    return
            time.sleep(0.01)
        self.close(force=True)
        raise RuntimeError("host scheduler did not become ready")

    def close(self, *, force: bool = False) -> None:
        process = self._process
        if process is None:
            return
        run_nonce = self._run_nonce
        shutdown_error: OSError | RuntimeError | None = None
        if process.poll() is None and force:
            process.terminate()
        elif process.poll() is None:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.connect(str(self.socket_path))
                    connection.sendall(
                        _CONTROL_FRAME.pack(
                            _KERNEL_MAGIC,
                            _KERNEL_PROTOCOL_VERSION,
                            _SHUTDOWN_MESSAGE,
                            0,
                            0,
                            b"",
                            b"",
                            b"",
                        )
                    )
                    response = bytearray()
                    while len(response) < _CONTROL_FRAME.size:
                        block = connection.recv(_CONTROL_FRAME.size - len(response))
                        if not block:
                            break
                        response.extend(block)
                    if len(response) != _CONTROL_FRAME.size:
                        raise RuntimeError("host scheduler shutdown receipt is partial")
                    values = _CONTROL_FRAME.unpack(response)
                    if (
                        values[0] != _KERNEL_MAGIC
                        or values[1] != _KERNEL_PROTOCOL_VERSION
                        or values[2] != _RELEASED_MESSAGE
                    ):
                        raise RuntimeError("host scheduler shutdown receipt is invalid")
            except (OSError, RuntimeError) as error:
                shutdown_error = error
                process.terminate()
        runtime_statistics: dict[str, object] | None = None
        try:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired as error:
                if not force and shutdown_error is None:
                    shutdown_error = RuntimeError(
                        "host scheduler did not exit after its shutdown receipt"
                    )
                    shutdown_error.__cause__ = error
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired as terminate_error:
                    if not force and shutdown_error is None:
                        shutdown_error = RuntimeError("host scheduler did not exit after terminate")
                        shutdown_error.__cause__ = terminate_error
                    process.kill()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired as kill_error:
                        if shutdown_error is None:
                            shutdown_error = RuntimeError("host scheduler did not exit after kill")
                            shutdown_error.__cause__ = kill_error
            if not force and shutdown_error is None:
                if process.returncode != 0:
                    stderr = (process.stderr.read() if process.stderr is not None else b"").decode(
                        "utf-8", errors="replace"
                    )
                    shutdown_error = RuntimeError(
                        "host scheduler exited unsuccessfully after its shutdown "
                        f"receipt: {stderr.strip()}"
                    )
                else:
                    stdout = (process.stdout.read() if process.stdout is not None else b"").decode(
                        "utf-8", errors="strict"
                    )
                    runtime_statistics = self._parse_runtime_statistics(stdout)
                    if runtime_statistics["worker_threads"] != self.worker_threads:
                        raise RuntimeError(
                            "host scheduler runtime worker-thread receipt differs "
                            "from configuration"
                        )
                    if runtime_statistics["request_threads"] != self.request_threads:
                        raise RuntimeError(
                            "host scheduler runtime request-thread receipt differs "
                            "from configuration"
                        )
                    terminal_io = runtime_statistics["terminal_process_io"]
                    if not isinstance(terminal_io, dict):
                        raise RuntimeError("native scheduler terminal I/O is unavailable")
                    if (
                        self._task_started_monotonic is None
                        or self._process_create_time is None
                        or self._process_start_time_ticks is None
                    ):
                        raise RuntimeError("native scheduler terminal I/O identity is unavailable")
                    self._terminal_process_io_receipt = {
                        "schema_version": TERMINAL_PROCESS_IO_RECEIPT_SCHEMA_VERSION,
                        "pid": process.pid,
                        "parent_pid": os.getpid(),
                        "create_time": self._process_create_time,
                        "start_time_ticks": self._process_start_time_ticks,
                        "task_started_monotonic": self._task_started_monotonic,
                        "captured_monotonic": time.monotonic(),
                        "read_bytes": terminal_io["read_bytes"],
                        "write_bytes": terminal_io["write_bytes"],
                    }
        finally:
            self._process = None
            self.socket_path.unlink(missing_ok=True)
            if run_nonce is not None:
                self._cleanup_owned_segments(process.pid, run_nonce)
            self._run_nonce = None
        if shutdown_error is not None:
            raise RuntimeError(
                "host scheduler graceful shutdown failed without fallback"
            ) from shutdown_error
        self._runtime_statistics = runtime_statistics

    def _parse_runtime_statistics(self, raw: str) -> dict[str, object]:
        lines = tuple(line.strip() for line in raw.splitlines() if line.strip())
        if len(lines) != 1:
            raise RuntimeError("host scheduler runtime statistics must contain exactly one record")
        try:
            decoded = json.loads(lines[0])
        except json.JSONDecodeError as error:
            raise RuntimeError("host scheduler runtime statistics are not valid JSON") from error
        if not isinstance(decoded, dict):
            raise RuntimeError("host scheduler runtime statistics must be an object")
        expected_keys = {
            "schema_version",
            "worker_threads",
            "request_threads",
            "receipt_writer_threads",
            "peak_active_requests",
            "peak_distinct_client_pids",
            "request_queue",
            "work_queue",
            "task_receipts",
            "terminal_process_io",
        }
        if set(decoded) != expected_keys:
            raise RuntimeError("host scheduler runtime statistics fields are invalid")
        if decoded["schema_version"] != "stage05.2-native-scheduler-runtime-v4":
            raise RuntimeError("host scheduler runtime statistics schema is invalid")
        terminal_io = decoded["terminal_process_io"]
        if not isinstance(terminal_io, dict) or set(terminal_io) != {
            "read_bytes",
            "write_bytes",
        }:
            raise RuntimeError("host scheduler terminal I/O fields are invalid")
        for field in ("read_bytes", "write_bytes"):
            self._require_nonnegative_integer(terminal_io[field], f"terminal_process_io.{field}")
        for field in (
            "worker_threads",
            "request_threads",
            "receipt_writer_threads",
            "peak_active_requests",
            "peak_distinct_client_pids",
        ):
            self._require_nonnegative_integer(decoded[field], field)
        if decoded["receipt_writer_threads"] != 1:
            raise RuntimeError("host scheduler must own exactly one task-receipt writer")
        request_queue = self._validate_queue_statistics(
            decoded["request_queue"], request_queue=True
        )
        work_queue = self._validate_queue_statistics(decoded["work_queue"], request_queue=False)
        for queue_name, statistics in (
            ("request", request_queue),
            ("work", work_queue),
        ):
            if statistics["queue_full_count"] != 0 or statistics["rejected_count"] != 0:
                raise RuntimeError(f"host scheduler {queue_name} queue overflowed or rejected work")
        if request_queue["pending"] != 0:
            raise RuntimeError("host scheduler request queue was not drained")
        if work_queue["pending"] != 0 or work_queue["active"] != 0:
            raise RuntimeError("host scheduler work queue was not drained")
        decoded["task_receipts"] = self._validate_task_receipts(
            decoded["task_receipts"],
            completed_tasks=self._require_nonnegative_integer(
                work_queue["completed"], "work_queue.completed"
            ),
        )
        return cast(dict[str, object], decoded)

    def _validate_task_receipts(
        self,
        value: object,
        *,
        completed_tasks: int,
    ) -> dict[str, object]:
        output_path = self._task_receipt_output_path
        if output_path is None:
            raise RuntimeError("host scheduler task-receipt target is unavailable")
        return validate_native_work_task_receipt_stream(
            value,
            expected_path=output_path,
            completed_tasks=completed_tasks,
            worker_threads=self.worker_threads,
        )

    @classmethod
    def _validate_queue_statistics(
        cls, value: object, *, request_queue: bool
    ) -> Mapping[str, object]:
        if not isinstance(value, dict):
            raise RuntimeError("host scheduler queue statistics must be an object")
        common_fields = {
            "pending",
            "peak_pending",
            "queue_full_count",
            "rejected_count",
            "completed",
            "total_wait_seconds",
            "maximum_wait_seconds",
            "total_service_seconds",
            "maximum_service_seconds",
            "wait_histogram",
            "service_histogram",
        }
        expected_fields = (
            common_fields
            if request_queue
            else common_fields
            | {
                "active",
                "peak_active",
            }
        )
        if set(value) != expected_fields:
            raise RuntimeError("host scheduler queue statistics fields are invalid")
        integer_fields = {
            "pending",
            "peak_pending",
            "queue_full_count",
            "rejected_count",
            "completed",
        }
        if not request_queue:
            integer_fields |= {"active", "peak_active"}
        for field in integer_fields:
            cls._require_nonnegative_integer(value[field], field)
        for field in (
            "total_wait_seconds",
            "maximum_wait_seconds",
            "total_service_seconds",
            "maximum_service_seconds",
        ):
            cls._require_nonnegative_number(value[field], field)
        for field in ("wait_histogram", "service_histogram"):
            histogram = value[field]
            if (
                not isinstance(histogram, Sequence)
                or isinstance(histogram, (str, bytes, bytearray))
                or len(histogram) != 32
            ):
                raise RuntimeError("host scheduler queue histogram must have exactly 32 bins")
            for index, count in enumerate(histogram):
                cls._require_nonnegative_integer(count, f"{field}[{index}]")
        return cast(Mapping[str, object], value)

    @staticmethod
    def _require_nonnegative_integer(value: object, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"host scheduler runtime statistic {field} must be non-negative int")
        return value

    @staticmethod
    def _require_nonnegative_number(value: object, field: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise RuntimeError(
                f"host scheduler runtime statistic {field} must be finite and non-negative"
            )
        return float(value)

    @staticmethod
    def _cleanup_owned_segments(process_id: int, run_nonce: str) -> None:
        prefix = f"evrptw-s52-kernel-{process_id}-{run_nonce}-"
        for segment in Path("/dev/shm").glob(f"{prefix}*"):
            if segment.is_file() and segment.name.startswith(prefix):
                segment.unlink(missing_ok=True)

    @property
    def process_id(self) -> int:
        process = self._process
        if process is None or process.poll() is not None:
            raise RuntimeError("host scheduler is not running")
        return process.pid

    @property
    def is_running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    @property
    def request_thread_count(self) -> int:
        """Configured request-worker count passed to the native service."""

        return self.request_threads

    def observed_cpu_affinity(self) -> tuple[int, ...]:
        """Return the scheduler process's effective CPU set from Linux."""

        try:
            return tuple(sorted(os.sched_getaffinity(self.process_id)))
        except AttributeError as error:
            raise RuntimeError(
                "native scheduler CPU affinity observation is unavailable"
            ) from error

    def observed_thread_count(self) -> int:
        task_directory = Path("/proc") / str(self.process_id) / "task"
        return len(tuple(task_directory.iterdir()))

    def observed_request_thread_count(self) -> int:
        """Return the scheduler's live request-worker count from Linux thread names."""

        task_directory = Path("/proc") / str(self.process_id) / "task"
        count = 0
        for task in task_directory.iterdir():
            try:
                if task.joinpath("comm").read_text(encoding="utf-8").strip() == "s52-request":
                    count += 1
            except (FileNotFoundError, PermissionError):
                continue
        return count

    def observed_task_receipt_writer_thread_count(self) -> int:
        """Return the live bounded receipt-writer count from Linux thread names."""

        task_directory = Path("/proc") / str(self.process_id) / "task"
        count = 0
        for task in task_directory.iterdir():
            try:
                if task.joinpath("comm").read_text(encoding="utf-8").strip() == "s52-task-write":
                    count += 1
            except (FileNotFoundError, PermissionError):
                continue
        return count

    @property
    def runtime_statistics(self) -> dict[str, object]:
        """Return the validated, post-drain scheduler shutdown statistics."""

        if self._process is not None:
            raise RuntimeError("host scheduler is still running")
        if self._runtime_statistics is None:
            raise RuntimeError("host scheduler runtime statistics are unavailable")
        return dict(self._runtime_statistics)

    @property
    def terminal_process_io_receipt(self) -> dict[str, object]:
        """Return the scheduler's cooperative final cumulative I/O receipt."""

        if self._process is not None:
            raise RuntimeError("host scheduler is still running")
        if self._terminal_process_io_receipt is None:
            raise RuntimeError("host scheduler terminal I/O receipt is unavailable")
        return dict(self._terminal_process_io_receipt)

    def __enter__(self) -> NativeHostScheduler:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def dispatch_full_native_alns(
    socket_path: str,
    *arrays: npt.NDArray[Any],
) -> object:
    """Submit one all-or-nothing solve through native UDS/POSIX shared memory."""

    from evrptw import _core as native_core

    entrypoint = cast(Callable[..., object], native_core.full_native_alns_host_v2)
    return entrypoint(socket_path, *arrays)


__all__ = ("NativeHostScheduler", "dispatch_full_native_alns")
