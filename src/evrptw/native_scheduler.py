"""Host-wide Stage 5.2 scheduler control and shared-memory transport."""

from __future__ import annotations

import os
import secrets
import socket
import struct
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy.typing as npt

_READY_TIMEOUT_SECONDS = 10.0
_NUMERIC_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


_CONTROL_FRAME = struct.Struct("<QIIQQ128s64s192s")
_KERNEL_MAGIC = 0x4556525054574B32
_KERNEL_PROTOCOL_VERSION = 2
_SHUTDOWN_MESSAGE = 6
_RELEASED_MESSAGE = 4


@dataclass(slots=True)
class NativeHostScheduler:
    """One temporary run-wide scheduler with one shared 24-thread compute pool."""

    socket_path: Path
    worker_threads: int = 24
    enable_fault_injection: bool = False
    production_fault: str | None = None
    _process: subprocess.Popen[bytes] | None = None
    _run_nonce: str | None = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("host scheduler is already started")
        if self.production_fault not in {None, "pause_before_execute"}:
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
        command = [
            str(executable),
            str(self.socket_path),
            str(self.worker_threads),
            run_nonce,
        ]
        if self.enable_fault_injection:
            command.append("--enable-fault-injection")
        if self.production_fault is not None:
            command.append(f"--production-fault={self.production_fault}")
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=scheduler_environment,
        )
        self._process = process
        self._run_nonce = run_nonce
        deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stderr = (
                    process.stderr.read() if process.stderr is not None else b""
                ).decode("utf-8", errors="replace")
                self._process = None
                self.socket_path.unlink(missing_ok=True)
                self._cleanup_owned_segments(process.pid, run_nonce)
                self._run_nonce = None
                raise RuntimeError(
                    f"host scheduler exited before becoming ready: {stderr.strip()}"
                )
            if self.socket_path.exists():
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                        probe.connect(str(self.socket_path))
                except ConnectionRefusedError:
                    time.sleep(0.01)
                    continue
                expected_threads = self.worker_threads + 7
                if self.observed_thread_count() == expected_threads:
                    return
            time.sleep(0.01)
        self.close(force=True)
        raise RuntimeError("host scheduler did not become ready")

    def close(self, *, force: bool = False) -> None:
        process = self._process
        if process is None:
            return
        run_nonce = self._run_nonce
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
            except (OSError, RuntimeError):
                process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5.0)
        self._process = None
        self.socket_path.unlink(missing_ok=True)
        if run_nonce is not None:
            self._cleanup_owned_segments(process.pid, run_nonce)
        self._run_nonce = None

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

    def observed_thread_count(self) -> int:
        task_directory = Path("/proc") / str(self.process_id) / "task"
        return len(tuple(task_directory.iterdir()))

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
