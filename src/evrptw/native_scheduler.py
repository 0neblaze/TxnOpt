"""Host-wide Stage 5.2 scheduler control and shared-memory transport."""

from __future__ import annotations

import multiprocessing
import os
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
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


def _scheduler_service_main(socket_path: str, worker_threads: int) -> None:
    from evrptw import _core as native_core

    entrypoint = cast(Callable[[str, int], None], native_core.run_host_scheduler_service_v2)
    entrypoint(socket_path, worker_threads)


@dataclass(slots=True)
class NativeHostScheduler:
    """One temporary run-wide scheduler with 24 request worker threads."""

    socket_path: Path
    worker_threads: int = 24
    _process: BaseProcess | None = None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("host scheduler is already started")
        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=_scheduler_service_main,
            args=(str(self.socket_path), self.worker_threads),
            name="evrptw-native-host-scheduler",
        )
        previous_environment = {
            name: os.environ.get(name) for name in _NUMERIC_THREAD_ENVIRONMENT
        }
        try:
            os.environ.update(_NUMERIC_THREAD_ENVIRONMENT)
            process.start()
        finally:
            for name, previous in previous_environment.items():
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous
        self._process = process
        deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if not process.is_alive():
                process.join(timeout=1.0)
                self._process = None
                self.socket_path.unlink(missing_ok=True)
                raise RuntimeError("host scheduler exited before becoming ready")
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
        if process.is_alive() and force:
            process.terminate()
        elif process.is_alive():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.connect(str(self.socket_path))
                    connection.sendall(b"X")
            except OSError:
                process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        self._process = None
        self.socket_path.unlink(missing_ok=True)

    @property
    def process_id(self) -> int:
        process = self._process
        if process is None or process.pid is None or not process.is_alive():
            raise RuntimeError("host scheduler is not running")
        return process.pid

    @property
    def is_running(self) -> bool:
        process = self._process
        return process is not None and process.is_alive()

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

    entrypoint = cast(Callable[..., object], native_core.dispatch_host_scheduler_v2)
    return entrypoint(socket_path, *arrays)


__all__ = ("NativeHostScheduler", "dispatch_full_native_alns")
