"""Host-wide Stage 5.2 scheduler control and shared-memory transport."""

from __future__ import annotations

import json
import multiprocessing
import pickle
import socket
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

_FRAME = struct.Struct("!Q")
_READY_TIMEOUT_SECONDS = 10.0
_IPC_TIMEOUT_SECONDS = 130.0
_MAX_CONTROL_BYTES = 1 << 20


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        chunk = connection.recv(size - len(payload))
        if not chunk:
            raise RuntimeError("host scheduler received a partial IPC frame")
        payload.extend(chunk)
    return bytes(payload)


def _recv_frame(connection: socket.socket) -> bytes:
    size = _FRAME.unpack(_recv_exact(connection, _FRAME.size))[0]
    if size <= 0 or size > _MAX_CONTROL_BYTES:
        raise RuntimeError("host scheduler control frame size is invalid")
    return _recv_exact(connection, size)


def _send_frame(connection: socket.socket, payload: bytes) -> None:
    connection.sendall(_FRAME.pack(len(payload)) + payload)


def _array_descriptor(
    array: npt.NDArray[np.generic],
) -> tuple[dict[str, object], shared_memory.SharedMemory]:
    contiguous = np.ascontiguousarray(array)
    segment = shared_memory.SharedMemory(create=True, size=max(1, contiguous.nbytes))
    if contiguous.nbytes:
        target = np.ndarray(contiguous.shape, dtype=contiguous.dtype, buffer=segment.buf)
        target[...] = contiguous
    return (
        {
            "name": segment.name,
            "shape": list(contiguous.shape),
            "dtype": contiguous.dtype.str,
            "nbytes": contiguous.nbytes,
        },
        segment,
    )


def _scheduler_service_main(socket_path: str, worker_threads: int) -> None:
    from evrptw import _core as native_core

    entrypoint = cast(Callable[[str, int], None], native_core.run_host_scheduler_service_v1)
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
        process.start()
        self._process = process
        deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if not process.is_alive():
                raise RuntimeError("host scheduler exited before becoming ready")
            if self.socket_path.exists():
                return
            time.sleep(0.01)
        self.close(force=True)
        raise RuntimeError("host scheduler did not become ready")

    def close(self, *, force: bool = False) -> None:
        process = self._process
        if process is None:
            return
        if process.is_alive() and not force:
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.connect(str(self.socket_path))
                    connection.sendall(b"X")
            except OSError:
                force = True
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
    *arrays: npt.NDArray[np.generic],
) -> object:
    """Submit one all-or-nothing full solve through shared memory and UDS control."""

    descriptors: list[dict[str, object]] = []
    segments: list[shared_memory.SharedMemory] = []
    try:
        for array in arrays:
            descriptor, segment = _array_descriptor(array)
            descriptors.append(descriptor)
            segments.append(segment)
        request = json.dumps(
            {"operation": "full_native_alns_v1", "arrays": descriptors},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(_IPC_TIMEOUT_SECONDS)
            connection.connect(socket_path)
            _send_frame(connection, request)
            response = json.loads(_recv_frame(connection))
        if not isinstance(response, dict) or response.get("ok") is not True:
            message = response.get("error") if isinstance(response, dict) else response
            raise RuntimeError(f"host scheduler transaction failed: {message}")
        output_name = response.get("output_name")
        output_size = response.get("output_size")
        if not isinstance(output_name, str) or not isinstance(output_size, int) or output_size <= 0:
            raise RuntimeError("host scheduler returned an invalid output descriptor")
        output = shared_memory.SharedMemory(name=output_name)
        try:
            output_buffer = output.buf
            if output_buffer is None:
                raise RuntimeError("host scheduler output shared memory is unavailable")
            return pickle.loads(bytes(output_buffer[:output_size]))
        finally:
            output.close()
            output.unlink()
    except (OSError, EOFError, pickle.UnpicklingError) as error:
        raise RuntimeError("host scheduler IPC failed without fallback") from error
    finally:
        for segment in segments:
            segment.close()
            segment.unlink()


__all__ = ("NativeHostScheduler", "dispatch_full_native_alns")
