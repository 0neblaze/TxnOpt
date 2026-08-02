"""Host-wide Stage 5.2 scheduler control and shared-memory transport."""

from __future__ import annotations

import json
import multiprocessing
import pickle
import socket
import struct
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

_FRAME = struct.Struct("!Q")
_READY_TIMEOUT_SECONDS = 10.0
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


def _handle_request(connection: socket.socket) -> None:
    segments: list[shared_memory.SharedMemory] = []
    try:
        request = json.loads(_recv_frame(connection))
        if not isinstance(request, dict) or request.get("operation") != "full_native_alns_v1":
            raise RuntimeError("host scheduler received an unknown operation")
        raw_arrays = request.get("arrays")
        if not isinstance(raw_arrays, list) or len(raw_arrays) != 12:
            raise RuntimeError("host scheduler received an invalid SoA descriptor set")
        arrays: list[npt.NDArray[np.generic]] = []
        for raw in raw_arrays:
            if not isinstance(raw, dict):
                raise RuntimeError("host scheduler array descriptor is invalid")
            name = raw.get("name")
            shape = raw.get("shape")
            dtype = raw.get("dtype")
            nbytes = raw.get("nbytes")
            if (
                not isinstance(name, str)
                or not isinstance(shape, list)
                or not all(isinstance(value, int) and value >= 0 for value in shape)
                or not isinstance(dtype, str)
                or not isinstance(nbytes, int)
                or nbytes < 0
            ):
                raise RuntimeError("host scheduler array descriptor fields are invalid")
            segment = shared_memory.SharedMemory(name=name)
            segments.append(segment)
            array = np.ndarray(tuple(shape), dtype=np.dtype(dtype), buffer=segment.buf)
            if array.nbytes != nbytes or not array.flags.c_contiguous:
                raise RuntimeError("host scheduler shared-memory array does not reconcile")
            arrays.append(array)

        from evrptw import _core as native_core

        entrypoint = cast(Callable[..., object], native_core.full_native_alns_v1)
        result = entrypoint(*arrays)
        encoded = pickle.dumps(result, protocol=5)
        output = shared_memory.SharedMemory(create=True, size=max(1, len(encoded)))
        try:
            output_buffer = output.buf
            if output_buffer is None:
                raise RuntimeError("host scheduler output shared memory is unavailable")
            output_buffer[: len(encoded)] = encoded
            _send_frame(
                connection,
                json.dumps(
                    {
                        "ok": True,
                        "output_name": output.name,
                        "output_size": len(encoded),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
        finally:
            output.close()
    except BaseException as error:
        response = json.dumps(
            {"ok": False, "error_type": type(error).__name__, "error": str(error)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        with suppress(OSError):
            _send_frame(connection, response)
    finally:
        for segment in segments:
            segment.close()
        connection.close()


def _scheduler_service_main(socket_path: str, worker_threads: int) -> None:
    endpoint = Path(socket_path)
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    endpoint.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(64)
    try:
        with ThreadPoolExecutor(max_workers=worker_threads) as executor:
            while True:
                connection, _ = server.accept()
                try:
                    control = connection.recv(1, socket.MSG_PEEK)
                except OSError:
                    connection.close()
                    continue
                if control == b"X":
                    connection.recv(1)
                    connection.close()
                    break
                executor.submit(_handle_request, connection)
    finally:
        server.close()
        endpoint.unlink(missing_ok=True)


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
