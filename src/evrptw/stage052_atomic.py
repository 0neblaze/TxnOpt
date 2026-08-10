"""Linux no-replace publication primitive for immutable Stage 5.2 evidence."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path
from typing import Final

_AT_FDCWD: Final = -100
_RENAME_NOREPLACE: Final = 1


def publish_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename ``source`` while refusing an existing destination."""

    if sys.platform != "linux":
        raise RuntimeError("Stage 5.2 immutable publication requires Linux/WSL")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Linux renameat2 is unavailable for immutable publication")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if (
        renameat2(
            _AT_FDCWD,
            source_bytes,
            _AT_FDCWD,
            destination_bytes,
            _RENAME_NOREPLACE,
        )
        == 0
    ):
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)
