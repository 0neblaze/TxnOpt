"""Observable process-memory release for Stage 5.2 producer phases."""

from __future__ import annotations

import ctypes
import gc
import os
import sysconfig
from typing import Any

import psutil  # type: ignore[import-untyped]
import pyarrow as pa


def release_stage052_process_memory() -> dict[str, Any]:
    """Release reclaimable Python, Arrow, and libc pages and report the result."""

    process = psutil.Process()
    rss_before_release = process.memory_info().rss
    collected_objects = gc.collect()
    rss_after_gc = process.memory_info().rss

    arrow_pool = pa.default_memory_pool()
    arrow_bytes_before_release = arrow_pool.bytes_allocated()
    arrow_pool.release_unused()
    arrow_bytes_after_release = arrow_pool.bytes_allocated()
    rss_after_arrow_release = process.memory_info().rss
    if arrow_bytes_after_release > arrow_bytes_before_release:
        raise RuntimeError("PyArrow memory-pool release increased live allocations")

    system_allocator_trim_available = False
    system_allocator_trim_result: int | None = None
    try:
        process_image = ctypes.CDLL(None, use_errno=True)
        malloc_trim = process_image.malloc_trim
    except AttributeError:
        pass
    else:
        system_allocator_trim_available = True
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        system_allocator_trim_result = int(malloc_trim(0))
    rss_after_system_allocator_trim = process.memory_info().rss

    return {
        "python_allocator": os.environ.get("PYTHONMALLOC", "default"),
        "python_with_mimalloc": bool(sysconfig.get_config_var("WITH_MIMALLOC")),
        "gc_collected_objects": collected_objects,
        "rss_bytes_before_release": rss_before_release,
        "rss_bytes_after_gc": rss_after_gc,
        "rss_bytes_before_arrow_release": rss_after_gc,
        "arrow_memory_pool_backend": arrow_pool.backend_name,
        "arrow_bytes_before_release": arrow_bytes_before_release,
        "arrow_bytes_after_release": arrow_bytes_after_release,
        "rss_bytes_after_arrow_release": rss_after_arrow_release,
        "system_allocator_trim_available": system_allocator_trim_available,
        "system_allocator_trim_result": system_allocator_trim_result,
        "rss_bytes_after_system_allocator_trim": rss_after_system_allocator_trim,
    }
