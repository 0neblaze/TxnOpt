"""Bounded native-round resource soak with a signed, claim-limited receipt."""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import numpy as np

from txnopt import _native
from txnopt_evidence.codec import write_signed_json


def run_native_resource_soak(
    *,
    rounds: int,
    workers: int,
    warmup_rounds: int,
    maximum_rss_growth_kib: int,
) -> dict[str, object]:
    if min(rounds, workers, maximum_rss_growth_kib) <= 0 or warmup_rounds < 0:
        raise ValueError("soak rounds, workers, and RSS limit must be positive")
    proc = Path("/proc/self")
    if not proc.is_dir():
        raise RuntimeError("resource soak requires Linux procfs")
    context = _native.EVRPTWContext(
        np.asarray((0, 1, 1), dtype=np.int64),
        np.asarray((0.0, 1.0, 1.0), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((100.0, 100.0, 100.0), dtype=np.float64),
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        np.asarray(((0.0, 1.0, 2.0), (1.0, 0.0, 1.0), (2.0, 1.0, 0.0))),
        np.ones((3, 3), dtype=np.uint8),
        np.asarray((100.0, 10.0, 1.0, 0.1, 1.0), dtype=np.float64),
        workers,
    )
    offsets = np.asarray((0, 1, 2), dtype=np.int64)
    indices = np.asarray((1, 2), dtype=np.int64)
    for _ in range(warmup_rounds):
        context.exact_round_v1(offsets, indices, math.inf, 64)
    gc.collect()
    before = _resource_snapshot(proc)
    started_ns = time.monotonic_ns()
    last_receipt: _native.NativeRoundReceipt | None = None
    for round_index in range(rounds):
        output = context.exact_round_v1(offsets, indices, math.inf, 64)
        last_receipt = output[9]
        if round_index % 1_000 == 0:
            gc.collect()
    elapsed_ns = time.monotonic_ns() - started_ns
    del output
    gc.collect()
    after = _resource_snapshot(proc)
    rss_growth_kib = after["rss_kib"] - before["rss_kib"]
    status = (
        "PASS"
        if rss_growth_kib <= maximum_rss_growth_kib
        and after["thread_count"] == before["thread_count"]
        and after["fd_count"] == before["fd_count"]
        and last_receipt is not None
        and last_receipt["fallback_count"] == 0
        else "FAIL"
    )
    return {
        "schema_version": "txnopt-native-resource-soak-v1",
        "status": status,
        "rounds": rounds,
        "warmup_rounds": warmup_rounds,
        "workers": workers,
        "elapsed_ns": elapsed_ns,
        "before": before,
        "after": after,
        "rss_growth_kib": rss_growth_kib,
        "maximum_rss_growth_kib": maximum_rss_growth_kib,
        "fallback_count": 0 if last_receipt is None else last_receipt["fallback_count"],
        "native_round_call_count": (
            0 if last_receipt is None else last_receipt["round_call_count"]
        ),
        "native_build_attestation": dict(_native.BUILD_ATTESTATION),
    }


def write_resource_soak_receipt(
    output: Path,
    *,
    rounds: int,
    workers: int,
    warmup_rounds: int,
    maximum_rss_growth_kib: int,
) -> str:
    result = run_native_resource_soak(
        rounds=rounds,
        workers=workers,
        warmup_rounds=warmup_rounds,
        maximum_rss_growth_kib=maximum_rss_growth_kib,
    )
    if result["status"] != "PASS":
        raise RuntimeError(f"native resource soak failed: {result}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return write_signed_json(output, result)


def _resource_snapshot(proc: Path) -> dict[str, int]:
    status = (proc / "status").read_text(encoding="utf-8")
    rss_line = next(
        (line for line in status.splitlines() if line.startswith("VmRSS:")),
        None,
    )
    if rss_line is None:
        raise RuntimeError("procfs status does not contain VmRSS")
    return {
        "rss_kib": int(rss_line.split()[1]),
        "thread_count": sum(1 for _ in (proc / "task").iterdir()),
        "fd_count": sum(1 for _ in (proc / "fd").iterdir()),
    }


__all__ = ["run_native_resource_soak", "write_resource_soak_receipt"]
