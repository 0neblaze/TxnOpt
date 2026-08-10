"""Paired A/B calibration for Stage 5.2 process-tree telemetry overhead."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from evrptw.runtime_envelope import ProcessTreeMonitor
from evrptw.stage052_atomic import publish_no_replace
from evrptw.stage052_performance import (
    TelemetryOverheadReceipt,
    require_clean_repository_root,
)

TELEMETRY_SAMPLE_SCHEMA_VERSION: Final = "stage05.2-representative-telemetry-sample-v3"

_CHILD_WORKLOAD: Final = """
import hashlib
import sys

iterations = int(sys.argv[1])
state = b"stage05.2-telemetry-overhead-v1"
for ordinal in range(iterations):
    state = hashlib.sha256(state + ordinal.to_bytes(8, "little")).digest()
print(state.hex())
"""

_REPRESENTATIVE_SURFACE_FIELDS: Final = (
    "semantic_telemetry",
    "physical_telemetry",
    "persistence",
    "independent_replay",
)


class TelemetryOverheadError(RuntimeError):
    """The A/B workload or its evidence failed validation."""


@dataclass(frozen=True, slots=True)
class TelemetryWorkloadSample:
    """One semantic fingerprint plus monitored evidence from a real axis."""

    fingerprint: bytes
    resource_summary: Mapping[str, object]
    workload_evidence: Mapping[str, object]


def _invoke_workload(iterations: int) -> bytes:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _CHILD_WORKLOAD, str(iterations)],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": "0",
        },
    )
    if completed.returncode != 0:
        raise TelemetryOverheadError(
            "telemetry A/B child failed: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    output = completed.stdout.strip()
    if len(output) != 64 or any(byte not in b"0123456789abcdef" for byte in output):
        raise TelemetryOverheadError("telemetry A/B child output is not deterministic SHA-256")
    return output


def measure_telemetry_overhead(
    *,
    iterations: int,
    repeat_count: int = 5,
    sample_interval_seconds: float = 0.05,
    minimum_unmonitored_seconds: float = 0.5,
    workload: Callable[[int], bytes] = _invoke_workload,
) -> TelemetryOverheadReceipt:
    """Run an alternating paired monitor-off/monitor-on process-tree workload."""

    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if (
        isinstance(repeat_count, bool)
        or not isinstance(repeat_count, int)
        or repeat_count < 5
        or repeat_count % 2 == 0
    ):
        raise ValueError("repeat_count must be odd and at least 5")
    if sample_interval_seconds <= 0.0 or minimum_unmonitored_seconds < 0.0:
        raise ValueError("telemetry timing parameters are invalid")
    allowed_cpu_count = len(os.sched_getaffinity(0))
    if allowed_cpu_count <= 0:
        raise TelemetryOverheadError("CPU affinity is empty")

    warm_output = workload(iterations)
    outputs = {warm_output}
    off_seconds: list[float] = []
    on_seconds: list[float] = []
    summaries: list[Mapping[str, object]] = []
    orders: list[str] = []

    def run_unmonitored() -> tuple[float, bytes]:
        started = time.perf_counter()
        output = workload(iterations)
        return time.perf_counter() - started, output

    def run_monitored() -> tuple[float, bytes, Mapping[str, object]]:
        started = time.perf_counter()
        with ProcessTreeMonitor(sample_interval_seconds=sample_interval_seconds) as monitor:
            output = workload(iterations)
        elapsed = time.perf_counter() - started
        statistics = monitor.statistics(
            elapsed_seconds=elapsed,
            compute_thread_limit=allowed_cpu_count,
        )
        return elapsed, output, statistics

    for index in range(repeat_count):
        order = "off-on" if index % 2 == 0 else "on-off"
        orders.append(order)
        if order == "off-on":
            off, off_output = run_unmonitored()
            on, on_output, summary = run_monitored()
        else:
            on, on_output, summary = run_monitored()
            off, off_output = run_unmonitored()
        off_seconds.append(off)
        on_seconds.append(on)
        summaries.append(summary)
        outputs.update((off_output, on_output))
    if len(outputs) != 1:
        raise TelemetryOverheadError("telemetry on/off workload output diverged")
    if min(off_seconds) < minimum_unmonitored_seconds:
        raise TelemetryOverheadError(
            "telemetry A/B workload is too short for a trustworthy 2% gate"
        )
    return TelemetryOverheadReceipt(
        unmonitored_seconds=tuple(off_seconds),
        monitored_seconds=tuple(on_seconds),
        pair_orders=tuple(orders),
        sample_interval_seconds=sample_interval_seconds,
        workload_output_sha256=hashlib.sha256(next(iter(outputs))).hexdigest(),
        monitored_resource_summaries=tuple(summaries),
        workload_evidence={
            "kind": "synthetic-process-tree-probe",
            "mode": "synthetic",
            "axis": "sha256-loop",
            "exact_calls": 0,
            "instance": "synthetic",
            "semantic_telemetry": False,
            "physical_telemetry": False,
            "persistence": False,
            "independent_replay": False,
            "fingerprints_identical": True,
        },
    )


def measure_representative_telemetry_overhead(
    *,
    run_sample: Callable[[bool, int], TelemetryWorkloadSample],
    repeat_count: int = 5,
    sample_interval_seconds: float = 0.05,
    minimum_unmonitored_seconds: float = 0.5,
) -> TelemetryOverheadReceipt:
    """Measure full semantic/physical/persistence/replay overhead on one real axis."""

    if (
        isinstance(repeat_count, bool)
        or not isinstance(repeat_count, int)
        or repeat_count < 5
        or repeat_count % 2 == 0
    ):
        raise ValueError("repeat_count must be odd and at least 5")
    if sample_interval_seconds <= 0.0 or minimum_unmonitored_seconds < 0.0:
        raise ValueError("telemetry timing parameters are invalid")

    off_seconds: list[float] = []
    on_seconds: list[float] = []
    summaries: list[Mapping[str, object]] = []
    orders: list[str] = []
    paired_sample_evidence: list[Mapping[str, object]] = []

    def base_evidence(sample: TelemetryWorkloadSample) -> dict[str, object]:
        return {
            key: value
            for key, value in sample.workload_evidence.items()
            if key not in _REPRESENTATIVE_SURFACE_FIELDS
        }

    def timed(enabled: bool, sample_index: int) -> tuple[float, TelemetryWorkloadSample]:
        started = time.perf_counter()
        result = run_sample(enabled, sample_index)
        elapsed = time.perf_counter() - started
        if len(result.fingerprint) != 64 or any(
            byte not in b"0123456789abcdef" for byte in result.fingerprint
        ):
            raise TelemetryOverheadError("representative workload fingerprint is invalid")
        if any(
            result.workload_evidence.get(field) is not enabled
            for field in _REPRESENTATIVE_SURFACE_FIELDS
        ):
            raise TelemetryOverheadError("representative telemetry on/off surface is invalid")
        if result.workload_evidence.get("minimal_validator_replay") is not True:
            raise TelemetryOverheadError("representative minimal validation is missing")
        return elapsed, result

    def sample_evidence(
        *,
        enabled: bool,
        sample_index: int,
        elapsed_seconds: float,
        sample: TelemetryWorkloadSample,
    ) -> dict[str, object]:
        return {
            "enabled": enabled,
            "sample_index": sample_index,
            "elapsed_seconds": elapsed_seconds,
            "fingerprint": sample.fingerprint.decode("ascii"),
            "resource_summary": dict(sample.resource_summary),
            "workload_evidence": dict(sample.workload_evidence),
        }

    # Warm both paths once so import, parser, allocator, and native pool startup
    # do not systematically favor whichever side happens to run second.
    warm_off_seconds, warm_off = timed(False, -2)
    warm_on_seconds, warm_on = timed(True, -1)
    fingerprints = {warm_off.fingerprint, warm_on.fingerprint}
    evidence_rows: list[Mapping[str, object]] = [
        base_evidence(warm_off),
        base_evidence(warm_on),
    ]
    warm_sample_evidence = [
        sample_evidence(
            enabled=False,
            sample_index=-2,
            elapsed_seconds=warm_off_seconds,
            sample=warm_off,
        ),
        sample_evidence(
            enabled=True,
            sample_index=-1,
            elapsed_seconds=warm_on_seconds,
            sample=warm_on,
        ),
    ]

    for index in range(repeat_count):
        order = "off-on" if index % 2 == 0 else "on-off"
        orders.append(order)
        if order == "off-on":
            off, off_sample = timed(False, index * 2)
            on, on_sample = timed(True, index * 2 + 1)
        else:
            on, on_sample = timed(True, index * 2)
            off, off_sample = timed(False, index * 2 + 1)
        off_seconds.append(off)
        on_seconds.append(on)
        fingerprints.update((off_sample.fingerprint, on_sample.fingerprint))
        summaries.append(dict(on_sample.resource_summary))
        evidence_rows.extend((base_evidence(off_sample), base_evidence(on_sample)))
        paired_sample_evidence.append(
            {
                "pair_index": index,
                "order": order,
                "unmonitored": sample_evidence(
                    enabled=False,
                    sample_index=(index * 2 if order == "off-on" else index * 2 + 1),
                    elapsed_seconds=off,
                    sample=off_sample,
                ),
                "monitored": sample_evidence(
                    enabled=True,
                    sample_index=(index * 2 + 1 if order == "off-on" else index * 2),
                    elapsed_seconds=on,
                    sample=on_sample,
                ),
            }
        )

    if len(fingerprints) != 1:
        raise TelemetryOverheadError("telemetry on/off representative semantics diverged")
    first_evidence = dict(evidence_rows[0])
    if any(dict(row) != first_evidence for row in evidence_rows[1:]):
        raise TelemetryOverheadError("representative telemetry evidence changed between pairs")
    if min(off_seconds) < minimum_unmonitored_seconds:
        raise TelemetryOverheadError(
            "representative telemetry A/B workload is too short for a trustworthy 2% gate"
        )
    first_evidence["fingerprints_identical"] = True
    first_evidence.update({field: True for field in _REPRESENTATIVE_SURFACE_FIELDS})
    first_evidence["unmonitored_telemetry_surface"] = {
        field: False for field in _REPRESENTATIVE_SURFACE_FIELDS
    }
    first_evidence["monitored_telemetry_surface"] = {
        field: True for field in _REPRESENTATIVE_SURFACE_FIELDS
    }
    first_evidence["warm_sample_evidence"] = warm_sample_evidence
    first_evidence["paired_sample_evidence"] = paired_sample_evidence
    return TelemetryOverheadReceipt(
        unmonitored_seconds=tuple(off_seconds),
        monitored_seconds=tuple(on_seconds),
        pair_orders=tuple(orders),
        sample_interval_seconds=sample_interval_seconds,
        workload_output_sha256=hashlib.sha256(next(iter(fingerprints))).hexdigest(),
        monitored_resource_summaries=tuple(summaries),
        workload_evidence=first_evidence,
    )


def _atomic_signed_json(path: Path, payload: Mapping[str, object]) -> None:
    sidecar = Path(f"{path}.sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"telemetry overhead output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    path_published = False
    sidecar_published = False
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        with temporary_sidecar.open("x", encoding="ascii") as stream:
            stream.write(digest + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_no_replace(temporary_sidecar, sidecar)
        sidecar_published = True
        publish_no_replace(temporary, path)
        path_published = True
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        if not path_published and sidecar_published:
            sidecar.unlink(missing_ok=True)
        raise


def write_telemetry_overhead_receipt(
    path: Path,
    receipt: TelemetryOverheadReceipt,
) -> None:
    _atomic_signed_json(path, receipt.to_dict())


def load_telemetry_overhead_receipt(path: Path) -> TelemetryOverheadReceipt:
    sidecar = Path(f"{path}.sha256")
    encoded = path.read_bytes()
    if sidecar.read_text(encoding="ascii").strip() != hashlib.sha256(encoded).hexdigest():
        raise TelemetryOverheadError("telemetry overhead receipt SHA-256 mismatch")
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TelemetryOverheadError("telemetry overhead receipt is not valid JSON") from error
    if not isinstance(payload, dict):
        raise TelemetryOverheadError("telemetry overhead receipt root is not an object")
    try:
        return TelemetryOverheadReceipt.from_dict(payload)
    except ValueError as error:
        raise TelemetryOverheadError("telemetry overhead receipt is invalid") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure Stage 5.2 process-tree telemetry overhead with paired A/B runs"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        required=True,
        help="Explicit clean ext4 Git worktree used for source identity",
    )
    parser.add_argument("--iterations", type=int, default=3_000_000)
    parser.add_argument("--repeat-count", type=int, default=5)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.05)
    parser.add_argument("--minimum-unmonitored-seconds", type=float, default=0.5)
    arguments = parser.parse_args(argv)
    require_clean_repository_root(arguments.repository_root)
    try:
        receipt = measure_telemetry_overhead(
            iterations=arguments.iterations,
            repeat_count=arguments.repeat_count,
            sample_interval_seconds=arguments.sample_interval_seconds,
            minimum_unmonitored_seconds=arguments.minimum_unmonitored_seconds,
        )
        write_telemetry_overhead_receipt(arguments.output, receipt)
        receipt.require_passed()
    except (OSError, ValueError, TelemetryOverheadError) as error:
        parser.exit(1, f"{error}\n")
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
