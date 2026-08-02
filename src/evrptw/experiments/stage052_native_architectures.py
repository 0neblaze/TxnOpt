"""Five-mode Stage 5.2 native-architecture comparison runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from evrptw.alns import ALNSResult, solve_alns
from evrptw.cache_incremental import CacheIncrementalConfig
from evrptw.candidate_control import CandidateControlConfig
from evrptw.candidate_transaction import NativeCandidateTransactionConfig
from evrptw.exact_deadline import ExactDeadlineConfig
from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES
from evrptw.measurement import CheapScreeningConfig, MeasurementConfig
from evrptw.native_execution import Stage052NativeExecutionConfig
from evrptw.native_kernels import NativeKernelConfig
from evrptw.native_scheduler import NativeHostScheduler
from evrptw.parser import parse_schneider
from evrptw.repository import repository_root
from evrptw.stage04 import Stage04Config
from evrptw.validation import validate_routes

SCHEMA_VERSION = "stage05.2-native-architecture-comparison-v1"
SEEDS = (2014, 2015, 2016)
PAIRED_INSTANCES = ("c101C5", "c101_21", "r101_21", "rc101_21")
AXIS_NAMES = ("fixed_work", "wall_clock_30")
SHARD_PROCESSES = 6
THREADS_PER_SHARD = 4
TOTAL_COMPUTE_THREADS = 24


class ArchitectureMode(StrEnum):
    CURRENT_STAGE052 = "current_stage052"
    PYTHON_CANDIDATE_CONTROL = "python_candidate_control"
    PER_SOLVE_RUNTIME = "per_solve_runtime"
    FULL_NATIVE_ALNS = "full_native_alns"
    HOST_SCHEDULER = "host_scheduler"


MODES = tuple(ArchitectureMode)


@dataclass(frozen=True, slots=True)
class ArchitectureAxisTask:
    scope: str
    repeat: int
    axis: str
    instance_name: str
    seed: int
    benchmark_dir: Path
    output_root: Path
    run_labels: dict[str, str]
    scheduler_socket_path: str
    wheel_sha256: str
    native_sha256: str
    revision: str


def run_labels_for_scope(scope: str, attempt: int) -> dict[str, str]:
    if scope not in {"paired", "pilot"} or attempt <= 0:
        raise ValueError("native architecture scope/attempt is invalid")
    return {
        mode.value: (
            f"stage05.2_native_architecture_{mode.value}_{scope}_attempt{attempt:02d}"
        )
        for mode in MODES
    }


def build_axis_plan(
    scope: str,
    *,
    attempt: int,
    benchmark_dir: Path,
    output_root: Path,
    scheduler_socket_path: str,
    wheel_sha256: str,
    native_sha256: str,
    revision: str,
) -> tuple[ArchitectureAxisTask, ...]:
    instances: tuple[str, ...]
    axes: tuple[str, ...]
    if scope == "paired":
        instances = PAIRED_INSTANCES
        repeats = range(3)
        axes = AXIS_NAMES
    elif scope == "pilot":
        instances = tuple(FORMAL_INSTANCES)
        repeats = range(1)
        axes = ("wall_clock_30",)
    else:
        raise ValueError("scope must be paired or pilot")
    labels = run_labels_for_scope(scope, attempt)
    return tuple(
        ArchitectureAxisTask(
            scope=scope,
            repeat=repeat,
            axis=axis,
            instance_name=instance_name,
            seed=seed,
            benchmark_dir=benchmark_dir,
            output_root=output_root,
            run_labels=labels,
            scheduler_socket_path=scheduler_socket_path,
            wheel_sha256=wheel_sha256,
            native_sha256=native_sha256,
            revision=revision,
        )
        for repeat in repeats
        for axis in axes
        for instance_name in instances
        for seed in SEEDS
    )


def rotated_modes(task: ArchitectureAxisTask) -> tuple[ArchitectureMode, ...]:
    instance_order = (
        PAIRED_INSTANCES
        if task.scope == "paired"
        else tuple(FORMAL_INSTANCES)
    )
    axis_order = AXIS_NAMES if task.scope == "paired" else ("wall_clock_30",)
    rotation = (
        task.repeat * len(axis_order) * len(instance_order) * len(SEEDS)
        + axis_order.index(task.axis) * len(instance_order) * len(SEEDS)
        + instance_order.index(task.instance_name) * len(SEEDS)
        + SEEDS.index(task.seed)
    ) % len(MODES)
    return MODES[rotation:] + MODES[:rotation]


def expected_axis_count(scope: str) -> int:
    if scope == "paired":
        return len(PAIRED_INSTANCES) * len(SEEDS) * 3 * len(AXIS_NAMES) * len(MODES)
    if scope == "pilot":
        return len(FORMAL_INSTANCES) * len(SEEDS) * len(MODES)
    raise ValueError("scope must be paired or pilot")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_signed_json(path: Path, payload: object) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_bytes(payload) + b"\n"
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_bytes(data)
    temporary.replace(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(hashlib.sha256(data).hexdigest() + "\n", encoding="ascii")
    return len(data) + sidecar.stat().st_size


def _set_artifact_size(payload: dict[str, object]) -> None:
    payload["artifact_bytes"] = 0
    for _ in range(4):
        size = len(_canonical_bytes(payload)) + 1 + 65
        if payload["artifact_bytes"] == size:
            return
        payload["artifact_bytes"] = size
    raise RuntimeError("artifact byte count did not converge")


def _native_config(
    mode: ArchitectureMode,
    *,
    scheduler_socket_path: str | None = None,
) -> Stage052NativeExecutionConfig:
    if mode not in {
        ArchitectureMode.PER_SOLVE_RUNTIME,
        ArchitectureMode.FULL_NATIVE_ALNS,
        ArchitectureMode.HOST_SCHEDULER,
    }:
        raise ValueError("mode does not use the explicit native execution protocol")
    return Stage052NativeExecutionConfig(
        mode=mode.value,  # type: ignore[arg-type]
        native_kernel_config=NativeKernelConfig(),
        candidate_transaction_config=NativeCandidateTransactionConfig(),
        candidate_control_config=CandidateControlConfig(worker_count=THREADS_PER_SHARD),
        shard_processes=SHARD_PROCESSES,
        compute_threads_per_shard=THREADS_PER_SHARD,
        scheduler_socket_path=scheduler_socket_path,
    )


def _thread_count() -> int:
    return len(tuple((Path("/proc") / str(os.getpid()) / "task").iterdir()))


def _rss_bytes() -> int:
    fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
    if len(fields) < 2:
        raise RuntimeError("cannot read process RSS from /proc/self/statm")
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def _metric_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _solve_mode(
    mode: ArchitectureMode,
    task: ArchitectureAxisTask,
) -> tuple[ALNSResult, float, dict[str, object]]:
    instance = replace(
        parse_schneider(task.benchmark_dir / f"{task.instance_name}.txt"),
        distance_backend="native",
    )
    fixed_work = task.axis == "fixed_work"
    exact_deadline = (
        ExactDeadlineConfig.fixed_exact_calls(100, watchdog_seconds=120.0)
        if fixed_work
        else ExactDeadlineConfig.wall_clock()
    )
    cpu_before = os.times()
    threads_before = _thread_count()
    started = time.perf_counter()
    common: dict[str, object] = {
        "seed": task.seed,
        "max_iterations": 1000,
        "time_limit_seconds": 120.0 if fixed_work else 30.0,
        "operator_profile": "stage02_constraint_guided",
        "measurement_config": MeasurementConfig(),
        "screening_config": CheapScreeningConfig(),
        "cache_incremental_config": CacheIncrementalConfig(enabled=True),
        "backend": "cpu_batch",
        "batch_size": 128,
        "termination_mode": "fixed_work" if fixed_work else "wall_clock",
        "exact_deadline_config": exact_deadline,
        "stage04_config": Stage04Config(),
    }
    if mode is ArchitectureMode.CURRENT_STAGE052:
        result = solve_alns(
            instance,
            **common,  # type: ignore[arg-type]
            native_kernel_config=NativeKernelConfig(),
            candidate_transaction_config=NativeCandidateTransactionConfig(),
        )
    elif mode is ArchitectureMode.PYTHON_CANDIDATE_CONTROL:
        result = solve_alns(
            instance,
            **common,  # type: ignore[arg-type]
            candidate_control_config=CandidateControlConfig(
                worker_count=THREADS_PER_SHARD
            ),
        )
    else:
        result = solve_alns(
            instance,
            **common,  # type: ignore[arg-type]
            native_execution_config=_native_config(
                mode,
                scheduler_socket_path=(
                    task.scheduler_socket_path
                    if mode is ArchitectureMode.HOST_SCHEDULER
                    else None
                ),
            ),
        )
    solver_seconds = time.perf_counter() - started
    cpu_after = os.times()
    report = validate_routes(instance, [list(route) for route in result.routes])
    if not report.feasible or result.objective is None:
        raise RuntimeError("architecture axis returned an invalid or objective-less solution")
    if result.objective.key != (
        report.vehicle_count,
        round(report.total_distance, 9),
        round(report.total_charging_time, 9),
        result.objective.charging_count,
    ):
        raise RuntimeError("architecture axis objective does not replay")
    topology: dict[str, object] = {
        "shard_processes": SHARD_PROCESSES,
        "threads_per_shard": THREADS_PER_SHARD,
        "compute_thread_limit": TOTAL_COMPUTE_THREADS,
        "scheduler_threads": 24 if mode is ArchitectureMode.HOST_SCHEDULER else 0,
        "process_id": os.getpid(),
        "threads_before": threads_before,
        "threads_after": _thread_count(),
        "cpu_seconds": (
            cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
        ),
        "cpu_utilization_percent_of_one_core": (
            100.0
            * (
                cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system
            )
            / max(solver_seconds, 1e-12)
        ),
        "rss_bytes": _rss_bytes(),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    return result, solver_seconds, topology


def _result_payload(
    task: ArchitectureAxisTask,
    mode: ArchitectureMode,
    result: ALNSResult,
    solver_seconds: float,
    topology: dict[str, object],
) -> dict[str, object]:
    assert result.objective is not None
    backend = result.backend_metrics
    screening = result.screening_statistics
    candidate_transactions = result.candidate_transaction_statistics
    return {
        "schema_version": SCHEMA_VERSION,
        "run_label": task.run_labels[mode.value],
        "scope": task.scope,
        "repeat": task.repeat,
        "axis": task.axis,
        "mode": mode.value,
        "instance": task.instance_name,
        "seed": task.seed,
        "status": "completed",
        "revision": task.revision,
        "wheel_sha256": task.wheel_sha256,
        "native_sha256": task.native_sha256,
        "solver_seconds": solver_seconds,
        "objective": list(result.objective.key),
        "routes": [list(route) for route in result.routes],
        "customer_sequences": [list(route) for route in result.customer_sequences],
        "validator_passed": True,
        "iterations": result.iterations,
        "effective_iterations": result.effective_iterations,
        "accepted_moves": result.accepted_moves,
        "rejected_moves": result.rejected_moves,
        "exact_started_calls": result.exact_started_calls,
        "exact_completed_calls": result.exact_completed_calls,
        "exact_interrupted_calls": result.exact_interrupted_calls,
        "candidate_work_hash": result.candidate_work_hash,
        "route_result_hash": result.route_result_hash,
        "trajectory": [dict(event) for event in result.neighborhood_events],
        "stage04_events": [dict(event) for event in result.stage04_event_log],
        "candidate_transaction_events": [
            dict(event) for event in result.candidate_transaction_events
        ],
        "candidate_control_statistics": result.candidate_control_statistics,
        "candidate_transaction_statistics": candidate_transactions,
        "native_execution_statistics": result.native_execution_statistics,
        "backend_metrics": backend,
        "screening_statistics": screening,
        "cache_incremental_statistics": result.cache_incremental_statistics,
        "topology": topology,
        "throughput": {
            "effective_iterations_per_second": result.effective_iterations
            / max(solver_seconds, 1e-12),
            "exact_started_per_second": result.exact_started_calls
            / max(solver_seconds, 1e-12),
            "candidate_transactions_per_second": (
                _metric_int(candidate_transactions.get("transactions", 0))
                + _metric_int(
                    candidate_transactions.get("native_candidate_transactions", 0)
                )
            )
            / max(solver_seconds, 1e-12),
            "screened_routes_per_second": (
                _metric_int(screening.get("total_routes", 0))
                or _metric_int(screening.get("screening_calls", 0))
            )
            / max(solver_seconds, 1e-12),
        },
        "cache_memory_bytes": _metric_int(
            result.cache_incremental_statistics.get("estimated_memory_bytes", 0)
        ),
    }


def _axis_path(task: ArchitectureAxisTask, mode: ArchitectureMode) -> Path:
    return (
        task.output_root
        / task.run_labels[mode.value]
        / "axes"
        / f"repeat{task.repeat + 1}"
        / task.axis
        / task.instance_name
        / f"{task.seed}.json"
    )


def _run_group(task: ArchitectureAxisTask) -> list[str]:
    written: list[str] = []
    for mode in rotated_modes(task):
        path = _axis_path(task, mode)
        persistence_started = time.perf_counter()
        try:
            result, solver_seconds, topology = _solve_mode(mode, task)
            payload = _result_payload(task, mode, result, solver_seconds, topology)
        except BaseException as error:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "run_label": task.run_labels[mode.value],
                "scope": task.scope,
                "repeat": task.repeat,
                "axis": task.axis,
                "mode": mode.value,
                "instance": task.instance_name,
                "seed": task.seed,
                "status": "failed",
                "revision": task.revision,
                "wheel_sha256": task.wheel_sha256,
                "native_sha256": task.native_sha256,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        payload["persistence_seconds"] = time.perf_counter() - persistence_started
        _set_artifact_size(payload)
        observed_bytes = _write_signed_json(path, payload)
        if observed_bytes != payload["artifact_bytes"]:
            raise RuntimeError("artifact byte count does not reconcile")
        written.append(str(path))
    return written


def run_experiment(
    scope: str,
    *,
    attempt: int,
    output_root: Path,
    wheel_path: Path,
    max_workers: int = SHARD_PROCESSES,
) -> dict[str, object]:
    root = repository_root()
    if max_workers != SHARD_PROCESSES:
        raise ValueError("Stage 5.2 comparison requires exactly six shard processes")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("native architecture experiment requires a clean worktree")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not wheel_path.is_file():
        raise FileNotFoundError("the frozen comparison wheel does not exist")
    from evrptw import _core as native_core

    native_path = Path(str(native_core.__file__))
    labels = run_labels_for_scope(scope, attempt)
    for label in labels.values():
        if (output_root / label).exists():
            raise FileExistsError(f"run label already exists and cannot be reused: {label}")
    scheduler_path = output_root / f".native-scheduler-{scope}-attempt{attempt:02d}.sock"
    plan = build_axis_plan(
        scope,
        attempt=attempt,
        benchmark_dir=root / "data" / "schneider",
        output_root=output_root,
        scheduler_socket_path=str(scheduler_path),
        wheel_sha256=_sha256_path(wheel_path),
        native_sha256=_sha256_path(native_path),
        revision=revision,
    )
    for label in labels.values():
        (output_root / label).mkdir(parents=True)
    started = time.time()
    written: list[str] = []
    with (
        NativeHostScheduler(scheduler_path, worker_threads=24),
        ProcessPoolExecutor(max_workers=max_workers) as executor,
    ):
        futures = [executor.submit(_run_group, task) for task in plan]
        for future in as_completed(futures):
            written.extend(future.result())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "attempt": attempt,
        "run_labels": labels,
        "revision": revision,
        "wheel_path": str(wheel_path.resolve()),
        "wheel_sha256": _sha256_path(wheel_path),
        "native_path": str(native_path.resolve()),
        "native_sha256": _sha256_path(native_path),
        "axis_count": len(written),
        "expected_axis_count": expected_axis_count(scope),
        "started_unix": started,
        "completed_unix": time.time(),
        "topology": {
            "shard_processes": SHARD_PROCESSES,
            "threads_per_shard": THREADS_PER_SHARD,
            "host_scheduler_threads": 24,
            "compute_thread_limit": TOTAL_COMPUTE_THREADS,
        },
        "mode_order_policy": "rotated_by_repeat_instance_seed_axis",
        "formal_started": False,
    }
    if manifest["axis_count"] != manifest["expected_axis_count"]:
        raise RuntimeError("native architecture experiment axis count is incomplete")
    for label in labels.values():
        _write_signed_json(output_root / label / "run_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("paired", "pilot"))
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--wheel", type=Path, required=True)
    arguments = parser.parse_args(argv)
    manifest = run_experiment(
        arguments.scope,
        attempt=arguments.attempt,
        output_root=arguments.output_root,
        wheel_path=arguments.wheel,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ArchitectureAxisTask",
    "ArchitectureMode",
    "AXIS_NAMES",
    "MODES",
    "PAIRED_INSTANCES",
    "SCHEMA_VERSION",
    "SEEDS",
    "build_axis_plan",
    "expected_axis_count",
    "rotated_modes",
    "run_experiment",
    "run_labels_for_scope",
)
