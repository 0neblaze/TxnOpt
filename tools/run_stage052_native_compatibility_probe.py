"""Run the frozen Stage 5.2 compatibility axes without native execution v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from evrptw.alns import solve_alns
from evrptw.cache_incremental import CacheIncrementalConfig
from evrptw.candidate_transaction import NativeCandidateTransactionConfig
from evrptw.exact_deadline import ExactDeadlineConfig
from evrptw.measurement import CheapScreeningConfig, MeasurementConfig
from evrptw.native_kernels import NativeKernelConfig
from evrptw.parser import parse_schneider
from evrptw.stage04 import Stage04Config
from evrptw.validation import validate_routes

INSTANCES = ("c101C5", "c101_21", "r101_21", "rc101_21")
SEEDS = (2014, 2015, 2016)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_solution(
    root: Path,
    instance_name: str,
    seed: int,
) -> tuple[Path, dict[str, object]]:
    matches = tuple(
        root.glob(
            f"batch*/{instance_name}/{seed}/"
            f"stage05.2_benchmark_attempt72_solution_{instance_name}_{seed}.json"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"accepted attempt72 source identity is not unique: {instance_name}/{seed}"
        )
    payload = json.loads(matches[0].read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("accepted attempt72 solution root is invalid")
    axes = payload.get("axes")
    axis = axes.get("wall_clock_30") if isinstance(axes, dict) else None
    if not isinstance(axis, dict):
        raise RuntimeError("accepted attempt72 wall-clock solution is missing")
    return matches[0], axis


def run_probe(
    *,
    benchmark_dir: Path,
    attempt72_root: Path,
    output_path: Path,
    runtime_identity: str,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for instance_name in INSTANCES:
        for seed in SEEDS:
            source_path, source_axis = _source_solution(
                attempt72_root,
                instance_name,
                seed,
            )
            instance = replace(
                parse_schneider(benchmark_dir / f"{instance_name}.txt"),
                distance_backend="native",
            )
            started = time.perf_counter()
            result = solve_alns(
                instance,
                seed=seed,
                max_iterations=1000,
                time_limit_seconds=120.0,
                operator_profile="stage02_constraint_guided",
                measurement_config=MeasurementConfig(),
                screening_config=CheapScreeningConfig(),
                cache_incremental_config=CacheIncrementalConfig(enabled=True),
                backend="cpu_batch",
                batch_size=128,
                termination_mode="fixed_work",
                exact_deadline_config=ExactDeadlineConfig.fixed_exact_calls(
                    100,
                    watchdog_seconds=120.0,
                ),
                stage04_config=Stage04Config(),
                native_kernel_config=NativeKernelConfig(),
                candidate_transaction_config=NativeCandidateTransactionConfig(),
                native_execution_config=None,
            )
            report = validate_routes(instance, [list(route) for route in result.routes])
            if not report.feasible or result.objective is None:
                raise RuntimeError("compatibility probe produced an invalid solution")
            records.append(
                {
                    "instance": instance_name,
                    "seed": seed,
                    "validator_passed": True,
                    "objective": list(result.objective.key),
                    "routes": [list(route) for route in result.routes],
                    "customer_sequences": [
                        list(route) for route in result.customer_sequences
                    ],
                    "candidate_work_hash": result.candidate_work_hash,
                    "route_result_hash": result.route_result_hash,
                    "exact_started_calls": result.exact_started_calls,
                    "exact_completed_calls": result.exact_completed_calls,
                    "effective_iterations": result.effective_iterations,
                    "termination_reason": result.termination_reason,
                    "fallback_count": result.native_execution_statistics.get(
                        "fallback_count", 0
                    ),
                    "solver_seconds": time.perf_counter() - started,
                    "source_solution_path": str(source_path),
                    "source_solution_sha256": _sha256(source_path),
                    "accepted_attempt72_wall_clock_objective": source_axis.get(
                        "objective_key"
                    ),
                }
            )
    payload: dict[str, object] = {
        "schema_version": "stage05.2-native-none-compatibility-v1",
        "runtime_identity": runtime_identity,
        "axis_count": len(records),
        "native_execution_config": None,
        "records": records,
    }
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if output_path.exists() or output_path.with_suffix(
        output_path.suffix + ".sha256"
    ).exists():
        raise FileExistsError(f"compatibility probe output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "\n",
        encoding="ascii",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--attempt72-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-identity", required=True)
    arguments = parser.parse_args(argv)
    payload = run_probe(
        benchmark_dir=arguments.benchmark_dir,
        attempt72_root=arguments.attempt72_root,
        output_path=arguments.output,
        runtime_identity=arguments.runtime_identity,
    )
    print(json.dumps({"axis_count": payload["axis_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
