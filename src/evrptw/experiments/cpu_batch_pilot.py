"""Independent CPU exact-charging batch evidence generator.

This pilot is not Stage 3.3.  It compares the scalar reference with ordered
CPU batching using identical fixed work, screening and bounded route-cache
semantics.  Raw evidence stays below ``results/`` and only the review command
may publish a tracked summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evrptw.alns import ALNSResult, solve_alns
from evrptw.cache_incremental import CacheIncrementalConfig
from evrptw.charging import ChargingSubproblemResult
from evrptw.cpu_batch import ExactChargingBackend, solve_exact_charging_batch
from evrptw.environment import collect_environment
from evrptw.experiments.stage02_route_reduction import FORMAL_SEEDS
from evrptw.experiments.stage02_route_reduction import load_config as load_stage02_config
from evrptw.measurement import CheapScreeningConfig, MeasurementConfig, canonical_route_key
from evrptw.models import Instance
from evrptw.parser import parse_schneider
from evrptw.validation import validate_routes

SCHEMA_VERSION = "cpu-batch-pilot-v1"
RUN_LABEL = "cpu_batch_pilot_attempt01"
PILOT_INSTANCES = ("c101C5", "c101_21", "r101_21", "rc101_21")
PILOT_SEEDS = tuple(FORMAL_SEEDS)
HUNDRED_CUSTOMER_INSTANCES = frozenset({"c101_21", "r101_21", "rc101_21"})
DEFAULT_BATCH_SIZE = 128
DEFAULT_WARMUPS = 3
DEFAULT_MEASUREMENTS = 5
DEFAULT_FIXED_ITERATIONS = 40
DEFAULT_WATCHDOG_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class PilotConfig:
    root: Path
    benchmark_dir: Path
    stage02_config: Path
    output_dir: Path
    summary_dir: Path
    instances: tuple[str, ...] = PILOT_INSTANCES
    seeds: tuple[int, ...] = PILOT_SEEDS
    batch_size: int = DEFAULT_BATCH_SIZE
    warmups: int = DEFAULT_WARMUPS
    measurements: int = DEFAULT_MEASUREMENTS
    fixed_iterations: int = DEFAULT_FIXED_ITERATIONS
    watchdog_seconds: float = DEFAULT_WATCHDOG_SECONDS

    @classmethod
    def defaults(cls, root: Path | None = None) -> PilotConfig:
        resolved_root = root or Path(__file__).resolve().parents[3]
        return cls(
            root=resolved_root,
            benchmark_dir=resolved_root / "data" / "schneider",
            stage02_config=resolved_root / "configs" / "stage02_constraint_guided.toml",
            output_dir=resolved_root / "results" / RUN_LABEL,
            summary_dir=resolved_root / "experiments" / "summaries",
        )


def run_replay(config: PilotConfig) -> dict[str, Path]:
    """Create scalar candidate-work manifests and replay both CPU backends."""

    _validate_config(config)
    _prepare_output(config.output_dir, require_empty=True)
    stage02 = load_stage02_config(config.stage02_config)
    rows: list[dict[str, object]] = []
    for instance_name in config.instances:
        instance = _load_instance(config, instance_name)
        for seed in config.seeds:
            reference = _run_fixed_work(
                instance,
                seed,
                config,
                stage02.vehicle_operator_config,
                ExactChargingBackend.CPU_SCALAR,
            )
            if (
                not reference.feasible
                or reference.watchdog_triggered
                or reference.effective_iterations != config.fixed_iterations
                or reference.measurement_trace is None
                or reference.measurement_trace.reconcile(reference)["status"] != "pass"
            ):
                raise RuntimeError(
                    f"incomplete scalar candidate work for {instance_name}/{seed}"
                )
            work_items = _work_items(reference, seed)
            work_hash = _candidate_work_hash(work_items, seed)
            manifest_path = (
                config.output_dir / "candidate_work" / f"{instance_name}_{seed}.json"
            )
            _write_json(
                manifest_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "instance": instance_name,
                    "seed": seed,
                    "candidate_work_hash": work_hash,
                    "requests": work_items,
                },
            )
            orders = tuple(_order_from_item(item) for item in work_items)
            scalar_reference = solve_exact_charging_batch(
                instance,
                orders,
                backend=ExactChargingBackend.CPU_SCALAR,
                batch_size=1,
            ).results
            for backend, batch_size in (
                (ExactChargingBackend.CPU_SCALAR, 1),
                (ExactChargingBackend.CPU_BATCH, config.batch_size),
            ):
                for _ in range(config.warmups):
                    solve_exact_charging_batch(
                        instance,
                        orders,
                        backend=backend,
                        batch_size=batch_size,
                    )
                samples: list[float] = []
                latest_metrics: dict[str, object] = {}
                correctness = True
                for _ in range(config.measurements):
                    replay = solve_exact_charging_batch(
                        instance,
                        orders,
                        backend=backend,
                        batch_size=batch_size,
                    )
                    samples.append(replay.metrics.total_seconds)
                    latest_metrics = replay.metrics.to_dict()
                    correctness = correctness and _results_equal(
                        replay.results,
                        scalar_reference,
                    )
                rows.append(
                    {
                        "instance": instance_name,
                        "seed": seed,
                        "backend": backend.value,
                        "batch_size": batch_size,
                    "request_count": len(orders),
                    "warmups": config.warmups,
                    "measurements": config.measurements,
                        "candidate_work_hash": work_hash,
                        "correctness": correctness,
                        "total_seconds_median": statistics.median(samples),
                        "total_seconds_samples": samples,
                        "metrics": latest_metrics,
                    }
                )
    replay_path = config.output_dir / "replay_results.json"
    _write_json(replay_path, {"schema_version": SCHEMA_VERSION, "rows": rows})
    gate = _evaluate_replay_gate(rows, config)
    gate_path = config.output_dir / "replay_gate.json"
    _write_json(gate_path, gate)
    _write_json(config.output_dir / "environment.json", collect_environment())
    return {"replay_results": replay_path, "replay_gate": gate_path}


def run_paired(config: PilotConfig) -> dict[str, Path]:
    """Run fixed-work end-to-end scalar/batch pairs after replay passes."""

    gate = _read_json(config.output_dir / "replay_gate.json")
    if gate.get("status") != "PASS_REPLAY":
        raise RuntimeError("paired CPU benchmark is blocked because replay did not pass")
    stage02 = load_stage02_config(config.stage02_config)
    rows: list[dict[str, object]] = []
    for instance_name in config.instances:
        instance = _load_instance(config, instance_name)
        for seed in config.seeds:
            manifest = _read_json(
                config.output_dir / "candidate_work" / f"{instance_name}_{seed}.json"
            )
            expected_work_hash = str(manifest["candidate_work_hash"])
            scalar = _run_fixed_work(
                instance,
                seed,
                config,
                stage02.vehicle_operator_config,
                ExactChargingBackend.CPU_SCALAR,
            )
            batch = _run_fixed_work(
                instance,
                seed,
                config,
                stage02.vehicle_operator_config,
                ExactChargingBackend.CPU_BATCH,
            )
            scalar_summary = _alns_summary(scalar, instance, seed)
            batch_summary = _alns_summary(batch, instance, seed)
            valid = all(
                (
                    scalar_summary["candidate_work_hash"] == expected_work_hash,
                    batch_summary["candidate_work_hash"] == expected_work_hash,
                    scalar_summary["exact_calls"] == batch_summary["exact_calls"],
                    scalar_summary["route_result_hash"]
                    == batch_summary["route_result_hash"],
                    scalar_summary["objective_key"] == batch_summary["objective_key"],
                    scalar_summary["effective_iterations"]
                    == batch_summary["effective_iterations"]
                    == config.fixed_iterations,
                    scalar_summary["validator_feasible"] is True,
                    batch_summary["validator_feasible"] is True,
                    scalar_summary["trace_reconciliation_status"] == "pass",
                    batch_summary["trace_reconciliation_status"] == "pass",
                    scalar_summary["watchdog_triggered"] is False,
                    batch_summary["watchdog_triggered"] is False,
                )
            )
            scalar_seconds = _as_float(scalar_summary["runtime_seconds"])
            batch_seconds = _as_float(batch_summary["runtime_seconds"])
            saving = (
                (scalar_seconds - batch_seconds) / scalar_seconds
                if scalar_seconds > 0.0
                else float("-inf")
            )
            rows.append(
                {
                    "instance": instance_name,
                    "seed": seed,
                    "valid": valid,
                    "fixed_iterations": config.fixed_iterations,
                    "scalar_runtime_seconds": scalar_seconds,
                    "batch_runtime_seconds": batch_seconds,
                    "saving_fraction": saving,
                    "speedup": scalar_seconds / batch_seconds if batch_seconds > 0 else 0.0,
                    "scalar": scalar_summary,
                    "batch": batch_summary,
                }
            )
    paired_path = config.output_dir / "paired_results.json"
    _write_json(paired_path, {"schema_version": SCHEMA_VERSION, "rows": rows})
    gate_payload = _runner_paired_gate(rows)
    gate_path = config.output_dir / "paired_gate.json"
    _write_json(gate_path, gate_payload)
    manifest_path = _finalize_raw_manifest(config.output_dir, gate_payload["status"])
    return {
        "paired_results": paired_path,
        "paired_gate": gate_path,
        "raw_manifest": manifest_path,
    }


def _runner_paired_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    """Apply the predeclared correctness and end-to-end adoption gate."""

    expected_keys = {
        (instance, seed)
        for instance in PILOT_INSTANCES
        for seed in PILOT_SEEDS
    }
    rows_by_key: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row.get("instance")), _as_int(row.get("seed")))
        rows_by_key.setdefault(key, []).append(row)
    scope_complete = set(rows_by_key) == expected_keys and all(
        len(group) == 1 for group in rows_by_key.values()
    )
    all_valid = scope_complete and all(bool(row.get("valid")) for row in rows)
    family_savings: dict[str, float] = {}
    for instance in sorted(HUNDRED_CUSTOMER_INSTANCES):
        values = [
            _as_float(row["saving_fraction"])
            for row in rows
            if row.get("instance") == instance and bool(row.get("valid"))
        ]
        if values:
            family_savings[instance] = statistics.median(values)
    median_family_saving = (
        statistics.median(family_savings.values())
        if len(family_savings) == len(HUNDRED_CUSTOMER_INSTANCES)
        else float("-inf")
    )
    positive_families = sum(value > 0.0 for value in family_savings.values())
    protocol_complete = scope_complete and all(
        _as_int(row.get("fixed_iterations")) == DEFAULT_FIXED_ITERATIONS
        for row in rows
    )
    adopt = (
        all_valid
        and protocol_complete
        and median_family_saving >= 0.10
        and positive_families >= 2
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "ADOPT_CPU_BATCH_DEFAULT" if adopt else "KEEP_CPU_SCALAR_DEFAULT"
        ),
        "all_pairs_valid": all_valid,
        "protocol_complete": protocol_complete,
        "family_median_saving_fraction": family_savings,
        "median_family_saving_fraction": median_family_saving,
        "positive_families": positive_families,
        "criterion": (
            "all pairs valid; median of three 100-customer family medians >= 0.10; "
            "at least two families improve"
        ),
    }


def _run_fixed_work(
    instance: Instance,
    seed: int,
    config: PilotConfig,
    vehicle_operator_config: Any,
    backend: ExactChargingBackend,
) -> ALNSResult:
    return solve_alns(
        instance,
        seed=seed,
        max_iterations=config.fixed_iterations,
        time_limit_seconds=config.watchdog_seconds,
        operator_profile="stage02_constraint_guided",
        vehicle_operator_config=vehicle_operator_config,
        measurement_config=MeasurementConfig(),
        screening_config=CheapScreeningConfig(),
        cache_incremental_config=CacheIncrementalConfig(
            enabled=True,
            max_entries=50_000,
            max_memory_bytes=512 * 1024 * 1024,
        ),
        backend=backend,
        batch_size=(1 if backend is ExactChargingBackend.CPU_SCALAR else config.batch_size),
        termination_mode="fixed_work",
    )


def _work_items(result: ALNSResult, seed: int) -> list[dict[str, object]]:
    trace = result.measurement_trace
    if trace is None:
        raise RuntimeError("candidate-work manifest requires a Stage03Trace")
    items: list[dict[str, object]] = []
    for record in trace.route_evaluations:
        if record.kind != "exact_call":
            continue
        sequence = trace.route_dictionary[record.route_key]
        if canonical_route_key(sequence) != record.route_key:
            raise RuntimeError(f"trace route key is not canonical: {record.route_key}")
        items.append(
            {
                "request_order": len(items),
                "seed": seed,
                "iteration": record.iteration,
                "lane": record.lane,
                "operator": record.operator,
                "route_sequence": list(sequence),
                "canonical_route_key": record.route_key,
            }
        )
    if not items:
        raise RuntimeError("candidate-work manifest contains no exact charging calls")
    return items


def _candidate_work_hash(items: list[dict[str, object]], seed: int) -> str:
    return _payload_hash({"seed": seed, "requests": items})


def _alns_summary(result: ALNSResult, instance: Instance, seed: int) -> dict[str, object]:
    validator = validate_routes(instance, [list(route) for route in result.routes])
    trace = result.measurement_trace
    return {
        "runtime_seconds": result.runtime_seconds,
        "exact_charging_seconds": result.charging_subproblem_time,
        "exact_calls": result.charging_subproblem_calls,
        "effective_iterations": result.effective_iterations,
        "watchdog_triggered": result.watchdog_triggered,
        "objective_key": list(result.objective.key) if result.objective is not None else [],
        "route_result_hash": _payload_hash(
            {
                "routes": [list(route) for route in result.routes],
                "customer_sequences": [list(route) for route in result.customer_sequences],
            }
        ),
        "candidate_work_hash": _candidate_work_hash(_work_items(result, seed), seed),
        "validator_feasible": validator.feasible,
        "trace_reconciliation_status": (
            str(trace.reconcile(result)["status"]) if trace is not None else "missing"
        ),
        "backend_metrics": result.backend_metrics,
        "routes": [list(route) for route in result.routes],
        "customer_sequences": [list(route) for route in result.customer_sequences],
    }


def _evaluate_replay_gate(
    rows: list[dict[str, object]],
    config: PilotConfig,
) -> dict[str, object]:
    correctness = all(bool(row.get("correctness")) for row in rows)
    savings: list[float] = []
    for instance in HUNDRED_CUSTOMER_INSTANCES:
        for seed in PILOT_SEEDS:
            matches = [
                row
                for row in rows
                if row.get("instance") == instance and _as_int(row.get("seed")) == seed
            ]
            scalar = next(
                (
                    _as_float(row["total_seconds_median"])
                    for row in matches
                    if row.get("backend") == ExactChargingBackend.CPU_SCALAR.value
                ),
                None,
            )
            batch = next(
                (
                    _as_float(row["total_seconds_median"])
                    for row in matches
                    if row.get("backend") == ExactChargingBackend.CPU_BATCH.value
                ),
                None,
            )
            if scalar is not None and batch is not None and scalar > 0:
                savings.append((scalar - batch) / scalar)
    positive_signal = bool(savings) and statistics.median(savings) > 0.0
    protocol_complete = (
        config.instances == PILOT_INSTANCES
        and config.seeds == PILOT_SEEDS
        and config.warmups == DEFAULT_WARMUPS
        and config.measurements == DEFAULT_MEASUREMENTS
        and config.fixed_iterations == DEFAULT_FIXED_ITERATIONS
    )
    if correctness and positive_signal:
        status = "PASS_REPLAY" if protocol_complete else "PASS_REPLAY_SMOKE"
    else:
        status = "STOP_BEFORE_PAIRED"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "correctness": correctness,
        "protocol_complete": protocol_complete,
        "median_100_customer_saving_fraction": (
            statistics.median(savings) if savings else float("-inf")
        ),
    }


def _results_equal(
    actual: tuple[ChargingSubproblemResult, ...],
    expected: tuple[ChargingSubproblemResult, ...],
) -> bool:
    return len(actual) == len(expected) and all(
        _charging_result_hash(left) == _charging_result_hash(right)
        for left, right in zip(actual, expected, strict=True)
    )


def _charging_result_hash(result: ChargingSubproblemResult) -> str:
    return _payload_hash(
        {
            "feasible": result.feasible,
            "route": list(result.route),
            "distance": result.distance,
            "total_energy": result.total_energy,
            "charged_energy": result.charged_energy,
            "charging_time": result.charging_time,
            "failure_reason": result.failure_reason,
        }
    )


def _order_from_item(item: dict[str, object]) -> tuple[str, ...]:
    raw = item.get("route_sequence")
    if not isinstance(raw, list):
        raise ValueError("candidate-work route_sequence must be a list")
    order = tuple(str(name) for name in raw)
    if item.get("canonical_route_key") != canonical_route_key(order):
        raise ValueError("candidate-work canonical route key does not match")
    return order


def _load_instance(config: PilotConfig, name: str) -> Instance:
    path = config.benchmark_dir / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return parse_schneider(path)


def _validate_config(config: PilotConfig) -> None:
    if config.batch_size <= 0 or config.warmups < 0 or config.measurements <= 0:
        raise ValueError("CPU batch size and measurement counts must be positive")
    if config.fixed_iterations <= 0 or config.watchdog_seconds <= 0:
        raise ValueError("fixed iterations and watchdog must be positive")
    try:
        config.output_dir.relative_to(config.root / "results")
    except ValueError as error:
        raise ValueError("CPU pilot output must be below results/") from error


def _prepare_output(path: Path, *, require_empty: bool = False) -> None:
    if require_empty and path.exists() and any(path.iterdir()):
        raise FileExistsError(f"CPU pilot output already exists: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _finalize_raw_manifest(output_dir: Path, status: object) -> Path:
    manifest_path = output_dir / "raw_manifest.json"
    sidecar_path = output_dir / "raw_manifest.sha256"
    files: dict[str, dict[str, object]] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path not in {manifest_path, sidecar_path}:
            files[str(path.relative_to(output_dir))] = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
    _write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "run_label": output_dir.name,
            "status": status,
            "stage03_3_claim": False,
            "files": files,
        },
    )
    sidecar_path.write_text(_sha256(manifest_path) + "\n", encoding="utf-8")
    return manifest_path


def _payload_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON payload is not an object: {path}")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected numeric value, got {type(value).__name__}")
    return float(value)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected integer value, got {type(value).__name__}")
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the independent CPU batch pilot")
    parser.add_argument("command", choices=("replay", "paired", "all"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--fixed-iterations", type=int, default=DEFAULT_FIXED_ITERATIONS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--measurements", type=int, default=DEFAULT_MEASUREMENTS)
    arguments = parser.parse_args()
    defaults = PilotConfig.defaults()
    config = PilotConfig(
        root=defaults.root,
        benchmark_dir=(arguments.benchmark_dir or defaults.benchmark_dir).resolve(),
        stage02_config=defaults.stage02_config,
        output_dir=(arguments.output_dir or defaults.output_dir).resolve(),
        summary_dir=(arguments.summary_dir or defaults.summary_dir).resolve(),
        fixed_iterations=arguments.fixed_iterations,
        warmups=arguments.warmups,
        measurements=arguments.measurements,
    )
    if arguments.command in {"replay", "all"}:
        run_replay(config)
    if arguments.command in {"paired", "all"}:
        run_paired(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
