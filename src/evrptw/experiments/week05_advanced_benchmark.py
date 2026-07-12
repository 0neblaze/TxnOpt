from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from evrptw.alns import ALNSResult, solve_alns
from evrptw.baselines.ga_vrptw import solve_ga_vrptw
from evrptw.baselines.ortools_vrptw import solve_vrptw
from evrptw.benchmark import InstanceAudit, audit_instance
from evrptw.bpc import BPCResult, solve_branch_price_and_cut
from evrptw.environment import collect_environment
from evrptw.models import Instance
from evrptw.parser import parse_schneider
from evrptw.validation import SolutionReport, validate_routes

DEFAULT_PRIMARY_INSTANCES = (
    "c101C5", "r105C5", "rc105C5",
    "c104C10", "r103C10", "rc102C10",
    "c106C15", "r105C15", "rc103C15",
    "c101_21", "r101_21", "rc101_21",
)
DEFAULT_SEEDS = (2014, 2015, 2016)

PER_RUN_FIELDS = (
    "experiment_id", "start_utc", "end_utc", "benchmark_type", "instance",
    "source_instance", "algorithm", "algorithm_role", "algorithm_version", "seed",
    "customer_count", "station_count", "vehicle_capacity", "battery_capacity",
    "battery_theoretical_lower_bound", "battery_structural_lower_bound", "battery_ratio",
    "time_window_pressure", "distance_metric", "vehicle_count", "objective_value",
    "total_distance", "total_energy", "total_charged_energy", "total_charging_time",
    "status", "feasible", "proven_optimal", "best_known_value", "relative_gap", "lower_bound",
    "root_lower_bound", "final_lower_bound", "incumbent", "optimality_gap",
    "first_feasible_time", "best_time", "runtime_seconds", "cpu_time_seconds",
    "wall_clock_seconds", "iterations", "accepted_moves", "rejected_moves",
    "alns_destroy_statistics", "alns_repair_statistics", "charging_subproblem_calls",
    "charging_subproblem_average_seconds", "bpc_nodes", "bpc_generated_columns",
    "bpc_active_columns", "bpc_pricing_iterations", "forward_labels", "backward_labels",
    "joined_labels", "labels_pruned", "cuts_added", "peak_memory_mb", "failure_reason",
    "raw_log_path", "solution_path", "threads", "time_limit_seconds",
)


def run_week05_advanced_benchmark(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    instance_names: tuple[str, ...] = DEFAULT_PRIMARY_INSTANCES,
    stress_instance_names: tuple[str, ...] = ("c101C5", "r105C5", "rc105C5"),
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    alns_iterations: int = 1_000,
    time_limit_seconds: float = 30.0,
    ga_population_size: int = 60,
    ga_generations: int = 80,
    include_stress: bool = True,
) -> dict[str, Path]:
    if not seeds:
        raise ValueError("at least one random seed is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    solution_dir = output_dir / "solutions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    solution_dir.mkdir(parents=True, exist_ok=True)

    environment_path = output_dir / "environment.json"
    _write_json(
        environment_path,
        {
            "environment": collect_environment(),
            "ram_bytes": _physical_memory_bytes(),
            "gpu": {"used": False, "device": None},
            "threads": 1,
            "seeds": seeds,
            "time_limit_seconds": time_limit_seconds,
            "install_command": "uv sync --all-groups",
            "run_command": (
                "uv run python -m evrptw.experiments.week05_advanced_benchmark "
                f"--benchmark-dir {benchmark_dir} --output-dir {output_dir}"
            ),
            "environment_variables": {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "")},
        },
    )

    instances = [_load_named_instance(benchmark_dir, name) for name in instance_names]
    audits: list[InstanceAudit] = []
    rows: list[dict[str, Any]] = []
    for instance in instances:
        audit = audit_instance(instance, source_path=_instance_path(benchmark_dir, instance.name))
        _require_structural_feasibility(audit, "Primary")
        audits.append(audit)
        rows.extend(
            _run_method_set(
                instance,
                audit,
                benchmark_type="Primary",
                source_instance=instance.name,
                seeds=seeds,
                alns_iterations=alns_iterations,
                time_limit_seconds=time_limit_seconds,
                ga_population_size=ga_population_size,
                ga_generations=ga_generations,
                raw_dir=raw_dir,
                solution_dir=solution_dir,
            )
        )

    if include_stress:
        stress_sources = [
            instance for instance in instances if instance.name in set(stress_instance_names)
        ]
        missing_stress = set(stress_instance_names) - {item.name for item in stress_sources}
        if missing_stress:
            raise ValueError(
                "stress instances must also be Primary instances: "
                + ", ".join(sorted(missing_stress))
            )
        for original in stress_sources:
            original_audit = audit_instance(original)
            stress_capacity = max(
                original_audit.battery_bounds.theoretical_lower_bound,
                original_audit.battery_bounds.structural_lower_bound * 1.05,
            )
            stress = replace(
                original,
                name=f"{original.name}_battery_stress",
                vehicle=replace(original.vehicle, battery_capacity=stress_capacity),
            )
            audit = audit_instance(stress, source_path=_instance_path(benchmark_dir, original.name))
            _require_structural_feasibility(audit, "Stress")
            audits.append(audit)
            rows.extend(
                _run_method_set(
                    stress,
                    audit,
                    benchmark_type="Stress",
                    source_instance=original.name,
                    seeds=seeds,
                    alns_iterations=alns_iterations,
                    time_limit_seconds=time_limit_seconds,
                    ga_population_size=ga_population_size,
                    ga_generations=ga_generations,
                    raw_dir=raw_dir,
                    solution_dir=solution_dir,
                )
            )

    per_run_path = output_dir / "per_run_results.csv"
    audit_path = output_dir / "instance_audits.csv"
    summary_path = output_dir / "summary_results.csv"
    failure_path = output_dir / "failure_cases.csv"
    _write_csv(per_run_path, PER_RUN_FIELDS, rows)
    _write_csv(audit_path, tuple(audits[0].to_row()), [audit.to_row() for audit in audits])
    _write_csv(summary_path, _summary_fields(), _summarize(rows))
    failure_rows = [row for row in rows if row["status"] != "feasible"]
    _write_csv(failure_path, PER_RUN_FIELDS, failure_rows)
    return {
        "environment": environment_path,
        "instance_audits": audit_path,
        "per_run": per_run_path,
        "summary": summary_path,
        "failures": failure_path,
        "raw_dir": raw_dir,
        "solution_dir": solution_dir,
    }


def _run_method_set(
    instance: Instance,
    audit: InstanceAudit,
    *,
    benchmark_type: str,
    source_instance: str,
    seeds: tuple[int, ...],
    alns_iterations: int,
    time_limit_seconds: float,
    ga_population_size: int,
    ga_generations: int,
    raw_dir: Path,
    solution_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(instance.customers) <= 8:
        bpc = _timed_call(
            partial(
                solve_branch_price_and_cut,
                instance,
                time_limit_seconds=time_limit_seconds,
                max_customers=8,
            )
        )
        bpc_row = _bpc_row(
            instance, audit, benchmark_type, source_instance, bpc, time_limit_seconds
        )
        _persist_row(bpc_row, bpc.value, raw_dir, solution_dir)
        rows.append(bpc_row)
        best_known = (
            bpc.value.incumbent if math.isfinite(bpc.value.incumbent) else float("nan")
        )
    else:
        bpc_row = _not_applicable_bpc_row(
            instance,
            audit,
            benchmark_type,
            source_instance,
            time_limit_seconds,
        )
        _persist_row(bpc_row, {"routes": []}, raw_dir, solution_dir)
        rows.append(bpc_row)
        best_known = float("nan")

    for seed in seeds:
        alns = _timed_call(
            partial(
                solve_alns,
                instance,
                seed=seed,
                max_iterations=alns_iterations,
                time_limit_seconds=time_limit_seconds,
            )
        )
        row = _alns_row(
            instance, audit, benchmark_type, source_instance, seed, alns,
            best_known, time_limit_seconds,
        )
        _persist_row(row, alns.value, raw_dir, solution_dir)
        rows.append(row)

    vehicle_count = len(instance.customers)
    ortools = _timed_call(
        partial(
            solve_vrptw,
            instance,
            vehicle_count=vehicle_count,
            time_limit_seconds=max(1, math.ceil(time_limit_seconds)),
        )
    )
    row = _baseline_row(
        instance, audit, benchmark_type, source_instance, "OR_TOOLS_VRPTW", 0,
        "classical VRPTW baseline; no charging optimization", ortools, best_known,
        time_limit_seconds,
    )
    _persist_row(row, ortools.value, raw_dir, solution_dir)
    rows.append(row)

    for seed in seeds:
        ga = _timed_call(
            partial(
                solve_ga_vrptw,
                instance,
                seed=seed,
                population_size=ga_population_size,
                generations=ga_generations,
                time_limit_seconds=time_limit_seconds,
            )
        )
        row = _baseline_row(
            instance, audit, benchmark_type, source_instance, "GA_VRPTW", seed,
            "weak baseline; chromosome decoder has no exact station insertion", ga,
            best_known, time_limit_seconds,
        )
        _persist_row(row, ga.value, raw_dir, solution_dir)
        rows.append(row)
    return rows


@dataclass(frozen=True, slots=True)
class _TimedValue:
    value: Any
    start_utc: str
    end_utc: str
    wall_seconds: float
    cpu_seconds: float
    peak_memory_mb: float


def _timed_call(function: Callable[[], Any]) -> _TimedValue:
    start_utc = datetime.now(UTC)
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    value = function()
    return _TimedValue(
        value,
        start_utc.isoformat(),
        datetime.now(UTC).isoformat(),
        time.perf_counter() - wall_start,
        time.process_time() - cpu_start,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0),
    )


def _common_row(
    instance: Instance,
    audit: InstanceAudit,
    benchmark_type: str,
    source_instance: str,
    algorithm: str,
    role: str,
    seed: int,
    timed: _TimedValue,
    time_limit_seconds: float,
) -> dict[str, Any]:
    identifier = f"{benchmark_type.lower()}-{instance.name}-{algorithm.lower()}-{seed}"
    bounds = audit.battery_bounds
    return {
        "experiment_id": identifier,
        "start_utc": timed.start_utc,
        "end_utc": timed.end_utc,
        "benchmark_type": benchmark_type,
        "instance": instance.name,
        "source_instance": source_instance,
        "algorithm": algorithm,
        "algorithm_role": role,
        "algorithm_version": _git_revision(),
        "seed": seed,
        "customer_count": len(instance.customers),
        "station_count": len(instance.stations),
        "vehicle_capacity": instance.vehicle.load_capacity,
        "battery_capacity": instance.vehicle.battery_capacity,
        "battery_theoretical_lower_bound": bounds.theoretical_lower_bound,
        "battery_structural_lower_bound": bounds.structural_lower_bound,
        "battery_ratio": bounds.stress_ratio,
        "time_window_pressure": 1.0,
        "distance_metric": audit.distance_metric,
        "cpu_time_seconds": timed.cpu_seconds,
        "wall_clock_seconds": timed.wall_seconds,
        "peak_memory_mb": timed.peak_memory_mb,
        "threads": 1,
        "time_limit_seconds": time_limit_seconds,
    }


def _bpc_row(
    instance: Instance,
    audit: InstanceAudit,
    benchmark_type: str,
    source_instance: str,
    timed: _TimedValue,
    time_limit_seconds: float,
) -> dict[str, Any]:
    result: BPCResult = timed.value
    report = _validate(instance, result.routes, result.objective_value)
    row = _common_row(
        instance, audit, benchmark_type, source_instance, "BRANCH_PRICE_AND_CUT",
        "small-scale exact theoretical reference", 0, timed, time_limit_seconds,
    )
    row.update(_empty_algorithm_fields())
    row.update(_report_fields(report))
    row.update({
        "status": "feasible" if report.feasible else result.status,
        "feasible": report.feasible,
        "proven_optimal": result.proven_optimal,
        "best_known_value": result.incumbent,
        "relative_gap": result.optimality_gap,
        "lower_bound": result.final_lower_bound,
        "root_lower_bound": result.root_lower_bound,
        "final_lower_bound": result.final_lower_bound,
        "incumbent": result.incumbent,
        "optimality_gap": result.optimality_gap,
        "runtime_seconds": result.runtime_seconds,
        "bpc_nodes": result.branch_nodes,
        "bpc_generated_columns": result.generated_columns,
        "bpc_active_columns": result.active_columns,
        "bpc_pricing_iterations": result.pricing_iterations,
        "forward_labels": result.forward_labels,
        "backward_labels": result.backward_labels,
        "joined_labels": result.joined_labels,
        "labels_pruned": result.labels_pruned_capacity + result.labels_pruned_infeasible,
        "cuts_added": result.cuts_added,
        "failure_reason": result.failure_reason,
    })
    return row


def _alns_row(
    instance: Instance,
    audit: InstanceAudit,
    benchmark_type: str,
    source_instance: str,
    seed: int,
    timed: _TimedValue,
    best_known: float,
    time_limit_seconds: float,
) -> dict[str, Any]:
    result: ALNSResult = timed.value
    report = _validate(instance, result.routes, result.objective_value)
    row = _common_row(
        instance, audit, benchmark_type, source_instance, "ALNS_EXACT_CHARGING",
        "primary ALNS-based matheuristic", seed, timed, time_limit_seconds,
    )
    row.update(_empty_algorithm_fields())
    row.update(_report_fields(report))
    row.update({
        "status": "feasible" if report.feasible else "invalid",
        "feasible": report.feasible,
        "proven_optimal": False,
        "best_known_value": best_known,
        "relative_gap": _relative_gap(result.objective_value, best_known),
        "first_feasible_time": result.first_feasible_time,
        "best_time": result.best_time,
        "runtime_seconds": result.runtime_seconds,
        "iterations": result.iterations,
        "accepted_moves": result.accepted_moves,
        "rejected_moves": result.rejected_moves,
        "alns_destroy_statistics": json.dumps(result.destroy_statistics, sort_keys=True),
        "alns_repair_statistics": json.dumps(result.repair_statistics, sort_keys=True),
        "charging_subproblem_calls": result.charging_subproblem_calls,
        "charging_subproblem_average_seconds": (
            result.charging_subproblem_time / result.charging_subproblem_calls
            if result.charging_subproblem_calls else 0.0
        ),
        "forward_labels": result.charging_labels_generated,
        "labels_pruned": result.charging_labels_pruned,
        "failure_reason": result.failure_reason,
    })
    return row


def _baseline_row(
    instance: Instance,
    audit: InstanceAudit,
    benchmark_type: str,
    source_instance: str,
    algorithm: str,
    seed: int,
    role: str,
    timed: _TimedValue,
    best_known: float,
    time_limit_seconds: float,
) -> dict[str, Any]:
    result = timed.value
    routes = result["routes"] if isinstance(result, dict) else result.routes
    claimed = (
        result.get("validation_total_distance")
        if isinstance(result, dict)
        else result.objective_value
    )
    report = _validate(instance, routes, claimed)
    row = _common_row(
        instance, audit, benchmark_type, source_instance, algorithm, role, seed,
        timed, time_limit_seconds,
    )
    row.update(_empty_algorithm_fields())
    row.update(_report_fields(report))
    row.update({
        "status": "feasible" if report.feasible else "invalid",
        "feasible": report.feasible,
        "proven_optimal": False,
        "best_known_value": best_known,
        "relative_gap": _relative_gap(report.total_distance, best_known) if report.feasible else "",
        "runtime_seconds": timed.wall_seconds,
        "failure_reason": "" if report.feasible else _violations(report),
    })
    return row


def _not_applicable_bpc_row(
    instance: Instance,
    audit: InstanceAudit,
    benchmark_type: str,
    source_instance: str,
    time_limit_seconds: float,
) -> dict[str, Any]:
    captured = datetime.now(UTC).isoformat()
    timed = _TimedValue(None, captured, captured, 0.0, 0.0, 0.0)
    row = _common_row(
        instance,
        audit,
        benchmark_type,
        source_instance,
        "BRANCH_PRICE_AND_CUT",
        "small-scale exact theoretical reference",
        0,
        timed,
        time_limit_seconds,
    )
    row.update(_empty_algorithm_fields())
    row.update({
        "vehicle_count": "",
        "objective_value": "",
        "total_distance": "",
        "total_energy": "",
        "total_charged_energy": "",
        "total_charging_time": "",
        "status": "not_applicable",
        "feasible": False,
        "proven_optimal": False,
        "runtime_seconds": 0.0,
        "failure_reason": "exact enumerated-column BPC is limited to at most 8 customers",
    })
    return row


def _validate(instance: Instance, routes: Any, objective: float | None) -> SolutionReport:
    route_lists = [list(route) for route in routes]
    claimed = objective if objective is not None and math.isfinite(float(objective)) else None
    return validate_routes(instance, route_lists, claimed_objective=claimed)


def _report_fields(report: SolutionReport) -> dict[str, Any]:
    return {
        "vehicle_count": report.vehicle_count,
        "objective_value": report.total_distance,
        "total_distance": report.total_distance,
        "total_energy": report.total_energy,
        "total_charged_energy": report.total_charged_energy,
        "total_charging_time": report.total_charging_time,
    }


def _empty_algorithm_fields() -> dict[str, Any]:
    return {field: "" for field in PER_RUN_FIELDS if field not in {
        "experiment_id", "start_utc", "end_utc", "benchmark_type", "instance",
        "source_instance", "algorithm", "algorithm_role", "algorithm_version", "seed",
        "customer_count", "station_count", "vehicle_capacity", "battery_capacity",
        "battery_theoretical_lower_bound", "battery_structural_lower_bound", "battery_ratio",
        "time_window_pressure", "distance_metric", "cpu_time_seconds", "wall_clock_seconds",
        "peak_memory_mb", "threads", "time_limit_seconds", "vehicle_count", "objective_value",
        "total_distance", "total_energy", "total_charged_energy", "total_charging_time",
    }}


def _persist_row(row: dict[str, Any], result: Any, raw_dir: Path, solution_dir: Path) -> None:
    identifier = str(row["experiment_id"])
    raw_path = raw_dir / f"{identifier}.json"
    solution_path = solution_dir / f"{identifier}.json"
    row["raw_log_path"] = str(raw_path.resolve())
    row["solution_path"] = str(solution_path.resolve())
    payload = asdict(result) if hasattr(result, "__dataclass_fields__") else result
    _write_json(raw_path, {"record": row, "solver_result": payload})
    routes = payload.get("routes", []) if isinstance(payload, dict) else []
    _write_json(solution_path, {"instance": row["instance"], "routes": routes})


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["benchmark_type"], row["instance"], row["algorithm"])
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (benchmark, instance, algorithm), group in sorted(groups.items()):
        feasible_values = [float(row["objective_value"]) for row in group if row["feasible"]]
        runtimes = [float(row["runtime_seconds"]) for row in group]
        output.append({
            "benchmark_type": benchmark,
            "instance": instance,
            "algorithm": algorithm,
            "runs": len(group),
            "feasible_runs": len(feasible_values),
            "feasibility_rate": len(feasible_values) / len(group),
            "best": min(feasible_values) if feasible_values else "",
            "mean": statistics.fmean(feasible_values) if feasible_values else "",
            "median": statistics.median(feasible_values) if feasible_values else "",
            "worst": max(feasible_values) if feasible_values else "",
            "standard_deviation": statistics.pstdev(feasible_values) if feasible_values else "",
            "mean_runtime": statistics.fmean(runtimes),
            "gap_to_best_known": _mean_numeric(group, "relative_gap"),
        })
    return output


def _summary_fields() -> tuple[str, ...]:
    return (
        "benchmark_type", "instance", "algorithm", "runs", "feasible_runs",
        "feasibility_rate", "best", "mean", "median", "worst", "standard_deviation",
        "mean_runtime", "gap_to_best_known",
    )


def _mean_numeric(rows: list[dict[str, Any]], field: str) -> float | str:
    values = [float(row[field]) for row in rows if row[field] not in ("", None)]
    return statistics.fmean(values) if values else ""


def _relative_gap(value: float, reference: float) -> float | str:
    if not math.isfinite(value) or not math.isfinite(reference):
        return ""
    return (value - reference) / max(abs(reference), 1e-9)


def _violations(report: SolutionReport) -> str:
    values = [*report.violations]
    values.extend(violation for route in report.routes for violation in route.violations)
    return " | ".join(values)


def _require_structural_feasibility(audit: InstanceAudit, benchmark: str) -> None:
    if not audit.structurally_feasible:
        failures = " | ".join(audit.failures)
        raise ValueError(f"{benchmark} instance {audit.instance} failed audit: {failures}")


def _load_named_instance(directory: Path, name: str) -> Instance:
    return parse_schneider(_instance_path(directory, name))


def _instance_path(directory: Path, name: str) -> Path:
    matches = list(directory.glob(f"{name}.txt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one Schneider file for {name}, found {len(matches)}")
    return matches[0]


def _git_revision() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    return revision + ("+dirty" if dirty.stdout else "")


def _physical_memory_bytes() -> int | None:
    if platform.system() != "Darwin":
        return None
    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"],
        capture_output=True,
        text=True,
        check=False,
    )
    return int(result.stdout) if result.returncode == 0 else None


def _write_json(path: Path, payload: Any) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(rendered, encoding="utf-8")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Week 5 advanced EVRP-TW benchmark")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/schneider"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/week05_advanced"))
    parser.add_argument("--instances", default=",".join(DEFAULT_PRIMARY_INSTANCES))
    parser.add_argument("--stress-instances", default="c101C5,r105C5,rc105C5")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--alns-iterations", type=int, default=1_000)
    parser.add_argument("--time-limit-seconds", type=float, default=30.0)
    parser.add_argument("--ga-population-size", type=int, default=60)
    parser.add_argument("--ga-generations", type=int, default=80)
    parser.add_argument("--skip-stress", action="store_true")
    args = parser.parse_args()
    outputs = run_week05_advanced_benchmark(
        benchmark_dir=args.benchmark_dir,
        output_dir=args.output_dir,
        instance_names=tuple(item.strip() for item in args.instances.split(",") if item.strip()),
        stress_instance_names=tuple(
            item.strip() for item in args.stress_instances.split(",") if item.strip()
        ),
        seeds=tuple(int(item) for item in args.seeds.split(",") if item.strip()),
        alns_iterations=args.alns_iterations,
        time_limit_seconds=args.time_limit_seconds,
        ga_population_size=args.ga_population_size,
        ga_generations=args.ga_generations,
        include_stress=not args.skip_stress,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
