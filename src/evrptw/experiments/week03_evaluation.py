from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from evrptw.baselines.ga_vrptw import solve_ga_vrptw
from evrptw.baselines.ortools_vrptw import solve_vrptw
from evrptw.environment import collect_environment
from evrptw.experiments.week02_baseline_comparison import (
    generate_week02_instance,
    write_schneider_instance,
)
from evrptw.metrics import RouteMetrics, evaluate_route_metrics
from evrptw.models import Instance, Vehicle
from evrptw.repairs import insert_charging_stations

PER_RUN_FIELDS = [
    "instance",
    "size",
    "method",
    "seed",
    "feasible",
    "objective_value",
    "runtime_seconds",
    "vehicle_count",
    "capacity_violations",
    "time_window_violations",
    "energy_violations",
    "coverage_violations",
    "charging_count",
    "charging_time",
    "first_infeasible_step",
    "total_violation_count",
    "raw_json",
]

SUMMARY_FIELDS = [
    "size",
    "method",
    "runs",
    "feasible_runs",
    "feasible_rate",
    "best_feasible_objective",
    "avg_feasible_objective",
    "objective_stddev",
    "avg_runtime_seconds",
    "avg_total_violation_count",
    "avg_capacity_violations",
    "avg_time_window_violations",
    "avg_energy_violations",
    "avg_charging_count",
    "avg_charging_time",
]


def generate_week03_instance(
    customer_count: int,
    *,
    seed: int,
    battery_capacity: float = 80.0,
) -> Instance:
    base = generate_week02_instance(customer_count, seed=seed)
    vehicle = Vehicle(
        battery_capacity=battery_capacity,
        load_capacity=base.vehicle.load_capacity,
        consumption_rate=base.vehicle.consumption_rate,
        inverse_refueling_rate=base.vehicle.inverse_refueling_rate,
        average_velocity=base.vehicle.average_velocity,
    )
    return replace(base, name=f"week03_generated_{customer_count}_seed_{seed}", vehicle=vehicle)


def run_week03_evaluation(
    *,
    output_dir: Path,
    summary_dir: Path | None = Path("experiments/summaries"),
    scales: tuple[int, ...] = (50, 100, 200),
    seeds: tuple[int, ...] = (2014, 2015, 2016),
    battery_capacity: float = 80.0,
    ga_population_size: int = 100,
    ga_generations: int = 200,
    crossover_probability: float = 0.8,
    mutation_probability: float = 0.2,
    ortools_time_limit_seconds: int = 30,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    instance_dir = output_dir / "instances"
    per_run_csv = output_dir / "week03_per_run_results.csv"
    summary_csv = output_dir / "week03_summary_results.csv"
    failure_cases_md = output_dir / "week03_failure_cases.md"
    environment_json = output_dir / "week03_environment.json"

    _write_json(environment_json, collect_environment())

    rows: list[dict[str, Any]] = []
    for scale in scales:
        for seed in seeds:
            instance = generate_week03_instance(
                scale,
                seed=seed,
                battery_capacity=battery_capacity,
            )
            write_schneider_instance(instance, instance_dir / f"{instance.name}.txt")
            vehicle_count = _recommended_vehicle_count(instance)

            ortools_result = solve_vrptw(
                instance,
                vehicle_count=vehicle_count,
                time_limit_seconds=ortools_time_limit_seconds,
            )
            rows.append(
                _record_baseline_result(
                    instance=instance,
                    seed=seed,
                    method="OR_TOOLS_VRPTW",
                    routes=_as_route_lists(ortools_result["routes"]),
                    runtime_seconds=float(ortools_result["runtime_seconds"]),
                    raw_dir=raw_dir,
                    payload={"baseline": ortools_result},
                )
            )
            rows.append(
                _record_repaired_result(
                    instance=instance,
                    seed=seed,
                    method="OR_TOOLS_VRPTW_CHARGING_REPAIR",
                    routes=_as_route_lists(ortools_result["routes"]),
                    baseline_runtime_seconds=float(ortools_result["runtime_seconds"]),
                    raw_dir=raw_dir,
                    payload={"baseline": ortools_result},
                )
            )

            ga_result = solve_ga_vrptw(
                instance,
                seed=seed,
                population_size=ga_population_size,
                generations=ga_generations,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
            )
            ga_payload = ga_result.to_dict()
            ga_routes = [list(route) for route in ga_result.routes]
            rows.append(
                _record_baseline_result(
                    instance=instance,
                    seed=seed,
                    method="GA_VRPTW",
                    routes=ga_routes,
                    runtime_seconds=ga_result.runtime_seconds,
                    raw_dir=raw_dir,
                    payload={"baseline": ga_payload},
                )
            )
            rows.append(
                _record_repaired_result(
                    instance=instance,
                    seed=seed,
                    method="GA_VRPTW_CHARGING_REPAIR",
                    routes=ga_routes,
                    baseline_runtime_seconds=ga_result.runtime_seconds,
                    raw_dir=raw_dir,
                    payload={"baseline": ga_payload},
                )
            )

    _write_csv(per_run_csv, PER_RUN_FIELDS, rows)
    summary_rows = _summary_rows(rows)
    _write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
    _write_failure_cases(failure_cases_md, rows)

    outputs = {
        "per_run_csv": per_run_csv,
        "summary_csv": summary_csv,
        "failure_cases_md": failure_cases_md,
        "environment_json": environment_json,
        "raw_dir": raw_dir,
        "instance_dir": instance_dir,
    }

    if summary_dir is not None:
        summary_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(summary_dir / "week03_per_run_results.csv", PER_RUN_FIELDS, rows)
        _write_csv(summary_dir / "week03_summary_results.csv", SUMMARY_FIELDS, summary_rows)
        outputs["summary_copy_dir"] = summary_dir

    return outputs


def _record_baseline_result(
    *,
    instance: Instance,
    seed: int,
    method: str,
    routes: list[list[str]],
    runtime_seconds: float,
    raw_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    metrics = evaluate_route_metrics(
        instance,
        routes,
        method=method,
        seed=seed,
        runtime_seconds=runtime_seconds,
    )
    raw_json = raw_dir / f"{instance.name}_{method.lower()}.json"
    _write_json(
        raw_json,
        {
            **payload,
            "routes": routes,
            "metrics": metrics.to_row(),
        },
    )
    return _row_with_raw_path(metrics, raw_json)


def _record_repaired_result(
    *,
    instance: Instance,
    seed: int,
    method: str,
    routes: list[list[str]],
    baseline_runtime_seconds: float,
    raw_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    repair = insert_charging_stations(instance, routes)
    repaired_routes = [list(route) for route in repair.routes]
    metrics = evaluate_route_metrics(
        instance,
        repaired_routes,
        method=method,
        seed=seed,
        runtime_seconds=baseline_runtime_seconds + repair.runtime_seconds,
    )
    raw_json = raw_dir / f"{instance.name}_{method.lower()}.json"
    _write_json(
        raw_json,
        {
            **payload,
            "repair": repair.to_dict(),
            "routes": repaired_routes,
            "metrics": metrics.to_row(),
        },
    )
    return _row_with_raw_path(metrics, raw_json)


def _row_with_raw_path(metrics: RouteMetrics, raw_json: Path) -> dict[str, Any]:
    row = metrics.to_row()
    row["raw_json"] = str(raw_json)
    return row


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["size"]), str(row["method"])), []).append(row)

    summaries: list[dict[str, Any]] = []
    for (size, method), group in sorted(grouped.items()):
        feasible = [row for row in group if bool(row["feasible"])]
        feasible_objectives = [float(row["objective_value"]) for row in feasible]
        summaries.append(
            {
                "size": size,
                "method": method,
                "runs": len(group),
                "feasible_runs": len(feasible),
                "feasible_rate": len(feasible) / len(group),
                "best_feasible_objective": _min_or_blank(feasible_objectives),
                "avg_feasible_objective": _mean_or_blank(feasible_objectives),
                "objective_stddev": _stdev_or_blank(feasible_objectives),
                "avg_runtime_seconds": _mean(float(row["runtime_seconds"]) for row in group),
                "avg_total_violation_count": _mean(
                    float(row["total_violation_count"]) for row in group
                ),
                "avg_capacity_violations": _mean(
                    float(row["capacity_violations"]) for row in group
                ),
                "avg_time_window_violations": _mean(
                    float(row["time_window_violations"]) for row in group
                ),
                "avg_energy_violations": _mean(
                    float(row["energy_violations"]) for row in group
                ),
                "avg_charging_count": _mean(float(row["charging_count"]) for row in group),
                "avg_charging_time": _mean(float(row["charging_time"]) for row in group),
            }
        )
    return summaries


def _write_failure_cases(path: Path, rows: list[dict[str, Any]]) -> None:
    failures = [
        row
        for row in rows
        if not bool(row["feasible"]) or int(row["total_violation_count"]) > 0
    ]
    failures.sort(key=lambda row: (int(row["size"]), str(row["method"]), int(row["seed"])))

    lines = [
        "# Week 3 Failure Cases",
        "",
        "Failure cases are retained because infeasible EVRP-TW routes are useful evidence.",
        "",
        "| Instance | Size | Method | Seed | Objective | Violated constraints | "
        "First infeasible step | Diagnosis | Possible fix |",
        "|---|---:|---|---:|---:|---|---|---|---|",
    ]
    for row in failures[:12]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["instance"]),
                    str(row["size"]),
                    str(row["method"]),
                    str(row["seed"]),
                    _format_float(float(row["objective_value"])),
                    _violated_constraint_summary(row),
                    str(row["first_infeasible_step"]) or "n/a",
                    _diagnosis(row),
                    _possible_fix(row),
                ]
            )
            + " |"
        )

    if not failures:
        lines.append("| n/a | 0 | n/a | 0 | 0 | none | n/a | no failures found | n/a |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _violated_constraint_summary(row: dict[str, Any]) -> str:
    parts = []
    for label, field in (
        ("capacity", "capacity_violations"),
        ("time_window", "time_window_violations"),
        ("energy", "energy_violations"),
        ("coverage", "coverage_violations"),
    ):
        count = int(row[field])
        if count:
            parts.append(f"{label}={count}")
    return ", ".join(parts) if parts else "none"


def _diagnosis(row: dict[str, Any]) -> str:
    if int(row["energy_violations"]):
        return "route distance exceeds available battery between charging opportunities"
    if int(row["time_window_violations"]):
        return "arrival time exceeds at least one customer due date"
    if int(row["capacity_violations"]):
        return "route demand exceeds vehicle load capacity"
    if int(row["coverage_violations"]):
        return "customer coverage or route structure is invalid"
    return "feasible route retained for comparison"


def _possible_fix(row: dict[str, Any]) -> str:
    if int(row["energy_violations"]):
        return "add stronger charging insertion or split the route"
    if int(row["time_window_violations"]):
        return "add time-window repair or reorder customers"
    if int(row["capacity_violations"]):
        return "split overloaded route before validation"
    if int(row["coverage_violations"]):
        return "repair customer assignment and route endpoints"
    return "none"


def _as_route_lists(value: object) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    routes: list[list[str]] = []
    for route in value:
        if isinstance(route, list):
            routes.append([str(node) for node in route])
    return routes


def _recommended_vehicle_count(instance: Instance) -> int:
    total_demand = sum(customer.demand for customer in instance.customers)
    return max(1, math.ceil(total_demand / instance.vehicle.load_capacity) + 1)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_object:
        writer = csv.DictWriter(file_object, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def _mean_or_blank(values: list[float]) -> float | str:
    if not values:
        return ""
    return _mean(values)


def _min_or_blank(values: list[float]) -> float | str:
    if not values:
        return ""
    return min(values)


def _stdev_or_blank(values: list[float]) -> float | str:
    if len(values) < 2:
        return ""
    return statistics.stdev(values)


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week 3 EVRP-TW evaluation")
    parser.add_argument("--output-dir", type=Path, default=Path("results/week03"))
    parser.add_argument("--summary-dir", type=Path, default=Path("experiments/summaries"))
    parser.add_argument("--scales", default="50,100,200")
    parser.add_argument("--seeds", default="2014,2015,2016")
    parser.add_argument("--battery-capacity", type=float, default=80.0)
    parser.add_argument("--ga-population-size", type=int, default=100)
    parser.add_argument("--ga-generations", type=int, default=200)
    parser.add_argument("--crossover-probability", type=float, default=0.8)
    parser.add_argument("--mutation-probability", type=float, default=0.2)
    parser.add_argument("--ortools-time-limit-seconds", type=int, default=30)
    arguments = parser.parse_args()

    outputs = run_week03_evaluation(
        output_dir=arguments.output_dir,
        summary_dir=arguments.summary_dir,
        scales=_parse_int_tuple(arguments.scales),
        seeds=_parse_int_tuple(arguments.seeds),
        battery_capacity=arguments.battery_capacity,
        ga_population_size=arguments.ga_population_size,
        ga_generations=arguments.ga_generations,
        crossover_probability=arguments.crossover_probability,
        mutation_probability=arguments.mutation_probability,
        ortools_time_limit_seconds=arguments.ortools_time_limit_seconds,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
