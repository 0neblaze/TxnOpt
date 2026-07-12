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
from evrptw.experiments.week02_baseline_comparison import write_schneider_instance
from evrptw.experiments.week04_extension import generate_week04_instance
from evrptw.metrics import evaluate_route_metrics
from evrptw.models import Instance
from evrptw.repairs import (
    InfrastructureAugmentationResult,
    augment_charging_infrastructure,
    split_routes_after_anticipatory_repair,
)

PER_RUN_FIELDS = [
    "infrastructure_variant",
    "added_station_count",
    "instance",
    "size",
    "battery_capacity",
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
    "infrastructure_variant",
    "size",
    "method",
    "runs",
    "feasible_runs",
    "feasible_rate",
    "best_feasible_objective",
    "avg_feasible_objective",
    "objective_stddev",
    "avg_runtime_seconds",
    "avg_vehicle_count",
    "avg_added_station_count",
    "avg_total_violation_count",
    "avg_capacity_violations",
    "avg_time_window_violations",
    "avg_energy_violations",
    "avg_charging_count",
    "avg_charging_time",
]


def run_week05_infrastructure_augmentation(
    *,
    output_dir: Path,
    summary_dir: Path | None = Path("experiments/summaries"),
    scales: tuple[int, ...] = (50, 100, 200),
    seeds: tuple[int, ...] = (2014, 2015, 2016),
    battery_capacity: float = 60.0,
    ga_population_size: int = 100,
    ga_generations: int = 200,
    crossover_probability: float = 0.8,
    mutation_probability: float = 0.2,
    ortools_time_limit_seconds: int = 30,
    min_reserve_ratio: float = 0.0,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    instance_dir = output_dir / "instances"
    manifest_dir = output_dir / "manifests"
    per_run_csv = output_dir / "week05_infrastructure_per_run_results.csv"
    summary_csv = output_dir / "week05_infrastructure_summary_results.csv"
    failure_cases_md = output_dir / "week05_infrastructure_failure_cases.md"
    acceptance_md = output_dir / "week05_infrastructure_acceptance.md"
    environment_json = output_dir / "week05_infrastructure_environment.json"

    _write_json(
        environment_json,
        {
            "environment": collect_environment(),
            "experiment": {
                "scales": scales,
                "seeds": seeds,
                "battery_capacity": battery_capacity,
                "ga_population_size": ga_population_size,
                "ga_generations": ga_generations,
                "crossover_probability": crossover_probability,
                "mutation_probability": mutation_probability,
                "ortools_time_limit_seconds": ortools_time_limit_seconds,
                "min_reserve_ratio": min_reserve_ratio,
                "infrastructure_variants": ["original", "augmented"],
                "augmentation_rule": "one midpoint station per structurally unreachable customer",
            },
        },
    )

    rows: list[dict[str, Any]] = []
    for scale in scales:
        for seed in seeds:
            original = generate_week04_instance(
                scale,
                seed=seed,
                battery_capacity=battery_capacity,
            )
            augmented = augment_charging_infrastructure(original)
            augmented_instance = replace(
                augmented.instance,
                name=f"{original.name}_infrastructure_augmented",
            )
            manifest_path = manifest_dir / f"{original.name}_augmentation_manifest.json"
            _write_json(manifest_path, augmented.to_dict())

            rows.extend(
                _run_variant(
                    instance=original,
                    infrastructure_variant="original",
                    augmentation=None,
                    manifest_path=None,
                    seed=seed,
                    raw_dir=raw_dir,
                    instance_dir=instance_dir,
                    ga_population_size=ga_population_size,
                    ga_generations=ga_generations,
                    crossover_probability=crossover_probability,
                    mutation_probability=mutation_probability,
                    ortools_time_limit_seconds=ortools_time_limit_seconds,
                    min_reserve_ratio=min_reserve_ratio,
                )
            )
            rows.extend(
                _run_variant(
                    instance=augmented_instance,
                    infrastructure_variant="augmented",
                    augmentation=augmented,
                    manifest_path=manifest_path,
                    seed=seed,
                    raw_dir=raw_dir,
                    instance_dir=instance_dir,
                    ga_population_size=ga_population_size,
                    ga_generations=ga_generations,
                    crossover_probability=crossover_probability,
                    mutation_probability=mutation_probability,
                    ortools_time_limit_seconds=ortools_time_limit_seconds,
                    min_reserve_ratio=min_reserve_ratio,
                )
            )

    _write_csv(per_run_csv, PER_RUN_FIELDS, rows)
    summary_rows = _summary_rows(rows)
    _write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
    _write_failure_cases(failure_cases_md, rows)
    _write_acceptance_report(acceptance_md, rows)

    outputs = {
        "per_run_csv": per_run_csv,
        "summary_csv": summary_csv,
        "failure_cases_md": failure_cases_md,
        "acceptance_md": acceptance_md,
        "environment_json": environment_json,
        "raw_dir": raw_dir,
        "instance_dir": instance_dir,
        "manifest_dir": manifest_dir,
    }
    if summary_dir is not None:
        summary_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(
            summary_dir / "week05_infrastructure_per_run_results.csv",
            PER_RUN_FIELDS,
            rows,
        )
        _write_csv(
            summary_dir / "week05_infrastructure_summary_results.csv",
            SUMMARY_FIELDS,
            summary_rows,
        )
        outputs["summary_copy_dir"] = summary_dir
    return outputs


def _run_variant(
    *,
    instance: Instance,
    infrastructure_variant: str,
    augmentation: InfrastructureAugmentationResult | None,
    manifest_path: Path | None,
    seed: int,
    raw_dir: Path,
    instance_dir: Path,
    ga_population_size: int,
    ga_generations: int,
    crossover_probability: float,
    mutation_probability: float,
    ortools_time_limit_seconds: int,
    min_reserve_ratio: float,
) -> list[dict[str, Any]]:
    write_schneider_instance(instance, instance_dir / f"{instance.name}.txt")
    added_station_count = 0 if augmentation is None else augmentation.added_station_count
    vehicle_count = _recommended_vehicle_count(instance)

    ortools_result = solve_vrptw(
        instance,
        vehicle_count=vehicle_count,
        time_limit_seconds=ortools_time_limit_seconds,
    )
    rows = [
        _record_split_result(
            instance=instance,
            infrastructure_variant=infrastructure_variant,
            added_station_count=added_station_count,
            method="OR_TOOLS_VRPTW_ANTICIPATORY_SPLIT_REPAIR",
            routes=_as_route_lists(ortools_result["routes"]),
            seed=seed,
            constructor_runtime_seconds=float(ortools_result["runtime_seconds"]),
            raw_dir=raw_dir,
            payload={
                "constructor": ortools_result,
                "augmentation": None if augmentation is None else augmentation.to_dict(),
                "augmentation_manifest": None if manifest_path is None else str(manifest_path),
            },
            min_reserve_ratio=min_reserve_ratio,
        )
    ]

    ga_result = solve_ga_vrptw(
        instance,
        seed=seed,
        population_size=ga_population_size,
        generations=ga_generations,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
    )
    rows.append(
        _record_split_result(
            instance=instance,
            infrastructure_variant=infrastructure_variant,
            added_station_count=added_station_count,
            method="GA_VRPTW_ANTICIPATORY_SPLIT_REPAIR",
            routes=[list(route) for route in ga_result.routes],
            seed=seed,
            constructor_runtime_seconds=ga_result.runtime_seconds,
            raw_dir=raw_dir,
            payload={
                "constructor": ga_result.to_dict(),
                "augmentation": None if augmentation is None else augmentation.to_dict(),
                "augmentation_manifest": None if manifest_path is None else str(manifest_path),
            },
            min_reserve_ratio=min_reserve_ratio,
        )
    )
    return rows


def _record_split_result(
    *,
    instance: Instance,
    infrastructure_variant: str,
    added_station_count: int,
    method: str,
    routes: list[list[str]],
    seed: int,
    constructor_runtime_seconds: float,
    raw_dir: Path,
    payload: dict[str, Any],
    min_reserve_ratio: float,
) -> dict[str, Any]:
    repair = split_routes_after_anticipatory_repair(
        instance,
        routes,
        min_reserve_ratio=min_reserve_ratio,
    )
    repaired_routes = [list(route) for route in repair.routes]
    metrics = evaluate_route_metrics(
        instance,
        repaired_routes,
        method=method,
        seed=seed,
        runtime_seconds=constructor_runtime_seconds + repair.runtime_seconds,
    )
    raw_json = raw_dir / f"{instance.name}_{method.lower()}.json"
    _write_json(
        raw_json,
        {
            **payload,
            "infrastructure_variant": infrastructure_variant,
            "added_station_count": added_station_count,
            "post_processing": repair.to_dict(),
            "routes": repaired_routes,
            "metrics": metrics.to_row(),
        },
    )
    row = metrics.to_row()
    row["infrastructure_variant"] = infrastructure_variant
    row["added_station_count"] = added_station_count
    row["battery_capacity"] = instance.vehicle.battery_capacity
    row["raw_json"] = str(raw_json)
    return row


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                str(row["infrastructure_variant"]),
                int(row["size"]),
                str(row["method"]),
            ),
            [],
        ).append(row)

    summaries: list[dict[str, Any]] = []
    for (variant, size, method), group in sorted(grouped.items()):
        feasible = [row for row in group if bool(row["feasible"])]
        feasible_objectives = [float(row["objective_value"]) for row in feasible]
        summaries.append(
            {
                "infrastructure_variant": variant,
                "size": size,
                "method": method,
                "runs": len(group),
                "feasible_runs": len(feasible),
                "feasible_rate": len(feasible) / len(group),
                "best_feasible_objective": _min_or_blank(feasible_objectives),
                "avg_feasible_objective": _mean_or_blank(feasible_objectives),
                "objective_stddev": _stdev_or_blank(feasible_objectives),
                "avg_runtime_seconds": _mean(float(row["runtime_seconds"]) for row in group),
                "avg_vehicle_count": _mean(float(row["vehicle_count"]) for row in group),
                "avg_added_station_count": _mean(
                    float(row["added_station_count"]) for row in group
                ),
                "avg_total_violation_count": _mean(
                    float(row["total_violation_count"]) for row in group
                ),
                "avg_capacity_violations": _mean(
                    float(row["capacity_violations"]) for row in group),
                "avg_time_window_violations": _mean(
                    float(row["time_window_violations"]) for row in group),
                "avg_energy_violations": _mean(
                    float(row["energy_violations"]) for row in group),
                "avg_charging_count": _mean(float(row["charging_count"]) for row in group),
                "avg_charging_time": _mean(float(row["charging_time"]) for row in group),
            }
        )
    return summaries


def _write_failure_cases(path: Path, rows: list[dict[str, Any]]) -> None:
    failures = [row for row in rows if not bool(row["feasible"])]
    failures.sort(
        key=lambda row: (
            str(row["infrastructure_variant"]),
            int(row["size"]),
            str(row["method"]),
            int(row["seed"]),
        )
    )
    lines = [
        "# Week 5 Infrastructure Augmentation Failure Cases",
        "",
        "Original cases are retained as a structural-infeasibility control. "
        "Augmented cases must be feasible.",
        "",
        "| Variant | Size | Method | Seed | Added stations | Energy violations | "
        "First infeasible step |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in failures:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["infrastructure_variant"]),
                    str(row["size"]),
                    str(row["method"]),
                    str(row["seed"]),
                    str(row["added_station_count"]),
                    str(row["energy_violations"]),
                    str(row["first_infeasible_step"]) or "n/a",
                ]
            )
            + " |"
        )
    if not failures:
        lines.append("| n/a | 0 | n/a | 0 | 0 | 0 | no failures |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_acceptance_report(path: Path, rows: list[dict[str, Any]]) -> None:
    augmented_rows = [row for row in rows if row["infrastructure_variant"] == "augmented"]
    failures = [
        row
        for row in augmented_rows
        if not bool(row["feasible"])
        or any(
            int(row[field])
            for field in (
                "capacity_violations",
                "time_window_violations",
                "energy_violations",
                "coverage_violations",
            )
        )
    ]
    lines = [
        "# Week 5 Infrastructure Augmentation Acceptance",
        "",
        f"- Augmented runs: {len(augmented_rows)}",
        f"- Augmented feasible runs: {sum(bool(row['feasible']) for row in augmented_rows)}",
        f"- Augmented zero-violation failures: {len(failures)}",
    ]
    if failures:
        lines.extend(["", "## Failed augmented runs"])
        lines.extend(
            f"- {row['instance']} / {row['method']} / seed {row['seed']}"
            for row in failures
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        raise RuntimeError("infrastructure augmentation did not reach zero-violation feasibility")

    lines.extend(["", "Result: every augmented run is feasible with zero violations."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recommended_vehicle_count(instance: Instance) -> int:
    total_demand = sum(customer.demand for customer in instance.customers)
    return max(1, math.ceil(total_demand / instance.vehicle.load_capacity) + 1)


def _as_route_lists(value: object) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    return [[str(node) for node in route] for route in value if isinstance(route, list)]


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
    return _mean(values) if values else ""


def _min_or_blank(values: list[float]) -> float | str:
    return min(values) if values else ""


def _stdev_or_blank(values: list[float]) -> float | str:
    return statistics.stdev(values) if len(values) > 1 else ""


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Week 5 infrastructure-augmentation EVRP-TW experiments"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/week05_infrastructure"))
    parser.add_argument("--summary-dir", type=Path, default=Path("experiments/summaries"))
    parser.add_argument("--scales", default="50,100,200")
    parser.add_argument("--seeds", default="2014,2015,2016")
    parser.add_argument("--battery-capacity", type=float, default=60.0)
    parser.add_argument("--ga-population-size", type=int, default=100)
    parser.add_argument("--ga-generations", type=int, default=200)
    parser.add_argument("--crossover-probability", type=float, default=0.8)
    parser.add_argument("--mutation-probability", type=float, default=0.2)
    parser.add_argument("--ortools-time-limit-seconds", type=int, default=30)
    parser.add_argument("--min-reserve-ratio", type=float, default=0.0)
    arguments = parser.parse_args()

    outputs = run_week05_infrastructure_augmentation(
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
        min_reserve_ratio=arguments.min_reserve_ratio,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
