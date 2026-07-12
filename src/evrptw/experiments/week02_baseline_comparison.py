from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evrptw.baselines.ga_vrptw import solve_ga_vrptw
from evrptw.baselines.ortools_vrptw import solve_vrptw
from evrptw.environment import collect_environment
from evrptw.models import Instance, Node, NodeType, Vehicle


def generate_week02_instance(customer_count: int, *, seed: int = 2014) -> Instance:
    if customer_count <= 0:
        raise ValueError("customer_count must be positive")

    rng = random.Random(seed + customer_count)
    horizon = float(10_000 + customer_count * 100)
    nodes: list[Node] = [
        Node("D0", NodeType.DEPOT, 50.0, 50.0, 0.0, 0.0, horizon, 0.0),
    ]

    station_coordinates = ((20.0, 20.0), (20.0, 80.0), (80.0, 20.0), (80.0, 80.0), (50.0, 50.0))
    for index, (x_coordinate, y_coordinate) in enumerate(station_coordinates, start=1):
        nodes.append(
            Node(
                f"S{index}",
                NodeType.STATION,
                x_coordinate,
                y_coordinate,
                0.0,
                0.0,
                horizon,
                0.0,
            )
        )

    for index in range(1, customer_count + 1):
        nodes.append(
            Node(
                f"C{index}",
                NodeType.CUSTOMER,
                rng.uniform(0.0, 100.0),
                rng.uniform(0.0, 100.0),
                float(rng.randint(1, 5)),
                0.0,
                horizon,
                5.0,
            )
        )

    vehicle = Vehicle(
        battery_capacity=1_000_000.0,
        load_capacity=50.0,
        consumption_rate=1.0,
        inverse_refueling_rate=0.1,
        average_velocity=1.0,
    )
    return Instance(f"week02_generated_{customer_count}", tuple(nodes), vehicle)


def write_schneider_instance(instance: Instance, path: Path) -> None:
    lines = ["StringID Type x y demand ReadyTime DueDate ServiceTime"]
    for node in instance.nodes:
        lines.append(
            f"{node.name} {node.kind.value} {node.x:.6f} {node.y:.6f} "
            f"{node.demand:.6f} {node.ready_time:.6f} {node.due_date:.6f} "
            f"{node.service_time:.6f}"
        )
    vehicle = instance.vehicle
    lines.extend(
        [
            "",
            f"Q / {vehicle.battery_capacity:.6f} /",
            f"C / {vehicle.load_capacity:.6f} /",
            f"r / {vehicle.consumption_rate:.6f} /",
            f"g / {vehicle.inverse_refueling_rate:.6f} /",
            f"v / {vehicle.average_velocity:.6f} /",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recommended_vehicle_count(instance: Instance) -> int:
    total_demand = sum(customer.demand for customer in instance.customers)
    return max(1, math.ceil(total_demand / instance.vehicle.load_capacity) + 1)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _violation_count(payload: dict[str, Any]) -> int:
    violations = payload.get("violations", [])
    return len(violations) if isinstance(violations, list) else 0


def run_week02_comparison(
    *,
    output_dir: Path,
    scales: tuple[int, ...] = (50, 100, 200),
    seed: int = 2014,
    ga_population_size: int = 100,
    ga_generations: int = 200,
    crossover_probability: float = 0.8,
    mutation_probability: float = 0.2,
    ortools_time_limit_seconds: int = 30,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    instance_dir = output_dir / "instances"
    results_csv = output_dir / "week02_results.csv"
    convergence_csv = output_dir / "week02_convergence.csv"
    environment_json = output_dir / "week02_environment.json"

    _write_json(environment_json, collect_environment())

    result_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []

    for scale in scales:
        instance = generate_week02_instance(scale, seed=seed)
        write_schneider_instance(instance, instance_dir / f"{instance.name}.txt")
        vehicle_count = _recommended_vehicle_count(instance)

        ortools_result = solve_vrptw(
            instance,
            vehicle_count=vehicle_count,
            time_limit_seconds=ortools_time_limit_seconds,
        )
        ortools_json = raw_dir / f"{instance.name}_ortools_vrptw.json"
        _write_json(ortools_json, ortools_result)
        result_rows.append(
            {
                "instance": instance.name,
                "size": scale,
                "method": "OR_TOOLS_VRPTW",
                "objective_value": ortools_result["validation_total_distance"],
                "feasibility_status": "feasible"
                if ortools_result["feasible"]
                else "infeasible",
                "runtime_seconds": ortools_result["runtime_seconds"],
                "vehicle_count": len(ortools_result["routes"]),
                "violation_count": _violation_count(ortools_result),
                "raw_json": str(ortools_json),
            }
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
        ga_json = raw_dir / f"{instance.name}_ga_vrptw.json"
        _write_json(ga_json, ga_payload)
        result_rows.append(
            {
                "instance": instance.name,
                "size": scale,
                "method": "GA_VRPTW",
                "objective_value": ga_result.objective_value,
                "feasibility_status": "feasible" if ga_result.feasible else "infeasible",
                "runtime_seconds": ga_result.runtime_seconds,
                "vehicle_count": ga_result.vehicle_count,
                "violation_count": len(ga_result.violations),
                "raw_json": str(ga_json),
            }
        )
        for record in ga_result.convergence:
            row = asdict(record)
            row.update({"instance": instance.name, "size": scale, "method": "GA_VRPTW"})
            convergence_rows.append(row)

    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with results_csv.open("w", encoding="utf-8", newline="") as file_object:
        writer = csv.DictWriter(
            file_object,
            fieldnames=[
                "instance",
                "size",
                "method",
                "objective_value",
                "feasibility_status",
                "runtime_seconds",
                "vehicle_count",
                "violation_count",
                "raw_json",
            ],
        )
        writer.writeheader()
        writer.writerows(result_rows)

    with convergence_csv.open("w", encoding="utf-8", newline="") as file_object:
        writer = csv.DictWriter(
            file_object,
            fieldnames=[
                "instance",
                "size",
                "method",
                "generation",
                "best_objective",
                "average_objective",
                "feasible_rate",
                "best_feasible",
            ],
        )
        writer.writeheader()
        writer.writerows(convergence_rows)

    return {
        "results_csv": results_csv,
        "convergence_csv": convergence_csv,
        "environment_json": environment_json,
        "raw_dir": raw_dir,
        "instance_dir": instance_dir,
    }


def _parse_scales(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week 2 baseline comparisons")
    parser.add_argument("--output-dir", type=Path, default=Path("results/week02"))
    parser.add_argument("--scales", default="50,100,200")
    parser.add_argument("--seed", type=int, default=2014)
    parser.add_argument("--ga-population-size", type=int, default=100)
    parser.add_argument("--ga-generations", type=int, default=200)
    parser.add_argument("--crossover-probability", type=float, default=0.8)
    parser.add_argument("--mutation-probability", type=float, default=0.2)
    parser.add_argument("--ortools-time-limit-seconds", type=int, default=30)
    arguments = parser.parse_args()

    outputs = run_week02_comparison(
        output_dir=arguments.output_dir,
        scales=_parse_scales(arguments.scales),
        seed=arguments.seed,
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
