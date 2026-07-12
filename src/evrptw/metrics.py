from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from evrptw.models import Instance, NodeType
from evrptw.validation import validate_routes


@dataclass(frozen=True, slots=True)
class RouteMetrics:
    instance: str
    size: int
    method: str
    seed: int
    feasible: bool
    objective_value: float
    runtime_seconds: float
    vehicle_count: int
    capacity_violations: int
    time_window_violations: int
    energy_violations: int
    coverage_violations: int
    charging_count: int
    charging_time: float
    first_infeasible_step: str
    total_violation_count: int

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_route_metrics(
    instance: Instance,
    routes: list[list[str]],
    *,
    method: str,
    seed: int,
    runtime_seconds: float,
) -> RouteMetrics:
    report = validate_routes(instance, routes)
    route_violations = [
        violation for route_report in report.routes for violation in route_report.violations
    ]
    violations = [*report.violations, *route_violations]
    charging_count, charging_time = _charging_statistics(instance, routes)

    return RouteMetrics(
        instance=instance.name,
        size=len(instance.customers),
        method=method,
        seed=seed,
        feasible=report.feasible,
        objective_value=report.total_distance,
        runtime_seconds=runtime_seconds,
        vehicle_count=len(routes),
        capacity_violations=_count_matching(violations, ("capacity", "load")),
        time_window_violations=_count_matching(
            violations,
            ("due date", "time window", "arrival"),
        ),
        energy_violations=_count_matching(violations, ("battery", "energy")),
        coverage_violations=_count_matching(
            violations,
            ("unvisited", "visited more than once", "unknown", "start and end"),
        ),
        charging_count=charging_count,
        charging_time=charging_time,
        first_infeasible_step=_first_infeasible_step(instance, routes, violations),
        total_violation_count=len(violations),
    )


def _count_matching(violations: list[str], patterns: tuple[str, ...]) -> int:
    lowered = tuple(pattern.lower() for pattern in patterns)
    return sum(
        1
        for violation in violations
        if any(pattern in violation.lower() for pattern in lowered)
    )


def _charging_statistics(instance: Instance, routes: list[list[str]]) -> tuple[int, float]:
    by_name = instance.by_name
    charging_count = 0
    charging_time = 0.0

    for route in routes:
        if not route or route[0] not in by_name:
            continue
        battery = instance.vehicle.battery_capacity
        for origin_name, destination_name in zip(route, route[1:], strict=False):
            if origin_name not in by_name or destination_name not in by_name:
                continue
            origin = by_name[origin_name]
            destination = by_name[destination_name]
            battery -= origin.distance_to(destination) * instance.vehicle.consumption_rate
            if destination.kind is NodeType.STATION:
                recharge = max(0.0, instance.vehicle.battery_capacity - battery)
                charging_time += recharge * instance.vehicle.inverse_refueling_rate
                charging_count += 1
                battery = instance.vehicle.battery_capacity

    return charging_count, charging_time


def _first_infeasible_step(
    instance: Instance,
    routes: list[list[str]],
    solution_violations: list[str],
) -> str:
    by_name = instance.by_name
    depot = instance.depot
    visits: Counter[str] = Counter()

    for route_index, route in enumerate(routes, start=1):
        if len(route) < 2 or route[0] != depot.name or route[-1] != depot.name:
            return f"route {route_index}: route must start and end at depot"
        unknown = [name for name in route if name not in by_name]
        if unknown:
            return f"route {route_index}: unknown node {unknown[0]}"

        route_load = sum(
            by_name[name].demand for name in route if by_name[name].kind is NodeType.CUSTOMER
        )
        if route_load > instance.vehicle.load_capacity + 1e-9:
            return f"route {route_index}: load exceeds capacity"

        battery = instance.vehicle.battery_capacity
        elapsed = max(0.0, depot.ready_time)
        for leg_index, (origin_name, destination_name) in enumerate(
            zip(route, route[1:], strict=False),
            start=1,
        ):
            origin = by_name[origin_name]
            destination = by_name[destination_name]
            distance = origin.distance_to(destination)
            battery -= distance * instance.vehicle.consumption_rate
            elapsed += distance / instance.vehicle.average_velocity
            if battery < -1e-9:
                return (
                    f"route {route_index} leg {leg_index} "
                    f"{origin_name}->{destination_name}: battery depleted"
                )

            elapsed = max(elapsed, destination.ready_time)
            if elapsed > destination.due_date + 1e-9:
                return (
                    f"route {route_index} leg {leg_index} "
                    f"{origin_name}->{destination_name}: due date exceeded"
                )

            if destination.kind is NodeType.CUSTOMER:
                visits[destination.name] += 1
                elapsed += destination.service_time
            elif destination.kind is NodeType.STATION:
                recharge = max(0.0, instance.vehicle.battery_capacity - battery)
                elapsed += recharge * instance.vehicle.inverse_refueling_rate
                battery = instance.vehicle.battery_capacity

    if solution_violations:
        return solution_violations[0]
    return ""
