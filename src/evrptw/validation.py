from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from evrptw.models import Instance, NodeType


@dataclass(frozen=True, slots=True)
class RouteReport:
    route: tuple[str, ...]
    distance: float
    finish_time: float
    total_energy: float
    charged_energy: float
    charging_time: float
    violations: tuple[str, ...]

    @property
    def feasible(self) -> bool:
        return not self.violations


@dataclass(frozen=True, slots=True)
class SolutionReport:
    vehicle_count: int
    total_distance: float
    total_energy: float
    total_charged_energy: float
    total_charging_time: float
    routes: tuple[RouteReport, ...]
    violations: tuple[str, ...]

    @property
    def feasible(self) -> bool:
        return not self.violations and all(route.feasible for route in self.routes)


def validate_routes(
    instance: Instance,
    routes: list[list[str]],
    *,
    claimed_objective: float | None = None,
) -> SolutionReport:
    by_name = instance.by_name
    depot = instance.depot
    visits: Counter[str] = Counter()
    route_reports: list[RouteReport] = []
    solution_violations: list[str] = []

    for raw_route in routes:
        route = tuple(raw_route)
        violations: list[str] = []
        if len(route) < 2 or route[0] != depot.name or route[-1] != depot.name:
            violations.append("route must start and end at the depot")
        unknown = [name for name in route if name not in by_name]
        if unknown:
            violations.append(f"unknown nodes: {', '.join(sorted(set(unknown)))}")
            route_reports.append(RouteReport(route, 0.0, 0.0, 0.0, 0.0, 0.0, tuple(violations)))
            continue
        if depot.name in route[1:-1]:
            violations.append("depot may only appear at route start and end")

        customer_demand = sum(
            by_name[name].demand for name in route if by_name[name].kind is NodeType.CUSTOMER
        )
        if customer_demand > instance.vehicle.load_capacity + 1e-9:
            violations.append(
                f"load {customer_demand:.6g} exceeds capacity "
                f"{instance.vehicle.load_capacity:.6g}"
            )

        battery = instance.vehicle.battery_capacity
        time = max(0.0, depot.ready_time)
        distance = 0.0
        total_energy = 0.0
        charged_energy = 0.0
        charging_time = 0.0
        for leg_index, (origin_name, destination_name) in enumerate(
            zip(route, route[1:], strict=False), start=1
        ):
            origin = by_name[origin_name]
            destination = by_name[destination_name]
            leg_distance = origin.distance_to(destination)
            distance += leg_distance
            leg_energy = leg_distance * instance.vehicle.consumption_rate
            total_energy += leg_energy
            battery -= leg_energy
            time += leg_distance / instance.vehicle.average_velocity

            if battery < -1e-9:
                violations.append(
                    f"battery depleted on leg {leg_index} "
                    f"({origin_name}->{destination_name})"
                )

            time = max(time, destination.ready_time)
            if time > destination.due_date + 1e-9:
                violations.append(
                    f"arrival at {destination_name} ({time:.6g}) exceeds due date "
                    f"{destination.due_date:.6g}"
                )

            if destination.kind is NodeType.CUSTOMER:
                visits[destination.name] += 1
                time += destination.service_time
            elif destination.kind is NodeType.STATION:
                recharge = max(0.0, instance.vehicle.battery_capacity - battery)
                recharge_time = recharge * instance.vehicle.inverse_refueling_rate
                charged_energy += recharge
                charging_time += recharge_time
                time += recharge_time
                battery = instance.vehicle.battery_capacity

        route_reports.append(
            RouteReport(
                route,
                distance,
                time,
                total_energy,
                charged_energy,
                charging_time,
                tuple(violations),
            )
        )

    expected = {customer.name for customer in instance.customers}
    missing = sorted(name for name in expected if visits[name] == 0)
    duplicates = sorted(name for name, count in visits.items() if count > 1)
    if missing:
        solution_violations.append(f"unvisited customers: {', '.join(missing)}")
    if duplicates:
        solution_violations.append(f"customers visited more than once: {', '.join(duplicates)}")

    total_distance = sum(report.distance for report in route_reports)
    if claimed_objective is not None and abs(claimed_objective - total_distance) > 1e-6:
        solution_violations.append(
            f"claimed objective {claimed_objective:.12g} differs from recomputed "
            f"distance {total_distance:.12g}"
        )

    return SolutionReport(
        vehicle_count=len(routes),
        total_distance=total_distance,
        total_energy=sum(report.total_energy for report in route_reports),
        total_charged_energy=sum(report.charged_energy for report in route_reports),
        total_charging_time=sum(report.charging_time for report in route_reports),
        routes=tuple(route_reports),
        violations=tuple(solution_violations),
    )
