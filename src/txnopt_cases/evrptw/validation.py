"""Independent EVRPTW route validation behind the TxnOpt case boundary."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import hypot
from typing import Protocol

_CUSTOMER_KIND = "c"
_STATION_KIND = "f"


class _NodeLike(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> object: ...

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

    @property
    def demand(self) -> float: ...

    @property
    def ready_time(self) -> float: ...

    @property
    def due_date(self) -> float: ...

    @property
    def service_time(self) -> float: ...

class _VehicleLike(Protocol):
    @property
    def battery_capacity(self) -> float: ...

    @property
    def load_capacity(self) -> float: ...

    @property
    def consumption_rate(self) -> float: ...

    @property
    def inverse_refueling_rate(self) -> float: ...

    @property
    def average_velocity(self) -> float: ...


class _InstanceLike(Protocol):
    @property
    def by_name(self) -> Mapping[str, _NodeLike]: ...

    @property
    def depot(self) -> _NodeLike: ...

    @property
    def customers(self) -> tuple[_NodeLike, ...]: ...

    @property
    def vehicle(self) -> _VehicleLike: ...


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


class RouteReportCache:
    """Bounded solve-local LRU for independently reconstructed exact routes."""

    __slots__ = (
        "_capacity",
        "_entries",
        "_evictions",
        "_hits",
        "_instance",
        "_misses",
    )

    def __init__(self, capacity: int = 65_536) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("route report cache capacity must be a positive integer")
        self._capacity = capacity
        self._entries: OrderedDict[tuple[str, ...], RouteReport] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._instance: _InstanceLike | None = None

    @property
    def statistics(self) -> Mapping[str, int]:
        return {
            "route_validation_cache_hits": self._hits,
            "route_validation_cache_misses": self._misses,
            "route_validation_cache_size": len(self._entries),
            "route_validation_cache_capacity": self._capacity,
            "route_validation_cache_evictions": self._evictions,
        }

    def resolve(self, instance: _InstanceLike, route: tuple[str, ...]) -> RouteReport:
        if self._instance is None:
            self._instance = instance
        elif self._instance is not instance:
            raise ValueError("a route report cache cannot cross instance contexts")
        try:
            report = self._entries.pop(route)
        except KeyError:
            report = _validate_route(instance, route)
            self._entries[route] = report
            self._misses += 1
            if len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
                self._evictions += 1
        else:
            self._entries[route] = report
            self._hits += 1
        return report


def validate_routes(
    instance: _InstanceLike,
    routes: Sequence[Sequence[str]],
    *,
    claimed_objective: float | None = None,
    route_report_cache: RouteReportCache | None = None,
) -> SolutionReport:
    by_name = instance.by_name
    visits: Counter[str] = Counter()
    route_reports: list[RouteReport] = []
    solution_violations: list[str] = []

    for raw_route in routes:
        route = tuple(raw_route)
        report = (
            _validate_route(instance, route)
            if route_report_cache is None
            else route_report_cache.resolve(instance, route)
        )
        route_reports.append(report)
        if all(name in by_name for name in route):
            visits.update(
                name
                for name in route
                if str(by_name[name].kind) == _CUSTOMER_KIND
            )

    expected = {customer.name for customer in instance.customers}
    missing = sorted(name for name in expected if visits[name] == 0)
    duplicates = sorted(name for name, count in visits.items() if count > 1)
    if missing:
        solution_violations.append(f"unvisited customers: {', '.join(missing)}")
    if duplicates:
        solution_violations.append(
            f"customers visited more than once: {', '.join(duplicates)}"
        )

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


def _validate_route(instance: _InstanceLike, route: tuple[str, ...]) -> RouteReport:
    by_name = instance.by_name
    depot = instance.depot
    violations: list[str] = []
    if len(route) < 2 or route[0] != depot.name or route[-1] != depot.name:
        violations.append("route must start and end at the depot")
    unknown = [name for name in route if name not in by_name]
    if unknown:
        violations.append(f"unknown nodes: {', '.join(sorted(set(unknown)))}")
        return RouteReport(route, 0.0, 0.0, 0.0, 0.0, 0.0, tuple(violations))
    if depot.name in route[1:-1]:
        violations.append("depot may only appear at route start and end")

    customer_demand = sum(
        by_name[name].demand
        for name in route
        if str(by_name[name].kind) == _CUSTOMER_KIND
    )
    if customer_demand > instance.vehicle.load_capacity + 1e-9:
        violations.append(
            f"load {customer_demand:.6g} exceeds capacity "
            f"{instance.vehicle.load_capacity:.6g}"
        )

    battery = instance.vehicle.battery_capacity
    current_time = max(0.0, depot.ready_time)
    distance = 0.0
    total_energy = 0.0
    charged_energy = 0.0
    charging_time = 0.0
    for leg_index, (origin_name, destination_name) in enumerate(
        zip(route, route[1:], strict=False), start=1
    ):
        origin = by_name[origin_name]
        destination = by_name[destination_name]
        leg_distance = hypot(origin.x - destination.x, origin.y - destination.y)
        distance += leg_distance
        leg_energy = leg_distance * instance.vehicle.consumption_rate
        total_energy += leg_energy
        battery -= leg_energy
        current_time += leg_distance / instance.vehicle.average_velocity

        if battery < -1e-9:
            violations.append(
                f"battery depleted on leg {leg_index} ({origin_name}->{destination_name})"
            )

        current_time = max(current_time, destination.ready_time)
        if current_time > destination.due_date + 1e-9:
            violations.append(
                f"arrival at {destination_name} ({current_time:.6g}) "
                f"exceeds due date {destination.due_date:.6g}"
            )

        if str(destination.kind) == _CUSTOMER_KIND:
            current_time += destination.service_time
        elif str(destination.kind) == _STATION_KIND:
            recharge = max(0.0, instance.vehicle.battery_capacity - battery)
            recharge_time = recharge * instance.vehicle.inverse_refueling_rate
            charged_energy += recharge
            charging_time += recharge_time
            current_time += recharge_time
            battery = instance.vehicle.battery_capacity

    return RouteReport(
        route,
        distance,
        current_time,
        total_energy,
        charged_energy,
        charging_time,
        tuple(violations),
    )


__all__ = ["RouteReport", "RouteReportCache", "SolutionReport", "validate_routes"]
