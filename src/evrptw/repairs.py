from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from evrptw.models import Instance, Node, NodeType
from evrptw.validation import SolutionReport, validate_routes


@dataclass(frozen=True, slots=True)
class ChargingRepairResult:
    routes: tuple[tuple[str, ...], ...]
    inserted_station_count: int
    unrepaired_legs: tuple[str, ...]
    runtime_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RouteSplitRepairResult:
    routes: tuple[tuple[str, ...], ...]
    inserted_station_count: int
    split_count: int
    added_vehicle_count: int
    unrepaired_reasons: tuple[str, ...]
    runtime_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AugmentedStation:
    customer_name: str
    station_name: str
    anchor_name: str
    x: float
    y: float
    original_minimum_closed_energy: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InfrastructureAugmentationResult:
    instance: Instance
    stations: tuple[AugmentedStation, ...]

    @property
    def added_station_count(self) -> int:
        return len(self.stations)

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_name": self.instance.name,
            "added_station_count": self.added_station_count,
            "stations": [station.to_dict() for station in self.stations],
        }


def insert_charging_stations(
    instance: Instance,
    routes: list[list[str]],
) -> ChargingRepairResult:
    started_at = time.perf_counter()
    by_name = instance.by_name
    repaired_routes: list[tuple[str, ...]] = []
    inserted_station_count = 0
    unrepaired_legs: list[str] = []

    for route_index, route in enumerate(routes, start=1):
        if len(route) < 2 or any(node_name not in by_name for node_name in route):
            repaired_routes.append(tuple(route))
            continue

        battery = instance.vehicle.battery_capacity
        repaired = [route[0]]

        for origin_name, destination_name in zip(route, route[1:], strict=False):
            origin = by_name[origin_name]
            destination = by_name[destination_name]
            required_energy = (
                origin.distance_to(destination) * instance.vehicle.consumption_rate
            )

            if battery + 1e-9 < required_energy:
                station_name = _best_station_for_leg(
                    instance,
                    origin_name=origin_name,
                    destination_name=destination_name,
                    available_battery=battery,
                )
                if station_name is None:
                    unrepaired_legs.append(
                        f"route {route_index} {origin_name}->{destination_name}"
                    )
                else:
                    station = by_name[station_name]
                    battery -= origin.distance_to(station) * instance.vehicle.consumption_rate
                    repaired.append(station_name)
                    inserted_station_count += 1
                    battery = instance.vehicle.battery_capacity
                    origin = station
                    required_energy = (
                        origin.distance_to(destination) * instance.vehicle.consumption_rate
                    )

            battery -= required_energy
            repaired.append(destination_name)
            if destination_name in by_name and by_name[destination_name].kind is NodeType.STATION:
                battery = instance.vehicle.battery_capacity

        repaired_routes.append(tuple(repaired))

    return ChargingRepairResult(
        routes=tuple(repaired_routes),
        inserted_station_count=inserted_station_count,
        unrepaired_legs=tuple(unrepaired_legs),
        runtime_seconds=time.perf_counter() - started_at,
    )


def insert_anticipatory_charging_stations(
    instance: Instance,
    routes: list[list[str]],
    *,
    min_reserve_ratio: float = 0.0,
) -> ChargingRepairResult:
    started_at = time.perf_counter()
    by_name = instance.by_name
    repaired_routes: list[tuple[str, ...]] = []
    inserted_station_count = 0
    unrepaired_legs: list[str] = []

    for route_index, route in enumerate(routes, start=1):
        if len(route) < 2 or any(node_name not in by_name for node_name in route):
            repaired_routes.append(tuple(route))
            continue

        battery = instance.vehicle.battery_capacity
        repaired = [route[0]]

        for origin_name, destination_name in zip(route, route[1:], strict=False):
            origin = by_name[origin_name]
            destination = by_name[destination_name]
            required_energy = (
                origin.distance_to(destination) * instance.vehicle.consumption_rate
            )
            battery_after_direct = battery - required_energy

            if _leg_needs_anticipatory_charge(
                instance,
                destination_name=destination_name,
                battery_after_leg=battery_after_direct,
                min_reserve_ratio=min_reserve_ratio,
            ):
                station_name = _best_anticipatory_station_for_leg(
                    instance,
                    origin_name=origin_name,
                    destination_name=destination_name,
                    available_battery=battery,
                    min_reserve_ratio=min_reserve_ratio,
                )
                if station_name is None:
                    unrepaired_legs.append(
                        f"route {route_index} {origin_name}->{destination_name}"
                    )
                else:
                    station = by_name[station_name]
                    battery -= origin.distance_to(station) * instance.vehicle.consumption_rate
                    repaired.append(station_name)
                    inserted_station_count += 1
                    battery = instance.vehicle.battery_capacity
                    origin = station
                    required_energy = (
                        origin.distance_to(destination) * instance.vehicle.consumption_rate
                    )

            battery -= required_energy
            repaired.append(destination_name)
            if destination.kind is NodeType.STATION:
                battery = instance.vehicle.battery_capacity

        repaired_routes.append(tuple(repaired))

    return ChargingRepairResult(
        routes=tuple(repaired_routes),
        inserted_station_count=inserted_station_count,
        unrepaired_legs=tuple(unrepaired_legs),
        runtime_seconds=time.perf_counter() - started_at,
    )


def augment_charging_infrastructure(instance: Instance) -> InfrastructureAugmentationResult:
    """Add one deterministic midpoint charging station for each unreachable customer cycle."""
    safe_nodes = (instance.depot, *instance.stations)
    horizon = max(node.due_date for node in instance.nodes)
    additions: list[Node] = []
    manifest: list[AugmentedStation] = []

    for customer in instance.customers:
        minimum_closed_energy = min(
            (
                safe_start.distance_to(customer) + customer.distance_to(safe_end)
            )
            * instance.vehicle.consumption_rate
            for safe_start in safe_nodes
            for safe_end in safe_nodes
        )
        if minimum_closed_energy <= instance.vehicle.battery_capacity + 1e-9:
            continue

        anchor = min(safe_nodes, key=lambda node: (node.distance_to(customer), node.name))
        anchor_energy = anchor.distance_to(customer) * instance.vehicle.consumption_rate
        if anchor_energy > instance.vehicle.battery_capacity + 1e-9:
            raise ValueError(
                "one midpoint station cannot make the customer cycle energy-feasible; "
                "multiple infrastructure stations are required"
            )
        station_name = f"F_AUG_{customer.name}"
        additions.append(
            Node(
                name=station_name,
                kind=NodeType.STATION,
                x=(anchor.x + customer.x) / 2,
                y=(anchor.y + customer.y) / 2,
                demand=0.0,
                ready_time=0.0,
                due_date=horizon,
                service_time=0.0,
            )
        )
        manifest.append(
            AugmentedStation(
                customer_name=customer.name,
                station_name=station_name,
                anchor_name=anchor.name,
                x=(anchor.x + customer.x) / 2,
                y=(anchor.y + customer.y) / 2,
                original_minimum_closed_energy=minimum_closed_energy,
            )
        )

    return InfrastructureAugmentationResult(
        instance=Instance(
            name=instance.name,
            nodes=(*instance.nodes, *additions),
            vehicle=instance.vehicle,
        ),
        stations=tuple(manifest),
    )


def split_routes_after_anticipatory_repair(
    instance: Instance,
    routes: list[list[str]],
    *,
    min_reserve_ratio: float = 0.0,
) -> RouteSplitRepairResult:
    """Split raw customer routes only when a split strictly reduces energy violations."""
    started_at = time.perf_counter()
    _validate_raw_customer_routes(instance, routes)

    current_raw_routes = tuple(tuple(route) for route in routes)
    current_repair = insert_anticipatory_charging_stations(
        instance,
        [list(route) for route in current_raw_routes],
        min_reserve_ratio=min_reserve_ratio,
    )
    current_report = validate_routes(instance, [list(route) for route in current_repair.routes])
    split_count = 0
    unrepaired_reasons: list[str] = []

    while not current_report.feasible:
        current_counts = _violation_counts(current_report)
        if any(current_counts[name] for name in ("coverage", "capacity", "time_window")):
            unrepaired_reasons.append(
                "no-progress: anticipatory repair has non-energy violations; "
                "route splitting will not accept a structurally weaker candidate"
            )
            break
        if current_counts["energy"] == 0:
            unrepaired_reasons.append(
                "no-progress: solution remains infeasible for an unsupported violation type"
            )
            break

        candidates: list[
            tuple[
                tuple[int, int, int, int, int, float],
                tuple[tuple[str, ...], ...],
                ChargingRepairResult,
                SolutionReport,
            ]
        ] = []
        for route_index, route in enumerate(current_raw_routes):
            customers = route[1:-1]
            for split_index in range(1, len(customers)):
                depot = instance.depot.name
                candidate_raw_routes = (
                    *current_raw_routes[:route_index],
                    (depot, *customers[:split_index], depot),
                    (depot, *customers[split_index:], depot),
                    *current_raw_routes[route_index + 1 :],
                )
                candidate_repair = insert_anticipatory_charging_stations(
                    instance,
                    [list(candidate_route) for candidate_route in candidate_raw_routes],
                    min_reserve_ratio=min_reserve_ratio,
                )
                candidate_report = validate_routes(
                    instance,
                    [list(candidate_route) for candidate_route in candidate_repair.routes],
                )
                candidate_counts = _violation_counts(candidate_report)
                if any(
                    candidate_counts[name] > current_counts[name]
                    for name in ("coverage", "capacity", "time_window")
                ):
                    continue
                if candidate_counts["energy"] >= current_counts["energy"]:
                    continue
                candidates.append(
                    (
                        _candidate_rank(candidate_report),
                        candidate_raw_routes,
                        candidate_repair,
                        candidate_report,
                    )
                )

        if not candidates:
            unrepaired_reasons.append(
                "no-progress: no customer-boundary split strictly reduced energy violations"
            )
            break

        _, current_raw_routes, current_repair, current_report = min(
            candidates,
            key=lambda candidate: candidate[0],
        )
        split_count += 1

    if not current_report.feasible:
        unrepaired_reasons.extend(current_repair.unrepaired_legs)

    return RouteSplitRepairResult(
        routes=current_repair.routes,
        inserted_station_count=current_repair.inserted_station_count,
        split_count=split_count,
        added_vehicle_count=len(current_repair.routes) - len(routes),
        unrepaired_reasons=tuple(dict.fromkeys(unrepaired_reasons)),
        runtime_seconds=time.perf_counter() - started_at,
    )


def _validate_raw_customer_routes(instance: Instance, routes: list[list[str]]) -> None:
    report = validate_routes(instance, routes)
    counts = _violation_counts(report)
    if counts["coverage"] or counts["capacity"] or counts["time_window"]:
        raise ValueError(
            "route splitting requires complete raw VRPTW routes without coverage, capacity, "
            "or time-window violations"
        )

    by_name = instance.by_name
    depot = instance.depot.name
    for route in routes:
        if route[0] != depot or route[-1] != depot:
            raise ValueError("route splitting requires routes that start and end at the depot")
        if any(by_name[name].kind is not NodeType.CUSTOMER for name in route[1:-1]):
            raise ValueError(
                "route splitting requires raw customer routes without charging stations"
            )


def _violation_counts(report: SolutionReport) -> dict[str, int]:
    violations = [
        *report.violations,
        *(violation for route_report in report.routes for violation in route_report.violations),
    ]
    return {
        "coverage": sum(
            "unvisited" in violation
            or "visited more than once" in violation
            or "unknown" in violation
            or "start and end" in violation
            for violation in violations
        ),
        "capacity": sum("load" in violation or "capacity" in violation for violation in violations),
        "time_window": sum(
            "arrival" in violation or "due date" in violation or "time window" in violation
            for violation in violations
        ),
        "energy": sum("battery" in violation or "energy" in violation for violation in violations),
    }


def _candidate_rank(report: SolutionReport) -> tuple[int, int, int, int, int, float]:
    counts = _violation_counts(report)
    return (
        counts["coverage"],
        counts["capacity"],
        counts["time_window"],
        counts["energy"],
        report.vehicle_count,
        report.total_distance,
    )


def _best_station_for_leg(
    instance: Instance,
    *,
    origin_name: str,
    destination_name: str,
    available_battery: float,
) -> str | None:
    by_name = instance.by_name
    origin = by_name[origin_name]
    destination = by_name[destination_name]
    direct_distance = origin.distance_to(destination)
    candidates: list[tuple[float, str]] = []

    for station in instance.stations:
        energy_to_station = origin.distance_to(station) * instance.vehicle.consumption_rate
        energy_station_to_destination = (
            station.distance_to(destination) * instance.vehicle.consumption_rate
        )
        if energy_to_station > available_battery + 1e-9:
            continue
        if energy_station_to_destination > instance.vehicle.battery_capacity + 1e-9:
            continue
        extra_distance = (
            origin.distance_to(station)
            + station.distance_to(destination)
            - direct_distance
        )
        candidates.append((extra_distance, station.name))

    if not candidates:
        return None
    return min(candidates)[1]


def _leg_needs_anticipatory_charge(
    instance: Instance,
    *,
    destination_name: str,
    battery_after_leg: float,
    min_reserve_ratio: float,
) -> bool:
    if battery_after_leg < -1e-9:
        return True

    reserve = instance.vehicle.battery_capacity * min_reserve_ratio
    if battery_after_leg + 1e-9 < reserve:
        return True

    destination = instance.by_name[destination_name]
    if destination.kind in {NodeType.DEPOT, NodeType.STATION}:
        return False

    return not _has_reachable_safe_node(
        instance,
        origin_name=destination_name,
        available_battery=battery_after_leg,
    )


def _best_anticipatory_station_for_leg(
    instance: Instance,
    *,
    origin_name: str,
    destination_name: str,
    available_battery: float,
    min_reserve_ratio: float,
) -> str | None:
    by_name = instance.by_name
    origin = by_name[origin_name]
    destination = by_name[destination_name]
    direct_distance = origin.distance_to(destination)
    candidates: list[tuple[float, float, str]] = []

    for station in instance.stations:
        energy_to_station = origin.distance_to(station) * instance.vehicle.consumption_rate
        energy_station_to_destination = (
            station.distance_to(destination) * instance.vehicle.consumption_rate
        )
        battery_after_destination = instance.vehicle.battery_capacity - (
            energy_station_to_destination
        )
        if energy_to_station > available_battery + 1e-9:
            continue
        if energy_station_to_destination > instance.vehicle.battery_capacity + 1e-9:
            continue
        if _leg_needs_anticipatory_charge(
            instance,
            destination_name=destination_name,
            battery_after_leg=battery_after_destination,
            min_reserve_ratio=min_reserve_ratio,
        ):
            continue

        extra_distance = (
            origin.distance_to(station)
            + station.distance_to(destination)
            - direct_distance
        )
        candidates.append((extra_distance, -battery_after_destination, station.name))

    if not candidates:
        return None
    return min(candidates)[2]


def _has_reachable_safe_node(
    instance: Instance,
    *,
    origin_name: str,
    available_battery: float,
) -> bool:
    by_name = instance.by_name
    origin = by_name[origin_name]
    safe_nodes = (instance.depot, *instance.stations)

    return any(
        safe_node.name != origin_name
        and origin.distance_to(safe_node) * instance.vehicle.consumption_rate
        <= available_battery + 1e-9
        for safe_node in safe_nodes
    )
