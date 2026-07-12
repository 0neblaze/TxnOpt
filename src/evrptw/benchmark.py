from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evrptw.models import Instance, Node
from evrptw.parser import parse_schneider

_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class BatteryBounds:
    theoretical_lower_bound: float
    structural_lower_bound: float
    experimental_capacity: float
    stress_ratio: float


@dataclass(frozen=True, slots=True)
class InstanceAudit:
    instance: str
    source_path: str
    customer_count: int
    station_count: int
    load_capacity: float
    battery_capacity: float
    consumption_rate: float
    inverse_refueling_rate: float
    average_velocity: float
    depot_due_date: float
    earliest_customer_ready: float
    latest_customer_due: float
    distance_metric: str
    battery_bounds: BatteryBounds
    structurally_feasible: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        bounds = row.pop("battery_bounds")
        row.update(bounds)
        row["failures"] = " | ".join(self.failures)
        row["warnings"] = " | ".join(self.warnings)
        return row


def calculate_battery_bounds(instance: Instance) -> BatteryBounds:
    """Calculate energy bounds from depot/station-to-customer reachability.

    The theoretical bound is the largest unavoidable single incident leg over all
    customers. The structural bound is the largest, over customers, of the cheapest
    recharge-node -> customer -> recharge-node energy segment. Recharge nodes are the
    depot and stations. This is a necessary customer-level bound, not a proof that a
    complete multi-customer EVRP-TW solution exists.
    """

    recharge_nodes = (instance.depot, *instance.stations)
    rate = instance.vehicle.consumption_rate
    if not instance.customers:
        return BatteryBounds(0.0, 0.0, instance.vehicle.battery_capacity, float("inf"))

    theoretical = 0.0
    structural = 0.0
    for customer in instance.customers:
        incoming = min(_energy(node, customer, rate) for node in recharge_nodes)
        outgoing = min(_energy(customer, node, rate) for node in recharge_nodes)
        theoretical = max(theoretical, incoming, outgoing)
        structural = max(structural, incoming + outgoing)

    ratio = (
        instance.vehicle.battery_capacity / structural
        if structural > _EPSILON
        else float("inf")
    )
    return BatteryBounds(theoretical, structural, instance.vehicle.battery_capacity, ratio)


def audit_instance(instance: Instance, *, source_path: str | Path = "") -> InstanceAudit:
    failures: list[str] = []
    warnings: list[str] = []
    vehicle = instance.vehicle
    depot = instance.depot
    bounds = calculate_battery_bounds(instance)
    recharge_nodes = (depot, *instance.stations)

    if not instance.customers:
        failures.append("instance has no customers")
    if not instance.stations:
        warnings.append("instance has no charging stations")
    if vehicle.battery_capacity + _EPSILON < bounds.theoretical_lower_bound:
        failures.append("battery below theoretical single-leg lower bound")
    if vehicle.battery_capacity + _EPSILON < bounds.structural_lower_bound:
        failures.append("battery below customer-level structural lower bound")
    if depot.ready_time > depot.due_date + _EPSILON:
        failures.append("depot time window is empty")

    for customer in instance.customers:
        if customer.demand > vehicle.load_capacity + _EPSILON:
            failures.append(f"{customer.name}: demand exceeds vehicle capacity")
        if customer.ready_time > customer.due_date + _EPSILON:
            failures.append(f"{customer.name}: time window is empty")

        earliest_from_depot = max(
            depot.ready_time + depot.distance_to(customer) / vehicle.average_velocity,
            customer.ready_time,
        )
        if earliest_from_depot > customer.due_date + _EPSILON:
            failures.append(f"{customer.name}: unreachable from depot before due date")

        inbound = [
            node
            for node in recharge_nodes
            if _energy(node, customer, vehicle.consumption_rate)
            <= vehicle.battery_capacity + _EPSILON
        ]
        outbound = [
            node
            for node in recharge_nodes
            if _energy(customer, node, vehicle.consumption_rate)
            <= vehicle.battery_capacity + _EPSILON
        ]
        if not inbound:
            failures.append(f"{customer.name}: no battery-feasible inbound recharge node")
        if not outbound:
            failures.append(f"{customer.name}: no battery-feasible return recharge node")

        shortest_round_trip = 2.0 * depot.distance_to(customer) / vehicle.average_velocity
        direct_finish = depot.ready_time + shortest_round_trip + customer.service_time
        if direct_finish > depot.due_date + _EPSILON:
            failures.append(f"{customer.name}: direct service cannot fit depot horizon")

    return InstanceAudit(
        instance=instance.name,
        source_path=str(Path(source_path).resolve()) if source_path else "",
        customer_count=len(instance.customers),
        station_count=len(instance.stations),
        load_capacity=vehicle.load_capacity,
        battery_capacity=vehicle.battery_capacity,
        consumption_rate=vehicle.consumption_rate,
        inverse_refueling_rate=vehicle.inverse_refueling_rate,
        average_velocity=vehicle.average_velocity,
        depot_due_date=depot.due_date,
        earliest_customer_ready=min((node.ready_time for node in instance.customers), default=0.0),
        latest_customer_due=max((node.due_date for node in instance.customers), default=0.0),
        distance_metric="EUC_2D (unrounded Euclidean distance)",
        battery_bounds=bounds,
        structurally_feasible=not failures,
        failures=tuple(dict.fromkeys(failures)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def audit_schneider_directory(directory: str | Path) -> tuple[InstanceAudit, ...]:
    root = Path(directory)
    instance_paths = sorted(
        path for path in root.glob("*.txt") if path.name.lower() != "readme.txt"
    )
    if not instance_paths:
        raise ValueError(f"no Schneider instances found in {root}")
    return tuple(audit_instance(parse_schneider(path), source_path=path) for path in instance_paths)


def _energy(origin: Node, destination: Node, consumption_rate: float) -> float:
    return origin.distance_to(destination) * consumption_rate
