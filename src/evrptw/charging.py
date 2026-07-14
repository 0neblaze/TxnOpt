from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

from evrptw.models import Instance, Node, NodeType
from evrptw.validation import validate_routes

_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class ChargingLabel:
    progress: int
    node_name: str
    elapsed_time: float
    battery: float
    distance: float
    total_energy: float
    charged_energy: float
    charging_time: float
    path: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChargingSubproblemResult:
    feasible: bool
    route: tuple[str, ...]
    distance: float
    total_energy: float
    charged_energy: float
    charging_time: float
    labels_generated: int
    labels_expanded: int
    labels_pruned: int
    runtime_seconds: float
    failure_reason: str


@dataclass(order=True, slots=True)
class _QueueEntry:
    priority: tuple[float, float, int]
    serial: int
    label: ChargingLabel = field(compare=False)


def solve_exact_charging(
    instance: Instance,
    customer_order: list[str] | tuple[str, ...],
) -> ChargingSubproblemResult:
    """Solve station insertion exactly for a fixed customer sequence.

    The charging policy is the Schneider full-recharge, linear-time policy used by
    the repository validator. Customer assignment and order are fixed. Labels retain
    every nondominated combination of elapsed time, battery and distance at each
    (served-customer count, current node) state.
    """

    started = time.perf_counter()
    order = tuple(customer_order)
    failure = _validate_order(instance, order)
    if failure:
        return _failure_result(started, failure)

    depot = instance.depot
    initial = ChargingLabel(
        progress=0,
        node_name=depot.name,
        elapsed_time=max(0.0, depot.ready_time),
        battery=instance.vehicle.battery_capacity,
        distance=0.0,
        total_energy=0.0,
        charged_energy=0.0,
        charging_time=0.0,
        path=(depot.name,),
    )
    labels: dict[tuple[int, str], list[ChargingLabel]] = {(0, depot.name): [initial]}
    queue: list[_QueueEntry] = [_QueueEntry(_queue_priority(initial), 0, initial)]
    generated = 1
    expanded = 0
    pruned = 0
    serial = 1
    best: ChargingLabel | None = None

    while queue:
        entry = heapq.heappop(queue)
        label = entry.label
        state_labels = labels.get((label.progress, label.node_name), [])
        if label not in state_labels:
            continue
        if best is not None and label.distance >= best.distance - _EPSILON:
            pruned += 1
            continue
        expanded += 1

        for destination, new_progress in _successors(instance, order, label):
            candidate = _extend(instance, label, destination, new_progress)
            generated += 1
            if candidate is None:
                pruned += 1
                continue
            if new_progress == len(order) and destination.kind is NodeType.DEPOT:
                if best is None or _better_terminal(candidate, best):
                    best = candidate
                continue

            key = (candidate.progress, candidate.node_name)
            current = labels.setdefault(key, [])
            if any(_dominates(existing, candidate) for existing in current):
                pruned += 1
                continue
            survivors = [existing for existing in current if not _dominates(candidate, existing)]
            pruned += len(current) - len(survivors)
            survivors.append(candidate)
            labels[key] = survivors
            heapq.heappush(
                queue,
                _QueueEntry(
                    _queue_priority(candidate),
                    serial,
                    candidate,
                ),
            )
            serial += 1

    runtime = time.perf_counter() - started
    if best is None:
        return ChargingSubproblemResult(
            False,
            (),
            float("inf"),
            0.0,
            0.0,
            0.0,
            generated,
            expanded,
            pruned,
            runtime,
            "no feasible station-insertion pattern for fixed customer order",
        )

    route = list(best.path)
    report = validate_routes(instance, [route])
    route_reasons = [violation for item in report.routes for violation in item.violations]
    routed_customers = tuple(
        name for name in route if instance.by_name[name].kind is NodeType.CUSTOMER
    )
    if route_reasons or routed_customers != order:
        reasons = [*route_reasons]
        if routed_customers != order:
            reasons.append("route customer sequence differs from fixed order")
        raise RuntimeError("exact charging returned an invalid route: " + " | ".join(reasons))
    return ChargingSubproblemResult(
        True,
        best.path,
        best.distance,
        best.total_energy,
        best.charged_energy,
        best.charging_time,
        generated,
        expanded,
        pruned,
        runtime,
        "",
    )


def _validate_order(instance: Instance, order: tuple[str, ...]) -> str:
    customer_names = {customer.name for customer in instance.customers}
    if len(set(order)) != len(order):
        return "customer order contains duplicates"
    unknown = [name for name in order if name not in customer_names]
    if unknown:
        return "customer order contains non-customer nodes: " + ", ".join(unknown)
    demand = sum(instance.by_name[name].demand for name in order)
    if demand > instance.vehicle.load_capacity + _EPSILON:
        return "fixed route demand exceeds vehicle capacity"
    return ""


def _successors(
    instance: Instance,
    order: tuple[str, ...],
    label: ChargingLabel,
) -> tuple[tuple[Node, int], ...]:
    current = instance.by_name[label.node_name]
    candidates: list[tuple[Node, int]] = []
    if label.progress < len(order):
        candidates.append((instance.by_name[order[label.progress]], label.progress + 1))
        candidates.extend(
            (station, label.progress)
            for station in instance.stations
            if station.name != current.name
        )
    else:
        candidates.append((instance.depot, label.progress))
        candidates.extend(
            (station, label.progress)
            for station in instance.stations
            if station.name != current.name
        )
    return tuple(candidates)


def _extend(
    instance: Instance,
    label: ChargingLabel,
    destination: Node,
    progress: int,
) -> ChargingLabel | None:
    origin = instance.by_name[label.node_name]
    distance = origin.distance_to(destination)
    energy = distance * instance.vehicle.consumption_rate
    travel_time = distance / instance.vehicle.average_velocity
    return _extend_with_transition(
        instance,
        label,
        destination,
        progress,
        distance=distance,
        energy=energy,
        travel_time=travel_time,
    )


def _queue_priority(label: ChargingLabel) -> tuple[float, float, int]:
    """Keep the scalar and batched label queue ordering identical."""

    return (label.distance, label.elapsed_time, -label.progress)


def _extend_with_transition(
    instance: Instance,
    label: ChargingLabel,
    destination: Node,
    progress: int,
    *,
    distance: float,
    energy: float,
    travel_time: float,
) -> ChargingLabel | None:
    if energy > label.battery + _EPSILON:
        return None

    battery = max(0.0, label.battery - energy)
    elapsed = label.elapsed_time + travel_time
    elapsed = max(elapsed, destination.ready_time)
    if elapsed > destination.due_date + _EPSILON:
        return None

    charged = 0.0
    charge_time = 0.0
    if destination.kind is NodeType.CUSTOMER:
        elapsed += destination.service_time
    elif destination.kind is NodeType.STATION:
        charged = instance.vehicle.battery_capacity - battery
        charge_time = charged * instance.vehicle.inverse_refueling_rate
        elapsed += charge_time
        if elapsed > destination.due_date + _EPSILON:
            return None
        battery = instance.vehicle.battery_capacity

    return ChargingLabel(
        progress=progress,
        node_name=destination.name,
        elapsed_time=elapsed,
        battery=battery,
        distance=label.distance + distance,
        total_energy=label.total_energy + energy,
        charged_energy=label.charged_energy + charged,
        charging_time=label.charging_time + charge_time,
        path=(*label.path, destination.name),
    )


def _dominates(left: ChargingLabel, right: ChargingLabel) -> bool:
    no_worse = (
        left.elapsed_time <= right.elapsed_time + _EPSILON
        and left.battery + _EPSILON >= right.battery
        and left.distance <= right.distance + _EPSILON
    )
    strictly_better = (
        left.elapsed_time < right.elapsed_time - _EPSILON
        or left.battery > right.battery + _EPSILON
        or left.distance < right.distance - _EPSILON
    )
    return no_worse and strictly_better


def _better_terminal(candidate: ChargingLabel, incumbent: ChargingLabel) -> bool:
    candidate_key = (candidate.distance, candidate.elapsed_time)
    incumbent_key = (incumbent.distance, incumbent.elapsed_time)
    return candidate_key < incumbent_key


def _failure_result(started: float, reason: str) -> ChargingSubproblemResult:
    return ChargingSubproblemResult(
        False,
        (),
        float("inf"),
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
        time.perf_counter() - started,
        reason,
    )
