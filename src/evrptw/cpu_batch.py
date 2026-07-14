"""CPU-only exact-charging backends.

The scalar label-setting algorithm in :mod:`evrptw.charging` remains the
reference implementation.  The batched backend advances several independent
route searches together while preserving request order, queue ordering,
dominance pruning and exact result semantics.
"""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from evrptw.charging import (
    _EPSILON,
    ChargingLabel,
    ChargingSubproblemResult,
    _better_terminal,
    _dominates,
    _extend_with_transition,
    _failure_result,
    _queue_priority,
    _QueueEntry,
    _successors,
    _validate_order,
    solve_exact_charging,
)
from evrptw.models import Instance, Node, NodeType
from evrptw.validation import validate_routes


class ExactChargingBackend(StrEnum):
    """Supported CPU exact-charging execution backends."""

    CPU_SCALAR = "cpu_scalar"
    CPU_BATCH = "cpu_batch"


class ExactBatchDeadlineExceeded(RuntimeError):
    """The cooperative CPU batch crossed its deadline before completion."""

    def __init__(self, completed_exact_calls: int = 0) -> None:
        self.completed_exact_calls = completed_exact_calls
        super().__init__("CPU exact-charging batch deadline exceeded")


@dataclass(slots=True)
class BackendMetrics:
    """CPU timing and work counters emitted by one backend invocation."""

    backend: str
    batch_size: int
    total_seconds: float = 0.0
    transition_seconds: float = 0.0
    label_management_seconds: float = 0.0
    work_batches: int = 0
    transition_batches: int = 0
    transitions: int = 0
    exact_calls: int = 0

    def add(self, other: BackendMetrics) -> None:
        if self.backend != other.backend:
            raise ValueError(
                f"cannot merge backend metrics {self.backend!r} and {other.backend!r}"
            )
        for field_name in (
            "total_seconds",
            "transition_seconds",
            "label_management_seconds",
        ):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        for field_name in ("work_batches", "transition_batches", "transitions", "exact_calls"):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        self.batch_size = max(self.batch_size, other.batch_size)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "batch_size": self.batch_size,
            "total_seconds": self.total_seconds,
            "transition_seconds": self.transition_seconds,
            "label_management_seconds": self.label_management_seconds,
            "work_batches": self.work_batches,
            "transition_batches": self.transition_batches,
            "transitions": self.transitions,
            "exact_calls": self.exact_calls,
        }


@dataclass(frozen=True, slots=True)
class TransitionArithmetic:
    """Transition arithmetic before label-management semantics."""

    distance: float
    energy: float
    travel_time: float


@dataclass(frozen=True, slots=True)
class _TransitionRequest:
    instance: Instance
    route_index: int
    label: ChargingLabel
    origin: Node
    destination: Node
    progress: int


@dataclass(slots=True)
class _SearchState:
    route_index: int
    order: tuple[str, ...]
    started: float
    labels: dict[tuple[int, str], list[ChargingLabel]]
    queue: list[_QueueEntry]
    generated: int = 1
    expanded: int = 0
    pruned: int = 0
    serial: int = 1
    best: ChargingLabel | None = None


class _TransitionBackend(Protocol):
    def evaluate(
        self,
        requests: Sequence[_TransitionRequest],
    ) -> tuple[TransitionArithmetic, ...]: ...


class _CPUTransitionBackend:
    def __init__(self, metrics: BackendMetrics) -> None:
        self.metrics = metrics

    def evaluate(
        self,
        requests: Sequence[_TransitionRequest],
    ) -> tuple[TransitionArithmetic, ...]:
        started = time.perf_counter()
        values: list[TransitionArithmetic] = []
        for request in requests:
            distance = math.hypot(
                request.origin.x - request.destination.x,
                request.origin.y - request.destination.y,
            )
            values.append(
                TransitionArithmetic(
                    distance=distance,
                    energy=distance * request.instance.vehicle.consumption_rate,
                    travel_time=distance / request.instance.vehicle.average_velocity,
                )
            )
        self.metrics.transition_seconds += time.perf_counter() - started
        return tuple(values)


@dataclass(frozen=True, slots=True)
class BatchChargingResult:
    """Results and auditable backend counters for one ordered work batch."""

    results: tuple[ChargingSubproblemResult, ...]
    metrics: BackendMetrics


def solve_exact_charging_batch(
    instance: Instance,
    customer_orders: Sequence[Sequence[str]],
    *,
    backend: ExactChargingBackend | str = ExactChargingBackend.CPU_BATCH,
    batch_size: int = 128,
    deadline: float | None = None,
) -> BatchChargingResult:
    """Evaluate fixed customer orders with an explicit CPU backend."""

    selected = ExactChargingBackend(backend)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if deadline is not None and time.perf_counter() >= deadline:
        raise ExactBatchDeadlineExceeded()
    orders = tuple(tuple(order) for order in customer_orders)
    metrics = BackendMetrics(
        selected.value,
        batch_size,
        work_batches=(len(orders) if selected is ExactChargingBackend.CPU_SCALAR else bool(orders)),
        exact_calls=len(orders),
    )
    started = time.perf_counter()
    if selected is ExactChargingBackend.CPU_SCALAR:
        scalar_results = tuple(solve_exact_charging(instance, order) for order in orders)
        metrics.transitions = sum(
            max(0, result.labels_generated - 1) for result in scalar_results
        )
        metrics.transition_batches = len(scalar_results)
        metrics.label_management_seconds = time.perf_counter() - started
        metrics.total_seconds = metrics.label_management_seconds
        return BatchChargingResult(scalar_results, metrics)

    transition_backend: _TransitionBackend = _CPUTransitionBackend(metrics)
    states: list[_SearchState] = []
    results: list[ChargingSubproblemResult | None] = [None] * len(orders)
    for index, order in enumerate(orders):
        failure = _validate_order(instance, order)
        if failure:
            results[index] = _failure_result(started, failure)
            continue
        initial = ChargingLabel(
            progress=0,
            node_name=instance.depot.name,
            elapsed_time=max(0.0, instance.depot.ready_time),
            battery=instance.vehicle.battery_capacity,
            distance=0.0,
            total_energy=0.0,
            charged_energy=0.0,
            charging_time=0.0,
            path=(instance.depot.name,),
        )
        states.append(
            _SearchState(
                route_index=index,
                order=order,
                started=started,
                labels={(0, instance.depot.name): [initial]},
                queue=[_QueueEntry(_queue_priority(initial), 0, initial)],
            )
        )

    states_by_index = {state.route_index: state for state in states}
    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            raise ExactBatchDeadlineExceeded()
        requests: list[_TransitionRequest] = []
        progressed = False
        for state in states:
            entry = _pop_live_entry(state)
            if entry is None:
                continue
            label = entry.label
            if state.best is not None and label.distance >= state.best.distance - _EPSILON:
                state.pruned += 1
                progressed = True
                continue
            state.expanded += 1
            progressed = True
            current = instance.by_name[label.node_name]
            for destination, progress in _successors(instance, state.order, label):
                requests.append(
                    _TransitionRequest(
                        instance=instance,
                        route_index=state.route_index,
                        label=label,
                        origin=current,
                        destination=destination,
                        progress=progress,
                    )
                )
        if not requests:
            if not progressed:
                break
            continue

        for offset in range(0, len(requests), batch_size):
            if deadline is not None and time.perf_counter() >= deadline:
                raise ExactBatchDeadlineExceeded()
            chunk = requests[offset : offset + batch_size]
            metrics.transition_batches += 1
            metrics.transitions += len(chunk)
            arithmetic = transition_backend.evaluate(chunk)
            apply_started = time.perf_counter()
            for request, values in zip(chunk, arithmetic, strict=True):
                state = states_by_index[request.route_index]
                state.generated += 1
                candidate = _extend_with_transition(
                    instance,
                    request.label,
                    request.destination,
                    request.progress,
                    distance=values.distance,
                    energy=values.energy,
                    travel_time=values.travel_time,
                )
                if candidate is None:
                    state.pruned += 1
                    continue
                if (
                    request.progress == len(state.order)
                    and request.destination.kind is NodeType.DEPOT
                ):
                    if state.best is None or _better_terminal(candidate, state.best):
                        state.best = candidate
                    continue
                key = (candidate.progress, candidate.node_name)
                current_labels = state.labels.setdefault(key, [])
                if any(_dominates(existing, candidate) for existing in current_labels):
                    state.pruned += 1
                    continue
                survivors = [
                    existing
                    for existing in current_labels
                    if not _dominates(candidate, existing)
                ]
                state.pruned += len(current_labels) - len(survivors)
                survivors.append(candidate)
                state.labels[key] = survivors
                state.serial += 1
                heapq.heappush(
                    state.queue,
                    _QueueEntry(_queue_priority(candidate), state.serial, candidate),
                )
            metrics.label_management_seconds += time.perf_counter() - apply_started
            if deadline is not None and time.perf_counter() >= deadline:
                raise ExactBatchDeadlineExceeded()

    for state in states:
        if state.best is None:
            results[state.route_index] = ChargingSubproblemResult(
                False,
                (),
                float("inf"),
                0.0,
                0.0,
                0.0,
                state.generated,
                state.expanded,
                state.pruned,
                time.perf_counter() - state.started,
                "no feasible station-insertion pattern for fixed customer order",
            )
            continue
        routed_customers = tuple(
            name
            for name in state.best.path
            if instance.by_name[name].kind is NodeType.CUSTOMER
        )
        if routed_customers != state.order:
            raise RuntimeError(
                "batched exact charging returned a route with a different customer order"
            )
        report = validate_routes(instance, [list(state.best.path)])
        route_reasons = [
            violation
            for item in report.routes
            for violation in item.violations
        ]
        if route_reasons:
            raise RuntimeError(
                "batched exact charging returned an invalid route: "
                + " | ".join(route_reasons)
            )
        route_report = report.routes[0]
        results[state.route_index] = ChargingSubproblemResult(
            True,
            state.best.path,
            route_report.distance,
            route_report.total_energy,
            route_report.charged_energy,
            route_report.charging_time,
            state.generated,
            state.expanded,
            state.pruned,
            time.perf_counter() - state.started,
            "",
        )

    metrics.total_seconds = time.perf_counter() - started
    metrics.label_management_seconds = max(
        0.0,
        metrics.total_seconds
        - metrics.transition_seconds,
    )
    completed = tuple(result for result in results if result is not None)
    if len(completed) != len(orders):
        raise RuntimeError("batched exact charging lost a result while preserving request order")
    return BatchChargingResult(completed, metrics)


def _pop_live_entry(state: _SearchState) -> _QueueEntry | None:
    while state.queue:
        entry = heapq.heappop(state.queue)
        key = (entry.label.progress, entry.label.node_name)
        if entry.label in state.labels.get(key, []):
            return entry
    return None
