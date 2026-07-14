"""Explicit exact-charging backends for the Apple Metal pilot.

The label-setting algorithm in :mod:`evrptw.charging` remains the reference
implementation. This module adds a batch seam around its transition
arithmetic. CPU code still owns the priority queues, dominance checks and path
reconstruction; a Metal backend is allowed to calculate only distance, energy
and travel-time arithmetic for an ordered batch of transitions.

The module deliberately has no implicit fallback. A requested Metal backend
either runs through the native bridge or raises a typed error which the pilot
runner records as an invalid run.
"""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast

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

_native_core: Any | None
try:  # The extension is optional on non-Apple development machines.
    from evrptw import _core as _native_core
except ImportError:  # pragma: no cover - depends on the local build environment
    _native_core = None


class ExactChargingBackend(StrEnum):
    """Supported exact-charging execution backends."""

    CPU_SCALAR = "cpu_scalar"
    CPU_BATCH = "cpu_batch"
    METAL_BATCH = "metal_batch"


class GPUBackendError(RuntimeError):
    """Base class for errors that invalidate a GPU pilot run."""


class MetalUnavailableError(GPUBackendError):
    """The native Metal bridge or a usable Metal device is unavailable."""


class MetalPrecisionError(GPUBackendError):
    """GPU transition arithmetic is outside the declared precision bound."""


class LabelBufferOverflowError(GPUBackendError):
    """The bounded batch label buffer would overflow."""


@dataclass(slots=True)
class BackendMetrics:
    """Timing and work counters emitted by one backend invocation."""

    backend: str
    batch_size: int
    total_seconds: float = 0.0
    kernel_seconds: float = 0.0
    transfer_seconds: float = 0.0
    packing_seconds: float = 0.0
    unpacking_seconds: float = 0.0
    cpu_postprocess_seconds: float = 0.0
    initialization_seconds: float = 0.0
    batch_launches: int = 0
    transitions: int = 0
    exact_calls: int = 0

    def add(self, other: BackendMetrics) -> None:
        if self.backend != other.backend:
            raise ValueError(
                f"cannot merge backend metrics {self.backend!r} and {other.backend!r}"
            )
        for field_name in (
            "total_seconds",
            "kernel_seconds",
            "transfer_seconds",
            "packing_seconds",
            "unpacking_seconds",
            "cpu_postprocess_seconds",
            "initialization_seconds",
        ):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        for field_name in ("batch_launches", "transitions", "exact_calls"):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        self.batch_size = max(self.batch_size, other.batch_size)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "batch_size": self.batch_size,
            "total_seconds": self.total_seconds,
            "kernel_seconds": self.kernel_seconds,
            "transfer_seconds": self.transfer_seconds,
            "packing_seconds": self.packing_seconds,
            "unpacking_seconds": self.unpacking_seconds,
            "cpu_postprocess_seconds": self.cpu_postprocess_seconds,
            "initialization_seconds": self.initialization_seconds,
            "batch_launches": self.batch_launches,
            "transitions": self.transitions,
            "exact_calls": self.exact_calls,
        }


@dataclass(frozen=True, slots=True)
class TransitionArithmetic:
    """Arithmetic returned by a transition backend before host-side semantics."""

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
        self.metrics.kernel_seconds += time.perf_counter() - started
        return tuple(values)


class _MetalTransitionBackend:
    _RELATIVE_TOLERANCE = 2e-6

    def __init__(self, metrics: BackendMetrics) -> None:
        if _native_core is None or not hasattr(_native_core, "metal_backend_info"):
            raise MetalUnavailableError(
                "Metal backend requested but the native extension has no Metal bridge"
            )
        try:
            info = dict(_native_core.metal_backend_info())
        except BaseException as error:
            raise MetalUnavailableError("Metal capability query failed") from error
        if not bool(info.get("available", False)):
            reason = str(info.get("reason", "no usable Metal device"))
            raise MetalUnavailableError(reason)
        self.metrics = metrics
        self.metrics.initialization_seconds = float(
            cast(Any, info.get("initialization_seconds", 0.0))
        )

    def evaluate(
        self,
        requests: Sequence[_TransitionRequest],
    ) -> tuple[TransitionArithmetic, ...]:
        if not requests:
            return ()
        try:
            import numpy as np
        except ImportError as error:  # pragma: no cover - NumPy is a project dependency
            raise MetalUnavailableError("Metal batch packing requires NumPy") from error

        packed_started = time.perf_counter()
        dx = np.asarray(
            [request.origin.x - request.destination.x for request in requests],
            dtype=np.float32,
        )
        dy = np.asarray(
            [request.origin.y - request.destination.y for request in requests],
            dtype=np.float32,
        )
        consumption = np.asarray(
            [request.instance.vehicle.consumption_rate for request in requests],
            dtype=np.float32,
        )
        velocity = np.asarray(
            [request.instance.vehicle.average_velocity for request in requests],
            dtype=np.float32,
        )
        self.metrics.packing_seconds += time.perf_counter() - packed_started
        if _native_core is None:
            raise MetalUnavailableError("Metal native bridge disappeared after capability probe")
        result = _native_core.metal_transition_batch(dx, dy, consumption, velocity)
        self.metrics.transfer_seconds += float(cast(Any, result["transfer_seconds"]))
        self.metrics.kernel_seconds += float(cast(Any, result["kernel_seconds"]))
        self.metrics.initialization_seconds += float(
            cast(Any, result.get("initialization_seconds", 0.0))
        )

        unpacked_started = time.perf_counter()
        outputs = np.asarray(result["outputs"], dtype=np.float32)
        if outputs.shape != (len(requests), 3):
            raise MetalPrecisionError(
                "Metal transition kernel returned an unexpected output shape"
            )
        values: list[TransitionArithmetic] = []
        for request, output in zip(requests, outputs, strict=True):
            distance = float(output[0])
            energy = float(output[1])
            travel_time = float(output[2])
            if not all(math.isfinite(value) for value in (distance, energy, travel_time)):
                raise MetalPrecisionError("Metal transition kernel returned a non-finite value")
            reference_distance = math.hypot(
                request.origin.x - request.destination.x,
                request.origin.y - request.destination.y,
            )
            reference_energy = reference_distance * request.instance.vehicle.consumption_rate
            reference_time = reference_distance / request.instance.vehicle.average_velocity
            for name, actual, reference in (
                ("distance", distance, reference_distance),
                ("energy", energy, reference_energy),
                ("travel_time", travel_time, reference_time),
            ):
                tolerance = max(1e-7, self._RELATIVE_TOLERANCE * max(1.0, abs(reference)))
                if abs(actual - reference) > tolerance:
                    raise MetalPrecisionError(
                        f"Metal {name} arithmetic exceeded tolerance: "
                        f"actual={actual!r}, reference={reference!r}, tolerance={tolerance!r}"
                    )
            values.append(
                TransitionArithmetic(distance, energy, travel_time)
            )
        self.metrics.unpacking_seconds += time.perf_counter() - unpacked_started
        return tuple(values)


def metal_backend_info() -> dict[str, object]:
    """Return the local Metal capability without selecting a fallback backend."""

    if _native_core is None or not hasattr(_native_core, "metal_backend_info"):
        return {
            "available": False,
            "reason": "native extension was built without the Metal bridge",
        }
    try:
        return dict(_native_core.metal_backend_info())
    except BaseException as error:
        return {"available": False, "reason": f"capability query failed: {error}"}


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
    label_buffer_capacity: int = 1_000_000,
) -> BatchChargingResult:
    """Evaluate fixed customer orders with an explicit arithmetic backend.

    ``cpu_scalar`` delegates to the scalar reference. ``cpu_batch`` and
    ``metal_batch`` use the same host-side label manager, but process the
    transition arithmetic for multiple active route searches together. The
    request and result order is stable and is part of the pilot replay hash.
    """

    selected = ExactChargingBackend(backend)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if label_buffer_capacity <= 0:
        raise LabelBufferOverflowError("label buffer capacity must be positive")
    orders = tuple(tuple(order) for order in customer_orders)
    metrics = BackendMetrics(selected.value, batch_size, exact_calls=len(orders))
    started = time.perf_counter()
    if selected is ExactChargingBackend.CPU_SCALAR:
        scalar_results = tuple(solve_exact_charging(instance, order) for order in orders)
        metrics.transitions = sum(
            max(0, result.labels_generated - 1) for result in scalar_results
        )
        metrics.batch_launches = len(scalar_results)
        metrics.cpu_postprocess_seconds = time.perf_counter() - started
        metrics.total_seconds = metrics.cpu_postprocess_seconds
        return BatchChargingResult(scalar_results, metrics)

    transition_backend: _TransitionBackend
    if selected is ExactChargingBackend.CPU_BATCH:
        transition_backend = _CPUTransitionBackend(metrics)
    else:
        transition_backend = _MetalTransitionBackend(metrics)
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
            chunk = requests[offset : offset + batch_size]
            metrics.batch_launches += 1
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
                if sum(len(labels) for labels in state.labels.values()) > label_buffer_capacity:
                    raise LabelBufferOverflowError(
                        f"route {state.route_index} exceeded label buffer capacity "
                        f"{label_buffer_capacity}"
                    )
                state.serial += 1
                heapq.heappush(
                    state.queue,
                    _QueueEntry(_queue_priority(candidate), state.serial, candidate),
                )
            metrics.cpu_postprocess_seconds += time.perf_counter() - apply_started

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
    metrics.cpu_postprocess_seconds = max(
        0.0,
        metrics.total_seconds
        - metrics.kernel_seconds
        - metrics.transfer_seconds
        - metrics.packing_seconds
        - metrics.unpacking_seconds,
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
