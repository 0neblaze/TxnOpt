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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

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
from evrptw.native_kernels import NativeKernelConfig, NativeKernelRuntime
from evrptw.validation import validate_routes


class ExactChargingBackend(StrEnum):
    """Supported CPU exact-charging execution backends."""

    CPU_SCALAR = "cpu_scalar"
    CPU_BATCH = "cpu_batch"


class ExactBatchDeadlineExceeded(RuntimeError):
    """The cooperative CPU batch crossed its deadline before completion."""

    def __init__(
        self,
        *,
        started_exact_calls: int = 0,
        completed_exact_calls: int = 0,
        metrics: BackendMetrics | None = None,
        completed_indices: tuple[int, ...] = (),
    ) -> None:
        self.started_exact_calls = started_exact_calls
        self.completed_exact_calls = completed_exact_calls
        self.interrupted_exact_calls = started_exact_calls - completed_exact_calls
        self.metrics = metrics or BackendMetrics("cpu_batch", 0)
        self.completed_indices = completed_indices
        super().__init__("CPU exact-charging batch deadline exceeded")

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Preserve structured deadline evidence across spawned workers."""

        return (
            _restore_exact_batch_deadline_exceeded,
            (
                self.started_exact_calls,
                self.completed_exact_calls,
                self.metrics,
                self.completed_indices,
            ),
        )


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
    batch_launches: int = 0
    packing_seconds: float = 0.0
    unpacking_seconds: float = 0.0
    checkpoint_count: int = 0
    started_calls: int = 0
    completed_calls: int = 0
    interrupted_calls: int = 0
    native_kernel_seconds: float = 0.0
    native_invocations: int = 0
    native_fallbacks: int = 0
    launch_occupancies: list[int] = field(default_factory=list)

    def add(self, other: BackendMetrics) -> None:
        if self.backend != other.backend:
            raise ValueError(f"cannot merge backend metrics {self.backend!r} and {other.backend!r}")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in other.launch_occupancies
            )
            or len(other.launch_occupancies) != other.batch_launches
            or sum(other.launch_occupancies) != other.exact_calls
        ):
            raise ValueError(
                "backend launch occupancies must reconcile with launches and exact calls"
            )
        for field_name in (
            "total_seconds",
            "transition_seconds",
            "label_management_seconds",
            "packing_seconds",
            "unpacking_seconds",
            "native_kernel_seconds",
        ):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        for field_name in (
            "work_batches",
            "transition_batches",
            "transitions",
            "exact_calls",
            "batch_launches",
            "checkpoint_count",
            "started_calls",
            "completed_calls",
            "interrupted_calls",
            "native_invocations",
            "native_fallbacks",
        ):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        self.launch_occupancies.extend(other.launch_occupancies)
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
            "batch_launches": self.batch_launches,
            "packing_seconds": self.packing_seconds,
            "unpacking_seconds": self.unpacking_seconds,
            "checkpoint_count": self.checkpoint_count,
            "started_calls": self.started_calls,
            "completed_calls": self.completed_calls,
            "interrupted_calls": self.interrupted_calls,
            "native_kernel_seconds": self.native_kernel_seconds,
            "native_invocations": self.native_invocations,
            "native_fallbacks": self.native_fallbacks,
            "launch_occupancies": list(self.launch_occupancies),
        }


def _restore_exact_batch_deadline_exceeded(
    started_exact_calls: int,
    completed_exact_calls: int,
    metrics: BackendMetrics,
    completed_indices: tuple[int, ...],
) -> ExactBatchDeadlineExceeded:
    return ExactBatchDeadlineExceeded(
        started_exact_calls=started_exact_calls,
        completed_exact_calls=completed_exact_calls,
        metrics=metrics,
        completed_indices=completed_indices,
    )


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
    def __init__(self, metrics: BackendMetrics, instance: Instance) -> None:
        self.metrics = metrics
        self._distance = instance.distance
        self._consumption_rate = instance.vehicle.consumption_rate
        self._average_velocity = instance.vehicle.average_velocity

    def evaluate(
        self,
        requests: Sequence[_TransitionRequest],
    ) -> tuple[TransitionArithmetic, ...]:
        started = time.perf_counter()
        values: list[TransitionArithmetic] = []
        append = values.append
        distance_between = self._distance
        consumption_rate = self._consumption_rate
        average_velocity = self._average_velocity
        for request in requests:
            distance = distance_between(
                request.origin.name,
                request.destination.name,
            )
            append(
                TransitionArithmetic(
                    distance=distance,
                    energy=distance * consumption_rate,
                    travel_time=distance / average_velocity,
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
    native_kernel_config: NativeKernelConfig | None = None,
    native_runtime: NativeKernelRuntime | None = None,
) -> BatchChargingResult:
    """Evaluate fixed customer orders with an explicit CPU backend."""

    selected = ExactChargingBackend(backend)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    orders = tuple(tuple(order) for order in customer_orders)
    metrics = BackendMetrics(
        selected.value,
        batch_size,
        work_batches=(len(orders) if selected is ExactChargingBackend.CPU_SCALAR else bool(orders)),
        exact_calls=len(orders),
        batch_launches=int(bool(orders)),
        started_calls=len(orders),
        launch_occupancies=([len(orders)] if orders else []),
    )
    started = time.perf_counter()
    completed_indices: set[int] = set()

    def checkpoint() -> None:
        metrics.checkpoint_count += 1
        now = time.perf_counter()
        if deadline is None or now < deadline:
            return
        metrics.completed_calls = len(completed_indices)
        metrics.interrupted_calls = len(orders) - len(completed_indices)
        metrics.total_seconds = max(0.0, now - started)
        raise ExactBatchDeadlineExceeded(
            started_exact_calls=len(orders),
            completed_exact_calls=len(completed_indices),
            metrics=metrics,
            completed_indices=tuple(sorted(completed_indices)),
        )

    checkpoint()
    if selected is ExactChargingBackend.CPU_SCALAR:
        if native_kernel_config is not None or native_runtime is not None:
            raise ValueError("native exact charging requires backend='cpu_batch'")
        scalar_results = tuple(solve_exact_charging(instance, order) for order in orders)
        metrics.transitions = sum(max(0, result.labels_generated - 1) for result in scalar_results)
        metrics.transition_batches = len(scalar_results)
        metrics.label_management_seconds = time.perf_counter() - started
        metrics.total_seconds = metrics.label_management_seconds
        metrics.completed_calls = len(orders)
        return BatchChargingResult(scalar_results, metrics)

    if native_kernel_config is not None or native_runtime is not None:
        runtime = native_runtime
        if runtime is None:
            if native_kernel_config is None:
                raise RuntimeError("native exact charging is missing its explicit configuration")
            runtime = NativeKernelRuntime.build(instance, native_kernel_config)
        elif native_kernel_config is not None and runtime.config != native_kernel_config:
            raise ValueError("native runtime and native kernel configuration disagree")
        runtime.context.assert_matches(instance)
        return _solve_exact_charging_native(
            instance,
            orders,
            runtime=runtime,
            batch_size=batch_size,
            deadline=deadline,
            started=started,
            metrics=metrics,
            checkpoint=checkpoint,
            completed_indices=completed_indices,
        )

    packing_started = time.perf_counter()
    transition_backend: _TransitionBackend = _CPUTransitionBackend(metrics, instance)
    states: list[_SearchState] = []
    results: list[ChargingSubproblemResult | None] = [None] * len(orders)
    for index, order in enumerate(orders):
        failure = _validate_order(instance, order)
        if failure:
            results[index] = _failure_result(started, failure)
            completed_indices.add(index)
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
    metrics.packing_seconds = time.perf_counter() - packing_started

    states_by_index = {state.route_index: state for state in states}
    while True:
        requests: list[_TransitionRequest] = []
        progressed = False
        for state in states:
            entry = _pop_live_entry(state)
            if entry is None:
                completed_indices.add(state.route_index)
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
        checkpoint()
        if not requests:
            if not progressed:
                break
            continue

        for offset in range(0, len(requests), batch_size):
            checkpoint()
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
                    existing for existing in current_labels if not _dominates(candidate, existing)
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

    unpacking_started = time.perf_counter()
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
            completed_indices.add(state.route_index)
            continue
        routed_customers = tuple(
            name for name in state.best.path if instance.by_name[name].kind is NodeType.CUSTOMER
        )
        if routed_customers != state.order:
            raise RuntimeError(
                "batched exact charging returned a route with a different customer order"
            )
        report = validate_routes(instance, [list(state.best.path)])
        route_reasons = [violation for item in report.routes for violation in item.violations]
        if route_reasons:
            raise RuntimeError(
                "batched exact charging returned an invalid route: " + " | ".join(route_reasons)
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
        completed_indices.add(state.route_index)

    metrics.unpacking_seconds = time.perf_counter() - unpacking_started
    metrics.total_seconds = time.perf_counter() - started
    metrics.label_management_seconds = max(
        0.0,
        metrics.total_seconds - metrics.transition_seconds,
    )
    completed = tuple(result for result in results if result is not None)
    if len(completed) != len(orders):
        raise RuntimeError("batched exact charging lost a result while preserving request order")
    metrics.completed_calls = len(orders)
    metrics.interrupted_calls = 0
    return BatchChargingResult(completed, metrics)


def _solve_exact_charging_native(
    instance: Instance,
    orders: tuple[tuple[str, ...], ...],
    *,
    runtime: NativeKernelRuntime,
    batch_size: int,
    deadline: float | None,
    started: float,
    metrics: BackendMetrics,
    checkpoint: Callable[[], None],
    completed_indices: set[int],
    native_payload: object | None = None,
    native_kernel_seconds: float | None = None,
) -> BatchChargingResult:
    """Execute the whole ordered label-setting batch through the native ABI."""

    if not runtime.config.exact_charging:
        raise RuntimeError("native runtime does not enable exact charging")
    results: list[ChargingSubproblemResult | None] = [None] * len(orders)
    valid_orders: list[tuple[str, ...]] = []
    valid_indices: list[int] = []
    packing_started = time.perf_counter()
    for index, order in enumerate(orders):
        failure = _validate_order(instance, order)
        if failure:
            results[index] = _failure_result(started, failure)
            completed_indices.add(index)
            continue
        valid_orders.append(order)
        valid_indices.append(index)

    context = runtime.context
    offsets = np.empty(len(valid_orders) + 1, dtype=np.int64)
    offsets[0] = 0
    flat_indices: list[int] = []
    for index, order in enumerate(valid_orders, start=1):
        flat_indices.extend(context.name_to_index[name] for name in order)
        offsets[index] = len(flat_indices)
    order_offsets = np.ascontiguousarray(offsets, dtype=np.int64)
    order_indices = np.ascontiguousarray(flat_indices, dtype=np.int64)
    deadline_remaining = np.ascontiguousarray(
        [float("inf") if deadline is None else max(0.0, deadline - time.perf_counter())],
        dtype=np.float64,
    )
    batch_size_array = np.ascontiguousarray([batch_size], dtype=np.int64)
    metrics.packing_seconds += (
        runtime.claim_context_packing_seconds() + time.perf_counter() - packing_started
    )
    checkpoint()
    if not valid_orders:
        metrics.completed_calls = len(orders)
        metrics.interrupted_calls = 0
        metrics.total_seconds = time.perf_counter() - started
        metrics.label_management_seconds = max(
            0.0,
            metrics.total_seconds - metrics.packing_seconds,
        )
        completed = tuple(result for result in results if result is not None)
        if len(completed) != len(orders):
            raise RuntimeError("native exact charging lost a prevalidated result")
        return BatchChargingResult(completed, metrics)

    try:
        from evrptw import _core as native_core
    except Exception as error:
        raise RuntimeError(
            "native exact charging requested but evrptw._core is unavailable"
        ) from error

    native_started = time.perf_counter()
    payload = (
        native_core.exact_charging_batch_numeric(
            context.node_kind,
            context.ready_time,
            context.due_date,
            context.service_time,
            context.distance,
            context.vehicle,
            order_offsets,
            order_indices,
            deadline_remaining,
            batch_size_array,
        )
        if native_payload is None
        else native_payload
    )
    native_completed = time.perf_counter()
    native_elapsed = (
        native_completed - native_started
        if native_kernel_seconds is None
        else native_kernel_seconds
    )
    if native_elapsed < 0.0 or not math.isfinite(native_elapsed):
        raise RuntimeError("native exact charging elapsed time is invalid")
    metrics.native_kernel_seconds += native_elapsed
    metrics.native_invocations += 1
    (
        path_offsets,
        path_indices,
        statuses,
        reasons,
        route_metrics,
        label_counters,
        batch_counters,
    ) = _validate_native_payload(payload, len(valid_orders), len(context.node_names))

    metrics.transition_batches += int(batch_counters[5])
    metrics.transitions += int(batch_counters[6])
    metrics.checkpoint_count += int(batch_counters[7])
    metrics.work_batches = max(metrics.work_batches, int(batch_counters[4]))
    metrics.batch_launches = max(metrics.batch_launches, int(batch_counters[8]))
    if int(batch_counters[0]) != len(valid_orders):
        raise RuntimeError("native exact charging route-count reconciliation failed")
    if int(batch_counters[9]) != batch_size:
        raise RuntimeError("native exact charging batch-size reconciliation failed")

    unpacking_started = time.perf_counter()
    interrupted_indices: list[int] = []
    native_completed_count = 0
    for native_index, original_index in enumerate(valid_indices):
        status = int(statuses[native_index])
        reason = int(reasons[native_index])
        if status == 2:
            if reason != 2:
                raise RuntimeError("native interrupted result has an invalid reason code")
            interrupted_indices.append(original_index)
            continue
        native_completed_count += 1
        completed_indices.add(original_index)
        counters = label_counters[native_index]
        if status == 1:
            if reason != 1:
                raise RuntimeError("native infeasible result has an invalid reason code")
            results[original_index] = ChargingSubproblemResult(
                False,
                (),
                float("inf"),
                0.0,
                0.0,
                0.0,
                int(counters[0]),
                int(counters[1]),
                int(counters[2]),
                native_elapsed,
                "no feasible station-insertion pattern for fixed customer order",
            )
            continue
        if status != 0 or reason != 0:
            raise RuntimeError("native exact charging returned an unknown status/reason code")
        first = int(path_offsets[native_index])
        last = int(path_offsets[native_index + 1])
        route = tuple(context.node_names[int(index)] for index in path_indices[first:last])
        routed_customers = tuple(
            name for name in route if instance.by_name[name].kind is NodeType.CUSTOMER
        )
        if routed_customers != orders[original_index]:
            raise RuntimeError("native exact charging changed the fixed customer order")
        report = validate_routes(instance, [list(route)])
        violations = [violation for item in report.routes for violation in item.violations]
        if violations:
            raise RuntimeError(
                "native exact charging returned an invalid route: " + " | ".join(violations)
            )
        route_report = report.routes[0]
        native_values = route_metrics[native_index]
        expected_values = (
            route_report.distance,
            route_report.total_energy,
            route_report.charged_energy,
            route_report.charging_time,
        )
        if not all(
            math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-9)
            for actual, expected in zip(native_values, expected_values, strict=True)
        ):
            raise RuntimeError("native exact charging objective metrics failed replay")
        results[original_index] = ChargingSubproblemResult(
            True,
            route,
            route_report.distance,
            route_report.total_energy,
            route_report.charged_energy,
            route_report.charging_time,
            int(counters[0]),
            int(counters[1]),
            int(counters[2]),
            native_elapsed,
            "",
        )
    metrics.unpacking_seconds += time.perf_counter() - unpacking_started

    if int(batch_counters[1]) != len(valid_orders):
        raise RuntimeError("native exact charging started-call reconciliation failed")
    if int(batch_counters[2]) != native_completed_count:
        raise RuntimeError("native exact charging completed-call reconciliation failed")
    if int(batch_counters[3]) != len(interrupted_indices):
        raise RuntimeError("native exact charging interrupted-call reconciliation failed")
    metrics.completed_calls = len(completed_indices)
    metrics.interrupted_calls = len(orders) - metrics.completed_calls
    metrics.total_seconds = time.perf_counter() - started
    metrics.label_management_seconds = metrics.native_kernel_seconds
    if interrupted_indices:
        raise ExactBatchDeadlineExceeded(
            started_exact_calls=len(orders),
            completed_exact_calls=len(completed_indices),
            metrics=metrics,
            completed_indices=tuple(sorted(completed_indices)),
        )
    completed = tuple(result for result in results if result is not None)
    if len(completed) != len(orders):
        raise RuntimeError("native exact charging lost an ordered route result")
    return BatchChargingResult(completed, metrics)


def decode_exact_charging_batch_numeric(
    instance: Instance,
    customer_orders: Sequence[Sequence[str]],
    *,
    native_runtime: NativeKernelRuntime,
    batch_size: int,
    payload: object,
    native_kernel_seconds: float,
) -> BatchChargingResult:
    """Replay and decode an exact payload produced inside a larger native call."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    orders = tuple(tuple(order) for order in customer_orders)
    native_runtime.context.assert_matches(instance)
    metrics = BackendMetrics(
        ExactChargingBackend.CPU_BATCH.value,
        batch_size,
        work_batches=bool(orders),
        exact_calls=len(orders),
        batch_launches=int(bool(orders)),
        started_calls=len(orders),
        launch_occupancies=([len(orders)] if orders else []),
    )
    started = time.perf_counter()
    completed_indices: set[int] = set()

    def checkpoint() -> None:
        metrics.checkpoint_count += 1

    return _solve_exact_charging_native(
        instance,
        orders,
        runtime=native_runtime,
        batch_size=batch_size,
        deadline=None,
        started=started,
        metrics=metrics,
        checkpoint=checkpoint,
        completed_indices=completed_indices,
        native_payload=payload,
        native_kernel_seconds=native_kernel_seconds,
    )


def _validate_native_payload(
    payload: object,
    route_count: int,
    node_count: int,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    if not isinstance(payload, tuple) or len(payload) != 7:
        raise RuntimeError("native exact charging returned an invalid payload tuple")

    def require_array(
        value: object,
        dtype: np.dtype[Any],
        shape: tuple[int, ...],
        name: str,
    ) -> np.ndarray[Any, Any]:
        if not isinstance(value, np.ndarray):
            raise RuntimeError(f"native exact charging {name} is not a numeric array")
        if value.dtype != dtype or value.shape != shape or not value.flags.c_contiguous:
            raise RuntimeError(f"native exact charging {name} has an invalid schema")
        return value

    path_offsets = require_array(payload[0], np.dtype(np.int64), (route_count + 1,), "path_offsets")
    path_indices_value = payload[1]
    if (
        not isinstance(path_indices_value, np.ndarray)
        or path_indices_value.dtype != np.dtype(np.int64)
        or path_indices_value.ndim != 1
        or not path_indices_value.flags.c_contiguous
    ):
        raise RuntimeError("native exact charging path_indices has an invalid schema")
    path_indices = cast(npt.NDArray[np.int64], path_indices_value)
    statuses = require_array(payload[2], np.dtype(np.int64), (route_count,), "status")
    reasons = require_array(payload[3], np.dtype(np.int64), (route_count,), "reason")
    route_metrics = require_array(payload[4], np.dtype(np.float64), (route_count, 4), "metrics")
    label_counters = require_array(
        payload[5], np.dtype(np.int64), (route_count, 3), "label_counters"
    )
    batch_counters = require_array(payload[6], np.dtype(np.int64), (10,), "batch_counters")
    if int(path_offsets[0]) != 0 or int(path_offsets[-1]) != len(path_indices):
        raise RuntimeError("native exact charging path CSR offsets are inconsistent")
    if np.any(path_offsets[1:] < path_offsets[:-1]):
        raise RuntimeError("native exact charging path CSR offsets are not monotonic")
    if len(path_indices) and (int(path_indices.min()) < 0 or int(path_indices.max()) >= node_count):
        raise RuntimeError("native exact charging path CSR contains an invalid node index")
    return (
        cast(npt.NDArray[np.int64], path_offsets),
        path_indices,
        cast(npt.NDArray[np.int64], statuses),
        cast(npt.NDArray[np.int64], reasons),
        cast(npt.NDArray[np.float64], route_metrics),
        cast(npt.NDArray[np.int64], label_counters),
        cast(npt.NDArray[np.int64], batch_counters),
    )


def _pop_live_entry(state: _SearchState) -> _QueueEntry | None:
    while state.queue:
        entry = heapq.heappop(state.queue)
        key = (entry.label.progress, entry.label.node_name)
        if entry.label in state.labels.get(key, []):
            return entry
    return None
