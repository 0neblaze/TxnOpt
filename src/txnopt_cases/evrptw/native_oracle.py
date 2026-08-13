"""EVRPTW Oracle backed by one ``txnopt-native-round-v1`` call per round."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from txnopt import _native
from txnopt_cases.evrptw.charging import ChargingSubproblemResult
from txnopt_cases.evrptw.models import Instance, NodeType
from txnopt_cases.evrptw.oracle import EVRPTWOracle, EVRPTWPlan, EVRPTWSolution

_KIND_CODE = {
    NodeType.DEPOT: 0,
    NodeType.CUSTOMER: 1,
    NodeType.STATION: 2,
}


class NativeEVRPTWOracle(EVRPTWOracle):
    """Pack one immutable context and keep parallelism below the round seam."""

    internal_parallelism = True

    def __init__(
        self,
        instance: Instance,
        *,
        worker_count: int,
        batch_size: int = 64,
    ) -> None:
        super().__init__(instance)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("native EVRPTW batch_size must be a positive integer")
        self._worker_count = worker_count
        self._batch_size = batch_size
        self._names = tuple(node.name for node in instance.nodes)
        self._name_to_index = {
            name: index for index, name in enumerate(self._names)
        }
        distance = np.asarray(
            [
                [instance.distance(origin, destination) for destination in self._names]
                for origin in self._names
            ],
            dtype=np.float64,
        )
        reachable = _optimistic_reachability(instance, distance)
        self._context = _native.EVRPTWContext(
            np.asarray([_KIND_CODE[node.kind] for node in instance.nodes], dtype=np.int64),
            np.asarray([node.demand for node in instance.nodes], dtype=np.float64),
            np.asarray([node.ready_time for node in instance.nodes], dtype=np.float64),
            np.asarray([node.due_date for node in instance.nodes], dtype=np.float64),
            np.asarray([node.service_time for node in instance.nodes], dtype=np.float64),
            np.ascontiguousarray(distance),
            reachable,
            np.asarray(
                (
                    instance.vehicle.battery_capacity,
                    instance.vehicle.load_capacity,
                    instance.vehicle.consumption_rate,
                    instance.vehicle.inverse_refueling_rate,
                    instance.vehicle.average_velocity,
                ),
                dtype=np.float64,
            ),
            worker_count,
        )
        self._last_receipt: Mapping[str, object] | None = None

    @property
    def worker_count(self) -> int:
        return self._worker_count

    @property
    def last_receipt(self) -> Mapping[str, object] | None:
        return self._last_receipt

    def evaluate_batch(
        self,
        candidates: Sequence[EVRPTWPlan],
        *,
        work_budget: int,
        deadline_ns: int | None,
    ) -> Sequence[EVRPTWSolution]:
        expected_work = sum(self.work_units(candidate) for candidate in candidates)
        if work_budget != expected_work:
            raise ValueError("native EVRPTW work budget must cover every exact route")
        routes = tuple(
            route for candidate in candidates for route in candidate.customer_routes
        )
        offsets = [0]
        indices: list[int] = []
        for route in routes:
            indices.extend(self._name_to_index[name] for name in route)
            offsets.append(len(indices))
        deadline_seconds = math.inf
        if deadline_ns is not None:
            deadline_seconds = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
            if deadline_seconds <= 0.0:
                raise TimeoutError("native EVRPTW round has no remaining deadline")
        output = self._context.exact_round_v1(
            np.asarray(offsets, dtype=np.int64),
            np.asarray(indices, dtype=np.int64),
            deadline_seconds,
            self._batch_size,
        )
        (
            path_offsets,
            path_indices,
            statuses,
            reasons,
            metrics,
            label_counters,
            semantic_counters,
            resolution_order,
            _physical_receipts,
            receipt,
        ) = output
        self._validate_projection(
            route_count=len(routes),
            path_offsets=path_offsets,
            statuses=statuses,
            reasons=reasons,
            metrics=metrics,
            label_counters=label_counters,
            semantic_counters=semantic_counters,
            resolution_order=resolution_order,
            receipt=receipt,
        )
        self._last_receipt = MappingProxyType(dict(receipt))
        if receipt["phase"] == "INTERRUPTED":
            raise TimeoutError("native EVRPTW exact round crossed its deadline")

        exact_results: list[ChargingSubproblemResult] = []
        for route_index in range(len(routes)):
            first = int(path_offsets[route_index])
            last = int(path_offsets[route_index + 1])
            status = int(statuses[route_index])
            metric = metrics[route_index]
            labels = label_counters[route_index]
            feasible = status == 0
            exact_results.append(
                ChargingSubproblemResult(
                    feasible=feasible,
                    route=(
                        tuple(self._names[int(index)] for index in path_indices[first:last])
                        if feasible
                        else ()
                    ),
                    distance=float(metric[0]),
                    total_energy=float(metric[1]),
                    charged_energy=float(metric[2]),
                    charging_time=float(metric[3]),
                    labels_generated=int(labels[0]),
                    labels_expanded=int(labels[1]),
                    labels_pruned=int(labels[2]),
                    runtime_seconds=0.0,
                    failure_reason=(
                        ""
                        if feasible
                        else "no feasible station-insertion pattern for fixed customer order"
                    ),
                )
            )

        solutions: list[EVRPTWSolution] = []
        route_offset = 0
        for candidate in candidates:
            route_end = route_offset + len(candidate.customer_routes)
            solutions.append(
                self._solution_from_exact(
                    candidate,
                    exact_results[route_offset:route_end],
                )
            )
            route_offset = route_end
        return tuple(solutions)

    @staticmethod
    def _validate_projection(
        *,
        route_count: int,
        path_offsets: npt.NDArray[np.int64],
        statuses: npt.NDArray[np.int64],
        reasons: npt.NDArray[np.int64],
        metrics: npt.NDArray[np.float64],
        label_counters: npt.NDArray[np.int64],
        semantic_counters: npt.NDArray[np.int64],
        resolution_order: npt.NDArray[np.int64],
        receipt: _native.NativeRoundReceipt,
    ) -> None:
        if (
            path_offsets.shape != (route_count + 1,)
            or statuses.shape != (route_count,)
            or reasons.shape != (route_count,)
            or metrics.shape != (route_count, 4)
            or label_counters.shape != (route_count, 3)
            or semantic_counters.shape != (5,)
            or semantic_counters.tolist()[:2] != [route_count, route_count]
            or int(semantic_counters[2]) + int(semantic_counters[3]) != route_count
            or resolution_order.tolist() != list(range(route_count))
            or receipt["protocol"] != "txnopt-native-round-v1"
            or receipt["fallback_count"] != 0
            or receipt["parallel_route_threshold"] != receipt["worker_count"] * 2
            or receipt["scheduled_worker_count"] < 1
            or receipt["scheduled_worker_count"] > receipt["worker_count"]
            or receipt["execution_policy"]
            not in {"serial_configured", "serial_small_batch", "parallel"}
            or (
                receipt["worker_count"] == 1
                and receipt["execution_policy"] != "serial_configured"
            )
            or (
                receipt["worker_count"] > 1
                and route_count < receipt["parallel_route_threshold"]
                and receipt["execution_policy"] != "serial_small_batch"
            )
            or (
                receipt["worker_count"] > 1
                and route_count >= receipt["parallel_route_threshold"]
                and receipt["execution_policy"] != "parallel"
            )
        ):
            raise RuntimeError("txnopt-native-round-v1 projection is inconsistent")


__all__ = ["NativeEVRPTWOracle"]


def _optimistic_reachability(
    instance: Instance,
    distance: npt.NDArray[np.float64],
) -> npt.NDArray[np.uint8]:
    """Pack the safe full-battery station/depot transitive reachability bound."""

    capacity = instance.vehicle.battery_capacity
    rate = instance.vehicle.consumption_rate
    safe = tuple(
        index
        for index, node in enumerate(instance.nodes)
        if node.kind in {NodeType.DEPOT, NodeType.STATION}
    )
    reachable = np.zeros((len(instance.nodes), len(instance.nodes)), dtype=np.uint8)
    for origin in range(len(instance.nodes)):
        frontier = [origin]
        visited_safe: set[int] = set()
        while frontier:
            current = frontier.pop()
            for destination in range(len(instance.nodes)):
                if distance[current, destination] * rate <= capacity + 1e-9:
                    reachable[origin, destination] = 1
            for safe_node in safe:
                if (
                    safe_node not in visited_safe
                    and distance[current, safe_node] * rate <= capacity + 1e-9
                ):
                    visited_safe.add(safe_node)
                    frontier.append(safe_node)
    return np.ascontiguousarray(reachable)
