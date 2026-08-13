"""EVRPTW plan values and exact-charging Oracle implementation."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from txnopt_cases.evrptw.charging import ChargingSubproblemResult, solve_exact_charging
from txnopt_cases.evrptw.models import Instance, NodeType
from txnopt_cases.evrptw.objective import SolutionObjective
from txnopt_cases.evrptw.validation import validate_routes

_DEFAULT_ROUTE_SCREEN_CACHE_CAPACITY: Final = 65_536


@dataclass(frozen=True, slots=True)
class EVRPTWPlan:
    customer_routes: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not self.customer_routes or any(not route for route in self.customer_routes):
            raise ValueError("EVRPTW plans require non-empty customer routes")


@dataclass(frozen=True, slots=True)
class EVRPTWSolution:
    plan: EVRPTWPlan
    routes: tuple[tuple[str, ...], ...]
    feasible: bool
    objective_value: SolutionObjective | None
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.feasible:
            if self.objective_value is None:
                raise ValueError("a feasible EVRPTW solution requires an objective")
            if len(self.routes) != len(self.plan.customer_routes):
                raise ValueError("exact routes must align with customer routes")
            if self.failure_reasons:
                raise ValueError("a feasible EVRPTW solution cannot have failure reasons")
        elif self.objective_value is not None or not self.failure_reasons:
            raise ValueError("an infeasible EVRPTW result requires failure reasons only")


class EVRPTWOracle:
    """Safe plan screening, ordered exact charging, and independent validation."""

    deterministic = True
    parallel_safe = True
    internal_parallelism = False

    def __init__(
        self,
        instance: Instance,
        *,
        route_screen_cache_capacity: int = _DEFAULT_ROUTE_SCREEN_CACHE_CAPACITY,
    ) -> None:
        if (
            isinstance(route_screen_cache_capacity, bool)
            or not isinstance(route_screen_cache_capacity, int)
            or route_screen_cache_capacity <= 0
        ):
            raise ValueError("route_screen_cache_capacity must be a positive integer")
        self._instance = instance
        self._instance_digest = _instance_digest(instance)
        self._route_screen_cache_capacity = route_screen_cache_capacity
        self._route_screen_cache: OrderedDict[tuple[str, ...], bool] = OrderedDict()
        self._route_screen_cache_hits = 0
        self._route_screen_cache_misses = 0
        self._route_screen_cache_evictions = 0

    @property
    def screening_statistics(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "route_screen_cache_hits": self._route_screen_cache_hits,
                "route_screen_cache_misses": self._route_screen_cache_misses,
                "route_screen_cache_size": len(self._route_screen_cache),
                "route_screen_cache_capacity": self._route_screen_cache_capacity,
                "route_screen_cache_evictions": self._route_screen_cache_evictions,
            }
        )

    def stable_key(self, candidate: EVRPTWPlan) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "instance": self._instance_digest,
                    "customer_routes": candidate.customer_routes,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def state_digest(self, state: EVRPTWSolution) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "instance": self._instance_digest,
                    "plan": state.plan.customer_routes,
                    "routes": state.routes,
                    "feasible": state.feasible,
                    "objective": (
                        None
                        if state.objective_value is None
                        else state.objective_value.key
                    ),
                    "failure_reasons": state.failure_reasons,
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def work_units(self, candidate: EVRPTWPlan) -> int:
        return len(candidate.customer_routes)

    def screen(self, candidates: Sequence[EVRPTWPlan]) -> Sequence[bool]:
        return tuple(self._screen_one(candidate) for candidate in candidates)

    def evaluate_batch(
        self,
        candidates: Sequence[EVRPTWPlan],
        *,
        work_budget: int,
        deadline_ns: int | None,
    ) -> Sequence[EVRPTWSolution]:
        if work_budget != sum(self.work_units(candidate) for candidate in candidates):
            raise ValueError("EVRPTW work budget must cover the complete ordered batch")
        results: list[EVRPTWSolution] = []
        for candidate in candidates:
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("EVRPTW exact batch crossed its deadline")
            results.append(self._evaluate_one(candidate))
        return tuple(results)

    def validate(self, state: EVRPTWSolution) -> None:
        if not self._screen_one(state.plan):
            raise ValueError("EVRPTW solution plan failed safe screening")
        if not state.feasible:
            if state.routes:
                raise ValueError("an infeasible EVRPTW result cannot expose partial routes")
            return
        report = validate_routes(self._instance, state.routes)
        if not report.feasible:
            raise ValueError(f"invalid EVRPTW solution: {report.violations}")
        objective = SolutionObjective.from_report(self._instance, report)
        claimed_objective = state.objective_value
        if claimed_objective is None or objective.key != claimed_objective.key:
            raise ValueError("EVRPTW objective differs from independent reconstruction")
        customer_routes = tuple(
            tuple(
                name
                for name in route
                if self._instance.by_name[name].kind is NodeType.CUSTOMER
            )
            for route in state.routes
        )
        if customer_routes != state.plan.customer_routes:
            raise ValueError("exact routes changed the declared customer order")

    def objective(self, state: EVRPTWSolution) -> SolutionObjective:
        self.validate(state)
        if state.objective_value is None:
            raise ValueError("an infeasible EVRPTW result has no objective")
        return state.objective_value

    def solve_initial(self, plan: EVRPTWPlan) -> EVRPTWSolution:
        """Build and validate one initial state outside the runtime work budget."""

        if not self._screen_one(plan):
            raise ValueError("initial EVRPTW plan failed safe screening")
        state = self._evaluate_one(plan)
        self.validate(state)
        if not state.feasible:
            raise ValueError("initial EVRPTW plan is exact-infeasible")
        return state

    def _screen_one(self, candidate: EVRPTWPlan) -> bool:
        expected = {customer.name for customer in self._instance.customers}
        flattened = tuple(name for route in candidate.customer_routes for name in route)
        counts = Counter(flattened)
        if set(flattened) != expected or any(count != 1 for count in counts.values()):
            return False
        by_name = self._instance.by_name
        for route in candidate.customer_routes:
            if any(
                name not in by_name or by_name[name].kind is not NodeType.CUSTOMER
                for name in route
            ):
                return False
            if (
                sum(by_name[name].demand for name in route)
                > self._instance.vehicle.load_capacity + 1e-9
            ):
                return False
            try:
                cached = self._route_screen_cache.pop(route)
            except KeyError:
                cached = _screen_route_safely(self._instance, route)
                self._route_screen_cache[route] = cached
                self._route_screen_cache_misses += 1
                if len(self._route_screen_cache) > self._route_screen_cache_capacity:
                    self._route_screen_cache.popitem(last=False)
                    self._route_screen_cache_evictions += 1
            else:
                self._route_screen_cache[route] = cached
                self._route_screen_cache_hits += 1
            if not cached:
                return False
        return True

    def _evaluate_one(self, candidate: EVRPTWPlan) -> EVRPTWSolution:
        if not self._screen_one(candidate):
            raise ValueError("EVRPTW candidate failed safe screening")
        exact_results = tuple(
            solve_exact_charging(self._instance, route)
            for route in candidate.customer_routes
        )
        return self._solution_from_exact(candidate, exact_results)

    def _solution_from_exact(
        self,
        candidate: EVRPTWPlan,
        exact_results: Sequence[ChargingSubproblemResult],
    ) -> EVRPTWSolution:
        failures = tuple(
            f"route {index}: {result.failure_reason}"
            for index, result in enumerate(exact_results)
            if not result.feasible
        )
        if failures:
            return EVRPTWSolution(candidate, (), False, None, failures)
        routes = tuple(result.route for result in exact_results)
        report = validate_routes(self._instance, routes)
        if not report.feasible:
            raise RuntimeError(
                "exact charging and the independent EVRPTW validator disagree: "
                + " | ".join(report.violations)
            )
        solution = EVRPTWSolution(
            candidate,
            routes,
            True,
            SolutionObjective.from_report(self._instance, report),
        )
        self.validate(solution)
        return solution


def _instance_digest(instance: Instance) -> str:
    payload = {
        "name": instance.name,
        "nodes": [
            (
                node.name,
                node.kind.value,
                node.x,
                node.y,
                node.demand,
                node.ready_time,
                node.due_date,
                node.service_time,
            )
            for node in instance.nodes
        ],
        "vehicle": (
            instance.vehicle.battery_capacity,
            instance.vehicle.load_capacity,
            instance.vehicle.consumption_rate,
            instance.vehicle.inverse_refueling_rate,
            instance.vehicle.average_velocity,
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _screen_route_safely(instance: Instance, route: tuple[str, ...]) -> bool:
    """Mirror the native optimistic screening contract without another ABI call."""

    epsilon = 1e-9
    by_name = instance.by_name
    chain = (instance.depot.name, *route, instance.depot.name)
    current_time = max(0.0, instance.depot.ready_time)
    earliest: dict[str, float] = {}
    min_slack = float("inf")
    for origin, destination in zip(chain, chain[1:], strict=False):
        node = by_name[destination]
        current_time += instance.distance(origin, destination) / instance.vehicle.average_velocity
        current_time = max(current_time, node.ready_time)
        slack = node.due_date - current_time
        min_slack = min(min_slack, slack)
        if slack < -epsilon:
            return False
        if node.kind is NodeType.CUSTOMER:
            earliest[destination] = current_time
            current_time += node.service_time

    latest_departure = instance.depot.due_date
    latest_arrivals: dict[str, float] = {}
    for index in range(len(chain) - 2, -1, -1):
        origin_node = by_name[chain[index]]
        destination_node = by_name[chain[index + 1]]
        if destination_node.kind is NodeType.CUSTOMER:
            latest_arrival = min(
                destination_node.due_date,
                latest_departure - destination_node.service_time,
            )
            latest_arrivals[destination_node.name] = latest_arrival
        else:
            latest_arrival = min(destination_node.due_date, latest_departure)
        latest_departure = latest_arrival - (
            instance.distance(origin_node.name, destination_node.name)
            / instance.vehicle.average_velocity
        )
    if any(latest_arrivals[name] - arrival < -epsilon for name, arrival in earliest.items()):
        return False
    if min_slack < -epsilon:
        return False

    if any(
        not _energy_reachable_optimistically(instance, origin, destination, epsilon=epsilon)
        for origin, destination in zip(chain, chain[1:], strict=False)
    ):
        return False
    recharge_nodes = (instance.depot, *instance.stations)
    structural_energy_lower_bound = max(
        (
            min(instance.distance(node.name, customer) for node in recharge_nodes)
            + min(instance.distance(customer, node.name) for node in recharge_nodes)
        )
        * instance.vehicle.consumption_rate
        for customer in route
    )
    return structural_energy_lower_bound <= instance.vehicle.battery_capacity + epsilon


def _energy_reachable_optimistically(
    instance: Instance,
    origin: str,
    destination: str,
    *,
    epsilon: float,
) -> bool:
    if origin == destination:
        return True
    capacity = instance.vehicle.battery_capacity
    rate = instance.vehicle.consumption_rate
    frontier = [origin]
    visited_recharge_nodes: set[str] = set()
    while frontier:
        current = frontier.pop()
        if instance.distance(current, destination) * rate <= capacity + epsilon:
            return True
        for station in instance.stations:
            if station.name == current or station.name in visited_recharge_nodes:
                continue
            if instance.distance(current, station.name) * rate <= capacity + epsilon:
                visited_recharge_nodes.add(station.name)
                frontier.append(station.name)
    return False


__all__ = ["EVRPTWOracle", "EVRPTWPlan", "EVRPTWSolution"]
