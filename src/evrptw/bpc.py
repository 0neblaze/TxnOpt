from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from evrptw.charging import ChargingSubproblemResult, solve_exact_charging
from evrptw.models import Instance
from evrptw.validation import validate_routes

_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class RouteColumn:
    customers: tuple[str, ...]
    route: tuple[str, ...]
    cost: float
    charging: ChargingSubproblemResult


@dataclass(frozen=True, slots=True)
class BidirectionalPricingResult:
    columns: tuple[RouteColumn, ...]
    forward_labels: int
    backward_labels: int
    joined_labels: int
    labels_pruned_capacity: int
    labels_pruned_infeasible: int
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class BPCResult:
    status: str
    proven_optimal: bool
    routes: tuple[tuple[str, ...], ...]
    objective_value: float
    root_lower_bound: float
    final_lower_bound: float
    incumbent: float
    optimality_gap: float
    branch_nodes: int
    generated_columns: int
    active_columns: int
    pricing_iterations: int
    forward_labels: int
    backward_labels: int
    joined_labels: int
    labels_pruned_capacity: int
    labels_pruned_infeasible: int
    cuts_added: int
    runtime_seconds: float
    failure_reason: str


@dataclass(order=True, slots=True)
class _BranchNode:
    priority: float
    serial: int
    fixed_zero: frozenset[int] = field(compare=False)
    fixed_one: frozenset[int] = field(compare=False)


@dataclass(frozen=True, slots=True)
class _MasterResult:
    feasible: bool
    objective: float
    values: np.ndarray
    active: tuple[int, ...]
    lower_bound: float
    pricing_iterations: int


def generate_columns_bidirectionally(
    instance: Instance,
    *,
    max_customers: int = 8,
) -> BidirectionalPricingResult:
    """Generate every feasible elementary route by joining forward/backward labels."""

    started = time.perf_counter()
    customers = tuple(customer.name for customer in instance.customers)
    if len(customers) > max_customers:
        raise ValueError(
            f"bidirectional exact pricing supports at most {max_customers} customers, "
            f"received {len(customers)}"
        )
    split = math.ceil(len(customers) / 2)
    forward: set[tuple[str, ...]] = {()}
    backward: set[tuple[str, ...]] = {()}
    pruned_capacity = 0

    for length in range(1, split + 1):
        for label in itertools.permutations(customers, length):
            if _demand(instance, label) <= instance.vehicle.load_capacity + _EPSILON:
                forward.add(label)
            else:
                pruned_capacity += 1
    for length in range(1, len(customers) - split + 1):
        for label in itertools.permutations(customers, length):
            if _demand(instance, label) <= instance.vehicle.load_capacity + _EPSILON:
                backward.add(label)
            else:
                pruned_capacity += 1

    sequences: set[tuple[str, ...]] = set()
    joined = 0
    for left in forward:
        if left:
            sequences.add(left)
        left_set = set(left)
        for right_reversed in backward:
            right = tuple(reversed(right_reversed))
            if (not right or left_set.isdisjoint(right)) and (left or right):
                joined += 1
                sequences.add((*left, *right))

    columns: list[RouteColumn] = []
    pruned_infeasible = 0
    for sequence in sorted(sequences, key=lambda item: (len(item), item)):
        if _demand(instance, sequence) > instance.vehicle.load_capacity + _EPSILON:
            pruned_capacity += 1
            continue
        charging = solve_exact_charging(instance, sequence)
        if not charging.feasible:
            pruned_infeasible += 1
            continue
        columns.append(RouteColumn(sequence, charging.route, charging.distance, charging))

    return BidirectionalPricingResult(
        tuple(columns),
        len(forward),
        len(backward),
        joined,
        pruned_capacity,
        pruned_infeasible,
        time.perf_counter() - started,
    )


def solve_branch_price_and_cut(
    instance: Instance,
    *,
    time_limit_seconds: float = 60.0,
    max_customers: int = 8,
) -> BPCResult:
    started = time.perf_counter()
    pricing = generate_columns_bidirectionally(instance, max_customers=max_customers)
    columns = pricing.columns
    customers = tuple(customer.name for customer in instance.customers)
    if not columns:
        return _bpc_failure(started, pricing, "pricing generated no feasible columns")
    singleton = {
        index
        for index, column in enumerate(columns)
        if len(column.customers) == 1
    }
    if {name for index in singleton for name in columns[index].customers} != set(customers):
        return _bpc_failure(
            started,
            pricing,
            "at least one customer has no feasible singleton column",
        )

    queue = [_BranchNode(0.0, 0, frozenset(), frozenset())]
    serial = 1
    incumbent = float("inf")
    incumbent_columns: tuple[int, ...] = ()
    root_lower_bound = float("inf")
    final_lower_bound = float("inf")
    explored = 0
    pricing_iterations = 0
    active_seen: set[int] = set(singleton)
    timed_out = False

    while queue:
        if time.perf_counter() - started >= time_limit_seconds:
            timed_out = True
            break
        node = heapq.heappop(queue)
        master = _solve_node(
            instance,
            columns,
            singleton,
            node.fixed_zero,
            node.fixed_one,
            exhaustive_pool=explored > 0,
        )
        explored += 1
        pricing_iterations += master.pricing_iterations
        active_seen.update(master.active)
        if explored == 1:
            root_lower_bound = master.lower_bound
        if not master.feasible or master.lower_bound >= incumbent - _EPSILON:
            continue
        final_lower_bound = min((item.priority for item in queue), default=master.lower_bound)

        fractional = [
            (column_index, value)
            for column_index, value in zip(master.active, master.values, strict=True)
            if _EPSILON < value < 1.0 - _EPSILON
        ]
        if not fractional:
            chosen = tuple(
                column_index
                for column_index, value in zip(master.active, master.values, strict=True)
                if value >= 1.0 - _EPSILON
            )
            if master.objective < incumbent:
                incumbent = master.objective
                incumbent_columns = chosen
            continue

        branch_index, _ = min(fractional, key=lambda item: abs(item[1] - 0.5))
        heapq.heappush(
            queue,
            _BranchNode(
                master.lower_bound,
                serial,
                node.fixed_zero | {branch_index},
                node.fixed_one,
            ),
        )
        serial += 1
        heapq.heappush(
            queue,
            _BranchNode(
                master.lower_bound,
                serial,
                node.fixed_zero,
                node.fixed_one | {branch_index},
            ),
        )
        serial += 1

    if math.isinf(incumbent):
        return _bpc_failure(started, pricing, "branch-and-price found no integer incumbent")
    if not timed_out:
        final_lower_bound = incumbent
    elif queue:
        final_lower_bound = min(item.priority for item in queue)
    gap = max(0.0, (incumbent - final_lower_bound) / max(abs(incumbent), _EPSILON))
    routes = tuple(columns[index].route for index in incumbent_columns)
    report = validate_routes(instance, [list(route) for route in routes])
    if not report.feasible:
        raise RuntimeError("Branch-Price-and-Cut incumbent failed unified validation")
    return BPCResult(
        status="timeout" if timed_out else "optimal",
        proven_optimal=not timed_out and gap <= _EPSILON,
        routes=routes,
        objective_value=incumbent,
        root_lower_bound=root_lower_bound,
        final_lower_bound=final_lower_bound,
        incumbent=incumbent,
        optimality_gap=gap,
        branch_nodes=explored,
        generated_columns=len(columns),
        active_columns=len(active_seen),
        pricing_iterations=pricing_iterations,
        forward_labels=pricing.forward_labels,
        backward_labels=pricing.backward_labels,
        joined_labels=pricing.joined_labels,
        labels_pruned_capacity=pricing.labels_pruned_capacity,
        labels_pruned_infeasible=pricing.labels_pruned_infeasible,
        cuts_added=explored,
        runtime_seconds=time.perf_counter() - started,
        failure_reason="",
    )


def _solve_node(
    instance: Instance,
    columns: tuple[RouteColumn, ...],
    initial: set[int],
    fixed_zero: frozenset[int],
    fixed_one: frozenset[int],
    *,
    exhaustive_pool: bool,
) -> _MasterResult:
    customers = tuple(customer.name for customer in instance.customers)
    customer_index = {name: index for index, name in enumerate(customers)}
    if exhaustive_pool:
        active = sorted(set(range(len(columns))) - set(fixed_zero))
    else:
        active = sorted((initial | set(fixed_one)) - set(fixed_zero))
    iterations = 0
    while True:
        iterations += 1
        matrix = np.zeros((len(customers), len(active)))
        for position, column_index in enumerate(active):
            for customer in columns[column_index].customers:
                matrix[customer_index[customer], position] = 1.0
        costs = np.array([columns[index].cost for index in active])
        fleet_lb = math.ceil(
            sum(customer.demand for customer in instance.customers)
            / instance.vehicle.load_capacity
        )
        bounds = [
            (1.0, 1.0) if column_index in fixed_one else (0.0, 1.0)
            for column_index in active
        ]
        result = linprog(
            costs,
            A_ub=-np.ones((1, len(active))),
            b_ub=np.array([-float(fleet_lb)]),
            A_eq=matrix,
            b_eq=np.ones(len(customers)),
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            return _MasterResult(
                False,
                float("inf"),
                np.array([]),
                tuple(active),
                float("inf"),
                iterations,
            )

        if exhaustive_pool:
            values = np.asarray(result.x, dtype=float)
            objective = float(result.fun)
            return _MasterResult(True, objective, values, tuple(active), objective, iterations)

        duals = result.eqlin.marginals
        fleet_dual = float(result.ineqlin.marginals[0])
        candidates: list[tuple[float, int]] = []
        active_set = set(active)
        for index, column in enumerate(columns):
            if index in active_set or index in fixed_zero:
                continue
            covered_duals = sum(
                duals[customer_index[name]] for name in column.customers
            )
            reduced_cost = column.cost - covered_duals
            reduced_cost += fleet_dual
            if reduced_cost < -_EPSILON:
                candidates.append((reduced_cost, index))
        if not candidates:
            values = np.asarray(result.x, dtype=float)
            objective = float(result.fun)
            return _MasterResult(True, objective, values, tuple(active), objective, iterations)
        active.extend(index for _, index in sorted(candidates)[: min(20, len(candidates))])
        active.sort()


def _demand(instance: Instance, customers: tuple[str, ...]) -> float:
    return sum(instance.by_name[name].demand for name in customers)


def _bpc_failure(
    started: float,
    pricing: BidirectionalPricingResult,
    reason: str,
) -> BPCResult:
    return BPCResult(
        "infeasible",
        False,
        (),
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
        0,
        len(pricing.columns),
        0,
        0,
        pricing.forward_labels,
        pricing.backward_labels,
        pricing.joined_labels,
        pricing.labels_pruned_capacity,
        pricing.labels_pruned_infeasible,
        0,
        time.perf_counter() - started,
        reason,
    )
