from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass

from evrptw.charging import ChargingSubproblemResult, solve_exact_charging
from evrptw.models import Instance, Node, NodeType
from evrptw.objective import (
    ObjectiveComparison,
    SolutionObjective,
    accept_annealing_move,
    compare_objectives,
)
from evrptw.validation import validate_routes

_INFEASIBLE_COST = 1e12


@dataclass(slots=True)
class OperatorStatistics:
    calls: int = 0
    accepted: int = 0
    improved: int = 0
    best: int = 0
    weight: float = 1.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ALNSResult:
    feasible: bool
    routes: tuple[tuple[str, ...], ...]
    customer_sequences: tuple[tuple[str, ...], ...]
    objective: SolutionObjective | None
    vehicle_count: int
    total_energy: float
    total_charged_energy: float
    total_charging_time: float
    iterations: int
    accepted_moves: int
    improving_moves: int
    rejected_moves: int
    first_feasible_time: float
    best_time: float
    runtime_seconds: float
    charging_subproblem_calls: int
    charging_subproblem_time: float
    charging_labels_generated: int
    charging_labels_pruned: int
    destroy_statistics: dict[str, dict[str, float | int]]
    repair_statistics: dict[str, dict[str, float | int]]
    failure_reason: str

    @property
    def objective_value(self) -> float:
        return self.objective.total_distance if self.objective is not None else float("inf")


@dataclass(frozen=True, slots=True)
class _EvaluatedSolution:
    sequences: tuple[tuple[str, ...], ...]
    charging: tuple[ChargingSubproblemResult, ...]
    feasible: bool
    objective: SolutionObjective | None


class _Evaluator:
    def __init__(self, instance: Instance, *, deadline: float) -> None:
        self.instance = instance
        self.deadline = deadline
        self.cache: dict[tuple[str, ...], ChargingSubproblemResult] = {}
        self.calls = 0
        self.runtime = 0.0
        self.labels_generated = 0
        self.labels_pruned = 0

    def route(self, sequence: tuple[str, ...]) -> ChargingSubproblemResult:
        if sequence not in self.cache:
            if time.perf_counter() >= self.deadline:
                raise _TimeLimitReached
            result = solve_exact_charging(self.instance, sequence)
            self.cache[sequence] = result
            self.calls += 1
            self.runtime += result.runtime_seconds
            self.labels_generated += result.labels_generated
            self.labels_pruned += result.labels_pruned
        return self.cache[sequence]

    def solution(self, sequences: tuple[tuple[str, ...], ...]) -> _EvaluatedSolution:
        clean = tuple(sequence for sequence in sequences if sequence)
        charging = tuple(self.route(sequence) for sequence in clean)
        feasible = bool(clean) and all(result.feasible for result in charging)
        if not feasible:
            return _EvaluatedSolution(clean, charging, False, None)
        charging_count = sum(
            1
            for result in charging
            for name in result.route
            if self.instance.by_name[name].kind is NodeType.STATION
        )
        objective = SolutionObjective(
            len(clean),
            sum(result.distance for result in charging),
            sum(result.charging_time for result in charging),
            charging_count,
        )
        return _EvaluatedSolution(clean, charging, True, objective)


class _TimeLimitReached(Exception):
    pass


def solve_alns(
    instance: Instance,
    *,
    seed: int,
    max_iterations: int = 2_000,
    time_limit_seconds: float = 60.0,
    removal_fraction: float = 0.2,
) -> ALNSResult:
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    if not 0.0 < removal_fraction <= 1.0:
        raise ValueError("removal_fraction must be in (0, 1]")

    started = time.perf_counter()
    rng = random.Random(seed)
    evaluator = _Evaluator(instance, deadline=started + time_limit_seconds)
    try:
        initial_sequences = _construct_initial_solution(instance, evaluator)
    except _TimeLimitReached:
        return _failed_result(started, evaluator, "time limit reached during initial construction")
    current = evaluator.solution(initial_sequences)
    if not current.feasible:
        return _failed_result(started, evaluator, "no feasible singleton initial solution")
    if current.objective is None:
        raise RuntimeError("feasible ALNS initial solution is missing its objective")

    best = current
    first_feasible_time = time.perf_counter() - started
    best_time = first_feasible_time
    destroy_stats = {name: OperatorStatistics() for name in ("random", "worst", "related")}
    repair_stats = {name: OperatorStatistics() for name in ("greedy", "regret2", "energy")}
    accepted = 0
    improved = 0
    rejected = 0
    completed_iterations = 0
    initial_temperature = max(1.0, current.objective.total_distance * 0.05)

    for iteration in range(max_iterations):
        elapsed = time.perf_counter() - started
        if elapsed >= time_limit_seconds:
            break
        completed_iterations = iteration + 1
        destroy_name = _weighted_choice(rng, destroy_stats)
        repair_name = _weighted_choice(rng, repair_stats)
        destroy_stats[destroy_name].calls += 1
        repair_stats[repair_name].calls += 1

        remove_count = max(1, math.ceil(len(instance.customers) * removal_fraction))
        if len(instance.customers) > 20:
            remove_count = min(remove_count, 3)
        partial, removed = _destroy(instance, current.sequences, remove_count, destroy_name, rng)
        try:
            candidate_sequences = _repair(partial, removed, repair_name, evaluator, instance, rng)
            candidate = evaluator.solution(candidate_sequences)
        except _TimeLimitReached:
            break

        temperature = initial_temperature * max(0.001, 1.0 - iteration / max_iterations)
        accept = (
            candidate.feasible
            and candidate.objective is not None
            and accept_annealing_move(
                current.objective,
                candidate.objective,
                temperature=max(temperature, 1e-12),
                random_draw=rng.random(),
            )
        )
        if not accept:
            rejected += 1
            _update_weight(destroy_stats[destroy_name], 0.0)
            _update_weight(repair_stats[repair_name], 0.0)
            continue

        accepted += 1
        destroy_stats[destroy_name].accepted += 1
        repair_stats[repair_name].accepted += 1
        reward = 1.0
        if candidate.objective is None:
            raise RuntimeError("accepted ALNS candidate is missing its objective")
        if compare_objectives(candidate.objective, current.objective) is ObjectiveComparison.BETTER:
            improved += 1
            reward = 4.0
            destroy_stats[destroy_name].improved += 1
            repair_stats[repair_name].improved += 1
        current = candidate
        if best.objective is None:
            raise RuntimeError("feasible ALNS incumbent is missing its objective")
        if compare_objectives(candidate.objective, best.objective) is ObjectiveComparison.BETTER:
            best = candidate
            best_time = time.perf_counter() - started
            reward = 8.0
            destroy_stats[destroy_name].best += 1
            repair_stats[repair_name].best += 1
        _update_weight(destroy_stats[destroy_name], reward)
        _update_weight(repair_stats[repair_name], reward)

    routes = tuple(result.route for result in best.charging)
    report = validate_routes(instance, [list(route) for route in routes])
    if not report.feasible:
        raise RuntimeError("ALNS best solution failed the unified validator")
    if best.objective is None:
        raise RuntimeError("feasible ALNS result is missing its objective")
    return ALNSResult(
        feasible=True,
        routes=routes,
        customer_sequences=best.sequences,
        objective=best.objective,
        vehicle_count=len(routes),
        total_energy=sum(result.total_energy for result in best.charging),
        total_charged_energy=sum(result.charged_energy for result in best.charging),
        total_charging_time=sum(result.charging_time for result in best.charging),
        iterations=completed_iterations,
        accepted_moves=accepted,
        improving_moves=improved,
        rejected_moves=rejected,
        first_feasible_time=first_feasible_time,
        best_time=best_time,
        runtime_seconds=time.perf_counter() - started,
        charging_subproblem_calls=evaluator.calls,
        charging_subproblem_time=evaluator.runtime,
        charging_labels_generated=evaluator.labels_generated,
        charging_labels_pruned=evaluator.labels_pruned,
        destroy_statistics={name: stats.to_dict() for name, stats in destroy_stats.items()},
        repair_statistics={name: stats.to_dict() for name, stats in repair_stats.items()},
        failure_reason="",
    )


def _construct_initial_solution(
    instance: Instance,
    evaluator: _Evaluator,
) -> tuple[tuple[str, ...], ...]:
    sequences: list[tuple[str, ...]] = []
    customers = sorted(
        instance.customers,
        key=lambda node: (node.due_date, node.ready_time, node.name),
    )
    if len(customers) > 20:
        return _construct_large_initial_solution(instance, customers, evaluator)
    for customer in customers:
        best: tuple[float, int, tuple[str, ...]] | None = None
        for route_index in range(len(sequences) + 1):
            base = sequences[route_index] if route_index < len(sequences) else ()
            for position in range(len(base) + 1):
                candidate = (*base[:position], customer.name, *base[position:])
                result = evaluator.route(candidate)
                if not result.feasible:
                    continue
                old_distance = evaluator.route(base).distance if base else 0.0
                key = (result.distance - old_distance, route_index, candidate)
                if best is None or key < best:
                    best = key
        if best is None:
            return ()
        _, route_index, sequence = best
        if route_index == len(sequences):
            sequences.append(sequence)
        else:
            sequences[route_index] = sequence
    return tuple(sequences)


def _construct_large_initial_solution(
    instance: Instance,
    customers: list[Node],
    evaluator: _Evaluator,
) -> tuple[tuple[str, ...], ...]:
    from evrptw.baselines.ortools_vrptw import solve_vrptw

    minimum_fleet = math.ceil(
        sum(customer.demand for customer in customers) / instance.vehicle.load_capacity
    )
    constructor = solve_vrptw(
        instance,
        vehicle_count=min(
            len(customers),
            max(minimum_fleet + 5, math.ceil(len(customers) / 4)),
        ),
        time_limit_seconds=1,
        first_solution_strategy="PARALLEL_CHEAPEST_INSERTION",
    )
    routes = constructor["routes"]
    if routes:
        sequences: list[tuple[str, ...]] = []
        for route in routes:
            customer_sequence = tuple(
                name for name in route if name != instance.depot.name
            )
            split = _split_until_charging_feasible(customer_sequence, evaluator)
            if not split:
                return ()
            sequences.extend(split)
        return tuple(sequences)

    return _construct_sequential_initial_solution(customers, evaluator)


def _construct_sequential_initial_solution(
    customers: list[Node],
    evaluator: _Evaluator,
) -> tuple[tuple[str, ...], ...]:
    sequences: list[tuple[str, ...]] = []
    current: tuple[str, ...] = ()
    for customer in customers:
        candidates: list[tuple[float, tuple[str, ...]]] = []
        for position in range(len(current) + 1):
            candidate = (*current[:position], customer.name, *current[position:])
            result = evaluator.route(candidate)
            if result.feasible:
                candidates.append((result.distance, candidate))
        if candidates:
            current = min(candidates)[1]
            continue
        if current:
            sequences.append(current)
        current = (customer.name,)
        if not evaluator.route(current).feasible:
            return ()
    if current:
        sequences.append(current)
    return tuple(sequences)


def _split_until_charging_feasible(
    sequence: tuple[str, ...],
    evaluator: _Evaluator,
) -> tuple[tuple[str, ...], ...]:
    if evaluator.route(sequence).feasible:
        return (sequence,)
    if len(sequence) <= 1:
        return ()
    midpoint = len(sequence) // 2
    left = _split_until_charging_feasible(sequence[:midpoint], evaluator)
    right = _split_until_charging_feasible(sequence[midpoint:], evaluator)
    return (*left, *right) if left and right else ()


def _destroy(
    instance: Instance,
    sequences: tuple[tuple[str, ...], ...],
    count: int,
    name: str,
    rng: random.Random,
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    customers = [customer for sequence in sequences for customer in sequence]
    count = min(count, len(customers))
    if name == "random":
        removed = rng.sample(customers, count)
    elif name == "worst":
        contribution: list[tuple[float, str]] = []
        by_name = instance.by_name
        for sequence in sequences:
            chain = (instance.depot.name, *sequence, instance.depot.name)
            for index, customer in enumerate(sequence, start=1):
                before = by_name[chain[index - 1]]
                node = by_name[customer]
                after = by_name[chain[index + 1]]
                saving = (
                    before.distance_to(node)
                    + node.distance_to(after)
                    - before.distance_to(after)
                )
                contribution.append((saving, customer))
        removed = [customer for _, customer in sorted(contribution, reverse=True)[:count]]
    else:
        anchor = rng.choice(customers)
        anchor_node = instance.by_name[anchor]
        removed = [
            customer
            for _, customer in sorted(
                (anchor_node.distance_to(instance.by_name[name]), name) for name in customers
            )[:count]
        ]
    removed_set = set(removed)
    partial = tuple(
        tuple(name for name in sequence if name not in removed_set)
        for sequence in sequences
    )
    return tuple(sequence for sequence in partial if sequence), tuple(removed)


def _repair(
    partial: tuple[tuple[str, ...], ...],
    removed: tuple[str, ...],
    name: str,
    evaluator: _Evaluator,
    instance: Instance,
    rng: random.Random,
) -> tuple[tuple[str, ...], ...]:
    sequences = list(partial)
    pending = list(removed)
    while pending:
        options: dict[str, list[tuple[float, int, tuple[str, ...]]]] = {}
        for customer in pending:
            options[customer] = _insertion_options(sequences, customer, evaluator, instance, name)
        feasible_customers = [customer for customer, values in options.items() if values]
        if not feasible_customers:
            return ()
        if name == "regret2":
            customer = max(
                feasible_customers,
                key=lambda item: (
                    (options[item][1][0] - options[item][0][0])
                    if len(options[item]) > 1
                    else _INFEASIBLE_COST,
                    rng.random(),
                ),
            )
        else:
            customer = min(feasible_customers, key=lambda item: (options[item][0][0], item))
        _, route_index, sequence = options[customer][0]
        if route_index == len(sequences):
            sequences.append(sequence)
        else:
            sequences[route_index] = sequence
        pending.remove(customer)
    return tuple(sequences)


def _insertion_options(
    sequences: list[tuple[str, ...]],
    customer: str,
    evaluator: _Evaluator,
    instance: Instance,
    mode: str,
) -> list[tuple[float, int, tuple[str, ...]]]:
    options: list[tuple[float, int, tuple[str, ...]]] = []
    for route_index in range(len(sequences) + 1):
        base = sequences[route_index] if route_index < len(sequences) else ()
        old = evaluator.route(base).distance if base else 0.0
        for position in range(len(base) + 1):
            candidate = (*base[:position], customer, *base[position:])
            result = evaluator.route(candidate)
            if not result.feasible:
                continue
            score = result.distance - old
            if mode == "energy":
                score += 0.05 * result.charged_energy
            demand = sum(instance.by_name[name].demand for name in candidate)
            if demand > instance.vehicle.load_capacity + 1e-9:
                continue
            options.append((score, route_index, candidate))
    return sorted(options)


def _weighted_choice(rng: random.Random, statistics: dict[str, OperatorStatistics]) -> str:
    names = list(statistics)
    return rng.choices(names, weights=[statistics[name].weight for name in names], k=1)[0]


def _update_weight(statistics: OperatorStatistics, reward: float, reaction: float = 0.2) -> None:
    statistics.weight = max(0.05, (1.0 - reaction) * statistics.weight + reaction * reward)


def _failed_result(started: float, evaluator: _Evaluator, reason: str) -> ALNSResult:
    return ALNSResult(
        feasible=False,
        routes=(),
        customer_sequences=(),
        objective=None,
        vehicle_count=0,
        total_energy=0.0,
        total_charged_energy=0.0,
        total_charging_time=0.0,
        iterations=0,
        accepted_moves=0,
        improving_moves=0,
        rejected_moves=0,
        first_feasible_time=float("inf"),
        best_time=float("inf"),
        runtime_seconds=time.perf_counter() - started,
        charging_subproblem_calls=evaluator.calls,
        charging_subproblem_time=evaluator.runtime,
        charging_labels_generated=evaluator.labels_generated,
        charging_labels_pruned=evaluator.labels_pruned,
        destroy_statistics={},
        repair_statistics={},
        failure_reason=reason,
    )
