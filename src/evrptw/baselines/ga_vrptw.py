from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from evrptw.models import Instance, NodeType
from evrptw.validation import validate_routes


@dataclass(frozen=True, slots=True)
class ConvergenceRecord:
    generation: int
    best_objective: float
    average_objective: float
    feasible_rate: float
    best_feasible: bool


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    chromosome: tuple[str, ...]
    routes: tuple[tuple[str, ...], ...]
    objective: float
    score: float
    feasible: bool
    violations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GeneticResult:
    method: str
    instance_name: str
    customer_count: int
    seed: int
    population_size: int
    configured_generations: int
    generations: int
    crossover_probability: float
    mutation_probability: float
    objective_value: float
    score: float
    feasible: bool
    runtime_seconds: float
    time_limit_seconds: float | None
    terminated_by_time_limit: bool
    vehicle_count: int
    routes: tuple[tuple[str, ...], ...]
    violations: tuple[str, ...]
    convergence: tuple[ConvergenceRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _route_violations(instance: Instance, route: Sequence[str]) -> list[str]:
    by_name = instance.by_name
    depot = instance.depot
    violations: list[str] = []
    if len(route) < 2 or route[0] != depot.name or route[-1] != depot.name:
        violations.append("route must start and end at the depot")
    if any(name not in by_name for name in route):
        violations.append("route contains unknown nodes")
        return violations

    load = sum(by_name[name].demand for name in route if by_name[name].kind is NodeType.CUSTOMER)
    if load > instance.vehicle.load_capacity + 1e-9:
        violations.append("route load exceeds vehicle capacity")

    battery = instance.vehicle.battery_capacity
    elapsed = max(0.0, depot.ready_time)
    for origin_name, destination_name in zip(route, route[1:], strict=False):
        origin = by_name[origin_name]
        destination = by_name[destination_name]
        leg_distance = origin.distance_to(destination)
        battery -= leg_distance * instance.vehicle.consumption_rate
        elapsed += leg_distance / instance.vehicle.average_velocity
        if battery < -1e-9:
            violations.append("route depletes battery")
        elapsed = max(elapsed, destination.ready_time)
        if elapsed > destination.due_date + 1e-9:
            violations.append("route violates a time window")
        if destination.kind is NodeType.CUSTOMER:
            elapsed += destination.service_time
        elif destination.kind is NodeType.STATION:
            recharge = max(0.0, instance.vehicle.battery_capacity - battery)
            elapsed += recharge * instance.vehicle.inverse_refueling_rate
            battery = instance.vehicle.battery_capacity
    return violations


def decode_chromosome(instance: Instance, chromosome: Sequence[str]) -> list[list[str]]:
    depot = instance.depot.name
    routes: list[list[str]] = []
    current: list[str] = []

    for customer_name in chromosome:
        proposed = [depot, *current, customer_name, depot]
        if not current or not _route_violations(instance, proposed):
            current.append(customer_name)
            continue
        routes.append([depot, *current, depot])
        current = [customer_name]

    if current:
        routes.append([depot, *current, depot])
    return routes


def _evaluate(instance: Instance, chromosome: Sequence[str]) -> CandidateEvaluation:
    routes = decode_chromosome(instance, chromosome)
    report = validate_routes(instance, routes)
    route_violations = [
        violation for route_report in report.routes for violation in route_report.violations
    ]
    violations = (*report.violations, *route_violations)
    penalty = 1_000_000.0 * len(violations)
    score = report.total_distance + penalty
    return CandidateEvaluation(
        chromosome=tuple(chromosome),
        routes=tuple(tuple(route) for route in routes),
        objective=report.total_distance,
        score=score,
        feasible=report.feasible,
        violations=tuple(violations),
    )


def _roulette_select(
    population: Sequence[tuple[str, ...]],
    evaluations: Sequence[CandidateEvaluation],
    rng: random.Random,
) -> tuple[str, ...]:
    weights = [1.0 / max(evaluation.score, 1e-9) for evaluation in evaluations]
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for chromosome, weight in zip(population, weights, strict=True):
        cumulative += weight
        if cumulative >= threshold:
            return chromosome
    return population[-1]


def _pmx_child(
    first_parent: Sequence[str],
    second_parent: Sequence[str],
    left: int,
    right: int,
) -> tuple[str, ...]:
    child: list[str | None] = [None] * len(first_parent)
    child[left : right + 1] = first_parent[left : right + 1]

    for index in range(left, right + 1):
        gene = second_parent[index]
        if gene in child:
            continue
        position = index
        while child[position] is not None:
            mapped_gene = first_parent[position]
            position = second_parent.index(mapped_gene)
        child[position] = gene

    for index, gene in enumerate(second_parent):
        if child[index] is None:
            child[index] = gene
    return tuple(gene for gene in child if gene is not None)


def _pmx_crossover(
    first_parent: Sequence[str],
    second_parent: Sequence[str],
    rng: random.Random,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(first_parent) < 2:
        return tuple(first_parent), tuple(second_parent)
    left, right = sorted(rng.sample(range(len(first_parent)), 2))
    return (
        _pmx_child(first_parent, second_parent, left, right),
        _pmx_child(second_parent, first_parent, left, right),
    )


def _inverse_mutation(chromosome: Sequence[str], rng: random.Random) -> tuple[str, ...]:
    if len(chromosome) < 2:
        return tuple(chromosome)
    left, right = sorted(rng.sample(range(len(chromosome)), 2))
    mutated = list(chromosome)
    mutated[left : right + 1] = reversed(mutated[left : right + 1])
    return tuple(mutated)


def _convergence_record(
    generation: int,
    evaluations: Sequence[CandidateEvaluation],
) -> ConvergenceRecord:
    best = min(evaluations, key=lambda evaluation: evaluation.score)
    return ConvergenceRecord(
        generation=generation,
        best_objective=best.objective,
        average_objective=sum(evaluation.objective for evaluation in evaluations)
        / len(evaluations),
        feasible_rate=sum(1 for evaluation in evaluations if evaluation.feasible)
        / len(evaluations),
        best_feasible=best.feasible,
    )


def solve_ga_vrptw(
    instance: Instance,
    *,
    seed: int = 2014,
    population_size: int = 100,
    generations: int = 200,
    crossover_probability: float = 0.8,
    mutation_probability: float = 0.2,
    time_limit_seconds: float | None = None,
) -> GeneticResult:
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if generations < 0:
        raise ValueError("generations must be non-negative")
    if time_limit_seconds is not None and time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive when provided")

    rng = random.Random(seed)
    customers = tuple(customer.name for customer in instance.customers)
    if not customers:
        raise ValueError("GA baseline requires at least one customer")

    population = [tuple(rng.sample(customers, len(customers))) for _ in range(population_size)]
    convergence: list[ConvergenceRecord] = []
    started_at = time.perf_counter()

    evaluations = [_evaluate(instance, chromosome) for chromosome in population]
    convergence.append(_convergence_record(0, evaluations))

    completed_generations = 0
    terminated_by_time_limit = False
    for generation in range(1, generations + 1):
        if (
            time_limit_seconds is not None
            and time.perf_counter() - started_at >= time_limit_seconds
        ):
            terminated_by_time_limit = True
            break
        best = min(evaluations, key=lambda evaluation: evaluation.score)
        offspring: list[tuple[str, ...]] = [best.chromosome]

        while len(offspring) < population_size:
            first = _roulette_select(population, evaluations, rng)
            second = _roulette_select(population, evaluations, rng)
            if rng.random() < crossover_probability:
                first_child, second_child = _pmx_crossover(first, second, rng)
            else:
                first_child, second_child = tuple(first), tuple(second)
            if rng.random() < mutation_probability:
                first_child = _inverse_mutation(first_child, rng)
            if rng.random() < mutation_probability:
                second_child = _inverse_mutation(second_child, rng)
            offspring.append(first_child)
            if len(offspring) < population_size:
                offspring.append(second_child)

        population = offspring
        evaluations = [_evaluate(instance, chromosome) for chromosome in population]
        convergence.append(_convergence_record(generation, evaluations))
        completed_generations = generation

    best = min(evaluations, key=lambda evaluation: evaluation.score)
    runtime_seconds = time.perf_counter() - started_at
    return GeneticResult(
        method="GA_VRPTW",
        instance_name=instance.name,
        customer_count=len(instance.customers),
        seed=seed,
        population_size=population_size,
        configured_generations=generations,
        generations=completed_generations,
        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,
        objective_value=best.objective,
        score=best.score,
        feasible=best.feasible,
        runtime_seconds=runtime_seconds,
        time_limit_seconds=time_limit_seconds,
        terminated_by_time_limit=terminated_by_time_limit,
        vehicle_count=len(best.routes),
        routes=best.routes,
        violations=best.violations,
        convergence=tuple(convergence),
    )
