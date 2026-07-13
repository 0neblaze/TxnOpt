from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field, replace

from evrptw.charging import ChargingSubproblemResult, solve_exact_charging
from evrptw.models import Instance, Node
from evrptw.neighborhoods import (
    NeighborhoodEvent,
    OperatorProfile,
    RouteSequences,
    VehicleOperatorConfig,
    propose_ejection_chain,
    propose_relocate,
    propose_route_elimination,
    propose_route_merge,
    propose_route_segment_destroy,
    propose_swap,
    propose_two_opt_star,
    repair_vehicle_count_aware,
)
from evrptw.objective import (
    ObjectiveComparison,
    SolutionObjective,
    accept_annealing_move,
    compare_objectives,
)
from evrptw.validation import validate_routes

_INFEASIBLE_COST = 1e12
_QUALITY_NEIGHBORHOOD_ORDER = (
    "relocate",
    "swap",
    "two_opt_star",
    "route_segment_destroy",
    "ejection_chain",
)
_QUALITY_NEIGHBORHOODS = frozenset(_QUALITY_NEIGHBORHOOD_ORDER)


@dataclass(slots=True)
class OperatorStatistics:
    calls: int = 0
    feasible_repairs: int = 0
    accepted: int = 0
    improved: int = 0
    best: int = 0
    vehicle_reductions: int = 0
    distance_improvements: int = 0
    rejected: int = 0
    prefilter_passed: int = 0
    prefilter_rejected: int = 0
    new_routes_created: int = 0
    exact_route_evaluations: int = 0
    candidate_proposals: int = 0
    feasible_candidates: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    weight: float = 1.0

    def to_dict(self) -> dict[str, object]:
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
    destroy_statistics: dict[str, dict[str, object]]
    repair_statistics: dict[str, dict[str, object]]
    operator_profile: str
    neighborhood_statistics: dict[str, dict[str, object]]
    neighborhood_events: tuple[dict[str, object], ...]
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
        objective = sum(
            (
                SolutionObjective.from_route(
                    self.instance,
                    result.route,
                    total_distance=result.distance,
                    total_charging_time=result.charging_time,
                )
                for result in charging
            ),
            start=SolutionObjective.zero(),
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
    operator_profile: OperatorProfile | str = OperatorProfile.STAGE02_ROUTE_QUALITY,
    vehicle_operator_config: VehicleOperatorConfig | None = None,
) -> ALNSResult:
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    if not 0.0 < removal_fraction <= 1.0:
        raise ValueError("removal_fraction must be in (0, 1]")
    profile = OperatorProfile(operator_profile)
    vehicle_config = vehicle_operator_config or VehicleOperatorConfig()

    started = time.perf_counter()
    rng = random.Random(seed)
    evaluator = _Evaluator(instance, deadline=started + time_limit_seconds)
    try:
        initial_sequences = _construct_initial_solution(instance, evaluator)
    except _TimeLimitReached:
        return _failed_result(
            started,
            evaluator,
            "time limit reached during initial construction",
            operator_profile=profile.value,
        )
    current = evaluator.solution(initial_sequences)
    if not current.feasible:
        return _failed_result(
            started,
            evaluator,
            "no feasible singleton initial solution",
            operator_profile=profile.value,
        )
    if current.objective is None:
        raise RuntimeError("feasible ALNS initial solution is missing its objective")

    best = current
    quality_probe_current = current
    quality_probe_best = current
    first_feasible_time = time.perf_counter() - started
    best_time = first_feasible_time
    destroy_stats = {name: OperatorStatistics() for name in ("random", "worst", "related")}
    standard_repair_stats = {
        name: OperatorStatistics() for name in ("greedy", "regret2", "energy")
    }
    repair_stats = dict(standard_repair_stats)
    if profile in (
        OperatorProfile.STAGE02_ROUTE_REDUCTION,
        OperatorProfile.STAGE02_ROUTE_QUALITY,
    ):
        repair_stats["vehicle_count_aware"] = OperatorStatistics()
    neighborhood_names = _neighborhood_names(profile)
    neighborhood_stats = {
        name: OperatorStatistics() for name in neighborhood_names
    } if profile is not OperatorProfile.BASELINE else {}
    neighborhood_events: list[dict[str, object]] = []
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
        destroy_name = ""
        repair_name = ""
        selected_neighborhood = ""
        move_events: tuple[NeighborhoodEvent, ...] = ()
        shadow_neighborhood = ""
        shadow_events: tuple[NeighborhoodEvent, ...] = ()
        shadow_candidate: _EvaluatedSolution | None = None
        if profile is OperatorProfile.BASELINE:
            destroy_name = _weighted_choice(rng, destroy_stats)
            repair_name = _weighted_choice(rng, standard_repair_stats)
            destroy_stats[destroy_name].calls += 1
            repair_stats[repair_name].calls += 1

            remove_count = max(1, math.ceil(len(instance.customers) * removal_fraction))
            if len(instance.customers) > 20:
                remove_count = min(remove_count, 3)
            partial, removed = _destroy(
                instance, current.sequences, remove_count, destroy_name, rng
            )
            try:
                candidate_sequences = _repair(
                    partial, removed, repair_name, evaluator, instance, rng
                )
                candidate = evaluator.solution(candidate_sequences)
            except _TimeLimitReached:
                break
        else:
            selected_neighborhood = _select_stage02_neighborhood(
                iteration,
                rng,
                neighborhood_stats,
                include_quality=profile is not OperatorProfile.STAGE02_ROUTE_QUALITY,
            )
            neighborhood_stats[selected_neighborhood].calls += 1
            try:
                if selected_neighborhood == "route_elimination":
                    proposal = propose_route_elimination(
                        instance,
                        current.sequences,
                        evaluator,
                        config=vehicle_config,
                    )
                    candidate_sequences = proposal.sequences or ()
                    move_events = proposal.events
                elif selected_neighborhood == "route_merge":
                    proposal = propose_route_merge(
                        instance,
                        current.sequences,
                        evaluator,
                        config=vehicle_config,
                    )
                    candidate_sequences = proposal.sequences or ()
                    move_events = proposal.events
                elif selected_neighborhood == "relocate":
                    proposal = propose_relocate(
                        instance,
                        current.sequences,
                        evaluator,
                        config=vehicle_config,
                    )
                    candidate_sequences = proposal.sequences or ()
                    move_events = proposal.events
                elif selected_neighborhood == "swap":
                    proposal = propose_swap(
                        instance,
                        current.sequences,
                        evaluator,
                        config=vehicle_config,
                    )
                    candidate_sequences = proposal.sequences or ()
                    move_events = proposal.events
                elif selected_neighborhood == "two_opt_star":
                    proposal = propose_two_opt_star(
                        instance,
                        current.sequences,
                        evaluator,
                        config=vehicle_config,
                    )
                    candidate_sequences = proposal.sequences or ()
                    move_events = proposal.events
                elif selected_neighborhood == "route_segment_destroy":
                    proposal = propose_route_segment_destroy(
                        instance,
                        current.sequences,
                        evaluator,
                        config=vehicle_config,
                    )
                    candidate_sequences = proposal.sequences or ()
                    move_events = proposal.events
                elif selected_neighborhood == "ejection_chain":
                    proposal = propose_ejection_chain(
                        instance,
                        current.sequences,
                        evaluator,
                        config=vehicle_config,
                    )
                    candidate_sequences = proposal.sequences or ()
                    move_events = proposal.events
                else:
                    destroy_name = _weighted_choice(rng, destroy_stats)
                    destroy_stats[destroy_name].calls += 1
                    remove_count = max(1, math.ceil(len(instance.customers) * removal_fraction))
                    if len(instance.customers) > 20:
                        remove_count = min(remove_count, 3)
                    partial, removed = _destroy(
                        instance, current.sequences, remove_count, destroy_name, rng
                    )
                    if selected_neighborhood == "vehicle_count_aware_repair":
                        repair_name = "vehicle_count_aware"
                        repair_stats[repair_name].calls += 1
                        before_calls = evaluator.calls
                        repair = repair_vehicle_count_aware(
                            partial,
                            removed,
                            evaluator,
                            instance,
                            config=vehicle_config,
                            allow_new_routes=True,
                        )
                        candidate_sequences = repair.sequences or ()
                        move_events = (
                            NeighborhoodEvent(
                                "vehicle_count_aware_repair",
                                "candidate_proposed"
                                if repair.sequences is not None
                                else "failed",
                                repair.failure_reason or "existing_route_repair",
                                removed_customers=removed,
                                candidate_vehicle_delta=(
                                    len(candidate_sequences) - len(current.sequences)
                                    if repair.sequences is not None
                                    else None
                                ),
                                candidate_feasible=repair.sequences is not None,
                                new_routes_created=repair.new_routes_created,
                                exact_route_evaluations=evaluator.calls - before_calls,
                            ),
                        )
                    else:
                        repair_name = _weighted_choice(rng, standard_repair_stats)
                        repair_stats[repair_name].calls += 1
                        candidate_sequences = _repair(
                            partial, removed, repair_name, evaluator, instance, rng
                        )
                        move_events = (
                            NeighborhoodEvent(
                                "standard",
                                "proposal",
                                f"{destroy_name}+{repair_name}",
                                removed_customers=removed,
                            ),
                        )
                candidate = evaluator.solution(candidate_sequences)
            except _TimeLimitReached:
                timeout_event = _event_record(
                    NeighborhoodEvent(
                        selected_neighborhood,
                        "time_limit",
                        "time_limit_reached_during_neighborhood",
                    ),
                    iteration,
                )
                timeout_event.update(
                    {
                        "accepted": False,
                        "vehicle_reduction": False,
                        "distance_improvement": False,
                        "candidate_objective_key": (),
                    }
                )
                neighborhood_events.append(timeout_event)
                break

            _record_neighborhood_proposal(
                neighborhood_stats[selected_neighborhood],
                move_events,
                candidate,
                current,
            )
            if profile is OperatorProfile.STAGE02_ROUTE_QUALITY:
                shadow_neighborhood = _quality_shadow_neighborhood(iteration)
                if shadow_neighborhood:
                    neighborhood_stats[shadow_neighborhood].calls += 1
                    try:
                        shadow_sequences, shadow_events = _quality_shadow_proposal(
                            shadow_neighborhood,
                            instance,
                            quality_probe_current.sequences,
                            evaluator,
                            vehicle_config,
                        )
                        shadow_candidate = evaluator.solution(shadow_sequences)
                    except _TimeLimitReached:
                        neighborhood_events.append(
                            _event_record(
                                NeighborhoodEvent(
                                    shadow_neighborhood,
                                    "time_limit",
                                    "time_limit_reached_during_quality_probe",
                                ),
                                iteration,
                            )
                        )
                        break
                    _record_neighborhood_proposal(
                        neighborhood_stats[shadow_neighborhood],
                        shadow_events,
                        shadow_candidate,
                        quality_probe_current,
                    )
                    quality_comparison = (
                        compare_objectives(
                            shadow_candidate.objective,
                            quality_probe_current.objective,
                        )
                        if shadow_candidate.feasible
                        and shadow_candidate.objective is not None
                        and quality_probe_current.objective is not None
                        else ObjectiveComparison.WORSE
                    )
                    quality_probe_accept = (
                        shadow_candidate.feasible
                        and quality_comparison is not ObjectiveComparison.WORSE
                    )
                    quality_probe_vehicle_reduction = bool(
                        shadow_candidate.objective is not None
                        and quality_probe_current.objective is not None
                        and shadow_candidate.objective.vehicle_count
                        < quality_probe_current.objective.vehicle_count
                    )
                    quality_probe_distance_improvement = bool(
                        shadow_candidate.objective is not None
                        and quality_probe_current.objective is not None
                        and shadow_candidate.objective.total_distance
                        < quality_probe_current.objective.total_distance - 1e-9
                    )
                    neighborhood_events.extend(
                        _annotated_event_record(
                            event,
                            iteration=iteration,
                            accepted=quality_probe_accept,
                            vehicle_reduction=quality_probe_vehicle_reduction,
                            distance_improvement=quality_probe_distance_improvement,
                            candidate=shadow_candidate,
                        )
                        for event in shadow_events
                    )
                    if quality_probe_accept and shadow_candidate.objective is not None:
                        shadow_statistics = neighborhood_stats[shadow_neighborhood]
                        shadow_statistics.accepted += 1
                        reward = 1.0
                        if quality_comparison is ObjectiveComparison.BETTER:
                            shadow_statistics.improved += 1
                            reward = 4.0
                        quality_probe_current = shadow_candidate
                        if (
                            quality_probe_best.objective is None
                            or compare_objectives(
                                shadow_candidate.objective,
                                quality_probe_best.objective,
                            )
                            is ObjectiveComparison.BETTER
                        ):
                            quality_probe_best = shadow_candidate
                            shadow_statistics.best += 1
                            reward = 8.0
                            if (
                                best.objective is None
                                or compare_objectives(
                                    shadow_candidate.objective,
                                    best.objective,
                                )
                                is ObjectiveComparison.BETTER
                            ):
                                best = shadow_candidate
                                best_time = time.perf_counter() - started
                        _update_weight(shadow_statistics, reward)

        temperature = initial_temperature * max(0.001, 1.0 - iteration / max_iterations)
        quality_candidate_is_worse = (
            profile is OperatorProfile.STAGE02_ROUTE_QUALITY
            and selected_neighborhood in _QUALITY_NEIGHBORHOODS
            and candidate.objective is not None
            and compare_objectives(candidate.objective, current.objective)
            is ObjectiveComparison.WORSE
        )
        accept = False if quality_candidate_is_worse else (
            candidate.feasible
            and candidate.objective is not None
            and accept_annealing_move(
                current.objective,
                candidate.objective,
                temperature=max(temperature, 1e-12),
                random_draw=rng.random(),
            )
        )
        if profile is not OperatorProfile.BASELINE:
            vehicle_reduction = bool(
                candidate.objective is not None
                and current.objective is not None
                and candidate.objective.vehicle_count < current.objective.vehicle_count
            )
            distance_improvement = bool(
                candidate.objective is not None
                and current.objective is not None
                and candidate.objective.total_distance
                < current.objective.total_distance - 1e-9
            )
            neighborhood_events.extend(
                _annotated_event_record(
                    event,
                    iteration=iteration,
                    accepted=accept,
                    vehicle_reduction=vehicle_reduction,
                    distance_improvement=distance_improvement,
                    candidate=candidate,
                )
                for event in move_events
            )
        if not accept:
            rejected += 1
            if profile is OperatorProfile.BASELINE:
                _update_weight(destroy_stats[destroy_name], 0.0)
                _update_weight(repair_stats[repair_name], 0.0)
            else:
                neighborhood_stats[selected_neighborhood].rejected += 1
                _update_weight(neighborhood_stats[selected_neighborhood], 0.0)
                if destroy_name:
                    _update_weight(destroy_stats[destroy_name], 0.0)
                if repair_name:
                    _update_weight(repair_stats[repair_name], 0.0)
            continue

        accepted += 1
        if profile is OperatorProfile.BASELINE:
            destroy_stats[destroy_name].accepted += 1
            repair_stats[repair_name].accepted += 1
        else:
            neighborhood_stats[selected_neighborhood].accepted += 1
            _update_weight(neighborhood_stats[selected_neighborhood], 1.0)
            if destroy_name:
                destroy_stats[destroy_name].accepted += 1
            if repair_name:
                repair_stats[repair_name].accepted += 1
        reward = 1.0
        if candidate.objective is None:
            raise RuntimeError("accepted ALNS candidate is missing its objective")
        if compare_objectives(candidate.objective, current.objective) is ObjectiveComparison.BETTER:
            improved += 1
            reward = 4.0
            if profile is OperatorProfile.BASELINE:
                destroy_stats[destroy_name].improved += 1
                repair_stats[repair_name].improved += 1
            else:
                neighborhood_stats[selected_neighborhood].improved += 1
                if destroy_name:
                    destroy_stats[destroy_name].improved += 1
                if repair_name:
                    repair_stats[repair_name].improved += 1
        current = candidate
        if best.objective is None:
            raise RuntimeError("feasible ALNS incumbent is missing its objective")
        if compare_objectives(candidate.objective, best.objective) is ObjectiveComparison.BETTER:
            best = candidate
            best_time = time.perf_counter() - started
            reward = 8.0
            if profile is OperatorProfile.BASELINE:
                destroy_stats[destroy_name].best += 1
                repair_stats[repair_name].best += 1
            else:
                neighborhood_stats[selected_neighborhood].best += 1
                if destroy_name:
                    destroy_stats[destroy_name].best += 1
                if repair_name:
                    repair_stats[repair_name].best += 1
        if profile is OperatorProfile.BASELINE:
            _update_weight(destroy_stats[destroy_name], reward)
            _update_weight(repair_stats[repair_name], reward)
        else:
            _update_weight(neighborhood_stats[selected_neighborhood], reward)
            if destroy_name:
                _update_weight(destroy_stats[destroy_name], reward)
            if repair_name:
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
        operator_profile=profile.value,
        neighborhood_statistics={
            name: stats.to_dict() for name, stats in neighborhood_stats.items()
        },
        neighborhood_events=tuple(neighborhood_events),
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
            customer_sequence = tuple(name for name in route if name != instance.depot.name)
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
                    before.distance_to(node) + node.distance_to(after) - before.distance_to(after)
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
        tuple(name for name in sequence if name not in removed_set) for sequence in sequences
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


def _select_stage02_neighborhood(
    iteration: int,
    rng: random.Random,
    statistics: dict[str, OperatorStatistics],
    *,
    include_quality: bool = True,
) -> str:
    warmup: tuple[str, ...]
    if include_quality and "relocate" in statistics:
        warmup = (
            "route_elimination",
            "vehicle_count_aware_repair",
            "route_merge",
            "relocate",
            "swap",
            "two_opt_star",
            "route_segment_destroy",
            "ejection_chain",
        )
    else:
        warmup = ("route_elimination", "vehicle_count_aware_repair", "route_merge")
    if iteration < len(warmup):
        return warmup[iteration]
    if include_quality:
        return _weighted_choice(rng, statistics)
    legacy_names = (
        "standard",
        "vehicle_count_aware_repair",
        "route_elimination",
        "route_merge",
    )
    return _weighted_choice(rng, {name: statistics[name] for name in legacy_names})


def _neighborhood_names(profile: OperatorProfile) -> tuple[str, ...]:
    if profile is OperatorProfile.STAGE02_ROUTE_REDUCTION:
        return (
            "standard",
            "vehicle_count_aware_repair",
            "route_elimination",
            "route_merge",
        )
    if profile is OperatorProfile.STAGE02_ROUTE_QUALITY:
        return (
            "standard",
            "vehicle_count_aware_repair",
            "route_elimination",
            "route_merge",
            "relocate",
            "swap",
            "two_opt_star",
            "route_segment_destroy",
            "ejection_chain",
        )
    return ()


def _quality_shadow_neighborhood(iteration: int) -> str:
    return (
        _QUALITY_NEIGHBORHOOD_ORDER[iteration]
        if iteration < len(_QUALITY_NEIGHBORHOOD_ORDER)
        else ""
    )


def _quality_shadow_proposal(
    operator: str,
    instance: Instance,
    sequences: RouteSequences,
    evaluator: _Evaluator,
    config: VehicleOperatorConfig,
) -> tuple[RouteSequences, tuple[NeighborhoodEvent, ...]]:
    probe_budget = config.quality_probe_exact_evaluation_budget
    probe_config = replace(
        config,
        relocate_exact_evaluation_budget=min(
            config.relocate_exact_evaluation_budget, probe_budget
        ),
        swap_exact_evaluation_budget=min(config.swap_exact_evaluation_budget, probe_budget),
        two_opt_star_exact_evaluation_budget=min(
            config.two_opt_star_exact_evaluation_budget, probe_budget
        ),
        route_segment_exact_evaluation_budget=min(
            config.route_segment_exact_evaluation_budget, probe_budget
        ),
        ejection_chain_exact_evaluation_budget=min(
            config.ejection_chain_exact_evaluation_budget, probe_budget
        ),
    )
    if operator == "relocate":
        proposal = propose_relocate(instance, sequences, evaluator, config=probe_config)
    elif operator == "swap":
        proposal = propose_swap(instance, sequences, evaluator, config=probe_config)
    elif operator == "two_opt_star":
        proposal = propose_two_opt_star(instance, sequences, evaluator, config=probe_config)
    elif operator == "route_segment_destroy":
        proposal = propose_route_segment_destroy(
            instance, sequences, evaluator, config=probe_config
        )
    elif operator == "ejection_chain":
        proposal = propose_ejection_chain(instance, sequences, evaluator, config=probe_config)
    else:
        raise ValueError(f"unsupported Stage 2.2 shadow operator: {operator}")
    return proposal.sequences or (), proposal.events


def _record_neighborhood_proposal(
    statistics: OperatorStatistics,
    events: tuple[NeighborhoodEvent, ...],
    candidate: _EvaluatedSolution,
    current: _EvaluatedSolution,
) -> None:
    statistics.prefilter_passed += sum(event.prefilter_passed for event in events)
    statistics.prefilter_rejected += sum(
        event.status == "prefilter_rejected" for event in events
    )
    statistics.new_routes_created += sum(event.new_routes_created for event in events)
    statistics.exact_route_evaluations += sum(
        event.exact_route_evaluations for event in events
    )
    statistics.candidate_proposals += sum(
        event.status == "candidate_proposed" for event in events
    )
    statistics.feasible_candidates += sum(event.candidate_feasible for event in events)
    failure_statuses = {
        "failed",
        "prefilter_rejected",
        "exact_infeasible",
        "budget_exhausted",
        "not_applicable",
        "time_limit",
    }
    for event in events:
        if event.status in failure_statuses:
            statistics.failure_reasons[event.reason] = (
                statistics.failure_reasons.get(event.reason, 0) + 1
            )
    if candidate.feasible:
        statistics.feasible_repairs += 1
    elif not events:
        statistics.failure_reasons["candidate_infeasible"] = (
            statistics.failure_reasons.get("candidate_infeasible", 0) + 1
        )
    if candidate.objective is None or current.objective is None:
        return
    if candidate.objective.vehicle_count < current.objective.vehicle_count:
        statistics.vehicle_reductions += 1
    if candidate.objective.total_distance < current.objective.total_distance - 1e-9:
        statistics.distance_improvements += 1


def _event_record(event: NeighborhoodEvent, iteration: int) -> dict[str, object]:
    record = event.to_dict()
    record["iteration"] = iteration
    return record


def _annotated_event_record(
    event: NeighborhoodEvent,
    *,
    iteration: int,
    accepted: bool,
    vehicle_reduction: bool,
    distance_improvement: bool,
    candidate: _EvaluatedSolution,
) -> dict[str, object]:
    record = _event_record(event, iteration)
    record.update(
        {
            "accepted": accepted,
            "vehicle_reduction": vehicle_reduction,
            "distance_improvement": distance_improvement,
            "candidate_objective_key": (
                candidate.objective.key if candidate.objective is not None else ()
            ),
        }
    )
    return record


def _failed_result(
    started: float,
    evaluator: _Evaluator,
    reason: str,
    *,
    operator_profile: str = OperatorProfile.BASELINE.value,
) -> ALNSResult:
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
        operator_profile=operator_profile,
        neighborhood_statistics={},
        neighborhood_events=(),
        failure_reason=reason,
    )
