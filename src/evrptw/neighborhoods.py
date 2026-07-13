from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Protocol

from evrptw.charging import ChargingSubproblemResult
from evrptw.models import Instance, NodeType
from evrptw.objective import SolutionObjective

CustomerSequence = tuple[str, ...]
RouteSequences = tuple[CustomerSequence, ...]
_EPSILON = 1e-9


class OperatorProfile(StrEnum):
    BASELINE = "baseline"
    STAGE02_ROUTE_REDUCTION = "stage02_route_reduction"


class RouteEvaluator(Protocol):
    calls: int

    def route(self, sequence: CustomerSequence) -> ChargingSubproblemResult:
        """Evaluate one customer sequence with the exact charging subproblem."""


@dataclass(frozen=True, slots=True)
class VehicleOperatorConfig:
    max_route_elimination_attempts: int = 8
    route_elimination_exact_evaluation_budget: int = 256
    route_merge_exact_evaluation_budget: int = 64
    vehicle_repair_exact_evaluation_budget: int = 256

    def __post_init__(self) -> None:
        for name, value in (
            ("max_route_elimination_attempts", self.max_route_elimination_attempts),
            (
                "route_elimination_exact_evaluation_budget",
                self.route_elimination_exact_evaluation_budget,
            ),
            ("route_merge_exact_evaluation_budget", self.route_merge_exact_evaluation_budget),
            ("vehicle_repair_exact_evaluation_budget", self.vehicle_repair_exact_evaluation_budget),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")


_DEFAULT_VEHICLE_OPERATOR_CONFIG = VehicleOperatorConfig()


@dataclass(frozen=True, slots=True)
class NeighborhoodEvent:
    operator: str
    status: str
    reason: str
    route_indices: tuple[int, ...] = ()
    removed_customers: tuple[str, ...] = ()
    candidate_customer_sequence: tuple[str, ...] = ()
    candidate_vehicle_delta: int | None = None
    candidate_feasible: bool = False
    prefilter_passed: bool = False
    new_routes_created: int = 0
    exact_route_evaluations: int = 0
    selection_rank: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MoveProposal:
    operator: str
    sequences: RouteSequences | None
    events: tuple[NeighborhoodEvent, ...]


@dataclass(frozen=True, slots=True)
class RepairResult:
    sequences: RouteSequences | None
    new_routes_created: int
    exact_route_evaluations: int
    failure_reason: str


@dataclass(frozen=True, slots=True)
class RouteScreenResult:
    accepted: bool
    reason: str
    demand: float
    optimistic_finish_time: float
    energy_reachable: bool


@dataclass(frozen=True, slots=True)
class _InsertionOption:
    route_index: int
    position: int
    sequence: CustomerSequence
    objective: SolutionObjective


@dataclass(frozen=True, slots=True)
class _RouteProfile:
    index: int
    sequence: CustomerSequence
    demand: float
    distance: float
    charging_time: float
    charging_count: int


class _EvaluationBudgetExceeded(RuntimeError):
    pass


def screen_route_candidate(instance: Instance, sequence: CustomerSequence) -> RouteScreenResult:
    """Run safe, optimistic checks before exact charging evaluation.

    The time propagation ignores charging and uses direct Euclidean legs, so a rejected
    candidate is necessarily infeasible under the repository's charging model. Energy
    reachability starts every fixed node with a full battery, which is also optimistic.
    """

    by_name = instance.by_name
    unknown = [name for name in sequence if name not in by_name]
    if unknown or any(by_name[name].kind is not NodeType.CUSTOMER for name in sequence):
        return RouteScreenResult(False, "route_structure_prefilter", 0.0, 0.0, False)

    demand = sum(by_name[name].demand for name in sequence)
    if demand > instance.vehicle.load_capacity + _EPSILON:
        return RouteScreenResult(False, "capacity_prefilter", demand, 0.0, False)

    current_time = max(0.0, instance.depot.ready_time)
    chain = (instance.depot.name, *sequence, instance.depot.name)
    for origin_name, destination_name in zip(chain, chain[1:], strict=False):
        origin = by_name[origin_name]
        destination = by_name[destination_name]
        current_time += origin.distance_to(destination) / instance.vehicle.average_velocity
        current_time = max(current_time, destination.ready_time)
        if destination.kind is NodeType.CUSTOMER:
            if current_time > destination.due_date + _EPSILON:
                return RouteScreenResult(
                    False, "time_window_prefilter", demand, current_time, False
                )
            current_time += destination.service_time

    for origin_name, destination_name in zip(chain, chain[1:], strict=False):
        if not _energy_reachable_optimistically(instance, origin_name, destination_name):
            return RouteScreenResult(False, "energy_prefilter", demand, current_time, False)

    return RouteScreenResult(True, "", demand, current_time, True)


def repair_vehicle_count_aware(
    partial: RouteSequences,
    removed: tuple[str, ...],
    evaluator: RouteEvaluator,
    instance: Instance,
    *,
    config: VehicleOperatorConfig,
    allow_new_routes: bool,
) -> RepairResult:
    """Repair customers using existing routes first, then optionally add routes."""

    strict = _repair_pass(
        partial,
        removed,
        evaluator,
        instance,
        allow_new_routes=False,
        budget=config.vehicle_repair_exact_evaluation_budget,
    )
    if (
        strict.sequences is not None
        or not allow_new_routes
        or strict.failure_reason != "no_existing_route_insertion"
    ):
        return strict

    fallback = _repair_pass(
        partial,
        removed,
        evaluator,
        instance,
        allow_new_routes=True,
        budget=config.vehicle_repair_exact_evaluation_budget,
    )
    return RepairResult(
        fallback.sequences,
        fallback.new_routes_created,
        strict.exact_route_evaluations + fallback.exact_route_evaluations,
        fallback.failure_reason,
    )


def propose_route_elimination(
    instance: Instance,
    sequences: RouteSequences,
    evaluator: RouteEvaluator,
    *,
    config: VehicleOperatorConfig = _DEFAULT_VEHICLE_OPERATOR_CONFIG,
) -> MoveProposal:
    """Try to eliminate one complete route and reinsert its customers elsewhere."""

    if len(sequences) <= 1:
        event = NeighborhoodEvent(
            "route_elimination", "not_applicable", "only_one_route", candidate_feasible=False
        )
        return MoveProposal("route_elimination", None, (event,))

    events: list[NeighborhoodEvent] = []
    profiles: list[_RouteProfile] = []
    for index, sequence in enumerate(sequences):
        result = evaluator.route(sequence)
        if not result.feasible:
            events.append(
                NeighborhoodEvent(
                    "route_elimination",
                    "failed",
                    "source_route_infeasible",
                    route_indices=(index,),
                    removed_customers=sequence,
                    exact_route_evaluations=1,
                    selection_rank=index + 1,
                )
            )
            continue
        objective = _route_objective(instance, result)
        profiles.append(
            _RouteProfile(
                index,
                sequence,
                sum(instance.by_name[name].demand for name in sequence),
                objective.total_distance,
                objective.total_charging_time,
                objective.charging_count,
            )
        )

    if len(profiles) != len(sequences):
        return MoveProposal("route_elimination", None, tuple(events))

    ordered = sorted(
        profiles,
        key=lambda profile: (
            len(profile.sequence),
            -profile.distance,
            -profile.charging_time,
            -profile.charging_count,
            profile.index,
        ),
    )
    for rank, profile in enumerate(ordered[: config.max_route_elimination_attempts], start=1):
        partial = tuple(
            sequence for index, sequence in enumerate(sequences) if index != profile.index
        )
        before_calls = evaluator.calls
        elimination_config = replace(
            config,
            vehicle_repair_exact_evaluation_budget=(
                config.route_elimination_exact_evaluation_budget
            ),
        )
        repair = repair_vehicle_count_aware(
            partial,
            profile.sequence,
            evaluator,
            instance,
            config=elimination_config,
            allow_new_routes=False,
        )
        exact_calls = evaluator.calls - before_calls
        if repair.sequences is not None and len(repair.sequences) == len(sequences) - 1:
            events.append(
                NeighborhoodEvent(
                    "route_elimination",
                    "candidate_proposed",
                    "route_eliminated",
                    route_indices=(profile.index,),
                    removed_customers=profile.sequence,
                    candidate_vehicle_delta=-1,
                    candidate_feasible=True,
                    prefilter_passed=True,
                    new_routes_created=repair.new_routes_created,
                    exact_route_evaluations=exact_calls,
                    selection_rank=rank,
                )
            )
            return MoveProposal("route_elimination", repair.sequences, tuple(events))

        events.append(
            NeighborhoodEvent(
                "route_elimination",
                "failed",
                repair.failure_reason or "route_elimination_repair_failed",
                route_indices=(profile.index,),
                removed_customers=profile.sequence,
                candidate_feasible=False,
                prefilter_passed=True,
                new_routes_created=repair.new_routes_created,
                exact_route_evaluations=exact_calls,
                selection_rank=rank,
            )
        )

    if not events:
        events.append(
            NeighborhoodEvent("route_elimination", "failed", "no_candidate_route")
        )
    return MoveProposal("route_elimination", None, tuple(events))


def propose_route_merge(
    instance: Instance,
    sequences: RouteSequences,
    evaluator: RouteEvaluator,
    *,
    config: VehicleOperatorConfig = _DEFAULT_VEHICLE_OPERATOR_CONFIG,
) -> MoveProposal:
    """Try promising route pairs, screening before exact charging evaluation."""

    if len(sequences) <= 1:
        event = NeighborhoodEvent("route_merge", "not_applicable", "only_one_route")
        return MoveProposal("route_merge", None, (event,))

    profiles: list[_RouteProfile] = []
    events: list[NeighborhoodEvent] = []
    for index, sequence in enumerate(sequences):
        result = evaluator.route(sequence)
        if not result.feasible:
            events.append(
                NeighborhoodEvent(
                    "route_merge",
                    "failed",
                    "source_route_infeasible",
                    route_indices=(index,),
                    exact_route_evaluations=1,
                )
            )
            continue
        objective = _route_objective(instance, result)
        profiles.append(
            _RouteProfile(
                index,
                sequence,
                sum(instance.by_name[name].demand for name in sequence),
                objective.total_distance,
                objective.total_charging_time,
                objective.charging_count,
            )
        )
    if len(profiles) != len(sequences):
        return MoveProposal("route_merge", None, tuple(events))

    pairs = sorted(
        (
            (
                len(left.sequence) + len(right.sequence),
                left.demand + right.demand,
                -(left.distance + right.distance),
                -(left.charging_time + right.charging_time),
                left.index,
                right.index,
            ),
            left,
            right,
        )
        for position, left in enumerate(profiles)
        for right in profiles[position + 1 :]
    )

    best: tuple[SolutionObjective, CustomerSequence, int, int] | None = None
    exact_evaluations = 0
    for _, left, right in pairs:
        for source, target in ((left, right), (right, left)):
            for position in range(len(target.sequence) + 1):
                merged = (
                    target.sequence[:position]
                    + source.sequence
                    + target.sequence[position:]
                )
                screen = screen_route_candidate(instance, merged)
                if not screen.accepted:
                    events.append(
                        NeighborhoodEvent(
                            "route_merge",
                            "prefilter_rejected",
                            screen.reason,
                            route_indices=(left.index, right.index),
                            removed_customers=source.sequence,
                            candidate_customer_sequence=merged,
                            prefilter_passed=False,
                        )
                    )
                    continue
                if exact_evaluations >= config.route_merge_exact_evaluation_budget:
                    events.append(
                        NeighborhoodEvent(
                            "route_merge",
                            "budget_exhausted",
                            "route_merge_exact_evaluation_budget",
                            route_indices=(left.index, right.index),
                            candidate_customer_sequence=merged,
                            prefilter_passed=True,
                            exact_route_evaluations=exact_evaluations,
                        )
                    )
                    break
                before_calls = evaluator.calls
                result = evaluator.route(merged)
                exact_evaluations += max(1, evaluator.calls - before_calls)
                if not result.feasible:
                    events.append(
                        NeighborhoodEvent(
                            "route_merge",
                            "exact_infeasible",
                            result.failure_reason or "exact_charging_infeasible",
                            route_indices=(left.index, right.index),
                            candidate_customer_sequence=merged,
                            prefilter_passed=True,
                            exact_route_evaluations=1,
                        )
                    )
                    continue
                objective = _route_objective(instance, result)
                candidate_key = (objective.key, merged, left.index, right.index)
                if best is None or candidate_key < (best[0].key, best[1], best[2], best[3]):
                    best = (objective, merged, left.index, right.index)
                events.append(
                    NeighborhoodEvent(
                        "route_merge",
                        "feasible_candidate",
                        "exact_charging_feasible",
                        route_indices=(left.index, right.index),
                        candidate_customer_sequence=merged,
                        candidate_feasible=True,
                        prefilter_passed=True,
                        exact_route_evaluations=1,
                    )
                )
            if exact_evaluations >= config.route_merge_exact_evaluation_budget:
                break
        if exact_evaluations >= config.route_merge_exact_evaluation_budget:
            break

    if best is None:
        return MoveProposal("route_merge", None, tuple(events))

    _, merged, left_index, right_index = best
    new_sequences = list(sequences)
    new_sequences[left_index] = merged
    del new_sequences[right_index]
    events.append(
        NeighborhoodEvent(
            "route_merge",
            "candidate_proposed",
            "route_merged",
            route_indices=(left_index, right_index),
            candidate_customer_sequence=merged,
            candidate_vehicle_delta=-1,
            candidate_feasible=True,
            prefilter_passed=True,
            exact_route_evaluations=exact_evaluations,
        )
    )
    return MoveProposal("route_merge", tuple(new_sequences), tuple(events))


def _repair_pass(
    partial: RouteSequences,
    removed: tuple[str, ...],
    evaluator: RouteEvaluator,
    instance: Instance,
    *,
    allow_new_routes: bool,
    budget: int,
) -> RepairResult:
    sequences = list(partial)
    pending = list(removed)
    before_calls = evaluator.calls
    new_routes = 0

    try:
        while pending:
            options_by_customer = {
                customer: _insertion_options(
                    sequences,
                    customer,
                    evaluator,
                    instance,
                    before_calls=before_calls,
                    budget=budget,
                )
                for customer in pending
            }
            feasible_customers = [
                customer for customer, options in options_by_customer.items() if options
            ]
            if feasible_customers:
                customer = min(
                    feasible_customers,
                    key=lambda name: _customer_priority(name, options_by_customer[name]),
                )
                option = options_by_customer[customer][0]
                if option.route_index == len(sequences):
                    raise RuntimeError("new-route insertion leaked into existing-only repair")
                sequences[option.route_index] = option.sequence
                pending.remove(customer)
                continue

            if not allow_new_routes:
                return RepairResult(
                    None,
                    new_routes,
                    evaluator.calls - before_calls,
                    "no_existing_route_insertion",
                )

            customer = min(pending)
            if evaluator.calls - before_calls >= budget:
                raise _EvaluationBudgetExceeded
            singleton = (customer,)
            result = evaluator.route(singleton)
            if not result.feasible:
                return RepairResult(
                    None,
                    new_routes,
                    evaluator.calls - before_calls,
                    "new_singleton_route_infeasible",
                )
            sequences.append(singleton)
            pending.remove(customer)
            new_routes += 1
    except _EvaluationBudgetExceeded:
        return RepairResult(
            None,
            new_routes,
            evaluator.calls - before_calls,
            "evaluation_budget_exhausted",
        )

    return RepairResult(tuple(sequences), new_routes, evaluator.calls - before_calls, "")


def _insertion_options(
    sequences: list[CustomerSequence],
    customer: str,
    evaluator: RouteEvaluator,
    instance: Instance,
    *,
    before_calls: int,
    budget: int,
) -> list[_InsertionOption]:
    options: list[_InsertionOption] = []
    for route_index, base in enumerate(sequences):
        base_demand = sum(instance.by_name[name].demand for name in base)
        if (
            base_demand + instance.by_name[customer].demand
            > instance.vehicle.load_capacity + _EPSILON
        ):
            continue
        for position in range(len(base) + 1):
            if evaluator.calls - before_calls >= budget:
                raise _EvaluationBudgetExceeded
            candidate = (*base[:position], customer, *base[position:])
            result = evaluator.route(candidate)
            if not result.feasible:
                continue
            options.append(
                _InsertionOption(
                    route_index,
                    position,
                    candidate,
                    _route_objective(instance, result),
                )
            )
    return sorted(
        options,
        key=lambda option: (
            option.objective.key,
            option.route_index,
            option.position,
            option.sequence,
        ),
    )


def _customer_priority(customer: str, options: list[_InsertionOption]) -> tuple[float, float, str]:
    if len(options) < 2:
        regret = math.inf
    else:
        regret = options[1].objective.total_distance - options[0].objective.total_distance
    return (float(len(options)), -regret, customer)


def _route_objective(instance: Instance, result: ChargingSubproblemResult) -> SolutionObjective:
    if not result.feasible:
        raise ValueError("cannot build a route objective for an infeasible route")
    return SolutionObjective.from_route(
        instance,
        result.route,
        total_distance=result.distance,
        total_charging_time=result.charging_time,
    )


def _energy_reachable_optimistically(
    instance: Instance,
    origin_name: str,
    destination_name: str,
) -> bool:
    if origin_name == destination_name:
        return True
    by_name = instance.by_name
    origin = by_name[origin_name]
    destination = by_name[destination_name]
    capacity = instance.vehicle.battery_capacity
    rate = instance.vehicle.consumption_rate
    frontier = [origin]
    visited_safe_nodes: set[str] = set()
    while frontier:
        current = frontier.pop()
        if current.distance_to(destination) * rate <= capacity + _EPSILON:
            return True
        if current.name != origin.name and current.kind not in (NodeType.DEPOT, NodeType.STATION):
            continue
        for station in instance.stations:
            if station.name == current.name or station.name in visited_safe_nodes:
                continue
            if current.distance_to(station) * rate <= capacity + _EPSILON:
                visited_safe_nodes.add(station.name)
                frontier.append(station)
    return False
