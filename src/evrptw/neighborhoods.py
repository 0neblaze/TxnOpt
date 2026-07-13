from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
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
    STAGE02_ROUTE_QUALITY = "stage02_route_quality"
    STAGE02_CONSTRAINT_GUIDED = "stage02_constraint_guided"


class ConstraintRemovalOperator(StrEnum):
    STATION_PRESSURE = "station_pressure"
    TIME_WINDOW_CONFLICT = "time_window_conflict"
    WORST_ENERGY_DETOUR = "worst_energy_detour"
    SHAW_RELATED = "shaw_related"


class RemovalTier(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class RouteEvaluator(Protocol):
    calls: int

    def route(self, sequence: CustomerSequence) -> ChargingSubproblemResult:
        """Evaluate one customer sequence with the exact charging subproblem."""


class RouteEvaluationDeadlineExceeded(RuntimeError):
    """Signal a cooperative exact-route deadline without losing operator events."""

    def __init__(
        self,
        sequence: CustomerSequence | None = None,
        *,
        exact_route_evaluations: int = 0,
    ) -> None:
        super().__init__("route evaluation deadline exceeded")
        self.sequence = sequence
        self.exact_route_evaluations = exact_route_evaluations


@dataclass(frozen=True, slots=True)
class VehicleOperatorConfig:
    max_route_elimination_attempts: int = 8
    route_elimination_exact_evaluation_budget: int = 256
    route_merge_exact_evaluation_budget: int = 64
    vehicle_repair_exact_evaluation_budget: int = 256
    vehicle_reduction_refinement_exact_evaluation_budget: int = 512
    relocate_exact_evaluation_budget: int = 48
    swap_exact_evaluation_budget: int = 48
    two_opt_star_exact_evaluation_budget: int = 32
    route_segment_exact_evaluation_budget: int = 32
    ejection_chain_exact_evaluation_budget: int = 24
    quality_probe_exact_evaluation_budget: int = 4
    quality_route_segment_probe_exact_evaluation_budget: int = 4
    constraint_lane_time_budget_seconds: float = 2.0
    route_segment_min_length: int = 2
    route_segment_max_length: int = 5
    ejection_chain_max_depth: int = 3
    ejection_chain_beam_width: int = 16
    constraint_probe_exact_evaluation_budget: int = 16
    station_pressure_exact_evaluation_budget: int = 48
    time_window_conflict_exact_evaluation_budget: int = 48
    worst_energy_detour_exact_evaluation_budget: int = 48
    shaw_related_exact_evaluation_budget: int = 48
    small_removal_min_fraction: float = 0.05
    small_removal_max_fraction: float = 0.10
    medium_removal_min_fraction: float = 0.10
    medium_removal_max_fraction: float = 0.20
    large_removal_min_fraction: float = 0.20
    large_removal_max_fraction: float = 0.35
    medium_stagnation_threshold: int = 4
    large_stagnation_threshold: int = 8
    exploration_period: int = 3

    def __post_init__(self) -> None:
        for name, value in (
            ("max_route_elimination_attempts", self.max_route_elimination_attempts),
            (
                "route_elimination_exact_evaluation_budget",
                self.route_elimination_exact_evaluation_budget,
            ),
            ("route_merge_exact_evaluation_budget", self.route_merge_exact_evaluation_budget),
            ("vehicle_repair_exact_evaluation_budget", self.vehicle_repair_exact_evaluation_budget),
            (
                "vehicle_reduction_refinement_exact_evaluation_budget",
                self.vehicle_reduction_refinement_exact_evaluation_budget,
            ),
            ("relocate_exact_evaluation_budget", self.relocate_exact_evaluation_budget),
            ("swap_exact_evaluation_budget", self.swap_exact_evaluation_budget),
            (
                "two_opt_star_exact_evaluation_budget",
                self.two_opt_star_exact_evaluation_budget,
            ),
            ("route_segment_exact_evaluation_budget", self.route_segment_exact_evaluation_budget),
            (
                "ejection_chain_exact_evaluation_budget",
                self.ejection_chain_exact_evaluation_budget,
            ),
            ("quality_probe_exact_evaluation_budget", self.quality_probe_exact_evaluation_budget),
            (
                "quality_route_segment_probe_exact_evaluation_budget",
                self.quality_route_segment_probe_exact_evaluation_budget,
            ),
            ("constraint_lane_time_budget_seconds", self.constraint_lane_time_budget_seconds),
            ("route_segment_min_length", self.route_segment_min_length),
            ("route_segment_max_length", self.route_segment_max_length),
            ("ejection_chain_max_depth", self.ejection_chain_max_depth),
            ("ejection_chain_beam_width", self.ejection_chain_beam_width),
            (
                "constraint_probe_exact_evaluation_budget",
                self.constraint_probe_exact_evaluation_budget,
            ),
            (
                "station_pressure_exact_evaluation_budget",
                self.station_pressure_exact_evaluation_budget,
            ),
            (
                "time_window_conflict_exact_evaluation_budget",
                self.time_window_conflict_exact_evaluation_budget,
            ),
            (
                "worst_energy_detour_exact_evaluation_budget",
                self.worst_energy_detour_exact_evaluation_budget,
            ),
            ("shaw_related_exact_evaluation_budget", self.shaw_related_exact_evaluation_budget),
            ("medium_stagnation_threshold", self.medium_stagnation_threshold),
            ("large_stagnation_threshold", self.large_stagnation_threshold),
            ("exploration_period", self.exploration_period),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.route_segment_min_length > self.route_segment_max_length:
            raise ValueError("route_segment_min_length must not exceed route_segment_max_length")
        for name, lower, upper in (
            (
                "small_removal",
                self.small_removal_min_fraction,
                self.small_removal_max_fraction,
            ),
            (
                "medium_removal",
                self.medium_removal_min_fraction,
                self.medium_removal_max_fraction,
            ),
            (
                "large_removal",
                self.large_removal_min_fraction,
                self.large_removal_max_fraction,
            ),
        ):
            if not 0.0 < lower <= upper <= 1.0:
                raise ValueError(f"{name} fractions must satisfy 0 < min <= max <= 1")
        if self.medium_stagnation_threshold > self.large_stagnation_threshold:
            raise ValueError(
                "medium_stagnation_threshold must not exceed large_stagnation_threshold"
            )


_DEFAULT_VEHICLE_OPERATOR_CONFIG = VehicleOperatorConfig()


@dataclass(frozen=True, slots=True)
class NeighborhoodEvent:
    operator: str
    status: str
    reason: str
    route_indices: tuple[int, ...] = ()
    affected_route_indices: tuple[int, ...] = ()
    removed_customers: tuple[str, ...] = ()
    candidate_customer_sequence: tuple[str, ...] = ()
    candidate_route_sequences: tuple[CustomerSequence, ...] = ()
    candidate_vehicle_delta: int | None = None
    candidate_feasible: bool = False
    prefilter_passed: bool = False
    new_routes_created: int = 0
    exact_route_evaluations: int = 0
    selection_rank: int = 0
    chain_depth: int = 0
    segment_length: int = 0
    track: str = "legacy"
    constraint_category: str = ""
    removal_tier: str = ""
    removal_size_requested: int = 0
    removal_size_actual: int = 0
    stagnation_iterations: int = 0
    removal_trigger: str = ""
    reset_observed: bool = False
    ranking_score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MoveProposal:
    operator: str
    sequences: RouteSequences | None
    events: tuple[NeighborhoodEvent, ...]


@dataclass(frozen=True, slots=True)
class RemovalSizeSelection:
    tier: RemovalTier
    requested_count: int
    lower_bound: int
    upper_bound: int
    stagnation_iterations: int
    trigger_reason: str
    reset_observed: bool = False


@dataclass(frozen=True, slots=True)
class RemovalProposal:
    operator: str
    partial: RouteSequences | None
    removed_customers: tuple[str, ...]
    selection: RemovalSizeSelection
    scores: tuple[tuple[str, float], ...]
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


def select_dynamic_removal_size(
    customer_count: int,
    stagnation_iterations: int,
    iteration: int,
    *,
    config: VehicleOperatorConfig = _DEFAULT_VEHICLE_OPERATOR_CONFIG,
    global_best_reset: bool = False,
) -> RemovalSizeSelection:
    """Select a deterministic removal tier for the Stage 2.3 lane.

    The tier is driven by stagnation first.  Periodic exploration may promote the
    selected tier by one level only after medium stagnation has been reached; it
    never bypasses the stagnation thresholds, resets stagnation, or reduces a tier.
    Counts use the configured lower bound so the selection is reproducible and the
    actual removal remains inside the configured interval after clamping.
    """

    if customer_count < 0:
        raise ValueError("customer_count must be non-negative")
    if stagnation_iterations < 0:
        raise ValueError("stagnation_iterations must be non-negative")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")

    if stagnation_iterations >= config.large_stagnation_threshold:
        tier = RemovalTier.LARGE
        trigger = "large_stagnation"
    elif stagnation_iterations >= config.medium_stagnation_threshold:
        tier = RemovalTier.MEDIUM
        trigger = "medium_stagnation"
    else:
        tier = RemovalTier.SMALL
        trigger = "stagnation_baseline"

    if (
        iteration > 0
        and iteration % config.exploration_period == 0
        and stagnation_iterations > config.medium_stagnation_threshold
    ):
        promoted = _next_removal_tier(tier)
        if _removal_tier_rank(promoted) > _removal_tier_rank(tier):
            tier = promoted
            trigger = f"{trigger}+periodic_exploration"

    if customer_count <= 1:
        return RemovalSizeSelection(
            tier,
            0,
            0,
            0,
            stagnation_iterations,
            "no_removable_customer",
            global_best_reset,
        )

    minimum, maximum = _removal_fraction_bounds(tier, config)
    upper_customer_bound = customer_count - 1
    lower_bound = max(1, min(upper_customer_bound, math.ceil(customer_count * minimum)))
    upper_bound = max(
        lower_bound,
        min(upper_customer_bound, math.floor(customer_count * maximum)),
    )
    requested = lower_bound
    return RemovalSizeSelection(
        tier,
        requested,
        lower_bound,
        upper_bound,
        stagnation_iterations,
        trigger,
        global_best_reset,
    )


def propose_constraint_removal(
    instance: Instance,
    sequences: RouteSequences,
    evaluator: RouteEvaluator,
    *,
    operator: ConstraintRemovalOperator | str,
    selection: RemovalSizeSelection,
    seed: int = 0,
    rng: random.Random | None = None,
    precomputed_routes: Mapping[CustomerSequence, ChargingSubproblemResult] | None = None,
) -> RemovalProposal:
    """Rank and remove customers using one explicit constraint signal.

    This public seam only performs the removal.  Repair remains a separate call so
    the ALNS layer can record the exact repair outcome and vehicle delta.  Every
    source route is evaluated once unless the caller supplies its already available
    exact route results; ties are resolved by route index and customer name.
    """

    selected_operator = ConstraintRemovalOperator(operator)
    operator_name = selected_operator.value
    random_source = rng if rng is not None else random.Random(seed)
    all_customers = [name for sequence in sequences for name in sequence]
    if len(set(all_customers)) != len(all_customers):
        raise ValueError("constraint removal requires unique customer coverage")
    if any(name not in instance.by_name or instance.by_name[name].kind is not NodeType.CUSTOMER
           for name in all_customers):
        raise ValueError("constraint removal requires customer-only route sequences")
    if len(all_customers) <= 1 or selection.requested_count <= 0:
        event = NeighborhoodEvent(
            operator_name,
            "not_applicable",
            "no_removable_customer",
            track="constraint_lane",
            constraint_category=operator_name,
            removal_tier=selection.tier.value,
            removal_size_requested=selection.requested_count,
            removal_size_actual=0,
            stagnation_iterations=selection.stagnation_iterations,
            removal_trigger=selection.trigger_reason,
            reset_observed=selection.reset_observed,
        )
        return RemovalProposal(
            operator_name,
            None,
            (),
            selection,
            (),
            (event,),
        )

    events: list[NeighborhoodEvent] = []
    scores: list[tuple[str, float, int]] = []
    before_calls = evaluator.calls
    anchor: str | None = None
    if selected_operator is ConstraintRemovalOperator.SHAW_RELATED:
        anchor = random_source.choice(sorted(all_customers))

    try:
        for route_index, sequence in enumerate(sequences):
            used_precomputed = (
                precomputed_routes is not None and sequence in precomputed_routes
            )
            result = (
                precomputed_routes[sequence]
                if precomputed_routes is not None and used_precomputed
                else evaluator.route(sequence)
            )
            if not result.feasible:
                events.append(
                    NeighborhoodEvent(
                        operator_name,
                        "exact_infeasible",
                        result.failure_reason or "source_route_infeasible",
                        route_indices=(route_index,),
                        affected_route_indices=(route_index,),
                        candidate_route_sequences=(sequence,),
                        prefilter_passed=True,
                        exact_route_evaluations=0 if used_precomputed else 1,
                        track="constraint_lane",
                        constraint_category=operator_name,
                        removal_tier=selection.tier.value,
                        removal_size_requested=selection.requested_count,
                        stagnation_iterations=selection.stagnation_iterations,
                        removal_trigger=selection.trigger_reason,
                        reset_observed=selection.reset_observed,
                    )
                )
                continue
            route_scores = _constraint_route_scores(
                instance,
                sequence,
                result,
                selected_operator,
                anchor=anchor,
            )
            scores.extend(
                (name, score, route_index) for name, score in route_scores.items()
            )
    except RouteEvaluationDeadlineExceeded:
        events.append(
            NeighborhoodEvent(
                operator_name,
                "time_limit",
                "time_limit_reached_during_constraint_ranking",
                prefilter_passed=True,
                exact_route_evaluations=evaluator.calls - before_calls,
                track="constraint_lane",
                constraint_category=operator_name,
                # This is a probe-not-started event, not a dynamic removal
                # event.  Keep it outside the configured count gate while
                # retaining the selected tier/count in the reason.
                removal_trigger=(
                    "probe_not_started:"
                    f"tier={selection.tier.value};"
                    f"requested={selection.requested_count};"
                    f"{selection.trigger_reason}"
                ),
                reset_observed=selection.reset_observed,
            )
        )
        return RemovalProposal(operator_name, None, (), selection, (), tuple(events))

    exact_evaluations = evaluator.calls - before_calls
    if not scores:
        if not events:
            events.append(
                NeighborhoodEvent(
                    operator_name,
                    "failed",
                    "no_rankable_customer",
                    track="constraint_lane",
                    constraint_category=operator_name,
                    removal_tier=selection.tier.value,
                    removal_size_requested=selection.requested_count,
                    stagnation_iterations=selection.stagnation_iterations,
                    removal_trigger=selection.trigger_reason,
                    reset_observed=selection.reset_observed,
                )
            )
        return RemovalProposal(operator_name, None, (), selection, (), tuple(events))

    if selected_operator is ConstraintRemovalOperator.SHAW_RELATED:
        ordered = sorted(scores, key=lambda item: (item[1], item[2], item[0]))
    else:
        ordered = sorted(scores, key=lambda item: (-item[1], item[2], item[0]))
    actual_count = min(selection.requested_count, len(ordered), len(all_customers) - 1)
    chosen = tuple(item[0] for item in ordered[:actual_count])
    chosen_set = set(chosen)
    partial = tuple(
        tuple(name for name in sequence if name not in chosen_set)
        for sequence in sequences
    )
    partial = tuple(sequence for sequence in partial if sequence)
    affected = tuple(
        index
        for index, sequence in enumerate(sequences)
        if any(name in chosen_set for name in sequence)
    )
    score_map = {name: score for name, score, _ in scores}
    events.append(
        NeighborhoodEvent(
            operator_name,
            "candidate_proposed",
            "constraint_ranked_removal",
            route_indices=affected,
            affected_route_indices=affected,
            removed_customers=chosen,
            candidate_route_sequences=partial,
            candidate_feasible=False,
            prefilter_passed=True,
            exact_route_evaluations=exact_evaluations,
            selection_rank=1,
            track="constraint_lane",
            constraint_category=operator_name,
            removal_tier=selection.tier.value,
            removal_size_requested=selection.requested_count,
            removal_size_actual=actual_count,
            stagnation_iterations=selection.stagnation_iterations,
            removal_trigger=selection.trigger_reason,
            reset_observed=selection.reset_observed,
            ranking_score=score_map[chosen[0]] if chosen else 0.0,
        )
    )
    return RemovalProposal(
        operator_name,
        partial,
        chosen,
        selection,
        tuple((name, score) for name, score, _ in ordered),
        tuple(events),
    )


def _next_removal_tier(tier: RemovalTier) -> RemovalTier:
    if tier is RemovalTier.SMALL:
        return RemovalTier.MEDIUM
    if tier is RemovalTier.MEDIUM:
        return RemovalTier.LARGE
    return RemovalTier.LARGE


def _removal_tier_rank(tier: RemovalTier) -> int:
    return {
        RemovalTier.SMALL: 0,
        RemovalTier.MEDIUM: 1,
        RemovalTier.LARGE: 2,
    }[tier]


def _removal_fraction_bounds(
    tier: RemovalTier,
    config: VehicleOperatorConfig,
) -> tuple[float, float]:
    if tier is RemovalTier.SMALL:
        return config.small_removal_min_fraction, config.small_removal_max_fraction
    if tier is RemovalTier.MEDIUM:
        return config.medium_removal_min_fraction, config.medium_removal_max_fraction
    return config.large_removal_min_fraction, config.large_removal_max_fraction


def _constraint_route_scores(
    instance: Instance,
    sequence: CustomerSequence,
    result: ChargingSubproblemResult,
    operator: ConstraintRemovalOperator,
    *,
    anchor: str | None,
) -> dict[str, float]:
    if operator is ConstraintRemovalOperator.STATION_PRESSURE:
        return _station_pressure_scores(instance, sequence, result)
    if operator is ConstraintRemovalOperator.TIME_WINDOW_CONFLICT:
        return _time_window_conflict_scores(instance, result)
    if operator is ConstraintRemovalOperator.WORST_ENERGY_DETOUR:
        return _worst_energy_detour_scores(instance, result)
    if anchor is None:
        raise ValueError("Shaw-related removal requires a seeded anchor")
    return _shaw_related_scores(instance, sequence, anchor)


def _station_pressure_scores(
    instance: Instance,
    sequence: CustomerSequence,
    result: ChargingSubproblemResult,
) -> dict[str, float]:
    path = result.route
    customer_positions = [
        (position, name)
        for position, name in enumerate(path)
        if instance.by_name[name].kind is NodeType.CUSTOMER
    ]
    station_count = sum(
        instance.by_name[name].kind is NodeType.STATION for name in path
    )
    route_pressure = (
        2.0 * station_count
        + result.charged_energy
        + 10.0 * result.charging_time
    )
    output: dict[str, float] = {}
    for index, (_position, customer) in enumerate(customer_positions):
        left = customer_positions[index - 1][0] if index else 0
        right = (
            customer_positions[index + 1][0]
            if index + 1 < len(customer_positions)
            else len(path) - 1
        )
        local_path = path[left : right + 1]
        local_distance = _path_distance(instance, local_path)
        direct_distance = instance.by_name[path[left]].distance_to(instance.by_name[path[right]])
        local_stations = sum(
            instance.by_name[name].kind is NodeType.STATION for name in local_path
        )
        output[customer] = (
            local_distance - direct_distance
            + 2.0 * local_stations
            + route_pressure / max(1, len(customer_positions))
        )
    return output


def _time_window_conflict_scores(
    instance: Instance,
    result: ChargingSubproblemResult,
) -> dict[str, float]:
    path = result.route
    current_time = max(0.0, instance.depot.ready_time)
    battery = instance.vehicle.battery_capacity
    scores: dict[str, float] = {}
    for origin_name, destination_name in zip(path, path[1:], strict=False):
        origin = instance.by_name[origin_name]
        destination = instance.by_name[destination_name]
        distance = origin.distance_to(destination)
        battery -= distance * instance.vehicle.consumption_rate
        current_time += distance / instance.vehicle.average_velocity
        current_time = max(current_time, destination.ready_time)
        if destination.kind is NodeType.STATION:
            charged = instance.vehicle.battery_capacity - max(0.0, battery)
            current_time += charged * instance.vehicle.inverse_refueling_rate
            battery = instance.vehicle.battery_capacity
        elif destination.kind is NodeType.CUSTOMER:
            scores[destination_name] = destination.due_date - current_time
            current_time += destination.service_time
    return {name: -slack for name, slack in scores.items()}


def _worst_energy_detour_scores(
    instance: Instance,
    result: ChargingSubproblemResult,
) -> dict[str, float]:
    path = result.route
    customer_positions = [
        (position, name)
        for position, name in enumerate(path)
        if instance.by_name[name].kind is NodeType.CUSTOMER
    ]
    output: dict[str, float] = {}
    for index, (_position, customer) in enumerate(customer_positions):
        left = customer_positions[index - 1][0] if index else 0
        right = (
            customer_positions[index + 1][0]
            if index + 1 < len(customer_positions)
            else len(path) - 1
        )
        actual = _path_distance(instance, path[left : right + 1])
        direct = instance.by_name[path[left]].distance_to(instance.by_name[path[right]])
        output[customer] = actual - direct
    return output


def _shaw_related_scores(
    instance: Instance,
    sequence: CustomerSequence,
    anchor: str,
) -> dict[str, float]:
    anchor_node = instance.by_name[anchor]
    max_distance = max(
        (anchor_node.distance_to(node) for node in instance.customers),
        default=1.0,
    )
    max_time = max((node.due_date for node in instance.customers), default=1.0)
    max_demand = max((node.demand for node in instance.customers), default=1.0)
    output: dict[str, float] = {}
    for name in sequence:
        node = instance.by_name[name]
        distance = anchor_node.distance_to(node) / max(max_distance, _EPSILON)
        time_difference = (
            abs(anchor_node.ready_time - node.ready_time)
            + abs(anchor_node.due_date - node.due_date)
        ) / max(max_time, _EPSILON)
        demand_difference = abs(anchor_node.demand - node.demand) / max(max_demand, _EPSILON)
        energy_penalty = 0.0
        if not _energy_reachable_optimistically(instance, anchor, name):
            energy_penalty = 1.0
        output[name] = distance + 0.25 * time_difference + 0.25 * demand_difference + energy_penalty
    return output


def _path_distance(instance: Instance, path: tuple[str, ...]) -> float:
    return sum(
        instance.by_name[left].distance_to(instance.by_name[right])
        for left, right in zip(path, path[1:], strict=False)
    )


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


def repair_vehicle_reduction_refinement(
    partial: RouteSequences,
    removed: tuple[str, ...],
    evaluator: RouteEvaluator,
    instance: Instance,
    *,
    budget: int,
) -> RepairResult:
    """Regret-repair a reduced-fleet candidate without creating a route.

    This bounded intensification is used only after a legacy-lane move has
    reduced the vehicle count.  It deliberately keeps the existing route set,
    uses the exact route evaluator only after safe screening, and uses explicit
    deterministic tie-breaks so a late wall-clock timeout cannot change the
    insertion ordering silently.
    """

    if budget <= 0:
        raise ValueError("budget must be positive")
    sequences = list(partial)
    pending = list(removed)
    before_calls = evaluator.calls
    try:
        while pending:
            options_by_customer = {
                customer: _regret_insertion_options(
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
                customer
                for customer, options in options_by_customer.items()
                if options
            ]
            if not feasible_customers:
                return RepairResult(
                    None,
                    0,
                    evaluator.calls - before_calls,
                    "no_existing_route_insertion",
                )

            customer = max(
                feasible_customers,
                key=lambda item: (
                    (
                        float("inf")
                        if len(options_by_customer[item]) < 2
                        else options_by_customer[item][1][0]
                        - options_by_customer[item][0][0]
                    ),
                    item,
                ),
            )
            _delta, route_index, _position, sequence = options_by_customer[customer][0]
            sequences[route_index] = sequence
            pending.remove(customer)
    except _EvaluationBudgetExceeded:
        return RepairResult(
            None,
            0,
            evaluator.calls - before_calls,
            "evaluation_budget_exhausted",
        )
    return RepairResult(
        tuple(sequences),
        0,
        evaluator.calls - before_calls,
        "",
    )


def repair_constraint_removal(
    original: RouteSequences,
    partial: RouteSequences,
    removed: tuple[str, ...],
    evaluator: RouteEvaluator,
    instance: Instance,
    *,
    budget: int = 16,
) -> RepairResult:
    """Repair a constraint probe into existing routes under an exact budget.

    The repair is deliberately bounded and deterministic: it restores every removed
    customer by trying capacity/time-window/energy-screened insertions into the
    existing routes, then exact-evaluates only the changed route.  It never creates
    a route and never silently falls back to the original solution.
    """

    if budget <= 0:
        raise ValueError("budget must be positive")
    partial_customers = [name for sequence in partial for name in sequence]
    original_customers = [name for sequence in original for name in sequence]
    if (
        len(set(original_customers)) != len(original_customers)
        or len(set(removed)) != len(removed)
        or sorted((*partial_customers, *removed)) != sorted(original_customers)
    ):
        return RepairResult(None, 0, 0, "constraint_removal_customer_coverage")

    result = _repair_pass(
        partial,
        removed,
        evaluator,
        instance,
        allow_new_routes=False,
        budget=budget,
    )
    if result.sequences is None and result.failure_reason == "no_existing_route_insertion":
        return RepairResult(
            None,
            0,
            result.exact_route_evaluations,
            "constraint_removal_no_existing_route_insertion",
        )
    if result.sequences == original:
        return RepairResult(
            None,
            0,
            result.exact_route_evaluations,
            "constraint_removal_no_change",
        )
    return result


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
        before_calls = evaluator.calls
        result = evaluator.route(sequence)
        exact_delta = evaluator.calls - before_calls
        if not result.feasible:
            events.append(
                NeighborhoodEvent(
                    "route_merge",
                    "failed",
                    "source_route_infeasible",
                    route_indices=(index,),
                    exact_route_evaluations=exact_delta,
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
                requires_exact_evaluation = not _is_cached_route(evaluator, merged)
                if (
                    requires_exact_evaluation
                    and exact_evaluations >= config.route_merge_exact_evaluation_budget
                ):
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
                exact_delta = evaluator.calls - before_calls
                exact_evaluations += exact_delta
                if not result.feasible:
                    events.append(
                        NeighborhoodEvent(
                            "route_merge",
                            "exact_infeasible",
                            result.failure_reason or "exact_charging_infeasible",
                            route_indices=(left.index, right.index),
                            candidate_customer_sequence=merged,
                            prefilter_passed=True,
                            exact_route_evaluations=exact_delta,
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
                        exact_route_evaluations=exact_delta,
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
            # Exact calls belong to the individual feasible/infeasible probe
            # events above; the final proposal only selects the best one.
            exact_route_evaluations=0,
        )
    )
    return MoveProposal("route_merge", tuple(new_sequences), tuple(events))


@dataclass(frozen=True, slots=True)
class _CandidateDescription:
    changes: tuple[tuple[int, CustomerSequence], ...]
    removed_customers: tuple[str, ...] = ()
    chain_depth: int = 0
    segment_length: int = 0


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    objective: SolutionObjective | None
    reason: str
    prefilter_passed: bool
    exact_route_evaluations: int


def propose_relocate(
    instance: Instance,
    sequences: RouteSequences,
    evaluator: RouteEvaluator,
    *,
    config: VehicleOperatorConfig = _DEFAULT_VEHICLE_OPERATOR_CONFIG,
    precomputed_routes: Mapping[CustomerSequence, ChargingSubproblemResult] | None = None,
) -> MoveProposal:
    """Move one customer between two existing routes without changing fleet size."""

    if len(sequences) <= 1:
        return MoveProposal(
            "relocate",
            None,
            (NeighborhoodEvent("relocate", "not_applicable", "only_one_route"),),
        )
    return _search_changed_candidates(
        "relocate",
        instance,
        sequences,
        evaluator,
        _relocate_candidates(sequences),
        budget=config.relocate_exact_evaluation_budget,
        precomputed_routes=precomputed_routes,
    )


def propose_swap(
    instance: Instance,
    sequences: RouteSequences,
    evaluator: RouteEvaluator,
    *,
    config: VehicleOperatorConfig = _DEFAULT_VEHICLE_OPERATOR_CONFIG,
    precomputed_routes: Mapping[CustomerSequence, ChargingSubproblemResult] | None = None,
) -> MoveProposal:
    """Exchange one customer from each of two existing routes."""

    if len(sequences) <= 1:
        return MoveProposal(
            "swap",
            None,
            (NeighborhoodEvent("swap", "not_applicable", "only_one_route"),),
        )
    return _search_changed_candidates(
        "swap",
        instance,
        sequences,
        evaluator,
        _swap_candidates(sequences),
        budget=config.swap_exact_evaluation_budget,
        precomputed_routes=precomputed_routes,
    )


def propose_two_opt_star(
    instance: Instance,
    sequences: RouteSequences,
    evaluator: RouteEvaluator,
    *,
    config: VehicleOperatorConfig = _DEFAULT_VEHICLE_OPERATOR_CONFIG,
    precomputed_routes: Mapping[CustomerSequence, ChargingSubproblemResult] | None = None,
) -> MoveProposal:
    """Exchange tails of two routes at customer-to-customer cut points."""

    if len(sequences) <= 1:
        return MoveProposal(
            "two_opt_star",
            None,
            (NeighborhoodEvent("two_opt_star", "not_applicable", "only_one_route"),),
        )
    return _search_changed_candidates(
        "two_opt_star",
        instance,
        sequences,
        evaluator,
        _two_opt_star_candidates(sequences),
        budget=config.two_opt_star_exact_evaluation_budget,
        precomputed_routes=precomputed_routes,
    )


def propose_route_segment_destroy(
    instance: Instance,
    sequences: RouteSequences,
    evaluator: RouteEvaluator,
    *,
    config: VehicleOperatorConfig = _DEFAULT_VEHICLE_OPERATOR_CONFIG,
) -> MoveProposal:
    """Remove a contiguous segment and repair it only into existing routes."""

    operator = "route_segment_destroy"
    min_length = config.route_segment_min_length
    max_length = config.route_segment_max_length
    events: list[NeighborhoodEvent] = []
    exact_used = 0
    candidate_limit = max(16, config.route_segment_exact_evaluation_budget * 4)
    considered = 0

    for source_index, source in enumerate(sequences):
        if len(source) <= min_length:
            continue
        for segment_length in range(min_length, min(max_length, len(source) - 1) + 1):
            for start in range(len(source) - segment_length + 1):
                if considered >= candidate_limit:
                    events.append(
                        NeighborhoodEvent(
                            operator,
                            "budget_exhausted",
                            "route_segment_candidate_budget",
                            route_indices=(source_index,),
                            affected_route_indices=(source_index,),
                            segment_length=segment_length,
                        )
                    )
                    break
                considered += 1
                segment = source[start : start + segment_length]
                partial = list(sequences)
                partial[source_index] = source[:start] + source[start + segment_length :]
                remaining = config.route_segment_exact_evaluation_budget - exact_used
                if remaining <= 0:
                    events.append(
                        NeighborhoodEvent(
                            operator,
                            "budget_exhausted",
                            "route_segment_exact_evaluation_budget",
                            route_indices=(source_index,),
                            affected_route_indices=(source_index,),
                            removed_customers=segment,
                            segment_length=segment_length,
                        )
                    )
                    break
                repair_config = replace(
                    config,
                    vehicle_repair_exact_evaluation_budget=remaining,
                )
                repair = repair_vehicle_count_aware(
                    tuple(partial),
                    segment,
                    evaluator,
                    instance,
                    config=repair_config,
                    allow_new_routes=False,
                )
                exact_used += repair.exact_route_evaluations
                if repair.sequences is None or len(repair.sequences) != len(sequences):
                    events.append(
                        NeighborhoodEvent(
                            operator,
                            "failed",
                            repair.failure_reason or "route_segment_repair_failed",
                            route_indices=(source_index,),
                            affected_route_indices=(source_index,),
                            removed_customers=segment,
                            candidate_vehicle_delta=0,
                            candidate_feasible=False,
                            prefilter_passed=repair.exact_route_evaluations > 0,
                            new_routes_created=repair.new_routes_created,
                            exact_route_evaluations=repair.exact_route_evaluations,
                            segment_length=segment_length,
                        )
                    )
                    continue

                affected = tuple(
                    index
                    for index, (before, after) in enumerate(
                        zip(sequences, repair.sequences, strict=True)
                    )
                    if before != after
                )
                if not affected:
                    events.append(
                        NeighborhoodEvent(
                            operator,
                            "failed",
                            "route_segment_no_change",
                            route_indices=(source_index,),
                            affected_route_indices=(),
                            removed_customers=segment,
                            candidate_vehicle_delta=0,
                            candidate_feasible=False,
                            prefilter_passed=True,
                            new_routes_created=repair.new_routes_created,
                            exact_route_evaluations=repair.exact_route_evaluations,
                            segment_length=segment_length,
                            selection_rank=considered,
                        )
                    )
                    continue
                events.append(
                    NeighborhoodEvent(
                        operator,
                        "candidate_proposed",
                        "route_segment_repaired",
                        route_indices=(source_index,),
                        affected_route_indices=affected,
                        removed_customers=segment,
                        candidate_route_sequences=repair.sequences,
                        candidate_vehicle_delta=0,
                        candidate_feasible=True,
                        prefilter_passed=True,
                        new_routes_created=repair.new_routes_created,
                        exact_route_evaluations=repair.exact_route_evaluations,
                        segment_length=segment_length,
                        selection_rank=considered,
                    )
                )
                return MoveProposal(operator, repair.sequences, tuple(events))
            if exact_used >= config.route_segment_exact_evaluation_budget:
                break
        if exact_used >= config.route_segment_exact_evaluation_budget:
            break

    if not events:
        events.append(
            NeighborhoodEvent(operator, "failed", "no_route_with_segment_length")
        )
    return MoveProposal(operator, None, tuple(events))


def propose_ejection_chain(
    instance: Instance,
    sequences: RouteSequences,
    evaluator: RouteEvaluator,
    *,
    config: VehicleOperatorConfig = _DEFAULT_VEHICLE_OPERATOR_CONFIG,
    precomputed_routes: Mapping[CustomerSequence, ChargingSubproblemResult] | None = None,
) -> MoveProposal:
    """Search bounded relocate/ejection chains over existing routes.

    A chain removes one pending customer from a source route, inserts it into a
    different route while ejecting one customer, and repeats for at most the
    configured depth.  The final pending customer is inserted without an
    ejection.  Route count therefore remains unchanged and every customer is
    present exactly once in a completed proposal.
    """

    operator = "ejection_chain"
    if len(sequences) <= 1:
        return MoveProposal(
            operator,
            None,
            (NeighborhoodEvent(operator, "not_applicable", "only_one_route"),),
        )
    if config.ejection_chain_max_depth < 1:
        raise ValueError("ejection_chain_max_depth must be positive")

    events: list[NeighborhoodEvent] = []
    exact_used = 0
    base_results = _base_route_results(
        evaluator,
        sequences,
        precomputed_routes=precomputed_routes,
    )
    considered = 0
    candidate_limit = max(16, config.ejection_chain_exact_evaluation_budget * 4)
    best: tuple[SolutionObjective, RouteSequences, int, tuple[int, ...]] | None = None

    initial_states = [
        (tuple(
            sequence[:position] + sequence[position + 1 :]
            if index == source_index
            else sequence
            for index, sequence in enumerate(sequences)
        ), customer, 0)
        for source_index, sequence in enumerate(sequences)
        if len(sequence) > 1
        for position, customer in enumerate(sequence)
    ]
    beam = sorted(
        initial_states,
        key=lambda state: (
            _route_sequence_distance(instance, state[0]),
            state[0],
            state[1],
        ),
    )[: config.ejection_chain_beam_width]
    seen: set[tuple[RouteSequences, str, int]] = set()

    for _ in range(config.ejection_chain_max_depth):
        next_beam: list[tuple[RouteSequences, str, int]] = []
        for state_sequences, pending, ejection_depth in beam:
            state_key = (state_sequences, pending, ejection_depth)
            if state_key in seen:
                continue
            seen.add(state_key)
            chain_depth = ejection_depth + 1
            for target_index, target in enumerate(state_sequences):
                for position in range(len(target) + 1):
                    if considered >= candidate_limit:
                        break
                    candidate_routes = list(state_sequences)
                    candidate_routes[target_index] = (
                        target[:position] + (pending,) + target[position:]
                    )
                    candidate = tuple(candidate_routes)
                    if candidate == sequences:
                        continue
                    changes = tuple(
                        (index, route)
                        for index, (before, route) in enumerate(
                            zip(sequences, candidate, strict=True)
                        )
                        if before != route
                    )
                    if not changes:
                        continue
                    considered += 1
                    description = _CandidateDescription(
                        changes,
                        chain_depth=chain_depth,
                    )
                    evaluation = _evaluate_changed_candidate(
                        instance,
                        evaluator,
                        changes,
                        base_results=base_results,
                        exact_used=exact_used,
                        budget=config.ejection_chain_exact_evaluation_budget,
                        budget_reason="ejection_chain_exact_evaluation_budget",
                    )
                    exact_used += evaluation.exact_route_evaluations
                    events.append(
                        _candidate_event(
                            operator,
                            description,
                            evaluation,
                            considered,
                        )
                    )
                    if evaluation.objective is not None:
                        candidate_key = (
                            evaluation.objective.key,
                            candidate,
                            chain_depth,
                            tuple(index for index, _ in changes),
                        )
                        if best is None or candidate_key < (
                            best[0].key,
                            best[1],
                            best[2],
                            best[3],
                        ):
                            best = (
                                evaluation.objective,
                                candidate,
                                chain_depth,
                                tuple(index for index, _ in changes),
                            )
                    if evaluation.reason == "ejection_chain_exact_evaluation_budget":
                        break
                if exact_used >= config.ejection_chain_exact_evaluation_budget:
                    break

            if exact_used >= config.ejection_chain_exact_evaluation_budget:
                break
            if ejection_depth + 1 >= config.ejection_chain_max_depth:
                continue
            for target_index, target in enumerate(state_sequences):
                if not target:
                    continue
                for position, ejected in enumerate(target):
                    next_routes = list(state_sequences)
                    next_routes[target_index] = (
                        target[:position] + (pending,) + target[position + 1 :]
                    )
                    next_state = tuple(next_routes)
                    if next_state == state_sequences:
                        continue
                    if any(
                        not screen_route_candidate(instance, route).accepted
                        for route in next_state
                    ):
                        continue
                    next_beam.append((next_state, ejected, ejection_depth + 1))
        if exact_used >= config.ejection_chain_exact_evaluation_budget:
            break
        beam = sorted(
            next_beam,
            key=lambda state: (
                _route_sequence_distance(instance, state[0]),
                state[0],
                state[1],
                state[2],
            ),
        )[: config.ejection_chain_beam_width]
        if not beam:
            break

    if best is None:
        if not events:
            events.append(NeighborhoodEvent(operator, "failed", "no_feasible_candidate"))
        return MoveProposal(operator, None, tuple(events))
    events.append(
        NeighborhoodEvent(
            operator,
            "candidate_proposed",
            "ejection_chain_completed",
            affected_route_indices=best[3],
            candidate_route_sequences=best[1],
            candidate_vehicle_delta=0,
            candidate_feasible=True,
            prefilter_passed=True,
            chain_depth=best[2],
        )
    )
    return MoveProposal(operator, best[1], tuple(events))


def _search_changed_candidates(
    operator: str,
    instance: Instance,
    sequences: RouteSequences,
    evaluator: RouteEvaluator,
    candidates: Iterable[_CandidateDescription],
    *,
    budget: int,
    precomputed_routes: Mapping[CustomerSequence, ChargingSubproblemResult] | None = None,
) -> MoveProposal:
    events: list[NeighborhoodEvent] = []
    exact_used = 0
    base_results = _base_route_results(
        evaluator,
        sequences,
        precomputed_routes=precomputed_routes,
    )
    candidate_limit = max(16, budget * 4)
    best: tuple[SolutionObjective, RouteSequences, _CandidateDescription] | None = None
    for rank, description in enumerate(candidates, start=1):
        if rank > candidate_limit:
            events.append(
                NeighborhoodEvent(
                    operator,
                    "budget_exhausted",
                    f"{operator}_candidate_budget",
                )
            )
            break
        evaluation = _evaluate_changed_candidate(
            instance,
            evaluator,
            description.changes,
            base_results=base_results,
            exact_used=exact_used,
            budget=budget,
            budget_reason=f"{operator}_exact_evaluation_budget",
        )
        exact_used += evaluation.exact_route_evaluations
        events.append(_candidate_event(operator, description, evaluation, rank))
        if evaluation.objective is not None:
            candidate = _apply_changes(sequences, description.changes)
            candidate_key = (evaluation.objective.key, candidate)
            if best is None or candidate_key < (best[0].key, best[1]):
                best = (evaluation.objective, candidate, description)
        if evaluation.reason == f"{operator}_exact_evaluation_budget":
            break
    if best is None:
        if not events:
            events.append(NeighborhoodEvent(operator, "failed", "no_feasible_candidate"))
        return MoveProposal(operator, None, tuple(events))
    events.append(
        _candidate_proposed_event(
            operator,
            best[2],
            reason=f"{operator}_candidate",
        )
    )
    return MoveProposal(operator, best[1], tuple(events))


def _relocate_candidates(sequences: RouteSequences) -> Iterable[_CandidateDescription]:
    for source_index, source in enumerate(sequences):
        if len(source) <= 1:
            continue
        for source_position, customer in enumerate(source):
            source_without = source[:source_position] + source[source_position + 1 :]
            for target_index, target in enumerate(sequences):
                if target_index == source_index:
                    continue
                for target_position in range(len(target) + 1):
                    target_with = (
                        target[:target_position]
                        + (customer,)
                        + target[target_position:]
                    )
                    yield_changes = _ordered_changes(
                        (source_index, source_without), (target_index, target_with)
                    )
                    yield _CandidateDescription(
                        yield_changes,
                        removed_customers=(customer,),
                    )


def _swap_candidates(sequences: RouteSequences) -> Iterable[_CandidateDescription]:
    for left_index, left in enumerate(sequences):
        for right_index in range(left_index + 1, len(sequences)):
            right = sequences[right_index]
            for left_position, left_customer in enumerate(left):
                for right_position, right_customer in enumerate(right):
                    new_left = (
                        left[:left_position]
                        + (right_customer,)
                        + left[left_position + 1 :]
                    )
                    new_right = (
                        right[:right_position]
                        + (left_customer,)
                        + right[right_position + 1 :]
                    )
                    yield _CandidateDescription(
                        _ordered_changes(
                            (left_index, new_left),
                            (right_index, new_right),
                        ),
                        removed_customers=(left_customer, right_customer),
                    )


def _two_opt_star_candidates(sequences: RouteSequences) -> Iterable[_CandidateDescription]:
    for left_index, left in enumerate(sequences):
        for right_index in range(left_index + 1, len(sequences)):
            right = sequences[right_index]
            for left_cut in range(1, len(left)):
                for right_cut in range(1, len(right)):
                    new_left = left[:left_cut] + right[right_cut:]
                    new_right = right[:right_cut] + left[left_cut:]
                    if not new_left or not new_right:
                        continue
                    yield _CandidateDescription(
                        _ordered_changes(
                            (left_index, new_left),
                            (right_index, new_right),
                        )
                    )


def _ordered_changes(
    *changes: tuple[int, CustomerSequence],
) -> tuple[tuple[int, CustomerSequence], ...]:
    return tuple(sorted(changes, key=lambda change: change[0]))


def _evaluate_changed_candidate(
    instance: Instance,
    evaluator: RouteEvaluator,
    changes: tuple[tuple[int, CustomerSequence], ...],
    *,
    base_results: tuple[ChargingSubproblemResult, ...],
    exact_used: int,
    budget: int,
    budget_reason: str,
) -> _CandidateEvaluation:
    if not changes:
        return _CandidateEvaluation(None, "no_changed_route", False, 0)
    if len({index for index, _ in changes}) != len(changes):
        return _CandidateEvaluation(None, "duplicate_changed_route", False, 0)
    for _, sequence in changes:
        screen = screen_route_candidate(instance, sequence)
        if not screen.accepted:
            return _CandidateEvaluation(None, screen.reason, False, 0)
    required = sum(
        not _is_cached_route(evaluator, sequence) for _, sequence in changes
    )
    if exact_used + required > budget:
        return _CandidateEvaluation(
            None,
            budget_reason,
            True,
            0,
        )
    results: list[ChargingSubproblemResult] = []
    exact_evaluations = 0
    for _, sequence in changes:
        before_calls = evaluator.calls
        results.append(evaluator.route(sequence))
        exact_evaluations += evaluator.calls - before_calls
    if not all(result.feasible for result in results):
        reason = next(
            (
                result.failure_reason or "exact_charging_infeasible"
                for result in results
                if not result.feasible
            ),
        )
        return _CandidateEvaluation(None, reason, True, exact_evaluations)
    candidate_results = list(base_results)
    for (index, _), result in zip(changes, results, strict=True):
        candidate_results[index] = result
    objective = sum(
        (_route_objective(instance, result) for result in candidate_results),
        start=SolutionObjective.zero(),
    )
    return _CandidateEvaluation(
        objective,
        "exact_charging_feasible",
        True,
        exact_evaluations,
    )


def _is_cached_route(
    evaluator: RouteEvaluator,
    sequence: CustomerSequence,
) -> bool:
    cache = getattr(evaluator, "cache", None)
    return isinstance(cache, Mapping) and sequence in cache


def _base_route_results(
    evaluator: RouteEvaluator,
    sequences: RouteSequences,
    *,
    precomputed_routes: Mapping[CustomerSequence, ChargingSubproblemResult] | None = None,
) -> tuple[ChargingSubproblemResult, ...]:
    return tuple(
        precomputed_routes[sequence]
        if precomputed_routes is not None and sequence in precomputed_routes
        else evaluator.route(sequence)
        for sequence in sequences
    )


def _candidate_event(
    operator: str,
    description: _CandidateDescription,
    evaluation: _CandidateEvaluation,
    rank: int,
) -> NeighborhoodEvent:
    if evaluation.objective is not None:
        status = "feasible_candidate"
    elif evaluation.reason.endswith("_exact_evaluation_budget"):
        status = "budget_exhausted"
    elif evaluation.prefilter_passed:
        status = "exact_infeasible"
    else:
        status = "prefilter_rejected"
    indices = tuple(index for index, _ in description.changes)
    routes = tuple(sequence for _, sequence in description.changes)
    return NeighborhoodEvent(
        operator,
        status,
        evaluation.reason,
        route_indices=indices,
        affected_route_indices=indices,
        removed_customers=description.removed_customers,
        candidate_customer_sequence=routes[0] if len(routes) == 1 else (),
        candidate_route_sequences=routes,
        candidate_feasible=evaluation.objective is not None,
        prefilter_passed=evaluation.prefilter_passed,
        exact_route_evaluations=evaluation.exact_route_evaluations,
        selection_rank=rank,
        chain_depth=description.chain_depth,
        segment_length=description.segment_length,
    )


def _candidate_proposed_event(
    operator: str,
    description: _CandidateDescription,
    *,
    reason: str,
) -> NeighborhoodEvent:
    indices = tuple(index for index, _ in description.changes)
    routes = tuple(sequence for _, sequence in description.changes)
    return NeighborhoodEvent(
        operator,
        "candidate_proposed",
        reason,
        route_indices=indices,
        affected_route_indices=indices,
        removed_customers=description.removed_customers,
        candidate_customer_sequence=routes[0] if len(routes) == 1 else (),
        candidate_route_sequences=routes,
        candidate_vehicle_delta=0,
        candidate_feasible=True,
        prefilter_passed=True,
        chain_depth=description.chain_depth,
        segment_length=description.segment_length,
    )


def _apply_changes(
    sequences: RouteSequences,
    changes: tuple[tuple[int, CustomerSequence], ...],
) -> RouteSequences:
    by_index = dict(changes)
    return tuple(by_index.get(index, sequence) for index, sequence in enumerate(sequences))


def _route_sequence_distance(instance: Instance, sequences: RouteSequences) -> float:
    depot = instance.depot
    return sum(
        sum(
            origin.distance_to(destination)
            for origin, destination in zip(
                (depot, *(instance.by_name[name] for name in sequence), depot),
                (*(instance.by_name[name] for name in sequence), depot),
                strict=False,
            )
        )
        for sequence in sequences
    )


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
            if not screen_route_candidate(instance, candidate).accepted:
                continue
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


def _regret_insertion_options(
    sequences: list[CustomerSequence],
    customer: str,
    evaluator: RouteEvaluator,
    instance: Instance,
    *,
    before_calls: int,
    budget: int,
) -> list[tuple[float, int, int, CustomerSequence]]:
    options: list[tuple[float, int, int, CustomerSequence]] = []
    for route_index, base in enumerate(sequences):
        base_demand = sum(instance.by_name[name].demand for name in base)
        if (
            base_demand + instance.by_name[customer].demand
            > instance.vehicle.load_capacity + _EPSILON
        ):
            continue
        if evaluator.calls - before_calls >= budget:
            raise _EvaluationBudgetExceeded
        old_distance = evaluator.route(base).distance if base else 0.0
        for position in range(len(base) + 1):
            if evaluator.calls - before_calls >= budget:
                raise _EvaluationBudgetExceeded
            candidate = (*base[:position], customer, *base[position:])
            if not screen_route_candidate(instance, candidate).accepted:
                continue
            result = evaluator.route(candidate)
            if result.feasible:
                options.append(
                    (
                        result.distance - old_distance,
                        route_index,
                        position,
                        candidate,
                    )
                )
    return sorted(options, key=lambda option: (option[0], option[1], option[2], option[3]))


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
