from __future__ import annotations

from dataclasses import dataclass, replace

from evrptw import neighborhoods
from evrptw.charging import ChargingSubproblemResult
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.neighborhoods import (
    ConstraintRemovalOperator,
    RemovalTier,
    RepairResult,
    VehicleOperatorConfig,
    propose_constraint_removal,
    propose_ejection_chain,
    propose_relocate,
    propose_route_elimination,
    propose_route_merge,
    propose_route_segment_destroy,
    propose_swap,
    propose_two_opt_star,
    repair_constraint_removal,
    repair_vehicle_count_aware,
    repair_vehicle_reduction_refinement,
    screen_route_candidate,
    select_dynamic_removal_size,
)


@dataclass
class FakeEvaluator:
    instance: Instance
    maximum_customers_per_route: int = 2
    calls: int = 0
    candidate_transaction_enabled: bool = False
    pair_pruning_enabled: bool = False

    def route(self, sequence: tuple[str, ...]) -> ChargingSubproblemResult:
        self.calls += 1
        route = (self.instance.depot.name, *sequence, self.instance.depot.name)
        feasible = len(sequence) <= self.maximum_customers_per_route
        distance = float(len(sequence) * 10 + sum(int(name[1:]) for name in sequence))
        return ChargingSubproblemResult(
            feasible=feasible,
            route=route if feasible else (),
            distance=distance if feasible else float("inf"),
            total_energy=distance if feasible else 0.0,
            charged_energy=0.0,
            charging_time=0.0,
            labels_generated=1,
            labels_expanded=1,
            labels_pruned=0,
            runtime_seconds=0.0,
            failure_reason="" if feasible else "fake route capacity limit",
        )


def _instance(*, battery_capacity: float = 100.0, load_capacity: float = 10.0) -> Instance:
    return Instance(
        "neighborhood_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 1000.0, 0.0),
            Node("F1", NodeType.STATION, 5.0, 0.0, 0.0, 0.0, 1000.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 1000.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 1000.0, 0.0),
            Node("C3", NodeType.CUSTOMER, 3.0, 0.0, 1.0, 0.0, 1000.0, 0.0),
            Node("C4", NodeType.CUSTOMER, 4.0, 0.0, 1.0, 0.0, 1000.0, 0.0),
        ),
        Vehicle(battery_capacity, load_capacity, 1.0, 0.1, 1.0),
    )


def test_route_elimination_reinserts_every_customer_without_creating_route() -> None:
    instance = _instance()
    evaluator = FakeEvaluator(instance)

    proposal = propose_route_elimination(
        instance,
        (("C1",), ("C2",), ("C3",)),
        evaluator,
        config=VehicleOperatorConfig(max_route_elimination_attempts=3),
    )

    assert proposal.sequences is not None
    assert len(proposal.sequences) == 2
    assert sorted(name for route in proposal.sequences for name in route) == ["C1", "C2", "C3"]
    assert proposal.operator == "route_elimination"
    assert any(event.status == "candidate_proposed" for event in proposal.events)
    assert all(event.new_routes_created == 0 for event in proposal.events)


def test_route_elimination_reports_failure_when_existing_routes_cannot_accept_customer() -> None:
    instance = _instance()
    evaluator = FakeEvaluator(instance, maximum_customers_per_route=1)

    proposal = propose_route_elimination(
        instance,
        (("C1",), ("C2",), ("C3",)),
        evaluator,
        config=VehicleOperatorConfig(max_route_elimination_attempts=2),
    )

    assert proposal.sequences is None
    assert any(event.reason == "no_existing_route_insertion" for event in proposal.events)


def test_vehicle_count_aware_repair_creates_route_only_after_existing_routes_fail() -> None:
    instance = _instance()
    evaluator = FakeEvaluator(instance, maximum_customers_per_route=1)

    result = repair_vehicle_count_aware(
        (("C1",),),
        ("C2",),
        evaluator,
        instance,
        config=VehicleOperatorConfig(),
        allow_new_routes=True,
    )

    assert result.sequences == (("C1",), ("C2",))
    assert result.new_routes_created == 1
    assert result.failure_reason == ""


def test_vehicle_count_aware_repair_does_not_fallback_after_budget_exhaustion() -> None:
    instance = _instance()
    evaluator = FakeEvaluator(instance, maximum_customers_per_route=2)
    config = VehicleOperatorConfig(vehicle_repair_exact_evaluation_budget=1)

    result = repair_vehicle_count_aware(
        (("C1",),),
        ("C2",),
        evaluator,
        instance,
        config=config,
        allow_new_routes=True,
    )

    assert result.sequences is None
    assert result.new_routes_created == 0
    assert result.failure_reason == "evaluation_budget_exhausted"


def test_vehicle_reduction_refinement_uses_existing_routes_and_is_bounded() -> None:
    instance = _instance()
    evaluator = FakeEvaluator(instance, maximum_customers_per_route=2)

    result = repair_vehicle_reduction_refinement(
        (("C1",), ("C2",)),
        ("C3",),
        evaluator,
        instance,
        budget=64,
    )

    assert result.sequences is not None
    assert len(result.sequences) == 2
    assert result.new_routes_created == 0
    assert sorted(name for route in result.sequences for name in route) == [
        "C1",
        "C2",
        "C3",
    ]
    assert result.exact_route_evaluations <= 64


def test_vehicle_repair_screens_time_windows_before_exact_evaluation() -> None:
    base = _instance()
    instance = Instance(
        base.name,
        tuple(
            replace(node, due_date=0.1) if node.name == "C2" else node
            for node in base.nodes
        ),
        base.vehicle,
    )
    evaluator = FakeEvaluator(instance)

    result = repair_vehicle_count_aware(
        (("C1",),),
        ("C2",),
        evaluator,
        instance,
        config=VehicleOperatorConfig(),
        allow_new_routes=False,
    )

    assert result.sequences is None
    assert result.failure_reason == "no_existing_route_insertion"
    assert evaluator.calls == 0


def test_route_merge_capacity_prefilter_avoids_exact_merged_route_evaluation() -> None:
    instance = _instance(load_capacity=1.0)
    evaluator = FakeEvaluator(
        instance,
        candidate_transaction_enabled=True,
        pair_pruning_enabled=True,
    )
    sequences = (("C1",), ("C2",))

    before = evaluator.calls
    proposal = propose_route_merge(instance, sequences, evaluator)

    assert proposal.sequences is None
    assert evaluator.calls == before + len(sequences)
    aggregate = next(
        event
        for event in proposal.events
        if event.status == "pair_prefilter_rejected_aggregate"
    )
    assert aggregate.reason == "capacity_prefilter"
    assert aggregate.route_indices == (0, 1)
    assert aggregate.aggregate_count == 4
    assert len(aggregate.candidate_pool_hash) == 64
    assert aggregate.candidate_route_sequences == sequences


def test_route_merge_pair_capacity_hash_is_stable_and_boundary_is_inclusive() -> None:
    rejected_instance = _instance(load_capacity=1.0)
    sequences = (("C1",), ("C2",))

    first = propose_route_merge(
        rejected_instance,
        sequences,
        FakeEvaluator(
            rejected_instance,
            candidate_transaction_enabled=True,
            pair_pruning_enabled=True,
        ),
    )
    second = propose_route_merge(
        rejected_instance,
        sequences,
        FakeEvaluator(
            rejected_instance,
            candidate_transaction_enabled=True,
            pair_pruning_enabled=True,
        ),
    )
    first_event = next(
        event for event in first.events if event.status == "pair_prefilter_rejected_aggregate"
    )
    second_event = next(
        event for event in second.events if event.status == "pair_prefilter_rejected_aggregate"
    )

    assert first_event.candidate_pool_hash == second_event.candidate_pool_hash

    boundary_instance = _instance(load_capacity=2.0)
    boundary = propose_route_merge(
        boundary_instance,
        sequences,
        FakeEvaluator(
            boundary_instance,
            candidate_transaction_enabled=True,
            pair_pruning_enabled=True,
        ),
    )
    assert all(
        event.status != "pair_prefilter_rejected_aggregate" for event in boundary.events
    )
    assert boundary.sequences is not None


def test_route_merge_pair_pruning_is_stage052_opt_in() -> None:
    instance = _instance(load_capacity=1.0)

    historical = propose_route_merge(instance, (("C1",), ("C2",)), FakeEvaluator(instance))

    assert all(
        event.status != "pair_prefilter_rejected_aggregate"
        for event in historical.events
    )


def test_native_route_merge_preserves_duplicate_candidate_positions() -> None:
    instance = _instance()

    class RecordingEvaluator(FakeEvaluator):
        ordered_candidates: tuple[tuple[str, ...], ...] = ()

        def candidate_route_batch(
            self,
            sequences: tuple[tuple[str, ...], ...],
            **_kwargs: object,
        ) -> tuple[ChargingSubproblemResult, ...]:
            self.ordered_candidates = sequences
            return tuple(self.route(sequence) for sequence in sequences)

    evaluator = RecordingEvaluator(
        instance,
        candidate_transaction_enabled=True,
        pair_pruning_enabled=True,
    )

    proposal = propose_route_merge(instance, (("C1",), ("C2",)), evaluator)

    assert proposal.sequences is not None
    assert evaluator.ordered_candidates == (
        ("C1", "C2"),
        ("C2", "C1"),
        ("C2", "C1"),
        ("C1", "C2"),
    )


def test_route_merge_produces_one_route_after_safe_prefilters() -> None:
    instance = _instance()
    evaluator = FakeEvaluator(instance)

    proposal = propose_route_merge(instance, (("C1",), ("C2",)), evaluator)

    assert proposal.sequences is not None
    assert len(proposal.sequences) == 1
    assert sorted(proposal.sequences[0]) == ["C1", "C2"]
    assert any(event.status == "candidate_proposed" for event in proposal.events)


def test_route_screen_rejects_energy_unreachable_sequence_without_solver() -> None:
    instance = _instance(battery_capacity=1.0)

    result = screen_route_candidate(instance, ("C3",))

    assert result.accepted is False
    assert result.reason == "energy_prefilter"


def test_relocate_keeps_route_count_and_coverage() -> None:
    instance = _instance()
    proposal = propose_relocate(
        instance,
        (("C1", "C2"), ("C3", "C4")),
        FakeEvaluator(instance, maximum_customers_per_route=3),
    )

    assert proposal.sequences is not None
    assert len(proposal.sequences) == 2
    assert sorted(name for route in proposal.sequences for name in route) == [
        "C1",
        "C2",
        "C3",
        "C4",
    ]
    assert any(event.status == "candidate_proposed" for event in proposal.events)
    assert all(event.candidate_vehicle_delta in (None, 0) for event in proposal.events)


def test_cross_route_search_reuses_precomputed_unchanged_routes() -> None:
    instance = _instance()
    sequences = (("C1", "C2"), ("C3", "C4"))
    evaluator = FakeEvaluator(instance, maximum_customers_per_route=3)
    precomputed = {sequence: evaluator.route(sequence) for sequence in sequences}
    calls_before_proposal = evaluator.calls

    proposal = propose_relocate(
        instance,
        sequences,
        evaluator,
        config=VehicleOperatorConfig(relocate_exact_evaluation_budget=2),
        precomputed_routes=precomputed,
    )

    assert proposal.sequences is not None
    assert evaluator.calls - calls_before_proposal <= 2


def test_swap_keeps_route_count_and_records_affected_routes() -> None:
    instance = _instance()
    proposal = propose_swap(
        instance,
        (("C1", "C2"), ("C3", "C4")),
        FakeEvaluator(instance, maximum_customers_per_route=2),
    )

    assert proposal.sequences is not None
    assert len(proposal.sequences) == 2
    assert any(event.status == "candidate_proposed" for event in proposal.events)
    assert any(event.affected_route_indices == (0, 1) for event in proposal.events)


def test_two_opt_star_exchanges_tails_only_after_prefilters() -> None:
    instance = _instance()
    proposal = propose_two_opt_star(
        instance,
        (("C1", "C2"), ("C3", "C4")),
        FakeEvaluator(instance, maximum_customers_per_route=2),
    )

    assert proposal.sequences is not None
    assert len(proposal.sequences) == 2
    assert any(event.operator == "two_opt_star" for event in proposal.events)
    assert any(event.candidate_route_sequences for event in proposal.events)


def test_route_segment_destroy_repairs_into_existing_routes() -> None:
    instance = _instance()
    proposal = propose_route_segment_destroy(
        instance,
        (("C1", "C2", "C3"), ("C4",)),
        FakeEvaluator(instance, maximum_customers_per_route=4),
        config=VehicleOperatorConfig(
            route_segment_min_length=2,
            route_segment_max_length=2,
        ),
    )

    assert proposal.sequences is not None
    assert len(proposal.sequences) == 2
    assert any(event.status == "candidate_proposed" for event in proposal.events)
    assert all(event.new_routes_created == 0 for event in proposal.events)


def test_route_segment_destroy_rejects_noop_repair(monkeypatch) -> None:
    instance = _instance()

    def return_original_sequence(*args, **kwargs) -> RepairResult:
        return RepairResult((("C1", "C2", "C3"),), 0, 0, "")

    monkeypatch.setattr(
        neighborhoods,
        "repair_vehicle_count_aware",
        return_original_sequence,
    )
    proposal = propose_route_segment_destroy(
        instance,
        (("C1", "C2", "C3"),),
        FakeEvaluator(instance, maximum_customers_per_route=4),
        config=VehicleOperatorConfig(
            route_segment_min_length=2,
            route_segment_max_length=2,
        ),
    )

    assert proposal.sequences is None
    assert any(event.reason == "route_segment_no_change" for event in proposal.events)


def test_ejection_chain_is_bounded_and_can_complete_a_capacity_blocked_relocate() -> None:
    instance = _instance()
    proposal = propose_ejection_chain(
        instance,
        (("C1", "C2"), ("C3", "C4")),
        FakeEvaluator(instance, maximum_customers_per_route=2),
        config=VehicleOperatorConfig(
            ejection_chain_max_depth=3,
            ejection_chain_beam_width=16,
            ejection_chain_exact_evaluation_budget=24,
        ),
    )

    assert proposal.sequences is not None
    assert len(proposal.sequences) == 2
    assert any(event.status == "candidate_proposed" for event in proposal.events)
    assert max(event.chain_depth for event in proposal.events) <= 3
    assert sorted(name for route in proposal.sequences for name in route) == [
        "C1",
        "C2",
        "C3",
        "C4",
    ]


def test_dynamic_removal_size_is_deterministic_bounded_and_escalates() -> None:
    config = VehicleOperatorConfig(
        small_removal_min_fraction=0.05,
        small_removal_max_fraction=0.10,
        medium_removal_min_fraction=0.10,
        medium_removal_max_fraction=0.20,
        large_removal_min_fraction=0.20,
        large_removal_max_fraction=0.35,
        medium_stagnation_threshold=4,
        large_stagnation_threshold=8,
        exploration_period=3,
    )

    first = select_dynamic_removal_size(100, 0, 0, config=config)
    medium = select_dynamic_removal_size(100, 4, 1, config=config)
    large = select_dynamic_removal_size(100, 5, 6, config=config)
    stagnated = select_dynamic_removal_size(100, 8, 1, config=config)

    assert first.tier is RemovalTier.SMALL
    assert first.requested_count == 5
    assert medium.tier is RemovalTier.MEDIUM
    assert medium.requested_count == 10
    assert large.tier is RemovalTier.LARGE
    assert large.requested_count == 20
    assert stagnated.tier is RemovalTier.LARGE
    assert all(1 <= item.requested_count <= 99 for item in (first, medium, large, stagnated))
    assert first == select_dynamic_removal_size(100, 0, 0, config=config)
    assert select_dynamic_removal_size(100, 0, 6, config=config).tier is RemovalTier.SMALL


def test_constraint_removal_public_seam_covers_all_operators_deterministically() -> None:
    instance = _instance()
    sequences = (("C1", "C2"), ("C3", "C4"))
    config = VehicleOperatorConfig()
    selection = select_dynamic_removal_size(4, 0, 0, config=config)
    expected_customers = sorted(name for route in sequences for name in route)

    for operator in ConstraintRemovalOperator:
        first = propose_constraint_removal(
            instance,
            sequences,
            FakeEvaluator(instance),
            operator=operator,
            selection=selection,
            seed=2014,
        )
        second = propose_constraint_removal(
            instance,
            sequences,
            FakeEvaluator(instance),
            operator=operator,
            selection=selection,
            seed=2014,
        )

        assert first.partial is not None
        assert first.removed_customers
        assert len(first.removed_customers) == selection.requested_count
        assert sorted(first.removed_customers) == sorted(
            name
            for name in expected_customers
            if name not in {item for route in first.partial for item in route}
        )
        assert first.removed_customers == second.removed_customers
        assert first.events[-1].operator == operator.value
        assert first.events[-1].removal_tier == RemovalTier.SMALL.value
        assert first.events[-1].removal_size_actual == len(first.removed_customers)


def test_constraint_removal_uses_precomputed_route_evaluations_when_supplied() -> None:
    instance = _instance()
    sequences = (("C1", "C2"), ("C3", "C4"))
    evaluator = FakeEvaluator(instance)
    precomputed = {
        sequence: evaluator.route(sequence)
        for sequence in sequences
    }
    evaluator.calls = 0
    proposal = propose_constraint_removal(
        instance,
        sequences,
        evaluator,
        operator=ConstraintRemovalOperator.TIME_WINDOW_CONFLICT,
        selection=select_dynamic_removal_size(4, 0, 0),
        precomputed_routes=precomputed,
    )

    assert proposal.partial is not None
    assert evaluator.calls == 0
    assert all(event.exact_route_evaluations == 0 for event in proposal.events)


def test_constraint_removal_repair_inserts_into_existing_routes_only() -> None:
    instance = _instance()
    evaluator = FakeEvaluator(instance)
    original = (("C1", "C2"), ("C3", "C4"))
    selection = select_dynamic_removal_size(4, 0, 0)
    proposal = propose_constraint_removal(
        instance,
        original,
        evaluator,
        operator=ConstraintRemovalOperator.STATION_PRESSURE,
        selection=selection,
    )

    assert proposal.partial is not None
    before = evaluator.calls
    repair = repair_constraint_removal(
        original,
        proposal.partial,
        proposal.removed_customers,
        evaluator,
        instance,
    )

    assert repair.sequences is not None
    assert repair.sequences != original
    assert sorted(name for route in repair.sequences for name in route) == [
        "C1",
        "C2",
        "C3",
        "C4",
    ]
    assert repair.new_routes_created == 0
    assert repair.exact_route_evaluations == evaluator.calls - before
