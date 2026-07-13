from __future__ import annotations

from dataclasses import dataclass, replace

from evrptw.charging import ChargingSubproblemResult
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.neighborhoods import (
    VehicleOperatorConfig,
    propose_ejection_chain,
    propose_relocate,
    propose_route_elimination,
    propose_route_merge,
    propose_route_segment_destroy,
    propose_swap,
    propose_two_opt_star,
    repair_vehicle_count_aware,
    screen_route_candidate,
)


@dataclass
class FakeEvaluator:
    instance: Instance
    maximum_customers_per_route: int = 2
    calls: int = 0

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
    evaluator = FakeEvaluator(instance)
    sequences = (("C1",), ("C2",))

    before = evaluator.calls
    proposal = propose_route_merge(instance, sequences, evaluator)

    assert proposal.sequences is None
    assert evaluator.calls == before + len(sequences)
    assert any(event.reason == "capacity_prefilter" for event in proposal.events)


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
