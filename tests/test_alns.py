from __future__ import annotations

from evrptw.alns import solve_alns
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.objective import SolutionObjective
from evrptw.validation import validate_routes


def _instance() -> Instance:
    return Instance(
        "alns_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("F1", NodeType.STATION, 5.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 8.0, 1.0, 1.0, 0.0, 100.0, 1.0),
            Node("C2", NodeType.CUSTOMER, 8.0, -1.0, 1.0, 0.0, 100.0, 1.0),
            Node("C3", NodeType.CUSTOMER, 2.0, 1.0, 1.0, 0.0, 100.0, 1.0),
        ),
        Vehicle(11.0, 3.0, 1.0, 0.1, 1.0),
    )


def test_alns_returns_reproducible_unified_validator_feasible_solution() -> None:
    instance = _instance()
    first = solve_alns(instance, seed=2014, max_iterations=60, time_limit_seconds=2.0)
    second = solve_alns(instance, seed=2014, max_iterations=60, time_limit_seconds=2.0)

    assert first.feasible is True
    assert first.customer_sequences == second.customer_sequences
    assert first.objective_value == second.objective_value
    assert first.vehicle_count >= 1
    assert first.charging_subproblem_calls > 0
    assert set(first.destroy_statistics) == {"random", "worst", "related"}
    assert set(first.repair_statistics) == {
        "greedy",
        "regret2",
        "energy",
        "vehicle_count_aware",
    }
    assert set(first.neighborhood_statistics) == {
        "standard",
        "vehicle_count_aware_repair",
        "route_elimination",
        "route_merge",
    }
    assert first.neighborhood_events
    assert all(
        {"iteration", "accepted", "vehicle_reduction", "distance_improvement"}
        <= set(event)
        for event in first.neighborhood_events
    )
    report = validate_routes(instance, [list(route) for route in first.routes])
    assert report.feasible
    assert first.objective is not None
    assert first.objective.key == SolutionObjective.from_report(instance, report).key
    assert first.objective_value == first.objective.total_distance


def test_alns_baseline_profile_preserves_historical_operator_surface() -> None:
    result = solve_alns(
        _instance(),
        seed=2014,
        max_iterations=10,
        time_limit_seconds=1.0,
        operator_profile="baseline",
    )

    assert result.feasible is True
    assert result.operator_profile == "baseline"
    assert result.neighborhood_statistics == {}
    assert result.neighborhood_events == ()
    assert set(result.repair_statistics) == {"greedy", "regret2", "energy"}
