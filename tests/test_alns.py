from __future__ import annotations

from evrptw.alns import _annotated_event_record, _EvaluatedSolution, solve_alns
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.neighborhoods import NeighborhoodEvent
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
    assert set(first.neighborhood_statistics) >= {
        "standard",
        "vehicle_count_aware_repair",
        "route_elimination",
        "route_merge",
        "relocate",
        "swap",
        "two_opt_star",
        "route_segment_destroy",
        "ejection_chain",
        "station_pressure",
        "time_window_conflict",
        "worst_energy_detour",
        "shaw_related",
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


def test_only_selected_feasible_probe_is_marked_accepted() -> None:
    candidate = _EvaluatedSolution(
        (),
        (),
        True,
        SolutionObjective(1, 10.0, 0.0, 0),
    )
    probe = NeighborhoodEvent(
        "relocate",
        "feasible_candidate",
        "probe",
        candidate_feasible=True,
    )
    selected = NeighborhoodEvent(
        "relocate",
        "candidate_proposed",
        "selected",
        candidate_feasible=True,
    )

    probe_record = _annotated_event_record(
        probe,
        iteration=1,
        accepted=True,
        vehicle_reduction=False,
        distance_improvement=True,
        candidate=candidate,
    )
    selected_record = _annotated_event_record(
        selected,
        iteration=1,
        accepted=True,
        vehicle_reduction=False,
        distance_improvement=True,
        candidate=candidate,
    )

    assert probe_record["accepted"] is False
    assert selected_record["accepted"] is True


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


def test_alns_stage02_route_reduction_profile_remains_explicitly_reproducible() -> None:
    result = solve_alns(
        _instance(),
        seed=2014,
        max_iterations=10,
        time_limit_seconds=1.0,
        operator_profile="stage02_route_reduction",
    )

    assert result.feasible is True
    assert set(result.neighborhood_statistics) == {
        "standard",
        "vehicle_count_aware_repair",
        "route_elimination",
        "route_merge",
    }


def test_alns_stage02_constraint_guided_profile_records_constraint_lane_and_cache_metrics() -> None:
    result = solve_alns(
        _instance(),
        seed=2014,
        max_iterations=20,
        time_limit_seconds=1.0,
        operator_profile="stage02_constraint_guided",
    )

    assert result.feasible is True
    assert result.operator_profile == "stage02_constraint_guided"
    assert set(result.constraint_operator_statistics) == {
        "station_pressure",
        "time_window_conflict",
        "worst_energy_detour",
        "shaw_related",
    }
    assert result.cache_hits >= 0
    assert result.cache_misses == result.unique_route_evaluations
    assert 0 <= result.effective_iterations <= result.iterations
    assert set(result.removal_tier_counts) >= {"small", "medium"}
    assert result.maximum_stagnation >= 0
    constraint_events = [
        event
        for event in result.neighborhood_events
        if event.get("track") == "constraint_lane"
    ]
    assert constraint_events
    assert {
        "station_pressure",
        "time_window_conflict",
        "worst_energy_detour",
        "shaw_related",
    } <= {str(event["operator"]) for event in constraint_events}
    assert all(
        int(event["removal_size_actual"]) <= int(event["removal_size_requested"])
        for event in constraint_events
        if int(event["removal_size_requested"])
    )
