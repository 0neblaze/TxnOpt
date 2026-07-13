from __future__ import annotations

import pytest

import evrptw.alns as alns_module
from evrptw.alns import MeasurementConfig, Stage03ExecutionError, Stage03Trace, solve_alns
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.neighborhoods import OperatorProfile
from evrptw.objective import SolutionObjective


def _instance() -> Instance:
    return Instance(
        "stage03_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("F1", NodeType.STATION, 5.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 8.0, 1.0, 1.0, 0.0, 100.0, 1.0),
            Node("C2", NodeType.CUSTOMER, 8.0, -1.0, 1.0, 0.0, 100.0, 1.0),
            Node("C3", NodeType.CUSTOMER, 2.0, 1.0, 1.0, 0.0, 100.0, 1.0),
        ),
        Vehicle(11.0, 3.0, 1.0, 0.1, 1.0),
    )


def test_stage03_trace_keeps_canonical_route_dictionary_and_call_states() -> None:
    trace = Stage03Trace(MeasurementConfig())

    exact_id = trace.record_route_evaluation(
        ("C1", "C2"),
        lane="legacy",
        iteration=3,
        operator="relocate",
        kind="exact_call",
        started_at=0.1,
        completed_at=0.4,
        exact_started=True,
        exact_completed=True,
        feasible=True,
        failure_reason="",
        labels_generated=4,
        labels_expanded=3,
        labels_pruned=1,
    )
    cache_id = trace.record_route_evaluation(
        ("C1", "C2"),
        lane="legacy",
        iteration=3,
        operator="relocate",
        kind="cache_hit",
        started_at=0.5,
        completed_at=0.5,
        exact_started=False,
        exact_completed=False,
        feasible=True,
        failure_reason="",
    )

    assert exact_id == 1
    assert cache_id == 2
    assert len(trace.route_dictionary) == 1
    assert trace.started_calls == 1
    assert trace.completed_calls == 1
    assert trace.exact_calls == 1
    assert trace.cache_hits == 1
    assert trace.route_evaluations[0].route_key == trace.route_evaluations[1].route_key
    assert trace.to_dict()["route_dictionary"] == {
        trace.route_evaluations[0].route_key: ["C1", "C2"]
    }


def test_stage03_trace_records_precomputed_routes_and_deadline_boundaries() -> None:
    trace = Stage03Trace(MeasurementConfig())
    trace.record_route_evaluation(
        ("C1",),
        lane="quality_shadow",
        iteration=1,
        operator="relocate",
        kind="precomputed_route",
        started_at=0.0,
        completed_at=0.0,
        exact_started=False,
        exact_completed=False,
        feasible=True,
        failure_reason="",
    )
    trace.record_deadline_boundary(
        lane="legacy",
        iteration=2,
        operator="route_merge",
        boundary="after_exact_call",
        route_sequence=("C1",),
        exact_call_id=1,
    )

    assert trace.precomputed_routes == 1
    assert trace.deadline_events == 1
    assert trace.events[-1]["event_type"] == "deadline_boundary"
    assert trace.events[-1]["boundary"] == "after_exact_call"


def test_measurement_is_opt_in_and_reconciles_exact_calls() -> None:
    instance = _instance()
    disabled = solve_alns(
        instance,
        seed=2014,
        max_iterations=8,
        time_limit_seconds=1.0,
        operator_profile=OperatorProfile.BASELINE,
    )
    measured = solve_alns(
        instance,
        seed=2014,
        max_iterations=8,
        time_limit_seconds=1.0,
        operator_profile=OperatorProfile.STAGE02_CONSTRAINT_GUIDED,
        measurement_config=MeasurementConfig(),
    )

    assert disabled.measurement_trace is None
    assert measured.measurement_trace is not None
    trace = measured.measurement_trace
    assert trace.started_calls >= trace.completed_calls
    assert trace.completed_calls == measured.charging_subproblem_calls
    assert trace.exact_calls == measured.charging_subproblem_calls
    assert trace.cache_hits == measured.cache_hits
    assert trace.reconcile(measured)["status"] == "pass"
    assert measured.objective is not None
    assert measured.objective.key == SolutionObjective(
        measured.vehicle_count,
        measured.objective.total_distance,
        measured.objective.total_charging_time,
        measured.objective.charging_count,
    ).key


def test_accepted_stage03_candidate_has_complete_vehicle_first_state() -> None:
    result = solve_alns(
        _instance(),
        seed=2014,
        max_iterations=12,
        time_limit_seconds=1.0,
        operator_profile=OperatorProfile.STAGE02_CONSTRAINT_GUIDED,
        measurement_config=MeasurementConfig(),
    )

    assert result.measurement_trace is not None
    accepted = [
        event
        for event in result.measurement_trace.events
        if event.get("event_type") == "candidate_state" and event.get("accepted") is True
    ]
    assert accepted
    for event in accepted:
        assert event["candidate_route_keys"]
        assert event["candidate_objective_key"]
        assert event["candidate_vehicle_count"] <= event["current_vehicle_count"]


def test_measured_execution_failure_keeps_partial_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("synthetic exact failure")

    monkeypatch.setattr(alns_module, "_solve_alns", fail)

    with pytest.raises(Stage03ExecutionError) as caught:
        solve_alns(
            _instance(),
            seed=2014,
            max_iterations=1,
            time_limit_seconds=1.0,
            operator_profile=OperatorProfile.BASELINE,
            measurement_config=MeasurementConfig(),
        )

    assert caught.value.trace.finished_at is not None
    assert any(
        event.get("event_type") == "execution_error"
        for event in caught.value.trace.events
    )
