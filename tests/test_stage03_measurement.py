from __future__ import annotations

from types import SimpleNamespace

import pytest

import evrptw.alns as alns_module
from evrptw.alns import MeasurementConfig, Stage03ExecutionError, Stage03Trace, solve_alns
from evrptw.cache_incremental import CacheIncrementalConfig
from evrptw.exact_deadline import ExactDeadlineConfig
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
    payload = trace.to_dict()
    assert payload["route_dictionary"] == {
        trace.route_evaluations[0].route_key: ["C1", "C2"]
    }
    assert "runtime_semantic_events" not in payload
    assert trace.runtime_semantic_events == ()
    assert Stage03Trace.from_dict(payload).runtime_semantic_events == ()


def test_runtime_semantic_journal_is_explicit_and_round_trips() -> None:
    trace = Stage03Trace(
        MeasurementConfig(record_runtime_semantic_events=True)
    )
    trace.record_route_evaluation(
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

    payload = trace.to_dict()
    assert payload["config"]["record_runtime_semantic_events"] is True
    assert payload["runtime_semantic_events"] == [
        {
            "event_type": "exact_route_result",
            "evaluation_id": 1,
            "route_key": trace.route_evaluations[0].route_key,
            "lane": "legacy",
            "iteration": 3,
            "operator": "relocate",
            "exact_started": True,
            "exact_completed": True,
            "feasible": True,
            "failure_reason": "",
            "deadline_boundary": "",
            "status": "completed_feasible",
            "semantic_stream": "exact_result",
            "semantic_event_id": 1,
            "runtime_causal_event_id": 1,
        }
    ]
    assert Stage03Trace.from_dict(payload).runtime_semantic_events == (
        payload["runtime_semantic_events"][0],
    )


def test_runtime_semantic_journal_rolls_back_atomically() -> None:
    trace = Stage03Trace(
        MeasurementConfig(record_runtime_semantic_events=True)
    )
    trace.record_runtime_semantic_event("operator", {"event_type": "before"})
    checkpoint = trace.snapshot_runtime_semantic_journal()
    trace.record_runtime_semantic_event("exact_work", {"event_type": "work"})
    trace.record_runtime_semantic_event("exact_result", {"event_type": "result"})

    trace.rollback_runtime_semantic_journal(checkpoint)

    assert trace.runtime_semantic_events == (
        {
            "event_type": "before",
            "semantic_stream": "operator",
            "semantic_event_id": 1,
            "runtime_causal_event_id": 1,
        },
    )
    assert trace.record_runtime_semantic_event(
        "deadline", {"event_type": "deadline"}
    ) == 2


def test_runtime_semantic_import_preserves_external_causal_ids_atomically() -> None:
    trace = Stage03Trace(
        MeasurementConfig(record_runtime_semantic_events=True)
    )
    trace.record_runtime_semantic_event("operator", {"event_type": "old"})

    trace.import_runtime_semantic_journal(
        (
            (1, "operator", {"event_type": "first"}),
            (2, "exact_work", {"event_type": "second"}),
        )
    )

    assert [
        event["runtime_causal_event_id"]
        for event in trace.runtime_semantic_events
    ] == [1, 2]
    assert [
        event["semantic_event_id"] for event in trace.runtime_semantic_events
    ] == [1, 2]

    before = trace.runtime_semantic_events
    with pytest.raises(
        ValueError,
        match="unique and contiguous",
    ):
        trace.import_runtime_semantic_journal(
            (
                (1, "operator", {"event_type": "first"}),
                (3, "exact_work", {"event_type": "gap"}),
            )
        )
    assert trace.runtime_semantic_events == before


def test_runtime_semantic_journal_rejects_hidden_payload() -> None:
    payload = Stage03Trace(MeasurementConfig()).to_dict()
    payload["runtime_semantic_events"] = [
        {
            "event_type": "forged",
            "semantic_stream": "operator",
            "semantic_event_id": 1,
        }
    ]

    with pytest.raises(
        ValueError,
        match="require their explicit measurement flag",
    ):
        Stage03Trace.from_dict(payload)


def test_runtime_semantic_journal_rejects_corrupt_persisted_causal_ids() -> None:
    trace = Stage03Trace(
        MeasurementConfig(record_runtime_semantic_events=True)
    )
    trace.record_runtime_semantic_event("operator", {"event_type": "first"})
    trace.record_runtime_semantic_event("exact_work", {"event_type": "second"})
    payload = trace.to_dict()
    runtime_events = payload["runtime_semantic_events"]
    assert isinstance(runtime_events, list)
    runtime_events[1]["runtime_causal_event_id"] = 3

    with pytest.raises(
        ValueError,
        match="unique and contiguous",
    ):
        Stage03Trace.from_dict(payload)


@pytest.mark.parametrize(
    ("declared_semantics", "expected_unique", "expected_status"),
    [
        (None, 2, "pass"),
        ("completed_cache_owner_identity_v2", 1, "pass"),
        ("completed_cache_owner_identity_v2", 2, "fail"),
        ("forged_semantics", 1, "fail"),
    ],
)
def test_trace_reconcile_preserves_legacy_interrupted_unique_semantics(
    declared_semantics: str | None,
    expected_unique: int,
    expected_status: str,
) -> None:
    trace = Stage03Trace(
        MeasurementConfig(),
        cache_incremental_config=CacheIncrementalConfig(enabled=True),
        exact_deadline_config=ExactDeadlineConfig.wall_clock(),
    )
    trace.record_route_evaluation(
        ("C1",),
        lane="legacy",
        iteration=1,
        operator="relocate",
        kind="exact_call",
        exact_started=True,
        exact_completed=True,
        feasible=True,
        failure_reason="",
        cache_key_digest="completed-digest",
    )
    trace.record_route_evaluation(
        ("C2",),
        lane="legacy",
        iteration=1,
        operator="relocate",
        kind="exact_call",
        exact_started=True,
        exact_completed=False,
        feasible=None,
        failure_reason="ExactBatchDeadlineExceeded",
        status="interrupted_deadline",
    )
    result_fields = {
        "charging_subproblem_calls": 1,
        "exact_started_calls": 2,
        "cache_hits": 0,
        "unique_route_evaluations": expected_unique,
        "destroy_statistics": {},
        "repair_statistics": {},
        "neighborhood_statistics": {},
        "effective_iterations": 0,
        "accepted_moves": 0,
        "rejected_moves": 0,
        "improving_moves": 0,
        "screening_statistics": {},
        "cache_incremental_statistics": {},
    }
    if declared_semantics is not None:
        result_fields["unique_route_semantics"] = declared_semantics
    result = SimpleNamespace(**result_fields)

    reconciliation = trace.reconcile(result)

    assert reconciliation["status"] == expected_status
    assert reconciliation["expected"]["unique_route_semantics"] == (
        declared_semantics or "started_lane_identity_legacy_v1"
    )


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
    assert (
        measured.objective.key
        == SolutionObjective(
            measured.vehicle_count,
            measured.objective.total_distance,
            measured.objective.total_charging_time,
            measured.objective.charging_count,
        ).key
    )


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
        assert event["candidate_full_route_keys"]
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
    assert any(event.get("event_type") == "execution_error" for event in caught.value.trace.events)
