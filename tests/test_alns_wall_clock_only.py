from __future__ import annotations

import time
from collections.abc import Mapping

import pytest

import evrptw.alns as alns_module
from evrptw.alns import ExactDeadlineConfig, solve_alns
from evrptw.measurement import (
    MeasurementConfig,
    RouteEvaluationTrace,
    ScreeningDecision,
)
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.stage04 import Stage04Config
from evrptw.validation import SolutionReport, validate_routes


class _SteppedClock:
    def __init__(self, step: float) -> None:
        self._now = -step
        self._step = step

    def perf_counter(self) -> float:
        self._now += self._step
        return self._now


class _CollectingTraceSink:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def append_route_evaluation(self, record: RouteEvaluationTrace) -> None:
        del record

    def append_event(self, event: Mapping[str, object]) -> None:
        self.events.append(dict(event))

    def append_screening_decision(self, decision: ScreeningDecision) -> None:
        del decision

    def append_incremental_propagation(
        self,
        propagation: Mapping[str, object],
    ) -> None:
        del propagation


def _single_customer_instance() -> Instance:
    return Instance(
        "alns_wall_clock_only",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("F1", NodeType.STATION, 5.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 100.0, 1.0),
        ),
        Vehicle(11.0, 3.0, 1.0, 0.1, 1.0),
    )


def test_none_iteration_limit_runs_until_the_wall_clock_deadline() -> None:
    result = solve_alns(
        _single_customer_instance(),
        seed=2014,
        max_iterations=None,
        time_limit_seconds=0.02,
        operator_profile="baseline",
    )

    assert result.feasible is True
    assert result.iterations > 0
    assert result.runtime_seconds >= 0.02
    assert result.termination_reason == "wall_clock_deadline"


def test_none_iteration_limit_rejects_fixed_work_termination() -> None:
    with pytest.raises(
        ValueError,
        match="max_iterations=None requires wall_clock termination",
    ):
        solve_alns(
            _single_customer_instance(),
            seed=2014,
            max_iterations=None,
            time_limit_seconds=1.0,
            termination_mode="fixed_work",
            operator_profile="baseline",
        )


def test_none_iteration_limit_rejects_fixed_exact_call_budget() -> None:
    with pytest.raises(
        ValueError,
        match="max_iterations=None requires wall_clock termination",
    ):
        solve_alns(
            _single_customer_instance(),
            seed=2014,
            max_iterations=None,
            time_limit_seconds=1.0,
            exact_deadline_config=ExactDeadlineConfig.fixed_exact_calls(
                1,
                watchdog_seconds=1.0,
            ),
            operator_profile="baseline",
        )


def test_none_iteration_limit_cools_annealing_by_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _SteppedClock(0.0001)
    monkeypatch.setattr(time, "perf_counter", clock.perf_counter)

    result = solve_alns(
        _single_customer_instance(),
        seed=2014,
        max_iterations=None,
        time_limit_seconds=0.1,
        operator_profile="baseline",
        stage04_config=Stage04Config(
            auto_temperature=False,
            reheat_enabled=False,
            restart_enabled=False,
            intensification_enabled=False,
        ),
    )

    initial_temperature = result.stage04_temperature_history[0][1]
    final_recorded_temperature = result.stage04_temperature_history[-1][1]
    assert result.termination_reason == "wall_clock_deadline"
    assert result.stage04_temperature_history[-1][0] >= 10
    assert final_recorded_temperature <= initial_temperature * 0.1


def test_candidate_acceptance_cannot_commit_after_wall_clock_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "perf_counter", lambda: clock["now"])

    def cross_deadline_before_acceptance(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        clock["now"] = 2.0
        return True

    monkeypatch.setattr(
        alns_module,
        "accept_annealing_move",
        cross_deadline_before_acceptance,
    )

    result = solve_alns(
        _single_customer_instance(),
        seed=2014,
        max_iterations=1,
        time_limit_seconds=1.0,
        operator_profile="baseline",
        measurement_config=MeasurementConfig(),
    )

    assert result.accepted_moves == 0
    assert result.rejected_moves == 1
    assert result.termination_reason == "wall_clock_deadline"
    assert result.measurement_trace is not None
    assert result.measurement_trace.deadline_events == 2
    assert {
        str(event["lane"])
        for event in result.measurement_trace.events
        if event.get("event_type") == "deadline_boundary"
    } == {"legacy", "solver_finalization"}


def test_final_validation_deadline_emits_a_trace_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": time.perf_counter()}
    started_at = clock["now"]
    sink = _CollectingTraceSink()
    monkeypatch.setattr(time, "perf_counter", lambda: clock["now"])

    def cross_deadline_during_final_validation(
        instance: Instance,
        routes: list[list[str]],
        *,
        claimed_objective: float | None = None,
    ) -> SolutionReport:
        clock["now"] = started_at + 2.0
        return validate_routes(
            instance,
            routes,
            claimed_objective=claimed_objective,
        )

    monkeypatch.setattr(
        "evrptw.alns.validate_routes",
        cross_deadline_during_final_validation,
    )

    result = solve_alns(
        _single_customer_instance(),
        seed=2014,
        max_iterations=1,
        time_limit_seconds=1.0,
        operator_profile="baseline",
        measurement_config=MeasurementConfig(stream_sink=sink),
    )

    assert result.termination_reason == "wall_clock_deadline"
    assert result.measurement_trace is not None
    assert result.measurement_trace.deadline_events == 1
    deadline_events = [
        event for event in sink.events if event.get("event_type") == "deadline_boundary"
    ]
    assert len(deadline_events) == 1
    assert deadline_events[0]["timestamp_seconds"] == pytest.approx(2.0, abs=0.1)
    assert deadline_events[0] | {"timestamp_seconds": 2.0} == {
        "event_type": "deadline_boundary",
        "timestamp_seconds": 2.0,
        "lane": "solver_finalization",
        "iteration": 1,
        "operator": "termination",
        "boundary": "solver_termination",
        "reason": (
            "overall wall-clock deadline was confirmed after final solution "
            "validation"
        ),
        "route_keys": (),
        "exact_call_id": None,
    }


def test_integer_iteration_limit_preserves_iteration_bounded_behavior() -> None:
    result = solve_alns(
        _single_customer_instance(),
        seed=2014,
        max_iterations=1,
        time_limit_seconds=1.0,
        operator_profile="baseline",
    )

    assert result.iterations == 1
    assert result.termination_reason == "iteration_limit"
