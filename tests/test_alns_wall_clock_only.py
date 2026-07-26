from __future__ import annotations

import time

import pytest

import evrptw.alns as alns_module
from evrptw.alns import ExactDeadlineConfig, solve_alns
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.stage04 import Stage04Config


class _SteppedClock:
    def __init__(self, step: float) -> None:
        self._now = -step
        self._step = step

    def perf_counter(self) -> float:
        self._now += self._step
        return self._now


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
    monkeypatch.setattr(alns_module.time, "perf_counter", lambda: clock["now"])

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
    )

    assert result.accepted_moves == 0
    assert result.rejected_moves == 1
    assert result.termination_reason == "wall_clock_deadline"


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
