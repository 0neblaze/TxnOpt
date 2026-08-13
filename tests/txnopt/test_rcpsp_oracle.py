from __future__ import annotations

from collections.abc import Sequence

from txnopt import RunConfig
from txnopt.runtime import SerialTxnRuntime
from txnopt_cases.rcpsp import (
    Activity,
    ActivityMode,
    RCPSPInstance,
    RCPSPOracle,
    RCPSPSchedule,
    RCPSPSearchKernel,
    RCPSPState,
    ScheduledActivity,
    validate_schedule,
)


def _instance() -> RCPSPInstance:
    zero = ActivityMode(0, (0,))
    work = ActivityMode(2, (2,))
    return RCPSPInstance(
        "tiny-resource-conflict",
        (
            Activity(0, (), (zero,)),
            Activity(1, (0,), (work,)),
            Activity(2, (0,), (work,)),
            Activity(3, (1, 2), (zero,)),
        ),
        (2,),
    )


def test_rcpsp_oracle_solves_and_independent_validator_rebuilds_schedule() -> None:
    oracle = RCPSPOracle(_instance(), seed=2014)
    state = RCPSPState((0, 1, 2, 3), (0, 0, 0, 0))
    assert oracle.screen((state,)) == (True,)
    assert oracle.lower_bound(state) == 2
    schedule = oracle.evaluate_batch((state,), work_budget=1, deadline_ns=None)[0]
    report = validate_schedule(_instance(), schedule)
    assert report.feasible is True
    assert schedule.makespan == report.makespan == 4
    assert oracle.objective(schedule) == 4
    assert len(oracle.stable_key(state)) == 64
    assert len(oracle.state_digest(schedule)) == 64


def test_rcpsp_safe_screen_rejects_precedence_and_mode_errors() -> None:
    oracle = RCPSPOracle(_instance(), seed=0)
    bad_order = RCPSPState((1, 0, 2, 3), (0, 0, 0, 0))
    bad_mode = RCPSPState((0, 1, 2, 3), (0, 0, 9, 0))
    assert oracle.screen((bad_order, bad_mode)) == (False, False)


def test_rcpsp_independent_validator_detects_resource_overlap() -> None:
    state = RCPSPState((0, 1, 2, 3), (0, 0, 0, 0))
    invalid = RCPSPSchedule(
        state=state,
        activities=(
            ScheduledActivity(0, 0, 0, 0),
            ScheduledActivity(1, 0, 0, 2),
            ScheduledActivity(2, 0, 0, 2),
            ScheduledActivity(3, 0, 2, 2),
        ),
        makespan=2,
    )
    report = validate_schedule(_instance(), invalid)
    assert report.feasible is False
    assert "resource 0 exceeds capacity at time 0" in report.violations


class _AlternateOrderKernel:
    def propose(
        self,
        snapshot: RCPSPSchedule,
        *,
        round_id: int,
        random_tape: Sequence[int],
    ) -> Sequence[RCPSPState]:
        return (RCPSPState((0, 2, 1, 3), (0, 0, 0, 0)),)

    def decide(
        self,
        snapshot: RCPSPSchedule,
        candidates: Sequence[RCPSPState],
        evaluated_states: Sequence[RCPSPSchedule],
        *,
        round_id: int,
    ) -> RCPSPSchedule:
        return evaluated_states[0]


def test_rcpsp_oracle_runs_through_the_generic_serial_contract() -> None:
    oracle = RCPSPOracle(_instance(), seed=2014)
    initial_state = RCPSPState((0, 1, 2, 3), (0, 0, 0, 0))
    initial_schedule = oracle.evaluate_batch(
        (initial_state,),
        work_budget=1,
        deadline_ns=None,
    )[0]
    config = RunConfig(
        seed=2014,
        workers=1,
        execution_mode="serial",
        fixed_work=1,
        max_rounds=1,
    )
    first = SerialTxnRuntime[RCPSPSchedule, RCPSPState, int]().run(
        initial_schedule,
        kernel=_AlternateOrderKernel(),
        oracle=oracle,
        config=config,
    )
    second = SerialTxnRuntime[RCPSPSchedule, RCPSPState, int]().run(
        initial_schedule,
        kernel=_AlternateOrderKernel(),
        oracle=oracle,
        config=config,
    )
    assert first.objective == 4
    assert first.termination_reason == "max_rounds"
    assert first.semantic_digest == second.semantic_digest


def test_rcpsp_kernel_generates_block_reinsertions_and_mode_changes() -> None:
    work_slow = ActivityMode(3, (2,))
    base = _instance()
    instance = RCPSPInstance(
        "tiny-multimode",
        (
            base.activities[0],
            Activity(1, (0,), (base.activities[1].modes[0], work_slow)),
            base.activities[2],
            base.activities[3],
        ),
        base.renewable_capacities,
    )
    oracle = RCPSPOracle(instance, seed=2014)
    initial_state = RCPSPState((0, 1, 2, 3), (0, 0, 0, 0))
    initial = oracle.evaluate_batch(
        (initial_state,),
        work_budget=1,
        deadline_ns=None,
    )[0]
    candidates = tuple(
        RCPSPSearchKernel(instance).propose(
            initial,
            round_id=0,
            random_tape=(7,),
        )
    )

    bounded = RCPSPSearchKernel(instance, max_candidates=4).propose(
        initial,
        round_id=0,
        random_tape=(7,),
    )

    assert len(bounded) <= 16
    assert RCPSPSearchKernel(instance, max_candidates=4).admission_limit == 4
    assert len(candidates) <= 64
    assert any(
        candidate.activity_order != initial_state.activity_order
        for candidate in candidates
    )
    assert any(candidate.mode_vector != initial_state.mode_vector for candidate in candidates)
    assert tuple(oracle.screen(candidates)).count(True) >= 1
