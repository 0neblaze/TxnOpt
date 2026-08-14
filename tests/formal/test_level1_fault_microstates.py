from __future__ import annotations

import hashlib
import itertools
import threading
from collections.abc import Mapping, Sequence
from typing import Literal

import pytest

from txnopt import RunConfig, RunResult
from txnopt.runtime import OracleWorkerError, PythonTxnRuntime

_CANDIDATES = (1, 2, 3)
_COMPLETION_ORDERS = tuple(itertools.permutations(_CANDIDATES))


class _Kernel:
    def propose(
        self,
        snapshot: int,
        *,
        round_id: int,
        random_tape: Sequence[int],
    ) -> Sequence[int]:
        return _CANDIDATES

    def decide(
        self,
        snapshot: int,
        candidates: Sequence[int],
        evaluated_states: Sequence[int],
        *,
        round_id: int,
    ) -> int:
        assert tuple(candidates) == _CANDIDATES
        assert tuple(evaluated_states) == _CANDIDATES
        return evaluated_states[0]


class _Oracle:
    deterministic = True
    parallel_safe = True
    internal_parallelism = False

    def stable_key(self, candidate: int) -> str:
        return f"microstate:{candidate}"

    def state_digest(self, state: int) -> str:
        return hashlib.sha256(str(state).encode()).hexdigest()

    def work_units(self, candidate: int) -> int:
        return 1

    def screen(self, candidates: Sequence[int]) -> Sequence[bool]:
        return tuple(True for _candidate in candidates)

    def evaluate_batch(
        self,
        candidates: Sequence[int],
        *,
        work_budget: int,
        deadline_ns: int | None,
    ) -> Sequence[int]:
        assert work_budget == len(candidates)
        return tuple(candidates)

    def validate(self, state: int) -> None:
        if state < 0:
            raise ValueError("negative state")

    def objective(self, state: int) -> int:
        return state


class _CompletionController:
    def __init__(
        self,
        completion_order: tuple[int, ...],
        *,
        failing_candidate: int | None = None,
    ) -> None:
        self._completion_order = completion_order
        self._failing_candidate = failing_candidate
        self._started = {candidate: threading.Event() for candidate in _CANDIDATES}
        self._released = {candidate: threading.Event() for candidate in _CANDIDATES}
        self._finished = {candidate: threading.Event() for candidate in _CANDIDATES}
        self.observed_order: list[int] = []
        self.controller_error: BaseException | None = None

    def evaluate(self, candidate: int) -> int:
        self._started[candidate].set()
        if not self._released[candidate].wait(timeout=2.0):
            raise RuntimeError(f"candidate {candidate} was never released")
        try:
            if candidate == self._failing_candidate:
                raise OracleWorkerError(f"candidate {candidate} failed")
            return candidate
        finally:
            self.observed_order.append(candidate)
            self._finished[candidate].set()

    def release_in_order(self) -> None:
        try:
            for event in self._started.values():
                if not event.wait(timeout=2.0):
                    raise RuntimeError("parallel evaluator did not start every candidate")
            for candidate in self._completion_order:
                self._released[candidate].set()
                if not self._finished[candidate].wait(timeout=2.0):
                    raise RuntimeError(f"candidate {candidate} did not finish")
        except BaseException as error:
            self.controller_error = error
        finally:
            for event in self._released.values():
                event.set()


class _ControlledOracle(_Oracle):
    def __init__(self, controller: _CompletionController) -> None:
        self._controller = controller

    def evaluate_batch(
        self,
        candidates: Sequence[int],
        *,
        work_budget: int,
        deadline_ns: int | None,
    ) -> Sequence[int]:
        assert len(candidates) == 1
        assert work_budget == 1
        return (self._controller.evaluate(candidates[0]),)


def _parallel_config(mode: Literal["barrier", "ordered"]) -> RunConfig:
    return RunConfig(
        seed=2014,
        workers=3,
        execution_mode=mode,
        fixed_work=3,
        speculation_window=3 if mode == "ordered" else 0,
        trace_policy="semantic_and_physical",
        max_rounds=1,
    )


def _run_controlled(
    *,
    mode: Literal["barrier", "ordered"],
    completion_order: tuple[int, ...],
    failing_candidate: int | None = None,
) -> tuple[RunResult[int, int], _CompletionController, tuple[Mapping[str, object], ...]]:
    controller = _CompletionController(
        completion_order,
        failing_candidate=failing_candidate,
    )
    physical: list[tuple[Mapping[str, object], ...]] = []
    release_thread = threading.Thread(
        target=controller.release_in_order,
        name="txnopt-microstate-controller",
    )
    release_thread.start()
    try:
        result = PythonTxnRuntime[int, int, int](
            physical_event_sink=physical.append
        ).run(
            0,
            kernel=_Kernel(),
            oracle=_ControlledOracle(controller),
            config=_parallel_config(mode),
        )
    finally:
        release_thread.join(timeout=3.0)
    assert not release_thread.is_alive()
    if controller.controller_error is not None:
        raise controller.controller_error
    assert len(physical) == 1
    return result, controller, physical[0]


@pytest.mark.parametrize("mode", ("barrier", "ordered"))  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("completion_order", _COMPLETION_ORDERS)  # type: ignore[untyped-decorator]
def test_all_completion_orders_preserve_the_committed_trace(
    mode: Literal["barrier", "ordered"],
    completion_order: tuple[int, ...],
) -> None:
    serial = PythonTxnRuntime[int, int, int]().run(
        0,
        kernel=_Kernel(),
        oracle=_Oracle(),
        config=RunConfig(
            seed=2014,
            workers=1,
            execution_mode="serial",
            fixed_work=3,
            max_rounds=1,
        ),
    )
    result, controller, _physical = _run_controlled(
        mode=mode,
        completion_order=completion_order,
    )

    assert tuple(controller.observed_order) == completion_order
    assert result.last_committed_state == serial.last_committed_state == 1
    assert result.objective == serial.objective == 1
    assert result.semantic_digest == serial.semantic_digest


@pytest.mark.parametrize("mode", ("barrier", "ordered"))  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("completion_order", _COMPLETION_ORDERS)  # type: ignore[untyped-decorator]
@pytest.mark.parametrize("failing_candidate", _CANDIDATES)  # type: ignore[untyped-decorator]
def test_every_worker_failure_position_preserves_the_last_committed_prefix(
    mode: Literal["barrier", "ordered"],
    completion_order: tuple[int, ...],
    failing_candidate: int,
) -> None:
    result, controller, physical = _run_controlled(
        mode=mode,
        completion_order=completion_order,
        failing_candidate=failing_candidate,
    )

    assert tuple(controller.observed_order) == completion_order
    assert result.termination_reason == "worker_failure"
    assert result.last_committed_state == 0
    assert result.objective == 0
    waste = next(event for event in physical if event["event"] == "t4_waste_observation")
    assert waste["bound_satisfied"] is True


class _DeadlineClock:
    def __init__(self, *, expire_on_call: int) -> None:
        self._expire_on_call = expire_on_call
        self.calls = 0

    def __call__(self) -> int:
        self.calls += 1
        return 1 if self.calls >= self._expire_on_call else 0


@pytest.mark.parametrize("expire_on_call", (2, 3))  # type: ignore[untyped-decorator]
def test_every_runtime_deadline_checkpoint_returns_the_last_committed_prefix(
    expire_on_call: int,
) -> None:
    clock = _DeadlineClock(expire_on_call=expire_on_call)
    result = PythonTxnRuntime[int, int, int](clock_ns=clock).run(
        0,
        kernel=_Kernel(),
        oracle=_Oracle(),
        config=RunConfig(
            seed=2014,
            workers=1,
            execution_mode="serial",
            deadline_seconds=0.000000001,
            max_rounds=1,
        ),
    )

    assert clock.calls >= expire_on_call
    assert result.termination_reason == "deadline"
    assert result.last_committed_state == 0
    assert result.objective == 0
