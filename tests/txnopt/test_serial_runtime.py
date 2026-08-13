from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pytest

from txnopt import RunConfig
from txnopt.runtime import RuntimeContractError, SerialTxnRuntime


class IncrementKernel:
    def __init__(self) -> None:
        self.tapes: list[tuple[int, ...]] = []

    def propose(
        self,
        snapshot: int,
        *,
        round_id: int,
        random_tape: Sequence[int],
    ) -> Sequence[int]:
        self.tapes.append(tuple(random_tape))
        return (snapshot + 1,)

    def decide(
        self,
        snapshot: int,
        candidates: Sequence[int],
        evaluated_states: Sequence[int],
        *,
        round_id: int,
    ) -> int:
        return evaluated_states[0]


class IntegerOracle:
    deterministic = True

    def __init__(self) -> None:
        self.evaluated_batches: list[tuple[int, ...]] = []

    def stable_key(self, candidate: int) -> str:
        return f"integer:{candidate}"

    def state_digest(self, state: int) -> str:
        return hashlib.sha256(str(state).encode()).hexdigest()

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
        self.evaluated_batches.append(tuple(candidates))
        return tuple(candidates)

    def validate(self, state: int) -> None:
        if state < 0:
            raise ValueError("negative state")

    def objective(self, state: int) -> int:
        return state


def test_serial_runtime_is_deterministic_and_returns_last_fixed_work_prefix() -> None:
    config = RunConfig(
        seed=2014,
        workers=1,
        execution_mode="serial",
        fixed_work=3,
        max_rounds=10,
    )
    first_kernel = IncrementKernel()
    first = SerialTxnRuntime[int, int, int]().run(
        0,
        kernel=first_kernel,
        oracle=IntegerOracle(),
        config=config,
    )
    second_kernel = IncrementKernel()
    second = SerialTxnRuntime[int, int, int]().run(
        0,
        kernel=second_kernel,
        oracle=IntegerOracle(),
        config=config,
    )
    assert first.last_committed_state == 3
    assert first.objective == 3
    assert first.termination_reason == "fixed_work_exhausted"
    assert first.semantic_digest == second.semantic_digest
    assert first_kernel.tapes == second_kernel.tapes


def test_serial_runtime_discards_a_batch_that_crosses_deadline() -> None:
    ticks = iter((0, 1, 200))
    runtime = SerialTxnRuntime[int, int, int](clock_ns=lambda: next(ticks))
    result = runtime.run(
        0,
        kernel=IncrementKernel(),
        oracle=IntegerOracle(),
        config=RunConfig(
            seed=0,
            workers=1,
            execution_mode="serial",
            deadline_seconds=0.0000001,
            max_rounds=10,
        ),
    )
    assert result.last_committed_state == 0
    assert result.objective == 0
    assert result.termination_reason == "deadline"


def test_serial_runtime_commits_cache_only_with_completed_round() -> None:
    class RepeatKernel(IncrementKernel):
        def propose(
            self,
            snapshot: int,
            *,
            round_id: int,
            random_tape: Sequence[int],
        ) -> Sequence[int]:
            return (1, 1)

    oracle = IntegerOracle()
    result = SerialTxnRuntime[int, int, int]().run(
        0,
        kernel=RepeatKernel(),
        oracle=oracle,
        config=RunConfig(
            seed=0,
            workers=1,
            execution_mode="serial",
            fixed_work=2,
            max_rounds=2,
        ),
    )
    assert result.last_committed_state == 1
    assert oracle.evaluated_batches == [(1,)]
    assert result.termination_reason == "max_rounds"


def test_serial_runtime_rejects_undeclared_determinism() -> None:
    oracle = IntegerOracle()
    oracle.deterministic = False
    with pytest.raises(RuntimeContractError, match="deterministic"):
        SerialTxnRuntime[int, int, int]().run(
            0,
            kernel=IncrementKernel(),
            oracle=oracle,
            config=RunConfig(
                seed=0,
                workers=1,
                execution_mode="serial",
                fixed_work=1,
            ),
        )


def test_serial_runtime_rejects_unvalidated_kernel_state() -> None:
    class InvalidKernel(IncrementKernel):
        def decide(
            self,
            snapshot: int,
            candidates: Sequence[int],
            evaluated_states: Sequence[int],
            *,
            round_id: int,
        ) -> int:
            return 999

    with pytest.raises(RuntimeContractError, match="neither"):
        SerialTxnRuntime[int, int, int]().run(
            0,
            kernel=InvalidKernel(),
            oracle=IntegerOracle(),
            config=RunConfig(
                seed=0,
                workers=1,
                execution_mode="serial",
                fixed_work=1,
            ),
        )
