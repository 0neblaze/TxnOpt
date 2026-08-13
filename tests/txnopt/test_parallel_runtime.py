from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence

import pytest

from txnopt import RunConfig
from txnopt.runtime import PythonTxnRuntime, RuntimeContractError


class CandidateKernel:
    def propose(
        self,
        snapshot: int,
        *,
        round_id: int,
        random_tape: Sequence[int],
    ) -> Sequence[int]:
        return (snapshot + 1, snapshot + 2, snapshot + 3, snapshot + 4)

    def decide(
        self,
        snapshot: int,
        candidates: Sequence[int],
        evaluated_states: Sequence[int],
        *,
        round_id: int,
    ) -> int:
        assert tuple(evaluated_states) == tuple(candidates)
        return evaluated_states[0]


class ParallelIntegerOracle:
    deterministic = True
    parallel_safe = True
    internal_parallelism = False

    def stable_key(self, candidate: int) -> str:
        return f"parallel-integer:{candidate}"

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
        if len(candidates) == 1:
            time.sleep((5 - candidates[0]) * 0.001)
        return tuple(candidates)

    def validate(self, state: int) -> None:
        if state < 0:
            raise ValueError("negative state")

    def objective(self, state: int) -> int:
        return state


def _config(mode: str) -> RunConfig:
    return RunConfig(
        seed=2014,
        workers=1 if mode == "serial" else 4,
        execution_mode=mode,  # type: ignore[arg-type]
        fixed_work=4,
        speculation_window=4 if mode == "ordered" else 0,
        max_rounds=1,
    )


def test_all_execution_modes_refine_to_the_same_committed_trace() -> None:
    results = {
        mode: PythonTxnRuntime[int, int, int]().run(
            0,
            kernel=CandidateKernel(),
            oracle=ParallelIntegerOracle(),
            config=_config(mode),
        )
        for mode in ("serial", "barrier", "ordered")
    }

    assert {result.last_committed_state for result in results.values()} == {1}
    assert {result.semantic_digest for result in results.values()} == {
        results["serial"].semantic_digest
    }


def test_parallel_modes_fail_closed_without_an_explicit_safety_declaration() -> None:
    oracle = ParallelIntegerOracle()
    oracle.parallel_safe = False

    with pytest.raises(RuntimeContractError, match="parallel-safe"):
        PythonTxnRuntime[int, int, int]().run(
            0,
            kernel=CandidateKernel(),
            oracle=oracle,
            config=_config("barrier"),
        )
