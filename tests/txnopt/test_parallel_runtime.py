from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence

import pytest

from txnopt import RunConfig
from txnopt.runtime import OracleWorkerError, PythonTxnRuntime, RuntimeContractError


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


def test_ordered_failure_binds_t4_to_the_full_atomic_transaction_window() -> None:
    class WideKernel(CandidateKernel):
        admission_limit = 64

        def propose(
            self,
            snapshot: int,
            *,
            round_id: int,
            random_tape: Sequence[int],
        ) -> Sequence[int]:
            return tuple(range(1, 65))

    class LateFailureOracle(ParallelIntegerOracle):
        def evaluate_batch(
            self,
            candidates: Sequence[int],
            *,
            work_budget: int,
            deadline_ns: int | None,
        ) -> Sequence[int]:
            if candidates == (61,):
                raise OracleWorkerError("candidate 61 failed")
            return tuple(candidates)

    physical: list[tuple[Mapping[str, object], ...]] = []
    result = PythonTxnRuntime[int, int, int](physical_event_sink=physical.append).run(
        0,
        kernel=WideKernel(),
        oracle=LateFailureOracle(),
        config=RunConfig(
            seed=2014,
            workers=4,
            execution_mode="ordered",
            fixed_work=100,
            speculation_window=4,
            trace_policy="semantic_and_physical",
            max_rounds=1,
        ),
    )

    assert result.termination_reason == "worker_failure"
    observation = next(
        event for event in physical[0] if event["event"] == "t4_waste_observation"
    )
    assert observation["uncommitted_window"] == 64
    assert observation["observed_discarded_work_units"] > 4
    assert observation["discarded_work_bound_units"] == 64
    assert observation["bound_satisfied"] is True
