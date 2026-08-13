from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import pytest

from txnopt import RunConfig
from txnopt._internal.cache import CacheLookup, InMemoryCacheStore
from txnopt.runtime import OracleWorkerError, RuntimeContractError, SerialTxnRuntime


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
    parallel_safe = True
    internal_parallelism = False

    def __init__(self) -> None:
        self.evaluated_batches: list[tuple[int, ...]] = []

    def stable_key(self, candidate: int) -> str:
        return f"integer:{candidate}"

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


def test_worker_failure_returns_only_the_last_committed_prefix() -> None:
    class FailingOracle(IntegerOracle):
        def evaluate_batch(
            self,
            candidates: Sequence[int],
            *,
            work_budget: int,
            deadline_ns: int | None,
        ) -> Sequence[int]:
            if candidates == (2,):
                raise OracleWorkerError("simulated worker crash")
            return super().evaluate_batch(
                candidates,
                work_budget=work_budget,
                deadline_ns=deadline_ns,
            )

    cache = InMemoryCacheStore[int]()
    result = SerialTxnRuntime[int, int, int](cache_factory=lambda: cache).run(
        0,
        kernel=IncrementKernel(),
        oracle=FailingOracle(),
        config=RunConfig(
            seed=0,
            workers=1,
            execution_mode="serial",
            fixed_work=3,
        ),
    )

    assert result.last_committed_state == 1
    assert result.termination_reason == "worker_failure"
    assert cache.lookup(("integer:1", "integer:2")).values == (1, None)


def test_validation_failure_rolls_back_the_incomplete_round() -> None:
    class InvalidResultOracle(IntegerOracle):
        def evaluate_batch(
            self,
            candidates: Sequence[int],
            *,
            work_budget: int,
            deadline_ns: int | None,
        ) -> Sequence[int]:
            if candidates == (2,):
                return (-2,)
            return super().evaluate_batch(
                candidates,
                work_budget=work_budget,
                deadline_ns=deadline_ns,
            )

    cache = InMemoryCacheStore[int]()
    result = SerialTxnRuntime[int, int, int](cache_factory=lambda: cache).run(
        0,
        kernel=IncrementKernel(),
        oracle=InvalidResultOracle(),
        config=RunConfig(
            seed=0,
            workers=1,
            execution_mode="serial",
            fixed_work=3,
        ),
    )

    assert result.last_committed_state == 1
    assert result.termination_reason == "validation_failure"
    assert cache.lookup(("integer:1", "integer:2")).values == (1, None)


def test_cache_write_failure_rolls_back_state_and_cache() -> None:
    def reject(_entries: object) -> None:
        raise OSError("simulated cache failure")

    cache = InMemoryCacheStore[int](on_commit=reject)
    result = SerialTxnRuntime[int, int, int](cache_factory=lambda: cache).run(
        0,
        kernel=IncrementKernel(),
        oracle=IntegerOracle(),
        config=RunConfig(
            seed=0,
            workers=1,
            execution_mode="serial",
            fixed_work=1,
        ),
    )

    assert result.last_committed_state == 0
    assert result.termination_reason == "cache_write_failure"
    assert cache.snapshot() == (0, {})


def test_stale_cache_snapshot_interrupts_before_evaluation() -> None:
    class InterferingCache(InMemoryCacheStore[int]):
        def lookup(self, keys: Sequence[str]) -> CacheLookup[int]:
            external = self.begin()
            self.commit(external)
            return super().lookup(keys)

    oracle = IntegerOracle()
    result = SerialTxnRuntime[int, int, int](cache_factory=InterferingCache).run(
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

    assert result.last_committed_state == 0
    assert result.termination_reason == "stale_snapshot"
    assert oracle.evaluated_batches == []


def test_runtime_emits_a_detached_complete_semantic_stream() -> None:
    streams: list[tuple[Mapping[str, object], ...]] = []
    runtime = SerialTxnRuntime[int, int, int](semantic_event_sink=streams.append)
    result = runtime.run(
        0,
        kernel=IncrementKernel(),
        oracle=IntegerOracle(),
        config=RunConfig(
            seed=0,
            workers=1,
            execution_mode="serial",
            fixed_work=1,
            max_rounds=1,
        ),
    )

    assert len(streams) == 1
    assert streams[0][0]["event"] == "run_open"
    assert streams[0][-1] == {"event": "run_terminated", "reason": "max_rounds"}
    assert (
        hashlib.sha256(
            json.dumps(
                streams[0],
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        == result.semantic_digest
    )


def test_physical_timing_is_separate_from_the_semantic_digest() -> None:
    first_physical: list[tuple[Mapping[str, object], ...]] = []
    second_physical: list[tuple[Mapping[str, object], ...]] = []
    config = RunConfig(
        seed=0,
        workers=1,
        execution_mode="serial",
        fixed_work=1,
        trace_policy="semantic_and_physical",
        max_rounds=1,
    )
    first_ticks = iter((10, 20))
    second_ticks = iter((100, 300))
    first = SerialTxnRuntime[int, int, int](
        clock_ns=lambda: next(first_ticks),
        physical_event_sink=first_physical.append,
    ).run(0, kernel=IncrementKernel(), oracle=IntegerOracle(), config=config)
    second = SerialTxnRuntime[int, int, int](
        clock_ns=lambda: next(second_ticks),
        physical_event_sink=second_physical.append,
    ).run(0, kernel=IncrementKernel(), oracle=IntegerOracle(), config=config)

    assert first.semantic_digest == second.semantic_digest
    assert first.physical_artifact_ref == "txnopt-physical-trace-v1:external"
    assert first_physical != second_physical
