from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import pytest

from txnopt import RunConfig
from txnopt._internal.budget import BudgetLedger
from txnopt._internal.cache import (
    CacheCommitReceipt,
    CacheLookup,
    CacheTxn,
    InMemoryCacheStore,
)
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


def test_published_cache_with_malformed_receipt_is_observable_and_fail_fast() -> None:
    class MalformedReceiptCache(InMemoryCacheStore[int]):
        def commit(self, transaction: CacheTxn[int]) -> object:
            super().commit(transaction)
            return object()

    cache = MalformedReceiptCache()
    semantic: list[tuple[Mapping[str, object], ...]] = []
    physical: list[tuple[Mapping[str, object], ...]] = []

    with pytest.raises(RuntimeContractError, match="malformed receipt"):
        SerialTxnRuntime[int, int, int](
            cache_factory=lambda: cache,
            semantic_event_sink=semantic.append,
            physical_event_sink=physical.append,
        ).run(
            0,
            kernel=IncrementKernel(),
            oracle=IntegerOracle(),
            config=RunConfig(
                seed=0,
                workers=1,
                execution_mode="serial",
                fixed_work=1,
                trace_policy="semantic_and_physical",
                max_rounds=1,
            ),
        )

    assert cache.snapshot() == (1, {"integer:1": 1})
    assert semantic[0][-2]["phase"] == "COMMITTED"
    assert semantic[0][-2]["cache_receipt_status"] == (
        "reconstructed_after_contract_failure"
    )
    assert semantic[0][-1] == {
        "event": "run_terminated",
        "reason": "runtime_contract_error",
    }
    assert len(physical) == 1
    assert physical[0][0]["termination_reason"] == "runtime_contract_error"


def test_unverifiable_cache_outcome_is_neither_committed_nor_aborted() -> None:
    class UnverifiablePublishedCache(InMemoryCacheStore[int]):
        def commit(self, transaction: CacheTxn[int]) -> object:
            super().commit(transaction)
            return object()

        def snapshot(self) -> tuple[int, Mapping[str, int]]:
            raise OSError("simulated snapshot failure after publication")

    cache = UnverifiablePublishedCache()
    semantic: list[tuple[Mapping[str, object], ...]] = []
    physical: list[tuple[Mapping[str, object], ...]] = []

    with pytest.raises(RuntimeContractError, match="cannot be verified") as caught:
        SerialTxnRuntime[int, int, int](
            cache_factory=lambda: cache,
            semantic_event_sink=semantic.append,
            physical_event_sink=physical.append,
        ).run(
            0,
            kernel=IncrementKernel(),
            oracle=IntegerOracle(),
            config=RunConfig(
                seed=0,
                workers=1,
                execution_mode="serial",
                fixed_work=1,
                trace_policy="semantic_and_physical",
                max_rounds=1,
            ),
        )

    assert any("snapshot verification failed" in note for note in caught.value.__notes__)
    assert semantic[0][-2]["phase"] == "INTERRUPTED"
    assert semantic[0][-2]["cache_outcome"] == "unknown"
    assert all(event.get("phase") != "ABORTED" for event in semantic[0])
    assert all(event.get("phase") != "COMMITTED" for event in semantic[0])
    assert semantic[0][-1]["reason"] == "cache_outcome_unknown"
    assert len(physical) == 1


def test_unpublished_cache_cannot_forge_a_valid_commit_receipt() -> None:
    class ForgedReceiptCache(InMemoryCacheStore[int]):
        def commit(self, transaction: CacheTxn[int]) -> object:
            del transaction
            return CacheCommitReceipt(0, 1, 1)

    cache = ForgedReceiptCache()
    semantic: list[tuple[Mapping[str, object], ...]] = []

    with pytest.raises(RuntimeContractError, match="unverified outcome"):
        SerialTxnRuntime[int, int, int](
            cache_factory=lambda: cache,
            semantic_event_sink=semantic.append,
        ).run(
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

    assert cache.snapshot() == (0, {})
    assert semantic[0][-2]["phase"] == "ABORTED"
    assert all(event.get("phase") != "COMMITTED" for event in semantic[0])


def test_closed_but_unpublished_cache_is_reported_as_outcome_unknown() -> None:
    class ClosedWithoutPublicationCache(InMemoryCacheStore[int]):
        def commit(self, transaction: CacheTxn[int]) -> object:
            transaction.closed = True
            transaction.staged.clear()
            return object()

    cache = ClosedWithoutPublicationCache()
    semantic: list[tuple[Mapping[str, object], ...]] = []

    with pytest.raises(RuntimeContractError, match="differs from"):
        SerialTxnRuntime[int, int, int](
            cache_factory=lambda: cache,
            semantic_event_sink=semantic.append,
        ).run(
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

    assert cache.snapshot() == (0, {})
    assert semantic[0][-2]["phase"] == "INTERRUPTED"
    assert semantic[0][-2]["cache_outcome"] == "unknown"
    assert all(event.get("phase") != "COMMITTED" for event in semantic[0])


def test_stale_cache_snapshot_interrupts_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterferingCache(InMemoryCacheStore[int]):
        def lookup(self, keys: Sequence[str]) -> CacheLookup[int]:
            external = self.begin()
            self.commit(external)
            return super().lookup(keys)

    ledgers: list[BudgetLedger] = []

    class TrackingBudget(BudgetLedger):
        def __init__(self, limit: int | None) -> None:
            super().__init__(limit)
            ledgers.append(self)

    monkeypatch.setattr("txnopt.runtime.BudgetLedger", TrackingBudget)
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
    assert len(ledgers) == 1
    assert ledgers[0].reserved == 0
    assert ledgers[0].started == 0


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


def test_admission_limited_runtime_stops_after_the_canonical_admissible_prefix() -> None:
    class LimitedKernel(IncrementKernel):
        admission_limit = 2

        def propose(
            self,
            snapshot: int,
            *,
            round_id: int,
            random_tape: Sequence[int],
        ) -> Sequence[int]:
            return (1, 2, 3, 4, 5, 6)

    class PrefixOracle(IntegerOracle):
        def __init__(self) -> None:
            super().__init__()
            self.keyed: list[int] = []
            self.screened_batches: list[tuple[int, ...]] = []

        def stable_key(self, candidate: int) -> str:
            self.keyed.append(candidate)
            return super().stable_key(candidate)

        def screen(self, candidates: Sequence[int]) -> Sequence[bool]:
            batch = tuple(candidates)
            self.screened_batches.append(batch)
            return tuple(candidate % 2 == 0 for candidate in batch)

    streams: list[tuple[Mapping[str, object], ...]] = []
    oracle = PrefixOracle()
    result = SerialTxnRuntime[int, int, int](
        semantic_event_sink=streams.append
    ).run(
        0,
        kernel=LimitedKernel(),
        oracle=oracle,
        config=RunConfig(
            seed=0,
            workers=1,
            execution_mode="serial",
            fixed_work=2,
            max_rounds=1,
        ),
    )

    assert result.last_committed_state == 2
    assert oracle.keyed == [1, 2, 3, 4]
    assert oracle.screened_batches == [(1, 2), (3, 4)]
    screening = next(event for event in streams[0] if event["event"] == "candidate_screening")
    assert screening == {
        "event": "candidate_screening",
        "proposed_count": 6,
        "screened_candidate_count": 4,
        "unique_count": 4,
        "screened_admissible_count": 2,
        "admitted_count": 2,
        "admission_limit": 2,
        "screening_complete": False,
    }


def test_contract_failure_after_reservation_rolls_back_audits_and_stays_fail_fast() -> None:
    class BrokenOracle(IntegerOracle):
        def evaluate_batch(
            self,
            candidates: Sequence[int],
            *,
            work_budget: int,
            deadline_ns: int | None,
        ) -> Sequence[int]:
            raise RuntimeContractError("simulated malformed exact receipt")

    physical: list[tuple[Mapping[str, object], ...]] = []
    cache = InMemoryCacheStore[int]()
    cost_ticks = iter((10, 30))
    runtime = SerialTxnRuntime[int, int, int](
        cost_clock_ns=lambda: next(cost_ticks),
        cache_factory=lambda: cache,
        physical_event_sink=physical.append,
    )

    with pytest.raises(RuntimeContractError, match="malformed exact receipt"):
        runtime.run(
            0,
            kernel=IncrementKernel(),
            oracle=BrokenOracle(),
            config=RunConfig(
                seed=0,
                workers=1,
                execution_mode="serial",
                fixed_work=1,
                trace_policy="semantic_and_physical",
                max_rounds=1,
            ),
        )

    assert cache.snapshot() == (0, {})
    assert len(physical) == 1
    assert physical[0][0]["event"] == "run_observation"
    assert physical[0][0]["termination_reason"] == "runtime_contract_error"
    waste = physical[0][1]
    assert waste["event"] == "t4_waste_observation"
    assert waste["reason"] == "runtime_contract_error"
    assert waste["observed_discarded_work_units"] == 1
    assert waste["bound_satisfied"] is True
    assert waste["measured_cmax_ns"] == 20
    assert waste["observed_discarded_cost_upper_ns"] == 20
    assert waste["discarded_cost_bound_ns"] == 20


def test_contract_failure_during_preparation_closes_cache_transaction() -> None:
    class FailingLookupCache(InMemoryCacheStore[int]):
        transaction: CacheTxn[int] | None = None

        def begin(self) -> CacheTxn[int]:
            transaction = super().begin()
            self.transaction = transaction
            return transaction

        def lookup(self, keys: Sequence[str]) -> CacheLookup[int]:
            raise RuntimeContractError("simulated lookup contract failure")

    cache = FailingLookupCache()
    semantic: list[tuple[Mapping[str, object], ...]] = []
    physical: list[tuple[Mapping[str, object], ...]] = []
    with pytest.raises(RuntimeContractError, match="lookup contract failure"):
        SerialTxnRuntime[int, int, int](
            cache_factory=lambda: cache,
            semantic_event_sink=semantic.append,
            physical_event_sink=physical.append,
        ).run(
            0,
            kernel=IncrementKernel(),
            oracle=IntegerOracle(),
            config=RunConfig(
                seed=0,
                workers=1,
                execution_mode="serial",
                fixed_work=1,
                trace_policy="semantic_and_physical",
                max_rounds=1,
            ),
        )

    assert cache.transaction is not None
    assert cache.transaction.closed is True
    assert cache.snapshot() == (0, {})
    assert semantic[0][-2]["phase"] == "ABORTED"
    assert semantic[0][-1] == {
        "event": "run_terminated",
        "reason": "runtime_contract_error",
    }
    assert len(physical) == 1
    assert len(physical[0]) == 1


def test_rollback_callback_failure_does_not_hide_the_primary_contract_error() -> None:
    class FailingRollbackCache(InMemoryCacheStore[int]):
        transaction: CacheTxn[int] | None = None

        def begin(self) -> CacheTxn[int]:
            transaction = super().begin()
            self.transaction = transaction
            return transaction

        def rollback(self, transaction: CacheTxn[int]) -> None:
            raise OSError("simulated rollback callback failure")

    class BrokenOracle(IntegerOracle):
        def evaluate_batch(
            self,
            candidates: Sequence[int],
            *,
            work_budget: int,
            deadline_ns: int | None,
        ) -> Sequence[int]:
            raise RuntimeContractError("primary oracle contract failure")

    cache = FailingRollbackCache()
    physical: list[tuple[Mapping[str, object], ...]] = []
    with pytest.raises(RuntimeContractError, match="primary oracle contract failure") as caught:
        SerialTxnRuntime[int, int, int](
            cache_factory=lambda: cache,
            physical_event_sink=physical.append,
        ).run(
            0,
            kernel=IncrementKernel(),
            oracle=BrokenOracle(),
            config=RunConfig(
                seed=0,
                workers=1,
                execution_mode="serial",
                fixed_work=1,
                trace_policy="semantic_and_physical",
                max_rounds=1,
            ),
        )

    assert any("secondary cleanup failure" in note for note in caught.value.__notes__)
    assert cache.transaction is not None
    assert cache.transaction.closed is True
    assert cache.transaction.staged == {}
    assert len(physical) == 1
    assert physical[0][0]["termination_reason"] == "runtime_contract_error"
    assert physical[0][1]["bound_satisfied"] is True


def test_invalid_kernel_decision_emits_one_aborted_semantic_prefix_before_raising() -> None:
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

    semantic: list[tuple[Mapping[str, object], ...]] = []
    physical: list[tuple[Mapping[str, object], ...]] = []

    with pytest.raises(RuntimeContractError, match="neither"):
        SerialTxnRuntime[int, int, int](
            semantic_event_sink=semantic.append,
            physical_event_sink=physical.append,
        ).run(
            0,
            kernel=InvalidKernel(),
            oracle=IntegerOracle(),
            config=RunConfig(
                seed=0,
                workers=1,
                execution_mode="serial",
                fixed_work=1,
                trace_policy="semantic_and_physical",
                max_rounds=1,
            ),
        )

    assert len(semantic) == 1
    assert semantic[0][-2]["event"] == "candidate_transaction"
    assert semantic[0][-2]["phase"] == "ABORTED"
    assert semantic[0][-1] == {
        "event": "run_terminated",
        "reason": "runtime_contract_error",
    }
    assert len(physical) == 1


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
