from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pytest

from evrptw.alns import _Evaluator, solve_alns
from evrptw.cache_incremental import CacheIncrementalConfig, RouteEvaluationCache
from evrptw.candidate_transaction import (
    CandidateScreeningBatch,
    CandidateTransactionDeadlineExceeded,
    CandidateTransactionRequest,
    NativeCandidateTransactionConfig,
    NativeCandidateTransactionRuntime,
    execute_candidate_transaction,
    native_screen_candidate_batch,
)
from evrptw.charging import ChargingSubproblemResult
from evrptw.cpu_batch import ExactChargingBackend
from evrptw.measurement import CheapScreeningConfig
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.native_kernels import NativeKernelConfig, NativeKernelRuntime


def _screening(
    candidates: tuple[tuple[str, ...], ...],
) -> CandidateScreeningBatch:
    count = len(candidates)
    codes = np.zeros((count, 16), dtype=np.int64)
    codes[:, 0] = 1
    statuses = np.zeros(count, dtype=np.int64)
    duplicate_of = np.full(count, -1, dtype=np.int64)
    for index, sequence in enumerate(candidates):
        if sequence == ("reject",):
            codes[index, 0] = 0
            codes[index, 1] = 2
        if index == 3:
            statuses[index] = 1
            duplicate_of[index] = 0
    return CandidateScreeningBatch(
        candidates,
        np.arange(count, dtype=np.int64),
        statuses,
        duplicate_of,
        codes,
        np.zeros((count, 15), dtype=np.float64),
        np.arange(count + 1, dtype=np.int64),
        np.zeros(count, dtype=np.int64),
        np.array(
            [count, len({sequence for sequence in candidates}), 0, 0, count],
            dtype=np.int64,
        ),
        {
            "input_candidates": len(candidates),
            "unique_candidates": len({sequence for sequence in candidates}),
        },
        "a" * 64,
    )


def _fixture_instance(name: str = "candidate_transaction_fixture") -> Instance:
    return Instance(
        name,
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 100.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(1.0, 1.0, 1.0, 0.1, 1.0),
        distance_backend="python",
    )


def _feasible_result() -> ChargingSubproblemResult:
    return ChargingSubproblemResult(
        True,
        ("D0", "C1", "D0"),
        2.0,
        0.0,
        0.0,
        0.0,
        1,
        0,
        0,
        0.0,
        "",
    )


def test_candidate_transaction_preserves_order_budget_and_atomic_cache() -> None:
    candidates = (("exact",), ("reject",), ("cached",), ("exact",), ("skipped",))
    staged: dict[tuple[str, ...], str] = {}
    committed: dict[tuple[str, ...], str] = {}
    exact_calls: list[tuple[tuple[str, ...], ...]] = []
    rollbacks: list[str] = []

    def exact_batch(sequences: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
        exact_calls.append(sequences)
        return ("exact:E",)

    result = execute_candidate_transaction(
        CandidateTransactionRequest(
            candidates,
            lane="constraint",
            operator="route_merge",
            iteration=7,
            exact_budget=1,
            deadline=10.0,
        ),
        screen_batch=_screening,
        cache_lookup=lambda sequence: "cache:C" if sequence == ("cached",) else None,
        exact_batch=exact_batch,
        stage_cache_write=lambda sequence, value: staged.__setitem__(sequence, value),
        commit_cache_writes=lambda: committed.update(staged),
        rollback_cache_writes=rollbacks.append,
        rejected_result=lambda _sequence, reason, _status: f"reject:{reason}",
        skipped_result=lambda reason: f"skip:{reason}",
        clock=lambda: 1.0,
    )

    assert result.ordered_results == (
        "exact:E",
        "reject:capacity_prefilter",
        "cache:C",
        "exact:E",
        "skip:operator_exact_budget_exhausted",
    )
    assert exact_calls == [(("exact",),)]
    assert committed == {("exact",): "exact:E"}
    assert rollbacks == []
    assert result.audit.exact_misses == 1
    assert result.audit.cache_hits == 1
    assert result.audit.budget_skips == 1
    assert result.audit.duplicate_candidates == 1
    assert len(result.audit.transaction_sha256) == 64


def test_native_candidate_screening_matches_valid_invalid_duplicate_and_cache_hit() -> None:
    instance = Instance(
        "candidate_transaction_fixture",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 100.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(100.0, 1.0, 1.0, 0.1, 1.0),
        distance_backend="python",
    )
    runtime = NativeKernelRuntime.build(instance, NativeKernelConfig())
    candidates = (("C1",), ("C1",), ("C1", "C2"), ())

    batch = native_screen_candidate_batch(
        instance,
        candidates,
        native_runtime=runtime,
        transaction_runtime=NativeCandidateTransactionRuntime(NativeCandidateTransactionConfig()),
        negative_cache={(): "route_structure_prefilter"},
        deadline=time.perf_counter() + 10.0,
    )

    assert batch.sequences == candidates
    assert tuple(batch.native_status(index) for index in range(4)) == (
        "screened",
        "duplicate",
        "screened",
        "negative_cache_hit",
    )
    assert tuple(batch.accepted(index) for index in range(4)) == (
        True,
        True,
        False,
        False,
    )
    assert batch.reason(2) == "capacity_prefilter"
    assert batch.reason(3) == "route_structure_prefilter"
    assert batch.counters == {
        "input_candidates": 4,
        "unique_candidates": 3,
        "duplicate_candidates": 1,
        "negative_cache_hits": 1,
        "screened_candidates": 2,
    }
    assert runtime.statistics()["native_screening_batch_occupancies"] == (4,)


@pytest.mark.parametrize("tampered_field", ("accepted", "metric", "counter"))
def test_native_candidate_screening_digest_binds_all_returned_fields(
    monkeypatch: pytest.MonkeyPatch,
    tampered_field: str,
) -> None:
    from evrptw import _core

    instance = _fixture_instance("candidate_transaction_digest_fixture")
    native_runtime = NativeKernelRuntime.build(instance, NativeKernelConfig())
    transaction_runtime = NativeCandidateTransactionRuntime(NativeCandidateTransactionConfig())
    original = _core.screen_route_batch_transaction_v2

    def tampered(*args: object) -> tuple[object, ...]:
        payload = list(original(*args))
        target = {"accepted": 3, "metric": 4, "counter": 5}[tampered_field]
        changed = payload[target].copy()
        if tampered_field == "accepted":
            changed[0, 0] = 1 - changed[0, 0]
        elif tampered_field == "metric":
            changed[0, 3] += 1.0
        else:
            changed[4] += 1
        payload[target] = changed
        return tuple(payload)

    monkeypatch.setattr(_core, "screen_route_batch_transaction_v2", tampered)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        native_screen_candidate_batch(
            instance,
            (("C1",),),
            native_runtime=native_runtime,
            transaction_runtime=transaction_runtime,
            negative_cache={},
            deadline=time.perf_counter() + 10.0,
        )


def test_negative_cache_packing_appends_without_repacking_existing_buffer() -> None:
    runtime = NativeCandidateTransactionRuntime(NativeCandidateTransactionConfig())
    mapping = {"C1": 1, "C2": 2}
    negative_cache: dict[tuple[str, ...], str] = {}

    first = runtime.packed_negative_cache(negative_cache, mapping)
    offsets_buffer = runtime._negative_cache_offsets
    indices_buffer = runtime._negative_cache_indices
    negative_cache[("C1",)] = "capacity_prefilter"
    runtime.commit_negative_cache_entries(
        {("C1",): "capacity_prefilter"},
        mapping,
    )
    second = runtime.packed_negative_cache(negative_cache, mapping)

    assert runtime._negative_cache_offsets is offsets_buffer
    assert runtime._negative_cache_indices is indices_buffer
    assert first[0].tolist() == [0]
    assert second[0].tolist() == [0, 1]
    assert second[1].tolist() == [1]
    assert second[2].tolist() == [2]


def test_route_cache_commit_restores_exact_and_negative_state_on_store_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _fixture_instance("candidate_transaction_atomic_cache_fixture")
    route_cache = RouteEvaluationCache(
        instance,
        CacheIncrementalConfig(enabled=True, max_entries=2),
    )
    result = _feasible_result()
    route_cache.store(("C1",), result)
    before_statistics = route_cache.statistics_dict()
    evaluator = _Evaluator(
        instance,
        deadline=time.perf_counter() + 10.0,
        route_cache=route_cache,
        negative_screening_sequences={("C2",): "capacity_prefilter"},
    )
    evaluator.pending_candidate_cache[("new-1",)] = result
    evaluator.pending_candidate_cache[("new-2",)] = result
    evaluator.pending_negative_screening_sequences[("reject",)] = "capacity_prefilter"
    original_store = route_cache.store
    calls = 0

    def fail_second(
        sequence: tuple[str, ...],
        value: ChargingSubproblemResult,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second store failure")
        return original_store(sequence, value)

    monkeypatch.setattr(route_cache, "store", fail_second)
    with pytest.raises(RuntimeError, match="injected second store failure"):
        evaluator._commit_pending_candidate_cache()

    assert route_cache.statistics_dict() == before_statistics
    assert route_cache.contains(("C1",))
    assert not route_cache.contains(("new-1",))
    assert not route_cache.contains(("new-2",))
    assert evaluator.negative_screening_sequences == {("C2",): "capacity_prefilter"}
    evaluator._discard_pending_candidate_cache("test_cleanup")
    assert evaluator.pending_candidate_cache == {}
    assert evaluator.pending_negative_screening_sequences == {}


def test_route_cache_commit_rolls_back_exact_entries_when_negative_commit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _fixture_instance("candidate_transaction_cross_cache_atomic_fixture")
    route_cache = RouteEvaluationCache(
        instance,
        CacheIncrementalConfig(enabled=True, max_entries=2),
    )
    result = _feasible_result()
    route_cache.store(("C1",), result)
    before_statistics = route_cache.statistics_dict()
    transaction_runtime = NativeCandidateTransactionRuntime(
        NativeCandidateTransactionConfig()
    )
    evaluator = _Evaluator(
        instance,
        deadline=time.perf_counter() + 10.0,
        route_cache=route_cache,
        native_runtime=NativeKernelRuntime.build(instance, NativeKernelConfig()),
        candidate_transaction_runtime=transaction_runtime,
        negative_screening_sequences={("C2",): "capacity_prefilter"},
    )
    evaluator.pending_candidate_cache[("new-1",)] = result
    evaluator.pending_negative_screening_sequences[("reject",)] = "capacity_prefilter"

    def fail_negative_commit(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected negative cache failure")

    monkeypatch.setattr(
        NativeCandidateTransactionRuntime,
        "commit_negative_cache_entries",
        fail_negative_commit,
    )
    with pytest.raises(RuntimeError, match="injected negative cache failure"):
        evaluator._commit_pending_candidate_cache()

    assert route_cache.statistics_dict() == before_statistics
    assert route_cache.contains(("C1",))
    assert not route_cache.contains(("new-1",))
    assert evaluator.negative_screening_sequences == {("C2",): "capacity_prefilter"}


def test_worker_failure_rolls_back_staged_negative_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _fixture_instance("candidate_transaction_negative_rollback_fixture")
    native_runtime = NativeKernelRuntime.build(instance, NativeKernelConfig())
    evaluator = _Evaluator(
        instance,
        deadline=time.perf_counter() + 10.0,
        screening_config=CheapScreeningConfig(),
        backend=ExactChargingBackend.CPU_BATCH,
        native_runtime=native_runtime,
        candidate_transaction_runtime=NativeCandidateTransactionRuntime(
            NativeCandidateTransactionConfig()
        ),
        negative_screening_sequences={("original",): "capacity_prefilter"},
    )
    evaluator.operator = "route_merge"

    def fail_exact(*_args: object, **_kwargs: object) -> tuple[()]:
        raise RuntimeError("worker failed after screening")

    monkeypatch.setattr(evaluator, "_solve_uncached_batch", fail_exact)
    with pytest.raises(RuntimeError, match="worker failed after screening"):
        evaluator.candidate_route_batch(
            (("C1", "C2"), ("C1",)),
            exact_budget=1,
        )

    assert evaluator.negative_screening_sequences == {("original",): "capacity_prefilter"}
    assert evaluator.pending_negative_screening_sequences == {}


def test_route_evaluator_delegates_ordered_pool_to_native_candidate_transaction() -> None:
    instance = Instance(
        "candidate_transaction_evaluator_fixture",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 100.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(100.0, 1.0, 1.0, 0.1, 1.0),
        distance_backend="python",
    )
    native_runtime = NativeKernelRuntime.build(instance, NativeKernelConfig())
    transaction_runtime = NativeCandidateTransactionRuntime(NativeCandidateTransactionConfig())
    evaluator = _Evaluator(
        instance,
        deadline=time.perf_counter() + 10.0,
        screening_config=CheapScreeningConfig(),
        backend=ExactChargingBackend.CPU_BATCH,
        native_runtime=native_runtime,
        candidate_transaction_runtime=transaction_runtime,
    )
    evaluator.operator = "route_merge"
    evaluator.iteration = 4

    results = evaluator.candidate_route_batch(
        (("C1",), ("C1",), ("C1", "C2")),
        exact_budget=1,
        base_sequences=(("C1",), ("C1",), ("C1",)),
    )

    assert tuple(result.feasible for result in results) == (True, True, False)
    assert evaluator.calls == 1
    assert transaction_runtime.statistics()["native_candidate_transactions"] == 1
    assert transaction_runtime.events[0]["operator"] == "route_merge"
    assert transaction_runtime.events[0]["exact_misses"] == 1
    assert native_runtime.statistics()["native_screening_batch_occupancies"] == (3,)
    assert tuple(evaluator.propagation_snapshots) == (("C1",),)
    assert evaluator.incremental_propagations == 3


def test_route_evaluator_batched_screening_ablation_preserves_budget_and_order() -> None:
    instance = Instance(
        "candidate_batched_screening_fixture",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 100.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(100.0, 1.0, 1.0, 0.1, 1.0),
        distance_backend="python",
    )
    native_runtime = NativeKernelRuntime.build(instance, NativeKernelConfig())
    transaction_runtime = NativeCandidateTransactionRuntime(
        NativeCandidateTransactionConfig(implementation_mode="batched_screening")
    )
    evaluator = _Evaluator(
        instance,
        deadline=time.perf_counter() + 10.0,
        screening_config=CheapScreeningConfig(),
        backend=ExactChargingBackend.CPU_BATCH,
        native_runtime=native_runtime,
        candidate_transaction_runtime=transaction_runtime,
    )
    evaluator.operator = "route_merge"

    results = evaluator.candidate_route_batch(
        (("C1",), ("C1",), ("C2",)),
        exact_budget=1,
    )

    assert results[0].feasible
    assert results[1].feasible
    assert results[2].failure_reason == ("candidate_transaction:operator_exact_budget_exhausted")
    assert evaluator.calls == 1
    assert transaction_runtime.statistics()["native_candidate_transactions"] == 0
    assert native_runtime.statistics()["native_screening_batch_occupancies"] == (3,)


def test_solve_alns_candidate_transaction_is_explicit_and_requires_native_screening() -> None:
    instance = Instance(
        "candidate_transaction_solver_fixture",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(100.0, 1.0, 1.0, 0.1, 1.0),
        distance_backend="python",
    )
    transaction_config = NativeCandidateTransactionConfig()
    with pytest.raises(ValueError, match="require native kernels"):
        solve_alns(
            instance,
            seed=1,
            max_iterations=1,
            time_limit_seconds=1.0,
            candidate_transaction_config=transaction_config,
            screening_config=CheapScreeningConfig(),
        )
    with pytest.raises(ValueError, match="require cheap screening"):
        solve_alns(
            instance,
            seed=1,
            max_iterations=1,
            time_limit_seconds=1.0,
            candidate_transaction_config=transaction_config,
            native_kernel_config=NativeKernelConfig(),
        )

    result = solve_alns(
        instance,
        seed=1,
        max_iterations=1,
        time_limit_seconds=1.0,
        candidate_transaction_config=transaction_config,
        native_kernel_config=NativeKernelConfig(),
        screening_config=CheapScreeningConfig(),
    )

    assert result.feasible
    assert result.candidate_transaction_statistics["native_candidate_transaction_fallbacks"] == 0


def test_candidate_transaction_rolls_back_worker_failure_without_fallback() -> None:
    rollbacks: list[str] = []
    fallback_called = False

    def fail_exact(_sequences: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
        raise RuntimeError("worker failed")

    def forbidden_fallback(_sequence: tuple[str, ...]) -> str | None:
        nonlocal fallback_called
        fallback_called = True
        return None

    with pytest.raises(RuntimeError, match="worker failed"):
        execute_candidate_transaction(
            CandidateTransactionRequest(
                (("exact",),),
                lane="legacy",
                operator="route_merge",
                iteration=3,
                exact_budget=1,
                deadline=10.0,
            ),
            screen_batch=_screening,
            cache_lookup=forbidden_fallback,
            exact_batch=fail_exact,
            stage_cache_write=lambda _sequence, _value: None,
            commit_cache_writes=lambda: None,
            rollback_cache_writes=rollbacks.append,
            rejected_result=lambda _sequence, _reason, _status: "rejected",
            skipped_result=lambda _reason: "skipped",
            clock=lambda: 1.0,
        )

    assert fallback_called is True
    assert len(rollbacks) == 1
    assert "RuntimeError:worker failed" in rollbacks[0]


def test_candidate_transaction_checks_deadline_before_native_screening() -> None:
    screen_called = False
    rollbacks: list[str] = []

    def screen(
        candidates: tuple[tuple[str, ...], ...],
    ) -> CandidateScreeningBatch:
        nonlocal screen_called
        screen_called = True
        return _screening(candidates)

    with pytest.raises(CandidateTransactionDeadlineExceeded) as raised:
        execute_candidate_transaction(
            CandidateTransactionRequest(
                (("exact",),),
                lane="legacy",
                operator="route_merge",
                iteration=None,
                exact_budget=1,
                deadline=1.0,
            ),
            screen_batch=screen,
            cache_lookup=lambda _sequence: None,
            exact_batch=lambda _sequences: (),
            stage_cache_write=lambda _sequence, _value: None,
            commit_cache_writes=lambda: None,
            rollback_cache_writes=rollbacks.append,
            rejected_result=lambda _sequence, _reason, _status: "rejected",
            skipped_result=lambda _reason: "skipped",
            clock=lambda: 1.0,
        )

    assert raised.value.boundary == "before_native_screening"
    assert screen_called is False
    assert len(rollbacks) == 1


@pytest.mark.parametrize(
    "mutator",
    (
        lambda: NativeCandidateTransactionConfig(candidate_order_policy="distance"),
        lambda: NativeCandidateTransactionConfig(exact_budget_policy="top_k"),
        lambda: NativeCandidateTransactionConfig(cache_write_policy="immediate"),
        lambda: NativeCandidateTransactionConfig(failure_policy="python_fallback"),
        lambda: NativeCandidateTransactionConfig(implementation_mode="python_fallback"),
    ),
)
def test_candidate_transaction_config_rejects_semantic_drift(
    mutator: Callable[[], NativeCandidateTransactionConfig],
) -> None:
    with pytest.raises(ValueError):
        mutator()


def test_candidate_transaction_ablation_modes_are_explicit() -> None:
    assert (
        NativeCandidateTransactionConfig(implementation_mode="pair_pruning").implementation_mode
        == "pair_pruning"
    )
    assert (
        NativeCandidateTransactionConfig(
            implementation_mode="batched_screening"
        ).implementation_mode
        == "batched_screening"
    )
    assert NativeCandidateTransactionConfig().implementation_mode == ("candidate_transaction")
