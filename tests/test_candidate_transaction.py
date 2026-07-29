from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from evrptw.alns import _Evaluator, solve_alns
from evrptw.candidate_transaction import (
    CandidateScreeningBatch,
    CandidateScreeningDecision,
    CandidateTransactionDeadlineExceeded,
    CandidateTransactionRequest,
    NativeCandidateTransactionConfig,
    NativeCandidateTransactionRuntime,
    execute_candidate_transaction,
    native_screen_candidate_batch,
)
from evrptw.cpu_batch import ExactChargingBackend
from evrptw.measurement import CheapScreeningConfig
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.native_kernels import NativeKernelConfig, NativeKernelRuntime


def _screening(
    candidates: tuple[tuple[str, ...], ...],
) -> CandidateScreeningBatch:
    decisions = tuple(
        CandidateScreeningDecision(
            candidate_id=index,
            sequence=sequence,
            accepted=sequence != ("reject",),
            reason="capacity_prefilter" if sequence == ("reject",) else "",
            native_status="duplicate" if index == 3 else "screened",
            duplicate_of=0 if index == 3 else None,
        )
        for index, sequence in enumerate(candidates)
    )
    return CandidateScreeningBatch(
        decisions,
        {
            "input_candidates": len(candidates),
            "unique_candidates": len({sequence for sequence in candidates}),
        },
        "a" * 64,
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
        rejected_result=lambda decision: f"reject:{decision.reason}",
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
        negative_cache={(): "route_structure_prefilter"},
        deadline=time.perf_counter() + 10.0,
    )

    assert tuple(decision.sequence for decision in batch.decisions) == candidates
    assert tuple(decision.native_status for decision in batch.decisions) == (
        "screened",
        "duplicate",
        "screened",
        "negative_cache_hit",
    )
    assert tuple(decision.accepted for decision in batch.decisions) == (
        True,
        True,
        False,
        False,
    )
    assert batch.decisions[2].reason == "capacity_prefilter"
    assert batch.decisions[3].reason == "route_structure_prefilter"
    assert batch.counters == {
        "input_candidates": 4,
        "unique_candidates": 3,
        "duplicate_candidates": 1,
        "negative_cache_hits": 1,
        "screened_candidates": 2,
    }
    assert runtime.statistics()["native_screening_batch_occupancies"] == (4,)


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
    transaction_runtime = NativeCandidateTransactionRuntime(
        NativeCandidateTransactionConfig()
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
    assert results[2].failure_reason == (
        "candidate_transaction:operator_exact_budget_exhausted"
    )
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
    assert (
        result.candidate_transaction_statistics["native_candidate_transaction_fallbacks"]
        == 0
    )


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
            rejected_result=lambda _decision: "rejected",
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
            rejected_result=lambda _decision: "rejected",
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
    assert NativeCandidateTransactionConfig(
        implementation_mode="pair_pruning"
    ).implementation_mode == "pair_pruning"
    assert NativeCandidateTransactionConfig(
        implementation_mode="batched_screening"
    ).implementation_mode == "batched_screening"
    assert NativeCandidateTransactionConfig().implementation_mode == (
        "candidate_transaction"
    )
