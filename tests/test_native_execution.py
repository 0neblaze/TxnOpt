from __future__ import annotations

import hashlib
import os
import random
import socket
import struct
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from evrptw.alns import _destroy, _Evaluator, solve_alns
from evrptw.cache_incremental import (
    CacheIncrementalConfig,
    RouteEvaluationCache,
    charging_result_semantic_digest,
    charging_result_semantic_payload,
    estimate_cache_entry_bytes,
)
from evrptw.candidate_control import CandidateControlConfig, CandidateControlRuntime
from evrptw.candidate_transaction import (
    BoundedNegativeSequenceCache,
    NativeCandidateTransactionConfig,
    NativeCandidateTransactionRuntime,
)
from evrptw.charging import solve_exact_charging
from evrptw.exact_deadline import ExactCallController, ExactDeadlineConfig
from evrptw.experiments.stage052_native_architectures import (
    _semantic_candidate_trajectory,
)
from evrptw.measurement import CheapScreeningConfig, MeasurementConfig, Stage03Trace
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.native_execution import (
    NATIVE_EXECUTION_SCHEMA_VERSION,
    NativeCandidateRoundRequest,
    NativeCandidateRoundResult,
    Stage052NativeExecutionConfig,
    execute_native_candidate_round,
)
from evrptw.native_kernels import NativeKernelConfig, NativeKernelRuntime
from evrptw.native_scheduler import NativeHostScheduler
from evrptw.objective import SolutionObjective, accept_annealing_move
from evrptw.parser import parse_schneider
from evrptw.stage04 import Stage04Config
from evrptw.warm_start import (
    WarmStartValidationConfig,
    canonical_customer_sequences_sha256,
)


def _fixture_instance() -> Instance:
    return Instance(
        "native_execution_fixture",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 100.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(10.0, 100.0, 1.0, 0.1, 1.0),
        distance_backend="python",
    )


def _candidate_control_ranking_fixture() -> Instance:
    """Make lexical and full-screening distance ranks intentionally disagree."""

    return Instance(
        "candidate_control_ranking_fixture",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 10.0, 0.0, 1.0, 0.0, 100.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(10.0, 100.0, 1.0, 0.1, 1.0),
        distance_backend="python",
    )


def _candidate_plan_fixture() -> Instance:
    return Instance(
        "candidate_plan_fixture",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 1_000.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 1_000.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 1_000.0, 0.0),
            Node("C3", NodeType.CUSTOMER, 3.0, 0.0, 1.0, 0.0, 1_000.0, 0.0),
            Node("C4", NodeType.CUSTOMER, 4.0, 0.0, 1.0, 0.0, 1_000.0, 0.0),
        ),
        Vehicle(100.0, 100.0, 1.0, 0.1, 1.0),
        distance_backend="python",
    )


def _per_solve_config() -> Stage052NativeExecutionConfig:
    return _native_config("per_solve_runtime")


def _native_config(mode: str) -> Stage052NativeExecutionConfig:
    return Stage052NativeExecutionConfig(
        mode=mode,
        native_kernel_config=NativeKernelConfig(),
        candidate_transaction_config=NativeCandidateTransactionConfig(),
        candidate_control_config=CandidateControlConfig(worker_count=1),
        shard_processes=6,
        compute_threads_per_shard=4,
    )


def _full_native_solve_kwargs() -> dict[str, object]:
    return {
        "initial_customer_sequences": (("C1", "C2"),),
        "screening_config": CheapScreeningConfig(),
        "cache_incremental_config": CacheIncrementalConfig(enabled=True),
        "stage04_config": Stage04Config(),
    }


def test_per_solve_native_execution_protocol_is_explicit_and_fail_fast() -> None:
    config = _per_solve_config()

    assert config.schema_version == NATIVE_EXECUTION_SCHEMA_VERSION
    assert config.worker_protocol == "candidate_round_soa_v2"
    assert config.compute_thread_limit == 24
    assert config.fallback_allowed is False
    assert config.failure_policy == "fail_fast_no_fallback"


@pytest.mark.parametrize(
    ("mode", "expected_protocol"),
    [
        ("full_native_alns", "full_solve_soa_v2"),
        ("host_scheduler", "unix_shm_scheduler_v1"),
    ],
)
def test_native_execution_mode_selects_one_fixed_protocol(
    mode: str,
    expected_protocol: str,
) -> None:
    config = Stage052NativeExecutionConfig(
        mode=mode,
        native_kernel_config=NativeKernelConfig(),
        candidate_transaction_config=NativeCandidateTransactionConfig(),
        candidate_control_config=CandidateControlConfig(worker_count=1),
        shard_processes=6,
        compute_threads_per_shard=4,
    )

    assert config.worker_protocol == expected_protocol
    assert config.compute_thread_limit == 24


def test_native_execution_rejects_ambiguous_or_oversubscribed_topology() -> None:
    common = {
        "native_kernel_config": NativeKernelConfig(),
        "candidate_transaction_config": NativeCandidateTransactionConfig(),
        "candidate_control_config": CandidateControlConfig(worker_count=1),
    }

    with pytest.raises(ValueError, match="exactly 24 compute threads"):
        Stage052NativeExecutionConfig(
            mode="per_solve_runtime",
            shard_processes=6,
            compute_threads_per_shard=3,
            **common,
        )
    with pytest.raises(ValueError, match="disabled native configuration is ambiguous"):
        Stage052NativeExecutionConfig(
            mode="per_solve_runtime",
            shard_processes=6,
            compute_threads_per_shard=4,
            enabled=False,
            **common,
        )


def test_explicit_native_protocol_is_the_only_guard_bypass() -> None:
    instance = _fixture_instance()
    native = NativeKernelConfig()
    control = CandidateControlConfig(worker_count=1)

    with pytest.raises(ValueError, match="explicit native worker protocol"):
        solve_alns(
            instance,
            seed=2014,
            max_iterations=1,
            time_limit_seconds=2.0,
            native_kernel_config=native,
            candidate_control_config=control,
        )

    result = solve_alns(
        instance,
        seed=2014,
        max_iterations=1,
        time_limit_seconds=2.0,
        screening_config=CheapScreeningConfig(),
        native_execution_config=_per_solve_config(),
    )

    assert result.feasible
    assert result.native_execution_statistics["mode"] == "per_solve_runtime"
    assert result.native_execution_statistics["worker_protocol"] == "candidate_round_soa_v2"
    assert result.native_execution_statistics["fallback_count"] == 0


def test_explicit_warm_start_validation_does_not_enable_candidate_control(
    tmp_path: Path,
) -> None:
    instance = _fixture_instance()
    supplied = (("C1", "C2"),)
    source_path = tmp_path / "source-solution.json"
    source_path.write_text("{}", encoding="utf-8")
    provenance = {
        "source_solution_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_solution_path": str(source_path),
        "source_customer_sequences_sha256": canonical_customer_sequences_sha256(
            supplied
        ),
    }

    with pytest.raises(ValueError, match="requires candidate control or the explicit"):
        solve_alns(
            instance,
            seed=2014,
            max_iterations=1,
            time_limit_seconds=2.0,
            initial_customer_sequences=supplied,
            initial_solution_provenance=provenance,
        )

    result = solve_alns(
        instance,
        seed=2014,
        max_iterations=1,
        time_limit_seconds=2.0,
        initial_customer_sequences=supplied,
        initial_solution_provenance=provenance,
        warm_start_validation_config=WarmStartValidationConfig(),
    )

    assert result.feasible
    assert result.initial_customer_sequences == supplied
    assert result.candidate_control_statistics == {}


def test_none_preserves_current_stage052_candidate_transaction_path() -> None:
    result = solve_alns(
        _fixture_instance(),
        seed=2014,
        max_iterations=1,
        time_limit_seconds=2.0,
        screening_config=CheapScreeningConfig(),
        native_kernel_config=NativeKernelConfig(),
        candidate_transaction_config=NativeCandidateTransactionConfig(),
        native_execution_config=None,
    )

    assert result.feasible
    assert result.native_execution_statistics == {}
    assert result.candidate_control_statistics == {}


def test_native_candidate_round_is_one_structured_soa_call() -> None:
    from evrptw import _core as native_core

    instance = _fixture_instance()
    context = NativeKernelRuntime.build(instance, NativeKernelConfig()).context
    candidates = (("C1",), ("C2",), ("C2", "C1"))
    route_offsets = np.asarray([0, 1, 2, 4], dtype=np.int64)
    route_indices = np.asarray(
        [context.name_to_index[name] for route in candidates for name in route],
        dtype=np.int64,
    )
    candidate_ids = np.arange(len(candidates), dtype=np.int64)
    lexical_rank = np.asarray(
        [
            rank
            for rank, _name in sorted(
                enumerate(context.node_names),
                key=lambda item: item[1],
            )
        ],
        dtype=np.int64,
    )

    payload = native_core.candidate_round_transaction_v1(
        context.node_kind,
        context.demand,
        context.ready_time,
        context.due_date,
        context.service_time,
        context.distance,
        context.reachable,
        context.vehicle,
        route_offsets,
        route_indices,
        candidate_ids,
        lexical_rank,
        np.asarray([1.0, context.reachability_epsilon, 0.0, 0.0], dtype=np.float64),
        np.zeros((len(candidates), 6), dtype=np.float64),
        np.asarray([0], dtype=np.int64),
        np.asarray([], dtype=np.int64),
        np.asarray([], dtype=np.int64),
        np.asarray([1, 0, 0], dtype=np.int64),
        np.asarray([2, 1, 4], dtype=np.int64),
        np.asarray([10.0], dtype=np.float64),
        np.asarray([128], dtype=np.int64),
        np.asarray([7, 11, 13], dtype=np.int64),
    )

    assert isinstance(payload, tuple) and len(payload) == 10
    screening, resolutions, sources, cache_journal = payload[:4]
    exact_candidate_ids, completion_order, exact_payload = payload[4:7]
    counters, timings, transaction_sha256 = payload[7:]
    assert isinstance(screening, tuple) and len(screening) == 7
    assert resolutions.shape == (3,)
    assert sources.shape == (3,)
    assert cache_journal.shape == (3, 3)
    assert exact_candidate_ids.shape == (1,)
    assert np.array_equal(completion_order, exact_candidate_ids)
    assert isinstance(exact_payload, tuple) and len(exact_payload) == 7
    assert counters.shape == (10,)
    assert timings.shape == (4,)
    assert len(transaction_sha256) == 64
    assert set(transaction_sha256) <= set("0123456789abcdef")


def test_native_candidate_round_decodes_exact_results_and_records_one_invocation() -> None:
    instance = _fixture_instance()
    native_runtime = NativeKernelRuntime.build(instance, NativeKernelConfig())
    transaction_runtime = NativeCandidateTransactionRuntime(
        NativeCandidateTransactionConfig()
    )
    request = NativeCandidateRoundRequest(
        candidates=(("C1",), ("C2",), ("C2", "C1")),
        cache_hit_flags=(True, False, False),
        proposal_top_k=2,
        exact_budget=1,
        deadline=100.0,
        batch_size=128,
        lane="constraint",
        operator="relocate",
        iteration=7,
        compute_threads=4,
    )

    result = execute_native_candidate_round(
        instance,
        request,
        native_runtime=native_runtime,
        transaction_runtime=transaction_runtime,
        negative_cache={},
        clock=lambda: 90.0,
    )

    assert result.resolutions == ("cache_hit", "exact", "not_selected")
    assert result.exact_candidate_ids == (1,)
    assert len(result.exact_results) == 1 and result.exact_results[0].feasible
    assert result.completion_order == (1,)
    assert transaction_runtime.transaction_count == 1
    assert native_runtime.screening_batch_invocations == 1


def test_evaluator_per_solve_protocol_crosses_python_native_boundary_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw import _core as native_core

    instance = _fixture_instance()
    execution = _per_solve_config()
    native_runtime = NativeKernelRuntime.build(
        instance,
        execution.native_kernel_config,
    )
    transaction_runtime = NativeCandidateTransactionRuntime(
        execution.candidate_transaction_config
    )
    control_runtime = CandidateControlRuntime(execution.candidate_control_config)
    control_runtime.begin_round(7, lane="constraint")
    evaluator = _Evaluator(
        instance,
        deadline=time.perf_counter() + 10.0,
        lane="constraint",
        screening_config=CheapScreeningConfig(),
        backend="cpu_batch",
        candidate_control_runtime=control_runtime,
        candidate_transaction_runtime=transaction_runtime,
        native_runtime=native_runtime,
        native_execution_config=execution,
    )
    evaluator.set_measurement_context(
        lane="constraint",
        iteration=7,
        operator="relocate",
    )
    original = native_core.candidate_round_transaction_v2
    invocations = 0

    def counted(*args: object) -> object:
        nonlocal invocations
        invocations += 1
        return original(*args)

    def forbidden(*_args: object) -> object:
        raise AssertionError("per-solve protocol called a second native entry point")

    monkeypatch.setattr(native_core, "candidate_round_transaction_v2", counted)
    monkeypatch.setattr(native_core, "screen_route_batch_transaction_v2", forbidden)
    monkeypatch.setattr(native_core, "exact_charging_batch_numeric", forbidden)

    results = evaluator.candidate_route_batch(
        (("C1",), ("C2",), ("C2", "C1")),
        exact_budget=1,
    )

    assert invocations == 1
    assert len(results) == 3 and results[0].feasible
    assert results[1].failure_reason == "candidate_control:not_selected"
    assert results[2].failure_reason == "candidate_control:not_selected"
    assert transaction_runtime.transaction_count == 1
    assert transaction_runtime.protocol_invocations == 1
    assert control_runtime._pool is None


def test_per_solve_fixed_work_round_matches_python_candidate_control() -> None:
    instance = _fixture_instance()
    candidates = (("C1",), ("C2",), ("C2", "C1"))
    control_config = CandidateControlConfig(worker_count=1)
    python_control = CandidateControlRuntime(control_config)
    python_control.begin_round(7, lane="constraint")
    python_trace = Stage03Trace(MeasurementConfig())
    python_evaluator = _Evaluator(
        instance,
        deadline=time.perf_counter() + 10.0,
        lane="constraint",
        measurement_trace=python_trace,
        screening_config=CheapScreeningConfig(),
        backend="cpu_batch",
        candidate_control_runtime=python_control,
    )
    python_evaluator.set_measurement_context(
        lane="constraint",
        iteration=7,
        operator="relocate",
    )

    execution = _per_solve_config()
    native_control = CandidateControlRuntime(execution.candidate_control_config)
    native_control.begin_round(7, lane="constraint")
    native_trace = Stage03Trace(MeasurementConfig())
    native_evaluator = _Evaluator(
        instance,
        deadline=time.perf_counter() + 10.0,
        lane="constraint",
        measurement_trace=native_trace,
        screening_config=CheapScreeningConfig(),
        backend="cpu_batch",
        candidate_control_runtime=native_control,
        candidate_transaction_runtime=NativeCandidateTransactionRuntime(
            execution.candidate_transaction_config
        ),
        native_runtime=NativeKernelRuntime.build(
            instance,
            execution.native_kernel_config,
        ),
        native_execution_config=execution,
    )
    native_evaluator.set_measurement_context(
        lane="constraint",
        iteration=7,
        operator="relocate",
    )

    python_results = python_evaluator.candidate_route_batch(candidates)
    native_results = native_evaluator.candidate_route_batch(
        candidates,
        exact_budget=1,
    )

    assert [charging_result_semantic_payload(result) for result in native_results] == [
        charging_result_semantic_payload(result) for result in python_results
    ]
    assert native_control.candidate_work_hash == python_control.candidate_work_hash
    assert native_control.route_result_hash == python_control.route_result_hash
    assert [
        (event.route_key, event.exact_started, event.exact_completed, event.status)
        for event in native_trace.route_evaluations
    ] == [
        (event.route_key, event.exact_started, event.exact_completed, event.status)
        for event in python_trace.route_evaluations
    ]
    python_decisions = [
        event
        for event in python_control.events
        if event["event_type"]
        in {"candidate_control_decision", "candidate_control_decision_aggregate"}
    ]
    native_decisions = [
        event
        for event in native_control.events
        if event["event_type"]
        in {"candidate_control_decision", "candidate_control_decision_aggregate"}
    ]
    assert native_decisions == python_decisions


def test_per_solve_preserves_python_prescreened_ranking_semantics() -> None:
    instance = _candidate_control_ranking_fixture()
    candidates = (("C1",), ("C2",))
    control_config = CandidateControlConfig(worker_count=1)
    python_control = CandidateControlRuntime(control_config)
    python_evaluator = _Evaluator(
        instance,
        deadline=time.perf_counter() + 10.0,
        lane="initialization",
        screening_config=CheapScreeningConfig(),
        backend="cpu_batch",
        candidate_control_runtime=python_control,
    )

    execution = _per_solve_config()
    native_control = CandidateControlRuntime(execution.candidate_control_config)
    native_evaluator = _Evaluator(
        instance,
        deadline=time.perf_counter() + 10.0,
        lane="initialization",
        screening_config=CheapScreeningConfig(),
        backend="cpu_batch",
        candidate_control_runtime=native_control,
        candidate_transaction_runtime=NativeCandidateTransactionRuntime(
            execution.candidate_transaction_config
        ),
        native_runtime=NativeKernelRuntime.build(
            instance,
            execution.native_kernel_config,
        ),
        native_execution_config=execution,
    )

    python_results = python_evaluator.candidate_route_batch(
        candidates,
        prescreened=True,
    )
    native_results = native_evaluator.candidate_route_batch(
        candidates,
        prescreened=True,
        exact_budget=1,
    )

    assert [charging_result_semantic_payload(result) for result in native_results] == [
        charging_result_semantic_payload(result) for result in python_results
    ]
    assert native_control.candidate_work_hash == python_control.candidate_work_hash
    assert native_control.route_result_hash == python_control.route_result_hash


def test_native_candidate_round_hash_mismatch_fails_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw import _core as native_core

    instance = _fixture_instance()
    native_runtime = NativeKernelRuntime.build(instance, NativeKernelConfig())
    transaction_runtime = NativeCandidateTransactionRuntime(
        NativeCandidateTransactionConfig()
    )
    original = native_core.candidate_round_transaction_v2

    def corrupt_hash(*args: object) -> tuple[object, ...]:
        payload = original(*args)
        return (*payload[:-1], "0" * 64)

    monkeypatch.setattr(native_core, "candidate_round_transaction_v2", corrupt_hash)

    with pytest.raises(RuntimeError, match="transaction SHA-256 mismatch"):
        execute_native_candidate_round(
            instance,
            NativeCandidateRoundRequest(
                candidates=(("C1",),),
                cache_hit_flags=(False,),
                proposal_top_k=1,
                exact_budget=1,
                deadline=100.0,
                batch_size=128,
                lane="constraint",
                operator="relocate",
                iteration=7,
            ),
            native_runtime=native_runtime,
            transaction_runtime=transaction_runtime,
            negative_cache={},
            clock=lambda: 90.0,
        )

    assert transaction_runtime.transaction_count == 0
    assert transaction_runtime.fallback_count == 0
    assert native_runtime.screening_batch_invocations == 0


def test_native_candidate_round_cache_commit_failure_rolls_back_all_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _fixture_instance()
    execution = _per_solve_config()
    route_cache = RouteEvaluationCache(
        instance,
        CacheIncrementalConfig(enabled=True, max_entries=4),
    )
    route_cache.store(("C1",), solve_exact_charging(instance, ("C1",)))
    route_cache.store(("C2",), solve_exact_charging(instance, ("C2",)))
    cache_before = route_cache.snapshot_state()
    control_runtime = CandidateControlRuntime(execution.candidate_control_config)
    control_runtime.begin_round(7, lane="constraint")
    control_before = control_runtime.snapshot_protocol_state()
    transaction_runtime = NativeCandidateTransactionRuntime(
        execution.candidate_transaction_config
    )
    native_runtime = NativeKernelRuntime.build(
        instance,
        execution.native_kernel_config,
    )
    evaluator = _Evaluator(
        instance,
        deadline=time.perf_counter() + 10.0,
        lane="constraint",
        screening_config=CheapScreeningConfig(),
        cache_incremental_config=CacheIncrementalConfig(enabled=True),
        route_cache=route_cache,
        backend="cpu_batch",
        candidate_control_runtime=control_runtime,
        candidate_transaction_runtime=transaction_runtime,
        native_runtime=native_runtime,
        native_execution_config=execution,
    )
    evaluator.set_measurement_context(
        lane="constraint",
        iteration=7,
        operator="relocate",
    )

    def fail_commit() -> None:
        raise RuntimeError("injected cache commit failure")

    monkeypatch.setattr(evaluator, "_commit_pending_candidate_cache", fail_commit)

    with pytest.raises(RuntimeError, match="injected cache commit failure"):
        evaluator.candidate_route_batch((("C1",), ("C2",)), exact_budget=1)

    assert route_cache.snapshot_state() == cache_before
    assert control_runtime.snapshot_protocol_state() == control_before
    assert transaction_runtime.transaction_count == 0
    assert transaction_runtime.fallback_count == 0
    assert native_runtime.screening_batch_invocations == 0


def test_native_candidate_round_failure_does_not_refund_started_exact_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _fixture_instance()
    execution = _per_solve_config()
    route_cache = RouteEvaluationCache(
        instance,
        CacheIncrementalConfig(enabled=True, max_entries=4),
    )
    cache_before = route_cache.snapshot_state()
    control_runtime = CandidateControlRuntime(execution.candidate_control_config)
    control_runtime.begin_round(7, lane="constraint")
    exact_controller = ExactCallController(
        ExactDeadlineConfig.fixed_exact_calls(1, watchdog_seconds=120.0)
    )
    transaction_runtime = NativeCandidateTransactionRuntime(
        execution.candidate_transaction_config
    )
    native_runtime = NativeKernelRuntime.build(
        instance,
        execution.native_kernel_config,
    )
    evaluator = _Evaluator(
        instance,
        deadline=time.perf_counter() + 10.0,
        lane="constraint",
        screening_config=CheapScreeningConfig(),
        cache_incremental_config=CacheIncrementalConfig(enabled=True),
        route_cache=route_cache,
        backend="cpu_batch",
        exact_call_controller=exact_controller,
        candidate_control_runtime=control_runtime,
        candidate_transaction_runtime=transaction_runtime,
        native_runtime=native_runtime,
        native_execution_config=execution,
    )
    evaluator.set_measurement_context(
        lane="constraint",
        iteration=7,
        operator="relocate",
    )

    def fail_commit() -> None:
        raise RuntimeError("injected cache commit failure")

    monkeypatch.setattr(evaluator, "_commit_pending_candidate_cache", fail_commit)

    with pytest.raises(RuntimeError, match="injected cache commit failure"):
        evaluator.candidate_route_batch((("C1",),), exact_budget=1)

    assert route_cache.snapshot_state() == cache_before
    assert evaluator.pending_candidate_cache == {}
    assert exact_controller.started_calls == 1
    assert exact_controller.completed_calls == 1
    assert exact_controller.interrupted_calls == 0
    assert evaluator.backend_metrics.started_calls == 1
    assert evaluator.backend_metrics.completed_calls == 1
    assert evaluator.calls == 1
    control_after = control_runtime.snapshot_protocol_state()
    assert control_after.round_used == 1
    assert control_after.candidate_work_count == 0
    assert control_after.route_result_count == 0
    assert control_runtime.events[-1]["status"] == "aborted_transaction_consumed"
    assert transaction_runtime.transaction_count == 0
    assert transaction_runtime.fallback_count == 0
    assert native_runtime.screening_batch_invocations == 0


def test_native_candidate_round_one_and_four_threads_are_semantically_identical() -> None:
    instance = _fixture_instance()
    candidates = (("C1",), ("C2",), ("C2", "C1"))

    def execute(compute_threads: int) -> NativeCandidateRoundResult:
        return execute_native_candidate_round(
            instance,
            NativeCandidateRoundRequest(
                candidates=candidates,
                cache_hit_flags=(False, False, False),
                proposal_top_k=2,
                exact_budget=2,
                deadline=100.0,
                batch_size=128,
                lane="constraint",
                operator="relocate",
                iteration=7,
                compute_threads=compute_threads,
            ),
            native_runtime=NativeKernelRuntime.build(instance, NativeKernelConfig()),
            transaction_runtime=NativeCandidateTransactionRuntime(
                NativeCandidateTransactionConfig()
            ),
            negative_cache={},
            clock=lambda: 90.0,
        )

    serial = execute(1)
    parallel = execute(4)
    assert serial.resolutions == parallel.resolutions
    assert serial.exact_candidate_ids == parallel.exact_candidate_ids
    assert serial.completion_order == parallel.completion_order
    assert serial.transaction_sha256 == parallel.transaction_sha256
    assert [charging_result_semantic_payload(result) for result in serial.exact_results] == [
        charging_result_semantic_payload(result) for result in parallel.exact_results
    ]


@pytest.mark.parametrize("seed", [0, 1, 2014, 2014 ^ 0x5EED23, 2**32 + 17])
def test_native_python_random_matches_python313_call_sequence(seed: int) -> None:
    from evrptw import _core as native_core

    bounds = np.asarray([1, 2, 3, 17, 2**16, 2**32], dtype=np.int64)
    weights = np.asarray([0.25, 1.5, 0.0, 3.25], dtype=np.float64)
    native = native_core.python_random_golden_v1(
        seed,
        12,
        bounds,
        100,
        17,
        weights,
        25,
    )
    python = random.Random(seed)
    expected_random = [python.random() for _ in range(12)]
    expected_bounded = [python.randrange(int(bound)) for bound in bounds]
    expected_sample = python.sample(range(100), 17)
    expected_weighted = python.choices(range(len(weights)), weights=weights, k=1)[0]
    expected_shuffle = list(range(25))
    python.shuffle(expected_shuffle)

    assert native[0].tolist() == expected_random
    assert native[1].tolist() == expected_bounded
    assert native[2].tolist() == expected_sample
    assert native[3] == expected_weighted
    assert native[4].tolist() == expected_shuffle


@pytest.mark.parametrize(
    ("operation", "destroy_name"),
    ((0, "random"), (1, "worst"), (2, "related")),
)
@pytest.mark.parametrize("seed", (2014, 2015, 2016))
def test_native_legacy_destroy_matches_python(
    operation: int,
    destroy_name: str,
    seed: int,
) -> None:
    from evrptw import _core as native_core

    instance = _fixture_instance()
    sequences = (("C1", "C2"),)
    expected_partial, expected_removed = _destroy(
        instance,
        sequences,
        1,
        destroy_name,
        random.Random(seed),
    )
    context = NativeKernelRuntime.build(instance, NativeKernelConfig()).context
    offsets = np.asarray([0, 2], dtype=np.int64)
    indices = np.asarray(
        [context.name_to_index[name] for name in sequences[0]],
        dtype=np.int64,
    )
    ordered_names = sorted(context.node_names)
    rank_by_name = {name: rank for rank, name in enumerate(ordered_names)}
    lexical_rank = np.asarray(
        [rank_by_name[name] for name in context.node_names],
        dtype=np.int64,
    )
    partial_offsets, partial_indices, removed_indices = native_core.legacy_destroy_v2(
        seed,
        operation,
        1,
        offsets,
        indices,
        context.distance,
        lexical_rank,
        context.name_to_index[instance.depot.name],
    )
    actual_partial = tuple(
        tuple(
            context.node_names[int(index)]
            for index in partial_indices[
                int(partial_offsets[route]) : int(partial_offsets[route + 1])
            ]
        )
        for route in range(len(partial_offsets) - 1)
    )
    actual_removed = tuple(context.node_names[int(index)] for index in removed_indices)

    assert actual_partial == expected_partial
    assert actual_removed == expected_removed


def test_native_insertion_candidate_plans_match_controlled_python_enumeration() -> None:
    from evrptw import _core as native_core

    instance = _fixture_instance()
    context = NativeKernelRuntime.build(instance, NativeKernelConfig()).context
    customer = "C2"
    current_offsets = np.asarray([0, 1], dtype=np.int64)
    current_indices = np.asarray(
        [context.name_to_index["C1"]],
        dtype=np.int64,
    )
    plan_offsets, route_offsets, route_indices, metadata = (
        native_core.insertion_candidate_plans_v2(
            current_offsets,
            current_indices,
            context.name_to_index[customer],
            context.demand,
            instance.vehicle.load_capacity,
            1e-9,
        )
    )
    actual = tuple(
        tuple(
            tuple(
                context.node_names[int(index)]
                for index in route_indices[
                    int(route_offsets[route]) : int(route_offsets[route + 1])
                ]
            )
            for route in range(
                int(plan_offsets[plan]),
                int(plan_offsets[plan + 1]),
            )
        )
        for plan in range(len(plan_offsets) - 1)
    )
    expected = (
        (("C2", "C1"),),
        (("C1", "C2"),),
        (("C1",), ("C2",)),
    )

    assert actual == expected
    assert metadata.tolist() == [[0, 0], [0, 1], [1, 0]]


def test_full_native_initialization_matches_python_exact_objective_and_budget() -> None:
    from evrptw import _core as native_core

    instance = _fixture_instance()
    context = NativeKernelRuntime.build(instance, NativeKernelConfig()).context
    route = ("C1", "C2")
    route_offsets = np.asarray([0, 2], dtype=np.int64)
    route_indices = np.asarray(
        [context.name_to_index[name] for name in route],
        dtype=np.int64,
    )
    control = np.asarray([2014, 1, 128, 4, 100], dtype=np.int64)
    deadline = np.asarray([10.0], dtype=np.float64)

    payload = native_core.full_native_initialize_v2(
        context.node_kind,
        context.ready_time,
        context.due_date,
        context.service_time,
        context.distance,
        context.vehicle,
        route_offsets,
        route_indices,
        control,
        deadline,
    )
    expected = solve_exact_charging(instance, route)

    assert payload[1].tolist() == [1, 0]
    assert payload[2].tolist() == [expected.distance, expected.charging_time]
    assert payload[3].tolist() == [1, 1, 0, 0]

    insufficient = control.copy()
    insufficient[4] = 0
    with pytest.raises(RuntimeError, match="does not fit the exact-call budget"):
        native_core.full_native_initialize_v2(
            context.node_kind,
            context.ready_time,
            context.due_date,
            context.service_time,
            context.distance,
            context.vehicle,
            route_offsets,
            route_indices,
            insufficient,
            deadline,
        )


def test_native_objective_acceptance_matches_vehicle_first_python_policy() -> None:
    from evrptw import _core as native_core

    current = [
        SolutionObjective(3, 100.0, 10.0, 2),
        SolutionObjective(2, 100.0, 10.0, 2),
        SolutionObjective(2, 100.0, 10.0, 2),
        SolutionObjective(2, 100.0, 10.0, 2),
        SolutionObjective(2, 100.0, 10.0, 2),
    ]
    candidates = [
        SolutionObjective(2, 1000.0, 100.0, 20),
        SolutionObjective(3, 1.0, 0.0, 0),
        SolutionObjective(2, 99.0, 20.0, 5),
        SolutionObjective(2, 101.0, 10.0, 2),
        SolutionObjective(2, 100.0, 11.0, 2),
    ]
    temperatures = np.asarray([1.0, 1.0, 1.0, 10.0, 10.0], dtype=np.float64)
    draws = np.asarray([0.5, 0.5, 0.5, 0.01, 0.0], dtype=np.float64)

    observed = native_core.native_objective_acceptance_v1(
        np.asarray(
            [(item.vehicle_count, item.charging_count) for item in current],
            dtype=np.int64,
        ),
        np.asarray(
            [(item.total_distance, item.total_charging_time) for item in current],
            dtype=np.float64,
        ),
        np.asarray(
            [(item.vehicle_count, item.charging_count) for item in candidates],
            dtype=np.int64,
        ),
        np.asarray(
            [
                (item.total_distance, item.total_charging_time)
                for item in candidates
            ],
            dtype=np.float64,
        ),
        temperatures,
        draws,
    )
    expected = [
        accept_annealing_move(
            current_item,
            candidate_item,
            temperature=float(temperature),
            random_draw=float(draw),
        )
        for current_item, candidate_item, temperature, draw in zip(
            current,
            candidates,
            temperatures,
            draws,
            strict=True,
        )
    ]

    assert observed.tolist() == [int(value) for value in expected]


def test_native_stage04_segment_update_matches_python_config() -> None:
    from evrptw import _core as native_core

    config = Stage04Config()
    weights = np.asarray([1.0, 2.0, 0.2], dtype=np.float64)
    rewards = np.asarray([20.0, 1.0, 0.0], dtype=np.float64)
    calls = np.asarray([5, 4, 8], dtype=np.int64)
    updated, statuses = native_core.stage04_segment_update_v1(
        weights,
        rewards,
        calls,
        np.asarray(
            [
                config.weight_reaction,
                config.weight_floor,
                config.weight_smoothing,
                config.min_calls_per_operator,
            ],
            dtype=np.float64,
        ),
    )
    expected = [
        (
            config.apply_segment_update(float(weight), float(reward), int(call_count))
            if call_count >= config.min_calls_per_operator
            else float(weight)
        )
        for weight, reward, call_count in zip(weights, rewards, calls, strict=True)
    ]

    assert updated.tolist() == expected
    assert statuses.tolist() == [1, 0, 1]


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"abc",
        bytes(range(55)),
        bytes(range(56)),
        bytes(range(64)),
        bytes(range(255)) * 3,
    ],
)
def test_native_sha256_matches_hashlib_across_block_boundaries(payload: bytes) -> None:
    from evrptw import _core as native_core

    digest, hexadecimal = native_core.native_sha256_v1(
        np.frombuffer(payload, dtype=np.uint8),
    )
    expected = hashlib.sha256(payload).digest()

    assert digest.tobytes() == expected
    assert hexadecimal == expected.hex()


@pytest.mark.parametrize(
    ("operation", "generator_name"),
    [
        (0, "_relocate_candidates"),
        (1, "_swap_candidates"),
        (2, "_two_opt_star_candidates"),
    ],
)
def test_native_changed_candidate_pool_matches_python_order(
    operation: int,
    generator_name: str,
) -> None:
    from evrptw import _core as native_core
    from evrptw import neighborhoods

    sequences = ((1, 2), (3, 4, 5), (6,))
    offsets = np.asarray([0, 2, 5, 6], dtype=np.int64)
    indices = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
    payload = native_core.changed_candidate_pool_v1(operation, offsets, indices)
    changed_routes, change_offsets, change_indices, removed_offsets, removed_indices = (
        payload
    )
    observed: list[tuple[tuple[tuple[int, tuple[int, ...]], ...], tuple[int, ...]]] = []
    for candidate in range(len(changed_routes)):
        changes = tuple(
            (
                int(changed_routes[candidate, ordinal]),
                tuple(
                    int(value)
                    for value in change_indices[
                        int(change_offsets[candidate * 2 + ordinal]) : int(
                            change_offsets[candidate * 2 + ordinal + 1]
                        )
                    ]
                ),
            )
            for ordinal in range(2)
        )
        removed = tuple(
            int(value)
            for value in removed_indices[
                int(removed_offsets[candidate]) : int(removed_offsets[candidate + 1])
            ]
        )
        observed.append((changes, removed))
    generator = getattr(neighborhoods, generator_name)
    expected = [
        (candidate.changes, candidate.removed_customers)
        for candidate in generator(sequences)
    ]

    assert observed == expected

    plan_offsets, plan_route_offsets, plan_route_indices = (
        native_core.assemble_changed_candidate_plans_v1(
            offsets,
            indices,
            changed_routes,
            change_offsets,
            change_indices,
        )
    )
    observed_plans = tuple(
        tuple(
            tuple(
                int(value)
                for value in plan_route_indices[
                    int(plan_route_offsets[route]) : int(plan_route_offsets[route + 1])
                ]
            )
            for route in range(
                int(plan_offsets[candidate]), int(plan_offsets[candidate + 1])
            )
        )
        for candidate in range(len(observed))
    )
    expected_plans = tuple(
        neighborhoods._apply_changes(sequences, candidate.changes)
        for candidate in generator(sequences)
    )

    assert observed_plans == expected_plans


def test_native_candidate_plan_ranking_matches_python_rank_key() -> None:
    from evrptw import _core as native_core

    current = ((1, 2), (3,))
    plans = (
        ((1, 3), (2,)),
        ((1, 2), (3,)),
        ((1, 2, 3),),
        ((2, 1), (3,)),
    )
    per_route_lower_bounds = (
        (2.0, 1.0),
        (2.0, 1.0),
        (4.0,),
        (2.0, 1.0),
    )

    plan_offsets = [0]
    route_offsets = [0]
    route_indices: list[int] = []
    lower_bounds: list[float] = []
    for plan, plan_bounds in zip(plans, per_route_lower_bounds, strict=True):
        for route, route_bound in zip(plan, plan_bounds, strict=True):
            route_indices.extend(route)
            route_offsets.append(len(route_indices))
            lower_bounds.append(route_bound)
        plan_offsets.append(len(route_offsets) - 1)
    current_offsets = [0]
    current_indices: list[int] = []
    for route in current:
        current_indices.extend(route)
        current_offsets.append(len(current_indices))
    attempted = np.asarray([0, 1, 0, 0], dtype=np.int64)
    ranked, selected, integer_metrics, float_metrics = (
        native_core.rank_candidate_plans_v1(
            np.asarray(plan_offsets, dtype=np.int64),
            np.asarray(route_offsets, dtype=np.int64),
            np.asarray(route_indices, dtype=np.int64),
            np.asarray(lower_bounds, dtype=np.float64),
            np.asarray(current_offsets, dtype=np.int64),
            np.asarray(current_indices, dtype=np.int64),
            np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
            attempted,
            2,
        )
    )
    current_set = set(current)
    rank_keys = tuple(
        (
            len(plan),
            sum(plan_bounds),
            sum(route not in current_set for route in plan),
            plan,
            ordinal,
        )
        for ordinal, (plan, plan_bounds) in enumerate(
            zip(plans, per_route_lower_bounds, strict=True)
        )
    )
    expected_ranked = sorted(range(len(plans)), key=rank_keys.__getitem__)
    expected_selected = [
        ordinal for ordinal in expected_ranked if not attempted[ordinal]
    ][:2]

    assert ranked.tolist() == expected_ranked
    assert selected.tolist() == expected_selected
    assert integer_metrics.tolist() == [
        [len(plan), sum(route not in current_set for route in plan)] for plan in plans
    ]
    assert float_metrics.tolist() == [sum(bounds) for bounds in per_route_lower_bounds]


@pytest.mark.parametrize(
    ("operation", "generator_name"),
    [(0, "_relocate_candidates"), (1, "_swap_candidates"), (2, "_two_opt_star_candidates")],
)
def test_native_changed_plan_selection_matches_python_screen_and_rank(
    operation: int,
    generator_name: str,
) -> None:
    from evrptw import _core as native_core
    from evrptw import neighborhoods

    instance = _candidate_plan_fixture()
    context = NativeKernelRuntime.build(instance, NativeKernelConfig()).context
    sequences = (("C1", "C2"), ("C3", "C4"))
    route_offsets = np.asarray([0, 2, 4], dtype=np.int64)
    route_indices = np.asarray([1, 2, 3, 4], dtype=np.int64)
    generator = getattr(neighborhoods, generator_name)
    descriptions = tuple(generator(sequences))
    plans = tuple(
        neighborhoods._apply_changes(sequences, description.changes)
        for description in descriptions
    )
    attempted = np.zeros(len(plans), dtype=np.int64)
    if len(attempted) > 1:
        attempted[1] = 1
    payload = native_core.changed_candidate_plan_selection_v1(
        operation,
        context.node_kind,
        context.demand,
        context.ready_time,
        context.due_date,
        context.service_time,
        context.distance,
        context.reachable,
        context.vehicle,
        np.arange(len(context.node_names), dtype=np.int64),
        route_offsets,
        route_indices,
        np.asarray([1.0, context.reachability_epsilon, 0.0, 0.0], dtype=np.float64),
        np.asarray([0], dtype=np.int64),
        np.asarray([], dtype=np.int64),
        np.asarray([], dtype=np.int64),
        attempted,
        2,
        4,
    )
    eligible = payload[3]
    ranked = payload[4]
    selected = payload[5]
    integer_metrics = payload[6]
    float_metrics = payload[7]
    current_set = set(sequences)
    screens = tuple(
        tuple(
            neighborhoods.screen_route_candidate(instance, route, full=True)
            for route in plan
        )
        for plan in plans
    )
    expected_eligible = [int(all(screen.accepted for screen in plan)) for plan in screens]
    expected_keys = tuple(
        (
            len(plan),
            sum(screen.distance_lower_bound for screen in plan_screens),
            sum(route not in current_set for route in plan),
            plan,
            ordinal,
        )
        for ordinal, (plan, plan_screens) in enumerate(zip(plans, screens, strict=True))
    )
    expected_ranked = sorted(
        (index for index, valid in enumerate(expected_eligible) if valid),
        key=expected_keys.__getitem__,
    )
    expected_selected = [index for index in expected_ranked if not attempted[index]][:2]

    assert eligible.tolist() == expected_eligible
    assert ranked.tolist() == expected_ranked
    assert selected.tolist() == expected_selected
    assert integer_metrics.tolist() == [
        [len(plan), sum(route not in current_set for route in plan)] for plan in plans
    ]
    assert float_metrics.tolist() == [key[1] for key in expected_keys]
    assert isinstance(payload[8], str) and len(payload[8]) == 64


def test_native_attempted_plan_identity_is_transactional_across_ordinals() -> None:
    from evrptw import _core as native_core

    # Plan 0 and plan 2 are identical but have different proposal ordinals.
    plans = (((1, 2), (3,)), ((1, 3), (2,)), ((1, 2), (3,)))
    plan_offsets = [0]
    route_offsets = [0]
    route_indices: list[int] = []
    for plan in plans:
        for route in plan:
            route_indices.extend(route)
            route_offsets.append(len(route_indices))
        plan_offsets.append(len(route_offsets) - 1)
    packed_plans = np.asarray(plan_offsets, dtype=np.int64)
    packed_routes = np.asarray(route_offsets, dtype=np.int64)
    packed_indices = np.asarray(route_indices, dtype=np.int64)
    attempted = native_core.NativeAttemptedPlanSetV2()

    assert attempted.lookup(packed_plans, packed_routes, packed_indices).tolist() == [0, 0, 0]
    statuses = attempted.begin_mark_many_atomic(
        packed_plans,
        packed_routes,
        packed_indices,
        np.asarray([0, 1], dtype=np.int64),
    )
    assert statuses.tolist() == [1, 1]
    assert attempted.lookup(packed_plans, packed_routes, packed_indices).tolist() == [1, 1, 1]
    assert attempted.rollback_mark_batch() == 0
    assert attempted.lookup(packed_plans, packed_routes, packed_indices).tolist() == [0, 0, 0]

    attempted.begin_mark_many_atomic(
        packed_plans,
        packed_routes,
        packed_indices,
        np.asarray([2], dtype=np.int64),
    )
    assert attempted.commit_mark_batch() == 1
    assert attempted.lookup(packed_plans, packed_routes, packed_indices).tolist() == [1, 0, 1]


def test_native_attempted_plan_invalid_batch_is_atomic() -> None:
    from evrptw import _core as native_core

    plan_offsets = np.asarray([0, 2, 4], dtype=np.int64)
    route_offsets = np.asarray([0, 2, 3, 5, 6], dtype=np.int64)
    route_indices = np.asarray([1, 2, 3, 1, 3, 2], dtype=np.int64)
    attempted = native_core.NativeAttemptedPlanSetV2()

    with pytest.raises(ValueError, match="unique valid plan rows"):
        attempted.begin_mark_many_atomic(
            plan_offsets,
            route_offsets,
            route_indices,
            np.asarray([0, 99], dtype=np.int64),
        )

    assert attempted.size() == 0
    assert attempted.lookup(plan_offsets, route_offsets, route_indices).tolist() == [0, 0]


def test_native_route_cache_atomic_lru_matches_python_cache() -> None:
    from evrptw import _core as native_core

    instance = _fixture_instance()
    config = CacheIncrementalConfig(
        enabled=True,
        max_entries=2,
        max_memory_bytes=1_000_000,
    )
    python_cache = RouteEvaluationCache(instance, config)
    native_cache = native_core.NativeRouteCacheV2(
        config.max_entries,
        config.max_memory_bytes,
    )
    routes = (("C1",), ("C2",), ("C1", "C2"))
    results = tuple(solve_exact_charging(instance, route) for route in routes)
    node_index = {node.name: index for index, node in enumerate(instance.nodes)}

    def packed(selected: tuple[tuple[str, ...], ...]) -> tuple[np.ndarray, np.ndarray]:
        offsets = [0]
        indices: list[int] = []
        for route in selected:
            indices.extend(node_index[name] for name in route)
            offsets.append(len(indices))
        return (
            np.asarray(offsets, dtype=np.int64),
            np.asarray(indices, dtype=np.int64),
        )

    def hashes(selected_results: tuple[object, ...]) -> np.ndarray:
        return np.asarray(
            [
                list(bytes.fromhex(charging_result_semantic_digest(result)))
                for result in selected_results
            ],
            dtype=np.uint8,
        )

    def entry_sizes(selected_results: tuple[object, ...]) -> np.ndarray:
        return np.asarray(
            [estimate_cache_entry_bytes(result) for result in selected_results],
            dtype=np.int64,
        )

    initial_offsets, initial_indices = packed(routes[:2])
    python_cache.store_many_atomic(tuple(zip(routes[:2], results[:2], strict=True)))
    native_statuses, native_evictions, _ = native_cache.begin_store_many_atomic(
        initial_offsets,
        initial_indices,
        hashes(results[:2]),
        entry_sizes(results[:2]),
    )
    native_cache.commit_store_batch()
    assert native_statuses.tolist() == [0, 0]
    assert native_evictions.tolist() == [0, 0]

    first_offsets, first_indices = packed((routes[0],))
    assert python_cache.lookup(routes[0]).hit
    native_hits, native_hashes, native_statistics = native_cache.lookup_many(
        first_offsets,
        first_indices,
    )
    assert native_hits.tolist() == [1]
    assert native_hashes[0].tolist() == hashes((results[0],))[0].tolist()
    assert native_statistics.tolist() == list(python_cache.statistics.to_dict().values())


    third_offsets, third_indices = packed((routes[2],))
    python_batch = python_cache.begin_store_many_atomic(((routes[2], results[2]),))
    native_statuses, native_evictions, _ = native_cache.begin_store_many_atomic(
        third_offsets,
        third_indices,
        hashes((results[2],)),
        entry_sizes((results[2],)),
    )
    assert native_statuses.tolist() == [0]
    assert native_evictions.tolist() == [1]
    python_cache.rollback_store_batch(python_batch)
    native_statistics = native_cache.rollback_store_batch()
    assert native_statistics.tolist() == list(python_cache.statistics.to_dict().values())

    snapshot = native_cache.snapshot()
    snapshot_offsets, snapshot_indices, snapshot_hashes, snapshot_bytes, _ = snapshot
    native_routes = tuple(
        tuple(
            int(value)
            for value in snapshot_indices[
                int(snapshot_offsets[index]) : int(snapshot_offsets[index + 1])
            ]
        )
        for index in range(len(snapshot_offsets) - 1)
    )
    python_routes = tuple(
        tuple(node_index[name] for name in key.customer_sequence)
        for key in python_cache._entries
    )
    assert native_routes == python_routes
    expected_by_route = dict(zip(routes, results, strict=True))
    ordered_results = tuple(
        expected_by_route[key.customer_sequence] for key in python_cache._entries
    )
    assert snapshot_hashes.tolist() == hashes(ordered_results).tolist()
    assert snapshot_bytes.tolist() == entry_sizes(ordered_results).tolist()

    with pytest.raises(RuntimeError, match="semantic conflict"):
        python_cache.begin_store_many_atomic(((routes[0], results[1]),))
    with pytest.raises(RuntimeError, match="semantic conflict"):
        native_cache.begin_store_many_atomic(
            first_offsets,
            first_indices,
            hashes((results[1],)),
            entry_sizes((results[1],)),
        )

    python_batch = python_cache.begin_store_many_atomic(((routes[2], results[2]),))
    python_cache.commit_store_batch(python_batch)
    native_cache.begin_store_many_atomic(
        third_offsets,
        third_indices,
        hashes((results[2],)),
        entry_sizes((results[2],)),
    )
    native_statistics = native_cache.commit_store_batch()
    assert native_statistics.tolist() == list(python_cache.statistics.to_dict().values())


def test_native_route_cache_restores_typed_exact_payload_on_hit() -> None:
    from evrptw import _core as native_core

    instance = _fixture_instance()
    routes = (("C1",), ("C2",), ("C1", "C2"))
    results = tuple(solve_exact_charging(instance, route) for route in routes)
    node_index = {node.name: index for index, node in enumerate(instance.nodes)}

    def packed(selected: tuple[tuple[str, ...], ...]) -> tuple[np.ndarray, np.ndarray]:
        offsets = [0]
        indices: list[int] = []
        for route in selected:
            indices.extend(node_index[name] for name in route)
            offsets.append(len(indices))
        return np.asarray(offsets, dtype=np.int64), np.asarray(indices, dtype=np.int64)

    selected = routes[:2]
    selected_results = results[:2]
    route_offsets, route_indices = packed(selected)
    path_offsets, path_indices = packed(
        tuple(tuple(result.route) for result in selected_results)
    )
    metrics = np.asarray(
        [
            [
                result.distance,
                result.total_energy,
                result.charged_energy,
                result.charging_time,
            ]
            for result in selected_results
        ],
        dtype=np.float64,
    )
    labels = np.asarray(
        [
            [result.labels_generated, result.labels_expanded, result.labels_pruned]
            for result in selected_results
        ],
        dtype=np.int64,
    )
    semantic_hashes = np.asarray(
        [
            list(bytes.fromhex(charging_result_semantic_digest(result)))
            for result in selected_results
        ],
        dtype=np.uint8,
    )
    entry_bytes = np.asarray(
        [estimate_cache_entry_bytes(result) for result in selected_results],
        dtype=np.int64,
    )
    cache = native_core.NativeRouteCacheV2(2, 1_000_000)
    statuses, evictions, _ = cache.begin_store_exact_many_atomic(
        route_offsets,
        route_indices,
        path_offsets,
        path_indices,
        np.zeros(2, dtype=np.int64),
        np.zeros(2, dtype=np.int64),
        metrics,
        labels,
        semantic_hashes,
        entry_bytes,
    )
    assert statuses.tolist() == [0, 0]
    assert evictions.tolist() == [0, 0]
    cache.commit_store_batch()

    state_before = cache.snapshot()
    cache.begin_protocol_transaction()
    cache.lookup_exact_many(*packed((routes[0],)))
    third_route_offsets, third_route_indices = packed((routes[2],))
    third_path_offsets, third_path_indices = packed((tuple(results[2].route),))
    third_metrics = np.asarray(
        [[
            results[2].distance,
            results[2].total_energy,
            results[2].charged_energy,
            results[2].charging_time,
        ]],
        dtype=np.float64,
    )
    third_labels = np.asarray(
        [[
            results[2].labels_generated,
            results[2].labels_expanded,
            results[2].labels_pruned,
        ]],
        dtype=np.int64,
    )
    third_hash = np.asarray(
        [list(bytes.fromhex(charging_result_semantic_digest(results[2])))],
        dtype=np.uint8,
    )
    third_bytes = np.asarray(
        [estimate_cache_entry_bytes(results[2])],
        dtype=np.int64,
    )
    cache.begin_store_exact_many_atomic(
        third_route_offsets,
        third_route_indices,
        third_path_offsets,
        third_path_indices,
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        third_metrics,
        third_labels,
        third_hash,
        third_bytes,
    )
    cache.commit_store_batch()
    cache.lookup_exact_many(third_route_offsets, third_route_indices)
    cache.rollback_protocol_transaction()
    state_after = cache.snapshot()
    assert [item.tolist() for item in state_after] == [
        item.tolist() for item in state_before
    ]

    lookup_offsets, lookup_indices = packed((routes[1], routes[2]))
    payload = cache.lookup_exact_many(lookup_offsets, lookup_indices)
    (
        hits,
        cached_path_offsets,
        cached_path_indices,
        cached_statuses,
        cached_reasons,
        cached_metrics,
        cached_labels,
        cached_hashes,
        _statistics,
    ) = payload
    assert hits.tolist() == [1, 0]
    assert cached_path_offsets.tolist() == [0, len(results[1].route), len(results[1].route)]
    assert cached_path_indices.tolist() == [
        node_index[name] for name in results[1].route
    ]
    assert cached_statuses.tolist() == [0, -1]
    assert cached_reasons.tolist() == [0, -1]
    assert cached_metrics[0].tolist() == metrics[1].tolist()
    assert cached_labels[0].tolist() == labels[1].tolist()
    assert cached_hashes[0].tolist() == semantic_hashes[1].tolist()

    conflicting_metrics = metrics.copy()
    conflicting_metrics[0, 0] += 1.0
    with pytest.raises(RuntimeError, match="exact payload conflict"):
        cache.begin_store_exact_many_atomic(
            route_offsets,
            route_indices,
            path_offsets,
            path_indices,
            np.zeros(2, dtype=np.int64),
            np.zeros(2, dtype=np.int64),
            conflicting_metrics,
            labels,
            semantic_hashes,
            entry_bytes,
        )


def test_native_negative_route_cache_generation_matches_python() -> None:
    from evrptw import _core as native_core

    python_cache = BoundedNegativeSequenceCache(capacity=2)
    native_cache = native_core.NativeNegativeRouteCacheV2(2)
    node_index = {"C1": 1, "C2": 2, "C3": 3}

    def packed(selected: tuple[tuple[str, ...], ...]) -> tuple[np.ndarray, np.ndarray]:
        offsets = [0]
        indices: list[int] = []
        for route in selected:
            indices.extend(node_index[name] for name in route)
            offsets.append(len(indices))
        return (
            np.asarray(offsets, dtype=np.int64),
            np.asarray(indices, dtype=np.int64),
        )

    def expected_statistics() -> list[int]:
        statistics = python_cache.statistics()
        return [
            int(statistics["stores"]),
            int(statistics["evictions"]),
            int(statistics["rollovers"]),
            int(statistics["current_entries"]),
            int(statistics["peak_entries"]),
        ]

    initial_routes = (("C1",), ("C2",))
    initial_offsets, initial_indices = packed(initial_routes)
    python_batch = python_cache.begin_store_many_atomic(
        {initial_routes[0]: "capacity_prefilter", initial_routes[1]: "energy_prefilter"}
    )
    native_summary = native_cache.begin_store_many_atomic(
        initial_offsets,
        initial_indices,
        np.asarray([2, 7], dtype=np.int64),
    )
    assert native_summary.tolist() == [2, 0, 0]
    python_cache.commit_store_batch(python_batch)
    assert native_cache.commit_store_batch().tolist() == expected_statistics()

    first_offsets, first_indices = packed((initial_routes[0],))
    hits, reasons, statistics = native_cache.lookup_many(first_offsets, first_indices)
    assert hits.tolist() == [1]
    assert reasons.tolist() == [2]
    assert statistics.tolist() == expected_statistics()

    with pytest.raises(RuntimeError, match="reason changed"):
        python_cache.begin_store_many_atomic({initial_routes[0]: "energy_prefilter"})
    with pytest.raises(RuntimeError, match="reason changed"):
        native_cache.begin_store_many_atomic(
            first_offsets,
            first_indices,
            np.asarray([7], dtype=np.int64),
        )

    replacement = (("C3",),)
    replacement_offsets, replacement_indices = packed(replacement)
    python_batch = python_cache.begin_store_many_atomic(
        {replacement[0]: "forward_time_window_prefilter"}
    )
    native_summary = native_cache.begin_store_many_atomic(
        replacement_offsets,
        replacement_indices,
        np.asarray([3], dtype=np.int64),
    )
    assert native_summary.tolist() == [1, 2, 1]
    python_cache.rollback_store_batch(python_batch)
    assert native_cache.rollback_store_batch().tolist() == expected_statistics()

    snapshot_offsets, snapshot_indices, snapshot_reasons, snapshot_statistics = (
        native_cache.snapshot()
    )
    assert snapshot_offsets.tolist() == [0, 1, 2]
    assert snapshot_indices.tolist() == [1, 2]
    assert snapshot_reasons.tolist() == [2, 7]
    assert snapshot_statistics.tolist() == expected_statistics()

    python_batch = python_cache.begin_store_many_atomic(
        {replacement[0]: "forward_time_window_prefilter"}
    )
    python_cache.commit_store_batch(python_batch)
    native_cache.begin_store_many_atomic(
        replacement_offsets,
        replacement_indices,
        np.asarray([3], dtype=np.int64),
    )
    assert native_cache.commit_store_batch().tolist() == expected_statistics()
    hits, reasons, _ = native_cache.lookup_many(replacement_offsets, replacement_indices)
    assert hits.tolist() == [1]
    assert reasons.tolist() == [3]


def test_native_budget_state_matches_python_controllers_and_rollback() -> None:
    from evrptw import _core as native_core

    exact = ExactCallController(
        ExactDeadlineConfig.fixed_exact_calls(5, watchdog_seconds=120.0)
    )
    candidate = CandidateControlRuntime(
        CandidateControlConfig(max_exact_calls_per_round=2, worker_count=1)
    )
    native = native_core.NativeBudgetStateV2(5, 2)

    candidate.begin_round(7, lane="quality_shadow")
    state = native.begin_round(4, 7)
    assert state.tolist()[:5] == [1, 4, 7, 0, 2]
    assert native.exact_remaining() == 5
    assert native.candidate_round_remaining() == 2
    assert candidate.reserve(3, atomic=True, context="plan") == 0
    assert native.reserve_round(3, True).tolist() == [3, 0, 2]
    assert candidate.reserve(1, atomic=True, context="plan") == 1
    assert native.reserve_round(1, True).tolist() == [1, 1, 1]

    exact_reservation = exact.reserve(3)
    native_reservation = native.reserve_exact(3)
    assert native_reservation.tolist() == [
        exact_reservation.requested,
        exact_reservation.granted,
    ]
    exact.complete(2)
    exact.interrupt(1)
    native.complete_exact(2)
    native.interrupt_exact(1)
    snapshot = native.snapshot().copy()
    candidate_snapshot = candidate.snapshot_protocol_state()
    assert snapshot.tolist() == [1, 4, 7, 1, 1, 3, 2, 1, 0]
    assert native.exact_remaining() == 2
    assert native.candidate_round_remaining() == 1

    candidate.reserve(1, atomic=False, context="temporary")
    native.reserve_round(1, False)
    native.reserve_exact(2)
    candidate.rollback_protocol_state(candidate_snapshot)
    state = native.restore(snapshot)
    assert candidate.round_remaining == int(state[4]) == 1
    assert state.tolist() == snapshot.tolist()

    exact_reservation = exact.reserve(4)
    native_reservation = native.reserve_exact(4)
    assert native_reservation.tolist() == [
        exact_reservation.requested,
        exact_reservation.granted,
    ]
    exact.complete(2)
    state = native.complete_exact(2)
    assert state.tolist()[5:] == [
        exact.started_calls,
        exact.completed_calls,
        exact.interrupted_calls,
        exact.budget_exhaustions,
    ]
    with pytest.raises(RuntimeError, match="invalid completed"):
        native.complete_exact(2)

    candidate.finish_round()
    assert native.finish_round().tolist()[:5] == [0, -1, -1, 0, 2]


def test_exact_budget_states_reject_double_classification() -> None:
    from evrptw import _core as native_core

    exact = ExactCallController(
        ExactDeadlineConfig.fixed_exact_calls(1, watchdog_seconds=120.0)
    )
    native = native_core.NativeBudgetStateV2(1, 1)

    assert exact.reserve(1).granted == 1
    assert native.reserve_exact(1).tolist() == [1, 1]
    exact.interrupt(1)
    native.interrupt_exact(1)

    with pytest.raises(RuntimeError, match="invalid completed exact-call count"):
        exact.complete(1)
    with pytest.raises(RuntimeError, match="invalid completed exact-call count"):
        native.complete_exact(1)


@pytest.mark.external_data
@pytest.mark.skipif(
    os.environ.get("EVRPTW_RUN_NATIVE_REAL_DIFFERENTIAL") != "1",
    reason="opt-in real 100-customer native differential gate",
)
@pytest.mark.parametrize(
    ("instance_name", "seed"),
    [
        (instance_name, seed)
        for instance_name in ("c101C5", "c101_21", "r101_21", "rc101_21")
        for seed in (2014, 2015, 2016)
    ],
)
def test_per_solve_real_fixed_work_matches_four_worker_python_control(
    instance_name: str,
    seed: int,
) -> None:
    instance_path = Path(f"data/schneider/{instance_name}.txt")
    if not instance_path.exists():
        pytest.skip("Schneider benchmark data are not linked")
    instance = replace(parse_schneider(instance_path), distance_backend="native")
    control = CandidateControlConfig(worker_count=4)
    common = {
        "seed": seed,
        "max_iterations": 1000,
        "time_limit_seconds": 120.0,
        "operator_profile": "stage02_constraint_guided",
        "measurement_config": MeasurementConfig(),
        "screening_config": CheapScreeningConfig(),
        "cache_incremental_config": CacheIncrementalConfig(enabled=True),
        "backend": "cpu_batch",
        "batch_size": 128,
        "termination_mode": "fixed_work",
        "exact_deadline_config": ExactDeadlineConfig.fixed_exact_calls(
            100,
            watchdog_seconds=120.0,
        ),
        "stage04_config": Stage04Config(),
    }

    python_result = solve_alns(
        instance,
        **common,  # type: ignore[arg-type]
        candidate_control_config=control,
    )
    native_result = solve_alns(
        instance,
        **common,  # type: ignore[arg-type]
        native_execution_config=replace(
            _per_solve_config(),
            candidate_control_config=control,
        ),
    )

    assert python_result.feasible and native_result.feasible
    assert native_result.objective == python_result.objective
    assert native_result.customer_sequences == python_result.customer_sequences
    assert native_result.candidate_work_hash == python_result.candidate_work_hash
    assert native_result.route_result_hash == python_result.route_result_hash
    assert native_result.exact_started_calls == python_result.exact_started_calls
    assert native_result.exact_completed_calls == python_result.exact_completed_calls
    telemetry_fields = {
        "failure_reasons",
        "prefilter_passed",
        "prefilter_rejected",
    }
    for operator, python_statistics in python_result.neighborhood_statistics.items():
        native_statistics = native_result.neighborhood_statistics[operator]
        assert {
            key: value
            for key, value in native_statistics.items()
            if key not in telemetry_fields
        } == {
            key: value
            for key, value in python_statistics.items()
            if key not in telemetry_fields
        }, operator
    assert _semantic_candidate_trajectory(native_result) == _semantic_candidate_trajectory(
        python_result
    )
    assert native_result.stage04_statistics == python_result.stage04_statistics
    assert native_result.stage04_event_log == python_result.stage04_event_log
    assert native_result.measurement_trace is not None
    assert python_result.measurement_trace is not None
    assert [
        (
            row.route_key,
            row.kind,
            row.exact_started,
            row.exact_completed,
            row.cache_key_digest,
        )
        for row in native_result.measurement_trace.route_evaluations
    ] == [
        (
            row.route_key,
            row.kind,
            row.exact_started,
            row.exact_completed,
            row.cache_key_digest,
        )
        for row in python_result.measurement_trace.route_evaluations
    ]


def test_full_native_v2_packs_one_complete_soa_and_refuses_prototype_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw import _core as native_core

    instance = _fixture_instance()
    original = native_core.full_native_alns_v2
    invocations = 0

    def counted(*args: object) -> object:
        nonlocal invocations
        invocations += 1
        return original(*args)

    monkeypatch.setattr(native_core, "full_native_alns_v2", counted)
    with pytest.raises(
        RuntimeError,
        match="semantic engine is incomplete; refusing prototype fallback",
    ):
        solve_alns(
            instance,
            seed=2014,
            max_iterations=1,
            time_limit_seconds=2.0,
            **_full_native_solve_kwargs(),  # type: ignore[arg-type]
            native_execution_config=_native_config("full_native_alns"),
        )

    assert invocations == 1


def test_host_scheduler_v1_cannot_masquerade_as_full_native_v2(
    tmp_path: Path,
) -> None:
    instance = _fixture_instance()
    endpoint = tmp_path / "native-scheduler.sock"
    with (
        NativeHostScheduler(endpoint),
        pytest.raises(RuntimeError, match="invalid SoA descriptor set"),
    ):
        solve_alns(
            instance,
            seed=2014,
            max_iterations=10,
            time_limit_seconds=2.0,
            **_full_native_solve_kwargs(),  # type: ignore[arg-type]
            native_execution_config=replace(
                _native_config("host_scheduler"),
                scheduler_socket_path=str(endpoint),
            ),
        )


def test_host_scheduler_exit_fails_fast_without_local_fallback(tmp_path: Path) -> None:
    endpoint = tmp_path / "native-scheduler.sock"
    scheduler = NativeHostScheduler(endpoint)
    scheduler.start()
    scheduler.close(force=True)

    with pytest.raises(RuntimeError, match="IPC failed without fallback"):
        solve_alns(
            _fixture_instance(),
            seed=2014,
            max_iterations=1,
            time_limit_seconds=2.0,
            **_full_native_solve_kwargs(),  # type: ignore[arg-type]
            native_execution_config=replace(
                _native_config("host_scheduler"),
                scheduler_socket_path=str(endpoint),
            ),
        )


def test_host_scheduler_partial_ipc_rolls_back_and_keeps_service_usable(
    tmp_path: Path,
) -> None:
    endpoint = tmp_path / "native-scheduler.sock"
    with NativeHostScheduler(endpoint):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(endpoint))
            connection.sendall(struct.pack("!Q", 100) + b"{}")

        with pytest.raises(RuntimeError, match="invalid SoA descriptor set"):
            solve_alns(
                _fixture_instance(),
                seed=2014,
                max_iterations=1,
                time_limit_seconds=2.0,
                **_full_native_solve_kwargs(),  # type: ignore[arg-type]
                native_execution_config=replace(
                    _native_config("host_scheduler"),
                    scheduler_socket_path=str(endpoint),
                ),
            )


def test_host_scheduler_start_failure_cleans_process_state(tmp_path: Path) -> None:
    endpoint = tmp_path / "native-scheduler.sock"
    scheduler = NativeHostScheduler(endpoint, worker_threads=23)

    with pytest.raises(RuntimeError, match="exited before becoming ready"):
        scheduler.start()

    assert not scheduler.is_running
    assert not endpoint.exists()
