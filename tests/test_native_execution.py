from __future__ import annotations

import os
import random
import socket
import struct
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from evrptw.alns import _Evaluator, solve_alns
from evrptw.cache_incremental import (
    CacheIncrementalConfig,
    RouteEvaluationCache,
    charging_result_semantic_payload,
)
from evrptw.candidate_control import CandidateControlConfig, CandidateControlRuntime
from evrptw.candidate_transaction import (
    NativeCandidateTransactionConfig,
    NativeCandidateTransactionRuntime,
)
from evrptw.charging import solve_exact_charging
from evrptw.exact_deadline import ExactDeadlineConfig
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
from evrptw.parser import parse_schneider
from evrptw.stage04 import Stage04Config


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
        ("full_native_alns", "full_solve_soa_v1"),
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


def test_full_native_alns_calls_cpp_once_and_matches_python_minimal_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw import _core as native_core

    instance = _fixture_instance()
    python_result = solve_alns(
        instance,
        seed=2014,
        max_iterations=1,
        time_limit_seconds=2.0,
        screening_config=CheapScreeningConfig(),
        candidate_control_config=CandidateControlConfig(worker_count=1),
    )
    original = native_core.full_native_alns_v1
    invocations = 0

    def counted(*args: object) -> object:
        nonlocal invocations
        invocations += 1
        return original(*args)

    monkeypatch.setattr(native_core, "full_native_alns_v1", counted)
    native_result = solve_alns(
        instance,
        seed=2014,
        max_iterations=1,
        time_limit_seconds=2.0,
        screening_config=CheapScreeningConfig(),
        native_execution_config=_native_config("full_native_alns"),
    )

    assert invocations == 1
    assert native_result.routes == python_result.routes
    assert native_result.customer_sequences == python_result.customer_sequences
    assert native_result.objective == python_result.objective
    assert native_result.native_execution_statistics["mode"] == "full_native_alns"
    assert native_result.native_execution_statistics["fallback_count"] == 0


def test_host_scheduler_uses_uds_shared_memory_and_matches_direct_full_native(
    tmp_path: Path,
) -> None:
    instance = _fixture_instance()
    direct = solve_alns(
        instance,
        seed=2014,
        max_iterations=10,
        time_limit_seconds=2.0,
        screening_config=CheapScreeningConfig(),
        native_execution_config=_native_config("full_native_alns"),
    )
    endpoint = tmp_path / "native-scheduler.sock"
    with NativeHostScheduler(endpoint):
        scheduled = solve_alns(
            instance,
            seed=2014,
            max_iterations=10,
            time_limit_seconds=2.0,
            screening_config=CheapScreeningConfig(),
            native_execution_config=replace(
                _native_config("host_scheduler"),
                scheduler_socket_path=str(endpoint),
            ),
        )

    assert scheduled.objective == direct.objective
    assert scheduled.routes == direct.routes
    assert scheduled.neighborhood_events == direct.neighborhood_events
    assert scheduled.candidate_work_hash == direct.candidate_work_hash
    assert scheduled.native_execution_statistics["mode"] == "host_scheduler"
    assert scheduled.native_execution_statistics["fallback_count"] == 0


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
            screening_config=CheapScreeningConfig(),
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

        result = solve_alns(
            _fixture_instance(),
            seed=2014,
            max_iterations=1,
            time_limit_seconds=2.0,
            screening_config=CheapScreeningConfig(),
            native_execution_config=replace(
                _native_config("host_scheduler"),
                scheduler_socket_path=str(endpoint),
            ),
        )

    assert result.feasible
    assert result.native_execution_statistics["fallback_count"] == 0


def test_host_scheduler_start_failure_cleans_process_state(tmp_path: Path) -> None:
    endpoint = tmp_path / "native-scheduler.sock"
    scheduler = NativeHostScheduler(endpoint, worker_threads=23)

    with pytest.raises(RuntimeError, match="exited before becoming ready"):
        scheduler.start()

    assert not scheduler.is_running
    assert not endpoint.exists()
