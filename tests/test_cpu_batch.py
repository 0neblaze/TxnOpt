from __future__ import annotations

import itertools
import pickle
import random
from dataclasses import replace

import numpy as np
import pytest

from evrptw import _core as native_core
from evrptw.alns import solve_alns
from evrptw.cache_incremental import CacheIncrementalConfig
from evrptw.charging import solve_exact_charging
from evrptw.cpu_batch import (
    BackendMetrics,
    ExactBatchDeadlineExceeded,
    ExactChargingBackend,
    solve_exact_charging_batch,
)
from evrptw.exact_deadline import ExactDeadlineConfig
from evrptw.measurement import CheapScreeningConfig, MeasurementConfig
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.native_kernels import NativeKernelConfig, NativeKernelRuntime


def _instance() -> Instance:
    return Instance(
        "cpu_batch_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("F1", NodeType.STATION, 4.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("F2", NodeType.STATION, 0.0, 4.0, 0.0, 0.0, 200.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 8.0, 0.0, 1.0, 0.0, 100.0, 1.0),
            Node("C2", NodeType.CUSTOMER, 8.0, 4.0, 1.0, 0.0, 100.0, 1.0),
        ),
        Vehicle(20.0, 5.0, 1.0, 0.1, 1.0),
    )


def _signature(result: object) -> tuple[object, ...]:
    charging = result  # type: ignore[assignment]
    return (
        charging.feasible,
        charging.route,
        charging.distance,
        charging.total_energy,
        charging.charged_energy,
        charging.charging_time,
        charging.labels_generated,
        charging.labels_expanded,
        charging.labels_pruned,
        charging.failure_reason,
    )


def test_exact_batch_deadline_exception_round_trips_across_process_boundary() -> None:
    metrics = BackendMetrics(
        "cpu_batch",
        8,
        started_calls=3,
        completed_calls=2,
        interrupted_calls=1,
    )
    original = ExactBatchDeadlineExceeded(
        started_exact_calls=3,
        completed_exact_calls=2,
        metrics=metrics,
        completed_indices=(0, 2),
    )

    restored = pickle.loads(pickle.dumps(original))

    assert isinstance(restored, ExactBatchDeadlineExceeded)
    assert restored.started_exact_calls == 3
    assert restored.completed_exact_calls == 2
    assert restored.interrupted_exact_calls == 1
    assert restored.metrics.to_dict() == metrics.to_dict()
    assert restored.completed_indices == (0, 2)


def test_cpu_batch_matches_scalar_and_exposes_only_cpu_backends() -> None:
    instance = _instance()
    orders = (("C1",), ("C2",), ("C1", "C2"), ("C2", "C1"))

    batch = solve_exact_charging_batch(instance, orders, batch_size=2)

    assert tuple(ExactChargingBackend) == (
        ExactChargingBackend.CPU_SCALAR,
        ExactChargingBackend.CPU_BATCH,
    )
    assert batch.metrics.backend == "cpu_batch"
    assert batch.metrics.native_invocations == 0
    assert batch.metrics.native_fallbacks == 0
    assert batch.metrics.work_batches == 1
    assert batch.metrics.batch_launches == 1
    assert batch.metrics.launch_occupancies == [len(orders)]
    assert batch.metrics.transition_batches > 0
    assert batch.metrics.transitions > 0
    assert [_signature(result) for result in batch.results] == [
        _signature(solve_exact_charging(instance, order)) for order in orders
    ]


def test_backend_metrics_reject_non_reconciling_launch_occupancies_atomically() -> None:
    aggregate = BackendMetrics("cpu_batch", 128)
    invalid = BackendMetrics(
        "cpu_batch",
        128,
        exact_calls=2,
        batch_launches=1,
        launch_occupancies=[1],
    )

    with pytest.raises(ValueError, match="reconcile"):
        aggregate.add(invalid)

    assert aggregate.exact_calls == 0
    assert aggregate.batch_launches == 0
    assert aggregate.launch_occupancies == []


def test_cpu_batch_preserves_request_order_and_boundary_results() -> None:
    instance = _instance()
    station_instance = Instance(
        "cpu_batch_station_edge",
        tuple(node for node in instance.nodes if node.name != "C2"),
        Vehicle(10.0, 5.0, 1.0, 0.1, 1.0),
    )
    tight_window_instance = Instance(
        "cpu_batch_time_edge",
        tuple(
            replace(node, due_date=3.0) if node.name == "F1" else node
            for node in station_instance.nodes
        ),
        station_instance.vehicle,
    )
    capacity_instance = Instance(
        "cpu_batch_capacity_edge",
        station_instance.nodes,
        Vehicle(10.0, 0.5, 1.0, 0.1, 1.0),
    )
    cases = (
        (station_instance, (("C1",),)),
        (instance, (("C1", "C1"), ("unknown",))),
        (tight_window_instance, (("C1",),)),
        (capacity_instance, (("C1",),)),
    )

    for edge_instance, orders in cases:
        actual = solve_exact_charging_batch(edge_instance, orders, batch_size=32)
        expected = tuple(solve_exact_charging(edge_instance, order) for order in orders)
        assert [_signature(result) for result in actual.results] == [
            _signature(result) for result in expected
        ]


def test_native_exact_batch_matches_python_for_randomized_ordered_work() -> None:
    instance = _instance()
    rng = random.Random(2014)
    orders = list(itertools.permutations(("C1", "C2")))
    orders.extend([("C1",), ("C2",), (), ("C1", "C1"), ("unknown",)])
    rng.shuffle(orders)

    native = solve_exact_charging_batch(
        instance,
        tuple(orders),
        batch_size=3,
        native_kernel_config=NativeKernelConfig(),
    )
    expected = tuple(solve_exact_charging(instance, order) for order in orders)

    assert [_signature(result) for result in native.results] == [
        _signature(result) for result in expected
    ]
    assert native.metrics.native_invocations == 1
    assert native.metrics.native_fallbacks == 0
    assert native.metrics.native_kernel_seconds > 0.0
    assert native.metrics.packing_seconds > 0.0
    assert native.metrics.completed_calls == len(orders)
    assert native.metrics.launch_occupancies == [len(orders)]


def test_native_runtime_reuses_one_instance_context() -> None:
    instance = _instance()
    runtime = NativeKernelRuntime.build(instance, NativeKernelConfig())

    first = solve_exact_charging_batch(instance, (("C1",),), native_runtime=runtime)
    second = solve_exact_charging_batch(instance, (("C2",),), native_runtime=runtime)

    assert first.metrics.packing_seconds > second.metrics.packing_seconds
    assert first.metrics.native_invocations == second.metrics.native_invocations == 1
    assert first.metrics.native_fallbacks == second.metrics.native_fallbacks == 0


def test_native_exact_batch_rejects_invalid_payload_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = 0

    def invalid_payload(*_args: object) -> tuple[np.ndarray[object, object], ...]:
        nonlocal called
        called += 1
        return (np.array([], dtype=np.int64),)

    monkeypatch.setattr(native_core, "exact_charging_batch_numeric", invalid_payload)
    with pytest.raises(RuntimeError, match="invalid payload tuple"):
        solve_exact_charging_batch(
            _instance(),
            (("C1",),),
            native_kernel_config=NativeKernelConfig(),
        )
    assert called == 1


def test_native_exact_batch_propagates_native_failure_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NativeFailure(RuntimeError):
        pass

    def fail(*_args: object) -> None:
        raise NativeFailure("native sentinel")

    monkeypatch.setattr(native_core, "exact_charging_batch_numeric", fail)
    with pytest.raises(NativeFailure, match="native sentinel"):
        solve_exact_charging_batch(
            _instance(),
            (("C1",),),
            native_kernel_config=NativeKernelConfig(),
        )


def test_cpu_batch_deadline_fails_before_unbounded_transition_work() -> None:
    try:
        solve_exact_charging_batch(
            _instance(),
            (("C1",), ("C2",)),
            deadline=0.0,
        )
    except ExactBatchDeadlineExceeded as error:
        assert error.completed_exact_calls == 0
    else:
        raise AssertionError("expired CPU batch deadline did not fail fast")


def test_native_batch_expired_deadline_never_invokes_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args: object) -> None:
        raise AssertionError("expired deadline invoked native work")

    monkeypatch.setattr(native_core, "exact_charging_batch_numeric", unexpected)
    with pytest.raises(ExactBatchDeadlineExceeded) as caught:
        solve_exact_charging_batch(
            _instance(),
            (("C1",), ("C2",)),
            deadline=0.0,
            native_kernel_config=NativeKernelConfig(),
        )
    assert caught.value.started_exact_calls == 2
    assert caught.value.completed_exact_calls == 0


def test_native_partial_deadline_maps_completed_indices_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = solve_exact_charging(_instance(), ("C1",))
    names = {node.name: index for index, node in enumerate(_instance().nodes)}

    def partial(*_args: object) -> tuple[np.ndarray[object, object], ...]:
        return (
            np.array([0, 3, 3], dtype=np.int64),
            np.array([names[name] for name in expected.route], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
            np.array([0, 2], dtype=np.int64),
            np.array(
                [
                    [
                        expected.distance,
                        expected.total_energy,
                        expected.charged_energy,
                        expected.charging_time,
                    ],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    [
                        expected.labels_generated,
                        expected.labels_expanded,
                        expected.labels_pruned,
                    ],
                    [1, 0, 0],
                ],
                dtype=np.int64,
            ),
            np.array([2, 2, 1, 1, 1, 1, 1, 1, 1, 128], dtype=np.int64),
        )

    monkeypatch.setattr(native_core, "exact_charging_batch_numeric", partial)
    with pytest.raises(ExactBatchDeadlineExceeded) as caught:
        solve_exact_charging_batch(
            _instance(),
            (("C1",), ("C1", "C1"), ("C2",)),
            batch_size=128,
            native_kernel_config=NativeKernelConfig(),
        )
    assert caught.value.started_exact_calls == 3
    assert caught.value.completed_exact_calls == 2
    assert caught.value.completed_indices == (0, 1)
    assert caught.value.metrics.native_invocations == 1
    assert caught.value.metrics.native_fallbacks == 0


def test_cpu_batch_preserves_fixed_work_with_local_cache_enabled() -> None:
    common = {
        "seed": 2014,
        "max_iterations": 5,
        "time_limit_seconds": 10.0,
        "operator_profile": "stage02_constraint_guided",
        "measurement_config": MeasurementConfig(),
        "termination_mode": "fixed_work",
    }

    scalar = solve_alns(_instance(), backend="cpu_scalar", batch_size=1, **common)
    batch = solve_alns(_instance(), backend="cpu_batch", batch_size=128, **common)

    assert scalar.effective_iterations == batch.effective_iterations == 5
    assert scalar.charging_subproblem_calls == batch.charging_subproblem_calls
    assert scalar.cache_hits == batch.cache_hits
    assert scalar.objective is not None and batch.objective is not None
    assert scalar.objective.key == batch.objective.key
    assert int(batch.backend_metrics["work_batches"]) < batch.charging_subproblem_calls
    assert sum(batch.backend_metrics["launch_occupancies"]) == int(
        batch.backend_metrics["exact_calls"]
    )
    assert scalar.measurement_trace is not None and batch.measurement_trace is not None
    scalar_work = [
        (record.lane, record.iteration, record.operator, record.route_key, record.kind)
        for record in scalar.measurement_trace.route_evaluations
    ]
    batch_work = [
        (record.lane, record.iteration, record.operator, record.route_key, record.kind)
        for record in batch.measurement_trace.route_evaluations
    ]
    assert scalar_work == batch_work


def test_native_alns_fixed_work_matches_python_and_records_no_fallback() -> None:
    common = {
        "seed": 2014,
        "max_iterations": 5,
        "time_limit_seconds": 10.0,
        "operator_profile": "stage02_constraint_guided",
        "measurement_config": MeasurementConfig(),
        "termination_mode": "fixed_work",
        "backend": "cpu_batch",
    }
    python = solve_alns(_instance(), **common)
    native = solve_alns(
        _instance(),
        native_kernel_config=NativeKernelConfig(),
        **common,
    )

    assert python.effective_iterations == native.effective_iterations == 5
    assert python.charging_subproblem_calls == native.charging_subproblem_calls
    assert python.objective is not None and native.objective is not None
    assert python.objective.key == native.objective.key
    assert int(native.backend_metrics["native_invocations"]) > 0
    assert int(native.backend_metrics["native_fallbacks"]) == 0
    assert len(native.backend_metrics["launch_occupancies"]) == int(
        native.backend_metrics["batch_launches"]
    )
    assert sum(native.backend_metrics["launch_occupancies"]) == int(
        native.backend_metrics["exact_calls"]
    )


def test_native_exact_budget_boundary_matches_python_without_partial_acceptance() -> None:
    common = {
        "seed": 2014,
        "max_iterations": 1000,
        "time_limit_seconds": 10.0,
        "operator_profile": "stage02_constraint_guided",
        "measurement_config": MeasurementConfig(),
        "backend": "cpu_batch",
        "disable_cache": True,
        "exact_deadline_config": ExactDeadlineConfig.fixed_exact_calls(
            10,
            watchdog_seconds=10.0,
        ),
    }

    python = solve_alns(_instance(), **common)
    native = solve_alns(
        _instance(),
        native_kernel_config=NativeKernelConfig(),
        **common,
    )

    for result in (python, native):
        assert result.exact_started_calls == 10
        assert result.exact_completed_calls == 10
        assert result.exact_interrupted_calls == 0
        assert result.exact_budget_exhaustions == 1
        assert result.termination_reason == "exact_call_budget_exhausted"
    assert python.objective is not None and native.objective is not None
    assert python.objective.key == native.objective.key
    assert python.effective_iterations == native.effective_iterations


def test_solve_alns_defaults_to_reviewed_cpu_batch_backend() -> None:
    common = {
        "seed": 2014,
        "max_iterations": 2,
        "time_limit_seconds": 10.0,
        "operator_profile": "stage02_constraint_guided",
        "termination_mode": "fixed_work",
    }

    default = solve_alns(_instance(), **common)
    explicit = solve_alns(_instance(), backend="cpu_batch", **common)

    assert default.charging_backend == "cpu_batch"
    assert default.objective is not None and explicit.objective is not None
    assert default.objective.key == explicit.objective.key


def test_cpu_batch_batches_stage032_screened_cache_misses() -> None:
    common = {
        "seed": 2014,
        "max_iterations": 5,
        "time_limit_seconds": 10.0,
        "operator_profile": "stage02_constraint_guided",
        "measurement_config": MeasurementConfig(),
        "screening_config": CheapScreeningConfig(),
        "cache_incremental_config": CacheIncrementalConfig(
            enabled=True,
            max_entries=128,
            max_memory_bytes=1_000_000,
        ),
        "termination_mode": "fixed_work",
    }

    scalar = solve_alns(_instance(), backend="cpu_scalar", batch_size=1, **common)
    batch = solve_alns(_instance(), backend="cpu_batch", batch_size=128, **common)

    assert scalar.effective_iterations == batch.effective_iterations == 5
    assert scalar.charging_subproblem_calls == batch.charging_subproblem_calls
    assert scalar.objective is not None and batch.objective is not None
    assert scalar.objective.key == batch.objective.key
    assert int(batch.backend_metrics["work_batches"]) < batch.charging_subproblem_calls
    assert batch.measurement_trace is not None
    assert batch.measurement_trace.reconcile(batch)["status"] == "pass"


def test_cpu_batch_preserves_bounded_lru_semantics() -> None:
    common = {
        "seed": 2014,
        "max_iterations": 5,
        "time_limit_seconds": 10.0,
        "operator_profile": "stage02_constraint_guided",
        "measurement_config": MeasurementConfig(),
        "screening_config": CheapScreeningConfig(),
        "cache_incremental_config": CacheIncrementalConfig(
            enabled=True,
            max_entries=1,
            max_memory_bytes=1_000_000,
        ),
        "termination_mode": "fixed_work",
    }

    scalar = solve_alns(_instance(), backend="cpu_scalar", batch_size=1, **common)
    batch = solve_alns(_instance(), backend="cpu_batch", batch_size=128, **common)

    assert scalar.charging_subproblem_calls == batch.charging_subproblem_calls
    scalar_cache = scalar.cache_incremental_statistics["route_cache"]
    batch_cache = batch.cache_incremental_statistics["route_cache"]
    assert isinstance(scalar_cache, dict) and isinstance(batch_cache, dict)
    for field in (
        "cache_lookups",
        "cache_hits",
        "cache_misses",
        "cache_stores",
        "cache_evictions",
        "entries_current",
    ):
        assert scalar_cache[field] == batch_cache[field]
    assert scalar.objective is not None and batch.objective is not None
    assert scalar.objective.key == batch.objective.key
