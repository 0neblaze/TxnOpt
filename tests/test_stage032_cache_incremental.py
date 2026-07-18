from __future__ import annotations

from dataclasses import replace

import pytest

from evrptw.alns import solve_alns
from evrptw.cache_incremental import (
    CacheIncrementalConfig,
    RouteEvaluationCache,
    StationReachabilityIndex,
    build_route_propagation_snapshot,
    canonical_instance_hash,
    estimate_cache_entry_bytes,
    incremental_route_propagation,
)
from evrptw.charging import ChargingSubproblemResult
from evrptw.experiments.stage03_measurement import _validate_stage032_run_label
from evrptw.measurement import CheapScreeningConfig, MeasurementConfig, Stage03Trace
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.native_kernels import NativeKernelConfig, NativeKernelRuntime
from evrptw.neighborhoods import repair_vehicle_reduction_refinement


def _instance(*, due_c3: float = 100.0, battery: float = 5.0) -> Instance:
    return Instance(
        "stage032_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("F1", NodeType.STATION, 3.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 100.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 100.0, 0.0),
            Node("C3", NodeType.CUSTOMER, 6.0, 0.0, 1.0, 0.0, due_c3, 0.0),
        ),
        Vehicle(battery, 10.0, 1.0, 1.0, 1.0),
    )


def _result(*, feasible: bool) -> ChargingSubproblemResult:
    return ChargingSubproblemResult(
        feasible=feasible,
        route=("D0", "C1", "D0") if feasible else (),
        distance=2.0 if feasible else float("inf"),
        total_energy=2.0 if feasible else 0.0,
        charged_energy=0.0,
        charging_time=0.0,
        labels_generated=1,
        labels_expanded=1,
        labels_pruned=0,
        runtime_seconds=0.0,
        failure_reason="" if feasible else "known infeasible",
    )


def test_route_cache_key_contains_instance_sequence_configuration_and_objective() -> None:
    instance = _instance()
    config = CacheIncrementalConfig(
        enabled=True,
        instance_hash="instance-a",
        charging_configuration_version="charging-a",
    )
    cache = RouteEvaluationCache(instance, config)

    key = cache.make_key(("C1", "C2"))

    assert key.instance_hash == "instance-a"
    assert key.customer_sequence == ("C1", "C2")
    assert key.charging_configuration_version == "charging-a"
    assert key.objective_schema_version == "vehicles,distance,charging_time,charging_count"
    assert key.digest != cache.make_key(("C2", "C1")).digest
    assert (
        key.digest
        != RouteEvaluationCache(
            instance,
            replace(config, instance_hash="instance-b"),
        )
        .make_key(("C1", "C2"))
        .digest
    )


def test_stage032_runner_labels_are_canonical_and_non_overlapping() -> None:
    _validate_stage032_run_label("stage03.2_cache_incremental_attempt01")
    _validate_stage032_run_label("stage03.2_cache_incremental_rerun99")

    with pytest.raises(ValueError):
        _validate_stage032_run_label("stage032_cache_incremental_attempt01")


def test_route_cache_reuses_feasible_and_infeasible_results() -> None:
    instance = _instance()
    cache = RouteEvaluationCache(
        instance,
        CacheIncrementalConfig(enabled=True, instance_hash=canonical_instance_hash(instance)),
    )

    first = cache.lookup(("C1",))
    cache.store(("C1",), _result(feasible=True))
    feasible_hit = cache.lookup(("C1",))
    cache.lookup(("C2",))
    cache.store(("C2",), _result(feasible=False))
    infeasible_hit = cache.lookup(("C2",))

    assert not first.hit
    assert feasible_hit.hit and feasible_hit.result is not None and feasible_hit.result.feasible
    assert infeasible_hit.hit and infeasible_hit.result is not None
    assert not infeasible_hit.result.feasible
    assert cache.statistics.hits == 2
    assert cache.statistics.misses == 2


def test_cache_entry_size_excludes_volatile_runtime() -> None:
    result = _result(feasible=True)

    assert estimate_cache_entry_bytes(result) == estimate_cache_entry_bytes(
        replace(result, runtime_seconds=123.456789)
    )


def test_route_cache_lru_eviction_is_observable_and_bounded() -> None:
    instance = _instance()
    cache = RouteEvaluationCache(
        instance,
        CacheIncrementalConfig(enabled=True, max_entries=1, max_memory_bytes=10_000),
    )

    cache.store(("C1",), _result(feasible=True))
    cache.store(("C2",), _result(feasible=True))

    assert not cache.contains(("C1",))
    assert cache.contains(("C2",))
    assert cache.statistics.evictions == 1
    assert cache.statistics.entries_current == 1
    assert cache.statistics.bytes_current <= cache.config.max_memory_bytes


def test_station_reachability_bitset_matches_safe_frontier_example() -> None:
    index = StationReachabilityIndex(_instance(battery=4.0))

    assert index.can_reach("D0", "C3") is True
    assert index.can_reach("C1", "C3") is True
    assert index.to_dict()["origin_bitsets"]["C1"] & 2
    assert index.bitset_for("D0") & 1


def test_incremental_relocation_propagation_reuses_prefix_and_matches_distance() -> None:
    instance = _instance()
    base = build_route_propagation_snapshot(instance, ("C1", "C2"))

    result = incremental_route_propagation(instance, base, ("C1", "C3"))

    assert result.status == "incremental"
    assert result.accepted
    assert result.distance_lower_bound == 12.0
    assert result.reused_prefix_edges >= 1
    assert result.recomputed_forward_edges > 0


def test_incremental_propagation_reports_time_window_failure_and_fallback() -> None:
    instance = _instance(due_c3=2.0)
    base = build_route_propagation_snapshot(instance, ("C1", "C2"))

    rejected = incremental_route_propagation(instance, base, ("C1", "C3"))
    fallback = incremental_route_propagation(instance, base, ("C1", "C9"))

    assert not rejected.accepted
    assert rejected.first_failed_check == "forward_time_window"
    assert fallback.status == "fallback"
    assert fallback.first_failed_check == "route_structure"


def test_native_incremental_wrapper_matches_python_and_records_invocation() -> None:
    instance = _instance()
    base = build_route_propagation_snapshot(instance, ("C1", "C2"))
    runtime = NativeKernelRuntime.build(instance, NativeKernelConfig())

    expected = incremental_route_propagation(instance, base, ("C1", "C3"))
    actual = incremental_route_propagation(
        instance,
        base,
        ("C1", "C3"),
        native_runtime=runtime,
    )

    assert actual == expected
    assert runtime.propagation_invocations == 1
    assert runtime.fallback_count == 0


def test_native_incremental_wrapper_fails_fast_when_core_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _instance()
    base = build_route_propagation_snapshot(instance, ("C1", "C2"))
    runtime = NativeKernelRuntime.build(instance, NativeKernelConfig())

    def fail(_function_name: str) -> object:
        raise RuntimeError("native propagation failed")

    monkeypatch.setattr("evrptw.cache_incremental._require_native_core", fail)

    with pytest.raises(RuntimeError, match="native propagation failed"):
        incremental_route_propagation(
            instance,
            base,
            ("C1", "C3"),
            native_runtime=runtime,
        )

    assert runtime.fallback_count == 0


def test_native_incremental_context_is_reused_across_calls() -> None:
    instance = _instance()
    base = build_route_propagation_snapshot(instance, ("C1", "C2"))
    runtime = NativeKernelRuntime.build(instance, NativeKernelConfig())
    context_identity = id(runtime.context)

    incremental_route_propagation(instance, base, ("C1", "C3"), native_runtime=runtime)
    incremental_route_propagation(instance, base, ("C2", "C3"), native_runtime=runtime)

    assert id(runtime.context) == context_identity
    assert runtime.propagation_invocations == 2


def test_refinement_uses_precomputed_current_route_without_unchanged_exact_call() -> None:
    instance = _instance(battery=20.0)

    class FakeEvaluator:
        calls = 0

        def __init__(self) -> None:
            self.statuses: list[tuple[tuple[str, ...], str]] = []

        def route_with_status(
            self, sequence: tuple[str, ...], status: str
        ) -> ChargingSubproblemResult:
            self.calls += 1
            self.statuses.append((sequence, status))
            return _result(feasible=True)

    evaluator = FakeEvaluator()
    base = ("C1", "C2")
    result = repair_vehicle_reduction_refinement(
        (base,),
        ("C3",),
        evaluator,
        instance,
        budget=8,
        precomputed_routes={base: _result(feasible=True)},
    )

    assert result.sequences is not None
    assert evaluator.calls > 0
    assert (base, "unchanged") not in evaluator.statuses


def test_stage032_solve_trace_round_trip_and_result_reconciliation() -> None:
    result = solve_alns(
        _instance(battery=20.0),
        seed=2014,
        max_iterations=12,
        time_limit_seconds=1.0,
        operator_profile="stage02_constraint_guided",
        measurement_config=MeasurementConfig(),
        screening_config=CheapScreeningConfig(),
        cache_incremental_config=CacheIncrementalConfig(
            enabled=True,
            max_entries=64,
            max_memory_bytes=1_000_000,
        ),
    )

    assert result.feasible
    assert result.measurement_trace is not None
    trace = result.measurement_trace
    assert trace.trace_schema_version == "stage03-trace-v2"
    assert trace.cache_incremental_config is not None
    assert trace.cache_incremental_counts["cache_lookups"] > 0
    assert trace.precomputed_routes > 0
    assert all(
        evaluation.route_change_status in {"changed", "unchanged"}
        for evaluation in trace.route_evaluations
    )
    assert trace.reconcile(result)["status"] == "pass"

    restored = Stage03Trace.from_dict(trace.to_dict())
    assert restored.trace_schema_version == "stage03-trace-v2"
    assert restored.cache_incremental_counts == trace.cache_incremental_counts
    assert restored.route_dictionary == trace.route_dictionary


def test_stage032_disabled_configuration_preserves_uninstrumented_result_shape() -> None:
    result = solve_alns(
        _instance(battery=20.0),
        seed=2014,
        max_iterations=8,
        time_limit_seconds=1.0,
        operator_profile="baseline",
        cache_incremental_config=CacheIncrementalConfig(enabled=False),
    )

    assert result.feasible
    assert result.measurement_trace is None
    assert result.cache_incremental_statistics == {}
