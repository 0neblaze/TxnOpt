from __future__ import annotations

from dataclasses import replace

import pytest

from evrptw.alns import solve_alns
from evrptw.charging import solve_exact_charging
from evrptw.gpu_batch import (
    ExactChargingBackend,
    LabelBufferOverflowError,
    MetalUnavailableError,
    solve_exact_charging_batch,
)
from evrptw.measurement import MeasurementConfig
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.validation import validate_routes


def _instance() -> Instance:
    return Instance(
        "gpu_batch_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("F1", NodeType.STATION, 4.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("F2", NodeType.STATION, 0.0, 4.0, 0.0, 0.0, 200.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 8.0, 0.0, 1.0, 0.0, 100.0, 1.0),
            Node("C2", NodeType.CUSTOMER, 8.0, 4.0, 1.0, 0.0, 100.0, 1.0),
        ),
        Vehicle(20.0, 5.0, 1.0, 0.1, 1.0),
    )


def _single_customer_instance() -> Instance:
    instance = _instance()
    return Instance(
        instance.name,
        tuple(node for node in instance.nodes if node.name != "C2"),
        instance.vehicle,
    )


def _result_signature(result: object) -> tuple[object, ...]:
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


def test_cpu_batch_matches_scalar_exact_result_and_validator() -> None:
    instance = _instance()
    orders = (("C1",), ("C2",), ("C1", "C2"), ("C2", "C1"))

    batch = solve_exact_charging_batch(
        instance,
        orders,
        backend=ExactChargingBackend.CPU_BATCH,
        batch_size=2,
    )

    assert batch.metrics.backend == ExactChargingBackend.CPU_BATCH.value
    assert batch.metrics.batch_launches > 0
    assert batch.metrics.transitions > 0
    assert len(batch.results) == len(orders)
    for order, actual in zip(orders, batch.results, strict=True):
        expected = solve_exact_charging(instance, order)
        assert _result_signature(actual) == _result_signature(expected)
        if actual.feasible and set(order) == {customer.name for customer in instance.customers}:
            assert validate_routes(instance, [list(actual.route)]).feasible

    single = solve_exact_charging_batch(
        _single_customer_instance(),
        (("C1",),),
        backend=ExactChargingBackend.CPU_BATCH,
    )
    assert validate_routes(_single_customer_instance(), [list(single.results[0].route)]).feasible


def test_cpu_batch_preserves_request_order_for_feasible_and_infeasible_routes() -> None:
    instance = _instance()
    orders = (("C1",), ("C1", "C1"), ("unknown",))

    result = solve_exact_charging_batch(
        instance,
        orders,
        backend="cpu_batch",
        batch_size=128,
    )

    assert result.results[0].feasible is True
    assert result.results[1].failure_reason == "customer order contains duplicates"
    assert "non-customer" in result.results[2].failure_reason


def test_batch_label_capacity_overflow_fails_without_cpu_fallback() -> None:
    with pytest.raises(LabelBufferOverflowError):
        solve_exact_charging_batch(
            _instance(),
            (("C1",), ("C2",)),
            backend=ExactChargingBackend.CPU_BATCH,
            batch_size=128,
            label_buffer_capacity=1,
        )


def test_metal_backend_is_explicit_and_never_silently_falls_back() -> None:
    try:
        result = solve_exact_charging_batch(
            _instance(),
            (("C1",),),
            backend=ExactChargingBackend.METAL_BATCH,
            batch_size=32,
        )
    except MetalUnavailableError:
        return
    assert result.metrics.backend == ExactChargingBackend.METAL_BATCH.value
    assert result.results[0].feasible is True


def test_fixed_work_alns_keeps_candidate_work_and_objective_identical() -> None:
    common = {
        "seed": 2014,
        "max_iterations": 5,
        "time_limit_seconds": 10.0,
        "operator_profile": "stage02_constraint_guided",
        "measurement_config": MeasurementConfig(),
        "termination_mode": "fixed_work",
        "disable_cache": True,
    }
    scalar = solve_alns(_instance(), backend="cpu_scalar", batch_size=1, **common)
    batch = solve_alns(_instance(), backend="cpu_batch", batch_size=128, **common)

    assert scalar.feasible is True
    assert batch.feasible is True
    assert scalar.effective_iterations == batch.effective_iterations == 5
    assert scalar.charging_subproblem_calls == batch.charging_subproblem_calls
    assert scalar.objective is not None and batch.objective is not None
    assert scalar.objective.key == batch.objective.key
    assert scalar.measurement_trace is not None and batch.measurement_trace is not None
    scalar_work = [
        (record.lane, record.iteration, record.operator, record.route_key)
        for record in scalar.measurement_trace.route_evaluations
        if record.kind == "exact_call"
    ]
    batch_work = [
        (record.lane, record.iteration, record.operator, record.route_key)
        for record in batch.measurement_trace.route_evaluations
        if record.kind == "exact_call"
    ]
    assert scalar_work == batch_work


def test_transition_edge_cases_match_scalar_reference() -> None:
    instance = _instance()
    station_instance = Instance(
        "gpu_batch_station_edge",
        tuple(node for node in instance.nodes if node.name != "C2"),
        Vehicle(10.0, 5.0, 1.0, 0.1, 1.0),
    )
    tight_window_instance = Instance(
        "gpu_batch_time_edge",
        tuple(
            replace(node, due_date=3.0) if node.name == "F1" else node
            for node in station_instance.nodes
        ),
        station_instance.vehicle,
    )
    capacity_instance = Instance(
        "gpu_batch_capacity_edge",
        station_instance.nodes,
        Vehicle(10.0, 0.5, 1.0, 0.1, 1.0),
    )
    cases = (
        (station_instance, (("C1",),)),
        (instance, (("C1", "C1"),)),
        (
            Instance(
                instance.name,
                tuple(
                    node
                    for node in instance.nodes
                    if node.name not in {"C2"}
                ),
                Vehicle(7.0, 5.0, 1.0, 0.1, 1.0),
            ),
            (("C1",),),
        ),
        (tight_window_instance, (("C1",),)),
        (capacity_instance, (("C1",),)),
    )
    for edge_instance, orders in cases:
        expected = tuple(solve_exact_charging(edge_instance, order) for order in orders)
        actual = solve_exact_charging_batch(
            edge_instance,
            orders,
            backend=ExactChargingBackend.CPU_BATCH,
            batch_size=32,
        ).results
        assert [_result_signature(item) for item in actual] == [
            _result_signature(item) for item in expected
        ]

    try:
        metal = tuple(
            solve_exact_charging_batch(
                edge_instance,
                orders,
                backend=ExactChargingBackend.METAL_BATCH,
                batch_size=32,
            ).results
            for edge_instance, orders in cases
        )
    except MetalUnavailableError:
        return
    for (edge_instance, orders), actual in zip(cases, metal, strict=True):
        expected = tuple(solve_exact_charging(edge_instance, order) for order in orders)
        assert [_result_signature(item) for item in actual] == [
            _result_signature(item) for item in expected
        ]
