from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from evrptw._core import (
    exact_charging_batch_numeric,
    propagate_routes_numeric,
    screen_routes_numeric,
)
from evrptw.cache_incremental import (
    IncrementalPropagationResult,
    RoutePropagationSnapshot,
    build_route_propagation_snapshot,
    incremental_route_propagation,
)
from evrptw.charging import solve_exact_charging
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.neighborhoods import _energy_reachable_optimistically, screen_route_candidate
from evrptw.validation import validate_routes


def _instance(
    *,
    station_due: float = 200.0,
    symmetric_stations: bool = False,
) -> Instance:
    station_nodes = (
        (
            Node("F1", NodeType.STATION, 2.0, 1.0, 0.0, 0.0, station_due, 0.0),
            Node("F2", NodeType.STATION, 2.0, -1.0, 0.0, 0.0, station_due, 0.0),
        )
        if symmetric_stations
        else (Node("F1", NodeType.STATION, 4.0, 0.0, 0.0, 0.0, station_due, 0.0),)
    )
    customer_x = 4.0 if symmetric_stations else 8.0
    battery = 2.5 if symmetric_stations else 10.0
    return Instance(
        "native_exact_fixture",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            *station_nodes,
            Node("C1", NodeType.CUSTOMER, customer_x, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(battery, 5.0, 1.0, 0.1, 1.0),
        distance_backend="python",
    )


def _pack(
    instance: Instance,
    orders: tuple[tuple[str, ...], ...],
    *,
    deadline_remaining: float = float("inf"),
    batch_size: int = 128,
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], ...]:
    node_index = {node.name: index for index, node in enumerate(instance.nodes)}
    kind_codes = {
        NodeType.DEPOT: 0,
        NodeType.CUSTOMER: 1,
        NodeType.STATION: 2,
    }
    offsets = [0]
    indices: list[int] = []
    for order in orders:
        indices.extend(node_index[name] for name in order)
        offsets.append(len(indices))
    return (
        np.asarray([kind_codes[node.kind] for node in instance.nodes], dtype=np.int64),
        np.asarray([node.ready_time for node in instance.nodes], dtype=np.float64),
        np.asarray([node.due_date for node in instance.nodes], dtype=np.float64),
        np.asarray([node.service_time for node in instance.nodes], dtype=np.float64),
        np.asarray(
            [
                [instance.distance(origin.name, destination.name) for destination in instance.nodes]
                for origin in instance.nodes
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                instance.vehicle.battery_capacity,
                instance.vehicle.load_capacity,
                instance.vehicle.consumption_rate,
                instance.vehicle.inverse_refueling_rate,
                instance.vehicle.average_velocity,
            ],
            dtype=np.float64,
        ),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        np.asarray([deadline_remaining], dtype=np.float64),
        np.asarray([batch_size], dtype=np.int64),
    )


def _decode_paths(
    instance: Instance,
    offsets: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    indices: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(instance.nodes[int(index)].name for index in indices[offsets[i] : offsets[i + 1]])
        for i in range(len(offsets) - 1)
    )


def test_exact_charging_batch_numeric_rejects_non_c_contiguous_arrays() -> None:
    inputs = list(_pack(_instance(), (("C1",),)))
    inputs[4] = np.asfortranarray(inputs[4])

    with pytest.raises(ValueError, match="C-contiguous"):
        exact_charging_batch_numeric(*inputs)


def test_exact_charging_batch_numeric_rejects_wrong_dtype_and_shape() -> None:
    inputs = list(_pack(_instance(), (("C1",),)))
    inputs[0] = inputs[0].astype(np.int32)

    with pytest.raises(ValueError, match="requested numeric dtype"):
        exact_charging_batch_numeric(*inputs)

    inputs = list(_pack(_instance(), (("C1",),)))
    inputs[4] = np.zeros((len(inputs[0]), len(inputs[0]) + 1), dtype=np.float64)
    with pytest.raises(ValueError, match="shape"):
        exact_charging_batch_numeric(*inputs)

    inputs = list(_pack(_instance(), (("C1",),)))
    inputs[6] = np.asarray([0, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="span order_indices"):
        exact_charging_batch_numeric(*inputs)

    inputs = list(_pack(_instance(), (("C1",),)))
    inputs[7] = np.asarray([0], dtype=np.int64)
    with pytest.raises(ValueError, match="customer node indices"):
        exact_charging_batch_numeric(*inputs)


def test_exact_charging_batch_numeric_frozen_fixture_matches_python() -> None:
    instance = _instance()
    orders = (("C1",), ())
    path_offsets, path_indices, status, reason, metrics, counters, batch = (
        exact_charging_batch_numeric(*_pack(instance, orders, batch_size=2))
    )

    expected = tuple(solve_exact_charging(instance, order) for order in orders)
    assert _decode_paths(instance, path_offsets, path_indices) == tuple(
        result.route for result in expected
    )
    np.testing.assert_array_equal(status, np.asarray([0, 0], dtype=np.int64))
    np.testing.assert_array_equal(reason, np.asarray([0, 0], dtype=np.int64))
    np.testing.assert_allclose(
        metrics,
        np.asarray(
            [
                [
                    result.distance,
                    result.total_energy,
                    result.charged_energy,
                    result.charging_time,
                ]
                for result in expected
            ]
        ),
    )
    np.testing.assert_array_equal(
        counters,
        np.asarray(
            [
                [result.labels_generated, result.labels_expanded, result.labels_pruned]
                for result in expected
            ],
            dtype=np.int64,
        ),
    )
    assert batch.tolist() == [2, 2, 2, 0, 1, batch[5], batch[6], batch[7], 1, 2]
    assert batch[5] > 0
    assert batch[6] == sum(result.labels_generated - 1 for result in expected)
    assert batch[7] > 0


def test_exact_charging_batch_numeric_matches_small_brute_force() -> None:
    instance = _instance()
    native = exact_charging_batch_numeric(*_pack(instance, (("C1",),)))
    native_route = _decode_paths(instance, native[0], native[1])[0]

    feasible: list[tuple[float, tuple[str, ...]]] = []
    for length in range(1, 4):
        for middle in product(("F1", "C1"), repeat=length):
            if tuple(name for name in middle if name == "C1") != ("C1",):
                continue
            route = ("D0", *middle, "D0")
            report = validate_routes(instance, [route])
            if report.feasible:
                feasible.append((report.total_distance, route))
    assert native_route == min(feasible, key=lambda item: item[0])[1]


def test_exact_charging_batch_numeric_randomized_differential() -> None:
    random = np.random.default_rng(20260718)
    for case in range(12):
        coordinates = random.uniform(-4.0, 4.0, size=(6, 2))
        coordinates[0] = (0.0, 0.0)
        nodes = [Node("D0", NodeType.DEPOT, *coordinates[0], 0.0, 0.0, 500.0, 0.0)]
        nodes.extend(
            Node(f"F{i}", NodeType.STATION, *coordinates[i], 0.0, 0.0, 500.0, 0.0)
            for i in (1, 2)
        )
        nodes.extend(
            Node(f"C{i}", NodeType.CUSTOMER, *coordinates[i], 1.0, 0.0, 500.0, 0.25)
            for i in (3, 4, 5)
        )
        instance = Instance(
            f"random_{case}",
            tuple(nodes),
            Vehicle(12.0, 10.0, 1.0, 0.05, 1.0),
            distance_backend="python",
        )
        order_names = ["C3", "C4", "C5"]
        orders: list[tuple[str, ...]] = []
        for _ in range(4):
            random.shuffle(order_names)
            orders.append(tuple(order_names[: int(random.integers(0, 4))]))
        order_tuple = tuple(orders)
        native = exact_charging_batch_numeric(*_pack(instance, order_tuple, batch_size=3))
        expected = tuple(solve_exact_charging(instance, order) for order in order_tuple)
        assert _decode_paths(instance, native[0], native[1]) == tuple(
            result.route for result in expected
        )
        np.testing.assert_array_equal(
            native[2], np.asarray([0 if result.feasible else 1 for result in expected])
        )
        np.testing.assert_allclose(
            native[4],
            np.asarray(
                [
                    [
                        result.distance,
                        result.total_energy,
                        result.charged_energy,
                        result.charging_time,
                    ]
                    for result in expected
                ]
            ),
        )
        np.testing.assert_array_equal(
            native[5],
            np.asarray(
                [
                    [result.labels_generated, result.labels_expanded, result.labels_pruned]
                    for result in expected
                ]
            ),
        )


def test_exact_charging_batch_numeric_stable_tie_order_matches_python() -> None:
    instance = Instance(
        "native_stable_tie",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("F1", NodeType.STATION, 3.0, 1.0, 0.0, 0.0, 200.0, 0.0),
            Node("F2", NodeType.STATION, 3.0, -1.0, 0.0, 0.0, 200.0, 0.0),
            Node("F3", NodeType.STATION, 7.5, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 8.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(7.2, 5.0, 1.0, 0.1, 1.0),
        distance_backend="python",
    )
    native = exact_charging_batch_numeric(*_pack(instance, (("C1",),), batch_size=1))
    expected = solve_exact_charging(instance, ("C1",))
    route = _decode_paths(instance, native[0], native[1])[0]

    assert route == expected.route
    assert route[1] == "F1"
    np.testing.assert_array_equal(
        native[5],
        np.asarray(
            [[expected.labels_generated, expected.labels_expanded, expected.labels_pruned]]
        ),
    )


def test_exact_charging_batch_numeric_rechecks_station_due_after_recharge() -> None:
    instance = _instance(station_due=4.1)
    native = exact_charging_batch_numeric(*_pack(instance, (("C1",),)))
    expected = solve_exact_charging(instance, ("C1",))

    assert expected.feasible is False
    assert _decode_paths(instance, native[0], native[1]) == ((),)
    np.testing.assert_array_equal(native[2], np.asarray([1], dtype=np.int64))
    np.testing.assert_array_equal(native[3], np.asarray([1], dtype=np.int64))
    np.testing.assert_array_equal(
        native[5],
        np.asarray(
            [[expected.labels_generated, expected.labels_expanded, expected.labels_pruned]]
        ),
    )


def test_exact_charging_batch_numeric_batch_boundaries_preserve_results() -> None:
    instance = _instance()
    orders = (("C1",), (), ("C1",))
    single = exact_charging_batch_numeric(*_pack(instance, orders, batch_size=1))
    wide = exact_charging_batch_numeric(*_pack(instance, orders, batch_size=128))

    for index in range(6):
        if index == 4:
            np.testing.assert_allclose(single[index], wide[index])
        else:
            np.testing.assert_array_equal(single[index], wide[index])
    assert single[6][5] > wide[6][5]
    assert single[6][6] == wide[6][6]


def test_exact_charging_batch_numeric_deadline_interrupts_without_fallback() -> None:
    instance = _instance()
    native = exact_charging_batch_numeric(
        *_pack(instance, (("C1",), ("C1",)), deadline_remaining=0.0)
    )

    np.testing.assert_array_equal(native[0], np.asarray([0, 0, 0], dtype=np.int64))
    assert native[1].size == 0
    np.testing.assert_array_equal(native[2], np.asarray([2, 2], dtype=np.int64))
    np.testing.assert_array_equal(native[3], np.asarray([2, 2], dtype=np.int64))
    assert native[6].tolist() == [2, 2, 0, 2, 1, 0, 0, 1, 1, 128]


_SCREEN_REASON = {
    0: "",
    1: "route_structure_prefilter",
    2: "capacity_prefilter",
    3: "forward_time_window_prefilter",
    4: "backward_time_window_prefilter",
    5: "time_window_slack_prefilter",
    6: "single_segment_energy_prefilter",
    7: "structural_energy_prefilter",
    8: "time_window_prefilter",
    9: "energy_prefilter",
}
_CHECK_NAME = {
    1: "route_structure",
    2: "capacity_lower_bound",
    3: "forward_time_window",
    4: "backward_time_window",
    5: "time_window_slack",
    6: "shortest_distance_lower_bound",
    7: "single_segment_battery_reachability",
    8: "structural_energy_lower_bound",
}
_CHECK_STATUS = {0: "fail", 1: "pass", 2: "recorded"}


def _numeric_context(instance: Instance) -> tuple[np.ndarray, ...]:
    kind_codes = {
        NodeType.DEPOT: 0,
        NodeType.CUSTOMER: 1,
        NodeType.STATION: 2,
    }
    distance = np.asarray(
        [
            [instance.distance(left.name, right.name) for right in instance.nodes]
            for left in instance.nodes
        ],
        dtype=np.float64,
    )
    vehicle = np.asarray(
        [
            instance.vehicle.battery_capacity,
            instance.vehicle.load_capacity,
            instance.vehicle.consumption_rate,
            instance.vehicle.inverse_refueling_rate,
            instance.vehicle.average_velocity,
        ],
        dtype=np.float64,
    )
    return (
        np.asarray([kind_codes[node.kind] for node in instance.nodes], dtype=np.int64),
        np.asarray([node.demand for node in instance.nodes], dtype=np.float64),
        np.asarray([node.ready_time for node in instance.nodes], dtype=np.float64),
        np.asarray([node.due_date for node in instance.nodes], dtype=np.float64),
        np.asarray([node.service_time for node in instance.nodes], dtype=np.float64),
        distance,
        vehicle,
    )


def _screen_pack(
    instance: Instance,
    sequence: tuple[str, ...],
    *,
    full: bool = True,
    reference_distance: float | None = None,
    incremental: IncrementalPropagationResult | None = None,
) -> tuple[np.ndarray, ...]:
    kind, demand, ready, due, service, distance, vehicle = _numeric_context(instance)
    node_index = {node.name: index for index, node in enumerate(instance.nodes)}
    reachable = np.asarray(
        [
            [
                _energy_reachable_optimistically(instance, left.name, right.name)
                for right in instance.nodes
            ]
            for left in instance.nodes
        ],
        dtype=np.uint8,
    )
    incremental_values = (
        np.asarray(
            [
                1.0,
                incremental.distance_lower_bound,
                incremental.min_time_window_slack,
                incremental.finish_time,
                float(incremental.forward_feasible),
                float(incremental.backward_feasible),
            ],
            dtype=np.float64,
        )
        if incremental is not None and incremental.status == "incremental"
        else np.zeros(6, dtype=np.float64)
    )
    return (
        kind,
        demand,
        ready,
        due,
        service,
        distance,
        reachable,
        vehicle,
        np.asarray([node_index.get(name, -1) for name in sequence], dtype=np.int64),
        np.asarray(
            [
                float(full),
                1e-9,
                reference_distance if reference_distance is not None else 0.0,
                float(reference_distance is not None),
            ],
            dtype=np.float64,
        ),
        incremental_values,
    )


def _assert_screen_matches_python(
    instance: Instance,
    sequence: tuple[str, ...],
    *,
    full: bool = True,
    reference_distance: float | None = None,
    incremental: IncrementalPropagationResult | None = None,
) -> None:
    codes, metrics = screen_routes_numeric(
        *_screen_pack(
            instance,
            sequence,
            full=full,
            reference_distance=reference_distance,
            incremental=incremental,
        )
    )
    expected = screen_route_candidate(
        instance,
        sequence,
        full=full,
        reference_distance=reference_distance,
        incremental_metrics=incremental,
    )
    assert bool(codes[0]) is expected.accepted
    assert _SCREEN_REASON[int(codes[1])] == expected.reason
    assert bool(codes[3]) is expected.single_segment_reachable
    assert metrics[0] == pytest.approx(expected.demand)
    assert metrics[1] == pytest.approx(expected.optimistic_finish_time)
    assert metrics[2] == pytest.approx(expected.min_time_window_slack)
    assert metrics[3] == pytest.approx(expected.distance_lower_bound)
    if expected.distance_increment_lower_bound is None:
        assert np.isnan(metrics[4])
    else:
        assert metrics[4] == pytest.approx(expected.distance_increment_lower_bound)
    assert metrics[5] == pytest.approx(expected.structural_energy_lower_bound)
    if full:
        event_count = int(codes[7])
        actual_events = [
            (
                _CHECK_NAME[int(codes[8 + index]) // 10],
                _CHECK_STATUS[int(codes[8 + index]) % 10],
                float(metrics[7 + index]),
            )
            for index in range(event_count)
        ]
        assert len(actual_events) == len(expected.checks)
        for actual, trace in zip(actual_events, expected.checks, strict=True):
            assert actual[:2] == (trace.check, trace.status)
            assert actual[2] == pytest.approx(float(trace.value))


def test_screen_routes_numeric_frozen_and_failure_differential() -> None:
    instance = _instance()
    _assert_screen_matches_python(instance, ("C1",), reference_distance=20.0)
    _assert_screen_matches_python(instance, ("unknown",))
    _assert_screen_matches_python(instance, ("F1",))
    _assert_screen_matches_python(instance, ("C1", "C1"))
    _assert_screen_matches_python(
        Instance(
            instance.name,
            instance.nodes,
            replace(instance.vehicle, load_capacity=0.5),
            distance_backend="python",
        ),
        ("C1",),
    )
    tight = Instance(
        instance.name,
        tuple(
            replace(node, due_date=7.0) if node.name == "C1" else node
            for node in instance.nodes
        ),
        instance.vehicle,
        distance_backend="python",
    )
    _assert_screen_matches_python(tight, ("C1",))
    _assert_screen_matches_python(tight, ("C1",), full=False)


def test_screen_routes_numeric_randomized_python_differential() -> None:
    random = np.random.default_rng(8181)
    for case in range(15):
        coordinates = random.uniform(-5.0, 5.0, size=(5, 2))
        coordinates[0] = 0.0
        instance = Instance(
            f"screen_random_{case}",
            (
                Node("D0", NodeType.DEPOT, *coordinates[0], 0.0, 0.0, 100.0, 0.0),
                *(
                    Node(
                        f"C{index}",
                        NodeType.CUSTOMER,
                        *coordinates[index],
                        1.0,
                        float(random.uniform(0.0, 3.0)),
                        float(random.uniform(8.0, 30.0)),
                        float(random.uniform(0.0, 2.0)),
                    )
                    for index in range(1, 5)
                ),
            ),
            Vehicle(100.0, 10.0, 1.0, 0.1, 1.0),
            distance_backend="python",
        )
        names = ["C1", "C2", "C3", "C4"]
        random.shuffle(names)
        _assert_screen_matches_python(
            instance,
            tuple(names[: int(random.integers(0, 5))]),
            reference_distance=float(random.uniform(0.0, 30.0)),
        )


def test_screen_routes_numeric_consumes_incremental_result_without_python_replay() -> None:
    instance = _instance()
    accepted = IncrementalPropagationResult(
        "incremental",
        "incremental_propagation",
        ("C1",),
        ("C1",),
        16.0,
        92.0,
        16.0,
        True,
        True,
        "",
        2,
        0,
        0,
        0,
    )
    rejected = replace(
        accepted,
        reason="forward_time_window_prefilter",
        min_time_window_slack=-1.0,
        finish_time=8.0,
        forward_feasible=False,
        first_failed_check="forward_time_window",
    )
    _assert_screen_matches_python(instance, ("C1",), incremental=accepted)
    _assert_screen_matches_python(instance, ("C1",), incremental=rejected)


def _propagation_pack(
    instance: Instance,
    base: RoutePropagationSnapshot,
    candidate: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    kind, _demand, ready, due, service, distance, vehicle = _numeric_context(instance)
    node_index = {node.name: index for index, node in enumerate(instance.nodes)}
    return (
        kind,
        ready,
        due,
        service,
        distance,
        vehicle,
        np.asarray([node_index[name] for name in base.chain], dtype=np.int64),
        np.asarray(
            [
                node_index[instance.depot.name],
                *[node_index.get(name, -1) for name in candidate],
                node_index[instance.depot.name],
            ],
            dtype=np.int64,
        ),
        np.asarray(base.edge_distances, dtype=np.float64),
        np.asarray(base.earliest_arrivals, dtype=np.float64),
        np.asarray(base.latest_departures, dtype=np.float64),
        np.asarray([1e-9], dtype=np.float64),
    )


def _assert_propagation_matches_python(
    instance: Instance,
    base: RoutePropagationSnapshot,
    candidate: tuple[str, ...],
) -> None:
    codes, metrics = propagate_routes_numeric(*_propagation_pack(instance, base, candidate))
    expected = incremental_route_propagation(instance, base, candidate)
    status = "incremental" if codes[0] == 0 else "fallback"
    reasons = {
        0: "incremental_propagation",
        1: "unchanged_route",
        2: "route_structure_requires_full_propagation",
        3: "forward_time_window_prefilter",
        4: "backward_time_window_prefilter",
        5: "time_window_slack_prefilter",
    }
    failures = {
        0: "",
        1: "route_structure",
        2: "forward_time_window",
        3: "backward_time_window",
        4: "time_window_slack",
    }
    assert status == expected.status
    assert reasons[int(codes[1])] == expected.reason
    assert failures[int(codes[2])] == expected.first_failed_check
    assert bool(codes[3]) is expected.forward_feasible
    assert bool(codes[4]) is expected.backward_feasible
    assert tuple(int(value) for value in codes[5:9]) == (
        expected.reused_prefix_edges,
        expected.reused_suffix_edges,
        expected.recomputed_forward_edges,
        expected.recomputed_backward_edges,
    )
    np.testing.assert_allclose(
        metrics,
        np.asarray(
            [
                expected.distance_lower_bound,
                expected.min_time_window_slack,
                expected.finish_time,
            ]
        ),
    )


def test_propagate_routes_numeric_frozen_unchanged_and_fallback_differential() -> None:
    instance = Instance(
        "propagation_fixture",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 20.0, 1.0),
            Node("C2", NodeType.CUSTOMER, 4.0, 0.0, 1.0, 0.0, 30.0, 1.0),
            Node("C3", NodeType.CUSTOMER, 6.0, 0.0, 1.0, 0.0, 40.0, 1.0),
        ),
        Vehicle(20.0, 10.0, 1.0, 0.1, 1.0),
        distance_backend="python",
    )
    base = build_route_propagation_snapshot(instance, ("C1", "C2", "C3"))
    _assert_propagation_matches_python(instance, base, ("C1", "C3", "C2"))
    _assert_propagation_matches_python(instance, base, base.sequence)
    _assert_propagation_matches_python(instance, base, ("C1", "C1"))
    _assert_propagation_matches_python(instance, base, ("unknown",))


def test_propagate_routes_numeric_randomized_python_differential() -> None:
    random = np.random.default_rng(9917)
    for case in range(20):
        coordinates = random.uniform(-5.0, 5.0, size=(6, 2))
        coordinates[0] = 0.0
        instance = Instance(
            f"propagation_random_{case}",
            (
                Node("D0", NodeType.DEPOT, *coordinates[0], 0.0, 0.0, 100.0, 0.0),
                *(
                    Node(
                        f"C{index}",
                        NodeType.CUSTOMER,
                        *coordinates[index],
                        1.0,
                        float(random.uniform(0.0, 3.0)),
                        float(random.uniform(8.0, 40.0)),
                        float(random.uniform(0.0, 2.0)),
                    )
                    for index in range(1, 6)
                ),
            ),
            Vehicle(100.0, 10.0, 1.0, 0.1, 1.0),
            distance_backend="python",
        )
        base_names = [f"C{index}" for index in range(1, 6)]
        random.shuffle(base_names)
        base = build_route_propagation_snapshot(instance, tuple(base_names))
        candidate = base_names.copy()
        random.shuffle(candidate)
        _assert_propagation_matches_python(instance, base, tuple(candidate))


def test_screen_and_propagation_numeric_schema_fail_fast() -> None:
    instance = _instance()
    screen_inputs = list(_screen_pack(instance, ("C1",)))
    screen_inputs[5] = np.asfortranarray(screen_inputs[5])
    with pytest.raises(ValueError, match="C-contiguous"):
        screen_routes_numeric(*screen_inputs)

    base = build_route_propagation_snapshot(instance, ("C1",))
    propagation_inputs = list(_propagation_pack(instance, base, ("C1",)))
    propagation_inputs[8] = np.asarray([], dtype=np.float64)
    with pytest.raises(ValueError, match="inconsistent lengths"):
        propagate_routes_numeric(*propagation_inputs)
