from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from evrptw.alns import (
    CheapScreeningConfig,
    MeasurementConfig,
    _Evaluator,
    solve_alns,
)
from evrptw.experiments.stage03_measurement import _scope_instances, load_config
from evrptw.experiments.stage03_measurement_review import _screening_trace_ok
from evrptw.measurement import Stage03Trace
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.neighborhoods import screen_route_candidate
from evrptw.parser import parse_schneider


def _instance() -> Instance:
    return Instance(
        "stage031_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("F1", NodeType.STATION, 5.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 8.0, 1.0, 1.0, 0.0, 100.0, 1.0),
            Node("C2", NodeType.CUSTOMER, 8.0, -1.0, 1.0, 0.0, 100.0, 1.0),
        ),
        Vehicle(11.0, 3.0, 1.0, 0.1, 1.0),
    )


def test_full_screen_records_all_safe_metrics_without_distance_pruning() -> None:
    result = screen_route_candidate(_instance(), ("C1", "C2"), full=True)

    assert result.accepted
    names = {check.check for check in result.checks}
    assert {
        "route_structure",
        "capacity_lower_bound",
        "forward_time_window",
        "backward_time_window",
        "time_window_slack",
        "shortest_distance_lower_bound",
        "single_segment_battery_reachability",
        "structural_energy_lower_bound",
    } <= names
    assert result.distance_lower_bound > 0.0
    assert any(check.status == "recorded" for check in result.checks)


def test_full_screen_rejects_capacity_and_structural_energy_safely() -> None:
    capacity = replace(_instance(), vehicle=replace(_instance().vehicle, load_capacity=1.0))
    capacity_result = screen_route_candidate(capacity, ("C1", "C2"), full=True)
    assert not capacity_result.accepted
    assert capacity_result.reason == "capacity_prefilter"
    assert capacity_result.first_failed_check == "capacity_lower_bound"
    assert capacity_result.distance_lower_bound > 0.0

    structural = Instance(
        "structural",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 4.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(5.0, 2.0, 1.0, 0.1, 1.0),
    )
    structural_result = screen_route_candidate(structural, ("C1",), full=True)
    assert not structural_result.accepted
    assert structural_result.reason == "structural_energy_prefilter"


def test_negative_screening_cache_blocks_exact_and_records_independent_hits() -> None:
    instance = replace(
        _instance(),
        nodes=tuple(
            replace(node, due_date=0.1) if node.name == "C2" else node
            for node in _instance().nodes
        ),
    )
    trace = Stage03Trace(MeasurementConfig(), screening_config=CheapScreeningConfig())
    evaluator = _Evaluator(
        instance,
        deadline=1e9,
        measurement_trace=trace,
        screening_config=CheapScreeningConfig(),
        negative_screening_cache={},
    )

    first = evaluator.route(("C2",))
    second = evaluator.route(("C2",))

    assert not first.feasible and not second.feasible
    assert evaluator.calls == 0
    assert evaluator.screening_rejections == 1
    assert evaluator.screening_cache_hits == 1
    assert evaluator.screening_exact_call_blocked == 2
    assert [item.status for item in trace.screening_decisions] == [
        "rejected",
        "negative_cache_hit",
    ]
    assert not trace.route_evaluations


def test_screening_pass_allows_exact_and_exact_cache_remains_distinct() -> None:
    trace = Stage03Trace(MeasurementConfig(), screening_config=CheapScreeningConfig())
    evaluator = _Evaluator(
        _instance(),
        deadline=1e9,
        measurement_trace=trace,
        screening_config=CheapScreeningConfig(),
        negative_screening_cache={},
    )
    first = evaluator.route(("C1",))
    second = evaluator.route(("C1",))

    assert first.feasible and second.feasible
    assert evaluator.calls == 1
    assert [item.kind for item in trace.route_evaluations] == ["exact_call", "cache_hit"]
    assert all(item.status == "pass" for item in trace.screening_decisions)
    assert trace.screening_counts["screening_cache_hits"] == 0


def test_stage031_trace_round_trip_and_result_reconciliation() -> None:
    result = solve_alns(
        _instance(),
        seed=2014,
        max_iterations=8,
        time_limit_seconds=1.0,
        operator_profile="stage02_constraint_guided",
        measurement_config=MeasurementConfig(),
        screening_config=CheapScreeningConfig(),
    )
    assert result.measurement_trace is not None
    trace = result.measurement_trace
    assert trace.reconcile(result)["status"] == "pass"
    assert _screening_trace_ok(trace)
    restored = Stage03Trace.from_dict(trace.to_dict())
    assert restored.screening_counts == trace.screening_counts
    assert _screening_trace_ok(restored)


def test_stage031_requires_route_dictionary_for_auditable_screening() -> None:
    with pytest.raises(ValueError, match="record_route_dictionary"):
        Stage03Trace(
            MeasurementConfig(record_route_dictionary=False),
            screening_config=CheapScreeningConfig(),
        )


def test_stage031_replay_rejects_tampered_negative_cache_semantics() -> None:
    instance = replace(
        _instance(),
        nodes=tuple(
            replace(node, due_date=0.1) if node.name == "C2" else node
            for node in _instance().nodes
        ),
    )
    trace = Stage03Trace(MeasurementConfig(), screening_config=CheapScreeningConfig())
    evaluator = _Evaluator(
        instance,
        deadline=1e9,
        measurement_trace=trace,
        screening_config=CheapScreeningConfig(),
        negative_screening_cache={},
    )
    evaluator.route(("C2",))
    evaluator.route(("C2",))
    hit_index = 1
    trace.screening_decisions[hit_index] = replace(
        trace.screening_decisions[hit_index], negative_cache_hit=False
    )
    assert not _screening_trace_ok(trace)


def test_stage031_config_and_scope_are_exact() -> None:
    config = load_config(Path("configs/stage031_cheap_screening.toml"))
    assert config.screening_config == CheapScreeningConfig()
    assert len(_scope_instances("smoke")) == 6
    assert len(set(_scope_instances("smoke"))) == 6


def test_real_c5_feasible_route_is_not_rejected_by_full_screen() -> None:
    instance = parse_schneider(Path("data/schneider/c101C5.txt"))
    result = screen_route_candidate(
        instance,
        tuple(customer.name for customer in instance.customers[:1]),
        full=True,
    )
    assert result.accepted
