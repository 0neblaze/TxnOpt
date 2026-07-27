from __future__ import annotations

import pytest

from evrptw.metrics import evaluate_route_metrics
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.validation import validate_routes


def test_route_metrics_classify_energy_and_charging_fields() -> None:
    instance = Instance(
        "metrics_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 1_000.0, 0.0),
            Node("S1", NodeType.STATION, 30.0, 0.0, 0.0, 0.0, 1_000.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 60.0, 0.0, 1.0, 0.0, 1_000.0, 0.0),
        ),
        Vehicle(
            battery_capacity=50.0,
            load_capacity=10.0,
            consumption_rate=1.0,
            inverse_refueling_rate=0.1,
            average_velocity=1.0,
        ),
    )

    metrics = evaluate_route_metrics(
        instance,
        [["D0", "S1", "C1", "D0"]],
        method="TEST",
        seed=1,
        runtime_seconds=0.5,
    )

    assert metrics.feasible is False
    assert metrics.energy_violations == 1
    assert metrics.charging_count == 1
    assert metrics.charging_time == 3.0
    assert metrics.first_infeasible_step == "route 1 leg 3 C1->D0: battery depleted"


def test_validator_recomputes_energy_charging_and_rejects_false_objective() -> None:
    instance = Instance(
        "validator_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("S1", NodeType.STATION, 3.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 6.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(8.0, 10.0, 1.0, 0.1, 1.0),
    )

    report = validate_routes(
        instance,
        [["D0", "S1", "C1", "S1", "D0"]],
        claimed_objective=999.0,
    )

    assert report.feasible is False
    assert report.total_distance == 12.0
    assert report.total_energy == 12.0
    assert report.total_charged_energy == 9.0
    assert report.total_charging_time == pytest.approx(0.9)
    assert "claimed objective" in report.violations[0]
