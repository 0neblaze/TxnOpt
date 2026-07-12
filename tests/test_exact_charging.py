from __future__ import annotations

import pytest

from evrptw.charging import solve_exact_charging
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.validation import validate_routes


def _instance(*, station_due: float = 100.0) -> Instance:
    return Instance(
        "charging_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("F1", NodeType.STATION, 4.0, 0.0, 0.0, 0.0, station_due, 0.0),
            Node("C1", NodeType.CUSTOMER, 8.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(10.0, 5.0, 1.0, 0.1, 1.0),
    )


def test_exact_charging_inserts_station_and_returns_validator_feasible_route() -> None:
    instance = _instance()
    result = solve_exact_charging(instance, ["C1"])

    assert result.feasible is True
    assert result.route == ("D0", "F1", "C1", "F1", "D0")
    assert result.distance == 16.0
    assert result.charged_energy == 12.0
    assert result.charging_time == pytest.approx(1.2)
    assert result.labels_generated > 0
    assert validate_routes(instance, [list(result.route)]).feasible


def test_exact_charging_reports_no_pattern_without_fabricating_route() -> None:
    result = solve_exact_charging(_instance(station_due=3.0), ["C1"])

    assert result.feasible is False
    assert result.route == ()
    assert "no feasible" in result.failure_reason


def test_exact_charging_rejects_invalid_fixed_order() -> None:
    result = solve_exact_charging(_instance(), ["C1", "C1"])

    assert result.feasible is False
    assert result.labels_generated == 0
    assert result.failure_reason == "customer order contains duplicates"
