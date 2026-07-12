from __future__ import annotations

from evrptw.benchmark import audit_instance, calculate_battery_bounds
from evrptw.models import Instance, Node, NodeType, Vehicle


def _instance(*, battery: float = 12.0, demand: float = 1.0) -> Instance:
    return Instance(
        "audit_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("S1", NodeType.STATION, 10.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 6.0, 0.0, demand, 0.0, 100.0, 1.0),
        ),
        Vehicle(battery, 5.0, 1.0, 0.1, 1.0),
    )


def test_battery_bounds_use_recharge_customer_structure() -> None:
    bounds = calculate_battery_bounds(_instance())

    assert bounds.theoretical_lower_bound == 4.0
    assert bounds.structural_lower_bound == 8.0
    assert bounds.experimental_capacity == 12.0
    assert bounds.stress_ratio == 1.5


def test_audit_rejects_structurally_impossible_battery_and_demand() -> None:
    audit = audit_instance(_instance(battery=7.0, demand=6.0))

    assert audit.structurally_feasible is False
    assert "battery below customer-level structural lower bound" in audit.failures
    assert "C1: demand exceeds vehicle capacity" in audit.failures


def test_audit_accepts_structurally_sound_toy_instance() -> None:
    audit = audit_instance(_instance())

    assert audit.structurally_feasible is True
    assert audit.failures == ()
    assert audit.distance_metric.startswith("EUC_2D")
