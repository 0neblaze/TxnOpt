from __future__ import annotations

import pytest

from evrptw.bpc import generate_columns_bidirectionally, solve_branch_price_and_cut
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.validation import validate_routes


def _instance() -> Instance:
    return Instance(
        "bpc_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("F1", NodeType.STATION, 4.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 7.0, 1.0, 1.0, 0.0, 100.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 7.0, -1.0, 1.0, 0.0, 100.0, 0.0),
            Node("C3", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(10.0, 2.0, 1.0, 0.1, 1.0),
    )


def test_bidirectional_pricing_generates_feasible_elementary_columns() -> None:
    result = generate_columns_bidirectionally(_instance())

    assert result.forward_labels > 1
    assert result.backward_labels > 1
    assert result.joined_labels > 1
    assert result.columns
    assert all(len(set(column.customers)) == len(column.customers) for column in result.columns)


def test_branch_price_and_cut_proves_small_instance_and_validates_incumbent() -> None:
    instance = _instance()
    result = solve_branch_price_and_cut(instance, time_limit_seconds=10.0)

    assert result.status == "optimal"
    assert result.proven_optimal is True
    assert result.objective_value == pytest.approx(result.root_lower_bound)
    assert result.final_lower_bound == pytest.approx(result.incumbent)
    assert result.optimality_gap == pytest.approx(0.0)
    assert result.generated_columns >= result.active_columns
    assert result.pricing_iterations >= 1
    assert validate_routes(instance, [list(route) for route in result.routes]).feasible


def test_exact_pricing_fails_fast_above_documented_size_limit() -> None:
    instance = _instance()
    with pytest.raises(ValueError, match="at most 2 customers"):
        generate_columns_bidirectionally(instance, max_customers=2)
