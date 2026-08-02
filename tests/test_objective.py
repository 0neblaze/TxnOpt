from __future__ import annotations

import pytest

from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.objective import (
    ObjectiveComparison,
    SolutionObjective,
    accept_annealing_move,
    compare_objectives,
    objective_key_from_exact_numeric,
)
from evrptw.validation import validate_routes


def _objective(
    vehicles: int,
    distance: float,
    charging_time: float = 0.0,
    charging_count: int = 0,
) -> SolutionObjective:
    return SolutionObjective(vehicles, distance, charging_time, charging_count)


def test_lexicographic_objective_uses_every_level_in_declared_order() -> None:
    assert (
        compare_objectives(_objective(1, 100.0), _objective(2, 1.0)) is ObjectiveComparison.BETTER
    )
    assert (
        compare_objectives(_objective(2, 10.0), _objective(2, 11.0)) is ObjectiveComparison.BETTER
    )
    assert (
        compare_objectives(_objective(2, 10.0, 4.0), _objective(2, 10.0, 5.0))
        is ObjectiveComparison.BETTER
    )
    assert (
        compare_objectives(_objective(2, 10.0, 4.0, 1), _objective(2, 10.0, 4.0, 2))
        is ObjectiveComparison.BETTER
    )


def test_normalized_key_makes_sub_nanounit_noise_equal_and_order_transitive() -> None:
    first = _objective(1, 10.0)
    noise = _objective(1, 10.0 + 0.4e-9)
    later = _objective(1, 10.0 + 1.4e-9)

    assert compare_objectives(first, noise) is ObjectiveComparison.EQUAL
    assert compare_objectives(first, later) is ObjectiveComparison.BETTER
    assert first.key < later.key


def test_numeric_exact_objective_uses_canonical_rounding_and_station_count() -> None:
    key = objective_key_from_exact_numeric(
        [0, 1, 2],
        [0, 2, 1, 2, 0],
        [[5.0000000005, 0.0, 0.0, 1.25]],
        vehicle_count=1,
        station_kind=2,
    )

    assert key == SolutionObjective(1, 5.0000000005, 1.25, 2).key


@pytest.mark.parametrize(
    "values",
    [
        (-1, 1.0, 1.0, 0),
        (1, -1.0, 1.0, 0),
        (1, float("nan"), 1.0, 0),
        (1, 1.0, float("inf"), 0),
        (1, 1.0, 1.0, -1),
    ],
)
def test_objective_rejects_invalid_values(values: tuple[int, float, float, int]) -> None:
    with pytest.raises(ValueError):
        SolutionObjective(*values)


def test_objective_from_validator_report_counts_station_visits() -> None:
    instance = Instance(
        "objective_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("F1", NodeType.STATION, 2.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 4.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(4.0, 2.0, 1.0, 0.1, 1.0),
    )
    report = validate_routes(instance, [["D0", "F1", "C1", "F1", "D0"]])

    objective = SolutionObjective.from_report(instance, report)

    assert objective.key == _objective(1, 8.0, 0.6, 2).key

    route_objective = SolutionObjective.from_route(
        instance,
        ("D0", "F1", "C1", "F1", "D0"),
        total_distance=8.0,
        total_charging_time=0.6,
    )
    assert route_objective.key == objective.key


def test_annealing_policy_never_accepts_more_vehicles_and_always_accepts_fewer() -> None:
    assert not accept_annealing_move(
        _objective(1, 100.0), _objective(2, 1.0), temperature=100.0, random_draw=0.0
    )
    assert accept_annealing_move(
        _objective(2, 1.0), _objective(1, 100.0), temperature=0.01, random_draw=0.999
    )


def test_annealing_only_randomizes_worse_distance_with_same_vehicle_count() -> None:
    current = _objective(2, 10.0, 5.0, 1)
    assert accept_annealing_move(
        current, _objective(2, 11.0, 1.0, 0), temperature=10.0, random_draw=0.5
    )
    assert not accept_annealing_move(
        current, _objective(2, 10.0, 6.0, 1), temperature=10.0, random_draw=0.0
    )
