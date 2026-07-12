from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from evrptw.models import Instance, NodeType
from evrptw.validation import SolutionReport

OBJECTIVE_PRECISION_DIGITS = 9


def count_charging_visits(instance: Instance, routes: Iterable[Iterable[str]]) -> int:
    """Count charging-station visits using the instance's node types."""

    return sum(
        1 for route in routes for name in route if instance.by_name[name].kind is NodeType.STATION
    )


class ObjectiveComparison(StrEnum):
    BETTER = "better"
    EQUAL = "equal"
    WORSE = "worse"


@dataclass(frozen=True, slots=True)
class SolutionObjective:
    vehicle_count: int
    total_distance: float
    total_charging_time: float
    charging_count: int

    def __post_init__(self) -> None:
        if self.vehicle_count < 0:
            raise ValueError("vehicle_count must be non-negative")
        if self.charging_count < 0:
            raise ValueError("charging_count must be non-negative")
        for name, value in (
            ("total_distance", self.total_distance),
            ("total_charging_time", self.total_charging_time),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def key(self) -> tuple[int, float, float, int]:
        return (
            self.vehicle_count,
            round(self.total_distance, OBJECTIVE_PRECISION_DIGITS),
            round(self.total_charging_time, OBJECTIVE_PRECISION_DIGITS),
            self.charging_count,
        )

    @classmethod
    def zero(cls) -> SolutionObjective:
        return cls(0, 0.0, 0.0, 0)

    @classmethod
    def from_report(cls, instance: Instance, report: SolutionReport) -> SolutionObjective:
        if not report.feasible:
            raise ValueError("cannot construct an objective from an infeasible solution report")
        return cls(
            report.vehicle_count,
            report.total_distance,
            report.total_charging_time,
            count_charging_visits(instance, (route.route for route in report.routes)),
        )

    @classmethod
    def from_route(
        cls,
        instance: Instance,
        route: Iterable[str],
        *,
        total_distance: float,
        total_charging_time: float,
    ) -> SolutionObjective:
        route_tuple = tuple(route)
        return cls(
            1,
            total_distance,
            total_charging_time,
            count_charging_visits(instance, (route_tuple,)),
        )

    def __add__(self, other: SolutionObjective) -> SolutionObjective:
        if not isinstance(other, SolutionObjective):
            return NotImplemented
        return SolutionObjective(
            self.vehicle_count + other.vehicle_count,
            self.total_distance + other.total_distance,
            self.total_charging_time + other.total_charging_time,
            self.charging_count + other.charging_count,
        )


def compare_objectives(left: SolutionObjective, right: SolutionObjective) -> ObjectiveComparison:
    if left.key < right.key:
        return ObjectiveComparison.BETTER
    if left.key > right.key:
        return ObjectiveComparison.WORSE
    return ObjectiveComparison.EQUAL


def accept_annealing_move(
    current: SolutionObjective,
    candidate: SolutionObjective,
    *,
    temperature: float,
    random_draw: float,
) -> bool:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not 0.0 <= random_draw <= 1.0:
        raise ValueError("random_draw must be in [0, 1]")
    if candidate.vehicle_count < current.vehicle_count:
        return True
    if candidate.vehicle_count > current.vehicle_count:
        return False

    comparison = compare_objectives(candidate, current)
    if comparison is not ObjectiveComparison.WORSE:
        return True
    if candidate.key[1] == current.key[1]:
        return False
    distance_delta = candidate.total_distance - current.total_distance
    return random_draw < math.exp(-distance_delta / temperature)
