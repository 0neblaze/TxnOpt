"""RCPSP domain values shared by kernels, the CP-SAT oracle, and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ActivityMode:
    duration: int
    renewable_demands: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.duration, bool)
            or not isinstance(self.duration, int)
            or self.duration < 0
        ):
            raise ValueError("mode duration must be a non-negative integer")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.renewable_demands
        ):
            raise ValueError("renewable demands must be non-negative integers")


@dataclass(frozen=True, slots=True)
class Activity:
    activity_id: int
    predecessors: tuple[int, ...]
    modes: tuple[ActivityMode, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.activity_id, bool)
            or not isinstance(self.activity_id, int)
            or self.activity_id < 0
        ):
            raise ValueError("activity_id must be a non-negative integer")
        if len(set(self.predecessors)) != len(self.predecessors) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.predecessors
        ):
            raise ValueError("predecessors must be unique non-negative integers")
        if self.activity_id in self.predecessors:
            raise ValueError("an activity cannot precede itself")
        if not self.modes:
            raise ValueError("every activity requires at least one mode")
        widths = {len(mode.renewable_demands) for mode in self.modes}
        if len(widths) != 1:
            raise ValueError("all modes of one activity must share a resource width")


@dataclass(frozen=True, slots=True)
class RCPSPInstance:
    name: str
    activities: tuple[Activity, ...]
    renewable_capacities: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("instance name cannot be empty")
        if not self.activities:
            raise ValueError("instance requires activities")
        identifiers = tuple(activity.activity_id for activity in self.activities)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("activity identifiers must be unique")
        identifier_set = set(identifiers)
        if any(
            predecessor not in identifier_set
            for activity in self.activities
            for predecessor in activity.predecessors
        ):
            raise ValueError("predecessor references an unknown activity")
        if not self.renewable_capacities or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.renewable_capacities
        ):
            raise ValueError("renewable capacities must be non-negative integers")
        if any(
            len(mode.renewable_demands) != len(self.renewable_capacities)
            for activity in self.activities
            for mode in activity.modes
        ):
            raise ValueError("mode resource width differs from instance capacities")

    @property
    def by_id(self) -> dict[int, Activity]:
        return {activity.activity_id: activity for activity in self.activities}

    @property
    def digest(self) -> str:
        payload = {
            "name": self.name,
            "capacities": self.renewable_capacities,
            "activities": [
                {
                    "id": activity.activity_id,
                    "predecessors": activity.predecessors,
                    "modes": [
                        {
                            "duration": mode.duration,
                            "renewable_demands": mode.renewable_demands,
                        }
                        for mode in activity.modes
                    ],
                }
                for activity in sorted(self.activities, key=lambda item: item.activity_id)
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class RCPSPState:
    """A precedence-feasible order paired positionally with one mode index."""

    activity_order: tuple[int, ...]
    mode_vector: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.activity_order:
            raise ValueError("activity_order cannot be empty")
        if len(set(self.activity_order)) != len(self.activity_order):
            raise ValueError("activity_order must contain each activity once")
        if any(
            isinstance(activity, bool) or not isinstance(activity, int) or activity < 0
            for activity in self.activity_order
        ):
            raise ValueError("activity identifiers must be non-negative integers")
        if len(self.mode_vector) != len(self.activity_order):
            raise ValueError("mode_vector must align with activity_order")
        if any(
            isinstance(mode, bool) or not isinstance(mode, int) or mode < 0
            for mode in self.mode_vector
        ):
            raise ValueError("mode indices must be non-negative integers")

    @property
    def mode_by_activity(self) -> dict[int, int]:
        return dict(zip(self.activity_order, self.mode_vector, strict=True))


@dataclass(frozen=True, slots=True)
class ScheduledActivity:
    activity_id: int
    mode_index: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class RCPSPSchedule:
    state: RCPSPState
    activities: tuple[ScheduledActivity, ...]
    makespan: int

    def __post_init__(self) -> None:
        if len({item.activity_id for item in self.activities}) != len(self.activities):
            raise ValueError("scheduled activities must be unique")
        if (
            isinstance(self.makespan, bool)
            or not isinstance(self.makespan, int)
            or self.makespan < 0
        ):
            raise ValueError("makespan must be a non-negative integer")


__all__ = [
    "Activity",
    "ActivityMode",
    "RCPSPInstance",
    "RCPSPSchedule",
    "RCPSPState",
    "ScheduledActivity",
]
