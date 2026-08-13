"""Strict parser for PSPLIB single-mode ``.sm`` RCPSP instances."""

from __future__ import annotations

import re
from pathlib import Path

from txnopt_cases.rcpsp.model import Activity, ActivityMode, RCPSPInstance, RCPSPState

_INTEGER_ROW = re.compile(r"^\s*\d+(?:\s+\d+)*\s*$")


def parse_psplib_sm(path: str | Path) -> RCPSPInstance:
    """Parse one PSPLIB single-mode instance without trusting section order gaps."""

    source = Path(path).resolve(strict=True)
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    precedence_start = _section(lines, "PRECEDENCE RELATIONS:")
    requests_start = _section(lines, "REQUESTS/DURATIONS:")
    capacities_start = _section(lines, "RESOURCEAVAILABILITIES:")
    if not precedence_start < requests_start < capacities_start:
        raise ValueError("PSPLIB sections are not in canonical order")

    capacity_rows = [
        tuple(int(value) for value in line.split())
        for line in lines[capacities_start + 1 :]
        if _INTEGER_ROW.fullmatch(line)
    ]
    if len(capacity_rows) != 1 or not capacity_rows[0]:
        raise ValueError("PSPLIB resource capacities must contain one numeric row")
    capacities = capacity_rows[0]

    successors: dict[int, tuple[int, ...]] = {}
    mode_counts: dict[int, int] = {}
    for line_number, line in enumerate(
        lines[precedence_start + 1 : requests_start],
        start=precedence_start + 2,
    ):
        if not _INTEGER_ROW.fullmatch(line):
            continue
        values = tuple(int(value) for value in line.split())
        if len(values) < 3:
            raise ValueError(f"invalid PSPLIB precedence row at line {line_number}")
        activity_id, mode_count, successor_count, *successor_values = values
        if activity_id in successors:
            raise ValueError(f"duplicate PSPLIB activity {activity_id}")
        if mode_count <= 0 or successor_count != len(successor_values):
            raise ValueError(f"invalid PSPLIB precedence counts at line {line_number}")
        successors[activity_id] = tuple(successor_values)
        mode_counts[activity_id] = mode_count
    if not successors:
        raise ValueError("PSPLIB precedence section contains no activities")

    mode_rows: dict[tuple[int, int], ActivityMode] = {}
    previous_activity: int | None = None
    width = len(capacities)
    for line_number, line in enumerate(
        lines[requests_start + 1 : capacities_start],
        start=requests_start + 2,
    ):
        if not _INTEGER_ROW.fullmatch(line):
            continue
        values = tuple(int(value) for value in line.split())
        if len(values) == width + 3:
            activity_id, mode_number, duration, *demands = values
            previous_activity = activity_id
        elif len(values) == width + 2 and previous_activity is not None:
            activity_id = previous_activity
            mode_number, duration, *demands = values
        else:
            raise ValueError(f"invalid PSPLIB request row at line {line_number}")
        key = (activity_id, mode_number)
        if key in mode_rows or activity_id not in successors:
            raise ValueError(f"invalid PSPLIB activity/mode identity at line {line_number}")
        mode_rows[key] = ActivityMode(duration, tuple(demands))

    predecessors: dict[int, list[int]] = {activity_id: [] for activity_id in successors}
    for activity_id, raw_successors in successors.items():
        for successor in raw_successors:
            if successor not in predecessors:
                raise ValueError(f"PSPLIB successor {successor} is not an activity")
            predecessors[successor].append(activity_id)

    activities: list[Activity] = []
    for activity_id in sorted(successors):
        modes = tuple(
            mode_rows[(activity_id, mode_number)]
            for mode_number in range(1, mode_counts[activity_id] + 1)
            if (activity_id, mode_number) in mode_rows
        )
        if len(modes) != mode_counts[activity_id]:
            raise ValueError(f"PSPLIB activity {activity_id} has incomplete mode rows")
        activities.append(
            Activity(
                activity_id=activity_id,
                predecessors=tuple(sorted(predecessors[activity_id])),
                modes=modes,
            )
        )
    return RCPSPInstance(source.stem, tuple(activities), capacities)


def precedence_feasible_initial_state(instance: RCPSPInstance) -> RCPSPState:
    """Return the lexicographically smallest topological order and first modes."""

    remaining = {
        activity.activity_id: set(activity.predecessors)
        for activity in instance.activities
    }
    order: list[int] = []
    while remaining:
        ready = min(
            (activity_id for activity_id, predecessors in remaining.items() if not predecessors),
            default=None,
        )
        if ready is None:
            raise ValueError("RCPSP precedence graph contains a cycle")
        order.append(ready)
        del remaining[ready]
        for predecessors in remaining.values():
            predecessors.discard(ready)
    return RCPSPState(tuple(order), (0,) * len(order))


def _section(lines: list[str], heading: str) -> int:
    matches = [index for index, line in enumerate(lines) if line.strip() == heading]
    if len(matches) != 1:
        raise ValueError(f"PSPLIB file requires exactly one {heading} section")
    return matches[0]


__all__ = ["parse_psplib_sm", "precedence_feasible_initial_state"]
