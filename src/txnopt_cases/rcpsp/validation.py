"""Independent RCPSP schedule validation without solver callbacks."""

from __future__ import annotations

from dataclasses import dataclass

from txnopt_cases.rcpsp.model import RCPSPInstance, RCPSPSchedule


@dataclass(frozen=True, slots=True)
class RCPSPValidationReport:
    feasible: bool
    makespan: int
    violations: tuple[str, ...]


def validate_schedule(
    instance: RCPSPInstance,
    schedule: RCPSPSchedule,
) -> RCPSPValidationReport:
    violations: list[str] = []
    by_id = instance.by_id
    expected = set(by_id)
    state_ids = set(schedule.state.activity_order)
    scheduled = {item.activity_id: item for item in schedule.activities}
    if state_ids != expected:
        violations.append("state activity set differs from instance")
    if set(scheduled) != expected:
        violations.append("scheduled activity set differs from instance")

    position = {
        activity_id: index
        for index, activity_id in enumerate(schedule.state.activity_order)
    }
    mode_by_activity = schedule.state.mode_by_activity
    for activity_id, activity in by_id.items():
        mode_index = mode_by_activity.get(activity_id)
        item = scheduled.get(activity_id)
        if mode_index is None or not 0 <= mode_index < len(activity.modes):
            violations.append(f"activity {activity_id} has an invalid mode")
            continue
        if item is None:
            continue
        mode = activity.modes[mode_index]
        if item.mode_index != mode_index:
            violations.append(f"activity {activity_id} mode differs from state")
        if item.start < 0 or item.end != item.start + mode.duration:
            violations.append(f"activity {activity_id} has invalid start/end")
        for predecessor in activity.predecessors:
            if position.get(predecessor, len(position)) >= position.get(
                activity_id, -1
            ):
                violations.append(
                    f"activity order violates precedence {predecessor}->{activity_id}"
                )
            predecessor_item = scheduled.get(predecessor)
            if predecessor_item is not None and predecessor_item.end > item.start:
                violations.append(
                    f"schedule violates precedence {predecessor}->{activity_id}"
                )

    ordered_items = [
        scheduled[activity_id]
        for activity_id in schedule.state.activity_order
        if activity_id in scheduled
    ]
    if any(
        left.start > right.start
        for left, right in zip(ordered_items, ordered_items[1:], strict=False)
    ):
        violations.append("scheduled starts violate the declared activity order")

    boundaries = sorted(
        {time for item in scheduled.values() for time in (item.start, item.end)}
    )
    for time in boundaries:
        for resource_index, capacity in enumerate(instance.renewable_capacities):
            demand = 0
            for item in scheduled.values():
                active_activity = by_id.get(item.activity_id)
                if active_activity is None or not (item.start <= time < item.end):
                    continue
                mode_index = mode_by_activity.get(item.activity_id)
                if (
                    mode_index is None
                    or not 0 <= mode_index < len(active_activity.modes)
                ):
                    continue
                demand += active_activity.modes[mode_index].renewable_demands[
                    resource_index
                ]
            if demand > capacity:
                violations.append(
                    f"resource {resource_index} exceeds capacity at time {time}"
                )

    recomputed_makespan = max((item.end for item in scheduled.values()), default=0)
    if schedule.makespan != recomputed_makespan:
        violations.append("declared makespan differs from scheduled activities")
    return RCPSPValidationReport(
        feasible=not violations,
        makespan=recomputed_makespan,
        violations=tuple(violations),
    )


__all__ = ["RCPSPValidationReport", "validate_schedule"]
