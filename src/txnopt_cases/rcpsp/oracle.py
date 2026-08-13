"""Fixed-seed, single-thread CP-SAT exact scheduling oracle for RCPSP."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence

from ortools.sat.python import cp_model

from txnopt_cases.rcpsp.model import (
    RCPSPInstance,
    RCPSPSchedule,
    RCPSPState,
    ScheduledActivity,
)
from txnopt_cases.rcpsp.validation import validate_schedule


class RCPSPOracle:
    """Exact schedule repair for a fixed order and mode vector."""

    deterministic = True
    parallel_safe = True
    internal_parallelism = False

    def __init__(self, instance: RCPSPInstance, *, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("RCPSP oracle seed must be a non-negative integer")
        self._instance = instance
        self._seed = seed

    def stable_key(self, candidate: RCPSPState) -> str:
        payload = {
            "instance": self._instance.digest,
            "activity_order": candidate.activity_order,
            "mode_vector": candidate.mode_vector,
        }
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()

    def state_digest(self, state: RCPSPSchedule) -> str:
        payload = {
            "instance": self._instance.digest,
            "state": {
                "activity_order": state.state.activity_order,
                "mode_vector": state.state.mode_vector,
            },
            "activities": [
                (item.activity_id, item.mode_index, item.start, item.end)
                for item in state.activities
            ],
            "makespan": state.makespan,
        }
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()

    def work_units(self, candidate: RCPSPState) -> int:
        return 1

    def screen(self, candidates: Sequence[RCPSPState]) -> Sequence[bool]:
        return tuple(self._screen_one(candidate) for candidate in candidates)

    def lower_bound(self, candidate: RCPSPState) -> int:
        """Return the precedence-chain duration lower bound for one mode vector."""

        if not self._screen_one(candidate):
            raise ValueError("cannot compute a lower bound for an invalid candidate")
        by_id = self._instance.by_id
        modes = candidate.mode_by_activity
        earliest_end: dict[int, int] = {}
        for activity_id in candidate.activity_order:
            activity = by_id[activity_id]
            earliest_start = max(
                (earliest_end[predecessor] for predecessor in activity.predecessors),
                default=0,
            )
            earliest_end[activity_id] = (
                earliest_start + activity.modes[modes[activity_id]].duration
            )
        return max(earliest_end.values(), default=0)

    def evaluate_batch(
        self,
        candidates: Sequence[RCPSPState],
        *,
        work_budget: int,
        deadline_ns: int | None,
    ) -> Sequence[RCPSPSchedule]:
        if work_budget != len(candidates):
            raise ValueError("RCPSP work budget must cover the complete ordered batch")
        schedules: list[RCPSPSchedule] = []
        for candidate in candidates:
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("RCPSP exact batch crossed its deadline")
            schedules.append(self._solve(candidate, deadline_ns=deadline_ns))
        return tuple(schedules)

    def validate(self, state: RCPSPSchedule) -> None:
        report = validate_schedule(self._instance, state)
        if not report.feasible:
            raise ValueError(f"invalid RCPSP schedule: {report.violations}")

    def objective(self, state: RCPSPSchedule) -> int:
        self.validate(state)
        return state.makespan

    def _screen_one(self, candidate: RCPSPState) -> bool:
        by_id = self._instance.by_id
        if set(candidate.activity_order) != set(by_id):
            return False
        position = {
            activity_id: index
            for index, activity_id in enumerate(candidate.activity_order)
        }
        modes = candidate.mode_by_activity
        for activity_id, activity in by_id.items():
            mode_index = modes.get(activity_id)
            if mode_index is None or not 0 <= mode_index < len(activity.modes):
                return False
            if any(
                position[predecessor] >= position[activity_id]
                for predecessor in activity.predecessors
            ):
                return False
            mode = activity.modes[mode_index]
            if any(
                demand > capacity
                for demand, capacity in zip(
                    mode.renewable_demands,
                    self._instance.renewable_capacities,
                    strict=True,
                )
            ):
                return False
        return True

    def _solve(
        self,
        candidate: RCPSPState,
        *,
        deadline_ns: int | None,
    ) -> RCPSPSchedule:
        if not self._screen_one(candidate):
            raise ValueError("RCPSP candidate failed safe screening")
        by_id = self._instance.by_id
        modes = candidate.mode_by_activity
        horizon = sum(
            by_id[activity_id].modes[modes[activity_id]].duration
            for activity_id in candidate.activity_order
        )
        model = cp_model.CpModel()
        starts: dict[int, cp_model.IntVar] = {}
        ends: dict[int, cp_model.IntVar] = {}
        intervals: dict[int, cp_model.IntervalVar] = {}
        for activity_id in candidate.activity_order:
            duration = by_id[activity_id].modes[modes[activity_id]].duration
            starts[activity_id] = model.new_int_var(0, horizon, f"start_{activity_id}")
            ends[activity_id] = model.new_int_var(0, horizon, f"end_{activity_id}")
            intervals[activity_id] = model.new_interval_var(
                starts[activity_id],
                duration,
                ends[activity_id],
                f"interval_{activity_id}",
            )
        for activity_id, activity in by_id.items():
            for predecessor in activity.predecessors:
                model.add(ends[predecessor] <= starts[activity_id])
        for left, right in zip(
            candidate.activity_order,
            candidate.activity_order[1:],
            strict=False,
        ):
            model.add(starts[left] <= starts[right])
        for resource_index, capacity in enumerate(self._instance.renewable_capacities):
            model.add_cumulative(
                [intervals[activity_id] for activity_id in candidate.activity_order],
                [
                    by_id[activity_id]
                    .modes[modes[activity_id]]
                    .renewable_demands[resource_index]
                    for activity_id in candidate.activity_order
                ],
                capacity,
            )
        makespan = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(makespan, list(ends.values()))
        model.minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = self._seed
        if deadline_ns is not None:
            remaining = (deadline_ns - time.monotonic_ns()) / 1_000_000_000
            if remaining <= 0.0:
                raise TimeoutError("RCPSP exact repair has no remaining time")
            solver.parameters.max_time_in_seconds = remaining
        status = solver.solve(model)
        if status != cp_model.OPTIMAL:
            if deadline_ns is not None and status in {
                cp_model.UNKNOWN,
                cp_model.FEASIBLE,
            }:
                raise TimeoutError("RCPSP exact repair did not prove optimal before deadline")
            raise RuntimeError(f"RCPSP exact repair was not optimal: {solver.status_name(status)}")
        schedule = RCPSPSchedule(
            state=candidate,
            activities=tuple(
                ScheduledActivity(
                    activity_id=activity_id,
                    mode_index=modes[activity_id],
                    start=solver.value(starts[activity_id]),
                    end=solver.value(ends[activity_id]),
                )
                for activity_id in candidate.activity_order
            ),
            makespan=solver.value(makespan),
        )
        self.validate(schedule)
        return schedule


__all__ = ["RCPSPOracle"]
