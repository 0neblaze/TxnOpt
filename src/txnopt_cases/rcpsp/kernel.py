"""Deterministic RCPSP proposal and decision kernel."""

from __future__ import annotations

from collections.abc import Sequence

from txnopt_cases.rcpsp.model import RCPSPInstance, RCPSPSchedule, RCPSPState


class RCPSPSearchKernel:
    """Generate activity-block reinsertion and mode-change candidates."""

    def __init__(
        self,
        instance: RCPSPInstance,
        *,
        max_block_size: int = 3,
        max_candidates: int = 64,
    ) -> None:
        if (
            isinstance(max_block_size, bool)
            or not isinstance(max_block_size, int)
            or max_block_size <= 0
        ):
            raise ValueError("max_block_size must be a positive integer")
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates <= 0
        ):
            raise ValueError("max_candidates must be a positive integer")
        self._instance = instance
        self._max_block_size = max_block_size
        self._max_candidates = max_candidates

    @property
    def admission_limit(self) -> int:
        return self._max_candidates

    def propose(
        self,
        snapshot: RCPSPSchedule,
        *,
        round_id: int,
        random_tape: Sequence[int],
    ) -> Sequence[RCPSPState]:
        if not random_tape:
            raise ValueError("RCPSP proposal requires a non-empty random tape")
        order = snapshot.state.activity_order
        mode_by_activity = snapshot.state.mode_by_activity
        proposals: list[RCPSPState] = []

        for block_size in range(1, min(self._max_block_size, len(order)) + 1):
            for start in range(0, len(order) - block_size + 1):
                block = order[start : start + block_size]
                remainder = (*order[:start], *order[start + block_size :])
                for insertion in range(len(remainder) + 1):
                    moved = (
                        *remainder[:insertion],
                        *block,
                        *remainder[insertion:],
                    )
                    if moved == order:
                        continue
                    proposals.append(
                        RCPSPState(
                            activity_order=moved,
                            mode_vector=tuple(mode_by_activity[item] for item in moved),
                        )
                    )

        by_id = self._instance.by_id
        for position, activity_id in enumerate(order):
            for mode_index in range(len(by_id[activity_id].modes)):
                if mode_index == mode_by_activity[activity_id]:
                    continue
                modes = list(snapshot.state.mode_vector)
                modes[position] = mode_index
                proposals.append(RCPSPState(order, tuple(modes)))

        unique = tuple(dict.fromkeys(proposals))
        if not unique:
            return ()
        offset = (random_tape[0] + round_id) % len(unique)
        rotated = (*unique[offset:], *unique[:offset])
        return rotated[: self._max_candidates * 4]

    def decide(
        self,
        snapshot: RCPSPSchedule,
        candidates: Sequence[RCPSPState],
        evaluated_states: Sequence[RCPSPSchedule],
        *,
        round_id: int,
    ) -> RCPSPSchedule:
        if len(candidates) != len(evaluated_states):
            raise ValueError("RCPSP decision inputs must preserve candidate order")
        best = min(
            evaluated_states,
            key=lambda state: (
                state.makespan,
                state.state.activity_order,
                state.state.mode_vector,
            ),
            default=snapshot,
        )
        return best if best.makespan < snapshot.makespan else snapshot


__all__ = ["RCPSPSearchKernel"]
