"""Deterministic EVRPTW route proposal and vehicle-first decision kernel."""

from __future__ import annotations

from collections.abc import Sequence

from txnopt_cases.evrptw.neighborhoods import canonical_neighborhood_plans
from txnopt_cases.evrptw.objective import ObjectiveComparison, compare_objectives
from txnopt_cases.evrptw.oracle import EVRPTWPlan, EVRPTWSolution


class EVRPTWSearchKernel:
    """Generate bounded relocate, swap, and route-merge plan candidates."""

    def __init__(self, *, max_candidates: int = 64) -> None:
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates <= 0
        ):
            raise ValueError("max_candidates must be a positive integer")
        self._max_candidates = max_candidates

    def propose(
        self,
        snapshot: EVRPTWSolution,
        *,
        round_id: int,
        random_tape: Sequence[int],
    ) -> Sequence[EVRPTWPlan]:
        if not snapshot.feasible:
            raise ValueError("EVRPTW proposal snapshot must be feasible")
        if not random_tape:
            raise ValueError("EVRPTW proposal requires a non-empty random tape")
        unique = canonical_neighborhood_plans(snapshot.plan.customer_routes)
        if not unique:
            return ()
        offset = (random_tape[0] + round_id) % len(unique)
        rotated = (*unique[offset:], *unique[:offset])
        return rotated[: self._max_candidates]

    def decide(
        self,
        snapshot: EVRPTWSolution,
        candidates: Sequence[EVRPTWPlan],
        evaluated_states: Sequence[EVRPTWSolution],
        *,
        round_id: int,
    ) -> EVRPTWSolution:
        if len(candidates) != len(evaluated_states):
            raise ValueError("EVRPTW decision inputs must preserve candidate order")
        incumbent = snapshot
        if incumbent.objective_value is None:
            raise ValueError("EVRPTW incumbent must have an objective")
        for result in evaluated_states:
            if not result.feasible or result.objective_value is None:
                continue
            if (
                compare_objectives(result.objective_value, incumbent.objective_value)
                is ObjectiveComparison.BETTER
            ):
                incumbent = result
        return incumbent

__all__ = ["EVRPTWSearchKernel"]
