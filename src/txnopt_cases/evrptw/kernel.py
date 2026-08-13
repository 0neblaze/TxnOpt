"""Deterministic EVRPTW route proposal and vehicle-first decision kernel."""

from __future__ import annotations

from collections.abc import Sequence

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
        routes = snapshot.plan.customer_routes
        proposals = [
            *self._relocations(routes),
            *self._swaps(routes),
            *self._merges(routes),
        ]
        unique = tuple(dict.fromkeys(proposals))
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

    @staticmethod
    def _relocations(
        routes: tuple[tuple[str, ...], ...],
    ) -> tuple[EVRPTWPlan, ...]:
        proposals: list[EVRPTWPlan] = []
        for source_index, source in enumerate(routes):
            for source_position, customer in enumerate(source):
                for target_index, target in enumerate(routes):
                    for target_position in range(len(target) + 1):
                        if source_index == target_index and target_position in {
                            source_position,
                            source_position + 1,
                        }:
                            continue
                        changed = list(routes)
                        shortened = (*source[:source_position], *source[source_position + 1 :])
                        if source_index == target_index:
                            insertion = target_position - (
                                1 if target_position > source_position else 0
                            )
                            changed[source_index] = (
                                *shortened[:insertion],
                                customer,
                                *shortened[insertion:],
                            )
                        else:
                            changed[source_index] = shortened
                            changed[target_index] = (
                                *target[:target_position],
                                customer,
                                *target[target_position:],
                            )
                            changed = [route for route in changed if route]
                        proposals.append(EVRPTWPlan(tuple(changed)))
        return tuple(proposals)

    @staticmethod
    def _swaps(routes: tuple[tuple[str, ...], ...]) -> tuple[EVRPTWPlan, ...]:
        positions = tuple(
            (route_index, position)
            for route_index, route in enumerate(routes)
            for position in range(len(route))
        )
        proposals: list[EVRPTWPlan] = []
        for left_index, left in enumerate(positions):
            for right in positions[left_index + 1 :]:
                changed = [list(route) for route in routes]
                changed[left[0]][left[1]], changed[right[0]][right[1]] = (
                    changed[right[0]][right[1]],
                    changed[left[0]][left[1]],
                )
                proposals.append(EVRPTWPlan(tuple(tuple(route) for route in changed)))
        return tuple(proposals)

    @staticmethod
    def _merges(routes: tuple[tuple[str, ...], ...]) -> tuple[EVRPTWPlan, ...]:
        proposals: list[EVRPTWPlan] = []
        for left in range(len(routes)):
            for right in range(left + 1, len(routes)):
                for merged in (routes[left] + routes[right], routes[right] + routes[left]):
                    changed = [
                        route
                        for index, route in enumerate(routes)
                        if index not in {left, right}
                    ]
                    changed.insert(left, merged)
                    proposals.append(EVRPTWPlan(tuple(changed)))
        return tuple(proposals)


__all__ = ["EVRPTWSearchKernel"]
