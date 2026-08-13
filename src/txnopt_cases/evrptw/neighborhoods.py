"""Pure EVRPTW plan neighborhoods behind one case-owned proposal seam."""

from __future__ import annotations

from txnopt_cases.evrptw.oracle import EVRPTWPlan

CustomerRoutes = tuple[tuple[str, ...], ...]


def relocate_plans(routes: CustomerRoutes) -> tuple[EVRPTWPlan, ...]:
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


def swap_plans(routes: CustomerRoutes) -> tuple[EVRPTWPlan, ...]:
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


def merge_plans(routes: CustomerRoutes) -> tuple[EVRPTWPlan, ...]:
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


def canonical_neighborhood_plans(routes: CustomerRoutes) -> tuple[EVRPTWPlan, ...]:
    """Return the de-duplicated canonical relocate/swap/merge plan order."""

    return tuple(
        dict.fromkeys((*merge_plans(routes), *relocate_plans(routes), *swap_plans(routes)))
    )


__all__ = [
    "canonical_neighborhood_plans",
    "merge_plans",
    "relocate_plans",
    "swap_plans",
]
