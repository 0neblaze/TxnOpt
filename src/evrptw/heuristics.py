from __future__ import annotations

import random
from dataclasses import dataclass

from evrptw._core import route_distance, two_opt_delta


@dataclass(frozen=True, slots=True)
class SearchResult:
    route: tuple[int, ...]
    distance: float
    accepted_moves: int
    seed: int


def deterministic_two_opt(
    points: list[tuple[float, float]],
    initial_route: list[int],
    *,
    seed: int,
    iterations: int = 500,
) -> SearchResult:
    if len(initial_route) < 4 or initial_route[0] != initial_route[-1]:
        raise ValueError("route must be a closed tour with at least two interior nodes")

    rng = random.Random(seed)
    route = list(initial_route)
    accepted = 0
    for _ in range(iterations):
        first, second = sorted(rng.sample(range(1, len(route) - 1), 2))
        delta = two_opt_delta(points, route, first, second)
        if delta < -1e-12:
            route[first : second + 1] = reversed(route[first : second + 1])
            accepted += 1

    return SearchResult(
        route=tuple(route),
        distance=float(route_distance(points, route)),
        accepted_moves=accepted,
        seed=seed,
    )
