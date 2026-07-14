from collections.abc import Sequence

Point = tuple[float, float]

def route_distance(points: Sequence[Point], route: Sequence[int]) -> float: ...

def two_opt_delta(
    points: Sequence[Point], route: Sequence[int], first: int, second: int
) -> float: ...
