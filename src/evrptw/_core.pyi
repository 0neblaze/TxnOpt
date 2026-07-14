from collections.abc import Sequence

Point = tuple[float, float]

def route_distance(points: Sequence[Point], route: Sequence[int]) -> float: ...

def two_opt_delta(
    points: Sequence[Point], route: Sequence[int], first: int, second: int
) -> float: ...

def metal_backend_info() -> dict[str, object]: ...

def metal_transition_batch(
    dx: object,
    dy: object,
    consumption: object,
    velocity: object,
) -> dict[str, object]: ...
