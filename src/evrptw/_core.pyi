from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Point = tuple[float, float]

def distance_matrix(points: npt.ArrayLike) -> npt.NDArray[np.float64]: ...

def route_distance(points: Sequence[Point], route: Sequence[int]) -> float: ...

def two_opt_delta(
    points: Sequence[Point], route: Sequence[int], first: int, second: int
) -> float: ...
