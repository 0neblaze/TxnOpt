from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Point = tuple[float, float]

def pack_stage052_screening_occurrences(
    events: Sequence[object],
    definition_cache: dict[object, int],
    negative_evidence_cache: dict[object, tuple[object, ...]],
    first_event_id: int,
) -> tuple[
    tuple[
        list[int],
        list[int | None],
        list[float],
        list[float],
        list[int | None],
        list[int],
    ],
    list[tuple[object, int, list[int]]],
    list[int],
]: ...
def pack_stage052_neighborhood_events(
    events: Sequence[object],
    lane_ids: dict[str, int],
    operator_ids: dict[str, int],
    extras_cache: dict[tuple[object, ...], str],
    allowed_fields: frozenset[str],
    missing_extra: object,
    stable_dictionary_id: object,
    json_text: object,
    first_event_id: int,
) -> tuple[tuple[list[object], ...], list[int]]: ...
def distance_matrix(points: npt.ArrayLike) -> npt.NDArray[np.float64]: ...
def route_distance(points: Sequence[Point], route: Sequence[int]) -> float: ...
def two_opt_delta(
    points: Sequence[Point], route: Sequence[int], first: int, second: int
) -> float: ...
def exact_charging_batch_numeric(
    node_kind: npt.NDArray[np.int64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    vehicle: npt.NDArray[np.float64],
    order_offsets: npt.NDArray[np.int64],
    order_indices: npt.NDArray[np.int64],
    deadline_remaining: npt.NDArray[np.float64],
    batch_size: npt.NDArray[np.int64],
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]: ...
def screen_routes_numeric(
    node_kind: npt.NDArray[np.int64],
    demand: npt.NDArray[np.float64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    reachable: npt.NDArray[np.uint8],
    vehicle: npt.NDArray[np.float64],
    route_indices: npt.NDArray[np.int64],
    options: npt.NDArray[np.float64],
    incremental: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
]: ...
def propagate_routes_numeric(
    node_kind: npt.NDArray[np.int64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    vehicle: npt.NDArray[np.float64],
    base_chain: npt.NDArray[np.int64],
    candidate_chain: npt.NDArray[np.int64],
    base_edge_distances: npt.NDArray[np.float64],
    base_earliest_arrivals: npt.NDArray[np.float64],
    base_latest_departures: npt.NDArray[np.float64],
    epsilon: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
]: ...
