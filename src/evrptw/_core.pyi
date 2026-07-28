from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Point = tuple[float, float]

class Stage052ReplayState:
    def __init__(
        self,
        axis_budgets: dict[str, int],
        persistence_ledgers: dict[str, tuple[tuple[int, str], ...]] = {},
    ) -> None: ...
    def consume(self, encoded_columns: dict[str, object]) -> None: ...
    def finish(self) -> dict[str, object]: ...

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
def create_stage052_screening_definition_cache(capacity: int = 262144) -> object: ...
def create_stage052_definition_identity_store(capacity: int) -> object: ...
def register_stage052_definition_identities(
    identity_store: object,
    definitions: Sequence[object],
) -> tuple[object, ...]: ...
def stage052_definition_identity_store_size(identity_store: object) -> int: ...
def pack_stage052_screening_transactions(
    events: Sequence[object],
    definition_cache: object,
    negative_evidence_cache: dict[object, tuple[object, ...]],
    lane_ids: dict[str, int],
    operator_ids: dict[str, int],
    route_ids: dict[str, int],
    resolve_route_id: object,
    stable_dictionary_id: object,
    definition_identity: object,
    first_event_id: int,
) -> tuple[
    tuple[
        list[int],
        list[int],
        list[float],
        list[float],
        list[int | None],
        list[int],
    ],
    list[tuple[int, bytes, bytes, dict[str, object], tuple[object, ...]]],
    list[int],
    list[tuple[str, int, tuple[str, ...]]],
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
def pack_stage052_deferred_sparse_events(
    events: Sequence[object],
    route_ids: dict[str, int],
    lane_ids: dict[str, int],
    operator_ids: dict[str, int],
    route_evaluation_extras_cache: dict[tuple[object, ...], str],
    cache_event_extras_cache: dict[tuple[object, ...], str],
    resolve_route_id: object,
    stable_dictionary_id: object,
    json_text: object,
    first_event_id: int,
) -> tuple[
    tuple[list[object], ...],
    list[tuple[int, int]],
    list[tuple[str, int, tuple[str, ...]]],
]: ...
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
