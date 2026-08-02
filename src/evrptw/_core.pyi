from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

def python_random_golden_v1(
    seed: int,
    random_count: int,
    randbelow_bounds: npt.NDArray[np.int64],
    sample_population: int,
    sample_size: int,
    weights: npt.NDArray[np.float64],
    shuffle_size: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    int,
    npt.NDArray[np.int64],
]: ...
def native_sha256_v1(
    payload: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], str]: ...
def native_objective_acceptance_v1(
    current_integer: npt.NDArray[np.int64],
    current_float: npt.NDArray[np.float64],
    candidate_integer: npt.NDArray[np.int64],
    candidate_float: npt.NDArray[np.float64],
    temperatures: npt.NDArray[np.float64],
    random_draws: npt.NDArray[np.float64],
) -> npt.NDArray[np.int64]: ...
def stage04_segment_update_v1(
    weights: npt.NDArray[np.float64],
    reward_sums: npt.NDArray[np.float64],
    calls: npt.NDArray[np.int64],
    options: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]: ...
def changed_candidate_pool_v1(
    operation: int,
    route_offsets: npt.NDArray[np.int64],
    route_indices: npt.NDArray[np.int64],
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]: ...
def assemble_changed_candidate_plans_v1(
    current_route_offsets: npt.NDArray[np.int64],
    current_route_indices: npt.NDArray[np.int64],
    changed_route_indices: npt.NDArray[np.int64],
    change_offsets: npt.NDArray[np.int64],
    change_indices: npt.NDArray[np.int64],
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]: ...
def rank_candidate_plans_v1(
    plan_offsets: npt.NDArray[np.int64],
    route_offsets: npt.NDArray[np.int64],
    route_indices: npt.NDArray[np.int64],
    route_distance_lower_bounds: npt.NDArray[np.float64],
    current_route_offsets: npt.NDArray[np.int64],
    current_route_indices: npt.NDArray[np.int64],
    lexical_rank: npt.NDArray[np.int64],
    attempted_flags: npt.NDArray[np.int64],
    top_k: int,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
]: ...
def changed_candidate_plan_selection_v1(
    operation: int,
    node_kind: npt.NDArray[np.int64],
    demand: npt.NDArray[np.float64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    reachable: npt.NDArray[np.uint8],
    vehicle: npt.NDArray[np.float64],
    lexical_rank: npt.NDArray[np.int64],
    current_route_offsets: npt.NDArray[np.int64],
    current_route_indices: npt.NDArray[np.int64],
    screening_options: npt.NDArray[np.float64],
    negative_offsets: npt.NDArray[np.int64],
    negative_indices: npt.NDArray[np.int64],
    negative_reason_codes: npt.NDArray[np.int64],
    attempted_flags: npt.NDArray[np.int64],
    top_k: int,
    worker_count: int,
) -> tuple[
    object,
    object,
    object,
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    str,
]: ...

class NativeRouteCacheV2:
    def __init__(self, max_entries: int, max_memory_bytes: int) -> None: ...
    def lookup_many(
        self,
        route_offsets: npt.NDArray[np.int64],
        route_indices: npt.NDArray[np.int64],
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.uint8],
        npt.NDArray[np.int64],
    ]: ...
    def begin_store_exact_many_atomic(
        self,
        route_offsets: npt.NDArray[np.int64],
        route_indices: npt.NDArray[np.int64],
        path_offsets: npt.NDArray[np.int64],
        path_indices: npt.NDArray[np.int64],
        result_statuses: npt.NDArray[np.int64],
        reason_codes: npt.NDArray[np.int64],
        result_metrics: npt.NDArray[np.float64],
        label_counters: npt.NDArray[np.int64],
        semantic_hashes: npt.NDArray[np.uint8],
        entry_bytes: npt.NDArray[np.int64],
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]: ...
    def lookup_exact_many(
        self,
        route_offsets: npt.NDArray[np.int64],
        route_indices: npt.NDArray[np.int64],
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.uint8],
        npt.NDArray[np.int64],
    ]: ...
    def begin_protocol_transaction(self) -> None: ...
    def commit_protocol_transaction(self) -> npt.NDArray[np.int64]: ...
    def rollback_protocol_transaction(self) -> npt.NDArray[np.int64]: ...
    def begin_store_many_atomic(
        self,
        route_offsets: npt.NDArray[np.int64],
        route_indices: npt.NDArray[np.int64],
        semantic_hashes: npt.NDArray[np.uint8],
        entry_bytes: npt.NDArray[np.int64],
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]: ...
    def commit_store_batch(self) -> npt.NDArray[np.int64]: ...
    def rollback_store_batch(self) -> npt.NDArray[np.int64]: ...
    def snapshot(self) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.uint8],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]: ...

class NativeNegativeRouteCacheV2:
    def __init__(self, capacity: int) -> None: ...
    def lookup_many(
        self,
        route_offsets: npt.NDArray[np.int64],
        route_indices: npt.NDArray[np.int64],
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]: ...
    def begin_store_many_atomic(
        self,
        route_offsets: npt.NDArray[np.int64],
        route_indices: npt.NDArray[np.int64],
        reason_codes: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]: ...
    def commit_store_batch(self) -> npt.NDArray[np.int64]: ...
    def rollback_store_batch(self) -> npt.NDArray[np.int64]: ...
    def snapshot(self) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]: ...

class NativeBudgetStateV2:
    def __init__(self, exact_budget: int, round_budget: int) -> None: ...
    def begin_round(self, lane_id: int, iteration: int) -> npt.NDArray[np.int64]: ...
    def finish_round(self) -> npt.NDArray[np.int64]: ...
    def reserve_round(
        self, requested: int, atomic: bool
    ) -> npt.NDArray[np.int64]: ...
    def reserve_exact(self, requested: int) -> npt.NDArray[np.int64]: ...
    def complete_exact(self, count: int) -> npt.NDArray[np.int64]: ...
    def interrupt_exact(self, count: int) -> npt.NDArray[np.int64]: ...
    def snapshot(self) -> npt.NDArray[np.int64]: ...
    def restore(
        self, snapshot: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]: ...
    def state(self) -> npt.NDArray[np.int64]: ...

__build_git_revision__: str

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
def stage052_screening_definition_cache_size(definition_cache: object) -> int: ...
def stage052_screening_definition_cache_capacity(definition_cache: object) -> int: ...
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
def screen_route_batch_transaction_v2(
    node_kind: npt.NDArray[np.int64],
    demand: npt.NDArray[np.float64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    reachable: npt.NDArray[np.uint8],
    vehicle: npt.NDArray[np.float64],
    route_offsets: npt.NDArray[np.int64],
    route_indices: npt.NDArray[np.int64],
    candidate_ids: npt.NDArray[np.int64],
    options: npt.NDArray[np.float64],
    incremental: npt.NDArray[np.float64],
    negative_offsets: npt.NDArray[np.int64],
    negative_indices: npt.NDArray[np.int64],
    negative_reason_codes: npt.NDArray[np.int64],
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    str,
]: ...
def candidate_round_transaction_v1(
    node_kind: npt.NDArray[np.int64],
    demand: npt.NDArray[np.float64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    reachable: npt.NDArray[np.uint8],
    vehicle: npt.NDArray[np.float64],
    route_offsets: npt.NDArray[np.int64],
    route_indices: npt.NDArray[np.int64],
    candidate_ids: npt.NDArray[np.int64],
    lexical_rank: npt.NDArray[np.int64],
    options: npt.NDArray[np.float64],
    incremental: npt.NDArray[np.float64],
    negative_offsets: npt.NDArray[np.int64],
    negative_indices: npt.NDArray[np.int64],
    negative_reason_codes: npt.NDArray[np.int64],
    cache_hit_flags: npt.NDArray[np.int64],
    control: npt.NDArray[np.int64],
    deadline_remaining: npt.NDArray[np.float64],
    batch_size: npt.NDArray[np.int64],
    context_ids: npt.NDArray[np.int64],
) -> tuple[
    tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        str,
    ],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    str,
]: ...
candidate_round_transaction_v2 = candidate_round_transaction_v1
def full_native_alns_v1(
    node_kind: npt.NDArray[np.int64],
    demand: npt.NDArray[np.float64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    vehicle: npt.NDArray[np.float64],
    lexical_rank: npt.NDArray[np.int64],
    initial_route_offsets: npt.NDArray[np.int64],
    initial_route_indices: npt.NDArray[np.int64],
    control: npt.NDArray[np.int64],
    deadline_remaining: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    str,
]: ...
def full_native_alns_v2(
    node_kind: npt.NDArray[np.int64],
    demand: npt.NDArray[np.float64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    vehicle: npt.NDArray[np.float64],
    lexical_rank: npt.NDArray[np.int64],
    initial_route_offsets: npt.NDArray[np.int64],
    initial_route_indices: npt.NDArray[np.int64],
    control: npt.NDArray[np.int64],
    deadline_remaining: npt.NDArray[np.float64],
    protocol_control: npt.NDArray[np.int64],
    protocol_options: npt.NDArray[np.float64],
    stage04_integer: npt.NDArray[np.int64],
    stage04_float: npt.NDArray[np.float64],
    operator_integer: npt.NDArray[np.int64],
    operator_float: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    str,
]: ...
def full_native_initialize_v2(
    node_kind: npt.NDArray[np.int64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    vehicle: npt.NDArray[np.float64],
    initial_route_offsets: npt.NDArray[np.int64],
    initial_route_indices: npt.NDArray[np.int64],
    control: npt.NDArray[np.int64],
    deadline_remaining: npt.NDArray[np.float64],
) -> tuple[
    tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
]: ...
def run_host_scheduler_service_v1(socket_path: str, worker_threads: int) -> None: ...
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
