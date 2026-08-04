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
def dynamic_removal_selection_v2(
    customer_count: int,
    stagnation_iterations: int,
    iteration: int,
    thresholds: npt.NDArray[np.int64],
    fractions: npt.NDArray[np.float64],
    global_best_reset: bool,
) -> npt.NDArray[np.int64]: ...
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
def prepare_candidate_plans_v2(
    plan_offsets: npt.NDArray[np.int64],
    route_offsets: npt.NDArray[np.int64],
    route_indices: npt.NDArray[np.int64],
    expected_customer_indices: npt.NDArray[np.int64],
    node_kind: npt.NDArray[np.int64],
    lexical_rank: npt.NDArray[np.int64],
    complete_customer_indices: npt.NDArray[np.int64],
    customer_kind: int,
    allow_partial_customer_coverage: bool,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]: ...
def decide_candidate_plans_v2(
    plan_offsets: npt.NDArray[np.int64],
    coverage_eligible: npt.NDArray[np.int64],
    screening_passed: npt.NDArray[np.int64],
    attempted_flags: npt.NDArray[np.int64],
    current_route_count: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ...
def order_feasible_candidate_plans_v2(
    plan_offsets: npt.NDArray[np.int64],
    route_offsets: npt.NDArray[np.int64],
    route_indices: npt.NDArray[np.int64],
    objective_integer: npt.NDArray[np.int64],
    objective_float: npt.NDArray[np.float64],
    lexical_rank: npt.NDArray[np.int64],
    feasible_plan_ids: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]: ...
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
    def inject_protocol_journal_failure_once(self) -> None: ...
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
    def exact_remaining(self) -> int: ...
    def candidate_round_remaining(self) -> int: ...
    def complete_exact(self, count: int) -> npt.NDArray[np.int64]: ...
    def interrupt_exact(self, count: int) -> npt.NDArray[np.int64]: ...
    def snapshot(self) -> npt.NDArray[np.int64]: ...
    def restore(
        self, snapshot: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]: ...
    def state(self) -> npt.NDArray[np.int64]: ...

class NativeAttemptedPlanSetV2:
    def __init__(self) -> None: ...
    def lookup(
        self,
        plan_offsets: npt.NDArray[np.int64],
        route_offsets: npt.NDArray[np.int64],
        route_indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]: ...
    def begin_mark_many_atomic(
        self,
        plan_offsets: npt.NDArray[np.int64],
        route_offsets: npt.NDArray[np.int64],
        route_indices: npt.NDArray[np.int64],
        plan_ids: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]: ...
    def commit_mark_batch(self) -> int: ...
    def rollback_mark_batch(self) -> int: ...
    def size(self) -> int: ...

class NativeSearchEngineV2:
    def __init__(
        self,
        exact_budget: int,
        round_budget: int,
        cache_entries: int,
        cache_memory_bytes: int,
        negative_cache_entries: int,
        proposal_top_k: int,
        screening_epsilon: float,
        worker_count: int,
    ) -> None: ...
    def configure_node_names(
        self,
        name_offsets: npt.NDArray[np.int64],
        name_bytes: npt.NDArray[np.uint8],
    ) -> None: ...
    def initialize(
        self,
        node_kind: npt.NDArray[np.int64],
        demand: npt.NDArray[np.float64],
        ready_time: npt.NDArray[np.float64],
        due_date: npt.NDArray[np.float64],
        service_time: npt.NDArray[np.float64],
        distance: npt.NDArray[np.float64],
        reachable: npt.NDArray[np.uint8],
        vehicle: npt.NDArray[np.float64],
        lexical_rank: npt.NDArray[np.int64],
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
    def evaluate_plans(
        self,
        plan_offsets: npt.NDArray[np.int64],
        route_offsets: npt.NDArray[np.int64],
        route_indices: npt.NDArray[np.int64],
        context_ids: npt.NDArray[np.int64],
        deadline_remaining: npt.NDArray[np.float64],
        batch_size: npt.NDArray[np.int64],
        expected_customer_indices: npt.NDArray[np.int64],
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        str,
    ]: ...
    def legacy_route_elimination_probe(
        self,
        iteration: int,
        max_attempts: int,
        route_change_limit: int,
        deadline_remaining: npt.NDArray[np.float64],
        batch_size: npt.NDArray[np.int64],
        defer_acceptance: bool = False,
    ) -> tuple[object, ...]: ...
    def apply_legacy_candidate(
        self,
        temperature: float,
        random_draw: float,
    ) -> tuple[int, int, int]: ...
    def legacy_vehicle_reduction_refinement(
        self,
        iteration: int,
        evaluation_budget: int,
        deadline_remaining: npt.NDArray[np.float64],
        batch_size: npt.NDArray[np.int64],
    ) -> tuple[object, ...]: ...
    def quality_changed_probe(
        self,
        operation: int,
        iteration: int,
        deadline_remaining: npt.NDArray[np.float64],
        batch_size: npt.NDArray[np.int64],
    ) -> tuple[object, ...]: ...
    def run_three_lane_bootstrap(
        self,
        max_route_elimination_attempts: int,
        refinement_budget: int,
        route_change_limit: int,
        thresholds: npt.NDArray[np.int64],
        fractions: npt.NDArray[np.float64],
        deadline_remaining: npt.NDArray[np.float64],
        batch_size: npt.NDArray[np.int64],
    ) -> tuple[object, ...]: ...
    def run_three_lane_followup(
        self,
        iteration: int,
        max_iterations: int,
        removal_fraction: float,
        route_elimination_max_attempts: int,
        refinement_budget: int,
        route_segment_min_length: int,
        route_segment_max_length: int,
        route_segment_budget: int,
        ejection_chain_budget: int,
        ejection_chain_max_depth: int,
        ejection_chain_beam_width: int,
        route_change_limit: int,
        thresholds: npt.NDArray[np.int64],
        fractions: npt.NDArray[np.float64],
        deadline_remaining: npt.NDArray[np.float64],
        batch_size: npt.NDArray[np.int64],
    ) -> tuple[object, ...]: ...
    def constraint_probe(
        self,
        operation: int,
        requested_count: int,
        seed: int,
        context_ids: npt.NDArray[np.int64],
        deadline_remaining: npt.NDArray[np.float64],
        batch_size: npt.NDArray[np.int64],
        route_change_limit: int,
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
        tuple[
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
        ]
        | None,
        tuple[
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            npt.NDArray[np.float64],
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            str,
        ]
        | None,
    ]: ...
    def apply_last_candidate(
        self,
        temperature: float,
        random_draw: float,
    ) -> tuple[int, int, int]: ...
    def configure_stage04(
        self,
        integer_config: npt.NDArray[np.int64],
        float_config: npt.NDArray[np.float64],
    ) -> None: ...
    def initialize_stage04_search(
        self,
        deadline_remaining: npt.NDArray[np.float64],
        batch_size: npt.NDArray[np.int64],
    ) -> tuple[object, ...]: ...
    def constraint_stage04_state(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]: ...
    def full_stage04_state(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]: ...
    def record_constraint_stage04_outcome(
        self,
        iteration: int,
        operation: int,
        accepted: bool,
        comparison: int,
        is_global_best: bool,
        vehicle_reduction: bool,
    ) -> None: ...
    def finish_stage04_iteration(
        self,
        iteration: int,
        budget_boundary: bool,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
    ]: ...
    def constraint_iteration(
        self,
        iteration: int,
        stagnation_iterations: int,
        global_best_reset: bool,
        thresholds: npt.NDArray[np.int64],
        fractions: npt.NDArray[np.float64],
        deadline_remaining: npt.NDArray[np.float64],
        batch_size: npt.NDArray[np.int64],
        route_change_limit: int,
    ) -> tuple[
        npt.NDArray[np.int64],
        tuple[
            tuple[
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                npt.NDArray[np.float64],
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
            ],
            tuple[
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
            ]
            | None,
            tuple[
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                npt.NDArray[np.float64],
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                npt.NDArray[np.int64],
                str,
            ]
            | None,
        ],
        npt.NDArray[np.int64],
    ]: ...
    def dynamic_removal_selection_state(
        self,
        expected_iteration: int,
    ) -> npt.NDArray[np.int64]: ...
    def constraint_removal_state(
        self,
        expected_iteration: int,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]: ...
    def constraint_repair_state(
        self,
        expected_iteration: int,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]: ...
    def run_constraint_search(
        self,
        start_iteration: int,
        iteration_count: int,
        initial_stagnation_iterations: int,
        thresholds: npt.NDArray[np.int64],
        fractions: npt.NDArray[np.float64],
        deadline_remaining: npt.NDArray[np.float64],
        batch_size: npt.NDArray[np.int64],
        route_change_limit: int,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.uint8],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        str,
    ]: ...
    def run_global_search(
        self,
        start_iteration: int,
        iteration_count: int,
        initial_stagnation_iterations: int,
        thresholds: npt.NDArray[np.int64],
        fractions: npt.NDArray[np.float64],
        deadline_remaining: npt.NDArray[np.float64],
        batch_size: npt.NDArray[np.int64],
        route_change_limit: int,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        str,
    ]: ...
    def inject_global_search_envelope_failure_once(self) -> None: ...
    def initialized(self) -> bool: ...
    def inject_commit_failure_once(self, step: int) -> None: ...
    def inject_constraint_probe_envelope_failure_once(self) -> None: ...
    def inject_constraint_iteration_deadline_before_commit_once(self) -> None: ...
    def inject_constraint_search_deadline_after_completed_once(
        self, completed_iterations: int
    ) -> None: ...
    def state(
        self,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        int,
        npt.NDArray[np.int64],
    ]: ...
    def cache_execution_coverage_receipt(
        self,
    ) -> tuple[npt.NDArray[np.int64], str]: ...
    def solution_state(
        self,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
    ]: ...
    def lane_solution_state(
        self,
        lane: int,
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
    ]: ...
    def best_solution_payload(
        self,
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
        ],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
    ]: ...

__build_git_revision__: str

def stage052_native_architecture_capabilities_v2() -> npt.NDArray[np.int64]: ...

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
def candidate_control_repair_v2(
    node_kind: npt.NDArray[np.int64],
    demand: npt.NDArray[np.float64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    reachable: npt.NDArray[np.uint8],
    vehicle: npt.NDArray[np.float64],
    lexical_rank: npt.NDArray[np.int64],
    partial_route_offsets: npt.NDArray[np.int64],
    partial_route_indices: npt.NDArray[np.int64],
    removed_customer_indices: npt.NDArray[np.int64],
    epsilon: float,
    route_change_limit: int,
    allow_new_routes: bool,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]: ...
def constraint_removal_v2(
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
    route_offsets: npt.NDArray[np.int64],
    route_indices: npt.NDArray[np.int64],
    path_offsets: npt.NDArray[np.int64],
    path_indices: npt.NDArray[np.int64],
    result_metrics: npt.NDArray[np.float64],
    requested_count: int,
    seed: int,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
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
def candidate_round_transaction_v2(
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
    resource_receipt: npt.NDArray[np.int64],
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
    str,
]: ...
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
def native_search_request_receipt_v2(
    node_kind: npt.NDArray[np.int64],
    demand: npt.NDArray[np.float64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    reachable: npt.NDArray[np.uint8],
    vehicle: npt.NDArray[np.float64],
    lexical_rank: npt.NDArray[np.int64],
    node_name_offsets: npt.NDArray[np.int64],
    node_name_bytes: npt.NDArray[np.uint8],
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
) -> tuple[npt.NDArray[np.int64], str]: ...
def native_search_initial_state_v2(
    node_kind: npt.NDArray[np.int64],
    demand: npt.NDArray[np.float64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    reachable: npt.NDArray[np.uint8],
    vehicle: npt.NDArray[np.float64],
    lexical_rank: npt.NDArray[np.int64],
    node_name_offsets: npt.NDArray[np.int64],
    node_name_bytes: npt.NDArray[np.uint8],
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
    tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    str,
    str,
]: ...
def full_native_alns_v2(
    node_kind: npt.NDArray[np.int64],
    demand: npt.NDArray[np.float64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    reachable: npt.NDArray[np.uint8],
    vehicle: npt.NDArray[np.float64],
    lexical_rank: npt.NDArray[np.int64],
    node_name_offsets: npt.NDArray[np.int64],
    node_name_bytes: npt.NDArray[np.uint8],
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
    object,
    object,
    object,
    object,
    tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        str,
    ],
    tuple[npt.NDArray[np.int64], str, str, str, str, object],
]: ...
def native_search_request_host_receipt_v2(
    socket_path: str,
    node_kind: npt.NDArray[np.int64],
    demand: npt.NDArray[np.float64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    reachable: npt.NDArray[np.uint8],
    vehicle: npt.NDArray[np.float64],
    lexical_rank: npt.NDArray[np.int64],
    node_name_offsets: npt.NDArray[np.int64],
    node_name_bytes: npt.NDArray[np.uint8],
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
) -> tuple[npt.NDArray[np.int64], str]: ...
def native_search_initial_state_host_v2(
    socket_path: str,
    node_kind: npt.NDArray[np.int64],
    demand: npt.NDArray[np.float64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    reachable: npt.NDArray[np.uint8],
    vehicle: npt.NDArray[np.float64],
    lexical_rank: npt.NDArray[np.int64],
    node_name_offsets: npt.NDArray[np.int64],
    node_name_bytes: npt.NDArray[np.uint8],
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
    tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.float64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int64],
    str,
    str,
]: ...
def full_native_alns_host_v2(
    socket_path: str,
    node_kind: npt.NDArray[np.int64],
    demand: npt.NDArray[np.float64],
    ready_time: npt.NDArray[np.float64],
    due_date: npt.NDArray[np.float64],
    service_time: npt.NDArray[np.float64],
    distance: npt.NDArray[np.float64],
    reachable: npt.NDArray[np.uint8],
    vehicle: npt.NDArray[np.float64],
    lexical_rank: npt.NDArray[np.int64],
    node_name_offsets: npt.NDArray[np.int64],
    node_name_bytes: npt.NDArray[np.uint8],
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
) -> tuple[object, ...]: ...

def _test_native_kernel_fault_v2(socket_path: str, fault: str) -> None: ...
def _test_full_native_initial_mirror_fault_v2(code: int) -> None: ...
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
