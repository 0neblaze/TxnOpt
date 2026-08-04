"""Explicit Stage 5.2 native execution protocols.

This module is the only public configuration seam that may combine the
Stage 5.2 native kernels and candidate transaction with historical Stage 3.4
Candidate Control.  Passing ``None`` to :func:`evrptw.alns.solve_alns` keeps
the historical guards and execution paths unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from evrptw.cache_incremental import CacheIncrementalConfig
from evrptw.candidate_control import (
    CandidateControlConfig,
    stable_candidate_payload_hash,
)
from evrptw.candidate_transaction import (
    SCREEN_REASON_BY_CODE,
    CandidateScreeningBatch,
    CandidateTransactionAudit,
    CustomerSequence,
    NativeCandidateTransactionConfig,
    NativeCandidateTransactionRuntime,
    decode_native_candidate_screening_payload,
)
from evrptw.charging import ChargingSubproblemResult, solve_exact_charging
from evrptw.cpu_batch import BackendMetrics, decode_exact_charging_batch_numeric
from evrptw.measurement import CheapScreeningConfig
from evrptw.models import Instance
from evrptw.native_kernels import NativeKernelConfig, NativeKernelRuntime
from evrptw.neighborhoods import VehicleOperatorConfig, screen_route_candidate
from evrptw.objective import SolutionObjective
from evrptw.stage04 import Stage04Config
from evrptw.validation import validate_routes

NATIVE_EXECUTION_SCHEMA_VERSION = "stage05.2-native-execution-v2"

_SCREEN_CHECK_BY_CODE = {
    1: "route_structure",
    2: "capacity_lower_bound",
    3: "forward_time_window",
    4: "backward_time_window",
    5: "time_window_slack",
    6: "shortest_distance_lower_bound",
    7: "single_segment_battery_reachability",
    8: "structural_energy_lower_bound",
}
_SCREEN_CHECK_STATUS_BY_CODE = {0: "fail", 1: "pass", 2: "recorded"}

NativeExecutionMode = Literal[
    "per_solve_runtime",
    "full_native_alns",
    "host_scheduler",
]
NativeWorkerProtocol = Literal[
    "candidate_round_soa_v2",
    "full_solve_soa_v2",
    "unix_shm_scheduler_v2",
]

_PROTOCOL_BY_MODE: dict[NativeExecutionMode, NativeWorkerProtocol] = {
    "per_solve_runtime": "candidate_round_soa_v2",
    "full_native_alns": "full_solve_soa_v2",
    "host_scheduler": "unix_shm_scheduler_v2",
}

NativeCandidateResolution = Literal[
    "screening_rejected",
    "cache_hit",
    "exact",
    "not_selected",
    "round_budget_exhausted",
    "duplicate",
]
_RESOLUTION_BY_CODE: dict[int, NativeCandidateResolution] = {
    0: "screening_rejected",
    1: "cache_hit",
    2: "exact",
    3: "not_selected",
    4: "round_budget_exhausted",
    5: "duplicate",
}

FULL_NATIVE_OPERATOR_NAMES = (
    "standard",
    "vehicle_count_aware_repair",
    "route_elimination",
    "route_merge",
    "relocate",
    "swap",
    "two_opt_star",
    "route_segment_destroy",
    "ejection_chain",
    "station_pressure",
    "time_window_conflict",
    "worst_energy_detour",
    "shaw_related",
    "vehicle_reduction_refinement",
    "destroy:random",
    "destroy:worst",
    "destroy:related",
    "repair:greedy",
    "repair:regret2",
    "repair:energy",
)

FULL_NATIVE_STAGE04_INTEGER_FIELDS = (
    "enabled",
    "segment_length",
    "min_calls_per_operator",
    "fixed_weights",
    "auto_temperature",
    "temperature_sample_size",
    "reheat_enabled",
    "reheat_stagnation_threshold",
    "max_reheats",
    "restart_enabled",
    "restart_stagnation_threshold",
    "max_restarts",
    "intensification_enabled",
    "intensification_iterations",
    "acceptance_rate_window",
)
FULL_NATIVE_STAGE04_FLOAT_FIELDS = (
    "weight_reaction",
    "weight_floor",
    "weight_smoothing",
    "fixed_weight_value",
    "reward_rejected",
    "reward_accepted_worse",
    "reward_accepted_equal",
    "reward_distance_improvement",
    "reward_vehicle_reduction",
    "reward_new_global_best",
    "reward_new_global_best_vehicle_reduction",
    "temperature_target_acceptance_rate",
    "temperature_fallback_fraction",
    "reheat_factor",
    "intensification_removal_fraction",
)
FULL_NATIVE_OPERATOR_INTEGER_FIELDS = (
    "max_route_elimination_attempts",
    "route_elimination_exact_evaluation_budget",
    "route_merge_exact_evaluation_budget",
    "vehicle_repair_exact_evaluation_budget",
    "vehicle_reduction_refinement_exact_evaluation_budget",
    "relocate_exact_evaluation_budget",
    "swap_exact_evaluation_budget",
    "two_opt_star_exact_evaluation_budget",
    "route_segment_exact_evaluation_budget",
    "ejection_chain_exact_evaluation_budget",
    "quality_probe_exact_evaluation_budget",
    "quality_route_segment_probe_exact_evaluation_budget",
    "route_segment_min_length",
    "route_segment_max_length",
    "ejection_chain_max_depth",
    "ejection_chain_beam_width",
    "constraint_probe_exact_evaluation_budget",
    "station_pressure_exact_evaluation_budget",
    "time_window_conflict_exact_evaluation_budget",
    "worst_energy_detour_exact_evaluation_budget",
    "shaw_related_exact_evaluation_budget",
    "medium_stagnation_threshold",
    "large_stagnation_threshold",
    "exploration_period",
)
FULL_NATIVE_OPERATOR_FLOAT_FIELDS = (
    "constraint_lane_time_budget_seconds",
    "small_removal_min_fraction",
    "small_removal_max_fraction",
    "medium_removal_min_fraction",
    "medium_removal_max_fraction",
    "large_removal_min_fraction",
    "large_removal_max_fraction",
)


@dataclass(frozen=True, slots=True)
class NativeCandidateRoundRequest:
    """One complete candidate round expressed as immutable protocol inputs."""

    candidates: tuple[CustomerSequence, ...]
    cache_hit_flags: tuple[bool, ...]
    proposal_top_k: int
    exact_budget: int
    deadline: float
    batch_size: int
    lane: str
    operator: str
    iteration: int | None
    compute_threads: int = 1
    full_screening: bool = True
    incremental: npt.NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("a native candidate round must not be empty")
        if len(self.cache_hit_flags) != len(self.candidates):
            raise ValueError("cache-hit flags must align with candidate rows")
        if self.proposal_top_k <= 0:
            raise ValueError("native candidate-round top-k must be positive")
        if self.exact_budget < 0:
            raise ValueError("native candidate-round exact budget must be non-negative")
        if not math.isfinite(self.deadline):
            raise ValueError("native candidate-round deadline must be finite")
        if self.batch_size <= 0:
            raise ValueError("native candidate-round batch size must be positive")
        if self.compute_threads <= 0:
            raise ValueError("native candidate-round compute threads must be positive")


@dataclass(frozen=True, slots=True)
class NativeCandidateResourceReceipt:
    """Out-of-band accounting written before result-payload validation."""

    phase: int
    started_calls: int
    completed_calls: int
    interrupted_calls: int
    fallback_count: int
    fail_closed: bool = False

    @property
    def work_started(self) -> bool:
        return self.started_calls > 0


class NativeCandidateRoundFailure(RuntimeError):
    """A failed native round whose resource receipt remains authoritative."""

    def __init__(
        self,
        message: str,
        *,
        resource_receipt: NativeCandidateResourceReceipt,
    ) -> None:
        super().__init__(message)
        self.resource_receipt = resource_receipt


@dataclass(frozen=True, slots=True)
class NativeCandidateRoundResult:
    """Strictly decoded output of one native candidate-round invocation."""

    screening: CandidateScreeningBatch
    resolutions: tuple[NativeCandidateResolution, ...]
    resolution_sources: tuple[int, ...]
    cache_journal: npt.NDArray[np.int64]
    exact_candidate_ids: tuple[int, ...]
    exact_results: tuple[ChargingSubproblemResult, ...]
    backend_metrics: BackendMetrics
    completion_order: tuple[int, ...]
    counters: Mapping[str, int]
    timings: Mapping[str, float]
    transaction_sha256: str
    audit: CandidateTransactionAudit
    resource_receipt: NativeCandidateResourceReceipt


@dataclass(frozen=True, slots=True)
class NativeInitialStateReceipt:
    """Hashed ownership receipt for the full-native initial-state operation."""

    host_owned: bool
    operation_count: int
    request_sha256: str
    state_sha256: str
    transaction_sha256: str


@dataclass(frozen=True, slots=True)
class FullNativeALNSResult:
    """Strict output of one full-native solve dispatch."""

    customer_sequences: tuple[CustomerSequence, ...]
    exact_results: tuple[ChargingSubproblemResult, ...]
    backend_metrics: BackendMetrics
    counters: Mapping[str, int]
    timings: Mapping[str, float]
    trajectory: tuple[Mapping[str, object], ...]
    semantic_stream: NativeGlobalSemanticStream | NativeThreeLaneSemanticStream
    transaction_sha256: str
    candidate_work_hash: str
    route_result_hash: str
    exact_journal_events: tuple[Mapping[str, object], ...]
    exact_journal_sha256: str
    candidate_work: tuple[Mapping[str, object], ...]
    route_results: tuple[Mapping[str, object], ...]
    control_journal_events: tuple[Mapping[str, object], ...]
    control_journal_sha256: str
    control_journal_statistics: Mapping[str, int]
    route_cache_statistics: Mapping[str, int]
    screening_statistics: Mapping[str, object]
    causal_journal: NativeCausalJournal
    initial_state_receipt: NativeInitialStateReceipt


@dataclass(frozen=True, slots=True)
class NativeCausalJournal:
    """Strict typed receipt for native cross-domain event order."""

    event_ids: npt.NDArray[np.int64]
    stream_codes: npt.NDArray[np.int64]
    event_codes: npt.NDArray[np.int64]
    lane_ids: npt.NDArray[np.int64]
    operator_ids: npt.NDArray[np.int64]
    iterations: npt.NDArray[np.int64]
    transaction_ids: npt.NDArray[np.int64]
    subject_ids: npt.NDArray[np.int64]
    status_codes: npt.NDArray[np.int64]
    flags: npt.NDArray[np.int64]
    stream_counts: npt.NDArray[np.int64]
    transaction_sha256: str

    @property
    def complete_search_semantics(self) -> bool:
        """Return true only when every search-decision domain is populated."""

        required_codes = (0, 1, 2, 3, 4, 5, 7)
        if not all(int(self.stream_counts[code]) > 0 for code in required_codes):
            return False
        terminal_rows = np.flatnonzero(self.stream_codes == 7)
        if (
            len(terminal_rows) != 1
            or int(terminal_rows[0]) != len(self.event_ids) - 1
            or int(self.event_codes[terminal_rows[0]]) != 5
        ):
            return False
        terminal_reason = int(self.status_codes[terminal_rows[0]])
        expected_deadline_rows = 1 if terminal_reason in {1, 2, 3} else 0
        return int(self.stream_counts[6]) == expected_deadline_rows


@dataclass(frozen=True, slots=True)
class NativeConstraintSemanticStream:
    """Independently validated typed events from the native constraint loop."""

    event_integer: npt.NDArray[np.int64]
    event_objective_integer: npt.NDArray[np.int64]
    event_objective: npt.NDArray[np.float64]
    plan_offsets: npt.NDArray[np.int64]
    route_offsets: npt.NDArray[np.int64]
    route_indices: npt.NDArray[np.int64]
    candidate_hashes: npt.NDArray[np.uint8]
    stage04_status: npt.NDArray[np.int64]
    stage04_weights: npt.NDArray[np.float64]
    stage04_calls: npt.NDArray[np.int64]
    stage04_rewards: npt.NDArray[np.float64]
    termination: npt.NDArray[np.int64]
    transaction_sha256: str


@dataclass(frozen=True, slots=True)
class NativeGlobalSemanticStream:
    """Validated global-iteration events projected into the Python event schema."""

    neighborhood_events: tuple[Mapping[str, object], ...]
    event_integer: npt.NDArray[np.int64]
    stage04_calls: npt.NDArray[np.int64]
    termination: npt.NDArray[np.int64]
    transaction_sha256: str
    initial_temperature: float


@dataclass(frozen=True, slots=True)
class NativeThreeLaneSemanticStream:
    """Independently replayed semantic projection of one full three-lane iteration."""

    neighborhood_events: tuple[Mapping[str, object], ...]
    operator_weights: npt.NDArray[np.float64]
    operator_rewards: npt.NDArray[np.float64]
    operator_calls: npt.NDArray[np.int64]
    operator_totals: npt.NDArray[np.int64]
    operator_activity: npt.NDArray[np.int64]
    termination: npt.NDArray[np.int64]
    transaction_sha256: str
    stage04_events: tuple[Mapping[str, object], ...] = ()
    stage04_control: npt.NDArray[np.int64] | None = None
    initial_temperature: float | None = None
    operator_replay_events: tuple[Mapping[str, object], ...] = ()


def _append_u64(evidence: bytearray, value: int) -> None:
    evidence.extend((value & ((1 << 64) - 1)).to_bytes(8, "little"))


def _append_typed_values(
    evidence: bytearray,
    values: npt.NDArray[np.generic],
) -> None:
    flattened = values.reshape(-1)
    _append_u64(evidence, len(flattened))
    if values.dtype == np.dtype(np.uint8):
        evidence.extend(flattened.tobytes(order="C"))
        return
    if values.dtype == np.dtype(np.int64):
        for value in flattened:
            _append_u64(evidence, int(value))
        return
    if values.dtype == np.dtype(np.float64):
        for value in flattened:
            converted = float(value)
            bits = (
                0x7FF8000000000000
                if math.isnan(converted)
                else struct.unpack("<Q", struct.pack("<d", converted))[0]
            )
            _append_u64(evidence, bits)
        return
    raise RuntimeError("native semantic evidence has an unsupported dtype")


def _append_typed_array(
    evidence: bytearray,
    values: npt.NDArray[np.generic],
) -> None:
    _append_u64(evidence, values.ndim)
    for dimension in values.shape:
        _append_u64(evidence, dimension)
    _append_typed_values(evidence, values)


def _readonly_copy[T: np.generic](values: npt.NDArray[T]) -> npt.NDArray[T]:
    copied = np.ascontiguousarray(values.copy())
    copied.flags.writeable = False
    return copied


def _semantic_event_aggregate(
    event: Mapping[str, object],
    flag: str,
) -> int:
    """Apply NeighborhoodEvent aggregate-count semantics to one boolean flag."""

    return cast(int, event.get("aggregate_count", 1)) if bool(event[flag]) else 0


def _constraint_score_vectors(
    removal: tuple[object, ...],
    removed_indices: npt.NDArray[np.int64],
    name: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Validate the complete per-customer ranking journal for one removal."""

    score_nodes = _require_vector(removal[3], f"{name} score nodes")
    scores = cast(
        npt.NDArray[np.float64],
        _require_array(
            removal[4],
            dtype=np.dtype(np.float64),
            shape=(len(score_nodes),),
            name=f"{name} scores",
        ),
    )
    score_routes = _require_vector(removal[5], f"{name} score routes")
    removed_count = len(removed_indices)
    if (
        len(score_routes) != len(score_nodes)
        or len(score_nodes) < removed_count
        or not np.array_equal(score_nodes[:removed_count], removed_indices)
        or np.any(score_nodes < 0)
        or np.any(score_routes < 0)
        or np.any(~np.isfinite(scores))
    ):
        raise RuntimeError(f"native constraint {name} ranking journal is invalid")
    return scores, score_routes


def decode_native_constraint_semantic_stream(
    payload: object,
) -> NativeConstraintSemanticStream:
    """Validate native event SoA without trusting producer-side hashes."""

    if not isinstance(payload, tuple) or len(payload) != 13:
        raise RuntimeError("native constraint semantic stream has an invalid tuple")
    event_integer = payload[0]
    if (
        not isinstance(event_integer, np.ndarray)
        or event_integer.dtype != np.dtype(np.int64)
        or event_integer.ndim != 2
        or event_integer.shape[1] != 16
        or not event_integer.flags.c_contiguous
    ):
        raise RuntimeError("native constraint semantic event integers have an invalid schema")
    event_count = event_integer.shape[0]
    expected_arrays: tuple[tuple[object, np.dtype[np.generic], tuple[int, ...], str], ...] = (
        (payload[1], np.dtype(np.int64), (event_count, 6), "objective integers"),
        (payload[2], np.dtype(np.float64), (event_count, 6), "objective floats"),
        (payload[6], np.dtype(np.uint8), (event_count, 32), "candidate hashes"),
        (payload[7], np.dtype(np.int64), (event_count, 4), "Stage 4 statuses"),
        (payload[8], np.dtype(np.float64), (event_count, 4, 2), "Stage 4 weights"),
        (payload[9], np.dtype(np.int64), (event_count, 4), "Stage 4 calls"),
        (payload[10], np.dtype(np.float64), (event_count, 4), "Stage 4 rewards"),
    )
    validated: list[npt.NDArray[np.generic]] = []
    for value, dtype, shape, name in expected_arrays:
        validated.append(_require_array(value, dtype=dtype, shape=shape, name=name))
    event_objective_integer = cast(npt.NDArray[np.int64], validated[0])
    event_objective = cast(npt.NDArray[np.float64], validated[1])
    candidate_hashes = cast(npt.NDArray[np.uint8], validated[2])
    stage04_status = cast(npt.NDArray[np.int64], validated[3])
    stage04_weights = cast(npt.NDArray[np.float64], validated[4])
    stage04_calls = cast(npt.NDArray[np.int64], validated[5])
    stage04_rewards = cast(npt.NDArray[np.float64], validated[6])
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(13,),
            name="termination",
        ),
    )
    plan_offsets = _require_vector(payload[3], "constraint semantic plan offsets")
    route_offsets = _require_vector(payload[4], "constraint semantic route offsets")
    route_indices = _require_vector(payload[5], "constraint semantic route indices")
    if (
        len(plan_offsets) != event_count + 1
        or len(route_offsets) == 0
        or int(plan_offsets[0]) != 0
        or int(route_offsets[0]) != 0
        or int(plan_offsets[-1]) != len(route_offsets) - 1
        or int(route_offsets[-1]) != len(route_indices)
        or np.any(plan_offsets[:-1] > plan_offsets[1:])
        or np.any(route_offsets[:-1] > route_offsets[1:])
    ):
        raise RuntimeError("native constraint semantic offsets are invalid")
    if (
        np.any(event_integer[:, 0] != 0)
        or np.any(event_integer[:, 1] != 2)
        or np.any(event_integer[:, 3] < 9)
        or np.any(event_integer[:, 3] > 10)
        or np.any(event_integer[1:, 2] != event_integer[:-1, 2] + 1)
    ):
        raise RuntimeError("native constraint semantic event identity is invalid")
    if event_count:
        candidate_vehicle_present = event_objective_integer[:, 0] >= 0
        candidate_charging_present = event_objective_integer[:, 1] >= 0
        candidate_present = candidate_vehicle_present & candidate_charging_present
        candidate_route_counts = plan_offsets[1:] - plan_offsets[:-1]
        if (
            np.any(event_objective_integer[:, 2:] < 0)
            or np.any(candidate_vehicle_present != candidate_charging_present)
            or np.any(event_objective_integer[~candidate_present, :2] != -1)
            or np.any(
                event_objective_integer[candidate_present, 0]
                != candidate_route_counts[candidate_present]
            )
            or np.any(event_integer[:, 14] != event_objective_integer[:, 2])
            or np.any(event_integer[:, 15] != event_objective_integer[:, 4])
            or np.any(event_integer[:, 6].astype(bool) != candidate_present)
            or np.any(np.isfinite(event_objective[:, :2]) != candidate_present[:, None])
            or np.any(event_objective[candidate_present, :2] < 0.0)
            or np.any(~np.isfinite(event_objective[:, 2:]))
            or np.any(event_objective[:, 2:] < 0.0)
        ):
            raise RuntimeError("native constraint semantic objectives are invalid")
    if (
        np.any((stage04_status < -1) | (stage04_status > 1))
        or np.any(stage04_calls < 0)
        or np.any(~np.isfinite(stage04_weights))
        or np.any(stage04_weights < 0.0)
        or np.any(~np.isfinite(stage04_rewards))
        or np.any(stage04_rewards < 0.0)
    ):
        raise RuntimeError("native constraint semantic Stage 4 state is invalid")
    terminal_reason = int(termination[0])
    requested_iterations = int(termination[2])
    exact_budget_limit = int(termination[3])
    entry_exact = termination[4:7]
    final_exact = termination[7:10]
    terminal_tail = termination[10:13]
    event_exact_deltas = event_integer[:, 10:13].sum(axis=0, dtype=np.int64)
    budget_configured = exact_budget_limit >= 0
    budget_reached = budget_configured and int(final_exact[0]) == exact_budget_limit
    deadline_has_terminal_work = bool(np.any(terminal_tail != 0))
    if (
        terminal_reason not in {0, 1, 2}
        or int(termination[1]) != event_count
        or requested_iterations <= 0
        or exact_budget_limit < -1
        or np.any(entry_exact < 0)
        or np.any(final_exact < entry_exact)
        or (
            budget_configured
            and (
                int(entry_exact[0]) > exact_budget_limit
                or int(final_exact[0]) > exact_budget_limit
            )
        )
        or np.any(terminal_tail < 0)
        or int(entry_exact[1]) + int(entry_exact[2]) > int(entry_exact[0])
        or int(final_exact[1]) + int(final_exact[2]) > int(final_exact[0])
        or np.any(final_exact - entry_exact - event_exact_deltas != terminal_tail)
        or (
            terminal_reason == 0
            and (event_count != requested_iterations or budget_reached)
        )
        or (
            terminal_reason == 1
            and not budget_reached
        )
        or (
            terminal_reason == 2
            and (
                event_count >= requested_iterations
                or (budget_reached and not deadline_has_terminal_work)
            )
        )
        or (terminal_reason != 2 and np.any(terminal_tail != 0))
    ):
        raise RuntimeError("native constraint semantic termination is invalid")

    for event in range(event_count):
        identity = bytearray(b"stage05.2-native-candidate-route-identity-v2")
        first_route = int(plan_offsets[event])
        last_route = int(plan_offsets[event + 1])
        for route in range(first_route, last_route):
            _append_typed_values(
                identity,
                route_indices[int(route_offsets[route]) : int(route_offsets[route + 1])],
            )
        observed = bytes(candidate_hashes[event])
        if hashlib.sha256(identity).digest() != observed:
            raise RuntimeError("native candidate identity SHA-256 mismatch")

    transaction_sha256 = payload[12]
    if not isinstance(transaction_sha256, str) or not _is_sha256(transaction_sha256):
        raise RuntimeError("native constraint semantic stream SHA-256 is invalid")
    evidence = bytearray(b"stage05.2-native-constraint-semantic-stream-v2")
    for values in (
        event_integer,
        event_objective_integer,
        event_objective,
        plan_offsets,
        route_offsets,
        route_indices,
        candidate_hashes,
        stage04_status,
        stage04_weights,
        stage04_calls,
        stage04_rewards,
        termination,
    ):
        _append_typed_array(evidence, values)
    if hashlib.sha256(evidence).hexdigest() != transaction_sha256:
        raise RuntimeError("native constraint semantic stream SHA-256 mismatch")
    return NativeConstraintSemanticStream(
        event_integer=_readonly_copy(event_integer),
        event_objective_integer=_readonly_copy(event_objective_integer),
        event_objective=_readonly_copy(event_objective),
        plan_offsets=_readonly_copy(plan_offsets),
        route_offsets=_readonly_copy(route_offsets),
        route_indices=_readonly_copy(route_indices),
        candidate_hashes=_readonly_copy(candidate_hashes),
        stage04_status=_readonly_copy(stage04_status),
        stage04_weights=_readonly_copy(stage04_weights),
        stage04_calls=_readonly_copy(stage04_calls),
        stage04_rewards=_readonly_copy(stage04_rewards),
        termination=_readonly_copy(termination),
        transaction_sha256=transaction_sha256,
    )


def decode_native_global_semantic_stream(
    instance: Instance,
    payload: object,
    *,
    node_names: tuple[str, ...],
    initial_customer_sequences: tuple[CustomerSequence, ...],
    stage04_config: Stage04Config,
    expected_start_iteration: int,
    expected_iteration_count: int,
) -> NativeGlobalSemanticStream:
    """Independently validate and project one native global semantic stream."""

    if expected_start_iteration < 0 or expected_iteration_count <= 0:
        raise ValueError("native global semantic expected iteration range is invalid")
    if not stage04_config.enabled:
        raise ValueError("native global semantic replay requires enabled Stage 4")
    if tuple(node.name for node in instance.nodes) != node_names:
        raise RuntimeError("native global semantic node identity is invalid")
    if not isinstance(payload, tuple) or len(payload) != 15:
        raise RuntimeError("native global semantic stream has an invalid tuple")
    events = payload[0]
    if (
        not isinstance(events, np.ndarray)
        or events.dtype != np.dtype(np.int64)
        or events.ndim != 2
        or events.shape[1] != 26
        or not events.flags.c_contiguous
    ):
        raise RuntimeError("native global semantic events have an invalid schema")
    event_count = events.shape[0]
    ranking = cast(
        npt.NDArray[np.float64],
        _require_array(
            payload[1],
            dtype=np.dtype(np.float64),
            shape=(event_count,),
            name="global ranking",
        ),
    )
    removed_offsets = _require_vector(payload[2], "global removed offsets")
    removed_indices = _require_vector(payload[3], "global removed indices")
    plan_offsets = _require_vector(payload[4], "global plan offsets")
    route_offsets = _require_vector(payload[5], "global route offsets")
    route_indices = _require_vector(payload[6], "global route indices")
    objective_integer = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[7],
            dtype=np.dtype(np.int64),
            shape=(event_count, 2),
            name="global objective integers",
        ),
    )
    objective_float = cast(
        npt.NDArray[np.float64],
        _require_array(
            payload[8],
            dtype=np.dtype(np.float64),
            shape=(event_count, 2),
            name="global objective floats",
        ),
    )
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[13],
            dtype=np.dtype(np.int64),
            shape=(11,),
            name="global termination",
        ),
    )
    terminal_reason = int(termination[0])
    completed_iterations = int(termination[2])
    exact_budget_limit = int(termination[4])
    entry_exact = termination[5:8]
    final_exact = termination[8:11]
    exact_delta = final_exact - entry_exact
    stage04_status = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[9],
            dtype=np.dtype(np.int64),
            shape=(completed_iterations, 4),
            name="global Stage 4 status",
        ),
    )
    stage04_weights = cast(
        npt.NDArray[np.float64],
        _require_array(
            payload[10],
            dtype=np.dtype(np.float64),
            shape=(completed_iterations, 4, 2),
            name="global Stage 4 weights",
        ),
    )
    stage04_calls = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(completed_iterations, 4),
            name="global Stage 4 calls",
        ),
    )
    stage04_rewards = cast(
        npt.NDArray[np.float64],
        _require_array(
            payload[12],
            dtype=np.dtype(np.float64),
            shape=(completed_iterations, 4),
            name="global Stage 4 rewards",
        ),
    )
    if (
        len(removed_offsets) != event_count + 1
        or len(plan_offsets) != event_count + 1
        or len(route_offsets) == 0
        or int(removed_offsets[0]) != 0
        or int(removed_offsets[-1]) != len(removed_indices)
        or int(plan_offsets[0]) != 0
        or int(plan_offsets[-1]) != len(route_offsets) - 1
        or int(route_offsets[0]) != 0
        or int(route_offsets[-1]) != len(route_indices)
        or np.any(removed_offsets[:-1] > removed_offsets[1:])
        or np.any(plan_offsets[:-1] > plan_offsets[1:])
        or np.any(route_offsets[:-1] > route_offsets[1:])
    ):
        raise RuntimeError("native global semantic offsets are invalid")
    if (
        np.any(events[:, 0] != 0)
        or np.any(events[:, 1] < 0)
        or np.any(events[:, 1] > 2)
        or np.any(events[:, 3] < 0)
        or np.any(events[:, 3] >= len(FULL_NATIVE_OPERATOR_NAMES))
        or np.any(events[:, 4] < 0)
        or np.any(events[:, 4] > 2)
        or np.any(events[:, 5] < 0)
        or np.any(events[:, 5] > 4)
        or np.any(events[:, 2] < expected_start_iteration)
        or np.any(events[:, 2] >= expected_start_iteration + completed_iterations)
        or np.any(~np.isin(events[:, [6, 7, 8, 9, 10, 23]], (0, 1)))
        or np.any(events[:, [11, 12, 13, 14, 18, 19, 21, 22]] < 0)
        or np.any(events[:, 6] > events[:, 10])
        or np.any(events[:, 16:18] < -1)
        or np.any(events[:, 20] < -2)
        or np.any(events[:, 24] < -1)
        or np.any(events[:, 24] > 2)
        or np.any(events[:, 25] < -1)
        or np.any(events[:, 25] > 6)
        or np.any(~np.isfinite(ranking))
    ):
        raise RuntimeError("native global semantic event values are invalid")
    all_node_indices = np.concatenate((removed_indices, route_indices))
    if np.any(all_node_indices < 0) or np.any(all_node_indices >= len(node_names)):
        raise RuntimeError("native global semantic event contains an unknown node")
    objective_present = objective_integer[:, 0] >= 0
    if (
        np.any((objective_integer[:, 0] >= 0) != (objective_integer[:, 1] >= 0))
        or np.any(objective_integer[~objective_present] != -1)
        or np.any(objective_integer[objective_present] < 0)
        or np.any(np.isfinite(objective_float) != objective_present[:, None])
        or np.any(objective_float[objective_present] < 0.0)
        or np.any((stage04_status < -1) | (stage04_status > 1))
        or np.any(stage04_calls < 0)
        or np.any(~np.isfinite(stage04_weights))
        or np.any(stage04_weights < 0.0)
        or np.any(~np.isfinite(stage04_rewards))
        or np.any(stage04_rewards < 0.0)
        or terminal_reason not in {0, 1, 2}
        or (terminal_reason == 0 and completed_iterations != expected_iteration_count)
        or (
            terminal_reason == 1
            and not 0 <= completed_iterations <= expected_iteration_count
        )
        or (terminal_reason == 2 and completed_iterations != 0)
        or int(termination[1]) != expected_start_iteration
        or int(termination[3]) != expected_start_iteration + completed_iterations
        or exact_budget_limit < -1
        or np.any(entry_exact < 0)
        or np.any(final_exact < entry_exact)
        or int(entry_exact[1]) + int(entry_exact[2]) != int(entry_exact[0])
        or int(final_exact[1]) + int(final_exact[2]) != int(final_exact[0])
        or entry_exact.tolist() != [len(initial_customer_sequences)] * 2 + [0]
        or (
            exact_budget_limit >= 0
            and (
                int(entry_exact[0]) > exact_budget_limit
                or int(final_exact[0]) > exact_budget_limit
            )
        )
        or (
            terminal_reason == 0
            and exact_budget_limit >= 0
            and int(final_exact[0]) == exact_budget_limit
        )
        or (
            terminal_reason == 1
            and (
                exact_budget_limit < 0
                or int(final_exact[0]) != exact_budget_limit
            )
        )
        or (
            terminal_reason != 2
            and (
                int(exact_delta[0]) != int(events[:, 11].sum(dtype=np.int64))
                or exact_delta.tolist() != [int(exact_delta[0]), int(exact_delta[0]), 0]
            )
        )
    ):
        raise RuntimeError("native global semantic state is invalid")
    initial_names = tuple(name for route in initial_customer_sequences for name in route)
    expected_customer_names = {customer.name for customer in instance.customers}
    if (
        len(initial_customer_sequences) != 1
        or len(initial_names) != len(expected_customer_names)
        or set(initial_names) != expected_customer_names
    ):
        raise RuntimeError("native global initial incumbent identity is invalid")
    initial_exact = solve_exact_charging(instance, initial_customer_sequences[0])
    if not initial_exact.feasible:
        raise RuntimeError("native global initial incumbent is infeasible")
    initial_temperature = max(
        1.0,
        initial_exact.distance * stage04_config.temperature_fallback_fraction,
    )
    transaction_sha256 = payload[14]
    if not isinstance(transaction_sha256, str) or not _is_sha256(transaction_sha256):
        raise RuntimeError("native global semantic SHA-256 is invalid")
    evidence = bytearray(b"stage05.2-native-global-semantic-stream-v2")
    for values in (
        events,
        ranking,
        removed_offsets,
        removed_indices,
        plan_offsets,
        route_offsets,
        route_indices,
        objective_integer,
        objective_float,
        stage04_status,
        stage04_weights,
        stage04_calls,
        stage04_rewards,
        termination,
    ):
        _append_typed_array(evidence, values)
    if hashlib.sha256(evidence).hexdigest() != transaction_sha256:
        raise RuntimeError("native global semantic stream SHA-256 mismatch")
    if completed_iterations == 0:
        if (
            terminal_reason not in {1, 2}
            or event_count != 0
            or removed_offsets.tolist() != [0]
            or len(removed_indices) != 0
            or plan_offsets.tolist() != [0]
            or route_offsets.tolist() != [0]
            or len(route_indices) != 0
            or objective_integer.shape != (0, 2)
            or objective_float.shape != (0, 2)
            or stage04_status.shape != (0, 4)
            or stage04_weights.shape != (0, 4, 2)
            or stage04_calls.shape != (0, 4)
            or stage04_rewards.shape != (0, 4)
        ):
            raise RuntimeError("native global empty terminal is invalid")
        return NativeGlobalSemanticStream(
            neighborhood_events=(),
            event_integer=_readonly_copy(events),
            stage04_calls=_readonly_copy(stage04_calls),
            termination=_readonly_copy(termination),
            transaction_sha256=transaction_sha256,
            initial_temperature=initial_temperature,
        )
    # The native one-route controller returns one canonical event window per
    # completed round. Keep windows keyed by their encoded iteration instead
    # of assuming all rows belong to iteration zero.
    iteration_windows: list[npt.NDArray[np.int64]] = []
    observed_iterations = events[:, 2]
    if np.any(observed_iterations[:-1] > observed_iterations[1:]):
        raise RuntimeError("native global canonical event sequence is invalid")
    for iteration in range(
        expected_start_iteration,
        expected_start_iteration + completed_iterations,
    ):
        rows = np.flatnonzero(observed_iterations == iteration).astype(
            np.int64,
            copy=False,
        )
        if rows.size == 0:
            raise RuntimeError("native global canonical event sequence is invalid")
        iteration_windows.append(rows)
    if sum(int(rows.size) for rows in iteration_windows) != event_count:
        raise RuntimeError("native global canonical event sequence is invalid")

    status_by_code = {0: "not_applicable", 1: "candidate_proposed", 2: "failed"}
    reason_by_code = {
        0: "only_one_route",
        1: "constraint_ranked_removal",
        2: "constraint_removal_repaired",
        3: "constraint_repair_infeasible",
        4: "no_removable_customer",
    }

    def project_events() -> tuple[dict[str, object], ...]:
        projected: list[dict[str, object]] = []
        for row_index in range(event_count):
            row = events[row_index]
            first_removed = int(removed_offsets[row_index])
            last_removed = int(removed_offsets[row_index + 1])
            removed_customers = tuple(
                node_names[int(node)]
                for node in removed_indices[first_removed:last_removed]
            )
            first_route = int(plan_offsets[row_index])
            last_route = int(plan_offsets[row_index + 1])
            candidate_routes = tuple(
                tuple(
                    node_names[int(node)]
                    for node in route_indices[
                        int(route_offsets[route]) : int(route_offsets[route + 1])
                    ]
                )
                for route in range(first_route, last_route)
            )
            operator = FULL_NATIVE_OPERATOR_NAMES[int(row[3])]
            constraint_event = int(row[15]) == 2
            objective_key: tuple[int, float, float, int] | tuple[()] = ()
            if objective_present[row_index]:
                objective_key = SolutionObjective(
                    vehicle_count=int(objective_integer[row_index, 0]),
                    total_distance=float(objective_float[row_index, 0]),
                    total_charging_time=float(objective_float[row_index, 1]),
                    charging_count=int(objective_integer[row_index, 1]),
                ).key
            projected.append(
                {
                    "operator": operator,
                    "status": status_by_code[int(row[4])],
                    "reason": reason_by_code[int(row[5])],
                    "route_indices": (() if int(row[16]) < 0 else (int(row[16]),)),
                    "affected_route_indices": (
                        () if int(row[17]) < 0 else (int(row[17]),)
                    ),
                    "removed_customers": removed_customers,
                    "candidate_customer_sequence": (),
                    "candidate_route_sequences": candidate_routes,
                    "candidate_vehicle_delta": (
                        None if int(row[20]) == -2 else int(row[20])
                    ),
                    "candidate_feasible": bool(row[10]),
                    "prefilter_passed": bool(row[9]),
                    "new_routes_created": int(row[21]),
                    "exact_route_evaluations": int(row[11]),
                    "selection_rank": int(row[12]),
                    "chain_depth": int(row[19]),
                    "segment_length": int(row[18]),
                    "track": "constraint_lane" if constraint_event else "legacy",
                    "constraint_category": operator if constraint_event else "",
                    "removal_tier": "small" if int(row[24]) == 0 else "",
                    "removal_size_requested": int(row[13]),
                    "removal_size_actual": int(row[14]),
                    "stagnation_iterations": int(row[22]),
                    "removal_trigger": (
                        "stagnation_baseline"
                        if int(row[25]) == 0
                        else "no_removable_customer"
                        if int(row[25]) == 1
                        else ""
                    ),
                    "reset_observed": bool(row[23]),
                    "ranking_score": float(ranking[row_index]),
                    "iteration": int(row[2]),
                    "accepted": bool(row[6]),
                    "vehicle_reduction": bool(row[7]),
                    "distance_improvement": bool(row[8]),
                    "candidate_objective_key": objective_key,
                }
            )
        return tuple(projected)

    if len(expected_customer_names) == 1:
        initial_exact = solve_exact_charging(instance, initial_customer_sequences[0])
        if not initial_exact.feasible:
            raise RuntimeError("native global no-removable replay is invalid")
        if any(rows.size != 3 for rows in iteration_windows):
            raise RuntimeError("native global no-removable replay is invalid")
        expected_stage04_calls = np.zeros(
            (completed_iterations, 4), dtype=np.int64
        )
        running_stage04_calls = np.zeros(4, dtype=np.int64)
        for window_index, rows in enumerate(iteration_windows):
            constraint_operator = int(events[int(rows[1]), 3]) - 9
            if constraint_operator not in range(4):
                raise RuntimeError("native global no-removable operator is invalid")
            running_stage04_calls[constraint_operator] += 1
            expected_stage04_calls[window_index] = running_stage04_calls
        if (
            removed_offsets.tolist() != [0] * (event_count + 1)
            or len(removed_indices) != 0
            or plan_offsets.tolist() != [0] * (event_count + 1)
            or route_offsets.tolist() != [0]
            or len(route_indices) != 0
            or np.any(objective_present)
            or np.any(ranking != 0.0)
            or not np.array_equal(stage04_calls, expected_stage04_calls)
            or np.any(stage04_status != -1)
            or not np.array_equal(
                stage04_weights,
                np.ones((completed_iterations, 4, 2)),
            )
            or not np.array_equal(
                stage04_rewards,
                np.zeros((completed_iterations, 4)),
            )
        ):
            raise RuntimeError("native global no-removable replay is invalid")
        for rows in iteration_windows:
            window = events[rows]
            if (
                int(window[0, 1]) != 1
                or int(window[0, 3]) != 4
                or np.any(window[0, [4, 5, 6, 7, 8, 9, 10]] != 0)
                or np.any(window[0, [15, 16, 17, 20, 24, 25]] != [0, -1, -1, -2, -1, -1])
                or int(window[1, 1]) != 2
                or not 9 <= int(window[1, 3]) <= 12
                or int(window[1, 4]) != 0
                or int(window[1, 5]) != 4
                or np.any(window[1, [6, 7, 8, 9, 10]] != 0)
                or np.any(window[1, [15, 16, 17, 20, 24, 25]] != [2, -1, -1, -2, 0, 1])
                or int(window[2, 1]) != 0
                or int(window[2, 3]) != 2
                or np.any(window[2, [4, 5, 6, 7, 8, 9, 10]] != 0)
                or np.any(window[2, [15, 16, 17, 20, 24, 25]] != [0, -1, -1, -2, -1, -1])
            ):
                raise RuntimeError("native global no-removable replay is invalid")
        return NativeGlobalSemanticStream(
            neighborhood_events=project_events(),
            event_integer=_readonly_copy(events),
            stage04_calls=_readonly_copy(stage04_calls),
            termination=_readonly_copy(termination),
            transaction_sha256=transaction_sha256,
            initial_temperature=initial_temperature,
        )

    if completed_iterations > 1:
        if any(rows.size not in {3, 4} for rows in iteration_windows):
            raise RuntimeError("native global canonical event sequence is invalid")
        if terminal_reason == 0 and any(rows.size != 4 for rows in iteration_windows):
            raise RuntimeError("native global canonical event sequence is invalid")
        if terminal_reason == 1 and any(
            rows.size != 4 for rows in iteration_windows[:-1]
        ):
            raise RuntimeError("native global canonical event sequence is invalid")

        index_by_name = {name: index for index, name in enumerate(node_names)}
        expected_customer_nodes = {
            index_by_name[customer.name] for customer in instance.customers
        }

        def event_routes(row_index: int) -> tuple[CustomerSequence, ...]:
            first_plan = int(plan_offsets[row_index])
            last_plan = int(plan_offsets[row_index + 1])
            if last_plan - first_plan != 1:
                return ()
            first_route = int(route_offsets[first_plan])
            last_route = int(route_offsets[last_plan])
            return tuple(
                tuple(
                    node_names[int(node)]
                    for node in route_indices[
                        int(route_offsets[route]) : int(route_offsets[route + 1])
                    ]
                )
                for route in range(first_route, last_route)
            )

        def event_removed(row_index: int) -> tuple[int, ...]:
            first = int(removed_offsets[row_index])
            last = int(removed_offsets[row_index + 1])
            return tuple(int(node) for node in removed_indices[first:last])

        def replay_routes(
            routes: tuple[CustomerSequence, ...],
        ) -> SolutionObjective | None:
            objective = SolutionObjective.zero()
            for sequence in routes:
                exact = solve_exact_charging(instance, sequence)
                if not exact.feasible:
                    return None
                objective += SolutionObjective.from_route(
                    instance,
                    exact.route,
                    total_distance=exact.distance,
                    total_charging_time=exact.charging_time,
                )
            return objective

        previous_routes: tuple[CustomerSequence, ...] = initial_customer_sequences
        previous_objective = replay_routes(previous_routes)
        if previous_objective is None:
            raise RuntimeError("native global initial objective replay is infeasible")
        expected_stage_status = np.full((completed_iterations, 4), -1, dtype=np.int64)
        expected_stage_weights = np.ones(
            (completed_iterations, 4, 2),
            dtype=np.float64,
        )
        expected_stage_rewards = np.zeros(
            (completed_iterations, 4),
            dtype=np.float64,
        )
        for round_index, rows in enumerate(iteration_windows):
            quality_row = events[rows[0]]
            ranked_row = events[rows[1]]
            repaired_row = events[rows[2]]
            legacy_row = events[rows[3]] if rows.size == 4 else None
            constraint_operator = int(ranked_row[3])
            if (
                int(quality_row[1]) != 1
                or int(quality_row[3]) != 4
                or int(quality_row[4]) != 0
                or int(quality_row[5]) != 0
                or int(quality_row[15]) != 0
                or int(quality_row[20]) != -2
                or int(ranked_row[1]) != 2
                or not 9 <= constraint_operator <= 12
                or int(ranked_row[4]) != 1
                or int(ranked_row[5]) != 1
                or int(ranked_row[9]) != 1
                or int(ranked_row[12]) != 1
                or int(ranked_row[15]) != 2
                or int(ranked_row[17]) != 0
                or int(repaired_row[1]) != 2
                or int(repaired_row[3]) != constraint_operator
                or int(repaired_row[15]) != 2
                or int(repaired_row[17]) != 0
                or (
                    legacy_row is not None
                    and (
                        int(legacy_row[1]) != 0
                        or int(legacy_row[3]) != 2
                        or int(legacy_row[4]) != 0
                        or int(legacy_row[5]) != 0
                    )
                )
            ):
                raise RuntimeError("native global canonical event sequence is invalid")
            removed = event_removed(int(rows[1]))
            repaired_routes = event_routes(int(rows[2]))
            partial_routes = event_routes(int(rows[1]))
            if (
                not removed
                or int(ranked_row[13]) != len(removed)
                or int(ranked_row[14]) != len(removed)
                or len(partial_routes) == 0
                or len(repaired_routes) == 0
            ):
                raise RuntimeError("native global canonical route semantics are invalid")
            removed_set = set(removed)
            partial_node_set = {
                index_by_name[name]
                for route in partial_routes
                for name in route
            }
            repaired_node_set = {
                index_by_name[name]
                for route in repaired_routes
                for name in route
            }
            if (
                len(removed_set) != len(removed)
                or removed_set & partial_node_set
                or removed_set | partial_node_set != expected_customer_nodes
                or repaired_node_set != expected_customer_nodes
            ):
                raise RuntimeError("native global canonical route semantics are invalid")
            candidate_feasible = bool(repaired_row[10])
            candidate_objective = replay_routes(repaired_routes)
            reported_candidate = (
                SolutionObjective(
                    vehicle_count=int(objective_integer[rows[2], 0]),
                    total_distance=float(objective_float[rows[2], 0]),
                    total_charging_time=float(objective_float[rows[2], 1]),
                    charging_count=int(objective_integer[rows[2], 1]),
                )
                if objective_present[rows[2]]
                else None
            )
            if (
                (reported_candidate is None) != (candidate_objective is None)
                or (
                    reported_candidate is not None
                    and candidate_objective is not None
                    and reported_candidate.key != candidate_objective.key
                )
                or candidate_feasible != (candidate_objective is not None)
            ):
                raise RuntimeError("native global objective replay is invalid")
            if candidate_objective is None:
                comparison = "worse"
                candidate_better = False
                candidate_equal = False
            else:
                candidate_better = candidate_objective.key < previous_objective.key
                candidate_equal = candidate_objective.key == previous_objective.key
                comparison = (
                    "better"
                    if candidate_better
                    else "equal"
                    if candidate_equal
                    else "worse"
                )
            accepted = bool(repaired_row[6])
            vehicle_reduction = bool(repaired_row[7])
            distance_improvement = bool(repaired_row[8])
            expected_vehicle_reduction = bool(
                candidate_objective is not None
                and candidate_objective.vehicle_count < previous_objective.vehicle_count
            )
            expected_distance_improvement = bool(
                candidate_objective is not None
                and candidate_objective.total_distance
                < previous_objective.total_distance - 1e-9
            )
            if (
                vehicle_reduction != expected_vehicle_reduction
                or distance_improvement != expected_distance_improvement
                or (terminal_reason == 1 and round_index == completed_iterations - 1 and accepted)
            ):
                raise RuntimeError("native global objective replay is invalid")
            if accepted:
                if candidate_objective is None or candidate_objective.key > previous_objective.key:
                    raise RuntimeError("native global objective replay is invalid")
                previous_routes = repaired_routes
                previous_objective = candidate_objective
            operation = constraint_operator - 9
            if np.any(stage04_calls[round_index] != np.eye(4, dtype=np.int64)[operation]):
                raise RuntimeError("native global Stage 4 replay is invalid")
            expected_reward = stage04_config.reward_for(
                accepted=accepted,
                comparison=comparison,
                is_global_best=accepted and candidate_better,
                vehicle_reduction=expected_vehicle_reduction,
            )
            expected_stage_rewards[round_index, operation] = expected_reward
            if (
                (expected_start_iteration + round_index + 1)
                % stage04_config.segment_length
                == 0
                and terminal_reason != 1
                and not stage04_config.fixed_weights
            ):
                expected_stage_status[round_index] = np.asarray(
                    [
                        1
                        if int(calls) >= stage04_config.min_calls_per_operator
                        else 0
                        for calls in stage04_calls[round_index]
                    ],
                    dtype=np.int64,
                )
        if (
            not np.array_equal(stage04_status, expected_stage_status)
            or not np.allclose(
                stage04_weights,
                expected_stage_weights,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.allclose(
                stage04_rewards,
                expected_stage_rewards,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise RuntimeError("native global Stage 4 replay is invalid")
        return NativeGlobalSemanticStream(
            neighborhood_events=project_events(),
            event_integer=_readonly_copy(events),
            stage04_calls=_readonly_copy(stage04_calls),
            termination=_readonly_copy(termination),
            transaction_sha256=transaction_sha256,
            initial_temperature=initial_temperature,
        )

    canonical_columns = np.asarray(
        [1, 3, 4, 5, 9, 10, 15, 16, 20, 24, 25],
        dtype=np.int64,
    )
    candidate_feasible = bool(events[2, 10]) if event_count >= 3 else False
    normal_canonical_values = np.asarray(
        [
                [1, 4, 0, 0, 0, 0, 0, -1, -2, -1, -1],
            [2, 9, 1, 1, 1, 0, 2, 0, -2, 0, 0],
            [
                2,
                    9,
                1 if candidate_feasible else 2,
                2 if candidate_feasible else 3,
                1,
                1 if candidate_feasible else 0,
                2,
                -1,
                0 if candidate_feasible else -2,
                0,
                0,
            ],
                [0, 2, 0, 0, 0, 0, 0, -1, -2, -1, -1],
        ],
        dtype=np.int64,
    )
    expected_event_count = 3 if terminal_reason == 1 else 4
    canonical_values = normal_canonical_values[:expected_event_count]
    removed_count = int(events[1, 14]) if event_count == expected_event_count else -1
    expected_removed_offsets = [0, 0, removed_count, removed_count * 2]
    expected_plan_offsets = [0, 0, 1, 2]
    expected_objective_presence = [
        False,
        candidate_feasible,
        candidate_feasible,
    ]
    if terminal_reason == 0:
        expected_removed_offsets.append(removed_count * 2)
        expected_plan_offsets.append(2)
        expected_objective_presence.append(False)
    if (
        event_count != expected_event_count
        or not np.array_equal(events[:, canonical_columns], canonical_values)
        or np.any(events[:, 2] != expected_start_iteration)
        or events[1, 12] != 1
        or events[1, 13] <= 0
        or events[1, 13] != removed_count
        or events[2, 13] != events[1, 13]
        or events[2, 14] != removed_count
        or events[1, 22] != 0
        or events[2, 22] != 0
        or events[1, 17] != 0
        or events[2, 17] not in {-1, 0}
        or removed_offsets.tolist() != expected_removed_offsets
        or plan_offsets.tolist() != expected_plan_offsets
        or len(route_offsets) != 3
        or not np.array_equal(
            removed_indices[:removed_count],
            removed_indices[removed_count:],
        )
        or objective_present.tolist() != expected_objective_presence
        or not np.array_equal(objective_integer[1], objective_integer[2])
        or not np.array_equal(objective_float[1], objective_float[2], equal_nan=True)
        or stage04_calls.tolist() != [[1, 0, 0, 0]]
        or np.any(stage04_rewards[:, 1:] != 0.0)
    ):
        raise RuntimeError("native global canonical event sequence is invalid")
    partial_nodes = route_indices[int(route_offsets[0]) : int(route_offsets[1])]
    repaired_nodes = route_indices[int(route_offsets[1]) : int(route_offsets[2])]
    removed_nodes = removed_indices[:removed_count]
    index_by_name = {name: index for index, name in enumerate(node_names)}
    expected_customer_nodes = {
        index_by_name[customer.name] for customer in instance.customers
    }
    if (
        len(set(int(node) for node in removed_nodes)) != len(removed_nodes)
        or len(set(int(node) for node in partial_nodes)) != len(partial_nodes)
        or len(set(int(node) for node in repaired_nodes)) != len(repaired_nodes)
        or set(int(node) for node in partial_nodes) & set(int(node) for node in removed_nodes)
        or set(int(node) for node in partial_nodes) | set(int(node) for node in removed_nodes)
        != expected_customer_nodes
        or set(int(node) for node in repaired_nodes) != expected_customer_nodes
    ):
        raise RuntimeError("native global canonical route semantics are invalid")
    def replay_objective(
        sequences: tuple[CustomerSequence, ...],
    ) -> SolutionObjective | None:
        objective = SolutionObjective.zero()
        for sequence in sequences:
            exact = solve_exact_charging(instance, sequence)
            if not exact.feasible:
                return None
            objective += SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
        return objective

    initial_objective = replay_objective(initial_customer_sequences)
    if initial_objective is None:
        raise RuntimeError("native global initial objective replay is infeasible")
    repaired_sequences = (
        tuple(node_names[int(node)] for node in repaired_nodes),
    )
    replayed_candidate = replay_objective(repaired_sequences)
    reported_candidate = (
        SolutionObjective(
            vehicle_count=int(objective_integer[2, 0]),
            total_distance=float(objective_float[2, 0]),
            total_charging_time=float(objective_float[2, 1]),
            charging_count=int(objective_integer[2, 1]),
        )
        if objective_present[2]
        else None
    )
    candidate_better = (
        replayed_candidate is not None
        and replayed_candidate.key < initial_objective.key
    )
    candidate_equal = (
        replayed_candidate is not None
        and replayed_candidate.key == initial_objective.key
    )
    expected_vehicle_reduction = bool(
        replayed_candidate is not None
        and replayed_candidate.vehicle_count < initial_objective.vehicle_count
    )
    expected_distance_improvement = bool(
        replayed_candidate is not None
        and replayed_candidate.total_distance < initial_objective.total_distance - 1e-9
    )
    accepted = bool(events[2, 6])
    if (
        (reported_candidate is None) != (replayed_candidate is None)
        or (
            reported_candidate is not None
            and replayed_candidate is not None
            and reported_candidate.key != replayed_candidate.key
        )
        or candidate_feasible != (replayed_candidate is not None)
        or bool(events[2, 7]) != expected_vehicle_reduction
        or bool(events[2, 8]) != expected_distance_improvement
        or (terminal_reason == 1 and accepted)
    ):
        raise RuntimeError("native global objective replay is invalid")
    comparison = "better" if candidate_better else "equal" if candidate_equal else "worse"
    expected_reward = stage04_config.reward_for(
        accepted=accepted,
        comparison=comparison,
        is_global_best=accepted and candidate_better,
        vehicle_reduction=expected_vehicle_reduction,
    )
    expected_status = np.full((1, 4), -1, dtype=np.int64)
    expected_weights = np.ones((1, 4, 2), dtype=np.float64)
    is_segment_boundary = (
        (expected_start_iteration + expected_iteration_count)
        % stage04_config.segment_length
        == 0
    )
    if (
        is_segment_boundary
        and terminal_reason != 1
        and not stage04_config.fixed_weights
    ):
        expected_status[0] = [
            1 if calls >= stage04_config.min_calls_per_operator else 0
            for calls in stage04_calls[0]
        ]
        if expected_status[0, 0] == 1:
            expected_weights[0, 0, 1] = stage04_config.apply_segment_update(
                1.0,
                expected_reward,
                1,
            )
    expected_rewards = np.asarray(
        [[expected_reward, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    if (
        not np.array_equal(stage04_status, expected_status)
        or not np.allclose(stage04_weights, expected_weights, rtol=0.0, atol=1e-12)
        or not np.allclose(stage04_rewards, expected_rewards, rtol=0.0, atol=1e-12)
    ):
        raise RuntimeError("native global Stage 4 replay is invalid")
    return NativeGlobalSemanticStream(
        neighborhood_events=project_events(),
        event_integer=_readonly_copy(events),
        stage04_calls=_readonly_copy(stage04_calls),
        termination=_readonly_copy(termination),
        transaction_sha256=transaction_sha256,
        initial_temperature=initial_temperature,
    )


def _append_native_nested_evidence(evidence: bytearray, value: object) -> None:
    if value is None:
        evidence.extend(b"N")
    elif isinstance(value, tuple):
        evidence.extend(b"T")
        _append_u64(evidence, len(value))
        for item in value:
            _append_native_nested_evidence(evidence, item)
    elif isinstance(value, np.ndarray):
        evidence.extend(b"A" + value.dtype.str.encode() + b"\0")
        _append_typed_array(evidence, value)
    elif isinstance(value, str):
        encoded = value.encode()
        evidence.extend(b"S")
        _append_u64(evidence, len(encoded))
        evidence.extend(encoded)
    elif isinstance(value, bool):
        evidence.extend(b"B" + bytes((int(value),)))
    elif isinstance(value, int):
        evidence.extend(b"I")
        _append_u64(evidence, value)
    elif isinstance(value, float):
        evidence.extend(b"F")
        _append_u64(evidence, struct.unpack("<Q", struct.pack("<d", value))[0])
    else:
        raise RuntimeError("native semantic evidence contains an unsupported value")


def _verify_native_three_lane_semantic_hash(payload: tuple[object, ...]) -> str:
    evidence = bytearray(b"stage05.2-native-three-lane-semantic-stream-v2")
    _append_native_nested_evidence(evidence, payload[:13])
    producer_sha256 = payload[13]
    if (
        not isinstance(producer_sha256, str)
        or not _is_sha256(producer_sha256)
        or hashlib.sha256(evidence).hexdigest() != producer_sha256
    ):
        raise RuntimeError("native three-lane semantic stream SHA-256 mismatch")
    return producer_sha256


def _native_refinement_outcome(
    metadata: npt.NDArray[np.int64],
) -> tuple[str, str, bool]:
    """Project the typed native refinement outcome onto Python semantics."""

    if metadata.shape != (4,):
        raise RuntimeError("native refinement metadata shape is invalid")
    failure_code = int(metadata[0])
    selected = int(metadata[1])
    if failure_code not in (0, 1, 2) or selected not in (0, 1):
        raise RuntimeError("native refinement metadata outcome is invalid")
    if selected:
        if failure_code != 0:
            raise RuntimeError("native selected refinement has a failure code")
        return "candidate_proposed", "vehicle_reduction_refined", True
    return (
        "failed",
        {
            0: "refinement_not_better",
            1: "no_existing_route_insertion",
            2: "evaluation_budget_exhausted",
        }[failure_code],
        False,
    )


def _python_neighborhood_event_projection(
    events: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Match ``NeighborhoodEvent.to_dict`` omission rules after replay."""

    projected: list[Mapping[str, object]] = []
    for raw in events:
        event = dict(raw)
        if event.get("aggregate_count") == 1:
            event.pop("aggregate_count")
        if not event.get("candidate_pool_hash"):
            event.pop("candidate_pool_hash", None)
        projected.append(event)
    return tuple(projected)


def decode_native_three_lane_semantic_stream(
    instance: Instance,
    payload: object,
    *,
    node_names: tuple[str, ...],
    initial_customer_sequences: tuple[CustomerSequence, ...],
) -> NativeThreeLaneSemanticStream:
    """Replay one multi-route bootstrap without trusting native search decisions."""

    if not isinstance(payload, tuple) or len(payload) != 14:
        raise RuntimeError("native three-lane semantic payload has an invalid tuple")
    producer_sha256 = _verify_native_three_lane_semantic_hash(payload)
    if tuple(node.name for node in instance.nodes) != node_names:
        raise RuntimeError("native three-lane semantic node identity is invalid")
    expected_customers = {customer.name for customer in instance.customers}
    flattened_initial = tuple(
        customer for route in initial_customer_sequences for customer in route
    )
    if (
        not initial_customer_sequences
        or len(flattened_initial) != len(expected_customers)
        or set(flattened_initial) != expected_customers
    ):
        raise RuntimeError("native three-lane initial incumbent identity is invalid")
    name_by_index = node_names
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="three-lane termination",
        ),
    )
    full_stage04_value = payload[12]
    if not isinstance(full_stage04_value, tuple) or len(full_stage04_value) != 4:
        raise RuntimeError("native three-lane full Stage 4 has an invalid tuple")
    weights = cast(
        npt.NDArray[np.float64],
        _require_array(
            full_stage04_value[0],
            dtype=np.dtype(np.float64),
            shape=(len(FULL_NATIVE_OPERATOR_NAMES),),
            name="full Stage 4 weights",
        ),
    )
    rewards = cast(
        npt.NDArray[np.float64],
        _require_array(
            full_stage04_value[1],
            dtype=np.dtype(np.float64),
            shape=(len(FULL_NATIVE_OPERATOR_NAMES),),
            name="full Stage 4 rewards",
        ),
    )
    calls = cast(
        npt.NDArray[np.int64],
        _require_array(
            full_stage04_value[2],
            dtype=np.dtype(np.int64),
            shape=(len(FULL_NATIVE_OPERATOR_NAMES),),
            name="full Stage 4 calls",
        ),
    )
    totals = cast(
        npt.NDArray[np.int64],
        _require_array(
            full_stage04_value[3],
            dtype=np.dtype(np.int64),
            shape=(len(FULL_NATIVE_OPERATOR_NAMES), 8),
            name="full Stage 4 totals",
        ),
    )
    if (
        np.any(weights <= 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(rewards < 0.0)
        or np.any(~np.isfinite(rewards))
        or np.any(calls < 0)
        or np.any(totals < 0)
        or int(termination[0]) not in {0, 1, 2}
        or int(termination[2]) != int(termination[3]) + int(termination[4])
        or int(termination[5]) not in {0, 1}
    ):
        raise RuntimeError("native three-lane final state is invalid")

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native three-lane {name} has an invalid tuple")
        return value

    def unpack_routes(
        offsets_value: object,
        indices_value: object,
        name: str,
    ) -> tuple[CustomerSequence, ...]:
        offsets = _require_vector(offsets_value, f"{name} offsets")
        indices = _require_vector(indices_value, f"{name} indices")
        if (
            len(offsets) < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or np.any(offsets[:-1] >= offsets[1:])
            or np.any(indices < 0)
            or np.any(indices >= len(name_by_index))
        ):
            raise RuntimeError(f"native three-lane {name} route SoA is invalid")
        routes = tuple(
            tuple(
                name_by_index[int(index)]
                for index in indices[int(offsets[route]) : int(offsets[route + 1])]
            )
            for route in range(len(offsets) - 1)
        )
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(set(flattened)):
            raise RuntimeError(f"native three-lane {name} repeats a customer")
        return routes

    def replay_objective(routes: tuple[CustomerSequence, ...]) -> SolutionObjective:
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(expected_customers) or set(flattened) != expected_customers:
            raise RuntimeError("native three-lane candidate customer identity is invalid")
        objective = SolutionObjective.zero()
        for route in routes:
            exact = solve_exact_charging(instance, route)
            if not exact.feasible:
                raise RuntimeError("native three-lane candidate objective replay is infeasible")
            objective += SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
        return objective

    def require_plan_transaction(
        value: object,
        plan_count: int,
        name: str,
    ) -> tuple[object, ...]:
        transaction = require_tuple(value, 13, f"{name} transaction")
        _require_array(
            transaction[1],
            dtype=np.dtype(np.int64),
            shape=(plan_count,),
            name=f"{name} plan statuses",
        )
        _require_array(
            transaction[2],
            dtype=np.dtype(np.int64),
            shape=(plan_count, 2),
            name=f"{name} objective integers",
        )
        _require_array(
            transaction[3],
            dtype=np.dtype(np.float64),
            shape=(plan_count, 2),
            name=f"{name} objective floats",
        )
        _require_vector(transaction[5], f"{name} exact rows")
        _require_vector(transaction[11], f"{name} feasible order")
        transaction_hash = transaction[12]
        if not isinstance(transaction_hash, str) or not _is_sha256(transaction_hash):
            raise RuntimeError(f"native three-lane {name} transaction hash is invalid")
        return transaction

    def reported_objective(
        transaction: tuple[object, ...],
        plan: int,
        replayed: SolutionObjective,
        name: str,
    ) -> tuple[int, float, float, int]:
        integers = cast(npt.NDArray[np.int64], transaction[2])
        floats = cast(npt.NDArray[np.float64], transaction[3])
        reported = SolutionObjective(
            vehicle_count=int(integers[plan, 0]),
            total_distance=float(floats[plan, 0]),
            total_charging_time=float(floats[plan, 1]),
            charging_count=int(integers[plan, 1]),
        )
        if reported.key != replayed.key:
            raise RuntimeError(f"native three-lane {name} objective replay mismatch")
        return reported.key

    def event(
        operator: str,
        status: str,
        reason: str,
        *,
        iteration: int = 0,
        **changes: object,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "operator": operator,
            "status": status,
            "reason": reason,
            "route_indices": (),
            "affected_route_indices": (),
            "removed_customers": (),
            "candidate_customer_sequence": (),
            "candidate_route_sequences": (),
            "candidate_vehicle_delta": None,
            "candidate_feasible": False,
            "prefilter_passed": False,
            "new_routes_created": 0,
            "exact_route_evaluations": 0,
            "selection_rank": 0,
            "chain_depth": 0,
            "segment_length": 0,
            "track": "legacy",
            "constraint_category": "",
            "removal_tier": "",
            "removal_size_requested": 0,
            "removal_size_actual": 0,
            "stagnation_iterations": 0,
            "removal_trigger": "",
            "reset_observed": False,
            "ranking_score": 0.0,
            "iteration": iteration,
            "accepted": False,
            "vehicle_reduction": False,
            "distance_improvement": False,
            "candidate_objective_key": (),
        }
        result.update(changes)
        return result

    def activity_from_events(
        events: Sequence[Mapping[str, object]],
    ) -> npt.NDArray[np.int64]:
        activity = np.zeros((len(FULL_NATIVE_OPERATOR_NAMES), 8), dtype=np.int64)
        by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
        for item in events:
            operator_index = by_name[str(item["operator"])]
            activity[operator_index, 0] = max(
                int(activity[operator_index, 0]),
                int(bool(item["candidate_feasible"])),
            )
            activity[operator_index, 1] += int(bool(item["prefilter_passed"]))
            activity[operator_index, 2] += cast(int, item["exact_route_evaluations"])
            activity[operator_index, 3] += int(item["status"] == "candidate_proposed")
            activity[operator_index, 4] += int(bool(item["candidate_feasible"]))
            activity[operator_index, 5] += int(bool(item["vehicle_reduction"]))
            activity[operator_index, 6] += int(bool(item["distance_improvement"]))
        refinement_index = FULL_NATIVE_OPERATOR_NAMES.index(
            "vehicle_reduction_refinement"
        )
        activity[refinement_index, 5] = 0
        activity[refinement_index, 0] = int(
            any(
                str(item["operator"]) == "vehicle_reduction_refinement"
                and (
                    bool(item["candidate_feasible"])
                    or str(item["reason"]) == "refinement_not_better"
                )
                for item in events
            )
        )
        return activity

    initialization = require_tuple(payload[10], 5, "Stage 4 initialization")
    if not isinstance(initialization[0], float):
        raise RuntimeError(
            "native three-lane Stage 4 initial temperature has an invalid type"
        )
    initial_temperature = initialization[0]
    temperature_deltas = cast(
        npt.NDArray[np.float64],
        _require_array(
            initialization[1],
            dtype=np.dtype(np.float64),
            shape=(len(cast(npt.NDArray[np.float64], initialization[1])),),
            name="Stage 4 temperature deltas",
        ),
    )
    temperature_plan_offsets = _require_vector(
        initialization[2], "Stage 4 temperature plan offsets"
    )
    temperature_route_offsets = _require_vector(
        initialization[3], "Stage 4 temperature route offsets"
    )
    temperature_route_indices = _require_vector(
        initialization[4], "Stage 4 temperature route indices"
    )
    if (
        not math.isfinite(initial_temperature)
        or initial_temperature < 1.0
        or np.any(~np.isfinite(temperature_deltas))
        or np.any(temperature_deltas <= 0.0)
        or len(temperature_plan_offsets) < 1
        or len(temperature_route_offsets) < 1
        or int(temperature_plan_offsets[0]) != 0
        or int(temperature_plan_offsets[-1]) != len(temperature_route_offsets) - 1
        or int(temperature_route_offsets[0]) != 0
        or int(temperature_route_offsets[-1]) != len(temperature_route_indices)
        or np.any(temperature_plan_offsets[:-1] > temperature_plan_offsets[1:])
        or np.any(temperature_route_offsets[:-1] > temperature_route_offsets[1:])
        or np.any(temperature_route_indices < 0)
        or np.any(temperature_route_indices >= len(name_by_index))
    ):
        raise RuntimeError("native three-lane Stage 4 initialization is invalid")

    initial_objective = replay_objective(initial_customer_sequences)
    if int(termination[0]) in {1, 2}:
        best = require_tuple(payload[9], 5, "terminal best")
        terminal_routes = unpack_routes(best[0], best[1], "terminal best")
        terminal_objective = replay_objective(terminal_routes)
        terminal_expected_routes = initial_customer_sequences
        terminal_expected_objective = initial_objective
        if int(termination[0]) == 1 and (
            terminal_routes != terminal_expected_routes
            or terminal_objective.key != terminal_expected_objective.key
        ):
            raise RuntimeError(
                "native three-lane terminal changed the last complete incumbent"
            )
        terminal_events: list[dict[str, object]] = []
        if payload[2] is not None:
            quality = require_tuple(payload[2], 4, "terminal quality")
            quality_pool = require_tuple(quality[0], 5, "terminal quality pool")
            changed_routes = cast(
                npt.NDArray[np.int64],
                _require_array(
                    quality_pool[0],
                    dtype=np.dtype(np.int64),
                    shape=(len(cast(npt.NDArray[np.int64], quality_pool[0])), 2),
                    name="terminal quality changed routes",
                ),
            )
            pool_count = changed_routes.shape[0]
            change_offsets = _require_vector(
                quality_pool[1], "terminal quality change offsets"
            )
            change_indices = _require_vector(
                quality_pool[2], "terminal quality change indices"
            )
            moved_customers = _require_vector(
                quality_pool[4], "terminal quality moved customers"
            )
            terminal_quality_plans: list[tuple[CustomerSequence, ...]] = []
            terminal_quality_sources: list[int] = []
            terminal_seen_quality: set[tuple[CustomerSequence, ...]] = set()
            for candidate in range(pool_count):
                routes = list(initial_customer_sequences)
                for changed_ordinal in range(2):
                    route = int(changed_routes[candidate, changed_ordinal])
                    begin = int(change_offsets[candidate * 2 + changed_ordinal])
                    end = int(change_offsets[candidate * 2 + changed_ordinal + 1])
                    routes[route] = tuple(
                        name_by_index[int(index)] for index in change_indices[begin:end]
                    )
                plan = tuple(routes)
                if plan not in terminal_seen_quality:
                    terminal_seen_quality.add(plan)
                    terminal_quality_plans.append(plan)
                    terminal_quality_sources.append(candidate)
            quality_outcome = cast(
                npt.NDArray[np.int64],
                _require_array(
                    quality[2],
                    dtype=np.dtype(np.int64),
                    shape=(4,),
                    name="terminal quality outcome",
                ),
            )
            selected_quality = int(quality_outcome[0])
            pool_hash = hashlib.sha256()
            for plan in terminal_quality_plans:
                pool_hash.update(json.dumps(plan, separators=(",", ":")).encode())
            terminal_events.append(
                event(
                    "relocate",
                    "candidate_pool_aggregate",
                    "relocate_complete_candidate_pool",
                    aggregate_count=len(terminal_quality_plans),
                    candidate_pool_hash=pool_hash.hexdigest(),
                )
            )
            if selected_quality >= 0:
                if selected_quality >= len(terminal_quality_plans):
                    raise RuntimeError(
                        "native three-lane terminal quality selection is invalid"
                    )
                quality_transaction = require_plan_transaction(
                    quality[1], len(terminal_quality_plans), "terminal quality"
                )
                quality_routes = terminal_quality_plans[selected_quality]
                quality_objective = replay_objective(quality_routes)
                quality_objective_key = reported_objective(
                    quality_transaction,
                    selected_quality,
                    quality_objective,
                    "terminal quality",
                )
                source = terminal_quality_sources[selected_quality]
                moved = int(moved_customers[source])
                if int(termination[0]) == 1 and (
                    bool(quality_outcome[1]) or bool(quality_outcome[2])
                ):
                    raise RuntimeError(
                        "native three-lane terminal accepted an incomplete quality move"
                    )
                terminal_events.append(
                    event(
                        "relocate",
                        "candidate_proposed",
                        "relocate_candidate",
                        route_indices=tuple(int(v) for v in changed_routes[source]),
                        affected_route_indices=tuple(
                            int(v) for v in changed_routes[source]
                        ),
                        removed_customers=(name_by_index[moved],),
                        candidate_route_sequences=tuple(
                            quality_routes[index]
                            for index in changed_routes[source]
                        ),
                        candidate_vehicle_delta=len(quality_routes)
                        - len(initial_customer_sequences),
                        candidate_feasible=True,
                        prefilter_passed=True,
                        accepted=bool(quality_outcome[1]),
                        vehicle_reduction=bool(quality_outcome[3]),
                        distance_improvement=quality_objective.total_distance
                        < initial_objective.total_distance - 1e-9,
                        candidate_objective_key=quality_objective_key,
                    )
                )
                terminal_events[0]["candidate_objective_key"] = quality_objective_key
                if bool(quality_outcome[2]):
                    if not bool(quality_outcome[1]):
                        raise RuntimeError(
                            "native three-lane terminal best quality was not accepted"
                        )
                    terminal_expected_routes = quality_routes
                    terminal_expected_objective = quality_objective
            else:
                terminal_events.append(
                    event(
                        "relocate",
                        "candidate_control_skipped",
                        "no_selected_complete_plan_feasible",
                    )
                )
        if payload[3] is not None or payload[4] is not None or payload[5] is not None:
            raise RuntimeError(
                "native three-lane terminal retained post-boundary semantic state"
            )
        if (
            terminal_routes != terminal_expected_routes
            or terminal_objective.key != terminal_expected_objective.key
        ):
            raise RuntimeError(
                "native three-lane terminal best does not match completed transactions"
            )
        operator_replay_events = list(terminal_events)
        if payload[0] is not None:
            terminal_legacy = require_tuple(payload[0], 8, "terminal legacy")
            terminal_plan_offsets = _require_vector(
                terminal_legacy[2], "terminal legacy plan offsets"
            )
            terminal_route_offsets = _require_vector(
                terminal_legacy[3], "terminal legacy route offsets"
            )
            terminal_route_indices = _require_vector(
                terminal_legacy[4], "terminal legacy route indices"
            )
            terminal_legacy_outcome = cast(
                npt.NDArray[np.int64],
                _require_array(
                    terminal_legacy[6],
                    dtype=np.dtype(np.int64),
                    shape=(5,),
                    name="terminal legacy outcome",
                ),
            )
            terminal_legacy_order = _require_vector(
                terminal_legacy[0], "terminal legacy route order"
            )
            terminal_legacy_attempts = cast(
                npt.NDArray[np.int64],
                _require_array(
                    terminal_legacy[1],
                    dtype=np.dtype(np.int64),
                    shape=(len(terminal_legacy_order), 6),
                    name="terminal legacy attempts",
                ),
            )
            if (
                not np.array_equal(
                    terminal_legacy_order, terminal_legacy_attempts[:, 0]
                )
                or np.any(terminal_legacy_order < 0)
                or np.any(terminal_legacy_order >= len(initial_customer_sequences))
            ):
                raise RuntimeError(
                    "native three-lane terminal legacy attempt journal is invalid"
                )
            operator_replay_events.extend(
                event(
                    "route_elimination",
                    "failed",
                    "no_existing_route_insertion"
                    if int(attempt[2]) == 1
                    else "singleton_route_screening_rejected",
                    route_indices=(int(attempt[0]),),
                    removed_customers=initial_customer_sequences[int(attempt[0])],
                    selection_rank=int(attempt[1]),
                )
                for attempt in terminal_legacy_attempts
                if int(attempt[2]) != 0
            )
            if int(terminal_legacy_outcome[0]) >= 0:
                selected_plan = int(terminal_legacy_outcome[0])
                source_route = int(terminal_legacy_outcome[4])
                if (
                    selected_plan + 1 >= len(terminal_plan_offsets)
                    or source_route < 0
                    or source_route >= len(initial_customer_sequences)
                ):
                    raise RuntimeError(
                        "native three-lane terminal legacy selection is invalid"
                    )
                first_route = int(terminal_plan_offsets[selected_plan])
                last_route = int(terminal_plan_offsets[selected_plan + 1])
                candidate_routes = tuple(
                    tuple(
                        name_by_index[int(index)]
                        for index in terminal_route_indices[
                            int(terminal_route_offsets[route]) : int(
                                terminal_route_offsets[route + 1]
                            )
                        ]
                    )
                    for route in range(first_route, last_route)
                )
                terminal_transaction = require_tuple(
                    terminal_legacy[5], 13, "terminal legacy transaction"
                )
                candidate_objective = replay_objective(candidate_routes)
                candidate_key = reported_objective(
                    terminal_transaction,
                    selected_plan,
                    candidate_objective,
                    "terminal legacy",
                )
                exact_evaluations = len(
                    _require_vector(
                        terminal_transaction[5], "terminal legacy exact"
                    )
                )
                operator_replay_events.append(
                    event(
                        "route_elimination",
                        "candidate_proposed",
                        "route_eliminated_and_repaired",
                        route_indices=(source_route,),
                        affected_route_indices=tuple(range(len(candidate_routes))),
                        removed_customers=initial_customer_sequences[source_route],
                        candidate_route_sequences=candidate_routes,
                        candidate_vehicle_delta=len(candidate_routes)
                        - len(initial_customer_sequences),
                        candidate_feasible=True,
                        prefilter_passed=True,
                        exact_route_evaluations=exact_evaluations,
                        selection_rank=selected_plan + 1,
                        accepted=False,
                        vehicle_reduction=candidate_objective.vehicle_count
                        < initial_objective.vehicle_count,
                        distance_improvement=candidate_objective.total_distance
                        < initial_objective.total_distance - 1e-9,
                        candidate_objective_key=candidate_key,
                    )
                )
            else:
                if int(terminal_legacy_outcome[0]) != -1:
                    raise RuntimeError(
                        "native three-lane terminal legacy failure journal is invalid"
                    )
                terminal_exact_evaluations = 0
                if terminal_legacy[5] is not None:
                    terminal_transaction = require_tuple(
                        terminal_legacy[5], 13, "terminal legacy transaction"
                    )
                    terminal_exact_evaluations = len(
                        _require_vector(
                            terminal_transaction[5], "terminal legacy exact"
                        )
                    )
                operator_replay_events.append(
                    event(
                        "route_elimination",
                        "candidate_control_skipped",
                        "no_selected_complete_plan_feasible",
                        exact_route_evaluations=terminal_exact_evaluations,
                    )
                )
        refinement_failure_code: int | None = None
        if payload[1] is not None:
            terminal_refinement = require_tuple(
                payload[1], 4, "terminal refinement"
            )
            refinement_metadata = cast(
                npt.NDArray[np.int64],
                _require_array(
                    terminal_refinement[0],
                    dtype=np.dtype(np.int64),
                    shape=(4,),
                    name="terminal refinement metadata",
                ),
            )
            refinement_status, refinement_reason, refinement_selected = (
                _native_refinement_outcome(refinement_metadata)
            )
            if int(termination[0]) == 1 and refinement_selected:
                # Python retains the work counters for a refinement that found
                # a candidate at the exact-call boundary, but atomically
                # discards its outcome classification with the incomplete
                # enclosing ALNS iteration.  Keep this replay-only marker out
                # of the public neighborhood stream while preserving that
                # neutral accounting contract.
                operator_replay_events.append(
                    event(
                        "vehicle_reduction_refinement",
                        "candidate_proposed",
                        "vehicle_reduction_refined",
                        candidate_feasible=True,
                        prefilter_passed=True,
                        exact_route_evaluations=int(refinement_metadata[2]),
                        distance_improvement=True,
                        accepted=False,
                        _operator_budget_discarded=True,
                    )
                )
            else:
                # At a fixed-work boundary Python keeps the proposal-level
                # repair reason in the neighborhood event, while its legacy
                # operator statistics classify the enclosing interruption as
                # a refinement time-limit failure.  Preserve both projections.
                refinement_failure_code = (
                    1
                    if int(termination[0]) == 1
                    and int(refinement_metadata[0]) != 0
                    else {
                        "refinement_not_better": 2,
                        "no_existing_route_insertion": 3,
                        "evaluation_budget_exhausted": 4,
                    }[refinement_reason]
                )
                operator_replay_events.append(
                    event(
                        "vehicle_reduction_refinement",
                        refinement_status,
                        refinement_reason,
                        prefilter_passed=True,
                        exact_route_evaluations=int(refinement_metadata[2]),
                    )
                )
        activity = activity_from_events(operator_replay_events)
        if refinement_failure_code is not None:
            refinement_index = FULL_NATIVE_OPERATOR_NAMES.index(
                "vehicle_reduction_refinement"
            )
            activity[refinement_index, 5] = 0
            activity[refinement_index, 7] = refinement_failure_code
        return NativeThreeLaneSemanticStream(
            neighborhood_events=tuple(terminal_events),
            operator_weights=_readonly_copy(weights),
            operator_rewards=_readonly_copy(rewards),
            operator_calls=_readonly_copy(calls),
            operator_totals=_readonly_copy(totals),
            operator_activity=_readonly_copy(activity),
            termination=_readonly_copy(termination),
            transaction_sha256=producer_sha256,
            initial_temperature=initial_temperature,
            operator_replay_events=tuple(operator_replay_events),
        )

    legacy = require_tuple(payload[0], 8, "legacy")
    legacy_plan_offsets = _require_vector(legacy[2], "legacy plan offsets")
    legacy_route_offsets = _require_vector(legacy[3], "legacy route offsets")
    legacy_route_indices = _require_vector(legacy[4], "legacy route indices")
    if (
        int(legacy_plan_offsets[0]) != 0
        or int(legacy_plan_offsets[-1]) != len(legacy_route_offsets) - 1
        or int(legacy_route_offsets[0]) != 0
        or int(legacy_route_offsets[-1]) != len(legacy_route_indices)
    ):
        raise RuntimeError("native three-lane legacy plan SoA is invalid")
    legacy_plan_count = len(legacy_plan_offsets) - 1
    legacy_outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[6], dtype=np.dtype(np.int64), shape=(5,), name="legacy outcome"
        ),
    )
    selected_legacy = int(legacy_outcome[0])
    selected_source = int(legacy_outcome[4])
    if selected_legacy < 0:
        if legacy_plan_count != 0 or legacy[5] is not None or selected_source != -1:
            raise RuntimeError("native three-lane rejected legacy selection is invalid")
        if payload[1] is not None or payload[4] is not None:
            raise RuntimeError("native three-lane rejected legacy retained acceptance state")
        rejected_events: list[dict[str, object]] = []

        quality = require_tuple(payload[2], 4, "rejected quality")
        quality_pool = require_tuple(quality[0], 5, "rejected quality pool")
        changed_routes = cast(
            npt.NDArray[np.int64],
            _require_array(
                quality_pool[0],
                dtype=np.dtype(np.int64),
                shape=(len(cast(npt.NDArray[np.int64], quality_pool[0])), 2),
                name="rejected quality changed routes",
            ),
        )
        change_offsets = _require_vector(
            quality_pool[1], "rejected quality change offsets"
        )
        change_indices = _require_vector(
            quality_pool[2], "rejected quality change indices"
        )
        moved_customers = _require_vector(
            quality_pool[4], "rejected quality moved customers"
        )
        rejected_quality_plans: list[tuple[CustomerSequence, ...]] = []
        rejected_quality_sources: list[int] = []
        rejected_seen_quality: set[tuple[CustomerSequence, ...]] = set()
        for candidate in range(changed_routes.shape[0]):
            routes = list(initial_customer_sequences)
            for changed_ordinal in range(2):
                route = int(changed_routes[candidate, changed_ordinal])
                if route < 0 or route >= len(routes):
                    raise RuntimeError(
                        "native three-lane rejected quality route is invalid"
                    )
                begin = int(change_offsets[candidate * 2 + changed_ordinal])
                end = int(change_offsets[candidate * 2 + changed_ordinal + 1])
                routes[route] = tuple(
                    name_by_index[int(index)] for index in change_indices[begin:end]
                )
            plan = tuple(routes)
            if plan not in rejected_seen_quality:
                rejected_seen_quality.add(plan)
                rejected_quality_plans.append(plan)
                rejected_quality_sources.append(candidate)
        # A one-route incumbent has no relocate pair and no safely removable
        # customer for the constraint lane.  The native producer represents
        # those two legitimate Stage 2.3 no-op probes with empty pools and no
        # Candidate Control transaction.  Treating the missing transaction as
        # corruption would force one-route solves onto the obsolete synthetic
        # global controller and lose their real operator/RNG/Stage 4 sequence.
        if (
            len(expected_customers) == 1
            and not rejected_quality_plans
            and quality[1] is None
        ):
            quality_outcome = cast(
                npt.NDArray[np.int64],
                _require_array(
                    quality[2],
                    dtype=np.dtype(np.int64),
                    shape=(4,),
                    name="one-route quality outcome",
                ),
            )
            constraint = require_tuple(payload[3], 3, "one-route constraint")
            selection = cast(
                npt.NDArray[np.int64],
                _require_array(
                    constraint[0],
                    dtype=np.dtype(np.int64),
                    shape=(7,),
                    name="one-route constraint selection",
                ),
            )
            constraint_probe = require_tuple(
                constraint[1], 3, "one-route constraint probe"
            )
            removal = require_tuple(
                constraint_probe[0], 7, "one-route constraint removal"
            )
            removal_metadata = cast(
                npt.NDArray[np.int64],
                _require_array(
                    removal[6],
                    dtype=np.dtype(np.int64),
                    shape=(3,),
                    name="one-route constraint removal metadata",
                ),
            )
            constraint_outcome = cast(
                npt.NDArray[np.int64],
                _require_array(
                    constraint[2],
                    dtype=np.dtype(np.int64),
                    shape=(6,),
                    name="one-route constraint outcome",
                ),
            )
            empty_removal_vectors = all(
                len(_require_vector(removal[index], "one-route removal vector")) == 0
                for index in (1, 2, 3, 5)
            ) and len(
                cast(
                    npt.NDArray[np.float64],
                    _require_array(
                        removal[4],
                        dtype=np.dtype(np.float64),
                        shape=(0,),
                        name="one-route removal scores",
                    ),
                )
            ) == 0
            if (
                not np.array_equal(quality_outcome, np.array([-1, 0, 0, 0]))
                or constraint_probe[1] is not None
                or constraint_probe[2] is not None
                or not np.array_equal(
                    removal_metadata, np.array([2, -1, 0], dtype=np.int64)
                )
                or not empty_removal_vectors
                or np.any(constraint_outcome != 0)
                or int(selection[0]) != 0
            ):
                raise RuntimeError(
                    "native three-lane one-route no-op semantics are invalid"
                )
            best = require_tuple(payload[9], 5, "one-route best")
            if unpack_routes(best[0], best[1], "one-route best") != (
                initial_customer_sequences
            ):
                raise RuntimeError(
                    "native three-lane one-route bootstrap changed global best"
                )
            one_route_events = [
                event("relocate", "not_applicable", "only_one_route"),
                event(
                    "station_pressure",
                    "not_applicable",
                    "no_removable_customer",
                    track="constraint_lane",
                    constraint_category="station_pressure",
                    removal_tier="small",
                    removal_size_requested=int(selection[1]),
                    removal_size_actual=int(selection[2]),
                    removal_trigger="no_removable_customer",
                ),
                event(
                    "route_elimination",
                    "not_applicable",
                    "only_one_route",
                ),
            ]
            activity = activity_from_events(one_route_events)
            return NativeThreeLaneSemanticStream(
                neighborhood_events=tuple(one_route_events),
                operator_weights=_readonly_copy(weights),
                operator_rewards=_readonly_copy(rewards),
                operator_calls=_readonly_copy(calls),
                operator_totals=_readonly_copy(totals),
                operator_activity=_readonly_copy(activity),
                termination=_readonly_copy(termination),
                transaction_sha256=producer_sha256,
                initial_temperature=initial_temperature,
                operator_replay_events=tuple(one_route_events),
            )
        rejected_quality_transaction: tuple[object, ...] | None = (
            None
            if not rejected_quality_plans and quality[1] is None
            else require_plan_transaction(
                quality[1], len(rejected_quality_plans), "rejected quality"
            )
        )
        quality_outcome = cast(
            npt.NDArray[np.int64],
            _require_array(
                quality[2],
                dtype=np.dtype(np.int64),
                shape=(4,),
                name="rejected quality outcome",
            ),
        )
        pool_hash = hashlib.sha256()
        for plan in rejected_quality_plans:
            pool_hash.update(json.dumps(plan, separators=(",", ":")).encode())
        selected_quality = int(quality_outcome[0])
        quality_one_route_noop = (
            len(initial_customer_sequences) == 1
            and not rejected_quality_plans
            and selected_quality == -1
        )
        if quality_one_route_noop:
            rejected_events.append(
                event("relocate", "not_applicable", "only_one_route")
            )
        else:
            rejected_events.append(
                event(
                    "relocate",
                    "candidate_pool_aggregate",
                    "relocate_complete_candidate_pool",
                    aggregate_count=len(rejected_quality_plans),
                    candidate_pool_hash=pool_hash.hexdigest(),
                )
            )
        if selected_quality < 0:
            if bool(quality_outcome[1]) or bool(quality_outcome[2]):
                raise RuntimeError("native three-lane rejected quality outcome is invalid")
            if not quality_one_route_noop:
                rejected_events.append(
                    event(
                        "relocate",
                        "candidate_control_skipped",
                        "no_selected_complete_plan_feasible",
                    )
                )
        else:
            if selected_quality >= len(rejected_quality_plans):
                raise RuntimeError("native three-lane rejected quality selection is invalid")
            if rejected_quality_transaction is None:
                raise RuntimeError(
                    "native three-lane selected quality has no transaction"
                )
            quality_routes = rejected_quality_plans[selected_quality]
            quality_objective = replay_objective(quality_routes)
            quality_key = reported_objective(
                rejected_quality_transaction,
                selected_quality,
                quality_objective,
                "rejected quality",
            )
            source = rejected_quality_sources[selected_quality]
            moved = int(moved_customers[source])
            rejected_events.append(
                event(
                    "relocate",
                    "candidate_proposed",
                    "relocate_candidate",
                    route_indices=tuple(int(v) for v in changed_routes[source]),
                    affected_route_indices=tuple(
                        int(v) for v in changed_routes[source]
                    ),
                    removed_customers=(name_by_index[moved],),
                    candidate_route_sequences=tuple(
                        quality_routes[index]
                        for index in changed_routes[source]
                    ),
                    candidate_vehicle_delta=len(quality_routes)
                    - len(initial_customer_sequences),
                    candidate_feasible=True,
                    prefilter_passed=True,
                    accepted=bool(quality_outcome[1]),
                    vehicle_reduction=bool(quality_outcome[3]),
                    distance_improvement=quality_objective.total_distance
                    < initial_objective.total_distance - 1e-9,
                    candidate_objective_key=quality_key,
                )
            )
            rejected_events[0]["candidate_objective_key"] = quality_key

        constraint = require_tuple(payload[3], 3, "rejected constraint")
        selection = cast(
            npt.NDArray[np.int64],
            _require_array(
                constraint[0],
                dtype=np.dtype(np.int64),
                shape=(7,),
                name="rejected constraint selection",
            ),
        )
        constraint_probe = require_tuple(
            constraint[1], 3, "rejected constraint probe"
        )
        removal = require_tuple(
            constraint_probe[0], 7, "rejected constraint removal"
        )
        removed_indices = _require_vector(
            removal[2], "rejected constraint removed customers"
        )
        removed_customers = tuple(name_by_index[int(index)] for index in removed_indices)
        remaining_routes = unpack_routes(
            removal[0], removal[1], "rejected constraint partial"
        )
        repair = require_tuple(constraint_probe[1], 3, "rejected constraint repair")
        repaired_routes = unpack_routes(
            repair[0], repair[1], "rejected constraint repaired"
        )
        constraint_transaction = require_plan_transaction(
            constraint_probe[2], 1, "rejected constraint"
        )
        constraint_objective = replay_objective(repaired_routes)
        constraint_key = reported_objective(
            constraint_transaction, 0, constraint_objective, "rejected constraint"
        )
        constraint_outcome = cast(
            npt.NDArray[np.int64],
            _require_array(
                constraint[2],
                dtype=np.dtype(np.int64),
                shape=(6,),
                name="rejected constraint outcome",
            ),
        )
        operator = FULL_NATIVE_OPERATOR_NAMES[int(constraint_outcome[0]) + 9]
        removal_scores, removal_route_indices = _constraint_score_vectors(
            removal, removed_indices, "rejected constraint removal"
        )
        if len(removal_route_indices) == 0:
            raise RuntimeError(
                "native three-lane rejected constraint route identity is missing"
            )
        affected = tuple(
            index
            for index, (before, after) in enumerate(
                zip(initial_customer_sequences, repaired_routes, strict=False)
            )
            if before != after
        )
        candidate_ready = bool(constraint_outcome[2])
        constraint_statuses = _require_vector(
            constraint_transaction[1], "rejected constraint statuses"
        )
        constraint_exact_rows = _require_vector(
            constraint_transaction[5], "rejected constraint exact rows"
        )
        if len(constraint_statuses) != 1:
            raise RuntimeError(
                "native three-lane rejected constraint status count is invalid"
            )
        no_change = (
            not candidate_ready
            and int(constraint_statuses[0]) == 5
            and repaired_routes == initial_customer_sequences
        )
        if not candidate_ready and not no_change:
            raise RuntimeError(
                "native three-lane rejected constraint lost a non-incumbent candidate"
            )
        ranked_key: tuple[int, float, float, int] | tuple[()] = (
            () if no_change else constraint_key
        )
        rejected_events.append(
            event(
                operator,
                "candidate_proposed",
                "constraint_ranked_removal",
                route_indices=(int(removal_route_indices[0]),),
                affected_route_indices=(int(removal_route_indices[0]),),
                removed_customers=removed_customers,
                candidate_route_sequences=remaining_routes,
                prefilter_passed=True,
                selection_rank=1,
                ranking_score=float(removal_scores[0]),
                candidate_objective_key=ranked_key,
                track="constraint_lane",
                constraint_category=operator,
                removal_tier="small" if int(selection[0]) == 0 else "",
                removal_size_requested=int(selection[1]),
                removal_size_actual=int(selection[2]),
                removal_trigger="stagnation_baseline",
            )
        )
        if no_change:
            if len(constraint_exact_rows) != 0:
                raise RuntimeError(
                    "native three-lane rejected no-change constraint started exact work"
                )
            rejected_events.append(
                event(
                    operator,
                    "failed",
                    "constraint_removal_no_change",
                    removed_customers=removed_customers,
                    prefilter_passed=True,
                    track="constraint_lane",
                    constraint_category=operator,
                    removal_tier="small" if int(selection[0]) == 0 else "",
                    removal_size_requested=int(selection[1]),
                    removal_size_actual=int(selection[2]),
                    removal_trigger="stagnation_baseline",
                )
            )
        else:
            rejected_events.append(
                event(
                    operator,
                    "candidate_proposed",
                    "constraint_removal_repaired",
                    affected_route_indices=affected,
                    removed_customers=removed_customers,
                    candidate_route_sequences=repaired_routes,
                    candidate_vehicle_delta=len(repaired_routes)
                    - len(initial_customer_sequences),
                    candidate_feasible=True,
                    prefilter_passed=True,
                    exact_route_evaluations=len(constraint_exact_rows),
                    accepted=bool(constraint_outcome[3]),
                    vehicle_reduction=bool(constraint_outcome[5]),
                    distance_improvement=constraint_objective.total_distance
                    < initial_objective.total_distance - 1e-9,
                    candidate_objective_key=constraint_key,
                    track="constraint_lane",
                    constraint_category=operator,
                    removal_tier="small" if int(selection[0]) == 0 else "",
                    removal_size_requested=int(selection[1]),
                    removal_size_actual=int(selection[2]),
                    removal_trigger="stagnation_baseline",
                )
            )

        profile_order = _require_vector(legacy[0], "rejected legacy profile order")
        attempts = cast(
            npt.NDArray[np.int64],
            _require_array(
                legacy[1],
                dtype=np.dtype(np.int64),
                shape=(len(profile_order), 6),
                name="rejected legacy attempts",
            ),
        )
        if not np.array_equal(profile_order, attempts[:, 0]):
            raise RuntimeError("native three-lane rejected legacy order is invalid")
        for attempt in attempts:
            source = int(attempt[0])
            failure_code = int(attempt[2])
            if failure_code == 0:
                raise RuntimeError(
                    "native three-lane rejected legacy lost a feasible plan"
                )
            rejected_events.append(
                event(
                    "route_elimination",
                    "failed",
                    "no_existing_route_insertion"
                    if failure_code == 1
                    else "singleton_route_screening_rejected",
                    route_indices=(source,),
                    removed_customers=initial_customer_sequences[source],
                    selection_rank=int(attempt[1]),
                )
            )
        if len(initial_customer_sequences) == 1 and len(profile_order) == 0:
            rejected_events.append(
                event(
                    "route_elimination",
                    "not_applicable",
                    "only_one_route",
                )
            )
        else:
            rejected_events.append(
                event(
                    "route_elimination",
                    "candidate_control_skipped",
                    "no_selected_complete_plan_feasible",
                )
            )
        best = require_tuple(payload[9], 5, "rejected best")
        best_routes = unpack_routes(best[0], best[1], "rejected best")
        if best_routes != initial_customer_sequences:
            raise RuntimeError("native three-lane rejected bootstrap changed global best")
        activity = activity_from_events(rejected_events)
        return NativeThreeLaneSemanticStream(
            neighborhood_events=tuple(rejected_events),
            operator_weights=_readonly_copy(weights),
            operator_rewards=_readonly_copy(rewards),
            operator_calls=_readonly_copy(calls),
            operator_totals=_readonly_copy(totals),
            operator_activity=_readonly_copy(activity),
            termination=_readonly_copy(termination),
            transaction_sha256=producer_sha256,
            initial_temperature=initial_temperature,
        )

    legacy_transaction = require_plan_transaction(
        legacy[5], legacy_plan_count, "legacy"
    )
    if (
        selected_legacy >= legacy_plan_count
        or selected_source < 0
        or selected_source >= len(initial_customer_sequences)
    ):
        raise RuntimeError("native three-lane legacy selection is invalid")
    first_legacy_route = int(legacy_plan_offsets[selected_legacy])
    last_legacy_route = int(legacy_plan_offsets[selected_legacy + 1])
    selected_legacy_offsets = legacy_route_offsets[
        first_legacy_route : last_legacy_route + 1
    ] - int(legacy_route_offsets[first_legacy_route])
    selected_legacy_indices = legacy_route_indices[
        int(legacy_route_offsets[first_legacy_route]) : int(
            legacy_route_offsets[last_legacy_route]
        )
    ]
    legacy_routes = unpack_routes(
        np.ascontiguousarray(selected_legacy_offsets, dtype=np.int64),
        np.ascontiguousarray(selected_legacy_indices, dtype=np.int64),
        "selected legacy plan",
    )
    legacy_objective = replay_objective(legacy_routes)
    legacy_objective_key = reported_objective(
        legacy_transaction, selected_legacy, legacy_objective, "legacy"
    )

    refinement = require_tuple(payload[1], 4, "refinement")
    refinement_metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            refinement[0],
            dtype=np.dtype(np.int64),
            shape=(4,),
            name="refinement metadata",
        ),
    )
    refinement_removed_indices = _require_vector(
        refinement[1], "refinement removed customers"
    )
    if np.any(refinement_removed_indices < 0) or np.any(
        refinement_removed_indices >= len(name_by_index)
    ):
        raise RuntimeError("native three-lane refinement removed customer is invalid")
    refinement_removed = tuple(
        name_by_index[int(index)] for index in refinement_removed_indices
    )
    refinement_status, refinement_reason, refinement_selected = (
        _native_refinement_outcome(refinement_metadata)
    )
    refinement_partial = tuple(
        tuple(customer for customer in route if customer not in refinement_removed)
        for route in legacy_routes
    )
    refinement_partial = tuple(route for route in refinement_partial if route)
    refinement_routes = (
        unpack_routes(
            require_tuple(refinement[2], 3, "refinement plan")[1],
            require_tuple(refinement[2], 3, "refinement plan")[2],
            "refinement final plan",
        )
        if refinement_selected
        else refinement_partial
    )
    legacy_candidate_routes = refinement_routes if refinement_selected else legacy_routes
    legacy_candidate_objective = (
        replay_objective(legacy_candidate_routes)
        if refinement_selected
        else legacy_objective
    )
    legacy_candidate_objective_key = (
        legacy_candidate_objective.key
        if refinement_selected
        else legacy_objective_key
    )

    quality = require_tuple(payload[2], 4, "quality")
    quality_pool = require_tuple(quality[0], 5, "quality pool")
    changed_routes = cast(
        npt.NDArray[np.int64],
        _require_array(
            quality_pool[0],
            dtype=np.dtype(np.int64),
            shape=(len(cast(npt.NDArray[np.int64], quality_pool[0])), 2),
            name="quality changed routes",
        ),
    )
    pool_count = changed_routes.shape[0]
    change_offsets = _require_vector(quality_pool[1], "quality change offsets")
    change_indices = _require_vector(quality_pool[2], "quality change indices")
    moved_customers = _require_vector(quality_pool[4], "quality moved customers")
    if len(change_offsets) != pool_count * 2 + 1 or len(moved_customers) != pool_count:
        raise RuntimeError("native three-lane quality pool offsets are invalid")
    quality_plans: list[tuple[CustomerSequence, ...]] = []
    quality_sources: list[int] = []
    seen_quality: set[tuple[CustomerSequence, ...]] = set()
    for candidate in range(pool_count):
        routes = list(initial_customer_sequences)
        for changed_ordinal in range(2):
            route = int(changed_routes[candidate, changed_ordinal])
            if route < 0 or route >= len(routes):
                raise RuntimeError("native three-lane quality changed route is invalid")
            begin = int(change_offsets[candidate * 2 + changed_ordinal])
            end = int(change_offsets[candidate * 2 + changed_ordinal + 1])
            routes[route] = tuple(
                name_by_index[int(index)] for index in change_indices[begin:end]
            )
        plan = tuple(routes)
        if plan not in seen_quality:
            seen_quality.add(plan)
            quality_plans.append(plan)
            quality_sources.append(candidate)
    quality_transaction = require_plan_transaction(
        quality[1], len(quality_plans), "quality"
    )
    quality_outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            quality[2], dtype=np.dtype(np.int64), shape=(4,), name="quality outcome"
        ),
    )
    selected_quality = int(quality_outcome[0])
    quality_selected = selected_quality >= 0
    if selected_quality >= len(quality_plans):
        raise RuntimeError(
            "native three-lane quality selection is invalid: "
            f"selected={selected_quality}, canonical_pool={len(quality_plans)}, "
            f"raw_pool={pool_count}, outcome={quality_outcome.tolist()}"
        )
    if not quality_selected:
        if selected_quality != -1 or np.any(quality_outcome[1:] != 0):
            raise RuntimeError("native three-lane skipped quality outcome is invalid")
        quality_routes = initial_customer_sequences
        quality_objective = initial_objective
        normal_quality_key: tuple[int, float, float, int] | tuple[()] = ()
        quality_changed: tuple[int, ...] = ()
        moved_customer_index: int | None = None
    else:
        quality_routes = quality_plans[selected_quality]
        quality_objective = replay_objective(quality_routes)
        normal_quality_key = reported_objective(
            quality_transaction, selected_quality, quality_objective, "quality"
        )
        quality_source = quality_sources[selected_quality]
        quality_changed = tuple(
            int(value) for value in changed_routes[quality_source]
        )
        moved_customer_index = int(moved_customers[quality_source])
        if moved_customer_index < 0 or moved_customer_index >= len(name_by_index):
            raise RuntimeError("native three-lane quality moved customer is invalid")
    pool_hash = hashlib.sha256()
    for plan in quality_plans:
        pool_hash.update(json.dumps(plan, separators=(",", ":")).encode())

    constraint = require_tuple(payload[3], 3, "constraint")
    selection = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[0],
            dtype=np.dtype(np.int64),
            shape=(7,),
            name="constraint selection",
        ),
    )
    constraint_probe = require_tuple(constraint[1], 3, "constraint probe")
    removal = require_tuple(constraint_probe[0], 7, "constraint removal")
    removed_indices = _require_vector(removal[2], "constraint removed customers")
    removed_customers = tuple(name_by_index[int(index)] for index in removed_indices)
    remaining_routes = unpack_routes(removal[0], removal[1], "constraint partial")
    repair = require_tuple(constraint_probe[1], 3, "constraint repair")
    repaired_routes = unpack_routes(repair[0], repair[1], "constraint repaired")
    constraint_transaction = require_plan_transaction(
        constraint_probe[2], 1, "constraint"
    )
    constraint_outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[2],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="constraint outcome",
        ),
    )
    constraint_statuses = _require_vector(
        constraint_transaction[1], "constraint statuses"
    )
    constraint_prepared = bool(constraint_outcome[2])
    constraint_exact_feasible = int(constraint_statuses[0]) == 5
    if constraint_prepared and not constraint_exact_feasible:
        raise RuntimeError("native three-lane constraint candidate state is invalid")
    constraint_objective_key: tuple[int, float, float, int] | tuple[()]
    if constraint_prepared:
        constraint_objective = replay_objective(repaired_routes)
        constraint_objective_key = reported_objective(
            constraint_transaction, 0, constraint_objective, "constraint"
        )
    else:
        constraint_objective = initial_objective
        constraint_objective_key = ()
    removal_scores, removal_route_indices = _constraint_score_vectors(
        removal, removed_indices, "constraint removal"
    )
    if len(removal_route_indices) == 0:
        raise RuntimeError("native three-lane constraint route identity is missing")
    constraint_route_indices = tuple(
        sorted(dict.fromkeys(
            int(value) for value in removal_route_indices[: len(removed_indices)]
        ))
    )
    constraint_affected = tuple(
        index
        for index, (before, after) in enumerate(
            zip(initial_customer_sequences, repaired_routes, strict=False)
        )
        if before != after
    )

    legacy_acceptance = require_tuple(payload[4], 3, "legacy acceptance")
    legacy_accepted = bool(legacy_acceptance[0])
    legacy_best = bool(legacy_acceptance[1])
    legacy_vehicle_reduction = bool(legacy_acceptance[2])
    quality_accepted = bool(quality_outcome[1])
    quality_best = bool(quality_outcome[2])
    constraint_accepted = bool(constraint_outcome[3])
    operator_events: list[dict[str, object]] = [
        event(
            "relocate",
            "candidate_pool_aggregate",
            "relocate_complete_candidate_pool",
            aggregate_count=len(quality_plans),
            candidate_pool_hash=pool_hash.hexdigest(),
            candidate_objective_key=normal_quality_key,
        ),
        event(
            "relocate",
            "candidate_proposed" if quality_selected else "candidate_control_skipped",
            "relocate_candidate" if quality_selected else "no_selected_complete_plan_feasible",
            route_indices=quality_changed,
            affected_route_indices=quality_changed,
            removed_customers=(
                (name_by_index[moved_customer_index],)
                if moved_customer_index is not None
                else ()
            ),
            candidate_route_sequences=(
                tuple(quality_routes[index] for index in quality_changed)
                if quality_selected
                else ()
            ),
            candidate_vehicle_delta=(
                len(quality_routes) - len(initial_customer_sequences)
                if quality_selected
                else None
            ),
            candidate_feasible=quality_selected,
            prefilter_passed=quality_selected,
            accepted=quality_accepted,
            vehicle_reduction=bool(quality_outcome[3]),
            distance_improvement=quality_selected
            and quality_objective.total_distance
            < initial_objective.total_distance - 1e-9,
            candidate_objective_key=normal_quality_key,
        ),
        event(
            FULL_NATIVE_OPERATOR_NAMES[int(constraint_outcome[0]) + 9],
            "candidate_proposed",
            "constraint_ranked_removal",
            route_indices=constraint_route_indices,
            affected_route_indices=constraint_route_indices,
            removed_customers=removed_customers,
            candidate_route_sequences=remaining_routes,
            prefilter_passed=True,
            selection_rank=1,
            track="constraint_lane",
            constraint_category=FULL_NATIVE_OPERATOR_NAMES[
                int(constraint_outcome[0]) + 9
            ],
            removal_tier="small" if int(selection[0]) == 0 else "",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            removal_trigger="stagnation_baseline",
            ranking_score=float(removal_scores[0]),
            candidate_objective_key=constraint_objective_key,
        ),
        event(
            FULL_NATIVE_OPERATOR_NAMES[int(constraint_outcome[0]) + 9],
            "candidate_proposed" if constraint_prepared else "failed",
            (
                "constraint_removal_repaired"
                if constraint_prepared
                else "constraint_removal_no_change"
                if constraint_exact_feasible
                else "constraint_repair_infeasible"
            ),
            affected_route_indices=(
                constraint_affected
                if constraint_prepared or not constraint_exact_feasible
                else ()
            ),
            removed_customers=removed_customers,
            candidate_route_sequences=(
                repaired_routes
                if constraint_prepared or not constraint_exact_feasible
                else ()
            ),
            candidate_vehicle_delta=(
                len(repaired_routes) - len(initial_customer_sequences)
                if constraint_prepared
                else None
            ),
            candidate_feasible=constraint_prepared,
            prefilter_passed=True,
            exact_route_evaluations=(
                len(_require_vector(constraint_transaction[5], "constraint exact rows"))
                if constraint_prepared
                else 0
            ),
            track="constraint_lane",
            constraint_category=FULL_NATIVE_OPERATOR_NAMES[
                int(constraint_outcome[0]) + 9
            ],
            removal_tier="small" if int(selection[0]) == 0 else "",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            removal_trigger="stagnation_baseline",
            accepted=constraint_accepted,
            vehicle_reduction=bool(constraint_outcome[5]),
            distance_improvement=constraint_prepared
            and constraint_objective.total_distance
            < initial_objective.total_distance - 1e-9,
            candidate_objective_key=constraint_objective_key,
        ),
        event(
            "route_elimination",
            "candidate_proposed",
            "route_eliminated",
            route_indices=(selected_source,),
            removed_customers=initial_customer_sequences[selected_source],
            candidate_vehicle_delta=len(legacy_routes) - len(initial_customer_sequences),
            candidate_feasible=True,
            prefilter_passed=True,
            exact_route_evaluations=len(
                _require_vector(legacy_transaction[5], "legacy exact rows")
            ),
            accepted=legacy_accepted,
            vehicle_reduction=legacy_vehicle_reduction,
            distance_improvement=legacy_objective.total_distance
            < initial_objective.total_distance - 1e-9,
            candidate_objective_key=legacy_candidate_objective_key,
        ),
        event(
            "vehicle_reduction_refinement",
            refinement_status,
            refinement_reason,
            affected_route_indices=tuple(
                index
                for index, (before, after) in enumerate(
                    zip(initial_customer_sequences, refinement_routes, strict=False)
                )
                if before != after
            ),
            removed_customers=refinement_removed,
            candidate_route_sequences=refinement_routes,
            candidate_vehicle_delta=(
                len(refinement_routes) - len(initial_customer_sequences)
                if refinement_selected
                else None
            ),
            candidate_feasible=refinement_selected,
            prefilter_passed=bool(refinement_partial),
            exact_route_evaluations=int(refinement_metadata[2]),
            accepted=refinement_selected and legacy_accepted,
            vehicle_reduction=refinement_selected,
            distance_improvement=refinement_selected
            and legacy_candidate_objective.total_distance
            < initial_objective.total_distance - 1e-9,
            candidate_objective_key=legacy_candidate_objective_key,
        ),
    ]
    if refinement_selected:
        operator_events[-1].update(
            {
                "_operator_comparison": "better",
                "_operator_is_global_best": True,
                "_operator_vehicle_reduction": True,
            }
        )
    quality_expected_accept = quality_selected and (
        quality_objective.key <= initial_objective.key
    )
    quality_expected_best = quality_expected_accept and (
        quality_objective.key < initial_objective.key
    )
    global_after_quality = (
        quality_objective if quality_expected_best else initial_objective
    )
    constraint_expected_accept = constraint_prepared and (
        constraint_objective.key <= initial_objective.key
    )
    constraint_expected_best = constraint_expected_accept and (
        constraint_objective.key < global_after_quality.key
    )
    global_after_constraint = (
        constraint_objective if constraint_expected_best else global_after_quality
    )
    legacy_expected_accept = legacy_candidate_objective.key <= initial_objective.key
    legacy_expected_best = legacy_expected_accept and (
        legacy_candidate_objective.key < global_after_constraint.key
    )
    if (
        quality_accepted != quality_expected_accept
        or quality_best != quality_expected_best
        or constraint_accepted != constraint_expected_accept
        or bool(constraint_outcome[4]) != constraint_expected_best
        or legacy_accepted != legacy_expected_accept
        or legacy_best != legacy_expected_best
    ):
        raise RuntimeError("native three-lane acceptance trajectory is invalid")
    expected_best_routes = (
        legacy_candidate_routes
        if legacy_expected_best
        else repaired_routes
        if constraint_expected_best
        else quality_routes
        if quality_expected_best
        else initial_customer_sequences
    )
    best = require_tuple(payload[9], 5, "best")
    if unpack_routes(best[0], best[1], "best") != expected_best_routes:
        raise RuntimeError("native three-lane best trajectory replay is invalid")

    return NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(operator_events),
        operator_weights=_readonly_copy(weights),
        operator_rewards=_readonly_copy(rewards),
        operator_calls=_readonly_copy(calls),
        operator_totals=_readonly_copy(totals),
        operator_activity=_readonly_copy(activity_from_events(operator_events)),
        termination=_readonly_copy(termination),
        transaction_sha256=producer_sha256,
        initial_temperature=initial_temperature,
    )


def _verify_native_three_lane_search_hash(payload: tuple[object, ...]) -> str:
    if len(payload) != 2 or not isinstance(payload[0], tuple):
        raise RuntimeError("native three-lane search payload has an invalid tuple")
    evidence = bytearray(b"stage05.2-native-three-lane-search-stream-v2")
    _append_native_nested_evidence(evidence, payload[0])
    producer_sha256 = payload[1]
    if (
        not isinstance(producer_sha256, str)
        or not _is_sha256(producer_sha256)
        or hashlib.sha256(evidence).hexdigest() != producer_sha256
    ):
        raise RuntimeError("native three-lane search stream SHA-256 mismatch")
    return producer_sha256


def _native_three_lane_legacy_payload_kind(value: object) -> str:
    """Identify a legacy-lane payload from its typed shape, never route length."""

    if not isinstance(value, tuple) or not value:
        return "invalid"
    metadata = value[0]
    if not isinstance(metadata, np.ndarray) or metadata.dtype != np.dtype(np.int64):
        return "invalid"
    if len(value) == 2 and metadata.ndim == 1 and metadata.shape == (4,):
        return "simple_rejection"
    if len(value) == 7:
        if metadata.ndim == 1 and metadata.shape == (4,):
            return "route_merge"
        if metadata.ndim == 1 and metadata.shape == (8,):
            return "weighted"
        return "invalid"
    if len(value) == 8:
        pool = value[1]
        if (
            isinstance(pool, np.ndarray)
            and pool.dtype == np.dtype(np.int64)
            and pool.ndim == 2
            and pool.shape[1] == 6
        ):
            return "route_elimination"
        if metadata.ndim == 1 and metadata.shape == (8,):
            return "weighted"
    return "invalid"


def _decode_native_one_customer_three_lane_search(
    instance: Instance,
    iteration_payloads: tuple[object, ...],
    *,
    node_names: tuple[str, ...],
    initial_customer_sequences: tuple[CustomerSequence, ...],
    search_sha256: str,
) -> NativeThreeLaneSemanticStream:
    """Replay the real three-lane controller on the one-customer boundary."""

    first_payload = cast(tuple[object, ...], iteration_payloads[0])
    first = decode_native_three_lane_semantic_stream(
        instance,
        first_payload,
        node_names=node_names,
        initial_customer_sequences=initial_customer_sequences,
    )
    customer = initial_customer_sequences[0][0]
    objective = SolutionObjective.zero()
    exact = solve_exact_charging(instance, (customer,))
    if not exact.feasible:
        raise RuntimeError("native one-customer incumbent replay is infeasible")
    objective += SolutionObjective.from_route(
        instance,
        exact.route,
        total_distance=exact.distance,
        total_charging_time=exact.charging_time,
    )

    def event(
        operator: str,
        status: str,
        reason: str,
        iteration: int,
        **changes: object,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "operator": operator,
            "status": status,
            "reason": reason,
            "route_indices": (),
            "affected_route_indices": (),
            "removed_customers": (),
            "candidate_customer_sequence": (),
            "candidate_route_sequences": (),
            "candidate_vehicle_delta": None,
            "candidate_feasible": False,
            "prefilter_passed": False,
            "new_routes_created": 0,
            "exact_route_evaluations": 0,
            "selection_rank": 0,
            "chain_depth": 0,
            "segment_length": 0,
            "track": "legacy",
            "constraint_category": "",
            "removal_tier": "",
            "removal_size_requested": 0,
            "removal_size_actual": 0,
            "stagnation_iterations": 0,
            "removal_trigger": "",
            "reset_observed": False,
            "ranking_score": 0.0,
            "iteration": iteration,
            "accepted": False,
            "vehicle_reduction": False,
            "distance_improvement": False,
            "candidate_objective_key": (),
        }
        result.update(changes)
        return result

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native one-customer {name} has an invalid tuple")
        return value

    def verify_unchanged_state(value: object, size: int, name: str) -> None:
        state = require_tuple(value, size, name)
        offsets = _require_vector(state[0], f"{name} offsets")
        indices = _require_vector(state[1], f"{name} indices")
        if not np.array_equal(offsets, np.array([0, 1], dtype=np.int64)) or not np.array_equal(
            indices, np.array([node_names.index(customer)], dtype=np.int64)
        ):
            raise RuntimeError(f"native one-customer {name} changed its route")

    events = list(first.neighborhood_events)
    latest_weights = first.operator_weights
    latest_rewards = first.operator_rewards
    latest_calls = first.operator_calls
    latest_totals = first.operator_totals
    latest_termination = first.termination
    stage04_events = list(first.stage04_events)
    latest_stage04_control = first.stage04_control
    quality_operators = (
        "relocate",
        "swap",
        "two_opt_star",
        "route_segment_destroy",
        "ejection_chain",
    )
    constraint_operators = (
        "station_pressure",
        "time_window_conflict",
        "worst_energy_detour",
        "shaw_related",
    )
    destroy_names = ("random", "worst", "related")
    repair_names = ("greedy", "regret2", "energy")
    expected_stagnation = 1
    for iteration, raw_payload in enumerate(iteration_payloads[1:], start=1):
        payload = require_tuple(raw_payload, 14, f"iteration {iteration}")
        _verify_native_three_lane_semantic_hash(payload)
        for index in range(6, 9):
            verify_unchanged_state(payload[index], 4, f"iteration {iteration} state {index}")
        verify_unchanged_state(payload[9], 5, f"iteration {iteration} best")

        quality = payload[2]
        if iteration < 3:
            quality_payload = require_tuple(quality, 4, "empty quality probe")
            quality_pool = require_tuple(quality_payload[0], 5, "empty quality pool")
            changed = cast(
                npt.NDArray[np.int64],
                _require_array(
                    quality_pool[0],
                    dtype=np.dtype(np.int64),
                    shape=(0, 2),
                    name="one-customer quality changed routes",
                ),
            )
            outcome = _require_vector(quality_payload[2], "one-customer quality outcome")
            if (
                len(changed) != 0
                or quality_payload[1] is not None
                or not np.array_equal(outcome, np.array([-1, 0, 0, 0]))
            ):
                raise RuntimeError("native one-customer quality no-op is invalid")
            events.append(
                event(
                    quality_operators[iteration],
                    "not_applicable",
                    "only_one_route",
                    iteration,
                )
            )
        elif iteration == 3:
            quality_payload = require_tuple(quality, 7, "route segment probe")
            metadata = cast(
                npt.NDArray[np.int64],
                _require_array(
                    quality_payload[0],
                    dtype=np.dtype(np.int64),
                    shape=(0, 8),
                    name="one-customer route segment metadata",
                ),
            )
            outcome = _require_vector(quality_payload[5], "route segment outcome")
            if (
                len(metadata) != 0
                or quality_payload[3] is not None
                or quality_payload[4] is not None
                or not np.array_equal(outcome, np.array([-1, 0, 0, 0]))
            ):
                raise RuntimeError("native one-customer route segment no-op is invalid")
            events.append(
                event(
                    "route_segment_destroy",
                    "failed",
                    "no_route_with_segment_length",
                    iteration,
                )
            )
        elif iteration == 4:
            quality_payload = require_tuple(quality, 4, "ejection chain probe")
            outcome = _require_vector(quality_payload[2], "ejection chain outcome")
            if (
                quality_payload[0] is not None
                or quality_payload[1] is not None
                or not np.array_equal(outcome, np.array([-1, 0, 0, 0]))
            ):
                raise RuntimeError("native one-customer ejection-chain no-op is invalid")
            events.append(
                event("ejection_chain", "not_applicable", "only_one_route", iteration)
            )
        elif quality is not None:
            raise RuntimeError("native one-customer post-warm-up quality probe is invalid")

        if payload[3] is not None:
            constraint = require_tuple(payload[3], 3, "constraint no-op")
            selection = _require_vector(constraint[0], "constraint selection")
            probe = require_tuple(constraint[1], 3, "constraint probe")
            removal = require_tuple(probe[0], 7, "constraint removal")
            removal_metadata = _require_vector(removal[6], "constraint removal metadata")
            outcome = _require_vector(constraint[2], "constraint outcome")
            if len(selection) != 7 or len(outcome) != 6:
                raise RuntimeError("native one-customer constraint shape is invalid")
            operator_id = int(outcome[0])
            if (
                operator_id not in range(4)
                or probe[1] is not None
                or probe[2] is not None
                or not np.array_equal(
                    removal_metadata, np.array([2, -1, 0], dtype=np.int64)
                )
                or np.any(outcome[1:] != 0)
                or int(selection[0]) not in range(3)
                or int(selection[4]) != expected_stagnation
            ):
                raise RuntimeError("native one-customer constraint no-op is invalid")
            operator = constraint_operators[operator_id]
            events.append(
                event(
                    operator,
                    "not_applicable",
                    "no_removable_customer",
                    iteration,
                    track="constraint_lane",
                    constraint_category=operator,
                    removal_tier=("small", "medium", "large")[int(selection[0])],
                    removal_size_requested=int(selection[1]),
                    removal_size_actual=int(selection[2]),
                    stagnation_iterations=int(selection[4]),
                    removal_trigger="no_removable_customer",
                    reset_observed=bool(selection[6]),
                )
            )

        legacy = payload[0]
        if iteration in (1, 3):
            legacy_payload = require_tuple(legacy, 7, "vehicle repair")
            metadata = _require_vector(legacy_payload[0], "vehicle repair metadata")
            removed = _require_vector(legacy_payload[1], "vehicle repair removed")
            acceptance = require_tuple(payload[4], 3, "vehicle repair acceptance")
            transaction = require_tuple(legacy_payload[5], 13, "vehicle repair transaction")
            statuses = _require_vector(transaction[1], "vehicle repair statuses")
            if (
                len(metadata) != 8
                or int(metadata[0]) != iteration
                or int(metadata[7]) != -2
                or not np.array_equal(removed, np.array([node_names.index(customer)]))
                or not np.array_equal(statuses, np.array([5]))
                or acceptance != (1, 0, 0)
            ):
                raise RuntimeError("native one-customer vehicle repair is invalid")
            events.append(
                event(
                    "vehicle_count_aware_repair",
                    "candidate_proposed",
                    "existing_route_repair",
                    iteration,
                    removed_customers=(customer,),
                    _operator_destroy_name=destroy_names[int(metadata[1])],
                    candidate_vehicle_delta=0,
                    candidate_feasible=True,
                    new_routes_created=int(metadata[4]),
                    accepted=True,
                    candidate_objective_key=objective.key,
                )
            )
        elif iteration == 2:
            legacy_payload = require_tuple(legacy, 2, "route merge no-op")
            metadata = _require_vector(legacy_payload[0], "route merge metadata")
            if not np.array_equal(metadata, np.array([2, 3, 1, 0])) or payload[4] is not None:
                raise RuntimeError("native one-customer route merge no-op is invalid")
            events.append(
                event("route_merge", "not_applicable", "only_one_route", iteration)
            )
        elif iteration == 4:
            legacy_payload = require_tuple(legacy, 8, "standard probe")
            metadata = _require_vector(legacy_payload[0], "standard metadata")
            removed = _require_vector(legacy_payload[1], "standard removed")
            transaction = require_tuple(legacy_payload[6], 13, "standard transaction")
            statuses = _require_vector(transaction[1], "standard statuses")
            if (
                len(metadata) != 8
                or int(metadata[0]) != iteration
                or int(metadata[1]) != 0
                or not np.array_equal(removed, np.array([node_names.index(customer)]))
                or not np.array_equal(statuses, np.array([1]))
                or payload[4] is not None
            ):
                raise RuntimeError("native one-customer standard rejection is invalid")
            events.append(
                event(
                    "standard",
                    "proposal",
                    f"{destroy_names[int(metadata[2])]}+{repair_names[int(metadata[3])]}",
                    iteration,
                    removed_customers=(customer,),
                    _operator_destroy_name=destroy_names[int(metadata[2])],
                    _operator_repair_name=repair_names[int(metadata[3])],
                )
            )
        else:
            legacy_kind = _native_three_lane_legacy_payload_kind(legacy)
            if legacy_kind == "simple_rejection":
                legacy_payload = require_tuple(legacy, 2, "simple legacy rejection")
                metadata = _require_vector(
                    legacy_payload[0], "simple legacy rejection metadata"
                )
                if (
                    not np.array_equal(
                        metadata,
                        np.array(
                            [iteration, int(metadata[1]), 1, 0], dtype=np.int64
                        ),
                    )
                    or int(metadata[1]) not in (2, 3)
                    or payload[4] is not None
                ):
                    raise RuntimeError(
                        "native one-customer simple legacy rejection is invalid"
                    )
                operator = (
                    "route_elimination" if int(metadata[1]) == 2 else "route_merge"
                )
                events.append(
                    event(operator, "not_applicable", "only_one_route", iteration)
                )
            elif legacy_kind == "weighted" and isinstance(legacy, tuple):
                if len(legacy) == 7:
                    metadata = _require_vector(legacy[0], "weighted vehicle repair")
                    removed = _require_vector(legacy[1], "weighted vehicle removed")
                    acceptance = require_tuple(payload[4], 3, "weighted acceptance")
                    transaction = require_tuple(
                        legacy[5], 13, "weighted vehicle transaction"
                    )
                    statuses = _require_vector(
                        transaction[1], "weighted vehicle statuses"
                    )
                    if (
                        len(metadata) != 8
                        or int(metadata[0]) != iteration
                        or int(metadata[7]) != -2
                        or not np.array_equal(
                            removed, np.array([node_names.index(customer)])
                        )
                        or not np.array_equal(statuses, np.array([5]))
                        or acceptance != (1, 0, 0)
                    ):
                        raise RuntimeError(
                            "native one-customer weighted vehicle repair is invalid"
                        )
                    events.append(
                        event(
                            "vehicle_count_aware_repair",
                            "candidate_proposed",
                            "existing_route_repair",
                            iteration,
                            removed_customers=(customer,),
                            _operator_destroy_name=destroy_names[int(metadata[1])],
                            candidate_vehicle_delta=0,
                            candidate_feasible=True,
                            new_routes_created=int(metadata[4]),
                            accepted=True,
                            candidate_objective_key=objective.key,
                        )
                    )
                elif len(legacy) == 8:
                    metadata = _require_vector(legacy[0], "weighted standard metadata")
                    removed = _require_vector(legacy[1], "weighted standard removed")
                    transaction = require_tuple(
                        legacy[6], 13, "weighted standard transaction"
                    )
                    statuses = _require_vector(
                        transaction[1], "weighted standard statuses"
                    )
                    if (
                        len(metadata) != 8
                        or int(metadata[0]) != iteration
                        or int(metadata[1]) != 0
                        or not np.array_equal(
                            removed, np.array([node_names.index(customer)])
                        )
                        or not np.array_equal(statuses, np.array([1]))
                        or payload[4] is not None
                    ):
                        raise RuntimeError(
                            "native one-customer weighted standard rejection is invalid"
                        )
                    events.append(
                        event(
                            "standard",
                            "proposal",
                            f"{destroy_names[int(metadata[2])]}+"
                            f"{repair_names[int(metadata[3])]}",
                            iteration,
                            removed_customers=(customer,),
                            _operator_destroy_name=destroy_names[int(metadata[2])],
                            _operator_repair_name=repair_names[int(metadata[3])],
                        )
                    )
                else:
                    raise RuntimeError(
                        "native one-customer weighted legacy tuple is invalid"
                    )
            else:
                raise RuntimeError(
                    "native one-customer post-warm-up legacy probe is invalid"
                )

        stage = require_tuple(payload[12], 4, "Stage 4 state")
        latest_weights = cast(
            npt.NDArray[np.float64],
            _require_array(
                stage[0],
                dtype=np.dtype(np.float64),
                shape=(len(FULL_NATIVE_OPERATOR_NAMES),),
                name="one-customer Stage 4 weights",
            ),
        )
        latest_rewards = cast(
            npt.NDArray[np.float64],
            _require_array(
                stage[1],
                dtype=np.dtype(np.float64),
                shape=(len(FULL_NATIVE_OPERATOR_NAMES),),
                name="one-customer Stage 4 rewards",
            ),
        )
        latest_calls = cast(
            npt.NDArray[np.int64],
            _require_array(
                stage[2],
                dtype=np.dtype(np.int64),
                shape=(len(FULL_NATIVE_OPERATOR_NAMES),),
                name="one-customer Stage 4 calls",
            ),
        )
        latest_totals = cast(
            npt.NDArray[np.int64],
            _require_array(
                stage[3],
                dtype=np.dtype(np.int64),
                shape=(len(FULL_NATIVE_OPERATOR_NAMES), 8),
                name="one-customer Stage 4 totals",
            ),
        )
        latest_termination = cast(
            npt.NDArray[np.int64],
            _require_array(
                payload[11],
                dtype=np.dtype(np.int64),
                shape=(6,),
                name="one-customer termination",
            ),
        )
        if int(latest_termination[0]) != 0 or int(latest_termination[5]) != iteration + 1:
            raise RuntimeError("native one-customer termination is invalid")
        stage_boundary = require_tuple(payload[5], 6, "Stage 4 boundary")
        stage_control = cast(
            npt.NDArray[np.int64],
            _require_array(
                stage_boundary[4],
                dtype=np.dtype(np.int64),
                shape=(7,),
                name="one-customer Stage 4 control",
            ),
        )
        stage_control_float = cast(
            npt.NDArray[np.float64],
            _require_array(
                stage_boundary[5],
                dtype=np.dtype(np.float64),
                shape=(1,),
                name="one-customer Stage 4 control float",
            ),
        )
        if (
            np.any(stage_control < 0)
            or int(stage_control[0]) not in (0, 1)
            or int(stage_control[2]) not in (0, 1)
            or int(stage_control[4]) not in (0, 1)
            or not math.isfinite(float(stage_control_float[0]))
            or float(stage_control_float[0]) < 0.0
        ):
            raise RuntimeError("native one-customer Stage 4 control is invalid")
        if bool(stage_control[0]):
            stage04_events.append(
                {
                    "type": "stage04_reheat",
                    "iteration": iteration,
                    "reheat_count": int(stage_control[1]),
                    "reheat_floor": float(stage_control_float[0]),
                    "stagnation_iterations": int(stage_control[6]),
                }
            )
        if bool(stage_control[2]):
            stage04_events.append(
                {
                    "type": "stage04_restart",
                    "iteration": iteration,
                    "restart_count": int(stage_control[3]),
                    "intensification": bool(stage_control[4]),
                    "stagnation_at_trigger": 0,
                }
            )
        if (
            latest_stage04_control is not None
            and bool(latest_stage04_control[4])
            and not bool(stage_control[4])
        ):
            stage04_events.append(
                {
                    "type": "stage04_intensification_end",
                    "iteration": iteration,
                }
            )
        latest_stage04_control = _readonly_copy(stage_control)
        expected_stagnation = 0 if int(stage_control[2]) else expected_stagnation + 1

    activity = np.zeros((len(FULL_NATIVE_OPERATOR_NAMES), 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    feasible_repairs: set[tuple[int, int]] = set()
    for item in events:
        index = by_name[str(item["operator"])]
        if bool(item["candidate_feasible"]):
            feasible_repairs.add((index, cast(int, item["iteration"])))
        activity[index, 1] += _semantic_event_aggregate(item, "prefilter_passed")
        activity[index, 2] += cast(int, item["exact_route_evaluations"])
        activity[index, 3] += int(item["status"] == "candidate_proposed")
        activity[index, 4] += _semantic_event_aggregate(item, "candidate_feasible")
        activity[index, 5] += int(bool(item["vehicle_reduction"]))
        activity[index, 6] += int(bool(item["distance_improvement"]))
    for index, _ in feasible_repairs:
        activity[index, 0] += 1
    return NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(events),
        operator_weights=_readonly_copy(latest_weights),
        operator_rewards=_readonly_copy(latest_rewards),
        operator_calls=_readonly_copy(latest_calls),
        operator_totals=_readonly_copy(latest_totals),
        operator_activity=_readonly_copy(activity),
        termination=_readonly_copy(latest_termination),
        transaction_sha256=search_sha256,
        stage04_events=tuple(stage04_events),
        stage04_control=latest_stage04_control,
        initial_temperature=first.initial_temperature,
        operator_replay_events=tuple(events),
    )


def decode_native_three_lane_search_semantic_stream(
    instance: Instance,
    payload: object,
    *,
    node_names: tuple[str, ...],
    initial_customer_sequences: tuple[CustomerSequence, ...],
    vehicle_operator_config: VehicleOperatorConfig,
) -> NativeThreeLaneSemanticStream:
    """Replay the two-iteration v2 search envelope event by event."""

    if not isinstance(payload, tuple):
        raise RuntimeError("native three-lane search payload has an invalid type")
    producer_sha256 = _verify_native_three_lane_search_hash(payload)
    iteration_payloads = cast(tuple[object, ...], payload[0])
    if len(iteration_payloads) < 2:
        raise RuntimeError("native three-lane search iteration count is invalid")
    if (
        len(initial_customer_sequences) == 1
        and len(initial_customer_sequences[0]) == 1
        and len(instance.customers) == 1
    ):
        return _decode_native_one_customer_three_lane_search(
            instance,
            iteration_payloads,
            node_names=node_names,
            initial_customer_sequences=initial_customer_sequences,
            search_sha256=producer_sha256,
        )
    first_payload = cast(tuple[object, ...], iteration_payloads[0])
    second_payload = cast(tuple[object, ...], iteration_payloads[1])
    first = decode_native_three_lane_semantic_stream(
        instance,
        first_payload,
        node_names=node_names,
        initial_customer_sequences=initial_customer_sequences,
    )
    _verify_native_three_lane_semantic_hash(second_payload)
    if tuple(node.name for node in instance.nodes) != node_names:
        raise RuntimeError("native three-lane search node identity is invalid")
    if len(second_payload) != 14:
        raise RuntimeError("native three-lane follow-up has an invalid tuple")

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native three-lane follow-up {name} is invalid")
        return value

    def unpack_routes(
        offsets_value: object,
        indices_value: object,
        name: str,
    ) -> tuple[CustomerSequence, ...]:
        offsets = _require_vector(offsets_value, f"{name} offsets")
        indices = _require_vector(indices_value, f"{name} indices")
        if (
            len(offsets) < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or np.any(offsets[:-1] >= offsets[1:])
            or np.any(indices < 0)
            or np.any(indices >= len(node_names))
        ):
            raise RuntimeError(f"native three-lane follow-up {name} SoA is invalid")
        routes = tuple(
            tuple(
                node_names[int(index)]
                for index in indices[
                    int(offsets[route]) : int(offsets[route + 1])
                ]
            )
            for route in range(len(offsets) - 1)
        )
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(set(flattened)):
            raise RuntimeError(f"native three-lane follow-up {name} repeats customers")
        return routes

    expected_customers = {customer.name for customer in instance.customers}

    def replay_objective(routes: tuple[CustomerSequence, ...]) -> SolutionObjective:
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(expected_customers) or set(flattened) != expected_customers:
            raise RuntimeError("native three-lane follow-up customer identity is invalid")
        objective = SolutionObjective.zero()
        for route in routes:
            exact = solve_exact_charging(instance, route)
            if not exact.feasible:
                raise RuntimeError("native three-lane follow-up replay is infeasible")
            objective += SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
        return objective

    def transaction(value: object, count: int, name: str) -> tuple[object, ...]:
        result = require_tuple(value, 13, f"{name} transaction")
        _require_array(
            result[1],
            dtype=np.dtype(np.int64),
            shape=(count,),
            name=f"{name} statuses",
        )
        _require_array(
            result[2],
            dtype=np.dtype(np.int64),
            shape=(count, 2),
            name=f"{name} objective integers",
        )
        _require_array(
            result[3],
            dtype=np.dtype(np.float64),
            shape=(count, 2),
            name=f"{name} objective floats",
        )
        _require_vector(result[5], f"{name} exact rows")
        _require_vector(result[11], f"{name} feasible order")
        if not isinstance(result[12], str) or not _is_sha256(result[12]):
            raise RuntimeError(f"native three-lane follow-up {name} hash is invalid")
        return result

    def objective_key(
        value: tuple[object, ...],
        plan: int,
        replayed: SolutionObjective,
        name: str,
    ) -> tuple[int, float, float, int]:
        integers = cast(npt.NDArray[np.int64], value[2])
        floats = cast(npt.NDArray[np.float64], value[3])
        reported = SolutionObjective(
            vehicle_count=int(integers[plan, 0]),
            total_distance=float(floats[plan, 0]),
            total_charging_time=float(floats[plan, 1]),
            charging_count=int(integers[plan, 1]),
        )
        if reported.key != replayed.key:
            raise RuntimeError(
                f"native three-lane follow-up {name} objective mismatch"
            )
        return reported.key

    def event(operator: str, status: str, reason: str, **changes: object) -> dict[str, object]:
        result: dict[str, object] = {
            "operator": operator,
            "status": status,
            "reason": reason,
            "route_indices": (),
            "affected_route_indices": (),
            "removed_customers": (),
            "candidate_customer_sequence": (),
            "candidate_route_sequences": (),
            "candidate_vehicle_delta": None,
            "candidate_feasible": False,
            "prefilter_passed": False,
            "new_routes_created": 0,
            "exact_route_evaluations": 0,
            "selection_rank": 0,
            "chain_depth": 0,
            "segment_length": 0,
            "track": "legacy",
            "constraint_category": "",
            "removal_tier": "",
            "removal_size_requested": 0,
            "removal_size_actual": 0,
            "stagnation_iterations": 0,
            "removal_trigger": "",
            "reset_observed": False,
            "ranking_score": 0.0,
            "iteration": 1,
            "accepted": False,
            "vehicle_reduction": False,
            "distance_improvement": False,
            "candidate_objective_key": (),
        }
        result.update(changes)
        return result

    prior_state_payloads = (
        *(require_tuple(first_payload[index], 4, f"prior state {index}") for index in range(6, 9)),
        require_tuple(first_payload[9], 5, "prior state 9"),
    )
    prior_legacy, prior_quality, prior_constraint, prior_best = tuple(
        unpack_routes(state[0], state[1], f"prior state {index}")
        for index, state in enumerate(prior_state_payloads, start=6)
    )
    prior_legacy_objective = replay_objective(prior_legacy)
    prior_quality_objective = replay_objective(prior_quality)
    prior_constraint_objective = replay_objective(prior_constraint)
    prior_best_objective = replay_objective(prior_best)

    quality = require_tuple(second_payload[2], 4, "quality")
    quality_pool = require_tuple(quality[0], 5, "quality pool")
    changed = cast(
        npt.NDArray[np.int64],
        _require_array(
            quality_pool[0],
            dtype=np.dtype(np.int64),
            shape=(len(cast(npt.NDArray[np.int64], quality_pool[0])), 2),
            name="follow-up quality changed routes",
        ),
    )
    pool_count = changed.shape[0]
    change_offsets = _require_vector(quality_pool[1], "follow-up quality offsets")
    change_indices = _require_vector(quality_pool[2], "follow-up quality indices")
    moved = _require_vector(quality_pool[4], "follow-up quality moved customers")
    if len(change_offsets) != pool_count * 2 + 1 or len(moved) != pool_count * 2:
        raise RuntimeError("native three-lane follow-up swap pool is invalid")
    quality_plans: list[tuple[CustomerSequence, ...]] = []
    quality_sources: list[int] = []
    seen_quality: set[tuple[CustomerSequence, ...]] = set()
    for candidate in range(pool_count):
        routes = list(prior_quality)
        for changed_ordinal in range(2):
            route = int(changed[candidate, changed_ordinal])
            begin = int(change_offsets[candidate * 2 + changed_ordinal])
            end = int(change_offsets[candidate * 2 + changed_ordinal + 1])
            routes[route] = tuple(
                node_names[int(index)] for index in change_indices[begin:end]
            )
        plan = tuple(routes)
        if plan not in seen_quality:
            seen_quality.add(plan)
            quality_plans.append(plan)
            quality_sources.append(candidate)
    quality_transaction = transaction(quality[1], len(quality_plans), "quality")
    quality_outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            quality[2],
            dtype=np.dtype(np.int64),
            shape=(4,),
            name="follow-up quality outcome",
        ),
    )
    selected_quality = int(quality_outcome[0])
    quality_selected = selected_quality >= 0
    if selected_quality >= len(quality_plans):
        raise RuntimeError("native three-lane follow-up quality selection is invalid")
    if not quality_selected:
        if selected_quality != -1 or np.any(quality_outcome[1:] != 0):
            raise RuntimeError(
                "native three-lane follow-up skipped quality outcome is invalid"
            )
        quality_routes = prior_quality
        quality_objective = prior_quality_objective
        quality_key: tuple[int, float, float, int] | tuple[()] = ()
        quality_source: int | None = None
        quality_moved: tuple[str, ...] = ()
    else:
        quality_routes = quality_plans[selected_quality]
        quality_objective = replay_objective(quality_routes)
        quality_key = objective_key(
            quality_transaction, selected_quality, quality_objective, "quality"
        )
        quality_source = quality_sources[selected_quality]
        quality_moved = tuple(
            node_names[int(value)]
            for value in moved[quality_source * 2 : quality_source * 2 + 2]
        )
    pool_hash = hashlib.sha256()
    for plan in quality_plans:
        pool_hash.update(json.dumps(plan, separators=(",", ":")).encode())

    constraint = require_tuple(second_payload[3], 3, "constraint")
    selection = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[0],
            dtype=np.dtype(np.int64),
            shape=(7,),
            name="follow-up constraint selection",
        ),
    )
    constraint_probe = require_tuple(constraint[1], 3, "constraint probe")
    removal = require_tuple(constraint_probe[0], 7, "constraint removal")
    removed_indices = _require_vector(removal[2], "follow-up constraint removed")
    constraint_removed = tuple(node_names[int(index)] for index in removed_indices)
    constraint_partial = unpack_routes(removal[0], removal[1], "constraint partial")
    repair = require_tuple(constraint_probe[1], 3, "constraint repair")
    constraint_routes = unpack_routes(repair[0], repair[1], "constraint repaired")
    constraint_transaction = transaction(constraint_probe[2], 1, "constraint")
    constraint_outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[2],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="follow-up constraint outcome",
        ),
    )
    if int(constraint_outcome[0]) != 1:
        raise RuntimeError("native three-lane follow-up constraint operator is invalid")
    constraint_statuses = _require_vector(
        constraint_transaction[1], "follow-up constraint statuses"
    )
    constraint_prepared = bool(constraint_outcome[2])
    constraint_exact_feasible = int(constraint_statuses[0]) == 5
    if constraint_prepared and not constraint_exact_feasible:
        raise RuntimeError(
            "native three-lane follow-up constraint candidate state is invalid"
        )
    constraint_key: tuple[int, float, float, int] | tuple[()]
    if constraint_prepared:
        constraint_objective = replay_objective(constraint_routes)
        constraint_key = objective_key(
            constraint_transaction, 0, constraint_objective, "constraint"
        )
    else:
        constraint_objective = prior_constraint_objective
        constraint_key = ()
    removal_scores, removal_route_indices = _constraint_score_vectors(
        removal, removed_indices, "follow-up constraint"
    )
    if len(removal_route_indices) == 0:
        raise RuntimeError("native three-lane follow-up constraint route is missing")
    constraint_route_indices = tuple(
        sorted(dict.fromkeys(
            int(value) for value in removal_route_indices[: len(removed_indices)]
        ))
    )

    legacy = require_tuple(second_payload[0], 7, "legacy repair")
    legacy_metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[0],
            dtype=np.dtype(np.int64),
            shape=(8,),
            name="follow-up legacy metadata",
        ),
    )
    if (
        int(legacy_metadata[0]) != 1
        or int(legacy_metadata[1]) not in range(3)
        or int(legacy_metadata[2]) <= 0
        or int(legacy_metadata[3]) != 0
        or int(legacy_metadata[4]) != 0
        or int(legacy_metadata[5]) != 0
        or int(legacy_metadata[6]) < 0
        or int(legacy_metadata[7]) != -2
    ):
        raise RuntimeError("native three-lane follow-up legacy decision is invalid")
    legacy_removed_indices = _require_vector(legacy[1], "follow-up legacy removed")
    legacy_removed = tuple(node_names[int(index)] for index in legacy_removed_indices)
    legacy_repair = require_tuple(legacy[4], 3, "legacy repair result")
    legacy_routes = unpack_routes(legacy_repair[0], legacy_repair[1], "legacy repaired")
    legacy_transaction = transaction(legacy[5], 1, "legacy")
    legacy_objective = replay_objective(legacy_routes)
    legacy_key = objective_key(legacy_transaction, 0, legacy_objective, "legacy")
    legacy_acceptance = require_tuple(second_payload[4], 3, "legacy acceptance")
    quality_accepted = bool(quality_outcome[1])
    constraint_accepted = bool(constraint_outcome[3])
    legacy_accepted = bool(legacy_acceptance[0])

    def valid_acceptance(
        candidate: SolutionObjective,
        incumbent: SolutionObjective,
        accepted: bool,
    ) -> bool:
        if candidate.vehicle_count > incumbent.vehicle_count:
            return not accepted
        if candidate.key <= incumbent.key:
            return accepted
        return candidate.vehicle_count == incumbent.vehicle_count

    quality_best = (
        quality_selected
        and quality_accepted
        and quality_objective.key < prior_best_objective.key
    )
    best_after_quality = quality_objective if quality_best else prior_best_objective
    constraint_best = (
        constraint_prepared
        and
        constraint_accepted and constraint_objective.key < best_after_quality.key
    )
    best_after_constraint = (
        constraint_objective if constraint_best else best_after_quality
    )
    legacy_best = legacy_accepted and legacy_objective.key < best_after_constraint.key
    quality_vehicle_reduction = (
        quality_selected
        and
        quality_accepted
        and quality_objective.vehicle_count < prior_quality_objective.vehicle_count
    )
    constraint_vehicle_reduction = (
        constraint_prepared
        and
        constraint_accepted
        and constraint_objective.vehicle_count
        < prior_constraint_objective.vehicle_count
    )
    legacy_vehicle_reduction = (
        legacy_accepted
        and legacy_objective.vehicle_count < prior_legacy_objective.vehicle_count
    )
    if (
        (
            not quality_selected
            and quality_accepted
        )
        or (
            quality_selected
            and not valid_acceptance(
                quality_objective, prior_quality_objective, quality_accepted
            )
        )
        or bool(quality_outcome[2]) != quality_best
        or bool(quality_outcome[3]) != quality_vehicle_reduction
        or (
            constraint_prepared
            and not valid_acceptance(
                constraint_objective, prior_constraint_objective, constraint_accepted
            )
        )
        or (not constraint_prepared and constraint_accepted)
        or bool(constraint_outcome[4]) != constraint_best
        or bool(constraint_outcome[5]) != constraint_vehicle_reduction
        or not valid_acceptance(
            legacy_objective, prior_legacy_objective, legacy_accepted
        )
        or bool(legacy_acceptance[1]) != legacy_best
        or bool(legacy_acceptance[2]) != legacy_vehicle_reduction
    ):
        raise RuntimeError("native three-lane follow-up acceptance replay mismatch")

    expected_legacy = legacy_routes if legacy_accepted else prior_legacy
    expected_quality = quality_routes if quality_accepted else prior_quality
    expected_constraint = constraint_routes if constraint_accepted else prior_constraint
    expected_best = (
        legacy_routes
        if legacy_best
        else constraint_routes
        if constraint_best
        else quality_routes
        if quality_best
        else prior_best
    )
    final_state_payloads = (
        *(
            require_tuple(second_payload[index], 4, f"final state {index}")
            for index in range(6, 9)
        ),
        require_tuple(second_payload[9], 5, "final state 9"),
    )
    final_states = tuple(
        unpack_routes(state[0], state[1], f"final state {index}")
        for index, state in enumerate(final_state_payloads, start=6)
    )
    if final_states != (
        expected_legacy,
        expected_quality,
        expected_constraint,
        expected_best,
    ):
        raise RuntimeError("native three-lane follow-up lane state replay mismatch")
    expected_best_objective = (
        legacy_objective
        if legacy_best
        else constraint_objective
        if constraint_best
        else quality_objective
        if quality_best
        else prior_best_objective
    )
    if replay_objective(final_states[3]).key != expected_best_objective.key:
        raise RuntimeError("native three-lane follow-up global best is invalid")

    constraint_affected = tuple(
        index
        for index, (before, after) in enumerate(
            zip(prior_constraint, constraint_routes, strict=False)
        )
        if before != after
    )
    quality_changed = (
        tuple(int(value) for value in changed[quality_source])
        if quality_source is not None
        else ()
    )
    followup_events = (
        event(
            "swap",
            "candidate_pool_aggregate",
            "swap_complete_candidate_pool",
            aggregate_count=len(quality_plans),
            candidate_pool_hash=pool_hash.hexdigest(),
            candidate_objective_key=quality_key,
        ),
        event(
            "swap",
            "candidate_proposed" if quality_selected else "candidate_control_skipped",
            "swap_candidate" if quality_selected else "no_selected_complete_plan_feasible",
            route_indices=quality_changed,
            affected_route_indices=quality_changed,
            removed_customers=quality_moved,
            candidate_route_sequences=(
                tuple(quality_routes[index] for index in quality_changed)
                if quality_selected
                else ()
            ),
            candidate_vehicle_delta=(
                len(quality_routes) - len(prior_quality)
                if quality_selected
                else None
            ),
            candidate_feasible=quality_selected,
            prefilter_passed=quality_selected,
            accepted=quality_accepted,
            distance_improvement=quality_selected
            and quality_objective.total_distance
            < prior_quality_objective.total_distance - 1e-9,
            candidate_objective_key=quality_key,
        ),
        event(
            "time_window_conflict",
            "candidate_proposed",
            "constraint_ranked_removal",
            route_indices=constraint_route_indices,
            affected_route_indices=constraint_route_indices,
            removed_customers=constraint_removed,
            candidate_route_sequences=constraint_partial,
            prefilter_passed=True,
            selection_rank=1,
            track="constraint_lane",
            constraint_category="time_window_conflict",
            removal_tier="small",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            stagnation_iterations=int(selection[4]),
            removal_trigger="stagnation_baseline",
            reset_observed=bool(selection[6]),
            ranking_score=float(removal_scores[0]),
            candidate_objective_key=constraint_key,
        ),
        event(
            "time_window_conflict",
            "candidate_proposed" if constraint_prepared else "failed",
            (
                "constraint_removal_repaired"
                if constraint_prepared
                else "constraint_removal_no_change"
                if constraint_exact_feasible
                else "constraint_repair_infeasible"
            ),
            affected_route_indices=(
                constraint_affected
                if constraint_prepared or not constraint_exact_feasible
                else ()
            ),
            removed_customers=constraint_removed,
            candidate_route_sequences=(
                constraint_routes
                if constraint_prepared or not constraint_exact_feasible
                else ()
            ),
            candidate_vehicle_delta=(
                len(constraint_routes) - len(prior_constraint)
                if constraint_prepared
                else None
            ),
            candidate_feasible=constraint_prepared,
            prefilter_passed=True,
            exact_route_evaluations=(
                len(_require_vector(constraint_transaction[5], "constraint exact rows"))
                if constraint_prepared
                else 0
            ),
            track="constraint_lane",
            constraint_category="time_window_conflict",
            removal_tier="small",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            stagnation_iterations=int(selection[4]),
            removal_trigger="stagnation_baseline",
            reset_observed=bool(selection[6]),
            accepted=constraint_accepted,
            distance_improvement=constraint_prepared
            and constraint_objective.total_distance
            < prior_constraint_objective.total_distance - 1e-9,
            candidate_objective_key=constraint_key,
        ),
        event(
            "vehicle_count_aware_repair",
            "candidate_proposed",
            "existing_route_repair",
            removed_customers=legacy_removed,
            _operator_destroy_name=("random", "worst", "related")[
                int(legacy_metadata[1])
            ],
            candidate_vehicle_delta=len(legacy_routes) - len(prior_legacy),
            candidate_feasible=True,
            new_routes_created=int(legacy_metadata[4]),
            exact_route_evaluations=len(
                _require_vector(legacy_transaction[5], "legacy exact rows")
            ),
            accepted=legacy_accepted,
            distance_improvement=legacy_objective.total_distance
            < prior_legacy_objective.total_distance - 1e-9,
            candidate_objective_key=legacy_key,
        ),
    )
    all_events = (*first.neighborhood_events, *followup_events)

    full_stage04 = require_tuple(second_payload[12], 4, "full Stage 4")
    weights = cast(
        npt.NDArray[np.float64],
        _require_array(
            full_stage04[0],
            dtype=np.dtype(np.float64),
            shape=(len(FULL_NATIVE_OPERATOR_NAMES),),
            name="follow-up Stage 4 weights",
        ),
    )
    rewards = cast(
        npt.NDArray[np.float64],
        _require_array(
            full_stage04[1],
            dtype=np.dtype(np.float64),
            shape=(len(FULL_NATIVE_OPERATOR_NAMES),),
            name="follow-up Stage 4 rewards",
        ),
    )
    calls = cast(
        npt.NDArray[np.int64],
        _require_array(
            full_stage04[2],
            dtype=np.dtype(np.int64),
            shape=(len(FULL_NATIVE_OPERATOR_NAMES),),
            name="follow-up Stage 4 calls",
        ),
    )
    totals = cast(
        npt.NDArray[np.int64],
        _require_array(
            full_stage04[3],
            dtype=np.dtype(np.int64),
            shape=(len(FULL_NATIVE_OPERATOR_NAMES), 8),
            name="follow-up Stage 4 totals",
        ),
    )
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            second_payload[11],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="follow-up termination",
        ),
    )
    if (
        int(termination[0]) != 0
        or int(termination[1]) == 0
        or int(termination[2]) != int(termination[3]) + int(termination[4])
        or int(termination[5]) != 2
        or np.any(weights <= 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(rewards < 0.0)
        or np.any(calls < 0)
        or np.any(totals < 0)
    ):
        raise RuntimeError("native three-lane follow-up final state is invalid")
    activity = np.zeros((len(FULL_NATIVE_OPERATOR_NAMES), 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    for item in all_events:
        index = by_name[str(item["operator"])]
        activity[index, 0] = max(
            int(activity[index, 0]), int(bool(item["candidate_feasible"]))
        )
        activity[index, 1] += _semantic_event_aggregate(item, "prefilter_passed")
        activity[index, 2] += cast(int, item["exact_route_evaluations"])
        activity[index, 3] += int(item["status"] == "candidate_proposed")
        activity[index, 4] += _semantic_event_aggregate(item, "candidate_feasible")
        activity[index, 5] += int(bool(item["vehicle_reduction"]))
        activity[index, 6] += int(bool(item["distance_improvement"]))
    refinement_index = FULL_NATIVE_OPERATOR_NAMES.index(
        "vehicle_reduction_refinement"
    )
    activity[refinement_index, 5] = 0
    activity[refinement_index, 0] = first.operator_activity[refinement_index, 0]
    activity[refinement_index, 7] = first.operator_activity[refinement_index, 7]
    second_stream = NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(all_events),
        operator_weights=_readonly_copy(weights),
        operator_rewards=_readonly_copy(rewards),
        operator_calls=_readonly_copy(calls),
        operator_totals=_readonly_copy(totals),
        operator_activity=_readonly_copy(activity),
        termination=_readonly_copy(termination),
        transaction_sha256=producer_sha256,
        initial_temperature=first.initial_temperature,
    )
    if len(iteration_payloads) == 2:
        return second_stream
    third_stream = _decode_native_three_lane_third_iteration(
        instance,
        cast(tuple[object, ...], iteration_payloads[2]),
        previous_payload=second_payload,
        previous_stream=second_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
    )
    if len(iteration_payloads) == 3:
        return third_stream
    fourth_stream = _decode_native_three_lane_fourth_iteration(
        instance,
        cast(tuple[object, ...], iteration_payloads[3]),
        previous_payload=cast(tuple[object, ...], iteration_payloads[2]),
        previous_stream=third_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
    )
    if len(iteration_payloads) == 4:
        return fourth_stream
    fifth_stream = _decode_native_three_lane_fifth_iteration(
        instance,
        cast(tuple[object, ...], iteration_payloads[4]),
        previous_payload=cast(tuple[object, ...], iteration_payloads[3]),
        previous_stream=fourth_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
    )
    if len(iteration_payloads) == 5:
        return fifth_stream
    sixth_payload = cast(tuple[object, ...], iteration_payloads[5])
    sixth_legacy = sixth_payload[0] if sixth_payload else None
    sixth_metadata = (
        _require_vector(sixth_legacy[0], "sixth legacy metadata dispatch")
        if isinstance(sixth_legacy, tuple) and sixth_legacy
        else np.empty(0, dtype=np.int64)
    )
    if _native_three_lane_legacy_payload_kind(sixth_legacy) == "weighted":
        sixth_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            sixth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[4]),
            previous_stream=fifth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=5,
            completed_iterations=6,
        )
    else:
        sixth_operator_id = (
            2
            if isinstance(sixth_legacy, tuple) and len(sixth_legacy) == 8
            else 3
            if isinstance(sixth_legacy, tuple) and len(sixth_legacy) == 7
            else int(sixth_metadata[1])
        )
        sixth_operator = FULL_NATIVE_OPERATOR_NAMES[sixth_operator_id]
        sixth_stream = _decode_native_three_lane_rejection_only_iteration(
            sixth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[4]),
            previous_stream=fifth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=5,
            operator=sixth_operator,
            operator_id=sixth_operator_id,
            completed_iterations=6,
            instance=instance,
        )
    if len(iteration_payloads) == 6:
        return sixth_stream
    seventh_payload = cast(tuple[object, ...], iteration_payloads[6])
    seventh_legacy = seventh_payload[0] if seventh_payload else None
    seventh_metadata = (
        seventh_legacy[0]
        if isinstance(seventh_legacy, tuple) and seventh_legacy
        else None
    )
    if isinstance(seventh_metadata, np.ndarray) and seventh_metadata.shape == (8,):
        seventh_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            seventh_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[5]),
            previous_stream=sixth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=6,
            completed_iterations=7,
            allow_constraint_no_change=seventh_payload[3] is not None,
            vehicle_operator_config=vehicle_operator_config,
        )
    else:
        seventh_stream = _decode_native_three_lane_seventh_iteration(
            instance,
            seventh_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[5]),
            previous_stream=sixth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=6,
            constraint_operator="time_window_conflict",
            constraint_operator_id=1,
            legacy_operator="route_merge",
            legacy_operator_id=3,
            expected_legacy_metadata=(6, 3, 1, 0),
            legacy_reason="only_one_route",
            expected_legacy_statuses=(),
            expected_legacy_exact_rows=(),
            expected_selection=(2, 1, 1, 1, 5, 4, 0),
            removal_trigger="medium_stagnation+periodic_exploration",
            completed_iterations=7,
            vehicle_operator_config=vehicle_operator_config,
        )
    if len(iteration_payloads) == 7:
        return seventh_stream
    eighth_payload = cast(tuple[object, ...], iteration_payloads[7])
    eighth_legacy = eighth_payload[0] if eighth_payload else None
    eighth_metadata = (
        _require_vector(eighth_legacy[0], "eighth legacy metadata")
        if isinstance(eighth_legacy, tuple) and eighth_legacy
        else np.empty(0, dtype=np.int64)
    )
    if _native_three_lane_legacy_payload_kind(eighth_legacy) == "weighted":
        eighth_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            eighth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[6]),
            previous_stream=seventh_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=7,
            completed_iterations=8,
            allow_constraint_no_change=eighth_payload[3] is not None,
            vehicle_operator_config=vehicle_operator_config,
        )
    else:
        eighth_is_merge = (
            isinstance(eighth_metadata, np.ndarray)
            and eighth_metadata.shape == (4,)
            and int(eighth_metadata[1]) == 3
        )
        eighth_stream = _decode_native_three_lane_rejection_only_iteration(
            eighth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[6]),
            previous_stream=seventh_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=7,
            operator="route_merge" if eighth_is_merge else "route_elimination",
            operator_id=3 if eighth_is_merge else 2,
            completed_iterations=8,
            instance=instance if eighth_is_merge else None,
        )
    if len(iteration_payloads) == 8:
        return eighth_stream
    ninth_payload = cast(tuple[object, ...], iteration_payloads[8])
    ninth_legacy = ninth_payload[0] if ninth_payload else None
    ninth_metadata = (
        ninth_legacy[0]
        if isinstance(ninth_legacy, tuple) and ninth_legacy
        else None
    )
    if (
        isinstance(ninth_legacy, tuple)
        and len(ninth_legacy) == 8
        and isinstance(ninth_metadata, np.ndarray)
        and ninth_metadata.shape == (8,)
    ):
        ninth_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            ninth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[7]),
            previous_stream=eighth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=8,
            completed_iterations=9,
            allow_constraint_no_change=ninth_payload[3] is not None,
            vehicle_operator_config=vehicle_operator_config,
        )
    else:
        ninth_is_route_elimination = (
            isinstance(ninth_metadata, np.ndarray)
            and ninth_metadata.shape == (2,)
        )
        ninth_stream = _decode_native_three_lane_rejection_only_iteration(
            ninth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[7]),
            previous_stream=eighth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=8,
            operator=(
                "route_elimination" if ninth_is_route_elimination else "route_merge"
            ),
            operator_id=2 if ninth_is_route_elimination else 3,
            completed_iterations=9,
            instance=None if ninth_is_route_elimination else instance,
        )
    if len(iteration_payloads) == 9:
        return ninth_stream
    tenth_payload = cast(tuple[object, ...], iteration_payloads[9])
    tenth_legacy = tenth_payload[0] if tenth_payload else None
    tenth_metadata = (
        tenth_legacy[0]
        if isinstance(tenth_legacy, tuple) and tenth_legacy
        else None
    )
    if _native_three_lane_legacy_payload_kind(tenth_legacy) == "weighted":
        tenth_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            tenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[8]),
            previous_stream=ninth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=9,
            completed_iterations=10,
            allow_constraint_no_change=True,
            vehicle_operator_config=vehicle_operator_config,
        )
    elif (
        isinstance(tenth_metadata, np.ndarray)
        and tenth_metadata.shape == (4,)
        and tenth_payload[3] is not None
    ):
        tenth_constraint = cast(tuple[object, ...], tenth_payload[3])
        tenth_selection = tuple(
            int(value)
            for value in _require_vector(
                tenth_constraint[0], "tenth constraint selection"
            )
        )
        tenth_outcome = _require_vector(
            tenth_constraint[2], "tenth constraint outcome"
        )
        tenth_constraint_id = int(tenth_outcome[0])
        tenth_legacy_operator_id = int(tenth_metadata[1])
        if tenth_legacy_operator_id not in (2, 3):
            raise RuntimeError("tenth constraint legacy operator is invalid")
        tenth_constraint_names = (
            "station_pressure",
            "time_window_conflict",
            "worst_energy_detour",
            "shaw_related",
        )
        tenth_stream = _decode_native_three_lane_constraint_no_change_iteration(
            instance,
            tenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[8]),
            previous_stream=ninth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=9,
            constraint_operator=tenth_constraint_names[tenth_constraint_id],
            constraint_operator_id=tenth_constraint_id,
            expected_selection=tenth_selection,
            removal_tier=("small", "medium", "large")[tenth_selection[0]],
            removal_trigger=(
                "large_stagnation"
                if tenth_selection[4]
                >= vehicle_operator_config.large_stagnation_threshold
                else "medium_stagnation"
                if tenth_selection[4]
                >= vehicle_operator_config.medium_stagnation_threshold
                else "stagnation_baseline"
            ),
            legacy_operator=(
                "route_merge"
                if tenth_legacy_operator_id == 3
                else "route_elimination"
            ),
            legacy_operator_id=tenth_legacy_operator_id,
            completed_iterations=10,
        )
    else:
        tenth_stream = _decode_native_three_lane_constraint_no_change_iteration(
            instance,
            tenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[8]),
            previous_stream=ninth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=9,
            constraint_operator="shaw_related",
            constraint_operator_id=3,
            expected_selection=(2, 1, 1, 1, 8, 2, 0),
            removal_tier="large",
            removal_trigger="large_stagnation",
            legacy_operator="route_elimination",
            legacy_operator_id=2,
            completed_iterations=10,
        )
    if len(iteration_payloads) == 10:
        return tenth_stream
    eleventh_payload = cast(tuple[object, ...], iteration_payloads[10])
    eleventh_legacy = eleventh_payload[0] if eleventh_payload else None
    eleventh_metadata = (
        eleventh_legacy[0]
        if isinstance(eleventh_legacy, tuple) and eleventh_legacy
        else None
    )
    if (
        isinstance(eleventh_legacy, tuple)
        and isinstance(eleventh_metadata, np.ndarray)
        and eleventh_metadata.shape in {(2,), (4,)}
    ):
        eleventh_is_route_elimination = (
            eleventh_metadata.shape == (2,)
            or (
                eleventh_metadata.shape == (4,)
                and int(eleventh_metadata[1]) == 2
            )
        )
        if eleventh_metadata.shape == (4,) and int(eleventh_metadata[1]) not in (
            2,
            3,
        ):
            raise RuntimeError("eleventh legacy operator is invalid")
        eleventh_stream = _decode_native_three_lane_rejection_only_iteration(
            eleventh_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[9]),
            previous_stream=tenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=10,
            operator=(
                "route_elimination"
                if eleventh_is_route_elimination
                else "route_merge"
            ),
            operator_id=2 if eleventh_is_route_elimination else 3,
            completed_iterations=11,
            instance=None if eleventh_is_route_elimination else instance,
        )
    else:
        eleventh_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            eleventh_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[9]),
            previous_stream=tenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=10,
            completed_iterations=11,
            allow_constraint_no_change=eleventh_payload[3] is not None,
            vehicle_operator_config=vehicle_operator_config,
        )
    if len(iteration_payloads) == 11:
        return eleventh_stream
    twelfth_payload = cast(tuple[object, ...], iteration_payloads[11])
    twelfth_legacy = twelfth_payload[0] if twelfth_payload else None
    twelfth_metadata = (
        twelfth_legacy[0]
        if isinstance(twelfth_legacy, tuple) and twelfth_legacy
        else None
    )
    twelfth_is_vehicle_repair = (
        isinstance(twelfth_metadata, np.ndarray)
        and twelfth_metadata.shape == (8,)
    )
    if twelfth_is_vehicle_repair:
        twelfth_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            twelfth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[10]),
            previous_stream=eleventh_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=11,
            completed_iterations=12,
            vehicle_operator_config=vehicle_operator_config,
        )
    else:
        twelfth_is_merge = (
            isinstance(twelfth_metadata, np.ndarray)
            and twelfth_metadata.shape == (4,)
            and int(twelfth_metadata[1]) == 3
        )
        twelfth_stream = _decode_native_three_lane_rejection_only_iteration(
            twelfth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[10]),
            previous_stream=eleventh_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=11,
            operator="route_merge" if twelfth_is_merge else "route_elimination",
            operator_id=3 if twelfth_is_merge else 2,
            completed_iterations=12,
            instance=instance if twelfth_is_merge else None,
        )
    if len(iteration_payloads) == 12:
        return twelfth_stream
    thirteenth_payload = cast(tuple[object, ...], iteration_payloads[12])
    thirteenth_constraint = cast(tuple[object, ...], thirteenth_payload[3])
    thirteenth_selection = tuple(
        int(value)
        for value in _require_vector(
            thirteenth_constraint[0], "thirteenth constraint selection"
        )
    )
    thirteenth_outcome = _require_vector(
        thirteenth_constraint[2], "thirteenth constraint outcome"
    )
    thirteenth_constraint_id = int(thirteenth_outcome[0])
    thirteenth_legacy = cast(tuple[object, ...], thirteenth_payload[0])
    thirteenth_legacy_metadata = _require_vector(
        thirteenth_legacy[0], "thirteenth legacy metadata"
    )
    thirteenth_is_merge = (
        len(thirteenth_legacy) == 7
        and len(thirteenth_legacy_metadata) == 4
        and int(thirteenth_legacy_metadata[1]) == 3
    )
    thirteenth_stagnation = thirteenth_selection[4]
    thirteenth_baseline_tier = (
        2
        if thirteenth_stagnation
        >= vehicle_operator_config.large_stagnation_threshold
        else 1
        if thirteenth_stagnation
        >= vehicle_operator_config.medium_stagnation_threshold
        else 0
    )
    thirteenth_trigger = (
        "large_stagnation"
        if thirteenth_baseline_tier == 2
        else "medium_stagnation"
        if thirteenth_baseline_tier == 1
        else "stagnation_baseline"
    )
    if thirteenth_selection[0] == thirteenth_baseline_tier + 1:
        thirteenth_trigger = f"{thirteenth_trigger}+periodic_exploration"
    if _native_three_lane_legacy_payload_kind(thirteenth_legacy) == "weighted":
        thirteenth_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            thirteenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[11]),
            previous_stream=twelfth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=12,
            completed_iterations=13,
            allow_constraint_no_change=True,
            vehicle_operator_config=vehicle_operator_config,
        )
    elif bool(thirteenth_outcome[2]):
        thirteenth_legacy_operator_id = int(thirteenth_legacy_metadata[1])
        if thirteenth_legacy_operator_id not in (2, 3):
            raise RuntimeError("thirteenth legacy operator is invalid")
        thirteenth_stream = _decode_native_three_lane_seventh_iteration(
            instance,
            thirteenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[11]),
            previous_stream=twelfth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=12,
            constraint_operator=(
                "station_pressure",
                "time_window_conflict",
                "worst_energy_detour",
                "shaw_related",
            )[thirteenth_constraint_id],
            constraint_operator_id=thirteenth_constraint_id,
            legacy_operator=(
                "route_merge"
                if thirteenth_legacy_operator_id == 3
                else "route_elimination"
            ),
            legacy_operator_id=thirteenth_legacy_operator_id,
            expected_legacy_metadata=tuple(
                int(value) for value in thirteenth_legacy_metadata
            ),
            legacy_reason="only_one_route",
            expected_legacy_statuses=(),
            expected_legacy_exact_rows=(),
            expected_selection=thirteenth_selection,
            removal_trigger=thirteenth_trigger,
            completed_iterations=13,
            vehicle_operator_config=vehicle_operator_config,
        )
    else:
        thirteenth_stream = _decode_native_three_lane_constraint_no_change_iteration(
            instance,
            thirteenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[11]),
            previous_stream=twelfth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=12,
            constraint_operator=(
                "station_pressure",
                "time_window_conflict",
                "worst_energy_detour",
                "shaw_related",
            )[thirteenth_constraint_id],
            constraint_operator_id=thirteenth_constraint_id,
            expected_selection=thirteenth_selection,
            removal_tier=("small", "medium", "large")[thirteenth_selection[0]],
            removal_trigger=thirteenth_trigger,
            legacy_operator=(
                "route_merge" if thirteenth_is_merge else "route_elimination"
            ),
            legacy_operator_id=3 if thirteenth_is_merge else 2,
            completed_iterations=13,
        )
    if len(iteration_payloads) == 13:
        return thirteenth_stream
    fourteenth_payload = cast(tuple[object, ...], iteration_payloads[13])
    fourteenth_legacy = fourteenth_payload[0] if fourteenth_payload else None
    fourteenth_metadata = (
        _require_vector(fourteenth_legacy[0], "fourteenth legacy metadata")
        if isinstance(fourteenth_legacy, tuple) and fourteenth_legacy
        else np.empty(0, dtype=np.int64)
    )
    if _native_three_lane_legacy_payload_kind(fourteenth_legacy) == "weighted":
        fourteenth_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            fourteenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[12]),
            previous_stream=thirteenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=13,
            completed_iterations=14,
            allow_constraint_no_change=fourteenth_payload[3] is not None,
            vehicle_operator_config=vehicle_operator_config,
        )
    else:
        fourteenth_is_merge = int(fourteenth_metadata[1]) == 3
        fourteenth_stream = _decode_native_three_lane_rejection_only_iteration(
            fourteenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[12]),
            previous_stream=thirteenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=13,
            operator="route_merge" if fourteenth_is_merge else "route_elimination",
            operator_id=3 if fourteenth_is_merge else 2,
            completed_iterations=14,
            instance=instance if fourteenth_is_merge else None,
        )
    if len(iteration_payloads) == 14:
        return fourteenth_stream
    fifteenth_payload = cast(tuple[object, ...], iteration_payloads[14])
    fifteenth_legacy = cast(tuple[object, ...], fifteenth_payload[0])
    fifteenth_metadata = _require_vector(
        fifteenth_legacy[0], "fifteenth legacy metadata"
    )
    if _native_three_lane_legacy_payload_kind(fifteenth_legacy) == "weighted":
        fifteenth_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            fifteenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[13]),
            previous_stream=fourteenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=14,
            completed_iterations=15,
            allow_constraint_no_change=fifteenth_payload[3] is not None,
            vehicle_operator_config=vehicle_operator_config,
        )
    else:
        fifteenth_is_merge = int(fifteenth_metadata[1]) == 3
        fifteenth_stream = _decode_native_three_lane_rejection_only_iteration(
            fifteenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[13]),
            previous_stream=fourteenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=14,
            operator="route_merge" if fifteenth_is_merge else "route_elimination",
            operator_id=3 if fifteenth_is_merge else 2,
            completed_iterations=15,
            instance=instance if fifteenth_is_merge else None,
        )
    if len(iteration_payloads) == 15:
        return fifteenth_stream
    sixteenth_payload = cast(tuple[object, ...], iteration_payloads[15])
    sixteenth_legacy = cast(tuple[object, ...], sixteenth_payload[0])
    if _native_three_lane_legacy_payload_kind(sixteenth_legacy) == "weighted":
        sixteenth_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            sixteenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[14]),
            previous_stream=fifteenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=15,
            completed_iterations=16,
            allow_constraint_no_change=True,
            vehicle_operator_config=vehicle_operator_config,
        )
    else:
        sixteenth_stream = _decode_native_three_lane_seventh_iteration(
            instance,
            sixteenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[14]),
            previous_stream=fifteenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=15,
            constraint_operator="time_window_conflict",
            constraint_operator_id=1,
            legacy_operator="route_elimination",
            legacy_operator_id=2,
            expected_legacy_metadata=(15, 2, 1, 0),
            legacy_reason="only_one_route",
            expected_legacy_statuses=(),
            expected_legacy_exact_rows=(),
            expected_selection=(2, 1, 1, 1, 14, 2, 0),
            removal_trigger="large_stagnation",
            completed_iterations=16,
            vehicle_operator_config=vehicle_operator_config,
        )
    if len(iteration_payloads) == 16:
        return sixteenth_stream
    seventeenth_payload = cast(tuple[object, ...], iteration_payloads[16])
    seventeenth_legacy = cast(tuple[object, ...], seventeenth_payload[0])
    seventeenth_metadata = _require_vector(
        seventeenth_legacy[0], "seventeenth legacy metadata"
    )
    if _native_three_lane_legacy_payload_kind(seventeenth_legacy) == "weighted":
        seventeenth_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            seventeenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[15]),
            previous_stream=sixteenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=16,
            completed_iterations=17,
            allow_constraint_no_change=seventeenth_payload[3] is not None,
            vehicle_operator_config=vehicle_operator_config,
        )
    else:
        seventeenth_is_merge = int(seventeenth_metadata[1]) == 3
        seventeenth_stream = _decode_native_three_lane_rejection_only_iteration(
            seventeenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[15]),
            previous_stream=sixteenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=16,
            operator=(
                "route_merge" if seventeenth_is_merge else "route_elimination"
            ),
            operator_id=3 if seventeenth_is_merge else 2,
            completed_iterations=17,
            instance=instance if seventeenth_is_merge else None,
        )
    if len(iteration_payloads) == 17:
        return seventeenth_stream
    eighteenth_payload = cast(tuple[object, ...], iteration_payloads[17])
    eighteenth_legacy = cast(tuple[object, ...], eighteenth_payload[0])
    eighteenth_metadata = _require_vector(
        eighteenth_legacy[0], "eighteenth legacy metadata"
    )
    if _native_three_lane_legacy_payload_kind(eighteenth_legacy) == "weighted":
        eighteenth_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            eighteenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[16]),
            previous_stream=seventeenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=17,
            completed_iterations=18,
            allow_constraint_no_change=eighteenth_payload[3] is not None,
            vehicle_operator_config=vehicle_operator_config,
        )
    else:
        eighteenth_is_merge = int(eighteenth_metadata[1]) == 3
        eighteenth_stream = _decode_native_three_lane_rejection_only_iteration(
            eighteenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[16]),
            previous_stream=seventeenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=17,
            operator=(
                "route_merge" if eighteenth_is_merge else "route_elimination"
            ),
            operator_id=3 if eighteenth_is_merge else 2,
            completed_iterations=18,
            instance=instance if eighteenth_is_merge else None,
        )
    if len(iteration_payloads) == 18:
        return eighteenth_stream
    nineteenth_payload = cast(tuple[object, ...], iteration_payloads[18])
    nineteenth_legacy = cast(tuple[object, ...], nineteenth_payload[0])
    if _native_three_lane_legacy_payload_kind(nineteenth_legacy) == "weighted":
        nineteenth_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            nineteenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[17]),
            previous_stream=eighteenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=18,
            completed_iterations=19,
            allow_constraint_no_change=True,
            vehicle_operator_config=vehicle_operator_config,
        )
    elif len(nineteenth_legacy) == 7:
        nineteenth_constraint = cast(tuple[object, ...], nineteenth_payload[3])
        nineteenth_selection = tuple(
            int(value)
            for value in _require_vector(
                nineteenth_constraint[0], "nineteenth constraint selection"
            )
        )
        nineteenth_outcome = _require_vector(
            nineteenth_constraint[2], "nineteenth constraint outcome"
        )
        nineteenth_constraint_id = int(nineteenth_outcome[0])
        nineteenth_stream = _decode_native_three_lane_constraint_no_change_iteration(
            instance,
            nineteenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[17]),
            previous_stream=eighteenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=18,
            constraint_operator=(
                "station_pressure",
                "time_window_conflict",
                "worst_energy_detour",
                "shaw_related",
            )[nineteenth_constraint_id],
            constraint_operator_id=nineteenth_constraint_id,
            expected_selection=nineteenth_selection,
            removal_tier=("small", "medium", "large")[nineteenth_selection[0]],
            removal_trigger="large_stagnation",
            legacy_operator="route_merge",
            legacy_operator_id=3,
            completed_iterations=19,
        )
    else:
        nineteenth_stream = _decode_native_three_lane_seventh_iteration(
            instance,
            nineteenth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[17]),
            previous_stream=eighteenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=18,
            constraint_operator="worst_energy_detour",
            constraint_operator_id=2,
            legacy_operator="standard",
            legacy_operator_id=0,
            expected_legacy_metadata=(18, 0, 2, 1, 1, 0, -1, 0),
            legacy_reason="related+regret2",
            expected_legacy_statuses=(1, 1, 1, 1, 0),
            expected_legacy_exact_rows=(),
            expected_selection=(2, 1, 1, 1, 17, 2, 0),
            removal_trigger="large_stagnation",
            completed_iterations=19,
            vehicle_operator_config=vehicle_operator_config,
        )
    if len(iteration_payloads) == 19:
        return nineteenth_stream
    twentieth_payload = cast(tuple[object, ...], iteration_payloads[19])
    twentieth_legacy = cast(tuple[object, ...], twentieth_payload[0])
    twentieth_metadata = _require_vector(
        twentieth_legacy[0], "twentieth legacy metadata"
    )
    if _native_three_lane_legacy_payload_kind(twentieth_legacy) == "weighted":
        twentieth_stream = _decode_native_three_lane_sixth_iteration(
            instance,
            twentieth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[18]),
            previous_stream=nineteenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=19,
            completed_iterations=20,
            allow_constraint_no_change=twentieth_payload[3] is not None,
            vehicle_operator_config=vehicle_operator_config,
        )
    else:
        twentieth_is_merge = int(twentieth_metadata[1]) == 3
        twentieth_stream = _decode_native_three_lane_rejection_only_iteration(
            twentieth_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[18]),
            previous_stream=nineteenth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=19,
            operator="route_merge" if twentieth_is_merge else "route_elimination",
            operator_id=3 if twentieth_is_merge else 2,
            completed_iterations=20,
            instance=instance if twentieth_is_merge else None,
        )
    if len(iteration_payloads) == 20:
        return twentieth_stream
    twenty_first_payload = cast(tuple[object, ...], iteration_payloads[20])
    twenty_first_legacy = cast(tuple[object, ...], twenty_first_payload[0])
    twenty_first_metadata = _require_vector(
        twenty_first_legacy[0], "twenty-first legacy metadata"
    )
    if _native_three_lane_legacy_payload_kind(twenty_first_legacy) == "weighted":
        stream = _decode_native_three_lane_sixth_iteration(
            instance,
            twenty_first_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[19]),
            previous_stream=twentieth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=20,
            completed_iterations=21,
            allow_constraint_no_change=twenty_first_payload[3] is not None,
            vehicle_operator_config=vehicle_operator_config,
        )
    else:
        twenty_first_is_merge = int(twenty_first_metadata[1]) == 3
        twenty_first_boundary = cast(tuple[object, ...], twenty_first_payload[5])
        twenty_first_control = _require_vector(
            twenty_first_boundary[4], "twenty-first Stage 4 control"
        )
        stream = _decode_native_three_lane_rejection_only_iteration(
            twenty_first_payload,
            previous_payload=cast(tuple[object, ...], iteration_payloads[19]),
            previous_stream=twentieth_stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=20,
            operator=(
                "route_merge" if twenty_first_is_merge else "route_elimination"
            ),
            operator_id=3 if twenty_first_is_merge else 2,
            completed_iterations=21,
            restart_to_best=bool(twenty_first_control[2]),
            instance=instance if twenty_first_is_merge else None,
        )
    for iteration in range(21, len(iteration_payloads)):
        current_payload = cast(tuple[object, ...], iteration_payloads[iteration])
        generic_legacy = current_payload[0] if len(current_payload) == 14 else None
        if not isinstance(generic_legacy, tuple) or len(generic_legacy) not in (2, 7, 8):
            raise RuntimeError(
                "native three-lane generic follow-up operator is not implemented"
            )
        metadata = _require_vector(generic_legacy[0], "generic legacy metadata")
        legacy_kind = _native_three_lane_legacy_payload_kind(generic_legacy)
        aggregate_merge = legacy_kind == "route_merge"
        full_route_elimination = legacy_kind == "route_elimination"
        simple_rejection = legacy_kind == "simple_rejection"
        rejection_like = (
            aggregate_merge or full_route_elimination or simple_rejection
        )
        if legacy_kind == "invalid":
            raise RuntimeError(
                "native three-lane generic follow-up payload schema is invalid"
            )
        if rejection_like:
            if aggregate_merge:
                operator = "route_merge"
                operator_id = 3
            elif full_route_elimination:
                operator = "route_elimination"
                operator_id = 2
            else:
                operator_id = int(metadata[1])
                if operator_id not in (2, 3):
                    raise RuntimeError(
                        "native three-lane generic rejection operator is invalid"
                    )
                operator = (
                    "route_elimination" if operator_id == 2 else "route_merge"
                )
            current_termination = _require_vector(
                current_payload[11], "generic rejection termination"
            )
            current_termination_reason = int(current_termination[0])
            if current_termination_reason not in (0, 3) or (
                current_termination_reason != 0
                and iteration != len(iteration_payloads) - 1
            ):
                raise RuntimeError(
                    "native three-lane generic rejection boundary is invalid"
                )
            if current_payload[3] is not None:
                constraint = cast(tuple[object, ...], current_payload[3])
                selection = _require_vector(
                    constraint[0], "generic constraint selection"
                )
                outcome = _require_vector(
                    constraint[2], "generic constraint outcome"
                )
                if len(selection) != 7 or len(outcome) != 6:
                    raise RuntimeError(
                        "native three-lane generic constraint payload is invalid"
                    )
                operator_names = (
                    "station_pressure",
                    "time_window_conflict",
                    "worst_energy_detour",
                    "shaw_related",
                )
                constraint_operator_id = int(outcome[0])
                tier_names = ("small", "medium", "large")
                tier_index = int(selection[0])
                if constraint_operator_id not in range(4) or tier_index not in range(3):
                    raise RuntimeError(
                        "native three-lane generic constraint identity is invalid"
                    )
                stagnation = int(selection[4])
                if stagnation >= vehicle_operator_config.large_stagnation_threshold:
                    baseline_tier = 2
                    trigger = "large_stagnation"
                elif stagnation >= vehicle_operator_config.medium_stagnation_threshold:
                    baseline_tier = 1
                    trigger = "medium_stagnation"
                else:
                    baseline_tier = 0
                    trigger = "stagnation_baseline"
                if tier_index == baseline_tier + 1:
                    trigger = f"{trigger}+periodic_exploration"
                elif tier_index != baseline_tier:
                    raise RuntimeError(
                        "native three-lane generic constraint tier is invalid"
                    )
                stream = _decode_native_three_lane_constraint_no_change_iteration(
                    instance,
                    current_payload,
                    previous_payload=cast(
                        tuple[object, ...], iteration_payloads[iteration - 1]
                    ),
                    previous_stream=stream,
                    node_names=node_names,
                    search_sha256=producer_sha256,
                    iteration=iteration,
                    constraint_operator=operator_names[constraint_operator_id],
                    constraint_operator_id=constraint_operator_id,
                    expected_selection=tuple(int(value) for value in selection),
                    removal_tier=tier_names[tier_index],
                    removal_trigger=trigger,
                    legacy_operator=operator,
                    legacy_operator_id=operator_id,
                    completed_iterations=iteration + 1,
                    termination_reason=current_termination_reason,
                )
                continue
            boundary = cast(tuple[object, ...], current_payload[5])
            control = _require_vector(
                boundary[4], "generic rejection Stage 4 control"
            )
            stream = _decode_native_three_lane_rejection_only_iteration(
                current_payload,
                previous_payload=cast(
                    tuple[object, ...], iteration_payloads[iteration - 1]
                ),
                previous_stream=stream,
                node_names=node_names,
                search_sha256=producer_sha256,
                iteration=iteration,
                operator=operator,
                operator_id=operator_id,
                completed_iterations=iteration + 1,
                restart_to_best=bool(control[2]),
                instance=instance if aggregate_merge else None,
                termination_reason=current_termination_reason,
            )
            continue
        current_termination = _require_vector(
            current_payload[11], "generic follow-up termination"
        )
        current_termination_reason = int(current_termination[0])
        if current_termination_reason not in (0, 3) or (
            current_termination_reason != 0
            and iteration != len(iteration_payloads) - 1
        ):
            raise RuntimeError(
                "native three-lane generic termination boundary is invalid"
            )
        stream = _decode_native_three_lane_sixth_iteration(
            instance,
            current_payload,
            previous_payload=cast(
                tuple[object, ...], iteration_payloads[iteration - 1]
            ),
            previous_stream=stream,
            node_names=node_names,
            search_sha256=producer_sha256,
            iteration=iteration,
            completed_iterations=iteration + 1,
            allow_constraint_no_change=current_payload[3] is not None,
            vehicle_operator_config=vehicle_operator_config,
            termination_reason=current_termination_reason,
        )
    return stream


def _canonical_route_merge_projection(
    instance: Instance,
    routes: tuple[CustomerSequence, ...],
) -> tuple[
    tuple[tuple[CustomerSequence, int, int], ...],
    dict[str, int],
    str,
]:
    """Rebuild Python's globally deduplicated controlled merge pool."""

    profiles: list[tuple[int, CustomerSequence, float, float, float]] = []
    for index, route in enumerate(routes):
        exact = solve_exact_charging(instance, route)
        if not exact.feasible:
            raise RuntimeError("canonical route-merge source is infeasible")
        profiles.append(
            (
                index,
                route,
                sum(instance.by_name[name].demand for name in route),
                exact.distance,
                exact.charging_time,
            )
        )
    pairs = sorted(
        (
            (
                len(left[1]) + len(right[1]),
                left[2] + right[2],
                -(left[3] + right[3]),
                -(left[4] + right[4]),
                left[0],
                right[0],
            ),
            left,
            right,
        )
        for position, left in enumerate(profiles)
        for right in profiles[position + 1 :]
    )
    seen: set[CustomerSequence] = set()
    candidates: list[tuple[CustomerSequence, int, int]] = []
    reason_counts: dict[str, int] = {}
    digest = hashlib.sha256()
    for _, left, right in pairs:
        for source, target in ((left, right), (right, left)):
            for position in range(len(target[1]) + 1):
                merged = (
                    target[1][:position]
                    + source[1]
                    + target[1][position:]
                )
                if merged in seen:
                    continue
                seen.add(merged)
                screen = screen_route_candidate(instance, merged)
                if screen.accepted:
                    candidates.append((merged, left[0], right[0]))
                    continue
                reason_counts[screen.reason] = reason_counts.get(screen.reason, 0) + 1
                digest.update(
                    json.dumps(
                        (screen.reason, merged),
                        separators=(",", ":"),
                    ).encode()
                )
    return tuple(candidates), reason_counts, digest.hexdigest()


def _decode_native_three_lane_third_general(
    instance: Instance,
    payload: tuple[object, ...],
    *,
    previous_payload: tuple[object, ...],
    previous_stream: NativeThreeLaneSemanticStream,
    node_names: tuple[str, ...],
    search_sha256: str,
) -> NativeThreeLaneSemanticStream:
    """Replay a general two-route third iteration without fixture assumptions."""

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native three-lane third general {name} is invalid")
        return value

    def unpack_soa(
        offsets_value: object,
        indices_value: object,
        name: str,
    ) -> tuple[CustomerSequence, ...]:
        offsets = _require_vector(offsets_value, f"{name} offsets")
        indices = _require_vector(indices_value, f"{name} indices")
        if (
            len(offsets) < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or np.any(offsets[:-1] >= offsets[1:])
            or np.any(indices < 0)
            or np.any(indices >= len(node_names))
        ):
            raise RuntimeError(f"native three-lane third general {name} SoA is invalid")
        return tuple(
            tuple(
                node_names[int(index)]
                for index in indices[
                    int(offsets[route]) : int(offsets[route + 1])
                ]
            )
            for route in range(len(offsets) - 1)
        )

    def unpack_state(value: object, name: str) -> tuple[CustomerSequence, ...]:
        state = require_tuple(value, 5 if "best" in name else 4, name)
        return unpack_soa(state[0], state[1], name)

    expected_customers = {customer.name for customer in instance.customers}

    def replay(routes: tuple[CustomerSequence, ...]) -> SolutionObjective:
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(expected_customers) or set(flattened) != expected_customers:
            raise RuntimeError(
                "native three-lane third general customer identity is invalid"
            )
        objective = SolutionObjective.zero()
        for route in routes:
            exact = solve_exact_charging(instance, route)
            if not exact.feasible:
                raise RuntimeError("native three-lane third general replay is infeasible")
            objective += SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
        return objective

    def event(operator: str, status: str, reason: str, **changes: object) -> dict[str, object]:
        result: dict[str, object] = {
            "operator": operator,
            "status": status,
            "reason": reason,
            "route_indices": (),
            "affected_route_indices": (),
            "removed_customers": (),
            "candidate_customer_sequence": (),
            "candidate_route_sequences": (),
            "candidate_vehicle_delta": None,
            "candidate_feasible": False,
            "prefilter_passed": False,
            "new_routes_created": 0,
            "exact_route_evaluations": 0,
            "selection_rank": 0,
            "chain_depth": 0,
            "segment_length": 0,
            "track": "legacy",
            "constraint_category": "",
            "removal_tier": "",
            "removal_size_requested": 0,
            "removal_size_actual": 0,
            "stagnation_iterations": 0,
            "removal_trigger": "",
            "reset_observed": False,
            "ranking_score": 0.0,
            "iteration": 2,
            "accepted": False,
            "vehicle_reduction": False,
            "distance_improvement": False,
            "candidate_objective_key": (),
        }
        result.update(changes)
        return result

    prior_legacy = unpack_state(previous_payload[6], "prior legacy")
    prior_quality = unpack_state(previous_payload[7], "prior quality")
    prior_constraint = unpack_state(previous_payload[8], "prior constraint")
    prior_best = unpack_state(previous_payload[9], "prior best")
    prior_quality_objective = replay(prior_quality)
    prior_constraint_objective = replay(prior_constraint)
    prior_best_objective = replay(prior_best)

    quality = require_tuple(payload[2], 4, "quality")
    quality_pool = require_tuple(quality[0], 5, "quality pool")
    changed = cast(
        npt.NDArray[np.int64],
        _require_array(
            quality_pool[0],
            dtype=np.dtype(np.int64),
            shape=(len(cast(npt.NDArray[np.int64], quality_pool[0])), 2),
            name="third general quality changed routes",
        ),
    )
    change_offsets = _require_vector(quality_pool[1], "third general quality offsets")
    change_indices = _require_vector(quality_pool[2], "third general quality indices")
    if len(change_offsets) != changed.shape[0] * 2 + 1:
        raise RuntimeError("native three-lane third general quality pool is invalid")
    quality_plans: list[tuple[CustomerSequence, ...]] = []
    quality_sources: list[int] = []
    seen_quality: set[tuple[CustomerSequence, ...]] = set()
    for candidate in range(changed.shape[0]):
        routes = list(prior_quality)
        for ordinal in range(2):
            route = int(changed[candidate, ordinal])
            begin = int(change_offsets[candidate * 2 + ordinal])
            end = int(change_offsets[candidate * 2 + ordinal + 1])
            routes[route] = tuple(
                node_names[int(index)] for index in change_indices[begin:end]
            )
        plan = tuple(routes)
        if plan not in seen_quality:
            seen_quality.add(plan)
            quality_plans.append(plan)
            quality_sources.append(candidate)
    quality_transaction = require_tuple(quality[1], 13, "quality transaction")
    quality_statuses = _require_vector(
        quality_transaction[1], "third general quality statuses"
    )
    quality_outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            quality[2],
            dtype=np.dtype(np.int64),
            shape=(4,),
            name="third general quality outcome",
        ),
    )
    selected_quality = int(quality_outcome[0])
    if len(quality_statuses) != len(quality_plans):
        raise RuntimeError("native three-lane third general quality selection is invalid")
    if np.any((quality_statuses < 0) | (quality_statuses > 5)):
        raise RuntimeError("native three-lane third general quality statuses are invalid")
    if selected_quality < -1 or selected_quality >= len(quality_plans):
        raise RuntimeError("native three-lane third general quality selection is invalid")
    quality_accepted = bool(quality_outcome[1])
    quality_routes = prior_quality
    quality_objective = prior_quality_objective
    quality_source: int | None = None
    quality_key: tuple[int, float, float, int] | tuple[()] = ()
    if selected_quality >= 0:
        if int(quality_statuses[selected_quality]) != 5:
            raise RuntimeError(
                "native three-lane third general selected quality status is invalid"
            )
        quality_routes = quality_plans[selected_quality]
        quality_objective = replay(quality_routes)
        quality_integers = cast(npt.NDArray[np.int64], quality_transaction[2])
        quality_floats = cast(npt.NDArray[np.float64], quality_transaction[3])
        quality_reported = SolutionObjective(
            int(quality_integers[selected_quality, 0]),
            float(quality_floats[selected_quality, 0]),
            float(quality_floats[selected_quality, 1]),
            int(quality_integers[selected_quality, 1]),
        )
        if quality_reported.key != quality_objective.key:
            raise RuntimeError(
                "native three-lane third general quality objective mismatch"
            )
        quality_source = quality_sources[selected_quality]
        quality_key = quality_objective.key
        if quality_objective.vehicle_count > prior_quality_objective.vehicle_count:
            if quality_accepted:
                raise RuntimeError(
                    "native three-lane third accepted a larger quality fleet"
                )
        elif quality_objective.key <= prior_quality_objective.key and not quality_accepted:
            raise RuntimeError(
                "native three-lane third rejected a non-worse quality move"
            )
    elif quality_accepted or bool(quality_outcome[2]) or bool(quality_outcome[3]):
        raise RuntimeError("native three-lane third rejected quality flags are invalid")
    quality_best = quality_accepted and quality_objective.key < prior_best_objective.key
    if bool(quality_outcome[2]) != quality_best:
        raise RuntimeError("native three-lane third quality best flag is invalid")
    quality_hash = hashlib.sha256()
    for plan in quality_plans:
        quality_hash.update(json.dumps(plan, separators=(",", ":")).encode())
    quality_changed = (
        tuple(int(value) for value in changed[quality_source])
        if quality_source is not None
        else ()
    )
    quality_events = (
        event(
            "two_opt_star",
            "candidate_pool_aggregate",
            "two_opt_star_complete_candidate_pool",
            aggregate_count=len(quality_plans),
            candidate_pool_hash=quality_hash.hexdigest(),
            candidate_objective_key=quality_key,
        ),
        event(
            "two_opt_star",
            "candidate_proposed"
            if quality_source is not None
            else "candidate_control_skipped",
            "two_opt_star_candidate"
            if quality_source is not None
            else "no_selected_complete_plan_feasible",
            route_indices=quality_changed,
            affected_route_indices=quality_changed,
            candidate_route_sequences=tuple(
                quality_routes[index] for index in quality_changed
            ),
            candidate_vehicle_delta=(
                len(quality_routes) - len(prior_quality)
                if quality_source is not None
                else None
            ),
            candidate_feasible=quality_source is not None,
            prefilter_passed=quality_source is not None,
            accepted=quality_accepted,
            vehicle_reduction=bool(quality_outcome[3]),
            distance_improvement=quality_objective.total_distance
            < prior_quality_objective.total_distance - 1e-9,
            candidate_objective_key=quality_key,
        ),
    )

    constraint = require_tuple(payload[3], 3, "constraint")
    selection = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[0],
            dtype=np.dtype(np.int64),
            shape=(7,),
            name="third general constraint selection",
        ),
    )
    probe = require_tuple(constraint[1], 3, "constraint probe")
    removal = require_tuple(probe[0], 7, "constraint removal")
    outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[2],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="third general constraint outcome",
        ),
    )
    if (
        int(outcome[0]) != 2
        or not 0 <= int(outcome[1]) < 2**32
        or np.any((outcome[2:] < 0) | (outcome[2:] > 1))
    ):
        raise RuntimeError("native three-lane third general constraint operator is invalid")
    removal_metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            removal[6],
            dtype=np.dtype(np.int64),
            shape=(3,),
            name="third general constraint removal metadata",
        ),
    )
    removed_indices = _require_vector(removal[2], "third general constraint removed")
    removed = tuple(node_names[int(index)] for index in removed_indices)
    removal_failed = int(removal_metadata[0]) != 0
    if removal_failed:
        if len(removed_indices) != 0 or int(removal_metadata[2]) != 0:
            raise RuntimeError(
                "native three-lane third general failed constraint removal is invalid"
            )
        partial_routes: tuple[CustomerSequence, ...] = ()
    else:
        partial_routes = unpack_soa(removal[0], removal[1], "constraint partial")
    scores, score_routes = _constraint_score_vectors(
        removal, removed_indices, "third general constraint"
    )
    repair_payload = probe[1]
    transaction_payload = probe[2]
    repaired_routes: tuple[CustomerSequence, ...] = ()
    repair_failed = False
    if repair_payload is None:
        if transaction_payload is not None:
            raise RuntimeError(
                "native three-lane third general constraint transaction lacks repair"
            )
        repair_failed = True
    else:
        repair = require_tuple(repair_payload, 3, "constraint repair")
        repair_metadata = cast(
            npt.NDArray[np.int64],
            _require_array(
                repair[2],
                dtype=np.dtype(np.int64),
                shape=(7,),
                name="third general constraint repair metadata",
            ),
        )
        repair_failed = int(repair_metadata[0]) != 0
        if repair_failed:
            if transaction_payload is not None:
                raise RuntimeError(
                    "native three-lane third general failed repair has a transaction"
                )
        else:
            repaired_routes = unpack_soa(repair[0], repair[1], "constraint repaired")
            if transaction_payload is None:
                raise RuntimeError(
                    "native three-lane third general repaired constraint lacks transaction"
                )
    transaction: tuple[object, ...] | None = None
    statuses = np.empty(0, dtype=np.int64)
    exact_rows = np.empty(0, dtype=np.int64)
    if transaction_payload is not None:
        transaction = require_tuple(transaction_payload, 13, "constraint transaction")
        statuses = _require_vector(transaction[1], "third general constraint statuses")
        if len(statuses) != 1 or np.any((statuses < 0) | (statuses > 5)):
            raise RuntimeError(
                "native three-lane third general constraint statuses are invalid"
            )
        _require_array(
            transaction[2],
            dtype=np.dtype(np.int64),
            shape=(1, 2),
            name="third general constraint objective integers",
        )
        _require_array(
            transaction[3],
            dtype=np.dtype(np.float64),
            shape=(1, 2),
            name="third general constraint objective floats",
        )
        exact_rows = _require_vector(
            transaction[5], "third general constraint exact rows"
        )
    constraint_prepared = bool(outcome[2])
    constraint_exact_feasible = statuses.tolist() == [5]
    no_change = (
        not constraint_prepared
        and constraint_exact_feasible
        and repaired_routes == prior_constraint
    )
    if constraint_prepared and (
        transaction is None or repair_failed or not constraint_exact_feasible
    ):
        raise RuntimeError("native three-lane third general constraint state is invalid")
    if not constraint_prepared and constraint_exact_feasible and not no_change:
        raise RuntimeError(
            "native three-lane third general rejected constraint changed its state"
        )
    if not constraint_prepared and np.any(outcome[3:] != 0):
        raise RuntimeError("native three-lane third general no-change journal is invalid")
    constraint_objective = prior_constraint_objective
    constraint_key: tuple[int, float, float, int] | tuple[()] = ()
    if constraint_prepared and transaction is not None:
        constraint_objective = replay(repaired_routes)
        constraint_integers = cast(npt.NDArray[np.int64], transaction[2])
        constraint_floats = cast(npt.NDArray[np.float64], transaction[3])
        constraint_reported = SolutionObjective(
            int(constraint_integers[0, 0]),
            float(constraint_floats[0, 0]),
            float(constraint_floats[0, 1]),
            int(constraint_integers[0, 1]),
        )
        if constraint_reported.key != constraint_objective.key:
            raise RuntimeError(
                "native three-lane third general constraint objective mismatch"
            )
        constraint_accepted = bool(outcome[3])
        if constraint_objective.vehicle_count > prior_constraint_objective.vehicle_count:
            if constraint_accepted:
                raise RuntimeError(
                    "native three-lane third accepted a larger constraint fleet"
                )
        elif (
            constraint_objective.key <= prior_constraint_objective.key
            and not constraint_accepted
        ):
            raise RuntimeError(
                "native three-lane third rejected a non-worse constraint move"
            )
        constraint_key = constraint_objective.key
    elif constraint_exact_feasible and transaction is not None:
        # A feasible transaction can still be discarded when it is exactly the
        # incumbent; native marks that candidate as not prepared.
        transaction_integers = cast(npt.NDArray[np.int64], transaction[2])
        transaction_floats = cast(npt.NDArray[np.float64], transaction[3])
        reported = SolutionObjective(
            int(transaction_integers[0, 0]),
            float(transaction_floats[0, 0]),
            float(transaction_floats[0, 1]),
            int(transaction_integers[0, 1]),
        )
        replayed = replay(repaired_routes)
        if reported.key != replayed.key:
            raise RuntimeError(
                "native three-lane third general constraint no-change objective mismatch"
            )
        if replayed.key != prior_constraint_objective.key:
            raise RuntimeError(
                "native three-lane third general rejected constraint objective mismatch"
            )
        # The transaction was feasible, but native discarded this exact
        # incumbent candidate before preparation.  Keep the event's candidate
        # key empty, matching the Python lane's "no selected candidate"
        # semantics; the replay above still proves the typed objective receipt
        # and incumbent identity.
        constraint_key = ()
    elif transaction is not None and not repair_failed:
        # Exact-infeasible, budget-skipped, or not-selected plans have no
        # replayable objective, but their typed status and route receipt remain
        # valid evidence of a rejected constraint candidate.
        if np.any(outcome[3:] != 0):
            raise RuntimeError(
                "native three-lane third general infeasible constraint flags are invalid"
            )
    else:
        if np.any(outcome[2:] != 0):
            raise RuntimeError(
                "native three-lane third general failed constraint flags are invalid"
            )
    constraint_accepted = bool(outcome[3]) if constraint_prepared else False
    constraint_route_indices = tuple(
        sorted(
            dict.fromkeys(
                int(value) for value in score_routes[: len(removed_indices)]
            )
        )
    )
    constraint_events = (
        event(
            "worst_energy_detour",
            "candidate_proposed",
            "constraint_ranked_removal",
            route_indices=constraint_route_indices,
            affected_route_indices=constraint_route_indices,
            removed_customers=removed,
            candidate_route_sequences=partial_routes,
            prefilter_passed=True,
            selection_rank=1,
            track="constraint_lane",
            constraint_category="worst_energy_detour",
            removal_tier="small",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            stagnation_iterations=int(selection[4]),
            removal_trigger="stagnation_baseline",
            reset_observed=bool(selection[6]),
            ranking_score=float(scores[0]) if len(scores) else 0.0,
            candidate_objective_key=constraint_key,
        ),
        event(
            "worst_energy_detour",
            "candidate_proposed" if constraint_prepared else "failed",
            (
                "constraint_removal_repaired"
                if constraint_prepared
                else "constraint_removal_no_change"
                if no_change
                else "constraint_repair_infeasible"
            ),
            affected_route_indices=(
                tuple(
                    index
                    for index, (before, after) in enumerate(
                        zip(prior_constraint, repaired_routes, strict=False)
                    )
                    if before != after
                )
                if constraint_prepared or not constraint_exact_feasible
                else ()
            ),
            removed_customers=removed,
            candidate_route_sequences=(
                repaired_routes
                if constraint_prepared or not constraint_exact_feasible
                else ()
            ),
            candidate_vehicle_delta=(
                len(repaired_routes) - len(prior_constraint)
                if constraint_prepared
                else None
            ),
            candidate_feasible=constraint_prepared,
            prefilter_passed=True,
            exact_route_evaluations=len(exact_rows) if transaction is not None else 0,
            track="constraint_lane",
            constraint_category="worst_energy_detour",
            removal_tier="small",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            stagnation_iterations=int(selection[4]),
            removal_trigger="stagnation_baseline",
            reset_observed=bool(selection[6]),
            accepted=constraint_accepted,
            vehicle_reduction=bool(outcome[5]),
            distance_improvement=constraint_prepared
            and constraint_objective.total_distance
            < prior_constraint_objective.total_distance - 1e-9,
            candidate_objective_key=constraint_key,
        ),
    )

    legacy = require_tuple(payload[0], 7, "route merge")
    legacy_metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[0],
            dtype=np.dtype(np.int64),
            shape=(4,),
            name="third general route-merge metadata",
        ),
    )
    merge_pool = require_tuple(legacy[1], 4, "route-merge pool")
    merge_offsets = _require_vector(merge_pool[0], "route-merge offsets")
    merge_indices = _require_vector(merge_pool[1], "route-merge indices")
    merge_metadata = cast(npt.NDArray[np.int64], merge_pool[2])
    screening_reasons = _require_vector(legacy[2], "route-merge screening reasons")
    plan_payload = require_tuple(legacy[3], 4, "route-merge plans")
    merge_plan_offsets = _require_vector(
        plan_payload[0], "route-merge plan offsets"
    )
    merge_plan_count = len(merge_plan_offsets) - 1
    pruning = _require_vector(merge_pool[3], "route-merge pruning")
    if (
        legacy_metadata.tolist()
        != [2, 3, len(prior_legacy), len(merge_offsets) - 1]
        or merge_metadata.shape != (len(merge_offsets) - 1, 5)
        or len(screening_reasons) != len(merge_offsets) - 1
        or len(pruning) != 2
        or int(merge_plan_offsets[0]) != 0
        or int(merge_plan_offsets[-1])
        != len(_require_vector(plan_payload[1], "route-merge route offsets")) - 1
        or cast(npt.NDArray[np.int64], legacy[5]).tolist() != [-1] * 5
    ):
        raise RuntimeError("native three-lane third general route-merge journal is invalid")
    merge_transaction: tuple[object, ...] | None = None
    if merge_plan_count == 0:
        if legacy[4] is not None:
            raise RuntimeError(
                "native three-lane third general empty merge transaction is invalid"
            )
    else:
        merge_transaction = require_tuple(legacy[4], 13, "route-merge transaction")
        merge_statuses = _require_vector(
            merge_transaction[1], "route-merge statuses"
        )
        if len(merge_statuses) != merge_plan_count or np.any(merge_statuses == 5):
            raise RuntimeError(
                "native three-lane third general merge selection is invalid"
            )
    canonical_candidates, reason_counts, prefilter_sha256 = (
        _canonical_route_merge_projection(instance, prior_legacy)
    )
    source_candidates = _require_vector(
        plan_payload[3], "route-merge source candidates"
    )
    native_candidates = tuple(
        tuple(
            node_names[int(index)]
            for index in merge_indices[
                int(merge_offsets[candidate]) : int(merge_offsets[candidate + 1])
            ]
        )
        for candidate in range(len(merge_offsets) - 1)
    )
    if tuple(native_candidates[int(source)] for source in source_candidates) != tuple(
        candidate[0] for candidate in canonical_candidates
    ):
        raise RuntimeError(
            "native three-lane third general merge candidate order diverged"
        )
    merge_events: list[dict[str, object]] = [
        event(
            "route_merge",
            "prefilter_rejected_aggregate",
            reason,
            aggregate_count=count,
            candidate_pool_hash=prefilter_sha256,
        )
        for reason, count in sorted(reason_counts.items())
    ]
    if merge_transaction is not None:
        merge_statuses = _require_vector(
            merge_transaction[1], "route-merge statuses"
        )
        result_counts: dict[tuple[str, str], int] = {}
        result_digest = hashlib.sha256()
        for merged, _, _ in canonical_candidates:
            exact = solve_exact_charging(instance, merged)
            if exact.feasible:
                raise RuntimeError(
                    "native three-lane third general omitted a feasible merge"
                )
            status = "exact_infeasible"
            reason = exact.failure_reason or "exact_charging_infeasible"
            result_counts[(status, reason)] = (
                result_counts.get((status, reason), 0) + 1
            )
            result_digest.update(
                json.dumps(
                    (status, reason, merged),
                    separators=(",", ":"),
                ).encode()
            )
        merge_events.extend(
            [
                event(
                    "route_merge",
                    f"{status}_aggregate",
                    reason,
                    aggregate_count=count,
                    prefilter_passed=True,
                    candidate_pool_hash=result_digest.hexdigest(),
                )
                for (status, reason), count in sorted(result_counts.items())
            ]
        )

    expected_quality = quality_routes if quality_accepted else prior_quality
    expected_constraint = (
        repaired_routes if constraint_accepted else prior_constraint
    )
    best_after_quality = quality_objective if quality_best else prior_best_objective
    constraint_best = constraint_accepted and (
        constraint_objective.key < best_after_quality.key
    )
    if bool(outcome[4]) != constraint_best:
        raise RuntimeError("native three-lane third constraint best flag is invalid")
    expected_best = (
        repaired_routes
        if constraint_best
        else quality_routes
        if quality_best
        else prior_best
    )
    final_states = (
        unpack_state(payload[6], "final legacy"),
        unpack_state(payload[7], "final quality"),
        unpack_state(payload[8], "final constraint"),
        unpack_state(payload[9], "final best"),
    )
    if final_states != (
        prior_legacy,
        expected_quality,
        expected_constraint,
        expected_best,
    ) or replay(final_states[3]).key != (
        constraint_objective.key
        if constraint_best
        else quality_objective.key
        if quality_best
        else prior_best_objective.key
    ):
        raise RuntimeError("native three-lane third general lane state mismatch")

    stage04 = require_tuple(payload[12], 4, "Stage 4")
    weights = cast(npt.NDArray[np.float64], stage04[0])
    rewards = cast(npt.NDArray[np.float64], stage04[1])
    calls = cast(npt.NDArray[np.int64], stage04[2])
    totals = cast(npt.NDArray[np.int64], stage04[3])
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="third general termination",
        ),
    )
    if (
        weights.shape != (len(FULL_NATIVE_OPERATOR_NAMES),)
        or rewards.shape != weights.shape
        or calls.shape != weights.shape
        or totals.shape != (len(FULL_NATIVE_OPERATOR_NAMES), 8)
        or int(termination[0]) != 0
        or int(termination[5]) != 3
    ):
        raise RuntimeError("native three-lane third general final state is invalid")
    all_events = (
        *previous_stream.neighborhood_events,
        *quality_events,
        *constraint_events,
        *merge_events,
    )
    activity = np.zeros((len(FULL_NATIVE_OPERATOR_NAMES), 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    for item in all_events:
        index = by_name[str(item["operator"])]
        activity[index, 0] = max(
            int(activity[index, 0]), int(bool(item["candidate_feasible"]))
        )
        activity[index, 1] += _semantic_event_aggregate(item, "prefilter_passed")
        activity[index, 2] += cast(int, item["exact_route_evaluations"])
        activity[index, 3] += int(item["status"] == "candidate_proposed")
        activity[index, 4] += _semantic_event_aggregate(item, "candidate_feasible")
        activity[index, 5] += int(bool(item["vehicle_reduction"]))
        activity[index, 6] += int(bool(item["distance_improvement"]))
    refinement_index = FULL_NATIVE_OPERATOR_NAMES.index(
        "vehicle_reduction_refinement"
    )
    activity[refinement_index, 5] = 0
    activity[refinement_index, 0] = previous_stream.operator_activity[
        refinement_index, 0
    ]
    activity[refinement_index, 7] = previous_stream.operator_activity[
        refinement_index, 7
    ]
    return NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(all_events),
        operator_weights=_readonly_copy(weights),
        operator_rewards=_readonly_copy(rewards),
        operator_calls=_readonly_copy(calls),
        operator_totals=_readonly_copy(totals),
        operator_activity=_readonly_copy(activity),
        termination=_readonly_copy(termination),
        transaction_sha256=search_sha256,
        initial_temperature=previous_stream.initial_temperature,
    )


def _decode_native_three_lane_third_iteration(
    instance: Instance,
    payload: tuple[object, ...],
    *,
    previous_payload: tuple[object, ...],
    previous_stream: NativeThreeLaneSemanticStream,
    node_names: tuple[str, ...],
    search_sha256: str,
) -> NativeThreeLaneSemanticStream:
    """Replay the route-merge/two-opt-star/energy third iteration."""

    _verify_native_three_lane_semantic_hash(payload)
    if len(payload) != 14:
        raise RuntimeError("native three-lane third iteration is invalid")
    if isinstance(payload[0], tuple) and len(payload[0]) == 7:
        return _decode_native_three_lane_third_general(
            instance,
            payload,
            previous_payload=previous_payload,
            previous_stream=previous_stream,
            node_names=node_names,
            search_sha256=search_sha256,
        )

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native three-lane third {name} is invalid")
        return value

    def unpack_routes(value: object, name: str) -> tuple[CustomerSequence, ...]:
        state = require_tuple(value, 5 if "best" in name else 4, name)
        offsets = _require_vector(state[0], f"{name} offsets")
        indices = _require_vector(state[1], f"{name} indices")
        if (
            len(offsets) < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or np.any(offsets[:-1] >= offsets[1:])
            or np.any(indices < 0)
            or np.any(indices >= len(node_names))
        ):
            raise RuntimeError(f"native three-lane third {name} SoA is invalid")
        return tuple(
            tuple(
                node_names[int(index)]
                for index in indices[
                    int(offsets[route]) : int(offsets[route + 1])
                ]
            )
            for route in range(len(offsets) - 1)
        )

    expected_customers = {customer.name for customer in instance.customers}

    def replay(routes: tuple[CustomerSequence, ...]) -> SolutionObjective:
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(expected_customers) or set(flattened) != expected_customers:
            raise RuntimeError("native three-lane third customer identity is invalid")
        objective = SolutionObjective.zero()
        for route in routes:
            exact = solve_exact_charging(instance, route)
            if not exact.feasible:
                raise RuntimeError("native three-lane third replay is infeasible")
            objective += SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
        return objective

    def event(operator: str, status: str, reason: str, **changes: object) -> dict[str, object]:
        result: dict[str, object] = {
            "operator": operator,
            "status": status,
            "reason": reason,
            "route_indices": (),
            "affected_route_indices": (),
            "removed_customers": (),
            "candidate_customer_sequence": (),
            "candidate_route_sequences": (),
            "candidate_vehicle_delta": None,
            "candidate_feasible": False,
            "prefilter_passed": False,
            "new_routes_created": 0,
            "exact_route_evaluations": 0,
            "selection_rank": 0,
            "chain_depth": 0,
            "segment_length": 0,
            "track": "legacy",
            "constraint_category": "",
            "removal_tier": "",
            "removal_size_requested": 0,
            "removal_size_actual": 0,
            "stagnation_iterations": 0,
            "removal_trigger": "",
            "reset_observed": False,
            "ranking_score": 0.0,
            "iteration": 2,
            "accepted": False,
            "vehicle_reduction": False,
            "distance_improvement": False,
            "candidate_objective_key": (),
        }
        result.update(changes)
        return result

    prior_legacy = unpack_routes(previous_payload[6], "prior legacy")
    prior_quality = unpack_routes(previous_payload[7], "prior quality")
    prior_constraint = unpack_routes(previous_payload[8], "prior constraint")
    prior_best = unpack_routes(previous_payload[9], "prior best")
    prior_constraint_objective = replay(prior_constraint)
    prior_best_objective = replay(prior_best)

    legacy = require_tuple(payload[0], 2, "legacy")
    legacy_metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[0],
            dtype=np.dtype(np.int64),
            shape=(4,),
            name="third legacy metadata",
        ),
    )
    if legacy_metadata.tolist() != [2, 3, 1, 0] or payload[4] is not None:
        raise RuntimeError("native three-lane third route-merge decision is invalid")

    quality = require_tuple(payload[2], 4, "quality")
    quality_pool = require_tuple(quality[0], 5, "quality pool")
    changed = cast(
        npt.NDArray[np.int64],
        _require_array(
            quality_pool[0],
            dtype=np.dtype(np.int64),
            shape=(len(cast(npt.NDArray[np.int64], quality_pool[0])), 2),
            name="third quality changed routes",
        ),
    )
    quality_change_offsets = _require_vector(
        quality_pool[1], "third quality offsets"
    )
    quality_change_indices = _require_vector(
        quality_pool[2], "third quality indices"
    )
    quality_outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            quality[2],
            dtype=np.dtype(np.int64),
            shape=(4,),
            name="third quality outcome",
        ),
    )
    if (
        len(quality_change_offsets) != changed.shape[0] * 2 + 1
        or quality_outcome.tolist() != [-1, 0, 0, 0]
    ):
        raise RuntimeError("native three-lane third two-opt-star decision is invalid")
    quality_plans: list[tuple[CustomerSequence, ...]] = []
    seen_quality: set[tuple[CustomerSequence, ...]] = set()
    for candidate in range(changed.shape[0]):
        routes = list(prior_quality)
        for ordinal in range(2):
            route = int(changed[candidate, ordinal])
            if route < 0 or route >= len(routes):
                raise RuntimeError("native three-lane third quality route is invalid")
            begin = int(quality_change_offsets[candidate * 2 + ordinal])
            end = int(quality_change_offsets[candidate * 2 + ordinal + 1])
            routes[route] = tuple(
                node_names[int(index)] for index in quality_change_indices[begin:end]
            )
        plan = tuple(routes)
        if plan not in seen_quality:
            seen_quality.add(plan)
            quality_plans.append(plan)
    if quality[1] is None:
        if quality_plans:
            raise RuntimeError(
                "native three-lane third quality transaction is missing"
            )
    else:
        quality_transaction = require_tuple(
            quality[1], 13, "quality transaction"
        )
        quality_statuses = _require_vector(
            quality_transaction[1], "third quality statuses"
        )
        if len(quality_statuses) != len(quality_plans) or np.any(
            quality_statuses == 5
        ):
            raise RuntimeError(
                "native three-lane third skipped quality journal is invalid"
            )
    quality_pool_hash = hashlib.sha256()
    for plan in quality_plans:
        quality_pool_hash.update(
            json.dumps(plan, separators=(",", ":")).encode()
        )

    constraint = require_tuple(payload[3], 3, "constraint")
    selection = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[0],
            dtype=np.dtype(np.int64),
            shape=(7,),
            name="third constraint selection",
        ),
    )
    probe = require_tuple(constraint[1], 3, "constraint probe")
    removal = require_tuple(probe[0], 7, "constraint removal")
    repair = require_tuple(probe[1], 3, "constraint repair")
    transaction = require_tuple(probe[2], 13, "constraint transaction")
    outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[2],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="third constraint outcome",
        ),
    )
    if outcome.tolist()[:1] != [2] or not bool(outcome[3]):
        raise RuntimeError("native three-lane third constraint decision is invalid")
    removed_indices = _require_vector(removal[2], "third constraint removed")
    removed = tuple(node_names[int(index)] for index in removed_indices)
    partial_state = (removal[0], removal[1], None, None)
    repaired_state = (repair[0], repair[1], None, None)
    partial_routes = unpack_routes(partial_state, "constraint partial")
    repaired_routes = unpack_routes(repaired_state, "constraint repaired")
    candidate_objective = replay(repaired_routes)
    objective_integer = cast(npt.NDArray[np.int64], transaction[2])
    objective_float = cast(npt.NDArray[np.float64], transaction[3])
    reported = SolutionObjective(
        int(objective_integer[0, 0]),
        float(objective_float[0, 0]),
        float(objective_float[0, 1]),
        int(objective_integer[0, 1]),
    )
    if reported.key != candidate_objective.key:
        raise RuntimeError("native three-lane third constraint objective mismatch")
    affected = tuple(
        index
        for index, (before, after) in enumerate(
            zip(prior_constraint, repaired_routes, strict=False)
        )
        if before != after
    )
    route_indices = _require_vector(removal[5], "third constraint routes")
    scores = cast(npt.NDArray[np.float64], removal[4])
    followup_events = (
        event(
            "two_opt_star",
            "candidate_pool_aggregate",
            "two_opt_star_complete_candidate_pool",
            aggregate_count=len(quality_plans),
            candidate_pool_hash=quality_pool_hash.hexdigest(),
        ),
        event(
            "two_opt_star",
            "candidate_control_skipped",
            "no_selected_complete_plan_feasible",
        ),
        event(
            "worst_energy_detour",
            "candidate_proposed",
            "constraint_ranked_removal",
            route_indices=(int(route_indices[0]),),
            affected_route_indices=(int(route_indices[0]),),
            removed_customers=removed,
            candidate_route_sequences=partial_routes,
            prefilter_passed=True,
            selection_rank=1,
            track="constraint_lane",
            constraint_category="worst_energy_detour",
            removal_tier="small",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            stagnation_iterations=int(selection[4]),
            removal_trigger="stagnation_baseline",
            reset_observed=bool(selection[6]),
            ranking_score=float(scores[0]),
            candidate_objective_key=candidate_objective.key,
        ),
        event(
            "worst_energy_detour",
            "candidate_proposed",
            "constraint_removal_repaired",
            affected_route_indices=affected,
            removed_customers=removed,
            candidate_route_sequences=repaired_routes,
            candidate_vehicle_delta=len(repaired_routes) - len(prior_constraint),
            candidate_feasible=True,
            prefilter_passed=True,
            exact_route_evaluations=len(
                _require_vector(transaction[5], "third constraint exact rows")
            ),
            track="constraint_lane",
            constraint_category="worst_energy_detour",
            removal_tier="small",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            stagnation_iterations=int(selection[4]),
            removal_trigger="stagnation_baseline",
            reset_observed=bool(selection[6]),
            accepted=True,
            vehicle_reduction=candidate_objective.vehicle_count
            < prior_constraint_objective.vehicle_count,
            distance_improvement=candidate_objective.total_distance
            < prior_constraint_objective.total_distance - 1e-9,
            candidate_objective_key=candidate_objective.key,
        ),
        event("route_merge", "not_applicable", "only_one_route"),
    )
    expected_states = (prior_legacy, prior_quality, repaired_routes, prior_best)
    final_states = (
        unpack_routes(payload[6], "final legacy"),
        unpack_routes(payload[7], "final quality"),
        unpack_routes(payload[8], "final constraint"),
        unpack_routes(payload[9], "final best"),
    )
    if final_states != expected_states or replay(final_states[3]).key != prior_best_objective.key:
        raise RuntimeError("native three-lane third lane state mismatch")

    stage04 = require_tuple(payload[12], 4, "Stage 4")
    weights = cast(npt.NDArray[np.float64], stage04[0])
    rewards = cast(npt.NDArray[np.float64], stage04[1])
    calls = cast(npt.NDArray[np.int64], stage04[2])
    totals = cast(npt.NDArray[np.int64], stage04[3])
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="third termination",
        ),
    )
    if (
        weights.shape != (len(FULL_NATIVE_OPERATOR_NAMES),)
        or rewards.shape != weights.shape
        or calls.shape != weights.shape
        or totals.shape != (len(FULL_NATIVE_OPERATOR_NAMES), 8)
        or termination.tolist()[-1] != 3
        or int(termination[0]) != 0
    ):
        raise RuntimeError("native three-lane third final state is invalid")
    all_events = (*previous_stream.neighborhood_events, *followup_events)
    activity = np.zeros((len(FULL_NATIVE_OPERATOR_NAMES), 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    for item in all_events:
        index = by_name[str(item["operator"])]
        activity[index, 0] = max(
            int(activity[index, 0]), int(bool(item["candidate_feasible"]))
        )
        activity[index, 1] += _semantic_event_aggregate(item, "prefilter_passed")
        activity[index, 2] += cast(int, item["exact_route_evaluations"])
        activity[index, 3] += int(item["status"] == "candidate_proposed")
        activity[index, 4] += _semantic_event_aggregate(item, "candidate_feasible")
        activity[index, 5] += int(bool(item["vehicle_reduction"]))
        activity[index, 6] += int(bool(item["distance_improvement"]))
    refinement_index = FULL_NATIVE_OPERATOR_NAMES.index(
        "vehicle_reduction_refinement"
    )
    activity[refinement_index, 5] = 0
    activity[refinement_index, 0] = previous_stream.operator_activity[
        refinement_index, 0
    ]
    activity[refinement_index, 7] = previous_stream.operator_activity[
        refinement_index, 7
    ]
    return NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(all_events),
        operator_weights=_readonly_copy(weights),
        operator_rewards=_readonly_copy(rewards),
        operator_calls=_readonly_copy(calls),
        operator_totals=_readonly_copy(totals),
        operator_activity=_readonly_copy(activity),
        termination=_readonly_copy(termination),
        transaction_sha256=search_sha256,
        initial_temperature=previous_stream.initial_temperature,
    )


def _decode_native_three_lane_fourth_general(
    instance: Instance,
    payload: tuple[object, ...],
    *,
    previous_payload: tuple[object, ...],
    previous_stream: NativeThreeLaneSemanticStream,
    node_names: tuple[str, ...],
    search_sha256: str,
) -> NativeThreeLaneSemanticStream:
    """Independently project a real route-merge fourth iteration."""

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native three-lane fourth general {name} is invalid")
        return value

    def unpack_soa(
        offsets_value: object,
        indices_value: object,
        name: str,
    ) -> tuple[CustomerSequence, ...]:
        offsets = _require_vector(offsets_value, f"{name} offsets")
        indices = _require_vector(indices_value, f"{name} indices")
        if (
            len(offsets) < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or np.any(offsets[:-1] >= offsets[1:])
            or np.any(indices < 0)
            or np.any(indices >= len(node_names))
        ):
            raise RuntimeError(
                f"native three-lane fourth general {name} SoA is invalid"
            )
        routes = tuple(
            tuple(
                node_names[int(index)]
                for index in indices[
                    int(offsets[route]) : int(offsets[route + 1])
                ]
            )
            for route in range(len(offsets) - 1)
        )
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(set(flattened)):
            raise RuntimeError(
                f"native three-lane fourth general {name} repeats customers"
            )
        return routes

    def unpack_state(value: object, name: str) -> tuple[CustomerSequence, ...]:
        state = require_tuple(value, 5 if "best" in name else 4, name)
        return unpack_soa(state[0], state[1], name)

    expected_customers = {customer.name for customer in instance.customers}

    def replay(routes: tuple[CustomerSequence, ...]) -> SolutionObjective:
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(expected_customers) or set(flattened) != expected_customers:
            raise RuntimeError(
                "native three-lane fourth general customer identity is invalid"
            )
        objective = SolutionObjective.zero()
        for route in routes:
            exact = solve_exact_charging(instance, route)
            if not exact.feasible:
                raise RuntimeError(
                    "native three-lane fourth general replay is infeasible"
                )
            objective += SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
        return objective

    def event(
        operator: str,
        status: str,
        reason: str,
        **changes: object,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "operator": operator,
            "status": status,
            "reason": reason,
            "route_indices": (),
            "affected_route_indices": (),
            "removed_customers": (),
            "candidate_customer_sequence": (),
            "candidate_route_sequences": (),
            "candidate_vehicle_delta": None,
            "candidate_feasible": False,
            "prefilter_passed": False,
            "new_routes_created": 0,
            "exact_route_evaluations": 0,
            "selection_rank": 0,
            "chain_depth": 0,
            "segment_length": 0,
            "track": "legacy",
            "constraint_category": "",
            "removal_tier": "",
            "removal_size_requested": 0,
            "removal_size_actual": 0,
            "stagnation_iterations": 0,
            "removal_trigger": "",
            "reset_observed": False,
            "ranking_score": 0.0,
            "iteration": 3,
            "accepted": False,
            "vehicle_reduction": False,
            "distance_improvement": False,
            "candidate_objective_key": (),
        }
        result.update(changes)
        return result

    prior_legacy = unpack_state(previous_payload[6], "prior legacy")
    prior_quality = unpack_state(previous_payload[7], "prior quality")
    prior_constraint = unpack_state(previous_payload[8], "prior constraint")
    prior_best = unpack_state(previous_payload[9], "prior best")
    prior_best_objective = replay(prior_best)

    quality = require_tuple(payload[2], 7, "quality")
    if (
        not isinstance(quality[0], np.ndarray)
        or quality[0].dtype != np.dtype(np.int64)
        or quality[0].ndim != 2
        or quality[0].shape[1] != 8
        or not quality[0].flags.c_contiguous
    ):
        raise RuntimeError(
            "native three-lane fourth general quality attempts are invalid: "
            f"type={type(quality[0]).__name__}, "
            f"dtype={getattr(quality[0], 'dtype', None)}, "
            f"shape={getattr(quality[0], 'shape', None)}, "
            f"contiguous={getattr(getattr(quality[0], 'flags', None), 'c_contiguous', None)}"
        )
    attempts = cast(npt.NDArray[np.int64], quality[0])
    removed_offsets = _require_vector(
        quality[1], "fourth general quality removed offsets"
    )
    removed_indices = _require_vector(
        quality[2], "fourth general quality removed indices"
    )
    quality_outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            quality[5],
            dtype=np.dtype(np.int64),
            shape=(4,),
            name="fourth general quality outcome",
        ),
    )
    if (
        len(removed_offsets) != len(attempts) + 1
        or int(removed_offsets[0]) != 0
        or int(removed_offsets[-1]) != len(removed_indices)
        or np.any(removed_offsets[:-1] > removed_offsets[1:])
        or int(quality_outcome[0]) < -1
        or np.any((quality_outcome[1:] != 0) & (quality_outcome[1:] != 1))
    ):
        raise RuntimeError(
            "native three-lane fourth general quality journal is invalid: "
            f"attempts={attempts.tolist()}, repair_present={quality[3] is not None}, "
            f"transaction_present={quality[4] is not None}, "
            f"outcome={cast(npt.NDArray[np.int64], quality[5]).tolist()}"
        )
    operator_config = VehicleOperatorConfig()
    candidate_limit = max(
        16,
        operator_config.quality_route_segment_probe_exact_evaluation_budget * 4,
    )
    quality_events: list[dict[str, object]] = []
    quality_candidate_attempts: list[
        tuple[npt.NDArray[np.int64], tuple[str, ...]]
    ] = []
    attempt_index = 0
    considered = 0
    for source_index, source in enumerate(prior_quality):
        if quality_candidate_attempts:
            break
        if len(source) <= operator_config.route_segment_min_length:
            continue
        last_length = min(operator_config.route_segment_max_length, len(source) - 1)
        for segment_length in range(
            operator_config.route_segment_min_length,
            last_length + 1,
        ):
            if quality_candidate_attempts:
                break
            for start in range(len(source) - segment_length + 1):
                if considered >= candidate_limit:
                    quality_events.append(
                        event(
                            "route_segment_destroy",
                            "budget_exhausted",
                            "route_segment_candidate_budget",
                            route_indices=(source_index,),
                            affected_route_indices=(source_index,),
                            segment_length=segment_length,
                        )
                    )
                    break
                if attempt_index >= len(attempts):
                    if quality_candidate_attempts:
                        break
                    raise RuntimeError(
                        "native three-lane fourth general quality attempt is missing"
                    )
                considered += 1
                row = attempts[attempt_index]
                expected_removed = source[start : start + segment_length]
                actual_removed = tuple(
                    node_names[int(index)]
                    for index in removed_indices[
                        int(removed_offsets[attempt_index]) : int(
                            removed_offsets[attempt_index + 1]
                        )
                    ]
                )
                if (
                    row.tolist()
                    != [
                        source_index,
                        start,
                        segment_length,
                        0,
                        0,
                        considered,
                        int(row[6]),
                        0,
                    ]
                    or int(row[6]) not in (0, 1)
                    or actual_removed != expected_removed
                ):
                    raise RuntimeError(
                        "native three-lane fourth general quality attempt diverged"
                    )
                if bool(row[6]):
                    quality_candidate_attempts.append((row, actual_removed))
                else:
                    quality_events.append(
                        event(
                            "route_segment_destroy",
                            "failed",
                            "route_segment_no_change",
                            route_indices=(source_index,),
                            removed_customers=actual_removed,
                            candidate_vehicle_delta=0,
                            prefilter_passed=True,
                            selection_rank=considered,
                            segment_length=segment_length,
                        )
                    )
                attempt_index += 1
                if quality_candidate_attempts:
                    break
    if attempt_index != len(attempts):
        raise RuntimeError(
            "native three-lane fourth general quality attempt is extraneous"
        )
    quality_expected_state = prior_quality
    if quality_candidate_attempts:
        if (
            quality[3] is None
            or quality[4] is None
            or int(quality_outcome[0]) not in range(len(quality_candidate_attempts))
        ):
            raise RuntimeError(
                "native three-lane fourth general quality selection is invalid"
            )
        quality_repair = require_tuple(quality[3], 3, "quality repair")
        quality_routes = unpack_soa(
            quality_repair[0], quality_repair[1], "quality repaired"
        )
        quality_transaction = require_tuple(
            quality[4], 13, "quality transaction"
        )
        quality_statuses = _require_vector(
            quality_transaction[1], "fourth general quality statuses"
        )
        quality_exact_rows = _require_vector(
            quality_transaction[5], "fourth general quality exact rows"
        )
        selected_quality = int(quality_outcome[0])
        if (
            len(quality_statuses) != len(quality_candidate_attempts)
            or int(quality_statuses[selected_quality]) != 5
        ):
            raise RuntimeError(
                "native three-lane fourth general quality transaction is invalid"
            )
        quality_objective = replay(quality_routes)
        quality_integers = cast(
            npt.NDArray[np.int64], quality_transaction[2]
        )
        quality_floats = cast(npt.NDArray[np.float64], quality_transaction[3])
        reported_quality = SolutionObjective(
            int(quality_integers[selected_quality, 0]),
            float(quality_floats[selected_quality, 0]),
            float(quality_floats[selected_quality, 1]),
            int(quality_integers[selected_quality, 1]),
        )
        if reported_quality.key != quality_objective.key:
            raise RuntimeError(
                "native three-lane fourth general quality objective mismatch"
            )
        selected_attempt, selected_removed = quality_candidate_attempts[
            selected_quality
        ]
        quality_affected = tuple(
            index
            for index, (before, after) in enumerate(
                zip(prior_quality, quality_routes, strict=False)
            )
            if before != after
        )
        quality_events.append(
            event(
                "route_segment_destroy",
                "candidate_proposed",
                "route_segment_repaired",
                route_indices=(int(selected_attempt[0]),),
                affected_route_indices=quality_affected,
                removed_customers=selected_removed,
                candidate_route_sequences=quality_routes,
                candidate_vehicle_delta=len(quality_routes) - len(prior_quality),
                candidate_feasible=True,
                prefilter_passed=True,
                exact_route_evaluations=len(quality_exact_rows),
                selection_rank=int(selected_attempt[5]),
                segment_length=int(selected_attempt[2]),
                accepted=bool(quality_outcome[1]),
                vehicle_reduction=quality_objective.vehicle_count
                < replay(prior_quality).vehicle_count,
                distance_improvement=quality_objective.total_distance
                < replay(prior_quality).total_distance - 1e-9,
                candidate_objective_key=reported_quality.key,
            )
        )
        if bool(quality_outcome[1]):
            quality_expected_state = quality_routes
    elif (
        quality[3] is not None
        or quality[4] is not None
        or quality_outcome.tolist() != [-1, 0, 0, 0]
    ):
        raise RuntimeError(
            "native three-lane fourth general empty quality result is invalid"
        )

    constraint = require_tuple(payload[3], 3, "constraint")
    selection = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[0],
            dtype=np.dtype(np.int64),
            shape=(7,),
            name="fourth general constraint selection",
        ),
    )
    probe = require_tuple(constraint[1], 3, "constraint probe")
    removal = require_tuple(probe[0], 7, "constraint removal")
    constraint_repair = require_tuple(probe[1], 3, "constraint repair")
    constraint_transaction = require_tuple(probe[2], 13, "constraint transaction")
    constraint_outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[2],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="fourth general constraint outcome",
        ),
    )
    constraint_removed_indices = _require_vector(
        removal[2], "fourth general constraint removed"
    )
    constraint_removed = tuple(
        node_names[int(index)] for index in constraint_removed_indices
    )
    constraint_partial = unpack_soa(
        removal[0], removal[1], "constraint partial"
    )
    constraint_routes = unpack_soa(
        constraint_repair[0], constraint_repair[1], "constraint repaired"
    )
    constraint_scores, constraint_score_routes = _constraint_score_vectors(
        removal, constraint_removed_indices, "fourth general constraint"
    )
    constraint_route_indices = tuple(
        sorted(
            dict.fromkeys(
                int(value)
                for value in constraint_score_routes[: len(constraint_removed_indices)]
            )
        )
    )
    constraint_statuses = _require_vector(
        constraint_transaction[1], "fourth general constraint statuses"
    )
    constraint_exact_rows = _require_vector(
        constraint_transaction[5], "fourth general constraint exact rows"
    )
    constraint_no_change = (
        int(constraint_outcome[0]) == 3
        and not bool(constraint_outcome[2])
        and constraint_statuses.tolist() == [5]
        and len(constraint_exact_rows) == 0
        and constraint_routes == prior_constraint
    )
    if (
        not constraint_no_change
        or np.any(constraint_outcome[3:] != 0)
        or not constraint_route_indices
    ):
        raise RuntimeError(
            "native three-lane fourth general constraint journal is invalid"
        )
    constraint_events = (
        event(
            "shaw_related",
            "candidate_proposed",
            "constraint_ranked_removal",
            route_indices=constraint_route_indices,
            affected_route_indices=constraint_route_indices,
            removed_customers=constraint_removed,
            candidate_route_sequences=constraint_partial,
            prefilter_passed=True,
            selection_rank=1,
            track="constraint_lane",
            constraint_category="shaw_related",
            removal_tier="small",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            stagnation_iterations=int(selection[4]),
            removal_trigger="stagnation_baseline",
            reset_observed=bool(selection[6]),
            ranking_score=float(constraint_scores[0]),
        ),
        event(
            "shaw_related",
            "failed",
            "constraint_removal_no_change",
            removed_customers=constraint_removed,
            prefilter_passed=True,
            track="constraint_lane",
            constraint_category="shaw_related",
            removal_tier="small",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            stagnation_iterations=int(selection[4]),
            removal_trigger="stagnation_baseline",
            reset_observed=bool(selection[6]),
        ),
    )

    legacy = require_tuple(payload[0], 7, "route merge")
    legacy_metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[0],
            dtype=np.dtype(np.int64),
            shape=(4,),
            name="fourth general route-merge metadata",
        ),
    )
    merge_pool = require_tuple(legacy[1], 4, "route-merge pool")
    merge_offsets = _require_vector(merge_pool[0], "fourth general merge offsets")
    merge_indices = _require_vector(merge_pool[1], "fourth general merge indices")
    merge_metadata = cast(npt.NDArray[np.int64], merge_pool[2])
    screening_reasons = _require_vector(
        legacy[2], "fourth general merge screening reasons"
    )
    plan_payload = require_tuple(legacy[3], 4, "route-merge plans")
    source_candidates = _require_vector(
        plan_payload[3], "fourth general merge source candidates"
    )
    merge_transaction = (
        None
        if legacy[4] is None
        else require_tuple(legacy[4], 13, "route-merge transaction")
    )
    merge_outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[5],
            dtype=np.dtype(np.int64),
            shape=(5,),
            name="fourth general merge outcome",
        ),
    )
    merge_count = len(merge_offsets) - 1
    merge_statuses = (
        np.empty(0, dtype=np.int64)
        if merge_transaction is None
        else _require_vector(
            merge_transaction[1], "fourth general merge statuses"
        )
    )
    merge_exact_rows = (
        np.empty(0, dtype=np.int64)
        if merge_transaction is None
        else _require_vector(
            merge_transaction[5], "fourth general merge exact rows"
        )
    )
    if (
        legacy_metadata.tolist() != [3, 3, len(prior_legacy), merge_count]
        or int(merge_offsets[0]) != 0
        or int(merge_offsets[-1]) != len(merge_indices)
        or merge_metadata.shape != (merge_count, 5)
        or len(screening_reasons) != merge_count
        or len(source_candidates) != len(merge_statuses)
        or payload[4] is not None
    ):
        raise RuntimeError(
            "native three-lane fourth general route-merge journal is invalid"
        )
    native_candidates = tuple(
        tuple(
            node_names[int(index)]
            for index in merge_indices[
                int(merge_offsets[candidate]) : int(merge_offsets[candidate + 1])
            ]
        )
        for candidate in range(merge_count)
    )

    canonical_candidate_projection, prefilter_counts, prefilter_sha256 = (
        _canonical_route_merge_projection(instance, prior_legacy)
    )
    canonical_candidates = list(canonical_candidate_projection)
    native_passed = tuple(
        native_candidates[int(source)] for source in source_candidates
    )
    if native_passed != tuple(candidate[0] for candidate in canonical_candidates):
        raise RuntimeError(
            "native three-lane fourth general merge candidate order diverged"
        )

    merge_events: list[dict[str, object]] = [
        event(
            "route_merge",
            "prefilter_rejected_aggregate",
            reason,
            aggregate_count=count,
            candidate_pool_hash=prefilter_sha256,
        )
        for reason, count in sorted(prefilter_counts.items())
    ]
    result_counts: dict[tuple[str, str], int] = {}
    result_digest = hashlib.sha256()
    feasible: list[tuple[SolutionObjective, CustomerSequence, int, int, int]] = []
    for plan, (merged, left, right) in enumerate(canonical_candidates):
        exact = solve_exact_charging(instance, merged)
        if exact.feasible:
            status = "feasible_candidate"
            reason = "exact_charging_feasible"
            objective = SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
            feasible.append((objective, merged, left, right, plan))
        else:
            status = "exact_infeasible"
            reason = exact.failure_reason or "exact_charging_infeasible"
        result_counts[(status, reason)] = result_counts.get((status, reason), 0) + 1
        result_digest.update(
            json.dumps((status, reason, merged), separators=(",", ":")).encode()
        )
    merge_events.extend(
        event(
            "route_merge",
            f"{status}_aggregate",
            reason,
            candidate_feasible=status == "feasible_candidate",
            prefilter_passed=True,
            aggregate_count=count,
            candidate_pool_hash=result_digest.hexdigest(),
        )
        for (status, reason), count in sorted(result_counts.items())
    )
    if feasible:
        raise RuntimeError(
            "native three-lane fourth general feasible merge is not implemented"
        )
    if (
        merge_outcome.tolist() != [-1, -1, -1, -1, -1]
        or np.any((merge_statuses != 1) & (merge_statuses != 4))
    ):
        raise RuntimeError(
            "native three-lane fourth general merge decision is invalid"
        )
    if len(merge_exact_rows) == 0 and canonical_candidates:
        merge_events.append(
            event(
                "route_merge",
                "budget_exhausted",
                "candidate_control_round_budget",
            )
        )

    final_states = (
        unpack_state(payload[6], "final legacy"),
        unpack_state(payload[7], "final quality"),
        unpack_state(payload[8], "final constraint"),
        unpack_state(payload[9], "final best"),
    )
    if final_states != (
        prior_legacy,
        quality_expected_state,
        prior_constraint,
        prior_best,
    ) or replay(final_states[3]).key != prior_best_objective.key:
        raise RuntimeError(
            "native three-lane fourth general lane state mismatch"
        )

    all_events = (
        *previous_stream.neighborhood_events,
        *quality_events,
        *constraint_events,
        *merge_events,
    )
    stage04 = require_tuple(payload[12], 4, "Stage 4")
    operator_count = len(FULL_NATIVE_OPERATOR_NAMES)
    weights = cast(npt.NDArray[np.float64], stage04[0])
    rewards = cast(npt.NDArray[np.float64], stage04[1])
    calls = cast(npt.NDArray[np.int64], stage04[2])
    totals = cast(npt.NDArray[np.int64], stage04[3])
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="fourth general termination",
        ),
    )
    if (
        weights.shape != (operator_count,)
        or rewards.shape != weights.shape
        or calls.shape != weights.shape
        or totals.shape != (operator_count, 8)
        or int(termination[0]) != 0
        or int(termination[5]) != 4
        or np.any(weights <= 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(rewards < 0.0)
        or np.any(calls < 0)
        or np.any(totals < 0)
    ):
        raise RuntimeError(
            "native three-lane fourth general final state is invalid"
        )
    activity = np.zeros((operator_count, 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    for item in all_events:
        index = by_name[str(item["operator"])]
        activity[index, 0] = max(
            int(activity[index, 0]), int(bool(item["candidate_feasible"]))
        )
        activity[index, 1] += _semantic_event_aggregate(item, "prefilter_passed")
        activity[index, 2] += cast(int, item["exact_route_evaluations"])
        activity[index, 3] += int(item["status"] == "candidate_proposed")
        activity[index, 4] += _semantic_event_aggregate(item, "candidate_feasible")
        activity[index, 5] += int(bool(item["vehicle_reduction"]))
        activity[index, 6] += int(bool(item["distance_improvement"]))
    refinement_index = FULL_NATIVE_OPERATOR_NAMES.index(
        "vehicle_reduction_refinement"
    )
    activity[refinement_index, 5] = 0
    activity[refinement_index, 0] = previous_stream.operator_activity[
        refinement_index, 0
    ]
    activity[refinement_index, 7] = previous_stream.operator_activity[
        refinement_index, 7
    ]
    return NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(all_events),
        operator_weights=_readonly_copy(weights),
        operator_rewards=_readonly_copy(rewards),
        operator_calls=_readonly_copy(calls),
        operator_totals=_readonly_copy(totals),
        operator_activity=_readonly_copy(activity),
        termination=_readonly_copy(termination),
        transaction_sha256=search_sha256,
        initial_temperature=previous_stream.initial_temperature,
    )


def _decode_native_three_lane_fourth_iteration(
    instance: Instance,
    payload: tuple[object, ...],
    *,
    previous_payload: tuple[object, ...],
    previous_stream: NativeThreeLaneSemanticStream,
    node_names: tuple[str, ...],
    search_sha256: str,
) -> NativeThreeLaneSemanticStream:
    """Replay the standard/route-segment/Shaw fourth iteration."""

    _verify_native_three_lane_semantic_hash(payload)
    if len(payload) != 14:
        raise RuntimeError("native three-lane fourth iteration is invalid")
    if isinstance(payload[0], tuple) and len(payload[0]) == 7:
        return _decode_native_three_lane_fourth_general(
            instance,
            payload,
            previous_payload=previous_payload,
            previous_stream=previous_stream,
            node_names=node_names,
            search_sha256=search_sha256,
        )

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native three-lane fourth {name} is invalid")
        return value

    def unpack_state(value: object, name: str) -> tuple[CustomerSequence, ...]:
        state = require_tuple(value, 5 if "best" in name else 4, name)
        return unpack_soa(state[0], state[1], name)

    def unpack_soa(
        offsets_value: object,
        indices_value: object,
        name: str,
    ) -> tuple[CustomerSequence, ...]:
        offsets = _require_vector(offsets_value, f"{name} offsets")
        indices = _require_vector(indices_value, f"{name} indices")
        if (
            len(offsets) < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or np.any(offsets[:-1] >= offsets[1:])
            or np.any(indices < 0)
            or np.any(indices >= len(node_names))
        ):
            raise RuntimeError(f"native three-lane fourth {name} SoA is invalid")
        routes = tuple(
            tuple(
                node_names[int(index)]
                for index in indices[int(offsets[route]) : int(offsets[route + 1])]
            )
            for route in range(len(offsets) - 1)
        )
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(set(flattened)):
            raise RuntimeError(f"native three-lane fourth {name} repeats customers")
        return routes

    expected_customers = {customer.name for customer in instance.customers}

    def replay(routes: tuple[CustomerSequence, ...]) -> SolutionObjective:
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(expected_customers) or set(flattened) != expected_customers:
            raise RuntimeError("native three-lane fourth customer identity is invalid")
        objective = SolutionObjective.zero()
        for route in routes:
            exact = solve_exact_charging(instance, route)
            if not exact.feasible:
                raise RuntimeError("native three-lane fourth replay is infeasible")
            objective += SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
        return objective

    def transaction(value: object, count: int, name: str) -> tuple[object, ...]:
        result = require_tuple(value, 13, f"{name} transaction")
        _require_array(
            result[1],
            dtype=np.dtype(np.int64),
            shape=(count,),
            name=f"fourth {name} statuses",
        )
        _require_array(
            result[2],
            dtype=np.dtype(np.int64),
            shape=(count, 2),
            name=f"fourth {name} objective integers",
        )
        _require_array(
            result[3],
            dtype=np.dtype(np.float64),
            shape=(count, 2),
            name=f"fourth {name} objective floats",
        )
        _require_vector(result[5], f"fourth {name} exact rows")
        _require_vector(result[11], f"fourth {name} feasible order")
        if not isinstance(result[12], str) or not _is_sha256(result[12]):
            raise RuntimeError(f"native three-lane fourth {name} hash is invalid")
        return result

    def reported_objective(
        value: tuple[object, ...],
        plan: int,
        replayed: SolutionObjective,
        name: str,
    ) -> tuple[int, float, float, int]:
        integers = cast(npt.NDArray[np.int64], value[2])
        floats = cast(npt.NDArray[np.float64], value[3])
        reported = SolutionObjective(
            vehicle_count=int(integers[plan, 0]),
            total_distance=float(floats[plan, 0]),
            total_charging_time=float(floats[plan, 1]),
            charging_count=int(integers[plan, 1]),
        )
        if reported.key != replayed.key:
            raise RuntimeError(f"native three-lane fourth {name} objective mismatch")
        return reported.key

    def event(
        operator: str,
        status: str,
        reason: str,
        **changes: object,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "operator": operator,
            "status": status,
            "reason": reason,
            "route_indices": (),
            "affected_route_indices": (),
            "removed_customers": (),
            "candidate_customer_sequence": (),
            "candidate_route_sequences": (),
            "candidate_vehicle_delta": None,
            "candidate_feasible": False,
            "prefilter_passed": False,
            "new_routes_created": 0,
            "exact_route_evaluations": 0,
            "selection_rank": 0,
            "chain_depth": 0,
            "segment_length": 0,
            "track": "legacy",
            "constraint_category": "",
            "removal_tier": "",
            "removal_size_requested": 0,
            "removal_size_actual": 0,
            "stagnation_iterations": 0,
            "removal_trigger": "",
            "reset_observed": False,
            "ranking_score": 0.0,
            "iteration": 3,
            "accepted": False,
            "vehicle_reduction": False,
            "distance_improvement": False,
            "candidate_objective_key": (),
        }
        result.update(changes)
        return result

    prior_legacy = unpack_state(previous_payload[6], "prior legacy")
    prior_quality = unpack_state(previous_payload[7], "prior quality")
    prior_constraint = unpack_state(previous_payload[8], "prior constraint")
    prior_best = unpack_state(previous_payload[9], "prior best")
    prior_legacy_objective = replay(prior_legacy)
    prior_quality_objective = replay(prior_quality)
    prior_constraint_objective = replay(prior_constraint)
    prior_best_objective = replay(prior_best)

    legacy = require_tuple(payload[0], 8, "legacy")
    legacy_metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[0],
            dtype=np.dtype(np.int64),
            shape=(8,),
            name="fourth legacy metadata",
        ),
    )
    if legacy_metadata.tolist()[:5] != [3, 0, 0, 1, 1]:
        raise RuntimeError("native three-lane fourth standard selection is invalid")
    legacy_removed_indices = _require_vector(legacy[1], "fourth legacy removed")
    legacy_removed = tuple(node_names[int(index)] for index in legacy_removed_indices)
    if len(legacy_removed) != 1:
        raise RuntimeError("native three-lane fourth standard removal is invalid")
    unpack_soa(legacy[2], legacy[3], "legacy partial")
    repair = require_tuple(legacy[4], 3, "legacy repair")
    legacy_routes = unpack_soa(repair[0], repair[1], "legacy repaired")
    if legacy[5] is not None:
        raise RuntimeError(
            "native three-lane fourth standard selection was replayed twice"
        )
    insertion_transaction = require_tuple(legacy[6], 13, "insertion transaction")
    insertion_count = len(
        _require_vector(insertion_transaction[1], "fourth insertion statuses")
    )
    transaction(insertion_transaction, insertion_count, "insertion")
    insertion_exact_rows = _require_vector(
        insertion_transaction[5], "fourth insertion exact rows"
    )
    insertion_counters = cast(
        npt.NDArray[np.int64],
        _require_array(
            insertion_transaction[7],
            dtype=np.dtype(np.int64),
            shape=(8,),
            name="fourth insertion counters",
        ),
    )
    if (
        int(insertion_counters[0]) != insertion_count
        or int(insertion_counters[5]) != len(insertion_exact_rows)
        or np.any(insertion_exact_rows < 0)
        or len(set(int(row) for row in insertion_exact_rows))
        != len(insertion_exact_rows)
    ):
        raise RuntimeError("native three-lane fourth insertion charging is invalid")
    selected_insertion = int(legacy_metadata[6])
    legacy_selected = selected_insertion >= 0
    legacy_key: tuple[int, float, float, int] | tuple[()] = ()
    legacy_objective = prior_legacy_objective
    if legacy_selected:
        if selected_insertion >= insertion_count:
            raise RuntimeError("native three-lane fourth insertion selection is invalid")
        legacy_objective = replay(legacy_routes)
        legacy_key = reported_objective(
            insertion_transaction, selected_insertion, legacy_objective, "legacy"
        )
        legacy_acceptance = require_tuple(payload[4], 3, "legacy acceptance")
        if any(value not in (0, 1) for value in legacy_acceptance):
            raise RuntimeError("native three-lane fourth legacy acceptance mismatch")
        legacy_accepted = bool(legacy_acceptance[0])
    else:
        if (
            selected_insertion != -1
            or len(_require_vector(insertion_transaction[11], "insertion feasible"))
            or payload[4] is not None
        ):
            raise RuntimeError("native three-lane fourth insertion rejection is invalid")
        legacy_accepted = False

    quality = require_tuple(payload[2], 7, "quality")
    if (
        not isinstance(quality[0], np.ndarray)
        or quality[0].dtype != np.dtype(np.int64)
        or quality[0].ndim != 2
        or quality[0].shape[1] != 8
        or not quality[0].flags.c_contiguous
    ):
        raise RuntimeError("native three-lane fourth quality attempts are invalid")
    attempts = cast(npt.NDArray[np.int64], quality[0])
    removed_offsets = _require_vector(quality[1], "fourth quality removed offsets")
    removed_indices = _require_vector(quality[2], "fourth quality removed indices")
    if (
        len(removed_offsets) != len(attempts) + 1
        or int(removed_offsets[0]) != 0
        or int(removed_offsets[-1]) != len(removed_indices)
        or np.any(removed_offsets[:-1] > removed_offsets[1:])
    ):
        raise RuntimeError("native three-lane fourth route-segment offsets are invalid")
    quality_outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            quality[5],
            dtype=np.dtype(np.int64),
            shape=(4,),
            name="fourth quality outcome",
        ),
    )
    quality_removed = tuple(
        tuple(
            node_names[int(index)]
            for index in removed_indices[
                int(removed_offsets[row]) : int(removed_offsets[row + 1])
            ]
        )
        for row in range(len(attempts))
    )
    quality_routes = prior_quality
    quality_objective = prior_quality_objective
    quality_key: tuple[int, float, float, int] | tuple[()] = ()
    if len(attempts) == 0:
        if (
            len(removed_offsets) != 1
            or len(removed_indices) != 0
            or quality[3] is not None
            or quality[4] is not None
            or quality_outcome.tolist() != [-1, 0, 0, 0]
        ):
            raise RuntimeError(
                "native three-lane fourth empty quality result is invalid"
            )
    else:
        if (
            np.any(attempts[:, :3] < 0)
            or np.any(attempts[:, 2] == 0)
            or np.any(attempts[:, 3:6] < 0)
            or np.any((attempts[:, 6:] != 0) & (attempts[:, 6:] != 1))
            or np.any(attempts[:-1, 6] != 0)
            or int(attempts[-1, 6]) != 1
            or np.any(removed_offsets[:-1] >= removed_offsets[1:])
        ):
            raise RuntimeError(
                "native three-lane fourth route-segment order is invalid"
            )
        selected_repair = require_tuple(quality[3], 3, "quality repair")
        quality_routes = unpack_soa(
            selected_repair[0], selected_repair[1], "quality repaired"
        )
        quality_transaction = transaction(quality[4], 1, "quality")
        quality_objective = replay(quality_routes)
        quality_key = reported_objective(
            quality_transaction, 0, quality_objective, "quality"
        )
        if (
            int(quality_outcome[0]) != 0
            or np.any((quality_outcome[1:] != 0) & (quality_outcome[1:] != 1))
        ):
            raise RuntimeError(
                "native three-lane fourth quality acceptance mismatch"
            )

    constraint = require_tuple(payload[3], 3, "constraint")
    selection = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[0],
            dtype=np.dtype(np.int64),
            shape=(7,),
            name="fourth constraint selection",
        ),
    )
    probe = require_tuple(constraint[1], 3, "constraint probe")
    removal = require_tuple(probe[0], 7, "constraint removal")
    constraint_repair = require_tuple(probe[1], 3, "constraint repair")
    constraint_transaction = transaction(probe[2], 1, "constraint")
    constraint_outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[2],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="fourth constraint outcome",
        ),
    )
    if (
        int(constraint_outcome[0]) != 3
        or np.any((constraint_outcome[2:] != 0) & (constraint_outcome[2:] != 1))
    ):
        raise RuntimeError("native three-lane fourth constraint decision is invalid")
    constraint_removed_indices = _require_vector(
        removal[2], "fourth constraint removed"
    )
    constraint_removed = tuple(
        node_names[int(index)] for index in constraint_removed_indices
    )
    constraint_partial = unpack_soa(
        removal[0], removal[1], "constraint partial"
    )
    constraint_routes = unpack_soa(
        constraint_repair[0], constraint_repair[1], "constraint repaired"
    )
    constraint_objective = replay(constraint_routes)
    constraint_scores, constraint_route_indices = _constraint_score_vectors(
        removal, constraint_removed_indices, "fourth constraint"
    )
    if len(constraint_route_indices) == 0:
        raise RuntimeError("native three-lane fourth constraint route is missing")
    constraint_affected = tuple(
        index
        for index, (before, after) in enumerate(
            zip(prior_constraint, constraint_routes, strict=False)
        )
        if before != after
    )
    constraint_statuses = _require_vector(
        constraint_transaction[1], "fourth constraint statuses"
    )
    constraint_exact_rows = _require_vector(
        constraint_transaction[5], "fourth constraint exact rows"
    )
    constraint_prepared = bool(constraint_outcome[2])
    constraint_key: tuple[int, float, float, int] | tuple[()] = ()
    if constraint_prepared:
        if constraint_statuses.tolist() != [5]:
            raise RuntimeError(
                "native three-lane fourth prepared constraint is invalid"
            )
        constraint_key = reported_objective(
            constraint_transaction, 0, constraint_objective, "constraint"
        )
    constraint_no_change = (
        not constraint_prepared
        and constraint_statuses.tolist() == [5]
        and len(constraint_exact_rows) == 0
        and constraint_routes == prior_constraint
    )
    constraint_infeasible = (
        not constraint_prepared
        and not constraint_no_change
        and len(_require_vector(constraint_transaction[11], "constraint feasible"))
        == 0
        and np.all(constraint_statuses != 5)
    )
    if not constraint_prepared and not constraint_no_change and not constraint_infeasible:
        raise RuntimeError("native three-lane fourth constraint journal is invalid")

    final_states = (
        unpack_state(payload[6], "final legacy"),
        unpack_state(payload[7], "final quality"),
        unpack_state(payload[8], "final constraint"),
        unpack_state(payload[9], "final best"),
    )
    expected_quality = quality_routes if bool(quality_outcome[1]) else prior_quality
    expected_constraint = (
        constraint_routes if bool(constraint_outcome[3]) else prior_constraint
    )
    expected_legacy = legacy_routes if legacy_accepted else prior_legacy
    expected_states = (
        expected_legacy,
        expected_quality,
        expected_constraint,
        prior_best,
    )
    if (
        final_states != expected_states
        or replay(final_states[3]).key != prior_best_objective.key
    ):
        raise RuntimeError("native three-lane fourth lane state mismatch")

    quality_events = [
        event(
            "route_segment_destroy",
            "failed",
            "route_segment_no_change",
            route_indices=(int(attempt[0]),),
            removed_customers=quality_removed[index],
            candidate_vehicle_delta=0,
            prefilter_passed=True,
            selection_rank=int(attempt[5]),
            segment_length=int(attempt[2]),
            candidate_objective_key=prior_quality_objective.key,
        )
        for index, attempt in enumerate(attempts[:-1])
    ]
    quality_affected = tuple(
        index
        for index, (before, after) in enumerate(
            zip(prior_quality, quality_routes, strict=False)
        )
        if before != after
    )
    if len(attempts) == 0:
        quality_events.append(
            event(
                "route_segment_destroy",
                "failed",
                "no_route_with_segment_length",
            )
        )
    else:
        selected_attempt = attempts[-1]
        quality_events.append(
            event(
                "route_segment_destroy",
                "candidate_proposed",
                "route_segment_repaired",
                route_indices=(int(selected_attempt[0]),),
                affected_route_indices=quality_affected,
                removed_customers=quality_removed[-1],
                candidate_route_sequences=quality_routes,
                candidate_vehicle_delta=len(quality_routes) - len(prior_quality),
                candidate_feasible=True,
                prefilter_passed=True,
                selection_rank=int(selected_attempt[5]),
                segment_length=int(selected_attempt[2]),
                accepted=bool(quality_outcome[1]),
                vehicle_reduction=quality_objective.vehicle_count
                < prior_quality_objective.vehicle_count,
                distance_improvement=quality_objective.total_distance
                < prior_quality_objective.total_distance - 1e-9,
                candidate_objective_key=quality_key,
            )
        )
    constraint_ranked_event = event(
        "shaw_related",
        "candidate_proposed",
        "constraint_ranked_removal",
        route_indices=(int(constraint_route_indices[0]),),
        affected_route_indices=(int(constraint_route_indices[0]),),
        removed_customers=constraint_removed,
        candidate_route_sequences=constraint_partial,
        prefilter_passed=True,
        selection_rank=1,
        track="constraint_lane",
        constraint_category="shaw_related",
        removal_tier="small",
        removal_size_requested=int(selection[1]),
        removal_size_actual=int(selection[2]),
        stagnation_iterations=int(selection[4]),
        removal_trigger="stagnation_baseline",
        reset_observed=bool(selection[6]),
        ranking_score=float(constraint_scores[0]),
        candidate_objective_key=()
        if constraint_no_change
        else constraint_key,
    )
    constraint_result_event = (
        event(
            "shaw_related",
            "failed",
            "constraint_removal_no_change",
            removed_customers=constraint_removed,
            prefilter_passed=True,
            track="constraint_lane",
            constraint_category="shaw_related",
            removal_tier="small",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            stagnation_iterations=int(selection[4]),
            removal_trigger="stagnation_baseline",
            reset_observed=bool(selection[6]),
        )
        if constraint_no_change
        else event(
            "shaw_related",
            "failed",
            "constraint_repair_infeasible",
            affected_route_indices=constraint_affected,
            removed_customers=constraint_removed,
            candidate_route_sequences=constraint_routes,
            prefilter_passed=True,
            exact_route_evaluations=len(constraint_exact_rows),
            track="constraint_lane",
            constraint_category="shaw_related",
            removal_tier="small",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            stagnation_iterations=int(selection[4]),
            removal_trigger="stagnation_baseline",
            reset_observed=bool(selection[6]),
        )
        if constraint_infeasible
        else event(
            "shaw_related",
            "candidate_proposed",
            "constraint_removal_repaired",
            affected_route_indices=constraint_affected,
            removed_customers=constraint_removed,
            candidate_route_sequences=constraint_routes,
            candidate_vehicle_delta=len(constraint_routes) - len(prior_constraint),
            candidate_feasible=True,
            prefilter_passed=True,
            exact_route_evaluations=len(constraint_exact_rows),
            track="constraint_lane",
            constraint_category="shaw_related",
            removal_tier="small",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            stagnation_iterations=int(selection[4]),
            removal_trigger="stagnation_baseline",
            reset_observed=bool(selection[6]),
            accepted=bool(constraint_outcome[3]),
            vehicle_reduction=constraint_objective.vehicle_count
            < prior_constraint_objective.vehicle_count,
            distance_improvement=constraint_objective.total_distance
            < prior_constraint_objective.total_distance - 1e-9,
            candidate_objective_key=constraint_key,
        )
    )
    followup_events = (
        *quality_events,
        constraint_ranked_event,
        constraint_result_event,
        event(
            "standard",
            "proposal",
            "random+regret2",
            removed_customers=legacy_removed,
            _operator_feasible_repair=legacy_selected,
            accepted=legacy_accepted,
            vehicle_reduction=legacy_selected
            and legacy_objective.vehicle_count
            < prior_legacy_objective.vehicle_count,
            distance_improvement=legacy_selected
            and legacy_objective.total_distance
            < prior_legacy_objective.total_distance - 1e-9,
            candidate_objective_key=legacy_key,
        ),
    )
    all_events = (*previous_stream.neighborhood_events, *followup_events)

    stage04 = require_tuple(payload[12], 4, "Stage 4")
    operator_count = len(FULL_NATIVE_OPERATOR_NAMES)
    weights = cast(
        npt.NDArray[np.float64],
        _require_array(
            stage04[0],
            dtype=np.dtype(np.float64),
            shape=(operator_count,),
            name="fourth Stage 4 weights",
        ),
    )
    rewards = cast(
        npt.NDArray[np.float64],
        _require_array(
            stage04[1],
            dtype=np.dtype(np.float64),
            shape=(operator_count,),
            name="fourth Stage 4 rewards",
        ),
    )
    calls = cast(
        npt.NDArray[np.int64],
        _require_array(
            stage04[2],
            dtype=np.dtype(np.int64),
            shape=(operator_count,),
            name="fourth Stage 4 calls",
        ),
    )
    totals = cast(
        npt.NDArray[np.int64],
        _require_array(
            stage04[3],
            dtype=np.dtype(np.int64),
            shape=(operator_count, 8),
            name="fourth Stage 4 totals",
        ),
    )
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="fourth termination",
        ),
    )
    if (
        int(termination[0]) != 0
            or int(termination[1]) != int(previous_stream.termination[1])
        or int(termination[2]) < 0
        or int(termination[3]) != int(termination[2])
        or int(termination[4]) != 0
        or int(termination[5]) != 4
        or np.any(weights <= 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(rewards < 0.0)
        or np.any(calls < 0)
        or np.any(totals < 0)
    ):
        raise RuntimeError("native three-lane fourth final state is invalid")

    activity = np.zeros((operator_count, 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    for item in all_events:
        index = by_name[str(item["operator"])]
        activity[index, 0] = max(
            int(activity[index, 0]), int(bool(item["candidate_feasible"]))
        )
        activity[index, 1] += _semantic_event_aggregate(item, "prefilter_passed")
        activity[index, 2] += cast(int, item["exact_route_evaluations"])
        activity[index, 3] += int(item["status"] == "candidate_proposed")
        activity[index, 4] += _semantic_event_aggregate(item, "candidate_feasible")
        activity[index, 5] += int(bool(item["vehicle_reduction"]))
        activity[index, 6] += int(bool(item["distance_improvement"]))
    activity[by_name["standard"], 0] = int(legacy_selected)
    refinement_index = FULL_NATIVE_OPERATOR_NAMES.index(
        "vehicle_reduction_refinement"
    )
    activity[refinement_index, 5] = 0
    activity[refinement_index, 0] = previous_stream.operator_activity[
        refinement_index, 0
    ]
    activity[refinement_index, 7] = previous_stream.operator_activity[
        refinement_index, 7
    ]
    return NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(all_events),
        operator_weights=_readonly_copy(weights),
        operator_rewards=_readonly_copy(rewards),
        operator_calls=_readonly_copy(calls),
        operator_totals=_readonly_copy(totals),
        operator_activity=_readonly_copy(activity),
        termination=_readonly_copy(termination),
        transaction_sha256=search_sha256,
        initial_temperature=previous_stream.initial_temperature,
    )


def _decode_native_three_lane_fifth_general(
    instance: Instance,
    payload: tuple[object, ...],
    *,
    previous_payload: tuple[object, ...],
    previous_stream: NativeThreeLaneSemanticStream,
    node_names: tuple[str, ...],
    search_sha256: str,
) -> NativeThreeLaneSemanticStream:
    """Replay a real route-elimination/ejection-chain fifth iteration."""

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native three-lane fifth general {name} is invalid")
        return value

    def unpack_soa(
        offsets_value: object,
        indices_value: object,
        name: str,
    ) -> tuple[CustomerSequence, ...]:
        offsets = _require_vector(offsets_value, f"{name} offsets")
        indices = _require_vector(indices_value, f"{name} indices")
        if (
            len(offsets) < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or np.any(offsets[:-1] >= offsets[1:])
            or np.any(indices < 0)
            or np.any(indices >= len(node_names))
        ):
            raise RuntimeError(f"native three-lane fifth general {name} SoA is invalid")
        routes = tuple(
            tuple(
                node_names[int(index)]
                for index in indices[
                    int(offsets[route]) : int(offsets[route + 1])
                ]
            )
            for route in range(len(offsets) - 1)
        )
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(set(flattened)):
            raise RuntimeError(
                f"native three-lane fifth general {name} repeats customers"
            )
        return routes

    def unpack_state(value: object, name: str) -> tuple[CustomerSequence, ...]:
        state = require_tuple(value, 5 if "best" in name else 4, name)
        return unpack_soa(state[0], state[1], name)

    expected_customers = {customer.name for customer in instance.customers}

    def replay(routes: tuple[CustomerSequence, ...]) -> SolutionObjective:
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(expected_customers) or set(flattened) != expected_customers:
            raise RuntimeError(
                "native three-lane fifth general customer identity is invalid"
            )
        objective = SolutionObjective.zero()
        for route in routes:
            exact = solve_exact_charging(instance, route)
            if not exact.feasible:
                raise RuntimeError("native three-lane fifth general replay is infeasible")
            objective += SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
        return objective

    def event(
        operator: str,
        status: str,
        reason: str,
        **changes: object,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "operator": operator,
            "status": status,
            "reason": reason,
            "route_indices": (),
            "affected_route_indices": (),
            "removed_customers": (),
            "candidate_customer_sequence": (),
            "candidate_route_sequences": (),
            "candidate_vehicle_delta": None,
            "candidate_feasible": False,
            "prefilter_passed": False,
            "new_routes_created": 0,
            "exact_route_evaluations": 0,
            "selection_rank": 0,
            "chain_depth": 0,
            "segment_length": 0,
            "track": "legacy",
            "constraint_category": "",
            "removal_tier": "",
            "removal_size_requested": 0,
            "removal_size_actual": 0,
            "stagnation_iterations": 0,
            "removal_trigger": "",
            "reset_observed": False,
            "ranking_score": 0.0,
            "iteration": 4,
            "accepted": False,
            "vehicle_reduction": False,
            "distance_improvement": False,
            "candidate_objective_key": (),
        }
        result.update(changes)
        return result

    prior_legacy = unpack_state(previous_payload[6], "prior legacy")
    prior_quality = unpack_state(previous_payload[7], "prior quality")
    prior_constraint = unpack_state(previous_payload[8], "prior constraint")
    prior_best = unpack_state(previous_payload[9], "prior best")
    prior_legacy_objective = replay(prior_legacy)
    prior_best_objective = replay(prior_best)

    legacy = require_tuple(payload[0], 8, "route elimination")
    profile_order = _require_vector(legacy[0], "fifth general elimination order")
    attempts = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[1],
            dtype=np.dtype(np.int64),
            shape=(len(profile_order), 6),
            name="fifth general elimination attempts",
        ),
    )
    plan_offsets = _require_vector(legacy[2], "fifth general elimination plans")
    route_offsets = _require_vector(legacy[3], "fifth general elimination routes")
    route_indices = _require_vector(legacy[4], "fifth general elimination indices")
    transaction = require_tuple(legacy[5], 13, "elimination transaction")
    outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[6],
            dtype=np.dtype(np.int64),
            shape=(5,),
            name="fifth general elimination outcome",
        ),
    )
    plan_count = len(plan_offsets) - 1
    statuses = _require_vector(transaction[1], "fifth general elimination statuses")
    if (
        len(profile_order) == 0
        or np.any(profile_order < 0)
        or np.any(profile_order >= len(prior_legacy))
        or np.any(attempts[:, 0] != profile_order)
        or np.any(attempts[:, 1] != np.arange(1, len(attempts) + 1))
        or int(plan_offsets[0]) != 0
        or int(plan_offsets[-1]) != len(route_offsets) - 1
        or int(route_offsets[0]) != 0
        or int(route_offsets[-1]) != len(route_indices)
        or len(statuses) != plan_count
        or int(outcome[0]) < 0
        or int(outcome[0]) >= plan_count
        or int(statuses[int(outcome[0])]) != 5
    ):
        raise RuntimeError(
            "native three-lane fifth general elimination journal is invalid"
        )
    selected_plan = int(outcome[0])
    selected_first_route = int(plan_offsets[selected_plan])
    selected_last_route = int(plan_offsets[selected_plan + 1])
    selected_offsets = np.ascontiguousarray(
        route_offsets[selected_first_route : selected_last_route + 1]
        - int(route_offsets[selected_first_route]),
        dtype=np.int64,
    )
    selected_indices = np.ascontiguousarray(
        route_indices[
            int(route_offsets[selected_first_route]) : int(
                route_offsets[selected_last_route]
            )
        ],
        dtype=np.int64,
    )
    selected_routes = unpack_soa(
        selected_offsets, selected_indices, "selected elimination plan"
    )
    selected_objective = replay(selected_routes)
    objective_integers = cast(npt.NDArray[np.int64], transaction[2])
    objective_floats = cast(npt.NDArray[np.float64], transaction[3])
    reported = SolutionObjective(
        int(objective_integers[selected_plan, 0]),
        float(objective_floats[selected_plan, 0]),
        float(objective_floats[selected_plan, 1]),
        int(objective_integers[selected_plan, 1]),
    )
    if reported.key != selected_objective.key:
        raise RuntimeError(
            "native three-lane fifth general elimination objective mismatch"
        )
    selected_source = int(outcome[4])
    if selected_source < 0 or selected_source >= len(prior_legacy):
        raise RuntimeError(
            "native three-lane fifth general elimination source is invalid"
        )
    acceptance = require_tuple(payload[4], 3, "legacy acceptance")
    if any(value not in (0, 1) for value in acceptance):
        raise RuntimeError(
            "native three-lane fifth general elimination acceptance is invalid"
        )
    legacy_accepted = bool(acceptance[0])
    legacy_best = bool(acceptance[1])
    legacy_vehicle_reduction = bool(acceptance[2])
    if (
        not legacy_accepted
        or not legacy_best
        or not legacy_vehicle_reduction
        or selected_objective.vehicle_count >= prior_legacy_objective.vehicle_count
        or selected_objective.key >= prior_best_objective.key
    ):
        raise RuntimeError(
            "native three-lane fifth general elimination decision diverged"
        )
    elimination_events: list[dict[str, object]] = []
    failure_reasons = {1: "no_existing_route_insertion"}
    for attempt in attempts:
        failure = int(attempt[2])
        if failure == 0:
            continue
        if failure not in failure_reasons or int(attempt[3]) != -1:
            raise RuntimeError(
                "native three-lane fifth general elimination failure is unknown"
            )
        source = int(attempt[0])
        elimination_events.append(
            event(
                "route_elimination",
                "failed",
                failure_reasons[failure],
                route_indices=(source,),
                removed_customers=prior_legacy[source],
                selection_rank=int(attempt[1]),
                candidate_objective_key=selected_objective.key,
            )
        )
    elimination_events.append(
        event(
            "route_elimination",
            "candidate_proposed",
            "route_eliminated",
            route_indices=(selected_source,),
            removed_customers=prior_legacy[selected_source],
            candidate_vehicle_delta=-1,
            candidate_feasible=True,
            prefilter_passed=True,
            exact_route_evaluations=len(
                _require_vector(transaction[5], "fifth general elimination exact rows")
            ),
            accepted=True,
            vehicle_reduction=True,
            distance_improvement=selected_objective.total_distance
            < prior_legacy_objective.total_distance - 1e-9,
            candidate_objective_key=selected_objective.key,
        )
    )

    refinement = require_tuple(payload[1], 4, "refinement")
    refinement_metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            refinement[0],
            dtype=np.dtype(np.int64),
            shape=(4,),
            name="fifth general refinement metadata",
        ),
    )
    refinement_removed_indices = _require_vector(
        refinement[1], "fifth general refinement removed"
    )
    refinement_removed = tuple(
        node_names[int(index)] for index in refinement_removed_indices
    )
    refinement_status, refinement_reason, refinement_selected = (
        _native_refinement_outcome(refinement_metadata)
    )
    if refinement_selected:
        raise RuntimeError(
            "native three-lane fifth general selected refinement is not supported"
        )
    refinement_partial = tuple(
        tuple(customer for customer in route if customer not in refinement_removed)
        for route in selected_routes
    )
    refinement_partial = tuple(route for route in refinement_partial if route)
    refinement_event = event(
        "vehicle_reduction_refinement",
        refinement_status,
        refinement_reason,
        affected_route_indices=tuple(
            index
            for index, (before, after) in enumerate(
                zip(prior_legacy, refinement_partial, strict=False)
            )
            if before != after
        ),
        removed_customers=refinement_removed,
        candidate_route_sequences=refinement_partial,
        prefilter_passed=bool(refinement_partial),
        exact_route_evaluations=int(refinement_metadata[2]),
        candidate_objective_key=selected_objective.key,
    )

    quality = require_tuple(payload[2], 4, "quality")
    pool = require_tuple(quality[0], 5, "quality pool")
    quality_plan_offsets = _require_vector(pool[0], "fifth general quality plans")
    quality_route_offsets = _require_vector(pool[1], "fifth general quality routes")
    quality_route_indices = _require_vector(pool[2], "fifth general quality indices")
    change_offsets = _require_vector(pool[3], "fifth general quality changes")
    change_indices = _require_vector(pool[4], "fifth general quality change indices")
    quality_count = len(quality_plan_offsets) - 1
    journal = require_tuple(quality[1], 4, "quality journal")
    quality_statuses = _require_vector(journal[0], "fifth general quality statuses")
    exact_deltas = _require_vector(journal[3], "fifth general quality exact deltas")
    quality_outcome = cast(npt.NDArray[np.int64], quality[2])
    if (
        quality_count <= 0
        or len(quality_statuses) != quality_count
        or np.any(quality_statuses != 0)
        or np.any(exact_deltas != 0)
        or quality_outcome.tolist() != [-1, 0, 0, 0]
        or int(quality_plan_offsets[0]) != 0
        or int(quality_plan_offsets[-1]) != len(quality_route_offsets) - 1
        or int(quality_route_offsets[0]) != 0
        or int(quality_route_offsets[-1]) != len(quality_route_indices)
        or len(change_offsets) != quality_count + 1
    ):
        raise RuntimeError(
            "native three-lane fifth general quality journal is invalid"
        )
    quality_events: list[dict[str, object]] = []
    for candidate in range(quality_count):
        plan = tuple(
            tuple(
                node_names[int(index)]
                for index in quality_route_indices[
                    int(quality_route_offsets[route]) : int(
                        quality_route_offsets[route + 1]
                    )
                ]
            )
            for route in range(
                int(quality_plan_offsets[candidate]),
                int(quality_plan_offsets[candidate + 1]),
            )
        )
        changes = change_indices[
            int(change_offsets[candidate]) : int(change_offsets[candidate + 1])
        ]
        if len(changes) < 2:
            raise RuntimeError(
                "native three-lane fifth general quality change is invalid"
            )
        changed = tuple(int(index) for index in changes[:-1])
        depth = int(changes[-1])
        changed_sequences = tuple(plan[index] for index in changed)
        rejection = None
        for index in changed:
            screen = screen_route_candidate(instance, plan[index], full=True)
            if not screen.accepted:
                rejection = screen
                break
        if rejection is None:
            raise RuntimeError(
                "native three-lane fifth general quality rejection cannot be replayed"
            )
        quality_events.append(
            event(
                "ejection_chain",
                "prefilter_rejected",
                rejection.reason,
                route_indices=changed,
                affected_route_indices=changed,
                candidate_customer_sequence=(
                    changed_sequences[0] if len(changed_sequences) == 1 else ()
                ),
                candidate_route_sequences=changed_sequences,
                chain_depth=depth,
                selection_rank=candidate + 1,
            )
        )
    if payload[3] is not None:
        raise RuntimeError(
            "native three-lane fifth general ran an unscheduled constraint lane"
        )

    final_states = (
        unpack_state(payload[6], "final legacy"),
        unpack_state(payload[7], "final quality"),
        unpack_state(payload[8], "final constraint"),
        unpack_state(payload[9], "final best"),
    )
    if final_states != (
        selected_routes,
        prior_quality,
        prior_constraint,
        selected_routes,
    ) or replay(final_states[3]).key != selected_objective.key:
        raise RuntimeError("native three-lane fifth general lane state mismatch")

    all_events = (
        *previous_stream.neighborhood_events,
        *quality_events,
        *elimination_events,
        refinement_event,
    )
    stage04 = require_tuple(payload[12], 4, "Stage 4")
    operator_count = len(FULL_NATIVE_OPERATOR_NAMES)
    weights = cast(npt.NDArray[np.float64], stage04[0])
    rewards = cast(npt.NDArray[np.float64], stage04[1])
    calls = cast(npt.NDArray[np.int64], stage04[2])
    totals = cast(npt.NDArray[np.int64], stage04[3])
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="fifth general termination",
        ),
    )
    if (
        weights.shape != (operator_count,)
        or rewards.shape != weights.shape
        or calls.shape != weights.shape
        or totals.shape != (operator_count, 8)
        or int(termination[0]) != 0
        or int(termination[5]) != 5
        or np.any(weights <= 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(rewards < 0.0)
        or np.any(calls < 0)
        or np.any(totals < 0)
    ):
        raise RuntimeError("native three-lane fifth general final state is invalid")
    activity = np.zeros((operator_count, 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    grouped: dict[tuple[int, str], list[Mapping[str, object]]] = {}
    for item in all_events:
        operator = str(item["operator"])
        index = by_name[operator]
        activity[index, 1] += _semantic_event_aggregate(item, "prefilter_passed")
        activity[index, 2] += cast(int, item["exact_route_evaluations"])
        activity[index, 3] += int(item["status"] == "candidate_proposed")
        activity[index, 4] += _semantic_event_aggregate(item, "candidate_feasible")
        if operator != "vehicle_reduction_refinement":
            activity[index, 5] += int(bool(item["vehicle_reduction"]))
        activity[index, 6] += int(bool(item["distance_improvement"]))
        grouped.setdefault((cast(int, item["iteration"]), operator), []).append(item)
    for (_, operator), items in grouped.items():
        index = by_name[operator]
        feasible_repair = any(bool(item["candidate_feasible"]) for item in items)
        if operator == "vehicle_reduction_refinement":
            feasible_repair = any(
                bool(item["prefilter_passed"]) and item["status"] != "time_limit"
                for item in items
            )
        activity[index, 0] += int(feasible_repair)
    refinement_index = by_name["vehicle_reduction_refinement"]
    activity[refinement_index, 5] = 0
    activity[refinement_index, 7] = previous_stream.operator_activity[
        refinement_index, 7
    ]
    return NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(all_events),
        operator_weights=_readonly_copy(weights),
        operator_rewards=_readonly_copy(rewards),
        operator_calls=_readonly_copy(calls),
        operator_totals=_readonly_copy(totals),
        operator_activity=_readonly_copy(activity),
        termination=_readonly_copy(termination),
        transaction_sha256=search_sha256,
        initial_temperature=previous_stream.initial_temperature,
    )


def _decode_native_three_lane_fifth_iteration(
    instance: Instance,
    payload: tuple[object, ...],
    *,
    previous_payload: tuple[object, ...],
    previous_stream: NativeThreeLaneSemanticStream,
    node_names: tuple[str, ...],
    search_sha256: str,
) -> NativeThreeLaneSemanticStream:
    """Replay the related/energy and ejection-chain fifth iteration."""

    _verify_native_three_lane_semantic_hash(payload)
    if len(payload) != 14:
        raise RuntimeError("native three-lane fifth iteration is invalid")
    if (
        isinstance(payload[0], tuple)
        and len(payload[0]) == 8
        and isinstance(payload[0][1], np.ndarray)
        and payload[0][1].ndim == 2
    ):
        return _decode_native_three_lane_fifth_general(
            instance,
            payload,
            previous_payload=previous_payload,
            previous_stream=previous_stream,
            node_names=node_names,
            search_sha256=search_sha256,
        )

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native three-lane fifth {name} is invalid")
        return value

    def unpack_soa(
        offsets_value: object,
        indices_value: object,
        name: str,
    ) -> tuple[CustomerSequence, ...]:
        offsets = _require_vector(offsets_value, f"fifth {name} offsets")
        indices = _require_vector(indices_value, f"fifth {name} indices")
        if (
            len(offsets) < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or np.any(offsets[:-1] >= offsets[1:])
            or np.any(indices < 0)
            or np.any(indices >= len(node_names))
        ):
            raise RuntimeError(f"native three-lane fifth {name} SoA is invalid")
        routes = tuple(
            tuple(
                node_names[int(index)]
                for index in indices[int(offsets[route]) : int(offsets[route + 1])]
            )
            for route in range(len(offsets) - 1)
        )
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(set(flattened)):
            raise RuntimeError(f"native three-lane fifth {name} repeats customers")
        return routes

    def unpack_state(value: object, name: str) -> tuple[CustomerSequence, ...]:
        state = require_tuple(value, 5 if "best" in name else 4, name)
        return unpack_soa(state[0], state[1], name)

    expected_customers = {customer.name for customer in instance.customers}

    def replay(routes: tuple[CustomerSequence, ...]) -> SolutionObjective:
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(expected_customers) or set(flattened) != expected_customers:
            raise RuntimeError("native three-lane fifth customer identity is invalid")
        objective = SolutionObjective.zero()
        for route in routes:
            exact = solve_exact_charging(instance, route)
            if not exact.feasible:
                raise RuntimeError("native three-lane fifth replay is infeasible")
            objective += SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
        return objective

    def event(
        operator: str,
        status: str,
        reason: str,
        **changes: object,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "operator": operator,
            "status": status,
            "reason": reason,
            "route_indices": (),
            "affected_route_indices": (),
            "removed_customers": (),
            "candidate_customer_sequence": (),
            "candidate_route_sequences": (),
            "candidate_vehicle_delta": None,
            "candidate_feasible": False,
            "prefilter_passed": False,
            "new_routes_created": 0,
            "exact_route_evaluations": 0,
            "selection_rank": 0,
            "chain_depth": 0,
            "segment_length": 0,
            "track": "legacy",
            "constraint_category": "",
            "removal_tier": "",
            "removal_size_requested": 0,
            "removal_size_actual": 0,
            "stagnation_iterations": 0,
            "removal_trigger": "",
            "reset_observed": False,
            "ranking_score": 0.0,
            "iteration": 4,
            "accepted": False,
            "vehicle_reduction": False,
            "distance_improvement": False,
            "candidate_objective_key": (),
        }
        result.update(changes)
        return result

    prior_legacy = unpack_state(previous_payload[6], "prior legacy")
    prior_quality = unpack_state(previous_payload[7], "prior quality")
    prior_constraint = unpack_state(previous_payload[8], "prior constraint")
    prior_best = unpack_state(previous_payload[9], "prior best")
    prior_legacy_objective = replay(prior_legacy)
    prior_quality_objective = replay(prior_quality)
    prior_best_objective = replay(prior_best)

    if not isinstance(payload[0], tuple) or len(payload[0]) not in (7, 8):
        raise RuntimeError("native three-lane fifth legacy payload is invalid")
    legacy = cast(tuple[object, ...], payload[0])
    legacy_metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[0],
            dtype=np.dtype(np.int64),
            shape=(8,),
            name="fifth legacy metadata",
        ),
    )
    legacy_operator = 1 if len(legacy) == 7 else 0
    valid_standard = (
        legacy_operator == 0
        and int(legacy_metadata[0]) == 4
        and int(legacy_metadata[1]) == 0
        and int(legacy_metadata[2]) in range(3)
        and int(legacy_metadata[3]) in range(3)
        and int(legacy_metadata[4]) > 0
        and int(legacy_metadata[5]) == 0
        and int(legacy_metadata[6]) >= -1
        and int(legacy_metadata[7]) >= 0
    )
    valid_vehicle_repair = (
        legacy_operator == 1
        and int(legacy_metadata[0]) == 4
        and int(legacy_metadata[1]) in (0, 1, 2)
        and int(legacy_metadata[2]) > 0
        and int(legacy_metadata[3]) == 0
        and int(legacy_metadata[4]) == 0
        and int(legacy_metadata[7]) == -2
    )
    if not valid_standard and not valid_vehicle_repair:
        raise RuntimeError("native three-lane fifth legacy selection is invalid")
    legacy_removed_indices = _require_vector(legacy[1], "fifth legacy removed")
    legacy_removed = tuple(node_names[int(index)] for index in legacy_removed_indices)
    expected_removed = int(
        legacy_metadata[4] if legacy_operator == 0 else legacy_metadata[2]
    )
    if len(legacy_removed) != expected_removed:
        raise RuntimeError("native three-lane fifth legacy journal is invalid")
    transaction_index = 6 if legacy_operator == 0 else 5
    state_index = 7 if legacy_operator == 0 else 6
    if legacy_operator == 0 and legacy[5] is not None:
        raise RuntimeError("native three-lane fifth standard journal is invalid")
    transaction_payload = legacy[transaction_index]
    if transaction_payload is None:
        raise RuntimeError("native three-lane fifth legacy transaction is missing")
    insertion = require_tuple(
        transaction_payload, 13, "legacy insertion transaction"
    )
    insertion_statuses = _require_vector(insertion[1], "fifth insertion statuses")
    insertion_integers = cast(
        npt.NDArray[np.int64],
        _require_array(
            insertion[2],
            dtype=np.dtype(np.int64),
            shape=(len(insertion_statuses), 2),
            name="fifth insertion objective integers",
        ),
    )
    insertion_floats = cast(
        npt.NDArray[np.float64],
        _require_array(
            insertion[3],
            dtype=np.dtype(np.float64),
            shape=(len(insertion_statuses), 2),
            name="fifth insertion objective floats",
        ),
    )
    selected_insertion = int(
        legacy_metadata[6] if legacy_operator == 0 else legacy_metadata[5]
    )
    legacy_selected = selected_insertion >= 0
    if (
        selected_insertion >= len(insertion_statuses)
        or (legacy_operator == 1 and selected_insertion != 0)
        or (legacy_selected and int(insertion_statuses[selected_insertion]) != 5)
    ):
        raise RuntimeError("native three-lane fifth attempted-plan replay is invalid")
    legacy_state = unpack_state(legacy[state_index], "legacy embedded state")
    if legacy_state != prior_legacy:
        raise RuntimeError("native three-lane fifth changed rejected legacy state")
    legacy_repair = require_tuple(legacy[4], 3, "legacy repair")
    legacy_routes = unpack_soa(
        legacy_repair[0], legacy_repair[1], "legacy repaired"
    )
    legacy_key: tuple[int, float, float, int] | tuple[()] = ()
    if legacy_selected:
        legacy_objective = replay(legacy_routes)
        reported = SolutionObjective(
            int(insertion_integers[selected_insertion, 0]),
            float(insertion_floats[selected_insertion, 0]),
            float(insertion_floats[selected_insertion, 1]),
            int(insertion_integers[selected_insertion, 1]),
        )
        if reported.key != legacy_objective.key:
            raise RuntimeError("native three-lane fifth legacy objective mismatch")
        legacy_key = reported.key
        legacy_acceptance = require_tuple(payload[4], 3, "legacy acceptance")
        if any(value not in (0, 1) for value in legacy_acceptance):
            raise RuntimeError("native three-lane fifth legacy acceptance is invalid")
        legacy_accepted = bool(legacy_acceptance[0])
        legacy_best = legacy_accepted and legacy_objective.key < prior_best_objective.key
        legacy_vehicle_reduction = (
            legacy_accepted
            and legacy_objective.vehicle_count
            < prior_legacy_objective.vehicle_count
        )
        if (
            bool(legacy_acceptance[1]) != legacy_best
            or bool(legacy_acceptance[2]) != legacy_vehicle_reduction
            or (
                legacy_objective.vehicle_count > prior_legacy_objective.vehicle_count
                and legacy_accepted
            )
            or (legacy_objective.key <= prior_legacy_objective.key and not legacy_accepted)
        ):
            raise RuntimeError("native three-lane fifth legacy acceptance mismatch")
    else:
        if payload[4] is not None:
            raise RuntimeError("native three-lane fifth legacy rejection is invalid")
        legacy_accepted = False
        legacy_best = False
        legacy_vehicle_reduction = False

    quality = require_tuple(payload[2], 4, "quality")
    pool = require_tuple(quality[0], 5, "quality pool")
    plan_offsets = _require_vector(pool[0], "fifth quality plan offsets")
    route_offsets = _require_vector(pool[1], "fifth quality route offsets")
    route_indices = _require_vector(pool[2], "fifth quality route indices")
    change_offsets = _require_vector(pool[3], "fifth quality change offsets")
    change_indices = _require_vector(pool[4], "fifth quality change indices")
    candidate_count = len(plan_offsets) - 1
    if (
        candidate_count <= 0
        or int(plan_offsets[0]) != 0
        or int(plan_offsets[-1]) != len(route_offsets) - 1
        or np.any(plan_offsets[:-1] >= plan_offsets[1:])
        or int(route_offsets[0]) != 0
        or int(route_offsets[-1]) != len(route_indices)
        or np.any(route_offsets[:-1] >= route_offsets[1:])
        or len(change_offsets) != candidate_count + 1
        or int(change_offsets[0]) != 0
        or int(change_offsets[-1]) != len(change_indices)
        or np.any(change_offsets[:-1] >= change_offsets[1:])
    ):
        raise RuntimeError("native three-lane fifth quality pool is invalid")
    candidate_plans = tuple(
        tuple(
            tuple(
                node_names[int(index)]
                for index in route_indices[
                    int(route_offsets[route]) : int(route_offsets[route + 1])
                ]
            )
            for route in range(int(plan_offsets[plan]), int(plan_offsets[plan + 1]))
        )
        for plan in range(candidate_count)
    )
    candidate_changes: list[tuple[tuple[int, ...], int]] = []
    for candidate in range(candidate_count):
        values = change_indices[
            int(change_offsets[candidate]) : int(change_offsets[candidate + 1])
        ]
        if len(values) < 2:
            raise RuntimeError("native three-lane fifth candidate changes are invalid")
        candidate_changes.append(
            (tuple(int(value) for value in values[:-1]), int(values[-1]))
        )
    journal = require_tuple(quality[1], 4, "quality journal")
    statuses = cast(
        npt.NDArray[np.int64],
        _require_array(
            journal[0],
            dtype=np.dtype(np.int64),
            shape=(candidate_count,),
            name="fifth quality statuses",
        ),
    )
    objective_integers = cast(
        npt.NDArray[np.int64],
        _require_array(
            journal[1],
            dtype=np.dtype(np.int64),
            shape=(candidate_count, 2),
            name="fifth quality objective integers",
        ),
    )
    objective_floats = cast(
        npt.NDArray[np.float64],
        _require_array(
            journal[2],
            dtype=np.dtype(np.float64),
            shape=(candidate_count, 2),
            name="fifth quality objective floats",
        ),
    )
    exact_deltas = cast(
        npt.NDArray[np.int64],
        _require_array(
            journal[3],
            dtype=np.dtype(np.int64),
            shape=(candidate_count,),
            name="fifth quality exact deltas",
        ),
    )
    if np.any(~np.isin(statuses, np.asarray([0, 3, 4, 5], dtype=np.int64))):
        raise RuntimeError("native three-lane fifth quality journal is invalid")
    if np.any(exact_deltas < 0) or np.any(
        exact_deltas[np.isin(statuses, np.asarray([0, 3], dtype=np.int64))] != 0
    ):
        raise RuntimeError("native three-lane fifth quality exact journal is invalid")
    candidate_reasons: list[str] = []
    for candidate, plan in enumerate(candidate_plans):
        changed, _depth = candidate_changes[candidate]
        if any(index < 0 or index >= len(plan) for index in changed):
            raise RuntimeError("native three-lane fifth changed route is invalid")
        status = int(statuses[candidate])
        if status == 5:
            objective = replay(plan)
            reported = SolutionObjective(
                int(objective_integers[candidate, 0]),
                float(objective_floats[candidate, 0]),
                float(objective_floats[candidate, 1]),
                int(objective_integers[candidate, 1]),
            )
            if reported.key != objective.key:
                raise RuntimeError("native three-lane fifth quality objective mismatch")
            candidate_reasons.append("exact_charging_feasible")
            continue
        if status == 4:
            candidate_reasons.append("exact_charging_infeasible")
            continue
        if status == 3:
            candidate_reasons.append("candidate_control:round_budget_exhausted")
            continue
        rejection = None
        for index in changed:
            screen = screen_route_candidate(instance, plan[index], full=True)
            if not screen.accepted:
                rejection = screen
                break
        if rejection is None:
            raise RuntimeError(
                "native three-lane fifth screening rejection cannot be replayed"
            )
        candidate_reasons.append(rejection.reason)
    outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            quality[2],
            dtype=np.dtype(np.int64),
            shape=(4,),
            name="fifth quality outcome",
        ),
    )
    if (
        int(outcome[0]) < 0
        or int(outcome[0]) >= candidate_count
        or int(statuses[int(outcome[0])]) != 5
        or any(int(value) not in (0, 1) for value in outcome[1:])
    ):
        raise RuntimeError("native three-lane fifth quality selection is invalid")
    selected_plan = candidate_plans[int(outcome[0])]
    selected_objective = replay(selected_plan)
    selected_key = selected_objective.key
    selected_reported = SolutionObjective(
        int(objective_integers[int(outcome[0]), 0]),
        float(objective_floats[int(outcome[0]), 0]),
        float(objective_floats[int(outcome[0]), 1]),
        int(objective_integers[int(outcome[0]), 1]),
    )
    if selected_reported.key != selected_key:
        raise RuntimeError("native three-lane fifth selected objective is invalid")
    quality_accepted = bool(outcome[1])
    quality_best = quality_accepted and selected_key < prior_best_objective.key
    quality_vehicle_reduction = (
        quality_accepted
        and selected_objective.vehicle_count < prior_quality_objective.vehicle_count
    )
    if (
        bool(outcome[2]) != quality_best
        or bool(outcome[3]) != quality_vehicle_reduction
        or (
            selected_objective.vehicle_count > prior_quality_objective.vehicle_count
            and quality_accepted
        )
        or (selected_key <= prior_quality_objective.key and not quality_accepted)
    ):
        raise RuntimeError("native three-lane fifth quality acceptance mismatch")
    if payload[3] is not None:
        raise RuntimeError("native three-lane fifth ran an unscheduled constraint lane")

    final_states = (
        unpack_state(payload[6], "final legacy"),
        unpack_state(payload[7], "final quality"),
        unpack_state(payload[8], "final constraint"),
        unpack_state(payload[9], "final best"),
    )
    expected_legacy = legacy_routes if legacy_accepted else prior_legacy
    expected_quality = selected_plan if quality_accepted else prior_quality
    expected_best = (
        legacy_routes
        if legacy_best
        else selected_plan
        if quality_best
        else prior_best
    )
    if final_states != (
        expected_legacy,
        expected_quality,
        prior_constraint,
        expected_best,
    ):
        raise RuntimeError("native three-lane fifth lane state mismatch")
    expected_best_objective = (
        legacy_objective
        if legacy_best
        else selected_objective
        if quality_best
        else prior_best_objective
    )
    if replay(final_states[3]).key != expected_best_objective.key:
        raise RuntimeError("native three-lane fifth global best mismatch")

    candidate_events = []
    for candidate, plan in enumerate(candidate_plans):
        changed, depth = candidate_changes[candidate]
        changed_sequences = tuple(plan[index] for index in changed)
        status = int(statuses[candidate])
        reason = candidate_reasons[candidate]
        candidate_feasible = status == 5
        prefilter_passed = status != 0
        event_status = (
            "feasible_candidate"
            if candidate_feasible
            else "exact_infeasible"
            if prefilter_passed
            else "prefilter_rejected"
        )
        candidate_events.append(
            event(
                "ejection_chain",
                event_status,
                reason,
                route_indices=changed,
                affected_route_indices=changed,
                candidate_customer_sequence=(
                    changed_sequences[0] if len(changed_sequences) == 1 else ()
                ),
                candidate_route_sequences=changed_sequences,
                candidate_feasible=candidate_feasible,
                prefilter_passed=prefilter_passed,
                exact_route_evaluations=int(exact_deltas[candidate]),
                selection_rank=candidate + 1,
                chain_depth=depth,
                candidate_objective_key=selected_key,
            )
        )
    selected_changes, selected_depth = candidate_changes[int(outcome[0])]
    followup_events = (
        *candidate_events,
        event(
            "ejection_chain",
            "candidate_proposed",
            "ejection_chain_completed",
            affected_route_indices=selected_changes,
            candidate_route_sequences=selected_plan,
            candidate_vehicle_delta=0,
            candidate_feasible=True,
            prefilter_passed=True,
            chain_depth=selected_depth,
            accepted=quality_accepted,
            vehicle_reduction=quality_vehicle_reduction,
            distance_improvement=selected_objective.total_distance
            < prior_quality_objective.total_distance - 1e-9,
            candidate_objective_key=selected_key,
        ),
        *(
            (
                event(
                    "standard",
                    "proposal",
                    f"{('random', 'worst', 'related')[int(legacy_metadata[2])]}+"
                    f"{('greedy', 'regret2', 'energy')[int(legacy_metadata[3])]}",
                    removed_customers=legacy_removed,
                    _operator_feasible_repair=legacy_selected,
                    candidate_objective_key=legacy_key,
                ),
            )
            if legacy_operator == 0
            else (
                event(
                    "vehicle_count_aware_repair",
                    "candidate_proposed",
                    "existing_route_repair",
                    removed_customers=legacy_removed,
                    _operator_destroy_name=("random", "worst", "related")[
                        int(legacy_metadata[1])
                    ],
                    candidate_vehicle_delta=len(legacy_routes) - len(prior_legacy),
                    candidate_feasible=True,
                    prefilter_passed=bool(
                        len(_require_vector(insertion[5], "fifth legacy exact rows"))
                    ),
                    new_routes_created=int(legacy_metadata[4]),
                    exact_route_evaluations=len(
                        _require_vector(insertion[5], "fifth legacy exact rows")
                    ),
                    accepted=legacy_accepted,
                    vehicle_reduction=legacy_vehicle_reduction,
                    distance_improvement=legacy_objective.total_distance
                    < prior_legacy_objective.total_distance - 1e-9,
                    candidate_objective_key=legacy_key,
                ),
            )
        ),
    )
    all_events = (*previous_stream.neighborhood_events, *followup_events)

    stage04 = require_tuple(payload[12], 4, "Stage 4")
    operator_count = len(FULL_NATIVE_OPERATOR_NAMES)
    weights = cast(
        npt.NDArray[np.float64],
        _require_array(
            stage04[0],
            dtype=np.dtype(np.float64),
            shape=(operator_count,),
            name="fifth weights",
        ),
    )
    rewards = cast(
        npt.NDArray[np.float64],
        _require_array(
            stage04[1],
            dtype=np.dtype(np.float64),
            shape=(operator_count,),
            name="fifth rewards",
        ),
    )
    calls = cast(
        npt.NDArray[np.int64],
        _require_array(
            stage04[2],
            dtype=np.dtype(np.int64),
            shape=(operator_count,),
            name="fifth calls",
        ),
    )
    totals = cast(
        npt.NDArray[np.int64],
        _require_array(
            stage04[3],
            dtype=np.dtype(np.int64),
            shape=(operator_count, 8),
            name="fifth totals",
        ),
    )
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="fifth termination",
        ),
    )
    if (
        int(termination[0]) != 0
        or int(termination[1]) != int(previous_stream.termination[1])
        or int(termination[2]) < 0
        or int(termination[3]) != int(termination[2])
        or int(termination[4]) != 0
        or int(termination[5]) != 5
        or np.any(weights <= 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(rewards < 0.0)
        or np.any(calls < 0)
        or np.any(totals < 0)
    ):
        raise RuntimeError("native three-lane fifth final state is invalid")
    activity = np.zeros((operator_count, 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    for item in all_events:
        index = by_name[str(item["operator"])]
        activity[index, 0] = max(
            int(activity[index, 0]), int(bool(item["candidate_feasible"]))
        )
        activity[index, 1] += _semantic_event_aggregate(item, "prefilter_passed")
        activity[index, 2] += cast(int, item["exact_route_evaluations"])
        activity[index, 3] += int(item["status"] == "candidate_proposed")
        activity[index, 4] += _semantic_event_aggregate(item, "candidate_feasible")
        activity[index, 5] += int(bool(item["vehicle_reduction"]))
        activity[index, 6] += int(bool(item["distance_improvement"]))
    activity[:, 0] = previous_stream.operator_activity[:, 0]
    legacy_operator_name = (
        "standard" if legacy_operator == 0 else "vehicle_count_aware_repair"
    )
    activity[by_name[legacy_operator_name], 0] += int(legacy_selected)
    activity[by_name["ejection_chain"], 0] += int(np.any(statuses == 5))
    refinement_index = by_name["vehicle_reduction_refinement"]
    activity[refinement_index, 5] = 0
    activity[refinement_index, 0] = previous_stream.operator_activity[
        refinement_index, 0
    ]
    activity[refinement_index, 7] = previous_stream.operator_activity[
        refinement_index, 7
    ]
    return NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(all_events),
        operator_weights=_readonly_copy(weights),
        operator_rewards=_readonly_copy(rewards),
        operator_calls=_readonly_copy(calls),
        operator_totals=_readonly_copy(totals),
        operator_activity=_readonly_copy(activity),
        termination=_readonly_copy(termination),
        transaction_sha256=search_sha256,
        initial_temperature=previous_stream.initial_temperature,
    )


def _decode_native_three_lane_sixth_iteration(
    instance: Instance,
    payload: tuple[object, ...],
    *,
    previous_payload: tuple[object, ...],
    previous_stream: NativeThreeLaneSemanticStream,
    node_names: tuple[str, ...],
    search_sha256: str,
    iteration: int,
    completed_iterations: int,
    allow_constraint_no_change: bool = False,
    vehicle_operator_config: VehicleOperatorConfig | None = None,
    termination_reason: int = 0,
) -> NativeThreeLaneSemanticStream:
    """Replay the weighted vehicle-count-aware sixth iteration."""

    _verify_native_three_lane_semantic_hash(payload)
    if len(payload) != 14:
        raise RuntimeError("native three-lane sixth iteration is invalid")

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native three-lane sixth {name} is invalid")
        return value

    def unpack(value: object, name: str) -> tuple[CustomerSequence, ...]:
        state = require_tuple(value, 5 if "best" in name else 4, name)
        offsets = _require_vector(state[0], f"sixth {name} offsets")
        indices = _require_vector(state[1], f"sixth {name} indices")
        if (
            len(offsets) < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or np.any(offsets[:-1] >= offsets[1:])
            or np.any(indices < 0)
            or np.any(indices >= len(node_names))
        ):
            raise RuntimeError(f"native three-lane sixth {name} SoA is invalid")
        return tuple(
            tuple(
                node_names[int(index)]
                for index in indices[int(offsets[route]) : int(offsets[route + 1])]
            )
            for route in range(len(offsets) - 1)
        )

    expected_customers = {customer.name for customer in instance.customers}

    def replay(routes: tuple[CustomerSequence, ...]) -> SolutionObjective:
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(expected_customers) or set(flattened) != expected_customers:
            raise RuntimeError("native three-lane sixth customer identity is invalid")
        objective = SolutionObjective.zero()
        for route in routes:
            exact = solve_exact_charging(instance, route)
            if not exact.feasible:
                raise RuntimeError("native three-lane sixth replay is infeasible")
            objective += SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
        return objective

    prior_states = (
        unpack(previous_payload[6], "prior legacy"),
        unpack(previous_payload[7], "prior quality"),
        unpack(previous_payload[8], "prior constraint"),
        unpack(previous_payload[9], "prior best"),
    )
    if not isinstance(payload[0], tuple) or len(payload[0]) not in (7, 8):
        raise RuntimeError("native three-lane sixth legacy payload is invalid")
    legacy = cast(tuple[object, ...], payload[0])
    metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[0],
            dtype=np.dtype(np.int64),
            shape=(8,),
            name="sixth legacy metadata",
        ),
    )
    operator_id = 1 if len(legacy) == 7 else 0
    valid_standard = (
        operator_id == 0
        and int(metadata[0]) == iteration
        and int(metadata[1]) == 0
        and int(metadata[2]) in range(3)
        and int(metadata[3]) in range(3)
        and int(metadata[4]) > 0
        and int(metadata[5]) == 0
        and int(metadata[6]) >= -1
        and int(metadata[7]) >= 0
    )
    valid_vehicle_repair = (
        operator_id == 1
        and int(metadata[0]) == iteration
        and int(metadata[1]) in (0, 1, 2)
        and int(metadata[2]) > 0
        and int(metadata[3]) == 0
        and int(metadata[4]) == 0
        and int(metadata[5]) >= 0
        and int(metadata[6]) >= 0
        and int(metadata[7]) == -2
    )
    if not valid_standard and not valid_vehicle_repair:
        raise RuntimeError("native three-lane sixth legacy decision is invalid")
    removed_indices = _require_vector(legacy[1], "sixth legacy removed")
    removed = tuple(node_names[int(index)] for index in removed_indices)
    if len(removed) == 0 or len(removed) != int(metadata[4] if operator_id == 0 else metadata[2]):
        raise RuntimeError("native three-lane sixth removal is invalid")
    repair = require_tuple(legacy[4], 3, "legacy repair")
    repaired_state = (repair[0], repair[1], None, None)
    repaired_routes = unpack(repaired_state, "legacy repaired")
    transaction_index = 5 if operator_id == 1 else 6
    transaction = require_tuple(
        legacy[transaction_index], 13, "legacy transaction"
    )
    statuses = _require_vector(transaction[1], "sixth legacy statuses")
    exact_rows = _require_vector(transaction[5], "sixth legacy exact rows")
    transaction_budget = _require_vector(
        transaction[10], "sixth legacy budget state"
    )
    if len(transaction_budget) < 8:
        raise RuntimeError("native three-lane sixth budget state is invalid")
    selected_index = int(metadata[5] if operator_id == 1 else metadata[6])
    candidate_feasible = selected_index >= 0
    if (
        selected_index >= len(statuses)
        or (candidate_feasible and int(statuses[selected_index]) != 5)
        or np.any(exact_rows < 0)
    ):
        raise RuntimeError("native three-lane sixth transaction is invalid")
    main_exact_work = (
        int(metadata[7]) if operator_id == 0 else len(exact_rows)
    )
    expected_budget_state = previous_stream.termination[2:5].astype(
        np.int64, copy=True
    )
    expected_budget_state[0] += main_exact_work
    expected_budget_state[1] += main_exact_work
    if payload[1] is not None:
        refinement_payload = require_tuple(payload[1], 4, "refinement")
        refinement_metadata = cast(
            npt.NDArray[np.int64],
            _require_array(
                refinement_payload[0],
                dtype=np.dtype(np.int64),
                shape=(4,),
                name="sixth refinement metadata",
            ),
        )
        if (
            int(refinement_metadata[0]) not in (0, 1, 2)
            or int(refinement_metadata[1]) not in (0, 1)
            or int(refinement_metadata[2]) < 0
            or int(refinement_metadata[3]) <= 0
        ):
            raise RuntimeError("native three-lane refinement metadata is invalid")
        expected_budget_state[0] += int(refinement_metadata[2])
        expected_budget_state[1] += int(refinement_metadata[2])
    prior_legacy_objective = replay(prior_states[0])
    prior_best_objective = replay(prior_states[3])
    reported: SolutionObjective | None = None
    accepted = False
    new_best = False
    vehicle_reduction = False
    if candidate_feasible:
        repaired_objective = replay(repaired_routes)
        integers = cast(npt.NDArray[np.int64], transaction[2])
        floats = cast(npt.NDArray[np.float64], transaction[3])
        reported = SolutionObjective(
            int(integers[selected_index, 0]),
            float(floats[selected_index, 0]),
            float(floats[selected_index, 1]),
            int(integers[selected_index, 1]),
        )
        if reported.key != repaired_objective.key:
            raise RuntimeError("native three-lane sixth objective mismatch")
        acceptance = require_tuple(payload[4], 3, "legacy acceptance")
        if any(value not in (0, 1) for value in acceptance):
            raise RuntimeError("native three-lane sixth acceptance is invalid")
        accepted = bool(acceptance[0])
        new_best = accepted and reported.key < prior_best_objective.key
        vehicle_reduction = (
            accepted and reported.vehicle_count < prior_legacy_objective.vehicle_count
        )
        if (
            bool(acceptance[1]) != new_best
            or bool(acceptance[2]) != vehicle_reduction
            or (reported.vehicle_count > prior_legacy_objective.vehicle_count and accepted)
            or (reported.key <= prior_legacy_objective.key and not accepted)
        ):
            raise RuntimeError("native three-lane sixth acceptance mismatch")
    elif payload[4] is not None:
        raise RuntimeError("native three-lane sixth rejected acceptance is invalid")
    if payload[2] is not None or (
        payload[3] is not None and not allow_constraint_no_change
    ):
        raise RuntimeError("native three-lane sixth lane schedule is invalid")
    expected_constraint = prior_states[2]
    constraint_improved_best = False
    if payload[3] is not None:
        preliminary_constraint = require_tuple(payload[3], 3, "constraint")
        preliminary_probe = require_tuple(
            preliminary_constraint[1], 3, "constraint probe"
        )
        preliminary_outcome = _require_vector(
            preliminary_constraint[2], "sixth constraint outcome"
        )
        if len(preliminary_outcome) != 6:
            raise RuntimeError("native three-lane sixth constraint outcome is invalid")
        if bool(preliminary_outcome[3]):
            preliminary_repair = require_tuple(
                preliminary_probe[1], 3, "constraint repair"
            )
            expected_constraint = unpack(
                (preliminary_repair[0], preliminary_repair[1], None, None),
                "constraint repaired",
            )
        constraint_improved_best = bool(preliminary_outcome[4])
    final_states = (
        unpack(payload[6], "final legacy"),
        unpack(payload[7], "final quality"),
        unpack(payload[8], "final constraint"),
        unpack(payload[9], "final best"),
    )
    boundary_preview = require_tuple(payload[5], 6, "Stage 4 boundary preview")
    control_preview = _require_vector(
        boundary_preview[4], "sixth Stage 4 control preview"
    )
    if len(control_preview) != 7:
        raise RuntimeError("native three-lane sixth Stage 4 control is invalid")
    expected_best = (
        repaired_routes
        if new_best
        else expected_constraint
        if constraint_improved_best
        else prior_states[3]
    )
    expected_legacy = (
        expected_best
        if bool(control_preview[2])
        else repaired_routes
        if accepted
        else prior_states[0]
    )
    if final_states != (
        expected_legacy,
        prior_states[1],
        expected_constraint,
        expected_best,
    ):
        raise RuntimeError(
            "native three-lane sixth lane state mismatch: "
            f"iteration={iteration}, operator_id={operator_id}, accepted={accepted}, "
            f"new_best={new_best}, constraint_improved_best="
            f"{constraint_improved_best}, expected="
            f"{(expected_legacy, prior_states[1], expected_constraint, expected_best)!r}, "
            f"actual={final_states!r}"
        )

    operator_name = "vehicle_count_aware_repair" if operator_id == 1 else "standard"
    destroy_names = ("random", "worst", "related")
    repair_names = ("greedy", "regret2", "energy")
    item: dict[str, object] = {
        "operator": operator_name,
        "status": "candidate_proposed" if operator_id == 1 else "proposal",
        "reason": (
            "existing_route_repair"
            if operator_id == 1
            else f"{destroy_names[int(metadata[2])]}+{repair_names[int(metadata[3])]}"
        ),
        "route_indices": (),
        "affected_route_indices": (),
        "removed_customers": removed,
        "candidate_customer_sequence": (),
        "candidate_route_sequences": (),
        "candidate_vehicle_delta": (
            len(repaired_routes) - len(prior_states[0]) if operator_id == 1 else None
        ),
        "candidate_feasible": candidate_feasible if operator_id == 1 else False,
        "prefilter_passed": bool(len(exact_rows)) if operator_id == 1 else False,
        "new_routes_created": int(metadata[4]) if operator_id == 1 else 0,
        "exact_route_evaluations": len(exact_rows) if operator_id == 1 else 0,
        "selection_rank": 0,
        "chain_depth": 0,
        "segment_length": 0,
        "track": "legacy",
        "constraint_category": "",
        "removal_tier": "",
        "removal_size_requested": 0,
        "removal_size_actual": 0,
        "stagnation_iterations": 0,
        "removal_trigger": "",
        "reset_observed": False,
        "ranking_score": 0.0,
        "iteration": iteration,
        "_operator_feasible_repair": candidate_feasible,
        "_operator_destroy_name": (
            destroy_names[int(metadata[1])] if operator_id == 1 else ""
        ),
        "accepted": accepted,
        "vehicle_reduction": vehicle_reduction,
        "distance_improvement": (
            reported is not None
            and reported.total_distance < prior_legacy_objective.total_distance - 1e-9
        ),
        "candidate_objective_key": reported.key if reported is not None else (),
        "_operator_comparison": (
            "better"
            if reported is not None
            and reported.key < prior_legacy_objective.key
            else "equal"
            if reported is not None
            and reported.key == prior_legacy_objective.key
            else "worse"
        ),
        "_operator_is_global_best": new_best,
        "_operator_vehicle_reduction": vehicle_reduction,
    }
    constraint_events: tuple[dict[str, object], ...] = ()
    constraint_candidate_prepared = False
    constraint_operator = ""
    if payload[3] is not None:
        if vehicle_operator_config is None:
            raise RuntimeError(
                "native three-lane combined constraint replay lacks its config"
            )
        constraint = require_tuple(payload[3], 3, "constraint")
        selection = cast(
            npt.NDArray[np.int64],
            _require_array(
                constraint[0],
                dtype=np.dtype(np.int64),
                shape=(7,),
                name="sixth constraint selection",
            ),
        )
        probe = require_tuple(constraint[1], 3, "constraint probe")
        removal = require_tuple(probe[0], 7, "constraint removal")
        constraint_repair = require_tuple(probe[1], 3, "constraint repair")
        constraint_transaction = require_tuple(
            probe[2], 13, "constraint transaction"
        )
        constraint_budget = _require_vector(
            constraint_transaction[10], "sixth constraint budget state"
        )
        if len(constraint_budget) < 8:
            raise RuntimeError(
                "native three-lane constraint budget state is invalid"
            )
        outcome = cast(
            npt.NDArray[np.int64],
            _require_array(
                constraint[2],
                dtype=np.dtype(np.int64),
                shape=(6,),
                name="sixth constraint outcome",
            ),
        )
        constraint_candidate_prepared = bool(outcome[2])
        constraint_accepted = bool(outcome[3])
        if (
            int(outcome[0]) not in range(4)
            or not 0 <= int(outcome[1]) < 2**32
            or any(int(value) not in (0, 1) for value in outcome[2:])
            or (constraint_accepted and not constraint_candidate_prepared)
            or ((bool(outcome[4]) or bool(outcome[5])) and not constraint_accepted)
        ):
            raise RuntimeError("native three-lane constraint outcome is invalid")
        constraint_removed_indices = _require_vector(
            removal[2], "sixth constraint removed"
        )
        constraint_removed = tuple(
            node_names[int(index)] for index in constraint_removed_indices
        )
        partial_routes = unpack(
            (removal[0], removal[1], None, None), "constraint partial"
        )
        constraint_routes = unpack(
            (constraint_repair[0], constraint_repair[1], None, None),
            "constraint repaired",
        )
        constraint_statuses = _require_vector(
            constraint_transaction[1], "sixth constraint statuses"
        )
        constraint_exact_rows = _require_vector(
            constraint_transaction[5], "sixth constraint exact rows"
        )
        constraint_status_values = constraint_statuses.tolist()
        constraint_exact_infeasible = constraint_status_values == [4]
        constraint_no_change = constraint_status_values == [5]
        if (
            not (constraint_exact_infeasible or constraint_no_change)
            or np.any(constraint_exact_rows < 0)
            or (
                constraint_no_change
                and not constraint_candidate_prepared
                and constraint_routes != prior_states[2]
            )
        ):
            raise RuntimeError(
                "native three-lane constraint no-change journal is invalid"
            )
        expected_budget_state[0] += len(constraint_exact_rows)
        expected_budget_state[1] += len(constraint_exact_rows)
        constraint_objective: SolutionObjective | None = None
        reported_constraint: SolutionObjective | None = None
        if not constraint_exact_infeasible:
            constraint_integers = cast(
                npt.NDArray[np.int64], constraint_transaction[2]
            )
            constraint_floats = cast(
                npt.NDArray[np.float64], constraint_transaction[3]
            )
            constraint_objective = replay(constraint_routes)
            reported_constraint = SolutionObjective(
                int(constraint_integers[0, 0]),
                float(constraint_floats[0, 0]),
                float(constraint_floats[0, 1]),
                int(constraint_integers[0, 1]),
            )
            if reported_constraint.key != constraint_objective.key:
                raise RuntimeError(
                    "native three-lane constraint no-change objective mismatch"
                )
        scores, removal_routes = _constraint_score_vectors(
            removal,
            constraint_removed_indices,
            "sixth constraint no-change",
        )
        if len(removal_routes) == 0:
            raise RuntimeError(
                "native three-lane constraint no-change route is missing"
            )
        tier_names = ("small", "medium", "large")
        tier_index = int(selection[0])
        if tier_index not in range(len(tier_names)):
            raise RuntimeError("native three-lane constraint tier is invalid")
        stagnation = int(selection[4])
        if stagnation >= vehicle_operator_config.large_stagnation_threshold:
            baseline_tier = 2
            trigger = "large_stagnation"
        elif stagnation >= vehicle_operator_config.medium_stagnation_threshold:
            baseline_tier = 1
            trigger = "medium_stagnation"
        else:
            baseline_tier = 0
            trigger = "stagnation_baseline"
        if tier_index > baseline_tier:
            if (
                iteration <= 0
                or iteration % vehicle_operator_config.exploration_period != 0
                or stagnation
                <= vehicle_operator_config.medium_stagnation_threshold
                or tier_index != baseline_tier + 1
            ):
                raise RuntimeError(
                    "native three-lane constraint promotion is invalid"
                )
            trigger = f"{trigger}+periodic_exploration"
        elif tier_index != baseline_tier:
            raise RuntimeError("native three-lane constraint tier disagrees with config")
        operator_names = (
            "station_pressure",
            "time_window_conflict",
            "worst_energy_detour",
            "shaw_related",
        )
        constraint_operator = operator_names[int(outcome[0])]

        def constraint_event(
            status: str,
            reason: str,
            **changes: object,
        ) -> dict[str, object]:
            result: dict[str, object] = {
                "operator": constraint_operator,
                "status": status,
                "reason": reason,
                "route_indices": (),
                "affected_route_indices": (),
                "removed_customers": constraint_removed,
                "candidate_customer_sequence": (),
                "candidate_route_sequences": (),
                "candidate_vehicle_delta": None,
                "candidate_feasible": False,
                "prefilter_passed": True,
                "new_routes_created": 0,
                "exact_route_evaluations": 0,
                "selection_rank": 0,
                "chain_depth": 0,
                "segment_length": 0,
                "track": "constraint_lane",
                "constraint_category": constraint_operator,
                "removal_tier": tier_names[tier_index],
                "removal_size_requested": int(selection[1]),
                "removal_size_actual": len(constraint_removed),
                "stagnation_iterations": stagnation,
                "removal_trigger": trigger,
                "reset_observed": bool(selection[6]),
                "ranking_score": 0.0,
                "iteration": iteration,
                "accepted": False,
                "vehicle_reduction": False,
                "distance_improvement": False,
                "candidate_objective_key": (),
            }
            result.update(changes)
            return result

        constraint_affected = tuple(
            index
            for index, (before, after) in enumerate(
                zip(prior_states[2], constraint_routes, strict=False)
            )
            if before != after
        )
        constraint_route_indices = tuple(
            sorted(
                {
                    int(route)
                    for route in removal_routes[: len(constraint_removed)]
                }
            )
        )
        constraint_events = (
            constraint_event(
                "candidate_proposed",
                "constraint_ranked_removal",
                route_indices=constraint_route_indices,
                affected_route_indices=constraint_route_indices,
                candidate_route_sequences=partial_routes,
                selection_rank=1,
                ranking_score=float(scores[0]),
                candidate_objective_key=(
                    reported_constraint.key
                    if constraint_candidate_prepared
                    and reported_constraint is not None
                    else ()
                ),
            ),
            (
                constraint_event(
                    "candidate_proposed",
                    "constraint_removal_repaired",
                    affected_route_indices=constraint_affected,
                    candidate_route_sequences=constraint_routes,
                    candidate_vehicle_delta=(
                        len(constraint_routes) - len(prior_states[2])
                    ),
                    candidate_feasible=True,
                    exact_route_evaluations=len(constraint_exact_rows),
                    accepted=constraint_accepted,
                    vehicle_reduction=(
                        constraint_objective is not None
                        and constraint_objective.vehicle_count
                        < replay(prior_states[2]).vehicle_count
                    ),
                    distance_improvement=(
                        constraint_objective is not None
                        and constraint_objective.total_distance
                        < replay(prior_states[2]).total_distance - 1e-9
                    ),
                    candidate_objective_key=(
                        reported_constraint.key
                        if reported_constraint is not None
                        else ()
                    ),
                )
                if constraint_candidate_prepared
                else constraint_event(
                    "failed",
                    (
                        "constraint_repair_infeasible"
                        if constraint_exact_infeasible
                        else "constraint_removal_no_change"
                    ),
                    affected_route_indices=(
                        constraint_affected if constraint_exact_infeasible else ()
                    ),
                    candidate_route_sequences=(
                        constraint_routes if constraint_exact_infeasible else ()
                    ),
                    exact_route_evaluations=(
                        len(constraint_exact_rows)
                        if constraint_exact_infeasible
                        else 0
                    ),
                )
            ),
        )
    all_events = (
        *previous_stream.neighborhood_events,
        *constraint_events,
        item,
    )
    boundary = require_tuple(payload[5], 6, "Stage 4 boundary")
    control = cast(
        npt.NDArray[np.int64],
        _require_array(
            boundary[4],
            dtype=np.dtype(np.int64),
            shape=(7,),
            name="sixth Stage 4 control",
        ),
    )
    control_float = cast(
        npt.NDArray[np.float64],
        _require_array(
            boundary[5],
            dtype=np.dtype(np.float64),
            shape=(1,),
            name="sixth Stage 4 control float",
        ),
    )
    if (
        np.any(control < 0)
        or not math.isfinite(float(control_float[0]))
        or float(control_float[0]) < 0.0
    ):
        raise RuntimeError("native three-lane sixth Stage 4 state is invalid")
    stage04_events = list(previous_stream.stage04_events)
    if bool(control[0]):
        stage04_events.append(
            {
                "type": "stage04_reheat",
                "iteration": iteration,
                "reheat_count": int(control[1]),
                "reheat_floor": float(control_float[0]),
                "stagnation_iterations": int(control[6]),
            }
        )
    if bool(control[2]):
        stage04_events.append(
            {
                "type": "stage04_restart",
                "iteration": iteration,
                "restart_count": int(control[3]),
                "intensification": bool(control[4]),
                "stagnation_at_trigger": 0,
            }
        )
    if (
        previous_stream.stage04_control is not None
        and bool(previous_stream.stage04_control[4])
        and not bool(control[4])
    ):
        stage04_events.append(
            {
                "type": "stage04_intensification_end",
                "iteration": iteration,
            }
        )
    stage04 = require_tuple(payload[12], 4, "Stage 4")
    operator_count = len(FULL_NATIVE_OPERATOR_NAMES)
    weights = cast(
        npt.NDArray[np.float64],
        _require_array(
            stage04[0],
            dtype=np.dtype(np.float64),
            shape=(operator_count,),
            name="sixth weights",
        ),
    )
    rewards = cast(
        npt.NDArray[np.float64],
        _require_array(
            stage04[1],
            dtype=np.dtype(np.float64),
            shape=(operator_count,),
            name="sixth rewards",
        ),
    )
    calls = cast(
        npt.NDArray[np.int64],
        _require_array(
            stage04[2],
            dtype=np.dtype(np.int64),
            shape=(operator_count,),
            name="sixth calls",
        ),
    )
    totals = cast(
        npt.NDArray[np.int64],
        _require_array(
            stage04[3],
            dtype=np.dtype(np.int64),
            shape=(operator_count, 8),
            name="sixth totals",
        ),
    )
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="sixth termination",
        ),
    )
    if (
        int(termination[0]) != termination_reason
        or int(termination[1]) != int(previous_stream.termination[1])
        or not np.array_equal(termination[2:5], expected_budget_state)
        or int(termination[2]) < int(previous_stream.termination[2])
        or int(termination[3]) < int(previous_stream.termination[3])
        or int(termination[4]) < int(previous_stream.termination[4])
        or int(termination[2]) != int(termination[3]) + int(termination[4])
        or int(termination[5]) != completed_iterations
        or np.any(weights <= 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(rewards < 0.0)
        or np.any(calls < 0)
        or np.any(totals < 0)
    ):
        raise RuntimeError(
            "native three-lane sixth final state is invalid: "
            f"termination={termination.tolist()}, "
            f"previous={previous_stream.termination.tolist()}, "
            f"expected_budget_state={expected_budget_state.tolist()}, "
            f"exact_rows={len(exact_rows)}"
        )
    activity = np.zeros((operator_count, 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    for event_item in all_events:
        index = by_name[str(event_item["operator"])]
        activity[index, 0] = max(
            int(activity[index, 0]), int(bool(event_item["candidate_feasible"]))
        )
        activity[index, 1] += _semantic_event_aggregate(
            event_item, "prefilter_passed"
        )
        activity[index, 2] += cast(int, event_item["exact_route_evaluations"])
        activity[index, 3] += int(event_item["status"] == "candidate_proposed")
        activity[index, 4] += _semantic_event_aggregate(
            event_item, "candidate_feasible"
        )
        activity[index, 5] += int(bool(event_item["vehicle_reduction"]))
        activity[index, 6] += int(bool(event_item["distance_improvement"]))
    activity[:, 0] = previous_stream.operator_activity[:, 0]
    operator_index = by_name[operator_name]
    activity[operator_index, 0] += int(candidate_feasible)
    if constraint_operator:
        activity[by_name[constraint_operator], 0] += int(
            constraint_candidate_prepared
        )
    refinement_index = by_name["vehicle_reduction_refinement"]
    activity[refinement_index, 5] = 0
    activity[refinement_index, 0] = previous_stream.operator_activity[
        refinement_index, 0
    ]
    activity[refinement_index, 7] = previous_stream.operator_activity[
        refinement_index, 7
    ]
    return NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(all_events),
        operator_weights=_readonly_copy(weights),
        operator_rewards=_readonly_copy(rewards),
        operator_calls=_readonly_copy(calls),
        operator_totals=_readonly_copy(totals),
        operator_activity=_readonly_copy(activity),
        termination=_readonly_copy(termination),
        transaction_sha256=search_sha256,
        stage04_events=tuple(stage04_events),
        stage04_control=_readonly_copy(control),
        initial_temperature=previous_stream.initial_temperature,
    )


def _decode_native_three_lane_seventh_general(
    instance: Instance,
    payload: tuple[object, ...],
    *,
    previous_payload: tuple[object, ...],
    previous_stream: NativeThreeLaneSemanticStream,
    node_names: tuple[str, ...],
    search_sha256: str,
    iteration: int,
    completed_iterations: int,
    vehicle_operator_config: VehicleOperatorConfig,
) -> NativeThreeLaneSemanticStream:
    """Replay a real all-screened route merge and constraint no-change."""

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native three-lane seventh general {name} is invalid")
        return value

    def unpack_soa(
        offsets_value: object,
        indices_value: object,
        name: str,
    ) -> tuple[CustomerSequence, ...]:
        offsets = _require_vector(offsets_value, f"seventh general {name} offsets")
        indices = _require_vector(indices_value, f"seventh general {name} indices")
        if (
            len(offsets) < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or np.any(offsets[:-1] >= offsets[1:])
            or np.any(indices < 0)
            or np.any(indices >= len(node_names))
        ):
            raise RuntimeError(
                f"native three-lane seventh general {name} SoA is invalid"
            )
        return tuple(
            tuple(
                node_names[int(index)]
                for index in indices[
                    int(offsets[route]) : int(offsets[route + 1])
                ]
            )
            for route in range(len(offsets) - 1)
        )

    def unpack_state(value: object, name: str) -> tuple[CustomerSequence, ...]:
        state = require_tuple(value, 5 if "best" in name else 4, name)
        return unpack_soa(state[0], state[1], name)

    expected_customers = {customer.name for customer in instance.customers}

    def replay(routes: tuple[CustomerSequence, ...]) -> SolutionObjective:
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(expected_customers) or set(flattened) != expected_customers:
            raise RuntimeError(
                "native three-lane seventh general customer identity is invalid"
            )
        objective = SolutionObjective.zero()
        for route in routes:
            exact = solve_exact_charging(instance, route)
            if not exact.feasible:
                raise RuntimeError(
                    "native three-lane seventh general replay is infeasible"
                )
            objective += SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
        return objective

    def event(
        operator: str,
        status: str,
        reason: str,
        **changes: object,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "operator": operator,
            "status": status,
            "reason": reason,
            "route_indices": (),
            "affected_route_indices": (),
            "removed_customers": (),
            "candidate_customer_sequence": (),
            "candidate_route_sequences": (),
            "candidate_vehicle_delta": None,
            "candidate_feasible": False,
            "prefilter_passed": False,
            "new_routes_created": 0,
            "exact_route_evaluations": 0,
            "selection_rank": 0,
            "chain_depth": 0,
            "segment_length": 0,
            "track": "legacy",
            "constraint_category": "",
            "removal_tier": "",
            "removal_size_requested": 0,
            "removal_size_actual": 0,
            "stagnation_iterations": 0,
            "removal_trigger": "",
            "reset_observed": False,
            "ranking_score": 0.0,
            "iteration": iteration,
            "accepted": False,
            "vehicle_reduction": False,
            "distance_improvement": False,
            "candidate_objective_key": (),
        }
        result.update(changes)
        return result

    prior_states = (
        unpack_state(previous_payload[6], "prior legacy"),
        unpack_state(previous_payload[7], "prior quality"),
        unpack_state(previous_payload[8], "prior constraint"),
        unpack_state(previous_payload[9], "prior best"),
    )
    legacy = require_tuple(payload[0], 7, "route merge")
    metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[0],
            dtype=np.dtype(np.int64),
            shape=(4,),
            name="seventh general merge metadata",
        ),
    )
    pool = require_tuple(legacy[1], 4, "merge pool")
    merge_offsets = _require_vector(pool[0], "seventh general merge offsets")
    merge_indices = _require_vector(pool[1], "seventh general merge indices")
    screening_reasons = _require_vector(
        legacy[2], "seventh general merge screening"
    )
    plans = require_tuple(legacy[3], 4, "merge plans")
    if (
        metadata.tolist()
        != [iteration, 3, len(prior_states[0]), len(merge_offsets) - 1]
        or int(merge_offsets[0]) != 0
        or int(merge_offsets[-1]) != len(merge_indices)
        or len(screening_reasons) != len(merge_offsets) - 1
        or np.any(screening_reasons == 0)
        or _require_vector(plans[0], "seventh general plan offsets").tolist()
        != [0]
        or legacy[4] is not None
        or cast(npt.NDArray[np.int64], legacy[5]).tolist() != [-1] * 5
        or unpack_state(legacy[6], "legacy embedded state") != prior_states[0]
    ):
        raise RuntimeError(
            "native three-lane seventh general merge journal is invalid"
        )
    canonical_candidates, reason_counts, prefilter_sha256 = (
        _canonical_route_merge_projection(instance, prior_states[0])
    )
    if canonical_candidates:
        raise RuntimeError(
            "native three-lane seventh general omitted a merge candidate"
        )
    merge_events = tuple(
        event(
            "route_merge",
            "prefilter_rejected_aggregate",
            reason,
            aggregate_count=count,
            candidate_pool_hash=prefilter_sha256,
        )
        for reason, count in sorted(reason_counts.items())
    )

    constraint = require_tuple(payload[3], 3, "constraint")
    selection = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[0],
            dtype=np.dtype(np.int64),
            shape=(7,),
            name="seventh general constraint selection",
        ),
    )
    probe = require_tuple(constraint[1], 3, "constraint probe")
    removal = require_tuple(probe[0], 7, "constraint removal")
    repair = require_tuple(probe[1], 3, "constraint repair")
    transaction = require_tuple(probe[2], 13, "constraint transaction")
    outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[2],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="seventh general constraint outcome",
        ),
    )
    removed_indices = _require_vector(removal[2], "seventh general removed")
    removed = tuple(node_names[int(index)] for index in removed_indices)
    partial_routes = unpack_soa(removal[0], removal[1], "constraint partial")
    repaired_routes = unpack_soa(repair[0], repair[1], "constraint repaired")
    statuses = _require_vector(transaction[1], "seventh general statuses")
    exact_rows = _require_vector(transaction[5], "seventh general exact rows")
    scores, score_routes = _constraint_score_vectors(
        removal, removed_indices, "seventh general constraint"
    )
    route_indices = tuple(
        sorted(
            dict.fromkeys(
                int(value) for value in score_routes[: len(removed_indices)]
            )
        )
    )
    constraint_operator_id = int(outcome[0])
    tier_index = int(selection[0])
    if (
        constraint_operator_id not in range(4)
        or tier_index not in range(3)
        or np.any(outcome[2:] != 0)
        or statuses.tolist() != [5]
        or len(exact_rows) != 0
        or repaired_routes != prior_states[2]
        or not route_indices
    ):
        raise RuntimeError(
            "native three-lane seventh general constraint journal is invalid"
        )
    constraint_operator = (
        "station_pressure",
        "time_window_conflict",
        "worst_energy_detour",
        "shaw_related",
    )[constraint_operator_id]
    stagnation = int(selection[4])
    if stagnation >= vehicle_operator_config.large_stagnation_threshold:
        baseline_tier = 2
        removal_trigger = "large_stagnation"
    elif stagnation >= vehicle_operator_config.medium_stagnation_threshold:
        baseline_tier = 1
        removal_trigger = "medium_stagnation"
    else:
        baseline_tier = 0
        removal_trigger = "stagnation_baseline"
    if tier_index == baseline_tier + 1:
        removal_trigger = f"{removal_trigger}+periodic_exploration"
    elif tier_index != baseline_tier:
        raise RuntimeError(
            "native three-lane seventh general constraint tier is invalid"
        )
    removal_tier = ("small", "medium", "large")[tier_index]
    constraint_events = (
        event(
            constraint_operator,
            "candidate_proposed",
            "constraint_ranked_removal",
            route_indices=route_indices,
            affected_route_indices=route_indices,
            removed_customers=removed,
            candidate_route_sequences=partial_routes,
            prefilter_passed=True,
            selection_rank=1,
            track="constraint_lane",
            constraint_category=constraint_operator,
            removal_tier=removal_tier,
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            stagnation_iterations=int(selection[4]),
            removal_trigger=removal_trigger,
            reset_observed=bool(selection[6]),
            ranking_score=float(scores[0]),
        ),
        event(
            constraint_operator,
            "failed",
            "constraint_removal_no_change",
            removed_customers=removed,
            prefilter_passed=True,
            track="constraint_lane",
            constraint_category=constraint_operator,
            removal_tier=removal_tier,
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            stagnation_iterations=int(selection[4]),
            removal_trigger=removal_trigger,
            reset_observed=bool(selection[6]),
        ),
    )
    if payload[1] is not None or payload[2] is not None or payload[4] is not None:
        raise RuntimeError(
            "native three-lane seventh general lane schedule is invalid"
        )
    final_states = (
        unpack_state(payload[6], "final legacy"),
        unpack_state(payload[7], "final quality"),
        unpack_state(payload[8], "final constraint"),
        unpack_state(payload[9], "final best"),
    )
    if final_states != prior_states or replay(final_states[3]).key != replay(
        prior_states[3]
    ).key:
        raise RuntimeError(
            "native three-lane seventh general lane state mismatch"
        )
    all_events = (
        *previous_stream.neighborhood_events,
        *constraint_events,
        *merge_events,
    )

    boundary = require_tuple(payload[5], 6, "Stage 4 boundary")
    control = cast(npt.NDArray[np.int64], boundary[4])
    control_float = cast(npt.NDArray[np.float64], boundary[5])
    if control.shape != (7,) or control_float.shape != (1,):
        raise RuntimeError(
            "native three-lane seventh general Stage 4 boundary is invalid"
        )
    stage04_events = list(previous_stream.stage04_events)
    if bool(control[0]):
        stage04_events.append(
            {
                "type": "stage04_reheat",
                "iteration": iteration,
                "reheat_count": int(control[1]),
                "reheat_floor": float(control_float[0]),
                "stagnation_iterations": int(control[6]),
            }
        )
    if bool(control[2]):
        stage04_events.append(
            {
                "type": "stage04_restart",
                "iteration": iteration,
                "restart_count": int(control[3]),
                "intensification": bool(control[4]),
                "stagnation_at_trigger": 0,
            }
        )
    stage04 = require_tuple(payload[12], 4, "Stage 4")
    operator_count = len(FULL_NATIVE_OPERATOR_NAMES)
    weights = cast(npt.NDArray[np.float64], stage04[0])
    rewards = cast(npt.NDArray[np.float64], stage04[1])
    calls = cast(npt.NDArray[np.int64], stage04[2])
    totals = cast(npt.NDArray[np.int64], stage04[3])
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="seventh general termination",
        ),
    )
    if (
        weights.shape != (operator_count,)
        or rewards.shape != weights.shape
        or calls.shape != weights.shape
        or totals.shape != (operator_count, 8)
        or termination.tolist()
        != [
            0,
            int(previous_stream.termination[1]),
            int(previous_stream.termination[2]) + len(exact_rows),
            int(previous_stream.termination[3]) + len(exact_rows),
            int(previous_stream.termination[4]),
            completed_iterations,
        ]
    ):
        raise RuntimeError(
            "native three-lane seventh general final state is invalid"
        )
    activity = np.zeros((operator_count, 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    grouped: dict[tuple[int, str], list[Mapping[str, object]]] = {}
    for item in all_events:
        operator = str(item["operator"])
        index = by_name[operator]
        activity[index, 1] += _semantic_event_aggregate(item, "prefilter_passed")
        activity[index, 2] += cast(int, item["exact_route_evaluations"])
        activity[index, 3] += int(item["status"] == "candidate_proposed")
        activity[index, 4] += _semantic_event_aggregate(item, "candidate_feasible")
        if operator != "vehicle_reduction_refinement":
            activity[index, 5] += int(bool(item["vehicle_reduction"]))
        activity[index, 6] += int(bool(item["distance_improvement"]))
        grouped.setdefault((cast(int, item["iteration"]), operator), []).append(item)
    for (_, operator), items in grouped.items():
        index = by_name[operator]
        feasible_repair = any(bool(item["candidate_feasible"]) for item in items)
        if operator == "vehicle_reduction_refinement":
            feasible_repair = any(
                bool(item["prefilter_passed"]) and item["status"] != "time_limit"
                for item in items
            )
        activity[index, 0] += int(feasible_repair)
    refinement_index = by_name["vehicle_reduction_refinement"]
    activity[refinement_index, 5] = 0
    activity[refinement_index, 7] = previous_stream.operator_activity[
        refinement_index, 7
    ]
    return NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(all_events),
        operator_weights=_readonly_copy(weights),
        operator_rewards=_readonly_copy(rewards),
        operator_calls=_readonly_copy(calls),
        operator_totals=_readonly_copy(totals),
        operator_activity=_readonly_copy(activity),
        termination=_readonly_copy(termination),
        transaction_sha256=search_sha256,
        stage04_events=tuple(stage04_events),
        stage04_control=_readonly_copy(control),
        initial_temperature=previous_stream.initial_temperature,
    )


def _decode_native_three_lane_seventh_iteration(
    instance: Instance,
    payload: tuple[object, ...],
    *,
    previous_payload: tuple[object, ...],
    previous_stream: NativeThreeLaneSemanticStream,
    node_names: tuple[str, ...],
    search_sha256: str,
    iteration: int,
    constraint_operator: str,
    constraint_operator_id: int,
    legacy_operator: str,
    legacy_operator_id: int,
    expected_legacy_metadata: tuple[int, ...],
    legacy_reason: str,
    expected_legacy_statuses: tuple[int, ...],
    expected_legacy_exact_rows: tuple[int, ...],
    expected_selection: tuple[int, ...],
    removal_trigger: str,
    completed_iterations: int,
    vehicle_operator_config: VehicleOperatorConfig | None = None,
) -> NativeThreeLaneSemanticStream:
    """Replay periodic constraint exploration and route-merge rejection."""

    _verify_native_three_lane_semantic_hash(payload)
    if len(payload) != 14:
        raise RuntimeError("native three-lane seventh iteration is invalid")
    if isinstance(payload[0], tuple) and len(payload[0]) == 7:
        if vehicle_operator_config is None:
            raise RuntimeError(
                "native three-lane seventh general replay lacks its config"
            )
        return _decode_native_three_lane_seventh_general(
            instance,
            payload,
            previous_payload=previous_payload,
            previous_stream=previous_stream,
            node_names=node_names,
            search_sha256=search_sha256,
            iteration=iteration,
            completed_iterations=completed_iterations,
            vehicle_operator_config=vehicle_operator_config,
        )

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native three-lane seventh {name} is invalid")
        return value

    def unpack_soa(
        offsets_value: object,
        indices_value: object,
        name: str,
    ) -> tuple[CustomerSequence, ...]:
        offsets = _require_vector(offsets_value, f"seventh {name} offsets")
        indices = _require_vector(indices_value, f"seventh {name} indices")
        if (
            len(offsets) < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or np.any(offsets[:-1] >= offsets[1:])
            or np.any(indices < 0)
            or np.any(indices >= len(node_names))
        ):
            raise RuntimeError(f"native three-lane seventh {name} SoA is invalid")
        return tuple(
            tuple(
                node_names[int(index)]
                for index in indices[
                    int(offsets[route]) : int(offsets[route + 1])
                ]
            )
            for route in range(len(offsets) - 1)
        )

    def unpack_state(value: object, name: str) -> tuple[CustomerSequence, ...]:
        state = require_tuple(value, 5 if "best" in name else 4, name)
        return unpack_soa(state[0], state[1], name)

    expected_customers = {customer.name for customer in instance.customers}

    def replay(routes: tuple[CustomerSequence, ...]) -> SolutionObjective:
        flattened = tuple(customer for route in routes for customer in route)
        if (
            len(flattened) != len(expected_customers)
            or set(flattened) != expected_customers
        ):
            raise RuntimeError("native three-lane seventh customer identity is invalid")
        objective = SolutionObjective.zero()
        for route in routes:
            exact = solve_exact_charging(instance, route)
            if not exact.feasible:
                raise RuntimeError("native three-lane seventh replay is infeasible")
            objective += SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
        return objective

    def transaction(value: object, name: str) -> tuple[object, ...]:
        result = require_tuple(value, 13, f"{name} transaction")
        _require_array(
            result[1],
            dtype=np.dtype(np.int64),
            shape=(1,),
            name=f"seventh {name} statuses",
        )
        _require_array(
            result[2],
            dtype=np.dtype(np.int64),
            shape=(1, 2),
            name=f"seventh {name} objective integers",
        )
        _require_array(
            result[3],
            dtype=np.dtype(np.float64),
            shape=(1, 2),
            name=f"seventh {name} objective floats",
        )
        _require_vector(result[5], f"seventh {name} exact rows")
        if not isinstance(result[12], str) or not _is_sha256(result[12]):
            raise RuntimeError(f"native three-lane seventh {name} hash is invalid")
        return result

    def reported_objective(
        value: tuple[object, ...],
        replayed: SolutionObjective,
        name: str,
    ) -> tuple[int, float, float, int]:
        integers = cast(npt.NDArray[np.int64], value[2])
        floats = cast(npt.NDArray[np.float64], value[3])
        reported = SolutionObjective(
            vehicle_count=int(integers[0, 0]),
            total_distance=float(floats[0, 0]),
            total_charging_time=float(floats[0, 1]),
            charging_count=int(integers[0, 1]),
        )
        if reported.key != replayed.key:
            raise RuntimeError(f"native three-lane seventh {name} objective mismatch")
        return reported.key

    def event(
        operator: str,
        status: str,
        reason: str,
        **changes: object,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "operator": operator,
            "status": status,
            "reason": reason,
            "route_indices": (),
            "affected_route_indices": (),
            "removed_customers": (),
            "candidate_customer_sequence": (),
            "candidate_route_sequences": (),
            "candidate_vehicle_delta": None,
            "candidate_feasible": False,
            "prefilter_passed": False,
            "new_routes_created": 0,
            "exact_route_evaluations": 0,
            "selection_rank": 0,
            "chain_depth": 0,
            "segment_length": 0,
            "track": "legacy",
            "constraint_category": "",
            "removal_tier": "",
            "removal_size_requested": 0,
            "removal_size_actual": 0,
            "stagnation_iterations": 0,
            "removal_trigger": "",
            "reset_observed": False,
            "ranking_score": 0.0,
            "iteration": iteration,
            "accepted": False,
            "vehicle_reduction": False,
            "distance_improvement": False,
            "candidate_objective_key": (),
        }
        result.update(changes)
        return result

    prior_states = (
        unpack_state(previous_payload[6], "prior legacy"),
        unpack_state(previous_payload[7], "prior quality"),
        unpack_state(previous_payload[8], "prior constraint"),
        unpack_state(previous_payload[9], "prior best"),
    )
    legacy = require_tuple(
        payload[0], 8 if legacy_operator == "standard" else 2, "legacy"
    )
    legacy_metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[0],
            dtype=np.dtype(np.int64),
            shape=(len(expected_legacy_metadata),),
            name="seventh legacy metadata",
        ),
    )
    if legacy_metadata.tolist() != list(expected_legacy_metadata):
        raise RuntimeError("native three-lane seventh route-merge decision is invalid")
    legacy_removed: tuple[str, ...] = ()
    if legacy_operator == "standard":
        legacy_removed = tuple(
            node_names[int(index)]
            for index in _require_vector(legacy[1], "seventh legacy removed")
        )
        legacy_transaction = require_tuple(
            legacy[6], 13, "legacy insertion transaction"
        )
        legacy_statuses = _require_vector(
            legacy_transaction[1], "legacy statuses"
        )
        legacy_exact_rows = _require_vector(
            legacy_transaction[5], "legacy exact rows"
        )
        if (
            legacy[5] is not None
            or len(legacy_statuses) == 0
            or np.any(legacy_statuses < 0)
            or np.any(legacy_statuses > 5)
            or np.any(legacy_exact_rows < 0)
            or len(set(int(row) for row in legacy_exact_rows))
            != len(legacy_exact_rows)
            or len(_require_vector(legacy_transaction[11], "legacy attempted"))
        ):
            raise RuntimeError("native three-lane seventh legacy journal is invalid")
        legacy_state = unpack_state(legacy[7], "legacy state")
    else:
        legacy_state = unpack_state(legacy[1], "legacy state")
    if legacy_state != prior_states[0]:
        raise RuntimeError("native three-lane seventh legacy state is invalid")

    constraint = require_tuple(payload[3], 3, "constraint")
    selection = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[0],
            dtype=np.dtype(np.int64),
            shape=(7,),
            name="seventh constraint selection",
        ),
    )
    if selection.tolist() != list(expected_selection):
        raise RuntimeError("native three-lane seventh stagnation state is invalid")
    probe = require_tuple(constraint[1], 3, "constraint probe")
    removal = require_tuple(probe[0], 7, "constraint removal")
    repair = require_tuple(probe[1], 3, "constraint repair")
    constraint_transaction = transaction(probe[2], "constraint")
    outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[2],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="seventh constraint outcome",
        ),
    )
    if (
        int(outcome[0]) != constraint_operator_id
        or not 0 <= int(outcome[1]) < 2**32
        or outcome[2:4].tolist() != [1, 1]
        or any(int(value) not in (0, 1) for value in outcome[4:])
    ):
        raise RuntimeError("native three-lane seventh constraint decision is invalid")
    removed_indices = _require_vector(removal[2], "seventh constraint removed")
    removed = tuple(node_names[int(index)] for index in removed_indices)
    partial_routes = unpack_soa(removal[0], removal[1], "constraint partial")
    repaired_routes = unpack_soa(repair[0], repair[1], "constraint repaired")
    constraint_objective = replay(repaired_routes)
    constraint_key = reported_objective(
        constraint_transaction,
        constraint_objective,
        "constraint",
    )
    prior_constraint_objective = replay(prior_states[2])
    prior_best_objective = replay(prior_states[3])
    expected_constraint_best = constraint_objective.key < prior_best_objective.key
    expected_vehicle_reduction = (
        constraint_objective.vehicle_count
        < prior_constraint_objective.vehicle_count
    )
    if (
        bool(outcome[4]) != expected_constraint_best
        or bool(outcome[5]) != expected_vehicle_reduction
    ):
        raise RuntimeError(
            "native three-lane seventh constraint objective flags are invalid"
        )
    statuses = _require_vector(
        constraint_transaction[1], "seventh constraint statuses"
    )
    exact_rows = _require_vector(
        constraint_transaction[5], "seventh constraint exact rows"
    )
    if (
        statuses.tolist() != [5]
        or np.any(exact_rows < 0)
        or len(set(int(row) for row in exact_rows)) != len(exact_rows)
    ):
        raise RuntimeError("native three-lane seventh cache transaction is invalid")
    scores, route_indices = _constraint_score_vectors(
        removal, removed_indices, "seventh constraint"
    )
    if len(route_indices) == 0:
        raise RuntimeError("native three-lane seventh constraint route is missing")
    affected = tuple(
        index
        for index, (before, after) in enumerate(
            zip(prior_states[2], repaired_routes, strict=False)
        )
        if before != after
    )

    if payload[1] is not None or payload[2] is not None or payload[4] is not None:
        raise RuntimeError("native three-lane seventh lane schedule is invalid")
    final_states = (
        unpack_state(payload[6], "final legacy"),
        unpack_state(payload[7], "final quality"),
        unpack_state(payload[8], "final constraint"),
        unpack_state(payload[9], "final best"),
    )
    if final_states != (
        prior_states[0],
        prior_states[1],
        repaired_routes,
        repaired_routes if bool(outcome[4]) else prior_states[3],
    ):
        raise RuntimeError("native three-lane seventh lane state mismatch")

    followup_events = (
        event(
            constraint_operator,
            "candidate_proposed",
            "constraint_ranked_removal",
            route_indices=(int(route_indices[0]),),
            affected_route_indices=(int(route_indices[0]),),
            removed_customers=removed,
            candidate_route_sequences=partial_routes,
            prefilter_passed=True,
            selection_rank=1,
            track="constraint_lane",
            constraint_category=constraint_operator,
            removal_tier="large",
            removal_size_requested=int(selection[1]),
            removal_size_actual=len(removed),
            stagnation_iterations=int(selection[4]),
            removal_trigger=removal_trigger,
            reset_observed=bool(selection[6]),
            ranking_score=float(scores[0]),
            candidate_objective_key=constraint_key,
        ),
        event(
            constraint_operator,
            "candidate_proposed",
            "constraint_removal_repaired",
            affected_route_indices=affected,
            removed_customers=removed,
            candidate_route_sequences=repaired_routes,
            candidate_vehicle_delta=len(repaired_routes) - len(prior_states[2]),
            candidate_feasible=True,
            prefilter_passed=True,
            exact_route_evaluations=len(exact_rows),
            track="constraint_lane",
            constraint_category=constraint_operator,
            removal_tier="large",
            removal_size_requested=int(selection[1]),
            removal_size_actual=len(removed),
            stagnation_iterations=int(selection[4]),
            removal_trigger=removal_trigger,
            reset_observed=bool(selection[6]),
            accepted=True,
            vehicle_reduction=constraint_objective.vehicle_count
            < replay(prior_states[2]).vehicle_count,
            distance_improvement=constraint_objective.total_distance
            < replay(prior_states[2]).total_distance - 1e-9,
            candidate_objective_key=constraint_key,
        ),
        event(
            legacy_operator,
            "proposal" if legacy_operator == "standard" else "not_applicable",
            legacy_reason,
            removed_customers=legacy_removed,
        ),
    )
    all_events = (*previous_stream.neighborhood_events, *followup_events)

    boundary = require_tuple(payload[5], 6, "Stage 4 boundary")
    control = cast(
        npt.NDArray[np.int64],
        _require_array(
            boundary[4],
            dtype=np.dtype(np.int64),
            shape=(7,),
            name="seventh Stage 4 control",
        ),
    )
    control_float = cast(
        npt.NDArray[np.float64],
        _require_array(
            boundary[5],
            dtype=np.dtype(np.float64),
            shape=(1,),
            name="seventh Stage 4 control float",
        ),
    )
    if (
        np.any(control < 0)
        or not math.isfinite(float(control_float[0]))
        or float(control_float[0]) < 0.0
    ):
        raise RuntimeError("native three-lane seventh Stage 4 state is invalid")
    stage04_events = list(previous_stream.stage04_events)
    if bool(control[0]):
        stage04_events.append(
            {
                "type": "stage04_reheat",
                "iteration": iteration,
                "reheat_count": int(control[1]),
                "reheat_floor": float(control_float[0]),
                "stagnation_iterations": int(control[6]),
            }
        )
    if bool(control[2]):
        stage04_events.append(
            {
                "type": "stage04_restart",
                "iteration": iteration,
                "restart_count": int(control[3]),
                "intensification": bool(control[4]),
                "stagnation_at_trigger": 0,
            }
        )

    stage04 = require_tuple(payload[12], 4, "Stage 4")
    operator_count = len(FULL_NATIVE_OPERATOR_NAMES)
    weights = cast(
        npt.NDArray[np.float64],
        _require_array(
            stage04[0],
            dtype=np.dtype(np.float64),
            shape=(operator_count,),
            name="seventh weights",
        ),
    )
    rewards = cast(
        npt.NDArray[np.float64],
        _require_array(
            stage04[1],
            dtype=np.dtype(np.float64),
            shape=(operator_count,),
            name="seventh rewards",
        ),
    )
    calls = cast(
        npt.NDArray[np.int64],
        _require_array(
            stage04[2],
            dtype=np.dtype(np.int64),
            shape=(operator_count,),
            name="seventh calls",
        ),
    )
    totals = cast(
        npt.NDArray[np.int64],
        _require_array(
            stage04[3],
            dtype=np.dtype(np.int64),
            shape=(operator_count, 8),
            name="seventh totals",
        ),
    )
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="seventh termination",
        ),
    )
    if (
        termination.tolist()
        != [
            0,
            int(previous_stream.termination[1]),
            int(previous_stream.termination[2]) + len(exact_rows),
            int(previous_stream.termination[3]) + len(exact_rows),
            int(previous_stream.termination[4]),
            completed_iterations,
        ]
        or np.any(weights <= 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(rewards < 0.0)
        or np.any(calls < 0)
        or np.any(totals < 0)
    ):
        raise RuntimeError("native three-lane seventh final state is invalid")

    activity = np.zeros((operator_count, 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    for event_item in all_events:
        index = by_name[str(event_item["operator"])]
        activity[index, 0] = max(
            int(activity[index, 0]), int(bool(event_item["candidate_feasible"]))
        )
        activity[index, 1] += _semantic_event_aggregate(
            event_item, "prefilter_passed"
        )
        activity[index, 2] += cast(int, event_item["exact_route_evaluations"])
        activity[index, 3] += int(event_item["status"] == "candidate_proposed")
        activity[index, 4] += _semantic_event_aggregate(
            event_item, "candidate_feasible"
        )
        activity[index, 5] += int(bool(event_item["vehicle_reduction"]))
        activity[index, 6] += int(bool(event_item["distance_improvement"]))
    activity[:, 0] = previous_stream.operator_activity[:, 0]
    constraint_index = by_name[constraint_operator]
    activity[constraint_index, 0] = (
        previous_stream.operator_activity[constraint_index, 0] + 1
    )
    refinement_index = by_name["vehicle_reduction_refinement"]
    activity[refinement_index, 5] = 0
    activity[refinement_index, 0] = previous_stream.operator_activity[
        refinement_index, 0
    ]
    activity[refinement_index, 7] = previous_stream.operator_activity[
        refinement_index, 7
    ]
    return NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(all_events),
        operator_weights=_readonly_copy(weights),
        operator_rewards=_readonly_copy(rewards),
        operator_calls=_readonly_copy(calls),
        operator_totals=_readonly_copy(totals),
        operator_activity=_readonly_copy(activity),
        termination=_readonly_copy(termination),
        transaction_sha256=search_sha256,
        stage04_events=tuple(stage04_events),
        stage04_control=_readonly_copy(control),
        initial_temperature=previous_stream.initial_temperature,
    )


def _decode_native_three_lane_rejection_only_iteration(
    payload: tuple[object, ...],
    *,
    previous_payload: tuple[object, ...],
    previous_stream: NativeThreeLaneSemanticStream,
    node_names: tuple[str, ...],
    search_sha256: str,
    iteration: int,
    operator: str,
    operator_id: int,
    completed_iterations: int,
    restart_to_best: bool = False,
    instance: Instance | None = None,
    termination_reason: int = 0,
) -> NativeThreeLaneSemanticStream:
    """Replay a weighted one-route operator rejection without hidden work."""

    _verify_native_three_lane_semantic_hash(payload)
    if len(payload) != 14:
        raise RuntimeError("native three-lane rejection iteration is invalid")

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native three-lane rejection {name} is invalid")
        return value

    def unpack(value: object, name: str) -> tuple[CustomerSequence, ...]:
        state = require_tuple(value, 5 if "best" in name else 4, name)
        offsets = _require_vector(state[0], f"rejection {name} offsets")
        indices = _require_vector(state[1], f"rejection {name} indices")
        if (
            len(offsets) < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or np.any(offsets[:-1] >= offsets[1:])
            or np.any(indices < 0)
            or np.any(indices >= len(node_names))
        ):
            raise RuntimeError(f"native three-lane rejection {name} SoA is invalid")
        return tuple(
            tuple(
                node_names[int(index)]
                for index in indices[
                    int(offsets[route]) : int(offsets[route + 1])
                ]
            )
            for route in range(len(offsets) - 1)
        )

    prior_states = (
        unpack(previous_payload[6], "prior legacy"),
        unpack(previous_payload[7], "prior quality"),
        unpack(previous_payload[8], "prior constraint"),
        unpack(previous_payload[9], "prior best"),
    )
    aggregate_merge: tuple[tuple[str, int], ...] | None = None
    route_elimination_events: tuple[dict[str, object], ...] | None = None
    merge_pool_sha256 = ""
    if isinstance(payload[0], tuple) and len(payload[0]) == 7:
        if instance is None or operator != "route_merge" or operator_id != 3:
            raise RuntimeError(
                "native three-lane aggregate merge lacks its replay context"
            )
        legacy = cast(tuple[object, ...], payload[0])
        metadata = cast(
            npt.NDArray[np.int64],
            _require_array(
                legacy[0],
                dtype=np.dtype(np.int64),
                shape=(4,),
                name="rejection aggregate-merge metadata",
            ),
        )
        pool = require_tuple(legacy[1], 4, "aggregate merge pool")
        merge_offsets = _require_vector(pool[0], "aggregate merge offsets")
        merge_indices = _require_vector(pool[1], "aggregate merge indices")
        screening_reasons = _require_vector(
            legacy[2], "aggregate merge screening reasons"
        )
        plans = require_tuple(legacy[3], 4, "aggregate merge plans")
        if (
            metadata.tolist()
            != [iteration, operator_id, len(prior_states[0]), len(merge_offsets) - 1]
            or int(merge_offsets[0]) != 0
            or int(merge_offsets[-1]) != len(merge_indices)
            or len(screening_reasons) != len(merge_offsets) - 1
            or np.any(screening_reasons == 0)
            or _require_vector(plans[0], "aggregate merge plan offsets").tolist()
            != [0]
            or legacy[4] is not None
            or _require_vector(legacy[5], "aggregate merge outcome").tolist()
            != [-1] * 5
            or unpack(legacy[6], "aggregate merge state") != prior_states[0]
        ):
            raise RuntimeError(
                "native three-lane aggregate merge journal is invalid"
            )
        canonical_candidates, reason_counts, merge_pool_sha256 = (
            _canonical_route_merge_projection(instance, prior_states[0])
        )
        if canonical_candidates:
            raise RuntimeError(
                "native three-lane aggregate merge omitted a candidate"
            )
        aggregate_merge = tuple(sorted(reason_counts.items()))
    elif isinstance(payload[0], tuple) and len(payload[0]) == 8:
        if operator != "route_elimination" or operator_id != 2:
            raise RuntimeError(
                "native three-lane route elimination identity is invalid"
            )
        legacy = cast(tuple[object, ...], payload[0])
        profile_order = _require_vector(
            legacy[0], "rejection route-elimination order"
        )
        attempts = cast(
            npt.NDArray[np.int64],
            _require_array(
                legacy[1],
                dtype=np.dtype(np.int64),
                shape=(len(profile_order), 6),
                name="rejection route-elimination attempts",
            ),
        )
        plan_offsets = _require_vector(
            legacy[2], "rejection route-elimination plan offsets"
        )
        route_offsets = _require_vector(
            legacy[3], "rejection route-elimination route offsets"
        )
        route_indices = _require_vector(
            legacy[4], "rejection route-elimination route indices"
        )
        outcome = _require_vector(
            legacy[6], "rejection route-elimination outcome"
        )
        if (
            not np.array_equal(profile_order, attempts[:, 0])
            or np.any(attempts[:, 0] < 0)
            or np.any(attempts[:, 0] >= len(prior_states[0]))
            or np.any(attempts[:, 2] == 0)
            or plan_offsets.tolist() != [0]
            or route_offsets.tolist() != [0]
            or len(route_indices) != 0
            or legacy[5] is not None
            or outcome.tolist() != [-1] * 5
            or unpack(legacy[7], "route-elimination state") != prior_states[0]
        ):
            raise RuntimeError(
                "native three-lane route-elimination journal is invalid"
            )
        route_elimination_events = tuple(
            {
                "operator": "route_elimination",
                "status": "failed",
                "reason": (
                    "no_existing_route_insertion"
                    if int(attempt[2]) == 1
                    else "singleton_route_screening_rejected"
                ),
                "route_indices": (int(attempt[0]),),
                "affected_route_indices": (),
                "removed_customers": prior_states[0][int(attempt[0])],
                "candidate_customer_sequence": (),
                "candidate_route_sequences": (),
                "candidate_vehicle_delta": None,
                "candidate_feasible": False,
                "prefilter_passed": False,
                "new_routes_created": 0,
                "exact_route_evaluations": 0,
                "selection_rank": int(attempt[1]),
                "chain_depth": 0,
                "segment_length": 0,
                "track": "legacy",
                "constraint_category": "",
                "removal_tier": "",
                "removal_size_requested": 0,
                "removal_size_actual": 0,
                "stagnation_iterations": 0,
                "removal_trigger": "",
                "reset_observed": False,
                "ranking_score": 0.0,
                "iteration": iteration,
                "accepted": False,
                "vehicle_reduction": False,
                "distance_improvement": False,
                "candidate_objective_key": (),
            }
            for attempt in attempts
        )
    else:
        legacy = require_tuple(payload[0], 2, "legacy")
        metadata = cast(
            npt.NDArray[np.int64],
            _require_array(
                legacy[0],
                dtype=np.dtype(np.int64),
                shape=(4,),
                name="rejection legacy metadata",
            ),
        )
        if metadata.tolist() != [iteration, operator_id, 1, 0]:
            raise RuntimeError("native three-lane rejection decision is invalid")
        if unpack(legacy[1], "legacy state") != prior_states[0]:
            raise RuntimeError("native three-lane rejection changed its legacy state")
    if any(payload[index] is not None for index in (1, 2, 3, 4)):
        raise RuntimeError("native three-lane rejection lane schedule is invalid")
    final_states = (
        unpack(payload[6], "final legacy"),
        unpack(payload[7], "final quality"),
        unpack(payload[8], "final constraint"),
        unpack(payload[9], "final best"),
    )
    expected_states = (
        prior_states[3] if restart_to_best else prior_states[0],
        prior_states[1],
        prior_states[2],
        prior_states[3],
    )
    if final_states != expected_states:
        raise RuntimeError("native three-lane rejection modified solver state")
    boundary = require_tuple(payload[5], 6, "Stage 4 boundary")
    control = cast(
        npt.NDArray[np.int64],
        _require_array(
            boundary[4],
            dtype=np.dtype(np.int64),
            shape=(7,),
            name="rejection Stage 4 control",
        ),
    )
    control_float = cast(
        npt.NDArray[np.float64],
        _require_array(
            boundary[5],
            dtype=np.dtype(np.float64),
            shape=(1,),
            name="rejection Stage 4 control float",
        ),
    )
    if (
        np.any(control < 0)
        or not math.isfinite(float(control_float[0]))
        or float(control_float[0]) < 0.0
    ):
        raise RuntimeError("native three-lane rejection Stage 4 state is invalid")
    if restart_to_best and not bool(control[2]):
        raise RuntimeError("native three-lane rejection omitted its restart boundary")
    stage04_events = list(previous_stream.stage04_events)
    if bool(control[0]):
        stage04_events.append(
            {
                "type": "stage04_reheat",
                "iteration": iteration,
                "reheat_count": int(control[1]),
                "reheat_floor": float(control_float[0]),
                "stagnation_iterations": int(control[6]),
            }
        )
    if bool(control[2]):
        stage04_events.append(
            {
                "type": "stage04_restart",
                "iteration": iteration,
                "restart_count": int(control[3]),
                "intensification": bool(control[4]),
                "stagnation_at_trigger": 0,
            }
        )

    event: dict[str, object] = {
        "operator": operator,
        "status": "not_applicable",
        "reason": "only_one_route",
        "route_indices": (),
        "affected_route_indices": (),
        "removed_customers": (),
        "candidate_customer_sequence": (),
        "candidate_route_sequences": (),
        "candidate_vehicle_delta": None,
        "candidate_feasible": False,
        "prefilter_passed": False,
        "new_routes_created": 0,
        "exact_route_evaluations": 0,
        "selection_rank": 0,
        "chain_depth": 0,
        "segment_length": 0,
        "track": "legacy",
        "constraint_category": "",
        "removal_tier": "",
        "removal_size_requested": 0,
        "removal_size_actual": 0,
        "stagnation_iterations": 0,
        "removal_trigger": "",
        "reset_observed": False,
        "ranking_score": 0.0,
        "iteration": iteration,
        "accepted": False,
        "vehicle_reduction": False,
        "distance_improvement": False,
        "candidate_objective_key": (),
    }
    if route_elimination_events is not None:
        new_events = (
            *route_elimination_events,
            {
                **event,
                "operator": "route_elimination",
                "status": "candidate_control_skipped",
                "reason": "no_selected_complete_plan_feasible",
            },
        )
    elif aggregate_merge is None:
        new_events = (event,)
    else:
        new_events = tuple(
            {
                **event,
                "status": "prefilter_rejected_aggregate",
                "reason": reason,
                "aggregate_count": count,
                "candidate_pool_hash": merge_pool_sha256,
            }
            for reason, count in aggregate_merge
        )
    all_events = (*previous_stream.neighborhood_events, *new_events)
    stage04 = require_tuple(payload[12], 4, "Stage 4")
    operator_count = len(FULL_NATIVE_OPERATOR_NAMES)
    weights = cast(
        npt.NDArray[np.float64],
        _require_array(
            stage04[0],
            dtype=np.dtype(np.float64),
            shape=(operator_count,),
            name="rejection weights",
        ),
    )
    rewards = cast(
        npt.NDArray[np.float64],
        _require_array(
            stage04[1],
            dtype=np.dtype(np.float64),
            shape=(operator_count,),
            name="rejection rewards",
        ),
    )
    calls = cast(
        npt.NDArray[np.int64],
        _require_array(
            stage04[2],
            dtype=np.dtype(np.int64),
            shape=(operator_count,),
            name="rejection calls",
        ),
    )
    totals = cast(
        npt.NDArray[np.int64],
        _require_array(
            stage04[3],
            dtype=np.dtype(np.int64),
            shape=(operator_count, 8),
            name="rejection totals",
        ),
    )
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="rejection termination",
        ),
    )
    if (
        termination.tolist()
        != [
            termination_reason,
            int(previous_stream.termination[1]),
            int(previous_stream.termination[2]),
            int(previous_stream.termination[3]),
            int(previous_stream.termination[4]),
            completed_iterations,
        ]
        or np.any(weights <= 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(rewards < 0.0)
        or np.any(calls < 0)
        or np.any(totals < 0)
    ):
        raise RuntimeError("native three-lane rejection final state is invalid")

    activity = np.zeros((operator_count, 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    for event_item in all_events:
        index = by_name[str(event_item["operator"])]
        activity[index, 1] += _semantic_event_aggregate(
            event_item, "prefilter_passed"
        )
        activity[index, 2] += cast(int, event_item["exact_route_evaluations"])
        activity[index, 3] += int(event_item["status"] == "candidate_proposed")
        activity[index, 4] += _semantic_event_aggregate(
            event_item, "candidate_feasible"
        )
        activity[index, 5] += int(bool(event_item["vehicle_reduction"]))
        activity[index, 6] += int(bool(event_item["distance_improvement"]))
    activity[:, 0] = previous_stream.operator_activity[:, 0]
    refinement_index = by_name["vehicle_reduction_refinement"]
    activity[refinement_index, 5] = 0
    activity[refinement_index, 7] = previous_stream.operator_activity[
        refinement_index, 7
    ]
    return NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(all_events),
        operator_weights=_readonly_copy(weights),
        operator_rewards=_readonly_copy(rewards),
        operator_calls=_readonly_copy(calls),
        operator_totals=_readonly_copy(totals),
        operator_activity=_readonly_copy(activity),
        termination=_readonly_copy(termination),
        transaction_sha256=search_sha256,
        stage04_events=tuple(stage04_events),
        stage04_control=_readonly_copy(control),
        initial_temperature=previous_stream.initial_temperature,
    )


def _decode_native_three_lane_constraint_no_change_iteration(
    instance: Instance,
    payload: tuple[object, ...],
    *,
    previous_payload: tuple[object, ...],
    previous_stream: NativeThreeLaneSemanticStream,
    node_names: tuple[str, ...],
    search_sha256: str,
    iteration: int,
    constraint_operator: str,
    constraint_operator_id: int,
    expected_selection: tuple[int, ...],
    removal_tier: str,
    removal_trigger: str,
    legacy_operator: str,
    legacy_operator_id: int,
    completed_iterations: int,
    termination_reason: int = 0,
) -> NativeThreeLaneSemanticStream:
    """Replay a constraint no-change failure followed by legacy rejection."""

    _verify_native_three_lane_semantic_hash(payload)
    if len(payload) != 14:
        raise RuntimeError("native constraint no-change iteration is invalid")

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native constraint no-change {name} is invalid")
        return value

    def unpack_soa(
        offsets_value: object,
        indices_value: object,
        name: str,
    ) -> tuple[CustomerSequence, ...]:
        offsets = _require_vector(offsets_value, f"no-change {name} offsets")
        indices = _require_vector(indices_value, f"no-change {name} indices")
        if (
            len(offsets) < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or np.any(offsets[:-1] >= offsets[1:])
            or np.any(indices < 0)
            or np.any(indices >= len(node_names))
        ):
            raise RuntimeError(f"native constraint no-change {name} SoA is invalid")
        return tuple(
            tuple(
                node_names[int(index)]
                for index in indices[
                    int(offsets[route]) : int(offsets[route + 1])
                ]
            )
            for route in range(len(offsets) - 1)
        )

    def unpack_state(value: object, name: str) -> tuple[CustomerSequence, ...]:
        state = require_tuple(value, 5 if "best" in name else 4, name)
        return unpack_soa(state[0], state[1], name)

    def replay(routes: tuple[CustomerSequence, ...]) -> SolutionObjective:
        expected = {customer.name for customer in instance.customers}
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(expected) or set(flattened) != expected:
            raise RuntimeError("native constraint no-change identity is invalid")
        objective = SolutionObjective.zero()
        for route in routes:
            exact = solve_exact_charging(instance, route)
            if not exact.feasible:
                raise RuntimeError("native constraint no-change replay is infeasible")
            objective += SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
        return objective

    def event(
        operator: str,
        status: str,
        reason: str,
        **changes: object,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "operator": operator,
            "status": status,
            "reason": reason,
            "route_indices": (),
            "affected_route_indices": (),
            "removed_customers": (),
            "candidate_customer_sequence": (),
            "candidate_route_sequences": (),
            "candidate_vehicle_delta": None,
            "candidate_feasible": False,
            "prefilter_passed": False,
            "new_routes_created": 0,
            "exact_route_evaluations": 0,
            "selection_rank": 0,
            "chain_depth": 0,
            "segment_length": 0,
            "track": "legacy",
            "constraint_category": "",
            "removal_tier": "",
            "removal_size_requested": 0,
            "removal_size_actual": 0,
            "stagnation_iterations": 0,
            "removal_trigger": "",
            "reset_observed": False,
            "ranking_score": 0.0,
            "iteration": iteration,
            "accepted": False,
            "vehicle_reduction": False,
            "distance_improvement": False,
            "candidate_objective_key": (),
        }
        result.update(changes)
        return result

    prior_states = (
        unpack_state(previous_payload[6], "prior legacy"),
        unpack_state(previous_payload[7], "prior quality"),
        unpack_state(previous_payload[8], "prior constraint"),
        unpack_state(previous_payload[9], "prior best"),
    )
    aggregate_legacy_events: tuple[dict[str, object], ...] = ()
    if isinstance(payload[0], tuple) and len(payload[0]) == 7:
        if legacy_operator != "route_merge" or legacy_operator_id != 3:
            raise RuntimeError(
                "native constraint no-change aggregate legacy identity is invalid"
            )
        legacy = cast(tuple[object, ...], payload[0])
        legacy_metadata = cast(
            npt.NDArray[np.int64],
            _require_array(
                legacy[0],
                dtype=np.dtype(np.int64),
                shape=(4,),
                name="no-change aggregate-merge metadata",
            ),
        )
        pool = require_tuple(legacy[1], 4, "aggregate merge pool")
        merge_offsets = _require_vector(pool[0], "aggregate merge offsets")
        merge_indices = _require_vector(pool[1], "aggregate merge indices")
        screening_reasons = _require_vector(
            legacy[2], "aggregate merge screening reasons"
        )
        plans = require_tuple(legacy[3], 4, "aggregate merge plans")
        if (
            legacy_metadata.tolist()
            != [iteration, legacy_operator_id, len(prior_states[0]), len(merge_offsets) - 1]
            or int(merge_offsets[0]) != 0
            or int(merge_offsets[-1]) != len(merge_indices)
            or len(screening_reasons) != len(merge_offsets) - 1
            or np.any(screening_reasons == 0)
            or _require_vector(plans[0], "aggregate merge plan offsets").tolist()
            != [0]
            or legacy[4] is not None
            or _require_vector(legacy[5], "aggregate merge outcome").tolist()
            != [-1] * 5
            or unpack_state(legacy[6], "aggregate merge state") != prior_states[0]
        ):
            raise RuntimeError(
                "native constraint no-change aggregate merge journal is invalid"
            )
        canonical_candidates, reason_counts, pool_sha256 = (
            _canonical_route_merge_projection(instance, prior_states[0])
        )
        if canonical_candidates:
            raise RuntimeError(
                "native constraint no-change aggregate merge omitted a candidate"
            )
        aggregate_legacy_events = tuple(
            event(
                "route_merge",
                "prefilter_rejected_aggregate",
                reason,
                aggregate_count=count,
                candidate_pool_hash=pool_sha256,
            )
            for reason, count in sorted(reason_counts.items())
        )
    elif isinstance(payload[0], tuple) and len(payload[0]) == 8:
        if legacy_operator != "route_elimination" or legacy_operator_id != 2:
            raise RuntimeError(
                "native constraint no-change route-elimination identity is invalid"
            )
        legacy = cast(tuple[object, ...], payload[0])
        legacy_header = _require_vector(
            legacy[0], "no-change route-elimination order"
        )
        attempts = cast(
            npt.NDArray[np.int64],
            _require_array(
                legacy[1],
                dtype=np.dtype(np.int64),
                shape=(len(legacy_header), 6),
                name="no-change route-elimination attempts",
            ),
        )
        if (
                len(legacy_header) == 0
                or len(legacy_header) > len(prior_states[0])
                or not np.array_equal(legacy_header, attempts[:, 0])
                or len(set(int(value) for value in legacy_header))
                != len(legacy_header)
                or np.any(legacy_header < 0)
                or np.any(legacy_header >= len(prior_states[0]))
            or np.any(attempts[:, 2] == 0)
            or _require_vector(
                legacy[2], "no-change route-elimination plan offsets"
            ).tolist()
            != [0]
            or _require_vector(
                legacy[3], "no-change route-elimination route offsets"
            ).tolist()
            != [0]
            or len(
                _require_vector(
                    legacy[4], "no-change route-elimination route indices"
                )
            )
            != 0
            or legacy[5] is not None
            or _require_vector(
                legacy[6], "no-change route-elimination outcome"
            ).tolist()
            != [-1] * 5
            or unpack_state(legacy[7], "route-elimination state")
            != prior_states[0]
        ):
            raise RuntimeError(
                "native constraint no-change route-elimination journal is invalid"
            )
        aggregate_legacy_events = (
            *(
                event(
                    "route_elimination",
                    "failed",
                    "no_existing_route_insertion"
                    if int(attempt[2]) == 1
                    else "singleton_route_screening_rejected",
                    route_indices=(int(attempt[0]),),
                    removed_customers=prior_states[0][int(attempt[0])],
                    selection_rank=int(attempt[1]),
                )
                for attempt in attempts
            ),
            event(
                "route_elimination",
                "candidate_control_skipped",
                "no_selected_complete_plan_feasible",
            ),
        )
    else:
        legacy = require_tuple(payload[0], 2, "legacy")
        legacy_metadata = cast(
            npt.NDArray[np.int64],
            _require_array(
                legacy[0],
                dtype=np.dtype(np.int64),
                shape=(4,),
                name="no-change legacy metadata",
            ),
        )
        if (
            legacy_metadata.tolist() != [iteration, legacy_operator_id, 1, 0]
            or unpack_state(legacy[1], "legacy state") != prior_states[0]
        ):
            raise RuntimeError("native constraint no-change legacy decision is invalid")

    constraint = require_tuple(payload[3], 3, "constraint")
    selection = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[0],
            dtype=np.dtype(np.int64),
            shape=(7,),
            name="no-change constraint selection",
        ),
    )
    if selection.tolist() != list(expected_selection):
        raise RuntimeError("native constraint no-change stagnation state is invalid")
    probe = require_tuple(constraint[1], 3, "constraint probe")
    removal = require_tuple(probe[0], 7, "constraint removal")
    repair = require_tuple(probe[1], 3, "constraint repair")
    outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            constraint[2],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="no-change constraint outcome",
        ),
    )
    if (
        int(outcome[0]) != constraint_operator_id
        or not 0 <= int(outcome[1]) < 2**32
        or outcome[2:].tolist() != [0, 0, 0, 0]
    ):
        raise RuntimeError("native constraint no-change outcome is invalid")
    removed_indices = _require_vector(removal[2], "no-change removed")
    removed = tuple(node_names[int(index)] for index in removed_indices)
    partial_routes = unpack_soa(removal[0], removal[1], "constraint partial")
    repair_counters = _require_vector(repair[2], "no-change repair counters")
    repair_failed = int(repair_counters[0]) != 0
    constraint_exact_infeasible = False
    repaired_routes: tuple[CustomerSequence, ...] = ()
    exact_rows = np.empty(0, dtype=np.int64)
    if repair_failed:
        if (
            _require_vector(repair[0], "failed repair offsets").tolist() != [0]
            or len(_require_vector(repair[1], "failed repair indices")) != 0
            or probe[2] is not None
        ):
            raise RuntimeError("native constraint failed repair journal is invalid")
    else:
        repaired_routes = unpack_soa(repair[0], repair[1], "constraint repaired")
        transaction = require_tuple(probe[2], 13, "constraint transaction")
        statuses = _require_vector(transaction[1], "no-change statuses")
        exact_rows = _require_vector(transaction[5], "no-change exact rows")
        constraint_exact_infeasible = statuses.tolist() == [4]
        if (
            statuses.tolist() not in ([4], [5])
            or (statuses.tolist() == [5] and len(exact_rows) != 0)
            or (statuses.tolist() == [5] and repaired_routes != prior_states[2])
        ):
            raise RuntimeError("native constraint no-change cache journal is invalid")
        if not constraint_exact_infeasible:
            objective_integers = cast(
                npt.NDArray[np.int64],
                _require_array(
                    transaction[2],
                    dtype=np.dtype(np.int64),
                    shape=(1, 2),
                    name="no-change objective integers",
                ),
            )
            objective_floats = cast(
                npt.NDArray[np.float64],
                _require_array(
                    transaction[3],
                    dtype=np.dtype(np.float64),
                    shape=(1, 2),
                    name="no-change objective floats",
                ),
            )
            reported = SolutionObjective(
                int(objective_integers[0, 0]),
                float(objective_floats[0, 0]),
                float(objective_floats[0, 1]),
                int(objective_integers[0, 1]),
            )
            if reported.key != replay(repaired_routes).key:
                raise RuntimeError("native constraint no-change objective mismatch")
    scores, removal_routes = _constraint_score_vectors(
        removal, removed_indices, "no-change"
    )
    if len(removal_routes) == 0:
        raise RuntimeError("native constraint no-change route is missing")
    affected_routes = tuple(
        sorted({int(route) for route in removal_routes[: len(removed)]})
    )
    repaired_affected = tuple(
        index
        for index, (before, after) in enumerate(
            zip(prior_states[2], repaired_routes, strict=False)
        )
        if before != after
    )

    if payload[1] is not None or payload[2] is not None or payload[4] is not None:
        raise RuntimeError("native constraint no-change lane schedule is invalid")
    final_states = (
        unpack_state(payload[6], "final legacy"),
        unpack_state(payload[7], "final quality"),
        unpack_state(payload[8], "final constraint"),
        unpack_state(payload[9], "final best"),
    )
    boundary_preview = require_tuple(payload[5], 6, "Stage 4 boundary preview")
    control_preview = _require_vector(
        boundary_preview[4], "no-change Stage 4 control preview"
    )
    if len(control_preview) != 7:
        raise RuntimeError("native constraint no-change Stage 4 control is invalid")
    expected_final_states = (
        prior_states[3] if bool(control_preview[2]) else prior_states[0],
        prior_states[1],
        prior_states[2],
        prior_states[3],
    )
    if final_states != expected_final_states:
        raise RuntimeError("native constraint no-change modified solver state")

    followup_events = (
        event(
            constraint_operator,
            "candidate_proposed",
            "constraint_ranked_removal",
            route_indices=affected_routes,
            affected_route_indices=affected_routes,
            removed_customers=removed,
            candidate_route_sequences=partial_routes,
            prefilter_passed=True,
            selection_rank=1,
            track="constraint_lane",
            constraint_category=constraint_operator,
            removal_tier=removal_tier,
            removal_size_requested=int(selection[1]),
            removal_size_actual=len(removed),
            stagnation_iterations=int(selection[4]),
            removal_trigger=removal_trigger,
            ranking_score=float(scores[0]),
        ),
        event(
            constraint_operator,
            "failed",
            (
                "constraint_removal_no_existing_route_insertion"
                if repair_failed
                else "constraint_repair_infeasible"
                if constraint_exact_infeasible
                else "constraint_removal_no_change"
            ),
            affected_route_indices=(
                repaired_affected if constraint_exact_infeasible else ()
            ),
            removed_customers=removed,
            candidate_route_sequences=(
                repaired_routes if constraint_exact_infeasible else ()
            ),
            prefilter_passed=True,
            exact_route_evaluations=(
                len(exact_rows) if constraint_exact_infeasible else 0
            ),
            track="constraint_lane",
            constraint_category=constraint_operator,
            removal_tier=removal_tier,
            removal_size_requested=int(selection[1]),
            removal_size_actual=len(removed),
            stagnation_iterations=int(selection[4]),
            removal_trigger=removal_trigger,
        ),
        *(
            aggregate_legacy_events
            if aggregate_legacy_events
            else (event(legacy_operator, "not_applicable", "only_one_route"),)
        ),
    )
    all_events = (*previous_stream.neighborhood_events, *followup_events)
    boundary = require_tuple(payload[5], 6, "Stage 4 boundary")
    control = cast(
        npt.NDArray[np.int64],
        _require_array(
            boundary[4],
            dtype=np.dtype(np.int64),
            shape=(7,),
            name="no-change Stage 4 control",
        ),
    )
    control_float = cast(
        npt.NDArray[np.float64],
        _require_array(
            boundary[5],
            dtype=np.dtype(np.float64),
            shape=(1,),
            name="no-change Stage 4 control float",
        ),
    )
    if (
        np.any(control < 0)
        or not math.isfinite(float(control_float[0]))
        or float(control_float[0]) < 0.0
    ):
        raise RuntimeError("native constraint no-change Stage 4 state is invalid")
    stage04_events = list(previous_stream.stage04_events)
    if bool(control[0]):
        stage04_events.append(
            {
                "type": "stage04_reheat",
                "iteration": iteration,
                "reheat_count": int(control[1]),
                "reheat_floor": float(control_float[0]),
                "stagnation_iterations": int(control[6]),
            }
        )
    if bool(control[2]):
        stage04_events.append(
            {
                "type": "stage04_restart",
                "iteration": iteration,
                "restart_count": int(control[3]),
                "intensification": bool(control[4]),
                "stagnation_at_trigger": 0,
            }
        )
    stage04 = require_tuple(payload[12], 4, "Stage 4")
    operator_count = len(FULL_NATIVE_OPERATOR_NAMES)
    weights = cast(
        npt.NDArray[np.float64],
        _require_array(
            stage04[0],
            dtype=np.dtype(np.float64),
            shape=(operator_count,),
            name="no-change weights",
        ),
    )
    rewards = cast(
        npt.NDArray[np.float64],
        _require_array(
            stage04[1],
            dtype=np.dtype(np.float64),
            shape=(operator_count,),
            name="no-change rewards",
        ),
    )
    calls = cast(
        npt.NDArray[np.int64],
        _require_array(
            stage04[2],
            dtype=np.dtype(np.int64),
            shape=(operator_count,),
            name="no-change calls",
        ),
    )
    totals = cast(
        npt.NDArray[np.int64],
        _require_array(
            stage04[3],
            dtype=np.dtype(np.int64),
            shape=(operator_count, 8),
            name="no-change totals",
        ),
    )
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="no-change termination",
        ),
    )
    if (
        termination.tolist()
        != [
            termination_reason,
            int(previous_stream.termination[1]),
            int(previous_stream.termination[2]) + len(exact_rows),
            int(previous_stream.termination[3]) + len(exact_rows),
            int(previous_stream.termination[4]),
            completed_iterations,
        ]
        or np.any(weights <= 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(rewards < 0.0)
        or np.any(calls < 0)
        or np.any(totals < 0)
    ):
        raise RuntimeError("native constraint no-change final state is invalid")
    activity = np.zeros((operator_count, 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    for event_item in all_events:
        index = by_name[str(event_item["operator"])]
        activity[index, 1] += _semantic_event_aggregate(
            event_item, "prefilter_passed"
        )
        activity[index, 2] += cast(int, event_item["exact_route_evaluations"])
        activity[index, 3] += int(event_item["status"] == "candidate_proposed")
        activity[index, 4] += _semantic_event_aggregate(
            event_item, "candidate_feasible"
        )
        activity[index, 5] += int(bool(event_item["vehicle_reduction"]))
        activity[index, 6] += int(bool(event_item["distance_improvement"]))
    activity[:, 0] = previous_stream.operator_activity[:, 0]
    refinement_index = by_name["vehicle_reduction_refinement"]
    activity[refinement_index, 5] = 0
    activity[refinement_index, 7] = previous_stream.operator_activity[
        refinement_index, 7
    ]
    return NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(all_events),
        operator_weights=_readonly_copy(weights),
        operator_rewards=_readonly_copy(rewards),
        operator_calls=_readonly_copy(calls),
        operator_totals=_readonly_copy(totals),
        operator_activity=_readonly_copy(activity),
        termination=_readonly_copy(termination),
        transaction_sha256=search_sha256,
        stage04_events=tuple(stage04_events),
        stage04_control=_readonly_copy(control),
        initial_temperature=previous_stream.initial_temperature,
    )


def _decode_native_three_lane_standard_energy_rejection_iteration(
    instance: Instance,
    payload: tuple[object, ...],
    *,
    previous_payload: tuple[object, ...],
    previous_stream: NativeThreeLaneSemanticStream,
    node_names: tuple[str, ...],
    search_sha256: str,
    iteration: int,
    destroy_name: str,
    repair_name: str,
    expected_metadata: tuple[int, ...],
    expected_statuses: tuple[int, ...],
    expected_exact_rows: tuple[int, ...],
    expected_exact_call_delta: int,
    candidate_prepared: bool,
    feasible_repair: bool,
    completed_iterations: int,
) -> NativeThreeLaneSemanticStream:
    """Replay a standard destroy/repair rejection with an audited journal."""

    _verify_native_three_lane_semantic_hash(payload)
    if len(payload) != 14:
        raise RuntimeError("native standard-energy iteration is invalid")

    def require_tuple(value: object, size: int, name: str) -> tuple[object, ...]:
        if not isinstance(value, tuple) or len(value) != size:
            raise RuntimeError(f"native standard-energy {name} is invalid")
        return value

    def unpack_soa(
        offsets_value: object,
        indices_value: object,
        name: str,
    ) -> tuple[CustomerSequence, ...]:
        offsets = _require_vector(offsets_value, f"standard-energy {name} offsets")
        indices = _require_vector(indices_value, f"standard-energy {name} indices")
        if (
            len(offsets) < 2
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(indices)
            or np.any(offsets[:-1] >= offsets[1:])
            or np.any(indices < 0)
            or np.any(indices >= len(node_names))
        ):
            raise RuntimeError(f"native standard-energy {name} SoA is invalid")
        return tuple(
            tuple(
                node_names[int(index)]
                for index in indices[
                    int(offsets[route]) : int(offsets[route + 1])
                ]
            )
            for route in range(len(offsets) - 1)
        )

    def unpack_state(value: object, name: str) -> tuple[CustomerSequence, ...]:
        state = require_tuple(value, 5 if "best" in name else 4, name)
        return unpack_soa(state[0], state[1], name)

    def replay(routes: tuple[CustomerSequence, ...]) -> SolutionObjective:
        expected = {customer.name for customer in instance.customers}
        flattened = tuple(customer for route in routes for customer in route)
        if len(flattened) != len(expected) or set(flattened) != expected:
            raise RuntimeError("native standard-energy customer identity is invalid")
        objective = SolutionObjective.zero()
        for route in routes:
            exact = solve_exact_charging(instance, route)
            if not exact.feasible:
                raise RuntimeError("native standard-energy replay is infeasible")
            objective += SolutionObjective.from_route(
                instance,
                exact.route,
                total_distance=exact.distance,
                total_charging_time=exact.charging_time,
            )
        return objective

    prior_states = (
        unpack_state(previous_payload[6], "prior legacy"),
        unpack_state(previous_payload[7], "prior quality"),
        unpack_state(previous_payload[8], "prior constraint"),
        unpack_state(previous_payload[9], "prior best"),
    )
    legacy = require_tuple(payload[0], 8, "legacy")
    metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[0],
            dtype=np.dtype(np.int64),
            shape=(8,),
            name="standard-energy metadata",
        ),
    )
    if metadata.tolist() != list(expected_metadata):
        raise RuntimeError("native standard-energy decision is invalid")
    removed_indices = _require_vector(legacy[1], "standard-energy removed")
    removed = tuple(node_names[int(index)] for index in removed_indices)
    repaired = require_tuple(legacy[4], 3, "energy repair")
    repaired_routes = unpack_soa(repaired[0], repaired[1], "energy repaired")
    repaired_objective = replay(repaired_routes) if candidate_prepared else None
    if legacy[5] is not None:
        raise RuntimeError("native standard-energy duplicated its final transaction")
    transaction = require_tuple(legacy[6], 13, "energy insertion transaction")
    statuses = _require_vector(transaction[1], "standard-energy statuses")
    exact_rows = _require_vector(transaction[5], "standard-energy exact rows")
    if (
        len(statuses) == 0
        or np.any(statuses < 0)
        or np.any(statuses > 5)
        or np.any(exact_rows < 0)
        or len(set(int(row) for row in exact_rows)) != len(exact_rows)
    ):
        raise RuntimeError("native standard-energy insertion journal is invalid")
    if tuple(int(status) for status in statuses) != expected_statuses:
        raise RuntimeError("native standard-energy insertion statuses diverged")
    if tuple(int(row) for row in exact_rows) != expected_exact_rows:
        raise RuntimeError("native standard-energy exact order diverged")
    objective_integers = cast(
        npt.NDArray[np.int64],
        _require_array(
            transaction[2],
            dtype=np.dtype(np.int64),
            shape=(len(statuses), 2),
            name="standard-energy objective integers",
        ),
    )
    objective_floats = cast(
        npt.NDArray[np.float64],
        _require_array(
            transaction[3],
            dtype=np.dtype(np.float64),
            shape=(len(statuses), 2),
            name="standard-energy objective floats",
        ),
    )
    candidate_objective_key: tuple[object, ...]
    selected_plan = int(metadata[6])
    if candidate_prepared != (selected_plan >= 0):
        raise RuntimeError("native standard-energy candidate state is inconsistent")
    if candidate_prepared:
        if selected_plan >= len(statuses):
            raise RuntimeError("native standard-energy selected plan is invalid")
        reported = SolutionObjective(
            int(objective_integers[selected_plan, 0]),
            float(objective_floats[selected_plan, 0]),
            float(objective_floats[selected_plan, 1]),
            int(objective_integers[selected_plan, 1]),
        )
        if repaired_objective is None or reported.key != repaired_objective.key:
            raise RuntimeError("native standard-energy objective mismatch")
        acceptance = require_tuple(payload[4], 3, "legacy acceptance")
        if acceptance != (0, 0, 0):
            raise RuntimeError("native standard-energy rejection is invalid")
        candidate_objective_key = reported.key
    else:
        if payload[4] is not None or len(_require_vector(transaction[11], "attempted")):
            raise RuntimeError("native standard-energy unexpected candidate state")
        candidate_objective_key = ()
    if payload[1] is not None or payload[2] is not None or payload[3] is not None:
        raise RuntimeError("native standard-energy lane schedule is invalid")
    final_states = (
        unpack_state(payload[6], "final legacy"),
        unpack_state(payload[7], "final quality"),
        unpack_state(payload[8], "final constraint"),
        unpack_state(payload[9], "final best"),
    )
    if final_states != prior_states:
        raise RuntimeError("native standard-energy rejection modified solver state")
    boundary = require_tuple(payload[5], 6, "Stage 4 boundary")
    control = cast(
        npt.NDArray[np.int64],
        _require_array(
            boundary[4],
            dtype=np.dtype(np.int64),
            shape=(7,),
            name="standard-energy Stage 4 control",
        ),
    )
    control_float = cast(
        npt.NDArray[np.float64],
        _require_array(
            boundary[5],
            dtype=np.dtype(np.float64),
            shape=(1,),
            name="standard-energy Stage 4 control float",
        ),
    )
    if (
        np.any(control < 0)
        or not math.isfinite(float(control_float[0]))
        or float(control_float[0]) < 0.0
    ):
        raise RuntimeError("native standard-energy reheat state is invalid")

    event: dict[str, object] = {
        "operator": "standard",
        "status": "proposal",
        "reason": f"{destroy_name}+{repair_name}",
        "route_indices": (),
        "affected_route_indices": (),
        "removed_customers": removed,
        "candidate_customer_sequence": (),
        "candidate_route_sequences": (),
        "candidate_vehicle_delta": None,
        "candidate_feasible": False,
        "prefilter_passed": False,
        "new_routes_created": 0,
        "exact_route_evaluations": 0,
        "selection_rank": 0,
        "chain_depth": 0,
        "segment_length": 0,
        "track": "legacy",
        "constraint_category": "",
        "removal_tier": "",
        "removal_size_requested": 0,
        "removal_size_actual": 0,
        "stagnation_iterations": 0,
        "removal_trigger": "",
        "reset_observed": False,
        "ranking_score": 0.0,
        "iteration": iteration,
        "accepted": False,
        "vehicle_reduction": False,
        "distance_improvement": False,
        "candidate_objective_key": candidate_objective_key,
        "_operator_feasible_repair": feasible_repair,
    }
    all_events = (*previous_stream.neighborhood_events, event)
    stage04 = require_tuple(payload[12], 4, "Stage 4")
    operator_count = len(FULL_NATIVE_OPERATOR_NAMES)
    weights = cast(
        npt.NDArray[np.float64],
        _require_array(
            stage04[0],
            dtype=np.dtype(np.float64),
            shape=(operator_count,),
            name="standard-energy weights",
        ),
    )
    rewards = cast(
        npt.NDArray[np.float64],
        _require_array(
            stage04[1],
            dtype=np.dtype(np.float64),
            shape=(operator_count,),
            name="standard-energy rewards",
        ),
    )
    calls = cast(
        npt.NDArray[np.int64],
        _require_array(
            stage04[2],
            dtype=np.dtype(np.int64),
            shape=(operator_count,),
            name="standard-energy calls",
        ),
    )
    totals = cast(
        npt.NDArray[np.int64],
        _require_array(
            stage04[3],
            dtype=np.dtype(np.int64),
            shape=(operator_count, 8),
            name="standard-energy totals",
        ),
    )
    termination = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[11],
            dtype=np.dtype(np.int64),
            shape=(6,),
            name="standard-energy termination",
        ),
    )
    if (
        int(termination[0]) != 0
        or int(termination[1]) != int(previous_stream.termination[1])
        or int(termination[2]) < int(previous_stream.termination[2])
        or int(termination[2]) - int(previous_stream.termination[2])
        != expected_exact_call_delta
        or int(termination[3]) - int(previous_stream.termination[3])
        != int(termination[2]) - int(previous_stream.termination[2])
        or int(termination[4]) != int(previous_stream.termination[4])
        or int(termination[5]) != completed_iterations
        or np.any(weights <= 0.0)
        or np.any(~np.isfinite(weights))
        or np.any(rewards < 0.0)
        or np.any(calls < 0)
        or np.any(totals < 0)
    ):
        raise RuntimeError("native standard-energy final state is invalid")
    activity = np.zeros((operator_count, 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    for event_item in all_events:
        index = by_name[str(event_item["operator"])]
        activity[index, 1] += _semantic_event_aggregate(
            event_item, "prefilter_passed"
        )
        activity[index, 2] += cast(int, event_item["exact_route_evaluations"])
        activity[index, 3] += int(event_item["status"] == "candidate_proposed")
        activity[index, 4] += _semantic_event_aggregate(
            event_item, "candidate_feasible"
        )
        activity[index, 5] += int(bool(event_item["vehicle_reduction"]))
        activity[index, 6] += int(bool(event_item["distance_improvement"]))
    activity[:, 0] = previous_stream.operator_activity[:, 0]
    standard_index = by_name["standard"]
    activity[standard_index, 0] += int(feasible_repair)
    refinement_index = by_name["vehicle_reduction_refinement"]
    activity[refinement_index, 5] = 0
    activity[refinement_index, 7] = previous_stream.operator_activity[
        refinement_index, 7
    ]
    return NativeThreeLaneSemanticStream(
        neighborhood_events=tuple(all_events),
        operator_weights=_readonly_copy(weights),
        operator_rewards=_readonly_copy(rewards),
        operator_calls=_readonly_copy(calls),
        operator_totals=_readonly_copy(totals),
        operator_activity=_readonly_copy(activity),
        termination=_readonly_copy(termination),
        transaction_sha256=search_sha256,
        stage04_events=(
            *previous_stream.stage04_events,
            *(
                (
                    {
                        "type": "stage04_reheat",
                        "iteration": iteration,
                        "reheat_count": int(control[1]),
                        "reheat_floor": float(control_float[0]),
                        "stagnation_iterations": int(control[6]),
                    },
                )
                if bool(control[0])
                else ()
            ),
            *(
                (
                    {
                        "type": "stage04_restart",
                        "iteration": iteration,
                        "restart_count": int(control[3]),
                        "intensification": bool(control[4]),
                        "stagnation_at_trigger": 0,
                    },
                )
                if bool(control[2])
                else ()
            ),
        ),
        stage04_control=_readonly_copy(control),
        initial_temperature=previous_stream.initial_temperature,
    )


def _pack_integer_config(
    config: object,
    fields: tuple[str, ...],
) -> npt.NDArray[np.int64]:
    values: list[int] = []
    for name in fields:
        value = getattr(config, name)
        if isinstance(value, bool):
            values.append(int(value))
        elif isinstance(value, int):
            values.append(value)
        else:
            raise TypeError(f"full native integer config field {name} is not integral")
    return np.ascontiguousarray(values, dtype=np.int64)


def _pack_float_config(
    config: object,
    fields: tuple[str, ...],
) -> npt.NDArray[np.float64]:
    values: list[float] = []
    for name in fields:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"full native float config field {name} is not numeric")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"full native float config field {name} is not finite")
        values.append(converted)
    return np.ascontiguousarray(values, dtype=np.float64)


def _decode_full_native_exact_journal(
    instance: Instance,
    payload: object,
    *,
    native_runtime: NativeKernelRuntime,
    batch_size: int,
) -> tuple[
    str,
    str,
    tuple[Mapping[str, object], ...],
    str,
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    """Independently replay every completed native exact batch."""

    if not isinstance(payload, tuple) or len(payload) != 2:
        raise RuntimeError("full native exact journal has an invalid envelope")
    batches_value, producer_sha256 = payload
    if not isinstance(batches_value, tuple):
        raise RuntimeError("full native exact journal batches are invalid")
    evidence = bytearray(b"stage05.2-native-exact-journal-v2")
    lane_names = (
        "initialization",
        "legacy",
        "quality_shadow",
        "constraint_lane",
    )
    operator_names = (
        "initial_solution",
        "initialization",
        "standard",
        "greedy",
        "regret2",
        "energy",
        "route_elimination",
        "route_merge",
        "vehicle_count_aware_repair",
        "vehicle_reduction_refinement",
        "relocate",
        "swap",
        "two_opt_star",
        "route_segment_destroy",
        "ejection_chain",
        "station_pressure",
        "time_window_conflict",
        "worst_energy_detour",
        "shaw_related",
    )
    lane_by_id = {_stable_int63(name): name for name in lane_names}
    operator_by_id = {_stable_int63(name): name for name in operator_names}
    if len(lane_by_id) != len(lane_names) or len(operator_by_id) != len(operator_names):
        raise RuntimeError("full native exact journal context IDs collide")

    candidate_work: list[dict[str, object]] = []
    route_results: list[dict[str, object]] = []
    events: list[Mapping[str, object]] = []
    total_routes = 0
    for batch_ordinal, batch_value in enumerate(batches_value):
        if not isinstance(batch_value, tuple) or len(batch_value) != 9:
            raise RuntimeError("full native exact journal batch is invalid")
        _append_native_nested_evidence(evidence, batch_value)
        context = cast(
            npt.NDArray[np.int64],
            _require_array(
                batch_value[0],
                dtype=np.dtype(np.int64),
                shape=(3,),
                name="full native exact journal context",
            ),
        )
        route_offsets = _require_vector(
            batch_value[1], "full native exact journal route offsets"
        )
        route_indices = _require_vector(
            batch_value[2], "full native exact journal route indices"
        )
        route_count = len(route_offsets) - 1
        if (
            route_count <= 0
            or int(route_offsets[0]) != 0
            or int(route_offsets[-1]) != len(route_indices)
            or np.any(route_offsets[:-1] > route_offsets[1:])
            or np.any(route_indices < 0)
            or np.any(route_indices >= len(native_runtime.context.node_names))
        ):
            raise RuntimeError("full native exact journal route SoA is invalid")
        try:
            lane = lane_by_id[int(context[0])]
            operator = operator_by_id[int(context[1])]
        except KeyError as error:
            raise RuntimeError(
                "full native exact journal contains an unknown context ID"
            ) from error
        iteration = None if int(context[2]) < 0 else int(context[2])
        sequences = tuple(
            tuple(
                native_runtime.context.node_names[int(index)]
                for index in route_indices[
                    int(route_offsets[route]) : int(route_offsets[route + 1])
                ]
            )
            for route in range(route_count)
        )
        batch_counters = np.ascontiguousarray(
            [
                route_count,
                route_count,
                route_count,
                0,
                1,
                0,
                0,
                0,
                1,
                batch_size,
            ],
            dtype=np.int64,
        )
        decoded = decode_exact_charging_batch_numeric(
            instance,
            sequences,
            native_runtime=native_runtime,
            batch_size=batch_size,
            payload=(*batch_value[3:9], batch_counters),
            native_kernel_seconds=0.0,
        )
        candidate_work.append(
            {
                "lane": lane,
                "iteration": iteration,
                "operator": operator,
                "sequences": [list(sequence) for sequence in sequences],
            }
        )
        for sequence, result in zip(sequences, decoded.results, strict=True):
            result_payload: dict[str, object] = {
                "feasible": result.feasible,
                "route": list(result.route),
                "distance": result.distance,
                "total_energy": result.total_energy,
                "charged_energy": result.charged_energy,
                "charging_time": result.charging_time,
                "labels_generated": result.labels_generated,
                "labels_expanded": result.labels_expanded,
                "labels_pruned": result.labels_pruned,
                "failure_reason": result.failure_reason,
            }
            route_results.append(
                {"sequence": list(sequence), "result": result_payload}
            )
            events.append(
                {
                    "event_type": "candidate_route_result",
                    "status": "complete",
                    "lane": lane,
                    "iteration": iteration,
                    "operator": operator,
                    "customer_sequences": [list(sequence), list(result.route)],
                    **result_payload,
                }
            )
        events.append(
            {
                "event_type": "parallel_batch",
                "status": "native_complete",
                "worker_count": 0,
                "worker_protocol": "full_solve_soa_v2",
                "submission_order": list(range(route_count)),
                "completion_order": list(range(route_count)),
                "merge_order": list(range(route_count)),
                "customer_sequences": [list(sequence) for sequence in sequences],
                "lane": lane,
                "iteration": iteration,
                "operator": operator,
                "batch_ordinal": batch_ordinal,
            }
        )
        total_routes += route_count
    if (
        not isinstance(producer_sha256, str)
        or not _is_sha256(producer_sha256)
        or hashlib.sha256(evidence).hexdigest() != producer_sha256
    ):
        raise RuntimeError("full native exact journal SHA-256 mismatch")
    if total_routes == 0:
        raise RuntimeError("full native exact journal contains no completed work")
    return (
        stable_candidate_payload_hash(candidate_work),
        stable_candidate_payload_hash(route_results),
        tuple(events),
        producer_sha256,
        tuple(candidate_work),
        tuple(route_results),
    )


def _decode_full_native_control_journal(
    payload: object,
    *,
    node_names: tuple[str, ...],
) -> tuple[
    tuple[Mapping[str, object], ...],
    str,
    Mapping[str, int],
    Mapping[str, int],
    Mapping[str, object],
]:
    """Validate committed candidate-plan decisions and cache lifecycle totals."""

    if not isinstance(payload, tuple) or len(payload) != 3:
        raise RuntimeError("full native control journal has an invalid envelope")
    batches_value, screening_payload, producer_sha256 = payload
    if not isinstance(screening_payload, tuple) or len(screening_payload) != 11:
        raise RuntimeError("full native screening payload is invalid")
    (
        screening_statistics_value,
        reason_codes_value,
        reason_counts_value,
        screening_timing_value,
        screening_contexts_value,
        screening_route_offsets_value,
        screening_route_indices_value,
        screening_codes_value,
        screening_metrics_value,
        screening_flags_value,
        screening_occupancies_value,
    ) = screening_payload
    if not isinstance(batches_value, tuple):
        raise RuntimeError("full native control journal batches are invalid")
    evidence = bytearray(b"stage05.2-native-control-journal-v2")
    lane_names = ("initialization", "legacy", "quality_shadow", "constraint_lane")
    operator_names = (
        "initial_solution",
        "initialization",
        "standard",
        "greedy",
        "regret2",
        "energy",
        "route_elimination",
        "route_merge",
        "vehicle_count_aware_repair",
        "vehicle_reduction_refinement",
        "relocate",
        "swap",
        "two_opt_star",
        "route_segment_destroy",
        "ejection_chain",
        "station_pressure",
        "time_window_conflict",
        "worst_energy_detour",
        "shaw_related",
    )
    lane_by_id = {_stable_int63(name): name for name in lane_names}
    operator_by_id = {_stable_int63(name): name for name in operator_names}
    decision_status = {
        0: "ineligible",
        1: "already_attempted",
        2: "selected",
        3: "not_selected",
    }
    events: list[Mapping[str, object]] = []
    decision_count = 0
    selected_count = 0
    skipped_count = 0
    final_cache_statistics: npt.NDArray[np.int64] | None = None
    for batch_ordinal, batch_value in enumerate(batches_value):
        if not isinstance(batch_value, tuple) or len(batch_value) != 14:
            raise RuntimeError("full native control journal batch is invalid")
        _append_native_nested_evidence(evidence, batch_value)
        context = cast(
            npt.NDArray[np.int64],
            _require_array(
                batch_value[0],
                dtype=np.dtype(np.int64),
                shape=(3,),
                name="full native control context",
            ),
        )
        try:
            lane = lane_by_id[int(context[0])]
            operator = operator_by_id[int(context[1])]
        except KeyError as error:
            raise RuntimeError(
                "full native control journal contains an unknown context ID"
            ) from error
        iteration = int(context[2])
        warm_start = lane == "initialization" and operator == "initial_solution"
        plan_offsets = _require_vector(batch_value[1], "control plan offsets")
        route_offsets = _require_vector(batch_value[2], "control route offsets")
        route_indices = _require_vector(batch_value[3], "control route indices")
        plan_count = len(plan_offsets) - 1
        route_count = len(route_offsets) - 1
        if (
            plan_count <= 0
            or route_count <= 0
            or int(plan_offsets[0]) != 0
            or int(plan_offsets[-1]) != route_count
            or np.any(plan_offsets[:-1] > plan_offsets[1:])
            or int(route_offsets[0]) != 0
            or int(route_offsets[-1]) != len(route_indices)
            or np.any(route_offsets[:-1] > route_offsets[1:])
            or np.any(route_indices < 0)
            or np.any(route_indices >= len(node_names))
        ):
            raise RuntimeError("full native control plan SoA is invalid")
        ranked = _require_vector(batch_value[4], "control ranked plans")
        selected = _require_vector(batch_value[5], "control selected plans")
        decisions = _require_vector(batch_value[6], "control decision codes")
        statuses = _require_vector(batch_value[7], "control plan statuses")
        ranking_integer = cast(
            npt.NDArray[np.int64],
            _require_array(
                batch_value[8],
                dtype=np.dtype(np.int64),
                shape=(plan_count, 2),
                name="control ranking integer metrics",
            ),
        )
        ranking_float = cast(
            npt.NDArray[np.float64],
            _require_array(
                batch_value[9],
                dtype=np.dtype(np.float64),
                shape=(plan_count,),
                name="control ranking float metrics",
            ),
        )
        resolutions = _require_vector(batch_value[10], "control route resolutions")
        cache_statistics = cast(
            npt.NDArray[np.int64],
            _require_array(
                batch_value[11],
                dtype=np.dtype(np.int64),
                shape=(11,),
                name="control cache statistics",
            ),
        )
        budget_state = cast(
            npt.NDArray[np.int64],
            _require_array(
                batch_value[12],
                dtype=np.dtype(np.int64),
                shape=(9,),
                name="control budget state",
            ),
        )
        protocol_flags = cast(
            npt.NDArray[np.int64],
            _require_array(
                batch_value[13],
                dtype=np.dtype(np.int64),
                shape=(4,),
                name="control protocol flags",
            ),
        )
        if np.any((protocol_flags != 0) & (protocol_flags != 1)):
            raise RuntimeError("full native control protocol flags are invalid")
        implementation_internal = bool(protocol_flags[0])
        if iteration < 0 and not (
            iteration == -1 and (warm_start or implementation_internal)
        ):
            raise RuntimeError("full native control iteration is invalid")
        if (
            len(ranked) != plan_count
            or set(int(value) for value in ranked) != set(range(plan_count))
            or len(decisions) != plan_count
            or len(statuses) != plan_count
            or len(resolutions) != route_count
            or any(int(value) not in decision_status for value in decisions)
            or any(int(value) < 0 or int(value) >= plan_count for value in selected)
            or (
                not warm_start
                and not implementation_internal
                and int(budget_state[2]) != iteration
            )
        ):
            raise RuntimeError("full native control journal arrays do not reconcile")
        selected_set = {int(value) for value in selected}
        if selected_set != {
            plan for plan, code in enumerate(decisions) if int(code) == 2
        }:
            raise RuntimeError("full native control selection and decisions disagree")
        routes = tuple(
            tuple(
                node_names[int(index)]
                for index in route_indices[
                    int(route_offsets[route]) : int(route_offsets[route + 1])
                ]
            )
            for route in range(route_count)
        )
        if warm_start:
            for result in ("miss", "hit"):
                for sequence in routes:
                    events.append(
                        {
                            "event_type": "cache_event",
                            "operation": "lookup",
                            "result": result,
                            "lane": lane,
                            "iteration": None,
                            "operator": operator,
                            "customer_sequence": list(sequence),
                            "batch_ordinal": batch_ordinal,
                        }
                    )
            events.append(
                {
                    "event_type": "candidate_initial_solution",
                    "status": "verified",
                    "lane": lane,
                    "iteration": None,
                    "operator": operator,
                    "customer_sequences": [list(sequence) for sequence in routes],
                    "batch_ordinal": batch_ordinal,
                }
            )
            events.append(
                {
                    "event_type": "candidate_plan_decision",
                    "status": "selected",
                    "lane": lane,
                    "iteration": None,
                    "operator": operator,
                    "candidate_id": 0,
                    "rank": 1,
                    "vehicle_count": int(ranking_integer[0, 0]),
                    "changed_route_count": int(ranking_integer[0, 1]),
                    "optimistic_total_distance": float(ranking_float[0]),
                    "customer_sequences": [list(sequence) for sequence in routes],
                    "proposal_ordinal": 0,
                    "transaction_status_code": int(statuses[0]),
                    "batch_ordinal": batch_ordinal,
                }
            )
        elif not implementation_internal:
            for rank, raw_plan in enumerate(ranked, start=1):
                plan = int(raw_plan)
                code = int(decisions[plan])
                sequences = routes[
                    int(plan_offsets[plan]) : int(plan_offsets[plan + 1])
                ]
                events.append(
                    {
                        "event_type": "candidate_plan_decision",
                        "status": decision_status[code],
                        "lane": lane,
                        "iteration": iteration,
                        "operator": operator,
                        "candidate_id": plan,
                        "rank": rank,
                        "vehicle_count": int(ranking_integer[plan, 0]),
                        "changed_route_count": int(ranking_integer[plan, 1]),
                        "optimistic_total_distance": float(ranking_float[plan]),
                        "customer_sequences": [
                            list(sequence) for sequence in sequences
                        ],
                        "proposal_ordinal": plan,
                        "transaction_status_code": int(statuses[plan]),
                        "batch_ordinal": batch_ordinal,
                    }
                )
        if not warm_start:
            for raw_plan in selected:
                plan = int(raw_plan)
                for route in range(
                    int(plan_offsets[plan]), int(plan_offsets[plan + 1])
                ):
                    resolution = int(resolutions[route])
                    if resolution == 0:
                        continue
                    events.append(
                        {
                            "event_type": "cache_event",
                            "operation": "lookup",
                            "result": "hit" if resolution == 1 else "miss",
                            "lane": lane,
                            "iteration": iteration,
                            "operator": operator,
                            "customer_sequence": list(routes[route]),
                            "batch_ordinal": batch_ordinal,
                        }
                    )
        events.append(
            {
                "event_type": "candidate_cache_transaction",
                "status": "committed",
                "lane": lane,
                "iteration": iteration,
                "operator": operator,
                "lookups": int(np.count_nonzero(resolutions)),
                "hits": int(np.count_nonzero(resolutions == 1)),
                "exact_stores": int(np.count_nonzero(resolutions == 2)),
                "cache_statistics": cache_statistics.tolist(),
                "budget_state": budget_state.tolist(),
                "batch_ordinal": batch_ordinal,
            }
        )
        if not implementation_internal:
            decision_count += plan_count
            selected_count += len(selected_set)
            skipped_count += plan_count - len(selected_set)
        final_cache_statistics = cache_statistics
    screening_statistics_array = cast(
        npt.NDArray[np.int64],
        _require_array(
            screening_statistics_value,
            dtype=np.dtype(np.int64),
            shape=(8,),
            name="full native screening statistics",
        ),
    )
    reason_codes = _require_vector(reason_codes_value, "screening reason codes")
    reason_counts = _require_vector(reason_counts_value, "screening reason counts")
    screening_timing = cast(
        npt.NDArray[np.float64],
        _require_array(
            screening_timing_value,
            dtype=np.dtype(np.float64),
            shape=(1,),
            name="full native screening timing",
        ),
    )
    screening_occupancies = _require_vector(
        screening_occupancies_value,
        "full native screening occupancies",
    )
    if np.any(screening_occupancies < 0):
        raise RuntimeError("full native screening occupancy is negative")
    screening_contexts = cast(
        npt.NDArray[np.int64],
        _require_array(
            screening_contexts_value,
            dtype=np.dtype(np.int64),
            shape=(len(cast(npt.NDArray[np.int64], screening_contexts_value)), 3),
            name="full native screening contexts",
        ),
    )
    screening_row_count = screening_contexts.shape[0]
    screening_route_offsets = _require_vector(
        screening_route_offsets_value, "full native screening route offsets"
    )
    screening_route_indices = _require_vector(
        screening_route_indices_value, "full native screening route indices"
    )
    screening_codes = cast(
        npt.NDArray[np.int64],
        _require_array(
            screening_codes_value,
            dtype=np.dtype(np.int64),
            shape=(screening_row_count, 16),
            name="full native screening codes",
        ),
    )
    screening_metrics = cast(
        npt.NDArray[np.float64],
        _require_array(
            screening_metrics_value,
            dtype=np.dtype(np.float64),
            shape=(screening_row_count, 15),
            name="full native screening metrics",
        ),
    )
    screening_flags = cast(
        npt.NDArray[np.int64],
        _require_array(
            screening_flags_value,
            dtype=np.dtype(np.int64),
            shape=(screening_row_count, 4),
            name="full native screening flags",
        ),
    )
    if (
        len(reason_codes) != len(reason_counts)
        or np.any(reason_counts <= 0)
        or len(set(int(code) for code in reason_codes)) != len(reason_codes)
        or not math.isfinite(float(screening_timing[0]))
        or float(screening_timing[0]) < 0.0
        or int(screening_statistics_array[0])
        != int(screening_statistics_array[1] + screening_statistics_array[2])
        or int(screening_statistics_array[5])
        != int(screening_statistics_array[6] + screening_statistics_array[7])
        or len(screening_route_offsets) != screening_row_count + 1
        or int(screening_route_offsets[0]) != 0
        or int(screening_route_offsets[-1]) != len(screening_route_indices)
        or np.any(screening_route_offsets[:-1] > screening_route_offsets[1:])
        or np.any(screening_route_indices < 0)
        or np.any(screening_route_indices >= len(node_names))
        or np.any(
            (screening_flags[:, :3] != 0) & (screening_flags[:, :3] != 1)
        )
        or np.any(screening_flags[:, 3] < 0)
        or screening_row_count != int(screening_statistics_array[0])
        or int(np.count_nonzero(screening_flags[:, 0] & screening_flags[:, 2]))
        != int(screening_statistics_array[5])
    ):
        raise RuntimeError("full native screening statistics do not reconcile")
    for row in range(screening_row_count):
        try:
            lane = lane_by_id[int(screening_contexts[row, 0])]
            operator = operator_by_id[int(screening_contexts[row, 1])]
        except KeyError as error:
            raise RuntimeError(
                "full native screening journal contains an unknown context ID"
            ) from error
        check_count = int(screening_codes[row, 7])
        if check_count < 0 or check_count > 8:
            raise RuntimeError("full native screening check count is invalid")
        checks: list[dict[str, object]] = []
        for check_ordinal in range(check_count):
            packed_check = int(screening_codes[row, 8 + check_ordinal])
            check_code, status_code = divmod(packed_check, 10)
            if (
                check_code not in _SCREEN_CHECK_BY_CODE
                or status_code not in _SCREEN_CHECK_STATUS_BY_CODE
            ):
                raise RuntimeError("full native screening check code is invalid")
            checks.append(
                {
                    "check": _SCREEN_CHECK_BY_CODE[check_code],
                    "status": _SCREEN_CHECK_STATUS_BY_CODE[status_code],
                    "value": float(screening_metrics[row, 7 + check_ordinal]),
                    "reason": (
                        SCREEN_REASON_BY_CODE[int(screening_codes[row, 1])]
                        if status_code == 0
                        else ""
                    ),
                }
            )
        reason_code = int(screening_codes[row, 1])
        failed_check_code = int(screening_codes[row, 2])
        if reason_code not in SCREEN_REASON_BY_CODE or (
            failed_check_code != 0
            and failed_check_code not in _SCREEN_CHECK_BY_CODE
        ):
            raise RuntimeError("full native screening outcome code is invalid")
        sequence = tuple(
            node_names[int(index)]
            for index in screening_route_indices[
                int(screening_route_offsets[row]) : int(
                    screening_route_offsets[row + 1]
                )
            ]
        )
        accepted = bool(screening_codes[row, 0])
        events.append(
            {
                "event_type": "screening_decision",
                "status": "pass" if accepted else "rejected",
                "lane": lane,
                "iteration": int(screening_contexts[row, 2]),
                "operator": operator,
                "customer_sequence": list(sequence),
                "first_failed_check": (
                    _SCREEN_CHECK_BY_CODE.get(failed_check_code, "")
                ),
                "reason": SCREEN_REASON_BY_CODE[reason_code],
                "checks": checks,
                "demand": float(screening_metrics[row, 0]),
                "min_time_window_slack": float(screening_metrics[row, 2]),
                "distance_lower_bound": float(screening_metrics[row, 3]),
                "distance_increment_lower_bound": (
                    None
                    if math.isnan(float(screening_metrics[row, 4]))
                    else float(screening_metrics[row, 4])
                ),
                "single_segment_reachable": bool(screening_metrics[row, 6]),
                "structural_energy_lower_bound": float(
                    screening_metrics[row, 5]
                ),
                "negative_cache_hit": bool(screening_flags[row, 1]),
                "exact_call_blocked": not accepted,
                "physical_evaluated": bool(screening_flags[row, 0]),
                "physical_owner": bool(screening_flags[row, 2]),
                "reachability_queries": int(screening_flags[row, 3]),
                "screening_ordinal": row,
            }
        )
    _append_native_nested_evidence(evidence, screening_payload)
    if (
        not isinstance(producer_sha256, str)
        or not _is_sha256(producer_sha256)
        or hashlib.sha256(evidence).hexdigest() != producer_sha256
    ):
        raise RuntimeError("full native control journal SHA-256 mismatch")
    if final_cache_statistics is None:
        raise RuntimeError("full native control journal contains no transactions")
    cache_names = (
        "cache_lookups",
        "cache_hits",
        "cache_misses",
        "cache_stores",
        "cache_evictions",
        "cache_oversize_not_cached",
        "entries_current",
        "entries_peak",
        "bytes_current",
        "bytes_peak",
        "unique_route_evaluations",
    )
    return (
        tuple(events),
        producer_sha256,
        {
            "candidate_decisions": decision_count,
            "selected_candidates": selected_count,
            "skipped_candidates": skipped_count,
        },
        {
            name: int(final_cache_statistics[index])
            for index, name in enumerate(cache_names)
        },
        {
            "screening_calls": int(screening_statistics_array[5]),
            "screening_passes": int(screening_statistics_array[6]),
            "screening_rejections": int(screening_statistics_array[7]),
            "screening_cache_hits": int(screening_statistics_array[3]),
            "screening_exact_call_blocked": int(screening_statistics_array[4]),
            "screening_runtime_seconds": float(screening_timing[0]),
            "screening_reason_counts": {
                SCREEN_REASON_BY_CODE.get(int(code), f"unknown:{int(code)}"): int(count)
                for code, count in zip(reason_codes, reason_counts, strict=True)
            },
            "screening_semantic_decisions": int(screening_statistics_array[0]),
            "screening_semantic_passes": int(screening_statistics_array[1]),
            "screening_semantic_rejections": int(screening_statistics_array[2]),
            "screening_semantic_event_count": screening_row_count,
            "screening_physical_owner_count": int(
                np.count_nonzero(screening_flags[:, 0] & screening_flags[:, 2])
            ),
            "station_reachability_queries": int(np.sum(screening_flags[:, 3])),
            "candidate_screening_occupancies": [
                int(value) for value in screening_occupancies
            ],
            "candidate_screening_maximum_occupancy": int(
                np.max(screening_occupancies, initial=0)
            ),
        },
    )


def _decode_full_native_causal_journal(payload: object) -> NativeCausalJournal:
    """Validate the native cross-domain causal receipt without filling gaps."""

    if not isinstance(payload, tuple) or len(payload) != 12:
        raise RuntimeError("full native causal journal has an invalid envelope")
    raw_columns = payload[:10]
    columns = tuple(
        _require_vector(value, f"full native causal journal column {index}")
        for index, value in enumerate(raw_columns)
    )
    event_count = len(columns[0])
    if any(len(column) != event_count for column in columns):
        raise RuntimeError("full native causal journal columns do not align")
    event_ids = columns[0]
    if not np.array_equal(event_ids, np.arange(event_count, dtype=np.int64)):
        raise RuntimeError("full native causal journal IDs are not contiguous")
    stream_codes = columns[1]
    if np.any(stream_codes < 0) or np.any(stream_codes >= 8):
        raise RuntimeError("full native causal journal has an unknown stream code")
    event_codes = columns[2]
    if np.any(event_codes < 1) or np.any(event_codes > 11):
        raise RuntimeError("full native causal journal has an invalid event code")
    allowed_events = {
        0: frozenset({9}),
        1: frozenset({1}),
        2: frozenset({2}),
        3: frozenset({7, 8}),
        4: frozenset({3, 6}),
        5: frozenset({10}),
        6: frozenset({4, 11}),
        7: frozenset({5}),
    }
    if any(
        int(event) not in allowed_events[int(stream)]
        for stream, event in zip(stream_codes, event_codes, strict=True)
    ):
        raise RuntimeError(
            "full native causal event does not belong to its stream"
        )
    lane_ids = columns[3]
    operator_ids = columns[4]
    iterations = columns[5]
    transaction_ids = columns[6]
    subject_ids = columns[7]
    status_codes = columns[8]
    flags = columns[9]
    if (
        np.any(iterations < -1)
        or np.any(status_codes < 0)
        or np.any(flags < 0)
    ):
        raise RuntimeError("full native causal event fields are invalid")
    boundary = (stream_codes == 6) | (stream_codes == 7)
    if (
        np.any(lane_ids[boundary] != -1)
        or np.any(operator_ids[boundary] != -1)
        or np.any(iterations[boundary] < 0)
        or np.any(transaction_ids[boundary] != -1)
        or np.any(subject_ids[boundary] != -1)
        or np.any(lane_ids[~boundary] < 0)
        or np.any(operator_ids[~boundary] < 0)
        or np.any(transaction_ids[~boundary] < 0)
        or np.any(subject_ids[~boundary] < 0)
    ):
        raise RuntimeError("full native causal transaction fields are invalid")
    terminal_rows = np.flatnonzero(stream_codes == 7)
    if (
        len(terminal_rows) != 1
        or int(terminal_rows[0]) != event_count - 1
        or int(event_codes[terminal_rows[0]]) != 5
        or int(status_codes[terminal_rows[0]]) not in {0, 1, 2, 3}
        or int(flags[terminal_rows[0]]) != 0
    ):
        raise RuntimeError("full native causal termination is invalid")
    terminal_reason = int(status_codes[terminal_rows[0]])
    deadline_rows = np.flatnonzero(stream_codes == 6)
    expected_deadline_rows = 1 if terminal_reason in {1, 2, 3} else 0
    if len(deadline_rows) != expected_deadline_rows:
        raise RuntimeError("full native causal deadline boundary is incomplete")
    if deadline_rows.size:
        row = int(deadline_rows[0])
        if (
            row != event_count - 2
            or int(status_codes[row]) != terminal_reason
            or int(flags[row]) != 1
            or (
                terminal_reason == 2
                and int(event_codes[row]) != 4
            )
            or (
                terminal_reason != 2
                and int(event_codes[row]) != 11
            )
        ):
            raise RuntimeError("full native causal deadline boundary is invalid")
    status_domains = {
        1: frozenset({0, 1, 2, 3}),
        2: frozenset({0, 1}),
        3: frozenset({0, 1}),
        7: frozenset({0, 1}),
        8: frozenset({0, 1}),
        9: frozenset({0, 1}),
        10: frozenset({0, 1}),
    }
    for row, event in enumerate(event_codes):
        domain = status_domains.get(int(event))
        if domain is not None and int(status_codes[row]) not in domain:
            raise RuntimeError("full native causal status code is invalid")
        if int(event) == 6 and (
            int(subject_ids[row]) <= 0
            or int(status_codes[row]) != int(subject_ids[row])
        ):
            raise RuntimeError("full native causal exact-work row is invalid")
    stream_counts = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[10],
            dtype=np.dtype(np.int64),
            shape=(8,),
            name="full native causal stream counts",
        ),
    )
    observed_counts = np.bincount(stream_codes, minlength=8).astype(
        np.int64,
        copy=False,
    )
    if not np.array_equal(stream_counts, observed_counts):
        raise RuntimeError("full native causal stream counts do not reconcile")
    producer_sha256 = payload[11]
    evidence = bytearray(b"stage05.2-native-causal-journal-v2")
    for column in (*columns, stream_counts):
        _append_typed_array(evidence, column)
    if (
        not isinstance(producer_sha256, str)
        or not _is_sha256(producer_sha256)
        or hashlib.sha256(evidence).hexdigest() != producer_sha256
    ):
        raise RuntimeError("full native causal journal SHA-256 mismatch")
    readonly = tuple(_readonly_copy(column) for column in columns)
    return NativeCausalJournal(
        event_ids=readonly[0],
        stream_codes=readonly[1],
        event_codes=readonly[2],
        lane_ids=readonly[3],
        operator_ids=readonly[4],
        iterations=readonly[5],
        transaction_ids=readonly[6],
        subject_ids=readonly[7],
        status_codes=readonly[8],
        flags=readonly[9],
        stream_counts=_readonly_copy(stream_counts),
        transaction_sha256=producer_sha256,
    )


def _decode_native_initial_state_receipt(
    payload: object,
) -> NativeInitialStateReceipt:
    """Validate the causal ownership receipt for native initial-state work."""

    if not isinstance(payload, tuple) or len(payload) != 4:
        raise RuntimeError("full native initial-state receipt has an invalid tuple")
    flags = _require_array(
        payload[0],
        dtype=np.dtype(np.int64),
        shape=(2,),
        name="full native initial-state receipt flags",
    )
    host_owned = int(flags[0])
    operation_count = int(flags[1])
    if host_owned not in (0, 1) or operation_count != host_owned:
        raise RuntimeError("full native initial-state receipt ownership is invalid")
    request_sha256, state_sha256, transaction_sha256 = payload[1:]
    if not all(
        isinstance(value, str) and _is_sha256(value)
        for value in (request_sha256, state_sha256, transaction_sha256)
    ):
        raise RuntimeError("full native initial-state receipt SHA-256 is invalid")
    evidence = bytearray(b"stage05.2-native-initial-state-receipt-v2")
    evidence.extend(flags.tobytes(order="C"))
    evidence.extend(request_sha256.encode("ascii"))
    evidence.extend(state_sha256.encode("ascii"))
    if hashlib.sha256(evidence).hexdigest() != transaction_sha256:
        raise RuntimeError("full native initial-state receipt SHA-256 mismatch")
    return NativeInitialStateReceipt(
        host_owned=bool(host_owned),
        operation_count=operation_count,
        request_sha256=request_sha256,
        state_sha256=state_sha256,
        transaction_sha256=transaction_sha256,
    )


def execute_full_native_alns(
    instance: Instance,
    *,
    seed: int,
    max_iterations: int,
    deadline: float,
    batch_size: int,
    compute_threads: int,
    native_runtime: NativeKernelRuntime,
    initial_customer_sequences: tuple[CustomerSequence, ...],
    candidate_control_config: CandidateControlConfig,
    stage04_config: Stage04Config,
    vehicle_operator_config: VehicleOperatorConfig,
    screening_config: CheapScreeningConfig,
    cache_incremental_config: CacheIncrementalConfig,
    removal_fraction: float,
    termination_mode: str,
    operator_profile: str,
    exact_call_budget: int | None = None,
    dispatcher: Callable[..., object] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> FullNativeALNSResult:
    """Dispatch one instance/seed to the full-native ALNS ABI."""

    if max_iterations <= 0 or batch_size <= 0 or compute_threads <= 0:
        raise ValueError("full native iteration/batch/thread counts must be positive")
    if not initial_customer_sequences:
        raise ValueError("full native v2 requires an explicit non-empty warm start")
    if operator_profile != "stage02_constraint_guided":
        raise ValueError("full native v2 requires the Stage 2.3 operator profile")
    if termination_mode not in {"fixed_work", "wall_clock"}:
        raise ValueError("full native v2 termination mode is invalid")
    if not 0.0 < removal_fraction <= 1.0:
        raise ValueError("full native v2 removal fraction must be in (0, 1]")
    if not stage04_config.enabled:
        raise ValueError("full native v2 requires enabled Stage 4 search control")
    if not screening_config.enabled or not cache_incremental_config.enabled:
        raise ValueError("full native v2 requires screening and cache/incremental control")
    if deadline - clock() <= 0.0:
        raise RuntimeError("full native ALNS reached its deadline before dispatch")
    context = native_runtime.context
    context.assert_matches(instance)
    ordered_names = sorted(context.node_names)
    rank_by_name = {name: rank for rank, name in enumerate(ordered_names)}
    lexical_rank = np.ascontiguousarray(
        [rank_by_name[name] for name in context.node_names],
        dtype=np.int64,
    )
    encoded_node_names = tuple(name.encode("utf-8") for name in context.node_names)
    node_name_offsets = np.ascontiguousarray(
        np.cumsum([0, *(len(value) for value in encoded_node_names)]),
        dtype=np.int64,
    )
    node_name_bytes = np.frombuffer(b"".join(encoded_node_names), dtype=np.uint8).copy()
    initial_offsets, initial_indices = _pack_routes(
        initial_customer_sequences,
        context.name_to_index,
    )
    if exact_call_budget is not None and exact_call_budget <= 0:
        raise ValueError("full native exact-call budget must be positive")
    control = np.ascontiguousarray(
        [
            seed,
            max_iterations,
            batch_size,
            compute_threads,
            -1 if exact_call_budget is None else exact_call_budget,
        ],
        dtype=np.int64,
    )
    protocol_control = np.ascontiguousarray(
        [
            1 if termination_mode == "fixed_work" else 0,
            candidate_control_config.proposal_top_k,
            candidate_control_config.max_exact_calls_per_round,
            candidate_control_config.worker_count,
            candidate_control_config.fixed_work_exhaustion_rounds,
            candidate_control_config.min_iterations_before_exhaustion,
            1 if screening_config.negative_sequence_cache else 0,
            cache_incremental_config.max_entries,
            cache_incremental_config.max_memory_bytes,
            1 if cache_incremental_config.shared_across_lanes else 0,
            1 if cache_incremental_config.incremental_relocate else 0,
            1 if cache_incremental_config.incremental_swap else 0,
            1 if cache_incremental_config.station_reachability_bitset else 0,
        ],
        dtype=np.int64,
    )
    protocol_options = np.ascontiguousarray(
        [removal_fraction, screening_config.epsilon],
        dtype=np.float64,
    )
    stage04_integer = _pack_integer_config(
        stage04_config,
        FULL_NATIVE_STAGE04_INTEGER_FIELDS,
    )
    stage04_float = _pack_float_config(
        stage04_config,
        FULL_NATIVE_STAGE04_FLOAT_FIELDS,
    )
    operator_integer = _pack_integer_config(
        vehicle_operator_config,
        FULL_NATIVE_OPERATOR_INTEGER_FIELDS,
    )
    operator_float = _pack_float_config(
        vehicle_operator_config,
        FULL_NATIVE_OPERATOR_FLOAT_FIELDS,
    )
    remaining = deadline - clock()
    if remaining <= 0.0:
        raise RuntimeError("full native ALNS reached its deadline during input packing")
    deadline_remaining = np.ascontiguousarray([remaining], dtype=np.float64)

    from evrptw import _core as native_core

    native_entrypoint = native_core.full_native_alns_v2 if dispatcher is None else dispatcher
    payload = native_entrypoint(
        context.node_kind,
        context.demand,
        context.ready_time,
        context.due_date,
        context.service_time,
        context.distance,
        context.reachable,
        context.vehicle,
        lexical_rank,
        node_name_offsets,
        node_name_bytes,
        initial_offsets,
        initial_indices,
        control,
        deadline_remaining,
        protocol_control,
        protocol_options,
        stage04_integer,
        stage04_float,
        operator_integer,
        operator_float,
    )
    if not isinstance(payload, tuple) or len(payload) != 13:
        raise RuntimeError("full native ALNS returned an invalid payload tuple")
    route_offsets = _require_vector(payload[0], "full native route offsets")
    route_indices = _require_vector(payload[1], "full native route indices")
    if (
        len(route_offsets) < 2
        or int(route_offsets[0]) != 0
        or int(route_offsets[-1]) != len(route_indices)
        or np.any(route_offsets[:-1] > route_offsets[1:])
    ):
        raise RuntimeError("full native ALNS returned invalid route offsets")
    customer_sequences = tuple(
        tuple(
            context.node_names[int(index)]
            for index in route_indices[
                int(route_offsets[route]) : int(route_offsets[route + 1])
            ]
        )
        for route in range(len(route_offsets) - 1)
    )
    counters_array = _require_array(
        payload[3],
        dtype=np.dtype(np.int64),
        shape=(8,),
        name="full native counters",
    )
    timings_array = _require_array(
        payload[4],
        dtype=np.dtype(np.float64),
        shape=(13,),
        name="full native timings",
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in timings_array):
        raise RuntimeError("full native ALNS returned invalid timings")
    if not math.isclose(
        float(timings_array[0] + timings_array[1]),
        float(timings_array[2]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("full native ALNS timing intervals do not reconcile")
    telemetry_values = timings_array[4:13]
    if any(float(value) != int(value) for value in telemetry_values):
        raise RuntimeError("full native ALNS concurrency telemetry is not integral")
    initial_state_receipt = _decode_native_initial_state_receipt(payload[12])
    if (
        int(timings_array[12]) != initial_state_receipt.operation_count
        or bool(int(timings_array[7])) != initial_state_receipt.host_owned
    ):
        raise RuntimeError(
            "full native initial-state ownership receipt does not reconcile"
        )
    if int(timings_array[5]) != 0:
        raise RuntimeError("full native ALNS returned with active native work")
    if int(timings_array[7]) not in (0, 1):
        raise RuntimeError("full native ALNS shared-pool telemetry is invalid")
    if int(timings_array[8]) not in (1, 4, 24):
        raise RuntimeError("full native ALNS work-pool size is invalid")
    if int(timings_array[7]) == 1:
        if int(timings_array[8]) != 24 or int(timings_array[9]) != 0:
            raise RuntimeError("host scheduler thread topology is invalid")
        if int(timings_array[10]) <= 0:
            raise RuntimeError("host scheduler reported no remote kernel requests")
        if int(timings_array[11]) <= 0:
            raise RuntimeError("host scheduler reported no screening-batch requests")
        if int(timings_array[12]) != 1:
            raise RuntimeError(
                "host scheduler did not own exactly one initial search state"
            )
    elif (
        int(timings_array[9]) != 0
        or int(timings_array[10]) != 0
        or int(timings_array[11]) != 0
        or int(timings_array[12]) != 0
    ):
        raise RuntimeError("local full-native reported host-only telemetry")
    trajectory_array = payload[5]
    if (
        not isinstance(trajectory_array, np.ndarray)
        or trajectory_array.dtype != np.dtype(np.int64)
        or trajectory_array.ndim != 2
        or trajectory_array.shape[1] != 7
        or not trajectory_array.flags.c_contiguous
    ):
        raise RuntimeError("full native ALNS returned an invalid trajectory array")
    if np.any(trajectory_array[:, 1] < 0) or np.any(
        trajectory_array[:, 1] >= len(FULL_NATIVE_OPERATOR_NAMES)
    ):
        raise RuntimeError("full native ALNS returned an unknown operator id")
    transaction_sha256 = payload[6]
    if not isinstance(transaction_sha256, str) or not _is_sha256(transaction_sha256):
        raise RuntimeError("full native ALNS returned an invalid transaction SHA-256")
    three_lane_semantics = isinstance(payload[7], tuple) and len(payload[7]) in {
        2,
        14,
    }
    semantic_stream: NativeGlobalSemanticStream | NativeThreeLaneSemanticStream
    if three_lane_semantics:
        semantic_stream = (
            decode_native_three_lane_search_semantic_stream(
                instance,
                payload[7],
                node_names=context.node_names,
                initial_customer_sequences=initial_customer_sequences,
                vehicle_operator_config=vehicle_operator_config,
            )
            if len(cast(tuple[object, ...], payload[7])) == 2
            else decode_native_three_lane_semantic_stream(
                instance,
                payload[7],
                node_names=context.node_names,
                initial_customer_sequences=initial_customer_sequences,
            )
        )
    else:
        semantic_stream = decode_native_global_semantic_stream(
            instance,
            payload[7],
            node_names=context.node_names,
            initial_customer_sequences=initial_customer_sequences,
            stage04_config=stage04_config,
            expected_start_iteration=0,
            expected_iteration_count=max_iterations,
        )
    semantic_stream = replace(
        semantic_stream,
        neighborhood_events=_python_neighborhood_event_projection(
            semantic_stream.neighborhood_events
        ),
    )
    backend_metrics = _decode_full_native_backend_metrics(
        payload[8],
        batch_size=batch_size,
        exact_seconds=float(timings_array[1]),
    )
    (
        candidate_work_hash,
        route_result_hash,
        exact_journal_events,
        exact_journal_sha256,
        candidate_work,
        route_results,
    ) = _decode_full_native_exact_journal(
        instance,
        payload[9],
        native_runtime=native_runtime,
        batch_size=batch_size,
    )
    (
        control_journal_events,
        control_journal_sha256,
        control_journal_statistics,
        route_cache_statistics,
        screening_statistics,
    ) = _decode_full_native_control_journal(
        payload[10],
        node_names=context.node_names,
    )
    causal_journal = _decode_full_native_causal_journal(payload[11])
    expected_sha256 = _full_native_digest(
        route_offsets=route_offsets,
        route_indices=route_indices,
        exact_payload=payload[2],
        counters=counters_array,
        trajectory=trajectory_array,
        semantic_sha256=semantic_stream.transaction_sha256,
        backend_payload=payload[8],
        exact_journal_sha256=exact_journal_sha256,
        control_journal_sha256=control_journal_sha256,
        causal_journal_sha256=causal_journal.transaction_sha256,
        initial_state_receipt=payload[12],
    )
    if transaction_sha256 != expected_sha256:
        raise RuntimeError("full native ALNS transaction SHA-256 mismatch")
    if (
        not three_lane_semantics
        and int(semantic_stream.termination[2]) != int(counters_array[0])
    ):
        raise RuntimeError("full native ALNS semantic iteration count mismatch")
    decoded = decode_exact_charging_batch_numeric(
        instance,
        customer_sequences,
        native_runtime=native_runtime,
        batch_size=batch_size,
        payload=payload[2],
        native_kernel_seconds=0.0,
    )
    if three_lane_semantics:
        raw_semantics = cast(tuple[object, ...], payload[7])
        final_semantics = (
            cast(
                tuple[object, ...],
                cast(tuple[object, ...], raw_semantics[0])[-1],
            )
            if len(raw_semantics) == 2
            else raw_semantics
        )
        raw_best = cast(tuple[object, ...], final_semantics[9])
        replayed_best = _pack_routes(customer_sequences, context.name_to_index)
        if not (
            np.array_equal(
                _require_vector(raw_best[0], "full native best offsets"),
                replayed_best[0],
            )
            and np.array_equal(
                _require_vector(raw_best[1], "full native best indices"),
                replayed_best[1],
            )
        ):
            raise RuntimeError("full native three-lane best projection is invalid")
    else:
        _verify_full_native_best_projection(
            instance,
            initial_customer_sequences=initial_customer_sequences,
            semantic_stream=cast(NativeGlobalSemanticStream, semantic_stream),
            returned_customer_sequences=customer_sequences,
        )
    counters = {
        "iterations": int(counters_array[0]),
        "exact_started_calls": int(counters_array[1]),
        "exact_completed_calls": int(counters_array[2]),
        "accepted_moves": int(counters_array[3]),
        "improving_moves": int(counters_array[4]),
        "rejected_moves": int(counters_array[5]),
        "interrupted_calls": int(counters_array[6]),
        "fallback_count": int(counters_array[7]),
    }
    if counters["fallback_count"] != 0:
        raise RuntimeError("full native ALNS reported a forbidden fallback")
    if (
        counters["exact_started_calls"] != backend_metrics.started_calls
        or counters["exact_completed_calls"] != backend_metrics.completed_calls
        or counters["interrupted_calls"] != backend_metrics.interrupted_calls
    ):
        raise RuntimeError("full native ALNS backend counters do not reconcile")
    return FullNativeALNSResult(
        customer_sequences=customer_sequences,
        exact_results=decoded.results,
        backend_metrics=backend_metrics,
        counters=counters,
        timings={
            "search_seconds": float(timings_array[0]),
            "exact_seconds": float(timings_array[1]),
            "total_seconds": float(timings_array[2]),
            "queue_wait_seconds": float(timings_array[3]),
            "work_pool_peak_active_tasks": float(timings_array[4]),
            "work_pool_active_tasks_at_return": float(timings_array[5]),
            "queue_depth_on_submit": float(timings_array[6]),
            "shared_work_pool": float(timings_array[7]),
            "work_pool_thread_count": float(timings_array[8]),
            "client_dispatch_thread_count": float(timings_array[9]),
            "remote_kernel_request_count": float(timings_array[10]),
            "screening_batch_request_count": float(timings_array[11]),
            "initial_state_request_count": float(timings_array[12]),
        },
        trajectory=tuple(
            {
                "iteration": int(row[0]),
                "operator_id": int(row[1]),
                "operator": FULL_NATIVE_OPERATOR_NAMES[int(row[1])],
                "route_count": int(row[2]),
                "started_exact_calls": int(row[3]),
                "status_code": int(row[4]),
                "accepted": int(row[5]),
                "vehicle_reduction": int(row[6]),
            }
            for row in trajectory_array
        ),
        semantic_stream=semantic_stream,
        transaction_sha256=transaction_sha256,
        candidate_work_hash=candidate_work_hash,
        route_result_hash=route_result_hash,
        exact_journal_events=exact_journal_events,
        exact_journal_sha256=exact_journal_sha256,
        candidate_work=candidate_work,
        route_results=route_results,
        control_journal_events=control_journal_events,
        control_journal_sha256=control_journal_sha256,
        control_journal_statistics=control_journal_statistics,
        route_cache_statistics=route_cache_statistics,
        screening_statistics=screening_statistics,
        causal_journal=causal_journal,
        initial_state_receipt=initial_state_receipt,
    )


def _decode_candidate_resource_receipt(
    values: npt.NDArray[np.int64],
    *,
    fail_closed_started_calls: int,
) -> NativeCandidateResourceReceipt:
    """Decode caller-owned receipt memory, charging conservatively if corrupt."""

    raw = tuple(int(value) for value in values)
    if (
        len(raw) == 6
        and raw[0] == 2
        and 0 <= raw[1] <= 4
        and all(value >= 0 for value in raw[2:])
        and raw[5] == 0
        and (raw[1] >= 2 or raw[2:5] == (0, 0, 0))
        and (raw[1] < 2 or raw[3] + raw[4] <= raw[2])
    ):
        unaccounted = raw[2] - raw[3] - raw[4]
        return NativeCandidateResourceReceipt(
            phase=raw[1],
            started_calls=raw[2],
            completed_calls=raw[3],
            interrupted_calls=raw[4] + unaccounted,
            fallback_count=raw[5],
            fail_closed=unaccounted > 0,
        )
    return NativeCandidateResourceReceipt(
        phase=-1,
        started_calls=fail_closed_started_calls,
        completed_calls=0,
        interrupted_calls=fail_closed_started_calls,
        fallback_count=0,
        fail_closed=True,
    )


def _reconcile_candidate_resource_receipt_fail_closed(
    receipt: NativeCandidateResourceReceipt,
    backend_metrics: BackendMetrics,
) -> NativeCandidateResourceReceipt:
    """Merge disagreeing authorities without ever lowering observed work."""

    started = max(receipt.started_calls, backend_metrics.started_calls)
    completed = min(
        started,
        max(receipt.completed_calls, backend_metrics.completed_calls),
    )
    return NativeCandidateResourceReceipt(
        phase=receipt.phase,
        started_calls=started,
        completed_calls=completed,
        interrupted_calls=started - completed,
        fallback_count=max(receipt.fallback_count, backend_metrics.native_fallbacks),
        fail_closed=True,
    )


def _execute_native_candidate_round(
    instance: Instance,
    request: NativeCandidateRoundRequest,
    *,
    native_runtime: NativeKernelRuntime,
    transaction_runtime: NativeCandidateTransactionRuntime,
    negative_cache: Mapping[CustomerSequence, str],
    resource_receipt_values: npt.NDArray[np.int64],
    clock: Callable[[], float] = time.perf_counter,
    record_transaction: bool = True,
    record_runtime: bool = True,
) -> NativeCandidateRoundResult:
    """Execute screening, ranking, cache decisions and exact work in one C++ call."""

    native_runtime.context.assert_matches(instance)
    remaining = request.deadline - clock()
    if remaining <= 0.0:
        raise RuntimeError("native candidate round reached its deadline before dispatch")
    context = native_runtime.context
    route_offsets, route_indices = _pack_routes(
        request.candidates,
        context.name_to_index,
    )
    candidate_ids = np.arange(len(request.candidates), dtype=np.int64)
    ordered_names = sorted(context.node_names)
    rank_by_name = {name: rank for rank, name in enumerate(ordered_names)}
    lexical_rank = np.ascontiguousarray(
        [rank_by_name[name] for name in context.node_names],
        dtype=np.int64,
    )
    options = np.ascontiguousarray(
        [float(request.full_screening), context.reachability_epsilon, 0.0, 0.0],
        dtype=np.float64,
    )
    incremental = (
        np.zeros((len(request.candidates), 6), dtype=np.float64)
        if request.incremental is None
        else cast(
            npt.NDArray[np.float64],
            _require_array(
                request.incremental,
                dtype=np.dtype(np.float64),
                shape=(len(request.candidates), 6),
                name="incremental",
            ),
        )
    )
    negative_offsets, negative_indices, negative_reason_codes = (
        transaction_runtime.packed_negative_cache(
            negative_cache,
            context.name_to_index,
        )
    )
    cache_hit_flags = np.ascontiguousarray(request.cache_hit_flags, dtype=np.int64)
    control = np.ascontiguousarray(
        [request.proposal_top_k, request.exact_budget, request.compute_threads],
        dtype=np.int64,
    )
    deadline_remaining = np.ascontiguousarray([remaining], dtype=np.float64)
    batch_size = np.ascontiguousarray([request.batch_size], dtype=np.int64)
    context_ids = np.ascontiguousarray(
        [
            _stable_int63(request.lane),
            _stable_int63(request.operator),
            -1 if request.iteration is None else request.iteration,
        ],
        dtype=np.int64,
    )

    from evrptw import _core as native_core

    payload = native_core.candidate_round_transaction_v2(
        context.node_kind,
        context.demand,
        context.ready_time,
        context.due_date,
        context.service_time,
        context.distance,
        context.reachable,
        context.vehicle,
        route_offsets,
        route_indices,
        candidate_ids,
        lexical_rank,
        options,
        incremental,
        negative_offsets,
        negative_indices,
        negative_reason_codes,
        cache_hit_flags,
        control,
        deadline_remaining,
        batch_size,
        context_ids,
        resource_receipt_values,
    )
    if not isinstance(payload, tuple) or len(payload) != 10:
        raise RuntimeError("native candidate round returned an invalid payload tuple")
    screening = decode_native_candidate_screening_payload(
        payload[0],
        candidates=request.candidates,
        candidate_ids=candidate_ids,
        route_offsets=route_offsets,
        route_indices=route_indices,
    )
    resolution_codes = _require_array(
        payload[1],
        dtype=np.dtype(np.int64),
        shape=(len(request.candidates),),
        name="resolution codes",
    )
    try:
        resolutions = tuple(
            _RESOLUTION_BY_CODE[int(code)] for code in resolution_codes
        )
    except KeyError as error:
        raise RuntimeError("native candidate round returned an unknown resolution") from error
    sources = _require_array(
        payload[2],
        dtype=np.dtype(np.int64),
        shape=(len(request.candidates),),
        name="resolution sources",
    )
    cache_journal = cast(
        npt.NDArray[np.int64],
        _require_array(
            payload[3],
            dtype=np.dtype(np.int64),
            shape=(len(request.candidates), 3),
            name="cache journal",
        ),
    )
    if not np.array_equal(cache_journal[:, 0], candidate_ids):
        raise RuntimeError("native candidate round changed cache-journal identities")
    exact_ids_array = _require_vector(payload[4], "exact candidate ids")
    completion_array = _require_vector(payload[5], "completion order")
    exact_ids = tuple(int(value) for value in exact_ids_array)
    completion_order = tuple(int(value) for value in completion_array)
    if len(set(exact_ids)) != len(exact_ids) or any(
        candidate_id < 0 or candidate_id >= len(request.candidates)
        for candidate_id in exact_ids
    ):
        raise RuntimeError("native candidate round returned invalid exact identities")
    if sorted(completion_order) != sorted(exact_ids):
        raise RuntimeError("native candidate round completion order is not a permutation")
    counters_array = _require_array(
        payload[7],
        dtype=np.dtype(np.int64),
        shape=(10,),
        name="counters",
    )
    timings_array = _require_array(
        payload[8],
        dtype=np.dtype(np.float64),
        shape=(4,),
        name="timings",
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in timings_array):
        raise RuntimeError("native candidate round returned invalid timings")
    transaction_sha256 = payload[9]
    if not isinstance(transaction_sha256, str) or not _is_sha256(transaction_sha256):
        raise RuntimeError("native candidate round returned an invalid transaction SHA-256")
    expected_transaction_sha256 = _candidate_round_digest(
        context_ids=context_ids,
        resolution_codes=resolution_codes,
        sources=sources,
        cache_journal=cache_journal,
        exact_candidate_ids=exact_ids_array,
        completion_order=completion_array,
        exact_payload=payload[6],
        screening_sha256=screening.candidate_pool_hash,
    )
    if transaction_sha256 != expected_transaction_sha256:
        raise RuntimeError("native candidate round transaction SHA-256 mismatch")

    resource_receipt = _decode_candidate_resource_receipt(
        resource_receipt_values,
        fail_closed_started_calls=request.exact_budget,
    )
    if resource_receipt.phase != 4 or resource_receipt.fail_closed:
        raise RuntimeError("native candidate round resource receipt is incomplete")

    exact_orders = tuple(request.candidates[index] for index in exact_ids)
    decoded = decode_exact_charging_batch_numeric(
        instance,
        exact_orders,
        native_runtime=native_runtime,
        batch_size=request.batch_size,
        payload=payload[6],
        native_kernel_seconds=float(timings_array[1]),
    )
    if record_runtime:
        native_runtime.record_screening(
            float(timings_array[0]),
            batch_candidates=len(request.candidates),
        )
    counters = {
        "input_candidates": int(counters_array[0]),
        "rankable_candidates": int(counters_array[1]),
        "selected_candidates": int(counters_array[2]),
        "screening_rejections": int(counters_array[3]),
        "negative_cache_hits": int(counters_array[4]),
        "cache_hits": int(counters_array[5]),
        "exact_misses": int(counters_array[6]),
        "budget_skips": int(counters_array[7]),
        "duplicate_candidates": int(counters_array[8]),
        "fallback_count": int(counters_array[9]),
    }
    if counters["input_candidates"] != len(request.candidates):
        raise RuntimeError("native candidate round input counter diverged")
    if counters["exact_misses"] != len(exact_ids):
        raise RuntimeError("native candidate round exact counter diverged")
    if counters["fallback_count"] != 0:
        raise RuntimeError("native candidate round reported a forbidden fallback")
    reason_counts: dict[str, int] = {}
    for index in range(len(request.candidates)):
        reason = SCREEN_REASON_BY_CODE[int(screening.codes[index, 1])]
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    screening_passes = sum(
        screening.accepted(index) for index in range(len(request.candidates))
    )
    audit = CandidateTransactionAudit(
        lane=request.lane,
        operator=request.operator,
        iteration=request.iteration,
        input_candidates=len(request.candidates),
        candidates=request.candidates,
        exact_budget=request.exact_budget,
        screening_integrity_evidence=screening.integrity_evidence(),
        screening_passes=screening_passes,
        screening_rejections=counters["screening_rejections"],
        screening_cache_hits=counters["negative_cache_hits"],
        screening_exact_call_blocked=len(request.candidates) - screening_passes,
        screening_reason_counts=dict(sorted(reason_counts.items())),
        cache_hits=counters["cache_hits"],
        exact_misses=counters["exact_misses"],
        budget_skips=counters["budget_skips"],
        duplicate_candidates=counters["duplicate_candidates"],
        screening_pool_hash=screening.candidate_pool_hash,
        transaction_sha256=transaction_sha256,
    )
    if record_transaction:
        transaction_runtime.record(audit)
    return NativeCandidateRoundResult(
        screening=screening,
        resolutions=resolutions,
        resolution_sources=tuple(int(value) for value in sources),
        cache_journal=cache_journal,
        exact_candidate_ids=exact_ids,
        exact_results=decoded.results,
        backend_metrics=decoded.metrics,
        completion_order=completion_order,
        counters=counters,
        timings={
            "screening_seconds": float(timings_array[0]),
            "exact_seconds": float(timings_array[1]),
            "total_seconds": float(timings_array[2]),
            "queue_wait_seconds": float(timings_array[3]),
        },
        transaction_sha256=transaction_sha256,
        audit=audit,
        resource_receipt=resource_receipt,
    )


def execute_native_candidate_round(
    instance: Instance,
    request: NativeCandidateRoundRequest,
    *,
    native_runtime: NativeKernelRuntime,
    transaction_runtime: NativeCandidateTransactionRuntime,
    negative_cache: Mapping[CustomerSequence, str],
    clock: Callable[[], float] = time.perf_counter,
    record_transaction: bool = True,
    record_runtime: bool = True,
) -> NativeCandidateRoundResult:
    """Execute one round and preserve native accounting across every failure path."""

    resource_receipt_values = np.zeros(6, dtype=np.int64)
    resource_receipt_values[0] = 2
    try:
        result = _execute_native_candidate_round(
            instance,
            request,
            native_runtime=native_runtime,
            transaction_runtime=transaction_runtime,
            negative_cache=negative_cache,
            resource_receipt_values=resource_receipt_values,
            clock=clock,
            record_transaction=record_transaction,
            record_runtime=record_runtime,
        )
    except NativeCandidateRoundFailure:
        raise
    except BaseException as error:
        receipt = _decode_candidate_resource_receipt(
            resource_receipt_values,
            fail_closed_started_calls=request.exact_budget,
        )
        # An error before the C++ entry point preserves its historical Python
        # exception type; once native validation began, callers need the typed
        # receipt even if no exact work was selected.
        if receipt.phase == 0 and not receipt.fail_closed:
            raise
        raise NativeCandidateRoundFailure(
            str(error),
            resource_receipt=receipt,
        ) from error
    if result.resource_receipt.started_calls != result.backend_metrics.started_calls:
        fail_closed_receipt = _reconcile_candidate_resource_receipt_fail_closed(
            result.resource_receipt,
            result.backend_metrics,
        )
        raise NativeCandidateRoundFailure(
            "native candidate-round backend and resource receipts diverged",
            resource_receipt=fail_closed_receipt,
        )
    if (
        result.resource_receipt.completed_calls
        != result.backend_metrics.completed_calls
        or result.resource_receipt.interrupted_calls
        != result.backend_metrics.interrupted_calls
    ):
        fail_closed_receipt = _reconcile_candidate_resource_receipt_fail_closed(
            result.resource_receipt,
            result.backend_metrics,
        )
        raise NativeCandidateRoundFailure(
            "native candidate-round completion receipt diverged",
            resource_receipt=fail_closed_receipt,
        )
    return result


def _pack_routes(
    routes: tuple[CustomerSequence, ...],
    name_to_index: Mapping[str, int],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    offsets = [0]
    indices: list[int] = []
    for route in routes:
        try:
            indices.extend(name_to_index[name] for name in route)
        except KeyError as error:
            raise ValueError(
                f"candidate route contains unknown customer {error.args[0]!r}"
            ) from error
        offsets.append(len(indices))
    return (
        np.ascontiguousarray(offsets, dtype=np.int64),
        np.ascontiguousarray(indices, dtype=np.int64),
    )


def _stable_int63(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _require_array(
    value: object,
    *,
    dtype: np.dtype[np.generic],
    shape: tuple[int, ...],
    name: str,
) -> npt.NDArray[np.generic]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != dtype
        or value.shape != shape
        or not value.flags.c_contiguous
    ):
        raise RuntimeError(f"native candidate round {name} has an invalid schema")
    return value


def _require_vector(value: object, name: str) -> npt.NDArray[np.int64]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.int64)
        or value.ndim != 1
        or not value.flags.c_contiguous
    ):
        raise RuntimeError(f"native candidate round {name} has an invalid schema")
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _candidate_round_digest(
    *,
    context_ids: npt.NDArray[np.int64],
    resolution_codes: npt.NDArray[np.generic],
    sources: npt.NDArray[np.generic],
    cache_journal: npt.NDArray[np.int64],
    exact_candidate_ids: npt.NDArray[np.int64],
    completion_order: npt.NDArray[np.int64],
    exact_payload: object,
    screening_sha256: str,
) -> str:
    evidence = bytearray(b"stage05.2-candidate-round-transaction-v2")
    for values in (
        context_ids,
        resolution_codes,
        sources,
        cache_journal.reshape(-1),
        exact_candidate_ids,
        completion_order,
    ):
        for value in values:
            evidence.extend(int(value).to_bytes(8, "little", signed=True))
    if not isinstance(exact_payload, tuple) or len(exact_payload) != 7:
        raise RuntimeError("native candidate round exact payload has an invalid schema")
    for value in exact_payload:
        if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
            raise RuntimeError("native candidate round exact payload is not contiguous")
        evidence.extend(value.tobytes(order="C"))
    evidence.extend(screening_sha256.encode("ascii"))
    return hashlib.sha256(evidence).hexdigest()


def _full_native_digest(
    *,
    route_offsets: npt.NDArray[np.int64],
    route_indices: npt.NDArray[np.int64],
    exact_payload: object,
    counters: npt.NDArray[np.generic],
    trajectory: npt.NDArray[np.int64],
    semantic_sha256: str,
    backend_payload: object,
    exact_journal_sha256: str,
    control_journal_sha256: str,
    causal_journal_sha256: str,
    initial_state_receipt: object,
) -> str:
    evidence = bytearray(b"stage05.2-full-native-alns-v2")
    evidence.extend(route_offsets.tobytes(order="C"))
    evidence.extend(route_indices.tobytes(order="C"))
    if not isinstance(exact_payload, tuple) or len(exact_payload) != 7:
        raise RuntimeError("full native ALNS exact payload has an invalid schema")
    for value in exact_payload:
        if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
            raise RuntimeError("full native ALNS exact payload is not contiguous")
        evidence.extend(value.tobytes(order="C"))
    evidence.extend(counters.tobytes(order="C"))
    evidence.extend(trajectory.tobytes(order="C"))
    if not _is_sha256(semantic_sha256):
        raise RuntimeError("full native ALNS semantic SHA-256 is invalid")
    evidence.extend(semantic_sha256.encode("ascii"))
    if not isinstance(backend_payload, tuple) or len(backend_payload) != 3:
        raise RuntimeError("full native ALNS backend payload has an invalid schema")
    for value in backend_payload[:2]:
        if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
            raise RuntimeError("full native ALNS backend payload is not contiguous")
        evidence.extend(value.tobytes(order="C"))
    if not _is_sha256(exact_journal_sha256):
        raise RuntimeError("full native ALNS exact journal SHA-256 is invalid")
    evidence.extend(exact_journal_sha256.encode("ascii"))
    if not _is_sha256(control_journal_sha256):
        raise RuntimeError("full native ALNS control journal SHA-256 is invalid")
    evidence.extend(control_journal_sha256.encode("ascii"))
    if not _is_sha256(causal_journal_sha256):
        raise RuntimeError("full native ALNS causal journal SHA-256 is invalid")
    evidence.extend(causal_journal_sha256.encode("ascii"))
    receipt = _decode_native_initial_state_receipt(initial_state_receipt)
    flags = cast(tuple[object, ...], initial_state_receipt)[0]
    evidence.extend(cast(npt.NDArray[np.int64], flags).tobytes(order="C"))
    evidence.extend(receipt.request_sha256.encode("ascii"))
    evidence.extend(receipt.state_sha256.encode("ascii"))
    evidence.extend(receipt.transaction_sha256.encode("ascii"))
    return hashlib.sha256(evidence).hexdigest()


def _decode_full_native_backend_metrics(
    payload: object,
    *,
    batch_size: int,
    exact_seconds: float,
) -> BackendMetrics:
    if not isinstance(payload, tuple) or len(payload) != 3:
        raise RuntimeError("full native ALNS backend payload has an invalid schema")
    counters = _require_array(
        payload[0],
        dtype=np.dtype(np.int64),
        shape=(10,),
        name="full native backend counters",
    )
    occupancies = _require_vector(payload[1], "full native launch occupancies")
    timing = _require_array(
        payload[2],
        dtype=np.dtype(np.float64),
        shape=(1,),
        name="full native backend timing",
    )
    values = tuple(int(value) for value in counters)
    if any(value < 0 for value in values[:9]) or values[9] != batch_size:
        raise RuntimeError("full native ALNS backend counters are invalid")
    if (
        values[0] != values[1]
        or values[1] != values[2] + values[3]
        or values[4] != values[8]
        or values[8] != len(occupancies)
        or any(int(value) <= 0 for value in occupancies)
        or sum(int(value) for value in occupancies) != values[0]
    ):
        raise RuntimeError("full native ALNS backend counters do not reconcile")
    measured_exact_seconds = float(timing[0])
    if (
        not math.isfinite(measured_exact_seconds)
        or measured_exact_seconds < 0.0
        or measured_exact_seconds != exact_seconds
    ):
        raise RuntimeError("full native ALNS backend timing does not reconcile")
    return BackendMetrics(
        backend="cpu_batch",
        batch_size=batch_size,
        total_seconds=measured_exact_seconds,
        label_management_seconds=measured_exact_seconds,
        work_batches=values[4],
        transition_batches=values[5],
        transitions=values[6],
        exact_calls=values[0],
        batch_launches=values[8],
        checkpoint_count=values[7],
        started_calls=values[1],
        completed_calls=values[2],
        interrupted_calls=values[3],
        native_kernel_seconds=measured_exact_seconds,
        native_invocations=values[8],
        launch_occupancies=[int(value) for value in occupancies],
    )


def _verify_full_native_best_projection(
    instance: Instance,
    *,
    initial_customer_sequences: tuple[CustomerSequence, ...],
    semantic_stream: NativeGlobalSemanticStream,
    returned_customer_sequences: tuple[CustomerSequence, ...],
) -> None:
    def replay_objective(
        customer_sequences: tuple[CustomerSequence, ...],
    ) -> SolutionObjective:
        exact_results = tuple(
            solve_exact_charging(instance, sequence) for sequence in customer_sequences
        )
        if any(not result.feasible for result in exact_results):
            raise RuntimeError("full native semantic replay found an infeasible solution")
        report = validate_routes(
            instance,
            [list(result.route) for result in exact_results],
        )
        if not report.feasible:
            raise RuntimeError("full native semantic replay failed the unified validator")
        return SolutionObjective.from_report(instance, report)

    current_sequences = initial_customer_sequences
    current_objective = replay_objective(current_sequences)
    best_sequences = current_sequences
    best_objective = current_objective
    for event in semantic_stream.neighborhood_events:
        candidate_sequences = cast(
            tuple[CustomerSequence, ...],
            event["candidate_route_sequences"],
        )
        candidate_feasible = bool(event["candidate_feasible"])
        accepted = bool(event["accepted"])
        if accepted and (not candidate_feasible or not candidate_sequences):
            raise RuntimeError("full native semantic replay accepted a missing candidate")
        if not candidate_feasible or not candidate_sequences:
            continue
        candidate_objective = replay_objective(candidate_sequences)
        event_objective = cast(tuple[int, float, float, int], event["candidate_objective_key"])
        if candidate_objective.key != event_objective:
            raise RuntimeError("full native semantic candidate objective failed replay")
        if bool(event["vehicle_reduction"]) != (
            candidate_objective.vehicle_count < current_objective.vehicle_count
        ):
            raise RuntimeError("full native semantic vehicle-reduction flag failed replay")
        if bool(event["distance_improvement"]) != _native_distance_improved(
            candidate_objective,
            current_objective,
        ):
            raise RuntimeError("full native semantic distance-improvement flag failed replay")
        if accepted:
            current_sequences = candidate_sequences
            current_objective = candidate_objective
            if current_objective.key < best_objective.key:
                best_sequences = current_sequences
                best_objective = current_objective
    if returned_customer_sequences != best_sequences:
        raise RuntimeError("full native ALNS returned current state instead of global best")


def _native_distance_improved(
    candidate: SolutionObjective,
    current: SolutionObjective,
) -> bool:
    return (
        candidate.vehicle_count == current.vehicle_count
        and candidate.total_distance < current.total_distance - 1e-9
    )


@dataclass(frozen=True, slots=True)
class Stage052NativeExecutionConfig:
    """Fail-fast selection of one experimental Stage 5.2 native architecture."""

    mode: NativeExecutionMode
    native_kernel_config: NativeKernelConfig
    candidate_transaction_config: NativeCandidateTransactionConfig
    candidate_control_config: CandidateControlConfig
    shard_processes: int
    compute_threads_per_shard: int
    schema_version: str = NATIVE_EXECUTION_SCHEMA_VERSION
    enabled: bool = True
    scheduler_threads: int = 24
    control_channel: str = "unix_domain"
    data_plane: str = "shared_memory_soa"
    failure_policy: str = "fail_fast_no_fallback"
    fallback_allowed: bool = False
    scheduler_socket_path: str | None = None

    def __post_init__(self) -> None:
        if not self.enabled:
            raise ValueError("disabled native configuration is ambiguous; pass None instead")
        if self.schema_version != NATIVE_EXECUTION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Stage 5.2 native execution schema {self.schema_version}"
            )
        if self.mode not in _PROTOCOL_BY_MODE:
            raise ValueError(f"unsupported Stage 5.2 native execution mode {self.mode}")
        if self.shard_processes <= 0 or self.compute_threads_per_shard <= 0:
            raise ValueError("native execution topology counts must be positive")
        if self.compute_thread_limit != 24:
            raise ValueError("Stage 5.2 comparisons require exactly 24 compute threads")
        if self.scheduler_threads != 24:
            raise ValueError("the host scheduler requires exactly 24 scheduler threads")
        if self.control_channel != "unix_domain":
            raise ValueError("the host scheduler requires a Unix-domain control channel")
        if self.data_plane != "shared_memory_soa":
            raise ValueError("native execution requires the shared-memory SoA data plane")
        if self.failure_policy != "fail_fast_no_fallback" or self.fallback_allowed:
            raise ValueError("native execution failures must not fall back")
        if self.scheduler_socket_path is not None and not self.scheduler_socket_path:
            raise ValueError("native scheduler socket path must be non-empty when supplied")
        if not self.candidate_control_config.enabled:
            raise ValueError("native execution requires enabled Candidate Control")
        if self.candidate_transaction_config.implementation_mode != "candidate_transaction":
            raise ValueError("native execution requires the complete candidate transaction")

    @property
    def worker_protocol(self) -> NativeWorkerProtocol:
        return _PROTOCOL_BY_MODE[self.mode]

    @property
    def compute_thread_limit(self) -> int:
        return self.shard_processes * self.compute_threads_per_shard

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["worker_protocol"] = self.worker_protocol
        payload["compute_thread_limit"] = self.compute_thread_limit
        return payload


__all__ = (
    "NATIVE_EXECUTION_SCHEMA_VERSION",
    "NativeExecutionMode",
    "NativeInitialStateReceipt",
    "NativeCandidateResolution",
    "NativeCandidateResourceReceipt",
    "NativeCandidateRoundFailure",
    "NativeCandidateRoundRequest",
    "NativeCandidateRoundResult",
    "NativeConstraintSemanticStream",
    "NativeGlobalSemanticStream",
    "NativeThreeLaneSemanticStream",
    "FullNativeALNSResult",
    "FULL_NATIVE_OPERATOR_NAMES",
    "NativeWorkerProtocol",
    "Stage052NativeExecutionConfig",
    "execute_native_candidate_round",
    "decode_native_constraint_semantic_stream",
    "decode_native_global_semantic_stream",
    "decode_native_three_lane_semantic_stream",
    "execute_full_native_alns",
)
