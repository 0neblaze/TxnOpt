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
from dataclasses import asdict, dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from evrptw.cache_incremental import CacheIncrementalConfig
from evrptw.candidate_control import CandidateControlConfig
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
from evrptw.neighborhoods import VehicleOperatorConfig
from evrptw.objective import SolutionObjective
from evrptw.stage04 import Stage04Config
from evrptw.validation import validate_routes

NATIVE_EXECUTION_SCHEMA_VERSION = "stage05.2-native-execution-v2"

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
        )
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
        expected_events = np.asarray(
            [
                    [1, 4, 0, 0, 0, 0, 0, -1, -2, -1, -1],
                [2, 9, 0, 4, 0, 0, 2, -1, -2, 0, 1],
                    [0, 2, 0, 0, 0, 0, 0, -1, -2, -1, -1],
            ],
            dtype=np.int64,
        )
        single_columns = np.asarray(
            [1, 3, 4, 5, 9, 10, 15, 16, 20, 24, 25],
            dtype=np.int64,
        )
        initial_exact = solve_exact_charging(instance, initial_customer_sequences[0])
        if (
            event_count != 3
            or terminal_reason != 0
            or not initial_exact.feasible
            or not np.array_equal(events[:, single_columns], expected_events)
            or np.any(events[:, 2] != expected_start_iteration)
            or removed_offsets.tolist() != [0, 0, 0, 0]
            or len(removed_indices) != 0
            or plan_offsets.tolist() != [0, 0, 0, 0]
            or route_offsets.tolist() != [0]
            or len(route_indices) != 0
            or np.any(objective_present)
            or np.any(ranking != 0.0)
            or stage04_calls.tolist() != [[0, 0, 0, 0]]
            or stage04_status.tolist() != [[-1, -1, -1, -1]]
            or not np.array_equal(stage04_weights, np.ones((1, 4, 2)))
            or not np.array_equal(stage04_rewards, np.zeros((1, 4)))
        ):
            raise RuntimeError("native global no-removable replay is invalid")
        return NativeGlobalSemanticStream(
            neighborhood_events=project_events(),
            event_integer=_readonly_copy(events),
            stage04_calls=_readonly_copy(stage04_calls),
            termination=_readonly_copy(termination),
            transaction_sha256=transaction_sha256,
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
        len(initial_customer_sequences) < 2
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
        if activity[refinement_index, 1] > 0:
            activity[refinement_index, 0] = 1
        return activity

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
                        candidate_route_sequences=quality_routes,
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
        activity = activity_from_events(terminal_events)
        if payload[0] is not None:
            terminal_legacy = require_tuple(payload[0], 8, "terminal legacy")
            terminal_legacy_outcome = cast(
                npt.NDArray[np.int64],
                _require_array(
                    terminal_legacy[6],
                    dtype=np.dtype(np.int64),
                    shape=(5,),
                    name="terminal legacy outcome",
                ),
            )
            if int(terminal_legacy_outcome[0]) >= 0:
                terminal_transaction = require_tuple(
                    terminal_legacy[5], 13, "terminal legacy transaction"
                )
                legacy_index = FULL_NATIVE_OPERATOR_NAMES.index("route_elimination")
                activity[legacy_index] = [
                    1,
                    1,
                    len(_require_vector(terminal_transaction[5], "terminal legacy exact")),
                    1,
                    1,
                    1,
                    int(
                        float(cast(npt.NDArray[np.float64], terminal_transaction[3])[
                            int(terminal_legacy_outcome[0]), 0
                        ])
                        < initial_objective.total_distance - 1e-9
                    ),
                    0,
                ]
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
            refinement_index = FULL_NATIVE_OPERATOR_NAMES.index(
                "vehicle_reduction_refinement"
            )
            activity[refinement_index] = [
                int(refinement_metadata[0]) == 0,
                1,
                int(refinement_metadata[2]),
                0,
                0,
                0,
                0,
                1 if int(refinement_metadata[0]) != 0 else 2,
            ]
        return NativeThreeLaneSemanticStream(
            neighborhood_events=tuple(terminal_events),
            operator_weights=_readonly_copy(weights),
            operator_rewards=_readonly_copy(rewards),
            operator_calls=_readonly_copy(calls),
            operator_totals=_readonly_copy(totals),
            operator_activity=_readonly_copy(activity),
            termination=_readonly_copy(termination),
            transaction_sha256=producer_sha256,
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
        quality_transaction = require_plan_transaction(
            quality[1], len(rejected_quality_plans), "rejected quality"
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
        rejected_events.append(
            event(
                "relocate",
                "candidate_pool_aggregate",
                "relocate_complete_candidate_pool",
                aggregate_count=len(rejected_quality_plans),
                candidate_pool_hash=pool_hash.hexdigest(),
            )
        )
        selected_quality = int(quality_outcome[0])
        if selected_quality < 0:
            if bool(quality_outcome[1]) or bool(quality_outcome[2]):
                raise RuntimeError("native three-lane rejected quality outcome is invalid")
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
            quality_routes = rejected_quality_plans[selected_quality]
            quality_objective = replay_objective(quality_routes)
            quality_key = reported_objective(
                quality_transaction,
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
                    candidate_route_sequences=quality_routes,
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
        removal_scores = cast(
            npt.NDArray[np.float64],
            _require_array(
                removal[4],
                dtype=np.dtype(np.float64),
                shape=(len(removed_indices) * 2 + 2,),
                name="rejected constraint removal scores",
            ),
        )
        removal_route_indices = _require_vector(
            removal[5], "rejected constraint route indices"
        )
        affected = tuple(
            index
            for index, (before, after) in enumerate(
                zip(initial_customer_sequences, repaired_routes, strict=False)
            )
            if before != after
        )
        rejected_events.extend(
            (
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
                    candidate_objective_key=constraint_key,
                    track="constraint_lane",
                    constraint_category=operator,
                    removal_tier="small" if int(selection[0]) == 0 else "",
                    removal_size_requested=int(selection[1]),
                    removal_size_actual=int(selection[2]),
                    removal_trigger="stagnation_baseline",
                ),
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
                    exact_route_evaluations=len(
                        _require_vector(
                            constraint_transaction[5],
                            "rejected constraint exact rows",
                        )
                    ),
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
                ),
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
    refinement_selected = bool(refinement_metadata[1])
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
    if selected_quality < 0 or selected_quality >= len(quality_plans):
        raise RuntimeError("native three-lane quality selection is invalid")
    quality_routes = quality_plans[selected_quality]
    quality_objective = replay_objective(quality_routes)
    quality_objective_key = reported_objective(
        quality_transaction, selected_quality, quality_objective, "quality"
    )
    quality_source = quality_sources[selected_quality]
    quality_changed = tuple(int(value) for value in changed_routes[quality_source])
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
    constraint_objective = replay_objective(repaired_routes)
    constraint_objective_key = reported_objective(
        constraint_transaction, 0, constraint_objective, "constraint"
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
    removal_scores = cast(
        npt.NDArray[np.float64],
        _require_array(
            removal[4],
            dtype=np.dtype(np.float64),
            shape=(len(removed_indices) * 2 + 2,),
            name="constraint removal scores",
        ),
    )
    removal_route_indices = _require_vector(removal[5], "constraint route indices")
    if len(removal_route_indices) == 0:
        raise RuntimeError("native three-lane constraint route identity is missing")
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
            candidate_objective_key=quality_objective_key,
        ),
        event(
            "relocate",
            "candidate_proposed",
            "relocate_candidate",
            route_indices=quality_changed,
            affected_route_indices=quality_changed,
            removed_customers=(name_by_index[moved_customer_index],),
            candidate_route_sequences=quality_routes,
            candidate_vehicle_delta=len(quality_routes) - len(initial_customer_sequences),
            candidate_feasible=True,
            prefilter_passed=True,
            accepted=quality_accepted,
            vehicle_reduction=bool(quality_outcome[3]),
            distance_improvement=quality_objective.total_distance
            < initial_objective.total_distance - 1e-9,
            candidate_objective_key=quality_objective_key,
        ),
        event(
            FULL_NATIVE_OPERATOR_NAMES[int(constraint_outcome[0]) + 9],
            "candidate_proposed",
            "constraint_ranked_removal",
            route_indices=(int(removal_route_indices[0]),),
            affected_route_indices=(int(removal_route_indices[0]),),
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
            "candidate_proposed",
            "constraint_removal_repaired",
            affected_route_indices=constraint_affected,
            removed_customers=removed_customers,
            candidate_route_sequences=repaired_routes,
            candidate_vehicle_delta=len(repaired_routes)
            - len(initial_customer_sequences),
            candidate_feasible=True,
            prefilter_passed=True,
            exact_route_evaluations=len(
                _require_vector(constraint_transaction[5], "constraint exact rows")
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
            distance_improvement=constraint_objective.total_distance
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
            candidate_objective_key=legacy_objective_key,
        ),
        event(
            "vehicle_reduction_refinement",
            "candidate_proposed" if refinement_selected else "failed",
            "vehicle_reduction_refined" if refinement_selected else "refinement_not_better",
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
            distance_improvement=refinement_selected
            and legacy_objective.total_distance < initial_objective.total_distance - 1e-9,
            candidate_objective_key=legacy_objective_key,
        ),
    ]
    quality_expected_accept = quality_objective.key <= initial_objective.key
    quality_expected_best = quality_expected_accept and (
        quality_objective.key < initial_objective.key
    )
    global_after_quality = (
        quality_objective if quality_expected_best else initial_objective
    )
    constraint_expected_accept = constraint_objective.key <= initial_objective.key
    constraint_expected_best = constraint_expected_accept and (
        constraint_objective.key < global_after_quality.key
    )
    global_after_constraint = (
        constraint_objective if constraint_expected_best else global_after_quality
    )
    legacy_expected_accept = legacy_objective.key <= initial_objective.key
    legacy_expected_best = legacy_expected_accept and (
        legacy_objective.key < global_after_constraint.key
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
        legacy_routes
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


def decode_native_three_lane_search_semantic_stream(
    instance: Instance,
    payload: object,
    *,
    node_names: tuple[str, ...],
    initial_customer_sequences: tuple[CustomerSequence, ...],
) -> NativeThreeLaneSemanticStream:
    """Replay the two-iteration v2 search envelope event by event."""

    if not isinstance(payload, tuple):
        raise RuntimeError("native three-lane search payload has an invalid type")
    producer_sha256 = _verify_native_three_lane_search_hash(payload)
    iteration_payloads = cast(tuple[object, ...], payload[0])
    if len(iteration_payloads) not in {
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
    }:
        raise RuntimeError("native three-lane search iteration count is invalid")
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
    if selected_quality < 0 or selected_quality >= len(quality_plans):
        raise RuntimeError("native three-lane follow-up quality selection is invalid")
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
    constraint_objective = replay_objective(constraint_routes)
    constraint_key = objective_key(
        constraint_transaction, 0, constraint_objective, "constraint"
    )
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
    removal_scores = cast(
        npt.NDArray[np.float64],
        _require_array(
            removal[4],
            dtype=np.dtype(np.float64),
            shape=(len(removed_indices) * 2 + 2,),
            name="follow-up constraint scores",
        ),
    )
    removal_route_indices = _require_vector(
        removal[5], "follow-up constraint route indices"
    )
    if len(removal_route_indices) == 0:
        raise RuntimeError("native three-lane follow-up constraint route is missing")

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
        legacy_metadata[:6].tolist() != [1, 0, 1, 0, 0, 0]
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
    expected_quality_accepted = quality_objective.key <= prior_quality_objective.key
    expected_constraint_accepted = (
        constraint_objective.key <= prior_constraint_objective.key
    )
    expected_legacy_accepted = legacy_objective.key <= prior_legacy_objective.key
    if (
        quality_accepted != expected_quality_accepted
        or bool(quality_outcome[2])
        or constraint_accepted != expected_constraint_accepted
        or bool(constraint_outcome[4])
        or legacy_accepted != expected_legacy_accepted
        or bool(legacy_acceptance[1])
        or bool(legacy_acceptance[2])
    ):
        raise RuntimeError("native three-lane follow-up acceptance replay mismatch")

    expected_legacy = legacy_routes if legacy_accepted else prior_legacy
    expected_quality = quality_routes if quality_accepted else prior_quality
    expected_constraint = constraint_routes if constraint_accepted else prior_constraint
    expected_best = prior_best
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
    if replay_objective(final_states[3]).key != prior_best_objective.key:
        raise RuntimeError("native three-lane follow-up changed the global best")

    constraint_affected = tuple(
        index
        for index, (before, after) in enumerate(
            zip(prior_constraint, constraint_routes, strict=False)
        )
        if before != after
    )
    quality_changed = tuple(int(value) for value in changed[quality_source])
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
            "candidate_proposed",
            "swap_candidate",
            route_indices=quality_changed,
            affected_route_indices=quality_changed,
            removed_customers=quality_moved,
            candidate_route_sequences=quality_routes,
            candidate_vehicle_delta=len(quality_routes) - len(prior_quality),
            candidate_feasible=True,
            prefilter_passed=True,
            accepted=quality_accepted,
            distance_improvement=quality_objective.total_distance
            < prior_quality_objective.total_distance - 1e-9,
            candidate_objective_key=quality_key,
        ),
        event(
            "time_window_conflict",
            "candidate_proposed",
            "constraint_ranked_removal",
            route_indices=(int(removal_route_indices[0]),),
            affected_route_indices=(int(removal_route_indices[0]),),
            removed_customers=constraint_removed,
            candidate_route_sequences=constraint_partial,
            prefilter_passed=True,
            selection_rank=1,
            track="constraint_lane",
            constraint_category="time_window_conflict",
            removal_tier="small",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            removal_trigger="stagnation_baseline",
            reset_observed=bool(selection[6]),
            ranking_score=float(removal_scores[0]),
            candidate_objective_key=constraint_key,
        ),
        event(
            "time_window_conflict",
            "candidate_proposed",
            "constraint_removal_repaired",
            affected_route_indices=constraint_affected,
            removed_customers=constraint_removed,
            candidate_route_sequences=constraint_routes,
            candidate_vehicle_delta=len(constraint_routes) - len(prior_constraint),
            candidate_feasible=True,
            prefilter_passed=True,
            exact_route_evaluations=len(
                _require_vector(constraint_transaction[5], "constraint exact rows")
            ),
            track="constraint_lane",
            constraint_category="time_window_conflict",
            removal_tier="small",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            removal_trigger="stagnation_baseline",
            reset_observed=bool(selection[6]),
            accepted=constraint_accepted,
            distance_improvement=constraint_objective.total_distance
            < prior_constraint_objective.total_distance - 1e-9,
            candidate_objective_key=constraint_key,
        ),
        event(
            "vehicle_count_aware_repair",
            "candidate_proposed",
            "existing_route_repair",
            removed_customers=legacy_removed,
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
        termination.tolist()[:2] != [0, -1]
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
        activity[index, 1] += int(bool(item["prefilter_passed"]))
        activity[index, 2] += cast(int, item["exact_route_evaluations"])
        activity[index, 3] += int(item["status"] == "candidate_proposed")
        activity[index, 4] += int(bool(item["candidate_feasible"]))
        activity[index, 5] += int(bool(item["vehicle_reduction"]))
        activity[index, 6] += int(bool(item["distance_improvement"]))
    refinement_index = FULL_NATIVE_OPERATOR_NAMES.index(
        "vehicle_reduction_refinement"
    )
    if activity[refinement_index, 1] > 0:
        activity[refinement_index, 0] = 1
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
    sixth_stream = _decode_native_three_lane_sixth_iteration(
        instance,
        cast(tuple[object, ...], iteration_payloads[5]),
        previous_payload=cast(tuple[object, ...], iteration_payloads[4]),
        previous_stream=fifth_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
        iteration=5,
        expected_metadata=(5, 1, 1, 0, 0, 0, 0, -2),
        completed_iterations=6,
    )
    if len(iteration_payloads) == 6:
        return sixth_stream
    seventh_stream = _decode_native_three_lane_seventh_iteration(
        instance,
        cast(tuple[object, ...], iteration_payloads[6]),
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
    )
    if len(iteration_payloads) == 7:
        return seventh_stream
    eighth_stream = _decode_native_three_lane_rejection_only_iteration(
        cast(tuple[object, ...], iteration_payloads[7]),
        previous_payload=cast(tuple[object, ...], iteration_payloads[6]),
        previous_stream=seventh_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
        iteration=7,
        operator="route_elimination",
        operator_id=2,
        completed_iterations=8,
    )
    if len(iteration_payloads) == 8:
        return eighth_stream
    ninth_stream = _decode_native_three_lane_rejection_only_iteration(
        cast(tuple[object, ...], iteration_payloads[8]),
        previous_payload=cast(tuple[object, ...], iteration_payloads[7]),
        previous_stream=eighth_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
        iteration=8,
        operator="route_merge",
        operator_id=3,
        completed_iterations=9,
    )
    if len(iteration_payloads) == 9:
        return ninth_stream
    tenth_stream = _decode_native_three_lane_constraint_no_change_iteration(
        instance,
        cast(tuple[object, ...], iteration_payloads[9]),
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
    eleventh_stream = _decode_native_three_lane_standard_energy_rejection_iteration(
        instance,
        cast(tuple[object, ...], iteration_payloads[10]),
        previous_payload=cast(tuple[object, ...], iteration_payloads[9]),
        previous_stream=tenth_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
        iteration=10,
        destroy_name="random",
        repair_name="energy",
        expected_metadata=(10, 0, 0, 2, 1, 0, 0, 1),
        expected_statuses=(5, 1, 1, 1, 0),
        expected_exact_rows=(0,),
        expected_exact_call_delta=2,
        candidate_prepared=True,
        feasible_repair=True,
        completed_iterations=11,
    )
    if len(iteration_payloads) == 11:
        return eleventh_stream
    twelfth_stream = _decode_native_three_lane_rejection_only_iteration(
        cast(tuple[object, ...], iteration_payloads[11]),
        previous_payload=cast(tuple[object, ...], iteration_payloads[10]),
        previous_stream=eleventh_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
        iteration=11,
        operator="route_elimination",
        operator_id=2,
        completed_iterations=12,
    )
    if len(iteration_payloads) == 12:
        return twelfth_stream
    thirteenth_stream = _decode_native_three_lane_constraint_no_change_iteration(
        instance,
        cast(tuple[object, ...], iteration_payloads[12]),
        previous_payload=cast(tuple[object, ...], iteration_payloads[11]),
        previous_stream=twelfth_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
        iteration=12,
        constraint_operator="station_pressure",
        constraint_operator_id=0,
        expected_selection=(2, 1, 1, 1, 11, 2, 0),
        removal_tier="large",
        removal_trigger="large_stagnation",
        legacy_operator="route_elimination",
        legacy_operator_id=2,
        completed_iterations=13,
    )
    if len(iteration_payloads) == 13:
        return thirteenth_stream
    fourteenth_stream = _decode_native_three_lane_standard_energy_rejection_iteration(
        instance,
        cast(tuple[object, ...], iteration_payloads[13]),
        previous_payload=cast(tuple[object, ...], iteration_payloads[12]),
        previous_stream=thirteenth_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
        iteration=13,
        destroy_name="random",
        repair_name="greedy",
        expected_metadata=(13, 0, 0, 0, 1, 0, -1, 0),
        expected_statuses=(1, 1, 1, 1, 0),
        expected_exact_rows=(),
        expected_exact_call_delta=0,
        candidate_prepared=False,
        feasible_repair=False,
        completed_iterations=14,
    )
    if len(iteration_payloads) == 14:
        return fourteenth_stream
    fifteenth_stream = _decode_native_three_lane_rejection_only_iteration(
        cast(tuple[object, ...], iteration_payloads[14]),
        previous_payload=cast(tuple[object, ...], iteration_payloads[13]),
        previous_stream=fourteenth_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
        iteration=14,
        operator="route_merge",
        operator_id=3,
        completed_iterations=15,
    )
    if len(iteration_payloads) == 15:
        return fifteenth_stream
    sixteenth_stream = _decode_native_three_lane_seventh_iteration(
        instance,
        cast(tuple[object, ...], iteration_payloads[15]),
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
    )
    if len(iteration_payloads) == 16:
        return sixteenth_stream
    seventeenth_stream = _decode_native_three_lane_rejection_only_iteration(
        cast(tuple[object, ...], iteration_payloads[16]),
        previous_payload=cast(tuple[object, ...], iteration_payloads[15]),
        previous_stream=sixteenth_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
        iteration=16,
        operator="route_elimination",
        operator_id=2,
        completed_iterations=17,
    )
    if len(iteration_payloads) == 17:
        return seventeenth_stream
    eighteenth_stream = _decode_native_three_lane_rejection_only_iteration(
        cast(tuple[object, ...], iteration_payloads[17]),
        previous_payload=cast(tuple[object, ...], iteration_payloads[16]),
        previous_stream=seventeenth_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
        iteration=17,
        operator="route_merge",
        operator_id=3,
        completed_iterations=18,
    )
    if len(iteration_payloads) == 18:
        return eighteenth_stream
    nineteenth_stream = _decode_native_three_lane_seventh_iteration(
        instance,
        cast(tuple[object, ...], iteration_payloads[18]),
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
    )
    if len(iteration_payloads) == 19:
        return nineteenth_stream
    twentieth_stream = _decode_native_three_lane_sixth_iteration(
        instance,
        cast(tuple[object, ...], iteration_payloads[19]),
        previous_payload=cast(tuple[object, ...], iteration_payloads[18]),
        previous_stream=nineteenth_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
        iteration=19,
        expected_metadata=(19, 2, 1, 0, 0, 0, 0, -2),
        completed_iterations=20,
    )
    if len(iteration_payloads) == 20:
        return twentieth_stream
    return _decode_native_three_lane_rejection_only_iteration(
        cast(tuple[object, ...], iteration_payloads[20]),
        previous_payload=cast(tuple[object, ...], iteration_payloads[19]),
        previous_stream=twentieth_stream,
        node_names=node_names,
        search_sha256=producer_sha256,
        iteration=20,
        operator="route_merge",
        operator_id=3,
        completed_iterations=21,
        restart_to_best=True,
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
    changed = cast(npt.NDArray[np.int64], quality_pool[0])
    if (
        changed.shape != (0, 2)
        or _require_vector(quality_pool[1], "third quality offsets").tolist() != [0]
        or len(_require_vector(quality_pool[2], "third quality indices")) != 0
        or quality[1] is not None
        or cast(npt.NDArray[np.int64], quality[2]).tolist() != [-1, 0, 0, 0]
    ):
        raise RuntimeError("native three-lane third two-opt-star decision is invalid")

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
            aggregate_count=0,
            candidate_pool_hash=hashlib.sha256().hexdigest(),
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
        activity[index, 1] += int(bool(item["prefilter_passed"]))
        activity[index, 2] += cast(int, item["exact_route_evaluations"])
        activity[index, 3] += int(item["status"] == "candidate_proposed")
        activity[index, 4] += int(bool(item["candidate_feasible"]))
        activity[index, 5] += int(bool(item["vehicle_reduction"]))
        activity[index, 6] += int(bool(item["distance_improvement"]))
    refinement_index = FULL_NATIVE_OPERATOR_NAMES.index(
        "vehicle_reduction_refinement"
    )
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
    legacy_transaction = transaction(legacy[5], 1, "legacy")
    insertion_transaction = require_tuple(legacy[6], 13, "insertion transaction")
    insertion_count = len(
        _require_vector(insertion_transaction[1], "fourth insertion statuses")
    )
    transaction(insertion_transaction, insertion_count, "insertion")
    if len(_require_vector(insertion_transaction[5], "fourth insertion exact rows")) != 2:
        raise RuntimeError("native three-lane fourth insertion charging is invalid")
    legacy_objective = replay(legacy_routes)
    legacy_key = reported_objective(
        legacy_transaction, 0, legacy_objective, "legacy"
    )
    legacy_acceptance = require_tuple(payload[4], 3, "legacy acceptance")
    if (
        legacy_acceptance != (1, 0, 0)
        or legacy_objective.key != prior_legacy_objective.key
    ):
        raise RuntimeError("native three-lane fourth legacy acceptance mismatch")

    quality = require_tuple(payload[2], 7, "quality")
    attempts = cast(
        npt.NDArray[np.int64],
        _require_array(
            quality[0],
            dtype=np.dtype(np.int64),
            shape=(2, 8),
            name="fourth quality attempts",
        ),
    )
    if attempts.tolist() != [
        [1, 0, 2, 0, 0, 1, 0, 0],
        [1, 1, 2, 0, 0, 2, 1, 0],
    ]:
        raise RuntimeError("native three-lane fourth route-segment order is invalid")
    removed_offsets = _require_vector(quality[1], "fourth quality removed offsets")
    removed_indices = _require_vector(quality[2], "fourth quality removed indices")
    if removed_offsets.tolist() != [0, 2, 4]:
        raise RuntimeError("native three-lane fourth route-segment offsets are invalid")
    quality_removed = tuple(
        tuple(
            node_names[int(index)]
            for index in removed_indices[
                int(removed_offsets[row]) : int(removed_offsets[row + 1])
            ]
        )
        for row in range(2)
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
    quality_outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            quality[5],
            dtype=np.dtype(np.int64),
            shape=(4,),
            name="fourth quality outcome",
        ),
    )
    if quality_outcome.tolist() != [0, 1, 0, 0]:
        raise RuntimeError("native three-lane fourth quality acceptance mismatch")

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
    if int(constraint_outcome[0]) != 3 or not bool(constraint_outcome[3]):
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
    constraint_key = reported_objective(
        constraint_transaction, 0, constraint_objective, "constraint"
    )
    constraint_scores = cast(
        npt.NDArray[np.float64],
        _require_array(
            removal[4],
            dtype=np.dtype(np.float64),
            shape=(len(constraint_removed_indices) * 2 + 2,),
            name="fourth constraint scores",
        ),
    )
    constraint_route_indices = _require_vector(
        removal[5], "fourth constraint route indices"
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

    final_states = (
        unpack_state(payload[6], "final legacy"),
        unpack_state(payload[7], "final quality"),
        unpack_state(payload[8], "final constraint"),
        unpack_state(payload[9], "final best"),
    )
    expected_states = (legacy_routes, quality_routes, constraint_routes, prior_best)
    if (
        final_states != expected_states
        or replay(final_states[3]).key != prior_best_objective.key
    ):
        raise RuntimeError("native three-lane fourth lane state mismatch")

    followup_events = (
        event(
            "route_segment_destroy",
            "failed",
            "route_segment_no_change",
            route_indices=(int(attempts[0, 0]),),
            removed_customers=quality_removed[0],
            candidate_vehicle_delta=0,
            prefilter_passed=True,
            selection_rank=int(attempts[0, 5]),
            segment_length=int(attempts[0, 2]),
            candidate_objective_key=prior_quality_objective.key,
        ),
        event(
            "route_segment_destroy",
            "candidate_proposed",
            "route_segment_repaired",
            route_indices=(int(attempts[1, 0]),),
            affected_route_indices=(int(attempts[1, 0]),),
            removed_customers=quality_removed[1],
            candidate_route_sequences=quality_routes,
            candidate_vehicle_delta=len(quality_routes) - len(prior_quality),
            candidate_feasible=True,
            prefilter_passed=True,
            selection_rank=int(attempts[1, 5]),
            segment_length=int(attempts[1, 2]),
            accepted=True,
            candidate_objective_key=quality_key,
        ),
        event(
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
            candidate_objective_key=constraint_key,
        ),
        event(
            "shaw_related",
            "candidate_proposed",
            "constraint_removal_repaired",
            affected_route_indices=constraint_affected,
            removed_customers=constraint_removed,
            candidate_route_sequences=constraint_routes,
            candidate_vehicle_delta=len(constraint_routes) - len(prior_constraint),
            candidate_feasible=True,
            prefilter_passed=True,
            exact_route_evaluations=len(
                _require_vector(
                    constraint_transaction[5], "fourth constraint exact rows"
                )
            ),
            track="constraint_lane",
            constraint_category="shaw_related",
            removal_tier="small",
            removal_size_requested=int(selection[1]),
            removal_size_actual=int(selection[2]),
            stagnation_iterations=int(selection[4]),
            removal_trigger="stagnation_baseline",
            reset_observed=bool(selection[6]),
            accepted=True,
            vehicle_reduction=constraint_objective.vehicle_count
            < prior_constraint_objective.vehicle_count,
            distance_improvement=constraint_objective.total_distance
            < prior_constraint_objective.total_distance - 1e-9,
            candidate_objective_key=constraint_key,
        ),
        event(
            "standard",
            "proposal",
            "random+regret2",
            removed_customers=legacy_removed,
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
        termination.tolist() != [0, -1, 40, 40, 0, 4]
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
        activity[index, 1] += int(bool(item["prefilter_passed"]))
        activity[index, 2] += cast(int, item["exact_route_evaluations"])
        activity[index, 3] += int(item["status"] == "candidate_proposed")
        activity[index, 4] += int(bool(item["candidate_feasible"]))
        activity[index, 5] += int(bool(item["vehicle_reduction"]))
        activity[index, 6] += int(bool(item["distance_improvement"]))
    activity[by_name["standard"], 0] = 1
    refinement_index = FULL_NATIVE_OPERATOR_NAMES.index(
        "vehicle_reduction_refinement"
    )
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
    unpack_state(previous_payload[7], "prior quality")
    prior_constraint = unpack_state(previous_payload[8], "prior constraint")
    prior_best = unpack_state(previous_payload[9], "prior best")
    prior_best_objective = replay(prior_best)

    legacy = require_tuple(payload[0], 8, "legacy")
    legacy_metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[0],
            dtype=np.dtype(np.int64),
            shape=(8,),
            name="fifth legacy metadata",
        ),
    )
    if legacy_metadata.tolist() != [4, 0, 2, 2, 1, 0, -1, 0]:
        raise RuntimeError("native three-lane fifth legacy selection is invalid")
    legacy_removed_indices = _require_vector(legacy[1], "fifth legacy removed")
    legacy_removed = tuple(node_names[int(index)] for index in legacy_removed_indices)
    if legacy_removed != ("C4",) or legacy[5] is not None or payload[4] is not None:
        raise RuntimeError("native three-lane fifth legacy rejection is invalid")
    insertion = require_tuple(legacy[6], 13, "legacy insertion transaction")
    insertion_statuses = _require_vector(insertion[1], "fifth insertion statuses")
    if insertion_statuses.tolist() != [1, 1, 1, 1, 0]:
        raise RuntimeError("native three-lane fifth attempted-plan replay is invalid")
    legacy_state = unpack_state(legacy[7], "legacy embedded state")
    if legacy_state != prior_legacy:
        raise RuntimeError("native three-lane fifth changed rejected legacy state")

    quality = require_tuple(payload[2], 4, "quality")
    pool = require_tuple(quality[0], 5, "quality pool")
    plan_offsets = _require_vector(pool[0], "fifth quality plan offsets")
    route_offsets = _require_vector(pool[1], "fifth quality route offsets")
    route_indices = _require_vector(pool[2], "fifth quality route indices")
    change_offsets = _require_vector(pool[3], "fifth quality change offsets")
    change_indices = _require_vector(pool[4], "fifth quality change indices")
    candidate_count = len(plan_offsets) - 1
    if (
        candidate_count != 10
        or int(plan_offsets[0]) != 0
        or int(plan_offsets[-1]) != len(route_offsets) - 1
        or int(route_offsets[0]) != 0
        or int(route_offsets[-1]) != len(route_indices)
        or len(change_offsets) != candidate_count + 1
        or int(change_offsets[0]) != 0
        or int(change_offsets[-1]) != len(change_indices)
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
    if (
        statuses.tolist() != [5] * candidate_count
        or exact_deltas.tolist() != [1, 0, 0, 0, 1, 0, 0, 1, 1, 0]
        or np.any(objective_integers[:, 0] != 2)
        or np.any(objective_integers[:, 1] != 0)
        or np.any(~np.isfinite(objective_floats))
    ):
        raise RuntimeError("native three-lane fifth quality journal is invalid")
    outcome = cast(
        npt.NDArray[np.int64],
        _require_array(
            quality[2],
            dtype=np.dtype(np.int64),
            shape=(4,),
            name="fifth quality outcome",
        ),
    )
    if outcome.tolist() != [2, 1, 0, 0]:
        raise RuntimeError("native three-lane fifth quality selection is invalid")
    selected_plan = candidate_plans[int(outcome[0])]
    selected_objective = replay(selected_plan)
    selected_key = selected_objective.key
    if selected_key != (2, 10.0, 0.0, 0):
        raise RuntimeError("native three-lane fifth quality objective is invalid")
    if payload[3] is not None:
        raise RuntimeError("native three-lane fifth ran an unscheduled constraint lane")

    final_states = (
        unpack_state(payload[6], "final legacy"),
        unpack_state(payload[7], "final quality"),
        unpack_state(payload[8], "final constraint"),
        unpack_state(payload[9], "final best"),
    )
    if final_states != (prior_legacy, selected_plan, prior_constraint, prior_best):
        raise RuntimeError("native three-lane fifth lane state mismatch")
    if replay(final_states[3]).key != prior_best_objective.key:
        raise RuntimeError("native three-lane fifth changed the global best")

    candidate_events = []
    for candidate, plan in enumerate(candidate_plans):
        changed, depth = candidate_changes[candidate]
        changed_sequences = tuple(plan[index] for index in changed)
        candidate_events.append(
            event(
                "ejection_chain",
                "feasible_candidate",
                "exact_charging_feasible",
                route_indices=changed,
                affected_route_indices=changed,
                candidate_customer_sequence=(
                    changed_sequences[0] if len(changed_sequences) == 1 else ()
                ),
                candidate_route_sequences=changed_sequences,
                candidate_feasible=True,
                prefilter_passed=True,
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
            accepted=True,
            candidate_objective_key=selected_key,
        ),
        event(
            "standard",
            "proposal",
            "related+energy",
            removed_customers=legacy_removed,
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
        termination.tolist() != [0, -1, 44, 44, 0, 5]
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
        activity[index, 1] += int(bool(item["prefilter_passed"]))
        activity[index, 2] += cast(int, item["exact_route_evaluations"])
        activity[index, 3] += int(item["status"] == "candidate_proposed")
        activity[index, 4] += int(bool(item["candidate_feasible"]))
        activity[index, 5] += int(bool(item["vehicle_reduction"]))
        activity[index, 6] += int(bool(item["distance_improvement"]))
    activity[by_name["standard"], 0] = previous_stream.operator_activity[
        by_name["standard"], 0
    ]
    refinement_index = by_name["vehicle_reduction_refinement"]
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
    expected_metadata: tuple[int, ...],
    completed_iterations: int,
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
    legacy = require_tuple(payload[0], 7, "legacy")
    metadata = cast(
        npt.NDArray[np.int64],
        _require_array(
            legacy[0],
            dtype=np.dtype(np.int64),
            shape=(8,),
            name="sixth legacy metadata",
        ),
    )
    if metadata.tolist() != list(expected_metadata):
        raise RuntimeError("native three-lane sixth legacy decision is invalid")
    removed_indices = _require_vector(legacy[1], "sixth legacy removed")
    removed = tuple(node_names[int(index)] for index in removed_indices)
    repair = require_tuple(legacy[4], 3, "legacy repair")
    repaired_state = (repair[0], repair[1], None, None)
    repaired_routes = unpack(repaired_state, "legacy repaired")
    transaction = require_tuple(legacy[5], 13, "legacy transaction")
    statuses = _require_vector(transaction[1], "sixth legacy statuses")
    exact_rows = _require_vector(transaction[5], "sixth legacy exact rows")
    if statuses.tolist() != [5] or len(exact_rows) != 0:
        raise RuntimeError("native three-lane sixth transaction is invalid")
    repaired_objective = replay(repaired_routes)
    integers = cast(npt.NDArray[np.int64], transaction[2])
    floats = cast(npt.NDArray[np.float64], transaction[3])
    reported = SolutionObjective(
        int(integers[0, 0]),
        float(floats[0, 0]),
        float(floats[0, 1]),
        int(integers[0, 1]),
    )
    if reported.key != repaired_objective.key:
        raise RuntimeError("native three-lane sixth objective mismatch")
    if payload[2] is not None or payload[3] is not None or payload[4] != (1, 0, 0):
        raise RuntimeError("native three-lane sixth lane schedule is invalid")
    final_states = (
        unpack(payload[6], "final legacy"),
        unpack(payload[7], "final quality"),
        unpack(payload[8], "final constraint"),
        unpack(payload[9], "final best"),
    )
    if final_states != (repaired_routes, *prior_states[1:]):
        raise RuntimeError("native three-lane sixth lane state mismatch")

    item: dict[str, object] = {
        "operator": "vehicle_count_aware_repair",
        "status": "candidate_proposed",
        "reason": "existing_route_repair",
        "route_indices": (),
        "affected_route_indices": (),
        "removed_customers": removed,
        "candidate_customer_sequence": (),
        "candidate_route_sequences": (),
        "candidate_vehicle_delta": len(repaired_routes) - len(prior_states[0]),
        "candidate_feasible": True,
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
        "accepted": True,
        "vehicle_reduction": False,
        "distance_improvement": False,
        "candidate_objective_key": reported.key,
    }
    all_events = (*previous_stream.neighborhood_events, item)
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
        termination.tolist()
        != [
            0,
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
        raise RuntimeError("native three-lane sixth final state is invalid")
    activity = np.zeros((operator_count, 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    for event_item in all_events:
        index = by_name[str(event_item["operator"])]
        activity[index, 0] = max(
            int(activity[index, 0]), int(bool(event_item["candidate_feasible"]))
        )
        activity[index, 1] += int(bool(event_item["prefilter_passed"]))
        activity[index, 2] += cast(int, event_item["exact_route_evaluations"])
        activity[index, 3] += int(event_item["status"] == "candidate_proposed")
        activity[index, 4] += int(bool(event_item["candidate_feasible"]))
        activity[index, 5] += int(bool(event_item["vehicle_reduction"]))
        activity[index, 6] += int(bool(event_item["distance_improvement"]))
    activity[:, 0] = previous_stream.operator_activity[:, 0]
    repair_index = by_name["vehicle_count_aware_repair"]
    activity[repair_index, 0] = (
        previous_stream.operator_activity[repair_index, 0] + 1
    )
    refinement_index = by_name["vehicle_reduction_refinement"]
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
) -> NativeThreeLaneSemanticStream:
    """Replay periodic constraint exploration and route-merge rejection."""

    _verify_native_three_lane_semantic_hash(payload)
    if len(payload) != 14:
        raise RuntimeError("native three-lane seventh iteration is invalid")

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
        if (
            legacy[5] is not None
            or _require_vector(legacy_transaction[1], "legacy statuses").tolist()
            != list(expected_legacy_statuses)
            or _require_vector(legacy_transaction[5], "legacy exact rows").tolist()
            != list(expected_legacy_exact_rows)
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
        or outcome[2:].tolist() != [1, 1, 0, 0]
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
    statuses = _require_vector(
        constraint_transaction[1], "seventh constraint statuses"
    )
    exact_rows = _require_vector(
        constraint_transaction[5], "seventh constraint exact rows"
    )
    if statuses.tolist() != [5] or len(exact_rows) != 0:
        raise RuntimeError("native three-lane seventh cache transaction is invalid")
    scores = cast(
        npt.NDArray[np.float64],
        _require_array(
            removal[4],
            dtype=np.dtype(np.float64),
            shape=(len(removed_indices) * 2 + 2,),
            name="seventh constraint scores",
        ),
    )
    route_indices = _require_vector(
        removal[5], "seventh constraint route indices"
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
        prior_states[3],
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
        raise RuntimeError("native three-lane seventh final state is invalid")

    activity = np.zeros((operator_count, 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    for event_item in all_events:
        index = by_name[str(event_item["operator"])]
        activity[index, 0] = max(
            int(activity[index, 0]), int(bool(event_item["candidate_feasible"]))
        )
        activity[index, 1] += int(bool(event_item["prefilter_passed"]))
        activity[index, 2] += cast(int, event_item["exact_route_evaluations"])
        activity[index, 3] += int(event_item["status"] == "candidate_proposed")
        activity[index, 4] += int(bool(event_item["candidate_feasible"]))
        activity[index, 5] += int(bool(event_item["vehicle_reduction"]))
        activity[index, 6] += int(bool(event_item["distance_improvement"]))
    activity[:, 0] = previous_stream.operator_activity[:, 0]
    constraint_index = by_name[constraint_operator]
    activity[constraint_index, 0] = (
        previous_stream.operator_activity[constraint_index, 0] + 1
    )
    refinement_index = by_name["vehicle_reduction_refinement"]
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
    all_events = (*previous_stream.neighborhood_events, event)
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
            0,
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
        activity[index, 1] += int(bool(event_item["prefilter_passed"]))
        activity[index, 2] += cast(int, event_item["exact_route_evaluations"])
        activity[index, 3] += int(event_item["status"] == "candidate_proposed")
        activity[index, 4] += int(bool(event_item["candidate_feasible"]))
        activity[index, 5] += int(bool(event_item["vehicle_reduction"]))
        activity[index, 6] += int(bool(event_item["distance_improvement"]))
    activity[:, 0] = previous_stream.operator_activity[:, 0]
    refinement_index = by_name["vehicle_reduction_refinement"]
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
    transaction = require_tuple(probe[2], 13, "constraint transaction")
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
    repaired_routes = unpack_soa(repair[0], repair[1], "constraint repaired")
    if repaired_routes != prior_states[2]:
        raise RuntimeError("native constraint no-change repair changed its lane")
    statuses = _require_vector(transaction[1], "no-change statuses")
    exact_rows = _require_vector(transaction[5], "no-change exact rows")
    if statuses.tolist() != [5] or len(exact_rows) != 0:
        raise RuntimeError("native constraint no-change cache journal is invalid")
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
    scores = cast(
        npt.NDArray[np.float64],
        _require_array(
            removal[4],
            dtype=np.dtype(np.float64),
            shape=(len(removed_indices) * 2 + 2,),
            name="no-change scores",
        ),
    )
    removal_routes = _require_vector(removal[5], "no-change route indices")
    if len(removal_routes) == 0:
        raise RuntimeError("native constraint no-change route is missing")

    if payload[1] is not None or payload[2] is not None or payload[4] is not None:
        raise RuntimeError("native constraint no-change lane schedule is invalid")
    final_states = (
        unpack_state(payload[6], "final legacy"),
        unpack_state(payload[7], "final quality"),
        unpack_state(payload[8], "final constraint"),
        unpack_state(payload[9], "final best"),
    )
    if final_states != prior_states:
        raise RuntimeError("native constraint no-change modified solver state")

    followup_events = (
        event(
            constraint_operator,
            "candidate_proposed",
            "constraint_ranked_removal",
            route_indices=(int(removal_routes[0]),),
            affected_route_indices=(int(removal_routes[0]),),
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
            "constraint_removal_no_change",
            removed_customers=removed,
            prefilter_passed=True,
            track="constraint_lane",
            constraint_category=constraint_operator,
            removal_tier=removal_tier,
            removal_size_requested=int(selection[1]),
            removal_size_actual=len(removed),
            stagnation_iterations=int(selection[4]),
            removal_trigger=removal_trigger,
        ),
        event(legacy_operator, "not_applicable", "only_one_route"),
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
            0,
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
        raise RuntimeError("native constraint no-change final state is invalid")
    activity = np.zeros((operator_count, 8), dtype=np.int64)
    by_name = {name: index for index, name in enumerate(FULL_NATIVE_OPERATOR_NAMES)}
    for event_item in all_events:
        index = by_name[str(event_item["operator"])]
        activity[index, 1] += int(bool(event_item["prefilter_passed"]))
        activity[index, 2] += cast(int, event_item["exact_route_evaluations"])
        activity[index, 3] += int(event_item["status"] == "candidate_proposed")
        activity[index, 4] += int(bool(event_item["candidate_feasible"]))
        activity[index, 5] += int(bool(event_item["vehicle_reduction"]))
        activity[index, 6] += int(bool(event_item["distance_improvement"]))
    activity[:, 0] = previous_stream.operator_activity[:, 0]
    refinement_index = by_name["vehicle_reduction_refinement"]
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
    repaired_objective = replay(repaired_routes)
    if legacy[5] is not None:
        raise RuntimeError("native standard-energy duplicated its final transaction")
    transaction = require_tuple(legacy[6], 13, "energy insertion transaction")
    statuses = _require_vector(transaction[1], "standard-energy statuses")
    exact_rows = _require_vector(transaction[5], "standard-energy exact rows")
    if (
        statuses.tolist() != list(expected_statuses)
        or exact_rows.tolist() != list(expected_exact_rows)
    ):
        raise RuntimeError("native standard-energy insertion journal is invalid")
    objective_integers = cast(
        npt.NDArray[np.int64],
        _require_array(
            transaction[2],
            dtype=np.dtype(np.int64),
            shape=(len(expected_statuses), 2),
            name="standard-energy objective integers",
        ),
    )
    objective_floats = cast(
        npt.NDArray[np.float64],
        _require_array(
            transaction[3],
            dtype=np.dtype(np.float64),
            shape=(len(expected_statuses), 2),
            name="standard-energy objective floats",
        ),
    )
    candidate_objective_key: tuple[object, ...]
    if candidate_prepared:
        reported = SolutionObjective(
            int(objective_integers[0, 0]),
            float(objective_floats[0, 0]),
            float(objective_floats[0, 1]),
            int(objective_integers[0, 1]),
        )
        if reported.key != repaired_objective.key:
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
        termination.tolist()
        != [
            0,
            int(previous_stream.termination[1]),
            int(previous_stream.termination[2]) + expected_exact_call_delta,
            int(previous_stream.termination[3]) + expected_exact_call_delta,
            int(previous_stream.termination[4]),
            completed_iterations,
        ]
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
        activity[index, 1] += int(bool(event_item["prefilter_passed"]))
        activity[index, 2] += cast(int, event_item["exact_route_evaluations"])
        activity[index, 3] += int(event_item["status"] == "candidate_proposed")
        activity[index, 4] += int(bool(event_item["candidate_feasible"]))
        activity[index, 5] += int(bool(event_item["vehicle_reduction"]))
        activity[index, 6] += int(bool(event_item["distance_improvement"]))
    activity[:, 0] = previous_stream.operator_activity[:, 0]
    standard_index = by_name["standard"]
    activity[standard_index, 0] += int(feasible_repair)
    refinement_index = by_name["vehicle_reduction_refinement"]
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
    remaining = deadline - clock()
    if remaining <= 0.0:
        raise RuntimeError("full native ALNS reached its deadline before dispatch")
    context = native_runtime.context
    context.assert_matches(instance)
    ordered_names = sorted(context.node_names)
    rank_by_name = {name: rank for rank, name in enumerate(ordered_names)}
    lexical_rank = np.ascontiguousarray(
        [rank_by_name[name] for name in context.node_names],
        dtype=np.int64,
    )
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
    deadline_remaining = np.ascontiguousarray([remaining], dtype=np.float64)
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
    if not isinstance(payload, tuple) or len(payload) != 9:
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
        shape=(4,),
        name="full native timings",
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in timings_array):
        raise RuntimeError("full native ALNS returned invalid timings")
    if not math.isclose(
        float(timings_array[0] + timings_array[1] + timings_array[3]),
        float(timings_array[2]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("full native ALNS timing intervals do not reconcile")
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
    backend_metrics = _decode_full_native_backend_metrics(
        payload[8],
        batch_size=batch_size,
        exact_seconds=float(timings_array[1]),
    )
    expected_sha256 = _full_native_digest(
        route_offsets=route_offsets,
        route_indices=route_indices,
        exact_payload=payload[2],
        counters=counters_array,
        trajectory=trajectory_array,
        semantic_sha256=semantic_stream.transaction_sha256,
        backend_payload=payload[8],
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
    )


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
    "NativeCandidateResolution",
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
