"""Explicit Stage 5.2 native execution protocols.

This module is the only public configuration seam that may combine the
Stage 5.2 native kernels and candidate transaction with historical Stage 3.4
Candidate Control.  Passing ``None`` to :func:`evrptw.alns.solve_alns` keeps
the historical guards and execution paths unchanged.
"""

from __future__ import annotations

import hashlib
import math
import struct
import time
from collections.abc import Callable, Mapping
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
    "route_merge",
    "route_elimination",
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
    semantic_stream: NativeGlobalSemanticStream
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
        or np.any(event_integer[:, 3] < 7)
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
                [1, 2, 0, 0, 0, 0, 0, -1, -2, -1, -1],
                [2, 7, 0, 4, 0, 0, 2, -1, -2, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, -1, -2, -1, -1],
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
            [1, 2, 0, 0, 0, 0, 0, -1, -2, -1, -1],
            [2, 7, 1, 1, 1, 0, 2, 0, -2, 0, 0],
            [
                2,
                7,
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
            [0, 1, 0, 0, 0, 0, 0, -1, -2, -1, -1],
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
    if int(semantic_stream.termination[2]) != int(counters_array[0]):
        raise RuntimeError("full native ALNS semantic iteration count mismatch")
    decoded = decode_exact_charging_batch_numeric(
        instance,
        customer_sequences,
        native_runtime=native_runtime,
        batch_size=batch_size,
        payload=payload[2],
        native_kernel_seconds=0.0,
    )
    _verify_full_native_best_projection(
        instance,
        initial_customer_sequences=initial_customer_sequences,
        semantic_stream=semantic_stream,
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
    "FullNativeALNSResult",
    "FULL_NATIVE_OPERATOR_NAMES",
    "NativeWorkerProtocol",
    "Stage052NativeExecutionConfig",
    "execute_native_candidate_round",
    "decode_native_constraint_semantic_stream",
    "decode_native_global_semantic_stream",
    "execute_full_native_alns",
)
