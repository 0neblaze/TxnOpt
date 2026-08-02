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
from evrptw.charging import ChargingSubproblemResult
from evrptw.cpu_batch import BackendMetrics, decode_exact_charging_batch_numeric
from evrptw.measurement import CheapScreeningConfig
from evrptw.models import Instance
from evrptw.native_kernels import NativeKernelConfig, NativeKernelRuntime
from evrptw.neighborhoods import VehicleOperatorConfig
from evrptw.stage04 import Stage04Config

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
    if not isinstance(payload, tuple) or len(payload) != 7:
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
    expected_sha256 = _full_native_digest(
        route_offsets=route_offsets,
        route_indices=route_indices,
        exact_payload=payload[2],
        counters=counters_array,
        trajectory=trajectory_array,
    )
    if transaction_sha256 != expected_sha256:
        raise RuntimeError("full native ALNS transaction SHA-256 mismatch")
    decoded = decode_exact_charging_batch_numeric(
        instance,
        customer_sequences,
        native_runtime=native_runtime,
        batch_size=batch_size,
        payload=payload[2],
        native_kernel_seconds=float(timings_array[2]),
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
    return FullNativeALNSResult(
        customer_sequences=customer_sequences,
        exact_results=decoded.results,
        backend_metrics=decoded.metrics,
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
    return hashlib.sha256(evidence).hexdigest()


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
    "FullNativeALNSResult",
    "FULL_NATIVE_OPERATOR_NAMES",
    "NativeWorkerProtocol",
    "Stage052NativeExecutionConfig",
    "execute_native_candidate_round",
    "decode_native_constraint_semantic_stream",
    "execute_full_native_alns",
)
