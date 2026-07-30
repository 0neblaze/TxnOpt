from __future__ import annotations

import itertools
import math
import random
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from typing import Any, cast

import numpy as np

from evrptw.cache_incremental import (
    CacheIncrementalConfig,
    CacheStore,
    RouteCacheWriteBatch,
    RouteEvaluationCache,
    RoutePropagationSnapshot,
    StationReachabilityIndex,
    build_route_propagation_snapshot,
    canonical_instance_hash,
    incremental_route_propagation,
)
from evrptw.candidate_control import (
    CandidateControlConfig,
    CandidateControlRuntime,
    CandidatePlan,
)
from evrptw.candidate_transaction import (
    STAGE052_NEGATIVE_SCREENING_RESULT_CACHE_ENTRIES,
    STAGE052_NEGATIVE_SEQUENCE_CACHE_ENTRIES,
    BoundedNegativeSequenceCache,
    BoundedScreeningResultCache,
    CandidateTransactionDeadlineExceeded,
    CandidateTransactionRequest,
    NativeCandidateTransactionConfig,
    NativeCandidateTransactionRuntime,
    NegativeCacheCommit,
    NegativeSequenceCacheBatch,
    execute_candidate_transaction,
    native_screen_candidate_batch,
)
from evrptw.charging import ChargingSubproblemResult, solve_exact_charging
from evrptw.cpu_batch import (
    BackendMetrics,
    ExactBatchDeadlineExceeded,
    ExactChargingBackend,
    solve_exact_charging_batch,
)
from evrptw.exact_deadline import ExactCallController, ExactDeadlineConfig
from evrptw.measurement import (
    COMPLETED_UNIQUE_ROUTE_SEMANTICS,
    CheapScreeningConfig,
    MeasurementConfig,
    ScreeningCheckTrace,
    Stage03ExecutionError,
    Stage03Trace,
    route_result_fields,
)
from evrptw.models import Instance, Node
from evrptw.native_kernels import NativeKernelConfig, NativeKernelRuntime
from evrptw.neighborhoods import (
    ConstraintRemovalOperator,
    NeighborhoodEvent,
    OperatorProfile,
    RemovalSizeSelection,
    RemovalTier,
    RouteEvaluationDeadlineExceeded,
    RouteSequences,
    ScreeningResult,
    VehicleOperatorConfig,
    propose_constraint_removal,
    propose_ejection_chain,
    propose_relocate,
    propose_route_elimination,
    propose_route_merge,
    propose_route_segment_destroy,
    propose_swap,
    propose_two_opt_star,
    repair_constraint_removal,
    repair_vehicle_count_aware,
    repair_vehicle_reduction_refinement,
    screen_route_candidate,
    select_dynamic_removal_size,
)
from evrptw.objective import (
    ObjectiveComparison,
    SolutionObjective,
    accept_annealing_move,
    compare_objectives,
)
from evrptw.stage04 import Stage04Config
from evrptw.validation import validate_routes

__all__ = (
    "ALNSResult",
    "CacheIncrementalConfig",
    "CandidateControlConfig",
    "CheapScreeningConfig",
    "ExactDeadlineConfig",
    "MeasurementConfig",
    "NativeKernelConfig",
    "Stage03ExecutionError",
    "Stage03Trace",
    "Stage04Config",
    "solve_alns",
)

_INFEASIBLE_COST = 1e12
_QUALITY_NEIGHBORHOOD_ORDER = (
    "relocate",
    "swap",
    "two_opt_star",
    "route_segment_destroy",
    "ejection_chain",
)
_QUALITY_NEIGHBORHOODS = frozenset(_QUALITY_NEIGHBORHOOD_ORDER)
_CONSTRAINT_REMOVAL_ORDER = (
    ConstraintRemovalOperator.STATION_PRESSURE.value,
    ConstraintRemovalOperator.TIME_WINDOW_CONFLICT.value,
    ConstraintRemovalOperator.WORST_ENERGY_DETOUR.value,
    ConstraintRemovalOperator.SHAW_RELATED.value,
)
_CONSTRAINT_REMOVAL_NEIGHBORHOODS = frozenset(_CONSTRAINT_REMOVAL_ORDER)
_NEGATIVE_SEQUENCE_CACHE_HIT_CHECKS = (
    ScreeningCheckTrace(
        "negative_sequence_cache",
        "hit",
        True,
        "reused a previously recorded safe screening rejection",
    ),
)


@dataclass(slots=True)
class OperatorStatistics:
    calls: int = 0
    feasible_repairs: int = 0
    accepted: int = 0
    improved: int = 0
    best: int = 0
    vehicle_reductions: int = 0
    accepted_vehicle_reductions: int = 0
    distance_improvements: int = 0
    rejected: int = 0
    # Stage 4 six-category split of *accepted* moves.
    accepted_improving: int = 0
    accepted_equal: int = 0
    accepted_worse: int = 0
    prefilter_passed: int = 0
    prefilter_rejected: int = 0
    new_routes_created: int = 0
    exact_route_evaluations: int = 0
    candidate_proposals: int = 0
    feasible_candidates: int = 0
    failure_reasons: dict[str, int] = field(default_factory=dict)
    weight: float = 1.0
    # Segment-based accumulation (Stage 4).
    segment_reward_sum: float = 0.0
    segment_calls: int = 0
    weight_history: list[tuple[int, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ALNSResult:
    feasible: bool
    routes: tuple[tuple[str, ...], ...]
    customer_sequences: tuple[tuple[str, ...], ...]
    objective: SolutionObjective | None
    vehicle_count: int
    total_energy: float
    total_charged_energy: float
    total_charging_time: float
    iterations: int
    accepted_moves: int
    improving_moves: int
    rejected_moves: int
    first_feasible_time: float
    best_time: float
    runtime_seconds: float
    charging_subproblem_calls: int
    charging_subproblem_time: float
    charging_labels_generated: int
    charging_labels_pruned: int
    destroy_statistics: dict[str, dict[str, object]]
    repair_statistics: dict[str, dict[str, object]]
    operator_profile: str
    neighborhood_statistics: dict[str, dict[str, object]]
    neighborhood_events: tuple[dict[str, object], ...]
    failure_reason: str
    cache_hits: int = 0
    cache_misses: int = 0
    unique_route_evaluations: int = 0
    unique_route_semantics: str = COMPLETED_UNIQUE_ROUTE_SEMANTICS
    effective_iterations: int = 0
    removal_tier_counts: dict[str, int] = field(default_factory=dict)
    maximum_stagnation: int = 0
    constraint_operator_statistics: dict[str, dict[str, object]] = field(default_factory=dict)
    measurement_trace: Stage03Trace | None = None
    screening_statistics: dict[str, object] = field(default_factory=dict)
    cache_incremental_statistics: dict[str, object] = field(default_factory=dict)
    charging_backend: str = ExactChargingBackend.CPU_BATCH.value
    batch_size: int = 128
    backend_metrics: dict[str, object] = field(default_factory=dict)
    termination_mode: str = "wall_clock"
    watchdog_triggered: bool = False
    exact_started_calls: int = 0
    exact_completed_calls: int = 0
    exact_interrupted_calls: int = 0
    exact_budget_exhaustions: int = 0
    termination_reason: str = ""
    exact_deadline_statistics: dict[str, object] = field(default_factory=dict)
    candidate_control_statistics: dict[str, object] = field(default_factory=dict)
    candidate_transaction_statistics: dict[str, object] = field(default_factory=dict)
    candidate_transaction_events: tuple[dict[str, object], ...] = ()
    candidate_work_hash: str = ""
    route_result_hash: str = ""
    # Stage 4 adaptive-weight and search-control statistics.
    stage04_statistics: dict[str, object] = field(default_factory=dict)
    stage04_weight_history: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    stage04_temperature_history: tuple[tuple[int, float], ...] = ()
    stage04_event_log: tuple[dict[str, object], ...] = ()
    initial_routes: tuple[tuple[str, ...], ...] = ()
    initial_customer_sequences: tuple[tuple[str, ...], ...] = ()
    initial_objective: SolutionObjective | None = None
    iteration_limit_completed_at_seconds: float | None = None

    @property
    def objective_value(self) -> float:
        return self.objective.total_distance if self.objective is not None else float("inf")


@dataclass(frozen=True, slots=True)
class _EvaluatedSolution:
    sequences: tuple[tuple[str, ...], ...]
    charging: tuple[ChargingSubproblemResult, ...]
    feasible: bool
    objective: SolutionObjective | None


def _evaluated_full_routes(
    solution: _EvaluatedSolution,
) -> tuple[tuple[str, ...], ...] | None:
    """Return complete depot/station routes only for a feasible evaluated solution."""

    if not solution.feasible:
        return None
    routes = tuple(tuple(result.route) for result in solution.charging)
    if len(routes) != len(solution.sequences) or any(not route for route in routes):
        raise RuntimeError("feasible evaluated solution lacks complete charging routes")
    return routes


@dataclass(slots=True)
class _IncumbentRouteLedger:
    by_lane: dict[
        str,
        dict[tuple[str, ...], ChargingSubproblemResult],
    ] = field(default_factory=dict)

    def remember(self, lane: str, solution: _EvaluatedSolution) -> None:
        self.by_lane[lane] = dict(zip(solution.sequences, solution.charging, strict=True))

    def get(
        self,
        sequence: tuple[str, ...],
    ) -> ChargingSubproblemResult | None:
        for lane in sorted(self.by_lane):
            result = self.by_lane[lane].get(sequence)
            if result is not None:
                return result
        return None


def _candidate_control_skip_result(reason: str) -> ChargingSubproblemResult:
    return ChargingSubproblemResult(
        False,
        (),
        float("inf"),
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
        0.0,
        f"candidate_control:{reason}",
    )


def _candidate_transaction_skip_result(reason: str) -> ChargingSubproblemResult:
    return ChargingSubproblemResult(
        False,
        (),
        float("inf"),
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
        0.0,
        f"candidate_transaction:{reason}",
    )


class _Evaluator:
    def __init__(
        self,
        instance: Instance,
        *,
        deadline: float,
        measurement_trace: Stage03Trace | None = None,
        lane: str = "legacy",
        screening_config: CheapScreeningConfig | None = None,
        negative_screening_cache: (
            dict[str, ScreeningResult]
            | BoundedScreeningResultCache[ScreeningResult]
            | None
        ) = None,
        negative_screening_sequences: (
            dict[tuple[str, ...], str] | BoundedNegativeSequenceCache | None
        ) = None,
        cache_incremental_config: CacheIncrementalConfig | None = None,
        route_cache: RouteEvaluationCache | None = None,
        backend: ExactChargingBackend | str = ExactChargingBackend.CPU_SCALAR,
        batch_size: int = 128,
        disable_cache: bool = False,
        batch_work_enabled: bool = False,
        exact_call_controller: ExactCallController | None = None,
        candidate_control_runtime: CandidateControlRuntime | None = None,
        candidate_transaction_runtime: NativeCandidateTransactionRuntime | None = None,
        incumbent_route_ledger: _IncumbentRouteLedger | None = None,
        native_runtime: NativeKernelRuntime | None = None,
    ) -> None:
        self.instance = instance
        self.deadline = deadline
        self.measurement_trace = measurement_trace
        self.lane = lane
        self.incumbent_lane = lane
        self.screening_config = (
            screening_config if screening_config is not None and screening_config.enabled else None
        )
        self.cache_incremental_config = (
            cache_incremental_config
            if cache_incremental_config is not None and cache_incremental_config.enabled
            else None
        )
        self.route_cache = route_cache
        if disable_cache and route_cache is not None:
            raise ValueError("disable_cache cannot be combined with a Stage 3.2 route cache")
        self.local_cache_enabled = not disable_cache
        self.backend = ExactChargingBackend(backend)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.batch_work_enabled = batch_work_enabled
        self.exact_call_controller = exact_call_controller
        self.candidate_control_runtime = candidate_control_runtime
        self.candidate_transaction_runtime = candidate_transaction_runtime
        self.incumbent_route_ledger = incumbent_route_ledger
        self.native_runtime = native_runtime
        self.pending_candidate_cache: dict[tuple[str, ...], ChargingSubproblemResult] = {}
        self.pending_negative_screening_sequences: dict[tuple[str, ...], str] = {}
        self.backend_metrics = BackendMetrics(self.backend.value, batch_size)
        self.reachability_index = (
            StationReachabilityIndex(instance)
            if self.cache_incremental_config is not None
            and self.cache_incremental_config.station_reachability_bitset
            else None
        )
        self.propagation_snapshots: dict[tuple[str, ...], RoutePropagationSnapshot] = {}
        self.incremental_propagations = 0
        self.incremental_fallbacks = 0
        self.incremental_reused_prefix_edges = 0
        self.incremental_reused_suffix_edges = 0
        self.negative_screening_cache = (
            negative_screening_cache if negative_screening_cache is not None else {}
        )
        self.negative_screening_sequences = (
            negative_screening_sequences if negative_screening_sequences is not None else {}
        )
        self.iteration: int | None = None
        self.operator = "initialization"
        self.cache: dict[tuple[str, ...], ChargingSubproblemResult] = {}
        self.evaluated_routes: set[tuple[str, ...]] = set()
        self.evaluated_route_keys: set[tuple[str, tuple[str, ...]]] = set()
        self.calls = 0
        self.cache_hits = 0
        self.runtime = 0.0
        self.labels_generated = 0
        self.labels_pruned = 0
        self.screening_calls = 0
        self.screening_passes = 0
        self.screening_rejections = 0
        self.screening_cache_hits = 0
        self.screening_exact_call_blocked = 0
        self.screening_runtime = 0.0
        self.screening_reason_counts: dict[str, int] = {}

    def _record_exact_budget_boundary(self) -> None:
        controller = self.exact_call_controller
        if controller is None or controller.boundary_recorded:
            return
        controller.boundary_recorded = True
        if self.measurement_trace is not None:
            self.measurement_trace.events.append(
                {
                    "event_type": "exact_budget_boundary",
                    "status": "budget_exhausted",
                    "timestamp_seconds": self.measurement_trace._offset(),
                    "lane": self.lane,
                    "iteration": self.iteration,
                    "operator": self.operator,
                    "exact_call_budget": controller.budget,
                    "started_calls": controller.started_calls,
                    "completed_calls": controller.completed_calls,
                    "interrupted_calls": controller.interrupted_calls,
                }
            )

    def _discard_pending_candidate_cache(self, reason: str) -> None:
        if not self.pending_candidate_cache and not self.pending_negative_screening_sequences:
            return
        discarded = len(self.pending_candidate_cache)
        discarded_negative = len(self.pending_negative_screening_sequences)
        self.pending_candidate_cache.clear()
        self.pending_negative_screening_sequences.clear()
        if self.measurement_trace is not None:
            self.measurement_trace.events.append(
                {
                    "event_type": "candidate_cache_rollback",
                    "status": "discarded",
                    "reason": reason,
                    "discarded_entries": discarded,
                    "discarded_negative_entries": discarded_negative,
                    "timestamp_seconds": self.measurement_trace._offset(),
                    "lane": self.lane,
                    "iteration": self.iteration,
                    "operator": self.operator,
                }
            )

    def _commit_pending_candidate_cache(self) -> None:
        if not self.pending_candidate_cache and not self.pending_negative_screening_sequences:
            return
        pending = tuple(self.pending_candidate_cache.items())
        pending_negative = dict(self.pending_negative_screening_sequences)
        local_insertions: list[tuple[str, ...]] = []
        negative_insertions: list[tuple[str, ...]] = []
        negative_sequence_batch: NegativeSequenceCacheBatch | None = None
        route_cache_batch: RouteCacheWriteBatch | None = None
        native_negative_commit: NegativeCacheCommit | None = None
        stores: tuple[CacheStore, ...] = ()
        try:
            if self.route_cache is None:
                if self.local_cache_enabled:
                    for sequence, result in pending:
                        if sequence in self.cache:
                            raise RuntimeError(
                                "atomic candidate cache batch contains a local non-miss key"
                            )
                        local_insertions.append(sequence)
                        self.cache[sequence] = result
            else:
                route_cache_batch = self.route_cache.begin_store_many_atomic(pending)
                stores = route_cache_batch.stores
            if isinstance(
                self.negative_screening_sequences,
                BoundedNegativeSequenceCache,
            ):
                negative_sequence_batch = (
                    self.negative_screening_sequences.begin_store_many_atomic(
                        pending_negative
                    )
                )
            else:
                for sequence, reason in pending_negative.items():
                    existing = self.negative_screening_sequences.get(sequence)
                    if existing is not None and existing != reason:
                        raise RuntimeError(
                            "candidate negative cache reason changed during commit"
                        )
                    if existing is None:
                        negative_insertions.append(sequence)
                        self.negative_screening_sequences[sequence] = reason
            if pending_negative and self.candidate_transaction_runtime is not None:
                if self.native_runtime is None:
                    raise RuntimeError(
                        "candidate transaction negative-cache commit lacks native runtime"
                    )
                if (
                    negative_sequence_batch is not None
                    and negative_sequence_batch.rollover
                ):
                    native_negative_commit = (
                        self.candidate_transaction_runtime.replace_negative_cache_entries(
                            self.negative_screening_sequences,
                            self.native_runtime.context.name_to_index,
                        )
                    )
                else:
                    native_negative_commit = (
                        self.candidate_transaction_runtime.commit_negative_cache_entries(
                            pending_negative,
                            self.native_runtime.context.name_to_index,
                        )
                    )
            for store in stores:
                if self.measurement_trace is None:
                    continue
                for evicted in store.evicted:
                    self.measurement_trace.record_cache_event(
                        operation="evict",
                        route_key=evicted.route_key,
                        cache_key_digest=evicted.digest,
                        lane=self.lane,
                        iteration=self.iteration,
                        operator=self.operator,
                        reason="lru_capacity_or_memory",
                        current_entries=store.current_entries,
                        current_bytes=store.current_bytes,
                    )
                self.measurement_trace.record_cache_event(
                    operation=("store" if store.stored else "oversize_not_cached"),
                    route_key=store.key.route_key,
                    cache_key_digest=store.key.digest,
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    reason=store.reason,
                    entry_bytes=store.entry_bytes,
                    current_entries=store.current_entries,
                    current_bytes=store.current_bytes,
                )
            if self.measurement_trace is not None:
                self.measurement_trace.events.append(
                    {
                        "event_type": "candidate_cache_commit",
                        "status": "committed",
                        "committed_entries": len(pending),
                        "committed_negative_entries": len(pending_negative),
                        "timestamp_seconds": self.measurement_trace._offset(),
                        "lane": self.lane,
                        "iteration": self.iteration,
                        "operator": self.operator,
                    }
                )
            if route_cache_batch is not None:
                assert self.route_cache is not None
                self.route_cache.commit_store_batch(route_cache_batch)
            if native_negative_commit is not None:
                assert self.candidate_transaction_runtime is not None
                self.candidate_transaction_runtime.commit_negative_cache_batch(
                    native_negative_commit
                )
            if negative_sequence_batch is not None:
                assert isinstance(
                    self.negative_screening_sequences,
                    BoundedNegativeSequenceCache,
                )
                self.negative_screening_sequences.commit_store_batch(
                    negative_sequence_batch
                )
        except BaseException:
            if native_negative_commit is not None and native_negative_commit.active:
                assert self.candidate_transaction_runtime is not None
                self.candidate_transaction_runtime.rollback_negative_cache_batch(
                    native_negative_commit
                )
            if route_cache_batch is not None and route_cache_batch.active:
                assert self.route_cache is not None
                self.route_cache.rollback_store_batch(route_cache_batch)
            for sequence in local_insertions:
                self.cache.pop(sequence, None)
            if negative_sequence_batch is not None and negative_sequence_batch.active:
                assert isinstance(
                    self.negative_screening_sequences,
                    BoundedNegativeSequenceCache,
                )
                self.negative_screening_sequences.rollback_store_batch(
                    negative_sequence_batch
                )
            else:
                assert isinstance(self.negative_screening_sequences, dict)
                for sequence in negative_insertions:
                    self.negative_screening_sequences.pop(sequence, None)
            raise
        self.pending_candidate_cache.clear()
        self.pending_negative_screening_sequences.clear()

    @contextmanager
    def measurement_context(
        self,
        *,
        lane: str | None = None,
        iteration: int | None = None,
        operator: str | None = None,
    ) -> Iterator[None]:
        previous = (self.lane, self.iteration, self.operator)
        if self.measurement_trace is not None:
            if lane is not None:
                self.lane = lane
            self.iteration = iteration
            if operator is not None:
                self.operator = operator
        try:
            yield
        finally:
            if self.measurement_trace is not None:
                self.lane, self.iteration, self.operator = previous

    def set_measurement_context(
        self,
        *,
        lane: str,
        iteration: int | None,
        operator: str,
    ) -> None:
        if self.measurement_trace is not None:
            self.lane = lane
            self.iteration = iteration
            self.operator = operator

    @property
    def screening_enabled(self) -> bool:
        return self.screening_config is not None

    @property
    def candidate_control_enabled(self) -> bool:
        return self.candidate_control_runtime is not None

    @property
    def candidate_transaction_enabled(self) -> bool:
        runtime = self.candidate_transaction_runtime
        return runtime is not None and runtime.config.implementation_mode in {
            "batched_screening",
            "candidate_transaction",
        }

    @property
    def pair_pruning_enabled(self) -> bool:
        return self.candidate_transaction_runtime is not None

    @property
    def candidate_control_route_change_limit(self) -> int | None:
        runtime = self.candidate_control_runtime
        if runtime is None:
            return None
        return runtime.config.max_exact_calls_per_round

    def candidate_route_batch(
        self,
        sequences: Sequence[tuple[str, ...]],
        *,
        route_change_status: str = "changed",
        prescreened: bool = False,
        exact_budget: int | None = None,
        base_sequences: Sequence[tuple[str, ...]] | None = None,
    ) -> tuple[ChargingSubproblemResult, ...]:
        clean = tuple(sequence for sequence in sequences if sequence)
        if self.candidate_transaction_runtime is not None:
            if self.candidate_transaction_runtime.config.implementation_mode == "pair_pruning":
                return self.route_batch(
                    clean,
                    route_change_status=route_change_status,
                )
            if exact_budget is None:
                raise ValueError("native candidate transaction requires the operator exact budget")
            return self._native_candidate_route_batch(
                clean,
                route_change_status=route_change_status,
                exact_budget=exact_budget,
                base_sequences=base_sequences,
            )
        if self.candidate_control_runtime is not None:
            return self._controlled_candidate_route_batch(
                clean,
                route_change_status=route_change_status,
                prescreened=prescreened,
            )
        return self.route_batch(
            clean,
            route_change_status=route_change_status,
        )

    def _native_candidate_route_batch(
        self,
        clean: tuple[tuple[str, ...], ...],
        *,
        route_change_status: str,
        exact_budget: int,
        base_sequences: Sequence[tuple[str, ...]] | None,
    ) -> tuple[ChargingSubproblemResult, ...]:
        runtime = self.candidate_transaction_runtime
        native_runtime = self.native_runtime
        if runtime is None or native_runtime is None:
            raise RuntimeError("native candidate transaction runtime is incomplete")
        if not clean:
            return ()
        if base_sequences is not None and len(base_sequences) != len(clean):
            raise ValueError("candidate transaction base sequences must align with candidates")
        effective_budget = exact_budget
        controller = self.exact_call_controller
        if controller is not None and controller.budget is not None:
            effective_budget = min(
                effective_budget,
                max(0, controller.budget - controller.started_calls),
            )

        incremental_rows = np.zeros((len(clean), 6), dtype=np.float64)
        if base_sequences is not None:
            epsilon = (
                self.screening_config.epsilon
                if self.screening_config is not None
                else native_runtime.context.reachability_epsilon
            )
            for index, (base_sequence, candidate) in enumerate(
                zip(base_sequences, clean, strict=True)
            ):
                base_snapshot = self.propagation_snapshots.get(base_sequence)
                if base_snapshot is None:
                    base_snapshot = build_route_propagation_snapshot(
                        self.instance,
                        base_sequence,
                        epsilon=epsilon,
                    )
                    self.propagation_snapshots[base_sequence] = base_snapshot
                propagation = incremental_route_propagation(
                    self.instance,
                    base_snapshot,
                    candidate,
                    epsilon=epsilon,
                    native_runtime=native_runtime,
                )
                if propagation.status == "incremental":
                    self.incremental_propagations += 1
                    self.incremental_reused_prefix_edges += propagation.reused_prefix_edges
                    self.incremental_reused_suffix_edges += propagation.reused_suffix_edges
                    incremental_rows[index] = (
                        1.0,
                        propagation.distance_lower_bound,
                        propagation.min_time_window_slack,
                        propagation.finish_time,
                        float(propagation.forward_feasible),
                        float(propagation.backward_feasible),
                    )
                else:
                    self.incremental_fallbacks += 1
                if self.measurement_trace is not None:
                    self.measurement_trace.record_incremental_propagation(
                        operator=self.operator,
                        lane=self.lane,
                        iteration=self.iteration,
                        base_sequence=base_sequence,
                        candidate_sequence=candidate,
                        status=propagation.status,
                        reason=propagation.reason,
                        distance_lower_bound=propagation.distance_lower_bound,
                        min_time_window_slack=propagation.min_time_window_slack,
                        finish_time=propagation.finish_time,
                        reused_prefix_edges=propagation.reused_prefix_edges,
                        reused_suffix_edges=propagation.reused_suffix_edges,
                        recomputed_forward_edges=propagation.recomputed_forward_edges,
                        recomputed_backward_edges=propagation.recomputed_backward_edges,
                    )

        def screen_batch(
            candidates: tuple[tuple[str, ...], ...],
        ) -> Any:
            batch = native_screen_candidate_batch(
                self.instance,
                candidates,
                native_runtime=native_runtime,
                transaction_runtime=runtime,
                negative_cache=self.negative_screening_sequences,
                deadline=self.deadline,
                incremental=incremental_rows,
            )
            self.screening_calls += len(batch.sequences)
            self.screening_passes += sum(
                batch.accepted(index) for index in range(len(batch.sequences))
            )
            self.screening_rejections += sum(
                not batch.accepted(index) and batch.native_status(index) != "negative_cache_hit"
                for index in range(len(batch.sequences))
            )
            self.screening_cache_hits += batch.counters["negative_cache_hits"]
            self.screening_exact_call_blocked += sum(
                not batch.accepted(index) for index in range(len(batch.sequences))
            )
            for index in range(len(batch.sequences)):
                reason = batch.reason(index)
                if reason:
                    self.screening_reason_counts[reason] = (
                        self.screening_reason_counts.get(reason, 0) + 1
                    )
            return batch

        def rejected(
            sequence: tuple[str, ...],
            reason: str,
            native_status: str,
        ) -> ChargingSubproblemResult:
            if native_status != "negative_cache_hit":
                self.pending_negative_screening_sequences[sequence] = reason
            return ChargingSubproblemResult(
                False,
                (),
                float("inf"),
                0.0,
                0.0,
                0.0,
                0,
                0,
                0,
                0.0,
                f"cheap_screening:{reason}",
            )

        try:
            if runtime.config.implementation_mode == "batched_screening":
                screening = screen_batch(clean)
                screening_event = {
                    "event_type": "native_candidate_screening_batch",
                    "status": "committed",
                    "lane": self.lane,
                    "iteration": self.iteration,
                    "operator": self.operator,
                    "input_candidates": len(clean),
                    "candidates": clean,
                    "screening_pool_hash": screening.candidate_pool_hash,
                    "screening_integrity_evidence": screening.integrity_evidence(),
                    "counters": dict(screening.counters),
                }
                resolved: list[ChargingSubproblemResult | None] = [None] * len(clean)
                waiting_indices: dict[tuple[str, ...], list[int]] = {}
                known_results: dict[tuple[str, ...], ChargingSubproblemResult] = {}
                exact_sequences: list[tuple[str, ...]] = []
                for index, sequence in enumerate(screening.sequences):
                    if not screening.accepted(index):
                        resolved[index] = rejected(
                            sequence,
                            screening.reason(index),
                            screening.native_status(index),
                        )
                        continue
                    if sequence in waiting_indices:
                        waiting_indices[sequence].append(index)
                        continue
                    known = known_results.get(sequence)
                    if known is not None:
                        resolved[index] = known
                        continue
                    cached = self._lookup_cached_result(
                        sequence,
                        route_change_status,
                    )[0]
                    if cached is not None:
                        known_results[sequence] = cached
                        resolved[index] = cached
                        continue
                    if len(exact_sequences) >= effective_budget:
                        skipped = _candidate_transaction_skip_result(
                            "operator_exact_budget_exhausted"
                        )
                        known_results[sequence] = skipped
                        resolved[index] = skipped
                        continue
                    waiting_indices[sequence] = [index]
                    exact_sequences.append(sequence)
                if time.perf_counter() >= self.deadline:
                    raise CandidateTransactionDeadlineExceeded("before_exact_batch")
                exact_results = self._solve_uncached_batch(
                    tuple(exact_sequences),
                    route_change_status,
                )
                for sequence, result in zip(
                    exact_sequences,
                    exact_results,
                    strict=True,
                ):
                    known_results[sequence] = result
                    for index in waiting_indices[sequence]:
                        resolved[index] = result
                if any(result is None for result in resolved):
                    raise RuntimeError("batched screening lost an ordered result")
                self._commit_pending_candidate_cache()
                if self.measurement_trace is not None:
                    screening_passes = sum(
                        screening.accepted(index) for index in range(len(screening.sequences))
                    )
                    screening_cache_hits = screening.counters["negative_cache_hits"]
                    screening_exact_call_blocked = len(screening.sequences) - screening_passes
                    screening_rejections = (
                        screening_exact_call_blocked - screening_cache_hits
                    )
                    screening_reason_counts: dict[str, int] = {}
                    for index in range(len(screening.sequences)):
                        reason = screening.reason(index)
                        if reason:
                            screening_reason_counts[reason] = (
                                screening_reason_counts.get(reason, 0) + 1
                            )
                    self.measurement_trace.record_screening_aggregate(
                        {
                            "timestamp_seconds": self.measurement_trace._offset(),
                            "screening_passes": screening_passes,
                            "screening_rejections": screening_rejections,
                            "screening_cache_hits": screening_cache_hits,
                            "screening_exact_call_blocked": screening_exact_call_blocked,
                            "screening_reason_counts": dict(
                                sorted(screening_reason_counts.items())
                            ),
                            **screening_event,
                        },
                        calls=len(screening.sequences),
                        passes=screening_passes,
                        rejections=screening_rejections,
                        cache_hits=screening_cache_hits,
                        exact_call_blocked=screening_exact_call_blocked,
                        reason_counts=screening_reason_counts,
                    )
                return tuple(result for result in resolved if result is not None)
            transaction = execute_candidate_transaction(
                CandidateTransactionRequest(
                    clean,
                    lane=self.lane,
                    operator=self.operator,
                    iteration=self.iteration,
                    exact_budget=effective_budget,
                    deadline=self.deadline,
                ),
                screen_batch=screen_batch,
                cache_lookup=lambda sequence: self._lookup_cached_result(
                    sequence,
                    route_change_status,
                )[0],
                exact_batch=lambda sequences: self._solve_uncached_batch(
                    sequences,
                    route_change_status,
                    stage_candidate_transaction=True,
                ),
                stage_cache_write=lambda sequence, result: self.pending_candidate_cache.__setitem__(
                    sequence, result
                ),
                commit_cache_writes=self._commit_pending_candidate_cache,
                rollback_cache_writes=self._discard_pending_candidate_cache,
                rejected_result=rejected,
                skipped_result=_candidate_transaction_skip_result,
            )
        except CandidateTransactionDeadlineExceeded as error:
            self._discard_pending_candidate_cache(
                f"candidate_transaction_deadline:{error.boundary}"
            )
            if self.measurement_trace is not None:
                self.measurement_trace.record_deadline_boundary(
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    boundary=error.boundary,
                    reason=str(error),
                )
            raise _TimeLimitReached(clean[0]) from error
        except BaseException as error:
            self._discard_pending_candidate_cache(
                f"candidate_transaction_failure:{type(error).__name__}:{error}"
            )
            raise
        runtime.record(transaction.audit)
        if self.measurement_trace is not None:
            self.measurement_trace.record_screening_aggregate(
                {
                    "event_type": "native_candidate_transaction",
                    "status": "committed",
                    "timestamp_seconds": self.measurement_trace._offset(),
                    **asdict(transaction.audit),
                },
                calls=transaction.audit.input_candidates,
                passes=transaction.audit.screening_passes,
                rejections=transaction.audit.screening_rejections,
                cache_hits=transaction.audit.screening_cache_hits,
                exact_call_blocked=transaction.audit.screening_exact_call_blocked,
                reason_counts=transaction.audit.screening_reason_counts,
            )
        return transaction.ordered_results

    def record_candidate_screening_aggregate(
        self,
        counts: Mapping[str, int],
        candidate_pool_hash: str,
    ) -> None:
        runtime = self.candidate_control_runtime
        if runtime is None:
            return
        runtime.events.append(
            {
                "event_type": "candidate_screening_aggregate",
                "status": "aggregated",
                "lane": self.lane,
                "iteration": self.iteration,
                "operator": self.operator,
                "counts": dict(sorted(counts.items())),
                "aggregate_count": sum(counts.values()),
                "candidate_pool_hash": candidate_pool_hash,
            }
        )

    def remember_incumbent(self, solution: _EvaluatedSolution) -> None:
        """Retain only the current lane incumbent outside the bounded LRU."""

        ledger = self.incumbent_route_ledger
        if ledger is not None:
            ledger.remember(self.incumbent_lane, solution)

    def incumbent_route_result(
        self,
        sequence: tuple[str, ...],
    ) -> ChargingSubproblemResult | None:
        ledger = self.incumbent_route_ledger
        return None if ledger is None else ledger.get(sequence)

    def select_feasible_candidate_plan(
        self,
        plans: Sequence[tuple[tuple[str, ...], ...]],
        *,
        current_sequences: tuple[tuple[str, ...], ...],
        allow_vehicle_increase: bool = False,
    ) -> tuple[tuple[str, ...], ...] | None:
        """Rank complete solution states and exact-evaluate selected plans atomically."""

        feasible = self.evaluate_feasible_candidate_plans(
            plans,
            current_sequences=current_sequences,
            allow_vehicle_increase=allow_vehicle_increase,
        )
        return feasible[0] if feasible else None

    def evaluate_feasible_candidate_plans(
        self,
        plans: Sequence[tuple[tuple[str, ...], ...]],
        *,
        current_sequences: tuple[tuple[str, ...], ...],
        allow_vehicle_increase: bool = False,
    ) -> tuple[tuple[tuple[str, ...], ...], ...]:
        """Return selected feasible plans in deterministic objective order."""

        runtime = self.candidate_control_runtime
        if runtime is None:
            raise RuntimeError("complete candidate-plan selection requires Stage 3.4")
        current_set = set(current_sequences)
        screen_cache: dict[tuple[str, ...], ScreeningResult] = {}

        def candidate_screen(sequence: tuple[str, ...]) -> ScreeningResult:
            decision = screen_cache.get(sequence)
            if decision is None:
                decision = screen_route_candidate(
                    self.instance,
                    sequence,
                    full=True,
                    native_runtime=self.native_runtime,
                )
                screen_cache[sequence] = decision
            return decision

        rankable: list[CandidatePlan] = []
        for ordinal, sequences in enumerate(plans):
            if not allow_vehicle_increase and len(sequences) > len(current_sequences):
                runtime.events.append(
                    {
                        "event_type": "candidate_plan_screening",
                        "status": "vehicle_increase_rejected",
                        "lane": self.lane,
                        "iteration": self.iteration,
                        "operator": self.operator,
                        "proposal_ordinal": ordinal,
                        "vehicle_count": len(sequences),
                    }
                )
                continue
            screens = tuple(candidate_screen(sequence) for sequence in sequences)
            if not all(screen.accepted for screen in screens):
                runtime.events.append(
                    {
                        "event_type": "candidate_plan_screening",
                        "status": "screening_rejected",
                        "lane": self.lane,
                        "iteration": self.iteration,
                        "operator": self.operator,
                        "proposal_ordinal": ordinal,
                        "vehicle_count": len(sequences),
                        "customer_sequences": [list(sequence) for sequence in sequences],
                        "reasons": [screen.reason for screen in screens if not screen.accepted],
                    }
                )
                continue
            rankable.append(
                CandidatePlan(
                    candidate_id=ordinal,
                    customer_sequences=sequences,
                    vehicle_count=len(sequences),
                    optimistic_total_distance=sum(
                        screen.distance_lower_bound for screen in screens
                    ),
                    changed_route_count=sum(sequence not in current_set for sequence in sequences),
                    proposal_ordinal=ordinal,
                )
            )
        selected = runtime.select_plans(
            rankable,
            lane=self.lane,
            iteration=self.iteration or 0,
            operator=self.operator,
        )
        feasible: list[_EvaluatedSolution] = []
        precomputed = {
            sequence: result
            for sequence in current_sequences
            if (result := self.incumbent_route_result(sequence)) is not None
        }
        for plan in selected:
            candidate = self.solution(
                plan.customer_sequences,
                precomputed_routes=precomputed,
            )
            if any(
                result.failure_reason.startswith("candidate_control:")
                for result in candidate.charging
            ):
                continue
            runtime.mark_plan_attempted(
                plan,
                lane=self.lane,
                iteration=self.iteration,
                operator=self.operator,
            )
            if not candidate.feasible or candidate.objective is None:
                continue
            feasible.append(candidate)
        feasible.sort(
            key=lambda candidate: (
                candidate.objective.key if candidate.objective is not None else (),
                candidate.sequences,
            )
        )
        return tuple(candidate.sequences for candidate in feasible)

    @property
    def cache_incremental_enabled(self) -> bool:
        return self.cache_incremental_config is not None and self.route_cache is not None

    def has_cached_route(self, sequence: tuple[str, ...]) -> bool:
        if not self.local_cache_enabled:
            return False
        if self.route_cache is not None:
            return self.route_cache.contains(sequence)
        return sequence in self.cache

    def screen(
        self,
        sequence: tuple[str, ...],
        *,
        reference_distance: float | None = None,
        base_sequence: tuple[str, ...] | None = None,
        operator: str = "",
    ) -> ScreeningResult:
        """Run the shared screening seam and record one independent decision."""

        if self.screening_config is None:
            raise RuntimeError("screen() called while cheap screening is disabled")
        if time.perf_counter() >= self.deadline:
            if self.measurement_trace is not None:
                self.measurement_trace.record_deadline_boundary(
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    boundary="before_cheap_screening",
                    route_sequence=sequence,
                )
            raise _TimeLimitReached(sequence)

        started = time.perf_counter()
        key = (
            self.measurement_trace.register_route(sequence)
            if self.measurement_trace is not None
            else "route:" + "|".join(f"{len(name)}:{name}" for name in sequence)
        )
        cached = (
            self.negative_screening_cache.get(key)
            if self.screening_config.negative_sequence_cache
            else None
        )
        negative_cache_hit = cached is not None
        incremental_metrics = None
        if (
            cached is None
            and self.cache_incremental_enabled
            and base_sequence is not None
            and operator in {"relocate", "swap"}
        ):
            base_snapshot = self.propagation_snapshots.get(base_sequence)
            if base_snapshot is None:
                base_snapshot = build_route_propagation_snapshot(
                    self.instance,
                    base_sequence,
                    epsilon=self.screening_config.epsilon,
                )
                self.propagation_snapshots[base_sequence] = base_snapshot
            incremental_metrics = incremental_route_propagation(
                self.instance,
                base_snapshot,
                sequence,
                epsilon=self.screening_config.epsilon,
                native_runtime=self.native_runtime,
            )
            if incremental_metrics.status == "fallback":
                self.incremental_fallbacks += 1
            else:
                self.incremental_propagations += 1
                self.incremental_reused_prefix_edges += incremental_metrics.reused_prefix_edges
                self.incremental_reused_suffix_edges += incremental_metrics.reused_suffix_edges
        result = (
            cached
            if cached is not None
            else screen_route_candidate(
                self.instance,
                sequence,
                full=True,
                reference_distance=(
                    reference_distance
                    if reference_distance is not None
                    else (
                        self.propagation_snapshots[base_sequence].total_distance
                        if base_sequence is not None and base_sequence in self.propagation_snapshots
                        else None
                    )
                ),
                epsilon=self.screening_config.epsilon,
                reachability_index=self.reachability_index,
                incremental_metrics=(
                    incremental_metrics
                    if incremental_metrics is not None
                    and incremental_metrics.status == "incremental"
                    else None
                ),
                native_runtime=self.native_runtime,
            )
        )
        if (
            self.cache_incremental_enabled
            and cached is None
            and sequence not in self.propagation_snapshots
        ):
            self.propagation_snapshots[sequence] = build_route_propagation_snapshot(
                self.instance,
                sequence,
                epsilon=self.screening_config.epsilon,
            )
        completed = time.perf_counter()
        self.screening_runtime += completed - started
        self.screening_calls += 1
        if negative_cache_hit:
            self.screening_cache_hits += 1
        elif result.accepted:
            self.screening_passes += 1
        else:
            self.screening_rejections += 1
            if self.screening_config.negative_sequence_cache:
                if isinstance(
                    self.negative_screening_cache,
                    BoundedScreeningResultCache,
                ):
                    self.negative_screening_cache.store(key, result)
                else:
                    self.negative_screening_cache[key] = result
        if result.reason:
            self.screening_reason_counts[result.reason] = (
                self.screening_reason_counts.get(result.reason, 0) + 1
            )
        blocked = not result.accepted
        if blocked:
            self.screening_exact_call_blocked += 1
        status = (
            "negative_cache_hit"
            if negative_cache_hit
            else "pass"
            if result.accepted
            else "rejected"
        )
        if self.measurement_trace is not None:
            decision_checks = (
                _NEGATIVE_SEQUENCE_CACHE_HIT_CHECKS if negative_cache_hit else result.checks
            )
            self.measurement_trace.record_screening_decision(
                sequence,
                lane=self.lane,
                iteration=self.iteration,
                operator=self.operator,
                status=status,
                first_failed_check=result.first_failed_check,
                reason=result.reason,
                checks=decision_checks,
                demand=result.demand,
                min_time_window_slack=result.min_time_window_slack,
                distance_lower_bound=result.distance_lower_bound,
                distance_increment_lower_bound=result.distance_increment_lower_bound,
                single_segment_reachable=result.single_segment_reachable,
                structural_energy_lower_bound=result.structural_energy_lower_bound,
                negative_cache_hit=negative_cache_hit,
                exact_call_blocked=blocked,
                started_at=self.measurement_trace._offset(started),
                completed_at=self.measurement_trace._offset(completed),
                registered_route_key=key,
                negative_evidence_token=id(result) if negative_cache_hit else None,
            )
            if incremental_metrics is not None:
                self.measurement_trace.record_incremental_propagation(
                    operator=operator or self.operator,
                    lane=self.lane,
                    iteration=self.iteration,
                    base_sequence=base_sequence or (),
                    candidate_sequence=sequence,
                    status=incremental_metrics.status,
                    reason=incremental_metrics.reason,
                    distance_lower_bound=incremental_metrics.distance_lower_bound,
                    min_time_window_slack=incremental_metrics.min_time_window_slack,
                    finish_time=incremental_metrics.finish_time,
                    reused_prefix_edges=incremental_metrics.reused_prefix_edges,
                    reused_suffix_edges=incremental_metrics.reused_suffix_edges,
                    recomputed_forward_edges=incremental_metrics.recomputed_forward_edges,
                    recomputed_backward_edges=incremental_metrics.recomputed_backward_edges,
                )
            if completed >= self.deadline:
                self.measurement_trace.record_deadline_boundary(
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    boundary="after_cheap_screening",
                    route_sequence=sequence,
                    reason="cheap screening completed at or after the lane deadline",
                )
        if completed >= self.deadline:
            self._discard_pending_candidate_cache("deadline_after_screening")
            raise _TimeLimitReached(sequence)
        return result

    def route_with_status(
        self,
        sequence: tuple[str, ...],
        route_change_status: str,
    ) -> ChargingSubproblemResult:
        return self.route(sequence, route_change_status=route_change_status)

    def screening_statistics(self) -> dict[str, object]:
        statistics: dict[str, object] = {
            "screening_calls": self.screening_calls,
            "screening_passes": self.screening_passes,
            "screening_rejections": self.screening_rejections,
            "screening_cache_hits": self.screening_cache_hits,
            "screening_exact_call_blocked": self.screening_exact_call_blocked,
            "screening_runtime_seconds": self.screening_runtime,
            "screening_reason_counts": dict(sorted(self.screening_reason_counts.items())),
        }
        if isinstance(
            self.negative_screening_cache,
            BoundedScreeningResultCache,
        ):
            statistics["negative_screening_result_cache"] = (
                self.negative_screening_cache.statistics()
            )
        return statistics

    def incremental_statistics(self) -> dict[str, object]:
        reachability = (
            self.reachability_index.to_dict() if self.reachability_index is not None else {}
        )
        return {
            "incremental_propagations": self.incremental_propagations,
            "incremental_fallbacks": self.incremental_fallbacks,
            "incremental_reused_prefix_edges": self.incremental_reused_prefix_edges,
            "incremental_reused_suffix_edges": self.incremental_reused_suffix_edges,
            "station_reachability": reachability,
        }

    def _lookup_cached_result(
        self,
        sequence: tuple[str, ...],
        route_change_status: str,
    ) -> tuple[ChargingSubproblemResult | None, str]:
        """Perform one ordered cache lookup and emit its observable result."""

        pending = self.pending_candidate_cache.get(sequence)
        if pending is not None:
            cache_key_digest = (
                self.route_cache.make_key(sequence).digest if self.route_cache is not None else ""
            )
            self.cache_hits += 1
            if self.measurement_trace is not None:
                fields = route_result_fields(pending)
                route_key = self.measurement_trace.register_route(sequence)
                self.measurement_trace.record_cache_event(
                    operation="candidate_pending_hit",
                    route_key=route_key,
                    cache_key_digest=cache_key_digest,
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                )
                self.measurement_trace.record_route_evaluation(
                    sequence,
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    kind="cache_hit",
                    exact_started=False,
                    exact_completed=False,
                    cache_key_digest=cache_key_digest,
                    route_change_status=route_change_status,
                    **fields,
                )
            return pending, cache_key_digest
        if time.perf_counter() >= self.deadline:
            self._discard_pending_candidate_cache("deadline_before_route_evaluation")
            if self.measurement_trace is not None:
                self.measurement_trace.record_deadline_boundary(
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    boundary="before_route_evaluation",
                    route_sequence=sequence,
                )
            raise _TimeLimitReached(sequence)
        cache_key_digest = ""
        cached: ChargingSubproblemResult | None = None
        if self.route_cache is not None:
            lookup = self.route_cache.lookup(sequence)
            cache_key_digest = lookup.key.digest
            if self.measurement_trace is not None:
                self.measurement_trace.record_cache_event(
                    operation="lookup",
                    route_key=lookup.key.route_key,
                    cache_key_digest=cache_key_digest,
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    current_entries=lookup.current_entries,
                    current_bytes=lookup.current_bytes,
                )
                self.measurement_trace.record_cache_event(
                    operation="hit" if lookup.hit else "miss",
                    route_key=lookup.key.route_key,
                    cache_key_digest=cache_key_digest,
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    current_entries=lookup.current_entries,
                    current_bytes=lookup.current_bytes,
                )
            if lookup.hit:
                cached = lookup.result
        elif self.local_cache_enabled:
            cached = self.cache.get(sequence)
        if cached is None:
            return None, cache_key_digest

        self.cache_hits += 1
        if self.measurement_trace is not None:
            started = time.perf_counter()
            fields = route_result_fields(cached)
            self.measurement_trace.record_route_evaluation(
                sequence,
                lane=self.lane,
                iteration=self.iteration,
                operator=self.operator,
                kind="cache_hit",
                started_at=self.measurement_trace._offset(started),
                completed_at=self.measurement_trace._offset(),
                exact_started=False,
                exact_completed=False,
                cache_key_digest=cache_key_digest,
                route_change_status=route_change_status,
                **fields,
            )
        return cached, cache_key_digest

    def route(
        self,
        sequence: tuple[str, ...],
        *,
        route_change_status: str = "unknown",
    ) -> ChargingSubproblemResult:
        if self.cache_incremental_enabled and route_change_status == "unknown":
            # Initial and newly created routes are changes relative to the
            # current candidate; never leave their Stage 3.2 status ambiguous.
            route_change_status = "changed"
        incumbent_result = self.incumbent_route_result(sequence)
        if route_change_status == "unchanged" and incumbent_result is not None:
            return self._precomputed_route(
                sequence,
                incumbent_result,
            )
        if self.screening_config is not None:
            screen = self.screen(sequence, operator=self.operator)
            if not screen.accepted:
                return ChargingSubproblemResult(
                    False,
                    (),
                    float("inf"),
                    0.0,
                    0.0,
                    0.0,
                    0,
                    0,
                    0,
                    0.0,
                    f"cheap_screening:{screen.reason}",
                )
        cached, cache_key_digest = self._lookup_cached_result(
            sequence,
            route_change_status,
        )
        if cached is not None:
            return cached
        if self.candidate_control_runtime is not None:
            granted = self.candidate_control_runtime.reserve(
                1,
                atomic=True,
                context=f"{self.lane}:{self.operator}:single_route",
            )
            if granted == 0:
                return _candidate_control_skip_result("round_budget_exhausted")
        reservation = (
            self.exact_call_controller.reserve(1)
            if self.exact_call_controller is not None
            else None
        )
        if reservation is not None and reservation.granted == 0:
            self._discard_pending_candidate_cache("exact_call_budget_exhausted")
            self._record_exact_budget_boundary()
            raise _TimeLimitReached(sequence)
        started = time.perf_counter()
        if self.measurement_trace is not None:
            started_offset = self.measurement_trace._offset(started)
        else:
            started_offset = 0.0
        try:
            if self.backend is ExactChargingBackend.CPU_SCALAR:
                result = solve_exact_charging(self.instance, sequence)
                self.backend_metrics.work_batches += 1
                self.backend_metrics.transition_batches += 1
                self.backend_metrics.exact_calls += 1
                self.backend_metrics.batch_launches += 1
                self.backend_metrics.started_calls += 1
                self.backend_metrics.completed_calls += 1
                self.backend_metrics.launch_occupancies.append(1)
                self.backend_metrics.transitions += max(0, result.labels_generated - 1)
                self.backend_metrics.total_seconds += result.runtime_seconds
                self.backend_metrics.label_management_seconds += result.runtime_seconds
            else:
                batch = (
                    self.candidate_control_runtime.solve_batch(
                        self.instance,
                        (sequence,),
                        batch_size=self.batch_size,
                        deadline=self.deadline,
                        lane=self.lane,
                        iteration=self.iteration,
                        operator=self.operator,
                    )
                    if self.candidate_control_runtime is not None
                    else solve_exact_charging_batch(
                        self.instance,
                        (sequence,),
                        backend=self.backend,
                        batch_size=self.batch_size,
                        deadline=self.deadline,
                        native_runtime=self.native_runtime,
                    )
                )
                result = batch.results[0]
                self.backend_metrics.add(batch.metrics)
        except BaseException as error:
            self._discard_pending_candidate_cache(f"exact_call_interrupted:{type(error).__name__}")
            completed_on_error = (
                error.completed_exact_calls if isinstance(error, ExactBatchDeadlineExceeded) else 0
            )
            if completed_on_error == 1:
                self.evaluated_routes.add(sequence)
                self.evaluated_route_keys.add((self.lane, sequence))
            if self.exact_call_controller is not None:
                self.exact_call_controller.complete(completed_on_error)
                self.exact_call_controller.interrupt(1 - completed_on_error)
            if isinstance(error, ExactBatchDeadlineExceeded):
                self.backend_metrics.add(error.metrics)
                self.calls += error.completed_exact_calls
                self.runtime += error.metrics.total_seconds
            if self.measurement_trace is not None:
                self.measurement_trace.record_route_evaluation(
                    sequence,
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    kind="exact_call",
                    started_at=started_offset,
                    completed_at=self.measurement_trace._offset(),
                    exact_started=True,
                    exact_completed=completed_on_error == 1,
                    feasible=None,
                    failure_reason=f"{type(error).__name__}: {error}",
                    cache_key_digest=(
                        self.route_cache.make_key(sequence).digest
                        if self.route_cache is not None
                        else ""
                    ),
                    route_change_status=route_change_status,
                )
            if isinstance(error, ExactBatchDeadlineExceeded):
                raise _TimeLimitReached(
                    sequence,
                    exact_route_evaluations=error.completed_exact_calls,
                ) from error
            raise
        exact_completed_at = time.perf_counter()
        transactional_deadline = (
            self.exact_call_controller is not None and exact_completed_at >= self.deadline
        )
        if transactional_deadline:
            assert self.exact_call_controller is not None
            if self.backend_metrics.completed_calls < 1:
                raise RuntimeError("deadline transaction lacks one completed backend call")
            self.backend_metrics.completed_calls -= 1
            self.backend_metrics.interrupted_calls += 1
            self.exact_call_controller.interrupt(1)
            late_exact_call_id = None
            if self.measurement_trace is not None:
                late_exact_call_id = self.measurement_trace.record_route_evaluation(
                    sequence,
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    kind="exact_call",
                    started_at=started_offset,
                    completed_at=self.measurement_trace._offset(exact_completed_at),
                    exact_started=True,
                    exact_completed=False,
                    feasible=None,
                    failure_reason="exact call transaction completed at or after the lane deadline",
                    cache_key_digest=(
                        self.route_cache.make_key(sequence).digest
                        if self.route_cache is not None
                        else ""
                    ),
                    route_change_status=route_change_status,
                )
                self.measurement_trace.record_deadline_boundary(
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    boundary="after_exact_call",
                    route_sequence=sequence,
                    exact_call_id=late_exact_call_id,
                    reason="exact call transaction completed at or after the lane deadline",
                )
            self._discard_pending_candidate_cache("deadline_after_exact_call")
            raise _TimeLimitReached(sequence, exact_route_evaluations=0)
        self.evaluated_routes.add(sequence)
        self.evaluated_route_keys.add((self.lane, sequence))
        if self.exact_call_controller is not None:
            self.exact_call_controller.complete(1)
        if self.exact_call_controller is not None and not transactional_deadline:
            self.pending_candidate_cache[sequence] = result
        elif not transactional_deadline and self.route_cache is None and self.local_cache_enabled:
            # Stage 0--3.1 retain their historical lane-local cache.  Stage
            # 3.2 has only the bounded RouteEvaluationCache so the memory cap
            # applies to every exact result held by the evaluator.
            self.cache[sequence] = result
        cache_key_digest = ""
        if self.exact_call_controller is not None and self.route_cache is not None:
            cache_key_digest = self.route_cache.make_key(sequence).digest
        if (
            self.exact_call_controller is None
            and not transactional_deadline
            and self.route_cache is not None
        ):
            store = self.route_cache.store(sequence, result)
            cache_key_digest = store.key.digest
            if self.measurement_trace is not None:
                for evicted in store.evicted:
                    self.measurement_trace.record_cache_event(
                        operation="evict",
                        route_key=evicted.route_key,
                        cache_key_digest=evicted.digest,
                        lane=self.lane,
                        iteration=self.iteration,
                        operator=self.operator,
                        reason="lru_capacity_or_memory",
                        current_entries=store.current_entries,
                        current_bytes=store.current_bytes,
                    )
                self.measurement_trace.record_cache_event(
                    operation=("store" if store.stored else "oversize_not_cached"),
                    route_key=store.key.route_key,
                    cache_key_digest=store.key.digest,
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    reason=store.reason,
                    entry_bytes=store.entry_bytes,
                    current_entries=store.current_entries,
                    current_bytes=store.current_bytes,
                )
        self.calls += 1
        self.runtime += result.runtime_seconds
        self.labels_generated += result.labels_generated
        self.labels_pruned += result.labels_pruned
        exact_call_id: int | None = None
        if self.measurement_trace is not None:
            fields = route_result_fields(result)
            exact_call_id = self.measurement_trace.record_route_evaluation(
                sequence,
                lane=self.lane,
                iteration=self.iteration,
                operator=self.operator,
                kind="exact_call",
                started_at=started_offset,
                completed_at=self.measurement_trace._offset(exact_completed_at),
                exact_started=True,
                exact_completed=True,
                cache_key_digest=cache_key_digest,
                route_change_status=route_change_status,
                **fields,
            )
        if time.perf_counter() >= self.deadline:
            self._discard_pending_candidate_cache("deadline_after_exact_call")
            if self.measurement_trace is not None:
                self.measurement_trace.record_deadline_boundary(
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    boundary="after_exact_call",
                    route_sequence=sequence,
                    exact_call_id=exact_call_id,
                    reason="exact call completed after the lane deadline",
                )
            raise _TimeLimitReached(sequence, exact_route_evaluations=1)
        return result

    def solution(
        self,
        sequences: tuple[tuple[str, ...], ...],
        *,
        precomputed_routes: dict[tuple[str, ...], ChargingSubproblemResult] | None = None,
    ) -> _EvaluatedSolution:
        clean = tuple(sequence for sequence in sequences if sequence)
        if precomputed_routes is None and self.backend is not ExactChargingBackend.CPU_SCALAR:
            charging = self.route_batch(clean, route_change_status="changed")
        elif precomputed_routes is not None and self.candidate_control_runtime is not None:
            missing = tuple(sequence for sequence in clean if sequence not in precomputed_routes)
            missing_results = iter(self.route_batch(missing, route_change_status="changed"))
            charging = tuple(
                self._precomputed_route(sequence, precomputed_routes[sequence])
                if sequence in precomputed_routes
                else next(missing_results)
                for sequence in clean
            )
        else:
            charging = tuple(
                self._precomputed_route(sequence, precomputed_routes[sequence])
                if precomputed_routes is not None and sequence in precomputed_routes
                else self.route(sequence, route_change_status="changed")
                for sequence in clean
            )
        feasible = bool(clean) and all(result.feasible for result in charging)
        objective = (
            sum(
                (
                    SolutionObjective.from_route(
                        self.instance,
                        result.route,
                        total_distance=result.distance,
                        total_charging_time=result.charging_time,
                    )
                    for result in charging
                ),
                start=SolutionObjective.zero(),
            )
            if feasible
            else None
        )
        if self.exact_call_controller is not None and time.perf_counter() >= self.deadline:
            self._discard_pending_candidate_cache("deadline_before_candidate_commit")
            if self.measurement_trace is not None:
                self.measurement_trace.record_deadline_boundary(
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    boundary="before_candidate_commit",
                    reason="candidate completed at or after the exact deadline",
                )
            raise _TimeLimitReached(clean[-1] if clean else ())
        self._commit_pending_candidate_cache()
        if not feasible:
            return _EvaluatedSolution(clean, charging, False, None)
        if objective is None:
            raise RuntimeError("feasible candidate is missing its objective")
        return _EvaluatedSolution(clean, charging, True, objective)

    def route_batch(
        self,
        sequences: Sequence[tuple[str, ...]],
        *,
        route_change_status: str = "changed",
        candidate_pool: bool = False,
    ) -> tuple[ChargingSubproblemResult, ...]:
        """Evaluate ordered routes while batching CPU cache misses."""

        clean = tuple(sequence for sequence in sequences if sequence)
        if not clean:
            return ()
        if self.candidate_control_runtime is not None and candidate_pool:
            return self._controlled_candidate_route_batch(
                clean,
                route_change_status=route_change_status,
            )
        if self.backend is ExactChargingBackend.CPU_SCALAR or len(clean) == 1:
            return tuple(
                self.route(sequence, route_change_status=route_change_status) for sequence in clean
            )
        resolved: list[ChargingSubproblemResult | None] = [None] * len(clean)
        pending_sequences: list[tuple[str, ...]] = []
        pending_indices: list[int] = []
        pending_set: set[tuple[str, ...]] = set()
        cache_available = self.route_cache is not None or self.local_cache_enabled

        def flush_pending() -> None:
            if not pending_sequences:
                return
            batch_results = self._solve_uncached_batch(
                tuple(pending_sequences),
                route_change_status,
            )
            for request_index, result in zip(
                pending_indices,
                batch_results,
                strict=True,
            ):
                resolved[request_index] = result
            pending_sequences.clear()
            pending_indices.clear()
            pending_set.clear()

        for index, sequence in enumerate(clean):
            incumbent_result = self.incumbent_route_result(sequence)
            if route_change_status == "unchanged" and incumbent_result is not None:
                flush_pending()
                resolved[index] = self._precomputed_route(
                    sequence,
                    incumbent_result,
                )
                continue
            if self.screening_config is not None:
                screen = self.screen(sequence, operator=self.operator)
                if not screen.accepted:
                    flush_pending()
                    resolved[index] = ChargingSubproblemResult(
                        False,
                        (),
                        float("inf"),
                        0.0,
                        0.0,
                        0.0,
                        0,
                        0,
                        0,
                        0.0,
                        f"cheap_screening:{screen.reason}",
                    )
                    continue
            if cache_available and (sequence in pending_set or self.has_cached_route(sequence)):
                flush_pending()
            cached, _ = self._lookup_cached_result(
                sequence,
                route_change_status,
            )
            if cached is not None:
                resolved[index] = cached
                continue
            pending_sequences.append(sequence)
            pending_indices.append(index)
            pending_set.add(sequence)
        flush_pending()
        completed = tuple(result for result in resolved if result is not None)
        if len(completed) != len(clean):
            raise RuntimeError("CPU batch lost a cached route result")
        return completed

    def _controlled_candidate_route_batch(
        self,
        clean: tuple[tuple[str, ...], ...],
        *,
        route_change_status: str,
        prescreened: bool = False,
    ) -> tuple[ChargingSubproblemResult, ...]:
        runtime = self.candidate_control_runtime
        if runtime is None:
            raise RuntimeError("controlled route batch requires Stage 3.4 runtime")
        resolved: list[ChargingSubproblemResult | None] = [None] * len(clean)
        rankable: list[tuple[int, tuple[str, ...], float]] = []
        for index, sequence in enumerate(clean):
            lower_bound = 0.0
            if self.screening_config is not None:
                screen = (
                    screen_route_candidate(
                        self.instance,
                        sequence,
                        native_runtime=self.native_runtime,
                    )
                    if prescreened
                    else self.screen(sequence, operator=self.operator)
                )
                lower_bound = screen.distance_lower_bound
                if not screen.accepted:
                    resolved[index] = _candidate_control_skip_result(
                        f"screening_rejected:{screen.reason}"
                    )
                    continue
            rankable.append((index, sequence, lower_bound))
        selected = runtime.select_route_candidates(
            rankable,
            lane=self.lane,
            iteration=self.iteration,
            operator=self.operator,
        )
        selected_misses: list[int] = []
        for index in selected:
            cached, _ = self._lookup_cached_result(
                clean[index],
                route_change_status,
            )
            if cached is not None:
                resolved[index] = cached
            else:
                selected_misses.append(index)
        granted = runtime.reserve(
            len(selected_misses),
            atomic=True,
            context=f"{self.lane}:{self.operator}:candidate_pool",
        )
        active_indices = tuple(selected_misses[:granted])
        active_sequences = tuple(clean[index] for index in active_indices)
        if active_sequences:
            active_results = self._solve_uncached_batch(
                active_sequences,
                route_change_status,
                candidate_control_reserved=True,
            )
            for index, result in zip(active_indices, active_results, strict=True):
                resolved[index] = result
        selected_set = set(active_indices)
        top_k_set = set(selected)
        for index, _sequence, _lower_bound in rankable:
            if resolved[index] is not None:
                continue
            reason = (
                "round_budget_exhausted"
                if index in top_k_set and index not in selected_set
                else "not_selected"
            )
            resolved[index] = _candidate_control_skip_result(reason)
        completed = tuple(result for result in resolved if result is not None)
        if len(completed) != len(clean):
            raise RuntimeError("candidate control lost an ordered route result")
        return completed

    def _solve_uncached_batch(
        self,
        sequences: tuple[tuple[str, ...], ...],
        route_change_status: str,
        *,
        candidate_control_reserved: bool = False,
        stage_candidate_transaction: bool = False,
    ) -> tuple[ChargingSubproblemResult, ...]:
        """Solve a non-empty ordered group of known cache misses."""

        if not sequences:
            return ()
        if self.candidate_control_runtime is not None and not candidate_control_reserved:
            granted = self.candidate_control_runtime.reserve(
                len(sequences),
                atomic=True,
                context=f"{self.lane}:{self.operator}:complete_candidate",
            )
            if granted == 0:
                return tuple(
                    _candidate_control_skip_result("round_budget_exhausted") for _ in sequences
                )
        if (
            self.candidate_control_runtime is not None
            and self.exact_call_controller is not None
            and self.exact_call_controller.budget is not None
            and self.exact_call_controller.budget - self.exact_call_controller.started_calls
            < len(sequences)
        ):
            self.candidate_control_runtime.events.append(
                {
                    "event_type": "candidate_control_budget",
                    "status": "global_budget_atomic_skip",
                    "context": f"{self.lane}:{self.operator}:complete_candidate",
                    "requested": len(sequences),
                    "granted": 0,
                    "remaining": (
                        self.exact_call_controller.budget - self.exact_call_controller.started_calls
                    ),
                    "iteration": self.iteration,
                }
            )
            return tuple(
                _candidate_control_skip_result("global_budget_atomic_skip") for _ in sequences
            )
        if time.perf_counter() >= self.deadline:
            self._discard_pending_candidate_cache("deadline_before_exact_batch")
            raise _TimeLimitReached(sequences[0])
        reservation = (
            self.exact_call_controller.reserve(len(sequences))
            if self.exact_call_controller is not None
            else None
        )
        if reservation is not None and reservation.granted == 0:
            self._discard_pending_candidate_cache("exact_call_budget_exhausted")
            self._record_exact_budget_boundary()
            raise _TimeLimitReached(sequences[0])
        active_sequences = (
            sequences[: reservation.granted] if reservation is not None else sequences
        )
        partial_budget_batch = reservation is not None and reservation.partial
        batch_started = time.perf_counter()
        started_offset = (
            self.measurement_trace._offset(batch_started)
            if self.measurement_trace is not None
            else 0.0
        )
        try:
            batch = (
                self.candidate_control_runtime.solve_batch(
                    self.instance,
                    active_sequences,
                    batch_size=self.batch_size,
                    deadline=self.deadline,
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                )
                if self.candidate_control_runtime is not None
                else solve_exact_charging_batch(
                    self.instance,
                    active_sequences,
                    backend=self.backend,
                    batch_size=self.batch_size,
                    deadline=self.deadline,
                    native_runtime=self.native_runtime,
                )
            )
        except BaseException as error:
            self._discard_pending_candidate_cache(f"exact_batch_interrupted:{type(error).__name__}")
            completed_indices = (
                set(error.completed_indices)
                if isinstance(error, ExactBatchDeadlineExceeded)
                else set()
            )
            for index in completed_indices:
                sequence = active_sequences[index]
                self.evaluated_routes.add(sequence)
                self.evaluated_route_keys.add((self.lane, sequence))
            if self.exact_call_controller is not None:
                self.exact_call_controller.complete(len(completed_indices))
                self.exact_call_controller.interrupt(len(active_sequences) - len(completed_indices))
            if isinstance(error, ExactBatchDeadlineExceeded):
                self.backend_metrics.add(error.metrics)
                self.calls += len(completed_indices)
                self.runtime += error.metrics.total_seconds
            if self.measurement_trace is not None:
                completed_offset = self.measurement_trace._offset()
                for index, sequence in enumerate(active_sequences):
                    self.measurement_trace.record_route_evaluation(
                        sequence,
                        lane=self.lane,
                        iteration=self.iteration,
                        operator=self.operator,
                        kind="exact_call",
                        started_at=started_offset,
                        completed_at=completed_offset,
                        exact_started=True,
                        exact_completed=index in completed_indices,
                        feasible=None,
                        failure_reason=f"{type(error).__name__}: {error}",
                        cache_key_digest=(
                            self.route_cache.make_key(sequence).digest
                            if self.route_cache is not None
                            else ""
                        ),
                        route_change_status=route_change_status,
                    )
            if isinstance(error, ExactBatchDeadlineExceeded):
                raise _TimeLimitReached(
                    sequences[0],
                    exact_route_evaluations=error.completed_exact_calls,
                ) from error
            raise
        batch_completed = time.perf_counter()
        transactional_deadline = (
            self.exact_call_controller is not None and batch_completed >= self.deadline
        )
        if transactional_deadline:
            assert self.exact_call_controller is not None
            if batch.metrics.completed_calls < len(batch.results):
                raise RuntimeError(
                    "deadline transaction backend completed-call count is incomplete"
                )
            batch.metrics.completed_calls -= len(batch.results)
            batch.metrics.interrupted_calls += len(batch.results)
            self.exact_call_controller.interrupt(len(batch.results))
        elif self.exact_call_controller is not None:
            self.exact_call_controller.complete(len(batch.results))
        self.backend_metrics.add(batch.metrics)
        if transactional_deadline:
            if self.measurement_trace is not None:
                completed_offset = self.measurement_trace._offset(batch_completed)
                for sequence in active_sequences:
                    self.measurement_trace.record_route_evaluation(
                        sequence,
                        lane=self.lane,
                        iteration=self.iteration,
                        operator=self.operator,
                        kind="exact_call",
                        started_at=started_offset,
                        completed_at=completed_offset,
                        exact_started=True,
                        exact_completed=False,
                        feasible=None,
                        failure_reason=(
                            "exact batch transaction completed at or after the lane deadline"
                        ),
                        cache_key_digest=(
                            self.route_cache.make_key(sequence).digest
                            if self.route_cache is not None
                            else ""
                        ),
                        route_change_status=route_change_status,
                    )
                self.measurement_trace.record_deadline_boundary(
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    boundary="after_exact_batch",
                    route_sequence=active_sequences[-1],
                    reason="CPU exact batch transaction completed at or after the lane deadline",
                )
            self._discard_pending_candidate_cache("deadline_after_exact_batch")
            raise _TimeLimitReached(active_sequences[-1], exact_route_evaluations=0)
        self.calls += len(batch.results)
        self.runtime += batch.metrics.total_seconds
        self.labels_generated += sum(item.labels_generated for item in batch.results)
        self.labels_pruned += sum(item.labels_pruned for item in batch.results)
        self.evaluated_routes.update(active_sequences)
        self.evaluated_route_keys.update((self.lane, sequence) for sequence in active_sequences)
        for sequence, result in zip(active_sequences, batch.results, strict=True):
            cache_key_digest = (
                self.route_cache.make_key(sequence).digest if self.route_cache is not None else ""
            )
            if stage_candidate_transaction or (
                self.exact_call_controller is not None
                and not (partial_budget_batch or transactional_deadline)
            ):
                self.pending_candidate_cache[sequence] = result
            elif (
                not partial_budget_batch
                and not transactional_deadline
                and self.route_cache is None
                and self.local_cache_enabled
            ):
                self.cache[sequence] = result
            if (
                self.exact_call_controller is None
                and not partial_budget_batch
                and not transactional_deadline
                and self.route_cache is not None
            ):
                store = self.route_cache.store(sequence, result)
                cache_key_digest = store.key.digest
                if self.measurement_trace is not None:
                    for evicted in store.evicted:
                        self.measurement_trace.record_cache_event(
                            operation="evict",
                            route_key=evicted.route_key,
                            cache_key_digest=evicted.digest,
                            lane=self.lane,
                            iteration=self.iteration,
                            operator=self.operator,
                            reason="lru_capacity_or_memory",
                            current_entries=store.current_entries,
                            current_bytes=store.current_bytes,
                        )
                    self.measurement_trace.record_cache_event(
                        operation=("store" if store.stored else "oversize_not_cached"),
                        route_key=store.key.route_key,
                        cache_key_digest=store.key.digest,
                        lane=self.lane,
                        iteration=self.iteration,
                        operator=self.operator,
                        reason=store.reason,
                        entry_bytes=store.entry_bytes,
                        current_entries=store.current_entries,
                        current_bytes=store.current_bytes,
                    )
            if self.measurement_trace is not None:
                fields = route_result_fields(result)
                self.measurement_trace.record_route_evaluation(
                    sequence,
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    kind="exact_call",
                    started_at=started_offset,
                    completed_at=self.measurement_trace._offset(batch_completed),
                    exact_started=True,
                    exact_completed=True,
                    cache_key_digest=cache_key_digest,
                    route_change_status=route_change_status,
                    **fields,
                )
        if partial_budget_batch:
            self._discard_pending_candidate_cache("partial_exact_call_budget_batch")
            self._record_exact_budget_boundary()
            raise _TimeLimitReached(
                active_sequences[-1],
                exact_route_evaluations=len(active_sequences),
            )
        if time.perf_counter() >= self.deadline:
            self._discard_pending_candidate_cache("deadline_after_exact_batch")
            if self.measurement_trace is not None:
                self.measurement_trace.record_deadline_boundary(
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    boundary="after_exact_batch",
                    route_sequence=active_sequences[-1],
                    reason="CPU exact batch completed at or after the lane deadline",
                )
            raise _TimeLimitReached(
                active_sequences[-1],
                exact_route_evaluations=len(active_sequences),
            )
        return batch.results

    def _precomputed_route(
        self,
        sequence: tuple[str, ...],
        result: ChargingSubproblemResult,
    ) -> ChargingSubproblemResult:
        if self.measurement_trace is not None and time.perf_counter() >= self.deadline:
            if self.measurement_trace is not None:
                self.measurement_trace.record_deadline_boundary(
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    boundary="before_precomputed_route",
                    route_sequence=sequence,
                )
            raise _TimeLimitReached(sequence)
        if self.measurement_trace is not None:
            fields = route_result_fields(result)
            started = time.perf_counter()
            self.measurement_trace.record_route_evaluation(
                sequence,
                lane=self.lane,
                iteration=self.iteration,
                operator=self.operator,
                kind="precomputed_route",
                started_at=self.measurement_trace._offset(started),
                completed_at=self.measurement_trace._offset(),
                exact_started=False,
                exact_completed=False,
                route_change_status="unchanged",
                cache_key_digest=(
                    self.route_cache.make_key(sequence).digest
                    if self.route_cache is not None
                    else ""
                ),
                **fields,
            )
        if self.measurement_trace is not None and time.perf_counter() >= self.deadline:
            if self.measurement_trace is not None:
                self.measurement_trace.record_deadline_boundary(
                    lane=self.lane,
                    iteration=self.iteration,
                    operator=self.operator,
                    boundary="after_precomputed_route",
                    route_sequence=sequence,
                    reason="precomputed route completed after the lane deadline",
                )
            raise _TimeLimitReached(sequence)
        return result


def _unchanged_precomputed_routes(
    evaluator: _Evaluator,
    current: _EvaluatedSolution,
    candidate_sequences: tuple[tuple[str, ...], ...],
) -> dict[tuple[str, ...], ChargingSubproblemResult] | None:
    """Reuse only routes that are unchanged in the Stage 3.2 candidate.

    This helper is deliberately opt-in.  Historical Stage 0--3.1 paths keep
    their lane-local evaluation behaviour, while Stage 3.2 never re-enters
    exact charging for a route that survived an operator unchanged.
    """

    if not evaluator.cache_incremental_enabled:
        return None
    current_results = {
        sequence: result
        for sequence, result in zip(current.sequences, current.charging, strict=True)
    }
    return {
        sequence: current_results[sequence]
        for sequence in candidate_sequences
        if sequence in current_results
    }


class _TimeLimitReached(RouteEvaluationDeadlineExceeded):
    pass


class _NeighborhoodEventStream(list[dict[str, object]]):
    """Historical list by default; zero-retention live stream when a sink is set."""

    def __init__(
        self,
        sink: Callable[[Mapping[str, object]], None] | None = None,
    ) -> None:
        super().__init__()
        self._sink = sink
        self.emitted_count = 0

    def append(self, event: dict[str, object]) -> None:
        if self._sink is None:
            super().append(event)
        else:
            self._sink(event)
            self.emitted_count += 1

    def extend(self, events: Iterable[dict[str, object]]) -> None:
        if self._sink is None:
            super().extend(events)
            return
        for event in events:
            self.append(event)


def _solve_alns(
    instance: Instance,
    *,
    seed: int,
    max_iterations: int | None = 2_000,
    time_limit_seconds: float = 60.0,
    removal_fraction: float = 0.2,
    operator_profile: OperatorProfile | str = OperatorProfile.STAGE02_CONSTRAINT_GUIDED,
    vehicle_operator_config: VehicleOperatorConfig | None = None,
    measurement_trace: Stage03Trace | None = None,
    screening_config: CheapScreeningConfig | None = None,
    cache_incremental_config: CacheIncrementalConfig | None = None,
    backend: ExactChargingBackend | str = ExactChargingBackend.CPU_BATCH,
    batch_size: int = 128,
    termination_mode: str = "wall_clock",
    disable_cache: bool = False,
    exact_call_controller: ExactCallController | None = None,
    candidate_control_runtime: CandidateControlRuntime | None = None,
    candidate_transaction_config: NativeCandidateTransactionConfig | None = None,
    initial_customer_sequences: tuple[tuple[str, ...], ...] | None = None,
    initial_solution_provenance: Mapping[str, object] | None = None,
    stage04_config: Stage04Config | None = None,
    native_kernel_config: NativeKernelConfig | None = None,
    neighborhood_event_sink: Callable[[Mapping[str, object]], None] | None = None,
) -> ALNSResult:
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    if not 0.0 < removal_fraction <= 1.0:
        raise ValueError("removal_fraction must be in (0, 1]")
    if termination_mode not in {"wall_clock", "fixed_work"}:
        raise ValueError("termination_mode must be 'wall_clock' or 'fixed_work'")
    fixed_exact_calls = (
        exact_call_controller is not None
        and exact_call_controller.config.mode == "exact_call_budget"
    )
    batch_work_enabled = termination_mode == "fixed_work" or fixed_exact_calls
    if max_iterations is None and batch_work_enabled:
        raise ValueError("max_iterations=None requires wall_clock termination")
    profile = OperatorProfile(operator_profile)
    vehicle_config = vehicle_operator_config or VehicleOperatorConfig()
    cache_enabled = cache_incremental_config is not None and cache_incremental_config.enabled
    if cache_enabled and (screening_config is None or not screening_config.enabled):
        raise ValueError(
            "Stage 3.2 cache/incremental evaluation requires enabled Stage 3.1 screening"
        )
    if cache_incremental_config is not None and cache_incremental_config.enabled:
        cache_incremental_config = replace(
            cache_incremental_config,
            instance_hash=(
                cache_incremental_config.instance_hash or canonical_instance_hash(instance)
            ),
        )

    started = time.perf_counter()
    rng = random.Random(seed)
    constraint_rng = random.Random(seed ^ 0x5EED23)
    overall_deadline = started + time_limit_seconds
    constraint_lane_budget = (
        min(
            vehicle_config.constraint_lane_time_budget_seconds,
            time_limit_seconds * 0.25,
        )
        if profile is OperatorProfile.STAGE02_CONSTRAINT_GUIDED and not fixed_exact_calls
        else 0.0
    )
    legacy_lane_budget = time_limit_seconds - constraint_lane_budget
    if legacy_lane_budget <= 0.0:
        raise ValueError("constraint lane time budget must be smaller than the time limit")
    legacy_deadline = started + legacy_lane_budget
    negative_screening_cache: (
        dict[str, ScreeningResult]
        | BoundedScreeningResultCache[ScreeningResult]
        | None
    ) = (
        (
            BoundedScreeningResultCache(
                capacity=STAGE052_NEGATIVE_SCREENING_RESULT_CACHE_ENTRIES
            )
            if candidate_transaction_config is not None
            else {}
        )
        if screening_config is not None and screening_config.enabled
        else None
    )
    negative_screening_sequences: (
        dict[tuple[str, ...], str] | BoundedNegativeSequenceCache | None
    ) = (
        BoundedNegativeSequenceCache(
            capacity=STAGE052_NEGATIVE_SEQUENCE_CACHE_ENTRIES
        )
        if candidate_transaction_config is not None
        else None
    )
    candidate_transaction_runtime = (
        NativeCandidateTransactionRuntime(candidate_transaction_config)
        if candidate_transaction_config is not None
        else None
    )
    shared_route_cache = (
        RouteEvaluationCache(instance, cache_incremental_config)
        if cache_enabled
        and cache_incremental_config is not None
        and cache_incremental_config.shared_across_lanes
        else None
    )
    lane_route_caches: list[RouteEvaluationCache | None]
    if shared_route_cache is not None:
        lane_route_caches = [shared_route_cache] * 3
    else:
        lane_route_caches = [
            (
                RouteEvaluationCache(instance, cache_incremental_config)
                if cache_enabled and cache_incremental_config is not None
                else None
            )
            for _ in range(3)
        ]
    incumbent_route_ledger = (
        _IncumbentRouteLedger() if candidate_control_runtime is not None else None
    )
    native_runtime = (
        NativeKernelRuntime.build(instance, native_kernel_config)
        if native_kernel_config is not None
        else None
    )
    evaluator = _Evaluator(
        instance,
        deadline=legacy_deadline
        if profile is OperatorProfile.STAGE02_CONSTRAINT_GUIDED
        else overall_deadline,
        measurement_trace=measurement_trace,
        lane="legacy",
        screening_config=screening_config,
        negative_screening_cache=negative_screening_cache,
        negative_screening_sequences=negative_screening_sequences,
        cache_incremental_config=cache_incremental_config,
        route_cache=lane_route_caches[0],
        backend=backend,
        batch_size=batch_size,
        disable_cache=disable_cache,
        batch_work_enabled=batch_work_enabled,
        exact_call_controller=exact_call_controller,
        candidate_control_runtime=candidate_control_runtime,
        candidate_transaction_runtime=candidate_transaction_runtime,
        incumbent_route_ledger=incumbent_route_ledger,
        native_runtime=native_runtime,
    )
    quality_evaluator = _Evaluator(
        instance,
        deadline=legacy_deadline
        if profile is OperatorProfile.STAGE02_CONSTRAINT_GUIDED
        else overall_deadline,
        measurement_trace=measurement_trace,
        lane="quality_shadow",
        screening_config=screening_config,
        negative_screening_cache=negative_screening_cache,
        negative_screening_sequences=negative_screening_sequences,
        cache_incremental_config=cache_incremental_config,
        route_cache=lane_route_caches[1],
        backend=backend,
        batch_size=batch_size,
        disable_cache=disable_cache,
        batch_work_enabled=batch_work_enabled,
        exact_call_controller=exact_call_controller,
        candidate_control_runtime=candidate_control_runtime,
        candidate_transaction_runtime=candidate_transaction_runtime,
        incumbent_route_ledger=incumbent_route_ledger,
        native_runtime=native_runtime,
    )
    # The constraint lane is the explicit slice after the legacy/quality lane,
    # not an unbounded second 30-second budget.  Spell out the endpoint so the
    # configured 0.1-second reservation remains auditable in the code path.
    constraint_deadline = legacy_deadline + constraint_lane_budget
    constraint_evaluator = _Evaluator(
        instance,
        deadline=constraint_deadline,
        measurement_trace=measurement_trace,
        lane="constraint_lane",
        screening_config=screening_config,
        negative_screening_cache=negative_screening_cache,
        negative_screening_sequences=negative_screening_sequences,
        cache_incremental_config=cache_incremental_config,
        route_cache=lane_route_caches[2],
        backend=backend,
        batch_size=batch_size,
        disable_cache=disable_cache,
        batch_work_enabled=batch_work_enabled,
        exact_call_controller=exact_call_controller,
        candidate_control_runtime=candidate_control_runtime,
        candidate_transaction_runtime=candidate_transaction_runtime,
        incumbent_route_ledger=incumbent_route_ledger,
        native_runtime=native_runtime,
    )
    try:
        with evaluator.measurement_context(
            lane="initialization", iteration=None, operator="initial_solution"
        ):
            if initial_customer_sequences is None:
                initial_sequences = _construct_initial_solution(instance, evaluator)
            else:
                if candidate_control_runtime is None:
                    raise RuntimeError("inherited initialization requires candidate control")
                candidate_control_runtime.events.append(
                    {
                        "event_type": "candidate_initial_solution",
                        "status": "submitted",
                        "customer_sequences": [
                            list(sequence) for sequence in initial_customer_sequences
                        ],
                        **dict(initial_solution_provenance or {}),
                    }
                )
                selected_initial = evaluator.select_feasible_candidate_plan(
                    (initial_customer_sequences,),
                    current_sequences=(),
                    allow_vehicle_increase=True,
                )
                initial_sequences = selected_initial or ()
            if candidate_control_runtime is not None and initial_customer_sequences is None:
                initial_sequences = _refine_controlled_initial_solution(
                    instance,
                    initial_sequences,
                    evaluator,
                    vehicle_config,
                )
    except _TimeLimitReached:
        return _failed_result(
            started,
            evaluator,
            "time limit reached during initial construction",
            operator_profile=profile.value,
            termination_mode=termination_mode,
        )
    with evaluator.measurement_context(
        lane="initialization", iteration=None, operator="initial_solution"
    ):
        current = evaluator.solution(initial_sequences)
    if initial_customer_sequences is not None and candidate_control_runtime is not None:
        candidate_control_runtime.events.append(
            {
                "event_type": "candidate_initial_solution",
                "status": "verified" if current.feasible else "verification_failed",
                "customer_sequences": [list(sequence) for sequence in initial_sequences],
                "objective_key": current.objective.key if current.objective is not None else (),
                **dict(initial_solution_provenance or {}),
            }
        )
    if not current.feasible:
        return _failed_result(
            started,
            evaluator,
            "no feasible singleton initial solution",
            operator_profile=profile.value,
            termination_mode=termination_mode,
        )
    if current.objective is None:
        raise RuntimeError("feasible ALNS initial solution is missing its objective")
    evaluator.remember_incumbent(current)
    quality_evaluator.remember_incumbent(current)
    constraint_evaluator.remember_incumbent(current)

    initial_routes = tuple(result.route for result in current.charging)
    initial_customer_sequences = current.sequences
    initial_objective = current.objective
    best = current
    quality_probe_current = current
    quality_probe_best = current
    constraint_lane_current = current
    constraint_lane_best = current
    first_feasible_time = time.perf_counter() - started
    best_time = first_feasible_time
    destroy_stats = {name: OperatorStatistics() for name in ("random", "worst", "related")}
    standard_repair_stats = {name: OperatorStatistics() for name in ("greedy", "regret2", "energy")}
    repair_stats = dict(standard_repair_stats)
    if profile in (
        OperatorProfile.STAGE02_ROUTE_REDUCTION,
        OperatorProfile.STAGE02_ROUTE_QUALITY,
        OperatorProfile.STAGE02_CONSTRAINT_GUIDED,
    ):
        repair_stats["vehicle_count_aware"] = OperatorStatistics()
    neighborhood_names = _neighborhood_names(profile)
    neighborhood_stats = (
        {name: OperatorStatistics() for name in neighborhood_names}
        if profile is not OperatorProfile.BASELINE
        else {}
    )
    refinement_stats = OperatorStatistics()
    if profile is OperatorProfile.STAGE02_CONSTRAINT_GUIDED:
        neighborhood_stats["vehicle_reduction_refinement"] = refinement_stats
    neighborhood_events = _NeighborhoodEventStream(neighborhood_event_sink)
    accepted = 0
    improved = 0
    rejected = 0
    completed_iterations = 0
    effective_iterations = 0
    last_effective_iteration_completed_at_seconds: float | None = None
    stagnation_iterations = 0
    maximum_stagnation = 0
    global_best_reset_pending = False
    removal_tier_counts = {tier.value: 0 for tier in RemovalTier}

    # ── Stage 4 adaptive weights and search control ──────────────
    stage04_enabled = stage04_config is not None and stage04_config.enabled
    stage04_events: list[dict[str, object]] = []
    temperature_history: list[tuple[int, float]] = []
    reheat_count = 0
    restart_count = 0
    reheat_floor = 0.0
    intensification_active = False
    intensification_remaining = 0
    acceptance_window: list[bool] = []

    if stage04_enabled and stage04_config is not None:
        if stage04_config.auto_temperature:
            initial_temperature = _estimate_initial_temperature(
                instance,
                current,
                evaluator,
                rng,
                stage04_config,
            )
        else:
            initial_temperature = max(
                1.0, current.objective.total_distance * stage04_config.temperature_fallback_fraction
            )
        if not stage04_config.fixed_weights:
            stage04_events.append(
                {
                    "type": "stage04_config",
                    "segment_length": stage04_config.segment_length,
                    "min_calls_per_operator": stage04_config.min_calls_per_operator,
                    "auto_temperature": stage04_config.auto_temperature,
                    "initial_temperature": initial_temperature,
                    "reheat_enabled": stage04_config.reheat_enabled,
                    "restart_enabled": stage04_config.restart_enabled,
                    "intensification_enabled": stage04_config.intensification_enabled,
                    "fixed_weights": stage04_config.fixed_weights,
                }
            )
            temperature_history.append((0, initial_temperature))
    else:
        initial_temperature = max(1.0, current.objective.total_distance * 0.05)

    watchdog_triggered = False
    candidate_exhausted = False
    no_exact_rounds = 0
    iteration_numbers: Iterator[int] = (
        iter(range(max_iterations)) if max_iterations is not None else itertools.count()
    )
    for iteration in iteration_numbers:
        if exact_call_controller is not None and exact_call_controller.budget_reached:
            evaluator._record_exact_budget_boundary()
            break
        elapsed = time.perf_counter() - started
        if elapsed >= time_limit_seconds:
            watchdog_triggered = termination_mode == "fixed_work" or (
                exact_call_controller is not None
                and exact_call_controller.config.mode == "exact_call_budget"
            )
            break
        if candidate_control_runtime is not None:
            candidate_control_runtime.begin_round(iteration)
        round_started_calls = (
            exact_call_controller.started_calls if exact_call_controller is not None else 0
        )
        completed_iterations = iteration + 1
        destroy_name = ""
        repair_name = ""
        selected_neighborhood = ""
        move_events: tuple[NeighborhoodEvent, ...] = ()
        shadow_neighborhood = ""
        shadow_events: tuple[NeighborhoodEvent, ...] = ()
        shadow_candidate: _EvaluatedSolution | None = None
        global_best_improved = False
        main_global_best_improved = False
        main_lane_timed_out = False
        refinement_selected = False
        if profile is OperatorProfile.BASELINE:
            destroy_name = _weighted_choice(rng, destroy_stats)
            repair_name = _weighted_choice(rng, standard_repair_stats)
            destroy_stats[destroy_name].calls += 1
            repair_stats[repair_name].calls += 1
            evaluator.set_measurement_context(
                lane="legacy", iteration=iteration, operator=f"{destroy_name}+{repair_name}"
            )
            if measurement_trace is not None:
                measurement_trace.record_operator_call(
                    lane="legacy",
                    iteration=iteration,
                    operator=destroy_name,
                    statistics_group="destroy_statistics",
                )
                measurement_trace.record_operator_call(
                    lane="legacy",
                    iteration=iteration,
                    operator=repair_name,
                    statistics_group="repair_statistics",
                )

            remove_count = max(1, math.ceil(len(instance.customers) * removal_fraction))
            if len(instance.customers) > 20:
                remove_count = min(remove_count, 3)
            partial, removed = _destroy(
                instance, current.sequences, remove_count, destroy_name, rng
            )
            try:
                candidate_sequences = _repair(
                    partial, removed, repair_name, evaluator, instance, rng
                )
                candidate = evaluator.solution(
                    candidate_sequences,
                    precomputed_routes=_unchanged_precomputed_routes(
                        evaluator, current, candidate_sequences
                    ),
                )
            except _TimeLimitReached:
                destroy_stats[destroy_name].rejected += 1
                repair_stats[repair_name].rejected += 1
                if stage04_enabled and stage04_config is not None:
                    _stage04_accumulate(destroy_stats[destroy_name], stage04_config.reward_rejected)
                    _stage04_accumulate(repair_stats[repair_name], stage04_config.reward_rejected)
                break
        else:
            selected_neighborhood = _select_stage02_neighborhood(
                iteration,
                rng,
                neighborhood_stats,
                include_quality=False,
                include_constraint=False,
            )
            neighborhood_stats[selected_neighborhood].calls += 1
            evaluator.set_measurement_context(
                lane="legacy", iteration=iteration, operator=selected_neighborhood
            )
            if measurement_trace is not None:
                measurement_trace.record_operator_call(
                    lane="legacy",
                    iteration=iteration,
                    operator=selected_neighborhood,
                )
            try:
                if selected_neighborhood == "route_elimination":
                    proposal = propose_route_elimination(
                        instance,
                        current.sequences,
                        evaluator,
                        config=vehicle_config,
                    )
                    candidate_sequences = proposal.sequences or ()
                    move_events = proposal.events
                elif selected_neighborhood == "route_merge":
                    proposal = propose_route_merge(
                        instance,
                        current.sequences,
                        evaluator,
                        config=vehicle_config,
                    )
                    candidate_sequences = proposal.sequences or ()
                    move_events = proposal.events
                elif selected_neighborhood == "relocate":
                    proposal = propose_relocate(
                        instance,
                        current.sequences,
                        evaluator,
                        config=vehicle_config,
                    )
                    candidate_sequences = proposal.sequences or ()
                    move_events = proposal.events
                elif selected_neighborhood == "swap":
                    proposal = propose_swap(
                        instance,
                        current.sequences,
                        evaluator,
                        config=vehicle_config,
                    )
                    candidate_sequences = proposal.sequences or ()
                    move_events = proposal.events
                elif selected_neighborhood == "two_opt_star":
                    proposal = propose_two_opt_star(
                        instance,
                        current.sequences,
                        evaluator,
                        config=vehicle_config,
                    )
                    candidate_sequences = proposal.sequences or ()
                    move_events = proposal.events
                elif selected_neighborhood == "route_segment_destroy":
                    proposal = propose_route_segment_destroy(
                        instance,
                        current.sequences,
                        evaluator,
                        config=vehicle_config,
                    )
                    candidate_sequences = proposal.sequences or ()
                    move_events = proposal.events
                elif selected_neighborhood == "ejection_chain":
                    proposal = propose_ejection_chain(
                        instance,
                        current.sequences,
                        evaluator,
                        config=vehicle_config,
                    )
                    candidate_sequences = proposal.sequences or ()
                    move_events = proposal.events
                else:
                    destroy_name = _weighted_choice(rng, destroy_stats)
                    destroy_stats[destroy_name].calls += 1
                    evaluator.set_measurement_context(
                        lane="legacy", iteration=iteration, operator=destroy_name
                    )
                    if measurement_trace is not None:
                        measurement_trace.record_operator_call(
                            lane="legacy",
                            iteration=iteration,
                            operator=destroy_name,
                            statistics_group="destroy_statistics",
                        )
                    remove_count = max(1, math.ceil(len(instance.customers) * removal_fraction))
                    if len(instance.customers) > 20:
                        remove_count = min(remove_count, 3)
                    partial, removed = _destroy(
                        instance, current.sequences, remove_count, destroy_name, rng
                    )
                    if selected_neighborhood == "vehicle_count_aware_repair":
                        repair_name = "vehicle_count_aware"
                        repair_stats[repair_name].calls += 1
                        evaluator.set_measurement_context(
                            lane="legacy", iteration=iteration, operator=repair_name
                        )
                        if measurement_trace is not None:
                            measurement_trace.record_operator_call(
                                lane="legacy",
                                iteration=iteration,
                                operator=repair_name,
                                statistics_group="repair_statistics",
                            )
                        before_calls = evaluator.calls
                        repair = repair_vehicle_count_aware(
                            partial,
                            removed,
                            evaluator,
                            instance,
                            config=vehicle_config,
                            allow_new_routes=True,
                        )
                        candidate_sequences = repair.sequences or ()
                        move_events = (
                            NeighborhoodEvent(
                                "vehicle_count_aware_repair",
                                "candidate_proposed" if repair.sequences is not None else "failed",
                                repair.failure_reason or "existing_route_repair",
                                removed_customers=removed,
                                candidate_vehicle_delta=(
                                    len(candidate_sequences) - len(current.sequences)
                                    if repair.sequences is not None
                                    else None
                                ),
                                candidate_feasible=repair.sequences is not None,
                                new_routes_created=repair.new_routes_created,
                                exact_route_evaluations=evaluator.calls - before_calls,
                            ),
                        )
                    else:
                        repair_name = _weighted_choice(rng, standard_repair_stats)
                        repair_stats[repair_name].calls += 1
                        evaluator.set_measurement_context(
                            lane="legacy", iteration=iteration, operator=repair_name
                        )
                        if measurement_trace is not None:
                            measurement_trace.record_operator_call(
                                lane="legacy",
                                iteration=iteration,
                                operator=repair_name,
                                statistics_group="repair_statistics",
                            )
                        candidate_sequences = _repair(
                            partial, removed, repair_name, evaluator, instance, rng
                        )
                        move_events = (
                            NeighborhoodEvent(
                                "standard",
                                "proposal",
                                f"{destroy_name}+{repair_name}",
                                removed_customers=removed,
                            ),
                        )
                evaluator.set_measurement_context(
                    lane="legacy", iteration=iteration, operator=selected_neighborhood
                )
                candidate = evaluator.solution(
                    candidate_sequences,
                    precomputed_routes=_unchanged_precomputed_routes(
                        evaluator, current, candidate_sequences
                    ),
                )
            except _TimeLimitReached as error:
                timeout = _deadline_event(
                    selected_neighborhood,
                    "time_limit_reached_during_neighborhood",
                    error,
                    current.sequences,
                )
                if (
                    profile is OperatorProfile.STAGE02_CONSTRAINT_GUIDED
                    and time.perf_counter() < overall_deadline
                ):
                    move_events = (timeout,)
                    candidate = current
                    main_lane_timed_out = True
                else:
                    neighborhood_events.append(_event_record(timeout, iteration))
                    neighborhood_stats[selected_neighborhood].rejected += 1
                    if destroy_name:
                        destroy_stats[destroy_name].rejected += 1
                    if repair_name:
                        repair_stats[repair_name].rejected += 1
                    if stage04_enabled and stage04_config is not None:
                        _stage04_accumulate(
                            neighborhood_stats[selected_neighborhood],
                            stage04_config.reward_rejected,
                        )
                        if destroy_name:
                            _stage04_accumulate(
                                destroy_stats[destroy_name],
                                stage04_config.reward_rejected,
                            )
                        if repair_name:
                            _stage04_accumulate(
                                repair_stats[repair_name],
                                stage04_config.reward_rejected,
                            )
                    break

            _record_neighborhood_proposal(
                neighborhood_stats[selected_neighborhood],
                move_events,
                candidate,
                current,
            )
            if (
                profile is OperatorProfile.STAGE02_CONSTRAINT_GUIDED
                and candidate.feasible
                and candidate.objective is not None
                and current.objective is not None
                and candidate.objective.vehicle_count < current.objective.vehicle_count
                and candidate.objective.vehicle_count
                <= max(1, math.ceil(len(instance.customers) / 5) - 1)
                and len(instance.customers) > 1
                and not main_lane_timed_out
            ):
                candidate_before_refinement = candidate
                refinement_count = max(1, min(3, len(instance.customers) - 1))
                refinement_partial, refinement_removed = _destroy(
                    instance,
                    candidate.sequences,
                    refinement_count,
                    "worst",
                    random.Random(seed ^ 0xA11CE ^ iteration),
                )
                refinement_before_calls = evaluator.calls
                refinement_candidate = _infeasible_solution()
                refinement_reason = ""
                evaluator.set_measurement_context(
                    lane="legacy", iteration=iteration, operator="vehicle_reduction_refinement"
                )
                try:
                    refinement = repair_vehicle_reduction_refinement(
                        refinement_partial,
                        refinement_removed,
                        evaluator,
                        instance,
                        budget=(
                            vehicle_config.vehicle_reduction_refinement_exact_evaluation_budget
                        ),
                        precomputed_routes={
                            sequence: result
                            for sequence, result in zip(
                                candidate_before_refinement.sequences,
                                candidate_before_refinement.charging,
                                strict=True,
                            )
                        },
                    )
                    if refinement.sequences is not None:
                        refinement_candidate = evaluator.solution(
                            refinement.sequences,
                            precomputed_routes=_unchanged_precomputed_routes(
                                evaluator, candidate_before_refinement, refinement.sequences
                            ),
                        )
                    else:
                        refinement_reason = refinement.failure_reason
                except _TimeLimitReached:
                    refinement_reason = "time_limit_reached_during_refinement"
                refinement_comparison = (
                    compare_objectives(
                        refinement_candidate.objective,
                        candidate_before_refinement.objective,
                    )
                    if refinement_candidate.feasible
                    and refinement_candidate.objective is not None
                    and candidate_before_refinement.objective is not None
                    else ObjectiveComparison.WORSE
                )
                if refinement_comparison is ObjectiveComparison.BETTER:
                    candidate = refinement_candidate
                    refinement_selected = True
                    refinement_status = "candidate_proposed"
                    refinement_reason = "vehicle_reduction_refined"
                    refinement_sequences = refinement_candidate.sequences
                else:
                    refinement_status = "failed"
                    refinement_reason = refinement_reason or "refinement_not_better"
                    refinement_sequences = refinement_partial
                refinement_event = NeighborhoodEvent(
                    "vehicle_reduction_refinement",
                    refinement_status,
                    refinement_reason,
                    affected_route_indices=tuple(
                        index
                        for index, (before, after) in enumerate(
                            zip(
                                current.sequences,
                                refinement_sequences,
                                strict=False,
                            )
                        )
                        if before != after
                    ),
                    removed_customers=refinement_removed,
                    candidate_route_sequences=refinement_sequences,
                    candidate_vehicle_delta=(
                        len(refinement_sequences) - len(current.sequences)
                        if refinement_selected
                        else None
                    ),
                    candidate_feasible=refinement_selected,
                    prefilter_passed=bool(refinement_partial),
                    exact_route_evaluations=evaluator.calls - refinement_before_calls,
                    track="legacy",
                )
                move_events = (*move_events, refinement_event)
                refinement_stats.calls += 1
                if not refinement_selected:
                    # A refinement call that did not replace the main candidate
                    # is a rejected operator outcome, even if that main
                    # candidate is later accepted.
                    refinement_stats.rejected += 1
                if measurement_trace is not None:
                    measurement_trace.record_operator_call(
                        lane="legacy",
                        iteration=iteration,
                        operator="vehicle_reduction_refinement",
                    )
                _record_neighborhood_proposal(
                    refinement_stats,
                    (refinement_event,),
                    refinement_candidate,
                    candidate_before_refinement,
                )
                evaluator.set_measurement_context(
                    lane="legacy", iteration=iteration, operator=selected_neighborhood
                )
            if (
                profile
                in (
                    OperatorProfile.STAGE02_ROUTE_QUALITY,
                    OperatorProfile.STAGE02_CONSTRAINT_GUIDED,
                )
                and not main_lane_timed_out
                and not (exact_call_controller is not None and exact_call_controller.budget_reached)
            ):
                shadow_neighborhood = _quality_shadow_neighborhood(iteration)
                if shadow_neighborhood:
                    neighborhood_stats[shadow_neighborhood].calls += 1
                    quality_evaluator.set_measurement_context(
                        lane="quality_shadow",
                        iteration=iteration,
                        operator=shadow_neighborhood,
                    )
                    if measurement_trace is not None:
                        measurement_trace.record_operator_call(
                            lane="quality_shadow",
                            iteration=iteration,
                            operator=shadow_neighborhood,
                        )
                    shadow_current_before = quality_probe_current
                    shadow_global_best_improved = False
                    try:
                        shadow_sequences, shadow_events = _quality_shadow_proposal(
                            shadow_neighborhood,
                            instance,
                            quality_probe_current.sequences,
                            quality_evaluator,
                            vehicle_config,
                            precomputed_routes={
                                sequence: charging
                                for sequence, charging in zip(
                                    quality_probe_current.sequences,
                                    quality_probe_current.charging,
                                    strict=True,
                                )
                            },
                        )
                        shadow_candidate = quality_evaluator.solution(
                            shadow_sequences,
                            precomputed_routes={
                                sequence: charging
                                for sequence, charging in zip(
                                    quality_probe_current.sequences,
                                    quality_probe_current.charging,
                                    strict=True,
                                )
                            },
                        )
                    except _TimeLimitReached as error:
                        shadow_events = (
                            _deadline_event(
                                shadow_neighborhood,
                                "time_limit_reached_during_quality_probe",
                                error,
                                quality_probe_current.sequences,
                            ),
                        )
                        shadow_candidate = _infeasible_solution()
                    _record_neighborhood_proposal(
                        neighborhood_stats[shadow_neighborhood],
                        shadow_events,
                        shadow_candidate,
                        quality_probe_current,
                    )
                    quality_comparison = (
                        compare_objectives(
                            shadow_candidate.objective,
                            quality_probe_current.objective,
                        )
                        if shadow_candidate.feasible
                        and shadow_candidate.objective is not None
                        and quality_probe_current.objective is not None
                        else ObjectiveComparison.WORSE
                    )
                    quality_probe_accept = (
                        shadow_candidate.feasible
                        and quality_comparison is not ObjectiveComparison.WORSE
                        and not (
                            exact_call_controller is not None
                            and exact_call_controller.budget_reached
                        )
                    )
                    if quality_probe_accept and time.perf_counter() >= quality_evaluator.deadline:
                        quality_probe_accept = False
                        if measurement_trace is not None:
                            measurement_trace.record_deadline_boundary(
                                lane="quality_shadow",
                                iteration=iteration,
                                operator=shadow_neighborhood,
                                boundary="before_candidate_commit",
                                reason=(
                                    "quality-shadow candidate transaction reached "
                                    "the lane deadline before commit"
                                ),
                            )
                    quality_probe_vehicle_reduction = bool(
                        shadow_candidate.objective is not None
                        and quality_probe_current.objective is not None
                        and shadow_candidate.objective.vehicle_count
                        < quality_probe_current.objective.vehicle_count
                    )
                    quality_probe_distance_improvement = bool(
                        shadow_candidate.objective is not None
                        and quality_probe_current.objective is not None
                        and shadow_candidate.objective.total_distance
                        < quality_probe_current.objective.total_distance - 1e-9
                    )
                    neighborhood_events.extend(
                        _annotated_event_record(
                            event,
                            iteration=iteration,
                            accepted=quality_probe_accept,
                            vehicle_reduction=quality_probe_vehicle_reduction,
                            distance_improvement=quality_probe_distance_improvement,
                            candidate=shadow_candidate,
                        )
                        for event in shadow_events
                    )
                    shadow_statistics = neighborhood_stats[shadow_neighborhood]
                    if quality_probe_accept and shadow_candidate.objective is not None:
                        shadow_statistics.accepted += 1
                        if quality_comparison is ObjectiveComparison.BETTER:
                            shadow_statistics.improved += 1
                            shadow_statistics.accepted_improving += 1
                        elif quality_comparison is ObjectiveComparison.EQUAL:
                            shadow_statistics.accepted_equal += 1
                        else:
                            shadow_statistics.accepted_worse += 1
                        if quality_probe_vehicle_reduction:
                            shadow_statistics.accepted_vehicle_reductions += 1
                        quality_probe_current = shadow_candidate
                        quality_evaluator.remember_incumbent(quality_probe_current)
                        if (
                            quality_probe_best.objective is None
                            or compare_objectives(
                                shadow_candidate.objective,
                                quality_probe_best.objective,
                            )
                            is ObjectiveComparison.BETTER
                        ):
                            quality_probe_best = shadow_candidate
                            if (
                                best.objective is None
                                or compare_objectives(
                                    shadow_candidate.objective,
                                    best.objective,
                                )
                                is ObjectiveComparison.BETTER
                            ):
                                best = shadow_candidate
                                best_time = time.perf_counter() - started
                                global_best_improved = True
                                shadow_global_best_improved = True
                                shadow_statistics.best += 1
                        if stage04_enabled and stage04_config is not None:
                            reward = stage04_config.reward_for(
                                accepted=True,
                                comparison=(
                                    "better"
                                    if quality_comparison is ObjectiveComparison.BETTER
                                    else "equal"
                                    if quality_comparison is ObjectiveComparison.EQUAL
                                    else "worse"
                                ),
                                is_global_best=shadow_global_best_improved,
                                vehicle_reduction=quality_probe_vehicle_reduction,
                            )
                            _stage04_accumulate(shadow_statistics, reward)
                        else:
                            reward = (
                                8.0
                                if shadow_global_best_improved
                                else (
                                    4.0 if quality_comparison is ObjectiveComparison.BETTER else 1.0
                                )
                            )
                            _update_weight(shadow_statistics, reward)
                    else:
                        shadow_statistics.rejected += 1
                        if stage04_enabled and stage04_config is not None:
                            _stage04_accumulate(shadow_statistics, stage04_config.reward_rejected)
                        else:
                            _update_weight(shadow_statistics, 0.0)
                    if measurement_trace is not None and shadow_candidate is not None:
                        measurement_trace.record_candidate_state(
                            lane="quality_shadow",
                            iteration=iteration,
                            operator=shadow_neighborhood,
                            current_sequences=shadow_current_before.sequences,
                            candidate_sequences=shadow_candidate.sequences,
                            candidate_full_routes=_evaluated_full_routes(shadow_candidate),
                            current_objective_key=(
                                shadow_current_before.objective.key
                                if shadow_current_before.objective is not None
                                else ()
                            ),
                            candidate_objective_key=(
                                shadow_candidate.objective.key
                                if shadow_candidate.objective is not None
                                else ()
                            ),
                            candidate_feasible=shadow_candidate.feasible,
                            accepted=quality_probe_accept,
                            global_best=shadow_global_best_improved,
                            status=(
                                "accepted"
                                if quality_probe_accept
                                else "time_limit"
                                if any(event.status == "time_limit" for event in shadow_events)
                                else "rejected"
                            ),
                            reason="quality shadow probe",
                        )

            if (
                profile is OperatorProfile.STAGE02_CONSTRAINT_GUIDED
                and (
                    iteration < len(_CONSTRAINT_REMOVAL_ORDER)
                    or iteration % vehicle_config.exploration_period == 0
                )
                and not (exact_call_controller is not None and exact_call_controller.budget_reached)
            ):
                constraint_operator = _select_constraint_operator(
                    iteration,
                    constraint_rng,
                    {name: neighborhood_stats[name] for name in _CONSTRAINT_REMOVAL_ORDER},
                )
                constraint_statistics = neighborhood_stats[constraint_operator]
                constraint_statistics.calls += 1
                constraint_evaluator.set_measurement_context(
                    lane="constraint_lane",
                    iteration=iteration,
                    operator=constraint_operator,
                )
                if measurement_trace is not None:
                    measurement_trace.record_operator_call(
                        lane="constraint_lane",
                        iteration=iteration,
                        operator=constraint_operator,
                    )
                selection = select_dynamic_removal_size(
                    len(instance.customers),
                    stagnation_iterations,
                    iteration,
                    config=vehicle_config,
                    global_best_reset=global_best_reset_pending,
                )
                removal_tier_counts[selection.tier.value] += 1
                constraint_current_before = constraint_lane_current
                constraint_global_best_improved = False
                try:
                    constraint_candidate, constraint_events = _constraint_lane_step(
                        instance,
                        constraint_lane_current,
                        constraint_evaluator,
                        vehicle_config,
                        operator=constraint_operator,
                        selection=selection,
                        seed=constraint_rng.randrange(2**32),
                    )
                except _TimeLimitReached as error:
                    constraint_events = (
                        replace(
                            _deadline_event(
                                constraint_operator,
                                "time_limit_reached_during_constraint_probe",
                                error,
                                constraint_lane_current.sequences,
                            ),
                            track="constraint_lane",
                            constraint_category=constraint_operator,
                            removal_tier="",
                            removal_size_requested=0,
                            removal_size_actual=0,
                            stagnation_iterations=selection.stagnation_iterations,
                            removal_trigger=(
                                "probe_not_started:"
                                f"tier={selection.tier.value};"
                                f"requested={selection.requested_count};"
                                f"{selection.trigger_reason}"
                            ),
                            reset_observed=selection.reset_observed,
                        ),
                    )
                    constraint_candidate = _infeasible_solution()

                _record_neighborhood_proposal(
                    constraint_statistics,
                    constraint_events,
                    constraint_candidate,
                    constraint_lane_current,
                )
                lane_comparison = (
                    compare_objectives(
                        constraint_candidate.objective,
                        constraint_lane_current.objective,
                    )
                    if constraint_candidate.feasible
                    and constraint_candidate.objective is not None
                    and constraint_lane_current.objective is not None
                    else ObjectiveComparison.WORSE
                )
                lane_accept = (
                    constraint_candidate.feasible
                    and constraint_candidate.objective is not None
                    and constraint_lane_current.objective is not None
                    and accept_annealing_move(
                        constraint_lane_current.objective,
                        constraint_candidate.objective,
                        temperature=max(
                            1.0,
                            constraint_lane_current.objective.total_distance * 0.05,
                        ),
                        random_draw=0.0,
                    )
                    and not (
                        exact_call_controller is not None and exact_call_controller.budget_reached
                    )
                )
                if lane_accept and time.perf_counter() >= constraint_evaluator.deadline:
                    lane_accept = False
                    if measurement_trace is not None:
                        measurement_trace.record_deadline_boundary(
                            lane="constraint_lane",
                            iteration=iteration,
                            operator=constraint_operator,
                            boundary="before_candidate_commit",
                            reason=(
                                "constraint-lane candidate transaction reached "
                                "the lane deadline before commit"
                            ),
                        )
                lane_vehicle_reduction = bool(
                    constraint_candidate.objective is not None
                    and constraint_lane_current.objective is not None
                    and constraint_candidate.objective.vehicle_count
                    < constraint_lane_current.objective.vehicle_count
                )
                lane_distance_improvement = bool(
                    constraint_candidate.objective is not None
                    and constraint_lane_current.objective is not None
                    and constraint_candidate.objective.total_distance
                    < constraint_lane_current.objective.total_distance - 1e-9
                )
                neighborhood_events.extend(
                    _annotated_event_record(
                        event,
                        iteration=iteration,
                        accepted=lane_accept,
                        vehicle_reduction=lane_vehicle_reduction,
                        distance_improvement=lane_distance_improvement,
                        candidate=constraint_candidate,
                    )
                    for event in constraint_events
                )
                if lane_accept and constraint_candidate.objective is not None:
                    constraint_statistics.accepted += 1
                    constraint_lane_current = constraint_candidate
                    constraint_evaluator.remember_incumbent(constraint_lane_current)
                    if lane_comparison is ObjectiveComparison.BETTER:
                        constraint_statistics.improved += 1
                        constraint_statistics.accepted_improving += 1
                    elif lane_comparison is ObjectiveComparison.EQUAL:
                        constraint_statistics.accepted_equal += 1
                    else:
                        constraint_statistics.accepted_worse += 1
                    if lane_vehicle_reduction:
                        constraint_statistics.accepted_vehicle_reductions += 1
                    if (
                        constraint_lane_best.objective is None
                        or compare_objectives(
                            constraint_candidate.objective,
                            constraint_lane_best.objective,
                        )
                        is ObjectiveComparison.BETTER
                    ):
                        constraint_lane_best = constraint_candidate
                    if (
                        best.objective is None
                        or compare_objectives(
                            constraint_candidate.objective,
                            best.objective,
                        )
                        is ObjectiveComparison.BETTER
                    ):
                        best = constraint_candidate
                        best_time = time.perf_counter() - started
                        global_best_improved = True
                        constraint_global_best_improved = True
                        constraint_statistics.best += 1
                else:
                    constraint_statistics.rejected += 1
                    constraint_statistics.failure_reasons["candidate_rejected"] = (
                        constraint_statistics.failure_reasons.get("candidate_rejected", 0) + 1
                    )
                if stage04_enabled and stage04_config is not None:
                    constraint_reward = stage04_config.reward_for(
                        accepted=lane_accept,
                        comparison=(
                            "better"
                            if lane_comparison is ObjectiveComparison.BETTER
                            else "equal"
                            if lane_comparison is ObjectiveComparison.EQUAL
                            else "worse"
                        ),
                        is_global_best=constraint_global_best_improved,
                        vehicle_reduction=lane_vehicle_reduction and lane_accept,
                    )
                    _stage04_accumulate(constraint_statistics, constraint_reward)
                if measurement_trace is not None:
                    measurement_trace.record_candidate_state(
                        lane="constraint_lane",
                        iteration=iteration,
                        operator=constraint_operator,
                        current_sequences=constraint_current_before.sequences,
                        candidate_sequences=constraint_candidate.sequences,
                        candidate_full_routes=_evaluated_full_routes(constraint_candidate),
                        current_objective_key=(
                            constraint_current_before.objective.key
                            if constraint_current_before.objective is not None
                            else ()
                        ),
                        candidate_objective_key=(
                            constraint_candidate.objective.key
                            if constraint_candidate.objective is not None
                            else ()
                        ),
                        candidate_feasible=constraint_candidate.feasible,
                        accepted=lane_accept,
                        global_best=constraint_global_best_improved,
                        status=(
                            "accepted"
                            if lane_accept
                            else "time_limit"
                            if any(event.status == "time_limit" for event in constraint_events)
                            else "rejected"
                        ),
                        reason="constraint lane probe",
                    )

        if exact_call_controller is not None and exact_call_controller.budget_reached:
            evaluator._record_exact_budget_boundary()
            reward_rejected = (
                stage04_config.reward_rejected
                if stage04_enabled and stage04_config is not None
                else 0.0
            )
            if profile is OperatorProfile.BASELINE:
                destroy_stats[destroy_name].rejected += 1
                repair_stats[repair_name].rejected += 1
                if stage04_enabled and stage04_config is not None:
                    _stage04_accumulate(destroy_stats[destroy_name], reward_rejected)
                    _stage04_accumulate(repair_stats[repair_name], reward_rejected)
            else:
                neighborhood_stats[selected_neighborhood].rejected += 1
                if destroy_name:
                    destroy_stats[destroy_name].rejected += 1
                if repair_name:
                    repair_stats[repair_name].rejected += 1
                if stage04_enabled and stage04_config is not None:
                    _stage04_accumulate(neighborhood_stats[selected_neighborhood], reward_rejected)
                    if destroy_name:
                        _stage04_accumulate(destroy_stats[destroy_name], reward_rejected)
                    if repair_name:
                        _stage04_accumulate(repair_stats[repair_name], reward_rejected)
            break

        previous_current = current
        cooling_progress = (
            iteration / max_iterations
            if max_iterations is not None
            else min(1.0, (time.perf_counter() - started) / time_limit_seconds)
        )
        if stage04_enabled:
            cooled = initial_temperature * max(0.001, 1.0 - cooling_progress)
            reheat_floor *= 0.99
            temperature = max(cooled, reheat_floor)
            if iteration % 10 == 0:
                temperature_history.append((iteration, temperature))
        else:
            temperature = initial_temperature * max(0.001, 1.0 - cooling_progress)
        quality_candidate_is_worse = (
            profile
            in (
                OperatorProfile.STAGE02_ROUTE_QUALITY,
                OperatorProfile.STAGE02_CONSTRAINT_GUIDED,
            )
            and selected_neighborhood in _QUALITY_NEIGHBORHOODS
            and candidate.objective is not None
            and compare_objectives(candidate.objective, current.objective)
            is ObjectiveComparison.WORSE
        )
        accept = (
            False
            if main_lane_timed_out or quality_candidate_is_worse
            else (
                candidate.feasible
                and candidate.objective is not None
                and accept_annealing_move(
                    current.objective,
                    candidate.objective,
                    temperature=max(temperature, 1e-12),
                    random_draw=rng.random(),
                )
            )
        )
        if accept and time.perf_counter() >= overall_deadline:
            accept = False
            main_lane_timed_out = True
            if measurement_trace is not None:
                measurement_trace.record_deadline_boundary(
                    lane="legacy",
                    iteration=iteration,
                    operator=(
                        selected_neighborhood
                        if selected_neighborhood
                        else f"{destroy_name}+{repair_name}"
                    ),
                    boundary="before_candidate_commit",
                    reason=(
                        "main candidate transaction reached the overall deadline before commit"
                    ),
                )
        if profile is not OperatorProfile.BASELINE:
            vehicle_reduction = bool(
                candidate.objective is not None
                and current.objective is not None
                and candidate.objective.vehicle_count < current.objective.vehicle_count
            )
            distance_improvement = bool(
                candidate.objective is not None
                and current.objective is not None
                and candidate.objective.total_distance < current.objective.total_distance - 1e-9
            )
            neighborhood_events.extend(
                _annotated_event_record(
                    event,
                    iteration=iteration,
                    accepted=accept,
                    vehicle_reduction=vehicle_reduction,
                    distance_improvement=distance_improvement,
                    candidate=candidate,
                )
                for event in move_events
            )
        if not accept:
            rejected += 1
            if refinement_selected:
                refinement_stats.rejected += 1
            reward_rejected = (
                stage04_config.reward_rejected
                if stage04_enabled and stage04_config is not None
                else 0.0
            )
            if profile is OperatorProfile.BASELINE:
                destroy_stats[destroy_name].rejected += 1
                repair_stats[repair_name].rejected += 1
                if stage04_enabled and stage04_config is not None:
                    _stage04_accumulate(destroy_stats[destroy_name], reward_rejected)
                    _stage04_accumulate(repair_stats[repair_name], reward_rejected)
                else:
                    _update_weight(destroy_stats[destroy_name], 0.0)
                    _update_weight(repair_stats[repair_name], 0.0)
            else:
                neighborhood_stats[selected_neighborhood].rejected += 1
                if destroy_name:
                    destroy_stats[destroy_name].rejected += 1
                if repair_name:
                    repair_stats[repair_name].rejected += 1
                if stage04_enabled and stage04_config is not None:
                    _stage04_accumulate(neighborhood_stats[selected_neighborhood], reward_rejected)
                    if destroy_name:
                        _stage04_accumulate(destroy_stats[destroy_name], reward_rejected)
                    if repair_name:
                        _stage04_accumulate(repair_stats[repair_name], reward_rejected)
                else:
                    _update_weight(neighborhood_stats[selected_neighborhood], 0.0)
                    if destroy_name:
                        _update_weight(destroy_stats[destroy_name], 0.0)
                    if repair_name:
                        _update_weight(repair_stats[repair_name], 0.0)
            if global_best_improved:
                stagnation_iterations = 0
                global_best_reset_pending = True
            else:
                stagnation_iterations += 1
                global_best_reset_pending = False
            maximum_stagnation = max(maximum_stagnation, stagnation_iterations)

            # ── Stage 4 segment-based weight update (reject path) ──
            if (
                stage04_enabled
                and stage04_config is not None
                and not stage04_config.fixed_weights
                and (iteration + 1) % stage04_config.segment_length == 0
            ):
                _apply_stage04_segment_update(
                    _stage04_adaptive_weight_statistics(
                        profile,
                        neighborhood_stats,
                        destroy_stats,
                        standard_repair_stats,
                    ),
                    stage04_config,
                    iteration,
                    stage04_events,
                )

            # ── Stage 4 reheating (reject path) ──────────────────
            if (
                stage04_enabled
                and stage04_config is not None
                and stage04_config.reheat_enabled
                and reheat_count < stage04_config.max_reheats
                and stagnation_iterations >= stage04_config.reheat_stagnation_threshold
            ):
                reheat_floor = initial_temperature * stage04_config.reheat_factor
                reheat_count += 1
                stage04_events.append(
                    {
                        "type": "stage04_reheat",
                        "iteration": iteration,
                        "reheat_count": reheat_count,
                        "reheat_floor": reheat_floor,
                        "stagnation_iterations": stagnation_iterations,
                    }
                )
                temperature_history.append((iteration, reheat_floor))

            # ── Stage 4 stagnation restart (reject path) ─────────
            if (
                stage04_enabled
                and stage04_config is not None
                and stage04_config.restart_enabled
                and restart_count < stage04_config.max_restarts
                and stagnation_iterations >= stage04_config.restart_stagnation_threshold
            ):
                current = best
                evaluator.remember_incumbent(current)
                stagnation_iterations = 0
                restart_count += 1
                reheat_floor = initial_temperature * stage04_config.reheat_factor
                if stage04_config.intensification_enabled and not intensification_active:
                    intensification_active = True
                    intensification_remaining = stage04_config.intensification_iterations
                stage04_events.append(
                    {
                        "type": "stage04_restart",
                        "iteration": iteration,
                        "restart_count": restart_count,
                        "intensification": intensification_active,
                        "stagnation_at_trigger": stagnation_iterations,
                    }
                )

            # ── Stage 4 acceptance-rate tracking (reject path) ────
            if stage04_enabled:
                acceptance_window.append(False)
                if len(acceptance_window) > 50:
                    acceptance_window.pop(0)

            effective_iterations += 1
            if measurement_trace is not None:
                measurement_trace.record_candidate_state(
                    lane="legacy",
                    iteration=iteration,
                    operator=(
                        selected_neighborhood
                        if selected_neighborhood
                        else f"{destroy_name}+{repair_name}"
                    ),
                    current_sequences=previous_current.sequences,
                    candidate_sequences=candidate.sequences,
                    candidate_full_routes=_evaluated_full_routes(candidate),
                    current_objective_key=(
                        previous_current.objective.key
                        if previous_current.objective is not None
                        else ()
                    ),
                    candidate_objective_key=(
                        candidate.objective.key if candidate.objective is not None else ()
                    ),
                    candidate_feasible=candidate.feasible,
                    accepted=False,
                    global_best=main_global_best_improved,
                    status="rejected" if not main_lane_timed_out else "time_limit",
                    reason=(
                        "main lane deadline reached"
                        if main_lane_timed_out
                        else "candidate rejected by lexicographic acceptance"
                    ),
                )
            continue

        accepted += 1
        if candidate.objective is None:
            raise RuntimeError("accepted ALNS candidate is missing its objective")
        if current.objective is None:
            raise RuntimeError("current ALNS solution is missing its objective")
        comparison_result = compare_objectives(candidate.objective, current.objective)
        is_better = comparison_result is ObjectiveComparison.BETTER
        is_equal = comparison_result is ObjectiveComparison.EQUAL
        is_worse = comparison_result is ObjectiveComparison.WORSE
        if refinement_selected:
            refinement_stats.accepted += 1
            if is_better:
                refinement_stats.improved += 1
                refinement_stats.accepted_improving += 1
            elif is_equal:
                refinement_stats.accepted_equal += 1
            elif is_worse:
                refinement_stats.accepted_worse += 1
        veh_reduction_accept = bool(
            candidate.objective is not None
            and current.objective is not None
            and candidate.objective.vehicle_count < current.objective.vehicle_count
        )
        if profile is OperatorProfile.BASELINE:
            destroy_stats[destroy_name].accepted += 1
            repair_stats[repair_name].accepted += 1
            if is_better:
                destroy_stats[destroy_name].improved += 1
                destroy_stats[destroy_name].accepted_improving += 1
                repair_stats[repair_name].improved += 1
                repair_stats[repair_name].accepted_improving += 1
            elif is_equal:
                destroy_stats[destroy_name].accepted_equal += 1
                repair_stats[repair_name].accepted_equal += 1
            elif is_worse:
                destroy_stats[destroy_name].accepted_worse += 1
                repair_stats[repair_name].accepted_worse += 1
        else:
            neighborhood_stats[selected_neighborhood].accepted += 1
            if is_better:
                neighborhood_stats[selected_neighborhood].improved += 1
                neighborhood_stats[selected_neighborhood].accepted_improving += 1
            elif is_equal:
                neighborhood_stats[selected_neighborhood].accepted_equal += 1
            elif is_worse:
                neighborhood_stats[selected_neighborhood].accepted_worse += 1
            if not (stage04_enabled and stage04_config is not None):
                _update_weight(neighborhood_stats[selected_neighborhood], 1.0)
            if destroy_name:
                destroy_stats[destroy_name].accepted += 1
                if is_better:
                    destroy_stats[destroy_name].accepted_improving += 1
                elif is_equal:
                    destroy_stats[destroy_name].accepted_equal += 1
                elif is_worse:
                    destroy_stats[destroy_name].accepted_worse += 1
            if repair_name:
                repair_stats[repair_name].accepted += 1
                if is_better:
                    repair_stats[repair_name].accepted_improving += 1
                elif is_equal:
                    repair_stats[repair_name].accepted_equal += 1
                elif is_worse:
                    repair_stats[repair_name].accepted_worse += 1
        if veh_reduction_accept:
            if profile is OperatorProfile.BASELINE:
                destroy_stats[destroy_name].accepted_vehicle_reductions += 1
                repair_stats[repair_name].accepted_vehicle_reductions += 1
            else:
                neighborhood_stats[selected_neighborhood].accepted_vehicle_reductions += 1
                if destroy_name:
                    destroy_stats[destroy_name].accepted_vehicle_reductions += 1
                if repair_name:
                    repair_stats[repair_name].accepted_vehicle_reductions += 1
            if refinement_selected:
                refinement_stats.accepted_vehicle_reductions += 1
        if is_better:
            improved += 1
        current = candidate
        evaluator.remember_incumbent(current)
        if best.objective is None:
            raise RuntimeError("feasible ALNS incumbent is missing its objective")
        is_new_global_best = (
            compare_objectives(candidate.objective, best.objective) is ObjectiveComparison.BETTER
        )
        if is_new_global_best:
            best = candidate
            best_time = time.perf_counter() - started
            global_best_improved = True
            main_global_best_improved = True
            if profile is OperatorProfile.BASELINE:
                destroy_stats[destroy_name].best += 1
                repair_stats[repair_name].best += 1
            else:
                neighborhood_stats[selected_neighborhood].best += 1
                if refinement_selected:
                    refinement_stats.best += 1
                if destroy_name:
                    destroy_stats[destroy_name].best += 1
                if repair_name:
                    repair_stats[repair_name].best += 1

        # ── Differentiated reward computation ────────────────────
        if stage04_enabled and stage04_config is not None:
            comparison_str = "better" if is_better else ("equal" if is_equal else "worse")
            reward = stage04_config.reward_for(
                accepted=True,
                comparison=comparison_str,
                is_global_best=is_new_global_best,
                vehicle_reduction=veh_reduction_accept,
            )
        else:
            reward = 1.0
            if is_better:
                reward = 4.0
            if is_new_global_best:
                reward = 8.0

        # ── Weight feedback ──────────────────────────────────────
        if stage04_enabled and stage04_config is not None:
            if profile is OperatorProfile.BASELINE:
                _stage04_accumulate(destroy_stats[destroy_name], reward)
                _stage04_accumulate(repair_stats[repair_name], reward)
            else:
                _stage04_accumulate(neighborhood_stats[selected_neighborhood], reward)
                if destroy_name:
                    _stage04_accumulate(destroy_stats[destroy_name], reward)
                if repair_name:
                    _stage04_accumulate(repair_stats[repair_name], reward)
        else:
            if profile is OperatorProfile.BASELINE:
                _update_weight(destroy_stats[destroy_name], reward)
                _update_weight(repair_stats[repair_name], reward)
            else:
                _update_weight(neighborhood_stats[selected_neighborhood], reward)
                if destroy_name:
                    _update_weight(destroy_stats[destroy_name], reward)
                if repair_name:
                    _update_weight(repair_stats[repair_name], reward)

        if measurement_trace is not None:
            measurement_trace.record_candidate_state(
                lane="legacy",
                iteration=iteration,
                operator=(
                    selected_neighborhood
                    if selected_neighborhood
                    else f"{destroy_name}+{repair_name}"
                ),
                current_sequences=previous_current.sequences,
                candidate_sequences=candidate.sequences,
                candidate_full_routes=_evaluated_full_routes(candidate),
                current_objective_key=(
                    previous_current.objective.key if previous_current.objective is not None else ()
                ),
                candidate_objective_key=(
                    candidate.objective.key if candidate.objective is not None else ()
                ),
                candidate_feasible=candidate.feasible,
                accepted=True,
                global_best=main_global_best_improved,
                status="accepted",
            )

        if global_best_improved:
            stagnation_iterations = 0
            global_best_reset_pending = True
        else:
            stagnation_iterations += 1
            global_best_reset_pending = False
        maximum_stagnation = max(maximum_stagnation, stagnation_iterations)

        # ── Stage 4 segment-based weight update ───────────────────
        if (
            stage04_enabled
            and stage04_config is not None
            and not stage04_config.fixed_weights
            and (iteration + 1) % stage04_config.segment_length == 0
        ):
            _apply_stage04_segment_update(
                _stage04_adaptive_weight_statistics(
                    profile,
                    neighborhood_stats,
                    destroy_stats,
                    standard_repair_stats,
                ),
                stage04_config,
                iteration,
                stage04_events,
            )

        # ── Stage 4 reheating ────────────────────────────────────
        if (
            stage04_enabled
            and stage04_config is not None
            and stage04_config.reheat_enabled
            and reheat_count < stage04_config.max_reheats
            and stagnation_iterations >= stage04_config.reheat_stagnation_threshold
        ):
            reheat_floor = initial_temperature * stage04_config.reheat_factor
            reheat_count += 1
            stage04_events.append(
                {
                    "type": "stage04_reheat",
                    "iteration": iteration,
                    "reheat_count": reheat_count,
                    "reheat_floor": reheat_floor,
                    "stagnation_iterations": stagnation_iterations,
                }
            )
            temperature_history.append((iteration, reheat_floor))

        # ── Stage 4 stagnation restart ───────────────────────────
        if (
            stage04_enabled
            and stage04_config is not None
            and stage04_config.restart_enabled
            and restart_count < stage04_config.max_restarts
            and stagnation_iterations >= stage04_config.restart_stagnation_threshold
        ):
            current = best
            evaluator.remember_incumbent(current)
            stagnation_iterations = 0
            restart_count += 1
            reheat_floor = initial_temperature * stage04_config.reheat_factor
            if stage04_config.intensification_enabled and not intensification_active:
                intensification_active = True
                intensification_remaining = stage04_config.intensification_iterations
            stage04_events.append(
                {
                    "type": "stage04_restart",
                    "iteration": iteration,
                    "restart_count": restart_count,
                    "intensification": intensification_active,
                    "stagnation_at_trigger": stagnation_iterations,
                }
            )

        # ── Stage 4 incumbent intensification ────────────────────
        if intensification_active:
            if intensification_remaining > 0:
                intensification_remaining -= 1
            else:
                intensification_active = False
                stage04_events.append(
                    {
                        "type": "stage04_intensification_end",
                        "iteration": iteration,
                    }
                )

        # ── Stage 4 acceptance-rate tracking ─────────────────────
        if stage04_enabled:
            acceptance_window.append(accept)
            if len(acceptance_window) > 50:
                acceptance_window.pop(0)

        effective_iterations += 1
        last_effective_iteration_completed_at_seconds = time.perf_counter() - started
        if (
            candidate_control_runtime is not None
            and fixed_exact_calls
            and exact_call_controller is not None
        ):
            if exact_call_controller.started_calls == round_started_calls:
                no_exact_rounds += 1
            else:
                no_exact_rounds = 0
            if (
                effective_iterations
                >= candidate_control_runtime.config.min_iterations_before_exhaustion
                and no_exact_rounds >= candidate_control_runtime.config.fixed_work_exhaustion_rounds
            ):
                candidate_exhausted = True
                break

    fixed_watchdog_mode = termination_mode == "fixed_work" or (
        exact_call_controller is not None
        and exact_call_controller.config.mode == "exact_call_budget"
    )
    budget_reached = exact_call_controller is not None and exact_call_controller.budget_reached
    if (
        fixed_watchdog_mode
        and max_iterations is not None
        and effective_iterations < max_iterations
        and not budget_reached
        and not candidate_exhausted
    ):
        watchdog_triggered = True

    lane_evaluators = (evaluator, quality_evaluator, constraint_evaluator)
    routes = tuple(result.route for result in best.charging)
    report = validate_routes(instance, [list(route) for route in routes])
    if not report.feasible:
        raise RuntimeError("ALNS best solution failed the unified validator")
    if best.objective is None:
        raise RuntimeError("feasible ALNS result is missing its objective")
    validated_objective = SolutionObjective.from_report(instance, report)
    if compare_objectives(validated_objective, best.objective) is not ObjectiveComparison.EQUAL:
        raise RuntimeError("ALNS solver objective differs from the unified validator objective")
    if candidate_control_runtime is not None:
        candidate_control_runtime.finish_round()
    exact_statistics = exact_call_controller.to_dict() if exact_call_controller is not None else {}
    solve_completed_at = time.perf_counter()
    termination_reason = (
        "exact_call_budget_exhausted"
        if exact_call_controller is not None and exact_call_controller.budget_reached
        else "candidate_control_exhausted"
        if candidate_exhausted
        else "watchdog_exhausted"
        if watchdog_triggered
        else "wall_clock_deadline"
        if solve_completed_at >= overall_deadline
        else "iteration_limit"
    )
    if termination_reason == "wall_clock_deadline" and measurement_trace is not None:
        measurement_trace.record_deadline_boundary(
            lane="solver_finalization",
            iteration=completed_iterations,
            operator="termination",
            boundary="solver_termination",
            reason=("overall wall-clock deadline was confirmed after final solution validation"),
        )
    return ALNSResult(
        feasible=True,
        routes=routes,
        customer_sequences=best.sequences,
        objective=best.objective,
        vehicle_count=len(routes),
        total_energy=sum(result.total_energy for result in best.charging),
        total_charged_energy=sum(result.charged_energy for result in best.charging),
        total_charging_time=sum(result.charging_time for result in best.charging),
        iterations=completed_iterations,
        accepted_moves=accepted,
        improving_moves=improved,
        rejected_moves=rejected,
        first_feasible_time=first_feasible_time,
        best_time=best_time,
        runtime_seconds=solve_completed_at - started,
        charging_subproblem_calls=sum(item.calls for item in lane_evaluators),
        charging_subproblem_time=sum(item.runtime for item in lane_evaluators),
        charging_labels_generated=sum(item.labels_generated for item in lane_evaluators),
        charging_labels_pruned=sum(item.labels_pruned for item in lane_evaluators),
        destroy_statistics={name: stats.to_dict() for name, stats in destroy_stats.items()},
        repair_statistics={name: stats.to_dict() for name, stats in repair_stats.items()},
        operator_profile=profile.value,
        neighborhood_statistics={
            name: stats.to_dict() for name, stats in neighborhood_stats.items()
        },
        neighborhood_events=tuple(neighborhood_events),
        failure_reason="",
        cache_hits=(
            _cache_statistics(lane_route_caches)["cache_hits"]
            if cache_enabled
            else sum(item.cache_hits for item in lane_evaluators)
        ),
        cache_misses=(
            _cache_statistics(lane_route_caches)["cache_misses"]
            if cache_enabled
            else sum(item.calls for item in lane_evaluators)
        ),
        unique_route_evaluations=(
            _completed_unique_route_evaluations(lane_evaluators)
            if exact_call_controller is not None
            else _cache_statistics(lane_route_caches)["unique_route_evaluations"]
            if cache_enabled
            else sum(
                len(item.cache) if item.local_cache_enabled else len(item.evaluated_route_keys)
                for item in lane_evaluators
            )
        ),
        effective_iterations=effective_iterations,
        removal_tier_counts=removal_tier_counts,
        maximum_stagnation=maximum_stagnation,
        constraint_operator_statistics={
            name: neighborhood_stats[name].to_dict()
            for name in _CONSTRAINT_REMOVAL_ORDER
            if name in neighborhood_stats
        },
        screening_statistics=_aggregate_screening_statistics(
            lane_evaluators,
            native_runtime=native_runtime,
        ),
        cache_incremental_statistics=_aggregate_cache_incremental_statistics(
            lane_evaluators,
            cache_incremental_config,
            lane_route_caches,
        ),
        charging_backend=ExactChargingBackend(backend).value,
        batch_size=batch_size,
        backend_metrics=_aggregate_backend_metrics(lane_evaluators),
        termination_mode=termination_mode,
        watchdog_triggered=watchdog_triggered,
        exact_started_calls=(
            exact_call_controller.started_calls if exact_call_controller is not None else 0
        ),
        exact_completed_calls=(
            exact_call_controller.completed_calls if exact_call_controller is not None else 0
        ),
        exact_interrupted_calls=(
            exact_call_controller.interrupted_calls if exact_call_controller is not None else 0
        ),
        exact_budget_exhaustions=(
            exact_call_controller.budget_exhaustions if exact_call_controller is not None else 0
        ),
        termination_reason=termination_reason,
        exact_deadline_statistics=exact_statistics,
        candidate_control_statistics=(
            candidate_control_runtime.statistics() if candidate_control_runtime is not None else {}
        ),
        candidate_transaction_statistics=(
            candidate_transaction_runtime.statistics()
            if candidate_transaction_runtime is not None
            else {}
        ),
        candidate_transaction_events=(
            tuple(candidate_transaction_runtime.events)
            if candidate_transaction_runtime is not None
            else ()
        ),
        candidate_work_hash=(
            candidate_control_runtime.candidate_work_hash
            if candidate_control_runtime is not None
            else ""
        ),
        route_result_hash=(
            candidate_control_runtime.route_result_hash
            if candidate_control_runtime is not None
            else ""
        ),
        stage04_statistics=(
            {
                "enabled": stage04_enabled,
                "reheat_count": reheat_count,
                "restart_count": restart_count,
                "intensification_active": intensification_active,
                "acceptance_rate": (
                    sum(acceptance_window) / len(acceptance_window) if acceptance_window else 0.0
                ),
                "segment_length": (
                    stage04_config.segment_length
                    if stage04_enabled and stage04_config is not None
                    else 0
                ),
                "min_calls_per_operator": (
                    stage04_config.min_calls_per_operator
                    if stage04_enabled and stage04_config is not None
                    else 0
                ),
                "fixed_weights": (
                    stage04_config.fixed_weights
                    if stage04_enabled and stage04_config is not None
                    else False
                ),
                "auto_temperature": (
                    stage04_config.auto_temperature
                    if stage04_enabled and stage04_config is not None
                    else False
                ),
                "initial_temperature": initial_temperature,
                "maximum_stagnation": maximum_stagnation,
            }
            if stage04_enabled
            else {}
        ),
        stage04_weight_history=(
            {
                name: list(stats.weight_history)
                for name, stats in (
                    neighborhood_stats.items()
                    if profile is not OperatorProfile.BASELINE
                    else {**destroy_stats, **repair_stats}.items()
                )
                if stats.weight_history
            }
            if stage04_enabled
            else {}
        ),
        stage04_temperature_history=tuple(temperature_history),
        stage04_event_log=tuple(stage04_events),
        initial_routes=initial_routes,
        initial_customer_sequences=initial_customer_sequences,
        initial_objective=initial_objective,
        iteration_limit_completed_at_seconds=(
            last_effective_iteration_completed_at_seconds
            if termination_reason == "iteration_limit"
            else None
        ),
    )


def solve_alns(
    instance: Instance,
    *,
    seed: int,
    max_iterations: int | None = 2_000,
    time_limit_seconds: float = 60.0,
    removal_fraction: float = 0.2,
    operator_profile: OperatorProfile | str = OperatorProfile.STAGE02_CONSTRAINT_GUIDED,
    vehicle_operator_config: VehicleOperatorConfig | None = None,
    measurement_config: MeasurementConfig | None = None,
    screening_config: CheapScreeningConfig | None = None,
    cache_incremental_config: CacheIncrementalConfig | None = None,
    backend: ExactChargingBackend | str = ExactChargingBackend.CPU_BATCH,
    batch_size: int = 128,
    termination_mode: str = "wall_clock",
    disable_cache: bool = False,
    exact_deadline_config: ExactDeadlineConfig | None = None,
    candidate_control_config: CandidateControlConfig | None = None,
    initial_customer_sequences: tuple[tuple[str, ...], ...] | None = None,
    initial_solution_provenance: Mapping[str, object] | None = None,
    stage04_config: Stage04Config | None = None,
    native_kernel_config: NativeKernelConfig | None = None,
    candidate_transaction_config: NativeCandidateTransactionConfig | None = None,
    neighborhood_event_sink: Callable[[Mapping[str, object]], None] | None = None,
) -> ALNSResult:
    """Solve ALNS with opt-in Stage 3.0--3.4 and Stage 4 evaluation layers.

    ``max_iterations=None`` selects wall-clock-only termination and therefore
    cannot be combined with fixed-work or fixed exact-call semantics.
    """

    if max_iterations is None and (
        termination_mode != "wall_clock"
        or (exact_deadline_config is not None and exact_deadline_config.mode == "exact_call_budget")
    ):
        raise ValueError("max_iterations=None requires wall_clock termination")
    if (
        exact_deadline_config is not None
        and ExactChargingBackend(backend) is not ExactChargingBackend.CPU_BATCH
    ):
        raise ValueError("Stage 3.3 exact deadline requires the cpu_batch backend")
    candidate_control_enabled = (
        candidate_control_config is not None and candidate_control_config.enabled
    )
    if (
        native_kernel_config is not None
        and ExactChargingBackend(backend) is not ExactChargingBackend.CPU_BATCH
    ):
        raise ValueError("native exact charging requires the cpu_batch backend")
    if native_kernel_config is not None and candidate_control_enabled:
        raise ValueError(
            "native kernels cannot be combined with candidate-control workers without "
            "an explicit native worker protocol"
        )
    candidate_transaction_enabled = candidate_transaction_config is not None
    if candidate_transaction_enabled and candidate_control_enabled:
        raise ValueError(
            "Stage 5.2 candidate transactions cannot modify the historical "
            "Stage 3.4 candidate-control path"
        )
    if candidate_transaction_enabled and native_kernel_config is None:
        raise ValueError("Stage 5.2 candidate transactions require native kernels")
    if candidate_transaction_enabled and (screening_config is None or not screening_config.enabled):
        raise ValueError("Stage 5.2 candidate transactions require cheap screening")
    if candidate_transaction_enabled and (
        ExactChargingBackend(backend) is not ExactChargingBackend.CPU_BATCH
    ):
        raise ValueError("Stage 5.2 candidate transactions require cpu_batch")
    if initial_customer_sequences is not None:
        if not candidate_control_enabled:
            raise ValueError("an inherited initial solution requires candidate control")
        supplied_customers = [
            customer for sequence in initial_customer_sequences for customer in sequence
        ]
        expected_customers = sorted(customer.name for customer in instance.customers)
        if sorted(supplied_customers) != expected_customers:
            raise ValueError("inherited initial solution must cover every customer exactly once")
    elif initial_solution_provenance is not None:
        raise ValueError("initial solution provenance requires inherited customer sequences")
    if (
        candidate_control_enabled
        and ExactChargingBackend(backend) is not ExactChargingBackend.CPU_BATCH
    ):
        raise ValueError("Stage 3.4 candidate control requires the cpu_batch backend")
    if (
        stage04_config is not None
        and stage04_config.enabled
        and ExactChargingBackend(backend) is not ExactChargingBackend.CPU_BATCH
    ):
        raise ValueError("Stage 4 adaptive weights requires the cpu_batch backend")
    candidate_control_runtime = (
        CandidateControlRuntime(candidate_control_config)
        if candidate_control_enabled and candidate_control_config is not None
        else None
    )
    exact_call_controller = (
        ExactCallController(exact_deadline_config) if exact_deadline_config is not None else None
    )
    if (
        exact_deadline_config is not None
        and exact_deadline_config.mode == "exact_call_budget"
        and exact_deadline_config.watchdog_seconds is not None
    ):
        time_limit_seconds = exact_deadline_config.watchdog_seconds

    screening_enabled = screening_config is not None and screening_config.enabled
    measurement_enabled = measurement_config is not None and measurement_config.enabled
    cache_incremental_enabled = (
        cache_incremental_config is not None and cache_incremental_config.enabled
    )
    if cache_incremental_enabled and not screening_enabled:
        raise ValueError(
            "Stage 3.2 cache/incremental evaluation requires enabled Stage 3.1 screening"
        )
    if cache_incremental_config is not None and cache_incremental_config.enabled:
        cache_incremental_config = replace(
            cache_incremental_config,
            instance_hash=(
                cache_incremental_config.instance_hash or canonical_instance_hash(instance)
            ),
        )
    if (
        not measurement_enabled
        and not screening_enabled
        and not cache_incremental_enabled
        and exact_deadline_config is None
    ):
        try:
            return _solve_alns(
                instance,
                seed=seed,
                max_iterations=max_iterations,
                time_limit_seconds=time_limit_seconds,
                removal_fraction=removal_fraction,
                operator_profile=operator_profile,
                vehicle_operator_config=vehicle_operator_config,
                screening_config=screening_config,
                cache_incremental_config=cache_incremental_config,
                backend=backend,
                batch_size=batch_size,
                termination_mode=termination_mode,
                disable_cache=disable_cache,
                exact_call_controller=exact_call_controller,
                candidate_control_runtime=candidate_control_runtime,
                initial_customer_sequences=initial_customer_sequences,
                initial_solution_provenance=initial_solution_provenance,
                stage04_config=stage04_config,
                native_kernel_config=native_kernel_config,
                candidate_transaction_config=candidate_transaction_config,
                neighborhood_event_sink=neighborhood_event_sink,
            )
        finally:
            if candidate_control_runtime is not None:
                candidate_control_runtime.close()
    trace_config = (
        measurement_config
        if measurement_config is not None and measurement_config.enabled
        else MeasurementConfig()
    )
    trace = Stage03Trace(
        trace_config,
        screening_config=screening_config,
        cache_incremental_config=(cache_incremental_config if cache_incremental_enabled else None),
        exact_deadline_config=exact_deadline_config,
        candidate_control_config=(candidate_control_config if candidate_control_enabled else None),
    )
    try:
        result = _solve_alns(
            instance,
            seed=seed,
            max_iterations=max_iterations,
            time_limit_seconds=time_limit_seconds,
            removal_fraction=removal_fraction,
            operator_profile=operator_profile,
            vehicle_operator_config=vehicle_operator_config,
            measurement_trace=trace,
            screening_config=screening_config,
            cache_incremental_config=cache_incremental_config,
            backend=backend,
            batch_size=batch_size,
            termination_mode=termination_mode,
            disable_cache=disable_cache,
            exact_call_controller=exact_call_controller,
            candidate_control_runtime=candidate_control_runtime,
            initial_customer_sequences=initial_customer_sequences,
            initial_solution_provenance=initial_solution_provenance,
            stage04_config=stage04_config,
            native_kernel_config=native_kernel_config,
            candidate_transaction_config=candidate_transaction_config,
            neighborhood_event_sink=neighborhood_event_sink,
        )
    except BaseException as error:
        if candidate_control_runtime is not None:
            candidate_control_runtime.close(cancel_futures=True, wait=False)
            _append_candidate_control_events(trace, candidate_control_runtime)
        trace.record_execution_error(error)
        trace.finish()
        raise Stage03ExecutionError(trace, error) from error
    if candidate_control_runtime is not None:
        candidate_control_runtime.close()
        _append_candidate_control_events(trace, candidate_control_runtime)
    trace.finish(result)
    return replace(result, measurement_trace=trace)


def _append_candidate_control_events(
    trace: Stage03Trace,
    runtime: CandidateControlRuntime,
) -> None:
    for runtime_event in runtime.events:
        event = dict(runtime_event)
        sequence = event.pop("customer_sequence", None)
        if isinstance(sequence, list):
            event["route_keys"] = (trace.register_route(tuple(str(name) for name in sequence)),)
        sequences = event.pop("customer_sequences", None)
        if isinstance(sequences, list):
            event["route_keys"] = tuple(
                trace.register_route(tuple(str(name) for name in item))
                for item in sequences
                if isinstance(item, list)
            )
        trace.events.append(event)


def _construct_initial_solution(
    instance: Instance,
    evaluator: _Evaluator,
) -> tuple[tuple[str, ...], ...]:
    sequences: list[tuple[str, ...]] = []
    customers = sorted(
        instance.customers,
        key=lambda node: (node.due_date, node.ready_time, node.name),
    )
    if len(customers) > 20:
        return _construct_large_initial_solution(instance, customers, evaluator)
    for customer in customers:
        best: tuple[float, int, tuple[str, ...]] | None = None
        if (
            evaluator.backend is ExactChargingBackend.CPU_SCALAR
            and not evaluator.batch_work_enabled
        ):
            for route_index in range(len(sequences) + 1):
                base = sequences[route_index] if route_index < len(sequences) else ()
                for position in range(len(base) + 1):
                    candidate = (*base[:position], customer.name, *base[position:])
                    result = evaluator.route(candidate)
                    if not result.feasible:
                        continue
                    old_distance = evaluator.route(base).distance if base else 0.0
                    key = (result.distance - old_distance, route_index, candidate)
                    if best is None or key < best:
                        best = key
            if best is None:
                return ()
            _, route_index, sequence = best
            if route_index == len(sequences):
                sequences.append(sequence)
            else:
                sequences[route_index] = sequence
            continue
        candidate_metadata: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []
        for route_index in range(len(sequences) + 1):
            base = sequences[route_index] if route_index < len(sequences) else ()
            for position in range(len(base) + 1):
                candidate = (*base[:position], customer.name, *base[position:])
                candidate_metadata.append((route_index, base, candidate))
        candidate_results = evaluator.route_batch(
            tuple(candidate for _, _, candidate in candidate_metadata),
            route_change_status="changed",
            candidate_pool=True,
        )
        nonempty_bases = [
            (route_index, base)
            for route_index, base, _ in candidate_metadata
            if base and route_index < len(sequences)
        ]
        base_results = _repair_base_route_results(
            evaluator,
            tuple(base for _, base in nonempty_bases),
        )
        old_distances = {
            route_index: result.distance
            for (route_index, _), result in zip(nonempty_bases, base_results, strict=True)
        }
        for (route_index, _base, candidate), result in zip(
            candidate_metadata,
            candidate_results,
            strict=True,
        ):
            if not result.feasible:
                continue
            old_distance = old_distances.get(route_index, 0.0)
            key = (result.distance - old_distance, route_index, candidate)
            if best is None or key < best:
                best = key
        if best is None:
            return ()
        _, route_index, sequence = best
        if route_index == len(sequences):
            sequences.append(sequence)
        else:
            sequences[route_index] = sequence
    return tuple(sequences)


def _refine_controlled_initial_solution(
    instance: Instance,
    sequences: tuple[tuple[str, ...], ...],
    evaluator: _Evaluator,
    vehicle_config: VehicleOperatorConfig,
) -> tuple[tuple[str, ...], ...]:
    """Greedily consume ranked feasible route merges before ALNS sampling."""

    current = sequences
    for _ in range(len(sequences)):
        proposal = propose_route_merge(
            instance,
            current,
            evaluator,
            config=vehicle_config,
        )
        if proposal.sequences is None or len(proposal.sequences) >= len(current):
            break
        candidate = evaluator.solution(proposal.sequences)
        if not candidate.feasible:
            break
        current = candidate.sequences
    return current


def _construct_large_initial_solution(
    instance: Instance,
    customers: list[Node],
    evaluator: _Evaluator,
) -> tuple[tuple[str, ...], ...]:
    from evrptw.baselines.ortools_vrptw import solve_vrptw

    minimum_fleet = math.ceil(
        sum(customer.demand for customer in customers) / instance.vehicle.load_capacity
    )
    constructor = solve_vrptw(
        instance,
        vehicle_count=min(
            len(customers),
            max(minimum_fleet + 5, math.ceil(len(customers) / 4)),
        ),
        time_limit_seconds=1,
        first_solution_strategy="PARALLEL_CHEAPEST_INSERTION",
    )
    routes = constructor["routes"]
    if routes:
        sequences: list[tuple[str, ...]] = []
        for route in routes:
            customer_sequence = tuple(name for name in route if name != instance.depot.name)
            split = _split_until_charging_feasible(customer_sequence, evaluator)
            if not split:
                return ()
            sequences.extend(split)
        return tuple(sequences)

    return _construct_sequential_initial_solution(customers, evaluator)


def _construct_sequential_initial_solution(
    customers: list[Node],
    evaluator: _Evaluator,
) -> tuple[tuple[str, ...], ...]:
    sequences: list[tuple[str, ...]] = []
    current: tuple[str, ...] = ()
    for customer in customers:
        if (
            evaluator.backend is ExactChargingBackend.CPU_SCALAR
            and not evaluator.batch_work_enabled
        ):
            scalar_candidates: list[tuple[float, tuple[str, ...]]] = []
            for position in range(len(current) + 1):
                candidate = (*current[:position], customer.name, *current[position:])
                result = evaluator.route(candidate)
                if result.feasible:
                    scalar_candidates.append((result.distance, candidate))
            if scalar_candidates:
                current = min(scalar_candidates)[1]
                continue
            if current:
                sequences.append(current)
            current = (customer.name,)
            if not evaluator.route(current).feasible:
                return ()
            continue
        candidate_sequences = [
            (*current[:position], customer.name, *current[position:])
            for position in range(len(current) + 1)
        ]
        candidate_results = evaluator.route_batch(
            tuple(candidate_sequences),
            route_change_status="changed",
            candidate_pool=True,
        )
        candidates: list[tuple[float, tuple[str, ...]]] = []
        for candidate, result in zip(candidate_sequences, candidate_results, strict=True):
            if result.feasible:
                candidates.append((result.distance, candidate))
        if candidates:
            current = min(candidates)[1]
            continue
        if current:
            sequences.append(current)
        current = (customer.name,)
        if not evaluator.route_batch((current,), route_change_status="changed")[0].feasible:
            return ()
    if current:
        sequences.append(current)
    return tuple(sequences)


def _split_until_charging_feasible(
    sequence: tuple[str, ...],
    evaluator: _Evaluator,
) -> tuple[tuple[str, ...], ...]:
    if evaluator.route(sequence).feasible:
        return (sequence,)
    if len(sequence) <= 1:
        return ()
    if evaluator.candidate_control_runtime is not None:
        for split_index in range(1, len(sequence)):
            left_part = sequence[:split_index]
            right_part = sequence[split_index:]
            if evaluator.route(left_part).feasible and evaluator.route(right_part).feasible:
                return (left_part, right_part)
    midpoint = len(sequence) // 2
    left_splits = _split_until_charging_feasible(sequence[:midpoint], evaluator)
    right_splits = _split_until_charging_feasible(sequence[midpoint:], evaluator)
    return (*left_splits, *right_splits) if left_splits and right_splits else ()


def _destroy(
    instance: Instance,
    sequences: tuple[tuple[str, ...], ...],
    count: int,
    name: str,
    rng: random.Random,
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    customers = [customer for sequence in sequences for customer in sequence]
    count = min(count, len(customers))
    if name == "random":
        removed = rng.sample(customers, count)
    elif name == "worst":
        contribution: list[tuple[float, str]] = []
        by_name = instance.by_name
        for sequence in sequences:
            chain = (instance.depot.name, *sequence, instance.depot.name)
            for index, customer in enumerate(sequence, start=1):
                before = by_name[chain[index - 1]]
                node = by_name[customer]
                after = by_name[chain[index + 1]]
                saving = (
                    before.distance_to(node) + node.distance_to(after) - before.distance_to(after)
                )
                contribution.append((saving, customer))
        removed = [customer for _, customer in sorted(contribution, reverse=True)[:count]]
    else:
        anchor = rng.choice(customers)
        anchor_node = instance.by_name[anchor]
        removed = [
            customer
            for _, customer in sorted(
                (anchor_node.distance_to(instance.by_name[name]), name) for name in customers
            )[:count]
        ]
    removed_set = set(removed)
    partial = tuple(
        tuple(name for name in sequence if name not in removed_set) for sequence in sequences
    )
    return tuple(sequence for sequence in partial if sequence), tuple(removed)


def _repair(
    partial: tuple[tuple[str, ...], ...],
    removed: tuple[str, ...],
    name: str,
    evaluator: _Evaluator,
    instance: Instance,
    rng: random.Random,
) -> tuple[tuple[str, ...], ...]:
    sequences = list(partial)
    pending = list(removed)
    while pending:
        options: dict[str, list[tuple[float, int, tuple[str, ...]]]] = {}
        for customer in pending:
            options[customer] = _insertion_options(sequences, customer, evaluator, instance, name)
        feasible_customers = [customer for customer, values in options.items() if values]
        if not feasible_customers:
            return ()
        if name == "regret2":
            customer = max(
                feasible_customers,
                key=lambda item: (
                    (options[item][1][0] - options[item][0][0])
                    if len(options[item]) > 1
                    else _INFEASIBLE_COST,
                    rng.random(),
                ),
            )
        else:
            customer = min(feasible_customers, key=lambda item: (options[item][0][0], item))
        _, route_index, sequence = options[customer][0]
        if route_index == len(sequences):
            sequences.append(sequence)
        else:
            sequences[route_index] = sequence
        pending.remove(customer)
    return tuple(sequences)


def _repair_base_route_results(
    evaluator: _Evaluator,
    bases: tuple[tuple[str, ...], ...],
) -> tuple[ChargingSubproblemResult, ...]:
    """Resolve repair bases with truthful current-versus-changed status."""

    results: dict[tuple[str, ...], ChargingSubproblemResult] = {}
    changed: list[tuple[str, ...]] = []
    for base in dict.fromkeys(bases):
        if evaluator.incumbent_route_result(base) is not None:
            results[base] = evaluator.route(
                base,
                route_change_status="unchanged",
            )
        else:
            changed.append(base)
    if changed:
        changed_results = evaluator.route_batch(
            tuple(changed),
            route_change_status="changed",
        )
        results.update(zip(changed, changed_results, strict=True))
    return tuple(results[base] for base in bases)


def _insertion_options(
    sequences: list[tuple[str, ...]],
    customer: str,
    evaluator: _Evaluator,
    instance: Instance,
    mode: str,
) -> list[tuple[float, int, tuple[str, ...]]]:
    options: list[tuple[float, int, tuple[str, ...]]] = []
    if evaluator.candidate_control_enabled:
        return _controlled_insertion_options(
            sequences,
            customer,
            evaluator,
            instance,
            mode,
        )
    if evaluator.backend is ExactChargingBackend.CPU_SCALAR and not evaluator.batch_work_enabled:
        for route_index in range(len(sequences) + 1):
            base = sequences[route_index] if route_index < len(sequences) else ()
            old = evaluator.route(base).distance if base else 0.0
            for position in range(len(base) + 1):
                candidate = (*base[:position], customer, *base[position:])
                result = evaluator.route(candidate)
                if not result.feasible:
                    continue
                score = result.distance - old
                if mode == "energy":
                    score += 0.05 * result.charged_energy
                demand = sum(instance.by_name[name].demand for name in candidate)
                if demand > instance.vehicle.load_capacity + 1e-9:
                    continue
                options.append((score, route_index, candidate))
        return sorted(options)
    base_entries = [
        (route_index, base)
        for route_index in range(len(sequences) + 1)
        for base in (sequences[route_index] if route_index < len(sequences) else (),)
        if base
    ]
    base_results = _repair_base_route_results(
        evaluator,
        tuple(base for _, base in base_entries),
    )
    old_distances = {
        route_index: result.distance
        for (route_index, _), result in zip(base_entries, base_results, strict=True)
    }
    candidate_metadata: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []
    for route_index in range(len(sequences) + 1):
        base = sequences[route_index] if route_index < len(sequences) else ()
        for position in range(len(base) + 1):
            candidate = (*base[:position], customer, *base[position:])
            candidate_metadata.append((route_index, base, candidate))
    candidate_results = evaluator.route_batch(
        tuple(candidate for _, _, candidate in candidate_metadata),
        route_change_status="changed",
        candidate_pool=True,
    )
    for (route_index, _, candidate), result in zip(
        candidate_metadata,
        candidate_results,
        strict=True,
    ):
        if not result.feasible:
            continue
        score = result.distance - old_distances.get(route_index, 0.0)
        if mode == "energy":
            score += 0.05 * result.charged_energy
        demand = sum(instance.by_name[name].demand for name in candidate)
        if demand > instance.vehicle.load_capacity + 1e-9:
            continue
        options.append((score, route_index, candidate))
    return sorted(options)


def _controlled_insertion_options(
    sequences: list[tuple[str, ...]],
    customer: str,
    evaluator: _Evaluator,
    instance: Instance,
    mode: str,
) -> list[tuple[float, int, tuple[str, ...]]]:
    options: list[tuple[float, int, tuple[str, ...]]] = []
    current = tuple(sequences)
    plan_metadata: dict[
        tuple[tuple[str, ...], ...],
        tuple[int, tuple[str, ...], tuple[str, ...]],
    ] = {}
    for route_index in range(len(sequences) + 1):
        base = sequences[route_index] if route_index < len(sequences) else ()
        for position in range(len(base) + 1):
            candidate = (*base[:position], customer, *base[position:])
            demand = sum(instance.by_name[name].demand for name in candidate)
            if demand > instance.vehicle.load_capacity + 1e-9:
                continue
            plan = (
                (*current[:route_index], candidate, *current[route_index + 1 :])
                if route_index < len(sequences)
                else (*current, candidate)
            )
            plan_metadata.setdefault(plan, (route_index, base, candidate))
    feasible_plans = evaluator.evaluate_feasible_candidate_plans(
        tuple(plan_metadata),
        current_sequences=current,
    )
    for plan in feasible_plans:
        route_index, base, candidate = plan_metadata[plan]
        result = evaluator.route(candidate)
        if not result.feasible:
            raise RuntimeError("selected insertion plan lost its exact route result")
        old_result = evaluator.incumbent_route_result(base) if base else None
        if base and old_result is None:
            old_result = evaluator.route(base)
        score = result.distance - (old_result.distance if old_result is not None else 0.0)
        if mode == "energy":
            score += 0.05 * result.charged_energy
        options.append((score, route_index, candidate))
    return sorted(options)


def _weighted_choice(rng: random.Random, statistics: dict[str, OperatorStatistics]) -> str:
    names = list(statistics)
    return rng.choices(names, weights=[statistics[name].weight for name in names], k=1)[0]


def _update_weight(statistics: OperatorStatistics, reward: float, reaction: float = 0.2) -> None:
    statistics.weight = max(0.05, (1.0 - reaction) * statistics.weight + reaction * reward)


def _estimate_initial_temperature(
    instance: Instance,
    current: _EvaluatedSolution,
    evaluator: _Evaluator,
    rng: random.Random,
    config: Stage04Config,
) -> float:
    """Estimate the SA initial temperature by sampling random worse moves.

    A small sample of random destroy-repair cycles is evaluated.  The
    positive distance deltas are used to compute the temperature that
    yields the target worse-solution acceptance rate::

        T = -mean_delta / ln(target_rate)

    Falls back to the legacy ``distance * fallback_fraction`` heuristic
    if sampling does not produce enough positive deltas.
    """
    if current.objective is None:
        return max(1.0, 1_000.0 * config.temperature_fallback_fraction)
    distance_deltas: list[float] = []
    sequences = list(current.sequences)
    sample = min(config.temperature_sample_size, max(5, len(instance.customers)))
    for _ in range(sample):
        if len(sequences) <= 1:
            break
        shuffled = sequences[:]
        rng.shuffle(shuffled)
        # merge two random routes as a proxy for a worse candidate
        idx1, idx2 = rng.randrange(len(shuffled)), rng.randrange(len(shuffled))
        if idx1 == idx2:
            continue
        r1, r2 = shuffled[idx1], shuffled[idx2]
        merged = (*r1, *r2)
        remaining = [s for j, s in enumerate(shuffled) if j not in (idx1, idx2)]
        candidate_sequences = (*remaining, merged)
        try:
            candidate = evaluator.solution(candidate_sequences)
        except _TimeLimitReached:
            continue
        if candidate.feasible and candidate.objective is not None and current.objective is not None:
            delta = candidate.objective.total_distance - current.objective.total_distance
            if delta > 0:
                distance_deltas.append(delta)
    if len(distance_deltas) < 3:
        return max(
            1.0,
            current.objective.total_distance * config.temperature_fallback_fraction,
        )
    mean_delta = sum(distance_deltas) / len(distance_deltas)
    if mean_delta <= 0:
        return max(
            1.0,
            current.objective.total_distance * config.temperature_fallback_fraction,
        )
    target = config.temperature_target_acceptance_rate
    temperature = -mean_delta / math.log(target)
    return max(1.0, temperature)


def _apply_stage04_segment_update(
    all_stats: dict[str, OperatorStatistics],
    config: Stage04Config,
    iteration: int,
    events: list[dict[str, object]],
) -> None:
    """Apply the segment-based weight update to every operator.

    Only operators with at least ``min_calls_per_operator`` calls in the
    current segment have their weight updated.  Segment accumulators are
    reset after the update.  The weight change is logged.
    """
    for name, stats in all_stats.items():
        role = name.partition(":")[0] if ":" in name else "unspecified"
        if stats.segment_calls < config.min_calls_per_operator:
            events.append(
                {
                    "type": "stage04_segment_skip",
                    "operator": name,
                    "role": role,
                    "iteration": iteration,
                    "segment_calls": stats.segment_calls,
                    "segment_reward_sum": stats.segment_reward_sum,
                    "minimum_calls": config.min_calls_per_operator,
                }
            )
            stats.segment_calls = 0
            stats.segment_reward_sum = 0.0
            continue
        old_weight = stats.weight
        new_weight = config.apply_segment_update(
            stats.weight,
            stats.segment_reward_sum,
            stats.segment_calls,
        )
        stats.weight = new_weight
        stats.weight_history.append((iteration, new_weight))
        events.append(
            {
                "type": "stage04_segment_update",
                "operator": name,
                "role": role,
                "iteration": iteration,
                "old_weight": old_weight,
                "new_weight": new_weight,
                "segment_calls": stats.segment_calls,
                "segment_reward_sum": stats.segment_reward_sum,
            }
        )
        stats.segment_calls = 0
        stats.segment_reward_sum = 0.0


def _stage04_accumulate(stats: OperatorStatistics, reward: float) -> None:
    """Accumulate reward into the segment buffer."""
    stats.segment_reward_sum += reward
    stats.segment_calls += 1


def _stage04_adaptive_weight_statistics(
    profile: OperatorProfile,
    neighborhood_statistics: dict[str, OperatorStatistics],
    destroy_statistics: dict[str, OperatorStatistics],
    repair_statistics: dict[str, OperatorStatistics],
) -> dict[str, OperatorStatistics]:
    """Return exactly the operators whose weights participate in selection."""

    selected: dict[str, OperatorStatistics] = {}
    if profile is not OperatorProfile.BASELINE:
        selected.update(
            {
                f"neighborhood:{name}": stats
                for name, stats in neighborhood_statistics.items()
                if name != "vehicle_reduction_refinement"
            }
        )
    selected.update({f"destroy:{name}": stats for name, stats in destroy_statistics.items()})
    selected.update({f"repair:{name}": stats for name, stats in repair_statistics.items()})
    return selected


def _select_stage02_neighborhood(
    iteration: int,
    rng: random.Random,
    statistics: dict[str, OperatorStatistics],
    *,
    include_quality: bool = True,
    include_constraint: bool = True,
) -> str:
    warmup: tuple[str, ...]
    if include_quality and "relocate" in statistics:
        warmup = (
            "route_elimination",
            "vehicle_count_aware_repair",
            "route_merge",
            "relocate",
            "swap",
            "two_opt_star",
            "route_segment_destroy",
            "ejection_chain",
        )
    else:
        warmup = ("route_elimination", "vehicle_count_aware_repair", "route_merge")
    if iteration < len(warmup):
        return warmup[iteration]
    if include_quality:
        allowed = {
            name: statistics[name]
            for name in statistics
            if include_constraint or name not in _CONSTRAINT_REMOVAL_NEIGHBORHOODS
        }
        return _weighted_choice(rng, allowed)
    legacy_names = (
        "standard",
        "vehicle_count_aware_repair",
        "route_elimination",
        "route_merge",
    )
    return _weighted_choice(rng, {name: statistics[name] for name in legacy_names})


def _select_constraint_operator(
    iteration: int,
    rng: random.Random,
    statistics: dict[str, OperatorStatistics],
) -> str:
    if iteration < len(_CONSTRAINT_REMOVAL_ORDER):
        return _CONSTRAINT_REMOVAL_ORDER[iteration]
    return _weighted_choice(rng, statistics)


def _neighborhood_names(profile: OperatorProfile) -> tuple[str, ...]:
    if profile is OperatorProfile.STAGE02_ROUTE_REDUCTION:
        return (
            "standard",
            "vehicle_count_aware_repair",
            "route_elimination",
            "route_merge",
        )
    if profile is OperatorProfile.STAGE02_ROUTE_QUALITY:
        return (
            "standard",
            "vehicle_count_aware_repair",
            "route_elimination",
            "route_merge",
            "relocate",
            "swap",
            "two_opt_star",
            "route_segment_destroy",
            "ejection_chain",
        )
    if profile is OperatorProfile.STAGE02_CONSTRAINT_GUIDED:
        return (
            "standard",
            "vehicle_count_aware_repair",
            "route_elimination",
            "route_merge",
            "relocate",
            "swap",
            "two_opt_star",
            "route_segment_destroy",
            "ejection_chain",
            *_CONSTRAINT_REMOVAL_ORDER,
        )
    return ()


def _quality_shadow_neighborhood(iteration: int) -> str:
    return (
        _QUALITY_NEIGHBORHOOD_ORDER[iteration]
        if iteration < len(_QUALITY_NEIGHBORHOOD_ORDER)
        else ""
    )


def _quality_shadow_proposal(
    operator: str,
    instance: Instance,
    sequences: RouteSequences,
    evaluator: _Evaluator,
    config: VehicleOperatorConfig,
    *,
    precomputed_routes: dict[tuple[str, ...], ChargingSubproblemResult],
) -> tuple[RouteSequences, tuple[NeighborhoodEvent, ...]]:
    probe_budget = config.quality_probe_exact_evaluation_budget
    probe_config = replace(
        config,
        relocate_exact_evaluation_budget=min(config.relocate_exact_evaluation_budget, probe_budget),
        swap_exact_evaluation_budget=min(config.swap_exact_evaluation_budget, probe_budget),
        two_opt_star_exact_evaluation_budget=min(
            config.two_opt_star_exact_evaluation_budget, probe_budget
        ),
        route_segment_exact_evaluation_budget=min(
            config.route_segment_exact_evaluation_budget,
            config.quality_route_segment_probe_exact_evaluation_budget,
        ),
        ejection_chain_exact_evaluation_budget=min(
            config.ejection_chain_exact_evaluation_budget, probe_budget
        ),
    )
    if operator == "relocate":
        proposal = propose_relocate(
            instance,
            sequences,
            evaluator,
            config=probe_config,
            precomputed_routes=precomputed_routes,
        )
    elif operator == "swap":
        proposal = propose_swap(
            instance,
            sequences,
            evaluator,
            config=probe_config,
            precomputed_routes=precomputed_routes,
        )
    elif operator == "two_opt_star":
        proposal = propose_two_opt_star(
            instance,
            sequences,
            evaluator,
            config=probe_config,
            precomputed_routes=precomputed_routes,
        )
    elif operator == "route_segment_destroy":
        proposal = propose_route_segment_destroy(
            instance, sequences, evaluator, config=probe_config
        )
    elif operator == "ejection_chain":
        proposal = propose_ejection_chain(
            instance,
            sequences,
            evaluator,
            config=probe_config,
            precomputed_routes=precomputed_routes,
        )
    else:
        raise ValueError(f"unsupported Stage 2.2 shadow operator: {operator}")
    return proposal.sequences or (), proposal.events


def _constraint_lane_step(
    instance: Instance,
    current: _EvaluatedSolution,
    evaluator: _Evaluator,
    config: VehicleOperatorConfig,
    *,
    operator: str,
    selection: RemovalSizeSelection,
    seed: int,
) -> tuple[_EvaluatedSolution, tuple[NeighborhoodEvent, ...]]:
    proposal = propose_constraint_removal(
        instance,
        current.sequences,
        evaluator,
        operator=operator,
        selection=selection,
        seed=seed,
        precomputed_routes={
            sequence: charging
            for sequence, charging in zip(current.sequences, current.charging, strict=True)
        },
    )
    events = list(proposal.events)
    if proposal.partial is None:
        return _infeasible_solution(), tuple(events)

    before_calls = evaluator.calls
    constraint_budget = min(
        config.constraint_probe_exact_evaluation_budget,
        {
            ConstraintRemovalOperator.STATION_PRESSURE.value: (
                config.station_pressure_exact_evaluation_budget
            ),
            ConstraintRemovalOperator.TIME_WINDOW_CONFLICT.value: (
                config.time_window_conflict_exact_evaluation_budget
            ),
            ConstraintRemovalOperator.WORST_ENERGY_DETOUR.value: (
                config.worst_energy_detour_exact_evaluation_budget
            ),
            ConstraintRemovalOperator.SHAW_RELATED.value: (
                config.shaw_related_exact_evaluation_budget
            ),
        }[operator],
    )
    before_calls = evaluator.calls
    try:
        repair = repair_constraint_removal(
            current.sequences,
            proposal.partial,
            proposal.removed_customers,
            evaluator,
            instance,
            budget=constraint_budget,
        )
        candidate_sequences = repair.sequences or ()
        candidate = evaluator.solution(
            candidate_sequences,
            precomputed_routes={
                sequence: charging
                for sequence, charging in zip(current.sequences, current.charging, strict=True)
            },
        )
    except _TimeLimitReached:
        events.append(
            NeighborhoodEvent(
                operator,
                "time_limit",
                "time_limit_reached_during_constraint_repair",
                removed_customers=proposal.removed_customers,
                candidate_route_sequences=proposal.partial,
                prefilter_passed=True,
                exact_route_evaluations=evaluator.calls - before_calls,
                track="constraint_lane",
                constraint_category=operator,
                removal_tier=selection.tier.value,
                removal_size_requested=selection.requested_count,
                removal_size_actual=len(proposal.removed_customers),
                stagnation_iterations=selection.stagnation_iterations,
                removal_trigger=selection.trigger_reason,
                reset_observed=selection.reset_observed,
            )
        )
        return _infeasible_solution(), tuple(events)
    events.append(
        NeighborhoodEvent(
            operator,
            "candidate_proposed" if candidate.feasible else "failed",
            "constraint_removal_repaired"
            if candidate.feasible
            else repair.failure_reason or "constraint_repair_infeasible",
            affected_route_indices=tuple(
                index
                for index, (before, after) in enumerate(
                    zip(current.sequences, candidate_sequences, strict=False)
                )
                if before != after
            ),
            removed_customers=proposal.removed_customers,
            candidate_route_sequences=candidate_sequences,
            candidate_vehicle_delta=(
                len(candidate_sequences) - len(current.sequences) if candidate.feasible else None
            ),
            candidate_feasible=candidate.feasible,
            prefilter_passed=bool(proposal.partial),
            new_routes_created=repair.new_routes_created,
            exact_route_evaluations=evaluator.calls - before_calls,
            track="constraint_lane",
            constraint_category=operator,
            removal_tier=selection.tier.value,
            removal_size_requested=selection.requested_count,
            removal_size_actual=len(proposal.removed_customers),
            stagnation_iterations=selection.stagnation_iterations,
            removal_trigger=selection.trigger_reason,
            reset_observed=selection.reset_observed,
        )
    )
    return candidate, tuple(events)


def _infeasible_solution() -> _EvaluatedSolution:
    return _EvaluatedSolution((), (), False, None)


def _record_neighborhood_proposal(
    statistics: OperatorStatistics,
    events: tuple[NeighborhoodEvent, ...],
    candidate: _EvaluatedSolution,
    current: _EvaluatedSolution,
) -> None:
    statistics.prefilter_passed += sum(
        event.aggregate_count for event in events if event.prefilter_passed
    )
    statistics.prefilter_rejected += sum(
        event.aggregate_count
        for event in events
        if event.status in {"prefilter_rejected", "prefilter_rejected_aggregate"}
    )
    statistics.new_routes_created += sum(event.new_routes_created for event in events)
    statistics.exact_route_evaluations += sum(event.exact_route_evaluations for event in events)
    statistics.candidate_proposals += sum(event.status == "candidate_proposed" for event in events)
    statistics.feasible_candidates += sum(
        event.aggregate_count for event in events if event.candidate_feasible
    )
    failure_statuses = {
        "failed",
        "prefilter_rejected",
        "exact_infeasible",
        "budget_exhausted",
        "not_applicable",
        "time_limit",
        "prefilter_rejected_aggregate",
        "exact_infeasible_aggregate",
        "candidate_control_skipped_aggregate",
    }
    for event in events:
        if event.status in failure_statuses:
            statistics.failure_reasons[event.reason] = (
                statistics.failure_reasons.get(event.reason, 0) + event.aggregate_count
            )
    if candidate.feasible:
        statistics.feasible_repairs += 1
    elif not events:
        statistics.failure_reasons["candidate_infeasible"] = (
            statistics.failure_reasons.get("candidate_infeasible", 0) + 1
        )
    if candidate.objective is None or current.objective is None:
        return
    if candidate.objective.vehicle_count < current.objective.vehicle_count:
        statistics.vehicle_reductions += 1
    if candidate.objective.total_distance < current.objective.total_distance - 1e-9:
        statistics.distance_improvements += 1


def _event_record(event: NeighborhoodEvent, iteration: int) -> dict[str, object]:
    record = event.to_dict()
    record["iteration"] = iteration
    return record


def _deadline_event(
    operator: str,
    reason: str,
    error: _TimeLimitReached,
    reference_sequences: tuple[tuple[str, ...], ...],
) -> NeighborhoodEvent:
    sequence = error.sequence
    affected = tuple(
        index
        for index, candidate_sequence in enumerate(reference_sequences)
        if sequence is not None and candidate_sequence == sequence
    )
    return NeighborhoodEvent(
        operator,
        "time_limit",
        reason,
        route_indices=affected,
        affected_route_indices=affected,
        candidate_route_sequences=(sequence,) if sequence is not None else (),
        prefilter_passed=error.exact_route_evaluations > 0,
        exact_route_evaluations=error.exact_route_evaluations,
    )


def _annotated_event_record(
    event: NeighborhoodEvent,
    *,
    iteration: int,
    accepted: bool,
    vehicle_reduction: bool,
    distance_improvement: bool,
    candidate: _EvaluatedSolution,
) -> dict[str, object]:
    record = _event_record(event, iteration)
    # A proposal may contain many feasible probes, but only its final
    # candidate_proposed event is the sequence actually returned to ALNS.
    # Do not report every feasible probe as accepted or improved.
    event_candidate = event.candidate_feasible and event.status == "candidate_proposed"
    record.update(
        {
            "accepted": accepted and event_candidate,
            "vehicle_reduction": vehicle_reduction if event_candidate else False,
            "distance_improvement": distance_improvement if event_candidate else False,
            "candidate_objective_key": (
                candidate.objective.key if candidate.objective is not None else ()
            ),
        }
    )
    return record


def _failed_result(
    started: float,
    evaluator: _Evaluator,
    reason: str,
    *,
    operator_profile: str = OperatorProfile.BASELINE.value,
    termination_mode: str = "wall_clock",
) -> ALNSResult:
    exact_controller = evaluator.exact_call_controller
    return ALNSResult(
        feasible=False,
        routes=(),
        customer_sequences=(),
        objective=None,
        vehicle_count=0,
        total_energy=0.0,
        total_charged_energy=0.0,
        total_charging_time=0.0,
        iterations=0,
        accepted_moves=0,
        improving_moves=0,
        rejected_moves=0,
        first_feasible_time=float("inf"),
        best_time=float("inf"),
        runtime_seconds=time.perf_counter() - started,
        charging_subproblem_calls=evaluator.calls,
        charging_subproblem_time=evaluator.runtime,
        charging_labels_generated=evaluator.labels_generated,
        charging_labels_pruned=evaluator.labels_pruned,
        destroy_statistics={},
        repair_statistics={},
        operator_profile=operator_profile,
        neighborhood_statistics={},
        neighborhood_events=(),
        failure_reason=reason,
        cache_hits=(
            _cache_statistics([evaluator.route_cache])["cache_hits"]
            if evaluator.cache_incremental_enabled
            else evaluator.cache_hits
        ),
        cache_misses=(
            _cache_statistics([evaluator.route_cache])["cache_misses"]
            if evaluator.cache_incremental_enabled
            else evaluator.calls
        ),
        unique_route_evaluations=(
            _completed_unique_route_evaluations((evaluator,))
            if exact_controller is not None
            else _cache_statistics([evaluator.route_cache])["unique_route_evaluations"]
            if evaluator.cache_incremental_enabled
            else (
                len(evaluator.cache)
                if evaluator.local_cache_enabled
                else len(evaluator.evaluated_route_keys)
            )
        ),
        effective_iterations=0,
        removal_tier_counts={tier.value: 0 for tier in RemovalTier},
        maximum_stagnation=0,
        constraint_operator_statistics={},
        screening_statistics=evaluator.screening_statistics(),
        cache_incremental_statistics=_aggregate_cache_incremental_statistics(
            (evaluator,),
            evaluator.cache_incremental_config,
            [evaluator.route_cache],
        ),
        charging_backend=evaluator.backend.value,
        batch_size=evaluator.batch_size,
        backend_metrics=evaluator.backend_metrics.to_dict(),
        termination_mode=termination_mode,
        watchdog_triggered=termination_mode == "fixed_work",
        exact_started_calls=(exact_controller.started_calls if exact_controller is not None else 0),
        exact_completed_calls=(
            exact_controller.completed_calls if exact_controller is not None else 0
        ),
        exact_interrupted_calls=(
            exact_controller.interrupted_calls if exact_controller is not None else 0
        ),
        exact_budget_exhaustions=(
            exact_controller.budget_exhaustions if exact_controller is not None else 0
        ),
        termination_reason=(
            "exact_call_budget_exhausted"
            if exact_controller is not None and exact_controller.budget_reached
            else "initialization_failed"
        ),
        exact_deadline_statistics=(
            exact_controller.to_dict() if exact_controller is not None else {}
        ),
    )


def _aggregate_backend_metrics(
    evaluators: tuple[_Evaluator, ...],
) -> dict[str, object]:
    if not evaluators:
        return {}
    backends = {evaluator.backend.value for evaluator in evaluators}
    if len(backends) != 1:
        raise RuntimeError(f"ALNS lanes used inconsistent charging backends: {backends}")
    aggregate = BackendMetrics(
        next(iter(backends)),
        max(evaluator.batch_size for evaluator in evaluators),
    )
    for evaluator in evaluators:
        aggregate.add(evaluator.backend_metrics)
    return aggregate.to_dict()


def _aggregate_screening_statistics(
    evaluators: tuple[_Evaluator, ...],
    *,
    native_runtime: NativeKernelRuntime | None = None,
) -> dict[str, object]:
    reason_counts: dict[str, int] = {}
    negative_result_cache_statistics: dict[str, object] | None = None
    total_runtime = 0.0
    totals = {
        "screening_calls": 0,
        "screening_passes": 0,
        "screening_rejections": 0,
        "screening_cache_hits": 0,
        "screening_exact_call_blocked": 0,
    }
    for evaluator in evaluators:
        statistics = cast(Any, evaluator.screening_statistics())
        raw_negative_result_cache = statistics.get(
            "negative_screening_result_cache"
        )
        if raw_negative_result_cache is not None:
            observed = dict(cast(Mapping[str, object], raw_negative_result_cache))
            if (
                negative_result_cache_statistics is not None
                and negative_result_cache_statistics != observed
            ):
                raise RuntimeError(
                    "ALNS lanes observed inconsistent bounded negative-cache state"
                )
            negative_result_cache_statistics = observed
        for field_name in totals:
            totals[field_name] += int(statistics[field_name])
        total_runtime += float(statistics["screening_runtime_seconds"])
        for reason, count in dict(statistics["screening_reason_counts"]).items():
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + int(count)
    native_statistics: dict[str, object] = (
        native_runtime.statistics() if native_runtime is not None else {}
    )
    output: dict[str, object] = {
        **totals,
        "screening_runtime_seconds": total_runtime,
        "screening_reason_counts": dict(sorted(reason_counts.items())),
        **native_statistics,
    }
    if negative_result_cache_statistics is not None:
        output["negative_screening_result_cache"] = (
            negative_result_cache_statistics
        )
    return output


def _unique_route_caches(
    route_caches: list[RouteEvaluationCache | None],
) -> tuple[RouteEvaluationCache, ...]:
    unique: list[RouteEvaluationCache] = []
    seen: set[int] = set()
    for cache in route_caches:
        if cache is None or id(cache) in seen:
            continue
        seen.add(id(cache))
        unique.append(cache)
    return tuple(unique)


def _cache_statistics(
    route_caches: list[RouteEvaluationCache | None],
) -> dict[str, int]:
    totals = {
        "cache_lookups": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_stores": 0,
        "cache_evictions": 0,
        "cache_oversize_not_cached": 0,
        "entries_current": 0,
        "entries_peak": 0,
        "bytes_current": 0,
        "bytes_peak": 0,
        "unique_route_evaluations": 0,
    }
    for cache in _unique_route_caches(route_caches):
        statistics = cache.statistics
        totals["cache_lookups"] += statistics.lookups
        totals["cache_hits"] += statistics.hits
        totals["cache_misses"] += statistics.misses
        totals["cache_stores"] += statistics.stores
        totals["cache_evictions"] += statistics.evictions
        totals["cache_oversize_not_cached"] += statistics.oversize_not_cached
        totals["entries_current"] += statistics.entries_current
        totals["entries_peak"] += statistics.entries_peak
        totals["bytes_current"] += statistics.bytes_current
        totals["bytes_peak"] += statistics.bytes_peak
        totals["unique_route_evaluations"] += statistics.unique_keys_seen
    return totals


def _completed_unique_route_evaluations(
    evaluators: tuple[_Evaluator, ...],
) -> int:
    """Count completed exact route identities using the cache ownership boundary."""

    route_caches = [evaluator.route_cache for evaluator in evaluators]
    shared_cache = (
        bool(route_caches)
        and all(cache is not None for cache in route_caches)
        and len({id(cache) for cache in route_caches}) == 1
    )
    if shared_cache:
        return len(
            {sequence for evaluator in evaluators for sequence in evaluator.evaluated_routes}
        )
    return sum(len(evaluator.evaluated_routes) for evaluator in evaluators)


def _aggregate_cache_incremental_statistics(
    evaluators: tuple[_Evaluator, ...],
    config: CacheIncrementalConfig | None,
    route_caches: list[RouteEvaluationCache | None],
) -> dict[str, object]:
    if config is None or not config.enabled:
        return {}
    cache_statistics = _cache_statistics(route_caches)
    incremental_totals: dict[str, int] = {
        "incremental_propagations": 0,
        "incremental_fallbacks": 0,
        "incremental_reused_prefix_edges": 0,
        "incremental_reused_suffix_edges": 0,
    }
    station_reachability: dict[str, object] = {}
    station_queries = 0
    for evaluator in evaluators:
        statistics = evaluator.incremental_statistics()
        for field_name in incremental_totals:
            incremental_totals[field_name] += int(cast(Any, statistics[field_name]))
        reachability = statistics["station_reachability"]
        if isinstance(reachability, dict) and not station_reachability:
            station_reachability = dict(reachability)
        if isinstance(reachability, dict):
            station_queries += int(reachability.get("queries", 0))
    station_reachability["queries"] = station_queries
    return {
        "schema_version": config.schema_version,
        "config": asdict(config),
        **cache_statistics,
        **incremental_totals,
        "route_cache": {
            **cache_statistics,
            "eviction_policy": config.eviction_policy,
            "max_entries": config.max_entries,
            "max_memory_bytes": config.max_memory_bytes,
            "instance_hash": config.instance_hash,
        },
        "station_reachability": station_reachability,
    }
