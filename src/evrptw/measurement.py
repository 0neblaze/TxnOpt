from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

from evrptw.candidate_control import CandidateControlConfig
from evrptw.exact_deadline import ExactDeadlineConfig
from evrptw.objective import ObjectiveComparison, SolutionObjective, compare_objectives

if TYPE_CHECKING:
    from evrptw.models import Instance


TRACE_SCHEMA_VERSION = "stage03-trace-v1"
CACHE_INCREMENTAL_TRACE_SCHEMA_VERSION = "stage03-trace-v2"
EXACT_DEADLINE_TRACE_SCHEMA_VERSION = "stage03-trace-v3"
CANDIDATE_CONTROL_TRACE_SCHEMA_VERSION = "stage03-trace-v4"
SCREENING_SCHEMA_VERSION = "stage031-screening-v1"
LEGACY_UNIQUE_ROUTE_SEMANTICS = "started_lane_identity_legacy_v1"
COMPLETED_UNIQUE_ROUTE_SEMANTICS = "completed_cache_owner_identity_v2"
ROUTE_EVALUATION_KINDS = frozenset({"exact_call", "cache_hit", "precomputed_route"})


class MeasurementTraceSink(Protocol):
    """Runtime-only consumer for bounded Stage 5.2 trace persistence."""

    def append_route_evaluation(self, record: RouteEvaluationTrace) -> None: ...

    def append_event(self, event: Mapping[str, object]) -> None: ...

    def append_screening_decision(self, decision: ScreeningDecision) -> None: ...

    def append_incremental_propagation(self, propagation: Mapping[str, object]) -> None: ...


def canonical_route_key(sequence: tuple[str, ...] | list[str]) -> str:
    """Return a collision-free, stable key for an ordered customer route."""

    values = tuple(sequence)
    return "route:" + "|".join(f"{len(name)}:{name}" for name in values)


@dataclass(frozen=True, slots=True)
class MeasurementConfig:
    """Opt-in controls for Stage 3.0 measurement.

    The configuration intentionally contains measurement-only switches.  It
    does not alter charging, neighbourhood, cache, or deadline behaviour.
    """

    enabled: bool = True
    schema_version: str = TRACE_SCHEMA_VERSION
    record_route_dictionary: bool = True
    record_operator_events: bool = True
    record_candidate_states: bool = True
    stream_sink: MeasurementTraceSink | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Stage 3.0 trace schema {self.schema_version}; "
                f"expected {TRACE_SCHEMA_VERSION}"
            )


def _measurement_config_payload(config: MeasurementConfig) -> dict[str, object]:
    """Serialize only stable controls; the live stream sink is never evidence."""

    return {
        "enabled": config.enabled,
        "schema_version": config.schema_version,
        "record_route_dictionary": config.record_route_dictionary,
        "record_operator_events": config.record_operator_events,
        "record_candidate_states": config.record_candidate_states,
    }


@dataclass(frozen=True, slots=True)
class CheapScreeningConfig:
    """Opt-in Stage 3.1 screening controls.

    The negative cache is deliberately the only cache introduced here.  Exact
    route-result caching remains the evaluator's existing Stage 2/3.0 cache and
    is recorded separately from screening evidence.
    """

    enabled: bool = True
    schema_version: str = SCREENING_SCHEMA_VERSION
    negative_sequence_cache: bool = True
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.schema_version != SCREENING_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Stage 3.1 screening schema {self.schema_version}; "
                f"expected {SCREENING_SCHEMA_VERSION}"
            )
        if self.epsilon <= 0.0:
            raise ValueError("screening epsilon must be positive")


@dataclass(frozen=True, slots=True)
class ScreeningCheckTrace:
    check: str
    status: str
    value: float | bool | None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ScreeningDecision:
    decision_id: int
    route_key: str
    lane: str
    iteration: int | None
    operator: str
    status: str
    first_failed_check: str
    reason: str
    checks: tuple[ScreeningCheckTrace, ...]
    demand: float
    min_time_window_slack: float
    distance_lower_bound: float
    distance_increment_lower_bound: float | None
    single_segment_reachable: bool
    structural_energy_lower_bound: float
    negative_cache_hit: bool
    exact_call_blocked: bool
    started_at: float
    completed_at: float
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class RouteEvaluationTrace:
    evaluation_id: int
    route_key: str
    lane: str
    iteration: int | None
    operator: str
    kind: str
    started_at: float
    completed_at: float | None
    duration_seconds: float
    exact_started: bool
    exact_completed: bool
    feasible: bool | None
    failure_reason: str
    labels_generated: int = 0
    labels_expanded: int = 0
    labels_pruned: int = 0
    deadline_boundary: str = ""
    cache_key_digest: str = ""
    route_change_status: str = "unknown"
    status: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ROUTE_EVALUATION_KINDS:
            raise ValueError(f"unsupported route evaluation kind: {self.kind}")
        if self.evaluation_id <= 0:
            raise ValueError("evaluation_id must be positive")
        if self.exact_completed and not self.exact_started:
            raise ValueError("an exact call cannot complete before it starts")
        if self.route_change_status not in {"changed", "unchanged", "unknown"}:
            raise ValueError(f"unsupported route change status: {self.route_change_status}")


@dataclass(slots=True)
class _StreamingTraceSummary:
    counts: Counter[str] = field(default_factory=Counter)
    operator_calls: Counter[str] = field(default_factory=Counter)
    operator_calls_by_group: dict[str, Counter[str]] = field(default_factory=dict)
    screening_reasons: Counter[str] = field(default_factory=Counter)
    cache_operations: Counter[str] = field(default_factory=Counter)
    legacy_exact_route_keys: set[tuple[str, str]] = field(default_factory=set)
    completed_shared_route_keys: set[str] = field(default_factory=set)
    completed_lane_route_keys: set[str | tuple[str, str]] = field(default_factory=set)
    legacy_candidate_states: int = 0
    accepted_legacy: int = 0
    rejected_legacy: int = 0
    improving_legacy: int = 0
    candidate_pending_cache_hits: int = 0


class _ExternalizedTraceList[T](list[T]):
    """Append-compatible list façade whose complete rows live in an external sink."""

    def __init__(
        self,
        *,
        family: str,
        sink: MeasurementTraceSink,
        observer: Any,
    ) -> None:
        super().__init__()
        self._family = family
        self._sink = sink
        self._observer = observer
        self._count = 0

    def append(self, value: T) -> None:
        self._observer(self._family, value)
        if self._family == "route_evaluations":
            self._sink.append_route_evaluation(cast(RouteEvaluationTrace, value))
        elif self._family == "events":
            self._sink.append_event(cast(Mapping[str, object], value))
        elif self._family == "screening_decisions":
            self._sink.append_screening_decision(cast(ScreeningDecision, value))
        elif self._family == "incremental_propagations":
            self._sink.append_incremental_propagation(cast(Mapping[str, object], value))
        else:  # pragma: no cover - construction owns the closed family set
            raise RuntimeError(f"unsupported externalized trace family: {self._family}")
        self._count += 1

    def extend(self, values: Iterable[T]) -> None:
        for value in values:
            self.append(value)

    def increment_external_count(self) -> None:
        """Advance a record emitted through a sink-specific typed fast path."""

        self._count += 1

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[T]:
        raise RuntimeError(
            f"{self._family} were externalized during solve and cannot be materialized in memory"
        )


class _MeasuredResult(Protocol):
    @property
    def charging_subproblem_calls(self) -> int: ...

    @property
    def cache_hits(self) -> int: ...

    @property
    def unique_route_evaluations(self) -> int: ...

    @property
    def neighborhood_statistics(self) -> dict[str, dict[str, object]]: ...

    @property
    def destroy_statistics(self) -> dict[str, dict[str, object]]: ...

    @property
    def repair_statistics(self) -> dict[str, dict[str, object]]: ...

    @property
    def accepted_moves(self) -> int: ...

    @property
    def rejected_moves(self) -> int: ...

    @property
    def improving_moves(self) -> int: ...


class _ChargingResult(Protocol):
    @property
    def feasible(self) -> bool: ...

    @property
    def failure_reason(self) -> str: ...

    @property
    def labels_generated(self) -> int: ...

    @property
    def labels_expanded(self) -> int: ...

    @property
    def labels_pruned(self) -> int: ...


@dataclass(slots=True)
class Stage03Trace:
    """Raw, append-only-in-practice evidence emitted by one measured solve."""

    config: MeasurementConfig = field(default_factory=MeasurementConfig)
    started_at_perf: float = field(default_factory=time.perf_counter, repr=False)
    route_dictionary: dict[str, tuple[str, ...]] = field(default_factory=dict)
    route_evaluations: list[RouteEvaluationTrace] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)
    finished_at: float | None = None
    result_summary: dict[str, object] = field(default_factory=dict)
    # These fields are appended after the Stage 3.0 fields so positional
    # construction of the v1 trace remains compatible.
    screening_config: CheapScreeningConfig | None = None
    screening_decisions: list[ScreeningDecision] = field(default_factory=list)
    cache_incremental_config: Any | None = None
    incremental_propagations: list[dict[str, object]] = field(default_factory=list)
    trace_schema_version: str = TRACE_SCHEMA_VERSION
    exact_deadline_config: ExactDeadlineConfig | None = None
    candidate_control_config: CandidateControlConfig | None = None
    _stream_summary: _StreamingTraceSummary | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._validate_screening_route_dictionary()
        if self.trace_schema_version not in (
            TRACE_SCHEMA_VERSION,
            CACHE_INCREMENTAL_TRACE_SCHEMA_VERSION,
            EXACT_DEADLINE_TRACE_SCHEMA_VERSION,
            CANDIDATE_CONTROL_TRACE_SCHEMA_VERSION,
        ):
            raise ValueError(f"unsupported Stage 3 trace schema {self.trace_schema_version}")
        if self.cache_incremental_config is not None:
            self.trace_schema_version = CACHE_INCREMENTAL_TRACE_SCHEMA_VERSION
        if self.exact_deadline_config is not None:
            self.trace_schema_version = EXACT_DEADLINE_TRACE_SCHEMA_VERSION
        if self.candidate_control_config is not None:
            self.trace_schema_version = CANDIDATE_CONTROL_TRACE_SCHEMA_VERSION
        if self.config.stream_sink is not None:
            if any(
                (
                    self.route_evaluations,
                    self.events,
                    self.screening_decisions,
                    self.incremental_propagations,
                )
            ):
                raise ValueError("a streaming trace must start with empty event families")
            self._stream_summary = _StreamingTraceSummary()
            sink = self.config.stream_sink
            self.route_evaluations = _ExternalizedTraceList(
                family="route_evaluations",
                sink=sink,
                observer=self._observe_streamed_record,
            )
            self.events = _ExternalizedTraceList(
                family="events",
                sink=sink,
                observer=self._observe_streamed_record,
            )
            self.screening_decisions = _ExternalizedTraceList(
                family="screening_decisions",
                sink=sink,
                observer=self._observe_streamed_record,
            )
            self.incremental_propagations = _ExternalizedTraceList(
                family="incremental_propagations",
                sink=sink,
                observer=self._observe_streamed_record,
            )

    @property
    def streamed_record_counts(self) -> dict[str, int]:
        summary = self._stream_summary
        if summary is None:
            return {}
        return {
            family: int(summary.counts[family])
            for family in (
                "route_evaluations",
                "events",
                "screening_decisions",
                "incremental_propagations",
            )
        }

    @property
    def stream_sink(self) -> MeasurementTraceSink | None:
        return self.config.stream_sink

    def _external_unique_route_count(self, semantics: str) -> int | None:
        """Read the exact bounded identity count exposed by a Stage 5.2 shard."""

        sink = cast(Any, self.config.stream_sink)
        if sink is None:
            return None
        counter = getattr(sink, "unique_route_identity_count", None)
        if callable(counter):
            return int(counter(semantics))
        shard = getattr(sink, "_shard", None)
        shard_counter = getattr(shard, "unique_route_identity_count", None)
        axis_name = getattr(sink, "axis_name", None)
        if callable(shard_counter) and isinstance(axis_name, str):
            return int(shard_counter(semantics, axis_name))
        return None

    @property
    def stream_unique_identity_hot_entries(self) -> int:
        """Bounded in-process identity state used by a streaming trace sink."""

        sink = cast(Any, self.config.stream_sink)
        if sink is None:
            return 0
        direct = getattr(sink, "unique_route_hot_entries", None)
        if isinstance(direct, int):
            return direct
        shard = getattr(sink, "_shard", None)
        return int(getattr(shard, "unique_route_hot_entries", 0))

    def _observe_streamed_record(self, family: str, value: object) -> None:
        summary = self._stream_summary
        if summary is None:
            raise RuntimeError("stream observer is unavailable")
        summary.counts[family] += 1
        if family == "route_evaluations":
            record = cast(RouteEvaluationTrace, value)
            summary.counts[f"route_kind:{record.kind}"] += 1
            summary.counts["exact_started"] += int(record.exact_started)
            summary.counts["exact_completed"] += int(record.exact_completed)
            summary.counts["interrupted"] += int(record.status == "interrupted_deadline")
            if self._external_unique_route_count("legacy_started") is None:
                if record.kind == "exact_call" and record.exact_started:
                    summary.legacy_exact_route_keys.add((record.lane, record.route_key))
                if record.kind == "exact_call" and record.exact_completed:
                    summary.completed_shared_route_keys.add(record.route_key)
                    summary.completed_lane_route_keys.add(
                        (
                            "legacy" if record.lane == "initialization" else record.lane,
                            record.route_key,
                        )
                    )
            return
        if family == "screening_decisions":
            decision = cast(ScreeningDecision, value)
            self._observe_streamed_screening_fields(
                status=decision.status,
                reason=decision.reason,
                negative_cache_hit=decision.negative_cache_hit,
                exact_call_blocked=decision.exact_call_blocked,
            )
            return
        if family == "incremental_propagations":
            propagation = cast(Mapping[str, object], value)
            summary.counts["incremental_fallback"] += int(propagation.get("status") == "fallback")
            return
        event = cast(Mapping[str, object], value)
        event_type = str(event.get("event_type", ""))
        summary.counts[f"event_type:{event_type}"] += 1
        if event_type == "operator_call":
            operator = str(event.get("operator", ""))
            group = str(event.get("statistics_group", ""))
            summary.operator_calls[operator] += 1
            summary.operator_calls_by_group.setdefault(group, Counter())[operator] += 1
        if event_type == "cache_event":
            operation = str(event.get("operation", ""))
            if operation == "lookup_result":
                summary.cache_operations["lookup"] += 1
                summary.cache_operations[str(event.get("lookup_result", ""))] += 1
            else:
                summary.cache_operations[operation] += 1
            if operation == "candidate_pending_hit":
                summary.candidate_pending_cache_hits += 1
        if event_type != "candidate_state" or event.get("lane") != "legacy":
            return
        summary.legacy_candidate_states += 1
        if event.get("accepted") is not True:
            summary.rejected_legacy += 1
            return
        summary.accepted_legacy += 1
        current = _objective_from_key(event.get("current_objective_key"))
        candidate = _objective_from_key(event.get("candidate_objective_key"))
        if (
            current is not None
            and candidate is not None
            and compare_objectives(candidate, current) is ObjectiveComparison.BETTER
        ):
            summary.improving_legacy += 1

    def _observe_streamed_screening_fields(
        self,
        *,
        status: str,
        reason: str,
        negative_cache_hit: bool,
        exact_call_blocked: bool,
    ) -> None:
        summary = self._stream_summary
        if summary is None:
            raise RuntimeError("stream observer is unavailable")
        summary.counts[f"screening_status:{status}"] += 1
        summary.counts["screening_negative_cache_hit"] += int(negative_cache_hit)
        summary.counts["screening_exact_call_blocked"] += int(exact_call_blocked)
        if reason:
            summary.screening_reasons[reason] += 1

    def _validate_screening_route_dictionary(self) -> None:
        if (
            self.screening_config is not None
            and self.screening_config.enabled
            and not self.config.record_route_dictionary
        ):
            raise ValueError("Stage 3.1 screening requires record_route_dictionary=True")

    def _offset(self, value: float | None = None) -> float:
        return (time.perf_counter() if value is None else value) - self.started_at_perf

    def register_route(self, sequence: tuple[str, ...] | list[str]) -> str:
        values = tuple(sequence)
        key = canonical_route_key(values)
        if self.config.record_route_dictionary and self.config.stream_sink is None:
            existing = self.route_dictionary.get(key)
            if existing is not None and existing != values:
                raise RuntimeError(f"canonical route key collision for {key}")
            self.route_dictionary[key] = values
        return key

    def record_route_evaluation(
        self,
        sequence: tuple[str, ...] | list[str],
        *,
        lane: str,
        iteration: int | None,
        operator: str,
        kind: str,
        started_at: float | None = None,
        completed_at: float | None = None,
        exact_started: bool,
        exact_completed: bool,
        feasible: bool | None,
        failure_reason: str,
        labels_generated: int = 0,
        labels_expanded: int = 0,
        labels_pruned: int = 0,
        deadline_boundary: str = "",
        cache_key_digest: str = "",
        route_change_status: str = "unknown",
        status: str = "",
    ) -> int:
        if kind not in ROUTE_EVALUATION_KINDS:
            raise ValueError(f"unsupported route evaluation kind: {kind}")
        key = self.register_route(sequence)
        started = self._offset() if started_at is None else started_at
        completed = self._offset() if completed_at is None and exact_completed else completed_at
        duration = max(0.0, (completed - started) if completed is not None else 0.0)
        resolved_status = status
        if not resolved_status:
            if kind != "exact_call":
                resolved_status = kind
            elif exact_completed and feasible is True:
                resolved_status = "completed_feasible"
            elif exact_completed and feasible is False:
                resolved_status = "completed_infeasible"
            elif exact_completed:
                resolved_status = "completed_discarded"
            elif exact_started:
                resolved_status = "interrupted_deadline"
            else:
                resolved_status = "not_started"
        record = RouteEvaluationTrace(
            evaluation_id=len(self.route_evaluations) + 1,
            route_key=key,
            lane=lane,
            iteration=iteration,
            operator=operator,
            kind=kind,
            started_at=started,
            completed_at=completed,
            duration_seconds=duration,
            exact_started=exact_started,
            exact_completed=exact_completed,
            feasible=feasible,
            failure_reason=failure_reason,
            labels_generated=labels_generated,
            labels_expanded=labels_expanded,
            labels_pruned=labels_pruned,
            deadline_boundary=deadline_boundary,
            cache_key_digest=cache_key_digest,
            route_change_status=route_change_status,
            status=resolved_status,
        )
        self.route_evaluations.append(record)
        return record.evaluation_id

    def record_cache_event(
        self,
        *,
        operation: str,
        route_key: str,
        cache_key_digest: str,
        lane: str,
        iteration: int | None,
        operator: str,
        **fields: object,
    ) -> None:
        self.events.append(
            {
                "event_type": "cache_event",
                "operation": operation,
                "route_key": route_key,
                "cache_key_digest": cache_key_digest,
                "lane": lane,
                "iteration": iteration,
                "operator": operator,
                "timestamp_seconds": self._offset(),
                **fields,
            }
        )

    def record_incremental_propagation(
        self,
        *,
        operator: str,
        lane: str,
        iteration: int | None,
        base_sequence: tuple[str, ...],
        candidate_sequence: tuple[str, ...],
        status: str,
        reason: str,
        distance_lower_bound: float,
        min_time_window_slack: float,
        finish_time: float,
        reused_prefix_edges: int,
        reused_suffix_edges: int,
        recomputed_forward_edges: int,
        recomputed_backward_edges: int,
    ) -> None:
        base_key = self.register_route(base_sequence)
        candidate_key = self.register_route(candidate_sequence)
        self.incremental_propagations.append(
            {
                "propagation_id": len(self.incremental_propagations) + 1,
                "lane": lane,
                "iteration": iteration,
                "operator": operator,
                "base_route_key": base_key,
                "candidate_route_key": candidate_key,
                "status": status,
                "reason": reason,
                "distance_lower_bound": float(distance_lower_bound),
                "min_time_window_slack": float(min_time_window_slack),
                "finish_time": float(finish_time),
                "reused_prefix_edges": int(reused_prefix_edges),
                "reused_suffix_edges": int(reused_suffix_edges),
                "recomputed_forward_edges": int(recomputed_forward_edges),
                "recomputed_backward_edges": int(recomputed_backward_edges),
                "timestamp_seconds": self._offset(),
            }
        )

    def record_deadline_boundary(
        self,
        *,
        lane: str,
        iteration: int | None,
        operator: str,
        boundary: str,
        route_sequence: tuple[str, ...] | list[str] = (),
        exact_call_id: int | None = None,
        reason: str = "deadline boundary reached",
    ) -> None:
        route_keys = (self.register_route(route_sequence),) if route_sequence else ()
        self.events.append(
            {
                "event_type": "deadline_boundary",
                "timestamp_seconds": self._offset(),
                "lane": lane,
                "iteration": iteration,
                "operator": operator,
                "boundary": boundary,
                "reason": reason,
                "route_keys": route_keys,
                "exact_call_id": exact_call_id,
            }
        )

    def record_operator_call(
        self,
        *,
        lane: str,
        iteration: int | None,
        operator: str,
        statistics_group: str = "neighborhood_statistics",
    ) -> None:
        if not self.config.record_operator_events:
            return
        self.events.append(
            {
                "event_type": "operator_call",
                "timestamp_seconds": self._offset(),
                "lane": lane,
                "iteration": iteration,
                "operator": operator,
                "statistics_group": statistics_group,
            }
        )

    def record_candidate_state(
        self,
        *,
        lane: str,
        iteration: int | None,
        operator: str,
        current_sequences: tuple[tuple[str, ...], ...],
        candidate_sequences: tuple[tuple[str, ...], ...],
        current_objective_key: tuple[int, float, float, int] | tuple[()],
        candidate_objective_key: tuple[int, float, float, int] | tuple[()],
        candidate_feasible: bool,
        accepted: bool,
        global_best: bool,
        status: str,
        reason: str = "",
    ) -> None:
        if not self.config.record_candidate_states:
            return
        current_keys = tuple(self.register_route(route) for route in current_sequences)
        candidate_keys = tuple(self.register_route(route) for route in candidate_sequences)
        current_vehicle_count = len(current_sequences)
        candidate_vehicle_count = len(candidate_sequences)
        self.events.append(
            {
                "event_type": "candidate_state",
                "timestamp_seconds": self._offset(),
                "lane": lane,
                "iteration": iteration,
                "operator": operator,
                "status": status,
                "reason": reason,
                "current_route_keys": current_keys,
                "candidate_route_keys": candidate_keys,
                "candidate_feasible": candidate_feasible,
                "accepted": accepted,
                "global_best": global_best,
                "current_objective_key": current_objective_key,
                "candidate_objective_key": candidate_objective_key,
                "current_vehicle_count": current_vehicle_count,
                "candidate_vehicle_count": candidate_vehicle_count,
                "candidate_vehicle_delta": candidate_vehicle_count - current_vehicle_count,
            }
        )

    def record_execution_error(self, error: BaseException) -> None:
        self.events.append(
            {
                "event_type": "execution_error",
                "timestamp_seconds": self._offset(),
                "error_type": type(error).__name__,
                "reason": str(error),
            }
        )

    def record_screening_decision(
        self,
        sequence: tuple[str, ...] | list[str],
        *,
        lane: str,
        iteration: int | None,
        operator: str,
        status: str,
        first_failed_check: str,
        reason: str,
        checks: tuple[ScreeningCheckTrace, ...],
        demand: float,
        min_time_window_slack: float,
        distance_lower_bound: float,
        distance_increment_lower_bound: float | None,
        single_segment_reachable: bool,
        structural_energy_lower_bound: float,
        negative_cache_hit: bool,
        exact_call_blocked: bool,
        started_at: float | None = None,
        completed_at: float | None = None,
        registered_route_key: str | None = None,
    ) -> int:
        """Append one auditable Stage 3.1 screening decision.

        Screening decisions deliberately have their own sequence instead of
        being represented as a route evaluation.  A rejected sequence therefore
        cannot inflate exact-call or route-cache counters.
        """

        key = (
            self.register_route(sequence)
            if registered_route_key is None
            else registered_route_key
        )
        started = self._offset() if started_at is None else started_at
        completed = self._offset() if completed_at is None else completed_at
        decision_id = len(self.screening_decisions) + 1
        normalized_demand = float(demand)
        normalized_slack = float(min_time_window_slack)
        normalized_distance = float(distance_lower_bound)
        normalized_increment = (
            None
            if distance_increment_lower_bound is None
            else float(distance_increment_lower_bound)
        )
        normalized_reachable = bool(single_segment_reachable)
        normalized_energy = float(structural_energy_lower_bound)
        normalized_negative_hit = bool(negative_cache_hit)
        normalized_blocked = bool(exact_call_blocked)
        fast_append = getattr(self.config.stream_sink, "append_screening_fields", None)
        if callable(fast_append):
            summary = self._stream_summary
            if summary is None or not isinstance(
                self.screening_decisions, _ExternalizedTraceList
            ):
                raise RuntimeError("typed screening stream is unavailable")
            summary.counts["screening_decisions"] += 1
            self._observe_streamed_screening_fields(
                status=status,
                reason=reason,
                negative_cache_hit=normalized_negative_hit,
                exact_call_blocked=normalized_blocked,
            )
            fast_append(
                decision_id,
                key,
                lane,
                iteration,
                operator,
                started,
                completed,
                status,
                reason,
                normalized_demand,
                normalized_increment,
                normalized_distance,
                normalized_blocked,
                first_failed_check,
                normalized_slack,
                normalized_negative_hit,
                normalized_reachable,
                normalized_energy,
                checks,
            )
            self.screening_decisions.increment_external_count()
            return decision_id
        decision = ScreeningDecision(
            decision_id=decision_id,
            route_key=key,
            lane=lane,
            iteration=iteration,
            operator=operator,
            status=status,
            first_failed_check=first_failed_check,
            reason=reason,
            checks=checks,
            demand=normalized_demand,
            min_time_window_slack=normalized_slack,
            distance_lower_bound=normalized_distance,
            distance_increment_lower_bound=normalized_increment,
            single_segment_reachable=normalized_reachable,
            structural_energy_lower_bound=normalized_energy,
            negative_cache_hit=normalized_negative_hit,
            exact_call_blocked=normalized_blocked,
            started_at=started,
            completed_at=completed,
            duration_seconds=max(0.0, completed - started),
        )
        self.screening_decisions.append(decision)
        return decision.decision_id

    def finish(self, result: object | None = None) -> None:
        self.finished_at = self._offset()
        if result is not None:
            fields = (
                "charging_subproblem_calls",
                "cache_hits",
                "cache_misses",
                "unique_route_evaluations",
                "unique_route_semantics",
                "iterations",
                "effective_iterations",
                "exact_started_calls",
                "exact_completed_calls",
                "exact_interrupted_calls",
                "exact_budget_exhaustions",
                "termination_reason",
                "candidate_work_hash",
                "route_result_hash",
            )
            self.result_summary = {
                field: getattr(result, field) for field in fields if hasattr(result, field)
            }
            if hasattr(result, "screening_statistics"):
                self.result_summary["screening_statistics"] = cast(Any, result).screening_statistics
            if hasattr(result, "cache_incremental_statistics"):
                self.result_summary["cache_incremental_statistics"] = cast(
                    Any, result
                ).cache_incremental_statistics
            if hasattr(result, "candidate_control_statistics"):
                self.result_summary["candidate_control_statistics"] = cast(
                    Any, result
                ).candidate_control_statistics

    @property
    def started_calls(self) -> int:
        if self._stream_summary is not None:
            return int(self._stream_summary.counts["exact_started"])
        return sum(record.exact_started for record in self.route_evaluations)

    @property
    def completed_calls(self) -> int:
        if self._stream_summary is not None:
            return int(self._stream_summary.counts["exact_completed"])
        return sum(record.exact_completed for record in self.route_evaluations)

    @property
    def exact_calls(self) -> int:
        if self._stream_summary is not None:
            return int(self._stream_summary.counts["route_kind:exact_call"])
        return sum(record.kind == "exact_call" for record in self.route_evaluations)

    @property
    def cache_hits(self) -> int:
        if self._stream_summary is not None:
            return int(self._stream_summary.counts["route_kind:cache_hit"])
        return sum(record.kind == "cache_hit" for record in self.route_evaluations)

    @property
    def precomputed_routes(self) -> int:
        if self._stream_summary is not None:
            return int(self._stream_summary.counts["route_kind:precomputed_route"])
        return sum(record.kind == "precomputed_route" for record in self.route_evaluations)

    @property
    def deadline_events(self) -> int:
        if self._stream_summary is not None:
            return int(self._stream_summary.counts["event_type:deadline_boundary"])
        return sum(event.get("event_type") == "deadline_boundary" for event in self.events)

    @property
    def operator_call_counts(self) -> dict[str, int]:
        if self._stream_summary is not None:
            return dict(sorted(self._stream_summary.operator_calls.items()))
        counts: Counter[str] = Counter()
        for event in self.events:
            if event.get("event_type") == "operator_call":
                counts[str(event["operator"])] += 1
        return dict(sorted(counts.items()))

    @property
    def screening_counts(self) -> dict[str, object]:
        if self._stream_summary is not None:
            summary = self._stream_summary
            return {
                "screening_calls": int(summary.counts["screening_decisions"]),
                "screening_passes": int(summary.counts["screening_status:pass"]),
                "screening_rejections": int(summary.counts["screening_status:rejected"]),
                "screening_cache_hits": int(summary.counts["screening_negative_cache_hit"]),
                "screening_exact_call_blocked": int(summary.counts["screening_exact_call_blocked"]),
                "screening_reason_counts": dict(sorted(summary.screening_reasons.items())),
            }
        reason_counts: Counter[str] = Counter()
        for decision in self.screening_decisions:
            if decision.reason:
                reason_counts[decision.reason] += 1
        return {
            "screening_calls": len(self.screening_decisions),
            "screening_passes": sum(
                decision.status == "pass" for decision in self.screening_decisions
            ),
            "screening_rejections": sum(
                decision.status == "rejected" for decision in self.screening_decisions
            ),
            "screening_cache_hits": sum(
                decision.negative_cache_hit for decision in self.screening_decisions
            ),
            "screening_exact_call_blocked": sum(
                decision.exact_call_blocked for decision in self.screening_decisions
            ),
            "screening_reason_counts": dict(sorted(reason_counts.items())),
        }

    @property
    def cache_incremental_counts(self) -> dict[str, int]:
        if self._stream_summary is not None:
            streamed_counts = self._stream_summary.cache_operations
            return {
                "cache_lookups": streamed_counts["lookup"],
                "cache_hits": streamed_counts["hit"],
                "cache_misses": streamed_counts["miss"],
                "cache_stores": streamed_counts["store"],
                "cache_evictions": streamed_counts["evict"],
                "cache_oversize_not_cached": streamed_counts["oversize_not_cached"],
                "incremental_propagations": int(
                    self._stream_summary.counts["incremental_propagations"]
                ),
                "incremental_fallbacks": int(self._stream_summary.counts["incremental_fallback"]),
            }
        counts: Counter[str] = Counter()
        for event in self.events:
            if event.get("event_type") != "cache_event":
                continue
            operation = str(event.get("operation", ""))
            if operation == "lookup_result":
                counts["lookup"] += 1
                counts[str(event.get("lookup_result", ""))] += 1
            else:
                counts[operation] += 1
        return {
            "cache_lookups": counts["lookup"],
            "cache_hits": counts["hit"],
            "cache_misses": counts["miss"],
            "cache_stores": counts["store"],
            "cache_evictions": counts["evict"],
            "cache_oversize_not_cached": counts["oversize_not_cached"],
            "incremental_propagations": len(self.incremental_propagations),
            "incremental_fallbacks": sum(
                item.get("status") == "fallback" for item in self.incremental_propagations
            ),
        }

    def reconcile(self, result: _MeasuredResult) -> dict[str, object]:
        stream_sink = cast(Any, self.config.stream_sink)
        finish_stream = getattr(stream_sink, "finish", None)
        if callable(finish_stream):
            finish_stream()
        expected_calls = int(result.charging_subproblem_calls)
        expected_started_calls = int(
            getattr(result, "exact_started_calls", expected_calls) or expected_calls
        )
        expected_cache_hits = int(result.cache_hits)
        expected_unique = int(result.unique_route_evaluations)
        expected_cache_incremental = getattr(result, "cache_incremental_statistics", {})
        candidate_pending_cache_hits = (
            self._stream_summary.candidate_pending_cache_hits
            if self._stream_summary is not None
            else sum(
                event.get("event_type") == "cache_event"
                and event.get("operation") == "candidate_pending_hit"
                for event in self.events
            )
        )
        expected_route_evaluation_cache_hits = expected_cache_hits + (
            candidate_pending_cache_hits
            if isinstance(expected_cache_incremental, dict) and bool(expected_cache_incremental)
            else 0
        )
        unique_route_semantics = str(
            getattr(
                result,
                "unique_route_semantics",
                LEGACY_UNIQUE_ROUTE_SEMANTICS,
            )
        )
        supported_unique_route_semantics = unique_route_semantics in {
            LEGACY_UNIQUE_ROUTE_SEMANTICS,
            COMPLETED_UNIQUE_ROUTE_SEMANTICS,
        }
        legacy_started_unique_semantics = unique_route_semantics == LEGACY_UNIQUE_ROUTE_SEMANTICS
        external_semantics = (
            "legacy_started"
            if legacy_started_unique_semantics
            else "completed_shared"
            if (
                self.cache_incremental_config is not None
                and self.cache_incremental_config.shared_across_lanes
            )
            else "completed_lane"
        )
        external_unique_count = self._external_unique_route_count(external_semantics)
        if self._stream_summary is not None and external_unique_count is not None:
            observed_unique_routes = external_unique_count
        elif self._stream_summary is not None:
            if legacy_started_unique_semantics:
                observed_unique_routes = len(self._stream_summary.legacy_exact_route_keys)
            elif (
                self.cache_incremental_config is not None
                and self.cache_incremental_config.shared_across_lanes
            ):
                observed_unique_routes = len(self._stream_summary.completed_shared_route_keys)
            else:
                observed_unique_routes = len(self._stream_summary.completed_lane_route_keys)
        else:
            identity_records = (
                record
                for record in self.route_evaluations
                if record.kind == "exact_call"
                and (
                    record.exact_started
                    if legacy_started_unique_semantics
                    else record.exact_completed
                )
            )
            exact_route_keys: set[str | tuple[str, str]]
            if legacy_started_unique_semantics:
                exact_route_keys = {(record.lane, record.route_key) for record in identity_records}
            elif (
                self.cache_incremental_config is not None
                and self.cache_incremental_config.shared_across_lanes
            ):
                exact_route_keys = {record.route_key for record in identity_records}
            else:
                exact_route_keys = {
                    (
                        "legacy" if record.lane == "initialization" else record.lane,
                        record.route_key,
                    )
                    for record in identity_records
                }
            observed_unique_routes = len(exact_route_keys)
        operator_call_counts: dict[str, dict[str, int]] = {}
        expected_operator_calls: dict[str, dict[str, int]] = {}
        for group, statistics in (
            ("destroy_statistics", result.destroy_statistics),
            ("repair_statistics", result.repair_statistics),
            ("neighborhood_statistics", result.neighborhood_statistics),
        ):
            observed = (
                self._stream_summary.operator_calls_by_group.get(group, Counter()).copy()
                if self._stream_summary is not None
                else Counter(
                    str(event["operator"])
                    for event in self.events
                    if event.get("event_type") == "operator_call"
                    and event.get("statistics_group") == group
                )
            )
            expected = {
                str(name): int(cast(Any, values.get("calls", 0)))
                for name, values in statistics.items()
            }
            observed.update({name: 0 for name in expected if name not in observed})
            operator_call_counts[group] = dict(sorted(observed.items()))
            expected_operator_calls[group] = dict(sorted(expected.items()))

        if self._stream_summary is not None:
            legacy_candidate_state_count = self._stream_summary.legacy_candidate_states
            accepted_legacy = self._stream_summary.accepted_legacy
            rejected_legacy = self._stream_summary.rejected_legacy
            improving_legacy = self._stream_summary.improving_legacy
        else:
            legacy_candidate_states = [
                event
                for event in self.events
                if event.get("event_type") == "candidate_state" and event.get("lane") == "legacy"
            ]
            legacy_candidate_state_count = len(legacy_candidate_states)
            accepted_legacy = sum(
                event.get("accepted") is True for event in legacy_candidate_states
            )
            rejected_legacy = sum(
                event.get("accepted") is not True for event in legacy_candidate_states
            )
            improving_legacy = 0
            for event in legacy_candidate_states:
                if event.get("accepted") is not True:
                    continue
                current = _objective_from_key(event.get("current_objective_key"))
                candidate = _objective_from_key(event.get("candidate_objective_key"))
                if (
                    current is not None
                    and candidate is not None
                    and compare_objectives(candidate, current) is ObjectiveComparison.BETTER
                ):
                    improving_legacy += 1
        checks = {
            "started_calls_not_less_than_completed": self.started_calls >= self.completed_calls,
            "completed_calls_equal_result": self.completed_calls == expected_calls,
            "exact_calls_equal_result": self.exact_calls == expected_started_calls,
            "unique_route_semantics_supported": supported_unique_route_semantics,
            "route_evaluation_cache_hits_equal_result": self.cache_hits
            == expected_route_evaluation_cache_hits,
            "unique_routes_equal_result": observed_unique_routes == expected_unique,
            "operator_calls_equal_result": operator_call_counts == expected_operator_calls,
            "legacy_candidate_states_equal_effective_iterations": legacy_candidate_state_count
            == int(getattr(result, "effective_iterations", 0)),
            "accepted_moves_equal_result": accepted_legacy == int(result.accepted_moves),
            "rejected_moves_equal_result": rejected_legacy == int(result.rejected_moves),
            "improving_moves_equal_result": improving_legacy == int(result.improving_moves),
        }
        expected_screening = getattr(result, "screening_statistics", {})
        if isinstance(expected_screening, dict) and expected_screening:
            observed_screening = self.screening_counts
            observed_screening = cast(dict[str, Any], observed_screening)
            expected_screening = cast(dict[str, Any], expected_screening)
            for field_name in (
                "screening_calls",
                "screening_passes",
                "screening_rejections",
                "screening_cache_hits",
                "screening_exact_call_blocked",
            ):
                checks[f"{field_name}_equal_result"] = int(
                    cast(Any, observed_screening.get(field_name, 0))
                ) == int(cast(Any, expected_screening.get(field_name, 0)))
            checks["screening_reason_counts_equal_result"] = observed_screening.get(
                "screening_reason_counts", {}
            ) == expected_screening.get("screening_reason_counts", {})
        if isinstance(expected_cache_incremental, dict) and expected_cache_incremental:
            observed_cache_incremental = self.cache_incremental_counts
            expected_cache_incremental = cast(dict[str, Any], expected_cache_incremental)
            for field_name in (
                "cache_lookups",
                "cache_hits",
                "cache_misses",
                "cache_stores",
                "cache_evictions",
                "cache_oversize_not_cached",
                "incremental_propagations",
                "incremental_fallbacks",
            ):
                checks[f"cache_incremental_{field_name}_equal_result"] = int(
                    observed_cache_incremental.get(field_name, 0)
                ) == int(expected_cache_incremental.get(field_name, 0))
        return {
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "observed": {
                "started_calls": self.started_calls,
                "completed_calls": self.completed_calls,
                "exact_calls": self.exact_calls,
                "cache_hits": self.cache_hits,
                "candidate_pending_cache_hits": candidate_pending_cache_hits,
                "result_cache_hits": expected_cache_hits,
                "precomputed_routes": self.precomputed_routes,
                "unique_route_evaluations": observed_unique_routes,
                "unique_route_semantics": unique_route_semantics,
                "operator_calls": operator_call_counts,
                "legacy_candidate_states": legacy_candidate_state_count,
                "accepted_moves": accepted_legacy,
                "rejected_moves": rejected_legacy,
                "improving_moves": improving_legacy,
                "screening": self.screening_counts,
                "cache_incremental": self.cache_incremental_counts,
            },
            "expected": {
                "charging_subproblem_calls": expected_calls,
                "cache_hits": expected_route_evaluation_cache_hits,
                "candidate_pending_cache_hits": candidate_pending_cache_hits,
                "result_cache_hits": expected_cache_hits,
                "unique_route_evaluations": expected_unique,
                "unique_route_semantics": unique_route_semantics,
                "operator_calls": expected_operator_calls,
                "accepted_moves": int(result.accepted_moves),
                "rejected_moves": int(result.rejected_moves),
                "improving_moves": int(result.improving_moves),
                "screening": expected_screening,
                "cache_incremental": expected_cache_incremental,
            },
        }

    def to_dict(self) -> dict[str, object]:
        if self._stream_summary is not None:
            raise RuntimeError(
                "full trace rows were externalized during solve; use to_index_dict() and "
                "the persisted Parquet stream"
            )
        return {
            "schema_version": self.config.schema_version,
            "config": _measurement_config_payload(self.config),
            "route_dictionary": {
                key: list(sequence) for key, sequence in sorted(self.route_dictionary.items())
            },
            "route_evaluations": [asdict(record) for record in self.route_evaluations],
            "events": list(self.events),
            "summary": {
                "started_calls": self.started_calls,
                "completed_calls": self.completed_calls,
                "exact_calls": self.exact_calls,
                "cache_hits": self.cache_hits,
                "precomputed_routes": self.precomputed_routes,
                "deadline_events": self.deadline_events,
                "interrupted_calls": sum(
                    record.status == "interrupted_deadline" for record in self.route_evaluations
                ),
                "budget_exhaustions": sum(
                    event.get("event_type") == "exact_budget_boundary" for event in self.events
                ),
                "route_evaluation_count": len(self.route_evaluations),
                "operator_call_counts": self.operator_call_counts,
                "screening": self.screening_counts,
            },
            "screening_config": (
                asdict(self.screening_config) if self.screening_config is not None else None
            ),
            "trace_schema_version": self.trace_schema_version,
            "cache_incremental_config": (
                asdict(self.cache_incremental_config)
                if self.cache_incremental_config is not None
                else None
            ),
            "cache_incremental_summary": self.cache_incremental_counts,
            "exact_deadline_config": (
                asdict(self.exact_deadline_config)
                if self.exact_deadline_config is not None
                else None
            ),
            "candidate_control_config": (
                asdict(self.candidate_control_config)
                if self.candidate_control_config is not None
                else None
            ),
            "incremental_propagations": list(self.incremental_propagations),
            "screening_decisions": [asdict(decision) for decision in self.screening_decisions],
            "finished_at": self.finished_at,
            "result_summary": dict(self.result_summary),
        }

    def to_index_dict(self) -> dict[str, object]:
        """Return scalar trace metadata without copying append-only event lists."""

        interrupted_calls = (
            int(self._stream_summary.counts["interrupted"])
            if self._stream_summary is not None
            else sum(record.status == "interrupted_deadline" for record in self.route_evaluations)
        )
        budget_exhaustions = (
            int(self._stream_summary.counts["event_type:exact_budget_boundary"])
            if self._stream_summary is not None
            else sum(event.get("event_type") == "exact_budget_boundary" for event in self.events)
        )
        return {
            "config": _measurement_config_payload(self.config),
            "summary": {
                "started_calls": self.started_calls,
                "completed_calls": self.completed_calls,
                "exact_calls": self.exact_calls,
                "cache_hits": self.cache_hits,
                "precomputed_routes": self.precomputed_routes,
                "deadline_events": self.deadline_events,
                "interrupted_calls": interrupted_calls,
                "budget_exhaustions": budget_exhaustions,
                "route_evaluation_count": len(self.route_evaluations),
                "operator_call_counts": self.operator_call_counts,
                "screening": self.screening_counts,
            },
            "result_summary": dict(self.result_summary),
            "streamed_record_counts": self.streamed_record_counts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Stage03Trace:
        trace_schema_version = str(payload.get("trace_schema_version", TRACE_SCHEMA_VERSION))
        config_payload = payload.get("config", {})
        config = MeasurementConfig(**config_payload)
        trace = cls(config)
        trace.trace_schema_version = trace_schema_version
        trace.route_dictionary = {
            str(key): tuple(str(value) for value in values)
            for key, values in dict(payload.get("route_dictionary", {})).items()
        }
        trace.route_evaluations = [
            RouteEvaluationTrace(
                evaluation_id=int(item["evaluation_id"]),
                route_key=str(item["route_key"]),
                lane=str(item["lane"]),
                iteration=(None if item.get("iteration") is None else int(item["iteration"])),
                operator=str(item["operator"]),
                kind=str(item["kind"]),
                started_at=float(item["started_at"]),
                completed_at=(
                    None if item.get("completed_at") is None else float(item["completed_at"])
                ),
                duration_seconds=float(item["duration_seconds"]),
                exact_started=bool(item["exact_started"]),
                exact_completed=bool(item["exact_completed"]),
                feasible=(None if item.get("feasible") is None else bool(item["feasible"])),
                failure_reason=str(item.get("failure_reason", "")),
                labels_generated=int(item.get("labels_generated", 0)),
                labels_expanded=int(item.get("labels_expanded", 0)),
                labels_pruned=int(item.get("labels_pruned", 0)),
                deadline_boundary=str(item.get("deadline_boundary", "")),
                cache_key_digest=str(item.get("cache_key_digest", "")),
                route_change_status=str(item.get("route_change_status", "unknown")),
                status=str(item.get("status", "")),
            )
            for item in payload.get("route_evaluations", [])
        ]
        trace.events = [dict(event) for event in payload.get("events", [])]
        trace.finished_at = (
            None if payload.get("finished_at") is None else float(payload["finished_at"])
        )
        trace.result_summary = dict(payload.get("result_summary", {}))
        screening_payload = payload.get("screening_config")
        if isinstance(screening_payload, dict):
            trace.screening_config = CheapScreeningConfig(**screening_payload)
            trace._validate_screening_route_dictionary()
        cache_payload = payload.get("cache_incremental_config")
        if isinstance(cache_payload, dict):
            from evrptw.cache_incremental import CacheIncrementalConfig

            trace.cache_incremental_config = CacheIncrementalConfig(**cache_payload)
            trace.trace_schema_version = CACHE_INCREMENTAL_TRACE_SCHEMA_VERSION
        exact_deadline_payload = payload.get("exact_deadline_config")
        if isinstance(exact_deadline_payload, dict):
            trace.exact_deadline_config = ExactDeadlineConfig(**exact_deadline_payload)
            trace.trace_schema_version = EXACT_DEADLINE_TRACE_SCHEMA_VERSION
        candidate_control_payload = payload.get("candidate_control_config")
        if isinstance(candidate_control_payload, dict):
            trace.candidate_control_config = CandidateControlConfig(**candidate_control_payload)
            trace.trace_schema_version = CANDIDATE_CONTROL_TRACE_SCHEMA_VERSION
        trace.incremental_propagations = [
            dict(item) for item in payload.get("incremental_propagations", [])
        ]
        trace.screening_decisions = [
            ScreeningDecision(
                decision_id=int(item["decision_id"]),
                route_key=str(item["route_key"]),
                lane=str(item["lane"]),
                iteration=(None if item.get("iteration") is None else int(item["iteration"])),
                operator=str(item["operator"]),
                status=str(item["status"]),
                first_failed_check=str(item.get("first_failed_check", "")),
                reason=str(item.get("reason", "")),
                checks=tuple(
                    ScreeningCheckTrace(
                        check=str(check["check"]),
                        status=str(check["status"]),
                        value=(
                            None
                            if check.get("value") is None
                            else bool(check["value"])
                            if isinstance(check.get("value"), bool)
                            else float(check["value"])
                        ),
                        reason=str(check.get("reason", "")),
                    )
                    for check in item.get("checks", [])
                ),
                demand=float(item.get("demand", 0.0)),
                min_time_window_slack=float(item.get("min_time_window_slack", 0.0)),
                distance_lower_bound=float(item.get("distance_lower_bound", 0.0)),
                distance_increment_lower_bound=(
                    None
                    if item.get("distance_increment_lower_bound") is None
                    else float(item["distance_increment_lower_bound"])
                ),
                single_segment_reachable=bool(item.get("single_segment_reachable", False)),
                structural_energy_lower_bound=float(item.get("structural_energy_lower_bound", 0.0)),
                negative_cache_hit=bool(item.get("negative_cache_hit", False)),
                exact_call_blocked=bool(item.get("exact_call_blocked", False)),
                started_at=float(item.get("started_at", 0.0)),
                completed_at=float(item.get("completed_at", 0.0)),
                duration_seconds=float(item.get("duration_seconds", 0.0)),
            )
            for item in payload.get("screening_decisions", [])
        ]
        return trace


class Stage03ExecutionError(RuntimeError):
    """A measured solve failed after its partial trace was captured."""

    def __init__(self, trace: Stage03Trace, original: BaseException) -> None:
        self.trace = trace
        self.original = original
        super().__init__(f"measured ALNS execution failed: {original}")


def route_result_fields(result: _ChargingResult) -> dict[str, Any]:
    """Extract the stable charging result fields used by the raw trace."""

    return {
        "feasible": bool(result.feasible),
        "failure_reason": str(result.failure_reason),
        "labels_generated": int(result.labels_generated),
        "labels_expanded": int(result.labels_expanded),
        "labels_pruned": int(result.labels_pruned),
    }


def _objective_from_key(value: object) -> SolutionObjective | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return SolutionObjective(
            int(value[0]),
            float(value[1]),
            float(value[2]),
            int(value[3]),
        )
    except (TypeError, ValueError):
        return None


def instance_route_key(instance: Instance, route: tuple[str, ...]) -> str:
    """Keep a typed seam for auditors that need to validate route membership."""

    unknown = [name for name in route if name not in instance.by_name]
    if unknown:
        raise ValueError(f"route contains unknown nodes: {unknown}")
    return canonical_route_key(route)
