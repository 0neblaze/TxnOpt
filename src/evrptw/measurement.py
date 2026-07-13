from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from evrptw.models import Instance


TRACE_SCHEMA_VERSION = "stage03-trace-v1"
ROUTE_EVALUATION_KINDS = frozenset(
    {"exact_call", "cache_hit", "precomputed_route"}
)


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

    def __post_init__(self) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Stage 3.0 trace schema {self.schema_version}; "
                f"expected {TRACE_SCHEMA_VERSION}"
            )


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

    def __post_init__(self) -> None:
        if self.kind not in ROUTE_EVALUATION_KINDS:
            raise ValueError(f"unsupported route evaluation kind: {self.kind}")
        if self.evaluation_id <= 0:
            raise ValueError("evaluation_id must be positive")
        if self.exact_completed and not self.exact_started:
            raise ValueError("an exact call cannot complete before it starts")


class _MeasuredResult(Protocol):
    @property
    def charging_subproblem_calls(self) -> int: ...

    @property
    def cache_hits(self) -> int: ...

    @property
    def unique_route_evaluations(self) -> int: ...

    @property
    def neighborhood_statistics(self) -> dict[str, dict[str, object]]: ...


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

    def _offset(self, value: float | None = None) -> float:
        return (time.perf_counter() if value is None else value) - self.started_at_perf

    def register_route(self, sequence: tuple[str, ...] | list[str]) -> str:
        values = tuple(sequence)
        key = canonical_route_key(values)
        if self.config.record_route_dictionary:
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
    ) -> int:
        if kind not in ROUTE_EVALUATION_KINDS:
            raise ValueError(f"unsupported route evaluation kind: {kind}")
        key = self.register_route(sequence)
        started = self._offset() if started_at is None else started_at
        completed = (
            self._offset() if completed_at is None and exact_completed else completed_at
        )
        duration = max(0.0, (completed - started) if completed is not None else 0.0)
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
        )
        self.route_evaluations.append(record)
        return record.evaluation_id

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

    def finish(self, result: object | None = None) -> None:
        self.finished_at = self._offset()
        if result is not None:
            fields = (
                "charging_subproblem_calls",
                "cache_hits",
                "cache_misses",
                "unique_route_evaluations",
                "iterations",
                "effective_iterations",
            )
            self.result_summary = {
                field: getattr(result, field)
                for field in fields
                if hasattr(result, field)
            }

    @property
    def started_calls(self) -> int:
        return sum(record.exact_started for record in self.route_evaluations)

    @property
    def completed_calls(self) -> int:
        return sum(record.exact_completed for record in self.route_evaluations)

    @property
    def exact_calls(self) -> int:
        return sum(record.kind == "exact_call" for record in self.route_evaluations)

    @property
    def cache_hits(self) -> int:
        return sum(record.kind == "cache_hit" for record in self.route_evaluations)

    @property
    def precomputed_routes(self) -> int:
        return sum(record.kind == "precomputed_route" for record in self.route_evaluations)

    @property
    def deadline_events(self) -> int:
        return sum(event.get("event_type") == "deadline_boundary" for event in self.events)

    @property
    def operator_call_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for event in self.events:
            if event.get("event_type") == "operator_call":
                counts[str(event["operator"])] += 1
        return dict(sorted(counts.items()))

    def reconcile(self, result: _MeasuredResult) -> dict[str, object]:
        expected_calls = int(result.charging_subproblem_calls)
        expected_cache_hits = int(result.cache_hits)
        expected_unique = int(result.unique_route_evaluations)
        expected_statistics = result.neighborhood_statistics
        observed_statistics = Counter(
            str(event["operator"])
            for event in self.events
            if event.get("event_type") == "operator_call"
            and event.get("statistics_group") == "neighborhood_statistics"
        )
        expected_operator_calls = {
            str(name): int(cast(Any, values.get("calls", 0)))
            for name, values in expected_statistics.items()
        }
        observed_statistics.update(
            {name: 0 for name in expected_operator_calls if name not in observed_statistics}
        )
        exact_route_keys = {
            (record.lane, record.route_key)
            for record in self.route_evaluations
            if record.kind == "exact_call"
        }
        checks = {
            "started_calls_not_less_than_completed": self.started_calls >= self.completed_calls,
            "completed_calls_equal_result": self.completed_calls == expected_calls,
            "exact_calls_equal_result": self.exact_calls == expected_calls,
            "cache_hits_equal_result": self.cache_hits == expected_cache_hits,
            "unique_routes_equal_result": len(exact_route_keys) == expected_unique,
            "operator_calls_equal_result": dict(sorted(observed_statistics.items()))
            == dict(sorted(expected_operator_calls.items())),
        }
        return {
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "observed": {
                "started_calls": self.started_calls,
                "completed_calls": self.completed_calls,
                "exact_calls": self.exact_calls,
                "cache_hits": self.cache_hits,
                "precomputed_routes": self.precomputed_routes,
                "unique_route_evaluations": len(exact_route_keys),
                "operator_calls": dict(sorted(observed_statistics.items())),
            },
            "expected": {
                "charging_subproblem_calls": expected_calls,
                "cache_hits": expected_cache_hits,
                "unique_route_evaluations": expected_unique,
                "operator_calls": expected_operator_calls,
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.config.schema_version,
            "config": asdict(self.config),
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
                "route_evaluation_count": len(self.route_evaluations),
                "operator_call_counts": self.operator_call_counts,
            },
            "finished_at": self.finished_at,
            "result_summary": dict(self.result_summary),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Stage03Trace:
        config_payload = payload.get("config", {})
        config = MeasurementConfig(**config_payload)
        trace = cls(config)
        trace.route_dictionary = {
            str(key): tuple(str(value) for value in values)
            for key, values in dict(payload.get("route_dictionary", {})).items()
        }
        trace.route_evaluations = [
            RouteEvaluationTrace(
                evaluation_id=int(item["evaluation_id"]),
                route_key=str(item["route_key"]),
                lane=str(item["lane"]),
                iteration=(
                    None if item.get("iteration") is None else int(item["iteration"])
                ),
                operator=str(item["operator"]),
                kind=str(item["kind"]),
                started_at=float(item["started_at"]),
                completed_at=(
                    None
                    if item.get("completed_at") is None
                    else float(item["completed_at"])
                ),
                duration_seconds=float(item["duration_seconds"]),
                exact_started=bool(item["exact_started"]),
                exact_completed=bool(item["exact_completed"]),
                feasible=(
                    None if item.get("feasible") is None else bool(item["feasible"])
                ),
                failure_reason=str(item.get("failure_reason", "")),
                labels_generated=int(item.get("labels_generated", 0)),
                labels_expanded=int(item.get("labels_expanded", 0)),
                labels_pruned=int(item.get("labels_pruned", 0)),
                deadline_boundary=str(item.get("deadline_boundary", "")),
            )
            for item in payload.get("route_evaluations", [])
        ]
        trace.events = [dict(event) for event in payload.get("events", [])]
        trace.finished_at = (
            None if payload.get("finished_at") is None else float(payload["finished_at"])
        )
        trace.result_summary = dict(payload.get("result_summary", {}))
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


def instance_route_key(instance: Instance, route: tuple[str, ...]) -> str:
    """Keep a typed seam for auditors that need to validate route membership."""

    unknown = [name for name in route if name not in instance.by_name]
    if unknown:
        raise ValueError(f"route contains unknown nodes: {unknown}")
    return canonical_route_key(route)
