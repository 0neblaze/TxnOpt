from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evrptw import artifacts as artifact_module
from evrptw.alns import _NeighborhoodEventStream
from evrptw.artifacts import (
    ArtifactBundleWriter,
    ArtifactIntegrityError,
    ArtifactReader,
    ArtifactRunContext,
    ArtifactStorageConfig,
    DeferredRouteEvaluation,
    DeferredScreeningDecision,
)
from evrptw.experiments import stage052_performance
from evrptw.experiments.stage052_performance_review import (
    replay_stage052_storage_semantics,
)
from evrptw.measurement import (
    MeasurementConfig,
    RouteEvaluationTrace,
    ScreeningCheckTrace,
    ScreeningDecision,
    Stage03Trace,
)


class _RecordingTraceSink:
    def __init__(self) -> None:
        self.records: list[tuple[str, object]] = []

    def append_route_evaluation(self, record: RouteEvaluationTrace) -> None:
        self.records.append(("route_evaluation", record))

    def append_event(self, event: Mapping[str, object]) -> None:
        self.records.append(("event", event))

    def append_screening_decision(self, decision: ScreeningDecision) -> None:
        self.records.append(("screening_decision", decision))

    def append_incremental_propagation(self, propagation: Mapping[str, object]) -> None:
        self.records.append(("incremental_propagation", propagation))


class _BufferedScreeningRecordingShard:
    screening_schema_version = "screening_decisions_v3"
    supports_buffered_screening_decisions = True

    def __init__(self) -> None:
        self.rows: list[object] = []

    @property
    def scratch_directory(self) -> Path:
        raise AssertionError("screening-only test must not use scratch storage")

    def append(
        self,
        *,
        route_dictionary: Mapping[str, object],
        critical_events: object,
        diagnostic_rows: object = (),
        cache_lookups_coalesced: bool = False,
    ) -> int:
        del route_dictionary, diagnostic_rows, cache_lookups_coalesced
        rows = list(critical_events)  # type: ignore[arg-type]
        self.rows.extend(rows)
        return len(rows)


def test_measurement_stream_sink_receives_each_family_without_retaining_rows() -> None:
    sink = _RecordingTraceSink()
    trace = Stage03Trace(MeasurementConfig(stream_sink=sink))

    trace.record_route_evaluation(
        ("C1",),
        lane="legacy",
        iteration=1,
        operator="repair",
        kind="exact_call",
        started_at=0.1,
        completed_at=0.2,
        exact_started=True,
        exact_completed=True,
        feasible=True,
        failure_reason="",
    )
    trace.events.append(
        {
            "event_type": "operator_call",
            "lane": "legacy",
            "iteration": 1,
            "operator": "repair",
            "statistics_group": "repair_statistics",
            "timestamp_seconds": 0.3,
        }
    )
    trace.record_screening_decision(
        ("C1",),
        lane="legacy",
        iteration=1,
        operator="repair",
        status="pass",
        first_failed_check="",
        reason="",
        checks=(ScreeningCheckTrace("capacity", "pass", True),),
        demand=1.0,
        min_time_window_slack=2.0,
        distance_lower_bound=3.0,
        distance_increment_lower_bound=None,
        single_segment_reachable=True,
        structural_energy_lower_bound=4.0,
        negative_cache_hit=False,
        exact_call_blocked=False,
        started_at=0.4,
        completed_at=0.5,
    )
    trace.record_incremental_propagation(
        operator="repair",
        lane="legacy",
        iteration=1,
        base_sequence=("C1",),
        candidate_sequence=("C1", "C2"),
        status="incremental",
        reason="",
        distance_lower_bound=1.0,
        min_time_window_slack=2.0,
        finish_time=3.0,
        reused_prefix_edges=1,
        reused_suffix_edges=0,
        recomputed_forward_edges=1,
        recomputed_backward_edges=1,
    )

    assert [family for family, _ in sink.records] == [
        "route_evaluation",
        "event",
        "screening_decision",
        "incremental_propagation",
    ]
    assert trace.streamed_record_counts == {
        "route_evaluations": 1,
        "events": 1,
        "screening_decisions": 1,
        "incremental_propagations": 1,
    }
    assert trace.started_calls == 1
    assert trace.completed_calls == 1
    assert trace.exact_calls == 1
    assert trace.operator_call_counts == {"repair": 1}
    assert trace.screening_counts["screening_calls"] == 1
    assert trace.cache_incremental_counts["incremental_propagations"] == 1
    reconciliation = trace.reconcile(
        SimpleNamespace(
            charging_subproblem_calls=1,
            exact_started_calls=1,
            cache_hits=0,
            unique_route_evaluations=1,
            unique_route_semantics="started_lane_identity_legacy_v1",
            destroy_statistics={},
            repair_statistics={"repair": {"calls": 1}},
            neighborhood_statistics={},
            effective_iterations=0,
            accepted_moves=0,
            rejected_moves=0,
            improving_moves=0,
            screening_statistics={
                "screening_calls": 1,
                "screening_passes": 1,
                "screening_rejections": 0,
                "screening_cache_hits": 0,
                "screening_exact_call_blocked": 0,
                "screening_reason_counts": {},
            },
            cache_incremental_statistics={
                "cache_lookups": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_stores": 0,
                "cache_evictions": 0,
                "cache_oversize_not_cached": 0,
                "incremental_propagations": 1,
                "incremental_fallbacks": 0,
            },
        )
    )
    assert reconciliation["status"] == "pass"
    with pytest.raises(RuntimeError, match="externalized"):
        list(trace.route_evaluations)


def test_default_measurement_trace_still_retains_historical_lists() -> None:
    trace = Stage03Trace(MeasurementConfig())
    trace.events.append({"event_type": "execution_error", "reason": "test"})

    assert trace.events == [{"event_type": "execution_error", "reason": "test"}]
    assert trace.streamed_record_counts == {}
    assert trace.to_dict()["events"] == trace.events


def test_stream_sink_is_runtime_only_and_never_serialized() -> None:
    trace = Stage03Trace(MeasurementConfig(stream_sink=_RecordingTraceSink()))

    index = trace.to_index_dict()

    assert "stream_sink" not in index["config"]


class _RecordingShard:
    def __init__(self, screening_schema_version: str = "screening_decisions_v3") -> None:
        self.events: list[dict[str, object]] = []
        self.flushes = 0
        self.append_calls = 0
        self._screening_schema_version = screening_schema_version
        self._scratch = tempfile.TemporaryDirectory()

    @property
    def screening_schema_version(self) -> str:
        return self._screening_schema_version

    @property
    def scratch_directory(self) -> Path:
        return Path(self._scratch.name)

    def append(
        self,
        *,
        route_dictionary: Mapping[str, object],
        critical_events: object,
        diagnostic_rows: object = (),
        cache_lookups_coalesced: bool = False,
    ) -> int:
        del route_dictionary, diagnostic_rows, cache_lookups_coalesced
        self.append_calls += 1
        rows = [
            {
                "benchmark_axis": row.axis_name,
                "record_type": row.marker,
                "kind": row.values[5],
            }
            if isinstance(row, DeferredRouteEvaluation)
            else dict(row)
            for row in critical_events  # type: ignore[union-attr]
        ]
        self.events.extend(rows)
        return len(rows)

    def flush(self) -> None:
        self.flushes += 1


def test_stage052_trace_sink_appends_to_open_shard_before_solver_returns() -> None:
    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="wall_clock_300",
    )
    trace = Stage03Trace(MeasurementConfig(stream_sink=sink))

    trace.record_route_evaluation(
        ("C1",),
        lane="legacy",
        iteration=1,
        operator="repair",
        kind="exact_call",
        started_at=0.1,
        completed_at=0.2,
        exact_started=True,
        exact_completed=True,
        feasible=True,
        failure_reason="",
    )

    sink.finish()
    assert len(shard.events) == 1
    assert shard.events[0]["benchmark_axis"] == "wall_clock_300"
    assert shard.events[0]["record_type"] == "route_evaluation"
    assert sink.event_count == 1
    assert sink.persisted_family_counts["route_evaluations"] == 1

    sink.append_neighborhood_event(
        {
            "event_type": "neighborhood_event",
            "lane": "legacy",
            "iteration": 1,
            "operator": "relocate",
            "status": "accepted",
        }
    )
    sink.finish()

    assert len(shard.events) == 2
    assert shard.events[-1]["record_type"] == "neighborhood_event"
    assert sink.persisted_family_counts["events"] == 1


def test_repeated_route_timings_and_exact_calls_are_retained() -> None:
    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="fixed_work",
        buffer_rows=1,
    )
    common = {
        "evaluation_id": 1,
        "route_key": "route:2:C1",
        "lane": "legacy",
        "iteration": 1,
        "operator": "repair",
        "started_at": 0.1,
        "completed_at": 0.2,
        "duration_seconds": 0.1,
        "exact_started": False,
        "exact_completed": False,
        "feasible": True,
        "failure_reason": "",
    }
    sink.append_route_evaluation(RouteEvaluationTrace(**common, kind="cache_hit"))
    sink.append_route_evaluation(
        RouteEvaluationTrace(
            **{**common, "evaluation_id": 2, "exact_started": True, "exact_completed": True},
            kind="exact_call",
        )
    )

    assert [event["kind"] for event in shard.events] == ["cache_hit", "exact_call"]


def test_stage052_stream_counts_follow_coalesced_physical_events() -> None:
    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="fixed_work",
    )
    common = {
        "event_type": "cache_event",
        "lane": "legacy",
        "iteration": 1,
        "operator": "repair",
        "route_key": "route:2:C1",
        "cache_key_digest": "a" * 64,
    }

    sink.append_event({**common, "operation": "lookup"})
    sink.append_event({**common, "operation": "hit"})
    sink.finish()

    assert len(shard.events) == 1
    assert shard.events[0]["operation"] == "lookup_result"
    assert sink.persisted_family_counts == {
        "events": 1,
        "incremental_propagations": 0,
        "route_evaluations": 0,
        "screening_decisions": 0,
    }


def test_neighborhood_rejections_and_failures_are_retained_in_order() -> None:
    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="fixed_work",
        buffer_rows=1,
    )

    sink.append_neighborhood_event(
        {
            "event_type": "neighborhood_event",
            "lane": "legacy",
            "operator": "relocate",
            "status": "prefilter_rejected",
            "reason": "forward_time_window_prefilter",
        }
    )
    sink.append_neighborhood_event(
        {
            "event_type": "neighborhood_event",
            "lane": "legacy",
            "operator": "relocate",
            "status": "failed",
            "reason": "evaluation_budget_exhausted",
        }
    )
    sink.finish()

    assert [event["status"] for event in shard.events] == [
        "prefilter_rejected",
        "failed",
    ]


def test_streamed_screening_definition_matches_mapping_normalization() -> None:
    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="fixed_work",
        buffer_rows=1,
    )
    decision = ScreeningDecision(
        decision_id=1,
        route_key="route:2:C1",
        lane="legacy",
        iteration=1,
        operator="repair",
        status="rejected",
        first_failed_check="capacity",
        reason="capacity",
        checks=(
            ScreeningCheckTrace("structure", "pass", True),
            ScreeningCheckTrace("capacity", "fail", 2.5, "capacity"),
        ),
        demand=2.5,
        min_time_window_slack=1.0,
        distance_lower_bound=3.0,
        distance_increment_lower_bound=None,
        single_segment_reachable=True,
        structural_energy_lower_bound=4.0,
        negative_cache_hit=False,
        exact_call_blocked=True,
        started_at=0.1,
        completed_at=0.2,
        duration_seconds=0.1,
    )

    sink.append_screening_decision(decision)
    streamed = shard.events[0]
    normalized = artifact_module._normalise_screening_definition(  # noqa: SLF001
        streamed,
        lane_id=11,
        operator_id=12,
        route_id=13,
    )

    assert normalized["status"] == decision.status
    assert normalized["reason"] == decision.reason
    assert normalized["benchmark_axis"] == "fixed_work"
    assert normalized["demand"] == decision.demand
    assert normalized["checks"] == [
        {
            "check": check.check,
            "status": check.status,
            "value_bool": check.value if isinstance(check.value, bool) else None,
            "value_float": (
                float(check.value)
                if isinstance(check.value, (int, float)) and not isinstance(check.value, bool)
                else None
            ),
            "value_text": check.value if isinstance(check.value, str) else None,
            "reason": check.reason,
        }
        for check in decision.checks
    ]


def test_v3_screening_bridge_flattens_decision_for_native_writer() -> None:
    shard = _BufferedScreeningRecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="fixed_work",
        buffer_rows=2,
    )
    decision = ScreeningDecision(
        decision_id=1,
        route_key="route:2:C1",
        lane="legacy",
        iteration=1,
        operator="repair",
        status="rejected",
        first_failed_check="capacity",
        reason="capacity",
        checks=(ScreeningCheckTrace("capacity", "fail", 2.5, "capacity"),),
        demand=2.5,
        min_time_window_slack=1.0,
        distance_lower_bound=3.0,
        distance_increment_lower_bound=None,
        single_segment_reachable=True,
        structural_energy_lower_bound=4.0,
        negative_cache_hit=False,
        exact_call_blocked=True,
        started_at=0.1,
        completed_at=0.2,
        duration_seconds=0.1,
    )

    sink.append_screening_decision(decision)
    sink.append_screening_decision(replace(decision, decision_id=2))

    assert len(shard.rows) == 2
    first, second = shard.rows
    assert isinstance(first, DeferredScreeningDecision)
    assert isinstance(second, DeferredScreeningDecision)
    assert first.axis_name == "fixed_work"
    assert first.values[0] == 1
    assert first.values[1] == decision.route_key
    assert second.values[0] == 2


def test_native_v3_screening_packer_preserves_positions_and_cache_identity() -> None:
    from evrptw import _core

    deferred = DeferredScreeningDecision(
        axis_name="fixed_work",
        values=(
            7,
            "route:2:C1",
            "legacy",
            3,
            "repair",
            0.1,
            0.2,
            "rejected",
            "capacity",
            2.5,
            None,
            3.0,
            True,
            "capacity",
            1.0,
            False,
            True,
            4.0,
            (("capacity", "fail", 2.5, "capacity"),),
            None,
        ),
    )
    definition_cache: dict[object, int] = {}

    columns, misses, non_screening_indices = _core.pack_stage052_screening_occurrences(
        (
            deferred,
            {"event": "not_screening"},
            deferred._replace(values=(8, *deferred.values[1:])),
        ),
        definition_cache,
        {},
        100,
    )

    assert columns[0] == [100, 102]
    assert columns[1] == [None, None]
    assert columns[5] == [7, 8]
    assert len(misses) == 1
    assert non_screening_indices == [1]
    definition_key, event_index, occurrence_indices = misses[0]
    assert event_index == 0
    assert occurrence_indices == [0, 1]

    definition_cache[definition_key] = 17
    cached_columns, cached_misses, cached_non_screening_indices = (
        _core.pack_stage052_screening_occurrences(
        (deferred,),
        definition_cache,
        {},
        200,
        )
    )

    assert cached_columns[0] == [200]
    assert cached_columns[1] == [17]
    assert cached_misses == []
    assert cached_non_screening_indices == []


def test_native_v3_screening_packer_rejects_negative_evidence_drift() -> None:
    from evrptw import _core

    deferred = DeferredScreeningDecision(
        axis_name="fixed_work",
        values=(
            7,
            "route:2:C1",
            "legacy",
            3,
            "repair",
            0.1,
            0.2,
            "rejected",
            "negative_cache",
            2.5,
            None,
            3.0,
            True,
            "negative_cache",
            1.0,
            True,
            True,
            4.0,
            (("capacity", "fail", 2.5, "capacity"),),
            True,
        ),
    )
    evidence_cache: dict[object, tuple[object, ...]] = {}
    _core.pack_stage052_screening_occurrences(
        (deferred,),
        {},
        evidence_cache,
        1,
    )
    drifted_values = list(deferred.values)
    drifted_values[8] = "changed_reason"
    with pytest.raises(ValueError, match="inconsistent evidence"):
        _core.pack_stage052_screening_occurrences(
            (deferred._replace(values=tuple(drifted_values)),),
            {},
            evidence_cache,
            2,
        )


def test_native_neighborhood_packer_matches_python_normalizer() -> None:
    from evrptw import _core

    event: dict[str, object] = {
        "event_type": "neighborhood_event",
        "record_type": None,
        "timestamp_seconds": True,
        "started_at": "1.25",
        "completed_at": 2,
        "duration_seconds": "invalid",
        "lane": None,
        "iteration": "7",
        "operator": "repair",
        "status": None,
        "feasible": True,
        "candidate_vehicle_count": "3",
        "candidate_objective_key": [3, 100.0, 2.0, 1],
        "route_indices": (0, 1),
    }
    lane_ids: dict[str, int] = {}
    operator_ids: dict[str, int] = {}
    columns, non_neighborhood_indices = _core.pack_stage052_neighborhood_events(
        (event, {"event_type": "neighborhood_event", "unsupported": 1}),
        lane_ids,
        operator_ids,
        {},
        artifact_module._NEIGHBORHOOD_FAST_FIELDS,  # noqa: SLF001
        artifact_module._MISSING_NEIGHBORHOOD_EXTRA,  # noqa: SLF001
        artifact_module._stable_dictionary_id,  # noqa: SLF001
        artifact_module._json_text,  # noqa: SLF001
        41,
    )

    assert non_neighborhood_indices == [1]
    assert lane_ids == {
        "None": artifact_module._stable_dictionary_id("lane:None")  # noqa: SLF001
    }
    assert operator_ids == {
        "repair": artifact_module._stable_dictionary_id(  # noqa: SLF001
            "operator:repair"
        )
    }
    native_row = tuple(column[0] for column in columns)
    python_row = artifact_module._normalise_neighborhood_event_values(  # noqa: SLF001
        event,
        event_id=41,
        lane_ids=lane_ids,
        operator_ids=operator_ids,
    )
    assert native_row == python_row


def test_v3_buffered_screening_bridge_roundtrips_through_real_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    occurrence_column_batch_sizes: list[int] = []
    event_column_batch_sizes: list[int] = []
    original_append_columns = artifact_module._StreamingParquetSink.append_columns  # noqa: SLF001

    def record_append_columns(
        self: artifact_module._StreamingParquetSink,  # noqa: SLF001
        columns: object,
    ) -> None:
        buffered = tuple(columns)  # type: ignore[arg-type]
        if "screening_occurrences" in self.path.name:
            occurrence_column_batch_sizes.append(len(buffered[0]))
        elif "_events_" in self.path.name:
            event_column_batch_sizes.append(len(buffered[0]))
        original_append_columns(self, buffered)

    monkeypatch.setattr(
        artifact_module._StreamingParquetSink,  # noqa: SLF001
        "append_columns",
        record_append_columns,
    )
    run_label = "stage05.2_artifact_streaming_attempt96"
    writer = ArtifactBundleWriter(
        tmp_path / "results" / run_label,
        ArtifactRunContext("stage05.2", "artifact_streaming", run_label),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    )
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,
        axis_name="fixed_work",
        buffer_rows=2,
        async_persistence=True,
    )
    for iteration in range(2):
        sink.append_event(
            {
                "event_type": "operator_call",
                "lane": "legacy",
                "iteration": iteration,
                "operator": "repair",
            }
        )
    decision = ScreeningDecision(
        decision_id=1,
        route_key="route:2:C1",
        lane="legacy",
        iteration=1,
        operator="repair",
        status="rejected",
        first_failed_check="capacity",
        reason="capacity",
        checks=(ScreeningCheckTrace("capacity", "fail", 9.0, "capacity"),),
        demand=9.0,
        min_time_window_slack=1.0,
        distance_lower_bound=3.0,
        distance_increment_lower_bound=None,
        single_segment_reachable=True,
        structural_energy_lower_bound=4.0,
        negative_cache_hit=False,
        exact_call_blocked=True,
        started_at=0.1,
        completed_at=0.2,
        duration_seconds=0.1,
    )
    sink.append_screening_decision(decision)
    sink.append_screening_decision(replace(decision, decision_id=2))
    sink.close()
    assert event_column_batch_sizes == [2]
    assert occurrence_column_batch_sizes == [2]
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    bundle = writer.finalize()

    rows = list(
        ArtifactReader(bundle.run_dir).iter_events(f"toy/2014/{run_label}_events_toy_2014.parquet")
    )
    screening_rows = [row for row in rows if row["event_type"] == "screening_decision"]
    assert [row["decision_id"] for row in screening_rows] == [1, 2]
    assert all(row["demand"] == pytest.approx(9.0) for row in screening_rows)
    assert screening_rows[0]["checks"] == [
        {"check": "capacity", "status": "fail", "value": 9.0, "reason": "capacity"}
    ]


def test_v2_screening_bridge_retains_the_complete_legacy_payload() -> None:
    shard = _RecordingShard("screening_decisions_v2")
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="fixed_work",
        buffer_rows=1,
    )
    decision = ScreeningDecision(
        decision_id=1,
        route_key="route:2:C1",
        lane="legacy",
        iteration=1,
        operator="repair",
        status="rejected",
        first_failed_check="capacity",
        reason="capacity",
        checks=(ScreeningCheckTrace("capacity", "fail", 2.5, "capacity"),),
        demand=2.5,
        min_time_window_slack=1.0,
        distance_lower_bound=3.0,
        distance_increment_lower_bound=None,
        single_segment_reachable=True,
        structural_energy_lower_bound=4.0,
        negative_cache_hit=False,
        exact_call_blocked=True,
        started_at=0.1,
        completed_at=0.2,
        duration_seconds=0.1,
    )

    sink.append_screening_decision(decision)

    assert len(shard.events) == 1
    streamed = shard.events[0]
    assert streamed["status"] == "rejected"
    assert streamed["first_failed_check"] == "capacity"
    assert streamed["duration_seconds"] == pytest.approx(0.1)
    assert streamed["checks"] == (
        {"check": "capacity", "status": "fail", "value": 2.5, "reason": "capacity"},
    )


def test_v2_screening_bridge_roundtrips_through_the_real_writer(tmp_path: Path) -> None:
    run_label = "stage05.2_artifact_streaming_attempt95"
    writer = ArtifactBundleWriter(
        tmp_path / "results" / run_label,
        ArtifactRunContext("stage05.2", "artifact_streaming", run_label),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v2",
        ),
    )
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,
        axis_name="fixed_work",
        buffer_rows=1,
    )
    decision = ScreeningDecision(
        decision_id=1,
        route_key="route:2:C1",
        lane="legacy",
        iteration=1,
        operator="repair",
        status="rejected",
        first_failed_check="capacity",
        reason="capacity",
        checks=(ScreeningCheckTrace("capacity", "fail", 9.0, "capacity"),),
        demand=9.0,
        min_time_window_slack=1.0,
        distance_lower_bound=3.0,
        distance_increment_lower_bound=None,
        single_segment_reachable=True,
        structural_energy_lower_bound=4.0,
        negative_cache_hit=False,
        exact_call_blocked=True,
        started_at=0.1,
        completed_at=0.2,
        duration_seconds=0.1,
    )
    sink.append_screening_decision(decision)
    sink.finish()
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    bundle = writer.finalize()

    rows = list(
        ArtifactReader(bundle.run_dir).iter_events(f"toy/2014/{run_label}_events_toy_2014.parquet")
    )
    assert rows[0]["status"] == "rejected"
    assert rows[0]["demand"] == pytest.approx(9.0)
    assert rows[0]["checks"] == [
        {"check": "capacity", "status": "fail", "value": 9.0, "reason": "capacity"}
    ]


def test_active_write_is_charged_to_persistence_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="fixed_work",
        buffer_rows=1,
    )
    original_append = shard.append

    def delayed_append(**kwargs: object) -> int:
        time.sleep(0.01)
        return original_append(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(shard, "append", delayed_append)

    sink.append_event(
        {
            "event_type": "candidate_state",
            "lane": "legacy",
            "operator": "repair",
            "timestamp_seconds": 0.1,
            "current_objective_key": [1, 2.0, 3.0, 4],
        }
    )

    assert sink.persistence_nanoseconds >= 9_000_000


def test_callback_artifact_preparation_is_charged_to_persistence_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="fixed_work",
        buffer_rows=100,
    )
    original = stage052_performance._route_reference_event  # noqa: SLF001

    def delayed_route_reference(*args: object, **kwargs: object) -> dict[str, object]:
        time.sleep(0.01)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        stage052_performance,
        "_route_reference_event",
        delayed_route_reference,
    )

    sink.append_neighborhood_event(
        {
            "event_type": "neighborhood_event",
            "lane": "legacy",
            "operator": "relocate",
            "status": "failed",
        }
    )

    assert sink.persistence_nanoseconds >= 9_000_000


def test_screening_identity_is_charged_to_persistence_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="fixed_work",
        buffer_rows=100,
    )
    original_queue = sink._queue_owned  # noqa: SLF001

    def delayed_queue(payload: dict[str, object]) -> None:
        time.sleep(0.01)
        original_queue(payload)

    monkeypatch.setattr(sink, "_queue_owned", delayed_queue)
    decision = ScreeningDecision(
        decision_id=1,
        route_key="route:2:C1",
        lane="legacy",
        iteration=1,
        operator="repair",
        status="rejected",
        first_failed_check="capacity",
        reason="capacity",
        checks=(ScreeningCheckTrace("capacity", "fail", 2.5, "capacity"),),
        demand=2.5,
        min_time_window_slack=1.0,
        distance_lower_bound=3.0,
        distance_increment_lower_bound=None,
        single_segment_reachable=True,
        structural_energy_lower_bound=4.0,
        negative_cache_hit=False,
        exact_call_blocked=True,
        started_at=0.1,
        completed_at=0.2,
        duration_seconds=0.1,
    )

    sink.append_screening_decision(decision)

    assert sink.persistence_nanoseconds >= 9_000_000


def test_negative_screening_tail_cache_fails_on_route_evidence_drift() -> None:
    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="fixed_work",
        buffer_rows=100,
    )
    base = ScreeningDecision(
        decision_id=1,
        route_key="route:2:C1",
        lane="legacy",
        iteration=1,
        operator="repair",
        status="negative_cache_hit",
        first_failed_check="capacity",
        reason="capacity",
        checks=(ScreeningCheckTrace("negative_sequence_cache", "hit", True, "reused"),),
        demand=2.5,
        min_time_window_slack=1.0,
        distance_lower_bound=3.0,
        distance_increment_lower_bound=None,
        single_segment_reachable=True,
        structural_energy_lower_bound=4.0,
        negative_cache_hit=True,
        exact_call_blocked=True,
        started_at=0.1,
        completed_at=0.2,
        duration_seconds=0.1,
    )
    sink.append_screening_decision(base)
    sink.append_screening_decision(base)

    with pytest.raises(RuntimeError, match="inconsistent evidence"):
        sink.append_screening_decision(replace(base, decision_id=2, demand=9.0))


def test_v3_negative_cache_hit_reuses_definition_without_check_iteration(
    tmp_path: Path,
) -> None:
    class NoIterationChecks(tuple[ScreeningCheckTrace, ...]):
        def __iter__(self) -> Any:
            raise AssertionError("cached negative screening checks were rebuilt")

    run_label = "stage05.2_artifact_streaming_attempt94"
    writer = ArtifactBundleWriter(
        tmp_path / "results" / run_label,
        ArtifactRunContext("stage05.2", "artifact_streaming", run_label),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    )
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,
        axis_name="fixed_work",
        buffer_rows=2,
    )
    base = ScreeningDecision(
        decision_id=1,
        route_key="route:2:C1",
        lane="legacy",
        iteration=1,
        operator="repair",
        status="negative_cache_hit",
        first_failed_check="capacity",
        reason="capacity",
        checks=(ScreeningCheckTrace("negative_sequence_cache", "hit", True, "reused"),),
        demand=2.5,
        min_time_window_slack=1.0,
        distance_lower_bound=3.0,
        distance_increment_lower_bound=None,
        single_segment_reachable=True,
        structural_energy_lower_bound=4.0,
        negative_cache_hit=True,
        exact_call_blocked=True,
        started_at=0.1,
        completed_at=0.2,
        duration_seconds=0.1,
    )
    sink.append_screening_decision(base)
    sink.append_screening_decision(
        replace(
            base,
            decision_id=2,
            checks=NoIterationChecks(base.checks),
        )
    )
    sink.close()
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    bundle = writer.finalize()
    rows = list(
        ArtifactReader(bundle.run_dir).iter_events(
            f"toy/2014/{run_label}_events_toy_2014.parquet"
        )
    )
    assert [row["decision_id"] for row in rows] == [1, 2]
    assert rows[0]["checks"] == rows[1]["checks"]


def test_v3_writer_fails_fast_on_negative_cache_evidence_drift(tmp_path: Path) -> None:
    run_label = "stage05.2_artifact_streaming_attempt92"
    writer = ArtifactBundleWriter(
        tmp_path / "results" / run_label,
        ArtifactRunContext("stage05.2", "artifact_streaming", run_label),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    )
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,
        axis_name="fixed_work",
        buffer_rows=2,
    )
    base = ScreeningDecision(
        decision_id=1,
        route_key="route:2:C1",
        lane="legacy",
        iteration=1,
        operator="repair",
        status="negative_cache_hit",
        first_failed_check="capacity",
        reason="capacity",
        checks=(ScreeningCheckTrace("negative_sequence_cache", "hit", True, "reused"),),
        demand=2.5,
        min_time_window_slack=1.0,
        distance_lower_bound=3.0,
        distance_increment_lower_bound=None,
        single_segment_reachable=True,
        structural_energy_lower_bound=4.0,
        negative_cache_hit=True,
        exact_call_blocked=True,
        started_at=0.1,
        completed_at=0.2,
        duration_seconds=0.1,
    )

    sink.append_screening_decision(base)
    with pytest.raises(ArtifactIntegrityError, match="inconsistent evidence"):
        sink.append_screening_decision(replace(base, decision_id=2, demand=9.0))


def test_v3_negative_cache_contexts_roundtrip_through_writer(tmp_path: Path) -> None:
    run_label = "stage05.2_artifact_streaming_attempt93"
    writer = ArtifactBundleWriter(
        tmp_path / "results" / run_label,
        ArtifactRunContext("stage05.2", "artifact_streaming", run_label),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    )
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,
        axis_name="fixed_work",
        buffer_rows=256,
    )
    base = ScreeningDecision(
        decision_id=1,
        route_key="route:2:C1",
        lane="lane-0",
        iteration=1,
        operator="operator-0",
        status="negative_cache_hit",
        first_failed_check="capacity",
        reason="capacity",
        checks=(ScreeningCheckTrace("negative_sequence_cache", "hit", True, "reused"),),
        demand=2.5,
        min_time_window_slack=1.0,
        distance_lower_bound=3.0,
        distance_increment_lower_bound=None,
        single_segment_reachable=True,
        structural_energy_lower_bound=4.0,
        negative_cache_hit=True,
        exact_call_blocked=True,
        started_at=0.1,
        completed_at=0.2,
        duration_seconds=0.1,
    )
    for index in range(100):
        sink.append_screening_decision(
            replace(
                base,
                decision_id=index + 1,
                lane=f"lane-{index}",
                operator=f"operator-{index}",
            )
        )

    sink.close()
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    bundle = writer.finalize()
    rows = list(
        ArtifactReader(bundle.run_dir).iter_events(
            f"toy/2014/{run_label}_events_toy_2014.parquet"
        )
    )
    assert [row["decision_id"] for row in rows] == list(range(1, 101))
    assert [row["lane"] for row in rows] == [
        f"fixed_work:lane-{index}" for index in range(100)
    ]
    assert [row["operator"] for row in rows] == [
        f"operator-{index}" for index in range(100)
    ]


def test_stage052_trace_sink_flushes_bounded_event_batches() -> None:
    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="wall_clock_300",
        buffer_rows=4,
        neighborhood_buffer_rows=4,
    )

    for iteration in range(5):
        sink.append_event(
            {
                "event_type": "candidate_state",
                "lane": "legacy",
                "iteration": iteration,
                "operator": "repair",
                "timestamp_seconds": float(iteration),
                "accepted": False,
                "global_best": False,
                "current_objective_key": [2, 10.0, 0.0, 0],
            }
        )

    assert shard.append_calls == 1
    assert len(shard.events) == 4
    sink.finish()
    assert shard.append_calls == 2
    assert len(shard.events) == 5


def test_stage052_trace_sink_async_pipeline_preserves_order_and_drains() -> None:
    original_switch_interval = sys.getswitchinterval()
    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="wall_clock_30",
        buffer_rows=2,
        async_persistence=True,
    )
    for iteration in range(5):
        sink.append_event(
            {
                "event_type": "operator_call",
                "lane": "legacy",
                "iteration": iteration,
                "operator": "repair",
            }
        )

    sink.finish()
    sink.freeze_solver_persistence_boundary()
    sink.close()
    assert sys.getswitchinterval() == original_switch_interval

    assert [event["iteration"] for event in shard.events] == list(range(5))
    pipeline = sink.persistence_pipeline
    assert pipeline.pop("writer_active_nanoseconds") > 0  # type: ignore[operator]
    writer_cpu_ns = pipeline.pop("writer_cpu_nanoseconds")
    assert pipeline.pop("producer_wait_nanoseconds") >= 0  # type: ignore[operator]
    producer_active_ns = pipeline.pop("producer_active_nanoseconds")
    union_ns = pipeline.pop("persistence_union_nanoseconds")
    solver_union_ns = pipeline.pop("solver_persistence_union_nanoseconds")
    solver_critical_ns = pipeline.pop("solver_persistence_critical_path_nanoseconds")
    solver_producer_ns = pipeline.pop("solver_producer_active_nanoseconds")
    solver_writer_cpu_ns = pipeline.pop("solver_writer_cpu_nanoseconds")
    assert 0 < producer_active_ns <= union_ns  # type: ignore[operator]
    assert 0 < solver_union_ns <= union_ns  # type: ignore[operator]
    assert 0 <= solver_writer_cpu_ns <= writer_cpu_ns  # type: ignore[operator]
    assert solver_critical_ns == max(solver_producer_ns, solver_writer_cpu_ns)
    ledger = pipeline.pop("batch_ledger")
    assert [entry["ordinal"] for entry in ledger] == [0, 1, 2]  # type: ignore[index, union-attr]
    assert [entry["row_count"] for entry in ledger] == [2, 2, 1]  # type: ignore[index, union-attr]
    assert pipeline == {
        "mode": "bounded_async_thread",
        "queue_max_batches": 1,
        "writer_thread_switch_interval_seconds": (
            stage052_performance.STAGE052_WRITER_THREAD_SWITCH_INTERVAL_SECONDS
        ),
        "submitted_batches": 3,
        "completed_batches": 3,
        "peak_queued_batches": 1,
    }


def test_async_pipeline_waits_for_writer_before_next_callback() -> None:
    writer_started = threading.Event()
    release_writer = threading.Event()

    class BlockingShard(_RecordingShard):
        def append(self, **kwargs: object) -> int:
            if not writer_started.is_set():
                writer_started.set()
                if not release_writer.wait(timeout=5.0):
                    raise RuntimeError("test writer release timed out")
            return super().append(**kwargs)

    shard = BlockingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="wall_clock_30",
        buffer_rows=2,
        async_persistence=True,
    )
    for iteration in range(2):
        sink.append_event(
            {
                "event_type": "operator_call",
                "lane": "legacy",
                "iteration": iteration,
                "operator": "repair",
            }
        )
    assert writer_started.wait(timeout=1.0)

    producer_completed = threading.Event()

    def append_while_writer_is_active() -> None:
        for iteration in (2, 3):
            sink.append_event(
                {
                    "event_type": "operator_call",
                    "lane": "legacy",
                    "iteration": iteration,
                    "operator": "repair",
                }
            )
        producer_completed.set()

    producer = threading.Thread(target=append_while_writer_is_active)
    producer.start()
    try:
        assert not producer_completed.wait(timeout=0.1)
    finally:
        release_writer.set()
        producer.join(timeout=5.0)
    assert not producer.is_alive()
    assert producer_completed.is_set()

    sink.close()
    assert [event["iteration"] for event in shard.events] == [0, 1, 2, 3]
    pipeline = sink.persistence_pipeline
    producer_ns = pipeline["producer_active_nanoseconds"]
    writer_ns = pipeline["writer_active_nanoseconds"]
    union_ns = pipeline["persistence_union_nanoseconds"]
    assert isinstance(producer_ns, int)
    assert isinstance(writer_ns, int)
    assert isinstance(union_ns, int)
    assert max(producer_ns, writer_ns) <= union_ns <= producer_ns + writer_ns


def test_stage052_trace_sink_rejects_oversized_async_callback_batch() -> None:
    with pytest.raises(ValueError, match="may not exceed one Parquet row group"):
        stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
            shard=_RecordingShard(),
            axis_name="fixed_work",
            buffer_rows=65_537,
            async_persistence=True,
        )


def test_stage052_trace_sink_async_pipeline_surfaces_writer_failure() -> None:
    original_switch_interval = sys.getswitchinterval()

    class FailingShard(_RecordingShard):
        def append(self, **_kwargs: object) -> int:
            raise RuntimeError("writer failed")

    shard = FailingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="wall_clock_30",
        buffer_rows=1,
        async_persistence=True,
    )
    sink.append_event(
        {
            "event_type": "operator_call",
            "lane": "legacy",
            "iteration": 1,
            "operator": "repair",
        }
    )

    with pytest.raises(RuntimeError, match="writer failed"):
        sink.finish()
    with pytest.raises(RuntimeError, match="writer failed"):
        sink.discard_pending()
    assert sys.getswitchinterval() == original_switch_interval


def test_async_writer_cpu_tick_is_bounded_by_elapsed_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((0, 1_000_000_000))
    monkeypatch.setattr(stage052_performance.time, "thread_time_ns", lambda: next(ticks))
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=_RecordingShard(),  # type: ignore[arg-type]
        axis_name="fixed_work",
        buffer_rows=1,
        async_persistence=True,
    )
    sink.append_event(
        {
            "event_type": "operator_call",
            "lane": "legacy",
            "iteration": 1,
            "operator": "repair",
        }
    )
    sink.finish()
    sink.freeze_solver_persistence_boundary()
    sink.close()

    pipeline = sink.persistence_pipeline
    assert pipeline["writer_cpu_nanoseconds"] <= pipeline["writer_active_nanoseconds"]
    assert (
        pipeline["solver_persistence_critical_path_nanoseconds"]
        <= pipeline["solver_persistence_union_nanoseconds"]
    )


def test_async_pipeline_ledger_is_independently_replayed_from_events(
    tmp_path: Path,
) -> None:
    def write(label: str, *, corrupt_digest: bool) -> Path:
        run_dir = tmp_path / label
        writer = ArtifactBundleWriter(
            run_dir,
            ArtifactRunContext("stage05.2", "native_kernels", label),
            ArtifactStorageConfig(
                storage_policy_version="artifact-storage-v2",
                screening_schema_version="screening_decisions_v3",
            ),
        )
        shard = writer.open_v2_shard(
            instance="toy",
            seed=2014,
            shard_ordinal=0,
            worker_identity="worker-0",
        )
        sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
            shard=shard,
            axis_name="fixed_work",
            buffer_rows=2,
            async_persistence=True,
        )
        for iteration in range(3):
            sink.append_event(
                {
                    "event_type": "operator_call",
                    "lane": "legacy",
                    "iteration": iteration,
                    "operator": "repair",
                }
            )
        sink.append_screening_decision(
            ScreeningDecision(
                decision_id=1,
                route_key="route:2:C1",
                lane="legacy",
                iteration=3,
                operator="repair",
                status="pass",
                first_failed_check="",
                reason="",
                checks=(ScreeningCheckTrace("capacity", "pass", 2.5, ""),),
                demand=2.5,
                min_time_window_slack=1.0,
                distance_lower_bound=3.0,
                distance_increment_lower_bound=None,
                single_segment_reachable=True,
                structural_energy_lower_bound=4.0,
                negative_cache_hit=False,
                exact_call_blocked=False,
                started_at=0.1,
                completed_at=0.2,
                duration_seconds=0.1,
            )
        )
        sink.append_route_evaluation(
            RouteEvaluationTrace(
                evaluation_id=1,
                route_key="route:2:C1",
                lane="legacy",
                iteration=4,
                operator="repair",
                kind="cache_hit",
                started_at=0.2,
                completed_at=0.3,
                duration_seconds=0.1,
                exact_started=False,
                exact_completed=False,
                feasible=True,
                failure_reason="",
            )
        )
        sink.finish()
        sink.freeze_solver_persistence_boundary()
        sink.close()
        pipeline = sink.persistence_pipeline
        if corrupt_digest:
            ledger = pipeline["batch_ledger"]
            assert isinstance(ledger, list)
            assert isinstance(ledger[0], dict)
            ledger[0]["event_token_sha256"] = "f" * 64
        shard.finalize(
            raw_payload={
                "instance": "toy",
                "seed": 2014,
                "axes": {"fixed_work": {"started_calls": 0, "completed_calls": 0}},
            },
            solution_payload={
                "instance": "toy",
                "seed": 2014,
                "axes": {
                    "fixed_work": {
                        "routes": [],
                        "objective_key": [0, 0.0, 0.0, 0],
                    }
                },
            },
            trace_payload={
                "axes": {
                    "fixed_work": {
                        "streamed_record_counts": {
                            "events": 3,
                            "incremental_propagations": 0,
                            "route_evaluations": 1,
                            "screening_decisions": 1,
                        },
                        "persistence_pipeline": pipeline,
                    }
                }
            },
            environment_payload={},
        )
        writer.finalize()
        return run_dir

    valid = write("stage05.2_native_kernels_attempt98", corrupt_digest=False)
    forged = write("stage05.2_native_kernels_attempt97", corrupt_digest=True)

    assert replay_stage052_storage_semantics(valid)
    with pytest.raises(ArtifactIntegrityError, match="batch digest mismatch"):
        replay_stage052_storage_semantics(forged)


def test_small_neighborhood_stream_stays_in_bounded_memory_until_canonical_merge() -> None:
    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="wall_clock_300",
        buffer_rows=4,
    )

    for iteration in range(4):
        sink.append_neighborhood_event(
            {
                "event_type": "neighborhood_event",
                "lane": "legacy",
                "iteration": iteration,
                "operator": "relocate",
                "status": "failed",
            }
        )

    assert sink._neighborhood_spool_path is None  # noqa: SLF001
    assert len(sink._neighborhood_buffer) == 4  # noqa: SLF001
    assert sink._event_buffer == []  # noqa: SLF001
    assert shard.events == []

    sink.finish()

    assert len(shard.events) == 4
    assert all(event["record_type"] == "neighborhood_event" for event in shard.events)
    assert [event["iteration"] for event in shard.events] == list(range(4))
    assert sink.event_count == 4
    assert sink.persisted_family_counts["events"] == 4
    assert sink._neighborhood_buffer == []  # noqa: SLF001
    sink.finish()
    assert len(shard.events) == 4


def test_large_neighborhood_stream_spills_to_measured_volume_until_canonical_merge() -> None:
    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="wall_clock_300",
        buffer_rows=4,
        neighborhood_buffer_rows=4,
    )

    for iteration in range(17):
        sink.append_neighborhood_event(
            {
                "event_type": "neighborhood_event",
                "lane": "legacy",
                "iteration": iteration,
                "operator": "relocate",
                "status": "failed",
            }
        )

    scratch_path = sink._neighborhood_spool_path  # noqa: SLF001
    assert scratch_path is not None
    assert scratch_path.is_relative_to(shard.scratch_directory)
    assert scratch_path.is_file()
    assert sink._neighborhood_buffer == []  # noqa: SLF001
    assert sink._event_buffer == []  # noqa: SLF001
    assert shard.events == []

    sink.finish()

    assert len(shard.events) == 17
    assert all(event["record_type"] == "neighborhood_event" for event in shard.events)
    sink.close()
    assert not scratch_path.exists()


def test_memory_neighborhood_drain_preparation_is_charged_to_persistence() -> None:
    class SlowIterationList(list[dict[str, object]]):
        def __iter__(self):  # type: ignore[no-untyped-def]
            time.sleep(0.02)
            return super().__iter__()

    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="fixed_work",
        buffer_rows=4,
    )
    sink._neighborhood_buffer = SlowIterationList(  # noqa: SLF001
        [
            {
                "event_type": "neighborhood_event",
                "record_type": "neighborhood_event",
                "lane": "legacy",
                "iteration": 1,
            }
        ]
    )

    before = sink.persistence_nanoseconds
    sink.finish()

    assert sink.persistence_nanoseconds - before >= 15_000_000


def test_spool_close_and_unlink_are_charged_to_persistence() -> None:
    shard = _RecordingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="fixed_work",
        buffer_rows=1,
        neighborhood_buffer_rows=1,
    )
    for iteration in range(2):
        sink.append_neighborhood_event(
            {
                "event_type": "neighborhood_event",
                "lane": "legacy",
                "iteration": iteration,
                "status": "failed",
            }
        )
    scratch_path = sink._neighborhood_spool_path  # noqa: SLF001
    assert scratch_path is not None
    before = sink.persistence_nanoseconds

    sink.discard_pending()

    assert sink.persistence_nanoseconds > before
    assert not scratch_path.exists()


def test_fdopen_failure_closes_descriptor_and_removes_scratch_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors: list[int] = []
    paths: list[Path] = []
    original_mkstemp = tempfile.mkstemp

    def recording_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        descriptor, raw_path = original_mkstemp(*args, **kwargs)
        descriptors.append(descriptor)
        paths.append(Path(raw_path))
        return descriptor, raw_path

    def failing_fdopen(*_args: object, **_kwargs: object) -> object:
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(stage052_performance.tempfile, "mkstemp", recording_mkstemp)
    monkeypatch.setattr(stage052_performance.os, "fdopen", failing_fdopen)
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=_RecordingShard(),  # type: ignore[arg-type]
        axis_name="fixed_work",
        buffer_rows=1,
        neighborhood_buffer_rows=1,
    )
    sink.append_neighborhood_event({"event_type": "neighborhood_event", "status": "failed"})

    with pytest.raises(OSError, match="fdopen failure"):
        sink.append_neighborhood_event({"event_type": "neighborhood_event", "status": "failed"})

    assert len(descriptors) == len(paths) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    assert not paths[0].exists()


def test_stage052_trace_sink_does_not_replay_a_failed_batch() -> None:
    class PartiallyFailingShard(_RecordingShard):
        def append(self, **kwargs: object) -> int:
            self.append_calls += 1
            raise OSError("injected partial write failure")

    shard = PartiallyFailingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="wall_clock_300",
        buffer_rows=1,
    )

    with pytest.raises(OSError, match="partial write failure"):
        sink.append_event(
            {
                "event_type": "candidate_state",
                "lane": "legacy",
                "iteration": 1,
                "operator": "repair",
                "timestamp_seconds": 1.0,
                "current_objective_key": [2, 10.0, 0.0, 0],
            }
        )
    sink.discard_pending()
    sink.close()

    assert shard.append_calls == 1


def test_spool_cleanup_failure_still_publishes_partial_shard_abort() -> None:
    original = RuntimeError("solver failed")

    class FailingStream:
        def discard_pending(self) -> None:
            raise OSError("spool unlink failed")

    class RecordingAbortShard:
        def __init__(self) -> None:
            self.aborted_with: BaseException | None = None

        def abort(self, error: BaseException) -> None:
            self.aborted_with = error

    shard = RecordingAbortShard()

    with pytest.raises(BaseExceptionGroup, match="cleanup also failed") as caught:
        stage052_performance._abort_v2_shard_after_failure(  # noqa: SLF001
            shard,
            active_trace_stream=FailingStream(),  # type: ignore[arg-type]
            original_error=original,
        )

    assert shard.aborted_with is original
    assert caught.value.exceptions[0] is original
    assert isinstance(caught.value.exceptions[1], OSError)


def test_concurrent_writer_failure_is_retained_with_solver_failure() -> None:
    original = RuntimeError("solver failed")

    class FailingShard(_RecordingShard):
        def __init__(self) -> None:
            super().__init__()
            self.aborted_with: BaseException | None = None

        def append(self, **_kwargs: object) -> int:
            raise OSError("writer failed")

        def abort(self, error: BaseException) -> None:
            self.aborted_with = error

    shard = FailingShard()
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,  # type: ignore[arg-type]
        axis_name="wall_clock_30",
        buffer_rows=1,
        async_persistence=True,
    )
    sink.append_event(
        {
            "event_type": "operator_call",
            "lane": "legacy",
            "iteration": 1,
            "operator": "repair",
        }
    )
    with pytest.raises(OSError, match="writer failed"):
        sink.finish()

    with pytest.raises(BaseExceptionGroup, match="cleanup also failed") as caught:
        stage052_performance._abort_v2_shard_after_failure(  # noqa: SLF001
            shard,
            active_trace_stream=sink,
            original_error=original,
        )

    assert shard.aborted_with is original
    assert caught.value.exceptions[0] is original
    assert isinstance(caught.value.exceptions[1], OSError)
    assert str(caught.value.exceptions[1]) == "writer failed"


def test_neighborhood_event_stream_externalizes_more_than_a_row_group() -> None:
    observed = 0

    def persist(_event: Mapping[str, object]) -> None:
        nonlocal observed
        observed += 1

    stream = _NeighborhoodEventStream(persist)
    count = 65_536 + 17

    stream.extend(
        {
            "event_type": "neighborhood_event",
            "iteration": iteration,
        }
        for iteration in range(count)
    )

    assert observed == count
    assert stream.emitted_count == count
    assert stream == []


def test_streamed_unique_route_reconciliation_stays_bounded_beyond_one_row_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact unique identities live on the shard volume, not in process sets/TMPDIR."""

    foreign_tmp = tmp_path / "foreign-tmp"
    foreign_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(foreign_tmp))
    run_label = "stage05.2_artifact_streaming_attempt97"
    writer = ArtifactBundleWriter(
        tmp_path / "staging" / run_label,
        ArtifactRunContext("stage05.2", "artifact_streaming", run_label),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    )
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    sink = stage052_performance._Stage052TraceStreamSink(  # noqa: SLF001
        shard=shard,
        axis_name="wall_clock_300",
    )
    trace = Stage03Trace(MeasurementConfig(stream_sink=sink))
    count = 65_536 + 1

    for index in range(count):
        trace.record_route_evaluation(
            (f"C{index}",),
            lane="legacy",
            iteration=index,
            operator="repair",
            kind="exact_call",
            started_at=float(index),
            completed_at=float(index) + 0.5,
            exact_started=True,
            exact_completed=True,
            feasible=True,
            failure_reason="",
        )

    reconciliation = trace.reconcile(
        SimpleNamespace(
            charging_subproblem_calls=count,
            exact_started_calls=count,
            cache_hits=0,
            unique_route_evaluations=count,
            unique_route_semantics="started_lane_identity_legacy_v1",
            destroy_statistics={},
            repair_statistics={},
            neighborhood_statistics={},
            effective_iterations=0,
            accepted_moves=0,
            rejected_moves=0,
            improving_moves=0,
            screening_statistics={},
            cache_incremental_statistics={},
        )
    )

    assert reconciliation["status"] == "pass"
    assert reconciliation["observed"]["unique_route_evaluations"] == count
    assert trace.stream_unique_identity_hot_entries <= shard.unique_route_hot_entry_limit
    assert shard.unique_route_identity_count("legacy_started", "wall_clock_300") == count
    assert shard.unique_route_hot_entries <= shard.unique_route_hot_entry_limit
    scratch = shard.unique_route_scratch_path
    assert scratch.is_relative_to(writer.run_dir / "toy" / "2014")
    assert not scratch.is_relative_to(foreign_tmp)
    shard.abort("bounded-state test complete")
    assert not scratch.exists()


def test_stage052_solver_installs_runtime_trace_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    expected = _RecordingTraceSink()

    def fake_solve(_instance: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(stage052_performance, "solve_alns", fake_solve)
    axis = stage052_performance.Stage052Axis(
        "wall_clock_300",
        "wall_clock",
        300.0,
        max_iterations=None,
    )
    config = SimpleNamespace(batch_size=8, native_kernels=object())
    stage04 = SimpleNamespace(
        screening_config=object(),
        cache_incremental_config=object(),
        stage04_config=object(),
    )
    stage02 = SimpleNamespace(vehicle_operator_config=object())
    instance = SimpleNamespace(distance_backend="native")

    result = stage052_performance._solve_stage052_axis(  # noqa: SLF001
        instance,  # type: ignore[arg-type]
        seed=2014,
        axis=axis,
        config=config,  # type: ignore[arg-type]
        stage04=stage04,
        stage02=stage02,
        trace_sink=expected,
    )

    assert result is not None
    measurement = captured["measurement_config"]
    assert isinstance(measurement, MeasurementConfig)
    assert measurement.stream_sink is expected
