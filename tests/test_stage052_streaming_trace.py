from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evrptw.alns import _NeighborhoodEventStream
from evrptw.artifacts import ArtifactBundleWriter, ArtifactRunContext, ArtifactStorageConfig
from evrptw.experiments import stage052_performance
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
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.flushes = 0

    def append(
        self,
        *,
        route_dictionary: Mapping[str, object],
        critical_events: object,
        diagnostic_rows: object = (),
    ) -> int:
        del route_dictionary, diagnostic_rows
        rows = [dict(row) for row in critical_events]  # type: ignore[union-attr]
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

    assert len(shard.events) == 1
    assert shard.events[0]["benchmark_axis"] == "wall_clock_300"
    assert shard.events[0]["record_type"] == "route_evaluation"
    assert sink.event_count == 1

    sink.append_neighborhood_event(
        {
            "event_type": "neighborhood_event",
            "lane": "legacy",
            "iteration": 1,
            "operator": "relocate",
            "status": "accepted",
        }
    )

    assert len(shard.events) == 2
    assert shard.events[-1]["record_type"] == "neighborhood_event"


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
