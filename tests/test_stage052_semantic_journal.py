from __future__ import annotations

import gzip
import json
import tempfile
import tracemalloc
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from evrptw.measurement import MeasurementConfig, Stage03Trace
from evrptw.stage052_physical_telemetry import iter_verified_physical_telemetry
from evrptw.stage052_semantic_journal import (
    iter_verified_semantic_journal,
    semantic_bundle_path,
    write_semantic_journal,
)


def _result_with_events(events: list[dict[str, object]]) -> SimpleNamespace:
    return SimpleNamespace(
        measurement_trace=SimpleNamespace(runtime_semantic_events=tuple(events)),
        effective_iterations=1,
        candidate_control_statistics={"enabled": False},
        objective=SimpleNamespace(key=(1, 2.0, 3.0, 4)),
        termination_reason="iteration_limit",
        iterations=1,
        exact_started_calls=1,
        exact_completed_calls=1,
        exact_interrupted_calls=0,
    )


def _complete_events(*, screening_count: int = 1) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(stream: str, event_type: str, **fields: object) -> None:
        event_id = len(rows) + 1
        rows.append(
            {
                "semantic_event_id": event_id,
                "runtime_causal_event_id": event_id,
                "semantic_stream": stream,
                "event_type": event_type,
                **fields,
            }
        )

    add(
        "candidate_state",
        "candidate_state",
        lane="legacy",
        iteration=0,
        operator="initial_solution",
        candidate_route_keys=["2:C1"],
        candidate_full_route_keys=["2:C1"],
        candidate_objective_key=[1, 2.0, 3.0, 4],
        candidate_feasible=True,
        accepted=True,
        status="accepted",
    )
    add("operator", "operator_event", status="accepted")
    add("stage04", "stage04_event", status="updated")
    add(
        "exact_work",
        "exact_batch_started",
        status="started",
        lane="legacy",
        iteration=0,
        operator="initial_solution",
        customer_sequences=[["C1"]],
        requested_calls=1,
        started_calls=1,
    )
    add(
        "candidate_transaction",
        "parallel_batch",
        status="native_complete",
        lane="legacy",
        iteration=0,
        operator="initial_solution",
        submission_order=[0],
        completion_order=[0],
        merge_order=[0],
        customer_sequences=[["C1"]],
    )
    add(
        "exact_result",
        "exact_route_result",
        status="feasible",
        lane="legacy",
        iteration=0,
        operator="initial_solution",
        route_key="route:2:C1",
        exact_started=True,
        exact_completed=True,
        feasible=True,
        deadline_boundary=False,
    )
    add("cache", "cache_event", operation="miss", route_key="2:C1")
    for ordinal in range(screening_count):
        add(
            "screening",
            "screening_decision",
            route_key="2:C1",
            status="passed",
            checks=[],
            demand=1.0,
            min_time_window_slack=2.0,
            distance_lower_bound=3.0,
            structural_energy_lower_bound=4.0,
            single_segment_reachable=True,
            negative_cache_hit=False,
            exact_call_blocked=False,
            ordinal=ordinal,
        )
    add(
        "termination",
        "termination",
        status="iteration_limit",
        iterations=1,
        effective_iterations=1,
        exact_started_calls=1,
        exact_completed_calls=1,
        exact_interrupted_calls=0,
        objective_key=[1, 2.0, 3.0, 4],
    )
    return rows


def test_semantic_journal_round_trip_is_atomic_signed_and_single_copy(
    tmp_path: Path,
) -> None:
    axis_path = tmp_path / "axis.json"
    descriptor = write_semantic_journal(
        axis_path,
        _result_with_events(_complete_events(screening_count=3)),
    )

    bundle = semantic_bundle_path(axis_path, descriptor)
    journal_path = bundle / str(descriptor["event_path"])
    sidecar = bundle / str(descriptor["sidecar_path"])
    physical = descriptor["physical_telemetry"]
    assert isinstance(physical, dict)
    batches = list(iter_verified_physical_telemetry(bundle, physical))
    assert batches[0]["completion_order"] == (0,)
    assert batches[0]["merge_order"] == (0,)
    assert batches[0]["observation_code"] == 0
    assert batches[0]["physical_task_receipts"] == ()
    assert journal_path.is_file()
    assert sidecar.is_file()
    assert (
        sidecar.read_text(encoding="ascii").strip() == sha256(journal_path.read_bytes()).hexdigest()
    )
    replayed = list(iter_verified_semantic_journal(axis_path, descriptor))
    assert len(replayed) == descriptor["event_count"]
    assert [event["semantic_sequence"] for event in replayed] == list(range(len(replayed)))
    assert descriptor["stream_counts"]["screening"] == 3
    persistence = descriptor["persistence"]
    assert isinstance(persistence, dict)
    for field_name in (
        "event_serialization_seconds",
        "compression_seconds",
        "write_seconds",
        "fsync_seconds",
        "hash_seconds",
        "atomic_publish_seconds",
        "total_seconds",
    ):
        assert float(persistence[field_name]) >= 0.0
    physical_persistence = physical["persistence"]
    assert isinstance(physical_persistence, dict)
    assert float(physical_persistence["total_seconds"]) >= 0.0
    assert not tuple(tmp_path.glob(".*.tmp-*"))


def test_physical_telemetry_preserves_process_wide_submission_ids(
    tmp_path: Path,
) -> None:
    events = _complete_events()
    parallel = next(event for event in events if event["event_type"] == "parallel_batch")
    parallel["submission_order"] = [7]
    parallel["completion_order"] = [7]
    parallel["merge_order"] = [7]
    axis_path = tmp_path / "axis.json"
    descriptor = write_semantic_journal(axis_path, _result_with_events(events))
    bundle = semantic_bundle_path(axis_path, descriptor)
    physical = descriptor["physical_telemetry"]
    assert isinstance(physical, dict)

    batches = list(iter_verified_physical_telemetry(bundle, physical))

    assert batches[0]["submission_order"] == (7,)
    assert batches[0]["completion_order"] == (7,)
    assert batches[0]["merge_order"] == (7,)


def test_physical_telemetry_persists_native_worker_times_separately(
    tmp_path: Path,
) -> None:
    events = _complete_events()
    parallel = next(event for event in events if event["event_type"] == "parallel_batch")
    parallel["semantic_completion_order"] = [0]
    parallel["physical_observation"] = "native_worker_observed"
    parallel["physical_task_receipts"] = [[0, 3, 0, 1, 5, 7, 11]]
    axis_path = tmp_path / "axis.json"
    descriptor = write_semantic_journal(axis_path, _result_with_events(events))
    bundle = semantic_bundle_path(axis_path, descriptor)
    physical = descriptor["physical_telemetry"]
    assert isinstance(physical, dict)

    batches = list(iter_verified_physical_telemetry(bundle, physical))

    assert batches[0]["semantic_completion_order"] == (0,)
    assert batches[0]["observation_code"] == 1
    assert batches[0]["physical_task_receipts"] == ((0, 3, 0, 1, 5, 7, 11),)


def test_physical_scheduling_changes_do_not_change_semantic_digest(tmp_path: Path) -> None:
    first_events = _complete_events()
    second_events = _complete_events()
    first_parallel = next(
        event for event in first_events if event["event_type"] == "parallel_batch"
    )
    second_parallel = next(
        event for event in second_events if event["event_type"] == "parallel_batch"
    )
    first_parallel.update(
        {
            "worker_count": 1,
            "worker_protocol": "serial",
            "submission_order": [7],
            "completion_order": [7],
            "merge_order": [7],
            "result_count": 99,
            "semantic_completion_order": [7],
            "physical_observation": "native_worker_observed",
            "physical_task_receipts": [[0, 0, 0, 1, 1, 2, 3]],
        }
    )
    second_parallel.update(
        {
            "worker_count": 4,
            "worker_protocol": "persistent_pool",
            "merge_order": [0],
            "semantic_completion_order": [0],
            "physical_observation": "native_worker_observed",
            "physical_task_receipts": [[0, 3, 0, 1, 10, 20, 30]],
        }
    )

    first_axis = tmp_path / "first" / "axis.json"
    second_axis = tmp_path / "second" / "axis.json"
    first_descriptor = write_semantic_journal(first_axis, _result_with_events(first_events))
    second_descriptor = write_semantic_journal(second_axis, _result_with_events(second_events))

    assert first_descriptor["sha256"] == second_descriptor["sha256"]
    first_physical = first_descriptor["physical_telemetry"]
    second_physical = second_descriptor["physical_telemetry"]
    assert isinstance(first_physical, dict) and isinstance(second_physical, dict)
    assert first_physical["global_receipt_sha256"] != second_physical["global_receipt_sha256"]
    assert list(
        iter_verified_physical_telemetry(
            semantic_bundle_path(first_axis, first_descriptor),
            first_physical,
        )
    )
    assert list(
        iter_verified_physical_telemetry(
            semantic_bundle_path(second_axis, second_descriptor),
            second_physical,
        )
    )


def test_semantic_journal_streaming_writer_has_bounded_incremental_memory(
    tmp_path: Path,
) -> None:
    events = _complete_events(screening_count=50_000)
    result = _result_with_events(events)
    tracemalloc.start()
    descriptor = write_semantic_journal(tmp_path / "axis.json", result)
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak_bytes < 32 * 1024 * 1024
    assert descriptor["compressed_bytes"] < 8 * 1024 * 1024
    assert descriptor["event_count"] == len(events)


def test_semantic_journal_externalizes_and_replays_runtime_spool_ledgers(
    tmp_path: Path,
) -> None:
    events = _complete_events(screening_count=600)
    trace = Stage03Trace(
        MeasurementConfig(
            record_runtime_semantic_events=True,
            externalize_runtime_semantic_events=True,
        )
    )
    for event in events:
        trace.record_runtime_semantic_event(
            str(event["semantic_stream"]),
            {
                key: value
                for key, value in event.items()
                if key
                not in {
                    "semantic_stream",
                    "semantic_event_id",
                    "runtime_causal_event_id",
                }
            },
        )
    result = _result_with_events([])
    result.measurement_trace = trace
    axis_path = tmp_path / "axis.json"

    descriptor = write_semantic_journal(axis_path, result)
    runtime = descriptor["runtime_spool_pipelines"]
    assert isinstance(runtime, dict)
    assert runtime["pipeline_count"] == 1
    assert "attempted_batch_ledger" not in runtime
    bundle = semantic_bundle_path(axis_path, descriptor)
    assert (bundle / str(runtime["path"])).is_file()
    assert len(list(iter_verified_semantic_journal(axis_path, descriptor))) == len(events)
    persistence = descriptor["persistence"]
    assert isinstance(persistence, dict)
    assert float(persistence["runtime_spool_union_seconds"]) > 0.0
    trace.release_runtime_semantic_storage()


def test_semantic_journal_replays_discarded_runtime_attempts_and_rejects_audit_tamper(
    tmp_path: Path,
) -> None:
    events = _complete_events(screening_count=600)
    trace = Stage03Trace(
        MeasurementConfig(
            record_runtime_semantic_events=True,
            externalize_runtime_semantic_events=True,
        )
    )

    def record(event: dict[str, object]) -> None:
        trace.record_runtime_semantic_event(
            str(event["semantic_stream"]),
            {
                key: value
                for key, value in event.items()
                if key
                not in {
                    "semantic_stream",
                    "semantic_event_id",
                    "runtime_causal_event_id",
                }
            },
        )

    for event in events[:300]:
        record(event)
    checkpoint = trace.snapshot_runtime_semantic_journal()
    for ordinal in range(300):
        trace.record_runtime_semantic_event(
            "screening",
            {
                "event_type": "screening_decision",
                "route_key": "discarded",
                "status": "passed",
                "ordinal": ordinal,
            },
        )
    trace.rollback_runtime_semantic_journal(checkpoint)
    for event in events[300:]:
        record(event)

    result = _result_with_events([])
    result.measurement_trace = trace
    axis_path = tmp_path / "axis.json"
    descriptor = write_semantic_journal(axis_path, result)
    assert len(list(iter_verified_semantic_journal(axis_path, descriptor))) == len(events)

    bundle = semantic_bundle_path(axis_path, descriptor)
    runtime_descriptor = descriptor["runtime_spool_pipelines"]
    assert isinstance(runtime_descriptor, dict)
    runtime_path = bundle / str(runtime_descriptor["path"])
    with gzip.open(runtime_path, "rb") as source:
        header = json.loads(source.readline())
        attempts = [json.loads(line) for line in source]
    assert any(record["row"]["status"] == "discarded" for record in attempts)
    audit = header["attempt_audit"]
    audit_path = bundle / str(audit["path"])
    raw = bytearray(audit_path.read_bytes())
    raw[-1] ^= 1
    audit_path.write_bytes(raw)

    with pytest.raises(RuntimeError, match="audit|corrupt|reconcile"):
        list(iter_verified_semantic_journal(axis_path, descriptor))
    trace.release_runtime_semantic_storage()


def test_semantic_journal_rejects_pipeline_ledger_tamper(tmp_path: Path) -> None:
    axis_path = tmp_path / "axis.json"
    descriptor = write_semantic_journal(
        axis_path,
        _result_with_events(_complete_events(screening_count=300)),
    )
    bundle = semantic_bundle_path(axis_path, descriptor)
    pipeline = descriptor["pipeline"]
    assert isinstance(pipeline, dict)
    ledger = pipeline["ledger"]
    assert isinstance(ledger, dict)
    ledger_path = bundle / str(ledger["path"])
    ledger_path.write_bytes(ledger_path.read_bytes() + b"tamper")

    with pytest.raises(RuntimeError, match="mismatch|does not reconcile"):
        list(iter_verified_semantic_journal(axis_path, descriptor))


def test_semantic_journal_rejects_tamper_and_path_escape(tmp_path: Path) -> None:
    axis_path = tmp_path / "axis.json"
    descriptor = write_semantic_journal(
        axis_path,
        _result_with_events(_complete_events()),
    )
    bundle = semantic_bundle_path(axis_path, descriptor)
    journal_path = bundle / str(descriptor["event_path"])
    journal_path.write_bytes(journal_path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        list(iter_verified_semantic_journal(axis_path, descriptor))

    escaped = {**descriptor, "path": "../outside.semantic.bundle"}
    with pytest.raises(ValueError, match="same directory"):
        list(iter_verified_semantic_journal(axis_path, escaped))


def test_semantic_journal_sidecar_failure_leaves_no_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axis_path = tmp_path / "axis.json"
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    trace = Stage03Trace(
        MeasurementConfig(
            record_runtime_semantic_events=True,
            externalize_runtime_semantic_events=True,
        )
    )
    for event in _complete_events():
        trace.record_runtime_semantic_event(
            str(event["semantic_stream"]),
            {
                key: value
                for key, value in event.items()
                if key
                not in {
                    "semantic_stream",
                    "semantic_event_id",
                    "runtime_causal_event_id",
                }
            },
        )
    result = _result_with_events([])
    result.measurement_trace = trace
    original_open = Path.open

    def fail_sidecar(path: Path, *args: object, **kwargs: object) -> object:
        if path.name == "events.sha256":
            raise OSError("injected sidecar failure")
        return original_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", fail_sidecar)
    with pytest.raises(OSError, match="injected sidecar failure"):
        write_semantic_journal(axis_path, result)

    assert not axis_path.with_suffix(".semantic.bundle").exists()
    assert not tuple(tmp_path.glob(".*.tmp-*"))
    assert trace.runtime_semantic_pipeline_evidence == ()
    assert not tuple(tmp_path.glob("stage052-runtime-semantic-*"))


def test_semantic_journal_publish_failure_leaves_no_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axis_path = tmp_path / "axis.json"
    bundle_path = axis_path.with_suffix(".semantic.bundle")

    def fail_publish(path: Path, target: Path) -> None:
        if target == bundle_path:
            raise OSError("injected publish failure")
        raise AssertionError("unexpected publish target")

    monkeypatch.setattr(
        "evrptw.stage052_semantic_journal.publish_no_replace",
        fail_publish,
    )
    with pytest.raises(OSError, match="injected publish failure"):
        write_semantic_journal(axis_path, _result_with_events(_complete_events()))
    assert not bundle_path.exists()
    assert not tuple(tmp_path.glob(".*.tmp-*"))


def test_semantic_journal_existing_bundle_does_not_detach_runtime_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    trace = Stage03Trace(
        MeasurementConfig(
            record_runtime_semantic_events=True,
            externalize_runtime_semantic_events=True,
        )
    )
    trace.record_runtime_semantic_event(
        "operator",
        {"event_type": "operator_event", "status": "accepted"},
    )
    result = _result_with_events([])
    result.measurement_trace = trace
    axis_path = tmp_path / "axis.json"
    axis_path.with_suffix(".semantic.bundle").mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        write_semantic_journal(axis_path, result)

    assert trace.runtime_semantic_pipeline_evidence == ()
    trace.release_runtime_semantic_storage()
    assert not tuple(tmp_path.glob("stage052-runtime-semantic-*"))


def test_semantic_journal_existing_bundle_is_immutable(tmp_path: Path) -> None:
    axis_path = tmp_path / "axis.json"
    bundle_path = axis_path.with_suffix(".semantic.bundle")
    bundle_path.mkdir()
    marker = bundle_path / "existing"
    marker.write_text("immutable\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_semantic_journal(axis_path, _result_with_events(_complete_events()))

    assert marker.read_text(encoding="utf-8") == "immutable\n"
    assert not tuple(tmp_path.glob(".*.tmp-*"))


def test_physical_telemetry_rejects_typed_array_tamper(tmp_path: Path) -> None:
    axis_path = tmp_path / "axis.json"
    descriptor = write_semantic_journal(
        axis_path,
        _result_with_events(_complete_events()),
    )
    bundle = semantic_bundle_path(axis_path, descriptor)
    physical = descriptor["physical_telemetry"]
    assert isinstance(physical, dict)
    files = physical["files"]
    assert isinstance(files, dict)
    completion = files["completion_order"]
    assert isinstance(completion, dict)
    completion_path = bundle / str(completion["path"])
    raw = bytearray(completion_path.read_bytes())
    raw[0] ^= 1
    completion_path.write_bytes(raw)

    with pytest.raises(RuntimeError, match="completion_order integrity mismatch"):
        list(iter_verified_physical_telemetry(bundle, physical))
