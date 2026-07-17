from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import evrptw.artifacts as artifacts_module
from evrptw.artifacts import (
    ArtifactBudgetExceeded,
    ArtifactBundleWriter,
    ArtifactReader,
    ArtifactRunContext,
    ArtifactStorageConfig,
    require_current_storage_config,
    storage_format_for_run,
    verify_manifest,
    write_solver_result_bundle,
)
from evrptw.measurement import MeasurementConfig, Stage03Trace, canonical_route_key


def _writer(tmp_path: Path, *, max_instance: int = 2 * 1024 * 1024) -> ArtifactBundleWriter:
    return ArtifactBundleWriter(
        tmp_path / "results" / "stage03.2_cache_incremental_attempt01",
        ArtifactRunContext(
            "stage03.2",
            "cache_incremental",
            "stage03.2_cache_incremental_attempt01",
        ),
        ArtifactStorageConfig(
            per_instance_seed_max_bytes=max_instance,
            per_run_max_bytes=16 * 1024 * 1024,
        ),
    )


def test_new_runs_require_canonical_label_and_enabled_storage() -> None:
    with pytest.raises(ValueError, match="enabled"):
        require_current_storage_config(
            "stage03.2_cache_incremental_attempt01",
            None,
        )
    with pytest.raises(ValueError, match="canonical"):
        require_current_storage_config("stage032_legacy_attempt01", ArtifactStorageConfig())


def test_v2_streams_shards_with_local_event_identity_and_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "stage05.2_artifact_streaming_attempt01"
    writer = ArtifactBundleWriter(
        run_dir,
        ArtifactRunContext(
            "stage05.2",
            "artifact_streaming",
            "stage05.2_artifact_streaming_attempt01",
        ),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    for ordinal, seed in enumerate((2014, 2015)):
        writer.write_instance_seed(
            instance="toy",
            seed=seed,
            shard_ordinal=ordinal,
            worker_identity=f"worker-{ordinal}",
            raw_payload={},
            solution_payload={},
            trace_payload={},
            environment_payload={},
            route_dictionary={"route:2:C1": ("C1",)},
            critical_events=[
                {
                    "record_type": "route_evaluation",
                    "event_type": "route_evaluation",
                    "route_key": "route:2:C1",
                    "exact_started": True,
                    "exact_completed": True,
                }
            ],
        )
    bundle = writer.finalize()
    reader = ArtifactReader(bundle.run_dir)
    for seed in (2014, 2015):
        prefix = f"toy/{seed}/stage05.2_artifact_streaming_attempt01"
        events = reader.read_events(f"{prefix}_events_toy_{seed}.parquet")
        assert events[0]["event_id"] == 1
        shard_manifest = reader.read_json(f"{prefix}_shard_manifest_toy_{seed}.json")
        assert shard_manifest["storage_policy_version"] == "artifact-storage-v2"
        assert shard_manifest["evidence_completeness"] == "complete"
    batches = list(
        reader.iter_parquet_batches(
            "toy/2014/stage05.2_artifact_streaming_attempt01_events_toy_2014.parquet",
            columns=("event_id", "event_type"),
        )
    )
    assert sum(batch.num_rows for batch in batches) == 1
    assert batches[0].schema.names == ["event_id", "event_type"]
    parquet = pq.ParquetFile(
        run_dir / "toy/2014/stage05.2_artifact_streaming_attempt01_events_toy_2014.parquet"
    )
    assert parquet.metadata.row_group(0).num_rows <= 65_536


def test_v2_shard_session_appends_axes_before_finalization(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "stage05.2_artifact_streaming_attempt02"
    writer = ArtifactBundleWriter(
        run_dir,
        ArtifactRunContext(
            "stage05.2",
            "artifact_streaming",
            "stage05.2_artifact_streaming_attempt02",
        ),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    shard.append(
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=(
            {
                "event_type": "route_evaluation",
                "benchmark_axis": "fixed_work",
                "route_key": "route:2:C1",
            },
        ),
        diagnostic_rows=(),
    )
    shard.flush()
    shard.append(
        route_dictionary={"route:2:C2": ("C2",)},
        critical_events=(
            {
                "event_type": "route_evaluation",
                "benchmark_axis": "wall_clock_30",
                "route_key": "route:2:C2",
            },
            {
                "event_type": "screening_decision",
                "benchmark_axis": "wall_clock_30",
                "route_key": "route:2:C2",
                "status": "pass",
                "decision_id": 1,
                "checks": ({"check": "capacity", "status": "pass", "value": True},),
            },
        ),
        diagnostic_rows=(),
    )
    assert shard.max_buffered_groups_observed <= 2
    shard.finalize(
        raw_payload={"axes": {}},
        solution_payload={"axes": {}},
        trace_payload={"axes": {}},
        environment_payload={},
    )
    bundle = writer.finalize()
    with pytest.raises(RuntimeError, match="finalized"):
        shard.abort("late abort")
    reader = ArtifactReader(bundle.run_dir)
    prefix = "toy/2014/stage05.2_artifact_streaming_attempt02"
    events = reader.read_events(f"{prefix}_events_toy_2014.parquet")
    assert [event["event_id"] for event in events] == [1, 2, 3]
    assert [event["event_type"] for event in events] == [
        "route_evaluation",
        "route_evaluation",
        "screening_decision",
    ]
    routes = reader.read_parquet(f"{prefix}_route_dictionary_toy_2014.parquet")
    assert {tuple(row["customer_sequence"]) for row in routes} == {("C1",), ("C2",)}


def test_v2_parent_adopts_worker_shards_without_reading_event_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "stage05.2_job_parallel_attempt01"
    context = ArtifactRunContext("stage05.2", "job_parallel", "stage05.2_job_parallel_attempt01")
    config = ArtifactStorageConfig(storage_policy_version="artifact-storage-v2")
    worker = ArtifactBundleWriter(run_dir, context, config)
    worker.write_instance_seed(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
        route_dictionary={},
        critical_events=[],
    )
    parent = ArtifactBundleWriter(run_dir, context, config)
    parent.adopt_v2_shards(expected_identities=(("toy", 2014),))
    bundle = parent.finalize()
    manifest = verify_manifest(bundle.run_dir)
    assert any(item["artifact_type"] == "shard_manifest" for item in manifest["artifacts"])


def test_v2_abort_closes_every_sink_after_one_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "results" / "stage05.2_artifact_streaming_attempt03"
    writer = ArtifactBundleWriter(
        run_dir,
        ArtifactRunContext(
            "stage05.2",
            "artifact_streaming",
            "stage05.2_artifact_streaming_attempt03",
        ),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    attempted: list[str] = []
    original_close = artifacts_module._StreamingParquetSink.close

    def injected_close(
        sink: artifacts_module._StreamingParquetSink,
    ) -> tuple[int, str]:
        attempted.append(sink.path.name)
        if "_events_" in sink.path.name:
            raise OSError("injected EIO")
        return original_close(sink)

    monkeypatch.setattr(artifacts_module._StreamingParquetSink, "close", injected_close)

    shard.abort("worker failed")

    assert len(attempted) == 5
    prefix = run_dir / "toy" / "2014" / "stage05.2_artifact_streaming_attempt03"
    failure = json.loads(Path(f"{prefix}_failure_toy_2014.json").read_text(encoding="utf-8"))
    assert failure["cleanup_failures"]
    manifest = json.loads(
        Path(f"{prefix}_shard_manifest_toy_2014.json").read_text(encoding="utf-8")
    )
    assert manifest["evidence_completeness"] == "partial"
    assert Path(f"{prefix}_shard_manifest_toy_2014.sha256").is_file()
    parent = ArtifactBundleWriter(run_dir, writer.context, writer.config)
    parent.adopt_v2_shards(expected_identities=(("toy", 2014),), require_complete=False)
    parent.finalize(status="partial", evidence_completeness="partial")
    verified = verify_manifest(run_dir)
    assert verified["evidence_completeness"] == "partial"


def test_v2_failure_shard_preserves_abrupt_worker_fragment(tmp_path: Path) -> None:
    run_label = "stage05.2_artifact_streaming_attempt04"
    run_dir = tmp_path / "results" / run_label
    context = ArtifactRunContext("stage05.2", "artifact_streaming", run_label)
    config = ArtifactStorageConfig(storage_policy_version="artifact-storage-v2")
    directory = run_dir / "toy" / "2014"
    directory.mkdir(parents=True)
    fragment = directory / f"{run_label}_events_toy_2014.parquet"
    fragment.write_bytes(b"abrupt worker fragment")
    original_failure = directory / f"{run_label}_failure_toy_2014.json"
    original_failure_bytes = b'{"specific_worker_error":"EIO"}'
    original_failure.write_bytes(original_failure_bytes)

    writer = ArtifactBundleWriter(run_dir, context, config)
    writer.write_v2_failure_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="failure-recorder",
        error="worker process exited",
    )

    assert fragment.read_bytes() == b"abrupt worker fragment"
    archived_failures = tuple(directory.glob(f"{run_label}_partial_fragment_failure_*.bin"))
    assert len(archived_failures) == 1
    assert archived_failures[0].read_bytes() == original_failure_bytes
    manifest_path = directory / f"{run_label}_shard_manifest_toy_2014.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["evidence_completeness"] == "partial"
    assert any(
        item["artifact_type"] == "partial_shard_fragment"
        and item["relative_path"] == str(fragment.relative_to(run_dir))
        for item in manifest["artifacts"]
    )
    assert manifest_path.with_suffix(".sha256").is_file()


def test_v2_finalize_sidecar_failure_recovers_adoptable_partial_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_label = "stage05.2_artifact_streaming_attempt05"
    run_dir = tmp_path / "results" / run_label
    context = ArtifactRunContext("stage05.2", "artifact_streaming", run_label)
    config = ArtifactStorageConfig(storage_policy_version="artifact-storage-v2")
    worker = ArtifactBundleWriter(run_dir, context, config)
    shard = worker.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    shard.append(
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=({"event_type": "failure_probe", "route_key": "route:2:C1"},),
    )
    original_write_text = Path.write_text
    injected = False

    def fail_first_shard_sidecar(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        nonlocal injected
        if not injected and path.name.endswith("_shard_manifest_toy_2014.sha256"):
            injected = True
            raise OSError("injected sidecar EIO")
        return original_write_text(
            path,
            data,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "write_text", fail_first_shard_sidecar)
    with pytest.raises(OSError, match="sidecar EIO"):
        shard.finalize(
            raw_payload={},
            solution_payload={},
            trace_payload={},
            environment_payload={},
            failure_payload={"specific_failure": "before manifest"},
        )
    assert injected

    parent = ArtifactBundleWriter(run_dir, context, config)
    parent.adopt_v2_shards(expected_identities=(("toy", 2014),), require_complete=False)
    parent.finalize(status="partial", evidence_completeness="partial")
    manifest = verify_manifest(run_dir)
    assert manifest["evidence_completeness"] == "partial"


def test_writer_uses_canonical_layout_and_manifest_checksums(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.write_instance_seed(
        instance="toy",
        seed=2014,
        raw_payload={"record": {"status": "feasible"}},
        solution_payload={"routes": [["C1"]]},
        trace_payload={
            "trace_schema_version": "stage03-trace-v3",
            "route_table": "route_dictionary.parquet",
        },
        environment_payload={"python": "3.13"},
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=[
            {
                "record_type": "route_evaluation",
                "event_type": "route_evaluation",
                "route_key": "route:2:C1",
                "kind": "exact_call",
                "exact_started": True,
                "exact_completed": True,
                "feasible": True,
            },
            {
                "record_type": "screening_decision",
                "event_type": "screening_decision",
                "route_key": "route:2:C1",
                "decision_id": 1,
                "status": "pass",
                "checks": [{"check": "capacity", "status": "pass", "value": True}],
            },
        ],
        diagnostic_rows=[
            {
                "lane": "legacy",
                "operator": "relocate",
                "metric": "candidate_rejected",
                "count": 4,
            }
        ],
    )
    result = writer.finalize()

    instance_dir = result.run_dir / "toy" / "2014"
    assert (
        instance_dir / "stage03.2_cache_incremental_attempt01_events_toy_2014.parquet"
    ).is_file()
    assert result.manifest_path.parent.name == "control"
    manifest = verify_manifest(result.run_dir)
    assert manifest["storage_policy_version"] == "artifact-storage-v1"
    assert manifest["artifact_status"]["failure"] == "not_applicable"
    assert storage_format_for_run(result.run_dir) == "current"

    event_path = instance_dir / "stage03.2_cache_incremental_attempt01_events_toy_2014.parquet"
    events = pq.read_table(event_path).to_pylist()
    assert [row["event_id"] for row in events] == [1, 2]
    assert events[0]["route_id"] == 1
    assert "C1" not in json.dumps(events[0], sort_keys=True)

    route_path = instance_dir / (
        "stage03.2_cache_incremental_attempt01_route_dictionary_toy_2014.parquet"
    )
    route_rows = pq.read_table(route_path).to_pylist()
    assert route_rows[0]["customer_sequence"] == ["C1"]

    checks_path = instance_dir / (
        "stage03.2_cache_incremental_attempt01_screening_checks_toy_2014.parquet"
    )
    assert pq.read_table(checks_path).to_pylist()[0]["check"] == "capacity"


def test_generic_solver_adapter_registers_routes_from_neighborhood_events(
    tmp_path: Path,
) -> None:
    writer = _writer(tmp_path)

    class InstanceStub:
        name = "toy"

    class ResultStub:
        routes = (("C1",),)
        feasible = True
        objective = type("Objective", (), {"key": (1, 2.0, 0.0, 0)})()
        measurement_trace = None
        neighborhood_events = (
            {
                "event_type": "candidate_state",
                "route_key": canonical_route_key(("C2",)),
            },
        )
        failure_reason = ""

    write_solver_result_bundle(
        writer,
        instance=InstanceStub(),
        seed=2014,
        result=ResultStub(),
        raw_record={"status": "feasible"},
        environment_payload={},
    )
    bundle = writer.finalize()
    route_rows = ArtifactReader(bundle.run_dir).read_parquet(
        "toy/2014/stage03.2_cache_incremental_attempt01_route_dictionary_toy_2014.parquet"
    )
    assert {tuple(row["customer_sequence"]) for row in route_rows} == {("C1",), ("C2",)}


def test_writer_preserves_partial_failure_when_budget_is_exceeded(tmp_path: Path) -> None:
    writer = _writer(tmp_path, max_instance=1)
    with pytest.raises(ArtifactBudgetExceeded):
        writer.write_instance_seed(
            instance="toy",
            seed=2014,
            raw_payload={"large": "payload"},
            solution_payload={"routes": []},
            trace_payload={},
            environment_payload={},
            route_dictionary={},
            critical_events=[],
        )

    failure = (
        tmp_path
        / "results"
        / "stage03.2_cache_incremental_attempt01"
        / "toy"
        / "2014"
        / "stage03.2_cache_incremental_attempt01_failure_toy_2014.json"
    )
    assert failure.is_file()
    assert json.loads(failure.read_text(encoding="utf-8"))["evidence_completeness"] == "partial"
    manifest = verify_manifest(tmp_path / "results" / "stage03.2_cache_incremental_attempt01")
    assert manifest["evidence_completeness"] == "partial"
    assert manifest["artifact_status"]["failure"] == "present"


def test_manifest_detects_parquet_tampering(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.write_instance_seed(
        instance="toy",
        seed=2014,
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
        route_dictionary={},
        critical_events=[],
    )
    result = writer.finalize()
    event_path = (
        result.run_dir
        / "toy"
        / "2014"
        / ("stage03.2_cache_incremental_attempt01_events_toy_2014.parquet")
    )
    event_path.write_bytes(event_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_manifest(result.run_dir)


def test_manifest_ignores_external_volume_appledouble_metadata(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.write_instance_seed(
        instance="toy",
        seed=2014,
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
        route_dictionary={},
        critical_events=[],
    )
    result = writer.finalize()
    (result.run_dir / "control" / "._fake_manifest.json").write_bytes(b"appledouble")
    (result.run_dir / "toy" / "2014" / "._events.parquet").write_bytes(b"appledouble")
    manifest = verify_manifest(result.run_dir)
    assert manifest["status"] == "complete"


def test_event_ids_are_global_and_trace_json_is_an_index(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    for seed in (2014, 2015):
        writer.write_instance_seed(
            instance="toy",
            seed=seed,
            raw_payload={},
            solution_payload={},
            trace_payload={"config": {"enabled": False}},
            environment_payload={},
            route_dictionary={"route:2:C1": ("C1",)},
            critical_events=[
                {
                    "record_type": "route_evaluation",
                    "event_type": "route_evaluation",
                    "route_key": "route:2:C1",
                    "kind": "exact_call",
                    "started_at": 0.1,
                    "completed_at": 0.2,
                    "exact_started": True,
                    "exact_completed": True,
                }
            ],
        )
    result = writer.finalize()
    reader = ArtifactReader(result.run_dir)
    first = reader.read_events(
        "toy/2014/stage03.2_cache_incremental_attempt01_events_toy_2014.parquet"
    )
    second = reader.read_events(
        "toy/2015/stage03.2_cache_incremental_attempt01_events_toy_2015.parquet"
    )
    assert first[0]["event_id"] == 1
    assert second[0]["event_id"] == 2
    trace = reader.read_json("toy/2014/stage03.2_cache_incremental_attempt01_trace_toy_2014.json")
    assert "events" not in trace
    assert trace["trace_storage_version"] == "stage03-trace-index-v2"
    assert set(trace["schema_fingerprints"]) == {
        "route_dictionary",
        "events",
        "screening_checks",
        "diagnostic",
    }
    assert (
        pq.ParquetFile(
            result.run_dir
            / "toy/2014/stage03.2_cache_incremental_attempt01_events_toy_2014.parquet"
        )
        .metadata.row_group(0)
        .column(1)
        .compression
        == "ZSTD"
    )


def test_reader_reconstructs_trace_from_current_columns(tmp_path: Path) -> None:
    trace = Stage03Trace(MeasurementConfig())
    trace.record_route_evaluation(
        ("C1",),
        lane="legacy",
        iteration=1,
        operator="relocate",
        kind="exact_call",
        started_at=0.1,
        completed_at=0.2,
        exact_started=True,
        exact_completed=True,
        feasible=True,
        failure_reason="",
    )
    writer = _writer(tmp_path)
    writer.write_instance_seed(
        instance="toy",
        seed=2014,
        raw_payload={},
        solution_payload={},
        trace_payload=trace.to_dict(),
        environment_payload={},
        route_dictionary=trace.route_dictionary,
        critical_events=[
            {
                "record_type": "route_evaluation",
                "event_type": "route_evaluation",
                "route_key": next(iter(trace.route_dictionary)),
                "evaluation_id": 1,
                "lane": "legacy",
                "iteration": 1,
                "operator": "relocate",
                "kind": "exact_call",
                "started_at": 0.1,
                "completed_at": 0.2,
                "duration_seconds": 0.1,
                "exact_started": True,
                "exact_completed": True,
                "feasible": True,
            }
        ],
    )
    result = writer.finalize()
    reader = ArtifactReader(result.run_dir)
    payload = reader.reconstruct_trace(
        "toy/2014/stage03.2_cache_incremental_attempt01_trace_toy_2014.json"
    )
    restored = Stage03Trace.from_dict(payload)
    assert restored.route_dictionary == trace.route_dictionary
    assert len(restored.route_evaluations) == 1


def test_cache_lookup_and_result_are_one_persisted_event(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.write_instance_seed(
        instance="toy",
        seed=2014,
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=[
            {
                "record_type": "cache_event",
                "event_type": "cache_event",
                "operation": "lookup",
                "route_key": "route:2:C1",
                "cache_key_digest": "digest",
                "lane": "legacy",
                "iteration": 1,
                "operator": "relocate",
                "current_entries": 0,
                "current_bytes": 0,
            },
            {
                "record_type": "cache_event",
                "event_type": "cache_event",
                "operation": "miss",
                "route_key": "route:2:C1",
                "cache_key_digest": "digest",
                "lane": "legacy",
                "iteration": 1,
                "operator": "relocate",
                "current_entries": 0,
                "current_bytes": 0,
            },
        ],
    )
    result = writer.finalize()
    event_path = (
        result.run_dir
        / "toy"
        / "2014"
        / ("stage03.2_cache_incremental_attempt01_events_toy_2014.parquet")
    )
    rows = pq.read_table(event_path).to_pylist()
    assert len(rows) == 1
    assert rows[0]["operation"] == "lookup_result"
    assert rows[0]["lane_id"] == 0
    assert rows[0]["operator_id"] == 0
    assert "lane" not in rows[0]
    assert json.loads(rows[0]["extras_json"])["lookup_result"] == "miss"


def test_event_storage_keeps_explicit_ordering_audit_lists(tmp_path: Path) -> None:
    writer = ArtifactBundleWriter(
        tmp_path / "results" / "stage03.4_control_parallel_attempt01",
        ArtifactRunContext("stage03.4", "control_parallel", "stage03.4_control_parallel_attempt01"),
    )
    writer.write_control(metadata={})
    writer.write_instance_seed(
        instance="toy",
        seed=2014,
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
        route_dictionary={},
        critical_events=[
            {
                "event_type": "parallel_batch",
                "submission_order": [0, 1],
                "completion_order": [1, 0],
                "merge_order": [0, 1],
                "chunk_sizes": [2, 1],
                "completed_indices": [0, 1, 2],
            }
        ],
        diagnostic_rows=[],
    )
    result = writer.finalize()
    event_path = (
        result.run_dir
        / "toy"
        / "2014"
        / ("stage03.4_control_parallel_attempt01_events_toy_2014.parquet")
    )
    extras = json.loads(pq.read_table(event_path).to_pylist()[0]["extras_json"])
    assert extras["submission_order"] == [0, 1]
    assert extras["completion_order"] == [1, 0]
    assert extras["merge_order"] == [0, 1]
    assert extras["chunk_sizes"] == [2, 1]
    assert extras["completed_indices"] == [0, 1, 2]
