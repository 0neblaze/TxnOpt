from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

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
        shard_manifest = reader.read_json(
            f"{prefix}_shard_manifest_toy_{seed}.json"
        )
        assert shard_manifest["storage_policy_version"] == "artifact-storage-v2"
        assert shard_manifest["evidence_completeness"] == "complete"
    batches = list(
        reader.iter_parquet_batches(
            "toy/2014/stage05.2_artifact_streaming_attempt01_events_toy_2014.parquet"
        )
    )
    assert sum(batch.num_rows for batch in batches) == 1
    parquet = pq.ParquetFile(
        run_dir
        / "toy/2014/stage05.2_artifact_streaming_attempt01_events_toy_2014.parquet"
    )
    assert parquet.metadata.row_group(0).num_rows <= 65_536


def test_v2_parent_adopts_worker_shards_without_reading_event_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "stage05.2_job_parallel_attempt01"
    context = ArtifactRunContext(
        "stage05.2", "job_parallel", "stage05.2_job_parallel_attempt01"
    )
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
                "checks": [
                    {"check": "capacity", "status": "pass", "value": True}
                ],
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
        instance_dir
        / "stage03.2_cache_incremental_attempt01_events_toy_2014.parquet"
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
    assert {
        tuple(row["customer_sequence"])
        for row in route_rows
    } == {("C1",), ("C2",)}


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
    manifest = verify_manifest(
        tmp_path / "results" / "stage03.2_cache_incremental_attempt01"
    )
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
    event_path = result.run_dir / "toy" / "2014" / (
        "stage03.2_cache_incremental_attempt01_events_toy_2014.parquet"
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
    trace = reader.read_json(
        "toy/2014/stage03.2_cache_incremental_attempt01_trace_toy_2014.json"
    )
    assert "events" not in trace
    assert trace["trace_storage_version"] == "stage03-trace-index-v2"
    assert set(trace["schema_fingerprints"]) == {
        "route_dictionary",
        "events",
        "screening_checks",
        "diagnostic",
    }
    assert pq.ParquetFile(
        result.run_dir
        / "toy/2014/stage03.2_cache_incremental_attempt01_events_toy_2014.parquet"
    ).metadata.row_group(0).column(1).compression == "ZSTD"


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
    event_path = result.run_dir / "toy" / "2014" / (
        "stage03.2_cache_incremental_attempt01_events_toy_2014.parquet"
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
        ArtifactRunContext(
            "stage03.4", "control_parallel", "stage03.4_control_parallel_attempt01"
        ),
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
    event_path = result.run_dir / "toy" / "2014" / (
        "stage03.4_control_parallel_attempt01_events_toy_2014.parquet"
    )
    extras = json.loads(pq.read_table(event_path).to_pylist()[0]["extras_json"])
    assert extras["submission_order"] == [0, 1]
    assert extras["completion_order"] == [1, 0]
    assert extras["merge_order"] == [0, 1]
    assert extras["chunk_sizes"] == [2, 1]
    assert extras["completed_indices"] == [0, 1, 2]
