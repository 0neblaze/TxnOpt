from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import evrptw.artifacts as artifacts_module
from evrptw.artifacts import (
    V2_SCREENING_DECISIONS_SCHEMA_V1,
    V3_SCREENING_DEFINITIONS_SCHEMA,
    V3_SCREENING_OCCURRENCES_SCHEMA,
    ArtifactBundleWriter,
    ArtifactIntegrityError,
    ArtifactReader,
    ArtifactRunContext,
    ArtifactStorageConfig,
    atomic_write_signed_json,
    signed_sidecar_matches,
)
from evrptw.stage052_campaign import AnytimeCheckpoint, BatchManifest, VolumeIdentity
from evrptw.stage052_evidence import (
    BatchPersistenceEnvelope,
    PersistenceInterval,
    Stage052PersistenceAttribution,
)


def _v3_writer(tmp_path: Path, *, attempt: int = 5) -> ArtifactBundleWriter:
    run_label = f"stage05.2_artifact_streaming_attempt{attempt:02d}"
    return ArtifactBundleWriter(
        tmp_path / "results" / run_label,
        ArtifactRunContext("stage05.2", "artifact_streaming", run_label),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    )


def _screening_event(*, decision_id: int, started_at: float) -> dict[str, object]:
    return {
        "event_type": "screening_decision",
        "benchmark_axis": "fixed_work",
        "route_key": "route:2:C1",
        "lane": "constraint",
        "operator": "station_pressure",
        "iteration": 4,
        "decision_id": decision_id,
        "status": "pass",
        "reason": "",
        "demand": 2.0,
        "distance_lower_bound": 3.5,
        "distance_increment_lower_bound": 0.5,
        "exact_call_blocked": False,
        "first_failed_check": "",
        "min_time_window_slack": 4.0,
        "negative_cache_hit": False,
        "single_segment_reachable": True,
        "structural_energy_lower_bound": 1.5,
        "started_at": started_at,
        "completed_at": started_at + 0.25,
        "duration_seconds": 0.25,
        "checks": (
            {
                "check": "capacity",
                "status": "pass",
                "value": True,
                "reason": "",
            },
        ),
    }


def test_trace_dictionary_ids_must_be_unique_canonical_decimals() -> None:
    with pytest.raises(ArtifactIntegrityError, match="duplicate or invalid"):
        artifacts_module._invert_trace_dictionary(  # noqa: SLF001
            {"1": "legacy", "01": "constraint"},
            "lane",
        )


def test_screening_check_group_has_a_fixed_eight_check_bound() -> None:
    rows = (
        {
            "decision_event_id": 1,
            "check_index": index,
            "check": f"check-{index}",
            "status": "pass",
            "value_type": "none",
            "reason": "",
        }
        for index in range(9)
    )

    with pytest.raises(ArtifactIntegrityError, match="eight-check"):
        list(artifacts_module._group_screening_checks(rows))


def test_signed_json_second_replace_failure_restores_previous_generation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign_manifest.json"
    atomic_write_signed_json(path, {"generation": 1})
    previous_payload = path.read_bytes()
    previous_sidecar = path.with_suffix(".sha256").read_bytes()
    calls = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected sidecar replace failure")
        source.replace(destination)

    with pytest.raises(OSError, match="sidecar replace"):
        atomic_write_signed_json(
            path,
            {"generation": 2},
            _replace=fail_second_replace,
        )

    assert path.read_bytes() == previous_payload
    assert path.with_suffix(".sha256").read_bytes() == previous_sidecar
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    )


@pytest.mark.parametrize(("replace_count", "expected_generation"), ((1, 1), (2, 2)))
def test_signed_json_process_death_leaves_a_verifiable_generation(
    tmp_path: Path,
    replace_count: int,
    expected_generation: int,
) -> None:
    path = tmp_path / "campaign_manifest.json"
    atomic_write_signed_json(path, {"generation": 1})
    program = """
import os
import sys
from pathlib import Path
from evrptw.artifacts import atomic_write_signed_json

calls = 0
stop_after = int(sys.argv[2])

def replace_then_exit(source: Path, destination: Path) -> None:
    global calls
    calls += 1
    os.replace(source, destination)
    if calls == stop_after:
        os._exit(73)

atomic_write_signed_json(
    Path(sys.argv[1]),
    {"generation": 2},
    _replace=replace_then_exit,
)
"""

    completed = subprocess.run(
        [sys.executable, "-c", program, str(path), str(replace_count)],
        check=False,
    )

    assert completed.returncode == 73
    assert json.loads(path.read_text(encoding="utf-8"))["generation"] == expected_generation
    assert signed_sidecar_matches(path)
    assert len(path.with_suffix(".sha256").read_text(encoding="utf-8").splitlines()) == 2

    atomic_write_signed_json(path, {"generation": 3})
    assert json.loads(path.read_text(encoding="utf-8"))["generation"] == 3
    assert len(path.with_suffix(".sha256").read_text(encoding="utf-8").splitlines()) == 1


def test_v3_writer_rejects_more_than_eight_screening_checks(tmp_path: Path) -> None:
    writer = _v3_writer(tmp_path, attempt=92)
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    event = {
        **_screening_event(decision_id=9, started_at=1.25),
        "checks": tuple(
            {
                "check": f"check-{index}",
                "status": "pass",
                "value": True,
                "reason": "",
            }
            for index in range(9)
        ),
    }

    with pytest.raises(ArtifactIntegrityError, match="eight-check"):
        shard.append(
            route_dictionary={"route:2:C1": ("C1",)},
            critical_events=(event,),
        )

    shard.abort("expected screening-domain rejection")
    writer.finalize(status="partial", evidence_completeness="partial")


def test_reader_accepts_pre_amendment_v2_manifest_without_physical_schema_field(
    tmp_path: Path,
) -> None:
    run_label = "stage05.2_native_kernels_attempt03"
    writer = ArtifactBundleWriter(
        tmp_path / run_label,
        ArtifactRunContext("stage05.2", "native_kernels", run_label),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v2",
        ),
    )
    bundle = writer.finalize()
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    del manifest["storage_policy"]["screening_schema_version"]
    bundle.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bundle.manifest_sidecar_path.write_text(
        hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )

    reader = ArtifactReader(bundle.run_dir)

    assert reader.manifest["storage_policy_version"] == "artifact-storage-v2"


def test_reader_accepts_only_the_signed_post_manifest_persistence_envelope(
    tmp_path: Path,
) -> None:
    writer = _v3_writer(tmp_path, attempt=98)
    bundle = writer.finalize()
    attribution = (
        bundle.run_dir
        / "control"
        / f"{writer.context.run_label}_persistence_attribution.json"
    )
    atomic_write_signed_json(
        attribution,
        {
            "run_label": writer.context.run_label,
            "schema_version": "stage05.2-persistence-attribution-v1",
        },
    )

    assert ArtifactReader(bundle.run_dir).manifest["run_label"] == writer.context.run_label

    attribution.with_suffix(".sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="persistence attribution sidecar"):
        ArtifactReader(bundle.run_dir)


def test_reader_rejects_foreign_batch_attribution_at_campaign_root(tmp_path: Path) -> None:
    writer = _v3_writer(tmp_path, attempt=99)
    bundle = writer.finalize()
    attribution = (
        bundle.run_dir
        / "control"
        / f"{writer.context.run_label}_batch0007_persistence_attribution.json"
    )
    atomic_write_signed_json(attribution, {"run_label": writer.context.run_label})

    with pytest.raises(ArtifactIntegrityError, match="directory role"):
        ArtifactReader(bundle.run_dir)


def test_reader_rejects_batch_state_envelope_at_campaign_root(
    tmp_path: Path,
) -> None:
    writer = _v3_writer(tmp_path, attempt=95)
    bundle = writer.finalize()
    envelope = bundle.run_dir / "batch_persistence_envelope.json"
    atomic_write_signed_json(
        envelope,
        {
            "schema_version": "stage05.2-batch-persistence-envelope-v1",
            "run_label": writer.context.run_label,
            "batch_id": "batch0001",
        },
    )

    with pytest.raises(ArtifactIntegrityError, match="canonical batch root"):
        ArtifactReader(bundle.run_dir)


def test_reader_accepts_strict_cross_bound_batch_post_manifest_envelopes(
    tmp_path: Path,
) -> None:
    run_label = "stage05.2_benchmark_attempt95"
    batch_id = "batch0001"
    batch_dir = tmp_path / "results" / run_label / batch_id
    writer = ArtifactBundleWriter(
        batch_dir,
        ArtifactRunContext("stage05.2", "benchmark", run_label),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    )
    bundle = writer.finalize()
    attribution = Stage052PersistenceAttribution(
        run_label=run_label,
        component="benchmark",
        scope="pilot",
        subject_id=batch_id,
        primary_manifest_relative_path=bundle.manifest_path.relative_to(batch_dir).as_posix(),
        primary_manifest_sha256=hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest(),
        solver_seconds=1.0,
        shard_persistence_seconds=0.0,
        control_intervals=(),
    )
    attribution_path = (
        batch_dir / "control" / f"{run_label}_{batch_id}_persistence_attribution.json"
    )
    atomic_write_signed_json(attribution_path, attribution.to_dict())
    orphan_envelope = BatchPersistenceEnvelope(
        run_label=run_label,
        batch_id=batch_id,
        base_attribution_sha256=hashlib.sha256(attribution_path.read_bytes()).hexdigest(),
        verified_manifest_sha256="4" * 64,
        archived_manifest_sha256="5" * 64,
        solver_seconds=1.0,
        base_persistence_seconds=0.0,
        state_intervals=(
            PersistenceInterval("verified_batch_manifest_write", 1, 1),
            PersistenceInterval("archived_batch_manifest_write", 2, 2),
        ),
    )
    orphan_path = batch_dir / "batch_persistence_envelope.json"
    atomic_write_signed_json(orphan_path, orphan_envelope.to_dict())
    with pytest.raises(ArtifactIntegrityError, match="requires the archived batch manifest"):
        ArtifactReader(batch_dir)
    orphan_path.unlink()
    orphan_path.with_suffix(".sha256").unlink()
    verified_batch = BatchManifest(
        run_label=run_label,
        batch_id=batch_id,
        status="verified",
        root_alias="staging",
        archive_root_alias="archive",
        logical_path=f"{run_label}/{batch_id}",
        volume=VolumeIdentity("device", "apfs"),
        shard_ids=("shard0001",),
        estimated_bytes=1,
        checksum_sha256="1" * 64,
        actual_bytes=1,
        row_count=0,
        physical_schema="screening_decisions_v3",
        resource_summary_sha256="2" * 64,
        persistence_attribution_sha256=hashlib.sha256(
            attribution_path.read_bytes()
        ).hexdigest(),
        control_persistence_seconds=0.0,
        persistence_ratio=0.0,
        shard_manifest_sha256_by_id={"shard0001": "3" * 64},
        shard_actual_bytes_by_id={"shard0001": 1},
    )
    batch_path = batch_dir / "batch_manifest.json"
    atomic_write_signed_json(batch_path, verified_batch.to_dict())

    ArtifactReader(batch_dir)

    archived_batch = verified_batch.mark_archived(
        root_alias="archive",
        volume=VolumeIdentity("device", "apfs"),
        transfer_mode="same_volume_atomic_rename",
        archive_transfer_seconds=0.0,
    )
    atomic_write_signed_json(batch_path, archived_batch.to_dict())

    ArtifactReader(batch_dir)

    envelope = BatchPersistenceEnvelope(
        run_label=run_label,
        batch_id=batch_id,
        base_attribution_sha256=hashlib.sha256(attribution_path.read_bytes()).hexdigest(),
        verified_manifest_sha256="4" * 64,
        archived_manifest_sha256=hashlib.sha256(batch_path.read_bytes()).hexdigest(),
        solver_seconds=1.0,
        base_persistence_seconds=0.0,
        state_intervals=(
            PersistenceInterval("verified_batch_manifest_write", 1, 1),
            PersistenceInterval("archived_batch_manifest_write", 2, 2),
        ),
    )
    atomic_write_signed_json(
        batch_dir / "batch_persistence_envelope.json",
        envelope.to_dict(),
    )

    ArtifactReader(batch_dir)

    tampered = envelope.to_dict()
    tampered["archived_manifest_sha256"] = "5" * 64
    atomic_write_signed_json(batch_dir / "batch_persistence_envelope.json", tampered)
    with pytest.raises(ArtifactIntegrityError, match="does not bind"):
        ArtifactReader(batch_dir)


def test_reader_does_not_broaden_the_post_manifest_envelope_exception(
    tmp_path: Path,
) -> None:
    writer = _v3_writer(tmp_path, attempt=96)
    bundle = writer.finalize()
    malformed = (
        bundle.run_dir
        / "control"
        / f"{writer.context.run_label}_batch7_persistence_attribution.json"
    )
    atomic_write_signed_json(malformed, {"run_label": writer.context.run_label})

    with pytest.raises(ArtifactIntegrityError, match="directory role"):
        ArtifactReader(bundle.run_dir)


def test_v3_writes_repeated_screening_definitions_once_and_occurrences_separately(
    tmp_path: Path,
) -> None:
    writer = _v3_writer(tmp_path)
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    shard.append(
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=(
            _screening_event(decision_id=9, started_at=1.25),
            _screening_event(decision_id=10, started_at=2.25),
        ),
    )
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    bundle = writer.finalize()

    prefix = bundle.run_dir / "toy" / "2014" / writer.context.run_label
    definitions_path = Path(f"{prefix}_screening_definitions_toy_2014.parquet")
    occurrences_path = Path(f"{prefix}_screening_occurrences_toy_2014.parquet")
    assert pq.read_table(definitions_path).num_rows == 1
    assert pq.read_table(occurrences_path).num_rows == 2

    reader = ArtifactReader(bundle.run_dir)
    events = reader.read_events(f"toy/2014/{writer.context.run_label}_events_toy_2014.parquet")
    assert [row["decision_id"] for row in events] == [9, 10]
    assert [row["embedded_checks"][0]["value"] for row in events] == [True, True]


def test_benchmark_v3_trace_is_declared_as_a_compact_index(tmp_path: Path) -> None:
    run_label = "stage05.2_benchmark_attempt01"
    writer = ArtifactBundleWriter(
        tmp_path / "results" / run_label,
        ArtifactRunContext("stage05.2", "benchmark", run_label),
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
    shard.append(route_dictionary={}, critical_events=())
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={"campaign_trace_schema_version": "stage05.2-campaign-trace-v1"},
        environment_payload={},
        anytime_checkpoints=(
            {
                "instance": "c101C5",
                "seed": 2014,
                "axis_budget_seconds": 30,
                "checkpoint_seconds": 1,
                "objective_key": [2, 100.0, 3.0, 1],
                "source": "verified_initial_incumbent",
                "incumbent_completed_at_seconds": 0.0,
                "incumbent_iteration": None,
            },
        ),
    )
    manifest = ArtifactReader(writer.finalize().run_dir).manifest

    traces = [item for item in manifest["artifacts"] if item["artifact_type"] == "trace"]
    assert len(traces) == 1
    assert traces[0]["artifact_subtype"] == "compact_index_v1"
    trace = ArtifactReader(writer.run_dir).read_json(traces[0]["relative_path"])
    assert trace["trace_storage_version"] == "stage03-trace-index-v2"
    assert trace["campaign_trace_schema_version"] == "stage05.2-campaign-trace-v1"
    checkpoint_refs = [
        item for item in manifest["artifacts"] if item["artifact_type"] == "anytime_checkpoints"
    ]
    assert len(checkpoint_refs) == 1
    assert checkpoint_refs[0]["artifact_subtype"] == "checkpoint_v1"
    checkpoint_rows = list(
        ArtifactReader(writer.run_dir).iter_parquet_rows(
            checkpoint_refs[0]["relative_path"],
            batch_size=1,
        )
    )
    assert AnytimeCheckpoint.from_dict(checkpoint_rows[0]).objective_key == (
        2,
        100.0,
        3.0,
        1,
    )


def _write_schema_bundle(
    tmp_path: Path,
    *,
    run_label: str,
    config: ArtifactStorageConfig,
) -> tuple[ArtifactReader, str]:
    component = run_label.removeprefix("stage05.2_").rsplit("_attempt", 1)[0]
    writer = ArtifactBundleWriter(
        tmp_path / "results" / run_label,
        ArtifactRunContext("stage05.2", component, run_label),
        config,
    )
    event = _screening_event(decision_id=9, started_at=1.25)
    if config.storage_policy_version == "artifact-storage-v2":
        shard = writer.open_v2_shard(
            instance="toy",
            seed=2014,
            shard_ordinal=0,
            worker_identity="worker-0",
        )
        shard.append(
            route_dictionary={"route:2:C1": ("C1",)},
            critical_events=(event,),
        )
        shard.finalize(
            raw_payload={},
            solution_payload={},
            trace_payload={},
            environment_payload={},
        )
    else:
        writer.write_instance_seed(
            instance="toy",
            seed=2014,
            raw_payload={},
            solution_payload={},
            trace_payload={},
            environment_payload={},
            route_dictionary={"route:2:C1": ("C1",)},
            critical_events=(event,),
        )
    bundle = writer.finalize()
    return (
        ArtifactReader(bundle.run_dir),
        f"toy/2014/{run_label}_trace_toy_2014.json",
    )


def test_v1_definition_json_v2_and_v3_expand_to_equal_logical_screening_events(
    tmp_path: Path,
) -> None:
    v1_reader, v1_trace = _write_schema_bundle(
        tmp_path,
        run_label="stage05.2_hot_path_attempt81",
        config=ArtifactStorageConfig(),
    )
    v2_reader, v2_trace = _write_schema_bundle(
        tmp_path,
        run_label="stage05.2_artifact_streaming_attempt82",
        config=ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    v3_reader, v3_trace = _write_schema_bundle(
        tmp_path,
        run_label="stage05.2_artifact_streaming_attempt83",
        config=ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    )

    logical = [
        reader.reconstruct_trace(trace)["screening_decisions"]
        for reader, trace in (
            (v1_reader, v1_trace),
            (v2_reader, v2_trace),
            (v3_reader, v3_trace),
        )
    ]
    assert logical[0] == logical[1] == logical[2]


def test_v3_transcodes_old_v2_arrow_batches_without_logical_drift(tmp_path: Path) -> None:
    source, source_trace_ref = _write_schema_bundle(
        tmp_path,
        run_label="stage05.2_artifact_streaming_attempt87",
        config=ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    source_items = {
        (str(item.get("artifact_type")), str(item.get("artifact_subtype"))): item
        for item in source.manifest["artifacts"]
        if isinstance(item, dict) and str(item.get("relative_path", "")).startswith("toy/2014/")
    }
    child_writer = _v3_writer(tmp_path, attempt=88)
    child = child_writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )

    def batches(artifact_type: str, subtype: str) -> object:
        reference = source_items[(artifact_type, subtype)]
        return pq.ParquetFile(source.run_dir / reference["relative_path"]).iter_batches(
            batch_size=65_536
        )

    source_trace = source.read_json(source_trace_ref)
    child.append_transcoded_v2_batches(
        route_batches=batches("route_dictionary", "canonical_routes"),
        critical_event_batches=batches("events", "critical"),
        screening_check_batches=batches("events", "screening_checks"),
        screening_decision_batches=batches("events", "screening_decisions_v2"),
        diagnostic_batches=batches("diagnostic", "aggregated"),
        lane_dictionary=source_trace["lane_dictionary"],
        operator_dictionary=source_trace["operator_dictionary"],
    )
    child.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    child_reader = ArtifactReader(child_writer.finalize().run_dir)
    source_events_ref = source_trace_ref.replace("_trace_", "_events_").replace(
        ".json", ".parquet"
    )
    child_events_ref = (
        f"toy/2014/{child_writer.context.run_label}_events_toy_2014.parquet"
    )

    assert list(source.iter_events(source_events_ref)) == list(
        child_reader.iter_events(child_events_ref)
    )


def test_iter_events_streams_v1_checks_as_writer_input(tmp_path: Path) -> None:
    reader, trace_ref = _write_schema_bundle(
        tmp_path,
        run_label="stage05.2_hot_path_attempt80",
        config=ArtifactStorageConfig(),
    )
    events_ref = trace_ref.replace("_trace_", "_events_").replace(".json", ".parquet")

    rows = list(reader.iter_events(events_ref, batch_size=1))

    assert rows[0]["lane"] == "constraint"
    assert rows[0]["operator"] == "station_pressure"
    assert rows[0]["checks"] == [
        {"check": "capacity", "status": "pass", "value": True, "reason": ""}
    ]


def test_v2_reader_screening_scratch_uses_explicit_staging_root_not_tmpdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader, trace_ref = _write_schema_bundle(
        tmp_path,
        run_label="stage05.2_artifact_streaming_attempt84",
        config=ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    events_ref = trace_ref.replace("_trace_", "_events_").replace(".json", ".parquet")
    foreign_tmp = tmp_path / "foreign-system-tmp"
    foreign_tmp.mkdir()
    staging_scratch = tmp_path / "external-staging" / "scratch"
    monkeypatch.setenv("TMPDIR", str(foreign_tmp))

    rows = list(
        reader.iter_events(
            events_ref,
            batch_size=1,
            scratch_root=staging_scratch,
        )
    )

    assert len(rows) == 1
    assert staging_scratch.is_dir()
    assert list(staging_scratch.iterdir()) == []
    assert list(foreign_tmp.iterdir()) == []


def test_v3_writer_never_calls_table_from_pylist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_arrow = artifacts_module.pa

    class GuardedTable:
        from_arrays = staticmethod(pa.Table.from_arrays)

        @staticmethod
        def from_pylist(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("v3 writer called Table.from_pylist")

    class GuardedArrow(SimpleNamespace):
        def __getattr__(self, name: str) -> object:
            return getattr(real_arrow, name)

    monkeypatch.setattr(artifacts_module, "pa", GuardedArrow(Table=GuardedTable))

    writer = _v3_writer(tmp_path, attempt=84)
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    shard.append(
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=(_screening_event(decision_id=9, started_at=1.25),),
    )
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    writer.finalize()


def test_iter_events_streams_v3_logical_rows_without_full_table_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = _v3_writer(tmp_path, attempt=85)
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
                "route_key": "route:2:C1",
                "lane": "legacy",
                "operator": "relocate",
                "iteration": 3,
                "exact_started": True,
            },
            _screening_event(decision_id=9, started_at=1.25),
            {
                "event_type": "deadline_boundary",
                "lane": "constraint",
                "operator": "station_pressure",
                "iteration": 5,
                "status": "complete",
            },
        ),
    )
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    bundle = writer.finalize()
    reader = ArtifactReader(bundle.run_dir)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("streaming API used a full-table reader")

    monkeypatch.setattr(reader, "read_events", forbidden)
    monkeypatch.setattr(artifacts_module.pq, "read_table", forbidden)
    rows = list(
        reader.iter_events(
            f"toy/2014/{writer.context.run_label}_events_toy_2014.parquet",
            batch_size=1,
        )
    )

    assert [row["event_type"] for row in rows] == [
        "route_evaluation",
        "screening_decision",
        "deadline_boundary",
    ]
    assert rows[0]["lane"] == "legacy"
    assert rows[1]["checks"] == [
        {"check": "capacity", "status": "pass", "value": True, "reason": ""}
    ]
    assert rows[1]["route_id"] is not None
    assert rows[1]["timestamp_seconds"] == 1.25
    assert rows[1]["duration_seconds"] == 0.25


def test_iter_events_rows_can_be_streamed_directly_into_a_v3_writer(
    tmp_path: Path,
) -> None:
    source_writer = _v3_writer(tmp_path, attempt=86)
    source_shard = source_writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="source",
    )
    source_shard.append(
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=(
            _screening_event(decision_id=9, started_at=1.25),
            {
                "event_type": "route_evaluation",
                "route_key": "route:2:C1",
                "lane": "legacy",
                "operator": "relocate",
                "iteration": 5,
                "exact_started": True,
            },
        ),
    )
    source_shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    source = source_writer.finalize()
    source_reader = ArtifactReader(source.run_dir)
    source_prefix = f"toy/2014/{source_writer.context.run_label}"

    target_writer = _v3_writer(tmp_path, attempt=87)
    target_shard = target_writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="target",
    )
    route_rows = source_reader.iter_parquet_rows(
        f"{source_prefix}_route_dictionary_toy_2014.parquet",
        batch_size=1,
    )
    target_shard.append(
        route_dictionary={
            str(row["canonical_route_key"]): tuple(row["customer_sequence"]) for row in route_rows
        },
        critical_events=source_reader.iter_events(
            f"{source_prefix}_events_toy_2014.parquet",
            batch_size=1,
        ),
    )
    target_shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    target = target_writer.finalize()
    target_reader = ArtifactReader(target.run_dir)
    target_prefix = f"toy/2014/{target_writer.context.run_label}"

    assert (
        source_reader.reconstruct_trace(f"{source_prefix}_trace_toy_2014.json")[
            "screening_decisions"
        ]
        == target_reader.reconstruct_trace(f"{target_prefix}_trace_toy_2014.json")[
            "screening_decisions"
        ]
    )


def test_iter_events_reads_the_historical_nested_v2_screening_schema(
    tmp_path: Path,
) -> None:
    writer = ArtifactBundleWriter(
        tmp_path / "results" / "stage05.2_artifact_streaming_attempt88",
        ArtifactRunContext(
            "stage05.2",
            "artifact_streaming",
            "stage05.2_artifact_streaming_attempt88",
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
        critical_events=(_screening_event(decision_id=9, started_at=1.25),),
    )
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    bundle = writer.finalize()
    prefix = f"toy/2014/{writer.context.run_label}"
    events_ref = f"{prefix}_events_toy_2014.parquet"
    current_reader = ArtifactReader(bundle.run_dir)
    expected = list(current_reader.iter_events(events_ref, batch_size=1))
    expanded = current_reader.read_events(events_ref)[0]
    extras = json.loads(str(expanded["extras_json"]))
    nested = {
        "event_id": expanded["event_id"],
        "timestamp_seconds": expanded["timestamp_seconds"],
        "started_at": expanded["started_at"],
        "completed_at": expanded["completed_at"],
        "duration_seconds": expanded["duration_seconds"],
        "lane_id": expanded["lane_id"],
        "iteration": expanded["iteration"],
        "operator_id": expanded["operator_id"],
        "route_id": expanded["route_id"],
        "status": expanded["status"],
        "reason": expanded["reason"],
        "decision_id": expanded["decision_id"],
        **extras,
        "checks": [
            {
                "check": "capacity",
                "status": "pass",
                "value_bool": True,
                "value_float": 1.0,
                "value_text": None,
                "reason": "",
            }
        ],
    }
    compact_path = (
        bundle.run_dir
        / "toy"
        / "2014"
        / f"{writer.context.run_label}_screening_decisions_toy_2014.parquet"
    )
    pq.write_table(
        pa.Table.from_pylist([nested], schema=V2_SCREENING_DECISIONS_SCHEMA_V1),
        compact_path,
    )

    historical_reader = ArtifactReader(bundle.run_dir, verify=False)
    assert list(historical_reader.iter_events(events_ref, batch_size=1)) == expected


def _write_single_v3_screening_bundle(
    tmp_path: Path,
    *,
    attempt: int,
) -> tuple[Path, str, Path, Path]:
    writer = _v3_writer(tmp_path, attempt=attempt)
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    shard.append(
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=(_screening_event(decision_id=9, started_at=1.25),),
    )
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    bundle = writer.finalize()
    prefix = bundle.run_dir / "toy" / "2014" / writer.context.run_label
    events_ref = f"toy/2014/{writer.context.run_label}_events_toy_2014.parquet"
    return (
        bundle.run_dir,
        events_ref,
        Path(f"{prefix}_screening_definitions_toy_2014.parquet"),
        Path(f"{prefix}_screening_occurrences_toy_2014.parquet"),
    )


def test_v3_producer_screening_scratch_stays_on_shard_volume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign_tmp = tmp_path / "foreign-tmp"
    foreign_tmp.mkdir()
    monkeypatch.setenv("TMPDIR", str(foreign_tmp))
    writer = _v3_writer(tmp_path / "staging", attempt=88)
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )

    shard.append(
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=(_screening_event(decision_id=9, started_at=1.25),),
    )

    store = shard._screening_definition_store
    assert store is not None
    scratch = Path(store._temporary_directory.name)
    assert scratch.is_relative_to(writer.run_dir / "toy" / "2014")
    assert not scratch.is_relative_to(foreign_tmp)
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    assert not scratch.exists()


def test_v3_reader_rejects_a_tampered_definition(tmp_path: Path) -> None:
    run_dir, events_ref, definitions_path, _ = _write_single_v3_screening_bundle(
        tmp_path,
        attempt=89,
    )
    definitions = pq.read_table(definitions_path).to_pylist()
    definitions[0]["status"] = "reject"
    pq.write_table(
        pa.Table.from_pylist(definitions, schema=V3_SCREENING_DEFINITIONS_SCHEMA),
        definitions_path,
    )

    with pytest.raises(ArtifactIntegrityError, match="definition hash mismatch"):
        list(ArtifactReader(run_dir, verify=False).iter_events(events_ref, batch_size=1))


def test_v3_reader_rejects_an_unknown_definition_reference(tmp_path: Path) -> None:
    run_dir, events_ref, _, occurrences_path = _write_single_v3_screening_bundle(
        tmp_path,
        attempt=90,
    )
    occurrences = pq.read_table(occurrences_path).to_pylist()
    occurrences[0]["definition_id"] += 1
    pq.write_table(
        pa.Table.from_pylist(occurrences, schema=V3_SCREENING_OCCURRENCES_SCHEMA),
        occurrences_path,
    )

    with pytest.raises(ArtifactIntegrityError, match="unknown definition"):
        list(ArtifactReader(run_dir, verify=False).iter_events(events_ref, batch_size=1))


def test_v3_row_groups_and_simultaneous_buffers_stay_bounded(tmp_path: Path) -> None:
    writer = _v3_writer(tmp_path, attempt=91)
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    shard.append(
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=(
            _screening_event(decision_id=index, started_at=float(index)) for index in range(65_537)
        ),
    )
    assert shard.max_buffered_groups_observed <= 2
    assert shard.max_pending_screening_transaction_rows_observed <= 1_024
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    bundle = writer.finalize()
    occurrences_path = (
        bundle.run_dir
        / "toy"
        / "2014"
        / f"{writer.context.run_label}_screening_occurrences_toy_2014.parquet"
    )
    parquet = pq.ParquetFile(occurrences_path)

    assert parquet.metadata.num_row_groups == 2
    assert [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ] == [65_536, 1]


def test_v3_occurrences_append_schema_ordered_values_without_row_mappings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_append = artifacts_module._StreamingParquetSink.append  # noqa: SLF001
    original_append_values = artifacts_module._StreamingParquetSink.append_values  # noqa: SLF001

    def reject_occurrence_mapping(
        sink: artifacts_module._StreamingParquetSink,  # noqa: SLF001
        row: dict[str, object],
    ) -> None:
        if sink.schema.equals(V3_SCREENING_OCCURRENCES_SCHEMA):
            raise AssertionError("v3 occurrence hotspot constructed a row mapping")
        original_append(sink, row)

    def reject_individual_occurrence_values(
        sink: artifacts_module._StreamingParquetSink,  # noqa: SLF001
        values: tuple[object, ...],
    ) -> None:
        if sink.schema.equals(V3_SCREENING_OCCURRENCES_SCHEMA):
            raise AssertionError("v3 occurrence hotspot appended one row at a time")
        original_append_values(sink, values)

    monkeypatch.setattr(
        artifacts_module._StreamingParquetSink,  # noqa: SLF001
        "append",
        reject_occurrence_mapping,
    )
    monkeypatch.setattr(
        artifacts_module._StreamingParquetSink,  # noqa: SLF001
        "append_values",
        reject_individual_occurrence_values,
    )
    writer = _v3_writer(tmp_path, attempt=92)
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )

    assert shard.append(
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=(_screening_event(decision_id=9, started_at=1.25),),
    ) == 1
    shard.abort("typed occurrence test complete")


def test_empty_v3_screening_transaction_does_not_flush_other_sinks(tmp_path: Path) -> None:
    writer = _v3_writer(tmp_path, attempt=93)
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    event = {
        "event_type": "route_evaluation",
        "route_key": "route:2:C1",
        "lane": "legacy",
        "operator": "relocate",
        "iteration": 3,
        "exact_started": True,
        "checks": (
            {"check": "capacity", "status": "pass", "value": True, "reason": ""},
        ),
    }
    for index in range(3):
        shard.append(
            route_dictionary={"route:2:C1": ("C1",)} if index == 0 else {},
            critical_events=(event,),
        )
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    bundle = writer.finalize()
    events_path = (
        bundle.run_dir
        / "toy"
        / "2014"
        / f"{writer.context.run_label}_events_toy_2014.parquet"
    )
    parquet = pq.ParquetFile(events_path)

    assert [
        parquet.metadata.row_group(index).num_rows
        for index in range(parquet.metadata.num_row_groups)
    ] == [3]


def test_v3_write_instance_seed_uses_the_same_physical_schema(tmp_path: Path) -> None:
    writer = _v3_writer(tmp_path, attempt=92)
    paths = writer.write_instance_seed(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=(_screening_event(decision_id=9, started_at=1.25),),
    )
    writer.finalize()

    assert "screening_definitions" in paths
    assert "screening_occurrences" in paths
    assert (writer.run_dir / paths["screening_definitions"]).is_file()
    assert (writer.run_dir / paths["screening_occurrences"]).is_file()


@pytest.mark.parametrize(
    "screening_schema_version",
    ["screening_decisions_v2", "screening_decisions_v3"],
)
def test_iter_events_resolves_evicted_definitions_from_bounded_disk_state(
    tmp_path: Path,
    screening_schema_version: str,
) -> None:
    attempt = 93 if screening_schema_version.endswith("v2") else 94
    run_label = f"stage05.2_artifact_streaming_attempt{attempt:02d}"
    writer = ArtifactBundleWriter(
        tmp_path / "results" / run_label,
        ArtifactRunContext("stage05.2", "artifact_streaming", run_label),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version=screening_schema_version,
        ),
    )
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    first = _screening_event(decision_id=1, started_at=1.0)
    second = {
        **_screening_event(decision_id=2, started_at=2.0),
        "route_key": "route:2:C2",
    }
    third = _screening_event(decision_id=3, started_at=3.0)
    shard.append(
        route_dictionary={"route:2:C1": ("C1",), "route:2:C2": ("C2",)},
        critical_events=(first, second, third),
    )
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    bundle = writer.finalize()

    rows = list(
        ArtifactReader(bundle.run_dir).iter_events(
            f"toy/2014/{run_label}_events_toy_2014.parquet",
            batch_size=1,
        )
    )
    assert [row["decision_id"] for row in rows] == [1, 2, 3]
    assert rows[0]["route_key"] == rows[2]["route_key"] != rows[1]["route_key"]
    assert rows[0]["route_id"] == rows[2]["route_id"] != rows[1]["route_id"]


def test_iter_events_restores_candidate_route_keys_for_global_best(
    tmp_path: Path,
) -> None:
    writer = _v3_writer(tmp_path, attempt=91)
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    route_keys = ("route:2:C1", "route:2:C2")
    shard.append(
        route_dictionary={route_keys[0]: ("C1",), route_keys[1]: ("C2",)},
        critical_events=(
            {
                "event_type": "candidate_state",
                "benchmark_axis": "wall_clock_30",
                "timestamp_seconds": 1.0,
                "iteration": 1,
                "status": "accepted",
                "candidate_feasible": True,
                "accepted": True,
                "global_best": True,
                "candidate_vehicle_count": 2,
                "candidate_vehicle_delta": 0,
                "candidate_route_keys": route_keys,
                "candidate_objective_key": [2, 10.0, 0.0, 0],
            },
        ),
    )
    shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    bundle = writer.finalize()

    rows = list(
        ArtifactReader(bundle.run_dir).iter_events(
            f"toy/2014/{writer.context.run_label}_events_toy_2014.parquet"
        )
    )

    assert rows[0]["candidate_route_keys"] == list(route_keys)
    assert len(rows[0]["candidate_route_ids"]) == 2


def test_v2_sequential_route_ids_migrate_to_v3_content_ids_by_key(
    tmp_path: Path,
) -> None:
    source_label = "stage05.2_artifact_streaming_attempt90"
    source_writer = ArtifactBundleWriter(
        tmp_path / "results" / source_label,
        ArtifactRunContext("stage05.2", "artifact_streaming", source_label),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v2",
        ),
    )
    route_keys = ("route:2:C1", "route:2:C2")
    source_writer.write_instance_seed(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="source-v2",
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
        route_dictionary={route_keys[0]: ("C1",), route_keys[1]: ("C2",)},
        critical_events=(
            {
                "event_type": "candidate_state",
                "benchmark_axis": "wall_clock_30",
                "timestamp_seconds": 1.0,
                "iteration": 1,
                "status": "accepted",
                "candidate_feasible": True,
                "accepted": True,
                "global_best": True,
                "route_key": route_keys[0],
                "route_keys": route_keys,
                "current_route_keys": (route_keys[0],),
                "candidate_route_keys": route_keys,
                "base_route_key": route_keys[0],
                "candidate_route_key": route_keys[1],
                "candidate_vehicle_count": 2,
                "candidate_vehicle_delta": 0,
                "candidate_objective_key": [2, 10.0, 0.0, 0],
            },
        ),
    )
    source = source_writer.finalize()
    source_event_ref = f"toy/2014/{source_label}_events_toy_2014.parquet"
    source_events = list(ArtifactReader(source.run_dir).iter_events(source_event_ref))
    assert source_events[0]["candidate_route_ids"] == [1, 2]

    target_writer = _v3_writer(tmp_path, attempt=89)
    target_shard = target_writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="target-v3",
    )
    target_shard.append(
        route_dictionary={route_keys[0]: ("C1",), route_keys[1]: ("C2",)},
        critical_events=source_events,
    )
    target_shard.finalize(
        raw_payload={},
        solution_payload={},
        trace_payload={},
        environment_payload={},
    )
    target = target_writer.finalize()
    target_ref = (
        f"toy/2014/{target_writer.context.run_label}_events_toy_2014.parquet"
    )
    migrated = list(ArtifactReader(target.run_dir).iter_events(target_ref))

    assert migrated[0]["route_key"] == route_keys[0]
    assert migrated[0]["route_keys"] == list(route_keys)
    assert migrated[0]["current_route_keys"] == [route_keys[0]]
    assert migrated[0]["candidate_route_keys"] == list(route_keys)
    assert migrated[0]["base_route_key"] == route_keys[0]
    assert migrated[0]["candidate_route_key"] == route_keys[1]
    assert migrated[0]["candidate_route_ids"] != [1, 2]


def test_v3_abort_retains_a_verified_partial_shard(tmp_path: Path) -> None:
    writer = _v3_writer(tmp_path, attempt=95)
    shard = writer.open_v2_shard(
        instance="toy",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    shard.append(
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=(_screening_event(decision_id=9, started_at=1.25),),
    )
    shard.flush()
    shard.abort("injected worker failure")
    bundle = writer.finalize(status="partial", evidence_completeness="partial")

    assert bundle.evidence_completeness == "partial"
    manifest = ArtifactReader(bundle.run_dir).manifest
    assert manifest["evidence_completeness"] == "partial"
    subtypes = {
        item["artifact_subtype"]
        for item in manifest["artifacts"]
        if item["artifact_type"] == "events"
    }
    assert {"screening_definitions_v3", "screening_occurrences_v3"} <= subtypes
    with pytest.raises(RuntimeError, match="not open"):
        shard.append(route_dictionary={}, critical_events=())
