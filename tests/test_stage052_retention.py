from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import threading
from dataclasses import replace
from pathlib import Path

import pytest

import evrptw.stage052_retention as retention
from evrptw.stage052_campaign import (
    BatchManifest,
    CampaignManifest,
    StorageRoot,
    StorageRootLocator,
    VolumeIdentity,
    directory_byte_count,
    directory_checksum,
    directory_file_count,
)
from evrptw.stage052_evidence import CAMPAIGN_PILOT_GATES
from evrptw.stage052_retention import (
    RetentionIntegrityError,
    Stage052RetentionPolicy,
    archive_stage052_inventory,
    archive_stage052_inventory_to_registry,
    audit_stage052_runs,
    create_supersession_receipt,
    load_retention_inventory,
    resolve_retained_run,
    resolve_retained_run_from_locator,
    write_retention_inventory,
    write_retention_registry,
)


def _run(
    root: Path,
    run_label: str,
    *,
    status: str,
    completeness: str,
    prerequisite: str | None = None,
) -> Path:
    run = root / run_label
    control = run / "control"
    control.mkdir(parents=True)
    payload = {
        "run_label": run_label,
        "status": status,
        "evidence_completeness": completeness,
        "source_commit": "a" * 40,
    }
    if prerequisite is not None:
        payload["prerequisite_run_label"] = prerequisite
    (control / f"{run_label}_manifest.json").write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    (run / "events.parquet").write_bytes(f"events:{run_label}".encode())
    return run


def _policy() -> Stage052RetentionPolicy:
    return Stage052RetentionPolicy(
        archive_root_alias="d_archive",
        archive_relative_base="stage05.2/history",
    )


def test_supersession_receipt_seals_interrupted_campaign_without_rewriting_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "results"
    run_label = "stage05.2_benchmark_attempt92"
    run = _run(
        source,
        run_label,
        status="planned",
        completeness="partial",
    )
    campaign = {
        "schema_version": "stage05.2-campaign-manifest-v1",
        "run_label": run_label,
        "status": "planned",
        "scope": "formal",
        "batches": [
            {"batch_id": f"batch{index:04d}", "status": "archived" if index <= 6 else "planned"}
            for index in range(1, 13)
        ],
    }
    campaign_bytes = (json.dumps(campaign, indent=2, sort_keys=True) + "\n").encode()
    campaign_path = run / "campaign_manifest.json"
    campaign_path.write_bytes(campaign_bytes)
    campaign_path.with_suffix(".sha256").write_text(
        hashlib.sha256(campaign_bytes).hexdigest() + "\n",
        encoding="ascii",
    )
    (run / "batch0007").mkdir()
    host_log = tmp_path / "task-host.log"
    host_log.write_text(
        "\n".join(
            (
                "HOST_STARTED 2026-07-28T20:00:39Z pid=460 ppid=459 "
                "run_id=20260728T200039Z-460 "
                "launch_nonce=fc094d4828bf457681bda14091f98e70 "
                "controller_sha256=" + "a" * 64,
                "HOST_CONTROLLER_FAILED 2026-07-29T02:14:23Z "
                "run_id=20260728T200039Z-460 exit_code=143",
                "HOST_STOPPED 2026-07-29T02:14:23Z pid=460 exit_code=143",
                "",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retention, "_matching_run_processes", lambda _run_label: ())
    original_manifest_sha256 = hashlib.sha256(campaign_path.read_bytes()).hexdigest()

    receipt_sha256 = create_supersession_receipt(
        run,
        host_log=host_log,
        host_run_id="20260728T200039Z-460",
        launch_nonce="fc094d4828bf457681bda14091f98e70",
        expected_exit_code=143,
        reason="operator_terminated_for_candidate_transaction_redesign",
        created_at_utc="2026-07-29T03:00:00Z",
    )
    inventory = audit_stage052_runs(source, _policy())

    assert hashlib.sha256(campaign_path.read_bytes()).hexdigest() == original_manifest_sha256
    assert inventory.records[0].status == "superseded"
    assert inventory.records[0].evidence_completeness == "partial"
    receipt_path = run / "control" / "supersession_receipt.json"
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == receipt_sha256
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["archived_batch_ids"] == [
        "batch0001",
        "batch0002",
        "batch0003",
        "batch0004",
        "batch0005",
        "batch0006",
    ]
    assert receipt["planned_batch_ids"] == [
        "batch0007",
        "batch0008",
        "batch0009",
        "batch0010",
        "batch0011",
        "batch0012",
    ]
    assert receipt["active_process_count"] == 0


def test_supersession_receipt_rejects_live_campaign_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run(
        tmp_path / "results",
        "stage05.2_benchmark_attempt92",
        status="planned",
        completeness="partial",
    )
    campaign = {
        "run_label": run.name,
        "status": "planned",
        "scope": "formal",
        "batches": [{"batch_id": "batch0001", "status": "archived"}],
    }
    payload = (json.dumps(campaign, sort_keys=True) + "\n").encode()
    (run / "campaign_manifest.json").write_bytes(payload)
    (run / "campaign_manifest.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n",
        encoding="ascii",
    )
    host_log = tmp_path / "task-host.log"
    host_log.write_text(
        "HOST_CONTROLLER_FAILED 2026-07-29T02:14:23Z "
        "run_id=20260728T200039Z-460 exit_code=143\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(retention, "_matching_run_processes", lambda _run_label: (1234,))

    with pytest.raises(RetentionIntegrityError, match="still has active processes"):
        create_supersession_receipt(
            run,
            host_log=host_log,
            host_run_id="20260728T200039Z-460",
            launch_nonce="fc094d4828bf457681bda14091f98e70",
            expected_exit_code=143,
            reason="operator_terminated_for_candidate_transaction_redesign",
        )


def _campaign_with_external_batch(
    source: Path,
    archive: Path,
) -> tuple[Path, Path, StorageRootLocator, Path]:
    run_label = "stage05.2_benchmark_attempt01"
    run = _run(
        source,
        run_label,
        status="complete",
        completeness="complete",
    )
    volume = VolumeIdentity(device_uuid="fixture-device", filesystem="fixturefs")
    locator = StorageRootLocator(
        {"d_archive": StorageRoot("d_archive", archive, volume)}
    )
    locator_path = source.parent / "stage052_storage_roots.local.toml"
    locator_path.write_text(
        "\n".join(
            (
                "[roots.d_archive]",
                f"absolute_path = {json.dumps(str(archive))}",
                'device_uuid = "fixture-device"',
                'filesystem = "fixturefs"',
                "",
            )
        ),
        encoding="utf-8",
    )
    batch_dir = archive / run_label / "batch0001"
    batch_dir.mkdir(parents=True)
    (batch_dir / "payload.bin").write_bytes(b"x" * 128)
    shard_ids = tuple(f"shard{index:04d}" for index in range(1, 37))
    batch = BatchManifest(
        run_label=run_label,
        batch_id="batch0001",
        status="archived",
        root_alias="d_archive",
        archive_root_alias="d_archive",
        logical_path=f"{run_label}/batch0001",
        volume=volume,
        shard_ids=shard_ids,
        estimated_bytes=128,
        checksum_sha256=directory_checksum(batch_dir),
        actual_bytes=directory_byte_count(batch_dir),
        row_count=1,
        physical_schema="screening_decisions_v3",
        resource_summary_sha256="b" * 64,
        persistence_attribution_sha256="c" * 64,
        control_persistence_seconds=0.01,
        persistence_ratio=0.01,
        shard_manifest_sha256_by_id={shard_id: "d" * 64 for shard_id in shard_ids},
        shard_actual_bytes_by_id={shard_id: 1 for shard_id in shard_ids},
        transfer_mode="same_volume_atomic_rename",
        archive_transfer_seconds=0.01,
    )
    batch_manifest = (
        json.dumps(batch.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode()
    (batch_dir / "batch_manifest.json").write_bytes(batch_manifest)
    (batch_dir / "batch_manifest.sha256").write_text(
        hashlib.sha256(batch_manifest).hexdigest() + "\n",
        encoding="ascii",
    )
    envelope = b'{"schema_version":"fixture-envelope-v1"}\n'
    envelope_sha256 = hashlib.sha256(envelope).hexdigest()
    (batch_dir / "batch_persistence_envelope.json").write_bytes(envelope)
    (batch_dir / "batch_persistence_envelope.sha256").write_text(
        envelope_sha256 + "\n",
        encoding="ascii",
    )
    campaign = CampaignManifest(
        run_label=run_label,
        status="complete",
        scope="pilot",
        configuration_sha256="e" * 64,
        prerequisite_review_sha256="f" * 64,
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=6,
        native_profile="stage05.2-native-kernels-v1",
        storage_policy_version="artifact-storage-v2",
        screening_schema_version="screening_decisions_v3",
        storage_roots={"d_archive": volume},
        shard_count=36,
        axis_count=36,
        declared_solver_seconds=1_080,
        checkpoint_count=144,
        batches=(batch,),
        batch_persistence_envelope_sha256_by_id={
            "batch0001": envelope_sha256
        },
    )
    campaign_path = run / "campaign_manifest.json"
    campaign_bytes = (
        json.dumps(campaign.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode()
    campaign_path.write_bytes(campaign_bytes)
    campaign_path.with_suffix(".sha256").write_text(
        hashlib.sha256(campaign_bytes).hexdigest() + "\n",
        encoding="ascii",
    )
    storage_identity = {
        "schema_version": "stage05.2-storage-publication-identity-v1",
        "run_label": run_label,
        "batches": [
            {
                "root_alias": "d_archive",
                "relative_path": f"{run_label}/batch0001",
                "file_count": directory_file_count(batch_dir),
                "byte_count": directory_byte_count(batch_dir),
                "tree_sha256": directory_checksum(batch_dir),
            }
        ],
    }
    review = {
        "schema_version": "stage05.2-campaign-review-v2",
        "run_label": run_label,
        "component": "benchmark",
        "scope": "pilot",
        "status": "READY_FOR_STAGE052_FORMAL_BENCHMARK",
        "gates": {
            gate: {"passed": True, "detail": "fixture"}
            for gate in sorted(CAMPAIGN_PILOT_GATES)
        },
        "review_execution_required": True,
        "storage_publication_identity": storage_identity,
        "storage_publication_identity_sha256": hashlib.sha256(
            json.dumps(
                storage_identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    review_dir = run / "review"
    review_dir.mkdir()
    review_manifest = review_dir / "review_manifest.json"
    review_manifest.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (review_dir / "review_execution.json").write_text(
        json.dumps(
            {
                "run_label": run_label,
                "finalized": True,
                "status": "completed",
                "systemd_service_result": "success",
                "cgroup_memory_peak_status": "verified",
                "raw_manifest_unchanged": True,
                "review_manifest_sha256": hashlib.sha256(
                    review_manifest.read_bytes()
                ).hexdigest(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return run, batch_dir, locator, locator_path


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def test_audit_records_status_identity_size_and_prerequisites(tmp_path: Path) -> None:
    source = tmp_path / "results"
    prerequisite = "stage05.2_hot_path_attempt03"
    _run(
        source,
        "stage05.2_artifact_streaming_attempt05",
        status="NOT_READY",
        completeness="partial",
        prerequisite=prerequisite,
    )

    inventory = audit_stage052_runs(
        source,
        _policy(),
        created_at_utc="2026-07-23T12:00:00Z",
    )

    assert inventory.schema_version == "stage05.2-retention-inventory-v1"
    assert inventory.directory_count == 1
    assert inventory.total_bytes > 0
    record = inventory.records[0]
    assert record.run_label == "stage05.2_artifact_streaming_attempt05"
    assert record.component == "artifact_streaming"
    assert record.status == "NOT_READY"
    assert record.evidence_completeness == "partial"
    assert record.source_commit == "a" * 40
    assert record.prerequisite_run_labels == (prerequisite,)
    assert record.file_count == 2
    assert len(record.tree_sha256) == 64
    assert record.disposition == "planned_archive"


def test_audit_selects_only_explicit_immutable_run_labels(tmp_path: Path) -> None:
    source = tmp_path / "results"
    selected = _run(
        source,
        "stage05.2_benchmark_attempt01",
        status="complete",
        completeness="complete",
    )
    untouched = _run(
        source,
        "stage05.2_benchmark_attempt02",
        status="failed",
        completeness="partial",
    )

    inventory = audit_stage052_runs(
        source,
        _policy(),
        run_labels=(selected.name,),
    )

    assert tuple(record.run_label for record in inventory.records) == (selected.name,)
    assert selected.is_dir()
    assert untouched.is_dir()
    with pytest.raises(RetentionIntegrityError, match="selected run is missing"):
        audit_stage052_runs(
            source,
            _policy(),
            run_labels=("stage05.2_benchmark_attempt03",),
        )


def test_repository_config_declares_single_current_retention_policy() -> None:
    repository = Path(__file__).resolve().parents[1]

    policy = Stage052RetentionPolicy.from_toml(
        repository / "configs" / "stage052_performance.toml"
    )

    assert policy == _policy()


def test_audit_uses_run_status_not_nested_prerequisite_status(tmp_path: Path) -> None:
    source = tmp_path / "results"
    run_label = "stage05.2_perf_baseline_attempt03"
    run = _run(source, run_label, status="complete", completeness="complete")
    metadata = {
        "run_label": run_label,
        "repository_revision": "b" * 40,
        "stage051_prerequisite": {"status": "READY_FOR_STAGE05_2"},
    }
    (run / "control" / f"{run_label}_run_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    review = run / "review"
    review.mkdir()
    (review / "review_manifest.json").write_text(
        json.dumps({"run_label": run_label, "status": "READY_FOR_STAGE052_HOT_PATH"}),
        encoding="utf-8",
    )

    record = audit_stage052_runs(source, _policy()).records[0]

    assert record.status == "READY_FOR_STAGE052_HOT_PATH"
    assert record.source_commit == "a" * 40 + "|" + "b" * 40


def test_audit_uses_current_review_status_not_failed_retry_history(tmp_path: Path) -> None:
    source = tmp_path / "results"
    run_label = "stage05.2_hot_path_attempt03"
    run = _run(source, run_label, status="complete", completeness="complete")
    review = run / "review"
    history = review / "history" / ("b" * 64)
    history.mkdir(parents=True)
    (history / "review_manifest.json").write_text(
        json.dumps({"run_label": run_label, "status": "NOT_READY"}),
        encoding="utf-8",
    )
    (review / "review_manifest.json").write_text(
        json.dumps(
            {
                "run_label": run_label,
                "status": "READY_FOR_STAGE052_ARTIFACT_STREAMING",
            }
        ),
        encoding="utf-8",
    )

    record = audit_stage052_runs(source, _policy()).records[0]

    assert record.status == "READY_FOR_STAGE052_ARTIFACT_STREAMING"


@pytest.mark.parametrize(
    ("status", "completeness"),
    [
        ("READY_FOR_STAGE052_JOB_PARALLEL", "complete"),
        ("NOT_READY", "complete"),
        ("failed", "partial"),
        ("partial", "partial"),
        ("complete", "complete"),
        ("active", "partial"),
        ("accepted", "complete"),
    ],
)
def test_archive_moves_every_terminal_status_after_verification(
    tmp_path: Path,
    status: str,
    completeness: str,
) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    run = _run(
        source,
        "stage05.2_native_kernels_attempt07",
        status=status,
        completeness=completeness,
    )
    inventory = audit_stage052_runs(
        source,
        _policy(),
        allow_unsealed=status == "active",
        expected_directory_count=1 if status == "active" else None,
        expected_total_bytes=_directory_bytes(run) if status == "active" else None,
    )
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)

    archived = archive_stage052_inventory(
        inventory_path,
        inventory_sha256=inventory_sha256,
        archive_root=archive,
    )

    destination = archive / "stage05.2/history" / run.name
    assert not run.exists()
    assert destination.is_dir()
    assert archived[0].disposition == "archived"
    assert archived[0].verification_status == "verified"
    assert archived[0].tree_sha256 == inventory.records[0].tree_sha256

    repeated = archive_stage052_inventory(
        inventory_path,
        inventory_sha256=inventory_sha256,
        archive_root=archive,
    )
    assert repeated[0].disposition == "already_archived"
    assert repeated[0].verification_status == "verified"


@pytest.mark.parametrize("status", ["active", "in_progress"])
def test_audit_rejects_active_run_without_explicit_historical_override(
    tmp_path: Path,
    status: str,
) -> None:
    source = tmp_path / "results"
    _run(
        source,
        "stage05.2_hot_path_attempt03",
        status=status,
        completeness="partial",
    )

    with pytest.raises(RetentionIntegrityError, match="not sealed"):
        audit_stage052_runs(source, _policy())

    with pytest.raises(RetentionIntegrityError, match="requires expected"):
        audit_stage052_runs(source, _policy(), allow_unsealed=True)


def test_archive_rejects_stale_inventory_without_removing_source(tmp_path: Path) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    run = _run(
        source,
        "stage05.2_job_parallel_attempt18",
        status="READY_FOR_STAGE052_NATIVE_KERNELS",
        completeness="complete",
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)
    (run / "events.parquet").write_bytes(b"changed after audit")

    with pytest.raises(RetentionIntegrityError, match="changed after audit"):
        archive_stage052_inventory(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive,
        )

    assert run.is_dir()
    assert not (archive / "stage05.2/history" / run.name).exists()


def test_archive_preflights_all_records_before_moving_any_source(tmp_path: Path) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    first = _run(
        source,
        "stage05.2_artifact_streaming_attempt05",
        status="complete",
        completeness="complete",
    )
    stale = _run(
        source,
        "stage05.2_native_kernels_attempt07",
        status="failed",
        completeness="partial",
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)
    (stale / "events.parquet").write_bytes(b"changed after audit")

    with pytest.raises(RetentionIntegrityError, match="changed after audit"):
        archive_stage052_inventory(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive,
        )

    assert first.is_dir()
    assert stale.is_dir()
    assert not (archive / "stage05.2/history" / first.name).exists()


def test_registry_conflict_is_rejected_before_archive_moves_source(tmp_path: Path) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    registry = tmp_path / "retention.csv"
    run = _run(
        source,
        "stage05.2_job_parallel_attempt18",
        status="complete",
        completeness="complete",
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)
    conflict = replace(inventory.records[0], tree_sha256="f" * 64)
    write_retention_registry(registry, (conflict,))

    with pytest.raises(RetentionIntegrityError, match="identity conflict"):
        archive_stage052_inventory_to_registry(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive,
            registry_path=registry,
        )

    assert run.is_dir()
    assert not (archive / "stage05.2/history" / run.name).exists()


def test_campaign_archive_preserves_and_reverifies_signed_external_batches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    registry = tmp_path / "retention.csv"
    run, batch_dir, locator, locator_path = _campaign_with_external_batch(
        source,
        archive,
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)

    archived = archive_stage052_inventory_to_registry(
        inventory_path,
        inventory_sha256=inventory_sha256,
        archive_root=archive,
        registry_path=registry,
        storage_root_locator=locator,
    )

    retained = archive / "stage05.2/history" / run.name
    assert archived[0].verification_status == "verified"
    assert not run.exists()
    assert retained.is_dir()
    assert batch_dir == archive / run.name / "batch0001"
    assert (batch_dir / "payload.bin").read_bytes() == b"x" * 128
    assert (
        resolve_retained_run_from_locator(
            run.name,
            registry_path=registry,
            storage_root_locator_path=locator_path,
        )
        == retained
    )

    (batch_dir / "payload.bin").write_bytes(b"tampered")
    with pytest.raises(
        RetentionIntegrityError,
        match="campaign external storage identity mismatch",
    ):
        resolve_retained_run_from_locator(
            run.name,
            registry_path=registry,
            storage_root_locator_path=locator_path,
        )


def test_ready_campaign_without_storage_identity_blocks_metadata_archive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    registry = tmp_path / "retention.csv"
    run, _, locator, _ = _campaign_with_external_batch(source, archive)
    review_path = run / "review" / "review_manifest.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review.pop("storage_publication_identity")
    review.pop("storage_publication_identity_sha256")
    review_path.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    execution_path = run / "review" / "review_execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["review_manifest_sha256"] = hashlib.sha256(
        review_path.read_bytes()
    ).hexdigest()
    execution_path.write_text(
        json.dumps(execution, sort_keys=True),
        encoding="utf-8",
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)

    with pytest.raises(
        RetentionIntegrityError,
        match="storage publication identity is missing",
    ):
        archive_stage052_inventory_to_registry(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive,
            registry_path=registry,
            storage_root_locator=locator,
        )

    assert run.is_dir()
    assert not (archive / "stage05.2/history" / run.name).exists()


def test_not_ready_campaign_with_storage_identity_archives_as_failed_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    registry = tmp_path / "retention.csv"
    run, batch_dir, locator, _ = _campaign_with_external_batch(source, archive)
    review_path = run / "review" / "review_manifest.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["status"] = "NOT_READY"
    first_gate = next(iter(review["gates"].values()))
    first_gate["passed"] = False
    review_path.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    execution_path = run / "review" / "review_execution.json"
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    execution["review_manifest_sha256"] = hashlib.sha256(
        review_path.read_bytes()
    ).hexdigest()
    execution_path.write_text(
        json.dumps(execution, sort_keys=True),
        encoding="utf-8",
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)

    archived = archive_stage052_inventory_to_registry(
        inventory_path,
        inventory_sha256=inventory_sha256,
        archive_root=archive,
        registry_path=registry,
        storage_root_locator=locator,
    )

    assert archived[0].verification_status == "verified"
    assert not run.exists()
    assert (archive / "stage05.2/history" / run.name).is_dir()
    assert batch_dir.is_dir()


def test_interrupted_campaign_external_batch_drift_blocks_metadata_archive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    registry = tmp_path / "retention.csv"
    run, batch_dir, locator, _ = _campaign_with_external_batch(source, archive)
    shutil.rmtree(run / "review")
    campaign_path = run / "campaign_manifest.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    campaign["status"] = "planned"
    campaign_path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    campaign_path.with_suffix(".sha256").write_text(
        hashlib.sha256(campaign_path.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)
    (batch_dir / "payload.bin").write_bytes(b"tampered")

    with pytest.raises(
        RetentionIntegrityError,
        match="campaign external storage identity mismatch",
    ):
        archive_stage052_inventory_to_registry(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive,
            registry_path=registry,
            storage_root_locator=locator,
        )

    assert run.is_dir()
    assert not (archive / "stage05.2/history" / run.name).exists()


def test_campaign_storage_parent_symlink_blocks_metadata_archive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    registry = tmp_path / "retention.csv"
    run, batch_dir, locator, _ = _campaign_with_external_batch(source, archive)
    signed_parent = batch_dir.parent
    physical_parent = archive / "physical-campaign-storage"
    signed_parent.rename(physical_parent)
    signed_parent.symlink_to(physical_parent, target_is_directory=True)
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)

    with pytest.raises(
        RetentionIntegrityError,
        match="path or transfer state is unsafe",
    ):
        archive_stage052_inventory_to_registry(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive,
            registry_path=registry,
            storage_root_locator=locator,
        )

    assert run.is_dir()
    assert not (archive / "stage05.2/history" / run.name).exists()


def test_campaign_incoming_transfer_blocks_metadata_archive(tmp_path: Path) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    registry = tmp_path / "retention.csv"
    run, batch_dir, locator, _ = _campaign_with_external_batch(source, archive)
    incoming = batch_dir.with_name(f"{batch_dir.name}.incoming")
    incoming.mkdir()
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)

    with pytest.raises(
        RetentionIntegrityError,
        match="path or transfer state is unsafe",
    ):
        archive_stage052_inventory_to_registry(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive,
            registry_path=registry,
            storage_root_locator=locator,
        )

    assert run.is_dir()
    assert not (archive / "stage05.2/history" / run.name).exists()


def test_campaign_manifest_sidecar_drift_blocks_metadata_archive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    registry = tmp_path / "retention.csv"
    run, _, locator, _ = _campaign_with_external_batch(source, archive)
    (run / "campaign_manifest.sha256").write_text("0" * 64 + "\n", encoding="ascii")
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)

    with pytest.raises(
        RetentionIntegrityError,
        match="campaign manifest is invalid",
    ):
        archive_stage052_inventory_to_registry(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive,
            registry_path=registry,
            storage_root_locator=locator,
        )

    assert run.is_dir()
    assert not (archive / "stage05.2/history" / run.name).exists()


def test_interrupted_move_retains_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    run = _run(
        source,
        "stage05.2_job_parallel_attempt18",
        status="complete",
        completeness="complete",
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)
    real_replace = os.replace

    def interrupted_replace(source_path: Path | str, destination_path: Path | str) -> None:
        if Path(source_path) == run:
            raise OSError("simulated move interruption")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(retention.os, "replace", interrupted_replace)

    with pytest.raises(OSError, match="simulated move interruption"):
        archive_stage052_inventory(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive,
        )

    assert run.is_dir()
    assert not (archive / "stage05.2/history" / run.name).exists()


def test_cross_volume_archive_copies_verifies_then_removes_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    run = _run(
        source,
        "stage05.2_job_parallel_attempt18",
        status="complete",
        completeness="complete",
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)
    monkeypatch.setattr(retention, "_same_volume", lambda _source, _target: False)

    records = archive_stage052_inventory(
        inventory_path,
        inventory_sha256=inventory_sha256,
        archive_root=archive,
    )

    assert not run.exists()
    assert (archive / "stage05.2/history" / run.name / "events.parquet").is_file()
    assert records[0].disposition == "archived"


def test_cross_volume_copy_interruption_is_safely_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    run = _run(
        source,
        "stage05.2_job_parallel_attempt18",
        status="complete",
        completeness="complete",
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)
    monkeypatch.setattr(retention, "_same_volume", lambda _source, _target: False)
    real_copytree = retention.shutil.copytree

    def interrupted_copytree(source_path: Path, destination_path: Path, **_: object) -> None:
        destination_path.mkdir(parents=True)
        (destination_path / "partial").write_text("partial", encoding="utf-8")
        raise OSError("simulated copy interruption")

    monkeypatch.setattr(retention.shutil, "copytree", interrupted_copytree)
    with pytest.raises(OSError, match="simulated copy interruption"):
        archive_stage052_inventory(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive,
        )
    assert run.is_dir()

    monkeypatch.setattr(retention.shutil, "copytree", real_copytree)
    records = archive_stage052_inventory(
        inventory_path,
        inventory_sha256=inventory_sha256,
        archive_root=archive,
    )
    assert records[0].verification_status == "verified"
    assert not run.exists()


def test_cross_volume_late_source_write_is_not_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    run = _run(
        source,
        "stage05.2_job_parallel_attempt18",
        status="complete",
        completeness="complete",
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)
    monkeypatch.setattr(retention, "_same_volume", lambda _source, _target: False)
    real_copytree = retention.shutil.copytree

    def copy_then_write(
        source_path: Path,
        destination_path: Path,
        *args: object,
        **kwargs: object,
    ) -> Path:
        copied = real_copytree(source_path, destination_path, *args, **kwargs)
        if source_path == run:
            (source_path / "late-write").write_text(
                "writer was still active", encoding="utf-8"
            )
        return copied

    monkeypatch.setattr(retention.shutil, "copytree", copy_then_write)

    with pytest.raises(RetentionIntegrityError, match="source changed during"):
        archive_stage052_inventory(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive,
        )

    assert (run / "late-write").is_file()
    assert (archive / "stage05.2/history" / run.name).is_dir()


def test_cross_volume_cleanup_interruption_uses_isolated_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    run = _run(
        source,
        "stage05.2_job_parallel_attempt18",
        status="complete",
        completeness="complete",
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)
    monkeypatch.setattr(retention, "_same_volume", lambda _source, _target: False)
    real_rmtree = retention.shutil.rmtree

    def interrupted_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if ".retention-cleanup." in path.name:
            raise OSError("simulated cleanup interruption")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(retention.shutil, "rmtree", interrupted_cleanup)
    with pytest.raises(OSError, match="simulated cleanup interruption"):
        archive_stage052_inventory(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive,
        )
    assert not run.exists()
    assert any(".retention-cleanup." in path.name for path in source.iterdir())

    monkeypatch.setattr(retention.shutil, "rmtree", real_rmtree)
    records = archive_stage052_inventory(
        inventory_path,
        inventory_sha256=inventory_sha256,
        archive_root=archive,
    )
    assert records[0].verification_status == "verified"
    assert not any(".retention-cleanup." in path.name for path in source.iterdir())


def test_archive_rejects_different_destination_without_removing_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    run = _run(
        source,
        "stage05.2_perf_baseline_attempt04",
        status="READY_FOR_STAGE052_HOT_PATH",
        completeness="complete",
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)
    destination = archive / "stage05.2/history" / run.name
    destination.mkdir(parents=True)
    (destination / "unrelated").write_text("collision", encoding="utf-8")

    with pytest.raises(RetentionIntegrityError, match="archive destination collision"):
        archive_stage052_inventory(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive,
        )

    assert run.is_dir()
    assert (destination / "unrelated").is_file()


def test_signed_inventory_rejects_tampering(tmp_path: Path) -> None:
    source = tmp_path / "results"
    _run(
        source,
        "stage05.2_hot_path_attempt03",
        status="READY_FOR_STAGE052_ARTIFACT_STREAMING",
        completeness="complete",
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)
    inventory_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RetentionIntegrityError, match="inventory SHA-256 mismatch"):
        load_retention_inventory(
            inventory_path,
            expected_sha256=inventory_sha256,
        )


def test_registry_is_lightweight_path_free_and_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    _run(
        source,
        "stage05.2_accelerator_pilot_attempt01",
        status="NOT_READY",
        completeness="complete",
    )
    inventory = audit_stage052_runs(
        source,
        _policy(),
        created_at_utc="2026-07-23T12:00:00Z",
    )
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)
    records = archive_stage052_inventory(
        inventory_path,
        inventory_sha256=inventory_sha256,
        archive_root=archive,
        archived_at_utc="2026-07-23T12:30:00Z",
    )
    registry = tmp_path / "stage05.2_retention_registry.csv"

    write_retention_registry(registry, records)

    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["archive_root_alias"] == "d_archive"
    assert rows[0]["archive_relative_path"].startswith("stage05.2/history/")
    assert rows[0]["archived_at_utc"] == "2026-07-23T12:30:00Z"
    assert str(tmp_path) not in registry.read_text(encoding="utf-8")


def test_registry_alias_resolves_archived_prerequisite_and_review(tmp_path: Path) -> None:
    source = tmp_path / "results"
    archive = tmp_path / "archive"
    run = _run(
        source,
        "stage05.2_native_kernels_attempt07",
        status="accepted",
        completeness="complete",
    )
    review = run / "review"
    review.mkdir()
    (review / "review_manifest.json").write_text(
        json.dumps({"run_label": run.name, "status": "accepted"}),
        encoding="utf-8",
    )
    inventory = audit_stage052_runs(source, _policy())
    inventory_path = tmp_path / "inventory.json"
    inventory_sha256 = write_retention_inventory(inventory_path, inventory)
    records = archive_stage052_inventory(
        inventory_path,
        inventory_sha256=inventory_sha256,
        archive_root=archive,
    )
    registry = tmp_path / "stage05.2_retention_registry.csv"
    write_retention_registry(registry, records)

    resolved = resolve_retained_run(
        run.name,
        registry_path=registry,
        archive_roots={"d_archive": archive},
    )

    assert json.loads(
        (resolved / "control" / f"{run.name}_manifest.json").read_text(encoding="utf-8")
    )["run_label"] == run.name
    assert json.loads(
        (resolved / "review" / "review_manifest.json").read_text(encoding="utf-8")
    )["run_label"] == run.name


def test_registry_merges_new_runs_without_losing_history(tmp_path: Path) -> None:
    registry = tmp_path / "stage05.2_retention_registry.csv"
    first_source = tmp_path / "first-results"
    second_source = tmp_path / "second-results"
    archive = tmp_path / "archive"
    first = _run(
        first_source,
        "stage05.2_hot_path_attempt03",
        status="complete",
        completeness="complete",
    )
    second = _run(
        second_source,
        "stage05.2_native_kernels_attempt07",
        status="NOT_READY",
        completeness="complete",
    )

    archived_records = []
    for source in (first_source, second_source):
        inventory = audit_stage052_runs(source, _policy())
        inventory_path = tmp_path / f"{source.name}.json"
        inventory_sha256 = write_retention_inventory(inventory_path, inventory)
        records = archive_stage052_inventory(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive,
        )
        write_retention_registry(registry, records)
        archived_records.extend(records)

    with registry.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run_label"] for row in rows] == sorted((first.name, second.name))

    conflicting = replace(archived_records[0], tree_sha256="f" * 64)
    with pytest.raises(RetentionIntegrityError, match="registry identity conflict"):
        write_retention_registry(registry, (conflicting,))


def test_concurrent_registry_merges_do_not_lose_rows(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    records = []
    for index, component in enumerate(("hot_path", "native_kernels"), start=1):
        source = tmp_path / f"results-{index}"
        _run(
            source,
            f"stage05.2_{component}_attempt0{index}",
            status="complete",
            completeness="complete",
        )
        inventory = audit_stage052_runs(source, _policy())
        inventory_path = tmp_path / f"inventory-{index}.json"
        inventory_sha256 = write_retention_inventory(inventory_path, inventory)
        records.append(
            archive_stage052_inventory(
                inventory_path,
                inventory_sha256=inventory_sha256,
                archive_root=archive,
            )[0]
        )
    registry = tmp_path / "registry.csv"
    failures: list[BaseException] = []

    def publish(record: retention.RetentionRecord) -> None:
        try:
            write_retention_registry(registry, (record,))
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=publish, args=(record,)) for record in records]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures
    with registry.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2
