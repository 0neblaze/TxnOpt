from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import evrptw.stage052_campaign as stage052_campaign
from evrptw.artifacts import atomic_write_signed_json
from evrptw.stage052_campaign import (
    PILOT_INSTANCE_NAMES,
    PILOT_SEEDS,
    AcceptedGlobalBest,
    AnytimeCheckpoint,
    ArchiveTransferCompletedError,
    ArchiveTransferError,
    BatchArchiver,
    BatchManifest,
    BatchPlan,
    BenchmarkCampaignConfig,
    BenchmarkPreflightObservation,
    CampaignIdentity,
    CampaignManifest,
    CampaignRecoveryController,
    CampaignRecoveryError,
    FileSystemArchiveIO,
    ManifestIntegrityError,
    PilotStorageObservation,
    ShardPlan,
    StorageRoot,
    StorageRootLocator,
    SystemLoadWindow,
    VolumeIdentity,
    authorize_resume,
    directory_byte_count,
    directory_checksum,
    directory_file_count,
    load_campaign_manifest,
)
from evrptw.stage052_evidence import BatchPersistenceEnvelope, PersistenceInterval


def _pilot_observations(
    *, small_bytes: int = 1_000, large_30_second_bytes: int = 2_000
) -> tuple[PilotStorageObservation, ...]:
    return tuple(
        PilotStorageObservation(
            family=family,
            customer_count=customer_count,
            budget_seconds=30,
            compressed_bytes=small_bytes,
        )
        for customer_count in (5, 10, 15)
        for family in ("C", "R", "RC")
    ) + tuple(
        PilotStorageObservation(
            family=family,
            customer_count=100,
            budget_seconds=30,
            compressed_bytes=large_30_second_bytes,
        )
        for family in ("C", "R", "RC")
    )


def _recovery_campaign(
    tmp_path: Path,
) -> tuple[
    Path,
    StorageRootLocator,
    CampaignManifest,
    CampaignIdentity,
    CampaignRecoveryController,
]:
    staging = tmp_path / "wsl-staging"
    archive = tmp_path / "e-archive"
    staging.mkdir()
    archive.mkdir()
    locator = StorageRootLocator(
        {
            "wsl_staging": StorageRoot(
                "wsl_staging", staging, VolumeIdentity("wsl-device", "ext4")
            ),
            "e_archive": StorageRoot(
                "e_archive", archive, VolumeIdentity("e-device", "ntfs")
            ),
        }
    )
    config = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_rerun16",
        staging_root_alias="wsl_staging",
        archive_root_aliases=("e_archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=6,
        native_profile="stage05.2-native-kernels-v1",
    )
    plan = config.build_plan(_pilot_observations())
    capacity = config.plan_archive_roots(
        plan,
        locator,
        free_bytes_by_alias={
            "wsl_staging": 2 * 1024**4,
            "e_archive": 2 * 1024**4,
        },
    )
    campaign = CampaignManifest.planned(
        config=config,
        plan=plan,
        capacity=capacity,
        locator=locator,
        configuration_sha256="1" * 64,
        prerequisite_review_sha256="2" * 64,
    )
    identity = CampaignIdentity(
        run_label=campaign.run_label,
        repository_revision="a" * 40,
        repository_tree="b" * 40,
        wheel_sha256="3" * 64,
        runtime_identity_sha256="4" * 64,
        configuration_sha256="5" * 64,
        prerequisite_review_sha256=campaign.prerequisite_review_sha256,
        producer_resource_contract_sha256="6" * 64,
        campaign_plan_sha256="7" * 64,
        lifecycle_plan_sha256="8" * 64,
        start_permit_sha256="9" * 64,
    )
    run_dir = staging / campaign.run_label
    controller = CampaignRecoveryController(run_dir=run_dir, locator=locator)
    controller.open(
        expected_identity=identity,
        fresh_campaign=campaign,
        resume_permit_path=None,
    )
    atomic_write_signed_json(run_dir / "campaign_manifest.json", campaign.to_dict())
    return run_dir, locator, campaign, identity, controller


def _host_loss_receipt(
    run_dir: Path, identity: CampaignIdentity
) -> Path:
    path, _ = atomic_write_signed_json(
        run_dir / "control" / "recovery" / "host-loss.json",
        {
            "schema_version": "stage05.2-unexpected-host-loss-receipt-v1",
            "run_label": identity.run_label,
            "cause": "unexpected_host_loss",
            "campaign_identity_sha256": identity.identity_sha256,
            "writer_absent": True,
            "service_state": "failed",
        },
    )
    return path


def _archive_first_recovery_batch(
    *,
    run_dir: Path,
    locator: StorageRootLocator,
    campaign: CampaignManifest,
) -> tuple[CampaignManifest, Path, str]:
    planned = campaign.batches[0]
    destination = locator.resolve("e_archive").absolute_path / planned.logical_path
    destination.mkdir(parents=True)
    shard_hashes, shard_bytes, axis_count = _write_synthetic_recovery_shards(
        destination, planned.shard_ids
    )
    verified = planned.mark_verified(
        checksum_sha256=directory_checksum(destination),
        actual_bytes=directory_byte_count(destination),
        row_count=axis_count,
        physical_schema="screening_decisions_v3",
        resource_summary_sha256="a" * 64,
        persistence_attribution_sha256="b" * 64,
        control_persistence_seconds=0.0,
        persistence_ratio=0.0,
        shard_manifest_sha256_by_id=shard_hashes,
        shard_actual_bytes_by_id=shard_bytes,
    )
    archived = verified.mark_archived(
        root_alias="e_archive",
        volume=locator.resolve("e_archive").volume,
        transfer_mode="cross_volume_verified_copy",
        archive_transfer_seconds=0.0,
    )
    verified_manifest_sha = hashlib.sha256(
        (json.dumps(verified.to_dict(), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()
    atomic_write_signed_json(
        destination / "batch_verified_receipt.json",
        {
            "schema_version": "stage05.2-batch-verified-receipt-v1",
            "run_label": campaign.run_label,
            "batch_id": verified.batch_id,
            "verified_manifest_sha256": verified_manifest_sha,
            "base_attribution_sha256": "d" * 64,
            "solver_seconds": 1.0,
            "base_persistence_seconds": 0.0,
            "verified_manifest_write_interval": {
                "label": "verified_batch_manifest_write",
                "started_ns": 1,
                "completed_ns": 2,
                "duration_seconds": 1e-9,
            },
        },
    )
    manifest_path, _ = atomic_write_signed_json(
        destination / "batch_manifest.json", archived.to_dict()
    )
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    envelope = BatchPersistenceEnvelope(
        run_label=campaign.run_label,
        batch_id=archived.batch_id,
        base_attribution_sha256="d" * 64,
        verified_manifest_sha256=verified_manifest_sha,
        archived_manifest_sha256=manifest_sha,
        solver_seconds=1.0,
        base_persistence_seconds=0.0,
        state_intervals=(
            PersistenceInterval(
                label="verified_batch_manifest_write",
                started_ns=1,
                completed_ns=2,
            ),
            PersistenceInterval(
                label="archived_batch_manifest_write",
                started_ns=3,
                completed_ns=4,
            ),
        ),
    )
    envelope_path, _ = atomic_write_signed_json(
        destination / "batch_persistence_envelope.json", envelope.to_dict()
    )
    updated = campaign.with_batch(archived).with_batch_persistence_envelope(
        archived.batch_id,
        hashlib.sha256(envelope_path.read_bytes()).hexdigest(),
    )
    atomic_write_signed_json(run_dir / "campaign_manifest.json", updated.to_dict())
    return updated, manifest_path, manifest_sha


def _write_verified_recovery_batch(
    *,
    run_dir: Path,
    locator: StorageRootLocator,
    campaign: CampaignManifest,
) -> tuple[CampaignManifest, BatchManifest, Path]:
    planned = campaign.batches[0]
    source = locator.resolve("wsl_staging").absolute_path / planned.logical_path
    source.mkdir(parents=True)
    shard_hashes, shard_bytes, axis_count = _write_synthetic_recovery_shards(
        source, planned.shard_ids
    )
    verified = planned.mark_verified(
        checksum_sha256=directory_checksum(source),
        actual_bytes=directory_byte_count(source),
        row_count=axis_count,
        physical_schema="screening_decisions_v3",
        resource_summary_sha256="a" * 64,
        persistence_attribution_sha256="b" * 64,
        control_persistence_seconds=0.0,
        persistence_ratio=0.0,
        shard_manifest_sha256_by_id=shard_hashes,
        shard_actual_bytes_by_id=shard_bytes,
    )
    verified_sha = hashlib.sha256(
        (json.dumps(verified.to_dict(), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()
    atomic_write_signed_json(source / "batch_manifest.json", verified.to_dict())
    atomic_write_signed_json(
        source / "batch_verified_receipt.json",
        {
            "schema_version": "stage05.2-batch-verified-receipt-v1",
            "run_label": campaign.run_label,
            "batch_id": verified.batch_id,
            "verified_manifest_sha256": verified_sha,
            "base_attribution_sha256": "d" * 64,
            "solver_seconds": 1.0,
            "base_persistence_seconds": 0.0,
            "verified_manifest_write_interval": PersistenceInterval(
                label="verified_batch_manifest_write",
                started_ns=1,
                completed_ns=2,
            ).to_dict(),
        },
    )
    updated = campaign.with_batch(verified)
    atomic_write_signed_json(run_dir / "campaign_manifest.json", updated.to_dict())
    return updated, verified, source


def _crash_recovery_worker(
    fault_point: str,
    run_dir: Path,
    staging: Path,
    archive: Path,
) -> None:
    locator = StorageRootLocator(
        {
            "wsl_staging": StorageRoot(
                "wsl_staging", staging, VolumeIdentity("wsl-device", "ext4")
            ),
            "e_archive": StorageRoot(
                "e_archive", archive, VolumeIdentity("e-device", "ntfs")
            ),
        }
    )
    campaign = load_campaign_manifest(run_dir / "campaign_manifest.json")
    controller = CampaignRecoveryController(run_dir=run_dir, locator=locator)
    if fault_point == "during_batch_compute":
        partial = run_dir / "batch0001"
        partial.mkdir()
        (partial / "partial.parquet").write_bytes(b"partial")
    elif fault_point in {
        "verified_before_archive",
        "archive_publish_before_state",
        "envelope_publish_before_state",
    }:
        verified_campaign, verified, source = _write_verified_recovery_batch(
            run_dir=run_dir,
            locator=locator,
            campaign=campaign,
        )
        if fault_point == "archive_publish_before_state":
            destination = (
                locator.resolve("e_archive").absolute_path / verified.logical_path
            )
            shutil.copytree(source, destination)
            archived = verified.mark_archived(
                root_alias="e_archive",
                volume=locator.resolve("e_archive").volume,
                transfer_mode="cross_volume_verified_copy",
                archive_transfer_seconds=0.0,
            )
            atomic_write_signed_json(
                destination / "batch_manifest.json", archived.to_dict()
            )
        elif fault_point == "envelope_publish_before_state":
            controller._finish_verified_batch(verified_campaign, verified)
            atomic_write_signed_json(
                run_dir / "campaign_manifest.json",
                verified_campaign.to_dict(),
            )
    elif fault_point == "final_batch_before_finalize":
        _archive_first_recovery_batch(
            run_dir=run_dir,
            locator=locator,
            campaign=campaign,
        )
    os._exit(86)


def _write_synthetic_recovery_shards(
    batch_dir: Path,
    shard_ids: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, int], int]:
    shard_root = batch_dir / "shards"
    shard_root.mkdir()
    hashes: dict[str, str] = {}
    byte_sizes: dict[str, int] = {}
    axis_total = 0
    for ordinal, shard_id in enumerate(shard_ids, start=1):
        axis_count = 3 if ordinal <= 200 else 2
        checkpoint_count = 12 if ordinal <= 280 else 11
        payload = {
            "shard_id": shard_id,
            "validator_passed": True,
            "objective_rows": [
                [1, float(ordinal), 0.0, 0] for _axis in range(axis_count)
            ],
            "checkpoints": list(range(checkpoint_count)),
        }
        path = shard_root / f"{shard_id}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        hashes[shard_id] = hashlib.sha256(path.read_bytes()).hexdigest()
        byte_sizes[shard_id] = path.stat().st_size
        axis_total += axis_count
    return hashes, byte_sizes, axis_total


def _synthetic_recovery_review(
    campaign: CampaignManifest,
    locator: StorageRootLocator,
) -> dict[str, object]:
    """Replay synthetic raw shards like the recovery gate's scientific core."""

    shard_ids: list[str] = []
    objective_rows: list[list[float | int]] = []
    checkpoint_count = 0
    validator_passed = True
    for batch in campaign.batches:
        assert batch.status == "archived"
        batch_dir = locator.resolve(batch.root_alias).absolute_path / batch.logical_path
        for shard_id in batch.shard_ids:
            path = batch_dir / "shards" / f"{shard_id}.json"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == (
                batch.shard_manifest_sha256_by_id or {}
            )[shard_id]
            assert path.stat().st_size == (batch.shard_actual_bytes_by_id or {})[
                shard_id
            ]
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["shard_id"] == shard_id
            shard_ids.append(shard_id)
            objective_rows.extend(payload["objective_rows"])
            checkpoint_count += len(payload["checkpoints"])
            validator_passed = validator_passed and payload["validator_passed"]
    return {
        "run_label": campaign.run_label,
        "scope": campaign.scope,
        "shard_ids": shard_ids,
        "shard_count": len(shard_ids),
        "axis_count": len(objective_rows),
        "checkpoint_count": checkpoint_count,
        "validator_passed": validator_passed,
        "objective_rows": objective_rows,
        "batches": [
            {
                "batch_id": batch.batch_id,
                "status": batch.status,
                "shard_ids": list(batch.shard_ids),
                "checksum_sha256": batch.checksum_sha256,
                "actual_bytes": batch.actual_bytes,
                "row_count": batch.row_count,
                "physical_schema": batch.physical_schema,
                "shard_manifest_sha256_by_id": dict(
                    batch.shard_manifest_sha256_by_id or {}
                ),
                "shard_actual_bytes_by_id": dict(
                    batch.shard_actual_bytes_by_id or {}
                ),
            }
            for batch in campaign.batches
        ],
    }


def test_recovery_fresh_open_rejects_an_existing_output_without_permit(
    tmp_path: Path,
) -> None:
    _run_dir, _locator, campaign, identity, controller = _recovery_campaign(tmp_path)

    with pytest.raises(CampaignRecoveryError, match="signed resume permit"):
        controller.open(
            expected_identity=identity,
            fresh_campaign=campaign,
            resume_permit_path=None,
        )


def test_recovery_discards_an_interrupted_batch_and_consumes_permit_once(
    tmp_path: Path,
) -> None:
    run_dir, locator, campaign, identity, controller = _recovery_campaign(tmp_path)
    partial = run_dir / "batch0001"
    (partial / "shard0001").mkdir(parents=True)
    (partial / "shard0001" / "partial.parquet").write_bytes(b"not-complete")
    receipt = _host_loss_receipt(run_dir, identity)
    permit, permit_path = authorize_resume(
        run_dir=run_dir,
        identity=identity,
        locator=locator,
        cause="unexpected_host_loss",
        stop_evidence_path=receipt,
        lifecycle_state="RUNNING",
        writer_active=False,
    )

    session = controller.open(
        expected_identity=identity,
        fresh_campaign=campaign,
        resume_permit_path=permit_path,
    )

    assert session.resumed is True
    assert session.epoch == 2
    assert session.completed_batch_ids == ()
    assert session.pending_batch_ids == tuple(
        batch.batch_id for batch in campaign.batches
    )
    assert not partial.exists()
    capsule = (
        locator.resolve("e_archive").absolute_path
        / ".campaign-recovery"
        / campaign.run_label
        / "epoch0002"
        / "batch0001"
    )
    assert (capsule / "interrupted_inventory.json").is_file()
    assert (capsule / "deletion_receipt.json").is_file()
    assert (
        run_dir / "control" / "recovery" / "consumed" / f"{permit.nonce}.json"
    ).is_file()
    with pytest.raises(CampaignRecoveryError):
        controller.open(
            expected_identity=identity,
            fresh_campaign=campaign,
            resume_permit_path=permit_path,
        )


def test_interrupted_batch_compaction_retries_after_inventory_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, locator, campaign, identity, controller = _recovery_campaign(tmp_path)
    partial = run_dir / "batch0001"
    partial.mkdir()
    (partial / "partial.bin").write_bytes(b"partial")
    permit, _permit_path = authorize_resume(
        run_dir=run_dir,
        identity=identity,
        locator=locator,
        cause="unexpected_host_loss",
        stop_evidence_path=_host_loss_receipt(run_dir, identity),
        lifecycle_state="RUNNING",
        writer_active=False,
    )
    real_rmtree = stage052_campaign.shutil.rmtree
    failed = False

    def fail_source_delete(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal failed
        if Path(path) == partial and not failed:
            failed = True
            raise OSError("injected after inventory")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(stage052_campaign.shutil, "rmtree", fail_source_delete)
    with pytest.raises(OSError, match="injected after inventory"):
        controller._compact_interrupted_batch(
            path=partial, batch=campaign.batches[0], permit=permit
        )
    monkeypatch.setattr(stage052_campaign.shutil, "rmtree", real_rmtree)
    controller._compact_interrupted_batch(
        path=partial, batch=campaign.batches[0], permit=permit
    )
    capsule = (
        locator.resolve("e_archive").absolute_path
        / ".campaign-recovery"
        / campaign.run_label
        / "epoch0002"
        / "batch0001"
    )
    assert not partial.exists()
    assert (capsule / "deletion_receipt.json").is_file()


def test_interrupted_batch_compaction_retries_after_delete_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, locator, campaign, identity, controller = _recovery_campaign(tmp_path)
    partial = run_dir / "batch0001"
    partial.mkdir()
    (partial / "partial.bin").write_bytes(b"partial")
    permit, _permit_path = authorize_resume(
        run_dir=run_dir,
        identity=identity,
        locator=locator,
        cause="unexpected_host_loss",
        stop_evidence_path=_host_loss_receipt(run_dir, identity),
        lifecycle_state="RUNNING",
        writer_active=False,
    )
    real_write = stage052_campaign.atomic_write_signed_json

    def fail_receipt(path: Path, payload: object) -> tuple[Path, Path]:
        if path.name == "deletion_receipt.json":
            raise OSError("injected before receipt")
        return real_write(path, payload)

    monkeypatch.setattr(stage052_campaign, "atomic_write_signed_json", fail_receipt)
    with pytest.raises(OSError, match="injected before receipt"):
        controller._compact_interrupted_batch(
            path=partial, batch=campaign.batches[0], permit=permit
        )
    monkeypatch.setattr(stage052_campaign, "atomic_write_signed_json", real_write)
    controller._compact_interrupted_batch(
        path=partial, batch=campaign.batches[0], permit=permit
    )
    assert not partial.exists()


def test_verified_recovery_rejects_divergent_archived_destination(
    tmp_path: Path,
) -> None:
    run_dir, locator, campaign, _identity, controller = _recovery_campaign(tmp_path)
    archived_campaign, _manifest_path, _manifest_sha = _archive_first_recovery_batch(
        run_dir=run_dir,
        locator=locator,
        campaign=campaign,
    )
    archived = archived_campaign.batches[0]
    verified = replace(
        archived,
        status="verified",
        root_alias="wsl_staging",
        volume=locator.resolve("wsl_staging").volume,
        transfer_mode=None,
        archive_transfer_seconds=None,
    )
    destination = locator.resolve("e_archive").absolute_path / archived.logical_path
    (destination / "payload.bin").write_bytes(b"diverged")

    with pytest.raises(CampaignRecoveryError, match="payload differs"):
        controller._finish_verified_batch(campaign, verified)


@pytest.mark.parametrize(
    ("fault_point", "expected_completed"),
    (
        ("before_batch_create", False),
        ("during_batch_compute", False),
        ("verified_before_archive", True),
        ("archive_publish_before_state", True),
        ("envelope_publish_before_state", True),
        ("final_batch_before_finalize", True),
    ),
)
def test_process_fault_injection_recovers_six_batch_boundaries(
    tmp_path: Path,
    fault_point: str,
    expected_completed: bool,
) -> None:
    run_dir, locator, campaign, identity, controller = _recovery_campaign(tmp_path)

    process = multiprocessing.get_context("spawn").Process(
        target=_crash_recovery_worker,
        args=(
            fault_point,
            run_dir,
            locator.resolve("wsl_staging").absolute_path,
            locator.resolve("e_archive").absolute_path,
        ),
    )
    process.start()
    process.join(10)
    assert process.exitcode == 86
    permit, permit_path = authorize_resume(
        run_dir=run_dir,
        identity=identity,
        locator=locator,
        cause="unexpected_host_loss",
        stop_evidence_path=_host_loss_receipt(run_dir, identity),
        lifecycle_state="RUNNING",
        writer_active=False,
    )
    session = controller.open(
        expected_identity=identity,
        fresh_campaign=campaign,
        resume_permit_path=permit_path,
    )
    semantic_state = (
        session.identity.identity_sha256,
        session.epoch,
        session.completed_batch_ids,
        session.pending_batch_ids,
    )
    expected_completed_ids = ("batch0001",) if expected_completed else ()
    expected_pending_ids = (
        ()
        if expected_completed
        else tuple(batch.batch_id for batch in campaign.batches)
    )
    assert semantic_state == (
        identity.identity_sha256,
        2,
        expected_completed_ids,
        expected_pending_ids,
    )
    assert (
        run_dir / "control" / "recovery" / "consumed" / f"{permit.nonce}.json"
    ).is_file()
    recovered_final = session.campaign
    if not expected_completed:
        recovered_final, _manifest_path, _manifest_sha = (
            _archive_first_recovery_batch(
                run_dir=run_dir,
                locator=locator,
                campaign=recovered_final,
            )
        )
    control_root = tmp_path / "uninterrupted-control"
    control_root.mkdir()
    control_run, control_locator, control_campaign, _identity, _controller = (
        _recovery_campaign(control_root)
    )
    control_final, _manifest_path, _manifest_sha = _archive_first_recovery_batch(
        run_dir=control_run,
        locator=control_locator,
        campaign=control_campaign,
    )
    recovered_review = _synthetic_recovery_review(recovered_final, locator)
    control_review = _synthetic_recovery_review(control_final, control_locator)
    assert recovered_review == control_review
    assert recovered_review["shard_count"] == 920
    assert recovered_review["axis_count"] == 2_040
    assert recovered_review["checkpoint_count"] == 10_400


def test_recovery_revalidates_completed_batch_without_changing_its_hash(
    tmp_path: Path,
) -> None:
    run_dir, locator, campaign, identity, controller = _recovery_campaign(tmp_path)
    campaign, manifest_path, before_sha = _archive_first_recovery_batch(
        run_dir=run_dir,
        locator=locator,
        campaign=campaign,
    )
    receipt = _host_loss_receipt(run_dir, identity)
    _permit, permit_path = authorize_resume(
        run_dir=run_dir,
        identity=identity,
        locator=locator,
        cause="unexpected_host_loss",
        stop_evidence_path=receipt,
        lifecycle_state="RUNNING",
        writer_active=False,
    )

    session = controller.open(
        expected_identity=identity,
        fresh_campaign=campaign,
        resume_permit_path=permit_path,
    )

    assert session.completed_batch_ids == ("batch0001",)
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == before_sha


def test_recovery_completes_after_permit_consumption_precedes_epoch_commit(
    tmp_path: Path,
) -> None:
    run_dir, locator, campaign, identity, controller = _recovery_campaign(tmp_path)
    receipt = _host_loss_receipt(run_dir, identity)
    permit, permit_path = authorize_resume(
        run_dir=run_dir,
        identity=identity,
        locator=locator,
        cause="unexpected_host_loss",
        stop_evidence_path=receipt,
        lifecycle_state="RUNNING",
        writer_active=False,
    )
    atomic_write_signed_json(
        run_dir / "control" / "recovery" / "consumed" / f"{permit.nonce}.json",
        {
            "schema_version": "stage05.2-resume-permit-consumption-v1",
            "run_label": identity.run_label,
            "epoch": permit.epoch,
            "resume_permit_sha256": hashlib.sha256(
                permit_path.read_bytes()
            ).hexdigest(),
            "nonce": permit.nonce,
            "status": "consumed",
        },
    )

    session = controller.open(
        expected_identity=identity,
        fresh_campaign=campaign,
        resume_permit_path=permit_path,
    )

    assert session.epoch == 2
    assert session.epoch_manifest_path.is_file()


def test_operator_stop_requires_a_separately_signed_pre_stop_intent(
    tmp_path: Path,
) -> None:
    run_dir, locator, _campaign, identity, _controller = _recovery_campaign(tmp_path)
    receipt, _ = atomic_write_signed_json(
        run_dir / "control" / "recovery" / "operator-stop.json",
        {
            "schema_version": "stage05.2-operator-stop-receipt-v1",
            "run_label": identity.run_label,
            "cause": "operator_stop",
            "campaign_identity_sha256": identity.identity_sha256,
            "writer_absent": True,
            "service_state": "inactive",
            "stop_intent_relative_path": "control/recovery/stop-intent.json",
            "stop_intent_sha256": "f" * 64,
        },
    )

    with pytest.raises(CampaignRecoveryError, match="stop intent"):
        authorize_resume(
            run_dir=run_dir,
            identity=identity,
            locator=locator,
            cause="operator_stop",
            stop_evidence_path=receipt,
            lifecycle_state="RUNNING",
            writer_active=False,
        )


@pytest.mark.parametrize(
    ("lifecycle_state", "writer_active", "match"),
    (("SEALED", False, "RUNNING"), ("RUNNING", True, "active")),
)
def test_recovery_rejects_terminal_lifecycle_or_active_writer(
    tmp_path: Path,
    lifecycle_state: str,
    writer_active: bool,
    match: str,
) -> None:
    run_dir, locator, _campaign, identity, _controller = _recovery_campaign(tmp_path)
    receipt = _host_loss_receipt(run_dir, identity)

    with pytest.raises(CampaignRecoveryError, match=match):
        authorize_resume(
            run_dir=run_dir,
            identity=identity,
            locator=locator,
            cause="unexpected_host_loss",
            stop_evidence_path=receipt,
            lifecycle_state=lifecycle_state,
            writer_active=writer_active,
        )


def test_recovery_rejects_identity_drift_and_unknown_reason(
    tmp_path: Path,
) -> None:
    run_dir, locator, _campaign, identity, _controller = _recovery_campaign(tmp_path)
    receipt = _host_loss_receipt(run_dir, identity)

    with pytest.raises(CampaignRecoveryError, match="cause"):
        authorize_resume(
            run_dir=run_dir,
            identity=identity,
            locator=locator,
            cause="resource_gate",
            stop_evidence_path=receipt,
            lifecycle_state="RUNNING",
            writer_active=False,
        )
    with pytest.raises(CampaignRecoveryError, match="identity"):
        authorize_resume(
            run_dir=run_dir,
            identity=replace(identity, wheel_sha256="0" * 64),
            locator=locator,
            cause="unexpected_host_loss",
            stop_evidence_path=receipt,
            lifecycle_state="RUNNING",
            writer_active=False,
        )


def test_recovery_rejects_a_tampered_permit_sidecar(
    tmp_path: Path,
) -> None:
    run_dir, locator, campaign, identity, controller = _recovery_campaign(tmp_path)
    receipt = _host_loss_receipt(run_dir, identity)
    _permit, permit_path = authorize_resume(
        run_dir=run_dir,
        identity=identity,
        locator=locator,
        cause="unexpected_host_loss",
        stop_evidence_path=receipt,
        lifecycle_state="RUNNING",
        writer_active=False,
    )
    permit_path.write_bytes(permit_path.read_bytes() + b" ")

    with pytest.raises(CampaignRecoveryError, match="control evidence"):
        controller.open(
            expected_identity=identity,
            fresh_campaign=campaign,
            resume_permit_path=permit_path,
        )


def test_recovery_rejects_a_cross_label_output_directory(tmp_path: Path) -> None:
    _run_dir, locator, campaign, identity, _controller = _recovery_campaign(tmp_path)
    wrong = CampaignRecoveryController(
        run_dir=tmp_path / "stage05.2_benchmark_rerun17",
        locator=locator,
    )

    with pytest.raises(CampaignRecoveryError, match="run label"):
        wrong.open(
            expected_identity=identity,
            fresh_campaign=campaign,
            resume_permit_path=None,
        )


def test_campaign_objective_keys_require_shared_canonical_precision() -> None:
    with pytest.raises(ValueError, match="canonical objective precision"):
        AcceptedGlobalBest(
            completed_at_seconds=1.0,
            iteration=1,
            objective_key=(2, 100.0000000004, 1.0, 1),
        )


def test_formal_campaign_plan_has_exact_scope_order_and_pilot_estimates() -> None:
    config = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("transfer_archive", "internal_archive"),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=4,
        native_profile="stage05.2-native-kernels-v1",
    )

    plan = config.build_plan(_pilot_observations())

    assert len(config.instances) == 92
    assert config.seeds == tuple(range(2014, 2024))
    assert len(plan.shards) == 920
    assert plan.axis_count == 2_040
    assert plan.declared_solver_seconds == 229_200
    assert plan.checkpoint_count == 10_400
    assert tuple(
        (shard.customer_count, shard.instance, shard.seed) for shard in plan.shards
    ) == tuple(sorted((shard.customer_count, shard.instance, shard.seed) for shard in plan.shards))
    assert {shard.estimated_bytes for shard in plan.shards if shard.customer_count < 100} == {1_500}
    assert {shard.estimated_bytes for shard in plan.shards if shard.customer_count == 100} == {
        39_000
    }
    assert {shard.max_iterations for shard in plan.shards if shard.customer_count < 100} == {1_000}
    assert {shard.max_iterations for shard in plan.shards if shard.customer_count == 100} == {None}
    assert config.to_dict()["storage_policy_version"] == "artifact-storage-v2"
    assert config.to_dict()["screening_schema_version"] == "screening_decisions_v3"
    assert config.to_dict()["selected_backend"] == "native_cpu"
    assert config.to_dict()["selected_exact_backend"] == "cpu_batch"


def test_pilot_campaign_uses_exact_stage0_scope_and_shared_manifest_contract(
    tmp_path: Path,
) -> None:
    locator = _root_locator(tmp_path)
    config = BenchmarkCampaignConfig.pilot(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("internal_archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=4,
        native_profile="stage05.2-native-kernels-v1",
    )

    plan = config.build_plan()

    assert config.scope == "pilot"
    assert config.seeds == PILOT_SEEDS
    assert tuple(item.instance for item in config.instances) == PILOT_INSTANCE_NAMES
    assert len(plan.shards) == 36
    assert plan.axis_count == 36
    assert plan.declared_solver_seconds == 1_080
    assert plan.checkpoint_count == 144
    assert plan.scope == "pilot"
    assert len(plan.batches) == 3
    assert all(shard.budgets_seconds == (30,) for shard in plan.shards)
    assert all(shard.scope == "pilot" for shard in plan.shards)
    assert all(shard.estimated_bytes == 2 * 1024**3 for shard in plan.shards)
    expected_customer_counts = {item.instance: item.customer_count for item in config.instances}
    assert tuple(
        (shard.customer_count, shard.instance, shard.seed) for shard in plan.shards
    ) == tuple(
        sorted(
            (expected_customer_counts[instance], instance, seed)
            for instance in PILOT_INSTANCE_NAMES
            for seed in PILOT_SEEDS
        )
    )

    capacity = config.plan_archive_roots(
        plan,
        locator,
        free_bytes_by_alias={
            "transfer_staging": 82 * 1024**3,
            "internal_archive": 200 * 1024**3 + plan.estimated_bytes,
        },
    )
    campaign = CampaignManifest.planned(
        config=config,
        plan=plan,
        capacity=capacity,
        locator=locator,
        configuration_sha256="a" * 64,
        prerequisite_review_sha256="b" * 64,
    )

    assert campaign.scope == "pilot"
    assert campaign.to_dict()["scope"] == "pilot"
    assert CampaignManifest.from_dict(campaign.to_dict()) == campaign

    with pytest.raises(ValueError, match="formal campaign instance scope"):
        replace(config, scope="formal")


def test_campaign_contract_can_bind_an_independently_promoted_cuda_backend() -> None:
    config = BenchmarkCampaignConfig.pilot(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("internal_archive",),
        selected_backend="cuda",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v1",
    )

    assert config.selected_backend == "cuda"
    assert config.to_dict()["selected_backend"] == "cuda"


def test_formal_campaign_uses_sequential_next_fit_without_splitting_shards() -> None:
    config = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("internal_archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v1",
    )

    plan = config.build_plan(
        _pilot_observations(
            small_bytes=1_000_000_000,
            large_30_second_bytes=100_000_000,
        )
    )

    assert tuple(batch.batch_id for batch in plan.batches) == tuple(
        f"batch{ordinal:04d}" for ordinal in range(1, len(plan.batches) + 1)
    )
    assert tuple(shard.shard_id for batch in plan.batches for shard in batch.shards) == tuple(
        shard.shard_id for shard in plan.shards
    )
    assert len(plan.batches) > 1
    assert all(batch.estimated_bytes <= 24 * 1024**3 for batch in plan.batches)
    for batch, following in zip(plan.batches, plan.batches[1:], strict=False):
        assert batch.estimated_bytes + following.shards[0].estimated_bytes > 24 * 1024**3


def test_formal_campaign_storage_limits_cannot_be_weakened() -> None:
    with pytest.raises(ValueError, match="storage byte limits are fixed"):
        BenchmarkCampaignConfig.formal(
            run_label="stage05.2_benchmark_attempt02",
            staging_root_alias="transfer_staging",
            archive_root_aliases=("internal_archive",),
            selected_backend="native_cpu",
            selected_exact_backend="cpu_batch",
            selected_workers=2,
            native_profile="stage05.2-native-kernels-v1",
            batch_target_bytes=10_000,
        )


def test_formal_campaign_rejects_an_indivisible_shard_over_two_gib() -> None:
    config = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("internal_archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v1",
    )
    observations = list(_pilot_observations())
    observations[-1] = PilotStorageObservation(
        family="RC",
        customer_count=100,
        budget_seconds=30,
        compressed_bytes=120_000_000,
    )

    with pytest.raises(ValueError, match="shard estimate exceeds 2 GiB"):
        config.build_plan(tuple(observations))


def test_shard_plan_encodes_small_iteration_cap_and_large_wall_clock_only() -> None:
    with pytest.raises(ValueError, match="small shard requires 1000 iterations"):
        ShardPlan(
            "shard0001",
            "c101C5",
            2014,
            5,
            "C",
            (30,),
            (1, 5, 10, 30, 60, 120, 300),
            1,
            None,
        )
    with pytest.raises(ValueError, match="large shard must be wall-clock only"):
        ShardPlan(
            "shard0001",
            "c101_21",
            2014,
            100,
            "C",
            (30, 60, 300),
            (1, 5, 10, 30, 60, 120, 300),
            1,
            1_000,
        )


def _root_locator(tmp_path: Path) -> StorageRootLocator:
    local = tmp_path / "stage052_storage_roots.local.toml"
    local.write_text(
        """
[roots.transfer_staging]
absolute_path = "/Volumes/TRANSFER/project/results"
device_uuid = "transfer-device"
filesystem = "exfat"

[roots.transfer_archive]
absolute_path = "/Volumes/TRANSFER/project/archive"
device_uuid = "transfer-device"
filesystem = "exfat"

[roots.internal_archive]
absolute_path = "/Users/example/project-results"
device_uuid = "internal-device"
filesystem = "apfs"
""".strip(),
        encoding="utf-8",
    )
    return StorageRootLocator.from_toml(local)


def test_storage_root_locator_keeps_absolute_paths_out_of_tracked_payload(
    tmp_path: Path,
) -> None:
    locator = _root_locator(tmp_path)

    assert locator.resolve("transfer_staging").absolute_path == Path(
        "/Volumes/TRANSFER/project/results"
    )
    tracked = locator.tracked_payload()
    encoded = json.dumps(tracked)
    assert "/Volumes/TRANSFER" not in encoded
    assert "/Users/example" not in encoded
    assert tracked["roots"]["internal_archive"] == {
        "device_uuid": "internal-device",
        "filesystem": "apfs",
    }


def test_storage_root_locator_refreshes_operational_volume_telemetry(
    tmp_path: Path,
) -> None:
    locator = _root_locator(tmp_path)

    refreshed = locator.with_observed_volumes(
        lambda _path: VolumeIdentity("observed-device", "xfs"),
        ("internal_archive",),
    )

    assert refreshed.resolve("internal_archive").volume == VolumeIdentity(
        "observed-device", "xfs"
    )
    assert refreshed.resolve("transfer_staging") == locator.resolve(
        "transfer_staging"
    )

    observed = {
        "/Volumes/TRANSFER/project/results": VolumeIdentity("transfer-device", "exfat"),
        "/Volumes/TRANSFER/project/archive": VolumeIdentity("transfer-device", "exfat"),
        "/Users/example/project-results": VolumeIdentity("internal-device", "apfs"),
    }
    locator.verify_all(lambda path: observed[str(path)])
    observed["/Users/example/project-results"] = VolumeIdentity("tampered", "apfs")
    with pytest.raises(RuntimeError, match="volume identity mismatch"):
        locator.verify_all(lambda path: observed[str(path)])


def test_campaign_capacity_preserves_external_workspace_and_internal_reserve(
    tmp_path: Path,
) -> None:
    locator = _root_locator(tmp_path)
    config = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("transfer_archive", "internal_archive"),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v1",
    )
    plan = config.build_plan(_pilot_observations())
    external_floor = 82 * 1024**3
    internal_free = plan.estimated_bytes

    capacity = config.plan_archive_roots(
        plan,
        locator,
        free_bytes_by_alias={
            "transfer_staging": external_floor,
            "transfer_archive": external_floor,
            "internal_archive": internal_free,
        },
    )

    assert len(capacity.assignments) == len(plan.batches)
    assert {assignment.root_alias for assignment in capacity.assignments} == {"internal_archive"}

    with pytest.raises(RuntimeError, match="staging capacity"):
        config.plan_archive_roots(
            plan,
            locator,
            free_bytes_by_alias={
                "transfer_staging": external_floor - 1,
                "transfer_archive": external_floor - 1,
                "internal_archive": internal_free,
            },
        )
    with pytest.raises(RuntimeError, match="campaign archive projection"):
        config.plan_archive_roots(
            plan,
            locator,
            free_bytes_by_alias={
                "transfer_staging": external_floor,
                "transfer_archive": external_floor,
                "internal_archive": internal_free - 1,
            },
        )


def test_campaign_preflight_treats_power_and_load_as_telemetry() -> None:
    config = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("internal_archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v1",
    )
    valid = BenchmarkPreflightObservation(
        power_source="AC Power",
        low_power_mode_enabled=False,
        windows=(
            SystemLoadWindow(100.0, 30.0, 3.5, 0.75),
            SystemLoadWindow(130.0, 30.0, 4.0, 0.99),
        ),
    )

    config.validate_preflight(valid)
    config.validate_preflight(
        BenchmarkPreflightObservation(
            "Battery Power",
            True,
            (valid.windows[0], SystemLoadWindow(130.0, 30.0, 40.0, 8.0)),
        )
    )

    with pytest.raises(RuntimeError, match="consecutive"):
        config.validate_preflight(
            BenchmarkPreflightObservation(
                "AC Power",
                False,
                (valid.windows[0], SystemLoadWindow(131.0, 30.0, 3.0, 0.5)),
            )
        )


def test_anytime_checkpoints_use_last_complete_global_best_at_each_boundary() -> None:
    checkpoints = AnytimeCheckpoint.for_axis(
        instance="c101C5",
        seed=2014,
        axis_budget_seconds=30,
        initial_objective_key=(3, 120.0, 4.0, 2),
        accepted_global_bests=(
            AcceptedGlobalBest(1.0, 3, (2, 110.0, 3.0, 2)),
            AcceptedGlobalBest(5.0001, 8, (2, 100.0, 2.0, 1)),
            AcceptedGlobalBest(11.0, 15, (2, 90.0, 1.0, 1)),
        ),
        max_iterations_completed_at_seconds=12.0,
        final_objective_key=(2, 90.0, 1.0, 1),
    )

    assert tuple(item.checkpoint_seconds for item in checkpoints) == (1, 5, 10, 30)
    assert checkpoints[0].source == "accepted_global_best"
    assert checkpoints[0].objective_key == (2, 110.0, 3.0, 2)
    assert checkpoints[1].source == "accepted_global_best"
    assert checkpoints[1].objective_key == (2, 110.0, 3.0, 2)
    assert checkpoints[2].objective_key == (2, 100.0, 2.0, 1)
    assert checkpoints[3].source == "final_incumbent_carry_forward"
    assert checkpoints[3].objective_key == (2, 90.0, 1.0, 1)


def test_anytime_checkpoints_use_verified_initial_incumbent_before_improvement() -> None:
    checkpoints = AnytimeCheckpoint.for_axis(
        instance="r101_21",
        seed=2023,
        axis_budget_seconds=60,
        initial_objective_key=(20, 2_000.0, 20.0, 10),
        accepted_global_bests=(AcceptedGlobalBest(10.0001, 2, (19, 1_900.0, 19.0, 9)),),
    )

    assert tuple(item.checkpoint_seconds for item in checkpoints) == (1, 5, 10, 30, 60)
    assert checkpoints[2].source == "verified_initial_incumbent"
    assert checkpoints[2].objective_key == (20, 2_000.0, 20.0, 10)
    assert checkpoints[3].source == "accepted_global_best"


def test_anytime_carry_forward_rejects_an_unverified_final_incumbent() -> None:
    with pytest.raises(ValueError, match="final incumbent does not match"):
        AnytimeCheckpoint.for_axis(
            instance="c101C5",
            seed=2014,
            axis_budget_seconds=30,
            initial_objective_key=(3, 120.0, 4.0, 2),
            accepted_global_bests=(),
            max_iterations_completed_at_seconds=12.0,
            final_objective_key=(1, 1.0, 0.0, 0),
        )

    with pytest.raises(ValueError, match="small instances"):
        AnytimeCheckpoint.for_axis(
            instance="r101_21",
            seed=2014,
            axis_budget_seconds=30,
            initial_objective_key=(20, 2_000.0, 20.0, 10),
            accepted_global_bests=(),
            max_iterations_completed_at_seconds=12.0,
            final_objective_key=(20, 2_000.0, 20.0, 10),
        )


def test_anytime_history_rejects_a_non_improving_global_best() -> None:
    with pytest.raises(ValueError, match="strict objective improvement"):
        AnytimeCheckpoint.for_axis(
            instance="c101C5",
            seed=2014,
            axis_budget_seconds=30,
            initial_objective_key=(2, 100.0, 1.0, 1),
            accepted_global_bests=(AcceptedGlobalBest(1.0, 1, (2, 100.0, 1.0, 1)),),
        )


def test_anytime_history_allows_multiple_global_bests_in_one_iteration() -> None:
    checkpoints = AnytimeCheckpoint.for_axis(
        instance="c101C5",
        seed=2014,
        axis_budget_seconds=30,
        initial_objective_key=(3, 120.0, 4.0, 2),
        accepted_global_bests=(
            AcceptedGlobalBest(1.0001, 7, (2, 110.0, 3.0, 2)),
            AcceptedGlobalBest(1.1, 7, (2, 100.0, 2.0, 1)),
        ),
    )

    assert checkpoints[0].objective_key == (3, 120.0, 4.0, 2)
    assert checkpoints[1].objective_key == (2, 100.0, 2.0, 1)

    with pytest.raises(ValueError, match="non-decreasing iteration"):
        AnytimeCheckpoint.for_axis(
            instance="c101C5",
            seed=2014,
            axis_budget_seconds=30,
            initial_objective_key=(3, 120.0, 4.0, 2),
            accepted_global_bests=(
                AcceptedGlobalBest(1.0, 7, (2, 110.0, 3.0, 2)),
                AcceptedGlobalBest(1.1, 6, (2, 100.0, 2.0, 1)),
            ),
        )


def test_campaign_and_batch_manifests_are_path_free_and_completion_gated(
    tmp_path: Path,
) -> None:
    locator = _root_locator(tmp_path)
    config = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("internal_archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=4,
        native_profile="stage05.2-native-kernels-v1",
    )
    plan = config.build_plan(_pilot_observations())
    capacity = config.plan_archive_roots(
        plan,
        locator,
        free_bytes_by_alias={
            "transfer_staging": 82 * 1024**3,
            "internal_archive": 200 * 1024**3 + plan.estimated_bytes,
        },
    )
    campaign = CampaignManifest.planned(
        config=config,
        plan=plan,
        capacity=capacity,
        locator=locator,
        configuration_sha256="a" * 64,
        prerequisite_review_sha256="b" * 64,
    )

    encoded = json.dumps(campaign.to_dict())
    assert "/Volumes/TRANSFER" not in encoded
    assert "/Users/example" not in encoded
    assert campaign.shard_count == 920
    assert campaign.axis_count == 2_040
    assert campaign.checkpoint_count == 10_400
    manifest_path = tmp_path / "campaign_manifest.json"
    manifest_bytes = json.dumps(campaign.to_dict(), sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_path.with_suffix(".sha256").write_text(
        hashlib.sha256(manifest_bytes).hexdigest() + "\n",
        encoding="utf-8",
    )
    assert load_campaign_manifest(manifest_path).to_dict() == campaign.to_dict()
    manifest_path.write_bytes(manifest_bytes + b"\n")
    with pytest.raises(ManifestIntegrityError, match="checksum mismatch"):
        load_campaign_manifest(manifest_path)
    tampered_batch = replace(
        campaign.batches[0],
        shard_ids=("shard9999", *campaign.batches[0].shard_ids[1:]),
    )
    with pytest.raises(ValueError, match="exact shard IDs"):
        replace(campaign, batches=(tampered_batch, *campaign.batches[1:]))
    with pytest.raises(RuntimeError, match="all batches must be archived"):
        campaign.mark_complete()

    batch = campaign.batches[0].mark_verified(
        checksum_sha256="c" * 64,
        actual_bytes=plan.batches[0].estimated_bytes,
        row_count=10_400,
        physical_schema="screening_decisions_v3",
        resource_summary_sha256="d" * 64,
        persistence_attribution_sha256="f" * 64,
        control_persistence_seconds=1.0,
        persistence_ratio=0.1,
        shard_manifest_sha256_by_id={
            shard_id: "e" * 64 for shard_id in campaign.batches[0].shard_ids
        },
        shard_actual_bytes_by_id={shard_id: 1 for shard_id in campaign.batches[0].shard_ids},
    )
    assert batch.to_dict()["root_alias"] == "transfer_staging"
    assert batch.to_dict()["volume_identity"] == {
        "device_uuid": "transfer-device",
        "filesystem": "exfat",
    }
    campaign = campaign.with_batch(batch)
    with pytest.raises(RuntimeError, match="all batches must be archived"):
        campaign.mark_complete()

    archived = batch.mark_archived(
        root_alias="internal_archive",
        volume=VolumeIdentity("internal-device", "apfs"),
        transfer_mode="cross_volume_verified_copy",
        archive_transfer_seconds=1.0,
    )
    completed = (
        campaign.with_batch(archived)
        .with_batch_persistence_envelope(archived.batch_id, "9" * 64)
        .mark_complete()
    )
    assert completed.status == "complete"

    failed = campaign.with_batch(batch.mark_failed("checksum mismatch")).mark_failed(
        "batch0001 failed"
    )
    assert failed.status == "failed"
    assert failed.batches[0].status == "failed"
    assert failed.batches[0].checksum_sha256 == "c" * 64


def test_batch_manifest_rejects_actual_bytes_above_32_gib_hard_cap(
    tmp_path: Path,
) -> None:
    locator = _root_locator(tmp_path)
    plan = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("internal_archive",),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=4,
        native_profile="stage05.2-native-kernels-v1",
    ).build_plan(_pilot_observations())
    batch = BatchManifest.planned(
        run_label=plan.run_label,
        plan=plan.batches[0],
        staging_root=locator.resolve("transfer_staging"),
        archive_root_alias="internal_archive",
    )

    with pytest.raises(ValueError, match="batch actual bytes exceed 32 GiB"):
        batch.mark_verified(
            checksum_sha256="c" * 64,
            actual_bytes=32 * 1024**3 + 1,
            row_count=1,
            physical_schema="screening_decisions_v3",
            resource_summary_sha256="d" * 64,
            persistence_attribution_sha256="f" * 64,
            control_persistence_seconds=1.0,
            persistence_ratio=0.1,
            shard_manifest_sha256_by_id={shard_id: "e" * 64 for shard_id in batch.shard_ids},
            shard_actual_bytes_by_id={shard_id: 1 for shard_id in batch.shard_ids},
        )


def test_batch_manifest_rejects_actual_shard_above_two_gib_hard_cap(
    tmp_path: Path,
) -> None:
    locator = _root_locator(tmp_path)
    shard = ShardPlan(
        shard_id="shard0001",
        instance="c101C5",
        seed=2014,
        customer_count=5,
        family="C",
        budgets_seconds=(30,),
        checkpoint_seconds=(1, 5, 10, 30, 60, 120, 300),
        estimated_bytes=1,
        max_iterations=1_000,
    )
    batch = BatchManifest.planned(
        run_label="stage05.2_benchmark_attempt02",
        plan=BatchPlan("batch0001", (shard,), 1),
        staging_root=locator.resolve("transfer_staging"),
        archive_root_alias="internal_archive",
    )

    with pytest.raises(ValueError, match="shard actual bytes exceed 2 GiB"):
        batch.mark_verified(
            checksum_sha256="c" * 64,
            actual_bytes=2 * 1024**3 + 1,
            row_count=1,
            physical_schema="screening_decisions_v3",
            resource_summary_sha256="d" * 64,
            persistence_attribution_sha256="f" * 64,
            control_persistence_seconds=1.0,
            persistence_ratio=0.1,
            shard_manifest_sha256_by_id={"shard0001": "e" * 64},
            shard_actual_bytes_by_id={"shard0001": 2 * 1024**3 + 1},
        )


@pytest.mark.parametrize("tamper", ("batch", "shard"))
def test_batch_manifest_parser_rejects_resigned_hard_cap_bypass(
    tmp_path: Path,
    tamper: str,
) -> None:
    locator = _root_locator(tmp_path)
    shard = ShardPlan(
        shard_id="shard0001",
        instance="c101C5",
        seed=2014,
        customer_count=5,
        family="C",
        budgets_seconds=(30,),
        checkpoint_seconds=(1, 5, 10, 30, 60, 120, 300),
        estimated_bytes=1,
        max_iterations=1_000,
    )
    planned = BatchManifest.planned(
        run_label="stage05.2_benchmark_attempt02",
        plan=BatchPlan("batch0001", (shard,), 1),
        staging_root=locator.resolve("transfer_staging"),
        archive_root_alias="internal_archive",
    )
    verified = planned.mark_verified(
        checksum_sha256="c" * 64,
        actual_bytes=1024,
        row_count=1,
        physical_schema="screening_decisions_v3",
        resource_summary_sha256="d" * 64,
        persistence_attribution_sha256="f" * 64,
        control_persistence_seconds=1.0,
        persistence_ratio=0.1,
        shard_manifest_sha256_by_id={"shard0001": "e" * 64},
        shard_actual_bytes_by_id={"shard0001": 512},
    )
    payload = verified.to_dict()
    if tamper == "batch":
        payload["actual_bytes"] = 40 * 1024**3
    else:
        payload["actual_bytes"] = 3 * 1024**3
        payload["shard_actual_bytes_by_id"] = {"shard0001": 3 * 1024**3}

    with pytest.raises(ValueError, match="hard cap"):
        BatchManifest.from_dict(payload)


def _verified_archive_batch(
    tmp_path: Path, *, same_volume: bool
) -> tuple[StorageRootLocator, BatchManifest, Path, Path]:
    source_root_path = tmp_path / "staging"
    destination_root_path = tmp_path / "archive"
    source_root_path.mkdir()
    destination_root_path.mkdir()
    source_volume = VolumeIdentity("external-device", "exfat")
    destination_volume = source_volume if same_volume else VolumeIdentity("internal-device", "apfs")
    locator = StorageRootLocator(
        {
            "staging": StorageRoot("staging", source_root_path, source_volume),
            "archive": StorageRoot("archive", destination_root_path, destination_volume),
        }
    )
    shard = ShardPlan(
        shard_id="shard0001",
        instance="c101C5",
        seed=2014,
        customer_count=5,
        family="C",
        budgets_seconds=(30,),
        checkpoint_seconds=(1, 5, 10, 30, 60, 120, 300),
        estimated_bytes=len(b"verified-evidence"),
        max_iterations=1_000,
    )
    batch = BatchManifest.planned(
        run_label="stage05.2_benchmark_attempt02",
        plan=BatchPlan("batch0001", (shard,), len(b"verified-evidence")),
        staging_root=locator.resolve("staging"),
        archive_root_alias="archive",
    )
    source = source_root_path / batch.logical_path
    destination = destination_root_path / batch.logical_path
    source.mkdir(parents=True)
    (source / "payload.bin").write_bytes(b"verified-evidence")
    batch = batch.mark_verified(
        checksum_sha256=directory_checksum(source),
        actual_bytes=len(b"verified-evidence"),
        row_count=1,
        physical_schema="screening_decisions_v3",
        resource_summary_sha256="d" * 64,
        persistence_attribution_sha256="f" * 64,
        control_persistence_seconds=1.0,
        persistence_ratio=0.1,
        shard_manifest_sha256_by_id={"shard0001": "e" * 64},
        shard_actual_bytes_by_id={"shard0001": len(b"verified-evidence")},
    )
    return locator, batch, source, destination


def test_batch_payload_checksum_excludes_only_the_top_level_manifest_envelope(
    tmp_path: Path,
) -> None:
    batch_dir = tmp_path / "batch0001"
    nested_control = batch_dir / "shard0001" / "control"
    nested_control.mkdir(parents=True)
    (batch_dir / "payload.bin").write_bytes(b"payload")
    (nested_control / "shard_manifest.json").write_bytes(b"nested manifest")

    checksum_before = directory_checksum(batch_dir)
    bytes_before = directory_byte_count(batch_dir)
    files_before = directory_file_count(batch_dir)
    (batch_dir / "batch_manifest.json").write_bytes(b"self-referencing envelope")
    (batch_dir / "batch_manifest.sha256").write_text("0" * 64 + "\n", encoding="utf-8")

    assert directory_checksum(batch_dir) == checksum_before
    assert directory_byte_count(batch_dir) == bytes_before
    assert directory_file_count(batch_dir) == files_before == 2
    (nested_control / "shard_manifest.json").write_bytes(b"tampered nested manifest")
    assert directory_checksum(batch_dir) != checksum_before


def test_same_volume_archive_uses_atomic_rename(tmp_path: Path) -> None:
    locator, batch, source, destination = _verified_archive_batch(tmp_path, same_volume=True)
    clock_values = iter((10.0, 12.5))

    archived = BatchArchiver(locator, clock=lambda: next(clock_values)).archive(batch)

    assert archived.status == "archived"
    assert archived.transfer_mode == "same_volume_atomic_rename"
    assert archived.archive_transfer_seconds == 2.5
    assert not source.exists()
    assert destination.is_dir()
    assert directory_checksum(destination) == batch.checksum_sha256


def test_cross_volume_archive_verifies_incoming_before_deleting_source(
    tmp_path: Path,
) -> None:
    locator, batch, source, destination = _verified_archive_batch(tmp_path, same_volume=False)

    archived = BatchArchiver(locator).archive(batch)

    assert archived.status == "archived"
    assert archived.transfer_mode == "cross_volume_verified_copy"
    assert not source.exists()
    assert destination.is_dir()
    assert not destination.with_name(f"{destination.name}.incoming").exists()


@pytest.mark.parametrize("same_volume", (True, False))
def test_archive_reports_final_destination_after_trailing_fsync_failure(
    tmp_path: Path,
    same_volume: bool,
) -> None:
    locator, batch, source, destination = _verified_archive_batch(
        tmp_path,
        same_volume=same_volume,
    )

    class TrailingFsyncFailureIO(FileSystemArchiveIO):
        def fsync_directory(self, path: Path) -> None:
            if not source.exists() and (
                (same_volume and path == destination.parent)
                or (not same_volume and path == source.parent)
            ):
                raise OSError("injected trailing fsync failure")
            super().fsync_directory(path)

    with pytest.raises(ArchiveTransferCompletedError) as caught:
        BatchArchiver(locator, io=TrailingFsyncFailureIO()).archive(batch)

    assert caught.value.batch.status == "archived"
    assert caught.value.destination == destination
    assert not source.exists()
    assert destination.is_dir()
    assert directory_checksum(destination) == batch.checksum_sha256


def test_cross_volume_archive_tampering_retains_source_and_incoming(
    tmp_path: Path,
) -> None:
    locator, batch, source, destination = _verified_archive_batch(tmp_path, same_volume=False)

    class TamperingIO(FileSystemArchiveIO):
        def fsync_tree(self, path: Path) -> None:
            super().fsync_tree(path)
            if path.name.endswith(".incoming"):
                (path / "payload.bin").write_bytes(b"tampered")

    with pytest.raises(ArchiveTransferError, match="incoming checksum mismatch"):
        BatchArchiver(locator, io=TamperingIO()).archive(batch)

    incoming = destination.with_name(f"{destination.name}.incoming")
    assert batch.status == "verified"
    assert source.is_dir()
    assert incoming.is_dir()
    assert not destination.exists()
