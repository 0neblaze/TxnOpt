"""Cryptographically bind an immutable Stage 5.2 archive to a replaced volume."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, cast

from evrptw.stage052_campaign import (
    CampaignManifest,
    StorageRootLocator,
    VolumeIdentity,
    directory_byte_count,
    directory_checksum,
    load_campaign_manifest,
)

STORAGE_MIGRATION_SCHEMA_VERSION: Final = "stage05.2-storage-migration-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DISK_FIELDS: Final = {"BusType", "FriendlyName", "Number", "SerialNumber"}


class StorageMigrationIntegrityError(RuntimeError):
    """A storage migration attestation does not match immutable evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _disk_identity(payload: object, *, field: str) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != _DISK_FIELDS:
        raise StorageMigrationIntegrityError(f"{field} disk identity is invalid")
    if (
        not isinstance(payload.get("BusType"), str)
        or not isinstance(payload.get("FriendlyName"), str)
        or isinstance(payload.get("Number"), bool)
        or not isinstance(payload.get("Number"), int)
        or not isinstance(payload.get("SerialNumber"), str)
    ):
        raise StorageMigrationIntegrityError(f"{field} disk identity types are invalid")
    return dict(payload)


def load_signed_storage_migration(path: Path) -> dict[str, object]:
    """Load an exact-schema migration attestation whose SHA sidecar matches."""

    try:
        payload_bytes = path.read_bytes()
        sidecar = path.with_suffix(f"{path.suffix}.sha256")
        declared = sidecar.read_text(encoding="ascii").strip()
    except OSError as error:
        raise StorageMigrationIntegrityError(
            f"storage migration attestation is unavailable: {path}"
        ) from error
    observed = hashlib.sha256(payload_bytes).hexdigest()
    if declared != observed or _SHA256.fullmatch(declared) is None:
        raise StorageMigrationIntegrityError("storage migration attestation SHA-256 mismatch")
    try:
        payload = json.loads(payload_bytes)
    except json.JSONDecodeError as error:
        raise StorageMigrationIntegrityError(
            "storage migration attestation is not valid JSON"
        ) from error
    expected = {
        "schema_version",
        "migration_id",
        "created_at_utc",
        "reason",
        "run_label",
        "archive_root_alias",
        "source_volume",
        "destination_volume",
        "source_machine_disk",
        "destination_machine_disk",
        "campaign_manifest_sha256",
        "standard_raw_manifest_sha256",
        "archived_batches",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise StorageMigrationIntegrityError(
            "storage migration attestation fields do not match its schema"
        )
    if payload.get("schema_version") != STORAGE_MIGRATION_SCHEMA_VERSION:
        raise StorageMigrationIntegrityError("storage migration schema is unsupported")
    for field in (
        "migration_id",
        "created_at_utc",
        "reason",
        "run_label",
        "archive_root_alias",
    ):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise StorageMigrationIntegrityError(f"storage migration {field} is invalid")
    for field in ("campaign_manifest_sha256", "standard_raw_manifest_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise StorageMigrationIntegrityError(f"storage migration {field} is invalid")
    for field in ("source_volume", "destination_volume"):
        value = payload.get(field)
        if not isinstance(value, Mapping):
            raise StorageMigrationIntegrityError(f"storage migration {field} is invalid")
        try:
            VolumeIdentity.from_dict(value)
        except (TypeError, ValueError) as error:
            raise StorageMigrationIntegrityError(
                f"storage migration {field} is invalid"
            ) from error
    _disk_identity(payload.get("source_machine_disk"), field="source")
    _disk_identity(payload.get("destination_machine_disk"), field="destination")
    batches = payload.get("archived_batches")
    if not isinstance(batches, list) or not batches:
        raise StorageMigrationIntegrityError("storage migration archived_batches is invalid")
    expected_batch_fields = {
        "batch_id",
        "logical_path",
        "directory_checksum_sha256",
        "byte_count",
    }
    for batch in batches:
        if not isinstance(batch, dict) or set(batch) != expected_batch_fields:
            raise StorageMigrationIntegrityError("storage migration batch fields are invalid")
        if (
            not isinstance(batch.get("batch_id"), str)
            or not isinstance(batch.get("logical_path"), str)
            or not isinstance(batch.get("directory_checksum_sha256"), str)
            or _SHA256.fullmatch(str(batch["directory_checksum_sha256"])) is None
            or isinstance(batch.get("byte_count"), bool)
            or not isinstance(batch.get("byte_count"), int)
            or int(batch["byte_count"]) < 0
        ):
            raise StorageMigrationIntegrityError("storage migration batch identity is invalid")
    return cast(dict[str, object], payload)


def verify_campaign_storage_migration(
    path: Path,
    *,
    campaign: CampaignManifest,
    campaign_dir: Path,
    locator: StorageRootLocator,
    volume_probe: Callable[[Path], VolumeIdentity],
) -> dict[str, object]:
    """Verify old-to-new archive identity and every archived batch byte."""

    payload = load_signed_storage_migration(path)
    alias = str(payload["archive_root_alias"])
    if (
        payload["run_label"] != campaign.run_label
        or payload["reason"] != "physical_archive_disk_replacement"
        or alias not in campaign.storage_roots
    ):
        raise StorageMigrationIntegrityError("storage migration campaign identity mismatch")
    source_volume = VolumeIdentity.from_dict(
        cast(Mapping[str, object], payload["source_volume"])
    )
    destination_volume = VolumeIdentity.from_dict(
        cast(Mapping[str, object], payload["destination_volume"])
    )
    configured = locator.resolve(alias)
    live_volume = volume_probe(configured.absolute_path)
    if (
        campaign.storage_roots[alias] != source_volume
        or configured.volume != destination_volume
        or live_volume != destination_volume
        or source_volume == destination_volume
    ):
        raise StorageMigrationIntegrityError("storage migration volume chain mismatch")
    campaign_manifest = campaign_dir / "campaign_manifest.json"
    standard_manifest = (
        campaign_dir / "control" / f"{campaign.run_label}_manifest.json"
    )
    if (
        _sha256(campaign_manifest) != payload["campaign_manifest_sha256"]
        or _sha256(standard_manifest) != payload["standard_raw_manifest_sha256"]
    ):
        raise StorageMigrationIntegrityError("storage migration manifest binding mismatch")
    recorded_batches = payload["archived_batches"]
    assert isinstance(recorded_batches, list)
    by_id = {str(item["batch_id"]): item for item in recorded_batches}
    if len(by_id) != len(recorded_batches) or set(by_id) != {
        batch.batch_id for batch in campaign.batches
    }:
        raise StorageMigrationIntegrityError("storage migration batch set mismatch")
    for batch in campaign.batches:
        item = by_id[batch.batch_id]
        batch_dir = configured.absolute_path.joinpath(*Path(batch.logical_path).parts)
        if (
            item["logical_path"] != batch.logical_path
            or item["directory_checksum_sha256"] != batch.checksum_sha256
            or item["byte_count"] != batch.actual_bytes
            or directory_checksum(batch_dir) != batch.checksum_sha256
            or directory_byte_count(batch_dir) != batch.actual_bytes
        ):
            raise StorageMigrationIntegrityError(
                f"storage migration batch identity mismatch: {batch.batch_id}"
            )
    return payload


def verify_successor_storage_migration_evidence(
    path: Path,
    *,
    evidence_dir: Path,
    locator: StorageRootLocator,
    volume_probe: Callable[[Path], VolumeIdentity],
) -> dict[str, object]:
    """Verify the reviewed campaign that authorizes a successor on new storage."""

    payload = load_signed_storage_migration(path)
    run_label = str(payload["run_label"])
    if evidence_dir.name != run_label:
        raise StorageMigrationIntegrityError(
            "storage migration evidence directory does not match its run label"
        )
    try:
        campaign = load_campaign_manifest(evidence_dir / "campaign_manifest.json")
    except (OSError, TypeError, ValueError) as error:
        raise StorageMigrationIntegrityError(
            "storage migration campaign manifest is unavailable or invalid"
        ) from error
    if (
        campaign.run_label != run_label
        or campaign.scope != "pilot"
        or campaign.status != "complete"
    ):
        raise StorageMigrationIntegrityError(
            "storage migration campaign is not a complete Pilot"
        )
    verified = verify_campaign_storage_migration(
        path,
        campaign=campaign,
        campaign_dir=evidence_dir,
        locator=locator,
        volume_probe=volume_probe,
    )
    try:
        review_path = evidence_dir / "review" / "review_manifest.json"
        execution_path = evidence_dir / "review" / "review_execution.json"
        review = json.loads(review_path.read_text(encoding="utf-8"))
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StorageMigrationIntegrityError(
            "storage migration independent review evidence is unavailable or invalid"
        ) from error
    raw_sha256 = str(payload["standard_raw_manifest_sha256"])
    migration_sha256 = _sha256(path)
    if (
        not isinstance(review, dict)
        or review.get("run_label") != run_label
        or review.get("component") != "benchmark"
        or review.get("scope") != "pilot"
        or review.get("status") != "READY_FOR_STAGE052_FORMAL_BENCHMARK"
        or review.get("raw_manifest_sha256") != raw_sha256
        or review.get("storage_migration_sha256") != migration_sha256
    ):
        raise StorageMigrationIntegrityError(
            "storage migration campaign review does not authorize a successor"
        )
    if (
        not isinstance(execution, dict)
        or execution.get("run_label") != run_label
        or execution.get("finalized") is not True
        or execution.get("status") != "completed"
        or execution.get("exit_code") != 0
        or execution.get("systemd_service_result") != "success"
        or execution.get("raw_manifest_unchanged") is not True
        or execution.get("raw_manifest_sha256_before") != raw_sha256
        or execution.get("raw_manifest_sha256_after") != raw_sha256
        or execution.get("review_manifest_sha256") != _sha256(review_path)
    ):
        raise StorageMigrationIntegrityError(
            "storage migration campaign review receipt is not finalized and successful"
        )
    return verified


def machine_identity_matches_storage_migration(
    frozen: Mapping[str, object],
    live: Mapping[str, object],
    migration: Mapping[str, object],
) -> bool:
    """Allow only the attested physical D-archive disk field to differ."""

    try:
        source_disk = _disk_identity(
            migration.get("source_machine_disk"), field="source"
        )
        destination_disk = _disk_identity(
            migration.get("destination_machine_disk"), field="destination"
        )
    except StorageMigrationIntegrityError:
        return False
    if frozen.get("d_archive_disk") != source_disk or live.get(
        "d_archive_disk"
    ) != destination_disk:
        return False
    frozen_copy = {key: value for key, value in frozen.items() if key != "memory_bytes"}
    live_copy = {key: value for key, value in live.items() if key != "memory_bytes"}
    frozen_copy["d_archive_disk"] = destination_disk
    for identity in (frozen_copy, live_copy):
        windows = identity.get("windows")
        if isinstance(windows, Mapping):
            normalized_windows = dict(windows)
            if normalized_windows.get("Caption") in {
                "Microsoft Windows 11 Pro for Workstations",
                "Microsoft Windows 11 专业工作站版",
            }:
                normalized_windows["Caption"] = "windows_11_pro_for_workstations"
            identity["windows"] = normalized_windows
    return frozen_copy == live_copy
