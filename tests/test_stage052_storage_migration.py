from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evrptw.stage052_storage_migration import (
    STORAGE_MIGRATION_SCHEMA_VERSION,
    StorageMigrationIntegrityError,
    load_signed_storage_migration,
    machine_identity_matches_storage_migration,
)


def _disk(name: str, serial: str) -> dict[str, object]:
    return {
        "BusType": "NVMe",
        "FriendlyName": name,
        "Number": 1,
        "SerialNumber": serial,
    }


def _payload() -> dict[str, object]:
    return {
        "schema_version": STORAGE_MIGRATION_SCHEMA_VERSION,
        "migration_id": "stage052-d-ssd-20260725",
        "created_at_utc": "2026-07-25T05:00:00Z",
        "reason": "physical_archive_disk_replacement",
        "run_label": "stage05.2_benchmark_attempt26",
        "archive_root_alias": "d_archive",
        "source_volume": {"device_uuid": "old-disk", "filesystem": "9p"},
        "destination_volume": {"device_uuid": "new-disk", "filesystem": "9p"},
        "source_machine_disk": _disk("old", "old-serial"),
        "destination_machine_disk": _disk("new", "new-serial"),
        "campaign_manifest_sha256": "a" * 64,
        "standard_raw_manifest_sha256": "b" * 64,
        "archived_batches": [
            {
                "batch_id": "batch0001",
                "logical_path": "stage05.2_benchmark_attempt26/batch0001",
                "directory_checksum_sha256": "c" * 64,
                "byte_count": 123,
            }
        ],
    }


def _write_signed(path: Path, payload: dict[str, object]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    path.with_suffix(f"{path.suffix}.sha256").write_text(
        f"{hashlib.sha256(data).hexdigest()}\n",
        encoding="ascii",
    )


def test_signed_storage_migration_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "migration.json"
    _write_signed(path, _payload())
    assert load_signed_storage_migration(path)["migration_id"] == (
        "stage052-d-ssd-20260725"
    )
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(StorageMigrationIntegrityError, match="SHA-256 mismatch"):
        load_signed_storage_migration(path)


def test_machine_identity_migration_allows_only_attested_archive_disk() -> None:
    migration = _payload()
    frozen = {
        "memory_bytes": 16,
        "cpu": {"Name": "same"},
        "d_archive_disk": migration["source_machine_disk"],
        "windows": {"Caption": "Microsoft Windows 11 专业工作站版"},
    }
    live = {
        "memory_bytes": 15,
        "cpu": {"Name": "same"},
        "d_archive_disk": migration["destination_machine_disk"],
        "windows": {"Caption": "Microsoft Windows 11 Pro for Workstations"},
    }
    assert machine_identity_matches_storage_migration(frozen, live, migration)
    live["cpu"] = {"Name": "different"}
    assert not machine_identity_matches_storage_migration(frozen, live, migration)
