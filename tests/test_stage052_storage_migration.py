from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evrptw.stage052_storage_migration import (
    STORAGE_MIGRATION_SCHEMA_VERSION,
    StorageMigrationIntegrityError,
    load_signed_storage_migration,
    machine_identity_matches_storage_migration,
    verify_successor_storage_migration_evidence,
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


def test_successor_migration_requires_finalized_independent_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "migration.json"
    payload = _payload()
    _write_signed(path, payload)
    evidence_dir = tmp_path / str(payload["run_label"])
    review_dir = evidence_dir / "review"
    review_dir.mkdir(parents=True)
    campaign_payload = {
        "run_label": payload["run_label"],
    }
    (evidence_dir / "campaign_manifest.json").write_text(
        json.dumps(campaign_payload),
        encoding="utf-8",
    )
    review = {
        "run_label": payload["run_label"],
        "component": "benchmark",
        "scope": "pilot",
        "status": "READY_FOR_STAGE052_FORMAL_BENCHMARK",
        "raw_manifest_sha256": payload["standard_raw_manifest_sha256"],
        "storage_migration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    review_path = review_dir / "review_manifest.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    execution = {
        "run_label": payload["run_label"],
        "finalized": True,
        "status": "completed",
        "exit_code": 0,
        "systemd_service_result": "success",
        "raw_manifest_unchanged": True,
        "raw_manifest_sha256_before": payload["standard_raw_manifest_sha256"],
        "raw_manifest_sha256_after": payload["standard_raw_manifest_sha256"],
        "review_manifest_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
    }
    execution_path = review_dir / "review_execution.json"
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    campaign = SimpleNamespace(
        run_label=payload["run_label"],
        scope="pilot",
        status="complete",
    )
    monkeypatch.setattr(
        "evrptw.stage052_storage_migration.load_campaign_manifest",
        lambda _: campaign,
    )
    monkeypatch.setattr(
        "evrptw.stage052_storage_migration.verify_campaign_storage_migration",
        lambda *_args, **_kwargs: payload,
    )

    assert (
        verify_successor_storage_migration_evidence(
            path,
            evidence_dir=evidence_dir,
            locator=object(),  # type: ignore[arg-type]
            volume_probe=lambda _: object(),  # type: ignore[return-value]
        )
        == payload
    )

    execution["finalized"] = False
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    with pytest.raises(
        StorageMigrationIntegrityError,
        match="receipt is not finalized and successful",
    ):
        verify_successor_storage_migration_evidence(
            path,
            evidence_dir=evidence_dir,
            locator=object(),  # type: ignore[arg-type]
            volume_probe=lambda _: object(),  # type: ignore[return-value]
        )
