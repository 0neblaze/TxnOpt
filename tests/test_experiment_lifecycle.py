from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import evrptw.experiment_lifecycle as lifecycle_module
import evrptw.storage_governance as storage_governance
from evrptw.artifacts import atomic_write_signed_json
from evrptw.experiment_lifecycle import (
    ClosePerformance,
    ExperimentCatalog,
    ExperimentLifecycleController,
    ExperimentPlan,
    LifecycleError,
    LifecycleState,
    RetentionClassV3,
    ReviewerStatus,
    _cli,
    _write_signed_json,
    audit_runner_entrypoints,
    load_historical_migration_gate,
    load_lifecycle_migration_ledger,
)
from evrptw.experiment_lifecycle import (
    main as lifecycle_main,
)
from evrptw.experiments.lifecycle_historical_review import (
    _inventory_generation,
    aggregate_historical_gate,
    apply_semantic_adjudication,
    execute_semantic_reviewer,
    review_historical_generation,
)
from evrptw.experiments.stage052_historical_semantic_review import (
    build_no_dependency_proof,
    review_historical_stage052,
)
from evrptw.experiments.stage052_historical_semantic_review import (
    main as historical_semantic_main,
)
from evrptw.stage052_campaign import (
    StorageRoot,
    StorageRootLocator,
    VolumeIdentity,
)
from evrptw.storage_governance import (
    StorageGovernanceError,
    build_cli_terminal_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
HASH = "a" * 64


def _catalog() -> ExperimentCatalog:
    return ExperimentCatalog.from_toml(ROOT / "configs/experiment_catalog.toml")


def _plan(tmp_path: Path, label: str) -> ExperimentPlan:
    return ExperimentPlan(
        run_label=label,
        run_dir=(tmp_path / label).resolve(),
        configuration_sha256=HASH,
        source_sha256="b" * 64,
        lock_sha256="c" * 64,
        environment_sha256="d" * 64,
        prerequisite_sha256_by_contract={},
        planned_archive_bytes=1024,
        max_batch_bytes=1024,
        max_run_bytes=1024,
        max_workspace_bytes=2048,
        workers=1,
        threads=1,
        processes=1,
        fallback_allowed=False,
        io_backend="windows_native",
        io_workers=4,
    )


def _controller(tmp_path: Path) -> ExperimentLifecycleController:
    return ExperimentLifecycleController(
        catalog=_catalog(),
        state_root=(tmp_path / "lifecycle").resolve(),
    )


def _start(
    controller: ExperimentLifecycleController,
    tmp_path: Path,
    label: str,
) -> ExperimentPlan:
    plan = _plan(tmp_path, label)
    controller.plan(plan)
    permit_path = controller.capacity_state_root / "permits" / f"{label}.json"
    _write_signed_json(
        permit_path,
        {
            "run_label": label,
            "status": "reserved",
            "stage_plan_sha256": plan.plan_sha256,
        },
    )
    ledger_path = controller.capacity_state_root / "capacity_ledger.json"
    reservations: dict[str, object] = {}
    if ledger_path.is_file():
        reservations = json.loads(ledger_path.read_text(encoding="utf-8"))[
            "reservations"
        ]
    reservations[label] = {
        "stage_id": "stage00",
        "stage_plan_sha256": plan.plan_sha256,
        "staging_root_alias": "wsl_staging",
        "host_root_alias": "d_host",
        "archive_root_alias": "e_archive",
        "planned_archive_bytes": plan.planned_archive_bytes,
        "max_active_workspace_bytes": plan.max_workspace_bytes,
        "projected_host_growth_bytes": plan.max_workspace_bytes,
        "status": "reserved",
    }
    _write_signed_json(
        ledger_path,
        {
            "schema_version": "experiment-storage-governance-v1",
            "reservations": reservations,
        },
    )
    assert controller.permit(label, storage_permit_path=permit_path).state == (
        LifecycleState.PERMITTED
    )
    assert controller.start(label, runtime_plan=plan).state == LifecycleState.RUNNING
    plan.run_dir.mkdir()
    return plan


def _seal_and_review(
    controller: ExperimentLifecycleController,
    tmp_path: Path,
    label: str,
    *,
    reviewer_status: ReviewerStatus,
    failure_code: str = "",
) -> None:
    artifact = tmp_path / label / "control" / "raw-evidence.txt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"raw-evidence")
    run_dir = tmp_path / label
    artifacts = [
        {
            "relative_path": path.relative_to(run_dir).as_posix(),
            "byte_size": path.stat().st_size,
            "checksum": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(item for item in run_dir.rglob("*") if item.is_file())
    ]
    manifest = tmp_path / label / "control" / "manifest.json"
    _write_signed_json(
        manifest,
        {
            "run_label": label,
            "status": "complete" if not failure_code else "partial",
            "evidence_completeness": "complete" if not failure_code else "partial",
            "artifacts": artifacts,
        },
    )
    controller.seal(label, manifest_path=manifest, failure_code=failure_code)
    review = tmp_path / label / "review" / "review.json"
    failure_identity = (
        {
            "component": "runner",
            "invariant_or_check": "manifest_integrity",
            "location": "control/manifest.json",
        }
        if reviewer_status
        in {
            ReviewerStatus.FAILED_KNOWN,
            ReviewerStatus.FAILED_UNKNOWN,
            ReviewerStatus.INVALID,
        }
        else {}
    )
    _write_signed_json(
        review,
        {
            "run_label": label,
            "status": reviewer_status.value,
            "failure_identity": failure_identity,
            "files": {},
        },
    )
    _review_execution(controller, label, review)
    controller.review(
        label,
        review_manifest_path=review,
    )


def _review_execution(
    controller: ExperimentLifecycleController,
    label: str,
    review_manifest: Path,
) -> None:
    record = next(item for item in controller.records() if item.run_label == label)
    spec = controller.catalog.for_run_label(label)
    execution_path = review_manifest.parent / "review_execution.json"
    _write_signed_json(
        execution_path,
        {
            "run_label": label,
            "status": "completed",
            "finalized": True,
            "exit_code": 0,
            "reviewer_module_name": spec.reviewer_module,
            "reviewer_installed_distribution_digest": "9" * 64,
            "raw_manifest_sha256_before": record.sealed_manifest_sha256,
            "raw_manifest_sha256_after": record.sealed_manifest_sha256,
            "raw_manifest_unchanged": True,
            "review_manifest_sha256": hashlib.sha256(
                review_manifest.read_bytes()
            ).hexdigest(),
        },
    )
    _write_signed_json(
        controller.state_root / "review-executions" / f"{label}.json",
        {
            "schema_version": "experiment-review-execution-binding-v1",
            "run_label": label,
            "review_execution_sha256": hashlib.sha256(
                execution_path.read_bytes()
            ).hexdigest(),
            "review_manifest_sha256": hashlib.sha256(
                review_manifest.read_bytes()
            ).hexdigest(),
            "sealed_manifest_sha256": record.sealed_manifest_sha256,
        },
    )


def _classification_context(
    controller: ExperimentLifecycleController,
    tmp_path: Path,
    label: str,
    *,
    publication_state: str,
) -> Path:
    record = next(item for item in controller.records() if item.run_label == label)
    path = tmp_path / f"{label}-classification.json"
    evidence_path = tmp_path / f"{label}-classification-evidence.json"
    evidence_sha256 = ""
    if publication_state != "none":
        _write_signed_json(
            evidence_path,
            {
                "run_label": label,
                "review_manifest_sha256": record.review_manifest_sha256,
                "publication_state": publication_state,
            },
        )
        evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    _write_signed_json(
        path,
        {
            "schema_version": "experiment-classification-context-v1",
            "run_label": label,
            "review_manifest_sha256": record.review_manifest_sha256,
            "publication_state": publication_state,
            "classification_evidence_relative_path": evidence_path.name,
            "classification_evidence_sha256": evidence_sha256,
            "rebuild_proof_sha256": "",
        },
    )
    return path


def _content_inventory(run_dir: Path, label: str) -> Path:
    files: list[dict[str, object]] = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        stat = path.stat()
        files.append(
            {
                "relative_path": path.relative_to(run_dir).as_posix(),
                "byte_count": stat.st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "modified_time_ns": stat.st_mtime_ns,
            }
        )
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item["relative_path"]).encode())
        digest.update(b"\0")
        digest.update(str(item["byte_count"]).encode())
        digest.update(b"\0")
        digest.update(str(item["sha256"]).encode())
        digest.update(b"\n")
    path = run_dir.parent / f"{label}-inventory.json"
    _write_signed_json(
        path,
        {
            "schema_version": "experiment-content-inventory-v1",
            "run_label": label,
            "source_tree_sha256": digest.hexdigest(),
            "files": files,
            "hashed_bytes_during_write": sum(int(item["byte_count"]) for item in files),
            "scanned_bytes_during_write": 0,
            "io_backend": "windows_native",
            "io_workers": 4,
            "backend_calibrated": True,
            "implicit_fallback": False,
        },
    )
    return path


def _storage_reconciliation(
    controller: ExperimentLifecycleController,
    tmp_path: Path,
    label: str,
) -> Path:
    record = next(item for item in controller.records() if item.run_label == label)
    evidence_sha256 = (
        record.retention_receipt_sha256
        if record.state == LifecycleState.RETAINED
        else record.compaction_receipt_sha256
    )
    permit_path = controller.capacity_state_root / "permits" / f"{label}.json"
    permit_sha256 = hashlib.sha256(permit_path.read_bytes()).hexdigest()
    ledger_path = controller.capacity_state_root / "capacity_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    reservation = ledger["reservations"][label]
    reservation.update(
        status="reconciled",
        outcome="retained",
        evidence_sha256=evidence_sha256,
        storage_permit_sha256=permit_sha256,
    )
    _write_signed_json(ledger_path, ledger)
    path = (
        controller.capacity_state_root
        / "permit_reconciliations"
        / f"{label}.json"
    )
    _write_signed_json(
        path,
        {
            "schema_version": "experiment-capacity-reconciliation-v1",
            "run_label": label,
            "outcome": "retained",
            "evidence_sha256": evidence_sha256,
            "storage_permit_sha256": permit_sha256,
        },
    )
    return path


def _retention_binding(
    controller: ExperimentLifecycleController,
    label: str,
    *,
    run_dir: Path,
    inventory: dict[str, object],
    performance: dict[str, object],
) -> Path:
    generation = 1
    relative_archive = Path(
        "stage00", "runs", label, f"generation-{generation:04d}"
    )
    archive_path = controller.archive_root / relative_archive
    shutil.copytree(run_dir, archive_path)
    files = inventory["files"]
    assert isinstance(files, list)
    source_bytes = sum(int(item["byte_count"]) for item in files)
    replay_path = (
        controller.storage_state_root
        / "retention_replays"
        / f"{label}-generation-{generation:04d}.json"
    )
    _write_signed_json(
        replay_path,
        {
            "schema_version": "experiment-retention-replay-v1",
            "run_label": label,
            "generation": generation,
            "archive_tree_sha256": inventory["source_tree_sha256"],
            "archive_file_count": len(files),
            "archive_byte_count": source_bytes,
            "verifier_identity_sha256": "2" * 64,
            "validator_replay_passed": True,
            "objective_replay_passed": True,
            "raw_review_replay_passed": True,
            "status": "passed",
        },
    )
    registry_path = controller.storage_state_root / "retention_registry_v2.json"
    records: list[dict[str, object]] = []
    if registry_path.is_file():
        records = json.loads(registry_path.read_text(encoding="utf-8"))["records"]
    records.append(
        {
            "run_label": label,
            "generation": generation,
            "archive_relative_path": relative_archive.as_posix(),
            "tree_sha256": inventory["source_tree_sha256"],
            "file_count": len(files),
            "byte_count": source_bytes,
            "verification_status": "verified",
        }
    )
    _write_signed_json(
        registry_path,
        {"schema_version": "experiment-retention-registry-v2", "records": records},
    )
    record = next(item for item in controller.records() if item.run_label == label)
    receipt = (
        controller.storage_state_root
        / "lifecycle_retention_receipts"
        / f"{label}.json"
    )
    _write_signed_json(
        receipt,
        {
            "schema_version": "experiment-lifecycle-retention-binding-v1",
            "run_label": label,
            "generation": generation,
            "verification_status": "verified",
            "retention_class": "current_accepted_full",
            "storage_permit_sha256": record.storage_permit_sha256,
            "archive_relative_path": relative_archive.as_posix(),
            "archive_tree_sha256": inventory["source_tree_sha256"],
            "verifier_identity_sha256": "2" * 64,
            "file_count": len(files),
            "byte_count": source_bytes,
            "validator_replay_passed": True,
            "objective_replay_passed": True,
            "raw_review_replay_passed": True,
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "replay_receipt_relative_path": replay_path.relative_to(
                controller.storage_state_root
            ).as_posix(),
            "replay_receipt_sha256": hashlib.sha256(
                replay_path.read_bytes()
            ).hexdigest(),
            "close_performance": performance,
        },
    )
    return receipt


def _close_current_accepted(
    controller: ExperimentLifecycleController,
    tmp_path: Path,
    label: str,
) -> Path:
    plan = _start(controller, tmp_path, label)
    raw = plan.run_dir / "raw" / "large.bin"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"raw-evidence")
    _seal_and_review(
        controller,
        tmp_path,
        label,
        reviewer_status=ReviewerStatus.ACCEPTED,
    )
    controller.classify(
        label,
        classification_context_path=_classification_context(
            controller, tmp_path, label, publication_state="current"
        ),
    )
    inventory_path = _content_inventory(plan.run_dir, label)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    source_bytes = sum(int(item["byte_count"]) for item in inventory["files"])
    performance = {
        "hashed_bytes": source_bytes,
        "source_bytes": source_bytes,
        "scan_passes": 1,
        "delete_traversals": 0,
        "duplicate_hashed_bytes": 0,
        "backend_calibrated": True,
        "implicit_fallback": False,
        "throughput_mib_per_second": 1.0,
        "cpu_utilization_percent": 10.0,
        "peak_memory_bytes": 1024,
        "io_utilization_percent": 10.0,
        "close_wall_seconds": 1.0,
        "backend": "windows_native",
        "workers": 1,
    }
    receipt = _retention_binding(
        controller,
        label,
        run_dir=plan.run_dir,
        inventory=inventory,
        performance=performance,
    )
    controller.mark_retained(
        label,
        retention_receipt_path=receipt,
        content_inventory_path=inventory_path,
    )
    controller.close(
        label,
        performance=ClosePerformance.from_dict(performance),
        storage_reconciliation_path=_storage_reconciliation(
            controller, tmp_path, label
        ),
    )
    return raw


def test_catalog_has_exactly_thirteen_governed_runner_entries() -> None:
    catalog = _catalog()

    assert len(catalog.specs) == 13
    assert catalog.for_run_label("stage05.2_benchmark_rerun03").experiment_id == (
        "stage052_performance"
    )
    with pytest.raises(LifecycleError, match="exactly one catalog"):
        catalog.for_run_label("stage06_unregistered_attempt01")


def test_v1_v2_migration_ledger_is_read_only_and_v3_is_new_truth() -> None:
    payload = load_lifecycle_migration_ledger(
        ROOT / "experiments/registries/experiment_lifecycle_v3_migration.json",
        repository=ROOT,
    )

    assert payload["new_write_schema_version"] == "experiment-lifecycle-v3"
    assert payload["v3_write_authority"] == "e_archive/.experiment-lifecycle"
    assert {item["role"] for item in payload["legacy_sources"]} == {
        "read_only_primary_legacy",
        "read_only_fallback",
    }


def test_historical_gate_rejects_handwritten_review_and_inventory(
    tmp_path: Path,
) -> None:
    migration_path = (
        ROOT
        / "experiments"
        / "registries"
        / "experiment_lifecycle_v3_migration.json"
    )
    migration = load_lifecycle_migration_ledger(migration_path, repository=ROOT)
    migration_sha256 = hashlib.sha256(migration_path.read_bytes()).hexdigest()
    gate_root = tmp_path / "historical-migration"
    archive_root = tmp_path / "archive"
    records: list[dict[str, object]] = []
    for label in migration["pending_historical_classification"]:
        assert isinstance(label, str)
        evidence_dir = gate_root / "evidence" / label
        inventory_path = evidence_dir / "content_inventory.json"
        inventory_sha256 = _write_signed_json(
            inventory_path,
            {
                    "schema_version": "experiment-content-inventory-v1",
                    "run_label": label,
                    "archive_root_alias": "e_archive",
                    "archive_root_resolved_path": str(archive_root.resolve()),
                    "archive_generation_relative_path": (
                        f"stage05.2/runs/{label}/generation-0001"
                    ),
                    "legacy_registry_sha256": HASH,
                    "legacy_tree_sha256": HASH,
                "files": [
                    {
                        "relative_path": "control/manifest.json",
                        "byte_count": 1,
                        "sha256": HASH,
                    }
                ],
            },
        )
        review_path = evidence_dir / "review_manifest.json"
        review_sha256 = _write_signed_json(
            review_path,
            {
                "run_label": label,
                "status": "FAILED_KNOWN",
                "retention_class": "unique_failure_capsule",
                "migration_ledger_sha256": migration_sha256,
                    "content_inventory_sha256": inventory_sha256,
                    "archive_root_alias": "e_archive",
                    "archive_root_resolved_path": str(archive_root.resolve()),
                    "archive_generation_relative_path": (
                        f"stage05.2/runs/{label}/generation-0001"
                    ),
                    "legacy_registry_sha256": HASH,
                    "legacy_tree_sha256": HASH,
                "reviewer_module_name": (
                    "evrptw.experiments.lifecycle_historical_review"
                ),
            },
        )
        records.append(
            {
                "run_label": label,
                "review_status": "FAILED_KNOWN",
                "retention_class": "unique_failure_capsule",
                "review_manifest_relative_path": review_path.relative_to(
                    gate_root
                ).as_posix(),
                "review_manifest_sha256": review_sha256,
                "content_inventory_relative_path": inventory_path.relative_to(
                    gate_root
                ).as_posix(),
                    "content_inventory_sha256": inventory_sha256,
                    "archive_root_alias": "e_archive",
                    "archive_root_resolved_path": str(archive_root.resolve()),
                    "archive_generation_relative_path": (
                        f"stage05.2/runs/{label}/generation-0001"
                    ),
                    "legacy_registry_sha256": HASH,
                    "legacy_tree_sha256": HASH,
                }
        )
    gate_path = gate_root / "gate.json"
    _write_signed_json(
        gate_path,
        {
            "schema_version": "experiment-lifecycle-historical-gate-v1",
            "migration_ledger_sha256": migration_sha256,
            "status": "complete",
            "records": records,
        },
    )

    with pytest.raises(LifecycleError):
        load_historical_migration_gate(
            gate_path,
            migration_ledger_path=migration_path,
            repository=ROOT,
        )


def test_historical_semantic_adjudication_closes_gate_only_when_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration_path = (
        ROOT
        / "experiments"
        / "registries"
        / "experiment_lifecycle_v3_migration.json"
    )
    migration = load_lifecycle_migration_ledger(migration_path, repository=ROOT)
    migration_sha256 = hashlib.sha256(migration_path.read_bytes()).hexdigest()
    output_root = tmp_path / "historical"
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    volume = VolumeIdentity("test-volume", "ext4")
    locator = StorageRootLocator(
        {"e_archive": StorageRoot("e_archive", archive_root, volume)}
    )
    monkeypatch.setattr(
        StorageRootLocator,
        "from_toml",
        classmethod(lambda _cls, _path: locator),
    )
    import evrptw.stage052_campaign_runner as campaign_runner

    monkeypatch.setattr(
        campaign_runner,
        "probe_volume_identity",
        lambda _path: volume,
    )
    trusted_registry_sha256 = str(
        next(
            item
            for item in migration["legacy_sources"]
            if item["schema_version"] == "experiment-retention-registry-v2"
        )["registry_sha256"]
    )
    labels = migration["pending_historical_classification"]
    assert isinstance(labels, list) and labels
    reviewer_module = "evrptw.experiments.stage052_historical_semantic_review"
    reviewer_path = (
        ROOT / "src/evrptw/experiments/stage052_historical_semantic_review.py"
    )
    reviewer_sha256 = hashlib.sha256(reviewer_path.read_bytes()).hexdigest()

    def adjudicate(label: str, inventory_path: Path) -> None:
        inventory_sha256 = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
        evidence_dir = output_root / "evidence" / label
        semantic_path = evidence_dir / "semantic_review.json"
        generation_relative = f"stage05.2/runs/{label}/generation-0001"
        _write_signed_json(
            semantic_path,
            {
                "schema_version": "experiment-historical-semantic-adjudication-v1",
                "run_label": label,
                "status": "FAILED_KNOWN",
                "retention_class": "unique_failure_capsule",
                "migration_ledger_sha256": migration_sha256,
                "content_inventory_sha256": inventory_sha256,
                "reviewer_module_name": reviewer_module,
                "failure_identity": {
                    "failure_code": "runner_failure",
                    "component": "runner",
                    "invariant_or_check": "startup",
                    "location": "control/manifest.json",
                },
            },
        )
        execution_path = evidence_dir / "semantic_review_execution.json"
        execution_sha256 = _write_signed_json(
            execution_path,
            {
                "schema_version": "experiment-historical-review-execution-v1",
                "run_label": label,
                "status": "completed",
                "exit_code": 0,
                "reviewer_module_name": reviewer_module,
                "reviewer_module_sha256": reviewer_sha256,
                "semantic_review_sha256": hashlib.sha256(
                    semantic_path.read_bytes()
                ).hexdigest(),
                "semantic_review_relative_path": "semantic_review.json",
                "migration_ledger_sha256_before": migration_sha256,
                "migration_ledger_sha256_after": migration_sha256,
                "content_inventory_sha256_before": inventory_sha256,
                "content_inventory_sha256_after": inventory_sha256,
                "archive_root_alias": "e_archive",
                "archive_root_resolved_path": str(archive_root.resolve()),
                "archive_generation_relative_path": generation_relative,
                "legacy_registry_sha256": trusted_registry_sha256,
                "legacy_tree_sha256": HASH,
                "command": [
                    sys.executable,
                    "-m",
                    reviewer_module,
                    "--migration-ledger",
                    str(migration_path.resolve()),
                    "--content-inventory",
                    str(inventory_path.resolve()),
                    "--run-label",
                    label,
                    "--output",
                    str(semantic_path.resolve()),
                    "--archive-root",
                    str(archive_root.resolve()),
                ],
            },
        )
        _write_signed_json(
            output_root / "executions" / f"{label}.json",
            {
                "schema_version": (
                    "experiment-historical-review-execution-binding-v1"
                ),
                "run_label": label,
                "review_execution_sha256": execution_sha256,
                "semantic_review_sha256": hashlib.sha256(
                    semantic_path.read_bytes()
                ).hexdigest(),
                "migration_ledger_sha256": migration_sha256,
                "content_inventory_sha256": inventory_sha256,
                "archive_root_alias": "e_archive",
                "archive_root_resolved_path": str(archive_root.resolve()),
                "archive_generation_relative_path": generation_relative,
                "legacy_registry_sha256": trusted_registry_sha256,
                "legacy_tree_sha256": HASH,
            },
        )
        apply_semantic_adjudication(
            catalog_path=ROOT / "configs" / "experiment_catalog.toml",
            migration_ledger_path=migration_path,
            output_root=output_root,
            run_label=label,
            semantic_review_path=semantic_path,
        )

    inventory_paths: dict[str, Path] = {}
    for raw_label in labels:
        label = str(raw_label)
        inventory_path = output_root / "evidence" / label / "content_inventory.json"
        _write_signed_json(
            inventory_path,
            {
                "schema_version": "experiment-content-inventory-v1",
                "run_label": label,
                "archive_root_alias": "e_archive",
                "archive_root_resolved_path": str(archive_root.resolve()),
                "archive_generation_relative_path": (
                    f"stage05.2/runs/{label}/generation-0001"
                ),
                "legacy_registry_sha256": trusted_registry_sha256,
                "legacy_tree_sha256": HASH,
                "files": [
                    {
                        "relative_path": "control/manifest.json",
                        "byte_count": 1,
                        "sha256": HASH,
                    }
                ],
            },
        )
        inventory_paths[label] = inventory_path

    first = str(labels[0])
    adjudicate(first, inventory_paths[first])

    with pytest.raises(LifecycleError, match="BLOCKED_RETENTION"):
        aggregate_historical_gate(
            repository=ROOT,
            catalog_path=ROOT / "configs" / "experiment_catalog.toml",
            migration_ledger_path=migration_path,
            output_root=output_root,
        )

    for raw_label in labels[1:]:
        label = str(raw_label)
        adjudicate(label, inventory_paths[label])
    gate_path = aggregate_historical_gate(
        repository=ROOT,
        catalog_path=ROOT / "configs" / "experiment_catalog.toml",
        migration_ledger_path=migration_path,
        output_root=output_root,
    )

    assert load_historical_migration_gate(
        gate_path,
        migration_ledger_path=migration_path,
        repository=ROOT,
    )["status"] == "complete"


def test_stage052_historical_semantic_reviewer_is_label_exact(
    tmp_path: Path,
) -> None:
    safe_label = "stage05.2_benchmark_attempt33"
    blocked_label = "stage05.2_hot_path_attempt04"
    keeper = "stage05.2_benchmark_attempt97"
    archive_root = tmp_path / "archive"
    generation_relative = (
        "stage05.2/runs/stage05.2_benchmark_attempt97/generation-0001"
    )
    generation = archive_root / generation_relative
    dependency_document = generation / "wsl_active/formal_memory_probe_report.json"
    dependency_document.parent.mkdir(parents=True)
    dependency_document.write_text(
        json.dumps({"run_label": keeper, "prerequisites": []}) + "\n",
        encoding="utf-8",
    )
    registry_path = archive_root / ".storage-governance/retention_registry_v2.json"
    _write_signed_json(
        registry_path,
        {
            "schema_version": "experiment-retention-registry-v2",
            "records": [
                {
                    "archive_relative_path": generation_relative,
                    "archive_root_alias": "e_archive",
                    "byte_count": dependency_document.stat().st_size,
                    "file_count": 1,
                    "generation": 1,
                    "retention_class": "unknown_full",
                    "run_label": keeper,
                    "tree_sha256": HASH,
                    "verification_status": "verified",
                }
            ],
        },
    )
    migration_path = tmp_path / "migration.json"
    _write_signed_json(
        migration_path,
        {
            "schema_version": "experiment-lifecycle-migration-v3",
            "pending_historical_classification": [safe_label, blocked_label],
            "protected_run_labels": [keeper],
            "legacy_sources": [
                {
                    "registry_sha256": hashlib.sha256(
                        registry_path.read_bytes()
                    ).hexdigest(),
                    "relative_path": (
                        ".storage-governance/retention_registry_v2.json"
                    ),
                    "role": "read_only_primary_legacy",
                    "root_alias": "e_archive",
                    "schema_version": "experiment-retention-registry-v2",
                }
            ],
            "protected_keeper_evidence": {
                keeper: {
                    "archive_relative_path": generation_relative,
                    "archive_root_alias": "e_archive",
                    "archive_tree_sha256": HASH,
                    "dependency_documents": [
                        {
                            "relative_path": (
                                "wsl_active/formal_memory_probe_report.json"
                            ),
                            "sha256": hashlib.sha256(
                                dependency_document.read_bytes()
                            ).hexdigest(),
                        }
                    ],
                }
            },
        },
    )

    def inventory(label: str) -> Path:
        path = tmp_path / label / "content_inventory.json"
        _write_signed_json(
            path,
            {
                "schema_version": "experiment-content-inventory-v1",
                "run_label": label,
                "files": [
                    {
                        "relative_path": "batch0001/batch_manifest.json",
                        "byte_count": 1,
                        "sha256": HASH,
                    }
                ],
            },
        )
        return path

    safe_inventory = inventory(safe_label)
    dependency_proof = build_no_dependency_proof(
        migration_ledger_path=migration_path,
        content_inventory_path=safe_inventory,
        run_label=safe_label,
        archive_root=archive_root,
        output_path=tmp_path / "dependency-proof.json",
    )
    safe = review_historical_stage052(
        migration_ledger_path=migration_path,
        content_inventory_path=safe_inventory,
        run_label=safe_label,
        dependency_proof_path=dependency_proof,
        archive_root=archive_root,
    )
    assert safe["retention_class"] == "superseded_metadata"
    assert safe["no_dependency_proof"] is True

    dependency_document.write_text(
        json.dumps({"run_label": keeper, "prerequisites": [safe_label]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(LifecycleError, match="document hash differs"):
        review_historical_stage052(
            migration_ledger_path=migration_path,
            content_inventory_path=safe_inventory,
            run_label=safe_label,
            dependency_proof_path=dependency_proof,
            archive_root=archive_root,
        )

    blocked = review_historical_stage052(
        migration_ledger_path=migration_path,
        content_inventory_path=inventory(blocked_label),
        run_label=blocked_label,
    )
    assert blocked["retention_class"] == "unknown_full"
    assert blocked["status"] == "INVALID"


def test_historical_inventory_uses_governance_tree_identity(
    tmp_path: Path,
) -> None:
    generation = tmp_path / "generation"
    (generation / "segment").mkdir(parents=True)
    (generation / "segment" / "first.json").write_text(
        '{"value": 1}\n', encoding="utf-8"
    )
    (generation / "second.bin").write_bytes(b"evidence")

    inventory, tree_sha256 = _inventory_generation(
        run_label="stage05.2_benchmark_attempt33",
        generation_dir=generation,
    )

    assert (
        inventory["file_count"],
        inventory["byte_count"],
        tree_sha256,
    ) == storage_governance.compute_tree_identity(generation)
    assert inventory["scan_passes"] == 1
    assert inventory["hash_backend"] == "python_thread_pool"
    assert inventory["hash_workers"] == 2


def test_historical_semantic_cli_writes_lifecycle_canonical_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = "stage05.2_hot_path_attempt04"
    migration_path = tmp_path / "migration.json"
    inventory_path = tmp_path / "inventory.json"
    output_path = tmp_path / "semantic.json"
    _write_signed_json(
        migration_path,
        {
            "schema_version": "experiment-lifecycle-migration-v3",
            "pending_historical_classification": [label],
            "protected_run_labels": ["stage05.2_benchmark_attempt97"],
        },
    )
    _write_signed_json(
        inventory_path,
        {
            "schema_version": "experiment-content-inventory-v1",
            "run_label": label,
            "files": [{"relative_path": "control/manifest.json"}],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage052_historical_semantic_review",
            "--migration-ledger",
            str(migration_path),
            "--content-inventory",
            str(inventory_path),
            "--run-label",
            label,
            "--output",
            str(output_path),
        ],
    )

    assert historical_semantic_main() == 0
    assert lifecycle_module._load_signed_json(output_path)["status"] == "INVALID"


def test_stage052_hot_path_requires_accepted_lineage_and_successor(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    current_label = "stage05.2_hot_path_attempt04"
    successor_label = "stage05.2_hot_path_attempt06"
    gate_names = {
        "exact_scope",
        "optimization_profile",
        "performance_promotion",
        "persistence_attribution",
        "prerequisite_performance_baseline",
        "replay_consistency",
        "runtime_identity",
        "source_snapshot",
        "staging_root_identity",
    }

    def gates(
        *failed: str, details: dict[str, str] | None = None
    ) -> dict[str, dict[str, object]]:
        detail_by_name = details or {}
        return {
            name: {
                "detail": detail_by_name.get(name, name),
                "passed": name not in failed,
            }
            for name in sorted(gate_names)
        }

    def write_json(path: Path, payload: object) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def raw_manifest(generation: Path, segment: str, label: str) -> str:
        path = generation / segment / "control" / f"{label}_manifest.json"
        digest = write_json(
            path,
            {"component": "hot_path", "run_label": label},
        )
        path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
        return digest

    def review_generation(
        generation: Path,
        relative_root: Path,
        generation_id: str,
    ) -> dict[str, str]:
        review_dir = generation / relative_root / "generations" / generation_id
        report = review_dir / "review_report.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("review\n", encoding="utf-8")
        relative = f"generations/{generation_id}/review_report.md"
        return {relative: hashlib.sha256(report.read_bytes()).hexdigest()}

    def archived_review(
        generation: Path,
        *,
        label: str,
        raw_sha256: str,
        status: str,
        gates: dict[str, dict[str, object]],
        generation_id: str,
        retry_history: list[str] | None = None,
    ) -> str:
        placeholder = generation / "d_history" / "review" / "history" / generation_id
        files = review_generation(generation, placeholder.relative_to(generation), "output")
        manifest = {
            "component": "hot_path",
            "files": files,
            "gates": gates,
            "raw_manifest_sha256": raw_sha256,
            "review_manifest_lineage_sha256": [],
            "run_label": label,
            "schema_version": "stage05.2-review-v1",
            "scope": "performance",
            "status": status,
        }
        if retry_history is not None:
            manifest["review_retry_history_sha256"] = retry_history
        manifest_path = placeholder / "review_manifest.json"
        digest = write_json(manifest_path, manifest)
        final = placeholder.with_name(digest)
        placeholder.rename(final)
        return digest

    current_relative = f"stage05.2/runs/{current_label}/generation-0001"
    current_generation = archive_root / current_relative
    current_raw = raw_manifest(current_generation, "d_history", current_label)
    retry_one = archived_review(
        current_generation,
        label=current_label,
        raw_sha256=current_raw,
        status="NOT_READY",
        gates=gates("runtime_identity"),
        generation_id="retry-one",
    )
    retry_two = archived_review(
        current_generation,
        label=current_label,
        raw_sha256=current_raw,
        status="NOT_READY",
        gates=gates("prerequisite_performance_baseline"),
        generation_id="retry-two",
        retry_history=[retry_one],
    )
    accepted = archived_review(
        current_generation,
        label=current_label,
        raw_sha256=current_raw,
        status="READY_FOR_STAGE052_ARTIFACT_STREAMING",
        gates=gates(),
        generation_id="accepted",
        retry_history=[retry_one, retry_two],
    )
    current_review_root = current_generation / "d_history" / "review"
    current_files = review_generation(
        current_generation,
        Path("d_history/review"),
        "terminal",
    )
    current_review = {
        "component": "hot_path",
        "files": current_files,
        "gates": gates(
            "prerequisite_performance_baseline",
            "source_snapshot",
            details={
                "prerequisite_performance_baseline": (
                    "producer metadata is not bound to the reviewed prerequisite role"
                ),
                "source_snapshot": (
                    "raw source snapshot identity does not match independent live replay"
                ),
            },
        ),
        "raw_manifest_sha256": current_raw,
        "review_manifest_lineage_sha256": [accepted],
        "review_retry_history_sha256": [retry_one, retry_two],
        "run_label": current_label,
        "schema_version": "stage05.2-review-v1",
        "scope": "performance",
        "status": "NOT_READY",
    }
    current_review_sha256 = write_json(
        current_review_root / "review_manifest.json", current_review
    )
    write_json(
        current_review_root / "review_execution.json",
        {
            "exit_code": 0,
            "finalized": True,
            "raw_manifest_sha256_after": current_raw,
            "raw_manifest_sha256_before": current_raw,
            "raw_manifest_unchanged": True,
            "review_manifest_sha256": current_review_sha256,
            "reviewer_module_name": (
                "evrptw.experiments.stage052_performance_review"
            ),
            "run_label": current_label,
            "schema_version": "stage05.2-review-execution-v1",
            "status": "completed",
        },
    )

    successor_relative = f"stage05.2/runs/{successor_label}/generation-0001"
    successor_generation = archive_root / successor_relative
    successor_raw = raw_manifest(successor_generation, "wsl_active", successor_label)
    successor_review_root = successor_generation / "wsl_active" / "review"
    successor_files = review_generation(
        successor_generation,
        Path("wsl_active/review"),
        "accepted",
    )
    successor_review = {
        "component": "hot_path",
        "files": successor_files,
        "gates": gates(),
        "raw_manifest_sha256": successor_raw,
        "review_manifest_lineage_sha256": [],
        "review_retry_history_sha256": [],
        "run_label": successor_label,
        "schema_version": "stage05.2-review-v1",
        "scope": "performance",
        "status": "READY_FOR_STAGE052_ARTIFACT_STREAMING",
    }
    successor_review_sha256 = write_json(
        successor_review_root / "review_manifest.json", successor_review
    )
    write_json(
        successor_review_root / "review_execution.json",
        {
            "exit_code": 0,
            "finalized": True,
            "raw_manifest_sha256_after": successor_raw,
            "raw_manifest_sha256_before": successor_raw,
            "raw_manifest_unchanged": True,
            "review_manifest_sha256": successor_review_sha256,
            "reviewer_module_name": (
                "evrptw.experiments.stage052_performance_review"
            ),
            "run_label": successor_label,
            "schema_version": "stage05.2-review-execution-v1",
            "status": "completed",
        },
    )

    current_inventory, current_tree = _inventory_generation(
        run_label=current_label,
        generation_dir=current_generation,
    )
    successor_inventory, successor_tree = _inventory_generation(
        run_label=successor_label,
        generation_dir=successor_generation,
    )
    registry_path = archive_root / ".storage-governance/retention_registry_v2.json"
    _write_signed_json(
        registry_path,
        {
            "schema_version": "experiment-retention-registry-v2",
            "records": [
                {
                    "archive_relative_path": current_relative,
                    "archive_root_alias": "e_archive",
                    "byte_count": current_inventory["byte_count"],
                    "file_count": current_inventory["file_count"],
                    "generation": 1,
                    "retention_class": "unknown_full",
                    "run_label": current_label,
                    "tree_sha256": current_tree,
                    "verification_status": "verified",
                },
                {
                    "archive_relative_path": successor_relative,
                    "archive_root_alias": "e_archive",
                    "byte_count": successor_inventory["byte_count"],
                    "file_count": successor_inventory["file_count"],
                    "generation": 1,
                    "retention_class": "unknown_full",
                    "run_label": successor_label,
                    "tree_sha256": successor_tree,
                    "verification_status": "verified",
                },
            ],
        },
    )
    migration_path = tmp_path / "migration.json"
    _write_signed_json(
        migration_path,
        {
            "schema_version": "experiment-lifecycle-migration-v3",
            "pending_historical_classification": [current_label],
            "protected_run_labels": ["stage05.2_benchmark_attempt97"],
            "legacy_sources": [
                {
                    "registry_sha256": hashlib.sha256(
                        registry_path.read_bytes()
                    ).hexdigest(),
                    "relative_path": (
                        ".storage-governance/retention_registry_v2.json"
                    ),
                    "role": "read_only_primary_legacy",
                    "root_alias": "e_archive",
                    "schema_version": "experiment-retention-registry-v2",
                }
            ],
        },
    )
    current_inventory.update(
        {
            "archive_root_alias": "e_archive",
            "archive_root_resolved_path": str(archive_root.resolve()),
            "archive_generation_relative_path": current_relative,
            "legacy_registry_sha256": hashlib.sha256(
                registry_path.read_bytes()
            ).hexdigest(),
            "legacy_tree_sha256": current_tree,
        }
    )
    inventory_path = tmp_path / "content_inventory.json"
    _write_signed_json(inventory_path, current_inventory)

    result = review_historical_stage052(
        migration_ledger_path=migration_path,
        content_inventory_path=inventory_path,
        run_label=current_label,
        archive_root=archive_root,
    )
    assert result["status"] == "PARTIAL"
    assert result["retention_class"] == "superseded_accepted_capsule"
    proof = result["supersession_proof"]
    assert isinstance(proof, dict)
    assert proof["accepted_review_manifest_sha256"] == accepted
    assert proof["successor_run_label"] == successor_label

    successor_review["status"] = "NOT_READY"
    write_json(successor_review_root / "review_manifest.json", successor_review)
    with pytest.raises(LifecycleError, match="successor differs"):
        review_historical_stage052(
            migration_ledger_path=migration_path,
            content_inventory_path=inventory_path,
            run_label=current_label,
            archive_root=archive_root,
        )


def test_historical_semantic_execution_rejects_archive_root_substitution(
    tmp_path: Path,
) -> None:
    label = "stage05.2_benchmark_attempt33"
    output_root = tmp_path / "historical"
    evidence_dir = output_root / "evidence" / label
    canonical_archive = tmp_path / "canonical-archive"
    mirror_archive = tmp_path / "mirror-archive"
    canonical_archive.mkdir()
    mirror_archive.mkdir()
    inventory_path = evidence_dir / "content_inventory.json"
    generation_relative = f"stage05.2/runs/{label}/generation-0001"
    _write_signed_json(
        inventory_path,
        {
            "schema_version": "experiment-content-inventory-v1",
            "run_label": label,
            "archive_root_alias": "e_archive",
            "archive_root_resolved_path": str(canonical_archive.resolve()),
            "archive_generation_relative_path": generation_relative,
            "legacy_registry_sha256": HASH,
            "legacy_tree_sha256": HASH,
            "files": [{"relative_path": "control/manifest.json"}],
        },
    )
    _write_signed_json(
        evidence_dir / "review_manifest.json",
        {
            "schema_version": "experiment-historical-migration-gate-v1",
            "run_label": label,
            "archive_root_alias": "e_archive",
            "archive_root_resolved_path": str(canonical_archive.resolve()),
            "archive_generation_relative_path": generation_relative,
            "legacy_registry_sha256": HASH,
            "legacy_tree_sha256": HASH,
        },
    )
    semantic_path = evidence_dir / "semantic_review.json"
    migration_path = (
        ROOT
        / "experiments"
        / "registries"
        / "experiment_lifecycle_v3_migration.json"
    )
    command = (
        sys.executable,
        "-m",
        "evrptw.experiments.stage052_historical_semantic_review",
        "--migration-ledger",
        str(migration_path.resolve()),
        "--content-inventory",
        str(inventory_path.resolve()),
        "--run-label",
        label,
        "--output",
        str(semantic_path.resolve()),
        "--archive-root",
        str(mirror_archive.resolve()),
    )

    with pytest.raises(LifecycleError, match="inputs are not canonical"):
        execute_semantic_reviewer(
            catalog_path=ROOT / "configs/experiment_catalog.toml",
            migration_ledger_path=migration_path,
            output_root=output_root,
            archive_root=canonical_archive,
            run_label=label,
            semantic_review_path=semantic_path,
            command=command,
        )


def test_physical_historical_review_rejects_mirror_and_fake_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evrptw.stage052_campaign_runner as campaign_runner

    repository = tmp_path / "repository"
    canonical_archive = tmp_path / "canonical-archive"
    mirror_archive = tmp_path / "mirror-archive"
    canonical_archive.mkdir()
    mirror_archive.mkdir()
    locator_path = repository / "configs/stage052_storage_roots.local.toml"
    locator_path.parent.mkdir(parents=True)
    locator_path.write_text(
        "\n".join(
            (
                "[roots.e_archive]",
                f'absolute_path = "{canonical_archive.as_posix()}"',
                'device_uuid = "test-volume"',
                'filesystem = "ext4"',
                "",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        campaign_runner,
        "probe_volume_identity",
        lambda _path: VolumeIdentity("test-volume", "ext4"),
    )
    v1_path = repository / "experiments/registries/stage05.2_retention_registry.csv"
    v1_path.parent.mkdir(parents=True)
    v1_path.write_text("trusted-v1\n", encoding="utf-8")
    trusted_registry = (
        canonical_archive / ".storage-governance/retention_registry_v2.json"
    )
    fake_registry = mirror_archive / ".storage-governance/retention_registry_v2.json"
    registry_payload = {
        "schema_version": "experiment-retention-registry-v2",
        "records": [],
    }
    _write_signed_json(trusted_registry, registry_payload)
    _write_signed_json(fake_registry, registry_payload)
    migration_path = repository / "experiments/registries/migration.json"
    _write_signed_json(
        migration_path,
        {
            "schema_version": "experiment-lifecycle-migration-v3",
            "new_write_schema_version": "experiment-lifecycle-v3",
            "v3_write_authority": "e_archive/.experiment-lifecycle",
            "legacy_sources": [
                {
                    "schema_version": "experiment-retention-registry-v2",
                    "role": "read_only_primary_legacy",
                    "root_alias": "e_archive",
                    "record_count": 1,
                    "relative_path": (
                        ".storage-governance/retention_registry_v2.json"
                    ),
                    "registry_sha256": hashlib.sha256(
                        trusted_registry.read_bytes()
                    ).hexdigest(),
                },
                {
                    "schema_version": "stage05.2-retention-v1",
                    "relative_path": (
                        "experiments/registries/stage05.2_retention_registry.csv"
                    ),
                    "registry_sha256": hashlib.sha256(
                        v1_path.read_bytes()
                    ).hexdigest(),
                },
            ],
            "protected_run_labels": ["stage05.2_benchmark_attempt97"],
            "pending_historical_classification": [
                "stage05.2_benchmark_attempt33"
            ],
        },
    )

    with pytest.raises(LifecycleError, match="not canonical e_archive"):
        review_historical_generation(
            repository=repository,
            archive_root=mirror_archive,
            migration_ledger_path=migration_path,
            registry_path=fake_registry,
            output_root=tmp_path / "output",
            run_label="stage05.2_benchmark_attempt33",
        )
    with pytest.raises(LifecycleError, match="not migration-ledger trusted"):
        review_historical_generation(
            repository=repository,
            archive_root=canonical_archive,
            migration_ledger_path=migration_path,
            registry_path=fake_registry,
            output_root=tmp_path / "output",
            run_label="stage05.2_benchmark_attempt33",
        )


def test_cli_exposes_the_complete_lifecycle_surface() -> None:
    help_text = _cli().format_help()

    for command in (
        "plan",
        "start",
        "seal",
        "review",
        "run-review",
        "adjudicate",
        "close",
        "status",
        "audit",
    ):
        assert command in help_text


def test_mutating_cli_requires_all_canonical_storage_roots(tmp_path: Path) -> None:
    with pytest.raises(LifecycleError, match="require canonical storage"):
        lifecycle_main(
            [
                "--catalog",
                str(ROOT / "configs/experiment_catalog.toml"),
                "--state-root",
                str((tmp_path / "state").resolve()),
                "plan",
                "--plan",
                str(tmp_path / "not-read.json"),
            ]
        )


def test_mutating_cli_rejects_roots_outside_the_locator(tmp_path: Path) -> None:
    locator = ROOT / "configs/stage052_storage_roots.local.toml"
    assert locator.is_file()
    with pytest.raises(LifecycleError, match="differ from the canonical locator"):
        lifecycle_main(
            [
                "--catalog",
                str(ROOT / "configs/experiment_catalog.toml"),
                "--state-root",
                str((tmp_path / "state").resolve()),
                "--storage-state-root",
                str((tmp_path / "storage").resolve()),
                "--capacity-state-root",
                str((tmp_path / "capacity").resolve()),
                "--archive-root",
                str((tmp_path / "archive").resolve()),
                "--storage-root-locator",
                str(locator),
                "plan",
                "--plan",
                str(tmp_path / "not-read.json"),
            ]
        )


def test_only_one_unclosed_top_level_experiment_is_permitted(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    first = _plan(tmp_path, "stage00_baseline_attempt01")

    assert controller.plan(first).state == LifecycleState.PLANNED
    assert controller.plan(first).plan_sha256 == first.plan_sha256
    with pytest.raises(LifecycleError, match="not CLOSED"):
        controller.plan(_plan(tmp_path, "stage01_objective_attempt01"))


def test_runtime_resource_drift_is_rejected_even_after_idempotent_start(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    plan = _start(controller, tmp_path, label)
    payload = plan.to_dict()
    payload["workers"] = 2
    drifted = ExperimentPlan.from_dict(payload)

    with pytest.raises(LifecycleError, match="runtime resources"):
        controller.start(label, runtime_plan=drifted)


def test_permit_retry_after_start_is_idempotent(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    _start(controller, tmp_path, label)

    resumed = controller.permit(
        label,
        storage_permit_path=controller.capacity_state_root
        / "permits"
        / f"{label}.json",
    )

    assert resumed.state == LifecycleState.RUNNING


def test_plan_rejects_extra_prerequisite_contracts(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    plan = replace(
        _plan(tmp_path, "stage01_objective_attempt01"),
        prerequisite_sha256_by_contract={
            "stage00_baseline": HASH,
            "undeclared_contract": HASH,
        },
    )

    with pytest.raises(LifecycleError, match="prerequisite contracts differ"):
        controller.plan(plan)


def test_review_rejects_manifest_without_artifact_inventory(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    plan = _start(controller, tmp_path, label)
    manifest = plan.run_dir / "control" / "manifest.json"
    _write_signed_json(
        manifest,
        {
            "run_label": label,
            "status": "complete",
            "evidence_completeness": "complete",
        },
    )
    controller.seal(label, manifest_path=manifest)
    review = plan.run_dir / "review" / "review.json"
    _write_signed_json(
        review,
        {"run_label": label, "status": "ACCEPTED", "files": {}},
    )
    _review_execution(controller, label, review)

    with pytest.raises(LifecycleError, match="no incremental artifact inventory"):
        controller.review(label, review_manifest_path=review)


def test_seal_accepts_the_existing_artifact_sidecar_convention(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    plan = _start(controller, tmp_path, label)
    artifact = plan.run_dir / "result.txt"
    artifact.write_bytes(b"result")
    manifest = plan.run_dir / "control" / "manifest.json"
    atomic_write_signed_json(
        manifest,
        {
            "run_label": label,
            "status": "complete",
            "evidence_completeness": "complete",
            "artifacts": [
                {
                    "relative_path": "result.txt",
                    "byte_size": artifact.stat().st_size,
                    "checksum": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ],
        },
    )

    assert manifest.with_suffix(".sha256").is_file()
    assert controller.seal(label, manifest_path=manifest).state == (
        LifecycleState.SEALED
    )


def test_terminal_manifest_reuses_signed_child_artifact_identities(
    tmp_path: Path,
) -> None:
    label = "stage05.2_resource_calibration_attempt01"
    output = tmp_path / label
    child = output / "workers4" / "c101_21" / "2014"
    raw = child / "raw" / "events.parquet"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"parquet")
    child_manifest = child / "control" / "child_manifest.json"
    atomic_write_signed_json(
        child_manifest,
        {
            "run_label": label,
            "status": "complete",
            "evidence_completeness": "complete",
            "artifacts": [
                {
                    "relative_path": "raw/events.parquet",
                    "byte_size": raw.stat().st_size,
                    "checksum": hashlib.sha256(raw.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    atomic_write_signed_json(
        output / "calibration_report.json",
        {"run_label": label, "status": "complete"},
    )

    terminal = build_cli_terminal_manifest(
        output_dir=output,
        run_label=label,
    )
    payload = json.loads(terminal.read_text(encoding="utf-8"))

    assert {item["relative_path"] for item in payload["artifacts"]} == {
        "calibration_report.json",
        "calibration_report.sha256",
        "workers4/c101_21/2014/control/child_manifest.json",
        "workers4/c101_21/2014/control/child_manifest.sha256",
        "workers4/c101_21/2014/raw/events.parquet",
    }
    omitted = child / "raw" / "omitted.bin"
    omitted.write_bytes(b"omitted")
    with pytest.raises(
        StorageGovernanceError, match="terminal manifest file set differs"
    ):
        build_cli_terminal_manifest(output_dir=output, run_label=label)
    omitted.unlink()
    raw.write_bytes(b"PARQUET")
    with pytest.raises(StorageGovernanceError, match="child artifact checksum differs"):
        build_cli_terminal_manifest(output_dir=output, run_label=label)


def test_failed_cli_attempt_persists_partial_inventory_before_sealing(
    tmp_path: Path,
) -> None:
    label = "stage00_baseline_attempt01"
    output = tmp_path / label
    output.mkdir()
    (output / "partial.log").write_text("started\n", encoding="utf-8")
    atomic_write_signed_json(
        output / "failure_summary.json",
        {"run_label": label, "status": "failed"},
    )
    manifest_path = storage_governance.build_cli_failure_manifest(
        output_dir=output,
        run_label=label,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["evidence_completeness"] == "partial"
    assert {item["relative_path"] for item in payload["artifacts"]} == {
        "failure_summary.json",
        "failure_summary.sha256",
        "partial.log",
    }


def test_failed_cli_attempt_capsules_a_broken_child_manifest(
    tmp_path: Path,
) -> None:
    label = "stage00_baseline_attempt01"
    output = tmp_path / label
    broken = output / "child" / "control" / "broken_manifest.json"
    broken.parent.mkdir(parents=True)
    broken.write_text("{broken", encoding="utf-8")
    atomic_write_signed_json(
        output / "failure_summary.json",
        {"run_label": label, "status": "failed"},
    )
    storage_governance.build_cli_failure_manifest(
        output_dir=output,
        run_label=label,
    )

    manifest = (
        output / "control" / f"{label}_failure_lifecycle_manifest.json"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["artifact_trust"] == "untrusted_failure_capsule"
    assert "child/control/broken_manifest.json" in {
        item["relative_path"] for item in payload["artifacts"]
    }


def test_failed_cli_attempt_creates_only_the_planned_missing_run_directory(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    shutil.copy2(ROOT / "configs/experiment_catalog.toml", config_dir)
    (config_dir / "runner.toml").write_text("schema = 1\n", encoding="utf-8")
    archive_root = tmp_path / "archive"
    staging_root = tmp_path / "staging"
    (config_dir / "stage052_storage_roots.local.toml").write_text(
        "\n".join(
            (
                "[roots.e_archive]",
                f'absolute_path = "{archive_root}"',
                'device_uuid = "archive"',
                'filesystem = "test"',
                "[roots.wsl_staging]",
                f'absolute_path = "{staging_root}"',
                'device_uuid = "staging"',
                'filesystem = "test"',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    controller = ExperimentLifecycleController(
        catalog=ExperimentCatalog.from_toml(
            config_dir / "experiment_catalog.toml"
        ),
        state_root=archive_root / ".experiment-lifecycle",
        storage_state_root=archive_root / ".storage-governance",
        capacity_state_root=staging_root / ".storage-governance",
        archive_root=archive_root,
    )
    label = "stage00_baseline_attempt01"
    plan = _start(controller, tmp_path, label)
    plan.run_dir.rmdir()

    assert storage_governance.seal_failed_cli_attempt(
        config_path=config_dir / "runner.toml",
        output_dir=plan.run_dir,
        run_label=label,
        error=RuntimeError("failed before producer mkdir"),
    )
    assert plan.run_dir.is_dir()
    assert controller._load(label).state == LifecycleState.SEALED


def test_terminal_manifest_detects_mutation_during_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    label = "stage05.2_resource_calibration_attempt01"
    output = tmp_path / label
    raw = output / "child" / "raw" / "events.bin"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"AAAA")
    child_manifest = output / "child" / "control" / "child_manifest.json"
    atomic_write_signed_json(
        child_manifest,
        {
            "run_label": label,
            "status": "complete",
            "evidence_completeness": "complete",
            "artifacts": [
                {
                    "relative_path": "raw/events.bin",
                    "byte_size": 4,
                    "checksum": hashlib.sha256(b"AAAA").hexdigest(),
                }
            ],
        },
    )
    original_sha256 = storage_governance._file_sha256

    def mutate_after_hash(path: Path) -> str:
        digest = original_sha256(path)
        if path == raw:
            raw.write_bytes(b"BBBB")
        return digest

    monkeypatch.setattr(storage_governance, "_file_sha256", mutate_after_hash)
    with pytest.raises(StorageGovernanceError, match="changed while hashing"):
        build_cli_terminal_manifest(output_dir=output, run_label=label)


def test_execute_reviewer_writes_uniform_receipt_and_allows_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    plan = _start(controller, tmp_path, label)
    artifact = plan.run_dir / "raw" / "result.json"
    artifact.parent.mkdir()
    artifact.write_bytes(b"{}\n")
    manifest = plan.run_dir / "control" / "manifest.json"
    _write_signed_json(
        manifest,
        {
            "run_label": label,
            "status": "complete",
            "evidence_completeness": "complete",
            "artifacts": [
                {
                    "relative_path": artifact.relative_to(plan.run_dir).as_posix(),
                    "byte_size": artifact.stat().st_size,
                    "checksum": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    controller.seal(label, manifest_path=manifest)
    review = plan.run_dir / "review" / "review.json"

    def fake_run(
        command: tuple[str, ...], *, check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert check is False
        atomic_write_signed_json(
            review,
            {"run_label": label, "status": "ACCEPTED", "files": {}},
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = controller.catalog.for_run_label(label)
    receipt = controller.execute_reviewer(
        label,
        raw_manifest_path=manifest,
        review_manifest_path=review,
        command=(sys.executable, "-m", spec.reviewer_module),
    )

    assert receipt == plan.run_dir / "review" / "review_execution.json"
    assert controller.review(label, review_manifest_path=review).state == (
        LifecycleState.REVIEWED
    )


def test_execute_reviewer_rejects_a_decorative_module_argument(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    plan = _start(controller, tmp_path, label)
    artifact = plan.run_dir / "raw" / "result.json"
    artifact.parent.mkdir()
    artifact.write_bytes(b"{}\n")
    manifest = plan.run_dir / "control" / "manifest.json"
    _write_signed_json(
        manifest,
        {
            "run_label": label,
            "status": "complete",
            "evidence_completeness": "complete",
            "artifacts": [
                {
                    "relative_path": artifact.relative_to(plan.run_dir).as_posix(),
                    "byte_size": artifact.stat().st_size,
                    "checksum": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    controller.seal(label, manifest_path=manifest)
    spec = controller.catalog.for_run_label(label)

    with pytest.raises(LifecycleError, match="does not invoke"):
        controller.execute_reviewer(
            label,
            raw_manifest_path=manifest,
            review_manifest_path=plan.run_dir / "review" / "review.json",
            command=(
                sys.executable,
                "-c",
                "raise SystemExit(0)",
                "-m",
                spec.reviewer_module,
            ),
        )


def test_execute_reviewer_rejects_same_size_raw_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    plan = _start(controller, tmp_path, label)
    artifact = plan.run_dir / "raw" / "result.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"AAAA")
    manifest = plan.run_dir / "control" / "manifest.json"
    _write_signed_json(
        manifest,
        {
            "run_label": label,
            "status": "complete",
            "evidence_completeness": "complete",
            "artifacts": [
                {
                    "relative_path": artifact.relative_to(plan.run_dir).as_posix(),
                    "byte_size": 4,
                    "checksum": hashlib.sha256(b"AAAA").hexdigest(),
                }
            ],
        },
    )
    controller.seal(label, manifest_path=manifest)
    review = plan.run_dir / "review" / "review.json"

    def fake_run(
        command: tuple[str, ...], *, check: bool
    ) -> subprocess.CompletedProcess[str]:
        artifact.write_bytes(b"BBBB")
        atomic_write_signed_json(
            review,
            {"run_label": label, "status": "ACCEPTED", "files": {}},
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = controller.catalog.for_run_label(label)
    with pytest.raises(LifecycleError, match="changed sealed raw artifact"):
        controller.execute_reviewer(
            label,
            raw_manifest_path=manifest,
            review_manifest_path=review,
            command=(sys.executable, "-m", spec.reviewer_module),
        )


def test_initial_transition_recovers_an_orphan_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path)
    plan = _plan(tmp_path, "stage00_baseline_attempt01")
    event_sidecar = (
        controller.state_root
        / "events"
        / plan.run_label
        / "0000-PLANNED.json.sha256"
    )
    original_replace = os.replace
    failed = False

    def interrupted_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        if Path(destination) == event_sidecar and not failed:
            failed = True
            raise OSError("simulated first-generation crash")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="simulated"):
        controller.plan(plan)
    monkeypatch.setattr(os, "replace", original_replace)

    recovered = controller.plan(plan)
    assert recovered.state == LifecycleState.PLANNED
    assert controller._record_path(plan.run_label).is_file()


def test_initial_transition_recovers_an_orphan_current_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path)
    plan = _plan(tmp_path, "stage00_baseline_attempt01")
    record_sidecar = controller._record_path(plan.run_label).with_suffix(
        ".json.sha256"
    )
    original_replace = os.replace
    failed = False

    def interrupted_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal failed
        if Path(destination) == record_sidecar and not failed:
            failed = True
            raise OSError("simulated current-record crash")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="current-record"):
        controller.plan(plan)
    monkeypatch.setattr(os, "replace", original_replace)

    assert controller.plan(plan).state == LifecycleState.PLANNED


@pytest.mark.parametrize(
    ("field", "value"),
    (("hashed_bytes", -1), ("throughput_mib_per_second", float("nan"))),
)
def test_close_performance_rejects_negative_and_non_finite_values(
    field: str,
    value: int | float,
) -> None:
    values: dict[str, object] = {
        "hashed_bytes": 1,
        "source_bytes": 1,
        "scan_passes": 1,
        "delete_traversals": 0,
        "duplicate_hashed_bytes": 0,
        "backend_calibrated": True,
        "implicit_fallback": False,
        "throughput_mib_per_second": 1.0,
        "cpu_utilization_percent": 1.0,
        "peak_memory_bytes": 1,
        "io_utilization_percent": 1.0,
        "close_wall_seconds": 1.0,
        "backend": "windows_native",
        "workers": 1,
    }
    values[field] = value

    with pytest.raises(LifecycleError, match="close performance gate failed"):
        ClosePerformance.from_dict(values).verify()


def test_complete_lifecycle_reaches_closed_only_after_retention_and_gate(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    _start(controller, tmp_path, label)
    _seal_and_review(
        controller,
        tmp_path,
        label,
        reviewer_status=ReviewerStatus.ACCEPTED,
    )

    decision = controller.classify(
        label,
        classification_context_path=_classification_context(
            controller, tmp_path, label, publication_state="current"
        ),
    )
    assert decision.retention_class == RetentionClassV3.CURRENT_ACCEPTED_FULL
    inventory_path = _content_inventory(tmp_path / label, label)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    source_bytes = sum(int(item["byte_count"]) for item in inventory["files"])
    performance = {
        "hashed_bytes": source_bytes,
        "source_bytes": source_bytes,
        "scan_passes": 1,
        "delete_traversals": 0,
        "duplicate_hashed_bytes": 0,
        "backend_calibrated": True,
        "implicit_fallback": False,
        "throughput_mib_per_second": 1.0,
        "cpu_utilization_percent": 10.0,
        "peak_memory_bytes": 1024,
        "io_utilization_percent": 10.0,
        "close_wall_seconds": 1.0,
        "backend": "windows_native",
        "workers": 1,
    }
    receipt = _retention_binding(
        controller,
        label,
        run_dir=tmp_path / label,
        inventory=inventory,
        performance=performance,
    )
    assert controller.mark_retained(
        label,
        retention_receipt_path=receipt,
        content_inventory_path=inventory_path,
    ).state == (
        LifecycleState.RETAINED
    )
    with pytest.raises(LifecycleError, match="performance gate"):
        controller.close(
            label,
            performance=ClosePerformance(
                hashed_bytes=2,
                source_bytes=1,
                scan_passes=2,
                delete_traversals=2,
                duplicate_hashed_bytes=1,
                backend_calibrated=False,
                implicit_fallback=True,
                throughput_mib_per_second=1.0,
                cpu_utilization_percent=10.0,
                peak_memory_bytes=1024,
                io_utilization_percent=10.0,
                close_wall_seconds=1.0,
                backend="windows_native",
                workers=1,
            ),
            storage_reconciliation_path=_storage_reconciliation(
                controller, tmp_path, label
            ),
        )
    closed = controller.close(
        label,
        performance=ClosePerformance(
            hashed_bytes=source_bytes,
            source_bytes=source_bytes,
            scan_passes=1,
            delete_traversals=0,
            duplicate_hashed_bytes=0,
            backend_calibrated=True,
            implicit_fallback=False,
            throughput_mib_per_second=1.0,
            cpu_utilization_percent=10.0,
            peak_memory_bytes=1024,
            io_utilization_percent=10.0,
            close_wall_seconds=1.0,
            backend="windows_native",
            workers=1,
        ),
        storage_reconciliation_path=_storage_reconciliation(
            controller, tmp_path, label
        ),
    )
    assert closed.state == LifecycleState.CLOSED
    assert controller.audit()["passed"] is True


def test_new_current_close_demotes_previous_current_in_same_transaction(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    first = "stage00_baseline_attempt01"
    second = "stage00_baseline_attempt02"
    first_raw = _close_current_accepted(controller, tmp_path, first)
    assert first_raw.is_file()

    _close_current_accepted(controller, tmp_path, second)

    first_record = next(
        item for item in controller.records() if item.run_label == first
    )
    assert first_record.state == LifecycleState.CLOSED
    assert first_record.retention_class == (
        RetentionClassV3.CURRENT_ACCEPTED_FULL
    )
    supersession = json.loads(
        (
            controller.state_root
            / "close"
            / "supersessions"
            / f"{first}.json"
        ).read_text(encoding="utf-8")
    )
    assert supersession["effective_retention_class"] == (
        RetentionClassV3.SUPERSEDED_ACCEPTED_CAPSULE.value
    )
    assert supersession["superseded_by"] == second
    first_status = controller.status_payload(first_record)
    assert first_status["effective_retention_class"] == (
        RetentionClassV3.SUPERSEDED_ACCEPTED_CAPSULE.value
    )
    assert first_status["superseded_by"] == second
    archived_raw = (
        controller.archive_root
        / "stage00"
        / "runs"
        / first
        / "generation-0001"
        / "raw"
        / "large.bin"
    )
    assert not archived_raw.exists()
    assert controller.audit()["passed"] is True


def test_retention_binding_rejects_archive_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    plan = _start(controller, tmp_path, label)
    _seal_and_review(
        controller,
        tmp_path,
        label,
        reviewer_status=ReviewerStatus.ACCEPTED,
    )
    controller.classify(
        label,
        classification_context_path=_classification_context(
            controller, tmp_path, label, publication_state="current"
        ),
    )
    inventory_path = _content_inventory(plan.run_dir, label)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    source_bytes = sum(int(item["byte_count"]) for item in inventory["files"])
    performance = {
        "hashed_bytes": source_bytes,
        "source_bytes": source_bytes,
        "scan_passes": 1,
        "delete_traversals": 0,
        "duplicate_hashed_bytes": 0,
        "backend_calibrated": True,
        "implicit_fallback": False,
        "throughput_mib_per_second": 1.0,
        "cpu_utilization_percent": 1.0,
        "peak_memory_bytes": 1024,
        "io_utilization_percent": 1.0,
        "close_wall_seconds": 1.0,
        "backend": "windows_native",
        "workers": 1,
    }
    receipt = _retention_binding(
        controller,
        label,
        run_dir=plan.run_dir,
        inventory=inventory,
        performance=performance,
    )
    archive_artifact = (
        controller.archive_root
        / "stage00"
        / "runs"
        / label
        / "generation-0001"
        / "control"
        / "raw-evidence.txt"
    )
    controller.mark_retained(
        label,
        retention_receipt_path=receipt,
        content_inventory_path=inventory_path,
    )
    original_verify = lifecycle_module._verify_tree_content

    def mutate_after_hash(
        root: Path,
        expected: tuple[lifecycle_module.FileIdentity, ...],
    ) -> tuple[str, int, int, dict[str, tuple[int, int]]]:
        result = original_verify(root, expected)
        archive_artifact.write_bytes(b"X" * len(b"raw-evidence"))
        return result

    monkeypatch.setattr(lifecycle_module, "_verify_tree_content", mutate_after_hash)

    with pytest.raises(LifecycleError, match="changed after close-time hashing"):
        controller.close(
            label,
            performance=ClosePerformance.from_dict(performance),
            storage_reconciliation_path=_storage_reconciliation(
                controller, tmp_path, label
            ),
        )


def test_unknown_failure_blocks_retention_and_the_next_experiment(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    _start(controller, tmp_path, label)
    _seal_and_review(
        controller,
        tmp_path,
        label,
        reviewer_status=ReviewerStatus.FAILED_UNKNOWN,
        failure_code="runner_failure",
    )

    decision = controller.classify(
        label,
        classification_context_path=_classification_context(
            controller, tmp_path, label, publication_state="none"
        ),
    )

    assert decision.blocked
    assert decision.retention_class == RetentionClassV3.UNKNOWN_FULL
    assert controller.records()[0].state == LifecycleState.BLOCKED_RETENTION
    with pytest.raises(LifecycleError, match="not CLOSED"):
        controller.plan(_plan(tmp_path, "stage01_objective_attempt01"))


def test_signed_adjudication_unblocks_unknown_failure(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    _start(controller, tmp_path, label)
    _seal_and_review(
        controller,
        tmp_path,
        label,
        reviewer_status=ReviewerStatus.FAILED_UNKNOWN,
        failure_code="runner_failure",
    )
    controller.classify(
        label,
        classification_context_path=_classification_context(
            controller, tmp_path, label, publication_state="none"
        ),
    )
    record_path = controller._record_path(label)
    adjudication = tmp_path / "unknown-adjudication.json"
    _write_signed_json(
        adjudication,
        {
            "run_label": label,
            "review_manifest_sha256": next(
                item for item in controller.records() if item.run_label == label
            ).review_manifest_sha256,
            "root_cause_id": "manifest-integrity-v1",
            "canonical_representative_run_label": label,
            "canonical_representative_record_sha256": hashlib.sha256(
                record_path.read_bytes()
            ).hexdigest(),
            "failure_code": "runner_failure",
            "failing_component": "runner",
            "invariant_or_check": "manifest_integrity",
            "failure_location": "control/manifest.json",
        },
    )

    adjudicated = controller.adjudicate(
        label,
        root_cause_id="manifest-integrity-v1",
        canonical_representative=label,
        adjudication_path=adjudication,
    )
    decision = controller.classify(
        label,
        classification_context_path=_classification_context(
            controller, tmp_path, label, publication_state="none"
        ),
    )

    assert adjudicated.reviewer_status == ReviewerStatus.FAILED_KNOWN
    assert decision.retention_class == RetentionClassV3.UNIQUE_FAILURE_CAPSULE
    assert controller.records()[0].state == LifecycleState.CLASSIFIED


def test_adjudication_rejects_a_running_record(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    _start(controller, tmp_path, label)

    with pytest.raises(LifecycleError, match="requires a reviewed lifecycle record"):
        controller.adjudicate(
            label,
            root_cause_id="premature",
            canonical_representative=label,
            adjudication_path=tmp_path / "not-read.json",
        )


def test_unique_failure_requires_exact_signed_root_cause_identity(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    _start(controller, tmp_path, label)
    _seal_and_review(
        controller,
        tmp_path,
        label,
        reviewer_status=ReviewerStatus.FAILED_KNOWN,
        failure_code="runner_failure",
    )
    adjudication = tmp_path / "adjudication.json"
    _write_signed_json(
        adjudication,
        {
            "run_label": label,
            "review_manifest_sha256": next(
                item for item in controller.records() if item.run_label == label
            ).review_manifest_sha256,
            "root_cause_id": "manifest-integrity-v1",
            "canonical_representative_run_label": label,
            "canonical_representative_record_sha256": (
                hashlib.sha256(controller._record_path(label).read_bytes()).hexdigest()
            ),
            "failure_code": "runner_failure",
            "failing_component": "runner",
            "invariant_or_check": "manifest_integrity",
            "failure_location": "control/manifest.json",
        },
    )
    controller.adjudicate(
        label,
        root_cause_id="manifest-integrity-v1",
        canonical_representative=label,
        adjudication_path=adjudication,
    )

    decision = controller.classify(
        label,
        classification_context_path=_classification_context(
            controller, tmp_path, label, publication_state="none"
        ),
    )

    assert decision.retention_class == RetentionClassV3.UNIQUE_FAILURE_CAPSULE


def test_compaction_plan_is_deterministic_and_apply_rejects_drift(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    plan = _start(controller, tmp_path, label)
    control = plan.run_dir / "control"
    control.mkdir(exist_ok=True)
    raw = plan.run_dir / "instance" / "events.parquet"
    raw.parent.mkdir()
    raw.write_bytes(b"raw-event-bytes")
    raw_stat = raw.stat()
    _write_signed_json(
        control / "manifest.json",
        {
            "run_label": label,
            "status": "complete",
            "evidence_completeness": "complete",
            "artifacts": [
                {
                    "relative_path": raw.relative_to(plan.run_dir).as_posix(),
                    "byte_size": raw.stat().st_size,
                    "checksum": hashlib.sha256(raw.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    review = plan.run_dir / "review" / "review.json"
    review.parent.mkdir()
    _write_signed_json(
        review, {"run_label": label, "status": "ACCEPTED", "files": {}}
    )
    controller.seal(label, manifest_path=control / "manifest.json")
    _review_execution(controller, label, review)
    controller.review(
        label,
        review_manifest_path=review,
    )
    controller.classify(
        label,
        classification_context_path=_classification_context(
            controller, tmp_path, label, publication_state="superseded"
        ),
    )
    compaction, _path = controller.prepare_compaction(
        label,
        run_dir=plan.run_dir,
        content_inventory_path=_content_inventory(plan.run_dir, label),
    )
    assert _path == (
        controller.state_root
        / "compaction"
        / "plans"
        / label
        / f"{compaction.plan_sha256}.json"
    )
    assert raw.relative_to(plan.run_dir).as_posix() in {
        item.relative_path for item in compaction.delete
    }
    added = plan.run_dir / "unexpected.tmp"
    added.write_bytes(b"late-writer")
    with pytest.raises(LifecycleError, match="file set differs"):
        controller.apply_compaction(
            compaction,
            expected_plan_sha256=compaction.plan_sha256,
            writer_is_active=lambda _path: False,
        )
    added.unlink()
    raw.write_bytes(b"tampered-bytes!")
    os.utime(raw, ns=(raw_stat.st_atime_ns, raw_stat.st_mtime_ns))
    with pytest.raises(LifecycleError, match="source drift"):
        controller.apply_compaction(
            compaction,
            expected_plan_sha256=compaction.plan_sha256,
            writer_is_active=lambda _path: False,
        )


def test_compaction_apply_deletes_only_controller_plan(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    label = "stage00_baseline_attempt01"
    plan = _start(controller, tmp_path, label)
    control = plan.run_dir / "control"
    control.mkdir(exist_ok=True)
    raw = plan.run_dir / "instance" / "events.parquet"
    raw.parent.mkdir()
    raw.write_bytes(b"raw-event-bytes")
    _write_signed_json(
        control / "manifest.json",
        {
            "run_label": label,
            "status": "complete",
            "evidence_completeness": "complete",
            "artifacts": [
                {
                    "relative_path": raw.relative_to(plan.run_dir).as_posix(),
                    "byte_size": raw.stat().st_size,
                    "checksum": hashlib.sha256(raw.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    review = plan.run_dir / "review" / "review.json"
    review.parent.mkdir()
    _write_signed_json(
        review, {"run_label": label, "status": "ACCEPTED", "files": {}}
    )
    controller.seal(label, manifest_path=control / "manifest.json")
    _review_execution(controller, label, review)
    controller.review(
        label,
        review_manifest_path=review,
    )
    controller.classify(
        label,
        classification_context_path=_classification_context(
            controller, tmp_path, label, publication_state="superseded"
        ),
    )
    compaction, _path = controller.prepare_compaction(
        label,
        run_dir=plan.run_dir,
        content_inventory_path=_content_inventory(plan.run_dir, label),
    )

    receipt = controller.apply_compaction(
        compaction,
        expected_plan_sha256=compaction.plan_sha256,
        writer_is_active=lambda _path: False,
    )

    assert not raw.exists()
    assert (control / "manifest.json").is_file()
    assert receipt.deleted_file_count == 1
    assert receipt.released_bytes == len(b"raw-event-bytes")
    assert controller.records()[0].state == LifecycleState.COMPACTED
    (controller.state_root / "compaction/ledger.csv").unlink()
    (controller.state_root / "compaction/deleted_files.csv").unlink()

    replayed = controller.apply_compaction(
        compaction,
        expected_plan_sha256=compaction.plan_sha256,
        writer_is_active=lambda _path: False,
    )

    assert replayed.receipt_sha256 == receipt.receipt_sha256
    deleted_rows = (
        controller.state_root / "compaction/deleted_files.csv"
    ).read_text(encoding="utf-8").splitlines()
    assert len(deleted_rows) == 2
    source_bytes = sum(
        item.byte_count for item in (*compaction.keep, *compaction.delete)
    )
    closed = controller.close(
        label,
        performance=ClosePerformance(
            hashed_bytes=compaction.hashed_bytes,
            source_bytes=source_bytes,
            scan_passes=compaction.scan_passes,
            delete_traversals=receipt.delete_traversals,
            duplicate_hashed_bytes=0,
            backend_calibrated=True,
            implicit_fallback=False,
            throughput_mib_per_second=1.0,
            cpu_utilization_percent=1.0,
            peak_memory_bytes=1024,
            io_utilization_percent=1.0,
            close_wall_seconds=1.0,
            backend=compaction.io_backend,
            workers=compaction.io_workers,
        ),
        storage_reconciliation_path=_storage_reconciliation(
            controller, tmp_path, label
        ),
    )
    assert closed.state == LifecycleState.CLOSED


def test_ci_audit_accepts_the_catalogued_runner_surface() -> None:
    audit_runner_entrypoints(ROOT, _catalog())
