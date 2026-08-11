from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import evrptw.experiment_lifecycle as lifecycle_module
import evrptw.stage052_retention as legacy_retention
import evrptw.storage_governance as storage_governance_module
from evrptw.stage052_campaign import (
    StorageRoot,
    StorageRootLocator,
    VolumeIdentity,
)
from evrptw.stage052_retention import RetentionRecord, write_retention_registry
from evrptw.storage_governance import (
    GIB,
    AdjudicationRecord,
    ExpectedSegmentTree,
    ExperimentStorageGovernance,
    GovernancePolicy,
    RebuildableAsset,
    RebuildableKind,
    RetentionClass,
    RetentionRequest,
    RetentionSegment,
    StartRequest,
    StorageCapacityError,
    StorageGovernanceError,
    SweepRequest,
    compute_tree_identity,
    compute_tree_sha256,
    preflight_cli_attempt,
    verify_storage_migration_attestation,
    write_adjudication_record,
    write_cleanup_confirmation_receipt,
    write_lifecycle_adjudication_record,
    write_migration_dry_run,
    write_retention_replay_receipt,
    write_storage_migration_attestation,
)


def _locator(tmp_path: Path) -> StorageRootLocator:
    return StorageRootLocator(
        {
            "wsl_staging": StorageRoot(
                "wsl_staging",
                tmp_path / "staging",
                VolumeIdentity("wsl-ext4", "ext4"),
            ),
            "d_host": StorageRoot(
                "d_host",
                tmp_path / "d-host",
                VolumeIdentity("d-nvme", "ntfs"),
            ),
            "e_archive": StorageRoot(
                "e_archive",
                tmp_path / "e-archive",
                VolumeIdentity("e-usb", "ntfs"),
            ),
        }
    )


def _replay_writer(
    tmp_path: Path,
    *,
    run_label: str,
    generation: int,
) -> Callable[[Path], Path]:
    def verify(archive_path: Path) -> Path:
        receipt = tmp_path / "replays" / f"{run_label}-{generation:04d}.json"
        write_retention_replay_receipt(
            receipt,
            run_label=run_label,
            generation=generation,
            archive_path=archive_path,
            verifier_identity_sha256="a" * 64,
            validator_replay_passed=True,
            objective_replay_passed=True,
            raw_review_replay_passed=True,
        )
        return receipt

    return verify


def test_repository_governance_policy_declares_balanced_reserves() -> None:
    policy = GovernancePolicy.from_toml(
        Path("configs/experiment_storage_governance.toml")
    )

    assert policy.archive_reserve_bytes == 0
    assert policy.host_reserve_bytes == 0
    assert policy.staging_safety_reserve_bytes == 50 * GIB
    assert policy.stage052_active_workspace_floor_bytes == 32 * GIB
    assert policy.maintenance_allowlist == (
        (".cache", "cache"),
        (".venv", "venv"),
        ("build", "build"),
        (".temporary-spool", "temporary_spool"),
    )


@pytest.mark.parametrize(
    "runner",
    (
        "stage00_baseline.py",
        "stage01_objective.py",
        "stage02_route_reduction.py",
        "stage02_route_quality.py",
        "stage02_constraint_guided.py",
        "stage03_measurement.py",
        "stage031_cheap_screening.py",
        "stage032_cache_incremental.py",
        "stage033_exact_deadline.py",
        "stage034_control_parallel.py",
        "stage04_weights.py",
        "stage051_best_known.py",
        "stage052_performance.py",
    ),
)
def test_every_stage_producer_cli_uses_shared_stop_gate(runner: str) -> None:
    source = (
        Path("src") / "evrptw" / "experiments" / runner
    ).read_text(encoding="utf-8")
    assert "preflight_cli_attempt(" in source


def test_missing_experiment_plan_cap_persists_rejection_without_run_directory(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "stage00_missing_plan.toml"
    config.write_text('[run]\nseeds = [2014]\n', encoding="utf-8")
    output = tmp_path / "results" / "stage00_baseline_attempt98"

    with pytest.raises(StorageGovernanceError, match="hard cap"):
        preflight_cli_attempt(config_path=config, output_dir=output)

    assert not output.exists()
    observations = tuple(
        (tmp_path / ".storage-governance" / "observations").glob(
            "plan-rejection-*.json"
        )
    )
    assert len(observations) == 1
    payload = json.loads(observations[0].read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "hard cap" in payload["plan_error"]
    assert observations[0].with_suffix(".json.sha256").is_file()


def test_cli_lifecycle_caps_keep_batch_and_run_budgets_distinct() -> None:
    assert storage_governance_module._artifact_storage_lifecycle_caps(
        (
            {
                "per_instance_seed_max_bytes": 1 * GIB,
                "per_run_max_bytes": 16 * GIB,
            },
        )
    ) == (1 * GIB, 16 * GIB)
    assert storage_governance_module._artifact_storage_lifecycle_caps(
        (
            {
                "per_batch_max_bytes": 2 * GIB,
                "per_instance_seed_max_bytes": 1 * GIB,
                "per_run_max_bytes": 16 * GIB,
            },
        )
    ) == (2 * GIB, 16 * GIB)


def test_cli_lifecycle_caps_reject_batch_above_run_budget() -> None:
    with pytest.raises(ValueError, match="batch hard cap is invalid"):
        storage_governance_module._artifact_storage_lifecycle_caps(
            (
                {
                    "per_instance_seed_max_bytes": 17 * GIB,
                    "per_run_max_bytes": 16 * GIB,
                },
            )
        )


def test_preflight_capacity_failure_is_observable_without_creating_run(
    tmp_path: Path,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    run_dir = locator.resolve("wsl_staging").absolute_path / "stage05.2_benchmark_attempt01"
    free = {
        locator.resolve("wsl_staging").absolute_path: 200 * GIB,
        locator.resolve("d_host").absolute_path: 500 * GIB,
        locator.resolve("e_archive").absolute_path: 50 * GIB,
    }
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda path: free[path],
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )

    with pytest.raises(StorageCapacityError) as raised:
        governance.preflight_run(
            StartRequest(
                stage_id="stage05.2",
                run_label=run_dir.name,
                run_dir=run_dir,
                staging_root_alias="wsl_staging",
                host_root_alias="d_host",
                archive_root_alias="e_archive",
                planned_archive_bytes=100 * GIB,
                max_active_workspace_bytes=32 * GIB,
                projected_host_growth_bytes=32 * GIB,
                stage_plan_sha256="1" * 64,
            )
        )

    assert not run_dir.exists()
    assert raised.value.observation["passed"] is False
    receipts = tuple((tmp_path / "state" / "observations").glob("*.json"))
    assert len(receipts) == 1
    persisted = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert persisted["deficits_by_alias"] == {"e_archive": 50 * GIB}
    assert receipts[0].with_suffix(".json.sha256").is_file()


def test_preflight_rejects_d_role_as_new_archive(tmp_path: Path) -> None:
    base = _locator(tmp_path)
    locator = StorageRootLocator(
        {
            **{alias: base.resolve(alias) for alias in base.aliases},
            "d_archive": StorageRoot(
                "d_archive",
                tmp_path / "d-archive",
                VolumeIdentity("d-nvme", "ntfs"),
            ),
        }
    )
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )
    run_dir = locator.resolve("wsl_staging").absolute_path / "stage05.2_benchmark_attempt01"

    with pytest.raises(StorageCapacityError, match="new experiment archives"):
        governance.preflight_run(
            StartRequest(
                stage_id="stage05.2",
                run_label=run_dir.name,
                run_dir=run_dir,
                staging_root_alias="wsl_staging",
                host_root_alias="d_host",
                archive_root_alias="d_archive",
                planned_archive_bytes=1 * GIB,
                max_active_workspace_bytes=32 * GIB,
                projected_host_growth_bytes=32 * GIB,
                stage_plan_sha256="1" * 64,
            )
        )

    assert not run_dir.exists()


def test_preflight_reserves_capacity_across_concurrent_run_labels(tmp_path: Path) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    free = {
        locator.resolve("wsl_staging").absolute_path: 200 * GIB,
        locator.resolve("d_host").absolute_path: 500 * GIB,
        locator.resolve("e_archive").absolute_path: 150 * GIB,
    }
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda path: free[path],
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )

    first = governance.preflight_run(
        StartRequest(
            stage_id="stage05.2",
            run_label="stage05.2_benchmark_attempt01",
            run_dir=(
                locator.resolve("wsl_staging").absolute_path
                / "stage05.2_benchmark_attempt01"
            ),
            staging_root_alias="wsl_staging",
            host_root_alias="d_host",
            archive_root_alias="e_archive",
            planned_archive_bytes=100 * GIB,
            max_active_workspace_bytes=32 * GIB,
            projected_host_growth_bytes=32 * GIB,
            stage_plan_sha256="1" * 64,
        )
    )

    assert first.permit_path.is_file()
    with pytest.raises(StorageCapacityError) as raised:
        governance.preflight_run(
            StartRequest(
                stage_id="stage05.2",
                run_label="stage05.2_benchmark_attempt02",
                run_dir=(
                    locator.resolve("wsl_staging").absolute_path
                    / "stage05.2_benchmark_attempt02"
                ),
                staging_root_alias="wsl_staging",
                host_root_alias="d_host",
                archive_root_alias="e_archive",
                planned_archive_bytes=100 * GIB,
                max_active_workspace_bytes=32 * GIB,
                projected_host_growth_bytes=32 * GIB,
                stage_plan_sha256="2" * 64,
            )
        )

    assert raised.value.observation["deficits_by_alias"] == {
        "e_archive": 50 * GIB
    }
    assert not (
        locator.resolve("wsl_staging").absolute_path
        / "stage05.2_benchmark_attempt02"
    ).exists()
    reconciliation = governance.reconcile_permit(
        first.run_label,
        outcome="aborted_audited",
        evidence_sha256="e" * 64,
    )
    assert reconciliation.is_file()
    third = StartRequest(
        stage_id="stage05.2",
        run_label="stage05.2_benchmark_attempt03",
        run_dir=(
            locator.resolve("wsl_staging").absolute_path
            / "stage05.2_benchmark_attempt03"
        ),
        staging_root_alias="wsl_staging",
        host_root_alias="d_host",
        archive_root_alias="e_archive",
        planned_archive_bytes=100 * GIB,
        max_active_workspace_bytes=32 * GIB,
        projected_host_growth_bytes=32 * GIB,
        stage_plan_sha256="3" * 64,
    )
    assert governance.preflight_run(third).run_label == third.run_label


def test_preflight_allows_only_monotonic_remaining_archive_projection(
    tmp_path: Path,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )
    request = StartRequest(
        stage_id="stage05.2",
        run_label="stage05.2_benchmark_attempt97",
        run_dir=(
            locator.resolve("wsl_staging").absolute_path
            / "stage05.2_benchmark_attempt97"
        ),
        staging_root_alias="wsl_staging",
        host_root_alias="d_host",
        archive_root_alias="e_archive",
        planned_archive_bytes=100 * GIB,
        max_active_workspace_bytes=32 * GIB,
        projected_host_growth_bytes=32 * GIB,
        stage_plan_sha256="7" * 64,
    )

    initial_permit = governance.preflight_run(request)
    initial_permit_sha256 = hashlib.sha256(
        initial_permit.permit_path.read_bytes()
    ).hexdigest()
    initial_permit_payload = json.loads(
        initial_permit.permit_path.read_text(encoding="utf-8")
    )
    final_permit = governance.preflight_run(
        replace(request, planned_archive_bytes=0)
    )
    final_observation = json.loads(
        final_permit.observation_path.read_text(encoding="utf-8")
    )
    assert final_observation["required_bytes_by_alias"]["e_archive"] == 0
    assert final_permit.permit_path == initial_permit.permit_path
    assert (
        hashlib.sha256(final_permit.permit_path.read_bytes()).hexdigest()
        == initial_permit_sha256
    )
    assert (
        json.loads(final_permit.permit_path.read_text(encoding="utf-8"))
        == initial_permit_payload
    )
    ledger = json.loads(
        (tmp_path / "state" / "capacity_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["reservations"][request.run_label]["planned_archive_bytes"] == 0
    reconciliation = governance.reconcile_permit(
        request.run_label,
        outcome="retained",
        evidence_sha256="8" * 64,
    )
    reconciliation_payload = json.loads(reconciliation.read_text(encoding="utf-8"))
    assert reconciliation_payload["storage_permit_sha256"] == initial_permit_sha256
    ledger_path = tmp_path / "state" / "capacity_ledger.json"
    ledger_before = ledger_path.read_bytes()
    reconciliation_before = reconciliation.read_bytes()
    reconciliation_sidecar_before = reconciliation.with_suffix(
        reconciliation.suffix + ".sha256"
    ).read_bytes()
    successor = governance.write_permit_reconciliation_successor(
        request.run_label,
        evidence_sha256="9" * 64,
    )
    successor_before = successor.read_bytes()
    successor_payload = json.loads(successor_before)
    assert successor_payload == {
        "created_at_utc": successor_payload["created_at_utc"],
        "outcome": "retained",
        "predecessor_evidence_sha256": "8" * 64,
        "predecessor_reconciliation_sha256": hashlib.sha256(
            reconciliation_before
        ).hexdigest(),
        "reason": "retention_completed_after_capacity_release",
        "run_label": request.run_label,
        "schema_version": "experiment-capacity-reconciliation-successor-v1",
        "storage_permit_sha256": initial_permit_sha256,
        "successor_evidence_sha256": "9" * 64,
    }
    assert governance.write_permit_reconciliation_successor(
        request.run_label,
        evidence_sha256="9" * 64,
    ) == successor
    assert governance.reconcile_permit(
        request.run_label,
        outcome="retained",
        evidence_sha256="8" * 64,
    ) == reconciliation
    assert successor.read_bytes() == successor_before
    assert ledger_path.read_bytes() == ledger_before
    assert reconciliation.read_bytes() == reconciliation_before
    assert (
        reconciliation.with_suffix(reconciliation.suffix + ".sha256").read_bytes()
        == reconciliation_sidecar_before
    )
    with pytest.raises(
        StorageGovernanceError,
        match="successor identity differs",
    ):
        governance.write_permit_reconciliation_successor(
            request.run_label,
            evidence_sha256="a" * 64,
        )
    partial_failure_record: dict[str, object] = {
        "schema_version": "experiment-lifecycle-v3",
        "run_label": request.run_label,
        "state": "RETAINED",
        "transition_ordinal": 7,
    }
    partial_failure_stable: dict[str, object] = {
        "schema_version": "experiment-lifecycle-close-failure-v1",
        "run_label": request.run_label,
        "failure_check": "storage_reconciliation_validation",
        "record_sha256": hashlib.sha256(
            storage_governance_module._lifecycle_record_bytes(
                partial_failure_record
            )
        ).hexdigest(),
        "record": partial_failure_record,
        "record_state": "RETAINED",
        "record_transition_ordinal": 7,
        "transition_event_sha256": "d" * 64,
        "storage_reconciliation_kind": "invalid_successor",
        "storage_reconciliation_sha256": hashlib.sha256(
            successor_before
        ).hexdigest(),
        "disposition_sha256": "a" * 64,
        "error_type": "LifecycleError",
        "error_message": (
            "storage permit reconciliation successor chain differs"
        ),
    }
    partial_failure_identity = hashlib.sha256(
        storage_governance_module._lifecycle_canonical_json(
            partial_failure_stable
        )
    ).hexdigest()
    partial_failed_close_receipt = (
        locator.resolve("e_archive").absolute_path
        / ".experiment-lifecycle"
        / "close"
        / "failures"
        / request.run_label
        / f"{partial_failure_identity}.json"
    )
    storage_governance_module._write_signed_json(
        partial_failed_close_receipt,
        {
            **partial_failure_stable,
            "failure_identity_sha256": partial_failure_identity,
            "created_at_utc": "2026-08-10T00:00:00+00:00",
        },
    )
    correction_path = (
        tmp_path
        / "state"
        / "permit_reconciliation_successor_corrections"
        / f"{request.run_label}.json"
    )
    with pytest.raises(
        StorageGovernanceError,
        match="failure record is invalid",
    ):
        governance.write_permit_reconciliation_successor_correction(
            request.run_label,
            evidence_sha256="a" * 64,
            failed_close_receipt_path=partial_failed_close_receipt,
        )
    assert not correction_path.exists()
    assert not correction_path.with_suffix(
        correction_path.suffix + ".sha256"
    ).exists()

    retention_performance = {
        "hashed_bytes": 1,
        "source_bytes": 1,
        "scan_passes": 1,
        "delete_traversals": 0,
        "duplicate_hashed_bytes": 0,
        "backend_calibrated": True,
        "implicit_fallback": False,
        "throughput_mib_per_second": 1.0,
        "cpu_utilization_percent": 10.0,
        "peak_memory_bytes": 1_024,
        "io_utilization_percent": 10.0,
        "close_wall_seconds": 1.0,
        "backend": "test",
        "workers": 1,
    }
    retention_path = (
        locator.resolve("e_archive").absolute_path
        / ".experiment-lifecycle"
        / "retention"
        / "receipts"
        / f"{request.run_label}.json"
    )
    storage_governance_module._write_signed_json(
        retention_path,
        {"close_performance": retention_performance},
    )
    retention_sha256 = hashlib.sha256(retention_path.read_bytes()).hexdigest()
    failure_record: dict[str, object] = {
        "schema_version": "experiment-lifecycle-v3",
        "run_label": request.run_label,
        "experiment_id": "stage052_performance",
        "state": "RETAINED",
        "plan_sha256": "7" * 64,
        "catalog_sha256": "2" * 64,
        "transition_ordinal": 7,
        "created_at_utc": "2026-08-10T00:00:00+00:00",
        "updated_at_utc": "2026-08-10T00:00:01+00:00",
        "storage_permit_sha256": initial_permit_sha256,
        "sealed_manifest_sha256": "3" * 64,
        "sealed_manifest_relative_path": "control/manifest.json",
        "reviewer_status": "FAILED_KNOWN",
        "review_manifest_sha256": "4" * 64,
        "failure_code": "invalid_manifest",
        "failure_component": "stage052_calibration",
        "failure_check": "resource_contract_not_bound_at_seal",
        "failure_location": "control/manifest.json",
        "root_cause_id": "unbound-producer-resource-contract-v1",
        "adjudication_sha256": "5" * 64,
        "canonical_representative": request.run_label,
        "retention_class": "unique_failure_full",
        "retention_receipt_sha256": retention_sha256,
        "content_inventory_sha256": "7" * 64,
        "compaction_plan_sha256": "",
        "compaction_receipt_sha256": "",
        "superseded_by": "",
        "blocked_reason": "",
    }
    current_record_path = (
        locator.resolve("e_archive").absolute_path
        / ".experiment-lifecycle"
        / "runs"
        / f"{request.run_label}.json"
    )
    lifecycle_module._write_signed_json(
        current_record_path,
        failure_record,
    )
    failure_record_sha256 = hashlib.sha256(
        storage_governance_module._lifecycle_record_bytes(failure_record)
    ).hexdigest()
    transition_path = (
        locator.resolve("e_archive").absolute_path
        / ".experiment-lifecycle"
        / "events"
        / request.run_label
        / "0007-RETAINED.json"
    )
    storage_governance_module._write_signed_json(
        transition_path,
        {
            **failure_record,
            "record_sha256": failure_record_sha256,
        },
    )
    failure_stable_payload: dict[str, object] = {
        "schema_version": "experiment-lifecycle-close-failure-v1",
        "run_label": request.run_label,
        "failure_check": "storage_reconciliation_validation",
        "record_sha256": failure_record_sha256,
        "record": failure_record,
        "record_state": "RETAINED",
        "record_transition_ordinal": 7,
        "transition_event_sha256": hashlib.sha256(
            transition_path.read_bytes()
        ).hexdigest(),
        "storage_reconciliation_kind": "invalid_successor",
        "storage_reconciliation_sha256": hashlib.sha256(
            successor_before
        ).hexdigest(),
        "disposition_sha256": retention_sha256,
        "error_type": "LifecycleError",
        "error_message": (
            "storage permit reconciliation successor chain differs"
        ),
    }
    failure_identity = hashlib.sha256(
        storage_governance_module._lifecycle_canonical_json(
            failure_stable_payload
        )
    ).hexdigest()
    failed_close_receipt = (
        locator.resolve("e_archive").absolute_path
        / ".experiment-lifecycle"
        / "close"
        / "failures"
        / request.run_label
        / f"{failure_identity}.json"
    )
    storage_governance_module._write_signed_json(
        failed_close_receipt,
        {
            **failure_stable_payload,
            "failure_identity_sha256": failure_identity,
            "created_at_utc": "2026-08-10T00:00:00+00:00",
        },
    )
    wrong_current_record = {
        **failure_record,
        "retention_receipt_sha256": "6" * 64,
    }
    wrong_current_stable = {
        **failure_stable_payload,
        "record_sha256": hashlib.sha256(
            storage_governance_module._lifecycle_record_bytes(
                wrong_current_record
            )
        ).hexdigest(),
        "record": wrong_current_record,
    }
    wrong_current_identity = hashlib.sha256(
        storage_governance_module._lifecycle_canonical_json(
            wrong_current_stable
        )
    ).hexdigest()
    wrong_current_receipt = failed_close_receipt.with_name(
        f"{wrong_current_identity}.json"
    )
    storage_governance_module._write_signed_json(
        wrong_current_receipt,
        {
            **wrong_current_stable,
            "failure_identity_sha256": wrong_current_identity,
            "created_at_utc": "2026-08-10T00:00:00+00:00",
        },
    )
    with pytest.raises(
        StorageGovernanceError,
        match="cannot be corrected",
    ):
        governance.write_permit_reconciliation_successor_correction(
            request.run_label,
            evidence_sha256=retention_sha256,
            failed_close_receipt_path=wrong_current_receipt,
        )
    assert not correction_path.exists()
    assert not correction_path.with_suffix(
        correction_path.suffix + ".sha256"
    ).exists()
    malformed_successor = dict(successor_payload)
    malformed_successor["unexpected"] = True
    storage_governance_module._write_signed_json(successor, malformed_successor)
    with pytest.raises(
        StorageGovernanceError,
        match="cannot be corrected",
    ):
        governance.write_permit_reconciliation_successor_correction(
            request.run_label,
            evidence_sha256=retention_sha256,
            failed_close_receipt_path=failed_close_receipt,
        )
    storage_governance_module._write_signed_json(successor, successor_payload)
    correction = governance.write_permit_reconciliation_successor_correction(
        request.run_label,
        evidence_sha256=retention_sha256,
        failed_close_receipt_path=failed_close_receipt,
    )
    correction_before = correction.read_bytes()
    correction_payload = json.loads(correction_before)
    assert correction_payload == {
        "created_at_utc": correction_payload["created_at_utc"],
        "invalid_successor_evidence_sha256": "9" * 64,
        "invalid_successor_reconciliation_sha256": hashlib.sha256(
            successor_before
        ).hexdigest(),
        "failed_close_receipt_identity_sha256": failure_identity,
        "failed_close_receipt_sha256": hashlib.sha256(
            failed_close_receipt.read_bytes()
        ).hexdigest(),
        "outcome": "retained",
        "predecessor_evidence_sha256": "8" * 64,
        "predecessor_reconciliation_sha256": hashlib.sha256(
            reconciliation_before
        ).hexdigest(),
        "reason": "corrected_successor_evidence_after_failed_close",
        "run_label": request.run_label,
        "schema_version": (
            "experiment-capacity-reconciliation-successor-correction-v1"
        ),
        "storage_permit_sha256": initial_permit_sha256,
        "successor_evidence_sha256": retention_sha256,
    }
    assert governance.write_permit_reconciliation_successor_correction(
        request.run_label,
        evidence_sha256=retention_sha256,
        failed_close_receipt_path=failed_close_receipt,
    ) == correction
    closed_record = {
        **failure_record,
        "state": "CLOSED",
        "transition_ordinal": 8,
        "updated_at_utc": "2026-08-10T00:00:02+00:00",
    }
    lifecycle_module._write_signed_json(current_record_path, closed_record)
    closed_transition_path = transition_path.with_name("0008-CLOSED.json")
    lifecycle_module._write_signed_json(
        closed_transition_path,
        {
            **closed_record,
            "record_sha256": hashlib.sha256(
                storage_governance_module._lifecycle_record_bytes(
                    closed_record
                )
            ).hexdigest(),
        },
    )
    close_path = (
        current_record_path.parents[1]
        / "close"
        / f"{request.run_label}.json"
    )
    close_payload = {
        "schema_version": "experiment-lifecycle-v3",
        "run_label": request.run_label,
        "status": "CLOSED",
        "record": closed_record,
        "storage_reconciliation_sha256": hashlib.sha256(
            correction.read_bytes()
        ).hexdigest(),
        "performance": retention_performance,
    }
    lifecycle_module._write_signed_json(close_path, close_payload)
    assert governance.write_permit_reconciliation_successor_correction(
        request.run_label,
        evidence_sha256=retention_sha256,
        failed_close_receipt_path=failed_close_receipt,
    ) == correction
    assert correction.read_bytes() == correction_before
    assert successor.read_bytes() == successor_before
    assert ledger_path.read_bytes() == ledger_before
    assert reconciliation.read_bytes() == reconciliation_before
    lifecycle_module._write_signed_json(
        close_path,
        {**close_payload, "performance": {"backend": "test"}},
    )
    with pytest.raises(
        StorageGovernanceError,
        match="cannot be corrected",
    ):
        governance.write_permit_reconciliation_successor_correction(
            request.run_label,
            evidence_sha256=retention_sha256,
            failed_close_receipt_path=failed_close_receipt,
        )
    lifecycle_module._write_signed_json(close_path, close_payload)
    drifted_performance = dict(retention_performance)
    drifted_performance["throughput_mib_per_second"] = 2.0
    lifecycle_module._write_signed_json(
        close_path,
        {**close_payload, "performance": drifted_performance},
    )
    with pytest.raises(
        StorageGovernanceError,
        match="cannot be corrected",
    ):
        governance.write_permit_reconciliation_successor_correction(
            request.run_label,
            evidence_sha256=retention_sha256,
            failed_close_receipt_path=failed_close_receipt,
        )
    lifecycle_module._write_signed_json(close_path, close_payload)
    with pytest.raises(
        StorageGovernanceError,
        match="cannot be corrected",
    ):
        governance.write_permit_reconciliation_successor_correction(
            request.run_label,
            evidence_sha256="b" * 64,
            failed_close_receipt_path=failed_close_receipt,
        )
    with pytest.raises(StorageCapacityError, match="conflicts"):
        governance.preflight_run(
            replace(request, planned_archive_bytes=1 * GIB)
        )


def test_preflight_maintenance_audit_discovers_exact_allowlist_assets(
    tmp_path: Path,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    cache = locator.resolve("wsl_staging").absolute_path / ".cache"
    cache.mkdir()
    (cache / "data.bin").write_bytes(b"cache")
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )
    permit = governance.preflight_run(
        StartRequest(
            stage_id="stage05.2",
            run_label="stage05.2_benchmark_attempt96",
            run_dir=(
                locator.resolve("wsl_staging").absolute_path
                / "stage05.2_benchmark_attempt96"
            ),
            staging_root_alias="wsl_staging",
            host_root_alias="d_host",
            archive_root_alias="e_archive",
            planned_archive_bytes=1 * GIB,
            max_active_workspace_bytes=32 * GIB,
            projected_host_growth_bytes=32 * GIB,
            stage_plan_sha256="8" * 64,
        )
    )

    audit = json.loads(
        permit.maintenance_audit_path.read_text(encoding="utf-8")
    )
    assert audit["candidate_paths"] == []
    assert audit["retained"] == [
        {
            "path": str(cache.resolve()),
            "reason": "retention_period",
        }
    ]


def test_existing_permit_remeasures_capacity_before_batch_dispatch(
    tmp_path: Path,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    free = {
        locator.resolve("wsl_staging").absolute_path: 200 * GIB,
        locator.resolve("d_host").absolute_path: 500 * GIB,
        locator.resolve("e_archive").absolute_path: 150 * GIB,
    }
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda path: free[path],
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )
    request = StartRequest(
        stage_id="stage05.2",
        run_label="stage05.2_benchmark_attempt41",
        run_dir=(
            locator.resolve("wsl_staging").absolute_path
            / "stage05.2_benchmark_attempt41"
        ),
        staging_root_alias="wsl_staging",
        host_root_alias="d_host",
        archive_root_alias="e_archive",
        planned_archive_bytes=100 * GIB,
        max_active_workspace_bytes=32 * GIB,
        projected_host_growth_bytes=32 * GIB,
        stage_plan_sha256="4" * 64,
    )

    governance.preflight_run(request)
    request.run_dir.mkdir()
    free[locator.resolve("e_archive").absolute_path] = 50 * GIB

    with pytest.raises(StorageCapacityError) as raised:
        governance.preflight_run(request)

    assert raised.value.observation["deficits_by_alias"] == {
        "e_archive": 50 * GIB
    }
    assert request.run_dir.is_dir()


def test_volume_probe_failure_is_persisted_before_run_directory_creation(
    tmp_path: Path,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    archive_path = locator.resolve("e_archive").absolute_path
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: (
            (_ for _ in ()).throw(RuntimeError("archive disconnected"))
            if path == archive_path
            else next(
                root.volume
                for root in (locator.resolve(alias) for alias in locator.aliases)
                if root.absolute_path == path
            )
        ),
    )
    run_dir = (
        locator.resolve("wsl_staging").absolute_path
        / "stage05.2_benchmark_attempt42"
    )

    with pytest.raises(StorageCapacityError) as raised:
        governance.preflight_run(
            StartRequest(
                stage_id="stage05.2",
                run_label=run_dir.name,
                run_dir=run_dir,
                staging_root_alias="wsl_staging",
                host_root_alias="d_host",
                archive_root_alias="e_archive",
                planned_archive_bytes=1 * GIB,
                max_active_workspace_bytes=32 * GIB,
                projected_host_growth_bytes=1 * GIB,
                stage_plan_sha256="5" * 64,
            )
        )

    measurement_errors = raised.value.observation["measurement_errors_by_alias"]
    assert isinstance(measurement_errors, dict)
    assert "archive disconnected" in measurement_errors["e_archive"]
    assert not run_dir.exists()
    observations = tuple((tmp_path / "state" / "observations").glob("*.json"))
    assert len(observations) == 1
    assert observations[0].with_suffix(".json.sha256").is_file()


def test_unknown_retention_is_full_and_resolves_through_verified_generation(
    tmp_path: Path,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    source = tmp_path / "source" / "stage05.2_benchmark_attempt03"
    (source / "control").mkdir(parents=True)
    (source / "raw" / "shard0001").mkdir(parents=True)
    (source / "control" / "manifest.json").write_text("manifest\n", encoding="utf-8")
    (source / "raw" / "shard0001" / "events.parquet").write_bytes(b"critical")
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        retention_state_root=tmp_path / "e-archive" / ".storage-governance",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )

    receipt = governance.retain_run(
        RetentionRequest(
            run_label=source.name,
            generation=1,
            retention_class=RetentionClass.UNKNOWN_FULL,
            archive_root_alias="e_archive",
            archive_relative_path=f"stage05.2/generations/{source.name}/0001",
            segments=(RetentionSegment("run", source, "."),),
        )
    )

    assert receipt.audit_only is False
    assert receipt.kept_file_count == 2
    assert receipt.omitted_file_count == 0
    assert source.is_dir()
    assert (
        tmp_path
        / "e-archive"
        / ".storage-governance"
        / "retention_registry_v2.json"
    ).is_file()
    assert not (tmp_path / "state" / "retention_registry_v2.json").exists()
    resolved = governance.resolve_run(source.name)
    assert resolved == receipt.archive_path
    assert (resolved / "control" / "manifest.json").read_text(encoding="utf-8") == (
        "manifest\n"
    )
    assert (resolved / "raw" / "shard0001" / "events.parquet").read_bytes() == b"critical"


def test_full_retention_limits_content_io_to_one_source_and_one_target_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    source = tmp_path / "source" / "stage05.2_benchmark_attempt42"
    source.mkdir(parents=True)
    payload = b"x" * (2 * 1024 * 1024)
    (source / "raw.bin").write_bytes(payload)
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )
    original_sha256 = storage_governance_module._file_sha256
    original_copy2 = storage_governance_module.shutil.copy2
    original_stream_copy = storage_governance_module._copy_file_with_sha256
    source_bytes_read = 0
    target_bytes_read = 0

    def counted_sha256(path: Path) -> str:
        nonlocal source_bytes_read, target_bytes_read
        if path.is_relative_to(source):
            source_bytes_read += path.stat().st_size
        elif path.is_relative_to(locator.resolve("e_archive").absolute_path):
            target_bytes_read += path.stat().st_size
        return original_sha256(path)

    def counted_copy2(source_path: Path, target_path: Path) -> None:
        nonlocal source_bytes_read
        source_bytes_read += source_path.stat().st_size
        original_copy2(source_path, target_path)

    def counted_stream_copy(
        task: tuple[
            str,
            Path,
            Path,
            storage_governance_module._FileSnapshot,
        ],
    ) -> storage_governance_module._FileIdentity:
        nonlocal source_bytes_read
        source_bytes_read += task[3].byte_count
        return original_stream_copy(task)

    monkeypatch.setattr(storage_governance_module, "_file_sha256", counted_sha256)
    monkeypatch.setattr(storage_governance_module.shutil, "copy2", counted_copy2)
    monkeypatch.setattr(
        storage_governance_module,
        "_copy_file_with_sha256",
        counted_stream_copy,
    )
    governance.retain_run(
        RetentionRequest(
            run_label=source.name,
            generation=1,
            retention_class=RetentionClass.UNKNOWN_FULL,
            archive_root_alias="e_archive",
            archive_relative_path=f"runs/{source.name}/generation-0001",
            segments=(RetentionSegment("run", source, "."),),
        )
    )

    assert source_bytes_read <= len(payload)
    assert target_bytes_read <= len(payload)


def test_full_retention_rejects_target_that_differs_from_signed_segment_tree(
    tmp_path: Path,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    source = tmp_path / "source" / "stage05.2_benchmark_attempt42"
    source.mkdir(parents=True)
    (source / "raw.bin").write_bytes(b"signed inventory payload")
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )
    destination = (
        locator.resolve("e_archive").absolute_path
        / "runs"
        / source.name
        / "generation-0001"
    )

    with pytest.raises(
        StorageGovernanceError,
        match="differs from signed inventory",
    ):
        governance.retain_run(
            RetentionRequest(
                run_label=source.name,
                generation=1,
                retention_class=RetentionClass.UNKNOWN_FULL,
                archive_root_alias="e_archive",
                archive_relative_path=f"runs/{source.name}/generation-0001",
                segments=(RetentionSegment("run", source, "run"),),
                copy_workers=4,
                expected_segment_trees=(
                    ExpectedSegmentTree(
                        segment_id="run",
                        file_count=1,
                        byte_count=len(b"signed inventory payload"),
                        tree_sha256="0" * 64,
                    ),
                ),
            )
        )

    assert not destination.exists()


def test_auto_native_copy_is_used_and_independently_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    source = tmp_path / "source" / "stage05.2_benchmark_attempt42"
    source.mkdir(parents=True)
    (source / "raw.bin").write_bytes(b"native copy payload")
    file_count, byte_count, tree_sha256 = compute_tree_identity(source)
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )
    native_calls: list[tuple[Path, Path, int]] = []

    def fake_windows_path(path: Path) -> str:
        return str(path)

    def fake_native_copy(
        *,
        source: Path,
        destination: Path,
        workers: int,
    ) -> dict[str, object]:
        native_calls.append((source, destination, workers))
        storage_governance_module.shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True,
        )
        return {
            "backend": "test_native",
            "source": str(source),
            "destination": str(destination),
            "workers": workers,
            "elapsed_seconds": 0.01,
            "exit_code": 1,
        }

    def fake_native_verify_retention_tree(
        *,
        root: Path,
        segments: tuple[RetentionSegment, ...],
        expected_trees: tuple[ExpectedSegmentTree, ...],
        workers: int,
    ) -> SimpleNamespace:
        expected_by_id = {tree.segment_id: tree for tree in expected_trees}
        segment_identities: dict[str, tuple[int, int, str]] = {}
        for segment in segments:
            observed = compute_tree_identity(root / segment.logical_prefix)
            expected = expected_by_id[segment.segment_id]
            assert observed == (
                expected.file_count,
                expected.byte_count,
                expected.tree_sha256,
            )
            segment_identities[segment.segment_id] = observed
        file_count, byte_count, tree_sha256 = compute_tree_identity(root)
        return SimpleNamespace(
            file_count=file_count,
            byte_count=byte_count,
            tree_sha256=tree_sha256,
            segment_identities=segment_identities,
            observation={
                "backend": "test_native_verifier",
                "workers": workers,
                "file_count": file_count,
                "byte_count": byte_count,
                "tree_sha256": tree_sha256,
            },
        )

    monkeypatch.setattr(
        storage_governance_module,
        "_windows_path_for_mounted_drive",
        fake_windows_path,
    )
    monkeypatch.setattr(
        storage_governance_module,
        "_native_copy_segment",
        fake_native_copy,
    )
    monkeypatch.setattr(
        storage_governance_module,
        "_native_verify_retention_tree",
        fake_native_verify_retention_tree,
    )
    receipt = governance.retain_run(
        RetentionRequest(
            run_label=source.name,
            generation=1,
            retention_class=RetentionClass.UNKNOWN_FULL,
            archive_root_alias="e_archive",
            archive_relative_path=f"runs/{source.name}/generation-0001",
            segments=(RetentionSegment("run", source, "run"),),
            copy_workers=32,
            copy_backend="auto_native",
            expected_segment_trees=(
                ExpectedSegmentTree(
                    segment_id="run",
                    file_count=file_count,
                    byte_count=byte_count,
                    tree_sha256=tree_sha256,
                ),
            ),
        )
    )

    assert len(native_calls) == 1
    native_source, native_destination, native_workers = native_calls[0]
    assert native_source == source
    assert native_destination.name == "run"
    assert native_destination.parent.name.startswith(".generation-0001.incoming-")
    assert native_workers == 32
    assert (receipt.archive_path / "run" / "raw.bin").read_bytes() == (
        b"native copy payload"
    )
    observations = tuple(
        (tmp_path / "state" / "retention_copy_observations").rglob("*.json")
    )
    assert len(observations) == 1


def test_windows_native_tree_helper_matches_canonical_segment_identities(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    first = target / "d_benchmark"
    second = target / "wsl_active"
    first.mkdir(parents=True)
    second.mkdir()
    (first / "first.bin").write_bytes(b"first")
    (second / "second.bin").write_bytes(b"second")
    helper = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "hash_retention_tree_windows.py"
    )

    completed = subprocess.run(
        (
            sys.executable,
            str(helper),
            "--root",
            str(target),
            "--workers",
            "4",
            "--segment",
            "d_benchmark=d_benchmark",
            "--segment",
            "wsl_active=wsl_active",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == (
        "experiment-retention-native-tree-verification-v1"
    )
    assert (
        payload["file_count"],
        payload["byte_count"],
        payload["tree_sha256"],
    ) == compute_tree_identity(target)
    assert (
        payload["segments"]["d_benchmark"]["file_count"],
        payload["segments"]["d_benchmark"]["byte_count"],
        payload["segments"]["d_benchmark"]["tree_sha256"],
    ) == compute_tree_identity(first)
    assert (
        payload["segments"]["wsl_active"]["file_count"],
        payload["segments"]["wsl_active"]["byte_count"],
        payload["segments"]["wsl_active"]["tree_sha256"],
    ) == compute_tree_identity(second)


def test_windows_native_mapping_helper_reuses_one_worker_pool(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    first = target / "runs" / "attempt01" / "generation-0001"
    second = target / "runs" / "attempt02" / "generation-0001"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "manifest.json").write_text("first", encoding="utf-8")
    (second / "raw.bin").write_bytes(b"second")
    helper = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "hash_retention_tree_windows.py"
    )
    request = {
        "root": str(target),
        "workers": 4,
        "mappings": [
            {
                "logical_id": "attempt01",
                "relative_path": "runs/attempt01/generation-0001",
            },
            {
                "logical_id": "attempt02",
                "relative_path": "runs/attempt02/generation-0001",
            },
            {
                "logical_id": "empty_attempt",
                "relative_path": "runs/empty/generation-0001",
            },
        ],
    }

    completed = subprocess.run(
        (sys.executable, str(helper), "--mapping-stdin"),
        input=json.dumps(request),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == (
        "experiment-retention-native-mapping-verification-v1"
    )
    assert payload["workers"] == 4
    assert (
        payload["mappings"]["attempt01"]["file_count"],
        payload["mappings"]["attempt01"]["byte_count"],
        payload["mappings"]["attempt01"]["tree_sha256"],
    ) == compute_tree_identity(first)
    assert (
        payload["mappings"]["attempt02"]["file_count"],
        payload["mappings"]["attempt02"]["byte_count"],
        payload["mappings"]["attempt02"]["tree_sha256"],
    ) == compute_tree_identity(second)
    assert (
        payload["mappings"]["empty_attempt"]["file_count"],
        payload["mappings"]["empty_attempt"]["byte_count"],
        payload["mappings"]["empty_attempt"]["tree_sha256"],
    ) == compute_tree_identity(target / "runs" / "empty" / "generation-0001")


def test_preflight_rejects_registered_run_label_with_signed_observation(
    tmp_path: Path,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    source = tmp_path / "source" / "stage05.2_benchmark_attempt43"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text("sealed", encoding="utf-8")
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )
    governance.retain_run(
        RetentionRequest(
            run_label=source.name,
            generation=1,
            retention_class=RetentionClass.UNKNOWN_FULL,
            archive_root_alias="e_archive",
            archive_relative_path=f"runs/{source.name}/generation-0001",
            segments=(RetentionSegment("run", source, "."),),
        )
    )
    run_dir = locator.resolve("wsl_staging").absolute_path / source.name

    with pytest.raises(StorageCapacityError) as raised:
        governance.preflight_run(
            StartRequest(
                stage_id="stage05.2",
                run_label=source.name,
                run_dir=run_dir,
                staging_root_alias="wsl_staging",
                host_root_alias="d_host",
                archive_root_alias="e_archive",
                planned_archive_bytes=1 * GIB,
                max_active_workspace_bytes=32 * GIB,
                projected_host_growth_bytes=1 * GIB,
                stage_plan_sha256="6" * 64,
            )
        )

    assert raised.value.observation["passed"] is False
    assert "already retained" in str(raised.value.observation["identity_errors"])
    assert not run_dir.exists()


def test_duplicate_failure_reduction_requires_signed_adjudication(tmp_path: Path) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    source = tmp_path / "source" / "stage05.2_benchmark_attempt04"
    source.mkdir(parents=True)
    (source / "failure.log").write_text("same root cause\n", encoding="utf-8")
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )

    with pytest.raises(StorageGovernanceError, match="adjudication"):
        governance.retain_run(
            RetentionRequest(
                run_label=source.name,
                generation=1,
                retention_class=RetentionClass.DUPLICATE_FAILURE_REDUCED,
                archive_root_alias="e_archive",
                archive_relative_path=f"stage05.2/generations/{source.name}/0001",
                segments=(RetentionSegment("run", source, "."),),
                adjudication=AdjudicationRecord(
                    run_label=source.name,
                    root_cause_id="sqlite-transaction-v1",
                    canonical_representative_run_label=(
                        "stage05.2_benchmark_attempt03"
                    ),
                    failure_location="artifact_store.register_many",
                    evidence_references=("docs/stage052_change_log.md#attempt60",),
                    adjudication_sha256="",
                ),
                keep_relative_paths=("failure.log",),
            )
        )


def test_native_performance_failure_rejects_generic_retention_replay(
    tmp_path: Path,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    label = "stage05.2_native_architecture_performance_calibration_attempt90"
    source = (tmp_path / "source" / label).resolve()
    (source / "control").mkdir(parents=True)
    (source / "review").mkdir()
    (source / "control" / f"{label}_failure_lifecycle_manifest.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (source / "review" / "review_manifest.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    adjudication_path = (tmp_path / "adjudication.json").resolve()
    adjudication = write_adjudication_record(
        adjudication_path,
        run_label=label,
        root_cause_id="dynamic-memory-identity-self-race-v1",
        canonical_representative_run_label=label,
        failure_location="failure_summary.json",
        evidence_references=("failure_summary.json",),
    )
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )

    with pytest.raises(
        StorageGovernanceError,
        match="independent retention replay failed",
    ):
        governance.retain_run(
            RetentionRequest(
                run_label=label,
                generation=1,
                retention_class=RetentionClass.UNIQUE_FAILURE_FULL,
                root_cause_id="dynamic-memory-identity-self-race-v1",
                adjudication=adjudication,
                adjudication_path=adjudication_path,
                archive_root_alias="e_archive",
                archive_relative_path=f"stage05.2/runs/{label}/generation-0001",
                segments=(RetentionSegment("run", source, "."),),
                replay_verifier=_replay_writer(
                    tmp_path,
                    run_label=label,
                    generation=1,
                ),
            )
        )


def test_duplicate_failure_retains_auditable_projection_and_trigger_shard(
    tmp_path: Path,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )
    representative = (
        tmp_path / "source" / "stage05.2_benchmark_attempt03"
    ).resolve()
    (representative / "raw").mkdir(parents=True)
    (representative / "raw" / "events.parquet").write_bytes(b"canonical full")
    representative_adjudication_path = (
        tmp_path / "adjudications" / f"{representative.name}.json"
    )
    representative_adjudication = write_adjudication_record(
        representative_adjudication_path,
        run_label=representative.name,
        root_cause_id="sqlite-transaction-v1",
        canonical_representative_run_label=representative.name,
        failure_location="artifact_store.register_many",
        evidence_references=("docs/stage052_change_log.md#attempt60",),
    )
    governance.retain_run(
        RetentionRequest(
            run_label=representative.name,
            generation=1,
            retention_class=RetentionClass.UNIQUE_FAILURE_FULL,
            root_cause_id="sqlite-transaction-v1",
            adjudication=representative_adjudication,
            adjudication_path=representative_adjudication_path,
            archive_root_alias="e_archive",
            archive_relative_path=(
                f"stage05.2/generations/{representative.name}/0001"
            ),
            segments=(RetentionSegment("run", representative, "."),),
            replay_verifier=_replay_writer(
                tmp_path,
                run_label=representative.name,
                generation=1,
            ),
        )
    )
    duplicate = (tmp_path / "source" / "stage05.2_benchmark_attempt04").resolve()
    for directory in (
        duplicate / "control",
        duplicate / "review",
        duplicate / "logs",
        duplicate / "raw" / "trigger",
        duplicate / "raw" / "unrelated",
    ):
        directory.mkdir(parents=True)
    files = {
        "control/manifest.json": b"manifest",
        "review/report.md": b"review",
        "logs/failure.log": b"failure",
        "raw/trigger/events.parquet": b"trigger",
        "raw/unrelated/events.parquet": b"large duplicate",
    }
    for relative, payload in files.items():
        duplicate.joinpath(*relative.split("/")).write_bytes(payload)
    adjudication_path = tmp_path / "adjudications" / f"{duplicate.name}.json"
    adjudication = write_adjudication_record(
        adjudication_path,
        run_label=duplicate.name,
        root_cause_id="sqlite-transaction-v1",
        canonical_representative_run_label=representative.name,
        failure_location="artifact_store.register_many",
        evidence_references=("docs/stage052_change_log.md#attempt60",),
    )

    receipt = governance.retain_run(
        RetentionRequest(
            run_label=duplicate.name,
            generation=1,
            retention_class=RetentionClass.DUPLICATE_FAILURE_REDUCED,
            archive_root_alias="e_archive",
            archive_relative_path=f"stage05.2/generations/{duplicate.name}/0001",
            segments=(RetentionSegment("run", duplicate, "."),),
            adjudication=adjudication,
            adjudication_path=adjudication_path,
            keep_relative_paths=(
                "control/manifest.json",
                "review/report.md",
                "logs/failure.log",
                "raw/trigger/events.parquet",
            ),
        )
    )

    assert receipt.audit_only is True
    assert receipt.kept_file_count == 4
    assert receipt.omitted_file_count == 1
    with pytest.raises(StorageGovernanceError, match="audit-only"):
        governance.resolve_run(duplicate.name)
    resolved = governance.resolve_run(duplicate.name, allow_audit_only=True)
    assert (resolved / "raw" / "trigger" / "events.parquet").is_file()
    assert not (resolved / "raw" / "unrelated" / "events.parquet").exists()
    projection = json.loads(
        (resolved / "retention_projection_manifest.json").read_text(encoding="utf-8")
    )
    assert projection["audit_only"] is True
    assert projection["root_cause_id"] == "sqlite-transaction-v1"
    assert len(projection["omitted_files"]) == 1


@pytest.mark.parametrize("unique_failure", (False, True))
def test_lifecycle_binding_preserves_storage_and_inventory_tree_identities(
    tmp_path: Path,
    unique_failure: bool,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    state_root = tmp_path / "state"
    retention_state_root = tmp_path / "retention-state"
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=state_root,
        retention_state_root=retention_state_root,
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )
    run_label = (
        "stage05.2_resource_calibration_attempt97"
        if unique_failure
        else "stage05.2_resource_calibration_attempt98"
    )
    source = tmp_path / "source" / run_label
    source.mkdir(parents=True)
    artifact = source / "manifest.json"
    artifact.write_bytes(b"sealed calibration")
    adjudication_path: Path | None = None
    adjudication: AdjudicationRecord | None = None
    if unique_failure:
        adjudication_path = (tmp_path / "adjudication.json").resolve()
        adjudication = write_lifecycle_adjudication_record(
            adjudication_path,
            run_label=run_label,
            review_manifest_sha256="b" * 64,
            root_cause_id="unbound-producer-resource-contract-v1",
            canonical_representative_run_label=run_label,
            canonical_representative_record_sha256="c" * 64,
            failure_code="invalid_manifest",
            failing_component="stage052_calibration",
            invariant_or_check="resource_contract_not_bound_at_seal",
            failure_location="control/lifecycle_manifest.json",
        )
    receipt = governance.retain_run(
        RetentionRequest(
            run_label=run_label,
            generation=1,
            retention_class=(
                RetentionClass.UNIQUE_FAILURE_FULL
                if unique_failure
                else RetentionClass.ACCEPTED_FULL
            ),
            root_cause_id=(
                "unbound-producer-resource-contract-v1" if unique_failure else ""
            ),
            adjudication=adjudication,
            adjudication_path=adjudication_path,
            archive_root_alias="e_archive",
            archive_relative_path=f"runs/{run_label}/generation-0001",
            segments=(RetentionSegment("run", source, "."),),
            replay_verifier=_replay_writer(
                tmp_path,
                run_label=run_label,
                generation=1,
            ),
        )
    )
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    lifecycle_digest = hashlib.sha256()
    lifecycle_digest.update(b"manifest.json\0")
    lifecycle_digest.update(str(artifact.stat().st_size).encode("ascii"))
    lifecycle_digest.update(b"\0")
    lifecycle_digest.update(artifact_sha256.encode("ascii"))
    lifecycle_digest.update(b"\n")
    lifecycle_tree_sha256 = lifecycle_digest.hexdigest()
    inventory_path = tmp_path / "content-inventory.json"
    storage_governance_module._write_signed_json(
        inventory_path,
        {
            "schema_version": "experiment-content-inventory-v1",
            "run_label": run_label,
            "source_tree_sha256": lifecycle_tree_sha256,
            "files": [
                {
                    "relative_path": "manifest.json",
                    "byte_count": artifact.stat().st_size,
                    "sha256": artifact_sha256,
                    "modified_time_ns": artifact.stat().st_mtime_ns,
                }
            ],
        },
    )
    permit_path = state_root / "permits" / f"{run_label}.json"
    permit_sha256 = storage_governance_module._write_signed_json(
        permit_path,
        {"run_label": run_label, "status": "reserved"},
    )

    lifecycle_class = (
        "unique_failure_full" if unique_failure else "current_accepted_full"
    )
    with pytest.raises(StorageGovernanceError, match="binding|governed retention"):
        governance.write_lifecycle_retention_binding(
            receipt,
            lifecycle_retention_class=(
                "current_accepted_full" if unique_failure else "unique_failure_full"
            ),
            storage_permit_sha256=permit_sha256,
            review_manifest_sha256="b" * 64,
            close_performance={"backend": "test"},
            content_inventory_path=inventory_path,
        )
    binding_path = governance.write_lifecycle_retention_binding(
        receipt,
        lifecycle_retention_class=lifecycle_class,
        storage_permit_sha256=permit_sha256,
        review_manifest_sha256="b" * 64,
        close_performance={"backend": "test"},
        content_inventory_path=inventory_path,
    )
    binding = storage_governance_module._load_signed_json(binding_path)

    assert binding["archive_tree_sha256"] == lifecycle_tree_sha256
    assert binding["storage_tree_sha256"] == receipt.tree_sha256
    assert binding["review_manifest_sha256"] == "b" * 64
    assert binding["root_cause_id"] == (
        "unbound-producer-resource-contract-v1" if unique_failure else ""
    )
    assert binding["adjudication_sha256"] == (
        adjudication.adjudication_sha256 if adjudication is not None else ""
    )
    assert binding["archive_tree_sha256"] != binding["storage_tree_sha256"]
    assert binding["content_inventory_sha256"] == hashlib.sha256(
        inventory_path.read_bytes()
    ).hexdigest()


def test_retention_retry_adopts_only_an_identical_published_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("sealed", encoding="utf-8")
    request = RetentionRequest(
        run_label="stage03.4_control_parallel_attempt98",
        generation=1,
        retention_class=RetentionClass.ACCEPTED_FULL,
        archive_root_alias="e_archive",
        archive_relative_path=(
            "runs/stage03.4_control_parallel_attempt98/generation-0001"
        ),
        segments=(
            RetentionSegment(
                segment_id="control",
                source_path=source,
                logical_prefix=".",
            ),
        ),
        replay_verifier=_replay_writer(
            tmp_path,
            run_label="stage03.4_control_parallel_attempt98",
            generation=1,
        ),
    )
    original_register = governance._register_retention_generation
    calls = 0

    def fail_once(**kwargs: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated registry interruption")
        return original_register(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(governance, "_register_retention_generation", fail_once)
    with pytest.raises(OSError, match="registry interruption"):
        governance.retain_run(request)

    destination = (
        tmp_path
        / "e-archive"
        / "runs"
        / "stage03.4_control_parallel_attempt98"
        / "generation-0001"
    )
    assert destination.is_dir()
    assert (source / "manifest.json").is_file()

    receipt = governance.retain_run(request)
    assert receipt.archive_path == destination
    assert governance.resolve_run(request.run_label) == destination
    registry_path = (
        governance.retention_state_root / "retention_registry_v2.json"
    )
    legacy_registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for record in legacy_registry["records"]:
        record.pop("adjudication_sha256")
    storage_governance_module._write_signed_json(registry_path, legacy_registry)
    assert governance.retain_run(request).archive_path == destination

    (destination / "manifest.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(StorageGovernanceError, match="already exists but differs"):
        governance.retain_run(
            replace(
                request,
                run_label="stage03.4_control_parallel_attempt99",
            )
        )


def test_retention_atomic_publish_recovers_from_transient_ntfs_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    source = tmp_path / "source" / "stage05.2_benchmark_attempt07"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text("sealed", encoding="utf-8")
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )
    original_replace = Path.replace
    publish_calls = 0

    def transient_replace(self: Path, target: Path) -> Path:
        nonlocal publish_calls
        if ".incoming-" in self.name and target.name == "generation-0001":
            publish_calls += 1
            if publish_calls <= 2:
                raise PermissionError("simulated transient NTFS directory lock")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", transient_replace)
    monkeypatch.setattr(storage_governance_module.time, "sleep", lambda _delay: None)
    receipt = governance.retain_run(
        RetentionRequest(
            run_label=source.name,
            generation=1,
            retention_class=RetentionClass.UNKNOWN_FULL,
            archive_root_alias="e_archive",
            archive_relative_path=f"runs/{source.name}/generation-0001",
            segments=(RetentionSegment("run", source, "."),),
        )
    )

    assert publish_calls == 3
    assert receipt.archive_path.is_dir()
    observation = json.loads(
        (
            tmp_path
            / "state"
            / "retention_publish_observations"
            / f"{source.name}-generation-0001.json"
        ).read_text(encoding="utf-8")
    )
    assert observation["status"] == "recovered"
    assert observation["successful_attempt"] == 3
    assert len(observation["attempts"]) == 2


def test_retention_atomic_publish_exhaustion_remains_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    source = tmp_path / "source" / "stage05.2_benchmark_attempt08"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text("sealed", encoding="utf-8")
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )
    original_replace = Path.replace
    publish_calls = 0

    def locked_replace(self: Path, target: Path) -> Path:
        nonlocal publish_calls
        if ".incoming-" in self.name and target.name == "generation-0001":
            publish_calls += 1
            raise PermissionError("simulated persistent NTFS directory lock")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", locked_replace)
    monkeypatch.setattr(storage_governance_module.time, "sleep", lambda _delay: None)
    with pytest.raises(PermissionError, match="persistent NTFS"):
        governance.retain_run(
            RetentionRequest(
                run_label=source.name,
                generation=1,
                retention_class=RetentionClass.UNKNOWN_FULL,
                archive_root_alias="e_archive",
                archive_relative_path=f"runs/{source.name}/generation-0001",
                segments=(RetentionSegment("run", source, "."),),
            )
        )

    assert publish_calls == 7
    assert (source / "manifest.json").is_file()
    destination_parent = (
        tmp_path / "e-archive" / "runs" / source.name
    )
    assert not (destination_parent / "generation-0001").exists()
    assert not tuple(destination_parent.glob(".generation-0001.incoming-*"))
    observation = json.loads(
        (
            tmp_path
            / "state"
            / "retention_publish_observations"
            / f"{source.name}-generation-0001.json"
        ).read_text(encoding="utf-8")
    )
    assert observation["status"] == "failed"
    assert len(observation["attempts"]) == 7


def test_full_retention_replay_failure_keeps_source_and_blocks_registry(
    tmp_path: Path,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    source = tmp_path / "source" / "stage04_adaptive_weights_attempt98"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text("sealed", encoding="utf-8")
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )

    with pytest.raises(StorageGovernanceError, match="independent retention replay"):
        governance.retain_run(
            RetentionRequest(
                run_label=source.name,
                generation=1,
                retention_class=RetentionClass.ACCEPTED_FULL,
                archive_root_alias="e_archive",
                archive_relative_path=f"runs/{source.name}/generation-0001",
                segments=(RetentionSegment("run", source, "."),),
                replay_verifier=lambda _path: (_ for _ in ()).throw(
                    RuntimeError("replay mismatch")
                ),
            )
        )

    assert (source / "manifest.json").is_file()
    with pytest.raises(StorageGovernanceError, match="not registered"):
        governance.resolve_run(source.name)


def test_retention_rejects_source_change_during_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    source = tmp_path / "source" / "stage03.4_control_parallel_attempt97"
    source.mkdir(parents=True)
    source_file = source / "manifest.json"
    source_file.write_text("sealed", encoding="utf-8")
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )
    original_stream_copy = storage_governance_module._copy_file_with_sha256

    def copy_then_mutate(
        task: tuple[
            str,
            Path,
            Path,
            storage_governance_module._FileSnapshot,
        ],
    ) -> storage_governance_module._FileIdentity:
        identity = original_stream_copy(task)
        source_file.write_text("changed after copy", encoding="utf-8")
        return identity

    monkeypatch.setattr(
        storage_governance_module,
        "_copy_file_with_sha256",
        copy_then_mutate,
    )

    with pytest.raises(StorageGovernanceError, match="source changed"):
        governance.retain_run(
            RetentionRequest(
                run_label=source.name,
                generation=1,
                retention_class=RetentionClass.ACCEPTED_FULL,
                archive_root_alias="e_archive",
                archive_relative_path=f"runs/{source.name}/generation-0001",
                segments=(RetentionSegment("run", source, "."),),
                replay_verifier=_replay_writer(
                    tmp_path,
                    run_label=source.name,
                    generation=1,
                ),
            )
        )
    assert not (
        locator.resolve("e_archive").absolute_path
        / "runs"
        / source.name
        / "generation-0001"
    ).exists()


def test_rebuildable_sweep_is_allowlisted_dry_run_first_and_lock_safe(
    tmp_path: Path,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
        keeper_reference_scanner=lambda _path: (),
    )
    allowed = tmp_path / "rebuildable"
    cache = allowed / "cache" / "route-cache"
    locked_spool = allowed / "spool" / "active"
    cache.mkdir(parents=True)
    locked_spool.mkdir(parents=True)
    (cache / "data.bin").write_bytes(b"rebuildable")
    (locked_spool / "data.bin").write_bytes(b"active")
    (locked_spool / ".lock").write_text("worker", encoding="utf-8")
    cache_tree_sha256 = compute_tree_sha256(cache)
    locked_tree_sha256 = compute_tree_sha256(locked_spool)
    request = SweepRequest(
        allowed_roots=(allowed,),
        retention_days=0,
        assets=(
            RebuildableAsset(
                path=cache,
                kind=RebuildableKind.CACHE,
                expected_tree_sha256=cache_tree_sha256,
            ),
            RebuildableAsset(
                path=locked_spool,
                kind=RebuildableKind.TEMPORARY_SPOOL,
                expected_tree_sha256=locked_tree_sha256,
            ),
        ),
        apply=False,
    )

    dry_run = governance.sweep_rebuildable_assets(request)
    assert dry_run.candidates == (cache,)
    assert dry_run.retained == ((locked_spool, "active_lock"),)
    assert cache.is_dir()
    with pytest.raises(StorageGovernanceError, match="approved signed dry run"):
        governance.sweep_rebuildable_assets(replace(request, apply=True))

    with pytest.raises(StorageGovernanceError, match="literal confirmation"):
        governance.sweep_rebuildable_assets(
            replace(
                request,
                apply=True,
                approved_dry_run_path=dry_run.receipt_path,
                approved_dry_run_sha256=dry_run.receipt_sha256,
            )
        )
    confirmation_path = tmp_path / "cleanup-confirmation.json"
    confirmation_sha256 = write_cleanup_confirmation_receipt(
        confirmation_path,
        dry_run_path=dry_run.receipt_path,
        dry_run_sha256=dry_run.receipt_sha256,
        confirmation_text="确认",
    )
    applied = governance.sweep_rebuildable_assets(
        replace(
            request,
            apply=True,
            approved_dry_run_path=dry_run.receipt_path,
            approved_dry_run_sha256=dry_run.receipt_sha256,
            confirmation_receipt_path=confirmation_path,
            confirmation_receipt_sha256=confirmation_sha256,
        )
    )
    assert applied.applied is True
    assert applied.receipt_path.is_file()
    assert not cache.exists()
    assert locked_spool.is_dir()


def test_rebuildable_sweep_fails_closed_without_independent_keeper_scan(
    tmp_path: Path,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    cache = tmp_path / "rebuildable" / "cache"
    cache.mkdir(parents=True)
    (cache / "data.bin").write_bytes(b"rebuildable")
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
    )

    receipt = governance.sweep_rebuildable_assets(
        SweepRequest(
            allowed_roots=(cache.parent,),
            retention_days=0,
            assets=(
                RebuildableAsset(
                    path=cache,
                    kind=RebuildableKind.CACHE,
                    expected_tree_sha256=compute_tree_sha256(cache),
                ),
            ),
            apply=False,
        )
    )

    assert receipt.candidates == ()
    assert receipt.retained == ((cache, "missing_keeper_reference_scan"),)
    assert cache.is_dir()


def test_rebuildable_sweep_never_selects_evidence_or_unsealed_runs(
    tmp_path: Path,
) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    allowed = tmp_path / "rebuildable"
    evidence_cache = allowed / "review"
    unsealed_spool = (
        allowed / "stage05.2_benchmark_attempt98" / "temporary-spool"
    )
    evidence_cache.mkdir(parents=True)
    unsealed_spool.mkdir(parents=True)
    (evidence_cache / "ordinary.bin").write_text(
        "protected",
        encoding="utf-8",
    )
    (unsealed_spool / "data.bin").write_bytes(b"active run")
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
        keeper_reference_scanner=lambda _path: (),
    )

    receipt = governance.sweep_rebuildable_assets(
        SweepRequest(
            allowed_roots=(allowed,),
            retention_days=0,
            assets=(
                RebuildableAsset(
                    path=evidence_cache,
                    kind=RebuildableKind.CACHE,
                    expected_tree_sha256=compute_tree_sha256(evidence_cache),
                ),
                RebuildableAsset(
                    path=unsealed_spool,
                    kind=RebuildableKind.TEMPORARY_SPOOL,
                    expected_tree_sha256=compute_tree_sha256(unsealed_spool),
                ),
            ),
        )
    )

    assert receipt.candidates == ()
    assert receipt.retained == (
        (evidence_cache, "protected_evidence"),
        (unsealed_spool, "unsealed_run"),
    )


def test_cross_alias_migration_attestation_requires_identical_trees(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "d-archive"
    destination_root = tmp_path / "e-archive"
    source = source_root / "stage05.2" / "attempt01"
    destination = destination_root / "stage05.2" / "attempt01"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    (source / "manifest.json").write_text("immutable", encoding="utf-8")
    (destination / "manifest.json").write_text("immutable", encoding="utf-8")
    attestation = tmp_path / "migration-attestation.json"
    dry_run = tmp_path / "migration-dry-run.json"
    dry_run_sha256 = write_migration_dry_run(
        dry_run,
        {
            "schema_version": "experiment-storage-migration-dry-run-v1",
            "source_deletion_authorized": False,
            "retention_default": "unknown_full",
            "full_retention_upper_bound_bytes": len(b"immutable"),
            "sources": [
                {
                    "logical_id": "stage05.2_attempt01",
                    "root_alias": "d_archive",
                    "relative_path": "stage05.2/attempt01",
                    "file_count": 1,
                    "byte_count": len(b"immutable"),
                    "tree_sha256": compute_tree_sha256(source),
                }
            ],
            "planned_mappings": [
                {
                    "logical_id": "stage05.2_attempt01",
                    "source_relative_path": "stage05.2/attempt01",
                    "destination_relative_path": "stage05.2/attempt01",
                    "file_count": 1,
                    "byte_count": len(b"immutable"),
                    "tree_sha256": compute_tree_sha256(source),
                    "source_root_alias": "d_archive",
                    "destination_root_alias": "e_archive",
                }
            ],
        },
    )
    wrong_source_projection = json.loads(
        dry_run.read_text(encoding="utf-8")
    )
    wrong_source_projection["sources"][0]["root_alias"] = "wrong_archive"
    with pytest.raises(StorageGovernanceError, match="sources do not match"):
        write_migration_dry_run(
            tmp_path / "wrong-source-dry-run.json",
            wrong_source_projection,
        )
    source_volume = VolumeIdentity("d-volume", "ntfs")
    destination_volume = VolumeIdentity("e-volume", "ntfs")

    digest = write_storage_migration_attestation(
        attestation,
        migration_id="stage052-d-to-e-20260731",
        source_root_alias="d_archive",
        destination_root_alias="e_archive",
        source_root=source_root,
        destination_root=destination_root,
        source_volume=source_volume,
        destination_volume=destination_volume,
        mappings=(("stage05.2_attempt01", "stage05.2/attempt01", "stage05.2/attempt01"),),
        dry_run_path=dry_run,
        dry_run_sha256=dry_run_sha256,
    )

    assert digest == attestation.with_suffix(".json.sha256").read_text(
        encoding="ascii"
    ).split()[0]
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    assert payload["source_root_alias"] == "d_archive"
    assert payload["destination_root_alias"] == "e_archive"
    assert payload["source_deletion_authorized"] is False
    verified = verify_storage_migration_attestation(
        attestation,
        expected_sha256=digest,
        source_root_alias="d_archive",
        destination_root_alias="e_archive",
        source_root=source_root,
        destination_root=destination_root,
        source_volume=source_volume,
        destination_volume=destination_volume,
        dry_run_path=dry_run,
        volume_probe=lambda path: (
            source_volume if path == source_root else destination_volume
        ),
    )
    assert verified["mappings"] == payload["mappings"]
    with pytest.raises(StorageGovernanceError, match="signed dry run"):
        write_storage_migration_attestation(
            tmp_path / "semantic-mismatch-attestation.json",
            migration_id="stage052-d-to-e-20260731-other",
            source_root_alias="d_archive",
            destination_root_alias="e_archive",
            source_root=source_root,
            destination_root=destination_root,
            source_volume=source_volume,
            destination_volume=destination_volume,
            mappings=(
                (
                    "different-logical-id",
                    "stage05.2/attempt01",
                    "stage05.2/attempt01",
                ),
            ),
            dry_run_path=dry_run,
            dry_run_sha256=dry_run_sha256,
        )
    (destination / "manifest.json").write_text("different", encoding="utf-8")
    with pytest.raises(StorageGovernanceError, match="tree mismatch"):
        write_storage_migration_attestation(
            tmp_path / "invalid-attestation.json",
            migration_id="stage052-d-to-e-20260731-retry",
            source_root_alias="d_archive",
            destination_root_alias="e_archive",
            source_root=source_root,
            destination_root=destination_root,
            source_volume=source_volume,
            destination_volume=destination_volume,
            mappings=(
                (
                    "stage05.2_attempt01",
                    "stage05.2/attempt01",
                    "stage05.2/attempt01",
                ),
            ),
            dry_run_path=dry_run,
            dry_run_sha256=dry_run_sha256,
        )
    with pytest.raises(StorageGovernanceError, match="unsafe"):
        write_storage_migration_attestation(
            tmp_path / "unsafe-attestation.json",
            migration_id="stage052-d-to-e-20260731-unsafe",
            source_root_alias="d_archive",
            destination_root_alias="e_archive",
            source_root=source_root,
            destination_root=destination_root,
            source_volume=source_volume,
            destination_volume=destination_volume,
            mappings=(("escape", "../escape", "stage05.2/attempt01"),),
            dry_run_path=dry_run,
            dry_run_sha256=dry_run_sha256,
        )


def test_resolver_falls_back_to_verified_stage052_v1_registry(tmp_path: Path) -> None:
    locator = _locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir()
    run_label = "stage05.2_native_kernels_attempt97"
    archive_relative_path = f"stage05.2/history/{run_label}"
    archived = (
        locator.resolve("e_archive").absolute_path
        / "stage05.2"
        / "history"
        / run_label
    )
    archived.mkdir(parents=True)
    (archived / "manifest.json").write_text("legacy", encoding="utf-8")
    file_count, byte_count, tree_sha256 = legacy_retention._tree_identity(archived)
    registry = tmp_path / "stage05.2_retention_registry.csv"
    write_retention_registry(
        registry,
        (
            RetentionRecord(
                run_label=run_label,
                component="native_kernels",
                status="accepted",
                evidence_completeness="complete",
                source_commit="a" * 40,
                prerequisite_run_labels=(),
                original_relative_path=run_label,
                file_count=file_count,
                byte_count=byte_count,
                tree_sha256=tree_sha256,
                archive_root_alias="e_archive",
                archive_relative_path=archive_relative_path,
                disposition="archived",
                verification_status="verified",
                archived_at_utc="2026-07-31T00:00:00Z",
            ),
        ),
    )
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy(),
        locator=locator,
        state_root=tmp_path / "state",
        free_space=lambda _path: 500 * GIB,
        volume_probe=lambda path: next(
            root.volume
            for root in (locator.resolve(alias) for alias in locator.aliases)
            if root.absolute_path == path
        ),
        legacy_registry_path=registry,
    )

    assert governance.resolve_run(run_label) == archived
