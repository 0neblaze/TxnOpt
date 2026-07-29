from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import pytest

import evrptw.artifacts as artifacts_module
import evrptw.experiments.stage052_performance as stage052_performance
import evrptw.experiments.stage052_performance_review as stage052_review
import evrptw.stage052_evidence as stage052_evidence
from evrptw._core import distance_matrix
from evrptw.artifacts import (
    ArtifactBundleWriter,
    ArtifactIntegrityError,
    ArtifactReader,
    ArtifactRunContext,
    ArtifactStorageConfig,
    expand_v2_screening_decision,
)
from evrptw.candidate_transaction import NativeCandidateTransactionConfig
from evrptw.experiments.stage052_performance import (
    PERFORMANCE_INSTANCES,
    PERFORMANCE_SEEDS,
    Stage052Axis,
    _accelerator_decision_inputs,
    _ensure_partial_shard_failure,
    _launch_occupancy_summary,
    _native_ablation_record,
    _require_clean_stage052_repository,
    _run_and_persist_v2_shard,
    _run_v2_shard_task,
    _run_v2_tasks,
    _ShardTask,
    _stage052_anytime_checkpoints,
    _verify_performance_staging_root,
    axes_for_scope,
    load_stage052_config,
    validate_stage052_run_label,
    verify_stage051_prerequisite,
)
from evrptw.experiments.stage052_performance_review import (
    _accelerator_pilot_metadata_matches,
    _audit_native_ablation_records,
    _audit_native_execution,
    _bind_persistence_attribution_review,
    _expected_native_screening_invocations,
    _prerequisite_binding_matches,
    _prior_review_manifest_history,
    _recompute_native_occupancies,
    _recompute_native_screening_batch_counters,
    _recompute_screening_hash,
    _recompute_transaction_hashes,
    _validate_native_shard_manifest_scope,
    _validate_stage052_staging_root_identity,
    render_semantic_mismatches,
    replay_stage052_storage_semantics,
    replay_stage052_storage_semantics_many,
    validate_per_run_scope,
    verify_stage052_review_prerequisite,
    write_semantic_mismatches,
)
from evrptw.experiments.stage052_performance_review import (
    _strict_float as _review_strict_float,
)
from evrptw.measurement import RouteEvaluationTrace
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.native_kernels import NATIVE_KERNEL_ABI_VERSION, NativeKernelConfig
from evrptw.stage052 import (
    STAGE052_MAXIMUM_PERSISTENCE_RATIO,
    AcceleratorDecision,
    ArtifactPersistenceObservation,
    ArtifactStorageObservation,
    PerformanceObservation,
    Stage052Component,
    decide_accelerator,
    evaluate_artifact_persistence,
    evaluate_artifact_storage_promotion,
    evaluate_promotion,
    formal_budget_matrix,
    select_worker_count,
    stage052_contract,
)
from evrptw.stage052_campaign import VolumeIdentity
from evrptw.stage052_evidence import (
    BatchPersistenceEnvelope,
    JobParallelSelectionIdentity,
    PersistenceInterval,
    ProcessTreeResourceSampler,
    Stage052PersistenceAttribution,
    Stage052PrerequisiteIdentity,
    collect_performance_provenance,
    create_stage052_runtime_identity,
    stage052_source_snapshot_contract,
    stage052_storage_root_binding,
    upsert_stage052_campaign_lock,
    validate_worker_ownership,
    verify_job_parallel_selection,
    verify_stage052_campaign_lock,
    verify_stage052_evidence_input,
    verify_stage052_prerequisite,
    verify_stage052_runtime_identity,
    verify_stage052_source_snapshot,
    verify_stage052_storage_root_binding,
)


def _bind_successful_review_execution(review_manifest: Path) -> None:
    raw_dir = review_manifest.parent.parent
    (review_manifest.parent / "review_execution.json").write_text(
        json.dumps(
            {
                "run_label": raw_dir.name,
                "finalized": True,
                "status": "completed",
                "systemd_service_result": "success",
                "cgroup_memory_peak_status": "verified",
                "raw_manifest_unchanged": True,
                "review_manifest_sha256": hashlib.sha256(review_manifest.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_source_snapshot_requires_clean_ext4_and_read_only_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    tracked = source / "tracked.txt"
    tracked.write_text("source", encoding="utf-8")
    subprocess.run(("git", "init", str(source)), check=True, capture_output=True)
    subprocess.run(("git", "-C", str(source), "add", "tracked.txt"), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Stage052 Test",
            "-c",
            "user.email=stage052@example.invalid",
            "commit",
            "-m",
            "snapshot",
        ),
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(
        stage052_evidence,
        "_findmnt_identity",
        lambda _path: {
            "source": "/dev/test",
            "filesystem": "ext4",
            "uuid": "test-uuid",
            "target": str(source),
        },
    )
    configs = source / "configs"
    configs.mkdir()
    lock = configs / "stage052_campaign_lock.local.json"
    lock_sidecar = configs / "stage052_campaign_lock.local.sha256"
    resource_contract = configs / "stage052_resource_calibration.local.json"
    resource_sidecar = configs / "stage052_resource_calibration.local.sha256"
    lock.write_text('{"schema_version":"test"}\n', encoding="utf-8")
    lock_sidecar.write_text(hashlib.sha256(lock.read_bytes()).hexdigest() + "\n")
    resource_contract.write_text('{"schema_version":"test-resource"}\n', encoding="utf-8")
    resource_sidecar.write_text(
        hashlib.sha256(resource_contract.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    lock.chmod(0o444)
    lock_sidecar.chmod(0o444)
    resource_contract.chmod(0o444)
    resource_sidecar.chmod(0o444)
    configs.chmod(0o555)
    tracked.chmod(0o444)
    source.chmod(0o555)

    observed = verify_stage052_source_snapshot(source)

    assert observed["read_only"] is True
    assert observed["tracked_file_count"] == 1
    assert set(observed["allowed_untracked_sha256"]) == {
        "configs/stage052_campaign_lock.local.json",
        "configs/stage052_campaign_lock.local.sha256",
        "configs/stage052_resource_calibration.local.json",
        "configs/stage052_resource_calibration.local.sha256",
    }
    source.chmod(0o755)
    injected = source / "sitecustomize.py"
    injected.write_text("raise RuntimeError('injected')\n", encoding="utf-8")
    injected.chmod(0o444)
    source.chmod(0o555)
    with pytest.raises(RuntimeError, match="unregistered untracked file"):
        verify_stage052_source_snapshot(source)
    source.chmod(0o755)
    injected.unlink()
    with pytest.raises(RuntimeError, match="writable tracked paths"):
        verify_stage052_source_snapshot(source)


def test_source_snapshot_contract_ignores_device_and_path_telemetry() -> None:
    frozen = {
        "repository_revision": "a" * 40,
        "mount": {
            "source": "/dev/sdd",
            "filesystem": "ext4",
            "uuid": "old-uuid",
            "target": "/sealed/old",
        },
        "tracked_file_count": 866,
        "allowed_untracked_sha256": {"data/schneider/c101.txt": "b" * 64},
        "read_only": True,
    }
    live = {
        **frozen,
        "mount": {
            "source": "/dev/sde",
            "filesystem": "ext4",
            "uuid": "new-uuid",
            "target": "/sealed/new",
        },
    }

    assert stage052_source_snapshot_contract(frozen) == (stage052_source_snapshot_contract(live))
    for field, value in (
        ("repository_revision", "c" * 40),
        ("tracked_file_count", 865),
        ("allowed_untracked_sha256", {"data/schneider/c101.txt": "d" * 64}),
    ):
        drifted = {**live, field: value}
        assert stage052_source_snapshot_contract(drifted) != (
            stage052_source_snapshot_contract(frozen)
        )
    invalid_filesystem = {
        **live,
        "mount": {**live["mount"], "filesystem": "9p"},
    }
    with pytest.raises(RuntimeError, match="hard contract is invalid"):
        stage052_source_snapshot_contract(invalid_filesystem)


def test_stage052_clean_check_defers_untracked_files_to_source_snapshot(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    tracked = repository / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(("git", "init", str(repository)), check=True, capture_output=True)
    subprocess.run(("git", "-C", str(repository), "add", "tracked.txt"), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Stage052 Test",
            "-c",
            "user.email=stage052@example.invalid",
            "commit",
            "-m",
            "snapshot",
        ),
        check=True,
        capture_output=True,
    )
    (repository / "local-only.json").write_text("{}\n", encoding="utf-8")

    _require_clean_stage052_repository(repository)

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean repository"):
        _require_clean_stage052_repository(repository)


def test_windows_command_output_uses_explicit_utf_encodings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, ...]] = []

    def run(
        arguments: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        observed.append(arguments)
        output = (
            "Windows 11 专业版".encode()
            if arguments[0] == "powershell.exe"
            else "WSL 版本: 2.9.3.0".encode("utf-16-le")
        )
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr=b"")

    monkeypatch.setattr(stage052_evidence.subprocess, "run", run)

    assert (
        stage052_evidence._run_command(("powershell.exe", "-Command", "Write-Output test"))
        == "Windows 11 专业版"
    )
    assert stage052_evidence._run_command(("wsl.exe", "--version")) == "WSL 版本: 2.9.3.0"
    assert "OutputEncoding" in observed[0][-1]


def test_persistence_attribution_recomputes_monotonic_control_intervals() -> None:
    attribution = Stage052PersistenceAttribution(
        run_label="stage05.2_artifact_streaming_attempt05",
        component="artifact_streaming",
        scope="performance",
        subject_id="run",
        primary_manifest_relative_path="control/manifest.json",
        primary_manifest_sha256="a" * 64,
        solver_seconds=8.0,
        shard_persistence_seconds=1.0,
        control_intervals=(
            PersistenceInterval("write_control", 10, 500_000_010),
            PersistenceInterval("finalize", 600_000_010, 1_100_000_010),
        ),
    )

    replayed = Stage052PersistenceAttribution.from_dict(attribution.to_dict())

    assert replayed.control_persistence_seconds == pytest.approx(1.0)
    assert replayed.total_persistence_seconds == pytest.approx(2.0)
    assert replayed.persistence_ratio == pytest.approx(0.2)
    assert attribution.to_dict()["maximum_persistence_ratio"] == pytest.approx(0.36)
    assert pytest.approx(0.36) == STAGE052_MAXIMUM_PERSISTENCE_RATIO


def test_persistence_attribution_keeps_v1_threshold_metadata_readable() -> None:
    attribution = Stage052PersistenceAttribution(
        run_label="stage05.2_native_kernels_attempt15",
        component="native_kernels",
        scope="performance",
        subject_id="run",
        primary_manifest_relative_path="control/manifest.json",
        primary_manifest_sha256="a" * 64,
        solver_seconds=65.0,
        shard_persistence_seconds=35.0,
        control_intervals=(),
    )
    legacy_payload = attribution.to_dict()
    legacy_payload["schema_version"] = "stage05.2-persistence-attribution-v1"
    legacy_payload["maximum_persistence_ratio"] = 0.30

    replayed = Stage052PersistenceAttribution.from_dict(legacy_payload)

    assert replayed.persistence_ratio == pytest.approx(0.35)


def test_batch_persistence_envelope_uses_v2_and_reads_v1_threshold_metadata() -> None:
    envelope = BatchPersistenceEnvelope(
        run_label="stage05.2_benchmark_attempt01",
        batch_id="batch0001",
        base_attribution_sha256="a" * 64,
        verified_manifest_sha256="b" * 64,
        archived_manifest_sha256="c" * 64,
        solver_seconds=65.0,
        base_persistence_seconds=35.0,
        state_intervals=(
            PersistenceInterval("verified_batch_manifest_write", 1, 1),
            PersistenceInterval("archived_batch_manifest_write", 2, 2),
        ),
    )

    current_payload = envelope.to_dict()
    assert current_payload["schema_version"] == ("stage05.2-batch-persistence-envelope-v2")
    assert current_payload["maximum_persistence_ratio"] == pytest.approx(0.36)
    assert BatchPersistenceEnvelope.from_dict(current_payload).persistence_ratio == (
        pytest.approx(0.35)
    )

    legacy_payload = dict(current_payload)
    legacy_payload["schema_version"] = "stage05.2-batch-persistence-envelope-v1"
    legacy_payload["maximum_persistence_ratio"] = 0.30
    assert BatchPersistenceEnvelope.from_dict(legacy_payload).persistence_ratio == (
        pytest.approx(0.35)
    )


def test_persistence_attribution_rejects_overlapping_control_intervals() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Stage052PersistenceAttribution(
            run_label="stage05.2_artifact_streaming_attempt05",
            component="artifact_streaming",
            scope="performance",
            subject_id="run",
            primary_manifest_relative_path="control/manifest.json",
            primary_manifest_sha256="a" * 64,
            solver_seconds=8.0,
            shard_persistence_seconds=1.0,
            control_intervals=(
                PersistenceInterval("first", 10, 20),
                PersistenceInterval("second", 19, 30),
            ),
        )


def _observation(
    instance: str,
    seed: int,
    seconds: float,
    *,
    customer_count: int = 100,
    semantic_digest: str = "same",
) -> PerformanceObservation:
    return PerformanceObservation(
        instance=instance,
        seed=seed,
        customer_count=customer_count,
        end_to_end_seconds=seconds,
        semantic_digest=semantic_digest,
    )


def test_stage052_components_have_one_strict_order() -> None:
    assert tuple(component.value for component in Stage052Component) == (
        "perf_baseline",
        "hot_path",
        "artifact_streaming",
        "job_parallel",
        "native_kernels",
        "accelerator_pilot",
        "benchmark",
    )


def test_stage052_contract_requires_accelerator_selection_then_new_pilot() -> None:
    pilot = stage052_contract(Stage052Component.BENCHMARK, "pilot")
    formal = stage052_contract(Stage052Component.BENCHMARK, "formal")

    assert pilot.prerequisite_component is Stage052Component.ACCELERATOR_PILOT
    assert pilot.prerequisite_status == "READY_FOR_STAGE052_BENCHMARK"
    assert pilot.prerequisites[0].role == "accelerator_selection"
    assert pilot.prerequisites[0].exact_run_label is None
    assert pilot.next_status == "READY_FOR_STAGE052_FORMAL_BENCHMARK"
    assert formal.prerequisite_component is Stage052Component.BENCHMARK
    assert formal.prerequisite_status == "READY_FOR_STAGE052_FORMAL_BENCHMARK"
    assert formal.next_status == "READY_FOR_STAGE05_3"
    assert pilot.required_backend == formal.required_backend == "cpu_batch"
    assert pilot.storage_policy_version == formal.storage_policy_version == ("artifact-storage-v2")
    assert (
        pilot.screening_schema_version
        == formal.screening_schema_version
        == ("screening_decisions_v3")
    )


def test_stage052_storage_amendment_contract_binds_current_predecessor() -> None:
    contract = stage052_contract(Stage052Component.ARTIFACT_STREAMING, "performance")
    requirements = {requirement.role: requirement for requirement in contract.prerequisites}

    assert tuple(requirements) == ("performance_baseline", "hot_path_predecessor")
    predecessor = requirements["hot_path_predecessor"]
    assert predecessor.exact_run_label is None
    assert predecessor.allowed_statuses == ("READY_FOR_STAGE052_ARTIFACT_STREAMING",)
    assert predecessor.requires_current_chain_identity


def test_stage052_campaign_lock_binds_exact_raw_review_runtime_identity(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "stage05.2_perf_baseline_attempt05"
    config = tmp_path / "stage052.toml"
    config.write_text("[stage05_2]\nschema_version='test'\n", encoding="utf-8")
    writer = ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext("stage05.2", "perf_baseline", raw_dir.name),
        ArtifactStorageConfig(),
    )
    writer.write_control(
        metadata={
            "run_label": raw_dir.name,
            "component": "perf_baseline",
            "scope": "performance",
            "runtime_identity": {"machine": "windows-wsl2-test"},
        },
        configuration_path=config,
    )
    writer.finalize()
    identity = Stage052PrerequisiteIdentity(
        run_label=raw_dir.name,
        component="perf_baseline",
        status="READY_FOR_STAGE052_HOT_PATH",
        repository_revision="a" * 40,
        configuration_sha256="b" * 64,
        raw_manifest_sha256="c" * 64,
        review_manifest_sha256="d" * 64,
    )
    lock_path = tmp_path / "campaign-lock.json"

    upsert_stage052_campaign_lock(lock_path, raw_dir=raw_dir, identity=identity)
    verify_stage052_campaign_lock(lock_path, raw_dir=raw_dir, identity=identity)

    changed = replace(identity, review_manifest_sha256="e" * 64)
    with pytest.raises(ArtifactIntegrityError, match="does not bind exact prerequisite"):
        verify_stage052_campaign_lock(lock_path, raw_dir=raw_dir, identity=changed)


def test_stage052_current_chain_prerequisites_reject_historical_physical_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contracts = (
        stage052_contract(Stage052Component.JOB_PARALLEL, "performance"),
        stage052_contract(Stage052Component.NATIVE_KERNELS, "performance"),
        stage052_contract(Stage052Component.ACCELERATOR_PILOT, "performance"),
        stage052_contract(Stage052Component.BENCHMARK, "pilot"),
        stage052_contract(Stage052Component.BENCHMARK, "formal"),
    )
    assert all(contract.prerequisites[0].requires_current_chain_identity for contract in contracts)
    requirement = contracts[0].prerequisites[0]
    identity = Stage052PrerequisiteIdentity(
        run_label="stage05.2_artifact_streaming_attempt04",
        component="artifact_streaming",
        status="READY_FOR_STAGE052_JOB_PARALLEL",
        repository_revision="a" * 40,
        configuration_sha256="b" * 64,
        raw_manifest_sha256="c" * 64,
        review_manifest_sha256="d" * 64,
        scope="performance",
    )

    monkeypatch.setattr(
        stage052_evidence,
        "verify_stage052_prerequisite",
        lambda *_args, **_kwargs: identity,
    )
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    (review_dir / "review_manifest.json").write_text("{}\n", encoding="utf-8")
    immutable_review = f"generations/{'e' * 64}/review_report.md"
    monkeypatch.setattr(
        stage052_evidence,
        "verify_stage052_review_files",
        lambda *_args, **_kwargs: {
            immutable_review: tmp_path / immutable_review,
            immutable_review.replace("review_report.md", "review_findings.csv"): tmp_path
            / "review_findings.csv",
            immutable_review.replace("review_report.md", "semantic_mismatches.csv"): tmp_path
            / "semantic_mismatches.csv",
        },
    )

    class HistoricalReader:
        manifest = {
            "artifacts": [
                {
                    "artifact_type": "manifest_metadata",
                    "relative_path": "control/metadata.json",
                }
            ]
        }

        def __init__(self, _path: Path) -> None:
            pass

        def read_json(self, _relative_path: str) -> dict[str, object]:
            return {
                "backend": "cpu_batch",
                "storage_policy_version": "artifact-storage-v2",
                "screening_schema_version": "screening_decisions_v2",
            }

    monkeypatch.setattr(stage052_evidence, "ArtifactReader", HistoricalReader)

    with pytest.raises(ArtifactIntegrityError, match="historical evidence"):
        verify_stage052_evidence_input(tmp_path, requirement)


def test_current_chain_campaign_uses_complete_campaign_review_generation() -> None:
    requirement = stage052_contract(Stage052Component.BENCHMARK, "formal").prerequisites[0]
    generation = "a" * 64
    campaign_files = {
        f"generations/{generation}/{name}": Path(name)
        for name in (
            "anytime_summary.csv",
            "budget_summary.csv",
            "failure_analysis.csv",
            "family_summary.csv",
            "gpu_decision.json",
            "per_run_results.csv",
            "performance_gates.csv",
            "persistence_summary.csv",
            "resource_summary.csv",
            "review_findings.csv",
            "review_report.md",
        )
    }

    stage052_evidence._verify_current_chain_review_file_surface(
        campaign_files,
        requirement,
    )


def test_current_chain_prerequisite_replays_frozen_producer_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requirement = stage052_contract(Stage052Component.JOB_PARALLEL, "performance").prerequisites[0]
    identity = Stage052PrerequisiteIdentity(
        run_label="stage05.2_artifact_streaming_attempt04",
        component="artifact_streaming",
        status="READY_FOR_STAGE052_JOB_PARALLEL",
        repository_revision="a" * 40,
        configuration_sha256="b" * 64,
        raw_manifest_sha256="c" * 64,
        review_manifest_sha256="d" * 64,
        scope="performance",
    )
    frozen_runtime = {"wheel_sha256": "e" * 64}
    monkeypatch.setattr(
        stage052_evidence,
        "verify_stage052_prerequisite",
        lambda *_args, **_kwargs: identity,
    )
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    (review_dir / "review_manifest.json").write_text("{}\n", encoding="utf-8")
    generation = f"generations/{'f' * 64}"
    monkeypatch.setattr(
        stage052_evidence,
        "verify_stage052_review_files",
        lambda *_args, **_kwargs: {
            f"{generation}/review_report.md": tmp_path / "review_report.md",
            f"{generation}/review_findings.csv": tmp_path / "review_findings.csv",
            f"{generation}/semantic_mismatches.csv": tmp_path / "semantic_mismatches.csv",
        },
    )

    class CurrentReader:
        manifest = {
            "artifacts": [
                {
                    "artifact_type": "manifest_metadata",
                    "relative_path": "control/metadata.json",
                }
            ]
        }

        def __init__(self, _path: Path) -> None:
            pass

        def read_json(self, _relative_path: str) -> dict[str, object]:
            return {
                "backend": "cpu_batch",
                "storage_policy_version": "artifact-storage-v2",
                "screening_schema_version": "screening_decisions_v3",
                "runtime_identity": frozen_runtime,
                "staging_root": {"alias": "wsl_staging"},
            }

    monkeypatch.setattr(stage052_evidence, "ArtifactReader", CurrentReader)
    current_root = tmp_path / "current"
    current_root.mkdir()
    producer_root = tmp_path / "producer"
    producer_root.mkdir()
    monkeypatch.setattr(stage052_evidence, "repository_root", lambda: current_root)
    monkeypatch.setattr(
        stage052_evidence,
        "verify_stage052_review_execution_receipt",
        lambda *_args, **_kwargs: {
            "working_directory": str(producer_root),
            "producer_repository_revision": "a" * 40,
        },
    )
    replayed: list[tuple[Path, str]] = []

    def replay_runtime(root: Path, revision: str) -> dict[str, object]:
        replayed.append((root, revision))
        return frozen_runtime

    monkeypatch.setattr(
        stage052_evidence,
        "verify_frozen_stage052_producer_runtime_identity",
        replay_runtime,
    )
    monkeypatch.setattr(
        stage052_evidence,
        "verify_stage052_storage_root_binding",
        lambda *_args, **_kwargs: None,
    )

    assert verify_stage052_evidence_input(tmp_path, requirement) == identity
    assert replayed == [(producer_root, "a" * 40)]


def test_current_chain_prerequisite_rejects_receipt_revision_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requirement = stage052_contract(Stage052Component.JOB_PARALLEL, "performance").prerequisites[0]
    identity = Stage052PrerequisiteIdentity(
        run_label="stage05.2_artifact_streaming_attempt04",
        component="artifact_streaming",
        status="READY_FOR_STAGE052_JOB_PARALLEL",
        repository_revision="a" * 40,
        configuration_sha256="b" * 64,
        raw_manifest_sha256="c" * 64,
        review_manifest_sha256="d" * 64,
        scope="performance",
    )
    monkeypatch.setattr(
        stage052_evidence,
        "verify_stage052_prerequisite",
        lambda *_args, **_kwargs: identity,
    )
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    (review_dir / "review_manifest.json").write_text("{}\n", encoding="utf-8")
    generation = f"generations/{'f' * 64}"
    monkeypatch.setattr(
        stage052_evidence,
        "verify_stage052_review_files",
        lambda *_args, **_kwargs: {
            f"{generation}/review_report.md": tmp_path / "review_report.md",
            f"{generation}/review_findings.csv": tmp_path / "review_findings.csv",
            f"{generation}/semantic_mismatches.csv": tmp_path / "semantic_mismatches.csv",
        },
    )

    class CurrentReader:
        manifest = {
            "artifacts": [
                {
                    "artifact_type": "manifest_metadata",
                    "relative_path": "control/metadata.json",
                }
            ]
        }

        def __init__(self, _path: Path) -> None:
            pass

        def read_json(self, _relative_path: str) -> dict[str, object]:
            return {
                "backend": "cpu_batch",
                "storage_policy_version": "artifact-storage-v2",
                "screening_schema_version": "screening_decisions_v3",
                "runtime_identity": {"wheel_sha256": "e" * 64},
                "staging_root": {"alias": "wsl_staging"},
            }

    monkeypatch.setattr(stage052_evidence, "ArtifactReader", CurrentReader)
    monkeypatch.setattr(
        stage052_evidence,
        "verify_stage052_review_execution_receipt",
        lambda *_args, **_kwargs: {
            "working_directory": str(tmp_path),
            "producer_repository_revision": "0" * 40,
        },
    )

    with pytest.raises(ArtifactIntegrityError, match="receipt revision"):
        verify_stage052_evidence_input(tmp_path, requirement)


def test_stage052_storage_root_binding_is_path_free_and_alias_scoped(
    tmp_path: Path,
) -> None:
    locator_path = tmp_path / "stage052_storage_roots.local.toml"
    locator_path.write_text(
        """
[roots.transfer_staging]
absolute_path = "/Volumes/TRANSFER/project/results"
device_uuid = "transfer-uuid"
filesystem = "ExFAT"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "staging_root": stage052_storage_root_binding(
            alias="transfer_staging",
            volume={"device_uuid": "transfer-uuid", "filesystem": "ExFAT"},
        )
    }

    binding = verify_stage052_storage_root_binding(
        metadata,
        locator_path=locator_path,
        expected_alias="transfer_staging",
    )

    assert binding == metadata["staging_root"]
    assert "/Volumes/TRANSFER" not in json.dumps(binding, sort_keys=True)


@pytest.mark.parametrize(
    "staging_root",
    (
        None,
        {
            "schema_version": "stage05.2-storage-root-binding-v1",
            "alias": "transfer_staging",
            "volume": {
                "device_uuid": "transfer-uuid",
                "filesystem": "ExFAT",
                "absolute_path": "/Volumes/TRANSFER/project/results",
            },
        },
    ),
)
def test_stage052_storage_root_binding_rejects_missing_or_pathful_identity(
    staging_root: object,
    tmp_path: Path,
) -> None:
    locator_path = tmp_path / "stage052_storage_roots.local.toml"
    locator_path.write_text(
        """
[roots.transfer_staging]
absolute_path = "/Volumes/TRANSFER/project/results"
device_uuid = "transfer-uuid"
filesystem = "ExFAT"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError, match="staging root"):
        verify_stage052_storage_root_binding(
            {"staging_root": staging_root},
            locator_path=locator_path,
            expected_alias="transfer_staging",
        )


def test_stage052_storage_root_binding_treats_device_change_as_telemetry(
    tmp_path: Path,
) -> None:
    locator_path = tmp_path / "stage052_storage_roots.local.toml"
    locator_path.write_text(
        """
[roots.transfer_staging]
absolute_path = "/Volumes/TRANSFER/project/results"
device_uuid = "configured-device"
filesystem = "ExFAT"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    observed = stage052_storage_root_binding(
        alias="transfer_staging",
        volume={"device_uuid": "observed-device", "filesystem": "APFS"},
    )

    assert (
        verify_stage052_storage_root_binding(
            {"staging_root": observed},
            locator_path=locator_path,
            expected_alias="transfer_staging",
        )
        == observed
    )


def test_performance_producer_verifies_exact_staging_path_and_live_volume(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    results = root / "results"
    results.mkdir(parents=True)
    locator_path = root / "configs" / "stage052_storage_roots.local.toml"
    locator_path.parent.mkdir()
    locator_path.write_text(
        f"""
[roots.wsl_staging]
absolute_path = "{results}"
device_uuid = "ext4-uuid"
filesystem = "ext4"
[roots.d_archive]
absolute_path = "{tmp_path / "archive"}"
device_uuid = "d-nvme"
filesystem = "9p"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    binding = _verify_performance_staging_root(
        root=root,
        locator_path=locator_path,
        staging_alias="wsl_staging",
        output_dir=results / "stage05.2_artifact_streaming_attempt05",
        volume_probe=lambda _path: VolumeIdentity("ext4-uuid", "ext4"),
    )

    assert binding["alias"] == "wsl_staging"
    assert "absolute_path" not in json.dumps(binding)


def test_performance_producer_records_observed_staging_volume_as_telemetry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    results = root / "results"
    results.mkdir(parents=True)
    locator_path = root / "storage.local.toml"
    locator_path.write_text(
        f"""
[roots.wsl_staging]
absolute_path = "{results}"
device_uuid = "ext4-uuid"
filesystem = "ext4"
[roots.d_archive]
absolute_path = "{tmp_path / "archive"}"
device_uuid = "d-nvme"
filesystem = "9p"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    binding = _verify_performance_staging_root(
        root=root,
        locator_path=locator_path,
        staging_alias="wsl_staging",
        output_dir=results / "stage05.2_job_parallel_attempt07",
        volume_probe=lambda _path: VolumeIdentity("internal-uuid", "APFS"),
    )

    assert binding["volume"] == {
        "device_uuid": "internal-uuid",
        "filesystem": "APFS",
    }


def test_performance_reviewer_independently_reprobes_staging_volume(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    staging = tmp_path / "external-staging"
    staging.mkdir(parents=True)
    locator_path = root / "storage.local.toml"
    root.mkdir()
    raw_dir = staging / "stage05.2_artifact_streaming_attempt99"
    raw_dir.mkdir()
    locator_path.write_text(
        f"""
[roots.transfer_staging]
absolute_path = "{staging}"
device_uuid = "transfer-uuid"
filesystem = "ExFAT"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "staging_root": stage052_storage_root_binding(
            alias="transfer_staging",
            volume={"device_uuid": "transfer-uuid", "filesystem": "ExFAT"},
        )
    }

    passed, detail = _validate_stage052_staging_root_identity(
        metadata,
        raw_dir=raw_dir,
        root=root,
        locator_path=locator_path,
        expected_alias="transfer_staging",
        volume_probe=lambda _path: VolumeIdentity("transfer-uuid", "ExFAT"),
    )
    changed, changed_detail = _validate_stage052_staging_root_identity(
        metadata,
        raw_dir=raw_dir,
        root=root,
        locator_path=locator_path,
        expected_alias="transfer_staging",
        volume_probe=lambda _path: VolumeIdentity("other-uuid", "APFS"),
    )

    assert passed
    assert "transfer_staging" in detail
    assert changed
    assert "current telemetry=other-uuid/APFS" in changed_detail


def test_current_staging_review_rejects_wrong_path_filesystem_and_aliases(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    staging = tmp_path / "stage052-active"
    archive = tmp_path / "archive"
    outside = tmp_path / "outside" / "stage05.2_perf_baseline_attempt99"
    root.mkdir()
    staging.mkdir()
    archive.mkdir()
    outside.mkdir(parents=True)
    raw_dir = staging / "stage05.2_perf_baseline_attempt99"
    raw_dir.mkdir()
    locator_path = root / "storage.local.toml"
    metadata = {
        "staging_root": stage052_storage_root_binding(
            alias="wsl_staging",
            volume={"device_uuid": "ext4-uuid", "filesystem": "ext4"},
        )
    }

    def write_locator(*, filesystem: str = "ext4", extra_alias: bool = False) -> None:
        extra_root = (
            '[roots.usb]\nabsolute_path = "/mnt/e"\ndevice_uuid = "usb"\nfilesystem = "ExFAT"'
            if extra_alias
            else ""
        )
        locator_path.write_text(
            f"""
[roots.wsl_staging]
absolute_path = "{staging}"
device_uuid = "ext4-uuid"
filesystem = "{filesystem}"

[roots.d_archive]
absolute_path = "{archive}"
device_uuid = "d-nvme"
filesystem = "9p"
{extra_root}
""".strip()
            + "\n",
            encoding="utf-8",
        )

    def probe(_path: Path) -> VolumeIdentity:
        return VolumeIdentity("ext4-uuid", "ext4")

    write_locator()
    passed, _ = _validate_stage052_staging_root_identity(
        metadata,
        raw_dir=raw_dir,
        root=root,
        locator_path=locator_path,
        volume_probe=probe,
    )
    wrong_path, wrong_path_detail = _validate_stage052_staging_root_identity(
        metadata,
        raw_dir=outside,
        root=root,
        locator_path=locator_path,
        volume_probe=probe,
    )
    write_locator(filesystem="ExFAT")
    exfat_metadata = {
        "staging_root": stage052_storage_root_binding(
            alias="wsl_staging",
            volume={"device_uuid": "ext4-uuid", "filesystem": "ExFAT"},
        )
    }
    wrong_filesystem, wrong_filesystem_detail = _validate_stage052_staging_root_identity(
        exfat_metadata,
        raw_dir=raw_dir,
        root=root,
        locator_path=locator_path,
        volume_probe=lambda _path: VolumeIdentity("ext4-uuid", "ExFAT"),
    )
    write_locator(extra_alias=True)
    wrong_aliases, wrong_aliases_detail = _validate_stage052_staging_root_identity(
        metadata,
        raw_dir=raw_dir,
        root=root,
        locator_path=locator_path,
        volume_probe=probe,
    )

    assert passed
    assert not wrong_path
    assert "outside" in wrong_path_detail
    assert not wrong_filesystem
    assert "not ext4" in wrong_filesystem_detail
    assert not wrong_aliases
    assert "aliases" in wrong_aliases_detail


def test_every_primary_write_review_binds_the_attribution_envelope(tmp_path: Path) -> None:
    raw_dir = tmp_path / "stage05.2_perf_baseline_attempt99"
    config = tmp_path / "stage052.toml"
    config.write_text("[stage05_2]\nschema_version='test'\n", encoding="utf-8")
    writer = ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext("stage05.2", "perf_baseline", raw_dir.name),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v1"),
    )
    writer.write_control(
        metadata={
            "run_label": raw_dir.name,
            "component": "perf_baseline",
            "scope": "performance",
            "persistence_attribution": "primary_active_writes_v1",
        },
        configuration_path=config,
    )
    bundle = writer.finalize()
    attribution = raw_dir / "control" / f"{raw_dir.name}_persistence_attribution.json"
    sidecar = attribution.with_suffix(".sha256")
    labels = (
        "parent_write_control",
        "parent_timing_and_per_run_control",
        "parent_resource_control",
        "parent_primary_manifest_finalize",
    )
    attribution_record = Stage052PersistenceAttribution(
        run_label=raw_dir.name,
        component="perf_baseline",
        scope="performance",
        subject_id="run",
        primary_manifest_relative_path=bundle.manifest_path.relative_to(raw_dir).as_posix(),
        primary_manifest_sha256=hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest(),
        solver_seconds=1.0,
        shard_persistence_seconds=0.1,
        control_intervals=tuple(
            PersistenceInterval(label, index * 2, index * 2 + 1)
            for index, label in enumerate(labels)
        ),
    )
    attribution.write_text(
        json.dumps(attribution_record.to_dict()) + "\n",
        encoding="utf-8",
    )
    sidecar.write_text(
        hashlib.sha256(attribution.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    review_manifest: dict[str, object] = {}
    metadata = {
        "component": "perf_baseline",
        "scope": "performance",
        "persistence_attribution": "primary_active_writes_v1",
    }

    _bind_persistence_attribution_review(
        review_manifest,
        raw_dir=raw_dir,
        metadata=metadata,
    )

    assert (
        review_manifest["persistence_attribution_sha256"]
        == hashlib.sha256(attribution.read_bytes()).hexdigest()
    )
    assert (
        review_manifest["persistence_attribution_sidecar_sha256"]
        == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    )

    sidecar.write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="envelope is missing"):
        _bind_persistence_attribution_review({}, raw_dir=raw_dir, metadata=metadata)

    wrong_identity = replace(attribution_record, component="hot_path")
    attribution.write_text(
        json.dumps(wrong_identity.to_dict()) + "\n",
        encoding="utf-8",
    )
    sidecar.write_text(
        hashlib.sha256(attribution.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactIntegrityError, match="identity is invalid"):
        _bind_persistence_attribution_review({}, raw_dir=raw_dir, metadata=metadata)


def test_stage052_current_chain_rejects_old_not_ready_remediation_input(
    tmp_path: Path,
) -> None:
    requirement = stage052_contract(
        Stage052Component.ARTIFACT_STREAMING, "performance"
    ).prerequisites[-1]
    raw_dir = tmp_path / "stage05.2_native_kernels_attempt03"
    config = tmp_path / "stage052.toml"
    config.write_text("[stage05_2]\nschema_version='test'\n", encoding="utf-8")
    config_digest = hashlib.sha256(config.read_bytes()).hexdigest()
    writer = ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext("stage05.2", "native_kernels", raw_dir.name),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    writer.write_control(
        metadata={
            "run_label": raw_dir.name,
            "component": "native_kernels",
            "scope": "performance",
            "repository_dirty": False,
            "repository_revision": "a" * 40,
            "configuration_sha256": config_digest,
        },
        configuration_path=config,
    )
    bundle = writer.finalize()
    review_dir = raw_dir / "review"
    review_dir.mkdir()
    report = review_dir / "review_report.md"
    findings = review_dir / "review_findings.csv"
    report.write_text("NOT_READY: persistence exceeded 30%\n", encoding="utf-8")
    findings.write_text("gate,passed\npersistence_ratio,False\n", encoding="utf-8")
    review_manifest = review_dir / "review_manifest.json"
    review_manifest.write_text(
        json.dumps(
            {
                "schema_version": "stage05.2-review-v1",
                "run_label": raw_dir.name,
                "component": "native_kernels",
                "scope": "performance",
                "status": "NOT_READY",
                "raw_manifest_sha256": hashlib.sha256(
                    bundle.manifest_path.read_bytes()
                ).hexdigest(),
                "gates": {"persistence_ratio": {"passed": False}},
                "files": {
                    findings.name: hashlib.sha256(findings.read_bytes()).hexdigest(),
                    report.name: hashlib.sha256(report.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError, match="component mismatch"):
        verify_stage052_evidence_input(raw_dir, requirement)


def test_artifact_persistence_gate_rejects_the_measured_e03_ratio() -> None:
    decision = evaluate_artifact_persistence(
        (
            ArtifactPersistenceObservation(
                solver_seconds=280.766185035,
                artifact_persistence_seconds=282.621038503,
            ),
        )
    )

    assert decision.ratio == pytest.approx(0.5016461621692021)
    assert not decision.passed
    assert "exceeds 36%" in decision.detail


def test_artifact_persistence_gate_uses_verified_solver_plus_new_persistence() -> None:
    decision = evaluate_artifact_persistence(
        (
            ArtifactPersistenceObservation(
                solver_seconds=280.766185035,
                artifact_persistence_seconds=100.0,
            ),
        )
    )

    assert decision.ratio == pytest.approx(100.0 / (280.766185035 + 100.0))
    assert decision.passed


def test_artifact_persistence_gate_uses_the_36_percent_boundary() -> None:
    passing = evaluate_artifact_persistence((ArtifactPersistenceObservation(65.0, 35.0),))
    failing = evaluate_artifact_persistence((ArtifactPersistenceObservation(63.0, 37.0),))

    assert passing.passed
    assert passing.maximum_ratio == pytest.approx(0.36)
    assert not failing.passed
    assert failing.maximum_ratio == pytest.approx(0.36)


def test_native_kernel_config_is_explicit_complete_and_serializable() -> None:
    config = NativeKernelConfig()
    assert config.to_dict() == {
        "enabled": True,
        "exact_charging": True,
        "screening": True,
        "propagation": True,
        "distance_matrix": True,
        "abi_version": NATIVE_KERNEL_ABI_VERSION,
        "context_policy": "pack_once_per_solve",
        "failure_policy": "fail_fast_no_fallback",
    }
    with pytest.raises(ValueError, match="pass None"):
        NativeKernelConfig(enabled=False)
    with pytest.raises(ValueError, match="complete native kernel set"):
        NativeKernelConfig(screening=False)
    with pytest.raises(ValueError, match="ABI"):
        NativeKernelConfig(abi_version="stale")
    with pytest.raises(ValueError, match="must not fall back"):
        NativeKernelConfig(failure_policy="fallback")


def test_stage052_config_declares_the_complete_native_kernel_profile() -> None:
    config = load_stage052_config(Path("configs/stage052_performance.toml"))
    assert config.native_kernels == NativeKernelConfig()
    assert config.v2_storage.screening_schema_version == "screening_decisions_v3"
    assert config.runtime_identity_manifest == Path("configs/stage052_runtime_identity.local.json")


def test_native_execution_audit_cross_checks_per_run_raw_and_trace(tmp_path: Path) -> None:
    run_label = "stage05.2_native_kernels_attempt99"
    raw_dir = tmp_path / run_label
    writer = ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext("stage05.2", "native_kernels", run_label),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    writer.write_control(metadata={"run_label": run_label})
    shard = raw_dir / "c101_21" / "2014"
    shard.mkdir(parents=True)
    backend = {
        "native_invocations": 2,
        "work_batches": 2,
        "batch_launches": 2,
        "exact_calls": 5,
        "completed_calls": 5,
        "launch_occupancies": [1, 4],
        "native_fallbacks": 0,
        "native_kernel_seconds": 0.1,
    }
    screening = {
        "screening_calls": 10,
        "screening_cache_hits": 2,
        "native_screening_invocations": 8,
        "native_screening_seconds": 0.2,
        "native_propagation_invocations": 3,
        "native_propagation_seconds": 0.05,
        "native_protocol_fallbacks": 0,
    }
    incremental = {"incremental_propagations": 2, "incremental_fallbacks": 1}
    raw_path = shard / f"{run_label}_raw_c101_21_2014.json"
    raw_path.write_text(
        json.dumps(
            {
                "run_label": run_label,
                "component": "native_kernels",
                "scope": "performance",
                "instance": "c101_21",
                "seed": 2014,
                "axes": {
                    "fixed_work": {
                        "backend_metrics": backend,
                        "objective_key": [1, 0.0, 0.0, 0],
                        "started_calls": 5,
                        "completed_calls": 5,
                        "effective_iterations": 3,
                        "unique_route_semantics": "completed_cache_owner_identity_v2",
                        "termination_reason": "exact_call_budget_exhausted",
                        "validator_passed": True,
                        "valid": True,
                        "trace_reconciliation": {
                            "status": "pass",
                            "checks": {"calls": True},
                            "expected": {
                                "unique_route_semantics": ("completed_cache_owner_identity_v2")
                            },
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    solution_path = shard / f"{run_label}_solution_c101_21_2014.json"
    solution_path.write_text(
        json.dumps(
            {
                "instance": "c101_21",
                "seed": 2014,
                "axes": {
                    "fixed_work": {
                        "objective_key": [1, 0.0, 0.0, 0],
                        "feasible": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    trace_path = shard / f"{run_label}_trace_c101_21_2014.json"
    trace_path.write_text(
        json.dumps(
            {
                "event_identity": {"shard_ordinal": 3, "local_field": "event_id"},
                "axes": {
                    "fixed_work": {
                        "persistence_pipeline": {
                            "mode": "bounded_async_thread",
                            "queue_max_batches": 1,
                            "writer_thread_switch_interval_seconds": (
                                stage052_performance.STAGE052_WRITER_THREAD_SWITCH_INTERVAL_SECONDS
                            ),
                            "submitted_batches": 2,
                            "completed_batches": 2,
                            "writer_active_nanoseconds": 8_000_000,
                            "writer_cpu_nanoseconds": 5_000_000,
                            "producer_active_nanoseconds": 6_000_000,
                            "persistence_union_nanoseconds": 10_000_000,
                            "solver_persistence_union_nanoseconds": 10_000_000,
                            "solver_persistence_critical_path_nanoseconds": 6_000_000,
                            "solver_producer_active_nanoseconds": 6_000_000,
                            "solver_writer_cpu_nanoseconds": 5_000_000,
                            "producer_wait_nanoseconds": 2_000_000,
                            "peak_queued_batches": 1,
                            "batch_ledger": [
                                {
                                    "ordinal": ordinal,
                                    "row_count": 1,
                                    "event_token_sha256": "0" * 64,
                                }
                                for ordinal in range(2)
                            ],
                        },
                        "result_summary": {
                            "screening_statistics": screening,
                            "cache_incremental_statistics": incremental,
                            "exact_started_calls": 5,
                            "exact_completed_calls": 5,
                            "effective_iterations": 3,
                            "unique_route_semantics": ("completed_cache_owner_identity_v2"),
                            "termination_reason": "exact_call_budget_exhausted",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    timing_path = raw_dir / "control" / f"{run_label}_timing_evidence.json"
    timing_path.write_text(
        json.dumps(
            {
                "schema_version": "stage05.2-timing-evidence-v1",
                "run_label": run_label,
                "component": "native_kernels",
                "rows": [
                    {
                        "instance": "c101_21",
                        "seed": 2014,
                        "axis": "fixed_work",
                        "axis_started_ns": 1_000_000_000,
                        "solver_started_ns": 1_000_000_000,
                        "solver_completed_ns": 1_100_000_000,
                        "live_stream_persistence_ns": 10_000_000,
                        "solver_interleaved_persistence_ns": 10_000_000,
                        "axis_completed_ns": 1_120_000_000,
                        "finalize_started_ns": 3_000_000_000,
                        "finalize_completed_ns": 3_030_000_000,
                        "axis_event_count": 2,
                        "total_event_count": 2,
                        "axis_count": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    writer.record_existing_file(raw_path, artifact_type="raw")
    writer.record_existing_file(solution_path, artifact_type="solution")
    writer.record_existing_file(trace_path, artifact_type="trace")
    writer.record_existing_file(timing_path, artifact_type="timing_evidence")
    bundle = writer.finalize()
    row = {
        "instance": "c101_21",
        "seed": 2014,
        "axis": "fixed_work",
        "native_invocations": 2,
        "native_fallbacks": 0,
        "native_kernel_seconds": 0.1,
        "native_screening_invocations": 8,
        "native_screening_seconds": 0.2,
        "native_propagation_invocations": 3,
        "native_propagation_seconds": 0.05,
        "native_protocol_fallbacks": 0,
        "batch_launches": 2,
        "exact_started_calls": 5,
        "exact_completed_calls": 5,
        "median_batch_occupancy": 2.5,
        "solver_seconds": 0.09,
        "artifact_persistence_seconds": 0.04,
        "end_to_end_seconds": 0.15,
    }

    passed, detail = _audit_native_execution(raw_dir, [row])
    assert passed, detail

    forged = {**row, "native_screening_invocations": 9}
    passed, detail = _audit_native_execution(raw_dir, [forged])
    assert not passed
    assert "binding mismatch" in detail

    forged_timing = {**row, "end_to_end_seconds": 0.01}
    passed, detail = _audit_native_execution(raw_dir, [forged_timing])
    assert not passed
    assert "binding mismatch" in detail

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_payload["event_identity"]["shard_ordinal"] = 4
    trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")
    manifest_payload = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    trace_relative = trace_path.relative_to(raw_dir).as_posix()
    trace_reference = next(
        item for item in manifest_payload["artifacts"] if item["relative_path"] == trace_relative
    )
    trace_reference["checksum"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    trace_reference["byte_size"] = trace_path.stat().st_size
    bundle.manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    bundle.manifest_sidecar_path.write_text(
        hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    passed, detail = _audit_native_execution(raw_dir, [row])
    assert not passed
    assert "trace event identity" in detail


def test_native_timing_rejects_overlapping_axes_and_early_finalization() -> None:
    axes = ("fixed_work_control", "fixed_work", "wall_clock_30")
    timings = [
        {
            "axis": axis,
            "axis_started_ns": 1_000 + index * 200,
            "axis_completed_ns": 1_100 + index * 200,
            "finalize_started_ns": 2_000,
        }
        for index, axis in enumerate(axes)
    ]
    stage052_review._validate_native_shard_timing_order(("c101_21", 2014), timings)

    overlapping = copy.deepcopy(timings)
    overlapping[1]["axis_started_ns"] = 1_050
    with pytest.raises(ArtifactIntegrityError, match="overlap"):
        stage052_review._validate_native_shard_timing_order(("c101_21", 2014), overlapping)

    early_finalize = copy.deepcopy(timings)
    for timing in early_finalize:
        timing["finalize_started_ns"] = 1_450
    with pytest.raises(ArtifactIntegrityError, match="precedes"):
        stage052_review._validate_native_shard_timing_order(("c101_21", 2014), early_finalize)


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf, "nan"))
def test_reviewer_rejects_non_finite_native_and_performance_numbers(invalid: object) -> None:
    with pytest.raises(ValueError, match="finite"):
        _review_strict_float(invalid)


def test_native_shard_manifest_scope_binds_complete_worker_artifacts_to_parent(
    tmp_path: Path,
) -> None:
    run_label = "stage05.2_native_kernels_attempt01"
    schema = (
        ("route_dictionary", "canonical_routes"),
        ("events", "critical"),
        ("events", "screening_checks"),
        ("events", "screening_definitions_v3"),
        ("events", "screening_occurrences_v3"),
        ("diagnostic", "aggregated"),
        ("raw", ""),
        ("solution", ""),
        ("environment", ""),
        ("trace", ""),
    )
    manifests: list[dict[str, object]] = []
    parent_artifacts: list[dict[str, object]] = []
    for ordinal, (instance, seed) in enumerate(
        (instance, seed) for instance in PERFORMANCE_INSTANCES for seed in PERFORMANCE_SEEDS
    ):
        artifacts = [
            {
                "artifact_type": artifact_type,
                "artifact_subtype": subtype,
                "relative_path": (
                    f"{instance}/{seed}/{run_label}_{artifact_type}_{index}_{instance}_{seed}.json"
                ),
                "checksum": f"{ordinal * len(schema) + index:064x}",
                "evidence_completeness": "complete",
                "storage_policy_version": "artifact-storage-v2",
            }
            for index, (artifact_type, subtype) in enumerate(schema)
        ]
        parent_artifacts.extend(copy.deepcopy(artifacts))
        manifest = {
            "schema_version": "artifact-storage-v2",
            "run_label": run_label,
            "instance": instance,
            "seed": seed,
            "shard_ordinal": ordinal,
            "event_identity": "shard_ordinal+shard_local_event_id",
            "evidence_completeness": "complete",
            "storage_policy_version": "artifact-storage-v2",
            "artifacts": artifacts,
        }
        manifests.append(manifest)
        directory = tmp_path / instance / str(seed)
        directory.mkdir(parents=True)
        manifest_path = directory / f"{run_label}_shard_manifest_{instance}_{seed}.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        sidecar_path = manifest_path.with_suffix(".sha256")
        sidecar_path.write_text(manifest_sha256 + "\n", encoding="utf-8")
        for artifact_type, path in (
            ("shard_manifest", manifest_path),
            ("shard_manifest_sidecar", sidecar_path),
        ):
            parent_artifacts.append(
                {
                    "artifact_type": artifact_type,
                    "artifact_subtype": "",
                    "relative_path": path.relative_to(tmp_path).as_posix(),
                    "checksum": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "byte_size": path.stat().st_size,
                    "evidence_completeness": "complete",
                    "storage_policy_version": "artifact-storage-v2",
                }
            )

    passed, detail = _validate_native_shard_manifest_scope(
        manifests,
        raw_dir=tmp_path,
        run_label=run_label,
        parent_artifacts=parent_artifacts,
    )
    assert passed, detail

    swapped_ordinals = copy.deepcopy(manifests)
    swapped_ordinals[0]["shard_ordinal"] = 1
    swapped_ordinals[1]["shard_ordinal"] = 0
    passed, detail = _validate_native_shard_manifest_scope(
        swapped_ordinals,
        raw_dir=tmp_path,
        run_label=run_label,
        parent_artifacts=parent_artifacts,
    )
    assert not passed
    assert "identity" in detail

    missing_schema = copy.deepcopy(manifests)
    missing_schema[0].pop("schema_version")
    passed, detail = _validate_native_shard_manifest_scope(
        missing_schema,
        raw_dir=tmp_path,
        run_label=run_label,
        parent_artifacts=parent_artifacts,
    )
    assert not passed
    assert "identity" in detail

    wrong_event_identity = copy.deepcopy(manifests)
    wrong_event_identity[0]["event_identity"] = "event_id"
    passed, detail = _validate_native_shard_manifest_scope(
        wrong_event_identity,
        raw_dir=tmp_path,
        run_label=run_label,
        parent_artifacts=parent_artifacts,
    )
    assert not passed
    assert "identity" in detail

    empty = copy.deepcopy(manifests)
    empty[0]["artifacts"] = []
    passed, detail = _validate_native_shard_manifest_scope(
        empty,
        raw_dir=tmp_path,
        run_label=run_label,
        parent_artifacts=parent_artifacts,
    )
    assert not passed
    assert "schema" in detail

    passed, detail = _validate_native_shard_manifest_scope(
        manifests[:-1],
        raw_dir=tmp_path,
        run_label=run_label,
        parent_artifacts=parent_artifacts,
    )
    assert not passed
    assert "count" in detail

    duplicate = copy.deepcopy(manifests)
    duplicate_artifacts = duplicate[0]["artifacts"]
    assert isinstance(duplicate_artifacts, list)
    duplicate_artifacts.append(copy.deepcopy(duplicate_artifacts[0]))
    passed, detail = _validate_native_shard_manifest_scope(
        duplicate,
        raw_dir=tmp_path,
        run_label=run_label,
        parent_artifacts=parent_artifacts,
    )
    assert not passed
    assert "schema" in detail

    cross_shard = copy.deepcopy(manifests)
    cross_artifacts = cross_shard[0]["artifacts"]
    assert isinstance(cross_artifacts, list) and isinstance(cross_artifacts[0], dict)
    cross_artifacts[0]["relative_path"] = str(parent_artifacts[-1]["relative_path"])
    passed, detail = _validate_native_shard_manifest_scope(
        cross_shard,
        raw_dir=tmp_path,
        run_label=run_label,
        parent_artifacts=parent_artifacts,
    )
    assert not passed
    assert "path" in detail

    checksum_mismatch = copy.deepcopy(manifests)
    mismatch_artifacts = checksum_mismatch[0]["artifacts"]
    assert isinstance(mismatch_artifacts, list) and isinstance(mismatch_artifacts[0], dict)
    mismatch_artifacts[0]["checksum"] = "f" * 64
    passed, detail = _validate_native_shard_manifest_scope(
        checksum_mismatch,
        raw_dir=tmp_path,
        run_label=run_label,
        parent_artifacts=parent_artifacts,
    )
    assert not passed
    assert "parent manifest" in detail

    first_instance, first_seed = PERFORMANCE_INSTANCES[0], PERFORMANCE_SEEDS[0]
    first_manifest = (
        tmp_path
        / first_instance
        / str(first_seed)
        / f"{run_label}_shard_manifest_{first_instance}_{first_seed}.json"
    )
    first_sidecar = first_manifest.with_suffix(".sha256")
    canonical_sidecar = first_sidecar.read_text(encoding="utf-8")
    first_sidecar.unlink()
    passed, detail = _validate_native_shard_manifest_scope(
        manifests,
        raw_dir=tmp_path,
        run_label=run_label,
        parent_artifacts=parent_artifacts,
    )
    assert not passed
    assert "cannot be read" in detail
    first_sidecar.write_text(canonical_sidecar, encoding="utf-8")

    def with_rebound_sidecar(content: str) -> list[dict[str, object]]:
        first_sidecar.write_text(content, encoding="utf-8")
        rebound = copy.deepcopy(parent_artifacts)
        relative = first_sidecar.relative_to(tmp_path).as_posix()
        reference = next(item for item in rebound if item["relative_path"] == relative)
        reference["checksum"] = hashlib.sha256(first_sidecar.read_bytes()).hexdigest()
        reference["byte_size"] = first_sidecar.stat().st_size
        return rebound

    mismatched_parent = with_rebound_sidecar("0" * 64 + "\n")
    passed, detail = _validate_native_shard_manifest_scope(
        manifests,
        raw_dir=tmp_path,
        run_label=run_label,
        parent_artifacts=mismatched_parent,
    )
    assert not passed
    assert "sidecar binding" in detail

    second_instance, second_seed = PERFORMANCE_INSTANCES[0], PERFORMANCE_SEEDS[1]
    second_manifest = (
        tmp_path
        / second_instance
        / str(second_seed)
        / f"{run_label}_shard_manifest_{second_instance}_{second_seed}.json"
    )
    wrong_pair_parent = with_rebound_sidecar(
        hashlib.sha256(second_manifest.read_bytes()).hexdigest() + "\n"
    )
    passed, detail = _validate_native_shard_manifest_scope(
        manifests,
        raw_dir=tmp_path,
        run_label=run_label,
        parent_artifacts=wrong_pair_parent,
    )
    assert not passed
    assert "sidecar binding" in detail


def test_formal_budget_matrix_is_the_declared_2040_runs() -> None:
    matrix = formal_budget_matrix()
    assert matrix.total_runs == 2040
    assert matrix.declared_solver_seconds == 229_200
    assert matrix.budgets_for_customer_count(15) == (30,)
    assert matrix.budgets_for_customer_count(100) == (30, 60, 300)


def test_promotion_requires_semantic_equality_and_15_percent_gain() -> None:
    previous = [
        _observation("c101_21", 2014, 100.0),
        _observation("r101_21", 2014, 100.0),
        _observation("rc101_21", 2014, 100.0),
    ]
    promoted = [
        _observation("c101_21", 2014, 84.0),
        _observation("r101_21", 2014, 84.0),
        _observation("rc101_21", 2014, 84.0),
    ]
    decision = evaluate_promotion(previous, promoted)
    assert decision.passed
    assert decision.aggregate_median_saving >= 0.15

    promoted[1] = _observation("r101_21", 2014, 84.0, semantic_digest="changed")
    decision = evaluate_promotion(previous, promoted)
    assert not decision.passed
    assert "semantic" in decision.detail


def test_promotion_uses_100_customer_pairs_and_keeps_c5_as_control() -> None:
    previous = [
        _observation("c101C5", 2014, 100.0, customer_count=5),
        _observation("c101_21", 2014, 100.0),
        _observation("r101_21", 2014, 100.0),
        _observation("rc101_21", 2014, 100.0),
    ]
    promoted = [
        _observation("c101C5", 2014, 100.0, customer_count=5),
        _observation("c101_21", 2014, 84.0),
        _observation("r101_21", 2014, 84.0),
        _observation("rc101_21", 2014, 84.0),
    ]

    decision = evaluate_promotion(previous, promoted)

    assert decision.passed
    assert decision.aggregate_median_saving == pytest.approx(0.16)

    incomplete = evaluate_promotion(previous[1:-1], promoted[1:-1])
    assert not incomplete.passed
    assert "families" in incomplete.detail


def test_worker_selection_is_fail_fast_and_memory_bounded() -> None:
    assert select_worker_count({1: 100.0, 2: 60.0, 4: 38.0}, {1: 4, 2: 8, 4: 11}) == 4
    assert select_worker_count({1: 100.0, 2: 60.0, 4: 50.0}, {1: 4, 2: 8, 4: 11}) == 2
    with pytest.raises(ValueError, match="NOT_READY"):
        select_worker_count({1: 100.0, 2: 80.0, 4: 30.0}, {1: 4, 2: 8, 4: 11})
    with pytest.raises(ValueError, match="finite"):
        select_worker_count({1: 100.0, 2: math.nan, 4: 30.0}, {1: 4, 2: 8, 4: 11})
    with pytest.raises(ValueError, match="finite"):
        select_worker_count({1: 100.0, 2: 60.0, 4: 30.0}, {1: 4, 2: math.inf, 4: 11})


def _storage_observation(
    *,
    policy: str,
    digest: str = "same",
    persistence_seconds: float = 2.0,
    end_to_end_seconds: float = 10.0,
    peak_rss_bytes: int = 50,
) -> ArtifactStorageObservation:
    return ArtifactStorageObservation(
        instance="c101_21",
        seed=2014,
        axis="fixed_work",
        storage_policy_version=policy,
        semantic_digest=digest,
        artifact_persistence_seconds=persistence_seconds,
        end_to_end_seconds=end_to_end_seconds,
        peak_rss_bytes=peak_rss_bytes,
    )


def test_artifact_storage_promotion_requires_replay_equality() -> None:
    baseline = [_storage_observation(policy="artifact-storage-v1", peak_rss_bytes=100)]
    predecessor = [_storage_observation(policy="artifact-storage-v1")]
    candidate = [_storage_observation(policy="artifact-storage-v2", digest="changed")]

    decision = evaluate_artifact_storage_promotion(baseline, predecessor, candidate)

    assert not decision.replay_equality_passed
    assert "semantic" in decision.replay_detail


def test_artifact_storage_promotion_enforces_persistence_and_half_rss() -> None:
    baseline = [_storage_observation(policy="artifact-storage-v1", peak_rss_bytes=100)]
    predecessor = [_storage_observation(policy="artifact-storage-v1")]
    passing = [_storage_observation(policy="artifact-storage-v2", peak_rss_bytes=50)]
    decision = evaluate_artifact_storage_promotion(baseline, predecessor, passing)
    assert decision.passed

    slow = [
        _storage_observation(
            policy="artifact-storage-v2",
            persistence_seconds=3.61,
            peak_rss_bytes=50,
        )
    ]
    assert not evaluate_artifact_storage_promotion(baseline, predecessor, slow).persistence_passed

    memory_heavy = [_storage_observation(policy="artifact-storage-v2", peak_rss_bytes=51)]
    assert not evaluate_artifact_storage_promotion(baseline, predecessor, memory_heavy).rss_passed


def test_artifact_storage_persistence_gate_uses_run_aggregate() -> None:
    baseline = [
        _storage_observation(policy="artifact-storage-v1", peak_rss_bytes=100),
        replace(
            _storage_observation(policy="artifact-storage-v1", peak_rss_bytes=100),
            instance="r101_21",
        ),
    ]
    predecessor = [
        _storage_observation(policy="artifact-storage-v1"),
        replace(
            _storage_observation(policy="artifact-storage-v1"),
            instance="r101_21",
        ),
    ]
    candidate = [
        _storage_observation(
            policy="artifact-storage-v2",
            persistence_seconds=0.8,
            end_to_end_seconds=1.0,
            peak_rss_bytes=50,
        ),
        replace(
            _storage_observation(
                policy="artifact-storage-v2",
                persistence_seconds=1.0,
                end_to_end_seconds=9.0,
                peak_rss_bytes=50,
            ),
            instance="r101_21",
        ),
    ]

    decision = evaluate_artifact_storage_promotion(baseline, predecessor, candidate)

    assert decision.persistence_passed
    assert "aggregate" in decision.persistence_detail


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf))
def test_artifact_storage_observation_rejects_non_finite_numbers(
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        _storage_observation(
            policy="artifact-storage-v2",
            persistence_seconds=invalid,
        )


def test_accelerator_is_skipped_below_occupancy_threshold() -> None:
    decision = decide_accelerator(median_batch_occupancy=31)
    assert decision is AcceleratorDecision.GPU_NOT_JUSTIFIED


def test_launch_occupancy_uses_the_true_median_not_the_arithmetic_mean() -> None:
    launches, median = _launch_occupancy_summary(
        {
            "batch_launches": 5,
            "exact_calls": 98,
            "launch_occupancies": [32, 32, 32, 1, 1],
        }
    )

    assert launches == 5
    assert median == 32
    assert median != pytest.approx(98 / 5)


def test_accelerator_decision_recomputes_exactly_nine_e_occupancies(
    tmp_path: Path,
) -> None:
    run_label = "stage05.2_native_kernels_attempt01"
    raw_dir = tmp_path / run_label
    writer = ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext("stage05.2", "native_kernels", run_label),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    writer.write_control(metadata={"run_label": run_label})
    values = iter(range(1, 10))
    expected_values: list[float] = []
    for instance in PERFORMANCE_INSTANCES:
        for seed in PERFORMANCE_SEEDS:
            occupancy = 1 if instance == "c101C5" else next(values)
            screening_occupancies = [occupancy, occupancy, 100] if occupancy == 9 else [occupancy]
            if instance != "c101C5":
                expected_values.append(float(occupancy))
            shard = raw_dir / instance / str(seed)
            shard.mkdir(parents=True)
            raw_path = shard / f"{run_label}_raw_{instance}_{seed}.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "instance": instance,
                        "seed": seed,
                        "component": "native_kernels",
                        "scope": "performance",
                        "axes": {
                            "fixed_work": {
                                "validator_passed": True,
                                "valid": True,
                                "candidate_transaction_statistics": {
                                    "native_candidate_transactions": len(screening_occupancies),
                                    "native_candidate_input_count": sum(screening_occupancies),
                                    "native_screening_occupancies": (screening_occupancies),
                                    "native_screening_median_occupancy": (
                                        statistics.median(screening_occupancies)
                                    ),
                                    "native_candidate_transaction_fallbacks": 0,
                                },
                                "candidate_transaction_events": [
                                    {
                                        "event_type": ("native_candidate_transaction"),
                                        "status": "committed",
                                        "input_candidates": value,
                                    }
                                    for value in screening_occupancies
                                ],
                                "backend_metrics": {
                                    "exact_calls": 1,
                                    "batch_launches": 1,
                                    "launch_occupancies": [1],
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            writer.record_existing_file(raw_path, artifact_type="raw")
    writer.finalize()

    payload = _accelerator_decision_inputs(
        raw_dir,
        prerequisite=SimpleNamespace(to_dict=lambda: {"run_label": run_label}),
        accelerator_backend="cuda",
    )
    assert payload["decision"] == "GPU_NOT_JUSTIFIED"
    assert payload["median_screening_occupancy"] == pytest.approx(5.0)
    assert payload["input_count"] == 9
    assert payload["gpu_rows_present"] is False
    independently_recomputed, independent_median = _recompute_native_occupancies(raw_dir)
    assert independently_recomputed == payload["inputs"]
    assert independent_median == pytest.approx(5.0)
    assert [item["median_screening_occupancy"] for item in payload["inputs"]] == expected_values


def test_native_ablation_semantics_replay_exact_cache_and_transaction_events() -> None:
    row = {
        "exact_started_calls": 1,
        "exact_completed_calls": 1,
        "cache_statistics": {
            "cache_lookups": 1,
            "cache_hits": 0,
            "cache_misses": 1,
            "cache_stores": 1,
            "cache_evictions": 0,
            "cache_oversize_not_cached": 0,
        },
        "candidate_transaction_statistics": {
            "native_candidate_transactions": 1,
            "native_screening_occupancies": [2],
            "native_candidate_transaction_fallbacks": 0,
        },
    }
    screening_codes = np.zeros((2, 16), dtype="<i8")
    screening_codes[:, 0] = 1
    transaction_event: dict[str, object] = {
        "event_type": "native_candidate_transaction",
        "status": "committed",
        "input_candidates": 2,
        "candidates": [["C1"], ["C2"]],
        "exact_budget": 1,
        "budget_skips": 1,
        "cache_hits": 0,
        "exact_misses": 1,
        "iteration": 1,
        "lane": "legacy",
        "operator": "route_merge",
        "screening_passes": 2,
        "screening_rejections": 0,
        "screening_cache_hits": 0,
        "screening_exact_call_blocked": 0,
        "screening_reason_counts": {},
        "screening_integrity_evidence": {
            "candidate_ids_le_hex": np.array([0, 1], dtype="<i8").tobytes().hex(),
            "statuses_le_hex": np.zeros(2, dtype="<i8").tobytes().hex(),
            "duplicate_of_le_hex": np.array([-1, -1], dtype="<i8").tobytes().hex(),
            "codes_le_hex": screening_codes.tobytes().hex(),
            "metrics_le_hex": np.zeros((2, 15), dtype="<f8").tobytes().hex(),
            "route_offsets_le_hex": np.array([0, 1, 2], dtype="<i8").tobytes().hex(),
            "route_indices_le_hex": np.array([1, 2], dtype="<i8").tobytes().hex(),
            "counters_le_hex": np.array([2, 2, 0, 0, 2], dtype="<i8").tobytes().hex(),
        },
    }
    screening_hash, transaction_hash = _recompute_transaction_hashes(transaction_event)
    transaction_event["screening_pool_hash"] = screening_hash
    transaction_event["transaction_sha256"] = transaction_hash
    records = {
        "trace_events": [
            {"event_type": "cache_event", "operation": "lookup"},
            {"event_type": "cache_event", "operation": "miss"},
            {"event_type": "cache_event", "operation": "store"},
        ],
        "route_evaluations": [
            {
                "event_type": "route_evaluation",
                "exact_started": True,
                "exact_completed": True,
            }
        ],
        "candidate_transaction_events": [transaction_event],
        "neighborhood_events": [],
    }

    assert (
        _audit_native_ablation_records(
            row,
            records,
            require_transaction=True,
        )
        == []
    )

    tampered = copy.deepcopy(records)
    tampered["trace_events"].extend(
        [
            {"event_type": "exact_budget_boundary"},
            {
                "event_type": "candidate_state",
                "accepted": True,
                "global_best": False,
            },
        ]
    )
    failures = _audit_native_ablation_records(
        row,
        tampered,
        require_transaction=True,
    )
    assert "accepted/global-best event follows a terminal boundary" in failures

    tampered_hash = copy.deepcopy(records)
    tampered_hash["candidate_transaction_events"][0]["transaction_sha256"] = "0" * 64
    failures = _audit_native_ablation_records(
        row,
        tampered_hash,
        require_transaction=True,
    )
    assert "candidate transaction hash recomputation failed" in failures


def test_native_screening_batch_counter_replay_includes_in_batch_cache_hits() -> None:
    screening_codes = np.zeros((2, 16), dtype="<i8")
    screening_codes[:, 0] = 1

    def transaction_event(*, statuses: tuple[int, int]) -> dict[str, object]:
        cache_hits = sum(status == 2 for status in statuses)
        event_codes = screening_codes.copy()
        for index, status in enumerate(statuses):
            if status == 2:
                event_codes[index, 0] = 0
                event_codes[index, 1] = 2
        event: dict[str, object] = {
            "event_type": "native_candidate_transaction",
            "status": "committed",
            "benchmark_axis": "fixed_work",
            "input_candidates": 2,
            "candidates": [["C1"], ["C2"]],
            "exact_budget": 2,
            "budget_skips": 0,
            "cache_hits": 0,
            "exact_misses": 2 - cache_hits,
            "iteration": 1,
            "lane": "legacy",
            "operator": "route_merge",
            "screening_passes": 2 - cache_hits,
            "screening_rejections": 0,
            "screening_cache_hits": cache_hits,
            "screening_exact_call_blocked": cache_hits,
            "screening_reason_counts": (
                {"capacity_prefilter": cache_hits} if cache_hits else {}
            ),
            "screening_integrity_evidence": {
                "candidate_ids_le_hex": np.array([0, 1], dtype="<i8").tobytes().hex(),
                "statuses_le_hex": np.array(statuses, dtype="<i8").tobytes().hex(),
                "duplicate_of_le_hex": np.array([-1, -1], dtype="<i8").tobytes().hex(),
                "codes_le_hex": event_codes.tobytes().hex(),
                "metrics_le_hex": np.zeros((2, 15), dtype="<f8").tobytes().hex(),
                "route_offsets_le_hex": np.array([0, 1, 2], dtype="<i8").tobytes().hex(),
                "route_indices_le_hex": np.array([1, 2], dtype="<i8").tobytes().hex(),
                "counters_le_hex": np.array(
                    [2, 2, 0, cache_hits, 2 - cache_hits],
                    dtype="<i8",
                )
                .tobytes()
                .hex(),
            },
        }
        screening_hash, transaction_hash = _recompute_transaction_hashes(event)
        event["screening_pool_hash"] = screening_hash
        event["transaction_sha256"] = transaction_hash
        return event

    replayed = _recompute_native_screening_batch_counters(
        [
            transaction_event(statuses=(0, 0)),
            transaction_event(statuses=(0, 2)),
        ]
    )

    assert replayed == {
        "fixed_work": {
            "batch_candidates": 4,
            "batch_cache_hits": 1,
            "batch_invocations": 2,
            "occupancies": [2, 2],
        }
    }
    assert (
        _expected_native_screening_invocations(
            {
                "screening_calls": 3448,
                "screening_cache_hits": 1101,
                "native_screening_batch_candidates": 4,
                "native_screening_batch_invocations": 2,
                "native_screening_batch_occupancies": [2, 2],
            },
            replayed["fixed_work"],
        )
        == 2346
    )

    tampered = transaction_event(statuses=(0, 2))
    tampered["screening_cache_hits"] = 0
    with pytest.raises(ArtifactIntegrityError, match="aggregate does not replay"):
        _recompute_native_screening_batch_counters([tampered])
    inconsistent_counters = transaction_event(statuses=(0, 2))
    inconsistent_counters["screening_integrity_evidence"]["counters_le_hex"] = np.array(
        [2, 1, 1, 0, 1],
        dtype="<i8",
    ).tobytes().hex()
    with pytest.raises(ArtifactIntegrityError, match="status counts"):
        _recompute_native_screening_batch_counters([inconsistent_counters])
    duplicate_event = transaction_event(statuses=(0, 0))
    duplicate_event["candidates"] = [["C1"], ["C1"]]
    duplicate_event["exact_misses"] = 1
    duplicate_evidence = duplicate_event["screening_integrity_evidence"]
    duplicate_evidence["statuses_le_hex"] = np.array([0, 1], dtype="<i8").tobytes().hex()
    duplicate_evidence["duplicate_of_le_hex"] = np.array([-1, 0], dtype="<i8").tobytes().hex()
    duplicate_evidence["route_indices_le_hex"] = np.array([1, 1], dtype="<i8").tobytes().hex()
    duplicate_evidence["counters_le_hex"] = np.array(
        [2, 1, 1, 0, 1],
        dtype="<i8",
    ).tobytes().hex()
    screening_hash, transaction_hash = _recompute_transaction_hashes(duplicate_event)
    duplicate_event["screening_pool_hash"] = screening_hash
    duplicate_event["transaction_sha256"] = transaction_hash
    assert _recompute_native_screening_batch_counters([duplicate_event])["fixed_work"][
        "batch_candidates"
    ] == 2
    duplicate_evidence["duplicate_of_le_hex"] = np.array(
        [-1, -1],
        dtype="<i8",
    ).tobytes().hex()
    with pytest.raises(ArtifactIntegrityError, match="duplicate identity"):
        _recompute_native_screening_batch_counters([duplicate_event])
    missing_duplicate_marker = transaction_event(statuses=(0, 0))
    missing_duplicate_marker["candidates"] = [["C1"], ["C1"]]
    missing_duplicate_marker["screening_integrity_evidence"][
        "route_indices_le_hex"
    ] = np.array([1, 1], dtype="<i8").tobytes().hex()
    with pytest.raises(ArtifactIntegrityError, match="repeated candidate"):
        _recompute_native_screening_batch_counters([missing_duplicate_marker])
    with pytest.raises(ArtifactIntegrityError, match="scalar counters are invalid"):
        _expected_native_screening_invocations(
            {
                "screening_calls": 1,
                "screening_cache_hits": 2,
            },
            None,
        )


def test_native_ablation_replays_batched_screening_bytes_and_lane_deadlines() -> None:
    screening_codes = np.zeros((2, 16), dtype="<i8")
    screening_codes[:, 0] = 1
    transaction_event: dict[str, object] = {
        "input_candidates": 2,
        "candidates": [["C1"], ["C2"]],
        "exact_budget": 1,
        "budget_skips": 1,
        "cache_hits": 0,
        "exact_misses": 1,
        "iteration": 2,
        "lane": "constraint",
        "operator": "route_merge",
        "screening_passes": 2,
        "screening_rejections": 0,
        "screening_cache_hits": 0,
        "screening_exact_call_blocked": 0,
        "screening_reason_counts": {},
        "screening_integrity_evidence": {
            "candidate_ids_le_hex": np.array([0, 1], dtype="<i8").tobytes().hex(),
            "statuses_le_hex": np.zeros(2, dtype="<i8").tobytes().hex(),
            "duplicate_of_le_hex": np.array([-1, -1], dtype="<i8").tobytes().hex(),
            "codes_le_hex": screening_codes.tobytes().hex(),
            "metrics_le_hex": np.zeros((2, 15), dtype="<f8").tobytes().hex(),
            "route_offsets_le_hex": np.array([0, 1, 2], dtype="<i8").tobytes().hex(),
            "route_indices_le_hex": np.array([1, 2], dtype="<i8").tobytes().hex(),
            "counters_le_hex": np.array([2, 2, 0, 0, 2], dtype="<i8").tobytes().hex(),
        },
    }
    screening_hash, _transaction_hash = _recompute_transaction_hashes(transaction_event)
    screening_event = {
        key: value
        for key, value in transaction_event.items()
        if key
            in {
                "candidates",
                "input_candidates",
                "screening_cache_hits",
                "screening_exact_call_blocked",
                "screening_integrity_evidence",
                "screening_passes",
                "screening_reason_counts",
                "screening_rejections",
            }
    }
    screening_event.update(
        {
            "event_type": "native_candidate_screening_batch",
            "status": "committed",
            "lane": "constraint",
            "iteration": 2,
            "operator": "route_merge",
            "screening_pool_hash": screening_hash,
        }
    )
    row = {
        "exact_started_calls": 0,
        "exact_completed_calls": 0,
        "cache_statistics": {},
        "candidate_transaction_statistics": {},
    }
    records = {
        "trace_events": [
            {"event_type": "deadline_boundary", "lane": "legacy"},
            {
                "event_type": "candidate_state",
                "lane": "constraint",
                "accepted": True,
                "global_best": False,
            },
            screening_event,
        ],
        "route_evaluations": [],
        "candidate_transaction_events": [],
        "neighborhood_events": [],
    }

    assert (
        _audit_native_ablation_records(
            row,
            records,
            require_transaction=False,
            require_batched_screening=True,
        )
        == []
    )
    assert _recompute_screening_hash(screening_event) == screening_hash

    tampered = copy.deepcopy(records)
    tampered["trace_events"][-1]["screening_integrity_evidence"][
        "statuses_le_hex"
    ] = np.array([1, 0], dtype="<i8").tobytes().hex()
    failures = _audit_native_ablation_records(
        row,
        tampered,
        require_transaction=False,
        require_batched_screening=True,
    )
    assert failures == [
        "batched screening evidence is invalid: "
        "transaction screening counters do not match candidate status counts"
    ]

    same_lane = copy.deepcopy(records)
    same_lane["trace_events"][1]["lane"] = "legacy"
    failures = _audit_native_ablation_records(
        row,
        same_lane,
        require_transaction=False,
        require_batched_screening=True,
    )
    assert "accepted/global-best event follows a terminal boundary" in failures


def test_accelerator_pilot_metadata_requires_six_workers_and_transaction_config() -> None:
    metadata = {
        "component": "accelerator_pilot",
        "backend": "cpu_batch",
        "execution_backend": "native_cpu",
        "accelerator_decision_mode": "accelerator_pilot",
        "native_kernel_config": NativeKernelConfig().to_dict(),
        "candidate_transaction_config": NativeCandidateTransactionConfig().to_dict(),
        "worker_count": 6,
        "staging_root": {"alias": "wsl_staging"},
    }

    assert _accelerator_pilot_metadata_matches(metadata, "native_cpu")
    assert not _accelerator_pilot_metadata_matches({**metadata, "worker_count": 4}, "native_cpu")
    assert not _accelerator_pilot_metadata_matches(
        {key: value for key, value in metadata.items() if key != "candidate_transaction_config"},
        "native_cpu",
    )


def test_pair_pruning_aggregate_is_independently_recomputed() -> None:
    pair_identity = {
        "left": {"index": 0, "sequence": ["C1"]},
        "reason": "capacity_prefilter",
        "right": {"index": 1, "sequence": ["C2"]},
        "schema_version": "route-merge-pair-pruning-v1",
        "skipped_candidate_count": 4,
    }
    digest = hashlib.sha256(
        json.dumps(
            pair_identity,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    row = {
        "exact_started_calls": 0,
        "exact_completed_calls": 0,
        "cache_statistics": {},
        "candidate_transaction_statistics": {},
    }
    records = {
        "trace_events": [],
        "route_evaluations": [],
        "candidate_transaction_events": [],
        "neighborhood_events": [
            {
                "operator": "route_merge",
                "status": "pair_prefilter_rejected_aggregate",
                "reason": "capacity_prefilter",
                "route_indices": [0, 1],
                "candidate_route_sequences": [["C1"], ["C2"]],
                "aggregate_count": 4,
                "candidate_pool_hash": digest,
            }
        ],
    }

    assert (
        _audit_native_ablation_records(
            row,
            records,
            require_transaction=False,
        )
        == []
    )
    records["neighborhood_events"][0]["aggregate_count"] = 5
    failures = _audit_native_ablation_records(
        row,
        records,
        require_transaction=False,
    )
    assert "pair-pruning aggregate recomputation failed" in failures


def test_native_ablation_record_reads_bounded_stream_audit_without_materializing_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route_evaluation = RouteEvaluationTrace(
        evaluation_id=1,
        route_key="route:2:C1",
        lane="legacy",
        iteration=1,
        operator="repair",
        kind="exact_call",
        started_at=0.1,
        completed_at=0.2,
        duration_seconds=0.1,
        exact_started=True,
        exact_completed=True,
        feasible=True,
        failure_reason="",
        labels_generated=0,
        labels_expanded=0,
        labels_pruned=0,
        deadline_boundary="",
        cache_key_digest="",
        route_change_status="changed",
        status="completed",
    )
    pair_event = {
        "operator": "route_merge",
        "status": "pair_prefilter_rejected_aggregate",
        "reason": "capacity_prefilter",
        "route_indices": [0, 1],
        "candidate_route_sequences": [["C1"], ["C2"]],
        "aggregate_count": 4,
        "candidate_pool_hash": "a" * 64,
    }

    class AuditSink:
        def native_ablation_records(self) -> dict[str, tuple[object, ...]]:
            return {
                "trace_events": (
                    {
                        "event_type": "candidate_state",
                        "lane": "legacy",
                        "iteration": 1,
                        "accepted": False,
                    },
                ),
                "route_evaluations": (route_evaluation,),
                "neighborhood_events": (pair_event,),
            }

    trace = SimpleNamespace(
        stream_sink=AuditSink(),
        reconcile=lambda _result: {"status": "pass"},
    )
    result = SimpleNamespace(
        measurement_trace=trace,
        objective=SimpleNamespace(key=(1, 2.0, 0.0, 0)),
        routes=(("D0", "C1", "D0"),),
        candidate_transaction_events=(),
        neighborhood_events=(),
        exact_started_calls=1,
        exact_completed_calls=1,
        cache_incremental_statistics={},
        exact_deadline_statistics={},
        candidate_transaction_statistics={},
        termination_reason="fixed_work_budget",
        backend_metrics={},
        screening_statistics={},
    )
    instance = Instance(
        "streaming_native_ablation",
        (
            Node("D0", NodeType.DEPOT, 0, 0, 0, 0, 100, 0),
            Node("C1", NodeType.CUSTOMER, 1, 0, 1, 0, 100, 0),
        ),
        Vehicle(10, 10, 1, 1, 1),
        distance_backend="python",
    )
    monkeypatch.setattr(
        stage052_performance,
        "validate_routes",
        lambda *_args, **_kwargs: SimpleNamespace(feasible=True),
    )

    record = _native_ablation_record(
        instance,
        result,
        implementation_mode="candidate_transaction",
        solver_seconds=1.0,
    )

    assert record["candidate_records"]["trace_events"][0]["event_type"] == "candidate_state"
    assert record["candidate_records"]["route_evaluations"][0]["route_key"] == "route:2:C1"
    assert record["candidate_records"]["neighborhood_events"] == [pair_event]


def test_instance_lookup_and_distance_matrix_are_stable() -> None:
    depot = Node("D0", NodeType.DEPOT, 0, 0, 0, 0, 100, 0)
    customer = Node("C1", NodeType.CUSTOMER, 3, 4, 1, 0, 100, 0)
    instance = Instance(
        "toy",
        (depot, customer),
        Vehicle(100, 10, 1, 1, 1),
        distance_backend="python",
    )
    assert instance.by_name is instance.by_name
    assert instance.customers is instance.customers
    assert instance.distance("D0", "C1") == 5.0


def test_stage052_runner_contract_has_canonical_labels_and_axes(
    tmp_path: Path,
) -> None:
    validate_stage052_run_label(
        "stage05.2_perf_baseline_attempt01", Stage052Component.PERF_BASELINE
    )
    with pytest.raises(ValueError, match="canonical"):
        validate_stage052_run_label(
            "stage052_perf_baseline_attempt01", Stage052Component.PERF_BASELINE
        )
    assert tuple(axis.name for axis in axes_for_scope("performance")) == (
        "fixed_work_control",
        "fixed_work",
        "wall_clock_30",
    )
    assert tuple(axis.name for axis in axes_for_scope("formal", customer_count=100)) == (
        "wall_clock_30",
        "wall_clock_60",
        "wall_clock_300",
    )
    assert axes_for_scope("pilot", customer_count=5)[0].max_iterations == 1000
    assert axes_for_scope("pilot", customer_count=100)[0].max_iterations is None
    assert axes_for_scope("formal", customer_count=15)[0].max_iterations == 1000
    assert all(axis.max_iterations is None for axis in axes_for_scope("formal", customer_count=100))

    publication = tmp_path / "stage051.json"
    publication.write_text(
        '{"run_label":"stage05.1_best_known_attempt06",'
        '"status":"READY_FOR_STAGE05_2",'
        '"comparison_baseline":"stage04_adaptive_weights_attempt15"}',
        encoding="utf-8",
    )
    assert verify_stage051_prerequisite(publication)["status"] == "READY_FOR_STAGE05_2"


def test_stage052_runner_checkpoints_use_only_completed_global_best_events() -> None:
    result = SimpleNamespace(
        measurement_trace=SimpleNamespace(
            events=[
                {
                    "event_type": "candidate_state",
                    "timestamp_seconds": 0.5,
                    "iteration": 0,
                    "accepted": False,
                    "global_best": False,
                    "current_objective_key": [3, 120.0, 4.0, 2],
                    "candidate_objective_key": [3, 130.0, 4.0, 2],
                },
                {
                    "event_type": "candidate_state",
                    "timestamp_seconds": 1.0,
                    "iteration": 1,
                    "accepted": True,
                    "global_best": True,
                    "current_objective_key": [3, 120.0, 4.0, 2],
                    "candidate_objective_key": [2, 110.0, 3.0, 1],
                },
                {
                    "event_type": "candidate_state",
                    "timestamp_seconds": 5.0001,
                    "iteration": 2,
                    "accepted": True,
                    "global_best": True,
                    "current_objective_key": [2, 110.0, 3.0, 1],
                    "candidate_objective_key": [2, 100.0, 2.0, 1],
                },
            ]
        ),
        initial_routes=(("D0", "C1", "D0"),),
        initial_objective=SimpleNamespace(key=(3, 120.0, 4.0, 2)),
        objective=SimpleNamespace(key=(2, 100.0, 2.0, 1)),
        termination_reason="wall_clock_deadline",
        runtime_seconds=30.0,
    )

    checkpoints = _stage052_anytime_checkpoints(
        result,
        instance_name="c101_21",
        seed=2014,
        axis=Stage052Axis("wall_clock_30", "wall_clock", 30.0, max_iterations=None),
    )

    assert tuple(item.checkpoint_seconds for item in checkpoints) == (1, 5, 10, 30)
    assert checkpoints[0].objective_key == (2, 110.0, 3.0, 1)
    assert checkpoints[1].objective_key == (2, 110.0, 3.0, 1)
    assert checkpoints[2].objective_key == (2, 100.0, 2.0, 1)


def test_stage052_iteration_limit_carry_uses_explicit_completion_timestamp() -> None:
    result = SimpleNamespace(
        measurement_trace=SimpleNamespace(
            events=[
                {
                    "event_type": "candidate_state",
                    "timestamp_seconds": 0.5,
                    "iteration": 0,
                    "accepted": False,
                    "global_best": False,
                    "current_objective_key": [3, 120.0, 4.0, 2],
                    "candidate_objective_key": [3, 130.0, 4.0, 2],
                },
                {
                    "event_type": "candidate_state",
                    "timestamp_seconds": 0.8,
                    "iteration": 1,
                    "accepted": True,
                    "global_best": True,
                    "current_objective_key": [3, 120.0, 4.0, 2],
                    "candidate_objective_key": [2, 100.0, 2.0, 1],
                },
            ]
        ),
        initial_routes=(("D0", "C1", "D0"),),
        initial_objective=SimpleNamespace(key=(3, 120.0, 4.0, 2)),
        objective=SimpleNamespace(key=(2, 100.0, 2.0, 1)),
        termination_reason="iteration_limit",
        iteration_limit_completed_at_seconds=5.1,
        runtime_seconds=5.2,
    )

    checkpoints = _stage052_anytime_checkpoints(
        result,
        instance_name="c101C5",
        seed=2014,
        axis=Stage052Axis("wall_clock_30", "wall_clock", 30.0, max_iterations=1000),
    )

    assert tuple(item.source for item in checkpoints[:2]) == (
        "accepted_global_best",
        "accepted_global_best",
    )
    assert checkpoints[2].source == "final_incumbent_carry_forward"


def test_stage052_reviewer_rejects_duplicate_or_missing_axes() -> None:
    rows = [
        {
            "instance": instance,
            "seed": str(seed),
            "axis": axis,
            "customer_count": "5" if instance == "c101C5" else "100",
            "validator_passed": "True",
            "failure_status": "",
        }
        for instance in PERFORMANCE_INSTANCES
        for seed in PERFORMANCE_SEEDS
        for axis in ("fixed_work_control", "fixed_work", "wall_clock_30")
    ]
    passed, detail = validate_per_run_scope(
        rows,
        instances=PERFORMANCE_INSTANCES,
        seeds=PERFORMANCE_SEEDS,
        axes=("fixed_work_control", "fixed_work", "wall_clock_30"),
    )
    assert passed, detail
    rows.append(dict(rows[0]))
    passed, detail = validate_per_run_scope(
        rows,
        instances=PERFORMANCE_INSTANCES,
        seeds=PERFORMANCE_SEEDS,
        axes=("fixed_work_control", "fixed_work", "wall_clock_30"),
    )
    assert not passed
    assert "duplicate" in detail


def test_stage052_reviewer_recomputes_customer_count_identity() -> None:
    rows = [
        {
            "instance": instance,
            "seed": str(seed),
            "axis": axis,
            "customer_count": "5" if instance == "c101C5" else "100",
            "validator_passed": "True",
            "failure_status": "",
        }
        for instance in PERFORMANCE_INSTANCES
        for seed in PERFORMANCE_SEEDS
        for axis in ("fixed_work_control", "fixed_work", "wall_clock_30")
    ]
    rows[9]["customer_count"] = "5"

    passed, detail = validate_per_run_scope(
        rows,
        instances=PERFORMANCE_INSTANCES,
        seeds=PERFORMANCE_SEEDS,
        axes=("fixed_work_control", "fixed_work", "wall_clock_30"),
    )

    assert not passed
    assert "customer_count mismatch" in detail


def test_stage052_review_prerequisite_verifies_identity_status_and_files(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "stage05.2_hot_path_attempt03"
    bundle = ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext("stage05.2", "hot_path", raw_dir.name),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v1"),
    ).finalize()
    review_dir = raw_dir / "review"
    review_dir.mkdir()
    report = review_dir / "review_report.md"
    findings = review_dir / "review_findings.csv"
    report.write_text("accepted\n", encoding="utf-8")
    findings.write_text("gate,passed\nall,True\n", encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    review_manifest = review_dir / "review_manifest.json"
    review_manifest.write_text(
        json.dumps(
            {
                "schema_version": "stage05.2-review-v1",
                "run_label": raw_dir.name,
                "component": "hot_path",
                "scope": "performance",
                "status": "READY_FOR_STAGE052_ARTIFACT_STREAMING",
                "raw_manifest_sha256": digest(bundle.manifest_path),
                "gates": {"all": {"passed": True}},
                "files": {
                    report.name: digest(report),
                    findings.name: digest(findings),
                },
            }
        ),
        encoding="utf-8",
    )

    verify_stage052_review_prerequisite(
        raw_dir,
        expected_component="hot_path",
        expected_status="READY_FOR_STAGE052_ARTIFACT_STREAMING",
    )
    report.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_stage052_review_prerequisite(
            raw_dir,
            expected_component="hot_path",
            expected_status="READY_FOR_STAGE052_ARTIFACT_STREAMING",
        )


def test_review_lineage_accepts_verified_append_only_prior_hashes(tmp_path: Path) -> None:
    run_label = "stage05.2_hot_path_attempt92"
    raw_dir = tmp_path / run_label
    ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext("stage05.2", "hot_path", run_label),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v1"),
    ).finalize()
    review_dir = raw_dir / "review"
    review_dir.mkdir()
    report = review_dir / "review_report.md"
    findings = review_dir / "review_findings.csv"
    report.write_text("review\n", encoding="utf-8")
    findings.write_text("gate,passed\nall,True\n", encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": "stage05.2-review-v1",
        "run_label": run_label,
        "component": "hot_path",
        "scope": "performance",
        "status": "READY_FOR_STAGE052_ARTIFACT_STREAMING",
        "gates": {"all": {"passed": True}},
        "files": {
            report.name: hashlib.sha256(report.read_bytes()).hexdigest(),
            findings.name: hashlib.sha256(findings.read_bytes()).hexdigest(),
        },
    }
    (review_dir / "review_manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    first_lineage, _ = _prior_review_manifest_history(raw_dir)
    payload.pop("files")
    payload["review_manifest_lineage_sha256"] = first_lineage
    stage052_review._publish_review_generation(
        review_dir=review_dir,
        findings=b"gate,passed\nall,True\n",
        report=b"second review\n",
        manifest=payload,
    )

    second_lineage, retry_history = _prior_review_manifest_history(raw_dir)

    assert retry_history == []
    assert second_lineage[:1] == first_lineage
    assert len(second_lineage) == 2


def test_prerequisite_receipt_must_bind_finalized_service_and_current_review(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "stage05.2_hot_path_attempt87"
    review_dir = raw_dir / "review"
    review_dir.mkdir(parents=True)
    review_manifest = review_dir / "review_manifest.json"
    review_manifest.write_text('{"status":"READY"}\n', encoding="utf-8")
    receipt = {
        "run_label": raw_dir.name,
        "finalized": True,
        "status": "completed",
        "systemd_service_result": "success",
        "cgroup_memory_peak_status": "verified",
        "raw_manifest_unchanged": True,
        "review_manifest_sha256": hashlib.sha256(review_manifest.read_bytes()).hexdigest(),
    }
    execution = review_dir / "review_execution.json"
    execution.write_text(json.dumps(receipt), encoding="utf-8")

    assert (
        stage052_evidence.verify_stage052_review_execution_receipt(raw_dir, review_manifest)[
            "status"
        ]
        == "completed"
    )
    receipt["cgroup_memory_peak_status"] = "unavailable"
    execution.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="execution receipt is invalid"):
        stage052_evidence.verify_stage052_review_execution_receipt(raw_dir, review_manifest)


def test_failed_review_is_archived_as_explicit_retry_history(tmp_path: Path) -> None:
    run_label = "stage05.2_hot_path_attempt91"
    raw_dir = tmp_path / run_label
    bundle = ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext("stage05.2", "hot_path", run_label),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v1"),
    ).finalize()
    review_dir = raw_dir / "review"
    review_dir.mkdir()
    report = review_dir / "review_report.md"
    findings = review_dir / "review_findings.csv"
    report.write_text("not ready\n", encoding="utf-8")
    findings.write_text("gate,passed\nruntime_identity,False\n", encoding="utf-8")
    prior_manifest = review_dir / "review_manifest.json"
    prior_manifest.write_text(
        json.dumps(
            {
                "schema_version": "stage05.2-review-v1",
                "run_label": run_label,
                "component": "hot_path",
                "scope": "performance",
                "status": "NOT_READY",
                "raw_manifest_sha256": hashlib.sha256(
                    bundle.manifest_path.read_bytes()
                ).hexdigest(),
                "gates": {"runtime_identity": {"passed": False}},
                "files": {
                    report.name: hashlib.sha256(report.read_bytes()).hexdigest(),
                    findings.name: hashlib.sha256(findings.read_bytes()).hexdigest(),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    prior_sha256 = hashlib.sha256(prior_manifest.read_bytes()).hexdigest()

    lineage, retry_history = _prior_review_manifest_history(raw_dir)

    assert lineage == []
    assert retry_history == [prior_sha256]
    assert (review_dir / "history" / prior_sha256 / "review_manifest.json").is_file()

    second = stage052_review._publish_review_generation(
        review_dir=review_dir,
        findings=b"gate,passed\nruntime_identity,False\n",
        report=b"still not ready\n",
        manifest={
            "schema_version": "stage05.2-review-v1",
            "run_label": run_label,
            "component": "hot_path",
            "scope": "performance",
            "status": "NOT_READY",
            "raw_manifest_sha256": hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest(),
            "review_manifest_lineage_sha256": [],
            "review_retry_history_sha256": retry_history,
            "gates": {"runtime_identity": {"passed": False}},
        },
    )
    second_sha256 = hashlib.sha256(second["review_manifest"].read_bytes()).hexdigest()

    lineage, retry_history = _prior_review_manifest_history(raw_dir)

    assert lineage == []
    assert retry_history == [prior_sha256, second_sha256]


@pytest.mark.parametrize(
    "failure_stage",
    ("review_findings.csv", "review_report.md", "review_manifest"),
)
def test_review_generation_publish_failure_preserves_and_archives_prior_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    run_label = "stage05.2_hot_path_attempt89"
    raw_dir = tmp_path / failure_stage / run_label
    bundle = ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext("stage05.2", "hot_path", run_label),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v1"),
    ).finalize()
    review_dir = raw_dir / "review"
    review_dir.mkdir()
    report = review_dir / "review_report.md"
    findings = review_dir / "review_findings.csv"
    report.write_text("accepted prior\n", encoding="utf-8")
    findings.write_text("gate,passed\nall,True\n", encoding="utf-8")
    prior_manifest = review_dir / "review_manifest.json"
    prior_manifest.write_text(
        json.dumps(
            {
                "schema_version": "stage05.2-review-v1",
                "run_label": run_label,
                "component": "hot_path",
                "scope": "performance",
                "status": "READY_FOR_STAGE052_ARTIFACT_STREAMING",
                "raw_manifest_sha256": hashlib.sha256(
                    bundle.manifest_path.read_bytes()
                ).hexdigest(),
                "gates": {"all": {"passed": True}},
                "files": {
                    findings.name: hashlib.sha256(findings.read_bytes()).hexdigest(),
                    report.name: hashlib.sha256(report.read_bytes()).hexdigest(),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    prior_bytes = {
        "review_manifest.json": prior_manifest.read_bytes(),
        findings.name: findings.read_bytes(),
        report.name: report.read_bytes(),
    }
    prior_sha256 = hashlib.sha256(prior_manifest.read_bytes()).hexdigest()
    lineage, retry_history = _prior_review_manifest_history(raw_dir)
    assert retry_history == []
    assert lineage == [prior_sha256]

    original_write = stage052_review._write_fsync

    def fail_at_selected_stage(path: Path, payload: bytes) -> None:
        if failure_stage in path.name:
            raise OSError(f"injected {failure_stage} publication failure")
        original_write(path, payload)

    monkeypatch.setattr(stage052_review, "_write_fsync", fail_at_selected_stage)
    manifest = {
        "schema_version": "stage05.2-review-v1",
        "run_label": run_label,
        "component": "hot_path",
        "scope": "performance",
        "status": "READY_FOR_STAGE052_ARTIFACT_STREAMING",
        "raw_manifest_sha256": hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest(),
        "review_manifest_lineage_sha256": lineage,
        "gates": {"all": {"passed": True}},
    }
    with pytest.raises(OSError, match="injected"):
        stage052_review._publish_review_generation(
            review_dir=review_dir,
            findings=b"gate,passed,detail\nall,True,re-reviewed\n",
            report=b"accepted re-review\n",
            manifest=manifest,
        )

    assert prior_manifest.read_bytes() == prior_bytes["review_manifest.json"]
    assert findings.read_bytes() == prior_bytes[findings.name]
    assert report.read_bytes() == prior_bytes[report.name]
    verify_stage052_review_prerequisite(
        raw_dir,
        expected_component="hot_path",
        expected_status="READY_FOR_STAGE052_ARTIFACT_STREAMING",
    )
    archive = review_dir / "history" / prior_sha256
    assert {path.name for path in archive.iterdir()} == set(prior_bytes)
    assert all((archive / name).read_bytes() == payload for name, payload in prior_bytes.items())

    monkeypatch.setattr(stage052_review, "_write_fsync", original_write)
    published = stage052_review._publish_review_generation(
        review_dir=review_dir,
        findings=b"gate,passed,detail\nall,True,re-reviewed\n",
        report=b"accepted re-review\n",
        manifest=manifest,
    )
    assert published["review_manifest"] == prior_manifest
    assert "generations" in published["review_findings"].parts
    (published["review_findings"].parent / "._review_findings.csv").write_bytes(
        b"AppleDouble metadata"
    )
    stage052_review._publish_review_generation(
        review_dir=review_dir,
        findings=b"gate,passed,detail\nall,True,re-reviewed\n",
        report=b"accepted re-review\n",
        manifest=manifest,
    )
    verify_stage052_review_prerequisite(
        raw_dir,
        expected_component="hot_path",
        expected_status="READY_FOR_STAGE052_ARTIFACT_STREAMING",
    )
    assert findings.read_bytes() == prior_bytes[findings.name]
    assert report.read_bytes() == prior_bytes[report.name]


def test_review_generation_streams_mismatch_file_into_publication(tmp_path: Path) -> None:
    mismatch_source = tmp_path / "mismatches.csv"
    mismatch_source.write_bytes(
        b"instance,seed,axis,ordinal,field,left_digest,right_digest\n"
        + b"c101_21,2014,fixed_work,1,event.status,left,right\n" * 10_000
    )
    review_dir = tmp_path / "review"

    outputs = stage052_review._publish_review_generation(
        review_dir=review_dir,
        findings=b"gate,passed,detail\nall,True,passed\n",
        report=b"accepted\n",
        semantic_mismatches=mismatch_source,
        manifest={
            "schema_version": "stage05.2-review-v1",
            "run_label": "stage05.2_hot_path_attempt99",
            "component": "hot_path",
            "scope": "performance",
            "status": "READY_FOR_STAGE052_ARTIFACT_STREAMING",
            "raw_manifest_sha256": "a" * 64,
            "gates": {"all": {"passed": True}},
        },
    )

    assert outputs["semantic_mismatches"].read_bytes() == mismatch_source.read_bytes()
    manifest = json.loads(outputs["review_manifest"].read_text(encoding="utf-8"))
    relative = outputs["semantic_mismatches"].relative_to(review_dir).as_posix()
    assert manifest["files"][relative] == hashlib.sha256(mismatch_source.read_bytes()).hexdigest()


def test_prior_review_archive_streams_published_mismatch_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "stage05.2_hot_path_attempt98"
    review_dir = raw_dir / "review"
    mismatch_source = tmp_path / "mismatches.csv"
    mismatch_source.write_bytes(
        b"instance,seed,axis,ordinal,field,left_digest,right_digest\n"
        + b"c101_21,2014,fixed_work,1,event.status,left,right\n" * 20_000
    )
    outputs = stage052_review._publish_review_generation(
        review_dir=review_dir,
        findings=b"gate,passed,detail\nall,True,passed\n",
        report=b"accepted\n",
        semantic_mismatches=mismatch_source,
        manifest={
            "schema_version": "stage05.2-review-v1",
            "run_label": raw_dir.name,
            "component": "hot_path",
            "scope": "performance",
            "status": "READY_FOR_STAGE052_ARTIFACT_STREAMING",
            "raw_manifest_sha256": "a" * 64,
            "gates": {"all": {"passed": True}},
        },
    )
    mismatch_path = outputs["semantic_mismatches"]
    manifest_path = outputs["review_manifest"]
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_read_bytes = Path.read_bytes

    def reject_mismatch_read_bytes(path: Path) -> bytes:
        if path == mismatch_path:
            raise AssertionError("large mismatch archive must not use Path.read_bytes()")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_mismatch_read_bytes)
    prior_sha256 = stage052_review._archive_prior_review_generation(
        raw_dir,
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
    )

    archived_mismatch = (
        review_dir / "history" / prior_sha256 / mismatch_path.relative_to(review_dir)
    )
    assert archived_mismatch.read_text(encoding="utf-8") == mismatch_source.read_text(
        encoding="utf-8"
    )


def test_stage052_producer_prerequisite_binds_raw_and_review_identity(tmp_path: Path) -> None:
    run_label = "stage05.2_artifact_streaming_attempt04"
    raw_dir = tmp_path / run_label
    config = tmp_path / "stage052.toml"
    config.write_text("[stage05_2]\nschema_version='test'\n", encoding="utf-8")
    config_digest = hashlib.sha256(config.read_bytes()).hexdigest()
    writer = ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext("stage05.2", "artifact_streaming", run_label),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    writer.write_control(
        metadata={
            "run_label": run_label,
            "component": "artifact_streaming",
            "scope": "performance",
            "repository_dirty": False,
            "repository_revision": "a" * 40,
            "configuration_sha256": config_digest,
        },
        configuration_path=config,
    )
    bundle = writer.finalize()
    review_dir = raw_dir / "review"
    review_dir.mkdir()
    report = review_dir / "review_report.md"
    findings = review_dir / "review_findings.csv"
    report.write_text("accepted\n", encoding="utf-8")
    findings.write_text("gate,passed\nall,True\n", encoding="utf-8")
    review_manifest = review_dir / "review_manifest.json"
    review_manifest.write_text(
        json.dumps(
            {
                "schema_version": "stage05.2-review-v1",
                "run_label": run_label,
                "component": "artifact_streaming",
                "scope": "performance",
                "status": "READY_FOR_STAGE052_JOB_PARALLEL",
                "review_execution_required": True,
                "raw_manifest_sha256": hashlib.sha256(
                    bundle.manifest_path.read_bytes()
                ).hexdigest(),
                "gates": {"all": {"passed": True}},
                "files": {
                    report.name: hashlib.sha256(report.read_bytes()).hexdigest(),
                    findings.name: hashlib.sha256(findings.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    _bind_successful_review_execution(review_manifest)

    identity = verify_stage052_prerequisite(
        raw_dir,
        expected_component="artifact_streaming",
        expected_status="READY_FOR_STAGE052_JOB_PARALLEL",
        expected_run_label=run_label,
    )
    assert identity.run_label == run_label
    assert identity.repository_revision == "a" * 40

    review_path = review_dir / "review_manifest.json"
    lineage, retry_history = _prior_review_manifest_history(raw_dir)
    assert retry_history == []
    assert len(lineage) == 1
    prior_review_sha256 = lineage[0]
    stronger_review = json.loads(review_path.read_text(encoding="utf-8"))
    stronger_review.pop("files")
    stronger_review["review_manifest_lineage_sha256"] = lineage
    stage052_review._publish_review_generation(
        review_dir=review_dir,
        findings=findings.read_bytes(),
        report=report.read_bytes(),
        manifest=stronger_review,
    )
    _bind_successful_review_execution(review_path)
    current_identity = verify_stage052_prerequisite(
        raw_dir,
        expected_component="artifact_streaming",
        expected_status="READY_FOR_STAGE052_JOB_PARALLEL",
    )
    assert _prerequisite_binding_matches(identity.to_dict(), current_identity, raw_dir)

    archive = review_dir / "history" / prior_review_sha256
    (archive / "._review_report.md").write_bytes(b"AppleDouble metadata")
    assert _prerequisite_binding_matches(identity.to_dict(), current_identity, raw_dir)
    unexpected = archive / "unexpected.txt"
    unexpected.write_text("not AppleDouble\n", encoding="utf-8")
    assert not _prerequisite_binding_matches(identity.to_dict(), current_identity, raw_dir)
    unexpected.unlink()
    missing_archive = archive.with_name(f"{archive.name}.missing")
    archive.rename(missing_archive)
    assert not _prerequisite_binding_matches(identity.to_dict(), current_identity, raw_dir)
    missing_archive.rename(archive)

    archived_report = archive / "review_report.md"
    accepted_archived_report = archived_report.read_bytes()
    archived_report.write_text("tampered archive\n", encoding="utf-8")
    assert not _prerequisite_binding_matches(identity.to_dict(), current_identity, raw_dir)
    archived_report.write_bytes(accepted_archived_report)

    accepted_current_manifest = review_path.read_bytes()
    current_review = json.loads(review_path.read_text(encoding="utf-8"))
    current_review["review_manifest_lineage_sha256"] = [prior_review_sha256, "b" * 64]
    review_path.write_text(json.dumps(current_review), encoding="utf-8")
    assert not _prerequisite_binding_matches(identity.to_dict(), current_identity, raw_dir)
    review_path.write_bytes(accepted_current_manifest)

    current_review = json.loads(review_path.read_text(encoding="utf-8"))
    published_report = review_dir / next(
        name for name in current_review["files"] if name.endswith("review_report.md")
    )
    accepted_report = published_report.read_bytes()
    published_report.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        verify_stage052_prerequisite(
            raw_dir,
            expected_component="artifact_streaming",
            expected_status="READY_FOR_STAGE052_JOB_PARALLEL",
        )
    published_report.write_bytes(accepted_report)
    manifest_payload = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    manifest_payload["post_review_mutation"] = True
    bundle.manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    bundle.manifest_sidecar_path.write_text(
        hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactIntegrityError, match="stale"):
        verify_stage052_prerequisite(
            raw_dir,
            expected_component="artifact_streaming",
            expected_status="READY_FOR_STAGE052_JOB_PARALLEL",
        )


def test_prerequisite_rejects_replaced_persistence_envelope(tmp_path: Path) -> None:
    run_label = "stage05.2_artifact_streaming_attempt95"
    raw_dir = tmp_path / run_label
    config = tmp_path / "stage052.toml"
    config.write_text("[stage05_2]\nschema_version='test'\n", encoding="utf-8")
    config_digest = hashlib.sha256(config.read_bytes()).hexdigest()
    writer = ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext("stage05.2", "artifact_streaming", run_label),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    writer.write_control(
        metadata={
            "run_label": run_label,
            "component": "artifact_streaming",
            "scope": "performance",
            "repository_dirty": False,
            "repository_revision": "a" * 40,
            "configuration_sha256": config_digest,
            "persistence_attribution": "primary_active_writes_v1",
        },
        configuration_path=config,
    )
    bundle = writer.finalize()
    attribution = raw_dir / "control" / f"{run_label}_persistence_attribution.json"
    attribution.write_text(
        json.dumps({"ratio": 0.2, "run_label": run_label}) + "\n",
        encoding="utf-8",
    )
    attribution_digest = hashlib.sha256(attribution.read_bytes()).hexdigest()
    sidecar = attribution.with_suffix(".sha256")
    sidecar.write_text(attribution_digest + "\n", encoding="utf-8")
    review_dir = raw_dir / "review"
    review_dir.mkdir()
    report = review_dir / "review_report.md"
    findings = review_dir / "review_findings.csv"
    report.write_text("accepted\n", encoding="utf-8")
    findings.write_text("gate,passed\nall,True\n", encoding="utf-8")
    review_manifest = review_dir / "review_manifest.json"
    review_manifest.write_text(
        json.dumps(
            {
                "schema_version": "stage05.2-review-v1",
                "run_label": run_label,
                "component": "artifact_streaming",
                "scope": "performance",
                "status": "READY_FOR_STAGE052_JOB_PARALLEL",
                "review_execution_required": True,
                "raw_manifest_sha256": hashlib.sha256(
                    bundle.manifest_path.read_bytes()
                ).hexdigest(),
                "persistence_attribution_sha256": attribution_digest,
                "persistence_attribution_sidecar_sha256": hashlib.sha256(
                    sidecar.read_bytes()
                ).hexdigest(),
                "gates": {"all": {"passed": True}},
                "files": {
                    report.name: hashlib.sha256(report.read_bytes()).hexdigest(),
                    findings.name: hashlib.sha256(findings.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    _bind_successful_review_execution(review_manifest)
    verify_stage052_prerequisite(
        raw_dir,
        expected_component="artifact_streaming",
        expected_status="READY_FOR_STAGE052_JOB_PARALLEL",
    )

    attribution.write_text(
        json.dumps({"ratio": 0.2, "replaced": True, "run_label": run_label}) + "\n",
        encoding="utf-8",
    )
    sidecar.write_text(
        hashlib.sha256(attribution.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactIntegrityError, match="persistence attribution"):
        verify_stage052_prerequisite(
            raw_dir,
            expected_component="artifact_streaming",
            expected_status="READY_FOR_STAGE052_JOB_PARALLEL",
        )


def test_job_parallel_selection_is_recomputed_and_binds_selected_run(tmp_path: Path) -> None:
    metrics = {
        "1": {
            "run_wall_seconds": 100.0,
            "aggregate_peak_rss_gib": 2.0,
            "speedup": 1.0,
        },
        "2": {
            "run_wall_seconds": 60.0,
            "aggregate_peak_rss_gib": 4.0,
            "speedup": 100.0 / 60.0,
        },
        "4": {
            "run_wall_seconds": 35.0,
            "aggregate_peak_rss_gib": 6.0,
            "speedup": 100.0 / 35.0,
        },
    }
    inputs = [
        "stage05.2_job_parallel_attempt04",
        "stage05.2_job_parallel_attempt05",
        "stage05.2_job_parallel_attempt06",
    ]
    input_raw_manifest_sha256: dict[str, str] = {}
    for run_label in inputs:
        bundle = ArtifactBundleWriter(
            tmp_path / run_label,
            ArtifactRunContext("stage05.2", "job_parallel", run_label),
            ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
        ).finalize()
        input_raw_manifest_sha256[run_label] = hashlib.sha256(
            bundle.manifest_path.read_bytes()
        ).hexdigest()
    raw_dir = tmp_path / inputs[2]
    review_dir = raw_dir / "review"
    review_dir.mkdir()
    selection = {
        "passed": True,
        "selected_workers": 4,
        "selected_run_label": inputs[2],
        "input_runs": inputs,
        "input_raw_manifest_sha256": input_raw_manifest_sha256,
        "resource_metrics": metrics,
    }
    review = {
        "selected_workers": 4,
        "selected_run_label": inputs[2],
        "input_runs": inputs,
        "input_raw_manifest_sha256": input_raw_manifest_sha256,
        "resource_metrics": metrics,
        "gates": {"worker_selection": selection},
    }
    review_path = review_dir / "review_manifest.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    digest = hashlib.sha256(review_path.read_bytes()).hexdigest()
    prerequisite = Stage052PrerequisiteIdentity(
        run_label=raw_dir.name,
        component="job_parallel",
        status="READY_FOR_STAGE052_NATIVE_KERNELS",
        repository_revision="a" * 40,
        configuration_sha256="b" * 64,
        raw_manifest_sha256=input_raw_manifest_sha256[inputs[2]],
        review_manifest_sha256=digest,
    )

    observed = verify_job_parallel_selection(raw_dir, prerequisite)
    assert isinstance(observed, JobParallelSelectionIdentity)
    assert observed.selected_workers == 4
    assert observed.selected_run_label == inputs[2]

    review["selected_run_label"] = inputs[1]
    review_path.write_text(json.dumps(review), encoding="utf-8")
    tampered = replace(
        prerequisite,
        review_manifest_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ArtifactIntegrityError, match="does not recompute"):
        verify_job_parallel_selection(raw_dir, tampered)


def test_process_tree_resource_summary_includes_live_child() -> None:
    sampler = ProcessTreeResourceSampler(
        run_label="stage05.2_job_parallel_attempt99",
        component="job_parallel",
        configured_worker_count=2,
        interval_seconds=0.01,
    )
    sampler.start()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; data=bytearray(8_000_000); time.sleep(.12)"],
    )
    child.wait(timeout=2.0)
    summary = sampler.stop()

    assert child.pid in summary.descendant_pids
    assert summary.aggregate_peak_rss_bytes > 8_000_000
    assert dict(summary.process_peak_rss_bytes)[child.pid] > 8_000_000
    assert summary.load1_sample_count == summary.sample_count
    assert 0.0 <= summary.load1_min <= summary.load1_mean <= summary.load1_max
    assert summary.sample_count >= 2
    assert summary.status == "complete"


def test_process_tree_resource_sampler_exposes_hard_rss_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self, pid: int, rss: int) -> None:
            self.pid = pid
            self.rss = rss

        def oneshot(self) -> FakeProcess:
            return self

        def __enter__(self) -> FakeProcess:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=self.rss)

        def cpu_times(self) -> SimpleNamespace:
            return SimpleNamespace(user=0.1, system=0.1)

    sampler = ProcessTreeResourceSampler(
        run_label="stage05.2_benchmark_attempt99",
        component="benchmark",
        configured_worker_count=6,
        interval_seconds=0.001,
        aggregate_rss_limit_bytes=1_000,
        per_process_rss_limit_bytes=900,
    )
    parent = FakeProcess(sampler.parent_pid, 200)
    oversized_worker = FakeProcess(999_997, 950)
    monkeypatch.setattr(sampler, "_processes", lambda: [parent, oversized_worker])

    sampler.start()
    deadline = time.monotonic() + 0.5
    while sampler.abort_reason() is None and time.monotonic() < deadline:
        time.sleep(0.001)
    reason = sampler.abort_reason()
    summary = sampler.stop()

    assert reason == (
        "aggregate RSS hard limit exceeded: observed=1150 limit=1000; "
        "process RSS hard limit exceeded: pid=999997 observed=950 limit=900"
    )
    assert summary.aggregate_peak_rss_bytes == 1_150
    assert dict(summary.process_peak_rss_bytes)[oversized_worker.pid] == 950
    task_started = False

    def task_runner(_task: object) -> list[dict[str, object]]:
        nonlocal task_started
        task_started = True
        return []

    with pytest.raises(
        RuntimeError,
        match="runtime guard aborted Stage 5.2 work: aggregate RSS hard limit exceeded",
    ):
        _run_v2_tasks(
            [SimpleNamespace(instance_name="c201_21", seed=2018)],  # type: ignore[arg-type]
            worker_count=1,
            abort_reason=sampler.abort_reason,
            _task_runner=task_runner,  # type: ignore[arg-type]
        )
    assert task_started is False


def test_process_tree_resource_summary_excludes_half_sampled_transient_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self, pid: int, *, disappears_before_cpu: bool = False) -> None:
            self.pid = pid
            self.disappears_before_cpu = disappears_before_cpu

        def oneshot(self) -> FakeProcess:
            return self

        def __enter__(self) -> FakeProcess:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=1024)

        def cpu_times(self) -> SimpleNamespace:
            if self.disappears_before_cpu:
                raise stage052_evidence.psutil.NoSuchProcess(self.pid)
            return SimpleNamespace(user=0.1, system=0.1)

    sampler = ProcessTreeResourceSampler(
        run_label="stage05.2_job_parallel_attempt99",
        component="job_parallel",
        configured_worker_count=1,
        interval_seconds=0.01,
    )
    parent = FakeProcess(sampler.parent_pid)
    transient = FakeProcess(999_999, disappears_before_cpu=True)
    monkeypatch.setattr(sampler, "_processes", lambda: [parent, transient])

    sampler.start()
    time.sleep(0.03)
    summary = sampler.stop()

    assert transient.pid not in summary.descendant_pids
    assert transient.pid not in dict(summary.process_peak_rss_bytes)


def test_process_tree_resource_summary_excludes_zero_rss_exited_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self, pid: int, rss: int) -> None:
            self.pid = pid
            self.rss = rss

        def oneshot(self) -> FakeProcess:
            return self

        def __enter__(self) -> FakeProcess:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=self.rss)

        def cpu_times(self) -> SimpleNamespace:
            return SimpleNamespace(user=0.1, system=0.1)

    sampler = ProcessTreeResourceSampler(
        run_label="stage05.2_native_kernels_attempt99",
        component="native_kernels",
        configured_worker_count=4,
        interval_seconds=0.01,
    )
    parent = FakeProcess(sampler.parent_pid, 1024)
    exited = FakeProcess(999_998, 0)
    monkeypatch.setattr(sampler, "_processes", lambda: [parent, exited])

    sampler.start()
    time.sleep(0.03)
    summary = sampler.stop()

    assert summary.sample_count >= 1
    assert parent.pid in dict(summary.process_peak_rss_bytes)
    assert exited.pid not in summary.descendant_pids
    assert exited.pid not in dict(summary.process_peak_rss_bytes)


def test_worker_ownership_requires_actual_sampled_pids() -> None:
    resource = {
        "schema_version": "stage05.2-run-resource-v2",
        "run_label": "stage05.2_job_parallel_attempt99",
        "component": "job_parallel",
        "configured_worker_count": 2,
        "measurement_scope": "task_scheduling_through_parent_control_preparation",
        "status": "complete",
        "run_wall_seconds": 10.0,
        "sample_interval_seconds": 0.05,
        "aggregate_peak_rss_bytes": 1024,
        "mean_active_cores": 1.5,
        "peak_active_cores": 2.0,
        "sample_count": 200,
        "parent_pid": 100,
        "descendant_pids": [201, 202, 301],
    }
    manifests = [
        {
            "run_label": "stage05.2_job_parallel_attempt99",
            "evidence_completeness": "complete",
            "shard_ordinal": 0,
            "worker_identity": "pid-201",
        },
        {
            "run_label": "stage05.2_job_parallel_attempt99",
            "evidence_completeness": "complete",
            "shard_ordinal": 1,
            "worker_identity": "pid-202",
        },
        {
            "run_label": "stage05.2_job_parallel_attempt99",
            "evidence_completeness": "complete",
            "shard_ordinal": 2,
            "worker_identity": "pid-201",
        },
    ]
    passed, _, owners = validate_worker_ownership(
        resource,
        manifests,
        expected_workers=2,
        expected_run_label="stage05.2_job_parallel_attempt99",
        expected_component="job_parallel",
    )
    assert passed
    assert owners == (201, 202)

    fake = [{**manifests[0], "worker_identity": "pid-0"}, *manifests[1:]]
    passed, detail, _ = validate_worker_ownership(
        resource,
        fake,
        expected_workers=2,
        expected_run_label="stage05.2_job_parallel_attempt99",
        expected_component="job_parallel",
    )
    assert not passed
    assert "not observed" in detail

    parent_owned = [{**manifests[0], "worker_identity": "pid-100"}, *manifests[1:]]
    passed, detail, _ = validate_worker_ownership(
        resource,
        parent_owned,
        expected_workers=2,
        expected_run_label="stage05.2_job_parallel_attempt99",
        expected_component="job_parallel",
    )
    assert not passed
    assert "not observed" in detail


def test_parallel_pool_terminates_all_workers_before_recording_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeProcess:
        pid = 999

        def terminate(self) -> None:
            events.append("terminate")

        def join(self, *, timeout: float) -> None:
            events.append(f"join:{timeout}")

        def is_alive(self) -> bool:
            return False

    class FakeFuture:
        def result(self) -> object:
            raise RuntimeError("worker failed")

        def cancel(self) -> None:
            events.append("cancel")

    class FailingExecutor:
        def __init__(self, **_: object) -> None:
            self._processes = {1: FakeProcess()}

        def submit(self, *_: object) -> FakeFuture:
            return FakeFuture()

        def shutdown(self, *, wait: bool, cancel_futures: bool = False) -> None:
            events.append(f"shutdown:{wait}:{cancel_futures}")

    tasks = [SimpleNamespace(instance_name="c101C5", seed=2014)]
    monkeypatch.setattr(stage052_performance, "ProcessPoolExecutor", FailingExecutor)
    monkeypatch.setattr(stage052_performance, "get_context", lambda _: object())
    monkeypatch.setattr(stage052_performance, "as_completed", lambda futures: iter(futures))
    monkeypatch.setattr(
        stage052_performance,
        "_ensure_partial_shard_failure",
        lambda *_: events.append("failure_evidence"),
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        _run_v2_tasks(tasks, worker_count=2)  # type: ignore[arg-type]

    assert events == [
        "cancel",
        "terminate",
        "join:2.0",
        "shutdown:True:True",
        "failure_evidence",
    ]


def test_serial_replay_aborts_current_bundle_on_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeFuture:
        def __init__(self, *, fails: bool) -> None:
            self.fails = fails

        def result(self) -> dict[tuple[str, int, str], str]:
            if self.fails:
                raise ArtifactIntegrityError("tampered replay")
            return {("c101C5", 2014, "fixed_work"): "ok"}

        def cancel(self) -> None:
            events.append("cancel")

    class FakeExecutor:
        def __init__(self, **_: object) -> None:
            return None

        def submit(self, _: object, raw_dir: Path) -> FakeFuture:
            return FakeFuture(fails=raw_dir.name == "tampered")

        def shutdown(self, *, wait: bool, cancel_futures: bool = False) -> None:
            events.append(f"shutdown:{wait}:{cancel_futures}")

    monkeypatch.setattr(stage052_review, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(stage052_review, "get_context", lambda _: object())
    monkeypatch.setattr(
        stage052_review,
        "abort_process_executor",
        lambda _: events.append("abort"),
    )

    with pytest.raises(ArtifactIntegrityError, match="tampered replay"):
        replay_stage052_storage_semantics_many((Path("tampered"), Path("healthy")))

    assert events == ["cancel", "abort"]


@pytest.mark.formal_environment
def test_performance_provenance_records_inputs_without_secret_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = tmp_path / "instance.txt"
    stage02 = tmp_path / "stage02.toml"
    stage04 = tmp_path / "stage04.toml"
    instance.write_text("instance", encoding="utf-8")
    stage02.write_text("stage02", encoding="utf-8")
    stage04.write_text("stage04", encoding="utf-8")
    native_extension = tmp_path / "_core.so"
    native_extension.write_bytes(b"native")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("SECRET_TOKEN", "must-not-be-recorded")

    provenance = collect_performance_provenance(
        instance_paths={"c101C5": instance},
        stage02_config_path=stage02,
        stage04_config_path=stage04,
        max_iterations=1000,
        batch_size=128,
        runtime_environment={
            "python": {"version": "3.13.13", "implementation": "CPython"},
            "system": {"platform": "macOS", "machine": "arm64", "cpu_count": 10},
            "packages": {"numpy": "2.4.1"},
            "native_extension": str(native_extension),
        },
    )

    assert provenance["instance_sha256"] == {
        "c101C5": hashlib.sha256(instance.read_bytes()).hexdigest()
    }
    environment = provenance["environment_variables"]
    assert isinstance(environment, dict)
    assert environment["PYTHONHASHSEED"] == "0"
    assert "SECRET_TOKEN" not in environment
    assert provenance["fallback_allowed"] is False


def test_performance_provenance_is_bound_to_captured_producer_not_reviewer(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    benchmark_dir = repository_root / "data" / "schneider"
    config_dir = repository_root / "configs"
    benchmark_dir.mkdir(parents=True)
    config_dir.mkdir()
    instance_hashes: dict[str, str] = {}
    for instance in stage052_review.PERFORMANCE_INSTANCES:
        path = benchmark_dir / f"{instance}.txt"
        path.write_text(instance, encoding="utf-8")
        instance_hashes[instance] = hashlib.sha256(path.read_bytes()).hexdigest()
    stage02 = config_dir / "stage02_constraint_guided.toml"
    stage04 = config_dir / "stage04_weights.toml"
    stage02.write_text("stage02", encoding="utf-8")
    stage04.write_text("stage04", encoding="utf-8")
    producer_native = tmp_path / "producer_core.so"
    producer_native.write_bytes(b"producer-native")
    producer_environment = {
        "python": {"version": "3.13.13", "implementation": "CPython"},
        "system": {
            "platform": "Linux-producer",
            "machine": "x86_64",
            "processor": "x86_64",
            "cpu_count": 24,
            "gpu_used": False,
        },
        "packages": {"numpy": "2.4.1", "pyarrow": "23.0.1"},
        "native_extension": str(producer_native),
    }
    provenance = {
        "schema_version": "stage05.2-performance-provenance-v1",
        "instance_sha256": instance_hashes,
        "warm_start": {"enabled": False, "source": None},
        "operator_surface": {
            "operator_profile": "stage02_constraint_guided",
            "stage02_config_sha256": hashlib.sha256(stage02.read_bytes()).hexdigest(),
            "stage04_config_sha256": hashlib.sha256(stage04.read_bytes()).hexdigest(),
        },
        "fixed_work_contract": {
            "exact_call_budget": 100,
            "watchdog_seconds": 120.0,
            "max_iterations": 1000,
            "batch_size": 128,
            "backend": "cpu_batch",
        },
        "worker_affinity": {"supported": True, "cpu_ids": [0, 1]},
        "environment_variables": {
            name: None for name in stage052_review._PERFORMANCE_ENVIRONMENT_VARIABLES
        },
        "background_load": {
            "load_average": [0.1, 0.2, 0.3],
            "process_status_counts": {"running": 1},
        },
        "power_mode": {
            "available": True,
            "source": "AC Power",
            "low_power_mode": 0,
        },
        "runtime_signature": {
            "python": producer_environment["python"],
            "system": producer_environment["system"],
            "packages": producer_environment["packages"],
            "native_extension_sha256": hashlib.sha256(producer_native.read_bytes()).hexdigest(),
        },
        "failure_policy": "abort_all_workers_without_fallback",
        "fallback_allowed": False,
    }
    metadata = {
        "environment": producer_environment,
        "optimization_profile": "native",
        "performance_provenance": provenance,
        "runtime_identity": {
            "python_version": "3.13.13",
            "dependency_versions": producer_environment["packages"],
            "native_extension_sha256": provenance["runtime_signature"]["native_extension_sha256"],
            "machine_identity": {"logical_cpu_count": 24},
        },
    }
    passed, detail = stage052_review._validate_performance_provenance(
        metadata,
        benchmark_dir=benchmark_dir,
    )

    assert passed, detail
    assert detail == "performance provenance passed"
    mismatched = {
        **metadata,
        "performance_provenance": {
            **provenance,
            "runtime_signature": {
                **provenance["runtime_signature"],
                "packages": {"numpy": "tampered"},
            },
        },
    }
    passed, detail = stage052_review._validate_performance_provenance(
        mismatched,
        benchmark_dir=benchmark_dir,
    )
    assert not passed
    assert detail == "runtime signature does not match the captured environment"


def test_stage052_runtime_identity_binds_wheel_python_native_and_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "evrptw_reproduction-0.1.0-cp313-cp313-macosx.whl"
    wheel.write_bytes(b"wheel-under-test")
    manifest = tmp_path / "stage052_runtime.local.json"
    monkeypatch.setattr(stage052_evidence, "_distribution_is_editable", lambda: False)
    monkeypatch.setattr(
        stage052_evidence,
        "_stage052_machine_identity",
        lambda: {"execution_environment": "windows11_wsl2_test"},
    )
    monkeypatch.setattr(stage052_evidence, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        stage052_evidence,
        "_findmnt_identity",
        lambda _path: {
            "source": "/dev/test",
            "filesystem": "ext4",
            "uuid": "test-uuid",
            "target": "/",
        },
    )

    created = create_stage052_runtime_identity(
        output_path=manifest,
        wheel_path=wheel,
        repository_revision="a" * 40,
    )
    verified = verify_stage052_runtime_identity(
        manifest,
        expected_repository_revision="a" * 40,
    )

    assert verified == created
    assert verified["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert "wheel_path" not in verified
    assert "python_executable" not in verified
    assert verified["installed_editable"] is False
    monkeypatch.setattr(
        stage052_evidence,
        "_stage052_machine_identity",
        lambda: {
            "execution_environment": "different-host-telemetry",
            "logical_cpu_count": 32,
        },
    )
    telemetry_drift = verify_stage052_runtime_identity(
        manifest,
        expected_repository_revision="a" * 40,
    )
    assert telemetry_drift["wheel_sha256"] == verified["wheel_sha256"]
    assert telemetry_drift["machine_identity"] != verified["machine_identity"]
    monkeypatch.setattr(
        stage052_review,
        "_verify_frozen_producer_runtime_identity",
        lambda *_args, **_kwargs: verified,
        raising=False,
    )
    passed, detail = stage052_review._validate_stage052_runtime_identity(
        {"repository_revision": "a" * 40, "runtime_identity": verified}
    )
    assert passed, detail
    passed, detail = stage052_review._validate_stage052_runtime_identity(
        {
            "repository_revision": "a" * 40,
            "runtime_identity": {**verified, "wheel_sha256": "0" * 64},
        }
    )
    assert not passed
    assert "does not match" in detail

    wheel.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="wheel hash"):
        verify_stage052_runtime_identity(
            manifest,
            expected_repository_revision="a" * 40,
        )


def test_stage052_reviewer_replays_the_receipt_bound_producer_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer_source = tmp_path / "sealed-producer"
    reviewer_source = tmp_path / "reviewer-worktree"
    producer_source.mkdir()
    reviewer_source.mkdir()
    (reviewer_source / ".ruff_cache").mkdir()
    receipt = tmp_path / "review_execution.json"
    receipt.write_text(
        json.dumps(
            {
                "producer_source_directory": str(producer_source),
                "working_directory": str(reviewer_source),
                "reviewer_working_directory": str(reviewer_source),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("STAGE052_REVIEW_EXECUTION_RECEIPT", str(receipt))
    monkeypatch.setattr(stage052_review, "find_repository_root", lambda: reviewer_source)

    source_identity = {
        "repository_revision": "a" * 40,
        "mount": {"filesystem": "ext4"},
        "tracked_file_count": 1,
        "allowed_untracked_sha256": {},
        "read_only": True,
    }

    def verify_source(root: Path) -> dict[str, object]:
        assert root == producer_source
        return source_identity

    def verify_runtime(root: Path, revision: str) -> dict[str, object]:
        assert root == producer_source
        assert revision == "a" * 40
        return {"wheel_sha256": "b" * 64}

    monkeypatch.setattr(stage052_review, "verify_stage052_source_snapshot", verify_source)
    monkeypatch.setattr(
        stage052_review,
        "_verify_frozen_producer_runtime_identity",
        verify_runtime,
    )

    passed, detail = stage052_review._validate_stage052_source_snapshot(
        {"source_snapshot": source_identity}
    )
    assert passed, detail
    passed, detail = stage052_review._validate_stage052_runtime_identity(
        {
            "repository_revision": "a" * 40,
            "runtime_identity": {"wheel_sha256": "b" * 64},
        }
    )
    assert passed, detail
    receipt.write_text(
        json.dumps(
            {
                "working_directory": str(producer_source),
                "reviewer_working_directory": str(reviewer_source),
            }
        ),
        encoding="utf-8",
    )
    passed, detail = stage052_review._validate_stage052_source_snapshot(
        {"source_snapshot": source_identity}
    )
    assert passed, detail


def test_review_runtime_machine_comparison_excludes_only_wsl_memory_limit() -> None:
    frozen = {
        "execution_environment": "windows11_wsl2",
        "memory_bytes": 15 * 1024**3,
        "logical_cpu_count": 24,
        "nvidia_gpu": {"name": "test-gpu"},
    }
    current = {**frozen, "memory_bytes": 16 * 1024**3}

    assert stage052_evidence._same_producer_machine_ignoring_review_memory(frozen, current)
    assert not stage052_evidence._same_producer_machine_ignoring_review_memory(
        frozen,
        {**current, "logical_cpu_count": 12},
    )


def test_review_runtime_machine_comparison_normalizes_windows_caption_locale() -> None:
    frozen = {
        "execution_environment": "windows11_wsl2",
        "windows": {
            "Caption": "Microsoft Windows 11 专业工作站版",
            "Version": "10.0.26200",
            "BuildNumber": "26200",
            "TotalVisibleMemorySize": 32629612,
        },
    }
    current = {
        **frozen,
        "windows": {
            **frozen["windows"],
            "Caption": "Microsoft Windows 11 Pro for Workstations",
        },
    }

    assert stage052_evidence._same_producer_machine_ignoring_review_memory(frozen, current)
    assert not stage052_evidence._same_producer_machine_ignoring_review_memory(
        frozen,
        {
            **current,
            "windows": {
                **current["windows"],
                "Caption": "Microsoft Windows 11 Home",
            },
        },
    )


def test_windows_operating_system_identity_uses_locale_independent_sku() -> None:
    chinese = {
        "Caption": "Microsoft Windows 11 专业工作站版",
        "Version": "10.0.26200",
        "BuildNumber": "26200",
        "TotalVisibleMemorySize": 32629612,
        "OperatingSystemSKU": 161,
    }
    english = {
        **chinese,
        "Caption": "Microsoft Windows 11 Pro for Workstations",
    }

    expected = {
        "Version": "10.0.26200",
        "BuildNumber": "26200",
        "TotalVisibleMemorySize": 32629612,
        "OperatingSystemSKU": 161,
    }
    assert stage052_evidence._canonical_windows_operating_system_identity(chinese) == expected
    assert stage052_evidence._canonical_windows_operating_system_identity(english) == expected
    with pytest.raises(RuntimeError, match="OperatingSystemSKU"):
        stage052_evidence._canonical_windows_operating_system_identity(
            {**english, "OperatingSystemSKU": True}
        )


def test_runtime_signature_is_self_consistent_without_impersonating_producer() -> None:
    environment = {
        "python": {"version": "3.13.13", "implementation": "CPython"},
        "system": {"machine": "arm64"},
        "packages": {"numpy": "2.4.6"},
        "native_extension": "/historical/evrptw/_core.cpython-313-darwin.so",
    }
    historical_signature = {
        "python": environment["python"],
        "system": environment["system"],
        "packages": environment["packages"],
        "native_extension_sha256": "a" * 64,
    }
    passed, detail = stage052_review._validate_captured_runtime_signature(
        environment=environment,
        runtime_signature=historical_signature,
        optimization_profile="python",
    )
    assert passed, detail

    passed, detail = stage052_review._validate_captured_runtime_signature(
        environment=environment,
        runtime_signature=historical_signature,
        optimization_profile="native",
    )
    assert passed, detail

    passed, detail = stage052_review._validate_captured_runtime_signature(
        environment=environment,
        runtime_signature=historical_signature,
        optimization_profile="unknown",
    )
    assert not passed
    assert "optimization profile" in detail

    malformed = {**historical_signature, "native_extension_sha256": "not-a-hash"}
    passed, _ = stage052_review._validate_captured_runtime_signature(
        environment=environment,
        runtime_signature=malformed,
        optimization_profile="python",
    )
    assert not passed


def test_reviewer_infers_historical_v2_screening_schema_from_artifacts() -> None:
    reader = SimpleNamespace(
        manifest={
            "storage_policy": {"storage_policy_version": "artifact-storage-v2"},
            "artifacts": [
                {
                    "artifact_type": "events",
                    "artifact_subtype": "screening_decisions_v2",
                }
            ],
        }
    )

    assert stage052_review._screening_schema_version(reader) == "screening_decisions_v2"


def test_reviewer_recognises_v1_screening_schema_embedded_in_critical_events() -> None:
    trace_paths = ("c101C5/2014/trace.json", "c101C5/2015/trace.json")
    traces = {
        path: {
            "events_ref": path.replace("trace.json", "events.parquet"),
            "screening_checks_ref": path.replace("trace.json", "checks.parquet"),
        }
        for path in trace_paths
    }
    reader = SimpleNamespace(
        manifest={
            "storage_policy": {"screening_schema_version": "screening_decisions_v1"},
            "artifacts": [
                {"artifact_type": "events", "artifact_subtype": "critical"},
                {"artifact_type": "events", "artifact_subtype": "screening_checks"},
                *({"artifact_type": "trace", "relative_path": path} for path in trace_paths),
            ],
        },
        read_json=lambda path: traces[path],
    )

    assert stage052_review._screening_schema_version(reader) == "screening_decisions_v1"


def test_reviewer_rejects_v1_declaration_with_v2_trace_reference() -> None:
    reader = SimpleNamespace(
        manifest={
            "storage_policy": {"screening_schema_version": "screening_decisions_v1"},
            "artifacts": [
                {"artifact_type": "events", "artifact_subtype": "critical"},
                {"artifact_type": "events", "artifact_subtype": "screening_checks"},
                {"artifact_type": "trace", "relative_path": "trace.json"},
            ],
        },
        read_json=lambda _path: {
            "screening_schema_version": "screening_decisions_v1",
            "events_ref": "events.parquet",
            "screening_checks_ref": "checks.parquet",
            "screening_decisions_ref": "screening_decisions.parquet",
        },
    )

    assert stage052_review._screening_schema_version(reader) is None


def test_reviewer_rejects_declared_and_physical_screening_schema_mismatch() -> None:
    mismatch = SimpleNamespace(
        manifest={
            "storage_policy": {"screening_schema_version": "screening_decisions_v2"},
            "artifacts": [
                {"artifact_type": "events", "artifact_subtype": "screening_definitions_v3"},
                {"artifact_type": "events", "artifact_subtype": "screening_occurrences_v3"},
            ],
        }
    )
    mixed = SimpleNamespace(
        manifest={
            "storage_policy": {},
            "artifacts": [
                {"artifact_type": "events", "artifact_subtype": "screening_decisions_v2"},
                {"artifact_type": "events", "artifact_subtype": "screening_definitions_v3"},
                {"artifact_type": "events", "artifact_subtype": "screening_occurrences_v3"},
            ],
        }
    )

    assert stage052_review._screening_schema_version(mismatch) is None
    assert stage052_review._screening_schema_version(mixed) is None


def test_storage_digest_ignores_derived_producer_instrumentation() -> None:
    historical = {
        "objective_key": [2, 10.0, 0.0, 0],
        "backend_metrics": {"exact_calls": 100},
    }
    current = {
        **historical,
        "anytime_checkpoints": [],
        "initial_objective_key": [],
        "iteration_limit_completed_at_seconds": 0.75,
        "trace_reconciliation": {"status": "pass", "checks": {"new_name": True}},
        "unique_route_semantics": "completed_cache_owner_identity_v2",
        "backend_metrics": {
            "exact_calls": 100,
            "label_management_seconds": 0.5,
            "transition_seconds": 0.1,
            "launch_occupancies": [1, 2],
            "native_invocations": 0,
            "native_fallbacks": 0,
        },
    }

    historical_semantics = stage052_review._canonical_axis_semantics(
        raw_axis={**historical, "started_calls": 100, "completed_calls": 100},
        solution_axis={"objective_key": [2, 10.0, 0.0, 0]},
        trace={},
        axis="fixed_work",
    )
    current_semantics = stage052_review._canonical_axis_semantics(
        raw_axis={**current, "started_calls": 100, "completed_calls": 100},
        solution_axis={
            "objective_key": [2, 10.0, 0.0, 0],
            "initial_objective_key": [],
            "initial_routes": [],
        },
        trace={},
        axis="fixed_work",
    )

    assert current_semantics == historical_semantics

    native_trace = {
        "axes": {
            "fixed_work": {
                "result_summary": {
                    "screening_statistics": {
                        "native_screening_invocations": 0,
                        "native_propagation_invocations": 0,
                        "native_protocol_fallbacks": 0,
                    }
                }
            }
        }
    }
    native_observability_baseline = stage052_review._canonical_axis_semantics(
        raw_axis={**current, "started_calls": 100, "completed_calls": 100},
        solution_axis={"objective_key": [2, 10.0, 0.0, 0]},
        trace=native_trace,
        axis="fixed_work",
    )
    native_observability_semantics = stage052_review._canonical_axis_semantics(
        raw_axis={
            **current,
            "started_calls": 100,
            "completed_calls": 100,
            "backend_metrics": {
                **current["backend_metrics"],
                "checkpoint_count": 824,
                "native_invocations": 12,
            },
        },
        solution_axis={"objective_key": [2, 10.0, 0.0, 0]},
        trace={
            "axes": {
                "fixed_work": {
                    "result_summary": {
                        "screening_statistics": {
                            "native_screening_invocations": 4499,
                            "native_propagation_invocations": 17,
                            "native_protocol_fallbacks": 0,
                        }
                    }
                }
            }
        },
        axis="fixed_work",
    )
    assert native_observability_semantics == native_observability_baseline

    historical_trace_semantics = stage052_review._canonical_axis_semantics(
        raw_axis={**historical, "started_calls": 100, "completed_calls": 100},
        solution_axis={"objective_key": [2, 10.0, 0.0, 0]},
        trace={"axes": {"fixed_work": {"result_summary": {"exact_calls": 100}}}},
        axis="fixed_work",
    )
    current_trace_semantics = stage052_review._canonical_axis_semantics(
        raw_axis={**historical, "started_calls": 100, "completed_calls": 100},
        solution_axis={"objective_key": [2, 10.0, 0.0, 0]},
        trace={
            "axes": {
                "fixed_work": {
                    "unique_route_semantics": "completed_cache_owner_identity_v2",
                    "result_summary": {
                        "exact_calls": 100,
                        "unique_route_semantics": "completed_cache_owner_identity_v2",
                    },
                }
            }
        },
        axis="fixed_work",
    )

    assert current_trace_semantics == historical_trace_semantics
    backend_metrics = current["backend_metrics"]
    assert isinstance(backend_metrics, dict)
    backend_metrics["native_fallbacks"] = 1
    fallback_semantics = stage052_review._canonical_axis_semantics(
        raw_axis={**current, "started_calls": 100, "completed_calls": 100},
        solution_axis={"objective_key": [2, 10.0, 0.0, 0]},
        trace={},
        axis="fixed_work",
    )
    assert fallback_semantics != historical_semantics
    mutated_semantics = stage052_review._canonical_axis_semantics(
        raw_axis={
            **current,
            "started_calls": 100,
            "completed_calls": 100,
            "unique_route_semantics": "started_lane_identity_legacy_v1",
        },
        solution_axis={"objective_key": [2, 10.0, 0.0, 0]},
        trace={},
        axis="fixed_work",
    )
    assert mutated_semantics != historical_semantics
    with pytest.raises(ArtifactIntegrityError, match="reconciliation did not pass"):
        stage052_review._canonical_axis_semantics(
            raw_axis={
                **current,
                "started_calls": 100,
                "completed_calls": 100,
                "trace_reconciliation": {"status": "fail", "checks": {"x": False}},
            },
            solution_axis={"objective_key": [2, 10.0, 0.0, 0]},
            trace={},
            axis="fixed_work",
        )


def test_streamed_record_counts_must_match_independent_replay() -> None:
    observed = {
        "events": 3,
        "incremental_propagations": 1,
        "route_evaluations": 2,
        "screening_decisions": 4,
    }
    stage052_review._validate_streamed_record_counts(observed, observed)

    with pytest.raises(ArtifactIntegrityError, match="do not match"):
        stage052_review._validate_streamed_record_counts(
            {**observed, "screening_decisions": 999},
            observed,
        )
    with pytest.raises(ArtifactIntegrityError, match="do not match"):
        stage052_review._validate_streamed_record_counts(
            {"events": 3},
            observed,
        )
    with pytest.raises(ArtifactIntegrityError, match="required for v3"):
        stage052_review._validate_streamed_record_counts(None, observed, required=True)


def test_native_persistence_pipeline_requires_bounded_complete_fifo_batches() -> None:
    valid = {
        "mode": "bounded_async_thread",
        "queue_max_batches": 1,
        "writer_thread_switch_interval_seconds": (
            stage052_performance.STAGE052_WRITER_THREAD_SWITCH_INTERVAL_SECONDS
        ),
        "submitted_batches": 10,
        "completed_batches": 10,
        "writer_active_nanoseconds": 100,
        "writer_cpu_nanoseconds": 70,
        "producer_active_nanoseconds": 80,
        "persistence_union_nanoseconds": 150,
        "solver_persistence_union_nanoseconds": 120,
        "solver_persistence_critical_path_nanoseconds": 70,
        "solver_producer_active_nanoseconds": 60,
        "solver_writer_cpu_nanoseconds": 70,
        "producer_wait_nanoseconds": 50,
        "peak_queued_batches": 1,
        "batch_ledger": [
            {
                "ordinal": ordinal,
                "row_count": 1,
                "event_token_sha256": "0" * 64,
            }
            for ordinal in range(10)
        ],
    }
    stage052_review._validate_persistence_pipeline(valid)

    with pytest.raises(ArtifactIntegrityError, match="pipeline evidence is invalid"):
        stage052_review._validate_persistence_pipeline({**valid, "completed_batches": 9})
    with pytest.raises(ArtifactIntegrityError, match="pipeline evidence is invalid"):
        stage052_review._validate_persistence_pipeline(
            {**valid, "solver_persistence_union_nanoseconds": 69}
        )
    oversized = dict(valid)
    oversized["batch_ledger"] = [
        {**entry, "row_count": 65_537} if index == 0 else entry
        for index, entry in enumerate(valid["batch_ledger"])
    ]
    with pytest.raises(ArtifactIntegrityError, match="batch ledger is invalid"):
        stage052_review._validate_persistence_pipeline(oversized)
    with pytest.raises(ArtifactIntegrityError, match="pipeline evidence is missing"):
        stage052_review._validate_persistence_pipeline(None)


def test_persistence_pipeline_metadata_is_nonsemantic() -> None:
    base = stage052_review._canonical_semantic_value({"objective_key": [2, 10.0, 0.0, 0]})
    pipelined = stage052_review._canonical_semantic_value(
        {
            "objective_key": [2, 10.0, 0.0, 0],
            "persistence_pipeline": {
                "mode": "bounded_async_thread",
                "submitted_batches": 10,
            },
        }
    )

    assert pipelined == base


def test_native_screening_diagnostic_float_canonicalization_is_narrow() -> None:
    base = {
        "event_type": "screening_decision",
        "status": "rejected",
        "reason": "forward_time_window_prefilter",
        "min_time_window_slack": -110.24146527838172,
        "distance_lower_bound": 263.64617382155,
        "checks": [
            {
                "check": "forward_time_window",
                "status": "fail",
                "reason": "forward_time_window_prefilter",
                "value": -110.24146527838172,
            }
        ],
    }
    native_roundoff = {
        **base,
        "min_time_window_slack": -110.24146527838175,
        "distance_lower_bound": 263.64617382155006,
        "checks": [{**base["checks"][0], "value": -110.24146527838175}],
    }

    canonical = stage052_review._canonicalize_screening_diagnostic_floats(base)
    assert canonical == stage052_review._canonicalize_screening_diagnostic_floats(native_roundoff)
    assert canonical != stage052_review._canonicalize_screening_diagnostic_floats(
        {**native_roundoff, "distance_lower_bound": 263.646173823}
    )
    assert canonical != stage052_review._canonicalize_screening_diagnostic_floats(
        {**native_roundoff, "status": "pass"}
    )
    assert stage052_review._canonicalize_screening_diagnostic_floats(
        {"event_type": "route_evaluation", "distance_lower_bound": 263.64617382155006}
    ) == {"event_type": "route_evaluation", "distance_lower_bound": 263.64617382155006}


@pytest.mark.formal_environment
def test_storage_semantic_replay_is_independent_and_equal_for_v1_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def write(
        policy: str,
        component: str,
        label: str,
        *,
        axis: str = "fixed_work",
        route_evaluation_status: str = "completed_feasible",
        extra_solution_axis: bool = False,
        extra_unreferenced_route: bool = False,
        fixed_route_customer: str = "C1",
        cache_bytes: int = 396,
        screening_schema_version: str = "screening_decisions_v2",
        repeated_event_count: int = 0,
    ) -> Path:
        run_dir = tmp_path / label
        storage_kwargs: dict[str, object] = {"storage_policy_version": policy}
        if policy == "artifact-storage-v2":
            storage_kwargs["screening_schema_version"] = screening_schema_version
        writer = ArtifactBundleWriter(
            run_dir,
            ArtifactRunContext("stage05.2", component, label),
            ArtifactStorageConfig(**storage_kwargs),
        )
        solution_axes: dict[str, object] = {
            axis: {
                "routes": [["C1"]],
                "objective_key": [1, 10.0, 0.0, 0],
            }
        }
        if extra_solution_axis:
            solution_axes["forged_axis"] = {
                "routes": [["C1"]],
                "objective_key": [1, 10.0, 0.0, 0],
            }
        fixed_route_key = "route:2:C1"
        route_dictionary = {fixed_route_key: (fixed_route_customer,)}
        if extra_unreferenced_route:
            route_dictionary["route:2:W1"] = ("W1",)
        repeated_events = [
            {
                "event_type": "candidate_state",
                "benchmark_axis": axis,
                "status": "rejected",
                "accepted": False,
                "global_best": False,
                "operation": "candidate_rejected",
                "cache_key_digest": f"repeat-{index}",
            }
            for index in range(repeated_event_count)
        ]
        writer.write_instance_seed(
            instance="c101_21",
            seed=2014,
            raw_payload={
                "instance": "c101_21",
                "seed": 2014,
                "axes": {axis: {"started_calls": 100, "completed_calls": 100}},
            },
            solution_payload={
                "instance": "c101_21",
                "seed": 2014,
                "axes": solution_axes,
            },
            trace_payload={},
            environment_payload={},
            route_dictionary=route_dictionary,
            critical_events=[
                {
                    "event_type": "candidate_state",
                    "benchmark_axis": axis,
                    "status": "rejected",
                    "accepted": False,
                    "global_best": False,
                    "operation": "candidate_rejected",
                    "cache_key_digest": "abc",
                },
                {
                    "event_type": "route_evaluation",
                    "benchmark_axis": axis,
                    "status": route_evaluation_status,
                    "kind": "exact_call",
                    "exact_started": True,
                    "exact_completed": True,
                    "evaluation_id": 1,
                    "route_key": fixed_route_key,
                },
                {
                    "event_type": "cache_event",
                    "benchmark_axis": axis,
                    "operation": "store",
                    "cache_key_digest": "abc",
                    "entry_bytes": cache_bytes,
                    "current_bytes": cache_bytes,
                    "current_entries": 1,
                },
                *repeated_events,
            ],
            shard_ordinal=0 if policy == "artifact-storage-v2" else None,
            worker_identity=("worker-0" if policy == "artifact-storage-v2" else None),
        )
        writer.finalize()
        return run_dir

    v1 = write(
        "artifact-storage-v1",
        "hot_path",
        "stage05.2_hot_path_attempt99",
    )
    v2 = write(
        "artifact-storage-v2",
        "artifact_streaming",
        "stage05.2_artifact_streaming_attempt99",
    )
    v3 = write(
        "artifact-storage-v2",
        "artifact_streaming",
        "stage05.2_artifact_streaming_attempt91",
        screening_schema_version="screening_decisions_v3",
    )

    released_artifacts: list[str] = []
    original_artifact_cache_drop = artifacts_module._drop_file_page_cache

    def record_artifact_cache_drop(handle: object) -> None:
        released_artifacts.append(str(getattr(handle, "name", "")))
        original_artifact_cache_drop(handle)

    monkeypatch.setattr(
        artifacts_module,
        "_drop_file_page_cache",
        record_artifact_cache_drop,
    )
    assert replay_stage052_storage_semantics(v1) == replay_stage052_storage_semantics(v2)
    assert any(path.endswith(".parquet") for path in released_artifacts)
    assert replay_stage052_storage_semantics(v1) == replay_stage052_storage_semantics(v3)
    serial_replays = replay_stage052_storage_semantics_many((v1, v2))
    assert serial_replays[0] == serial_replays[1]
    volatile_cache_bytes = write(
        "artifact-storage-v2",
        "artifact_streaming",
        "stage05.2_artifact_streaming_attempt90",
        cache_bytes=397,
    )
    assert replay_stage052_storage_semantics(v1) == replay_stage052_storage_semantics(
        volatile_cache_bytes
    )
    wall_route_only = write(
        "artifact-storage-v2",
        "artifact_streaming",
        "stage05.2_artifact_streaming_attempt96",
        extra_unreferenced_route=True,
    )
    assert replay_stage052_storage_semantics(v1) == replay_stage052_storage_semantics(
        wall_route_only
    )

    changed_event = write(
        "artifact-storage-v2",
        "artifact_streaming",
        "stage05.2_artifact_streaming_attempt98",
        route_evaluation_status="completed_infeasible",
    )
    assert replay_stage052_storage_semantics(v1) != replay_stage052_storage_semantics(changed_event)
    mismatch_lines = render_semantic_mismatches(changed_event, (v1,)).decode().splitlines()
    assert mismatch_lines[0] == ("instance,seed,axis,ordinal,field,left_digest,right_digest")
    assert mismatch_lines[1].startswith("c101_21,2014,fixed_work,2,event.propagation_status,")
    streamed_mismatches = tmp_path / "streamed-mismatches.csv"
    field_progress = tmp_path / "field-progress.jsonl"
    monkeypatch.setenv("STAGE052_REVIEW_PROGRESS_LOG", str(field_progress))
    write_semantic_mismatches(changed_event, (v1,), streamed_mismatches)
    assert streamed_mismatches.read_bytes() == ("\n".join(mismatch_lines) + "\n").encode()
    field_events = [
        json.loads(line)
        for line in field_progress.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"]
        in {"semantic_spool_bundle_complete", "semantic_stream_bundle_complete"}
    ]
    assert {event["event"] for event in field_events} == {
        "semantic_spool_bundle_complete",
        "semantic_stream_bundle_complete",
    }
    assert len({event["pid"] for event in field_events}) == 2
    assert all(event["pid"] != os.getpid() for event in field_events)

    wall_clock_prior = write(
        "artifact-storage-v2",
        "hot_path",
        "stage05.2_hot_path_attempt88",
        axis="wall_clock_30",
    )
    wall_clock_candidate = write(
        "artifact-storage-v2",
        "hot_path",
        "stage05.2_hot_path_attempt89",
        axis="wall_clock_30",
        route_evaluation_status="completed_infeasible",
    )
    summarized_wall_clock = (
        render_semantic_mismatches(
            wall_clock_candidate,
            (wall_clock_prior,),
            detailed_axis_prefixes=("fixed_work",),
        )
        .decode()
        .splitlines()
    )
    assert len(summarized_wall_clock) == 2
    assert summarized_wall_clock[1].startswith("c101_21,2014,wall_clock_30,-1,axis_digest_summary,")
    ordered_replays = replay_stage052_storage_semantics_many((changed_event, v1))
    assert ordered_replays == [
        replay_stage052_storage_semantics(changed_event),
        replay_stage052_storage_semantics(v1),
    ]

    progress_log = tmp_path / "review-progress.jsonl"
    previous_progress = os.environ.get("STAGE052_REVIEW_PROGRESS_LOG")
    os.environ["STAGE052_REVIEW_PROGRESS_LOG"] = str(progress_log)
    try:
        replay_stage052_storage_semantics_many((v1, v2))
    finally:
        if previous_progress is None:
            os.environ.pop("STAGE052_REVIEW_PROGRESS_LOG", None)
        else:
            os.environ["STAGE052_REVIEW_PROGRESS_LOG"] = previous_progress
    replay_starts = [
        json.loads(line)
        for line in progress_log.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"] == "storage_replay_start"
    ]
    worker_pids = [int(event["pid"]) for event in replay_starts]
    assert len(worker_pids) == 2
    assert len(set(worker_pids)) == 2
    assert os.getpid() not in worker_pids

    changed_route = write(
        "artifact-storage-v2",
        "artifact_streaming",
        "stage05.2_artifact_streaming_attempt95",
        fixed_route_customer="C2",
    )
    assert replay_stage052_storage_semantics(v1) != replay_stage052_storage_semantics(changed_route)

    extra_axis = write(
        "artifact-storage-v2",
        "artifact_streaming",
        "stage05.2_artifact_streaming_attempt97",
        extra_solution_axis=True,
    )
    with pytest.raises(ArtifactIntegrityError, match="axis identity mismatch"):
        replay_stage052_storage_semantics(extra_axis)

    fail_fast_progress = tmp_path / "fail-fast-progress.jsonl"
    previous_progress = os.environ.get("STAGE052_REVIEW_PROGRESS_LOG")
    os.environ["STAGE052_REVIEW_PROGRESS_LOG"] = str(fail_fast_progress)
    try:
        with pytest.raises(ArtifactIntegrityError, match="axis identity mismatch"):
            replay_stage052_storage_semantics_many((extra_axis, v1))
    finally:
        if previous_progress is None:
            os.environ.pop("STAGE052_REVIEW_PROGRESS_LOG", None)
        else:
            os.environ["STAGE052_REVIEW_PROGRESS_LOG"] = previous_progress
    started_directories = [
        json.loads(line)["raw_dir"]
        for line in fail_fast_progress.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"] == "bundle_replay_start"
    ]
    assert started_directories == [str(extra_axis.resolve())]

    memory_fixture = write(
        "artifact-storage-v2",
        "artifact_streaming",
        "stage05.2_artifact_streaming_attempt89",
        repeated_event_count=20_000,
    )
    spool_root = tmp_path / "spool-root"
    spool_root.mkdir()
    spool_progress = tmp_path / "spool-progress.jsonl"
    previous_tmp = os.environ.get("STAGE052_REVIEW_TMPDIR")
    previous_progress = os.environ.get("STAGE052_REVIEW_PROGRESS_LOG")
    os.environ["STAGE052_REVIEW_TMPDIR"] = str(spool_root)
    os.environ["STAGE052_REVIEW_PROGRESS_LOG"] = str(spool_progress)
    monkeypatch.setattr(
        stage052_review,
        "_SQLITE_CACHE_RELEASE_INTERVAL_RECORDS",
        1,
    )
    large_mismatches = tmp_path / "large-mismatches.csv"
    try:
        write_semantic_mismatches(memory_fixture, (v1,), large_mismatches)
    finally:
        if previous_tmp is None:
            os.environ.pop("STAGE052_REVIEW_TMPDIR", None)
        else:
            os.environ["STAGE052_REVIEW_TMPDIR"] = previous_tmp
        if previous_progress is None:
            os.environ.pop("STAGE052_REVIEW_PROGRESS_LOG", None)
        else:
            os.environ["STAGE052_REVIEW_PROGRESS_LOG"] = previous_progress
    with large_mismatches.open("rb") as mismatch_handle:
        assert sum(1 for _ in mismatch_handle) > 20_000
    spool_sizes = [
        int(event["spool_bytes"])
        for event in map(
            json.loads,
            spool_progress.read_text(encoding="utf-8").splitlines(),
        )
        if event["event"] == "semantic_spool_bundle_complete"
    ]
    streamed_sizes = [
        int(event["spool_bytes"])
        for event in map(
            json.loads,
            spool_progress.read_text(encoding="utf-8").splitlines(),
        )
        if event["event"] == "semantic_stream_bundle_complete"
    ]
    assert len(spool_sizes) == 1
    assert streamed_sizes == spool_sizes
    assert max(spool_sizes) < 16 * 1024 * 1024
    assert any(
        event["event"] == "semantic_spool_cache_release"
        for event in map(
            json.loads,
            spool_progress.read_text(encoding="utf-8").splitlines(),
        )
    )
    assert not any(spool_root.iterdir())

    cache_flushes = 0
    original_cache_flush = stage052_review._flush_sync_and_drop_file_cache

    def record_cache_flush(handle: object) -> None:
        nonlocal cache_flushes
        cache_flushes += 1
        original_cache_flush(handle)

    monkeypatch.setattr(stage052_review, "_CACHE_RELEASE_INTERVAL_BYTES", 1024 * 1024)
    monkeypatch.setattr(
        stage052_review,
        "_SQLITE_CACHE_RELEASE_INTERVAL_RECORDS",
        10_000,
    )
    monkeypatch.setattr(
        stage052_review,
        "_flush_sync_and_drop_file_cache",
        record_cache_flush,
    )
    left_only_mismatches = tmp_path / "left-only-mismatches.csv"
    write_semantic_mismatches(v1, (memory_fixture,), left_only_mismatches)
    with left_only_mismatches.open("rb") as mismatch_handle:
        assert sum(1 for _ in mismatch_handle) > 20_000
    assert cache_flushes > 1
    replay_probe = """
import json
import os
import sys
from pathlib import Path
from evrptw.experiments.stage052_performance_review import replay_stage052_storage_semantics

replay_stage052_storage_semantics(Path(sys.argv[1]))
print(json.dumps({"pid": os.getpid()}))
"""

    def replay_peak(path: Path) -> dict[str, int]:
        process = subprocess.Popen(
            (sys.executable, "-c", replay_probe, str(path)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        observed = psutil.Process(process.pid)
        peak_rss_bytes = 0
        while process.poll() is None:
            try:
                peak_rss_bytes = max(peak_rss_bytes, observed.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                break
            time.sleep(0.002)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise AssertionError(stderr)
        payload = json.loads(stdout.splitlines()[-1])
        return {"pid": int(payload["pid"]), "peak_rss_bytes": peak_rss_bytes}

    small_peak = replay_peak(v2)
    large_peak = replay_peak(memory_fixture)
    assert small_peak["pid"] != large_peak["pid"]
    assert large_peak["peak_rss_bytes"] - small_peak["peak_rss_bytes"] < 16 * 1024 * 1024


def test_native_distance_matrix_matches_worked_euclidean_fixture() -> None:
    points = np.asarray(((0.0, 0.0), (3.0, 4.0), (6.0, 8.0)), dtype=np.float64)
    observed = distance_matrix(points)
    assert observed.tolist() == [
        [0.0, 5.0, 10.0],
        [5.0, 0.0, 5.0],
        [10.0, 5.0, 0.0],
    ]


def test_storage_replay_aborts_executor_when_spawn_submit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = SimpleNamespace()
    aborted: list[object] = []

    monkeypatch.setattr(stage052_review, "ProcessPoolExecutor", lambda **_: executor)
    executor.submit = lambda *_: (_ for _ in ()).throw(RuntimeError("spawn failed"))
    monkeypatch.setattr(stage052_review, "abort_process_executor", aborted.append)

    with pytest.raises(RuntimeError, match="spawn failed"):
        replay_stage052_storage_semantics_many((tmp_path,))

    assert aborted == [executor]


def test_semantic_spool_lookup_and_order_use_primary_key_without_temp_sort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STAGE052_REVIEW_TMPDIR", str(tmp_path))
    with stage052_review._semantic_record_spool() as connection:
        connection.execute(
            "INSERT INTO semantic_records VALUES (?, ?, ?, ?, ?, ?)",
            ("c101_21", 2014, "fixed_work", 0, "left", b"payload"),
        )
        plan = connection.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT instance, seed, axis, ordinal, digest
            FROM semantic_records
            ORDER BY instance, seed, axis, ordinal
            """
        ).fetchall()

        details = " ".join(str(row[-1]) for row in plan)
        assert "TEMP B-TREE" not in details
        key = ("c101_21", 2014, "fixed_work", 0)
        assert list(stage052_review._semantic_spool_keys(connection)) == [(key, "left")]
        assert stage052_review._semantic_spool_payload(connection, key) == (
            "left",
            b"payload",
        )
        identity = key[:3]
        assert list(stage052_review._semantic_spool_tail_payloads(connection, {identity: 0})) == [
            (key, b"payload")
        ]
        assert list(stage052_review._semantic_spool_tail_payloads(connection, {identity: 1})) == []


def test_storage_replay_expands_compact_v2_screening_decisions(
    tmp_path: Path,
) -> None:
    def payloads() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
        raw = {
            "instance": "c101_21",
            "seed": 2014,
            "axes": {"fixed_work": {"started_calls": 0, "completed_calls": 0}},
        }
        solution = {
            "instance": "c101_21",
            "seed": 2014,
            "axes": {
                "fixed_work": {
                    "routes": [["C1"]],
                    "objective_key": [1, 10.0, 0.0, 0],
                }
            },
        }
        events = [
            {
                "event_type": "screening_decision",
                "record_type": "screening_decision",
                "benchmark_axis": "fixed_work",
                "route_key": "route:2:C1",
                "lane": "fixed_work:legacy",
                "operator": "relocate",
                "status": "pass",
                "reason": "",
                "decision_id": 1,
                "demand": 1.0,
                "distance_increment_lower_bound": None,
                "distance_lower_bound": 2.0,
                "exact_call_blocked": False,
                "first_failed_check": "",
                "min_time_window_slack": 3.0,
                "negative_cache_hit": False,
                "single_segment_reachable": True,
                "structural_energy_lower_bound": 4.0,
                "checks": [{"check": "capacity", "status": "pass", "value": True}],
            }
        ]
        return raw, solution, events

    v1 = tmp_path / "stage05.2_hot_path_attempt94"
    v1_writer = ArtifactBundleWriter(
        v1,
        ArtifactRunContext("stage05.2", "hot_path", v1.name),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v1"),
    )
    raw, solution, events = payloads()
    v1_writer.write_instance_seed(
        instance="c101_21",
        seed=2014,
        raw_payload=raw,
        solution_payload=solution,
        trace_payload={},
        environment_payload={},
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=events,
    )
    v1_writer.finalize()

    v2 = tmp_path / "stage05.2_artifact_streaming_attempt94"
    v2_writer = ArtifactBundleWriter(
        v2,
        ArtifactRunContext("stage05.2", "artifact_streaming", v2.name),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    raw, solution, events = payloads()
    shard = v2_writer.open_v2_shard(
        instance="c101_21",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    shard.append(
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=events,
    )
    shard.finalize(
        raw_payload=raw,
        solution_payload=solution,
        trace_payload={},
        environment_payload={},
    )
    v2_writer.finalize()

    v3 = tmp_path / "stage05.2_artifact_streaming_attempt93"
    v3_writer = ArtifactBundleWriter(
        v3,
        ArtifactRunContext("stage05.2", "artifact_streaming", v3.name),
        ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
    )
    raw, solution, events = payloads()
    v3_shard = v3_writer.open_v2_shard(
        instance="c101_21",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    v3_shard.append(
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=events,
    )
    v3_shard.finalize(
        raw_payload=raw,
        solution_payload=solution,
        trace_payload={},
        environment_payload={},
    )
    v3_writer.finalize()

    assert replay_stage052_storage_semantics(v1) == replay_stage052_storage_semantics(v2)
    assert replay_stage052_storage_semantics(v1) == replay_stage052_storage_semantics(v3)
    v1_trace = ArtifactReader(v1).reconstruct_trace(
        "c101_21/2014/stage05.2_hot_path_attempt94_trace_c101_21_2014.json"
    )
    v2_trace = ArtifactReader(v2).reconstruct_trace(
        "c101_21/2014/stage05.2_artifact_streaming_attempt94_trace_c101_21_2014.json"
    )
    assert v1_trace["screening_decisions"] == v2_trace["screening_decisions"]


def test_definition_encoded_screening_requires_known_untampered_definition() -> None:
    definition = {
        "benchmark_axis": "fixed_work",
        "checks": [
            {
                "check": "capacity",
                "reason": "",
                "status": "pass",
                "value_bool": True,
                "value_float": 1.0,
                "value_text": None,
            }
        ],
        "lane_id": 1,
        "operator_id": 2,
        "route_id": 3,
        "status": "pass",
    }
    definition_json = json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
    definition_id = (
        int.from_bytes(hashlib.sha256(definition_json).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
    )
    definitions: dict[int, dict[str, object]] = {}
    first = {
        "event_id": 1,
        "definition_id": definition_id,
        "definition_json": definition_json,
        "started_at": 1.25,
        "completed_at": 1.5,
        "iteration": 4,
        "decision_id": 9,
    }

    expanded = expand_v2_screening_decision(first, definitions=definitions)
    assert expanded["embedded_checks"][0]["value"] is True
    assert expanded["timestamp_seconds"] == 1.25
    assert expanded["duration_seconds"] == 0.25
    second = {**first, "event_id": 2, "definition_json": None}
    assert expand_v2_screening_decision(second, definitions=definitions)["route_id"] == 3

    with pytest.raises(ArtifactIntegrityError, match="unknown definition"):
        expand_v2_screening_decision({**second, "definition_id": definition_id + 1}, definitions={})
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        expand_v2_screening_decision({**first, "definition_id": definition_id + 1}, definitions={})


def test_v2_all_artifact_preparation_is_charged_without_gc_or_double_counting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flush_delay_seconds = 0.03
    postprocess_delay_seconds = 0.02
    live_append_delay_seconds = 0.025
    diagnostic_append_delay_seconds = 0.015

    class FakeShard:
        def append(self, **kwargs: object) -> int:
            critical_events = kwargs.get("critical_events")
            if isinstance(critical_events, tuple) and critical_events:
                time.sleep(live_append_delay_seconds)
            diagnostic_rows = kwargs.get("diagnostic_rows")
            if isinstance(diagnostic_rows, tuple) and diagnostic_rows:
                time.sleep(diagnostic_append_delay_seconds)
            return len(critical_events) if isinstance(critical_events, tuple) else 0

        def flush(self) -> None:
            time.sleep(flush_delay_seconds)

        def finalize(self, **_: object) -> None:
            return None

        def abort(self, _: BaseException) -> None:
            return None

    class FakeWriter:
        def open_v2_shard(self, **_: object) -> FakeShard:
            return FakeShard()

    def delayed_reconcile(_: object) -> dict[str, str]:
        time.sleep(postprocess_delay_seconds)
        return {"status": "pass"}

    trace = SimpleNamespace(
        route_dictionary={},
        reconcile=delayed_reconcile,
        to_index_dict=lambda: {},
    )
    objective = SimpleNamespace(
        key=(1, 1.0, 0.0, 0),
        vehicle_count=1,
        total_distance=1.0,
        total_charging_time=0.0,
        charging_count=0,
    )
    result = SimpleNamespace(
        measurement_trace=trace,
        objective=objective,
        routes=(("D0", "D0"),),
        feasible=True,
        exact_started_calls=1,
        exact_completed_calls=1,
        neighborhood_events=(),
        charging_backend="cpu_batch",
        backend_metrics={
            "work_batches": 1,
            "batch_launches": 1,
            "exact_calls": 1,
            "launch_occupancies": [1],
        },
        screening_statistics={},
        runtime_seconds=0.0,
        effective_iterations=1,
        unique_route_semantics="completed_cache_owner_identity_v2",
        termination_reason="fixed_work_budget",
    )

    def solve_with_live_event(*_args: object, **kwargs: object) -> object:
        sink = kwargs["trace_sink"]
        assert isinstance(sink, stage052_performance._Stage052TraceStreamSink)
        sink.append_event(
            {
                "event_type": "timing_diagnostic",
                "lane": "legacy",
                "iteration": 1,
                "operator": "test",
                "reason": "timing-only fixture",
            }
        )
        return result

    monkeypatch.setattr(stage052_performance, "_solve_stage052_axis", solve_with_live_event)
    monkeypatch.setattr(
        stage052_performance,
        "_iter_stage052_axis_events",
        lambda **kwargs: iter(()),
    )
    monkeypatch.setattr(
        stage052_performance,
        "validate_routes",
        lambda *args, **kwargs: SimpleNamespace(feasible=True),
    )
    monkeypatch.setattr(stage052_performance, "collect_environment", lambda: {})
    monkeypatch.setattr(stage052_performance, "_peak_rss_bytes", lambda: 1)

    axis = stage052_performance.Stage052Axis(
        name="fixed_work",
        termination_mode="fixed_work",
        time_limit_seconds=1.0,
        exact_call_budget=1,
    )
    task = _ShardTask(
        root=tmp_path,
        config_path=tmp_path / "unused.toml",
        run_dir=tmp_path / "results" / "stage05.2_artifact_streaming_attempt94",
        run_label="stage05.2_artifact_streaming_attempt94",
        component="artifact_streaming",
        scope="performance",
        instance_name="c101C5",
        customer_count=5,
        seed=2014,
        shard_ordinal=0,
        worker_count=1,
        storage=ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )

    rows = _run_and_persist_v2_shard(
        task,
        writer=FakeWriter(),  # type: ignore[arg-type]
        config=SimpleNamespace(),  # type: ignore[arg-type]
        stage04=SimpleNamespace(),
        stage02=SimpleNamespace(),
        instance=SimpleNamespace(),  # type: ignore[arg-type]
        axes=(axis,),
        storage=task.storage,
    )

    persistence = float(rows[0]["artifact_persistence_seconds"])
    solver_seconds = float(rows[0]["solver_seconds"])
    timing = rows[0]["_timing_evidence"]
    assert isinstance(timing, dict)
    recomputed_solver = (
        timing["solver_completed_ns"]
        - timing["solver_started_ns"]
        - timing["solver_interleaved_persistence_ns"]
    ) / 1_000_000_000
    recomputed_post_solver = (
        timing["axis_completed_ns"] - timing["solver_completed_ns"]
    ) / 1_000_000_000
    recomputed_finalize = (
        timing["finalize_completed_ns"] - timing["finalize_started_ns"]
    ) / 1_000_000_000
    assert timing["axis_started_ns"] == timing["solver_started_ns"]
    expected_charged_delays = (
        flush_delay_seconds
        + postprocess_delay_seconds
        + live_append_delay_seconds
        + diagnostic_append_delay_seconds
    )
    assert persistence >= expected_charged_delays * 0.9
    assert persistence < (expected_charged_delays * 2.0)
    assert solver_seconds == pytest.approx(recomputed_solver)
    assert persistence == pytest.approx(
        recomputed_finalize + timing["live_stream_persistence_ns"] / 1_000_000_000
    )
    assert recomputed_post_solver >= postprocess_delay_seconds * 0.9
    postsolve_persistence = (
        timing["live_stream_persistence_ns"] - timing["solver_interleaved_persistence_ns"]
    ) / 1_000_000_000
    assert postsolve_persistence == pytest.approx(
        timing["postsolve_artifact_preparation_ns"] / 1_000_000_000
    )
    assert timing["post_artifact_gc_ns"] == (
        timing["axis_completed_ns"] - timing["artifact_preparation_completed_ns"]
    )
    assert float(rows[0]["end_to_end_seconds"]) == pytest.approx(
        solver_seconds + persistence + recomputed_post_solver - postsolve_persistence
    )


def test_v2_pre_open_failure_publishes_partial_shard_evidence(
    tmp_path: Path,
) -> None:
    run_label = "stage05.2_artifact_streaming_attempt93"
    task = _ShardTask(
        root=tmp_path,
        config_path=tmp_path / "missing.toml",
        run_dir=tmp_path / "results" / run_label,
        run_label=run_label,
        component="artifact_streaming",
        scope="performance",
        instance_name="c101_21",
        customer_count=100,
        seed=2014,
        shard_ordinal=0,
        worker_count=1,
        storage=ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )

    with pytest.raises(FileNotFoundError):
        _run_v2_shard_task(task)

    prefix = task.run_dir / "c101_21" / "2014" / run_label
    failure = Path(f"{prefix}_failure_c101_21_2014.json")
    manifest = Path(f"{prefix}_shard_manifest_c101_21_2014.json")
    sidecar = Path(f"{prefix}_shard_manifest_c101_21_2014.sha256")
    assert failure.is_file()
    assert sidecar.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["evidence_completeness"] == "partial"


def test_v2_recovery_replaces_invalid_manifest_and_preserves_it(
    tmp_path: Path,
) -> None:
    run_label = "stage05.2_artifact_streaming_attempt92"
    task = _ShardTask(
        root=tmp_path,
        config_path=tmp_path / "missing.toml",
        run_dir=tmp_path / "results" / run_label,
        run_label=run_label,
        component="artifact_streaming",
        scope="performance",
        instance_name="c101_21",
        customer_count=100,
        seed=2014,
        shard_ordinal=0,
        worker_count=1,
        storage=ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    directory = task.run_dir / task.instance_name / str(task.seed)
    directory.mkdir(parents=True)
    manifest = directory / (f"{run_label}_shard_manifest_{task.instance_name}_{task.seed}.json")
    original_manifest = b'{"truncated":true}'
    manifest.write_bytes(original_manifest)
    manifest.with_suffix(".sha256").write_text("wrong\n", encoding="utf-8")

    _ensure_partial_shard_failure(task, "worker exited")

    recovered = json.loads(manifest.read_text(encoding="utf-8"))
    sidecar = manifest.with_suffix(".sha256")
    assert recovered["evidence_completeness"] == "partial"
    assert (
        sidecar.read_text(encoding="utf-8").strip()
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    archived = tuple(
        directory.glob(f"{run_label}_partial_fragment_manifest_{task.instance_name}_*.bin")
    )
    assert len(archived) == 1
    assert archived[0].read_bytes() == original_manifest
