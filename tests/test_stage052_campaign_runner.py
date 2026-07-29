from __future__ import annotations

import hashlib
import json
import os
import plistlib
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import evrptw.stage052_campaign_runner as campaign_runner
from evrptw.artifacts import ArtifactReader, ArtifactStorageConfig
from evrptw.candidate_transaction import NativeCandidateTransactionConfig
from evrptw.experiments import stage052_performance
from evrptw.experiments.stage052_performance import (
    _build_campaign_batch_tasks,
    _logical_event_row_count,
    _run_v2_tasks,
    load_stage052_config,
)
from evrptw.stage052_campaign import (
    BatchManifest,
    BatchPlan,
    BenchmarkCampaignConfig,
    BenchmarkPreflightObservation,
    CampaignManifest,
    PilotStorageObservation,
    ShardPlan,
    StorageRoot,
    StorageRootLocator,
    SystemLoadWindow,
    VolumeIdentity,
    directory_byte_count,
    directory_checksum,
    load_batch_manifest,
    load_campaign_manifest,
)
from evrptw.stage052_campaign_runner import (
    ArchivedBatchEvidence,
    ArchivedBatchStateWriteError,
    BatchRuntimeEvidence,
    BenchmarkExecutionLock,
    MachineSnapshot,
    RollingCampaignCapacityError,
    WindowsWslMachineSnapshotSource,
    campaign_configuration_selection_sha256,
    campaign_runtime_contract_sha256,
    campaign_runtime_selection_sha256,
    collect_preflight_observation,
    probe_volume_identity,
    validate_batch_measurements,
    verify_campaign_root_locations,
    verify_campaign_successor_revision,
    verify_rolling_campaign_capacity,
)
from evrptw.stage052_evidence import (
    PersistenceInterval,
    RunResourceSummary,
    Stage052PersistenceAttribution,
    validate_worker_ownership,
)
from evrptw.stage052_platform import WindowsWslPowerStatus
from evrptw.stage052_resources import ProducerResourceContract


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _record_stage052_worker_pid(_: object) -> list[dict[str, object]]:
    return [{"worker_pid": os.getpid()}]


def test_recycled_worker_ownership_allows_more_pids_than_concurrency() -> None:
    resource = {
        "schema_version": "stage05.2-run-resource-v2",
        "run_label": "stage05.2_benchmark_attempt99",
        "component": "benchmark",
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
            "run_label": "stage05.2_benchmark_attempt99",
            "evidence_completeness": "complete",
            "shard_ordinal": ordinal,
            "worker_identity": f"pid-{pid}",
        }
        for ordinal, pid in enumerate((201, 202, 301))
    ]

    passed, _, owners = validate_worker_ownership(
        resource,
        manifests,
        expected_workers=2,
        expected_run_label="stage05.2_benchmark_attempt99",
        expected_component="benchmark",
    )

    assert passed
    assert owners == (201, 202, 301)


def test_parallel_shards_recycle_worker_after_each_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    executor_options: list[dict[str, object]] = []

    class FakeFuture:
        def result(self) -> list[dict[str, object]]:
            return []

    class RecordingExecutor:
        def __init__(self, **options: object) -> None:
            self.ordinal = len(executor_options) + 1
            executor_options.append(options)
            events.append(f"create:{self.ordinal}")

        def submit(self, *_: object) -> FakeFuture:
            events.append(f"submit:{self.ordinal}")
            return FakeFuture()

        def shutdown(self, *, wait: bool) -> None:
            assert wait
            events.append(f"shutdown:{self.ordinal}")

    tasks = [
        SimpleNamespace(instance_name="c101C5", seed=2014),
        SimpleNamespace(instance_name="c101C5", seed=2015),
        SimpleNamespace(instance_name="c101C5", seed=2016),
    ]
    monkeypatch.setattr(stage052_performance, "ProcessPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(stage052_performance, "get_context", lambda _: object())
    monkeypatch.setattr(stage052_performance, "as_completed", lambda futures: iter(futures))

    assert _run_v2_tasks(tasks, worker_count=2) == []  # type: ignore[arg-type]
    assert events == [
        "create:1",
        "submit:1",
        "submit:1",
        "shutdown:1",
        "create:2",
        "submit:2",
        "shutdown:2",
    ]
    assert len(executor_options) == 2
    for options in executor_options:
        assert options["max_workers"] == 2
        assert options["max_tasks_per_child"] == 1
        assert (
            options["initializer"]
            is stage052_performance._warm_v2_worker_artifact_runtime
        )


def test_parallel_shards_abort_active_wave_on_resource_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeFuture:
        def cancel(self) -> bool:
            events.append("cancel")
            return True

    class RecordingExecutor:
        def __init__(self, **_options: object) -> None:
            events.append("create")

        def submit(self, *_args: object) -> FakeFuture:
            events.append("submit")
            return FakeFuture()

    tasks = [
        SimpleNamespace(instance_name="c201_21", seed=2018),
        SimpleNamespace(instance_name="c201_21", seed=2019),
    ]
    abort_checks = 0

    def abort_reason() -> str | None:
        nonlocal abort_checks
        abort_checks += 1
        if abort_checks == 1:
            return None
        return "aggregate RSS hard limit exceeded: observed=1001 limit=1000"

    monkeypatch.setattr(stage052_performance, "ProcessPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(stage052_performance, "get_context", lambda _: object())
    monkeypatch.setattr(
        stage052_performance,
        "abort_process_executor",
        lambda _executor: events.append("abort"),
    )
    monkeypatch.setattr(
        stage052_performance,
        "_ensure_partial_shard_failure",
        lambda task, _error: events.append(f"partial:{task.seed}"),
    )

    with pytest.raises(
        RuntimeError,
        match="runtime guard aborted Stage 5.2 work: aggregate RSS hard limit exceeded",
    ):
        _run_v2_tasks(  # type: ignore[arg-type]
            tasks,
            worker_count=2,
            abort_reason=abort_reason,
        )

    assert events == [
        "create",
        "submit",
        "submit",
        "cancel",
        "cancel",
        "abort",
        "partial:2018",
        "partial:2019",
    ]
    assert abort_checks == 2


def test_parallel_shards_use_a_fresh_spawned_pid_per_task() -> None:
    tasks = [
        SimpleNamespace(instance_name="c101C5", seed=seed)
        for seed in range(2014, 2020)
    ]

    rows = _run_v2_tasks(
        tasks,  # type: ignore[arg-type]
        worker_count=2,
        _task_runner=_record_stage052_worker_pid,  # type: ignore[arg-type]
    )

    worker_pids = [row["worker_pid"] for row in rows]
    assert len(worker_pids) == len(tasks)
    assert len(set(worker_pids)) == len(tasks)


def test_windows_wsl_snapshot_uses_windows_power_and_wsl_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_calls = 0

    def native_power() -> WindowsWslPowerStatus:
        nonlocal native_calls
        native_calls += 1
        return WindowsWslPowerStatus(True, False, 76, 8, "balanced-guid")

    monkeypatch.setattr(
        campaign_runner,
        "read_windows_wsl_power_status",
        native_power,
    )
    monkeypatch.setattr(campaign_runner, "read_wsl_ac_power_online", lambda: True)
    monkeypatch.setattr(campaign_runner.os, "getloadavg", lambda: (1.25, 0.0, 0.0))
    monkeypatch.setattr(campaign_runner, "_unrelated_user_cpu_seconds", lambda: {})

    source = WindowsWslMachineSnapshotSource()
    source.refresh_native_status()
    observed = source()
    source()

    assert observed.power_source == "AC Power"
    assert observed.low_power_mode_enabled is False
    assert observed.load1 == 1.25
    assert native_calls == 1


def test_windows_wsl_snapshot_fails_on_unknown_power_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        campaign_runner,
        "read_windows_wsl_power_status",
        lambda: (_ for _ in ()).throw(RuntimeError("AC power state mismatch")),
    )

    with pytest.raises(RuntimeError, match="AC power state mismatch"):
        WindowsWslMachineSnapshotSource()()


def test_native_power_boundary_allows_charge_progress_but_binds_invariants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = iter(
        (
            WindowsWslPowerStatus(True, False, 50, 8, "balanced-guid"),
            WindowsWslPowerStatus(True, False, 75, 1, "balanced-guid"),
        )
    )
    monkeypatch.setattr(
        campaign_runner,
        "read_windows_wsl_power_status",
        lambda: next(observations),
    )
    source = WindowsWslMachineSnapshotSource()
    source.refresh_native_status()

    evidence = source.verify_native_status_unchanged()

    assert evidence["invariants_unchanged"] is True
    assert evidence["before"]["battery_life_percent"] == 50
    assert evidence["after"]["battery_life_percent"] == 75


def test_unrelated_process_sampler_fails_on_unparseable_ps_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        campaign_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            (), 0, stdout="123 malformed-row\n", stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match="invalid non-empty ps process row"):
        campaign_runner._unrelated_user_cpu_seconds()


def test_volume_probe_resolves_a_directory_to_its_containing_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def run(
        arguments: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        calls.append(arguments)
        if arguments[0] == "df":
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=(
                    "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
                    "/dev/disk4s1 1 1 1 1% /Volumes/TRANSFER\n"
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=plistlib.dumps(
                {
                    "VolumeUUID": "transfer-uuid",
                    "FilesystemName": "ExFAT",
                }
            ),
            stderr=b"",
        )

    monkeypatch.setattr(campaign_runner.subprocess, "run", run)
    monkeypatch.setattr(campaign_runner.sys, "platform", "darwin")

    identity = probe_volume_identity(tmp_path)

    assert identity == VolumeIdentity("transfer-uuid", "ExFAT")
    assert calls == [
        ("df", "-P", str(tmp_path)),
        ("diskutil", "info", "-plist", "/dev/disk4s1"),
    ]


def test_batch_manifest_row_count_uses_only_logical_event_rows() -> None:
    rows = [
        {"_timing_evidence": {"axis_event_count": 7}},
        {"_timing_evidence": {"axis_event_count": 11}},
    ]

    assert _logical_event_row_count(rows) == 18


def _accepted_f02_payloads() -> tuple[dict[str, object], dict[str, object], str]:
    native = {
        "enabled": True,
        "exact_charging": True,
        "screening": True,
        "propagation": True,
        "distance_matrix": True,
        "abi_version": "stage05.2-native-kernels-v2",
        "context_policy": "pack_once_per_solve",
        "failure_policy": "fail_fast_no_fallback",
    }
    runtime = {
        "schema_version": "stage05.2-runtime-identity-v2",
        "repository_revision": "a" * 40,
        "wheel_filename": "runtime.whl",
        "wheel_sha256": "b" * 64,
        "python_version": "3.13.13",
        "python_executable_sha256": "c" * 64,
        "native_extension_sha256": "d" * 64,
        "dependency_versions": {"numpy": "2.4.6"},
        "dependency_manifest_sha256": "e" * 64,
        "installed_distribution_sha256": "6" * 64,
        "installed_editable": False,
    }
    inputs = {
        "instance_sha256": {"c101_21": "f" * 64},
        "stage02_config_sha256": "1" * 64,
        "stage04_config_sha256": "2" * 64,
    }
    raw_manifest_sha = "3" * 64
    metadata: dict[str, object] = {
        "run_label": "stage05.2_accelerator_pilot_attempt02",
        "component": "accelerator_pilot",
        "scope": "performance",
        "backend": "cpu_batch",
        "execution_backend": "native_cpu",
        "optimization_profile": "native",
        "worker_count": 2,
        "native_kernel_config": native,
        "candidate_transaction_config": (
            NativeCandidateTransactionConfig().to_dict()
        ),
        "repository_revision": "a" * 40,
        "runtime_identity": runtime,
        "configuration_sha256": "4" * 64,
        "performance_provenance": inputs,
        "storage_policy_version": "artifact-storage-v2",
        "screening_schema_version": "screening_decisions_v3",
    }
    review: dict[str, object] = {
        "run_label": metadata["run_label"],
        "component": metadata["component"],
        "scope": metadata["scope"],
        "status": "READY_FOR_STAGE052_BENCHMARK",
        "raw_manifest_sha256": raw_manifest_sha,
        "selected_backend": "native_cpu",
        "accelerator_decision": "GPU_NOT_JUSTIFIED",
        "selected_optimization_profile": "native",
        "selected_exact_backend": "cpu_batch",
        "selected_workers": 2,
        "native_configuration": native,
        "candidate_transaction_configuration": (
            NativeCandidateTransactionConfig().to_dict()
        ),
    }
    return metadata, review, raw_manifest_sha


def test_execution_lock_binds_every_selected_f02_input() -> None:
    metadata, review, raw_manifest_sha = _accepted_f02_payloads()

    lock = BenchmarkExecutionLock.from_accepted_evidence(
        metadata=metadata,
        review_manifest=review,
        raw_manifest_sha256=raw_manifest_sha,
        expected_scope="performance",
        expected_status="READY_FOR_STAGE052_BENCHMARK",
    )

    assert lock.selected_backend == "native_cpu"
    assert lock.selected_exact_backend == "cpu_batch"
    assert lock.selected_workers == 2
    assert lock.repository_revision == "a" * 40
    assert lock.runtime_identity_sha256 == _sha256_json(metadata["runtime_identity"])
    provenance = metadata["performance_provenance"]
    assert isinstance(provenance, dict)
    assert lock.input_provenance_sha256 == _sha256_json(
        {key: value for key, value in provenance.items() if key != "instance_sha256"}
    )
    assert lock.native_config_sha256 == _sha256_json(metadata["native_kernel_config"])
    assert lock.raw_manifest_sha256 == raw_manifest_sha
    assert lock.staging_root_alias is None
    assert lock.archive_root_aliases_exercised == ()


def test_execution_lock_binds_promoted_cuda_backend() -> None:
    metadata, review, raw_manifest_sha = _accepted_f02_payloads()
    metadata["execution_backend"] = "cuda"
    metadata["optimization_profile"] = "cuda"
    review["selected_backend"] = "cuda"
    review["accelerator_decision"] = "ACCELERATOR_PROMOTED"
    review["selected_optimization_profile"] = "cuda"

    lock = BenchmarkExecutionLock.from_accepted_evidence(
        metadata=metadata,
        review_manifest=review,
        raw_manifest_sha256=raw_manifest_sha,
        expected_scope="performance",
        expected_status="READY_FOR_STAGE052_BENCHMARK",
    )

    assert lock.selected_backend == "cuda"


def test_execution_lock_freezes_accepted_g01_storage_root_alias_set() -> None:
    metadata, review, raw_manifest_sha = _accepted_f02_payloads()
    metadata.update(
        {
            "run_label": "stage05.2_benchmark_attempt01",
            "component": "benchmark",
            "scope": "pilot",
        }
    )
    review.update(
        {
            "run_label": metadata["run_label"],
            "component": metadata["component"],
            "scope": metadata["scope"],
            "status": "READY_FOR_STAGE052_FORMAL_BENCHMARK",
            "staging_root_alias": "transfer_staging",
            "archive_root_aliases_exercised": [
                "transfer_archive",
                "internal_archive",
            ],
        }
    )

    lock = BenchmarkExecutionLock.from_accepted_evidence(
        metadata=metadata,
        review_manifest=review,
        raw_manifest_sha256=raw_manifest_sha,
        expected_scope="pilot",
        expected_status="READY_FOR_STAGE052_FORMAL_BENCHMARK",
    )

    assert lock.staging_root_alias == "transfer_staging"
    assert lock.archive_root_aliases_exercised == (
        "internal_archive",
        "transfer_archive",
    )
    lock.verify_planned_storage_roots(
        staging_root_alias="transfer_staging",
        planned_archive_root_aliases=("transfer_archive", "internal_archive"),
    )
    lock.verify_planned_storage_roots(
        staging_root_alias="transfer_staging",
        planned_archive_root_aliases=("internal_archive",),
    )


@pytest.mark.parametrize(
    ("staging_root_alias", "archive_root_aliases", "message"),
    (
        ("other_staging", ("transfer_archive", "internal_archive"), "staging"),
        (
            "transfer_staging",
            ("transfer_archive", "internal_archive", "unexpected_archive"),
            "drill coverage",
        ),
    ),
)
def test_execution_lock_rejects_formal_storage_root_drift_before_dispatch(
    staging_root_alias: str,
    archive_root_aliases: tuple[str, ...],
    message: str,
) -> None:
    metadata, review, raw_manifest_sha = _accepted_f02_payloads()
    metadata.update(
        {
            "run_label": "stage05.2_benchmark_attempt01",
            "component": "benchmark",
            "scope": "pilot",
        }
    )
    review.update(
        {
            "run_label": metadata["run_label"],
            "component": metadata["component"],
            "scope": metadata["scope"],
            "status": "READY_FOR_STAGE052_FORMAL_BENCHMARK",
            "staging_root_alias": "transfer_staging",
            "archive_root_aliases_exercised": [
                "transfer_archive",
                "internal_archive",
            ],
        }
    )
    lock = BenchmarkExecutionLock.from_accepted_evidence(
        metadata=metadata,
        review_manifest=review,
        raw_manifest_sha256=raw_manifest_sha,
        expected_scope="pilot",
        expected_status="READY_FOR_STAGE052_FORMAL_BENCHMARK",
    )
    batch_dispatches: list[str] = []

    with pytest.raises(RuntimeError, match=message):
        lock.verify_planned_storage_roots(
            staging_root_alias=staging_root_alias,
            planned_archive_root_aliases=archive_root_aliases,
        )
        batch_dispatches.append("batch0001")

    assert batch_dispatches == []


def test_execution_lock_rejects_g01_review_without_complete_root_aliases() -> None:
    metadata, review, raw_manifest_sha = _accepted_f02_payloads()
    metadata.update(
        {
            "run_label": "stage05.2_benchmark_attempt01",
            "component": "benchmark",
            "scope": "pilot",
        }
    )
    review.update(
        {
            "run_label": metadata["run_label"],
            "component": metadata["component"],
            "scope": metadata["scope"],
            "status": "READY_FOR_STAGE052_FORMAL_BENCHMARK",
            "staging_root_alias": "transfer_staging",
            "archive_root_aliases_exercised": ["transfer_staging"],
        }
    )

    with pytest.raises(RuntimeError, match="storage root aliases"):
        BenchmarkExecutionLock.from_accepted_evidence(
            metadata=metadata,
            review_manifest=review,
            raw_manifest_sha256=raw_manifest_sha,
            expected_scope="pilot",
            expected_status="READY_FOR_STAGE052_FORMAL_BENCHMARK",
        )


def test_execution_lock_rejects_backend_or_review_identity_drift() -> None:
    metadata, review, raw_manifest_sha = _accepted_f02_payloads()
    metadata["backend"] = "cpu_scalar"

    with pytest.raises(RuntimeError, match="cpu_batch"):
        BenchmarkExecutionLock.from_accepted_evidence(
            metadata=metadata,
            review_manifest=review,
            raw_manifest_sha256=raw_manifest_sha,
            expected_scope="performance",
            expected_status="READY_FOR_STAGE052_BENCHMARK",
        )
    metadata["backend"] = "cpu_batch"
    review["raw_manifest_sha256"] = "9" * 64
    with pytest.raises(RuntimeError, match="raw manifest"):
        BenchmarkExecutionLock.from_accepted_evidence(
            metadata=metadata,
            review_manifest=review,
            raw_manifest_sha256=raw_manifest_sha,
            expected_scope="performance",
            expected_status="READY_FOR_STAGE052_BENCHMARK",
        )
    review["raw_manifest_sha256"] = raw_manifest_sha
    review["selected_workers"] = 4
    with pytest.raises(RuntimeError, match="review worker"):
        BenchmarkExecutionLock.from_accepted_evidence(
            metadata=metadata,
            review_manifest=review,
            raw_manifest_sha256=raw_manifest_sha,
            expected_scope="performance",
            expected_status="READY_FOR_STAGE052_BENCHMARK",
        )


def test_execution_lock_rejects_current_runtime_or_worker_drift() -> None:
    metadata, review, raw_manifest_sha = _accepted_f02_payloads()
    lock = BenchmarkExecutionLock.from_accepted_evidence(
        metadata=metadata,
        review_manifest=review,
        raw_manifest_sha256=raw_manifest_sha,
        expected_scope="performance",
        expected_status="READY_FOR_STAGE052_BENCHMARK",
    )

    with pytest.raises(RuntimeError, match="worker"):
        lock.verify_current_execution(
            selected_backend="native_cpu",
            selected_exact_backend="cpu_batch",
            selected_workers=4,
            repository_revision="a" * 40,
            configuration_sha256="4" * 64,
            runtime_identity=metadata["runtime_identity"],
            input_provenance=metadata["performance_provenance"],
            native_kernel_config=metadata["native_kernel_config"],
        )
    runtime_payload = metadata["runtime_identity"]
    assert isinstance(runtime_payload, dict)
    drifted_runtime = dict(runtime_payload)
    drifted_runtime["wheel_sha256"] = "9" * 64
    with pytest.raises(RuntimeError, match="runtime"):
        lock.verify_current_execution(
            selected_backend="native_cpu",
            selected_exact_backend="cpu_batch",
            selected_workers=2,
            repository_revision="a" * 40,
            configuration_sha256="4" * 64,
            runtime_identity=drifted_runtime,
            input_provenance=metadata["performance_provenance"],
            native_kernel_config=metadata["native_kernel_config"],
        )


def test_execution_lock_ignores_only_same_revision_runtime_telemetry_and_paths() -> None:
    metadata, review, raw_manifest_sha = _accepted_f02_payloads()
    runtime = metadata["runtime_identity"]
    assert isinstance(runtime, dict)
    runtime.update(
        {
            "machine_identity": {
                "memory_bytes": 16 * 1024**3,
                "host_system": "Linux",
            },
            "source_repository_mount": {
                "filesystem": "ext4",
                "source": "/dev/sdd",
                "target": "/",
                "uuid": "old",
            },
            "native_extension": "/sealed/old/evrptw/_core.so",
            "python_executable": "/sealed/old/bin/python",
            "source_repository_root": "/sealed/old/producer-source",
            "wheel_path": "/sealed/old/wheels/runtime.whl",
            "machine_load_telemetry": {"temperature_c": 65.0},
        }
    )
    lock = BenchmarkExecutionLock.from_accepted_evidence(
        metadata=metadata,
        review_manifest=review,
        raw_manifest_sha256=raw_manifest_sha,
        expected_scope="performance",
        expected_status="READY_FOR_STAGE052_BENCHMARK",
    )
    current = {
        **runtime,
        "machine_identity": {
            "memory_bytes": 16 * 1024**3 - 4096,
            "host_system": "Linux",
        },
        "source_repository_mount": {
            "filesystem": "ext4",
            "source": "/dev/sde",
            "target": "/",
            "uuid": "new",
        },
        "native_extension": "/sealed/new/evrptw/_core.so",
        "python_executable": "/sealed/new/bin/python",
        "source_repository_root": "/sealed/new/producer-source",
        "wheel_path": "/sealed/new/wheels/runtime.whl",
        "machine_load_telemetry": {"temperature_c": 91.0},
    }

    assert campaign_runtime_contract_sha256(current) == (
        campaign_runtime_contract_sha256(runtime)
    )
    lock.verify_current_execution(
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        repository_revision="a" * 40,
        configuration_sha256="4" * 64,
        runtime_identity=current,
        input_provenance=metadata["performance_provenance"],
        native_kernel_config=metadata["native_kernel_config"],
    )

    for field in (
        "native_extension_sha256",
        "python_executable_sha256",
        "wheel_sha256",
    ):
        drifted = {**current, field: "9" * 64}
        with pytest.raises(RuntimeError, match="runtime contract"):
            lock.verify_current_execution(
                selected_backend="native_cpu",
                selected_exact_backend="cpu_batch",
                selected_workers=2,
                repository_revision="a" * 40,
                configuration_sha256="4" * 64,
                runtime_identity=drifted,
                input_provenance=metadata["performance_provenance"],
                native_kernel_config=metadata["native_kernel_config"],
            )
    incomplete = dict(current)
    del incomplete["python_version"]
    with pytest.raises(RuntimeError, match="hard contract is incomplete"):
        campaign_runtime_contract_sha256(incomplete)


def test_execution_lock_excludes_archive_device_telemetry_from_successor_identity(
    tmp_path: Path,
) -> None:
    metadata, review, raw_manifest_sha = _accepted_f02_payloads()
    source_disk = {
        "BusType": "NVMe",
        "FriendlyName": "original",
        "Number": 1,
        "SerialNumber": "original-serial",
    }
    base_runtime = dict(metadata["runtime_identity"])
    base_runtime["machine_identity"] = {
        "d_archive_disk": source_disk,
        "memory_bytes": 16 * 1024**3,
        "host_system": "Linux",
    }
    metadata["runtime_identity"] = base_runtime
    lock = BenchmarkExecutionLock.from_accepted_evidence(
        metadata=metadata,
        review_manifest=review,
        raw_manifest_sha256=raw_manifest_sha,
        expected_scope="performance",
        expected_status="READY_FOR_STAGE052_BENCHMARK",
    )
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.email", "test@example.com"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.name", "Stage 5.2 test"),
        check=True,
    )
    allowed = repository / "src" / "evrptw" / "stage052_campaign_runner.py"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("v1\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(repository), "add", "."), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-qm", "predecessor"),
        check=True,
    )
    predecessor_revision = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    allowed.write_text("v2\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(repository), "add", "."), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-qm", "successor"),
        check=True,
    )
    current_revision = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lock = replace(lock, repository_revision=predecessor_revision)

    runtime = dict(metadata["runtime_identity"])
    machine = dict(runtime["machine_identity"])
    destination_disk = {
        "BusType": "NVMe",
        "FriendlyName": "replacement",
        "Number": 1,
        "SerialNumber": "replacement-serial",
    }
    machine["d_archive_disk"] = destination_disk
    machine["memory_bytes"] = int(machine["memory_bytes"]) - 4096
    runtime["machine_identity"] = machine
    migration = {
        "source_machine_disk": source_disk,
        "destination_machine_disk": destination_disk,
    }

    lock.verify_current_execution(
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        repository_revision=current_revision,
        configuration_sha256="4" * 64,
        runtime_identity=runtime,
        input_provenance=metadata["performance_provenance"],
        native_kernel_config=metadata["native_kernel_config"],
        repository=repository,
        storage_migration=migration,
    )

    bad_migration = dict(migration)
    bad_migration["destination_machine_disk"] = source_disk
    lock.verify_current_execution(
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        repository_revision=current_revision,
        configuration_sha256="4" * 64,
        runtime_identity=runtime,
        input_provenance=metadata["performance_provenance"],
        native_kernel_config=metadata["native_kernel_config"],
        repository=repository,
        storage_migration=bad_migration,
    )

    drifted_runtime = dict(runtime)
    drifted_machine = dict(machine)
    drifted_machine["host_system"] = "different"
    drifted_runtime["machine_identity"] = drifted_machine
    lock.verify_current_execution(
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        repository_revision=current_revision,
        configuration_sha256="4" * 64,
        runtime_identity=drifted_runtime,
        input_provenance=metadata["performance_provenance"],
        native_kernel_config=metadata["native_kernel_config"],
        repository=repository,
        storage_migration=migration,
    )


def test_campaign_successor_revision_allows_only_g_governance_paths(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.email", "test@example.com"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.name", "Stage 5.2 test"),
        check=True,
    )
    allowed = repository / "src" / "evrptw" / "stage052_campaign_runner.py"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("v1\n", encoding="utf-8")
    allowed_config_test = repository / "tests" / "test_stage052_campaign.py"
    allowed_config_test.parent.mkdir(parents=True)
    allowed_config_test.write_text("v1\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(repository), "add", "."), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-qm", "base"),
        check=True,
    )
    predecessor = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    allowed.write_text("v2\n", encoding="utf-8")
    allowed_config_test.write_text("v2\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(repository), "commit", "-qam", "G fix"), check=True)
    successor = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert verify_campaign_successor_revision(
        repository,
        predecessor_revision=predecessor,
        current_revision=successor,
    ) == (
        "src/evrptw/stage052_campaign_runner.py",
        "tests/test_stage052_campaign.py",
    )

    forbidden = repository / "src" / "evrptw" / "objective.py"
    forbidden.write_text("changed\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(repository), "add", "."), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-qm", "forbidden"),
        check=True,
    )
    forbidden_successor = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(RuntimeError, match="non-G paths"):
        verify_campaign_successor_revision(
            repository,
            predecessor_revision=predecessor,
            current_revision=forbidden_successor,
        )


def test_campaign_successor_revision_accepts_only_exact_pinned_producer_fix(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "-C", str(repository), "init", "-q"), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.email", "test@example.com"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.name", "Stage 5.2 test"),
        check=True,
    )
    source_repository = Path(__file__).resolve().parents[1]
    pinned_paths = (
        "src/evrptw/alns.py",
        "tests/test_alns_wall_clock_only.py",
        "tests/test_artifacts_v3.py",
        "src/evrptw/candidate_transaction.py",
        "src/evrptw/measurement.py",
        "tests/test_candidate_transaction.py",
    )
    for index, relative in enumerate(pinned_paths):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            (source_repository / relative).read_bytes()
            if index < 2
            else b"pre-fix\n"
        )
    subprocess.run(("git", "-C", str(repository), "add", "."), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-qm", "base"),
        check=True,
    )
    predecessor = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    for relative in pinned_paths:
        (repository / relative).write_bytes((source_repository / relative).read_bytes())
    subprocess.run(("git", "-C", str(repository), "commit", "-qam", "producer fix"), check=True)
    successor = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert verify_campaign_successor_revision(
        repository,
        predecessor_revision=predecessor,
        current_revision=successor,
    ) == tuple(sorted(pinned_paths[2:]))

    with (repository / pinned_paths[0]).open("ab") as stream:
        stream.write(b"# unapproved change\n")
    subprocess.run(("git", "-C", str(repository), "commit", "-qam", "drift"), check=True)
    drifted = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(RuntimeError, match="pinned producer-fix content"):
        verify_campaign_successor_revision(
            repository,
            predecessor_revision=predecessor,
            current_revision=drifted,
        )


def test_campaign_runtime_selection_hash_excludes_only_g_wheel_identity() -> None:
    metadata, _, _ = _accepted_f02_payloads()
    runtime = metadata["runtime_identity"]
    assert isinstance(runtime, dict)
    runtime["dependency_versions"] = {
        "evrptw-reproduction": "0.1.0",
        "numpy": "2.4.6",
    }
    runtime["dependency_manifest_sha256"] = "4" * 64
    successor = {
        **runtime,
        "repository_revision": "9" * 40,
        "wheel_filename": "successor.whl",
        "wheel_sha256": "8" * 64,
        "installed_distribution_sha256": "7" * 64,
        "python_executable": "/sealed/runtime/bin/python3.13",
        "native_extension": "/sealed/runtime/site-packages/evrptw/_core.so",
        "source_repository_root": "/sealed/runtime/producer-source",
        "wheel_path": "/sealed/runtime/wheels/successor.whl",
    }

    assert campaign_runtime_selection_sha256(successor) == (
        campaign_runtime_selection_sha256(runtime)
    )
    successor["native_extension_sha256"] = "6" * 64
    successor["dependency_manifest_sha256"] = "5" * 64
    dependencies = successor["dependency_versions"]
    assert isinstance(dependencies, dict)
    successor["dependency_versions"] = {
        **dependencies,
        "reproducible-evrptw": "0.1.0",
    }
    assert campaign_runtime_selection_sha256(successor) == (
        campaign_runtime_selection_sha256(runtime)
    )
    successor_dependencies = successor["dependency_versions"]
    assert isinstance(successor_dependencies, dict)
    successor["dependency_versions"] = {
        **successor_dependencies,
        "numpy": "999.0",
    }
    assert campaign_runtime_selection_sha256(successor) != (
        campaign_runtime_selection_sha256(runtime)
    )


def test_campaign_configuration_selection_excludes_only_resource_tuning() -> None:
    accepted = b"""
[campaign]
storage_root_locator = "roots.toml"
staging_root_alias = "wsl_staging"

[artifact_storage_v2]
compression = "zstd"
"""
    calibrated = b"""
[campaign]
storage_root_locator = "roots.toml"
resource_calibration_contract = "resource.json"
staging_root_alias = "wsl_staging"

[artifact_storage_v2]
compression = "zstd"
parquet_row_group_size = 262144
parquet_queue_depth = 2
"""
    changed_science = calibrated.replace(b'compression = "zstd"', b'compression = "snappy"')

    assert campaign_configuration_selection_sha256(accepted) == (
        campaign_configuration_selection_sha256(calibrated)
    )
    assert campaign_configuration_selection_sha256(changed_science) != (
        campaign_configuration_selection_sha256(calibrated)
    )


def test_campaign_runtime_selection_hash_normalizes_attested_archive_disk() -> None:
    source_disk = {
        "BusType": "NVMe",
        "FriendlyName": "original",
        "Number": 1,
        "SerialNumber": "original-serial",
    }
    frozen_machine = {
        "memory_bytes": 16 * 1024**3,
        "cpu": {"Name": "same"},
        "d_archive_disk": source_disk,
    }
    frozen = {
        "schema_version": "stage05.2-runtime-identity-v2",
        "repository_revision": "1" * 40,
        "wheel_filename": "original.whl",
        "wheel_sha256": "2" * 64,
        "installed_distribution_sha256": "3" * 64,
        "native_extension_sha256": "4" * 64,
        "machine_identity": frozen_machine,
    }
    destination_disk = {
        "BusType": "NVMe",
        "FriendlyName": "replacement",
        "Number": 1,
        "SerialNumber": "replacement-serial",
    }
    live = {
        **frozen,
        "repository_revision": "9" * 40,
        "wheel_filename": "successor.whl",
        "wheel_sha256": "8" * 64,
        "installed_distribution_sha256": "7" * 64,
        "machine_identity": {
            **frozen_machine,
            "memory_bytes": int(frozen_machine["memory_bytes"]) - 4096,
            "d_archive_disk": destination_disk,
        },
    }
    migration = {
        "source_machine_disk": source_disk,
        "destination_machine_disk": destination_disk,
    }

    assert campaign_runtime_selection_sha256(frozen) == (
        campaign_runtime_selection_sha256(
            live,
            storage_migration=migration,
        )
    )


def _pilot_config() -> BenchmarkCampaignConfig:
    return BenchmarkCampaignConfig.pilot(
        run_label="stage05.2_benchmark_attempt01",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("transfer_archive", "internal_archive"),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v2",
    )


def test_stage052_config_routes_campaign_storage_through_ignored_locator() -> None:
    config = load_stage052_config(Path("configs/stage052_performance.toml"))

    assert config.storage_root_locator == Path("configs/stage052_storage_roots.local.toml")
    assert config.staging_root_alias == "wsl_staging"
    assert config.archive_root_aliases == ("d_archive",)


def test_campaign_root_location_drift_is_rejected_before_writes(tmp_path: Path) -> None:
    ext4 = VolumeIdentity("wsl-ext4", "ext4")
    d_drive = VolumeIdentity("d-nvme", "9p")
    locator = StorageRootLocator(
        {
            "wsl_staging": StorageRoot(
                "wsl_staging", Path("/mnt/d/wrong-staging"), ext4
            ),
            "d_archive": StorageRoot(
                "d_archive", Path("/mnt/d/FURP-2026-Yiyang-GUO-EVRP-TW-results"), d_drive
            ),
        }
    )

    with pytest.raises(RuntimeError, match="ext4"):
        verify_campaign_root_locations(repository_root=tmp_path, locator=locator)
    assert not Path("/mnt/d/wrong-staging").exists()


@pytest.mark.parametrize(
    ("archive_path", "filesystem"),
    ((Path("/mnt/e/stage052"), "9p"), (Path("/mnt/d/stage052"), "exfat")),
)
def test_campaign_root_preflight_rejects_forbidden_storage(
    tmp_path: Path,
    archive_path: Path,
    filesystem: str,
) -> None:
    locator = StorageRootLocator(
        {
            "wsl_staging": StorageRoot(
                "wsl_staging", tmp_path / "staging", VolumeIdentity("wsl", "ext4")
            ),
            "d_archive": StorageRoot(
                "d_archive", archive_path, VolumeIdentity("archive", filesystem)
            ),
        }
    )

    with pytest.raises(RuntimeError, match="forbidden|ExFAT"):
        verify_campaign_root_locations(repository_root=tmp_path, locator=locator)


def test_wsl_volume_probe_records_ext4_mount_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(arguments: tuple[str, ...], **_: object) -> SimpleNamespace:
        assert arguments[:2] == ("findmnt", "--json")
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "filesystems": [
                        {
                            "source": "/dev/sdd",
                            "fstype": "ext4",
                            "uuid": "wsl-ext4-uuid",
                        }
                    ]
                }
            )
        )

    monkeypatch.setattr(campaign_runner.subprocess, "run", fake_run)

    assert campaign_runner.probe_volume_identity(Path("/home/user/staging")) == (
        VolumeIdentity("wsl-ext4-uuid", "ext4")
    )


@pytest.mark.formal_environment
def test_wsl_volume_probe_records_d_nvme_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_run(arguments: tuple[str, ...], **_: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert arguments[:2] == ("findmnt", "--json")
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "filesystems": [
                            {"source": "D:\\", "fstype": "9p", "uuid": None}
                        ]
                    }
                )
            )
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "FriendlyName": "Samsung SSD 990 EVO Plus 1TB",
                    "SerialNumber": "0025_3854_5141_BD22.",
                    "BusType": "NVMe",
                }
            )
        )

    monkeypatch.setattr(campaign_runner.subprocess, "run", fake_run)

    assert campaign_runner.probe_volume_identity(Path("/mnt/d/archive")) == (
        VolumeIdentity("nvme-samsung-ssd-990-evo-plus-1tb-0025-3854-5141-bd22", "9p")
    )


def test_rolling_capacity_failure_carries_complete_observation(tmp_path: Path) -> None:
    config = _pilot_config()
    locator = _dispatcher_locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir(parents=True)
    plan = config.build_plan()
    capacity = config.plan_archive_roots(
        plan,
        locator,
        free_bytes_by_alias={alias: 200 * 1024**3 for alias in locator.aliases},
    )
    campaign = CampaignManifest.planned(
        config=config,
        plan=plan,
        capacity=capacity,
        locator=locator,
        configuration_sha256="a" * 64,
        prerequisite_review_sha256="b" * 64,
    )

    with pytest.raises(RollingCampaignCapacityError) as captured:
        verify_rolling_campaign_capacity(
            config=config,
            campaign=campaign,
            locator=locator,
            batch_id="batch0001",
            phase="pre_dispatch",
            free_space=lambda _path: 0,
            volume_probe=lambda path: next(
                root.volume
                for alias in locator.aliases
                for root in (locator.resolve(alias),)
                if root.absolute_path == path
            ),
        )

    observation = captured.value.observation
    assert observation["passed"] is False
    assert observation["phase"] == "pre_dispatch"
    assert observation["free_bytes_by_device"]
    assert observation["required_bytes_by_device"]
    assert observation["deficits_by_device"]


def test_campaign_task_ordinals_cover_exact_pilot_and_formal_bounds() -> None:
    pilot_plan = _pilot_config().build_plan()
    pilot_tasks = [
        task
        for batch in pilot_plan.batches
        for task in _build_campaign_batch_tasks(
            root=Path("/repo"),
            config_path=Path("/repo/config.toml"),
            batch_dir=Path("/staging") / batch.batch_id,
            run_label=pilot_plan.run_label,
            scope="pilot",
            worker_count=2,
            storage=ArtifactStorageConfig(
                storage_policy_version="artifact-storage-v2",
                screening_schema_version="screening_decisions_v3",
            ),
            shards=batch.shards,
        )
    ]
    assert len(pilot_tasks) == 36
    assert [task.shard_ordinal for task in pilot_tasks] == list(range(36))

    observations = tuple(
        PilotStorageObservation(family, count, 30, 1024)
        for family in ("C", "R", "RC")
        for count in (5, 10, 15, 100)
    )
    formal = BenchmarkCampaignConfig.formal(
        run_label="stage05.2_benchmark_attempt02",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("transfer_archive", "internal_archive"),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v2",
    ).build_plan(observations)
    formal_tasks = [
        task
        for batch in formal.batches
        for task in _build_campaign_batch_tasks(
            root=Path("/repo"),
            config_path=Path("/repo/config.toml"),
            batch_dir=Path("/staging") / batch.batch_id,
            run_label=formal.run_label,
            scope="formal",
            worker_count=2,
            storage=ArtifactStorageConfig(
                storage_policy_version="artifact-storage-v2",
                screening_schema_version="screening_decisions_v3",
            ),
            shards=batch.shards,
        )
    ]
    assert len(formal_tasks) == 920
    assert [task.shard_ordinal for task in formal_tasks] == list(range(920))
    assert sum(len(shard.budgets_seconds) for shard in formal.shards) == 2_040
    assert formal.checkpoint_count == 10_400


def test_preflight_collects_two_consecutive_exact_windows() -> None:
    now = 0.0

    def clock() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    def snapshot() -> MachineSnapshot:
        return MachineSnapshot("AC Power", False, 1.5, 0.25)

    observation = collect_preflight_observation(
        _pilot_config(),
        snapshot=snapshot,
        monotonic=clock,
        sleep=sleep,
        sample_interval_seconds=10.0,
    )

    assert len(observation.windows) == 2
    assert observation.windows[0].started_at_seconds == 0.0
    assert observation.windows[1].started_at_seconds == 30.0
    assert now == 60.0
    _pilot_config().validate_preflight(observation)


def test_batch_handoff_preflight_ignores_prior_campaign_load_history() -> None:
    now = 0.0

    def clock() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    observation = collect_preflight_observation(
        _pilot_config(),
        snapshot=lambda: MachineSnapshot("AC Power", False, 8.56982421875, 0.0),
        monotonic=clock,
        sleep=sleep,
        sample_interval_seconds=10.0,
        require_idle_load=False,
    )

    assert max(window.maximum_load1 for window in observation.windows) == 8.56982421875
    assert now == 60.0


def test_campaign_start_preflight_records_non_idle_load_as_telemetry() -> None:
    now = 0.0

    def clock() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    observation = collect_preflight_observation(
        _pilot_config(),
        snapshot=lambda: MachineSnapshot("Battery Power", True, 40.1, 6.0),
        monotonic=clock,
        sleep=sleep,
        sample_interval_seconds=10.0,
    )

    assert observation.power_source == "Battery Power"
    assert observation.low_power_mode_enabled is True
    assert max(window.maximum_load1 for window in observation.windows) == 40.1


def test_preflight_power_source_transition_is_nonblocking_telemetry() -> None:
    now = 0.0

    def clock() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    observation = collect_preflight_observation(
        _pilot_config(),
        snapshot=lambda: MachineSnapshot(
            "AC Power" if now < 30.0 else "Battery Power",
            now >= 30.0,
            40.0,
            8.0,
        ),
        monotonic=clock,
        sleep=sleep,
        sample_interval_seconds=10.0,
    )

    assert observation.power_source == "AC Power"
    assert observation.low_power_mode_enabled is True
    assert max(window.maximum_load1 for window in observation.windows) == 40.0


def test_preflight_windows_remain_consecutive_with_clock_call_overhead() -> None:
    now = 0.0

    def clock() -> float:
        nonlocal now
        now += 0.000001
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    observation = collect_preflight_observation(
        _pilot_config(),
        snapshot=lambda: MachineSnapshot("AC Power", False, 1.0, 0.1),
        monotonic=clock,
        sleep=sleep,
        sample_interval_seconds=10.0,
    )

    assert tuple(window.started_at_seconds for window in observation.windows) == (0.0, 30.0)
    _pilot_config().validate_preflight(observation)


def test_preflight_uses_full_window_cpu_time_delta_not_instantaneous_peak() -> None:
    now = 0.0

    def clock() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    def snapshot() -> MachineSnapshot:
        cpu_seconds = min(now, 1.0) * 2.0
        return MachineSnapshot(
            "AC Power",
            False,
            1.0,
            2.0 if now < 1.0 else 0.0,
            sampled_at_seconds=now,
            unrelated_process_cpu_seconds={999: cpu_seconds},
        )

    observation = collect_preflight_observation(
        _pilot_config(),
        snapshot=snapshot,
        monotonic=clock,
        sleep=sleep,
        sample_interval_seconds=1.0,
    )

    assert observation.windows[0].maximum_unrelated_process_average_cores == pytest.approx(
        2.0 / 30.0
    )
    assert observation.windows[1].maximum_unrelated_process_average_cores == 0.0


def test_preflight_allows_one_process_averaging_a_full_core_for_window() -> None:
    now = 0.0

    def clock() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    def snapshot() -> MachineSnapshot:
        return MachineSnapshot(
            "AC Power",
            False,
            1.0,
            1.0,
            sampled_at_seconds=now,
            unrelated_process_cpu_seconds={999: now},
        )

    observation = collect_preflight_observation(
        _pilot_config(),
        snapshot=snapshot,
        monotonic=clock,
        sleep=sleep,
        sample_interval_seconds=1.0,
    )

    assert observation.windows[0].maximum_unrelated_process_average_cores == 1.0


def test_preflight_allows_one_core_process_created_after_the_window_baseline() -> None:
    now = 0.0

    def clock() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    def snapshot() -> MachineSnapshot:
        counters = {} if now == 0.0 else {999: now + 0.1}
        return MachineSnapshot(
            "AC Power",
            False,
            1.0,
            0.0,
            sampled_at_seconds=now,
            unrelated_process_cpu_seconds=counters,
        )

    observation = collect_preflight_observation(
        _pilot_config(),
        snapshot=snapshot,
        monotonic=clock,
        sleep=sleep,
        sample_interval_seconds=1.0,
    )

    assert observation.windows[0].maximum_unrelated_process_average_cores == pytest.approx(
        30.1 / 30.0
    )


def test_preflight_conservatively_accounts_for_a_process_that_exits() -> None:
    snapshots = (
        MachineSnapshot(
            "AC Power",
            False,
            1.0,
            0.0,
            sampled_at_seconds=0.0,
            unrelated_process_cpu_seconds={},
        ),
        MachineSnapshot(
            "AC Power",
            False,
            1.0,
            0.0,
            sampled_at_seconds=29.0,
            unrelated_process_cpu_seconds={999: 28.0},
        ),
        MachineSnapshot(
            "AC Power",
            False,
            1.0,
            0.0,
            sampled_at_seconds=30.0,
            unrelated_process_cpu_seconds={},
        ),
    )

    evidence = BatchRuntimeEvidence.from_snapshots(
        snapshots,
        config=_pilot_config(),
        logical_cpu_count=24,
    )

    assert evidence.passed is True
    assert evidence.maximum_unrelated_process_average_cores == pytest.approx(52.0 / 30.0)


def test_runtime_evidence_records_power_or_load_drift_as_telemetry() -> None:
    evidence = BatchRuntimeEvidence.from_snapshots(
        (
            MachineSnapshot("AC Power", False, 1.0, 0.1),
            MachineSnapshot("Battery Power", False, 32.1, 1.0),
        ),
        config=_pilot_config(),
    )

    assert evidence.passed is True
    assert evidence.power_sources == ("AC Power", "Battery Power")
    assert evidence.maximum_load1 == 32.1


@pytest.mark.formal_environment
def test_runtime_evidence_allows_the_audited_24_thread_machine_headroom() -> None:
    config = BenchmarkCampaignConfig.pilot(
        run_label="stage05.2_benchmark_attempt01",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("transfer_archive", "internal_archive"),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=4,
        native_profile="stage05.2-native-kernels-v2",
    )

    evidence = BatchRuntimeEvidence.from_snapshots(
        (MachineSnapshot("AC Power", False, 31.9, 0.0),),
        config=config,
    )

    assert evidence.passed is True
    assert evidence.maximum_permitted_load1 == 32.0


@pytest.mark.formal_environment
def test_runtime_evidence_allows_attempt56_observed_campaign_load() -> None:
    config = BenchmarkCampaignConfig.pilot(
        run_label="stage05.2_benchmark_attempt01",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("transfer_archive", "internal_archive"),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=4,
        native_profile="stage05.2-native-kernels-v2",
    )

    evidence = BatchRuntimeEvidence.from_snapshots(
        (MachineSnapshot("AC Power", False, 21.60009765625, 0.0),),
        config=config,
    )

    assert evidence.passed is True


def test_runtime_evidence_records_load_beyond_old_machine_headroom() -> None:
    config = BenchmarkCampaignConfig.pilot(
        run_label="stage05.2_benchmark_attempt01",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("transfer_archive", "internal_archive"),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=4,
        native_profile="stage05.2-native-kernels-v2",
    )

    evidence = BatchRuntimeEvidence.from_snapshots(
        (MachineSnapshot("AC Power", False, 32.1, 0.0),),
        config=config,
    )

    assert evidence.passed is True
    assert evidence.maximum_permitted_load1 == 32.0
    assert evidence.failure_reason == ""


def test_runtime_evidence_accepts_any_logical_cpu_count_sufficient_for_workers() -> None:
    evidence = BatchRuntimeEvidence.from_snapshots(
        (MachineSnapshot("AC Power", False, 1.0, 0.0),),
        config=_pilot_config(),
        logical_cpu_count=16,
    )

    assert evidence.passed is True
    assert evidence.logical_cpu_count == 16
    assert evidence.failure_reason == ""


def test_runtime_evidence_rejects_logical_cpu_count_below_workers() -> None:
    evidence = BatchRuntimeEvidence.from_snapshots(
        (MachineSnapshot("AC Power", False, 1.0, 0.0),),
        config=_pilot_config(),
        logical_cpu_count=1,
    )

    assert evidence.passed is False
    assert "locked worker count" in evidence.failure_reason


@pytest.mark.formal_environment
def test_runtime_evidence_allows_full_window_unrelated_core_average() -> None:
    evidence = BatchRuntimeEvidence.from_snapshots(
        (
            MachineSnapshot(
                "AC Power",
                False,
                1.0,
                0.0,
                sampled_at_seconds=0.0,
                unrelated_process_cpu_seconds={999: 10.0},
            ),
            MachineSnapshot(
                "AC Power",
                False,
                1.0,
                0.0,
                sampled_at_seconds=30.0,
                unrelated_process_cpu_seconds={999: 40.0},
            ),
        ),
        config=_pilot_config(),
    )

    assert evidence.passed is True
    assert evidence.failure_reason == ""


def test_runtime_evidence_records_four_unrelated_cores_as_telemetry() -> None:
    evidence = BatchRuntimeEvidence.from_snapshots(
        (
            MachineSnapshot(
                "AC Power",
                False,
                1.0,
                0.0,
                sampled_at_seconds=0.0,
                unrelated_process_cpu_seconds={999: 0.0},
            ),
            MachineSnapshot(
                "AC Power",
                False,
                1.0,
                0.0,
                sampled_at_seconds=30.0,
                unrelated_process_cpu_seconds={999: 120.0},
            ),
        ),
        config=_pilot_config(),
    )

    assert evidence.passed is True
    assert evidence.maximum_unrelated_process_average_cores == 4.0


@pytest.mark.formal_environment
def test_runtime_monitor_waits_for_a_full_cpu_window_before_rejecting_exit() -> None:
    monitor = campaign_runner.BatchRuntimeMonitor(
        _pilot_config(),
        snapshot=lambda: MachineSnapshot("AC Power", False, 1.0, 0.0),
    )
    monitor._samples = [
        MachineSnapshot(
            "AC Power",
            False,
            1.0,
            0.0,
            sampled_at_seconds=0.0,
            unrelated_process_cpu_seconds={287: 0.0},
        ),
        MachineSnapshot(
            "AC Power",
            False,
            1.0,
            0.0,
            sampled_at_seconds=1.0,
            unrelated_process_cpu_seconds={287: 0.0},
        ),
        MachineSnapshot(
            "AC Power",
            False,
            1.0,
            0.0,
            sampled_at_seconds=2.1,
            unrelated_process_cpu_seconds={},
        ),
    ]

    assert monitor.abort_reason() is None


@pytest.mark.formal_environment
def test_runtime_monitor_allows_one_full_core_on_audited_24_thread_machine() -> None:
    monitor = campaign_runner.BatchRuntimeMonitor(
        _pilot_config(),
        snapshot=lambda: MachineSnapshot("AC Power", False, 1.0, 0.0),
    )
    monitor._samples = [
        MachineSnapshot(
            "AC Power",
            False,
            1.0,
            0.0,
            sampled_at_seconds=0.0,
            unrelated_process_cpu_seconds={999: 0.0},
        ),
        MachineSnapshot(
            "AC Power",
            False,
            1.0,
            0.0,
            sampled_at_seconds=30.0,
            unrelated_process_cpu_seconds={999: 30.0},
        ),
    ]

    assert monitor.abort_reason() is None


@pytest.mark.formal_environment
def test_runtime_monitor_does_not_abort_for_unrelated_cpu_telemetry() -> None:
    monitor = campaign_runner.BatchRuntimeMonitor(
        _pilot_config(),
        snapshot=lambda: MachineSnapshot("AC Power", False, 1.0, 0.0),
    )
    monitor._samples = [
        MachineSnapshot(
            "AC Power",
            False,
            1.0,
            0.0,
            sampled_at_seconds=0.0,
            unrelated_process_cpu_seconds={999: 0.0},
        ),
        MachineSnapshot(
            "AC Power",
            False,
            1.0,
            0.0,
            sampled_at_seconds=30.0,
            unrelated_process_cpu_seconds={999: 120.0},
        ),
    ]

    assert monitor.abort_reason() is None


def test_failed_batch_runtime_evidence_can_be_persisted(
    tmp_path: Path,
) -> None:
    records: list[tuple[Path, dict[str, object]]] = []
    writer = SimpleNamespace(
        record_existing_file=lambda path, **details: records.append((path, details))
    )
    (tmp_path / "control").mkdir()
    evidence = BatchRuntimeEvidence.from_snapshots(
        (
            MachineSnapshot(
                "AC Power",
                False,
                1.0,
                0.0,
                sampled_at_seconds=0.0,
                unrelated_process_cpu_seconds={999: 0.0},
            ),
            MachineSnapshot(
                "AC Power",
                False,
                1.0,
                0.0,
                sampled_at_seconds=30.0,
                unrelated_process_cpu_seconds={999: 120.0},
            ),
        ),
        config=_pilot_config(),
        logical_cpu_count=1,
    )

    path = stage052_performance._record_batch_runtime_evidence(
        writer=writer,
        batch_dir=tmp_path,
        run_label="stage05.2_benchmark_attempt34",
        batch_id="batch0001",
        runtime_evidence=evidence,
    )

    assert json.loads(path.read_text(encoding="utf-8"))["passed"] is False
    assert records == [
        (
            path,
            {
                "artifact_type": "batch_runtime_evidence",
                "artifact_subtype": "batch0001",
                "retention_class": "control",
                "storage_format": "json_control",
            },
        )
    ]


def _resource_summary(*, aggregate_rss: int = 2_000_000_000) -> RunResourceSummary:
    return RunResourceSummary(
        schema_version="stage05.2-run-resource-v3",
        run_label="stage05.2_benchmark_attempt01",
        component="benchmark",
        configured_worker_count=2,
        measurement_scope="process_tree",
        run_wall_seconds=30.0,
        sample_interval_seconds=0.05,
        parent_pid=100,
        descendant_pids=(101, 102),
        aggregate_peak_rss_bytes=aggregate_rss,
        process_peak_rss_bytes=((100, 100_000_000), (101, 500_000_000), (102, 600_000_000)),
        mean_active_cores=1.5,
        peak_active_cores=2.0,
        load1_min=1.0,
        load1_mean=1.2,
        load1_max=1.4,
        load1_sample_count=2,
        sample_count=2,
        status="complete",
    )


def _pilot_batch_rows() -> tuple[BatchPlan, list[dict[str, object]]]:
    batch = _pilot_config().build_plan().batches[0]
    rows = [
        {
            "instance": shard.instance,
            "seed": shard.seed,
            "axis": "wall_clock_30",
            "backend": "cpu_batch",
            "worker_count": 2,
            "solver_seconds": 10.0,
            "artifact_persistence_seconds": 1.0,
            "native_fallbacks": 0,
            "native_protocol_fallbacks": 0,
        }
        for shard in batch.shards
    ]
    return batch, rows


@pytest.mark.formal_environment
def test_batch_measurements_enforce_persistence_resource_and_runtime_gates() -> None:
    batch, rows = _pilot_batch_rows()
    runtime = BatchRuntimeEvidence.from_snapshots(
        (MachineSnapshot("AC Power", False, 1.0, 0.1),),
        config=_pilot_config(),
    )

    ratio = validate_batch_measurements(
        batch=batch,
        rows=rows,
        resource_summary=_resource_summary(),
        runtime_evidence=runtime,
        expected_workers=2,
    )
    assert ratio == pytest.approx(1.0 / 11.0)

    rows[0]["artifact_persistence_seconds"] = 100.0
    with pytest.raises(RuntimeError, match="persistence"):
        validate_batch_measurements(
            batch=batch,
            rows=rows,
            resource_summary=_resource_summary(),
            runtime_evidence=runtime,
            expected_workers=2,
        )


@pytest.mark.formal_environment
def test_benchmark_campaign_uses_relaxed_g_resource_limits() -> None:
    batch, rows = _pilot_batch_rows()
    runtime = BatchRuntimeEvidence.from_snapshots(
        (MachineSnapshot("AC Power", False, 1.0, 0.1),),
        config=_pilot_config(),
    )
    relaxed = replace(
        _resource_summary(aggregate_rss=19 * 1024**3),
        process_peak_rss_bytes=(
            (100, 100_000_000),
            (101, 7 * 1024**3),
            (102, 7 * 1024**3),
        ),
    )
    assert validate_batch_measurements(
        batch=batch,
        rows=rows,
        resource_summary=relaxed,
        runtime_evidence=runtime,
        expected_workers=2,
    ) == pytest.approx(1.0 / 11.0)

    with pytest.raises(RuntimeError, match="campaign lock"):
        validate_batch_measurements(
            batch=batch,
            rows=rows,
            resource_summary=replace(relaxed, aggregate_peak_rss_bytes=21 * 1024**3),
            runtime_evidence=runtime,
            expected_workers=2,
        )
    with pytest.raises(RuntimeError, match="per-worker"):
        validate_batch_measurements(
            batch=batch,
            rows=rows,
            resource_summary=replace(
                relaxed,
                process_peak_rss_bytes=(
                    (100, 100_000_000),
                    (101, 9 * 1024**3),
                    (102, 7 * 1024**3),
                ),
            ),
            runtime_evidence=runtime,
            expected_workers=2,
        )


def _dispatcher_locator(tmp_path: Path) -> StorageRootLocator:
    external = VolumeIdentity("external-device", "exfat")
    internal = VolumeIdentity("internal-device", "apfs")
    d_drive = VolumeIdentity("d-nvme-device", "9p")
    return StorageRootLocator(
        {
            "wsl_staging": StorageRoot("wsl_staging", tmp_path / "wsl-active", internal),
            "d_archive": StorageRoot("d_archive", tmp_path / "archive-d", d_drive),
            "transfer_staging": StorageRoot("transfer_staging", tmp_path / "results", external),
            "transfer_archive": StorageRoot(
                "transfer_archive", tmp_path / "archive-external", external
            ),
            "internal_archive": StorageRoot(
                "internal_archive", tmp_path / "archive-internal", internal
            ),
        }
    )


@pytest.mark.parametrize("archive_alias", ("transfer_archive", "internal_archive"))
def test_archive_state_write_failure_exposes_completed_transfer_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_alias: str,
) -> None:
    locator = _dispatcher_locator(tmp_path)
    for alias in locator.aliases:
        locator.resolve(alias).absolute_path.mkdir(parents=True)
    payload = b"verified archive payload"
    shard = ShardPlan(
        shard_id="shard0001",
        instance="c101C5",
        seed=2014,
        customer_count=5,
        family="C",
        budgets_seconds=(30,),
        checkpoint_seconds=(1, 5, 10, 30, 60, 120, 300),
        estimated_bytes=len(payload),
        max_iterations=1_000,
        scope="pilot",
    )
    batch = BatchManifest.planned(
        run_label="stage05.2_benchmark_attempt01",
        plan=BatchPlan("batch0001", (shard,), len(payload)),
        staging_root=locator.resolve("transfer_staging"),
        archive_root_alias=archive_alias,
    )
    source = locator.resolve("transfer_staging").absolute_path.joinpath(
        *Path(batch.logical_path).parts
    )
    source.mkdir(parents=True)
    (source / "payload.bin").write_bytes(payload)
    batch = batch.mark_verified(
        checksum_sha256=directory_checksum(source),
        actual_bytes=directory_byte_count(source),
        row_count=1,
        physical_schema="screening_decisions_v3",
        resource_summary_sha256="1" * 64,
        persistence_attribution_sha256="2" * 64,
        control_persistence_seconds=0.0,
        persistence_ratio=0.0,
        shard_manifest_sha256_by_id={"shard0001": "3" * 64},
        shard_actual_bytes_by_id={"shard0001": len(payload)},
    )
    monkeypatch.setattr(
        campaign_runner,
        "atomic_write_signed_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected seal failure")),
    )

    with pytest.raises(ArchivedBatchStateWriteError, match="final state write") as caught:
        campaign_runner.archive_verified_batch_with_evidence(batch=batch, locator=locator)

    destination = locator.resolve(archive_alias).absolute_path.joinpath(
        *Path(batch.logical_path).parts
    )
    assert caught.value.batch.status == "archived"
    assert caught.value.destination == destination
    assert not source.exists()
    assert destination.is_dir()
    assert directory_checksum(destination) == batch.checksum_sha256


def _valid_preflight() -> BenchmarkPreflightObservation:
    return BenchmarkPreflightObservation(
        power_source="AC Power",
        low_power_mode_enabled=False,
        windows=(
            SystemLoadWindow(0.0, 30.0, 1.0, 0.1),
            SystemLoadWindow(30.0, 30.0, 1.0, 0.1),
        ),
    )


def _patch_dispatcher_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    locator: StorageRootLocator,
) -> list[str]:
    producer_contract = ProducerResourceContract(
        selected_workers=4,
        available_memory_bytes=32 * 1024**3,
        selected_aggregate_peak_rss_bytes=8 * 1024**3,
        selected_per_worker_peak_rss_bytes=3 * 1024**3,
        aggregate_memory_limit_bytes=10 * 1024**3,
        per_worker_memory_limit_bytes=4 * 1024**3,
        semantic_digest="9" * 64,
        calibration_digest="8" * 64,
        row_group_size=262_144,
        queue_depth=2,
    )
    lock = SimpleNamespace(
        selected_workers=4,
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        verify_current_execution=lambda **_kwargs: None,
        to_dict=lambda: {
            "selected_backend": "native_cpu",
            "selected_exact_backend": "cpu_batch",
            "selected_workers": 4,
        },
        with_producer_resource_contract=lambda contract: {
            "selected_backend": "native_cpu",
            "selected_exact_backend": "cpu_batch",
            "selected_workers": contract.selected_workers,
            "producer_resource_contract": contract.to_dict(),
        },
    )
    monkeypatch.setattr(
        stage052_performance,
        "load_benchmark_execution_lock",
        lambda *_args, **_kwargs: lock,
    )
    monkeypatch.setattr(
        stage052_performance,
        "load_producer_resource_contract",
        lambda _path: producer_contract,
    )
    monkeypatch.setattr(stage052_performance, "_git", lambda *_args: "a" * 40)
    monkeypatch.setattr(
        stage052_performance,
        "verify_stage052_source_snapshot",
        lambda _root: {
            "repository_revision": "a" * 40,
            "mount": {"filesystem": "ext4", "uuid": "test-uuid"},
            "tracked_file_count": 100,
            "read_only": True,
        },
    )
    monkeypatch.setattr(
        stage052_performance,
        "verify_stage052_runtime_identity",
        lambda *_args, **_kwargs: {"runtime_identity_sha256": "b" * 64},
    )
    monkeypatch.setattr(stage052_performance, "collect_environment", lambda: {})
    monkeypatch.setattr(
        stage052_performance,
        "WindowsWslMachineSnapshotSource",
        lambda: SimpleNamespace(
            refresh_native_status=lambda: None,
            verify_native_status_unchanged=lambda: None,
            __call__=lambda: MachineSnapshot("AC Power", False, 0.0, 0.0),
        ),
    )
    monkeypatch.setattr(
        stage052_performance,
        "collect_performance_provenance",
        lambda **_kwargs: {"stage02_config_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        stage052_performance.StorageRootLocator,
        "from_toml",
        lambda _path: locator,
    )
    monkeypatch.setattr(
        stage052_performance,
        "verify_campaign_root_locations",
        lambda **_kwargs: None,
    )
    roots_by_path = {
        root.absolute_path: root.volume
        for alias in locator.aliases
        for root in (locator.resolve(alias),)
    }
    monkeypatch.setattr(
        stage052_performance,
        "probe_volume_identity",
        lambda path: roots_by_path[path],
    )
    monkeypatch.setattr(stage052_performance, "free_bytes", lambda _path: 1 << 50)
    preflight_calls: list[str] = []

    def collect_preflight(*_args: object, **_kwargs: object) -> BenchmarkPreflightObservation:
        preflight_calls.append("preflight")
        return _valid_preflight()

    monkeypatch.setattr(stage052_performance, "collect_preflight_observation", collect_preflight)

    def dry_run(path: Path, payload: dict[str, object]) -> tuple[Path, Path]:
        return stage052_performance.atomic_write_signed_json(path, payload)

    monkeypatch.setattr(
        stage052_performance,
        "_exercise_archive_roots",
        lambda **kwargs: dry_run(
            kwargs["output_dir"] / "control" / "archive_dry_run.json",
            {"passed": True},
        ),
    )
    monkeypatch.setattr(
        stage052_performance,
        "_exercise_publication_transaction",
        lambda output_dir, _run_label, **_kwargs: dry_run(
            output_dir / "control" / "publication_dry_run.json",
            {"passed": True},
        ),
    )
    monkeypatch.setattr(
        stage052_performance,
        "_exercise_raw_replay_drill",
        lambda **kwargs: dry_run(
            kwargs["output_dir"] / "control" / "raw_replay_drill.json",
            {"passed": True},
        ),
    )
    return preflight_calls


def _fake_batch_result(
    plan: Any,
    planned_manifest: Any,
) -> Any:
    rows = [
        {
            "customer_count": shard.customer_count,
            "instance": shard.instance,
            "seed": shard.seed,
            "axis": "wall_clock_30",
        }
        for shard in plan.shards
    ]
    checkpoints = [
        {
            "customer_count": shard.customer_count,
            "instance": shard.instance,
            "seed": shard.seed,
            "axis": "wall_clock_30",
            "axis_budget_seconds": 30,
            "checkpoint_seconds": checkpoint,
        }
        for shard in plan.shards
        for checkpoint in (1, 5, 10, 30)
    ]
    shard_hashes = {shard.shard_id: "d" * 64 for shard in plan.shards}
    shard_bytes = {shard.shard_id: 1 for shard in plan.shards}
    verified = planned_manifest.mark_verified(
        checksum_sha256="e" * 64,
        actual_bytes=max(1, len(plan.shards)),
        row_count=len(rows),
        physical_schema="screening_decisions_v3",
        resource_summary_sha256="f" * 64,
        persistence_attribution_sha256="a" * 64,
        control_persistence_seconds=0.1,
        persistence_ratio=0.1,
        shard_manifest_sha256_by_id=shard_hashes,
        shard_actual_bytes_by_id=shard_bytes,
    )
    attribution = Stage052PersistenceAttribution(
        run_label=verified.run_label,
        component="benchmark",
        scope="pilot",
        subject_id=verified.batch_id,
        primary_manifest_relative_path="manifest.json",
        primary_manifest_sha256="b" * 64,
        solver_seconds=10.0,
        shard_persistence_seconds=0.1,
        control_intervals=(),
    )
    return stage052_performance._VerifiedBatchExecution(
        rows=rows,
        checkpoints=checkpoints,
        batch=verified,
        base_attribution=attribution,
        verified_manifest_sha256="c" * 64,
        verified_manifest_write_interval=PersistenceInterval(
            "verified_batch_manifest_write", 1, 2
        ),
    )


def _run_patched_pilot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_batch_id: str | None = None,
    fail_archive_state_write: bool = False,
    fail_rolling_summary_once: bool = False,
    fail_failure_drill_summary: bool = False,
    fail_capacity_deficit: bool = False,
    fail_campaign_attribution: bool = False,
    attribution_precommit_status: list[str] | None = None,
) -> tuple[Path, list[str], list[str]]:
    locator = _dispatcher_locator(tmp_path)
    preflight_calls = _patch_dispatcher_dependencies(monkeypatch, locator=locator)
    config_path = Path("configs/stage052_performance.toml").resolve()
    config = load_stage052_config(config_path)
    prerequisite_dir = tmp_path / "stage05.2_accelerator_pilot_attempt02"
    review_dir = prerequisite_dir / "review"
    review_dir.mkdir(parents=True)
    (review_dir / "review_manifest.json").write_text(
        json.dumps({"status": "READY_FOR_STAGE052_BENCHMARK"}) + "\n",
        encoding="utf-8",
    )
    batch_calls: list[str] = []

    def run_batch(**kwargs: Any) -> Any:
        plan = kwargs["plan"]
        batch_calls.append(plan.batch_id)
        if plan.batch_id == fail_batch_id:
            raise RuntimeError("injected batch failure")
        return _fake_batch_result(plan, kwargs["planned_manifest"])

    def archive_batch(
        *,
        batch: Any,
        locator: StorageRootLocator,
        _state_writer: Any = None,
    ) -> ArchivedBatchEvidence:
        if _state_writer is not None:
            return campaign_runner.archive_verified_batch_with_evidence(
                batch=batch,
                locator=locator,
                _state_writer=_state_writer,
            )
        destination = locator.resolve(batch.archive_root_alias)
        archived = batch.mark_archived(
            root_alias=batch.archive_root_alias,
            volume=destination.volume,
            transfer_mode="same_volume_atomic_rename",
            archive_transfer_seconds=0.0,
        )
        batch_dir = destination.absolute_path.joinpath(*Path(batch.logical_path).parts)
        batch_dir.mkdir(parents=True, exist_ok=True)
        if fail_archive_state_write:
            raise ArchivedBatchStateWriteError(
                batch=archived,
                destination=batch_dir,
                cause=OSError("injected archived manifest failure"),
            )
        manifest_path, _ = stage052_performance.atomic_write_signed_json(
            batch_dir / "batch_manifest.json",
            archived.to_dict(),
        )
        return ArchivedBatchEvidence(
            batch=archived,
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            state_write_interval=PersistenceInterval(
                "archived_batch_manifest_write", 3, 4
            ),
            manifest_path=manifest_path,
        )

    monkeypatch.setattr(stage052_performance, "_run_benchmark_batch", run_batch)
    monkeypatch.setattr(
        stage052_performance,
        "archive_verified_batch_with_evidence",
        archive_batch,
    )
    if fail_capacity_deficit:

        def reject_capacity(**kwargs: Any) -> dict[str, object]:
            batch_id = str(kwargs["batch_id"])
            phase = str(kwargs["phase"])
            observation = {
                "schema_version": "stage05.2-rolling-capacity-v1",
                "run_label": "stage05.2_benchmark_attempt01",
                "batch_id": batch_id,
                "phase": phase,
                "free_bytes_by_device": {"transfer-volume": 1},
                "required_bytes_by_device": {"transfer-volume": 2},
                "deficits_by_device": {"transfer-volume": 1},
                "passed": False,
            }
            raise RollingCampaignCapacityError(
                observation=observation,
                message="injected rolling capacity deficit",
            )

        monkeypatch.setattr(
            stage052_performance,
            "verify_rolling_campaign_capacity",
            reject_capacity,
        )
    output_dir = tmp_path / "wsl-active" / "stage05.2_benchmark_attempt01"
    if fail_rolling_summary_once:
        real_atomic_write = stage052_performance.atomic_write_signed_json
        injected = False

        def fail_one_summary(
            path: Path,
            payload: dict[str, object],
            **kwargs: Any,
        ) -> tuple[Path, Path]:
            nonlocal injected
            if (
                not injected
                and path.name.endswith("_rolling_capacity.json")
                and payload.get("status") == "in_progress"
            ):
                injected = True
                raise OSError("injected rolling summary interruption")
            return real_atomic_write(path, payload, **kwargs)

        monkeypatch.setattr(
            stage052_performance,
            "atomic_write_signed_json",
            fail_one_summary,
        )
    elif fail_failure_drill_summary:
        real_atomic_write = stage052_performance.atomic_write_signed_json
        injected = False

        def fail_one_drill(
            path: Path,
            payload: dict[str, object],
            **kwargs: Any,
        ) -> tuple[Path, Path]:
            nonlocal injected
            if not injected and path.name.endswith("_failure_state_drill.json"):
                injected = True
                raise OSError("injected failure drill summary interruption")
            return real_atomic_write(path, payload, **kwargs)

        monkeypatch.setattr(
            stage052_performance,
            "atomic_write_signed_json",
            fail_one_drill,
        )
    elif fail_campaign_attribution:
        real_atomic_write = stage052_performance.atomic_write_signed_json

        def fail_attribution(
            path: Path,
            payload: dict[str, object],
            **kwargs: Any,
        ) -> tuple[Path, Path]:
            if path.name == "stage05.2_benchmark_attempt01_persistence_attribution.json":
                if attribution_precommit_status is not None:
                    attribution_precommit_status.append(
                        load_campaign_manifest(
                            path.parent.parent / "campaign_manifest.json"
                        ).status
                    )
                raise OSError("injected campaign attribution failure")
            return real_atomic_write(path, payload, **kwargs)

        monkeypatch.setattr(
            stage052_performance,
            "atomic_write_signed_json",
            fail_attribution,
        )

    def invoke() -> dict[str, Path]:
        return stage052_performance._run_benchmark_campaign(
            root=tmp_path,
            config_path=config_path,
            output_dir=output_dir,
            run_label="stage05.2_benchmark_attempt01",
            scope="pilot",
            worker_count=4,
            config=config,
            stage051_prerequisite={"status": "READY_FOR_STAGE05_2"},
            component_prerequisites={"accelerator_selection": {"status": "accepted"}},
            resolved_prerequisite_dirs={"accelerator_selection": prerequisite_dir},
            storage=config.v2_storage,
        )

    if (
        fail_batch_id is None
        and not fail_archive_state_write
        and not fail_rolling_summary_once
        and not fail_failure_drill_summary
        and not fail_capacity_deficit
        and not fail_campaign_attribution
    ):
        invoke()
    else:
        expected = "injected batch failure"
        error_type: type[BaseException] = RuntimeError
        if fail_archive_state_write:
            expected = "final state write failed"
        elif fail_rolling_summary_once:
            expected = "rolling summary interruption"
            error_type = OSError
        elif fail_failure_drill_summary:
            expected = "failure drill summary interruption"
            error_type = OSError
        elif fail_capacity_deficit:
            expected = "rolling capacity deficit"
            error_type = RollingCampaignCapacityError
        elif fail_campaign_attribution:
            expected = "campaign attribution failure"
            error_type = OSError
        with pytest.raises(error_type, match=expected):
            invoke()
    return output_dir, preflight_calls, batch_calls


def test_g01_dispatcher_archives_all_batches_and_completes_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, preflight_calls, batch_calls = _run_patched_pilot(tmp_path, monkeypatch)

    manifest_path = output_dir / "campaign_manifest.json"
    manifest = load_campaign_manifest(manifest_path)
    assert manifest_path.is_file()
    assert manifest_path.with_suffix(".sha256").is_file()
    assert manifest.status == "complete"
    assert manifest.shard_count == 36
    assert manifest.checkpoint_count == 144
    assert len(manifest.batches) > 1
    assert all(batch.status == "archived" for batch in manifest.batches)
    standard = ArtifactReader(output_dir)
    assert standard.manifest["status"] == "complete"
    assert not any(
        item["artifact_type"] == "campaign_manifest"
        for item in standard.manifest["artifacts"]
    )
    assert batch_calls == [batch.batch_id for batch in manifest.batches]
    assert preflight_calls == ["preflight"]
    aggregate_path = (
        output_dir / "control" / "stage05.2_benchmark_attempt01_anytime_checkpoints.json"
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["row_count"] == 144
    failure_summary = json.loads(
        (
            output_dir
            / "control/stage05.2_benchmark_attempt01_failure_state_drill.json"
        ).read_text(encoding="utf-8")
    )
    failed_batch = load_batch_manifest(
        output_dir / str(failure_summary["failed_batch_relative_path"])
    )
    failed_campaign = load_campaign_manifest(
        output_dir / str(failure_summary["failed_campaign_relative_path"])
    )
    assert failed_batch.status == "failed"
    assert failed_campaign.status == "failed"
    worker_manifest_path = output_dir / str(
        failure_summary["worker_manifest_relative_path"]
    )
    worker_manifest = json.loads(worker_manifest_path.read_text(encoding="utf-8"))
    assert worker_manifest["evidence_completeness"] == "partial"
    assert worker_manifest["worker_identity"] == "failure-recorder"
    assert worker_manifest_path.with_suffix(".sha256").read_text(
        encoding="utf-8"
    ).strip() == hashlib.sha256(worker_manifest_path.read_bytes()).hexdigest()
    archive_manifest_path = output_dir / str(
        failure_summary["archive_manifest_relative_path"]
    )
    archive_drill = load_batch_manifest(archive_manifest_path)
    assert archive_drill.status == "archived"
    assert archive_drill.transfer_mode == "same_volume_atomic_rename"
    assert directory_checksum(archive_manifest_path.parent) == archive_drill.checksum_sha256
    assert directory_byte_count(archive_manifest_path.parent) == archive_drill.actual_bytes


def test_g01_dispatcher_stops_after_failed_batch_and_preserves_partial_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, _preflight_calls, batch_calls = _run_patched_pilot(
        tmp_path,
        monkeypatch,
        fail_batch_id="batch0002",
    )

    manifest = load_campaign_manifest(output_dir / "campaign_manifest.json")
    assert manifest.status == "failed"
    assert manifest.batches[0].status == "archived"
    assert manifest.batches[1].status == "failed"
    assert all(batch.status == "planned" for batch in manifest.batches[2:])
    assert batch_calls == ["batch0001", "batch0002"]


def test_g01_campaign_records_archived_state_when_archive_seal_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, _preflight_calls, batch_calls = _run_patched_pilot(
        tmp_path,
        monkeypatch,
        fail_archive_state_write=True,
    )

    campaign = load_campaign_manifest(output_dir / "campaign_manifest.json")
    assert campaign.status == "failed"
    assert campaign.batches[0].status == "archived"
    assert all(batch.status == "planned" for batch in campaign.batches[1:])
    assert batch_calls == ["batch0001"]


def test_rolling_capacity_journal_survives_summary_write_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, _preflight_calls, batch_calls = _run_patched_pilot(
        tmp_path,
        monkeypatch,
        fail_rolling_summary_once=True,
    )

    journals = sorted(
        (output_dir / "control/rolling_capacity_observations").glob("*.json")
    )
    assert len(journals) == 1
    assert journals[0].with_suffix(".sha256").read_text(encoding="utf-8").strip() == (
        hashlib.sha256(journals[0].read_bytes()).hexdigest()
    )
    assert json.loads(journals[0].read_text(encoding="utf-8"))["phase"] == "pre_dispatch"
    assert batch_calls == []


def test_rolling_capacity_deficit_is_signed_before_campaign_aborts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, _preflight_calls, batch_calls = _run_patched_pilot(
        tmp_path,
        monkeypatch,
        fail_capacity_deficit=True,
    )

    journals = sorted(
        (output_dir / "control/rolling_capacity_observations").glob("*.json")
    )
    assert len(journals) == 1
    journal = json.loads(journals[0].read_text(encoding="utf-8"))
    assert journal["passed"] is False
    assert journal["deficits_by_device"] == {"transfer-volume": 1}
    assert journals[0].with_suffix(".sha256").read_text(encoding="utf-8").strip() == (
        hashlib.sha256(journals[0].read_bytes()).hexdigest()
    )
    summary = json.loads(
        (
            output_dir
            / "control/stage05.2_benchmark_attempt01_rolling_capacity.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["status"] == "failed"
    assert summary["passed"] is False
    assert summary["observations"] == [journal]
    assert load_campaign_manifest(output_dir / "campaign_manifest.json").status == "failed"
    assert batch_calls == []


def test_post_complete_attribution_failure_reseals_campaign_as_failed_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    precommit_status: list[str] = []
    output_dir, _preflight_calls, batch_calls = _run_patched_pilot(
        tmp_path,
        monkeypatch,
        fail_campaign_attribution=True,
        attribution_precommit_status=precommit_status,
    )

    campaign = load_campaign_manifest(output_dir / "campaign_manifest.json")
    reader = ArtifactReader(output_dir)
    assert campaign.status == "failed"
    assert "campaign attribution failure" in str(campaign.failure_reason)
    assert reader.manifest["status"] == "partial"
    assert reader.manifest["evidence_completeness"] == "partial"
    assert len(batch_calls) == len(campaign.batches)
    assert precommit_status == ["planned"]


def test_campaign_failure_sealer_retries_campaign_after_primary_is_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_label = "stage05.2_benchmark_attempt01"
    output_dir = tmp_path / run_label
    locator = _dispatcher_locator(tmp_path)
    config = _pilot_config()
    plan = config.build_plan()
    capacity = config.plan_archive_roots(
        plan,
        locator,
        free_bytes_by_alias={alias: 200 * 1024**3 for alias in locator.aliases},
    )
    campaign = CampaignManifest.planned(
        config=config,
        plan=plan,
        capacity=capacity,
        locator=locator,
        configuration_sha256="a" * 64,
        prerequisite_review_sha256="b" * 64,
    )
    real_persist = stage052_performance.persist_campaign_manifest
    real_persist(output_dir, campaign)
    calls = 0

    def fail_first_campaign_transition(*args: Any, **kwargs: Any) -> Path:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected campaign transition failure")
        return real_persist(*args, **kwargs)

    monkeypatch.setattr(
        stage052_performance,
        "persist_campaign_manifest",
        fail_first_campaign_transition,
    )
    storage = ArtifactStorageConfig(
        storage_policy_version="artifact-storage-v2",
        screening_schema_version="screening_decisions_v3",
    )
    error = RuntimeError("injected startup failure")

    stage052_performance._seal_campaign_startup_failure(
        output_dir=output_dir,
        run_label=run_label,
        storage=storage,
        error=error,
    )
    assert ArtifactReader(output_dir).manifest["status"] == "partial"
    assert load_campaign_manifest(output_dir / "campaign_manifest.json").status == "planned"

    stage052_performance._seal_campaign_startup_failure(
        output_dir=output_dir,
        run_label=run_label,
        storage=storage,
        error=error,
    )
    failed = load_campaign_manifest(output_dir / "campaign_manifest.json")
    assert failed.status == "failed"

    failed_sha = hashlib.sha256((output_dir / "campaign_manifest.json").read_bytes()).hexdigest()
    stage052_performance._seal_campaign_startup_failure(
        output_dir=output_dir,
        run_label=run_label,
        storage=storage,
        error=error,
    )
    assert hashlib.sha256((output_dir / "campaign_manifest.json").read_bytes()).hexdigest() == (
        failed_sha
    )


def test_campaign_startup_failure_is_sealed_as_partial_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir, _preflight_calls, batch_calls = _run_patched_pilot(
        tmp_path,
        monkeypatch,
        fail_failure_drill_summary=True,
    )

    reader = ArtifactReader(output_dir)
    campaign = load_campaign_manifest(output_dir / "campaign_manifest.json")
    assert reader.manifest["evidence_completeness"] == "partial"
    assert campaign.status == "failed"
    assert any(
        item["artifact_type"] == "startup_failure"
        for item in reader.manifest["artifacts"]
    )
    assert batch_calls == []


def test_nested_corrupt_manifest_cannot_suppress_top_level_failure_seal(
    tmp_path: Path,
) -> None:
    run_label = "stage05.2_benchmark_attempt01"
    output_dir = tmp_path / run_label
    nested = output_dir / "batch0001" / "control"
    nested.mkdir(parents=True)
    (nested / f"{run_label}_manifest.json").write_text("{broken", encoding="utf-8")

    stage052_performance._seal_campaign_startup_failure(
        output_dir=output_dir,
        run_label=run_label,
        storage=ArtifactStorageConfig(
            storage_policy_version="artifact-storage-v2",
            screening_schema_version="screening_decisions_v3",
        ),
        error=RuntimeError("injected startup failure"),
    )

    reader = ArtifactReader(output_dir)
    assert reader.result.manifest_path == output_dir / "control" / f"{run_label}_manifest.json"
    assert reader.manifest["status"] == "partial"
    assert any(
        item["artifact_type"] == "startup_failure"
        for item in reader.manifest["artifacts"]
    )
