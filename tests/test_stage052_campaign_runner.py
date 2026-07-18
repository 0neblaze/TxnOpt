from __future__ import annotations

import hashlib
import json
import plistlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import evrptw.stage052_campaign_runner as campaign_runner
from evrptw.artifacts import ArtifactReader, ArtifactStorageConfig
from evrptw.experiments import stage052_performance
from evrptw.experiments.stage052_performance import (
    _build_campaign_batch_tasks,
    _logical_event_row_count,
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
    collect_preflight_observation,
    probe_volume_identity,
    validate_batch_measurements,
    verify_campaign_root_locations,
    verify_rolling_campaign_capacity,
)
from evrptw.stage052_evidence import (
    PersistenceInterval,
    RunResourceSummary,
    Stage052PersistenceAttribution,
)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
        "abi_version": "stage05.2-native-kernels-v1",
        "context_policy": "pack_once_per_solve",
        "failure_policy": "fail_fast_no_fallback",
    }
    runtime = {
        "schema_version": "stage05.2-runtime-identity-v1",
        "repository_revision": "a" * 40,
        "python_sha256": "b" * 64,
        "wheel_sha256": "c" * 64,
        "native_extension_sha256": "d" * 64,
        "dependency_lock_sha256": "e" * 64,
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


def test_execution_lock_binds_promoted_metal_backend() -> None:
    metadata, review, raw_manifest_sha = _accepted_f02_payloads()
    metadata["execution_backend"] = "metal"
    metadata["optimization_profile"] = "metal"
    review["selected_backend"] = "metal"
    review["accelerator_decision"] = "ACCELERATOR_PROMOTED"
    review["selected_optimization_profile"] = "metal"

    lock = BenchmarkExecutionLock.from_accepted_evidence(
        metadata=metadata,
        review_manifest=review,
        raw_manifest_sha256=raw_manifest_sha,
        expected_scope="performance",
        expected_status="READY_FOR_STAGE052_BENCHMARK",
    )

    assert lock.selected_backend == "metal"


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


def _pilot_config() -> BenchmarkCampaignConfig:
    return BenchmarkCampaignConfig.pilot(
        run_label="stage05.2_benchmark_attempt01",
        staging_root_alias="transfer_staging",
        archive_root_aliases=("transfer_archive", "internal_archive"),
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        selected_workers=2,
        native_profile="stage05.2-native-kernels-v1",
    )


def test_stage052_config_routes_campaign_storage_through_ignored_locator() -> None:
    config = load_stage052_config(Path("configs/stage052_performance.toml"))

    assert config.storage_root_locator == Path("configs/stage052_storage_roots.local.toml")
    assert config.staging_root_alias == "transfer_staging"
    assert config.archive_root_aliases == ("transfer_archive", "internal_archive")


def test_campaign_root_location_drift_is_rejected_before_writes(tmp_path: Path) -> None:
    volume = VolumeIdentity("test-volume", "apfs")
    locator = StorageRootLocator(
        {
            "transfer_staging": StorageRoot(
                "transfer_staging", tmp_path / "results", volume
            ),
            "transfer_archive": StorageRoot(
                "transfer_archive",
                Path("/Volumes/TRANSFER/FURP-2026-Yiyang-GUO-EVRP-TW-results"),
                volume,
            ),
            "internal_archive": StorageRoot(
                "internal_archive", tmp_path / "wrong-internal-root", volume
            ),
        }
    )

    with pytest.raises(RuntimeError, match="absolute locations"):
        verify_campaign_root_locations(repository_root=tmp_path, locator=locator)
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "wrong-internal-root").exists()


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
        native_profile="stage05.2-native-kernels-v1",
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


def test_preflight_rejects_one_process_averaging_a_full_core_for_window() -> None:
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

    with pytest.raises(RuntimeError, match="full CPU core"):
        collect_preflight_observation(
            _pilot_config(),
            snapshot=snapshot,
            monotonic=clock,
            sleep=sleep,
            sample_interval_seconds=1.0,
        )


def test_preflight_counts_a_process_created_after_the_window_baseline() -> None:
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

    with pytest.raises(RuntimeError, match="full CPU core"):
        collect_preflight_observation(
            _pilot_config(),
            snapshot=snapshot,
            monotonic=clock,
            sleep=sleep,
            sample_interval_seconds=1.0,
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
        logical_cpu_count=12,
    )

    assert evidence.passed is False
    assert evidence.maximum_unrelated_process_average_cores == pytest.approx(40.0 / 30.0)


def test_runtime_evidence_rejects_power_or_load_drift() -> None:
    evidence = BatchRuntimeEvidence.from_snapshots(
        (
            MachineSnapshot("AC Power", False, 1.0, 0.1),
            MachineSnapshot("Battery Power", False, 4.1, 1.0),
        ),
        config=_pilot_config(),
    )

    assert evidence.passed is False
    assert "power" in evidence.failure_reason
    assert "load1" in evidence.failure_reason


def test_runtime_evidence_rejects_full_window_unrelated_core_average() -> None:
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

    assert evidence.passed is False
    assert "unrelated" in evidence.failure_reason


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


def _dispatcher_locator(tmp_path: Path) -> StorageRootLocator:
    external = VolumeIdentity("external-device", "exfat")
    internal = VolumeIdentity("internal-device", "apfs")
    return StorageRootLocator(
        {
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
    lock = SimpleNamespace(
        selected_workers=2,
        selected_backend="native_cpu",
        selected_exact_backend="cpu_batch",
        verify_current_execution=lambda **_kwargs: None,
        to_dict=lambda: {
            "selected_backend": "native_cpu",
            "selected_exact_backend": "cpu_batch",
            "selected_workers": 2,
        },
    )
    monkeypatch.setattr(
        stage052_performance,
        "load_benchmark_execution_lock",
        lambda *_args, **_kwargs: lock,
    )
    monkeypatch.setattr(stage052_performance, "_git", lambda *_args: "a" * 40)
    monkeypatch.setattr(
        stage052_performance,
        "verify_stage052_runtime_identity",
        lambda *_args, **_kwargs: {"runtime_identity_sha256": "b" * 64},
    )
    monkeypatch.setattr(stage052_performance, "collect_environment", lambda: {})
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
    output_dir = tmp_path / "results" / "stage05.2_benchmark_attempt01"
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
            worker_count=2,
            config=config,
            stage051_prerequisite={"status": "READY_FOR_STAGE05_2"},
            component_prerequisites={"accelerator_decision": {"status": "accepted"}},
            resolved_prerequisite_dirs={"accelerator_decision": prerequisite_dir},
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
