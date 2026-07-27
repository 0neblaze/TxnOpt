"""Fail-fast execution boundary for Stage 5.2 benchmark campaigns.

The campaign contract in :mod:`evrptw.stage052_campaign` owns path-free
planning and state.  This module binds that contract to accepted predecessor
evidence and to the local machine boundaries used by the experiment runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import plistlib
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from evrptw.artifacts import (
    ArtifactReader,
    atomic_write_signed_json,
    signed_sidecar_matches,
)
from evrptw.stage052 import STAGE052_MAXIMUM_PERSISTENCE_RATIO
from evrptw.stage052_campaign import (
    RUNTIME_LOAD_POLICY,
    ArchiveTransferCompletedError,
    BatchArchiver,
    BatchManifest,
    BatchPlan,
    BenchmarkCampaignConfig,
    BenchmarkPreflightObservation,
    CampaignManifest,
    PilotStorageObservation,
    ProcessCpuCounterSample,
    StorageRootLocator,
    SystemLoadWindow,
    VolumeIdentity,
    directory_byte_count,
    directory_checksum,
    load_campaign_manifest,
    maximum_process_average_cores,
    maximum_process_average_cores_over_windows,
)
from evrptw.stage052_evidence import (
    STAGE052_RESOURCE_SCHEMA_VERSION,
    PersistenceInterval,
    RunResourceSummary,
    verify_stage052_campaign_gate_set,
    verify_stage052_review_files,
)
from evrptw.stage052_platform import (
    WindowsWslPowerStatus,
    read_windows_wsl_power_status,
    read_wsl_ac_power_online,
)

PER_WORKER_RSS_LIMIT_BYTES = 8 * 1024**3
AGGREGATE_RSS_LIMIT_BYTES = 20 * 1024**3
STAGE052_MINIMUM_FREE_BYTES = 50 * 1024**3
_CAMPAIGN_SUCCESSOR_ALLOWED_PATHS = frozenset(
    {
        "AGENTS.md",
        "docs/stage052_change_log.md",
        "docs/stage052_performance_benchmark_workflow.md",
        "experiments/migrations/stage052_d_archive_ssd_20260725.json",
        "experiments/migrations/stage052_d_archive_ssd_20260725.json.sha256",
        "src/evrptw/artifacts.py",
        "src/evrptw/experiments/stage052_campaign_review.py",
        "src/evrptw/experiments/stage052_performance.py",
        "src/evrptw/experiments/stage052_performance_review.py",
        "src/evrptw/stage052_campaign.py",
        "src/evrptw/stage052_campaign_runner.py",
        "src/evrptw/stage052_evidence.py",
        "src/evrptw/stage052_platform.py",
        "src/evrptw/stage052_review_service.py",
        "src/evrptw/stage052_storage_migration.py",
        "tests/test_stage052_campaign_review.py",
        "tests/test_stage052_campaign_runner.py",
        "tests/test_stage052_platform.py",
        "tests/test_stage052_review_service.py",
        "tests/test_stage052_storage_migration.py",
        "tests/test_artifacts_v3.py",
    }
)
_CAMPAIGN_SUCCESSOR_PINNED_PRODUCER_FIXES = {
    "src/evrptw/alns.py": "392164b1fd53541c2a13924dcc95ceb82a71d591c025910070594c297582163a",
    "tests/test_alns_wall_clock_only.py": (
        "0bf741d1e4929564882dd8ff48775f0c589e957328aa00c7309176656fd16481"
    ),
    "tests/test_artifacts_v3.py": (
        "4da79f2adaf7c0f14394c3a3698009f4d290d99df40797f5197b157ccadbbd66"
    ),
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def campaign_runtime_selection_sha256(
    value: Mapping[str, object],
    *,
    storage_migration: Mapping[str, object] | None = None,
) -> str:
    """Hash the frozen runtime selection while excluding G-only wheel identity."""

    if storage_migration is not None:
        value = _runtime_selection_with_attested_archive_source(
            value,
            storage_migration,
        )
    excluded = {
        "installed_distribution_sha256",
        "repository_revision",
        "wheel_filename",
        "wheel_sha256",
    }
    selection = {key: item for key, item in value.items() if key not in excluded}
    machine = selection.get("machine_identity")
    if isinstance(machine, Mapping):
        selection["machine_identity"] = {
            key: item for key, item in machine.items() if key != "memory_bytes"
        }
    return _canonical_sha256(
        selection
    )


def _runtime_selection_with_attested_archive_source(
    runtime: Mapping[str, object],
    storage_migration: Mapping[str, object],
) -> dict[str, object]:
    """Normalize only an attested archive-disk replacement to its source identity."""

    machine = _mapping(runtime.get("machine_identity"), "runtime machine identity")
    observed_disk = _mapping(
        machine.get("d_archive_disk"),
        "runtime D archive disk identity",
    )
    source_disk = _mapping(
        storage_migration.get("source_machine_disk"),
        "storage migration source disk identity",
    )
    destination_disk = _mapping(
        storage_migration.get("destination_machine_disk"),
        "storage migration destination disk identity",
    )
    if observed_disk != destination_disk:
        raise RuntimeError(
            "benchmark runtime archive disk differs from the attested migration destination"
        )
    normalized_machine = dict(machine)
    normalized_machine["d_archive_disk"] = dict(source_disk)
    normalized_runtime = dict(runtime)
    normalized_runtime["machine_identity"] = normalized_machine
    return normalized_runtime


def verify_campaign_successor_revision(
    repository: Path,
    *,
    predecessor_revision: str,
    current_revision: str,
) -> tuple[str, ...]:
    """Allow a newer revision only when every change is confined to G governance."""

    for label, revision in (
        ("predecessor", predecessor_revision),
        ("current", current_revision),
    ):
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise RuntimeError(f"campaign {label} repository revision is invalid")
    resolved = repository.resolve()
    ancestor = subprocess.run(
        (
            "git",
            "-C",
            str(resolved),
            "merge-base",
            "--is-ancestor",
            predecessor_revision,
            current_revision,
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            "benchmark repository revision is not a descendant of the accepted selection"
        )
    result = subprocess.run(
        (
            "git",
            "-C",
            str(resolved),
            "diff",
            "--name-only",
            "-z",
            predecessor_revision,
            current_revision,
        ),
        check=True,
        capture_output=True,
        timeout=10.0,
    )
    changed_paths = tuple(
        sorted(
            path.decode("utf-8")
            for path in result.stdout.split(b"\0")
            if path
        )
    )
    forbidden = tuple(
        path
        for path in changed_paths
        if path not in _CAMPAIGN_SUCCESSOR_ALLOWED_PATHS
        and path not in _CAMPAIGN_SUCCESSOR_PINNED_PRODUCER_FIXES
    )
    if not changed_paths:
        raise RuntimeError("campaign successor revision has no recorded changes")
    if forbidden:
        raise RuntimeError(
            "campaign successor revision changes non-G paths: " + ", ".join(forbidden)
        )
    pinned_paths = frozenset(_CAMPAIGN_SUCCESSOR_PINNED_PRODUCER_FIXES)
    pinned_changed = frozenset(changed_paths) & pinned_paths
    if pinned_changed and pinned_changed != pinned_paths:
        raise RuntimeError("campaign successor revision has an incomplete pinned producer fix")
    for path in sorted(pinned_changed):
        content = subprocess.run(
            ("git", "-C", str(resolved), "show", f"{current_revision}:{path}"),
            check=True,
            capture_output=True,
            timeout=10.0,
        ).stdout
        if (
            hashlib.sha256(content).hexdigest()
            != _CAMPAIGN_SUCCESSOR_PINNED_PRODUCER_FIXES[path]
        ):
            raise RuntimeError(
                f"campaign successor revision changes pinned producer-fix content: {path}"
            )
    return changed_paths


def _input_lock_payload(value: Mapping[str, object]) -> dict[str, object]:
    """Remove scope/runtime observations while retaining stable solver inputs."""

    excluded = {
        "instance_sha256",
        "background_load",
        "power_mode",
        "runtime_signature",
        "process_status_counts",
        "load_average",
    }
    return {key: item for key, item in value.items() if key not in excluded}


def _instance_hashes(value: Mapping[str, object]) -> dict[str, str]:
    raw = value.get("instance_sha256")
    if (
        not isinstance(raw, Mapping)
        or not raw
        or any(
            not isinstance(instance, str)
            or not instance
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for instance, digest in raw.items()
        )
    ):
        raise RuntimeError("accepted benchmark instance hashes are invalid")
    return {str(instance): str(digest) for instance, digest in raw.items()}


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RuntimeError(f"accepted benchmark {field} must be an object")
    return {str(key): item for key, item in value.items()}


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"accepted benchmark {field} must be a SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class MachineSnapshot:
    """One power/load sample at the local machine boundary."""

    power_source: str
    low_power_mode_enabled: bool
    load1: float
    unrelated_process_average_cores: float
    sampled_at_seconds: float | None = None
    unrelated_process_cpu_seconds: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.power_source:
            raise ValueError("machine power_source is required")
        if not isinstance(self.low_power_mode_enabled, bool):
            raise ValueError("machine low_power_mode_enabled must be boolean")
        if (
            not math.isfinite(self.load1)
            or self.load1 < 0.0
            or not math.isfinite(self.unrelated_process_average_cores)
            or self.unrelated_process_average_cores < 0.0
        ):
            raise ValueError("machine load sample is invalid")
        if self.sampled_at_seconds is not None and (
            not math.isfinite(self.sampled_at_seconds) or self.sampled_at_seconds < 0.0
        ):
            raise ValueError("machine sample timestamp is invalid")
        if any(
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or isinstance(seconds, bool)
            or not isinstance(seconds, int | float)
            or not math.isfinite(float(seconds))
            or float(seconds) < 0.0
            for pid, seconds in self.unrelated_process_cpu_seconds.items()
        ):
            raise ValueError("machine process CPU counters are invalid")


def _maximum_window_process_average_cores(
    snapshots: Sequence[MachineSnapshot],
    *,
    logical_cpu_count: int | None = None,
) -> float:
    """Return a replayable conservative CPU-time average over the full window."""

    if not snapshots:
        raise ValueError("machine window requires at least one snapshot")
    counter_snapshots = [
        sample
        for sample in snapshots
        if sample.sampled_at_seconds is not None
    ]
    if len(counter_snapshots) >= 2:
        samples = tuple(
            ProcessCpuCounterSample(
                sampled_at_seconds=cast(float, sample.sampled_at_seconds),
                cpu_seconds_by_pid=sample.unrelated_process_cpu_seconds,
            )
            for sample in counter_snapshots
        )
        return maximum_process_average_cores(
            samples,
            logical_cpu_count=logical_cpu_count or (os.cpu_count() or 1),
        )
    # Compatibility for injected/legacy snapshots without cumulative counters:
    # use the full-window sample mean, never an instantaneous maximum.
    return sum(sample.unrelated_process_average_cores for sample in snapshots) / len(
        snapshots
    )


@dataclass(frozen=True, slots=True)
class BatchRuntimeEvidence:
    """Continuous power/load evidence for one campaign batch."""

    sample_count: int
    maximum_load1: float
    maximum_permitted_load1: float
    maximum_unrelated_process_average_cores: float
    power_sources: tuple[str, ...]
    low_power_mode_observed: bool
    passed: bool
    failure_reason: str
    logical_cpu_count: int | None = None
    process_cpu_samples: tuple[ProcessCpuCounterSample, ...] = ()

    @classmethod
    def from_snapshots(
        cls,
        snapshots: tuple[MachineSnapshot, ...],
        *,
        config: BenchmarkCampaignConfig,
        logical_cpu_count: int | None = None,
    ) -> BatchRuntimeEvidence:
        if not snapshots:
            raise ValueError("batch runtime evidence requires at least one sample")
        sources = tuple(sorted({sample.power_source for sample in snapshots}))
        maximum_load1 = max(sample.load1 for sample in snapshots)
        counter_samples = tuple(
            ProcessCpuCounterSample(
                sampled_at_seconds=float(sample.sampled_at_seconds),
                cpu_seconds_by_pid=sample.unrelated_process_cpu_seconds,
            )
            for sample in snapshots
            if sample.sampled_at_seconds is not None
        )
        recorded_logical_cpu_count = logical_cpu_count or (os.cpu_count() or 1)
        if (
            isinstance(recorded_logical_cpu_count, bool)
            or recorded_logical_cpu_count <= 0
        ):
            raise ValueError("batch runtime logical CPU count is invalid")
        maximum_unrelated = (
            maximum_process_average_cores_over_windows(
                counter_samples,
                logical_cpu_count=recorded_logical_cpu_count,
                window_seconds=config.preflight_window_seconds,
            )
            if len(counter_samples) >= 2
            else _maximum_window_process_average_cores(snapshots)
        )
        maximum_permitted_load1 = RUNTIME_LOAD_POLICY.runtime_maximum_load1
        low_power = any(sample.low_power_mode_enabled for sample in snapshots)
        failures: list[str] = []
        if sources != (config.required_power_source,):
            failures.append("power source drift")
        if low_power:
            failures.append("low power mode enabled")
        if maximum_load1 > maximum_permitted_load1:
            failures.append(f"load1 exceeded {maximum_permitted_load1:.1f}")
        if (
            recorded_logical_cpu_count is not None
            and recorded_logical_cpu_count != RUNTIME_LOAD_POLICY.logical_cpu_count
        ):
            failures.append(
                "logical CPU count differs from the frozen "
                f"{RUNTIME_LOAD_POLICY.logical_cpu_count}-thread machine"
            )
        if maximum_unrelated >= config.maximum_unrelated_process_average_cores:
            failures.append("unrelated process averaged one full core")
        return cls(
            sample_count=len(snapshots),
            maximum_load1=maximum_load1,
            maximum_permitted_load1=maximum_permitted_load1,
            maximum_unrelated_process_average_cores=maximum_unrelated,
            power_sources=sources,
            low_power_mode_observed=low_power,
            passed=not failures,
            failure_reason="; ".join(failures),
            logical_cpu_count=recorded_logical_cpu_count,
            process_cpu_samples=counter_samples if len(counter_samples) >= 2 else (),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "stage05.2-batch-runtime-evidence-v1",
            "sample_count": self.sample_count,
            "maximum_load1": self.maximum_load1,
            "maximum_permitted_load1": self.maximum_permitted_load1,
            "maximum_unrelated_process_average_cores": (
                self.maximum_unrelated_process_average_cores
            ),
            "power_sources": list(self.power_sources),
            "low_power_mode_observed": self.low_power_mode_observed,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
            "logical_cpu_count": self.logical_cpu_count,
            "process_cpu_samples": [
                sample.to_dict() for sample in self.process_cpu_samples
            ],
        }


def collect_preflight_observation(
    config: BenchmarkCampaignConfig,
    *,
    snapshot: Callable[[], MachineSnapshot],
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    sample_interval_seconds: float = 1.0,
    require_idle_load: bool = True,
) -> BenchmarkPreflightObservation:
    """Measure exactly two consecutive preflight windows."""

    if not math.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0.0:
        raise ValueError("preflight sample interval must be positive")
    campaign_started = monotonic()
    observed_power: str | None = None
    observed_low_power = False
    windows: list[SystemLoadWindow] = []
    for window_index in range(config.preflight_window_count):
        window_started = window_index * config.preflight_window_seconds
        absolute_deadline = campaign_started + (
            (window_index + 1) * config.preflight_window_seconds
        )
        maximum_load1 = 0.0
        window_snapshots: list[MachineSnapshot] = []
        while True:
            sample = snapshot()
            window_snapshots.append(sample)
            if observed_power is None:
                observed_power = sample.power_source
            elif sample.power_source != observed_power:
                raise RuntimeError("power source changed during campaign preflight")
            observed_low_power = observed_low_power or sample.low_power_mode_enabled
            maximum_load1 = max(maximum_load1, sample.load1)
            remaining = absolute_deadline - monotonic()
            if remaining <= 0.0:
                break
            sleep(min(sample_interval_seconds, remaining))
        replay_samples = tuple(
            ProcessCpuCounterSample(
                sampled_at_seconds=float(sample.sampled_at_seconds),
                cpu_seconds_by_pid=sample.unrelated_process_cpu_seconds,
            )
            for sample in window_snapshots
            if sample.sampled_at_seconds is not None
        )
        windows.append(
            SystemLoadWindow(
                started_at_seconds=window_started,
                duration_seconds=config.preflight_window_seconds,
                maximum_load1=maximum_load1,
                maximum_unrelated_process_average_cores=(
                    _maximum_window_process_average_cores(window_snapshots)
                ),
                logical_cpu_count=(os.cpu_count() or 1) if len(replay_samples) >= 2 else None,
                process_cpu_samples=replay_samples if len(replay_samples) >= 2 else (),
            )
        )
    observation = BenchmarkPreflightObservation(
        power_source=observed_power or "unknown",
        low_power_mode_enabled=observed_low_power,
        windows=tuple(windows),
    )
    config.validate_preflight(observation, require_idle_load=require_idle_load)
    return observation


def _non_negative_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"batch {field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(f"batch {field} must be finite and non-negative")
    return result


def validate_batch_measurements(
    *,
    batch: BatchPlan,
    rows: Sequence[Mapping[str, object]],
    resource_summary: RunResourceSummary,
    runtime_evidence: BatchRuntimeEvidence,
    expected_workers: int,
    additional_persistence_seconds: float = 0.0,
) -> float:
    """Enforce per-batch geometry, persistence, memory, and fallback gates."""

    expected = {
        (shard.instance, shard.seed, f"wall_clock_{budget}")
        for shard in batch.shards
        for budget in shard.budgets_seconds
    }
    observed: set[tuple[str, int, str]] = set()
    solver_seconds = 0.0
    persistence_seconds = _non_negative_number(
        additional_persistence_seconds,
        "additional_persistence_seconds",
    )
    for row in rows:
        instance = row.get("instance")
        seed = row.get("seed")
        axis = row.get("axis")
        if (
            not isinstance(instance, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(axis, str)
        ):
            raise RuntimeError("batch per-run identity is invalid")
        identity = (instance, seed, axis)
        if identity in observed:
            raise RuntimeError(f"batch per-run identity is duplicate: {identity}")
        observed.add(identity)
        if row.get("backend") != "cpu_batch":
            raise RuntimeError("batch backend is not cpu_batch")
        if row.get("worker_count") != expected_workers:
            raise RuntimeError("batch row worker count differs from selected workers")
        for field_name in ("native_fallbacks", "native_protocol_fallbacks"):
            value = row.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value != 0:
                raise RuntimeError(f"batch {field_name} must be zero")
        solver_seconds += _non_negative_number(row.get("solver_seconds"), "solver_seconds")
        persistence_seconds += _non_negative_number(
            row.get("artifact_persistence_seconds"),
            "artifact_persistence_seconds",
        )
    if observed != expected:
        raise RuntimeError(
            f"batch axis geometry mismatch: expected={sorted(expected)} observed={sorted(observed)}"
        )
    denominator = solver_seconds + persistence_seconds
    if denominator <= 0.0:
        raise RuntimeError("batch solver plus persistence time must be positive")
    persistence_ratio = persistence_seconds / denominator
    if persistence_ratio > STAGE052_MAXIMUM_PERSISTENCE_RATIO:
        raise RuntimeError(
            "batch aggregate persistence ratio exceeds "
            f"{STAGE052_MAXIMUM_PERSISTENCE_RATIO:.0%}: {persistence_ratio:.9f}"
        )
    if not runtime_evidence.passed:
        raise RuntimeError(f"batch runtime power/load violation: {runtime_evidence.failure_reason}")
    if (
        resource_summary.schema_version != STAGE052_RESOURCE_SCHEMA_VERSION
        or resource_summary.configured_worker_count != expected_workers
        or resource_summary.status != "complete"
    ):
        raise RuntimeError("batch resource summary identity is invalid")
    if resource_summary.aggregate_peak_rss_bytes > AGGREGATE_RSS_LIMIT_BYTES:
        raise RuntimeError("batch process-tree aggregate RSS exceeds 20 GiB")
    descendants = set(resource_summary.descendant_pids)
    process_peaks = dict(resource_summary.process_peak_rss_bytes)
    if not descendants or not descendants.issubset(process_peaks):
        raise RuntimeError("batch worker process RSS evidence is incomplete")
    if any(
        process_peaks[pid] > PER_WORKER_RSS_LIMIT_BYTES for pid in descendants
    ):
        raise RuntimeError("batch per-worker RSS exceeds the fixed limit")
    return persistence_ratio


@dataclass(frozen=True, slots=True)
class BenchmarkExecutionLock:
    """Execution identity frozen by accepted F02 or G01 evidence."""

    prerequisite_run_label: str
    raw_manifest_sha256: str
    selected_backend: str
    selected_exact_backend: str
    selected_workers: int
    repository_revision: str
    runtime_identity_sha256: str
    runtime_selection_sha256: str
    input_provenance_sha256: str
    configuration_sha256: str
    native_config_sha256: str
    native_kernel_config: Mapping[str, object]
    instance_sha256: Mapping[str, str]
    staging_root_alias: str | None = None
    archive_root_aliases_exercised: tuple[str, ...] = ()

    @classmethod
    def from_accepted_evidence(
        cls,
        *,
        metadata: Mapping[str, object],
        review_manifest: Mapping[str, object],
        raw_manifest_sha256: str,
        expected_scope: str,
        expected_status: str,
    ) -> BenchmarkExecutionLock:
        """Bind every performance-affecting field from one accepted review."""

        raw_sha = _sha256(raw_manifest_sha256, "raw manifest digest")
        run_label = metadata.get("run_label")
        if not isinstance(run_label, str) or not run_label:
            raise RuntimeError("accepted benchmark run_label is missing")
        for field_name in ("run_label", "component", "scope"):
            if review_manifest.get(field_name) != metadata.get(field_name):
                raise RuntimeError(
                    f"accepted benchmark review {field_name} identity mismatch"
                )
        if metadata.get("scope") != expected_scope:
            raise RuntimeError("accepted benchmark predecessor scope mismatch")
        if review_manifest.get("status") != expected_status:
            raise RuntimeError("accepted benchmark predecessor review status mismatch")
        if review_manifest.get("raw_manifest_sha256") != raw_sha:
            raise RuntimeError("accepted benchmark review/raw manifest identity mismatch")
        selected_backend = review_manifest.get("selected_backend")
        raw_accelerator_decision = review_manifest.get("accelerator_decision")
        accelerator_decision = (
            raw_accelerator_decision if isinstance(raw_accelerator_decision, str) else ""
        )
        expected_backend = {
            "GPU_NOT_JUSTIFIED": "native_cpu",
            "NATIVE_CPU_RETAINED": "native_cpu",
            "ACCELERATOR_PROMOTED": "cuda",
        }.get(accelerator_decision)
        if (
            expected_backend is None
            or selected_backend != expected_backend
            or metadata.get("backend") != "cpu_batch"
            or metadata.get("execution_backend") != selected_backend
            or review_manifest.get("selected_exact_backend") != "cpu_batch"
        ):
            raise RuntimeError(
                "Stage 5.2 benchmark execution backend does not match the accepted "
                "accelerator decision/cpu_batch exact backend"
            )
        expected_profile = "cuda" if selected_backend == "cuda" else "native"
        if (
            metadata.get("optimization_profile") != expected_profile
            or review_manifest.get("selected_optimization_profile") != expected_profile
        ):
            raise RuntimeError("accepted accelerator optimization profile is inconsistent")
        workers = metadata.get("worker_count")
        if isinstance(workers, bool) or workers not in {2, 4}:
            raise RuntimeError("accepted benchmark worker selection must be 2 or 4")
        if review_manifest.get("selected_workers") != workers:
            raise RuntimeError("accepted benchmark review worker selection mismatch")
        revision = metadata.get("repository_revision")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise RuntimeError("accepted benchmark repository revision is invalid")
        native = _mapping(metadata.get("native_kernel_config"), "native kernel config")
        required_native = {
            "enabled": True,
            "exact_charging": True,
            "screening": True,
            "propagation": True,
            "distance_matrix": True,
            "abi_version": "stage05.2-native-kernels-v1",
            "context_policy": "pack_once_per_solve",
            "failure_policy": "fail_fast_no_fallback",
        }
        if native != required_native:
            raise RuntimeError("accepted benchmark native kernel profile is incomplete")
        if review_manifest.get("native_configuration") != native:
            raise RuntimeError("accepted benchmark review native configuration mismatch")
        compatibility_native = review_manifest.get("native_kernel_config")
        if compatibility_native is not None and compatibility_native != native:
            raise RuntimeError("accepted benchmark review native compatibility field mismatch")
        runtime = _mapping(metadata.get("runtime_identity"), "runtime identity")
        inputs = _mapping(metadata.get("performance_provenance"), "input provenance")
        config_sha = _sha256(metadata.get("configuration_sha256"), "configuration digest")
        if metadata.get("storage_policy_version") != "artifact-storage-v2":
            raise RuntimeError("accepted benchmark storage policy is not artifact-storage-v2")
        if metadata.get("screening_schema_version") != "screening_decisions_v3":
            raise RuntimeError("accepted benchmark physical schema is not screening_decisions_v3")
        staging_root_alias: str | None = None
        archive_root_aliases_exercised: tuple[str, ...] = ()
        if expected_scope == "pilot":
            raw_staging_alias = review_manifest.get("staging_root_alias")
            raw_archive_aliases = review_manifest.get("archive_root_aliases_exercised")
            if (
                not isinstance(raw_staging_alias, str)
                or not raw_staging_alias
                or not isinstance(raw_archive_aliases, list)
                or not raw_archive_aliases
                or any(not isinstance(alias, str) or not alias for alias in raw_archive_aliases)
                or len(set(raw_archive_aliases)) != len(raw_archive_aliases)
                or raw_staging_alias in raw_archive_aliases
            ):
                raise RuntimeError(
                    "accepted G01 review storage root aliases are incomplete or invalid"
                )
            staging_root_alias = raw_staging_alias
            archive_root_aliases_exercised = tuple(sorted(raw_archive_aliases))
        return cls(
            prerequisite_run_label=run_label,
            raw_manifest_sha256=raw_sha,
            selected_backend=str(selected_backend),
            selected_exact_backend="cpu_batch",
            selected_workers=workers,
            repository_revision=revision,
            runtime_identity_sha256=_canonical_sha256(runtime),
            runtime_selection_sha256=campaign_runtime_selection_sha256(runtime),
            input_provenance_sha256=_canonical_sha256(_input_lock_payload(inputs)),
            configuration_sha256=config_sha,
            native_config_sha256=_canonical_sha256(native),
            native_kernel_config=dict(native),
            instance_sha256=_instance_hashes(inputs),
            staging_root_alias=staging_root_alias,
            archive_root_aliases_exercised=archive_root_aliases_exercised,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "prerequisite_run_label": self.prerequisite_run_label,
            "raw_manifest_sha256": self.raw_manifest_sha256,
            "selected_backend": self.selected_backend,
            "selected_exact_backend": self.selected_exact_backend,
            "selected_workers": self.selected_workers,
            "repository_revision": self.repository_revision,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "runtime_selection_sha256": self.runtime_selection_sha256,
            "input_provenance_sha256": self.input_provenance_sha256,
            "configuration_sha256": self.configuration_sha256,
            "native_config_sha256": self.native_config_sha256,
            "native_kernel_config": dict(self.native_kernel_config),
            "instance_sha256": dict(sorted(self.instance_sha256.items())),
            "staging_root_alias": self.staging_root_alias,
            "archive_root_aliases_exercised": list(self.archive_root_aliases_exercised),
        }

    def verify_planned_storage_roots(
        self,
        *,
        staging_root_alias: str,
        planned_archive_root_aliases: tuple[str, ...],
    ) -> None:
        """Reject Formal storage-root drift before any batch is dispatched."""

        if self.staging_root_alias is None or not self.archive_root_aliases_exercised:
            raise RuntimeError(
                "accepted G01 review does not freeze Formal storage root aliases"
            )
        if staging_root_alias != self.staging_root_alias:
            raise RuntimeError(
                "Formal staging root alias differs from the accepted G01 review"
            )
        if (
            not planned_archive_root_aliases
            or any(
                not isinstance(alias, str) or not alias
                for alias in planned_archive_root_aliases
            )
            or len(set(planned_archive_root_aliases)) != len(planned_archive_root_aliases)
            or staging_root_alias in planned_archive_root_aliases
        ):
            raise RuntimeError("Formal planned archive root alias set is invalid")
        if not set(planned_archive_root_aliases).issubset(
            self.archive_root_aliases_exercised
        ):
            raise RuntimeError(
                "Formal planned archive root aliases exceed accepted G01 drill coverage"
            )

    def verify_current_execution(
        self,
        *,
        selected_backend: str,
        selected_exact_backend: str,
        selected_workers: int,
        repository_revision: str,
        configuration_sha256: str,
        runtime_identity: object,
        input_provenance: object,
        native_kernel_config: object,
        repository: Path | None = None,
        storage_migration: Mapping[str, object] | None = None,
    ) -> None:
        """Reject any execution drift from the accepted predecessor lock."""

        if selected_backend != self.selected_backend:
            raise RuntimeError("benchmark execution backend differs from accepted selection")
        if selected_exact_backend != self.selected_exact_backend:
            raise RuntimeError("benchmark exact backend differs from accepted selection")
        if selected_workers != self.selected_workers:
            raise RuntimeError("benchmark worker count differs from accepted selection")
        if configuration_sha256 != self.configuration_sha256:
            raise RuntimeError("benchmark configuration differs from accepted selection")
        current_runtime = _mapping(runtime_identity, "runtime identity")
        if repository_revision == self.repository_revision:
            if _canonical_sha256(current_runtime) != self.runtime_identity_sha256:
                raise RuntimeError("benchmark runtime identity differs from accepted selection")
        else:
            if repository is None:
                raise RuntimeError(
                    "benchmark successor revision requires an auditable repository"
                )
            verify_campaign_successor_revision(
                repository,
                predecessor_revision=self.repository_revision,
                current_revision=repository_revision,
            )
            current_selection_sha256 = campaign_runtime_selection_sha256(
                current_runtime
            )
            if (
                current_selection_sha256 != self.runtime_selection_sha256
                and storage_migration is not None
            ):
                current_selection_sha256 = campaign_runtime_selection_sha256(
                    _runtime_selection_with_attested_archive_source(
                        current_runtime,
                        storage_migration,
                    )
                )
            if current_selection_sha256 != self.runtime_selection_sha256:
                raise RuntimeError(
                    "benchmark runtime selection differs from accepted selection"
                )
        current_inputs = _mapping(input_provenance, "input provenance")
        if _canonical_sha256(_input_lock_payload(current_inputs)) != self.input_provenance_sha256:
            raise RuntimeError("benchmark input provenance differs from accepted selection")
        current_instance_hashes = _instance_hashes(current_inputs)
        if any(
            current_instance_hashes.get(instance) != digest
            for instance, digest in self.instance_sha256.items()
        ):
            raise RuntimeError("benchmark instance inputs differ from accepted selection")
        if (
            _canonical_sha256(_mapping(native_kernel_config, "native kernel config"))
            != self.native_config_sha256
        ):
            raise RuntimeError("benchmark native configuration differs from accepted selection")


def load_benchmark_execution_lock(
    prerequisite_dir: Path,
    *,
    expected_scope: str,
    expected_status: str,
) -> BenchmarkExecutionLock:
    """Load the verified producer metadata and its current accepted review."""

    reader = ArtifactReader(prerequisite_dir)
    references = [
        item
        for item in reader.manifest.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("artifact_type") == "manifest_metadata"
    ]
    if len(references) != 1:
        raise RuntimeError("accepted benchmark predecessor lacks one metadata artifact")
    relative_path = references[0].get("relative_path")
    if not isinstance(relative_path, str):
        raise RuntimeError("accepted benchmark metadata path is invalid")
    metadata = reader.read_json(relative_path)
    review_path = prerequisite_dir / "review" / "review_manifest.json"
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("accepted benchmark review manifest is unreadable") from error
    if not isinstance(review, Mapping):
        raise RuntimeError("accepted benchmark review manifest must be an object")
    expected_review_identity = (
        ("stage05.2-campaign-review-v1", "benchmark")
        if expected_scope == "pilot"
        else ("stage05.2-review-v1", "accelerator_pilot")
    )
    if (
        review.get("schema_version") != expected_review_identity[0]
        or review.get("component") != expected_review_identity[1]
    ):
        raise RuntimeError("accepted benchmark review schema/component is invalid")
    if expected_scope == "pilot":
        verify_stage052_campaign_gate_set(review, scope="pilot")
    else:
        gates = review.get("gates")
        if (
            not isinstance(gates, Mapping)
            or not gates
            or any(
                not isinstance(gate, Mapping) or gate.get("passed") is not True
                for gate in gates.values()
            )
        ):
            raise RuntimeError("accepted benchmark review contains a failed or invalid gate")
    verified_review_files = verify_stage052_review_files(prerequisite_dir, review)
    if any(
        len(Path(relative).parts) != 3
        or Path(relative).parts[0] != "generations"
        for relative in verified_review_files
    ):
        raise RuntimeError("accepted benchmark review must use an immutable generation")
    if metadata.get("persistence_attribution") == "primary_active_writes_v1":
        attribution_path = (
            prerequisite_dir
            / "control"
            / f"{prerequisite_dir.name}_persistence_attribution.json"
        )
        attribution_sidecar = attribution_path.with_suffix(".sha256")
        if (
            not attribution_path.is_file()
            or not attribution_sidecar.is_file()
            or not signed_sidecar_matches(attribution_path, attribution_sidecar)
            or review.get("persistence_attribution_sha256")
            != _file_sha256(attribution_path)
            or review.get("persistence_attribution_sidecar_sha256")
            != _file_sha256(attribution_sidecar)
        ):
            raise RuntimeError("accepted benchmark persistence attribution is stale")
    manifest_path = reader.result.manifest_path
    if expected_scope == "pilot":
        campaign_path = prerequisite_dir / "campaign_manifest.json"
        campaign = load_campaign_manifest(campaign_path)
        if (
            campaign.status != "complete"
            or campaign.scope != "pilot"
            or campaign.run_label != prerequisite_dir.name
            or review.get("raw_campaign_manifest_sha256")
            != _file_sha256(campaign_path)
        ):
            raise RuntimeError("accepted G01 campaign manifest binding is stale")
    return BenchmarkExecutionLock.from_accepted_evidence(
        metadata=metadata,
        review_manifest=review,
        raw_manifest_sha256=_file_sha256(manifest_path),
        expected_scope=expected_scope,
        expected_status=expected_status,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def probe_volume_identity(path: Path) -> VolumeIdentity:
    """Return the mounted volume UUID/filesystem for one local root."""

    if sys.platform == "darwin":
        return _probe_macos_volume_identity(path)
    completed = subprocess.run(
        (
            "findmnt",
            "--json",
            "--target",
            str(path),
            "--output",
            "SOURCE,FSTYPE,UUID",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
        filesystems = payload["filesystems"]
        mount = filesystems[0]
        source = str(mount["source"])
        filesystem = str(mount["fstype"])
        uuid_value = mount.get("uuid")
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"findmnt returned invalid volume metadata for {path}") from error
    if isinstance(uuid_value, str) and uuid_value:
        return VolumeIdentity(device_uuid=uuid_value, filesystem=filesystem)
    drive_match = re.fullmatch(r"([A-Za-z]):\\", source)
    if drive_match is None:
        raise RuntimeError(f"mounted volume identity is incomplete for {path}: {source}")
    return VolumeIdentity(
        device_uuid=_windows_nvme_identity(drive_match.group(1)),
        filesystem=filesystem,
    )


def _windows_nvme_identity(drive_letter: str) -> str:
    powershell = shutil.which("pwsh.exe") or shutil.which("powershell.exe")
    if powershell is None:
        raise RuntimeError("Windows drive identity requires PowerShell")
    script = (
        f"$disk = Get-Partition -DriveLetter '{drive_letter}' | Get-Disk; "
        "[pscustomobject]@{FriendlyName=$disk.FriendlyName;"
        "SerialNumber=$disk.SerialNumber;BusType=[string]$disk.BusType} "
        "| ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        (powershell, "-NoLogo", "-NoProfile", "-Command", script),
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
        friendly_name = str(payload["FriendlyName"])
        serial_number = str(payload["SerialNumber"])
        bus_type = str(payload["BusType"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("PowerShell returned invalid Windows disk identity") from error
    if bus_type.casefold() != "nvme" or not friendly_name or not serial_number:
        raise RuntimeError("D archive must resolve to an identified NVMe disk")
    identity = re.sub(
        r"[^a-z0-9]+",
        "-",
        f"{bus_type}-{friendly_name}-{serial_number}".casefold(),
    ).strip("-")
    if not identity:
        raise RuntimeError("Windows NVMe identity is empty")
    return identity


def _probe_macos_volume_identity(path: Path) -> VolumeIdentity:
    filesystem_result = subprocess.run(
        ("df", "-P", str(path)),
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in filesystem_result.stdout.splitlines() if line.strip()]
    if len(lines) != 2:
        raise RuntimeError(f"df returned invalid mount metadata for {path}")
    device = lines[1].split(maxsplit=1)[0]
    if not device.startswith("/dev/"):
        raise RuntimeError(f"df returned an invalid device for {path}: {device}")
    completed = subprocess.run(
        ("diskutil", "info", "-plist", device),
        check=True,
        capture_output=True,
    )
    try:
        payload = plistlib.loads(completed.stdout)
    except plistlib.InvalidFileException as error:
        raise RuntimeError(f"diskutil returned invalid volume metadata for {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"diskutil returned invalid volume metadata for {path}")
    device_uuid = payload.get("VolumeUUID") or payload.get("DiskUUID")
    filesystem = payload.get("FilesystemName") or payload.get("FilesystemType")
    if not isinstance(device_uuid, str) or not isinstance(filesystem, str):
        raise RuntimeError(f"volume identity is incomplete for {path}")
    return VolumeIdentity(device_uuid=device_uuid, filesystem=filesystem)


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def verify_campaign_root_locations(
    *,
    repository_root: Path,
    locator: StorageRootLocator,
) -> None:
    """Reject local locator drift before creating or writing any campaign root."""

    del repository_root
    if locator.aliases != ("d_archive", "wsl_staging"):
        raise RuntimeError("Stage 5.2 storage locator must contain only d_archive/wsl_staging")
    staging = locator.resolve("wsl_staging")
    archive = locator.resolve("d_archive")
    staging_path = staging.absolute_path.resolve()
    archive_path = archive.absolute_path.resolve()
    if staging.volume.filesystem.casefold() != "ext4" or staging_path.is_relative_to(
        Path("/mnt")
    ):
        raise RuntimeError("Stage 5.2 wsl_staging must be on WSL2 native ext4")
    if not archive_path.is_relative_to(Path("/mnt/d")):
        raise RuntimeError("Stage 5.2 archive path is forbidden outside the D drive")
    if archive.volume.filesystem.casefold() in {"exfat", "vfat", "fat", "fat32"}:
        raise RuntimeError("Stage 5.2 D archive cannot use ExFAT or removable FAT storage")
    if "usb" in archive.volume.device_uuid.casefold():
        raise RuntimeError("Stage 5.2 D archive cannot use USB storage")


def verify_rolling_campaign_capacity(
    *,
    config: BenchmarkCampaignConfig,
    campaign: CampaignManifest,
    locator: StorageRootLocator,
    batch_id: str,
    phase: str,
    free_space: Callable[[Path], int] = free_bytes,
    volume_probe: Callable[[Path], VolumeIdentity] = probe_volume_identity,
) -> dict[str, object]:
    """Re-probe capacity before dispatch/archive and preserve every reserve."""

    if phase not in {"pre_dispatch", "pre_archive", "post_archive"}:
        raise ValueError("rolling capacity phase is invalid")
    if campaign.run_label != config.run_label or campaign.scope != config.scope:
        raise ValueError("rolling capacity campaign/config identity mismatch")
    batches = {batch.batch_id: batch for batch in campaign.batches}
    current = batches.get(batch_id)
    if current is None:
        raise ValueError("rolling capacity batch is not in the campaign")
    expected_status = {
        "pre_dispatch": "planned",
        "pre_archive": "verified",
        "post_archive": "archived",
    }[phase]
    if current.status != expected_status:
        raise RuntimeError(
            f"rolling capacity {phase} requires {expected_status} batch state"
        )

    aliases = tuple(
        dict.fromkeys((config.staging_root_alias, *config.archive_root_aliases))
    )
    locator.verify_all(volume_probe, aliases)
    roots = {alias: locator.resolve(alias) for alias in aliases}
    free_by_device: dict[str, int] = {}
    for root in roots.values():
        measured = free_space(root.absolute_path)
        if isinstance(measured, bool) or not isinstance(measured, int) or measured < 0:
            raise RuntimeError("rolling capacity free-byte measurement is invalid")
        device = root.volume.device_uuid
        free_by_device[device] = min(free_by_device.get(device, measured), measured)

    staging_device = roots[config.staging_root_alias].volume.device_uuid
    required_by_device: dict[str, int] = {}
    for alias in config.archive_root_aliases:
        root = roots[alias]
        device = root.volume.device_uuid
        reserve = (
            config.external_safety_reserve_bytes
            + config.external_active_workspace_bytes
            if device == staging_device
            else config.internal_safety_reserve_bytes
        )
        required_by_device[device] = max(required_by_device.get(device, 0), reserve)

    current_index = tuple(batch.batch_id for batch in campaign.batches).index(batch_id)
    future = campaign.batches[current_index + 1 :]
    if any(batch.status != "planned" for batch in future):
        raise RuntimeError("rolling capacity future batch states are not planned")
    projected = campaign.batches[current_index:] if phase == "pre_dispatch" else future
    for batch in projected:
        device = roots[batch.archive_root_alias].volume.device_uuid
        required_by_device[device] = (
            required_by_device.get(device, 0) + batch.estimated_bytes
        )

    if phase == "pre_archive":
        if current.actual_bytes is None:
            raise RuntimeError("verified batch has no actual byte count")
        target_device = roots[current.archive_root_alias].volume.device_uuid
        if target_device != staging_device:
            required_by_device[target_device] = (
                required_by_device.get(target_device, 0) + current.actual_bytes
            )
        staging_required = config.external_safety_reserve_bytes
        if future:
            if target_device == staging_device:
                staging_required = (
                    config.external_safety_reserve_bytes
                    + config.external_active_workspace_bytes
                )
            else:
                staging_required = max(
                    staging_required,
                    config.external_safety_reserve_bytes
                    + config.external_active_workspace_bytes
                    - current.actual_bytes,
                )
        required_by_device[staging_device] = max(
            required_by_device.get(staging_device, 0),
            staging_required,
        )
    else:
        staging_reserve = config.external_safety_reserve_bytes
        if phase == "pre_dispatch" or future:
            staging_reserve += config.external_active_workspace_bytes
        required_by_device[staging_device] = max(
            required_by_device.get(staging_device, 0),
            staging_reserve,
        )

    deficits = {
        device: required - free_by_device.get(device, 0)
        for device, required in required_by_device.items()
        if free_by_device.get(device, 0) < required
    }
    if deficits:
        observation = {
            "schema_version": "stage05.2-rolling-capacity-v1",
            "run_label": campaign.run_label,
            "batch_id": batch_id,
            "phase": phase,
            "free_bytes_by_device": dict(sorted(free_by_device.items())),
            "required_bytes_by_device": dict(sorted(required_by_device.items())),
            "deficits_by_device": dict(sorted(deficits.items())),
            "passed": False,
        }
        raise RollingCampaignCapacityError(
            observation=observation,
            message=(
                f"rolling campaign capacity cannot preserve reserves at {phase}: "
                f"{deficits}"
            ),
        )
    return {
        "schema_version": "stage05.2-rolling-capacity-v1",
        "run_label": campaign.run_label,
        "batch_id": batch_id,
        "phase": phase,
        "free_bytes_by_device": dict(sorted(free_by_device.items())),
        "required_bytes_by_device": dict(sorted(required_by_device.items())),
        "passed": True,
    }


class RollingCampaignCapacityError(RuntimeError):
    """A failed reserve check carrying the complete durable observation."""

    def __init__(self, *, observation: Mapping[str, object], message: str) -> None:
        super().__init__(message)
        self.observation = dict(observation)


def campaign_control_paths(run_dir: Path, run_label: str) -> dict[str, Path]:
    control = run_dir / "control"
    return {
        "campaign_manifest": run_dir / "campaign_manifest.json",
        "campaign_plan": control / f"{run_label}_campaign_plan.json",
        "preflight": control / f"{run_label}_campaign_preflight.json",
        "archive_dry_run": control / f"{run_label}_archive_dry_run.json",
        "publication_dry_run": control / f"{run_label}_publication_dry_run.json",
        "failure_state_drill": control / f"{run_label}_failure_state_drill.json",
        "raw_replay_drill": control / f"{run_label}_raw_replay_drill.json",
        "rolling_capacity": control / f"{run_label}_rolling_capacity.json",
        "persistence_attribution": control
        / f"{run_label}_persistence_attribution.json",
    }


def persist_campaign_manifest(run_dir: Path, manifest: CampaignManifest) -> Path:
    path = campaign_control_paths(run_dir, manifest.run_label)["campaign_manifest"]
    atomic_write_signed_json(path, manifest.to_dict())
    loaded = load_campaign_manifest(path)
    if loaded != manifest:
        raise RuntimeError("atomic campaign manifest read-back mismatch")
    return path


class BatchRuntimeMonitor:
    """Continuously sample batch power/load and expose a pool-abort reason."""

    def __init__(
        self,
        config: BenchmarkCampaignConfig,
        *,
        snapshot: Callable[[], MachineSnapshot],
        interval_seconds: float = 1.0,
    ) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("batch runtime sampling interval must be positive")
        self._config = config
        self._snapshot = snapshot
        self._interval_seconds = interval_seconds
        self._samples: list[MachineSnapshot] = []
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("batch runtime monitor was already started")
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="stage052-batch-runtime-monitor",
            daemon=True,
        )
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self._snapshot()
            except BaseException as error:
                with self._lock:
                    self._error = error
                self._stop.set()
                return
            with self._lock:
                self._samples.append(sample)
            if not BatchRuntimeEvidence.from_snapshots((sample,), config=self._config).passed:
                self._stop.set()
                return
            self._stop.wait(self._interval_seconds)

    def abort_reason(self) -> str | None:
        with self._lock:
            if self._error is not None:
                return f"batch runtime sampling failed: {type(self._error).__name__}: {self._error}"
            samples = tuple(self._samples)
        if not samples:
            return None
        evidence = BatchRuntimeEvidence.from_snapshots(samples, config=self._config)
        return None if evidence.passed else evidence.failure_reason

    def stop(self) -> BatchRuntimeEvidence:
        if self._thread is None:
            raise RuntimeError("batch runtime monitor was not started")
        self._stop.set()
        self._thread.join(timeout=max(5.0, self._interval_seconds * 2.0))
        if self._thread.is_alive():
            raise RuntimeError("batch runtime monitor did not stop")
        with self._lock:
            error = self._error
            samples = tuple(self._samples)
        if error is not None:
            raise RuntimeError(
                f"batch runtime sampling failed: {type(error).__name__}: {error}"
            ) from error
        if not samples:
            raise RuntimeError("batch runtime monitor captured no samples")
        return BatchRuntimeEvidence.from_snapshots(samples, config=self._config)


def load_pilot_storage_observations(
    prerequisite_dir: Path,
    *,
    locator: StorageRootLocator,
) -> tuple[PilotStorageObservation, ...]:
    """Re-verify archived G01 bytes and derive G02 next-fit estimates."""

    run_label = prerequisite_dir.name
    paths = campaign_control_paths(prerequisite_dir, run_label)
    manifest = load_campaign_manifest(paths["campaign_manifest"])
    if manifest.scope != "pilot" or manifest.status != "complete":
        raise RuntimeError("G02 requires one complete accepted G01 campaign")
    expected_roots = set(manifest.storage_roots)
    if expected_roots != set(locator.aliases):
        raise RuntimeError("G01/G02 storage root alias set differs")
    locator.verify_all(probe_volume_identity, tuple(sorted(expected_roots)))
    try:
        plan_payload = json.loads(paths["campaign_plan"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("accepted G01 campaign plan is unreadable") from error
    if not isinstance(plan_payload, Mapping) or (
        plan_payload.get("scope") != "pilot"
        or plan_payload.get("shard_count") != 36
        or plan_payload.get("axis_count") != 36
        or plan_payload.get("checkpoint_count") != 144
    ):
        raise RuntimeError("accepted G01 campaign plan geometry is invalid")
    shards = plan_payload.get("shards")
    if not isinstance(shards, list) or len(shards) != 36:
        raise RuntimeError("accepted G01 campaign plan shard list is invalid")
    actual_by_shard: dict[str, int] = {}
    for batch in manifest.batches:
        if batch.status != "archived" or batch.shard_actual_bytes_by_id is None:
            raise RuntimeError("accepted G01 batch is not archived with shard bytes")
        archive_root = locator.resolve(batch.root_alias)
        batch_path = archive_root.absolute_path.joinpath(*Path(batch.logical_path).parts)
        if (
            directory_checksum(batch_path) != batch.checksum_sha256
            or directory_byte_count(batch_path) != batch.actual_bytes
        ):
            raise RuntimeError(f"accepted G01 archived batch checksum failed: {batch.batch_id}")
        overlap = set(actual_by_shard).intersection(batch.shard_actual_bytes_by_id)
        if overlap:
            raise RuntimeError(f"accepted G01 shard bytes are duplicate: {sorted(overlap)}")
        actual_by_shard.update(batch.shard_actual_bytes_by_id)
    observations: list[PilotStorageObservation] = []
    for raw in shards:
        if not isinstance(raw, Mapping):
            raise RuntimeError("accepted G01 campaign shard is invalid")
        shard_id = raw.get("shard_id")
        family = raw.get("family")
        customer_count = raw.get("customer_count")
        budgets = raw.get("budgets_seconds")
        if (
            not isinstance(shard_id, str)
            or not isinstance(family, str)
            or isinstance(customer_count, bool)
            or not isinstance(customer_count, int)
            or budgets != [30]
            or shard_id not in actual_by_shard
        ):
            raise RuntimeError("accepted G01 campaign shard storage identity is invalid")
        observations.append(
            PilotStorageObservation(
                family=family,
                customer_count=customer_count,
                budget_seconds=30,
                compressed_bytes=actual_by_shard[shard_id],
            )
        )
    if len(observations) != 36:
        raise RuntimeError("accepted G01 storage observations are incomplete")
    return tuple(observations)


def archive_verified_batch(
    *,
    batch: BatchManifest,
    locator: StorageRootLocator,
) -> BatchManifest:
    """Archive a verified batch; retained for non-campaign dry-run callers."""

    return archive_verified_batch_with_evidence(batch=batch, locator=locator).batch


@dataclass(frozen=True, slots=True)
class ArchivedBatchEvidence:
    """Archived state plus the separately timed final manifest write."""

    batch: BatchManifest
    manifest_sha256: str
    state_write_interval: PersistenceInterval
    manifest_path: Path


class ArchivedBatchStateWriteError(RuntimeError):
    """Archive transfer completed, but its final signed state could not be sealed."""

    def __init__(
        self,
        *,
        batch: BatchManifest,
        destination: Path,
        cause: Exception,
    ) -> None:
        super().__init__(
            "archived batch transfer completed but final state write failed: "
            f"{type(cause).__name__}: {cause}"
        )
        self.batch = batch
        self.destination = destination


def archive_verified_batch_with_evidence(
    *,
    batch: BatchManifest,
    locator: StorageRootLocator,
    _state_writer: Callable[
        [Path, Mapping[str, object]], tuple[Path, Path]
    ]
    | None = None,
) -> ArchivedBatchEvidence:
    state_writer = atomic_write_signed_json if _state_writer is None else _state_writer
    try:
        archived = BatchArchiver(locator).archive(batch)
    except ArchiveTransferCompletedError as error:
        raise ArchivedBatchStateWriteError(
            batch=error.batch,
            destination=error.destination,
            cause=error,
        ) from error
    destination_root = locator.resolve(archived.root_alias)
    destination = destination_root.absolute_path.joinpath(*Path(archived.logical_path).parts)
    started_ns = time.monotonic_ns()
    try:
        manifest_path, _ = state_writer(
            destination / "batch_manifest.json",
            archived.to_dict(),
        )
        completed_ns = time.monotonic_ns()
        if (
            directory_checksum(destination) != archived.checksum_sha256
            or directory_byte_count(destination) != archived.actual_bytes
        ):
            raise RuntimeError("archived batch changed after manifest state update")
    except Exception as error:
        raise ArchivedBatchStateWriteError(
            batch=archived,
            destination=destination,
            cause=error,
        ) from error
    return ArchivedBatchEvidence(
        batch=archived,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        state_write_interval=PersistenceInterval(
            label="archived_batch_manifest_write",
            started_ns=started_ns,
            completed_ns=completed_ns,
        ),
        manifest_path=manifest_path,
    )


class WindowsWslMachineSnapshotSource:
    """Sample Windows power state plus WSL2 load and unrelated CPU usage."""

    def __init__(self) -> None:
        self._native_status: WindowsWslPowerStatus | None = None

    def refresh_native_status(self) -> WindowsWslPowerStatus:
        """Refresh native Windows state outside the measured runtime interval."""

        observed = read_windows_wsl_power_status()
        self._native_status = observed
        return observed

    def verify_native_status_unchanged(self) -> dict[str, object]:
        """Verify stable native invariants and return both boundary observations."""

        expected = self._native_status
        if expected is None:
            raise RuntimeError("Windows native power status was not sampled at preflight")
        observed = read_windows_wsl_power_status()
        expected_invariants = (
            expected.ac_online,
            expected.battery_saver,
            expected.active_power_scheme,
        )
        observed_invariants = (
            observed.ac_online,
            observed.battery_saver,
            observed.active_power_scheme,
        )
        if observed_invariants != expected_invariants:
            raise RuntimeError(
                "Windows native power state changed during the benchmark batch: "
                f"expected={expected!r} observed={observed!r}"
            )
        return {
            "schema_version": "stage05.2-native-power-boundary-v1",
            "before": _windows_power_status_to_dict(expected),
            "after": _windows_power_status_to_dict(observed),
            "stable_invariants": [
                "ac_online",
                "battery_saver",
                "active_power_scheme",
            ],
            "invariants_unchanged": True,
        }

    def __call__(self) -> MachineSnapshot:
        power = self._native_status
        if power is None:
            power = self.refresh_native_status()
        ac_online = read_wsl_ac_power_online()
        return MachineSnapshot(
            power_source="AC Power" if ac_online else "Battery Power",
            low_power_mode_enabled=power.battery_saver,
            load1=float(os.getloadavg()[0]),
            unrelated_process_average_cores=0.0,
            sampled_at_seconds=time.monotonic(),
            unrelated_process_cpu_seconds=_unrelated_user_cpu_seconds(),
        )


def _parse_process_cpu_time(value: str) -> float:
    day_parts = value.split("-", maxsplit=1)
    days = 0
    clock = value
    if len(day_parts) == 2:
        try:
            days = int(day_parts[0])
        except ValueError as error:
            raise RuntimeError(f"invalid process CPU time: {value}") from error
        clock = day_parts[1]
    parts = clock.split(":")
    try:
        if len(parts) == 2:
            hours = 0
            minutes = int(parts[0])
            seconds = float(parts[1])
        elif len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
        else:
            raise ValueError
    except ValueError as error:
        raise RuntimeError(f"invalid process CPU time: {value}") from error
    return days * 86_400.0 + hours * 3_600.0 + minutes * 60.0 + seconds


def _windows_power_status_to_dict(status: WindowsWslPowerStatus) -> dict[str, object]:
    return {
        "ac_online": status.ac_online,
        "battery_saver": status.battery_saver,
        "battery_life_percent": status.battery_life_percent,
        "battery_flag": status.battery_flag,
        "active_power_scheme": status.active_power_scheme,
    }


def _unrelated_user_cpu_seconds() -> dict[int, float]:
    completed = subprocess.run(
        ("ps", "-axo", "pid=,ppid=,uid=,time="),
        check=True,
        capture_output=True,
        text=True,
    )
    current_pid = os.getpid()
    current_uid = os.getuid()
    records: list[tuple[int, int, int, float]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise RuntimeError(f"invalid non-empty ps process row: {line!r}")
        try:
            records.append(
                (
                    int(fields[0]),
                    int(fields[1]),
                    int(fields[2]),
                    _parse_process_cpu_time(fields[3]),
                )
            )
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"cannot parse ps process row: {line!r}") from error
    related = {current_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent_pid, _, _ in records:
            if parent_pid in related and pid not in related:
                related.add(pid)
                changed = True
    return {
        pid: cpu_seconds
        for pid, _, uid, cpu_seconds in records
        if uid == current_uid and pid not in related
    }
