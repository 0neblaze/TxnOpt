"""Five-mode Stage 5.2 native-architecture comparison runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import re
import resource
import shutil
import subprocess
import time
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol, TypedDict, cast
from urllib.parse import unquote, urlparse

import numpy as np

from evrptw.alns import ALNSResult, solve_alns
from evrptw.cache_incremental import CacheIncrementalConfig
from evrptw.candidate_control import CandidateControlConfig
from evrptw.candidate_transaction import NativeCandidateTransactionConfig
from evrptw.exact_deadline import ExactDeadlineConfig
from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES
from evrptw.measurement import CheapScreeningConfig, MeasurementConfig
from evrptw.models import Instance, NodeType
from evrptw.native_execution import Stage052NativeExecutionConfig
from evrptw.native_kernels import NativeKernelConfig
from evrptw.native_scheduler import NativeHostScheduler
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.repository import repository_root
from evrptw.runtime_envelope import ProcessTreeMonitor
from evrptw.stage04 import Stage04Config
from evrptw.stage052_atomic import publish_no_replace
from evrptw.stage052_continuity_lease import require_owned
from evrptw.stage052_performance import (
    ExecutionTopology,
    FrozenPerformanceProfile,
    FrozenRuntimeBinding,
    load_frozen_profile,
)
from evrptw.stage052_performance import (
    performance_topology_key as frozen_performance_topology_key,
)
from evrptw.stage052_semantic_journal import (
    PARALLEL_BATCH_NON_CANONICAL_FIELDS,
    SEMANTIC_TRAJECTORY_IMPLEMENTATION_STATUSES,
    semantic_bundle_path,
    write_semantic_journal,
)
from evrptw.validation import validate_routes
from evrptw.warm_start import (
    WarmStartValidationConfig,
    canonical_customer_sequences_sha256,
)
from tools.native_build_attestation import (
    committed_source_attestation,
    committed_wheel_project_entry_sha256,
    validate_scheduler_build_attestation,
)

SCHEMA_VERSION = "stage05.2-native-architecture-comparison-v9"
PREVIOUS_COMPARISON_SCHEMA_VERSION = "stage05.2-native-architecture-comparison-v7"
AXIS_PERSISTENCE_RECEIPT_SCHEMA_VERSION = "stage05.2-axis-persistence-receipt-v1"
SEEDS = (2014, 2015, 2016)
PAIRED_INSTANCES = ("c101C5", "c101_21", "r101_21", "rc101_21")
AXIS_NAMES = ("fixed_work", "wall_clock_30")
SHARD_PROCESSES = 6
THREADS_PER_SHARD = 4
TOTAL_COMPUTE_THREADS = 24
WORKLOAD_CLASSES = ("c5", "100-customer")
WARM_START_SCHEMA_VERSION = "stage05.2-native-architecture-warm-start-v2"
CALIBRATION_REVIEW_SCHEMA_VERSION = (
    "stage05.2-native-architecture-performance-calibration-review-v1"
)
CALIBRATION_REVIEW_QUALIFICATION = "QUALIFIED_FOR_ATTEMPT08"
PAIRED_REVIEW_MANIFEST_SCHEMA_VERSION = "stage05.2-native-architecture-review-manifest-v2"
PAIRED_REVIEW_EXECUTION_SCHEMA_VERSION = "experiment-review-execution-v1"
PAIRED_REVIEWER_MODULE_NAME = "evrptw.experiments.stage052_native_architecture_review"
NATIVE_ARCHITECTURE_CAPABILITY_NAMES = (
    "host_candidate_transaction_scheduler",
    "whole_search_gil_released",
    "single_host_24_thread_compute_pool",
    "runtime_semantic_event_journal",
)

WarmStartIdentity = tuple[str, int]
WarmStartRecord = tuple[tuple[tuple[str, ...], ...], dict[str, object]]


class NativeBuildAttestation(Protocol):
    __build_git_revision__: str
    __build_git_tree__: str
    __build_source_manifest_sha256__: str
    __build_tracked_file_count__: int
    __build_source_dirty__: bool
    __build_development_override__: bool
    __build_cpp_source_kind__: str
    __build_source_attestation_version__: int
    __build_performance_profile__: str
    __build_compiler_id__: str
    __build_compiler_version__: str
    __build_interprocedural_optimization__: bool
    __build_host_native__: bool


class WheelReceipt(TypedDict):
    wheel_path: str
    wheel_sha256: str
    direct_url_path: str
    package_path: str
    native_path: str
    native_sha256: str
    scheduler_path: str
    scheduler_sha256: str
    build_git_revision: str
    build_git_tree: str
    build_source_manifest_sha256: str
    build_tracked_file_count: int
    build_source_dirty: bool
    build_development_override: bool
    build_cpp_source_kind: str
    build_source_attestation_version: int
    build_performance_profile: str
    build_compiler_id: str
    build_compiler_version: str
    build_interprocedural_optimization: bool
    build_host_native: bool
    wheel_entry_sha256: dict[str, str]
    native_wheel_entry: str
    scheduler_wheel_entry: str
    runner_wheel_entry: str
    scheduler_build_attestation: dict[str, object]


class ArchitectureMode(StrEnum):
    CURRENT_STAGE052 = "current_stage052"
    PYTHON_CANDIDATE_CONTROL = "python_candidate_control"
    PER_SOLVE_RUNTIME = "per_solve_runtime"
    FULL_NATIVE_ALNS = "full_native_alns"
    HOST_SCHEDULER = "host_scheduler"


MODES = tuple(ArchitectureMode)


class ArchitectureAxisExecutionFailed(RuntimeError):
    """Pickle-safe worker failure carrying the immutable failed-axis receipt."""

    def __init__(self, axis_path: str, error_type: str, error: str) -> None:
        self.axis_path = axis_path
        self.error_type = error_type
        self.error = error
        super().__init__(axis_path, error_type, error)

    def __str__(self) -> str:
        return f"{self.error_type}: {self.error} (failed axis: {self.axis_path})"


@dataclass(frozen=True, slots=True)
class ArchitectureAxisTask:
    scope: str
    repeat: int
    axis: str
    instance_name: str
    seed: int
    benchmark_dir: Path
    output_root: Path
    run_labels: dict[str, str]
    scheduler_socket_path: str
    wheel_sha256: str
    native_sha256: str
    scheduler_sha256: str
    revision: str
    initial_customer_sequences: tuple[tuple[str, ...], ...]
    initial_solution_provenance: dict[str, object]
    scheduler_process_id: int | None = None
    performance_profile_sha256: str | None = None
    topology_key: str | None = None
    axis_cpu_ids: tuple[int, ...] = ()
    shard_processes: int = SHARD_PROCESSES
    threads_per_shard: int = THREADS_PER_SHARD
    total_compute_threads: int = TOTAL_COMPUTE_THREADS
    scheduler_cpu_ids: tuple[int, ...] = ()
    scheduler_request_threads: int = SHARD_PROCESSES
    allow_affinity_overlap: bool = False
    fixed_work_exact_calls: int = 100
    fixed_work_max_iterations: int = 1000
    fixed_work_watchdog_seconds: float = 120.0
    exact_batch_size: int = 128


def workload_class_for_instance(instance_name: str) -> str:
    return "100-customer" if instance_name.endswith("_21") else "c5"


def performance_family_for_instance(instance_name: str) -> str:
    lowered = instance_name.lower()
    if not lowered.endswith("_21"):
        return "C5"
    if lowered.startswith("rc"):
        return "RC"
    if lowered.startswith("r"):
        return "R"
    return "C"


def performance_topology_key(
    mode: ArchitectureMode,
    workload_class: str,
) -> str:
    return frozen_performance_topology_key(mode.value, workload_class)


def _load_signed_performance_profile(path: Path) -> FrozenPerformanceProfile:
    data = path.read_bytes()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError("frozen performance profile SHA-256 sidecar is missing")
    declared = sidecar.read_text(encoding="ascii").strip()
    observed = hashlib.sha256(data).hexdigest()
    if declared != observed:
        raise RuntimeError("frozen performance profile byte hash mismatch")
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise RuntimeError("frozen performance profile is not valid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("frozen performance profile root must be an object")
    return load_frozen_profile(payload)


def _load_qualified_calibration_review(
    path: Path,
    *,
    review_execution_path: Path,
    calibration_run_label: str,
    revision: str,
    git_tree: str,
    source_manifest_sha256: str,
    wheel_sha256: str,
    native_sha256: str,
    scheduler_sha256: str,
    performance_profile_path: Path,
    performance_profile_file_sha256: str,
    performance_profile_sha256: str,
    selected_build_profile: str,
) -> dict[str, object]:
    if (
        re.fullmatch(
            r"stage05\.2_native_architecture_performance_calibration_(?:attempt|rerun)[0-9]{2}",
            calibration_run_label,
        )
        is None
    ):
        raise RuntimeError("calibration review run label is not canonical")
    data = path.read_bytes()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError("calibration review SHA-256 sidecar is missing")
    declared = sidecar.read_text(encoding="ascii").strip()
    observed = hashlib.sha256(data).hexdigest()
    if declared != observed:
        raise RuntimeError("calibration review byte hash mismatch")
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise RuntimeError("calibration review is not valid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("calibration review root must be an object")
    if (
        payload.get("schema_version") != CALIBRATION_REVIEW_SCHEMA_VERSION
        or payload.get("status") != "qualified"
        or payload.get("qualification") != CALIBRATION_REVIEW_QUALIFICATION
        or payload.get("calibration_run_label") != calibration_run_label
        or payload.get("repository_revision") != revision
        or payload.get("git_tree") != git_tree
        or payload.get("source_manifest_sha256") != source_manifest_sha256
        or payload.get("selected_build_profile") != selected_build_profile
        or payload.get("wheel_sha256") != wheel_sha256
        or payload.get("native_sha256") != native_sha256
        or payload.get("scheduler_sha256") != scheduler_sha256
        or payload.get("profile_sha256") != performance_profile_file_sha256
        or payload.get("profile_canonical_sha256") != performance_profile_sha256
        or payload.get("rederived_profile_canonical_sha256") != performance_profile_sha256
        or payload.get("fixed_work_semantics_identical") is not True
        or payload.get("validator_objective_replay_passed") is not True
        or payload.get("selector_recomputation_passed") is not True
        or payload.get("queue_full_count") != 0
        or payload.get("rejected_count") != 0
        or payload.get("swap_used") is not False
        or payload.get("formal_started") is not False
        or payload.get("cuda_started") is not False
        or payload.get("attempt08_started") is not False
    ):
        raise RuntimeError("calibration review is not a qualified independent replay")
    for field in (
        "observation_count",
        "raw_axis_replay_count",
        "resource_summary_replay_count",
    ):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"calibration review {field} is invalid")
    reviewer_source = payload.get("reviewer_source_sha256")
    if (
        not isinstance(reviewer_source, str)
        or re.fullmatch(r"[0-9a-f]{64}", reviewer_source) is None
    ):
        raise RuntimeError("calibration reviewer source identity is invalid")
    execution_data = review_execution_path.read_bytes()
    execution_sidecar = review_execution_path.with_suffix(review_execution_path.suffix + ".sha256")
    execution_sha256 = hashlib.sha256(execution_data).hexdigest()
    if (
        not execution_sidecar.is_file()
        or execution_sidecar.read_text(encoding="ascii").strip() != execution_sha256
    ):
        raise RuntimeError("calibration review execution receipt is unsigned")
    try:
        execution = json.loads(execution_data)
    except json.JSONDecodeError as error:
        raise RuntimeError("calibration review execution receipt is invalid JSON") from error
    command = execution.get("command") if isinstance(execution, dict) else None
    if (
        not isinstance(execution, dict)
        or execution.get("schema_version") != "experiment-review-execution-v1"
        or execution.get("run_label") != calibration_run_label
        or execution.get("status") != "completed"
        or execution.get("finalized") is not True
        or execution.get("exit_code") != 0
        or execution.get("reviewer_module_name")
        != "evrptw.experiments.stage052_performance_calibration_review"
        or execution.get("reviewer_installed_distribution_digest") != reviewer_source
        or execution.get("raw_manifest_sha256_before") != payload.get("calibration_manifest_sha256")
        or execution.get("raw_manifest_sha256_after") != payload.get("calibration_manifest_sha256")
        or execution.get("raw_manifest_unchanged") is not True
        or execution.get("review_manifest_sha256") != observed
        or not isinstance(command, list)
        or len(command) < 3
        or command[1:3] != ["-m", "evrptw.experiments.stage052_performance_calibration_review"]
    ):
        raise RuntimeError("calibration review execution receipt does not replay")
    if payload.get("storage_alias") != "stage052-performance-calibration-run":
        raise RuntimeError("calibration review storage alias is invalid")
    run_roots = [
        parent for parent in path.resolve().parents if parent.name == calibration_run_label
    ]
    if len(run_roots) != 1:
        raise RuntimeError("calibration review run namespace is ambiguous")
    calibration_run_dir = run_roots[0]

    def resolve_review_reference(field: str) -> Path:
        raw = payload.get(field)
        if not isinstance(raw, str) or not raw:
            raise RuntimeError(f"calibration review {field} is invalid")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts or "\\" in raw:
            raise RuntimeError(f"calibration review {field} is not portable")
        resolved = (calibration_run_dir / Path(*relative.parts)).resolve()
        try:
            resolved.relative_to(calibration_run_dir)
        except ValueError as error:
            raise RuntimeError(f"calibration review {field} escapes its run") from error
        return resolved

    recorded_profile_path = resolve_review_reference("profile_relative_path")
    if recorded_profile_path.resolve() != performance_profile_path.resolve():
        raise RuntimeError("calibration review references a different frozen profile")
    receipt_sha256 = payload.get("calibration_receipt_sha256")
    if not isinstance(receipt_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", receipt_sha256) is None:
        raise RuntimeError("calibration review receipt identity is invalid")
    receipt_path = resolve_review_reference("calibration_receipt_relative_path")
    if receipt_path.parent.name != calibration_run_label or not receipt_path.is_file():
        raise RuntimeError("calibration review receipt path is outside its run namespace")
    receipt_data = receipt_path.read_bytes()
    receipt_sidecar = receipt_path.with_suffix(receipt_path.suffix + ".sha256")
    if (
        hashlib.sha256(receipt_data).hexdigest() != receipt_sha256
        or not receipt_sidecar.is_file()
        or receipt_sidecar.read_text(encoding="ascii").strip() != receipt_sha256
    ):
        raise RuntimeError("calibration review receipt dependency changed")
    return {
        "path": str(path.resolve()),
        "file_sha256": observed,
        "review_execution_path": str(review_execution_path.resolve()),
        "review_execution_sha256": execution_sha256,
        "calibration_run_label": calibration_run_label,
        "calibration_receipt_path": str(receipt_path),
        "calibration_receipt_sha256": receipt_sha256,
        "profile_sha256": performance_profile_sha256,
        "selected_build_profile": selected_build_profile,
        "raw_axis_replay_count": payload["raw_axis_replay_count"],
        "resource_summary_replay_count": payload["resource_summary_replay_count"],
        "qualification": CALIBRATION_REVIEW_QUALIFICATION,
    }


def _recompute_paired_review_inventory(
    results_root: Path,
    paired_attempt: int,
) -> dict[str, object]:
    # Local import avoids a module cycle: the independent reviewer imports the
    # runner's frozen scope constants, while Pilot calls this only at runtime.
    from evrptw.experiments.stage052_native_architecture_review import (
        _raw_axis_inventory,
        load_records,
    )

    return _raw_axis_inventory(
        load_records("paired", attempt=paired_attempt, results_root=results_root)
    )


def _load_qualified_paired_review(
    path: Path,
    *,
    review_execution_path: Path,
    paired_results_root: Path,
    paired_attempt: int,
    revision: str,
    wheel_sha256: str,
    native_sha256: str,
    scheduler_sha256: str,
    performance_profile_sha256: str,
) -> dict[str, object]:
    data = path.read_bytes()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError("paired review SHA-256 sidecar is missing")
    declared = sidecar.read_text(encoding="ascii").strip()
    observed = hashlib.sha256(data).hexdigest()
    if declared != observed:
        raise RuntimeError("paired review byte hash mismatch")
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise RuntimeError("paired review is not valid JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError("paired review root must be an object")
    expected_labels = sorted(run_labels_for_scope("paired", paired_attempt).values())
    identity = payload.get("producer_identity")
    reviewer = payload.get("reviewer_provenance")
    recorded_inventory = payload.get("raw_axis_inventory")
    if (
        payload.get("scope") != "paired"
        or payload.get("attempt") != paired_attempt
        or payload.get("axis_count") != expected_axis_count("paired")
        or payload.get("axis_replay_passed") is not True
        or payload.get("semantic_gates_passed") is not True
        or payload.get("qualification_passed") is not True
        or payload.get("review_status") != "COMPARISON_COMPLETE_QUALIFIED"
        or payload.get("replay_failures") != []
        or not isinstance(identity, dict)
        or not isinstance(reviewer, dict)
        or not isinstance(recorded_inventory, dict)
    ):
        raise RuntimeError("paired review is not a qualified 360/360 independent replay")
    expected_identity: dict[str, object] = {
        "repository_revisions": [revision],
        "wheel_sha256": [wheel_sha256],
        "native_sha256": [native_sha256],
        "scheduler_sha256": [scheduler_sha256],
        "performance_profile_sha256": [performance_profile_sha256],
        "run_labels": expected_labels,
    }
    for field, value in expected_identity.items():
        if identity.get(field) != value:
            raise RuntimeError(f"paired review producer identity mismatch: {field}")
    reviewer_source_path = Path(__file__).with_name("stage052_native_architecture_review.py")
    reviewer_source_sha256 = _sha256_path(reviewer_source_path)
    if (
        reviewer.get("repository_revision") != revision
        or reviewer.get("source_path")
        != "src/evrptw/experiments/stage052_native_architecture_review.py"
        or reviewer.get("source_sha256") != reviewer_source_sha256
    ):
        raise RuntimeError("paired review provenance is not bound to the campaign revision")
    recomputed_inventory = _recompute_paired_review_inventory(
        paired_results_root,
        paired_attempt,
    )
    if recorded_inventory != recomputed_inventory:
        raise RuntimeError("paired review raw inventory does not match the 360 current axes")
    if recorded_inventory.get("axis_count") != expected_axis_count("paired") or not isinstance(
        recorded_inventory.get("tree_sha256"), str
    ):
        raise RuntimeError("paired review raw inventory is incomplete")
    manifest_path = path.with_name(path.stem + "_manifest.json")
    manifest_data = manifest_path.read_bytes()
    manifest_sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    manifest_sha256 = hashlib.sha256(manifest_data).hexdigest()
    if (
        not manifest_sidecar.is_file()
        or manifest_sidecar.read_text(encoding="ascii").strip() != manifest_sha256
    ):
        raise RuntimeError("paired review manifest is unsigned")
    try:
        manifest = json.loads(manifest_data)
    except json.JSONDecodeError as error:
        raise RuntimeError("paired review manifest is invalid JSON") from error
    if not isinstance(manifest, dict):
        raise RuntimeError("paired review manifest root must be an object")
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != PAIRED_REVIEW_MANIFEST_SCHEMA_VERSION
        or manifest.get("scope") != "paired"
        or manifest.get("attempt") != paired_attempt
        or manifest.get("axis_count") != expected_axis_count("paired")
        or manifest.get("status") != "COMPARISON_COMPLETE_QUALIFIED"
        or manifest.get("reviewer_module_name") != PAIRED_REVIEWER_MODULE_NAME
        or manifest.get("reviewer_provenance") != reviewer
        or manifest.get("producer_identity") != identity
        or manifest.get("raw_axis_inventory") != recorded_inventory
        or not isinstance(files, dict)
        or files.get(path.name) != observed
    ):
        raise RuntimeError("paired review manifest does not bind the qualified replay")
    for file_name, file_sha256 in files.items():
        if (
            not isinstance(file_name, str)
            or Path(file_name).name != file_name
            or not isinstance(file_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", file_sha256) is None
        ):
            raise RuntimeError("paired review manifest file inventory is invalid")
        dependency = manifest_path.parent / file_name
        if not dependency.is_file() or _sha256_path(dependency) != file_sha256:
            raise RuntimeError("paired review manifest dependency changed")
    execution_data = review_execution_path.read_bytes()
    execution_sidecar = review_execution_path.with_suffix(review_execution_path.suffix + ".sha256")
    execution_sha256 = hashlib.sha256(execution_data).hexdigest()
    if (
        not execution_sidecar.is_file()
        or execution_sidecar.read_text(encoding="ascii").strip() != execution_sha256
    ):
        raise RuntimeError("paired review execution receipt is unsigned")
    try:
        execution = json.loads(execution_data)
    except json.JSONDecodeError as error:
        raise RuntimeError("paired review execution receipt is invalid JSON") from error
    command = execution.get("command") if isinstance(execution, dict) else None
    raw_tree_sha256 = recorded_inventory["tree_sha256"]
    if (
        not isinstance(execution, dict)
        or execution.get("schema_version") != PAIRED_REVIEW_EXECUTION_SCHEMA_VERSION
        or execution.get("run_label")
        != f"stage05.2_native_architecture_paired_attempt{paired_attempt:02d}_review"
        or execution.get("status") != "completed"
        or execution.get("finalized") is not True
        or execution.get("exit_code") != 0
        or execution.get("reviewer_module_name") != PAIRED_REVIEWER_MODULE_NAME
        or execution.get("reviewer_installed_distribution_digest") != reviewer_source_sha256
        or execution.get("raw_manifest_sha256_before") != raw_tree_sha256
        or execution.get("raw_manifest_sha256_after") != raw_tree_sha256
        or execution.get("raw_manifest_unchanged") is not True
        or execution.get("raw_axis_count") != expected_axis_count("paired")
        or execution.get("review_manifest_sha256") != manifest_sha256
        or execution.get("review_json_sha256") != observed
        or not isinstance(command, list)
        or len(command) < 3
        or command[1:3] != ["-m", PAIRED_REVIEWER_MODULE_NAME]
    ):
        raise RuntimeError("paired review execution receipt does not replay")
    return {
        "path": str(path.resolve()),
        "file_sha256": observed,
        "review_manifest_path": str(manifest_path.resolve()),
        "review_manifest_sha256": manifest_sha256,
        "review_execution_path": str(review_execution_path.resolve()),
        "review_execution_sha256": execution_sha256,
        "raw_axis_inventory": recorded_inventory,
        "paired_attempt": paired_attempt,
        "review_schema_version": payload.get("schema_version"),
        "reviewer_provenance": reviewer,
        "producer_identity": identity,
        "axis_count": payload["axis_count"],
        "qualification_passed": True,
    }


def _performance_identity_blocks(
    plan: tuple[ArchitectureAxisTask, ...],
) -> tuple[tuple[ArchitectureAxisTask, ...], ...]:
    grouped: dict[tuple[int, str, str, str], list[ArchitectureAxisTask]] = {}
    for task in plan:
        key = (
            task.repeat,
            task.axis,
            workload_class_for_instance(task.instance_name),
            performance_family_for_instance(task.instance_name),
        )
        grouped.setdefault(key, []).append(task)
    return tuple(tuple(tasks) for tasks in grouped.values())


def _assign_performance_topology(
    task: ArchitectureAxisTask,
    topology: ExecutionTopology,
    *,
    shard_index: int,
    profile_sha256: str,
    topology_key: str,
) -> ArchitectureAxisTask:
    configured_cpu_ids = topology.shards[shard_index]
    axis_cpu_ids = (
        topology.cpu_ids if topology.affinity_policy == "free_scheduler" else configured_cpu_ids
    )
    return replace(
        task,
        performance_profile_sha256=profile_sha256,
        topology_key=topology_key,
        axis_cpu_ids=axis_cpu_ids,
        shard_processes=topology.shard_count,
        threads_per_shard=len(configured_cpu_ids),
        total_compute_threads=len(topology.cpu_ids),
        scheduler_cpu_ids=topology.scheduler_cpu_ids,
        scheduler_request_threads=topology.request_threads,
        allow_affinity_overlap=topology.allow_affinity_overlap,
    )


def _mode_task_batches(
    tasks: tuple[ArchitectureAxisTask, ...],
    topology: ExecutionTopology,
    *,
    profile_sha256: str,
    topology_key: str,
) -> tuple[tuple[ArchitectureAxisTask, ...], ...]:
    return tuple(
        tuple(
            _assign_performance_topology(
                task,
                topology,
                shard_index=index,
                profile_sha256=profile_sha256,
                topology_key=topology_key,
            )
            for index, task in enumerate(tasks[offset : offset + topology.shard_count])
        )
        for offset in range(0, len(tasks), topology.shard_count)
    )


def run_labels_for_scope(scope: str, attempt: int) -> dict[str, str]:
    if scope not in {"paired", "pilot"} or attempt <= 0:
        raise ValueError("native architecture scope/attempt is invalid")
    return {
        mode.value: (f"stage05.2_native_architecture_{mode.value}_{scope}_attempt{attempt:02d}")
        for mode in MODES
    }


def build_axis_plan(
    scope: str,
    *,
    attempt: int,
    benchmark_dir: Path,
    output_root: Path,
    scheduler_socket_path: str,
    wheel_sha256: str,
    native_sha256: str,
    scheduler_sha256: str,
    revision: str,
    warm_starts: dict[WarmStartIdentity, WarmStartRecord],
) -> tuple[ArchitectureAxisTask, ...]:
    instances: tuple[str, ...]
    axes: tuple[str, ...]
    if scope == "paired":
        instances = PAIRED_INSTANCES
        repeats = range(3)
        axes = AXIS_NAMES
    elif scope == "pilot":
        instances = tuple(FORMAL_INSTANCES)
        repeats = range(1)
        axes = ("wall_clock_30",)
    else:
        raise ValueError("scope must be paired or pilot")
    labels = run_labels_for_scope(scope, attempt)
    expected_identities = {(instance_name, seed) for instance_name in instances for seed in SEEDS}
    if set(warm_starts) != expected_identities:
        missing = sorted(expected_identities - set(warm_starts))
        extra = sorted(set(warm_starts) - expected_identities)
        raise ValueError(f"warm-start identity set mismatch: missing={missing}, extra={extra}")
    return tuple(
        ArchitectureAxisTask(
            scope=scope,
            repeat=repeat,
            axis=axis,
            instance_name=instance_name,
            seed=seed,
            benchmark_dir=benchmark_dir,
            output_root=output_root,
            run_labels=labels,
            scheduler_socket_path=scheduler_socket_path,
            wheel_sha256=wheel_sha256,
            native_sha256=native_sha256,
            scheduler_sha256=scheduler_sha256,
            revision=revision,
            initial_customer_sequences=warm_starts[(instance_name, seed)][0],
            initial_solution_provenance=dict(warm_starts[(instance_name, seed)][1]),
        )
        for repeat in repeats
        for axis in axes
        for instance_name in instances
        for seed in SEEDS
    )


def rotated_modes(task: ArchitectureAxisTask) -> tuple[ArchitectureMode, ...]:
    instance_order = PAIRED_INSTANCES if task.scope == "paired" else tuple(FORMAL_INSTANCES)
    axis_order = AXIS_NAMES if task.scope == "paired" else ("wall_clock_30",)
    rotation = (
        task.repeat * len(axis_order) * len(instance_order) * len(SEEDS)
        + axis_order.index(task.axis) * len(instance_order) * len(SEEDS)
        + instance_order.index(task.instance_name) * len(SEEDS)
        + SEEDS.index(task.seed)
    ) % len(MODES)
    return MODES[rotation:] + MODES[:rotation]


def expected_axis_count(scope: str) -> int:
    if scope == "paired":
        return len(PAIRED_INSTANCES) * len(SEEDS) * 3 * len(AXIS_NAMES) * len(MODES)
    if scope == "pilot":
        return len(FORMAL_INSTANCES) * len(SEEDS) * len(MODES)
    raise ValueError("scope must be paired or pilot")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_warm_start_bundle(
    path: Path,
    *,
    benchmark_dir: Path,
) -> dict[WarmStartIdentity, WarmStartRecord]:
    """Verify and decode one immutable cross-mode warm-start input bundle."""

    if not path.is_file():
        raise FileNotFoundError(f"warm-start bundle does not exist: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError("warm-start bundle SHA-256 sidecar is missing")
    bundle_sha256 = _sha256_path(path)
    expected_sha256 = sidecar.read_text(encoding="ascii").strip()
    if expected_sha256 != bundle_sha256:
        raise RuntimeError("warm-start bundle SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        WARM_START_SCHEMA_VERSION
    ):
        raise RuntimeError("warm-start bundle schema is invalid")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise RuntimeError("warm-start bundle records are missing")
    output: dict[WarmStartIdentity, WarmStartRecord] = {}
    instance_cache: dict[str, Instance] = {}
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise RuntimeError("warm-start bundle contains a non-object record")
        instance_name = raw.get("instance")
        seed = raw.get("seed")
        routes = raw.get("customer_sequences")
        if (
            not isinstance(instance_name, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(routes, list)
            or not routes
        ):
            raise RuntimeError("warm-start bundle record identity/routes are invalid")
        identity = (instance_name, seed)
        if identity in output:
            raise RuntimeError(f"duplicate warm-start identity: {identity}")
        instance = instance_cache.get(instance_name)
        if instance is None:
            instance = parse_schneider(benchmark_dir / f"{instance_name}.txt")
            instance_cache[instance_name] = instance
        sequences: list[tuple[str, ...]] = []
        for route in routes:
            if (
                not isinstance(route, list)
                or not route
                or not all(isinstance(name, str) for name in route)
            ):
                raise RuntimeError(f"warm-start route is invalid for {identity}")
            sequences.append(tuple(cast(str, name) for name in route))
        expected_customers = sorted(customer.name for customer in instance.customers)
        supplied_customers = sorted(name for route in sequences for name in route)
        if supplied_customers != expected_customers:
            raise RuntimeError(f"warm-start customer coverage mismatch for {identity}")
        source_sha256 = raw.get("source_solution_sha256")
        if (
            not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)
        ):
            raise RuntimeError(f"warm-start source hash is invalid for {identity}")
        source_path_value = raw.get("source_solution_path")
        source_axis = raw.get("source_axis")
        if (
            not isinstance(source_path_value, str)
            or not source_path_value
            or not isinstance(source_axis, str)
            or not source_axis
        ):
            raise RuntimeError(f"warm-start source path/axis is invalid for {identity}")
        source_path = Path(source_path_value)
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        source_path = source_path.resolve()
        if not source_path.is_file() or _sha256_path(source_path) != source_sha256:
            raise RuntimeError(f"warm-start source solution hash mismatch for {identity}")
        try:
            source_payload = json.loads(source_path.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"warm-start source solution is unreadable for {identity}"
            ) from error
        source_axes = source_payload.get("axes") if isinstance(source_payload, dict) else None
        source_record = source_axes.get(source_axis) if isinstance(source_axes, dict) else None
        if not isinstance(source_record, dict):
            raise RuntimeError(f"warm-start source axis is missing for {identity}")
        source_routes = source_record.get("routes")
        source_objective = source_record.get("objective_key")
        if not isinstance(source_routes, list) or not all(
            isinstance(route, list) and all(isinstance(name, str) for name in route)
            for route in source_routes
        ):
            raise RuntimeError(f"warm-start source routes are invalid for {identity}")
        source_sequences = tuple(
            tuple(
                cast(str, name)
                for name in route
                if cast(str, name) in instance.by_name
                and instance.by_name[cast(str, name)].kind is NodeType.CUSTOMER
            )
            for route in source_routes
        )
        if source_sequences != tuple(sequences):
            raise RuntimeError(f"warm-start routes diverge from source for {identity}")
        source_report = validate_routes(instance, source_routes)
        if not source_report.feasible:
            raise RuntimeError(f"warm-start source routes are infeasible for {identity}")
        recomputed_objective = list(SolutionObjective.from_report(instance, source_report).key)
        if source_objective != recomputed_objective:
            raise RuntimeError(f"warm-start source objective mismatch for {identity}")
        declared_objective = raw.get("source_objective_key")
        if declared_objective != recomputed_objective:
            raise RuntimeError(f"warm-start declared objective mismatch for {identity}")
        sequence_sha256 = canonical_customer_sequences_sha256(tuple(sequences))
        output[identity] = (
            tuple(sequences),
            {
                "source_stage": raw.get("source_stage", "comparison_warm_start"),
                "source_run_label": raw.get("source_run_label", ""),
                "source_instance": instance_name,
                "source_seed": seed,
                "source_solution_sha256": source_sha256,
                "source_solution_path": str(source_path),
                "source_axis": source_axis,
                "source_customer_sequences_sha256": sequence_sha256,
                "source_objective_key": recomputed_objective,
                "warm_start_bundle_sha256": bundle_sha256,
                "warm_start_bundle_path": str(path.resolve()),
            },
        )
    return output


def _validate_native_build_attestation(
    native_core: NativeBuildAttestation,
    *,
    expected_revision: str,
    expected_tree: str,
    expected_source_manifest_sha256: str,
    expected_tracked_file_count: int,
) -> None:
    def is_lower_hex(value: object, length: int) -> bool:
        return (
            isinstance(value, str)
            and len(value) == length
            and all(character in "0123456789abcdef" for character in value)
        )

    try:
        build_revision = native_core.__build_git_revision__
        build_tree = native_core.__build_git_tree__
        source_manifest_sha256 = native_core.__build_source_manifest_sha256__
        tracked_file_count = native_core.__build_tracked_file_count__
        source_dirty = native_core.__build_source_dirty__
        development_override = native_core.__build_development_override__
        cpp_source_kind = native_core.__build_cpp_source_kind__
        attestation_version = native_core.__build_source_attestation_version__
        performance_profile = native_core.__build_performance_profile__
        compiler_id = native_core.__build_compiler_id__
        compiler_version = native_core.__build_compiler_version__
        interprocedural_optimization = native_core.__build_interprocedural_optimization__
        host_native = native_core.__build_host_native__
    except AttributeError as error:
        raise RuntimeError("installed native wheel lacks source attestation") from error
    if not is_lower_hex(build_revision, 40) or build_revision != expected_revision:
        raise RuntimeError("installed native wheel was not built from the recorded revision")
    if not is_lower_hex(build_tree, 40) or build_tree != expected_tree:
        raise RuntimeError("installed native wheel Git tree does not match the checkout")
    if type(attestation_version) is not int or attestation_version != 1:
        raise RuntimeError("installed native wheel has an unknown source attestation")
    if (
        not is_lower_hex(source_manifest_sha256, 64)
        or source_manifest_sha256 != expected_source_manifest_sha256
    ):
        raise RuntimeError("installed native wheel source manifest does not match Git")
    if type(tracked_file_count) is not int or tracked_file_count <= 0:
        raise RuntimeError("installed native wheel has an invalid tracked-file count")
    if tracked_file_count != expected_tracked_file_count:
        raise RuntimeError("installed native wheel tracked-file count does not match Git")
    if type(source_dirty) is not bool:
        raise RuntimeError("installed native wheel has an invalid dirty-source flag")
    if source_dirty:
        raise RuntimeError("installed native wheel was built from a dirty source tree")
    if type(development_override) is not bool:
        raise RuntimeError("installed native wheel has an invalid development override")
    if development_override:
        raise RuntimeError("installed native wheel used the development build override")
    if cpp_source_kind != "git_blob_snapshot":
        raise RuntimeError("installed native wheel did not use a Git-blob C++ snapshot")
    expected_profile_flags = {
        "portable-o3": (False, False),
        "portable-lto": (True, False),
        "host-native-lto": (True, True),
    }
    if performance_profile not in expected_profile_flags:
        raise RuntimeError("installed native wheel has an invalid performance profile")
    if not isinstance(compiler_id, str) or not compiler_id:
        raise RuntimeError("installed native wheel has an invalid compiler identity")
    if not isinstance(compiler_version, str) or not compiler_version:
        raise RuntimeError("installed native wheel has an invalid compiler version")
    if type(interprocedural_optimization) is not bool or type(host_native) is not bool:
        raise RuntimeError("installed native wheel has invalid performance flags")
    if (interprocedural_optimization, host_native) != expected_profile_flags[performance_profile]:
        raise RuntimeError("installed native wheel performance flags contradict profile")


def _verify_installed_project_files(
    wheel_path: Path,
    *,
    site_packages: Path,
    required_entry_sha256: Mapping[str, str] | None = None,
    expected_source_entries: Mapping[str, str] | None = None,
    generated_entries: set[str] | None = None,
) -> dict[str, str]:
    wheel_entry_sha256: dict[str, str] = {}
    with zipfile.ZipFile(wheel_path) as archive:
        project_entries = tuple(
            entry
            for entry in archive.infolist()
            if not entry.is_dir() and entry.filename.startswith(("evrptw/", "tools/"))
        )
        if not project_entries:
            raise RuntimeError("supplied wheel contains no project files")
        for entry in project_entries:
            entry_path = PurePosixPath(entry.filename)
            if (
                entry_path.is_absolute()
                or ".." in entry_path.parts
                or "\\" in entry.filename
                or entry.filename in wheel_entry_sha256
            ):
                raise RuntimeError("supplied wheel has an unsafe project entry")
            installed_path = site_packages / entry.filename
            if not installed_path.is_file():
                raise RuntimeError(f"installed wheel file is missing: {entry.filename}")
            expected_sha256 = hashlib.sha256(archive.read(entry)).hexdigest()
            if _sha256_path(installed_path) != expected_sha256:
                raise RuntimeError(
                    f"installed wheel file differs from its archive: {entry.filename}"
                )
            wheel_entry_sha256[entry.filename] = expected_sha256
    installed_project_files = {
        str(path.relative_to(site_packages)).replace(os.sep, "/")
        for package_name in ("evrptw", "tools")
        for path in (site_packages / package_name).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    unexpected_files = sorted(installed_project_files - set(wheel_entry_sha256))
    if unexpected_files:
        raise RuntimeError(
            "installed project files are absent from the supplied wheel: "
            + ", ".join(unexpected_files[:8])
        )
    if expected_source_entries is not None:
        source_entries = {
            entry for entry in wheel_entry_sha256 if entry not in (generated_entries or set())
        }
        expected_source_entry_hashes = dict(expected_source_entries)
        if source_entries != set(expected_source_entry_hashes) or any(
            wheel_entry_sha256[entry] != expected_sha256
            for entry, expected_sha256 in expected_source_entry_hashes.items()
        ):
            raise RuntimeError("wheel project source inventory does not match Git")
    for required_entry, expected_sha256 in (required_entry_sha256 or {}).items():
        if wheel_entry_sha256.get(required_entry) != expected_sha256:
            raise RuntimeError(f"required wheel entry is not hash-bound: {required_entry}")
    return dict(sorted(wheel_entry_sha256.items()))


def _verify_installed_wheel(
    wheel_path: Path,
    *,
    expected_revision: str,
    expected_tree: str,
    expected_source_manifest_sha256: str,
    expected_tracked_file_count: int,
    expected_source_entries: Mapping[str, str],
) -> WheelReceipt:
    """Prove that the executing distribution was installed from the supplied wheel."""

    resolved_wheel = wheel_path.resolve()
    wheel_sha256 = _sha256_path(resolved_wheel)
    distribution = importlib.metadata.distribution("reproducible-evrptw")
    direct_url_entry = next(
        (
            entry
            for entry in distribution.files or ()
            if str(entry).endswith(".dist-info/direct_url.json")
        ),
        None,
    )
    if direct_url_entry is None:
        raise RuntimeError("installed distribution has no direct_url.json wheel receipt")
    direct_url_path = Path(str(distribution.locate_file(direct_url_entry)))
    if not direct_url_path.is_file():
        raise RuntimeError("installed distribution has no direct_url.json wheel receipt")
    direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
    if not isinstance(direct_url, dict):
        raise RuntimeError("installed wheel receipt has an invalid schema")
    archive_info = direct_url.get("archive_info")
    source_url = direct_url.get("url")
    if not isinstance(archive_info, dict) or not isinstance(source_url, str):
        raise RuntimeError("executing distribution is not a non-editable wheel install")
    parsed = urlparse(source_url)
    installed_source = Path(unquote(parsed.path)).resolve()
    if parsed.scheme != "file" or installed_source != resolved_wheel:
        raise RuntimeError("executing distribution was installed from a different wheel")
    receipt_hash = archive_info.get("hash")
    expected_receipt_hash = f"sha256={wheel_sha256}"
    if receipt_hash != expected_receipt_hash:
        raise RuntimeError("installed wheel receipt SHA-256 does not match supplied wheel")
    import evrptw
    from evrptw import _core as native_core

    site_packages = direct_url_path.parent.parent.resolve()
    package_path = Path(str(evrptw.__file__)).resolve()
    native_path = Path(str(native_core.__file__)).resolve()
    scheduler_path = native_path.with_name("_native_host_scheduler")
    if (
        not package_path.is_relative_to(site_packages)
        or not native_path.is_relative_to(site_packages)
        or not scheduler_path.is_relative_to(site_packages)
    ):
        raise RuntimeError("comparison runner imported source outside the installed wheel")
    if not scheduler_path.is_file() or not os.access(scheduler_path, os.X_OK):
        raise RuntimeError("installed wheel has no executable native host scheduler")
    _validate_native_build_attestation(
        native_core,
        expected_revision=expected_revision,
        expected_tree=expected_tree,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_tracked_file_count=expected_tracked_file_count,
    )
    native_sha256 = _sha256_path(native_path)
    scheduler_sha256 = _sha256_path(scheduler_path)
    runner_path = Path(__file__).resolve()
    if not runner_path.is_relative_to(site_packages):
        raise RuntimeError("comparison runner was imported outside the installed wheel")
    native_wheel_entry = native_path.relative_to(site_packages).as_posix()
    scheduler_wheel_entry = scheduler_path.relative_to(site_packages).as_posix()
    runner_wheel_entry = runner_path.relative_to(site_packages).as_posix()
    wheel_entry_sha256 = _verify_installed_project_files(
        resolved_wheel,
        site_packages=site_packages,
        required_entry_sha256={
            native_wheel_entry: native_sha256,
            scheduler_wheel_entry: scheduler_sha256,
            runner_wheel_entry: _sha256_path(runner_path),
        },
        expected_source_entries=expected_source_entries,
        generated_entries={native_wheel_entry, scheduler_wheel_entry},
    )
    scheduler_attestation_raw = subprocess.run(
        [str(scheduler_path), "--build-attestation"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    try:
        scheduler_attestation = json.loads(scheduler_attestation_raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("native scheduler build attestation is invalid") from error
    if not isinstance(scheduler_attestation, dict):
        raise RuntimeError("native scheduler build attestation is invalid")
    validate_scheduler_build_attestation(
        scheduler_attestation,
        revision=native_core.__build_git_revision__,
        git_tree=native_core.__build_git_tree__,
        source_manifest_sha256=native_core.__build_source_manifest_sha256__,
        tracked_file_count=native_core.__build_tracked_file_count__,
        performance_profile=native_core.__build_performance_profile__,
        compiler_id=native_core.__build_compiler_id__,
        compiler_version=native_core.__build_compiler_version__,
        interprocedural_optimization=(native_core.__build_interprocedural_optimization__),
        host_native=native_core.__build_host_native__,
    )
    return {
        "wheel_path": str(resolved_wheel),
        "wheel_sha256": wheel_sha256,
        "direct_url_path": str(direct_url_path.resolve()),
        "package_path": str(package_path),
        "native_path": str(native_path),
        "native_sha256": native_sha256,
        "scheduler_path": str(scheduler_path),
        "scheduler_sha256": scheduler_sha256,
        "build_git_revision": native_core.__build_git_revision__,
        "build_git_tree": native_core.__build_git_tree__,
        "build_source_manifest_sha256": (native_core.__build_source_manifest_sha256__),
        "build_tracked_file_count": native_core.__build_tracked_file_count__,
        "build_source_dirty": False,
        "build_development_override": False,
        "build_cpp_source_kind": native_core.__build_cpp_source_kind__,
        "build_source_attestation_version": 1,
        "build_performance_profile": native_core.__build_performance_profile__,
        "build_compiler_id": native_core.__build_compiler_id__,
        "build_compiler_version": native_core.__build_compiler_version__,
        "build_interprocedural_optimization": (native_core.__build_interprocedural_optimization__),
        "build_host_native": native_core.__build_host_native__,
        "wheel_entry_sha256": wheel_entry_sha256,
        "native_wheel_entry": native_wheel_entry,
        "scheduler_wheel_entry": scheduler_wheel_entry,
        "runner_wheel_entry": runner_wheel_entry,
        "scheduler_build_attestation": dict(scheduler_attestation),
    }


def _require_native_architecture_capabilities() -> dict[str, bool]:
    """Fail before any run label exists when the native design is incomplete."""

    from evrptw import _core as native_core

    raw = native_core.stage052_native_architecture_capabilities_v2()
    if (
        not isinstance(raw, np.ndarray)
        or raw.dtype != np.dtype(np.int64)
        or raw.shape != (len(NATIVE_ARCHITECTURE_CAPABILITY_NAMES),)
        or not raw.flags.c_contiguous
        or np.any((raw != 0) & (raw != 1))
    ):
        raise RuntimeError("native architecture capability receipt is invalid")
    capabilities = {
        name: bool(raw[index]) for index, name in enumerate(NATIVE_ARCHITECTURE_CAPABILITY_NAMES)
    }
    missing = [name for name, available in capabilities.items() if not available]
    if missing:
        raise RuntimeError(
            "native architecture campaign is blocked by incomplete capabilities: "
            + ", ".join(missing)
        )
    return capabilities


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_size(payload: object) -> int:
    encoder = json.JSONEncoder(
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return sum(len(chunk.encode("utf-8")) for chunk in encoder.iterencode(payload)) + 1


def _write_signed_json_with_receipt(
    path: Path,
    payload: object,
) -> tuple[int, dict[str, object]]:
    write_chunk_bytes = 512 * 1024
    total_started = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"signed JSON target already exists: {path}")
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + f".tmp-{os.getpid()}")
    write_seconds = 0.0
    encoding_seconds = 0.0
    hash_seconds = 0.0
    fsync_seconds = 0.0
    atomic_publish_seconds = 0.0
    data_bytes = 0
    write_batch_count = 0
    maximum_write_batch_bytes = 0
    path_published = False
    sidecar_published = False
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        with temporary.open("xb") as data_stream:
            buffer = bytearray()

            def persist_buffer() -> None:
                nonlocal data_bytes
                nonlocal hash_seconds
                nonlocal maximum_write_batch_bytes
                nonlocal write_batch_count
                nonlocal write_seconds
                if not buffer:
                    return
                batch_bytes = len(buffer)
                started = time.perf_counter()
                digest.update(buffer)
                hash_seconds += time.perf_counter() - started
                started = time.perf_counter()
                written = data_stream.write(buffer)
                write_seconds += time.perf_counter() - started
                if written != batch_bytes:
                    raise OSError("signed JSON streaming write was incomplete")
                data_bytes += written
                write_batch_count += 1
                maximum_write_batch_bytes = max(
                    maximum_write_batch_bytes,
                    batch_bytes,
                )
                buffer.clear()

            encoding_started = time.perf_counter()
            for token in encoder.iterencode(payload):
                encoded = token.encode("utf-8")
                offset = 0
                while offset < len(encoded):
                    available = write_chunk_bytes - len(buffer)
                    consumed = min(available, len(encoded) - offset)
                    buffer.extend(memoryview(encoded)[offset : offset + consumed])
                    offset += consumed
                    if len(buffer) == write_chunk_bytes:
                        encoding_seconds += time.perf_counter() - encoding_started
                        persist_buffer()
                        encoding_started = time.perf_counter()
            buffer.extend(b"\n")
            encoding_seconds += time.perf_counter() - encoding_started
            persist_buffer()
            started = time.perf_counter()
            data_stream.flush()
            write_seconds += time.perf_counter() - started
            started = time.perf_counter()
            os.fsync(data_stream.fileno())
            fsync_seconds += time.perf_counter() - started
        with temporary_sidecar.open("x", encoding="ascii") as sidecar_stream:
            started = time.perf_counter()
            sidecar_stream.write(digest.hexdigest() + "\n")
            sidecar_stream.flush()
            write_seconds += time.perf_counter() - started
            started = time.perf_counter()
            os.fsync(sidecar_stream.fileno())
            fsync_seconds += time.perf_counter() - started
        started = time.perf_counter()
        publish_no_replace(temporary_sidecar, sidecar)
        sidecar_published = True
        publish_no_replace(temporary, path)
        path_published = True
        atomic_publish_seconds += time.perf_counter() - started
        parent_fd = os.open(path.parent, os.O_RDONLY)
        try:
            started = time.perf_counter()
            os.fsync(parent_fd)
            fsync_seconds += time.perf_counter() - started
        finally:
            os.close(parent_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        if not path_published and sidecar_published:
            sidecar.unlink(missing_ok=True)
        raise
    sidecar_bytes = sidecar.stat().st_size
    receipt = {
        "schema_version": "stage05.2-signed-json-persistence-v2",
        "encoding_seconds": encoding_seconds,
        "hash_seconds": hash_seconds,
        "write_seconds": write_seconds,
        "fsync_seconds": fsync_seconds,
        "atomic_publish_seconds": atomic_publish_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "data_bytes": data_bytes,
        "sidecar_bytes": sidecar_bytes,
        "write_chunk_bytes": write_chunk_bytes,
        "write_batch_count": write_batch_count,
        "maximum_write_batch_bytes": maximum_write_batch_bytes,
    }
    return data_bytes + sidecar_bytes, receipt


def _write_signed_json(path: Path, payload: object) -> int:
    observed_bytes, _receipt = _write_signed_json_with_receipt(path, payload)
    return observed_bytes


def _set_artifact_size(
    payload: dict[str, object],
    *,
    external_bytes: int = 0,
) -> None:
    payload["artifact_bytes"] = 0
    for _ in range(4):
        size = _canonical_json_size(payload) + 65 + external_bytes
        if payload["artifact_bytes"] == size:
            return
        payload["artifact_bytes"] = size
    raise RuntimeError("artifact byte count did not converge")


def _axis_persistence_receipt_path(axis_path: Path) -> Path:
    return axis_path.with_suffix(axis_path.suffix + ".persistence")


def _axis_persistence_descriptor(axis_path: Path) -> dict[str, str]:
    receipt_path = _axis_persistence_receipt_path(axis_path)
    return {
        "schema_version": AXIS_PERSISTENCE_RECEIPT_SCHEMA_VERSION,
        "path": receipt_path.name,
        "sidecar_path": receipt_path.name + ".sha256",
    }


def _set_axis_persistence_receipt_size(
    receipt: dict[str, object],
    *,
    primary_artifact_bytes: int,
) -> None:
    receipt["persistence_receipt_bytes"] = 0
    receipt["artifact_bytes"] = primary_artifact_bytes
    for _ in range(8):
        receipt_bytes = _canonical_json_size(receipt) + 65
        artifact_bytes = primary_artifact_bytes + receipt_bytes
        if (
            receipt["persistence_receipt_bytes"] == receipt_bytes
            and receipt["artifact_bytes"] == artifact_bytes
        ):
            return
        receipt["persistence_receipt_bytes"] = receipt_bytes
        receipt["artifact_bytes"] = artifact_bytes
    raise RuntimeError("axis persistence receipt byte count did not converge")


def _native_config(
    mode: ArchitectureMode,
    task: ArchitectureAxisTask,
    *,
    scheduler_socket_path: str | None = None,
    task_receipt_path: Path | None = None,
) -> Stage052NativeExecutionConfig:
    if mode not in {
        ArchitectureMode.PER_SOLVE_RUNTIME,
        ArchitectureMode.FULL_NATIVE_ALNS,
        ArchitectureMode.HOST_SCHEDULER,
    }:
        raise ValueError("mode does not use the explicit native execution protocol")
    profile_bound = task.performance_profile_sha256 is not None
    threads_per_shard = task.threads_per_shard if profile_bound else THREADS_PER_SHARD
    shard_processes = task.shard_processes if profile_bound else SHARD_PROCESSES
    scheduler_threads = (
        len(task.scheduler_cpu_ids)
        if profile_bound and mode is ArchitectureMode.HOST_SCHEDULER
        else (task.total_compute_threads if profile_bound else TOTAL_COMPUTE_THREADS)
    )
    return Stage052NativeExecutionConfig(
        mode=mode.value,  # type: ignore[arg-type]
        native_kernel_config=NativeKernelConfig(),
        candidate_transaction_config=NativeCandidateTransactionConfig(),
        candidate_control_config=CandidateControlConfig(
            worker_count=threads_per_shard,
            stage052_calibrated_workers=profile_bound,
        ),
        shard_processes=shard_processes,
        compute_threads_per_shard=threads_per_shard,
        scheduler_threads=scheduler_threads,
        scheduler_socket_path=scheduler_socket_path,
        total_compute_threads=(task.total_compute_threads if profile_bound else None),
        frozen_performance_profile_sha256=task.performance_profile_sha256,
        axis_cpu_ids=task.axis_cpu_ids,
        scheduler_cpu_ids=task.scheduler_cpu_ids,
        allow_affinity_overlap=task.allow_affinity_overlap,
        task_receipt_path=(
            str(task_receipt_path)
            if mode
            in {
                ArchitectureMode.PER_SOLVE_RUNTIME,
                ArchitectureMode.FULL_NATIVE_ALNS,
            }
            and task_receipt_path is not None
            else None
        ),
    )


def _thread_count() -> int:
    return len(tuple((Path("/proc") / str(os.getpid()) / "task").iterdir()))


def _rss_bytes() -> int:
    fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
    if len(fields) < 2:
        raise RuntimeError("cannot read process RSS from /proc/self/statm")
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def _runtime_cgroup_snapshot(
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    membership_path: Path = Path("/proc/self/cgroup"),
) -> dict[str, object]:
    try:
        membership = membership_path.read_text(encoding="ascii")
        relative = next(
            line.split("::", 1)[1] for line in membership.splitlines() if line.startswith("0::")
        ).lstrip("/")
        root = cgroup_root.resolve()
        directory = (root / relative).resolve()
        directory.relative_to(root)
    except (OSError, StopIteration, ValueError, IndexError):
        return {
            "status": "unavailable",
            "cgroup_path": "unavailable",
        }

    def counter(name: str) -> int | str:
        try:
            text = (directory / name).read_text(encoding="ascii").strip()
            if text == "max":
                return "max"
            value = int(text)
            return value if value >= 0 else "unavailable"
        except (OSError, ValueError, UnicodeError):
            return "unavailable"

    def keyed_counters(name: str) -> dict[str, int] | str:
        try:
            values: dict[str, int] = {}
            for line in (directory / name).read_text(encoding="ascii").splitlines():
                key, raw_value = line.split()
                value = int(raw_value)
                if value < 0:
                    return "unavailable"
                values[key] = value
            return dict(sorted(values.items()))
        except (OSError, ValueError, UnicodeError):
            return "unavailable"

    io_totals: dict[str, int] | str
    try:
        totals = {
            "read_bytes": 0,
            "write_bytes": 0,
            "read_operations": 0,
            "write_operations": 0,
            "discard_bytes": 0,
            "discard_operations": 0,
        }
        field_map = {
            "rbytes": "read_bytes",
            "wbytes": "write_bytes",
            "rios": "read_operations",
            "wios": "write_operations",
            "dbytes": "discard_bytes",
            "dios": "discard_operations",
        }
        for line in (directory / "io.stat").read_text(encoding="ascii").splitlines():
            for field in line.split()[1:]:
                key, raw_value = field.split("=", 1)
                if key in field_map:
                    totals[field_map[key]] += int(raw_value)
        io_totals = totals
    except (OSError, ValueError, UnicodeError):
        io_totals = "unavailable"
    return {
        "status": "available",
        "cgroup_path": "/" + relative,
        "memory_current_bytes": counter("memory.current"),
        "memory_peak_bytes": counter("memory.peak"),
        "memory_swap_current_bytes": counter("memory.swap.current"),
        "memory_swap_peak_bytes": counter("memory.swap.peak"),
        "memory_events": keyed_counters("memory.events"),
        "io": io_totals,
    }


def _cgroup_counter_delta(
    before: Mapping[str, object],
    after: Mapping[str, object],
    field: str,
) -> int | str:
    before_events = before.get("memory_events")
    after_events = after.get("memory_events")
    if not isinstance(before_events, Mapping) or not isinstance(after_events, Mapping):
        return "unavailable"
    before_value = before_events.get(field)
    after_value = after_events.get(field)
    if (
        isinstance(before_value, bool)
        or not isinstance(before_value, int)
        or isinstance(after_value, bool)
        or not isinstance(after_value, int)
        or after_value < before_value
    ):
        return "unavailable"
    return after_value - before_value


def _cgroup_io_deltas(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, int] | str:
    before_io = before.get("io")
    after_io = after.get("io")
    fields = (
        "read_bytes",
        "write_bytes",
        "read_operations",
        "write_operations",
        "discard_bytes",
        "discard_operations",
    )
    if not isinstance(before_io, Mapping) or not isinstance(after_io, Mapping):
        return "unavailable"
    deltas: dict[str, int] = {}
    for field in fields:
        before_value = before_io.get(field)
        after_value = after_io.get(field)
        if (
            isinstance(before_value, bool)
            or not isinstance(before_value, int)
            or isinstance(after_value, bool)
            or not isinstance(after_value, int)
            or after_value < before_value
        ):
            return "unavailable"
        deltas[field] = after_value - before_value
    return deltas


def _metric_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _row_evidence(rows: Iterable[object]) -> dict[str, object]:
    digest = hashlib.sha256(b"stage05.2-row-evidence-v1\0")
    count = 0
    for row in rows:
        encoded = _canonical_bytes(_evidence_json_value(row))
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        count += 1
    return {"count": count, "sha256": digest.hexdigest()}


def _evidence_json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return _evidence_json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"nonfinite_float": "nan"}
        return {"nonfinite_float": "positive_inf" if value > 0.0 else "negative_inf"}
    if isinstance(value, dict):
        return {str(key): _evidence_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_evidence_json_value(item) for item in value]
    return value


def _canonical_trace_event(event: dict[str, object]) -> dict[str, object]:
    """Remove wall-clock telemetry from one replayable semantic event."""

    canonical = {
        key: value
        for key, value in event.items()
        if key not in {"timestamp_seconds", "duration_seconds"}
    }
    if canonical.get("event_type") in {
        "candidate_cache_transaction",
        "candidate_screening_aggregate",
    }:
        return {}
    if canonical.get("event_type") in {
        "deadline_boundary",
        "exact_budget_boundary",
        "candidate_control_boundary",
    }:
        if canonical.get("termination_boundary") is not True:
            return {}
        canonical.pop("termination_boundary", None)
    if canonical.get("event_type") == "screening_decision":
        raw_checks = canonical.get("checks", [])
        if isinstance(raw_checks, list):
            normalized_checks: list[dict[str, object]] = []
            boolean_checks = {
                "route_structure",
                "single_segment_battery_reachability",
            }
            for raw_check in raw_checks:
                if not isinstance(raw_check, dict):
                    continue
                check_name = str(raw_check.get("check", ""))
                check_value = raw_check.get("value")
                normalized_checks.append(
                    {
                        "check": check_name,
                        "status": raw_check.get("status"),
                        "value": (
                            bool(check_value) if check_name in boolean_checks else check_value
                        ),
                        "reason": raw_check.get("reason", ""),
                    }
                )
            canonical["checks"] = normalized_checks
        canonical = {
            key: value
            for key, value in canonical.items()
            if key
            in {
                "event_type",
                "route_key",
                "lane",
                "iteration",
                "operator",
                "status",
                "first_failed_check",
                "reason",
                "checks",
                "demand",
                "min_time_window_slack",
                "distance_lower_bound",
                "distance_increment_lower_bound",
                "single_segment_reachable",
                "structural_energy_lower_bound",
                "negative_cache_hit",
                "exact_call_blocked",
                "semantic_event_id",
            }
        }
    if canonical.get("event_type") == "candidate_control_budget":
        canonical.pop("accounting", None)
        context = canonical.get("context")
        if isinstance(context, str) and context.endswith(":native_candidate_round"):
            canonical["context"] = (
                context.removesuffix(":native_candidate_round") + ":candidate_pool"
            )
    if canonical.get("event_type") == "exact_batch_started":
        canonical.pop("transaction_sha256", None)
    if canonical.get("event_type") == "exact_route_result":
        canonical.pop("evaluation_id", None)
    if canonical.get("event_type") == "candidate_plan_decision":
        canonical.pop("batch_ordinal", None)
        canonical.pop("transaction_status_code", None)
    if canonical.get("event_type") == "cache_event":
        operation = canonical.get("operation")
        if operation in {"lookup", "store", "evict", "reconcile", "oversize_not_cached"}:
            return {}
        if operation in {"hit", "candidate_pending_hit", "miss"}:
            canonical = {
                key: value
                for key, value in canonical.items()
                if key
                in {
                    "event_type",
                    "route_key",
                    "lane",
                    "iteration",
                    "operator",
                    "semantic_event_id",
                }
            }
            canonical["event_type"] = "cache_lookup_result"
            canonical["status"] = "miss" if operation == "miss" else "hit"
    if canonical.get("event_type") == "parallel_batch":
        canonical["event_type"] = "candidate_batch_complete"
        canonical["status"] = "complete"
        for key in PARALLEL_BATCH_NON_CANONICAL_FIELDS:
            canonical.pop(key, None)
        sequences = canonical.get("customer_sequences")
        if not isinstance(sequences, list | tuple):
            raise ValueError("candidate batch requires customer_sequences")
        canonical["result_count"] = len(sequences)
    if canonical.get("event_type") == "candidate_state":
        route_keys = canonical.get("candidate_route_keys")
        if not isinstance(route_keys, list | tuple) or not all(
            isinstance(key, str) for key in route_keys
        ):
            raise ValueError("candidate_state requires candidate_route_keys as a string array")
        full_route_keys = canonical.get("candidate_full_route_keys", [])
        if not isinstance(full_route_keys, list | tuple) or not all(
            isinstance(key, str) for key in full_route_keys
        ):
            raise ValueError("candidate_full_route_keys must be a string array when present")
        canonical = {
            key: canonical[key]
            for key in (
                "event_type",
                "lane",
                "iteration",
                "operator",
                "candidate_objective_key",
                "candidate_feasible",
                "accepted",
                "status",
                "reason",
                "semantic_event_id",
            )
            if key in canonical
        }
        canonical["candidate_route_keys"] = list(route_keys)
        canonical["candidate_full_route_keys"] = list(full_route_keys)
        identity = {
            "candidate_route_keys": list(route_keys),
            "candidate_full_route_keys": list(full_route_keys),
        }
        canonical["candidate_id"] = hashlib.sha256(_canonical_bytes(identity)).hexdigest()
    return canonical


def _iter_semantic_candidate_trajectory(
    result: ALNSResult,
) -> Iterator[dict[str, object]]:
    quality_operators = {
        "relocate",
        "swap",
        "two_opt_star",
        "route_segment_destroy",
        "ejection_chain",
    }
    ordinal = 0
    for raw_event in result.neighborhood_events:
        if raw_event.get("status") in SEMANTIC_TRAJECTORY_IMPLEMENTATION_STATUSES:
            # Aggregate rows describe implementation-specific work rather than
            # search decisions.  They remain in raw neighborhood evidence (and
            # pair pruning also has a dedicated native-ablation stream), but
            # cannot shift cross-runtime canonical ordinals.
            continue
        event = {
            key: value
            for key, value in raw_event.items()
            if not str(key).startswith("_")
            and not (key == "aggregate_count" and value == 1)
            and not (key == "candidate_pool_hash" and not value)
        }
        operator = str(event.get("operator", ""))
        track = str(event.get("track", ""))
        lane = (
            "constraint_lane"
            if track == "constraint_lane"
            else "quality_shadow"
            if operator in quality_operators
            else "legacy"
        )
        identity = {
            "lane": lane,
            "iteration": event.get("iteration"),
            "operator": operator,
            "status": event.get("status"),
            "candidate_route_sequences": event.get("candidate_route_sequences", ()),
            "candidate_objective_key": event.get("candidate_objective_key", ()),
            "ordinal": ordinal,
        }
        yield {
            **event,
            "lane": lane,
            "candidate_id": hashlib.sha256(_canonical_bytes(identity)).hexdigest(),
        }
        ordinal += 1


def _semantic_candidate_trajectory(result: ALNSResult) -> list[dict[str, object]]:
    return list(_iter_semantic_candidate_trajectory(result))


def _canonical_event_rows(rows: Iterable[object]) -> list[dict[str, object]]:
    canonical_rows: list[dict[str, object]] = []
    for ordinal, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("canonical semantic streams require event objects")
        canonical = _evidence_json_value(_canonical_trace_event(dict(row)))
        if not isinstance(canonical, dict):
            raise AssertionError("canonical event normalization lost object identity")
        canonical_rows.append({**canonical, "stream_ordinal": ordinal})
    return canonical_rows


def _canonical_semantic_streams(result: ALNSResult) -> dict[str, list[dict[str, object]]]:
    trace = result.measurement_trace
    if trace is None:
        raise ValueError("canonical semantic streams require a measurement trace")
    stream_names = (
        "candidate_state",
        "operator",
        "stage04",
        "candidate_transaction",
        "exact_work",
        "exact_result",
        "cache",
        "screening",
        "deadline",
        "termination",
        "native_failure",
    )
    streams: dict[str, list[dict[str, object]]] = {name: [] for name in stream_names}
    runtime_events = trace.runtime_semantic_events
    runtime_ids = [event.get("semantic_event_id") for event in runtime_events]
    if runtime_ids != list(range(1, len(runtime_events) + 1)):
        raise ValueError("runtime semantic trace IDs are not contiguous")
    semantic_event_id = 0
    for raw_event in runtime_events:
        stream_name = raw_event.get("semantic_stream")
        if not isinstance(stream_name, str) or stream_name not in streams:
            raise ValueError("runtime semantic event names an unknown stream")
        source_runtime_event_id = raw_event.get("runtime_causal_event_id")
        if (
            isinstance(source_runtime_event_id, bool)
            or not isinstance(source_runtime_event_id, int)
            or source_runtime_event_id != raw_event.get("semantic_event_id")
        ):
            raise ValueError("runtime semantic event lost its causal source identity")
        event = {
            key: value
            for key, value in raw_event.items()
            if key
            not in {
                "semantic_stream",
                "runtime_causal_event_id",
                "runtime_native_event_id",
                "runtime_native_stream_code",
                "runtime_native_event_code",
                "runtime_native_lane_id",
                "runtime_native_operator_id",
                "runtime_native_iteration",
                "runtime_native_transaction_id",
                "runtime_native_subject_id",
                "runtime_native_status_code",
                "runtime_native_flags",
            }
        }
        canonical = _evidence_json_value(_canonical_trace_event(event))
        if not isinstance(canonical, dict):
            raise AssertionError("canonical event normalization lost object identity")
        if not canonical:
            continue
        runtime_event_id = canonical.pop("semantic_event_id", None)
        if not isinstance(runtime_event_id, int) or isinstance(runtime_event_id, bool):
            raise ValueError("runtime semantic event lost its event ID")
        semantic_event_id += 1
        streams[stream_name].append(
            {
                **canonical,
                "runtime_event_id": runtime_event_id,
                "semantic_event_id": semantic_event_id,
                "stream_ordinal": len(streams[stream_name]),
            }
        )
    required_nonempty = {
        "stage04",
        "exact_work",
        "exact_result",
        "cache",
        "screening",
        "termination",
    }
    if result.effective_iterations > 0:
        required_nonempty.update({"candidate_state", "operator"})
    if result.candidate_control_statistics.get("enabled") is True:
        required_nonempty.add("candidate_transaction")
    missing = sorted(name for name in required_nonempty if not streams[name])
    if missing:
        raise ValueError("runtime semantic journal is incomplete: " + ", ".join(missing))
    termination = streams["termination"]
    if result.objective is None:
        raise ValueError("runtime semantic termination requires an objective")
    if (
        len(termination) != 1
        or termination[0].get("event_type") != "termination"
        or termination[0].get("status") != result.termination_reason
        or termination[0].get("iterations") != result.iterations
        or termination[0].get("effective_iterations") != result.effective_iterations
        or termination[0].get("exact_started_calls") != result.exact_started_calls
        or termination[0].get("exact_completed_calls") != result.exact_completed_calls
        or termination[0].get("exact_interrupted_calls") != result.exact_interrupted_calls
        or termination[0].get("objective_key") != list(result.objective.key)
        or termination[0].get("semantic_event_id") != semantic_event_id
    ):
        raise ValueError("runtime semantic termination does not reconcile")
    if result.termination_reason != "iteration_limit" and not streams["deadline"]:
        raise ValueError("runtime semantic deadline boundary is missing")
    return streams


def _canonical_semantic_event_sequence(
    streams: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Merge runtime-stamped journals without inventing cross-stream order."""

    expected_streams = {
        "candidate_state",
        "operator",
        "stage04",
        "candidate_transaction",
        "exact_work",
        "exact_result",
        "cache",
        "screening",
        "deadline",
        "termination",
        "native_failure",
    }
    if set(streams) != expected_streams:
        raise ValueError("canonical semantic stream set is incomplete")
    if not any(streams.values()):
        raise ValueError("canonical semantic runtime journal cannot be empty")
    events: list[dict[str, object]] = []
    for stream_name, rows in streams.items():
        previous_event_id = 0
        for row in rows:
            event_id = row.get("semantic_event_id")
            if (
                isinstance(event_id, bool)
                or not isinstance(event_id, int)
                or event_id <= previous_event_id
            ):
                raise ValueError(
                    f"canonical semantic stream {stream_name} lacks ordered runtime event IDs"
                )
            previous_event_id = event_id
            events.append({**row, "semantic_stream": stream_name})

    def runtime_event_id(event: dict[str, object]) -> int:
        value = event["semantic_event_id"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise AssertionError("validated runtime event ID changed type")
        return value

    events.sort(key=runtime_event_id)
    event_ids = [runtime_event_id(event) for event in events]
    if event_ids != list(range(1, len(events) + 1)):
        raise ValueError("canonical semantic runtime event IDs must be unique and contiguous")
    return [{**event, "semantic_sequence": sequence} for sequence, event in enumerate(events)]


def _semantic_operator_statistics(result: ALNSResult) -> dict[str, dict[str, object]]:
    telemetry_fields = {
        "failure_reasons",
        "prefilter_passed",
        "prefilter_rejected",
    }
    return {
        operator: {key: value for key, value in statistics.items() if key not in telemetry_fields}
        for operator, statistics in sorted(result.neighborhood_statistics.items())
    }


def _measurement_evidence(result: ALNSResult) -> dict[str, object]:
    trace = result.measurement_trace
    candidate_trajectory_evidence = _row_evidence(_iter_semantic_candidate_trajectory(result))
    if trace is None:
        semantic = {
            "present": False,
            "exact_route_order": _row_evidence(()),
            "cache_lifecycle": _row_evidence(()),
            "deadline_boundaries": _row_evidence(()),
        }
    else:
        semantic_work_events = tuple(
            event
            for event in trace.runtime_semantic_events
            if event.get("semantic_stream") == "exact_work"
            and event.get("event_type") == "exact_batch_started"
        )
        exact_route_order = tuple(
            {
                "batch_ordinal": ordinal,
                "lane": event.get("lane"),
                "iteration": event.get("iteration"),
                "operator": event.get("operator"),
                "sequences": event.get("customer_sequences"),
            }
            for ordinal, event in enumerate(semantic_work_events)
        )
        semantic_cache_rows: list[dict[str, object]] = []
        for batch_ordinal, event in enumerate(semantic_work_events):
            for route_ordinal, sequence in enumerate(
                cast(list[list[str]], event.get("customer_sequences", []))
            ):
                semantic_cache_rows.append(
                    {
                        "batch_ordinal": batch_ordinal,
                        "route_ordinal": route_ordinal,
                        "lane": event.get("lane"),
                        "iteration": event.get("iteration"),
                        "operator": event.get("operator"),
                        "customer_sequence": sequence,
                        "transition": "exact_miss_to_committed_store",
                    }
                )
        cache_statistics = result.cache_incremental_statistics
        semantic_cache_rows.append(
            {
                "transition": "final_cache_state",
                **{
                    field: cache_statistics.get(field, 0)
                    for field in (
                        "cache_stores",
                        "cache_evictions",
                        "cache_oversize_not_cached",
                        "entries_current",
                        "entries_peak",
                        "bytes_current",
                        "bytes_peak",
                        "unique_route_evaluations",
                    )
                },
            }
        )
        canonical_routes = sorted(
            {
                tuple(sequence)
                for event in semantic_work_events
                for sequence in cast(list[list[str]], event.get("customer_sequences", []))
            }
        )
        deadline_boundaries = (
            {
                "evaluation_id": row.evaluation_id,
                "route_key": row.route_key,
                "deadline_boundary": row.deadline_boundary,
                "exact_started": row.exact_started,
                "exact_completed": row.exact_completed,
                "status": row.status,
            }
            for row in trace.route_evaluations
            if row.deadline_boundary
        )
        semantic = {
            "present": True,
            "exact_route_order": _row_evidence(exact_route_order),
            "exact_route_results": _row_evidence(result.route_result_events),
            "cache_lifecycle": _row_evidence(semantic_cache_rows),
            "deadline_boundaries": _row_evidence(deadline_boundaries),
            "route_dictionary": _row_evidence({"route": list(route)} for route in canonical_routes),
            # Only cross-adapter semantics participate in the transaction
            # digest. Native screening/pruning and incremental telemetry are
            # audited below but are allowed to use different implementations.
            "events": candidate_trajectory_evidence,
            "candidate_trajectory": dict(candidate_trajectory_evidence),
        }
        semantic["native_telemetry"] = {
            "screening_decisions": _row_evidence(asdict(row) for row in trace.screening_decisions),
            "incremental_propagations": _row_evidence(
                dict(row) for row in trace.incremental_propagations
            ),
        }
    digest_payload = {key: value for key, value in semantic.items() if key != "native_telemetry"}
    semantic["sha256"] = hashlib.sha256(_canonical_bytes(digest_payload)).hexdigest()
    return semantic


def _solve_mode(
    mode: ArchitectureMode,
    task: ArchitectureAxisTask,
    *,
    telemetry_enabled: bool = True,
    resource_telemetry_enabled: bool | None = None,
    task_receipt_path: Path | None = None,
) -> tuple[ALNSResult, float, dict[str, object]]:
    monitor_enabled = (
        telemetry_enabled if resource_telemetry_enabled is None else resource_telemetry_enabled
    )
    profile_bound = task.performance_profile_sha256 is not None
    if profile_bound:
        if not task.axis_cpu_ids:
            raise RuntimeError("profile-bound architecture axis has no CPU allocation")
        os.sched_setaffinity(0, task.axis_cpu_ids)
        if set(os.sched_getaffinity(0)) != set(task.axis_cpu_ids):
            raise RuntimeError("architecture axis affinity differs from frozen profile")
    threads_per_shard = task.threads_per_shard if profile_bound else THREADS_PER_SHARD
    shard_processes = task.shard_processes if profile_bound else SHARD_PROCESSES
    total_compute_threads = task.total_compute_threads if profile_bound else TOTAL_COMPUTE_THREADS
    instance = replace(
        parse_schneider(task.benchmark_dir / f"{task.instance_name}.txt"),
        distance_backend="native",
    )
    fixed_work = task.axis == "fixed_work"
    if (
        isinstance(task.fixed_work_exact_calls, bool)
        or task.fixed_work_exact_calls <= 0
        or isinstance(task.fixed_work_max_iterations, bool)
        or task.fixed_work_max_iterations <= 0
        or isinstance(task.fixed_work_watchdog_seconds, bool)
        or not math.isfinite(task.fixed_work_watchdog_seconds)
        or task.fixed_work_watchdog_seconds <= 0.0
        or isinstance(task.exact_batch_size, bool)
        or task.exact_batch_size <= 0
    ):
        raise RuntimeError("architecture axis fixed-work budget is invalid")
    exact_deadline = (
        ExactDeadlineConfig.fixed_exact_calls(
            task.fixed_work_exact_calls,
            watchdog_seconds=task.fixed_work_watchdog_seconds,
        )
        if fixed_work
        else ExactDeadlineConfig.wall_clock()
    )
    threads_before = _thread_count()
    common: dict[str, object] = {
        "seed": task.seed,
        "max_iterations": task.fixed_work_max_iterations if fixed_work else 1000,
        "time_limit_seconds": task.fixed_work_watchdog_seconds if fixed_work else 30.0,
        "operator_profile": "stage02_constraint_guided",
        "measurement_config": (
            MeasurementConfig(
                record_runtime_semantic_events=True,
                externalize_runtime_semantic_events=True,
            )
            if telemetry_enabled
            else None
        ),
        "screening_config": CheapScreeningConfig(),
        "cache_incremental_config": CacheIncrementalConfig(enabled=True),
        "backend": "cpu_batch",
        "batch_size": task.exact_batch_size,
        "termination_mode": "fixed_work" if fixed_work else "wall_clock",
        "exact_deadline_config": exact_deadline,
        "stage04_config": Stage04Config(),
        "initial_customer_sequences": task.initial_customer_sequences,
        "initial_solution_provenance": task.initial_solution_provenance,
        "warm_start_validation_config": WarmStartValidationConfig(),
    }
    scheduler_roots = (
        (task.scheduler_process_id,)
        if mode is ArchitectureMode.HOST_SCHEDULER and task.scheduler_process_id is not None
        else ()
    )

    def execute_solver() -> ALNSResult:
        if mode is ArchitectureMode.CURRENT_STAGE052:
            return solve_alns(
                instance,
                **common,  # type: ignore[arg-type]
                native_kernel_config=NativeKernelConfig(),
                candidate_transaction_config=NativeCandidateTransactionConfig(),
            )
        if mode is ArchitectureMode.PYTHON_CANDIDATE_CONTROL:
            return solve_alns(
                instance,
                **common,  # type: ignore[arg-type]
                candidate_control_config=CandidateControlConfig(
                    worker_count=threads_per_shard,
                    stage052_calibrated_workers=profile_bound,
                ),
            )
        native_config = _native_config(
            mode,
            task,
            scheduler_socket_path=(
                task.scheduler_socket_path if mode is ArchitectureMode.HOST_SCHEDULER else None
            ),
            task_receipt_path=task_receipt_path,
        )
        native_config.validate_runtime_affinity(sorted(os.sched_getaffinity(0)))
        return solve_alns(
            instance,
            **common,  # type: ignore[arg-type]
            native_execution_config=native_config,
        )

    # Per-axis telemetry includes the shared scheduler so an individual raw
    # bundle never omits part of its execution process tree.  These shared-root
    # values must not be summed across concurrent shard axes; the parent
    # mode-wave observation is the aggregate accounting source.
    if monitor_enabled:
        with ProcessTreeMonitor(additional_root_pids=scheduler_roots) as resource_monitor:
            started = time.perf_counter()
            result = execute_solver()
            solver_seconds = time.perf_counter() - started
        resource_statistics = resource_monitor.statistics(
            elapsed_seconds=solver_seconds,
            compute_thread_limit=(
                total_compute_threads
                if not profile_bound or mode is ArchitectureMode.HOST_SCHEDULER
                else len(task.axis_cpu_ids)
            ),
        )
    else:
        started = time.perf_counter()
        result = execute_solver()
        solver_seconds = time.perf_counter() - started
        resource_statistics = {"telemetry_status": "disabled"}
    report = validate_routes(instance, [list(route) for route in result.routes])
    if not report.feasible or result.objective is None:
        raise RuntimeError("architecture axis returned an invalid or objective-less solution")
    if result.objective.key != SolutionObjective.from_report(instance, report).key:
        raise RuntimeError("architecture axis objective does not replay")
    axis_compute_limit = (
        total_compute_threads
        if not profile_bound or mode is ArchitectureMode.HOST_SCHEDULER
        else len(task.axis_cpu_ids)
    )
    topology: dict[str, object] = {
        "shard_processes": shard_processes,
        "threads_per_shard": threads_per_shard,
        "compute_thread_limit": total_compute_threads,
        "axis_compute_thread_limit": axis_compute_limit,
        "scheduler_threads": (
            len(task.scheduler_cpu_ids)
            if mode is ArchitectureMode.HOST_SCHEDULER and profile_bound
            else (24 if mode is ArchitectureMode.HOST_SCHEDULER else 0)
        ),
        "effective_native_search_threads": (
            (len(task.scheduler_cpu_ids) if profile_bound else TOTAL_COMPUTE_THREADS)
            if mode is ArchitectureMode.HOST_SCHEDULER
            else threads_per_shard
        ),
        "performance_profile_sha256": task.performance_profile_sha256,
        "performance_topology_key": task.topology_key,
        "configured_axis_cpu_ids": list(task.axis_cpu_ids),
        "configured_scheduler_cpu_ids": list(task.scheduler_cpu_ids),
        "scheduler_request_threads": task.scheduler_request_threads,
        "allow_affinity_overlap": task.allow_affinity_overlap,
        "shared_native_work_pool": result.native_execution_statistics.get(
            "shared_native_work_pool", False
        ),
        "process_id": os.getpid(),
        "scheduler_process_id": task.scheduler_process_id,
        "shared_scheduler_resource_attribution": (
            "mode_wave_primary_axis_values_overlap"
            if mode is ArchitectureMode.HOST_SCHEDULER
            else "not_applicable"
        ),
        "threads_before": threads_before,
        "threads_after": _thread_count(),
        "rss_bytes": _rss_bytes(),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "shared_scheduler_accounting": (
            "mode_wave" if mode is ArchitectureMode.HOST_SCHEDULER else "none"
        ),
        **resource_statistics,
    }
    return result, solver_seconds, topology


def _result_payload(
    task: ArchitectureAxisTask,
    mode: ArchitectureMode,
    result: ALNSResult,
    solver_seconds: float,
    topology: dict[str, object],
    semantic_journal: Mapping[str, object],
) -> dict[str, object]:
    assert result.objective is not None
    backend = result.backend_metrics
    screening = result.screening_statistics
    candidate_transactions = result.candidate_transaction_statistics
    measurement_evidence = _measurement_evidence(result)
    native_fallback = result.native_execution_statistics.get("fallback_count", 0)
    if isinstance(native_fallback, bool) or not isinstance(native_fallback, int):
        raise RuntimeError("native fallback evidence has an invalid schema")
    native_full_mode = mode in {
        ArchitectureMode.FULL_NATIVE_ALNS,
        ArchitectureMode.HOST_SCHEDULER,
    }
    candidate_control_complete = result.native_execution_statistics.get(
        "candidate_control_semantics_complete"
    )
    stage04_complete = result.native_execution_statistics.get("stage04_semantics_complete")
    instrumentation_complete = result.native_execution_statistics.get("instrumentation_complete")
    raw_candidate_trajectory_evidence = measurement_evidence.get("candidate_trajectory")
    if not isinstance(raw_candidate_trajectory_evidence, dict):
        raise RuntimeError("candidate trajectory evidence is missing")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_label": task.run_labels[mode.value],
        "scope": task.scope,
        "repeat": task.repeat,
        "axis": task.axis,
        "fixed_work_budget": {
            "axis": "fixed_work",
            "exact_calls": task.fixed_work_exact_calls,
            "iterations": task.fixed_work_max_iterations,
            "watchdog_seconds": task.fixed_work_watchdog_seconds,
            "batch_size": task.exact_batch_size,
        },
        "mode": mode.value,
        "instance": task.instance_name,
        "seed": task.seed,
        "status": "completed",
        "revision": task.revision,
        "wheel_sha256": task.wheel_sha256,
        "native_sha256": task.native_sha256,
        "scheduler_sha256": task.scheduler_sha256,
        "solver_seconds": solver_seconds,
        "objective": list(result.objective.key),
        "routes": [list(route) for route in result.routes],
        "customer_sequences": [list(route) for route in result.customer_sequences],
        "validator_passed": True,
        "iterations": result.iterations,
        "effective_iterations": result.effective_iterations,
        "termination_reason": result.termination_reason,
        "accepted_moves": result.accepted_moves,
        "rejected_moves": result.rejected_moves,
        "exact_started_calls": result.exact_started_calls,
        "exact_completed_calls": result.exact_completed_calls,
        "exact_interrupted_calls": result.exact_interrupted_calls,
        "candidate_work_hash": result.candidate_work_hash,
        "route_result_hash": result.route_result_hash,
        "fallback_count": native_fallback,
        "semantic_trajectory": {
            "schema_version": "stage05.2-external-semantic-trajectory-v1",
            "source": "canonical_semantic_journal:operator",
            **raw_candidate_trajectory_evidence,
        },
        "canonical_semantic_journal": dict(semantic_journal),
        "trajectory": _row_evidence(dict(event) for event in result.neighborhood_events),
        "operator_statistics": result.neighborhood_statistics,
        "operator_semantic_statistics": _semantic_operator_statistics(result),
        "stage04_statistics": result.stage04_statistics,
        "stage04_events": _row_evidence(dict(event) for event in result.stage04_event_log),
        "candidate_transaction_events": _row_evidence(
            dict(event) for event in result.candidate_transaction_events
        ),
        "candidate_control_statistics": result.candidate_control_statistics,
        "candidate_transaction_statistics": candidate_transactions,
        "native_execution_statistics": result.native_execution_statistics,
        "backend_metrics": backend,
        "screening_statistics": screening,
        "cache_incremental_statistics": result.cache_incremental_statistics,
        "measurement_evidence": measurement_evidence,
        "semantic_completeness": {
            "candidate_control": (candidate_control_complete is True if native_full_mode else True),
            "stage04": stage04_complete is True if native_full_mode else True,
            "measurement_trace": bool(measurement_evidence["present"])
            and (instrumentation_complete is True if native_full_mode else True),
        },
        "topology": topology,
        "throughput": {
            "effective_iterations_per_second": result.effective_iterations
            / max(solver_seconds, 1e-12),
            "exact_started_per_second": result.exact_started_calls / max(solver_seconds, 1e-12),
            "candidate_transactions_per_second": (
                _metric_int(candidate_transactions.get("transactions", 0))
                + _metric_int(candidate_transactions.get("native_candidate_transactions", 0))
            )
            / max(solver_seconds, 1e-12),
            "screened_routes_per_second": (
                _metric_int(screening.get("total_routes", 0))
                or _metric_int(screening.get("screening_calls", 0))
            )
            / max(solver_seconds, 1e-12),
        },
        "cache_memory_bytes": _metric_int(result.cache_incremental_statistics.get("bytes_peak", 0)),
    }


def _axis_path(task: ArchitectureAxisTask, mode: ArchitectureMode) -> Path:
    return (
        task.output_root
        / task.run_labels[mode.value]
        / "axes"
        / f"repeat{task.repeat + 1}"
        / task.axis
        / task.instance_name
        / f"{task.seed}.json"
    )


def _axis_publication_targets(
    path: Path,
    *,
    task_receipt_path: Path | None,
) -> tuple[Path, ...]:
    persistence_path = _axis_persistence_receipt_path(path)
    targets = [
        path,
        path.with_suffix(path.suffix + ".sha256"),
        persistence_path,
        persistence_path.with_suffix(persistence_path.suffix + ".sha256"),
        semantic_bundle_path(path),
    ]
    if task_receipt_path is not None:
        targets.extend(
            (
                task_receipt_path,
                task_receipt_path.with_suffix(task_receipt_path.suffix + ".sha256"),
            )
        )
    return tuple(targets)


def _assert_axis_publication_namespace_empty(
    path: Path,
    *,
    task_receipt_path: Path | None,
) -> None:
    collisions = [
        target
        for target in _axis_publication_targets(
            path,
            task_receipt_path=task_receipt_path,
        )
        if target.exists() or target.is_symlink()
    ]
    if collisions:
        raise FileExistsError(
            "axis publication namespace is not empty: "
            + ", ".join(sorted(target.name for target in collisions))
        )


def _rollback_axis_publication(
    created_targets: set[Path],
) -> None:
    """Remove every target created by one failed terminal axis transaction."""

    for target in sorted(created_targets, key=lambda item: len(item.parts), reverse=True):
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)


def _persist_axis_terminal_transaction(
    *,
    path: Path,
    payload: dict[str, object],
    semantic_journal: Mapping[str, object] | None,
    task_receipt_path: Path | None,
    journal_persistence_seconds: float,
    journal_persistence_wall_seconds: float,
    startup_seconds: float,
    axis_started: float,
    created_targets: set[Path],
) -> str:
    """Publish the complete terminal axis target set or roll it all back."""

    persistence_path = _axis_persistence_receipt_path(path)
    terminal_targets = {
        path,
        path.with_suffix(path.suffix + ".sha256"),
        persistence_path,
        persistence_path.with_suffix(persistence_path.suffix + ".sha256"),
    }
    try:
        collisions = [
            target
            for target in terminal_targets
            if target not in created_targets and (target.exists() or target.is_symlink())
        ]
        if collisions:
            raise FileExistsError(
                "axis terminal publication target already exists: "
                + ", ".join(sorted(target.name for target in collisions))
            )
        external_bytes = 0
        if semantic_journal is not None:
            bundle_bytes = semantic_journal.get("bundle_bytes")
            if isinstance(bundle_bytes, bool) or not isinstance(bundle_bytes, int):
                raise RuntimeError("semantic journal byte accounting is invalid")
            external_bytes = bundle_bytes
        if task_receipt_path is not None:
            task_receipt_sidecar = task_receipt_path.with_suffix(
                task_receipt_path.suffix + ".sha256"
            )
            task_exists = task_receipt_path.is_file() and not task_receipt_path.is_symlink()
            sidecar_exists = (
                task_receipt_sidecar.is_file() and not task_receipt_sidecar.is_symlink()
            )
            if task_exists != sidecar_exists:
                raise RuntimeError("native work-task receipt publication is partial")
            if payload.get("status") == "completed" and not task_exists:
                raise RuntimeError("native work-task receipt artifacts are missing")
            if task_exists:
                external_bytes += (
                    task_receipt_path.stat().st_size + task_receipt_sidecar.stat().st_size
                )
        payload["persistence_seconds"] = journal_persistence_seconds
        payload["semantic_journal_wall_seconds"] = journal_persistence_wall_seconds
        payload["startup_seconds"] = startup_seconds
        payload["producer_pre_primary_publication_seconds"] = time.perf_counter() - axis_started
        payload.pop("end_to_end_seconds", None)
        payload["persistence_receipt"] = _axis_persistence_descriptor(path)
        payload["persistence_breakdown"] = {
            "receipt": payload["persistence_receipt"],
        }
        _set_artifact_size(payload, external_bytes=external_bytes)
        try:
            primary_bytes, signed_json_persistence = _write_signed_json_with_receipt(
                path,
                payload,
            )
        finally:
            for target in (path, path.with_suffix(path.suffix + ".sha256")):
                if target.exists() and not target.is_symlink():
                    created_targets.add(target)
        if primary_bytes + external_bytes != payload["artifact_bytes"]:
            raise RuntimeError("primary artifact byte count does not reconcile")
        signed_json_total = signed_json_persistence["total_seconds"]
        if (
            isinstance(signed_json_total, bool)
            or not isinstance(signed_json_total, int | float)
            or not math.isfinite(float(signed_json_total))
            or float(signed_json_total) < 0.0
        ):
            raise RuntimeError("signed JSON persistence timing is invalid")
        persistence_seconds = journal_persistence_seconds + float(signed_json_total)
        end_to_end_seconds = time.perf_counter() - axis_started
        persistence_receipt_path = _axis_persistence_receipt_path(path)
        persistence_receipt: dict[str, object] = {
            "schema_version": AXIS_PERSISTENCE_RECEIPT_SCHEMA_VERSION,
            "axis_path": path.name,
            "axis_sha256": path.with_suffix(path.suffix + ".sha256")
            .read_text(encoding="ascii")
            .strip(),
            "axis_status": payload["status"],
            "persistence_seconds": persistence_seconds,
            "end_to_end_seconds": end_to_end_seconds,
            "primary_artifact_bytes": primary_bytes + external_bytes,
            "persistence_breakdown": {
                "semantic_journal": (
                    semantic_journal.get("persistence")
                    if semantic_journal is not None
                    else "unavailable"
                ),
                "signed_json": signed_json_persistence,
            },
            "timing_scope": (
                "producer_pre_receipt: primary axis JSON, semantic journal, sidecars, "
                "fsync, and atomic publish; receipt publication is excluded here and "
                "included only in parent-observed mode-block timing"
            ),
        }
        _set_axis_persistence_receipt_size(
            persistence_receipt,
            primary_artifact_bytes=primary_bytes + external_bytes,
        )
        try:
            receipt_bytes = _write_signed_json(
                persistence_receipt_path,
                persistence_receipt,
            )
        finally:
            for target in (
                persistence_receipt_path,
                persistence_receipt_path.with_suffix(persistence_receipt_path.suffix + ".sha256"),
            ):
                if target.exists() and not target.is_symlink():
                    created_targets.add(target)
        if receipt_bytes != persistence_receipt["persistence_receipt_bytes"]:
            raise RuntimeError("axis persistence receipt byte count does not reconcile")
        return str(path)
    except BaseException as primary_error:
        try:
            _rollback_axis_publication(created_targets)
        except BaseException as rollback_error:
            raise BaseExceptionGroup(
                "axis terminal publication and rollback failed",
                [primary_error, rollback_error],
            ) from None
        raise


def _run_mode(
    task: ArchitectureAxisTask,
    mode: ArchitectureMode,
    *,
    resource_telemetry_enabled: bool = True,
) -> str:
    axis_started = time.perf_counter()
    path = _axis_path(task, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    task_receipt_path = (
        path.with_suffix(path.suffix + ".native-work-tasks.jsonl")
        if mode
        in {
            ArchitectureMode.PER_SOLVE_RUNTIME,
            ArchitectureMode.FULL_NATIVE_ALNS,
        }
        else None
    )
    _assert_axis_publication_namespace_empty(
        path,
        task_receipt_path=task_receipt_path,
    )
    created_targets: set[Path] = set()
    semantic_journal: dict[str, object] | None = None
    journal_persistence_seconds = 0.0
    journal_persistence_wall_seconds = 0.0
    startup_seconds = 0.0
    result: ALNSResult | None = None
    payload: dict[str, object] | None = None
    failure: BaseException | None = None
    try:
        solve_call_started = time.perf_counter()
        result, solver_seconds, topology = _solve_mode(
            mode,
            task,
            resource_telemetry_enabled=resource_telemetry_enabled,
            task_receipt_path=task_receipt_path,
        )
        solve_call_seconds = time.perf_counter() - solve_call_started
        startup_seconds = max(0.0, solve_call_seconds - solver_seconds)
        journal_started = time.perf_counter()
        semantic_journal = write_semantic_journal(path, result)
        created_targets.add(semantic_bundle_path(path, semantic_journal))
        journal_persistence_wall_seconds = time.perf_counter() - journal_started
        journal_persistence = semantic_journal.get("persistence")
        if not isinstance(journal_persistence, Mapping):
            raise RuntimeError("semantic journal persistence attribution is missing")
        formal_attributed = journal_persistence.get("formal_attributed_seconds")
        if (
            isinstance(formal_attributed, bool)
            or not isinstance(formal_attributed, int | float)
            or not math.isfinite(float(formal_attributed))
            or float(formal_attributed) < 0.0
        ):
            raise RuntimeError("semantic journal formal attribution is invalid")
        journal_persistence_seconds = float(formal_attributed)
        payload = _result_payload(
            task,
            mode,
            result,
            solver_seconds,
            topology,
            semantic_journal,
        )
    except BaseException as error:
        failure = error
    finally:
        if task_receipt_path is not None:
            for target in (
                task_receipt_path,
                task_receipt_path.with_suffix(task_receipt_path.suffix + ".sha256"),
            ):
                if target.exists() and not target.is_symlink():
                    created_targets.add(target)
        if result is not None and result.measurement_trace is not None:
            try:
                result.measurement_trace.release_runtime_semantic_storage()
            except BaseException as cleanup_error:
                failure = (
                    cleanup_error
                    if failure is None
                    else BaseExceptionGroup(
                        "axis solve and runtime semantic cleanup failed",
                        [failure, cleanup_error],
                    )
                )
    if failure is not None:
        if semantic_journal is not None:
            bundle = semantic_bundle_path(path, semantic_journal)
            try:
                _rollback_axis_publication({bundle})
            except BaseException as cleanup_error:
                raise BaseExceptionGroup(
                    "axis failure and semantic bundle rollback failed",
                    [failure, cleanup_error],
                ) from None
            created_targets.discard(bundle)
            semantic_journal = None
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_label": task.run_labels[mode.value],
            "scope": task.scope,
            "repeat": task.repeat,
            "axis": task.axis,
            "mode": mode.value,
            "instance": task.instance_name,
            "seed": task.seed,
            "status": "failed",
            "revision": task.revision,
            "wheel_sha256": task.wheel_sha256,
            "native_sha256": task.native_sha256,
            "scheduler_sha256": task.scheduler_sha256,
            "error_type": type(failure).__name__,
            "error": str(failure),
        }
    if payload is None:
        raise AssertionError("axis terminal payload was not constructed")
    terminal_path = _persist_axis_terminal_transaction(
        path=path,
        payload=payload,
        semantic_journal=semantic_journal,
        task_receipt_path=task_receipt_path,
        journal_persistence_seconds=journal_persistence_seconds,
        journal_persistence_wall_seconds=journal_persistence_wall_seconds,
        startup_seconds=startup_seconds,
        axis_started=axis_started,
        created_targets=created_targets,
    )
    if failure is not None:
        raise ArchitectureAxisExecutionFailed(
            terminal_path,
            type(failure).__name__,
            str(failure),
        )
    return terminal_path


def _run_group(task: ArchitectureAxisTask) -> list[str]:
    """Run one legacy test group; production campaigns use global mode waves."""

    return [_run_mode(task, mode) for mode in rotated_modes(task)]


def _latency_histogram_quantiles(
    histogram: object,
    *,
    maximum_seconds: object,
) -> dict[str, float | str]:
    if (
        not isinstance(histogram, list)
        or len(histogram) != 32
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in histogram
        )
        or isinstance(maximum_seconds, bool)
        or not isinstance(maximum_seconds, (int, float))
        or not math.isfinite(float(maximum_seconds))
        or float(maximum_seconds) < 0.0
    ):
        raise RuntimeError("native scheduler latency histogram is invalid")
    total = sum(histogram)
    if total == 0:
        return {
            "p50": "unavailable",
            "p95": "unavailable",
            "p99": "unavailable",
        }
    quantiles: dict[str, float | str] = {}
    for label, fraction in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
        target = max(1, math.ceil(total * fraction))
        cumulative = 0
        selected_bin = 31
        for index, count in enumerate(histogram):
            cumulative += count
            if cumulative >= target:
                selected_bin = index
                break
        if selected_bin == 31:
            estimate = float(maximum_seconds)
        else:
            upper_bound_seconds = (2 ** (selected_bin + 1)) / 1_000_000.0
            estimate = min(upper_bound_seconds, float(maximum_seconds))
        quantiles[label] = estimate
    return quantiles


def _scheduler_runtime_summary(
    runtime_statistics: dict[str, object],
) -> dict[str, object]:
    enriched = dict(runtime_statistics)
    latency: dict[str, object] = {}
    for queue_name in ("request_queue", "work_queue"):
        queue = runtime_statistics.get(queue_name)
        if not isinstance(queue, dict):
            raise RuntimeError("native scheduler queue receipt is missing")
        for counter in ("queue_full_count", "rejected_count"):
            if queue.get(counter) != 0:
                raise RuntimeError(f"native scheduler {queue_name} {counter} must remain zero")
        if queue.get("pending") != 0 or (queue_name == "work_queue" and queue.get("active") != 0):
            raise RuntimeError(f"native scheduler {queue_name} was not drained")
        latency[queue_name] = {
            "wait_seconds": _latency_histogram_quantiles(
                queue.get("wait_histogram"),
                maximum_seconds=queue.get("maximum_wait_seconds"),
            ),
            "service_seconds": _latency_histogram_quantiles(
                queue.get("service_histogram"),
                maximum_seconds=queue.get("maximum_service_seconds"),
            ),
        }
    enriched["latency_quantiles"] = latency
    return enriched


def _mode_wave_batches(
    plan: tuple[ArchitectureAxisTask, ...],
) -> tuple[tuple[ArchitectureAxisTask, ...], ...]:
    return tuple(
        tuple(plan[offset : offset + SHARD_PROCESSES])
        for offset in range(0, len(plan), SHARD_PROCESSES)
    )


def _configure_compute_envelope(
    runtime_binding: FrozenRuntimeBinding | None = None,
) -> dict[str, object]:
    available = sorted(os.sched_getaffinity(0))
    if runtime_binding is None:
        if len(available) < TOTAL_COMPUTE_THREADS:
            raise RuntimeError("Stage 5.2 comparison host exposes fewer than 24 logical CPUs")
        selected = available[:TOTAL_COMPUTE_THREADS]
    else:
        selected = list(runtime_binding.allowed_cpu_ids)
        if available != selected:
            raise RuntimeError("runtime CPU affinity differs from the frozen profile")
    os.sched_setaffinity(0, selected)
    thread_environment = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    os.environ.update(thread_environment)
    return {
        "available_logical_cpus": available,
        "selected_logical_cpus": selected,
        "selected_logical_cpu_count": len(selected),
        "thread_environment": thread_environment,
    }


def _require_campaign_identity(
    root: Path,
    *,
    continuity_lease_token: str,
    expected_revision: str,
) -> None:
    """Revalidate the single writer and frozen checkout around every mode wave."""

    require_owned(
        root,
        token=continuity_lease_token,
        allowed_phases=frozenset({"paired-campaign", "pilot-campaign"}),
    )
    observed_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_revision != expected_revision:
        raise RuntimeError("native architecture campaign Git revision changed")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("native architecture campaign worktree changed")


def _publish_campaign_failure_receipts(
    *,
    scope: str,
    attempt: int,
    output_root: Path,
    labels: Mapping[str, str],
    revision: str,
    performance_profile_sha256: str,
    written: Iterable[str],
    block_index: int,
    mode: ArchitectureMode,
    subwave_index: int,
    failure: BaseException,
) -> None:
    """Seal one immutable fail-fast campaign snapshot in every mode namespace."""

    root = output_root.resolve()

    def portable_axis_path(value: str) -> str:
        path = Path(value).resolve()
        try:
            return path.relative_to(root).as_posix()
        except ValueError as error:
            raise RuntimeError("campaign failure axis escaped the output root") from error

    completed_axes = sorted({portable_axis_path(value) for value in written})
    trigger_axis = (
        portable_axis_path(failure.axis_path)
        if isinstance(failure, ArchitectureAxisExecutionFailed)
        else None
    )
    payload = {
        "schema_version": "stage05.2-native-architecture-campaign-failure-v1",
        "scope": scope,
        "attempt": attempt,
        "status": "failed",
        "revision": revision,
        "performance_profile_sha256": performance_profile_sha256,
        "identity_block_index": block_index,
        "mode": mode.value,
        "subwave_index": subwave_index,
        "error_type": type(failure).__name__,
        "error": str(failure),
        "trigger_axis": trigger_axis,
        "completed_axis_count_at_failure": len(completed_axes),
        "completed_axes_at_failure": completed_axes,
        "expected_axis_count": expected_axis_count(scope),
        "snapshot_scope": "first-observed-failure-after-running-futures-drained",
        "formal_started": False,
        "cuda_started": False,
        "default_architecture_switched": False,
        "failed_unix": time.time(),
    }
    publication_errors: list[BaseException] = []
    for label in labels.values():
        try:
            _write_signed_json(output_root / label / "run_failure.json", payload)
        except BaseException as error:
            publication_errors.append(error)
    if publication_errors:
        raise BaseExceptionGroup(
            "campaign failure receipt publication failed",
            [failure, *publication_errors],
        )


def run_experiment(
    scope: str,
    *,
    attempt: int,
    output_root: Path,
    wheel_path: Path,
    warm_start_bundle_path: Path,
    continuity_lease_token: str,
    performance_profile_path: Path | None = None,
    calibration_review_path: Path | None = None,
    calibration_review_execution_path: Path | None = None,
    calibration_run_label: str | None = None,
    max_workers: int | None = None,
    paired_review_path: Path | None = None,
    paired_review_execution_path: Path | None = None,
    paired_attempt: int | None = None,
) -> dict[str, object]:
    root = repository_root()
    require_owned(
        root,
        token=continuity_lease_token,
        allowed_phases=frozenset({"paired-campaign", "pilot-campaign"}),
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("native architecture experiment requires a clean worktree")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    committed_attestation = committed_source_attestation(root, revision)
    expected_source_entries = committed_wheel_project_entry_sha256(root, revision)
    source_manifest_sha256 = cast(
        str,
        committed_attestation["source_manifest_sha256"],
    )
    tracked_file_count = cast(int, committed_attestation["tracked_file_count"])
    if not wheel_path.is_file():
        raise FileNotFoundError("the frozen comparison wheel does not exist")
    wheel_receipt = _verify_installed_wheel(
        wheel_path,
        expected_revision=revision,
        expected_tree=tree,
        expected_source_manifest_sha256=source_manifest_sha256,
        expected_tracked_file_count=tracked_file_count,
        expected_source_entries=expected_source_entries,
    )
    native_capabilities = _require_native_architecture_capabilities()
    if performance_profile_path is None:
        raise RuntimeError("native architecture campaign requires a frozen performance profile")
    performance_profile = _load_signed_performance_profile(performance_profile_path)
    runtime_binding = performance_profile.bind_runtime(wheel_receipt)
    profile_host_receipt = runtime_binding.host_receipt()
    if (
        calibration_review_path is None
        or calibration_review_execution_path is None
        or calibration_run_label is None
    ):
        raise RuntimeError(
            "native architecture campaign requires a signed qualified calibration review"
        )
    performance_profile_file_sha256 = _sha256_path(performance_profile_path)
    calibration_review_gate = _load_qualified_calibration_review(
        calibration_review_path,
        review_execution_path=calibration_review_execution_path,
        calibration_run_label=calibration_run_label,
        revision=revision,
        git_tree=tree,
        source_manifest_sha256=source_manifest_sha256,
        wheel_sha256=wheel_receipt["wheel_sha256"],
        native_sha256=wheel_receipt["native_sha256"],
        scheduler_sha256=wheel_receipt["scheduler_sha256"],
        performance_profile_path=performance_profile_path,
        performance_profile_file_sha256=performance_profile_file_sha256,
        performance_profile_sha256=performance_profile.canonical_sha256,
        selected_build_profile=runtime_binding.selected_build.name,
    )
    paired_review_gate: dict[str, object] | None = None
    if scope == "pilot":
        if (
            paired_review_path is None
            or paired_review_execution_path is None
            or paired_attempt is None
        ):
            raise RuntimeError(
                "pilot requires a signed qualified paired review and execution receipt"
            )
        paired_review_gate = _load_qualified_paired_review(
            paired_review_path,
            review_execution_path=paired_review_execution_path,
            paired_results_root=output_root,
            paired_attempt=paired_attempt,
            revision=revision,
            wheel_sha256=wheel_receipt["wheel_sha256"],
            native_sha256=wheel_receipt["native_sha256"],
            scheduler_sha256=wheel_receipt["scheduler_sha256"],
            performance_profile_sha256=performance_profile.canonical_sha256,
        )
    elif (
        paired_review_path is not None
        or paired_review_execution_path is not None
        or paired_attempt is not None
    ):
        raise RuntimeError("paired review input is valid only for Pilot")
    compute_envelope = _configure_compute_envelope(runtime_binding)
    configured_executor_workers = runtime_binding.max_executor_workers
    if max_workers is not None and max_workers != configured_executor_workers:
        raise ValueError("max_workers differs from the frozen performance profile")
    executor_workers = configured_executor_workers
    native_path = Path(wheel_receipt["native_path"])
    warm_starts = load_warm_start_bundle(
        warm_start_bundle_path,
        benchmark_dir=root / "data" / "schneider",
    )
    labels = run_labels_for_scope(scope, attempt)
    for label in labels.values():
        if (output_root / label).exists():
            raise FileExistsError(f"run label already exists and cannot be reused: {label}")
    scheduler_path = output_root / f".native-scheduler-{scope}-attempt{attempt:02d}.sock"
    plan = build_axis_plan(
        scope,
        attempt=attempt,
        benchmark_dir=root / "data" / "schneider",
        output_root=output_root,
        scheduler_socket_path=str(scheduler_path),
        wheel_sha256=wheel_receipt["wheel_sha256"],
        native_sha256=wheel_receipt["native_sha256"],
        scheduler_sha256=wheel_receipt["scheduler_sha256"],
        revision=revision,
        warm_starts=warm_starts,
    )
    for label in labels.values():
        (output_root / label).mkdir(parents=True)
    started = time.time()
    written: list[str] = []
    scheduler_observations: list[dict[str, object]] = []
    scheduler_observations_by_pid: dict[int, dict[str, object]] = {}
    mode_wave_resources: list[dict[str, object]] = []
    scheduler_startup_seconds = 0.0
    scheduler_shutdown_seconds = 0.0
    identity_blocks = _performance_identity_blocks(plan)
    for block_index, identity_block in enumerate(identity_blocks):
        workload_class = workload_class_for_instance(identity_block[0].instance_name)
        performance_family = performance_family_for_instance(identity_block[0].instance_name)
        if any(
            workload_class_for_instance(task.instance_name) != workload_class
            or performance_family_for_instance(task.instance_name) != performance_family
            for task in identity_block
        ):
            raise RuntimeError("performance identity block mixes workload classes or families")
        offset = block_index % len(MODES)
        mode_order = MODES[offset:] + MODES[:offset]
        for mode in mode_order:
            _require_campaign_identity(
                root,
                continuity_lease_token=continuity_lease_token,
                expected_revision=revision,
            )
            topology_key = performance_topology_key(mode, workload_class)
            topology = runtime_binding.topology_for(mode.value, workload_class)
            task_batches = _mode_task_batches(
                identity_block,
                topology,
                profile_sha256=performance_profile.canonical_sha256,
                topology_key=topology_key,
            )
            lifecycle = (
                runtime_binding.scheduler_lifecycle_for(workload_class)
                if mode is ArchitectureMode.HOST_SCHEDULER
                else "not-applicable"
            )
            scheduler_process_ids: list[int] = []
            axis_parent_terminal_timings: list[dict[str, object]] = []
            mode_scheduler_startup = 0.0
            mode_scheduler_shutdown = 0.0
            mode_started = time.perf_counter()
            cgroup_before = _runtime_cgroup_snapshot()
            if cgroup_before.get("status") != "available":
                raise RuntimeError("runtime cgroup resource accounting is unavailable")
            with ProcessTreeMonitor() as mode_monitor:
                shared_scheduler: NativeHostScheduler | None = None

                def start_scheduler(
                    subwave_index: int,
                    *,
                    frozen_topology: ExecutionTopology = topology,
                    current_block_index: int = block_index,
                    current_lifecycle: str = lifecycle,
                    process_ids: list[int] = scheduler_process_ids,
                ) -> NativeHostScheduler:
                    nonlocal mode_scheduler_startup, scheduler_startup_seconds
                    lifecycle_component = (
                        "mode-block" if subwave_index < 0 else f"subwave-{subwave_index + 1:03d}"
                    )
                    task_receipt_path = (
                        output_root
                        / labels[ArchitectureMode.HOST_SCHEDULER.value]
                        / (
                            f"scheduler-task-receipts-block-{current_block_index:04d}-"
                            f"{lifecycle_component}.jsonl"
                        )
                    )
                    scheduler = NativeHostScheduler(
                        scheduler_path,
                        worker_threads=len(frozen_topology.scheduler_cpu_ids),
                        request_threads=frozen_topology.request_threads,
                        cpu_affinity=frozen_topology.scheduler_cpu_ids,
                        task_receipt_path=task_receipt_path,
                    )
                    scheduler_started = time.perf_counter()
                    scheduler.start()
                    startup = time.perf_counter() - scheduler_started
                    mode_scheduler_startup += startup
                    scheduler_startup_seconds += startup
                    process_ids.append(scheduler.process_id)
                    observation: dict[str, object] = {
                        "identity_block_index": current_block_index,
                        "subwave_index": subwave_index,
                        "lifecycle": current_lifecycle,
                        "process_id": scheduler.process_id,
                        "observed_thread_count": scheduler.observed_thread_count(),
                        "configured_worker_threads": scheduler.worker_threads,
                        "configured_request_threads": scheduler.request_thread_count,
                        "observed_request_threads": (scheduler.observed_request_thread_count()),
                        "observed_receipt_writer_threads": (
                            scheduler.observed_task_receipt_writer_thread_count()
                        ),
                        "configured_cpu_affinity": list(frozen_topology.scheduler_cpu_ids),
                        "observed_cpu_affinity": list(scheduler.observed_cpu_affinity()),
                    }
                    scheduler_observations.append(observation)
                    scheduler_observations_by_pid[scheduler.process_id] = observation
                    return scheduler

                def stop_scheduler(scheduler: NativeHostScheduler) -> None:
                    nonlocal mode_scheduler_shutdown, scheduler_shutdown_seconds
                    scheduler_pid = scheduler.process_id
                    shutdown_started = time.perf_counter()
                    scheduler.close()
                    shutdown = time.perf_counter() - shutdown_started
                    mode_scheduler_shutdown += shutdown
                    scheduler_shutdown_seconds += shutdown
                    observation = scheduler_observations_by_pid[scheduler_pid]
                    observation["runtime_statistics"] = _scheduler_runtime_summary(
                        scheduler.runtime_statistics
                    )
                    observation["shutdown_seconds"] = shutdown

                with ProcessPoolExecutor(
                    max_workers=topology.shard_count,
                    mp_context=multiprocessing.get_context("spawn"),
                    max_tasks_per_child=1,
                ) as executor:
                    if mode is ArchitectureMode.HOST_SCHEDULER and lifecycle == "mode-block":
                        shared_scheduler = start_scheduler(-1)
                    try:
                        for subwave_index, task_batch in enumerate(task_batches):
                            scheduler = shared_scheduler
                            if mode is ArchitectureMode.HOST_SCHEDULER and lifecycle == "per-wave":
                                scheduler = start_scheduler(subwave_index)
                            scheduler_process_id = (
                                None if scheduler is None else scheduler.process_id
                            )
                            try:
                                futures: dict[
                                    Future[str],
                                    tuple[float, ArchitectureAxisTask],
                                ] = {}
                                for task in task_batch:
                                    submitted = time.perf_counter()
                                    future = executor.submit(
                                        _run_mode,
                                        replace(
                                            task,
                                            scheduler_process_id=scheduler_process_id,
                                        ),
                                        mode,
                                    )
                                    futures[future] = (submitted, task)
                                try:
                                    for future in as_completed(futures):
                                        submitted, task = futures[future]
                                        path = future.result()
                                        written.append(path)
                                        axis_parent_terminal_timings.append(
                                            {
                                                "repeat": task.repeat,
                                                "axis": task.axis,
                                                "instance": task.instance_name,
                                                "seed": task.seed,
                                                "subwave_index": subwave_index,
                                                "producer_parent_terminal_seconds": (
                                                    time.perf_counter() - submitted
                                                ),
                                            }
                                        )
                                except BaseException as error:
                                    if isinstance(error, ArchitectureAxisExecutionFailed):
                                        written.append(error.axis_path)
                                    for pending in futures:
                                        pending.cancel()
                                    # Keep the shared scheduler alive while already-running
                                    # clients finish their bounded rollback/publication path.
                                    # No later subwave or mode is submitted after this point.
                                    executor.shutdown(wait=True, cancel_futures=True)
                                    _publish_campaign_failure_receipts(
                                        scope=scope,
                                        attempt=attempt,
                                        output_root=output_root,
                                        labels=labels,
                                        revision=revision,
                                        performance_profile_sha256=(
                                            performance_profile.canonical_sha256
                                        ),
                                        written=written,
                                        block_index=block_index,
                                        mode=mode,
                                        subwave_index=subwave_index,
                                        failure=error,
                                    )
                                    raise RuntimeError(
                                        "native architecture campaign stopped at the first "
                                        "failed axis"
                                    ) from error
                            finally:
                                if scheduler is not None and scheduler is not shared_scheduler:
                                    stop_scheduler(scheduler)
                    finally:
                        if shared_scheduler is not None:
                            stop_scheduler(shared_scheduler)
            mode_elapsed = time.perf_counter() - mode_started
            mode_resource_statistics = mode_monitor.statistics(
                elapsed_seconds=mode_elapsed,
                compute_thread_limit=len(topology.cpu_ids),
            )
            cgroup_after = _runtime_cgroup_snapshot()
            if cgroup_after.get("status") != "available" or cgroup_after.get(
                "cgroup_path"
            ) != cgroup_before.get("cgroup_path"):
                raise RuntimeError("runtime cgroup resource identity changed")
            swap_counters = tuple(
                snapshot.get(field)
                for snapshot in (cgroup_before, cgroup_after)
                for field in (
                    "memory_swap_current_bytes",
                    "memory_swap_peak_bytes",
                )
            )
            if any(
                isinstance(value, bool) or not isinstance(value, int) for value in swap_counters
            ):
                raise RuntimeError("runtime cgroup swap accounting is unavailable")
            if any(value != 0 for value in swap_counters):
                raise RuntimeError("runtime cgroup used swap")
            memory_capacity = runtime_binding.memory_capacity_bytes
            memory_gate_bytes = math.floor(memory_capacity * 0.80)
            cgroup_peak = cgroup_after.get("memory_peak_bytes")
            aggregate_rss = mode_resource_statistics.get("peak_aggregate_rss_bytes")
            if (
                isinstance(cgroup_peak, bool)
                or not isinstance(cgroup_peak, int)
                or isinstance(aggregate_rss, bool)
                or not isinstance(aggregate_rss, int)
            ):
                raise RuntimeError("runtime peak memory accounting is unavailable")
            if cgroup_peak > memory_gate_bytes or aggregate_rss > memory_gate_bytes:
                raise RuntimeError("runtime memory use exceeded the 80% safety gate")
            memory_event_deltas = {
                field: _cgroup_counter_delta(cgroup_before, cgroup_after, field)
                for field in ("oom", "oom_kill")
            }
            if any(isinstance(value, str) or value != 0 for value in memory_event_deltas.values()):
                raise RuntimeError("runtime cgroup OOM accounting gate failed")
            io_deltas = _cgroup_io_deltas(cgroup_before, cgroup_after)
            if isinstance(io_deltas, str):
                raise RuntimeError("runtime cgroup I/O accounting is unavailable")
            raw_process_tree_cpu_seconds = mode_resource_statistics.get("process_tree_cpu_seconds")
            if (
                isinstance(raw_process_tree_cpu_seconds, bool)
                or not isinstance(raw_process_tree_cpu_seconds, int | float)
                or not math.isfinite(float(raw_process_tree_cpu_seconds))
                or float(raw_process_tree_cpu_seconds) < 0.0
            ):
                raise RuntimeError("runtime process-tree CPU accounting is invalid")
            process_tree_cpu_seconds = float(raw_process_tree_cpu_seconds)
            if len(axis_parent_terminal_timings) != len(identity_block):
                raise RuntimeError("mode-wave parent terminal timing inventory is incomplete")
            _require_campaign_identity(
                root,
                continuity_lease_token=continuity_lease_token,
                expected_revision=revision,
            )
            mode_wave_resources.append(
                {
                    "batch_index": block_index,
                    "identity_block_index": block_index,
                    "mode": mode.value,
                    "workload_class": workload_class,
                    "performance_family": performance_family,
                    "axis_count": len(identity_block),
                    "subwave_count": len(task_batches),
                    "worker_process_lifecycle": "one_shard_per_spawned_process",
                    "worker_multiprocessing_start_method": "spawn",
                    "worker_max_tasks_per_child": 1,
                    "scheduler_lifecycle": lifecycle,
                    "scheduler_process_id": (
                        scheduler_process_ids[0] if len(scheduler_process_ids) == 1 else None
                    ),
                    "scheduler_process_ids": scheduler_process_ids,
                    "scheduler_runtime_statistics": [
                        scheduler_observations_by_pid[process_id]["runtime_statistics"]
                        for process_id in scheduler_process_ids
                    ],
                    "scheduler_startup_seconds": mode_scheduler_startup,
                    "scheduler_shutdown_seconds": mode_scheduler_shutdown,
                    "performance_profile_sha256": (performance_profile.canonical_sha256),
                    "performance_topology_key": topology_key,
                    "execution_topology": topology.to_dict(),
                    "identities": [
                        {
                            "repeat": task.repeat,
                            "axis": task.axis,
                            "instance": task.instance_name,
                            "seed": task.seed,
                        }
                        for task in identity_block
                    ],
                    "axis_parent_terminal_timings": sorted(
                        axis_parent_terminal_timings,
                        key=lambda row: (
                            cast(int, row["repeat"]),
                            cast(str, row["axis"]),
                            cast(str, row["instance"]),
                            cast(int, row["seed"]),
                        ),
                    ),
                    "elapsed_seconds": mode_elapsed,
                    "axes_per_hour": 3600.0 * len(identity_block) / mode_elapsed,
                    "effective_cores": process_tree_cpu_seconds / mode_elapsed,
                    "cpu_utilization_fraction_of_compute_limit": (
                        process_tree_cpu_seconds / (mode_elapsed * len(topology.cpu_ids))
                    ),
                    "cgroup_before": cgroup_before,
                    "cgroup_after": cgroup_after,
                    "cgroup_memory_event_deltas": memory_event_deltas,
                    "cgroup_io_deltas": io_deltas,
                    "memory_gate_bytes": memory_gate_bytes,
                    **mode_resource_statistics,
                }
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "attempt": attempt,
        "run_labels": labels,
        "revision": revision,
        "git_tree": tree,
        "wheel_path": wheel_receipt["wheel_path"],
        "wheel_sha256": wheel_receipt["wheel_sha256"],
        "wheel_receipt": wheel_receipt,
        "native_architecture_capabilities": native_capabilities,
        "native_path": str(native_path.resolve()),
        "native_sha256": wheel_receipt["native_sha256"],
        "scheduler_path": wheel_receipt["scheduler_path"],
        "scheduler_sha256": wheel_receipt["scheduler_sha256"],
        "performance_profile_path": str(performance_profile_path.resolve()),
        "performance_profile_file_sha256": performance_profile_file_sha256,
        "performance_profile_sha256": performance_profile.canonical_sha256,
        "performance_profile": performance_profile.to_dict(),
        "performance_profile_host_receipt": profile_host_receipt,
        "calibration_review_gate": calibration_review_gate,
        "paired_review_gate": paired_review_gate,
        "warm_start_bundle_path": str(warm_start_bundle_path.resolve()),
        "warm_start_bundle_sha256": _sha256_path(warm_start_bundle_path),
        "axis_count": len(written),
        "expected_axis_count": expected_axis_count(scope),
        "started_unix": started,
        "completed_unix": time.time(),
        "topology": {
            "executor_worker_limit": executor_workers,
            "compute_thread_limit": len(runtime_binding.allowed_cpu_ids),
            "compute_envelope": compute_envelope,
            "frozen_topologies": {
                key: topology.to_dict()
                for key, topology in sorted(runtime_binding.topologies.items())
            },
            "scheduler_observed": scheduler_observations,
            "scheduler_startup_seconds": scheduler_startup_seconds,
            "scheduler_shutdown_seconds": scheduler_shutdown_seconds,
            "mode_wave_resources": mode_wave_resources,
        },
        "mode_order_policy": ("identity_blocks_rotated_by_mode_with_profile_frozen_subwaves"),
        "formal_started": False,
    }
    if manifest["axis_count"] != manifest["expected_axis_count"]:
        raise RuntimeError("native architecture experiment axis count is incomplete")
    for label in labels.values():
        _write_signed_json(output_root / label / "run_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("paired", "pilot"))
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--warm-start-bundle", type=Path, required=True)
    parser.add_argument("--performance-profile", type=Path, required=True)
    parser.add_argument("--calibration-review", type=Path, required=True)
    parser.add_argument("--calibration-review-execution", type=Path, required=True)
    parser.add_argument("--calibration-run-label", required=True)
    parser.add_argument("--continuity-lease-token", required=True)
    parser.add_argument("--paired-review", type=Path)
    parser.add_argument("--paired-review-execution", type=Path)
    parser.add_argument("--paired-attempt", type=int)
    arguments = parser.parse_args(argv)
    manifest = run_experiment(
        arguments.scope,
        attempt=arguments.attempt,
        output_root=arguments.output_root,
        wheel_path=arguments.wheel,
        warm_start_bundle_path=arguments.warm_start_bundle,
        continuity_lease_token=arguments.continuity_lease_token,
        performance_profile_path=arguments.performance_profile,
        calibration_review_path=arguments.calibration_review,
        calibration_review_execution_path=arguments.calibration_review_execution,
        calibration_run_label=arguments.calibration_run_label,
        paired_review_path=arguments.paired_review,
        paired_review_execution_path=arguments.paired_review_execution,
        paired_attempt=arguments.paired_attempt,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ArchitectureAxisTask",
    "ArchitectureMode",
    "AXIS_NAMES",
    "MODES",
    "PAIRED_INSTANCES",
    "SCHEMA_VERSION",
    "SEEDS",
    "build_axis_plan",
    "expected_axis_count",
    "load_warm_start_bundle",
    "_load_qualified_calibration_review",
    "_load_qualified_paired_review",
    "NATIVE_ARCHITECTURE_CAPABILITY_NAMES",
    "performance_family_for_instance",
    "rotated_modes",
    "run_experiment",
    "run_labels_for_scope",
)
