"""Stage 5.2 performance and benchmark experiment runner."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import re
import resource
import shutil
import statistics
import subprocess
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass, fields, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Protocol

from evrptw.alns import ALNSResult, solve_alns
from evrptw.artifacts import (
    ARTIFACT_STORAGE_V2,
    ArtifactBundleWriter,
    ArtifactIntegrityError,
    ArtifactReader,
    ArtifactRunContext,
    ArtifactStorageConfig,
    aggregate_diagnostic_events,
    atomic_write_signed_json,
    build_stage03_critical_events,
    iter_stage03_critical_events,
    signed_sidecar_matches,
)
from evrptw.best_known import BEST_KNOWN_VALUES
from evrptw.environment import collect_environment
from evrptw.exact_deadline import ExactDeadlineConfig
from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES
from evrptw.experiments.stage02_route_reduction import (
    load_config as load_stage02_config,
)
from evrptw.experiments.stage04_weights import load_stage04_config
from evrptw.measurement import (
    MeasurementConfig,
    MeasurementTraceSink,
    RouteEvaluationTrace,
    ScreeningDecision,
)
from evrptw.models import Instance
from evrptw.native_kernels import NativeKernelConfig
from evrptw.parser import parse_schneider
from evrptw.repository import repository_root
from evrptw.stage052 import (
    PerformanceObservation,
    Stage052Component,
    formal_budget_matrix,
    stage052_contract,
)
from evrptw.stage052_accelerator import (
    MetalPilotStatus,
    SubprocessMetalPilotExecutor,
    run_conditional_metal_pilot,
)
from evrptw.stage052_campaign import (
    CHECKPOINT_SECONDS,
    AcceptedGlobalBest,
    AnytimeCheckpoint,
    BatchManifest,
    BenchmarkCampaignConfig,
    CampaignManifest,
    ObjectiveKey,
    StorageRoot,
    StorageRootLocator,
    VolumeIdentity,
    directory_byte_count,
    directory_checksum,
    load_batch_manifest,
    load_campaign_manifest,
)
from evrptw.stage052_campaign_runner import (
    ArchivedBatchStateWriteError,
    BatchRuntimeMonitor,
    MacMachineSnapshotSource,
    RollingCampaignCapacityError,
    archive_verified_batch_with_evidence,
    campaign_control_paths,
    collect_preflight_observation,
    free_bytes,
    load_benchmark_execution_lock,
    load_pilot_storage_observations,
    persist_campaign_manifest,
    probe_volume_identity,
    validate_batch_measurements,
    verify_campaign_root_locations,
    verify_rolling_campaign_capacity,
)
from evrptw.stage052_evidence import (
    BatchPersistenceEnvelope,
    PersistenceInterval,
    ProcessTreeResourceSampler,
    RunResourceSummary,
    Stage052PersistenceAttribution,
    abort_process_executor,
    collect_performance_provenance,
    stage052_storage_root_binding,
    verify_job_parallel_selection,
    verify_stage052_evidence_input,
    verify_stage052_runtime_identity,
)
from evrptw.stage052_remediation import (
    Stage052RemediationConfig,
    Stage052RemediationResult,
    remediate_stage052_artifacts,
)
from evrptw.validation import validate_routes

STAGE052_SCHEMA_VERSION = "stage05.2-performance-v1"
PERFORMANCE_INSTANCES = ("c101C5", "c101_21", "r101_21", "rc101_21")
PERFORMANCE_SEEDS = (2014, 2015, 2016)
FORMAL_SEEDS = tuple(range(2014, 2024))

PER_RUN_FIELDS = (
    "instance",
    "seed",
    "axis",
    "customer_count",
    "component",
    "backend",
    "worker_count",
    "storage_policy_version",
    "persistence_attribution",
    "solver_seconds",
    "artifact_persistence_seconds",
    "end_to_end_seconds",
    "screening_seconds",
    "exact_seconds",
    "packing_seconds",
    "unpacking_seconds",
    "native_kernel_seconds",
    "native_invocations",
    "native_fallbacks",
    "native_screening_seconds",
    "native_screening_invocations",
    "native_propagation_seconds",
    "native_propagation_invocations",
    "native_protocol_fallbacks",
    "exact_started_calls",
    "exact_completed_calls",
    "effective_iterations",
    "batch_launches",
    "median_batch_occupancy",
    "peak_rss_bytes",
    "vehicle_count",
    "total_distance",
    "total_charging_time",
    "charging_count",
    "validator_passed",
    "semantic_digest",
    "termination_reason",
    "failure_status",
)


class _PersistenceRecorder:
    """Record non-overlapping primary active-write intervals."""

    def __init__(self) -> None:
        self._intervals: list[PersistenceInterval] = []

    @contextlib.contextmanager
    def record(self, label: str) -> Iterator[None]:
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            self._intervals.append(
                PersistenceInterval(
                    label=label,
                    started_ns=started_ns,
                    completed_ns=time.perf_counter_ns(),
                )
            )

    @property
    def intervals(self) -> tuple[PersistenceInterval, ...]:
        return tuple(self._intervals)

    def append(self, interval: PersistenceInterval) -> None:
        if self._intervals and interval.started_ns < self._intervals[-1].completed_ns:
            raise RuntimeError("persistence intervals must be appended in clock order")
        if any(item.label == interval.label for item in self._intervals):
            raise RuntimeError(f"duplicate persistence interval label: {interval.label}")
        self._intervals.append(interval)


@dataclass(frozen=True, slots=True)
class _VerifiedBatchExecution:
    rows: list[dict[str, object]]
    checkpoints: list[dict[str, object]]
    batch: BatchManifest
    base_attribution: Stage052PersistenceAttribution
    verified_manifest_sha256: str
    verified_manifest_write_interval: PersistenceInterval


def _write_persistence_attribution(
    *,
    run_dir: Path,
    run_label: str,
    component: str,
    scope: str,
    subject_id: str,
    primary_manifest_path: Path,
    rows: Sequence[Mapping[str, object]],
    recorder: _PersistenceRecorder,
) -> tuple[Stage052PersistenceAttribution, Path, Path]:
    attribution = Stage052PersistenceAttribution(
        run_label=run_label,
        component=component,
        scope=scope,
        subject_id=subject_id,
        primary_manifest_relative_path=primary_manifest_path.relative_to(run_dir).as_posix(),
        primary_manifest_sha256=_sha256(primary_manifest_path),
        solver_seconds=sum(_strict_float(row.get("solver_seconds")) for row in rows),
        shard_persistence_seconds=sum(
            _strict_float(row.get("artifact_persistence_seconds")) for row in rows
        ),
        control_intervals=recorder.intervals,
    )
    suffix = "" if subject_id == "run" else f"_{subject_id}"
    path = run_dir / "control" / f"{run_label}{suffix}_persistence_attribution.json"
    written, sidecar = atomic_write_signed_json(path, attribution.to_dict())
    return attribution, written, sidecar


@dataclass(frozen=True, slots=True)
class Stage052Axis:
    name: str
    termination_mode: str
    time_limit_seconds: float
    exact_call_budget: int | None = None
    instrumentation_enabled: bool = True
    max_iterations: int | None = 1000


@dataclass(frozen=True, slots=True)
class Stage052Config:
    benchmark_dir: Path
    stage051_manifest: Path
    stage04_config: Path
    stage02_config: Path
    max_iterations: int
    batch_size: int
    native_kernels: NativeKernelConfig
    v1_storage: ArtifactStorageConfig
    v2_storage: ArtifactStorageConfig
    runtime_identity_manifest: Path
    storage_root_locator: Path
    staging_root_alias: str
    archive_root_aliases: tuple[str, ...]
    metal_helper_path: Path | None


@dataclass(frozen=True, slots=True)
class _ShardTask:
    root: Path
    config_path: Path
    run_dir: Path
    run_label: str
    component: str
    scope: str
    instance_name: str
    customer_count: int
    seed: int
    shard_ordinal: int
    worker_count: int
    storage: ArtifactStorageConfig


def load_stage052_config(path: Path) -> Stage052Config:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    try:
        stage = payload["stage05_2"]
        runtime = payload["runtime"]
        campaign = payload["campaign"]
        accelerator = payload.get("accelerator", {})
        v1 = ArtifactStorageConfig(**dict(payload["artifact_storage_v1"]))
        v2 = ArtifactStorageConfig(**dict(payload["artifact_storage_v2"]))
        config = Stage052Config(
            benchmark_dir=Path(str(payload["benchmark"]["directory"])),
            stage051_manifest=Path(str(stage["stage051_manifest"])),
            stage04_config=Path(str(stage["stage04_config"])),
            stage02_config=Path(str(stage["stage02_config"])),
            max_iterations=int(runtime["max_iterations"]),
            batch_size=int(runtime["batch_size"]),
            native_kernels=NativeKernelConfig(**dict(payload["native_kernels"])),
            v1_storage=v1,
            v2_storage=v2,
            runtime_identity_manifest=Path(str(runtime["identity_manifest"])),
            storage_root_locator=Path(str(campaign["storage_root_locator"])),
            staging_root_alias=str(campaign["staging_root_alias"]),
            archive_root_aliases=tuple(str(item) for item in campaign["archive_root_aliases"]),
            metal_helper_path=(
                Path(str(accelerator["metal_helper_path"]))
                if isinstance(accelerator, Mapping) and accelerator.get("metal_helper_path")
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid Stage 5.2 configuration: {error}") from error
    if config.v1_storage.storage_policy_version != "artifact-storage-v1":
        raise ValueError("Stage 5.2 v1 baseline storage must use artifact-storage-v1")
    if config.v2_storage.storage_policy_version != ARTIFACT_STORAGE_V2:
        raise ValueError("Stage 5.2 streaming storage must use artifact-storage-v2")
    if config.v2_storage.screening_schema_version != "screening_decisions_v3":
        raise ValueError("current Stage 5.2 streaming storage requires screening_decisions_v3")
    if config.max_iterations != 1000 or config.batch_size <= 0:
        raise ValueError("Stage 5.2 requires 1000 iterations and a positive batch size")
    if config.staging_root_alias != "transfer_staging" or config.archive_root_aliases != (
        "transfer_archive",
        "internal_archive",
    ):
        raise ValueError("Stage 5.2 campaign storage root aliases are fixed")
    return config


def validate_stage052_run_label(run_label: str, component: Stage052Component | str) -> None:
    selected = Stage052Component(component)
    pattern = re.compile(rf"^stage05\.2_{re.escape(selected.value)}_(?:attempt|rerun)[0-9]{{2}}$")
    if pattern.fullmatch(run_label) is None:
        raise ValueError(
            "Stage 5.2 run label must be canonical: "
            f"stage05.2_{selected.value}_attemptNN or rerunNN"
        )


def _verify_performance_staging_root(
    *,
    root: Path,
    locator_path: Path,
    staging_alias: str,
    output_dir: Path,
    volume_probe: Callable[[Path], VolumeIdentity] | None = None,
) -> dict[str, object]:
    """Bind C--F evidence to the configured external staging volume."""

    locator = StorageRootLocator.from_toml(locator_path)
    staging = locator.resolve(staging_alias)
    expected_results = (root / "results").resolve()
    if staging.absolute_path.resolve() != expected_results:
        raise ValueError(
            "Stage 5.2 performance staging root must equal this repository's results root"
        )
    if output_dir.resolve().parent != expected_results:
        raise ValueError("Stage 5.2 performance output is outside the staging root")
    locator.verify_all(
        probe_volume_identity if volume_probe is None else volume_probe,
        (staging_alias,),
    )
    return stage052_storage_root_binding(
        alias=staging_alias,
        volume=staging.volume.to_dict(),
    )


def axes_for_scope(scope: str, *, customer_count: int | None = None) -> tuple[Stage052Axis, ...]:
    if scope == "performance":
        return (
            Stage052Axis("fixed_work_control", "fixed_work", 120.0, 100, False),
            Stage052Axis("fixed_work", "fixed_work", 120.0, 100),
            Stage052Axis("wall_clock_30", "wall_clock", 30.0),
        )
    if scope == "pilot":
        return (
            Stage052Axis(
                "wall_clock_30",
                "wall_clock",
                30.0,
                max_iterations=None if customer_count == 100 else 1000,
            ),
        )
    if scope == "formal":
        if customer_count is None:
            raise ValueError("formal scope requires customer_count")
        return tuple(
            Stage052Axis(
                f"wall_clock_{budget}",
                "wall_clock",
                float(budget),
                max_iterations=None if customer_count == 100 else 1000,
            )
            for budget in formal_budget_matrix().budgets_for_customer_count(customer_count)
        )
    raise ValueError(f"unsupported Stage 5.2 scope: {scope}")


def verify_stage051_prerequisite(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read Stage 5.1 prerequisite: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("Stage 5.1 prerequisite must be a JSON object")
    if (
        payload.get("run_label") != "stage05.1_best_known_attempt06"
        or payload.get("status") != "READY_FOR_STAGE05_2"
        or payload.get("comparison_baseline") != "stage04_adaptive_weights_attempt15"
    ):
        raise RuntimeError(
            "Stage 5.2 requires accepted stage05.1_best_known_attempt06 "
            "and stage04_adaptive_weights_attempt15"
        )
    return payload


def run_stage052(
    *,
    config_path: Path,
    output_dir: Path,
    run_label: str,
    component: Stage052Component | str,
    scope: str,
    worker_count: int = 1,
    prerequisite_dir: Path | None = None,
    prerequisite_dirs: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    """Execute one canonical Stage 5.2 component attempt."""

    selected = Stage052Component(component)
    validate_stage052_run_label(run_label, selected)
    contract = stage052_contract(selected, scope)
    if worker_count not in {1, 2, 4}:
        raise ValueError("Stage 5.2 worker_count must be 1, 2, or 4")
    root = repository_root()
    resolved_config = _resolve(root, config_path)
    resolved_output = _resolve(root, output_dir)
    if resolved_output != root / "results" / run_label:
        raise ValueError("Stage 5.2 output must be results/<canonical-run-label>")
    if resolved_output.exists():
        raise FileExistsError(resolved_output)
    _require_clean_repository(root)
    config = load_stage052_config(resolved_config)
    staging_root_binding: dict[str, object] | None = None
    if selected in {
        Stage052Component.ARTIFACT_STREAMING,
        Stage052Component.JOB_PARALLEL,
        Stage052Component.NATIVE_KERNELS,
        Stage052Component.ACCELERATOR_PILOT,
    }:
        staging_root_binding = _verify_performance_staging_root(
            root=root,
            locator_path=_resolve(root, config.storage_root_locator),
            staging_alias=config.staging_root_alias,
            output_dir=resolved_output,
        )
    prerequisite = verify_stage051_prerequisite(_resolve(root, config.stage051_manifest))
    component_prerequisite = None
    component_prerequisites: dict[str, object] = {}
    resolved_prerequisite_dirs: dict[str, Path] = {}
    job_parallel_selection = None
    accelerator_decision_payload: dict[str, object] | None = None
    if contract.prerequisites:
        supplied = dict(prerequisite_dirs or {})
        if prerequisite_dir is not None:
            if supplied:
                raise ValueError("use prerequisite_dir or prerequisite_dirs, not both")
            if len(contract.prerequisites) != 1:
                raise ValueError(f"{selected.value} requires named prerequisite role bindings")
            supplied[contract.prerequisites[0].role] = prerequisite_dir
        expected_roles = {requirement.role for requirement in contract.prerequisites}
        if set(supplied) != expected_roles:
            raise ValueError(
                "Stage 5.2 prerequisite roles mismatch: "
                f"expected={sorted(expected_roles)} observed={sorted(supplied)}"
            )
        identities: dict[str, Any] = {}
        for requirement in contract.prerequisites:
            resolved_input = _resolve(root, supplied[requirement.role])
            resolved_prerequisite_dirs[requirement.role] = resolved_input
            identities[requirement.role] = verify_stage052_evidence_input(
                resolved_input,
                requirement,
            )
        component_prerequisites = {
            role: identity.to_dict() for role, identity in identities.items()
        }
        if len(identities) == 1:
            component_prerequisite = next(iter(identities.values()))
        if selected is Stage052Component.NATIVE_KERNELS:
            selection_prerequisite = identities["worker_selection"]
            job_parallel_selection = verify_job_parallel_selection(
                resolved_prerequisite_dirs["worker_selection"],
                selection_prerequisite,
            )
            if worker_count != job_parallel_selection.selected_workers:
                raise ValueError(
                    "native_kernels worker_count must equal the independently selected "
                    f"D worker count ({job_parallel_selection.selected_workers})"
                )
        if selected is Stage052Component.ACCELERATOR_PILOT:
            if scope != "performance":
                raise ValueError("accelerator_pilot decision uses performance scope")
            prerequisite_path = resolved_prerequisite_dirs["native_selection"]
            predecessor_metadata = _stage052_metadata(prerequisite_path)
            predecessor_workers = _strict_int(
                predecessor_metadata.get("worker_count"), "worker_count"
            )
            if worker_count != predecessor_workers:
                raise ValueError(
                    "accelerator_pilot worker_count must equal the accepted E worker count "
                    f"({predecessor_workers})"
                )
            if predecessor_metadata.get("native_kernel_config") != NativeKernelConfig().to_dict():
                raise ValueError("accelerator_pilot requires complete accepted native kernels")
            accelerator_decision_payload = _accelerator_decision_inputs(
                prerequisite_path,
                prerequisite=identities["native_selection"],
                metal_helper_path=(
                    _resolve(root, config.metal_helper_path)
                    if config.metal_helper_path is not None
                    else None
                ),
            )
    instances, seeds = _scope_identities(scope)
    if contract.storage_policy_version == "artifact-storage-v1":
        storage = config.v1_storage
    else:
        storage = config.v2_storage
    if contract.worker_policy == "single" and worker_count != 1:
        raise ValueError(f"{selected.value} requires single-worker evidence")
    if selected is Stage052Component.JOB_PARALLEL and scope != "performance":
        raise ValueError("job_parallel selection uses the fixed performance scope")
    if scope == "formal" and selected is not Stage052Component.BENCHMARK:
        raise ValueError("only the benchmark component may run Formal scope")
    if selected is Stage052Component.BENCHMARK:
        return _run_benchmark_campaign(
            root=root,
            config_path=resolved_config,
            output_dir=resolved_output,
            run_label=run_label,
            scope=scope,
            worker_count=worker_count,
            config=config,
            stage051_prerequisite=prerequisite,
            component_prerequisites=component_prerequisites,
            resolved_prerequisite_dirs=resolved_prerequisite_dirs,
            storage=storage,
        )

    resolved_output.mkdir(parents=True)
    context = ArtifactRunContext("stage05.2", selected.value, run_label)
    parent_writer = ArtifactBundleWriter(resolved_output, context, storage)
    revision = _git(root, "rev-parse", "HEAD")
    runtime_identity = verify_stage052_runtime_identity(
        _resolve(root, config.runtime_identity_manifest),
        expected_repository_revision=revision,
    )
    environment = collect_environment()
    metadata = {
        "schema_version": STAGE052_SCHEMA_VERSION,
        "run_label": run_label,
        "component": selected.value,
        "scope": scope,
        "instances": list(instances),
        "seeds": list(seeds),
        "worker_count": worker_count,
        "storage_policy_version": storage.storage_policy_version,
        "backend": contract.required_backend,
        "execution_backend": _execution_backend(
            selected,
            accelerator_decision_payload=accelerator_decision_payload,
        ),
        "stage052_contract": {
            "component": contract.component.value,
            "scope": contract.scope,
            "prerequisite_component": (
                contract.prerequisite_component.value
                if contract.prerequisite_component is not None
                else None
            ),
            "prerequisite_status": contract.prerequisite_status,
            "prerequisites": [
                {
                    "role": requirement.role,
                    "component": requirement.component.value,
                    "scope": requirement.scope,
                    "allowed_statuses": list(requirement.allowed_statuses),
                    "exact_run_label": requirement.exact_run_label,
                    "requires_passed_review": requirement.requires_passed_review,
                    "requires_current_chain_identity": (
                        requirement.requires_current_chain_identity
                    ),
                }
                for requirement in contract.prerequisites
            ],
            "required_backend": contract.required_backend,
            "worker_policy": contract.worker_policy,
            "storage_policy_version": contract.storage_policy_version,
            "screening_schema_version": contract.screening_schema_version,
            "native_profile": contract.native_profile,
            "next_status": contract.next_status,
        },
        "screening_schema_version": contract.screening_schema_version,
        "optimization_profile": _accelerator_optimization_profile(
            selected,
            accelerator_decision_payload=accelerator_decision_payload,
        ),
        "native_kernel_config": (
            config.native_kernels.to_dict()
            if selected
            in {
                Stage052Component.NATIVE_KERNELS,
                Stage052Component.ACCELERATOR_PILOT,
                Stage052Component.BENCHMARK,
            }
            else None
        ),
        "persistence_attribution": (
            "decision_only_no_solver_persistence"
            if accelerator_decision_payload is not None
            else "primary_active_writes_v1"
        ),
        "repository_revision": revision,
        "repository_dirty": False,
        "runtime_identity": runtime_identity,
        "staging_root": staging_root_binding,
        "configuration_sha256": _sha256(resolved_config),
        "stage051_prerequisite": prerequisite,
        "component_prerequisite": (
            component_prerequisite.to_dict() if component_prerequisite is not None else None
        ),
        "component_prerequisites": component_prerequisites,
        "job_parallel_selection": (
            job_parallel_selection.to_dict() if job_parallel_selection is not None else None
        ),
        "accelerator_decision_mode": (
            _accelerator_mode(accelerator_decision_payload)
            if accelerator_decision_payload is not None
            else None
        ),
        "performance_provenance": collect_performance_provenance(
            instance_paths={
                instance: _resolve(root, config.benchmark_dir) / f"{instance}.txt"
                for instance in instances
            },
            stage02_config_path=_resolve(root, config.stage02_config),
            stage04_config_path=_resolve(root, config.stage04_config),
            max_iterations=config.max_iterations,
            batch_size=config.batch_size,
            runtime_environment=environment,
        ),
        "environment": environment,
    }
    persistence_recorder = _PersistenceRecorder()
    with persistence_recorder.record("parent_write_control"):
        parent_writer.write_control(metadata=metadata, configuration_path=resolved_config)
    if accelerator_decision_payload is not None:
        metal_mode = accelerator_decision_payload.get("schema_version") == (
            "stage05.2-accelerator-pilot-artifact-v1"
        )
        artifact_type = "metal_pilot" if metal_mode else "accelerator_decision"
        decision_path = resolved_output / "control" / f"{run_label}_{artifact_type}.json"
        decision_path.write_text(
            json.dumps(accelerator_decision_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        parent_writer.record_existing_file(
            decision_path,
            artifact_type=artifact_type,
            retention_class="control",
            storage_format="json_control",
        )
        pilot = accelerator_decision_payload.get("pilot")
        is_partial = (
            metal_mode
            and isinstance(pilot, Mapping)
            and pilot.get("status") == MetalPilotStatus.PARTIAL.value
        )
        bundle = parent_writer.finalize(
            status="partial" if is_partial else "complete",
            evidence_completeness="partial" if is_partial else "complete",
        )
        return {
            "run_dir": bundle.run_dir,
            artifact_type: decision_path,
            "manifest": bundle.manifest_path,
            "manifest_sidecar": bundle.manifest_sidecar_path,
        }
    tasks = _build_tasks(
        root=root,
        config_path=resolved_config,
        run_dir=resolved_output,
        run_label=run_label,
        component=selected,
        scope=scope,
        instances=instances,
        seeds=seeds,
        worker_count=worker_count,
        storage=storage,
    )
    rows: list[dict[str, object]] = []
    resource_sampler = (
        ProcessTreeResourceSampler(
            run_label=run_label,
            component=selected.value,
            configured_worker_count=worker_count,
            interval_seconds=0.05,
        )
        if selected
        in {
            Stage052Component.ARTIFACT_STREAMING,
            Stage052Component.JOB_PARALLEL,
            Stage052Component.NATIVE_KERNELS,
            Stage052Component.ACCELERATOR_PILOT,
            Stage052Component.BENCHMARK,
        }
        else None
    )
    if resource_sampler is not None:
        resource_sampler.start()
    try:
        if storage.storage_policy_version == ARTIFACT_STORAGE_V2:
            rows = _run_v2_tasks(tasks, worker_count=worker_count)
            parent_writer.adopt_v2_shards(
                expected_identities=tuple((task.instance_name, task.seed) for task in tasks)
            )
        else:
            for task in tasks:
                rows.extend(_run_and_persist_shard(task, writer=parent_writer))
    except BaseException as error:
        if storage.storage_policy_version == ARTIFACT_STORAGE_V2:
            for task in tasks:
                _ensure_partial_shard_failure(task, error)
            with contextlib.suppress(BaseException):
                parent_writer.adopt_v2_shards(
                    expected_identities=tuple((task.instance_name, task.seed) for task in tasks),
                    require_complete=False,
                )
        if resource_sampler is not None:
            with contextlib.suppress(BaseException):
                summary = resource_sampler.stop()
                _record_resource_summary(parent_writer, summary)
        parent_writer.finalize(status="partial", evidence_completeness="partial")
        raise
    remediation_result: Stage052RemediationResult | None = None
    if selected is Stage052Component.ARTIFACT_STREAMING:
        try:
            remediation_source = resolved_prerequisite_dirs["remediation_source"]
            remediation_result = remediate_stage052_artifacts(
                source_dir=remediation_source,
                child_dir=resolved_output / "remediation" / remediation_source.name,
                config=Stage052RemediationConfig(child_run_label=run_label),
            )
            _record_remediation_child(parent_writer, remediation_result)
        except BaseException:
            if resource_sampler is not None:
                with contextlib.suppress(BaseException):
                    summary = resource_sampler.stop()
                    _record_resource_summary(parent_writer, summary)
            parent_writer.finalize(status="partial", evidence_completeness="partial")
            raise
    rows.sort(
        key=lambda row: (
            str(row["instance"]),
            _strict_int(row["seed"], "seed"),
            str(row["axis"]),
        )
    )
    with persistence_recorder.record("parent_timing_and_per_run_control"):
        timing_evidence_path = _record_timing_evidence(parent_writer, rows)
        per_run_path = resolved_output / "control" / f"{run_label}_per_run_results.csv"
        _write_csv(per_run_path, PER_RUN_FIELDS, rows)
        parent_writer.record_existing_file(
            per_run_path,
            artifact_type="per_run_results",
            retention_class="control",
            storage_format="csv_control",
            row_count=len(rows),
        )
    resource_summary_path = None
    if resource_sampler is not None:
        summary = resource_sampler.stop()
        with persistence_recorder.record("parent_resource_control"):
            resource_summary_path = _record_resource_summary(parent_writer, summary)
    with persistence_recorder.record("parent_primary_manifest_finalize"):
        bundle = parent_writer.finalize()
    _, persistence_attribution_path, persistence_attribution_sidecar = (
        _write_persistence_attribution(
            run_dir=resolved_output,
            run_label=run_label,
            component=selected.value,
            scope=scope,
            subject_id="run",
            primary_manifest_path=bundle.manifest_path,
            rows=rows,
            recorder=persistence_recorder,
        )
    )
    outputs = {
        "run_dir": bundle.run_dir,
        "per_run_results": per_run_path,
        "manifest": bundle.manifest_path,
        "manifest_sidecar": bundle.manifest_sidecar_path,
        "persistence_attribution": persistence_attribution_path,
        "persistence_attribution_sidecar": persistence_attribution_sidecar,
    }
    if resource_summary_path is not None:
        outputs["resource_summary"] = resource_summary_path
    if timing_evidence_path is not None:
        outputs["timing_evidence"] = timing_evidence_path
    if remediation_result is not None:
        outputs["remediation_manifest"] = remediation_result.child_manifest_path
    return outputs


def _seal_campaign_startup_failure(
    *,
    output_dir: Path,
    run_label: str,
    storage: ArtifactStorageConfig,
    error: BaseException,
) -> None:
    """Seal files from an interrupted first-write phase as partial evidence."""

    primary_manifest = output_dir / "control" / f"{run_label}_manifest.json"
    primary_already_partial = False
    if primary_manifest.is_file() and primary_manifest.with_suffix(".sha256").is_file():
        try:
            sealed = ArtifactReader(output_dir)
            if (
                sealed.result.manifest_path.resolve() == primary_manifest.resolve()
                and sealed.manifest.get("status") in {"partial", "failed"}
                and sealed.manifest.get("evidence_completeness") == "partial"
            ):
                primary_already_partial = True
        except (ArtifactIntegrityError, OSError, TypeError, ValueError):
            pass
    campaign_path = output_dir / "campaign_manifest.json"
    if campaign_path.is_file():
        try:
            campaign = load_campaign_manifest(campaign_path)
            if campaign.status in {"planned", "complete"}:
                persist_campaign_manifest(
                    output_dir,
                    replace(
                        campaign,
                        status="failed",
                        failure_reason=f"{type(error).__name__}: {error}",
                    ),
                )
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
    if primary_already_partial:
        return
    failure_path, failure_sidecar = atomic_write_signed_json(
        output_dir / "control" / f"{run_label}_startup_failure.json",
        {
            "schema_version": "stage05.2-campaign-startup-failure-v1",
            "run_label": run_label,
            "status": "failed",
            "failure_reason": f"{type(error).__name__}: {error}",
        },
    )
    writer = ArtifactBundleWriter(
        output_dir,
        ArtifactRunContext("stage05.2", Stage052Component.BENCHMARK.value, run_label),
        storage,
    )
    excluded = {
        failure_path,
        failure_sidecar,
        primary_manifest,
        primary_manifest.with_suffix(".sha256"),
    }
    for path in sorted(output_dir.rglob("*")):
        if (
            not path.is_file()
            or path in excluded
            or any(part.startswith(".") for part in path.relative_to(output_dir).parts)
            or path.name.startswith("manifest_")
        ):
            continue
        writer.record_existing_file(
            path,
            artifact_type="startup_failure_retained",
            retention_class="diagnostic",
            storage_format=("sha256_control" if path.suffix == ".sha256" else "binary"),
        )
    writer.record_existing_file(
        failure_path,
        artifact_type="startup_failure",
        retention_class="diagnostic",
    )
    writer.record_existing_file(
        failure_sidecar,
        artifact_type="startup_failure_sidecar",
        retention_class="diagnostic",
        storage_format="sha256_control",
    )
    writer.finalize(status="partial", evidence_completeness="partial")


def _run_benchmark_campaign(
    *,
    root: Path,
    config_path: Path,
    output_dir: Path,
    run_label: str,
    scope: str,
    worker_count: int,
    config: Stage052Config,
    stage051_prerequisite: Mapping[str, object],
    component_prerequisites: Mapping[str, object],
    resolved_prerequisite_dirs: Mapping[str, Path],
    storage: ArtifactStorageConfig,
) -> dict[str, Path]:
    try:
        return _run_benchmark_campaign_impl(
            root=root,
            config_path=config_path,
            output_dir=output_dir,
            run_label=run_label,
            scope=scope,
            worker_count=worker_count,
            config=config,
            stage051_prerequisite=stage051_prerequisite,
            component_prerequisites=component_prerequisites,
            resolved_prerequisite_dirs=resolved_prerequisite_dirs,
            storage=storage,
        )
    except BaseException as error:
        if output_dir.is_dir():
            with contextlib.suppress(BaseException):
                _seal_campaign_startup_failure(
                    output_dir=output_dir,
                    run_label=run_label,
                    storage=storage,
                    error=error,
                )
        raise


def _run_benchmark_campaign_impl(
    *,
    root: Path,
    config_path: Path,
    output_dir: Path,
    run_label: str,
    scope: str,
    worker_count: int,
    config: Stage052Config,
    stage051_prerequisite: Mapping[str, object],
    component_prerequisites: Mapping[str, object],
    resolved_prerequisite_dirs: Mapping[str, Path],
    storage: ArtifactStorageConfig,
) -> dict[str, Path]:
    """Execute G01/G02 as immutable, archived batches rather than one task fan-out."""

    prerequisite_role = "accelerator_decision" if scope == "pilot" else "campaign_pilot"
    prerequisite_dir = resolved_prerequisite_dirs[prerequisite_role]
    expected_scope = "performance" if scope == "pilot" else "pilot"
    expected_status = (
        "READY_FOR_STAGE052_BENCHMARK"
        if scope == "pilot"
        else "READY_FOR_STAGE052_FORMAL_BENCHMARK"
    )
    selection_lock = load_benchmark_execution_lock(
        prerequisite_dir,
        expected_scope=expected_scope,
        expected_status=expected_status,
    )
    if worker_count != selection_lock.selected_workers:
        raise ValueError(
            "benchmark worker_count must equal the accepted predecessor selection "
            f"({selection_lock.selected_workers})"
        )
    revision = _git(root, "rev-parse", "HEAD")
    runtime_identity = verify_stage052_runtime_identity(
        _resolve(root, config.runtime_identity_manifest),
        expected_repository_revision=revision,
    )
    environment = collect_environment()
    instances, seeds = _scope_identities(scope)
    performance_provenance = collect_performance_provenance(
        instance_paths={
            instance: _resolve(root, config.benchmark_dir) / f"{instance}.txt"
            for instance in instances
        },
        stage02_config_path=_resolve(root, config.stage02_config),
        stage04_config_path=_resolve(root, config.stage04_config),
        max_iterations=config.max_iterations,
        batch_size=config.batch_size,
        runtime_environment=environment,
    )
    selection_lock.verify_current_execution(
        selected_backend=selection_lock.selected_backend,
        selected_exact_backend="cpu_batch",
        selected_workers=worker_count,
        repository_revision=revision,
        configuration_sha256=_sha256(config_path),
        runtime_identity=runtime_identity,
        input_provenance=performance_provenance,
        native_kernel_config=config.native_kernels.to_dict(),
    )
    if selection_lock.selected_backend == "metal":
        raise RuntimeError(
            "accepted F02 selected Metal, but no Stage 5.2 campaign Metal execution "
            "adapter is registered; native CPU fallback is forbidden"
        )
    locator_path = _resolve(root, config.storage_root_locator)
    locator = StorageRootLocator.from_toml(locator_path)
    verify_campaign_root_locations(repository_root=root, locator=locator)
    for alias in (config.staging_root_alias, *config.archive_root_aliases):
        locator.resolve(alias).absolute_path.mkdir(parents=True, exist_ok=True)
    locator.verify_all(
        probe_volume_identity,
        (config.staging_root_alias, *config.archive_root_aliases),
    )
    staging = locator.resolve(config.staging_root_alias)
    if staging.absolute_path.resolve() != (root / "results").resolve():
        raise RuntimeError("benchmark active writes must use the configured external results root")
    if output_dir != staging.absolute_path / run_label:
        raise RuntimeError("benchmark output is not below the verified staging root")
    campaign_config = (
        BenchmarkCampaignConfig.pilot(
            run_label=run_label,
            staging_root_alias=config.staging_root_alias,
            archive_root_aliases=config.archive_root_aliases,
            selected_backend=selection_lock.selected_backend,
            selected_exact_backend="cpu_batch",
            selected_workers=worker_count,
            native_profile="stage05.2-native-kernels-v1",
        )
        if scope == "pilot"
        else BenchmarkCampaignConfig.formal(
            run_label=run_label,
            staging_root_alias=config.staging_root_alias,
            archive_root_aliases=config.archive_root_aliases,
            selected_backend=selection_lock.selected_backend,
            selected_exact_backend="cpu_batch",
            selected_workers=worker_count,
            native_profile="stage05.2-native-kernels-v1",
        )
    )
    observations = (
        ()
        if scope == "pilot"
        else load_pilot_storage_observations(prerequisite_dir, locator=locator)
    )
    plan = campaign_config.build_plan(observations)
    expected_geometry = (
        (36, 36, 1_080, 144)
        if scope == "pilot"
        else (
            920,
            2_040,
            229_200,
            10_400,
        )
    )
    observed_geometry = (
        len(plan.shards),
        plan.axis_count,
        plan.declared_solver_seconds,
        plan.checkpoint_count,
    )
    if observed_geometry != expected_geometry:
        raise RuntimeError(f"benchmark {scope} geometry mismatch: {observed_geometry}")
    free_by_alias = {
        alias: free_bytes(locator.resolve(alias).absolute_path)
        for alias in (config.staging_root_alias, *config.archive_root_aliases)
    }
    capacity = campaign_config.plan_archive_roots(
        plan,
        locator,
        free_bytes_by_alias=free_by_alias,
    )
    if scope == "formal":
        selection_lock.verify_planned_storage_roots(
            staging_root_alias=config.staging_root_alias,
            planned_archive_root_aliases=tuple(
                dict.fromkeys(assignment.root_alias for assignment in capacity.assignments)
            ),
        )
    snapshot_source = MacMachineSnapshotSource()
    preflight = collect_preflight_observation(
        campaign_config,
        snapshot=snapshot_source,
    )
    output_dir.mkdir(parents=True)
    campaign_config_sha256 = _canonical_mapping_sha256(campaign_config.to_dict())
    prerequisite_review_path = prerequisite_dir / "review" / "review_manifest.json"
    prerequisite_review_sha256 = _sha256(prerequisite_review_path)
    exercised_aliases = (
        config.archive_root_aliases
        if scope == "pilot"
        else tuple(dict.fromkeys(assignment.root_alias for assignment in capacity.assignments))
    )
    metadata = {
        "schema_version": STAGE052_SCHEMA_VERSION,
        "run_label": run_label,
        "component": Stage052Component.BENCHMARK.value,
        "scope": scope,
        "instances": list(instances),
        "seeds": list(seeds),
        "backend": "cpu_batch",
        "execution_backend": selection_lock.selected_backend,
        "optimization_profile": (
            "metal" if selection_lock.selected_backend == "metal" else "native"
        ),
        "worker_count": worker_count,
        "native_profile": "stage05.2-native-kernels-v1",
        "native_kernel_config": config.native_kernels.to_dict(),
        "repository_revision": revision,
        "repository_dirty": False,
        "configuration_sha256": _sha256(config_path),
        "campaign_configuration_sha256": campaign_config_sha256,
        "campaign_prerequisite_review_sha256": prerequisite_review_sha256,
        "runtime_identity": runtime_identity,
        "performance_provenance": performance_provenance,
        "storage_policy_version": storage.storage_policy_version,
        "screening_schema_version": storage.screening_schema_version,
        "staging_root_alias": config.staging_root_alias,
        "staging_root": stage052_storage_root_binding(
            alias=config.staging_root_alias,
            volume=locator.resolve(config.staging_root_alias).volume.to_dict(),
        ),
        "planned_archive_root_aliases": list(config.archive_root_aliases),
        "archive_root_aliases_exercised": list(exercised_aliases),
        "storage_roots": locator.tracked_payload(
            (config.staging_root_alias, *config.archive_root_aliases)
        ),
        "campaign_capacity_plan": capacity.to_dict(),
        "persistence_attribution": "primary_active_writes_v1",
        "benchmark_execution_lock": selection_lock.to_dict(),
        "stage051_prerequisite": dict(stage051_prerequisite),
        "component_prerequisites": dict(component_prerequisites),
        "environment": environment,
    }
    campaign_persistence_recorder = _PersistenceRecorder()
    context = ArtifactRunContext("stage05.2", Stage052Component.BENCHMARK.value, run_label)
    parent_writer = ArtifactBundleWriter(output_dir, context, storage)
    with campaign_persistence_recorder.record("campaign_parent_write_control"):
        parent_writer.write_control(metadata=metadata, configuration_path=config_path)
    control_paths = campaign_control_paths(output_dir, run_label)
    with campaign_persistence_recorder.record("campaign_plan_write"):
        plan_path, plan_sidecar = atomic_write_signed_json(
            control_paths["campaign_plan"], plan.to_dict()
        )
    with campaign_persistence_recorder.record("campaign_preflight_write"):
        preflight_path, preflight_sidecar = atomic_write_signed_json(
            control_paths["preflight"],
            {
                "schema_version": "stage05.2-campaign-preflight-v1",
                "run_label": run_label,
                "scope": scope,
                "power_source": preflight.power_source,
                "low_power_mode_enabled": preflight.low_power_mode_enabled,
                "windows": [window.to_dict() for window in preflight.windows],
                "volume_identities": locator.tracked_payload(
                    (config.staging_root_alias, *config.archive_root_aliases)
                ),
                "free_bytes_by_alias": dict(sorted(free_by_alias.items())),
            },
        )
    campaign = CampaignManifest.planned(
        config=campaign_config,
        plan=plan,
        capacity=capacity,
        locator=locator,
        configuration_sha256=campaign_config_sha256,
        prerequisite_review_sha256=prerequisite_review_sha256,
    )
    with campaign_persistence_recorder.record("campaign_manifest_initial_write"):
        campaign_manifest_path = persist_campaign_manifest(output_dir, campaign)
    failure_drill_paths: tuple[Path, ...] | None = None
    if scope == "pilot":
        failure_drill_paths = _exercise_campaign_failure_state_machine(
            campaign,
            output_dir=output_dir,
            recorder=campaign_persistence_recorder,
            storage=storage,
        )
    all_rows: list[dict[str, object]] = []
    all_checkpoints: list[dict[str, object]] = []
    batch_persistence_envelopes: list[BatchPersistenceEnvelope] = []
    rolling_capacity_observations: list[dict[str, object]] = []
    rolling_capacity_path: Path | None = None
    rolling_capacity_sidecar: Path | None = None
    rolling_observation_paths: list[tuple[Path, Path]] = []

    def record_capacity(observation: dict[str, object]) -> None:
        nonlocal rolling_capacity_path, rolling_capacity_sidecar
        ordinal = len(rolling_capacity_observations) + 1
        journal_path = (
            output_dir
            / "control"
            / "rolling_capacity_observations"
            / (
                f"{ordinal:04d}_{observation['batch_id']}_"
                f"{observation['phase']}.json"
            )
        )
        with campaign_persistence_recorder.record(
            f"{observation['batch_id']}_{observation['phase']}_"
            "rolling_capacity_journal_write"
        ):
            journal_payload, journal_sidecar = atomic_write_signed_json(
                journal_path,
                observation,
            )
        rolling_observation_paths.append((journal_payload, journal_sidecar))
        rolling_capacity_observations.append(observation)
        label = f"{observation['batch_id']}_{observation['phase']}_rolling_capacity_write"
        with campaign_persistence_recorder.record(label):
            rolling_capacity_path, rolling_capacity_sidecar = atomic_write_signed_json(
                control_paths["rolling_capacity"],
                {
                    "schema_version": "stage05.2-rolling-capacity-summary-v1",
                    "run_label": run_label,
                    "scope": scope,
                    "observations": rolling_capacity_observations,
                    "passed": None,
                    "status": "in_progress",
                },
            )

    def verify_and_record_capacity(*, batch_id: str, phase: str) -> None:
        try:
            observation = verify_rolling_campaign_capacity(
                config=campaign_config,
                campaign=campaign,
                locator=locator,
                batch_id=batch_id,
                phase=phase,
                free_space=free_bytes,
                volume_probe=probe_volume_identity,
            )
        except RollingCampaignCapacityError as error:
            record_capacity(error.observation)
            raise
        record_capacity(observation)

    try:
        for batch_plan, planned_batch in zip(plan.batches, campaign.batches, strict=True):
            try:
                verify_and_record_capacity(
                    batch_id=planned_batch.batch_id,
                    phase="pre_dispatch",
                )
                execution = _run_benchmark_batch(
                    root=root,
                    config_path=config_path,
                    campaign_config=campaign_config,
                    campaign_configuration_sha256=campaign_config_sha256,
                    campaign_prerequisite_review_sha256=prerequisite_review_sha256,
                    config=config,
                    plan=batch_plan,
                    planned_manifest=planned_batch,
                    selection_lock=selection_lock.to_dict(),
                    repository_revision=revision,
                    runtime_identity=runtime_identity,
                    performance_provenance=performance_provenance,
                    locator=locator,
                    storage=storage,
                    snapshot_source=snapshot_source,
                )
                campaign = campaign.with_batch(execution.batch)
                with campaign_persistence_recorder.record(
                    f"{execution.batch.batch_id}_campaign_verified_state_write"
                ):
                    campaign_manifest_path = persist_campaign_manifest(output_dir, campaign)
                verify_and_record_capacity(
                    batch_id=execution.batch.batch_id,
                    phase="pre_archive",
                )
                archived_evidence = archive_verified_batch_with_evidence(
                    batch=execution.batch,
                    locator=locator,
                )
                campaign = campaign.with_batch(archived_evidence.batch)
                with campaign_persistence_recorder.record(
                    f"{execution.batch.batch_id}_campaign_archived_state_write"
                ):
                    campaign_manifest_path = persist_campaign_manifest(output_dir, campaign)
                envelope = BatchPersistenceEnvelope(
                    run_label=run_label,
                    batch_id=execution.batch.batch_id,
                    base_attribution_sha256=str(
                        execution.batch.persistence_attribution_sha256
                    ),
                    verified_manifest_sha256=execution.verified_manifest_sha256,
                    archived_manifest_sha256=archived_evidence.manifest_sha256,
                    solver_seconds=execution.base_attribution.solver_seconds,
                    base_persistence_seconds=(
                        execution.base_attribution.total_persistence_seconds
                    ),
                    state_intervals=(
                        execution.verified_manifest_write_interval,
                        archived_evidence.state_write_interval,
                    ),
                )
                with campaign_persistence_recorder.record(
                    f"{execution.batch.batch_id}_persistence_envelope_write"
                ):
                    envelope_path, _ = atomic_write_signed_json(
                        archived_evidence.manifest_path.parent
                        / "batch_persistence_envelope.json",
                        envelope.to_dict(),
                    )
                campaign = campaign.with_batch_persistence_envelope(
                    execution.batch.batch_id,
                    _sha256(envelope_path),
                )
                batch_persistence_envelopes.append(envelope)
                with campaign_persistence_recorder.record(
                    f"{execution.batch.batch_id}_campaign_envelope_state_write"
                ):
                    campaign_manifest_path = persist_campaign_manifest(output_dir, campaign)
                verify_and_record_capacity(
                    batch_id=execution.batch.batch_id,
                    phase="post_archive",
                )
                if envelope.persistence_ratio > 0.30:
                    raise RuntimeError(
                        "batch final persistence ratio exceeds 30%: "
                        f"{envelope.persistence_ratio:.9f}"
                    )
            except BaseException as error:
                if isinstance(error, ArchivedBatchStateWriteError):
                    campaign = campaign.with_batch(error.batch)
                    with campaign_persistence_recorder.record(
                        f"{planned_batch.batch_id}_campaign_archive_recovery_state_write"
                    ):
                        campaign_manifest_path = persist_campaign_manifest(
                            output_dir,
                            campaign,
                        )
                current_batch = next(
                    batch for batch in campaign.batches if batch.batch_id == planned_batch.batch_id
                )
                if current_batch.status in {"planned", "verified"}:
                    campaign = campaign.with_batch(
                        current_batch.mark_failed(f"{type(error).__name__}: {error}")
                    )
                with campaign_persistence_recorder.record(
                    f"{planned_batch.batch_id}_campaign_failure_state_write"
                ):
                    campaign_manifest_path = persist_campaign_manifest(output_dir, campaign)
                raise
            all_rows.extend(execution.rows)
            all_checkpoints.extend(execution.checkpoints)
        with campaign_persistence_recorder.record("campaign_rolling_capacity_complete_write"):
            rolling_capacity_path, rolling_capacity_sidecar = atomic_write_signed_json(
                control_paths["rolling_capacity"],
                {
                    "schema_version": "stage05.2-rolling-capacity-summary-v1",
                    "run_label": run_label,
                    "scope": scope,
                    "observations": rolling_capacity_observations,
                    "passed": True,
                    "status": "complete",
                },
            )
        archive_dry_run_path: Path | None = None
        archive_dry_run_sidecar: Path | None = None
        publication_dry_run_path: Path | None = None
        publication_dry_run_sidecar: Path | None = None
        raw_replay_drill_path: Path | None = None
        raw_replay_drill_sidecar: Path | None = None
        if scope == "pilot":
            raw_replay_drill_path, raw_replay_drill_sidecar = _exercise_raw_replay_drill(
                campaign=campaign,
                locator=locator,
                output_dir=output_dir,
                recorder=campaign_persistence_recorder,
            )
            archive_dry_run_path, archive_dry_run_sidecar = _exercise_archive_roots(
                output_dir=output_dir,
                run_label=run_label,
                locator=locator,
                staging_alias=config.staging_root_alias,
                archive_aliases=config.archive_root_aliases,
                recorder=campaign_persistence_recorder,
            )
            publication_dry_run_path, publication_dry_run_sidecar = (
                _exercise_publication_transaction(
                    output_dir,
                    run_label,
                    recorder=campaign_persistence_recorder,
                )
            )
        if len(all_rows) != plan.axis_count or len(all_checkpoints) != plan.checkpoint_count:
            raise RuntimeError(
                "campaign aggregate axis/checkpoint count mismatch: "
                f"rows={len(all_rows)} checkpoints={len(all_checkpoints)}"
            )
    except BaseException as error:
        with campaign_persistence_recorder.record("campaign_rolling_capacity_failure_write"):
            rolling_capacity_path, rolling_capacity_sidecar = atomic_write_signed_json(
                control_paths["rolling_capacity"],
                {
                    "schema_version": "stage05.2-rolling-capacity-summary-v1",
                    "run_label": run_label,
                    "scope": scope,
                    "observations": rolling_capacity_observations,
                    "passed": False,
                    "status": "failed",
                    "failure_reason": f"{type(error).__name__}: {error}",
                },
            )
        if campaign.status == "planned":
            campaign = campaign.mark_failed(f"{type(error).__name__}: {error}")
            with campaign_persistence_recorder.record("campaign_manifest_failed_write"):
                campaign_manifest_path = persist_campaign_manifest(output_dir, campaign)
        _record_campaign_control(parent_writer, plan_path, "campaign_plan")
        _record_campaign_control(parent_writer, plan_sidecar, "campaign_plan_sidecar")
        _record_campaign_control(parent_writer, preflight_path, "campaign_preflight")
        _record_campaign_control(parent_writer, preflight_sidecar, "campaign_preflight_sidecar")
        for observation_path, observation_sidecar in rolling_observation_paths:
            _record_campaign_control(
                parent_writer,
                observation_path,
                "rolling_capacity_observation",
            )
            _record_campaign_control(
                parent_writer,
                observation_sidecar,
                "rolling_capacity_observation_sidecar",
            )
        _record_campaign_control(
            parent_writer,
            rolling_capacity_path,
            "rolling_capacity",
        )
        assert rolling_capacity_sidecar is not None
        _record_campaign_control(
            parent_writer,
            rolling_capacity_sidecar,
            "rolling_capacity_sidecar",
        )
        if failure_drill_paths is not None:
            (
                failure_summary,
                failure_sidecar,
                partial_payload,
                failed_batch_path,
                failed_batch_sidecar,
                failed_campaign_path,
                failed_campaign_sidecar,
                atomic_state_path,
                atomic_state_sidecar,
                worker_failure_path,
                worker_manifest_path,
                worker_manifest_sidecar,
                archive_payload_path,
                archive_manifest_path,
                archive_manifest_sidecar,
            ) = failure_drill_paths
            _record_campaign_control(parent_writer, failure_summary, "failure_state_drill")
            _record_campaign_control(
                parent_writer,
                failure_sidecar,
                "failure_state_drill_sidecar",
            )
            parent_writer.record_existing_file(
                partial_payload,
                artifact_type="failure_state_drill_partial",
                retention_class="diagnostic",
                storage_format="binary",
            )
            for state_path, artifact_type in (
                (failed_batch_path, "failure_state_drill_failed_batch"),
                (failed_batch_sidecar, "failure_state_drill_failed_batch_sidecar"),
                (failed_campaign_path, "failure_state_drill_failed_campaign"),
                (failed_campaign_sidecar, "failure_state_drill_failed_campaign_sidecar"),
                (atomic_state_path, "failure_state_drill_atomic_state"),
                (atomic_state_sidecar, "failure_state_drill_atomic_state_sidecar"),
                (worker_manifest_path, "failure_state_drill_worker_manifest"),
                (worker_manifest_sidecar, "failure_state_drill_worker_manifest_sidecar"),
                (archive_manifest_path, "failure_state_drill_archive_manifest"),
                (archive_manifest_sidecar, "failure_state_drill_archive_manifest_sidecar"),
            ):
                _record_campaign_control(parent_writer, state_path, artifact_type)
            for payload_path, artifact_type, storage_format in (
                (
                    worker_failure_path,
                    "failure_state_drill_worker_failure",
                    "json_control",
                ),
                (archive_payload_path, "failure_state_drill_archive_payload", "binary"),
            ):
                parent_writer.record_existing_file(
                    payload_path,
                    artifact_type=artifact_type,
                    retention_class="diagnostic",
                    storage_format=storage_format,
                )
        parent_writer.finalize(status="partial", evidence_completeness="partial")
        raise
    all_rows.sort(
        key=lambda row: (
            _strict_int(row["customer_count"], "customer_count"),
            str(row["instance"]),
            _strict_int(row["seed"], "seed"),
            str(row["axis"]),
        )
    )
    all_checkpoints.sort(
        key=lambda row: (
            _strict_int(row["customer_count"], "customer_count"),
            str(row["instance"]),
            _strict_int(row["seed"], "seed"),
            _strict_int(row["axis_budget_seconds"], "axis_budget_seconds"),
            _strict_int(row["checkpoint_seconds"], "checkpoint_seconds"),
        )
    )
    per_run_path = output_dir / "control" / f"{run_label}_per_run_results.csv"
    with campaign_persistence_recorder.record("campaign_aggregate_per_run_write"):
        _write_csv(per_run_path, PER_RUN_FIELDS, all_rows)
        parent_writer.record_existing_file(
            per_run_path,
            artifact_type="per_run_results",
            retention_class="control",
            storage_format="csv_control",
            row_count=len(all_rows),
        )
    with campaign_persistence_recorder.record("campaign_anytime_checkpoints_write"):
        checkpoint_path, checkpoint_sidecar = atomic_write_signed_json(
            output_dir / "control" / f"{run_label}_anytime_checkpoints.json",
            {
                "schema_version": "stage05.2-anytime-checkpoints-v1",
                "run_label": run_label,
                "scope": scope,
                "row_count": len(all_checkpoints),
                "rows": all_checkpoints,
            },
        )
    assert rolling_capacity_path is not None and rolling_capacity_sidecar is not None
    for path, artifact_type in (
        (plan_path, "campaign_plan"),
        (plan_sidecar, "campaign_plan_sidecar"),
        (preflight_path, "campaign_preflight"),
        (preflight_sidecar, "campaign_preflight_sidecar"),
        (rolling_capacity_path, "rolling_capacity"),
        (rolling_capacity_sidecar, "rolling_capacity_sidecar"),
        (checkpoint_path, "anytime_checkpoints"),
        (checkpoint_sidecar, "anytime_checkpoints_sidecar"),
    ):
        _record_campaign_control(parent_writer, path, artifact_type)
    for observation_path, observation_sidecar in rolling_observation_paths:
        _record_campaign_control(
            parent_writer,
            observation_path,
            "rolling_capacity_observation",
        )
        _record_campaign_control(
            parent_writer,
            observation_sidecar,
            "rolling_capacity_observation_sidecar",
        )
    if archive_dry_run_path is not None and archive_dry_run_sidecar is not None:
        _record_campaign_control(parent_writer, archive_dry_run_path, "archive_dry_run")
        _record_campaign_control(parent_writer, archive_dry_run_sidecar, "archive_dry_run_sidecar")
    if publication_dry_run_path is not None and publication_dry_run_sidecar is not None:
        _record_campaign_control(parent_writer, publication_dry_run_path, "publication_dry_run")
        _record_campaign_control(
            parent_writer,
            publication_dry_run_sidecar,
            "publication_dry_run_sidecar",
        )
    if failure_drill_paths is not None:
        (
            failure_summary,
            failure_sidecar,
            partial_payload,
            failed_batch_path,
            failed_batch_sidecar,
            failed_campaign_path,
            failed_campaign_sidecar,
            atomic_state_path,
            atomic_state_sidecar,
            worker_failure_path,
            worker_manifest_path,
            worker_manifest_sidecar,
            archive_payload_path,
            archive_manifest_path,
            archive_manifest_sidecar,
        ) = failure_drill_paths
        _record_campaign_control(parent_writer, failure_summary, "failure_state_drill")
        _record_campaign_control(
            parent_writer,
            failure_sidecar,
            "failure_state_drill_sidecar",
        )
        parent_writer.record_existing_file(
            partial_payload,
            artifact_type="failure_state_drill_partial",
            retention_class="diagnostic",
            storage_format="binary",
        )
        for state_path, artifact_type in (
            (failed_batch_path, "failure_state_drill_failed_batch"),
            (failed_batch_sidecar, "failure_state_drill_failed_batch_sidecar"),
            (failed_campaign_path, "failure_state_drill_failed_campaign"),
            (failed_campaign_sidecar, "failure_state_drill_failed_campaign_sidecar"),
            (atomic_state_path, "failure_state_drill_atomic_state"),
            (atomic_state_sidecar, "failure_state_drill_atomic_state_sidecar"),
            (worker_manifest_path, "failure_state_drill_worker_manifest"),
            (worker_manifest_sidecar, "failure_state_drill_worker_manifest_sidecar"),
            (archive_manifest_path, "failure_state_drill_archive_manifest"),
            (archive_manifest_sidecar, "failure_state_drill_archive_manifest_sidecar"),
        ):
            _record_campaign_control(parent_writer, state_path, artifact_type)
        for payload_path, artifact_type, storage_format in (
            (
                worker_failure_path,
                "failure_state_drill_worker_failure",
                "json_control",
            ),
            (archive_payload_path, "failure_state_drill_archive_payload", "binary"),
        ):
            parent_writer.record_existing_file(
                payload_path,
                artifact_type=artifact_type,
                retention_class="diagnostic",
                storage_format=storage_format,
            )
    if raw_replay_drill_path is not None and raw_replay_drill_sidecar is not None:
        _record_campaign_control(parent_writer, raw_replay_drill_path, "raw_replay_drill")
        _record_campaign_control(
            parent_writer,
            raw_replay_drill_sidecar,
            "raw_replay_drill_sidecar",
        )
    with campaign_persistence_recorder.record("campaign_primary_manifest_finalize"):
        bundle = parent_writer.finalize()
    campaign_attribution = Stage052PersistenceAttribution(
        run_label=run_label,
        component=Stage052Component.BENCHMARK.value,
        scope=scope,
        subject_id="campaign",
        primary_manifest_relative_path=bundle.manifest_path.relative_to(output_dir).as_posix(),
        primary_manifest_sha256=_sha256(bundle.manifest_path),
        solver_seconds=sum(item.solver_seconds for item in batch_persistence_envelopes),
        shard_persistence_seconds=sum(
            item.total_persistence_seconds for item in batch_persistence_envelopes
        ),
        control_intervals=campaign_persistence_recorder.intervals,
    )
    campaign_attribution_path, campaign_attribution_sidecar = atomic_write_signed_json(
        output_dir / "control" / f"{run_label}_persistence_attribution.json",
        campaign_attribution.to_dict(),
    )
    if campaign_attribution.persistence_ratio > 0.30:
        raise RuntimeError(
            "campaign aggregate persistence ratio exceeds 30%: "
            f"{campaign_attribution.persistence_ratio:.9f}"
        )
    # The canonical campaign status is the final success commit.  Before this
    # point the signed attribution and its hard ratio gate cannot be bypassed.
    campaign = campaign.mark_complete()
    campaign_manifest_path = persist_campaign_manifest(output_dir, campaign)
    return {
        "run_dir": bundle.run_dir,
        "per_run_results": per_run_path,
        "anytime_checkpoints": checkpoint_path,
        "campaign_manifest": campaign_manifest_path,
        "rolling_capacity": rolling_capacity_path,
        "manifest": bundle.manifest_path,
        "manifest_sidecar": bundle.manifest_sidecar_path,
        "persistence_attribution": campaign_attribution_path,
        "persistence_attribution_sidecar": campaign_attribution_sidecar,
    }


def _run_benchmark_batch(
    *,
    root: Path,
    config_path: Path,
    campaign_config: BenchmarkCampaignConfig,
    campaign_configuration_sha256: str,
    campaign_prerequisite_review_sha256: str,
    config: Stage052Config,
    plan: Any,
    planned_manifest: BatchManifest,
    selection_lock: Mapping[str, object],
    repository_revision: str,
    runtime_identity: Mapping[str, object],
    performance_provenance: Mapping[str, object],
    locator: StorageRootLocator,
    storage: ArtifactStorageConfig,
    snapshot_source: MacMachineSnapshotSource,
) -> _VerifiedBatchExecution:
    """Run, verify, and seal one indivisible next-fit campaign batch."""

    staging_root = locator.resolve(campaign_config.staging_root_alias)
    batch_dir = staging_root.absolute_path.joinpath(*Path(planned_manifest.logical_path).parts)
    if batch_dir.exists():
        raise FileExistsError(
            "campaign batch already exists; resume and cross-attempt reuse are forbidden: "
            f"{batch_dir}"
        )
    batch_dir.mkdir(parents=True)
    context = ArtifactRunContext(
        "stage05.2", Stage052Component.BENCHMARK.value, campaign_config.run_label
    )
    writer = ArtifactBundleWriter(batch_dir, context, storage)
    batch_metadata = {
        "schema_version": "stage05.2-benchmark-batch-v1",
        "run_label": campaign_config.run_label,
        "component": Stage052Component.BENCHMARK.value,
        "scope": campaign_config.scope,
        "batch_id": plan.batch_id,
        "shard_ids": [shard.shard_id for shard in plan.shards],
        "backend": campaign_config.selected_exact_backend,
        "execution_backend": campaign_config.selected_backend,
        "worker_count": campaign_config.selected_workers,
        "native_profile": campaign_config.native_profile,
        "native_kernel_config": config.native_kernels.to_dict(),
        "repository_revision": repository_revision,
        "repository_dirty": False,
        "configuration_sha256": _sha256(config_path),
        "campaign_configuration_sha256": campaign_configuration_sha256,
        "campaign_prerequisite_review_sha256": campaign_prerequisite_review_sha256,
        "runtime_identity": dict(runtime_identity),
        "performance_provenance": dict(performance_provenance),
        "benchmark_execution_lock": dict(selection_lock),
        "storage_policy_version": storage.storage_policy_version,
        "screening_schema_version": storage.screening_schema_version,
        "staging_root_alias": campaign_config.staging_root_alias,
        "planned_archive_root_aliases": list(campaign_config.archive_root_aliases),
        "archive_root_aliases_exercised": [planned_manifest.archive_root_alias],
        "staging_volume_identity": staging_root.volume.to_dict(),
        "archive_volume_identity": locator.resolve(
            planned_manifest.archive_root_alias
        ).volume.to_dict(),
    }
    persistence_recorder = _PersistenceRecorder()
    with persistence_recorder.record("batch_write_control"):
        writer.write_control(metadata=batch_metadata, configuration_path=config_path)
    tasks = _build_campaign_batch_tasks(
        root=root,
        config_path=config_path,
        batch_dir=batch_dir,
        run_label=campaign_config.run_label,
        scope=campaign_config.scope,
        worker_count=campaign_config.selected_workers,
        storage=storage,
        shards=plan.shards,
    )
    resource_sampler = ProcessTreeResourceSampler(
        run_label=campaign_config.run_label,
        component=Stage052Component.BENCHMARK.value,
        configured_worker_count=campaign_config.selected_workers,
        interval_seconds=0.05,
    )
    runtime_monitor = BatchRuntimeMonitor(
        campaign_config,
        snapshot=snapshot_source,
        interval_seconds=1.0,
    )
    resource_started = False
    runtime_started = False
    try:
        batch_preflight = collect_preflight_observation(
            campaign_config,
            snapshot=snapshot_source,
        )
        batch_preflight_path = (
            batch_dir
            / "control"
            / (f"{campaign_config.run_label}_{plan.batch_id}_batch_preflight.json")
        )
        with persistence_recorder.record("batch_preflight_control"):
            batch_preflight_path.write_text(
                json.dumps(
                    {
                    "schema_version": "stage05.2-batch-preflight-v1",
                    "run_label": campaign_config.run_label,
                    "batch_id": plan.batch_id,
                    "power_source": batch_preflight.power_source,
                    "low_power_mode_enabled": batch_preflight.low_power_mode_enabled,
                    "windows": [window.to_dict() for window in batch_preflight.windows],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            writer.record_existing_file(
                batch_preflight_path,
                artifact_type="batch_preflight",
                artifact_subtype=plan.batch_id,
                retention_class="control",
                storage_format="json_control",
            )
        resource_sampler.start()
        resource_started = True
        runtime_monitor.start()
        runtime_started = True
        rows = _run_v2_tasks(
            tasks,
            worker_count=campaign_config.selected_workers,
            abort_reason=runtime_monitor.abort_reason,
        )
        writer.adopt_v2_shards(
            expected_identities=tuple((task.instance_name, task.seed) for task in tasks)
        )
        rows.sort(
            key=lambda row: (
                _strict_int(row["customer_count"], "customer_count"),
                str(row["instance"]),
                _strict_int(row["seed"], "seed"),
                str(row["axis"]),
            )
        )
        with persistence_recorder.record("batch_timing_and_per_run_control"):
            _record_timing_evidence(writer, rows)
            per_run_path = (
                batch_dir
                / "control"
                / (f"{campaign_config.run_label}_{plan.batch_id}_per_run_results.csv")
            )
            _write_csv(per_run_path, PER_RUN_FIELDS, rows)
            writer.record_existing_file(
                per_run_path,
                artifact_type="per_run_results",
                artifact_subtype=plan.batch_id,
                retention_class="control",
                storage_format="csv_control",
                row_count=len(rows),
            )
        resource_summary = resource_sampler.stop()
        resource_started = False
        with persistence_recorder.record("batch_resource_control"):
            resource_path = _record_resource_summary(writer, resource_summary)
        runtime_evidence = runtime_monitor.stop()
        runtime_started = False
        runtime_path = (
            batch_dir
            / "control"
            / (f"{campaign_config.run_label}_{plan.batch_id}_runtime_evidence.json")
        )
        with persistence_recorder.record("batch_runtime_control"):
            runtime_path.write_text(
                json.dumps(runtime_evidence.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            writer.record_existing_file(
                runtime_path,
                artifact_type="batch_runtime_evidence",
                artifact_subtype=plan.batch_id,
                retention_class="control",
                storage_format="json_control",
            )
        power_load_path = (
            batch_dir / "control" / (f"{campaign_config.run_label}_{plan.batch_id}_power_load.json")
        )
        with persistence_recorder.record("batch_power_load_control"):
            power_load_path.write_text(
                json.dumps(
                    {
                    "schema_version": "stage05.2-batch-power-load-v1",
                    "run_label": campaign_config.run_label,
                    "batch_id": plan.batch_id,
                    "status": "complete",
                    "preflight": {
                        "power_source": batch_preflight.power_source,
                        "low_power_mode_enabled": (batch_preflight.low_power_mode_enabled),
                        "windows": [window.to_dict() for window in batch_preflight.windows],
                    },
                    "runtime": {
                        "sample_count": runtime_evidence.sample_count,
                        "power_source_violations": (
                            0
                            if runtime_evidence.power_sources
                            == (campaign_config.required_power_source,)
                            else 1
                        ),
                        "low_power_mode_violations": (
                            1 if runtime_evidence.low_power_mode_observed else 0
                        ),
                        "maximum_load1": runtime_evidence.maximum_load1,
                        "maximum_unrelated_process_average_cores": (
                            runtime_evidence.maximum_unrelated_process_average_cores
                        ),
                        "logical_cpu_count": runtime_evidence.logical_cpu_count,
                        "process_cpu_samples": [
                            sample.to_dict()
                            for sample in runtime_evidence.process_cpu_samples
                        ],
                    },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            writer.record_existing_file(
                power_load_path,
                artifact_type="power_load",
                artifact_subtype=plan.batch_id,
                retention_class="control",
                storage_format="json_control",
            )
        checkpoints = _collect_batch_checkpoints(batch_dir, tasks)
        if len(checkpoints) != sum(shard.checkpoint_count for shard in plan.shards):
            raise RuntimeError(f"{plan.batch_id} checkpoint count does not match its plan")
        with persistence_recorder.record("batch_primary_manifest_finalize"):
            bundle = writer.finalize()
        attribution, persistence_attribution_path, _ = _write_persistence_attribution(
            run_dir=batch_dir,
            run_label=campaign_config.run_label,
            component=Stage052Component.BENCHMARK.value,
            scope=campaign_config.scope,
            subject_id=plan.batch_id,
            primary_manifest_path=bundle.manifest_path,
            rows=rows,
            recorder=persistence_recorder,
        )
        persistence_ratio = validate_batch_measurements(
            batch=plan,
            rows=rows,
            resource_summary=resource_summary,
            runtime_evidence=runtime_evidence,
            expected_workers=campaign_config.selected_workers,
            additional_persistence_seconds=attribution.control_persistence_seconds,
        )
        return _seal_verified_benchmark_batch(
            batch_dir=batch_dir,
            plan=plan,
            planned_manifest=planned_manifest,
            rows=rows,
            checkpoints=checkpoints,
            resource_path=resource_path,
            persistence_attribution_path=persistence_attribution_path,
            attribution=attribution,
            persistence_ratio=persistence_ratio,
            run_label=campaign_config.run_label,
        )
    except BaseException as error:
        if resource_started:
            with contextlib.suppress(BaseException):
                _record_resource_summary(writer, resource_sampler.stop())
        if runtime_started:
            with contextlib.suppress(BaseException):
                runtime_monitor.stop()
        for task in tasks:
            _ensure_partial_shard_failure(task, error)
        with contextlib.suppress(BaseException):
            writer.adopt_v2_shards(
                expected_identities=tuple((task.instance_name, task.seed) for task in tasks),
                require_complete=False,
            )
        with contextlib.suppress(BaseException):
            writer.finalize(status="partial", evidence_completeness="partial")
        failed = planned_manifest.mark_failed(f"{type(error).__name__}: {error}")
        with contextlib.suppress(BaseException):
            atomic_write_signed_json(batch_dir / "batch_manifest.json", failed.to_dict())
        raise


def _seal_verified_benchmark_batch(
    *,
    batch_dir: Path,
    plan: Any,
    planned_manifest: BatchManifest,
    rows: list[dict[str, object]],
    checkpoints: list[dict[str, object]],
    resource_path: Path,
    persistence_attribution_path: Path,
    attribution: Stage052PersistenceAttribution,
    persistence_ratio: float,
    run_label: str,
) -> _VerifiedBatchExecution:
    """Validate every sealed shard and commit the signed verified batch state."""

    logical_event_rows = _logical_event_row_count(rows)
    shard_manifest_sha256_by_id: dict[str, str] = {}
    shard_actual_bytes_by_id: dict[str, int] = {}
    for shard in plan.shards:
        shard_dir = batch_dir / shard.instance / str(shard.seed)
        manifest_path = shard_dir / (
            f"{run_label}_shard_manifest_{shard.instance}_{shard.seed}.json"
        )
        sidecar = manifest_path.with_suffix(".sha256")
        if (
            not manifest_path.is_file()
            or not sidecar.is_file()
            or sidecar.read_text(encoding="utf-8").strip() != _sha256(manifest_path)
        ):
            raise RuntimeError(f"sealed shard manifest failed: {shard.shard_id}")
        shard_manifest_sha256_by_id[shard.shard_id] = _sha256(manifest_path)
        shard_actual_bytes_by_id[shard.shard_id] = directory_byte_count(shard_dir)
    actual_bytes = directory_byte_count(batch_dir)
    verified = planned_manifest.mark_verified(
        checksum_sha256=directory_checksum(batch_dir),
        actual_bytes=actual_bytes,
        row_count=logical_event_rows,
        physical_schema="screening_decisions_v3",
        resource_summary_sha256=_sha256(resource_path),
        persistence_attribution_sha256=_sha256(persistence_attribution_path),
        control_persistence_seconds=attribution.control_persistence_seconds,
        persistence_ratio=persistence_ratio,
        shard_manifest_sha256_by_id=shard_manifest_sha256_by_id,
        shard_actual_bytes_by_id=shard_actual_bytes_by_id,
    )
    verified_write_started_ns = time.monotonic_ns()
    verified_manifest_path, _ = atomic_write_signed_json(
        batch_dir / "batch_manifest.json",
        verified.to_dict(),
    )
    verified_write_completed_ns = time.monotonic_ns()
    return _VerifiedBatchExecution(
        rows=rows,
        checkpoints=checkpoints,
        batch=verified,
        base_attribution=attribution,
        verified_manifest_sha256=_sha256(verified_manifest_path),
        verified_manifest_write_interval=PersistenceInterval(
            label="verified_batch_manifest_write",
            started_ns=verified_write_started_ns,
            completed_ns=verified_write_completed_ns,
        ),
    )


def _logical_event_row_count(rows: Sequence[Mapping[str, object]]) -> int:
    """Return the batch-manifest count independently replayed by the reviewer."""

    count = 0
    for row in rows:
        timing = row.get("_timing_evidence")
        if not isinstance(timing, Mapping):
            raise RuntimeError("batch row is missing shard event-count evidence")
        count += _strict_int(timing.get("axis_event_count"), "axis_event_count")
    return count


def _build_campaign_batch_tasks(
    *,
    root: Path,
    config_path: Path,
    batch_dir: Path,
    run_label: str,
    scope: str,
    worker_count: int,
    storage: ArtifactStorageConfig,
    shards: Sequence[Any],
) -> list[_ShardTask]:
    return [
        _ShardTask(
            root=root,
            config_path=config_path,
            run_dir=batch_dir,
            run_label=run_label,
            component=Stage052Component.BENCHMARK.value,
            scope=scope,
            instance_name=shard.instance,
            customer_count=shard.customer_count,
            seed=shard.seed,
            shard_ordinal=int(shard.shard_id.removeprefix("shard")) - 1,
            worker_count=worker_count,
            storage=storage,
        )
        for shard in shards
    ]


def _collect_batch_checkpoints(
    batch_dir: Path,
    tasks: Sequence[_ShardTask],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in tasks:
        raw_path = (
            task.run_dir
            / task.instance_name
            / str(task.seed)
            / (f"{task.run_label}_raw_{task.instance_name}_{task.seed}.json")
        )
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise RuntimeError("benchmark raw shard must be an object")
        axes = payload.get("axes")
        if not isinstance(axes, Mapping):
            raise RuntimeError("benchmark raw shard axes are missing")
        for axis_name, raw_axis in axes.items():
            if not isinstance(axis_name, str) or not isinstance(raw_axis, Mapping):
                raise RuntimeError("benchmark raw axis is invalid")
            checkpoints = raw_axis.get("anytime_checkpoints")
            if not isinstance(checkpoints, list):
                raise RuntimeError("benchmark raw axis checkpoints are missing")
            for raw_checkpoint in checkpoints:
                if not isinstance(raw_checkpoint, Mapping):
                    raise RuntimeError("benchmark checkpoint row is invalid")
                checkpoint = dict(raw_checkpoint)
                checkpoint["customer_count"] = task.customer_count
                checkpoint["axis"] = axis_name
                rows.append(checkpoint)
    return rows


def _canonical_mapping_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _record_campaign_control(
    writer: ArtifactBundleWriter,
    path: Path,
    artifact_type: str,
) -> None:
    writer.record_existing_file(
        path,
        artifact_type=artifact_type,
        retention_class="control",
        storage_format=("sha256_control" if path.suffix == ".sha256" else "json_control"),
    )


def _exercise_campaign_failure_state_machine(
    campaign: CampaignManifest,
    *,
    output_dir: Path,
    recorder: _PersistenceRecorder,
    storage: ArtifactStorageConfig,
) -> tuple[Path, ...]:
    """Exercise real worker, signed-state, and archive recovery paths."""

    injected_reason = "injected campaign failure-handling drill"
    drill_dir = output_dir / "control" / "failure_state_drill"
    drill_dir.mkdir(parents=True, exist_ok=False)
    partial_path = drill_dir / "injected_partial_payload.bin"
    expected_payload = hashlib.sha256(campaign.run_label.encode("utf-8")).digest() * 256
    injected_after_bytes = len(expected_payload) // 2
    try:
        with recorder.record("failure_state_drill_partial_write"), partial_path.open(
            "xb"
        ) as handle:
            handle.write(expected_payload[:injected_after_bytes])
            handle.flush()
            os.fsync(handle.fileno())
            raise OSError(injected_reason)
    except OSError as error:
        if str(error) != injected_reason:
            raise
    failed_batch = campaign.batches[0].mark_failed(injected_reason)
    failed_campaign = campaign.with_batch(failed_batch).mark_failed(injected_reason)
    if (
        failed_campaign.status != "failed"
        or failed_campaign.batches[0].status != "failed"
        or any(batch.status != "planned" for batch in failed_campaign.batches[1:])
    ):
        raise RuntimeError("campaign failure-handling drill did not preserve partial state")
    with recorder.record("failure_state_drill_failed_batch_write"):
        failed_batch_path, failed_batch_sidecar = atomic_write_signed_json(
            drill_dir / "failed_batch_manifest.json",
            failed_batch.to_dict(),
        )
    with recorder.record("failure_state_drill_failed_campaign_write"):
        failed_campaign_path, failed_campaign_sidecar = atomic_write_signed_json(
            drill_dir / "failed_campaign_manifest.json",
            failed_campaign.to_dict(),
        )
    if (
        load_batch_manifest(failed_batch_path) != failed_batch
        or load_campaign_manifest(failed_campaign_path) != failed_campaign
    ):
        raise RuntimeError("campaign failure drill signed state did not replay")
    with recorder.record("failure_state_drill_atomic_generation_write"):
        atomic_state_path, atomic_state_sidecar = atomic_write_signed_json(
            drill_dir / "atomic_state.json",
            {"run_label": campaign.run_label, "generation": 1},
        )
    prior_atomic_payload = atomic_state_path.read_bytes()
    prior_atomic_sidecar = atomic_state_sidecar.read_bytes()
    replace_calls = 0

    def fail_sidecar_replace(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected signed-state sidecar replace failure")
        os.replace(source, destination)

    try:
        with recorder.record("failure_state_drill_atomic_replace_failure"):
            atomic_write_signed_json(
                atomic_state_path,
                {"run_label": campaign.run_label, "generation": 2},
                _replace=fail_sidecar_replace,
            )
    except OSError as error:
        if "sidecar replace failure" not in str(error):
            raise
    if (
        atomic_state_path.read_bytes() != prior_atomic_payload
        or atomic_state_sidecar.read_bytes() != prior_atomic_sidecar
    ):
        raise RuntimeError("campaign failure drill did not recover signed state")

    worker_root = drill_dir / "worker_failure"
    worker_task = _ShardTask(
        root=output_dir,
        config_path=output_dir / "unused_failure_drill.toml",
        run_dir=worker_root,
        run_label=campaign.run_label,
        component=Stage052Component.BENCHMARK.value,
        scope=campaign.scope,
        instance_name="c101C5",
        customer_count=5,
        seed=2014,
        shard_ordinal=0,
        worker_count=campaign.selected_workers,
        storage=storage,
    )

    def inject_worker_failure(_task: _ShardTask) -> list[dict[str, object]]:
        raise RuntimeError("injected worker failure drill")

    try:
        with recorder.record("failure_state_drill_worker_failure_recovery"):
            _run_v2_shard_task(worker_task, _worker=inject_worker_failure)
    except RuntimeError as error:
        if str(error) != "injected worker failure drill":
            raise
    worker_shard_dir = worker_root / worker_task.instance_name / str(worker_task.seed)
    worker_failure_path = worker_shard_dir / (
        f"{campaign.run_label}_failure_{worker_task.instance_name}_{worker_task.seed}.json"
    )
    worker_manifest_path = worker_shard_dir / (
        f"{campaign.run_label}_shard_manifest_"
        f"{worker_task.instance_name}_{worker_task.seed}.json"
    )
    worker_manifest_sidecar = worker_manifest_path.with_suffix(".sha256")
    worker_manifest = json.loads(worker_manifest_path.read_text(encoding="utf-8"))
    if (
        worker_manifest_sidecar.read_text(encoding="utf-8").strip()
        != _sha256(worker_manifest_path)
        or worker_manifest.get("evidence_completeness") != "partial"
        or worker_manifest.get("worker_identity") != "failure-recorder"
    ):
        raise RuntimeError("worker failure drill did not seal a partial shard")

    archive_source_root = drill_dir / "archive_source"
    archive_destination_root = drill_dir / "archive_destination"
    archive_source_root.mkdir()
    archive_destination_root.mkdir()
    drill_volume = campaign.batches[0].volume
    archive_locator = StorageRootLocator(
        {
            "drill_staging": StorageRoot(
                "drill_staging", archive_source_root, drill_volume
            ),
            "drill_archive": StorageRoot(
                "drill_archive", archive_destination_root, drill_volume
            ),
        }
    )
    archive_logical_path = "archive_recovery/payload"
    archive_source = archive_source_root.joinpath(*Path(archive_logical_path).parts)
    archive_source.mkdir(parents=True)
    archive_payload_source = archive_source / "payload.bin"
    with recorder.record("failure_state_drill_archive_payload_write"):
        archive_payload_source.write_bytes(expected_payload)
    planned_archive_batch = replace(
        campaign.batches[0],
        root_alias="drill_staging",
        archive_root_alias="drill_archive",
        logical_path=archive_logical_path,
        volume=drill_volume,
    )
    shard_hashes = {shard_id: "a" * 64 for shard_id in planned_archive_batch.shard_ids}
    shard_bytes = {shard_id: 1 for shard_id in planned_archive_batch.shard_ids}
    verified_archive_batch = planned_archive_batch.mark_verified(
        checksum_sha256=directory_checksum(archive_source),
        actual_bytes=directory_byte_count(archive_source),
        row_count=0,
        physical_schema="screening_decisions_v3",
        resource_summary_sha256="b" * 64,
        persistence_attribution_sha256="c" * 64,
        control_persistence_seconds=0.0,
        persistence_ratio=0.0,
        shard_manifest_sha256_by_id=shard_hashes,
        shard_actual_bytes_by_id=shard_bytes,
    )

    def fail_archive_state_write(
        _path: Path,
        _payload: Mapping[str, object],
    ) -> tuple[Path, Path]:
        raise OSError("injected archive state write failure")

    try:
        with recorder.record("failure_state_drill_archive_recovery"):
            archive_verified_batch_with_evidence(
                batch=verified_archive_batch,
                locator=archive_locator,
                _state_writer=fail_archive_state_write,
            )
    except ArchivedBatchStateWriteError as error:
        archived_drill_batch = error.batch
        archive_destination = error.destination
    else:
        raise RuntimeError("archive recovery drill did not inject its state-write failure")
    archive_payload_path = archive_destination / "payload.bin"
    with recorder.record("failure_state_drill_archive_recovery_state_write"):
        archive_manifest_path, archive_manifest_sidecar = atomic_write_signed_json(
            archive_destination / "batch_manifest.json",
            archived_drill_batch.to_dict(),
        )
    if (
        archived_drill_batch.status != "archived"
        or archive_source.exists()
        or not archive_payload_path.is_file()
        or load_batch_manifest(archive_manifest_path) != archived_drill_batch
        or directory_checksum(archive_destination) != archived_drill_batch.checksum_sha256
        or directory_byte_count(archive_destination) != archived_drill_batch.actual_bytes
    ):
        raise RuntimeError("archive state-write recovery drill did not replay")
    with recorder.record("failure_state_drill_summary_write"):
        summary, sidecar = atomic_write_signed_json(
            campaign_control_paths(output_dir, campaign.run_label)["failure_state_drill"],
            {
                "schema_version": "stage05.2-failure-state-drill-v1",
                "run_label": campaign.run_label,
                "status": "passed",
                "failure_type": "injected_partial_write",
                "expected_total_bytes": len(expected_payload),
                "injected_after_bytes": injected_after_bytes,
                "partial_file_relative_path": partial_path.relative_to(output_dir).as_posix(),
                "partial_file_sha256": _sha256(partial_path),
                "failed_batch_relative_path": failed_batch_path.relative_to(
                    output_dir
                ).as_posix(),
                "failed_batch_sha256": _sha256(failed_batch_path),
                "failed_campaign_relative_path": failed_campaign_path.relative_to(
                    output_dir
                ).as_posix(),
                "failed_campaign_sha256": _sha256(failed_campaign_path),
                "atomic_state_relative_path": atomic_state_path.relative_to(
                    output_dir
                ).as_posix(),
                "atomic_state_sha256": _sha256(atomic_state_path),
                "atomic_state_generation": 1,
                "worker_failure_relative_path": worker_failure_path.relative_to(
                    output_dir
                ).as_posix(),
                "worker_failure_sha256": _sha256(worker_failure_path),
                "worker_manifest_relative_path": worker_manifest_path.relative_to(
                    output_dir
                ).as_posix(),
                "worker_manifest_sha256": _sha256(worker_manifest_path),
                "worker_injected_failure": "injected worker failure drill",
                "archive_payload_relative_path": archive_payload_path.relative_to(
                    output_dir
                ).as_posix(),
                "archive_payload_sha256": _sha256(archive_payload_path),
                "archive_manifest_relative_path": archive_manifest_path.relative_to(
                    output_dir
                ).as_posix(),
                "archive_manifest_sha256": _sha256(archive_manifest_path),
                "archive_transfer_mode": archived_drill_batch.transfer_mode,
                "archive_injected_failure": "injected archive state write failure",
            },
        )
    return (
        summary,
        sidecar,
        partial_path,
        failed_batch_path,
        failed_batch_sidecar,
        failed_campaign_path,
        failed_campaign_sidecar,
        atomic_state_path,
        atomic_state_sidecar,
        worker_failure_path,
        worker_manifest_path,
        worker_manifest_sidecar,
        archive_payload_path,
        archive_manifest_path,
        archive_manifest_sidecar,
    )


def _exercise_raw_replay_drill(
    *,
    campaign: CampaignManifest,
    locator: StorageRootLocator,
    output_dir: Path,
    recorder: _PersistenceRecorder,
) -> tuple[Path, Path]:
    """Reopen every archived pilot batch through the bounded artifact reader."""

    results: list[dict[str, object]] = []
    for batch in campaign.batches:
        if batch.status != "archived":
            raise RuntimeError("raw replay drill requires archived batches")
        root = locator.resolve(batch.root_alias)
        batch_dir = root.absolute_path.joinpath(*Path(batch.logical_path).parts)
        reader = ArtifactReader(batch_dir)
        if (
            directory_checksum(batch_dir) != batch.checksum_sha256
            or directory_byte_count(batch_dir) != batch.actual_bytes
        ):
            raise RuntimeError(f"raw replay drill checksum failed: {batch.batch_id}")
        results.append(
            {
                "batch_id": batch.batch_id,
                "raw_manifest_sha256": _sha256(reader.result.manifest_path),
                "directory_checksum_sha256": batch.checksum_sha256,
                "actual_bytes": batch.actual_bytes,
                "artifact_count": len(reader.manifest.get("artifacts", ())),
            }
        )
    with recorder.record("raw_replay_drill_summary_write"):
        return atomic_write_signed_json(
            campaign_control_paths(output_dir, campaign.run_label)["raw_replay_drill"],
            {
                "schema_version": "stage05.2-raw-replay-drill-v1",
                "run_label": campaign.run_label,
                "batch_count": len(results),
                "results": results,
                "passed": True,
            },
        )


def _exercise_archive_roots(
    *,
    output_dir: Path,
    run_label: str,
    locator: StorageRootLocator,
    staging_alias: str,
    archive_aliases: tuple[str, ...],
    recorder: _PersistenceRecorder,
) -> tuple[Path, Path]:
    """Exercise the exact same-volume/cross-volume transactions G02 may use."""

    staging = locator.resolve(staging_alias)
    results: list[dict[str, object]] = []
    for index, alias in enumerate(archive_aliases, start=1):
        logical_path = f"{run_label}/archive_dry_run/{alias}"
        source = staging.absolute_path.joinpath(*Path(logical_path).parts)
        if source.exists():
            raise FileExistsError(f"archive dry-run source already exists: {source}")
        source.mkdir(parents=True)
        payload_path = source / "archive_probe.bin"
        with recorder.record(f"archive_dry_run_{alias}_payload_write"):
            payload_path.write_bytes(f"{run_label}:{alias}\n".encode())
        payload_sha = _sha256(payload_path)
        actual_bytes = directory_byte_count(source)
        batch = BatchManifest(
            run_label=run_label,
            batch_id=f"batch{9000 + index:04d}",
            status="planned",
            root_alias=staging_alias,
            archive_root_alias=alias,
            logical_path=logical_path,
            volume=staging.volume,
            shard_ids=(f"shard{9000 + index:04d}",),
            estimated_bytes=actual_bytes,
        ).mark_verified(
            checksum_sha256=directory_checksum(source),
            actual_bytes=actual_bytes,
            row_count=0,
            physical_schema="archive_dry_run_v1",
            resource_summary_sha256=payload_sha,
            persistence_attribution_sha256=payload_sha,
            control_persistence_seconds=0.0,
            persistence_ratio=0.0,
            shard_manifest_sha256_by_id={f"shard{9000 + index:04d}": payload_sha},
            shard_actual_bytes_by_id={f"shard{9000 + index:04d}": actual_bytes},
        )
        with recorder.record(f"archive_dry_run_{alias}_verified_manifest_write"):
            atomic_write_signed_json(source / "batch_manifest.json", batch.to_dict())
        archived_evidence = archive_verified_batch_with_evidence(
            batch=batch,
            locator=locator,
        )
        recorder.append(
            replace(
                archived_evidence.state_write_interval,
                label=f"archive_dry_run_{alias}_archived_manifest_write",
            )
        )
        archived = archived_evidence.batch
        results.append(
            {
                "archive_root_alias": alias,
                "logical_path": logical_path,
                "volume_identity": archived.volume.to_dict(),
                "transfer_mode": archived.transfer_mode,
                "archive_transfer_seconds": archived.archive_transfer_seconds,
                "checksum_sha256": archived.checksum_sha256,
                "actual_bytes": archived.actual_bytes,
            }
        )
    with recorder.record("archive_dry_run_summary_write"):
        return atomic_write_signed_json(
            campaign_control_paths(output_dir, run_label)["archive_dry_run"],
            {
                "schema_version": "stage05.2-archive-dry-run-v1",
                "run_label": run_label,
                "staging_root_alias": staging_alias,
                "archive_root_aliases_exercised": list(archive_aliases),
                "results": results,
                "passed": True,
            },
        )


def _exercise_publication_transaction(
    output_dir: Path,
    run_label: str,
    *,
    recorder: _PersistenceRecorder,
) -> tuple[Path, Path]:
    """Verify data-first, trusted-manifest-last publication ordering locally."""

    transaction = output_dir / "control" / f".{run_label}_publication_dry_run.tmp"
    if transaction.exists():
        raise FileExistsError(transaction)
    transaction.mkdir()
    try:
        generation = transaction / "generation"
        generation.mkdir()
        payload = generation / "payload.json"
        with recorder.record("publication_dry_run_payload_write"):
            payload.write_text('{"status":"dry_run"}\n', encoding="utf-8")
            with payload.open("rb") as handle:
                os.fsync(handle.fileno())
        payload_sha = _sha256(payload)
        with recorder.record("publication_dry_run_trusted_manifest_write"):
            trusted_manifest, trusted_sidecar = atomic_write_signed_json(
                transaction / "trusted_manifest.json",
                {
                    "schema_version": "stage05.2-publication-dry-run-manifest-v1",
                    "run_label": run_label,
                    "payload_relative_path": "generation/payload.json",
                    "payload_sha256": payload_sha,
                    "status": "READY_DRY_RUN",
                },
            )
        if _sha256(payload) != payload_sha or not signed_sidecar_matches(
            trusted_manifest,
            trusted_sidecar,
        ):
            raise RuntimeError("publication dry-run verification failed")
        summary = {
            "schema_version": "stage05.2-publication-dry-run-v1",
            "run_label": run_label,
            "payload_sha256": payload_sha,
            "trusted_manifest_sha256": _sha256(trusted_manifest),
            "trusted_manifest_replaced_last": True,
            "passed": True,
        }
    finally:
        shutil.rmtree(transaction)
    with recorder.record("publication_dry_run_summary_write"):
        return atomic_write_signed_json(
            campaign_control_paths(output_dir, run_label)["publication_dry_run"],
            summary,
        )


def _record_remediation_child(
    writer: ArtifactBundleWriter,
    result: Stage052RemediationResult,
) -> None:
    summary_path = result.summary_path
    if not summary_path.is_file():
        raise RuntimeError("remediation summary is missing after child finalization")
    child_dir = result.child_manifest_path.parents[1]
    child = ArtifactReader(child_dir)
    for descriptor in child.manifest.get("artifacts", ()):
        if not isinstance(descriptor, Mapping):
            raise ArtifactIntegrityError("remediation child descriptor is invalid")
        child_path = child_dir / str(descriptor.get("relative_path", ""))
        writer.record_existing_file(
            child_path,
            artifact_type="remediation_child_payload",
            artifact_subtype=(
                f"{descriptor.get('artifact_type', '')}:"
                f"{descriptor.get('artifact_subtype', '')}"
            ),
            retention_class="critical",
            storage_format=str(descriptor.get("storage_format", "")),
            compression=str(descriptor.get("compression", "none")),
            row_count=(
                int(descriptor["row_count"])
                if descriptor.get("row_count") is not None
                else None
            ),
            schema_fingerprint=str(descriptor.get("schema_fingerprint", "")),
        )
    for path, artifact_type, storage_format in (
        (result.child_manifest_path, "remediation_child_manifest", "json_control"),
        (
            result.child_manifest_sidecar_path,
            "remediation_child_manifest_sidecar",
            "sha256_control",
        ),
        (summary_path, "remediation_summary", "json_control"),
    ):
        writer.record_existing_file(
            path,
            artifact_type=artifact_type,
            artifact_subtype=result.source_run_label,
            retention_class="critical",
            storage_format=storage_format,
        )


def _record_resource_summary(
    writer: ArtifactBundleWriter,
    summary: RunResourceSummary,
) -> Path:
    path = writer.run_dir / "control" / f"{writer.context.run_label}_resource_summary.json"
    path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    writer.record_existing_file(
        path,
        artifact_type="resource_summary",
        retention_class="control",
        storage_format="json_control",
    )
    return path


def _record_timing_evidence(
    writer: ArtifactBundleWriter,
    rows: Sequence[Mapping[str, object]],
) -> Path | None:
    timings = [row.get("_timing_evidence") for row in rows]
    if all(value is None for value in timings):
        return None
    if any(not isinstance(value, Mapping) for value in timings):
        raise RuntimeError("Stage 5.2 timing evidence is incomplete")
    path = writer.run_dir / "control" / f"{writer.context.run_label}_timing_evidence.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "stage05.2-timing-evidence-v1",
                "run_label": writer.context.run_label,
                "component": writer.context.component,
                "rows": timings,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    writer.record_existing_file(
        path,
        artifact_type="timing_evidence",
        retention_class="control",
        storage_format="json_control",
        row_count=len(timings),
    )
    return path


def _stage052_metadata(raw_dir: Path) -> dict[str, object]:
    reader = ArtifactReader(raw_dir)
    references = [
        item
        for item in reader.manifest.get("artifacts", [])
        if isinstance(item, dict) and item.get("artifact_type") == "manifest_metadata"
    ]
    if len(references) != 1:
        raise ValueError("Stage 5.2 predecessor must contain one metadata artifact")
    payload = reader.read_json(str(references[0]["relative_path"]))
    if not isinstance(payload, dict):
        raise TypeError("Stage 5.2 predecessor metadata must be an object")
    return payload


def _accelerator_decision_inputs(
    raw_dir: Path,
    *,
    prerequisite: object,
    metal_helper_path: Path | None = None,
) -> dict[str, object]:
    """Recompute E occupancy and run the conditional Metal branch when required."""

    reader = ArtifactReader(raw_dir)
    references = [
        item
        for item in reader.manifest.get("artifacts", [])
        if isinstance(item, dict) and item.get("artifact_type") == "raw"
    ]
    expected = {
        (instance, seed)
        for instance in ("c101_21", "r101_21", "rc101_21")
        for seed in PERFORMANCE_SEEDS
    }
    expected_all = {
        (instance, seed) for instance in PERFORMANCE_INSTANCES for seed in PERFORMANCE_SEEDS
    }
    values: dict[tuple[str, int], float] = {}
    raw_semantic_digests: dict[tuple[str, int], str] = {}
    native_observations: dict[tuple[str, int], PerformanceObservation] = {}
    observed_all: set[tuple[str, int]] = set()
    for reference in references:
        raw = reader.read_json(str(reference.get("relative_path", "")))
        if not isinstance(raw, Mapping):
            raise TypeError("accepted E raw payload must be an object")
        identity = (str(raw.get("instance", "")), _strict_int(raw.get("seed"), "seed"))
        if identity in observed_all:
            raise ValueError(f"duplicate E raw occupancy identity: {identity}")
        observed_all.add(identity)
        if (
            raw.get("component") != Stage052Component.NATIVE_KERNELS.value
            or raw.get("scope") != "performance"
        ):
            raise ValueError(f"invalid E raw occupancy source: {identity}")
        if identity not in expected:
            continue
        axes = raw.get("axes")
        fixed = axes.get("fixed_work") if isinstance(axes, Mapping) else None
        if (
            not isinstance(fixed, Mapping)
            or fixed.get("validator_passed") is not True
            or fixed.get("valid") is not True
        ):
            raise ValueError(f"invalid E fixed-work raw axis: {identity}")
        backend = fixed.get("backend_metrics")
        if not isinstance(backend, Mapping):
            raise ValueError(f"missing E raw backend metrics: {identity}")
        exact_calls = _strict_int(backend.get("exact_calls"), "exact_calls")
        batch_launches = _strict_int(backend.get("batch_launches"), "batch_launches")
        raw_occupancies = backend.get("launch_occupancies")
        if not isinstance(raw_occupancies, list):
            raise ValueError(f"missing E raw launch occupancies: {identity}")
        occupancies = [_strict_int(value, "launch_occupancy") for value in raw_occupancies]
        if (
            exact_calls <= 0
            or batch_launches <= 0
            or any(value <= 0 for value in occupancies)
            or len(occupancies) != batch_launches
            or sum(occupancies) != exact_calls
        ):
            raise ValueError(f"invalid E raw occupancy counters: {identity}")
        values[identity] = statistics.median(occupancies)
        semantic_digest = fixed.get("semantic_digest")
        if isinstance(semantic_digest, str) and semantic_digest:
            raw_semantic_digests[identity] = semantic_digest
    if observed_all != expected_all:
        raise ValueError(
            f"E raw shard scope mismatch: expected={sorted(expected_all)} "
            f"observed={sorted(observed_all)}"
        )
    if set(values) != expected:
        raise ValueError(
            f"E occupancy scope mismatch: expected={sorted(expected)} observed={sorted(values)}"
        )
    ordered = [
        {
            "instance": instance,
            "seed": seed,
            "axis": "fixed_work",
            "median_batch_occupancy": values[(instance, seed)],
        }
        for instance, seed in sorted(values)
    ]
    median = statistics.median(values.values())
    prerequisite_dict: dict[str, object] = getattr(prerequisite, "to_dict", lambda: {})()
    if median >= 32.0:
        per_run_references = [
            item
            for item in reader.manifest.get("artifacts", [])
            if isinstance(item, Mapping) and item.get("artifact_type") == "per_run_results"
        ]
        if len(per_run_references) != 1:
            raise ValueError("accepted E must contain one per-run result table for Metal")
        per_run_path = raw_dir / str(per_run_references[0].get("relative_path", ""))
        with per_run_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                identity = (str(row.get("instance", "")), _strict_int(row.get("seed"), "seed"))
                if identity not in expected or row.get("axis") != "fixed_work":
                    continue
                if identity in native_observations:
                    raise ValueError(f"duplicate E Metal performance row: {identity}")
                native_observations[identity] = PerformanceObservation(
                    instance=identity[0],
                    seed=identity[1],
                    customer_count=100,
                    end_to_end_seconds=_strict_float(row.get("end_to_end_seconds")),
                    semantic_digest=str(row.get("semantic_digest", "")),
                )
        if (
            set(native_observations) != expected
            or set(raw_semantic_digests) != expected
            or any(
                native_observations[identity].semantic_digest
                != raw_semantic_digests[identity]
                for identity in expected
            )
        ):
            raise ValueError(
                "E fixed-work raw/per-run performance semantics are incomplete or disagree"
            )
        executor = (
            SubprocessMetalPilotExecutor(metal_helper_path)
            if metal_helper_path is not None
            else None
        )
        pilot = run_conditional_metal_pilot(
            median_batch_occupancy=median,
            native_observations=tuple(
                native_observations[identity] for identity in sorted(native_observations)
            ),
            executor=executor,
        )
        return {
            "schema_version": "stage05.2-accelerator-pilot-artifact-v1",
            "occupancy_input_count": len(ordered),
            "occupancy_inputs": ordered,
            "native_prerequisite": prerequisite_dict,
            "pilot": pilot.to_dict(),
        }
    return {
        "schema_version": "stage05.2-accelerator-decision-v1",
        "decision_mode": "decision_only",
        "decision": "GPU_NOT_JUSTIFIED",
        "selected_backend": "native_cpu",
        "selected_exact_backend": "cpu_batch",
        "threshold": 32.0,
        "median_batch_occupancy": median,
        "input_count": len(ordered),
        "inputs": ordered,
        "native_prerequisite": prerequisite_dict,
        "gpu_rows_present": False,
        "fallback_used": False,
    }


def _accelerator_mode(payload: Mapping[str, object]) -> str:
    if payload.get("schema_version") == "stage05.2-accelerator-decision-v1":
        return "decision_only"
    if payload.get("schema_version") == "stage05.2-accelerator-pilot-artifact-v1":
        return "metal_pilot"
    raise ValueError("unknown accelerator evidence schema")


def _execution_backend(
    component: Stage052Component,
    *,
    accelerator_decision_payload: Mapping[str, object] | None,
) -> str | None:
    if component is Stage052Component.ACCELERATOR_PILOT:
        if accelerator_decision_payload is None:
            raise ValueError("accelerator evidence payload is missing")
        if _accelerator_mode(accelerator_decision_payload) == "decision_only":
            return "native_cpu"
        pilot = accelerator_decision_payload.get("pilot")
        if not isinstance(pilot, Mapping):
            raise ValueError("Metal pilot payload is missing")
        backend = pilot.get("selected_backend")
        if backend is not None and backend not in {"native_cpu", "metal"}:
            raise ValueError("Metal pilot selected backend is invalid")
        return backend if isinstance(backend, str) else None
    if component is Stage052Component.BENCHMARK:
        raise ValueError("benchmark execution backend must come from the accepted selection lock")
    if component is Stage052Component.NATIVE_KERNELS:
        return "native_cpu"
    return "python_cpu"


def _accelerator_optimization_profile(
    component: Stage052Component,
    *,
    accelerator_decision_payload: Mapping[str, object] | None,
) -> str | None:
    backend = _execution_backend(
        component,
        accelerator_decision_payload=accelerator_decision_payload,
    )
    if component is Stage052Component.ACCELERATOR_PILOT:
        if backend is None:
            return None
        return "metal" if backend == "metal" else "native"
    return _optimization_profile(component)


def _scope_identities(scope: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if scope == "performance":
        return PERFORMANCE_INSTANCES, PERFORMANCE_SEEDS
    if scope == "pilot":
        return tuple(FORMAL_INSTANCES), PERFORMANCE_SEEDS
    if scope == "formal":
        return tuple(sorted(record.instance for record in BEST_KNOWN_VALUES)), FORMAL_SEEDS
    raise ValueError("Stage 5.2 scope must be performance, pilot, or formal")


def _build_tasks(
    *,
    root: Path,
    config_path: Path,
    run_dir: Path,
    run_label: str,
    component: Stage052Component,
    scope: str,
    instances: Sequence[str],
    seeds: Sequence[int],
    worker_count: int,
    storage: ArtifactStorageConfig,
) -> list[_ShardTask]:
    customer_counts = {record.instance: record.customer_count for record in BEST_KNOWN_VALUES}
    tasks: list[_ShardTask] = []
    for ordinal, (instance, seed) in enumerate(
        identity for instance in instances for identity in ((instance, seed) for seed in seeds)
    ):
        tasks.append(
            _ShardTask(
                root=root,
                config_path=config_path,
                run_dir=run_dir,
                run_label=run_label,
                component=component.value,
                scope=scope,
                instance_name=instance,
                customer_count=customer_counts[instance],
                seed=seed,
                shard_ordinal=ordinal,
                worker_count=worker_count,
                storage=storage,
            )
        )
    return tasks


def _run_v2_tasks(
    tasks: Sequence[_ShardTask],
    *,
    worker_count: int,
    abort_reason: Callable[[], str | None] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if worker_count == 1:
        for task in tasks:
            reason = abort_reason() if abort_reason is not None else None
            if reason is not None:
                raise RuntimeError(f"runtime guard aborted Stage 5.2 work: {reason}")
            rows.extend(_run_v2_shard_task(task))
        return rows
    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_context("spawn"),
    )
    futures: dict[Any, _ShardTask] = {}
    try:
        futures = {executor.submit(_run_v2_shard_task, task): task for task in tasks}
        if abort_reason is None:
            for future in as_completed(futures):
                rows.extend(future.result())
        else:
            pending = set(futures)
            while pending:
                reason = abort_reason()
                if reason is not None:
                    raise RuntimeError(f"runtime guard aborted Stage 5.2 work: {reason}")
                completed, pending = wait(
                    pending,
                    timeout=0.5,
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    rows.extend(future.result())
    except BaseException as error:
        for future in futures:
            future.cancel()
        abort_error: BaseException | None = None
        try:
            abort_process_executor(executor)
        except BaseException as observed_abort_error:
            abort_error = observed_abort_error
        failure_error: BaseException = error
        if abort_error is not None:
            failure_error = RuntimeError(
                f"worker failure {type(error).__name__}: {error}; "
                f"process-pool abort failure {type(abort_error).__name__}: {abort_error}"
            )
        for task in tasks:
            _ensure_partial_shard_failure(task, failure_error)
        if abort_error is not None:
            raise failure_error from error
        raise
    executor.shutdown(wait=True)
    return rows


def _run_v2_shard_task(
    task: _ShardTask,
    *,
    _worker: Callable[[_ShardTask], list[dict[str, object]]] | None = None,
) -> list[dict[str, object]]:
    try:
        worker = _run_and_persist_shard if _worker is None else _worker
        return worker(task)
    except BaseException as error:
        _ensure_partial_shard_failure(task, error)
        raise


def _ensure_partial_shard_failure(task: _ShardTask, error: BaseException | str) -> None:
    directory = task.run_dir / task.instance_name / str(task.seed)
    shard_manifest = directory / (
        f"{task.run_label}_shard_manifest_{task.instance_name}_{task.seed}.json"
    )
    sidecar = shard_manifest.with_suffix(".sha256")
    if shard_manifest.is_file() and sidecar.is_file():
        try:
            payload = json.loads(shard_manifest.read_text(encoding="utf-8"))
            valid = (
                isinstance(payload, dict)
                and sidecar.read_text(encoding="utf-8").strip() == _sha256(shard_manifest)
                and payload.get("storage_policy_version") == ARTIFACT_STORAGE_V2
                and payload.get("run_label") == task.run_label
                and payload.get("instance") == task.instance_name
                and payload.get("seed") == task.seed
                and payload.get("shard_ordinal") == task.shard_ordinal
                and payload.get("evidence_completeness") in {"complete", "partial"}
                and isinstance(payload.get("artifacts"), list)
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            valid = False
        if valid:
            return
    writer = ArtifactBundleWriter(
        task.run_dir,
        ArtifactRunContext("stage05.2", task.component, task.run_label),
        task.storage,
    )
    writer.write_v2_failure_shard(
        instance=task.instance_name,
        seed=task.seed,
        shard_ordinal=task.shard_ordinal,
        worker_identity="failure-recorder",
        error=error,
    )


def _run_and_persist_shard(
    task: _ShardTask,
    *,
    writer: ArtifactBundleWriter | None = None,
) -> list[dict[str, object]]:
    config = load_stage052_config(task.config_path)
    stage04 = load_stage04_config(_resolve(task.root, config.stage04_config))
    stage02 = load_stage02_config(_resolve(task.root, config.stage02_config))
    instance = parse_schneider(
        _resolve(task.root, config.benchmark_dir) / f"{task.instance_name}.txt"
    )
    instance = replace(
        instance,
        distance_backend=_optimization_profile(Stage052Component(task.component)),
    )
    axes = axes_for_scope(task.scope, customer_count=task.customer_count)
    storage = (
        config.v1_storage if task.component in {"perf_baseline", "hot_path"} else config.v2_storage
    )
    if storage.storage_policy_version == ARTIFACT_STORAGE_V2:
        return _run_and_persist_v2_shard(
            task,
            writer=writer,
            config=config,
            stage04=stage04,
            stage02=stage02,
            instance=instance,
            axes=axes,
            storage=storage,
        )
    results: dict[str, ALNSResult] = {}
    solver_times: dict[str, float] = {}
    for axis in axes:
        started = time.perf_counter()
        exact_deadline = (
            ExactDeadlineConfig.fixed_exact_calls(
                axis.exact_call_budget or 100,
                watchdog_seconds=axis.time_limit_seconds,
            )
            if axis.termination_mode == "fixed_work"
            else ExactDeadlineConfig.wall_clock()
        )
        results[axis.name] = solve_alns(
            instance,
            seed=task.seed,
            max_iterations=axis.max_iterations,
            time_limit_seconds=axis.time_limit_seconds,
            operator_profile="stage02_constraint_guided",
            vehicle_operator_config=stage02.vehicle_operator_config,
            measurement_config=MeasurementConfig(enabled=axis.instrumentation_enabled),
            screening_config=stage04.screening_config,
            cache_incremental_config=stage04.cache_incremental_config,
            backend="cpu_batch",
            batch_size=config.batch_size,
            exact_deadline_config=exact_deadline,
            stage04_config=stage04.stage04_config,
            native_kernel_config=(
                config.native_kernels if instance.distance_backend == "native" else None
            ),
        )
        solver_times[axis.name] = time.perf_counter() - started
    shard_writer = writer or ArtifactBundleWriter(
        task.run_dir,
        ArtifactRunContext("stage05.2", task.component, task.run_label),
        storage,
    )
    persistence_started = time.perf_counter()
    semantic_digests, event_counts = _persist_shard(
        shard_writer,
        task=task,
        instance=instance,
        results=results,
        storage=storage,
    )
    persistence_seconds = time.perf_counter() - persistence_started
    peak_rss = _peak_rss_bytes()
    rows: list[dict[str, object]] = []
    total_events = sum(event_counts.values())
    for axis in axes:
        result = results[axis.name]
        backend = result.backend_metrics
        objective = result.objective
        report = validate_routes(instance, [list(route) for route in result.routes])
        batch_launches, median_batch_occupancy = _launch_occupancy_summary(backend)
        persistence_share = (
            persistence_seconds * event_counts[axis.name] / total_events
            if total_events
            else persistence_seconds / len(axes)
        )
        rows.append(
            {
                "instance": task.instance_name,
                "seed": task.seed,
                "axis": axis.name,
                "customer_count": task.customer_count,
                "component": task.component,
                "backend": result.charging_backend,
                "worker_count": task.worker_count,
                "storage_policy_version": storage.storage_policy_version,
                "persistence_attribution": "critical_event_rows",
                "solver_seconds": solver_times[axis.name],
                "artifact_persistence_seconds": persistence_share,
                "end_to_end_seconds": solver_times[axis.name] + persistence_share,
                "screening_seconds": result.screening_statistics.get(
                    "screening_runtime_seconds", 0.0
                ),
                "exact_seconds": backend.get("total_seconds", 0.0),
                "packing_seconds": backend.get("packing_seconds", 0.0),
                "unpacking_seconds": backend.get("unpacking_seconds", 0.0),
                "native_kernel_seconds": backend.get("native_kernel_seconds", 0.0),
                "native_invocations": backend.get("native_invocations", 0),
                "native_fallbacks": backend.get("native_fallbacks", 0),
                "native_screening_seconds": result.screening_statistics.get(
                    "native_screening_seconds", 0.0
                ),
                "native_screening_invocations": result.screening_statistics.get(
                    "native_screening_invocations", 0
                ),
                "native_propagation_seconds": result.screening_statistics.get(
                    "native_propagation_seconds", 0.0
                ),
                "native_propagation_invocations": result.screening_statistics.get(
                    "native_propagation_invocations", 0
                ),
                "native_protocol_fallbacks": result.screening_statistics.get(
                    "native_protocol_fallbacks", 0
                ),
                "exact_started_calls": result.exact_started_calls,
                "exact_completed_calls": result.exact_completed_calls,
                "effective_iterations": result.effective_iterations,
                "batch_launches": batch_launches,
                "median_batch_occupancy": median_batch_occupancy,
                "peak_rss_bytes": peak_rss,
                "vehicle_count": objective.vehicle_count if objective else "",
                "total_distance": objective.total_distance if objective else "",
                "total_charging_time": (objective.total_charging_time if objective else ""),
                "charging_count": objective.charging_count if objective else "",
                "validator_passed": report.feasible,
                "semantic_digest": semantic_digests[axis.name],
                "termination_reason": result.termination_reason,
                "failure_status": "" if report.feasible else "validator_failed",
            }
        )
    return rows


class _Stage052StreamingShard(Protocol):
    def append(
        self,
        *,
        route_dictionary: Mapping[str, Sequence[str]],
        critical_events: Iterable[Mapping[str, object]],
        diagnostic_rows: Iterable[Mapping[str, object]] = (),
    ) -> int: ...

    def flush(self) -> None: ...


class _Stage052TraceStreamSink(MeasurementTraceSink):
    """Bridge live measurement callbacks into one open typed Parquet shard."""

    def __init__(self, *, shard: _Stage052StreamingShard, axis_name: str) -> None:
        self._shard = shard
        self.axis_name = axis_name
        self.event_count = 0
        self.diagnostic_counts: Counter[tuple[str, str, str, str]] = Counter()
        self._pending_cache_lookup: dict[str, object] | None = None
        self._semantic_event_digest = hashlib.sha256()
        self._initial_objective_key: ObjectiveKey | None = None
        self._visible_global_bests: dict[int, AcceptedGlobalBest] = {}
        self._last_candidate_timestamp = 0.0
        self.persistence_nanoseconds = 0
        self._closed = False

    def append_route_evaluation(self, record: RouteEvaluationTrace) -> None:
        payload = {field.name: getattr(record, field.name) for field in fields(record)}
        payload["record_type"] = "route_evaluation"
        payload["event_type"] = "route_evaluation"
        self._queue(payload)

    def append_event(self, event: Mapping[str, object]) -> None:
        payload = dict(event)
        payload["record_type"] = str(event.get("event_type", "event"))
        self._queue(payload)

    def append_screening_decision(self, decision: ScreeningDecision) -> None:
        payload = {
            field.name: getattr(decision, field.name)
            for field in fields(decision)
            if field.name != "checks"
        }
        payload["checks"] = tuple(
            {
                "check": check.check,
                "status": check.status,
                "value": check.value,
                "reason": check.reason,
            }
            for check in decision.checks
        )
        payload["record_type"] = "screening_decision"
        payload["event_type"] = "screening_decision"
        self._queue(payload)

    def append_incremental_propagation(self, propagation: Mapping[str, object]) -> None:
        payload = dict(propagation)
        payload["record_type"] = "incremental_propagation"
        payload["event_type"] = "incremental_propagation"
        self._queue(payload)

    def append_neighborhood_events(
        self,
        events: Iterable[Mapping[str, object]],
        *,
        route_dictionary: dict[str, tuple[str, ...]],
    ) -> None:
        for raw_event in events:
            payload = _route_reference_event(raw_event, route_dictionary)
            payload["record_type"] = "neighborhood_event"
            self._queue(payload)

    def append_neighborhood_event(self, event: Mapping[str, object]) -> None:
        payload = _route_reference_event(event, {})
        payload["record_type"] = "neighborhood_event"
        self._queue(payload)

    def finish(self) -> None:
        self._flush_pending_lookup()

    def close(self) -> None:
        if self._closed:
            return
        self.finish()
        self._closed = True

    def semantic_digest(self, base_payload: Mapping[str, object]) -> str:
        self.finish()
        return hashlib.sha256(
            json.dumps(
                {
                    "base": dict(base_payload),
                    "event_stream_sha256": self._semantic_event_digest.hexdigest(),
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    @property
    def initial_objective_key(self) -> ObjectiveKey | None:
        return self._initial_objective_key

    @property
    def accepted_checkpoint_bests(self) -> tuple[AcceptedGlobalBest, ...]:
        unique: dict[tuple[float, int, ObjectiveKey], AcceptedGlobalBest] = {}
        for checkpoint in CHECKPOINT_SECONDS:
            current = self._visible_global_bests.get(checkpoint)
            if current is not None:
                unique[(current.completed_at_seconds, current.iteration, current.objective_key)] = (
                    current
                )
        return tuple(sorted(unique.values(), key=lambda value: value.completed_at_seconds))

    @property
    def last_candidate_timestamp(self) -> float:
        return self._last_candidate_timestamp

    def diagnostic_rows(
        self,
        *,
        run_label: str,
        instance: str,
        seed: int,
    ) -> Iterable[dict[str, object]]:
        for (lane, operator, reason, event_type), count in sorted(self.diagnostic_counts.items()):
            yield {
                "run_label": run_label,
                "instance": instance,
                "seed": seed,
                "lane": lane,
                "iteration": None,
                "operator": operator,
                "reason": reason,
                "metric": event_type,
                "count": count,
                "sum_value": None,
                "min_value": None,
                "max_value": None,
            }

    def _queue(self, raw_event: Mapping[str, object]) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed Stage 5.2 trace stream")
        event = dict(raw_event)
        event["lane"] = f"{self.axis_name}:{event.get('lane', '')}"
        event["benchmark_axis"] = self.axis_name
        if event.get("event_type") == "cache_event" and event.get("operation") == "lookup":
            self._flush_pending_lookup()
            self._pending_cache_lookup = event
            return
        pending = self._pending_cache_lookup
        if pending is not None:
            same_lookup = all(
                pending.get(field) == event.get(field)
                for field in (
                    "route_key",
                    "cache_key_digest",
                    "lane",
                    "iteration",
                    "operator",
                )
            )
            if (
                same_lookup
                and event.get("event_type") == "cache_event"
                and event.get("operation") in {"hit", "miss"}
            ):
                event["operation"] = "lookup_result"
                event["lookup_result"] = raw_event.get("operation")
                event["lookup_current_entries"] = pending.get("current_entries")
                event["lookup_current_bytes"] = pending.get("current_bytes")
                self._pending_cache_lookup = None
            else:
                self._flush_pending_lookup()
        self._append(event)

    def _flush_pending_lookup(self) -> None:
        pending = self._pending_cache_lookup
        if pending is None:
            return
        self._pending_cache_lookup = None
        self._append(pending)

    def _append(self, event: dict[str, object]) -> None:
        started_ns = time.perf_counter_ns()
        try:
            self._observe(event)
            persisted = self._shard.append(
                route_dictionary={},
                critical_events=(event,),
            )
        finally:
            self.persistence_nanoseconds += time.perf_counter_ns() - started_ns
        if persisted != 1:
            raise RuntimeError("Stage 5.2 trace sink did not persist exactly one logical event")
        self.event_count += 1

    def _observe(self, event: Mapping[str, object]) -> None:
        event_type = str(event.get("event_type", event.get("record_type", "event")))
        if event_type not in {
            "execution_error",
            "deadline_boundary",
            "cache_event",
            "screening_decision",
            "route_evaluation",
            "incremental_propagation",
        }:
            self.diagnostic_counts[
                (
                    str(event.get("lane", "")),
                    str(event.get("operator", "")),
                    str(event.get("reason", event.get("status", ""))),
                    event_type,
                )
            ] += 1
        if event_type in {"candidate_state", "cache_event", "exact_budget_boundary"}:
            self._semantic_event_digest.update(
                json.dumps(
                    {
                        "event_type": event_type,
                        "status": event.get("status"),
                        "accepted": event.get("accepted"),
                        "global_best": event.get("global_best"),
                        "operation": event.get("operation"),
                        "cache_key_digest": event.get("cache_key_digest"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
        if event_type != "candidate_state":
            return
        timestamp = _strict_float(event.get("timestamp_seconds"))
        self._last_candidate_timestamp = max(self._last_candidate_timestamp, timestamp)
        if self._initial_objective_key is None:
            self._initial_objective_key = _checkpoint_objective_key(
                event.get("current_objective_key")
            )
        if event.get("accepted") is not True or event.get("global_best") is not True:
            return
        current = AcceptedGlobalBest(
            completed_at_seconds=timestamp,
            iteration=_strict_int(event.get("iteration"), "checkpoint iteration"),
            objective_key=_checkpoint_objective_key(event.get("candidate_objective_key")),
        )
        for checkpoint in CHECKPOINT_SECONDS:
            if current.completed_at_seconds <= checkpoint:
                self._visible_global_bests[checkpoint] = current


def _run_and_persist_v2_shard(
    task: _ShardTask,
    *,
    writer: ArtifactBundleWriter | None,
    config: Stage052Config,
    stage04: Any,
    stage02: Any,
    instance: Instance,
    axes: Sequence[Stage052Axis],
    storage: ArtifactStorageConfig,
) -> list[dict[str, object]]:
    """Solve, append, and release one axis at a time for storage v2."""

    shard_writer = writer or ArtifactBundleWriter(
        task.run_dir,
        ArtifactRunContext("stage05.2", task.component, task.run_label),
        storage,
    )
    shard = shard_writer.open_v2_shard(
        instance=task.instance_name,
        seed=task.seed,
        shard_ordinal=task.shard_ordinal,
        worker_identity=f"pid-{os.getpid()}",
    )
    raw_axes: dict[str, object] = {}
    solution_axes: dict[str, object] = {}
    trace_axes: dict[str, object] = {}
    drafts: dict[str, dict[str, object]] = {}
    timing_by_axis: dict[str, dict[str, int]] = {}
    live_persistence_ns_by_axis: dict[str, int] = {}
    event_counts: dict[str, int] = {}
    failures: list[str] = []
    active_trace_stream: _Stage052TraceStreamSink | None = None
    try:
        for axis in axes:
            axis_started_ns = time.perf_counter_ns()
            solver_started_ns = axis_started_ns
            trace_stream = _Stage052TraceStreamSink(shard=shard, axis_name=axis.name)
            active_trace_stream = trace_stream
            result = _solve_stage052_axis(
                instance,
                seed=task.seed,
                axis=axis,
                config=config,
                stage04=stage04,
                stage02=stage02,
                trace_sink=trace_stream,
            )
            solver_completed_ns = time.perf_counter_ns()
            live_persistence_ns = trace_stream.persistence_nanoseconds
            live_persistence_ns_by_axis[axis.name] = live_persistence_ns
            solver_elapsed_ns = solver_completed_ns - solver_started_ns
            if live_persistence_ns > solver_elapsed_ns:
                raise RuntimeError("live trace persistence exceeds measured solver wall time")
            solver_seconds = (solver_elapsed_ns - live_persistence_ns) / 1_000_000_000
            trace = result.measurement_trace
            if trace is None:
                raise RuntimeError(f"Stage 5.2 {axis.name} result is missing its trace")
            objective_key = list(result.objective.key) if result.objective else []
            validation = validate_routes(instance, [list(route) for route in result.routes])
            reconciliation = trace.reconcile(result)
            valid = result.feasible and validation.feasible and reconciliation["status"] == "pass"
            if not valid:
                failures.append(f"{axis.name}: solver, validator, or trace reconciliation failed")
            route_dictionary: dict[str, tuple[str, ...]] = {}
            trace_stream.append_neighborhood_events(
                result.neighborhood_events,
                route_dictionary=route_dictionary,
            )
            trace_stream.finish()
            event_counts[axis.name] = trace_stream.event_count
            diagnostic_rows = tuple(
                trace_stream.diagnostic_rows(
                    run_label=task.run_label,
                    instance=task.instance_name,
                    seed=task.seed,
                )
            )
            shard.append(
                route_dictionary={},
                critical_events=(),
                diagnostic_rows=diagnostic_rows,
            )
            semantic_digest = trace_stream.semantic_digest(
                {
                    "objective_key": objective_key,
                    "routes": [list(route) for route in result.routes],
                    "started_calls": result.exact_started_calls,
                    "completed_calls": result.exact_completed_calls,
                }
            )
            initial_objective_key = (
                list(_stage052_initial_objective_key(result))
                if task.component == Stage052Component.BENCHMARK.value
                else []
            )
            initial_routes = (
                [list(route) for route in result.initial_routes]
                if task.component == Stage052Component.BENCHMARK.value
                else []
            )
            raw_axes[axis.name] = {
                "backend": result.charging_backend,
                "backend_metrics": result.backend_metrics,
                "objective_key": objective_key,
                "initial_objective_key": initial_objective_key,
                "runtime_seconds": result.runtime_seconds,
                "effective_iterations": result.effective_iterations,
                "unique_route_semantics": result.unique_route_semantics,
                "started_calls": result.exact_started_calls,
                "completed_calls": result.exact_completed_calls,
                "termination_reason": result.termination_reason,
                "iteration_limit_completed_at_seconds": (
                    getattr(result, "iteration_limit_completed_at_seconds", None)
                ),
                "semantic_digest": semantic_digest,
                "trace_reconciliation": reconciliation,
                "validator_passed": validation.feasible,
                "valid": valid,
                "anytime_checkpoints": [
                    checkpoint.to_dict()
                    for checkpoint in _stage052_anytime_checkpoints(
                        result,
                        instance_name=task.instance_name,
                        seed=task.seed,
                        axis=axis,
                        trace_stream=trace_stream,
                    )
                ]
                if task.component == Stage052Component.BENCHMARK.value
                else [],
            }
            solution_axes[axis.name] = {
                "routes": [list(route) for route in result.routes],
                "objective_key": objective_key,
                "initial_routes": initial_routes,
                "initial_objective_key": initial_objective_key,
                "feasible": result.feasible,
            }
            trace_axes[axis.name] = trace.to_index_dict()
            drafts[axis.name] = _stage052_row_draft(
                task=task,
                axis=axis,
                result=result,
                validation_passed=validation.feasible,
                axis_valid=valid,
                solver_seconds=solver_seconds,
                semantic_digest=semantic_digest,
                storage=storage,
            )
            trace_stream.close()
            active_trace_stream = None
            del result, trace, route_dictionary, diagnostic_rows
            gc.collect()
            timing_by_axis[axis.name] = {
                "axis_started_ns": axis_started_ns,
                "solver_started_ns": solver_started_ns,
                "solver_completed_ns": solver_completed_ns,
                "axis_completed_ns": time.perf_counter_ns(),
            }

        finalize_started_ns = time.perf_counter_ns()
        shard.flush()
        shard.finalize(
            raw_payload={
                "schema_version": STAGE052_SCHEMA_VERSION,
                "run_label": task.run_label,
                "component": task.component,
                "scope": task.scope,
                "instance": task.instance_name,
                "seed": task.seed,
                "worker_count": task.worker_count,
                "axes": raw_axes,
            },
            solution_payload={
                "schema_version": STAGE052_SCHEMA_VERSION,
                "instance": task.instance_name,
                "seed": task.seed,
                "axes": solution_axes,
            },
            trace_payload={
                "trace_schema_version": f"{STAGE052_SCHEMA_VERSION}-trace-v1",
                "campaign_trace_schema_version": (
                    "stage05.2-campaign-trace-v1"
                    if task.component == Stage052Component.BENCHMARK.value
                    else None
                ),
                "run_label": task.run_label,
                "instance": task.instance_name,
                "seed": task.seed,
                "axes": trace_axes,
            },
            environment_payload={
                **collect_environment(),
                "component": task.component,
                "scope": task.scope,
                "worker_count": task.worker_count,
                "peak_rss_bytes": _peak_rss_bytes(),
            },
            failure_payload=(
                {
                    "schema_version": STAGE052_SCHEMA_VERSION,
                    "instance": task.instance_name,
                    "seed": task.seed,
                    "reasons": failures,
                    "evidence_completeness": "complete",
                }
                if failures
                else None
            ),
            anytime_checkpoints=tuple(
                checkpoint
                for raw_axis in raw_axes.values()
                if isinstance(raw_axis, Mapping)
                for checkpoint in raw_axis.get("anytime_checkpoints", [])
                if isinstance(checkpoint, Mapping)
            )
            if task.component == Stage052Component.BENCHMARK.value
            else (),
        )
        finalize_completed_ns = time.perf_counter_ns()
        finalization_seconds = (finalize_completed_ns - finalize_started_ns) / 1_000_000_000
    except BaseException as error:
        if active_trace_stream is not None:
            active_trace_stream.close()
        shard.abort(error)
        raise

    total_events = sum(event_counts.values())
    rows: list[dict[str, object]] = []
    for axis in axes:
        finalization_share = (
            finalization_seconds * event_counts[axis.name] / total_events
            if total_events
            else finalization_seconds / len(axes)
        )
        # Validation, reconciliation, checkpoint construction, and GC are
        # post-solve computation rather than artifact persistence.  Only the
        # live stream writes and the final shard flush/seal are charged to the
        # persistence promotion gate.
        timing = timing_by_axis[axis.name]
        persistence_seconds = finalization_share
        persistence_seconds += live_persistence_ns_by_axis[axis.name] / 1_000_000_000
        row = drafts[axis.name]
        solver_value = row["solver_seconds"]
        if not isinstance(solver_value, (int, float)):
            raise TypeError("solver_seconds must be numeric")
        row["artifact_persistence_seconds"] = persistence_seconds
        row["end_to_end_seconds"] = (
            timing["axis_completed_ns"] - timing["axis_started_ns"]
        ) / 1_000_000_000 + finalization_share
        row["peak_rss_bytes"] = _peak_rss_bytes()
        row["_timing_evidence"] = {
            "instance": task.instance_name,
            "seed": task.seed,
            "axis": axis.name,
            **timing,
            "finalize_started_ns": finalize_started_ns,
            "finalize_completed_ns": finalize_completed_ns,
            "axis_event_count": event_counts[axis.name],
            "live_stream_persistence_ns": live_persistence_ns_by_axis[axis.name],
            "total_event_count": total_events,
            "axis_count": len(axes),
        }
        rows.append(row)
    return rows


def _solve_stage052_axis(
    instance: Instance,
    *,
    seed: int,
    axis: Stage052Axis,
    config: Stage052Config,
    stage04: Any,
    stage02: Any,
    trace_sink: MeasurementTraceSink | None = None,
) -> ALNSResult:
    exact_deadline = (
        ExactDeadlineConfig.fixed_exact_calls(
            axis.exact_call_budget or 100,
            watchdog_seconds=axis.time_limit_seconds,
        )
        if axis.termination_mode == "fixed_work"
        else ExactDeadlineConfig.wall_clock()
    )
    return solve_alns(
        instance,
        seed=seed,
        max_iterations=axis.max_iterations,
        time_limit_seconds=axis.time_limit_seconds,
        operator_profile="stage02_constraint_guided",
        vehicle_operator_config=stage02.vehicle_operator_config,
        measurement_config=MeasurementConfig(
            enabled=(axis.instrumentation_enabled or trace_sink is not None),
            stream_sink=trace_sink,
        ),
        screening_config=stage04.screening_config,
        cache_incremental_config=stage04.cache_incremental_config,
        backend="cpu_batch",
        batch_size=config.batch_size,
        exact_deadline_config=exact_deadline,
        stage04_config=stage04.stage04_config,
        native_kernel_config=(
            config.native_kernels if instance.distance_backend == "native" else None
        ),
        neighborhood_event_sink=(
            trace_sink.append_neighborhood_event
            if isinstance(trace_sink, _Stage052TraceStreamSink)
            else None
        ),
    )


def _checkpoint_objective_key(value: object) -> ObjectiveKey:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise RuntimeError("checkpoint objective must have four components")
    vehicle_count = _strict_int(value[0], "checkpoint vehicle_count")
    charging_count = _strict_int(value[3], "checkpoint charging_count")
    distance = _strict_float(value[1])
    charging_time = _strict_float(value[2])
    if vehicle_count <= 0 or charging_count < 0 or distance < 0.0 or charging_time < 0.0:
        raise RuntimeError("checkpoint objective contains an invalid value")
    return vehicle_count, distance, charging_time, charging_count


def _stage052_initial_objective_key(result: ALNSResult) -> ObjectiveKey:
    trace = result.measurement_trace
    if trace is None or result.initial_objective is None:
        raise RuntimeError("benchmark initial incumbent requires a complete trace/result")
    if not result.initial_routes:
        raise RuntimeError("benchmark initial incumbent routes are missing")
    return _checkpoint_objective_key(result.initial_objective.key)


def _stage052_anytime_checkpoints(
    result: ALNSResult,
    *,
    instance_name: str,
    seed: int,
    axis: Stage052Axis,
    trace_stream: _Stage052TraceStreamSink | None = None,
) -> tuple[AnytimeCheckpoint, ...]:
    """Extract complete accepted incumbents without trusting sampled snapshots."""

    if axis.termination_mode != "wall_clock":
        return ()
    trace = result.measurement_trace
    if trace is None or result.objective is None:
        raise RuntimeError("benchmark checkpoint extraction requires a complete trace/result")
    initial_objective = _stage052_initial_objective_key(result)
    if trace_stream is None:
        candidate_events = [
            event
            for event in trace.events
            if isinstance(event, Mapping) and event.get("event_type") == "candidate_state"
        ]
        accepted_global_bests = tuple(
            AcceptedGlobalBest(
                completed_at_seconds=_strict_float(event.get("timestamp_seconds")),
                iteration=_strict_int(event.get("iteration"), "checkpoint iteration"),
                objective_key=_checkpoint_objective_key(event.get("candidate_objective_key")),
            )
            for event in candidate_events
            if event.get("accepted") is True and event.get("global_best") is True
        )
    else:
        accepted_global_bests = trace_stream.accepted_checkpoint_bests
    max_iterations_completed_at_seconds: float | None = None
    final_objective_key: ObjectiveKey | None = None
    if result.termination_reason == "iteration_limit":
        max_iterations_completed_at_seconds = result.iteration_limit_completed_at_seconds
        if max_iterations_completed_at_seconds is None:
            raise RuntimeError("iteration-limit result is missing its completion timestamp")
        final_objective_key = _checkpoint_objective_key(result.objective.key)
    budget = int(axis.time_limit_seconds)
    if float(budget) != axis.time_limit_seconds:
        raise RuntimeError("benchmark checkpoint budget must be an integer number of seconds")
    return AnytimeCheckpoint.for_axis(
        instance=instance_name,
        seed=seed,
        axis_budget_seconds=budget,
        initial_objective_key=initial_objective,
        accepted_global_bests=accepted_global_bests,
        max_iterations_completed_at_seconds=max_iterations_completed_at_seconds,
        final_objective_key=final_objective_key,
    )


def _iter_stage052_axis_events(
    *,
    trace: object,
    neighborhood_events: Iterable[Mapping[str, object]],
    route_dictionary: dict[str, tuple[str, ...]],
    axis_name: str,
    diagnostic_counts: Counter[tuple[str, str, str, str]],
    digest: Any,
) -> Iterable[dict[str, object]]:
    for event in iter_stage03_critical_events(trace, neighborhood_events):
        record = _route_reference_event(event, route_dictionary, copy_event=False)
        record["lane"] = f"{axis_name}:{record.get('lane', '')}"
        record["benchmark_axis"] = axis_name
        event_type = str(record.get("event_type", record.get("record_type", "event")))
        if event_type not in {
            "execution_error",
            "deadline_boundary",
            "cache_event",
            "screening_decision",
            "route_evaluation",
            "incremental_propagation",
        }:
            diagnostic_counts[
                (
                    str(record.get("lane", "")),
                    str(record.get("operator", "")),
                    str(record.get("reason", record.get("status", ""))),
                    event_type,
                )
            ] += 1
        if event_type in {
            "candidate_state",
            "cache_event",
            "exact_budget_boundary",
        }:
            digest.update(
                json.dumps(
                    {
                        "event_type": event_type,
                        "status": record.get("status"),
                        "accepted": record.get("accepted"),
                        "global_best": record.get("global_best"),
                        "operation": record.get("operation"),
                        "cache_key_digest": record.get("cache_key_digest"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
        yield record


def _stage052_row_draft(
    *,
    task: _ShardTask,
    axis: Stage052Axis,
    result: ALNSResult,
    validation_passed: bool,
    axis_valid: bool,
    solver_seconds: float,
    semantic_digest: str,
    storage: ArtifactStorageConfig,
) -> dict[str, object]:
    backend = result.backend_metrics
    objective = result.objective
    batch_launches, median_batch_occupancy = _launch_occupancy_summary(backend)
    return {
        "instance": task.instance_name,
        "seed": task.seed,
        "axis": axis.name,
        "customer_count": task.customer_count,
        "component": task.component,
        "backend": result.charging_backend,
        "worker_count": task.worker_count,
        "storage_policy_version": storage.storage_policy_version,
        "persistence_attribution": "axis_stream_plus_finalization_rows",
        "solver_seconds": solver_seconds,
        "artifact_persistence_seconds": 0.0,
        "end_to_end_seconds": solver_seconds,
        "screening_seconds": result.screening_statistics.get("screening_runtime_seconds", 0.0),
        "exact_seconds": backend.get("total_seconds", 0.0),
        "packing_seconds": backend.get("packing_seconds", 0.0),
        "unpacking_seconds": backend.get("unpacking_seconds", 0.0),
        "native_kernel_seconds": backend.get("native_kernel_seconds", 0.0),
        "native_invocations": backend.get("native_invocations", 0),
        "native_fallbacks": backend.get("native_fallbacks", 0),
        "native_screening_seconds": result.screening_statistics.get(
            "native_screening_seconds", 0.0
        ),
        "native_screening_invocations": result.screening_statistics.get(
            "native_screening_invocations", 0
        ),
        "native_propagation_seconds": result.screening_statistics.get(
            "native_propagation_seconds", 0.0
        ),
        "native_propagation_invocations": result.screening_statistics.get(
            "native_propagation_invocations", 0
        ),
        "native_protocol_fallbacks": result.screening_statistics.get(
            "native_protocol_fallbacks", 0
        ),
        "exact_started_calls": result.exact_started_calls,
        "exact_completed_calls": result.exact_completed_calls,
        "effective_iterations": result.effective_iterations,
        "batch_launches": batch_launches,
        "median_batch_occupancy": median_batch_occupancy,
        "peak_rss_bytes": 0,
        "vehicle_count": objective.vehicle_count if objective else "",
        "total_distance": objective.total_distance if objective else "",
        "total_charging_time": objective.total_charging_time if objective else "",
        "charging_count": objective.charging_count if objective else "",
        "validator_passed": validation_passed,
        "semantic_digest": semantic_digest,
        "termination_reason": result.termination_reason,
        "failure_status": ("" if axis_valid else "solver_validator_or_trace_reconciliation_failed"),
    }


def _persist_shard(
    writer: ArtifactBundleWriter,
    *,
    task: _ShardTask,
    instance: Instance,
    results: Mapping[str, ALNSResult],
    storage: ArtifactStorageConfig,
) -> tuple[dict[str, str], dict[str, int]]:
    route_dictionary: dict[str, tuple[str, ...]] = {}
    critical_events: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    raw_axes: dict[str, object] = {}
    solution_axes: dict[str, object] = {}
    trace_axes: dict[str, object] = {}
    semantic_digests: dict[str, str] = {}
    event_counts: dict[str, int] = {}
    failures: list[str] = []
    for axis, result in results.items():
        trace = result.measurement_trace
        if trace is None:
            raise RuntimeError(f"Stage 5.2 {axis} result is missing its trace")
        route_dictionary.update(trace.route_dictionary)
        events = build_stage03_critical_events(trace, result.neighborhood_events)
        normalized_events: list[dict[str, object]] = []
        for event in events:
            record = _route_reference_event(event, route_dictionary)
            record["lane"] = f"{axis}:{record.get('lane', '')}"
            record["benchmark_axis"] = axis
            normalized_events.append(record)
        critical_events.extend(normalized_events)
        event_counts[axis] = len(normalized_events)
        diagnostic_rows.extend(
            aggregate_diagnostic_events(
                normalized_events,
                run_label=task.run_label,
                instance=task.instance_name,
                seed=task.seed,
            )
        )
        objective_key = list(result.objective.key) if result.objective else []
        validation = validate_routes(instance, [list(route) for route in result.routes])
        semantic_payload = {
            "objective_key": objective_key,
            "routes": [list(route) for route in result.routes],
            "started_calls": result.exact_started_calls,
            "completed_calls": result.exact_completed_calls,
            "candidate_decisions": [
                {
                    "event_type": event.get("event_type"),
                    "status": event.get("status"),
                    "accepted": event.get("accepted"),
                    "global_best": event.get("global_best"),
                    "operation": event.get("operation"),
                    "cache_key_digest": event.get("cache_key_digest"),
                }
                for event in normalized_events
                if event.get("event_type")
                in {"candidate_state", "cache_event", "exact_budget_boundary"}
            ],
        }
        semantic_digest = hashlib.sha256(
            json.dumps(
                semantic_payload,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        semantic_digests[axis] = semantic_digest
        reconciliation = trace.reconcile(result)
        valid = result.feasible and validation.feasible and reconciliation["status"] == "pass"
        if not valid:
            failures.append(f"{axis}: solver, validator, or trace reconciliation failed")
        raw_axes[axis] = {
            "backend": result.charging_backend,
            "backend_metrics": result.backend_metrics,
            "objective_key": objective_key,
            "runtime_seconds": result.runtime_seconds,
            "effective_iterations": result.effective_iterations,
            "unique_route_semantics": result.unique_route_semantics,
            "started_calls": result.exact_started_calls,
            "completed_calls": result.exact_completed_calls,
            "termination_reason": result.termination_reason,
            "semantic_digest": semantic_digest,
            "trace_reconciliation": reconciliation,
            "validator_passed": validation.feasible,
            "valid": valid,
        }
        solution_axes[axis] = {
            "routes": [list(route) for route in result.routes],
            "objective_key": objective_key,
            "feasible": result.feasible,
        }
        trace_payload = trace.to_dict()
        trace_axes[axis] = {
            "config": trace_payload["config"],
            "summary": trace_payload["summary"],
            "result_summary": trace_payload["result_summary"],
        }
    shard_ordinal: int | None = None
    worker_identity: str | None = None
    if storage.storage_policy_version == ARTIFACT_STORAGE_V2:
        shard_ordinal = task.shard_ordinal
        worker_identity = f"pid-{task.shard_ordinal % max(task.worker_count, 1)}"
    writer.write_instance_seed(
        instance=task.instance_name,
        seed=task.seed,
        raw_payload={
            "schema_version": STAGE052_SCHEMA_VERSION,
            "run_label": task.run_label,
            "component": task.component,
            "scope": task.scope,
            "instance": task.instance_name,
            "seed": task.seed,
            "worker_count": task.worker_count,
            "axes": raw_axes,
        },
        solution_payload={
            "schema_version": STAGE052_SCHEMA_VERSION,
            "instance": task.instance_name,
            "seed": task.seed,
            "axes": solution_axes,
        },
        trace_payload={
            "trace_schema_version": f"{STAGE052_SCHEMA_VERSION}-trace-v1",
            "axes": trace_axes,
        },
        environment_payload={
            **collect_environment(),
            "component": task.component,
            "scope": task.scope,
            "worker_count": task.worker_count,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
        route_dictionary=route_dictionary,
        critical_events=critical_events,
        diagnostic_rows=diagnostic_rows,
        failure_payload=(
            {
                "schema_version": STAGE052_SCHEMA_VERSION,
                "instance": task.instance_name,
                "seed": task.seed,
                "reasons": failures,
                "evidence_completeness": "complete",
            }
            if failures
            else None
        ),
        shard_ordinal=shard_ordinal,
        worker_identity=worker_identity,
    )
    return semantic_digests, event_counts


def _route_reference_event(
    event: Mapping[str, object],
    route_dictionary: dict[str, tuple[str, ...]],
    *,
    copy_event: bool = True,
) -> dict[str, object]:
    from evrptw.measurement import canonical_route_key

    output = dict(event) if copy_event or not isinstance(event, dict) else event
    raw_sequences = output.pop("customer_sequences", None)
    if raw_sequences is None:
        return output
    if not isinstance(raw_sequences, (list, tuple)):
        raise TypeError("candidate customer_sequences must be a sequence")
    route_keys: list[str] = []
    for raw_sequence in raw_sequences:
        if not isinstance(raw_sequence, (list, tuple)) or not all(
            isinstance(node, str) for node in raw_sequence
        ):
            raise TypeError("candidate route must contain node identifiers")
        sequence = tuple(raw_sequence)
        route_key = canonical_route_key(sequence)
        route_dictionary.setdefault(route_key, sequence)
        route_keys.append(route_key)
    output["route_keys"] = route_keys
    return output


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_clean_repository(root: Path) -> None:
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("Stage 5.2 runner requires a clean repository commit")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if value > 10_000_000 else value * 1024


def _optimization_profile(component: Stage052Component) -> str:
    if component is Stage052Component.PERF_BASELINE:
        return "none"
    if component in {
        Stage052Component.HOT_PATH,
        Stage052Component.ARTIFACT_STREAMING,
        Stage052Component.JOB_PARALLEL,
    }:
        return "python"
    return "native"


def _launch_occupancy_summary(backend: Mapping[str, object]) -> tuple[int, float]:
    """Return the real per-launch median, never an exact-calls/launches mean."""

    batch_launches = _strict_int(backend.get("batch_launches", 0), "batch_launches")
    exact_calls = _strict_int(backend.get("exact_calls", 0), "exact_calls")
    raw = backend.get("launch_occupancies")
    if not isinstance(raw, list):
        raise TypeError("launch_occupancies must be a list")
    occupancies = [_strict_int(value, "launch_occupancy") for value in raw]
    if any(value <= 0 for value in occupancies):
        raise ValueError("launch occupancies must be positive")
    if len(occupancies) != batch_launches or sum(occupancies) != exact_calls:
        raise ValueError(
            "launch occupancy count/sum must reconcile with batch_launches and exact_calls"
        )
    return batch_launches, statistics.median(occupancies) if occupancies else 0.0


def _strict_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    raise TypeError(f"{field} must be an integer")


def _strict_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("numeric value cannot be boolean")
    if isinstance(value, (int, float, str)):
        try:
            number = float(value)
        except ValueError as error:
            raise TypeError("numeric value must be parseable") from error
        if not math.isfinite(number):
            raise ValueError("numeric value must be finite")
        return number
    raise TypeError("numeric value must be a number")


def _parse_prerequisite_bindings(values: Sequence[str]) -> dict[str, Path]:
    bindings: dict[str, Path] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not role or not raw_path:
            raise ValueError("named prerequisite must use ROLE=PATH")
        if role in bindings:
            raise ValueError(f"duplicate prerequisite role: {role}")
        bindings[role] = Path(raw_path)
    return bindings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Stage 5.2 component")
    parser.add_argument("--config", type=Path, default=Path("configs/stage052_performance.toml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument(
        "--component",
        choices=tuple(component.value for component in Stage052Component),
        required=True,
    )
    parser.add_argument("--scope", choices=("performance", "pilot", "formal"), required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--prerequisite-dir", type=Path)
    parser.add_argument("--prerequisite", action="append", default=[])
    arguments = parser.parse_args()
    named_prerequisites = _parse_prerequisite_bindings(arguments.prerequisite)
    outputs = run_stage052(
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        run_label=arguments.run_label,
        component=arguments.component,
        scope=arguments.scope,
        worker_count=arguments.workers,
        prerequisite_dir=arguments.prerequisite_dir,
        prerequisite_dirs=named_prerequisites,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
