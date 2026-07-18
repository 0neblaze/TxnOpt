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
import statistics
import subprocess
import time
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from evrptw.alns import ALNSResult, solve_alns
from evrptw.artifacts import (
    ARTIFACT_STORAGE_V2,
    ArtifactBundleWriter,
    ArtifactReader,
    ArtifactRunContext,
    ArtifactStorageConfig,
    aggregate_diagnostic_events,
    build_stage03_critical_events,
    iter_stage03_critical_events,
)
from evrptw.best_known import BEST_KNOWN_VALUES
from evrptw.environment import collect_environment
from evrptw.exact_deadline import ExactDeadlineConfig
from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES
from evrptw.experiments.stage02_route_reduction import (
    load_config as load_stage02_config,
)
from evrptw.experiments.stage04_weights import load_stage04_config
from evrptw.measurement import MeasurementConfig
from evrptw.models import Instance
from evrptw.native_kernels import NativeKernelConfig
from evrptw.parser import parse_schneider
from evrptw.stage052 import Stage052Component, formal_budget_matrix
from evrptw.stage052_evidence import (
    ProcessTreeResourceSampler,
    RunResourceSummary,
    abort_process_executor,
    collect_performance_provenance,
    verify_job_parallel_selection,
    verify_stage052_prerequisite,
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


@dataclass(frozen=True, slots=True)
class Stage052Axis:
    name: str
    termination_mode: str
    time_limit_seconds: float
    exact_call_budget: int | None = None
    instrumentation_enabled: bool = True


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
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid Stage 5.2 configuration: {error}") from error
    if config.v1_storage.storage_policy_version != "artifact-storage-v1":
        raise ValueError("Stage 5.2 v1 baseline storage must use artifact-storage-v1")
    if config.v2_storage.storage_policy_version != ARTIFACT_STORAGE_V2:
        raise ValueError("Stage 5.2 streaming storage must use artifact-storage-v2")
    if config.max_iterations != 1000 or config.batch_size <= 0:
        raise ValueError("Stage 5.2 requires 1000 iterations and a positive batch size")
    return config


def validate_stage052_run_label(run_label: str, component: Stage052Component | str) -> None:
    selected = Stage052Component(component)
    pattern = re.compile(rf"^stage05\.2_{re.escape(selected.value)}_(?:attempt|rerun)[0-9]{{2}}$")
    if pattern.fullmatch(run_label) is None:
        raise ValueError(
            "Stage 5.2 run label must be canonical: "
            f"stage05.2_{selected.value}_attemptNN or rerunNN"
        )


def axes_for_scope(scope: str, *, customer_count: int | None = None) -> tuple[Stage052Axis, ...]:
    if scope == "performance":
        return (
            Stage052Axis("fixed_work_control", "fixed_work", 120.0, 100, False),
            Stage052Axis("fixed_work", "fixed_work", 120.0, 100),
            Stage052Axis("wall_clock_30", "wall_clock", 30.0),
        )
    if scope == "pilot":
        return (Stage052Axis("wall_clock_30", "wall_clock", 30.0),)
    if scope == "formal":
        if customer_count is None:
            raise ValueError("formal scope requires customer_count")
        return tuple(
            Stage052Axis(f"wall_clock_{budget}", "wall_clock", float(budget))
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
) -> dict[str, Path]:
    """Execute one canonical Stage 5.2 component attempt."""

    selected = Stage052Component(component)
    validate_stage052_run_label(run_label, selected)
    if worker_count not in {1, 2, 4}:
        raise ValueError("Stage 5.2 worker_count must be 1, 2, or 4")
    root = Path(__file__).resolve().parents[3]
    resolved_config = _resolve(root, config_path)
    resolved_output = _resolve(root, output_dir)
    if resolved_output != root / "results" / run_label:
        raise ValueError("Stage 5.2 output must be results/<canonical-run-label>")
    if resolved_output.exists():
        raise FileExistsError(resolved_output)
    _require_clean_repository(root)
    config = load_stage052_config(resolved_config)
    prerequisite = verify_stage051_prerequisite(_resolve(root, config.stage051_manifest))
    component_prerequisite = None
    job_parallel_selection = None
    accelerator_decision_payload: dict[str, object] | None = None
    prerequisite_contract = {
        Stage052Component.JOB_PARALLEL: (
            "artifact_streaming",
            "READY_FOR_STAGE052_JOB_PARALLEL",
        ),
        Stage052Component.NATIVE_KERNELS: (
            "job_parallel",
            "READY_FOR_STAGE052_NATIVE_KERNELS",
        ),
        Stage052Component.ACCELERATOR_PILOT: (
            "native_kernels",
            "READY_FOR_STAGE052_ACCELERATOR_DECISION",
        ),
    }.get(selected)
    if prerequisite_contract is not None:
        if prerequisite_dir is None:
            raise ValueError(f"{selected.value} requires --prerequisite-dir")
        expected_component, expected_status = prerequisite_contract
        component_prerequisite = verify_stage052_prerequisite(
            _resolve(root, prerequisite_dir),
            expected_component=expected_component,
            expected_status=expected_status,
            expected_run_label=(
                "stage05.2_artifact_streaming_attempt04"
                if selected is Stage052Component.JOB_PARALLEL
                else None
            ),
        )
        if selected is Stage052Component.NATIVE_KERNELS:
            job_parallel_selection = verify_job_parallel_selection(
                _resolve(root, prerequisite_dir), component_prerequisite
            )
            if worker_count != job_parallel_selection.selected_workers:
                raise ValueError(
                    "native_kernels worker_count must equal the independently selected "
                    f"D worker count ({job_parallel_selection.selected_workers})"
                )
        if selected is Stage052Component.ACCELERATOR_PILOT:
            if scope != "performance":
                raise ValueError("accelerator_pilot decision uses performance scope")
            prerequisite_path = _resolve(root, prerequisite_dir)
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
                prerequisite=component_prerequisite,
            )
            if _strict_float(accelerator_decision_payload["median_batch_occupancy"]) >= 32.0:
                raise RuntimeError(
                    "route-batch occupancy requires a real Metal pilot; "
                    "decision-only evidence is forbidden"
                )
    instances, seeds = _scope_identities(scope)
    if selected in {
        Stage052Component.PERF_BASELINE,
        Stage052Component.HOT_PATH,
    }:
        storage = config.v1_storage
        if worker_count != 1:
            raise ValueError("perf_baseline and hot_path are single-worker evidence")
    else:
        storage = config.v2_storage
    if selected is Stage052Component.JOB_PARALLEL and scope != "performance":
        raise ValueError("job_parallel selection uses the fixed performance scope")
    if scope == "formal" and selected is not Stage052Component.BENCHMARK:
        raise ValueError("only the benchmark component may run Formal scope")

    resolved_output.mkdir(parents=True)
    context = ArtifactRunContext("stage05.2", selected.value, run_label)
    parent_writer = ArtifactBundleWriter(resolved_output, context, storage)
    revision = _git(root, "rev-parse", "HEAD")
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
        "backend": "cpu_batch",
        "optimization_profile": _optimization_profile(selected),
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
        "persistence_attribution": "critical_event_rows",
        "repository_revision": revision,
        "repository_dirty": False,
        "configuration_sha256": _sha256(resolved_config),
        "stage051_prerequisite": prerequisite,
        "component_prerequisite": (
            component_prerequisite.to_dict() if component_prerequisite is not None else None
        ),
        "job_parallel_selection": (
            job_parallel_selection.to_dict() if job_parallel_selection is not None else None
        ),
        "accelerator_decision_mode": (
            "decision_only" if accelerator_decision_payload is not None else None
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
    parent_writer.write_control(metadata=metadata, configuration_path=resolved_config)
    if accelerator_decision_payload is not None:
        decision_path = resolved_output / "control" / f"{run_label}_accelerator_decision.json"
        decision_path.write_text(
            json.dumps(accelerator_decision_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        parent_writer.record_existing_file(
            decision_path,
            artifact_type="accelerator_decision",
            retention_class="control",
            storage_format="json_control",
        )
        bundle = parent_writer.finalize()
        return {
            "run_dir": bundle.run_dir,
            "accelerator_decision": decision_path,
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
            Stage052Component.JOB_PARALLEL,
            Stage052Component.NATIVE_KERNELS,
            Stage052Component.ACCELERATOR_PILOT,
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
    rows.sort(
        key=lambda row: (
            str(row["instance"]),
            _strict_int(row["seed"], "seed"),
            str(row["axis"]),
        )
    )
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
        resource_summary_path = _record_resource_summary(parent_writer, summary)
    bundle = parent_writer.finalize()
    outputs = {
        "run_dir": bundle.run_dir,
        "per_run_results": per_run_path,
        "manifest": bundle.manifest_path,
        "manifest_sidecar": bundle.manifest_sidecar_path,
    }
    if resource_summary_path is not None:
        outputs["resource_summary"] = resource_summary_path
    if timing_evidence_path is not None:
        outputs["timing_evidence"] = timing_evidence_path
    return outputs


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
) -> dict[str, object]:
    """Recompute the nine E fixed-work occupancy values without GPU execution."""

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
    if median >= 32.0:
        raise RuntimeError(
            "median route count per exact launch requires a real Metal pilot; "
            "GPU_NOT_JUSTIFIED decision-only evidence is forbidden"
        )
    prerequisite_dict: dict[str, object] = getattr(prerequisite, "to_dict", lambda: {})()
    return {
        "schema_version": "stage05.2-accelerator-decision-v1",
        "decision_mode": "decision_only",
        "decision": "GPU_NOT_JUSTIFIED",
        "threshold": 32.0,
        "median_batch_occupancy": median,
        "input_count": len(ordered),
        "inputs": ordered,
        "native_prerequisite": prerequisite_dict,
        "gpu_rows_present": False,
        "fallback_used": False,
    }


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


def _run_v2_tasks(tasks: Sequence[_ShardTask], *, worker_count: int) -> list[dict[str, object]]:
    if worker_count == 1:
        return [row for task in tasks for row in _run_v2_shard_task(task)]
    rows: list[dict[str, object]] = []
    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_context("spawn"),
    )
    futures: dict[Any, _ShardTask] = {}
    try:
        futures = {executor.submit(_run_v2_shard_task, task): task for task in tasks}
        for future in as_completed(futures):
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


def _run_v2_shard_task(task: _ShardTask) -> list[dict[str, object]]:
    try:
        return _run_and_persist_shard(task)
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
            max_iterations=config.max_iterations,
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
    event_counts: dict[str, int] = {}
    failures: list[str] = []
    try:
        for axis in axes:
            axis_started_ns = time.perf_counter_ns()
            solver_started_ns = axis_started_ns
            result = _solve_stage052_axis(
                instance,
                seed=task.seed,
                axis=axis,
                config=config,
                stage04=stage04,
                stage02=stage02,
            )
            solver_completed_ns = time.perf_counter_ns()
            solver_seconds = (solver_completed_ns - solver_started_ns) / 1_000_000_000
            trace = result.measurement_trace
            if trace is None:
                raise RuntimeError(f"Stage 5.2 {axis.name} result is missing its trace")
            objective_key = list(result.objective.key) if result.objective else []
            validation = validate_routes(instance, [list(route) for route in result.routes])
            reconciliation = trace.reconcile(result)
            valid = result.feasible and validation.feasible and reconciliation["status"] == "pass"
            if not valid:
                failures.append(f"{axis.name}: solver, validator, or trace reconciliation failed")
            digest = hashlib.sha256()
            digest.update(
                json.dumps(
                    {
                        "objective_key": objective_key,
                        "routes": [list(route) for route in result.routes],
                        "started_calls": result.exact_started_calls,
                        "completed_calls": result.exact_completed_calls,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
            diagnostic_counts: Counter[tuple[str, str, str, str]] = Counter()
            route_dictionary = dict(trace.route_dictionary)

            event_counts[axis.name] = shard.append(
                route_dictionary=route_dictionary,
                critical_events=_iter_stage052_axis_events(
                    trace=trace,
                    neighborhood_events=result.neighborhood_events,
                    route_dictionary=route_dictionary,
                    axis_name=axis.name,
                    diagnostic_counts=diagnostic_counts,
                    digest=digest,
                ),
            )
            diagnostic_rows = [
                {
                    "run_label": task.run_label,
                    "instance": task.instance_name,
                    "seed": task.seed,
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
                for (lane, operator, reason, event_type), count in sorted(diagnostic_counts.items())
            ]
            shard.append(
                route_dictionary={},
                critical_events=(),
                diagnostic_rows=diagnostic_rows,
            )
            semantic_digest = digest.hexdigest()
            raw_axes[axis.name] = {
                "backend": result.charging_backend,
                "backend_metrics": result.backend_metrics,
                "objective_key": objective_key,
                "runtime_seconds": result.runtime_seconds,
                "effective_iterations": result.effective_iterations,
                "started_calls": result.exact_started_calls,
                "completed_calls": result.exact_completed_calls,
                "termination_reason": result.termination_reason,
                "semantic_digest": semantic_digest,
                "trace_reconciliation": reconciliation,
                "validator_passed": validation.feasible,
                "valid": valid,
            }
            solution_axes[axis.name] = {
                "routes": [list(route) for route in result.routes],
                "objective_key": objective_key,
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
        )
        finalize_completed_ns = time.perf_counter_ns()
        finalization_seconds = (finalize_completed_ns - finalize_started_ns) / 1_000_000_000
    except BaseException as error:
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
        timing = timing_by_axis[axis.name]
        post_solver_seconds = (
            timing["axis_completed_ns"] - timing["solver_completed_ns"]
        ) / 1_000_000_000
        persistence_seconds = post_solver_seconds + finalization_share
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
        max_iterations=config.max_iterations,
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
    arguments = parser.parse_args()
    outputs = run_stage052(
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        run_label=arguments.run_label,
        component=arguments.component,
        scope=arguments.scope,
        worker_count=arguments.workers,
        prerequisite_dir=arguments.prerequisite_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
