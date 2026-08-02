"""Real-shard Stage 5.2 producer and persistence calibration.

The calibration corpus is read-only and never contributes campaign geometry or
readiness counts.  Its only durable output is a signed resource contract that
new Pilot and Formal campaigns freeze into their configuration and manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import psutil  # type: ignore[import-untyped]
import pyarrow.parquet as pq

from evrptw.artifacts import (
    ARTIFACT_STORAGE_V2,
    SCREENING_DECISIONS_V3,
    ArtifactStorageConfig,
    _StreamingParquetSink,
    atomic_write_signed_json,
    signed_sidecar_matches,
)
from evrptw.experiments.stage02_route_reduction import (
    load_config as load_stage02_config,
)
from evrptw.experiments.stage04_weights import load_stage04_config
from evrptw.experiments.stage052_performance import (
    Stage052Axis,
    _build_tasks,
    _optimization_profile,
    _run_and_persist_v2_shard,
    _run_v2_tasks,
    _ShardTask,
    axes_for_scope,
    load_stage052_config,
)
from evrptw.parser import parse_schneider
from evrptw.repository import repository_root
from evrptw.stage052 import Stage052Component
from evrptw.stage052_evidence import (
    CgroupV2MemorySource,
    ProcessTreeResourceSampler,
    RunResourceSummary,
    is_stage052_dedicated_cgroup_path,
)
from evrptw.stage052_memory import (
    release_stage052_process_memory,
    stage052_python_allocator,
)
from evrptw.stage052_resources import (
    ParquetBenchmark,
    ProducerBenchmark,
    ProducerResourceContract,
    ProducerSelection,
    derive_producer_resource_contract,
    select_parquet_configuration,
    select_producer_configuration,
)
from evrptw.storage_governance import (
    preflight_cli_attempt,
    seal_cli_attempt,
    seal_failed_cli_attempt,
)

_CALIBRATION_INSTANCES = ("c101_21", "r101_21", "rc101_21")
_CALIBRATION_SEEDS = (2014, 2015)
_FORMAL_MEMORY_INSTANCE = "r205_21"
_FORMAL_MEMORY_SEEDS = tuple(range(2014, 2022))
_PRODUCER_WORKERS = (4, 5, 6)
_PRODUCER_PROBE_WORKERS = (4, 5, 6, 8)
_PARQUET_CONFIGURATIONS = (
    (16_384, 1),
    (16_384, 2),
    (65_536, 1),
    (65_536, 2),
    (262_144, 1),
    (262_144, 2),
)
_FAILED_FORMAL_MEMORY_FLOOR_REASONS = (
    "aggregate RSS exceeds its campaign lock",
    "runtime guard aborted Stage 5.2 work: aggregate RSS hard limit exceeded:",
    "runtime guard aborted Stage 5.2 work: process RSS hard limit exceeded:",
    "runtime guard aborted Stage 5.2 work: cgroup v2 memory hard limit exceeded:",
)


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"calibration {field_name} is not an integer")
    return value


def _is_failed_formal_memory_floor_reason(reason: object) -> bool:
    """Recognize only fail-fast producer memory-limit failures."""

    return isinstance(reason, str) and any(
        marker in reason for marker in _FAILED_FORMAL_MEMORY_FLOOR_REASONS
    )


@dataclass(frozen=True, slots=True)
class MeasuredProducerCandidate:
    benchmark: ProducerBenchmark
    per_worker_peak_rss_bytes: int
    aggregate_memory_source: str = "process_tree_rss_telemetry"
    cgroup_path: str | None = None
    cgroup_swap_peak_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.per_worker_peak_rss_bytes <= 0:
            raise ValueError("per-worker peak RSS must be positive")
        if self.aggregate_memory_source not in {
            "cgroup_v2",
            "process_tree_rss_telemetry",
        }:
            raise ValueError("aggregate memory source is invalid")
        if self.aggregate_memory_source == "cgroup_v2":
            if (
                self.cgroup_path is None
                or not is_stage052_dedicated_cgroup_path(self.cgroup_path)
                or self.cgroup_swap_peak_bytes is None
                or self.cgroup_swap_peak_bytes < 0
            ):
                raise ValueError("cgroup v2 aggregate memory evidence is incomplete")
        elif self.cgroup_path is not None or self.cgroup_swap_peak_bytes is not None:
            raise ValueError("RSS telemetry cannot claim cgroup v2 evidence")


class FormalMemoryMeasurementError(RuntimeError):
    """A failed Formal probe with its completed resource observation attached."""

    def __init__(self, cause: BaseException, resource: RunResourceSummary) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.resource_summary = resource.to_dict()


@dataclass(frozen=True, slots=True)
class ProducerMemoryFloor:
    four_worker_aggregate_peak_rss_bytes: int
    per_worker_peak_rss_bytes: int
    source_sha256_by_batch: dict[str, str]

    def __post_init__(self) -> None:
        if (
            self.four_worker_aggregate_peak_rss_bytes <= 0
            or self.per_worker_peak_rss_bytes <= 0
            or set(self.source_sha256_by_batch) != {"batch0001", "batch0002"}
        ):
            raise ValueError("producer memory floor is incomplete")

    def projected_aggregate_peak_rss_bytes(self, workers: int) -> int:
        if workers not in _PRODUCER_PROBE_WORKERS:
            raise ValueError("producer memory floor workers must be 4, 5, 6, or 8")
        return ceil(self.four_worker_aggregate_peak_rss_bytes * workers / 4)


@dataclass(frozen=True, slots=True)
class FormalCampaignMemoryFloor:
    """One sealed, failed Formal batch observation with typed memory telemetry."""

    workers: int
    aggregate_peak_rss_bytes: int
    per_worker_peak_rss_bytes: int
    row_group_size: int
    queue_depth: int
    run_label: str
    batch_id: str
    resource_summary_sha256: str
    campaign_geometry_contribution: int = 0
    aggregate_memory_source: str = "process_tree_rss"
    aggregate_peak_memory_bytes: int | None = None
    cgroup_path: str | None = None
    cgroup_swap_peak_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.workers not in _PRODUCER_PROBE_WORKERS:
            raise ValueError("Formal campaign memory floor workers are invalid")
        if (
            self.aggregate_peak_rss_bytes <= 0
            or self.per_worker_peak_rss_bytes <= 0
        ):
            raise ValueError("Formal campaign memory floor RSS values must be positive")
        if self.row_group_size not in (16_384, 65_536, 262_144):
            raise ValueError("Formal campaign memory floor row group size is invalid")
        if self.queue_depth not in (1, 2):
            raise ValueError("Formal campaign memory floor queue depth is invalid")
        if (
            not self.run_label
            or not self.batch_id
            or len(self.resource_summary_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.resource_summary_sha256
            )
        ):
            raise ValueError("Formal campaign memory floor identity is invalid")
        if self.campaign_geometry_contribution != 0:
            raise ValueError("failed Formal memory floor cannot contribute campaign geometry")
        if self.aggregate_memory_source not in {"process_tree_rss", "cgroup_v2"}:
            raise ValueError("failed Formal aggregate memory source is invalid")
        if self.aggregate_memory_source == "cgroup_v2":
            if (
                self.aggregate_peak_memory_bytes is None
                or self.aggregate_peak_memory_bytes <= 0
                or self.cgroup_path is None
                or not is_stage052_dedicated_cgroup_path(self.cgroup_path)
                or self.cgroup_swap_peak_bytes != 0
            ):
                raise ValueError("failed Formal cgroup v2 memory evidence is incomplete")
        elif any(
            value is not None
            for value in (
                self.aggregate_peak_memory_bytes,
                self.cgroup_path,
                self.cgroup_swap_peak_bytes,
            )
        ):
            raise ValueError("failed Formal process RSS evidence claims cgroup fields")


def _load_signed_json_object(path: Path, *, evidence_name: str) -> dict[str, object]:
    if not signed_sidecar_matches(path, path.with_suffix(".sha256")):
        raise RuntimeError(f"{evidence_name} seal is invalid")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{evidence_name} is unreadable") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{evidence_name} is not a JSON object")
    return payload


def load_failed_formal_memory_floor(
    campaign_dir: Path,
    *,
    batch_id: str,
) -> FormalCampaignMemoryFloor:
    """Verify a failed Formal evidence chain and expose typed memory high-water marks."""

    if (
        len(batch_id) != len("batch0000")
        or not batch_id.startswith("batch")
        or not batch_id.removeprefix("batch").isdigit()
    ):
        raise RuntimeError("failed Formal batch ID is invalid")
    campaign_dir = campaign_dir.resolve(strict=True)
    campaign_manifest = _load_signed_json_object(
        campaign_dir / "campaign_manifest.json",
        evidence_name="failed Formal campaign manifest",
    )
    run_label = campaign_manifest.get("run_label")
    failure_reason = campaign_manifest.get("failure_reason")
    selected_workers = campaign_manifest.get("selected_workers")
    campaign_contract = campaign_manifest.get("producer_resource_contract")
    batches = campaign_manifest.get("batches")
    if (
        not isinstance(run_label, str)
        or re.fullmatch(
            r"stage05\.2_benchmark_(?:attempt|rerun)\d{2}",
            run_label,
        )
        is None
        or campaign_manifest.get("schema_version")
        != "stage05.2-campaign-manifest-v2"
        or campaign_manifest.get("scope") != "formal"
        or campaign_manifest.get("status") != "failed"
        or not _is_failed_formal_memory_floor_reason(failure_reason)
        or not isinstance(selected_workers, int)
        or isinstance(selected_workers, bool)
        or not isinstance(campaign_contract, dict)
        or campaign_contract.get("selected_workers") != selected_workers
        or not isinstance(batches, list)
    ):
        raise RuntimeError("failed Formal campaign identity is invalid")

    matching_batches = [
        item
        for item in batches
        if isinstance(item, dict) and item.get("batch_id") == batch_id
    ]
    if (
        len(matching_batches) != 1
        or matching_batches[0].get("run_label") != run_label
        or matching_batches[0].get("status") != "failed"
        or matching_batches[0].get("failure_reason") != failure_reason
    ):
        raise RuntimeError("failed Formal campaign batch binding is invalid")

    batch_dir = campaign_dir / batch_id
    batch_manifest = _load_signed_json_object(
        batch_dir / "batch_manifest.json",
        evidence_name="failed Formal batch manifest",
    )
    if (
        batch_manifest.get("schema_version") != "stage05.2-batch-manifest-v1"
        or batch_manifest.get("run_label") != run_label
        or batch_manifest.get("batch_id") != batch_id
        or batch_manifest.get("status") != "failed"
        or batch_manifest.get("failure_reason") != failure_reason
    ):
        raise RuntimeError("failed Formal batch manifest binding is invalid")

    artifact_manifest = _load_signed_json_object(
        batch_dir / "control" / f"{run_label}_manifest.json",
        evidence_name="failed Formal artifact manifest",
    )
    artifacts = artifact_manifest.get("artifacts")
    if (
        artifact_manifest.get("schema_version") != ARTIFACT_STORAGE_V2
        or artifact_manifest.get("run_label") != run_label
        or artifact_manifest.get("component") != "benchmark"
        or artifact_manifest.get("status") != "partial"
        or artifact_manifest.get("evidence_completeness") != "partial"
        or not isinstance(artifacts, list)
    ):
        raise RuntimeError("failed Formal artifact manifest is invalid")

    def bound_artifact(
        artifact_type: str,
        *,
        expected_relative_path: str,
    ) -> tuple[Path, dict[str, object]]:
        matches = [
            item
            for item in artifacts
            if isinstance(item, dict) and item.get("artifact_type") == artifact_type
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"failed Formal {artifact_type} resource binding is invalid"
            )
        record = matches[0]
        relative_path = record.get("relative_path")
        checksum = record.get("checksum")
        if (
            not isinstance(relative_path, str)
            or relative_path != expected_relative_path
            or not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise RuntimeError(
                f"failed Formal {artifact_type} resource binding is invalid"
            )
        path = (batch_dir / relative_path).resolve(strict=True)
        if not path.is_relative_to(batch_dir) or not path.is_file() or _sha256(path) != checksum:
            raise RuntimeError(
                f"failed Formal {artifact_type} resource binding is invalid"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"failed Formal {artifact_type} resource is unreadable"
            ) from error
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"failed Formal {artifact_type} resource is invalid"
            )
        return path, payload

    _, run_metadata = bound_artifact(
        "manifest_metadata",
        expected_relative_path=f"control/{run_label}_run_metadata.json",
    )
    resource_path, resource = bound_artifact(
        "resource_summary",
        expected_relative_path=f"control/{run_label}_resource_summary.json",
    )

    metadata_contract = run_metadata.get("producer_resource_contract")
    if (
        run_metadata.get("run_label") != run_label
        or run_metadata.get("batch_id") != batch_id
        or run_metadata.get("component") != "benchmark"
        or not isinstance(metadata_contract, dict)
        or metadata_contract.get("selected_workers") != selected_workers
        or metadata_contract.get("row_group_size")
        != campaign_contract.get("row_group_size")
        or metadata_contract.get("queue_depth") != campaign_contract.get("queue_depth")
    ):
        raise RuntimeError("failed Formal run metadata binding is invalid")
    process_peaks = resource.get("process_peak_rss_bytes")
    descendant_pids = resource.get("descendant_pids")
    parent_pid = resource.get("parent_pid")
    sample_count = resource.get("sample_count")
    process_peak_by_pid: dict[int, int] = {}
    if isinstance(process_peaks, dict):
        for raw_pid, raw_rss in process_peaks.items():
            if (
                not isinstance(raw_pid, str)
                or not raw_pid.isdigit()
                or isinstance(raw_rss, bool)
                or not isinstance(raw_rss, int)
                or raw_rss <= 0
            ):
                process_peak_by_pid = {}
                break
            process_peak_by_pid[int(raw_pid)] = raw_rss
    validated_descendant_pids: tuple[int, ...] | None = None
    if isinstance(descendant_pids, list):
        candidate_descendant_pids: list[int] = []
        for pid in descendant_pids:
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                candidate_descendant_pids = []
                break
            candidate_descendant_pids.append(pid)
        if (
            len(candidate_descendant_pids) >= selected_workers
            and len(set(candidate_descendant_pids))
            == len(candidate_descendant_pids)
        ):
            validated_descendant_pids = tuple(candidate_descendant_pids)
    resource_schema = resource.get("schema_version")
    if (
        resource_schema
        not in {"stage05.2-run-resource-v3", "stage05.2-run-resource-v4"}
        or resource.get("run_label") != run_label
        or resource.get("component") != "benchmark"
        or resource.get("configured_worker_count") != selected_workers
        or resource.get("measurement_scope")
        != "task_scheduling_through_parent_control_preparation"
        or resource.get("status") != "complete"
        or isinstance(parent_pid, bool)
        or not isinstance(parent_pid, int)
        or parent_pid <= 0
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or validated_descendant_pids is None
    ):
        raise RuntimeError("failed Formal resource summary is invalid")
    if (
        isinstance(failure_reason, str)
        and "cgroup v2 memory hard limit exceeded:" in failure_reason
        and resource_schema != "stage05.2-run-resource-v4"
    ):
        raise RuntimeError(
            "failed Formal cgroup v2 failure requires a v4 resource summary"
        )
    if (
        parent_pid in validated_descendant_pids
        or set(process_peak_by_pid) != set(validated_descendant_pids) | {parent_pid}
    ):
        raise RuntimeError("failed Formal resource summary PID binding is invalid")
    if resource_schema == "stage05.2-run-resource-v4" and (
        resource.get("aggregate_memory_source") != "cgroup_v2"
        or not is_stage052_dedicated_cgroup_path(
            str(resource.get("cgroup_path", ""))
        )
        or resource.get("cgroup_swap_peak_bytes") != 0
    ):
        raise RuntimeError("failed Formal cgroup v2 resource binding is invalid")
    if resource_schema == "stage05.2-run-resource-v4":
        failure_summary_path = campaign_dir / "failure_summary.json"
        failure_summary = _load_signed_json_object(
            failure_summary_path,
            evidence_name="failed Formal failure summary",
        )
        error_type = failure_summary.get("error_type")
        error_message = failure_summary.get("error_message")
        if (
            failure_summary.get("schema_version")
            != "experiment-cli-failure-summary-v1"
            or failure_summary.get("run_label") != run_label
            or failure_summary.get("status") != "failed"
            or failure_summary.get("failure_code") != "runner_failure"
            or error_type != "RuntimeError"
            or not isinstance(error_message, str)
            or failure_reason != f"{error_type}: {error_message}"
        ):
            raise RuntimeError("failed Formal failure summary binding is invalid")

        lifecycle_manifest_path = (
            campaign_dir
            / "control"
            / f"{run_label}_failure_lifecycle_manifest.json"
        )
        lifecycle_manifest = _load_signed_json_object(
            lifecycle_manifest_path,
            evidence_name="failed Formal lifecycle manifest",
        )
        lifecycle_artifacts = lifecycle_manifest.get("artifacts")
        if (
            lifecycle_manifest.get("schema_version")
            != "experiment-cli-failure-manifest-v1"
            or lifecycle_manifest.get("run_label") != run_label
            or lifecycle_manifest.get("status") != "failed"
            or lifecycle_manifest.get("evidence_completeness") != "partial"
            or lifecycle_manifest.get("artifact_trust")
            != "untrusted_failure_capsule"
            or not isinstance(lifecycle_artifacts, list)
        ):
            raise RuntimeError("failed Formal lifecycle manifest is invalid")

        def require_lifecycle_artifact(relative_path: str) -> None:
            matches = [
                item
                for item in lifecycle_artifacts
                if isinstance(item, dict)
                and item.get("relative_path") == relative_path
            ]
            path = (campaign_dir / relative_path).resolve(strict=True)
            stat = path.stat()
            if (
                len(matches) != 1
                or not path.is_relative_to(campaign_dir)
                or matches[0].get("checksum") != _sha256(path)
                or matches[0].get("byte_size") != stat.st_size
                or matches[0].get("modified_time_ns") != stat.st_mtime_ns
            ):
                raise RuntimeError(
                    "failed Formal lifecycle artifact binding is invalid: "
                    f"{relative_path}"
                )

        for relative_path in (
            "failure_summary.json",
            "failure_summary.sha256",
            "campaign_manifest.json",
            "campaign_manifest.sha256",
            f"{batch_id}/control/{run_label}_resource_summary.json",
        ):
            require_lifecycle_artifact(relative_path)
    aggregate_peak_rss = _integer(
        resource.get("aggregate_peak_rss_bytes"),
        "failed Formal aggregate RSS peak",
    )
    aggregate_peak_memory = (
        _integer(
            resource.get("aggregate_peak_memory_bytes"),
            "failed Formal cgroup memory peak",
        )
        if resource_schema == "stage05.2-run-resource-v4"
        else None
    )
    per_worker_peak = max(
        process_peak_by_pid[pid] for pid in validated_descendant_pids
    )
    row_group_size = _integer(
        metadata_contract.get("row_group_size"),
        "failed Formal row group size",
    )
    queue_depth = _integer(
        metadata_contract.get("queue_depth"),
        "failed Formal queue depth",
    )
    return FormalCampaignMemoryFloor(
        workers=selected_workers,
        aggregate_peak_rss_bytes=aggregate_peak_rss,
        per_worker_peak_rss_bytes=per_worker_peak,
        row_group_size=row_group_size,
        queue_depth=queue_depth,
        run_label=run_label,
        batch_id=batch_id,
        resource_summary_sha256=_sha256(resource_path),
        aggregate_memory_source=(
            "cgroup_v2"
            if resource_schema == "stage05.2-run-resource-v4"
            else "process_tree_rss"
        ),
        aggregate_peak_memory_bytes=aggregate_peak_memory,
        cgroup_path=(
            str(resource["cgroup_path"])
            if resource_schema == "stage05.2-run-resource-v4"
            else None
        ),
        cgroup_swap_peak_bytes=(
            _integer(
                resource.get("cgroup_swap_peak_bytes"),
                "failed Formal cgroup swap peak",
            )
            if resource_schema == "stage05.2-run-resource-v4"
            else None
        ),
    )


def load_attempt73_memory_floor(corpus_dir: Path) -> ProducerMemoryFloor:
    """Load the signed large-shard RSS floor without importing campaign geometry."""

    aggregate_peaks: list[int] = []
    per_worker_peaks: list[int] = []
    source_sha256_by_batch: dict[str, str] = {}
    for batch_id in ("batch0001", "batch0002"):
        batch_dir = corpus_dir / batch_id
        batch_manifest_path = batch_dir / "batch_manifest.json"
        if not signed_sidecar_matches(
            batch_manifest_path,
            batch_manifest_path.with_suffix(".sha256"),
        ):
            raise RuntimeError(f"Attempt73 {batch_id} manifest seal is invalid")
        try:
            batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Attempt73 {batch_id} manifest is unreadable") from error
        resource_path = (
            batch_dir
            / "control"
            / "stage05.2_benchmark_attempt73_resource_summary.json"
        )
        if (
            not isinstance(batch_manifest, dict)
            or batch_manifest.get("run_label") != "stage05.2_benchmark_attempt73"
            or batch_manifest.get("batch_id") != batch_id
            or batch_manifest.get("status") != "archived"
            or batch_manifest.get("resource_summary_sha256") != _sha256(resource_path)
        ):
            raise RuntimeError(f"Attempt73 {batch_id} resource binding is invalid")
        try:
            resource = json.loads(resource_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Attempt73 {batch_id} resource summary is unreadable") from error
        process_peaks = resource.get("process_peak_rss_bytes")
        if (
            not isinstance(resource, dict)
            or resource.get("configured_worker_count") != 4
            or resource.get("status") != "complete"
            or not isinstance(process_peaks, dict)
            or not process_peaks
        ):
            raise RuntimeError(f"Attempt73 {batch_id} resource summary is invalid")
        aggregate_peaks.append(
            _integer(resource.get("aggregate_peak_rss_bytes"), "aggregate peak RSS")
        )
        per_worker_peaks.append(
            max(_integer(value, "process peak RSS") for value in process_peaks.values())
        )
        source_sha256_by_batch[batch_id] = _sha256(resource_path)
    return ProducerMemoryFloor(
        four_worker_aggregate_peak_rss_bytes=max(aggregate_peaks),
        per_worker_peak_rss_bytes=max(per_worker_peaks),
        source_sha256_by_batch=source_sha256_by_batch,
    )


def apply_memory_floor(
    measurements: Sequence[MeasuredProducerCandidate],
    floor: ProducerMemoryFloor,
) -> tuple[MeasuredProducerCandidate, ...]:
    """Apply a conservative linear 4→5/6 projection from sealed long shards."""

    return tuple(
        MeasuredProducerCandidate(
            benchmark=replace(
                measurement.benchmark,
                aggregate_peak_rss_bytes=max(
                    measurement.benchmark.aggregate_peak_rss_bytes,
                    floor.projected_aggregate_peak_rss_bytes(
                        measurement.benchmark.workers
                    ),
                ),
            ),
            per_worker_peak_rss_bytes=max(
                measurement.per_worker_peak_rss_bytes,
                floor.per_worker_peak_rss_bytes,
            ),
        )
        for measurement in measurements
    )


class _PeakRssMonitor:
    def __init__(self, *, interval_seconds: float = 0.02) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._peak = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        return self._peak

    def _run(self) -> None:
        process = psutil.Process()
        while not self._stop.wait(self._interval_seconds):
            self._peak = max(self._peak, int(process.memory_info().rss))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_state(repository: Path) -> dict[str, object]:
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_paths = tuple(
            line
            for line in subprocess.run(
                ("git", "status", "--porcelain"),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            if line
        )
    except (OSError, subprocess.CalledProcessError):
        return {"repository_revision": None, "repository_dirty": None}
    return {
        "repository_revision": revision,
        "repository_dirty": bool(dirty_paths),
        "dirty_path_count": len(dirty_paths),
    }


def _calibration_provenance(
    *,
    repository: Path,
    config_path: Path,
    corpus_dir: Path,
) -> dict[str, object]:
    from evrptw import _core

    native_path = Path(str(_core.__file__)).resolve()
    corpus_manifests = tuple(
        path
        for path in (
            corpus_dir / "campaign_manifest.json",
            corpus_dir / "batch0001" / "batch_manifest.json",
            corpus_dir / "batch0002" / "batch_manifest.json",
        )
        if path.is_file()
    )
    return {
        **_source_state(repository),
        "python_abi": sys.implementation.cache_tag,
        "native_extension_sha256": _sha256(native_path),
        "configuration_sha256": _sha256(config_path) if config_path.is_file() else None,
        "corpus_manifest_sha256": {
            path.relative_to(corpus_dir).as_posix(): _sha256(path)
            for path in corpus_manifests
        },
        "producer_scope": {
            "instances": list(_CALIBRATION_INSTANCES),
            "seeds": list(_CALIBRATION_SEEDS),
            "termination": "fixed_work_100_exact_calls",
        },
    }


def _calibration_shard_runner(task: _ShardTask) -> list[dict[str, object]]:
    """Run one deterministic fixed-work shard in a fresh producer child."""

    config = load_stage052_config(task.config_path)
    stage04 = load_stage04_config(task.root / config.stage04_config)
    stage02 = load_stage02_config(task.root / config.stage02_config)
    instance = parse_schneider(
        task.root / config.benchmark_dir / f"{task.instance_name}.txt"
    )
    instance = replace(
        instance,
        distance_backend=_optimization_profile(Stage052Component.BENCHMARK),
    )
    axes = (
        Stage052Axis(
            name="fixed_work_calibration",
            termination_mode="fixed_work",
            time_limit_seconds=120.0,
            exact_call_budget=100,
            instrumentation_enabled=True,
            max_iterations=1_000,
        ),
    )
    return _run_and_persist_v2_shard(
        task,
        writer=None,
        config=config,
        stage04=stage04,
        stage02=stage02,
        instance=instance,
        axes=axes,
        storage=config.v2_storage,
    )


def _formal_memory_shard_runner(task: _ShardTask) -> list[dict[str, object]]:
    """Run one complete 30/60/300-second Formal shard for memory calibration."""

    config = load_stage052_config(task.config_path)
    stage04 = load_stage04_config(task.root / config.stage04_config)
    stage02 = load_stage02_config(task.root / config.stage02_config)
    instance = parse_schneider(
        task.root / config.benchmark_dir / f"{task.instance_name}.txt"
    )
    instance = replace(
        instance,
        distance_backend=_optimization_profile(Stage052Component.BENCHMARK),
    )
    return _run_and_persist_v2_shard(
        task,
        writer=None,
        config=config,
        stage04=stage04,
        stage02=stage02,
        instance=instance,
        axes=axes_for_scope("formal", customer_count=task.customer_count),
        storage=task.storage,
    )


def _summarize_producer_measurement(
    *,
    workers: int,
    rows: Sequence[dict[str, object]],
    elapsed: float,
    resource: RunResourceSummary,
    swap_peak_bytes: int,
) -> MeasuredProducerCandidate:
    rows = sorted(
        rows,
        key=lambda row: (
            str(row["instance"]),
            _integer(row["seed"], "seed"),
            str(row["axis"]),
        ),
    )
    aggregate_digest = hashlib.sha256(
        json.dumps(
            [
                {
                    "instance": row["instance"],
                    "seed": row["seed"],
                    "axis": row["axis"],
                    "semantic_digest": row["semantic_digest"],
                    "objective": [
                        row["vehicle_count"],
                        row["total_distance"],
                        row["total_charging_time"],
                        row["charging_count"],
                    ],
                    "exact_started_calls": row["exact_started_calls"],
                }
                for row in rows
            ],
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    total_exact_calls = sum(
        _integer(row["exact_started_calls"], "exact_started_calls") for row in rows
    )
    descendant_pids = set(resource.descendant_pids)
    per_worker_peak = max(
        (
            int(rss)
            for pid, rss in resource.process_peak_rss_bytes
            if pid in descendant_pids
        ),
        default=0,
    )
    fallback_count = sum(
        _integer(row["native_fallbacks"], "native_fallbacks")
        + _integer(row["native_protocol_fallbacks"], "native_protocol_fallbacks")
        for row in rows
    )
    return MeasuredProducerCandidate(
        benchmark=ProducerBenchmark(
            workers=workers,
            throughput=total_exact_calls / elapsed,
            aggregate_peak_rss_bytes=resource.aggregate_peak_memory_bytes,
            semantic_digest=aggregate_digest,
            swap_peak_bytes=swap_peak_bytes,
            fallback_count=fallback_count,
            resource_limit_exceeded=False,
        ),
        per_worker_peak_rss_bytes=per_worker_peak,
        aggregate_memory_source=resource.aggregate_memory_source,
        cgroup_path=resource.cgroup_path,
        cgroup_swap_peak_bytes=resource.cgroup_swap_peak_bytes,
    )


def benchmark_producer_candidate(
    *,
    workers: int,
    root: Path,
    config_path: Path,
    output_root: Path,
    run_label: str = "stage05.2_resource_calibration_attempt01",
) -> MeasuredProducerCandidate:
    """Execute the fixed six-shard high-memory scope with one worker candidate."""

    if workers not in _PRODUCER_PROBE_WORKERS:
        raise ValueError("producer calibration workers must be 4, 5, 6, or 8")
    config = load_stage052_config(config_path)
    candidate_dir = output_root / f"workers{workers}"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    tasks = _build_tasks(
        root=root,
        config_path=config_path,
        run_dir=candidate_dir,
        run_label=run_label,
        component=Stage052Component.BENCHMARK,
        scope="calibration",
        instances=_CALIBRATION_INSTANCES,
        seeds=_CALIBRATION_SEEDS,
        worker_count=workers,
        storage=config.v2_storage,
    )
    sampler = ProcessTreeResourceSampler(
        run_label=run_label,
        component="resource_calibration",
        configured_worker_count=workers,
        interval_seconds=0.02,
    )
    swap_baseline = int(psutil.swap_memory().used)
    swap_peak = swap_baseline
    swap_stop = threading.Event()

    def sample_swap() -> None:
        nonlocal swap_peak
        while not swap_stop.wait(0.02):
            swap_peak = max(swap_peak, int(psutil.swap_memory().used))

    swap_thread = threading.Thread(target=sample_swap, daemon=True)
    sampler.start()
    swap_thread.start()
    started = time.perf_counter()
    try:
        rows = _run_v2_tasks(
            tasks,
            worker_count=workers,
            _task_runner=_calibration_shard_runner,
        )
    finally:
        elapsed = time.perf_counter() - started
        swap_stop.set()
        swap_thread.join()
        resource = sampler.stop()
    rows.sort(key=lambda row: (str(row["instance"]), _integer(row["seed"], "seed")))
    if len(rows) != len(tasks) or any(row.get("failure_status") for row in rows):
        raise RuntimeError("producer calibration did not complete its exact fixed scope")
    aggregate_digest = hashlib.sha256(
        json.dumps(
            [
                {
                    "instance": row["instance"],
                    "seed": row["seed"],
                    "semantic_digest": row["semantic_digest"],
                    "objective": [
                        row["vehicle_count"],
                        row["total_distance"],
                        row["total_charging_time"],
                        row["charging_count"],
                    ],
                    "exact_started_calls": row["exact_started_calls"],
                }
                for row in rows
            ],
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    total_exact_calls = sum(
        _integer(row["exact_started_calls"], "exact_started_calls") for row in rows
    )
    per_worker_peak = max(
        (
            int(rss)
            for pid, rss in resource.process_peak_rss_bytes
            if pid in set(resource.descendant_pids)
        ),
        default=0,
    )
    fallback_count = sum(
        _integer(row["native_fallbacks"], "native_fallbacks")
        + _integer(row["native_protocol_fallbacks"], "native_protocol_fallbacks")
        for row in rows
    )
    return MeasuredProducerCandidate(
        benchmark=ProducerBenchmark(
            workers=workers,
            throughput=total_exact_calls / elapsed,
            aggregate_peak_rss_bytes=resource.aggregate_peak_rss_bytes,
            semantic_digest=aggregate_digest,
            swap_peak_bytes=max(0, swap_peak - swap_baseline),
            fallback_count=fallback_count,
            resource_limit_exceeded=False,
        ),
        per_worker_peak_rss_bytes=per_worker_peak,
    )


def benchmark_formal_memory_candidate(
    *,
    workers: int,
    row_group_size: int,
    queue_depth: int,
    root: Path,
    config_path: Path,
    output_root: Path,
    run_label: str,
    aggregate_memory_limit_bytes: int,
) -> MeasuredProducerCandidate:
    """Measure complete Formal axes on the failed high-memory R205 wave."""

    if workers not in _PRODUCER_PROBE_WORKERS:
        raise ValueError("Formal memory calibration workers must be 4, 5, 6, or 8")
    if row_group_size not in {16_384, 65_536, 262_144}:
        raise ValueError("Formal memory calibration row group size is invalid")
    if queue_depth not in {1, 2}:
        raise ValueError("Formal memory calibration queue depth is invalid")
    if aggregate_memory_limit_bytes <= 0:
        raise ValueError("Formal memory calibration aggregate limit must be positive")
    config = load_stage052_config(config_path)
    storage = replace(
        config.v2_storage,
        parquet_row_group_size=row_group_size,
        parquet_queue_depth=queue_depth,
    )
    candidate_dir = output_root / (
        f"formal-memory-workers{workers}-rg{row_group_size}-qd{queue_depth}"
    )
    candidate_dir.mkdir(parents=True, exist_ok=False)
    aggregate_memory_source = CgroupV2MemorySource.discover()
    (
        reset_memory_current,
        reset_memory_peak,
        reset_swap_current,
        reset_swap_peak,
    ) = aggregate_memory_source.reset_peaks()
    atomic_write_signed_json(
        output_root / "formal_memory_cgroup_peak_reset.json",
        {
            "schema_version": "stage05.2-cgroup-peak-reset-v1",
            "run_label": run_label,
            "component": "formal_memory_calibration",
            "cgroup_path": aggregate_memory_source.relative_path,
            "memory_current_bytes_after_reset": reset_memory_current,
            "memory_peak_bytes_after_reset": reset_memory_peak,
            "swap_current_bytes_after_reset": reset_swap_current,
            "swap_peak_bytes_after_reset": reset_swap_peak,
            "status": "verified",
        },
    )
    tasks = _build_tasks(
        root=root,
        config_path=config_path,
        run_dir=candidate_dir,
        run_label=run_label,
        component=Stage052Component.BENCHMARK,
        scope="formal_memory_calibration",
        instances=(_FORMAL_MEMORY_INSTANCE,),
        seeds=_FORMAL_MEMORY_SEEDS[:workers],
        worker_count=workers,
        storage=storage,
    )
    sampler = ProcessTreeResourceSampler(
        run_label=run_label,
        component="formal_memory_calibration",
        configured_worker_count=workers,
        interval_seconds=0.02,
        aggregate_memory_limit_bytes=aggregate_memory_limit_bytes,
        aggregate_memory_source=aggregate_memory_source,
    )
    swap_baseline = int(psutil.swap_memory().used)
    swap_peak = swap_baseline
    swap_stop = threading.Event()

    def sample_swap() -> None:
        nonlocal swap_peak
        while not swap_stop.wait(0.02):
            swap_peak = max(swap_peak, int(psutil.swap_memory().used))

    swap_thread = threading.Thread(target=sample_swap, daemon=True)
    sampler.start()
    swap_thread.start()
    started = time.perf_counter()
    measurement_error: BaseException | None = None
    rows: list[dict[str, object]] = []
    try:
        rows = _run_v2_tasks(
            tasks,
            worker_count=workers,
            abort_reason=sampler.abort_reason,
            _task_runner=_formal_memory_shard_runner,
        )
    except BaseException as error:
        measurement_error = error
    finally:
        elapsed = time.perf_counter() - started
        swap_stop.set()
        swap_thread.join()
        resource = sampler.stop()
    if measurement_error is not None:
        raise FormalMemoryMeasurementError(
            measurement_error,
            resource,
        ) from measurement_error
    if len(rows) != len(tasks) * 3 or any(row.get("failure_status") for row in rows):
        raise RuntimeError("Formal memory calibration did not complete its exact scope")
    return _summarize_producer_measurement(
        workers=workers,
        rows=rows,
        elapsed=elapsed,
        resource=resource,
        swap_peak_bytes=max(0, swap_peak - swap_baseline),
    )


def _write_parquet_copy(
    sources: Sequence[Path],
    destination_root: Path,
    *,
    row_group_size: int,
    queue_depth: int,
) -> None:
    storage = ArtifactStorageConfig(
        storage_policy_version=ARTIFACT_STORAGE_V2,
        screening_schema_version=SCREENING_DECISIONS_V3,
        compression_level=1,
        parquet_row_group_size=row_group_size,
        parquet_queue_depth=queue_depth,
    )
    for index, source in enumerate(sources):
        parquet = pq.ParquetFile(source)
        destination = destination_root / f"copy{index:02d}.parquet"
        sink = _StreamingParquetSink(destination, parquet.schema_arrow, storage)
        expected_rows = 0
        for batch in parquet.iter_batches(batch_size=65_536):
            sink.append_batch(batch)
            expected_rows += batch.num_rows
        observed_rows, schema_fingerprint = sink.close()
        if observed_rows != expected_rows or not schema_fingerprint:
            raise RuntimeError("Parquet calibration copy failed semantic row/schema replay")


def benchmark_parquet_configuration(
    *,
    corpus_dir: Path,
    row_group_size: int,
    queue_depth: int,
) -> ParquetBenchmark:
    """Replay the same two sealed large-shard event files through Zstandard 1."""

    sources = tuple(
        sorted(corpus_dir.glob("batch000[12]/c101_21/201[45]/*_events_*.parquet"))
    )
    if len(sources) != 2:
        raise RuntimeError("Attempt73 calibration corpus lacks two sealed c101_21 shards")
    semantic_digest = hashlib.sha256(
        json.dumps(
            [(str(path.relative_to(corpus_dir)), _sha256(path)) for path in sources],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with TemporaryDirectory(prefix="stage052-parquet-calibration-") as temporary:
        destination = Path(temporary)
        monitor = _PeakRssMonitor()
        monitor.start()
        started = time.perf_counter()
        _write_parquet_copy(
            sources,
            destination,
            row_group_size=row_group_size,
            queue_depth=queue_depth,
        )
        elapsed = time.perf_counter() - started
        peak_rss = monitor.stop()
    return ParquetBenchmark(
        row_group_size=row_group_size,
        queue_depth=queue_depth,
        persistence_seconds=elapsed,
        aggregate_peak_rss_bytes=peak_rss,
        semantic_digest=semantic_digest,
    )


def create_resource_contract(
    *,
    producer_measurements: Sequence[MeasuredProducerCandidate],
    parquet_measurements: Sequence[ParquetBenchmark],
    available_memory_bytes: int,
    locked_workers: int | None = None,
    locked_parquet_configuration: tuple[int, int] | None = None,
    python_allocator: str = "default",
) -> tuple[ProducerResourceContract, ProducerSelection, ParquetBenchmark]:
    """Apply all deterministic selection gates to measured observations."""

    producer_results = tuple(item.benchmark for item in producer_measurements)
    selection = select_producer_configuration(
        producer_results,
        available_memory_bytes=available_memory_bytes,
    )
    if locked_workers is not None:
        if locked_workers not in selection.candidate_workers:
            raise RuntimeError("locked producer worker count was not calibrated")
        rejected_reason = selection.rejected_reasons.get(locked_workers)
        if rejected_reason is not None:
            raise RuntimeError(
                f"locked {locked_workers}-worker producer is ineligible: "
                f"{rejected_reason}"
            )
        selection = ProducerSelection(
            selected_workers=locked_workers,
            semantic_digest=selection.semantic_digest,
            candidate_workers=selection.candidate_workers,
            rejected_reasons=selection.rejected_reasons,
        )
    selected_parquet = select_parquet_configuration(
        parquet_measurements,
        available_memory_bytes=available_memory_bytes,
        locked_configuration=locked_parquet_configuration,
    )
    selected_measurement = next(
        item
        for item in producer_measurements
        if item.benchmark.workers == selection.selected_workers
    )
    contract = derive_producer_resource_contract(
        results=producer_results,
        selection=selection,
        selected_per_worker_peak_rss_bytes=(
            selected_measurement.per_worker_peak_rss_bytes
        ),
        available_memory_bytes=available_memory_bytes,
        row_group_size=selected_parquet.row_group_size,
        queue_depth=selected_parquet.queue_depth,
        python_allocator=python_allocator,
    )
    return contract, selection, selected_parquet


def run_stage052_resource_calibration(
    *,
    corpus_dir: Path,
    output_root: Path,
    contract_path: Path,
    root: Path | None = None,
    config_path: Path | None = None,
    run_label: str = "stage05.2_resource_calibration_attempt01",
    require_clean_source: bool = False,
    memory_floor: ProducerMemoryFloor | None = None,
    producer_runner: Callable[..., MeasuredProducerCandidate] = benchmark_producer_candidate,
    parquet_runner: Callable[..., ParquetBenchmark] = benchmark_parquet_configuration,
    formal_memory_runner: Callable[
        ..., MeasuredProducerCandidate
    ] = benchmark_formal_memory_candidate,
    formal_campaign_memory_floor: FormalCampaignMemoryFloor | None = None,
    locked_workers: int = 6,
    locked_parquet_configuration: tuple[int, int] | None = None,
) -> ProducerResourceContract:
    """Run all candidates, select once, and atomically publish a signed contract."""

    repository = repository_root() if root is None else root.resolve()
    resolved_config = (
        repository / "configs/stage052_performance.toml"
        if config_path is None
        else config_path.resolve()
    )
    source_state = _source_state(repository)
    python_allocator = stage052_python_allocator()
    if require_clean_source and python_allocator != "malloc":
        raise RuntimeError(
            "resource calibration requires PYTHONMALLOC=malloc for bounded "
            "axis-internal reclamation"
        )
    if require_clean_source and (
        source_state.get("repository_revision") is None
        or source_state.get("repository_dirty") is not False
    ):
        raise RuntimeError("resource calibration requires a clean Git source revision")
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    memory = psutil.virtual_memory()
    memory_capacity = int(memory.total)
    observed_available_memory = int(memory.available)
    fresh_producer_measurements = tuple(
        producer_runner(
            workers=workers,
            root=repository,
            config_path=resolved_config,
            output_root=output_root,
            run_label=run_label,
        )
        for workers in _PRODUCER_WORKERS
    )
    memory_floor = (
        load_attempt73_memory_floor(corpus_dir)
        if memory_floor is None
        else memory_floor
    )
    producer_measurements = apply_memory_floor(
        fresh_producer_measurements,
        memory_floor,
    )
    parquet_measurements = tuple(
        parquet_runner(
            corpus_dir=corpus_dir,
            row_group_size=row_group_size,
            queue_depth=queue_depth,
        )
        for row_group_size, queue_depth in _PARQUET_CONFIGURATIONS
    )
    _, preliminary_selection, selected_parquet = create_resource_contract(
        producer_measurements=producer_measurements,
        parquet_measurements=parquet_measurements,
        available_memory_bytes=memory_capacity,
        locked_workers=locked_workers,
        locked_parquet_configuration=locked_parquet_configuration,
        python_allocator=python_allocator,
    )
    parent_memory_release_path = output_root / "formal_memory_parent_release.json"
    parent_memory_release = release_stage052_process_memory()
    atomic_write_signed_json(
        parent_memory_release_path,
        {
            "schema_version": "stage05.2-process-memory-release-v1",
            "run_label": run_label,
            "component": "formal_memory_calibration_parent",
            "memory_release": parent_memory_release,
            "status": "verified",
        },
    )
    parent_memory_release_sha256 = hashlib.sha256(
        parent_memory_release_path.read_bytes()
    ).hexdigest()
    formal_memory_measurement = formal_memory_runner(
        workers=preliminary_selection.selected_workers,
        row_group_size=selected_parquet.row_group_size,
        queue_depth=selected_parquet.queue_depth,
        root=repository,
        config_path=resolved_config,
        output_root=output_root,
        run_label=run_label,
        aggregate_memory_limit_bytes=memory_capacity,
    )
    formal_memory_measurement_path = output_root / "formal_memory_measurement.json"
    atomic_write_signed_json(
        formal_memory_measurement_path,
        {
            "schema_version": "stage05.2-formal-memory-measurement-v1",
            "run_label": run_label,
            "component": "formal_memory_calibration",
            "memory_capacity_bytes": memory_capacity,
            "measurement": asdict(formal_memory_measurement),
            "status": "measured_pending_contract_validation",
        },
    )
    formal_memory_measurement_sha256 = hashlib.sha256(
        formal_memory_measurement_path.read_bytes()
    ).hexdigest()
    formal_benchmark = formal_memory_measurement.benchmark
    if formal_benchmark.workers != preliminary_selection.selected_workers:
        raise RuntimeError("Formal memory measurement used the wrong producer worker count")
    if (
        formal_memory_measurement.aggregate_memory_source != "cgroup_v2"
        or formal_memory_measurement.cgroup_path is None
        or formal_memory_measurement.cgroup_swap_peak_bytes != 0
    ):
        raise RuntimeError(
            "Formal memory measurement must use a swap-free isolated cgroup v2 service"
        )
    # ``swap_peak_bytes`` is the host-wide psutil delta retained as telemetry.
    # Only the dedicated cgroup counter above can attribute swap to this Formal
    # workload; treating unrelated host activity as a hard gate would make the
    # signed resource contract non-reproducible.
    if formal_benchmark.fallback_count > 0 or formal_benchmark.resource_limit_exceeded:
        raise RuntimeError(
            "Formal memory measurement observed fallback or a resource limit"
        )
    if (
        formal_campaign_memory_floor is not None
        and formal_campaign_memory_floor.workers
        != preliminary_selection.selected_workers
    ):
        raise RuntimeError(
            "failed Formal memory floor worker count does not match "
            "the selected producer contract"
        )
    # The replacement aggregate limit is calibrated from this clean revision's
    # isolated cgroup measurement.  A predecessor process-RSS sum is not
    # commensurate with cgroup memory; a predecessor cgroup peak remains typed
    # and bound as failure provenance, but is not a replacement floor because
    # this calibration validates a source-level page-cache-control change.
    selected_aggregate_peak = formal_benchmark.aggregate_peak_rss_bytes
    selected_per_worker_peak = max(
        formal_memory_measurement.per_worker_peak_rss_bytes,
        (
            formal_campaign_memory_floor.per_worker_peak_rss_bytes
            if formal_campaign_memory_floor is not None
            else 0
        ),
    )
    producer_measurements = tuple(
        MeasuredProducerCandidate(
            benchmark=replace(
                measurement.benchmark,
                aggregate_peak_rss_bytes=selected_aggregate_peak,
            ),
            per_worker_peak_rss_bytes=max(
                measurement.per_worker_peak_rss_bytes,
                selected_per_worker_peak,
            ),
        )
        if measurement.benchmark.workers == preliminary_selection.selected_workers
        else measurement
        for measurement in producer_measurements
    )
    contract, selection, selected_parquet = create_resource_contract(
        producer_measurements=producer_measurements,
        parquet_measurements=parquet_measurements,
        available_memory_bytes=memory_capacity,
        locked_workers=locked_workers,
        locked_parquet_configuration=locked_parquet_configuration,
        python_allocator=python_allocator,
    )
    if selection.selected_workers != preliminary_selection.selected_workers:
        raise RuntimeError(
            "Formal memory measurement invalidated producer selection; "
            "rerun calibration with a viable worker set"
        )
    atomic_write_signed_json(contract_path, contract.to_dict())
    if not signed_sidecar_matches(contract_path, contract_path.with_suffix(".sha256")):
        raise RuntimeError("resource calibration contract sealing failed")
    report_schema = (
        "stage05.2-resource-calibration-report-v4"
        if formal_campaign_memory_floor is not None
        and formal_campaign_memory_floor.aggregate_memory_source == "cgroup_v2"
        else "stage05.2-resource-calibration-report-v3"
    )
    atomic_write_signed_json(
        output_root / "calibration_report.json",
        {
            "schema_version": report_schema,
            "corpus_role": "read_only_benchmark_differential_only",
            "corpus_path": str(corpus_dir),
            "campaign_geometry_contribution": 0,
            "run_label": run_label,
            "provenance": _calibration_provenance(
                repository=repository,
                config_path=resolved_config,
                corpus_dir=corpus_dir,
            ),
            "memory_capacity_bytes": memory_capacity,
            "observed_available_memory_bytes": observed_available_memory,
            "producer_measurements": [
                {
                    **asdict(item.benchmark),
                    "per_worker_peak_rss_bytes": item.per_worker_peak_rss_bytes,
                }
                for item in producer_measurements
            ],
            "fresh_producer_measurements": [
                {
                    **asdict(item.benchmark),
                    "per_worker_peak_rss_bytes": item.per_worker_peak_rss_bytes,
                }
                for item in fresh_producer_measurements
            ],
            "attempt73_memory_floor": asdict(memory_floor),
            "formal_memory_parent_release": parent_memory_release,
            "formal_memory_parent_release_sha256": parent_memory_release_sha256,
            "formal_memory_measurement": asdict(formal_memory_measurement),
            "formal_memory_measurement_sha256": formal_memory_measurement_sha256,
            "formal_campaign_memory_floor": (
                asdict(formal_campaign_memory_floor)
                if formal_campaign_memory_floor is not None
                else None
            ),
            "parquet_measurements": [asdict(item) for item in parquet_measurements],
            "selection": {
                "policy": "user_locked",
                "locked_workers": locked_workers,
                "selected_workers": selection.selected_workers,
                "parquet_policy": (
                    "user_locked"
                    if locked_parquet_configuration is not None
                    else "performance_selected"
                ),
                "rejected_reasons": dict(selection.rejected_reasons),
                "row_group_size": selected_parquet.row_group_size,
                "queue_depth": selected_parquet.queue_depth,
            },
            "contract": contract.to_dict(),
        },
    )
    return contract


def run_stage052_formal_memory_probe(
    *,
    workers: int,
    row_group_size: int,
    queue_depth: int,
    output_root: Path,
    root: Path | None = None,
    config_path: Path | None = None,
    run_label: str,
    runner: Callable[
        ..., MeasuredProducerCandidate
    ] = benchmark_formal_memory_candidate,
) -> dict[str, object]:
    """Seal one non-campaign complete-Formal producer memory observation."""

    repository = repository_root() if root is None else root.resolve()
    resolved_config = (
        repository / "configs/stage052_performance.toml"
        if config_path is None
        else config_path.resolve()
    )
    if output_root.exists():
        raise FileExistsError(output_root)
    formal_memory_instance_relative_path = (
        Path("data") / "schneider" / f"{_FORMAL_MEMORY_INSTANCE}.txt"
    )
    formal_memory_instance_path = repository / formal_memory_instance_relative_path
    path_component = repository
    for component in formal_memory_instance_relative_path.parts:
        path_component /= component
        if path_component.is_symlink():
            raise RuntimeError(
                "Formal memory probe exact input must be a snapshot-local "
                f"ordinary file: {formal_memory_instance_path}"
            )
    if not formal_memory_instance_path.is_file():
        raise FileNotFoundError(
            "Formal memory probe exact input is unavailable: "
            f"{formal_memory_instance_path}"
        )
    if not formal_memory_instance_path.resolve(strict=True).is_relative_to(repository):
        raise RuntimeError(
            "Formal memory probe exact input resolves outside the source snapshot: "
            f"{formal_memory_instance_path}"
        )
    if formal_memory_instance_path.stat().st_size <= 0:
        raise RuntimeError(
            "Formal memory probe exact input is empty: "
            f"{formal_memory_instance_path}"
        )
    output_root.mkdir(parents=True)
    memory_capacity = int(psutil.virtual_memory().total)
    operating_reserve_bytes = 1024**3
    if memory_capacity <= operating_reserve_bytes:
        raise RuntimeError("Formal memory probe host capacity is too small")
    aggregate_memory_limit_bytes = memory_capacity - operating_reserve_bytes
    base_payload: dict[str, object] = {
        "schema_version": "stage05.2-formal-memory-probe-v3",
        "run_label": run_label,
        "corpus_role": "exploratory_only_zero_campaign_geometry",
        "campaign_geometry_contribution": 0,
        "workers": workers,
        "row_group_size": row_group_size,
        "queue_depth": queue_depth,
        "memory_capacity_bytes": memory_capacity,
        "operating_reserve_bytes": operating_reserve_bytes,
        "aggregate_memory_source": "cgroup_v2",
        "aggregate_memory_abort_limit_bytes": aggregate_memory_limit_bytes,
        "formal_memory_scope": {
            "instance": _FORMAL_MEMORY_INSTANCE,
            "seeds": list(_FORMAL_MEMORY_SEEDS[:workers]),
            "axes": ["wall_clock_30", "wall_clock_60", "wall_clock_300"],
        },
        "provenance": {
            **_source_state(repository),
            "configuration_sha256": (
                _sha256(resolved_config) if resolved_config.is_file() else None
            ),
            "python_abi": sys.implementation.cache_tag,
        },
    }
    report_path = output_root / "formal_memory_probe_report.json"

    def seal(payload: dict[str, object]) -> None:
        atomic_write_signed_json(report_path, payload)
        if not signed_sidecar_matches(report_path, report_path.with_suffix(".sha256")):
            raise RuntimeError("Formal memory probe report sealing failed")

    try:
        measured = runner(
            workers=workers,
            row_group_size=row_group_size,
            queue_depth=queue_depth,
            root=repository,
            config_path=resolved_config,
            output_root=output_root,
            run_label=run_label,
            aggregate_memory_limit_bytes=aggregate_memory_limit_bytes,
        )
        if measured.benchmark.workers != workers:
            raise RuntimeError("Formal memory probe returned the wrong worker count")
    except BaseException as error:
        failed_payload = {
            **base_payload,
            "status": "failed",
            "failure_reason": f"{type(error).__name__}: {error}",
            "measurement": None,
            "resource_summary": getattr(error, "resource_summary", None),
        }
        try:
            seal(failed_payload)
        except BaseException as seal_error:
            raise RuntimeError(
                "Formal memory probe failed and its signed failure report "
                f"could not be sealed: {type(seal_error).__name__}: {seal_error}"
            ) from error
        raise
    payload = {
        **base_payload,
        "status": "complete",
        "failure_reason": None,
        "measurement": asdict(measured),
        "resource_summary": None,
    }
    seal(payload)
    return payload


def run_stage052_producer_probe(
    *,
    workers: int,
    corpus_dir: Path,
    output_root: Path,
    root: Path | None = None,
    config_path: Path | None = None,
    run_label: str,
) -> dict[str, object]:
    """Run one explicitly non-campaign producer concurrency probe."""

    repository = repository_root() if root is None else root.resolve()
    resolved_config = (
        repository / "configs/stage052_performance.toml"
        if config_path is None
        else config_path.resolve()
    )
    measured = benchmark_producer_candidate(
        workers=workers,
        root=repository,
        config_path=resolved_config,
        output_root=output_root,
        run_label=run_label,
    )
    floor = load_attempt73_memory_floor(corpus_dir)
    payload: dict[str, object] = {
        "schema_version": "stage05.2-producer-probe-v1",
        "corpus_role": "exploratory_only_zero_campaign_geometry",
        "campaign_geometry_contribution": 0,
        "workers": workers,
        "fresh_measurement": asdict(measured),
        "projected_long_shard_aggregate_peak_rss_bytes": (
            floor.projected_aggregate_peak_rss_bytes(workers)
        ),
        "long_shard_per_worker_peak_rss_bytes": floor.per_worker_peak_rss_bytes,
        "attempt73_resource_sha256_by_batch": floor.source_sha256_by_batch,
        "provenance": _calibration_provenance(
            repository=repository,
            config_path=resolved_config,
            corpus_dir=corpus_dir,
        ),
    }
    atomic_write_signed_json(output_root / "probe_report.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Stage 5.2 producer resources")
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--repository-root",
        type=Path,
        required=True,
        help="Explicit clean Git worktree used for source and input identity",
    )
    parser.add_argument("--contract-path", type=Path)
    parser.add_argument("--formal-memory-floor-dir", type=Path)
    parser.add_argument(
        "--formal-memory-floor-batch-id",
        default="batch0003",
    )
    parser.add_argument("--run-label", default="stage05.2_resource_calibration_attempt01")
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument(
        "--probe-workers",
        type=int,
        choices=_PRODUCER_PROBE_WORKERS,
    )
    parser.add_argument(
        "--formal-memory-probe-workers",
        type=int,
        choices=_PRODUCER_PROBE_WORKERS,
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        choices=(16_384, 65_536, 262_144),
    )
    parser.add_argument(
        "--queue-depth",
        type=int,
        choices=(1, 2),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage052_performance.toml"),
    )
    arguments = parser.parse_args()
    resolved_repository = arguments.repository_root.resolve()
    resolved_config = (
        arguments.config.resolve()
        if arguments.config.is_absolute()
        else (resolved_repository / arguments.config).resolve()
    )
    if (
        arguments.probe_workers is not None
        and arguments.formal_memory_probe_workers is not None
    ):
        parser.error(
            "--probe-workers and --formal-memory-probe-workers are mutually exclusive"
        )
    row_group_size = (
        65_536 if arguments.row_group_size is None else arguments.row_group_size
    )
    queue_depth = 2 if arguments.queue_depth is None else arguments.queue_depth

    def seal_output() -> None:
        seal_cli_attempt(
            config_path=resolved_config,
            output_dir=arguments.output_root.resolve(),
            run_label=arguments.run_label,
            manifest_path=None,
        )

    if arguments.formal_memory_probe_workers is not None:
        preflight_cli_attempt(
            config_path=resolved_config,
            output_dir=arguments.output_root.resolve(),
            run_label=arguments.run_label,
            workers=arguments.formal_memory_probe_workers,
            threads=arguments.formal_memory_probe_workers,
            processes=arguments.formal_memory_probe_workers,
            queue_depth=queue_depth,
            row_group_size=row_group_size,
        )
        try:
            payload = run_stage052_formal_memory_probe(
                workers=arguments.formal_memory_probe_workers,
                row_group_size=row_group_size,
                queue_depth=queue_depth,
                output_root=arguments.output_root.resolve(),
                root=resolved_repository,
                config_path=resolved_config,
                run_label=arguments.run_label,
            )
            seal_output()
        except BaseException as error:
            seal_failed_cli_attempt(
                config_path=resolved_config,
                output_dir=arguments.output_root.resolve(),
                run_label=arguments.run_label,
                error=error,
            )
            raise
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if arguments.probe_workers is not None:
        if arguments.corpus_dir is None:
            parser.error("--corpus-dir is required with --probe-workers")
        preflight_cli_attempt(
            config_path=resolved_config,
            output_dir=arguments.output_root.resolve(),
            run_label=arguments.run_label,
            workers=arguments.probe_workers,
            threads=arguments.probe_workers,
            processes=arguments.probe_workers,
            queue_depth=queue_depth,
            row_group_size=row_group_size,
        )
        try:
            payload = run_stage052_producer_probe(
                workers=arguments.probe_workers,
                corpus_dir=arguments.corpus_dir.resolve(),
                output_root=arguments.output_root.resolve(),
                root=resolved_repository,
                config_path=resolved_config,
                run_label=arguments.run_label,
            )
            seal_output()
        except BaseException as error:
            seal_failed_cli_attempt(
                config_path=resolved_config,
                output_dir=arguments.output_root.resolve(),
                run_label=arguments.run_label,
                error=error,
            )
            raise
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if arguments.corpus_dir is None:
        parser.error("--corpus-dir is required for resource calibration")
    if arguments.contract_path is None:
        parser.error(
            "--contract-path is required unless a producer probe mode is used"
        )
    if arguments.formal_memory_floor_dir is None:
        parser.error(
            "--formal-memory-floor-dir is required for resource calibration"
        )
    if (arguments.row_group_size is None) != (arguments.queue_depth is None):
        parser.error(
            "--row-group-size and --queue-depth must be supplied together "
            "when locking resource calibration"
        )
    locked_parquet_configuration = (
        None
        if arguments.row_group_size is None
        else (row_group_size, queue_depth)
    )
    formal_campaign_memory_floor = load_failed_formal_memory_floor(
        arguments.formal_memory_floor_dir.resolve(),
        batch_id=arguments.formal_memory_floor_batch_id,
    )
    preflight_cli_attempt(
        config_path=resolved_config,
        output_dir=arguments.output_root.resolve(),
        run_label=arguments.run_label,
        workers=6,
        threads=6,
        processes=6,
        queue_depth=queue_depth,
        row_group_size=row_group_size,
    )
    try:
        contract = run_stage052_resource_calibration(
            corpus_dir=arguments.corpus_dir.resolve(),
            output_root=arguments.output_root.resolve(),
            contract_path=arguments.contract_path.resolve(),
            root=resolved_repository,
            config_path=resolved_config,
            run_label=arguments.run_label,
            require_clean_source=not arguments.allow_dirty_source,
            formal_campaign_memory_floor=formal_campaign_memory_floor,
            locked_parquet_configuration=locked_parquet_configuration,
        )
        seal_output()
    except BaseException as error:
        seal_failed_cli_attempt(
            config_path=resolved_config,
            output_dir=arguments.output_root.resolve(),
            run_label=arguments.run_label,
            error=error,
        )
        raise
    print(json.dumps(contract.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
