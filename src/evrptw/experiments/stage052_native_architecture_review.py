"""Independent replay and reporting for Stage 5.2 native architectures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import struct
import subprocess
import sys
import time
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from typing import cast

import orjson

from evrptw.artifacts import ArtifactReader
from evrptw.cache_incremental import (
    RouteCacheKey,
    canonical_instance_hash,
    charging_result_semantic_digest,
    estimate_cache_entry_bytes,
)
from evrptw.charging import ChargingSubproblemResult, solve_exact_charging
from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES
from evrptw.experiments.stage052_native_architectures import (
    AXIS_NAMES,
    AXIS_PERSISTENCE_RECEIPT_SCHEMA_VERSION,
    CGROUP_IO_ACCOUNTING_SOURCE,
    LEGACY_PROFILE_COMPARISON_SCHEMA_VERSION,
    MODE_WAVE_RESOURCE_ACCOUNTING_SOURCE,
    MODES,
    PAIRED_INSTANCES,
    PREVIOUS_COMPARISON_SCHEMA_VERSION,
    PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION,
    PROCESS_PROFILE_COMPARISON_SCHEMA_VERSION,
    PROCESS_TREE_IO_ACCOUNTING_SOURCE,
    SCHEMA_VERSION,
    SEEDS,
    TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
    ArchitectureMode,
    _axis_persistence_receipt_path,
    expected_axis_count,
    run_labels_for_scope,
)
from evrptw.measurement import canonical_route_key
from evrptw.models import Instance, NodeType
from evrptw.neighborhoods import screen_route_candidate
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.repository import repository_root
from evrptw.runtime_envelope import (
    PROCESS_TREE_STATISTICS_FIELDS,
    TERMINAL_PROCESS_IO_RECEIPT_SCHEMA_VERSION,
)
from evrptw.stage052_performance import FrozenPerformanceProfile
from evrptw.stage052_physical_telemetry import (
    iter_verified_physical_telemetry,
)
from evrptw.stage052_replay import (
    replay_verified_shard,
    verified_artifact_shard_bundle,
)
from evrptw.stage052_semantic_journal import (
    SEMANTIC_STREAM_NAMES,
    SEMANTIC_TRAJECTORY_IMPLEMENTATION_STATUSES,
    iter_verified_native_control_events,
    iter_verified_semantic_journal,
    semantic_bundle_path,
)
from evrptw.validation import validate_routes
from tools.native_build_attestation import (
    committed_source_attestation,
    committed_wheel_project_entry_sha256,
    validate_scheduler_build_attestation,
)

REVIEW_SCHEMA_VERSION = "stage05.2-native-architecture-review-v13"
REVIEW_MANIFEST_SCHEMA_VERSION = "stage05.2-native-architecture-review-manifest-v2"
REVIEW_EXECUTION_SCHEMA_VERSION = "experiment-review-execution-v1"
REVIEWER_MODULE_NAME = "evrptw.experiments.stage052_native_architecture_review"
LEGACY_COMPARISON_SCHEMA_VERSION = "stage05.2-native-architecture-comparison-v3"
INLINE_SEMANTIC_COMPARISON_SCHEMA_VERSION = "stage05.2-native-architecture-comparison-v6"
EXTERNAL_SEMANTIC_COMPARISON_SCHEMA_VERSIONS = frozenset(
    {
        PREVIOUS_COMPARISON_SCHEMA_VERSION,
        LEGACY_PROFILE_COMPARISON_SCHEMA_VERSION,
        PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION,
        PROCESS_PROFILE_COMPARISON_SCHEMA_VERSION,
        TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }
)
PROFILE_COMPARISON_SCHEMA_VERSIONS = frozenset(
    {
        LEGACY_PROFILE_COMPARISON_SCHEMA_VERSION,
        PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION,
        PROCESS_PROFILE_COMPARISON_SCHEMA_VERSION,
        TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }
)
_RESOURCE_TELEMETRY_BASE_TOPOLOGY_FIELDS = frozenset(
    {
        "shard_processes",
        "threads_per_shard",
        "compute_thread_limit",
        "axis_compute_thread_limit",
        "scheduler_threads",
        "effective_native_search_threads",
        "performance_profile_sha256",
        "performance_topology_key",
        "configured_axis_cpu_ids",
        "configured_scheduler_cpu_ids",
        "scheduler_request_threads",
        "allow_affinity_overlap",
        "shared_native_work_pool",
        "process_id",
        "scheduler_process_id",
        "shared_scheduler_resource_attribution",
        "threads_before",
        "threads_after",
        "rss_bytes",
        "peak_rss_bytes",
        "cpu_affinity",
        "shared_scheduler_accounting",
    }
)
_RESOURCE_TELEMETRY_DISABLED_TOPOLOGY_FIELDS = (
    _RESOURCE_TELEMETRY_BASE_TOPOLOGY_FIELDS | {"telemetry_status"}
)
_CURRENT_WORKER_PSS_SAMPLE_ACCOUNTING_FIELDS = frozenset(
    {
        "worker_descendant_pss_complete_sample_count",
        "worker_descendant_pss_incomplete_sample_count",
    }
)
_CURRENT_CPU_QUANTIZATION_ACCOUNTING_FIELDS = frozenset(
    {"cpu_clock_tick_hz", "cpu_quantization_lane_count"}
)
_CURRENT_TERMINAL_PROCESS_IO_FIELDS = frozenset(
    {
        "terminal_process_io_receipts",
        "process_io_terminal_status",
        "process_io_uncovered_identities",
    }
)
_TICK_PROFILE_PROCESS_TREE_STATISTICS_FIELDS = (
    PROCESS_TREE_STATISTICS_FIELDS - _CURRENT_TERMINAL_PROCESS_IO_FIELDS
)
_PROCESS_PROFILE_PROCESS_TREE_STATISTICS_FIELDS = (
    _TICK_PROFILE_PROCESS_TREE_STATISTICS_FIELDS
    - _CURRENT_CPU_QUANTIZATION_ACCOUNTING_FIELDS
)
_PRIOR_PROFILE_PROCESS_TREE_STATISTICS_FIELDS = (
    _PROCESS_PROFILE_PROCESS_TREE_STATISTICS_FIELDS
    - _CURRENT_WORKER_PSS_SAMPLE_ACCOUNTING_FIELDS
)
FULL_NATIVE_SEMANTIC_MODES = frozenset(
    {
        ArchitectureMode.FULL_NATIVE_ALNS.value,
        ArchitectureMode.HOST_SCHEDULER.value,
    }
)
HISTORICAL_PILOT_ROOT = Path(
    "/mnt/e/Reproducible-EVRPTW-archive/stage05.2/runs/"
    "stage05.2_benchmark_attempt72/generation-0001/d_benchmark"
)
HISTORICAL_PILOT_RUN_LABEL = "stage05.2_benchmark_attempt72"
HISTORICAL_PILOT_REVISION = "a5cf00f7580fc2632179495a739a110786ace87d"
HISTORICAL_PILOT_RAW_MANIFEST_SHA256 = (
    "5aa8b773c39ea3e0a8cd31434756f4219c3ec51f430bdcae1e61107fdd3f38e4"
)
HISTORICAL_PILOT_REVIEW_MANIFEST_SHA256 = (
    "24de9cc93ad99f7d607617e11e8c9bee7fda224a421da1f97dd3af1bd6277727"
)


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    path: Path
    payload: Mapping[str, object]
    mode_wave: Mapping[str, object] | None = None

    @property
    def mode(self) -> ArchitectureMode:
        return ArchitectureMode(_string(self.payload, "mode"))

    @property
    def key(self) -> tuple[int, str, str, int]:
        return (
            _integer(self.payload, "repeat"),
            _string(self.payload, "axis"),
            _string(self.payload, "instance"),
            _integer(self.payload, "seed"),
        )


def _mode_wave_affinity_matches_profile(
    wave: Mapping[str, object],
    *,
    comparison_schema: str,
    allowed_cpu_ids: Sequence[int],
) -> bool:
    """Use process affinity as the v11 authority and retain older strict replay."""

    if wave.get("actual_affinity_union") != list(allowed_cpu_ids):
        return False
    return (
        comparison_schema
        in {
            PROCESS_PROFILE_COMPARISON_SCHEMA_VERSION,
            TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
            SCHEMA_VERSION,
        }
        or wave.get("thread_affinity_union") == list(allowed_cpu_ids)
    )


def _records_use_current_profile_schema(records: Iterable[ReviewRecord]) -> bool:
    """New qualification is available only for one all-current comparison schema."""

    return {_string(record.payload, "schema_version") for record in records} == {
        SCHEMA_VERSION
    }


def _resource_telemetry_topology_error(
    topology: Mapping[str, object],
    *,
    mode: ArchitectureMode,
    expected_resource_telemetry: bool,
    require_complete_schema: bool = True,
    comparison_schema: str = SCHEMA_VERSION,
) -> str | None:
    if expected_resource_telemetry:
        if "telemetry_status" in topology:
            return "enabled process-tree telemetry has a disabled-status marker"
        if comparison_schema == SCHEMA_VERSION:
            statistics_fields = PROCESS_TREE_STATISTICS_FIELDS
        elif comparison_schema == TICK_PROFILE_COMPARISON_SCHEMA_VERSION:
            statistics_fields = _TICK_PROFILE_PROCESS_TREE_STATISTICS_FIELDS
        elif comparison_schema == PROCESS_PROFILE_COMPARISON_SCHEMA_VERSION:
            statistics_fields = _PROCESS_PROFILE_PROCESS_TREE_STATISTICS_FIELDS
        else:
            statistics_fields = _PRIOR_PROFILE_PROCESS_TREE_STATISTICS_FIELDS
        if require_complete_schema and set(topology) != (
            _RESOURCE_TELEMETRY_BASE_TOPOLOGY_FIELDS
            | statistics_fields
        ):
            return "enabled process-tree telemetry schema is not exact"
        return None
    if not require_complete_schema:
        return "disabled process-tree telemetry is unavailable for a historical schema"
    if (
        mode is not ArchitectureMode.CURRENT_STAGE052
        or set(topology) != _RESOURCE_TELEMETRY_DISABLED_TOPOLOGY_FIELDS
        or topology.get("telemetry_status") != "disabled"
    ):
        return "disabled process-tree telemetry is not an exact authorized control"
    return None


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _sequence(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return value


def _persistence_breakdown_error(payload: Mapping[str, object]) -> str | None:
    breakdown = payload.get("persistence_breakdown")
    if not isinstance(breakdown, dict) or set(breakdown) != {
        "semantic_journal",
        "signed_json",
    }:
        return "persistence breakdown is missing or malformed"
    semantic = breakdown["semantic_journal"]
    signed = breakdown["signed_json"]
    semantic_fields = {
        "event_serialization_seconds",
        "compression_seconds",
        "write_seconds",
        "fsync_seconds",
        "hash_seconds",
        "atomic_publish_seconds",
        "pipeline_union_seconds",
        "runtime_spool_union_seconds",
        "publication_seconds",
        "formal_attributed_seconds",
        "total_seconds",
    }
    signed_fields = {
        "schema_version",
        "encoding_seconds",
        "hash_seconds",
        "write_seconds",
        "fsync_seconds",
        "atomic_publish_seconds",
        "total_seconds",
        "data_bytes",
        "sidecar_bytes",
    }
    signed_timing_fields = {
        "encoding_seconds",
        "hash_seconds",
        "write_seconds",
        "fsync_seconds",
        "atomic_publish_seconds",
        "total_seconds",
    }
    if not isinstance(semantic, dict) or set(semantic) != semantic_fields:
        return "semantic-journal persistence receipt is invalid"
    signed_schema = signed.get("schema_version") if isinstance(signed, dict) else None
    if signed_schema == "stage05.2-signed-json-persistence-v2":
        signed_fields |= {
            "write_chunk_bytes",
            "write_batch_count",
            "maximum_write_batch_bytes",
        }
    if (
        not isinstance(signed, dict)
        or set(signed) != signed_fields
        or signed_schema
        not in {
            "stage05.2-signed-json-persistence-v1",
            "stage05.2-signed-json-persistence-v2",
        }
    ):
        return "signed-JSON persistence receipt is invalid"
    if signed_schema == "stage05.2-signed-json-persistence-v2":
        for field_name in (
            "write_chunk_bytes",
            "write_batch_count",
            "maximum_write_batch_bytes",
        ):
            value = signed[field_name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                return f"signed-JSON chunk statistic is invalid: {field_name}"
        if (
            signed["write_chunk_bytes"] != 512 * 1024
            or signed["maximum_write_batch_bytes"] > signed["write_chunk_bytes"]
        ):
            return "signed-JSON chunk bounds are invalid"
    for receipt, timing_fields in (
        (semantic, semantic_fields),
        (signed, signed_timing_fields),
    ):
        for field_name in timing_fields:
            value = receipt[field_name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                return f"persistence timing is invalid: {field_name}"
    attributed = float(semantic["pipeline_union_seconds"]) + float(semantic["publication_seconds"])
    if not math.isclose(
        attributed,
        float(semantic["formal_attributed_seconds"]),
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        return "semantic-journal formal attribution does not reconcile"
    signed_component_total = sum(
        float(signed[field_name]) for field_name in signed_timing_fields - {"total_seconds"}
    )
    if signed_component_total > float(signed["total_seconds"]) + 1e-5:
        return "signed-JSON persistence timing components exceed total"
    for field_name in ("data_bytes", "sidecar_bytes"):
        value = signed[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return f"signed-JSON persistence bytes are invalid: {field_name}"
    persistence = payload.get("persistence_seconds")
    if (
        not isinstance(persistence, int | float)
        or isinstance(persistence, bool)
        or not math.isclose(
            float(persistence),
            float(semantic["formal_attributed_seconds"]) + float(signed["total_seconds"]),
            rel_tol=1e-9,
            abs_tol=1e-5,
        )
    ):
        return "persistence interval does not cover its timed components"
    return None


def _verify_signed_json(path: Path) -> Mapping[str, object]:
    data = path.read_bytes()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = sidecar.read_text(encoding="ascii").strip()
    observed = hashlib.sha256(data).hexdigest()
    if expected != observed:
        raise RuntimeError(f"SHA-256 mismatch: {path}")
    payload = orjson.loads(data)
    if not isinstance(payload, dict):
        raise RuntimeError(f"signed JSON root is not an object: {path}")
    return payload


def _axis_payload_with_persistence(
    path: Path,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Bind current axes to their separately signed, non-self-referential receipt."""

    result = dict(payload)
    if payload.get("schema_version") not in PROFILE_COMPARISON_SCHEMA_VERSIONS:
        return result
    descriptor = payload.get("persistence_receipt")
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "schema_version",
        "path",
        "sidecar_path",
    }:
        raise RuntimeError("axis persistence receipt descriptor is invalid")
    receipt_path = _axis_persistence_receipt_path(path)
    receipt_sidecar = receipt_path.with_suffix(receipt_path.suffix + ".sha256")
    if (
        descriptor.get("schema_version") != AXIS_PERSISTENCE_RECEIPT_SCHEMA_VERSION
        or descriptor.get("path") != receipt_path.name
        or descriptor.get("sidecar_path") != receipt_sidecar.name
        or receipt_path.is_symlink()
        or receipt_sidecar.is_symlink()
    ):
        raise RuntimeError("axis persistence receipt path is invalid")
    receipt = _verify_signed_json(receipt_path)
    required = {
        "schema_version",
        "axis_path",
        "axis_sha256",
        "axis_status",
        "persistence_seconds",
        "end_to_end_seconds",
        "primary_artifact_bytes",
        "persistence_receipt_bytes",
        "artifact_bytes",
        "persistence_breakdown",
        "timing_scope",
    }
    if set(receipt) != required:
        raise RuntimeError("axis persistence receipt fields are invalid")
    axis_data = path.read_bytes()
    axis_sidecar = path.with_suffix(path.suffix + ".sha256")
    axis_sha256 = hashlib.sha256(axis_data).hexdigest()
    if (
        receipt.get("schema_version") != AXIS_PERSISTENCE_RECEIPT_SCHEMA_VERSION
        or receipt.get("axis_path") != path.name
        or receipt.get("axis_sha256") != axis_sha256
        or receipt.get("axis_status") != payload.get("status")
        or axis_sidecar.read_text(encoding="ascii").strip() != axis_sha256
    ):
        raise RuntimeError("axis persistence receipt identity does not reconcile")
    external_bytes = 0
    journal = payload.get("canonical_semantic_journal")
    if journal is not None:
        if not isinstance(journal, Mapping):
            raise RuntimeError("axis semantic journal descriptor is invalid")
        bundle_bytes = journal.get("bundle_bytes")
        if isinstance(bundle_bytes, bool) or not isinstance(bundle_bytes, int):
            raise RuntimeError("axis semantic journal byte count is invalid")
        external_bytes = bundle_bytes
    if payload.get("mode") in {
        ArchitectureMode.PER_SOLVE_RUNTIME.value,
        ArchitectureMode.FULL_NATIVE_ALNS.value,
    }:
        task_receipt = path.with_suffix(path.suffix + ".native-work-tasks.jsonl")
        task_receipt_sidecar = task_receipt.with_suffix(task_receipt.suffix + ".sha256")
        if (
            task_receipt.is_symlink()
            or task_receipt_sidecar.is_symlink()
            or not task_receipt.is_file()
            or not task_receipt_sidecar.is_file()
        ):
            raise RuntimeError("axis native work-task receipt artifacts are missing")
        external_bytes += task_receipt.stat().st_size + task_receipt_sidecar.stat().st_size
    primary_bytes = len(axis_data) + axis_sidecar.stat().st_size + external_bytes
    persistence_receipt_bytes = receipt_path.stat().st_size + receipt_sidecar.stat().st_size
    for field, expected in (
        ("primary_artifact_bytes", primary_bytes),
        ("persistence_receipt_bytes", persistence_receipt_bytes),
        ("artifact_bytes", primary_bytes + persistence_receipt_bytes),
    ):
        value = receipt.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise RuntimeError(f"axis persistence receipt {field} does not reconcile")
    for field in ("persistence_seconds", "end_to_end_seconds"):
        value = receipt.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise RuntimeError(f"axis persistence receipt {field} is invalid")
    result["persistence_seconds"] = receipt["persistence_seconds"]
    result["producer_pre_receipt_seconds"] = receipt["end_to_end_seconds"]
    result["artifact_bytes"] = receipt["artifact_bytes"]
    result["persistence_breakdown"] = receipt["persistence_breakdown"]
    result["axis_persistence_receipt_evidence"] = dict(receipt)
    trajectory_descriptor = result.get("semantic_trajectory")
    if isinstance(trajectory_descriptor, Mapping):
        result["semantic_trajectory"] = _external_semantic_trajectory(
            path,
            result,
            trajectory_descriptor,
        )
    return result


_LARGE_REPLAY_ONLY_FIELDS = frozenset({"canonical_semantic_events", "canonical_semantic_streams"})


def _axis_review_projection(payload: Mapping[str, object]) -> dict[str, object]:
    """Retain report fields while releasing duplicated full semantic journals."""

    return {key: value for key, value in payload.items() if key not in _LARGE_REPLAY_ONLY_FIELDS}


def _load_axis_record(
    path: Path,
    *,
    verified_payload: Mapping[str, object] | None = None,
) -> ReviewRecord:
    raw_payload = verified_payload if verified_payload is not None else _verify_signed_json(path)
    payload = _axis_payload_with_persistence(path, raw_payload)
    if payload.get("schema_version") in PROFILE_COMPARISON_SCHEMA_VERSIONS:
        _review_axis_native_task_receipts(path, payload)
    return ReviewRecord(path, _axis_review_projection(payload))


def _review_build_attestation(manifest: Mapping[str, object]) -> tuple[str, str]:
    def lower_hex(value: object, length: int) -> bool:
        return (
            isinstance(value, str)
            and len(value) == length
            and all(character in "0123456789abcdef" for character in value)
        )

    revision = _string(manifest, "revision")
    git_tree = _string(manifest, "git_tree")
    comparison_schema = _string(manifest, "schema_version")
    if not lower_hex(revision, 40) or not lower_hex(git_tree, 40):
        raise RuntimeError("campaign Git revision/tree identity is invalid")
    observed_tree = subprocess.run(
        ("git", "rev-parse", f"{revision}^{{tree}}"),
        cwd=repository_root(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if git_tree != observed_tree:
        raise RuntimeError("campaign Git tree does not match its revision")
    committed_attestation = committed_source_attestation(
        repository_root(),
        revision,
    )
    receipt = _mapping(manifest, "wheel_receipt")

    expected_strings = {
        "build_git_revision": revision,
        "build_git_tree": git_tree,
        "wheel_sha256": _string(manifest, "wheel_sha256"),
        "native_sha256": _string(manifest, "native_sha256"),
        "scheduler_sha256": _string(manifest, "scheduler_sha256"),
    }
    if any(
        not lower_hex(value, 40 if field.startswith("build_git_") else 64)
        for field, value in expected_strings.items()
    ):
        raise RuntimeError("campaign wheel receipt identity has invalid hashes")
    if any(receipt.get(field) != value for field, value in expected_strings.items()):
        raise RuntimeError("campaign wheel receipt identity does not reconcile")
    source_manifest_sha256 = receipt.get("build_source_manifest_sha256")
    if not lower_hex(source_manifest_sha256, 64):
        raise RuntimeError("campaign source manifest SHA-256 is invalid")
    tracked_file_count = receipt.get("build_tracked_file_count")
    if isinstance(tracked_file_count, bool) or not isinstance(tracked_file_count, int):
        raise RuntimeError("campaign tracked-file count is invalid")
    if tracked_file_count <= 0:
        raise RuntimeError("campaign tracked-file count is invalid")
    if (
        source_manifest_sha256 != committed_attestation["source_manifest_sha256"]
        or tracked_file_count != committed_attestation["tracked_file_count"]
    ):
        raise RuntimeError("campaign source manifest does not match Git blobs")
    if receipt.get("build_source_dirty") is not False:
        raise RuntimeError("campaign wheel source is dirty")
    if receipt.get("build_development_override") is not False:
        raise RuntimeError("campaign wheel used a development build override")
    if receipt.get("build_cpp_source_kind") != "git_blob_snapshot":
        raise RuntimeError("campaign wheel did not use a Git-blob C++ snapshot")
    attestation_version = receipt.get("build_source_attestation_version")
    if isinstance(attestation_version, bool) or attestation_version != 1:
        raise RuntimeError("campaign source attestation version is invalid")
    performance_validation: dict[str, object] = {}
    if comparison_schema in PROFILE_COMPARISON_SCHEMA_VERSIONS:
        try:
            performance_profile = FrozenPerformanceProfile.from_dict(
                _mapping(manifest, "performance_profile")
            )
        except ValueError as error:
            raise RuntimeError("campaign performance profile is invalid") from error
        if manifest.get("performance_profile_sha256") != performance_profile.canonical_sha256:
            raise RuntimeError("campaign performance profile hash does not reconcile")
        artifact_identity = performance_profile.selected_build.artifact_identity
        if artifact_identity is None:
            raise RuntimeError("campaign performance profile lacks build identity")
        performance_validation = {
            "performance_profile": performance_profile.selected_build.name,
            "compiler_id": performance_profile.selected_build.compiler,
            "compiler_version": artifact_identity.compiler_version,
            "interprocedural_optimization": performance_profile.selected_build.lto,
            "host_native": performance_profile.selected_build.host_native,
        }
        receipt_fields = {
            "build_performance_profile": performance_validation["performance_profile"],
            "build_compiler_id": performance_validation["compiler_id"],
            "build_compiler_version": performance_validation["compiler_version"],
            "build_interprocedural_optimization": performance_validation[
                "interprocedural_optimization"
            ],
            "build_host_native": performance_validation["host_native"],
        }
        if any(receipt.get(field) != value for field, value in receipt_fields.items()):
            raise RuntimeError("campaign performance build identity does not reconcile")
    wheel_entries = receipt.get("wheel_entry_sha256")
    if (
        not isinstance(wheel_entries, dict)
        or not wheel_entries
        or any(
            not isinstance(path, str)
            or not path.startswith(("evrptw/", "tools/"))
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or str(PurePosixPath(path)) != path
            or not lower_hex(sha256, 64)
            for path, sha256 in wheel_entries.items()
        )
    ):
        raise RuntimeError("campaign wheel entry receipt is invalid")
    required_entries = {
        "native_wheel_entry": expected_strings["native_sha256"],
        "scheduler_wheel_entry": expected_strings["scheduler_sha256"],
        "runner_wheel_entry": None,
    }
    normalized_entries: dict[str, str] = {}
    for field, expected_sha256 in required_entries.items():
        entry = receipt.get(field)
        if not isinstance(entry, str):
            raise RuntimeError("campaign required wheel entry identity is missing")
        normalized = PurePosixPath(entry)
        if normalized.is_absolute() or ".." in normalized.parts or str(normalized) != entry:
            raise RuntimeError("campaign wheel entry path is not canonical")
        sha256 = wheel_entries.get(entry)
        if not lower_hex(sha256, 64) or (expected_sha256 is not None and sha256 != expected_sha256):
            raise RuntimeError("campaign required wheel entry is not hash-bound")
        normalized_entries[field] = entry
    native_entry = PurePosixPath(normalized_entries["native_wheel_entry"])
    if native_entry.parent != PurePosixPath("evrptw") or not native_entry.name.startswith("_core."):
        raise RuntimeError("campaign native extension wheel entry is not hash-bound")
    if normalized_entries["scheduler_wheel_entry"] != ("evrptw/_native_host_scheduler"):
        raise RuntimeError("campaign scheduler wheel entry is not hash-bound")
    if normalized_entries["runner_wheel_entry"] != (
        "evrptw/experiments/stage052_native_architectures.py"
    ):
        raise RuntimeError("campaign runner wheel entry is not hash-bound")
    expected_source_entries = committed_wheel_project_entry_sha256(repository_root(), revision)
    expected_wheel_entries = set(expected_source_entries) | {
        normalized_entries["native_wheel_entry"],
        normalized_entries["scheduler_wheel_entry"],
    }
    if set(wheel_entries) != expected_wheel_entries or any(
        wheel_entries[path] != expected_sha256
        for path, expected_sha256 in expected_source_entries.items()
    ):
        raise RuntimeError("campaign wheel project inventory does not match Git")
    scheduler_attestation = receipt.get("scheduler_build_attestation")
    if not isinstance(scheduler_attestation, dict):
        raise RuntimeError("campaign scheduler build attestation is invalid")
    validate_scheduler_build_attestation(
        scheduler_attestation,
        revision=revision,
        git_tree=git_tree,
        source_manifest_sha256=str(source_manifest_sha256),
        tracked_file_count=tracked_file_count,
        performance_profile=cast(str, performance_validation["performance_profile"])
        if performance_validation
        else None,
        compiler_id=cast(str, performance_validation["compiler_id"])
        if performance_validation
        else None,
        compiler_version=cast(str, performance_validation["compiler_version"])
        if performance_validation
        else None,
        interprocedural_optimization=cast(
            bool, performance_validation["interprocedural_optimization"]
        )
        if performance_validation
        else None,
        host_native=cast(bool, performance_validation["host_native"])
        if performance_validation
        else None,
    )
    return git_tree, str(source_manifest_sha256)


def _expected_keys(scope: str) -> set[tuple[int, str, str, int]]:
    if scope == "paired":
        return {
            (repeat, axis, instance, seed)
            for repeat in range(3)
            for axis in AXIS_NAMES
            for instance in PAIRED_INSTANCES
            for seed in SEEDS
        }
    if scope == "pilot":
        return {
            (0, "wall_clock_30", instance, seed) for instance in FORMAL_INSTANCES for seed in SEEDS
        }
    raise ValueError("scope must be paired or pilot")


def _review_nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{field} must be a non-negative integer")
    return value


def _review_nonnegative_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise RuntimeError(f"{field} must be a finite non-negative number")
    return float(value)


def _review_latency_quantiles(
    histogram: object,
    maximum_seconds: object,
) -> dict[str, float | str]:
    if (
        not isinstance(histogram, list)
        or len(histogram) != 32
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in histogram
        )
    ):
        raise RuntimeError("scheduler latency histogram is invalid")
    maximum = _review_nonnegative_number(
        maximum_seconds,
        "scheduler maximum latency",
    )
    total = sum(histogram)
    if total == 0:
        return {"p50": "unavailable", "p95": "unavailable", "p99": "unavailable"}
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
        quantiles[label] = (
            maximum if selected_bin == 31 else min((2 ** (selected_bin + 1)) / 1_000_000.0, maximum)
        )
    return quantiles


def _review_scheduler_queue_statistics(
    value: object,
    *,
    request_queue: bool,
    worker_threads: int,
) -> dict[str, dict[str, float | str]]:
    if not isinstance(value, dict):
        raise RuntimeError("scheduler queue statistics are missing")
    common_fields = {
        "pending",
        "peak_pending",
        "queue_full_count",
        "rejected_count",
        "completed",
        "total_wait_seconds",
        "maximum_wait_seconds",
        "total_service_seconds",
        "maximum_service_seconds",
        "wait_histogram",
        "service_histogram",
    }
    expected_fields = (
        common_fields
        if request_queue
        else common_fields
        | {
            "active",
            "peak_active",
        }
    )
    if set(value) != expected_fields:
        raise RuntimeError("scheduler queue statistics fields are invalid")
    integer_fields = {
        "pending",
        "peak_pending",
        "queue_full_count",
        "rejected_count",
        "completed",
    }
    if not request_queue:
        integer_fields |= {"active", "peak_active"}
    counters = {
        field: _review_nonnegative_integer(value[field], f"scheduler queue {field}")
        for field in integer_fields
    }
    numbers = {
        field: _review_nonnegative_number(value[field], f"scheduler queue {field}")
        for field in (
            "total_wait_seconds",
            "maximum_wait_seconds",
            "total_service_seconds",
            "maximum_service_seconds",
        )
    }
    wait_histogram = value["wait_histogram"]
    service_histogram = value["service_histogram"]
    wait_quantiles = _review_latency_quantiles(
        wait_histogram,
        value["maximum_wait_seconds"],
    )
    service_quantiles = _review_latency_quantiles(
        service_histogram,
        value["maximum_service_seconds"],
    )
    assert isinstance(wait_histogram, list)
    assert isinstance(service_histogram, list)
    if (
        sum(wait_histogram) != counters["completed"]
        or sum(service_histogram) != counters["completed"]
        or numbers["total_wait_seconds"] + 1e-12 < numbers["maximum_wait_seconds"]
        or numbers["total_service_seconds"] + 1e-12 < numbers["maximum_service_seconds"]
    ):
        raise RuntimeError("scheduler queue latency accounting does not reconcile")
    if (
        counters["pending"] != 0
        or counters["queue_full_count"] != 0
        or counters["rejected_count"] != 0
        or (not request_queue and counters["active"] != 0)
    ):
        raise RuntimeError("scheduler queue resource gate failed")
    maximum_pending = 64 if request_queue else max(1, worker_threads * 2)
    if counters["peak_pending"] > maximum_pending or (
        not request_queue and counters["peak_active"] > worker_threads
    ):
        raise RuntimeError("scheduler queue exceeded its bounded topology")
    return {
        "wait_seconds": wait_quantiles,
        "service_seconds": service_quantiles,
    }


def _review_scheduler_runtime_statistics(
    value: object,
    *,
    worker_threads: int,
    request_threads: int,
    evidence_root: Path,
) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("scheduler runtime statistics are missing")
    schema_version = value.get("schema_version")
    expected_fields = {
            "schema_version",
            "worker_threads",
            "request_threads",
            "receipt_writer_threads",
            "peak_active_requests",
            "peak_distinct_client_pids",
            "request_queue",
            "work_queue",
            "latency_quantiles",
            "task_receipts",
        }
    if schema_version == "stage05.2-native-scheduler-runtime-v4":
        expected_fields.add("terminal_process_io")
    elif schema_version != "stage05.2-native-scheduler-runtime-v3":
        raise RuntimeError("scheduler runtime statistics schema is invalid")
    if set(value) != expected_fields:
        raise RuntimeError("scheduler runtime statistics schema is invalid")
    if schema_version == "stage05.2-native-scheduler-runtime-v4":
        terminal_io = value.get("terminal_process_io")
        if not isinstance(terminal_io, Mapping) or set(terminal_io) != {
            "read_bytes",
            "write_bytes",
        }:
            raise RuntimeError("scheduler terminal process I/O is invalid")
        for name in ("read_bytes", "write_bytes"):
            _review_nonnegative_integer(
                terminal_io.get(name), f"scheduler terminal process I/O {name}"
            )
    if (
        _review_nonnegative_integer(value["worker_threads"], "scheduler worker_threads")
        != worker_threads
        or _review_nonnegative_integer(
            value["request_threads"],
            "scheduler request_threads",
        )
        != request_threads
        or _review_nonnegative_integer(
            value["receipt_writer_threads"],
            "scheduler receipt_writer_threads",
        )
        != 1
    ):
        raise RuntimeError("scheduler runtime topology does not replay")
    _review_nonnegative_integer(
        value["peak_active_requests"],
        "scheduler peak_active_requests",
    )
    _review_nonnegative_integer(
        value["peak_distinct_client_pids"],
        "scheduler peak_distinct_client_pids",
    )
    recomputed = {
        "request_queue": _review_scheduler_queue_statistics(
            value["request_queue"],
            request_queue=True,
            worker_threads=worker_threads,
        ),
        "work_queue": _review_scheduler_queue_statistics(
            value["work_queue"],
            request_queue=False,
            worker_threads=worker_threads,
        ),
    }
    work_queue = cast(Mapping[str, object], value["work_queue"])
    _review_scheduler_task_receipts(
        value["task_receipts"],
        evidence_root=evidence_root,
        worker_threads=worker_threads,
        completed_tasks=_review_nonnegative_integer(
            work_queue["completed"], "scheduler work completed"
        ),
    )
    reported = value["latency_quantiles"]
    if not isinstance(reported, dict) or set(reported) != set(recomputed):
        raise RuntimeError("scheduler latency quantiles are missing")
    for queue_name, recomputed_queue in recomputed.items():
        reported_queue = reported.get(queue_name)
        if not isinstance(reported_queue, dict) or set(reported_queue) != set(recomputed_queue):
            raise RuntimeError("scheduler latency quantiles are invalid")
        for latency_name, recomputed_quantiles in recomputed_queue.items():
            reported_quantiles = reported_queue.get(latency_name)
            if not isinstance(reported_quantiles, dict) or set(reported_quantiles) != set(
                recomputed_quantiles
            ):
                raise RuntimeError("scheduler latency quantiles are invalid")
            for quantile, expected in recomputed_quantiles.items():
                actual = reported_quantiles.get(quantile)
                if isinstance(expected, str):
                    matches = actual == expected
                else:
                    matches = (
                        not isinstance(actual, bool)
                        and isinstance(actual, int | float)
                        and math.isclose(
                            float(actual),
                            expected,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    )
                if not matches:
                    raise RuntimeError("scheduler latency quantiles do not independently replay")


def _review_scheduler_task_receipts(
    value: object,
    *,
    evidence_root: Path,
    worker_threads: int,
    completed_tasks: int,
) -> None:
    expected_fields = {
        "schema_version",
        "path",
        "sha256",
        "bytes",
        "count",
        "storage_model",
        "receipt_batch_capacity",
        "queue_bound_batches",
        "peak_queued_batches",
        "submitted_batches",
        "completed_batches",
        "dropped_count",
        "producer_wait_seconds",
        "writer_wall_seconds",
        "writer_cpu_seconds",
        "serialization_seconds",
        "write_seconds",
        "file_fsync_seconds",
        "atomic_publish_seconds",
        "parent_fsync_seconds",
        "sidecar_path",
        "sidecar_sha256",
        "validated_worker_indices",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise RuntimeError("scheduler task-receipt descriptor is invalid")
    raw_path = value.get("path")
    raw_sidecar = value.get("sidecar_path")
    if (
        not isinstance(raw_path, str)
        or not isinstance(raw_sidecar, str)
        or Path(raw_path).name != raw_path
        or Path(raw_sidecar).name != raw_sidecar
        or raw_sidecar != raw_path + ".sha256"
    ):
        raise RuntimeError("scheduler task-receipt path is invalid")
    path = evidence_root / raw_path
    sidecar = evidence_root / raw_sidecar
    if path.is_symlink() or sidecar.is_symlink() or not path.is_file() or not sidecar.is_file():
        raise RuntimeError("scheduler task-receipt files are missing")
    count = _review_nonnegative_integer(value.get("count"), "scheduler task count")
    batch_capacity = _review_nonnegative_integer(
        value.get("receipt_batch_capacity"), "scheduler task batch capacity"
    )
    queue_bound = _review_nonnegative_integer(
        value.get("queue_bound_batches"), "scheduler task queue bound"
    )
    peak_queued = _review_nonnegative_integer(
        value.get("peak_queued_batches"), "scheduler task peak queue"
    )
    submitted_batches = _review_nonnegative_integer(
        value.get("submitted_batches"), "scheduler task submitted batches"
    )
    completed_batches = _review_nonnegative_integer(
        value.get("completed_batches"), "scheduler task completed batches"
    )
    dropped = _review_nonnegative_integer(
        value.get("dropped_count"), "scheduler task dropped_count"
    )
    declared_bytes = _review_nonnegative_integer(value.get("bytes"), "scheduler task bytes")
    declared_sha256 = value.get("sha256")
    declared_sidecar_sha256 = value.get("sidecar_sha256")
    for field in (
        "producer_wait_seconds",
        "writer_wall_seconds",
        "writer_cpu_seconds",
        "serialization_seconds",
        "write_seconds",
        "file_fsync_seconds",
        "atomic_publish_seconds",
        "parent_fsync_seconds",
    ):
        _review_nonnegative_number(value.get(field), f"scheduler task {field}")
    if (
        value.get("schema_version") != "stage05.2-native-work-task-receipts-v3"
        or value.get("storage_model") != "bounded_async_fifo_stream"
        or count != completed_tasks
        or batch_capacity != 4_096
        or queue_bound != 1
        or peak_queued > queue_bound
        or submitted_batches != completed_batches
        or (count == 0) != (completed_batches == 0)
        or dropped != 0
        or declared_bytes != path.stat().st_size
        or not isinstance(declared_sha256, str)
        or not isinstance(declared_sidecar_sha256, str)
        or len(declared_sha256) != 64
        or len(declared_sidecar_sha256) != 64
        or any(character not in "0123456789abcdef" for character in declared_sha256)
        or any(character not in "0123456789abcdef" for character in declared_sidecar_sha256)
    ):
        raise RuntimeError("scheduler task-receipt counters are invalid")
    digest = hashlib.sha256()
    batch_digest = hashlib.sha256()
    task_sequences: set[int] = set()
    worker_indices: set[int] = set()
    observed_count = 0
    observed_batches = 0
    batch_count = 0
    batch_first: int | None = None
    batch_last: int | None = None
    trailer_seen = False
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError as error:
                raise RuntimeError("scheduler task-receipt JSONL is invalid") from error
            if not isinstance(row, dict):
                raise RuntimeError("scheduler task-receipt row is invalid")
            if row.get("kind") == "trailer":
                if (
                    trailer_seen
                    or batch_count != 0
                    or observed_count != count
                    or observed_batches != completed_batches
                    or row
                    != {
                        "kind": "trailer",
                        "schema_version": "stage05.2-native-work-task-receipts-v3",
                        "storage_model": "bounded_async_fifo_stream",
                        "receipt_batch_capacity": batch_capacity,
                        "queue_bound_batches": queue_bound,
                        "submitted_batches": submitted_batches,
                        "completed_batches": completed_batches,
                        "task_receipt_dropped_count": dropped,
                        "completed_tasks": completed_tasks,
                        "receipt_count": count,
                    }
                ):
                    raise RuntimeError("scheduler task-receipt trailer is invalid")
                trailer_seen = True
                continue
            if trailer_seen:
                raise RuntimeError("scheduler task receipt follows its trailer")
            if row.get("kind") == "batch":
                if (
                    set(row)
                    != {
                        "kind",
                        "batch_ordinal",
                        "row_count",
                        "first_task_sequence",
                        "last_task_sequence",
                        "sha256",
                    }
                    or row.get("batch_ordinal") != observed_batches
                    or row.get("row_count") != batch_count
                    or row.get("first_task_sequence") != batch_first
                    or row.get("last_task_sequence") != batch_last
                    or row.get("sha256") != batch_digest.hexdigest()
                    or not 0 < batch_count <= batch_capacity
                ):
                    raise RuntimeError("scheduler task-receipt batch ledger is invalid")
                observed_batches += 1
                batch_digest = hashlib.sha256()
                batch_count = 0
                batch_first = None
                batch_last = None
                continue
            if (
                set(row)
                != {
                    "kind",
                    "task_sequence",
                    "worker_index",
                    "first_index",
                    "last_index",
                    "submitted_nanoseconds",
                    "started_nanoseconds",
                    "completed_nanoseconds",
                }
                or row.get("kind") != "task"
            ):
                raise RuntimeError("scheduler task-receipt row fields are invalid")
            batch_digest.update(line)
            task_sequence = _review_nonnegative_integer(
                row["task_sequence"], "scheduler task sequence"
            )
            worker_index = _review_nonnegative_integer(row["worker_index"], "scheduler task worker")
            first = _review_nonnegative_integer(row["first_index"], "scheduler task first")
            last = _review_nonnegative_integer(row["last_index"], "scheduler task last")
            submitted = _review_nonnegative_integer(
                row["submitted_nanoseconds"], "scheduler task submitted"
            )
            started = _review_nonnegative_integer(
                row["started_nanoseconds"], "scheduler task started"
            )
            completed = _review_nonnegative_integer(
                row["completed_nanoseconds"], "scheduler task completed"
            )
            if (
                task_sequence in task_sequences
                or worker_index >= worker_threads
                or first >= last
                or not submitted <= started <= completed
            ):
                raise RuntimeError("scheduler task-receipt ordering is invalid")
            task_sequences.add(task_sequence)
            worker_indices.add(worker_index)
            if batch_first is None:
                batch_first = task_sequence
            batch_last = task_sequence
            batch_count += 1
            observed_count += 1
    sidecar_data = sidecar.read_bytes()
    if (
        digest.hexdigest() != declared_sha256
        or sidecar_data != (declared_sha256 + "\n").encode("ascii")
        or hashlib.sha256(sidecar_data).hexdigest() != declared_sidecar_sha256
        or observed_count != count
        or observed_batches != completed_batches
        or batch_count != 0
        or not trailer_seen
        or value.get("validated_worker_indices") != sorted(worker_indices)
    ):
        raise RuntimeError("scheduler task-receipt digest/count does not replay")


def _review_local_native_work_pool(
    value: object,
    *,
    evidence_root: Path,
    expected_schema: str,
) -> None:
    fields = {
        "schema_version",
        "thread_count",
        "maximum_pending_tasks",
        "pending_tasks",
        "active_tasks",
        "peak_pending_tasks",
        "peak_active_tasks",
        "queue_full_count",
        "rejected_count",
        "completed_tasks",
        "total_wait_seconds",
        "maximum_wait_seconds",
        "total_service_seconds",
        "maximum_service_seconds",
        "wait_histogram",
        "service_histogram",
        "task_receipt_dropped_count",
        "task_receipts",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise RuntimeError("local native work-pool receipt is invalid")
    if value.get("schema_version") != expected_schema:
        raise RuntimeError("local native work-pool schema is invalid")
    counters = {
        field: _review_nonnegative_integer(value.get(field), f"local work-pool {field}")
        for field in (
            "thread_count",
            "maximum_pending_tasks",
            "pending_tasks",
            "active_tasks",
            "peak_pending_tasks",
            "peak_active_tasks",
            "queue_full_count",
            "rejected_count",
            "completed_tasks",
            "task_receipt_dropped_count",
        )
    }
    if (
        counters["thread_count"] == 0
        or counters["maximum_pending_tasks"] == 0
        or counters["pending_tasks"] != 0
        or counters["active_tasks"] != 0
        or counters["queue_full_count"] != 0
        or counters["rejected_count"] != 0
        or counters["task_receipt_dropped_count"] != 0
        or counters["peak_pending_tasks"] > counters["maximum_pending_tasks"]
        or counters["peak_active_tasks"] > counters["thread_count"]
    ):
        raise RuntimeError("local native work-pool resource gate failed")
    timings = {
        field: _review_nonnegative_number(value.get(field), f"local work-pool {field}")
        for field in (
            "total_wait_seconds",
            "maximum_wait_seconds",
            "total_service_seconds",
            "maximum_service_seconds",
        )
    }
    if (
        timings["maximum_wait_seconds"] > timings["total_wait_seconds"] + 1e-12
        or timings["maximum_service_seconds"] > timings["total_service_seconds"] + 1e-12
    ):
        raise RuntimeError("local native work-pool timing does not reconcile")
    for field in ("wait_histogram", "service_histogram"):
        histogram = value.get(field)
        if (
            not isinstance(histogram, list)
            or len(histogram) != 32
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in histogram
            )
            or sum(histogram) != counters["completed_tasks"]
        ):
            raise RuntimeError("local native work-pool histogram does not reconcile")
    _review_scheduler_task_receipts(
        value.get("task_receipts"),
        evidence_root=evidence_root,
        worker_threads=counters["thread_count"],
        completed_tasks=counters["completed_tasks"],
    )


def _review_axis_native_task_receipts(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    mode = payload.get("mode")
    if mode == ArchitectureMode.PER_SOLVE_RUNTIME.value:
        candidate_statistics = payload.get("candidate_transaction_statistics")
        if not isinstance(candidate_statistics, Mapping):
            raise RuntimeError("per-solve candidate statistics are missing")
        work_pool = candidate_statistics.get("native_candidate_work_pool")
        if not isinstance(work_pool, Mapping) or work_pool.get("enabled") is not True:
            raise RuntimeError("per-solve native work-pool receipt is missing")
        _review_local_native_work_pool(
            {key: value for key, value in work_pool.items() if key != "enabled"},
            evidence_root=path.parent,
            expected_schema="stage05.2-candidate-round-work-pool-v3",
        )
    elif mode == ArchitectureMode.FULL_NATIVE_ALNS.value:
        native_statistics = payload.get("native_execution_statistics")
        if not isinstance(native_statistics, Mapping):
            raise RuntimeError("full-native execution statistics are missing")
        _review_local_native_work_pool(
            native_statistics.get("work_pool_statistics"),
            evidence_root=path.parent,
            expected_schema="stage05.2-full-native-work-pool-v3",
        )


def _review_process_tree_accounting(
    wave: Mapping[str, object],
) -> tuple[float, float, dict[str, int]]:
    raw_metrics = wave.get("process_metrics")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise RuntimeError("campaign process-tree rows are unavailable")
    fields = (
        "voluntary_context_switches",
        "involuntary_context_switches",
        "minor_faults",
        "major_faults",
        "read_bytes",
        "write_bytes",
        "schedstat_runtime_ns",
        "schedstat_runqueue_delay_ns",
        "schedstat_timeslices",
        "cpu_migrations",
    )
    totals = {field: 0 for field in fields}
    user_seconds = 0.0
    system_seconds = 0.0
    identities: set[tuple[int, float]] = set()
    affinity_union: set[int] = set()
    affinity_intersection: set[int] | None = None
    for index, raw_metric in enumerate(raw_metrics):
        if not isinstance(raw_metric, Mapping):
            raise RuntimeError("campaign process-tree row is invalid")
        pid = _review_nonnegative_integer(raw_metric.get("pid"), f"process_metrics[{index}].pid")
        create_time = _review_nonnegative_number(
            raw_metric.get("create_time"),
            f"process_metrics[{index}].create_time",
        )
        if pid <= 0 or create_time <= 0.0:
            raise RuntimeError("campaign process-tree identity is invalid")
        identity = (pid, create_time)
        if identity in identities:
            raise RuntimeError("campaign process-tree identity is duplicated")
        identities.add(identity)
        if _review_nonnegative_integer(
            raw_metric.get("sample_count"),
            f"process_metrics[{index}].sample_count",
        ) == 0:
            raise RuntimeError("campaign process-tree sample count is invalid")
        if raw_metric.get("cpu_baseline_source") not in {
            "monitor_start",
            "process_create_time",
        }:
            raise RuntimeError("campaign process-tree CPU baseline is invalid")
        user_seconds += _review_nonnegative_number(
            raw_metric.get("user_cpu_seconds"),
            f"process_metrics[{index}].user_cpu_seconds",
        )
        system_seconds += _review_nonnegative_number(
            raw_metric.get("system_cpu_seconds"),
            f"process_metrics[{index}].system_cpu_seconds",
        )
        counters = raw_metric.get("counters")
        if not isinstance(counters, Mapping) or set(counters) != set(fields):
            raise RuntimeError("campaign process-tree counter schema is invalid")
        for field in fields:
            totals[field] += _review_nonnegative_integer(
                counters.get(field),
                f"process_metrics[{index}].{field}",
            )
        if raw_metric.get("cpu_migrations") != counters["cpu_migrations"]:
            raise RuntimeError("campaign process-tree migration alias differs")
        schedstat = raw_metric.get("schedstat")
        if not isinstance(schedstat, Mapping) or dict(schedstat) != {
            "runtime_ns": counters["schedstat_runtime_ns"],
            "runqueue_delay_ns": counters["schedstat_runqueue_delay_ns"],
            "timeslices": counters["schedstat_timeslices"],
        }:
            raise RuntimeError("campaign process-tree schedstat row differs")
        affinity = raw_metric.get("last_affinity")
        if (
            not isinstance(affinity, list)
            or not affinity
            or any(
                isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
                for cpu in affinity
            )
            or affinity != sorted(set(affinity))
        ):
            raise RuntimeError("campaign process-tree affinity is invalid")
        selected = set(cast(list[int], affinity))
        affinity_union.update(selected)
        affinity_intersection = (
            selected
            if affinity_intersection is None
            else affinity_intersection.intersection(selected)
        )
    cpu_fields = {
        "process_tree_user_cpu_seconds": user_seconds,
        "process_tree_system_cpu_seconds": system_seconds,
        "process_tree_cpu_seconds": user_seconds + system_seconds,
        "process_tree_cpu_user_seconds": user_seconds,
        "process_tree_cpu_system_seconds": system_seconds,
    }
    mismatches = [
        name
        for name, expected in cpu_fields.items()
        if not math.isclose(
            _review_nonnegative_number(wave.get(name), f"campaign {name}"),
            expected,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ]
    expected_fields: dict[str, object] = {
        "process_tree_voluntary_context_switches": totals[
            "voluntary_context_switches"
        ],
        "process_tree_involuntary_context_switches": totals[
            "involuntary_context_switches"
        ],
        "process_tree_context_switches": totals["voluntary_context_switches"]
        + totals["involuntary_context_switches"],
        "process_tree_minor_faults": totals["minor_faults"],
        "process_tree_major_faults": totals["major_faults"],
        "process_tree_read_bytes": totals["read_bytes"],
        "process_tree_write_bytes": totals["write_bytes"],
        "process_tree_cpu_migrations": totals["cpu_migrations"],
        "process_tree_migration_count": totals["cpu_migrations"],
        "process_tree_schedstat": {
            "runtime_ns": totals["schedstat_runtime_ns"],
            "runqueue_delay_ns": totals["schedstat_runqueue_delay_ns"],
            "timeslices": totals["schedstat_timeslices"],
        },
        "actual_affinity_union": sorted(affinity_union),
        "actual_affinity_intersection": sorted(affinity_intersection or set()),
        "cpu_affinity_union": sorted(affinity_union),
        "cpu_affinity_intersection": sorted(affinity_intersection or set()),
        "actual_affinity_union_count": len(affinity_union),
        "actual_affinity_intersection_count": len(affinity_intersection or set()),
        "affinity_status": "available",
    }
    mismatches.extend(
        name for name, expected in expected_fields.items() if wave.get(name) != expected
    )
    if mismatches:
        raise RuntimeError(
            "campaign process-tree aggregates do not replay: "
            + ", ".join(sorted(mismatches))
        )
    return user_seconds, system_seconds, totals


def _review_process_tree_io(wave: Mapping[str, object]) -> dict[str, int]:
    _user_seconds, _system_seconds, totals = _review_process_tree_accounting(wave)
    return {field: totals[field] for field in ("read_bytes", "write_bytes")}


def _review_worker_terminal_io(wave: Mapping[str, object]) -> None:
    """Replay current mode-wave worker terminal I/O identities and counters."""

    raw_core = wave.get("terminal_process_io_receipts")
    raw_bound = wave.get("worker_terminal_io_receipts")
    raw_scheduler = wave.get("scheduler_terminal_io_receipts")
    raw_identities = wave.get("identities")
    if (
        not isinstance(raw_core, list)
        or not isinstance(raw_bound, list)
        or not isinstance(raw_scheduler, list)
        or not isinstance(raw_identities, list)
        or not raw_core
        or len(raw_core) < len(raw_identities)
        or len(raw_bound) + len(raw_scheduler) != len(raw_core)
    ):
        raise RuntimeError("campaign worker terminal I/O inventory is incomplete")
    monitor_start_wall = _review_nonnegative_number(
        wave.get("monitor_start_wall_time"),
        "campaign monitor start wall time",
    )
    monitor_start = _review_nonnegative_number(
        wave.get("monitor_start_monotonic"),
        "campaign monitor start",
    )
    monitor_end = _review_nonnegative_number(
        wave.get("monitor_end_monotonic"),
        "campaign monitor end",
    )
    if min(monitor_start_wall, monitor_start) <= 0.0 or monitor_end < monitor_start:
        raise RuntimeError("campaign terminal I/O monitor interval is invalid")
    root_process_id = _review_nonnegative_integer(
        wave.get("root_process_id"),
        "campaign root process ID",
    )
    core_fields = {
        "schema_version",
        "pid",
        "parent_pid",
        "create_time",
        "start_time_ticks",
        "task_started_monotonic",
        "captured_monotonic",
        "read_bytes",
        "write_bytes",
    }
    core_by_identity: dict[tuple[int, float], dict[str, object]] = {}
    for index, raw_receipt in enumerate(raw_core):
        if not isinstance(raw_receipt, Mapping):
            raise RuntimeError("campaign terminal process I/O receipt is invalid")
        receipt = dict(raw_receipt)
        if (
            set(receipt) != core_fields
            or receipt.get("schema_version")
            != TERMINAL_PROCESS_IO_RECEIPT_SCHEMA_VERSION
        ):
            raise RuntimeError("campaign terminal process I/O schema is invalid")
        pid = _review_nonnegative_integer(
            receipt.get("pid"),
            f"campaign terminal process I/O PID {index}",
        )
        create_time = _review_nonnegative_number(
            receipt.get("create_time"),
            f"campaign terminal process I/O create_time {index}",
        )
        start_time_ticks = _review_nonnegative_integer(
            receipt.get("start_time_ticks"),
            f"campaign terminal process I/O start_time_ticks {index}",
        )
        task_started = _review_nonnegative_number(
            receipt.get("task_started_monotonic"),
            f"campaign terminal process I/O task start {index}",
        )
        captured = _review_nonnegative_number(
            receipt.get("captured_monotonic"),
            f"campaign terminal process I/O capture time {index}",
        )
        if (
            pid <= 0
            or _review_nonnegative_integer(
                receipt.get("parent_pid"),
                "campaign terminal process I/O parent PID",
            )
            <= 0
            or create_time <= 0.0
            or start_time_ticks
            < _review_nonnegative_integer(
                wave.get("monitor_start_boot_time_ticks"),
                "campaign monitor start boot ticks",
            )
            or task_started < monitor_start
            or captured < task_started
            or captured > monitor_end
        ):
            raise RuntimeError("campaign terminal process I/O identity is invalid")
        for name in ("read_bytes", "write_bytes"):
            _review_nonnegative_integer(
                receipt.get(name),
                f"campaign terminal process I/O {name}",
            )
        identity = (pid, create_time)
        if identity in core_by_identity:
            raise RuntimeError("campaign terminal process I/O identity is duplicated")
        core_by_identity[identity] = receipt

    raw_metrics = wave.get("process_metrics")
    if not isinstance(raw_metrics, list):
        raise RuntimeError("campaign terminal process I/O rows are unavailable")
    metrics: dict[tuple[int, float], Mapping[str, object]] = {}
    metric_pids: set[int] = set()
    for raw_metric in raw_metrics:
        if not isinstance(raw_metric, Mapping):
            raise RuntimeError("campaign terminal process I/O row is invalid")
        identity = (
            _review_nonnegative_integer(raw_metric.get("pid"), "campaign process PID"),
            _review_nonnegative_number(
                raw_metric.get("create_time"),
                "campaign process create_time",
            ),
        )
        metric_pids.add(identity[0])
        metrics[identity] = raw_metric
    final_sample_index = (
        _review_nonnegative_integer(wave.get("sample_count"), "campaign sample count") - 1
    )
    if (
        wave.get("process_io_terminal_status") != "available"
        or wave.get("process_io_uncovered_identities") != []
        or final_sample_index < 0
    ):
        raise RuntimeError("campaign terminal process I/O coverage is unavailable")
    for identity, metric in metrics.items():
        terminal_evidence = metric.get("terminal_io_evidence")
        if terminal_evidence == "cooperative_receipt":
            if identity not in core_by_identity:
                raise RuntimeError("campaign cooperative terminal I/O row is unbound")
        elif terminal_evidence == "monitor_end_sample":
            if metric.get("last_sample_index") != final_sample_index:
                raise RuntimeError("campaign monitor-end terminal I/O row is invalid")
        else:
            raise RuntimeError("campaign process terminal I/O evidence is invalid")
    for identity, receipt in core_by_identity.items():
        worker_metric = metrics.get(identity)
        if worker_metric is None:
            raise RuntimeError("campaign terminal process I/O worker row is unavailable")
        counters = worker_metric.get("counters")
        if not isinstance(counters, Mapping) or any(
            counters.get(name) != receipt[name] for name in ("read_bytes", "write_bytes")
        ):
            raise RuntimeError("campaign terminal process I/O counters do not bind worker row")
        if (
            receipt["parent_pid"] not in metric_pids | {root_process_id}
            or worker_metric.get("parent_pid") != receipt["parent_pid"]
            or worker_metric.get("start_time_ticks") != receipt["start_time_ticks"]
            or worker_metric.get("cpu_baseline_source") != "process_create_time"
            or worker_metric.get("terminal_io_evidence") != "cooperative_receipt"
        ):
            raise RuntimeError("campaign terminal process I/O identity does not bind worker row")

    expected_axes: set[tuple[int, str, str, int]] = set()
    for raw_axis in raw_identities:
        if not isinstance(raw_axis, Mapping):
            raise RuntimeError("campaign terminal process I/O axis identity is invalid")
        expected_axes.add(
            (
                _review_nonnegative_integer(raw_axis.get("repeat"), "campaign repeat"),
                _string(raw_axis, "axis"),
                _string(raw_axis, "instance"),
                _review_nonnegative_integer(raw_axis.get("seed"), "campaign seed"),
            )
        )
    observed_axes: set[tuple[int, str, str, int]] = set()
    observed_processes: set[tuple[int, float]] = set()
    for raw_receipt in raw_bound:
        if not isinstance(raw_receipt, Mapping):
            raise RuntimeError("campaign bound terminal process I/O receipt is invalid")
        receipt = dict(raw_receipt)
        if set(receipt) != core_fields | {"repeat", "axis", "instance", "seed"}:
            raise RuntimeError("campaign bound terminal process I/O schema is invalid")
        axis_identity = (
            _review_nonnegative_integer(receipt.get("repeat"), "terminal I/O repeat"),
            _string(receipt, "axis"),
            _string(receipt, "instance"),
            _review_nonnegative_integer(receipt.get("seed"), "terminal I/O seed"),
        )
        process_identity = (
            _review_nonnegative_integer(receipt.get("pid"), "terminal I/O PID"),
            _review_nonnegative_number(receipt.get("create_time"), "terminal I/O create_time"),
        )
        core = {name: receipt[name] for name in core_fields}
        if (
            axis_identity not in expected_axes
            or process_identity in observed_processes
            or core_by_identity.get(process_identity) != core
        ):
            raise RuntimeError("campaign bound terminal process I/O receipt does not replay")
        observed_axes.add(axis_identity)
        observed_processes.add(process_identity)
    scheduler_processes: set[tuple[int, float]] = set()
    for raw_receipt in raw_scheduler:
        if not isinstance(raw_receipt, Mapping):
            raise RuntimeError("campaign scheduler terminal process I/O receipt is invalid")
        receipt = dict(raw_receipt)
        process_identity = (
            _review_nonnegative_integer(receipt.get("pid"), "scheduler terminal I/O PID"),
            _review_nonnegative_number(
                receipt.get("create_time"),
                "scheduler terminal I/O create_time",
            ),
        )
        if (
            set(receipt) != core_fields
            or process_identity in scheduler_processes
            or core_by_identity.get(process_identity) != receipt
        ):
            raise RuntimeError("campaign scheduler terminal process I/O does not replay")
        scheduler_processes.add(process_identity)
    scheduler_pids = wave.get("scheduler_process_ids")
    if not isinstance(scheduler_pids, list) or {
        identity[0] for identity in scheduler_processes
    } != set(scheduler_pids):
        raise RuntimeError("campaign scheduler terminal process I/O identity differs")
    if (
        observed_axes != expected_axes
        or observed_processes.intersection(scheduler_processes)
        or observed_processes | scheduler_processes != set(core_by_identity)
    ):
        raise RuntimeError("campaign bound terminal process I/O inventory differs")


def _review_current_cpu_budget(
    statistics: Mapping[str, object],
    *,
    elapsed_seconds: float,
    compute_limit: int,
    cpu_seconds: float,
    label: str,
) -> None:
    """Independently replay the current tick-quantized CPU accounting bound."""

    clock_tick_hz = _review_nonnegative_integer(
        statistics.get("cpu_clock_tick_hz"),
        f"{label} CPU clock tick frequency",
    )
    peak_processes = _review_nonnegative_integer(
        statistics.get("peak_concurrent_processes"),
        f"{label} peak concurrent process count",
    )
    effective_elapsed = _review_nonnegative_number(
        statistics.get("effective_elapsed_seconds"),
        f"{label} effective elapsed time",
    )
    if (
        clock_tick_hz == 0
        or peak_processes == 0
        or effective_elapsed <= 0.0
        or statistics.get("compute_thread_limit") != compute_limit
        or statistics.get("cpu_normalized_within_limit") is not True
    ):
        raise RuntimeError(f"{label} CPU accounting identity is invalid")
    try:
        observed_clock_tick_hz = int(os.sysconf("SC_CLK_TCK"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"{label} CPU clock tick frequency is unavailable") from error
    monitor_start = _review_nonnegative_number(
        statistics.get("monitor_start_monotonic"),
        f"{label} monitor start",
    )
    monitor_end = _review_nonnegative_number(
        statistics.get("monitor_end_monotonic"),
        f"{label} monitor end",
    )
    monitor_elapsed = monitor_end - monitor_start
    expected_effective_elapsed = elapsed_seconds
    if (
        clock_tick_hz != observed_clock_tick_hz
        or monitor_end < monitor_start
        or monitor_elapsed > elapsed_seconds + 1e-9
        or not math.isclose(
            effective_elapsed,
            expected_effective_elapsed,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise RuntimeError(f"{label} CPU accounting clock identity is invalid")
    allowed_cpu_seconds = elapsed_seconds * compute_limit
    expected_quantization_lanes = compute_limit
    if statistics.get("cpu_quantization_lane_count") != expected_quantization_lanes:
        raise RuntimeError(f"{label} CPU quantization lane count is invalid")
    expected_tolerance = max(
        1e-6,
        allowed_cpu_seconds * 1e-6,
        expected_quantization_lanes / clock_tick_hz,
    )
    reported_tolerance = _review_nonnegative_number(
        statistics.get("cpu_limit_tolerance_seconds"),
        f"{label} CPU accounting tolerance",
    )
    if (
        not math.isclose(
            reported_tolerance,
            expected_tolerance,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or cpu_seconds > allowed_cpu_seconds + expected_tolerance
    ):
        raise RuntimeError(f"{label} CPU accounting exceeds its quantized budget")
    expected_utilization = min(
        100.0,
        100.0 * cpu_seconds / allowed_cpu_seconds,
    )
    reported_utilization = _review_nonnegative_number(
        statistics.get("cpu_utilization_percent_of_compute_limit"),
        f"{label} compute-limit utilization",
    )
    if not math.isclose(
        reported_utilization,
        expected_utilization,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError(f"{label} normalized CPU utilization does not replay")


def _review_mode_wave_resources(
    wave: Mapping[str, object],
    *,
    comparison_schema: str = SCHEMA_VERSION,
) -> None:
    if (
        wave.get("worker_process_lifecycle") != "one_shard_per_spawned_process"
        or wave.get("worker_multiprocessing_start_method") != "spawn"
        or wave.get("worker_max_tasks_per_child") != 1
    ):
        raise RuntimeError("campaign worker process lifecycle is invalid")
    if comparison_schema in {
        PROCESS_PROFILE_COMPARISON_SCHEMA_VERSION,
        TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }:
        if (
            wave.get("resource_summary_accounting_source")
            != MODE_WAVE_RESOURCE_ACCOUNTING_SOURCE
        ):
            raise RuntimeError("campaign resource summary provider is invalid")
        _review_process_tree_accounting(wave)
    elif "resource_summary_accounting_source" in wave:
        raise RuntimeError("campaign legacy resource summary provider is invalid")
    if comparison_schema == SCHEMA_VERSION:
        _review_worker_terminal_io(wave)
    elif (
        "worker_terminal_io_receipts" in wave
        or "terminal_process_io_receipts" in wave
    ):
        raise RuntimeError("campaign historical resource evidence has terminal I/O fields")
    before = wave.get("cgroup_before")
    after = wave.get("cgroup_after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise RuntimeError("campaign cgroup resource evidence is missing")
    cgroup_path = before.get("cgroup_path")
    if (
        before.get("status") != "available"
        or after.get("status") != "available"
        or not isinstance(cgroup_path, str)
        or cgroup_path == "unavailable"
        or after.get("cgroup_path") != cgroup_path
    ):
        raise RuntimeError("campaign cgroup resource identity is invalid")
    counter_fields = (
        "memory_current_bytes",
        "memory_peak_bytes",
        "memory_swap_current_bytes",
        "memory_swap_peak_bytes",
    )
    for snapshot in (before, after):
        for field in counter_fields:
            _review_nonnegative_integer(
                snapshot.get(field),
                f"campaign cgroup {field}",
            )
    if any(
        snapshot[field] != 0
        for snapshot in (before, after)
        for field in ("memory_swap_current_bytes", "memory_swap_peak_bytes")
    ):
        raise RuntimeError("campaign cgroup swap gate failed")
    before_peak = cast(int, before["memory_peak_bytes"])
    after_peak = cast(int, after["memory_peak_bytes"])
    if after_peak < before_peak:
        raise RuntimeError("campaign cgroup peak memory is not monotonic")
    event_deltas = wave.get("cgroup_memory_event_deltas")
    if event_deltas != {"oom": 0, "oom_kill": 0}:
        raise RuntimeError("campaign cgroup OOM gate failed")
    io_fields = {
        "read_bytes",
        "write_bytes",
        "read_operations",
        "write_operations",
        "discard_bytes",
        "discard_operations",
    }
    reported_io = wave.get(
        "cgroup_io_deltas"
        if comparison_schema == LEGACY_PROFILE_COMPARISON_SCHEMA_VERSION
        else "io_accounting"
    )
    if not isinstance(reported_io, Mapping):
        raise RuntimeError("campaign I/O evidence is invalid")
    source = (
        CGROUP_IO_ACCOUNTING_SOURCE
        if comparison_schema == LEGACY_PROFILE_COMPARISON_SCHEMA_VERSION
        else reported_io.get("source")
    )
    before_io = before.get("io")
    after_io = after.get("io")
    cgroup_replayable = (
        isinstance(before_io, Mapping)
        and isinstance(after_io, Mapping)
        and set(before_io) == io_fields
        and set(after_io) == io_fields
    )
    if cgroup_replayable:
        if source != CGROUP_IO_ACCOUNTING_SOURCE:
            raise RuntimeError("campaign I/O provider did not prefer cgroup")
        replay_before_io = cast(Mapping[str, object], before_io)
        replay_after_io = cast(Mapping[str, object], after_io)
        if (
            comparison_schema == LEGACY_PROFILE_COMPARISON_SCHEMA_VERSION
            and set(reported_io) != io_fields
        ) or (
            comparison_schema
            in {
                PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION,
                PROCESS_PROFILE_COMPARISON_SCHEMA_VERSION,
                TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
                SCHEMA_VERSION,
            }
            and set(reported_io) != {"source", *io_fields}
        ):
            raise RuntimeError("campaign cgroup I/O evidence is invalid")
        for field in io_fields:
            before_value = _review_nonnegative_integer(
                replay_before_io[field],
                f"campaign cgroup I/O {field}",
            )
            after_value = _review_nonnegative_integer(
                replay_after_io[field],
                f"campaign cgroup I/O {field}",
            )
            if after_value < before_value or reported_io[field] != after_value - before_value:
                raise RuntimeError("campaign cgroup I/O deltas do not replay")
    elif comparison_schema == LEGACY_PROFILE_COMPARISON_SCHEMA_VERSION:
        raise RuntimeError("campaign legacy cgroup I/O evidence is invalid")
    elif before_io != "unavailable" or after_io != "unavailable":
        raise RuntimeError("campaign cgroup I/O availability is inconsistent")
    elif source == PROCESS_TREE_IO_ACCOUNTING_SOURCE:
        byte_fields = {"read_bytes", "write_bytes"}
        if set(reported_io) != {"source", *byte_fields}:
            raise RuntimeError("campaign process-tree I/O evidence is invalid")
        for field, expected in _review_process_tree_io(wave).items():
            if reported_io[field] != expected:
                raise RuntimeError("campaign process-tree I/O does not replay")
    else:
        raise RuntimeError("campaign I/O accounting source is invalid")
    memory_gate = _review_nonnegative_integer(
        wave.get("memory_gate_bytes"),
        "campaign memory gate",
    )
    aggregate_rss = _review_nonnegative_integer(
        wave.get("peak_aggregate_rss_bytes"),
        "campaign aggregate RSS",
    )
    aggregate_pss = _review_nonnegative_integer(
        wave.get("peak_aggregate_pss_bytes"),
        "campaign aggregate PSS",
    )
    if (
        memory_gate == 0
        or max(
            cast(int, before["memory_current_bytes"]),
            cast(int, after["memory_current_bytes"]),
            after_peak,
            aggregate_rss,
            aggregate_pss,
        )
        > memory_gate
    ):
        raise RuntimeError("campaign memory safety gate failed")
    elapsed = _review_nonnegative_number(
        wave.get("elapsed_seconds"),
        "campaign mode-wave elapsed time",
    )
    compute_limit = _review_nonnegative_integer(
        wave.get("compute_thread_limit"),
        "campaign mode-wave compute limit",
    )
    cpu_seconds = _review_nonnegative_number(
        wave.get("process_tree_cpu_seconds"),
        "campaign process-tree CPU time",
    )
    if elapsed <= 0.0 or compute_limit == 0:
        raise RuntimeError("campaign process-tree CPU accounting identity is invalid")
    if comparison_schema == SCHEMA_VERSION:
        _review_current_cpu_budget(
            wave,
            elapsed_seconds=elapsed,
            compute_limit=compute_limit,
            cpu_seconds=cpu_seconds,
            label="campaign process-tree",
        )
        derived_cpu_seconds = min(cpu_seconds, elapsed * compute_limit)
    else:
        if cpu_seconds > elapsed * compute_limit * 1.000001:
            raise RuntimeError("campaign process-tree CPU accounting exceeds its budget")
        derived_cpu_seconds = cpu_seconds
    axis_count = _review_nonnegative_integer(
        wave.get("axis_count"),
        "campaign mode-wave axis count",
    )
    axes_per_hour = _review_nonnegative_number(
        wave.get("axes_per_hour"),
        "campaign mode-wave axes per hour",
    )
    effective_cores = _review_nonnegative_number(
        wave.get("effective_cores"),
        "campaign mode-wave effective cores",
    )
    utilization = _review_nonnegative_number(
        wave.get("cpu_utilization_fraction_of_compute_limit"),
        "campaign mode-wave compute-limit utilization",
    )
    if (
        axis_count == 0
        or not math.isclose(
            axes_per_hour,
            3600.0 * axis_count / elapsed,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            effective_cores,
            derived_cpu_seconds / elapsed,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            utilization,
            derived_cpu_seconds / (elapsed * compute_limit),
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or utilization > 1.000001
    ):
        raise RuntimeError("campaign derived CPU/throughput metrics do not replay")
    thread_tree = wave.get("thread_tree")
    thread_metrics = wave.get("thread_metrics")
    thread_schedstat = wave.get("thread_tree_schedstat")
    if (
        wave.get("thread_tree_status") != "available"
        or not isinstance(thread_tree, Mapping)
        or thread_tree.get("status") != "available"
        or thread_tree.get("observation_capacity") != 4096
        or not isinstance(thread_metrics, list)
        or not thread_metrics
        or thread_tree.get("observed_thread_identities") != len(thread_metrics)
        or not isinstance(thread_schedstat, Mapping)
    ):
        raise RuntimeError("campaign thread-tree resource evidence is incomplete")
    unresolved = thread_tree.get("unresolved_thread_ids")
    missed_processes = _review_nonnegative_integer(
        thread_tree.get("sample_missed_processes"),
        "campaign missed thread-process samples",
    )
    unresolved_count = _review_nonnegative_integer(
        thread_tree.get("unresolved_thread_observation_count"),
        "campaign unresolved thread count",
    )
    if not isinstance(unresolved, list) or unresolved_count != len(unresolved):
        raise RuntimeError("campaign thread-tree sampling evidence is incomplete")
    unresolved_identities: set[tuple[int, float, int]] = set()
    for index, raw_unresolved in enumerate(unresolved):
        if not isinstance(raw_unresolved, Mapping) or set(raw_unresolved) != {
            "pid",
            "process_create_time",
            "tid",
        }:
            raise RuntimeError("campaign unresolved thread schema is invalid")
        unresolved_identity = (
            _review_nonnegative_integer(
                raw_unresolved.get("pid"),
                f"campaign unresolved thread[{index}].pid",
            ),
            _review_nonnegative_number(
                raw_unresolved.get("process_create_time"),
                f"campaign unresolved thread[{index}].create_time",
            ),
            _review_nonnegative_integer(
                raw_unresolved.get("tid"),
                f"campaign unresolved thread[{index}].tid",
            ),
        )
        if (
            min(unresolved_identity[0], unresolved_identity[2]) == 0
            or unresolved_identity in unresolved_identities
        ):
            raise RuntimeError("campaign unresolved thread identity is invalid")
        unresolved_identities.add(unresolved_identity)
    if comparison_schema not in {
        PROCESS_PROFILE_COMPARISON_SCHEMA_VERSION,
        TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
        SCHEMA_VERSION,
    } and (missed_processes or unresolved):
        raise RuntimeError("campaign thread-tree sampling evidence is incomplete")
    start_ticks = wave.get("monitor_start_boot_time_ticks")
    _review_nonnegative_integer(start_ticks, "campaign monitor boot-time tick marker")
    counter_names = (
        "voluntary_context_switches",
        "involuntary_context_switches",
        "minor_faults",
        "major_faults",
        "schedstat_runtime_ns",
        "schedstat_runqueue_delay_ns",
        "schedstat_timeslices",
        "cpu_migrations",
    )
    counter_totals = {name: 0 for name in counter_names}
    user_total = 0.0
    system_total = 0.0
    affinity_union: set[int] = set()
    affinity_intersection: set[int] | None = None
    identities: set[tuple[int, float, int, int]] = set()
    expected_row_fields = {
        "pid",
        "process_create_time",
        "tid",
        "thread_start_time_ticks",
        "sample_count",
        "cpu_baseline_source",
        "user_cpu_seconds",
        "system_cpu_seconds",
        "counters",
        "last_affinity",
    }
    for ordinal, raw_row in enumerate(thread_metrics):
        if not isinstance(raw_row, Mapping) or set(raw_row) != expected_row_fields:
            raise RuntimeError("campaign thread-tree row schema is invalid")
        pid = _review_nonnegative_integer(raw_row.get("pid"), f"thread row {ordinal} PID")
        tid = _review_nonnegative_integer(raw_row.get("tid"), f"thread row {ordinal} TID")
        thread_start = _review_nonnegative_integer(
            raw_row.get("thread_start_time_ticks"),
            f"thread row {ordinal} start ticks",
        )
        sample_count = _review_nonnegative_integer(
            raw_row.get("sample_count"),
            f"thread row {ordinal} sample count",
        )
        create_time = _review_nonnegative_number(
            raw_row.get("process_create_time"),
            f"thread row {ordinal} process create time",
        )
        if pid == 0 or tid == 0 or sample_count == 0 or create_time == 0.0:
            raise RuntimeError("campaign thread-tree row identity is invalid")
        identity = (pid, create_time, tid, thread_start)
        if identity in identities:
            raise RuntimeError("campaign thread-tree identity is duplicated")
        identities.add(identity)
        baseline_source = raw_row.get("cpu_baseline_source")
        if baseline_source not in {"monitor_start", "thread_start"}:
            raise RuntimeError("campaign thread-tree CPU baseline source is invalid")
        user_total += _review_nonnegative_number(
            raw_row.get("user_cpu_seconds"),
            f"thread row {ordinal} user CPU",
        )
        system_total += _review_nonnegative_number(
            raw_row.get("system_cpu_seconds"),
            f"thread row {ordinal} system CPU",
        )
        counters = raw_row.get("counters")
        if not isinstance(counters, Mapping) or set(counters) != set(counter_names):
            raise RuntimeError("campaign thread-tree row counters are invalid")
        for name in counter_names:
            counter_totals[name] += _review_nonnegative_integer(
                counters.get(name),
                f"thread row {ordinal} {name}",
            )
        affinity = raw_row.get("last_affinity")
        if (
            not isinstance(affinity, list)
            or not affinity
            or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in affinity)
            or affinity != sorted(set(affinity))
        ):
            raise RuntimeError("campaign thread-tree row affinity is invalid")
        affinity_set = set(affinity)
        affinity_union.update(affinity_set)
        if affinity_intersection is None:
            affinity_intersection = affinity_set
        else:
            affinity_intersection.intersection_update(affinity_set)
    reported_counters = thread_tree.get("counters")
    if (
        not isinstance(reported_counters, Mapping)
        or dict(reported_counters) != counter_totals
        or thread_tree.get("affinity_union") != sorted(affinity_union)
        or thread_tree.get("affinity_intersection") != sorted(affinity_intersection or set())
        or wave.get("thread_affinity_union") != sorted(affinity_union)
        or wave.get("thread_affinity_intersection") != sorted(affinity_intersection or set())
        or not math.isclose(
            _review_nonnegative_number(
                thread_tree.get("user_cpu_seconds"),
                "campaign thread-tree user CPU",
            ),
            user_total,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            _review_nonnegative_number(
                thread_tree.get("system_cpu_seconds"),
                "campaign thread-tree system CPU",
            ),
            system_total,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise RuntimeError("campaign thread-tree aggregates do not replay")
    expected_schedstat = {
        "runtime_ns": counter_totals["schedstat_runtime_ns"],
        "runqueue_delay_ns": counter_totals["schedstat_runqueue_delay_ns"],
        "timeslices": counter_totals["schedstat_timeslices"],
    }
    expected_context_switches = (
        counter_totals["voluntary_context_switches"]
        + counter_totals["involuntary_context_switches"]
    )
    if (
        dict(thread_schedstat) != expected_schedstat
        or wave.get("thread_tree_user_cpu_seconds") != user_total
        or wave.get("thread_tree_system_cpu_seconds") != system_total
        or wave.get("thread_tree_context_switches") != expected_context_switches
        or wave.get("thread_tree_cpu_migrations") != counter_totals["cpu_migrations"]
        or wave.get("thread_tree_minor_faults") != counter_totals["minor_faults"]
        or wave.get("thread_tree_major_faults") != counter_totals["major_faults"]
    ):
        raise RuntimeError("campaign thread-tree aliases do not replay")


def load_records(
    scope: str,
    *,
    attempt: int,
    results_root: Path,
) -> tuple[ReviewRecord, ...]:
    labels = run_labels_for_scope(scope, attempt)
    records: list[ReviewRecord] = []
    common_identity: tuple[str, str, str, str, str, str, str] | None = None
    for mode in MODES:
        mode_shared_review_started = time.perf_counter()
        run_dir = results_root / labels[mode.value]
        manifest = _verify_signed_json(run_dir / "run_manifest.json")
        if _string(manifest, "schema_version") in EXTERNAL_SEMANTIC_COMPARISON_SCHEMA_VERSIONS:
            git_tree, source_manifest_sha256 = _review_build_attestation(manifest)
            topology = _mapping(manifest, "topology")
            mode_waves = topology.get("mode_wave_resources")
            mode_waves_for_count = mode_waves if isinstance(mode_waves, list) else []
            scheduler_observed = topology.get("scheduler_observed")
            manifest_schema = _string(manifest, "schema_version")
            expected_scheduler_observations = (
                sum(
                    len(wave.get("scheduler_process_ids", []))
                    for wave in mode_waves_for_count
                    if isinstance(wave, dict)
                    and wave.get("mode") == ArchitectureMode.HOST_SCHEDULER.value
                    and isinstance(wave.get("scheduler_process_ids"), list)
                )
                if manifest_schema in PROFILE_COMPARISON_SCHEMA_VERSIONS
                else sum(
                    isinstance(wave, dict)
                    and wave.get("mode") == ArchitectureMode.HOST_SCHEDULER.value
                    for wave in mode_waves_for_count
                )
            )
            if (
                not isinstance(mode_waves, list)
                or not mode_waves
                or not isinstance(scheduler_observed, list)
                or len(scheduler_observed) != expected_scheduler_observations
            ):
                raise RuntimeError("campaign process-tree instrumentation is incomplete")
            for field in ("scheduler_startup_seconds", "scheduler_shutdown_seconds"):
                value = topology.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise RuntimeError(f"campaign {field} is missing or invalid")
            for wave in mode_waves:
                if not isinstance(wave, dict) or any(
                    field not in wave
                    for field in (
                        "mode",
                        "elapsed_seconds",
                        "process_tree_cpu_seconds",
                        "peak_aggregate_rss_bytes",
                        "peak_aggregate_pss_bytes",
                        "scheduler_process_id",
                        "scheduler_startup_seconds",
                        "scheduler_shutdown_seconds",
                    )
                ):
                    raise RuntimeError("campaign mode-wave resource evidence is incomplete")
                scheduler_process_id = wave["scheduler_process_id"]
                scheduler_process_ids = wave.get("scheduler_process_ids")
                if manifest_schema in PROFILE_COMPARISON_SCHEMA_VERSIONS:
                    valid_host_scheduler = (
                        isinstance(scheduler_process_ids, list)
                        and bool(scheduler_process_ids)
                        and all(
                            type(process_id) is int and process_id > 0
                            for process_id in scheduler_process_ids
                        )
                        and scheduler_process_id
                        == (scheduler_process_ids[0] if len(scheduler_process_ids) == 1 else None)
                    )
                    valid_non_scheduler = (
                        scheduler_process_id is None and scheduler_process_ids == []
                    )
                else:
                    valid_host_scheduler = (
                        type(scheduler_process_id) is int and scheduler_process_id > 0
                    )
                    valid_non_scheduler = scheduler_process_id is None
                if (
                    wave["mode"] == ArchitectureMode.HOST_SCHEDULER.value
                    and not valid_host_scheduler
                ) or (
                    wave["mode"] != ArchitectureMode.HOST_SCHEDULER.value
                    and not valid_non_scheduler
                ):
                    raise RuntimeError("host scheduler exists outside its exclusive mode wave")
                if manifest_schema in PROFILE_COMPARISON_SCHEMA_VERSIONS:
                    _review_mode_wave_resources(wave, comparison_schema=manifest_schema)
            if manifest_schema in PROFILE_COMPARISON_SCHEMA_VERSIONS:
                try:
                    performance_profile = FrozenPerformanceProfile.from_dict(
                        _mapping(manifest, "performance_profile")
                    )
                except ValueError as error:
                    raise RuntimeError("campaign performance profile is invalid") from error
                allowed_cpu_ids = list(performance_profile.host.allowed_cpu_ids)
                host_waves_by_block: dict[int, Mapping[str, object]] = {}
                for wave in mode_waves:
                    assert isinstance(wave, dict)
                    mode_name = wave.get("mode")
                    workload_class = wave.get("workload_class")
                    topology_key = wave.get("performance_topology_key")
                    expected_key = f"{mode_name}:{workload_class}"
                    if (
                        topology_key != expected_key
                        or topology_key not in performance_profile.topologies
                        or wave.get("performance_profile_sha256")
                        != performance_profile.canonical_sha256
                        or wave.get("execution_topology")
                        != performance_profile.topologies[topology_key].to_dict()
                        or wave.get("compute_thread_limit") != len(allowed_cpu_ids)
                        or wave.get("cpu_normalized_within_limit") is not True
                        or not _mode_wave_affinity_matches_profile(
                            wave,
                            comparison_schema=manifest_schema,
                            allowed_cpu_ids=allowed_cpu_ids,
                        )
                    ):
                        raise RuntimeError("campaign mode-wave performance profile does not replay")
                    utilization = wave.get("cpu_utilization_percent_of_compute_limit")
                    if (
                        isinstance(utilization, bool)
                        or not isinstance(utilization, int | float)
                        or not 0.0 <= float(utilization) <= 100.000001
                    ):
                        raise RuntimeError("campaign mode-wave CPU utilization exceeds its budget")
                    if mode_name == ArchitectureMode.HOST_SCHEDULER.value:
                        block_index = wave.get("identity_block_index")
                        if type(block_index) is not int:
                            raise RuntimeError("host scheduler identity block is invalid")
                        host_waves_by_block[block_index] = wave
                        runtime_statistics = wave.get("scheduler_runtime_statistics")
                        if not isinstance(runtime_statistics, list) or len(
                            runtime_statistics
                        ) != len(cast(list[object], wave["scheduler_process_ids"])):
                            raise RuntimeError("scheduler runtime statistics count is invalid")
                        for scheduler_statistics in runtime_statistics:
                            _review_scheduler_runtime_statistics(
                                scheduler_statistics,
                                worker_threads=len(
                                    performance_profile.topologies[topology_key].scheduler_cpu_ids
                                ),
                                request_threads=performance_profile.topologies[
                                    topology_key
                                ].request_threads,
                                evidence_root=(
                                    results_root / labels[ArchitectureMode.HOST_SCHEDULER.value]
                                ),
                            )
                    elif wave.get("scheduler_runtime_statistics") != []:
                        raise RuntimeError("scheduler runtime statistics exist outside host mode")
                for observation in scheduler_observed:
                    if not isinstance(observation, dict):
                        raise RuntimeError("scheduler observation is invalid")
                    block_index = observation.get("identity_block_index")
                    if type(block_index) is not int or block_index not in host_waves_by_block:
                        raise RuntimeError("scheduler observation has no host wave")
                    host_wave = host_waves_by_block[block_index]
                    topology_key = host_wave["performance_topology_key"]
                    assert isinstance(topology_key, str)
                    selected_topology = performance_profile.topologies[topology_key]
                    expected_affinity = list(selected_topology.scheduler_cpu_ids)
                    configured_workers = observation.get("configured_worker_threads")
                    configured_requests = observation.get("configured_request_threads")
                    if (
                        type(configured_workers) is not int
                        or type(configured_requests) is not int
                        or configured_workers != len(expected_affinity)
                        or configured_requests != selected_topology.request_threads
                        or observation.get("observed_request_threads") != configured_requests
                        or observation.get("observed_receipt_writer_threads") != 1
                        or observation.get("observed_thread_count")
                        != configured_workers + configured_requests + 2
                        or observation.get("configured_cpu_affinity") != expected_affinity
                        or observation.get("observed_cpu_affinity") != expected_affinity
                        or observation.get("runtime_statistics")
                        not in cast(
                            list[object],
                            host_wave["scheduler_runtime_statistics"],
                        )
                    ):
                        raise RuntimeError("scheduler topology observation does not replay")
        manifest_schema = _string(manifest, "schema_version")
        if manifest_schema not in EXTERNAL_SEMANTIC_COMPARISON_SCHEMA_VERSIONS:
            git_tree = "legacy-unattested"
            source_manifest_sha256 = "legacy-unattested"
        scheduler_identity = (
            _string(manifest, "scheduler_sha256")
            if manifest_schema in EXTERNAL_SEMANTIC_COMPARISON_SCHEMA_VERSIONS
            else str(manifest.get("scheduler_sha256") or "legacy-unattested")
        )
        identity = (
            manifest_schema,
            _string(manifest, "revision"),
            _string(manifest, "wheel_sha256"),
            _string(manifest, "native_sha256"),
            scheduler_identity,
            git_tree,
            source_manifest_sha256,
        )
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise RuntimeError(
                "comparison modes do not share one schema/commit/wheel/native identity"
            )
        paths = sorted((run_dir / "axes").rglob("*.json"))
        mode_wave_by_key: dict[tuple[int, str, str, int], Mapping[str, object]] = {}
        if manifest_schema in PROFILE_COMPARISON_SCHEMA_VERSIONS:
            if not isinstance(mode_waves, list):
                raise RuntimeError("campaign mode-wave resource evidence is missing")
            for raw_wave in mode_waves:
                assert isinstance(raw_wave, dict)
                if raw_wave.get("mode") != mode.value:
                    continue
                identities = raw_wave.get("identities")
                family = raw_wave.get("performance_family")
                if (
                    not isinstance(identities, list)
                    or not identities
                    or raw_wave.get("axis_count") != len(identities)
                    or family not in {"C5", "C", "R", "RC"}
                ):
                    raise RuntimeError("campaign mode-wave identity inventory is invalid")
                wave_keys: set[tuple[int, str, str, int]] = set()
                for raw_identity in identities:
                    if not isinstance(raw_identity, Mapping):
                        raise RuntimeError("campaign mode-wave axis identity is invalid")
                    try:
                        key = (
                            _integer(raw_identity, "repeat"),
                            _string(raw_identity, "axis"),
                            _string(raw_identity, "instance"),
                            _integer(raw_identity, "seed"),
                        )
                    except ValueError as error:
                        raise RuntimeError("campaign mode-wave axis identity is invalid") from error
                    expected_family = "C5" if not key[2].endswith("_21") else _family(key[2])
                    if family != expected_family or key in wave_keys or key in mode_wave_by_key:
                        raise RuntimeError("campaign mode-wave axis ownership is invalid")
                    wave_keys.add(key)
                raw_timings = raw_wave.get("axis_parent_terminal_timings")
                if not isinstance(raw_timings, list) or len(raw_timings) != len(wave_keys):
                    raise RuntimeError("campaign mode-wave axis timing inventory is invalid")
                timing_keys: set[tuple[int, str, str, int]] = set()
                for raw_timing in raw_timings:
                    if not isinstance(raw_timing, Mapping) or set(raw_timing) != {
                        "repeat",
                        "axis",
                        "instance",
                        "seed",
                        "subwave_index",
                        "producer_parent_terminal_seconds",
                    }:
                        raise RuntimeError("campaign mode-wave axis timing row is invalid")
                    try:
                        timing_key = (
                            _integer(raw_timing, "repeat"),
                            _string(raw_timing, "axis"),
                            _string(raw_timing, "instance"),
                            _integer(raw_timing, "seed"),
                        )
                        subwave_index = _integer(raw_timing, "subwave_index")
                        parent_terminal = _number(
                            raw_timing,
                            "producer_parent_terminal_seconds",
                        )
                    except ValueError as error:
                        raise RuntimeError(
                            "campaign mode-wave axis timing row is invalid"
                        ) from error
                    if subwave_index < 0 or parent_terminal <= 0.0 or timing_key in timing_keys:
                        raise RuntimeError("campaign mode-wave axis timing row is invalid")
                    timing_keys.add(timing_key)
                if timing_keys != wave_keys:
                    raise RuntimeError("campaign mode-wave axis timing ownership is invalid")
                for key in wave_keys:
                    mode_wave_by_key[key] = raw_wave
        mode_records_list: list[ReviewRecord] = []
        expected_run_label = labels[mode.value]
        for path in paths:
            full_payload = _verify_signed_json(path)
            record = ReviewRecord(path, full_payload)
            payload_scheduler_identity = (
                _string(record.payload, "scheduler_sha256")
                if manifest_schema in EXTERNAL_SEMANTIC_COMPARISON_SCHEMA_VERSIONS
                else str(record.payload.get("scheduler_sha256") or "legacy-unattested")
            )
            payload_identity = (
                _string(record.payload, "revision"),
                _string(record.payload, "wheel_sha256"),
                _string(record.payload, "native_sha256"),
                payload_scheduler_identity,
            )
            if _string(record.payload, "schema_version") != manifest_schema:
                raise RuntimeError(f"axis schema does not match its run manifest: {record.path}")
            if payload_identity != identity[1:5]:
                raise RuntimeError(f"axis identity does not match its run manifest: {record.path}")
            if _string(record.payload, "run_label") != expected_run_label:
                raise RuntimeError(f"axis run label mismatch: {record.path}")
            if record.mode is not mode:
                raise RuntimeError(f"axis mode identity mismatch for {mode.value}")
            loaded = _load_axis_record(path, verified_payload=full_payload)
            wave = mode_wave_by_key.get(loaded.key)
            if manifest_schema in PROFILE_COMPARISON_SCHEMA_VERSIONS and wave is None:
                raise RuntimeError("campaign axis has no mode-wave ownership")
            mode_records_list.append(ReviewRecord(loaded.path, loaded.payload, wave))
        mode_records = tuple(mode_records_list)
        keys = {record.key for record in mode_records}
        if keys != _expected_keys(scope) or len(mode_records) != len(keys):
            raise RuntimeError(f"axis identity set is incomplete or duplicated for {mode.value}")
        if manifest_schema in PROFILE_COMPARISON_SCHEMA_VERSIONS:
            unique_waves = {
                id(record.mode_wave): record.mode_wave
                for record in mode_records
                if record.mode_wave is not None
            }
            if not unique_waves:
                raise RuntimeError("current comparison mode has no shared review evidence")
            shared_seconds = (time.perf_counter() - mode_shared_review_started) / len(unique_waves)
            timed_waves = {
                identity: {
                    **wave,
                    "independent_shared_replay_seconds": shared_seconds,
                }
                for identity, wave in unique_waves.items()
            }
            mode_records = tuple(
                ReviewRecord(
                    record.path,
                    record.payload,
                    timed_waves[id(record.mode_wave)],
                )
                for record in mode_records
            )
        records.extend(mode_records)
    if len(records) != expected_axis_count(scope):
        raise RuntimeError("comparison record count does not match the fixed protocol")
    return tuple(records)


def _candidate_hash_json_value(value: object) -> object:
    """Recover Candidate Control's stable JSON projection from journal JSON."""

    if isinstance(value, dict):
        if set(value) == {"nonfinite_float"}:
            marker = value["nonfinite_float"]
            return {
                "nan": "NaN",
                "positive_inf": "+Infinity",
                "negative_inf": "-Infinity",
            }.get(marker, marker)
        return {str(key): _candidate_hash_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_candidate_hash_json_value(item) for item in value]
    return value


class _StreamingCandidateHash:
    def __init__(self) -> None:
        self._digest = hashlib.sha256(b"[")
        self.count = 0

    def append(self, value: object) -> None:
        if self.count:
            self._digest.update(b",")
        self._digest.update(
            json.dumps(
                _candidate_hash_json_value(value),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        self.count += 1

    def hexdigest(self) -> str:
        digest = self._digest.copy()
        digest.update(b"]")
        return digest.hexdigest()


class _RowEvidenceAccumulator:
    """Replay the producer's ordered row-evidence receipt without buffering rows."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256(b"stage05.2-row-evidence-v1\0")
        self.count = 0

    def append(self, value: object) -> None:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        self._digest.update(len(encoded).to_bytes(8, "little"))
        self._digest.update(encoded)
        self.count += 1

    def receipt(self) -> dict[str, object]:
        return {"count": self.count, "sha256": self._digest.hexdigest()}


def _trajectory_row(event: Mapping[str, object], ordinal: int) -> dict[str, object]:
    metadata = {
        "runtime_event_id",
        "semantic_event_id",
        "semantic_sequence",
        "semantic_stream",
        "stream_ordinal",
        "native_telemetry",
    }
    row = {
        key: value
        for key, value in event.items()
        if key not in metadata and not str(key).startswith("_")
    }
    if row.get("aggregate_count") == 1:
        row.pop("aggregate_count")
    if not row.get("candidate_pool_hash"):
        row.pop("candidate_pool_hash", None)
    operator = str(row.get("operator", ""))
    track = str(row.get("track", ""))
    quality_operators = {
        "relocate",
        "swap",
        "two_opt_star",
        "route_segment_destroy",
        "ejection_chain",
    }
    lane = (
        "constraint_lane"
        if track == "constraint_lane"
        else "quality_shadow"
        if operator in quality_operators
        else "legacy"
    )
    identity = {
        "lane": lane,
        "iteration": row.get("iteration"),
        "operator": operator,
        "status": row.get("status"),
        "candidate_route_sequences": row.get("candidate_route_sequences", []),
        "candidate_objective_key": row.get("candidate_objective_key", []),
        "ordinal": ordinal,
    }
    candidate_id = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {**row, "lane": lane, "candidate_id": candidate_id}


def _external_semantic_trajectory(
    axis_path: Path,
    payload: Mapping[str, object],
    descriptor: Mapping[str, object],
) -> list[dict[str, object]]:
    """Rebuild the candidate trajectory from the one canonical journal copy."""

    if set(descriptor) != {"schema_version", "source", "count", "sha256"} or (
        descriptor.get("schema_version") != "stage05.2-external-semantic-trajectory-v1"
        or descriptor.get("source") != "canonical_semantic_journal:operator"
    ):
        raise RuntimeError("external semantic trajectory descriptor is invalid")
    journal = payload.get("canonical_semantic_journal")
    if not isinstance(journal, Mapping):
        raise RuntimeError("external semantic trajectory journal is missing")
    rows: list[dict[str, object]] = []
    evidence = _RowEvidenceAccumulator()
    try:
        for event in iter_verified_semantic_journal(axis_path, journal):
            if (
                event.get("semantic_stream") != "operator"
                or event.get("status") in SEMANTIC_TRAJECTORY_IMPLEMENTATION_STATUSES
            ):
                continue
            row = _trajectory_row(event, len(rows))
            rows.append(row)
            evidence.append(row)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError("external semantic trajectory replay failed") from error
    if evidence.receipt() != {
        "count": descriptor.get("count"),
        "sha256": descriptor.get("sha256"),
    }:
        raise RuntimeError("external semantic trajectory digest does not reconcile")
    return rows


def _event_context(event: Mapping[str, object]) -> tuple[object, object, object]:
    return (event.get("lane"), event.get("iteration"), event.get("operator"))


def _context_sha256(event: Mapping[str, object]) -> str:
    lane, iteration, operator = _event_context(event)
    if not isinstance(lane, str) or not isinstance(operator, str):
        raise ValueError("journal event context is incomplete")
    return hashlib.sha256(
        json.dumps(
            {"iteration": iteration, "lane": lane, "operator": operator},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _route_sha256(sequence: object) -> str:
    if not isinstance(sequence, list) or not all(
        isinstance(customer, str) for customer in sequence
    ):
        raise ValueError("journal exact-work sequence is invalid")
    return hashlib.sha256(canonical_route_key(sequence).encode("utf-8")).hexdigest()


def _physical_expected_batches(
    axis_path: Path,
    descriptor: Mapping[str, object],
) -> Iterable[tuple[str, tuple[str, ...], int]]:
    exact_batches: deque[tuple[str, tuple[str, ...], int]] = deque()
    for event in iter_verified_semantic_journal(axis_path, descriptor):
        stream = event.get("semantic_stream")
        event_type = event.get("event_type")
        if event_type not in {"exact_batch_started", "candidate_batch_complete"}:
            continue
        sequences = event.get("customer_sequences")
        if not isinstance(sequences, list):
            raise ValueError("journal physical batch lacks customer sequences")
        iteration = event.get("iteration")
        if iteration is None:
            iteration_value = -1
        elif isinstance(iteration, int) and not isinstance(iteration, bool):
            iteration_value = iteration
        else:
            raise ValueError("journal exact-work iteration is invalid")
        identity = (
            _context_sha256(event),
            tuple(_route_sha256(sequence) for sequence in sequences),
            iteration_value,
        )
        if stream == "exact_work" and event_type == "exact_batch_started":
            exact_batches.append(identity)
            continue
        if stream != "candidate_transaction":
            continue
        while exact_batches and exact_batches[0] != identity:
            exact_batches.popleft()
        if not exact_batches:
            raise ValueError("physical batch has no preceding exact-work identity")
        exact_batches.popleft()
        yield identity


def _replay_physical_telemetry(
    payload: Mapping[str, object],
    *,
    axis_path: Path,
) -> str | None:
    descriptor = payload.get("canonical_semantic_journal")
    if not isinstance(descriptor, dict):
        return "physical telemetry semantic bundle descriptor is missing"
    physical = descriptor.get("physical_telemetry")
    if not isinstance(physical, dict):
        return "physical telemetry descriptor is missing"
    try:
        bundle = semantic_bundle_path(axis_path, descriptor)
        expected = _physical_expected_batches(axis_path, descriptor)
        observed = iter_verified_physical_telemetry(bundle, physical)
        count = 0
        sentinel = object()
        for expected_batch, observed_batch in zip_longest(
            expected,
            observed,
            fillvalue=sentinel,
        ):
            if expected_batch is sentinel or observed_batch is sentinel:
                return "physical telemetry batch count does not match parallel work"
            assert isinstance(expected_batch, tuple)
            assert isinstance(observed_batch, dict)
            context_sha256, route_sha256, iteration = expected_batch
            if (
                observed_batch.get("context_sha256") != context_sha256
                or observed_batch.get("route_key_sha256") != route_sha256
                or observed_batch.get("iteration") != iteration
            ):
                return f"physical telemetry batch {count} identity does not match parallel work"
            count += 1
        if physical.get("batch_count") != count:
            return "physical telemetry declared batch count does not replay"
    except (OSError, RuntimeError, ValueError) as error:
        return f"physical telemetry replay failed: {error}"
    return None


def _replay_measurement_evidence(
    payload: Mapping[str, object],
    *,
    axis_path: Path,
) -> str | None:
    """Rebuild semantic measurement receipts from the canonical journal."""

    descriptor = payload.get("canonical_semantic_journal")
    measurement = payload.get("measurement_evidence")
    if not isinstance(descriptor, dict) or not isinstance(measurement, dict):
        return "measurement evidence or semantic journal is missing"
    if measurement.get("present") is not True:
        return "measurement evidence unexpectedly declares no trace"
    exact_order = _RowEvidenceAccumulator()
    route_results = _RowEvidenceAccumulator()
    cache_lifecycle = _RowEvidenceAccumulator()
    deadline_boundaries = _RowEvidenceAccumulator()
    trajectory = _RowEvidenceAccumulator()
    canonical_routes: set[tuple[str, ...]] = set()
    committed_misses: set[str] = set()
    cache_stores = 0
    cache_evictions = 0
    cache_oversize = 0
    entries_current = 0
    entries_peak = 0
    bytes_current = 0
    bytes_peak = 0
    batch_ordinal = 0
    trajectory_ordinal = 0
    try:
        for event in iter_verified_semantic_journal(axis_path, descriptor):
            stream = event.get("semantic_stream")
            event_type = event.get("event_type")
            if stream == "exact_work" and event_type == "exact_batch_started":
                sequences = event.get("customer_sequences")
                if not isinstance(sequences, list) or not all(
                    isinstance(sequence, list)
                    and all(isinstance(customer, str) for customer in sequence)
                    for sequence in sequences
                ):
                    return "measurement exact-work sequence payload is invalid"
                exact_order.append(
                    {
                        "batch_ordinal": batch_ordinal,
                        "lane": event.get("lane"),
                        "iteration": event.get("iteration"),
                        "operator": event.get("operator"),
                        "sequences": sequences,
                    }
                )
                for route_ordinal, sequence in enumerate(sequences):
                    cache_lifecycle.append(
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
                    canonical_routes.add(tuple(sequence))
                batch_ordinal += 1
            elif stream == "candidate_transaction" and event_type == "candidate_route_result":
                sequences = event.get("customer_sequences")
                if (
                    not isinstance(sequences, list)
                    or len(sequences) != 2
                    or not isinstance(sequences[0], list)
                ):
                    return "measurement route-result sequence payload is invalid"
                route_results.append(
                    {
                        "sequence": sequences[0],
                        "result": {
                            field: event.get(field)
                            for field in (
                                "feasible",
                                "route",
                                "distance",
                                "total_energy",
                                "charged_energy",
                                "charging_time",
                                "labels_generated",
                                "labels_expanded",
                                "labels_pruned",
                                "failure_reason",
                            )
                        },
                    }
                )
            elif stream == "cache" and event_type == "cache_lookup_result":
                if event.get("cache_scope") == "committed" and event.get("status") == "miss":
                    route_key = event.get("route_key")
                    if not isinstance(route_key, str):
                        return "measurement cache miss route identity is invalid"
                    committed_misses.add(route_key)
            elif stream == "cache" and event_type == "cache_lifecycle":
                operation = event.get("status")
                cache_stores += int(operation == "store")
                cache_evictions += int(operation == "evict")
                cache_oversize += int(operation == "oversize_not_cached")
                raw_entries = event.get("current_entries")
                raw_bytes = event.get("current_bytes")
                if (
                    isinstance(raw_entries, bool)
                    or not isinstance(raw_entries, int)
                    or isinstance(raw_bytes, bool)
                    or not isinstance(raw_bytes, int)
                ):
                    return "measurement cache size telemetry is invalid"
                entries_current = raw_entries
                bytes_current = raw_bytes
                entries_peak = max(entries_peak, raw_entries)
                bytes_peak = max(bytes_peak, raw_bytes)
            elif stream == "exact_result" and event_type == "exact_route_result":
                if event.get("deadline_boundary"):
                    evaluation_id = event.get("evaluation_id")
                    if isinstance(evaluation_id, bool) or not isinstance(evaluation_id, int):
                        return "measurement deadline evaluation identity is invalid"
                    deadline_boundaries.append(
                        {
                            "evaluation_id": evaluation_id,
                            "route_key": event.get("route_key"),
                            "deadline_boundary": event.get("deadline_boundary"),
                            "exact_started": event.get("exact_started"),
                            "exact_completed": event.get("exact_completed"),
                            "status": event.get("status"),
                        }
                    )
            elif stream == "operator":
                if event.get("status") in SEMANTIC_TRAJECTORY_IMPLEMENTATION_STATUSES:
                    continue
                trajectory.append(_trajectory_row(event, trajectory_ordinal))
                trajectory_ordinal += 1
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return f"measurement journal replay failed: {error}"

    cache_lifecycle.append(
        {
            "transition": "final_cache_state",
            "cache_stores": cache_stores,
            "cache_evictions": cache_evictions,
            "cache_oversize_not_cached": cache_oversize,
            "entries_current": entries_current,
            "entries_peak": entries_peak,
            "bytes_current": bytes_current,
            "bytes_peak": bytes_peak,
            "unique_route_evaluations": len(committed_misses),
        }
    )
    route_dictionary = _RowEvidenceAccumulator()
    for route in sorted(canonical_routes):
        route_dictionary.append({"route": list(route)})
    expected = {
        "exact_route_order": exact_order.receipt(),
        "exact_route_results": route_results.receipt(),
        "cache_lifecycle": cache_lifecycle.receipt(),
        "deadline_boundaries": deadline_boundaries.receipt(),
        "route_dictionary": route_dictionary.receipt(),
        "events": trajectory.receipt(),
        "candidate_trajectory": trajectory.receipt(),
    }
    for field, receipt in expected.items():
        if measurement.get(field) != receipt:
            return f"measurement {field} does not replay from journal"
    return None


def _replay_raw_native_control_journal(
    payload: Mapping[str, object],
    *,
    axis_path: Path,
    instance: Instance,
) -> str | None:
    descriptor = payload.get("canonical_semantic_journal")
    native_statistics = payload.get("native_execution_statistics")
    raw_cache = payload.get("cache_incremental_statistics")
    raw_candidate_control = payload.get("candidate_control_statistics")
    if not isinstance(descriptor, dict) or not isinstance(native_statistics, dict):
        return "full-native control journal identity is missing"
    if not isinstance(raw_cache, dict) or not isinstance(raw_candidate_control, dict):
        return "full-native cache statistics are missing"
    raw_control_descriptor = descriptor.get("native_control_events")
    if not isinstance(raw_control_descriptor, dict) or raw_control_descriptor.get(
        "native_source_sha256"
    ) != native_statistics.get("control_journal_sha256"):
        return "full-native control source digest does not reconcile"
    cache_names = (
        "cache_lookups",
        "cache_hits",
        "cache_misses",
        "cache_stores",
        "cache_evictions",
        "cache_oversize_not_cached",
        "entries_current",
        "entries_peak",
        "bytes_current",
        "bytes_peak",
        "unique_route_evaluations",
    )
    expected_batch_ordinal = 0
    active_transaction: int | None = None
    lookup_count = 0
    hit_count = 0
    store_count = 0
    eviction_count = 0
    oversize_count = 0
    aggregate_count = 0
    transaction_reference_distance_count = 0
    transaction_reference_distance_skipped = False
    transaction_budget_skip_count = 0
    budget_skip_count = 0
    final_statistics: list[int] | None = None
    reference_exact_cache: dict[tuple[str, tuple[str, ...]], ChargingSubproblemResult] = {}
    current_control_schema = payload.get("schema_version") == SCHEMA_VERSION
    plan_decisions: dict[int, tuple[str, int, int, bool]] = {}
    candidate_misses: Counter[int] = Counter()
    candidate_exact_rows: Counter[int] = Counter()
    candidate_stores: set[int] = set()
    budget_skips: dict[int, tuple[int, int, int, int, int, bool]] = {}
    previous_budget_state: list[int] | None = None
    fixed_work = payload.get("fixed_work_budget")
    if current_control_schema:
        exact_budget = -1
        if payload.get("axis") == "fixed_work":
            if not isinstance(fixed_work, dict):
                return "full-native fixed-work budget is missing"
            raw_exact_budget = fixed_work.get("exact_calls")
            if (
                isinstance(raw_exact_budget, bool)
                or not isinstance(raw_exact_budget, int)
                or raw_exact_budget <= 0
            ):
                return "full-native fixed-work exact budget is invalid"
            exact_budget = raw_exact_budget
        round_budget = raw_candidate_control.get("max_exact_calls_per_round")
        if (
            isinstance(round_budget, bool)
            or not isinstance(round_budget, int)
            or round_budget <= 0
        ):
            return "full-native candidate round budget is invalid"
    try:
        for event in iter_verified_native_control_events(axis_path, descriptor):
            transaction_id = event.get("native_transaction_id")
            batch_ordinal = event.get("batch_ordinal")
            if (
                isinstance(transaction_id, bool)
                or not isinstance(transaction_id, int)
                or transaction_id < 0
                or isinstance(batch_ordinal, bool)
                or not isinstance(batch_ordinal, int)
                or batch_ordinal < 0
            ):
                return "native control transaction identity is invalid"
            if active_transaction is None:
                if batch_ordinal != expected_batch_ordinal:
                    return "native control batch ordinals are not contiguous"
                active_transaction = transaction_id
                lookup_count = hit_count = store_count = eviction_count = oversize_count = 0
                transaction_reference_distance_count = 0
                transaction_reference_distance_skipped = False
                transaction_budget_skip_count = 0
                plan_decisions = {}
                candidate_misses = Counter()
                candidate_exact_rows = Counter()
                candidate_stores = set()
                budget_skips = {}
            elif transaction_id != active_transaction or batch_ordinal != expected_batch_ordinal:
                return "native control transaction rows are not contiguous"
            event_type = event.get("event_type")
            if event_type == "cache_event":
                if event.get("operation") != "lookup" or event.get("result") not in {
                    "hit",
                    "miss",
                }:
                    return "native control cache lookup row is invalid"
                sequence = event.get("customer_sequence")
                if not isinstance(sequence, list) or not all(
                    isinstance(customer, str) for customer in sequence
                ):
                    return "native control cache lookup route is invalid"
                lookup_count += 1
                hit_count += int(event.get("result") == "hit")
                if current_control_schema:
                    candidate_id = event.get("candidate_id")
                    if (
                        isinstance(candidate_id, bool)
                        or not isinstance(candidate_id, int)
                        or candidate_id < 0
                    ):
                        return "native control cache lookup candidate is invalid"
                    if event.get("result") == "miss":
                        candidate_misses[candidate_id] += 1
            elif event_type == "cache_store_receipt":
                status = event.get("status")
                raw_evictions = event.get("eviction_count")
                entry_bytes = event.get("entry_bytes")
                if (
                    status not in {"store", "reconcile", "oversize_not_cached"}
                    or isinstance(raw_evictions, bool)
                    or not isinstance(raw_evictions, int)
                    or raw_evictions < 0
                    or isinstance(entry_bytes, bool)
                    or not isinstance(entry_bytes, int)
                    or entry_bytes <= 0
                    or status != "store"
                    and raw_evictions != 0
                ):
                    return "native control cache store receipt is invalid"
                store_count += int(status == "store")
                eviction_count += raw_evictions
                oversize_count += int(status == "oversize_not_cached")
                if current_control_schema:
                    candidate_id = event.get("candidate_id")
                    if (
                        isinstance(candidate_id, bool)
                        or not isinstance(candidate_id, int)
                        or candidate_id < 0
                    ):
                        return "native control cache store candidate is invalid"
                    candidate_stores.add(candidate_id)
                    candidate_exact_rows[candidate_id] += 1
            elif event_type == "candidate_cache_transaction":
                statistics_row = event.get("cache_statistics")
                raw_reference_distance = event.get("reference_distance_resolution")
                implementation_internal = event.get("implementation_internal")
                round_budget_suppressed = event.get("round_budget_suppressed")
                if (
                    event.get("status") != "committed"
                    or event.get("lookups") != lookup_count
                    or event.get("hits") != hit_count
                    or event.get("exact_stores") != store_count
                    or not isinstance(statistics_row, list)
                    or len(statistics_row) != len(cache_names)
                    or any(
                        isinstance(value, bool) or not isinstance(value, int) or value < 0
                        for value in statistics_row
                    )
                    or not isinstance(raw_reference_distance, bool)
                    or int(raw_reference_distance) != transaction_reference_distance_count
                    or not isinstance(implementation_internal, bool)
                    or current_control_schema
                    and not isinstance(round_budget_suppressed, bool)
                ):
                    return "native control cache transaction does not replay"
                if current_control_schema:
                    raw_budget_state = event.get("budget_state")
                    if (
                        not isinstance(raw_budget_state, list)
                        or len(raw_budget_state) != 9
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            for value in raw_budget_state
                        )
                    ):
                        return "native control budget state is invalid"
                    budget_state = cast(list[int], raw_budget_state)
                    round_active = budget_state[0] == 1
                    expected_round_remaining = (
                        max(0, cast(int, round_budget) - budget_state[3])
                        if round_active
                        else cast(int, round_budget)
                    )
                    expected_exact_exhausted = int(
                        exact_budget >= 0 and budget_state[5] >= exact_budget
                    )
                    if (
                        budget_state[0] not in {0, 1}
                        or budget_state[3] < 0
                        or budget_state[3] > cast(int, round_budget)
                        or budget_state[4] != expected_round_remaining
                        or budget_state[5] < 0
                        or (exact_budget >= 0 and budget_state[5] > exact_budget)
                        or budget_state[6] < 0
                        or budget_state[7] < 0
                        or budget_state[6] > budget_state[5]
                        or budget_state[7] > budget_state[5] - budget_state[6]
                        or budget_state[8] != expected_exact_exhausted
                        or (
                            round_active
                            and (budget_state[1] < 0 or budget_state[2] < 0)
                        )
                        or (
                            not round_active
                            and budget_state[1:4] != [-1, -1, 0]
                        )
                    ):
                        return "native control budget state does not match configured limits"
                    skipped_candidates = {
                        candidate_id
                        for candidate_id, (
                            _status,
                            transaction_status,
                            _rank,
                            _internal,
                        ) in plan_decisions.items()
                        if transaction_status == 3
                    }
                    if skipped_candidates != set(budget_skips):
                        return "native control budget skips do not bind selected plans"
                    if budget_skips and (
                        implementation_internal or raw_reference_distance
                    ):
                        return "implementation-internal transaction contains a budget skip"
                    if raw_reference_distance and (
                        not implementation_internal or plan_decisions or budget_skips
                    ):
                        return "native reference-distance transaction shape is invalid"
                    if implementation_internal and plan_decisions:
                        return "implementation-internal transaction contains plan decisions"
                    prior_started = (
                        0 if previous_budget_state is None else previous_budget_state[5]
                    )
                    prior_completed = (
                        0 if previous_budget_state is None else previous_budget_state[6]
                    )
                    prior_interrupted = (
                        0 if previous_budget_state is None else previous_budget_state[7]
                    )
                    transaction_exact_rows = sum(candidate_exact_rows.values())
                    if (
                        budget_state[5] != prior_started + transaction_exact_rows
                        or budget_state[6] != prior_completed + transaction_exact_rows
                        or budget_state[7] != prior_interrupted
                    ):
                        return "native control exact budget state does not replay"
                    transaction_iteration = event.get("iteration")
                    if not round_budget_suppressed and (
                        isinstance(transaction_iteration, bool)
                        or not isinstance(transaction_iteration, int)
                        or transaction_iteration < 0
                    ):
                        return "native control accounted round identity is invalid"
                    accounted_round = (
                        not round_budget_suppressed
                        and isinstance(transaction_iteration, int)
                        and not isinstance(transaction_iteration, bool)
                        and transaction_iteration >= 0
                    )
                    running_started = prior_started
                    running_round_used = 0
                    if (
                        accounted_round
                        and previous_budget_state is not None
                        and previous_budget_state[0] == 1
                        and previous_budget_state[2] == transaction_iteration
                    ):
                        running_round_used = previous_budget_state[3]
                    selected_plans = sorted(
                        (
                            (rank, candidate_id)
                            for candidate_id, (
                                status,
                                _transaction_status,
                                rank,
                                internal,
                            ) in plan_decisions.items()
                            if status == "selected" and internal == implementation_internal
                        )
                    )
                    for _rank, candidate_id in selected_plans:
                        receipt = budget_skips.get(candidate_id)
                        exact_rows = candidate_exact_rows[candidate_id]
                        if receipt is not None:
                            (
                                requested,
                                granted,
                                round_remaining,
                                exact_remaining,
                                available,
                                receipt_internal,
                            ) = receipt
                            expected_exact_remaining = (
                                -1
                                if exact_budget < 0
                                else max(0, exact_budget - running_started)
                            )
                            expected_round_remaining = max(
                                0, cast(int, round_budget) - running_round_used
                            )
                            expected_available = min(
                                expected_round_remaining,
                                requested
                                if expected_exact_remaining < 0
                                else expected_exact_remaining,
                            )
                            if (
                                plan_decisions.get(candidate_id)
                                != ("selected", 3, _rank, False)
                                or requested != candidate_misses[candidate_id]
                                or granted != 0
                                or round_remaining != expected_round_remaining
                                or exact_remaining != expected_exact_remaining
                                or available != expected_available
                                or requested <= available
                                or exact_rows != 0
                                or candidate_id in candidate_stores
                                or receipt_internal
                            ):
                                return "native control budget skip does not replay"
                        running_started += exact_rows
                        running_round_used += exact_rows
                    if implementation_internal and accounted_round:
                        running_round_used += transaction_exact_rows
                    if accounted_round and (
                        budget_state[0] != 1
                        or budget_state[2] != transaction_iteration
                        or budget_state[3] != running_round_used
                        or budget_state[4]
                        != max(0, cast(int, round_budget) - running_round_used)
                    ):
                        return "native control round budget state does not replay"
                    if not accounted_round:
                        if previous_budget_state is None:
                            expected_suppressed_round_state = [
                                0,
                                -1,
                                -1,
                                0,
                                cast(int, round_budget),
                            ]
                        else:
                            expected_suppressed_round_state = previous_budget_state[:5]
                        if budget_state[:5] != expected_suppressed_round_state:
                            return "suppressed native control round state changed"
                    if transaction_reference_distance_skipped:
                        exact_budget_exhausted = (
                            exact_budget >= 0 and budget_state[5] >= exact_budget
                        )
                        prior_round_exhausted = (
                            accounted_round
                            and previous_budget_state is not None
                            and previous_budget_state[4] == 0
                            and budget_state[:5] == previous_budget_state[:5]
                        )
                        if not exact_budget_exhausted and not prior_round_exhausted:
                            return "native reference-distance budget skip is not justified"
                    previous_budget_state = budget_state
                if transaction_budget_skip_count and raw_reference_distance:
                    return "reference-distance transaction contains a budget skip"
                assert all(isinstance(value, int) for value in statistics_row)
                statistics_values = cast(list[int], statistics_row)
                if (
                    statistics_values[0] < lookup_count
                    or statistics_values[1] < hit_count
                    or statistics_values[2] != statistics_values[0] - statistics_values[1]
                    or statistics_values[3] < store_count
                    or statistics_values[4] < eviction_count
                    or statistics_values[5] < oversize_count
                    or statistics_values[6] > statistics_values[7]
                    or statistics_values[8] > statistics_values[9]
                ):
                    return "native control cumulative cache statistics are invalid"
                if final_statistics is not None and any(
                    current < previous
                    for current, previous in zip(
                        statistics_values[:6] + statistics_values[7:8] + statistics_values[9:],
                        final_statistics[:6] + final_statistics[7:8] + final_statistics[9:],
                        strict=True,
                    )
                ):
                    return "native control cumulative cache statistics regressed"
                final_statistics = statistics_values
                aggregate_count += 1
                expected_batch_ordinal += 1
                active_transaction = None
            elif event_type == "reference_distance_resolution":
                sequence = event.get("customer_sequence")
                if not isinstance(sequence, list) or not all(
                    isinstance(customer, str) for customer in sequence
                ):
                    return "native reference-distance route is invalid"
                transaction_status_code = event.get("transaction_status_code")
                if event.get("status") == "budget_skipped":
                    if (
                        transaction_status_code != 3
                        or event.get("vehicle_count") != -1
                        or event.get("charging_count") != -1
                        or event.get("reference_distance") is not None
                        or event.get("reference_charging_time") is not None
                    ):
                        return "native skipped reference-distance receipt is invalid"
                    transaction_reference_distance_count += 1
                    transaction_reference_distance_skipped = True
                    continue
                customer_sequence = tuple(sequence)
                reference_key = (instance.name, customer_sequence)
                exact = reference_exact_cache.get(reference_key)
                if exact is None:
                    exact = solve_exact_charging(instance, customer_sequence)
                    reference_exact_cache[reference_key] = exact
                expected_status = "feasible" if exact.feasible else "infeasible"
                expected_transaction_status = 5 if exact.feasible else 4
                if (
                    event.get("status") != expected_status
                    or transaction_status_code != expected_transaction_status
                ):
                    return "native reference-distance feasibility does not replay"
                if exact.feasible:
                    charging_count = sum(
                        instance.by_name[node].kind == NodeType.STATION for node in exact.route
                    )
                    reference_distance = event.get("reference_distance")
                    reference_charging_time = event.get("reference_charging_time")
                    if (
                        event.get("vehicle_count") != 1
                        or event.get("charging_count") != charging_count
                        or isinstance(reference_distance, bool)
                        or not isinstance(reference_distance, int | float)
                        or isinstance(reference_charging_time, bool)
                        or not isinstance(reference_charging_time, int | float)
                        or not math.isclose(
                            float(reference_distance),
                            exact.distance,
                            rel_tol=1e-12,
                            abs_tol=1e-9,
                        )
                        or not math.isclose(
                            float(reference_charging_time),
                            exact.charging_time,
                            rel_tol=1e-12,
                            abs_tol=1e-9,
                        )
                    ):
                        return "native reference-distance objective does not replay"
                elif (
                    event.get("vehicle_count") != -1
                    or event.get("charging_count") != -1
                    or event.get("reference_distance") is not None
                    or event.get("reference_charging_time") is not None
                ):
                    return "native infeasible reference-distance receipt is invalid"
                transaction_reference_distance_count += 1
            elif event_type == "candidate_control_budget":
                raw_requested = event.get("requested")
                raw_granted = event.get("granted")
                raw_remaining = event.get("remaining")
                raw_round_remaining = event.get("round_remaining")
                raw_exact_remaining = event.get("exact_remaining")
                raw_available = event.get("available")
                raw_candidate_id = event.get("candidate_id")
                raw_transaction_status = event.get("transaction_status_code")
                raw_implementation_internal = event.get("implementation_internal")
                context = event.get("context")
                iteration = event.get("iteration")
                if (
                    event.get("status") != "budget_skipped"
                    or isinstance(raw_requested, bool)
                    or not isinstance(raw_requested, int)
                    or raw_requested <= 0
                    or isinstance(raw_granted, bool)
                    or not isinstance(raw_granted, int)
                    or raw_granted != 0
                    or isinstance(raw_remaining, bool)
                    or not isinstance(raw_remaining, int)
                    or raw_remaining < 0
                    or not isinstance(context, str)
                    or len(context.split(":")) != 3
                    or isinstance(iteration, bool)
                    or not isinstance(iteration, int)
                    or iteration < 0
                ):
                    return "native control budget-skip receipt is invalid"
                if current_control_schema:
                    if (
                        isinstance(raw_candidate_id, bool)
                        or not isinstance(raw_candidate_id, int)
                        or raw_candidate_id < 0
                        or raw_transaction_status != 3
                        or not isinstance(raw_implementation_internal, bool)
                        or raw_candidate_id in budget_skips
                        or isinstance(raw_round_remaining, bool)
                        or not isinstance(raw_round_remaining, int)
                        or raw_round_remaining != raw_remaining
                        or isinstance(raw_exact_remaining, bool)
                        or not isinstance(raw_exact_remaining, int)
                        or isinstance(raw_available, bool)
                        or not isinstance(raw_available, int)
                    ):
                        return "native control typed budget-skip receipt is invalid"
                    budget_skips[raw_candidate_id] = (
                        raw_requested,
                        raw_granted,
                        raw_remaining,
                        raw_exact_remaining,
                        raw_available,
                        raw_implementation_internal,
                    )
                budget_skip_count += int(raw_implementation_internal is not True)
                transaction_budget_skip_count += 1
            elif event_type == "candidate_plan_decision":
                if current_control_schema:
                    candidate_id = event.get("candidate_id")
                    transaction_status = event.get("transaction_status_code")
                    status = event.get("status")
                    implementation_internal = event.get("implementation_internal")
                    rank = event.get("rank")
                    if (
                        isinstance(candidate_id, bool)
                        or not isinstance(candidate_id, int)
                        or candidate_id < 0
                        or candidate_id in plan_decisions
                        or not isinstance(status, str)
                        or isinstance(transaction_status, bool)
                        or not isinstance(transaction_status, int)
                        or not isinstance(implementation_internal, bool)
                        or (
                            implementation_internal is False
                            and (
                                isinstance(rank, bool)
                                or not isinstance(rank, int)
                                or rank <= 0
                            )
                        )
                    ):
                        return "native control plan decision identity is invalid"
                    plan_decisions[candidate_id] = (
                        status,
                        transaction_status,
                        0 if implementation_internal else cast(int, rank),
                        implementation_internal,
                    )
            elif event_type != "candidate_initial_solution":
                return "native control journal event type is invalid"
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return f"native control journal replay failed: {error}"
    if active_transaction is not None or aggregate_count == 0 or final_statistics is None:
        return "native control journal has an incomplete transaction"
    for index, name in enumerate(cache_names):
        if raw_cache.get(name) != final_statistics[index]:
            return f"native control cache {name} does not match the axis"
    if raw_candidate_control.get("budget_skips") != budget_skip_count:
        return "native control budget-skip count does not match the axis"
    return None


_NATIVE_CANONICAL_TELEMETRY_FIELDS = (
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
)


class _NativeCanonicalProjectionHasher:
    def __init__(self) -> None:
        self._hasher = hashlib.sha256(b"stage05.2-native-canonical-row-projection-v1")
        self._stream_counts = [0] * 11
        self.count = 0

    def append(self, row: Mapping[str, object]) -> None:
        if set(row) != set(_NATIVE_CANONICAL_TELEMETRY_FIELDS):
            raise ValueError("native canonical telemetry fields are incomplete")
        values: list[int] = []
        for field in _NATIVE_CANONICAL_TELEMETRY_FIELDS:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("native canonical telemetry contains a non-integer")
            values.append(value)
        stream_code = values[1]
        if stream_code < 0 or stream_code >= len(self._stream_counts):
            raise ValueError("native canonical stream code is invalid")
        self._hasher.update(struct.pack("<10q", *values))
        self._stream_counts[stream_code] += 1
        self.count += 1

    def hexdigest(self) -> str:
        finished = self._hasher.copy()
        finished.update(struct.pack("<q", self.count))
        finished.update(struct.pack("<11q", *self._stream_counts))
        return finished.hexdigest()


def _native_canonical_receipt_error(
    payload: Mapping[str, object],
    projection: _NativeCanonicalProjectionHasher,
) -> str | None:
    statistics_raw = payload.get("native_execution_statistics")
    if not isinstance(statistics_raw, Mapping):
        return "full-native canonical journal statistics are missing"
    expected_count = statistics_raw.get("canonical_event_count")
    expected_sha256 = statistics_raw.get("canonical_event_journal_sha256")
    expected_projection_sha256 = statistics_raw.get("canonical_event_projection_sha256")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or not isinstance(expected_projection_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_projection_sha256)
    ):
        return "full-native canonical journal receipt is invalid"
    if projection.count != expected_count:
        return "full-native canonical telemetry coverage is incomplete"
    if projection.hexdigest() != expected_projection_sha256:
        return "full-native canonical telemetry digest does not replay"
    return None


def _replay_canonical_journal(
    payload: Mapping[str, object],
    *,
    axis_path: Path,
    instance: Instance,
) -> str | None:
    """Independently replay transactional, exact and cache journal state."""

    descriptor = payload.get("canonical_semantic_journal")
    if not isinstance(descriptor, dict):
        return "canonical semantic journal descriptor is missing"
    candidate_work_hash = _StreamingCandidateHash()
    route_result_hash = _StreamingCandidateHash()
    exact_queue: deque[tuple[str, tuple[object, object, object]]] = deque()
    candidate_expected: deque[tuple[str, tuple[object, object, object]]] = deque()
    candidate_results: deque[
        tuple[
            tuple[str, tuple[object, object, object]],
            tuple[str, ...],
            Mapping[str, object],
        ]
    ] = deque()
    committable_stores: dict[
        str,
        deque[tuple[tuple[object, object, object], int | None, int, str]],
    ] = defaultdict(deque)
    requires_candidate_route_results = bool(payload.get("route_result_hash"))
    raw_cache = payload.get("cache_incremental_statistics")
    if not isinstance(raw_cache, dict) or not raw_cache:
        return "cache incremental statistics are missing"
    raw_cache_config = raw_cache.get("config")
    if not isinstance(raw_cache_config, dict):
        return "cache configuration is missing"
    max_entries = raw_cache_config.get("max_entries")
    max_memory_bytes = raw_cache_config.get("max_memory_bytes")
    charging_configuration_version = raw_cache_config.get("charging_configuration_version")
    objective_schema_version = raw_cache_config.get("objective_schema_version")
    configured_instance_hash = raw_cache_config.get("instance_hash")
    if (
        raw_cache_config.get("enabled") is not True
        or raw_cache_config.get("eviction_policy") != "lru"
        or isinstance(max_entries, bool)
        or not isinstance(max_entries, int)
        or max_entries <= 0
        or isinstance(max_memory_bytes, bool)
        or not isinstance(max_memory_bytes, int)
        or max_memory_bytes <= 0
        or not isinstance(charging_configuration_version, str)
        or not charging_configuration_version
        or not isinstance(objective_schema_version, str)
        or not objective_schema_version
        or configured_instance_hash is not None
        and (not isinstance(configured_instance_hash, str) or not configured_instance_hash)
    ):
        return "cache configuration is invalid"
    instance_hash = configured_instance_hash or canonical_instance_hash(instance)
    cache_state: OrderedDict[str, tuple[str, int, str]] = OrderedDict()
    cache_seen_keys: set[str] = set()
    cache_counts: dict[str, int] = defaultdict(int)
    candidate_counts: dict[str, int] = defaultdict(int)
    rank_by_context: dict[tuple[object, object, object], list[int]] = defaultdict(list)
    candidate_ids_by_context: dict[tuple[object, object, object], set[int]] = defaultdict(set)
    native_plan_statuses: dict[
        tuple[tuple[object, object, object], int, int], tuple[str, int, int]
    ] = {}
    native_transaction_order: list[tuple[tuple[object, object, object], int]] = []
    native_candidate_misses: Counter[
        tuple[tuple[object, object, object], int, int]
    ] = Counter()
    native_candidate_writes: Counter[
        tuple[tuple[object, object, object], int, int]
    ] = Counter()
    native_candidate_evictions: Counter[
        tuple[tuple[object, object, object], int, int]
    ] = Counter()
    native_budget_skips: dict[
        tuple[tuple[object, object, object], int, int],
        tuple[int, int, int, int, int],
    ] = {}
    active_rounds: dict[tuple[object, object], int] = {}
    exact_started = 0
    exact_completed = 0
    exact_interrupted = 0
    cache_bytes_current = 0
    cache_entries_peak = 0
    cache_bytes_peak = 0
    pending_evictions: list[
        tuple[
            str,
            int,
            tuple[int, int],
            tuple[tuple[object, object, object], int, int] | None,
        ]
    ] = []
    native_event_id = 0
    native_canonical_projection = _NativeCanonicalProjectionHasher()
    current_native_schema = (
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("mode") in FULL_NATIVE_SEMANTIC_MODES
    )

    def expected_cache_digest(route_key: str) -> str:
        sequence = _route_sequence_from_canonical_key(route_key)
        return RouteCacheKey(
            instance_hash=instance_hash,
            customer_sequence=sequence,
            charging_configuration_version=charging_configuration_version,
            objective_schema_version=objective_schema_version,
        ).digest

    def cache_size_telemetry(event: Mapping[str, object]) -> tuple[int, int] | None:
        entries = event.get("current_entries")
        bytes_current = event.get("current_bytes")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (entries, bytes_current)
        ):
            return None
        assert isinstance(entries, int)
        assert isinstance(bytes_current, int)
        return entries, bytes_current

    def cache_transaction_identity(
        event: Mapping[str, object],
    ) -> tuple[tuple[object, object, object], int | None]:
        source_transaction_id = event.get("native_transaction_id")
        if source_transaction_id is not None and (
            isinstance(source_transaction_id, bool)
            or not isinstance(source_transaction_id, int)
            or source_transaction_id < 0
        ):
            raise ValueError("cache source transaction identity is invalid")
        raw_native = event.get("native_telemetry")
        if raw_native is None:
            return _event_context(event), source_transaction_id
        if not isinstance(raw_native, Mapping):
            raise ValueError("cache transaction native telemetry is invalid")
        transaction_id = raw_native.get("runtime_native_transaction_id")
        if (
            isinstance(transaction_id, bool)
            or not isinstance(transaction_id, int)
            or transaction_id < 0
        ):
            raise ValueError("cache transaction native identity is invalid")
        if source_transaction_id is not None and source_transaction_id != transaction_id:
            raise ValueError("cache source/native transaction identities diverge")
        return _event_context(event), transaction_id

    def native_candidate_key(
        event: Mapping[str, object],
    ) -> tuple[tuple[object, object, object], int, int] | None:
        context, transaction_id = cache_transaction_identity(event)
        candidate_id = event.get("candidate_id")
        if transaction_id is None:
            if current_native_schema and candidate_id is not None:
                raise ValueError("native candidate transaction identity is missing")
            return None
        if (
            isinstance(candidate_id, bool)
            or not isinstance(candidate_id, int)
            or candidate_id < 0
        ):
            raise ValueError("native cache candidate identity is invalid")
        transaction_key = (context, transaction_id)
        if transaction_key not in native_transaction_order:
            native_transaction_order.append(transaction_key)
        return context, transaction_id, candidate_id

    def replay_exact_result(
        sequence: tuple[str, ...],
        fields: Mapping[str, object],
    ) -> tuple[int, str] | None:
        def exact_float(value: object) -> float | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, int | float):
                return float(value)
            if isinstance(value, Mapping) and set(value) == {"nonfinite_float"}:
                marker = value.get("nonfinite_float")
                if marker == "positive_inf":
                    return math.inf
                if marker == "negative_inf":
                    return -math.inf
                if marker == "nan":
                    return math.nan
            return None

        feasible = fields.get("feasible")
        route = fields.get("route")
        failure_reason = fields.get("failure_reason")
        numeric_names = (
            "distance",
            "total_energy",
            "charged_energy",
            "charging_time",
        )
        numeric_values = [exact_float(fields.get(name)) for name in numeric_names]
        counter_names = ("labels_generated", "labels_expanded", "labels_pruned")
        counter_values = [fields.get(name) for name in counter_names]
        if (
            not isinstance(feasible, bool)
            or not isinstance(route, list)
            or not all(isinstance(node, str) for node in route)
            or not isinstance(failure_reason, str)
            or any(value is None for value in numeric_values)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counter_values
            )
        ):
            return None
        assert all(value is not None for value in numeric_values)
        assert all(isinstance(value, int) for value in counter_values)
        result = ChargingSubproblemResult(
            feasible=feasible,
            route=tuple(cast(list[str], route)),
            distance=cast(float, numeric_values[0]),
            total_energy=cast(float, numeric_values[1]),
            charged_energy=cast(float, numeric_values[2]),
            charging_time=cast(float, numeric_values[3]),
            labels_generated=cast(int, counter_values[0]),
            labels_expanded=cast(int, counter_values[1]),
            labels_pruned=cast(int, counter_values[2]),
            runtime_seconds=0.0,
            failure_reason=failure_reason,
        )
        routed_customers = tuple(
            node for node in result.route if instance.by_name[node].kind == NodeType.CUSTOMER
        )
        if result.feasible and routed_customers != sequence:
            return None
        return (
            estimate_cache_entry_bytes(result),
            charging_result_semantic_digest(result),
        )

    def validate_native_identity(event: Mapping[str, object]) -> str | None:
        nonlocal native_event_id
        raw_native_telemetry = event.get("native_telemetry")
        if raw_native_telemetry is None:
            if any(key.startswith("runtime_native_") for key in event):
                return "control-derived semantic row leaks native telemetry"
            return None
        if not isinstance(raw_native_telemetry, dict):
            return "native semantic telemetry is invalid"
        native_fields = _NATIVE_CANONICAL_TELEMETRY_FIELDS
        if set(raw_native_telemetry) != set(native_fields) or any(
            isinstance(raw_native_telemetry.get(field), bool)
            or not isinstance(raw_native_telemetry.get(field), int)
            for field in native_fields
        ):
            return "native semantic telemetry fields are incomplete"
        native_event_id += 1
        if raw_native_telemetry["runtime_native_event_id"] != native_event_id:
            return "native semantic event IDs are not contiguous"
        try:
            native_canonical_projection.append(cast(Mapping[str, object], raw_native_telemetry))
        except (OverflowError, struct.error, ValueError) as error:
            return f"native canonical telemetry replay failed: {error}"
        return None

    try:
        for event in iter_verified_semantic_journal(axis_path, descriptor):
            native_identity_error = validate_native_identity(event)
            if native_identity_error is not None:
                return native_identity_error
            stream = event.get("semantic_stream")
            event_type = event.get("event_type")
            if stream == "exact_work" and event_type == "exact_batch_started":
                sequences = event.get("customer_sequences")
                requested = event.get("requested_calls")
                started = event.get("started_calls")
                if (
                    not isinstance(sequences, list)
                    or isinstance(requested, bool)
                    or not isinstance(requested, int)
                    or isinstance(started, bool)
                    or not isinstance(started, int)
                    or started != len(sequences)
                    or not 0 <= started <= requested
                ):
                    return "exact-work batch accounting is invalid"
                context = _event_context(event)
                candidate_work_hash.append(
                    {
                        "lane": context[0],
                        "iteration": context[1],
                        "operator": context[2],
                        "sequences": sequences,
                    }
                )
                for sequence in sequences:
                    if not isinstance(sequence, list):
                        return "exact-work sequence is invalid"
                    identity = (canonical_route_key(sequence), context)
                    exact_queue.append(identity)
                    if requires_candidate_route_results:
                        candidate_expected.append(identity)
            elif stream == "exact_result" and event_type == "exact_route_result":
                started = event.get("exact_started")
                completed = event.get("exact_completed")
                if not isinstance(started, bool) or not isinstance(completed, bool):
                    return "exact-result state flags are invalid"
                exact_started += int(started)
                exact_completed += int(completed)
                exact_interrupted += int(
                    started and not completed or event.get("status") == "interrupted_deadline"
                )
                if started:
                    if not exact_queue:
                        return "exact-result has no preceding exact-work route"
                    expected_route, expected_context = exact_queue.popleft()
                    if (
                        event.get("route_key") != expected_route
                        or _event_context(event) != expected_context
                    ):
                        return "exact-result order/context does not match exact work"
                    if completed:
                        if requires_candidate_route_results:
                            if not candidate_results:
                                return "completed exact result lacks candidate result"
                            (
                                candidate_identity,
                                candidate_sequence,
                                candidate_fields,
                            ) = candidate_results.popleft()
                            if candidate_identity != (expected_route, expected_context):
                                return "candidate result order does not match exact result"
                            if candidate_fields.get("feasible") != event.get(
                                "feasible"
                            ) or candidate_fields.get("failure_reason") != event.get(
                                "failure_reason"
                            ):
                                return "candidate and exact result states diverge"
                            cache_identity = replay_exact_result(
                                candidate_sequence,
                                candidate_fields,
                            )
                            if cache_identity is None:
                                return "candidate exact result cannot reconstruct cache entry"
                            native_context = cache_transaction_identity(event)
                            committable_stores[expected_route].append(
                                (*native_context, *cache_identity)
                            )
                        else:
                            sequence = _route_sequence_from_canonical_key(expected_route)
                            exact = solve_exact_charging(instance, sequence)
                            if exact.feasible != event.get("feasible") or (
                                exact.failure_reason != event.get("failure_reason")
                            ):
                                return "exact result does not replay independently"
                            native_context = cache_transaction_identity(event)
                            committable_stores[expected_route].append(
                                (
                                    *native_context,
                                    estimate_cache_entry_bytes(exact),
                                    charging_result_semantic_digest(exact),
                                )
                            )
                    elif requires_candidate_route_results:
                        if not candidate_expected:
                            return "interrupted exact result lost candidate expectation"
                        if candidate_expected.popleft() != (
                            expected_route,
                            expected_context,
                        ):
                            return "interrupted exact result order diverges"
            elif stream == "candidate_transaction" and event_type == "candidate_route_result":
                sequences = event.get("customer_sequences")
                if (
                    not isinstance(sequences, list)
                    or len(sequences) != 2
                    or not all(isinstance(sequence, list) for sequence in sequences)
                ):
                    return "candidate route-result sequence payload is invalid"
                result_fields = {
                    field: event.get(field)
                    for field in (
                        "feasible",
                        "route",
                        "distance",
                        "total_energy",
                        "charged_energy",
                        "charging_time",
                        "labels_generated",
                        "labels_expanded",
                        "labels_pruned",
                        "failure_reason",
                    )
                }
                route_result_hash.append({"sequence": sequences[0], "result": result_fields})
                identity = (canonical_route_key(sequences[0]), _event_context(event))
                if not candidate_expected or candidate_expected.popleft() != identity:
                    return "candidate result has no matching exact-work route"
                candidate_results.append(
                    (identity, tuple(cast(list[str], sequences[0])), result_fields)
                )
            elif stream == "candidate_transaction" and event_type == "candidate_plan_decision":
                context = _event_context(event)
                rank = event.get("rank")
                candidate_id = event.get("candidate_id")
                if (
                    isinstance(rank, bool)
                    or not isinstance(rank, int)
                    or rank <= 0
                    or isinstance(candidate_id, bool)
                    or not isinstance(candidate_id, int)
                    or candidate_id < 0
                    or candidate_id in candidate_ids_by_context[context]
                ):
                    return "candidate-plan rank or identity is invalid"
                candidate_ids_by_context[context].add(candidate_id)
                rank_by_context[context].append(rank)
                status = event.get("status")
                plan_context, plan_native_transaction = cache_transaction_identity(event)
                if plan_native_transaction is not None:
                    native_plan_key = (
                        plan_context,
                        plan_native_transaction,
                        candidate_id,
                    )
                    transaction_status = event.get("transaction_status_code")
                    if (
                        native_plan_key in native_plan_statuses
                        or not isinstance(status, str)
                        or isinstance(transaction_status, bool)
                        or not isinstance(transaction_status, int)
                    ):
                        return "native candidate-plan identity is duplicated"
                    native_plan_statuses[native_plan_key] = (
                        status,
                        transaction_status,
                        rank,
                    )
                    transaction_key = (plan_context, plan_native_transaction)
                    if transaction_key not in native_transaction_order:
                        native_transaction_order.append(transaction_key)
                candidate_counts["candidate_decisions"] += 1
                candidate_counts[
                    "selected_candidates" if status == "selected" else "skipped_candidates"
                ] += 1
            elif stream == "candidate_transaction" and event_type in {
                "candidate_control_decision",
                "candidate_control_decision_aggregate",
            }:
                count = event.get("aggregate_count", 1)
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    return "candidate route-decision count is invalid"
                candidate_counts["candidate_decisions"] += 1
                candidate_counts[
                    "selected_candidates"
                    if event.get("status") == "selected"
                    else "skipped_candidates"
                ] += count
            elif stream == "candidate_transaction" and event_type == "candidate_control_budget":
                requested = event.get("requested")
                granted = event.get("granted")
                remaining = event.get("remaining")
                if any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in (requested, granted, remaining)
                ):
                    return "candidate budget fields are invalid"
                assert isinstance(requested, int)
                assert isinstance(granted, int)
                assert isinstance(remaining, int)
                if granted > requested:
                    return "candidate budget grant exceeds request"
                raw_native_transaction = event.get("native_transaction_id")
                if (
                    current_native_schema
                    and event.get("status") == "budget_skipped"
                    and raw_native_transaction is None
                ):
                    return "native candidate budget identity is missing"
                if raw_native_transaction is not None:
                    candidate_id = event.get("candidate_id")
                    round_remaining = event.get("round_remaining")
                    exact_remaining = event.get("exact_remaining")
                    available = event.get("available")
                    transaction_status = event.get("transaction_status_code")
                    raw_budget_context = event.get("context")
                    iteration = event.get("iteration")
                    if (
                        isinstance(raw_native_transaction, bool)
                        or not isinstance(raw_native_transaction, int)
                        or raw_native_transaction < 0
                        or isinstance(candidate_id, bool)
                        or not isinstance(candidate_id, int)
                        or candidate_id < 0
                        or round_remaining != remaining
                        or isinstance(exact_remaining, bool)
                        or not isinstance(exact_remaining, int)
                        or exact_remaining < -1
                        or isinstance(available, bool)
                        or not isinstance(available, int)
                        or available < 0
                        or transaction_status != 3
                        or not isinstance(raw_budget_context, str)
                        or len(raw_budget_context.split(":")) != 3
                        or isinstance(iteration, bool)
                        or not isinstance(iteration, int)
                        or iteration < 0
                    ):
                        return "native candidate budget identity is invalid"
                    lane, operator, _scope = raw_budget_context.split(":")
                    assert isinstance(exact_remaining, int)
                    assert isinstance(available, int)
                    native_plan_key = (
                        (lane, iteration, operator),
                        raw_native_transaction,
                        candidate_id,
                    )
                    plan_status = native_plan_statuses.get(native_plan_key)
                    if (
                        event.get("status") != "budget_skipped"
                        or plan_status is None
                        or plan_status[:2] != ("selected", 3)
                        or native_plan_key in native_budget_skips
                    ):
                        return "native candidate budget skip does not replay"
                    native_budget_skips[native_plan_key] = (
                        requested,
                        granted,
                        remaining,
                        exact_remaining,
                        available,
                    )
                candidate_counts["budget_events"] += 1
                candidate_counts["budget_skips"] += int(event.get("status") == "budget_skipped")
            elif stream == "candidate_transaction" and event_type == "candidate_control_round":
                key = (event.get("lane"), event.get("iteration"))
                budget = event.get("budget")
                if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
                    return "candidate round budget is invalid"
                status = event.get("status")
                if status == "started":
                    if key in active_rounds:
                        return "candidate round started twice"
                    active_rounds[key] = budget
                elif status == "completed":
                    used = event.get("used")
                    remainder = event.get("remainder")
                    if (
                        key not in active_rounds
                        or active_rounds.pop(key) != budget
                        or isinstance(used, bool)
                        or not isinstance(used, int)
                        or isinstance(remainder, bool)
                        or not isinstance(remainder, int)
                        or used < 0
                        or remainder < 0
                        or used + remainder != budget
                    ):
                        return "candidate round completion does not conserve budget"
                    candidate_counts["completed_rounds"] += 1
                    candidate_counts["maximum_exact_calls_per_round"] = max(
                        candidate_counts["maximum_exact_calls_per_round"], used
                    )
                    candidate_counts["total_round_remainder"] += remainder
                else:
                    return "candidate round status is invalid"
            elif stream == "cache" and event_type == "cache_lookup_result":
                route_key = event.get("route_key")
                status = event.get("status")
                scope = event.get("cache_scope")
                if not isinstance(route_key, str) or status not in {"hit", "miss"}:
                    return "cache lookup result is invalid"
                if pending_evictions:
                    return "cache eviction sequence is missing its store"
                if event.get("cache_key_digest") != expected_cache_digest(route_key):
                    return "cache lookup key digest does not replay"
                cache_seen_keys.add(route_key)
                if scope == "committed":
                    telemetry = cache_size_telemetry(event)
                    if telemetry is None:
                        return "cache lookup size telemetry is invalid"
                    cache_counts["cache_lookups"] += 1
                    cache_counts["cache_hits" if status == "hit" else "cache_misses"] += 1
                    if (status == "hit") != (route_key in cache_state):
                        return "committed cache lookup does not match replayed state"
                    if status == "hit":
                        cache_state.move_to_end(route_key)
                elif scope == "candidate_pending" and status == "hit":
                    telemetry = cache_size_telemetry(event)
                    has_size_telemetry = "current_entries" in event or "current_bytes" in event
                    if has_size_telemetry and telemetry is None:
                        return "candidate-pending cache size telemetry is invalid"
                    cache_counts["candidate_pending_hits"] += 1
                else:
                    return "cache lookup scope is invalid"
                if telemetry is not None and telemetry != (
                    len(cache_state),
                    cache_bytes_current,
                ):
                    return "cache lookup size telemetry does not replay"
                native_key = native_candidate_key(event)
                if native_key is not None and status == "miss":
                    native_candidate_misses[native_key] += 1
            elif stream == "cache" and event_type == "cache_lifecycle":
                route_key = event.get("route_key")
                operation = event.get("status")
                if not isinstance(route_key, str):
                    return "cache lifecycle route identity is invalid"
                if event.get("cache_key_digest") != expected_cache_digest(route_key):
                    return "cache lifecycle key digest does not replay"
                telemetry = cache_size_telemetry(event)
                if telemetry is None:
                    return "cache lifecycle size telemetry is invalid"
                if operation == "store":
                    if not committable_stores[route_key]:
                        return "cache store has no completed exact result"
                    entry_bytes = event.get("entry_bytes")
                    (
                        expected_context,
                        expected_native_transaction,
                        expected_entry_bytes,
                        expected_result_digest,
                    ) = committable_stores[route_key][0]
                    if cache_transaction_identity(event) != (
                        expected_context,
                        expected_native_transaction,
                    ):
                        return "cache store transaction does not match exact result"
                    committable_stores[route_key].popleft()
                    if (
                        isinstance(entry_bytes, bool)
                        or not isinstance(entry_bytes, int)
                        or entry_bytes <= 0
                        or entry_bytes != expected_entry_bytes
                        or entry_bytes > max_memory_bytes
                        or route_key in cache_state
                        or len(cache_state) >= max_entries
                        or cache_bytes_current + entry_bytes > max_memory_bytes
                    ):
                        return "cache store violates bounded LRU state"
                    pre_eviction_entries = len(cache_state) + len(pending_evictions)
                    pre_eviction_bytes = cache_bytes_current + sum(
                        evicted_bytes for _, evicted_bytes, _, _ in pending_evictions
                    )
                    remaining_entries = pre_eviction_entries
                    remaining_bytes = pre_eviction_bytes
                    for _, evicted_bytes, _, _ in pending_evictions:
                        if (
                            remaining_entries < max_entries
                            and remaining_bytes + entry_bytes <= max_memory_bytes
                        ):
                            return "cache eviction was not necessary for the pending store"
                        remaining_entries -= 1
                        remaining_bytes -= evicted_bytes
                    if (
                        remaining_entries >= max_entries
                        or remaining_bytes + entry_bytes > max_memory_bytes
                    ):
                        return "cache eviction sequence is insufficient for the pending store"
                    cache_state[route_key] = (
                        cast(str, event["cache_key_digest"]),
                        entry_bytes,
                        expected_result_digest,
                    )
                    cache_bytes_current += entry_bytes
                    cache_counts["cache_stores"] += 1
                    expected_size = (len(cache_state), cache_bytes_current)
                    store_native_key = native_candidate_key(event)
                    if any(
                        eviction_native_key != store_native_key
                        for _, _, _, eviction_native_key in pending_evictions
                    ):
                        return "cache eviction transaction does not match pending store"
                    if telemetry != expected_size or any(
                        observed != expected_size
                        for _, _, observed, _ in pending_evictions
                    ):
                        return "cache store size telemetry does not replay"
                    pending_evictions.clear()
                elif operation == "evict":
                    if route_key not in cache_state:
                        return "cache eviction targets an absent route"
                    if route_key != next(iter(cache_state)):
                        return "cache eviction does not target the LRU route"
                    stored_digest, stored_bytes, _ = cache_state.pop(route_key)
                    if stored_digest != event.get("cache_key_digest"):
                        return "cache eviction digest disagrees with stored entry"
                    cache_bytes_current -= stored_bytes
                    cache_counts["cache_evictions"] += 1
                    eviction_native_key = native_candidate_key(event)
                    pending_evictions.append(
                        (route_key, stored_bytes, telemetry, eviction_native_key)
                    )
                    if eviction_native_key is not None:
                        native_candidate_evictions[eviction_native_key] += 1
                elif operation == "reconcile":
                    if route_key not in cache_state:
                        return "cache reconciliation targets an absent route"
                    if pending_evictions:
                        return "cache eviction sequence is missing its store"
                    if not committable_stores[route_key]:
                        return "cache reconciliation has no completed exact result"
                    (
                        expected_context,
                        expected_native_transaction,
                        expected_entry_bytes,
                        expected_result_digest,
                    ) = committable_stores[route_key][0]
                    if cache_transaction_identity(event) != (
                        expected_context,
                        expected_native_transaction,
                    ):
                        return "cache reconciliation transaction does not match exact result"
                    committable_stores[route_key].popleft()
                    entry_bytes = event.get("entry_bytes")
                    stored_digest, stored_bytes, stored_result_digest = cache_state[route_key]
                    if (
                        stored_digest != event.get("cache_key_digest")
                        or expected_entry_bytes != stored_bytes
                        or event.get("pending_result_digest") != expected_result_digest
                        or event.get("existing_result_digest") != stored_result_digest
                        or entry_bytes is not None
                        and (
                            isinstance(entry_bytes, bool)
                            or not isinstance(entry_bytes, int)
                            or entry_bytes != stored_bytes
                        )
                        or telemetry != (len(cache_state), cache_bytes_current)
                    ):
                        return "cache reconciliation does not match stored entry"
                elif operation == "oversize_not_cached":
                    entry_bytes = event.get("entry_bytes")
                    if not committable_stores[route_key]:
                        return "cache oversize receipt has no completed exact result"
                    (
                        expected_context,
                        expected_native_transaction,
                        expected_entry_bytes,
                        _,
                    ) = committable_stores[route_key][0]
                    if cache_transaction_identity(event) != (
                        expected_context,
                        expected_native_transaction,
                    ):
                        return "cache oversize transaction does not match exact result"
                    committable_stores[route_key].popleft()
                    if (
                        pending_evictions
                        or route_key in cache_state
                        or isinstance(entry_bytes, bool)
                        or not isinstance(entry_bytes, int)
                        or entry_bytes != expected_entry_bytes
                        or entry_bytes <= max_memory_bytes
                        or telemetry != (len(cache_state), cache_bytes_current)
                    ):
                        return "cache oversize receipt does not replay"
                    cache_counts["cache_oversize_not_cached"] += 1
                else:
                    return "cache lifecycle operation is invalid"
                native_key = native_candidate_key(event)
                if native_key is not None and operation != "evict":
                    native_candidate_writes[native_key] += 1
                cache_entries_peak = max(cache_entries_peak, len(cache_state))
                cache_bytes_peak = max(cache_bytes_peak, cache_bytes_current)
        if payload.get("mode") in FULL_NATIVE_SEMANTIC_MODES:
            native_receipt_error = _native_canonical_receipt_error(
                payload,
                native_canonical_projection,
            )
            if native_receipt_error is not None:
                return native_receipt_error
        elif native_canonical_projection.count:
            return "non-full-native journal contains native canonical telemetry"
        if exact_queue:
            return "exact-work routes are missing exact-result records"
        if candidate_expected or candidate_results:
            return "candidate result transaction is incomplete"
        if active_rounds:
            return "candidate-control rounds are not closed"
        if pending_evictions:
            return "cache eviction sequence is missing its store"
        for ranks in rank_by_context.values():
            if sorted(ranks) != list(range(1, len(ranks) + 1)):
                return "candidate-plan ranks are not contiguous"
        if current_native_schema:
            raw_candidate_control = payload.get("candidate_control_statistics")
            fixed_work_budget = payload.get("fixed_work_budget")
            if not isinstance(raw_candidate_control, dict):
                return "native candidate-control statistics are missing"
            round_budget = raw_candidate_control.get("max_exact_calls_per_round")
            if (
                isinstance(round_budget, bool)
                or not isinstance(round_budget, int)
                or round_budget <= 0
            ):
                return "native candidate-control round budget is invalid"
            exact_budget = -1
            if payload.get("axis") == "fixed_work":
                if not isinstance(fixed_work_budget, dict):
                    return "native fixed-work budget is missing"
                raw_exact_budget = fixed_work_budget.get("exact_calls")
                if (
                    isinstance(raw_exact_budget, bool)
                    or not isinstance(raw_exact_budget, int)
                    or raw_exact_budget <= 0
                ):
                    return "native fixed-work exact budget is invalid"
                exact_budget = raw_exact_budget
            skipped_plan_keys = {
                key
                for key, (status, transaction_status, _rank) in native_plan_statuses.items()
                if status == "selected" and transaction_status == 3
            }
            if skipped_plan_keys != set(native_budget_skips):
                return "native canonical budget skips do not bind selected plans"
            replayed_exact_started = 0
            round_used_by_iteration: dict[int, int] = defaultdict(int)
            for transaction_context, transaction_id in native_transaction_order:
                plan_rows = sorted(
                    (
                        (rank, candidate_id, status, transaction_status)
                        for (
                            plan_context,
                            plan_transaction_id,
                            candidate_id,
                        ), (
                            status,
                            transaction_status,
                            rank,
                        ) in native_plan_statuses.items()
                        if plan_context == transaction_context
                        and plan_transaction_id == transaction_id
                        and status == "selected"
                    )
                )
                public_iteration = transaction_context[1]
                for _rank, candidate_id, status, transaction_status in plan_rows:
                    plan_key = (transaction_context, transaction_id, candidate_id)
                    exact_rows = native_candidate_writes[plan_key]
                    receipt = native_budget_skips.get(plan_key)
                    if receipt is not None:
                        (
                            requested,
                            granted,
                            round_remaining,
                            exact_remaining,
                            available,
                        ) = receipt
                        expected_exact_remaining = (
                            -1
                            if exact_budget < 0
                            else max(0, exact_budget - replayed_exact_started)
                        )
                        expected_round_remaining = (
                            max(
                                0,
                                round_budget
                                - round_used_by_iteration[public_iteration],
                            )
                            if isinstance(public_iteration, int)
                            and not isinstance(public_iteration, bool)
                            else round_budget
                        )
                        expected_available = min(
                            expected_round_remaining,
                            requested
                            if expected_exact_remaining < 0
                            else expected_exact_remaining,
                        )
                        if (
                            (status, transaction_status) != ("selected", 3)
                            or requested != native_candidate_misses[plan_key]
                            or granted != 0
                            or round_remaining != expected_round_remaining
                            or exact_remaining != expected_exact_remaining
                            or available != expected_available
                            or requested <= available
                            or exact_rows != 0
                            or native_candidate_evictions[plan_key] != 0
                        ):
                            return "native canonical budget skip does not replay"
                    replayed_exact_started += exact_rows
                    if (
                        isinstance(public_iteration, int)
                        and not isinstance(public_iteration, bool)
                    ):
                        round_used_by_iteration[public_iteration] += exact_rows
                plan_candidate_ids = {candidate_id for _, candidate_id, _, _ in plan_rows}
                replayed_exact_started += sum(
                    count
                    for (
                        write_context,
                        write_transaction_id,
                        write_candidate_id,
                    ), count in native_candidate_writes.items()
                    if write_context == transaction_context
                    and write_transaction_id == transaction_id
                    and write_candidate_id not in plan_candidate_ids
                )
            if replayed_exact_started != exact_started:
                return "native canonical exact-work budget does not replay"
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return f"canonical semantic replay failed: {error}"
    declared_candidate_work_hash = payload.get("candidate_work_hash")
    if not isinstance(declared_candidate_work_hash, str):
        return "candidate-work hash declaration is invalid"
    if (
        declared_candidate_work_hash
        and candidate_work_hash.hexdigest() != declared_candidate_work_hash
    ):
        return "candidate-work hash does not replay from journal"
    declared_route_result_hash = payload.get("route_result_hash")
    if not isinstance(declared_route_result_hash, str):
        return "route-result hash declaration is invalid"
    if declared_route_result_hash and route_result_hash.hexdigest() != declared_route_result_hash:
        return "route-result hash does not replay from journal"
    if (
        exact_started != payload.get("exact_started_calls")
        or exact_completed != payload.get("exact_completed_calls")
        or exact_interrupted != payload.get("exact_interrupted_calls")
    ):
        return "exact call counters do not replay from journal"
    raw_candidate = payload.get("candidate_control_statistics")
    if isinstance(raw_candidate, dict) and raw_candidate.get("enabled") is True:
        for field in (
            "candidate_decisions",
            "selected_candidates",
            "skipped_candidates",
            "budget_events",
            "budget_skips",
            "completed_rounds",
            "maximum_exact_calls_per_round",
            "total_round_remainder",
        ):
            if raw_candidate.get(field) != candidate_counts[field]:
                return f"candidate-control {field} does not replay from journal"
    replayed_cache = {
        "cache_lookups": cache_counts["cache_lookups"],
        "cache_hits": cache_counts["cache_hits"],
        "cache_misses": cache_counts["cache_misses"],
        "cache_stores": cache_counts["cache_stores"],
        "cache_evictions": cache_counts["cache_evictions"],
        "cache_oversize_not_cached": cache_counts["cache_oversize_not_cached"],
        "entries_current": len(cache_state),
        "entries_peak": cache_entries_peak,
        "bytes_current": cache_bytes_current,
        "bytes_peak": cache_bytes_peak,
        "unique_route_evaluations": len(cache_seen_keys),
    }
    for field, expected in replayed_cache.items():
        if raw_cache.get(field) != expected:
            return f"cache {field} does not replay from journal"
    raw_route_cache = raw_cache.get("route_cache")
    if not isinstance(raw_route_cache, dict):
        return "nested route cache statistics are missing"
    for field, expected in replayed_cache.items():
        if raw_route_cache.get(field) != expected:
            return f"nested route cache {field} does not replay from journal"
    if (
        raw_route_cache.get("eviction_policy") != "lru"
        or raw_route_cache.get("max_entries") != max_entries
        or raw_route_cache.get("max_memory_bytes") != max_memory_bytes
        or raw_route_cache.get("instance_hash") != configured_instance_hash
    ):
        return "nested route cache configuration does not replay"
    if payload.get("mode") in FULL_NATIVE_SEMANTIC_MODES and native_event_id == 0:
        return "full-native semantic journal has no native-backed events"
    return _replay_physical_telemetry(payload, axis_path=axis_path)


def _replay_record(
    record: ReviewRecord,
    benchmark_dir: Path,
    *,
    expected_resource_telemetry: bool = True,
) -> dict[str, object]:
    payload = _axis_payload_with_persistence(record.path, record.payload)
    comparison_schema = _string(payload, "schema_version")
    if comparison_schema not in {
        LEGACY_COMPARISON_SCHEMA_VERSION,
        INLINE_SEMANTIC_COMPARISON_SCHEMA_VERSION,
        PREVIOUS_COMPARISON_SCHEMA_VERSION,
        LEGACY_PROFILE_COMPARISON_SCHEMA_VERSION,
        PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION,
        PROCESS_PROFILE_COMPARISON_SCHEMA_VERSION,
        TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }:
        raise RuntimeError(f"unsupported comparison schema: {record.path}")
    if _string(payload, "status") != "completed":
        return {
            "valid": False,
            "reason": f"axis failed: {payload.get('error_type')}: {payload.get('error')}",
        }
    if comparison_schema in {
        INLINE_SEMANTIC_COMPARISON_SCHEMA_VERSION,
        PREVIOUS_COMPARISON_SCHEMA_VERSION,
        LEGACY_PROFILE_COMPARISON_SCHEMA_VERSION,
        PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION,
        PROCESS_PROFILE_COMPARISON_SCHEMA_VERSION,
        TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }:
        if "semantic_trajectory" not in payload or payload.get("semantic_trajectory") is None:
            return {
                "valid": False,
                "reason": "semantic trajectory replay failed: v4 evidence is missing",
            }
        try:
            _semantic_trajectory(payload)
            if comparison_schema == INLINE_SEMANTIC_COMPARISON_SCHEMA_VERSION:
                _canonical_semantic_events(payload)
        except ValueError as error:
            return {
                "valid": False,
                "reason": f"semantic trajectory replay failed: {error}",
            }
    instance_name = _string(payload, "instance")
    instance = parse_schneider(benchmark_dir / f"{instance_name}.txt")
    if comparison_schema in {
        INLINE_SEMANTIC_COMPARISON_SCHEMA_VERSION,
        PREVIOUS_COMPARISON_SCHEMA_VERSION,
        LEGACY_PROFILE_COMPARISON_SCHEMA_VERSION,
        PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION,
        PROCESS_PROFILE_COMPARISON_SCHEMA_VERSION,
        TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }:
        if comparison_schema in EXTERNAL_SEMANTIC_COMPARISON_SCHEMA_VERSIONS:
            if payload.get("mode") in FULL_NATIVE_SEMANTIC_MODES:
                control_replay_error = _replay_raw_native_control_journal(
                    payload,
                    axis_path=record.path,
                    instance=instance,
                )
                if control_replay_error is not None:
                    return {"valid": False, "reason": control_replay_error}
            replay_error = _replay_canonical_journal(
                payload,
                axis_path=record.path,
                instance=instance,
            )
            if replay_error is not None:
                return {"valid": False, "reason": replay_error}
            measurement_error = _replay_measurement_evidence(
                payload,
                axis_path=record.path,
            )
            if measurement_error is not None:
                return {"valid": False, "reason": measurement_error}
        try:
            screening_error = _replay_physical_screening(
                payload,
                instance,
                axis_path=record.path,
            )
        except (OSError, RuntimeError, ValueError) as error:
            screening_error = f"physical screening journal replay failed: {error}"
        if screening_error is not None:
            return {"valid": False, "reason": screening_error}
    raw_routes = _sequence(payload, "routes")
    routes: list[list[str]] = []
    for raw_route in raw_routes:
        if not isinstance(raw_route, list) or not all(isinstance(name, str) for name in raw_route):
            raise RuntimeError(f"invalid route schema: {record.path}")
        routes.append(list(raw_route))
    report = validate_routes(instance, routes)
    if not report.feasible:
        return {"valid": False, "reason": "unified validator replay failed"}
    objective = SolutionObjective.from_report(instance, report)
    recorded_objective = _sequence(payload, "objective")
    if list(objective.key) != recorded_objective:
        return {"valid": False, "reason": "objective replay mismatch"}
    fallback_count = payload.get("fallback_count")
    if isinstance(fallback_count, bool) or not isinstance(fallback_count, int):
        return {"valid": False, "reason": "fallback evidence is missing or malformed"}
    if fallback_count != 0:
        return {"valid": False, "reason": "native fallback count is non-zero"}
    if comparison_schema in EXTERNAL_SEMANTIC_COMPARISON_SCHEMA_VERSIONS and record.mode in {
        ArchitectureMode.FULL_NATIVE_ALNS,
        ArchitectureMode.HOST_SCHEDULER,
    }:
        kind_codes = {
            NodeType.DEPOT: 0,
            NodeType.CUSTOMER: 1,
            NodeType.STATION: 2,
        }
        receipt_error = _replay_initial_state_receipt(
            payload,
            record.mode,
            expected_node_kind=tuple(kind_codes[node.kind] for node in instance.nodes),
            expected_exact_batch_size=128,
        )
        if receipt_error is not None:
            return {"valid": False, "reason": receipt_error}
    if comparison_schema in EXTERNAL_SEMANTIC_COMPARISON_SCHEMA_VERSIONS:
        topology = payload.get("topology")
        if not isinstance(topology, dict):
            return {"valid": False, "reason": "process-tree topology is missing"}
        process_tree_fields = (
            "sample_count",
            "peak_concurrent_processes",
            "peak_aggregate_threads",
            "peak_aggregate_rss_bytes",
            "peak_aggregate_pss_bytes",
            "process_tree_cpu_seconds",
        )
        resource_error = _resource_telemetry_topology_error(
            topology,
            mode=record.mode,
            expected_resource_telemetry=expected_resource_telemetry,
            require_complete_schema=comparison_schema in PROFILE_COMPARISON_SCHEMA_VERSIONS,
            comparison_schema=comparison_schema,
        )
        if resource_error is not None:
            return {"valid": False, "reason": resource_error}
        if expected_resource_telemetry:
            for field in process_tree_fields:
                value = topology.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    return {
                        "valid": False,
                        "reason": f"process-tree field is missing or invalid: {field}",
                    }
        persistence = payload.get("persistence_seconds")
        if (
            isinstance(persistence, bool)
            or not isinstance(persistence, int | float)
            or not math.isfinite(float(persistence))
            or float(persistence) < 0.0
        ):
            return {"valid": False, "reason": "persistence interval is invalid"}
        if comparison_schema in PROFILE_COMPARISON_SCHEMA_VERSIONS:
            try:
                startup_seconds = _number(payload, "startup_seconds")
                producer_pre_receipt_seconds = _number(
                    payload,
                    "producer_pre_receipt_seconds",
                )
            except ValueError as error:
                return {"valid": False, "reason": str(error)}
            if startup_seconds < 0.0 or producer_pre_receipt_seconds < 0.0:
                return {
                    "valid": False,
                    "reason": "startup/producer timing is negative",
                }
            solver = _number(payload, "solver_seconds")
            if producer_pre_receipt_seconds + 1e-9 < startup_seconds + solver + float(persistence):
                return {
                    "valid": False,
                    "reason": ("producer timing does not cover startup, solver, and persistence"),
                }
            persistence_error = _persistence_breakdown_error(payload)
            if persistence_error is not None:
                return {"valid": False, "reason": persistence_error}
        if record.mode is ArchitectureMode.HOST_SCHEDULER:
            scheduler_pid = topology.get("scheduler_process_id")
            roots = topology.get("additional_root_pids")
            if (
                isinstance(scheduler_pid, bool)
                or not isinstance(scheduler_pid, int)
                or scheduler_pid <= 0
                or not isinstance(roots, list)
                or scheduler_pid not in roots
                or topology.get("shared_scheduler_resource_attribution")
                != "mode_wave_primary_axis_values_overlap"
            ):
                return {
                    "valid": False,
                    "reason": "host scheduler process-tree attribution is incomplete",
                }
    semantic_completeness = payload.get("semantic_completeness")
    semantics_complete = isinstance(semantic_completeness, dict) and all(
        semantic_completeness.get(field) is True
        for field in ("candidate_control", "stage04", "measurement_trace")
    )
    measurement = payload.get("measurement_evidence")
    if not isinstance(measurement, dict):
        return {"valid": False, "reason": "measurement evidence is missing"}
    recorded_measurement_hash = measurement.get("sha256")
    if not isinstance(recorded_measurement_hash, str):
        return {"valid": False, "reason": "measurement evidence hash is missing"}
    hashed_measurement = dict(measurement)
    del hashed_measurement["sha256"]
    if comparison_schema in EXTERNAL_SEMANTIC_COMPARISON_SCHEMA_VERSIONS:
        hashed_measurement.pop("native_telemetry", None)
    recomputed_measurement_hash = hashlib.sha256(
        json.dumps(
            hashed_measurement,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if recorded_measurement_hash != recomputed_measurement_hash:
        return {"valid": False, "reason": "measurement evidence hash mismatch"}
    for field in (
        "operator_statistics",
        "stage04_statistics",
        "stage04_events",
        "candidate_transaction_events",
    ):
        if field not in payload:
            return {"valid": False, "reason": f"{field} evidence is missing"}
    return {
        "valid": True,
        "semantics_complete": semantics_complete,
        "objective": objective.key,
        "routes": tuple(tuple(route) for route in routes),
    }


def _replay_initial_state_receipt(
    payload: Mapping[str, object],
    mode: ArchitectureMode,
    *,
    expected_node_kind: Sequence[int],
    expected_exact_batch_size: int,
) -> str | None:
    """Independently replay the persisted request-to-initial-state binding."""

    native = payload.get("native_execution_statistics")
    if not isinstance(native, dict):
        return "native execution evidence is missing"
    receipt = native.get("initial_state_receipt")
    if not isinstance(receipt, dict):
        return "initial-state ownership receipt is missing"
    if receipt.get("schema_version") != "stage05.2-native-initial-state-receipt-v3":
        return "initial-state ownership receipt schema is invalid"
    host_owned = receipt.get("host_owned")
    operation_count = receipt.get("operation_count")
    telemetry_count = native.get("initial_state_request_count")
    seed = payload.get("seed")
    expected_host_owned = mode is ArchitectureMode.HOST_SCHEDULER
    if (
        not isinstance(host_owned, bool)
        or isinstance(operation_count, bool)
        or not isinstance(operation_count, int)
        or isinstance(telemetry_count, bool)
        or not isinstance(telemetry_count, int)
        or host_owned is not expected_host_owned
        or operation_count != int(host_owned)
        or telemetry_count != operation_count
    ):
        return "initial-state ownership receipt does not reconcile"
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not -(1 << 63) <= seed <= (1 << 63) - 1
    ):
        return "initial-state ownership receipt seed is invalid"
    hashes = tuple(
        receipt.get(field)
        for field in (
            "request_sha256",
            "state_sha256",
            "initial_four_lane_state_sha256",
            "transaction_sha256",
        )
    )
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in hashes
    ):
        return "initial-state ownership receipt hash is invalid"
    (
        request_sha256,
        state_sha256,
        initial_four_lane_state_sha256,
        transaction_sha256,
    ) = hashes
    evidence = bytearray(b"stage05.2-native-initial-state-receipt-v3")
    evidence.extend(struct.pack("<qq", int(host_owned), operation_count))
    evidence.extend(str(request_sha256).encode("ascii"))
    evidence.extend(str(state_sha256).encode("ascii"))
    evidence.extend(str(initial_four_lane_state_sha256).encode("ascii"))
    if hashlib.sha256(evidence).hexdigest() != transaction_sha256:
        return "initial-state ownership receipt hash mismatch"
    return _replay_initial_four_lane_projection(
        receipt,
        request_sha256=str(request_sha256),
        initial_state_sha256=str(state_sha256),
        expected_sha256=str(initial_four_lane_state_sha256),
        expected_seed=seed,
        expected_node_kind=expected_node_kind,
        expected_exact_batch_size=expected_exact_batch_size,
    )


def _replay_initial_four_lane_projection(
    receipt: Mapping[str, object],
    *,
    request_sha256: str,
    initial_state_sha256: str,
    expected_sha256: str,
    expected_seed: int,
    expected_node_kind: Sequence[int],
    expected_exact_batch_size: int,
) -> str | None:
    """Rebuild the producer-independent initial four-lane state hash."""

    projection = receipt.get("initial_four_lane_projection")
    if not isinstance(projection, dict):
        return "initial four-lane projection is missing"
    if projection.get("schema_version") != "stage05.2-native-initial-four-lane-projection-v1":
        return "initial four-lane projection schema is invalid"
    int64_min = -(1 << 63)
    int64_max = (1 << 63) - 1

    def is_int64(value: object) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and int64_min <= value <= int64_max
        )

    def integers(field: str) -> list[int]:
        values = projection.get(field)
        if not isinstance(values, list) or any(not is_int64(value) for value in values):
            raise ValueError(f"initial four-lane {field} is invalid")
        return [int(value) for value in values]

    def floats(field: str) -> list[float]:
        values = projection.get(field)
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int | float) for value in values
        ):
            raise ValueError(f"initial four-lane {field} is invalid")
        try:
            output = [float(value) for value in values]
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"initial four-lane {field} is invalid") from error
        if len(output) != len(values) or any(not math.isfinite(value) for value in output):
            raise ValueError(f"initial four-lane {field} is invalid")
        return output

    def integer_matrix(field: str, width: int) -> list[int]:
        rows = projection.get(field)
        if not isinstance(rows, list) or any(
            not isinstance(row, list) or len(row) != width for row in rows
        ):
            raise ValueError(f"initial four-lane {field} matrix is invalid")
        flattened = [value for row in rows for value in row]
        if any(not is_int64(value) for value in flattened):
            raise ValueError(f"initial four-lane {field} matrix is invalid")
        return [int(value) for value in flattened]

    def float_matrix(field: str, width: int) -> list[float]:
        rows = projection.get(field)
        if not isinstance(rows, list) or any(
            not isinstance(row, list) or len(row) != width for row in rows
        ):
            raise ValueError(f"initial four-lane {field} matrix is invalid")
        flattened = [value for row in rows for value in row]
        if any(
            isinstance(value, bool) or not isinstance(value, int | float) for value in flattened
        ):
            raise ValueError(f"initial four-lane {field} matrix is invalid")
        try:
            output = [float(value) for value in flattened]
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(f"initial four-lane {field} matrix is invalid") from error
        if len(output) != len(flattened) or any(not math.isfinite(value) for value in output):
            raise ValueError(f"initial four-lane {field} matrix is invalid")
        return output

    try:
        route_offsets = integers("route_offsets")
        route_indices = integers("route_indices")
        path_offsets = integers("path_offsets")
        path_indices = integers("path_indices")
        statuses = integers("statuses")
        reasons = integers("reasons")
        metrics = float_matrix("metrics", 4)
        labels = integer_matrix("label_counters", 3)
        batch_counters = integers("batch_counters")
        completion_order = integers("completion_order")
        objective_integer = integers("objective_integer")
        objective_float = floats("objective_float")
        accounting = integers("accounting")
        rng_seeds = integers("rng_seeds")
        node_kind = integers("node_kind")
    except ValueError as error:
        return str(error)
    next_iteration = projection.get("next_iteration")
    lane_count = projection.get("lane_count")
    exact_batch_size = projection.get("exact_batch_size")
    if not is_int64(next_iteration) or not is_int64(lane_count) or not is_int64(exact_batch_size):
        return "initial four-lane scalar projection is invalid"
    assert isinstance(next_iteration, int)
    assert isinstance(lane_count, int)
    assert isinstance(exact_batch_size, int)
    next_iteration_value = int(next_iteration)
    lane_count_value = int(lane_count)
    exact_batch_size_value = int(exact_batch_size)
    if (
        len(route_offsets) < 2
        or route_offsets[0] != 0
        or route_offsets[-1] != len(route_indices)
        or any(left >= right for left, right in zip(route_offsets, route_offsets[1:], strict=False))
        or any(value < 0 for value in route_indices)
    ):
        return "initial four-lane route projection is invalid"
    route_count = len(route_offsets) - 1
    if (
        len(path_offsets) != route_count + 1
        or path_offsets[0] != 0
        or path_offsets[-1] != len(path_indices)
        or any(left >= right for left, right in zip(path_offsets, path_offsets[1:], strict=False))
        or any(value < 0 for value in path_indices)
        or len(statuses) != route_count
        or len(reasons) != route_count
        or statuses != [0] * route_count
        or reasons != [0] * route_count
        or len(metrics) != route_count * 4
        or any(value < 0.0 for value in metrics)
        or len(labels) != route_count * 3
        or any(value < 0 for value in labels)
        or len(batch_counters) != 10
        or sorted(completion_order) != list(range(route_count))
        or len(objective_integer) != 2
        or objective_integer[0] != route_count
        or any(value < 0 for value in objective_integer)
        or len(objective_float) != 2
        or any(value < 0.0 for value in objective_float)
        or accounting != [route_count, route_count, 0, 0]
        or rng_seeds != [expected_seed, expected_seed ^ 0x5EED23]
        or next_iteration_value != 0
        or lane_count_value != 4
        or not node_kind
        or any(kind not in {0, 1, 2} for kind in node_kind)
        or node_kind.count(0) != 1
        or exact_batch_size_value <= 0
        or node_kind != list(expected_node_kind)
        or exact_batch_size_value != expected_exact_batch_size
        or any(node >= len(node_kind) or node_kind[node] != 1 for node in route_indices)
        or sorted(route_indices) != [index for index, kind in enumerate(node_kind) if kind == 1]
        or projection.get("state_sha256") != expected_sha256
    ):
        return "initial four-lane projection values do not reconcile"
    depot = node_kind.index(0)
    charging_count = 0
    for route in range(route_count):
        path_first = path_offsets[route]
        path_last = path_offsets[route + 1]
        expected_customers = route_indices[route_offsets[route] : route_offsets[route + 1]]
        observed_customers: list[int] = []
        if (
            path_first < 0
            or path_first >= path_last
            or path_last > len(path_indices)
            or path_indices[path_first] != depot
            or path_indices[path_last - 1] != depot
        ):
            return "initial four-lane path projection is invalid"
        for position in range(path_first, path_last):
            node = path_indices[position]
            if node < 0 or node >= len(node_kind):
                return "initial four-lane path node is invalid"
            kind = node_kind[node]
            charging_count += int(kind == 2)
            if kind == 1:
                observed_customers.append(node)
            elif node == depot and position not in {path_first, path_last - 1}:
                return "initial four-lane path contains an interior depot"
        if observed_customers != expected_customers:
            return "initial four-lane customer path is invalid"
    total_distance = 0.0
    total_charging_time = 0.0
    for route in range(route_count):
        total_distance += metrics[route * 4]
        total_charging_time += metrics[route * 4 + 3]
    if (
        batch_counters[0:3] != [route_count] * 3
        or batch_counters[3] != 0
        or batch_counters[4] != 1
        or any(value < 0 for value in batch_counters[5:8])
        or batch_counters[8] != 1
        or batch_counters[9] != exact_batch_size_value
        or objective_integer != [route_count, charging_count]
        or objective_float != [total_distance, total_charging_time]
    ):
        return "initial four-lane exact/objective projection is invalid"

    def packed_integer(values: Sequence[int]) -> bytes:
        return struct.pack(f"<{len(values)}q", *values)

    def packed_float(values: Sequence[float]) -> bytes:
        return struct.pack(f"<{len(values)}d", *values)

    initial_state_evidence = bytearray(b"stage05.2-native-initial-search-state-v2")
    initial_state_evidence.extend(request_sha256.encode("ascii"))
    initial_arrays: tuple[tuple[Sequence[int] | Sequence[float], bytes], ...] = (
        (path_offsets, packed_integer(path_offsets)),
        (path_indices, packed_integer(path_indices)),
        (statuses, packed_integer(statuses)),
        (reasons, packed_integer(reasons)),
        (metrics, packed_float(metrics)),
        (labels, packed_integer(labels)),
        (batch_counters, packed_integer(batch_counters)),
        (completion_order, packed_integer(completion_order)),
        (objective_integer, packed_integer(objective_integer)),
        (objective_float, packed_float(objective_float)),
        (accounting, packed_integer(accounting)),
    )
    for values, raw in initial_arrays:
        initial_state_evidence.extend(struct.pack("<Q", len(values)))
        initial_state_evidence.extend(raw)
    if hashlib.sha256(initial_state_evidence).hexdigest() != initial_state_sha256:
        return "initial-state projection hash mismatch"

    lane_evidence = bytearray(b"stage05.2-native-lane-state-v2")
    arrays: tuple[tuple[Sequence[int] | Sequence[float], bytes], ...] = (
        (route_offsets, packed_integer(route_offsets)),
        (route_indices, packed_integer(route_indices)),
        (path_offsets, packed_integer(path_offsets)),
        (path_indices, packed_integer(path_indices)),
        (statuses, packed_integer(statuses)),
        (reasons, packed_integer(reasons)),
        (metrics, packed_float(metrics)),
        (labels, packed_integer(labels)),
        (batch_counters, packed_integer(batch_counters)),
        (completion_order, packed_integer(completion_order)),
        (objective_integer, packed_integer(objective_integer)),
        (objective_float, packed_float(objective_float)),
    )
    for values, raw in arrays:
        lane_evidence.extend(struct.pack("<Q", len(values)))
        lane_evidence.extend(raw)
    lane_sha256 = hashlib.sha256(lane_evidence).hexdigest()
    if projection.get("lane_sha256") != lane_sha256:
        return "initial four-lane lane hash mismatch"
    state_evidence = bytearray(b"stage05.2-native-initial-four-lane-state-v2")
    state_evidence.extend(request_sha256.encode("ascii"))
    state_evidence.extend(initial_state_sha256.encode("ascii"))
    state_evidence.extend(lane_sha256.encode("ascii") * 4)
    for values in (accounting, rng_seeds):
        state_evidence.extend(struct.pack("<Q", len(values)))
        state_evidence.extend(packed_integer(values))
    state_evidence.extend(struct.pack("<q", 0))
    state_evidence.extend(struct.pack("<Q", len(node_kind)))
    state_evidence.extend(packed_integer(node_kind))
    state_evidence.extend(struct.pack("<q", exact_batch_size_value))
    if hashlib.sha256(state_evidence).hexdigest() != expected_sha256:
        return "initial four-lane state hash mismatch"
    return None


def _common_prefix(left: Sequence[object], right: Sequence[object]) -> int:
    length = 0
    for left_event, right_event in zip(left, right, strict=False):
        if left_event != right_event:
            break
        length += 1
    return length


def _describe_first_divergence(
    baseline: Sequence[object],
    candidate: Sequence[object],
) -> dict[str, object] | None:
    index = _common_prefix(baseline, candidate)
    if index == len(baseline) == len(candidate):
        return None
    baseline_event = baseline[index] if index < len(baseline) else None
    candidate_event = candidate[index] if index < len(candidate) else None
    baseline_mapping = baseline_event if isinstance(baseline_event, dict) else {}
    candidate_mapping = candidate_event if isinstance(candidate_event, dict) else {}

    def coordinate_value(field: str) -> object:
        return (
            candidate_mapping[field] if field in candidate_mapping else baseline_mapping.get(field)
        )

    differing_fields: dict[str, object] = {}
    for field in sorted(set(baseline_mapping) | set(candidate_mapping)):
        baseline_present = field in baseline_mapping
        candidate_present = field in candidate_mapping
        baseline_value = baseline_mapping[field] if baseline_present else {"field_missing": True}
        candidate_value = candidate_mapping[field] if candidate_present else {"field_missing": True}
        if baseline_present != candidate_present or baseline_value != candidate_value:
            differing_fields[str(field)] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
            }
    return {
        "index": index,
        "lane": coordinate_value("lane"),
        "iteration": coordinate_value("iteration"),
        "operator": coordinate_value("operator"),
        "candidate_id": coordinate_value("candidate_id"),
        "differing_fields": differing_fields,
        "baseline": baseline_event,
        "candidate": candidate_event,
    }


def _semantic_trajectory(payload: Mapping[str, object]) -> list[object]:
    value = payload.get("semantic_trajectory")
    if value is not None:
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError("semantic_trajectory must be a list of event objects")
        quality_operators = {
            "relocate",
            "swap",
            "two_opt_star",
            "route_segment_destroy",
            "ejection_chain",
        }
        for ordinal, row in enumerate(value):
            operator = str(row.get("operator", ""))
            track = str(row.get("track", ""))
            lane = (
                "constraint_lane"
                if track == "constraint_lane"
                else "quality_shadow"
                if operator in quality_operators
                else "legacy"
            )
            if row.get("lane") != lane:
                raise ValueError("semantic candidate lane projection is inconsistent")
            identity = {
                "lane": lane,
                "iteration": row.get("iteration"),
                "operator": operator,
                "status": row.get("status"),
                "candidate_route_sequences": row.get("candidate_route_sequences", ()),
                "candidate_objective_key": row.get("candidate_objective_key", ()),
                "ordinal": ordinal,
            }
            expected_candidate_id = hashlib.sha256(
                json.dumps(
                    identity,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            if row.get("candidate_id") != expected_candidate_id:
                raise ValueError(
                    "semantic candidate_id does not match its canonical route projection"
                )
        return list(value)
    # Stage 5.2 comparison v3 retained only a count/hash summary. Keep it
    # readable for immutable attempt03 evidence, but do not pretend that it can
    # identify an event-level divergence.
    legacy = _mapping(payload, "trajectory")
    return [dict(legacy)]


_SEMANTIC_STREAM_NAMES = SEMANTIC_STREAM_NAMES


def _canonical_semantic_events(payload: Mapping[str, object]) -> list[dict[str, object]]:
    raw_streams = payload.get("canonical_semantic_streams")
    if not isinstance(raw_streams, dict) or set(raw_streams) != set(_SEMANTIC_STREAM_NAMES):
        raise ValueError("canonical semantic stream set is incomplete")
    streams: dict[str, list[dict[str, object]]] = {}
    for stream_name in _SEMANTIC_STREAM_NAMES:
        stream = raw_streams[stream_name]
        if not isinstance(stream, list) or not all(isinstance(row, dict) for row in stream):
            raise ValueError(f"canonical semantic stream {stream_name} is invalid")
        streams[stream_name] = [dict(row) for row in stream]
    screening_semantic_fields = {
        "event_type",
        "lane",
        "iteration",
        "operator",
        "candidate_id",
        "plan_id",
        "route_key",
        "status",
        "decision",
        "reason",
        "first_failed_check",
        "screening_passed",
        "exact_call_blocked",
        "negative_cache_hit",
        "cache_hit",
    }
    for event in streams["screening"]:
        if not screening_semantic_fields.intersection(event):
            raise ValueError("screening event lacks auditable semantic fields")
        raw_checks = event.get("checks")
        if raw_checks is not None and (
            not isinstance(raw_checks, list)
            or not all(isinstance(check, dict) for check in raw_checks)
        ):
            raise ValueError("screening checks are invalid")
    raw_events = payload.get("canonical_semantic_events")
    if not isinstance(raw_events, list) or not all(isinstance(row, dict) for row in raw_events):
        raise ValueError("explicit canonical semantic event sequence is missing")
    events = [dict(row) for row in raw_events]
    if not events:
        raise ValueError("canonical semantic event sequence cannot be empty")
    if len(events) != sum(len(rows) for rows in streams.values()):
        raise ValueError("canonical semantic event sequence has missing or duplicate rows")
    sequences = [event.get("semantic_sequence") for event in events]
    if sequences != list(range(len(events))):
        raise ValueError("canonical semantic sequence IDs are not contiguous")
    projected_streams: dict[str, list[dict[str, object]]] = {
        name: [] for name in _SEMANTIC_STREAM_NAMES
    }
    for event in events:
        raw_stream_name = event.get("semantic_stream")
        if not isinstance(raw_stream_name, str) or raw_stream_name not in projected_streams:
            raise ValueError("canonical semantic event names an unknown stream")
        projected_streams[raw_stream_name].append(
            {
                key: value
                for key, value in event.items()
                if key not in {"semantic_stream", "semantic_sequence"}
            }
        )
    if projected_streams != streams:
        raise ValueError("canonical semantic event sequence does not preserve every stream journal")
    runtime_event_ids = [event.get("semantic_event_id") for event in events]
    if runtime_event_ids != list(range(1, len(events) + 1)):
        raise ValueError("canonical semantic event sequence lacks contiguous runtime event IDs")
    mode = payload.get("mode")
    if isinstance(mode, str):
        source_runtime_ids: list[int] = []
        next_native_event_id = 1
        native_backed_count = 0
        native_fields = set(_NATIVE_CANONICAL_TELEMETRY_FIELDS)
        native_canonical_projection = _NativeCanonicalProjectionHasher()
        for event in events:
            event_id = event.get("runtime_event_id")
            if isinstance(event_id, bool) or not isinstance(event_id, int):
                raise ValueError("canonical semantic events lost strict runtime source order")
            source_runtime_ids.append(event_id)
            raw_native = event.get("native_telemetry")
            if raw_native is None:
                if any(key.startswith("runtime_native_") for key in event):
                    raise ValueError("control-derived semantic row leaks native telemetry")
                continue
            if mode not in FULL_NATIVE_SEMANTIC_MODES:
                raise ValueError("non-full-native semantic row carries native event telemetry")
            if (
                not isinstance(raw_native, dict)
                or set(raw_native) != native_fields
                or any(
                    isinstance(raw_native.get(field), bool)
                    or not isinstance(raw_native.get(field), int)
                    for field in native_fields
                )
            ):
                raise ValueError("native semantic telemetry fields are incomplete")
            if raw_native["runtime_native_event_id"] != next_native_event_id:
                raise ValueError("native semantic event IDs are not contiguous")
            next_native_event_id += 1
            native_backed_count += 1
            native_canonical_projection.append(cast(Mapping[str, object], raw_native))
        if mode in FULL_NATIVE_SEMANTIC_MODES and native_backed_count == 0:
            raise ValueError("full-native semantic journal has no native-backed events")
        if mode in FULL_NATIVE_SEMANTIC_MODES:
            native_receipt_error = _native_canonical_receipt_error(
                payload,
                native_canonical_projection,
            )
            if native_receipt_error is not None:
                raise ValueError(native_receipt_error)
        if any(
            source_runtime_ids[index - 1] >= source_runtime_ids[index]
            for index in range(1, len(source_runtime_ids))
        ):
            raise ValueError("canonical semantic events lost strict runtime source order")
        required_nonempty = {
            "stage04",
            "exact_work",
            "exact_result",
            "cache",
            "screening",
            "termination",
        }
        effective_iterations = payload.get("effective_iterations")
        if not (
            isinstance(effective_iterations, int)
            and not isinstance(effective_iterations, bool)
            and effective_iterations == 0
        ):
            required_nonempty.update({"candidate_state", "operator"})
        if mode != "current_stage052":
            required_nonempty.add("candidate_transaction")
        missing = sorted(name for name in required_nonempty if not streams[name])
        if missing:
            raise ValueError(
                "canonical semantic runtime journal is incomplete: " + ", ".join(missing)
            )
        termination = streams["termination"]
        allowed_termination = {
            "candidate_control_exhausted",
            "exact_call_budget_exhausted",
            "initialization_failed",
            "iteration_limit",
            "wall_clock_deadline",
            "watchdog_exhausted",
        }
        declared_termination = termination[0].get("status") if len(termination) == 1 else None
        if (
            len(termination) != 1
            or termination[0].get("event_type") != "termination"
            or declared_termination not in allowed_termination
            or events[-1].get("semantic_stream") != "termination"
        ):
            raise ValueError("canonical semantic termination is missing or inconsistent")
        if declared_termination != "iteration_limit" and not streams["deadline"]:
            raise ValueError("canonical semantic deadline boundary is missing")
        if "axis" in payload:
            expected_terminal_fields = {
                "iterations": payload.get("iterations"),
                "effective_iterations": payload.get("effective_iterations"),
                "exact_started_calls": payload.get("exact_started_calls"),
                "exact_completed_calls": payload.get("exact_completed_calls"),
                "exact_interrupted_calls": payload.get("exact_interrupted_calls"),
                "objective_key": payload.get("objective"),
            }
            if any(
                termination[0].get(field) != expected
                for field, expected in expected_terminal_fields.items()
            ):
                raise ValueError("canonical semantic termination counters do not reconcile")
    declared_started = payload.get("exact_started_calls")
    declared_completed = payload.get("exact_completed_calls")
    if (
        isinstance(declared_started, int)
        and not isinstance(declared_started, bool)
        and isinstance(declared_completed, int)
        and not isinstance(declared_completed, bool)
    ):
        journal_started = 0
        for event in streams["exact_work"]:
            started_calls = event.get("started_calls")
            if (
                event.get("event_type") == "exact_batch_started"
                and isinstance(started_calls, int)
                and not isinstance(started_calls, bool)
            ):
                journal_started += started_calls
        result_started = sum(
            event.get("exact_started") is True for event in streams["exact_result"]
        )
        result_completed = sum(
            event.get("exact_completed") is True for event in streams["exact_result"]
        )
        if journal_started != declared_started or result_started != declared_started:
            raise ValueError("semantic exact-start counters do not reconcile")
        if result_completed != declared_completed:
            raise ValueError("semantic exact-completion counters do not reconcile")
    return [
        {key: value for key, value in event.items() if key != "runtime_event_id"}
        for event in events
    ]


def _route_sequence_from_canonical_key(route_key: str) -> tuple[str, ...]:
    prefix = "route:"
    if not route_key.startswith(prefix):
        raise ValueError("physical screening route key is invalid")
    encoded = route_key[len(prefix) :]
    if not encoded:
        return ()
    sequence: list[str] = []
    for token in encoded.split("|"):
        length_text, separator, customer = token.partition(":")
        if not separator or not length_text.isdigit() or int(length_text) != len(customer):
            raise ValueError("physical screening route key is invalid")
        sequence.append(customer)
    return tuple(sequence)


def _replay_physical_screening(
    payload: Mapping[str, object],
    instance: Instance,
    *,
    axis_path: Path | None = None,
) -> str | None:
    """Independently recompute every safe physical screening decision."""

    if payload.get("schema_version") in EXTERNAL_SEMANTIC_COMPARISON_SCHEMA_VERSIONS:
        if axis_path is None:
            return "physical screening journal axis path is missing"
        descriptor = payload.get("canonical_semantic_journal")
        if not isinstance(descriptor, dict):
            return "physical screening journal descriptor is missing"
        raw_stream_counts = descriptor.get("stream_counts")
        if not isinstance(raw_stream_counts, dict):
            return "physical screening journal stream counts are missing"
        expected_screening_count = raw_stream_counts.get("screening")
        if (
            isinstance(expected_screening_count, bool)
            or not isinstance(expected_screening_count, int)
            or expected_screening_count <= 0
        ):
            return "physical screening journal count is invalid"

        def journal_screening() -> Iterable[object]:
            termination: Mapping[str, object] | None = None
            for event in iter_verified_semantic_journal(axis_path, descriptor):
                if event.get("semantic_stream") == "termination":
                    termination = event
                if event.get("semantic_stream") == "screening":
                    yield event
            if (
                termination is None
                or termination.get("event_type") != "termination"
                or termination.get("status") != payload.get("termination_reason")
                or termination.get("iterations") != payload.get("iterations")
                or termination.get("effective_iterations") != payload.get("effective_iterations")
                or termination.get("exact_started_calls") != payload.get("exact_started_calls")
                or termination.get("exact_completed_calls") != payload.get("exact_completed_calls")
                or termination.get("exact_interrupted_calls")
                != payload.get("exact_interrupted_calls")
                or termination.get("objective_key") != payload.get("objective")
            ):
                raise ValueError("semantic journal termination does not reconcile")

        raw_screening = journal_screening()
    else:
        raw_streams = payload.get("canonical_semantic_streams")
        if not isinstance(raw_streams, dict):
            return "physical screening stream set is missing"
        inline_screening = raw_streams.get("screening")
        if not isinstance(inline_screening, list):
            return "physical screening stream is missing"
        if not inline_screening:
            return "physical screening stream is empty"
        expected_screening_count = len(inline_screening)
        raw_screening = inline_screening
    raw_statistics = payload.get("screening_statistics")
    if not isinstance(raw_statistics, dict):
        return "physical screening statistics are missing"
    screening_calls = raw_statistics.get("screening_calls")
    screening_passes = raw_statistics.get("screening_passes")
    screening_rejections = raw_statistics.get("screening_rejections")
    screening_cache_hits = raw_statistics.get("screening_cache_hits")
    screening_blocked = raw_statistics.get("screening_exact_call_blocked")
    screening_counters = (
        screening_calls,
        screening_passes,
        screening_rejections,
        screening_cache_hits,
        screening_blocked,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in screening_counters
    ):
        return "physical screening statistics are invalid"
    assert isinstance(screening_calls, int)
    assert isinstance(screening_passes, int)
    assert isinstance(screening_rejections, int)
    assert isinstance(screening_cache_hits, int)
    assert isinstance(screening_blocked, int)
    observed_screening_count = 0
    observed_decision_count = 0
    observed_aggregate_count = 0
    observed_native_decisions = 0
    observed_native_passes = 0
    observed_native_rejections = 0
    observed_native_physical = 0
    observed_native_owners = 0
    observed_native_cache_hits = 0
    observed_native_reachability_queries = 0
    observed_native_reason_counts: dict[str, int] = {}
    for ordinal, raw_event in enumerate(raw_screening):
        observed_screening_count += 1
        if not isinstance(raw_event, dict):
            return f"physical screening row {ordinal} is invalid"
        if raw_event.get("event_type") == "candidate_screening_aggregate":
            observed_aggregate_count += 1
            aggregate_counters = tuple(
                raw_event.get(field)
                for field in (
                    "calls",
                    "passes",
                    "rejections",
                    "cache_hits",
                    "exact_call_blocked",
                )
            )
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in aggregate_counters
            ):
                return f"physical screening aggregate {ordinal} counters are invalid"
            calls, passes, rejections, cache_hits, blocked = aggregate_counters
            assert isinstance(calls, int)
            assert isinstance(passes, int)
            assert isinstance(rejections, int)
            assert isinstance(cache_hits, int)
            assert isinstance(blocked, int)
            if passes + blocked != calls or rejections + cache_hits != blocked:
                return f"physical screening aggregate {ordinal} does not conserve calls"
            reason_counts = raw_event.get("reason_counts")
            if not isinstance(reason_counts, dict) or any(
                not isinstance(reason, str)
                or not reason
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                for reason, count in reason_counts.items()
            ):
                return f"physical screening aggregate {ordinal} reasons are invalid"
            pool_hashes = (
                raw_event.get("candidate_pool_hash"),
                raw_event.get("screening_pool_hash"),
            )
            if not any(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in pool_hashes
            ):
                return f"physical screening aggregate {ordinal} pool hash is invalid"
            continue
        if raw_event.get("event_type") != "screening_decision":
            return f"physical screening row {ordinal} has an unknown event type"
        observed_decision_count += 1
        try:
            route_key = _string(raw_event, "route_key")
            sequence = _route_sequence_from_canonical_key(route_key)
            replayed = screen_route_candidate(instance, sequence, full=True)
        except (KeyError, TypeError, ValueError) as error:
            return f"physical screening row {ordinal} cannot be replayed: {error}"
        status = raw_event.get("status")
        negative_cache_hit = raw_event.get("negative_cache_hit")
        exact_call_blocked = raw_event.get("exact_call_blocked")
        if not isinstance(negative_cache_hit, bool) or not isinstance(exact_call_blocked, bool):
            return f"physical screening row {ordinal} flags are invalid"
        native_telemetry = raw_event.get("native_telemetry")
        if isinstance(native_telemetry, dict):
            native_fields = tuple(
                native_telemetry.get(field)
                for field in (
                    "runtime_native_lane_id",
                    "runtime_native_operator_id",
                    "runtime_native_iteration",
                    "runtime_native_transaction_id",
                    "runtime_native_subject_id",
                    "runtime_native_status_code",
                    "runtime_native_flags",
                )
            )
            if any(
                isinstance(value, bool) or not isinstance(value, int) for value in native_fields
            ):
                return f"physical screening row {ordinal} native telemetry is invalid"
            native_status_code = native_fields[-2]
            native_flags = native_fields[-1]
            assert isinstance(native_status_code, int)
            assert isinstance(native_flags, int)
            if native_flags < 0 or native_status_code not in {0, 1}:
                return f"physical screening row {ordinal} native codes are invalid"
            observed_native_decisions += 1
            observed_native_physical += int(bool(native_flags & 1))
            observed_native_cache_hits += int(bool(native_flags & 2))
            observed_native_owners += int(bool(native_flags & 4))
            observed_native_reachability_queries += native_flags >> 8
            if status == "pass":
                observed_native_passes += 1
                if native_status_code != 1:
                    return f"physical screening row {ordinal} native status is invalid"
            else:
                observed_native_rejections += 1
                if native_status_code != 0:
                    return f"physical screening row {ordinal} native status is invalid"
                reason = raw_event.get("reason")
                if not isinstance(reason, str) or not reason:
                    return f"physical screening row {ordinal} native reason is missing"
                observed_native_reason_counts[reason] = (
                    observed_native_reason_counts.get(reason, 0) + 1
                )
        if negative_cache_hit and isinstance(native_telemetry, dict):
            native_flags = native_telemetry.get("runtime_native_flags")
            if (
                status != "negative_cache_hit"
                or not exact_call_blocked
                or replayed.accepted
                or raw_event.get("reason") != replayed.reason
                or isinstance(native_flags, bool)
                or not isinstance(native_flags, int)
                or native_flags & 2 == 0
                or raw_event.get("checks") != []
                or any(
                    raw_event.get(field) not in {0, None}
                    for field in (
                        "demand",
                        "min_time_window_slack",
                        "distance_lower_bound",
                        "distance_increment_lower_bound",
                        "single_segment_reachable",
                        "structural_energy_lower_bound",
                    )
                )
            ):
                return f"physical screening row {ordinal} native cache replay mismatch"
            continue
        if isinstance(native_telemetry, dict) and raw_event.get("checks") == []:
            replay_reason = replayed.reason
            native_reason = raw_event.get("reason")
            compatible_reason = (
                native_reason == replay_reason
                or (
                    native_reason == "time_window_prefilter"
                    and replay_reason
                    in {
                        "forward_time_window_prefilter",
                        "backward_time_window_prefilter",
                        "time_window_slack_prefilter",
                    }
                )
                or (
                    native_reason == "energy_prefilter"
                    and replay_reason
                    in {
                        "single_segment_energy_prefilter",
                        "structural_energy_prefilter",
                    }
                )
            )
            if (
                (status == "pass") is not replayed.accepted
                or exact_call_blocked is replayed.accepted
                or (not replayed.accepted and not compatible_reason)
            ):
                return f"physical screening row {ordinal} native decision replay mismatch"
            continue
        expected_status = (
            "negative_cache_hit"
            if negative_cache_hit
            else "pass"
            if replayed.accepted
            else "rejected"
        )
        if (
            status != expected_status
            or exact_call_blocked is replayed.accepted
            or raw_event.get("reason") != replayed.reason
            or raw_event.get("first_failed_check") != replayed.first_failed_check
        ):
            return f"physical screening row {ordinal} decision replay mismatch"
        raw_checks = raw_event.get("checks")
        if not isinstance(raw_checks, list) or not all(
            isinstance(check, dict) for check in raw_checks
        ):
            return f"physical screening row {ordinal} checks are invalid"
        expected_checks: tuple[dict[str, object], ...]
        if negative_cache_hit:
            expected_checks = (
                {
                    "check": "negative_sequence_cache",
                    "status": "hit",
                    "value": True,
                    "reason": ("reused a previously recorded safe screening rejection"),
                },
            )
        else:
            expected_checks = tuple(
                {
                    "check": check.check,
                    "status": check.status,
                    "value": check.value,
                    "reason": check.reason,
                }
                for check in replayed.checks
            )
        if len(raw_checks) != len(expected_checks):
            return f"physical screening row {ordinal} check replay mismatch"
        for observed_check, expected_check in zip(raw_checks, expected_checks, strict=True):
            if (
                observed_check.get("check") != expected_check["check"]
                or observed_check.get("status") != expected_check["status"]
                or observed_check.get("reason", "") != expected_check["reason"]
            ):
                return f"physical screening row {ordinal} check replay mismatch"
            observed_value = observed_check.get("value")
            expected_value = expected_check["value"]
            if isinstance(expected_value, bool) or expected_value is None:
                if observed_value is not expected_value:
                    return f"physical screening row {ordinal} check replay mismatch"
            elif (
                isinstance(expected_value, bool)
                or not isinstance(expected_value, int | float)
                or isinstance(observed_value, bool)
                or not isinstance(observed_value, int | float)
                or not math.isfinite(float(observed_value))
                or not math.isclose(
                    float(observed_value),
                    float(expected_value),
                    rel_tol=1e-9,
                    abs_tol=1e-7,
                )
            ):
                return f"physical screening row {ordinal} check replay mismatch"
        expected_metrics = {
            "demand": replayed.demand,
            "min_time_window_slack": replayed.min_time_window_slack,
            "distance_lower_bound": replayed.distance_lower_bound,
            "structural_energy_lower_bound": replayed.structural_energy_lower_bound,
        }
        for field, expected in expected_metrics.items():
            observed = raw_event.get(field)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, int | float)
                or not math.isfinite(float(observed))
                or not math.isclose(float(observed), float(expected), rel_tol=1e-9, abs_tol=1e-7)
            ):
                return f"physical screening row {ordinal} metric replay mismatch: {field}"
        if raw_event.get("single_segment_reachable") is not (replayed.single_segment_reachable):
            return f"physical screening row {ordinal} reachability replay mismatch"
        distance_increment = raw_event.get("distance_increment_lower_bound")
        if distance_increment is not None and (
            isinstance(distance_increment, bool)
            or not isinstance(distance_increment, int | float)
            or not math.isfinite(float(distance_increment))
        ):
            return f"physical screening row {ordinal} increment bound is invalid"
    if observed_screening_count != expected_screening_count:
        return "physical screening journal count mismatch"
    physical_owner_count = raw_statistics.get("screening_physical_owner_count")
    if physical_owner_count is not None:
        semantic_decisions = raw_statistics.get("screening_semantic_decisions")
        semantic_event_count = raw_statistics.get("screening_semantic_event_count")
        semantic_passes = raw_statistics.get("screening_semantic_passes")
        semantic_rejections = raw_statistics.get("screening_semantic_rejections")
        native_counters = (
            physical_owner_count,
            semantic_decisions,
            semantic_event_count,
            semantic_passes,
            semantic_rejections,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in native_counters
        ):
            return "native physical screening counters are invalid"
        assert isinstance(physical_owner_count, int)
        assert isinstance(semantic_decisions, int)
        assert isinstance(semantic_event_count, int)
        assert isinstance(semantic_passes, int)
        assert isinstance(semantic_rejections, int)
        if (
            physical_owner_count != screening_calls
            or screening_passes + screening_rejections != screening_calls
            or semantic_passes + semantic_rejections != semantic_decisions
            or semantic_event_count != semantic_decisions
            or semantic_event_count != observed_decision_count
            or observed_aggregate_count != 0
            or observed_native_decisions != semantic_decisions
            or observed_native_passes != semantic_passes
            or observed_native_rejections != semantic_rejections
            or observed_native_owners != physical_owner_count
            or observed_native_physical < observed_native_owners
            or observed_native_cache_hits != screening_cache_hits
            or observed_native_reachability_queries
            != raw_statistics.get("station_reachability_queries")
            or observed_native_reason_counts != raw_statistics.get("screening_reason_counts")
        ):
            return "native physical screening statistics do not reconcile"
    elif (
        screening_passes + screening_blocked != screening_calls
        or screening_rejections + screening_cache_hits != screening_blocked
    ):
        return "physical screening statistics do not conserve calls"
    return None


def _comparison_semantic_events(payload: Mapping[str, object]) -> Sequence[object]:
    if payload.get("schema_version") in {
        INLINE_SEMANTIC_COMPARISON_SCHEMA_VERSION,
        PREVIOUS_COMPARISON_SCHEMA_VERSION,
        LEGACY_PROFILE_COMPARISON_SCHEMA_VERSION,
        PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION,
        PROCESS_PROFILE_COMPARISON_SCHEMA_VERSION,
        TICK_PROFILE_COMPARISON_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }:
        # First validate the complete, implementation-owned causal journal.
        # Cross-architecture equality must then use only the shared logical
        # projection: native batching legitimately changes the number and
        # global interleaving of screening, cache, transaction and exact-work
        # telemetry rows.  Those streams remain mandatory and are reconciled
        # independently by _canonical_semantic_events plus the dedicated
        # transaction/cache/deadline gates below.
        _canonical_semantic_events(payload)
        raw_streams = payload.get("canonical_semantic_streams")
        if not isinstance(raw_streams, dict):
            raise ValueError("canonical semantic stream set is missing")
        projection: list[dict[str, object]] = []

        def screening_projection() -> list[dict[str, object]]:
            # The complete physical screening stream was validated above.
            # Cross-mode equality uses the candidate-level logical decision:
            # native batching may screen a larger physical proposal pool, and
            # negative-cache hits may occur at different physical call sites,
            # without changing which canonical candidate is proposed, skipped,
            # accepted, or rejected.  The trajectory binds that decision to
            # stable candidate, lane, iteration, operator, route, and reason
            # fields while the raw stream remains independently auditable.
            return [
                {
                    "event_type": "logical_screening_decision",
                    **{
                        field: event[field]
                        for field in (
                            "candidate_id",
                            "lane",
                            "iteration",
                            "operator",
                            "status",
                            "reason",
                            "candidate_route_sequences",
                        )
                        if field in event
                    },
                }
                for event in _semantic_trajectory(payload)
                if isinstance(event, dict)
            ]

        def append(stream_name: str, event: Mapping[str, object]) -> None:
            normalized = {
                key: value
                for key, value in event.items()
                if key
                not in {
                    "runtime_event_id",
                    "semantic_event_id",
                    "stream_ordinal",
                    "semantic_sequence",
                }
            }
            if stream_name == "candidate_state":
                # Candidate-state is the shared lane decision.  Attempted or
                # fully charged route-key retention and explanatory wording
                # are implementation telemetry; routes/objectives and exact
                # order are verified by dedicated gates.
                normalized = {
                    key: normalized.get(key)
                    for key in (
                        "event_type",
                        "lane",
                        "iteration",
                        "operator",
                        "candidate_objective_key",
                        "candidate_feasible",
                        "accepted",
                        "status",
                    )
                }
                normalized["candidate_feasible"] = bool(normalized.get("candidate_objective_key"))
                normalized["candidate_id"] = hashlib.sha256(
                    json.dumps(
                        {
                            "lane": normalized.get("lane"),
                            "iteration": normalized.get("iteration"),
                            "operator": normalized.get("operator"),
                            "candidate_objective_key": normalized.get(
                                "candidate_objective_key", []
                            ),
                            "candidate_feasible": normalized.get("candidate_feasible"),
                            "accepted": normalized.get("accepted"),
                            "status": normalized.get("status"),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
            elif stream_name == "deadline":
                normalized["event_type"] = "termination_boundary"
            projection.append(
                {
                    **normalized,
                    "semantic_stream": stream_name,
                    "projection_ordinal": len(projection),
                }
            )

        # Candidate decisions come from the canonical operator projection,
        # which removes implementation-only aggregate pruning telemetry and
        # supplies stable lane/candidate identities for first-divergence
        # diagnostics.
        for event in _semantic_trajectory(payload):
            if not isinstance(event, dict):
                raise ValueError("semantic trajectory contains a non-object event")
            append("operator", event)
        for event in screening_projection():
            append("screening", event)
        for stream_name in (
            "candidate_state",
            "stage04",
            "exact_result",
            "deadline",
            "termination",
            "native_failure",
        ):
            stream = raw_streams.get(stream_name)
            if not isinstance(stream, list) or not all(isinstance(event, dict) for event in stream):
                raise ValueError(f"canonical semantic stream {stream_name} is invalid")
            for event in stream:
                append(stream_name, event)
        return projection
    return _semantic_trajectory(payload)


def _comparison_semantic_events_from_artifact(
    record: ReviewRecord,
) -> Sequence[object]:
    full_payload = _verify_signed_json(record.path)
    if full_payload.get("schema_version") in EXTERNAL_SEMANTIC_COMPARISON_SCHEMA_VERSIONS:
        descriptor = full_payload.get("canonical_semantic_journal")
        if not isinstance(descriptor, dict):
            raise ValueError("canonical semantic journal descriptor is missing")
        selected_names = (
            "candidate_state",
            "stage04",
            "exact_result",
            "deadline",
            "termination",
            "native_failure",
        )
        selected: dict[str, list[dict[str, object]]] = {name: [] for name in selected_names}
        for event in iter_verified_semantic_journal(record.path, descriptor):
            stream_name = event.get("semantic_stream")
            if isinstance(stream_name, str) and stream_name in selected:
                selected[stream_name].append(event)
        return _comparison_projection_from_selected(full_payload, selected)
    return _comparison_semantic_events(full_payload)


def _comparison_projection_from_selected(
    payload: Mapping[str, object],
    selected: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    """Build the bounded cross-mode projection from verified journal rows."""

    projection: list[dict[str, object]] = []

    def append(stream_name: str, event: Mapping[str, object]) -> None:
        normalized = {
            key: value
            for key, value in event.items()
            if key
            not in {
                "runtime_event_id",
                "semantic_event_id",
                "stream_ordinal",
                "semantic_sequence",
                "semantic_stream",
                "native_telemetry",
            }
        }
        if stream_name == "candidate_state":
            normalized = {
                key: normalized.get(key)
                for key in (
                    "event_type",
                    "lane",
                    "iteration",
                    "operator",
                    "candidate_objective_key",
                    "candidate_feasible",
                    "accepted",
                    "status",
                )
            }
            normalized["candidate_feasible"] = bool(normalized.get("candidate_objective_key"))
            normalized["candidate_id"] = hashlib.sha256(
                json.dumps(
                    {
                        "lane": normalized.get("lane"),
                        "iteration": normalized.get("iteration"),
                        "operator": normalized.get("operator"),
                        "candidate_objective_key": normalized.get("candidate_objective_key", []),
                        "candidate_feasible": normalized.get("candidate_feasible"),
                        "accepted": normalized.get("accepted"),
                        "status": normalized.get("status"),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        elif stream_name == "deadline":
            normalized["event_type"] = "termination_boundary"
        elif stream_name == "exact_result":
            normalized.pop("evaluation_id", None)
        projection.append(
            {
                **normalized,
                "semantic_stream": stream_name,
                "projection_ordinal": len(projection),
            }
        )

    trajectory = _semantic_trajectory(payload)
    for event in trajectory:
        if not isinstance(event, dict):
            raise ValueError("semantic trajectory contains a non-object event")
        append("operator", event)
    for event in trajectory:
        if not isinstance(event, dict):
            raise ValueError("semantic trajectory contains a non-object event")
        append(
            "screening",
            {
                "event_type": "logical_screening_decision",
                **{
                    field: event[field]
                    for field in (
                        "candidate_id",
                        "lane",
                        "iteration",
                        "operator",
                        "status",
                        "reason",
                        "candidate_route_sequences",
                    )
                    if field in event
                },
            },
        )
    for stream_name in (
        "candidate_state",
        "stage04",
        "exact_result",
        "deadline",
        "termination",
        "native_failure",
    ):
        stream = selected.get(stream_name)
        if stream is None:
            raise ValueError(f"canonical semantic stream {stream_name} is missing")
        for event in stream:
            append(stream_name, event)
    return projection


def _sample_quantile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0.0 <= fraction <= 1.0:
        raise ValueError("sample quantile requires values and a valid fraction")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired(values: Iterable[float]) -> dict[str, float | int | None]:
    items = tuple(values)
    return {
        "count": len(items),
        "median": statistics.median(items) if items else None,
        "p50": _sample_quantile(items, 0.50) if items else None,
        "p95": _sample_quantile(items, 0.95) if items else None,
        "p99": _sample_quantile(items, 0.99) if items else None,
        "minimum": min(items) if items else None,
        "maximum": max(items) if items else None,
    }


def _relative_time_improvement(candidate_seconds: float, baseline_seconds: float) -> float:
    """Return the fractional E2E time saved by a candidate."""

    if (
        not math.isfinite(candidate_seconds)
        or not math.isfinite(baseline_seconds)
        or candidate_seconds <= 0.0
        or baseline_seconds <= 0.0
    ):
        raise ValueError("relative time improvement requires positive finite timings")
    return 1.0 - candidate_seconds / baseline_seconds


def _family(instance: str) -> str:
    lowered = instance.lower()
    if lowered.startswith("rc"):
        return "RC"
    if lowered.startswith("r"):
        return "R"
    return "C"


def _vehicle_count(payload: Mapping[str, object]) -> float:
    objective = _sequence(payload, "objective")
    if not objective:
        raise ValueError("objective must contain vehicle count")
    value = objective[0]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("objective vehicle count must be numeric")
    return float(value)


def _producer_pre_receipt_seconds(record: ReviewRecord) -> float:
    """Return the producer lower bound before its terminal receipt publication."""

    if record.payload.get("schema_version") in PROFILE_COMPARISON_SCHEMA_VERSIONS:
        return _number(record.payload, "producer_pre_receipt_seconds")
    return _number(record.payload, "solver_seconds") + _number(
        record.payload, "persistence_seconds"
    )


def _producer_parent_terminal_seconds(record: ReviewRecord) -> float:
    """Return parent-observed completion after the worker's terminal publication."""

    if record.payload.get("schema_version") not in PROFILE_COMPARISON_SCHEMA_VERSIONS:
        return _producer_pre_receipt_seconds(record)
    wave = record.mode_wave
    if wave is None:
        raise ValueError("current axis lacks parent terminal timing evidence")
    raw_timings = _sequence(wave, "axis_parent_terminal_timings")
    matches: list[Mapping[str, object]] = []
    for raw_timing in raw_timings:
        if not isinstance(raw_timing, Mapping):
            raise ValueError("axis parent terminal timing row must be an object")
        try:
            timing_key = (
                _integer(raw_timing, "repeat"),
                _string(raw_timing, "axis"),
                _string(raw_timing, "instance"),
                _integer(raw_timing, "seed"),
            )
        except ValueError as error:
            raise ValueError("axis parent terminal timing identity is invalid") from error
        if timing_key == record.key:
            matches.append(raw_timing)
    if len(matches) != 1:
        raise ValueError("axis parent terminal timing identity is ambiguous")
    terminal = _number(matches[0], "producer_parent_terminal_seconds")
    if terminal + 1e-9 < _producer_pre_receipt_seconds(record):
        raise ValueError("parent terminal timing omits producer publication")
    return terminal


def _axis_end_to_end_seconds(record: ReviewRecord, replay_seconds: float) -> float:
    """Return complete per-axis producer terminal plus independent replay time."""

    if not math.isfinite(replay_seconds) or replay_seconds < 0.0:
        raise ValueError("axis replay timing must be non-negative and finite")
    return _producer_parent_terminal_seconds(record) + replay_seconds


def _topology_integer(
    payload: Mapping[str, object],
    field: str,
    *,
    legacy_field: str,
) -> int:
    topology = _mapping(payload, "topology")
    selected = field if field in topology else legacy_field
    return _integer(topology, selected)


def _numeric_values(
    rows: Iterable[Mapping[str, object]],
    field_name: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field_name)
        if (
            not isinstance(value, bool)
            and isinstance(value, int | float)
            and math.isfinite(float(value))
        ):
            values.append(float(value))
    return values


def _mode_wave_metrics(records: Sequence[ReviewRecord]) -> dict[str, object]:
    waves: list[Mapping[str, object]] = []
    seen: set[int] = set()
    for record in records:
        wave = record.mode_wave
        if wave is None or id(wave) in seen:
            continue
        seen.add(id(wave))
        waves.append(wave)

    scheduler_statistics: list[Mapping[str, object]] = []
    request_queues: list[Mapping[str, object]] = []
    work_queues: list[Mapping[str, object]] = []
    cgroup_after: list[Mapping[str, object]] = []
    io_accounting: list[Mapping[str, object]] = []
    schedstat: list[Mapping[str, object]] = []
    scheduler_accounting: list[Mapping[str, object]] = []
    for wave in waves:
        raw_scheduler = wave.get("scheduler_runtime_statistics")
        if isinstance(raw_scheduler, list):
            scheduler_statistics.extend(item for item in raw_scheduler if isinstance(item, Mapping))
        raw_after = wave.get("cgroup_after")
        if isinstance(raw_after, Mapping):
            cgroup_after.append(raw_after)
        raw_io = wave.get("io_accounting")
        if isinstance(raw_io, Mapping):
            io_accounting.append(raw_io)
        else:
            raw_legacy_io = wave.get("cgroup_io_deltas")
            if isinstance(raw_legacy_io, Mapping):
                io_accounting.append(
                    {"source": CGROUP_IO_ACCOUNTING_SOURCE, **raw_legacy_io}
                )
        use_process_accounting = (
            wave.get("resource_summary_accounting_source")
            == MODE_WAVE_RESOURCE_ACCOUNTING_SOURCE
        )
        raw_schedstat = wave.get(
            "process_tree_schedstat" if use_process_accounting else "thread_tree_schedstat"
        )
        if isinstance(raw_schedstat, Mapping):
            schedstat.append(raw_schedstat)
        scheduler_accounting.append(
            {
                "context_switches": wave.get(
                    "process_tree_context_switches"
                    if use_process_accounting
                    else "thread_tree_context_switches"
                ),
                "cpu_migrations": wave.get(
                    "process_tree_cpu_migrations"
                    if use_process_accounting
                    else "thread_tree_cpu_migrations"
                ),
                "minor_faults": wave.get(
                    "process_tree_minor_faults"
                    if use_process_accounting
                    else "thread_tree_minor_faults"
                ),
                "major_faults": wave.get(
                    "process_tree_major_faults"
                    if use_process_accounting
                    else "thread_tree_major_faults"
                ),
            }
        )
    for statistics_payload in scheduler_statistics:
        request = statistics_payload.get("request_queue")
        work = statistics_payload.get("work_queue")
        if isinstance(request, Mapping):
            request_queues.append(request)
        if isinstance(work, Mapping):
            work_queues.append(work)

    elapsed = _numeric_values(waves, "elapsed_seconds")
    request_wait = _numeric_values(request_queues, "total_wait_seconds")
    work_wait = _numeric_values(work_queues, "total_wait_seconds")
    total_scheduler_wait = sum(request_wait) + sum(work_wait)
    total_elapsed = sum(elapsed)
    return {
        "wave_count": len(waves),
        "elapsed_seconds": _paired(elapsed),
        "axes_per_hour": _paired(_numeric_values(waves, "axes_per_hour")),
        "effective_cores": _paired(_numeric_values(waves, "effective_cores")),
        "cpu_utilization_fraction_of_compute_limit": _paired(
            _numeric_values(waves, "cpu_utilization_fraction_of_compute_limit")
        ),
        "process_tree_cpu_seconds": _paired(_numeric_values(waves, "process_tree_cpu_seconds")),
        "user_cpu_seconds": _paired(_numeric_values(waves, "process_tree_user_cpu_seconds")),
        "system_cpu_seconds": _paired(_numeric_values(waves, "process_tree_system_cpu_seconds")),
        "run_queue_wait_seconds": _paired(
            [value / 1_000_000_000.0 for value in _numeric_values(schedstat, "runqueue_delay_ns")]
        ),
        "context_switches": _paired(
            _numeric_values(scheduler_accounting, "context_switches")
        ),
        "cpu_migrations": _paired(
            _numeric_values(scheduler_accounting, "cpu_migrations")
        ),
        "minor_faults": _paired(_numeric_values(scheduler_accounting, "minor_faults")),
        "major_faults": _paired(_numeric_values(scheduler_accounting, "major_faults")),
        "rss_bytes": _paired(_numeric_values(waves, "peak_aggregate_rss_bytes")),
        "pss_bytes": _paired(_numeric_values(waves, "peak_aggregate_pss_bytes")),
        "cgroup_memory_peak_bytes": _paired(_numeric_values(cgroup_after, "memory_peak_bytes")),
        "io_read_bytes": _paired(_numeric_values(io_accounting, "read_bytes")),
        "io_write_bytes": _paired(_numeric_values(io_accounting, "write_bytes")),
        "io_accounting_sources": sorted(
            {
                source
                for row in io_accounting
                if isinstance((source := row.get("source")), str)
            }
        ),
        "scheduler_startup_seconds": _paired(_numeric_values(waves, "scheduler_startup_seconds")),
        "scheduler_shutdown_seconds": _paired(_numeric_values(waves, "scheduler_shutdown_seconds")),
        "scheduler_queue_wait_ratio": (
            total_scheduler_wait / total_elapsed if total_elapsed > 0.0 else None
        ),
        "request_queue_wait_seconds": _paired(request_wait),
        "request_queue_peak_depth": _paired(_numeric_values(request_queues, "peak_pending")),
        "work_queue_wait_seconds": _paired(work_wait),
        "work_queue_peak_depth": _paired(_numeric_values(work_queues, "peak_pending")),
        "work_queue_peak_active": _paired(_numeric_values(work_queues, "peak_active")),
        "queue_full_count": sum(
            int(value)
            for value in (
                *_numeric_values(request_queues, "queue_full_count"),
                *_numeric_values(work_queues, "queue_full_count"),
            )
        ),
        "rejected_count": sum(
            int(value)
            for value in (
                *_numeric_values(request_queues, "rejected_count"),
                *_numeric_values(work_queues, "rejected_count"),
            )
        ),
    }


def _mode_metrics(
    records: Iterable[ReviewRecord],
    *,
    replay_seconds: Mapping[Path, float],
) -> dict[str, object]:
    values = tuple(record for record in records if record.payload.get("status") == "completed")
    payloads = tuple(record.payload for record in values)
    topologies = tuple(
        topology
        for record in values
        if isinstance((topology := record.payload.get("topology")), Mapping)
    )
    native_queue: list[float] = []
    occupancy: list[int] = []
    iterations_per_cpu_second: list[float] = []
    exact_per_cpu_second: list[float] = []
    for record in values:
        native = record.payload.get("native_execution_statistics")
        if isinstance(native, dict):
            queue = native.get("queue_wait_seconds")
            if isinstance(queue, int | float) and not isinstance(queue, bool):
                native_queue.append(float(queue))
        backend = record.payload.get("backend_metrics")
        if isinstance(backend, dict):
            launches = backend.get("launch_occupancies")
            if isinstance(launches, list):
                occupancy.extend(
                    item
                    for item in launches
                    if isinstance(item, int) and not isinstance(item, bool) and item > 0
                )
        topology = record.payload.get("topology")
        cpu_seconds = (
            topology.get("process_tree_cpu_seconds") if isinstance(topology, Mapping) else None
        )
        if (
            not isinstance(cpu_seconds, bool)
            and isinstance(cpu_seconds, int | float)
            and math.isfinite(float(cpu_seconds))
            and float(cpu_seconds) > 0.0
        ):
            iterations_per_cpu_second.append(
                _integer(record.payload, "effective_iterations") / float(cpu_seconds)
            )
            exact_per_cpu_second.append(
                _integer(record.payload, "exact_started_calls") / float(cpu_seconds)
            )
    return {
        "completed_axes": len(values),
        "startup_seconds": _paired(_numeric_values(payloads, "startup_seconds")),
        "solver_seconds": _paired(_number(record.payload, "solver_seconds") for record in values),
        "persistence_seconds": _paired(
            _number(record.payload, "persistence_seconds") for record in values
        ),
        "producer_pre_receipt_seconds": _paired(
            _producer_pre_receipt_seconds(record) for record in values
        ),
        "end_to_end_seconds": _paired(
            _axis_end_to_end_seconds(record, replay_seconds[record.path]) for record in values
        ),
        "effective_iterations": _paired(
            float(_integer(record.payload, "effective_iterations")) for record in values
        ),
        "exact_started_calls": _paired(
            float(_integer(record.payload, "exact_started_calls")) for record in values
        ),
        "vehicle_count": _paired(_vehicle_count(record.payload) for record in values),
        "effective_iterations_per_second": _paired(
            _number(_mapping(record.payload, "throughput"), "effective_iterations_per_second")
            for record in values
        ),
        "candidate_transactions_per_second": _paired(
            _number(_mapping(record.payload, "throughput"), "candidate_transactions_per_second")
            for record in values
        ),
        "screened_routes_per_second": _paired(
            _number(_mapping(record.payload, "throughput"), "screened_routes_per_second")
            for record in values
        ),
        "exact_started_per_second": _paired(
            _number(_mapping(record.payload, "throughput"), "exact_started_per_second")
            for record in values
        ),
        "cpu_utilization_percent_of_one_core": _paired(
            _number(
                _mapping(record.payload, "topology"),
                "cpu_utilization_percent_of_one_core",
            )
            for record in values
        ),
        "cpu_utilization_percent_of_compute_limit": _paired(
            _numeric_values(topologies, "cpu_utilization_percent_of_compute_limit")
        ),
        "process_tree_cpu_seconds": _paired(
            _numeric_values(topologies, "process_tree_cpu_seconds")
        ),
        "effective_iterations_per_cpu_second": _paired(iterations_per_cpu_second),
        "exact_started_per_cpu_second": _paired(exact_per_cpu_second),
        "rss_bytes": _paired(
            float(
                _topology_integer(
                    record.payload,
                    "peak_aggregate_rss_bytes",
                    legacy_field="rss_bytes",
                )
            )
            for record in values
        ),
        "pss_bytes": _paired(
            float(
                _topology_integer(
                    record.payload,
                    "peak_aggregate_pss_bytes",
                    legacy_field="rss_bytes",
                )
            )
            for record in values
        ),
        "cache_memory_bytes": _paired(
            float(_integer(record.payload, "cache_memory_bytes")) for record in values
        ),
        "artifact_bytes": _paired(
            float(_integer(record.payload, "artifact_bytes")) for record in values
        ),
        "queue_wait_seconds": _paired(native_queue),
        "exact_backend_batch_occupancy": _paired(float(value) for value in occupancy),
        "mode_wave_resources": _mode_wave_metrics(values),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        sidecar = path.with_suffix(".sha256")
    expected = sidecar.read_text(encoding="ascii").strip().split()[0]
    actual = _sha256(path)
    if expected != actual:
        raise ValueError(f"checksum mismatch: {path}")
    return actual


def _raw_axis_inventory(records: Iterable[ReviewRecord]) -> dict[str, object]:
    values = tuple(records)

    def inventory(selected: Iterable[ReviewRecord]) -> dict[str, object]:
        entries: list[dict[str, object]] = []
        json_bytes = 0
        sidecar_bytes = 0
        persistence_receipt_bytes = 0
        for record in sorted(
            selected,
            key=lambda item: (_string(item.payload, "run_label"), item.key),
        ):
            sidecar = record.path.with_suffix(record.path.suffix + ".sha256")
            raw_data = record.path.read_bytes()
            sidecar_data = sidecar.read_bytes()
            json_bytes += len(raw_data)
            sidecar_bytes += len(sidecar_data)
            entry: dict[str, object] = {
                "logical_axis": [
                    _string(record.payload, "run_label"),
                    *record.key,
                ],
                "json_sha256": hashlib.sha256(raw_data).hexdigest(),
                "sidecar_sha256": hashlib.sha256(sidecar_data).hexdigest(),
            }
            if record.payload.get("schema_version") in PROFILE_COMPARISON_SCHEMA_VERSIONS:
                receipt = _axis_persistence_receipt_path(record.path)
                receipt_sidecar = receipt.with_suffix(receipt.suffix + ".sha256")
                receipt_data = receipt.read_bytes()
                receipt_sidecar_data = receipt_sidecar.read_bytes()
                persistence_receipt_bytes += len(receipt_data) + len(receipt_sidecar_data)
                entry["persistence_receipt_sha256"] = hashlib.sha256(receipt_data).hexdigest()
                entry["persistence_receipt_sidecar_sha256"] = hashlib.sha256(
                    receipt_sidecar_data
                ).hexdigest()
            entries.append(entry)
        canonical = json.dumps(
            entries,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return {
            "axis_count": len(entries),
            "json_bytes": json_bytes,
            "sidecar_bytes": sidecar_bytes,
            "persistence_receipt_bytes": persistence_receipt_bytes,
            "tree_sha256": hashlib.sha256(
                b"stage05.2-native-axis-inventory-v2\0" + canonical
            ).hexdigest(),
        }

    aggregate = inventory(values)
    aggregate["algorithm"] = (
        "sha256(stage05.2-native-axis-inventory-v2\\0 + canonical JSON of logical axis, "
        "JSON/sidecar SHA-256, and persistence receipt/sidecar SHA-256)"
    )
    aggregate["by_mode"] = {
        mode.value: inventory(record for record in values if record.mode is mode) for mode in MODES
    }
    return aggregate


def _historical_cross_schema_adapter(
    root: Path,
    benchmark_dir: Path,
) -> dict[str, object]:
    """Project accepted v1 Pilot evidence onto the v2 core semantic fields."""

    rows: list[dict[str, object]] = []
    aggregate_digest = hashlib.sha256(b"stage05.2-attempt72-cross-schema-v1\0")
    for batch_dir in sorted(root.glob("batch[0-9][0-9][0-9][0-9]")):
        reader = ArtifactReader(batch_dir, verify=False)
        artifacts = reader.manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("historical artifact manifest has no artifact inventory")
        references = {
            str(item.get("relative_path")): item
            for item in artifacts
            if isinstance(item, dict) and isinstance(item.get("relative_path"), str)
        }
        event_paths = sorted(
            relative
            for relative, item in references.items()
            if item.get("artifact_type") == "events" and item.get("artifact_subtype") == "critical"
        )
        for event_relative in event_paths:
            event_path = Path(event_relative)
            if len(event_path.parts) < 3:
                raise ValueError("historical event path is not canonical")
            instance_name = event_path.parts[0]
            seed = int(event_path.parts[1])
            raw_relative = str(
                event_path.with_name(event_path.name.replace("_events_", "_raw_", 1)).with_suffix(
                    ".json"
                )
            )
            solution_relative = str(
                event_path.with_name(
                    event_path.name.replace("_events_", "_solution_", 1)
                ).with_suffix(".json")
            )
            trace_relative = str(
                event_path.with_name(event_path.name.replace("_events_", "_trace_", 1)).with_suffix(
                    ".json"
                )
            )
            trace = reader.read_json(trace_relative)
            trace_references = (
                trace.get("route_dictionary_ref"),
                trace.get("screening_definitions_ref"),
                trace.get("screening_occurrences_ref"),
            )
            required_paths = (
                event_relative,
                raw_relative,
                solution_relative,
                trace_relative,
                *trace_references,
            )
            for relative in required_paths:
                if not isinstance(relative, str) or relative not in references:
                    raise ValueError(f"historical cross-schema artifact is missing: {relative}")
                reference = references[relative]
                if reference.get("checksum") != _sha256(batch_dir / relative):
                    raise ValueError(f"historical cross-schema checksum mismatch: {relative}")

            raw = reader.read_json(raw_relative)
            solution = reader.read_json(solution_relative)
            raw_axes = raw.get("axes")
            solution_axes = solution.get("axes")
            raw_axis = raw_axes.get("wall_clock_30") if isinstance(raw_axes, dict) else None
            solution_axis = (
                solution_axes.get("wall_clock_30") if isinstance(solution_axes, dict) else None
            )
            if not isinstance(raw_axis, dict) or not isinstance(solution_axis, dict):
                raise ValueError("historical cross-schema wall-clock axis is missing")
            bundle = verified_artifact_shard_bundle(
                reader=reader,
                shard_id=f"{instance_name}/{seed}",
                event_relative_path=event_relative,
                axis_budgets={"wall_clock_30": 30},
            )
            summary = replay_verified_shard(bundle, backend="python_reference")
            event_counts = dict(summary.event_type_counts)
            exact_counts = dict(
                (axis, (started, completed))
                for axis, started, completed in summary.axis_exact_counts
            )
            if exact_counts.get("wall_clock_30") != (
                raw_axis.get("started_calls"),
                raw_axis.get("completed_calls"),
            ):
                raise ValueError("historical exact-call replay does not reconcile")
            required_families = {
                "cache_event",
                "candidate_cache_commit",
                "candidate_state",
                "route_evaluation",
            }
            if not required_families.issubset(event_counts) or summary.native_fallback_count != 0:
                raise ValueError("historical core transaction event projection is incomplete")

            instance = parse_schneider(benchmark_dir / f"{instance_name}.txt")
            routes = solution_axis.get("routes")
            if not isinstance(routes, list) or not all(
                isinstance(route, list) and all(isinstance(node, str) for node in route)
                for route in routes
            ):
                raise ValueError("historical solution routes are invalid")
            report = validate_routes(instance, routes)
            replay_objective = SolutionObjective.from_report(instance, report).key
            recorded_objective = solution_axis.get("objective_key")
            if (
                not report.feasible
                or list(replay_objective) != recorded_objective
                or recorded_objective != raw_axis.get("objective_key")
                or raw_axis.get("validator_passed") is not True
            ):
                raise ValueError("historical validator/objective replay does not reconcile")

            projection = {
                "instance": instance_name,
                "seed": seed,
                "validator_passed": True,
                "objective": list(replay_objective),
                "exact_started_calls": summary.exact_started,
                "exact_completed_calls": summary.exact_completed,
                "accepted_candidates": summary.accepted_candidates,
                "global_bests": summary.global_bests,
                "native_fallback_count": summary.native_fallback_count,
                "deadline_axes": list(summary.deadline_axes),
                "cache_events": event_counts["cache_event"],
                "candidate_cache_commits": event_counts["candidate_cache_commit"],
                "candidate_state_events": event_counts["candidate_state"],
                "route_evaluations": event_counts["route_evaluation"],
                "canonical_event_sha256": summary.event_token_sha256,
            }
            aggregate_digest.update(
                json.dumps(
                    projection,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
            rows.append(projection)
    expected = {(name, seed) for name in FORMAL_INSTANCES for seed in SEEDS}
    observed = {(_string(row, "instance"), _integer(row, "seed")) for row in rows}
    if observed != expected or len(rows) != len(observed):
        raise ValueError("historical cross-schema adapter geometry is not 12x3")
    return {
        "schema_version": "stage05.2-attempt72-cross-schema-v1",
        "passed": True,
        "axis_count": len(rows),
        "mapped_fields": [
            "validator",
            "objective",
            "candidate_state",
            "cache_lifecycle",
            "exact_started_completed",
            "deadline_boundaries",
            "fallback_count",
            "canonical_event_stream",
        ],
        "aggregate_projection_sha256": aggregate_digest.hexdigest(),
        "rows": rows,
    }


def _historical_pilot(
    root: Path = HISTORICAL_PILOT_ROOT,
    *,
    benchmark_dir: Path,
) -> tuple[dict[tuple[str, int], dict[str, object]], dict[str, object]]:
    if not root.is_dir():
        return {}, {
            "available": False,
            "identity_verified": False,
            "error": "historical archive is not available",
        }
    active = root.parent / "wsl_active"
    try:
        campaign_path = active / "campaign_manifest.json"
        raw_manifest_path = active / "control" / f"{HISTORICAL_PILOT_RUN_LABEL}_manifest.json"
        run_metadata_path = active / "control" / f"{HISTORICAL_PILOT_RUN_LABEL}_run_metadata.json"
        review_manifest_path = active / "review" / "review_manifest.json"
        review_execution_path = active / "review" / "review_execution.json"
        campaign_sha256 = _verify_sidecar(campaign_path)
        raw_manifest_sha256 = _verify_sidecar(raw_manifest_path)
        if raw_manifest_sha256 != HISTORICAL_PILOT_RAW_MANIFEST_SHA256:
            raise ValueError("historical raw manifest is not the accepted identity")
        review_manifest_sha256 = _sha256(review_manifest_path)
        if review_manifest_sha256 != HISTORICAL_PILOT_REVIEW_MANIFEST_SHA256:
            raise ValueError("historical review manifest is not the accepted identity")

        campaign = json.loads(campaign_path.read_bytes())
        raw_manifest = json.loads(raw_manifest_path.read_bytes())
        run_metadata = json.loads(run_metadata_path.read_bytes())
        review_manifest = json.loads(review_manifest_path.read_bytes())
        review_execution = json.loads(review_execution_path.read_bytes())
        if not all(
            isinstance(value, dict)
            for value in (
                campaign,
                raw_manifest,
                run_metadata,
                review_manifest,
                review_execution,
            )
        ):
            raise ValueError("historical identity documents must be JSON objects")
        if (
            campaign.get("run_label") != HISTORICAL_PILOT_RUN_LABEL
            or campaign.get("scope") != "pilot"
            or campaign.get("status") != "complete"
            or campaign.get("axis_count") != 36
            or campaign.get("shard_count") != 36
            or raw_manifest.get("run_label") != HISTORICAL_PILOT_RUN_LABEL
            or raw_manifest.get("status") != "complete"
            or run_metadata.get("repository_revision") != HISTORICAL_PILOT_REVISION
            or review_manifest.get("run_label") != HISTORICAL_PILOT_RUN_LABEL
            or review_manifest.get("scope") != "pilot"
            or review_manifest.get("status") != "READY_FOR_STAGE052_FORMAL_BENCHMARK"
            or review_manifest.get("raw_campaign_manifest_sha256") != campaign_sha256
            or review_manifest.get("raw_manifest_sha256") != raw_manifest_sha256
            or review_execution.get("review_manifest_sha256") != review_manifest_sha256
            or review_execution.get("raw_manifest_sha256_before") != raw_manifest_sha256
            or review_execution.get("raw_manifest_sha256_after") != raw_manifest_sha256
            or review_execution.get("producer_repository_revision") != HISTORICAL_PILOT_REVISION
        ):
            raise ValueError("historical accepted-Pilot identity fields do not match")
        gates = review_manifest.get("gates")
        if (
            not isinstance(gates, dict)
            or not gates
            or any(
                not isinstance(gate, dict) or gate.get("passed") is not True
                for gate in gates.values()
            )
        ):
            raise ValueError("historical accepted review gates do not all pass")

        batches = campaign.get("batches")
        if not isinstance(batches, list) or len(batches) != 3:
            raise ValueError("historical campaign must contain exactly three batches")
        records: dict[tuple[str, int], dict[str, object]] = {}
        for batch in batches:
            if not isinstance(batch, dict):
                raise ValueError("historical batch entry must be an object")
            batch_id = _string(batch, "batch_id")
            batch_root = root / batch_id
            batch_manifest_path = batch_root / "batch_manifest.json"
            _verify_sidecar(batch_manifest_path)
            batch_manifest = json.loads(batch_manifest_path.read_bytes())
            if batch_manifest != batch or batch.get("status") != "archived":
                raise ValueError(f"historical batch identity mismatch: {batch_id}")
            shard_hashes = batch.get("shard_manifest_sha256_by_id")
            if not isinstance(shard_hashes, dict) or len(shard_hashes) != 12:
                raise ValueError(f"historical batch shard map is invalid: {batch_id}")
            shard_paths = tuple(batch_root.glob("*/*/*_shard_manifest_*.json"))
            if len(shard_paths) != 12:
                raise ValueError(f"historical batch must contain 12 shard manifests: {batch_id}")
            for shard_path in shard_paths:
                shard = json.loads(shard_path.read_bytes())
                if not isinstance(shard, dict):
                    raise ValueError("historical shard manifest must be an object")
                ordinal = _integer(shard, "shard_ordinal") + 1
                shard_id = f"shard{ordinal:04d}"
                if shard_hashes.get(shard_id) != _sha256(shard_path):
                    raise ValueError(f"historical shard manifest mismatch: {shard_id}")
                if (
                    shard.get("run_label") != HISTORICAL_PILOT_RUN_LABEL
                    or shard.get("evidence_completeness") != "complete"
                ):
                    raise ValueError(f"historical shard is not complete: {shard_id}")
                artifacts = shard.get("artifacts")
                if not isinstance(artifacts, list):
                    raise ValueError("historical shard artifacts must be an array")
                raw_artifacts = [
                    artifact
                    for artifact in artifacts
                    if isinstance(artifact, dict) and artifact.get("artifact_type") == "raw"
                ]
                if len(raw_artifacts) != 1:
                    raise ValueError(f"historical shard raw identity is ambiguous: {shard_id}")
                raw_artifact = raw_artifacts[0]
                raw_path = batch_root / _string(raw_artifact, "relative_path")
                if raw_artifact.get("checksum") != _sha256(raw_path):
                    raise ValueError(f"historical raw checksum mismatch: {shard_id}")
                payload = json.loads(raw_path.read_bytes())
                if not isinstance(payload, dict):
                    raise ValueError("historical raw payload must be an object")
                axes = payload.get("axes")
                if not isinstance(axes, dict) or not isinstance(axes.get("wall_clock_30"), dict):
                    raise ValueError("historical raw payload lacks wall_clock_30")
                key = (_string(payload, "instance"), _integer(payload, "seed"))
                if key in records:
                    raise ValueError(f"duplicate historical instance/seed: {key}")
                records[key] = dict(axes["wall_clock_30"])
        expected_keys = {(name, seed) for name in FORMAL_INSTANCES for seed in SEEDS}
        if set(records) != expected_keys:
            raise ValueError("historical accepted-Pilot geometry does not match 12x3")
        cross_schema = _historical_cross_schema_adapter(root, benchmark_dir)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return {}, {
            "available": True,
            "identity_verified": False,
            "error": str(exc),
        }
    return records, {
        "available": True,
        "identity_verified": True,
        "error": None,
        "campaign_manifest_sha256": campaign_sha256,
        "raw_manifest_sha256": raw_manifest_sha256,
        "review_manifest_sha256": review_manifest_sha256,
        "producer_revision": HISTORICAL_PILOT_REVISION,
        "cross_schema_adapter": cross_schema,
    }


def review_records(
    records: tuple[ReviewRecord, ...],
    *,
    scope: str,
    benchmark_dir: Path,
) -> dict[str, object]:
    replay: dict[Path, dict[str, object]] = {}
    replay_seconds: dict[Path, float] = {}
    for record in records:
        replay_started = time.perf_counter()
        replay[record.path] = _replay_record(
            ReviewRecord(record.path, _verify_signed_json(record.path)),
            benchmark_dir,
        )
        replay_seconds[record.path] = time.perf_counter() - replay_started
    by_mode = {mode: tuple(record for record in records if record.mode is mode) for mode in MODES}
    by_identity: dict[tuple[int, str, str, int], dict[ArchitectureMode, ReviewRecord]] = (
        defaultdict(dict)
    )
    for record in records:
        by_identity[record.key][record.mode] = record

    differential: dict[str, dict[str, object]] = {}
    for mode in (
        ArchitectureMode.PER_SOLVE_RUNTIME,
        ArchitectureMode.FULL_NATIVE_ALNS,
        ArchitectureMode.HOST_SCHEDULER,
    ):
        comparisons = []
        for key, modes in by_identity.items():
            if key[1] != "fixed_work":
                continue
            baseline = modes[ArchitectureMode.PYTHON_CANDIDATE_CONTROL]
            candidate = modes[mode]
            if (
                baseline.payload.get("status") != "completed"
                or candidate.payload.get("status") != "completed"
            ):
                comparisons.append(
                    {
                        "key": key,
                        "baseline_status": baseline.payload.get("status"),
                        "candidate_status": candidate.payload.get("status"),
                        "objective_equal": False,
                        "routes_equal": False,
                        "exact_order_equal": False,
                        "exact_counts_equal": False,
                        "candidate_work_hash_equal": False,
                        "route_result_hash_equal": False,
                        "trajectory_equal": False,
                        "screening_semantics_equal": False,
                        "canonical_semantic_events_equal": False,
                        "operator_statistics_equal": False,
                        "stage04_state_equal": False,
                        "candidate_transaction_events_equal": False,
                        "cache_lifecycle_equal": False,
                        "deadline_boundaries_equal": False,
                        "measurement_transaction_hash_equal": False,
                        "common_prefix": 0,
                    }
                )
                continue
            baseline_trajectory = _semantic_trajectory(baseline.payload)
            candidate_trajectory = _semantic_trajectory(candidate.payload)
            baseline_semantic_events = _comparison_semantic_events_from_artifact(baseline)
            candidate_semantic_events = _comparison_semantic_events_from_artifact(candidate)
            baseline_screening_events = [
                event
                for event in baseline_semantic_events
                if isinstance(event, dict) and event.get("semantic_stream") == "screening"
            ]
            candidate_screening_events = [
                event
                for event in candidate_semantic_events
                if isinstance(event, dict) and event.get("semantic_stream") == "screening"
            ]
            baseline_measurement = _mapping(baseline.payload, "measurement_evidence")
            candidate_measurement = _mapping(candidate.payload, "measurement_evidence")
            exact_order_equal = baseline_measurement.get(
                "exact_route_order"
            ) == candidate_measurement.get("exact_route_order")
            exact_counts_equal = baseline.payload.get(
                "exact_started_calls"
            ) == candidate.payload.get("exact_started_calls") and baseline.payload.get(
                "exact_completed_calls"
            ) == candidate.payload.get("exact_completed_calls")
            candidate_work_hash_equal = baseline.payload.get(
                "candidate_work_hash"
            ) == candidate.payload.get("candidate_work_hash")
            route_result_hash_equal = baseline.payload.get(
                "route_result_hash"
            ) == candidate.payload.get("route_result_hash")
            cache_lifecycle_equal = baseline_measurement.get(
                "cache_lifecycle"
            ) == candidate_measurement.get("cache_lifecycle")
            deadline_boundaries_equal = baseline_measurement.get(
                "deadline_boundaries"
            ) == candidate_measurement.get("deadline_boundaries")
            measurement_transaction_hash_equal = baseline_measurement.get(
                "sha256"
            ) == candidate_measurement.get("sha256")
            candidate_transaction_semantics_equal = all(
                (
                    exact_order_equal,
                    exact_counts_equal,
                    candidate_work_hash_equal,
                    route_result_hash_equal,
                    cache_lifecycle_equal,
                    deadline_boundaries_equal,
                    measurement_transaction_hash_equal,
                )
            )
            common_prefix = _common_prefix(
                baseline_semantic_events,
                candidate_semantic_events,
            )
            first_divergence = _describe_first_divergence(
                baseline_semantic_events,
                candidate_semantic_events,
            )
            comparisons.append(
                {
                    "key": key,
                    "objective_equal": baseline.payload.get("objective")
                    == candidate.payload.get("objective"),
                    "routes_equal": baseline.payload.get("routes")
                    == candidate.payload.get("routes"),
                    "exact_order_equal": exact_order_equal,
                    "exact_counts_equal": exact_counts_equal,
                    "candidate_work_hash_equal": candidate_work_hash_equal,
                    "route_result_hash_equal": route_result_hash_equal,
                    "trajectory_equal": baseline_trajectory == candidate_trajectory,
                    "screening_semantics_equal": (
                        baseline_screening_events == candidate_screening_events
                    ),
                    "canonical_semantic_events_equal": (
                        baseline_semantic_events == candidate_semantic_events
                    ),
                    "operator_statistics_equal": baseline.payload.get(
                        "operator_semantic_statistics",
                        baseline.payload.get("operator_statistics"),
                    )
                    == candidate.payload.get(
                        "operator_semantic_statistics",
                        candidate.payload.get("operator_statistics"),
                    ),
                    "stage04_state_equal": (
                        baseline.payload.get("stage04_statistics")
                        == candidate.payload.get("stage04_statistics")
                        and baseline.payload.get("stage04_events")
                        == candidate.payload.get("stage04_events")
                    ),
                    "candidate_transaction_events_equal": (candidate_transaction_semantics_equal),
                    "candidate_transaction_telemetry_equal": baseline.payload.get(
                        "candidate_transaction_events"
                    )
                    == candidate.payload.get("candidate_transaction_events"),
                    "cache_lifecycle_equal": cache_lifecycle_equal,
                    "deadline_boundaries_equal": deadline_boundaries_equal,
                    "measurement_transaction_hash_equal": (measurement_transaction_hash_equal),
                    "common_prefix": common_prefix,
                    "first_divergence": first_divergence,
                }
            )
        differential[mode.value] = {
            "comparison_count": len(comparisons),
            "passed": bool(comparisons)
            and all(
                all(
                    bool(comparison[field])
                    for field in (
                        "objective_equal",
                        "routes_equal",
                        "exact_order_equal",
                        "exact_counts_equal",
                        "candidate_work_hash_equal",
                        "route_result_hash_equal",
                        "trajectory_equal",
                        "screening_semantics_equal",
                        "canonical_semantic_events_equal",
                        "operator_statistics_equal",
                        "stage04_state_equal",
                        "candidate_transaction_events_equal",
                        "cache_lifecycle_equal",
                        "deadline_boundaries_equal",
                        "measurement_transaction_hash_equal",
                    )
                )
                for comparison in comparisons
            ),
            "comparisons": comparisons,
        }

    mode_block_timings: dict[
        ArchitectureMode,
        dict[tuple[int, str, str], dict[str, object]],
    ] = {mode: {} for mode in MODES}
    mode_block_evidence_complete = all(record.mode_wave is not None for record in records)
    if mode_block_evidence_complete:
        grouped_blocks: dict[
            tuple[ArchitectureMode, int, str, str],
            list[ReviewRecord],
        ] = defaultdict(list)
        for record in records:
            assert record.mode_wave is not None
            family = _string(record.mode_wave, "performance_family")
            grouped_blocks[(record.mode, record.key[0], record.key[1], family)].append(record)
        for (mode, repeat, axis, family), block_records in grouped_blocks.items():
            waves = [record.mode_wave for record in block_records]
            first_wave = waves[0]
            assert first_wave is not None
            if any(wave != first_wave for wave in waves[1:]):
                raise RuntimeError("mode-block axes do not share one timing receipt")
            if _integer(first_wave, "axis_count") != len(block_records):
                raise RuntimeError("mode-block timing receipt has the wrong axis count")
            elapsed = _number(first_wave, "elapsed_seconds")
            replay_total = sum(replay_seconds[record.path] for record in block_records)
            shared_replay = _number(first_wave, "independent_shared_replay_seconds")
            mode_block_timings[mode][(repeat, axis, family)] = {
                "producer_mode_block_seconds": elapsed,
                "axis_independent_replay_seconds": replay_total,
                "shared_independent_replay_seconds": shared_replay,
                "independent_replay_seconds": replay_total + shared_replay,
                "end_to_end_seconds": elapsed + replay_total + shared_replay,
                "axis_count": len(block_records),
            }

    speedups: dict[str, dict[str, object]] = {}
    for mode in MODES:
        if mode is ArchitectureMode.CURRENT_STAGE052:
            continue
        versus_current: list[float] = []
        versus_python: list[float] = []
        wall_objective_not_worse = True
        for key, modes in by_identity.items():
            candidate = modes[mode]
            current_record = modes[ArchitectureMode.CURRENT_STAGE052]
            python_record = modes[ArchitectureMode.PYTHON_CANDIDATE_CONTROL]
            if any(
                record.payload.get("status") != "completed"
                for record in (candidate, current_record, python_record)
            ):
                continue
            candidate_seconds = _axis_end_to_end_seconds(
                candidate,
                replay_seconds[candidate.path],
            )
            current_seconds = _axis_end_to_end_seconds(
                current_record,
                replay_seconds[current_record.path],
            )
            python_seconds = _axis_end_to_end_seconds(
                python_record,
                replay_seconds[python_record.path],
            )
            versus_current.append(_relative_time_improvement(candidate_seconds, current_seconds))
            versus_python.append(_relative_time_improvement(candidate_seconds, python_seconds))
            if key[1] == "wall_clock_30":
                candidate_objective = tuple(_sequence(candidate.payload, "objective"))
                current_objective = tuple(
                    _sequence(
                        current_record.payload,
                        "objective",
                    )
                )
                wall_objective_not_worse = (
                    wall_objective_not_worse and candidate_objective <= current_objective
                )
        block_versus_current: list[float] = []
        block_versus_python: list[float] = []
        family_values: dict[str, list[float]] = defaultdict(list)
        for block_key, candidate_timing in mode_block_timings[mode].items():
            current_timing = mode_block_timings[ArchitectureMode.CURRENT_STAGE052].get(block_key)
            python_timing = mode_block_timings[ArchitectureMode.PYTHON_CANDIDATE_CONTROL].get(
                block_key
            )
            if current_timing is None or python_timing is None:
                continue
            candidate_block_seconds = _number(candidate_timing, "end_to_end_seconds")
            current_block_seconds = _number(current_timing, "end_to_end_seconds")
            python_block_seconds = _number(python_timing, "end_to_end_seconds")
            improvement = _relative_time_improvement(
                candidate_block_seconds,
                current_block_seconds,
            )
            block_versus_current.append(improvement)
            block_versus_python.append(
                _relative_time_improvement(
                    candidate_block_seconds,
                    python_block_seconds,
                )
            )
            if block_key[1] == "fixed_work" and block_key[2] in {"C", "R", "RC"}:
                family_values[block_key[2]].append(improvement)
        family_medians = {
            family: statistics.median(values) for family, values in family_values.items() if values
        }
        aggregate_100 = [value for values in family_values.values() for value in values]
        performance_qualified = (
            mode_block_evidence_complete
            and bool(aggregate_100)
            and statistics.median(aggregate_100) >= 0.15
            and set(family_medians) == {"C", "R", "RC"}
            and all(value >= -0.03 for value in family_medians.values())
            and wall_objective_not_worse
            and (
                mode
                not in {
                    ArchitectureMode.PER_SOLVE_RUNTIME,
                    ArchitectureMode.FULL_NATIVE_ALNS,
                    ArchitectureMode.HOST_SCHEDULER,
                }
                or bool(differential[mode.value]["passed"])
            )
        )
        speedups[mode.value] = {
            "versus_current": _paired(versus_current),
            "versus_python_candidate_control": _paired(versus_python),
            "mode_block_versus_current": _paired(block_versus_current),
            "mode_block_versus_python_candidate_control": _paired(block_versus_python),
            "aggregate_100_customer": _paired(aggregate_100),
            "family_median_improvement": family_medians,
            "wall_clock_objective_not_worse": wall_objective_not_worse,
            "improvement_definition": "1 - candidate_end_to_end / baseline_end_to_end",
            "timing_basis": "identity_mode_block_elapsed_plus_independent_axis_replay",
            "mode_block_evidence_complete": mode_block_evidence_complete,
            "mode_block_end_to_end": {
                f"repeat{key[0] + 1}:{key[1]}:{key[2]}": value
                for key, value in sorted(mode_block_timings[mode].items())
            },
            "performance_qualified": performance_qualified,
        }

    historical, historical_identity = (
        _historical_pilot(benchmark_dir=benchmark_dir)
        if scope == "pilot"
        else (
            {},
            {
                "available": False,
                "identity_verified": False,
                "error": "not assessed for paired scope",
            },
        )
    )
    historical_drift: list[dict[str, object]] = []
    if historical:
        for record in by_mode[ArchitectureMode.CURRENT_STAGE052]:
            historical_key = (
                _string(record.payload, "instance"),
                _integer(record.payload, "seed"),
            )
            prior = historical.get(historical_key)
            if prior is None:
                continue
            prior_seconds = _number(prior, "runtime_seconds")
            historical_drift.append(
                {
                    "instance": historical_key[0],
                    "seed": historical_key[1],
                    "solver_ratio_new_over_historical": _number(record.payload, "solver_seconds")
                    / prior_seconds,
                    "objective_equal": record.payload.get("objective")
                    == prior.get("objective_key"),
                    "validator_historical": prior.get("validator_passed"),
                }
            )

    axis_replay_passed = all(bool(value["valid"]) for value in replay.values())
    semantic_gates_passed = all(bool(value.get("semantics_complete")) for value in replay.values())
    architecture_modes = (
        ArchitectureMode.PER_SOLVE_RUNTIME,
        ArchitectureMode.FULL_NATIVE_ALNS,
        ArchitectureMode.HOST_SCHEDULER,
    )
    correctness_matrix_passed = all(
        bool(differential[mode.value]["passed"]) for mode in architecture_modes
    )
    eligible_production_modes = [
        mode.value
        for mode in architecture_modes
        if bool(differential[mode.value]["passed"])
        and bool(speedups[mode.value]["performance_qualified"])
    ]
    current_schema_qualified = _records_use_current_profile_schema(records)
    qualification_passed = (
        axis_replay_passed
        and semantic_gates_passed
        and correctness_matrix_passed
        and bool(eligible_production_modes)
        and current_schema_qualified
    )
    review_status = (
        "COMPARISON_COMPLETE_QUALIFIED"
        if qualification_passed
        else "COMPARISON_COMPLETE_NOT_QUALIFIED"
        if axis_replay_passed
        else "NOT_READY"
    )
    cuda_evaluation_condition = _scheduler_screening_occupancy(
        by_mode[ArchitectureMode.HOST_SCHEDULER]
    )
    replay_timing_values = tuple(replay_seconds.values())
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "scope": scope,
        "attempt": _review_attempt(records, scope),
        "axis_count": len(records),
        "axis_replay_passed": axis_replay_passed,
        "semantic_gates_passed": semantic_gates_passed,
        "replay_failures": [
            {"path": str(path), **dict(value)}
            for path, value in replay.items()
            if not bool(value["valid"])
        ],
        "replay_timing": {
            "seconds_by_axis": {
                str(path): seconds for path, seconds in sorted(replay_seconds.items())
            },
            "total_seconds": sum(replay_timing_values),
            "p50_seconds": (
                statistics.median(replay_timing_values) if replay_timing_values else None
            ),
            "p95_seconds": (
                sorted(replay_timing_values)[
                    max(0, math.ceil(0.95 * len(replay_timing_values)) - 1)
                ]
                if replay_timing_values
                else None
            ),
            "maximum_seconds": max(replay_timing_values) if replay_timing_values else None,
        },
        "mode_metrics": {
            mode.value: _mode_metrics(mode_records, replay_seconds=replay_seconds)
            for mode, mode_records in by_mode.items()
        },
        "differential_gates": differential,
        "performance": speedups,
        "historical_attempt72": {
            **historical_identity,
            "comparison_count": len(historical_drift),
            "drift": historical_drift,
        },
        "cuda_evaluation_condition": cuda_evaluation_condition,
        "cuda_evaluation_condition_met": bool(cuda_evaluation_condition["condition_met"]),
        "producer_identity": _producer_identity(records),
        "reviewer_provenance": _reviewer_provenance(),
        "raw_axis_inventory": _raw_axis_inventory(records),
        "formal_started": False,
        "production_default_changed": False,
        "correctness_matrix_passed": correctness_matrix_passed,
        "eligible_production_modes": eligible_production_modes,
        "current_schema_qualified": current_schema_qualified,
        "qualification_passed": qualification_passed,
        "review_status": review_status,
    }


def _scheduler_screening_occupancy(
    records: Iterable[ReviewRecord],
) -> dict[str, object]:
    occupancy: list[int] = []
    for record in records:
        native = record.payload.get("native_execution_statistics")
        if not isinstance(native, dict):
            continue
        values = native.get("candidate_screening_occupancies")
        if isinstance(values, list):
            occupancy.extend(
                value
                for value in values
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            )
    if not occupancy:
        return {
            "available": False,
            "condition_met": False,
            "sample_count": 0,
            "median": None,
            "p95": None,
            "maximum": None,
            "reason": "native candidate-screening occupancy is not recorded",
        }
    ordered = sorted(occupancy)
    median = float(statistics.median(ordered))
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    maximum = max(occupancy)
    return {
        "available": True,
        "condition_met": median >= 32.0,
        "sample_count": len(ordered),
        "median": median,
        "p95": p95,
        "maximum": maximum,
        "reason": "native candidate-screening median occupancy replayed",
    }


def _producer_identity(records: Iterable[ReviewRecord]) -> dict[str, object]:
    values = tuple(records)
    return {
        "repository_revisions": sorted({_string(record.payload, "revision") for record in values}),
        "wheel_sha256": sorted({_string(record.payload, "wheel_sha256") for record in values}),
        "native_sha256": sorted({_string(record.payload, "native_sha256") for record in values}),
        "scheduler_sha256": sorted(
            {
                value
                for record in values
                if isinstance(value := record.payload.get("scheduler_sha256"), str) and value
            }
        ),
        "performance_profile_sha256": sorted(
            {
                value
                for record in values
                if isinstance(
                    value := record.payload.get("performance_profile_sha256"),
                    str,
                )
                and value
            }
        ),
        "run_labels": sorted({_string(record.payload, "run_label") for record in values}),
    }


def _review_attempt(records: Iterable[ReviewRecord], scope: str) -> int | None:
    pattern = re.compile(rf"_{re.escape(scope)}_attempt([0-9]+)$")
    attempts: set[int] = set()
    values = tuple(records)
    for record in values:
        match = pattern.search(_string(record.payload, "run_label"))
        if match is None:
            return None
        attempts.add(int(match.group(1)))
    return next(iter(attempts)) if len(attempts) == 1 else None


def _reviewer_provenance() -> dict[str, object]:
    source_path = Path(__file__).resolve()
    repository_root = source_path.parents[3]
    revision: str | None = None
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        candidate = completed.stdout.strip()
        if len(candidate) == 40:
            revision = candidate
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "repository_revision": revision,
        "source_path": "src/evrptw/experiments/stage052_native_architecture_review.py",
        "source_sha256": _sha256(source_path),
    }


def render_report(review: Mapping[str, object]) -> str:
    metrics = _mapping(review, "mode_metrics")
    performance = _mapping(review, "performance")
    differential = _mapping(review, "differential_gates")
    historical = _mapping(review, "historical_attempt72")
    cuda = _mapping(review, "cuda_evaluation_condition")
    raw_eligible_modes = review.get("eligible_production_modes")
    eligible_modes = raw_eligible_modes if isinstance(raw_eligible_modes, list) else []
    lines = [
        "# Stage 5.2 五种原生架构同机对比报告",
        "",
        f"- Review status（审查状态）：`{_string(review, 'review_status')}`",
        f"- Raw replay（原始重放）：`{review.get('axis_replay_passed')}`",
        f"- Axis count（轴数）：`{review.get('axis_count')}`",
        "- Eligible production modes（合格生产候选）：`"
        + ", ".join(value for value in eligible_modes if isinstance(value, str))
        + "`",
        "- Formal：未启动；production default（生产默认值）：未改变。",
        "",
        "## 五模式事实表",
        "",
        "| Mode | Solver median s | Effective iterations median | Exact calls median | "
        "Queue wait median s | Exact batch occupancy median | RSS median MiB | "
        "Artifact median MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        raw = _mapping(metrics, mode.value)
        lines.append(
            "| "
            + mode.value
            + " | "
            + _median_text(raw, "solver_seconds")
            + " | "
            + _median_text(raw, "effective_iterations")
            + " | "
            + _median_text(raw, "exact_started_calls")
            + " | "
            + _median_text(raw, "queue_wait_seconds")
            + " | "
            + _median_text(raw, "exact_backend_batch_occupancy")
            + " | "
            + _median_scaled_text(raw, "rss_bytes", 1024**2)
            + " | "
            + _median_scaled_text(raw, "artifact_bytes", 1024**2)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Throughput / resource envelope（吞吐与资源边界）",
            "",
            "| Mode | Iter/s | Candidate tx/s | Screened routes/s | Exact/s | "
            "CPU % of one core | Cache MiB | Persistence median s | Vehicle median |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        raw = _mapping(metrics, mode.value)
        lines.append(
            f"| {mode.value} | {_median_text(raw, 'effective_iterations_per_second')} | "
            f"{_median_text(raw, 'candidate_transactions_per_second')} | "
            f"{_median_text(raw, 'screened_routes_per_second')} | "
            f"{_median_text(raw, 'exact_started_per_second')} | "
            f"{_median_text(raw, 'cpu_utilization_percent_of_one_core')} | "
            f"{_median_scaled_text(raw, 'cache_memory_bytes', 1024**2)} | "
            f"{_median_text(raw, 'persistence_seconds')} | "
            f"{_median_text(raw, 'vehicle_count')} |"
        )
    lines.extend(
        [
            "",
            "## Correctness gates（正确性门控）",
            "",
            "| Mode | Fixed-work differential | Performance qualified | "
            "100-customer median improvement | Wall objective not worse |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        perf = performance.get(mode.value)
        perf_map = perf if isinstance(perf, dict) else {}
        diff = differential.get(mode.value)
        diff_map = diff if isinstance(diff, dict) else {}
        aggregate = perf_map.get("aggregate_100_customer")
        aggregate_map = aggregate if isinstance(aggregate, dict) else {}
        lines.append(
            f"| {mode.value} | {diff_map.get('passed', 'baseline')} | "
            f"{perf_map.get('performance_qualified', 'baseline')} | "
            f"{aggregate_map.get('median', 'n/a')} | "
            f"{perf_map.get('wall_clock_objective_not_worse', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## 实现与运维决策矩阵",
            "",
            "下表是实现边界与故障域的事实/工程判断，不替用户选择生产路线。",
            "",
            "| Mode | Python calls | Build complexity | Failure domain | "
            "Recovery difficulty | Maintenance cost |",
            "|---|---|---|---|---|---|",
            "| current_stage052 | 每批 candidate transaction | 中 | solve-local | 低 | 低 |",
            "| python_candidate_control | 每轮 Python control + worker IPC | 低 | "
            "worker pool | 中 | 中 |",
            "| per_solve_runtime | 每 candidate round 一次 C++ | 中 | "
            "solve-local runtime | 中 | 中 |",
            "| full_native_alns | 每 instance/seed 一次 C++ | 高 | 单个 native solve | 高 | 高 |",
            "| host_scheduler | shard 通过 UDS/shared memory | 最高 | "
            "run-wide scheduler | 最高 | 最高 |",
            "",
            "## 当前实现与历史 Pilot",
            "",
            "本轮 `current_stage052` 是所有速度比值和质量差值的主要分母。"
            "Accepted Pilot `attempt72` 仅用于长期漂移核验，不替代同机实测。",
            "",
            f"- Historical available（历史证据可用）：`{historical.get('available')}`",
            f"- Historical identity verified（历史身份已核验）："
            f"`{historical.get('identity_verified')}`",
            f"- Historical comparisons（历史配对数）：`{historical.get('comparison_count')}`",
            f"- CUDA condition（CUDA 条件）："
            f"`{cuda.get('condition_met')}`；{cuda.get('reason')}；"
            "本轮未自动运行 CUDA。",
            "",
            "## Instrumentation limitations（测量边界）",
            "",
            "Exact batch occupancy（精确批量占用度）不是 native candidate-screening "
            "occupancy（原生候选筛选占用度），不用于 CUDA 门槛。"
            "`persistence_seconds` 来自绑定主轴 SHA-256 的独立签名持久化回执；"
            "轴 JSON 只流式写入一次。"
            "Host 单轴的 process tree（进程树）包含共享 scheduler service；"
            "并发轴之间存在重叠，因此总 CPU/RSS 只采用 mode-wave 主计量，不跨轴求和。",
            "",
            "## 边界",
            "",
            "未启动 Formal Rerun16，未 push，未清理或覆盖历史 evidence，未切换默认架构。",
        ]
    )
    return "\n".join(lines) + "\n"


def _median_text(payload: Mapping[str, object], key: str) -> str:
    raw = payload.get(key)
    if not isinstance(raw, dict) or raw.get("median") is None:
        return "n/a"
    return f"{float(raw['median']):.6g}"


def _median_scaled_text(payload: Mapping[str, object], key: str, scale: float) -> str:
    raw = payload.get(key)
    if not isinstance(raw, dict) or raw.get("median") is None:
        return "n/a"
    return f"{float(raw['median']) / scale:.3f}"


def write_review(
    review: Mapping[str, object],
    *,
    output_json: Path,
    output_markdown: Path,
    raw_inventory_before: Mapping[str, object] | None = None,
    raw_inventory_after: Mapping[str, object] | None = None,
    command: Sequence[str] | None = None,
) -> None:
    recorded_inventory = _mapping(review, "raw_axis_inventory")
    before = dict(raw_inventory_before or recorded_inventory)
    after = dict(raw_inventory_after or recorded_inventory)
    if before != recorded_inventory or after != recorded_inventory:
        raise RuntimeError("raw architecture evidence changed during independent review")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(review, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output_json.write_text(data, encoding="utf-8")
    output_json.with_suffix(output_json.suffix + ".sha256").write_text(
        hashlib.sha256(data.encode("utf-8")).hexdigest() + "\n",
        encoding="ascii",
    )
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(render_report(review), encoding="utf-8")
    reviewer = _mapping(review, "reviewer_provenance")
    producer = _mapping(review, "producer_identity")
    manifest = {
        "schema_version": REVIEW_MANIFEST_SCHEMA_VERSION,
        "scope": _string(review, "scope"),
        "attempt": _integer(review, "attempt"),
        "axis_count": _integer(review, "axis_count"),
        "status": _string(review, "review_status"),
        "reviewer_module_name": REVIEWER_MODULE_NAME,
        "reviewer_provenance": dict(reviewer),
        "producer_identity": dict(producer),
        "raw_axis_inventory": dict(recorded_inventory),
        "files": {
            output_json.name: _sha256(output_json),
            output_json.with_suffix(output_json.suffix + ".sha256").name: _sha256(
                output_json.with_suffix(output_json.suffix + ".sha256")
            ),
            output_markdown.name: _sha256(output_markdown),
        },
    }
    manifest_path = output_json.with_name(output_json.stem + "_manifest.json")
    manifest_data = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    manifest_path.write_text(manifest_data, encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_data.encode("utf-8")).hexdigest()
    manifest_path.with_suffix(manifest_path.suffix + ".sha256").write_text(
        manifest_sha256 + "\n",
        encoding="ascii",
    )
    reviewer_source_sha256 = _string(reviewer, "source_sha256")
    execution = {
        "schema_version": REVIEW_EXECUTION_SCHEMA_VERSION,
        "run_label": (
            f"stage05.2_native_architecture_{_string(review, 'scope')}_"
            f"attempt{_integer(review, 'attempt'):02d}_review"
        ),
        "status": "completed",
        "finalized": True,
        "exit_code": 0,
        "reviewer_module_name": REVIEWER_MODULE_NAME,
        "reviewer_installed_distribution_digest": reviewer_source_sha256,
        "raw_manifest_sha256_before": _string(before, "tree_sha256"),
        "raw_manifest_sha256_after": _string(after, "tree_sha256"),
        "raw_manifest_unchanged": before == after,
        "raw_axis_count": _integer(after, "axis_count"),
        "review_manifest_sha256": manifest_sha256,
        "review_json_sha256": _sha256(output_json),
        "command": list(command or (sys.executable, "-m", REVIEWER_MODULE_NAME)),
    }
    execution_path = output_json.with_name(output_json.stem + "_execution.json")
    execution_data = (
        json.dumps(
            execution,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    execution_path.write_text(execution_data, encoding="utf-8")
    execution_path.with_suffix(execution_path.suffix + ".sha256").write_text(
        hashlib.sha256(execution_data.encode("utf-8")).hexdigest() + "\n",
        encoding="ascii",
    )


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("paired", "pilot"))
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/schneider"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    arguments = parser.parse_args(raw_argv)
    records = load_records(
        arguments.scope,
        attempt=arguments.attempt,
        results_root=arguments.results_root,
    )
    raw_inventory_before = _raw_axis_inventory(records)
    review = review_records(
        records,
        scope=arguments.scope,
        benchmark_dir=arguments.benchmark_dir,
    )
    reloaded_records = load_records(
        arguments.scope,
        attempt=arguments.attempt,
        results_root=arguments.results_root,
    )
    raw_inventory_after = _raw_axis_inventory(reloaded_records)
    write_review(
        review,
        output_json=arguments.output_json,
        output_markdown=arguments.output_markdown,
        raw_inventory_before=raw_inventory_before,
        raw_inventory_after=raw_inventory_after,
        command=(sys.executable, "-m", REVIEWER_MODULE_NAME, *raw_argv),
    )
    print(json.dumps(review, indent=2, sort_keys=True))
    if review["review_status"] == "COMPARISON_COMPLETE_QUALIFIED":
        return 0
    if review["review_status"] == "COMPARISON_COMPLETE_NOT_QUALIFIED":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "HISTORICAL_PILOT_ROOT",
    "REVIEW_EXECUTION_SCHEMA_VERSION",
    "REVIEW_MANIFEST_SCHEMA_VERSION",
    "REVIEW_SCHEMA_VERSION",
    "REVIEWER_MODULE_NAME",
    "ReviewRecord",
    "_historical_pilot",
    "_raw_axis_inventory",
    "_scheduler_screening_occupancy",
    "load_records",
    "render_report",
    "review_records",
    "write_review",
)
