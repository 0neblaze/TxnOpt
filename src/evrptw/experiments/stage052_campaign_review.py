"""Independent bounded replay for Stage 5.2 benchmark campaigns.

The module does not trust producer readiness fields.  Geometry, logical event
ordering, BKS compatibility, resource limits, and every physical checksum are
recomputed from the archived campaign evidence before a review status is
published.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import plistlib
import re
import shutil
import statistics
import subprocess
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeGuard

import orjson

from evrptw.artifacts import (
    ANYTIME_CHECKPOINT_SCHEMA,
    DIAGNOSTIC_SCHEMA,
    EVENTS_SCHEMA,
    ROUTE_DICTIONARY_SCHEMA,
    SCREENING_CHECKS_SCHEMA,
    V2_PARQUET_ROW_GROUP_SIZE,
    V3_SCREENING_DEFINITIONS_SCHEMA,
    V3_SCREENING_OCCURRENCES_SCHEMA,
    ArtifactIntegrityError,
    ArtifactReader,
    artifact_schema_fingerprint,
    signed_sidecar_matches,
)
from evrptw.best_known import BEST_KNOWN_VALUES
from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES
from evrptw.models import Instance
from evrptw.objective import ObjectiveComparison, SolutionObjective, compare_objectives
from evrptw.parser import parse_schneider
from evrptw.repository import repository_root
from evrptw.stage052 import (
    STAGE052_MAXIMUM_PERSISTENCE_RATIO,
    Stage052Component,
    stage052_contract,
)
from evrptw.stage052_campaign import (
    CHECKPOINT_SECONDS,
    GIB,
    AcceptedGlobalBest,
    AnytimeCheckpoint,
    BatchManifest,
    BenchmarkCampaignConfig,
    CampaignManifest,
    CampaignPlan,
    PilotStorageObservation,
    ProcessCpuCounterSample,
    StorageRootLocator,
    VolumeIdentity,
    directory_byte_count,
    directory_checksum,
    load_batch_manifest,
    load_campaign_manifest,
    maximum_process_average_cores,
)
from evrptw.stage052_evidence import (
    STAGE052_RESOURCE_SCHEMA_VERSION,
    BatchPersistenceEnvelope,
    Stage052PersistenceAttribution,
    validate_worker_ownership,
    verify_stage052_campaign_gate_set,
    verify_stage052_evidence_input,
    verify_stage052_review_files,
    verify_stage052_source_snapshot,
)
from evrptw.stage052_retention import (
    load_retention_registry,
    resolve_retained_run_from_locator,
)
from evrptw.stage052_review_service import ReviewProcessMemoryGuard, ReviewProgressLog
from evrptw.validation import validate_routes

NOT_READY = "NOT_READY"
PILOT_READY = "READY_FOR_STAGE052_FORMAL_BENCHMARK"
FORMAL_READY = "READY_FOR_STAGE05_3"
FORMAL_SEEDS = tuple(range(2014, 2024))
PILOT_SEEDS = (2014, 2015, 2016)
PER_WORKER_RSS_LIMIT_BYTES = 4_357_382_144
PROCESS_TREE_RSS_LIMIT_BYTES = 12 * 1024**3
CAMPAIGN_REVIEW_SCHEMA = "stage05.2-campaign-review-v1"
_REVIEW_EXECUTION_ENV = "STAGE052_REVIEW_EXECUTION_RECEIPT"
COMPACT_TRACE_SCHEMA = "stage05.2-campaign-trace-v1"
COMPACT_TRACE_MAX_BYTES = 16 * 1024 * 1024
MAX_STREAM_AUDIT_KEYS = 65_536
COMPACT_TRACE_SCHEMA_FINGERPRINTS = {
    "route_dictionary": artifact_schema_fingerprint(ROUTE_DICTIONARY_SCHEMA),
    "events": artifact_schema_fingerprint(EVENTS_SCHEMA),
    "screening_checks": artifact_schema_fingerprint(SCREENING_CHECKS_SCHEMA),
    "screening_definitions": artifact_schema_fingerprint(V3_SCREENING_DEFINITIONS_SCHEMA),
    "screening_occurrences": artifact_schema_fingerprint(V3_SCREENING_OCCURRENCES_SCHEMA),
    "diagnostic": artifact_schema_fingerprint(DIAGNOSTIC_SCHEMA),
}


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _is_sha256(value: object) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stable_input_provenance(value: Mapping[str, object]) -> dict[str, object]:
    """Keep solver inputs while excluding per-run machine/scope observations."""

    excluded = {
        "instance_sha256",
        "background_load",
        "power_mode",
        "runtime_signature",
        "process_status_counts",
        "load_average",
    }
    return {key: item for key, item in value.items() if key not in excluded}


def _validated_instance_hashes(value: Mapping[str, object]) -> dict[str, str]:
    raw = value.get("instance_sha256")
    if (
        not isinstance(raw, Mapping)
        or not raw
        or any(
            not isinstance(instance, str) or not instance or not _is_sha256(digest)
            for instance, digest in raw.items()
        )
    ):
        raise ArtifactIntegrityError("campaign instance hash mapping is invalid")
    return {str(instance): str(digest) for instance, digest in raw.items()}


@dataclass(frozen=True, slots=True)
class CampaignSelectionLockAudit:
    """Independent F02 selection/provenance lock consumed by G01 and G02."""

    passed: bool
    detail: str
    selection_lock: dict[str, object]


def validate_campaign_selection_lock(
    *,
    campaign_backend: str,
    campaign_exact_backend: str,
    campaign_workers: int,
    campaign_native_profile: str,
    prerequisite_metadata: Mapping[str, object],
    prerequisite_review: Mapping[str, object],
    prerequisite_identity: Mapping[str, object],
) -> CampaignSelectionLockAudit:
    """Recompute the selected F02 execution identity without trusting its gate text."""

    native = prerequisite_metadata.get("native_kernel_config")
    runtime = prerequisite_metadata.get("runtime_identity")
    provenance = prerequisite_metadata.get("performance_provenance")
    review_native = prerequisite_review.get("native_configuration")
    alternate_review_native = prerequisite_review.get("native_kernel_config")
    if review_native is None:
        review_native = alternate_review_native
    failures: list[str] = []
    required_native_configuration = {
        "enabled": True,
        "exact_charging": True,
        "screening": True,
        "propagation": True,
        "distance_matrix": True,
        "abi_version": "stage05.2-native-kernels-v1",
        "context_policy": "pack_once_per_solve",
        "failure_policy": "fail_fast_no_fallback",
    }
    raw_accelerator_decision = prerequisite_review.get("accelerator_decision")
    accelerator_decision = (
        raw_accelerator_decision if isinstance(raw_accelerator_decision, str) else ""
    )
    expected_backend = {
        "GPU_NOT_JUSTIFIED": "native_cpu",
        "NATIVE_CPU_RETAINED": "native_cpu",
        "ACCELERATOR_PROMOTED": "cuda",
    }.get(accelerator_decision)
    expected_profile = "cuda" if expected_backend == "cuda" else "native"
    if (
        expected_backend is None
        or campaign_backend != expected_backend
        or campaign_exact_backend != "cpu_batch"
        or prerequisite_metadata.get("execution_backend") != expected_backend
        or prerequisite_metadata.get("backend") != "cpu_batch"
        or prerequisite_review.get("selected_backend") != expected_backend
        or prerequisite_review.get("selected_exact_backend") != "cpu_batch"
    ):
        failures.append("selected accelerator execution/cpu_batch exact backend does not agree")
    if (
        isinstance(campaign_workers, bool)
        or campaign_workers not in {2, 4}
        or prerequisite_metadata.get("worker_count") != campaign_workers
        or prerequisite_review.get("selected_workers") != campaign_workers
    ):
        failures.append("selected worker count does not agree")
    if (
        campaign_native_profile != "stage05.2-native-kernels-v1"
        or not isinstance(native, Mapping)
        or dict(native) != required_native_configuration
        or review_native != native
        or (alternate_review_native is not None and alternate_review_native != native)
    ):
        failures.append("native configuration does not agree")
    if (
        prerequisite_metadata.get("optimization_profile") != expected_profile
        or prerequisite_review.get("selected_optimization_profile") != expected_profile
    ):
        failures.append("selected accelerator optimization profile does not agree")
    if (
        prerequisite_metadata.get("storage_policy_version") != "artifact-storage-v2"
        or prerequisite_metadata.get("screening_schema_version") != "screening_decisions_v3"
    ):
        failures.append("F02 storage/schema lock is invalid")
    expected_identity = {
        "run_label": prerequisite_metadata.get("run_label"),
        "component": prerequisite_metadata.get("component"),
        "scope": prerequisite_metadata.get("scope"),
        "status": prerequisite_review.get("status"),
        "repository_revision": prerequisite_metadata.get("repository_revision"),
        "configuration_sha256": prerequisite_metadata.get("configuration_sha256"),
        "raw_manifest_sha256": prerequisite_review.get("raw_manifest_sha256"),
    }
    if any(prerequisite_identity.get(field) != value for field, value in expected_identity.items()):
        failures.append("verified F02 raw/review identity does not agree with metadata")
    if (
        prerequisite_metadata.get("component") != Stage052Component.ACCELERATOR_PILOT.value
        or prerequisite_metadata.get("scope") != "performance"
        or prerequisite_review.get("component") != Stage052Component.ACCELERATOR_PILOT.value
        or prerequisite_review.get("scope") != "performance"
        or prerequisite_review.get("status") != "READY_FOR_STAGE052_BENCHMARK"
    ):
        failures.append("F02 component/scope/status identity is invalid")
    revision = prerequisite_metadata.get("repository_revision")
    configuration = prerequisite_metadata.get("configuration_sha256")
    raw_hash = prerequisite_identity.get("raw_manifest_sha256")
    review_hash = prerequisite_identity.get("review_manifest_sha256")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or not _is_sha256(configuration)
        or not _is_sha256(raw_hash)
        or not _is_sha256(review_hash)
    ):
        failures.append("F02 revision/config/raw/review digest is invalid")
    runtime_valid = isinstance(runtime, Mapping) and (
        runtime.get("schema_version") == "stage05.2-runtime-identity-v2"
        and runtime.get("repository_revision") == revision
        and runtime.get("installed_editable") is False
        and all(
            _is_sha256(runtime.get(field))
            for field in (
                "wheel_sha256",
                "python_executable_sha256",
                "native_extension_sha256",
                "dependency_manifest_sha256",
                "installed_distribution_sha256",
            )
        )
    )
    if not runtime_valid:
        failures.append("F02 non-editable runtime identity is invalid")
    if not isinstance(provenance, Mapping):
        failures.append("F02 input provenance is missing")
    if failures:
        return CampaignSelectionLockAudit(False, "; ".join(failures), {})
    assert isinstance(native, Mapping)
    assert isinstance(runtime, Mapping)
    assert isinstance(provenance, Mapping)
    assert isinstance(revision, str)
    assert isinstance(configuration, str)
    assert isinstance(raw_hash, str)
    assert isinstance(review_hash, str)
    lock: dict[str, object] = {
        "prerequisite_run_label": str(prerequisite_identity["run_label"]),
        "raw_manifest_sha256": raw_hash,
        "accelerator_review_manifest_sha256": review_hash,
        "selected_backend": expected_backend,
        "selected_exact_backend": "cpu_batch",
        "selected_workers": campaign_workers,
        "native_profile": campaign_native_profile,
        "selected_optimization_profile": expected_profile,
        "accelerator_decision": accelerator_decision,
        "repository_revision": revision,
        "configuration_sha256": configuration,
        "runtime_identity_sha256": _canonical_sha256(runtime),
        "input_provenance_sha256": _canonical_sha256(_stable_input_provenance(provenance)),
        "instance_sha256": dict(sorted(_validated_instance_hashes(provenance).items())),
        "native_config_sha256": _canonical_sha256(native),
        "native_kernel_config": dict(native),
    }
    return CampaignSelectionLockAudit(
        True,
        "F02 execution/exact backend, workers, native, schema, runtime, and input lock passed",
        lock,
    )


def validate_compact_trace_index(
    trace: Mapping[str, object],
    *,
    byte_size: int,
) -> None:
    """Fail fast unless a materialized trace is a bounded compact axis index."""

    required_references = (
        "route_dictionary_ref",
        "events_ref",
        "screening_checks_ref",
        "screening_definitions_ref",
        "screening_occurrences_ref",
        "diagnostic_ref",
    )
    forbidden_collections = {
        "route_dictionary",
        "route_evaluations",
        "events",
        "screening_decisions",
        "incremental_propagations",
        "operator_calls",
        "candidate_states",
        "deadline_events",
    }
    event_identity = trace.get("event_identity")
    schema_fingerprints = trace.get("schema_fingerprints")
    if (
        isinstance(byte_size, bool)
        or byte_size < 0
        or byte_size > COMPACT_TRACE_MAX_BYTES
        or trace.get("campaign_trace_schema_version") != COMPACT_TRACE_SCHEMA
        or trace.get("trace_storage_version") != "stage03-trace-index-v2"
        or not isinstance(trace.get("run_label"), str)
        or not isinstance(trace.get("instance"), str)
        or not _is_int(trace.get("seed"))
        or not isinstance(trace.get("axes"), Mapping)
        or any(
            not isinstance(trace.get(reference), str) or not trace.get(reference)
            for reference in required_references
        )
        or trace.get("screening_schema_version") != "screening_decisions_v3"
        or not isinstance(trace.get("lane_dictionary"), Mapping)
        or not isinstance(trace.get("operator_dictionary"), Mapping)
        or not isinstance(schema_fingerprints, Mapping)
        or dict(schema_fingerprints) != COMPACT_TRACE_SCHEMA_FINGERPRINTS
        or not isinstance(event_identity, Mapping)
        or event_identity.get("local_field") != "event_id"
        or not _is_int(event_identity.get("shard_ordinal"))
        or bool(forbidden_collections.intersection(trace))
    ):
        raise ValueError("compact trace index schema or 16-MiB size boundary failed")


@dataclass(frozen=True, slots=True)
class CampaignGeometryRecord:
    """One independently observed ``(instance, seed)`` shard geometry."""

    shard_id: str
    batch_id: str
    instance: str
    seed: int
    customer_count: int
    budgets_seconds: tuple[int, ...]
    checkpoint_count: int


@dataclass(frozen=True, slots=True)
class CampaignGeometryAudit:
    passed: bool
    detail: str
    shard_count: int
    axis_count: int
    declared_solver_seconds: int
    checkpoint_count: int


def validate_formal_geometry(
    records: Iterable[CampaignGeometryRecord],
) -> CampaignGeometryAudit:
    """Recompute the exact Formal geometry without trusting producer totals."""

    observed = tuple(records)
    axis_count = sum(len(record.budgets_seconds) for record in observed)
    declared_seconds = sum(sum(record.budgets_seconds) for record in observed)
    checkpoint_count = sum(record.checkpoint_count for record in observed)
    totals = (len(observed), axis_count, declared_seconds, checkpoint_count)
    expected_totals = (920, 2_040, 229_200, 10_400)
    if totals != expected_totals:
        return CampaignGeometryAudit(
            False,
            "Formal totals mismatch: expected 920 shards, 2040 axes, "
            "229200 declared seconds, and 10400 checkpoints; "
            f"observed={totals}",
            *totals,
        )

    canonical_counts = {item.instance: item.customer_count for item in BEST_KNOWN_VALUES}
    expected_identities = tuple(
        sorted(
            (customer_count, instance, seed)
            for instance, customer_count in canonical_counts.items()
            for seed in FORMAL_SEEDS
        )
    )
    observed_identities = tuple(
        (record.customer_count, record.instance, record.seed) for record in observed
    )
    expected_shards = tuple(f"shard{ordinal:04d}" for ordinal in range(1, 921))
    observed_shards = tuple(record.shard_id for record in observed)
    if observed_identities != expected_identities or observed_shards != expected_shards:
        return CampaignGeometryAudit(
            False,
            "Formal shard identity/order mismatch for exact 92 instances and seeds 2014-2023",
            *totals,
        )
    if len({(record.batch_id, record.shard_id) for record in observed}) != 920:
        return CampaignGeometryAudit(
            False,
            "Formal batch/shard identities are duplicate",
            *totals,
        )
    for record in observed:
        expected_budgets = (30,) if record.customer_count < 100 else (30, 60, 300)
        expected_checkpoints = 4 if record.customer_count < 100 else 16
        if (
            canonical_counts.get(record.instance) != record.customer_count
            or record.seed not in FORMAL_SEEDS
            or record.budgets_seconds != expected_budgets
            or record.checkpoint_count != expected_checkpoints
        ):
            return CampaignGeometryAudit(
                False,
                f"non-canonical Formal shard geometry: {record.shard_id}",
                *totals,
            )
    return CampaignGeometryAudit(
        True,
        "exact 92-instance x 10-seed Formal geometry passed",
        *totals,
    )


@dataclass(frozen=True, slots=True)
class StreamedEventAudit:
    passed: bool
    detail: str
    event_count: int
    exact_started: int
    exact_completed: int
    accepted_candidates: int
    global_bests: int
    deadline_axes: tuple[str, ...]
    axis_exact_counts: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True, slots=True)
class AxisEvidenceReconciliationAudit:
    """Bounded reconciliation of raw wall-clock axes and streamed events."""

    passed: bool
    detail: str
    raw_exact_started: int
    raw_exact_completed: int


def validate_axis_event_reconciliation(
    raw_axes: Mapping[str, object],
    event_audit: StreamedEventAudit,
    *,
    customer_count: int,
) -> AxisEvidenceReconciliationAudit:
    """Require exact-call equality plus canonical wall-clock termination semantics."""

    failures: list[str] = []
    raw_started = 0
    raw_completed = 0
    expected_deadlines: set[str] = set()
    raw_by_axis: dict[str, tuple[int, int]] = {}
    for axis, raw_value in raw_axes.items():
        if (
            not axis.startswith("wall_clock_")
            or not axis.removeprefix("wall_clock_").isdigit()
            or not isinstance(raw_value, Mapping)
        ):
            failures.append(f"invalid raw wall-clock axis: {axis}")
            continue
        budget = int(axis.removeprefix("wall_clock_"))
        runtime = _finite_number(raw_value.get("runtime_seconds"))
        termination = raw_value.get("termination_reason")
        started = raw_value.get("started_calls")
        completed = raw_value.get("completed_calls")
        if not _is_int(started) or not _is_int(completed):
            failures.append(f"invalid raw exact counters on {axis}")
            continue
        raw_started += started
        raw_completed += completed
        raw_by_axis[axis] = (started, completed)
        if termination == "wall_clock_deadline":
            expected_deadlines.add(axis)
            if runtime is None or not math.isclose(
                runtime,
                budget,
                rel_tol=0.0,
                abs_tol=1.0,
            ):
                failures.append(f"deadline runtime mismatch on {axis}")
        elif termination == "iteration_limit":
            if customer_count >= 100 or runtime is None or runtime <= 0.0 or runtime > budget:
                failures.append(f"iteration-limit runtime is invalid on {axis}")
        else:
            failures.append(f"unsupported wall-clock termination on {axis}")
    if raw_started != event_audit.exact_started:
        failures.append("raw/event started exact-call counts differ")
    if raw_completed != event_audit.exact_completed:
        failures.append("raw/event completed exact-call counts differ")
    event_by_axis = {
        axis: (started, completed) for axis, started, completed in event_audit.axis_exact_counts
    }
    if raw_by_axis != event_by_axis:
        failures.append("raw/event per-axis exact-call counts differ")
    if expected_deadlines != set(event_audit.deadline_axes):
        failures.append("raw/event deadline axes differ")
    return AxisEvidenceReconciliationAudit(
        not failures,
        "; ".join(failures) if failures else "raw/event runtime and counters passed",
        raw_started,
        raw_completed,
    )


@dataclass(slots=True)
class _AxisEventState:
    budget_seconds: int
    last_evaluation_id: int = 0
    exact_started: int = 0
    exact_completed: int = 0
    deadline_seen: bool = False
    accepted_candidates: int = 0
    global_bests: int = 0


def audit_streamed_events(
    events: Iterable[Mapping[str, object]],
    axis_budgets: Mapping[str, int],
) -> StreamedEventAudit:
    """Audit ordering and transaction boundaries in one bounded event stream."""

    if not axis_budgets or any(
        not axis or not _is_int(budget) or budget <= 0 for axis, budget in axis_budgets.items()
    ):
        raise ValueError("axis_budgets must contain positive integer budgets")
    states = {axis: _AxisEventState(budget_seconds=budget) for axis, budget in axis_budgets.items()}
    previous_event_id = 0
    count = 0
    failures: list[str] = []
    observed_axes: set[str] = set()
    cache_keys: dict[str, set[str]] = {axis: set() for axis in axis_budgets}
    cache_misses: dict[str, set[str]] = {axis: set() for axis in axis_budgets}
    completed_exact_keys: dict[str, set[str]] = {axis: set() for axis in axis_budgets}
    for event in events:
        count += 1
        event_id = event.get("event_id")
        if not _is_int(event_id) or event_id <= previous_event_id:
            failures.append("event IDs are not strictly increasing")
            break
        previous_event_id = event_id
        raw_axis = event.get("benchmark_axis")
        axis = str(raw_axis) if raw_axis else str(event.get("lane", "")).partition(":")[0]
        state = states.get(axis)
        if state is None:
            failures.append(f"event refers to an unknown benchmark axis: {axis}")
            continue
        observed_axes.add(axis)
        event_type = str(event.get("event_type", event.get("record_type", "")))
        fallback_fields = (
            event.get("native_fallback"),
            event.get("native_fallbacks"),
            event.get("native_protocol_fallback"),
            event.get("native_protocol_fallbacks"),
        )
        explicit_fallback = any(
            value is True or (_is_int(value) and value > 0) for value in fallback_fields
        )
        failure_text = str(event.get("failure_reason", "")).casefold()
        if explicit_fallback or (event_type == "execution_error" and "fallback" in failure_text):
            failures.append(f"native/protocol fallback observed on {axis}")

        if event_type == "deadline_boundary":
            state.deadline_seen = True
            timestamp = _finite_number(event.get("timestamp_seconds"))
            if timestamp is None or not math.isclose(
                timestamp,
                state.budget_seconds,
                rel_tol=0.0,
                abs_tol=1.0,
            ):
                failures.append(f"deadline boundary timestamp mismatch on {axis}")

        if event_type == "route_evaluation":
            evaluation_id = event.get("evaluation_id")
            if not _is_int(evaluation_id) or evaluation_id != state.last_evaluation_id + 1:
                failures.append(f"route evaluation ordering mismatch on {axis}")
            else:
                state.last_evaluation_id = evaluation_id
        if event_type == "route_evaluation" and event.get("exact_started") is True:
            if state.deadline_seen:
                failures.append(f"exact work started after deadline on {axis}")
            state.exact_started += 1
            started_at = _finite_number(event.get("started_at"))
            completed_at = _finite_number(event.get("completed_at"))
            if started_at is None or started_at < 0.0:
                failures.append(f"invalid exact start time on {axis}")
            if event.get("exact_completed") is True:
                state.exact_completed += 1
                if (
                    completed_at is None
                    or started_at is None
                    or completed_at < started_at
                    or completed_at > state.budget_seconds
                ):
                    failures.append(f"exact completion crosses deadline on {axis}")
                digest = str(event.get("cache_key_digest", ""))
                if not digest or digest not in cache_misses[axis]:
                    failures.append(f"exact completion lacks a preceding cache miss on {axis}")
                else:
                    cache_misses[axis].discard(digest)
                    completed_exact_keys[axis].add(digest)
            if event.get("deadline_boundary"):
                state.deadline_seen = True

        if event_type == "cache_event":
            operation = str(event.get("operation", ""))
            digest = str(event.get("cache_key_digest", ""))
            if state.deadline_seen and operation == "store":
                failures.append(f"cache store observed after deadline on {axis}")
            if operation == "store":
                if not digest:
                    failures.append(f"cache store lacks key on {axis}")
                elif digest not in completed_exact_keys[axis]:
                    failures.append(f"cache store precedes exact completion on {axis}")
                completed_exact_keys[axis].discard(digest)
                cache_keys[axis].add(digest)
            elif operation == "evict":
                if digest not in cache_keys[axis]:
                    failures.append(f"cache eviction refers to an absent key on {axis}")
                cache_keys[axis].discard(digest)
            elif operation == "lookup_result":
                if event.get("lookup_result") == "hit":
                    if digest and digest not in cache_keys[axis]:
                        failures.append(f"cache hit precedes store on {axis}")
                elif event.get("lookup_result") == "miss":
                    if not digest:
                        failures.append(f"cache miss lacks key on {axis}")
                    cache_misses[axis].add(digest)

        if event_type == "candidate_state":
            accepted = event.get("accepted") is True
            global_best = event.get("global_best") is True
            if accepted:
                state.accepted_candidates += 1
                if state.deadline_seen:
                    failures.append(f"candidate accepted after deadline on {axis}")
                vehicle_delta = event.get("candidate_vehicle_delta")
                if not _is_int(vehicle_delta) or vehicle_delta > 0:
                    failures.append(f"accepted candidate increased vehicle count on {axis}")
                timestamp = _finite_number(event.get("timestamp_seconds"))
                if timestamp is None or timestamp > state.budget_seconds:
                    failures.append(f"candidate accepted beyond wall-clock budget on {axis}")
                if event.get("status") != "accepted" or event.get("candidate_feasible") is not True:
                    failures.append(f"accepted candidate transaction is inconsistent on {axis}")
            if global_best:
                state.global_bests += 1
                if not accepted:
                    failures.append(f"global best is not an accepted candidate on {axis}")
                route_keys = event.get("candidate_route_keys")
                if (
                    not isinstance(route_keys, (list, tuple))
                    or not route_keys
                    or any(not isinstance(key, str) or not key for key in route_keys)
                ):
                    failures.append(f"global best lacks candidate route identity on {axis}")
            cache_misses[axis].clear()
            completed_exact_keys[axis].clear()

        if any(
            len(items) > MAX_STREAM_AUDIT_KEYS
            for items in (
                cache_keys[axis],
                cache_misses[axis],
                completed_exact_keys[axis],
            )
        ):
            failures.append(f"bounded cache audit key limit exceeded on {axis}")
            break

    if observed_axes != set(axis_budgets):
        failures.append(
            "event stream axis coverage mismatch: "
            f"expected={sorted(axis_budgets)} observed={sorted(observed_axes)}"
        )
    started = sum(state.exact_started for state in states.values())
    completed = sum(state.exact_completed for state in states.values())
    if completed > started:
        failures.append("completed exact calls exceed started exact calls")
    accepted_count = sum(state.accepted_candidates for state in states.values())
    global_bests = sum(state.global_bests for state in states.values())
    return StreamedEventAudit(
        not failures,
        "; ".join(dict.fromkeys(failures)) if failures else "streamed event audit passed",
        count,
        started,
        completed,
        accepted_count,
        global_bests,
        tuple(sorted(axis for axis, state in states.items() if state.deadline_seen)),
        tuple(
            (axis, state.exact_started, state.exact_completed)
            for axis, state in sorted(states.items())
        ),
    )


@dataclass(frozen=True, slots=True)
class BKSReferenceAudit:
    passed: bool
    detail: str
    row_count: int


def validate_bks_reference(path: Path) -> BKSReferenceAudit:
    """Require exact incompatible BKS coverage and forbid gap computation."""

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            rows = tuple(reader)
    except OSError as error:
        return BKSReferenceAudit(False, f"cannot read BKS reference: {error}", 0)
    gap_fields = tuple(field for field in fields if "gap" in field.casefold())
    if gap_fields:
        return BKSReferenceAudit(False, f"BKS gap columns are forbidden: {gap_fields}", len(rows))
    expected = {item.instance for item in BEST_KNOWN_VALUES}
    observed = {str(row.get("instance", "")) for row in rows}
    if len(rows) != 92 or observed != expected:
        return BKSReferenceAudit(
            False,
            "BKS reference does not cover the exact 92 instances",
            len(rows),
        )
    incompatible = all(
        str(row.get("model_compatible", "")).strip().casefold() == "false" for row in rows
    )
    if not incompatible:
        return BKSReferenceAudit(False, "every BKS row must set model_compatible=False", len(rows))
    return BKSReferenceAudit(True, "exact incompatible BKS coverage passed", len(rows))


def review_status_for_scope(scope: str, *, passed: bool) -> str:
    """Map a passed review to the only status allowed for its scope."""

    if scope not in {"pilot", "formal"}:
        raise ValueError("campaign review scope must be pilot or formal")
    if not passed:
        return NOT_READY
    return PILOT_READY if scope == "pilot" else FORMAL_READY


@dataclass(slots=True)
class _ReviewEvidence:
    per_run_rows: list[dict[str, object]]
    checkpoint_rows: list[dict[str, object]]
    resource_rows: list[dict[str, object]]
    persistence_rows: list[dict[str, object]]
    geometry: list[CampaignGeometryRecord]
    selection_lock: dict[str, object]
    staging_root_aliases: set[str]
    archive_root_aliases: set[str]
    verified_archive_root_aliases: set[str]
    standard_raw_manifest_sha256: str
    campaign_persistence_attribution_sha256: str
    campaign_persistence_attribution_sidecar_sha256: str

    @classmethod
    def empty(cls) -> _ReviewEvidence:
        return cls([], [], [], [], [], {}, set(), set(), set(), "", "", "")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_int(value: object, field: str) -> int:
    if _is_int(value):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ArtifactIntegrityError(f"{field} must be an integer")


def _strict_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ArtifactIntegrityError(f"{field} must be numeric")
    try:
        result = float(value)
    except ValueError as error:
        raise ArtifactIntegrityError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise ArtifactIntegrityError(f"{field} must be finite")
    return result


def _json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(f"cannot read JSON evidence: {path}") from error
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError(f"JSON evidence is not an object: {path}")
    return payload


def _one_artifact(
    reader: ArtifactReader,
    artifact_type: str,
    *,
    subtype: str | None = None,
) -> Mapping[str, object]:
    matches = [
        item
        for item in reader.manifest.get("artifacts", ())
        if isinstance(item, Mapping)
        and item.get("artifact_type") == artifact_type
        and (subtype is None or item.get("artifact_subtype") == subtype)
    ]
    if len(matches) != 1:
        raise ArtifactIntegrityError(
            f"expected one {artifact_type}/{subtype or '*'} artifact, observed {len(matches)}"
        )
    return matches[0]


def _artifacts_for_directory(
    shard_manifest: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    payload = shard_manifest.get("artifacts")
    if not isinstance(payload, list) or any(not isinstance(item, Mapping) for item in payload):
        raise ArtifactIntegrityError("shard manifest artifacts are invalid")
    return tuple(item for item in payload if isinstance(item, Mapping))


def _verified_primary_descriptor(
    reader: ArtifactReader,
    nested: Mapping[str, object],
) -> Mapping[str, object]:
    """Bind one shard descriptor exactly to the verified batch manifest."""

    relative = nested.get("relative_path")
    matches = [
        item
        for item in reader.manifest.get("artifacts", ())
        if isinstance(item, Mapping) and item.get("relative_path") == relative
    ]
    if len(matches) != 1 or dict(matches[0]) != dict(nested):
        raise ArtifactIntegrityError(
            "shard artifact descriptor differs from the verified batch descriptor"
        )
    if not isinstance(relative, str):
        raise ArtifactIntegrityError("shard artifact relative path is invalid")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ArtifactIntegrityError("shard artifact relative path escapes the batch")
    physical = reader.result.run_dir / path
    byte_size = _strict_int(matches[0].get("byte_size"), "artifact byte_size")
    if not physical.is_file() or physical.stat().st_size != byte_size:
        raise ArtifactIntegrityError("shard artifact physical byte size differs")
    return matches[0]


def _read_bounded_compact_trace(
    reader: ArtifactReader,
    trace_ref: Mapping[str, object],
) -> dict[str, object]:
    """Enforce the physical 16 MiB boundary before materialising compact JSON."""

    descriptor = _verified_primary_descriptor(reader, trace_ref)
    byte_size = _strict_int(descriptor.get("byte_size"), "trace byte_size")
    if byte_size > COMPACT_TRACE_MAX_BYTES:
        raise ArtifactIntegrityError("compact trace index exceeds the 16 MiB bound")
    trace = reader.read_json(str(descriptor["relative_path"]))
    try:
        validate_compact_trace_index(trace, byte_size=byte_size)
    except ValueError as error:
        raise ArtifactIntegrityError(str(error)) from error
    return trace


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as error:
        raise ArtifactIntegrityError(f"cannot read CSV evidence: {path}") from error


def _sidecar_for(path: Path) -> Path:
    candidates = (
        path.with_suffix(".sha256"),
        path.with_suffix(path.suffix + ".sha256"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ArtifactIntegrityError(f"manifest sidecar is missing: {path}")


def _verify_manifest_sidecar(path: Path) -> None:
    sidecar = _sidecar_for(path)
    if not signed_sidecar_matches(path, sidecar):
        raise ArtifactIntegrityError(f"manifest sidecar mismatch: {path}")


def _batch_reader(batch_dir: Path, *, run_label: str) -> ArtifactReader:
    """Verify one batch bundle through the shared strict envelope reader."""

    reader = ArtifactReader(batch_dir)
    manifest = reader.manifest
    policy = manifest.get("storage_policy")
    if (
        manifest.get("run_label") != run_label
        or manifest.get("component") != Stage052Component.BENCHMARK.value
        or manifest.get("status") != "complete"
        or manifest.get("evidence_completeness") != "complete"
        or manifest.get("storage_policy_version") != "artifact-storage-v2"
        or not isinstance(policy, Mapping)
        or policy.get("screening_schema_version") != "screening_decisions_v3"
    ):
        raise ArtifactIntegrityError("batch artifact bundle identity is invalid")
    return reader


def _default_volume_probe(path: Path) -> VolumeIdentity:
    try:
        completed = subprocess.run(
            ("diskutil", "info", "-plist", str(path)),
            check=True,
            capture_output=True,
        )
        payload = plistlib.loads(completed.stdout)
        device_uuid = payload.get("VolumeUUID") or payload.get("DiskUUID")
        filesystem = payload.get("FilesystemName") or payload.get("FilesystemType")
    except (OSError, subprocess.CalledProcessError, plistlib.InvalidFileException) as error:
        raise ArtifactIntegrityError(f"cannot probe storage volume for {path}") from error
    if not isinstance(device_uuid, str) or not isinstance(filesystem, str):
        raise ArtifactIntegrityError(f"storage volume identity is incomplete for {path}")
    return VolumeIdentity(device_uuid=device_uuid, filesystem=filesystem)


def _axis_budget(axis: str) -> int:
    prefix = "wall_clock_"
    if not axis.startswith(prefix) or not axis.removeprefix(prefix).isdigit():
        raise ArtifactIntegrityError(f"unsupported campaign axis: {axis}")
    return int(axis.removeprefix(prefix))


def _objective_key(value: object, field: str) -> tuple[int, float, float, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ArtifactIntegrityError(f"{field} must contain four objective components")
    objective = SolutionObjective(
        vehicle_count=_strict_int(value[0], f"{field}.vehicle_count"),
        total_distance=_strict_float(value[1], f"{field}.total_distance"),
        total_charging_time=_strict_float(value[2], f"{field}.total_charging_time"),
        charging_count=_strict_int(value[3], f"{field}.charging_count"),
    )
    if objective.vehicle_count <= 0 or objective.key != tuple(value):
        raise ArtifactIntegrityError(f"{field} is not a canonical objective key")
    return objective.key


def _native_axis_valid(
    raw_axis: Mapping[str, object],
    trace_axis: Mapping[str, object],
) -> tuple[bool, str]:
    backend = raw_axis.get("backend_metrics")
    result = trace_axis.get("result_summary")
    if (
        raw_axis.get("backend") != "cpu_batch"
        or not isinstance(backend, Mapping)
        or not isinstance(result, Mapping)
    ):
        return False, "cpu_batch backend/result summary is missing"
    try:
        started = _strict_int(raw_axis.get("started_calls"), "started_calls")
        completed = _strict_int(raw_axis.get("completed_calls"), "completed_calls")
        exact_calls = _strict_int(backend.get("exact_calls"), "exact_calls")
        batch_launches = _strict_int(backend.get("batch_launches"), "batch_launches")
        work_batches = _strict_int(backend.get("work_batches"), "work_batches")
        invocations = _strict_int(backend.get("native_invocations"), "native_invocations")
        fallbacks = _strict_int(backend.get("native_fallbacks"), "native_fallbacks")
        occupancies_raw = backend.get("launch_occupancies")
        if not isinstance(occupancies_raw, list):
            raise ArtifactIntegrityError("launch_occupancies must be an array")
        occupancies = tuple(_strict_int(value, "launch_occupancy") for value in occupancies_raw)
        pipeline = trace_axis.get("persistence_pipeline")
        if not isinstance(pipeline, Mapping):
            raise ArtifactIntegrityError("bounded async persistence pipeline is missing")
        submitted_batches = _strict_int(pipeline.get("submitted_batches"), "submitted_batches")
        completed_batches = _strict_int(pipeline.get("completed_batches"), "completed_batches")
        writer_active_ns = _strict_int(
            pipeline.get("writer_active_nanoseconds"), "writer_active_nanoseconds"
        )
        writer_cpu_ns = _strict_int(
            pipeline.get("writer_cpu_nanoseconds"), "writer_cpu_nanoseconds"
        )
        producer_active_ns = _strict_int(
            pipeline.get("producer_active_nanoseconds"), "producer_active_nanoseconds"
        )
        union_ns = _strict_int(
            pipeline.get("persistence_union_nanoseconds"),
            "persistence_union_nanoseconds",
        )
        solver_union_ns = _strict_int(
            pipeline.get("solver_persistence_union_nanoseconds"),
            "solver_persistence_union_nanoseconds",
        )
        solver_critical_ns = _strict_int(
            pipeline.get("solver_persistence_critical_path_nanoseconds"),
            "solver_persistence_critical_path_nanoseconds",
        )
        solver_producer_ns = _strict_int(
            pipeline.get("solver_producer_active_nanoseconds"),
            "solver_producer_active_nanoseconds",
        )
        solver_writer_cpu_ns = _strict_int(
            pipeline.get("solver_writer_cpu_nanoseconds"),
            "solver_writer_cpu_nanoseconds",
        )
        producer_wait_ns = _strict_int(
            pipeline.get("producer_wait_nanoseconds"), "producer_wait_nanoseconds"
        )
        ledger = pipeline.get("batch_ledger")
        if not isinstance(ledger, list):
            raise ArtifactIntegrityError("bounded async persistence batch ledger is missing")
        for ordinal, entry in enumerate(ledger):
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"ordinal", "row_count", "event_token_sha256"}
                or _strict_int(entry.get("ordinal"), "batch ordinal") != ordinal
                or not (
                    0
                    < _strict_int(entry.get("row_count"), "batch row_count")
                    <= V2_PARQUET_ROW_GROUP_SIZE
                )
                or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("event_token_sha256", ""))) is None
            ):
                raise ArtifactIntegrityError("bounded async persistence batch ledger is invalid")
        if (
            pipeline.get("mode") != "bounded_async_thread"
            or _strict_int(pipeline.get("queue_max_batches"), "queue_max_batches") != 1
            or _strict_float(
                pipeline.get("writer_thread_switch_interval_seconds"),
                "writer_thread_switch_interval_seconds",
            )
            != 0.05
            or submitted_batches <= 0
            or completed_batches != submitted_batches
            or writer_active_ns <= 0
            or writer_cpu_ns < 0
            or producer_active_ns <= 0
            or producer_wait_ns < 0
            or solver_union_ns <= 0
            or solver_critical_ns != max(solver_producer_ns, solver_writer_cpu_ns)
            or solver_critical_ns > solver_union_ns
            or solver_producer_ns <= 0
            or solver_writer_cpu_ns < 0
            or solver_producer_ns > producer_active_ns
            or solver_writer_cpu_ns > writer_cpu_ns
            or max(writer_active_ns, producer_active_ns) > union_ns
            or union_ns > writer_active_ns + producer_active_ns
            or solver_union_ns > union_ns
            or _strict_int(pipeline.get("peak_queued_batches"), "peak_queued_batches") != 1
            or len(ledger) != submitted_batches
        ):
            raise ArtifactIntegrityError("bounded async persistence pipeline is invalid")
    except ArtifactIntegrityError as error:
        return False, str(error)
    screening = result.get("screening_statistics")
    if not isinstance(screening, Mapping):
        return False, "screening statistics are missing"
    try:
        protocol_fallbacks = _strict_int(
            screening.get("native_protocol_fallbacks"),
            "native_protocol_fallbacks",
        )
        trace_started = _strict_int(result.get("exact_started_calls"), "trace started calls")
        trace_completed = _strict_int(result.get("exact_completed_calls"), "trace completed calls")
    except ArtifactIntegrityError as error:
        return False, str(error)
    passed = (
        started == exact_calls == sum(occupancies)
        and completed == trace_completed
        and started == trace_started
        and batch_launches == work_batches == invocations == len(occupancies)
        and all(value > 0 for value in occupancies)
        and fallbacks == 0
        and protocol_fallbacks == 0
        and 0 <= completed <= started
    )
    return (
        passed,
        "native counters and no-fallback passed"
        if passed
        else "native counters, exact-call ordering, or no-fallback failed",
    )


def _pipeline_event_token_from_logical_row(
    row: Mapping[str, object],
) -> tuple[object, ...]:
    get = row.get
    kind = get("kind") or None
    operation = get("operation") or None
    status = get("status") or None
    reason = get("reason") or None
    return (
        str(get("event_type", get("record_type", "event"))),
        get("benchmark_axis"),
        get("lane"),
        get("iteration"),
        get("operator"),
        get("route_key"),
        get("decision_id"),
        kind,
        operation,
        status,
        reason,
        get("exact_started"),
        get("exact_completed"),
        get("feasible"),
    )


@dataclass(slots=True)
class _PipelineLedgerReplay:
    ledger: list[object]
    index: int = 0
    seen: int = 0
    hasher: Any = None

    def __post_init__(self) -> None:
        if self.hasher is None:
            self.hasher = hashlib.sha256()


def _audit_async_persistence_ledgers(
    events: Iterable[Mapping[str, object]],
    trace_axes: Mapping[str, object],
) -> None:
    """Recompute every physical async batch digest from the logical event stream."""

    states: dict[str, _PipelineLedgerReplay] = {}
    for axis, raw_trace_axis in trace_axes.items():
        if not isinstance(raw_trace_axis, Mapping):
            raise ArtifactIntegrityError("trace axis is invalid for persistence replay")
        pipeline = raw_trace_axis.get("persistence_pipeline")
        if not isinstance(pipeline, Mapping):
            raise ArtifactIntegrityError("persistence pipeline is missing for ledger replay")
        ledger = pipeline.get("batch_ledger")
        if not isinstance(ledger, list):
            raise ArtifactIntegrityError("persistence batch ledger is missing")
        states[str(axis)] = _PipelineLedgerReplay(ledger=list(ledger))
    for row in events:
        axis = str(row.get("benchmark_axis") or str(row.get("lane", "")).partition(":")[0])
        state = states.get(axis)
        if state is None:
            raise ArtifactIntegrityError(
                f"persistence event refers to an unknown benchmark axis: {axis}"
            )
        ledger = state.ledger
        index = state.index
        if index >= len(ledger):
            raise ArtifactIntegrityError(
                f"persistence ledger ended before the event stream: {axis}"
            )
        entry = ledger[index]
        hasher = state.hasher
        if not isinstance(entry, Mapping) or not hasattr(hasher, "update"):
            raise ArtifactIntegrityError("persistence ledger replay state is invalid")
        hasher.update(orjson.dumps(_pipeline_event_token_from_logical_row(row)) + b"\n")
        seen = state.seen + 1
        expected_rows = _strict_int(entry.get("row_count"), "persistence ledger row_count")
        if seen == expected_rows:
            if hasher.hexdigest() != entry.get("event_token_sha256"):
                raise ArtifactIntegrityError(
                    f"persistence batch digest does not replay: {axis}/{index}"
                )
            state.index = index + 1
            state.seen = 0
            state.hasher = hashlib.sha256()
        elif seen > expected_rows:
            raise ArtifactIntegrityError(f"persistence batch row count overflow: {axis}/{index}")
        else:
            state.seen = seen
    for axis, state in states.items():
        ledger = state.ledger
        if state.index != len(ledger) or state.seen != 0:
            raise ArtifactIntegrityError(
                f"persistence ledger does not cover the complete event stream: {axis}"
            )


def _route_sequence_from_key(route_key: str) -> list[str]:
    prefix = "route:"
    if not route_key.startswith(prefix):
        raise ArtifactIntegrityError(f"invalid candidate route key: {route_key}")
    encoded = route_key[len(prefix) :]
    if not encoded:
        return []
    sequence: list[str] = []
    for token in encoded.split("|"):
        length_text, separator, customer = token.partition(":")
        if not separator or not length_text.isdigit() or int(length_text) != len(customer):
            raise ArtifactIntegrityError(f"invalid candidate route key: {route_key}")
        sequence.append(customer)
    return sequence


def summarize_streamed_global_bests(
    events: Iterable[Mapping[str, object]],
    axis_budgets: Mapping[str, int],
    *,
    instance: Instance,
) -> dict[str, tuple[AcceptedGlobalBest, ...]]:
    last: dict[str, AcceptedGlobalBest | None] = {axis: None for axis in axis_budgets}
    visible: dict[str, dict[int, AcceptedGlobalBest]] = {axis: {} for axis in axis_budgets}
    for event in events:
        if (
            str(event.get("event_type", event.get("record_type", ""))) != "candidate_state"
            or event.get("accepted") is not True
            or event.get("global_best") is not True
        ):
            continue
        axis = str(event.get("benchmark_axis", ""))
        if axis not in last:
            raise ArtifactIntegrityError(f"global-best event has unknown axis: {axis}")
        route_keys = event.get("candidate_route_keys")
        if (
            not isinstance(route_keys, (list, tuple))
            or not route_keys
            or any(not isinstance(key, str) or not key for key in route_keys)
        ):
            raise ArtifactIntegrityError(
                f"global-best event lacks candidate route identity on {axis}"
            )
        candidate_routes = [_route_sequence_from_key(str(key)) for key in route_keys]
        report = validate_routes(instance, candidate_routes)
        if not report.feasible:
            raise ArtifactIntegrityError(f"global-best candidate routes fail validation on {axis}")
        replayed_objective = SolutionObjective.from_report(instance, report).key
        recorded_objective = _objective_key(
            event.get("candidate_objective_key"), "candidate objective"
        )
        if (
            compare_objectives(
                SolutionObjective(*replayed_objective),
                SolutionObjective(*recorded_objective),
            )
            is not ObjectiveComparison.EQUAL
        ):
            raise ArtifactIntegrityError(
                f"global-best candidate routes/objective mismatch on {axis}"
            )
        current = AcceptedGlobalBest(
            completed_at_seconds=_strict_float(
                event.get("timestamp_seconds"), "global-best completion time"
            ),
            iteration=_strict_int(event.get("iteration"), "global-best iteration"),
            objective_key=recorded_objective,
        )
        previous = last[axis]
        objective_not_better = previous is not None and (
            compare_objectives(
                SolutionObjective(*current.objective_key),
                SolutionObjective(*previous.objective_key),
            )
            is not ObjectiveComparison.BETTER
        )
        if previous is not None and (
            current.completed_at_seconds <= previous.completed_at_seconds
            or current.iteration < previous.iteration
            or objective_not_better
        ):
            raise ArtifactIntegrityError(f"global-best history is not strictly improving on {axis}")
        if current.completed_at_seconds > axis_budgets[axis]:
            raise ArtifactIntegrityError(f"global-best exceeds axis budget on {axis}")
        last[axis] = current
        for checkpoint in CHECKPOINT_SECONDS:
            if checkpoint <= axis_budgets[axis] and current.completed_at_seconds <= checkpoint:
                visible[axis][checkpoint] = current
    histories: dict[str, tuple[AcceptedGlobalBest, ...]] = {}
    for axis in axis_budgets:
        bounded: list[AcceptedGlobalBest] = []
        for checkpoint in sorted(visible[axis]):
            accepted_best = visible[axis][checkpoint]
            if not bounded or accepted_best != bounded[-1]:
                bounded.append(accepted_best)
        histories[axis] = tuple(bounded)
    return histories


def _checkpoint_evidence(
    *,
    reader: ArtifactReader,
    checkpoint_ref: Mapping[str, object],
    instance: str,
    seed: int,
    customer_count: int,
    raw_axes: Mapping[str, object],
    solution_axes: Mapping[str, object],
    histories: Mapping[str, tuple[AcceptedGlobalBest, ...]],
    verified_initial_objectives: Mapping[str, tuple[int, float, float, int]],
) -> list[dict[str, object]]:
    if checkpoint_ref.get("artifact_subtype") != "checkpoint_v1" or not checkpoint_ref.get(
        "schema_fingerprint"
    ):
        raise ArtifactIntegrityError(
            "anytime checkpoint artifact subtype/schema fingerprint is invalid"
        )
    path = str(checkpoint_ref.get("relative_path", ""))
    maximum_rows = 4 if customer_count < 100 else 16
    observed: list[AnytimeCheckpoint] = []
    for row in reader.iter_parquet_rows(
        path,
        schema=ANYTIME_CHECKPOINT_SCHEMA,
        batch_size=16,
    ):
        if len(observed) >= maximum_rows:
            raise ArtifactIntegrityError("anytime checkpoint row count exceeds its hard bound")
        observed.append(AnytimeCheckpoint.from_dict(row))
    expected: list[AnytimeCheckpoint] = []
    for axis in sorted(raw_axes, key=_axis_budget):
        raw_axis = raw_axes[axis]
        solution_axis = solution_axes.get(axis)
        if not isinstance(raw_axis, Mapping) or not isinstance(solution_axis, Mapping):
            raise ArtifactIntegrityError("checkpoint raw/solution axis is invalid")
        initial = verified_initial_objectives.get(axis)
        if initial is None:
            raise ArtifactIntegrityError(f"verified initial objective is missing on {axis}")
        raw_initial = _objective_key(raw_axis.get("initial_objective_key"), "initial_objective_key")
        if (
            compare_objectives(
                SolutionObjective(*raw_initial),
                SolutionObjective(*initial),
            )
            is not ObjectiveComparison.EQUAL
        ):
            raise ArtifactIntegrityError(f"raw initial objective failed replay on {axis}")
        final = _objective_key(solution_axis.get("objective_key"), "final objective")
        expected_final = histories[axis][-1].objective_key if histories[axis] else initial
        if (
            compare_objectives(
                SolutionObjective(*final),
                SolutionObjective(*expected_final),
            )
            is not ObjectiveComparison.EQUAL
        ):
            raise ArtifactIntegrityError(
                f"final incumbent does not match accepted history: {instance}/{seed}/{axis}"
            )
        runtime = _strict_float(raw_axis.get("runtime_seconds"), "runtime_seconds")
        termination = str(raw_axis.get("termination_reason", ""))
        completion_value = raw_axis.get("iteration_limit_completed_at_seconds")
        carry_time: float | None = None
        if termination == "iteration_limit":
            carry_time = _strict_float(
                completion_value,
                "iteration_limit_completed_at_seconds",
            )
            if customer_count >= 100 or carry_time > _axis_budget(axis) or carry_time > runtime:
                raise ArtifactIntegrityError(
                    f"iteration-limit completion timestamp is invalid on {axis}"
                )
        elif completion_value is not None:
            raise ArtifactIntegrityError(
                f"non-iteration termination carries an iteration completion on {axis}"
            )
        expected.extend(
            AnytimeCheckpoint.for_axis(
                instance=instance,
                seed=seed,
                axis_budget_seconds=_axis_budget(axis),
                initial_objective_key=initial,
                accepted_global_bests=histories[axis],
                max_iterations_completed_at_seconds=carry_time,
                final_objective_key=final if carry_time is not None else None,
            )
        )
    if tuple(item.to_dict() for item in observed) != tuple(item.to_dict() for item in expected):
        raise ArtifactIntegrityError(
            f"anytime checkpoints do not replay from accepted incumbents: {instance}/{seed}"
        )
    return [item.to_dict() for item in observed]


def _replay_shard(
    *,
    reader: ArtifactReader,
    shard_manifest: Mapping[str, object],
    benchmark_dir: Path,
    scope: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    instance_name = str(shard_manifest.get("instance", ""))
    seed = _strict_int(shard_manifest.get("seed"), "seed")
    customer_count = next(
        (record.customer_count for record in BEST_KNOWN_VALUES if record.instance == instance_name),
        None,
    )
    if customer_count is None:
        raise ArtifactIntegrityError(f"unknown campaign instance: {instance_name}")
    artifacts = _artifacts_for_directory(shard_manifest)
    for artifact in artifacts:
        _verified_primary_descriptor(reader, artifact)
    by_type: dict[tuple[str, str], Mapping[str, object]] = {}
    for item in artifacts:
        key = (str(item.get("artifact_type", "")), str(item.get("artifact_subtype", "")))
        if key in by_type:
            raise ArtifactIntegrityError(f"duplicate shard artifact identity: {key}")
        by_type[key] = item
    required_types = {
        "raw",
        "solution",
        "trace",
        "route_dictionary",
        "environment",
        "diagnostic",
        "anytime_checkpoints",
    }
    observed_types = {key[0] for key in by_type}
    if not required_types <= observed_types:
        raise ArtifactIntegrityError(
            f"shard raw/solution/trace/checkpoint evidence is incomplete: {instance_name}/{seed}"
        )
    event_subtypes = {key[1] for key in by_type if key[0] == "events"}
    if (
        not {
            "critical",
            "screening_checks",
            "screening_definitions_v3",
            "screening_occurrences_v3",
        }
        <= event_subtypes
        or "screening_decisions_v2" in event_subtypes
    ):
        raise ArtifactIntegrityError(
            f"shard does not use physical screening_decisions_v3: {instance_name}/{seed}"
        )
    raw_ref = next(item for item in artifacts if item.get("artifact_type") == "raw")
    solution_ref = next(item for item in artifacts if item.get("artifact_type") == "solution")
    trace_ref = next(item for item in artifacts if item.get("artifact_type") == "trace")
    if trace_ref.get("artifact_subtype") != "compact_index_v1":
        raise ArtifactIntegrityError("trace artifact is not the compact_index_v1 subtype")
    events_ref = by_type["events", "critical"]
    checkpoint_ref = next(
        item for item in artifacts if item.get("artifact_type") == "anytime_checkpoints"
    )
    raw = reader.read_json(str(raw_ref["relative_path"]))
    solution = reader.read_json(str(solution_ref["relative_path"]))
    trace = _read_bounded_compact_trace(reader, trace_ref)
    if (
        trace.get("run_label") != shard_manifest.get("run_label")
        or trace.get("instance") != instance_name
        or trace.get("seed") != seed
    ):
        raise ArtifactIntegrityError("compact trace index identity mismatch")
    expected_trace_references = {
        "route_dictionary_ref": next(
            item for item in artifacts if item.get("artifact_type") == "route_dictionary"
        ).get("relative_path"),
        "events_ref": events_ref.get("relative_path"),
        "screening_checks_ref": by_type["events", "screening_checks"].get("relative_path"),
        "screening_definitions_ref": by_type["events", "screening_definitions_v3"].get(
            "relative_path"
        ),
        "screening_occurrences_ref": by_type["events", "screening_occurrences_v3"].get(
            "relative_path"
        ),
        "diagnostic_ref": next(
            item for item in artifacts if item.get("artifact_type") == "diagnostic"
        ).get("relative_path"),
    }
    if any(trace.get(field) != value for field, value in expected_trace_references.items()):
        raise ArtifactIntegrityError("compact trace references do not match shard artifacts")
    schema_descriptors = {
        "route_dictionary": next(
            item for item in artifacts if item.get("artifact_type") == "route_dictionary"
        ),
        "events": events_ref,
        "screening_checks": by_type["events", "screening_checks"],
        "screening_definitions": by_type["events", "screening_definitions_v3"],
        "screening_occurrences": by_type["events", "screening_occurrences_v3"],
        "diagnostic": next(item for item in artifacts if item.get("artifact_type") == "diagnostic"),
    }
    if any(
        descriptor.get("schema_fingerprint") != COMPACT_TRACE_SCHEMA_FINGERPRINTS[schema_name]
        for schema_name, descriptor in schema_descriptors.items()
    ):
        raise ArtifactIntegrityError(
            "compact trace physical descriptors do not match the fixed schemas"
        )
    raw_axes = raw.get("axes")
    solution_axes = solution.get("axes")
    trace_axes = trace.get("axes")
    if not all(isinstance(value, Mapping) for value in (raw_axes, solution_axes, trace_axes)):
        raise ArtifactIntegrityError("raw/solution/trace axes are missing")
    assert isinstance(raw_axes, Mapping)
    assert isinstance(solution_axes, Mapping)
    assert isinstance(trace_axes, Mapping)
    expected_axes = (
        {"wall_clock_30"}
        if scope == "pilot" or customer_count < 100
        else {"wall_clock_30", "wall_clock_60", "wall_clock_300"}
    )
    if (
        set(raw_axes) != expected_axes
        or set(solution_axes) != expected_axes
        or set(trace_axes) != expected_axes
    ):
        raise ArtifactIntegrityError(f"campaign axis identity mismatch: {instance_name}/{seed}")
    instance = parse_schneider(benchmark_dir / f"{instance_name}.txt")
    replay_rows: list[dict[str, object]] = []
    verified_initial_objectives: dict[str, tuple[int, float, float, int]] = {}
    for axis in sorted(expected_axes, key=_axis_budget):
        raw_axis = raw_axes[axis]
        solution_axis = solution_axes[axis]
        trace_axis = trace_axes[axis]
        if not all(isinstance(value, Mapping) for value in (raw_axis, solution_axis, trace_axis)):
            raise ArtifactIntegrityError("campaign axis payload is invalid")
        assert isinstance(raw_axis, Mapping)
        assert isinstance(solution_axis, Mapping)
        assert isinstance(trace_axis, Mapping)
        routes = solution_axis.get("routes")
        initial_routes = solution_axis.get("initial_routes")
        if (
            not isinstance(routes, list)
            or any(not isinstance(route, list) for route in routes)
            or not isinstance(initial_routes, list)
            or not initial_routes
            or any(not isinstance(route, list) for route in initial_routes)
        ):
            raise ArtifactIntegrityError("solution routes are missing")
        report = validate_routes(instance, [[str(node) for node in route] for route in routes])
        objective = SolutionObjective.from_report(instance, report) if report.feasible else None
        initial_report = validate_routes(
            instance,
            [[str(node) for node in route] for route in initial_routes],
        )
        initial_objective = (
            SolutionObjective.from_report(instance, initial_report)
            if initial_report.feasible
            else None
        )
        recorded = _objective_key(solution_axis.get("objective_key"), "solution objective")
        raw_recorded = _objective_key(raw_axis.get("objective_key"), "raw objective")
        initial_recorded = _objective_key(
            solution_axis.get("initial_objective_key"),
            "solution initial objective",
        )
        raw_initial_recorded = _objective_key(
            raw_axis.get("initial_objective_key"),
            "raw initial objective",
        )
        native_passed, native_detail = _native_axis_valid(raw_axis, trace_axis)
        budget = _axis_budget(axis)
        if (
            objective is None
            or initial_objective is None
            or compare_objectives(objective, SolutionObjective(*recorded))
            is not ObjectiveComparison.EQUAL
            or compare_objectives(
                initial_objective,
                SolutionObjective(*initial_recorded),
            )
            is not ObjectiveComparison.EQUAL
            or compare_objectives(
                SolutionObjective(*raw_initial_recorded),
                SolutionObjective(*initial_recorded),
            )
            is not ObjectiveComparison.EQUAL
            or compare_objectives(
                SolutionObjective(*raw_recorded),
                SolutionObjective(*recorded),
            )
            is not ObjectiveComparison.EQUAL
            or raw_axis.get("valid") is not True
            or raw_axis.get("validator_passed") is not True
            or solution_axis.get("feasible") is not True
            or not native_passed
        ):
            raise ArtifactIntegrityError(
                f"validator/objective/native replay failed: {instance_name}/{seed}/{axis}; "
                f"{native_detail}"
            )
        verified_initial_objectives[axis] = initial_recorded
        replay_rows.append(
            {
                "instance": instance_name,
                "seed": seed,
                "axis": axis,
                "customer_count": customer_count,
                "family": "RC"
                if instance_name.casefold().startswith("rc")
                else ("R" if instance_name.casefold().startswith("r") else "C"),
                "budget_seconds": budget,
                "vehicle_count": recorded[0],
                "total_distance": recorded[1],
                "total_charging_time": recorded[2],
                "charging_count": recorded[3],
                "exact_started_calls": raw_axis.get("started_calls"),
                "exact_completed_calls": raw_axis.get("completed_calls"),
                "effective_iterations": raw_axis.get("effective_iterations"),
                "termination_reason": raw_axis.get("termination_reason"),
                "native_fallbacks": 0,
                "native_protocol_fallbacks": 0,
            }
        )
    axis_budgets = {axis: _axis_budget(axis) for axis in expected_axes}
    event_path = str(events_ref["relative_path"])
    _audit_async_persistence_ledgers(
        reader.iter_events(event_path, batch_size=1024),
        trace_axes,
    )
    event_audit = audit_streamed_events(
        reader.iter_events(event_path, batch_size=1024),
        axis_budgets,
    )
    if not event_audit.passed:
        raise ArtifactIntegrityError(
            f"event replay failed: {instance_name}/{seed}: {event_audit.detail}"
        )
    reconciliation = validate_axis_event_reconciliation(
        raw_axes,
        event_audit,
        customer_count=customer_count,
    )
    if not reconciliation.passed:
        raise ArtifactIntegrityError(
            "event/raw exact-call or deadline reconciliation failed: "
            f"{instance_name}/{seed}: {reconciliation.detail}"
        )
    histories = summarize_streamed_global_bests(
        reader.iter_events(event_path, batch_size=1024),
        axis_budgets,
        instance=instance,
    )
    checkpoints = _checkpoint_evidence(
        reader=reader,
        checkpoint_ref=checkpoint_ref,
        instance=instance_name,
        seed=seed,
        customer_count=customer_count,
        raw_axes=raw_axes,
        solution_axes=solution_axes,
        histories=histories,
        verified_initial_objectives=verified_initial_objectives,
    )
    return replay_rows, checkpoints, event_audit.event_count


def _batch_control_artifact(
    reader: ArtifactReader,
    artifact_type: str,
) -> tuple[Mapping[str, object], Path]:
    item = _one_artifact(reader, artifact_type)
    return item, reader.run_dir / str(item["relative_path"])


def _validate_power_load(
    payload: Mapping[str, object],
    *,
    run_label: str,
    batch_id: str,
) -> tuple[bool, str]:
    if (
        payload.get("schema_version") != "stage05.2-batch-power-load-v1"
        or payload.get("run_label") != run_label
        or payload.get("batch_id") != batch_id
        or payload.get("status") != "complete"
    ):
        return False, "batch power/load identity is invalid"
    preflight = payload.get("preflight")
    runtime = payload.get("runtime")
    if not isinstance(preflight, Mapping) or not isinstance(runtime, Mapping):
        return False, "batch power/load observations are missing"
    boundary_passed, boundary_detail = _validate_native_power_boundary(
        payload.get("native_power_boundary")
    )
    if not boundary_passed:
        return False, boundary_detail
    windows = preflight.get("windows")
    if (
        preflight.get("power_source") != "AC Power"
        or preflight.get("low_power_mode_enabled") is not False
        or not isinstance(windows, list)
        or len(windows) != 2
    ):
        return False, "batch AC/low-power/two-window preflight failed"
    previous_end: float | None = None
    for window in windows:
        if not isinstance(window, Mapping):
            return False, "batch preflight window is invalid"
        try:
            started = _strict_float(window.get("started_at_seconds"), "window start")
            duration = _strict_float(window.get("duration_seconds"), "window duration")
            load1 = _strict_float(window.get("maximum_load1"), "window load1")
            unrelated = _strict_float(
                window.get("maximum_unrelated_process_average_cores"),
                "window unrelated cores",
            )
            logical_cpu_count = _strict_int(
                window.get("logical_cpu_count"),
                "window logical CPU count",
            )
            raw_samples = window.get("process_cpu_samples")
            if not isinstance(raw_samples, list):
                raise ArtifactIntegrityError("window process CPU samples are missing")
            samples = tuple(
                ProcessCpuCounterSample.from_dict(sample)
                for sample in raw_samples
                if isinstance(sample, Mapping)
            )
            if len(samples) != len(raw_samples):
                raise ArtifactIntegrityError("window process CPU sample is invalid")
            replayed_unrelated = maximum_process_average_cores(
                samples,
                logical_cpu_count=logical_cpu_count,
            )
        except ArtifactIntegrityError as error:
            return False, str(error)
        except ValueError as error:
            return False, f"window process CPU replay failed: {error}"
        if (
            duration != 30.0
            or load1 > 4.0
            or unrelated >= 1.0
            or not math.isclose(unrelated, replayed_unrelated, rel_tol=0.0, abs_tol=1e-12)
            or (previous_end is not None and not math.isclose(started, previous_end))
        ):
            return False, "batch preflight load windows violate the fixed thresholds"
        previous_end = started + duration
    try:
        runtime_samples = _strict_int(runtime.get("sample_count"), "runtime sample_count")
        power_violations = _strict_int(
            runtime.get("power_source_violations"), "power source violations"
        )
        low_power_violations = _strict_int(
            runtime.get("low_power_mode_violations"), "low power violations"
        )
        maximum_load1 = _strict_float(runtime.get("maximum_load1"), "runtime load1")
        maximum_unrelated = _strict_float(
            runtime.get("maximum_unrelated_process_average_cores"),
            "runtime unrelated cores",
        )
        runtime_logical_cpu_count = _strict_int(
            runtime.get("logical_cpu_count"),
            "runtime logical CPU count",
        )
        raw_runtime_samples = runtime.get("process_cpu_samples")
        if not isinstance(raw_runtime_samples, list):
            raise ArtifactIntegrityError("runtime process CPU samples are missing")
        runtime_counter_samples = tuple(
            ProcessCpuCounterSample.from_dict(sample)
            for sample in raw_runtime_samples
            if isinstance(sample, Mapping)
        )
        if len(runtime_counter_samples) != len(raw_runtime_samples):
            raise ArtifactIntegrityError("runtime process CPU sample is invalid")
        replayed_runtime_unrelated = maximum_process_average_cores(
            runtime_counter_samples,
            logical_cpu_count=runtime_logical_cpu_count,
        )
    except ArtifactIntegrityError as error:
        return False, str(error)
    except ValueError as error:
        return False, f"runtime process CPU replay failed: {error}"
    passed = (
        runtime_samples > 0
        and power_violations == 0
        and low_power_violations == 0
        and maximum_load1 <= 4.0
        and maximum_unrelated < 1.0
        and math.isclose(
            maximum_unrelated,
            replayed_runtime_unrelated,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    return (
        passed,
        "continuous batch power/load sampling passed"
        if passed
        else "continuous batch power/load sampling violated a threshold",
    )


def _validate_native_power_boundary(payload: object) -> tuple[bool, str]:
    if not isinstance(payload, Mapping):
        return False, "native power boundary evidence is missing"
    before = payload.get("before")
    after = payload.get("after")
    if (
        payload.get("schema_version") != "stage05.2-native-power-boundary-v1"
        or payload.get("stable_invariants") != ["ac_online", "battery_saver", "active_power_scheme"]
        or payload.get("invariants_unchanged") is not True
        or not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
    ):
        return False, "native power boundary schema is invalid"
    for observation in (before, after):
        percent = observation.get("battery_life_percent")
        flag = observation.get("battery_flag")
        if (
            observation.get("ac_online") is not True
            or observation.get("battery_saver") is not False
            or not isinstance(observation.get("active_power_scheme"), str)
            or not str(observation.get("active_power_scheme")).strip()
            or isinstance(percent, bool)
            or not isinstance(percent, int)
            or percent not in range(0, 256)
            or isinstance(flag, bool)
            or not isinstance(flag, int)
        ):
            return False, "native power boundary observation is invalid"
    invariants = ("ac_online", "battery_saver", "active_power_scheme")
    if any(before.get(field) != after.get(field) for field in invariants):
        return False, "native power hard invariant changed across the batch"
    return True, "native power boundary invariants independently replayed"


def _validate_batch_resources(
    *,
    resource: dict[str, object],
    shards: list[dict[str, object]],
    workers: int,
    run_label: str,
) -> tuple[bool, str, dict[str, object]]:
    ownership, detail, owners = validate_worker_ownership(
        resource,
        shards,
        expected_workers=workers,
        expected_run_label=run_label,
        expected_component=Stage052Component.BENCHMARK.value,
    )
    if not ownership:
        return False, detail, {}
    if resource.get("schema_version") != STAGE052_RESOURCE_SCHEMA_VERSION:
        return False, "batch resource evidence must use resource v3", {}
    process_peaks = resource.get("process_peak_rss_bytes")
    if not isinstance(process_peaks, Mapping):
        return False, "batch process peak RSS mapping is missing", {}
    try:
        normalized_peaks = {
            int(str(pid)): _strict_int(value, "process peak RSS")
            for pid, value in process_peaks.items()
        }
        aggregate = _strict_int(resource.get("aggregate_peak_rss_bytes"), "aggregate peak RSS")
    except (ValueError, ArtifactIntegrityError) as error:
        return False, str(error), {}
    worker_peak = max((normalized_peaks.get(pid, 0) for pid in owners), default=0)
    passed = (
        worker_peak > 0
        and worker_peak <= PER_WORKER_RSS_LIMIT_BYTES
        and aggregate <= PROCESS_TREE_RSS_LIMIT_BYTES
    )
    summary = {
        "configured_workers": workers,
        "owner_pids": list(owners),
        "per_worker_peak_rss_bytes": worker_peak,
        "aggregate_peak_rss_bytes": aggregate,
        "load1_min": resource.get("load1_min"),
        "load1_mean": resource.get("load1_mean"),
        "load1_max": resource.get("load1_max"),
        "sample_count": resource.get("sample_count"),
    }
    return (
        passed,
        "per-worker and process-tree RSS limits passed"
        if passed
        else "per-worker 4357382144-byte or process-tree 12-GiB RSS limit failed",
        summary,
    )


def _validate_batch_metadata(
    metadata: Mapping[str, object],
    *,
    campaign: CampaignManifest,
    selection_lock: Mapping[str, object],
    source_snapshot: Mapping[str, object],
) -> tuple[bool, str, tuple[str, str, str]]:
    runtime = metadata.get("runtime_identity")
    provenance = metadata.get("performance_provenance")
    native = metadata.get("native_kernel_config")
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(provenance, Mapping)
        or not isinstance(native, Mapping)
    ):
        return False, "batch runtime/input/native identity is missing", ("", "", "")
    repository_revision = str(metadata.get("repository_revision", ""))
    campaign_configuration = str(metadata.get("campaign_configuration_sha256", ""))
    runtime_digest = _canonical_sha256(runtime)
    try:
        current_instance_hashes = _validated_instance_hashes(provenance)
    except ArtifactIntegrityError:
        current_instance_hashes = {}
    locked_instance_hashes = selection_lock.get("instance_sha256")
    locked_instances_match = isinstance(locked_instance_hashes, Mapping) and all(
        current_instance_hashes.get(str(instance)) == digest
        for instance, digest in locked_instance_hashes.items()
    )
    passed = (
        metadata.get("run_label") == campaign.run_label
        and metadata.get("component") == Stage052Component.BENCHMARK.value
        and metadata.get("scope") == campaign.scope
        and metadata.get("execution_backend") == campaign.selected_backend
        and campaign.selected_backend in {"native_cpu", "cuda"}
        and metadata.get("backend") == campaign.selected_exact_backend == "cpu_batch"
        and metadata.get("worker_count") == campaign.selected_workers
        and metadata.get("native_profile") == campaign.native_profile
        and metadata.get("storage_policy_version") == campaign.storage_policy_version
        and metadata.get("screening_schema_version") == campaign.screening_schema_version
        and campaign_configuration == campaign.configuration_sha256
        and metadata.get("campaign_prerequisite_review_sha256")
        == campaign.prerequisite_review_sha256
        and metadata.get("configuration_sha256") == selection_lock.get("configuration_sha256")
        and repository_revision == selection_lock.get("repository_revision")
        and runtime_digest == selection_lock.get("runtime_identity_sha256")
        and _canonical_sha256(_stable_input_provenance(provenance))
        == selection_lock.get("input_provenance_sha256")
        and locked_instances_match
        and _canonical_sha256(native) == selection_lock.get("native_config_sha256")
        and dict(native) == selection_lock.get("native_kernel_config")
        and campaign.selected_workers == selection_lock.get("selected_workers")
        and campaign.selected_backend == selection_lock.get("selected_backend")
        and campaign.selected_exact_backend == selection_lock.get("selected_exact_backend")
        and campaign.native_profile == selection_lock.get("native_profile")
        and len(repository_revision) == 40
        and all(character in "0123456789abcdef" for character in repository_revision)
        and metadata.get("repository_dirty") is False
        and metadata.get("source_snapshot") == source_snapshot
        and runtime.get("schema_version") == "stage05.2-runtime-identity-v2"
        and runtime.get("installed_editable") is False
        and runtime.get("repository_revision") == repository_revision
        and all(
            isinstance(runtime.get(field), str)
            and len(str(runtime.get(field))) == 64
            and all(character in "0123456789abcdef" for character in str(runtime.get(field)))
            for field in (
                "wheel_sha256",
                "python_executable_sha256",
                "native_extension_sha256",
                "dependency_manifest_sha256",
                "installed_distribution_sha256",
            )
        )
    )
    return (
        passed,
        "batch input/runtime/backend provenance passed"
        if passed
        else "batch input/runtime/backend provenance mismatch",
        (repository_revision, campaign_configuration, runtime_digest),
    )


def _persistence_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    batch_id: str,
) -> tuple[bool, str, dict[str, object]]:
    solver = 0.0
    persistence = 0.0
    try:
        for row in rows:
            solver += _strict_float(row.get("solver_seconds"), "solver_seconds")
            persistence += _strict_float(
                row.get("artifact_persistence_seconds"),
                "artifact_persistence_seconds",
            )
    except ArtifactIntegrityError as error:
        return False, str(error), {}
    ratio = persistence / (solver + persistence) if solver + persistence > 0.0 else math.inf
    summary = {
        "batch_id": batch_id,
        "solver_seconds": solver,
        "artifact_persistence_seconds": persistence,
        "persistence_ratio": ratio,
        "maximum_ratio": STAGE052_MAXIMUM_PERSISTENCE_RATIO,
    }
    return (
        ratio <= STAGE052_MAXIMUM_PERSISTENCE_RATIO,
        f"batch persistence ratio={ratio:.9f}",
        summary,
    )


def _audit_batch_persistence(
    *,
    batch_dir: Path,
    reader: ArtifactReader,
    batch: BatchManifest,
    campaign: CampaignManifest,
    staging_root_alias: str,
    rows: Sequence[Mapping[str, object]],
    run_label: str,
    scope: str,
    batch_id: str,
) -> tuple[bool, str, dict[str, object]]:
    """Independently replay one batch's shard and primary-control timing."""

    try:
        timing_ref = _one_artifact(reader, "timing_evidence")
        timing_payload = reader.read_json(str(timing_ref["relative_path"]))
        timing_rows = timing_payload.get("rows")
        if (
            timing_payload.get("schema_version") != "stage05.2-timing-evidence-v1"
            or timing_payload.get("run_label") != run_label
            or timing_payload.get("component") != Stage052Component.BENCHMARK.value
            or not isinstance(timing_rows, list)
        ):
            raise ArtifactIntegrityError("batch persistence timing identity is invalid")
        row_by_identity = {
            (
                str(row.get("instance", "")),
                _strict_int(row.get("seed"), "seed"),
                str(row.get("axis", "")),
            ): row
            for row in rows
        }
        if len(row_by_identity) != len(rows):
            raise ArtifactIntegrityError("batch persistence per-run identity is duplicate")
        pipeline_by_identity: dict[tuple[str, int, str], Mapping[str, object]] = {}
        for trace_ref in (
            item
            for item in reader.manifest.get("artifacts", ())
            if isinstance(item, Mapping) and item.get("artifact_type") == "trace"
        ):
            trace_payload = _read_bounded_compact_trace(reader, trace_ref)
            trace_instance = str(trace_payload.get("instance", ""))
            trace_seed = _strict_int(trace_payload.get("seed"), "trace seed")
            trace_axes = trace_payload.get("axes")
            if not isinstance(trace_axes, Mapping):
                raise ArtifactIntegrityError("batch trace axes are missing")
            for axis, trace_axis in trace_axes.items():
                if not isinstance(trace_axis, Mapping):
                    raise ArtifactIntegrityError("batch trace axis is invalid")
                pipeline = trace_axis.get("persistence_pipeline")
                if not isinstance(pipeline, Mapping):
                    raise ArtifactIntegrityError("batch persistence pipeline is missing")
                identity = (trace_instance, trace_seed, str(axis))
                if identity in pipeline_by_identity:
                    raise ArtifactIntegrityError("batch persistence pipeline identity is duplicate")
                pipeline_by_identity[identity] = pipeline
        if set(pipeline_by_identity) != set(row_by_identity):
            raise ArtifactIntegrityError("batch persistence pipeline scope mismatch")
        observed: set[tuple[str, int, str]] = set()
        solver_seconds = 0.0
        shard_persistence_seconds = 0.0
        for raw_timing in timing_rows:
            if not isinstance(raw_timing, Mapping):
                raise ArtifactIntegrityError("batch persistence timing row is invalid")
            identity = (
                str(raw_timing.get("instance", "")),
                _strict_int(raw_timing.get("seed"), "seed"),
                str(raw_timing.get("axis", "")),
            )
            if identity in observed or identity not in row_by_identity:
                raise ArtifactIntegrityError("batch persistence timing scope mismatch")
            observed.add(identity)
            solver_started_ns = _strict_int(
                raw_timing.get("solver_started_ns"), "solver_started_ns"
            )
            solver_completed_ns = _strict_int(
                raw_timing.get("solver_completed_ns"), "solver_completed_ns"
            )
            finalize_started_ns = _strict_int(
                raw_timing.get("finalize_started_ns"), "finalize_started_ns"
            )
            finalize_completed_ns = _strict_int(
                raw_timing.get("finalize_completed_ns"), "finalize_completed_ns"
            )
            live_ns = _strict_int(
                raw_timing.get("live_stream_persistence_ns"),
                "live_stream_persistence_ns",
            )
            solver_interleaved_ns = _strict_int(
                raw_timing.get("solver_interleaved_persistence_ns"),
                "solver_interleaved_persistence_ns",
            )
            pipeline = pipeline_by_identity[identity]
            solver_union_ns = _strict_int(
                pipeline.get("solver_persistence_union_nanoseconds"),
                "solver_persistence_union_nanoseconds",
            )
            persistence_union_ns = _strict_int(
                pipeline.get("persistence_union_nanoseconds"),
                "persistence_union_nanoseconds",
            )
            event_count = _strict_int(raw_timing.get("axis_event_count"), "axis_event_count")
            total_events = _strict_int(raw_timing.get("total_event_count"), "total_event_count")
            axis_count = _strict_int(raw_timing.get("axis_count"), "axis_count")
            if (
                solver_completed_ns <= solver_started_ns
                or finalize_completed_ns < finalize_started_ns
                or live_ns < 0
                or live_ns > solver_completed_ns - solver_started_ns
                or solver_interleaved_ns != solver_union_ns
                or solver_interleaved_ns > live_ns
                or persistence_union_ns > live_ns
                or event_count < 0
                or total_events < 0
                or axis_count <= 0
            ):
                raise ArtifactIntegrityError("batch persistence monotonic interval is invalid")
            audited_solver = (
                solver_completed_ns - solver_started_ns - solver_interleaved_ns
            ) / 1_000_000_000
            finalize_seconds = (finalize_completed_ns - finalize_started_ns) / 1_000_000_000
            finalize_share = (
                finalize_seconds * event_count / total_events
                if total_events
                else finalize_seconds / axis_count
            )
            audited_persistence = finalize_share + live_ns / 1_000_000_000
            row = row_by_identity[identity]
            if not math.isclose(
                _strict_float(row.get("solver_seconds"), "solver_seconds"),
                audited_solver,
                rel_tol=0.0,
                abs_tol=1e-9,
            ) or not math.isclose(
                _strict_float(
                    row.get("artifact_persistence_seconds"),
                    "artifact_persistence_seconds",
                ),
                audited_persistence,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ArtifactIntegrityError(
                    "batch per-run persistence differs from timing evidence"
                )
            solver_seconds += audited_solver
            shard_persistence_seconds += audited_persistence
        if observed != set(row_by_identity):
            raise ArtifactIntegrityError("batch persistence timing evidence is incomplete")

        attribution_path = (
            batch_dir / "control" / f"{run_label}_{batch_id}_persistence_attribution.json"
        )
        sidecar = attribution_path.with_suffix(".sha256")
        if (
            not attribution_path.is_file()
            or not sidecar.is_file()
            or not signed_sidecar_matches(attribution_path, sidecar)
        ):
            raise ArtifactIntegrityError("batch signed persistence attribution is invalid")
        attribution = Stage052PersistenceAttribution.from_dict(_json_object(attribution_path))
        required_labels = {
            "batch_write_control",
            "batch_preflight_control",
            "batch_timing_and_per_run_control",
            "batch_resource_control",
            "batch_runtime_control",
            "batch_power_load_control",
            "batch_primary_manifest_finalize",
        }
        if {interval.label for interval in attribution.control_intervals} != required_labels:
            raise ArtifactIntegrityError(
                "batch persistence attribution is missing a required active-write interval"
            )
        primary_manifest = batch_dir / attribution.primary_manifest_relative_path
        if (
            attribution.run_label != run_label
            or attribution.component != Stage052Component.BENCHMARK.value
            or attribution.scope != scope
            or attribution.subject_id != batch_id
            or primary_manifest.resolve() != reader.result.manifest_path.resolve()
            or _sha256(primary_manifest) != attribution.primary_manifest_sha256
            or not math.isclose(
                attribution.solver_seconds,
                solver_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                attribution.shard_persistence_seconds,
                shard_persistence_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ArtifactIntegrityError("batch persistence attribution does not replay")
        batch_attribution_sha = getattr(batch, "persistence_attribution_sha256", None)
        batch_control_seconds = getattr(batch, "control_persistence_seconds", None)
        batch_ratio = getattr(batch, "persistence_ratio", None)
        if (
            batch_attribution_sha != _sha256(attribution_path)
            or isinstance(batch_control_seconds, bool)
            or not isinstance(batch_control_seconds, int | float)
            or isinstance(batch_ratio, bool)
            or not isinstance(batch_ratio, int | float)
            or not math.isclose(
                float(batch_control_seconds),
                attribution.control_persistence_seconds,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(batch_ratio),
                attribution.persistence_ratio,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ArtifactIntegrityError("batch manifest persistence binding is invalid")
        envelope_path = batch_dir / "batch_persistence_envelope.json"
        envelope_sidecar = envelope_path.with_suffix(".sha256")
        if (
            not envelope_path.is_file()
            or not envelope_sidecar.is_file()
            or not signed_sidecar_matches(envelope_path, envelope_sidecar)
        ):
            raise ArtifactIntegrityError("batch final persistence envelope is invalid")
        envelope = BatchPersistenceEnvelope.from_dict(_json_object(envelope_path))
        archived_manifest_path = batch_dir / "batch_manifest.json"
        try:
            verified_batch = replace(
                batch,
                status="verified",
                root_alias=staging_root_alias,
                volume=campaign.storage_roots[staging_root_alias],
                transfer_mode=None,
                archive_transfer_seconds=None,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactIntegrityError(
                "verified batch manifest state cannot be reconstructed"
            ) from error
        verified_manifest_sha = hashlib.sha256(
            (json.dumps(verified_batch.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
        ).hexdigest()
        campaign_envelope_sha = campaign.batch_persistence_envelope_sha256_by_id.get(batch_id)
        if (
            envelope.run_label != run_label
            or envelope.batch_id != batch_id
            or envelope.base_attribution_sha256 != _sha256(attribution_path)
            or envelope.verified_manifest_sha256 != verified_manifest_sha
            or envelope.archived_manifest_sha256 != _sha256(archived_manifest_path)
            or campaign_envelope_sha != _sha256(envelope_path)
            or not math.isclose(
                envelope.solver_seconds,
                attribution.solver_seconds,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                envelope.base_persistence_seconds,
                attribution.total_persistence_seconds,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ArtifactIntegrityError("batch final persistence envelope does not replay")
        passed = envelope.persistence_ratio <= STAGE052_MAXIMUM_PERSISTENCE_RATIO
        return (
            passed,
            f"batch persistence ratio={envelope.persistence_ratio:.9f}",
            {
                "batch_id": batch_id,
                "solver_seconds": envelope.solver_seconds,
                "artifact_persistence_seconds": envelope.total_persistence_seconds,
                "control_persistence_seconds": (
                    attribution.control_persistence_seconds + envelope.state_persistence_seconds
                ),
                "persistence_ratio": envelope.persistence_ratio,
                "maximum_ratio": STAGE052_MAXIMUM_PERSISTENCE_RATIO,
            },
        )
    except (ArtifactIntegrityError, OSError, TypeError, ValueError) as error:
        return False, str(error), {}


def _audit_campaign_persistence(
    *,
    campaign_dir: Path,
    reader: ArtifactReader,
    campaign: CampaignManifest,
    batch_rows: Sequence[Mapping[str, object]],
    staging_root_alias: str,
    archive_root_aliases: tuple[str, ...],
) -> tuple[bool, str, dict[str, object], str, str]:
    """Replay batch totals plus every non-archive campaign active write."""

    try:
        path = campaign_dir / "control" / f"{campaign.run_label}_persistence_attribution.json"
        sidecar = path.with_suffix(".sha256")
        if not path.is_file() or not sidecar.is_file() or not signed_sidecar_matches(path, sidecar):
            raise ArtifactIntegrityError("campaign persistence attribution is unsigned")
        attribution = Stage052PersistenceAttribution.from_dict(_json_object(path))
        expected_labels = [
            "campaign_parent_write_control",
            "campaign_plan_write",
            "campaign_preflight_write",
            "campaign_manifest_initial_write",
        ]
        if campaign.scope == "pilot":
            expected_labels.extend(
                (
                    "failure_state_drill_partial_write",
                    "failure_state_drill_failed_batch_write",
                    "failure_state_drill_failed_campaign_write",
                    "failure_state_drill_atomic_generation_write",
                    "failure_state_drill_atomic_replace_failure",
                    "failure_state_drill_worker_failure_recovery",
                    "failure_state_drill_archive_payload_write",
                    "failure_state_drill_archive_recovery",
                    "failure_state_drill_archive_recovery_state_write",
                    "failure_state_drill_summary_write",
                )
            )
        for batch in campaign.batches:
            expected_labels.extend(
                (
                    f"{batch.batch_id}_pre_dispatch_rolling_capacity_journal_write",
                    f"{batch.batch_id}_pre_dispatch_rolling_capacity_write",
                    f"{batch.batch_id}_campaign_verified_state_write",
                    f"{batch.batch_id}_pre_archive_rolling_capacity_journal_write",
                    f"{batch.batch_id}_pre_archive_rolling_capacity_write",
                    f"{batch.batch_id}_campaign_archived_state_write",
                    f"{batch.batch_id}_persistence_envelope_write",
                    f"{batch.batch_id}_campaign_envelope_state_write",
                    f"{batch.batch_id}_post_archive_rolling_capacity_journal_write",
                    f"{batch.batch_id}_post_archive_rolling_capacity_write",
                )
            )
        expected_labels.append("campaign_rolling_capacity_complete_write")
        if campaign.scope == "pilot":
            expected_labels.append("raw_replay_drill_summary_write")
            for alias in archive_root_aliases:
                expected_labels.extend(
                    (
                        f"archive_dry_run_{alias}_payload_write",
                        f"archive_dry_run_{alias}_verified_manifest_write",
                        f"archive_dry_run_{alias}_archived_manifest_write",
                    )
                )
            expected_labels.extend(
                (
                    "archive_dry_run_summary_write",
                    "publication_dry_run_payload_write",
                    "publication_dry_run_trusted_manifest_write",
                    "publication_dry_run_summary_write",
                )
            )
        expected_labels.extend(
            (
                "campaign_aggregate_per_run_write",
                "campaign_anytime_checkpoints_write",
                "campaign_primary_manifest_finalize",
            )
        )
        if tuple(interval.label for interval in attribution.control_intervals) != tuple(
            expected_labels
        ):
            raise ArtifactIntegrityError("campaign persistence interval set/order is incomplete")
        solver_seconds = sum(
            _strict_float(row.get("solver_seconds"), "solver_seconds") for row in batch_rows
        )
        batch_persistence_seconds = sum(
            _strict_float(
                row.get("artifact_persistence_seconds"),
                "artifact_persistence_seconds",
            )
            for row in batch_rows
        )
        primary = campaign_dir / attribution.primary_manifest_relative_path
        if (
            attribution.run_label != campaign.run_label
            or attribution.component != Stage052Component.BENCHMARK.value
            or attribution.scope != campaign.scope
            or attribution.subject_id != "campaign"
            or staging_root_alias not in campaign.storage_roots
            or primary.resolve() != reader.result.manifest_path.resolve()
            or _sha256(primary) != attribution.primary_manifest_sha256
            or not math.isclose(
                attribution.solver_seconds,
                solver_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                attribution.shard_persistence_seconds,
                batch_persistence_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ArtifactIntegrityError("campaign persistence attribution does not replay")
        row = {
            "batch_id": "campaign_aggregate",
            "solver_seconds": attribution.solver_seconds,
            "artifact_persistence_seconds": attribution.total_persistence_seconds,
            "control_persistence_seconds": attribution.control_persistence_seconds,
            "persistence_ratio": attribution.persistence_ratio,
            "maximum_ratio": STAGE052_MAXIMUM_PERSISTENCE_RATIO,
        }
        return (
            attribution.persistence_ratio <= STAGE052_MAXIMUM_PERSISTENCE_RATIO,
            f"campaign aggregate persistence ratio={attribution.persistence_ratio:.9f}",
            row,
            _sha256(path),
            _sha256(sidecar),
        )
    except (ArtifactIntegrityError, OSError, TypeError, ValueError) as error:
        return False, str(error), {}, "", ""


def _audit_campaign_controls(
    *,
    campaign_dir: Path,
    reader: ArtifactReader,
    campaign: CampaignManifest,
    locator: StorageRootLocator,
    staging_root_alias: str,
    archive_root_aliases: tuple[str, ...],
) -> tuple[bool, str]:
    """Independently replay rolling-capacity and all G01 drill artifacts."""

    try:
        rolling_ref = _one_artifact(reader, "rolling_capacity")
        rolling = reader.read_json(str(rolling_ref["relative_path"]))
        observations = rolling.get("observations")
        expected_observations = tuple(
            (batch.batch_id, phase)
            for batch in campaign.batches
            for phase in ("pre_dispatch", "pre_archive", "post_archive")
        )
        if (
            rolling.get("schema_version") != "stage05.2-rolling-capacity-summary-v1"
            or rolling.get("run_label") != campaign.run_label
            or rolling.get("scope") != campaign.scope
            or rolling.get("status") != "complete"
            or rolling.get("passed") is not True
            or not isinstance(observations, list)
            or len(observations) != len(expected_observations)
        ):
            raise ArtifactIntegrityError("rolling capacity summary is incomplete")
        observed_identities: list[tuple[str, str]] = []
        journal_refs = [
            item
            for item in reader.manifest.get("artifacts", ())
            if isinstance(item, Mapping)
            and item.get("artifact_type") == "rolling_capacity_observation"
        ]
        if len(journal_refs) != len(expected_observations):
            raise ArtifactIntegrityError("rolling capacity journal coverage is incomplete")
        for observation_index, observation in enumerate(observations):
            if not isinstance(observation, Mapping):
                raise ArtifactIntegrityError("rolling capacity observation is invalid")
            free = observation.get("free_bytes_by_device")
            required = observation.get("required_bytes_by_device")
            batch_index = observation_index // 3
            phase = ("pre_dispatch", "pre_archive", "post_archive")[observation_index % 3]
            expected_required = _expected_rolling_capacity_required(
                campaign=campaign,
                batch_index=batch_index,
                phase=phase,
                staging_root_alias=staging_root_alias,
                archive_root_aliases=archive_root_aliases,
            )
            expected_devices = {volume.device_uuid for volume in campaign.storage_roots.values()}
            normalized_free = (
                {
                    str(device): _strict_int(value, "rolling free bytes")
                    for device, value in free.items()
                }
                if isinstance(free, Mapping)
                else {}
            )
            normalized_required = (
                {
                    str(device): _strict_int(value, "rolling required bytes")
                    for device, value in required.items()
                }
                if isinstance(required, Mapping)
                else {}
            )
            if (
                observation.get("schema_version") != "stage05.2-rolling-capacity-v1"
                or observation.get("run_label") != campaign.run_label
                or observation.get("passed") is not True
                or not isinstance(free, Mapping)
                or not isinstance(required, Mapping)
                or set(normalized_free) != expected_devices
                or normalized_required != expected_required
                or any(value < 0 for value in normalized_free.values())
                or any(
                    normalized_free[device] < value for device, value in normalized_required.items()
                )
            ):
                raise ArtifactIntegrityError("rolling capacity reserve replay failed")
            journal = reader.read_json(str(journal_refs[observation_index]["relative_path"]))
            if journal != dict(observation):
                raise ArtifactIntegrityError(
                    "rolling capacity summary differs from its signed append-only journal"
                )
            observed_identities.append(
                (str(observation.get("batch_id", "")), str(observation.get("phase", "")))
            )
        if tuple(observed_identities) != expected_observations:
            raise ArtifactIntegrityError("rolling capacity phase order is incomplete")

        if campaign.scope != "pilot":
            return True, "rolling capacity evidence replay passed"

        failure_ref = _one_artifact(reader, "failure_state_drill")
        failure = reader.read_json(str(failure_ref["relative_path"]))
        partial_ref = _one_artifact(reader, "failure_state_drill_partial")
        partial_relative = str(partial_ref["relative_path"])
        partial_path = campaign_dir / partial_relative
        if (
            failure.get("schema_version") != "stage05.2-failure-state-drill-v1"
            or failure.get("run_label") != campaign.run_label
            or failure.get("status") != "passed"
            or failure.get("failure_type") != "injected_partial_write"
            or failure.get("partial_file_relative_path") != partial_relative
            or failure.get("partial_file_sha256") != _sha256(partial_path)
            or _strict_int(failure.get("injected_after_bytes"), "injected bytes")
            != partial_path.stat().st_size
            or partial_path.stat().st_size
            >= _strict_int(failure.get("expected_total_bytes"), "expected bytes")
        ):
            raise ArtifactIntegrityError("failure-state partial write drill is invalid")
        failed_batch_ref = _one_artifact(reader, "failure_state_drill_failed_batch")
        failed_campaign_ref = _one_artifact(reader, "failure_state_drill_failed_campaign")
        failed_batch_relative = str(failed_batch_ref["relative_path"])
        failed_campaign_relative = str(failed_campaign_ref["relative_path"])
        failed_batch_path = campaign_dir / failed_batch_relative
        failed_campaign_path = campaign_dir / failed_campaign_relative
        atomic_state_ref = _one_artifact(reader, "failure_state_drill_atomic_state")
        atomic_state_relative = str(atomic_state_ref["relative_path"])
        atomic_state_path = campaign_dir / atomic_state_relative
        atomic_state = reader.read_json(atomic_state_relative)
        if (
            failure.get("failed_batch_relative_path") != failed_batch_relative
            or failure.get("failed_batch_sha256") != _sha256(failed_batch_path)
            or failure.get("failed_campaign_relative_path") != failed_campaign_relative
            or failure.get("failed_campaign_sha256") != _sha256(failed_campaign_path)
            or failure.get("atomic_state_relative_path") != atomic_state_relative
            or failure.get("atomic_state_sha256") != _sha256(atomic_state_path)
            or failure.get("atomic_state_generation") != 1
            or atomic_state != {"run_label": campaign.run_label, "generation": 1}
        ):
            raise ArtifactIntegrityError("failure-state signed manifest bindings are invalid")
        worker_failure_ref = _one_artifact(reader, "failure_state_drill_worker_failure")
        worker_manifest_ref = _one_artifact(reader, "failure_state_drill_worker_manifest")
        archive_payload_ref = _one_artifact(reader, "failure_state_drill_archive_payload")
        archive_manifest_ref = _one_artifact(reader, "failure_state_drill_archive_manifest")
        worker_failure_relative = str(worker_failure_ref["relative_path"])
        worker_manifest_relative = str(worker_manifest_ref["relative_path"])
        archive_payload_relative = str(archive_payload_ref["relative_path"])
        archive_manifest_relative = str(archive_manifest_ref["relative_path"])
        worker_failure_path = campaign_dir / worker_failure_relative
        worker_manifest_path = campaign_dir / worker_manifest_relative
        archive_payload_path = campaign_dir / archive_payload_relative
        archive_manifest_path = campaign_dir / archive_manifest_relative
        worker_failure = reader.read_json(worker_failure_relative)
        worker_manifest = reader.read_json(worker_manifest_relative)
        worker_artifacts = worker_manifest.get("artifacts")
        if (
            failure.get("worker_failure_relative_path") != worker_failure_relative
            or failure.get("worker_failure_sha256") != _sha256(worker_failure_path)
            or failure.get("worker_manifest_relative_path") != worker_manifest_relative
            or failure.get("worker_manifest_sha256") != _sha256(worker_manifest_path)
            or failure.get("worker_injected_failure") != "injected worker failure drill"
            or worker_failure.get("status") != "partial"
            or worker_failure.get("evidence_completeness") != "partial"
            or worker_failure.get("failure_reason") != "injected worker failure drill"
            or worker_manifest.get("evidence_completeness") != "partial"
            or worker_manifest.get("worker_identity") != "failure-recorder"
            or worker_manifest.get("instance") != "c101C5"
            or worker_manifest.get("seed") != 2014
            or worker_manifest.get("shard_ordinal") != 0
            or not isinstance(worker_artifacts, list)
            or len(worker_artifacts) != 1
            or not isinstance(worker_artifacts[0], Mapping)
            or worker_artifacts[0].get("artifact_type") != "failure"
            or worker_artifacts[0].get("checksum") != _sha256(worker_failure_path)
            or worker_artifacts[0].get("byte_size") != worker_failure_path.stat().st_size
        ):
            raise ArtifactIntegrityError("worker failure recovery drill is invalid")
        archived_drill_batch = load_batch_manifest(archive_manifest_path)
        if (
            failure.get("archive_payload_relative_path") != archive_payload_relative
            or failure.get("archive_payload_sha256") != _sha256(archive_payload_path)
            or failure.get("archive_manifest_relative_path") != archive_manifest_relative
            or failure.get("archive_manifest_sha256") != _sha256(archive_manifest_path)
            or failure.get("archive_transfer_mode") != "same_volume_atomic_rename"
            or failure.get("archive_injected_failure") != "injected archive state write failure"
            or archived_drill_batch.status != "archived"
            or archived_drill_batch.transfer_mode != "same_volume_atomic_rename"
            or directory_checksum(archive_manifest_path.parent)
            != archived_drill_batch.checksum_sha256
            or directory_byte_count(archive_manifest_path.parent)
            != archived_drill_batch.actual_bytes
        ):
            raise ArtifactIntegrityError("archive state-write recovery drill is invalid")
        failed_batch = load_batch_manifest(failed_batch_path)
        failed_campaign = load_campaign_manifest(failed_campaign_path)
        injected_reason = "injected campaign failure-handling drill"
        staging_volume = campaign.storage_roots[staging_root_alias]
        planned_batches = tuple(
            replace(
                batch,
                status="planned",
                root_alias=staging_root_alias,
                volume=staging_volume,
                checksum_sha256=None,
                actual_bytes=None,
                row_count=None,
                physical_schema=None,
                resource_summary_sha256=None,
                persistence_attribution_sha256=None,
                control_persistence_seconds=None,
                persistence_ratio=None,
                shard_manifest_sha256_by_id=None,
                shard_actual_bytes_by_id=None,
                transfer_mode=None,
                archive_transfer_seconds=None,
            )
            for batch in campaign.batches
        )
        planned_campaign = replace(
            campaign,
            status="planned",
            batches=planned_batches,
            batch_persistence_envelope_sha256_by_id={},
        )
        expected_failed_batch = planned_batches[0].mark_failed(injected_reason)
        expected_failed_campaign = planned_campaign.with_batch(expected_failed_batch).mark_failed(
            injected_reason
        )
        deterministic_payload = hashlib.sha256(campaign.run_label.encode("utf-8")).digest() * 256
        if (
            failed_batch.to_dict() != expected_failed_batch.to_dict()
            or failed_campaign.to_dict() != expected_failed_campaign.to_dict()
            or failure.get("expected_total_bytes") != len(deterministic_payload)
            or failure.get("injected_after_bytes") != len(deterministic_payload) // 2
            or partial_path.read_bytes() != deterministic_payload[: len(deterministic_payload) // 2]
        ):
            raise ArtifactIntegrityError("failure-state drill transition does not replay")

        raw_ref = _one_artifact(reader, "raw_replay_drill")
        raw_drill = reader.read_json(str(raw_ref["relative_path"]))
        raw_results = raw_drill.get("results")
        if (
            raw_drill.get("schema_version") != "stage05.2-raw-replay-drill-v1"
            or raw_drill.get("run_label") != campaign.run_label
            or raw_drill.get("passed") is not True
            or raw_drill.get("batch_count") != len(campaign.batches)
            or not isinstance(raw_results, list)
            or len(raw_results) != len(campaign.batches)
        ):
            raise ArtifactIntegrityError("raw replay drill summary is incomplete")
        raw_by_batch = {
            str(item.get("batch_id")): item for item in raw_results if isinstance(item, Mapping)
        }
        for batch in campaign.batches:
            item = raw_by_batch.get(batch.batch_id)
            root = locator.resolve(batch.root_alias)
            batch_dir = root.absolute_path.joinpath(*Path(batch.logical_path).parts)
            batch_reader = ArtifactReader(batch_dir)
            if (
                item is None
                or item.get("raw_manifest_sha256") != _sha256(batch_reader.result.manifest_path)
                or item.get("directory_checksum_sha256") != batch.checksum_sha256
                or item.get("actual_bytes") != batch.actual_bytes
                or item.get("artifact_count") != len(batch_reader.manifest.get("artifacts", ()))
            ):
                raise ArtifactIntegrityError("raw replay drill does not match archived batch")

        archive_ref = _one_artifact(reader, "archive_dry_run")
        archive = reader.read_json(str(archive_ref["relative_path"]))
        results = archive.get("results")
        if (
            archive.get("schema_version") != "stage05.2-archive-dry-run-v1"
            or archive.get("run_label") != campaign.run_label
            or archive.get("staging_root_alias") != staging_root_alias
            or archive.get("archive_root_aliases_exercised") != list(archive_root_aliases)
            or archive.get("passed") is not True
            or not isinstance(results, list)
            or len(results) != len(archive_root_aliases)
        ):
            raise ArtifactIntegrityError("archive dry-run summary is incomplete")
        observed_archive_aliases = tuple(
            str(item.get("archive_root_alias", "")) for item in results if isinstance(item, Mapping)
        )
        if observed_archive_aliases != archive_root_aliases:
            raise ArtifactIntegrityError("archive dry-run alias coverage is not exact")
        staging_root = locator.resolve(staging_root_alias)
        for item in results:
            if not isinstance(item, Mapping):
                raise ArtifactIntegrityError("archive dry-run result is invalid")
            alias = str(item.get("archive_root_alias", ""))
            logical_path = str(item.get("logical_path", ""))
            expected_logical_path = f"{campaign.run_label}/archive_dry_run/{alias}"
            if alias not in archive_root_aliases or logical_path != expected_logical_path:
                raise ArtifactIntegrityError("archive dry-run alias is invalid")
            root = locator.resolve(alias)
            archived_dir = root.absolute_path.joinpath(*Path(logical_path).parts)
            source_dir = staging_root.absolute_path.joinpath(*Path(logical_path).parts)
            incoming = archived_dir.with_name(f"{archived_dir.name}.incoming")
            expected_mode = (
                "same_volume_atomic_rename"
                if staging_root.volume.device_uuid == root.volume.device_uuid
                else "cross_volume_verified_copy"
            )
            archived = load_batch_manifest(archived_dir / "batch_manifest.json")
            if (
                archived.status != "archived"
                or archived.root_alias != alias
                or archived.logical_path != logical_path
                or archived.volume.to_dict() != item.get("volume_identity")
                or archived.transfer_mode != expected_mode
                or item.get("transfer_mode") != expected_mode
                or archived.archive_transfer_seconds != item.get("archive_transfer_seconds")
                or archived.checksum_sha256 != item.get("checksum_sha256")
                or archived.actual_bytes != item.get("actual_bytes")
                or directory_checksum(archived_dir) != archived.checksum_sha256
                or directory_byte_count(archived_dir) != archived.actual_bytes
                or source_dir.exists()
                or incoming.exists()
            ):
                raise ArtifactIntegrityError("archive dry-run batch does not replay")

        publication_ref = _one_artifact(reader, "publication_dry_run")
        publication = reader.read_json(str(publication_ref["relative_path"]))
        payload_bytes = b'{"status":"dry_run"}\n'
        payload_sha = hashlib.sha256(payload_bytes).hexdigest()
        trusted = {
            "schema_version": "stage05.2-publication-dry-run-manifest-v1",
            "run_label": campaign.run_label,
            "payload_relative_path": "generation/payload.json",
            "payload_sha256": payload_sha,
            "status": "READY_DRY_RUN",
        }
        trusted_sha = hashlib.sha256(
            (json.dumps(trusted, indent=2, sort_keys=True) + "\n").encode("utf-8")
        ).hexdigest()
        if (
            publication.get("schema_version") != "stage05.2-publication-dry-run-v1"
            or publication.get("run_label") != campaign.run_label
            or publication.get("payload_sha256") != payload_sha
            or publication.get("trusted_manifest_sha256") != trusted_sha
            or publication.get("trusted_manifest_replaced_last") is not True
            or publication.get("passed") is not True
        ):
            raise ArtifactIntegrityError("publication dry-run summary does not replay")
        return True, "rolling capacity and all pilot campaign drills replayed"
    except (ArtifactIntegrityError, OSError, TypeError, ValueError) as error:
        return False, str(error)


def _expected_rolling_capacity_required(
    *,
    campaign: CampaignManifest,
    batch_index: int,
    phase: str,
    staging_root_alias: str,
    archive_root_aliases: tuple[str, ...],
) -> dict[str, int]:
    """Independently derive the canonical per-device reserve for one phase."""

    if phase not in {"pre_dispatch", "pre_archive", "post_archive"}:
        raise ArtifactIntegrityError("rolling capacity phase is invalid")
    try:
        current = campaign.batches[batch_index]
        staging_device = campaign.storage_roots[staging_root_alias].device_uuid
    except (IndexError, KeyError) as error:
        raise ArtifactIntegrityError("rolling capacity campaign identity is invalid") from error
    required: dict[str, int] = {}
    for alias in archive_root_aliases:
        volume = campaign.storage_roots[alias]
        reserve = 52 * 1024**3 if volume.device_uuid == staging_device else 50 * 1024**3
        required[volume.device_uuid] = max(required.get(volume.device_uuid, 0), reserve)
    future = campaign.batches[batch_index + 1 :]
    projected = campaign.batches[batch_index:] if phase == "pre_dispatch" else future
    for batch in projected:
        device = campaign.storage_roots[batch.archive_root_alias].device_uuid
        required[device] = required.get(device, 0) + batch.estimated_bytes
    if phase == "pre_archive":
        if current.actual_bytes is None:
            raise ArtifactIntegrityError("rolling capacity actual bytes are missing")
        target = campaign.storage_roots[current.archive_root_alias].device_uuid
        if target != staging_device:
            required[target] = required.get(target, 0) + current.actual_bytes
        staging_required = 20 * 1024**3
        if future:
            staging_required = (
                52 * 1024**3
                if target == staging_device
                else max(20 * 1024**3, 52 * 1024**3 - current.actual_bytes)
            )
        required[staging_device] = max(required.get(staging_device, 0), staging_required)
    else:
        staging_reserve = 20 * 1024**3
        if phase == "pre_dispatch" or future:
            staging_reserve = 52 * 1024**3
        required[staging_device] = max(required.get(staging_device, 0), staging_reserve)
    return dict(sorted(required.items()))


def _verify_accelerator_prerequisite(
    raw_dir: Path,
    *,
    campaign: CampaignManifest,
) -> tuple[dict[str, object], str, dict[str, object]]:
    requirement = stage052_contract(Stage052Component.BENCHMARK, "pilot").prerequisites[0]
    identity = verify_stage052_evidence_input(raw_dir, requirement)
    reader = ArtifactReader(raw_dir)
    metadata_ref = _one_artifact(reader, "manifest_metadata")
    metadata = reader.read_json(str(metadata_ref["relative_path"]))
    review_path = raw_dir / "review" / "review_manifest.json"
    review = _json_object(review_path)
    audit = validate_campaign_selection_lock(
        campaign_backend=campaign.selected_backend,
        campaign_exact_backend=campaign.selected_exact_backend,
        campaign_workers=campaign.selected_workers,
        campaign_native_profile=campaign.native_profile,
        prerequisite_metadata=metadata,
        prerequisite_review=review,
        prerequisite_identity=identity.to_dict(),
    )
    if not audit.passed:
        raise ArtifactIntegrityError(audit.detail)
    payload = {
        **identity.to_dict(),
        "accelerator_decision": review.get("accelerator_decision"),
        "selection_lock": audit.selection_lock,
    }
    return payload, identity.review_manifest_sha256, audit.selection_lock


def _verify_campaign_review_prerequisite(
    raw_dir: Path,
) -> tuple[dict[str, object], str]:
    path = raw_dir / "review" / "review_manifest.json"
    payload = _json_object(path)
    expected = {
        "schema_version": CAMPAIGN_REVIEW_SCHEMA,
        "run_label": raw_dir.name,
        "component": Stage052Component.BENCHMARK.value,
        "scope": "pilot",
        "status": PILOT_READY,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ArtifactIntegrityError(
                f"accepted pilot review {field} mismatch: {payload.get(field)!r}"
            )
    verify_stage052_campaign_gate_set(payload, scope="pilot")
    campaign_path = raw_dir / "campaign_manifest.json"
    pilot_campaign = load_campaign_manifest(campaign_path)
    if (
        pilot_campaign.scope != "pilot"
        or pilot_campaign.status != "complete"
        or pilot_campaign.run_label != raw_dir.name
    ):
        raise ArtifactIntegrityError("accepted pilot campaign identity is invalid")
    if payload.get("raw_campaign_manifest_sha256") != _sha256(campaign_path):
        raise ArtifactIntegrityError("accepted pilot raw campaign hash mismatch")
    reader = ArtifactReader(raw_dir)
    if payload.get("raw_manifest_sha256") != _sha256(reader.result.manifest_path):
        raise ArtifactIntegrityError("accepted pilot standard raw manifest hash mismatch")
    attribution_path = raw_dir / "control" / f"{raw_dir.name}_persistence_attribution.json"
    attribution_sidecar = attribution_path.with_suffix(".sha256")
    if (
        not attribution_path.is_file()
        or not attribution_sidecar.is_file()
        or not signed_sidecar_matches(attribution_path, attribution_sidecar)
        or payload.get("persistence_attribution_sha256") != _sha256(attribution_path)
        or payload.get("persistence_attribution_sidecar_sha256") != _sha256(attribution_sidecar)
    ):
        raise ArtifactIntegrityError("accepted pilot campaign persistence attribution is stale")
    selection = payload.get("selection_lock")
    native = payload.get("native_configuration")
    raw_decision = selection.get("accelerator_decision") if isinstance(selection, Mapping) else None
    decision = raw_decision if isinstance(raw_decision, str) else ""
    expected_backend = {
        "GPU_NOT_JUSTIFIED": "native_cpu",
        "NATIVE_CPU_RETAINED": "native_cpu",
        "ACCELERATOR_PROMOTED": "cuda",
    }.get(decision)
    if (
        not isinstance(selection, Mapping)
        or expected_backend is None
        or payload.get("selected_backend") != expected_backend
        or payload.get("selected_exact_backend") != "cpu_batch"
        or payload.get("selected_workers") not in {2, 4}
        or payload.get("native_profile") != "stage05.2-native-kernels-v1"
        or not isinstance(native, Mapping)
        or payload.get("native_kernel_config") != native
        or selection.get("selected_backend") != payload.get("selected_backend")
        or selection.get("selected_exact_backend") != payload.get("selected_exact_backend")
        or selection.get("selected_workers") != payload.get("selected_workers")
        or selection.get("native_profile") != payload.get("native_profile")
        or selection.get("native_kernel_config") != native
        or selection.get("native_config_sha256") != _canonical_sha256(native)
        or not _is_sha256(selection.get("accelerator_review_manifest_sha256"))
        or pilot_campaign.selected_backend != payload.get("selected_backend")
        or pilot_campaign.selected_exact_backend != payload.get("selected_exact_backend")
        or pilot_campaign.selected_workers != payload.get("selected_workers")
        or pilot_campaign.native_profile != payload.get("native_profile")
    ):
        raise ArtifactIntegrityError("accepted pilot F02 selection lock is invalid")
    staging_alias = payload.get("staging_root_alias")
    archive_aliases = payload.get("archive_root_aliases_exercised")
    if (
        not isinstance(staging_alias, str)
        or not staging_alias
        or not isinstance(archive_aliases, list)
        or not archive_aliases
        or any(not isinstance(alias, str) for alias in archive_aliases)
        or staging_alias in archive_aliases
    ):
        raise ArtifactIntegrityError("accepted pilot staging/archive root evidence is invalid")
    verify_stage052_review_files(raw_dir, payload)
    return payload, _sha256(path)


def _pilot_geometry(records: Sequence[CampaignGeometryRecord]) -> CampaignGeometryAudit:
    axis_count = sum(len(record.budgets_seconds) for record in records)
    declared = sum(sum(record.budgets_seconds) for record in records)
    checkpoints = sum(record.checkpoint_count for record in records)
    totals = (len(records), axis_count, declared, checkpoints)
    if totals != (36, 36, 1_080, 144):
        return CampaignGeometryAudit(
            False,
            "pilot totals mismatch: expected 36 shards/axes, 1080 seconds, 144 checkpoints",
            *totals,
        )
    counts = {item.instance: item.customer_count for item in BEST_KNOWN_VALUES}
    expected = tuple(
        sorted(
            (counts[instance], instance, seed)
            for instance in FORMAL_INSTANCES
            for seed in PILOT_SEEDS
        )
    )
    observed = tuple((record.customer_count, record.instance, record.seed) for record in records)
    if (
        observed != expected
        or tuple(record.shard_id for record in records)
        != tuple(f"shard{index:04d}" for index in range(1, 37))
        or any(
            record.budgets_seconds != (30,) or record.checkpoint_count != 4 for record in records
        )
    ):
        return CampaignGeometryAudit(False, "pilot shard identity/order mismatch", *totals)
    return CampaignGeometryAudit(True, "exact 12-instance x 3-seed pilot geometry passed", *totals)


def _campaign_geometry(
    scope: str,
    records: Sequence[CampaignGeometryRecord],
) -> CampaignGeometryAudit:
    return validate_formal_geometry(records) if scope == "formal" else _pilot_geometry(records)


def _campaign_preflight_free_bytes(
    *,
    payload: Mapping[str, object],
    config: BenchmarkCampaignConfig,
    locator: StorageRootLocator,
) -> dict[str, int]:
    """Replay the signed campaign-level power/load and capacity snapshot."""

    expected_fields = {
        "schema_version",
        "run_label",
        "scope",
        "power_source",
        "low_power_mode_enabled",
        "windows",
        "volume_identities",
        "free_bytes_by_alias",
    }
    if set(payload) != expected_fields:
        raise ArtifactIntegrityError("campaign preflight fields do not match the schema")
    aliases = (config.staging_root_alias, *config.archive_root_aliases)
    if (
        payload.get("schema_version") != "stage05.2-campaign-preflight-v1"
        or payload.get("run_label") != config.run_label
        or payload.get("scope") != config.scope
        or payload.get("power_source") != "AC Power"
        or payload.get("low_power_mode_enabled") is not False
        or payload.get("volume_identities") != locator.tracked_payload(aliases)
    ):
        raise ArtifactIntegrityError("campaign preflight identity/power/volume evidence is invalid")
    windows = payload.get("windows")
    if not isinstance(windows, list) or len(windows) != 2:
        raise ArtifactIntegrityError("campaign preflight must contain two load windows")
    previous_end: float | None = None
    for raw_window in windows:
        if not isinstance(raw_window, Mapping):
            raise ArtifactIntegrityError("campaign preflight load window is invalid")
        expected_window_fields = {
            "started_at_seconds",
            "duration_seconds",
            "maximum_load1",
            "maximum_unrelated_process_average_cores",
            "logical_cpu_count",
            "process_cpu_samples",
        }
        if set(raw_window) != expected_window_fields:
            raise ArtifactIntegrityError("campaign preflight load-window schema is invalid")
        started = _strict_float(raw_window.get("started_at_seconds"), "preflight start")
        duration = _strict_float(raw_window.get("duration_seconds"), "preflight duration")
        load1 = _strict_float(raw_window.get("maximum_load1"), "preflight load1")
        recorded_cores = _strict_float(
            raw_window.get("maximum_unrelated_process_average_cores"),
            "preflight unrelated cores",
        )
        logical_cpu_count = _strict_int(
            raw_window.get("logical_cpu_count"),
            "preflight logical CPU count",
        )
        raw_samples = raw_window.get("process_cpu_samples")
        if not isinstance(raw_samples, list) or len(raw_samples) < 2:
            raise ArtifactIntegrityError("campaign preflight process samples are incomplete")
        samples = tuple(
            ProcessCpuCounterSample.from_dict(dict(sample))
            for sample in raw_samples
            if isinstance(sample, Mapping)
        )
        if len(samples) != len(raw_samples):
            raise ArtifactIntegrityError("campaign preflight process sample is invalid")
        replayed_cores = maximum_process_average_cores(
            samples,
            logical_cpu_count=logical_cpu_count,
        )
        if (
            duration != 30.0
            or load1 < 0.0
            or load1 > 4.0
            or recorded_cores < 0.0
            or recorded_cores >= 1.0
            or not math.isclose(replayed_cores, recorded_cores, rel_tol=0.0, abs_tol=1e-12)
            or (
                previous_end is not None
                and not math.isclose(started, previous_end, rel_tol=0.0, abs_tol=1e-9)
            )
        ):
            raise ArtifactIntegrityError("campaign preflight load-window replay failed")
        previous_end = started + duration
    raw_free = payload.get("free_bytes_by_alias")
    if not isinstance(raw_free, Mapping) or set(raw_free) != set(aliases):
        raise ArtifactIntegrityError("campaign preflight free-byte snapshot is incomplete")
    free_by_alias = {
        str(alias): _strict_int(value, "campaign preflight free bytes")
        for alias, value in raw_free.items()
    }
    if any(value < 0 for value in free_by_alias.values()):
        raise ArtifactIntegrityError("campaign preflight free bytes cannot be negative")
    return free_by_alias


def _independent_capacity_plan(
    *,
    config: BenchmarkCampaignConfig,
    plan: CampaignPlan,
    locator: StorageRootLocator,
    free_bytes_by_alias: Mapping[str, int],
) -> dict[str, object]:
    """Independently apply the fixed reserve and ordered first-fit rules."""

    aliases = (config.staging_root_alias, *config.archive_root_aliases)
    roots = {alias: locator.resolve(alias) for alias in aliases}
    free_by_device: dict[str, int] = {}
    for alias in aliases:
        device = roots[alias].volume.device_uuid
        measured = free_bytes_by_alias[alias]
        free_by_device[device] = min(free_by_device.get(device, measured), measured)
    staging_device = roots[config.staging_root_alias].volume.device_uuid
    external_floor = 82 * GIB
    if free_by_device[staging_device] < external_floor:
        raise ArtifactIntegrityError("campaign staging capacity is below the fixed reserve")
    representative_alias: dict[str, str] = {}
    usable: dict[str, int] = {}
    for alias in config.archive_root_aliases:
        root = roots[alias]
        device = root.volume.device_uuid
        if device in representative_alias:
            continue
        representative_alias[device] = alias
        if device == staging_device:
            reserve = external_floor
        else:
            filesystem = root.volume.filesystem.casefold()
            if alias == "d_archive":
                if filesystem not in {"9p", "ntfs"}:
                    raise ArtifactIntegrityError("D campaign archive must use WSL 9p/NTFS")
            elif filesystem != "apfs":
                raise ArtifactIntegrityError("historical internal archive must use APFS")
            reserve = 50 * GIB
        usable[device] = max(0, free_by_device[device] - reserve)
    if sum(usable.values()) < plan.estimated_bytes:
        raise ArtifactIntegrityError("campaign capacity is insufficient after fixed reserves")
    remaining = dict(usable)
    assignments: list[dict[str, object]] = []
    ordered_devices = tuple(representative_alias)
    for batch in plan.batches:
        selected = next(
            (device for device in ordered_devices if remaining[device] >= batch.estimated_bytes),
            None,
        )
        if selected is None:
            raise ArtifactIntegrityError("campaign capacity cannot place an indivisible batch")
        remaining[selected] -= batch.estimated_bytes
        assignments.append(
            {
                "batch_id": batch.batch_id,
                "root_alias": representative_alias[selected],
                "estimated_bytes": batch.estimated_bytes,
            }
        )
    return {
        "assignments": assignments,
        "usable_bytes_by_device": dict(sorted(usable.items())),
        "remaining_bytes_by_device": dict(sorted(remaining.items())),
    }


def audit_campaign_planning(
    *,
    campaign: CampaignManifest,
    config: BenchmarkCampaignConfig,
    expected_plan: CampaignPlan,
    plan_payload: Mapping[str, object],
    preflight_payload: Mapping[str, object],
    capacity_payload: Mapping[str, object],
    locator: StorageRootLocator,
) -> tuple[bool, str]:
    """Verify exact plan geometry, next-fit batches, reserves, and assignments."""

    try:
        configuration_sha256 = _canonical_sha256(config.to_dict())
        if (
            campaign.run_label != config.run_label
            or campaign.scope != config.scope
            or campaign.configuration_sha256 != configuration_sha256
            or dict(plan_payload) != expected_plan.to_dict()
        ):
            raise ArtifactIntegrityError("campaign configuration or signed plan does not replay")
        free_by_alias = _campaign_preflight_free_bytes(
            payload=preflight_payload,
            config=config,
            locator=locator,
        )
        expected_capacity = _independent_capacity_plan(
            config=config,
            plan=expected_plan,
            locator=locator,
            free_bytes_by_alias=free_by_alias,
        )
        if dict(capacity_payload) != expected_capacity:
            raise ArtifactIntegrityError("campaign capacity plan does not replay")
        raw_assignments = expected_capacity["assignments"]
        if not isinstance(raw_assignments, list):
            raise ArtifactIntegrityError("campaign capacity assignments are invalid")
        assignments = {
            str(item["batch_id"]): item for item in raw_assignments if isinstance(item, Mapping)
        }
        if len(campaign.batches) != len(expected_plan.batches):
            raise ArtifactIntegrityError("campaign manifest batch count differs from the plan")
        for planned, recorded in zip(expected_plan.batches, campaign.batches, strict=True):
            assignment = assignments.get(planned.batch_id)
            if (
                assignment is None
                or recorded.batch_id != planned.batch_id
                or recorded.shard_ids != tuple(shard.shard_id for shard in planned.shards)
                or recorded.estimated_bytes != planned.estimated_bytes
                or recorded.archive_root_alias != assignment.get("root_alias")
                or recorded.logical_path != f"{campaign.run_label}/{planned.batch_id}"
            ):
                raise ArtifactIntegrityError("campaign manifest batch/assignment identity drifted")
        return True, "campaign plan, preflight capacity, batches, and assignments replayed"
    except (ArtifactIntegrityError, KeyError, OSError, TypeError, ValueError) as error:
        return False, str(error)


def _formal_storage_observations_from_pilot(
    *,
    prerequisite_dir: Path,
    prerequisite_payload: Mapping[str, object],
    locator: StorageRootLocator,
) -> tuple[PilotStorageObservation, ...]:
    """Rebuild Formal estimates from the accepted pilot's archived shard bytes."""

    pilot_campaign = load_campaign_manifest(prerequisite_dir / "campaign_manifest.json")
    staging_alias = prerequisite_payload.get("staging_root_alias")
    raw_archive_aliases = prerequisite_payload.get("archive_root_aliases_exercised")
    if not isinstance(staging_alias, str) or not isinstance(raw_archive_aliases, list):
        raise ArtifactIntegrityError("accepted pilot root roles are invalid")
    archive_aliases = tuple(str(alias) for alias in raw_archive_aliases)
    pilot_config = BenchmarkCampaignConfig.pilot(
        run_label=pilot_campaign.run_label,
        staging_root_alias=staging_alias,
        archive_root_aliases=archive_aliases,
        selected_backend=pilot_campaign.selected_backend,
        selected_exact_backend=pilot_campaign.selected_exact_backend,
        selected_workers=pilot_campaign.selected_workers,
        native_profile=pilot_campaign.native_profile,
    )
    expected_pilot_plan = pilot_config.build_plan()
    pilot_reader = ArtifactReader(prerequisite_dir)
    pilot_plan_ref = _one_artifact(pilot_reader, "campaign_plan")
    pilot_plan_payload = pilot_reader.read_json(str(pilot_plan_ref["relative_path"]))
    if pilot_plan_payload != expected_pilot_plan.to_dict():
        raise ArtifactIntegrityError("accepted pilot signed campaign plan is non-canonical")
    actual_by_shard: dict[str, int] = {}
    for batch in pilot_campaign.batches:
        if batch.shard_actual_bytes_by_id is None:
            raise ArtifactIntegrityError("accepted pilot shard byte evidence is missing")
        root = locator.resolve(batch.root_alias)
        batch_dir = root.absolute_path.joinpath(*Path(batch.logical_path).parts)
        if (
            directory_checksum(batch_dir) != batch.checksum_sha256
            or directory_byte_count(batch_dir) != batch.actual_bytes
        ):
            raise ArtifactIntegrityError("accepted pilot archived batch no longer replays")
        overlap = set(actual_by_shard).intersection(batch.shard_actual_bytes_by_id)
        if overlap:
            raise ArtifactIntegrityError("accepted pilot contains duplicate shard byte evidence")
        actual_by_shard.update(batch.shard_actual_bytes_by_id)
    expected_ids = {shard.shard_id for shard in expected_pilot_plan.shards}
    if set(actual_by_shard) != expected_ids:
        raise ArtifactIntegrityError("accepted pilot shard byte coverage is incomplete")
    return tuple(
        PilotStorageObservation(
            family=shard.family,
            customer_count=shard.customer_count,
            budget_seconds=30,
            compressed_bytes=actual_by_shard[shard.shard_id],
        )
        for shard in expected_pilot_plan.shards
    )


def _audit_campaign(
    *,
    campaign_dir: Path,
    benchmark_dir: Path,
    bks_path: Path,
    scope: str,
    prerequisite_dir: Path,
    locator: StorageRootLocator,
    volume_probe: Callable[[Path], VolumeIdentity],
) -> tuple[
    dict[str, dict[str, object]],
    _ReviewEvidence,
    str,
    dict[str, object],
]:
    gates: dict[str, dict[str, object]] = {}
    evidence = _ReviewEvidence.empty()
    campaign_path = campaign_dir / "campaign_manifest.json"
    campaign = load_campaign_manifest(campaign_path)
    raw_manifest_hash = _sha256(campaign_path)
    standard_reader = ArtifactReader(campaign_dir)
    standard_metadata_ref = _one_artifact(standard_reader, "manifest_metadata")
    standard_metadata = standard_reader.read_json(str(standard_metadata_ref["relative_path"]))
    if (
        standard_reader.manifest.get("component") != Stage052Component.BENCHMARK.value
        or standard_reader.manifest.get("evidence_completeness") != "complete"
        or campaign.status != "complete"
    ):
        raise ArtifactIntegrityError(
            "standard raw envelope does not bind a complete canonical campaign manifest"
        )
    top_staging_alias = standard_metadata.get("staging_root_alias")
    top_archive_aliases = standard_metadata.get("planned_archive_root_aliases")
    if (
        not isinstance(top_staging_alias, str)
        or not isinstance(top_archive_aliases, list)
        or not top_archive_aliases
        or any(not isinstance(alias, str) for alias in top_archive_aliases)
        or len(set(top_archive_aliases)) != len(top_archive_aliases)
        or top_staging_alias in top_archive_aliases
        or top_staging_alias not in campaign.storage_roots
        or set(top_archive_aliases) != set(campaign.storage_roots).difference({top_staging_alias})
    ):
        raise ArtifactIntegrityError("campaign top-level root roles are invalid")
    evidence.standard_raw_manifest_sha256 = _sha256(standard_reader.result.manifest_path)
    campaign_identity = (
        campaign.status == "complete"
        and campaign.scope == scope
        and campaign.run_label == campaign_dir.name
        and campaign.selected_backend in {"native_cpu", "cuda"}
        and campaign.selected_exact_backend == "cpu_batch"
        and campaign.native_profile == "stage05.2-native-kernels-v1"
    )
    gates["campaign_identity"] = {
        "passed": campaign_identity,
        "detail": "complete canonical campaign manifest"
        if campaign_identity
        else "campaign identity/status/backend/native profile mismatch",
    }
    try:
        current_source_snapshot = verify_stage052_source_snapshot(repository_root())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        current_source_snapshot = {}
        gates["source_snapshot"] = {"passed": False, "detail": str(error)}
    else:
        source_passed = standard_metadata.get("source_snapshot") == current_source_snapshot
        gates["source_snapshot"] = {
            "passed": source_passed,
            "detail": (
                "clean ext4 read-only source snapshot independently replayed"
                if source_passed
                else "campaign source snapshot does not match independent replay"
            ),
        }

    prerequisite_payload: dict[str, object] = {}
    selection_lock: dict[str, object] = {}
    try:
        if scope == "formal":
            prerequisite_payload, prerequisite_hash = _verify_campaign_review_prerequisite(
                prerequisite_dir
            )
            raw_selection = prerequisite_payload.get("selection_lock")
            if not isinstance(raw_selection, Mapping):
                raise ArtifactIntegrityError("accepted pilot selection lock is missing")
            selection_lock = dict(raw_selection)
        else:
            prerequisite_payload, prerequisite_hash, selection_lock = (
                _verify_accelerator_prerequisite(
                    prerequisite_dir,
                    campaign=campaign,
                )
            )
        selection_matches = (
            selection_lock.get("selected_backend") == campaign.selected_backend
            and selection_lock.get("selected_exact_backend") == campaign.selected_exact_backend
            and selection_lock.get("selected_workers") == campaign.selected_workers
            and selection_lock.get("native_profile") == campaign.native_profile
            and selection_lock.get("native_kernel_config") is not None
        )
        prerequisite_ok = (
            campaign.prerequisite_review_sha256 == prerequisite_hash and selection_matches
        )
        prerequisite_payload["campaign_prerequisite_review_sha256"] = prerequisite_hash
    except (ArtifactIntegrityError, OSError, TypeError, ValueError) as error:
        prerequisite_ok = False
        prerequisite_hash = ""
        prerequisite_payload = {"error": str(error)}
        selection_lock = {}
    evidence.selection_lock = selection_lock
    gates["accepted_prerequisite"] = {
        "passed": prerequisite_ok,
        "detail": "campaign is bound to its exact accepted prerequisite review"
        if prerequisite_ok
        else f"campaign prerequisite binding failed: {prerequisite_payload}",
    }

    root_failures: list[str] = []
    for alias, expected in campaign.storage_roots.items():
        try:
            configured = locator.resolve(alias)
            observed = volume_probe(configured.absolute_path)
            if configured.volume != expected or observed != expected:
                root_failures.append(alias)
        except (ArtifactIntegrityError, KeyError, OSError, RuntimeError):
            root_failures.append(alias)
    gates["storage_roots"] = {
        "passed": not root_failures,
        "detail": "all archive root aliases and live volume identities passed"
        if not root_failures
        else f"storage root identity failures: {sorted(root_failures)}",
    }
    try:
        planning_config = (
            BenchmarkCampaignConfig.pilot(
                run_label=campaign.run_label,
                staging_root_alias=str(top_staging_alias),
                archive_root_aliases=tuple(str(alias) for alias in top_archive_aliases),
                selected_backend=campaign.selected_backend,
                selected_exact_backend=campaign.selected_exact_backend,
                selected_workers=campaign.selected_workers,
                native_profile=campaign.native_profile,
            )
            if scope == "pilot"
            else BenchmarkCampaignConfig.formal(
                run_label=campaign.run_label,
                staging_root_alias=str(top_staging_alias),
                archive_root_aliases=tuple(str(alias) for alias in top_archive_aliases),
                selected_backend=campaign.selected_backend,
                selected_exact_backend=campaign.selected_exact_backend,
                selected_workers=campaign.selected_workers,
                native_profile=campaign.native_profile,
            )
        )
        planning_observations = (
            ()
            if scope == "pilot"
            else _formal_storage_observations_from_pilot(
                prerequisite_dir=prerequisite_dir,
                prerequisite_payload=prerequisite_payload,
                locator=locator,
            )
        )
        expected_plan = planning_config.build_plan(planning_observations)
        plan_ref = _one_artifact(standard_reader, "campaign_plan")
        preflight_ref = _one_artifact(standard_reader, "campaign_preflight")
        raw_capacity = standard_metadata.get("campaign_capacity_plan")
        if (
            not isinstance(raw_capacity, Mapping)
            or standard_metadata.get("campaign_configuration_sha256")
            != campaign.configuration_sha256
        ):
            raise ArtifactIntegrityError("campaign capacity/configuration metadata is invalid")
        planning_passed, planning_detail = audit_campaign_planning(
            campaign=campaign,
            config=planning_config,
            expected_plan=expected_plan,
            plan_payload=standard_reader.read_json(str(plan_ref["relative_path"])),
            preflight_payload=standard_reader.read_json(str(preflight_ref["relative_path"])),
            capacity_payload=dict(raw_capacity),
            locator=locator,
        )
    except (
        ArtifactIntegrityError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        planning_passed = False
        planning_detail = str(error)
    gates["campaign_planning_replay"] = {
        "passed": planning_passed,
        "detail": planning_detail,
    }
    bks = validate_bks_reference(bks_path)
    gates["bks_model_compatibility"] = {
        "passed": bks.passed,
        "detail": bks.detail,
    }
    expected_input_instances = (
        tuple(FORMAL_INSTANCES)
        if scope == "pilot"
        else tuple(record.instance for record in BEST_KNOWN_VALUES)
    )
    live_instance_hashes: dict[str, str] = {}
    try:
        for instance in expected_input_instances:
            path = benchmark_dir / f"{instance}.txt"
            live_instance_hashes[instance] = _sha256(path)
    except OSError as error:
        gates["instance_input_hashes"] = {
            "passed": False,
            "detail": f"cannot hash canonical campaign instances: {error}",
        }
    else:
        gates["instance_input_hashes"] = {
            "passed": True,
            "detail": f"hashed {len(live_instance_hashes)} canonical instance inputs",
        }

    batch_failures: list[str] = []
    metadata_identities: set[tuple[str, str, str]] = set()
    observed_shard_ids: set[str] = set()
    aggregate_event_rows = 0
    for embedded_batch in campaign.batches:
        try:
            root = locator.resolve(embedded_batch.root_alias)
            batch_dir = root.absolute_path.joinpath(*Path(embedded_batch.logical_path).parts)
            disk_batch = load_batch_manifest(batch_dir / "batch_manifest.json")
            if disk_batch.to_dict() != embedded_batch.to_dict():
                raise ArtifactIntegrityError("campaign/batch manifest bidirectionality failed")
            staging_root = locator.resolve(str(top_staging_alias))
            expected_transfer_mode = (
                "same_volume_atomic_rename"
                if staging_root.volume.device_uuid == root.volume.device_uuid
                else "cross_volume_verified_copy"
            )
            source_dir = staging_root.absolute_path.joinpath(
                *Path(embedded_batch.logical_path).parts
            )
            incoming_dir = batch_dir.with_name(f"{batch_dir.name}.incoming")
            if (
                directory_checksum(batch_dir) != embedded_batch.checksum_sha256
                or directory_byte_count(batch_dir) != embedded_batch.actual_bytes
                or embedded_batch.physical_schema != "screening_decisions_v3"
                or embedded_batch.transfer_mode != expected_transfer_mode
                or source_dir.exists()
                or incoming_dir.exists()
            ):
                raise ArtifactIntegrityError(
                    "batch payload/archive transfer/checksum state mismatch"
                )
            reader = _batch_reader(batch_dir, run_label=campaign.run_label)
            metadata_ref, metadata_path = _batch_control_artifact(reader, "manifest_metadata")
            del metadata_ref
            metadata = _json_object(metadata_path)
            metadata_ok, metadata_detail, metadata_identity = _validate_batch_metadata(
                metadata,
                campaign=campaign,
                selection_lock=selection_lock,
                source_snapshot=current_source_snapshot,
            )
            if not metadata_ok:
                raise ArtifactIntegrityError(metadata_detail)
            provenance = metadata.get("performance_provenance")
            if (
                not isinstance(provenance, Mapping)
                or _validated_instance_hashes(provenance) != live_instance_hashes
            ):
                raise ArtifactIntegrityError(
                    "batch instance hashes differ from the live canonical inputs"
                )
            metadata_identities.add(metadata_identity)
            staging_alias = metadata.get("staging_root_alias")
            archive_aliases = metadata.get("archive_root_aliases_exercised")
            if (
                not isinstance(staging_alias, str)
                or staging_alias not in campaign.storage_roots
                or not isinstance(archive_aliases, list)
                or not archive_aliases
                or any(
                    not isinstance(alias, str) or alias not in campaign.storage_roots
                    for alias in archive_aliases
                )
                or staging_alias in archive_aliases
                or embedded_batch.archive_root_alias not in archive_aliases
            ):
                raise ArtifactIntegrityError(
                    "batch staging/archive root roles are invalid or mixed"
                )
            evidence.staging_root_aliases.add(staging_alias)
            evidence.archive_root_aliases.update(map(str, archive_aliases))
            shard_refs = [
                item
                for item in reader.manifest.get("artifacts", ())
                if isinstance(item, Mapping) and item.get("artifact_type") == "shard_manifest"
            ]
            if len(shard_refs) != len(embedded_batch.shard_ids):
                raise ArtifactIntegrityError("batch shard-manifest count mismatch")
            shard_payloads: list[dict[str, object]] = []
            replay_by_axis: dict[tuple[str, int, str], dict[str, object]] = {}
            checkpoint_count_before = len(evidence.checkpoint_rows)
            event_rows = 0
            digest_to_shard = {
                digest: shard_id
                for shard_id, digest in (embedded_batch.shard_manifest_sha256_by_id or {}).items()
            }
            for shard_ref in shard_refs:
                shard_path = batch_dir / str(shard_ref["relative_path"])
                shard_digest = _sha256(shard_path)
                shard_id = digest_to_shard.get(shard_digest)
                if shard_id is None or shard_id in observed_shard_ids:
                    raise ArtifactIntegrityError(
                        "batch shard checksum does not map uniquely to campaign shard ID"
                    )
                observed_shard_ids.add(shard_id)
                _verify_manifest_sidecar(shard_path)
                shard = _json_object(shard_path)
                shard_payloads.append(shard)
                expected_ordinal = int(shard_id.removeprefix("shard")) - 1
                if (
                    shard.get("run_label") != campaign.run_label
                    or shard.get("evidence_completeness") != "complete"
                    or shard.get("storage_policy_version") != "artifact-storage-v2"
                    or shard.get("shard_ordinal") != expected_ordinal
                ):
                    raise ArtifactIntegrityError("shard identity/completeness mismatch")
                shard_directory = shard_path.parent
                declared_shard_paths = {
                    str(item.get("relative_path", "")) for item in _artifacts_for_directory(shard)
                }
                physical_shard_paths = {
                    path.relative_to(batch_dir).as_posix()
                    for path in shard_directory.iterdir()
                    if path.is_file() and not path.name.startswith("._")
                }
                shard_envelope = {
                    shard_path.relative_to(batch_dir).as_posix(),
                    _sidecar_for(shard_path).relative_to(batch_dir).as_posix(),
                }
                if physical_shard_paths - shard_envelope != declared_shard_paths:
                    raise ArtifactIntegrityError(
                        "shard/raw/solution/trace/event bidirectionality failed"
                    )
                actual_shard_bytes = sum(
                    path.stat().st_size
                    for path in shard_directory.iterdir()
                    if path.is_file() and not path.name.startswith("._")
                )
                declared_shard_bytes = (embedded_batch.shard_actual_bytes_by_id or {}).get(shard_id)
                if actual_shard_bytes != declared_shard_bytes:
                    raise ArtifactIntegrityError("shard byte count mismatch")
                replay, checkpoints, logical_events = _replay_shard(
                    reader=reader,
                    shard_manifest=shard,
                    benchmark_dir=benchmark_dir,
                    scope=scope,
                )
                event_rows += logical_events
                evidence.checkpoint_rows.extend(checkpoints)
                for row in replay:
                    axis_identity = (
                        str(row["instance"]),
                        _strict_int(row["seed"], "seed"),
                        str(row["axis"]),
                    )
                    if axis_identity in replay_by_axis:
                        raise ArtifactIntegrityError("duplicate replayed axis identity")
                    replay_by_axis[axis_identity] = row
                customer_count = _strict_int(
                    next(iter(replay))["customer_count"] if replay else None,
                    "customer_count",
                )
                evidence.geometry.append(
                    CampaignGeometryRecord(
                        shard_id=shard_id,
                        batch_id=embedded_batch.batch_id,
                        instance=str(shard["instance"]),
                        seed=_strict_int(shard["seed"], "seed"),
                        customer_count=customer_count,
                        budgets_seconds=tuple(
                            _axis_budget(replay_identity[2])
                            for replay_identity in sorted(
                                replay_by_axis,
                                key=lambda item: (item[0], item[1], _axis_budget(item[2])),
                            )
                            if replay_identity[0] == shard["instance"]
                            and replay_identity[1] == shard["seed"]
                        ),
                        checkpoint_count=len(checkpoints),
                    )
                )
            if event_rows != embedded_batch.row_count:
                raise ArtifactIntegrityError("batch logical event row_count mismatch")
            aggregate_event_rows += event_rows

            per_run_ref, per_run_path = _batch_control_artifact(reader, "per_run_results")
            rows = _read_csv_rows(per_run_path)
            if per_run_ref.get("row_count") != len(rows) or len(rows) != len(replay_by_axis):
                raise ArtifactIntegrityError("batch per-run row count mismatch")
            normalized_rows: list[dict[str, object]] = []
            for row in rows:
                axis_identity = (
                    str(row.get("instance", "")),
                    _strict_int(row.get("seed"), "seed"),
                    str(row.get("axis", "")),
                )
                replayed = replay_by_axis.get(axis_identity)
                if replayed is None:
                    raise ArtifactIntegrityError("per-run axis is not in raw replay")
                recorded_objective = SolutionObjective(
                    vehicle_count=_strict_int(row.get("vehicle_count"), "vehicle_count"),
                    total_distance=_strict_float(row.get("total_distance"), "total_distance"),
                    total_charging_time=_strict_float(
                        row.get("total_charging_time"), "total_charging_time"
                    ),
                    charging_count=_strict_int(row.get("charging_count"), "charging_count"),
                )
                replayed_objective = SolutionObjective(
                    vehicle_count=_strict_int(replayed["vehicle_count"], "replayed vehicle_count"),
                    total_distance=_strict_float(
                        replayed["total_distance"], "replayed total_distance"
                    ),
                    total_charging_time=_strict_float(
                        replayed["total_charging_time"],
                        "replayed total_charging_time",
                    ),
                    charging_count=_strict_int(
                        replayed["charging_count"], "replayed charging_count"
                    ),
                )
                if (
                    compare_objectives(recorded_objective, replayed_objective)
                    is not ObjectiveComparison.EQUAL
                    or _strict_int(row.get("native_fallbacks"), "native_fallbacks") != 0
                    or _strict_int(
                        row.get("native_protocol_fallbacks"),
                        "native_protocol_fallbacks",
                    )
                    != 0
                ):
                    raise ArtifactIntegrityError("per-run/raw/native reconciliation failed")
                normalized_rows.append({**row, **replayed, "batch_id": embedded_batch.batch_id})
            persistence_ok, persistence_detail, persistence = _audit_batch_persistence(
                batch_dir=batch_dir,
                reader=reader,
                batch=embedded_batch,
                campaign=campaign,
                staging_root_alias=str(staging_alias),
                rows=normalized_rows,
                run_label=campaign.run_label,
                scope=scope,
                batch_id=embedded_batch.batch_id,
            )
            evidence.persistence_rows.append(persistence)
            if not persistence_ok:
                raise ArtifactIntegrityError(persistence_detail)
            evidence.per_run_rows.extend(normalized_rows)

            resource_ref, resource_path = _batch_control_artifact(reader, "resource_summary")
            resource = _json_object(resource_path)
            if _sha256(resource_path) != embedded_batch.resource_summary_sha256:
                raise ArtifactIntegrityError("batch resource summary hash mismatch")
            resource_ok, resource_detail, resource_row = _validate_batch_resources(
                resource=resource,
                shards=shard_payloads,
                workers=campaign.selected_workers,
                run_label=campaign.run_label,
            )
            if not resource_ok:
                raise ArtifactIntegrityError(resource_detail)
            evidence.resource_rows.append({"batch_id": embedded_batch.batch_id, **resource_row})
            power_ref, power_path = _batch_control_artifact(reader, "power_load")
            del power_ref
            power_ok, power_detail = _validate_power_load(
                _json_object(power_path),
                run_label=campaign.run_label,
                batch_id=embedded_batch.batch_id,
            )
            if not power_ok:
                raise ArtifactIntegrityError(power_detail)
            if len(evidence.checkpoint_rows) - checkpoint_count_before != sum(
                item.checkpoint_count
                for item in evidence.geometry
                if item.batch_id == embedded_batch.batch_id
            ):
                raise ArtifactIntegrityError("batch checkpoint count mismatch")
        except (
            ArtifactIntegrityError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            batch_failures.append(f"{embedded_batch.batch_id}: {error}")
            break

    gates["batch_shard_artifact_replay"] = {
        "passed": not batch_failures,
        "detail": "campaign->batch->shard->raw/solution/trace/event replay passed"
        if not batch_failures
        else "; ".join(batch_failures[:5]),
    }
    gates["runtime_provenance"] = {
        "passed": not batch_failures and len(metadata_identities) == 1,
        "detail": "all batches share one commit/config/non-editable runtime"
        if not batch_failures and len(metadata_identities) == 1
        else f"mixed or invalid runtime identities: {metadata_identities}",
    }
    planned_archive_aliases = {batch.archive_root_alias for batch in campaign.batches}
    root_roles_passed = (
        not batch_failures
        and len(evidence.staging_root_aliases) == 1
        and evidence.archive_root_aliases == planned_archive_aliases
        and evidence.staging_root_aliases.isdisjoint(evidence.archive_root_aliases)
    )
    gates["storage_root_roles"] = {
        "passed": root_roles_passed,
        "detail": (
            "one staging alias and the exact planned archive aliases passed"
            if root_roles_passed
            else "staging alias and planned archive aliases are mixed or incomplete"
        ),
    }
    if scope == "formal":
        pilot_aliases = prerequisite_payload.get("archive_root_aliases_exercised")
        pilot_staging = prerequisite_payload.get("staging_root_alias")
        alias_coverage = (
            isinstance(pilot_aliases, list)
            and planned_archive_aliases.issubset(set(map(str, pilot_aliases)))
            and len(evidence.staging_root_aliases) == 1
            and pilot_staging == next(iter(evidence.staging_root_aliases))
        )
        gates["pilot_archive_root_coverage"] = {
            "passed": alias_coverage,
            "detail": (
                "Formal planned archive aliases are covered by the accepted pilot drills"
                if alias_coverage
                else "Formal planned archive/staging aliases exceed accepted pilot coverage"
            ),
        }
    controls_passed, controls_detail = _audit_campaign_controls(
        campaign_dir=campaign_dir,
        reader=standard_reader,
        campaign=campaign,
        locator=locator,
        staging_root_alias=str(top_staging_alias),
        archive_root_aliases=tuple(str(alias) for alias in top_archive_aliases),
    )
    if scope == "pilot":
        if controls_passed:
            evidence.verified_archive_root_aliases.update(map(str, top_archive_aliases))
        controls_passed = controls_passed and root_roles_passed
        gates["pilot_campaign_drills"] = {
            "passed": controls_passed,
            "detail": controls_detail,
        }
    else:
        gates["rolling_capacity_replay"] = {
            "passed": controls_passed,
            "detail": controls_detail,
        }
    geometry = _campaign_geometry(scope, evidence.geometry)
    gates["campaign_geometry"] = {
        "passed": geometry.passed,
        "detail": geometry.detail,
    }
    expected_shards = campaign.shard_count
    gates["unique_shard_identity"] = {
        "passed": len(observed_shard_ids) == expected_shards,
        "detail": f"observed {len(observed_shard_ids)} unique shards; expected {expected_shards}",
    }
    (
        campaign_persistence_passed,
        campaign_persistence_detail,
        campaign_persistence_row,
        evidence.campaign_persistence_attribution_sha256,
        evidence.campaign_persistence_attribution_sidecar_sha256,
    ) = _audit_campaign_persistence(
        campaign_dir=campaign_dir,
        reader=standard_reader,
        campaign=campaign,
        batch_rows=evidence.persistence_rows,
        staging_root_alias=str(top_staging_alias),
        archive_root_aliases=tuple(str(alias) for alias in top_archive_aliases),
    )
    if campaign_persistence_row:
        evidence.persistence_rows.append(campaign_persistence_row)
    gates["persistence_ratio"] = {
        "passed": not batch_failures and campaign_persistence_passed,
        "detail": campaign_persistence_detail,
    }
    gates["resource_limits"] = {
        "passed": not batch_failures and len(evidence.resource_rows) == len(campaign.batches),
        "detail": "every batch passes per-worker and 12-GiB process-tree limits"
        if not batch_failures
        else "one or more batch resource gates failed",
    }
    gates["power_load"] = {
        "passed": not batch_failures,
        "detail": "preflight and continuous power/load gates passed"
        if not batch_failures
        else "one or more batch power/load gates failed",
    }
    if aggregate_event_rows <= 0:
        gates["event_evidence"] = {
            "passed": False,
            "detail": "campaign contains no logical event evidence",
        }
    else:
        gates["event_evidence"] = {
            "passed": True,
            "detail": f"streamed {aggregate_event_rows} logical events",
        }
    return gates, evidence, raw_manifest_hash, prerequisite_payload


def _csv_bytes(
    rows: Sequence[Mapping[str, object]],
    *,
    fallback_fields: Sequence[str],
) -> bytes:
    fields = list(fallback_fields)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return buffer.getvalue().encode("utf-8")


def _family_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("family", "")),
            _strict_int(row.get("customer_count"), "customer_count"),
            _strict_int(row.get("budget_seconds"), "budget_seconds"),
        )
        groups[key].append(row)
    output: list[dict[str, object]] = []
    for (family, customer_count, budget), items in sorted(groups.items()):
        output.append(
            {
                "family": family,
                "customer_count": customer_count,
                "budget_seconds": budget,
                "run_count": len(items),
                "median_solver_seconds": statistics.median(
                    _strict_float(item.get("solver_seconds"), "solver_seconds") for item in items
                ),
                "median_vehicle_count": statistics.median(
                    _strict_int(item.get("vehicle_count"), "vehicle_count") for item in items
                ),
                "median_total_distance": statistics.median(
                    _strict_float(item.get("total_distance"), "total_distance") for item in items
                ),
            }
        )
    return output


def _budget_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[_strict_int(row.get("budget_seconds"), "budget_seconds")].append(row)
    return [
        {
            "budget_seconds": budget,
            "run_count": len(items),
            "median_solver_seconds": statistics.median(
                _strict_float(item.get("solver_seconds"), "solver_seconds") for item in items
            ),
            "median_exact_started_calls": statistics.median(
                _strict_int(item.get("exact_started_calls"), "exact_started_calls")
                for item in items
            ),
            "median_effective_iterations": statistics.median(
                _strict_int(item.get("effective_iterations"), "effective_iterations")
                for item in items
            ),
        }
        for budget, items in sorted(groups.items())
    ]


def _review_payloads(
    *,
    run_label: str,
    status: str,
    gates: Mapping[str, Mapping[str, object]],
    evidence: _ReviewEvidence,
    prerequisite_payload: Mapping[str, object],
) -> dict[str, bytes]:
    findings = [
        {
            "gate": name,
            "passed": gate.get("passed"),
            "detail": gate.get("detail"),
        }
        for name, gate in sorted(gates.items())
    ]
    failures = [row for row in findings if row["passed"] is not True]
    report_lines = [
        f"# Stage 5.2 campaign review: {run_label}",
        "",
        f"Status: `{status}`",
        "",
    ]
    report_lines.extend(
        f"- {'PASS' if row['passed'] is True else 'FAIL'} `{row['gate']}`: {row['detail']}"
        for row in findings
    )
    return {
        "per_run_results": _csv_bytes(
            evidence.per_run_rows,
            fallback_fields=(
                "instance",
                "seed",
                "axis",
                "customer_count",
                "family",
                "budget_seconds",
            ),
        ),
        "family_summary": _csv_bytes(
            _family_summary(evidence.per_run_rows),
            fallback_fields=(
                "family",
                "customer_count",
                "budget_seconds",
                "run_count",
            ),
        ),
        "budget_summary": _csv_bytes(
            _budget_summary(evidence.per_run_rows),
            fallback_fields=("budget_seconds", "run_count"),
        ),
        "anytime_summary": _csv_bytes(
            evidence.checkpoint_rows,
            fallback_fields=(
                "instance",
                "seed",
                "axis_budget_seconds",
                "checkpoint_seconds",
                "objective_key",
            ),
        ),
        "resource_summary": _csv_bytes(
            evidence.resource_rows,
            fallback_fields=(
                "batch_id",
                "configured_workers",
                "per_worker_peak_rss_bytes",
                "aggregate_peak_rss_bytes",
            ),
        ),
        "persistence_summary": _csv_bytes(
            evidence.persistence_rows,
            fallback_fields=(
                "batch_id",
                "solver_seconds",
                "artifact_persistence_seconds",
                "persistence_ratio",
                "maximum_ratio",
            ),
        ),
        "performance_gates": _csv_bytes(
            findings,
            fallback_fields=("gate", "passed", "detail"),
        ),
        "gpu_decision": (
            json.dumps(
                {
                    "schema_version": "stage05.2-gpu-decision-publication-v1",
                    "run_label": run_label,
                    "decision": evidence.selection_lock.get("accelerator_decision"),
                    "selected_backend": evidence.selection_lock.get("selected_backend"),
                    "selected_exact_backend": evidence.selection_lock.get("selected_exact_backend"),
                    "selected_workers": evidence.selection_lock.get("selected_workers"),
                    "native_profile": evidence.selection_lock.get("native_profile"),
                    "native_config_sha256": evidence.selection_lock.get("native_config_sha256"),
                    "accelerator_review_manifest_sha256": evidence.selection_lock.get(
                        "accelerator_review_manifest_sha256"
                    ),
                    "campaign_prerequisite_review_sha256": prerequisite_payload.get(
                        "campaign_prerequisite_review_sha256"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
        "failure_analysis": _csv_bytes(
            failures,
            fallback_fields=("gate", "passed", "detail"),
        ),
        "review_findings": _csv_bytes(
            findings,
            fallback_fields=("gate", "passed", "detail"),
        ),
        "review_report": ("\n".join(report_lines) + "\n").encode("utf-8"),
    }


_REVIEW_FILENAMES = {
    "per_run_results": "per_run_results.csv",
    "family_summary": "family_summary.csv",
    "budget_summary": "budget_summary.csv",
    "anytime_summary": "anytime_summary.csv",
    "resource_summary": "resource_summary.csv",
    "persistence_summary": "persistence_summary.csv",
    "performance_gates": "performance_gates.csv",
    "gpu_decision": "gpu_decision.json",
    "failure_analysis": "failure_analysis.csv",
    "review_findings": "review_findings.csv",
    "review_report": "review_report.md",
}


def _write_fsync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verified_prior_campaign_review_history(
    campaign_dir: Path,
    payload: object,
) -> list[str]:
    if not isinstance(payload, Mapping):
        raise ArtifactIntegrityError("prior campaign review manifest must be an object")
    if (
        payload.get("schema_version") != CAMPAIGN_REVIEW_SCHEMA
        or payload.get("run_label") != campaign_dir.name
        or payload.get("scope") not in {"pilot", "formal"}
    ):
        raise ArtifactIntegrityError("prior campaign review identity is invalid")
    verify_stage052_review_files(campaign_dir, payload)
    raw_manifest_sha256 = _sha256(ArtifactReader(campaign_dir).result.manifest_path)
    if payload.get("raw_manifest_sha256") != raw_manifest_sha256:
        raise ArtifactIntegrityError("prior campaign review is stale for the raw manifest")
    campaign_manifest = campaign_dir / "campaign_manifest.json"
    if campaign_manifest.is_file() and payload.get("raw_campaign_manifest_sha256") != _sha256(
        campaign_manifest
    ):
        raise ArtifactIntegrityError("prior campaign review is stale for the campaign manifest")
    raw_history = payload.get("review_history")
    if (
        not isinstance(raw_history, list)
        or any(not isinstance(item, str) or not _is_sha256(item) for item in raw_history)
        or len(set(raw_history)) != len(raw_history)
    ):
        raise ArtifactIntegrityError("prior campaign review history is invalid")
    previous = payload.get("previous_review_manifest_sha256")
    if (raw_history and previous != raw_history[-1]) or (not raw_history and previous is not None):
        raise ArtifactIntegrityError("prior campaign review lineage is discontinuous")
    for index, digest in enumerate(raw_history):
        history_path = campaign_dir / "review" / "history" / digest / "review_manifest.json"
        if not history_path.is_file() or _sha256(history_path) != digest:
            raise ArtifactIntegrityError("prior campaign review history manifest is missing")
        try:
            historical = json.loads(history_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError(
                "prior campaign review history manifest is invalid"
            ) from error
        expected_prefix = raw_history[:index]
        if (
            not isinstance(historical, Mapping)
            or historical.get("schema_version") != CAMPAIGN_REVIEW_SCHEMA
            or historical.get("run_label") != campaign_dir.name
            or historical.get("review_history") != expected_prefix
            or historical.get("previous_review_manifest_sha256")
            != (expected_prefix[-1] if expected_prefix else None)
            or historical.get("raw_manifest_sha256") != raw_manifest_sha256
        ):
            raise ArtifactIntegrityError("prior campaign review history lineage is invalid")
        verify_stage052_review_files(campaign_dir, historical)
    return list(raw_history)


def _publish_review(
    *,
    campaign_dir: Path,
    manifest: Mapping[str, object],
    payloads: Mapping[str, bytes],
) -> dict[str, Path]:
    review_dir = campaign_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for key in sorted(payloads):
        digest.update(key.encode("utf-8") + b"\0" + payloads[key])
    generation_id = digest.hexdigest()
    generations = review_dir / "generations"
    generations.mkdir(exist_ok=True)
    generation = generations / generation_id
    temporary = generations / f".{generation_id}.{uuid.uuid4().hex}.tmp"
    if generation.exists():
        expected = {_REVIEW_FILENAMES[key] for key in payloads}
        observed = {
            path.name
            for path in generation.iterdir()
            if path.is_file() and not path.name.startswith("._")
        }
        if observed != expected or any(
            (generation / _REVIEW_FILENAMES[key]).read_bytes() != value
            for key, value in payloads.items()
        ):
            raise ArtifactIntegrityError("campaign review generation collision")
    else:
        temporary.mkdir()
        try:
            for key, value in payloads.items():
                _write_fsync(temporary / _REVIEW_FILENAMES[key], value)
            _fsync_directory(temporary)
            os.replace(temporary, generation)
            _fsync_directory(generations)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    publication_files = {
        key: {
            "relative_path": (generation / _REVIEW_FILENAMES[key])
            .relative_to(review_dir)
            .as_posix(),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        for key, value in payloads.items()
    }
    manifest_path = review_dir / "review_manifest.json"
    previous_sha256: str | None = None
    history: list[str] = []
    if manifest_path.is_file():
        previous_bytes = manifest_path.read_bytes()
        previous_sha256 = hashlib.sha256(previous_bytes).hexdigest()
        try:
            previous_payload = json.loads(previous_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError("prior campaign review manifest is invalid") from error
        history.extend(_verified_prior_campaign_review_history(campaign_dir, previous_payload))
        if previous_sha256 not in history:
            history.append(previous_sha256)
        history_dir = review_dir / "history" / previous_sha256
        history_dir.mkdir(parents=True, exist_ok=True)
        history_manifest = history_dir / "review_manifest.json"
        if history_manifest.exists():
            if history_manifest.read_bytes() != previous_bytes:
                raise ArtifactIntegrityError("campaign review history digest collision")
        else:
            _write_fsync(history_manifest, previous_bytes)
            _fsync_directory(history_dir)
            _fsync_directory(history_dir.parent)
    manifest_payload = {
        **manifest,
        "files": {item["relative_path"]: item["sha256"] for item in publication_files.values()},
        "publication_files": publication_files,
        "previous_review_manifest_sha256": previous_sha256,
        "review_history": history,
    }
    if os.environ.get(_REVIEW_EXECUTION_ENV):
        manifest_payload["review_execution_required"] = True
    manifest_bytes = (json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary_manifest = review_dir / f".review_manifest.{uuid.uuid4().hex}.tmp"
    try:
        _write_fsync(temporary_manifest, manifest_bytes)
        os.replace(temporary_manifest, manifest_path)
        _fsync_directory(review_dir)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    outputs = {key: generation / _REVIEW_FILENAMES[key] for key in payloads}
    outputs["review_manifest"] = manifest_path
    return outputs


def review_stage052_campaign(
    *,
    campaign_dir: Path,
    benchmark_dir: Path,
    bks_path: Path,
    scope: str,
    prerequisite_dir: Path,
    locator: StorageRootLocator,
    volume_probe: Callable[[Path], VolumeIdentity] = _default_volume_probe,
) -> dict[str, Path]:
    """Independently replay one G01 pilot or G02 Formal campaign."""

    if scope not in {"pilot", "formal"}:
        raise ValueError("campaign review scope must be pilot or formal")
    run_label = campaign_dir.name
    try:
        gates, evidence, raw_hash, prerequisite = _audit_campaign(
            campaign_dir=campaign_dir,
            benchmark_dir=benchmark_dir,
            bks_path=bks_path,
            scope=scope,
            prerequisite_dir=prerequisite_dir,
            locator=locator,
            volume_probe=volume_probe,
        )
        campaign = load_campaign_manifest(campaign_dir / "campaign_manifest.json")
        storage_aliases = sorted(campaign.storage_roots)
        archive_aliases = sorted(
            evidence.verified_archive_root_aliases
            if scope == "pilot"
            else evidence.archive_root_aliases
        )
        staging_alias = (
            next(iter(evidence.staging_root_aliases))
            if len(evidence.staging_root_aliases) == 1
            else None
        )
    except (
        ArtifactIntegrityError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        gates = {
            "campaign_replay": {
                "passed": False,
                "detail": f"campaign replay failed before completion: {error}",
            }
        }
        evidence = _ReviewEvidence.empty()
        campaign_path = campaign_dir / "campaign_manifest.json"
        raw_hash = _sha256(campaign_path) if campaign_path.is_file() else ""
        prerequisite = {"error": "prerequisite was not verified"}
        storage_aliases = []
        archive_aliases = []
        staging_alias = None
    selection_fields: dict[str, object] = {
        "selected_backend": evidence.selection_lock.get("selected_backend"),
        "selected_exact_backend": evidence.selection_lock.get("selected_exact_backend"),
        "selected_workers": evidence.selection_lock.get("selected_workers"),
        "native_profile": evidence.selection_lock.get("native_profile"),
        "native_kernel_config": evidence.selection_lock.get("native_kernel_config"),
        "native_configuration": evidence.selection_lock.get("native_kernel_config"),
        "accelerator_decision": evidence.selection_lock.get("accelerator_decision"),
        "selection_lock": dict(evidence.selection_lock),
    }
    if scope == "pilot":
        provisional_passed = bool(gates) and all(
            gate.get("passed") is True for gate in gates.values()
        )
        if provisional_passed:
            try:
                from tools.publish_stage052_artifacts import (
                    dry_run_stage052_publication,
                )

                provisional_status = review_status_for_scope("pilot", passed=True)
                provisional_payloads = _review_payloads(
                    run_label=run_label,
                    status=provisional_status,
                    gates=gates,
                    evidence=evidence,
                    prerequisite_payload=prerequisite,
                )
                with tempfile.TemporaryDirectory(
                    prefix="stage05.2-review-publication-dry-run-",
                    dir=campaign_dir,
                ) as temporary:
                    fake_campaign = Path(temporary) / run_label
                    fake_campaign.mkdir()
                    provisional_outputs = _publish_review(
                        campaign_dir=fake_campaign,
                        manifest={
                            "schema_version": CAMPAIGN_REVIEW_SCHEMA,
                            "run_label": run_label,
                            "component": Stage052Component.BENCHMARK.value,
                            "scope": "pilot",
                            "status": provisional_status,
                            "raw_manifest_sha256": (evidence.standard_raw_manifest_sha256),
                            "raw_campaign_manifest_sha256": raw_hash,
                            "persistence_attribution_sha256": (
                                evidence.campaign_persistence_attribution_sha256
                            ),
                            "persistence_attribution_sidecar_sha256": (
                                evidence.campaign_persistence_attribution_sidecar_sha256
                            ),
                            "storage_root_aliases": storage_aliases,
                            "staging_root_alias": staging_alias,
                            "archive_root_aliases_exercised": archive_aliases,
                            "campaign_prerequisite_review_sha256": prerequisite.get(
                                "campaign_prerequisite_review_sha256"
                            ),
                            **selection_fields,
                            "gates": gates,
                        },
                        payloads=provisional_payloads,
                    )
                    dry_run = dry_run_stage052_publication(
                        review_manifest=provisional_outputs["review_manifest"],
                        workspace_root=Path(temporary) / "publication-workspace",
                    )
                dry_run_passed = dry_run.get("status") == "passed"
                dry_run_detail = (
                    "atomic publisher dry run passed without tracked output"
                    if dry_run_passed
                    else f"publisher dry run returned {dry_run}"
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                dry_run_passed = False
                dry_run_detail = f"publisher dry run failed: {error}"
        else:
            dry_run_passed = False
            dry_run_detail = "publisher dry run withheld because an earlier pilot gate failed"
        gates["publication_dry_run"] = {
            "passed": dry_run_passed,
            "detail": dry_run_detail,
        }
    passed = bool(gates) and all(gate.get("passed") is True for gate in gates.values())
    status = review_status_for_scope(scope, passed=passed)
    manifest: dict[str, object] = {
        "schema_version": CAMPAIGN_REVIEW_SCHEMA,
        "run_label": run_label,
        "component": Stage052Component.BENCHMARK.value,
        "scope": scope,
        "status": status,
        "raw_manifest_sha256": evidence.standard_raw_manifest_sha256,
        "raw_campaign_manifest_sha256": raw_hash,
        "persistence_attribution_sha256": (evidence.campaign_persistence_attribution_sha256),
        "persistence_attribution_sidecar_sha256": (
            evidence.campaign_persistence_attribution_sidecar_sha256
        ),
        "storage_root_aliases": storage_aliases,
        "staging_root_alias": staging_alias,
        "archive_root_aliases_exercised": archive_aliases,
        "campaign_prerequisite_review_sha256": prerequisite.get(
            "campaign_prerequisite_review_sha256"
        ),
        **selection_fields,
        "gates": gates,
    }
    payloads = _review_payloads(
        run_label=run_label,
        status=status,
        gates=gates,
        evidence=evidence,
        prerequisite_payload=prerequisite,
    )
    return _publish_review(
        campaign_dir=campaign_dir,
        manifest=manifest,
        payloads=payloads,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--bks-path", type=Path, required=True)
    parser.add_argument("--scope", choices=("pilot", "formal"), required=True)
    parser.add_argument("--prerequisite-dir", type=Path, required=True)
    parser.add_argument("--storage-roots", type=Path, required=True)
    parser.add_argument(
        "--retention-registry",
        type=Path,
        default=Path("experiments/registries/stage05.2_retention_registry.csv"),
    )
    parser.add_argument("--progress-log", type=Path, required=True)
    parser.add_argument("--review-execution-receipt", type=Path, required=True)
    parser.add_argument("--max-aggregate-rss-gib", type=float, default=5.5)
    arguments = parser.parse_args()
    if arguments.max_aggregate_rss_gib <= 0.0:
        parser.error("--max-aggregate-rss-gib must be positive")
    root = repository_root()
    campaign_dir = (
        arguments.campaign_dir
        if arguments.campaign_dir.is_absolute()
        else root / arguments.campaign_dir
    ).resolve()
    registry_path = (root / arguments.retention_registry).resolve()
    locator_path = (root / arguments.storage_roots).resolve()
    if not registry_path.is_file() or not locator_path.is_file():
        parser.error("retention registry and storage-root locator are required")
    locator = StorageRootLocator.from_toml(locator_path)
    for record in load_retention_registry(registry_path):
        if record.run_label != campaign_dir.name:
            continue
        registered_path = locator.resolve(record.archive_root_alias).absolute_path.joinpath(
            *Path(record.archive_relative_path).parts
        )
        if registered_path.resolve() == campaign_dir:
            parser.error(
                "--campaign-dir cannot be immutable archived evidence; archived runs "
                "are read-only prerequisite/replay inputs"
            )
    ordinary_prerequisite = (
        arguments.prerequisite_dir
        if arguments.prerequisite_dir.is_absolute()
        else root / arguments.prerequisite_dir
    )
    prerequisite_dir = (
        ordinary_prerequisite.resolve()
        if ordinary_prerequisite.exists()
        else resolve_retained_run_from_locator(
            arguments.prerequisite_dir.as_posix(),
            registry_path=(root / arguments.retention_registry).resolve(),
            storage_root_locator_path=(root / arguments.storage_roots).resolve(),
        )
    )
    progress_path = arguments.progress_log.resolve()
    if progress_path == campaign_dir or campaign_dir in progress_path.parents:
        parser.error("--progress-log must be outside immutable raw evidence")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress = ReviewProgressLog(progress_path)
    guard = ReviewProcessMemoryGuard(
        limit_bytes=int(arguments.max_aggregate_rss_gib * 1024**3),
        progress=progress,
    )
    previous_progress = os.environ.get("STAGE052_REVIEW_PROGRESS_LOG")
    previous_receipt = os.environ.get(_REVIEW_EXECUTION_ENV)
    os.environ["STAGE052_REVIEW_PROGRESS_LOG"] = str(progress_path)
    os.environ[_REVIEW_EXECUTION_ENV] = str(arguments.review_execution_receipt.resolve())
    progress.emit(
        "campaign_review_start",
        campaign_dir=str(campaign_dir),
        scope=arguments.scope,
    )
    try:
        with guard:
            outputs = review_stage052_campaign(
                campaign_dir=campaign_dir,
                benchmark_dir=arguments.benchmark_dir,
                bks_path=arguments.bks_path,
                scope=arguments.scope,
                prerequisite_dir=prerequisite_dir,
                locator=locator,
            )
    except BaseException as error:
        progress.emit(
            "campaign_review_failed",
            error_type=type(error).__name__,
            error=str(error),
            aggregate_peak_rss_bytes=guard.peak_rss_bytes,
            aggregate_peak_swap_bytes=guard.peak_swap_bytes,
        )
        raise
    finally:
        if previous_progress is None:
            os.environ.pop("STAGE052_REVIEW_PROGRESS_LOG", None)
        else:
            os.environ["STAGE052_REVIEW_PROGRESS_LOG"] = previous_progress
        if previous_receipt is None:
            os.environ.pop(_REVIEW_EXECUTION_ENV, None)
        else:
            os.environ[_REVIEW_EXECUTION_ENV] = previous_receipt
    progress.emit(
        "campaign_review_complete",
        outputs={key: str(path) for key, path in outputs.items()},
        aggregate_peak_rss_bytes=guard.peak_rss_bytes,
        aggregate_peak_swap_bytes=guard.peak_swap_bytes,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
