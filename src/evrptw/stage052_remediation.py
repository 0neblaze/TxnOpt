"""Bounded Stage 5.2 C05 remediation of the withdrawn E03 evidence.

The public entry point verifies the signed E03 ``NOT_READY`` identity, streams
every logical event from the historical v2 bundle into a new v3 child bundle,
and independently streams the child back before reporting semantic equality.
The historical source is never modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from evrptw.artifacts import (
    ARTIFACT_STORAGE_V2,
    SCREENING_DECISIONS_V2,
    SCREENING_DECISIONS_V3,
    ArtifactBundleResult,
    ArtifactBundleWriter,
    ArtifactIntegrityError,
    ArtifactReader,
    ArtifactRunContext,
    ArtifactStorageConfig,
    artifact_context_from_run_label,
)
from evrptw.stage052 import (
    ArtifactPersistenceObservation,
    Stage052Component,
    evaluate_artifact_persistence,
    stage052_contract,
)
from evrptw.stage052_evidence import (
    Stage052PrerequisiteIdentity,
    verify_stage052_evidence_input,
)

E03_EVENT_COUNT = 37_373_993
E03_SHARD_COUNT = 12
E03_SOLVER_ROW_COUNT = 36
E03_IDENTITIES = tuple(
    (instance, seed)
    for instance in ("c101C5", "c101_21", "r101_21", "rc101_21")
    for seed in (2014, 2015, 2016)
)
E03_AXES = ("fixed_work", "fixed_work_control", "wall_clock_30")
_SUMMARY_SCHEMA_VERSION = "stage05.2-remediation-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class Stage052RemediationConfig:
    """Immutable C05 remediation protocol."""

    child_run_label: str = "stage05.2_artifact_streaming_attempt05"
    expected_shard_count: int = E03_SHARD_COUNT
    expected_solver_row_count: int = E03_SOLVER_ROW_COUNT
    expected_event_count: int = E03_EVENT_COUNT
    expected_identities: tuple[tuple[str, int], ...] = E03_IDENTITIES
    expected_axes: tuple[str, ...] = E03_AXES
    batch_size: int = 65_536
    maximum_persistence_ratio: float = 0.30
    per_instance_seed_max_bytes: int = 2 * 1024 * 1024 * 1024
    per_run_max_bytes: int = 32 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        context = artifact_context_from_run_label(self.child_run_label)
        if (
            context.stage_id != "stage05.2"
            or context.component != Stage052Component.ARTIFACT_STREAMING.value
        ):
            raise ValueError("C05 child run label must use artifact_streaming")
        for field, value in (
            ("expected_shard_count", self.expected_shard_count),
            ("expected_solver_row_count", self.expected_solver_row_count),
            ("expected_event_count", self.expected_event_count),
            ("batch_size", self.batch_size),
            ("per_instance_seed_max_bytes", self.per_instance_seed_max_bytes),
            ("per_run_max_bytes", self.per_run_max_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.per_instance_seed_max_bytes > self.per_run_max_bytes:
            raise ValueError("per-instance byte limit cannot exceed the run byte limit")
        if (
            len(self.expected_identities) != self.expected_shard_count
            or len(set(self.expected_identities)) != len(self.expected_identities)
            or any(
                not isinstance(instance, str)
                or not instance
                or isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed <= 0
                for instance, seed in self.expected_identities
            )
        ):
            raise ValueError("expected_identities must match the unique shard scope")
        if (
            not self.expected_axes
            or len(set(self.expected_axes)) != len(self.expected_axes)
            or any(not axis for axis in self.expected_axes)
            or len(self.expected_identities) * len(self.expected_axes)
            != self.expected_solver_row_count
        ):
            raise ValueError("expected_axes must match the exact solver-row scope")
        if (
            not math.isfinite(self.maximum_persistence_ratio)
            or not 0.0 < self.maximum_persistence_ratio < 1.0
        ):
            raise ValueError("maximum_persistence_ratio must be between zero and one")

    def storage_config(self) -> ArtifactStorageConfig:
        return ArtifactStorageConfig(
            storage_policy_version=ARTIFACT_STORAGE_V2,
            screening_schema_version=SCREENING_DECISIONS_V3,
            per_instance_seed_max_bytes=self.per_instance_seed_max_bytes,
            per_run_max_bytes=self.per_run_max_bytes,
        )


@dataclass(frozen=True, slots=True)
class Stage052RemediationResult:
    """Complete, independently replayed identity of one remediation child."""

    source_run_label: str
    child_run_label: str
    source_raw_manifest_sha256: str
    source_review_manifest_sha256: str
    source_semantic_digest: str
    child_semantic_digest: str
    source_event_count: int
    child_event_count: int
    shard_count: int
    verified_solver_seconds: float
    artifact_persistence_seconds: float
    persistence_ratio: float
    persistence_passed: bool
    child_manifest_path: Path
    child_manifest_sha256: str
    child_manifest_sidecar_path: Path
    child_manifest_sidecar_sha256: str

    @property
    def summary_path(self) -> Path:
        """Return the verification summary kept outside the timed child bundle."""

        child_dir = self.child_manifest_path.parents[1]
        return child_dir.parent / f"{self.child_run_label}_remediation_summary.json"

    def to_dict(self) -> dict[str, object]:
        output = asdict(self)
        for field in (
            "child_manifest_path",
            "child_manifest_sidecar_path",
        ):
            output[field] = str(output[field])
        return output


@dataclass(frozen=True, slots=True)
class Stage052SemanticSummary:
    """Canonical logical digest and event cardinality for one full bundle."""

    semantic_digest: str
    event_count: int
    shard_count: int


@dataclass(frozen=True, slots=True)
class _SourceShard:
    instance: str
    seed: int
    route_dictionary_ref: str
    events_ref: str
    screening_checks_ref: str
    screening_decisions_ref: str | None
    diagnostic_ref: str
    raw_ref: str
    solution_ref: str
    trace_ref: str
    environment_ref: str
    failure_ref: str | None


@dataclass(frozen=True, slots=True)
class _RouteDictionary:
    routes: Mapping[str, tuple[str, ...]]
    key_by_id: Mapping[int, str]


@dataclass(frozen=True, slots=True)
class _LogicalShardPayloads:
    raw: Mapping[str, object]
    solution: Mapping[str, object]
    trace: Mapping[str, object]
    environment: Mapping[str, object]
    failure: Mapping[str, object] | None


class _SemanticAccumulator:
    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.count = 0
        self._current_shard: tuple[str, int] | None = None
        self._streamed_counts: dict[tuple[str, int, str], dict[str, int]] = {}

    def begin_shard(self, instance: str, seed: int) -> None:
        self._current_shard = (instance, seed)
        self._digest.update(
            _canonical_json_bytes(
                {"record_type": "shard_identity", "instance": instance, "seed": seed}
            )
        )
        self._digest.update(b"\n")

    def observe_shard_payloads(self, payloads: _LogicalShardPayloads) -> None:
        for section, payload in (
            ("raw", payloads.raw),
            ("solution", payloads.solution),
            ("logical_trace", payloads.trace),
            ("environment", payloads.environment),
            ("failure", payloads.failure),
        ):
            self._observe_record(section, payload)

    def observe_route_dictionary(
        self,
        routes: Mapping[str, tuple[str, ...]],
    ) -> None:
        for route_key in sorted(routes):
            self._observe_record(
                "route_dictionary",
                {
                    "canonical_route_key": route_key,
                    "customer_sequence": list(routes[route_key]),
                },
            )

    def observe_diagnostic(self, row: Mapping[str, object]) -> dict[str, object]:
        physical = dict(row)
        logical = {key: value for key, value in physical.items() if key != "run_label"}
        self._observe_record("diagnostic", logical)
        return physical

    def observe(
        self,
        event: Mapping[str, object],
        *,
        route_key_by_id: Mapping[int, str],
    ) -> dict[str, object]:
        physical = dict(event)
        logical = _canonical_logical_event(physical, route_key_by_id=route_key_by_id)
        self._observe_record("event", logical)
        if self._current_shard is None:
            raise RuntimeError("semantic event observed before shard identity")
        axis = str(logical.get("benchmark_axis", ""))
        identity = (*self._current_shard, axis)
        counts = self._streamed_counts.setdefault(
            identity,
            {
                "events": 0,
                "incremental_propagations": 0,
                "route_evaluations": 0,
                "screening_decisions": 0,
            },
        )
        event_type = str(logical.get("event_type", ""))
        family = (
            "route_evaluations"
            if event_type == "route_evaluation"
            else "screening_decisions"
            if event_type == "screening_decision"
            else "incremental_propagations"
            if event_type == "incremental_propagation"
            else "events"
        )
        counts[family] += 1
        self.count += 1
        return logical

    def _observe_record(self, section: str, payload: object) -> None:
        self._digest.update(_canonical_json_bytes({"record_type": section, "payload": payload}))
        self._digest.update(b"\n")

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def streamed_counts(self, instance: str, seed: int, axis: str) -> dict[str, int]:
        return dict(
            self._streamed_counts.get(
                (instance, seed, axis),
                {
                    "events": 0,
                    "incremental_propagations": 0,
                    "route_evaluations": 0,
                    "screening_decisions": 0,
                },
            )
        )


def remediate_stage052_artifacts(
    *,
    source_dir: Path,
    child_dir: Path,
    config: Stage052RemediationConfig | None = None,
) -> Stage052RemediationResult:
    """Stream the signed E03 v2 bundle to an independently replayed v3 child.

    A semantic or completeness failure is raised immediately after a partial
    manifest, sidecar, and failure record have been retained in ``child_dir``.
    Invalid or stale source evidence is rejected before ``child_dir`` exists.
    """

    selected = config or Stage052RemediationConfig()
    requirement = next(
        item
        for item in stage052_contract(
            Stage052Component.ARTIFACT_STREAMING,
            scope="performance",
        ).prerequisites
        if item.role == "remediation_source"
    )
    source_identity = verify_stage052_evidence_input(source_dir, requirement)
    source_reader = ArtifactReader(source_dir, verify=False)
    _validate_source_storage(source_reader)
    verified_solver_seconds = _verified_solver_seconds(
        source_reader,
        expected_rows=selected.expected_solver_row_count,
        expected_identities=set(selected.expected_identities),
        expected_axes=set(selected.expected_axes),
    )
    shards = _source_shards(source_reader)
    observed_identities = {(shard.instance, shard.seed) for shard in shards}
    if observed_identities != set(selected.expected_identities):
        raise ArtifactIntegrityError(
            "E03 remediation shard identity mismatch: "
            f"expected={sorted(selected.expected_identities)} "
            f"observed={sorted(observed_identities)}"
        )
    if child_dir.exists():
        raise FileExistsError(f"remediation child already exists: {child_dir}")

    writer = ArtifactBundleWriter(
        child_dir,
        ArtifactRunContext(
            "stage05.2",
            Stage052Component.ARTIFACT_STREAMING.value,
            selected.child_run_label,
        ),
        selected.storage_config(),
    )
    source_accumulator = _stream_bundle_semantics(
        source_reader,
        expected_identities={(shard.instance, shard.seed) for shard in shards},
        batch_size=selected.batch_size,
        scratch_root=child_dir,
    )
    child_complete = False
    try:
        writer.write_control(
            metadata={
                "schema_version": _SUMMARY_SCHEMA_VERSION,
                "run_label": selected.child_run_label,
                "component": Stage052Component.ARTIFACT_STREAMING.value,
                "scope": "remediation",
                "repository_revision": source_identity.repository_revision,
                "repository_dirty": False,
                "source_run_label": source_identity.run_label,
                "source_raw_manifest_sha256": source_identity.raw_manifest_sha256,
                "source_review_manifest_sha256": source_identity.review_manifest_sha256,
            }
        )
        persistence_started = time.perf_counter()
        for ordinal, shard in enumerate(shards):
            payloads = _read_logical_shard_payloads(source_reader, shard)
            physical_trace = source_reader.read_json(shard.trace_ref)
            lane_dictionary, operator_dictionary = _trace_dictionaries(physical_trace)
            session = writer.open_v2_shard(
                instance=shard.instance,
                seed=shard.seed,
                shard_ordinal=ordinal,
                worker_identity="remediation-worker-0",
            )
            try:
                if shard.screening_decisions_ref is not None:
                    session.append_transcoded_v2_batches(
                        route_batches=_parquet_batches(
                            source_reader,
                            shard.route_dictionary_ref,
                            selected.batch_size,
                        ),
                        critical_event_batches=_parquet_batches(
                            source_reader,
                            shard.events_ref,
                            selected.batch_size,
                        ),
                        screening_check_batches=_parquet_batches(
                            source_reader,
                            shard.screening_checks_ref,
                            selected.batch_size,
                        ),
                        screening_decision_batches=_parquet_batches(
                            source_reader,
                            shard.screening_decisions_ref,
                            selected.batch_size,
                        ),
                        diagnostic_batches=_parquet_batches(
                            source_reader,
                            shard.diagnostic_ref,
                            selected.batch_size,
                        ),
                        lane_dictionary=lane_dictionary,
                        operator_dictionary=operator_dictionary,
                    )
                else:
                    route_dictionary = _read_route_dictionary(
                        source_reader,
                        shard.route_dictionary_ref,
                        batch_size=selected.batch_size,
                    )
                    session.append(
                        route_dictionary=route_dictionary.routes,
                        critical_events=source_reader.iter_events(
                            shard.events_ref,
                            batch_size=selected.batch_size,
                            scratch_root=child_dir,
                        ),
                        diagnostic_rows=source_reader.iter_parquet_rows(
                            shard.diagnostic_ref,
                            batch_size=selected.batch_size,
                        ),
                    )
                session.finalize(
                    raw_payload=payloads.raw,
                    solution_payload=payloads.solution,
                    trace_payload=_trace_with_streamed_counts(
                        payloads.trace,
                        source_accumulator,
                        instance=shard.instance,
                        seed=shard.seed,
                    ),
                    environment_payload=payloads.environment,
                    failure_payload=payloads.failure,
                )
            except BaseException as error:
                session.abort(error)
                raise
        preliminary = writer.finalize(status="partial", evidence_completeness="partial")
        initial_persistence_seconds = time.perf_counter() - persistence_started
        if source_accumulator.count != selected.expected_event_count:
            raise ArtifactIntegrityError(
                "E03 remediation event count mismatch: "
                f"expected={selected.expected_event_count} "
                f"observed={source_accumulator.count}"
            )

        child_reader = ArtifactReader(preliminary.run_dir)
        child_accumulator = _stream_bundle_semantics(
            child_reader,
            expected_identities={(shard.instance, shard.seed) for shard in shards},
            batch_size=selected.batch_size,
            scratch_root=child_dir,
        )
        if child_accumulator.count != source_accumulator.count:
            raise ArtifactIntegrityError(
                "remediation child event count mismatch: "
                f"source={source_accumulator.count} child={child_accumulator.count}"
            )
        if child_accumulator.hexdigest != source_accumulator.hexdigest:
            raise ArtifactIntegrityError("remediation canonical semantic digest mismatch")
        reverified_source = ArtifactReader(source_dir)
        if _sha256(reverified_source.result.manifest_path) != source_identity.raw_manifest_sha256:
            raise ArtifactIntegrityError("E03 source manifest changed during remediation")
        source_review_path = source_dir / "review" / "review_manifest.json"
        if _sha256(source_review_path) != source_identity.review_manifest_sha256:
            raise ArtifactIntegrityError("E03 source review changed during remediation")

        finalization_started = time.perf_counter()
        bundle = writer.finalize()
        artifact_persistence_seconds = (
            initial_persistence_seconds + time.perf_counter() - finalization_started
        )
        child_complete = True
        persistence = evaluate_artifact_persistence(
            (
                ArtifactPersistenceObservation(
                    solver_seconds=verified_solver_seconds,
                    artifact_persistence_seconds=artifact_persistence_seconds,
                ),
            ),
            maximum_ratio=selected.maximum_persistence_ratio,
        )
        verified_child = ArtifactReader(bundle.run_dir)
        result = _complete_result(
            source_identity=source_identity,
            source_accumulator=source_accumulator,
            child_accumulator=child_accumulator,
            shard_count=len(shards),
            verified_solver_seconds=verified_solver_seconds,
            artifact_persistence_seconds=artifact_persistence_seconds,
            persistence_ratio=persistence.ratio,
            persistence_passed=persistence.passed,
            bundle=bundle,
            verified_child=verified_child,
        )
        _write_json_fsync(
            result.summary_path,
            {
                "schema_version": _SUMMARY_SCHEMA_VERSION,
                "source_run_label": source_identity.run_label,
                "child_run_label": selected.child_run_label,
                "source_raw_manifest_sha256": source_identity.raw_manifest_sha256,
                "source_review_manifest_sha256": source_identity.review_manifest_sha256,
                "source_semantic_digest": source_accumulator.hexdigest,
                "child_semantic_digest": child_accumulator.hexdigest,
                "semantic_digest_scope": [
                    "raw",
                    "solution",
                    "logical_trace",
                    "environment",
                    "route_dictionary",
                    "events",
                    "diagnostic",
                    "failure",
                ],
                "source_event_count": source_accumulator.count,
                "child_event_count": child_accumulator.count,
                "shard_count": len(shards),
                "verified_solver_seconds": verified_solver_seconds,
                "artifact_persistence_seconds": artifact_persistence_seconds,
                "persistence_measurement_scope": (
                    "source_v2_stream_decode_plus_v3_child_write_partial_manifest_"
                    "and_final_control_manifest"
                ),
                "persistence_ratio": persistence.ratio,
                "persistence_passed": persistence.passed,
                "maximum_persistence_ratio": persistence.maximum_ratio,
                "storage_policy_version": ARTIFACT_STORAGE_V2,
                "screening_schema_version": SCREENING_DECISIONS_V3,
                "summary_timing_classification": (
                    "independent_verification_control_not_solver_persistence"
                ),
            },
        )
        return result
    except BaseException as error:
        if not child_complete:
            _retain_partial_child(writer, error)
        raise


def _validate_source_storage(reader: ArtifactReader) -> None:
    policy = reader.manifest.get("storage_policy")
    if not isinstance(policy, Mapping):
        raise ArtifactIntegrityError("E03 source lacks a storage policy")
    declared_schema = policy.get("screening_schema_version")
    artifacts = reader.manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ArtifactIntegrityError("E03 source artifact registry is invalid")
    observed_schemas = {
        str(item.get("artifact_subtype"))
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("artifact_type") == "events"
        and str(item.get("artifact_subtype", "")).startswith("screening_decisions_v")
    }
    schema_matches = (
        observed_schemas <= {SCREENING_DECISIONS_V2}
        if declared_schema == SCREENING_DECISIONS_V2
        else declared_schema is None and observed_schemas == {SCREENING_DECISIONS_V2}
    )
    if (
        reader.manifest.get("storage_policy_version") != ARTIFACT_STORAGE_V2
        or not schema_matches
    ):
        raise ArtifactIntegrityError(
            "E03 remediation requires historical artifact-storage-v2 with screening_decisions_v2"
        )


def _verified_solver_seconds(
    reader: ArtifactReader,
    *,
    expected_rows: int,
    expected_identities: set[tuple[str, int]],
    expected_axes: set[str],
) -> float:
    references = [
        item
        for item in reader.manifest.get("artifacts", ())
        if isinstance(item, Mapping) and item.get("artifact_type") == "per_run_results"
    ]
    if len(references) != 1:
        raise ArtifactIntegrityError("E03 must contain exactly one per-run result table")
    reference = references[0]
    if reference.get("row_count") != expected_rows:
        raise ArtifactIntegrityError("E03 solver row count does not match the manifest")
    path = reader.run_dir / str(reference["relative_path"])
    identities: set[tuple[str, int, str]] = set()
    total = 0.0
    count = 0
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                identity = (str(row["instance"]), int(row["seed"]), str(row["axis"]))
                if identity in identities:
                    raise ArtifactIntegrityError(f"duplicate E03 solver-row identity: {identity}")
                identities.add(identity)
                seconds = float(row["solver_seconds"])
                if not math.isfinite(seconds) or seconds <= 0.0:
                    raise ArtifactIntegrityError("E03 solver seconds must be finite and positive")
                total += seconds
                count += 1
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ArtifactIntegrityError):
            raise
        raise ArtifactIntegrityError("invalid E03 per-run solver evidence") from error
    if count != expected_rows:
        raise ArtifactIntegrityError(
            f"E03 solver row count mismatch: expected={expected_rows} observed={count}"
        )
    expected_solver_identities = {
        (instance, seed, axis) for instance, seed in expected_identities for axis in expected_axes
    }
    if identities != expected_solver_identities:
        raise ArtifactIntegrityError(
            "E03 solver-row identity mismatch: "
            f"expected={sorted(expected_solver_identities)} observed={sorted(identities)}"
        )
    return total


def _source_shards(reader: ArtifactReader) -> tuple[_SourceShard, ...]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for item in reader.manifest.get("artifacts", ()):
        if not isinstance(item, Mapping):
            continue
        relative = Path(str(item.get("relative_path", "")))
        if len(relative.parts) < 3 or relative.parts[0] == "control":
            continue
        try:
            identity = (relative.parts[0], int(relative.parts[1]))
        except ValueError as error:
            raise ArtifactIntegrityError(f"invalid E03 shard artifact path: {relative}") from error
        grouped.setdefault(identity, []).append(item)
    shards: list[_SourceShard] = []
    for (instance, seed), items in sorted(grouped.items()):
        shards.append(
            _SourceShard(
                instance=instance,
                seed=seed,
                route_dictionary_ref=_single_ref(
                    items,
                    artifact_type="route_dictionary",
                    artifact_subtype="canonical_routes",
                ),
                events_ref=_single_ref(
                    items,
                    artifact_type="events",
                    artifact_subtype="critical",
                ),
                screening_checks_ref=_single_ref(
                    items,
                    artifact_type="events",
                    artifact_subtype="screening_checks",
                ),
                screening_decisions_ref=_optional_ref(
                    items,
                    artifact_type="events",
                    artifact_subtype=SCREENING_DECISIONS_V2,
                ),
                diagnostic_ref=_single_ref(
                    items,
                    artifact_type="diagnostic",
                    artifact_subtype="aggregated",
                ),
                raw_ref=_single_ref(items, artifact_type="raw"),
                solution_ref=_single_ref(items, artifact_type="solution"),
                trace_ref=_single_ref(items, artifact_type="trace"),
                environment_ref=_single_ref(items, artifact_type="environment"),
                failure_ref=_optional_ref(items, artifact_type="failure"),
            )
        )
    return tuple(shards)


def _single_ref(
    items: Iterable[Mapping[str, Any]],
    *,
    artifact_type: str,
    artifact_subtype: str | None = None,
) -> str:
    references = [
        str(item["relative_path"])
        for item in items
        if item.get("artifact_type") == artifact_type
        and (artifact_subtype is None or item.get("artifact_subtype") == artifact_subtype)
    ]
    if len(references) != 1:
        raise ArtifactIntegrityError(
            f"E03 shard artifact identity mismatch for {artifact_type}/{artifact_subtype or '*'}"
        )
    return references[0]


def _optional_ref(
    items: Iterable[Mapping[str, Any]],
    *,
    artifact_type: str,
    artifact_subtype: str | None = None,
) -> str | None:
    references = [
        str(item["relative_path"])
        for item in items
        if item.get("artifact_type") == artifact_type
        and (artifact_subtype is None or item.get("artifact_subtype") == artifact_subtype)
    ]
    if len(references) > 1:
        raise ArtifactIntegrityError(f"E03 shard contains duplicate {artifact_type} artifacts")
    return references[0] if references else None


def _parquet_batches(
    reader: ArtifactReader,
    relative_path: str,
    batch_size: int,
) -> Iterable[Any]:
    yield from pq.ParquetFile(reader.run_dir / relative_path).iter_batches(
        batch_size=batch_size
    )


def _trace_dictionaries(
    trace: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    lane_dictionary = trace.get("lane_dictionary")
    operator_dictionary = trace.get("operator_dictionary")
    if not isinstance(lane_dictionary, Mapping) or not isinstance(
        operator_dictionary, Mapping
    ):
        raise ArtifactIntegrityError("old-v2 trace dictionaries are invalid")
    return lane_dictionary, operator_dictionary


def _read_route_dictionary(
    reader: ArtifactReader,
    relative_path: str,
    *,
    batch_size: int,
) -> _RouteDictionary:
    routes: dict[str, tuple[str, ...]] = {}
    key_by_id: dict[int, str] = {}
    for row in reader.iter_parquet_rows(relative_path, batch_size=batch_size):
        route_id = row.get("route_id")
        route_key = row.get("canonical_route_key")
        sequence = row.get("customer_sequence")
        if (
            isinstance(route_id, bool)
            or not isinstance(route_id, int)
            or not isinstance(route_key, str)
            or not isinstance(sequence, list)
        ):
            raise ArtifactIntegrityError("invalid E03 route dictionary row")
        if route_key in routes or route_id in key_by_id:
            raise ArtifactIntegrityError("duplicate E03 route dictionary identity")
        routes[route_key] = tuple(str(customer) for customer in sequence)
        key_by_id[route_id] = route_key
    return _RouteDictionary(routes, key_by_id)


def _read_logical_shard_payloads(
    reader: ArtifactReader,
    shard: _SourceShard,
) -> _LogicalShardPayloads:
    return _LogicalShardPayloads(
        raw=reader.read_json(shard.raw_ref),
        solution=reader.read_json(shard.solution_ref),
        trace=_logical_trace_payload(reader.read_json(shard.trace_ref)),
        environment=reader.read_json(shard.environment_ref),
        failure=(reader.read_json(shard.failure_ref) if shard.failure_ref is not None else None),
    )


def _observed_events(
    events: Iterable[Mapping[str, object]],
    accumulator: _SemanticAccumulator,
    *,
    route_key_by_id: Mapping[int, str],
) -> Iterable[dict[str, object]]:
    for event in events:
        yield accumulator.observe(event, route_key_by_id=route_key_by_id)


def _observed_diagnostics(
    rows: Iterable[Mapping[str, object]],
    accumulator: _SemanticAccumulator,
) -> Iterable[dict[str, object]]:
    for row in rows:
        yield accumulator.observe_diagnostic(row)


def replay_stage052_bundle_semantics(
    run_dir: Path,
    *,
    batch_size: int = 65_536,
    scratch_root: Path | None = None,
) -> Stage052SemanticSummary:
    """Independently stream every logical shard field in a verified bundle."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    reader = ArtifactReader(run_dir)
    accumulator = _stream_bundle_semantics(
        reader,
        expected_identities=None,
        batch_size=batch_size,
        scratch_root=scratch_root,
    )
    return Stage052SemanticSummary(
        semantic_digest=accumulator.hexdigest,
        event_count=accumulator.count,
        shard_count=len(_source_shards(reader)),
    )


def _stream_bundle_semantics(
    reader: ArtifactReader,
    *,
    expected_identities: set[tuple[str, int]] | None,
    batch_size: int,
    scratch_root: Path | None = None,
) -> _SemanticAccumulator:
    accumulator = _SemanticAccumulator()
    shards = _source_shards(reader)
    observed_identities = {(shard.instance, shard.seed) for shard in shards}
    if expected_identities is not None and observed_identities != expected_identities:
        raise ArtifactIntegrityError(
            "remediation child shard identity mismatch: "
            f"expected={sorted(expected_identities)} observed={sorted(observed_identities)}"
        )
    for shard in shards:
        accumulator.begin_shard(shard.instance, shard.seed)
        accumulator.observe_shard_payloads(_read_logical_shard_payloads(reader, shard))
        route_dictionary = _read_route_dictionary(
            reader,
            shard.route_dictionary_ref,
            batch_size=batch_size,
        )
        accumulator.observe_route_dictionary(route_dictionary.routes)
        for event in reader.iter_events(
            shard.events_ref,
            batch_size=batch_size,
            scratch_root=scratch_root,
        ):
            accumulator.observe(event, route_key_by_id=route_dictionary.key_by_id)
        for diagnostic in reader.iter_parquet_rows(
            shard.diagnostic_ref,
            batch_size=batch_size,
        ):
            accumulator.observe_diagnostic(diagnostic)
    return accumulator


def _canonical_logical_event(
    event: Mapping[str, object],
    *,
    route_key_by_id: Mapping[int, str],
) -> dict[str, object]:
    logical = dict(event)
    logical.pop("event_id", None)
    for physical, canonical in (
        ("route_id", "route_key"),
        ("base_route_id", "base_route_key"),
        ("candidate_route_id", "candidate_route_key"),
    ):
        raw_id = logical.pop(physical, None)
        if raw_id is not None:
            logical[canonical] = _resolve_route_key(raw_id, route_key_by_id)
    for physical, canonical in (
        ("route_ids", "route_keys"),
        ("current_route_ids", "current_route_keys"),
        ("candidate_route_ids", "candidate_route_keys"),
    ):
        raw_ids = logical.pop(physical, None)
        if raw_ids is not None:
            if not isinstance(raw_ids, list):
                raise ArtifactIntegrityError(f"event {physical} must be a list")
            logical[canonical] = [
                _resolve_route_key(route_id, route_key_by_id) for route_id in raw_ids
            ]
    return logical


def _resolve_route_key(value: object, route_key_by_id: Mapping[int, str]) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactIntegrityError("event route ID must be an integer")
    try:
        return route_key_by_id[value]
    except KeyError as error:
        raise ArtifactIntegrityError(f"event refers to unknown route ID: {value}") from error


def _logical_trace_payload(payload: Mapping[str, object]) -> dict[str, object]:
    physical_fields = {
        "trace_storage_version",
        "route_dictionary_ref",
        "events_ref",
        "screening_checks_ref",
        "screening_decisions_ref",
        "screening_definitions_ref",
        "screening_occurrences_ref",
        "diagnostic_ref",
        "lane_dictionary",
        "operator_dictionary",
        "schema_fingerprints",
        "screening_schema_version",
        "event_identity",
    }
    logical = {key: value for key, value in payload.items() if key not in physical_fields}
    axes = logical.get("axes")
    if isinstance(axes, Mapping):
        logical["axes"] = {
            str(axis): {
                key: value
                for key, value in axis_payload.items()
                if key != "streamed_record_counts"
            }
            if isinstance(axis_payload, Mapping)
            else axis_payload
            for axis, axis_payload in axes.items()
        }
    return logical


def _trace_with_streamed_counts(
    trace: Mapping[str, object],
    accumulator: _SemanticAccumulator,
    *,
    instance: str,
    seed: int,
) -> dict[str, object]:
    output = dict(trace)
    axes = trace.get("axes")
    if not isinstance(axes, Mapping):
        raise ArtifactIntegrityError("remediation trace axes are missing")
    output["axes"] = {
        str(axis): {
            **dict(axis_payload),
            "streamed_record_counts": accumulator.streamed_counts(
                instance,
                seed,
                str(axis),
            ),
        }
        if isinstance(axis_payload, Mapping)
        else axis_payload
        for axis, axis_payload in axes.items()
    }
    return output


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError("event is not canonical JSON data") from error


def _write_json_fsync(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _retain_partial_child(
    writer: ArtifactBundleWriter,
    error: BaseException,
) -> None:
    failure_path = (
        writer.run_dir / "control" / f"{writer.context.run_label}_remediation_failure.json"
    )
    try:
        _write_json_fsync(
            failure_path,
            {
                "schema_version": _SUMMARY_SCHEMA_VERSION,
                "run_label": writer.context.run_label,
                "status": "partial",
                "evidence_completeness": "partial",
                "error_type": type(error).__name__,
                "failure_reason": str(error),
            },
        )
        writer.record_existing_file(
            failure_path,
            artifact_type="failure",
            retention_class="critical",
            storage_format="json_control",
        )
        writer.finalize(status="partial", evidence_completeness="partial")
    except BaseException as retention_error:
        raise BaseExceptionGroup(
            "remediation failed and partial evidence retention also failed",
            [error, retention_error],
        ) from error


def _complete_result(
    *,
    source_identity: Stage052PrerequisiteIdentity,
    source_accumulator: _SemanticAccumulator,
    child_accumulator: _SemanticAccumulator,
    shard_count: int,
    verified_solver_seconds: float,
    artifact_persistence_seconds: float,
    persistence_ratio: float,
    persistence_passed: bool,
    bundle: ArtifactBundleResult,
    verified_child: ArtifactReader,
) -> Stage052RemediationResult:
    manifest_path = verified_child.result.manifest_path
    sidecar_path = bundle.manifest_sidecar_path
    manifest_sha256 = _sha256(manifest_path)
    if sidecar_path.read_text(encoding="utf-8").strip() != manifest_sha256:
        raise ArtifactIntegrityError("remediation child manifest sidecar mismatch")
    return Stage052RemediationResult(
        source_run_label=str(source_identity.run_label),
        child_run_label=str(verified_child.manifest["run_label"]),
        source_raw_manifest_sha256=str(source_identity.raw_manifest_sha256),
        source_review_manifest_sha256=str(source_identity.review_manifest_sha256),
        source_semantic_digest=source_accumulator.hexdigest,
        child_semantic_digest=child_accumulator.hexdigest,
        source_event_count=source_accumulator.count,
        child_event_count=child_accumulator.count,
        shard_count=shard_count,
        verified_solver_seconds=verified_solver_seconds,
        artifact_persistence_seconds=artifact_persistence_seconds,
        persistence_ratio=persistence_ratio,
        persistence_passed=persistence_passed,
        child_manifest_path=manifest_path,
        child_manifest_sha256=manifest_sha256,
        child_manifest_sidecar_path=sidecar_path,
        child_manifest_sidecar_sha256=_sha256(sidecar_path),
    )
