"""Fail-fast storage governance for experiment runs and retained evidence.

The public interface deliberately keeps capacity planning, retention, resolution,
and rebuildable-asset maintenance behind one module.  Callers provide immutable
requests; this module owns volume verification, durable observations, ledgers,
and filesystem transactions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import tomllib
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final, cast

from evrptw.stage052_campaign import StorageRoot, StorageRootLocator, VolumeIdentity

GIB: Final = 1024**3
POLICY_SCHEMA_VERSION: Final = "experiment-storage-governance-v1"
_RUN_LABEL: Final = re.compile(
    r"^stage(?P<major>0[0-8])(?:\.(?P<minor>[0-9]+))?_[a-z0-9_]+_"
    r"(?:attempt|rerun)[0-9]{2}$"
)
_SHA256: Final = re.compile(r"[0-9a-f]{64}")


class StorageGovernanceError(RuntimeError):
    """Storage governance could not prove that an operation was safe."""


class StorageCapacityError(StorageGovernanceError):
    """A start request cannot preserve every configured capacity reserve."""

    def __init__(self, *, observation: dict[str, object]) -> None:
        super().__init__(
            "experiment capacity stop gate rejected the run: deficits="
            f"{observation.get('deficits_by_alias', {})}, measurement_errors="
            f"{observation.get('measurement_errors_by_alias', {})}, identity_errors="
            f"{observation.get('identity_errors', [])}"
        )
        self.observation = observation


class RetentionClass(StrEnum):
    """Audited v2 disposition for one immutable run generation."""

    ACCEPTED_FULL = "accepted_full"
    UNIQUE_FAILURE_FULL = "unique_failure_full"
    DUPLICATE_FAILURE_REDUCED = "duplicate_failure_reduced"
    REBUILDABLE = "rebuildable"
    UNKNOWN_FULL = "unknown_full"

    @property
    def is_full(self) -> bool:
        return self in {
            RetentionClass.ACCEPTED_FULL,
            RetentionClass.UNIQUE_FAILURE_FULL,
            RetentionClass.UNKNOWN_FULL,
        }


@dataclass(frozen=True, slots=True)
class GovernancePolicy:
    """Hard reserves shared by every new Stage 0--8 experiment."""

    schema_version: str = POLICY_SCHEMA_VERSION
    archive_reserve_bytes: int = 0
    host_reserve_bytes: int = 0
    staging_safety_reserve_bytes: int = 50 * GIB
    stage052_active_workspace_floor_bytes: int = 32 * GIB
    maintenance_allowlist: tuple[tuple[str, str], ...] = (
        (".cache", "cache"),
        (".venv", "venv"),
        ("build", "build"),
        (".temporary-spool", "temporary_spool"),
    )

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_SCHEMA_VERSION:
            raise ValueError(f"unsupported storage governance policy: {self.schema_version}")
        nonnegative_values = (
            self.archive_reserve_bytes,
            self.host_reserve_bytes,
        )
        positive_values = (
            self.staging_safety_reserve_bytes,
            self.stage052_active_workspace_floor_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in nonnegative_values
        ) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in positive_values
        ):
            raise ValueError(
                "archive and host reserves must be non-negative integers; "
                "staging safety and active workspace reserves must be positive integers"
            )
        allowed_kinds = {"cache", "venv", "build", "temporary_spool"}
        for relative_path, kind in self.maintenance_allowlist:
            relative = PurePosixPath(relative_path)
            if (
                relative.is_absolute()
                or not relative.parts
                or str(relative) == "."
                or ".." in relative.parts
                or kind not in allowed_kinds
            ):
                raise ValueError(
                    "storage governance maintenance allowlist is invalid"
                )
        if len({path for path, _kind in self.maintenance_allowlist}) != len(
            self.maintenance_allowlist
        ):
            raise ValueError(
                "storage governance maintenance allowlist paths must be unique"
            )

    @classmethod
    def from_toml(cls, path: Path) -> GovernancePolicy:
        try:
            with path.open("rb") as handle:
                payload = tomllib.load(handle)
            schema_version = payload["schema_version"]
            capacity = payload["capacity"]
            maintenance = payload["maintenance"]
        except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
            raise ValueError(f"cannot read storage governance policy: {path}") from error
        if (
            not isinstance(schema_version, str)
            or not isinstance(capacity, dict)
            or not isinstance(maintenance, dict)
        ):
            raise ValueError("storage governance policy has invalid top-level fields")
        required = {
            "archive_reserve_bytes",
            "host_reserve_bytes",
            "staging_safety_reserve_bytes",
            "stage052_active_workspace_floor_bytes",
        }
        if not required.issubset(capacity):
            raise ValueError("storage governance capacity policy is incomplete")
        values: dict[str, int] = {}
        for field in required:
            value = capacity[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"storage governance capacity {field} is invalid")
            values[field] = value
        raw_allowlist = maintenance.get("exact_allowlist")
        if not isinstance(raw_allowlist, list) or not raw_allowlist:
            raise ValueError(
                "storage governance maintenance exact_allowlist is missing"
            )
        allowlist: list[tuple[str, str]] = []
        for record in raw_allowlist:
            if (
                not isinstance(record, dict)
                or set(record) != {"relative_path", "kind"}
                or not isinstance(record.get("relative_path"), str)
                or not isinstance(record.get("kind"), str)
            ):
                raise ValueError(
                    "storage governance maintenance allowlist record is invalid"
                )
            allowlist.append((record["relative_path"], record["kind"]))
        return cls(
            schema_version=schema_version,
            maintenance_allowlist=tuple(allowlist),
            **values,
        )


@dataclass(frozen=True, slots=True)
class StartRequest:
    """One complete, replayable experiment-capacity request."""

    stage_id: str
    run_label: str
    run_dir: Path
    staging_root_alias: str
    host_root_alias: str
    archive_root_alias: str
    planned_archive_bytes: int
    max_active_workspace_bytes: int
    projected_host_growth_bytes: int
    stage_plan_sha256: str

    def __post_init__(self) -> None:
        match = _RUN_LABEL.fullmatch(self.run_label)
        if match is None:
            raise ValueError(f"non-canonical governed run label: {self.run_label}")
        expected_stage = f"stage{match.group('major')}"
        if match.group("minor") is not None:
            expected_stage += f".{match.group('minor')}"
        if self.stage_id != expected_stage:
            raise ValueError("governed run label and stage_id disagree")
        if self.run_dir.name != self.run_label:
            raise ValueError("governed run directory must end with run_label")
        if not self.run_dir.is_absolute():
            raise ValueError("governed run directory must be absolute")
        aliases = (
            self.staging_root_alias,
            self.host_root_alias,
            self.archive_root_alias,
        )
        if any(re.fullmatch(r"[a-z][a-z0-9_]*", alias) is None for alias in aliases):
            raise ValueError("storage aliases must be canonical")
        if len(set(aliases)) != len(aliases):
            raise ValueError("staging, host, and archive aliases must be distinct")
        if (
            isinstance(self.planned_archive_bytes, bool)
            or not isinstance(self.planned_archive_bytes, int)
            or self.planned_archive_bytes < 0
        ):
            raise ValueError(
                "planned_archive_bytes must be a non-negative integer"
            )
        positive_sizes = (
            self.max_active_workspace_bytes,
            self.projected_host_growth_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in positive_sizes
        ):
            raise ValueError(
                "active workspace and host growth must be positive integers"
            )
        if _SHA256.fullmatch(self.stage_plan_sha256) is None:
            raise ValueError("stage_plan_sha256 must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class StartPermit:
    """Durable capacity reservation returned before a run directory is created."""

    run_label: str
    stage_plan_sha256: str
    observation_sha256: str
    observation_path: Path
    maintenance_audit_sha256: str
    maintenance_audit_path: Path
    permit_path: Path


@dataclass(frozen=True, slots=True)
class AdjudicationRecord:
    """Explicit root-cause decision; similarity of error text is never enough."""

    run_label: str
    root_cause_id: str
    canonical_representative_run_label: str
    failure_location: str
    evidence_references: tuple[str, ...]
    adjudication_sha256: str

    def __post_init__(self) -> None:
        if _RUN_LABEL.fullmatch(self.run_label) is None:
            raise ValueError("adjudication run_label is invalid")
        if _RUN_LABEL.fullmatch(self.canonical_representative_run_label) is None:
            raise ValueError("adjudication canonical representative is invalid")
        if self.run_label == self.canonical_representative_run_label:
            raise ValueError("duplicate failure cannot represent itself")
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.root_cause_id) is None:
            raise ValueError("root_cause_id must be canonical")
        if not self.failure_location or not self.evidence_references:
            raise ValueError("adjudication requires failure location and evidence")
        if self.adjudication_sha256 and _SHA256.fullmatch(
            self.adjudication_sha256
        ) is None:
            raise ValueError("adjudication_sha256 is invalid")

    @property
    def is_signed(self) -> bool:
        return _SHA256.fullmatch(self.adjudication_sha256) is not None


@dataclass(frozen=True, slots=True)
class RetentionSegment:
    """One physical source segment contributing to a logical run tree."""

    segment_id: str
    source_path: Path
    logical_prefix: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.segment_id) is None:
            raise ValueError("retention segment_id must be canonical")
        if not self.source_path.is_absolute():
            raise ValueError("retention segment source_path must be absolute")
        prefix = PurePosixPath(self.logical_prefix)
        if prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError("retention segment logical_prefix is unsafe")


@dataclass(frozen=True, slots=True)
class ExpectedSegmentTree:
    """Signed-inventory tree identity required from one copied segment."""

    segment_id: str
    file_count: int
    byte_count: int
    tree_sha256: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.segment_id) is None:
            raise ValueError("expected segment ID must be canonical")
        if (
            isinstance(self.file_count, bool)
            or not isinstance(self.file_count, int)
            or self.file_count < 0
            or isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or self.byte_count < 0
            or _SHA256.fullmatch(self.tree_sha256) is None
        ):
            raise ValueError("expected segment tree identity is invalid")


@dataclass(frozen=True, slots=True)
class RetentionRequest:
    """One immutable retention-generation transaction."""

    run_label: str
    generation: int
    retention_class: RetentionClass
    archive_root_alias: str
    archive_relative_path: str
    segments: tuple[RetentionSegment, ...]
    root_cause_id: str = ""
    adjudication: AdjudicationRecord | None = None
    adjudication_path: Path | None = None
    keep_relative_paths: tuple[str, ...] = ()
    replay_verifier: Callable[[Path], Path] | None = None
    copy_workers: int = 1
    copy_backend: str = "python_stream"
    expected_segment_trees: tuple[ExpectedSegmentTree, ...] = ()

    def __post_init__(self) -> None:
        if _RUN_LABEL.fullmatch(self.run_label) is None:
            raise ValueError("retention request run_label is invalid")
        if isinstance(self.generation, bool) or self.generation <= 0:
            raise ValueError("retention generation must be positive")
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.archive_root_alias) is None:
            raise ValueError("retention archive alias is invalid")
        relative = PurePosixPath(self.archive_relative_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("retention archive path is unsafe")
        if not self.segments:
            raise ValueError("retention request requires at least one source segment")
        if (
            isinstance(self.copy_workers, bool)
            or not isinstance(self.copy_workers, int)
            or not 1 <= self.copy_workers <= 32
        ):
            raise ValueError("retention copy_workers must be between 1 and 32")
        if self.copy_backend not in {"python_stream", "auto_native"}:
            raise ValueError("retention copy_backend is unsupported")
        segment_ids = tuple(segment.segment_id for segment in self.segments)
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("retention segment IDs must be unique")
        expected_ids = tuple(
            expected.segment_id for expected in self.expected_segment_trees
        )
        if self.expected_segment_trees and (
            len(set(expected_ids)) != len(expected_ids)
            or set(expected_ids) != set(segment_ids)
        ):
            raise ValueError(
                "expected segment trees must bind every source segment exactly once"
            )
        if self.adjudication is not None and self.adjudication.run_label != self.run_label:
            raise ValueError("adjudication and retention run_label disagree")
        if self.adjudication_path is not None and not self.adjudication_path.is_absolute():
            raise ValueError("adjudication_path must be absolute")
        if self.root_cause_id and re.fullmatch(
            r"[a-z0-9][a-z0-9._-]*", self.root_cause_id
        ) is None:
            raise ValueError("retention root_cause_id is invalid")
        if (
            self.retention_class == RetentionClass.UNIQUE_FAILURE_FULL
            and not self.root_cause_id
        ):
            raise ValueError("unique failure retention requires root_cause_id")
        if (
            self.retention_class
            in {
                RetentionClass.ACCEPTED_FULL,
                RetentionClass.UNIQUE_FAILURE_FULL,
            }
            and self.replay_verifier is None
        ):
            raise ValueError(
                "accepted and unique-failure full retention require independent replay"
            )
        for item in self.keep_relative_paths:
            path = PurePosixPath(item)
            if path.is_absolute() or not path.parts or ".." in path.parts:
                raise ValueError("retention keep path is unsafe")


@dataclass(frozen=True, slots=True)
class RetentionReceipt:
    """Verified outcome of one retention generation."""

    run_label: str
    generation: int
    retention_class: RetentionClass
    archive_path: Path
    tree_sha256: str
    kept_file_count: int
    kept_bytes: int
    omitted_file_count: int
    omitted_bytes: int
    audit_only: bool
    replay_receipt_sha256: str
    maintenance_audit_sha256: str
    maintenance_audit_path: Path
    registry_sha256: str


class RebuildableKind(StrEnum):
    """Exact classes eligible for maintenance after independent proof."""

    CACHE = "cache"
    VENV = "venv"
    BUILD = "build"
    TEMPORARY_SPOOL = "temporary_spool"


@dataclass(frozen=True, slots=True)
class RebuildableAsset:
    """One exact allowlisted asset and its pre-maintenance identity."""

    path: Path
    kind: RebuildableKind
    expected_tree_sha256: str
    keeper_references: tuple[str, ...] = ()
    rebuild_proof_path: Path | None = None
    rebuild_proof_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.path.is_absolute():
            raise ValueError("rebuildable asset path must be absolute")
        if _SHA256.fullmatch(self.expected_tree_sha256) is None:
            raise ValueError("rebuildable asset tree SHA-256 is invalid")
        if self.rebuild_proof_sha256 and _SHA256.fullmatch(
            self.rebuild_proof_sha256
        ) is None:
            raise ValueError("rebuild proof SHA-256 is invalid")


@dataclass(frozen=True, slots=True)
class SweepRequest:
    """A dry-run-first rebuildable maintenance request."""

    allowed_roots: tuple[Path, ...]
    assets: tuple[RebuildableAsset, ...]
    retention_days: int = 14
    apply: bool = False
    approved_dry_run_path: Path | None = None
    approved_dry_run_sha256: str = ""
    confirmation_receipt_path: Path | None = None
    confirmation_receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.allowed_roots or any(
            not root.is_absolute() for root in self.allowed_roots
        ):
            raise ValueError("sweep requires absolute allowed roots")
        if self.retention_days < 0:
            raise ValueError("retention_days cannot be negative")
        if self.approved_dry_run_path is not None and not (
            self.approved_dry_run_path.is_absolute()
            and _SHA256.fullmatch(self.approved_dry_run_sha256)
        ):
            raise ValueError("approved rebuildable dry-run identity is invalid")
        if not self.apply and (
            self.approved_dry_run_path is not None
            or self.approved_dry_run_sha256
            or self.confirmation_receipt_path is not None
            or self.confirmation_receipt_sha256
        ):
            raise ValueError("a dry run cannot carry an execution approval")
        if self.confirmation_receipt_path is not None and not (
            self.confirmation_receipt_path.is_absolute()
            and _SHA256.fullmatch(self.confirmation_receipt_sha256)
        ):
            raise ValueError("cleanup confirmation receipt identity is invalid")


@dataclass(frozen=True, slots=True)
class SweepReceipt:
    """Auditable maintenance decision; retained assets include a reason."""

    candidates: tuple[Path, ...]
    retained: tuple[tuple[Path, str], ...]
    candidate_bytes: int
    applied: bool
    receipt_path: Path
    receipt_sha256: str


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_signed_json(path: Path, payload: object) -> str:
    data = _canonical_json(payload)
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temporary = sidecar.with_name(f".{sidecar.name}.tmp")
    sidecar_temporary.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    sidecar_temporary.replace(sidecar)
    return digest


def write_adjudication_record(
    path: Path,
    *,
    run_label: str,
    root_cause_id: str,
    canonical_representative_run_label: str,
    failure_location: str,
    evidence_references: tuple[str, ...],
) -> AdjudicationRecord:
    """Write one explicit signed root-cause adjudication."""

    if not path.is_absolute():
        raise ValueError("adjudication path must be absolute")
    draft = AdjudicationRecord(
        run_label=run_label,
        root_cause_id=root_cause_id,
        canonical_representative_run_label=canonical_representative_run_label,
        failure_location=failure_location,
        evidence_references=evidence_references,
        adjudication_sha256="",
    )
    payload = {
        "schema_version": "experiment-root-cause-adjudication-v1",
        "run_label": draft.run_label,
        "root_cause_id": draft.root_cause_id,
        "canonical_representative_run_label": (
            draft.canonical_representative_run_label
        ),
        "failure_location": draft.failure_location,
        "evidence_references": list(draft.evidence_references),
    }
    digest = _write_signed_json(path, payload)
    return AdjudicationRecord(
        run_label=draft.run_label,
        root_cause_id=draft.root_cause_id,
        canonical_representative_run_label=draft.canonical_representative_run_label,
        failure_location=draft.failure_location,
        evidence_references=draft.evidence_references,
        adjudication_sha256=digest,
    )


def write_migration_dry_run(path: Path, payload: Mapping[str, object]) -> str:
    """Persist a signed, non-destructive migration projection."""

    if payload.get("schema_version") != "experiment-storage-migration-dry-run-v1":
        raise ValueError("migration dry-run schema is invalid")
    if payload.get("source_deletion_authorized") is not False:
        raise ValueError("a dry run cannot authorize source deletion")
    if payload.get("retention_default") != RetentionClass.UNKNOWN_FULL.value:
        raise ValueError("unadjudicated migration input must default to unknown_full")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("migration dry run requires source observations")
    planned = payload.get("planned_mappings")
    if planned is not None:
        if not isinstance(planned, list) or any(
            not isinstance(record, dict) for record in planned
        ):
            raise ValueError("migration dry-run planned mappings are invalid")
        _verify_migration_dry_run_projection(
            payload,
            [dict(record) for record in planned],
        )
    return _write_signed_json(path, dict(payload))


def write_retention_replay_receipt(
    path: Path,
    *,
    run_label: str,
    generation: int,
    archive_path: Path,
    verifier_identity_sha256: str,
    validator_replay_passed: bool,
    objective_replay_passed: bool,
    raw_review_replay_passed: bool,
) -> str:
    """Seal the independently produced full-retention replay result."""

    if (
        _RUN_LABEL.fullmatch(run_label) is None
        or isinstance(generation, bool)
        or generation <= 0
        or not archive_path.is_absolute()
        or _SHA256.fullmatch(verifier_identity_sha256) is None
    ):
        raise ValueError("retention replay identity is invalid")
    checks = {
        "validator_replay_passed": validator_replay_passed,
        "objective_replay_passed": objective_replay_passed,
        "raw_review_replay_passed": raw_review_replay_passed,
    }
    if not all(value is True for value in checks.values()):
        raise StorageGovernanceError(
            f"full-retention replay did not pass every check: {run_label}"
        )
    file_count, byte_count, tree_sha256 = _tree_identity(archive_path)
    return _write_signed_json(
        path,
        {
            "schema_version": "experiment-retention-replay-v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_label": run_label,
            "generation": generation,
            "archive_tree_sha256": tree_sha256,
            "archive_file_count": file_count,
            "archive_byte_count": byte_count,
            "verifier_identity_sha256": verifier_identity_sha256,
            **checks,
            "status": "passed",
        },
    )


def load_capacity_observation(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    """Load a signed capacity observation by its permit-bound identity."""

    if _SHA256.fullmatch(expected_sha256) is None:
        raise StorageGovernanceError("capacity observation SHA-256 is invalid")
    payload = _load_signed_json(path)
    if _file_sha256(path) != expected_sha256:
        raise StorageGovernanceError("capacity observation identity mismatch")
    return payload


def write_cleanup_confirmation_receipt(
    path: Path,
    *,
    dry_run_path: Path,
    dry_run_sha256: str,
    confirmation_text: str,
) -> str:
    """Bind a literal user confirmation to one exact rebuildable deletion list."""

    if confirmation_text != "确认":
        raise StorageGovernanceError(
            "rebuildable cleanup requires the literal confirmation text 确认"
        )
    if _SHA256.fullmatch(dry_run_sha256) is None:
        raise ValueError("cleanup dry-run SHA-256 is invalid")
    dry_run = _load_signed_json(dry_run_path)
    if (
        _file_sha256(dry_run_path) != dry_run_sha256
        or dry_run.get("schema_version") != "experiment-rebuildable-audit-v1"
        or dry_run.get("applied") is not False
    ):
        raise StorageGovernanceError("cleanup confirmation dry run is invalid")
    return _write_signed_json(
        path,
        {
            "schema_version": "experiment-cleanup-confirmation-v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "confirmation_text": confirmation_text,
            "approved_dry_run_sha256": dry_run_sha256,
            "request_identity_sha256": dry_run.get(
                "request_identity_sha256"
            ),
            "candidate_paths": dry_run.get("candidate_paths"),
            "candidate_bytes": dry_run.get("candidate_bytes"),
        },
    )


def _verify_migration_dry_run_projection(
    dry_run: Mapping[str, object],
    planned_mappings: list[dict[str, object]],
) -> None:
    if dry_run.get("planned_mappings") != planned_mappings:
        raise StorageGovernanceError(
            "storage migration mappings differ from the signed dry run"
        )
    upper_bound = dry_run.get("full_retention_upper_bound_bytes")
    sources = dry_run.get("sources")
    if (
        isinstance(upper_bound, bool)
        or not isinstance(upper_bound, int)
        or upper_bound < 0
        or not isinstance(sources, list)
    ):
        raise StorageGovernanceError(
            "storage migration dry-run byte projection is invalid"
        )
    logical_ids: set[object] = set()
    source_identities: set[tuple[object, object]] = set()
    destination_identities: set[tuple[object, object]] = set()
    expected_sources: list[dict[str, object]] = []
    mapping_bytes = 0
    for record in planned_mappings:
        byte_count = record.get("byte_count")
        logical_id = record.get("logical_id")
        source_root_alias = record.get("source_root_alias")
        source_relative_path = record.get("source_relative_path")
        destination_root_alias = record.get("destination_root_alias")
        destination_relative_path = record.get(
            "destination_relative_path"
        )
        file_count = record.get("file_count")
        tree_sha256 = record.get("tree_sha256")
        source_identity = (
            source_root_alias,
            source_relative_path,
        )
        destination_identity = (
            destination_root_alias,
            destination_relative_path,
        )
        if (
            not isinstance(logical_id, str)
            or not logical_id
            or not isinstance(source_root_alias, str)
            or not isinstance(source_relative_path, str)
            or not isinstance(destination_root_alias, str)
            or not isinstance(destination_relative_path, str)
            or isinstance(file_count, bool)
            or not isinstance(file_count, int)
            or file_count < 0
            or not isinstance(tree_sha256, str)
            or _SHA256.fullmatch(tree_sha256) is None
            or logical_id in logical_ids
            or source_identity in source_identities
            or destination_identity in destination_identities
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise StorageGovernanceError(
                "storage migration mapping identity is invalid or duplicated"
            )
        logical_ids.add(logical_id)
        source_identities.add(source_identity)
        destination_identities.add(destination_identity)
        mapping_bytes += byte_count
        expected_sources.append(
            {
                "logical_id": logical_id,
                "root_alias": source_root_alias,
                "relative_path": source_relative_path,
                "file_count": file_count,
                "byte_count": byte_count,
                "tree_sha256": tree_sha256,
            }
        )
    if sources != expected_sources:
        raise StorageGovernanceError(
            "storage migration dry-run sources do not match mappings"
        )
    source_bytes = 0
    for source in expected_sources:
        byte_count = source["byte_count"]
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise StorageGovernanceError(
                "storage migration dry-run source bytes are invalid"
            )
        source_bytes += byte_count
    if upper_bound != source_bytes or upper_bound != mapping_bytes:
        raise StorageGovernanceError(
            "storage migration dry-run byte projection does not match mappings"
        )


def write_storage_migration_attestation(
    path: Path,
    *,
    migration_id: str,
    source_root_alias: str,
    destination_root_alias: str,
    source_root: Path,
    destination_root: Path,
    source_volume: VolumeIdentity,
    destination_volume: VolumeIdentity,
    mappings: tuple[tuple[str, str, str], ...],
    dry_run_path: Path,
    dry_run_sha256: str,
) -> str:
    """Attest verified cross-alias trees without rewriting historical manifests."""

    if (
        source_root_alias == destination_root_alias
        or source_volume == destination_volume
        or _SHA256.fullmatch(dry_run_sha256) is None
        or not mappings
    ):
        raise ValueError("storage migration attestation identity is invalid")
    if dry_run_path.resolve().parent != path.resolve().parent:
        raise ValueError(
            "storage migration dry run and attestation must share one directory"
        )
    dry_run = _load_signed_json(dry_run_path)
    if (
        _file_sha256(dry_run_path) != dry_run_sha256
        or dry_run.get("schema_version")
        != "experiment-storage-migration-dry-run-v1"
        or dry_run.get("source_deletion_authorized") is not False
    ):
        raise StorageGovernanceError(
            "storage migration dry-run identity is invalid"
        )
    records: list[dict[str, object]] = []
    for logical_id, source_relative, destination_relative in mappings:
        if not logical_id:
            raise ValueError("storage migration logical identity is empty")
        source = _safe_root_relative_path(
            source_root,
            source_relative,
            description="storage migration source",
        )
        destination = _safe_root_relative_path(
            destination_root,
            destination_relative,
            description="storage migration destination",
        )
        source_count, source_bytes, source_tree = _tree_identity(source)
        destination_count, destination_bytes, destination_tree = _tree_identity(
            destination
        )
        if (
            source_count,
            source_bytes,
            source_tree,
        ) != (
            destination_count,
            destination_bytes,
            destination_tree,
        ):
            raise StorageGovernanceError(
                f"storage migration tree mismatch: {logical_id}"
            )
        records.append(
            {
                "logical_id": logical_id,
                "source_relative_path": source_relative,
                "destination_relative_path": destination_relative,
                "file_count": source_count,
                "byte_count": source_bytes,
                "tree_sha256": source_tree,
            }
        )
    expected_planned_mappings = [
        {
            **record,
            "source_root_alias": source_root_alias,
            "destination_root_alias": destination_root_alias,
        }
        for record in records
    ]
    _verify_migration_dry_run_projection(
        dry_run,
        expected_planned_mappings,
    )
    payload = {
        "schema_version": "experiment-storage-role-migration-v2",
        "migration_id": migration_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_root_alias": source_root_alias,
        "destination_root_alias": destination_root_alias,
        "source_volume_identity": source_volume.to_dict(),
        "destination_volume_identity": destination_volume.to_dict(),
        "dry_run_sha256": dry_run_sha256,
        "dry_run_relative_path": dry_run_path.name,
        "source_manifests_immutable": True,
        "source_deletion_authorized": False,
        "mappings": records,
    }
    return _write_signed_json(path, payload)


def write_storage_migration_attestation_from_dry_run(
    path: Path,
    *,
    migration_id: str,
    source_root_alias: str,
    destination_root_alias: str,
    source_volume: VolumeIdentity,
    destination_volume: VolumeIdentity,
    mappings: tuple[tuple[str, str, str], ...],
    dry_run_path: Path,
    dry_run_sha256: str,
) -> str:
    """Create an untrusted attestation projection for one later full replay."""

    if (
        source_root_alias == destination_root_alias
        or source_volume == destination_volume
        or _SHA256.fullmatch(dry_run_sha256) is None
        or not mappings
    ):
        raise ValueError("storage migration attestation identity is invalid")
    if dry_run_path.resolve().parent != path.resolve().parent:
        raise ValueError(
            "storage migration dry run and attestation must share one directory"
        )
    dry_run = _load_signed_json(dry_run_path)
    if (
        _file_sha256(dry_run_path) != dry_run_sha256
        or dry_run.get("schema_version")
        != "experiment-storage-migration-dry-run-v1"
        or dry_run.get("source_deletion_authorized") is not False
    ):
        raise StorageGovernanceError(
            "storage migration dry-run identity is invalid"
        )
    raw_planned = dry_run.get("planned_mappings")
    if not isinstance(raw_planned, list) or any(
        not isinstance(record, dict) for record in raw_planned
    ):
        raise StorageGovernanceError(
            "storage migration dry-run mappings are invalid"
        )
    planned = [dict(record) for record in raw_planned]
    _verify_migration_dry_run_projection(dry_run, planned)
    expected_mapping_keys = {
        (logical_id, source_relative, destination_relative)
        for logical_id, source_relative, destination_relative in mappings
    }
    records: list[dict[str, object]] = []
    for record in planned:
        key = (
            record.get("logical_id"),
            record.get("source_relative_path"),
            record.get("destination_relative_path"),
        )
        if (
            key not in expected_mapping_keys
            or record.get("source_root_alias") != source_root_alias
            or record.get("destination_root_alias") != destination_root_alias
        ):
            raise StorageGovernanceError(
                "storage migration mappings differ from the signed dry run"
            )
        records.append(
            {
                field: record[field]
                for field in (
                    "logical_id",
                    "source_relative_path",
                    "destination_relative_path",
                    "file_count",
                    "byte_count",
                    "tree_sha256",
                )
            }
        )
    if len(records) != len(expected_mapping_keys):
        raise StorageGovernanceError(
            "storage migration mappings differ from the signed dry run"
        )
    return _write_signed_json(
        path,
        {
            "schema_version": "experiment-storage-role-migration-v2",
            "migration_id": migration_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_root_alias": source_root_alias,
            "destination_root_alias": destination_root_alias,
            "source_volume_identity": source_volume.to_dict(),
            "destination_volume_identity": destination_volume.to_dict(),
            "dry_run_sha256": dry_run_sha256,
            "dry_run_relative_path": dry_run_path.name,
            "source_manifests_immutable": True,
            "source_deletion_authorized": False,
            "construction_status": "pending_independent_replay",
            "mappings": records,
        },
    )


def load_storage_migration_attestation(path: Path) -> dict[str, object]:
    """Load a signed v2 role-migration attestation for production dispatch."""

    payload = _load_signed_json(path)
    if payload.get("schema_version") != "experiment-storage-role-migration-v2":
        raise StorageGovernanceError(
            "storage migration attestation schema is unsupported"
        )
    return payload


def verify_storage_migration_attestation(
    path: Path,
    *,
    expected_sha256: str,
    source_root_alias: str,
    destination_root_alias: str,
    source_root: Path,
    destination_root: Path,
    source_volume: VolumeIdentity,
    destination_volume: VolumeIdentity,
    dry_run_path: Path,
    volume_probe: Callable[[Path], VolumeIdentity],
    workers: int = 1,
) -> dict[str, object]:
    """Reverify a cross-volume attestation before resolver or cleanup use."""

    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("storage migration attestation SHA-256 is invalid")
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 32
    ):
        raise ValueError("attestation workers must be between 1 and 32")
    payload = _load_signed_json(path)
    dry_run = _load_signed_json(dry_run_path)
    dry_run_sha256 = _file_sha256(dry_run_path)
    if (
        volume_probe(source_root) != source_volume
        or volume_probe(destination_root) != destination_volume
        or _file_sha256(path) != expected_sha256
        or payload.get("schema_version")
        != "experiment-storage-role-migration-v2"
        or payload.get("source_root_alias") != source_root_alias
        or payload.get("destination_root_alias") != destination_root_alias
        or payload.get("source_volume_identity") != source_volume.to_dict()
        or payload.get("destination_volume_identity")
        != destination_volume.to_dict()
        or payload.get("dry_run_sha256") != dry_run_sha256
        or payload.get("dry_run_relative_path") != dry_run_path.name
        or payload.get("source_manifests_immutable") is not True
        or payload.get("source_deletion_authorized") is not False
        or dry_run.get("schema_version")
        != "experiment-storage-migration-dry-run-v1"
        or dry_run.get("source_deletion_authorized") is not False
    ):
        raise StorageGovernanceError(
            "storage migration attestation identity does not replay"
        )
    raw_mappings = payload.get("mappings")
    if not isinstance(raw_mappings, list) or not raw_mappings:
        raise StorageGovernanceError(
            "storage migration attestation has no mappings"
        )
    replay_specs: list[
        tuple[str, Mapping[str, object], Path, Path]
    ] = []
    logical_ids: set[str] = set()
    for raw in raw_mappings:
        if not isinstance(raw, Mapping):
            raise StorageGovernanceError(
                "storage migration attestation mapping is invalid"
            )
        logical_id = raw.get("logical_id")
        source_relative = raw.get("source_relative_path")
        destination_relative = raw.get("destination_relative_path")
        if (
            not isinstance(logical_id, str)
            or not logical_id
            or logical_id in logical_ids
            or not isinstance(source_relative, str)
            or not isinstance(destination_relative, str)
        ):
            raise StorageGovernanceError(
                "storage migration attestation paths are invalid"
            )
        logical_ids.add(logical_id)
        source = _safe_root_relative_path(
            source_root,
            source_relative,
            description="storage migration source",
        )
        destination = _safe_root_relative_path(
            destination_root,
            destination_relative,
            description="storage migration destination",
        )
        replay_specs.append((logical_id, raw, source, destination))
    native_source_replay = (
        workers > 1
        and os.name != "nt"
        and _windows_path_for_mounted_drive(source_root) is not None
    )
    native_destination_replay = (
        workers > 1
        and os.name != "nt"
        and _windows_path_for_mounted_drive(destination_root) is not None
    )
    source_mappings = tuple(
        (logical_id, str(raw["source_relative_path"]))
        for logical_id, raw, _source, _destination in replay_specs
    )
    destination_mappings = tuple(
        (logical_id, str(raw["destination_relative_path"]))
        for logical_id, raw, _source, _destination in replay_specs
    )
    if native_source_replay and native_destination_replay:
        with ThreadPoolExecutor(max_workers=2) as native_executor:
            source_future = native_executor.submit(
                _native_verify_tree_mappings,
                root=source_root,
                mappings=source_mappings,
                workers=workers,
            )
            destination_future = native_executor.submit(
                _native_verify_tree_mappings,
                root=destination_root,
                mappings=destination_mappings,
                workers=workers,
            )
            source_identities = source_future.result()
            destination_identities = destination_future.result()
    elif native_destination_replay:
        with ThreadPoolExecutor(max_workers=1) as native_executor:
            destination_future = native_executor.submit(
                _native_verify_tree_mappings,
                root=destination_root,
                mappings=destination_mappings,
                workers=workers,
            )
            with ThreadPoolExecutor(max_workers=workers) as source_executor:
                source_identities = {
                    logical_id: identity
                    for (logical_id, _raw, _source, _destination), identity in zip(
                        replay_specs,
                        source_executor.map(
                            _tree_identity,
                            tuple(
                                source
                                for _logical_id, _raw, source, _destination
                                in replay_specs
                            ),
                        ),
                        strict=True,
                    )
                }
            destination_identities = destination_future.result()
    elif native_source_replay:
        with ThreadPoolExecutor(max_workers=1) as native_executor:
            source_future = native_executor.submit(
                _native_verify_tree_mappings,
                root=source_root,
                mappings=source_mappings,
                workers=workers,
            )
            with ThreadPoolExecutor(max_workers=workers) as destination_executor:
                destination_identities = {
                    logical_id: identity
                    for (logical_id, _raw, _source, _destination), identity in zip(
                        replay_specs,
                        destination_executor.map(
                            _tree_identity,
                            tuple(
                                destination
                                for _logical_id, _raw, _source, destination
                                in replay_specs
                            ),
                        ),
                        strict=True,
                    )
                }
            source_identities = source_future.result()
    else:
        tree_paths = tuple(
            path_item
            for _logical_id, _raw, source, destination in replay_specs
            for path_item in (source, destination)
        )
        if workers == 1:
            tree_identities = tuple(
                _tree_identity(item) for item in tree_paths
            )
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                tree_identities = tuple(
                    executor.map(_tree_identity, tree_paths)
                )
        source_identities = {
            logical_id: tree_identities[index * 2]
            for index, (logical_id, _raw, _source, _destination)
            in enumerate(replay_specs)
        }
        destination_identities = {
            logical_id: tree_identities[index * 2 + 1]
            for index, (logical_id, _raw, _source, _destination)
            in enumerate(replay_specs)
        }
    replayed_mappings: list[dict[str, object]] = []
    for logical_id, raw, _source, _destination in replay_specs:
        source_identity = source_identities[logical_id]
        destination_identity = destination_identities[logical_id]
        recorded_identity = (
            raw.get("file_count"),
            raw.get("byte_count"),
            raw.get("tree_sha256"),
        )
        if source_identity != destination_identity or source_identity != recorded_identity:
            raise StorageGovernanceError(
                "storage migration attestation tree no longer matches: "
                f"{raw.get('logical_id')}"
            )
        replayed_mappings.append(
            {
                **dict(raw),
                "source_root_alias": source_root_alias,
                "destination_root_alias": destination_root_alias,
            }
        )
    _verify_migration_dry_run_projection(dry_run, replayed_mappings)
    return payload


def resolve_run_from_locator(
    run_label: str,
    *,
    policy_path: Path,
    storage_root_locator_path: Path,
    legacy_registry_path: Path | None,
    volume_probe: Callable[[Path], VolumeIdentity],
    staging_root_alias: str = "wsl_staging",
    retention_root_alias: str = "e_archive",
) -> Path:
    """Resolve the newest verified v2 generation, then fall back to v1."""

    locator = StorageRootLocator.from_toml(storage_root_locator_path)
    staging_root = locator.resolve(staging_root_alias).absolute_path
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy.from_toml(policy_path),
        locator=locator,
        state_root=staging_root / ".storage-governance",
        retention_state_root=(
            locator.resolve(retention_root_alias).absolute_path
            / ".storage-governance"
        ),
        free_space=lambda path: shutil.disk_usage(path).free,
        volume_probe=volume_probe,
        legacy_registry_path=legacy_registry_path,
    )
    return governance.resolve_run(run_label)


def is_retained_path_from_locator(
    path: Path,
    *,
    policy_path: Path,
    storage_root_locator_path: Path,
    legacy_registry_path: Path | None,
    volume_probe: Callable[[Path], VolumeIdentity],
    staging_root_alias: str = "wsl_staging",
    retention_root_alias: str = "e_archive",
) -> bool:
    """Return whether a path is any immutable v2 or legacy v1 archive tree."""

    locator = StorageRootLocator.from_toml(storage_root_locator_path)
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy.from_toml(policy_path),
        locator=locator,
        state_root=(
            locator.resolve(staging_root_alias).absolute_path
            / ".storage-governance"
        ),
        retention_state_root=(
            locator.resolve(retention_root_alias).absolute_path
            / ".storage-governance"
        ),
        free_space=lambda candidate: shutil.disk_usage(candidate).free,
        volume_probe=volume_probe,
        legacy_registry_path=legacy_registry_path,
    )
    return governance.is_retained_path(path)


def preflight_cli_attempt(
    *,
    config_path: Path,
    output_dir: Path,
    run_label: str | None = None,
    workers: int = 1,
    threads: int = 1,
    processes: int = 1,
    batch_size: int = 1,
    queue_depth: int = 1,
    row_group_size: int = 1,
) -> StartPermit:
    """Apply the shared Stage 0--8 stop gate at every producer CLI boundary."""

    from evrptw.experiment_lifecycle import (
        ExperimentCatalog,
        ExperimentLifecycleController,
        LifecycleError,
        build_repository_plan,
        load_historical_migration_gate,
        register_lifecycle_writer,
    )

    resolved_output = output_dir.resolve()
    selected_label = run_label or resolved_output.name
    match = _RUN_LABEL.fullmatch(selected_label)
    if match is None:
        reason = "new Stage 0--8 CLI attempts require a canonical run label"
        _persist_cli_plan_rejection(
            config_path=config_path,
            output_dir=resolved_output,
            run_label=selected_label,
            reason=reason,
        )
        raise StorageGovernanceError(reason)
    if resolved_output.name != selected_label:
        reason = "CLI output directory must end with its run label"
        _persist_cli_plan_rejection(
            config_path=config_path,
            output_dir=resolved_output,
            run_label=selected_label,
            reason=reason,
        )
        raise StorageGovernanceError(reason)
    stage_id = f"stage{match.group('major')}"
    if match.group("minor") is not None:
        stage_id += f".{match.group('minor')}"
    try:
        resolved_config = config_path.resolve(strict=True)
        with resolved_config.open("rb") as handle:
            config_payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        reason = f"cannot read experiment plan: {config_path}"
        _persist_cli_plan_rejection(
            config_path=config_path,
            output_dir=resolved_output,
            run_label=selected_label,
            reason=f"{reason}: {type(error).__name__}: {error}",
        )
        raise StorageGovernanceError(reason) from error
    storage_sections = [
        value
        for key, value in config_payload.items()
        if key.startswith("artifact_storage") and isinstance(value, dict)
    ]
    if not storage_sections:
        reason = "experiment plan does not declare an artifact storage hard cap"
        _persist_cli_plan_rejection(
            config_path=resolved_config,
            output_dir=resolved_output,
            run_label=selected_label,
            reason=reason,
        )
        raise StorageGovernanceError(reason)
    per_run_caps: list[int] = []
    for section in storage_sections:
        value = section.get("per_run_max_bytes")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            reason = "experiment artifact storage run hard cap is missing"
            _persist_cli_plan_rejection(
                config_path=resolved_config,
                output_dir=resolved_output,
                run_label=selected_label,
                reason=reason,
            )
            raise StorageGovernanceError(reason)
        per_run_caps.append(value)
    planned_archive_bytes = max(per_run_caps)
    prerequisite_sha256_by_contract: dict[str, str] = {}
    if stage_id == "stage05.2":
        stage052 = config_payload.get("stage05_2")
        if not isinstance(stage052, dict) or not isinstance(
            stage052.get("stage051_manifest"), str
        ):
            raise StorageGovernanceError(
                "Stage 5.2 lifecycle plan requires the Stage 5.1 prerequisite"
            )
        prerequisite_path = resolved_config.parent.parent / str(
            stage052["stage051_manifest"]
        )
        try:
            prerequisite_sha256_by_contract["stage051_readiness"] = hashlib.sha256(
                prerequisite_path.read_bytes()
            ).hexdigest()
        except OSError as error:
            raise StorageGovernanceError(
                "Stage 5.2 lifecycle prerequisite cannot be read"
            ) from error
    locator_path = resolved_config.parent / "stage052_storage_roots.local.toml"
    policy_path = resolved_config.parent / "experiment_storage_governance.toml"
    catalog_path = resolved_config.parent / "experiment_catalog.toml"
    if not catalog_path.is_file():
        reason = "experiment catalog is missing at the producer boundary"
        _persist_cli_plan_rejection(
            config_path=resolved_config,
            output_dir=resolved_output,
            run_label=selected_label,
            reason=reason,
        )
        raise StorageGovernanceError(reason)
    catalog = ExperimentCatalog.from_toml(catalog_path)
    spec = catalog.for_run_label(selected_label)
    repository = resolved_config.parent.parent
    for contract, relative_path in spec.prerequisite_paths.items():
        prerequisite_path = repository.joinpath(*PurePosixPath(relative_path).parts)
        try:
            prerequisite_sha256 = _file_sha256(prerequisite_path)
        except OSError as error:
            raise StorageGovernanceError(
                f"catalog prerequisite cannot be read: {contract}"
            ) from error
        existing_prerequisite = prerequisite_sha256_by_contract.get(contract)
        if (
            existing_prerequisite is not None
            and existing_prerequisite != prerequisite_sha256
        ):
            raise StorageGovernanceError(
                f"config and catalog prerequisite identities differ: {contract}"
            )
        prerequisite_sha256_by_contract[contract] = prerequisite_sha256
    locator = StorageRootLocator.from_toml(locator_path)
    if stage_id == "stage05.2":
        historical_gate_path = (
            locator.resolve("e_archive").absolute_path
            / ".experiment-lifecycle"
            / "historical-migration"
            / "gate.json"
        )
        load_historical_migration_gate(
            historical_gate_path,
            migration_ledger_path=(
                repository
                / "experiments"
                / "registries"
                / "experiment_lifecycle_v3_migration.json"
            ),
            repository=repository,
        )
        prerequisite_sha256_by_contract["historical_migration"] = _file_sha256(
            historical_gate_path
        )
        prerequisite_sha256_by_contract["campaign_geometry"] = hashlib.sha256(
            _canonical_json(
                {
                    "run_label": selected_label,
                    "stage_id": stage_id,
                    "configuration_sha256": _file_sha256(resolved_config),
                }
            )
        ).hexdigest()
    from evrptw.stage052_campaign_runner import free_bytes, probe_volume_identity

    staging = locator.resolve("wsl_staging")
    legacy_registry_path = (
        resolved_config.parent.parent
        / "experiments"
        / "registries"
        / "stage05.2_retention_registry.csv"
    )
    governance = ExperimentStorageGovernance(
        policy=GovernancePolicy.from_toml(policy_path),
        locator=locator,
        state_root=staging.absolute_path / ".storage-governance",
        retention_state_root=(
            locator.resolve("e_archive").absolute_path / ".storage-governance"
        ),
        free_space=free_bytes,
        volume_probe=probe_volume_identity,
        legacy_registry_path=(
            legacy_registry_path if legacy_registry_path.is_file() else None
        ),
    )
    lifecycle_plan = build_repository_plan(
        repository=resolved_config.parent.parent,
        config_path=resolved_config,
        run_label=selected_label,
        run_dir=resolved_output,
        planned_archive_bytes=planned_archive_bytes,
        max_workspace_bytes=planned_archive_bytes,
        workers=workers,
        threads=threads,
        processes=processes,
        batch_size=batch_size,
        queue_depth=queue_depth,
        row_group_size=row_group_size,
        prerequisite_sha256_by_contract=prerequisite_sha256_by_contract,
    )
    lifecycle = ExperimentLifecycleController(
        catalog=catalog,
        state_root=locator.resolve("e_archive").absolute_path / ".experiment-lifecycle",
        storage_state_root=(
            locator.resolve("e_archive").absolute_path / ".storage-governance"
        ),
        capacity_state_root=staging.absolute_path / ".storage-governance",
        archive_root=locator.resolve("e_archive").absolute_path,
    )
    permit = governance.preflight_run(
        StartRequest(
            stage_id=stage_id,
            run_label=selected_label,
            run_dir=resolved_output,
            staging_root_alias="wsl_staging",
            host_root_alias="d_host",
            archive_root_alias="e_archive",
            planned_archive_bytes=planned_archive_bytes,
            max_active_workspace_bytes=planned_archive_bytes,
            projected_host_growth_bytes=planned_archive_bytes,
            stage_plan_sha256=lifecycle_plan.plan_sha256,
        )
    )
    try:
        lifecycle.plan(lifecycle_plan)
    except LifecycleError as error:
        rejection_path = _persist_cli_plan_rejection(
            config_path=resolved_config,
            output_dir=resolved_output,
            run_label=selected_label,
            reason=f"lifecycle admission failed: {error}",
        )
        governance.reconcile_permit(
            selected_label,
            outcome="aborted_audited",
            evidence_sha256=_file_sha256(rejection_path),
        )
        raise StorageGovernanceError(
            "lifecycle admission failed after capacity reservation"
        ) from error
    lifecycle.permit(selected_label, storage_permit_path=permit.permit_path)
    lifecycle.start(selected_label, runtime_plan=lifecycle_plan)
    register_lifecycle_writer(
        state_root=lifecycle.state_root,
        run_label=selected_label,
    )
    return permit


def seal_cli_attempt(
    *,
    config_path: Path,
    output_dir: Path,
    manifest_path: Path | None,
    run_label: str | None = None,
    failure_code: str = "",
) -> Path:
    """Stop the governed writer and atomically advance a finished CLI run to SEALED."""

    from evrptw.experiment_lifecycle import (
        ExperimentCatalog,
        ExperimentLifecycleController,
        release_lifecycle_writer,
    )

    resolved_config = config_path.resolve(strict=True)
    resolved_output = output_dir.resolve(strict=True)
    selected_label = run_label or resolved_output.name
    if resolved_output.name != selected_label:
        raise StorageGovernanceError("sealed CLI output identity differs")
    locator = StorageRootLocator.from_toml(
        resolved_config.parent / "stage052_storage_roots.local.toml"
    )
    archive_root = locator.resolve("e_archive").absolute_path
    lifecycle = ExperimentLifecycleController(
        catalog=ExperimentCatalog.from_toml(
            resolved_config.parent / "experiment_catalog.toml"
        ),
        state_root=archive_root / ".experiment-lifecycle",
        storage_state_root=archive_root / ".storage-governance",
        capacity_state_root=(
            locator.resolve("wsl_staging").absolute_path / ".storage-governance"
        ),
        archive_root=archive_root,
    )
    release_lifecycle_writer(
        state_root=lifecycle.state_root,
        run_label=selected_label,
    )
    with _state_lock(
        lifecycle.state_root / "writer-locks" / f"{selected_label}.lock"
    ):
        selected_manifest = (
            build_cli_terminal_manifest(
                output_dir=resolved_output,
                run_label=selected_label,
            )
            if manifest_path is None
            else manifest_path.resolve(strict=True)
        )
        lifecycle.seal(
            selected_label,
            manifest_path=selected_manifest,
            failure_code=failure_code,
        )
    return selected_manifest


def build_cli_terminal_manifest(
    *,
    output_dir: Path,
    run_label: str,
    status: str = "complete",
    evidence_completeness: str = "complete",
) -> Path:
    """Build an exact terminal inventory and verify every child artifact."""

    from evrptw.artifacts import atomic_write_signed_json, signed_sidecar_matches

    resolved_output = output_dir.resolve(strict=True)
    identities: dict[str, dict[str, object]] = {}
    child_terminal_states: list[tuple[str, str]] = []

    def add_identity(path: Path, *, checksum: str | None = None) -> None:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(resolved_output) or not resolved.is_file():
            raise StorageGovernanceError("terminal manifest path is outside its run")
        relative = resolved.relative_to(resolved_output).as_posix()
        stat_before = resolved.stat()
        digest = _file_sha256(resolved) if checksum is None else checksum
        stat_after = resolved.stat()
        if (
            stat_after.st_size != stat_before.st_size
            or stat_after.st_mtime_ns != stat_before.st_mtime_ns
        ):
            raise StorageGovernanceError(
                "terminal manifest artifact changed while hashing"
            )
        identity = {
            "relative_path": relative,
            "byte_size": stat_before.st_size,
            "checksum": digest,
            "modified_time_ns": stat_before.st_mtime_ns,
        }
        existing = identities.get(relative)
        if existing is not None and existing != identity:
            raise StorageGovernanceError("terminal manifest identity conflicts")
        identities[relative] = identity

    def replay_artifact_inventory(
        artifacts: list[object],
        *,
        artifact_root: Path,
    ) -> None:
        for raw in artifacts:
            if not isinstance(raw, dict):
                raise StorageGovernanceError("child artifact identity is invalid")
            relative = PurePosixPath(str(raw.get("relative_path", "")))
            checksum = str(raw.get("checksum", ""))
            byte_size = raw.get("byte_size")
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or _SHA256.fullmatch(checksum) is None
                or isinstance(byte_size, bool)
                or not isinstance(byte_size, int)
                or byte_size < 0
            ):
                raise StorageGovernanceError("child artifact identity is invalid")
            artifact_path = artifact_root.joinpath(*relative.parts)
            stat_before = artifact_path.stat()
            if not artifact_path.is_file() or stat_before.st_size != byte_size:
                raise StorageGovernanceError("child artifact size differs")
            observed_checksum = _file_sha256(artifact_path)
            stat_after = artifact_path.stat()
            if (
                stat_after.st_size != stat_before.st_size
                or stat_after.st_mtime_ns != stat_before.st_mtime_ns
            ):
                raise StorageGovernanceError("child artifact changed while hashing")
            if observed_checksum != checksum:
                raise StorageGovernanceError("child artifact checksum differs")
            add_identity(artifact_path, checksum=observed_checksum)

    child_manifests = tuple(
        sorted(
            path
            for path in resolved_output.glob("**/control/*_manifest.json")
            if not path.name.endswith("_lifecycle_manifest.json")
        )
    )
    for manifest_path in child_manifests:
        sidecar_path = manifest_path.with_suffix(".sha256")
        if not signed_sidecar_matches(manifest_path, sidecar_path):
            raise StorageGovernanceError("child artifact manifest signature differs")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StorageGovernanceError("child artifact manifest is invalid") from error
        artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or payload.get("run_label") != run_label
            or payload.get("status") not in {"complete", "partial", "failed"}
            or payload.get("evidence_completeness")
            not in {"complete", "partial", "legacy_unknown"}
            or not isinstance(artifacts, list)
        ):
            raise StorageGovernanceError("child artifact inventory is missing")
        child_terminal_states.append(
            (str(payload["status"]), str(payload["evidence_completeness"]))
        )
        child_root = manifest_path.parent.parent
        replay_artifact_inventory(artifacts, artifact_root=child_root)
        add_identity(manifest_path)
        add_identity(sidecar_path)
    shard_manifests = tuple(
        sorted(resolved_output.glob("**/*_shard_manifest_*.json"))
    )
    for manifest_path in shard_manifests:
        sidecar_path = manifest_path.with_suffix(".sha256")
        if not signed_sidecar_matches(manifest_path, sidecar_path):
            raise StorageGovernanceError("shard artifact manifest signature differs")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StorageGovernanceError("shard artifact manifest is invalid") from error
        artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
        instance = payload.get("instance") if isinstance(payload, dict) else None
        seed = payload.get("seed") if isinstance(payload, dict) else None
        completeness = (
            payload.get("evidence_completeness") if isinstance(payload, dict) else None
        )
        shard_ordinal = (
            payload.get("shard_ordinal") if isinstance(payload, dict) else None
        )
        worker_identity = (
            payload.get("worker_identity") if isinstance(payload, dict) else None
        )
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != "artifact-storage-v2"
            or payload.get("run_label") != run_label
            or completeness not in {"complete", "partial"}
            or not isinstance(artifacts, list)
            or not isinstance(instance, str)
            or not instance
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or manifest_path.parent.name != str(seed)
            or manifest_path.parent.parent.name != instance
            or isinstance(shard_ordinal, bool)
            or not isinstance(shard_ordinal, int)
            or shard_ordinal < 0
            or not isinstance(worker_identity, str)
            or not worker_identity
        ):
            raise StorageGovernanceError("shard artifact inventory is missing")
        expected_prefix = (instance, str(seed))
        if any(
            not isinstance(raw, dict)
            or PurePosixPath(str(raw.get("relative_path", ""))).parts[:2]
            != expected_prefix
            for raw in artifacts
        ):
            raise StorageGovernanceError("shard artifact path identity differs")
        child_terminal_states.append(
            ("complete" if completeness == "complete" else "partial", completeness)
        )
        replay_artifact_inventory(
            artifacts,
            artifact_root=manifest_path.parent.parent.parent,
        )
        add_identity(manifest_path)
        add_identity(sidecar_path)
    for post_manifest_path in sorted(
        resolved_output.glob("**/control/*_persistence_attribution.json")
    ):
        post_sidecar = post_manifest_path.with_suffix(".sha256")
        if not signed_sidecar_matches(post_manifest_path, post_sidecar):
            raise StorageGovernanceError(
                "post-manifest persistence attribution signature differs"
            )
        add_identity(post_manifest_path)
        add_identity(post_sidecar)
    root_control = resolved_output / "control"
    if root_control.is_dir():
        for control_path in sorted(root_control.glob("*.json")):
            if control_path.name.endswith("_lifecycle_manifest.json"):
                continue
            control_sidecar = control_path.with_suffix(".sha256")
            if not signed_sidecar_matches(control_path, control_sidecar):
                raise StorageGovernanceError("root control signature differs")
            add_identity(control_path)
            add_identity(control_sidecar)
    for root_file in sorted(item for item in resolved_output.iterdir() if item.is_file()):
        add_identity(root_file)
    if not identities:
        raise StorageGovernanceError("terminal manifest has no governed artifacts")
    if status == "complete" and (
        evidence_completeness != "complete"
        or any(
            child_status != "complete" or child_completeness != "complete"
            for child_status, child_completeness in child_terminal_states
        )
    ):
        raise StorageGovernanceError(
            "incomplete child evidence cannot become a complete terminal manifest"
        )
    manifest_path = (
        resolved_output / "control" / f"{run_label}_lifecycle_manifest.json"
    )
    terminal_sidecar = manifest_path.with_suffix(".sha256")
    observed: set[str] = set()
    for directory, directories, files in os.walk(resolved_output):
        directories.sort()
        files.sort()
        directory_path = Path(directory)
        if any((directory_path / name).is_symlink() for name in directories):
            raise StorageGovernanceError("terminal run contains a directory symlink")
        for name in files:
            path = directory_path / name
            if path in {manifest_path, terminal_sidecar}:
                continue
            if path.is_symlink() or not path.is_file():
                raise StorageGovernanceError("terminal run contains a non-regular file")
            observed.add(path.relative_to(resolved_output).as_posix())
    missing = set(identities) - observed
    unlisted = observed - set(identities)
    if missing:
        raise StorageGovernanceError(
            f"terminal manifest is missing declared files: {sorted(missing)[:5]}"
        )
    if unlisted and status != "complete":
        for unlisted_path in sorted(unlisted):
            add_identity(
                resolved_output.joinpath(*PurePosixPath(unlisted_path).parts)
            )
        unlisted = set()
    if unlisted:
        raise StorageGovernanceError(
            "terminal manifest file set differs: "
            f"unlisted={sorted(unlisted)[:5]}, missing=[]"
        )
    for terminal_relative, identity in identities.items():
        path = resolved_output.joinpath(*PurePosixPath(terminal_relative).parts)
        stat = path.stat()
        if (
            stat.st_size != identity["byte_size"]
            or stat.st_mtime_ns != identity["modified_time_ns"]
        ):
            raise StorageGovernanceError(
                "terminal manifest artifact changed after hashing"
            )
    atomic_write_signed_json(
        manifest_path,
        {
            "schema_version": "experiment-cli-terminal-manifest-v1",
            "run_label": run_label,
            "status": status,
            "evidence_completeness": evidence_completeness,
            "artifacts": [identities[key] for key in sorted(identities)],
        },
    )
    return manifest_path


def build_cli_failure_manifest(*, output_dir: Path, run_label: str) -> Path:
    """Inventory an untrusted failed run without interpreting broken child state."""

    from evrptw.artifacts import atomic_write_signed_json

    resolved_output = output_dir.resolve(strict=True)
    if not resolved_output.is_dir():
        raise StorageGovernanceError("failed CLI run directory is missing")
    manifest_path = (
        resolved_output / "control" / f"{run_label}_failure_lifecycle_manifest.json"
    )
    sidecar_path = manifest_path.with_suffix(".sha256")
    identities: list[dict[str, object]] = []
    for directory, directories, files in os.walk(resolved_output):
        directories.sort()
        files.sort()
        directory_path = Path(directory)
        if any((directory_path / name).is_symlink() for name in directories):
            raise StorageGovernanceError("failed CLI run contains a directory symlink")
        for name in files:
            path = directory_path / name
            if path in {manifest_path, sidecar_path}:
                continue
            if path.is_symlink() or not path.is_file():
                raise StorageGovernanceError("failed CLI run contains a non-regular file")
            stat = path.stat()
            identities.append(
                {
                    "relative_path": path.relative_to(resolved_output).as_posix(),
                    "byte_size": stat.st_size,
                    "checksum": _file_sha256(path),
                    "modified_time_ns": stat.st_mtime_ns,
                }
            )
    if not identities:
        raise StorageGovernanceError("failed CLI run has no failure evidence")
    for identity in identities:
        path = resolved_output.joinpath(
            *PurePosixPath(str(identity["relative_path"])).parts
        )
        stat = path.stat()
        if (
            stat.st_size != identity["byte_size"]
            or stat.st_mtime_ns != identity["modified_time_ns"]
        ):
            raise StorageGovernanceError("failed CLI artifact changed after hashing")
    atomic_write_signed_json(
        manifest_path,
        {
            "schema_version": "experiment-cli-failure-manifest-v1",
            "run_label": run_label,
            "status": "failed",
            "evidence_completeness": "partial",
            "artifact_trust": "untrusted_failure_capsule",
            "artifacts": sorted(
                identities,
                key=lambda item: str(item["relative_path"]),
            ),
        },
    )
    return manifest_path


def seal_failed_cli_attempt(
    *,
    config_path: Path,
    output_dir: Path,
    run_label: str,
    error: BaseException,
) -> bool:
    """Persist a controlled failure inventory and advance RUNNING to SEALED."""

    from evrptw.artifacts import atomic_write_signed_json
    from evrptw.experiment_lifecycle import (
        ExperimentCatalog,
        ExperimentLifecycleController,
        ExperimentPlan,
        LifecycleState,
        _load_signed_json,
        release_lifecycle_writer,
    )

    resolved_config = config_path.resolve(strict=True)
    locator = StorageRootLocator.from_toml(
        resolved_config.parent / "stage052_storage_roots.local.toml"
    )
    archive_root = locator.resolve("e_archive").absolute_path
    lifecycle = ExperimentLifecycleController(
        catalog=ExperimentCatalog.from_toml(
            resolved_config.parent / "experiment_catalog.toml"
        ),
        state_root=archive_root / ".experiment-lifecycle",
        storage_state_root=archive_root / ".storage-governance",
        capacity_state_root=(
            locator.resolve("wsl_staging").absolute_path / ".storage-governance"
        ),
        archive_root=archive_root,
    )
    active = tuple(
        record
        for record in lifecycle.records()
        if record.run_label == run_label and record.state == LifecycleState.RUNNING
    )
    if not active:
        return False
    if len(active) != 1:
        raise StorageGovernanceError("CLI lifecycle run identity is ambiguous")
    plan_payload = _load_signed_json(
        lifecycle.state_root / "plans" / f"{run_label}.json"
    )
    raw_plan = plan_payload.get("plan")
    if not isinstance(raw_plan, dict):
        raise StorageGovernanceError("failed CLI lifecycle plan is invalid")
    plan = ExperimentPlan.from_dict(raw_plan)
    resolved_output = output_dir.resolve()
    if (
        plan.run_dir != resolved_output
        or plan.plan_sha256 != active[0].plan_sha256
        or resolved_output.name != run_label
    ):
        raise StorageGovernanceError("failed CLI output differs from lifecycle plan")
    if resolved_output.exists() and not resolved_output.is_dir():
        raise StorageGovernanceError("failed CLI output is not a directory")
    resolved_output.mkdir(parents=True, exist_ok=True)
    release_lifecycle_writer(
        state_root=lifecycle.state_root,
        run_label=run_label,
    )
    with _state_lock(lifecycle.state_root / "writer-locks" / f"{run_label}.lock"):
        atomic_write_signed_json(
            resolved_output / "failure_summary.json",
            {
                "schema_version": "experiment-cli-failure-summary-v1",
                "run_label": run_label,
                "status": "failed",
                "failure_code": "runner_failure",
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        manifest = build_cli_failure_manifest(
            output_dir=resolved_output,
            run_label=run_label,
        )
        lifecycle.seal(
            run_label,
            manifest_path=manifest,
            failure_code="runner_failure",
        )
    return True


def _persist_cli_plan_rejection(
    *,
    config_path: Path,
    output_dir: Path,
    run_label: str,
    reason: str,
) -> Path:
    """Persist plan-level stop-gate failures even when volume probing cannot start."""

    config_parent = config_path.absolute().parent
    fallback_root = config_parent.parent / ".storage-governance"
    state_root = fallback_root
    locator_path = config_parent / "stage052_storage_roots.local.toml"
    try:
        locator = StorageRootLocator.from_toml(locator_path)
        state_root = locator.resolve("wsl_staging").absolute_path / ".storage-governance"
    except (OSError, KeyError, ValueError):
        pass
    token = (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')}-"
        f"{hashlib.sha256(run_label.encode('utf-8')).hexdigest()[:12]}"
    )
    observation_path = state_root / "observations" / f"plan-rejection-{token}.json"
    _write_signed_json(
        observation_path,
        {
            "schema_version": POLICY_SCHEMA_VERSION,
            "observed_at_utc": datetime.now(UTC).isoformat(),
            "stage_id": "",
            "run_label": run_label,
            "run_dir": str(output_dir),
            "configuration_path": str(config_path.absolute()),
            "stage_plan_sha256": "",
            "free_bytes_by_alias": {},
            "required_bytes_by_alias": {},
            "deficits_by_alias": {},
            "measurement_errors_by_alias": {},
            "plan_error": reason,
            "passed": False,
        },
    )
    return observation_path


def persist_cli_plan_rejection(
    *,
    config_path: Path,
    output_dir: Path,
    run_label: str,
    reason: str,
) -> Path:
    """Public fail-fast writer for a signed pre-admission rejection."""

    return _persist_cli_plan_rejection(
        config_path=config_path,
        output_dir=output_dir,
        run_label=run_label,
        reason=reason,
    )


def load_adjudication_record(
    path: Path,
    *,
    expected_sha256: str,
) -> AdjudicationRecord:
    """Load and reverify one explicitly approved adjudication record."""

    if _SHA256.fullmatch(expected_sha256) is None:
        raise StorageGovernanceError("expected adjudication SHA-256 is invalid")
    payload = _load_signed_json(path)
    observed = _file_sha256(path)
    if observed != expected_sha256:
        raise StorageGovernanceError("adjudication SHA-256 differs from approved identity")
    expected_fields = {
        "schema_version",
        "run_label",
        "root_cause_id",
        "canonical_representative_run_label",
        "failure_location",
        "evidence_references",
    }
    if set(payload) != expected_fields:
        raise StorageGovernanceError("adjudication fields do not match its schema")
    if payload["schema_version"] != "experiment-root-cause-adjudication-v1":
        raise StorageGovernanceError("unsupported adjudication schema")
    references = payload["evidence_references"]
    if not isinstance(references, list) or any(
        not isinstance(reference, str) for reference in references
    ):
        raise StorageGovernanceError("adjudication evidence references are invalid")
    run_label = payload["run_label"]
    root_cause_id = payload["root_cause_id"]
    canonical_representative_run_label = payload[
        "canonical_representative_run_label"
    ]
    failure_location = payload["failure_location"]
    if not all(
        isinstance(value, str)
        for value in (
            run_label,
            root_cause_id,
            canonical_representative_run_label,
            failure_location,
        )
    ):
        raise StorageGovernanceError("adjudication string fields are invalid")
    assert isinstance(run_label, str)
    assert isinstance(root_cause_id, str)
    assert isinstance(canonical_representative_run_label, str)
    assert isinstance(failure_location, str)
    return AdjudicationRecord(
        run_label=run_label,
        root_cause_id=root_cause_id,
        canonical_representative_run_label=canonical_representative_run_label,
        failure_location=failure_location,
        evidence_references=tuple(references),
        adjudication_sha256=observed,
    )


def _load_signed_json(path: Path) -> dict[str, object]:
    try:
        data = path.read_bytes()
        sidecar = path.with_suffix(path.suffix + ".sha256").read_text(
            encoding="utf-8"
        )
    except OSError as error:
        raise StorageGovernanceError(f"cannot read signed governance state: {path}") from error
    expected_fields = sidecar.strip().split()
    if len(expected_fields) != 2 or expected_fields[1] != path.name:
        raise StorageGovernanceError(f"invalid governance sidecar: {path}")
    observed = hashlib.sha256(data).hexdigest()
    if expected_fields[0] != observed:
        raise StorageGovernanceError(f"governance state SHA-256 mismatch: {path}")
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise StorageGovernanceError(f"invalid governance JSON: {path}") from error
    if not isinstance(payload, dict):
        raise StorageGovernanceError(f"governance state must be a JSON object: {path}")
    return payload


@contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            locking = cast(
                Callable[[int, int, int], None],
                vars(msvcrt)["locking"],
            )
            locking(descriptor, int(vars(msvcrt)["LK_LOCK"]), 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            locking = cast(
                Callable[[int, int, int], None],
                vars(msvcrt)["locking"],
            )
            locking(descriptor, int(vars(msvcrt)["LK_UNLCK"]), 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _reservation_mapping(payload: Mapping[str, object]) -> dict[str, dict[str, object]]:
    if payload.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise StorageGovernanceError("unsupported capacity-ledger schema")
    raw = payload.get("reservations")
    if not isinstance(raw, dict):
        raise StorageGovernanceError("capacity ledger reservations are invalid")
    reservations: dict[str, dict[str, object]] = {}
    for run_label, record in raw.items():
        if not isinstance(run_label, str) or not isinstance(record, dict):
            raise StorageGovernanceError("capacity ledger record is invalid")
        reservations[run_label] = dict(record)
    return reservations


def _positive_ledger_int(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StorageGovernanceError(f"capacity ledger {field} is invalid")
    return value


def _nonnegative_ledger_int(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StorageGovernanceError(f"capacity ledger {field} is invalid")
    return value


def _reservation_can_shrink(
    existing: Mapping[str, object],
    requested: Mapping[str, object],
) -> bool:
    """Permit only a monotonic archive projection reduction for one attempt."""

    fixed_fields = {
        "stage_id",
        "stage_plan_sha256",
        "staging_root_alias",
        "host_root_alias",
        "archive_root_alias",
        "max_active_workspace_bytes",
        "projected_host_growth_bytes",
        "status",
    }
    if any(existing.get(field) != requested.get(field) for field in fixed_fields):
        return False
    try:
        existing_archive = _nonnegative_ledger_int(
            existing,
            "planned_archive_bytes",
        )
        requested_archive = _nonnegative_ledger_int(
            requested,
            "planned_archive_bytes",
        )
    except StorageGovernanceError:
        return False
    return requested_archive <= existing_archive


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    relative_path: str
    byte_count: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    byte_count: int
    modified_ns: int
    changed_ns: int
    inode: int
    device: int


@dataclass(frozen=True, slots=True)
class _VerifiedTree:
    file_count: int
    byte_count: int
    tree_sha256: str
    segment_identities: Mapping[str, tuple[int, int, str]]
    observation: Mapping[str, object]


def _file_snapshot(path: Path) -> _FileSnapshot:
    stat = path.stat()
    return _FileSnapshot(
        byte_count=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        changed_ns=stat.st_ctime_ns,
        inode=stat.st_ino,
        device=stat.st_dev,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identities_tree(
    identities: Mapping[str, _FileIdentity],
) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    for relative in sorted(identities):
        digest.update(_canonical_json(identities[relative].to_dict()))
    return (
        len(identities),
        sum(identity.byte_count for identity in identities.values()),
        digest.hexdigest(),
    )


def _copy_file_with_sha256(
    task: tuple[str, Path, Path, _FileSnapshot],
) -> _FileIdentity:
    relative, source, target, expected_snapshot = task
    if _file_snapshot(source) != expected_snapshot:
        raise StorageGovernanceError(
            f"retention source changed before copy: {relative}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb") as source_handle, target.open("xb") as target_handle:
        while chunk := source_handle.read(8 * 1024 * 1024):
            target_handle.write(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
    shutil.copystat(source, target)
    if (
        byte_count != expected_snapshot.byte_count
        or _file_snapshot(source) != expected_snapshot
    ):
        raise StorageGovernanceError(
            f"retention source changed during copy: {relative}"
        )
    return _FileIdentity(relative, byte_count, digest.hexdigest())


def _windows_path_for_mounted_drive(path: Path) -> str | None:
    resolved = path.resolve()
    parts = resolved.parts
    if (
        len(parts) < 4
        or parts[0] != "/"
        or parts[1] != "mnt"
        or re.fullmatch(r"[a-zA-Z]", parts[2]) is None
    ):
        return None
    suffix = "\\".join(parts[3:])
    return f"{parts[2].upper()}:\\{suffix}"


def _native_copy_segment(
    *,
    source: Path,
    destination: Path,
    workers: int,
) -> dict[str, object]:
    source_windows = _windows_path_for_mounted_drive(source)
    destination_windows = _windows_path_for_mounted_drive(destination)
    if source_windows is None or destination_windows is None:
        raise StorageGovernanceError(
            "Windows-native copy requires mounted Windows source and destination"
        )
    executable = shutil.which("robocopy.exe")
    if executable is None:
        raise StorageGovernanceError(
            "Windows-native copy requires robocopy.exe"
        )
    destination.mkdir(parents=True, exist_ok=True)
    command = (
        executable,
        source_windows,
        destination_windows,
        "/E",
        "/COPY:DAT",
        "/DCOPY:DAT",
        "/R:1",
        "/W:1",
        f"/MT:{workers}",
        "/J",
        "/XJ",
        "/NFL",
        "/NDL",
        "/NJH",
        "/NJS",
        "/NP",
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed_seconds = time.perf_counter() - started
    if completed.returncode > 7:
        raise StorageGovernanceError(
            "Windows-native copy failed: "
            f"exit={completed.returncode}, stderr={completed.stderr.strip()}, "
            f"stdout={completed.stdout.strip()}"
        )
    return {
        "backend": "windows_robocopy",
        "source": str(source),
        "destination": str(destination),
        "workers": workers,
        "elapsed_seconds": elapsed_seconds,
        "exit_code": completed.returncode,
    }


def _native_verify_retention_tree(
    *,
    root: Path,
    segments: tuple[RetentionSegment, ...],
    expected_trees: tuple[ExpectedSegmentTree, ...],
    workers: int,
) -> _VerifiedTree:
    root_windows = _windows_path_for_mounted_drive(root)
    helper = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "hash_retention_tree_windows.py"
    )
    helper_windows = _windows_path_for_mounted_drive(helper)
    python_windows = shutil.which("python.exe")
    if (
        os.name == "nt"
        or root_windows is None
        or helper_windows is None
        or python_windows is None
        or not helper.is_file()
    ):
        raise StorageGovernanceError(
            "Windows-native target verification is unavailable"
        )
    command = [
        python_windows,
        helper_windows,
        "--root",
        root_windows,
        "--workers",
        str(workers),
    ]
    for segment in segments:
        command.extend(
            ("--segment", f"{segment.segment_id}={segment.logical_prefix}")
        )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise StorageGovernanceError(
            "Windows-native target verification failed: "
            f"exit={completed.returncode}, stderr={completed.stderr.strip()}, "
            f"stdout={completed.stdout.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise StorageGovernanceError(
            "Windows-native target verification returned invalid JSON"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != "experiment-retention-native-tree-verification-v1"
    ):
        raise StorageGovernanceError(
            "Windows-native target verification schema is invalid"
        )
    raw_segments = payload.get("segments")
    expected_by_id = {
        expected.segment_id: expected for expected in expected_trees
    }
    if (
        not isinstance(raw_segments, dict)
        or set(raw_segments) != set(expected_by_id)
    ):
        raise StorageGovernanceError(
            "Windows-native target segment identities are invalid"
        )
    segment_identities: dict[str, tuple[int, int, str]] = {}
    for segment_id, expected in expected_by_id.items():
        raw_identity = raw_segments.get(segment_id)
        if not isinstance(raw_identity, dict):
            raise StorageGovernanceError(
                "Windows-native target segment identity is invalid: "
                f"{segment_id}"
            )
        observed = (
            raw_identity.get("file_count"),
            raw_identity.get("byte_count"),
            raw_identity.get("tree_sha256"),
        )
        required = (
            expected.file_count,
            expected.byte_count,
            expected.tree_sha256,
        )
        if observed != required:
            raise StorageGovernanceError(
                "retained segment differs from signed inventory: "
                f"{segment_id}"
            )
        segment_identities[segment_id] = required
    file_count = payload.get("file_count")
    byte_count = payload.get("byte_count")
    tree_sha256 = payload.get("tree_sha256")
    if (
        isinstance(file_count, bool)
        or not isinstance(file_count, int)
        or file_count < 0
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or not isinstance(tree_sha256, str)
        or _SHA256.fullmatch(tree_sha256) is None
        or file_count != sum(identity[0] for identity in segment_identities.values())
        or byte_count != sum(identity[1] for identity in segment_identities.values())
    ):
        raise StorageGovernanceError(
            "Windows-native target tree identity is invalid"
        )
    return _VerifiedTree(
        file_count=file_count,
        byte_count=byte_count,
        tree_sha256=tree_sha256,
        segment_identities=segment_identities,
        observation={
            "backend": "windows_python_sha256",
            "workers": workers,
            "elapsed_seconds": elapsed_seconds,
            "verifier_pid": payload.get("verifier_pid"),
            "file_count": file_count,
            "byte_count": byte_count,
            "tree_sha256": tree_sha256,
        },
    )


def _native_verify_tree_mappings(
    *,
    root: Path,
    mappings: tuple[tuple[str, str], ...],
    workers: int,
) -> dict[str, tuple[int, int, str]]:
    """Rehash many destination generations in one Windows-native process."""

    root_windows = _windows_path_for_mounted_drive(root)
    helper = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "hash_retention_tree_windows.py"
    )
    helper_windows = _windows_path_for_mounted_drive(helper)
    python_windows = shutil.which("python.exe")
    if (
        os.name == "nt"
        or root_windows is None
        or helper_windows is None
        or python_windows is None
        or not helper.is_file()
    ):
        raise StorageGovernanceError(
            "Windows-native mapping verification is unavailable"
        )
    request = {
        "root": root_windows,
        "workers": workers,
        "mappings": [
            {
                "logical_id": logical_id,
                "relative_path": relative_path,
            }
            for logical_id, relative_path in mappings
        ],
    }
    completed = subprocess.run(
        (python_windows, helper_windows, "--mapping-stdin"),
        input=json.dumps(request),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise StorageGovernanceError(
            "Windows-native mapping verification failed: "
            f"exit={completed.returncode}, stderr={completed.stderr.strip()}, "
            f"stdout={completed.stdout.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise StorageGovernanceError(
            "Windows-native mapping verification returned invalid JSON"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != "experiment-retention-native-mapping-verification-v1"
        or payload.get("workers") != workers
    ):
        raise StorageGovernanceError(
            "Windows-native mapping verification schema is invalid"
        )
    raw_identities = payload.get("mappings")
    expected_ids = {logical_id for logical_id, _relative_path in mappings}
    if (
        not isinstance(raw_identities, dict)
        or set(raw_identities) != expected_ids
        or len(expected_ids) != len(mappings)
    ):
        raise StorageGovernanceError(
            "Windows-native mapping identities are invalid"
        )
    identities: dict[str, tuple[int, int, str]] = {}
    for logical_id in expected_ids:
        raw_identity = raw_identities.get(logical_id)
        if not isinstance(raw_identity, dict):
            raise StorageGovernanceError(
                "Windows-native mapping identity is invalid: "
                f"{logical_id}"
            )
        file_count = raw_identity.get("file_count")
        byte_count = raw_identity.get("byte_count")
        tree_sha256 = raw_identity.get("tree_sha256")
        if (
            isinstance(file_count, bool)
            or not isinstance(file_count, int)
            or file_count < 0
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not isinstance(tree_sha256, str)
            or _SHA256.fullmatch(tree_sha256) is None
        ):
            raise StorageGovernanceError(
                "Windows-native mapping identity is invalid: "
                f"{logical_id}"
            )
        identities[logical_id] = (
            file_count,
            byte_count,
            tree_sha256,
        )
    return identities


def _hash_source_file(
    task: tuple[str, Path, _FileSnapshot],
) -> _FileIdentity:
    relative, source, expected_snapshot = task
    if _file_snapshot(source) != expected_snapshot:
        raise StorageGovernanceError(
            f"retention source changed before verification: {relative}"
        )
    identity = _FileIdentity(
        relative,
        expected_snapshot.byte_count,
        _file_sha256(source),
    )
    if _file_snapshot(source) != expected_snapshot:
        raise StorageGovernanceError(
            f"retention source changed during verification: {relative}"
        )
    return identity


def _verify_target_file(
    task: tuple[str, Path, _FileIdentity],
) -> _FileIdentity:
    relative, target, expected = task
    if (
        not target.is_file()
        or target.stat().st_size != expected.byte_count
        or _file_sha256(target) != expected.sha256
    ):
        raise StorageGovernanceError(
            f"retained file verification failed: {relative}"
        )
    return expected


def _hash_target_file(
    task: tuple[str, Path],
) -> _FileIdentity:
    relative, target = task
    if not target.is_file():
        raise StorageGovernanceError(
            f"retention destination file is missing: {relative}"
        )
    snapshot = _file_snapshot(target)
    identity = _FileIdentity(relative, snapshot.byte_count, _file_sha256(target))
    if _file_snapshot(target) != snapshot:
        raise StorageGovernanceError(
            f"retention destination changed during verification: {relative}"
        )
    return identity


def _parallel_map[TaskT](
    function: Callable[[TaskT], _FileIdentity],
    tasks: tuple[TaskT, ...],
    *,
    workers: int,
) -> tuple[_FileIdentity, ...]:
    if workers == 1:
        return tuple(function(task) for task in tasks)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return tuple(executor.map(function, tasks))


def _safe_root_relative_path(
    root: Path,
    relative_path: str,
    *,
    description: str,
) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or str(relative) == "."
        or ".." in relative.parts
    ):
        raise StorageGovernanceError(f"{description} path is unsafe")
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*relative.parts).resolve()
    if not candidate.is_relative_to(resolved_root) or candidate == resolved_root:
        raise StorageGovernanceError(f"{description} path escapes its root")
    return candidate


def _segment_files(segment: RetentionSegment) -> tuple[tuple[Path, str], ...]:
    source = segment.source_path.resolve()
    if not source.is_dir():
        raise StorageGovernanceError(
            f"retention source segment is missing: {segment.source_path}"
        )
    files: list[tuple[Path, str]] = []
    prefix = PurePosixPath(segment.logical_prefix)
    for item in sorted(source.rglob("*"), key=lambda path: path.as_posix()):
        if item.is_symlink():
            raise StorageGovernanceError(f"retention source contains a symlink: {item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise StorageGovernanceError(
                f"retention source contains an unsupported entry: {item}"
            )
        relative = PurePosixPath(item.relative_to(source).as_posix())
        logical = relative if str(prefix) == "." else prefix / relative
        files.append((item, logical.as_posix()))
    return tuple(files)


def _is_duplicate_control_evidence(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    parts = tuple(part.casefold() for part in path.parts)
    name = path.name.casefold()
    return (
        bool(parts)
        and parts[0]
        in {
            "control",
            "review",
            "reviews",
            "log",
            "logs",
            "service_log",
            "service_logs",
            "failure",
            "failures",
        }
    ) or (
        "manifest" in name
        or "checksum" in name
        or "failure" in name
        or name.endswith(".sha256")
    )


def _tree_identity(path: Path) -> tuple[int, int, str]:
    files: dict[str, _FileIdentity] = {}
    for item in sorted(path.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if item.is_symlink():
            raise StorageGovernanceError(f"retained tree contains a symlink: {item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise StorageGovernanceError(f"retained tree entry is unsupported: {item}")
        relative = item.relative_to(path).as_posix()
        files[relative] = _FileIdentity(
            relative,
            item.stat().st_size,
            _file_sha256(item),
        )
    return _identities_tree(files)


def _protected_rebuildable_reason(path: Path) -> str | None:
    """Return why a path is categorically ineligible for automatic cleanup."""

    if any(_RUN_LABEL.fullmatch(part) is not None for part in path.parts):
        return "unsealed_run"
    protected_tokens = (
        "manifest",
        "registry",
        "review",
        "checksum",
        "source_snapshot",
        "sealed_source",
    )
    for item in (path, *path.rglob("*")):
        name = item.name.casefold()
        if name == ".git":
            return "git_repository"
        if any(token in name for token in protected_tokens) or (
            item.is_file() and name.endswith(".sha256")
        ):
            return "protected_evidence"
    return None


def compute_tree_sha256(path: Path) -> str:
    """Return the canonical tree identity used by governance receipts."""

    return _tree_identity(path)[2]


def compute_tree_identity(path: Path) -> tuple[int, int, str]:
    """Return canonical file count, byte count, and tree SHA-256."""

    return _tree_identity(path)


def compute_tree_identity_parallel(
    path: Path,
    *,
    workers: int,
) -> tuple[int, int, str]:
    """Return canonical tree identity with bounded parallel file hashing."""

    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 32
    ):
        raise ValueError("tree identity workers must be between 1 and 32")
    source = path.resolve()
    if not source.is_dir():
        raise StorageGovernanceError(f"tree identity source is missing: {path}")
    files: dict[str, Path] = {}
    for item in sorted(source.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if item.is_symlink():
            raise StorageGovernanceError(f"retained tree contains a symlink: {item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise StorageGovernanceError(
                f"retained tree entry is unsupported: {item}"
            )
        files[item.relative_to(source).as_posix()] = item
    tasks = tuple(
        (relative, item, _file_snapshot(item))
        for relative, item in files.items()
    )
    identities = {
        identity.relative_path: identity
        for identity in _parallel_map(
            _hash_source_file,
            tasks,
            workers=workers,
        )
    }
    return _identities_tree(identities)


def _verify_lifecycle_content_inventory(
    inventory_path: Path,
    *,
    run_label: str,
    archive_path: Path,
) -> tuple[str, int, int, str]:
    """Bind a lifecycle inventory tree to an already verified v2 archive."""

    inventory = _load_signed_json(inventory_path)
    files = inventory.get("files")
    if (
        inventory.get("schema_version") != "experiment-content-inventory-v1"
        or inventory.get("run_label") != run_label
        or not isinstance(files, list)
        or not files
    ):
        raise StorageGovernanceError("lifecycle content inventory is invalid")
    if any(not isinstance(item, Mapping) for item in files):
        raise StorageGovernanceError("lifecycle content inventory entry is invalid")
    mapping_files = cast(list[Mapping[str, object]], files)
    expected: dict[str, tuple[int, str]] = {}
    digest = hashlib.sha256()
    for raw_item in sorted(
        mapping_files,
        key=lambda item: str(item.get("relative_path", "")),
    ):
        relative = PurePosixPath(str(raw_item.get("relative_path", "")))
        byte_count = raw_item.get("byte_count")
        sha256 = str(raw_item.get("sha256", ""))
        relative_text = relative.as_posix()
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or relative_text in expected
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or _SHA256.fullmatch(sha256) is None
        ):
            raise StorageGovernanceError("lifecycle content inventory entry is invalid")
        expected[relative_text] = (byte_count, sha256)
        digest.update(relative_text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(byte_count).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256.encode("ascii"))
        digest.update(b"\n")
    expected_tree_sha256 = digest.hexdigest()
    if inventory.get("source_tree_sha256") != expected_tree_sha256:
        raise StorageGovernanceError("lifecycle content inventory tree differs")
    observed: dict[str, Path] = {}
    for path in sorted(archive_path.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise StorageGovernanceError("lifecycle archive contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise StorageGovernanceError("lifecycle archive entry is unsupported")
        observed[path.relative_to(archive_path).as_posix()] = path
    if set(observed) != set(expected):
        raise StorageGovernanceError("lifecycle archive file set differs")
    for observed_relative, path in observed.items():
        byte_count, sha256 = expected[observed_relative]
        if path.stat().st_size != byte_count or _file_sha256(path) != sha256:
            raise StorageGovernanceError(
                f"lifecycle archive content differs: {observed_relative}"
            )
    return (
        expected_tree_sha256,
        len(expected),
        sum(byte_count for byte_count, _sha256 in expected.values()),
        _file_sha256(inventory_path),
    )


def _maintenance_request_for_roots(
    policy: GovernancePolicy,
    roots: tuple[Path, ...],
) -> SweepRequest:
    resolved_roots = tuple(dict.fromkeys(root.resolve() for root in roots))
    assets: list[RebuildableAsset] = []
    for root in resolved_roots:
        for relative_path, raw_kind in policy.maintenance_allowlist:
            candidate = _safe_root_relative_path(
                root,
                relative_path,
                description="maintenance allowlist",
            )
            if not candidate.exists():
                continue
            assets.append(
                RebuildableAsset(
                    path=candidate,
                    kind=RebuildableKind(raw_kind),
                    expected_tree_sha256=_tree_identity(candidate)[2],
                )
            )
    return SweepRequest(
        allowed_roots=resolved_roots,
        assets=tuple(assets),
        apply=False,
    )


def _published_generation_matches(
    destination: Path,
    *,
    source_identities: Mapping[str, _FileIdentity],
    keep: set[str],
    projection: Mapping[str, object] | None,
) -> bool:
    if not destination.is_dir() or destination.is_symlink():
        return False
    actual_files: set[str] = set()
    for path in destination.rglob("*"):
        if path.is_symlink():
            return False
        if path.is_file():
            actual_files.add(path.relative_to(destination).as_posix())
    expected_files = set(keep)
    if projection is not None:
        manifest_path = destination / "retention_projection_manifest.json"
        try:
            observed_projection = _load_signed_json(manifest_path)
        except StorageGovernanceError:
            return False
        if observed_projection != projection:
            return False
        expected_files.update(
            {
                "retention_projection_manifest.json",
                "retention_projection_manifest.json.sha256",
            }
        )
    if actual_files != expected_files:
        return False
    for relative in keep:
        target = destination.joinpath(*PurePosixPath(relative).parts)
        identity = source_identities[relative]
        if (
            target.stat().st_size != identity.byte_count
            or _file_sha256(target) != identity.sha256
        ):
            return False
    return True


def _hash_full_destination(
    destination: Path,
    *,
    expected_files: set[str] | None,
    workers: int,
) -> dict[str, _FileIdentity]:
    if not destination.is_dir() or destination.is_symlink():
        raise StorageGovernanceError(
            f"retention destination is not a directory: {destination}"
        )
    actual_files: set[str] = set()
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise StorageGovernanceError(
                f"retention destination contains a symlink: {path}"
            )
        if path.is_file():
            actual_files.add(path.relative_to(destination).as_posix())
        elif not path.is_dir():
            raise StorageGovernanceError(
                f"retention destination entry is unsupported: {path}"
            )
    if expected_files is not None and actual_files != expected_files:
        raise StorageGovernanceError(
            f"retention destination file set differs: {destination}"
        )
    tasks = tuple(
        (
            relative,
            destination.joinpath(*PurePosixPath(relative).parts),
        )
        for relative in sorted(actual_files)
    )
    return {
        identity.relative_path: identity
        for identity in _parallel_map(
            _hash_target_file,
            tasks,
            workers=workers,
        )
    }


def _verify_full_destination(
    destination: Path,
    *,
    source_identities: Mapping[str, _FileIdentity],
    workers: int,
) -> None:
    target_identities = _hash_full_destination(
        destination,
        expected_files=set(source_identities),
        workers=workers,
    )
    if target_identities != source_identities:
        raise StorageGovernanceError(
            f"retained file verification failed: {destination}"
        )


def _complete_logical_segment_paths(
    *,
    segments: tuple[RetentionSegment, ...],
    identities: Mapping[str, _FileIdentity],
    known_paths: Mapping[str, tuple[str, str]],
) -> dict[str, tuple[str, str]]:
    completed = dict(known_paths)
    for logical_path in identities:
        if logical_path in completed:
            continue
        logical_parts = PurePosixPath(logical_path).parts
        matches: list[tuple[str, str]] = []
        for segment in segments:
            prefix = PurePosixPath(segment.logical_prefix)
            prefix_parts = () if str(prefix) == "." else prefix.parts
            if logical_parts[: len(prefix_parts)] != prefix_parts:
                continue
            relative_parts = logical_parts[len(prefix_parts) :]
            if relative_parts:
                matches.append(
                    (segment.segment_id, PurePosixPath(*relative_parts).as_posix())
                )
        if len(matches) != 1:
            raise StorageGovernanceError(
                "retained file does not map to exactly one segment: "
                f"{logical_path}"
            )
        completed[logical_path] = matches[0]
    if set(completed) != set(identities):
        raise StorageGovernanceError(
            "retention segment mapping differs from retained files"
        )
    return completed


def _verify_expected_segment_trees(
    *,
    expected_trees: tuple[ExpectedSegmentTree, ...],
    logical_segment_paths: Mapping[str, tuple[str, str]],
    identities: Mapping[str, _FileIdentity],
) -> None:
    if not expected_trees:
        return
    by_segment: dict[str, dict[str, _FileIdentity]] = {
        expected.segment_id: {} for expected in expected_trees
    }
    for logical_path, identity in identities.items():
        segment_id, segment_relative = logical_segment_paths[logical_path]
        by_segment[segment_id][segment_relative] = _FileIdentity(
            relative_path=segment_relative,
            byte_count=identity.byte_count,
            sha256=identity.sha256,
        )
    for expected in expected_trees:
        observed = _identities_tree(by_segment[expected.segment_id])
        required = (
            expected.file_count,
            expected.byte_count,
            expected.tree_sha256,
        )
        if observed != required:
            raise StorageGovernanceError(
                "retained segment differs from signed inventory: "
                f"{expected.segment_id}"
            )


def _registry_records(payload: Mapping[str, object]) -> list[dict[str, object]]:
    if payload.get("schema_version") != "experiment-retention-registry-v2":
        raise StorageGovernanceError("unsupported retention-registry schema")
    raw = payload.get("records")
    if not isinstance(raw, list):
        raise StorageGovernanceError("retention registry records are invalid")
    records: list[dict[str, object]] = []
    for record in raw:
        if not isinstance(record, dict):
            raise StorageGovernanceError("retention registry record is invalid")
        records.append(dict(record))
    return records


def _strict_registry_int(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StorageGovernanceError(f"retention registry {field} is invalid")
    return value


class ExperimentStorageGovernance:
    """Deep storage-governance module used by experiment runners."""

    def __init__(
        self,
        *,
        policy: GovernancePolicy,
        locator: StorageRootLocator,
        state_root: Path,
        free_space: Callable[[Path], int],
        volume_probe: Callable[[Path], VolumeIdentity],
        retention_state_root: Path | None = None,
        legacy_registry_path: Path | None = None,
        keeper_reference_scanner: Callable[[Path], tuple[str, ...]] | None = None,
        isolated_rebuild_verifier: Callable[[RebuildableAsset], bool] | None = None,
    ) -> None:
        if not state_root.is_absolute():
            raise ValueError("storage governance state_root must be absolute")
        if retention_state_root is not None and not retention_state_root.is_absolute():
            raise ValueError("retention governance state_root must be absolute")
        self.policy = policy
        self.locator = locator
        self.state_root = state_root
        self.retention_state_root = retention_state_root or state_root
        self._free_space = free_space
        self._volume_probe = volume_probe
        self._legacy_registry_path = legacy_registry_path
        self._keeper_reference_scanner = keeper_reference_scanner
        self._isolated_rebuild_verifier = isolated_rebuild_verifier

    def preflight_run(self, request: StartRequest) -> StartPermit:
        """Reserve capacity before any run output directory is created."""

        aliases = (
            request.staging_root_alias,
            request.host_root_alias,
            request.archive_root_alias,
        )
        roots: dict[str, StorageRoot] = {}
        measurement_errors: dict[str, str] = {}
        for alias in aliases:
            try:
                root = self.locator.resolve(alias)
                observed_volume = self._volume_probe(root.absolute_path)
                if observed_volume != root.volume:
                    raise StorageGovernanceError(
                        f"storage root volume identity mismatch: {alias}"
                    )
                roots[alias] = root
            except (KeyError, OSError, RuntimeError, StorageGovernanceError) as error:
                measurement_errors[alias] = f"{type(error).__name__}: {error}"
        ledger_path = self.state_root / "capacity_ledger.json"
        lock_path = self.state_root / "capacity_ledger.lock"
        with _state_lock(lock_path):
            if ledger_path.exists():
                reservations = _reservation_mapping(_load_signed_json(ledger_path))
            else:
                reservations = {}
            existing = reservations.get(request.run_label)
            maintenance_audit = self.sweep_rebuildable_assets(
                _maintenance_request_for_roots(
                    self.policy,
                    (
                        roots[request.staging_root_alias].absolute_path
                        if request.staging_root_alias in roots
                        else self.state_root,
                    ),
                )
            )
            request_identity = {
                "stage_id": request.stage_id,
                "stage_plan_sha256": request.stage_plan_sha256,
                "staging_root_alias": request.staging_root_alias,
                "host_root_alias": request.host_root_alias,
                "archive_root_alias": request.archive_root_alias,
                "planned_archive_bytes": request.planned_archive_bytes,
                "max_active_workspace_bytes": request.max_active_workspace_bytes,
                "projected_host_growth_bytes": request.projected_host_growth_bytes,
                "status": "reserved",
            }
            permit_path = self.state_root / "permits" / f"{request.run_label}.json"
            identity_errors: list[str] = []
            if request.archive_root_alias != "e_archive":
                identity_errors.append(
                    "new experiment archives must use e_archive; D roles are read-only"
                )
            unreconciled = sorted(
                label
                for label, reservation in reservations.items()
                if label != request.run_label and reservation.get("status") == "reserved"
            )
            if unreconciled:
                identity_errors.append(
                    "previous top-level storage permits are not reconciled: "
                    + ", ".join(unreconciled)
                )
            if existing is not None:
                if existing != request_identity and not _reservation_can_shrink(
                    existing,
                    request_identity,
                ):
                    identity_errors.append(
                        f"capacity reservation conflicts for {request.run_label}"
                    )
                if not permit_path.is_file():
                    identity_errors.append(
                        "existing capacity reservation has no immutable start permit"
                    )
                else:
                    try:
                        existing_permit = _load_signed_json(permit_path)
                    except StorageGovernanceError as error:
                        identity_errors.append(
                            f"existing start permit verification failed: {error}"
                        )
                    else:
                        if (
                            existing_permit.get("schema_version")
                            != POLICY_SCHEMA_VERSION
                            or existing_permit.get("run_label") != request.run_label
                            or existing_permit.get("stage_plan_sha256")
                            != request.stage_plan_sha256
                            or existing_permit.get("status") != "reserved"
                        ):
                            identity_errors.append(
                                "existing immutable start permit identity differs"
                            )
            elif request.run_dir.exists():
                identity_errors.append(
                    f"governed run path already exists without a permit: {request.run_dir}"
                )
            if existing is None:
                try:
                    if self._run_label_is_registered(request.run_label):
                        identity_errors.append(
                            f"governed run label is already retained: {request.run_label}"
                        )
                except StorageGovernanceError as error:
                    identity_errors.append(
                        f"retention registry verification failed: {error}"
                    )

            reserved_staging = 0
            reserved_host = 0
            reserved_archive = 0
            for reserved_label, reservation in reservations.items():
                if reserved_label == request.run_label:
                    continue
                if reservation.get("status") != "reserved":
                    continue
                if reservation.get("staging_root_alias") == request.staging_root_alias:
                    reserved_staging += _positive_ledger_int(
                        reservation,
                        "max_active_workspace_bytes",
                    )
                if reservation.get("host_root_alias") == request.host_root_alias:
                    reserved_host += _positive_ledger_int(
                        reservation,
                        "projected_host_growth_bytes",
                    )
                if reservation.get("archive_root_alias") == request.archive_root_alias:
                    reserved_archive += _nonnegative_ledger_int(
                        reservation,
                        "planned_archive_bytes",
                    )

            staging_workspace = request.max_active_workspace_bytes
            if request.stage_id == "stage05.2":
                staging_workspace = max(
                    staging_workspace,
                    self.policy.stage052_active_workspace_floor_bytes,
                )
            required = {
                request.staging_root_alias: (
                    self.policy.staging_safety_reserve_bytes
                    + reserved_staging
                    + staging_workspace
                ),
                request.host_root_alias: (
                    self.policy.host_reserve_bytes
                    + reserved_host
                    + request.projected_host_growth_bytes
                ),
                request.archive_root_alias: (
                    self.policy.archive_reserve_bytes
                    + reserved_archive
                    + request.planned_archive_bytes
                ),
            }
            free: dict[str, int] = {}
            for alias in aliases:
                measured_root = roots.get(alias)
                if measured_root is None:
                    continue
                try:
                    value = self._free_space(measured_root.absolute_path)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise ValueError("measurement is not a non-negative integer")
                    free[alias] = value
                except (OSError, RuntimeError, ValueError) as error:
                    measurement_errors[alias] = f"{type(error).__name__}: {error}"
            deficits = {
                alias: max(0, required[alias] - free.get(alias, 0))
                for alias in aliases
                if alias in measurement_errors
                or free.get(alias, 0) < required[alias]
            }
            observed_at = datetime.now(UTC).isoformat()
            observation: dict[str, object] = {
                "schema_version": POLICY_SCHEMA_VERSION,
                "observed_at_utc": observed_at,
                "stage_id": request.stage_id,
                "run_label": request.run_label,
                "run_dir": str(request.run_dir),
                "stage_plan_sha256": request.stage_plan_sha256,
                "free_bytes_by_alias": dict(sorted(free.items())),
                "required_bytes_by_alias": dict(sorted(required.items())),
                "volume_identities_by_alias": {
                    alias: roots[alias].volume.to_dict()
                    for alias in sorted(roots)
                },
                "deficits_by_alias": dict(sorted(deficits.items())),
                "measurement_errors_by_alias": dict(
                    sorted(measurement_errors.items())
                ),
                "identity_errors": identity_errors,
                "maintenance_audit_sha256": (
                    maintenance_audit.receipt_sha256
                ),
                "passed": (
                    not deficits
                    and not measurement_errors
                    and not identity_errors
                ),
            }
            timestamp = observed_at.replace(":", "").replace("+", "_")
            observation_path = (
                self.state_root
                / "observations"
                / f"{request.run_label}-{timestamp}.json"
            )
            observation_sha256 = _write_signed_json(observation_path, observation)
            if deficits or measurement_errors or identity_errors:
                raise StorageCapacityError(observation=observation)
            reservations[request.run_label] = request_identity
            ledger_payload = {
                "schema_version": POLICY_SCHEMA_VERSION,
                "reservations": dict(sorted(reservations.items())),
            }
            _write_signed_json(ledger_path, ledger_payload)
            if existing is None:
                permit_payload = {
                    "schema_version": POLICY_SCHEMA_VERSION,
                    "run_label": request.run_label,
                    "stage_plan_sha256": request.stage_plan_sha256,
                    "observation_sha256": observation_sha256,
                    "maintenance_audit_sha256": (
                        maintenance_audit.receipt_sha256
                    ),
                    "maintenance_audit_path": str(
                        maintenance_audit.receipt_path
                    ),
                    "status": "reserved",
                }
                _write_signed_json(permit_path, permit_payload)
            return StartPermit(
                run_label=request.run_label,
                stage_plan_sha256=request.stage_plan_sha256,
                observation_sha256=observation_sha256,
                observation_path=observation_path,
                maintenance_audit_sha256=(
                    maintenance_audit.receipt_sha256
                ),
                maintenance_audit_path=maintenance_audit.receipt_path,
                permit_path=permit_path,
            )

    def reconcile_permit(
        self,
        run_label: str,
        *,
        outcome: str,
        evidence_sha256: str,
    ) -> Path:
        """Release one reservation only after an explicit signed audit decision."""

        if _RUN_LABEL.fullmatch(run_label) is None:
            raise ValueError("permit run label is invalid")
        if outcome not in {"retained", "aborted_audited"}:
            raise ValueError("permit reconciliation outcome is invalid")
        if _SHA256.fullmatch(evidence_sha256) is None:
            raise ValueError("permit reconciliation evidence SHA-256 is invalid")
        ledger_path = self.state_root / "capacity_ledger.json"
        lock_path = self.state_root / "capacity_ledger.lock"
        receipt_path = self.state_root / "permit_reconciliations" / f"{run_label}.json"
        permit_path = self.state_root / "permits" / f"{run_label}.json"
        if not permit_path.is_file():
            raise StorageGovernanceError("capacity permit receipt does not exist")
        permit_sha256 = _file_sha256(permit_path)
        with _state_lock(lock_path):
            if not ledger_path.exists():
                raise StorageGovernanceError("capacity ledger does not exist")
            reservations = _reservation_mapping(_load_signed_json(ledger_path))
            reservation = reservations.get(run_label)
            if reservation is None:
                raise StorageGovernanceError("capacity permit does not exist")
            if reservation.get("status") == "reconciled":
                if (
                    reservation.get("outcome") != outcome
                    or reservation.get("evidence_sha256") != evidence_sha256
                    or reservation.get("storage_permit_sha256") != permit_sha256
                ):
                    raise StorageGovernanceError(
                        "capacity permit reconciliation identity differs"
                    )
                _write_signed_json(
                    receipt_path,
                    {
                        "schema_version": "experiment-capacity-reconciliation-v1",
                        "run_label": run_label,
                        "outcome": outcome,
                        "evidence_sha256": evidence_sha256,
                        "storage_permit_sha256": permit_sha256,
                        "reconciled_at_utc": datetime.now(UTC).isoformat(),
                    },
                )
                return receipt_path
            if reservation.get("status") != "reserved":
                raise StorageGovernanceError("capacity permit is not reserved")
            reconciled = dict(reservation)
            reconciled["status"] = "reconciled"
            reconciled["outcome"] = outcome
            reconciled["evidence_sha256"] = evidence_sha256
            reconciled["storage_permit_sha256"] = permit_sha256
            reservations[run_label] = reconciled
            receipt = {
                "schema_version": "experiment-capacity-reconciliation-v1",
                "run_label": run_label,
                "outcome": outcome,
                "evidence_sha256": evidence_sha256,
                "storage_permit_sha256": permit_sha256,
                "reconciled_at_utc": datetime.now(UTC).isoformat(),
            }
            _write_signed_json(
                ledger_path,
                {
                    "schema_version": POLICY_SCHEMA_VERSION,
                    "reservations": dict(sorted(reservations.items())),
                },
            )
            _write_signed_json(receipt_path, receipt)
        return receipt_path

    def write_lifecycle_retention_binding(
        self,
        receipt: RetentionReceipt,
        *,
        lifecycle_retention_class: str,
        storage_permit_sha256: str,
        close_performance: Mapping[str, object],
        content_inventory_path: Path,
    ) -> Path:
        """Bind a verified v2 archive generation to lifecycle-v3 close input."""

        if lifecycle_retention_class not in {
            "published_full",
            "current_accepted_full",
        }:
            raise StorageGovernanceError(
                "only full accepted retention can use a lifecycle binding receipt"
            )
        if (
            receipt.retention_class != RetentionClass.ACCEPTED_FULL
            or _SHA256.fullmatch(storage_permit_sha256) is None
            or not isinstance(close_performance, Mapping)
        ):
            raise StorageGovernanceError("lifecycle retention binding is invalid")
        archive_root = self.locator.resolve("e_archive").absolute_path.resolve()
        archive_path = receipt.archive_path.resolve()
        if not archive_path.is_relative_to(archive_root) or archive_path == archive_root:
            raise StorageGovernanceError("lifecycle archive is outside e_archive")
        registry_path = self.retention_state_root / "retention_registry_v2.json"
        if _file_sha256(registry_path) != receipt.registry_sha256:
            raise StorageGovernanceError("lifecycle retention registry differs")
        records = self._load_registry()
        matching = [
            record
            for record in records
            if record.get("run_label") == receipt.run_label
            and record.get("generation") == receipt.generation
        ]
        if not matching:
            raise StorageGovernanceError("lifecycle retention generation is absent")
        relative_archive = archive_path.relative_to(archive_root).as_posix()
        if any(
            record.get("archive_root_alias") != "e_archive"
            or record.get("archive_relative_path") != relative_archive
            or record.get("tree_sha256") != receipt.tree_sha256
            or record.get("file_count") != receipt.kept_file_count
            or record.get("byte_count") != receipt.kept_bytes
            or record.get("verification_status") != "verified"
            for record in matching
        ):
            raise StorageGovernanceError("lifecycle retention generation differs")
        replay_relatives = {
            str(record.get("replay_receipt_relative_path", ""))
            for record in matching
        }
        if len(replay_relatives) != 1 or not next(iter(replay_relatives)):
            raise StorageGovernanceError("lifecycle retention replay is missing")
        replay_relative = PurePosixPath(next(iter(replay_relatives)))
        if replay_relative.is_absolute() or ".." in replay_relative.parts:
            raise StorageGovernanceError("lifecycle retention replay path is unsafe")
        replay_path = self.retention_state_root.joinpath(*replay_relative.parts)
        replay = _load_signed_json(replay_path)
        if (
            _file_sha256(replay_path) != receipt.replay_receipt_sha256
            or replay.get("status") != "passed"
        ):
            raise StorageGovernanceError("lifecycle retention replay differs")
        (
            lifecycle_tree_sha256,
            lifecycle_file_count,
            lifecycle_byte_count,
            content_inventory_sha256,
        ) = _verify_lifecycle_content_inventory(
            content_inventory_path,
            run_label=receipt.run_label,
            archive_path=archive_path,
        )
        if (
            lifecycle_file_count != receipt.kept_file_count
            or lifecycle_byte_count != receipt.kept_bytes
        ):
            raise StorageGovernanceError(
                "lifecycle inventory counts differ from retained generation"
            )
        permit_path = self.state_root / "permits" / f"{receipt.run_label}.json"
        if (
            not permit_path.is_file()
            or _file_sha256(permit_path) != storage_permit_sha256
        ):
            raise StorageGovernanceError("lifecycle storage permit differs")
        path = (
            self.retention_state_root
            / "lifecycle_retention_receipts"
            / f"{receipt.run_label}.json"
        )
        _write_signed_json(
            path,
            {
                "schema_version": "experiment-lifecycle-retention-binding-v1",
                "run_label": receipt.run_label,
                "generation": receipt.generation,
                "retention_class": lifecycle_retention_class,
                "storage_permit_sha256": storage_permit_sha256,
                "verification_status": "verified",
                "archive_root_alias": "e_archive",
                "archive_relative_path": relative_archive,
                "archive_tree_sha256": lifecycle_tree_sha256,
                "storage_tree_sha256": receipt.tree_sha256,
                "file_count": receipt.kept_file_count,
                "byte_count": receipt.kept_bytes,
                "content_inventory_sha256": content_inventory_sha256,
                "registry_sha256": receipt.registry_sha256,
                "replay_receipt_relative_path": replay_relative.as_posix(),
                "replay_receipt_sha256": receipt.replay_receipt_sha256,
                "verifier_identity_sha256": replay.get(
                    "verifier_identity_sha256"
                ),
                "validator_replay_passed": replay.get(
                    "validator_replay_passed"
                ),
                "objective_replay_passed": replay.get(
                    "objective_replay_passed"
                ),
                "raw_review_replay_passed": replay.get(
                    "raw_review_replay_passed"
                ),
                "close_performance": dict(close_performance),
            },
        )
        return path

    def _observe_archive_capacity(
        self,
        *,
        request: RetentionRequest,
        archive_root: StorageRoot,
        phase: str,
        projected_bytes: int,
    ) -> None:
        required_bytes = self.policy.archive_reserve_bytes
        if phase == "pre_archive":
            required_bytes += projected_bytes
        measurement_error = ""
        free_bytes = 0
        try:
            measured = self._free_space(archive_root.absolute_path)
            if (
                isinstance(measured, bool)
                or not isinstance(measured, int)
                or measured < 0
            ):
                raise ValueError("measurement is not a non-negative integer")
            free_bytes = measured
        except (OSError, RuntimeError, ValueError) as error:
            measurement_error = f"{type(error).__name__}: {error}"
        passed = not measurement_error and free_bytes >= required_bytes
        observation: dict[str, object] = {
            "schema_version": "experiment-retention-capacity-v1",
            "observed_at_utc": datetime.now(UTC).isoformat(),
            "run_label": request.run_label,
            "generation": request.generation,
            "phase": phase,
            "archive_root_alias": request.archive_root_alias,
            "free_bytes": free_bytes,
            "required_bytes": required_bytes,
            "projected_bytes": projected_bytes,
            "measurement_error": measurement_error,
            "passed": passed,
        }
        _write_signed_json(
            self.state_root
            / "retention_capacity_observations"
            / (
                f"{request.run_label}-generation-{request.generation:04d}-"
                f"{phase}.json"
            ),
            observation,
        )
        if not passed:
            raise StorageCapacityError(observation=observation)

    def _publish_incoming_generation(
        self,
        *,
        request: RetentionRequest,
        incoming: Path,
        destination: Path,
    ) -> None:
        """Atomically publish after bounded retries for transient NTFS locks."""

        retry_delays = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
        observation_path = (
            self.state_root
            / "retention_publish_observations"
            / (
                f"{request.run_label}-generation-"
                f"{request.generation:04d}.json"
            )
        )
        errors: list[dict[str, object]] = []
        for attempt in range(1, len(retry_delays) + 2):
            try:
                incoming.replace(destination)
            except PermissionError as error:
                errors.append(
                    {
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "observed_at_utc": datetime.now(UTC).isoformat(),
                    }
                )
                exhausted = attempt > len(retry_delays)
                _write_signed_json(
                    observation_path,
                    {
                        "schema_version": (
                            "experiment-retention-atomic-publish-v1"
                        ),
                        "run_label": request.run_label,
                        "generation": request.generation,
                        "incoming_path": str(incoming),
                        "destination_path": str(destination),
                        "attempts": errors,
                        "status": "failed" if exhausted else "retrying",
                    },
                )
                if exhausted:
                    raise
                time.sleep(retry_delays[attempt - 1])
            else:
                if errors:
                    _write_signed_json(
                        observation_path,
                        {
                            "schema_version": (
                                "experiment-retention-atomic-publish-v1"
                            ),
                            "run_label": request.run_label,
                            "generation": request.generation,
                            "incoming_path": str(incoming),
                            "destination_path": str(destination),
                            "attempts": errors,
                            "successful_attempt": attempt,
                            "status": "recovered",
                        },
                    )
                return

    def retain_run(self, request: RetentionRequest) -> RetentionReceipt:
        """Publish one verified immutable full or reduced retention generation."""

        if request.retention_class == RetentionClass.REBUILDABLE:
            raise StorageGovernanceError(
                "rebuildable assets must use sweep_rebuildable_assets"
            )
        if request.retention_class == RetentionClass.DUPLICATE_FAILURE_REDUCED:
            adjudication = request.adjudication
            if (
                adjudication is None
                or not adjudication.is_signed
                or request.adjudication_path is None
            ):
                raise StorageGovernanceError(
                    "duplicate failure reduction requires a signed adjudication"
                )
            observed_adjudication = load_adjudication_record(
                request.adjudication_path,
                expected_sha256=adjudication.adjudication_sha256,
            )
            if observed_adjudication != adjudication:
                raise StorageGovernanceError(
                    "adjudication payload differs from the retention request"
                )
            if not request.keep_relative_paths:
                raise StorageGovernanceError(
                    "duplicate failure reduction requires explicit retained paths"
                )
            self._verify_canonical_failure(adjudication)
        elif request.adjudication is not None:
            raise StorageGovernanceError(
                "adjudication is valid only for duplicate failure reduction"
            )

        archive_root = self.locator.resolve(request.archive_root_alias)
        self.locator.verify_all(self._volume_probe, (request.archive_root_alias,))
        destination = archive_root.absolute_path.joinpath(
            *PurePosixPath(request.archive_relative_path).parts
        )
        resolved_archive = archive_root.absolute_path.resolve()
        resolved_destination_parent = destination.parent.resolve()
        if not resolved_destination_parent.is_relative_to(resolved_archive):
            raise StorageGovernanceError("retention destination escapes archive root")
        native_segment_ids = {
            segment.segment_id
            for segment in request.segments
            if (
                request.retention_class.is_full
                and request.copy_backend == "auto_native"
                and bool(request.expected_segment_trees)
                and _windows_path_for_mounted_drive(segment.source_path)
                is not None
                and _windows_path_for_mounted_drive(destination) is not None
            )
        }
        native_source_snapshots: dict[str, _FileSnapshot] = {}
        all_files: dict[str, Path] = {}
        logical_segment_paths: dict[str, tuple[str, str]] = {}
        for segment in request.segments:
            if segment.segment_id in native_segment_ids:
                source = segment.source_path.resolve()
                if not source.is_dir() or source.is_symlink():
                    raise StorageGovernanceError(
                        f"retention source segment is missing: {segment.source_path}"
                    )
                native_source_snapshots[segment.segment_id] = _file_snapshot(source)
                continue
            for source_file, logical_path in _segment_files(segment):
                if logical_path in all_files:
                    raise StorageGovernanceError(
                        f"retention segments collide at {logical_path}"
                    )
                all_files[logical_path] = source_file
                logical_segment_paths[logical_path] = (
                    segment.segment_id,
                    source_file.relative_to(segment.source_path.resolve()).as_posix(),
                )

        keep = set(all_files)
        if request.retention_class == RetentionClass.DUPLICATE_FAILURE_REDUCED:
            keep = set(request.keep_relative_paths)
            missing = sorted(keep.difference(all_files))
            if missing:
                raise StorageGovernanceError(
                    f"retention projection references missing paths: {missing}"
                )
            mandatory_control = {
                path
                for path in all_files
                if _is_duplicate_control_evidence(path)
            }
            missing_control = sorted(mandatory_control.difference(keep))
            if missing_control:
                raise StorageGovernanceError(
                    "duplicate retention omits required control evidence: "
                    f"{missing_control}"
                )
            raw_files = {
                path
                for path in all_files
                if PurePosixPath(path).parts
                and PurePosixPath(path).parts[0].casefold() == "raw"
            }
            retained_raw = raw_files.intersection(keep)
            if raw_files and not retained_raw:
                raise StorageGovernanceError(
                    "duplicate retention requires a failure-triggering raw shard"
                )
            selected_shards = {
                (
                    "/".join(PurePosixPath(path).parts[:2])
                    if len(PurePosixPath(path).parts) >= 2
                    else path
                )
                for path in retained_raw
            }
            if len(selected_shards) > 1:
                raise StorageGovernanceError(
                    "duplicate retention must select exactly one representative shard"
                )
            for shard_prefix in selected_shards:
                shard_files = {
                    path
                    for path in raw_files
                    if path == shard_prefix or path.startswith(f"{shard_prefix}/")
                }
                if not shard_files.issubset(keep):
                    raise StorageGovernanceError(
                        "duplicate retention representative shard is incomplete"
                    )
            unsupported = sorted(
                path
                for path in keep
                if path not in mandatory_control and path not in retained_raw
            )
            if unsupported:
                raise StorageGovernanceError(
                    "duplicate retention keeps unsupported evidence paths: "
                    f"{unsupported}"
                )
        source_snapshots = {
            relative: _file_snapshot(path)
            for relative, path in sorted(all_files.items())
        }
        source_identities: dict[str, _FileIdentity] = {}
        verified_full_tree: _VerifiedTree | None = None
        if request.retention_class == RetentionClass.DUPLICATE_FAILURE_REDUCED:
            source_identities = {
                relative: _FileIdentity(
                    relative_path=relative,
                    byte_count=source_snapshots[relative].byte_count,
                    sha256=_file_sha256(path),
                )
                for relative, path in sorted(all_files.items())
            }
        omitted = sorted(set(all_files).difference(keep))
        projection: dict[str, object] | None = None
        if request.retention_class == RetentionClass.DUPLICATE_FAILURE_REDUCED:
            adjudication = request.adjudication
            if adjudication is None:
                raise AssertionError("duplicate retention adjudication disappeared")
            projection = {
                "schema_version": "experiment-retention-projection-v2",
                "run_label": request.run_label,
                "generation": request.generation,
                "retention_class": request.retention_class.value,
                "audit_only": True,
                "adjudication_sha256": adjudication.adjudication_sha256,
                "root_cause_id": adjudication.root_cause_id,
                "canonical_representative_run_label": (
                    adjudication.canonical_representative_run_label
                ),
                "kept_files": [
                    source_identities[path].to_dict() for path in sorted(keep)
                ],
                "omitted_files": [
                    source_identities[path].to_dict() for path in omitted
                ],
            }
        self._observe_archive_capacity(
            request=request,
            archive_root=archive_root,
            phase="pre_archive",
            projected_bytes=(
                sum(snapshot.byte_count for snapshot in source_snapshots.values())
                + sum(
                    expected.byte_count
                    for expected in request.expected_segment_trees
                    if expected.segment_id in native_segment_ids
                )
            ),
        )
        incoming = destination.parent / f".{destination.name}.incoming-{uuid.uuid4().hex}"
        if destination.exists():
            if request.retention_class.is_full:
                if request.expected_segment_trees:
                    try:
                        if (
                            request.copy_backend == "auto_native"
                            and os.name != "nt"
                            and _windows_path_for_mounted_drive(destination)
                            is not None
                        ):
                            verified_full_tree = _native_verify_retention_tree(
                                root=destination,
                                segments=request.segments,
                                expected_trees=request.expected_segment_trees,
                                workers=request.copy_workers,
                            )
                        else:
                            source_identities = _hash_full_destination(
                                destination,
                                expected_files=(
                                    None
                                    if native_segment_ids
                                    else set(all_files)
                                ),
                                workers=request.copy_workers,
                            )
                            logical_segment_paths = (
                                _complete_logical_segment_paths(
                                    segments=request.segments,
                                    identities=source_identities,
                                    known_paths=logical_segment_paths,
                                )
                            )
                            _verify_expected_segment_trees(
                                expected_trees=request.expected_segment_trees,
                                logical_segment_paths=logical_segment_paths,
                                identities=source_identities,
                            )
                    except StorageGovernanceError as error:
                        raise StorageGovernanceError(
                            "retention generation already exists but differs: "
                            f"{destination}"
                        ) from error
                    if verified_full_tree is None:
                        keep = set(source_identities)
                else:
                    source_tasks = tuple(
                        (
                            relative,
                            all_files[relative],
                            source_snapshots[relative],
                        )
                        for relative in sorted(all_files)
                    )
                    source_identities = {
                        identity.relative_path: identity
                        for identity in _parallel_map(
                            _hash_source_file,
                            source_tasks,
                            workers=request.copy_workers,
                        )
                    }
                    try:
                        _verify_full_destination(
                            destination,
                            source_identities=source_identities,
                            workers=request.copy_workers,
                        )
                    except StorageGovernanceError as error:
                        raise StorageGovernanceError(
                            "retention generation already exists but differs: "
                            f"{destination}"
                        ) from error
            elif not _published_generation_matches(
                destination,
                source_identities=source_identities,
                keep=keep,
                projection=projection,
            ):
                raise StorageGovernanceError(
                    f"retention generation already exists but differs: {destination}"
                )
        else:
            incoming.mkdir(parents=True)
            try:
                if request.retention_class.is_full:
                    native_segments = [
                        segment
                        for segment in request.segments
                        if segment.segment_id in native_segment_ids
                    ]
                    stream_segment_ids = {
                        segment.segment_id
                        for segment in request.segments
                        if segment.segment_id not in native_segment_ids
                    }
                    native_observations = [
                        _native_copy_segment(
                            source=segment.source_path,
                            destination=(
                                incoming
                                if segment.logical_prefix == "."
                                else incoming.joinpath(
                                    *PurePosixPath(
                                        segment.logical_prefix
                                    ).parts
                                )
                            ),
                            workers=request.copy_workers,
                        )
                        for segment in native_segments
                    ]
                    copy_tasks = tuple(
                        (
                            relative,
                            all_files[relative],
                            incoming.joinpath(*PurePosixPath(relative).parts),
                            source_snapshots[relative],
                        )
                        for relative in sorted(keep)
                        if logical_segment_paths[relative][0]
                        in stream_segment_ids
                    )
                    streamed_identities = {
                        identity.relative_path: identity
                        for identity in _parallel_map(
                            _copy_file_with_sha256,
                            copy_tasks,
                            workers=request.copy_workers,
                        )
                    }
                    if (
                        request.copy_backend == "auto_native"
                        and request.expected_segment_trees
                        and os.name != "nt"
                        and _windows_path_for_mounted_drive(incoming)
                        is not None
                    ):
                        verified_full_tree = _native_verify_retention_tree(
                            root=incoming,
                            segments=request.segments,
                            expected_trees=request.expected_segment_trees,
                            workers=request.copy_workers,
                        )
                        source_identities = streamed_identities
                    else:
                        target_identities = _hash_full_destination(
                            incoming,
                            expected_files=(
                                None
                                if native_segment_ids
                                else set(all_files)
                            ),
                            workers=request.copy_workers,
                        )
                        if any(
                            target_identities[relative] != identity
                            for relative, identity in streamed_identities.items()
                        ):
                            raise StorageGovernanceError(
                                "retained streamed file verification failed: "
                                f"{request.run_label}"
                            )
                        logical_segment_paths = (
                            _complete_logical_segment_paths(
                                segments=request.segments,
                                identities=target_identities,
                                known_paths=logical_segment_paths,
                            )
                        )
                        _verify_expected_segment_trees(
                            expected_trees=request.expected_segment_trees,
                            logical_segment_paths=logical_segment_paths,
                            identities=target_identities,
                        )
                        source_identities = target_identities
                        keep = set(source_identities)
                    if native_observations or verified_full_tree is not None:
                        _write_signed_json(
                            self.retention_state_root
                            / "retention_copy_observations"
                            / request.run_label
                            / f"generation-{request.generation:04d}"
                            / f"{incoming.name}.json",
                            {
                                "schema_version": (
                                    "experiment-retention-copy-observation-v1"
                                ),
                                "created_at_utc": datetime.now(UTC).isoformat(),
                                "run_label": request.run_label,
                                "generation": request.generation,
                                "copy_backend": request.copy_backend,
                                "copy_workers": request.copy_workers,
                                "native_segments": native_observations,
                                "streamed_segment_ids": sorted(
                                    stream_segment_ids
                                ),
                                "target_verification": (
                                    verified_full_tree.observation
                                    if verified_full_tree is not None
                                    else "full_sha256_passed"
                                ),
                                "source_deletion_authorized": False,
                            },
                        )
                else:
                    for relative in sorted(keep):
                        target = incoming.joinpath(*PurePosixPath(relative).parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(all_files[relative], target)
                        identity = source_identities[relative]
                        if (
                            target.stat().st_size != identity.byte_count
                            or _file_sha256(target) != identity.sha256
                        ):
                            raise StorageGovernanceError(
                                f"retained file verification failed: {relative}"
                            )
                if projection is not None:
                    _write_signed_json(
                        incoming / "retention_projection_manifest.json",
                        projection,
                    )
                if request.retention_class.is_full:
                    if any(
                        _file_snapshot(path) != source_snapshots[relative]
                        for relative, path in all_files.items()
                    ) or any(
                        _file_snapshot(
                            next(
                                segment.source_path.resolve()
                                for segment in request.segments
                                if segment.segment_id == segment_id
                            )
                        )
                        != snapshot
                        for segment_id, snapshot in native_source_snapshots.items()
                    ):
                        raise StorageGovernanceError(
                            "retention source changed before atomic publication: "
                            f"{request.run_label}"
                        )
                else:
                    prepublication_source_identities = {
                        relative: _FileIdentity(
                            relative_path=relative,
                            byte_count=path.stat().st_size,
                            sha256=_file_sha256(path),
                        )
                        for relative, path in sorted(all_files.items())
                    }
                    if prepublication_source_identities != source_identities:
                        raise StorageGovernanceError(
                            "retention source changed before atomic publication: "
                            f"{request.run_label}"
                        )
                incoming.parent.mkdir(parents=True, exist_ok=True)
                self._publish_incoming_generation(
                    request=request,
                    incoming=incoming,
                    destination=destination,
                )
            except BaseException:
                if incoming.exists():
                    shutil.rmtree(incoming)
                raise

        if request.retention_class.is_full:
            if any(
                _file_snapshot(path) != source_snapshots[relative]
                for relative, path in all_files.items()
            ) or any(
                _file_snapshot(
                    next(
                        segment.source_path.resolve()
                        for segment in request.segments
                        if segment.segment_id == segment_id
                    )
                )
                != snapshot
                for segment_id, snapshot in native_source_snapshots.items()
            ):
                raise StorageGovernanceError(
                    f"retention source changed during publication: {request.run_label}"
                )
        else:
            observed_source_identities = {
                relative: _FileIdentity(
                    relative_path=relative,
                    byte_count=path.stat().st_size,
                    sha256=_file_sha256(path),
                )
                for relative, path in sorted(all_files.items())
            }
            if observed_source_identities != source_identities:
                raise StorageGovernanceError(
                    f"retention source changed during publication: {request.run_label}"
                )
        self._observe_archive_capacity(
            request=request,
            archive_root=archive_root,
            phase="post_archive",
            projected_bytes=0,
        )
        if request.retention_class.is_full:
            if verified_full_tree is not None:
                kept_file_count = verified_full_tree.file_count
                kept_bytes = verified_full_tree.byte_count
                tree_sha256 = verified_full_tree.tree_sha256
            else:
                kept_file_count, kept_bytes, tree_sha256 = _identities_tree(
                    {
                        relative: source_identities[relative]
                        for relative in keep
                    }
                )
        else:
            kept_file_count, kept_bytes, tree_sha256 = _tree_identity(destination)
        replay_receipt_sha256 = ""
        replay_receipt_relative_path = ""
        if request.replay_verifier is not None:
            try:
                replay_receipt_path = request.replay_verifier(destination)
                if not (
                    isinstance(replay_receipt_path, Path)
                    and replay_receipt_path.is_absolute()
                ):
                    raise StorageGovernanceError(
                        "retention replay verifier did not return an absolute receipt"
                    )
                replay_receipt = _load_signed_json(replay_receipt_path)
                replay_receipt_sha256 = _file_sha256(replay_receipt_path)
                expected_replay = {
                    "schema_version": "experiment-retention-replay-v1",
                    "run_label": request.run_label,
                    "generation": request.generation,
                    "archive_tree_sha256": tree_sha256,
                    "archive_file_count": kept_file_count,
                    "archive_byte_count": kept_bytes,
                    "validator_replay_passed": True,
                    "objective_replay_passed": True,
                    "raw_review_replay_passed": True,
                    "status": "passed",
                }
                if any(
                    replay_receipt.get(field) != value
                    for field, value in expected_replay.items()
                ) or _SHA256.fullmatch(
                    str(replay_receipt.get("verifier_identity_sha256", ""))
                ) is None:
                    raise StorageGovernanceError(
                        "retention replay receipt identity is invalid"
                    )
                governed_replay_path = (
                    self.retention_state_root
                    / "retention_replays"
                    / (
                        f"{request.run_label}-generation-"
                        f"{request.generation:04d}.json"
                    )
                )
                if governed_replay_path.exists():
                    governed_replay = _load_signed_json(governed_replay_path)
                    without_created_at = {
                        key: value
                        for key, value in replay_receipt.items()
                        if key != "created_at_utc"
                    }
                    governed_without_created_at = {
                        key: value
                        for key, value in governed_replay.items()
                        if key != "created_at_utc"
                    }
                    if governed_without_created_at != without_created_at:
                        raise StorageGovernanceError(
                            "retention replay generation already exists but differs"
                        )
                    replay_receipt_sha256 = _file_sha256(governed_replay_path)
                else:
                    replay_receipt_sha256 = _write_signed_json(
                        governed_replay_path,
                        replay_receipt,
                    )
                replay_receipt_relative_path = governed_replay_path.relative_to(
                    self.retention_state_root
                ).as_posix()
            except Exception as error:
                raise StorageGovernanceError(
                    f"independent retention replay failed: {request.run_label}"
                ) from error
        original_kept_bytes = (
            kept_bytes
            if request.retention_class.is_full
            else sum(source_identities[path].byte_count for path in keep)
        )
        omitted_bytes = (
            0
            if request.retention_class.is_full
            else sum(source_identities[path].byte_count for path in omitted)
        )
        registry_sha256 = self._register_retention_generation(
            request=request,
            tree_sha256=tree_sha256,
            file_count=kept_file_count,
            byte_count=kept_bytes,
            original_kept_bytes=original_kept_bytes,
            omitted_file_count=len(omitted),
            omitted_bytes=omitted_bytes,
            replay_receipt_sha256=replay_receipt_sha256,
            replay_receipt_relative_path=replay_receipt_relative_path,
        )
        maintenance_audit = self.sweep_rebuildable_assets(
            _maintenance_request_for_roots(
                self.policy,
                tuple(segment.source_path for segment in request.segments),
            )
        )
        return RetentionReceipt(
            run_label=request.run_label,
            generation=request.generation,
            retention_class=request.retention_class,
            archive_path=destination,
            tree_sha256=tree_sha256,
            kept_file_count=(
                kept_file_count
                if request.retention_class.is_full
                else len(keep)
            ),
            kept_bytes=original_kept_bytes,
            omitted_file_count=len(omitted),
            omitted_bytes=omitted_bytes,
            audit_only=(
                request.retention_class == RetentionClass.DUPLICATE_FAILURE_REDUCED
            ),
            replay_receipt_sha256=replay_receipt_sha256,
            maintenance_audit_sha256=maintenance_audit.receipt_sha256,
            maintenance_audit_path=maintenance_audit.receipt_path,
            registry_sha256=registry_sha256,
        )

    def sweep_rebuildable_assets(self, request: SweepRequest) -> SweepReceipt:
        """Audit or remove only exact, old, unreferenced, proven-rebuildable assets."""

        allowed_roots = tuple(root.resolve() for root in request.allowed_roots)
        request_identity = {
            "schema_version": "experiment-rebuildable-request-v1",
            "allowed_roots": [str(root) for root in allowed_roots],
            "retention_days": request.retention_days,
            "assets": [
                {
                    "path": str(asset.path.resolve()),
                    "kind": asset.kind.value,
                    "expected_tree_sha256": asset.expected_tree_sha256,
                    "keeper_references": sorted(set(asset.keeper_references)),
                    "rebuild_proof_path": (
                        str(asset.rebuild_proof_path.resolve())
                        if asset.rebuild_proof_path is not None
                        else ""
                    ),
                    "rebuild_proof_sha256": asset.rebuild_proof_sha256,
                }
                for asset in request.assets
            ],
        }
        request_identity_sha256 = hashlib.sha256(
            _canonical_json(request_identity)
        ).hexdigest()
        candidates: list[Path] = []
        retained: list[tuple[Path, str]] = []
        candidate_bytes = 0
        now_timestamp = datetime.now(UTC).timestamp()
        minimum_age_seconds = request.retention_days * 24 * 60 * 60
        seen: set[Path] = set()
        for asset in request.assets:
            path = asset.path.resolve()
            if path in seen:
                raise StorageGovernanceError(f"duplicate rebuildable asset: {path}")
            seen.add(path)
            if not any(path.is_relative_to(root) and path != root for root in allowed_roots):
                raise StorageGovernanceError(
                    f"rebuildable asset is outside its exact allowlist: {path}"
                )
            if not path.exists():
                retained.append((path, "missing"))
                continue
            if path.is_symlink():
                raise StorageGovernanceError(
                    f"rebuildable asset cannot be a symlink: {path}"
                )
            protected_reason = _protected_rebuildable_reason(path)
            if protected_reason is not None:
                retained.append((path, protected_reason))
                continue
            file_count, byte_count, observed_tree = _tree_identity(path)
            if file_count == 0:
                retained.append((path, "empty"))
                continue
            if observed_tree != asset.expected_tree_sha256:
                retained.append((path, "identity_mismatch"))
                continue
            files = tuple(item for item in path.rglob("*") if item.is_file())
            if any(
                item.name == ".lock" or item.name.endswith(".lock") for item in files
            ):
                retained.append((path, "active_lock"))
                continue
            newest_timestamp = max(
                [path.stat().st_mtime, *(item.stat().st_mtime for item in files)]
            )
            if now_timestamp - newest_timestamp < minimum_age_seconds:
                retained.append((path, "retention_period"))
                continue
            if self._keeper_reference_scanner is None:
                retained.append((path, "missing_keeper_reference_scan"))
                continue
            scanned_references = tuple(
                sorted(set(self._keeper_reference_scanner(path)))
            )
            if scanned_references != tuple(sorted(set(asset.keeper_references))):
                retained.append((path, "keeper_mapping_mismatch"))
                continue
            if scanned_references:
                retained.append((path, "keeper_reference"))
                continue
            if asset.kind in {RebuildableKind.VENV, RebuildableKind.BUILD}:
                if asset.rebuild_proof_path is None or not asset.rebuild_proof_sha256:
                    retained.append((path, "missing_rebuild_proof"))
                    continue
                try:
                    proof = _load_signed_json(asset.rebuild_proof_path)
                except StorageGovernanceError:
                    retained.append((path, "invalid_rebuild_proof"))
                    continue
                if (
                    _file_sha256(asset.rebuild_proof_path)
                    != asset.rebuild_proof_sha256
                    or proof.get("schema_version")
                    != "experiment-rebuild-proof-v1"
                    or proof.get("kind") != asset.kind.value
                    or proof.get("source_tree_sha256") != observed_tree
                    or proof.get("isolated_rebuild_match") is not True
                    or proof.get("smoke_test_passed") is not True
                ):
                    retained.append((path, "invalid_rebuild_proof"))
                    continue
                if (
                    self._isolated_rebuild_verifier is None
                    or not self._isolated_rebuild_verifier(asset)
                ):
                    retained.append((path, "isolated_rebuild_not_verified"))
                    continue
            candidates.append(path)
            candidate_bytes += byte_count

        decision = {
            "schema_version": "experiment-rebuildable-audit-v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "request_identity": request_identity,
            "request_identity_sha256": request_identity_sha256,
            "candidate_paths": [str(path) for path in candidates],
            "retained": [
                {"path": str(path), "reason": reason} for path, reason in retained
            ],
            "candidate_bytes": candidate_bytes,
            "applied": False,
        }
        receipt_root = self.state_root / "maintenance_audits"
        receipt_token = (
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')}-"
            f"{uuid.uuid4().hex}"
        )
        if request.apply:
            approved_path = request.approved_dry_run_path
            if approved_path is None:
                raise StorageGovernanceError(
                    "rebuildable cleanup requires an approved signed dry run"
                )
            confirmation_path = request.confirmation_receipt_path
            if confirmation_path is None:
                raise StorageGovernanceError(
                    "rebuildable cleanup requires a signed literal confirmation"
                )
            approved = _load_signed_json(approved_path)
            if (
                _file_sha256(approved_path) != request.approved_dry_run_sha256
                or approved.get("schema_version")
                != "experiment-rebuildable-audit-v1"
                or approved.get("applied") is not False
                or approved.get("request_identity_sha256")
                != request_identity_sha256
                or approved.get("candidate_paths")
                != [str(path) for path in candidates]
                or approved.get("retained") != decision["retained"]
                or approved.get("candidate_bytes") != candidate_bytes
            ):
                raise StorageGovernanceError(
                    "rebuildable cleanup differs from its approved dry run"
                )
            confirmation = _load_signed_json(confirmation_path)
            if (
                _file_sha256(confirmation_path)
                != request.confirmation_receipt_sha256
                or confirmation.get("schema_version")
                != "experiment-cleanup-confirmation-v1"
                or confirmation.get("confirmation_text") != "确认"
                or confirmation.get("approved_dry_run_sha256")
                != request.approved_dry_run_sha256
                or confirmation.get("request_identity_sha256")
                != request_identity_sha256
                or confirmation.get("candidate_paths")
                != [str(path) for path in candidates]
                or confirmation.get("candidate_bytes") != candidate_bytes
            ):
                raise StorageGovernanceError(
                    "rebuildable cleanup confirmation identity is invalid"
                )
            started_path = receipt_root / f"{receipt_token}-apply-started.json"
            started_sha256 = _write_signed_json(
                started_path,
                {
                    **decision,
                    "schema_version": "experiment-rebuildable-apply-v1",
                    "approved_dry_run_sha256": request.approved_dry_run_sha256,
                    "confirmation_receipt_sha256": (
                        request.confirmation_receipt_sha256
                    ),
                    "status": "started",
                },
            )
            deleted: list[str] = []
            try:
                assets_by_path = {
                    asset.path.resolve(): asset for asset in request.assets
                }
                for path in candidates:
                    resolved = path.resolve()
                    if not any(
                        resolved.is_relative_to(root) and resolved != root
                        for root in allowed_roots
                    ):
                        raise StorageGovernanceError(
                            "rebuildable asset escaped allowlist before removal: "
                            f"{path}"
                        )
                    asset = assets_by_path[resolved]
                    if (
                        not resolved.exists()
                        or _tree_identity(resolved)[2]
                        != asset.expected_tree_sha256
                    ):
                        raise StorageGovernanceError(
                            f"rebuildable asset changed before removal: {path}"
                        )
                    protected_reason = _protected_rebuildable_reason(resolved)
                    if protected_reason is not None:
                        raise StorageGovernanceError(
                            "rebuildable asset became protected before removal: "
                            f"{path} ({protected_reason})"
                        )
                    current_files = tuple(
                        item for item in resolved.rglob("*") if item.is_file()
                    )
                    if any(
                        item.name == ".lock" or item.name.endswith(".lock")
                        for item in current_files
                    ):
                        raise StorageGovernanceError(
                            f"rebuildable asset acquired an active lock: {path}"
                        )
                    if self._keeper_reference_scanner is None:
                        raise StorageGovernanceError(
                            "keeper reference scanner disappeared before removal"
                        )
                    if tuple(
                        sorted(set(self._keeper_reference_scanner(resolved)))
                    ):
                        raise StorageGovernanceError(
                            f"rebuildable asset gained a keeper reference: {path}"
                        )
                    if resolved.is_dir():
                        shutil.rmtree(resolved)
                    else:
                        resolved.unlink()
                    deleted.append(str(resolved))
            except BaseException as error:
                _write_signed_json(
                    receipt_root / f"{receipt_token}-apply-failed.json",
                    {
                        **decision,
                        "schema_version": "experiment-rebuildable-apply-v1",
                        "approved_dry_run_sha256": (
                            request.approved_dry_run_sha256
                        ),
                        "confirmation_receipt_sha256": (
                            request.confirmation_receipt_sha256
                        ),
                        "started_receipt_sha256": started_sha256,
                        "status": "failed",
                        "deleted_paths": deleted,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                raise
            receipt_path = receipt_root / f"{receipt_token}-apply-complete.json"
            receipt_sha256 = _write_signed_json(
                receipt_path,
                {
                    **decision,
                    "schema_version": "experiment-rebuildable-apply-v1",
                    "approved_dry_run_sha256": request.approved_dry_run_sha256,
                    "confirmation_receipt_sha256": (
                        request.confirmation_receipt_sha256
                    ),
                    "started_receipt_sha256": started_sha256,
                    "status": "complete",
                    "deleted_paths": deleted,
                    "applied": True,
                },
            )
        else:
            receipt_path = receipt_root / f"{receipt_token}-dry-run.json"
            receipt_sha256 = _write_signed_json(receipt_path, decision)
        return SweepReceipt(
            candidates=tuple(candidates),
            retained=tuple(retained),
            candidate_bytes=candidate_bytes,
            applied=request.apply,
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
        )

    def resolve_run(self, run_label: str, *, allow_audit_only: bool = False) -> Path:
        """Resolve and reverify the newest retained generation for one run."""

        if _RUN_LABEL.fullmatch(run_label) is None:
            raise ValueError("retained run label is invalid")
        records = self._load_registry()
        candidates = [record for record in records if record.get("run_label") == run_label]
        if not candidates:
            return self._resolve_legacy_run(run_label)
        generations = {
            _strict_registry_int(record, "generation") for record in candidates
        }
        generation = max(generations)
        selected = [
            record
            for record in candidates
            if _strict_registry_int(record, "generation") == generation
        ]
        if (
            not allow_audit_only
            and any(record.get("audit_only") is True for record in selected)
        ):
            raise StorageGovernanceError(
                f"retained run is audit-only and cannot be a prerequisite: {run_label}"
            )
        aliases = {record.get("archive_root_alias") for record in selected}
        paths = {record.get("archive_relative_path") for record in selected}
        tree_hashes = {record.get("tree_sha256") for record in selected}
        retention_classes = {record.get("retention_class") for record in selected}
        if (
            len(aliases) != 1
            or len(paths) != 1
            or len(tree_hashes) != 1
            or len(retention_classes) != 1
            or not all(
                record.get("verification_status") == "verified"
                for record in selected
            )
        ):
            raise StorageGovernanceError("retention generation records disagree")
        alias = next(iter(aliases))
        relative = next(iter(paths))
        expected_tree = next(iter(tree_hashes))
        if not isinstance(alias, str) or not isinstance(relative, str):
            raise StorageGovernanceError("retention generation location is invalid")
        root = self.locator.resolve(alias)
        self.locator.verify_all(self._volume_probe, (alias,))
        path = root.absolute_path.joinpath(*PurePosixPath(relative).parts)
        if not path.resolve().is_relative_to(root.absolute_path.resolve()):
            raise StorageGovernanceError("retained run escapes archive root")
        observed_count, observed_bytes, observed_tree = _tree_identity(path)
        if observed_tree != expected_tree:
            raise StorageGovernanceError(f"retained run tree changed: {run_label}")
        retention_class = next(iter(retention_classes))
        if retention_class in {
            RetentionClass.ACCEPTED_FULL.value,
            RetentionClass.UNIQUE_FAILURE_FULL.value,
        }:
            replay_paths = {
                record.get("replay_receipt_relative_path") for record in selected
            }
            replay_hashes = {
                record.get("replay_receipt_sha256") for record in selected
            }
            if len(replay_paths) != 1 or len(replay_hashes) != 1:
                raise StorageGovernanceError(
                    f"full retained run replay records disagree: {run_label}"
                )
            replay_relative = next(iter(replay_paths))
            replay_sha256 = next(iter(replay_hashes))
            if not isinstance(replay_relative, str) or _SHA256.fullmatch(
                str(replay_sha256)
            ) is None:
                raise StorageGovernanceError(
                    f"full retained run replay identity is missing: {run_label}"
                )
            replay_path = self.retention_state_root.joinpath(
                *PurePosixPath(replay_relative).parts
            )
            if not replay_path.resolve().is_relative_to(
                self.retention_state_root.resolve()
            ):
                raise StorageGovernanceError(
                    "retention replay receipt escapes state root"
                )
            replay = _load_signed_json(replay_path)
            if (
                _file_sha256(replay_path) != replay_sha256
                or replay.get("schema_version")
                != "experiment-retention-replay-v1"
                or replay.get("run_label") != run_label
                or replay.get("generation") != generation
                or replay.get("archive_tree_sha256") != observed_tree
                or replay.get("archive_file_count") != observed_count
                or replay.get("archive_byte_count") != observed_bytes
                or replay.get("validator_replay_passed") is not True
                or replay.get("objective_replay_passed") is not True
                or replay.get("raw_review_replay_passed") is not True
                or replay.get("status") != "passed"
            ):
                raise StorageGovernanceError(
                    f"full retained run replay verification failed: {run_label}"
                )
        return path

    def is_retained_path(self, path: Path) -> bool:
        """Recognize every immutable v2 generation and legacy v1 archive path."""

        resolved = path.resolve()
        for record in self._load_registry():
            alias = record.get("archive_root_alias")
            relative = record.get("archive_relative_path")
            if not isinstance(alias, str) or not isinstance(relative, str):
                raise StorageGovernanceError("retention registry location is invalid")
            candidate = self.locator.resolve(alias).absolute_path.joinpath(
                *PurePosixPath(relative).parts
            )
            if candidate.resolve() == resolved:
                return True
        if self._legacy_registry_path is not None:
            from evrptw.stage052_retention import load_retention_registry

            for legacy_record in load_retention_registry(self._legacy_registry_path):
                candidate = self.locator.resolve(
                    legacy_record.archive_root_alias
                ).absolute_path.joinpath(
                    *PurePosixPath(legacy_record.archive_relative_path).parts
                )
                if candidate.resolve() == resolved:
                    return True
        return False

    def _resolve_legacy_run(self, run_label: str) -> Path:
        if self._legacy_registry_path is None:
            raise StorageGovernanceError(f"retained run is not registered: {run_label}")
        from evrptw.stage052_retention import (
            RetentionIntegrityError,
            load_retention_registry,
            resolve_retained_run,
        )

        try:
            records = [
                record
                for record in load_retention_registry(self._legacy_registry_path)
                if record.run_label == run_label
            ]
            if len(records) != 1:
                raise StorageGovernanceError(
                    f"retained run is not registered in v2 or v1: {run_label}"
                )
            alias = records[0].archive_root_alias
            self.locator.verify_all(self._volume_probe, (alias,))
            return resolve_retained_run(
                run_label,
                registry_path=self._legacy_registry_path,
                archive_roots={alias: self.locator.resolve(alias).absolute_path},
            )
        except RetentionIntegrityError as error:
            raise StorageGovernanceError(
                f"legacy retained run verification failed: {run_label}"
            ) from error

    def _load_registry(self) -> list[dict[str, object]]:
        path = self.retention_state_root / "retention_registry_v2.json"
        if not path.exists():
            return []
        return _registry_records(_load_signed_json(path))

    def _run_label_is_registered(self, run_label: str) -> bool:
        if any(
            record.get("run_label") == run_label for record in self._load_registry()
        ):
            return True
        if self._legacy_registry_path is None:
            return False
        if not self._legacy_registry_path.exists():
            return False
        from evrptw.stage052_retention import load_retention_registry

        try:
            return any(
                record.run_label == run_label
                for record in load_retention_registry(self._legacy_registry_path)
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise StorageGovernanceError(
                "cannot verify legacy retention registry"
            ) from error

    def _verify_canonical_failure(self, adjudication: AdjudicationRecord) -> None:
        records = self._load_registry()
        matching = [
            record
            for record in records
            if record.get("run_label")
            == adjudication.canonical_representative_run_label
        ]
        if not matching:
            raise StorageGovernanceError(
                "adjudication canonical representative is not retained"
            )
        latest = max(_strict_registry_int(record, "generation") for record in matching)
        selected = [
            record
            for record in matching
            if _strict_registry_int(record, "generation") == latest
        ]
        if any(
            record.get("retention_class")
            != RetentionClass.UNIQUE_FAILURE_FULL.value
            or record.get("root_cause_id") != adjudication.root_cause_id
            or record.get("verification_status") != "verified"
            for record in selected
        ):
            raise StorageGovernanceError(
                "adjudication canonical representative is not a matching full failure"
            )
        self.resolve_run(adjudication.canonical_representative_run_label)

    def _register_retention_generation(
        self,
        *,
        request: RetentionRequest,
        tree_sha256: str,
        file_count: int,
        byte_count: int,
        original_kept_bytes: int,
        omitted_file_count: int,
        omitted_bytes: int,
        replay_receipt_sha256: str,
        replay_receipt_relative_path: str,
    ) -> str:
        registry_path = self.retention_state_root / "retention_registry_v2.json"
        lock_path = self.retention_state_root / "retention_registry_v2.lock"
        with _state_lock(lock_path):
            records = self._load_registry()
            additions: list[dict[str, object]] = [
                {
                    "schema_version": "experiment-retention-record-v2",
                    "run_label": request.run_label,
                    "segment_id": segment.segment_id,
                    "generation": request.generation,
                    "retention_class": request.retention_class.value,
                    "root_cause_id": (
                        request.root_cause_id
                        or (
                            request.adjudication.root_cause_id
                            if request.adjudication is not None
                            else ""
                        )
                    ),
                    "archive_root_alias": request.archive_root_alias,
                    "archive_relative_path": request.archive_relative_path,
                    "tree_sha256": tree_sha256,
                    "file_count": file_count,
                    "byte_count": byte_count,
                    "original_kept_bytes": original_kept_bytes,
                    "omitted_file_count": omitted_file_count,
                    "omitted_bytes": omitted_bytes,
                    "audit_only": (
                        request.retention_class
                        == RetentionClass.DUPLICATE_FAILURE_REDUCED
                    ),
                    "replay_receipt_sha256": replay_receipt_sha256,
                    "replay_receipt_relative_path": replay_receipt_relative_path,
                    "verification_status": "verified",
                }
                for segment in request.segments
            ]
            existing_by_key = {
                (
                    record.get("run_label"),
                    record.get("segment_id"),
                    record.get("generation"),
                ): record
                for record in records
            }
            for addition in additions:
                key = (
                    addition["run_label"],
                    addition["segment_id"],
                    addition["generation"],
                )
                existing = existing_by_key.get(key)
                if existing is not None and existing != addition:
                    raise StorageGovernanceError(
                        f"retention registry generation conflicts at {key}"
                    )
                if existing is None:
                    records.append(addition)
            records.sort(
                key=lambda record: (
                    str(record.get("run_label")),
                    _strict_registry_int(record, "generation"),
                    str(record.get("segment_id")),
                )
            )
            return _write_signed_json(
                registry_path,
                {
                    "schema_version": "experiment-retention-registry-v2",
                    "records": records,
                },
            )
