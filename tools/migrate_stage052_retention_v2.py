"""Resume-safe Stage 5.2 migration into the governed E-drive v2 archive.

This operational tool is intentionally conservative:

* every unadjudicated source defaults to ``unknown_full``;
* source trees are never modified or deleted;
* a capacity permit is acquired before any archive generation is created;
* each generation is copied through the storage-governance transaction;
* signed dry runs, cross-volume attestations, resolver replay, and a deletion
  candidate manifest are produced before a human may authorize deletion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import sysconfig
import types
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.machinery import ModuleSpec
from pathlib import Path, PurePosixPath
from typing import Any, Final

# The WSL runtime may provide compiled ``evrptw._core`` dependencies through an
# existing venv whose editable project checkout predates storage governance.
# Build an explicit package search path so Python loads governance code from this
# checkout and native dependencies from that venv, without modifying either.
_SOURCE_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "evrptw"
_RUNTIME_PACKAGE = Path(sysconfig.get_paths()["purelib"]) / "evrptw"
if "evrptw" not in sys.modules:
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not (
            str(getattr(finder, "__module__", "")).startswith(
                "__editable___evrptw_"
            )
            or str(getattr(finder, "__module__", ""))
            == "_reproducible_evrptw_editable"
        )
    ]
    _package_paths = [
        str(path)
        for path in (_SOURCE_PACKAGE, _RUNTIME_PACKAGE)
        if path.is_dir()
    ]
    if str(_SOURCE_PACKAGE) not in _package_paths:
        raise RuntimeError(f"source package is missing: {_SOURCE_PACKAGE}")
    _package = types.ModuleType("evrptw")
    _package.__file__ = str(_SOURCE_PACKAGE / "__init__.py")
    _package.__package__ = "evrptw"
    _package.__path__ = _package_paths
    _package.__spec__ = ModuleSpec("evrptw", loader=None, is_package=True)
    _package.__spec__.submodule_search_locations = _package_paths
    sys.modules["evrptw"] = _package

from evrptw.stage052_campaign import StorageRootLocator  # noqa: E402
from evrptw.stage052_campaign_runner import probe_volume_identity  # noqa: E402
from evrptw.storage_governance import (  # noqa: E402
    GIB,
    ExperimentStorageGovernance,
    GovernancePolicy,
    RetentionClass,
    RetentionRequest,
    RetentionSegment,
    StartRequest,
    StorageGovernanceError,
    compute_tree_identity,
    verify_storage_migration_attestation,
    write_migration_dry_run,
    write_storage_migration_attestation,
)

_RUN_LABEL: Final = re.compile(
    r"^stage05\.2_[a-z0-9_]+_(?:attempt|rerun)[0-9]{2}$"
)
_MIGRATION_LABEL: Final = "stage05.2_storage_migration_attempt01"
_MIGRATION_ID: Final = "stage052-retention-v2-20260731"
_GENERATION: Final = 1
_SCHEMA_PROGRESS: Final = "stage052-retention-migration-progress-v1"


@dataclass(frozen=True, slots=True)
class SourceSegment:
    """One immutable physical source included in a logical retained run."""

    run_label: str
    segment_id: str
    source_root_alias: str
    source_relative_path: str
    source_path: Path

    @property
    def logical_id(self) -> str:
        return f"{self.run_label}:{self.segment_id}"

    @property
    def destination_relative_path(self) -> str:
        return (
            f"stage05.2/runs/{self.run_label}/generation-"
            f"{_GENERATION:04d}/{self.segment_id}"
        )

    @property
    def run_destination_relative_path(self) -> str:
        return (
            f"stage05.2/runs/{self.run_label}/generation-{_GENERATION:04d}"
        )


@dataclass(frozen=True, slots=True)
class SegmentIdentity:
    """Signed dry-run identity for one source segment."""

    logical_id: str
    run_label: str
    segment_id: str
    source_root_alias: str
    source_relative_path: str
    destination_relative_path: str
    file_count: int
    byte_count: int
    tree_sha256: str


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_signed_json(path: Path, payload: object) -> str:
    data = _canonical_json(payload)
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temporary = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
    sidecar_temporary.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    os.replace(sidecar_temporary, sidecar)
    return digest


def _load_signed_json(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise StorageGovernanceError(f"signed JSON is incomplete: {path}")
    fields = sidecar.read_text(encoding="ascii").split()
    if len(fields) != 2 or fields[0] != _sha256(path) or fields[1] != path.name:
        raise StorageGovernanceError(f"signed JSON verification failed: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StorageGovernanceError(f"signed JSON must contain an object: {path}")
    return payload


def _log(event: str, **fields: object) -> None:
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "event": event,
        **fields,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def _safe_relative(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise StorageGovernanceError(f"unsafe relative path: {relative}")
    candidate = root.joinpath(*pure.parts)
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise StorageGovernanceError(f"path escapes root: {candidate}")
    return candidate


def _discover_sources(locator: StorageRootLocator) -> tuple[SourceSegment, ...]:
    wsl_root = locator.resolve("wsl_staging").absolute_path
    d_root = locator.resolve("d_archive").absolute_path
    history_root = d_root / "stage05.2" / "history"
    discovered: list[SourceSegment] = []

    def add_children(
        root: Path,
        *,
        source_root_alias: str,
        relative_prefix: str,
        segment_id: str,
        name_filter: re.Pattern[str],
    ) -> None:
        if not root.is_dir():
            raise StorageGovernanceError(f"migration source root is missing: {root}")
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            if not child.is_dir() or child.is_symlink():
                continue
            if name_filter.fullmatch(child.name) is None:
                continue
            relative = (
                f"{relative_prefix}/{child.name}" if relative_prefix else child.name
            )
            discovered.append(
                SourceSegment(
                    run_label=child.name,
                    segment_id=segment_id,
                    source_root_alias=source_root_alias,
                    source_relative_path=relative,
                    source_path=child.resolve(),
                )
            )

    add_children(
        wsl_root,
        source_root_alias="wsl_staging",
        relative_prefix="",
        segment_id="wsl_active",
        name_filter=_RUN_LABEL,
    )
    add_children(
        history_root,
        source_root_alias="d_archive",
        relative_prefix="stage05.2/history",
        segment_id="d_history",
        name_filter=_RUN_LABEL,
    )
    benchmark_filter = re.compile(
        r"^stage05\.2_benchmark_(?:attempt|rerun)[0-9]{2}$"
    )
    add_children(
        d_root,
        source_root_alias="d_archive",
        relative_prefix="",
        segment_id="d_benchmark",
        name_filter=benchmark_filter,
    )
    if not discovered:
        raise StorageGovernanceError("no canonical Stage 5.2 sources were discovered")
    keys = [
        (segment.run_label, segment.segment_id)
        for segment in discovered
    ]
    if len(keys) != len(set(keys)):
        raise StorageGovernanceError("duplicate migration source segment identity")
    return tuple(
        sorted(discovered, key=lambda item: (item.run_label, item.segment_id))
    )


def _inventory_segment(segment: SourceSegment) -> SegmentIdentity:
    file_count, byte_count, tree_sha256 = compute_tree_identity(segment.source_path)
    return SegmentIdentity(
        logical_id=segment.logical_id,
        run_label=segment.run_label,
        segment_id=segment.segment_id,
        source_root_alias=segment.source_root_alias,
        source_relative_path=segment.source_relative_path,
        destination_relative_path=segment.destination_relative_path,
        file_count=file_count,
        byte_count=byte_count,
        tree_sha256=tree_sha256,
    )


def _inventory(
    segments: tuple[SourceSegment, ...],
    *,
    progress_path: Path,
) -> tuple[SegmentIdentity, ...]:
    cached: dict[str, SegmentIdentity] = {}
    if progress_path.exists():
        payload = _load_signed_json(progress_path)
        if payload.get("schema_version") != _SCHEMA_PROGRESS:
            raise StorageGovernanceError("migration progress schema is invalid")
        records = payload.get("inventory")
        if not isinstance(records, list):
            raise StorageGovernanceError("migration inventory progress is invalid")
        for record in records:
            if not isinstance(record, dict):
                raise StorageGovernanceError("migration inventory record is invalid")
            identity = SegmentIdentity(**record)
            cached[identity.logical_id] = identity

    expected = {segment.logical_id: segment for segment in segments}
    if not set(cached).issubset(expected):
        raise StorageGovernanceError("migration progress contains stale source IDs")
    for logical_id, identity in cached.items():
        segment = expected[logical_id]
        fixed = (
            identity.run_label,
            identity.segment_id,
            identity.source_root_alias,
            identity.source_relative_path,
            identity.destination_relative_path,
        )
        current = (
            segment.run_label,
            segment.segment_id,
            segment.source_root_alias,
            segment.source_relative_path,
            segment.destination_relative_path,
        )
        if fixed != current:
            raise StorageGovernanceError(
                f"migration progress source mapping changed: {logical_id}"
            )

    for index, segment in enumerate(segments, start=1):
        if segment.logical_id in cached:
            continue
        _log(
            "inventory_started",
            index=index,
            total=len(segments),
            logical_id=segment.logical_id,
            source=str(segment.source_path),
        )
        cached[segment.logical_id] = _inventory_segment(segment)
        _write_signed_json(
            progress_path,
            {
                "schema_version": _SCHEMA_PROGRESS,
                "migration_id": _MIGRATION_ID,
                "source_deletion_authorized": False,
                "inventory": [
                    asdict(cached[key]) for key in sorted(cached)
                ],
            },
        )
        _log(
            "inventory_completed",
            index=index,
            total=len(segments),
            logical_id=segment.logical_id,
            file_count=cached[segment.logical_id].file_count,
            byte_count=cached[segment.logical_id].byte_count,
            tree_sha256=cached[segment.logical_id].tree_sha256,
        )
    return tuple(cached[key] for key in sorted(cached))


def _dry_run_payload(
    identities: tuple[SegmentIdentity, ...],
    *,
    source_root_alias: str,
) -> dict[str, object]:
    selected = [
        identity
        for identity in identities
        if identity.source_root_alias == source_root_alias
    ]
    if not selected:
        raise StorageGovernanceError(
            f"no migration records for source alias: {source_root_alias}"
        )
    sources = [
        {
            "logical_id": item.logical_id,
            "root_alias": item.source_root_alias,
            "relative_path": item.source_relative_path,
            "file_count": item.file_count,
            "byte_count": item.byte_count,
            "tree_sha256": item.tree_sha256,
        }
        for item in selected
    ]
    mappings = [
        {
            "logical_id": item.logical_id,
            "source_relative_path": item.source_relative_path,
            "destination_relative_path": item.destination_relative_path,
            "file_count": item.file_count,
            "byte_count": item.byte_count,
            "tree_sha256": item.tree_sha256,
            "source_root_alias": item.source_root_alias,
            "destination_root_alias": "e_archive",
        }
        for item in selected
    ]
    return {
        "schema_version": "experiment-storage-migration-dry-run-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "migration_id": _MIGRATION_ID,
        "source_deletion_authorized": False,
        "retention_default": RetentionClass.UNKNOWN_FULL.value,
        "full_retention_upper_bound_bytes": sum(
            item.byte_count for item in selected
        ),
        "sources": sources,
        "planned_mappings": mappings,
    }


def _write_dry_runs(
    identities: tuple[SegmentIdentity, ...],
    *,
    evidence_root: Path,
) -> dict[str, tuple[Path, str]]:
    result: dict[str, tuple[Path, str]] = {}
    for alias in ("wsl_staging", "d_archive"):
        path = evidence_root / f"{alias}-detailed-dry-run.json"
        payload = _dry_run_payload(identities, source_root_alias=alias)
        if path.exists():
            existing = _load_signed_json(path)
            invariant_fields = (
                "schema_version",
                "migration_id",
                "source_deletion_authorized",
                "retention_default",
                "full_retention_upper_bound_bytes",
                "sources",
                "planned_mappings",
            )
            if any(existing.get(field) != payload.get(field) for field in invariant_fields):
                raise StorageGovernanceError(
                    f"signed migration dry run differs from inventory: {path}"
                )
            digest = _sha256(path)
        else:
            digest = write_migration_dry_run(path, payload)
        result[alias] = (path, digest)
        _log(
            "dry_run_ready",
            source_root_alias=alias,
            path=str(path),
            sha256=digest,
            bytes=payload["full_retention_upper_bound_bytes"],
            mappings=len(payload["planned_mappings"]),  # type: ignore[arg-type]
        )
    return result


def _governance(
    *,
    repo_root: Path,
    locator: StorageRootLocator,
) -> ExperimentStorageGovernance:
    return ExperimentStorageGovernance(
        policy=GovernancePolicy.from_toml(
            repo_root / "configs" / "experiment_storage_governance.toml"
        ),
        locator=locator,
        state_root=(
            locator.resolve("wsl_staging").absolute_path / ".storage-governance"
        ),
        retention_state_root=(
            locator.resolve("e_archive").absolute_path / ".storage-governance"
        ),
        free_space=lambda path: shutil.disk_usage(path).free,
        volume_probe=probe_volume_identity,
        legacy_registry_path=(
            repo_root
            / "experiments"
            / "registries"
            / "stage05.2_retention_registry.csv"
        ),
    )


def _registry_records(retention_state_root: Path) -> list[dict[str, Any]]:
    path = retention_state_root / "retention_registry_v2.json"
    if not path.exists():
        return []
    payload = _load_signed_json(path)
    if payload.get("schema_version") != "experiment-retention-registry-v2":
        raise StorageGovernanceError("retention registry v2 schema is invalid")
    records = payload.get("records")
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise StorageGovernanceError("retention registry v2 records are invalid")
    return [dict(record) for record in records]


def _run_is_registered(
    run_label: str,
    segments: tuple[SourceSegment, ...],
    *,
    retention_state_root: Path,
) -> bool:
    records = [
        record
        for record in _registry_records(retention_state_root)
        if record.get("run_label") == run_label
        and record.get("generation") == _GENERATION
    ]
    expected_ids = {segment.segment_id for segment in segments}
    if not records:
        return False
    observed_ids = {record.get("segment_id") for record in records}
    expected_path = segments[0].run_destination_relative_path
    if (
        observed_ids != expected_ids
        or any(
            record.get("archive_root_alias") != "e_archive"
            or record.get("archive_relative_path") != expected_path
            or record.get("retention_class") != RetentionClass.UNKNOWN_FULL.value
            or record.get("verification_status") != "verified"
            for record in records
        )
    ):
        raise StorageGovernanceError(
            f"existing registry generation conflicts with migration: {run_label}"
        )
    return True


def _capacity_request(
    *,
    locator: StorageRootLocator,
    planned_archive_bytes: int,
    stage_plan_sha256: str,
) -> StartRequest:
    return StartRequest(
        stage_id="stage05.2",
        run_label=_MIGRATION_LABEL,
        run_dir=(
            locator.resolve("wsl_staging").absolute_path / _MIGRATION_LABEL
        ),
        staging_root_alias="wsl_staging",
        host_root_alias="d_archive",
        archive_root_alias="e_archive",
        planned_archive_bytes=planned_archive_bytes,
        max_active_workspace_bytes=32 * GIB,
        projected_host_growth_bytes=32 * GIB,
        stage_plan_sha256=stage_plan_sha256,
    )


def _group_segments(
    segments: tuple[SourceSegment, ...],
) -> dict[str, tuple[SourceSegment, ...]]:
    grouped: dict[str, list[SourceSegment]] = defaultdict(list)
    for segment in segments:
        grouped[segment.run_label].append(segment)
    return {
        run_label: tuple(sorted(items, key=lambda item: item.segment_id))
        for run_label, items in sorted(grouped.items())
    }


def _identity_lookup(
    identities: tuple[SegmentIdentity, ...],
) -> dict[str, SegmentIdentity]:
    return {identity.logical_id: identity for identity in identities}


def _verify_source_against_inventory(
    segments: tuple[SourceSegment, ...],
    identities: dict[str, SegmentIdentity],
) -> int:
    byte_count = 0
    for segment in segments:
        observed = _inventory_segment(segment)
        expected = identities[segment.logical_id]
        if observed != expected:
            raise StorageGovernanceError(
                f"source changed after signed dry run: {segment.logical_id}"
            )
        byte_count += expected.byte_count
    return byte_count


def _copy_all_runs(
    *,
    governance: ExperimentStorageGovernance,
    locator: StorageRootLocator,
    grouped: dict[str, tuple[SourceSegment, ...]],
    identities: tuple[SegmentIdentity, ...],
    stage_plan_sha256: str,
) -> str:
    retention_state_root = locator.resolve("e_archive").absolute_path / ".storage-governance"
    identity_by_id = _identity_lookup(identities)
    pending = {
        run_label: segments
        for run_label, segments in grouped.items()
        if not _run_is_registered(
            run_label,
            segments,
            retention_state_root=retention_state_root,
        )
    }
    remaining_bytes = sum(
        identity_by_id[segment.logical_id].byte_count
        for segments in pending.values()
        for segment in segments
    )
    permit = governance.preflight_run(
        _capacity_request(
            locator=locator,
            planned_archive_bytes=remaining_bytes,
            stage_plan_sha256=stage_plan_sha256,
        )
    )
    _log(
        "capacity_permit_reserved",
        run_label=permit.run_label,
        planned_archive_bytes=remaining_bytes,
        observation=str(permit.observation_path),
        observation_sha256=permit.observation_sha256,
    )

    completed = len(grouped) - len(pending)
    for run_label, segments in grouped.items():
        if run_label not in pending:
            resolved = governance.resolve_run(run_label)
            _log(
                "retention_resume_verified",
                run_label=run_label,
                archive_path=str(resolved),
            )
            continue
        current_bytes = _verify_source_against_inventory(
            segments,
            identity_by_id,
        )
        _log(
            "retention_started",
            run_label=run_label,
            completed=completed,
            total=len(grouped),
            source_bytes=current_bytes,
            segment_ids=[segment.segment_id for segment in segments],
        )
        receipt = governance.retain_run(
            RetentionRequest(
                run_label=run_label,
                generation=_GENERATION,
                retention_class=RetentionClass.UNKNOWN_FULL,
                archive_root_alias="e_archive",
                archive_relative_path=segments[0].run_destination_relative_path,
                segments=tuple(
                    RetentionSegment(
                        segment_id=segment.segment_id,
                        source_path=segment.source_path,
                        logical_prefix=segment.segment_id,
                    )
                    for segment in segments
                ),
            )
        )
        resolved = governance.resolve_run(run_label)
        if resolved != receipt.archive_path:
            raise StorageGovernanceError(
                f"resolver returned the wrong archive path: {run_label}"
            )
        remaining_bytes -= current_bytes
        completed += 1
        governance.preflight_run(
            _capacity_request(
                locator=locator,
                planned_archive_bytes=remaining_bytes,
                stage_plan_sha256=stage_plan_sha256,
            )
        )
        _log(
            "retention_completed",
            run_label=run_label,
            completed=completed,
            total=len(grouped),
            archive_path=str(receipt.archive_path),
            tree_sha256=receipt.tree_sha256,
            file_count=receipt.kept_file_count,
            byte_count=receipt.kept_bytes,
            remaining_planned_bytes=remaining_bytes,
            registry_sha256=receipt.registry_sha256,
        )
    registry_path = retention_state_root / "retention_registry_v2.json"
    return _sha256(registry_path)


def _attest(
    *,
    locator: StorageRootLocator,
    identities: tuple[SegmentIdentity, ...],
    dry_runs: dict[str, tuple[Path, str]],
    evidence_root: Path,
) -> dict[str, tuple[Path, str]]:
    result: dict[str, tuple[Path, str]] = {}
    destination = locator.resolve("e_archive")
    for source_alias in ("wsl_staging", "d_archive"):
        source = locator.resolve(source_alias)
        selected = tuple(
            item for item in identities if item.source_root_alias == source_alias
        )
        mappings = tuple(
            (
                item.logical_id,
                item.source_relative_path,
                item.destination_relative_path,
            )
            for item in selected
        )
        dry_run_path, dry_run_sha256 = dry_runs[source_alias]
        attestation_path = evidence_root / f"{source_alias}-to-e-attestation.json"
        digest = write_storage_migration_attestation(
            attestation_path,
            migration_id=f"{_MIGRATION_ID}-{source_alias}-to-e",
            source_root_alias=source_alias,
            destination_root_alias="e_archive",
            source_root=source.absolute_path,
            destination_root=destination.absolute_path,
            source_volume=source.volume,
            destination_volume=destination.volume,
            mappings=mappings,
            dry_run_path=dry_run_path,
            dry_run_sha256=dry_run_sha256,
        )
        verify_storage_migration_attestation(
            attestation_path,
            expected_sha256=digest,
            source_root_alias=source_alias,
            destination_root_alias="e_archive",
            source_root=source.absolute_path,
            destination_root=destination.absolute_path,
            source_volume=source.volume,
            destination_volume=destination.volume,
            dry_run_path=dry_run_path,
            volume_probe=probe_volume_identity,
        )
        result[source_alias] = (attestation_path, digest)
        _log(
            "attestation_verified",
            source_root_alias=source_alias,
            mappings=len(mappings),
            path=str(attestation_path),
            sha256=digest,
        )
    return result


def _replay_resolver(
    governance: ExperimentStorageGovernance,
    grouped: dict[str, tuple[SourceSegment, ...]],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index, run_label in enumerate(grouped, start=1):
        path = governance.resolve_run(run_label)
        file_count, byte_count, tree_sha256 = compute_tree_identity(path)
        results.append(
            {
                "run_label": run_label,
                "generation": _GENERATION,
                "archive_path": str(path),
                "file_count": file_count,
                "byte_count": byte_count,
                "tree_sha256": tree_sha256,
                "resolver_replay": "passed",
            }
        )
        _log(
            "resolver_replay_passed",
            index=index,
            total=len(grouped),
            run_label=run_label,
            tree_sha256=tree_sha256,
        )
    return results


def _deletion_candidate_manifest(
    *,
    identities: tuple[SegmentIdentity, ...],
    locator: StorageRootLocator,
    attestations: dict[str, tuple[Path, str]],
    resolver_receipt_sha256: str,
    evidence_root: Path,
) -> tuple[Path, str]:
    records: list[dict[str, object]] = []
    for identity in identities:
        source_root = locator.resolve(identity.source_root_alias).absolute_path
        source_path = _safe_relative(source_root, identity.source_relative_path)
        destination_path = _safe_relative(
            locator.resolve("e_archive").absolute_path,
            identity.destination_relative_path,
        )
        records.append(
            {
                "logical_id": identity.logical_id,
                "source_root_alias": identity.source_root_alias,
                "source_path": str(source_path),
                "destination_path": str(destination_path),
                "file_count": identity.file_count,
                "byte_count": identity.byte_count,
                "tree_sha256": identity.tree_sha256,
                "attestation_sha256": attestations[
                    identity.source_root_alias
                ][1],
                "registry_generation": _GENERATION,
                "resolver_replay_receipt_sha256": resolver_receipt_sha256,
                "deletion_authorized": False,
            }
        )
    path = evidence_root / "source-deletion-candidates.json"
    digest = _write_signed_json(
        path,
        {
            "schema_version": "stage052-source-deletion-candidates-v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "migration_id": _MIGRATION_ID,
            "source_deletion_authorized": False,
            "requires_literal_confirmation": "确认",
            "rollback_after_source_deletion": "none_single_media_archive",
            "candidate_count": len(records),
            "candidate_bytes": sum(
                identity.byte_count for identity in identities
            ),
            "candidates": records,
        },
    )
    return path, digest


def _volume_observations(locator: StorageRootLocator) -> dict[str, object]:
    result: dict[str, object] = {}
    for alias in ("wsl_staging", "d_archive", "e_archive"):
        root = locator.resolve(alias)
        observed = probe_volume_identity(root.absolute_path)
        if observed != root.volume:
            raise StorageGovernanceError(
                f"volume identity changed during migration: {alias}"
            )
        usage = shutil.disk_usage(root.absolute_path)
        result[alias] = {
            "path": str(root.absolute_path),
            "volume": observed.to_dict(),
            "free_bytes": usage.free,
            "total_bytes": usage.total,
        }
    return result


def _stage_plan_sha256(dry_runs: dict[str, tuple[Path, str]]) -> str:
    payload = {
        alias: {"path": str(path), "sha256": digest}
        for alias, (path, digest) in sorted(dry_runs.items())
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def migrate(repo_root: Path) -> None:
    _log(
        "runtime_modules",
        stage052_campaign=str(
            getattr(sys.modules["evrptw.stage052_campaign"], "__file__", "")
        ),
        stage052_campaign_runner=str(
            getattr(sys.modules["evrptw.stage052_campaign_runner"], "__file__", "")
        ),
        storage_governance=str(
            getattr(sys.modules["evrptw.storage_governance"], "__file__", "")
        ),
    )
    locator = StorageRootLocator.from_toml(
        repo_root / "configs" / "stage052_storage_roots.local.toml"
    )
    locator.verify_all(
        probe_volume_identity,
        ("wsl_staging", "d_archive", "e_archive"),
    )
    e_root = locator.resolve("e_archive").absolute_path
    evidence_root = (
        e_root / ".storage-governance" / "stage052-migration-20260731"
    )
    progress_path = evidence_root / "inventory-progress.json"
    segments = _discover_sources(locator)
    grouped = _group_segments(segments)
    _log(
        "sources_discovered",
        segment_count=len(segments),
        run_count=len(grouped),
        volume_observations=_volume_observations(locator),
    )
    identities = _inventory(segments, progress_path=progress_path)
    dry_runs = _write_dry_runs(identities, evidence_root=evidence_root)
    stage_plan_sha256 = _stage_plan_sha256(dry_runs)
    governance = _governance(repo_root=repo_root, locator=locator)
    registry_sha256 = _copy_all_runs(
        governance=governance,
        locator=locator,
        grouped=grouped,
        identities=identities,
        stage_plan_sha256=stage_plan_sha256,
    )
    attestations = _attest(
        locator=locator,
        identities=identities,
        dry_runs=dry_runs,
        evidence_root=evidence_root,
    )
    resolver_results = _replay_resolver(governance, grouped)
    resolver_receipt_path = evidence_root / "resolver-replay-receipt.json"
    resolver_receipt_sha256 = _write_signed_json(
        resolver_receipt_path,
        {
            "schema_version": "stage052-resolver-replay-v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "migration_id": _MIGRATION_ID,
            "retention_class": RetentionClass.UNKNOWN_FULL.value,
            "scientific_prerequisite_status": "not_adjudicated",
            "source_deletion_authorized": False,
            "registry_sha256": registry_sha256,
            "results": resolver_results,
            "status": "passed",
        },
    )
    deletion_path, deletion_sha256 = _deletion_candidate_manifest(
        identities=identities,
        locator=locator,
        attestations=attestations,
        resolver_receipt_sha256=resolver_receipt_sha256,
        evidence_root=evidence_root,
    )
    final_receipt_path = evidence_root / "migration-receipt.json"
    final_receipt_sha256 = _write_signed_json(
        final_receipt_path,
        {
            "schema_version": "stage052-storage-migration-receipt-v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "migration_id": _MIGRATION_ID,
            "source_deletion_authorized": False,
            "retention_default": RetentionClass.UNKNOWN_FULL.value,
            "run_count": len(grouped),
            "segment_count": len(identities),
            "source_bytes": sum(item.byte_count for item in identities),
            "registry_sha256": registry_sha256,
            "dry_run_sha256_by_alias": {
                alias: digest
                for alias, (_path, digest) in sorted(dry_runs.items())
            },
            "attestation_sha256_by_alias": {
                alias: digest
                for alias, (_path, digest) in sorted(attestations.items())
            },
            "resolver_replay_receipt_sha256": resolver_receipt_sha256,
            "deletion_candidate_manifest_sha256": deletion_sha256,
            "deletion_candidate_manifest_path": str(deletion_path),
            "capacity_after": _volume_observations(locator),
            "status": "verified_not_deleted",
        },
    )
    governance.reconcile_permit(
        _MIGRATION_LABEL,
        outcome="retained",
        evidence_sha256=final_receipt_sha256,
    )
    _log(
        "migration_completed",
        receipt=str(final_receipt_path),
        receipt_sha256=final_receipt_sha256,
        deletion_candidate_manifest=str(deletion_path),
        deletion_candidate_manifest_sha256=deletion_sha256,
        source_deletion_authorized=False,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy, verify, and register Stage 5.2 evidence on the governed "
            "E-drive archive without deleting any source."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Reproducible-EVRPTW checkout (default: current directory)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    try:
        migrate(repo_root)
    except BaseException as error:
        _log(
            "migration_failed",
            error_type=type(error).__name__,
            error=str(error),
            source_deletion_authorized=False,
        )
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
