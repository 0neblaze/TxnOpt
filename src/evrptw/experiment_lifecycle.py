"""Fail-fast experiment lifecycle and retention-v3 orchestration.

The storage-governance module owns physical volumes and archive generations.
This module owns the higher-level experiment state machine.  New experiment
facts are append-only, signed, and deterministic: neither a runner nor an AI
caller may directly choose a retention class or an ad-hoc deletion list.
"""

from __future__ import annotations

import argparse
import ast
import atexit
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import tomllib
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final, cast

LIFECYCLE_SCHEMA_VERSION: Final = "experiment-lifecycle-v3"
CATALOG_SCHEMA_VERSION: Final = "experiment-catalog-v1"
COMPACTION_SCHEMA_VERSION: Final = "experiment-compaction-v1"
MIGRATION_SCHEMA_VERSION: Final = "experiment-lifecycle-migration-v3"
HISTORICAL_GATE_SCHEMA_VERSION: Final = "experiment-lifecycle-historical-gate-v1"
HISTORICAL_COMPACTION_SCHEMA_VERSION: Final = (
    "experiment-historical-compaction-transaction-v1"
)
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_RUN_LABEL: Final = re.compile(
    r"stage0[0-8](?:\.[0-9]+)?_[a-z0-9_]+_(?:attempt|rerun)[0-9]{2}"
)
_WRITER_LEASES: list[tuple[int, Path, Path]] = []


class LifecycleError(RuntimeError):
    """A lifecycle invariant could not be proven."""


def _sync_parent_directory(path: Path) -> None:
    """Durably commit rename metadata on POSIX filesystems."""

    if os.name == "nt":
        return
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class LifecycleState(StrEnum):
    PLANNED = "PLANNED"
    PERMITTED = "PERMITTED"
    RUNNING = "RUNNING"
    SEALED = "SEALED"
    REVIEWED = "REVIEWED"
    CLASSIFIED = "CLASSIFIED"
    RETAINED = "RETAINED"
    COMPACTED = "COMPACTED"
    CLOSED = "CLOSED"
    BLOCKED_RETENTION = "BLOCKED_RETENTION"


class ReviewerStatus(StrEnum):
    ACCEPTED = "ACCEPTED"
    FAILED_KNOWN = "FAILED_KNOWN"
    FAILED_UNKNOWN = "FAILED_UNKNOWN"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"


class RetentionClassV3(StrEnum):
    PUBLISHED_FULL = "published_full"
    CURRENT_ACCEPTED_FULL = "current_accepted_full"
    SUPERSEDED_ACCEPTED_CAPSULE = "superseded_accepted_capsule"
    UNIQUE_FAILURE_CAPSULE = "unique_failure_capsule"
    DUPLICATE_FAILURE_METADATA = "duplicate_failure_metadata"
    SUPERSEDED_METADATA = "superseded_metadata"
    REBUILDABLE = "rebuildable"
    UNKNOWN_FULL = "unknown_full"

    @property
    def preserves_full_tree(self) -> bool:
        return self in {
            RetentionClassV3.PUBLISHED_FULL,
            RetentionClassV3.CURRENT_ACCEPTED_FULL,
            RetentionClassV3.UNKNOWN_FULL,
        }


_TRANSITIONS: Final[Mapping[LifecycleState, frozenset[LifecycleState]]] = {
    LifecycleState.PLANNED: frozenset(
        {LifecycleState.PERMITTED, LifecycleState.BLOCKED_RETENTION}
    ),
    LifecycleState.PERMITTED: frozenset({LifecycleState.RUNNING}),
    LifecycleState.RUNNING: frozenset({LifecycleState.SEALED}),
    LifecycleState.SEALED: frozenset({LifecycleState.REVIEWED}),
    LifecycleState.REVIEWED: frozenset(
        {LifecycleState.CLASSIFIED, LifecycleState.BLOCKED_RETENTION}
    ),
    LifecycleState.CLASSIFIED: frozenset(
        {LifecycleState.RETAINED, LifecycleState.COMPACTED}
    ),
    LifecycleState.RETAINED: frozenset(
        {LifecycleState.COMPACTED, LifecycleState.CLOSED}
    ),
    LifecycleState.COMPACTED: frozenset({LifecycleState.CLOSED}),
    LifecycleState.BLOCKED_RETENTION: frozenset({LifecycleState.CLASSIFIED}),
    LifecycleState.CLOSED: frozenset(),
}


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_signed_json(path: Path, payload: object) -> str:
    data = _canonical_json(payload) + b"\n"
    digest = _sha256_bytes(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.{uuid.uuid4().hex}.tmp")
    previous = path.with_name(f".{path.name}.previous")
    previous_sidecar = previous.with_suffix(previous.suffix + ".sha256")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary_sidecar.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{digest}  {path.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.is_file() and not sidecar.exists():
            if path.read_bytes() != data:
                raise LifecycleError(
                    f"orphan lifecycle payload differs from retry: {path}"
                )
            os.replace(temporary_sidecar, sidecar)
            _sync_parent_directory(path)
            temporary.unlink(missing_ok=True)
            return digest
        if path.is_file() and sidecar.is_file():
            existing_data = path.read_bytes()
            existing_sidecar = sidecar.read_text(encoding="utf-8")
            fields = existing_sidecar.strip().split()
            if (
                len(fields) == 2
                and fields[1] == path.name
                and fields[0] == _sha256_bytes(existing_data)
            ):
                with previous.open("wb") as handle:
                    handle.write(existing_data)
                    handle.flush()
                    os.fsync(handle.fileno())
                with previous_sidecar.open(
                    "w", encoding="utf-8", newline="\n"
                ) as handle:
                    handle.write(existing_sidecar)
                    handle.flush()
                    os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.replace(temporary_sidecar, sidecar)
        _sync_parent_directory(path)
        previous.unlink(missing_ok=True)
        previous_sidecar.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
    return digest


def _repair_orphan_lifecycle_sidecar(path: Path) -> None:
    """Recover the data-first half of an interrupted first-generation write."""

    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or sidecar.exists():
        return
    data = path.read_bytes()
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as error:
        raise LifecycleError(f"orphan lifecycle payload is invalid: {path}") from error
    if not isinstance(payload, dict):
        raise LifecycleError(f"orphan lifecycle payload is not an object: {path}")
    temporary = sidecar.with_name(f".{sidecar.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{_sha256_bytes(data)}  {path.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, sidecar)
        _sync_parent_directory(path)
    finally:
        temporary.unlink(missing_ok=True)


def register_lifecycle_writer(*, state_root: Path, run_label: str) -> None:
    """Hold the governed writer lease until normal or exceptional process exit."""

    if not state_root.is_absolute() or _RUN_LABEL.fullmatch(run_label) is None:
        raise ValueError("lifecycle writer identity is invalid")
    lock_path = state_root / "writer-locks" / f"{run_label}.lock"
    if any(existing_lock == lock_path for _fd, _marker, existing_lock in _WRITER_LEASES):
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    if os.name == "nt":
        import msvcrt

        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
        os.lseek(descriptor, 0, os.SEEK_SET)
        locking = cast(Callable[[int, int, int], None], vars(msvcrt)["locking"])
        locking(descriptor, int(vars(msvcrt)["LK_LOCK"]), 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_SH)
    marker_path = (
        state_root / "writer-markers" / run_label / f"{os.getpid()}.json"
    )
    _write_signed_json(
        marker_path,
        {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "run_label": run_label,
            "pid": os.getpid(),
            "status": "active",
        },
    )
    _WRITER_LEASES.append((descriptor, marker_path, lock_path))

    atexit.register(
        release_lifecycle_writer,
        state_root=state_root,
        run_label=run_label,
    )


def release_lifecycle_writer(*, state_root: Path, run_label: str) -> None:
    """Release the exact process-local writer lease before sealing a run."""

    matching = [
        lease
        for lease in _WRITER_LEASES
        if lease[2] == state_root / "writer-locks" / f"{run_label}.lock"
    ]
    for descriptor, marker_path, lock_path in matching:
        if os.name == "nt":
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            locking = cast(
                Callable[[int, int, int], None], vars(msvcrt)["locking"]
            )
            locking(descriptor, int(vars(msvcrt)["LK_UNLCK"]), 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        marker_path.unlink(missing_ok=True)
        marker_path.with_suffix(marker_path.suffix + ".sha256").unlink(
            missing_ok=True
        )
        _WRITER_LEASES.remove((descriptor, marker_path, lock_path))


def _load_signed_json(path: Path) -> dict[str, object]:
    previous = path.with_name(f".{path.name}.previous")
    previous_sidecar = previous.with_suffix(previous.suffix + ".sha256")

    def read_pair(data_path: Path, sidecar_path: Path) -> dict[str, object]:
        data = data_path.read_bytes()
        sidecar = sidecar_path.read_text(encoding="utf-8")
        fields = sidecar.strip().split()
        if len(fields) != 2 or fields[1] != path.name:
            raise LifecycleError(f"invalid lifecycle sidecar: {path}")
        if fields[0] != _sha256_bytes(data):
            raise LifecycleError(f"lifecycle SHA-256 mismatch: {path}")
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as error:
            raise LifecycleError(f"invalid lifecycle JSON: {path}") from error
        if not isinstance(payload, dict):
            raise LifecycleError(f"lifecycle state is not an object: {path}")
        return cast(dict[str, object], payload)

    try:
        return read_pair(path, path.with_suffix(path.suffix + ".sha256"))
    except (LifecycleError, OSError) as primary_error:
        try:
            return read_pair(previous, previous_sidecar)
        except (LifecycleError, OSError):
            raise LifecycleError(
                f"cannot read signed lifecycle state: {path}"
            ) from primary_error


def _external_sidecars(path: Path) -> tuple[Path, ...]:
    candidates = tuple(
        dict.fromkeys(
            (
                path.with_suffix(path.suffix + ".sha256"),
                path.with_suffix(".sha256"),
            )
        )
    )
    existing = tuple(candidate for candidate in candidates if candidate.is_file())
    if not existing:
        raise LifecycleError(f"external signed JSON has no sidecar: {path}")
    return existing


def _load_external_signed_json(path: Path) -> dict[str, object]:
    """Load producer/reviewer JSON without weakening controller-state signatures."""

    try:
        data = path.read_bytes()
        digest = _sha256_bytes(data)
        for sidecar in _external_sidecars(path):
            fields = sidecar.read_text(encoding="utf-8").strip().split()
            if not fields or fields[0] != digest:
                raise LifecycleError(f"external JSON SHA-256 mismatch: {path}")
        payload = json.loads(data)
    except (OSError, json.JSONDecodeError) as error:
        raise LifecycleError(f"cannot read external signed JSON: {path}") from error
    if not isinstance(payload, dict):
        raise LifecycleError(f"external signed JSON is not an object: {path}")
    return cast(dict[str, object], payload)


def load_lifecycle_migration_ledger(
    path: Path,
    *,
    repository: Path,
) -> dict[str, object]:
    """Verify the immutable v1/v2 compatibility anchor for v3 governance."""

    payload = _load_signed_json(path)
    if (
        payload.get("schema_version") != MIGRATION_SCHEMA_VERSION
        or payload.get("new_write_schema_version") != LIFECYCLE_SCHEMA_VERSION
        or payload.get("v3_write_authority") != "e_archive/.experiment-lifecycle"
    ):
        raise LifecycleError("lifecycle migration ledger contract is invalid")
    raw_sources = payload.get("legacy_sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != 2:
        raise LifecycleError("lifecycle migration ledger legacy sources are invalid")
    sources = tuple(
        cast(dict[str, object], item) for item in raw_sources if isinstance(item, dict)
    )
    if len(sources) != len(raw_sources):
        raise LifecycleError("lifecycle migration ledger source is invalid")
    schemas = {str(item.get("schema_version")) for item in sources}
    if schemas != {"experiment-retention-registry-v2", "stage05.2-retention-v1"}:
        raise LifecycleError("lifecycle migration ledger schemas are incomplete")
    v1 = next(
        item for item in sources if item.get("schema_version") == "stage05.2-retention-v1"
    )
    relative = PurePosixPath(str(v1.get("relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise LifecycleError("lifecycle migration v1 path is unsafe")
    v1_path = repository.resolve().joinpath(*relative.parts)
    if not v1_path.is_file() or _sha256_file(v1_path) != v1.get("registry_sha256"):
        raise LifecycleError("lifecycle migration v1 registry identity differs")
    v2 = next(
        item
        for item in sources
        if item.get("schema_version") == "experiment-retention-registry-v2"
    )
    if (
        v2.get("role") != "read_only_primary_legacy"
        or v2.get("root_alias") != "e_archive"
        or not isinstance(v2.get("record_count"), int)
        or cast(int, v2["record_count"]) <= 0
        or _SHA256.fullmatch(str(v2.get("registry_sha256", ""))) is None
    ):
        raise LifecycleError("lifecycle migration v2 registry identity is invalid")
    for field in ("protected_run_labels", "pending_historical_classification"):
        labels = payload.get(field)
        if not isinstance(labels, list) or not labels or not all(
            isinstance(label, str) and _RUN_LABEL.fullmatch(label) is not None
            for label in labels
        ):
            raise LifecycleError(f"lifecycle migration {field} is invalid")
    return payload


def load_historical_migration_gate(
    path: Path,
    *,
    migration_ledger_path: Path,
    repository: Path,
) -> dict[str, object]:
    """Require one independently reviewed disposition for every pending legacy run."""

    migration = load_lifecycle_migration_ledger(
        migration_ledger_path,
        repository=repository,
    )
    from evrptw.stage052_campaign import StorageRootLocator
    from evrptw.stage052_campaign_runner import probe_volume_identity

    try:
        locator_path = (
            repository.resolve(strict=True)
            / "configs"
            / "stage052_storage_roots.local.toml"
        ).resolve()
        locator = StorageRootLocator.from_toml(locator_path)
        locator.verify_all(probe_volume_identity, ("e_archive",))
        canonical_archive_root = locator.resolve("e_archive").absolute_path.resolve(
            strict=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise LifecycleError("canonical historical archive root is unavailable") from error
    legacy_sources = cast(list[dict[str, object]], migration["legacy_sources"])
    trusted_v2 = next(
        item
        for item in legacy_sources
        if item.get("schema_version") == "experiment-retention-registry-v2"
    )
    trusted_registry_sha256 = str(trusted_v2["registry_sha256"])
    gate = _load_signed_json(path)
    pending = migration["pending_historical_classification"]
    records = gate.get("records")
    if (
        gate.get("schema_version") != HISTORICAL_GATE_SCHEMA_VERSION
        or gate.get("migration_ledger_sha256")
        != _sha256_file(migration_ledger_path)
        or not isinstance(records, list)
        or any(not isinstance(item, dict) for item in records)
    ):
        raise LifecycleError("historical migration gate identity is invalid")
    by_label = {
        str(cast(dict[str, object], item).get("run_label")): cast(
            dict[str, object], item
        )
        for item in records
    }
    if set(by_label) != set(cast(list[str], pending)):
        raise LifecycleError("historical migration gate scope is incomplete")
    allowed = {
        RetentionClassV3.SUPERSEDED_ACCEPTED_CAPSULE.value,
        RetentionClassV3.UNIQUE_FAILURE_CAPSULE.value,
        RetentionClassV3.DUPLICATE_FAILURE_METADATA.value,
        RetentionClassV3.SUPERSEDED_METADATA.value,
        RetentionClassV3.REBUILDABLE.value,
    }

    def has_archive_identity(item: Mapping[str, object]) -> bool:
        generation = PurePosixPath(
            str(item.get("archive_generation_relative_path", ""))
        )
        return (
            item.get("archive_root_alias") == "e_archive"
            and Path(str(item.get("archive_root_resolved_path", ""))).resolve()
            == canonical_archive_root
            and bool(generation.parts)
            and not generation.is_absolute()
            and ".." not in generation.parts
            and _SHA256.fullmatch(
                str(item.get("legacy_registry_sha256", ""))
            )
            is not None
            and item.get("legacy_registry_sha256") == trusted_registry_sha256
            and _SHA256.fullmatch(str(item.get("legacy_tree_sha256", "")))
            is not None
        )

    def semantic_command_options(raw: object) -> tuple[list[str], dict[str, str]]:
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise LifecycleError("historical semantic command is invalid")
        command = cast(list[str], raw)
        reviewer_python = Path(command[0]) if command else Path()
        if (
            len(command) < 3
            or not reviewer_python.is_absolute()
            or not reviewer_python.is_file()
            or not reviewer_python.name.startswith("python")
            or command[1] != "-m"
        ):
            raise LifecycleError("historical semantic command prefix differs")
        tail = command[3:]
        allowed = {
            "--migration-ledger",
            "--content-inventory",
            "--run-label",
            "--output",
            "--dependency-proof",
            "--archive-root",
        }
        if len(tail) % 2:
            raise LifecycleError("historical semantic command options are invalid")
        options: dict[str, str] = {}
        for index in range(0, len(tail), 2):
            option = tail[index]
            value = tail[index + 1]
            if option not in allowed or option in options or not value:
                raise LifecycleError("historical semantic command options are invalid")
            options[option] = value
        if set(options) - {"--dependency-proof"} != allowed - {
            "--dependency-proof"
        }:
            raise LifecycleError("historical semantic command options are incomplete")
        return command, options

    unresolved = [
        label
        for label, item in by_label.items()
        if item.get("retention_class") not in allowed
        or item.get("review_status") not in {
            ReviewerStatus.FAILED_KNOWN.value,
            ReviewerStatus.PARTIAL.value,
            ReviewerStatus.INVALID.value,
        }
        or _SHA256.fullmatch(str(item.get("review_manifest_sha256", ""))) is None
        or _SHA256.fullmatch(str(item.get("content_inventory_sha256", ""))) is None
        or not has_archive_identity(item)
    ]
    if unresolved or gate.get("status") != "complete":
        raise LifecycleError(
            "historical migration remains BLOCKED_RETENTION: "
            + ", ".join(sorted(unresolved or by_label))
        )
    gate_root = path.resolve().parent
    migration_sha256 = _sha256_file(migration_ledger_path)
    for label, item in by_label.items():
        review_relative = PurePosixPath(
            str(item.get("review_manifest_relative_path", ""))
        )
        inventory_relative = PurePosixPath(
            str(item.get("content_inventory_relative_path", ""))
        )
        if any(
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            for relative in (review_relative, inventory_relative)
        ):
            raise LifecycleError("historical migration evidence path is unsafe")
        review_path = gate_root.joinpath(*review_relative.parts)
        inventory_path = gate_root.joinpath(*inventory_relative.parts)
        try:
            if (
                _sha256_file(review_path) != item["review_manifest_sha256"]
                or _sha256_file(inventory_path)
                != item["content_inventory_sha256"]
            ):
                raise LifecycleError("historical migration evidence identity differs")
            review = _load_signed_json(review_path)
            inventory = _load_signed_json(inventory_path)
        except OSError as error:
            raise LifecycleError(
                "historical migration evidence cannot be read"
            ) from error
        if (
            review.get("run_label") != label
            or review.get("migration_ledger_sha256") != migration_sha256
            or review.get("content_inventory_sha256")
            != item["content_inventory_sha256"]
            or review.get("retention_class") != item["retention_class"]
            or review.get("status") != item["review_status"]
            or review.get("reviewer_module_name")
            != "evrptw.experiments.lifecycle_historical_review"
            or any(
                review.get(field) != item.get(field)
                or inventory.get(field) != item.get(field)
                for field in (
                    "archive_root_alias",
                    "archive_root_resolved_path",
                    "archive_generation_relative_path",
                    "legacy_registry_sha256",
                    "legacy_tree_sha256",
                )
            )
        ):
            raise LifecycleError("historical migration review binding differs")
        files = inventory.get("files")
        if (
            inventory.get("schema_version")
            != "experiment-content-inventory-v1"
            or inventory.get("run_label") != label
            or not isinstance(files, list)
            or not files
        ):
            raise LifecycleError("historical migration content inventory is invalid")
        evidence_dir = review_path.parent
        execution_path = evidence_dir / "semantic_review_execution.json"
        binding_path = gate_root / "executions" / f"{label}.json"
        execution = _load_signed_json(execution_path)
        binding = _load_signed_json(binding_path)
        command, options = semantic_command_options(execution.get("command"))
        semantic_reviewer = str(review.get("semantic_reviewer_module_name", ""))
        if command[2] != semantic_reviewer:
            raise LifecycleError("historical semantic reviewer command differs")
        module_spec = importlib.util.find_spec(semantic_reviewer)
        if module_spec is None or module_spec.origin is None:
            raise LifecycleError("historical semantic reviewer is unavailable")
        module_path = Path(module_spec.origin).resolve(strict=True)
        semantic_relative = PurePosixPath(
            str(execution.get("semantic_review_relative_path", ""))
        )
        if (
            semantic_relative.is_absolute()
            or not semantic_relative.parts
            or ".." in semantic_relative.parts
        ):
            raise LifecycleError("historical semantic review path is unsafe")
        semantic_path = evidence_dir.joinpath(*semantic_relative.parts)
        semantic = _load_signed_json(semantic_path)
        try:
            historical_migration_path = Path(
                options["--migration-ledger"]
            ).resolve(strict=True)
        except OSError as error:
            raise LifecycleError(
                "historical semantic migration ledger is unavailable"
            ) from error
        if _sha256_file(historical_migration_path) != migration_sha256:
            raise LifecycleError("historical semantic migration ledger differs")
        expected_options = {
            "--content-inventory": str(inventory_path.resolve(strict=True)),
            "--run-label": label,
            "--output": str(semantic_path.resolve(strict=True)),
            "--archive-root": str(canonical_archive_root),
        }
        if options.get("--dependency-proof") is not None:
            proof_path = Path(options["--dependency-proof"]).resolve(strict=True)
            try:
                proof_path.relative_to(evidence_dir)
            except ValueError as error:
                raise LifecycleError(
                    "historical dependency proof path is unsafe"
                ) from error
            expected_options["--dependency-proof"] = str(proof_path)
        semantic_sha256 = _sha256_file(semantic_path)
        execution_sha256 = _sha256_file(execution_path)
        semantic_retention_class = str(semantic.get("retention_class", ""))
        semantic_status = str(semantic.get("status", ""))
        semantic_failure_identity = semantic.get("failure_identity")
        if (
            semantic.get("schema_version")
            != "experiment-historical-semantic-adjudication-v1"
            or semantic_status != review.get("status")
            or semantic_status != item.get("review_status")
            or semantic_retention_class != review.get("retention_class")
            or semantic_retention_class != item.get("retention_class")
        ):
            raise LifecycleError("historical semantic disposition differs")
        if semantic_retention_class == RetentionClassV3.SUPERSEDED_METADATA.value:
            proof_option = options.get("--dependency-proof")
            if proof_option is None:
                raise LifecycleError("historical superseded metadata lacks proof")
            proof_path = Path(proof_option).resolve(strict=True)
            proof = _load_signed_json(proof_path)
            if (
                semantic.get("no_dependency_proof") is not True
                or semantic.get("dependency_proof_sha256")
                != _sha256_file(proof_path)
                or proof.get("schema_version")
                != "stage052-historical-no-dependency-proof-v1"
                or proof.get("run_label") != label
                or proof.get("migration_ledger_sha256") != migration_sha256
                or proof.get("content_inventory_sha256")
                != item["content_inventory_sha256"]
                or proof.get("reference_hits") != []
                or proof.get("scan_completed") is not True
                or proof.get("producer_module") != semantic_reviewer
            ):
                raise LifecycleError("historical no-dependency proof differs")
        if semantic_retention_class in {
            RetentionClassV3.UNIQUE_FAILURE_CAPSULE.value,
            RetentionClassV3.DUPLICATE_FAILURE_METADATA.value,
        }:
            if (
                not isinstance(semantic_failure_identity, dict)
                or not all(
                    str(semantic_failure_identity.get(field, ""))
                    for field in (
                        "failure_code",
                        "component",
                        "invariant_or_check",
                        "location",
                    )
                )
                or semantic_failure_identity != review.get("failure_identity")
            ):
                raise LifecycleError("historical failure identity differs")
            if (
                semantic_retention_class
                == RetentionClassV3.DUPLICATE_FAILURE_METADATA.value
                and (
                    not semantic.get("canonical_representative")
                    or semantic.get("canonical_representative")
                    != review.get("canonical_representative")
                )
            ):
                raise LifecycleError("historical duplicate representative differs")
        if (
            semantic_retention_class
            == RetentionClassV3.SUPERSEDED_ACCEPTED_CAPSULE.value
        ):
            supersession_proof = semantic.get("supersession_proof")
            required_hashes = (
                "accepted_review_manifest_sha256",
                "accepted_raw_manifest_sha256",
                "terminal_review_manifest_sha256",
                "terminal_review_execution_sha256",
                "successor_archive_tree_sha256",
                "successor_raw_manifest_sha256",
                "successor_review_manifest_sha256",
                "successor_review_execution_sha256",
            )
            if (
                not isinstance(supersession_proof, dict)
                or supersession_proof != review.get("supersession_proof")
                or not str(supersession_proof.get("failure_code", ""))
                or not str(supersession_proof.get("successor_run_label", ""))
                or any(
                    _SHA256.fullmatch(str(supersession_proof.get(field, "")))
                    is None
                    for field in required_hashes
                )
            ):
                raise LifecycleError(
                    "historical superseded accepted proof differs"
                )
        if (
            semantic_retention_class == RetentionClassV3.REBUILDABLE.value
            and _SHA256.fullmatch(str(semantic.get("rebuild_proof_sha256", "")))
            is None
        ):
            raise LifecycleError("historical rebuild proof is invalid")
        identity_fields = (
            "archive_root_alias",
            "archive_root_resolved_path",
            "archive_generation_relative_path",
            "legacy_registry_sha256",
            "legacy_tree_sha256",
        )
        if (
            {key: value for key, value in options.items() if key != "--migration-ledger"}
            != expected_options
            or semantic_reviewer
            != "evrptw.experiments.stage052_historical_semantic_review"
            or execution.get("run_label") != label
            or execution.get("status") != "completed"
            or execution.get("exit_code") != 0
            or execution.get("reviewer_module_name") != semantic_reviewer
            or execution.get("reviewer_module_sha256") != _sha256_file(module_path)
            or execution.get("semantic_review_sha256") != semantic_sha256
            or execution.get("migration_ledger_sha256_before") != migration_sha256
            or execution.get("migration_ledger_sha256_after") != migration_sha256
            or execution.get("content_inventory_sha256_before")
            != item["content_inventory_sha256"]
            or execution.get("content_inventory_sha256_after")
            != item["content_inventory_sha256"]
            or any(execution.get(field) != item.get(field) for field in identity_fields)
            or binding.get("run_label") != label
            or binding.get("review_execution_sha256") != execution_sha256
            or binding.get("semantic_review_sha256") != semantic_sha256
            or binding.get("migration_ledger_sha256") != migration_sha256
            or binding.get("content_inventory_sha256")
            != item["content_inventory_sha256"]
            or any(binding.get(field) != item.get(field) for field in identity_fields)
            or review.get("semantic_review_execution_sha256") != execution_sha256
            or review.get("semantic_review_sha256") != semantic_sha256
            or semantic.get("run_label") != label
            or semantic.get("reviewer_module_name") != semantic_reviewer
            or semantic.get("migration_ledger_sha256") != migration_sha256
            or semantic.get("content_inventory_sha256")
            != item["content_inventory_sha256"]
        ):
            raise LifecycleError("historical semantic execution binding differs")
    return gate


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            locking = cast(
                Callable[[int, int, int], None], vars(msvcrt)["locking"]
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
                Callable[[int, int, int], None], vars(msvcrt)["locking"]
            )
            locking(descriptor, int(vars(msvcrt)["LK_UNLCK"]), 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    experiment_id: str
    stage_id: str
    component: str
    label_pattern: str
    runner_module: str
    additional_runner_modules: tuple[str, ...]
    reviewer_module: str
    prerequisite_contracts: tuple[str, ...]
    prerequisite_paths: Mapping[str, str]
    max_batch_bytes: int
    max_run_bytes: int
    max_archive_bytes: int
    max_workspace_bytes: int
    max_workers: int
    max_threads: int
    max_processes: int
    artifact_schema: str
    publication_role: str
    allowed_failure_codes: tuple[str, ...]
    retention_policy: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", self.experiment_id) is None:
            raise ValueError("experiment_id is not canonical")
        try:
            compiled = re.compile(self.label_pattern)
        except re.error as error:
            raise ValueError("catalog label_pattern is invalid") from error
        if compiled.fullmatch("stage00_invalid_attempt00") is not None:
            raise ValueError("catalog label_pattern is too broad")
        if not self.runner_module.startswith("evrptw.experiments."):
            raise ValueError("catalog runner_module is outside experiments")
        if any(
            not module.startswith("evrptw.experiments.")
            for module in self.additional_runner_modules
        ):
            raise ValueError("catalog additional runner module is outside experiments")
        if not self.reviewer_module.startswith("evrptw.experiments."):
            raise ValueError("catalog reviewer_module is outside experiments")
        if self.reviewer_module in {
            self.runner_module,
            *self.additional_runner_modules,
        }:
            raise ValueError("catalog reviewer must be independent from the producer")
        sizes = (
            self.max_batch_bytes,
            self.max_run_bytes,
            self.max_archive_bytes,
            self.max_workspace_bytes,
        )
        limits = (self.max_workers, self.max_threads, self.max_processes)
        if any(isinstance(value, bool) or value <= 0 for value in (*sizes, *limits)):
            raise ValueError("catalog resource limits must be positive integers")
        if self.max_batch_bytes > self.max_run_bytes:
            raise ValueError("catalog batch limit exceeds run limit")
        if not self.allowed_failure_codes:
            raise ValueError("catalog must declare controlled failure codes")
        if not set(self.prerequisite_paths).issubset(self.prerequisite_contracts):
            raise ValueError("catalog prerequisite paths are not declared contracts")
        if any(
            PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or not PurePosixPath(path).parts
            for path in self.prerequisite_paths.values()
        ):
            raise ValueError("catalog prerequisite path is unsafe")

    def matches(self, run_label: str) -> bool:
        return re.fullmatch(self.label_pattern, run_label) is not None


@dataclass(frozen=True, slots=True)
class ExperimentCatalog:
    specs: tuple[ExperimentSpec, ...]
    frozen_non_top_level_entrypoints: tuple[str, ...]
    historical_semantic_reviewer_modules: tuple[str, ...]
    catalog_sha256: str

    def __post_init__(self) -> None:
        identities = [spec.experiment_id for spec in self.specs]
        if not identities or len(set(identities)) != len(identities):
            raise ValueError("catalog experiment identities must be unique")
        if _SHA256.fullmatch(self.catalog_sha256) is None:
            raise ValueError("catalog SHA-256 is invalid")
        if any(
            not item.startswith("evrptw.experiments.")
            for item in self.frozen_non_top_level_entrypoints
        ):
            raise ValueError("frozen catalog entrypoint is invalid")
        if (
            not self.historical_semantic_reviewer_modules
            or any(
                not item.startswith("evrptw.experiments.")
                for item in self.historical_semantic_reviewer_modules
            )
        ):
            raise ValueError("historical semantic reviewer catalog is invalid")

    def for_run_label(self, run_label: str) -> ExperimentSpec:
        matches = tuple(spec for spec in self.specs if spec.matches(run_label))
        if len(matches) != 1:
            raise LifecycleError(
                f"run label must match exactly one catalog entry: {run_label}"
            )
        return matches[0]

    @classmethod
    def from_toml(cls, path: Path) -> ExperimentCatalog:
        try:
            data = path.read_bytes()
            payload = tomllib.loads(data.decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            raise LifecycleError(f"cannot load experiment catalog: {path}") from error
        if payload.get("schema_version") != CATALOG_SCHEMA_VERSION:
            raise LifecycleError("unsupported experiment catalog schema")
        records = payload.get("experiments")
        if not isinstance(records, list):
            raise LifecycleError("experiment catalog records are missing")
        frozen = payload.get("frozen_non_top_level_entrypoints")
        if not isinstance(frozen, list) or not all(
            isinstance(item, str) for item in frozen
        ):
            raise LifecycleError("frozen catalog entrypoints are invalid")
        historical_reviewers = payload.get("historical_semantic_reviewer_modules")
        if not isinstance(historical_reviewers, list) or not all(
            isinstance(item, str) for item in historical_reviewers
        ):
            raise LifecycleError("historical semantic reviewer catalog is invalid")
        specs: list[ExperimentSpec] = []
        for record in records:
            if not isinstance(record, dict):
                raise LifecycleError("experiment catalog record is invalid")
            try:
                specs.append(
                    ExperimentSpec(
                        experiment_id=str(record["experiment_id"]),
                        stage_id=str(record["stage_id"]),
                        component=str(record["component"]),
                        label_pattern=str(record["label_pattern"]),
                        runner_module=str(record["runner_module"]),
                        additional_runner_modules=tuple(
                            str(item)
                            for item in record.get("additional_runner_modules", [])
                        ),
                        reviewer_module=str(record["reviewer_module"]),
                        prerequisite_contracts=tuple(
                            str(item) for item in record["prerequisite_contracts"]
                        ),
                        prerequisite_paths={
                            str(key): str(value)
                            for key, value in cast(
                                dict[object, object],
                                record.get("prerequisite_paths", {}),
                            ).items()
                        },
                        max_batch_bytes=int(record["max_batch_bytes"]),
                        max_run_bytes=int(record["max_run_bytes"]),
                        max_archive_bytes=int(record["max_archive_bytes"]),
                        max_workspace_bytes=int(record["max_workspace_bytes"]),
                        max_workers=int(record["max_workers"]),
                        max_threads=int(record["max_threads"]),
                        max_processes=int(record["max_processes"]),
                        artifact_schema=str(record["artifact_schema"]),
                        publication_role=str(record["publication_role"]),
                        allowed_failure_codes=tuple(
                            str(item) for item in record["allowed_failure_codes"]
                        ),
                        retention_policy=str(record["retention_policy"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise LifecycleError("experiment catalog record is incomplete") from error
        return cls(
            tuple(specs),
            tuple(frozen),
            tuple(historical_reviewers),
            _sha256_bytes(data),
        )


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    run_label: str
    run_dir: Path
    configuration_sha256: str
    source_sha256: str
    lock_sha256: str
    environment_sha256: str
    prerequisite_sha256_by_contract: Mapping[str, str]
    planned_archive_bytes: int
    max_batch_bytes: int
    max_run_bytes: int
    max_workspace_bytes: int
    workers: int
    threads: int
    processes: int
    fallback_allowed: bool
    io_backend: str
    io_workers: int
    batch_size: int = 1
    queue_depth: int = 1
    row_group_size: int = 1
    repository_root: Path | None = None
    configuration_path: Path | None = None
    lock_path: Path | None = None

    def __post_init__(self) -> None:
        if _RUN_LABEL.fullmatch(self.run_label) is None:
            raise ValueError("lifecycle run label is invalid")
        if not self.run_dir.is_absolute() or self.run_dir.name != self.run_label:
            raise ValueError("lifecycle run directory is invalid")
        hashes = (
            self.configuration_sha256,
            self.source_sha256,
            self.lock_sha256,
            self.environment_sha256,
            *self.prerequisite_sha256_by_contract.values(),
        )
        if any(_SHA256.fullmatch(value) is None for value in hashes):
            raise ValueError("lifecycle plan contains an invalid SHA-256")
        values = (
            self.planned_archive_bytes,
            self.max_batch_bytes,
            self.max_run_bytes,
            self.max_workspace_bytes,
            self.workers,
            self.threads,
            self.processes,
            self.io_workers,
            self.batch_size,
            self.queue_depth,
            self.row_group_size,
        )
        if any(isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError("lifecycle plan resource values must be positive")
        if not self.io_backend:
            raise ValueError("lifecycle plan requires a calibrated I/O backend")
        optional_paths = (
            self.repository_root,
            self.configuration_path,
            self.lock_path,
        )
        if any(path is not None and not path.is_absolute() for path in optional_paths):
            raise ValueError("lifecycle runtime identity paths must be absolute")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_label": self.run_label,
            "run_dir": str(self.run_dir),
            "configuration_sha256": self.configuration_sha256,
            "source_sha256": self.source_sha256,
            "lock_sha256": self.lock_sha256,
            "environment_sha256": self.environment_sha256,
            "prerequisite_sha256_by_contract": dict(
                sorted(self.prerequisite_sha256_by_contract.items())
            ),
            "planned_archive_bytes": self.planned_archive_bytes,
            "max_batch_bytes": self.max_batch_bytes,
            "max_run_bytes": self.max_run_bytes,
            "max_workspace_bytes": self.max_workspace_bytes,
            "workers": self.workers,
            "threads": self.threads,
            "processes": self.processes,
            "fallback_allowed": self.fallback_allowed,
            "io_backend": self.io_backend,
            "io_workers": self.io_workers,
            "batch_size": self.batch_size,
            "queue_depth": self.queue_depth,
            "row_group_size": self.row_group_size,
            "repository_root": str(self.repository_root or ""),
            "configuration_path": str(self.configuration_path or ""),
            "lock_path": str(self.lock_path or ""),
        }

    @property
    def plan_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self.to_dict()))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExperimentPlan:
        prerequisites = payload.get("prerequisite_sha256_by_contract")
        if not isinstance(prerequisites, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in prerequisites.items()
        ):
            raise LifecycleError("lifecycle plan prerequisites are invalid")
        fallback_allowed = payload.get("fallback_allowed")
        if not isinstance(fallback_allowed, bool):
            raise LifecycleError("lifecycle plan fallback policy is invalid")
        try:
            return cls(
                run_label=str(payload["run_label"]),
                run_dir=Path(str(payload["run_dir"])).resolve(),
                configuration_sha256=str(payload["configuration_sha256"]),
                source_sha256=str(payload["source_sha256"]),
                lock_sha256=str(payload["lock_sha256"]),
                environment_sha256=str(payload["environment_sha256"]),
                prerequisite_sha256_by_contract=cast(
                    Mapping[str, str], prerequisites
                ),
                planned_archive_bytes=int(str(payload["planned_archive_bytes"])),
                max_batch_bytes=int(str(payload["max_batch_bytes"])),
                max_run_bytes=int(str(payload["max_run_bytes"])),
                max_workspace_bytes=int(str(payload["max_workspace_bytes"])),
                workers=int(str(payload["workers"])),
                threads=int(str(payload["threads"])),
                processes=int(str(payload["processes"])),
                fallback_allowed=fallback_allowed,
                io_backend=str(payload["io_backend"]),
                io_workers=int(str(payload["io_workers"])),
                batch_size=int(str(payload.get("batch_size", 1))),
                queue_depth=int(str(payload.get("queue_depth", 1))),
                row_group_size=int(str(payload.get("row_group_size", 1))),
                repository_root=(
                    Path(str(payload["repository_root"])).resolve()
                    if payload.get("repository_root")
                    else None
                ),
                configuration_path=(
                    Path(str(payload["configuration_path"])).resolve()
                    if payload.get("configuration_path")
                    else None
                ),
                lock_path=(
                    Path(str(payload["lock_path"])).resolve()
                    if payload.get("lock_path")
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LifecycleError("lifecycle plan is incomplete") from error


@dataclass(frozen=True, slots=True)
class RunLifecycleRecord:
    run_label: str
    experiment_id: str
    state: LifecycleState
    plan_sha256: str
    catalog_sha256: str
    transition_ordinal: int
    created_at_utc: str
    updated_at_utc: str
    storage_permit_sha256: str = ""
    sealed_manifest_sha256: str = ""
    sealed_manifest_relative_path: str = ""
    reviewer_status: ReviewerStatus | None = None
    review_manifest_sha256: str = ""
    failure_code: str = ""
    failure_component: str = ""
    failure_check: str = ""
    failure_location: str = ""
    root_cause_id: str = ""
    canonical_representative: str = ""
    retention_class: RetentionClassV3 | None = None
    retention_receipt_sha256: str = ""
    content_inventory_sha256: str = ""
    compaction_plan_sha256: str = ""
    compaction_receipt_sha256: str = ""
    superseded_by: str = ""
    blocked_reason: str = ""

    def __post_init__(self) -> None:
        if _RUN_LABEL.fullmatch(self.run_label) is None:
            raise ValueError("lifecycle record run label is invalid")
        if _SHA256.fullmatch(self.plan_sha256) is None:
            raise ValueError("lifecycle record plan SHA-256 is invalid")
        if _SHA256.fullmatch(self.catalog_sha256) is None:
            raise ValueError("lifecycle record catalog SHA-256 is invalid")
        if self.transition_ordinal < 0:
            raise ValueError("lifecycle transition ordinal is invalid")
        for value in (
            self.storage_permit_sha256,
            self.sealed_manifest_sha256,
            self.review_manifest_sha256,
            self.retention_receipt_sha256,
            self.content_inventory_sha256,
            self.compaction_plan_sha256,
            self.compaction_receipt_sha256,
        ):
            if value and _SHA256.fullmatch(value) is None:
                raise ValueError("lifecycle record contains an invalid SHA-256")
        if self.superseded_by and _RUN_LABEL.fullmatch(self.superseded_by) is None:
            raise ValueError("lifecycle superseding run label is invalid")
        sealed_relative = PurePosixPath(self.sealed_manifest_relative_path)
        if self.sealed_manifest_relative_path and (
            sealed_relative.is_absolute() or ".." in sealed_relative.parts
        ):
            raise ValueError("sealed manifest relative path is unsafe")
        if self.state == LifecycleState.BLOCKED_RETENTION and not self.blocked_reason:
            raise ValueError("blocked lifecycle record requires a reason")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "run_label": self.run_label,
            "experiment_id": self.experiment_id,
            "state": self.state.value,
            "plan_sha256": self.plan_sha256,
            "catalog_sha256": self.catalog_sha256,
            "transition_ordinal": self.transition_ordinal,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "storage_permit_sha256": self.storage_permit_sha256,
            "sealed_manifest_sha256": self.sealed_manifest_sha256,
            "sealed_manifest_relative_path": self.sealed_manifest_relative_path,
            "reviewer_status": (
                self.reviewer_status.value if self.reviewer_status is not None else None
            ),
            "review_manifest_sha256": self.review_manifest_sha256,
            "failure_code": self.failure_code,
            "failure_component": self.failure_component,
            "failure_check": self.failure_check,
            "failure_location": self.failure_location,
            "root_cause_id": self.root_cause_id,
            "canonical_representative": self.canonical_representative,
            "retention_class": (
                self.retention_class.value if self.retention_class is not None else None
            ),
            "retention_receipt_sha256": self.retention_receipt_sha256,
            "content_inventory_sha256": self.content_inventory_sha256,
            "compaction_plan_sha256": self.compaction_plan_sha256,
            "compaction_receipt_sha256": self.compaction_receipt_sha256,
            "superseded_by": self.superseded_by,
            "blocked_reason": self.blocked_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RunLifecycleRecord:
        if payload.get("schema_version") != LIFECYCLE_SCHEMA_VERSION:
            raise LifecycleError("unsupported lifecycle record schema")
        try:
            raw_status = payload.get("reviewer_status")
            raw_class = payload.get("retention_class")
            return cls(
                run_label=str(payload["run_label"]),
                experiment_id=str(payload["experiment_id"]),
                state=LifecycleState(str(payload["state"])),
                plan_sha256=str(payload["plan_sha256"]),
                catalog_sha256=str(payload["catalog_sha256"]),
                transition_ordinal=int(str(payload["transition_ordinal"])),
                created_at_utc=str(payload["created_at_utc"]),
                updated_at_utc=str(payload["updated_at_utc"]),
                storage_permit_sha256=str(payload.get("storage_permit_sha256", "")),
                sealed_manifest_sha256=str(
                    payload.get("sealed_manifest_sha256", "")
                ),
                sealed_manifest_relative_path=str(
                    payload.get("sealed_manifest_relative_path", "")
                ),
                reviewer_status=(
                    ReviewerStatus(str(raw_status)) if raw_status is not None else None
                ),
                review_manifest_sha256=str(
                    payload.get("review_manifest_sha256", "")
                ),
                failure_code=str(payload.get("failure_code", "")),
                failure_component=str(payload.get("failure_component", "")),
                failure_check=str(payload.get("failure_check", "")),
                failure_location=str(payload.get("failure_location", "")),
                root_cause_id=str(payload.get("root_cause_id", "")),
                canonical_representative=str(
                    payload.get("canonical_representative", "")
                ),
                retention_class=(
                    RetentionClassV3(str(raw_class)) if raw_class is not None else None
                ),
                retention_receipt_sha256=str(
                    payload.get("retention_receipt_sha256", "")
                ),
                content_inventory_sha256=str(
                    payload.get("content_inventory_sha256", "")
                ),
                compaction_plan_sha256=str(
                    payload.get("compaction_plan_sha256", "")
                ),
                compaction_receipt_sha256=str(
                    payload.get("compaction_receipt_sha256", "")
                ),
                superseded_by=str(payload.get("superseded_by", "")),
                blocked_reason=str(payload.get("blocked_reason", "")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LifecycleError("lifecycle record is invalid") from error


@dataclass(frozen=True, slots=True)
class RetentionDecision:
    run_label: str
    retention_class: RetentionClassV3
    reason_code: str
    blocked: bool
    canonical_representative: str = ""


@dataclass(frozen=True, slots=True)
class FileIdentity:
    relative_path: str
    byte_count: int
    sha256: str
    modified_time_ns: int

    def __post_init__(self) -> None:
        relative = PurePosixPath(self.relative_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("compaction file path is unsafe")
        if self.byte_count < 0 or self.modified_time_ns < 0:
            raise ValueError("compaction file stat is invalid")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("compaction file SHA-256 is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
            "modified_time_ns": self.modified_time_ns,
        }


def _validated_content_inventory(
    path: Path,
    *,
    run_label: str,
) -> tuple[dict[str, object], tuple[FileIdentity, ...]]:
    inventory = _load_signed_json(path)
    if (
        inventory.get("schema_version") != "experiment-content-inventory-v1"
        or inventory.get("run_label") != run_label
        or inventory.get("backend_calibrated") is not True
        or inventory.get("implicit_fallback") is not False
    ):
        raise LifecycleError("content inventory contract is invalid")
    raw_files = inventory.get("files")
    if not isinstance(raw_files, list):
        raise LifecycleError("content inventory files are invalid")
    try:
        identities = tuple(
            FileIdentity(
                relative_path=str(cast(dict[str, object], item)["relative_path"]),
                byte_count=int(str(cast(dict[str, object], item)["byte_count"])),
                sha256=str(cast(dict[str, object], item)["sha256"]),
                modified_time_ns=int(
                    str(cast(dict[str, object], item)["modified_time_ns"])
                ),
            )
            for item in raw_files
            if isinstance(item, dict)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LifecycleError("content inventory file identity is invalid") from error
    if len(identities) != len(raw_files) or len(
        {item.relative_path for item in identities}
    ) != len(identities):
        raise LifecycleError("content inventory file set is invalid")
    if _tree_sha256(identities) != inventory.get("source_tree_sha256"):
        raise LifecycleError("content inventory tree SHA-256 differs")
    return inventory, identities


def _inventory_from_manifests(
    *,
    run_label: str,
    run_dir: Path,
    raw_manifest_path: Path,
    raw_manifest: Mapping[str, object],
    review_manifest_path: Path,
    review_manifest: Mapping[str, object],
    io_backend: str,
    io_workers: int,
) -> dict[str, object]:
    """Reuse producer/reviewer checksums as the incremental content inventory."""

    raw_artifacts = raw_manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise LifecycleError(
            "sealed raw manifest has no incremental artifact inventory"
        )
    identities: dict[str, FileIdentity] = {}

    def add(relative: str, *, byte_count: int, sha256: str) -> None:
        candidate = PurePosixPath(relative)
        if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
            raise LifecycleError("manifest inventory path is unsafe")
        path = run_dir.joinpath(*candidate.parts)
        try:
            stat = path.stat()
        except OSError as error:
            raise LifecycleError(f"manifest inventory file is missing: {relative}") from error
        if (
            path.is_symlink()
            or not path.is_file()
            or stat.st_size != byte_count
            or _SHA256.fullmatch(sha256) is None
        ):
            raise LifecycleError(f"manifest inventory identity is invalid: {relative}")
        identity = FileIdentity(relative, byte_count, sha256, stat.st_mtime_ns)
        existing = identities.get(relative)
        if existing is not None and existing != identity:
            raise LifecycleError(f"manifest inventory identity conflicts: {relative}")
        identities[relative] = identity

    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            raise LifecycleError("raw manifest artifact inventory is invalid")
        try:
            add(
                str(raw["relative_path"]),
                byte_count=int(str(raw["byte_size"])),
                sha256=str(raw["checksum"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LifecycleError("raw manifest artifact identity is incomplete") from error
    review_files = review_manifest.get("files")
    if not isinstance(review_files, dict) or not all(
        isinstance(relative, str) and isinstance(sha256, str)
        for relative, sha256 in review_files.items()
    ):
        raise LifecycleError("review manifest file inventory is invalid")
    for relative, sha256 in cast(dict[str, str], review_files).items():
        path = review_manifest_path.parent / PurePosixPath(relative)
        add(
            path.relative_to(run_dir).as_posix(),
            byte_count=path.stat().st_size,
            sha256=sha256,
        )
    governed_documents = (
        raw_manifest_path,
        *_external_sidecars(raw_manifest_path),
        review_manifest_path,
        *_external_sidecars(review_manifest_path),
        review_manifest_path.parent / "review_execution.json",
        review_manifest_path.parent / "review_execution.json.sha256",
    )
    for path in governed_documents:
        relative = path.relative_to(run_dir).as_posix()
        add(relative, byte_count=path.stat().st_size, sha256=_sha256_file(path))
    observed: set[str] = set()
    for root, directories, files in os.walk(run_dir):
        directories.sort()
        files.sort()
        root_path = Path(root)
        if any((root_path / name).is_symlink() for name in directories):
            raise LifecycleError("manifest inventory contains a directory symlink")
        for name in files:
            path = root_path / name
            if path.is_symlink():
                raise LifecycleError("manifest inventory contains a file symlink")
            observed.add(path.relative_to(run_dir).as_posix())
    if observed != set(identities):
        missing = sorted(observed - set(identities))[:5]
        extra = sorted(set(identities) - observed)[:5]
        raise LifecycleError(
            f"manifest inventory file set differs: unlisted={missing}, missing={extra}"
        )
    ordered = tuple(sorted(identities.values(), key=lambda item: item.relative_path))
    return {
        "schema_version": "experiment-content-inventory-v1",
        "run_label": run_label,
        "source_tree_sha256": _tree_sha256(ordered),
        "files": [item.to_dict() for item in ordered],
        "hashed_bytes_during_write": sum(item.byte_count for item in ordered),
        "scanned_bytes_during_write": 0,
        "io_backend": io_backend,
        "io_workers": io_workers,
        "backend_calibrated": True,
        "implicit_fallback": False,
    }


def _verify_sealed_artifact_content(
    *, run_dir: Path, raw_manifest: Mapping[str, object]
) -> dict[str, tuple[int, int]]:
    """Hash each producer-declared artifact once after reviewer execution."""

    artifacts = raw_manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise LifecycleError("sealed raw manifest has no artifact inventory")
    snapshot: dict[str, tuple[int, int]] = {}
    for raw in artifacts:
        if not isinstance(raw, dict):
            raise LifecycleError("sealed raw artifact identity is invalid")
        relative = PurePosixPath(str(raw.get("relative_path", "")))
        expected_size = raw.get("byte_size")
        expected_sha256 = str(raw.get("checksum", ""))
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or _SHA256.fullmatch(expected_sha256) is None
        ):
            raise LifecycleError("sealed raw artifact identity is invalid")
        path = run_dir.joinpath(*relative.parts)
        stat_before = path.stat()
        if (
            not path.is_file()
            or path.is_symlink()
            or stat_before.st_size != expected_size
            or _sha256_file(path) != expected_sha256
        ):
            raise LifecycleError(
                f"independent reviewer changed sealed raw artifact: {relative}"
            )
        stat_after = path.stat()
        if (
            stat_after.st_size != stat_before.st_size
            or stat_after.st_mtime_ns != stat_before.st_mtime_ns
        ):
            raise LifecycleError(
                f"sealed raw artifact changed while hashing: {relative}"
            )
        snapshot[relative.as_posix()] = (
            stat_after.st_size,
            stat_after.st_mtime_ns,
        )
    return snapshot


def _verify_sealed_artifact_metadata(
    *, run_dir: Path, snapshot: Mapping[str, tuple[int, int]]
) -> None:
    for relative, expected in snapshot.items():
        path = run_dir.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.is_symlink():
            raise LifecycleError("sealed raw artifact disappeared after review")
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != expected:
            raise LifecycleError("sealed raw artifact changed after review validation")


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    run_label: str
    source_root: Path
    source_tree_sha256: str
    retention_class: RetentionClassV3
    keep: tuple[FileIdentity, ...]
    delete: tuple[FileIdentity, ...]
    scanned_bytes: int
    hashed_bytes: int
    scan_passes: int
    hash_passes: int
    io_backend: str
    io_workers: int

    def __post_init__(self) -> None:
        if (
            _SHA256.fullmatch(self.source_tree_sha256) is None
            or not self.io_backend
            or self.io_workers <= 0
            or self.scan_passes != 1
            or self.hash_passes != 1
            or self.scanned_bytes != sum(
                item.byte_count for item in (*self.keep, *self.delete)
            )
            or self.hashed_bytes != self.scanned_bytes
        ):
            raise ValueError("compaction plan inventory evidence is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": COMPACTION_SCHEMA_VERSION,
            "state": "PREPARED",
            "run_label": self.run_label,
            "source_root": str(self.source_root),
            "source_tree_sha256": self.source_tree_sha256,
            "retention_class": self.retention_class.value,
            "keep": [item.to_dict() for item in self.keep],
            "delete": [item.to_dict() for item in self.delete],
            "scanned_bytes": self.scanned_bytes,
            "hashed_bytes": self.hashed_bytes,
            "scan_passes": self.scan_passes,
            "hash_passes": self.hash_passes,
            "io_backend": self.io_backend,
            "io_workers": self.io_workers,
        }

    @property
    def plan_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self.to_dict()))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CompactionPlan:
        def identities(field: str) -> tuple[FileIdentity, ...]:
            raw = payload.get(field)
            if not isinstance(raw, list):
                raise LifecycleError("compaction plan file identities are invalid")
            try:
                result = tuple(
                    FileIdentity(
                        relative_path=str(item["relative_path"]),
                        byte_count=int(str(item["byte_count"])),
                        sha256=str(item["sha256"]),
                        modified_time_ns=int(str(item["modified_time_ns"])),
                    )
                    for item in raw
                    if isinstance(item, dict)
                )
            except (KeyError, TypeError, ValueError) as error:
                raise LifecycleError(
                    "compaction plan file identities are invalid"
                ) from error
            if len(result) != len(raw):
                raise LifecycleError("compaction plan file identities are invalid")
            return result

        if (
            payload.get("schema_version") != COMPACTION_SCHEMA_VERSION
            or payload.get("state") != "PREPARED"
        ):
            raise LifecycleError("compaction plan schema or state is invalid")
        try:
            return cls(
                run_label=str(payload["run_label"]),
                source_root=Path(str(payload["source_root"])).resolve(),
                source_tree_sha256=str(payload["source_tree_sha256"]),
                retention_class=RetentionClassV3(str(payload["retention_class"])),
                keep=identities("keep"),
                delete=identities("delete"),
                scanned_bytes=int(str(payload["scanned_bytes"])),
                hashed_bytes=int(str(payload["hashed_bytes"])),
                scan_passes=int(str(payload["scan_passes"])),
                hash_passes=int(str(payload["hash_passes"])),
                io_backend=str(payload["io_backend"]),
                io_workers=int(str(payload["io_workers"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LifecycleError("compaction plan is invalid") from error


@dataclass(frozen=True, slots=True)
class CompactionReceipt:
    run_label: str
    released_bytes: int
    deleted_file_count: int
    final_tree_sha256: str
    delete_traversals: int
    compaction_plan_sha256: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class ClosePerformance:
    hashed_bytes: int
    source_bytes: int
    scan_passes: int
    delete_traversals: int
    duplicate_hashed_bytes: int
    backend_calibrated: bool
    implicit_fallback: bool
    throughput_mib_per_second: float
    cpu_utilization_percent: float
    peak_memory_bytes: int
    io_utilization_percent: float
    close_wall_seconds: float
    backend: str
    workers: int

    def verify(self) -> None:
        failures: list[str] = []
        counts = (
            self.hashed_bytes,
            self.source_bytes,
            self.scan_passes,
            self.delete_traversals,
            self.duplicate_hashed_bytes,
            self.peak_memory_bytes,
            self.workers,
        )
        if any(isinstance(value, bool) or value < 0 for value in counts):
            failures.append("close counters contain a negative value")
        rates = (
            self.throughput_mib_per_second,
            self.cpu_utilization_percent,
            self.io_utilization_percent,
            self.close_wall_seconds,
        )
        if any(not math.isfinite(value) for value in rates):
            failures.append("close performance contains a non-finite value")
        if self.hashed_bytes > self.source_bytes:
            failures.append("same file content was hashed more than once")
        if self.scan_passes > 1:
            failures.append("close used more than one inventory traversal")
        if self.delete_traversals > 1:
            failures.append("close used more than one deletion traversal")
        if self.duplicate_hashed_bytes:
            failures.append("duplicate hash bytes are nonzero")
        if not self.backend_calibrated:
            failures.append("I/O backend is not calibrated")
        if self.implicit_fallback:
            failures.append("implicit I/O fallback occurred")
        if self.throughput_mib_per_second <= 0:
            failures.append("close throughput is not positive")
        if self.cpu_utilization_percent < 0:
            failures.append("close CPU utilization is invalid")
        if self.peak_memory_bytes < 0:
            failures.append("close peak memory is invalid")
        if self.io_utilization_percent < 0:
            failures.append("close I/O utilization is invalid")
        if self.close_wall_seconds <= 0:
            failures.append("close wall time is not positive")
        if not self.backend or self.workers <= 0:
            failures.append("close backend identity is incomplete")
        if failures:
            raise LifecycleError("close performance gate failed: " + "; ".join(failures))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ClosePerformance:
        booleans = (payload.get("backend_calibrated"), payload.get("implicit_fallback"))
        if not all(isinstance(value, bool) for value in booleans):
            raise LifecycleError("close performance booleans are invalid")
        try:
            return cls(
                hashed_bytes=int(str(payload["hashed_bytes"])),
                source_bytes=int(str(payload["source_bytes"])),
                scan_passes=int(str(payload["scan_passes"])),
                delete_traversals=int(str(payload["delete_traversals"])),
                duplicate_hashed_bytes=int(str(payload["duplicate_hashed_bytes"])),
                backend_calibrated=cast(bool, booleans[0]),
                implicit_fallback=cast(bool, booleans[1]),
                throughput_mib_per_second=float(
                    str(payload["throughput_mib_per_second"])
                ),
                cpu_utilization_percent=float(str(payload["cpu_utilization_percent"])),
                peak_memory_bytes=int(str(payload["peak_memory_bytes"])),
                io_utilization_percent=float(str(payload["io_utilization_percent"])),
                close_wall_seconds=float(str(payload["close_wall_seconds"])),
                backend=str(payload["backend"]),
                workers=int(str(payload["workers"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LifecycleError("close performance evidence is incomplete") from error


def _tree_sha256(identities: Sequence[FileIdentity]) -> str:
    digest = hashlib.sha256()
    for item in sorted(identities, key=lambda candidate: candidate.relative_path):
        digest.update(item.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.byte_count).encode("ascii"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _verify_tree_content(
    root: Path,
    expected: Sequence[FileIdentity],
) -> tuple[str, int, int, dict[str, tuple[int, int]]]:
    """Perform the single close-time content pass over an archived generation."""

    expected_by_path = {item.relative_path: item for item in expected}
    observed: list[FileIdentity] = []
    observed_paths: set[str] = set()
    for directory, directories, files in os.walk(root):
        directories.sort()
        files.sort()
        directory_path = Path(directory)
        if any((directory_path / name).is_symlink() for name in directories):
            raise LifecycleError("retention archive contains a directory symlink")
        for name in files:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise LifecycleError("retention archive contains a non-regular file")
            relative = path.relative_to(root).as_posix()
            identity = expected_by_path.get(relative)
            if identity is None:
                raise LifecycleError(
                    f"retention archive contains an untracked file: {relative}"
                )
            stat = path.stat()
            if stat.st_size != identity.byte_count:
                raise LifecycleError(
                    f"retention archive byte count differs: {relative}"
                )
            sha256 = _sha256_file(path)
            if sha256 != identity.sha256:
                raise LifecycleError(f"retention archive content differs: {relative}")
            observed_paths.add(relative)
            observed.append(
                FileIdentity(relative, stat.st_size, sha256, stat.st_mtime_ns)
            )
    missing = set(expected_by_path) - observed_paths
    if missing:
        raise LifecycleError(
            f"retention archive is missing inventory files: {sorted(missing)}"
        )
    return (
        _tree_sha256(observed),
        len(observed),
        sum(item.byte_count for item in observed),
        {
            item.relative_path: (item.byte_count, item.modified_time_ns)
            for item in observed
        },
    )


def _verify_tree_metadata_snapshot(
    root: Path, expected: Mapping[str, tuple[int, int]]
) -> None:
    """Close the content-hash TOCTOU window without a second content pass."""

    observed: dict[str, tuple[int, int]] = {}
    for directory, directories, files in os.walk(root):
        directories.sort()
        files.sort()
        directory_path = Path(directory)
        if any((directory_path / name).is_symlink() for name in directories):
            raise LifecycleError("retention archive contains a directory symlink")
        for name in files:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise LifecycleError("retention archive contains a non-regular file")
            stat = path.stat()
            observed[path.relative_to(root).as_posix()] = (
                stat.st_size,
                stat.st_mtime_ns,
            )
    if observed != expected:
        raise LifecycleError("retention archive changed after close-time hashing")


def _verify_tree_layout(
    root: Path,
    expected: Sequence[FileIdentity],
) -> tuple[int, int]:
    """Verify archive names and sizes before retention without rehashing content."""

    expected_by_path = {item.relative_path: item for item in expected}
    observed_paths: set[str] = set()
    observed_bytes = 0
    for directory, directories, files in os.walk(root):
        directories.sort()
        files.sort()
        directory_path = Path(directory)
        if any((directory_path / name).is_symlink() for name in directories):
            raise LifecycleError("retention archive contains a directory symlink")
        for name in files:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            expected_identity = expected_by_path.get(relative)
            if path.is_symlink() or not path.is_file() or expected_identity is None:
                raise LifecycleError("retention archive layout differs")
            size = path.stat().st_size
            if size != expected_identity.byte_count:
                raise LifecycleError(
                    f"retention archive byte count differs: {relative}"
                )
            observed_paths.add(relative)
            observed_bytes += size
    if observed_paths != set(expected_by_path):
        raise LifecycleError("retention archive file set differs")
    return len(observed_paths), observed_bytes


def _capsule_keep(
    relative_path: str,
    retention_class: RetentionClassV3,
    *,
    representative_shard_prefix: PurePosixPath | None = None,
) -> bool:
    path = PurePosixPath(relative_path)
    parts = set(path.parts)
    name = path.name.casefold()
    metadata = (
        "control" in parts
        or "review" in parts
        or "logs" in parts
        or "resource" in name
        or "manifest" in name
        or "checksum" in name
        or name.endswith(".sha256")
        or "failure" in name
        or "provenance" in name
        or "environment" in name
    )
    if retention_class in {
        RetentionClassV3.DUPLICATE_FAILURE_METADATA,
        RetentionClassV3.SUPERSEDED_METADATA,
    }:
        return metadata
    if retention_class in {
        RetentionClassV3.SUPERSEDED_ACCEPTED_CAPSULE,
        RetentionClassV3.UNIQUE_FAILURE_CAPSULE,
    }:
        if retention_class == RetentionClassV3.UNIQUE_FAILURE_CAPSULE:
            if representative_shard_prefix is None:
                return True
            prefix_parts = representative_shard_prefix.parts
            if path.parts[: len(prefix_parts)] == prefix_parts:
                return True
        return metadata or "solution" in name or "summary" in name
    return retention_class.preserves_full_tree


def _failure_shard_prefix(location: str) -> PurePosixPath | None:
    path = PurePosixPath(location)
    if (
        path.is_absolute()
        or len(path.parts) < 3
        or path.parts[0] in {"control", "review", "logs"}
    ):
        return None
    length = 3 if re.fullmatch(r"batch[0-9]{4}", path.parts[0]) else 2
    return PurePosixPath(*path.parts[:length])


def _historical_compaction_inputs(
    *,
    repository: Path,
    migration_ledger_path: Path,
    historical_gate_path: Path,
    archive_root: Path,
    run_label: str,
) -> tuple[CompactionPlan, dict[str, object], Path, Path]:
    """Rebuild one historical compaction plan from the signed migration gate."""

    gate = load_historical_migration_gate(
        historical_gate_path,
        migration_ledger_path=migration_ledger_path,
        repository=repository,
    )
    raw_records = gate.get("records")
    if not isinstance(raw_records, list):
        raise LifecycleError("historical migration gate records are invalid")
    matches = [
        cast(dict[str, object], item)
        for item in raw_records
        if isinstance(item, dict) and item.get("run_label") == run_label
    ]
    if len(matches) != 1:
        raise LifecycleError("historical compaction run is not uniquely gated")
    gate_record = matches[0]
    try:
        retention_class = RetentionClassV3(str(gate_record["retention_class"]))
    except (KeyError, ValueError) as error:
        raise LifecycleError("historical compaction retention class is invalid") from error
    if retention_class not in {
        RetentionClassV3.SUPERSEDED_METADATA,
        RetentionClassV3.SUPERSEDED_ACCEPTED_CAPSULE,
    }:
        raise LifecycleError("historical compaction class does not permit reduction")
    resolved_archive = archive_root.resolve()
    if (
        gate_record.get("archive_root_alias") != "e_archive"
        or Path(str(gate_record.get("archive_root_resolved_path", ""))).resolve()
        != resolved_archive
    ):
        raise LifecycleError("historical compaction archive root differs")
    generation_relative = PurePosixPath(
        str(gate_record.get("archive_generation_relative_path", ""))
    )
    if (
        generation_relative.is_absolute()
        or ".." in generation_relative.parts
        or run_label not in generation_relative.parts
        or re.fullmatch(r"generation-[0-9]{4}", generation_relative.name) is None
    ):
        raise LifecycleError("historical compaction generation path is invalid")
    source_root = resolved_archive.joinpath(*generation_relative.parts).resolve()
    if (
        not source_root.is_dir()
        or source_root.is_symlink()
        or not source_root.is_relative_to(resolved_archive)
    ):
        raise LifecycleError("historical compaction source generation is invalid")
    evidence_root = historical_gate_path.resolve(strict=True).parent

    def evidence_path(field: str) -> Path:
        relative = PurePosixPath(str(gate_record.get(field, "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise LifecycleError("historical compaction evidence path is unsafe")
        path = evidence_root.joinpath(*relative.parts).resolve(strict=True)
        if not path.is_relative_to(evidence_root):
            raise LifecycleError("historical compaction evidence escapes its root")
        return path

    inventory_path = evidence_path("content_inventory_relative_path")
    review_path = evidence_path("review_manifest_relative_path")
    if (
        _sha256_file(inventory_path)
        != gate_record.get("content_inventory_sha256")
        or _sha256_file(review_path) != gate_record.get("review_manifest_sha256")
    ):
        raise LifecycleError("historical compaction evidence identity differs")
    inventory = _load_signed_json(inventory_path)
    review = _load_signed_json(review_path)
    if (
        inventory.get("schema_version") != "experiment-content-inventory-v1"
        or inventory.get("run_label") != run_label
        or inventory.get("archive_root_alias") != "e_archive"
        or inventory.get("archive_root_resolved_path") != str(resolved_archive)
        or inventory.get("archive_generation_relative_path")
        != generation_relative.as_posix()
        or inventory.get("legacy_registry_sha256")
        != gate_record.get("legacy_registry_sha256")
        or inventory.get("legacy_tree_sha256")
        != gate_record.get("legacy_tree_sha256")
        or inventory.get("tree_sha256") != gate_record.get("legacy_tree_sha256")
        or inventory.get("scan_passes") != 1
        or review.get("run_label") != run_label
        or review.get("retention_class") != retention_class.value
        or review.get("content_inventory_sha256") != _sha256_file(inventory_path)
    ):
        raise LifecycleError("historical compaction inventory binding differs")
    raw_files = inventory.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise LifecycleError("historical compaction inventory files are invalid")
    try:
        identities = tuple(
            FileIdentity(
                relative_path=str(item["relative_path"]),
                byte_count=int(str(item["byte_count"])),
                sha256=str(item["sha256"]),
                modified_time_ns=int(str(item["modified_time_ns"])),
            )
            for item in raw_files
            if isinstance(item, dict)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LifecycleError(
            "historical compaction file identity is invalid"
        ) from error
    historical_tree_digest = hashlib.sha256()
    for item in identities:
        historical_tree_digest.update(
            (
                json.dumps(
                    {
                        "relative_path": item.relative_path,
                        "byte_count": item.byte_count,
                        "sha256": item.sha256,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        )
    if (
        len(identities) != len(raw_files)
        or len({item.relative_path for item in identities}) != len(identities)
        or historical_tree_digest.hexdigest() != inventory.get("tree_sha256")
        or inventory.get("file_count") != len(identities)
        or inventory.get("byte_count")
        != sum(item.byte_count for item in identities)
    ):
        raise LifecycleError("historical compaction inventory tree differs")
    keep = tuple(
        item
        for item in identities
        if _capsule_keep(item.relative_path, retention_class)
    )
    delete = tuple(item for item in identities if item not in keep)
    hash_backend = str(inventory.get("hash_backend", ""))
    hash_workers = inventory.get("hash_workers")
    if (
        not keep
        or not delete
        or not hash_backend
        or isinstance(hash_workers, bool)
        or not isinstance(hash_workers, int)
        or hash_workers <= 0
    ):
        raise LifecycleError("historical compaction reduction plan is invalid")
    plan = CompactionPlan(
        run_label=run_label,
        source_root=source_root,
        source_tree_sha256=str(inventory["tree_sha256"]),
        retention_class=retention_class,
        keep=keep,
        delete=delete,
        scanned_bytes=sum(item.byte_count for item in identities),
        hashed_bytes=sum(item.byte_count for item in identities),
        scan_passes=1,
        hash_passes=1,
        io_backend=hash_backend,
        io_workers=hash_workers,
    )
    return plan, gate_record, inventory_path, review_path


class ExperimentLifecycleController:
    """Single-writer controller for one repository's top-level experiments."""

    def __init__(
        self,
        *,
        catalog: ExperimentCatalog,
        state_root: Path,
        storage_state_root: Path | None = None,
        capacity_state_root: Path | None = None,
        archive_root: Path | None = None,
    ) -> None:
        if not state_root.is_absolute():
            raise ValueError("lifecycle state_root must be absolute")
        self.catalog = catalog
        self.state_root = state_root
        self.storage_state_root = (
            storage_state_root
            if storage_state_root is not None
            else state_root.parent / ".storage-governance"
        )
        self.capacity_state_root = (
            capacity_state_root
            if capacity_state_root is not None
            else self.storage_state_root
        )
        self.archive_root = archive_root if archive_root is not None else state_root.parent
        if (
            not self.storage_state_root.is_absolute()
            or not self.capacity_state_root.is_absolute()
            or not self.archive_root.is_absolute()
        ):
            raise ValueError("lifecycle storage bindings must be absolute")

    def prepare_historical_compaction(
        self,
        run_label: str,
        *,
        repository: Path,
        migration_ledger_path: Path,
        historical_gate_path: Path,
    ) -> tuple[CompactionPlan, Path, Path]:
        """Persist an exact plan without importing or mutating the historical run."""

        if self._record_path(run_label).exists():
            raise LifecycleError("historical compaction run is already lifecycle-managed")
        plan, gate_record, inventory_path, review_path = _historical_compaction_inputs(
            repository=repository,
            migration_ledger_path=migration_ledger_path,
            historical_gate_path=historical_gate_path,
            archive_root=self.archive_root,
            run_label=run_label,
        )
        plan_path = (
            self.state_root
            / "historical-compaction"
            / "plans"
            / run_label
            / f"{plan.plan_sha256}.json"
        )
        prepared_path = (
            self.state_root
            / "historical-compaction"
            / "prepared"
            / f"{run_label}.json"
        )
        expected = {
            "schema_version": HISTORICAL_COMPACTION_SCHEMA_VERSION,
            "run_label": run_label,
            "state": "PREPARED",
            "compaction_plan_sha256": plan.plan_sha256,
            "historical_gate_sha256": _sha256_file(historical_gate_path),
            "migration_ledger_sha256": _sha256_file(migration_ledger_path),
            "content_inventory_sha256": _sha256_file(inventory_path),
            "review_manifest_sha256": _sha256_file(review_path),
            "legacy_registry_sha256": gate_record["legacy_registry_sha256"],
            "source_tree_sha256": plan.source_tree_sha256,
            "retention_class": plan.retention_class.value,
            "source_root": str(plan.source_root),
            "keep_file_count": len(plan.keep),
            "keep_bytes": sum(item.byte_count for item in plan.keep),
            "delete_file_count": len(plan.delete),
            "delete_bytes": sum(item.byte_count for item in plan.delete),
        }
        with _exclusive_lock(
            self.state_root / "historical-compaction" / "transaction.lock"
        ):
            if plan_path.is_file():
                if CompactionPlan.from_dict(_load_signed_json(plan_path)) != plan:
                    raise LifecycleError("historical compaction plan identity differs")
            else:
                _write_signed_json(plan_path, plan.to_dict())
            if prepared_path.is_file():
                prepared = _load_signed_json(prepared_path)
                if any(prepared.get(key) != value for key, value in expected.items()):
                    raise LifecycleError(
                        "historical compaction prepared transaction differs"
                    )
            else:
                _write_signed_json(
                    prepared_path,
                    {**expected, "created_at_utc": datetime.now(UTC).isoformat()},
                )
        return plan, plan_path, prepared_path

    def apply_historical_compaction(
        self,
        run_label: str,
        *,
        repository: Path,
        migration_ledger_path: Path,
        historical_gate_path: Path,
        plan_path: Path,
        expected_plan_sha256: str,
        writer_is_active: Callable[[Path], bool],
    ) -> dict[str, object]:
        """Import, compact, and close one pre-lifecycle run as one transaction."""

        expected_plan_path = (
            self.state_root
            / "historical-compaction"
            / "plans"
            / run_label
            / f"{expected_plan_sha256}.json"
        ).resolve()
        if plan_path.resolve(strict=True) != expected_plan_path:
            raise LifecycleError("historical compaction plan path is not canonical")
        prepared_path = (
            self.state_root
            / "historical-compaction"
            / "prepared"
            / f"{run_label}.json"
        )
        prepared = _load_signed_json(prepared_path)
        if (
            prepared.get("schema_version")
            != HISTORICAL_COMPACTION_SCHEMA_VERSION
            or prepared.get("run_label") != run_label
            or prepared.get("state") not in {"PREPARED", "COMMITTED"}
            or prepared.get("compaction_plan_sha256") != expected_plan_sha256
        ):
            raise LifecycleError("historical compaction transaction is not prepared")
        recorded_plan = CompactionPlan.from_dict(_load_signed_json(plan_path))
        current_plan, gate_record, inventory_path, review_path = (
            _historical_compaction_inputs(
                repository=repository,
                migration_ledger_path=migration_ledger_path,
                historical_gate_path=historical_gate_path,
                archive_root=self.archive_root,
                run_label=run_label,
            )
        )
        if (
            recorded_plan != current_plan
            or recorded_plan.plan_sha256 != expected_plan_sha256
            or prepared.get("historical_gate_sha256")
            != _sha256_file(historical_gate_path)
            or prepared.get("migration_ledger_sha256")
            != _sha256_file(migration_ledger_path)
            or prepared.get("content_inventory_sha256")
            != _sha256_file(inventory_path)
            or prepared.get("review_manifest_sha256") != _sha256_file(review_path)
        ):
            raise LifecycleError("historical compaction authorization differs")
        completion_path = (
            self.state_root
            / "historical-compaction"
            / "receipts"
            / f"{run_label}.json"
        )
        unfinished = tuple(
            record
            for record in self.records()
            if record.run_label != run_label and not self._close_receipt_valid(record)
        )
        if unfinished:
            raise LifecycleError(
                "another lifecycle record is unfinished during historical compaction: "
                + ", ".join(item.run_label for item in unfinished)
            )
        if self._record_path(run_label).is_file():
            existing = self._load(run_label)
            if existing.state == LifecycleState.CLOSED:
                if not self._close_receipt_valid(existing):
                    raise LifecycleError("historical compaction close receipt is invalid")
                completion = _load_signed_json(completion_path)
                if (
                    completion.get("compaction_plan_sha256")
                    != expected_plan_sha256
                    or completion.get("status") != "COMMITTED"
                ):
                    raise LifecycleError(
                        "historical compaction completion identity differs"
                    )
                if prepared.get("state") != "COMMITTED":
                    repaired = dict(prepared)
                    repaired.update(
                        state="COMMITTED",
                        completion_receipt_sha256=_sha256_file(completion_path),
                        committed_at_utc=completion.get("committed_at_utc"),
                    )
                    _write_signed_json(prepared_path, repaired)
                return completion
            if prepared.get("state") == "COMMITTED":
                raise LifecycleError(
                    "historical compaction transaction is committed without CLOSED state"
                )
            if (
                existing.plan_sha256 != expected_plan_sha256
                or existing.compaction_plan_sha256 != expected_plan_sha256
                or existing.content_inventory_sha256
                != _sha256_file(inventory_path)
                or existing.retention_class != recorded_plan.retention_class
                or existing.state
                not in {LifecycleState.CLASSIFIED, LifecycleState.COMPACTED}
            ):
                raise LifecycleError("historical lifecycle import differs")
        else:
            review = _load_signed_json(review_path)
            failure = review.get("failure_identity")
            failure_mapping = failure if isinstance(failure, dict) else {}
            created_at = prepared.get("created_at_utc")
            if not isinstance(created_at, str) or not created_at:
                raise LifecycleError(
                    "historical compaction prepared timestamp is invalid"
                )
            now = created_at
            spec = self.catalog.for_run_label(run_label)
            imported = RunLifecycleRecord(
                run_label=run_label,
                experiment_id=spec.experiment_id,
                state=LifecycleState.CLASSIFIED,
                plan_sha256=expected_plan_sha256,
                catalog_sha256=self.catalog.catalog_sha256,
                transition_ordinal=0,
                created_at_utc=now,
                updated_at_utc=now,
                reviewer_status=ReviewerStatus(str(gate_record["review_status"])),
                review_manifest_sha256=_sha256_file(review_path),
                failure_code=str(failure_mapping.get("failure_code", "")),
                failure_component=str(failure_mapping.get("component", "")),
                failure_check=str(failure_mapping.get("invariant_or_check", "")),
                failure_location=str(failure_mapping.get("location", "")),
                retention_class=recorded_plan.retention_class,
                content_inventory_sha256=_sha256_file(inventory_path),
                compaction_plan_sha256=expected_plan_sha256,
            )
            import_path = (
                self.state_root
                / "historical-compaction"
                / "imports"
                / f"{run_label}.json"
            )
            _write_signed_json(
                import_path,
                {
                    "schema_version": HISTORICAL_COMPACTION_SCHEMA_VERSION,
                    "run_label": run_label,
                    "state": "CLASSIFIED",
                    "historical_gate_sha256": _sha256_file(historical_gate_path),
                    "content_inventory_sha256": _sha256_file(inventory_path),
                    "review_manifest_sha256": _sha256_file(review_path),
                    "compaction_plan_sha256": expected_plan_sha256,
                    "record": imported.to_dict(),
                },
            )
            self._write(imported)
        current = self._load(run_label)
        if current.state == LifecycleState.CLASSIFIED:
            receipt = self.apply_compaction(
                recorded_plan,
                expected_plan_sha256=expected_plan_sha256,
                writer_is_active=writer_is_active,
            )
        else:
            receipt_payload = _load_signed_json(
                self.state_root / "compaction" / "receipts" / f"{run_label}.json"
            )
            receipt = CompactionReceipt(
                run_label=run_label,
                released_bytes=int(str(receipt_payload["released_bytes"])),
                deleted_file_count=int(str(receipt_payload["deleted_file_count"])),
                final_tree_sha256=str(receipt_payload["final_tree_sha256"]),
                delete_traversals=int(str(receipt_payload["delete_traversals"])),
                compaction_plan_sha256=expected_plan_sha256,
                receipt_sha256=_sha256_file(
                    self.state_root
                    / "compaction"
                    / "receipts"
                    / f"{run_label}.json"
                ),
            )
        completion = {
            "schema_version": HISTORICAL_COMPACTION_SCHEMA_VERSION,
            "run_label": run_label,
            "status": "COMMITTED",
            "historical_gate_sha256": _sha256_file(historical_gate_path),
            "content_inventory_sha256": _sha256_file(inventory_path),
            "review_manifest_sha256": _sha256_file(review_path),
            "compaction_plan_sha256": expected_plan_sha256,
            "compaction_receipt_sha256": receipt.receipt_sha256,
            "released_bytes": receipt.released_bytes,
            "deleted_file_count": receipt.deleted_file_count,
            "final_tree_sha256": receipt.final_tree_sha256,
            "committed_at_utc": datetime.now(UTC).isoformat(),
        }
        completion_sha256 = _write_signed_json(completion_path, completion)
        record = self._load(run_label)
        if record.state != LifecycleState.COMPACTED:
            raise LifecycleError("historical compaction did not reach COMPACTED")
        closed_payload = record.to_dict()
        closed_payload.update(
            state=LifecycleState.CLOSED.value,
            transition_ordinal=record.transition_ordinal + 1,
            updated_at_utc=datetime.now(UTC).isoformat(),
        )
        closed = RunLifecycleRecord.from_dict(closed_payload)
        _write_signed_json(
            self.state_root / "close" / f"{run_label}.json",
            {
                "schema_version": LIFECYCLE_SCHEMA_VERSION,
                "run_label": run_label,
                "status": "CLOSED",
                "record": closed.to_dict(),
                "historical_compaction_receipt_sha256": completion_sha256,
            },
        )
        self._write(closed)
        committed_prepared = dict(prepared)
        committed_prepared.update(
            state="COMMITTED",
            completion_receipt_sha256=completion_sha256,
            committed_at_utc=datetime.now(UTC).isoformat(),
        )
        _write_signed_json(prepared_path, committed_prepared)
        return completion

    def _record_path(self, run_label: str) -> Path:
        return self.state_root / "runs" / f"{run_label}.json"

    def _writer_is_active(self, run_label: str, run_dir: Path) -> bool:
        if _writer_is_active(run_dir):
            return True
        marker_root = self.state_root / "writer-markers" / run_label
        if not marker_root.is_dir():
            return False
        for marker_path in marker_root.glob("*.json"):
            try:
                marker = _load_signed_json(marker_path)
                pid = int(str(marker["pid"]))
                os.kill(pid, 0)
            except (KeyError, LifecycleError, OSError, ValueError):
                continue
            if marker.get("run_label") == run_label and marker.get("status") == "active":
                return True
        return False

    def _load(self, run_label: str) -> RunLifecycleRecord:
        record_path = self._record_path(run_label)
        _repair_orphan_lifecycle_sidecar(record_path)
        return RunLifecycleRecord.from_dict(_load_signed_json(record_path))

    def _write(self, record: RunLifecycleRecord) -> str:
        """Append one transition event, then CAS the current record.

        The per-run lock is shared by every lifecycle mutation.  Writing the
        immutable event first makes an interrupted commit recoverable: a retry
        can finish the current-record replacement, while an existing different
        event or ordinal is a hard conflict.
        """

        record_path = self._record_path(record.run_label)
        event_path = (
            self.state_root
            / "events"
            / record.run_label
            / f"{record.transition_ordinal:04d}-{record.state.value}.json"
        )
        record_payload = record.to_dict()
        record_bytes = _canonical_json(record_payload) + b"\n"
        digest = _sha256_bytes(record_bytes)
        event = dict(record_payload)
        event["record_sha256"] = digest
        lock_path = self.state_root / "run-locks" / f"{record.run_label}.lock"
        with _exclusive_lock(lock_path):
            current: RunLifecycleRecord | None = None
            if record_path.is_file():
                _repair_orphan_lifecycle_sidecar(record_path)
                current = RunLifecycleRecord.from_dict(_load_signed_json(record_path))
            if current == record:
                if not event_path.is_file() or _load_signed_json(event_path) != event:
                    raise LifecycleError("current lifecycle record has no matching event")
                return digest
            expected_ordinal = record.transition_ordinal - 1
            if record.transition_ordinal == 0:
                if current is not None:
                    raise LifecycleError("initial lifecycle record already exists")
            elif current is None or current.transition_ordinal != expected_ordinal:
                raise LifecycleError("concurrent lifecycle transition conflict")
            if event_path.exists():
                _repair_orphan_lifecycle_sidecar(event_path)
                existing_event = _load_signed_json(event_path)
                if existing_event != event:
                    existing_payload = dict(existing_event)
                    existing_digest = str(existing_payload.pop("record_sha256", ""))
                    existing_record = RunLifecycleRecord.from_dict(existing_payload)
                    if (
                        existing_record.run_label != record.run_label
                        or existing_record.transition_ordinal
                        != record.transition_ordinal
                        or existing_record.state != record.state
                        or existing_digest
                        != _sha256_bytes(_canonical_json(existing_payload) + b"\n")
                    ):
                        raise LifecycleError(
                            "lifecycle transition event already differs"
                        )
                    return _write_signed_json(record_path, existing_payload)
            else:
                _write_signed_json(event_path, event)
            written = _write_signed_json(record_path, record_payload)
            if written != digest:
                raise LifecycleError("lifecycle record digest changed during commit")
            return written

    def records(self) -> tuple[RunLifecycleRecord, ...]:
        directory = self.state_root / "runs"
        if not directory.exists():
            return ()
        return tuple(
            sorted(
                (
                    RunLifecycleRecord.from_dict(_load_signed_json(path))
                    for path in directory.glob("*.json")
                ),
                key=lambda item: (item.created_at_utc, item.run_label),
            )
        )

    def status_payload(self, record: RunLifecycleRecord) -> dict[str, object]:
        """Project immutable records through committed supersession receipts."""

        payload = record.to_dict()
        payload["effective_retention_class"] = (
            record.retention_class.value if record.retention_class is not None else None
        )
        payload["superseded_by"] = ""
        receipt_path = (
            self.state_root
            / "close"
            / "supersessions"
            / f"{record.run_label}.json"
        )
        if not receipt_path.is_file():
            return payload
        receipt = _load_signed_json(receipt_path)
        successor_label = str(receipt.get("superseded_by", ""))
        transaction_path = (
            self.state_root
            / "close"
            / "supersession-transactions"
            / f"{successor_label}.json"
        )
        transaction = _load_signed_json(transaction_path)
        successor = self._load(successor_label)
        compaction_receipt_path = (
            self.state_root
            / "compaction"
            / "receipts"
            / f"{record.run_label}.json"
        )
        if (
            record.state != LifecycleState.CLOSED
            or not self._close_receipt_valid(record)
            or receipt.get("status") != "CLOSED"
            or receipt.get("run_label") != record.run_label
            or receipt.get("predecessor_record_sha256")
            != _sha256_file(self._record_path(record.run_label))
            or receipt.get("effective_retention_class")
            != RetentionClassV3.SUPERSEDED_ACCEPTED_CAPSULE.value
            or not successor_label
            or transaction.get("state") != "COMMITTED"
            or transaction.get("predecessor_run_label") != record.run_label
            or transaction.get("successor_run_label") != successor_label
            or transaction.get("supersession_receipt_sha256")
            != _sha256_file(receipt_path)
            or successor.state != LifecycleState.CLOSED
            or not self._close_receipt_valid(successor)
            or not compaction_receipt_path.is_file()
            or receipt.get("compaction_receipt_sha256")
            != _sha256_file(compaction_receipt_path)
        ):
            raise LifecycleError("supersession projection evidence differs")
        payload["effective_retention_class"] = (
            RetentionClassV3.SUPERSEDED_ACCEPTED_CAPSULE.value
        )
        payload["superseded_by"] = successor_label
        return payload

    def _close_receipt_valid(self, record: RunLifecycleRecord) -> bool:
        if record.state != LifecycleState.CLOSED:
            return False
        path = self.state_root / "close" / f"{record.run_label}.json"
        try:
            receipt = _load_signed_json(path)
        except LifecycleError:
            return False
        return (
            receipt.get("status") == "CLOSED"
            and receipt.get("run_label") == record.run_label
            and receipt.get("record") == record.to_dict()
        )

    def plan(self, plan: ExperimentPlan) -> RunLifecycleRecord:
        spec = self.catalog.for_run_label(plan.run_label)
        if plan.planned_archive_bytes > spec.max_archive_bytes:
            raise LifecycleError("planned archive bytes exceed the catalog limit")
        if plan.max_batch_bytes > spec.max_batch_bytes:
            raise LifecycleError("planned batch bytes exceed the catalog limit")
        if plan.max_run_bytes > spec.max_run_bytes:
            raise LifecycleError("planned run bytes exceed the catalog limit")
        if plan.max_workspace_bytes > spec.max_workspace_bytes:
            raise LifecycleError("planned workspace exceeds the catalog limit")
        if (
            plan.workers > spec.max_workers
            or plan.threads > spec.max_threads
            or plan.processes > spec.max_processes
        ):
            raise LifecycleError("planned concurrency exceeds the catalog limit")
        if plan.fallback_allowed:
            raise LifecycleError("runtime fallback is forbidden by the lifecycle")
        lock_path = self.state_root / "lifecycle.lock"
        with _exclusive_lock(lock_path):
            existing_path = self._record_path(plan.run_label)
            if existing_path.exists():
                existing = self._load(plan.run_label)
                if existing.plan_sha256 != plan.plan_sha256:
                    raise LifecycleError("existing lifecycle plan identity differs")
                return existing
            unfinished = tuple(
                record
                for record in self.records()
                if not self._close_receipt_valid(record)
            )
            if unfinished:
                labels = ", ".join(item.run_label for item in unfinished)
                raise LifecycleError(
                    f"previous top-level experiment is not CLOSED: {labels}"
                )
            unsuperseded_current: dict[str, list[str]] = {}
            for record in self.records():
                supersession_path = (
                    self.state_root
                    / "close"
                    / "supersessions"
                    / f"{record.run_label}.json"
                )
                if (
                    record.state == LifecycleState.CLOSED
                    and record.retention_class
                    == RetentionClassV3.CURRENT_ACCEPTED_FULL
                    and not supersession_path.is_file()
                ):
                    unsuperseded_current.setdefault(record.experiment_id, []).append(
                        record.run_label
                    )
            conflicts = {
                experiment: labels
                for experiment, labels in unsuperseded_current.items()
                if len(labels) > 1
            }
            if conflicts:
                raise LifecycleError(
                    "previous current-accepted close transaction is unfinished: "
                    f"{conflicts}"
                )
            declared = set(spec.prerequisite_contracts)
            supplied = set(plan.prerequisite_sha256_by_contract)
            if supplied != declared:
                raise LifecycleError(
                    "lifecycle prerequisite contracts differ: "
                    f"missing={sorted(declared - supplied)}, "
                    f"extra={sorted(supplied - declared)}"
                )
            if plan.run_dir.exists():
                raise LifecycleError("run directory exists before lifecycle permission")
            now = datetime.now(UTC).isoformat()
            record = RunLifecycleRecord(
                run_label=plan.run_label,
                experiment_id=spec.experiment_id,
                state=LifecycleState.PLANNED,
                plan_sha256=plan.plan_sha256,
                catalog_sha256=self.catalog.catalog_sha256,
                transition_ordinal=0,
                created_at_utc=now,
                updated_at_utc=now,
            )
            self._write(record)
            _write_signed_json(
                self.state_root / "plans" / f"{plan.run_label}.json",
                {
                    "schema_version": LIFECYCLE_SCHEMA_VERSION,
                    "experiment_id": spec.experiment_id,
                    "catalog_sha256": self.catalog.catalog_sha256,
                    "plan": plan.to_dict(),
                    "plan_sha256": plan.plan_sha256,
                },
            )
            return self._load(plan.run_label)

    def _transition(
        self,
        record: RunLifecycleRecord,
        state: LifecycleState,
        **changes: object,
    ) -> RunLifecycleRecord:
        if state not in _TRANSITIONS[record.state]:
            raise LifecycleError(
                f"illegal lifecycle transition: {record.state.value} -> {state.value}"
            )
        payload = record.to_dict()
        payload.update(changes)
        payload.update(
            state=state.value,
            transition_ordinal=record.transition_ordinal + 1,
            updated_at_utc=datetime.now(UTC).isoformat(),
        )
        if isinstance(payload.get("reviewer_status"), ReviewerStatus):
            payload["reviewer_status"] = cast(
                ReviewerStatus, payload["reviewer_status"]
            ).value
        if isinstance(payload.get("retention_class"), RetentionClassV3):
            payload["retention_class"] = cast(
                RetentionClassV3, payload["retention_class"]
            ).value
        updated = RunLifecycleRecord.from_dict(payload)
        self._write(updated)
        return self._load(record.run_label)

    def permit(self, run_label: str, *, storage_permit_path: Path) -> RunLifecycleRecord:
        record = self._load(run_label)
        permit = _load_signed_json(storage_permit_path)
        if (
            permit.get("run_label") != run_label
            or permit.get("status") != "reserved"
            or permit.get("stage_plan_sha256") != record.plan_sha256
        ):
            raise LifecycleError("storage permit does not bind the lifecycle plan")
        permit_sha256 = _sha256_file(storage_permit_path)
        if record.state != LifecycleState.PLANNED:
            if record.storage_permit_sha256 != permit_sha256:
                raise LifecycleError("resumed lifecycle storage permit differs")
            return record
        return self._transition(
            record,
            LifecycleState.PERMITTED,
            storage_permit_sha256=permit_sha256,
        )

    def start(
        self,
        run_label: str,
        *,
        runtime_plan: ExperimentPlan,
    ) -> RunLifecycleRecord:
        record = self._load(run_label)
        if runtime_plan.run_label != run_label:
            raise LifecycleError("runtime plan run_label differs from lifecycle record")
        if runtime_plan.plan_sha256 != record.plan_sha256:
            raise LifecycleError(
                "runtime resources or immutable inputs drifted after planning"
            )
        if runtime_plan.repository_root is not None:
            if (
                runtime_plan.configuration_path is None
                or runtime_plan.lock_path is None
            ):
                raise LifecycleError("runtime plan path identity is incomplete")

            def git(*arguments: str) -> str:
                completed = subprocess.run(
                    (
                        "git",
                        "-C",
                        str(runtime_plan.repository_root),
                        *arguments,
                    ),
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    raise LifecycleError("cannot re-read runtime source identity")
                return completed.stdout.strip()

            if git("status", "--porcelain", "--untracked-files=all"):
                raise LifecycleError("runtime source became dirty after planning")
            revision = git("rev-parse", "HEAD")
            tree = git("rev-parse", "HEAD^{tree}")
            live_source = _sha256_bytes(f"{revision}\n{tree}\n".encode("ascii"))
            live_environment = _sha256_bytes(
                _canonical_json(
                    {
                        "python": sys.version,
                        "executable": sys.executable,
                        "platform": platform.platform(),
                        "implementation": platform.python_implementation(),
                    }
                )
            )
            if (
                live_source != runtime_plan.source_sha256
                or _sha256_file(runtime_plan.configuration_path)
                != runtime_plan.configuration_sha256
                or _sha256_file(runtime_plan.lock_path) != runtime_plan.lock_sha256
                or live_environment != runtime_plan.environment_sha256
            ):
                raise LifecycleError("runtime immutable identity changed after planning")
        plan_payload = _load_signed_json(
            self.state_root / "plans" / f"{run_label}.json"
        )
        if plan_payload.get("plan_sha256") != record.plan_sha256:
            raise LifecycleError("persisted lifecycle plan no longer binds the record")
        if record.state == LifecycleState.RUNNING:
            return record
        return self._transition(record, LifecycleState.RUNNING)

    def seal(
        self,
        run_label: str,
        *,
        manifest_path: Path,
        failure_code: str = "",
    ) -> RunLifecycleRecord:
        record = self._load(run_label)
        spec = next(
            item
            for item in self.catalog.specs
            if item.experiment_id == record.experiment_id
        )
        if failure_code and failure_code not in spec.allowed_failure_codes:
            raise LifecycleError("runner used an uncontrolled failure code")
        plan_payload = _load_signed_json(
            self.state_root / "plans" / f"{run_label}.json"
        )
        raw_plan = plan_payload.get("plan")
        if not isinstance(raw_plan, dict):
            raise LifecycleError("sealed lifecycle plan is invalid")
        run_dir = ExperimentPlan.from_dict(raw_plan).run_dir
        try:
            manifest_path.resolve(strict=True).relative_to(run_dir)
        except (OSError, ValueError) as error:
            raise LifecycleError("sealed manifest is outside the governed run") from error
        if self._writer_is_active(run_label, run_dir):
            raise LifecycleError("cannot seal while a governed writer is active")
        manifest = _load_external_signed_json(manifest_path)
        if manifest.get("run_label") != run_label:
            raise LifecycleError("sealed manifest run identity differs")
        if manifest.get("status") not in {"complete", "partial", "failed"} or (
            manifest.get("evidence_completeness") not in {"complete", "partial"}
        ):
            raise LifecycleError("sealed manifest terminal evidence is incomplete")
        if manifest.get("status") != "complete" and not failure_code:
            raise LifecycleError("failed or partial seal requires a controlled failure code")
        manifest_sha256 = _sha256_file(manifest_path)
        return self._transition(
            record,
            LifecycleState.SEALED,
            sealed_manifest_sha256=manifest_sha256,
            sealed_manifest_relative_path=manifest_path.resolve().relative_to(
                run_dir
            ).as_posix(),
            failure_code=failure_code,
        )

    def review(
        self,
        run_label: str,
        *,
        review_manifest_path: Path,
    ) -> RunLifecycleRecord:
        record = self._load(run_label)
        spec = next(
            item
            for item in self.catalog.specs
            if item.experiment_id == record.experiment_id
        )
        review_payload = _load_external_signed_json(review_manifest_path)
        if review_payload.get("run_label") != run_label:
            raise LifecycleError("review manifest run identity differs")
        plan_payload = _load_signed_json(
            self.state_root / "plans" / f"{run_label}.json"
        )
        raw_plan = plan_payload.get("plan")
        if not isinstance(raw_plan, dict):
            raise LifecycleError("reviewed lifecycle plan is invalid")
        persisted_plan = ExperimentPlan.from_dict(raw_plan)
        review_root = persisted_plan.run_dir / "review"
        try:
            review_manifest_path.resolve(strict=True).relative_to(review_root)
        except (OSError, ValueError) as error:
            raise LifecycleError("review manifest is outside the governed review root") from error
        execution_path = review_root / "review_execution.json"
        execution = _load_signed_json(execution_path)
        execution_binding_path = (
            self.state_root / "review-executions" / f"{run_label}.json"
        )
        execution_binding = _load_signed_json(execution_binding_path)
        if (
            execution.get("run_label") != run_label
            or execution.get("status") != "completed"
            or execution.get("finalized") is not True
            or execution.get("exit_code") != 0
            or execution.get("reviewer_module_name") != spec.reviewer_module
            or execution.get("raw_manifest_sha256_before")
            != record.sealed_manifest_sha256
            or execution.get("raw_manifest_sha256_after")
            != record.sealed_manifest_sha256
            or execution.get("raw_manifest_unchanged") is not True
            or execution.get("review_manifest_sha256")
            != _sha256_file(review_manifest_path)
            or execution_binding.get("run_label") != run_label
            or execution_binding.get("review_execution_sha256")
            != _sha256_file(execution_path)
            or execution_binding.get("review_manifest_sha256")
            != _sha256_file(review_manifest_path)
            or _SHA256.fullmatch(
                str(execution.get("reviewer_installed_distribution_digest", ""))
            )
            is None
        ):
            raise LifecycleError("independent review execution identity differs")
        try:
            reviewer_status = ReviewerStatus(
                str(review_payload.get("lifecycle_status", review_payload["status"]))
            )
        except (KeyError, ValueError) as error:
            raise LifecycleError("review manifest has no controlled reviewer status") from error
        if reviewer_status == ReviewerStatus.ACCEPTED:
            published_status = str(review_payload.get("status", ""))
            is_readiness_status = published_status.startswith("READY_FOR_STAGE")
            if (
                published_status != ReviewerStatus.ACCEPTED.value
                and not is_readiness_status
            ):
                raise LifecycleError("accepted lifecycle review has no readiness status")
        failure_identity = review_payload.get("failure_identity", {})
        if not isinstance(failure_identity, dict):
            raise LifecycleError("review failure identity is invalid")
        failure_component = str(failure_identity.get("component", ""))
        failure_check = str(failure_identity.get("invariant_or_check", ""))
        failure_location = str(failure_identity.get("location", ""))
        if reviewer_status in {
            ReviewerStatus.FAILED_KNOWN,
            ReviewerStatus.FAILED_UNKNOWN,
            ReviewerStatus.INVALID,
        } and not all((failure_component, failure_check, failure_location)):
            raise LifecycleError("failed review requires structured failure identity")
        if reviewer_status != ReviewerStatus.ACCEPTED and not record.failure_code:
            raise LifecycleError("non-accepted review requires a controlled failure code")
        raw_manifest_path = persisted_plan.run_dir.joinpath(
            *PurePosixPath(record.sealed_manifest_relative_path).parts
        )
        raw_manifest = _load_external_signed_json(raw_manifest_path)
        prepared_inventory = _inventory_from_manifests(
            run_label=run_label,
            run_dir=persisted_plan.run_dir,
            raw_manifest_path=raw_manifest_path,
            raw_manifest=raw_manifest,
            review_manifest_path=review_manifest_path,
            review_manifest=review_payload,
            io_backend=persisted_plan.io_backend,
            io_workers=persisted_plan.io_workers,
        )
        inventory_sha256 = _write_signed_json(
            self.state_root
            / "retention"
            / "prepared-inventories"
            / f"{run_label}.json",
            prepared_inventory,
        )
        return self._transition(
            record,
            LifecycleState.REVIEWED,
            reviewer_status=reviewer_status,
            review_manifest_sha256=_sha256_file(review_manifest_path),
            failure_component=failure_component,
            failure_check=failure_check,
            failure_location=failure_location,
            content_inventory_sha256=inventory_sha256,
        )

    def execute_reviewer(
        self,
        run_label: str,
        *,
        raw_manifest_path: Path,
        review_manifest_path: Path,
        command: Sequence[str],
    ) -> Path:
        """Run any catalog reviewer and produce the uniform execution receipt."""

        record = self._load(run_label)
        if record.state != LifecycleState.SEALED:
            raise LifecycleError("independent reviewer requires a SEALED run")
        spec = next(
            item
            for item in self.catalog.specs
            if item.experiment_id == record.experiment_id
        )
        command_parts = tuple(command)
        module_spec = importlib.util.find_spec(spec.reviewer_module)
        if module_spec is None or module_spec.origin is None:
            raise LifecycleError("catalog reviewer module identity cannot be resolved")
        module_path = Path(module_spec.origin).resolve(strict=True)
        module_digest_before = _sha256_file(module_path)
        if (
            len(command_parts) < 3
            or Path(command_parts[0]).resolve() != Path(sys.executable).resolve()
            or command_parts[1:3] != ("-m", spec.reviewer_module)
        ):
            raise LifecycleError("review command does not invoke the catalog reviewer")
        plan_payload = _load_signed_json(
            self.state_root / "plans" / f"{run_label}.json"
        )
        raw_plan = plan_payload.get("plan")
        if not isinstance(raw_plan, dict):
            raise LifecycleError("review execution plan is invalid")
        plan = ExperimentPlan.from_dict(raw_plan)
        expected_raw = plan.run_dir.joinpath(
            *PurePosixPath(record.sealed_manifest_relative_path).parts
        ).resolve(strict=True)
        if raw_manifest_path.resolve(strict=True) != expected_raw:
            raise LifecycleError("review execution raw manifest differs from SEALED state")
        review_root = plan.run_dir / "review"
        try:
            review_manifest_path.resolve().relative_to(review_root.resolve())
        except ValueError as error:
            raise LifecycleError("review output is outside the governed review root") from error
        receipt_path = review_root / "review_execution.json"
        if (
            review_manifest_path.exists()
            or any(
                candidate.exists()
                for candidate in (
                    review_manifest_path.with_suffix(
                        review_manifest_path.suffix + ".sha256"
                    ),
                    review_manifest_path.with_suffix(".sha256"),
                    receipt_path,
                    receipt_path.with_suffix(receipt_path.suffix + ".sha256"),
                )
            )
        ):
            raise LifecycleError("independent review output already exists")
        before = _sha256_file(expected_raw)
        if before != record.sealed_manifest_sha256:
            raise LifecycleError("SEALED raw manifest changed before independent review")
        completed = subprocess.run(command_parts, check=False)
        after = _sha256_file(expected_raw)
        if completed.returncode != 0:
            raise LifecycleError(
                f"independent reviewer exited with code {completed.returncode}"
            )
        if after != before:
            raise LifecycleError("independent reviewer modified the raw manifest")
        if _sha256_file(module_path) != module_digest_before:
            raise LifecycleError("catalog reviewer module changed during execution")
        raw_payload = _load_external_signed_json(expected_raw)
        raw_snapshot = _verify_sealed_artifact_content(
            run_dir=plan.run_dir,
            raw_manifest=raw_payload,
        )
        try:
            review_payload = json.loads(review_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise LifecycleError("independent reviewer did not produce valid JSON") from error
        if not isinstance(review_payload, dict):
            raise LifecycleError("independent review manifest is not an object")
        external_sidecars = tuple(
            candidate
            for candidate in (
                review_manifest_path.with_suffix(
                    review_manifest_path.suffix + ".sha256"
                ),
                review_manifest_path.with_suffix(".sha256"),
            )
            if candidate.is_file()
        )
        if external_sidecars:
            _load_external_signed_json(review_manifest_path)
        else:
            _write_signed_json(review_manifest_path, review_payload)
        _write_signed_json(
            receipt_path,
            {
                "schema_version": "experiment-review-execution-v1",
                "run_label": run_label,
                "status": "completed",
                "finalized": True,
                "exit_code": 0,
                "reviewer_module_name": spec.reviewer_module,
                "reviewer_installed_distribution_digest": module_digest_before,
                "raw_manifest_sha256_before": before,
                "raw_manifest_sha256_after": after,
                "raw_manifest_unchanged": True,
                "review_manifest_sha256": _sha256_file(review_manifest_path),
                "command": list(command_parts),
            },
        )
        _write_signed_json(
            self.state_root / "review-executions" / f"{run_label}.json",
            {
                "schema_version": "experiment-review-execution-binding-v1",
                "run_label": run_label,
                "review_execution_sha256": _sha256_file(receipt_path),
                "review_manifest_sha256": _sha256_file(review_manifest_path),
                "sealed_manifest_sha256": before,
            },
        )
        _verify_sealed_artifact_metadata(
            run_dir=plan.run_dir,
            snapshot=raw_snapshot,
        )
        return receipt_path

    def adjudicate(
        self,
        run_label: str,
        *,
        root_cause_id: str,
        canonical_representative: str,
        adjudication_path: Path,
    ) -> RunLifecycleRecord:
        record = self._load(run_label)
        if record.state not in {
            LifecycleState.REVIEWED,
            LifecycleState.BLOCKED_RETENTION,
        }:
            raise LifecycleError(
                "root-cause adjudication requires a reviewed lifecycle record"
            )
        if not record.review_manifest_sha256:
            raise LifecycleError("root-cause adjudication has no bound review manifest")
        adjudication = _load_signed_json(adjudication_path)
        required = {
            "run_label": run_label,
            "review_manifest_sha256": record.review_manifest_sha256,
            "root_cause_id": root_cause_id,
            "canonical_representative_run_label": canonical_representative,
            "failure_code": record.failure_code,
            "failing_component": record.failure_component,
            "invariant_or_check": record.failure_check,
            "failure_location": record.failure_location,
        }
        if any(adjudication.get(key) != value for key, value in required.items()):
            raise LifecycleError("adjudication does not match the controlled failure identity")
        if canonical_representative == run_label:
            representative = record
        else:
            representative = self._load(canonical_representative)
            if (
                representative.state != LifecycleState.CLOSED
                or representative.retention_class
                not in {
                    RetentionClassV3.PUBLISHED_FULL,
                    RetentionClassV3.CURRENT_ACCEPTED_FULL,
                    RetentionClassV3.UNIQUE_FAILURE_CAPSULE,
                }
            ):
                raise LifecycleError("canonical representative is not closed and retained")
            failure_identity = (
                record.failure_code,
                record.failure_component,
                record.failure_check,
                record.failure_location,
            )
            representative_identity = (
                representative.failure_code,
                representative.failure_component,
                representative.failure_check,
                representative.failure_location,
            )
            if failure_identity != representative_identity:
                raise LifecycleError("canonical representative failure identity differs")
        representative_sha256 = _sha256_file(
            self._record_path(canonical_representative)
        )
        if (
            adjudication.get("canonical_representative_record_sha256")
            != representative_sha256
        ):
            raise LifecycleError("adjudication representative record SHA-256 differs")
        updated = replace(
            record,
            root_cause_id=root_cause_id,
            canonical_representative=canonical_representative,
            reviewer_status=(
                ReviewerStatus.FAILED_KNOWN
                if record.reviewer_status == ReviewerStatus.FAILED_UNKNOWN
                else record.reviewer_status
            ),
            transition_ordinal=record.transition_ordinal + 1,
            updated_at_utc=datetime.now(UTC).isoformat(),
        )
        self._write(updated)
        return self._load(run_label)

    def classify(
        self,
        run_label: str,
        *,
        classification_context_path: Path,
    ) -> RetentionDecision:
        record = self._load(run_label)
        if record.state not in {LifecycleState.REVIEWED, LifecycleState.BLOCKED_RETENTION}:
            raise LifecycleError("run must be REVIEWED before classification")
        context = _load_signed_json(classification_context_path)
        if (
            context.get("schema_version") != "experiment-classification-context-v1"
            or context.get("run_label") != run_label
            or context.get("review_manifest_sha256") != record.review_manifest_sha256
        ):
            raise LifecycleError("classification context does not bind the reviewed run")
        publication_state = str(context.get("publication_state", "none"))
        if publication_state not in {"published", "current", "superseded", "none"}:
            raise LifecycleError("classification publication state is invalid")
        evidence_sha256 = str(context.get("classification_evidence_sha256", ""))
        if publication_state != "none":
            relative_evidence = PurePosixPath(
                str(context.get("classification_evidence_relative_path", ""))
            )
            if (
                _SHA256.fullmatch(evidence_sha256) is None
                or relative_evidence.is_absolute()
                or ".." in relative_evidence.parts
            ):
                raise LifecycleError("classification publication evidence is invalid")
            evidence_path = classification_context_path.parent.joinpath(
                *relative_evidence.parts
            )
            evidence = _load_signed_json(evidence_path)
            if (
                _sha256_file(evidence_path) != evidence_sha256
                or evidence.get("run_label") != run_label
                or evidence.get("review_manifest_sha256")
                != record.review_manifest_sha256
                or evidence.get("publication_state") != publication_state
            ):
                raise LifecycleError("classification publication evidence differs")
        rebuild_proof_sha256 = str(context.get("rebuild_proof_sha256", ""))
        rebuildable_verified = False
        if rebuild_proof_sha256:
            relative_proof = PurePosixPath(
                str(context.get("rebuild_proof_relative_path", ""))
            )
            if relative_proof.is_absolute() or ".." in relative_proof.parts:
                raise LifecycleError("classification rebuild proof path is unsafe")
            proof_path = classification_context_path.parent.joinpath(
                *relative_proof.parts
            )
            proof = _load_signed_json(proof_path)
            rebuildable_verified = (
                _SHA256.fullmatch(rebuild_proof_sha256) is not None
                and _sha256_file(proof_path) == rebuild_proof_sha256
                and proof.get("run_label") == run_label
                and proof.get("status") == "verified"
                and proof.get("rebuildable") is True
            )
            if not rebuildable_verified:
                raise LifecycleError("classification rebuild proof differs")
        status = record.reviewer_status
        if publication_state == "published" and status == ReviewerStatus.ACCEPTED:
            selected = RetentionClassV3.PUBLISHED_FULL
            reason = "published_milestone"
        elif publication_state == "current" and status == ReviewerStatus.ACCEPTED:
            selected = RetentionClassV3.CURRENT_ACCEPTED_FULL
            reason = "current_accepted"
        elif publication_state == "superseded" and status == ReviewerStatus.ACCEPTED:
            selected = RetentionClassV3.SUPERSEDED_ACCEPTED_CAPSULE
            reason = "superseded_accepted"
        elif (
            status == ReviewerStatus.FAILED_KNOWN
            and record.root_cause_id
            and record.canonical_representative == run_label
        ):
            selected = RetentionClassV3.UNIQUE_FAILURE_CAPSULE
            reason = "known_unique_root_cause"
        elif (
            status == ReviewerStatus.FAILED_KNOWN
            and record.root_cause_id
            and record.canonical_representative
        ):
            selected = RetentionClassV3.DUPLICATE_FAILURE_METADATA
            reason = "known_duplicate_root_cause"
        elif rebuildable_verified:
            selected = RetentionClassV3.REBUILDABLE
            reason = "signed_rebuild_proof"
        elif publication_state == "superseded" and status in {
            ReviewerStatus.PARTIAL,
            ReviewerStatus.INVALID,
        }:
            selected = RetentionClassV3.SUPERSEDED_METADATA
            reason = "superseded_without_dependency"
        else:
            selected = RetentionClassV3.UNKNOWN_FULL
            reason = "root_cause_or_reference_unknown"
        blocked = selected == RetentionClassV3.UNKNOWN_FULL
        decision = RetentionDecision(
            run_label=run_label,
            retention_class=selected,
            reason_code=reason,
            blocked=blocked,
            canonical_representative=record.canonical_representative,
        )
        if blocked:
            if record.state != LifecycleState.BLOCKED_RETENTION:
                self._transition(
                    record,
                    LifecycleState.BLOCKED_RETENTION,
                    retention_class=selected,
                    blocked_reason=reason,
                )
        else:
            self._transition(
                record,
                LifecycleState.CLASSIFIED,
                retention_class=selected,
                blocked_reason="",
            )
        return decision

    def prepare_compaction(
        self,
        run_label: str,
        *,
        run_dir: Path,
        content_inventory_path: Path,
        supersession_transaction_path: Path | None = None,
    ) -> tuple[CompactionPlan, Path]:
        record = self._load(run_label)
        retention_class = record.retention_class
        supersession: dict[str, object] | None = None
        if supersession_transaction_path is not None:
            supersession = _load_signed_json(supersession_transaction_path)
            if (
                record.state != LifecycleState.CLOSED
                or supersession.get("state") != "PREPARED"
                or supersession.get("predecessor_run_label") != run_label
                or supersession.get("predecessor_record_sha256")
                != _sha256_file(self._record_path(run_label))
            ):
                raise LifecycleError("supersession transaction does not bind predecessor")
            retention_class = RetentionClassV3.SUPERSEDED_ACCEPTED_CAPSULE
        elif record.state != LifecycleState.CLASSIFIED or retention_class is None:
            raise LifecycleError("run must be CLASSIFIED before compaction planning")
        if retention_class is None:
            raise LifecycleError("compaction has no retention class")
        resolved_run_dir = run_dir.resolve()
        if (
            supersession is None
            and resolved_run_dir.name != run_label
        ) or (
            supersession is not None
            and (
                run_label not in resolved_run_dir.parts
                or re.fullmatch(r"generation-[0-9]{4}", resolved_run_dir.name) is None
            )
        ):
            raise LifecycleError("compaction run directory identity differs")
        inventory, identities = _validated_content_inventory(
            content_inventory_path,
            run_label=run_label,
        )
        if (
            record.content_inventory_sha256
            and _sha256_file(content_inventory_path)
            != record.content_inventory_sha256
        ):
            raise LifecycleError("content inventory differs from independent review")
        representative_prefix = (
            _failure_shard_prefix(record.failure_location)
            if retention_class == RetentionClassV3.UNIQUE_FAILURE_CAPSULE
            else None
        )
        keep = tuple(
            item
            for item in identities
            if _capsule_keep(
                item.relative_path,
                retention_class,
                representative_shard_prefix=representative_prefix,
            )
        )
        delete = tuple(item for item in identities if item not in keep)
        total_bytes = sum(item.byte_count for item in identities)
        plan = CompactionPlan(
            run_label=run_label,
            source_root=resolved_run_dir,
            source_tree_sha256=_tree_sha256(identities),
            retention_class=retention_class,
            keep=keep,
            delete=delete,
            scanned_bytes=total_bytes,
            hashed_bytes=total_bytes,
            scan_passes=1,
            hash_passes=1,
            io_backend=str(inventory.get("io_backend", "")),
            io_workers=int(str(inventory.get("io_workers", 0))),
        )
        path = (
            self.state_root
            / "compaction"
            / "plans"
            / run_label
            / f"{plan.plan_sha256}.json"
        )
        _write_signed_json(path, plan.to_dict())
        if supersession is not None and supersession_transaction_path is not None:
            supersession["compaction_plan_sha256"] = plan.plan_sha256
            _write_signed_json(supersession_transaction_path, supersession)
            return plan, path
        updated = replace(
            record,
            compaction_plan_sha256=plan.plan_sha256,
            transition_ordinal=record.transition_ordinal + 1,
            updated_at_utc=datetime.now(UTC).isoformat(),
        )
        self._write(updated)
        return plan, path

    def mark_retained(
        self,
        run_label: str,
        *,
        retention_receipt_path: Path,
        content_inventory_path: Path,
    ) -> RunLifecycleRecord:
        """Bind a signed physical-retention receipt before close."""

        record = self._load(run_label)
        expected_receipt_path = (
            self.storage_state_root
            / "lifecycle_retention_receipts"
            / f"{run_label}.json"
        ).resolve()
        if retention_receipt_path.resolve(strict=True) != expected_receipt_path:
            raise LifecycleError("retention receipt is outside storage governance")
        receipt = _load_signed_json(retention_receipt_path)
        inventory, identities = _validated_content_inventory(
            content_inventory_path,
            run_label=run_label,
        )
        if (
            record.content_inventory_sha256
            and _sha256_file(content_inventory_path)
            != record.content_inventory_sha256
        ):
            raise LifecycleError("content inventory differs from independent review")
        if (
            receipt.get("schema_version")
            != "experiment-lifecycle-retention-binding-v1"
            or receipt.get("run_label") != run_label
            or receipt.get("storage_permit_sha256") != record.storage_permit_sha256
        ):
            raise LifecycleError("retention receipt run identity differs")
        if receipt.get("verification_status") not in {"verified", "committed"}:
            raise LifecycleError("retention receipt is not verified")
        if receipt.get("retention_class") != (
            record.retention_class.value if record.retention_class is not None else None
        ):
            raise LifecycleError("retention receipt class differs")
        hashes = (
            str(receipt.get("archive_tree_sha256", "")),
            str(receipt.get("storage_tree_sha256", "")),
            str(receipt.get("verifier_identity_sha256", "")),
        )
        counts = (receipt.get("file_count"), receipt.get("byte_count"))
        replay_fields = (
            receipt.get("validator_replay_passed"),
            receipt.get("objective_replay_passed"),
            receipt.get("raw_review_replay_passed"),
        )
        if (
            any(_SHA256.fullmatch(value) is None for value in hashes)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts
            )
            or cast(int, counts[0]) == 0
            or replay_fields != (True, True, True)
        ):
            raise LifecycleError("retention receipt replay evidence is incomplete")
        if (
            receipt.get("archive_tree_sha256") != inventory["source_tree_sha256"]
            or receipt.get("file_count") != len(identities)
            or receipt.get("byte_count")
            != sum(item.byte_count for item in identities)
        ):
            raise LifecycleError("retention receipt differs from content inventory")
        generation = receipt.get("generation")
        relative_archive = PurePosixPath(
            str(receipt.get("archive_relative_path", ""))
        )
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation <= 0
            or relative_archive.is_absolute()
            or ".." in relative_archive.parts
            or run_label not in relative_archive.parts
            or f"generation-{generation:04d}" not in relative_archive.parts
        ):
            raise LifecycleError("retention archive generation identity is invalid")
        archive_path = self.archive_root.resolve().joinpath(*relative_archive.parts)
        if (
            not archive_path.is_dir()
            or not archive_path.resolve().is_relative_to(self.archive_root.resolve())
        ):
            raise LifecycleError("retention archive path is not canonical")
        archive_files, archive_bytes = _verify_tree_layout(
            archive_path,
            identities,
        )
        if (
            archive_files != receipt.get("file_count")
            or archive_bytes != receipt.get("byte_count")
        ):
            raise LifecycleError("retention archive no longer matches its receipt")
        registry_path = self.storage_state_root / "retention_registry_v2.json"
        if (
            not registry_path.is_file()
            or receipt.get("registry_sha256") != _sha256_file(registry_path)
        ):
            raise LifecycleError("retention registry identity differs")
        registry = _load_signed_json(registry_path)
        raw_records = registry.get("records")
        if not isinstance(raw_records, list):
            raise LifecycleError("retention registry records are invalid")
        matching_records = [
            item
            for item in raw_records
            if isinstance(item, dict)
            and item.get("run_label") == run_label
            and item.get("generation") == generation
        ]
        if not matching_records or any(
            item.get("archive_relative_path") != relative_archive.as_posix()
            or item.get("tree_sha256") != receipt.get("storage_tree_sha256")
            or item.get("file_count") != receipt.get("file_count")
            or item.get("byte_count") != receipt.get("byte_count")
            or item.get("verification_status") != "verified"
            for item in matching_records
        ):
            raise LifecycleError("retention registry does not bind the archive")
        replay_relative = PurePosixPath(
            str(receipt.get("replay_receipt_relative_path", ""))
        )
        if replay_relative.is_absolute() or ".." in replay_relative.parts:
            raise LifecycleError("retention replay path is unsafe")
        replay_path = self.storage_state_root.joinpath(*replay_relative.parts)
        replay = _load_signed_json(replay_path)
        if (
            receipt.get("replay_receipt_sha256") != _sha256_file(replay_path)
            or replay.get("run_label") != run_label
            or replay.get("generation") != generation
            or replay.get("archive_tree_sha256")
            != receipt.get("storage_tree_sha256")
            or replay.get("archive_file_count") != receipt.get("file_count")
            or replay.get("archive_byte_count") != receipt.get("byte_count")
            or replay.get("validator_replay_passed") is not True
            or replay.get("objective_replay_passed") is not True
            or replay.get("raw_review_replay_passed") is not True
            or replay.get("status") != "passed"
        ):
            raise LifecycleError("retention replay receipt differs")
        performance = receipt.get("close_performance")
        if not isinstance(performance, dict):
            raise LifecycleError("retention receipt close performance is missing")
        ClosePerformance.from_dict(performance).verify()
        inventory_path = (
            self.state_root / "retention" / "inventories" / f"{run_label}.json"
        )
        inventory_sha256 = _write_signed_json(inventory_path, inventory)
        stored_path = (
            self.state_root / "retention" / "receipts" / f"{run_label}.json"
        )
        stored_receipt = dict(receipt)
        stored_receipt["content_inventory_sha256"] = inventory_sha256
        stored_receipt["archive_path"] = str(archive_path.resolve())
        stored_sha256 = _write_signed_json(stored_path, stored_receipt)
        return self._transition(
            record,
            LifecycleState.RETAINED,
            retention_receipt_sha256=stored_sha256,
            content_inventory_sha256=inventory_sha256,
        )

    def apply_compaction(
        self,
        plan: CompactionPlan,
        *,
        expected_plan_sha256: str,
        writer_is_active: Callable[[Path], bool],
        supersession_transaction_path: Path | None = None,
    ) -> CompactionReceipt:
        if plan.plan_sha256 != expected_plan_sha256:
            raise LifecycleError("compaction plan SHA-256 differs")
        if writer_is_active(plan.source_root):
            raise LifecycleError("compaction target still has an active writer")
        record = self._load(plan.run_label)
        supersession: dict[str, object] | None = None
        if supersession_transaction_path is not None:
            supersession = _load_signed_json(supersession_transaction_path)
            if (
                record.state != LifecycleState.CLOSED
                or supersession.get("state") != "PREPARED"
                or supersession.get("predecessor_run_label") != plan.run_label
                or supersession.get("predecessor_record_sha256")
                != _sha256_file(self._record_path(plan.run_label))
                or supersession.get("compaction_plan_sha256") != plan.plan_sha256
                or plan.retention_class
                != RetentionClassV3.SUPERSEDED_ACCEPTED_CAPSULE
            ):
                raise LifecycleError("supersession transaction does not bind compaction")
        else:
            if record.compaction_plan_sha256 != plan.plan_sha256:
                raise LifecycleError("controller record does not bind the compaction plan")
            if record.state not in {
                LifecycleState.CLASSIFIED,
                LifecycleState.RETAINED,
                LifecycleState.COMPACTED,
            }:
                raise LifecycleError("run is not ready for compaction apply")
        lock = self.state_root / "compaction" / "compaction.lock"
        receipt_path = (
            self.state_root / "compaction" / "receipts" / f"{plan.run_label}.json"
        )
        applying_path = (
            self.state_root / "compaction" / "applying" / f"{plan.run_label}.json"
        )

        def verify_identity(item: FileIdentity) -> Path:
            path = plan.source_root.joinpath(*PurePosixPath(item.relative_path).parts)
            try:
                stat = path.stat()
            except OSError as error:
                raise LifecycleError(
                    f"compaction source drift: {item.relative_path}"
                ) from error
            if (
                stat.st_size != item.byte_count
                or stat.st_mtime_ns != item.modified_time_ns
                or _sha256_file(path) != item.sha256
            ):
                raise LifecycleError(f"compaction source drift: {item.relative_path}")
            return path

        def append_ledgers(receipt_sha256: str, final_tree: str) -> None:
            ledger_path = self.state_root / "compaction" / "ledger.csv"
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            existing_summary: set[tuple[str, str]] = set()
            if ledger_path.is_file():
                with ledger_path.open(encoding="utf-8", newline="") as handle:
                    existing_summary = {
                        (row["run_label"], row["plan_sha256"])
                        for row in csv.DictReader(handle)
                    }
            with ledger_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                if handle.tell() == 0:
                    writer.writerow(
                        (
                            "run_label",
                            "plan_sha256",
                            "receipt_sha256",
                            "released_bytes",
                            "final_tree_sha256",
                        )
                    )
                if (plan.run_label, plan.plan_sha256) not in existing_summary:
                    writer.writerow(
                        (
                            plan.run_label,
                            plan.plan_sha256,
                            receipt_sha256,
                            sum(item.byte_count for item in plan.delete),
                            final_tree,
                        )
                    )
                handle.flush()
                os.fsync(handle.fileno())
            file_ledger = self.state_root / "compaction" / "deleted_files.csv"
            existing_files: set[tuple[str, str, str]] = set()
            if file_ledger.is_file():
                with file_ledger.open(encoding="utf-8", newline="") as handle:
                    existing_files = {
                        (row["run_label"], row["plan_sha256"], row["relative_path"])
                        for row in csv.DictReader(handle)
                    }
            with file_ledger.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                if handle.tell() == 0:
                    writer.writerow(
                        (
                            "run_label",
                            "plan_sha256",
                            "relative_path",
                            "byte_count",
                            "sha256",
                            "modified_time_ns",
                        )
                    )
                for item in plan.delete:
                    key = (plan.run_label, plan.plan_sha256, item.relative_path)
                    if key not in existing_files:
                        writer.writerow(
                            (
                                plan.run_label,
                                plan.plan_sha256,
                                item.relative_path,
                                item.byte_count,
                                item.sha256,
                                item.modified_time_ns,
                            )
                        )
                handle.flush()
                os.fsync(handle.fileno())

        writer_lock = self.state_root / "writer-locks" / f"{plan.run_label}.lock"
        with _exclusive_lock(lock), _exclusive_lock(writer_lock):
            if writer_is_active(plan.source_root) or self._writer_is_active(
                plan.run_label, plan.source_root
            ):
                raise LifecycleError("compaction target still has an active writer")
            observed_files: dict[str, Path] = {}
            observed_directories: list[Path] = []
            for root, directories, files in os.walk(plan.source_root):
                directories.sort()
                files.sort()
                root_path = Path(root)
                for directory_name in directories:
                    directory = root_path / directory_name
                    if directory.is_symlink():
                        raise LifecycleError(
                            f"compaction source contains a symlink: {directory}"
                        )
                    observed_directories.append(directory)
                for file_name in files:
                    path = root_path / file_name
                    if path.is_symlink():
                        raise LifecycleError(
                            f"compaction source contains a symlink: {path}"
                        )
                    observed_files[path.relative_to(plan.source_root).as_posix()] = path
            expected_keep = {item.relative_path for item in plan.keep}
            expected_delete = {item.relative_path for item in plan.delete}
            observed = set(observed_files)

            def verify_final_layout() -> None:
                expected_directories: set[PurePosixPath] = {PurePosixPath(".")}
                expected_children: dict[PurePosixPath, set[str]] = {
                    PurePosixPath("."): set()
                }
                for relative in sorted(expected_keep):
                    path = PurePosixPath(relative)
                    parent = path.parent
                    expected_children.setdefault(parent, set()).add(path.name)
                    while str(parent) != ".":
                        expected_directories.add(parent)
                        expected_children.setdefault(parent.parent, set()).add(parent.name)
                        parent = parent.parent
                for relative_directory in sorted(
                    expected_directories,
                    key=lambda item: (len(item.parts), item.as_posix()),
                    reverse=True,
                ):
                    directory = (
                        plan.source_root
                        if str(relative_directory) == "."
                        else plan.source_root.joinpath(*relative_directory.parts)
                    )
                    if not directory.is_dir() or directory.is_symlink():
                        raise LifecycleError("compaction final directory layout differs")
                    actual = {item.name for item in directory.iterdir()}
                    if actual != expected_children.get(relative_directory, set()):
                        raise LifecycleError("compaction final file set differs")

            if receipt_path.is_file():
                payload = _load_signed_json(receipt_path)
                if (
                    payload.get("state") != "COMMITTED"
                    or payload.get("compaction_plan_sha256") != plan.plan_sha256
                ):
                    raise LifecycleError("existing compaction receipt identity differs")
                if observed != expected_keep:
                    raise LifecycleError("committed compaction file set differs")
                for item in plan.keep:
                    verify_identity(item)
                verify_final_layout()
                receipt_sha256 = _sha256_file(receipt_path)
                final_tree = str(payload["final_tree_sha256"])
                append_ledgers(receipt_sha256, final_tree)
            else:
                if applying_path.is_file():
                    applying = _load_signed_json(applying_path)
                    if (
                        applying.get("state") != "APPLYING"
                        or applying.get("compaction_plan_sha256") != plan.plan_sha256
                    ):
                        raise LifecycleError("existing APPLYING checkpoint identity differs")
                    allow_missing_delete = True
                    if not expected_keep.issubset(observed) or not observed.issubset(
                        expected_keep | expected_delete
                    ):
                        raise LifecycleError("resumed compaction file set differs")
                else:
                    if observed != expected_keep | expected_delete:
                        raise LifecycleError("compaction source file set differs")
                    allow_missing_delete = False
                verified_paths = {
                    item.relative_path: verify_identity(item)
                    for item in (*plan.keep, *plan.delete)
                    if item.relative_path in observed
                }
                if not applying_path.is_file():
                    _write_signed_json(
                        applying_path,
                        {
                            "schema_version": COMPACTION_SCHEMA_VERSION,
                            "run_label": plan.run_label,
                            "state": "APPLYING",
                            "compaction_plan_sha256": plan.plan_sha256,
                            "started_at_utc": datetime.now(UTC).isoformat(),
                        },
                    )
                for item in plan.delete:
                    delete_path = verified_paths.get(item.relative_path)
                    if delete_path is None:
                        if allow_missing_delete:
                            continue
                        raise LifecycleError(
                            f"compaction source drift: {item.relative_path}"
                        )
                    delete_path.unlink()
                keep_directories = {
                    plan.source_root.joinpath(*parent.parts)
                    for relative in expected_keep
                    for parent in PurePosixPath(relative).parents
                    if str(parent) != "."
                }
                for directory in sorted(
                    observed_directories,
                    key=lambda item: len(item.parts),
                    reverse=True,
                ):
                    if directory in keep_directories:
                        continue
                    try:
                        directory.rmdir()
                    except OSError as error:
                        raise LifecycleError(
                            f"compaction directory contains late data: {directory}"
                        ) from error
                verify_final_layout()
                final_tree = _tree_sha256(plan.keep)
                payload = {
                    "schema_version": COMPACTION_SCHEMA_VERSION,
                    "run_label": plan.run_label,
                    "state": "COMMITTED",
                    "compaction_plan_sha256": plan.plan_sha256,
                    "released_bytes": sum(item.byte_count for item in plan.delete),
                    "deleted_file_count": len(plan.delete),
                    "final_tree_sha256": final_tree,
                    "delete_traversals": 1,
                    "committed_at_utc": datetime.now(UTC).isoformat(),
                }
                receipt_sha256 = _write_signed_json(receipt_path, payload)
                append_ledgers(receipt_sha256, final_tree)
        record = self._load(plan.run_label)
        target_state = LifecycleState.COMPACTED
        if supersession is None and record.state == LifecycleState.CLASSIFIED:
            self._transition(
                record,
                target_state,
                compaction_receipt_sha256=receipt_sha256,
            )
        return CompactionReceipt(
            run_label=plan.run_label,
            released_bytes=sum(item.byte_count for item in plan.delete),
            deleted_file_count=len(plan.delete),
            final_tree_sha256=final_tree,
            delete_traversals=1,
            compaction_plan_sha256=plan.plan_sha256,
            receipt_sha256=receipt_sha256,
        )

    def _supersede_previous_current(self, current: RunLifecycleRecord) -> None:
        """Demote the prior current accepted run inside the new close transaction."""

        transaction_path = (
            self.state_root
            / "close"
            / "supersession-transactions"
            / f"{current.run_label}.json"
        )
        predecessor: RunLifecycleRecord | None = None
        if transaction_path.is_file():
            transaction = _load_signed_json(transaction_path)
            if transaction.get("successor_run_label") != current.run_label:
                raise LifecycleError("supersession transaction successor differs")
            predecessor = self._load(str(transaction["predecessor_run_label"]))
            if (
                transaction.get("state") == "COMMITTED"
                and predecessor.state == LifecycleState.CLOSED
                and self._close_receipt_valid(predecessor)
            ):
                receipt_path = (
                    self.state_root
                    / "close"
                    / "supersessions"
                    / f"{predecessor.run_label}.json"
                )
                if (
                    not receipt_path.is_file()
                    or transaction.get("supersession_receipt_sha256")
                    != _sha256_file(receipt_path)
                ):
                    raise LifecycleError("committed supersession receipt differs")
                return
        else:
            candidates = [
                item
                for item in self.records()
                if item.run_label != current.run_label
                and item.experiment_id == current.experiment_id
                and item.state == LifecycleState.CLOSED
                and item.retention_class == RetentionClassV3.CURRENT_ACCEPTED_FULL
                and not (
                    self.state_root
                    / "close"
                    / "supersessions"
                    / f"{item.run_label}.json"
                ).is_file()
            ]
            if len(candidates) > 1:
                raise LifecycleError("multiple current accepted predecessors exist")
            if not candidates:
                return
            predecessor = candidates[0]
            if not self._close_receipt_valid(predecessor):
                raise LifecycleError("current predecessor has no valid close receipt")
            _write_signed_json(
                transaction_path,
                {
                    "schema_version": LIFECYCLE_SCHEMA_VERSION,
                    "state": "PREPARED",
                    "successor_run_label": current.run_label,
                    "predecessor_run_label": predecessor.run_label,
                    "predecessor_record_sha256": _sha256_file(
                        self._record_path(predecessor.run_label)
                    ),
                },
            )

        if predecessor is None:
            raise LifecycleError("supersession transaction has no predecessor")
        inventory_path = (
            self.state_root
            / "retention"
            / "inventories"
            / f"{predecessor.run_label}.json"
        )
        if (
            not predecessor.content_inventory_sha256
            or _sha256_file(inventory_path) != predecessor.content_inventory_sha256
        ):
            raise LifecycleError("current predecessor inventory identity differs")
        retained_receipt = _load_signed_json(
            self.state_root
            / "retention"
            / "receipts"
            / f"{predecessor.run_label}.json"
        )
        archive_path = Path(str(retained_receipt.get("archive_path", ""))).resolve()
        if (
            retained_receipt.get("content_inventory_sha256")
            != predecessor.content_inventory_sha256
            or predecessor.run_label not in archive_path.parts
            or re.fullmatch(r"generation-[0-9]{4}", archive_path.name) is None
            or not archive_path.is_dir()
        ):
            raise LifecycleError("current predecessor archive binding differs")
        transaction = _load_signed_json(transaction_path)
        plan_sha256 = str(transaction.get("compaction_plan_sha256", ""))
        plan_path = (
            self.state_root
            / "compaction"
            / "plans"
            / predecessor.run_label
            / f"{plan_sha256}.json"
        )
        if plan_sha256:
            if not plan_path.is_file():
                raise LifecycleError("supersession compaction plan is missing")
            compaction_plan = CompactionPlan.from_dict(_load_signed_json(plan_path))
            if transaction.get("compaction_plan_sha256") != compaction_plan.plan_sha256:
                raise LifecycleError("supersession compaction plan binding differs")
        else:
            compaction_plan, _ = self.prepare_compaction(
                predecessor.run_label,
                run_dir=archive_path,
                content_inventory_path=inventory_path,
                supersession_transaction_path=transaction_path,
            )
        compaction = self.apply_compaction(
            compaction_plan,
            expected_plan_sha256=compaction_plan.plan_sha256,
            writer_is_active=_writer_is_active,
            supersession_transaction_path=transaction_path,
        )
        supersession_receipt = (
            self.state_root
            / "close"
            / "supersessions"
            / f"{predecessor.run_label}.json"
        )
        _write_signed_json(
            supersession_receipt,
            {
                "schema_version": LIFECYCLE_SCHEMA_VERSION,
                "status": "CLOSED",
                "run_label": predecessor.run_label,
                "superseded_by": current.run_label,
                "predecessor_record_sha256": _sha256_file(
                    self._record_path(predecessor.run_label)
                ),
                "effective_retention_class": (
                    RetentionClassV3.SUPERSEDED_ACCEPTED_CAPSULE.value
                ),
                "compaction_receipt_sha256": compaction.receipt_sha256,
            },
        )
        _write_signed_json(
            transaction_path,
            {
                "schema_version": LIFECYCLE_SCHEMA_VERSION,
                "state": "COMMITTED",
                "successor_run_label": current.run_label,
                "predecessor_run_label": predecessor.run_label,
                "supersession_receipt_sha256": _sha256_file(
                    supersession_receipt
                ),
            },
        )

    def close(
        self,
        run_label: str,
        *,
        performance: ClosePerformance,
        storage_reconciliation_path: Path,
    ) -> RunLifecycleRecord:
        """Close while holding the run's exclusive writer/generation lease."""

        with _exclusive_lock(
            self.state_root / "writer-locks" / f"{run_label}.lock"
        ):
            return self._close_exclusive(
                run_label,
                performance=performance,
                storage_reconciliation_path=storage_reconciliation_path,
            )

    def _close_exclusive(
        self,
        run_label: str,
        *,
        performance: ClosePerformance,
        storage_reconciliation_path: Path,
    ) -> RunLifecycleRecord:
        performance.verify()
        record = self._load(run_label)
        if record.state == LifecycleState.CLOSED:
            if not self._close_receipt_valid(record):
                raise LifecycleError("CLOSED record has no valid close receipt")
            if record.retention_class == RetentionClassV3.CURRENT_ACCEPTED_FULL:
                with _exclusive_lock(
                    self.state_root / "close" / "supersession.lock"
                ):
                    self._supersede_previous_current(record)
            return record
        if record.state not in {LifecycleState.RETAINED, LifecycleState.COMPACTED}:
            raise LifecycleError("run is not retained or compacted")
        if record.retention_class == RetentionClassV3.UNKNOWN_FULL:
            raise LifecycleError("unknown retention cannot be CLOSED")
        retained_archive_path: Path | None = None
        retained_archive_snapshot: dict[str, tuple[int, int]] | None = None
        if record.state == LifecycleState.COMPACTED:
            if not record.compaction_plan_sha256:
                raise LifecycleError("compacted run has no bound plan identity")
            plan_payload = _load_signed_json(
                self.state_root
                / "compaction"
                / "plans"
                / run_label
                / f"{record.compaction_plan_sha256}.json"
            )
            compaction_receipt = _load_signed_json(
                self.state_root / "compaction" / "receipts" / f"{run_label}.json"
            )
            expected_source_bytes = sum(
                int(str(item["byte_count"]))
                for field in ("keep", "delete")
                for item in cast(list[dict[str, object]], plan_payload[field])
            )
            if (
                performance.hashed_bytes != int(str(plan_payload["hashed_bytes"]))
                or performance.source_bytes != expected_source_bytes
                or performance.scan_passes != int(str(plan_payload["scan_passes"]))
                or performance.delete_traversals
                != int(str(compaction_receipt["delete_traversals"]))
                or performance.backend != plan_payload["io_backend"]
                or performance.workers != int(str(plan_payload["io_workers"]))
            ):
                raise LifecycleError("close performance differs from compaction evidence")
        else:
            retained = _load_signed_json(
                self.state_root / "retention" / "receipts" / f"{run_label}.json"
            )
            expected_performance = retained.get("close_performance")
            if not isinstance(expected_performance, dict) or performance != (
                ClosePerformance.from_dict(expected_performance)
            ):
                raise LifecycleError("close performance differs from retention evidence")
            inventory_path = (
                self.state_root / "retention" / "inventories" / f"{run_label}.json"
            )
            _inventory, identities = _validated_content_inventory(
                inventory_path,
                run_label=run_label,
            )
            archive_path = Path(str(retained.get("archive_path", ""))).resolve()
            if (
                not archive_path.is_dir()
                or not archive_path.is_relative_to(self.archive_root.resolve())
            ):
                raise LifecycleError("retained archive path is not canonical")
            (
                archive_tree,
                archive_files,
                archive_bytes,
                retained_archive_snapshot,
            ) = _verify_tree_content(archive_path, identities)
            retained_archive_path = archive_path
            if (
                archive_tree != retained.get("archive_tree_sha256")
                or archive_files != retained.get("file_count")
                or archive_bytes != retained.get("byte_count")
            ):
                raise LifecycleError("retained archive changed before close")
        disposition_sha256 = (
            record.retention_receipt_sha256
            if record.state == LifecycleState.RETAINED
            else record.compaction_receipt_sha256
        )
        expected_reconciliation_path = (
            self.capacity_state_root
            / "permit_reconciliations"
            / f"{run_label}.json"
        ).resolve()
        if storage_reconciliation_path.resolve(strict=True) != (
            expected_reconciliation_path
        ):
            raise LifecycleError("storage reconciliation is outside capacity governance")
        reconciliation = _load_signed_json(storage_reconciliation_path)
        if (
            reconciliation.get("schema_version")
            != "experiment-capacity-reconciliation-v1"
            or reconciliation.get("run_label") != run_label
            or reconciliation.get("outcome") != "retained"
            or reconciliation.get("evidence_sha256") != disposition_sha256
            or reconciliation.get("storage_permit_sha256")
            != record.storage_permit_sha256
        ):
            raise LifecycleError("storage permit reconciliation does not bind retention")
        permit_path = self.capacity_state_root / "permits" / f"{run_label}.json"
        if (
            not permit_path.is_file()
            or _sha256_file(permit_path) != record.storage_permit_sha256
        ):
            raise LifecycleError("capacity permit identity differs at close")
        permit = _load_signed_json(permit_path)
        ledger_path = self.capacity_state_root / "capacity_ledger.json"
        ledger = _load_signed_json(ledger_path)
        reservations = ledger.get("reservations")
        reservation = (
            reservations.get(run_label) if isinstance(reservations, dict) else None
        )
        if (
            permit.get("run_label") != run_label
            or permit.get("stage_plan_sha256") != record.plan_sha256
            or not isinstance(reservation, dict)
            or reservation.get("status") != "reconciled"
            or reservation.get("stage_plan_sha256") != record.plan_sha256
            or reservation.get("outcome") != "retained"
            or reservation.get("evidence_sha256") != disposition_sha256
        ):
            raise LifecycleError("capacity ledger does not reconcile the lifecycle")
        if retained_archive_path is not None and retained_archive_snapshot is not None:
            _verify_tree_metadata_snapshot(
                retained_archive_path,
                retained_archive_snapshot,
            )
        closed_payload = record.to_dict()
        closed_payload.update(
            state=LifecycleState.CLOSED.value,
            transition_ordinal=record.transition_ordinal + 1,
            updated_at_utc=datetime.now(UTC).isoformat(),
        )
        closed = RunLifecycleRecord.from_dict(closed_payload)
        _write_signed_json(
            self.state_root / "close" / f"{run_label}.json",
            {
                "schema_version": LIFECYCLE_SCHEMA_VERSION,
                "run_label": run_label,
                "status": "CLOSED",
                "record": closed.to_dict(),
                "storage_reconciliation_sha256": _sha256_file(
                    storage_reconciliation_path
                ),
                "performance": {
                    "hashed_bytes": performance.hashed_bytes,
                    "source_bytes": performance.source_bytes,
                    "scan_passes": performance.scan_passes,
                    "delete_traversals": performance.delete_traversals,
                    "duplicate_hashed_bytes": performance.duplicate_hashed_bytes,
                    "backend_calibrated": performance.backend_calibrated,
                    "implicit_fallback": performance.implicit_fallback,
                    "throughput_mib_per_second": performance.throughput_mib_per_second,
                    "cpu_utilization_percent": performance.cpu_utilization_percent,
                    "peak_memory_bytes": performance.peak_memory_bytes,
                    "io_utilization_percent": performance.io_utilization_percent,
                    "close_wall_seconds": performance.close_wall_seconds,
                    "backend": performance.backend,
                    "workers": performance.workers,
                },
            },
        )
        self._write(closed)
        committed = self._load(run_label)
        if committed.retention_class == RetentionClassV3.CURRENT_ACCEPTED_FULL:
            with _exclusive_lock(self.state_root / "close" / "supersession.lock"):
                self._supersede_previous_current(committed)
        return committed

    def audit(self) -> dict[str, object]:
        records = self.records()
        effective_records = [self.status_payload(item) for item in records]
        blocked = [
            item.run_label
            for item in records
            if item.state == LifecycleState.BLOCKED_RETENTION
        ]
        unfinished = [
            item.run_label for item in records if not self._close_receipt_valid(item)
        ]
        return {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "catalog_sha256": self.catalog.catalog_sha256,
            "record_count": len(records),
            "blocked_retention": blocked,
            "unfinished": unfinished,
            "effective_records": effective_records,
            "passed": not blocked and len(unfinished) <= 1,
        }


def audit_runner_entrypoints(repository: Path, catalog: ExperimentCatalog) -> None:
    """Static CI guard for governed producers and the complete controller surface."""

    failures: list[str] = []

    def parsed(path: Path) -> ast.Module:
        try:
            return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as error:
            raise LifecycleError(f"cannot parse governed entrypoint: {path}") from error

    def called_names(tree: ast.AST) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
        return names

    load_lifecycle_migration_ledger(
        repository / "experiments" / "registries" / "experiment_lifecycle_v3_migration.json",
        repository=repository,
    )
    catalogued_paths: set[Path] = set()
    for spec in catalog.specs:
        for module in (spec.runner_module, *spec.additional_runner_modules):
            relative = Path(*module.split(".")).with_suffix(".py")
            path = repository / "src" / relative
            catalogued_paths.add(path.resolve())
            if not path.is_file():
                failures.append(f"missing runner module: {module}")
                continue
            runner_calls = called_names(parsed(path))
            if "preflight_cli_attempt" not in runner_calls:
                failures.append(f"runner bypasses lifecycle preflight: {module}")
            if "seal_cli_attempt" not in runner_calls:
                failures.append(f"runner bypasses lifecycle seal: {module}")
            if "seal_failed_cli_attempt" not in runner_calls:
                failures.append(f"runner bypasses lifecycle failure seal: {module}")
        reviewer_path = (
            repository
            / "src"
            / Path(*spec.reviewer_module.split(".")).with_suffix(".py")
        )
        if not reviewer_path.is_file():
            failures.append(f"missing independent reviewer: {spec.reviewer_module}")
    for module in catalog.historical_semantic_reviewer_modules:
        reviewer_path = (
            repository / "src" / Path(*module.split(".")).with_suffix(".py")
        )
        if not reviewer_path.is_file():
            failures.append(f"missing historical semantic reviewer: {module}")
    governance_path = repository / "src" / "evrptw" / "storage_governance.py"
    governance_tree = parsed(governance_path)
    preflight_nodes = [
        node
        for node in governance_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "preflight_cli_attempt"
    ]
    if len(preflight_nodes) != 1:
        failures.append("shared lifecycle preflight definition is missing or ambiguous")
    else:
        missing = {"plan", "permit", "start"} - called_names(preflight_nodes[0])
        if missing:
            failures.append(
                "shared lifecycle preflight omits transitions: "
                + ", ".join(sorted(missing))
            )
    lifecycle_tree = parsed(repository / "src" / "evrptw" / "experiment_lifecycle.py")
    lifecycle_main_nodes = [
        node
        for node in lifecycle_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    if len(lifecycle_main_nodes) != 1:
        failures.append("shared lifecycle CLI main is missing or ambiguous")
    else:
        required_lifecycle_calls = {
            "plan",
            "permit",
            "start",
            "seal",
            "execute_reviewer",
            "review",
            "adjudicate",
            "classify",
            "mark_retained",
            "prepare_compaction",
            "apply_compaction",
            "close",
            "audit",
        }
        missing = required_lifecycle_calls - called_names(lifecycle_main_nodes[0])
        if missing:
            failures.append(
                "shared lifecycle CLI omits controller operations: "
                + ", ".join(sorted(missing))
            )
    frozen_paths = {
        (
            repository
            / "src"
            / Path(*module.split(".")).with_suffix(".py")
        ).resolve()
        for module in catalog.frozen_non_top_level_entrypoints
    }
    if any(not path.is_file() for path in frozen_paths):
        failures.append("catalogued frozen entrypoint is missing")
    experiments = repository / "src" / "evrptw" / "experiments"
    for path in experiments.glob("*.py"):
        tree = parsed(path)
        has_main = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "main"
            for node in tree.body
        )
        has_output_dir = any(
            isinstance(node, ast.Name) and node.id == "output_dir"
            for node in ast.walk(tree)
        )
        if (
            has_output_dir
            and has_main
            and path.resolve() not in catalogued_paths
            and path.resolve() not in frozen_paths
            and "_review" not in path.stem
            and path.stem != "__init__"
        ):
            failures.append(f"uncatalogued experiment entrypoint: {path.stem}")
    if failures:
        raise LifecycleError("; ".join(sorted(failures)))


def build_repository_plan(
    *,
    repository: Path,
    config_path: Path,
    run_label: str,
    run_dir: Path,
    planned_archive_bytes: int,
    max_workspace_bytes: int,
    max_batch_bytes: int | None = None,
    max_run_bytes: int | None = None,
    workers: int = 1,
    threads: int = 1,
    processes: int = 1,
    io_backend: str = "storage_governance_auto_native",
    io_workers: int = 1,
    batch_size: int = 1,
    queue_depth: int = 1,
    row_group_size: int = 1,
    prerequisite_sha256_by_contract: Mapping[str, str] | None = None,
) -> ExperimentPlan:
    """Build the identity shared by lifecycle and the physical storage permit."""

    resolved_repository = repository.resolve()

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(resolved_repository), *arguments),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise LifecycleError(
                f"cannot resolve lifecycle source identity: {completed.stderr.strip()}"
            )
        return completed.stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise LifecycleError("lifecycle planning requires a clean repository")
    revision = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    source_sha256 = _sha256_bytes(f"{revision}\n{tree}\n".encode("ascii"))
    lock_path = resolved_repository / "uv.lock"
    if not lock_path.is_file():
        raise LifecycleError("lifecycle planning requires the repository lock file")
    environment_sha256 = _sha256_bytes(
        _canonical_json(
            {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "implementation": platform.python_implementation(),
            }
        )
    )
    return ExperimentPlan(
        run_label=run_label,
        run_dir=run_dir.resolve(),
        configuration_sha256=_sha256_file(config_path.resolve(strict=True)),
        source_sha256=source_sha256,
        lock_sha256=_sha256_file(lock_path),
        environment_sha256=environment_sha256,
        prerequisite_sha256_by_contract=(
            prerequisite_sha256_by_contract or {}
        ),
        planned_archive_bytes=planned_archive_bytes,
        max_batch_bytes=max_batch_bytes or planned_archive_bytes,
        max_run_bytes=max_run_bytes or planned_archive_bytes,
        max_workspace_bytes=max_workspace_bytes,
        workers=workers,
        threads=threads,
        processes=processes,
        fallback_allowed=False,
        io_backend=io_backend,
        io_workers=io_workers,
        batch_size=batch_size,
        queue_depth=queue_depth,
        row_group_size=row_group_size,
        repository_root=resolved_repository,
        configuration_path=config_path.resolve(strict=True),
        lock_path=lock_path.resolve(strict=True),
    )


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="experiment-lifecycle")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--storage-state-root", type=Path)
    parser.add_argument("--capacity-state-root", type=Path)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--storage-root-locator", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--plan", type=Path, required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--run-label", required=True)
    start.add_argument("--storage-permit", type=Path, required=True)
    start.add_argument("--runtime-plan", type=Path, required=True)
    seal = subparsers.add_parser("seal")
    seal.add_argument("--run-label", required=True)
    seal.add_argument("--manifest", type=Path, required=True)
    seal.add_argument("--failure-code", default="")
    review = subparsers.add_parser("review")
    review.add_argument("--run-label", required=True)
    review.add_argument("--review-manifest", type=Path, required=True)
    run_review = subparsers.add_parser("run-review")
    run_review.add_argument("--run-label", required=True)
    run_review.add_argument("--raw-manifest", type=Path, required=True)
    run_review.add_argument("--review-manifest", type=Path, required=True)
    run_review.add_argument("reviewer_command", nargs=argparse.REMAINDER)
    adjudicate = subparsers.add_parser("adjudicate")
    adjudicate.add_argument("--run-label", required=True)
    adjudicate.add_argument("--adjudication", type=Path, required=True)
    adjudicate.add_argument("--root-cause-id", required=True)
    adjudicate.add_argument("--canonical-representative", required=True)
    prepare_historical = subparsers.add_parser("prepare-historical-compaction")
    prepare_historical.add_argument("--run-label", required=True)
    prepare_historical.add_argument("--repository", type=Path, required=True)
    prepare_historical.add_argument("--migration-ledger", type=Path, required=True)
    prepare_historical.add_argument("--historical-gate", type=Path, required=True)
    apply_historical = subparsers.add_parser("apply-historical-compaction")
    apply_historical.add_argument("--run-label", required=True)
    apply_historical.add_argument("--repository", type=Path, required=True)
    apply_historical.add_argument("--migration-ledger", type=Path, required=True)
    apply_historical.add_argument("--historical-gate", type=Path, required=True)
    apply_historical.add_argument("--plan", type=Path, required=True)
    apply_historical.add_argument("--expected-plan-sha256", required=True)
    close = subparsers.add_parser("close")
    close.add_argument("--run-label", required=True)
    close.add_argument("--run-dir", type=Path, required=True)
    close.add_argument("--retention-receipt", type=Path)
    close.add_argument("--performance", type=Path, required=True)
    close.add_argument("--storage-reconciliation", type=Path, required=True)
    close.add_argument("--content-inventory", type=Path, required=True)
    close.add_argument("--classification-context", type=Path, required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--run-label")
    audit = subparsers.add_parser("audit")
    audit.add_argument("--repository", type=Path)
    return parser


def _writer_is_active(run_dir: Path) -> bool:
    if (run_dir / ".writer-active").exists():
        return True
    pid_path = run_dir / "control" / "writer.pid"
    if not pid_path.is_file():
        return False
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _cli().parse_args(argv)
    root_arguments = (
        arguments.storage_state_root,
        arguments.capacity_state_root,
        arguments.archive_root,
        arguments.storage_root_locator,
    )
    if arguments.command not in {"status", "audit"} and any(
        path is None for path in root_arguments
    ):
        raise LifecycleError(
            "mutating lifecycle commands require canonical storage, capacity, "
            "and archive roots"
        )
    if arguments.command not in {"status", "audit"}:
        from evrptw.stage052_campaign import StorageRootLocator
        from evrptw.stage052_campaign_runner import probe_volume_identity

        catalog_path = arguments.catalog.resolve(strict=True)
        canonical_locator_path = (
            catalog_path.parent / "stage052_storage_roots.local.toml"
        ).resolve(strict=True)
        supplied_locator_path = cast(
            Path, arguments.storage_root_locator
        ).resolve(strict=True)
        if supplied_locator_path != canonical_locator_path:
            raise LifecycleError(
                "lifecycle CLI requires the repository canonical storage locator"
            )
        locator = StorageRootLocator.from_toml(canonical_locator_path)
        required_aliases = ("e_archive", "d_archive", "d_host", "wsl_staging")
        locator.verify_all(probe_volume_identity, required_aliases)
        expected_archive = locator.resolve("e_archive").absolute_path.resolve()
        expected_staging = locator.resolve("wsl_staging").absolute_path.resolve()
        expected_d_archive = locator.resolve("d_archive").absolute_path.resolve()
        expected_d_host = locator.resolve("d_host").absolute_path.resolve()
        if (
            expected_archive
            != Path("/mnt/e/Reproducible-EVRPTW-archive").resolve()
            or expected_d_host != Path("/mnt/d").resolve()
            or not expected_d_archive.is_relative_to(expected_d_host)
            or expected_staging.is_relative_to(Path("/mnt").resolve())
        ):
            raise LifecycleError("canonical storage locator violates role boundaries")
        supplied_roots = (
            arguments.state_root.resolve(),
            cast(Path, arguments.storage_state_root).resolve(),
            cast(Path, arguments.capacity_state_root).resolve(),
            cast(Path, arguments.archive_root).resolve(),
        )
        expected_roots = (
            expected_archive / ".experiment-lifecycle",
            expected_archive / ".storage-governance",
            expected_staging / ".storage-governance",
            expected_archive,
        )
        if supplied_roots != expected_roots:
            raise LifecycleError("lifecycle CLI roots differ from the canonical locator")
    controller = ExperimentLifecycleController(
        catalog=ExperimentCatalog.from_toml(arguments.catalog),
        state_root=arguments.state_root.resolve(),
        storage_state_root=(
            arguments.storage_state_root.resolve()
            if arguments.storage_state_root is not None
            else None
        ),
        capacity_state_root=(
            arguments.capacity_state_root.resolve()
            if arguments.capacity_state_root is not None
            else None
        ),
        archive_root=(
            arguments.archive_root.resolve()
            if arguments.archive_root is not None
            else None
        ),
    )
    if arguments.command == "plan":
        plan_payload = _load_signed_json(arguments.plan)
        raw_plan = plan_payload.get("plan", plan_payload)
        if not isinstance(raw_plan, dict):
            raise LifecycleError("signed plan payload is invalid")
        payload: object = controller.plan(ExperimentPlan.from_dict(raw_plan)).to_dict()
    elif arguments.command == "start":
        runtime_payload = _load_signed_json(arguments.runtime_plan)
        raw_runtime_plan = runtime_payload.get("plan", runtime_payload)
        if not isinstance(raw_runtime_plan, dict):
            raise LifecycleError("signed runtime plan payload is invalid")
        controller.permit(
            arguments.run_label,
            storage_permit_path=arguments.storage_permit,
        )
        payload = controller.start(
            arguments.run_label,
            runtime_plan=ExperimentPlan.from_dict(raw_runtime_plan),
        ).to_dict()
    elif arguments.command == "seal":
        payload = controller.seal(
            arguments.run_label,
            manifest_path=arguments.manifest,
            failure_code=arguments.failure_code,
        ).to_dict()
    elif arguments.command == "review":
        payload = controller.review(
            arguments.run_label,
            review_manifest_path=arguments.review_manifest,
        ).to_dict()
    elif arguments.command == "run-review":
        reviewer_command = tuple(arguments.reviewer_command)
        if reviewer_command[:1] == ("--",):
            reviewer_command = reviewer_command[1:]
        receipt = controller.execute_reviewer(
            arguments.run_label,
            raw_manifest_path=arguments.raw_manifest,
            review_manifest_path=arguments.review_manifest,
            command=reviewer_command,
        )
        payload = {
            "run_label": arguments.run_label,
            "review_execution": str(receipt),
            "review_execution_sha256": _sha256_file(receipt),
        }
    elif arguments.command == "adjudicate":
        payload = controller.adjudicate(
            arguments.run_label,
            root_cause_id=arguments.root_cause_id,
            canonical_representative=arguments.canonical_representative,
            adjudication_path=arguments.adjudication,
        ).to_dict()
    elif arguments.command == "prepare-historical-compaction":
        compaction, plan_path, prepared_path = controller.prepare_historical_compaction(
            arguments.run_label,
            repository=arguments.repository.resolve(),
            migration_ledger_path=arguments.migration_ledger.resolve(strict=True),
            historical_gate_path=arguments.historical_gate.resolve(strict=True),
        )
        payload = {
            "run_label": arguments.run_label,
            "state": "PREPARED",
            "compaction_plan_sha256": compaction.plan_sha256,
            "plan_path": str(plan_path),
            "prepared_path": str(prepared_path),
            "keep_file_count": len(compaction.keep),
            "keep_bytes": sum(item.byte_count for item in compaction.keep),
            "delete_file_count": len(compaction.delete),
            "delete_bytes": sum(item.byte_count for item in compaction.delete),
        }
    elif arguments.command == "apply-historical-compaction":
        payload = controller.apply_historical_compaction(
            arguments.run_label,
            repository=arguments.repository.resolve(),
            migration_ledger_path=arguments.migration_ledger.resolve(strict=True),
            historical_gate_path=arguments.historical_gate.resolve(strict=True),
            plan_path=arguments.plan.resolve(strict=True),
            expected_plan_sha256=arguments.expected_plan_sha256,
            writer_is_active=_writer_is_active,
        )
    elif arguments.command == "close":
        decision = controller.classify(
            arguments.run_label,
            classification_context_path=arguments.classification_context,
        )
        if decision.blocked:
            raise LifecycleError(
                "close refused because retention classification is unknown"
            )
        if decision.retention_class.preserves_full_tree:
            if arguments.retention_receipt is None:
                raise LifecycleError("full retention close requires a receipt")
            controller.mark_retained(
                arguments.run_label,
                retention_receipt_path=arguments.retention_receipt,
                content_inventory_path=arguments.content_inventory,
            )
        else:
            compaction, _path = controller.prepare_compaction(
                arguments.run_label,
                run_dir=arguments.run_dir.resolve(),
                content_inventory_path=arguments.content_inventory,
            )
            controller.apply_compaction(
                compaction,
                expected_plan_sha256=compaction.plan_sha256,
                writer_is_active=_writer_is_active,
            )
        performance_payload = _load_signed_json(arguments.performance)
        payload = controller.close(
            arguments.run_label,
            performance=ClosePerformance.from_dict(performance_payload),
            storage_reconciliation_path=arguments.storage_reconciliation,
        ).to_dict()
    elif arguments.command == "status":
        if arguments.run_label:
            payload = controller.status_payload(controller._load(arguments.run_label))
        else:
            payload = [
                controller.status_payload(record) for record in controller.records()
            ]
    else:
        if arguments.repository is not None:
            audit_runner_entrypoints(arguments.repository.resolve(), controller.catalog)
        payload = controller.audit()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
