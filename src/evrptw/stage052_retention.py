"""Stage 5.2 single-current-version retention and verified archival.

Run labels remain immutable experiment identities.  This module keeps bulky
run evidence out of the repository workspace by auditing complete directory
trees and moving them to the configured external archive only after their
inventory identity has been reverified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from evrptw.stage052_campaign import StorageRootLocator
    from evrptw.stage052_campaign_runner import probe_volume_identity

INVENTORY_SCHEMA_VERSION: Final = "stage05.2-retention-inventory-v1"
REGISTRY_SCHEMA_VERSION: Final = "stage05.2-retention-registry-v1"
POLICY_VERSION: Final = "stage05.2-single-current-v1"
SUPERSESSION_RECEIPT_SCHEMA_VERSION: Final = "stage05.2-supersession-receipt-v1"
_SUPERSESSION_RECEIPT_RELATIVE_PATH: Final = PurePosixPath(
    "control", "supersession_receipt.json"
)
_SUPERSESSION_RECEIPT_SIDECAR_RELATIVE_PATH: Final = PurePosixPath(
    "control", "supersession_receipt.sha256"
)
_RUN_LABEL = re.compile(
    r"^stage05\.2_(?P<component>[a-z0-9_]+)_(?:attempt|rerun)[0-9]{2}$"
)
_RUN_REFERENCE = re.compile(r"stage05\.2_[a-z0-9_]+_(?:attempt|rerun)[0-9]{2}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_STATUS_FIELDS: Final = frozenset(
    {"status", "review_status", "next_status", "readiness_status"}
)
_COMPLETENESS_FIELDS: Final = frozenset({"evidence_completeness"})
_SOURCE_COMMIT_FIELDS: Final = frozenset(
    {
        "source_commit",
        "source_revision",
        "repository_commit",
        "repository_revision",
        "git_commit",
    }
)
_REGISTRY_FIELDS: Final = (
    "schema_version",
    "run_label",
    "component",
    "status",
    "evidence_completeness",
    "source_commit",
    "prerequisite_run_labels",
    "original_relative_path",
    "file_count",
    "byte_count",
    "tree_sha256",
    "archive_root_alias",
    "archive_relative_path",
    "disposition",
    "verification_status",
    "archived_at_utc",
)


class RetentionIntegrityError(RuntimeError):
    """A retention operation could not prove that evidence remained intact."""


@dataclass(frozen=True, slots=True)
class Stage052RetentionPolicy:
    """One current implementation with full run evidence archived externally."""

    archive_root_alias: str
    archive_relative_base: str
    policy_version: str = POLICY_VERSION
    workspace_full_evidence: str = "active_only"
    completed_action: str = "archive"
    failed_action: str = "archive"

    def __post_init__(self) -> None:
        if self.policy_version != POLICY_VERSION:
            raise ValueError(f"unsupported Stage 5.2 retention policy: {self.policy_version}")
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.archive_root_alias) is None:
            raise ValueError("archive_root_alias must be a canonical storage alias")
        base = PurePosixPath(self.archive_relative_base)
        if base.is_absolute() or not base.parts or ".." in base.parts:
            raise ValueError("archive_relative_base must be a safe relative path")
        if self.workspace_full_evidence != "active_only":
            raise ValueError("Stage 5.2 workspace_full_evidence must be active_only")
        if self.completed_action != "archive" or self.failed_action != "archive":
            raise ValueError("complete and failed Stage 5.2 evidence must be archived")

    @classmethod
    def from_toml(cls, path: Path) -> Stage052RetentionPolicy:
        try:
            with path.open("rb") as handle:
                payload = tomllib.load(handle)
            raw = payload["retention"]
        except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
            raise ValueError(f"cannot read Stage 5.2 retention policy: {path}") from error
        if not isinstance(raw, dict):
            raise ValueError("Stage 5.2 retention policy must be a TOML table")
        required = {
            "policy_version",
            "archive_root_alias",
            "archive_relative_base",
            "workspace_full_evidence",
            "completed_action",
            "failed_action",
        }
        if set(raw) != required or any(not isinstance(raw[field], str) for field in required):
            raise ValueError("Stage 5.2 retention policy fields do not match its schema")
        return cls(**cast(dict[str, str], raw))

    def to_dict(self) -> dict[str, str]:
        return {
            "policy_version": self.policy_version,
            "archive_root_alias": self.archive_root_alias,
            "archive_relative_base": self.archive_relative_base,
            "workspace_full_evidence": self.workspace_full_evidence,
            "completed_action": self.completed_action,
            "failed_action": self.failed_action,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Stage052RetentionPolicy:
        expected = {
            "policy_version",
            "archive_root_alias",
            "archive_relative_base",
            "workspace_full_evidence",
            "completed_action",
            "failed_action",
        }
        if set(payload) != expected:
            raise ValueError("Stage 5.2 retention policy fields do not match its schema")
        if any(not isinstance(value, str) for value in payload.values()):
            raise ValueError("Stage 5.2 retention policy values must be strings")
        return cls(**cast(dict[str, str], dict(payload)))


@dataclass(frozen=True, slots=True)
class RetentionRecord:
    """Audited identity and final disposition of one Stage 5.2 run directory."""

    run_label: str
    component: str
    status: str
    evidence_completeness: str
    source_commit: str
    prerequisite_run_labels: tuple[str, ...]
    original_relative_path: str
    file_count: int
    byte_count: int
    tree_sha256: str
    archive_root_alias: str
    archive_relative_path: str
    disposition: str = "planned_archive"
    verification_status: str = "pending"
    archived_at_utc: str = ""

    def __post_init__(self) -> None:
        match = _RUN_LABEL.fullmatch(self.run_label)
        if match is None or match.group("component") != self.component:
            raise ValueError("retention record run label/component mismatch")
        if self.original_relative_path != self.run_label:
            raise ValueError("retention record source path must be the run label")
        if self.file_count < 0 or self.byte_count < 0:
            raise ValueError("retention record counts must be non-negative")
        if re.fullmatch(r"[0-9a-f]{64}", self.tree_sha256) is None:
            raise ValueError("retention record tree_sha256 is invalid")
        expected_archive = (
            PurePosixPath(self.archive_relative_path).parts[-1]
            if PurePosixPath(self.archive_relative_path).parts
            else ""
        )
        if expected_archive != self.run_label or ".." in PurePosixPath(
            self.archive_relative_path
        ).parts:
            raise ValueError("retention record archive path is unsafe")
        if tuple(sorted(set(self.prerequisite_run_labels))) != self.prerequisite_run_labels:
            raise ValueError("prerequisite run labels must be sorted and unique")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_label": self.run_label,
            "component": self.component,
            "status": self.status,
            "evidence_completeness": self.evidence_completeness,
            "source_commit": self.source_commit,
            "prerequisite_run_labels": list(self.prerequisite_run_labels),
            "original_relative_path": self.original_relative_path,
            "file_count": self.file_count,
            "byte_count": self.byte_count,
            "tree_sha256": self.tree_sha256,
            "archive_root_alias": self.archive_root_alias,
            "archive_relative_path": self.archive_relative_path,
            "disposition": self.disposition,
            "verification_status": self.verification_status,
            "archived_at_utc": self.archived_at_utc,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RetentionRecord:
        expected = {
            "run_label",
            "component",
            "status",
            "evidence_completeness",
            "source_commit",
            "prerequisite_run_labels",
            "original_relative_path",
            "file_count",
            "byte_count",
            "tree_sha256",
            "archive_root_alias",
            "archive_relative_path",
            "disposition",
            "verification_status",
            "archived_at_utc",
        }
        if set(payload) != expected:
            raise ValueError("retention record fields do not match its schema")
        prerequisites = payload["prerequisite_run_labels"]
        if not isinstance(prerequisites, list) or any(
            not isinstance(item, str) for item in prerequisites
        ):
            raise ValueError("prerequisite_run_labels must be an array of strings")
        integers = (payload["file_count"], payload["byte_count"])
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise ValueError("retention record counts must be integers")
        string_fields = expected - {
            "prerequisite_run_labels",
            "file_count",
            "byte_count",
        }
        if any(not isinstance(payload[field], str) for field in string_fields):
            raise ValueError("retention record scalar fields must be strings")
        return cls(
            **{field: cast(str, payload[field]) for field in string_fields},
            prerequisite_run_labels=tuple(cast(list[str], prerequisites)),
            file_count=cast(int, payload["file_count"]),
            byte_count=cast(int, payload["byte_count"]),
        )


@dataclass(frozen=True, slots=True)
class RetentionInventory:
    """Signed, local audit plan that binds every source directory byte."""

    source_root: str
    policy: Stage052RetentionPolicy
    records: tuple[RetentionRecord, ...]
    created_at_utc: str
    schema_version: str = INVENTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INVENTORY_SCHEMA_VERSION:
            raise ValueError("unsupported Stage 5.2 retention inventory schema")
        if not Path(self.source_root).is_absolute():
            raise ValueError("retention inventory source_root must be absolute")
        labels = tuple(record.run_label for record in self.records)
        if labels != tuple(sorted(set(labels))):
            raise ValueError("retention inventory records must be sorted and unique")
        if any(
            record.archive_root_alias != self.policy.archive_root_alias
            for record in self.records
        ):
            raise ValueError("retention record archive alias disagrees with policy")
        if any(
            record.archive_relative_path
            != PurePosixPath(
                self.policy.archive_relative_base,
                record.run_label,
            ).as_posix()
            for record in self.records
        ):
            raise ValueError("retention record archive path disagrees with policy")

    @property
    def directory_count(self) -> int:
        return len(self.records)

    @property
    def total_bytes(self) -> int:
        return sum(record.byte_count for record in self.records)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "source_root": self.source_root,
            "directory_count": self.directory_count,
            "total_bytes": self.total_bytes,
            "policy": self.policy.to_dict(),
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RetentionInventory:
        expected = {
            "schema_version",
            "created_at_utc",
            "source_root",
            "directory_count",
            "total_bytes",
            "policy",
            "records",
        }
        if set(payload) != expected:
            raise ValueError("retention inventory fields do not match its schema")
        policy = payload["policy"]
        records = payload["records"]
        if not isinstance(policy, dict) or not isinstance(records, list):
            raise ValueError("retention inventory policy/records are invalid")
        inventory = cls(
            schema_version=_string(payload, "schema_version"),
            created_at_utc=_string(payload, "created_at_utc"),
            source_root=_string(payload, "source_root"),
            policy=Stage052RetentionPolicy.from_dict(cast(dict[str, object], policy)),
            records=tuple(
                RetentionRecord.from_dict(cast(dict[str, object], item))
                for item in records
                if isinstance(item, dict)
            ),
        )
        if len(inventory.records) != len(records):
            raise ValueError("retention inventory record must be an object")
        if payload["directory_count"] != inventory.directory_count:
            raise ValueError("retention inventory directory_count mismatch")
        if payload["total_bytes"] != inventory.total_bytes:
            raise ValueError("retention inventory total_bytes mismatch")
        return inventory


def _string(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _tree_files(path: Path) -> tuple[Path, ...]:
    if not path.is_dir() or path.is_symlink():
        raise RetentionIntegrityError(f"run directory is missing or unsafe: {path}")
    files: list[Path] = []
    for item in path.rglob("*"):
        if item.is_symlink():
            raise RetentionIntegrityError(f"retention tree contains a symbolic link: {item}")
        if item.is_file():
            files.append(item)
        elif not item.is_dir():
            raise RetentionIntegrityError(f"retention tree contains an unsupported entry: {item}")
    return tuple(sorted(files, key=lambda item: item.relative_to(path).as_posix()))


def _tree_identity(path: Path) -> tuple[int, int, str]:
    return _tree_identity_excluding(path, ())


def _tree_identity_excluding(
    path: Path,
    excluded_relative_paths: Sequence[PurePosixPath],
) -> tuple[int, int, str]:
    digest = hashlib.sha256(b"stage05.2-retention-tree-v1\0")
    file_count = 0
    byte_count = 0
    excluded = {item.as_posix() for item in excluded_relative_paths}
    files = tuple(
        file_path
        for file_path in _tree_files(path)
        if file_path.relative_to(path).as_posix() not in excluded
    )
    for file_path in files:
        before = file_path.stat()
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(before.st_size.to_bytes(8, "big"))
        with file_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = file_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RetentionIntegrityError(f"file changed during retention audit: {file_path}")
        file_count += 1
        byte_count += before.st_size
    observed = tuple(
        file_path
        for file_path in _tree_files(path)
        if file_path.relative_to(path).as_posix() not in excluded
    )
    if files != observed:
        raise RetentionIntegrityError(f"retention tree changed during audit: {path}")
    return file_count, byte_count, digest.hexdigest()


def _matching_run_processes(run_label: str) -> tuple[int, ...]:
    """Return live producer/reviewer PIDs for one exact run identity."""

    matches: list[int] = []
    current_pid = os.getpid()
    markers = (
        "stage052_performance",
        "stage052_campaign",
        "stage052_review",
        "stage052_replay",
    )
    try:
        completed = subprocess.run(
            ("ps", "-eo", "pid=,args="),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RetentionIntegrityError("cannot prove superseded campaign process absence") from error
    for line in completed.stdout.splitlines():
        try:
            raw_pid, command = line.strip().split(maxsplit=1)
            pid = int(raw_pid)
        except ValueError:
            continue
        if (
            pid != current_pid
            and run_label in command
            and any(marker in command for marker in markers)
            and "stage052_retention" not in command
        ):
            matches.append(pid)
    return tuple(sorted(matches))


def _campaign_manifest_identity(run_dir: Path) -> tuple[dict[str, object], str]:
    manifest_path = run_dir / "campaign_manifest.json"
    sidecar_path = run_dir / "campaign_manifest.sha256"
    try:
        payload = manifest_path.read_bytes()
        sidecar = sidecar_path.read_text(encoding="ascii").strip()
        parsed = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetentionIntegrityError("superseded campaign manifest is unreadable") from error
    observed = hashlib.sha256(payload).hexdigest()
    if observed != sidecar:
        raise RetentionIntegrityError("superseded campaign manifest sidecar does not match")
    if not isinstance(parsed, dict):
        raise RetentionIntegrityError("superseded campaign manifest must be an object")
    return cast(dict[str, object], parsed), observed


def _supersession_batch_ids(
    campaign: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw_batches = campaign.get("batches")
    if not isinstance(raw_batches, list) or not raw_batches:
        raise RetentionIntegrityError("superseded campaign batches are missing")
    archived: list[str] = []
    planned: list[str] = []
    seen: set[str] = set()
    for raw_batch in raw_batches:
        if not isinstance(raw_batch, dict):
            raise RetentionIntegrityError("superseded campaign batch must be an object")
        batch_id = raw_batch.get("batch_id")
        status = raw_batch.get("status")
        if (
            not isinstance(batch_id, str)
            or re.fullmatch(r"batch[0-9]{4}", batch_id) is None
            or batch_id in seen
        ):
            raise RetentionIntegrityError("superseded campaign batch identity is invalid")
        seen.add(batch_id)
        if status == "archived":
            archived.append(batch_id)
        elif status == "planned":
            planned.append(batch_id)
        else:
            raise RetentionIntegrityError(
                "superseded campaign may contain only archived or planned batches"
            )
    return tuple(archived), tuple(planned)


def _supersession_host_evidence(
    host_log: Path,
    *,
    host_run_id: str,
    launch_nonce: str,
    expected_exit_code: int,
) -> tuple[str, tuple[str, ...]]:
    if re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9]+", host_run_id) is None:
        raise RetentionIntegrityError("supersession host run ID is invalid")
    if re.fullmatch(r"[0-9a-f]{32}", launch_nonce) is None:
        raise RetentionIntegrityError("supersession launch nonce is invalid")
    try:
        payload = host_log.read_bytes()
        lines = payload.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RetentionIntegrityError("supersession host log is unreadable") from error
    started = tuple(
        line
        for line in lines
        if "HOST_STARTED " in line
        and f"run_id={host_run_id} " in line
        and f"launch_nonce={launch_nonce} " in line
    )
    failed = tuple(
        line
        for line in lines
        if "HOST_CONTROLLER_FAILED " in line
        and f"run_id={host_run_id} " in line
        and f"exit_code={expected_exit_code}" in line
    )
    stopped = tuple(
        line
        for line in lines
        if "HOST_STOPPED " in line and f"exit_code={expected_exit_code}" in line
    )
    if len(started) != 1 or len(failed) != 1 or len(stopped) < 1:
        raise RetentionIntegrityError(
            "supersession host log does not contain one bound failed execution"
        )
    return hashlib.sha256(payload).hexdigest(), (started[0], failed[0], stopped[-1])


def create_supersession_receipt(
    run_dir: Path,
    *,
    host_log: Path,
    host_run_id: str,
    launch_nonce: str,
    expected_exit_code: int,
    reason: str,
    created_at_utc: str | None = None,
) -> str:
    """Seal one stopped, incomplete campaign without rewriting its signed manifest."""

    resolved = run_dir.resolve()
    if not resolved.is_dir() or _RUN_LABEL.fullmatch(resolved.name) is None:
        raise RetentionIntegrityError("supersession run directory is invalid")
    if re.fullmatch(r"[a-z0-9_]+", reason) is None:
        raise RetentionIntegrityError("supersession reason must be canonical")
    receipt_path = resolved / _SUPERSESSION_RECEIPT_RELATIVE_PATH
    sidecar_path = resolved / _SUPERSESSION_RECEIPT_SIDECAR_RELATIVE_PATH
    if receipt_path.exists() or sidecar_path.exists():
        raise RetentionIntegrityError("supersession receipt already exists")
    matching_processes = _matching_run_processes(resolved.name)
    if matching_processes:
        raise RetentionIntegrityError(
            f"superseded campaign still has active processes: {matching_processes}"
        )
    campaign, campaign_sha256 = _campaign_manifest_identity(resolved)
    if campaign.get("run_label") != resolved.name or campaign.get("status") != "planned":
        raise RetentionIntegrityError("only the exact planned campaign may be superseded")
    archived_batches, planned_batches = _supersession_batch_ids(campaign)
    if not archived_batches or not planned_batches:
        raise RetentionIntegrityError(
            "supersession requires both archived and planned campaign batches"
        )
    host_log_sha256, host_evidence = _supersession_host_evidence(
        host_log,
        host_run_id=host_run_id,
        launch_nonce=launch_nonce,
        expected_exit_code=expected_exit_code,
    )
    file_count, byte_count, tree_sha256 = _tree_identity(resolved)
    receipt = {
        "schema_version": SUPERSESSION_RECEIPT_SCHEMA_VERSION,
        "run_label": resolved.name,
        "status": "superseded",
        "evidence_completeness": "partial",
        "reason": reason,
        "campaign_manifest_sha256": campaign_sha256,
        "original_campaign_status": "planned",
        "archived_batch_ids": list(archived_batches),
        "planned_batch_ids": list(planned_batches),
        "host_log_sha256": host_log_sha256,
        "host_run_id": host_run_id,
        "launch_nonce": launch_nonce,
        "exit_code": expected_exit_code,
        "host_evidence": list(host_evidence),
        "active_process_count": 0,
        "pre_receipt_file_count": file_count,
        "pre_receipt_byte_count": byte_count,
        "pre_receipt_tree_sha256": tree_sha256,
        "created_at_utc": _utc_now() if created_at_utc is None else created_at_utc,
    }
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    receipt_sha256 = hashlib.sha256(encoded).hexdigest()
    _write_atomic(receipt_path, encoded)
    _write_atomic(sidecar_path, f"{receipt_sha256}\n".encode())
    return receipt_sha256


def _validated_supersession_receipt(run_dir: Path) -> dict[str, object] | None:
    receipt_path = run_dir / _SUPERSESSION_RECEIPT_RELATIVE_PATH
    sidecar_path = run_dir / _SUPERSESSION_RECEIPT_SIDECAR_RELATIVE_PATH
    if not receipt_path.exists() and not sidecar_path.exists():
        return None
    try:
        encoded = receipt_path.read_bytes()
        sidecar = sidecar_path.read_text(encoding="ascii").strip()
        payload = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetentionIntegrityError("supersession receipt is unreadable") from error
    if hashlib.sha256(encoded).hexdigest() != sidecar or not isinstance(payload, dict):
        raise RetentionIntegrityError("supersession receipt identity is invalid")
    receipt = cast(dict[str, object], payload)
    campaign, campaign_sha256 = _campaign_manifest_identity(run_dir)
    archived_batches, planned_batches = _supersession_batch_ids(campaign)
    required = {
        "schema_version": SUPERSESSION_RECEIPT_SCHEMA_VERSION,
        "run_label": run_dir.name,
        "status": "superseded",
        "evidence_completeness": "partial",
        "campaign_manifest_sha256": campaign_sha256,
        "original_campaign_status": "planned",
        "archived_batch_ids": list(archived_batches),
        "planned_batch_ids": list(planned_batches),
        "active_process_count": 0,
    }
    if any(receipt.get(field) != expected for field, expected in required.items()):
        raise RetentionIntegrityError("supersession receipt fields do not match the campaign")
    identity = _tree_identity_excluding(
        run_dir,
        (
            _SUPERSESSION_RECEIPT_RELATIVE_PATH,
            _SUPERSESSION_RECEIPT_SIDECAR_RELATIVE_PATH,
        ),
    )
    if identity != (
        receipt.get("pre_receipt_file_count"),
        receipt.get("pre_receipt_byte_count"),
        receipt.get("pre_receipt_tree_sha256"),
    ):
        raise RetentionIntegrityError("superseded campaign changed after receipt creation")
    matching_processes = _matching_run_processes(run_dir.name)
    if matching_processes:
        raise RetentionIntegrityError(
            f"superseded campaign still has active processes: {matching_processes}"
        )
    return receipt


def _metadata(
    run_dir: Path,
    run_label: str,
    *,
    strict: bool,
) -> tuple[str, str, str, tuple[str, ...]]:
    supersession_receipt = _validated_supersession_receipt(run_dir)
    statuses: set[str] = set()
    completeness: set[str] = set()
    commits: set[str] = set()
    references: set[str] = set()
    candidates = list(sorted((run_dir / "control").glob("**/*.json")))
    current_review = run_dir / "review" / "review_manifest.json"
    if current_review.is_file():
        candidates.append(current_review)
    for path in candidates:
        if path.name.startswith("._") or path.stat().st_size > 8 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            if strict:
                raise RetentionIntegrityError(
                    f"cannot parse retention metadata: {path}"
                ) from error
            continue
        references.update(_RUN_REFERENCE.findall(text))
        if not isinstance(payload, dict):
            continue
        for key in _STATUS_FIELDS:
            value = payload.get(key)
            if isinstance(value, str):
                statuses.add(value)
        for key in _COMPLETENESS_FIELDS:
            value = payload.get(key)
            if isinstance(value, str):
                completeness.add(value)
        for key in _SOURCE_COMMIT_FIELDS:
            value = payload.get(key)
            if isinstance(value, str) and _COMMIT.fullmatch(value):
                commits.add(value)
    references.discard(run_label)
    return (
        "superseded" if supersession_receipt is not None else _select_status(statuses),
        "partial"
        if supersession_receipt is not None
        else _select_completeness(completeness),
        "|".join(sorted(commits)),
        tuple(sorted(references)),
    )


def _select_status(statuses: set[str]) -> str:
    nonterminal = sorted(
        status
        for status in statuses
        if status.casefold() in {"active", "in_progress", "planned", "running"}
    )
    if nonterminal:
        return nonterminal[0]
    if "NOT_READY" in statuses:
        return "NOT_READY"
    failure = next(
        (status for status in sorted(statuses) if status.casefold() in {"failed", "error"}),
        None,
    )
    if failure is not None:
        return failure
    ready = sorted(status for status in statuses if status.startswith("READY_"))
    if ready:
        return ready[-1]
    if "complete" in statuses:
        return "complete"
    return "unknown" if not statuses else "|".join(sorted(statuses))


def _select_completeness(values: set[str]) -> str:
    if "partial" in values:
        return "partial"
    if "complete" in values:
        return "complete"
    return "unknown" if not values else "|".join(sorted(values))


def audit_stage052_runs(
    source_root: Path,
    policy: Stage052RetentionPolicy,
    *,
    created_at_utc: str | None = None,
    allow_unsealed: bool = False,
    expected_directory_count: int | None = None,
    expected_total_bytes: int | None = None,
    run_labels: Sequence[str] = (),
) -> RetentionInventory:
    """Read and hash every Stage 5.2 directory without changing the filesystem."""

    resolved = source_root.resolve()
    if not resolved.is_dir():
        raise RetentionIntegrityError(f"retention source root is missing: {resolved}")
    if len(set(run_labels)) != len(run_labels) or any(
        _RUN_LABEL.fullmatch(run_label) is None for run_label in run_labels
    ):
        raise RetentionIntegrityError(
            "retention run-label selection must be canonical and unique"
        )
    selected_dirs: tuple[Path, ...]
    if run_labels:
        selected_dirs = tuple(
            resolved / run_label for run_label in sorted(run_labels)
        )
        missing = tuple(path.name for path in selected_dirs if not path.is_dir())
        if missing:
            raise RetentionIntegrityError(
                "selected run is missing from the retention source root: "
                + ", ".join(missing)
            )
    else:
        selected_dirs = tuple(
            sorted(resolved.glob("stage05.2_*"), key=lambda path: path.name)
        )
    records: list[RetentionRecord] = []
    for run_dir in selected_dirs:
        if not run_dir.is_dir():
            continue
        match = _RUN_LABEL.fullmatch(run_dir.name)
        if match is None:
            raise RetentionIntegrityError(f"non-canonical Stage 5.2 run directory: {run_dir}")
        status, completeness, source_commit, prerequisites = _metadata(
            run_dir,
            run_dir.name,
            strict=not allow_unsealed,
        )
        terminal = (
            status
            in {
                "NOT_READY",
                "accepted",
                "complete",
                "error",
                "failed",
                "partial",
                "superseded",
            }
            or status.startswith("READY_")
            or status.startswith("GPU_NOT_JUSTIFIED")
        )
        if not allow_unsealed and (not terminal or completeness == "unknown"):
            raise RetentionIntegrityError(
                f"Stage 5.2 run is not sealed for archival: {run_dir.name}"
            )
        file_count, byte_count, tree_sha256 = _tree_identity(run_dir)
        archive_path = PurePosixPath(policy.archive_relative_base, run_dir.name).as_posix()
        records.append(
            RetentionRecord(
                run_label=run_dir.name,
                component=match.group("component"),
                status=status,
                evidence_completeness=completeness,
                source_commit=source_commit,
                prerequisite_run_labels=prerequisites,
                original_relative_path=run_dir.name,
                file_count=file_count,
                byte_count=byte_count,
                tree_sha256=tree_sha256,
                archive_root_alias=policy.archive_root_alias,
                archive_relative_path=archive_path,
            )
        )
    inventory = RetentionInventory(
        source_root=str(resolved),
        policy=policy,
        records=tuple(records),
        created_at_utc=_utc_now() if created_at_utc is None else created_at_utc,
    )
    if allow_unsealed and (
        expected_directory_count is None or expected_total_bytes is None
    ):
        raise RetentionIntegrityError(
            "historical unsealed audit requires expected directory count and bytes"
        )
    if (
        expected_directory_count is not None
        and inventory.directory_count != expected_directory_count
    ):
        raise RetentionIntegrityError("retention directory count differs from preflight")
    if expected_total_bytes is not None and inventory.total_bytes != expected_total_bytes:
        raise RetentionIntegrityError("retention byte count differs from preflight")
    return inventory


def _inventory_bytes(inventory: RetentionInventory) -> bytes:
    return (json.dumps(inventory.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_retention_inventory(path: Path, inventory: RetentionInventory) -> str:
    """Write a signed local inventory and return its SHA-256 identity."""

    payload = _inventory_bytes(inventory)
    sha256 = hashlib.sha256(payload).hexdigest()
    _write_atomic(path, payload)
    _write_atomic(path.with_suffix(f"{path.suffix}.sha256"), f"{sha256}\n".encode())
    return sha256


def load_retention_inventory(path: Path, *, expected_sha256: str) -> RetentionInventory:
    """Load only an inventory whose bytes match the explicitly approved hash."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RetentionIntegrityError(f"cannot read retention inventory: {path}") from error
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise RetentionIntegrityError(
            f"inventory SHA-256 mismatch: expected={expected_sha256} observed={observed}"
        )
    sidecar = path.with_suffix(f"{path.suffix}.sha256")
    try:
        sidecar_sha256 = sidecar.read_text(encoding="ascii").strip()
    except OSError as error:
        raise RetentionIntegrityError(
            f"cannot read retention inventory sidecar: {sidecar}"
        ) from error
    if sidecar_sha256 != expected_sha256:
        raise RetentionIntegrityError("inventory sidecar SHA-256 mismatch")
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RetentionIntegrityError("retention inventory is not valid JSON") from error
    if not isinstance(raw, dict):
        raise RetentionIntegrityError("retention inventory must be a JSON object")
    try:
        return RetentionInventory.from_dict(cast(dict[str, object], raw))
    except (TypeError, ValueError) as error:
        raise RetentionIntegrityError(f"invalid retention inventory: {error}") from error


def _matches_record(path: Path, record: RetentionRecord) -> bool:
    if not path.is_dir():
        return False
    return _tree_identity(path) == (
        record.file_count,
        record.byte_count,
        record.tree_sha256,
    )


def _campaign_storage_error(run_label: str, detail: str) -> RetentionIntegrityError:
    return RetentionIntegrityError(
        f"campaign external storage identity mismatch for {run_label}: {detail}"
    )


def _load_campaign_review(run_dir: Path) -> dict[str, object] | None:
    review_path = run_dir / "review" / "review_manifest.json"
    if not review_path.is_file():
        return None
    try:
        payload = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _campaign_storage_error(run_dir.name, "review manifest is unreadable") from error
    if not isinstance(payload, dict):
        raise _campaign_storage_error(run_dir.name, "review manifest must be an object")
    return cast(dict[str, object], payload)


def path_uses_symlink(path: Path) -> bool:
    """Return whether any lexical component of ``path`` is a symbolic link."""

    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _verify_campaign_external_storage(
    run_dir: Path,
    *,
    locator: StorageRootLocator | None,
) -> None:
    """Reverify archived batch trees without changing their signed paths."""

    review = _load_campaign_review(run_dir)
    campaign_path = run_dir / "campaign_manifest.json"
    ready_status_by_scope = {
        "pilot": "READY_FOR_STAGE052_FORMAL_BENCHMARK",
        "formal": "READY_FOR_STAGE05_3",
    }
    ready_statuses = frozenset(ready_status_by_scope.values())
    if not campaign_path.is_file():
        if review is not None and review.get("status") in ready_statuses:
            raise _campaign_storage_error(
                run_dir.name,
                "accepted review has no signed campaign manifest",
            )
        return
    run_label = run_dir.name
    try:
        from evrptw.stage052_campaign import (
            directory_byte_count,
            directory_checksum,
            directory_file_count,
            load_batch_manifest,
            load_campaign_manifest,
        )

        campaign_bytes = campaign_path.read_bytes()
        campaign_sha256 = hashlib.sha256(campaign_bytes).hexdigest()
        campaign_sidecar_sha256 = campaign_path.with_suffix(".sha256").read_text(
            encoding="ascii"
        ).strip()
        if campaign_sha256 != campaign_sidecar_sha256:
            raise ValueError("campaign manifest sidecar does not match")
        campaign = load_campaign_manifest(campaign_path)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise _campaign_storage_error(run_label, "campaign manifest is invalid") from error
    if campaign.run_label != run_label:
        raise _campaign_storage_error(run_label, "campaign run identity is invalid")
    if review is not None:
        review_status = review.get("status")
        if review_status in ready_statuses:
            pass
        elif isinstance(review_status, str) and review_status.startswith("READY_"):
            raise _campaign_storage_error(
                run_label,
                "campaign review readiness status is invalid",
            )
        else:
            review = None
    archived_batches = tuple(
        batch for batch in campaign.batches if batch.status == "archived"
    )
    if not archived_batches:
        if review is not None:
            raise _campaign_storage_error(
                run_label,
                "accepted review has no archived batches",
            )
        return
    if locator is None:
        raise _campaign_storage_error(run_label, "storage-root locator is required")

    rows: list[Mapping[str, object] | None]
    batches = archived_batches
    if review is None:
        rows = [None] * len(batches)
    else:
        scope = review.get("scope")
        scope_str = scope if isinstance(scope, str) else ""
        expected_status = ready_status_by_scope.get(
            scope_str
        )
        if (
            review.get("schema_version") != "stage05.2-campaign-review-v2"
            or review.get("run_label") != run_label
            or review.get("component") != "benchmark"
            or expected_status is None
            or review.get("status") != expected_status
            or review.get("review_execution_required") is not True
            or campaign.scope != scope
            or campaign.status != "complete"
            or len(archived_batches) != len(campaign.batches)
        ):
            raise _campaign_storage_error(run_label, "accepted review identity is invalid")
        try:
            from evrptw.stage052_evidence import (
                verify_stage052_campaign_gate_set,
                verify_stage052_review_execution_receipt,
            )

            verify_stage052_campaign_gate_set(review, scope=scope_str)
            verify_stage052_review_execution_receipt(
                run_dir,
                run_dir / "review" / "review_manifest.json",
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise _campaign_storage_error(
                run_label,
                "accepted review gates or execution receipt are invalid",
            ) from error

        storage_identity = review.get("storage_publication_identity")
        if not isinstance(storage_identity, Mapping):
            raise _campaign_storage_error(
                run_label,
                "storage publication identity is missing",
            )
        canonical_storage = json.dumps(
            storage_identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if (
            storage_identity.get("schema_version")
            != "stage05.2-storage-publication-identity-v1"
            or storage_identity.get("run_label") != run_label
            or review.get("storage_publication_identity_sha256")
            != hashlib.sha256(canonical_storage).hexdigest()
        ):
            raise _campaign_storage_error(
                run_label,
                "storage publication digest is invalid",
            )
        raw_rows = storage_identity.get("batches")
        if not isinstance(raw_rows, list) or not raw_rows:
            raise _campaign_storage_error(
                run_label,
                "storage publication batches are missing",
            )
        expected_row_fields = {
            "root_alias",
            "relative_path",
            "file_count",
            "byte_count",
            "tree_sha256",
        }
        if any(
            not isinstance(row, Mapping) or set(row) != expected_row_fields
            for row in raw_rows
        ):
            raise _campaign_storage_error(
                run_label,
                "storage publication row schema is invalid",
            )
        ordered_rows = sorted(
            raw_rows,
            key=lambda row: (
                str(cast(Mapping[str, object], row).get("relative_path", "")),
                str(cast(Mapping[str, object], row).get("root_alias", "")),
            ),
        )
        if raw_rows != ordered_rows or len(raw_rows) != len(batches):
            raise _campaign_storage_error(
                run_label,
                "storage publication rows are not canonical",
            )
        rows = [cast(Mapping[str, object], row) for row in raw_rows]

    for row, embedded_batch in zip(rows, batches, strict=True):
        root_alias = embedded_batch.root_alias
        relative_path = embedded_batch.logical_path
        if row is not None:
            row_root_alias = row.get("root_alias")
            row_relative_path = row.get("relative_path")
            if (
                not isinstance(row_root_alias, str)
                or not isinstance(row_relative_path, str)
                or isinstance(row.get("file_count"), bool)
                or not isinstance(row.get("file_count"), int)
                or cast(int, row["file_count"]) <= 0
                or isinstance(row.get("byte_count"), bool)
                or not isinstance(row.get("byte_count"), int)
                or cast(int, row["byte_count"]) <= 0
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(row.get("tree_sha256", "")),
                )
                is None
            ):
                raise _campaign_storage_error(
                    run_label,
                    f"{embedded_batch.batch_id} publication row values are invalid",
                )
            root_alias = row_root_alias
            relative_path = row_relative_path
        relative = PurePosixPath(relative_path)
        expected_relative = PurePosixPath(run_label, embedded_batch.batch_id)
        if (
            relative != expected_relative
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in relative_path
            or root_alias != embedded_batch.root_alias
            or embedded_batch.root_alias != embedded_batch.archive_root_alias
            or embedded_batch.status != "archived"
            or embedded_batch.logical_path != relative_path
            or (
                row is not None
                and (
                    row.get("byte_count") != embedded_batch.actual_bytes
                    or row.get("tree_sha256") != embedded_batch.checksum_sha256
                )
            )
        ):
            raise _campaign_storage_error(
                run_label,
                f"{embedded_batch.batch_id} disagrees with the signed campaign",
            )
        try:
            root_path = locator.resolve(root_alias).absolute_path.absolute()
            batch_dir = root_path.joinpath(*relative.parts)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise _campaign_storage_error(
                run_label,
                f"{embedded_batch.batch_id} cannot be resolved",
            ) from error
        if path_uses_symlink(root_path) or path_uses_symlink(batch_dir):
            raise _campaign_storage_error(
                run_label,
                f"{embedded_batch.batch_id} path or transfer state is unsafe",
            )
        try:
            root = root_path.resolve(strict=True)
            resolved_batch = batch_dir.resolve(strict=True)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise _campaign_storage_error(
                run_label,
                f"{embedded_batch.batch_id} cannot be resolved",
            ) from error
        if (
            not resolved_batch.is_relative_to(root)
            or batch_dir.is_symlink()
            or not batch_dir.is_dir()
            or batch_dir.with_name(f"{batch_dir.name}.incoming").exists()
        ):
            raise _campaign_storage_error(
                run_label,
                f"{embedded_batch.batch_id} path or transfer state is unsafe",
            )

        batch_manifest_path = batch_dir / "batch_manifest.json"
        batch_manifest_sidecar = batch_dir / "batch_manifest.sha256"
        envelope_path = batch_dir / "batch_persistence_envelope.json"
        envelope_sidecar = batch_dir / "batch_persistence_envelope.sha256"
        try:
            batch_manifest_bytes = batch_manifest_path.read_bytes()
            batch_manifest_sha256 = hashlib.sha256(batch_manifest_bytes).hexdigest()
            disk_batch = load_batch_manifest(batch_manifest_path)
            envelope_sha256 = hashlib.sha256(envelope_path.read_bytes()).hexdigest()
            expected_envelope_sha256 = (
                campaign.batch_persistence_envelope_sha256_by_id[
                    embedded_batch.batch_id
                ]
            )
            envelope_sidecar_sha256 = envelope_sidecar.read_text(
                encoding="ascii"
            ).strip()
            observed = {
                "root_alias": root_alias,
                "relative_path": relative_path,
                "file_count": directory_file_count(batch_dir),
                "byte_count": directory_byte_count(batch_dir),
                "tree_sha256": directory_checksum(batch_dir),
            }
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise _campaign_storage_error(
                run_label,
                f"{embedded_batch.batch_id} cannot be reverified",
            ) from error
        try:
            batch_manifest_sidecar_sha256 = batch_manifest_sidecar.read_text(
                encoding="ascii"
            ).strip()
        except OSError as error:
            raise _campaign_storage_error(
                run_label,
                f"{embedded_batch.batch_id} batch sidecar is missing",
            ) from error
        if (
            batch_manifest_sha256 != batch_manifest_sidecar_sha256
            or disk_batch.to_dict() != embedded_batch.to_dict()
            or envelope_sha256 != expected_envelope_sha256
            or envelope_sidecar_sha256 != expected_envelope_sha256
            or (
                cast(int, observed["file_count"]) <= 0
                or observed["byte_count"] != embedded_batch.actual_bytes
                or observed["tree_sha256"] != embedded_batch.checksum_sha256
            )
            or (row is not None and observed != dict(row))
        ):
            raise _campaign_storage_error(
                run_label,
                f"{embedded_batch.batch_id} content identity changed",
            )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_volume(source: Path, destination_parent: Path) -> bool:
    return source.stat().st_dev == destination_parent.stat().st_dev


def _fsync_tree(path: Path) -> None:
    if os.name == "nt":
        return
    for file_path in _tree_files(path):
        descriptor = os.open(file_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories = (path, *(item for item in path.rglob("*") if item.is_dir()))
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _copy_across_volumes(
    source: Path,
    destination: Path,
    record: RetentionRecord,
) -> None:
    temporary = destination.parent / f".{destination.name}.retention-copy.tmp"
    if temporary.exists():
        if not _matches_record(temporary, record):
            shutil.rmtree(temporary)
            shutil.copytree(source, temporary, copy_function=shutil.copy2)
    else:
        shutil.copytree(source, temporary, copy_function=shutil.copy2)
    _fsync_tree(temporary)
    if not _matches_record(temporary, record):
        raise RetentionIntegrityError(
            f"copied archive evidence verification failed for {record.run_label}"
        )
    os.replace(temporary, destination)
    _fsync_directory(destination.parent)
    if not _matches_record(destination, record):
        raise RetentionIntegrityError(
            f"archived evidence verification failed for {record.run_label}"
        )
    if not _matches_record(source, record):
        raise RetentionIntegrityError(
            f"source changed during cross-volume archive for {record.run_label}"
        )
    cleanup = source.parent / (
        f".{source.name}.retention-cleanup.{record.tree_sha256[:12]}"
    )
    if cleanup.exists():
        raise RetentionIntegrityError(
            f"retention cleanup quarantine already exists for {record.run_label}"
        )
    os.replace(source, cleanup)
    _fsync_directory(source.parent)
    shutil.rmtree(cleanup)


def _archive_record(
    source_root: Path,
    archive_root: Path,
    record: RetentionRecord,
    *,
    archived_at_utc: str,
) -> RetentionRecord:
    source = source_root / record.original_relative_path
    destination = archive_root.joinpath(*PurePosixPath(record.archive_relative_path).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not _matches_record(destination, record):
            raise RetentionIntegrityError(
                f"archive destination collision for {record.run_label}"
            )
        cleanup = source.parent / (
            f".{source.name}.retention-cleanup.{record.tree_sha256[:12]}"
        )
        if source.exists():
            if cleanup.exists():
                raise RetentionIntegrityError(
                    f"source and cleanup quarantine both exist for {record.run_label}"
                )
            if not _matches_record(source, record):
                raise RetentionIntegrityError(
                    f"source changed after audit for {record.run_label}"
                )
            os.replace(source, cleanup)
            _fsync_directory(source_root)
        if cleanup.exists():
            shutil.rmtree(cleanup)
        return replace(
            record,
            disposition="already_archived",
            verification_status="verified",
            archived_at_utc=archived_at_utc,
        )
    if not source.exists():
        raise RetentionIntegrityError(
            f"retention source and archive destination are both missing: {record.run_label}"
        )
    if not _matches_record(source, record):
        raise RetentionIntegrityError(f"source changed after audit for {record.run_label}")
    same_volume = _same_volume(source, destination.parent)
    if not same_volume:
        _copy_across_volumes(source, destination, record)
        _fsync_directory(source_root)
        return replace(
            record,
            disposition="archived",
            verification_status="verified",
            archived_at_utc=archived_at_utc,
        )
    os.replace(source, destination)
    try:
        _fsync_directory(destination.parent)
        _fsync_directory(source_root)
        if not _matches_record(destination, record):
            raise RetentionIntegrityError(
                f"archived evidence verification failed for {record.run_label}"
            )
    except BaseException:
        if not source.exists() and destination.exists():
            os.replace(destination, source)
        raise
    return replace(
        record,
        disposition="archived",
        verification_status="verified",
        archived_at_utc=archived_at_utc,
    )


def _preflight_archive_record(
    source_root: Path,
    archive_root: Path,
    record: RetentionRecord,
) -> None:
    source = source_root / record.original_relative_path
    destination = archive_root.joinpath(*PurePosixPath(record.archive_relative_path).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not _matches_record(destination, record):
            raise RetentionIntegrityError(
                f"archive destination collision for {record.run_label}"
            )
        if source.exists():
            cleanup = source.parent / (
                f".{source.name}.retention-cleanup.{record.tree_sha256[:12]}"
            )
            if cleanup.exists() or not _matches_record(source, record):
                raise RetentionIntegrityError(
                    f"source changed after audit for {record.run_label}"
                )
        return
    if not source.exists():
        raise RetentionIntegrityError(
            f"retention source and archive destination are both missing: {record.run_label}"
        )
    if not _matches_record(source, record):
        raise RetentionIntegrityError(f"source changed after audit for {record.run_label}")


def archive_stage052_inventory(
    inventory_path: Path,
    *,
    inventory_sha256: str,
    archive_root: Path,
    archived_at_utc: str | None = None,
) -> tuple[RetentionRecord, ...]:
    """Archive a signed audit plan, failing before deletion on any mismatch."""

    inventory = load_retention_inventory(
        inventory_path,
        expected_sha256=inventory_sha256,
    )
    source_root = Path(inventory.source_root)
    resolved_archive = archive_root.resolve()
    resolved_archive.mkdir(parents=True, exist_ok=True)
    timestamp = _utc_now() if archived_at_utc is None else archived_at_utc
    for record in inventory.records:
        _preflight_archive_record(source_root, resolved_archive, record)
    return tuple(
        _archive_record(
            source_root,
            resolved_archive,
            record,
            archived_at_utc=timestamp,
        )
        for record in inventory.records
    )


def archive_stage052_inventory_to_registry(
    inventory_path: Path,
    *,
    inventory_sha256: str,
    archive_root: Path,
    registry_path: Path,
    storage_root_locator: StorageRootLocator | None = None,
) -> tuple[RetentionRecord, ...]:
    """Preflight the registry under lock before moving any source evidence."""

    inventory = load_retention_inventory(
        inventory_path,
        expected_sha256=inventory_sha256,
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with _registry_lock(registry_path):
        existing = (
            {record.run_label: record for record in load_retention_registry(registry_path)}
            if registry_path.exists()
            else {}
        )
        for record in inventory.records:
            previous = existing.get(record.run_label)
            if previous is not None and _registry_identity(previous) != _registry_identity(record):
                raise RetentionIntegrityError(
                    f"retention registry identity conflict for {record.run_label}"
                )
            source = Path(inventory.source_root) / record.original_relative_path
            destination = archive_root.resolve().joinpath(
                *PurePosixPath(record.archive_relative_path).parts
            )
            metadata_root = source if source.is_dir() else destination
            _verify_campaign_external_storage(
                metadata_root,
                locator=storage_root_locator,
            )
        archived = archive_stage052_inventory(
            inventory_path,
            inventory_sha256=inventory_sha256,
            archive_root=archive_root,
        )
        _write_retention_registry_locked(registry_path, archived)
        return archived


@contextmanager
def _registry_lock(path: Path) -> Iterator[None]:
    lock_identity = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / f"evrptw-retention-{lock_identity}.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            vars(msvcrt)["locking"](
                handle.fileno(), vars(msvcrt)["LK_LOCK"], 1
            )
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                vars(msvcrt)["locking"](
                    handle.fileno(), vars(msvcrt)["LK_UNLCK"], 1
                )
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_retention_registry(path: Path, records: Sequence[RetentionRecord]) -> None:
    """Merge archived identities under an interprocess lock and publish atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with _registry_lock(path):
        _write_retention_registry_locked(path, records)


def _write_retention_registry_locked(
    path: Path,
    records: Sequence[RetentionRecord],
) -> None:
    merged = (
        {record.run_label: record for record in load_retention_registry(path)}
        if path.exists()
        else {}
    )
    for record in records:
        previous = merged.get(record.run_label)
        if previous is not None:
            if _registry_identity(previous) != _registry_identity(record):
                raise RetentionIntegrityError(
                    f"retention registry identity conflict for {record.run_label}"
                )
            continue
        merged[record.run_label] = record
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_REGISTRY_FIELDS, lineterminator="\n")
            writer.writeheader()
            for record in sorted(merged.values(), key=lambda item: item.run_label):
                writer.writerow(
                    {
                        "schema_version": REGISTRY_SCHEMA_VERSION,
                        **record.to_dict(),
                        "prerequisite_run_labels": "|".join(
                            record.prerequisite_run_labels
                        ),
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _registry_identity(record: RetentionRecord) -> tuple[object, ...]:
    return (
        record.run_label,
        record.component,
        record.status,
        record.evidence_completeness,
        record.source_commit,
        record.prerequisite_run_labels,
        record.original_relative_path,
        record.file_count,
        record.byte_count,
        record.tree_sha256,
        record.archive_root_alias,
        record.archive_relative_path,
    )


def load_retention_registry(path: Path) -> tuple[RetentionRecord, ...]:
    """Load the path-free registry and validate every retained run identity."""

    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(_REGISTRY_FIELDS):
                raise RetentionIntegrityError("retention registry fields do not match schema")
            records: list[RetentionRecord] = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise RetentionIntegrityError("retention registry row is malformed")
                if row["schema_version"] != REGISTRY_SCHEMA_VERSION:
                    raise RetentionIntegrityError("retention registry schema is unsupported")
                prerequisites = (
                    tuple(row["prerequisite_run_labels"].split("|"))
                    if row["prerequisite_run_labels"]
                    else ()
                )
                records.append(
                    RetentionRecord(
                        run_label=row["run_label"],
                        component=row["component"],
                        status=row["status"],
                        evidence_completeness=row["evidence_completeness"],
                        source_commit=row["source_commit"],
                        prerequisite_run_labels=prerequisites,
                        original_relative_path=row["original_relative_path"],
                        file_count=int(row["file_count"]),
                        byte_count=int(row["byte_count"]),
                        tree_sha256=row["tree_sha256"],
                        archive_root_alias=row["archive_root_alias"],
                        archive_relative_path=row["archive_relative_path"],
                        disposition=row["disposition"],
                        verification_status=row["verification_status"],
                        archived_at_utc=row["archived_at_utc"],
                    )
                )
    except RetentionIntegrityError:
        raise
    except (OSError, TypeError, ValueError, csv.Error) as error:
        raise RetentionIntegrityError(f"cannot read retention registry: {path}") from error
    labels = tuple(record.run_label for record in records)
    if labels != tuple(sorted(set(labels))):
        raise RetentionIntegrityError("retention registry rows must be sorted and unique")
    return tuple(records)


def resolve_retained_run(
    run_label: str,
    *,
    registry_path: Path,
    archive_roots: Mapping[str, Path],
) -> Path:
    """Resolve and reverify one archived run using only its label and root alias."""

    if _RUN_LABEL.fullmatch(run_label) is None:
        raise RetentionIntegrityError(f"invalid Stage 5.2 run label: {run_label}")
    matches = [
        record for record in load_retention_registry(registry_path) if record.run_label == run_label
    ]
    if len(matches) != 1:
        raise RetentionIntegrityError(
            f"retention registry must contain exactly one row for {run_label}"
        )
    record = matches[0]
    if record.verification_status != "verified" or record.disposition not in {
        "archived",
        "already_archived",
    }:
        raise RetentionIntegrityError(f"retained run is not verified: {run_label}")
    archive_root = archive_roots.get(record.archive_root_alias)
    if archive_root is None:
        raise RetentionIntegrityError(
            f"archive root alias is not configured: {record.archive_root_alias}"
        )
    resolved_root = archive_root.resolve()
    destination = resolved_root.joinpath(
        *PurePosixPath(record.archive_relative_path).parts
    ).resolve()
    if not destination.is_relative_to(resolved_root):
        raise RetentionIntegrityError(f"retained run path escapes archive root: {run_label}")
    if not _matches_record(destination, record):
        raise RetentionIntegrityError(f"retained run no longer matches registry: {run_label}")
    return destination


def resolve_retained_run_from_locator(
    run_label: str,
    *,
    registry_path: Path,
    storage_root_locator_path: Path,
) -> Path:
    """Resolve a retained run after verifying the local alias-to-volume binding."""

    from evrptw.stage052_campaign import StorageRootLocator
    locator = StorageRootLocator.from_toml(storage_root_locator_path)
    records = load_retention_registry(registry_path)
    aliases = tuple(
        sorted({record.archive_root_alias for record in records if record.run_label == run_label})
    )
    if len(aliases) != 1:
        raise RetentionIntegrityError(
            f"retention registry must bind one archive alias for {run_label}"
        )
    resolved = resolve_retained_run(
        run_label,
        registry_path=registry_path,
        archive_roots={alias: locator.resolve(alias).absolute_path for alias in aliases},
    )
    _verify_campaign_external_storage(resolved, locator=locator)
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="hash Stage 5.2 runs without changing them")
    audit.add_argument("--source-root", type=Path, required=True)
    audit.add_argument("--config", type=Path, required=True)
    audit.add_argument("--inventory", type=Path, required=True)
    audit.add_argument("--allow-historical-unsealed", action="store_true")
    audit.add_argument("--expected-directory-count", type=int)
    audit.add_argument("--expected-total-bytes", type=int)
    audit.add_argument(
        "--run-label",
        action="append",
        default=[],
        help="audit only this exact immutable run label; may be repeated",
    )
    archive = subparsers.add_parser("archive", help="archive an explicitly approved inventory")
    archive.add_argument("--inventory", type=Path, required=True)
    archive.add_argument("--inventory-sha256", required=True)
    archive.add_argument("--storage-root-locator", type=Path, required=True)
    archive.add_argument("--registry", type=Path, required=True)
    supersede = subparsers.add_parser(
        "supersede",
        help="seal one stopped partial campaign without rewriting its signed manifest",
    )
    supersede.add_argument("--run-dir", type=Path, required=True)
    supersede.add_argument("--host-log", type=Path, required=True)
    supersede.add_argument("--host-run-id", required=True)
    supersede.add_argument("--launch-nonce", required=True)
    supersede.add_argument("--expected-exit-code", type=int, required=True)
    supersede.add_argument("--reason", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "audit":
        policy = Stage052RetentionPolicy.from_toml(arguments.config)
        inventory = audit_stage052_runs(
            arguments.source_root,
            policy,
            allow_unsealed=arguments.allow_historical_unsealed,
            expected_directory_count=arguments.expected_directory_count,
            expected_total_bytes=arguments.expected_total_bytes,
            run_labels=tuple(arguments.run_label),
        )
        sha256 = write_retention_inventory(arguments.inventory, inventory)
        print(
            json.dumps(
                {
                    "directory_count": inventory.directory_count,
                    "total_bytes": inventory.total_bytes,
                    "inventory_sha256": sha256,
                    "inventory": str(arguments.inventory.resolve()),
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "supersede":
        receipt_sha256 = create_supersession_receipt(
            arguments.run_dir,
            host_log=arguments.host_log,
            host_run_id=arguments.host_run_id,
            launch_nonce=arguments.launch_nonce,
            expected_exit_code=arguments.expected_exit_code,
            reason=arguments.reason,
        )
        print(
            json.dumps(
                {
                    "run_label": arguments.run_dir.resolve().name,
                    "status": "superseded",
                    "evidence_completeness": "partial",
                    "receipt_sha256": receipt_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    from evrptw.stage052_campaign import StorageRootLocator
    approved = load_retention_inventory(
        arguments.inventory,
        expected_sha256=arguments.inventory_sha256,
    )
    locator = StorageRootLocator.from_toml(arguments.storage_root_locator)
    alias = approved.policy.archive_root_alias
    if alias == "e_archive":
        raise RetentionIntegrityError(
            "legacy Stage 5.2 retention v1 cannot publish to e_archive; "
            "use experiment retention v2 so historical source deletion remains "
            "a separately confirmed action"
        )
    locator.verify_all(probe_volume_identity, (alias,))
    records = archive_stage052_inventory_to_registry(
        arguments.inventory,
        inventory_sha256=arguments.inventory_sha256,
        archive_root=locator.resolve(alias).absolute_path,
        registry_path=arguments.registry,
        storage_root_locator=locator,
    )
    print(
        json.dumps(
            {
                "archived_runs": len(records),
                "verified_runs": sum(
                    record.verification_status == "verified" for record in records
                ),
                "registry": str(arguments.registry.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
