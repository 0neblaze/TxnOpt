"""Provider-neutral immutable archive port and local filesystem adapter."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Protocol, runtime_checkable

from txnopt_evidence.codec import canonical_json_bytes

_COMMIT_SCHEMA = "txnopt-archive-commit-v1"
_COMMIT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class ArchiveError(RuntimeError):
    """Base error for archive operations."""


class ArchiveIntegrityError(ArchiveError):
    """Raised when bytes differ from their declared immutable identity."""


class ArchiveNotFoundError(ArchiveError):
    """Raised when an immutable archive object is absent."""


class ArchiveConflictError(ArchiveError):
    """Raised when an immutable name is already bound to other bytes."""


@dataclass(frozen=True, slots=True)
class ArchiveObjectRef:
    sha256: str
    size: int

    def __post_init__(self) -> None:
        _validate_sha256(self.sha256)
        if self.size < 0:
            raise ValueError("archive object size must be non-negative")


@dataclass(frozen=True, slots=True)
class ArchiveObjectMeta:
    ref: ArchiveObjectRef
    verified: bool


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    relative_path: str
    ref: ArchiveObjectRef

    def __post_init__(self) -> None:
        _validate_relative_path(self.relative_path)


@dataclass(frozen=True, slots=True)
class ArchiveCommit:
    commit_id: str
    entries: tuple[ArchiveEntry, ...]
    schema_version: str = _COMMIT_SCHEMA

    def __post_init__(self) -> None:
        _validate_commit_id(self.commit_id)
        if self.schema_version != _COMMIT_SCHEMA:
            raise ValueError("archive commit schema differs from txnopt-archive-commit-v1")
        paths = tuple(entry.relative_path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("archive commit entries must be uniquely sorted by relative path")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "commit_id": self.commit_id,
            "objects": [
                {
                    "relative_path": entry.relative_path,
                    "sha256": entry.ref.sha256,
                    "size": entry.ref.size,
                }
                for entry in self.entries
            ],
        }


@dataclass(frozen=True, slots=True)
class ArchiveCommitRef:
    commit_id: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        _validate_commit_id(self.commit_id)
        _validate_sha256(self.sha256)
        if self.size < 0:
            raise ValueError("archive commit size must be non-negative")


@dataclass(frozen=True, slots=True)
class ArchiveVerificationReceipt:
    commit: ArchiveCommit
    ref: ArchiveCommitRef
    object_count: int
    total_size: int
    verified: bool


@dataclass(frozen=True, slots=True)
class ArchiveInventoryEntry:
    relative_path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        _validate_relative_path(self.relative_path)
        _validate_sha256(self.sha256)
        if self.size < 0:
            raise ValueError("archive inventory size must be non-negative")

    @property
    def ref(self) -> ArchiveObjectRef:
        return ArchiveObjectRef(sha256=self.sha256, size=self.size)


@dataclass(frozen=True, slots=True)
class ArchiveInventory:
    entries: tuple[ArchiveInventoryEntry, ...]
    tree_sha256: str
    total_size: int
    schema_version: str = "txnopt-archive-inventory-v1"

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tree_sha256": self.tree_sha256,
            "file_count": len(self.entries),
            "total_size": self.total_size,
            "files": [
                {
                    "relative_path": entry.relative_path,
                    "sha256": entry.sha256,
                    "size": entry.size,
                }
                for entry in self.entries
            ],
        }


@dataclass(frozen=True, slots=True)
class ArchiveRestoreReceipt:
    commit_ref: ArchiveCommitRef
    object_count: int
    total_size: int
    verified: bool


@runtime_checkable
class ArchiveStore(Protocol):
    """Immutable blob port; implementations may use local or object storage."""

    def put_blob(
        self,
        source: BinaryIO,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> ArchiveObjectRef: ...

    def open(self, ref: ArchiveObjectRef) -> BinaryIO: ...

    def head(self, ref: ArchiveObjectRef) -> ArchiveObjectMeta: ...

    def publish_commit(
        self,
        commit: ArchiveCommit,
        *,
        expected_absent: bool = True,
    ) -> ArchiveCommitRef: ...

    def verify_commit(self, ref: ArchiveCommitRef) -> ArchiveVerificationReceipt: ...


class LocalFilesystemArchiveStore:
    """Content-addressed immutable blobs stored below one local root."""

    def __init__(self, root: Path) -> None:
        if root.is_symlink():
            raise ValueError("archive root cannot be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise ValueError("archive root must be a directory")
        self._root = root.resolve(strict=True)
        self._staging = self._root / ".staging"
        self._staging.mkdir(exist_ok=True)

    def put_blob(
        self,
        source: BinaryIO,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> ArchiveObjectRef:
        _validate_sha256(expected_sha256)
        if expected_size < 0:
            raise ValueError("expected blob size must be non-negative")
        ref = ArchiveObjectRef(sha256=expected_sha256, size=expected_size)
        descriptor, temporary_name = tempfile.mkstemp(prefix="blob-", dir=self._staging)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        observed_size = 0
        try:
            with os.fdopen(descriptor, "wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    if not isinstance(chunk, bytes):
                        raise TypeError("archive blob source must yield bytes")
                    output.write(chunk)
                    digest.update(chunk)
                    observed_size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if observed_size != expected_size or digest.hexdigest() != expected_sha256:
                raise ArchiveIntegrityError("archive blob differs from expected digest or size")

            destination = self._blob_path(ref)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(temporary, destination)
                _fsync_directory(destination.parent)
            except FileExistsError:
                self._verified_meta(ref)
            return ref
        finally:
            temporary.unlink(missing_ok=True)

    def open(self, ref: ArchiveObjectRef) -> BinaryIO:
        path = self._blob_path(ref)
        if path.is_symlink():
            raise ArchiveIntegrityError("archive blob cannot be a symlink")
        try:
            return path.open("rb")
        except FileNotFoundError as error:
            raise ArchiveNotFoundError(f"archive blob is absent: {ref.sha256}") from error

    def head(self, ref: ArchiveObjectRef) -> ArchiveObjectMeta:
        return self._verified_meta(ref)

    def publish_commit(
        self,
        commit: ArchiveCommit,
        *,
        expected_absent: bool = True,
    ) -> ArchiveCommitRef:
        for entry in commit.entries:
            self._verified_meta(entry.ref)
        data = canonical_json_bytes(commit.to_payload(), pretty=True)
        ref = ArchiveCommitRef(
            commit_id=commit.commit_id,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        )
        destination = self._commit_path(ref.commit_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix="commit-", dir=self._staging)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, destination)
                _fsync_directory(destination.parent)
            except FileExistsError as error:
                if expected_absent:
                    raise ArchiveConflictError(
                        f"archive commit already exists: {commit.commit_id}"
                    ) from error
                if destination.is_symlink() or destination.read_bytes() != data:
                    raise ArchiveConflictError(
                        f"archive commit name is bound to other bytes: {commit.commit_id}"
                    ) from error
            return ref
        finally:
            temporary.unlink(missing_ok=True)

    def verify_commit(self, ref: ArchiveCommitRef) -> ArchiveVerificationReceipt:
        path = self._commit_path(ref.commit_id)
        if path.is_symlink():
            raise ArchiveIntegrityError("archive commit cannot be a symlink")
        try:
            data = path.read_bytes()
        except FileNotFoundError as error:
            raise ArchiveNotFoundError(f"archive commit is absent: {ref.commit_id}") from error
        if len(data) != ref.size or hashlib.sha256(data).hexdigest() != ref.sha256:
            raise ArchiveIntegrityError("archive commit differs from its immutable reference")
        try:
            payload: object = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArchiveIntegrityError("archive commit is not valid JSON") from error
        if canonical_json_bytes(payload, pretty=True) != data:
            raise ArchiveIntegrityError("archive commit is not canonical JSON")
        commit = _commit_from_payload(payload)
        if commit.commit_id != ref.commit_id:
            raise ArchiveIntegrityError("archive commit identity differs from its reference")
        total_size = 0
        for entry in commit.entries:
            self._verified_meta(entry.ref)
            total_size += entry.ref.size
        return ArchiveVerificationReceipt(
            commit=commit,
            ref=ref,
            object_count=len(commit.entries),
            total_size=total_size,
            verified=True,
        )

    def _blob_path(self, ref: ArchiveObjectRef) -> Path:
        return self._root / "blobs" / "sha256" / ref.sha256[:2] / ref.sha256

    def _commit_path(self, commit_id: str) -> Path:
        _validate_commit_id(commit_id)
        return self._root / "commits" / f"{commit_id}.json"

    def _verified_meta(self, ref: ArchiveObjectRef) -> ArchiveObjectMeta:
        path = self._blob_path(ref)
        if path.is_symlink():
            raise ArchiveIntegrityError("archive blob cannot be a symlink")
        if not path.is_file():
            raise ArchiveNotFoundError(f"archive blob is absent: {ref.sha256}")
        digest = hashlib.sha256()
        observed_size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                observed_size += len(chunk)
        if observed_size != ref.size or digest.hexdigest() != ref.sha256:
            raise ArchiveIntegrityError("archive blob differs from its immutable reference")
        return ArchiveObjectMeta(ref=ref, verified=True)


def inventory_tree(source: Path) -> ArchiveInventory:
    """Hash a regular-file tree without following symlinks."""

    if source.is_symlink():
        raise ValueError("archive inventory root cannot be a symlink")
    try:
        root = source.resolve(strict=True)
    except FileNotFoundError as error:
        raise ArchiveNotFoundError(f"archive inventory root is absent: {source}") from error
    if not root.is_dir():
        raise ValueError("archive inventory root must be a directory")
    entries: list[ArchiveInventoryEntry] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"archive inventory cannot contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"archive inventory accepts regular files only: {path}")
        sha256, size = _hash_file(path)
        entries.append(
            ArchiveInventoryEntry(
                relative_path=path.relative_to(root).as_posix(),
                sha256=sha256,
                size=size,
            )
        )
    payload = [
        {
            "relative_path": entry.relative_path,
            "sha256": entry.sha256,
            "size": entry.size,
        }
        for entry in entries
    ]
    return ArchiveInventory(
        entries=tuple(entries),
        tree_sha256=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        total_size=sum(entry.size for entry in entries),
    )


def mirror_tree(
    source: Path,
    *,
    store: ArchiveStore,
    commit_id: str,
) -> ArchiveCommitRef:
    """Copy one verified tree into immutable blobs and publish its commit last."""

    inventory = inventory_tree(source)
    root = source.resolve(strict=True)
    commit_entries: list[ArchiveEntry] = []
    for entry in inventory.entries:
        path = root.joinpath(*PurePosixPath(entry.relative_path).parts)
        if path.is_symlink():
            raise ValueError("archive source changed to a symlink after inventory")
        with path.open("rb") as handle:
            ref = store.put_blob(
                handle,
                expected_sha256=entry.sha256,
                expected_size=entry.size,
            )
        commit_entries.append(ArchiveEntry(relative_path=entry.relative_path, ref=ref))
    return store.publish_commit(
        ArchiveCommit(commit_id=commit_id, entries=tuple(commit_entries))
    )


def restore_commit(
    store: ArchiveStore,
    ref: ArchiveCommitRef,
    *,
    destination: Path,
) -> ArchiveRestoreReceipt:
    """Restore a verified commit through an invisible staging directory."""

    receipt = store.verify_commit(ref)
    if destination.exists() or destination.is_symlink():
        raise ArchiveConflictError(f"restore destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("restore destination parent cannot be a symlink")
    parent = destination.parent.resolve(strict=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.txnopt-restore-", dir=parent))
    try:
        for entry in receipt.commit.entries:
            output_path = staging.joinpath(*PurePosixPath(entry.relative_path).parts)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            observed_size = 0
            with store.open(entry.ref) as source, output_path.open("xb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    observed_size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if observed_size != entry.ref.size or digest.hexdigest() != entry.ref.sha256:
                raise ArchiveIntegrityError("restored blob differs from its immutable reference")
        if destination.exists() or destination.is_symlink():
            raise ArchiveConflictError(
                f"restore destination appeared during restore: {destination}"
            )
        os.rename(staging, destination)
        _fsync_directory(parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ArchiveRestoreReceipt(
        commit_ref=ref,
        object_count=receipt.object_count,
        total_size=receipt.total_size,
        verified=True,
    )


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("SHA-256 identity must be 64 lowercase hexadecimal characters")


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _validate_commit_id(value: str) -> None:
    if _COMMIT_ID.fullmatch(value) is None:
        raise ValueError("archive commit id is not a canonical path-safe identifier")


def _validate_relative_path(value: str) -> None:
    candidate = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or candidate.is_absolute()
        or value != candidate.as_posix()
        or value == "."
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("archive entry path must be canonical, relative, and traversal-free")


def _commit_from_payload(payload: object) -> ArchiveCommit:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "commit_id",
        "objects",
    }:
        raise ArchiveIntegrityError("archive commit field set differs from its schema")
    schema_version = payload.get("schema_version")
    commit_id = payload.get("commit_id")
    objects = payload.get("objects")
    if schema_version != _COMMIT_SCHEMA or not isinstance(commit_id, str):
        raise ArchiveIntegrityError("archive commit header differs from its schema")
    if not isinstance(objects, list):
        raise ArchiveIntegrityError("archive commit objects must be a list")
    entries: list[ArchiveEntry] = []
    for item in objects:
        if not isinstance(item, dict) or set(item) != {"relative_path", "sha256", "size"}:
            raise ArchiveIntegrityError("archive commit object field set differs")
        relative_path = item.get("relative_path")
        sha256 = item.get("sha256")
        size = item.get("size")
        if (
            not isinstance(relative_path, str)
            or not isinstance(sha256, str)
            or type(size) is not int
        ):
            raise ArchiveIntegrityError("archive commit object types differ")
        try:
            entries.append(
                ArchiveEntry(
                    relative_path=relative_path,
                    ref=ArchiveObjectRef(sha256=sha256, size=size),
                )
            )
        except ValueError as error:
            raise ArchiveIntegrityError("archive commit object identity differs") from error
    try:
        return ArchiveCommit(commit_id=commit_id, entries=tuple(entries))
    except ValueError as error:
        raise ArchiveIntegrityError("archive commit ordering or identity differs") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ArchiveError",
    "ArchiveCommit",
    "ArchiveCommitRef",
    "ArchiveConflictError",
    "ArchiveEntry",
    "ArchiveIntegrityError",
    "ArchiveInventory",
    "ArchiveInventoryEntry",
    "ArchiveNotFoundError",
    "ArchiveObjectMeta",
    "ArchiveObjectRef",
    "ArchiveRestoreReceipt",
    "ArchiveStore",
    "ArchiveVerificationReceipt",
    "LocalFilesystemArchiveStore",
    "inventory_tree",
    "mirror_tree",
    "restore_commit",
]
