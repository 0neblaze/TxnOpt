"""Provider-neutral immutable archive port and local filesystem adapter."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol, runtime_checkable

from txnopt_evidence.codec import canonical_json_bytes

_fcntl: Any = importlib.import_module("fcntl") if os.name == "posix" else None
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
    """Content-addressed immutable blobs stored below one POSIX local root."""

    def __init__(self, root: Path, *, create: bool = True) -> None:
        _require_posix_archive_primitives()
        self._root = _absolute_without_symlink_resolution(root)
        self._writable = create
        try:
            root_descriptor = _open_directory_path(self._root, create=create)
        except FileNotFoundError as error:
            raise ArchiveNotFoundError(f"archive root is absent: {self._root}") from error
        os.close(root_descriptor)
        if not create:
            return
        with self._locked_root() as root_descriptor:
            staging_descriptor = _open_child_directory(
                root_descriptor,
                ".staging",
                create=True,
            )
            try:
                _clean_staging_directory(staging_descriptor)
            finally:
                os.close(staging_descriptor)

    def put_blob(
        self,
        source: BinaryIO,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> ArchiveObjectRef:
        self._require_writable()
        _validate_sha256(expected_sha256)
        if expected_size < 0:
            raise ValueError("expected blob size must be non-negative")
        ref = ArchiveObjectRef(sha256=expected_sha256, size=expected_size)
        with self._locked_root() as root_descriptor:
            staging_descriptor = _open_child_directory(
                root_descriptor,
                ".staging",
                create=True,
            )
            try:
                temporary_name, descriptor = _create_staging_file(
                    staging_descriptor,
                    prefix="blob",
                )
            except Exception:
                os.close(staging_descriptor)
                raise
            try:
                digest = hashlib.sha256()
                observed_size = 0
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
                    raise ArchiveIntegrityError(
                        "archive blob differs from expected digest or size"
                    )

                prefix_descriptor = self._open_blob_prefix(
                    root_descriptor,
                    ref,
                    create=True,
                )
                try:
                    try:
                        os.link(
                            temporary_name,
                            ref.sha256,
                            src_dir_fd=staging_descriptor,
                            dst_dir_fd=prefix_descriptor,
                            follow_symlinks=False,
                        )
                        os.fsync(prefix_descriptor)
                    except FileExistsError:
                        verified_descriptor = self._open_verified_blob_descriptor(
                            root_descriptor,
                            ref,
                        )
                        os.close(verified_descriptor)
                finally:
                    os.close(prefix_descriptor)
            finally:
                try:
                    with suppress(FileNotFoundError):
                        os.unlink(temporary_name, dir_fd=staging_descriptor)
                finally:
                    os.close(staging_descriptor)
            return ref

    def open(self, ref: ArchiveObjectRef) -> BinaryIO:
        root_descriptor = _open_directory_path(self._root, create=False)
        try:
            descriptor = self._open_verified_blob_descriptor(root_descriptor, ref)
        finally:
            os.close(root_descriptor)
        return os.fdopen(descriptor, "rb")

    def head(self, ref: ArchiveObjectRef) -> ArchiveObjectMeta:
        root_descriptor = _open_directory_path(self._root, create=False)
        try:
            descriptor = self._open_verified_blob_descriptor(root_descriptor, ref)
            os.close(descriptor)
        finally:
            os.close(root_descriptor)
        return ArchiveObjectMeta(ref=ref, verified=True)

    def publish_commit(
        self,
        commit: ArchiveCommit,
        *,
        expected_absent: bool = True,
    ) -> ArchiveCommitRef:
        self._require_writable()
        data = canonical_json_bytes(commit.to_payload(), pretty=True)
        ref = ArchiveCommitRef(
            commit_id=commit.commit_id,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
        )
        with self._locked_root() as root_descriptor:
            for entry in commit.entries:
                verified_descriptor = self._open_verified_blob_descriptor(
                    root_descriptor,
                    entry.ref,
                )
                os.close(verified_descriptor)
            staging_descriptor = _open_child_directory(
                root_descriptor,
                ".staging",
                create=True,
            )
            try:
                temporary_name, descriptor = _create_staging_file(
                    staging_descriptor,
                    prefix="commit",
                )
            except Exception:
                os.close(staging_descriptor)
                raise
            try:
                with os.fdopen(descriptor, "wb") as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                commits_descriptor = _open_child_directory(
                    root_descriptor,
                    "commits",
                    create=True,
                )
                try:
                    commit_name = _commit_filename(commit.commit_id)
                    try:
                        os.link(
                            temporary_name,
                            commit_name,
                            src_dir_fd=staging_descriptor,
                            dst_dir_fd=commits_descriptor,
                            follow_symlinks=False,
                        )
                        os.fsync(commits_descriptor)
                    except FileExistsError as error:
                        if expected_absent:
                            raise ArchiveConflictError(
                                f"archive commit already exists: {commit.commit_id}"
                            ) from error
                        existing = _read_regular_file_at(
                            commits_descriptor,
                            commit_name,
                            object_kind="archive commit",
                        )
                        if existing != data:
                            raise ArchiveConflictError(
                                "archive commit name is bound to other bytes: "
                                f"{commit.commit_id}"
                            ) from error
                finally:
                    os.close(commits_descriptor)
            finally:
                try:
                    with suppress(FileNotFoundError):
                        os.unlink(temporary_name, dir_fd=staging_descriptor)
                finally:
                    os.close(staging_descriptor)
            return ref

    def verify_commit(self, ref: ArchiveCommitRef) -> ArchiveVerificationReceipt:
        root_descriptor = _open_directory_path(self._root, create=False)
        try:
            commits_descriptor = _open_child_directory(
                root_descriptor,
                "commits",
                create=False,
            )
            try:
                data = _read_regular_file_at(
                    commits_descriptor,
                    _commit_filename(ref.commit_id),
                    object_kind="archive commit",
                )
            finally:
                os.close(commits_descriptor)
            if len(data) != ref.size or hashlib.sha256(data).hexdigest() != ref.sha256:
                raise ArchiveIntegrityError(
                    "archive commit differs from its immutable reference"
                )
            try:
                payload: object = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ArchiveIntegrityError("archive commit is not valid JSON") from error
            if canonical_json_bytes(payload, pretty=True) != data:
                raise ArchiveIntegrityError("archive commit is not canonical JSON")
            commit = _commit_from_payload(payload)
            if commit.commit_id != ref.commit_id:
                raise ArchiveIntegrityError(
                    "archive commit identity differs from its reference"
                )
            total_size = 0
            for entry in commit.entries:
                descriptor = self._open_verified_blob_descriptor(root_descriptor, entry.ref)
                os.close(descriptor)
                total_size += entry.ref.size
            return ArchiveVerificationReceipt(
                commit=commit,
                ref=ref,
                object_count=len(commit.entries),
                total_size=total_size,
                verified=True,
            )
        finally:
            os.close(root_descriptor)

    @contextmanager
    def _locked_root(self) -> Iterator[int]:
        if _fcntl is None:  # pragma: no cover - guarded by constructor
            raise ArchiveError("POSIX archive writer locking is unavailable")
        root_descriptor = _open_directory_path(self._root, create=False)
        try:
            lock_descriptor = _open_lock_file(root_descriptor)
        except Exception:
            os.close(root_descriptor)
            raise
        locked = False
        try:
            _fcntl.flock(lock_descriptor, _fcntl.LOCK_EX)
            locked = True
            yield root_descriptor
        finally:
            try:
                if locked:
                    _fcntl.flock(lock_descriptor, _fcntl.LOCK_UN)
            finally:
                try:
                    os.close(lock_descriptor)
                finally:
                    os.close(root_descriptor)

    def _require_writable(self) -> None:
        if not self._writable:
            raise ArchiveError("archive store was opened in read-only mode")

    def _contains_path(self, path: Path) -> bool:
        return _path_is_within(self._root, _absolute_without_symlink_resolution(path))

    def _overlaps_path(self, path: Path) -> bool:
        candidate = _absolute_without_symlink_resolution(path)
        return _path_is_within(self._root, candidate) or _path_is_within(
            candidate,
            self._root,
        )

    def _open_blob_prefix(
        self,
        root_descriptor: int,
        ref: ArchiveObjectRef,
        *,
        create: bool,
    ) -> int:
        blobs_descriptor = _open_child_directory(
            root_descriptor,
            "blobs",
            create=create,
        )
        try:
            sha_descriptor = _open_child_directory(
                blobs_descriptor,
                "sha256",
                create=create,
            )
        finally:
            os.close(blobs_descriptor)
        try:
            return _open_child_directory(
                sha_descriptor,
                ref.sha256[:2],
                create=create,
            )
        finally:
            os.close(sha_descriptor)

    def _open_verified_blob_descriptor(
        self,
        root_descriptor: int,
        ref: ArchiveObjectRef,
    ) -> int:
        try:
            prefix_descriptor = self._open_blob_prefix(
                root_descriptor,
                ref,
                create=False,
            )
        except FileNotFoundError as error:
            raise ArchiveNotFoundError(
                f"archive blob is absent: {ref.sha256}"
            ) from error
        try:
            descriptor = _open_regular_file_at(
                prefix_descriptor,
                ref.sha256,
                object_kind="archive blob",
            )
        finally:
            os.close(prefix_descriptor)
        try:
            digest, size = _hash_descriptor(descriptor)
            if size != ref.size or digest != ref.sha256:
                raise ArchiveIntegrityError(
                    "archive blob differs from its immutable reference"
                )
            os.lseek(descriptor, 0, os.SEEK_SET)
        except Exception:
            os.close(descriptor)
            raise
        return descriptor


def inventory_tree(source: Path) -> ArchiveInventory:
    """Hash a regular-file tree without following symlinks."""

    _require_posix_archive_primitives()
    root_path = _absolute_without_symlink_resolution(source)
    try:
        root_descriptor = _open_directory_path(root_path, create=False)
    except FileNotFoundError as error:
        raise ArchiveNotFoundError(f"archive inventory root is absent: {source}") from error
    entries: list[ArchiveInventoryEntry] = []
    try:
        _inventory_directory(root_descriptor, PurePosixPath(), entries)
    finally:
        os.close(root_descriptor)
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

    _require_posix_archive_primitives()
    if isinstance(store, LocalFilesystemArchiveStore) and store._overlaps_path(source):
        raise ValueError("archive source and local store roots must not overlap")
    inventory = inventory_tree(source)
    root_descriptor = _open_directory_path(
        _absolute_without_symlink_resolution(source),
        create=False,
    )
    commit_entries: list[ArchiveEntry] = []
    try:
        for entry in inventory.entries:
            descriptor = _open_relative_regular_file(
                root_descriptor,
                PurePosixPath(entry.relative_path).parts,
                object_kind="archive source file",
            )
            with os.fdopen(descriptor, "rb") as handle:
                ref = store.put_blob(
                    handle,
                    expected_sha256=entry.sha256,
                    expected_size=entry.size,
                )
            commit_entries.append(
                ArchiveEntry(relative_path=entry.relative_path, ref=ref)
            )
    finally:
        os.close(root_descriptor)
    final_inventory = inventory_tree(source)
    if final_inventory != inventory:
        raise ArchiveIntegrityError("archive source tree changed during mirroring")
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

    _require_posix_archive_primitives()
    receipt = store.verify_commit(ref)
    destination_path = _absolute_without_symlink_resolution(destination)
    if isinstance(store, LocalFilesystemArchiveStore) and store._contains_path(
        destination_path
    ):
        raise ValueError("restore destination cannot be inside the local archive store")
    destination_name = destination_path.name
    if destination_name in {"", ".", ".."}:
        raise ValueError("restore destination must name one directory")
    parent_descriptor = _open_directory_path(destination_path.parent, create=True)
    staging_name = f".{destination_name}.txnopt-restore-{secrets.token_hex(16)}"
    try:
        try:
            os.stat(destination_name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ArchiveConflictError(
                f"restore destination already exists: {destination_path}"
            )
        os.mkdir(staging_name, mode=0o700, dir_fd=parent_descriptor)
        staging_descriptor = _open_child_directory(
            parent_descriptor,
            staging_name,
            create=False,
        )
    except Exception:
        os.close(parent_descriptor)
        raise
    published = False
    try:
        for entry in receipt.commit.entries:
            _restore_entry(store, entry, staging_descriptor)
        os.fsync(staging_descriptor)
        _assert_named_directory_identity(
            parent_descriptor,
            staging_name,
            staging_descriptor,
        )
        _rename_directory_noreplace(
            parent_descriptor,
            staging_name,
            destination_name,
        )
        published = True
        os.fsync(parent_descriptor)
    except Exception:
        raise
    finally:
        os.close(staging_descriptor)
        try:
            if not published:
                _remove_directory_at(parent_descriptor, staging_name)
        finally:
            os.close(parent_descriptor)
    return ArchiveRestoreReceipt(
        commit_ref=ref,
        object_count=receipt.object_count,
        total_size=receipt.total_size,
        verified=True,
    )


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("SHA-256 identity must be 64 lowercase hexadecimal characters")


def _hash_descriptor(descriptor: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _require_posix_archive_primitives() -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if (
        os.name != "posix"
        or not sys.platform.startswith("linux")
        or not hasattr(os, "O_NOFOLLOW")
        or _fcntl is None
        or renameat2 is None
    ):
        raise ArchiveError(
            "LocalFilesystemArchiveStore requires Linux no-follow and renameat2 primitives"
        )


def _path_is_within(parent: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _inventory_directory(
    directory_descriptor: int,
    prefix: PurePosixPath,
    entries: list[ArchiveInventoryEntry],
) -> None:
    with os.scandir(directory_descriptor) as scanned:
        names = tuple(sorted(entry.name for entry in scanned))
    for name in names:
        relative = prefix / name
        try:
            status = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError as error:
            raise ArchiveIntegrityError(
                f"archive source changed during inventory: {relative.as_posix()}"
            ) from error
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(
                f"archive inventory cannot contain symlinks: {relative.as_posix()}"
            )
        if stat.S_ISDIR(status.st_mode):
            child_descriptor = _open_child_directory(
                directory_descriptor,
                name,
                create=False,
            )
            try:
                _inventory_directory(child_descriptor, relative, entries)
            finally:
                os.close(child_descriptor)
            continue
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(
                "archive inventory accepts regular files only: "
                f"{relative.as_posix()}"
            )
        descriptor = _open_regular_file_at(
            directory_descriptor,
            name,
            object_kind="archive source file",
        )
        try:
            sha256, size = _hash_descriptor(descriptor)
        finally:
            os.close(descriptor)
        entries.append(
            ArchiveInventoryEntry(
                relative_path=relative.as_posix(),
                sha256=sha256,
                size=size,
            )
        )


def _open_relative_regular_file(
    root_descriptor: int,
    parts: tuple[str, ...],
    *,
    object_kind: str,
) -> int:
    if not parts:
        raise ArchiveIntegrityError("archive relative file path is empty")
    parent_descriptor = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            child = _open_child_directory(parent_descriptor, component, create=False)
            os.close(parent_descriptor)
            parent_descriptor = child
        return _open_regular_file_at(
            parent_descriptor,
            parts[-1],
            object_kind=object_kind,
        )
    finally:
        os.close(parent_descriptor)


def _open_directory_path(path: Path, *, create: bool) -> int:
    absolute = _absolute_without_symlink_resolution(path)
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:]:
            try:
                child = _open_child_directory(descriptor, component, create=create)
            except Exception:
                os.close(descriptor)
                raise
            os.close(descriptor)
            descriptor = child
    except Exception:
        raise
    return descriptor


def _open_child_directory(parent_descriptor: int, name: str, *, create: bool) -> int:
    if name in {"", ".", ".."} or "/" in name or "\\" in name:
        raise ArchiveIntegrityError("archive directory component is not canonical")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd=parent_descriptor)
    except FileNotFoundError:
        if not create:
            raise
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        except FileExistsError:
            pass
        try:
            return os.open(name, flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise ArchiveIntegrityError(
                f"archive directory is a symlink or not a safe real directory: {name}"
            ) from error
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ArchiveIntegrityError(
                f"archive directory is a symlink or not a safe real directory: {name}"
            ) from error
        raise


def _open_lock_file(root_descriptor: int) -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
    try:
        descriptor = os.open(
            ".writer.lock",
            flags,
            mode=0o600,
            dir_fd=root_descriptor,
        )
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ArchiveIntegrityError("archive writer lock cannot be a symlink") from error
        raise
    try:
        status = os.fstat(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    if not stat.S_ISREG(status.st_mode):
        os.close(descriptor)
        raise ArchiveIntegrityError("archive writer lock must be a regular file")
    return descriptor


def _clean_staging_directory(staging_descriptor: int) -> None:
    with os.scandir(staging_descriptor) as entries:
        names = tuple(entry.name for entry in entries)
    for name in names:
        try:
            status = os.stat(name, dir_fd=staging_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(status.st_mode):
            raise ArchiveIntegrityError(
                "archive staging contains a non-regular unexpected entry"
            )
        os.unlink(name, dir_fd=staging_descriptor)
    if names:
        os.fsync(staging_descriptor)


def _create_staging_file(directory_descriptor: int, *, prefix: str) -> tuple[str, int]:
    for _attempt in range(128):
        name = f"{prefix}-{secrets.token_hex(16)}"
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                mode=0o600,
                dir_fd=directory_descriptor,
            )
        except FileExistsError:
            continue
        return name, descriptor
    raise ArchiveConflictError("archive staging name allocation was exhausted")


def _open_regular_file_at(
    directory_descriptor: int,
    name: str,
    *,
    object_kind: str,
) -> int:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
    except FileNotFoundError as error:
        raise ArchiveNotFoundError(f"{object_kind} is absent: {name}") from error
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ArchiveIntegrityError(f"{object_kind} cannot be a symlink") from error
        raise
    try:
        status = os.fstat(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    if not stat.S_ISREG(status.st_mode):
        os.close(descriptor)
        raise ArchiveIntegrityError(f"{object_kind} must be a regular file")
    return descriptor


def _read_regular_file_at(
    directory_descriptor: int,
    name: str,
    *,
    object_kind: str,
) -> bytes:
    descriptor = _open_regular_file_at(
        directory_descriptor,
        name,
        object_kind=object_kind,
    )
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _commit_filename(commit_id: str) -> str:
    _validate_commit_id(commit_id)
    return f"{commit_id}.json"


def _restore_entry(
    store: ArchiveStore,
    entry: ArchiveEntry,
    staging_descriptor: int,
) -> None:
    parts = PurePosixPath(entry.relative_path).parts
    parent_descriptor = os.dup(staging_descriptor)
    try:
        for component in parts[:-1]:
            child = _open_child_directory(parent_descriptor, component, create=True)
            os.close(parent_descriptor)
            parent_descriptor = child
        output_descriptor = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            mode=0o600,
            dir_fd=parent_descriptor,
        )
        digest = hashlib.sha256()
        observed_size = 0
        try:
            with store.open(entry.ref) as source:
                while chunk := source.read(1024 * 1024):
                    if not isinstance(chunk, bytes):
                        raise TypeError("archive blob source must yield bytes")
                    _write_all(output_descriptor, chunk)
                    digest.update(chunk)
                    observed_size += len(chunk)
            os.fsync(output_descriptor)
        finally:
            os.close(output_descriptor)
        if observed_size != entry.ref.size or digest.hexdigest() != entry.ref.sha256:
            raise ArchiveIntegrityError(
                "restored blob differs from its immutable reference"
            )
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _rename_directory_noreplace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    function = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if function is None:
        raise ArchiveError("atomic no-replace directory publication is unavailable")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(destination_name),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ArchiveConflictError(
            f"restore destination appeared during restore: {destination_name}"
        )
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _assert_named_directory_identity(
    parent_descriptor: int,
    name: str,
    descriptor: int,
) -> None:
    expected = os.fstat(descriptor)
    try:
        observed = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ArchiveIntegrityError("restore staging directory name disappeared") from error
    if (
        not stat.S_ISDIR(observed.st_mode)
        or observed.st_dev != expected.st_dev
        or observed.st_ino != expected.st_ino
    ):
        raise ArchiveIntegrityError("restore staging directory identity changed")


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("archive restore write made no progress")
        view = view[written:]


def _remove_directory_at(parent_descriptor: int, name: str) -> None:
    try:
        status = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(status.st_mode):
        os.unlink(name, dir_fd=parent_descriptor)
        return
    try:
        directory_descriptor = _open_child_directory(
            parent_descriptor,
            name,
            create=False,
        )
    except FileNotFoundError:
        return
    try:
        with os.scandir(directory_descriptor) as entries:
            child_names = tuple(entry.name for entry in entries)
        for child_name in child_names:
            status = os.stat(
                child_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISDIR(status.st_mode):
                _remove_directory_at(directory_descriptor, child_name)
            else:
                os.unlink(child_name, dir_fd=directory_descriptor)
    finally:
        os.close(directory_descriptor)
    os.rmdir(name, dir_fd=parent_descriptor)


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
