"""Archive inventory primitives without a storage-backend abstraction.

The archive wire format is provider-specific. This module only inventories a
local evidence tree and opens the exact regular files named by that inventory.
It deliberately contains no generic store port or local archive adapter.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from txnopt_evidence.codec import canonical_json_bytes


class ArchiveError(RuntimeError):
    """Base error for archive inventory operations."""


class ArchiveIntegrityError(ArchiveError):
    """Raised when a source tree changes or violates its declared identity."""


class ArchiveNotFoundError(ArchiveError):
    """Raised when an inventory source is absent."""


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


@dataclass(frozen=True, slots=True)
class ArchiveInventory:
    entries: tuple[ArchiveInventoryEntry, ...]
    tree_sha256: str
    total_size: int
    recovery_source: str
    schema_version: str = "txnopt-archive-inventory-v1"

    def __post_init__(self) -> None:
        _validate_sha256(self.tree_sha256)
        paths = tuple(entry.relative_path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("archive inventory paths must be uniquely sorted")
        if self.total_size != sum(entry.size for entry in self.entries):
            raise ValueError("archive inventory total size differs from its entries")
        if not self.recovery_source:
            raise ValueError("archive inventory recovery source cannot be empty")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tree_sha256": self.tree_sha256,
            "file_count": len(self.entries),
            "total_size": self.total_size,
            "recovery_source": self.recovery_source,
            "files": [
                {
                    "relative_path": entry.relative_path,
                    "sha256": entry.sha256,
                    "size": entry.size,
                }
                for entry in self.entries
            ],
        }


def inventory_tree(source: Path) -> ArchiveInventory:
    """Hash a Linux regular-file tree without following any symlink."""

    _require_linux_no_follow()
    root = _absolute_without_symlink_resolution(source)
    try:
        root_descriptor = _open_directory_path(root)
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
        recovery_source=str(root),
    )


@contextmanager
def open_inventory_entry(
    source: Path,
    entry: ArchiveInventoryEntry,
) -> Iterator[BinaryIO]:
    """Open one inventoried source file through no-follow descriptors."""

    _require_linux_no_follow()
    root_descriptor = _open_directory_path(
        _absolute_without_symlink_resolution(source)
    )
    descriptor: int | None = None
    try:
        descriptor = _open_relative_regular_file(
            root_descriptor,
            PurePosixPath(entry.relative_path).parts,
        )
        digest, size = _hash_descriptor(descriptor)
        if digest != entry.sha256 or size != entry.size:
            raise ArchiveIntegrityError(
                f"archive source changed after inventory: {entry.relative_path}"
            )
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            yield handle
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(root_descriptor)


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
            raise ArchiveIntegrityError(
                f"archive inventory cannot contain symlinks: {relative.as_posix()}"
            )
        if stat.S_ISDIR(status.st_mode):
            child_descriptor = _open_child_directory(directory_descriptor, name)
            try:
                _inventory_directory(child_descriptor, relative, entries)
            finally:
                os.close(child_descriptor)
            continue
        if not stat.S_ISREG(status.st_mode):
            raise ArchiveIntegrityError(
                "archive inventory accepts regular files only: "
                f"{relative.as_posix()}"
            )
        descriptor = _open_regular_file_at(directory_descriptor, name)
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


def _open_relative_regular_file(root_descriptor: int, parts: tuple[str, ...]) -> int:
    if not parts:
        raise ArchiveIntegrityError("archive relative file path is empty")
    parent_descriptor = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            child = _open_child_directory(parent_descriptor, component)
            os.close(parent_descriptor)
            parent_descriptor = child
        return _open_regular_file_at(parent_descriptor, parts[-1])
    finally:
        os.close(parent_descriptor)


def _open_directory_path(path: Path) -> int:
    absolute = _absolute_without_symlink_resolution(path)
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:]:
            child = _open_child_directory(descriptor, component)
            os.close(descriptor)
            descriptor = child
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _open_child_directory(parent_descriptor: int, name: str) -> int:
    if name in {"", ".", ".."} or "/" in name or "\\" in name:
        raise ArchiveIntegrityError("archive directory component is not canonical")
    try:
        return os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ArchiveIntegrityError(
                f"archive source ancestry contains a symlink: {name}"
            ) from error
        raise


def _open_regular_file_at(directory_descriptor: int, name: str) -> int:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ArchiveIntegrityError("archive source file cannot be a symlink") from error
        raise
    try:
        status = os.fstat(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    if not stat.S_ISREG(status.st_mode):
        os.close(descriptor)
        raise ArchiveIntegrityError("archive source object must be a regular file")
    return descriptor


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


def _require_linux_no_follow() -> None:
    if (
        os.name != "posix"
        or not sys.platform.startswith("linux")
        or not hasattr(os, "O_NOFOLLOW")
    ):
        raise ArchiveError(
            "archive inventory requires Linux directory descriptors and O_NOFOLLOW"
        )


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("SHA-256 identity must be 64 lowercase hexadecimal characters")


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


__all__ = [
    "ArchiveError",
    "ArchiveIntegrityError",
    "ArchiveInventory",
    "ArchiveInventoryEntry",
    "ArchiveNotFoundError",
    "inventory_tree",
    "open_inventory_entry",
]
