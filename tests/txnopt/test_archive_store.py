from __future__ import annotations

import hashlib
import io
import os
import shutil
import sys
from pathlib import Path

import pytest

from txnopt_evidence.archive import (
    ArchiveCommit,
    ArchiveConflictError,
    ArchiveEntry,
    ArchiveIntegrityError,
    ArchiveNotFoundError,
    ArchiveObjectRef,
    ArchiveStore,
    LocalFilesystemArchiveStore,
    inventory_tree,
    mirror_tree,
    restore_commit,
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the current local archive adapter requires Linux filesystem primitives",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_local_archive_put_open_and_head_round_trip(tmp_path) -> None:
    payload = b"immutable TxnOpt evidence\n"
    store = LocalFilesystemArchiveStore(tmp_path / "archive")

    ref = store.put_blob(
        io.BytesIO(payload),
        expected_sha256=_sha256(payload),
        expected_size=len(payload),
    )

    assert ref.sha256 == _sha256(payload)
    assert ref.size == len(payload)
    assert store.head(ref).verified is True
    with store.open(ref) as handle:
        assert handle.read() == payload


def test_local_archive_rejects_wrong_blob_digest_without_publication(tmp_path) -> None:
    payload = b"bytes that must not be published"
    store = LocalFilesystemArchiveStore(tmp_path / "archive")

    with pytest.raises(ArchiveIntegrityError):
        store.put_blob(
            io.BytesIO(payload),
            expected_sha256="0" * 64,
            expected_size=len(payload),
        )

    assert not any((tmp_path / "archive" / "blobs").rglob("*"))


def test_local_archive_publishes_and_verifies_commit_marker(tmp_path) -> None:
    store = LocalFilesystemArchiveStore(tmp_path / "archive")
    first = b"first immutable object"
    second = b"second immutable object"
    first_ref = store.put_blob(
        io.BytesIO(first), expected_sha256=_sha256(first), expected_size=len(first)
    )
    second_ref = store.put_blob(
        io.BytesIO(second), expected_sha256=_sha256(second), expected_size=len(second)
    )
    commit = ArchiveCommit(
        commit_id="attempt-01",
        entries=(
            ArchiveEntry(relative_path="raw/first.bin", ref=first_ref),
            ArchiveEntry(relative_path="raw/second.bin", ref=second_ref),
        ),
    )

    commit_ref = store.publish_commit(commit)
    receipt = store.verify_commit(commit_ref)

    assert receipt.verified is True
    assert receipt.commit == commit
    assert receipt.object_count == 2
    assert receipt.total_size == len(first) + len(second)


def test_local_archive_commit_publication_is_conflict_safe_and_idempotent(tmp_path) -> None:
    store = LocalFilesystemArchiveStore(tmp_path / "archive")
    payload = b"one object"
    ref = store.put_blob(
        io.BytesIO(payload), expected_sha256=_sha256(payload), expected_size=len(payload)
    )
    commit = ArchiveCommit(
        commit_id="attempt-02",
        entries=(ArchiveEntry(relative_path="object.bin", ref=ref),),
    )

    published = store.publish_commit(commit)

    with pytest.raises(ArchiveConflictError):
        store.publish_commit(commit)
    assert store.publish_commit(commit, expected_absent=False) == published


class _InterruptedStream:
    def __init__(self, prefix: bytes) -> None:
        self._prefix = prefix
        self._reads = 0

    def read(self, _size: int = -1) -> bytes:
        self._reads += 1
        if self._reads == 1:
            return self._prefix
        raise OSError("simulated source interruption")


def test_interrupted_blob_write_remains_invisible_and_can_be_retried(tmp_path) -> None:
    payload = b"complete payload"
    ref = ArchiveObjectRef(sha256=_sha256(payload), size=len(payload))
    store = LocalFilesystemArchiveStore(tmp_path / "archive")

    with pytest.raises(OSError, match="simulated source interruption"):
        store.put_blob(
            _InterruptedStream(payload[:4]),  # type: ignore[arg-type]
            expected_sha256=ref.sha256,
            expected_size=ref.size,
        )
    with pytest.raises(ArchiveNotFoundError):
        store.head(ref)

    assert store.put_blob(
        io.BytesIO(payload), expected_sha256=ref.sha256, expected_size=ref.size
    ) == ref


def test_commit_verification_detects_a_truncated_local_blob(tmp_path) -> None:
    payload = b"payload protected by a commit"
    store_root = tmp_path / "archive"
    store = LocalFilesystemArchiveStore(store_root)
    ref = store.put_blob(
        io.BytesIO(payload), expected_sha256=_sha256(payload), expected_size=len(payload)
    )
    commit_ref = store.publish_commit(
        ArchiveCommit(
            commit_id="attempt-truncated",
            entries=(ArchiveEntry(relative_path="payload.bin", ref=ref),),
        )
    )

    (store_root / "blobs" / "sha256" / ref.sha256[:2] / ref.sha256).write_bytes(payload[:-1])

    with pytest.raises(ArchiveIntegrityError):
        store.verify_commit(commit_ref)

    with pytest.raises(ArchiveIntegrityError):
        store.open(ref)


def test_local_archive_mirror_verify_and_restore_round_trip(tmp_path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "alpha.txt").write_bytes(b"alpha\n")
    (source / "nested" / "beta.bin").write_bytes(b"\x00\x01beta")
    store = LocalFilesystemArchiveStore(tmp_path / "archive")

    commit_ref = mirror_tree(source, store=store, commit_id="mirror-attempt-01")
    destination = tmp_path / "restored"
    receipt = restore_commit(store, commit_ref, destination=destination)

    assert receipt.verified is True
    assert receipt.object_count == 2
    assert _tree_bytes(destination) == _tree_bytes(source)


def test_restore_rejects_existing_destination_and_entry_traversal(tmp_path) -> None:
    payload = b"safe"
    ref = ArchiveObjectRef(sha256=_sha256(payload), size=len(payload))
    with pytest.raises(ValueError):
        ArchiveEntry(relative_path="../escape", ref=ref)

    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.txt").write_bytes(payload)
    store = LocalFilesystemArchiveStore(tmp_path / "archive")
    commit_ref = mirror_tree(source, store=store, commit_id="mirror-attempt-02")
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(ArchiveConflictError):
        restore_commit(store, commit_ref, destination=destination)


@pytest.mark.parametrize("component", [".staging", "blobs"])
def test_local_archive_rejects_symlinked_internal_write_directory(
    tmp_path,
    component: str,
) -> None:
    store_root = tmp_path / "archive"
    store = LocalFilesystemArchiveStore(store_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    internal = store_root / component
    if internal.is_dir():
        shutil.rmtree(internal)
    internal.symlink_to(outside, target_is_directory=True)
    payload = b"must stay inside the archive"

    with pytest.raises(ArchiveIntegrityError):
        store.put_blob(
            io.BytesIO(payload),
            expected_sha256=_sha256(payload),
            expected_size=len(payload),
        )

    assert not any(outside.rglob("*"))


def test_local_archive_rejects_symlinked_commit_directory(tmp_path) -> None:
    store_root = tmp_path / "archive"
    store = LocalFilesystemArchiveStore(store_root)
    payload = b"safe blob"
    ref = store.put_blob(
        io.BytesIO(payload), expected_sha256=_sha256(payload), expected_size=len(payload)
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    commits = store_root / "commits"
    if commits.is_dir():
        shutil.rmtree(commits)
    commits.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArchiveIntegrityError):
        store.publish_commit(
            ArchiveCommit(
                commit_id="symlinked-commit",
                entries=(ArchiveEntry(relative_path="safe.bin", ref=ref),),
            )
        )

    assert not any(outside.rglob("*"))


def test_local_archive_rejects_symlink_in_root_ancestry(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArchiveIntegrityError, match="symlink"):
        LocalFilesystemArchiveStore(linked_root / "archive")

    assert not (outside / "archive").exists()


def test_inventory_rejects_symlink_in_source_ancestry(tmp_path) -> None:
    real_parent = tmp_path / "real-parent"
    source = real_parent / "source"
    source.mkdir(parents=True)
    (source / "evidence.txt").write_text("evidence\n", encoding="utf-8")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ArchiveIntegrityError, match="symlink"):
        inventory_tree(linked_parent / "source")


def test_local_archive_rejects_fifo_instead_of_blob_without_blocking(tmp_path) -> None:
    payload = b"immutable payload"
    store_root = tmp_path / "archive"
    store = LocalFilesystemArchiveStore(store_root)
    ref = store.put_blob(
        io.BytesIO(payload), expected_sha256=_sha256(payload), expected_size=len(payload)
    )
    blob = store_root / "blobs" / "sha256" / ref.sha256[:2] / ref.sha256
    blob.unlink()
    os.mkfifo(blob)

    with pytest.raises(ArchiveIntegrityError, match="regular file"):
        store.open(ref)


def test_local_archive_rejects_symlinked_writer_lock(tmp_path) -> None:
    store_root = tmp_path / "archive"
    store = LocalFilesystemArchiveStore(store_root)
    lock = store_root / ".writer.lock"
    lock.unlink()
    outside = tmp_path / "outside-lock"
    outside.write_text("outside\n", encoding="utf-8")
    lock.symlink_to(outside)
    payload = b"must not be written"

    with pytest.raises(ArchiveIntegrityError, match="lock"):
        store.put_blob(
            io.BytesIO(payload),
            expected_sha256=_sha256(payload),
            expected_size=len(payload),
        )

    assert outside.read_text(encoding="utf-8") == "outside\n"


def test_local_archive_recovers_invisible_staging_left_by_a_crash(tmp_path) -> None:
    store_root = tmp_path / "archive"
    LocalFilesystemArchiveStore(store_root)
    orphan = store_root / ".staging" / "orphaned-partial-write"
    orphan.write_bytes(b"partial")

    LocalFilesystemArchiveStore(store_root)

    assert not orphan.exists()


def test_read_only_archive_open_does_not_create_or_clean_state(tmp_path) -> None:
    absent = tmp_path / "absent-archive"
    with pytest.raises(ArchiveNotFoundError):
        LocalFilesystemArchiveStore(absent, create=False)
    assert not absent.exists()

    store_root = tmp_path / "archive"
    LocalFilesystemArchiveStore(store_root)
    orphan = store_root / ".staging" / "retained-until-next-writer"
    orphan.write_bytes(b"partial")

    read_only = LocalFilesystemArchiveStore(store_root, create=False)

    assert orphan.exists()
    with pytest.raises(RuntimeError, match="read-only"):
        read_only.put_blob(
            io.BytesIO(b"x"),
            expected_sha256=_sha256(b"x"),
            expected_size=1,
        )


class _SourceMutatingStore:
    def __init__(self, delegate: LocalFilesystemArchiveStore, source: Path) -> None:
        self._delegate = delegate
        self._source = source
        self._mutated = False

    def put_blob(self, source, *, expected_sha256: str, expected_size: int):  # type: ignore[no-untyped-def]
        ref = self._delegate.put_blob(
            source, expected_sha256=expected_sha256, expected_size=expected_size
        )
        if not self._mutated:
            self._mutated = True
            (self._source / "late-file.txt").write_text("late\n", encoding="utf-8")
        return ref

    def open(self, ref):  # type: ignore[no-untyped-def]
        return self._delegate.open(ref)

    def head(self, ref):  # type: ignore[no-untyped-def]
        return self._delegate.head(ref)

    def publish_commit(self, commit, *, expected_absent: bool = True):  # type: ignore[no-untyped-def]
        return self._delegate.publish_commit(commit, expected_absent=expected_absent)

    def verify_commit(self, ref):  # type: ignore[no-untyped-def]
        return self._delegate.verify_commit(ref)


class _DestinationRacingStore:
    def __init__(self, delegate: LocalFilesystemArchiveStore, destination: Path) -> None:
        self._delegate = delegate
        self._destination = destination
        self._created = False

    def put_blob(self, source, *, expected_sha256: str, expected_size: int):  # type: ignore[no-untyped-def]
        return self._delegate.put_blob(
            source, expected_sha256=expected_sha256, expected_size=expected_size
        )

    def open(self, ref):  # type: ignore[no-untyped-def]
        if not self._created:
            self._created = True
            self._destination.mkdir()
        return self._delegate.open(ref)

    def head(self, ref):  # type: ignore[no-untyped-def]
        return self._delegate.head(ref)

    def publish_commit(self, commit, *, expected_absent: bool = True):  # type: ignore[no-untyped-def]
        return self._delegate.publish_commit(commit, expected_absent=expected_absent)

    def verify_commit(self, ref):  # type: ignore[no-untyped-def]
        return self._delegate.verify_commit(ref)


def test_mirror_rejects_source_tree_change_before_commit_publication(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "initial.txt").write_text("initial\n", encoding="utf-8")
    delegate = LocalFilesystemArchiveStore(tmp_path / "archive")
    store: ArchiveStore = _SourceMutatingStore(delegate, source)

    with pytest.raises(ArchiveIntegrityError, match="changed during mirroring"):
        mirror_tree(source, store=store, commit_id="mutating-source")


def test_mirror_api_rejects_local_store_nested_inside_source(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.txt").write_text("evidence\n", encoding="utf-8")
    store_root = source / "archive"
    store = LocalFilesystemArchiveStore(store_root)

    with pytest.raises(ValueError, match="must not overlap"):
        mirror_tree(source, store=store, commit_id="overlapping-api")

    assert not (store_root / "commits").exists()


def test_restore_rejects_symlink_in_destination_ancestry(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.txt").write_text("safe\n", encoding="utf-8")
    store = LocalFilesystemArchiveStore(tmp_path / "archive")
    commit_ref = mirror_tree(source, store=store, commit_id="restore-symlink")
    outside = tmp_path / "outside"
    (outside / "nested").mkdir(parents=True)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArchiveIntegrityError, match="symlink"):
        restore_commit(
            store,
            commit_ref,
            destination=linked_parent / "nested" / "restored",
        )

    assert not (outside / "nested" / "restored").exists()


def test_restore_rejects_destination_inside_archive_store(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.txt").write_text("safe\n", encoding="utf-8")
    store_root = tmp_path / "archive"
    store = LocalFilesystemArchiveStore(store_root)
    commit_ref = mirror_tree(source, store=store, commit_id="restore-inside-store")

    with pytest.raises(ValueError, match="inside the local archive"):
        restore_commit(
            store,
            commit_ref,
            destination=store_root / "restored",
        )

    assert not (store_root / "restored").exists()


def test_restore_atomically_rejects_destination_created_during_copy(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.txt").write_text("safe\n", encoding="utf-8")
    delegate = LocalFilesystemArchiveStore(tmp_path / "archive")
    commit_ref = mirror_tree(source, store=delegate, commit_id="restore-race")
    destination = tmp_path / "racing-destination"
    store: ArchiveStore = _DestinationRacingStore(delegate, destination)

    with pytest.raises(ArchiveConflictError, match="appeared during restore"):
        restore_commit(store, commit_ref, destination=destination)

    assert destination.is_dir()
    assert not any(destination.iterdir())


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
