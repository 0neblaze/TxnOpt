from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from txnopt_evidence.archive import (
    ArchiveCommit,
    ArchiveConflictError,
    ArchiveEntry,
    ArchiveIntegrityError,
    ArchiveNotFoundError,
    ArchiveObjectRef,
    LocalFilesystemArchiveStore,
    mirror_tree,
    restore_commit,
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


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
