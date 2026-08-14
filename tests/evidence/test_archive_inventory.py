from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from txnopt_evidence.archive import (
    ArchiveIntegrityError,
    inventory_tree,
    open_inventory_entry,
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="archive source inventory requires Linux no-follow primitives",
)


def test_inventory_hashes_regular_files_and_reopens_exact_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "a.txt").write_bytes(b"a\n")
    (source / "nested" / "b.bin").write_bytes(b"\x00b")

    inventory = inventory_tree(source)

    assert inventory.schema_version == "txnopt-archive-inventory-v1"
    assert [entry.relative_path for entry in inventory.entries] == [
        "a.txt",
        "nested/b.bin",
    ]
    assert inventory.total_size == 4
    assert inventory.recovery_source == str(source)
    assert inventory.to_payload()["recovery_source"] == str(source)
    with open_inventory_entry(source, inventory.entries[1]) as handle:
        assert handle.read() == b"\x00b"


def test_inventory_rejects_symlinks_in_tree_or_ancestry(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    source = real_parent / "source"
    source.mkdir(parents=True)
    (source / "evidence.txt").write_text("evidence\n", encoding="utf-8")
    (source / "linked.txt").symlink_to(source / "evidence.txt")

    with pytest.raises(ArchiveIntegrityError, match="symlink"):
        inventory_tree(source)

    (source / "linked.txt").unlink()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ArchiveIntegrityError, match="symlink"):
        inventory_tree(linked_parent / "source")


def test_inventory_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    os.mkfifo(source / "fifo")

    with pytest.raises(ArchiveIntegrityError, match="regular files"):
        inventory_tree(source)


def test_reopen_rejects_source_mutation_after_inventory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    target = source / "evidence.bin"
    target.write_bytes(b"before")
    inventory = inventory_tree(source)
    target.write_bytes(b"after")

    with (
        pytest.raises(ArchiveIntegrityError, match="changed after inventory"),
        open_inventory_entry(source, inventory.entries[0]),
    ):
        pass


def test_removed_generic_archive_store_surface_is_absent() -> None:
    import txnopt_evidence.archive as archive

    assert not hasattr(archive, "ArchiveStore")
    assert not hasattr(archive, "LocalFilesystemArchiveStore")
