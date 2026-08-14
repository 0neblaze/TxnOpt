from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from txnopt_evidence.cli import main


def _signed_json(path: Path, payload: object) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{hashlib.sha256(data).hexdigest()}  {path.name}\n",
        encoding="utf-8",
    )


def test_cli_legacy_verify_reads_only_signed_legacy_json(
    tmp_path: Path,
    capsys: object,
) -> None:
    manifest = tmp_path / "manifest.json"
    _signed_json(manifest, {"schema_version": "txnopt-test-v1"})
    assert main(["legacy", "verify", str(manifest)]) == 0


def test_cli_run_and_replay_fail_closed_without_fallback(
    tmp_path: Path,
    capsys: object,
) -> None:
    path = tmp_path / "config.json"
    path.write_text("{}\n", encoding="utf-8")
    assert main(["run", "--config", str(path)]) == 2
    assert main(["replay", str(path), "--output-dir", str(tmp_path / "review")]) == 2


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="local archive adapter is Linux-only"
)
def test_cli_archive_inventory_mirror_verify_and_restore(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.json").write_text('{"status":"SEALED"}\n', encoding="utf-8")
    store = tmp_path / "archive"

    assert main(["archive", "inventory", str(source)]) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert inventory["schema_version"] == "txnopt-archive-inventory-v1"
    assert inventory["file_count"] == 1

    assert main(
        [
            "archive",
            "mirror",
            str(source),
            "--store",
            str(store),
            "--commit-id",
            "cli-attempt-01",
        ]
    ) == 0
    mirror = json.loads(capsys.readouterr().out)
    commit = mirror["commit_ref"]

    ref_arguments = [
        "--store",
        str(store),
        "--commit-id",
        commit["commit_id"],
        "--commit-sha256",
        commit["sha256"],
        "--commit-size",
        str(commit["size"]),
    ]
    assert main(["archive", "verify", *ref_arguments]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["verified"] is True

    destination = tmp_path / "restored"
    assert main(
        ["archive", "restore", *ref_arguments, "--destination", str(destination)]
    ) == 0
    restored = json.loads(capsys.readouterr().out)
    assert restored["verified"] is True
    assert (destination / "evidence.json").read_bytes() == (
        source / "evidence.json"
    ).read_bytes()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="local archive adapter is Linux-only"
)
def test_cli_archive_rejects_relative_or_worktree_local_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.txt").write_text("evidence\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(
        [
            "archive",
            "mirror",
            str(source),
            "--store",
            "relative-archive",
            "--commit-id",
            "relative-store",
        ]
    ) == 2

    worktree = tmp_path / "worktree"
    (worktree / ".git").mkdir(parents=True)
    assert main(
        [
            "archive",
            "mirror",
            str(source),
            "--store",
            str(worktree / "archive"),
            "--commit-id",
            "worktree-store",
        ]
    ) == 2


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="local archive adapter is Linux-only"
)
def test_cli_archive_rejects_overlapping_source_and_store(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.txt").write_text("evidence\n", encoding="utf-8")

    assert main(
        [
            "archive",
            "mirror",
            str(source),
            "--store",
            str(source / "archive"),
            "--commit-id",
            "overlapping-store",
        ]
    ) == 2


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="local archive adapter is Linux-only"
)
def test_cli_archive_verify_missing_store_is_read_only(tmp_path: Path) -> None:
    store = tmp_path / "missing-archive"

    assert main(
        [
            "archive",
            "verify",
            "--store",
            str(store),
            "--commit-id",
            "missing",
            "--commit-sha256",
            "0" * 64,
            "--commit-size",
            "0",
        ]
    ) == 2

    assert not store.exists()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="local archive adapter is Linux-only"
)
def test_cli_archive_rejects_rebuildable_cache_root(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.txt").write_text("evidence\n", encoding="utf-8")
    cache_home = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_home))

    assert main(
        [
            "archive",
            "mirror",
            str(source),
            "--store",
            str(cache_home / "txnopt" / ("a" * 40) / "archive"),
            "--commit-id",
            "cache-store",
        ]
    ) == 2
