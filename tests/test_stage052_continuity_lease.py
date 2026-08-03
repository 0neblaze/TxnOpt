from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from evrptw.stage052_continuity_lease import acquire, inspect, release, renew


def _repository(path: Path) -> Path:
    path.mkdir()
    subprocess.run(("git", "-C", str(path), "init", "-q"), check=True)
    subprocess.run(
        ("git", "-C", str(path), "config", "user.email", "lease@example.invalid"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(path), "config", "user.name", "Lease test"),
        check=True,
    )
    marker = path / "marker.txt"
    marker.write_text("lease\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(path), "add", "marker.txt"), check=True)
    subprocess.run(("git", "-C", str(path), "commit", "-qm", "base"), check=True)
    return path


def test_continuity_lease_is_atomic_renewable_and_identity_bound(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "repo")
    lease = acquire(
        repository,
        owner="thread:test",
        phase="unit-test",
        label=None,
        ttl_seconds=30,
    )
    token = str(lease["token"])
    try:
        observed = inspect(repository)
        assert observed is not None
        assert observed["owner"] == "thread:test"
        assert observed["phase"] == "unit-test"
        assert observed["label"] is None
        assert observed["boot_id"]
        assert observed["process_start_time"]
        with pytest.raises(RuntimeError, match="already held"):
            acquire(
                repository,
                owner="thread:duplicate",
                phase="must-not-start",
                label="attempt04",
            )
        updated = renew(
            repository,
            token=token,
            phase="verified",
            label="diagnostic-only",
        )
        assert updated["phase"] == "verified"
        assert updated["label"] == "diagnostic-only"
        assert float(updated["renewed_at_epoch"]) > float(lease["renewed_at_epoch"])
    finally:
        if inspect(repository) is not None:
            release(repository, token=token)
    assert inspect(repository) is None
