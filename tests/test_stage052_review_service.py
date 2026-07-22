from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

import evrptw.stage052_review_service as review_service
from evrptw.stage052_review_service import (
    ReviewServiceConfig,
    finalize_review_execution,
    launch_review_service,
    supervise_review,
)


def _config(tmp_path: Path, *, limit_bytes: int) -> ReviewServiceConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw_manifest = tmp_path / "raw_manifest.json"
    raw_manifest.write_text(
        '{"run_label":"stage05.2_hot_path_attempt99",'
        '"status":"complete","evidence_completeness":"complete"}\n',
        encoding="utf-8",
    )
    wheel = tmp_path / "reviewer.whl"
    wheel.write_bytes(b"reviewer-wheel")
    progress_log = tmp_path / "logs" / "progress.jsonl"
    return ReviewServiceConfig(
        unit="stage052-review-test",
        run_label="stage05.2_hot_path_attempt99",
        command=(
            sys.executable,
            "-c",
            (
                "import pathlib,sys; "
                "pathlib.Path(sys.argv[1]).write_text('progress\\n'); "
                "print('review complete')"
            ),
            str(progress_log),
        ),
        reviewer_python=Path(sys.executable),
        reviewer_revision="1" * 40,
        working_directory=Path.cwd(),
        log_directory=tmp_path / "logs",
        raw_manifest=raw_manifest,
        wheel_path=wheel,
        progress_log=progress_log,
        max_aggregate_rss_bytes=limit_bytes,
        sample_interval_seconds=0.01,
    )


def test_review_supervisor_writes_receipt_and_preserves_raw_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, limit_bytes=256 * 1024 * 1024)
    monkeypatch.setattr(
        review_service,
        "_validate_formal_execution_envelope",
        lambda _: {"producer_repository_revision": "2" * 40},
    )

    exit_code = supervise_review(config)
    monkeypatch.setenv("SERVICE_RESULT", "success")
    finalize_review_execution(config)

    assert exit_code == 0
    receipt = json.loads((config.log_directory / "review_execution.json").read_text())
    assert receipt["schema_version"] == "stage05.2-review-execution-v1"
    assert receipt["status"] == "completed"
    assert receipt["finalized"] is True
    assert receipt["exit_code"] == 0
    assert receipt["stop_reason"] == "process_exit"
    assert receipt["raw_manifest_sha256_before"] == receipt["raw_manifest_sha256_after"]
    assert receipt["wheel_sha256"]
    assert receipt["progress_log_sha256"]
    assert receipt["aggregate_peak_rss_bytes"] > 0
    assert "review complete" in (config.log_directory / "service.log").read_text()


def test_review_supervisor_terminates_process_tree_at_memory_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, limit_bytes=32 * 1024 * 1024)
    monkeypatch.setattr(
        review_service,
        "_validate_formal_execution_envelope",
        lambda _: {"producer_repository_revision": "2" * 40},
    )
    config = replace(
        config,
        command=(
            sys.executable,
            "-c",
            "import time; payload=bytearray(96*1024*1024); time.sleep(30)",
        ),
    )

    exit_code = supervise_review(config)

    assert exit_code != 0
    receipt = json.loads((config.log_directory / "review_execution.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["stop_reason"] == "memory_limit_exceeded"
    assert receipt["aggregate_peak_rss_bytes"] >= config.max_aggregate_rss_bytes
    assert receipt["raw_manifest_sha256_before"] == receipt["raw_manifest_sha256_after"]


def test_review_launcher_applies_fixed_systemd_process_tree_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, limit_bytes=int(5.5 * 1024**3))
    monkeypatch.setattr(
        review_service,
        "_validate_formal_execution_envelope",
        lambda _: {"producer_repository_revision": "2" * 40},
    )
    observed: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check
        observed.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(review_service.subprocess, "run", run)

    launch_review_service(config)

    assert len(observed) == 1
    command = observed[0]
    assert command[:3] == ("systemd-run", "--user", "--collect")
    assert "--property=MemoryHigh=5G" in command
    assert "--property=MemoryMax=6G" in command
    assert "--property=MemorySwapMax=2G" in command
    assert "--property=KillMode=control-group" in command
    assert "--property=Restart=no" in command
    assert "--property=OOMPolicy=stop" in command
    assert any(item.startswith("--property=ExecStopPost=") for item in command)


def test_exec_stop_post_finalizes_hard_oom_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, limit_bytes=int(5.5 * 1024**3))
    monkeypatch.setattr(
        review_service,
        "_validate_formal_execution_envelope",
        lambda _: {"producer_repository_revision": "2" * 40},
    )
    review_service._prepare_review_execution(config)
    monkeypatch.setattr(
        review_service,
        "_systemd_memory_peaks",
        lambda _: (6 * 1024**3, 512 * 1024**2),
    )
    monkeypatch.setenv("SERVICE_RESULT", "oom-kill")
    monkeypatch.setenv("EXIT_CODE", "killed")
    monkeypatch.setenv("EXIT_STATUS", "9")

    finalize_review_execution(config)

    receipt = json.loads((config.log_directory / "review_execution.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["finalized"] is True
    assert receipt["stop_reason"] == "systemd_memory_max"
    assert receipt["systemd_service_result"] == "oom-kill"
    assert receipt["systemd_exit_status"] == "9"
    assert receipt["raw_manifest_unchanged"] is True
    assert receipt["raw_manifest_sha256_after"] == receipt["raw_manifest_sha256_before"]
    assert receipt["aggregate_peak_rss_bytes"] == 6 * 1024**3
    assert receipt["aggregate_peak_swap_bytes"] == 512 * 1024**2
    assert receipt["duration_seconds"] is not None


def test_exec_stop_post_rehashes_raw_manifest_after_supervisor_is_killed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, limit_bytes=int(5.5 * 1024**3))
    monkeypatch.setattr(
        review_service,
        "_validate_formal_execution_envelope",
        lambda _: {"producer_repository_revision": "2" * 40},
    )
    monkeypatch.setattr(review_service, "_systemd_memory_peaks", lambda _: (None, None))
    review_service._prepare_review_execution(config)
    config.raw_manifest.write_text('{"tampered":true}\n', encoding="utf-8")
    monkeypatch.setenv("SERVICE_RESULT", "oom-kill")

    finalize_review_execution(config)

    receipt = json.loads((config.log_directory / "review_execution.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["finalized"] is True
    assert receipt["stop_reason"] == "raw_manifest_changed"
    assert receipt["raw_manifest_unchanged"] is False
    assert receipt["raw_manifest_sha256_after"] != receipt["raw_manifest_sha256_before"]


def test_formal_launcher_rejects_arbitrary_command_before_systemd(tmp_path: Path) -> None:
    config = _config(tmp_path, limit_bytes=int(5.5 * 1024**3))

    with pytest.raises(RuntimeError, match="isolated Stage 5.2 reviewer module"):
        launch_review_service(config)

    receipt = json.loads((config.log_directory / "review_execution.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["finalized"] is True
    assert receipt["stop_reason"] == "preflight_failed"


def test_reviewer_revision_is_bound_to_wheel_and_installed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "reviewer.whl"
    members = {
        "evrptw/reviewer.py": b"REVIEWER = True\n",
        "evrptw_reproduction-0.1.dist-info/METADATA": b"Name: evrptw-reproduction\n",
        "evrptw_reproduction-0.1.dist-info/RECORD": b"installed record differs\n",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    monkeypatch.setattr(review_service, "_require_clean_repository", lambda _: "1" * 40)
    provenance = review_service.seal_reviewer_wheel(wheel, "1" * 40)
    config = replace(_config(tmp_path / "config", limit_bytes=1024), wheel_path=wheel)

    assert review_service._verify_wheel_provenance(config, wheel) == provenance
    install_root = tmp_path / "installed"
    for name, content in members.items():
        if name.endswith(".dist-info/RECORD"):
            content = b"pip rewrote this file\n"
        destination = install_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    assert review_service._verify_installed_distribution_matches_wheel(wheel, install_root)

    with wheel.open("ab") as handle:
        handle.write(b"replacement")
    with pytest.raises(RuntimeError, match="provenance does not match"):
        review_service._verify_wheel_provenance(config, wheel)


def test_formal_source_allows_only_raw_bound_local_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    tracked = source / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q"), cwd=source, check=True)
    subprocess.run(("git", "add", "tracked.txt"), cwd=source, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=Stage052 Test",
            "-c",
            "user.email=stage052@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
        cwd=source,
        check=True,
    )
    local = source / "local.json"
    local.write_text('{"bound":true}\n', encoding="utf-8")
    digest = hashlib.sha256(local.read_bytes()).hexdigest()

    revision = review_service._require_clean_repository(
        source,
        allowed_untracked_sha256={"local.json": digest},
    )

    assert len(revision) == 40
    local.write_text('{"bound":false}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="unapproved local files"):
        review_service._require_clean_repository(
            source,
            allowed_untracked_sha256={"local.json": digest},
        )
