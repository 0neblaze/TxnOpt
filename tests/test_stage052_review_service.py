from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import evrptw.stage052_review_service as review_service
from evrptw.stage052_review_service import (
    ReviewServiceConfig,
    launch_review_service,
    supervise_review,
)


def _config(tmp_path: Path, *, limit_bytes: int) -> ReviewServiceConfig:
    raw_manifest = tmp_path / "raw_manifest.json"
    raw_manifest.write_text('{"status":"complete"}\n', encoding="utf-8")
    wheel = tmp_path / "reviewer.whl"
    wheel.write_bytes(b"reviewer-wheel")
    progress_log = tmp_path / "logs" / "progress.jsonl"
    return ReviewServiceConfig(
        unit="stage052-review-test",
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
        working_directory=Path.cwd(),
        log_directory=tmp_path / "logs",
        raw_manifest=raw_manifest,
        wheel_path=wheel,
        progress_log=progress_log,
        max_aggregate_rss_bytes=limit_bytes,
        sample_interval_seconds=0.01,
    )


def test_review_supervisor_writes_receipt_and_preserves_raw_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path, limit_bytes=256 * 1024 * 1024)

    exit_code = supervise_review(config)

    assert exit_code == 0
    receipt = json.loads((config.log_directory / "review_execution.json").read_text())
    assert receipt["schema_version"] == "stage05.2-review-execution-v1"
    assert receipt["status"] == "completed"
    assert receipt["exit_code"] == 0
    assert receipt["stop_reason"] == "process_exit"
    assert receipt["raw_manifest_sha256_before"] == receipt["raw_manifest_sha256_after"]
    assert receipt["wheel_sha256"]
    assert receipt["progress_log_sha256"]
    assert receipt["aggregate_peak_rss_bytes"] > 0
    assert "review complete" in (config.log_directory / "service.log").read_text()


def test_review_supervisor_terminates_process_tree_at_memory_limit(tmp_path: Path) -> None:
    config = _config(tmp_path, limit_bytes=32 * 1024 * 1024)
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
