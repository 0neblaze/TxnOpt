from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import evrptw.stage052_review_service as review_service
from evrptw.artifacts import (
    ArtifactBundleWriter,
    ArtifactRunContext,
    ArtifactStorageConfig,
)
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
        service_execution_path=(
            "/usr/bin:/usr/lib/wsl/lib:/mnt/c/WINDOWS/System32"
        ),
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
    monkeypatch.setattr(review_service, "_systemd_memory_peaks", lambda _: (123, 0))

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
    assert receipt["service_execution_path"] == config.service_execution_path
    assert receipt["progress_log_sha256"]
    assert receipt["aggregate_peak_rss_bytes"] > 0
    assert "review complete" in (config.log_directory / "service.log").read_text()


def test_finalizer_binds_successful_receipt_to_current_review_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path / "fixture", limit_bytes=int(5.5 * 1024**3))
    run_dir = tmp_path / "stage05.2_hot_path_attempt99"
    control = run_dir / "control"
    review = run_dir / "review"
    control.mkdir(parents=True)
    review.mkdir()
    raw_manifest = control / f"{run_dir.name}_manifest.json"
    raw_manifest.write_text(
        json.dumps(
            {
                "run_label": run_dir.name,
                "component": "hot_path",
                "status": "complete",
                "evidence_completeness": "complete",
            }
        ),
        encoding="utf-8",
    )
    review_manifest = review / "review_manifest.json"
    review_manifest.write_text('{"status":"READY_FOR_STAGE052_ARTIFACT_STREAMING"}\n')
    config = replace(config, raw_manifest=raw_manifest, log_directory=tmp_path / "logs")
    config.log_directory.mkdir()
    receipt = review_service._initial_receipt(config)
    receipt.update(
        {
            "status": "completed",
            "stop_reason": "process_exit",
            "raw_manifest_sha256_before": hashlib.sha256(
                raw_manifest.read_bytes()
            ).hexdigest(),
        }
    )
    (config.log_directory / "review_execution.json").write_text(
        json.dumps(receipt),
        encoding="utf-8",
    )
    monkeypatch.setenv("SERVICE_RESULT", "success")
    monkeypatch.setattr(review_service, "_systemd_memory_peaks", lambda _unit: (123, 0))

    finalize_review_execution(config)

    bound = json.loads((review / "review_execution.json").read_text(encoding="utf-8"))
    assert bound["review_manifest_sha256"] == hashlib.sha256(
        review_manifest.read_bytes()
    ).hexdigest()
    assert bound["cgroup_memory_peak_status"] == "verified"


def test_service_execution_path_binds_wsl_interoperability_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locations = {
        "nvidia-smi": "/usr/lib/wsl/lib/nvidia-smi",
        "powershell.exe": "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/powershell.exe",
        "wsl.exe": "/mnt/c/WINDOWS/System32/wsl.exe",
    }
    monkeypatch.setattr(review_service.shutil, "which", locations.get)

    service_path = review_service._service_execution_path()

    assert service_path.split(os.pathsep) == [
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
        "/usr/lib/wsl/lib",
        "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0",
        "/mnt/c/WINDOWS/System32",
    ]


def test_service_execution_path_fails_closed_when_a_formal_tool_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(review_service.shutil, "which", lambda _: None)

    with pytest.raises(FileNotFoundError, match="nvidia-smi"):
        review_service._service_execution_path()


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
    assert (
        "--setenv=PATH=/usr/bin:/usr/lib/wsl/lib:/mnt/c/WINDOWS/System32" in command
    )
    assert any(item.startswith("--property=ExecStopPost=") for item in command)


def test_formal_launcher_rejects_noncanonical_internal_rss_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, limit_bytes=6 * 1024**3)
    observed: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        review_service.subprocess,
        "run",
        lambda command, **_kwargs: observed.append(tuple(command)),
    )

    with pytest.raises(RuntimeError, match="fixed at 5.5 GiB"):
        launch_review_service(config)

    assert observed == []


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
    monkeypatch.setattr(review_service, "_systemd_memory_peaks", lambda _: (0, 0))
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


def test_exec_stop_post_fails_when_cgroup_peaks_are_unavailable(
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

    def unavailable(_: str) -> tuple[int, int]:
        raise review_service.ReviewResourceAccountingError("MemoryPeak unavailable")

    monkeypatch.setattr(review_service, "_systemd_memory_peaks", unavailable)
    monkeypatch.setenv("SERVICE_RESULT", "oom-kill")

    finalize_review_execution(config)

    receipt = json.loads((config.log_directory / "review_execution.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["finalized"] is True
    assert receipt["stop_reason"] == "resource_accounting_unavailable"
    assert receipt["cgroup_memory_peak_status"] == "unavailable"
    assert "MemoryPeak unavailable" in receipt["cgroup_memory_peak_error"]


def test_formal_launcher_rejects_arbitrary_command_before_systemd(tmp_path: Path) -> None:
    config = _config(tmp_path, limit_bytes=int(5.5 * 1024**3))

    with pytest.raises(RuntimeError, match="isolated Stage 5.2 reviewer module"):
        launch_review_service(config)

    receipt = json.loads((config.log_directory / "review_execution.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["finalized"] is True
    assert receipt["stop_reason"] == "preflight_failed"


def test_formal_preflight_rejects_noncanonical_manifest_in_control_directory(
    tmp_path: Path,
) -> None:
    run_label = "stage05.2_hot_path_attempt98"
    raw_dir = tmp_path / run_label
    bundle = ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext("stage05.2", "hot_path", run_label),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v1"),
    ).finalize()
    decoy = raw_dir / "control" / "decoy_manifest.json"
    decoy.write_bytes(bundle.manifest_path.read_bytes())

    with pytest.raises(RuntimeError, match="canonical signed bundle manifest"):
        review_service._canonical_raw_directory(decoy)


def test_formal_launcher_accepts_isolated_campaign_reviewer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_label = "stage05.2_benchmark_attempt99"
    raw_dir = tmp_path / "raw" / run_label
    control = raw_dir / "control"
    control.mkdir(parents=True)
    raw_manifest = control / f"{run_label}_manifest.json"
    raw_manifest.write_text(
        json.dumps(
            {
                "run_label": run_label,
                "component": "benchmark",
                "status": "complete",
                "evidence_completeness": "complete",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (control / f"{run_label}_run_metadata.json").write_text(
        '{"source_snapshot":{"allowed_untracked_sha256":{}}}\n',
        encoding="utf-8",
    )
    prerequisite = tmp_path / "stage05.2_accelerator_pilot_attempt99"
    prerequisite.mkdir()
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    bks_path = tmp_path / "bks.csv"
    bks_path.write_text("instance\n", encoding="utf-8")
    storage_roots = tmp_path / "roots.toml"
    storage_roots.write_text("[roots]\n", encoding="utf-8")
    registry = tmp_path / "retention.csv"
    registry.write_text("schema_version\n", encoding="utf-8")
    wheel = tmp_path / "reviewer.whl"
    wheel.write_bytes(b"reviewer-wheel")
    log_root = tmp_path / "review-logs"
    log_directory = log_root / run_label / "20260723T000000Z"
    progress_log = log_directory / "progress.jsonl"
    command = (
        sys.executable,
        "-I",
        "-m",
        "evrptw.experiments.stage052_campaign_review",
        "--campaign-dir",
        str(raw_dir),
        "--benchmark-dir",
        str(benchmark_dir),
        "--bks-path",
        str(bks_path),
        "--scope",
        "pilot",
        "--prerequisite-dir",
        str(prerequisite),
        "--storage-roots",
        str(storage_roots),
        "--retention-registry",
        str(registry),
        "--progress-log",
        str(progress_log),
        "--max-aggregate-rss-gib",
        "5.5",
        "--review-execution-receipt",
        str(log_directory / "review_execution.json"),
    )
    config = ReviewServiceConfig(
        unit="stage052-review-campaign-test",
        run_label=run_label,
        command=command,
        reviewer_python=Path(sys.executable),
        reviewer_revision="1" * 40,
        working_directory=Path.cwd(),
        log_directory=log_directory,
        raw_manifest=raw_manifest,
        wheel_path=wheel,
        service_execution_path="/usr/bin:/usr/lib/wsl/lib:/mnt/c/WINDOWS/System32",
        progress_log=progress_log,
        max_aggregate_rss_bytes=int(5.5 * 1024**3),
    )
    install_root = tmp_path / "installed"
    module_path = install_root / "evrptw/experiments/stage052_campaign_review.py"
    monkeypatch.setattr(review_service, "DEFAULT_LOG_ROOT", log_root)
    monkeypatch.setattr(
        review_service,
        "ArtifactReader",
        lambda _raw: SimpleNamespace(result=SimpleNamespace(manifest_path=raw_manifest)),
    )
    monkeypatch.setattr(review_service, "_require_clean_repository", lambda *_args, **_kw: "2" * 40)
    monkeypatch.setattr(review_service, "_verify_wheel_provenance", lambda *_: wheel)
    monkeypatch.setattr(
        review_service,
        "_reviewer_install_identity",
        lambda _python, _module: {
            "module_path": str(module_path),
            "distribution_root": str(install_root),
            "direct_url": wheel.resolve().as_uri(),
        },
    )
    monkeypatch.setattr(
        review_service,
        "_verify_installed_distribution_matches_wheel",
        lambda *_: "3" * 64,
    )
    observed: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check
        observed.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(review_service.subprocess, "run", run)

    launch_review_service(config)

    assert observed and observed[0][0] == "systemd-run"
    receipt = json.loads((log_directory / "review_execution.json").read_text())
    assert receipt["reviewer_module_name"] == (
        "evrptw.experiments.stage052_campaign_review"
    )


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
    monkeypatch.setattr(
        review_service,
        "_verify_wheel_source_matches_revision",
        lambda *_: "4" * 64,
    )
    monkeypatch.setattr(
        review_service,
        "_verify_wheel_rebuild_matches_revision",
        lambda *_: "5" * 64,
    )
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


def test_seal_reviewer_wheel_rejects_python_from_another_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    package = source / "src" / "evrptw"
    package.mkdir(parents=True)
    (package / "reviewer.py").write_text("REVISION = 'current'\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q"), cwd=source, check=True)
    subprocess.run(("git", "add", "."), cwd=source, check=True)
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
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    wheel = tmp_path / "reviewer.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("evrptw/reviewer.py", b"REVISION = 'old'\n")
    monkeypatch.chdir(source)

    with pytest.raises(RuntimeError, match="does not match clean source"):
        review_service.seal_reviewer_wheel(wheel, revision)


def test_seal_reviewer_wheel_requires_tracked_tools_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    package = source / "src" / "evrptw"
    package.mkdir(parents=True)
    (package / "reviewer.py").write_text("REVISION = 'current'\n", encoding="utf-8")
    tools_package = source / "tools"
    tools_package.mkdir()
    (tools_package / "__init__.py").write_text("", encoding="utf-8")
    (tools_package / "publisher.py").write_text("READY = True\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q"), cwd=source, check=True)
    subprocess.run(("git", "add", "."), cwd=source, check=True)
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
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    wheel = tmp_path / "reviewer.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("evrptw/reviewer.py", b"REVISION = 'current'\n")
        archive.writestr("tools/__init__.py", b"")
    monkeypatch.chdir(source)

    with pytest.raises(RuntimeError, match="omits clean source module: tools/publisher.py"):
        review_service.seal_reviewer_wheel(wheel, revision)


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


@pytest.mark.skipif(os.name == "nt", reason="Windows venv launchers are not symlinks")
def test_reviewer_python_identity_preserves_venv_symlink(tmp_path: Path) -> None:
    executable = tmp_path / "runtime-python"
    executable.symlink_to(Path(sys.executable))

    observed = review_service._absolute_executable(executable)

    assert observed == executable.absolute()
    assert observed != executable.resolve()
