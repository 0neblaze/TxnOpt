"""Resource-bounded systemd supervision for Stage 5.2 reviewers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]

REVIEW_EXECUTION_SCHEMA_VERSION = "stage05.2-review-execution-v1"
DEFAULT_LOG_ROOT = Path("/home/oneblaze/stage052-review-logs")
DEFAULT_MAX_AGGREGATE_RSS_BYTES = int(5.5 * 1024**3)
SYSTEMD_MEMORY_HIGH = "5G"
SYSTEMD_MEMORY_MAX = "6G"
SYSTEMD_MEMORY_SWAP_MAX = "2G"
_UNIT_PATTERN = re.compile(r"^[A-Za-z0-9_.@-]+$")


class ReviewMemoryLimitExceeded(RuntimeError):
    """Raised in the reviewer main thread when its process tree crosses the limit."""


@dataclass(frozen=True, slots=True)
class ReviewProgressLog:
    """Append-only fsynced JSONL progress for reviewer and spawned workers."""

    path: Path

    def emit(self, event: str, **details: object) -> None:
        payload = {
            "schema_version": "stage05.2-review-progress-v1",
            "timestamp": _utc_now(),
            "pid": os.getpid(),
            "event": event,
            **details,
        }
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class ReviewProcessMemoryGuard:
    """Interrupt the reviewer when aggregate parent/descendant RSS crosses its limit."""

    def __init__(
        self,
        *,
        limit_bytes: int,
        progress: ReviewProgressLog,
        interval_seconds: float = 0.1,
    ) -> None:
        if limit_bytes <= 0:
            raise ValueError("review process RSS limit must be positive")
        if interval_seconds <= 0.0:
            raise ValueError("review process sample interval must be positive")
        self.limit_bytes = limit_bytes
        self.progress = progress
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self.peak_swap_bytes = 0
        self.exceeded = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._previous_handler: Any = None
        self._last_progress_at = 0.0

    def __enter__(self) -> ReviewProcessMemoryGuard:
        self._previous_handler = signal.getsignal(signal.SIGTERM)

        def raise_limit(_: int, __: object) -> None:
            raise ReviewMemoryLimitExceeded(
                f"reviewer aggregate RSS exceeded {self.limit_bytes} bytes"
            )

        signal.signal(signal.SIGTERM, raise_limit)
        self._thread = threading.Thread(
            target=self._sample,
            name="stage052-review-memory-guard",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4.0))
            if self._thread.is_alive():
                raise RuntimeError("review process memory guard did not stop")
        signal.signal(signal.SIGTERM, self._previous_handler)

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss, swap = _sample_process_tree(os.getpid())
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self.peak_swap_bytes = max(self.peak_swap_bytes, swap)
            now = time.monotonic()
            if now - self._last_progress_at >= 5.0:
                self.progress.emit(
                    "memory_sample",
                    aggregate_rss_bytes=rss,
                    aggregate_swap_bytes=swap,
                    aggregate_peak_rss_bytes=self.peak_rss_bytes,
                    aggregate_peak_swap_bytes=self.peak_swap_bytes,
                    limit_bytes=self.limit_bytes,
                )
                self._last_progress_at = now
            if rss >= self.limit_bytes:
                self.exceeded = True
                self.progress.emit(
                    "memory_limit_exceeded",
                    aggregate_rss_bytes=rss,
                    aggregate_swap_bytes=swap,
                    limit_bytes=self.limit_bytes,
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return
            self._stop.wait(self.interval_seconds)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class ReviewServiceConfig:
    """Complete immutable execution envelope for one reviewer service."""

    unit: str
    command: tuple[str, ...]
    working_directory: Path
    log_directory: Path
    raw_manifest: Path
    wheel_path: Path
    progress_log: Path | None = None
    max_aggregate_rss_bytes: int = DEFAULT_MAX_AGGREGATE_RSS_BYTES
    sample_interval_seconds: float = 0.1

    def __post_init__(self) -> None:
        if not _UNIT_PATTERN.fullmatch(self.unit):
            raise ValueError(f"invalid systemd unit name: {self.unit}")
        if not self.command or not all(self.command):
            raise ValueError("review service command must be non-empty")
        if self.max_aggregate_rss_bytes <= 0:
            raise ValueError("review service RSS limit must be positive")
        if self.sample_interval_seconds <= 0.0:
            raise ValueError("review service sample interval must be positive")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        for key in ("working_directory", "log_directory", "raw_manifest", "wheel_path"):
            payload[key] = str(payload[key])
        payload["progress_log"] = (
            str(self.progress_log) if self.progress_log is not None else None
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ReviewServiceConfig:
        command = payload.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("review service command is invalid")
        return cls(
            unit=str(payload["unit"]),
            command=tuple(command),
            working_directory=Path(str(payload["working_directory"])),
            log_directory=Path(str(payload["log_directory"])),
            raw_manifest=Path(str(payload["raw_manifest"])),
            wheel_path=Path(str(payload["wheel_path"])),
            progress_log=(
                Path(str(payload["progress_log"]))
                if payload.get("progress_log") is not None
                else None
            ),
            max_aggregate_rss_bytes=int(payload["max_aggregate_rss_bytes"]),
            sample_interval_seconds=float(payload["sample_interval_seconds"]),
        )


def _processes(root_pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(root_pid)
        return [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return []


def _sample_process_tree(root_pid: int) -> tuple[int, int]:
    aggregate_rss = 0
    aggregate_swap = 0
    for process in _processes(root_pid):
        try:
            with process.oneshot():
                aggregate_rss += int(process.memory_info().rss)
                full_info = process.memory_full_info()
                aggregate_swap += int(getattr(full_info, "swap", 0))
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return aggregate_rss, aggregate_swap


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5.0)


def _repository_revision(working_directory: Path) -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=working_directory,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def supervise_review(config: ReviewServiceConfig) -> int:
    """Run one reviewer command, enforce its process-tree limit, and seal a receipt."""

    config.log_directory.mkdir(parents=True, exist_ok=False)
    if not config.working_directory.is_dir():
        raise FileNotFoundError(config.working_directory)
    if not config.raw_manifest.is_file():
        raise FileNotFoundError(config.raw_manifest)
    if not config.wheel_path.is_file():
        raise FileNotFoundError(config.wheel_path)
    raw_before = _sha256(config.raw_manifest)
    started_at = _utc_now()
    started = time.monotonic()
    peak_rss = 0
    peak_swap = 0
    stop_reason = "process_exit"
    service_log = config.log_directory / "service.log"
    exit_code = 1
    with service_log.open("xb") as output:
        process = subprocess.Popen(
            config.command,
            cwd=config.working_directory,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            rss, swap = _sample_process_tree(process.pid)
            peak_rss = max(peak_rss, rss)
            peak_swap = max(peak_swap, swap)
            if rss >= config.max_aggregate_rss_bytes:
                stop_reason = "memory_limit_exceeded"
                _terminate_process_group(process)
                break
            time.sleep(config.sample_interval_seconds)
        observed_exit_code = process.poll()
        exit_code = process.wait() if observed_exit_code is None else observed_exit_code
        final_rss, final_swap = _sample_process_tree(process.pid)
        peak_rss = max(peak_rss, final_rss)
        peak_swap = max(peak_swap, final_swap)
        output.flush()
        os.fsync(output.fileno())
    raw_after = _sha256(config.raw_manifest)
    if raw_after != raw_before:
        stop_reason = "raw_manifest_changed"
        exit_code = 1
    status = "completed" if exit_code == 0 and stop_reason == "process_exit" else "failed"
    receipt: dict[str, object] = {
        "schema_version": REVIEW_EXECUTION_SCHEMA_VERSION,
        "unit": config.unit,
        "status": status,
        "stop_reason": stop_reason,
        "exit_code": exit_code,
        "command": list(config.command),
        "working_directory": str(config.working_directory.resolve()),
        "python_executable": sys.executable,
        "repository_revision": _repository_revision(config.working_directory),
        "wheel_path": str(config.wheel_path.resolve()),
        "wheel_sha256": _sha256(config.wheel_path),
        "raw_manifest_path": str(config.raw_manifest.resolve()),
        "raw_manifest_sha256_before": raw_before,
        "raw_manifest_sha256_after": raw_after,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "duration_seconds": time.monotonic() - started,
        "aggregate_peak_rss_bytes": peak_rss,
        "aggregate_peak_swap_bytes": peak_swap,
        "max_aggregate_rss_bytes": config.max_aggregate_rss_bytes,
        "systemd_memory_high": SYSTEMD_MEMORY_HIGH,
        "systemd_memory_max": SYSTEMD_MEMORY_MAX,
        "systemd_memory_swap_max": SYSTEMD_MEMORY_SWAP_MAX,
        "service_log_sha256": _sha256(service_log),
        "progress_log_path": (
            str(config.progress_log.resolve()) if config.progress_log is not None else None
        ),
        "progress_log_sha256": (
            _sha256(config.progress_log)
            if config.progress_log is not None and config.progress_log.is_file()
            else None
        ),
    }
    _atomic_json(config.log_directory / "review_execution.json", receipt)
    return int(exit_code)


def launch_review_service(config: ReviewServiceConfig) -> None:
    """Create one transient user service without inheriting the caller's lifetime."""

    config.log_directory.parent.mkdir(parents=True, exist_ok=True)
    staging_directory = config.log_directory.with_name(f".{config.log_directory.name}.launch")
    staging_directory.mkdir(exist_ok=False)
    temporary_root = staging_directory / "tmp"
    temporary_root.mkdir()
    config_path = staging_directory / "service_config.json"
    _atomic_json(config_path, config.to_dict())
    command = (
        "systemd-run",
        "--user",
        "--collect",
        f"--unit={config.unit}",
        "--property=MemoryAccounting=yes",
        f"--property=MemoryHigh={SYSTEMD_MEMORY_HIGH}",
        f"--property=MemoryMax={SYSTEMD_MEMORY_MAX}",
        f"--property=MemorySwapMax={SYSTEMD_MEMORY_SWAP_MAX}",
        "--property=KillMode=control-group",
        "--property=Restart=no",
        "--property=OOMPolicy=stop",
        f"--property=WorkingDirectory={config.working_directory.resolve()}",
        f"--setenv=STAGE052_REVIEW_TMPDIR={temporary_root.resolve()}",
        sys.executable,
        "-m",
        "evrptw.stage052_review_service",
        "supervise",
        "--config",
        str(config_path.resolve()),
    )
    subprocess.run(command, check=True)


def _parse_command(values: Sequence[str]) -> tuple[str, ...]:
    command = tuple(values[1:] if values and values[0] == "--" else values)
    if not command:
        raise ValueError("review command is required after --")
    return command


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 5.2 review as a bounded service")
    subparsers = parser.add_subparsers(dest="action", required=True)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--run-label", required=True)
    launch.add_argument("--working-directory", type=Path, required=True)
    launch.add_argument("--raw-manifest", type=Path, required=True)
    launch.add_argument("--wheel-path", type=Path, required=True)
    launch.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    launch.add_argument("--max-aggregate-rss-gib", type=float, default=5.5)
    launch.add_argument("command", nargs=argparse.REMAINDER)
    supervise = subparsers.add_parser("supervise")
    supervise.add_argument("--config", type=Path, required=True)
    for action in ("status", "follow", "stop"):
        command_parser = subparsers.add_parser(action)
        command_parser.add_argument("--unit", required=True)
    receipt = subparsers.add_parser("receipt")
    receipt.add_argument("--log-dir", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _build_parser().parse_args()
    if arguments.action == "launch":
        timestamp = _timestamp()
        safe_label = re.sub(r"[^A-Za-z0-9_.-]", "-", arguments.run_label)
        unit = f"stage052-review-{safe_label}-{timestamp}"
        log_directory = arguments.log_root / arguments.run_label / timestamp
        command = list(_parse_command(arguments.command))
        progress_log = log_directory / "progress.jsonl"
        command.extend(
            (
                "--progress-log",
                str(progress_log),
                "--max-aggregate-rss-gib",
                str(arguments.max_aggregate_rss_gib),
            )
        )
        config = ReviewServiceConfig(
            unit=unit,
            command=tuple(command),
            working_directory=arguments.working_directory,
            log_directory=log_directory,
            raw_manifest=arguments.raw_manifest,
            wheel_path=arguments.wheel_path,
            progress_log=progress_log,
            max_aggregate_rss_bytes=int(arguments.max_aggregate_rss_gib * 1024**3),
        )
        launch_review_service(config)
        print(json.dumps({"unit": unit, "log_directory": str(log_directory)}))
        return 0
    if arguments.action == "supervise":
        payload = json.loads(arguments.config.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("review service config must be an object")
        try:
            return supervise_review(ReviewServiceConfig.from_dict(payload))
        finally:
            shutil.rmtree(arguments.config.parent, ignore_errors=True)
    if arguments.action == "receipt":
        print((arguments.log_dir / "review_execution.json").read_text(encoding="utf-8"), end="")
        return 0
    systemctl_action = "status" if arguments.action == "status" else "stop"
    command = ["systemctl", "--user", systemctl_action, arguments.unit]
    if arguments.action == "follow":
        command = ["journalctl", "--user", "--follow", "--unit", arguments.unit]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
