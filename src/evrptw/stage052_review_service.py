"""Resource-bounded systemd supervision for Stage 5.2 reviewers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import psutil  # type: ignore[import-untyped]

from evrptw.artifacts import ArtifactIntegrityError, ArtifactReader
from evrptw.stage052_resources import (
    ReviewMemoryContract,
    load_review_memory_contract,
)

REVIEW_EXECUTION_SCHEMA_VERSION = "stage05.2-review-execution-v2"
REVIEWER_WHEEL_PROVENANCE_SCHEMA_VERSION = "stage05.2-reviewer-wheel-provenance-v1"
DEFAULT_LOG_ROOT = (
    Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    / "reproducible-evrptw"
    / "stage052-review-logs"
)
DEFAULT_MAX_AGGREGATE_RSS_BYTES = int(5.5 * 1024**3)
DEFAULT_SYSTEMD_MEMORY_HIGH_BYTES = 5 * 1024**3
DEFAULT_SYSTEMD_MEMORY_MAX_BYTES = 6 * 1024**3
DEFAULT_SYSTEMD_MEMORY_SWAP_MAX_BYTES = 0
_UNIT_PATTERN = re.compile(r"^[A-Za-z0-9_.@-]+$")
_SERVICE_BASE_PATHS = (
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
)
_FORMAL_SERVICE_EXECUTABLES = ("nvidia-smi", "powershell.exe", "wsl.exe")
_PERFORMANCE_REVIEW_MODULE = "evrptw.experiments.stage052_performance_review"
_CAMPAIGN_REVIEW_MODULE = "evrptw.experiments.stage052_campaign_review"
_ALLOWED_REVIEW_MODULES = frozenset({_PERFORMANCE_REVIEW_MODULE, _CAMPAIGN_REVIEW_MODULE})
_REVIEW_EXECUTION_OPTION = "--review-execution-receipt"


class ReviewMemoryLimitExceeded(RuntimeError):
    """Raised in the reviewer main thread when its process tree crosses the limit."""


class ReviewServiceInterrupted(RuntimeError):
    """Raised when systemd or an operator interrupts the receipt supervisor."""


class ReviewResourceAccountingError(RuntimeError):
    """Raised when systemd cannot provide authoritative cgroup peak memory."""


@dataclass(frozen=True, slots=True)
class ReviewProgressLog:
    """Append-only fsynced JSONL progress for reviewer and spawned workers."""

    path: Path

    def emit(self, event: str, **details: object) -> None:
        payload = {
            "schema_version": "stage05.2-review-progress-v2",
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


def _absolute_executable(path: Path) -> Path:
    """Return an absolute executable path without dereferencing a venv symlink."""

    absolute = Path(os.path.abspath(path))
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    return absolute


def _service_execution_path() -> str:
    """Freeze the Linux and WSL interoperability tools needed by formal replay."""

    directories = list(_SERVICE_BASE_PATHS)
    for executable in _FORMAL_SERVICE_EXECUTABLES:
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(
                f"formal reviewer service executable is unavailable: {executable}"
            )
        parent = str(Path(resolved).absolute().parent)
        if parent not in directories:
            directories.append(parent)
    return os.pathsep.join(directories)


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
    run_label: str
    command: tuple[str, ...]
    reviewer_python: Path
    reviewer_revision: str
    working_directory: Path
    log_directory: Path
    raw_manifest: Path
    wheel_path: Path
    service_execution_path: str
    producer_source_directory: Path | None = None
    progress_log: Path | None = None
    review_memory_contract: Mapping[str, object] | None = None
    max_aggregate_rss_bytes: int = DEFAULT_MAX_AGGREGATE_RSS_BYTES
    systemd_memory_high_bytes: int = DEFAULT_SYSTEMD_MEMORY_HIGH_BYTES
    systemd_memory_max_bytes: int = DEFAULT_SYSTEMD_MEMORY_MAX_BYTES
    systemd_memory_swap_max_bytes: int = DEFAULT_SYSTEMD_MEMORY_SWAP_MAX_BYTES
    sample_interval_seconds: float = 0.1

    def __post_init__(self) -> None:
        if not _UNIT_PATTERN.fullmatch(self.unit):
            raise ValueError(f"invalid systemd unit name: {self.unit}")
        if not self.run_label or Path(self.run_label).name != self.run_label:
            raise ValueError(f"invalid review run label: {self.run_label}")
        if not self.command or not all(self.command):
            raise ValueError("review service command must be non-empty")
        if not re.fullmatch(r"[0-9a-f]{40}", self.reviewer_revision):
            raise ValueError("reviewer revision must be a full Git SHA-1")
        if not self.service_execution_path:
            raise ValueError("review service execution PATH must be non-empty")
        if self.max_aggregate_rss_bytes <= 0:
            raise ValueError("review service RSS limit must be positive")
        if not (
            0
            < self.systemd_memory_high_bytes
            < self.max_aggregate_rss_bytes
            < self.systemd_memory_max_bytes
        ):
            raise ValueError(
                "review service memory limits must satisfy High < guard < Max"
            )
        if self.systemd_memory_swap_max_bytes != 0:
            raise ValueError("review service swap must be disabled")
        if self.review_memory_contract is not None:
            contract = ReviewMemoryContract.from_dict(self.review_memory_contract)
            if (
                contract.memory_high_bytes != self.systemd_memory_high_bytes
                or contract.process_guard_bytes != self.max_aggregate_rss_bytes
                or contract.memory_max_bytes != self.systemd_memory_max_bytes
                or contract.memory_swap_max_bytes
                != self.systemd_memory_swap_max_bytes
            ):
                raise ValueError(
                    "review service limits differ from the Pilot memory contract"
                )
        if self.sample_interval_seconds <= 0.0:
            raise ValueError("review service sample interval must be positive")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        for key in (
            "reviewer_python",
            "working_directory",
            "log_directory",
            "raw_manifest",
            "wheel_path",
        ):
            payload[key] = str(payload[key])
        payload["producer_source_directory"] = (
            str(self.producer_source_directory)
            if self.producer_source_directory is not None
            else None
        )
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
            run_label=str(payload["run_label"]),
            command=tuple(command),
            reviewer_python=Path(str(payload["reviewer_python"])),
            reviewer_revision=str(payload["reviewer_revision"]),
            working_directory=Path(str(payload["working_directory"])),
            log_directory=Path(str(payload["log_directory"])),
            raw_manifest=Path(str(payload["raw_manifest"])),
            wheel_path=Path(str(payload["wheel_path"])),
            service_execution_path=str(payload["service_execution_path"]),
            producer_source_directory=(
                Path(str(payload["producer_source_directory"]))
                if payload.get("producer_source_directory") is not None
                else None
            ),
            progress_log=(
                Path(str(payload["progress_log"]))
                if payload.get("progress_log") is not None
                else None
            ),
            review_memory_contract=(
                dict(payload["review_memory_contract"])
                if isinstance(payload.get("review_memory_contract"), Mapping)
                else None
            ),
            max_aggregate_rss_bytes=int(payload["max_aggregate_rss_bytes"]),
            systemd_memory_high_bytes=int(
                payload.get(
                    "systemd_memory_high_bytes",
                    DEFAULT_SYSTEMD_MEMORY_HIGH_BYTES,
                )
            ),
            systemd_memory_max_bytes=int(
                payload.get(
                    "systemd_memory_max_bytes",
                    DEFAULT_SYSTEMD_MEMORY_MAX_BYTES,
                )
            ),
            systemd_memory_swap_max_bytes=int(
                payload.get(
                    "systemd_memory_swap_max_bytes",
                    DEFAULT_SYSTEMD_MEMORY_SWAP_MAX_BYTES,
                )
            ),
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
    if os.name == "nt":
        processes = list(reversed(_processes(process.pid)))
        for member in processes:
            try:
                member.terminate()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        _, alive = psutil.wait_procs(processes, timeout=5.0)
        for member in alive:
            try:
                member.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        process.wait(timeout=5.0)
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


def _require_clean_repository(
    working_directory: Path,
    *,
    allowed_untracked_sha256: dict[str, str] | None = None,
) -> str:
    revision = _repository_revision(working_directory)
    status = subprocess.run(
        ("git", "status", "--porcelain"),
        cwd=working_directory,
        check=True,
        capture_output=True,
        text=True,
    )
    allowed = allowed_untracked_sha256 or {}
    for line in status.stdout.splitlines():
        if not line.startswith("?? "):
            raise RuntimeError("formal reviewer working directory has tracked changes")
        relative = line[3:]
        expected_digest = allowed.get(relative)
        candidate = working_directory / relative
        if (
            expected_digest is None
            or not candidate.is_file()
            or _sha256(candidate) != expected_digest
        ):
            raise RuntimeError("formal reviewer working directory has unapproved local files")
    return revision


def _option_values(command: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(command):
        if item == option:
            if index + 1 >= len(command):
                raise ValueError(f"review command option has no value: {option}")
            values.append(command[index + 1])
        elif item.startswith(f"{option}="):
            values.append(item.split("=", 1)[1])
    return values


def _reviewer_install_identity(reviewer_python: Path, module_name: str) -> dict[str, str]:
    script = """
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse

reviewer = importlib.import_module(sys.argv[1])

distribution = metadata.distribution("reproducible-evrptw")
direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
print(json.dumps({
    "module_path": str(Path(reviewer.__file__).resolve()),
    "distribution_root": str(Path(distribution.locate_file(".")).resolve()),
    "direct_url": str(direct_url.get("url", "")),
}, sort_keys=True))
"""
    result = subprocess.run(
        (str(reviewer_python), "-I", "-c", script, module_name),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict) or any(
        not isinstance(payload.get(key), str)
        for key in ("module_path", "distribution_root", "direct_url")
    ):
        raise RuntimeError("reviewer installation identity is invalid")
    return {key: str(payload[key]) for key in payload}


def _wheel_provenance_path(wheel_path: Path) -> Path:
    return wheel_path.with_suffix(f"{wheel_path.suffix}.reviewer-provenance.json")


def _verify_wheel_source_matches_revision(wheel_path: Path, repository: Path) -> str:
    """Bind tracked Python, native, and build inputs to the clean revision."""

    result = subprocess.run(
        (
            "git",
            "ls-files",
            "-z",
            "--",
            "src/evrptw",
            "tools",
            "cpp",
            "CMakeLists.txt",
            "pyproject.toml",
        ),
        cwd=repository,
        check=True,
        capture_output=True,
    )
    relative_sources = tuple(
        Path(raw.decode("utf-8"))
        for raw in result.stdout.split(b"\0")
        if raw
    )
    if not relative_sources:
        raise RuntimeError("clean reviewer source contains no tracked Python modules")
    aggregate = hashlib.sha256(b"stage05.2-reviewer-source-v1\0")
    with zipfile.ZipFile(wheel_path) as archive:
        members = set(archive.namelist())
        for relative in sorted(relative_sources, key=lambda path: path.as_posix()):
            source = (repository / relative).read_bytes()
            member: str | None = None
            if relative.suffix in {".py", ".pyi"}:
                if relative.parts[:2] == ("src", "evrptw"):
                    member = Path(*relative.parts[1:]).as_posix()
                elif relative.parts[:1] == ("tools",):
                    member = relative.as_posix()
            if member is not None:
                if member not in members:
                    raise RuntimeError(f"reviewer wheel omits clean source module: {relative}")
                if archive.read(member) != source:
                    raise RuntimeError(f"reviewer wheel does not match clean source: {relative}")
            aggregate.update(relative.as_posix().encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(hashlib.sha256(source).digest())
    return aggregate.hexdigest()


def _wheel_member_digests(wheel_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel_path) as archive:
        return {
            member: hashlib.sha256(archive.read(member)).hexdigest()
            for member in sorted(archive.namelist())
            if not member.endswith("/") and not member.endswith(".dist-info/RECORD")
        }


def _verify_wheel_rebuild_matches_revision(wheel_path: Path, repository: Path) -> str:
    """Rebuild the clean revision and require every wheel member, including native, to match."""

    temporary_parent = Path("/tmp") if Path("/tmp").is_dir() else None
    with tempfile.TemporaryDirectory(
        prefix="stage052-reviewer-rebuild-",
        dir=temporary_parent,
    ) as directory:
        wheel_directory = Path(directory)
        subprocess.run(
            (
                str(_absolute_executable(Path(sys.executable))),
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--no-cache-dir",
                "--wheel-dir",
                str(wheel_directory),
                str(repository.resolve()),
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        rebuilt = tuple(wheel_directory.glob("*.whl"))
        if len(rebuilt) != 1:
            raise RuntimeError("clean reviewer rebuild did not produce exactly one wheel")
        declared = _wheel_member_digests(wheel_path)
        observed = _wheel_member_digests(rebuilt[0])
        if declared != observed:
            raise RuntimeError("reviewer wheel does not match clean native/source rebuild")
        return hashlib.sha256(
            json.dumps(declared, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def seal_reviewer_wheel(wheel_path: Path, reviewer_revision: str) -> Path:
    """Bind a clean reviewer source revision to one immutable wheel digest."""

    if not re.fullmatch(r"[0-9a-f]{40}", reviewer_revision):
        raise ValueError("reviewer revision must be a full Git SHA-1")
    observed_revision = _require_clean_repository(Path.cwd())
    if observed_revision != reviewer_revision:
        raise RuntimeError("reviewer revision does not match the clean build source")
    wheel = wheel_path.resolve(strict=True)
    source_digest = _verify_wheel_source_matches_revision(wheel, Path.cwd())
    rebuild_digest = _verify_wheel_rebuild_matches_revision(wheel, Path.cwd())
    provenance_path = _wheel_provenance_path(wheel)
    _atomic_json(
        provenance_path,
        {
            "schema_version": REVIEWER_WHEEL_PROVENANCE_SCHEMA_VERSION,
            "reviewer_revision": reviewer_revision,
            "wheel_filename": wheel.name,
            "wheel_sha256": _sha256(wheel),
            "reviewer_source_sha256": source_digest,
            "reviewer_rebuild_sha256": rebuild_digest,
            "sealed_at": _utc_now(),
        },
    )
    return provenance_path


def _verify_wheel_provenance(config: ReviewServiceConfig, wheel_path: Path) -> Path:
    provenance_path = _wheel_provenance_path(wheel_path)
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("reviewer wheel provenance must be a JSON object")
    expected = {
        "schema_version": REVIEWER_WHEEL_PROVENANCE_SCHEMA_VERSION,
        "reviewer_revision": config.reviewer_revision,
        "wheel_filename": wheel_path.name,
        "wheel_sha256": _sha256(wheel_path),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("reviewer wheel provenance does not match revision and wheel")
    source_digest = payload.get("reviewer_source_sha256")
    if (
        not isinstance(source_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None
    ):
        raise RuntimeError("reviewer wheel provenance source attestation is invalid")
    rebuild_digest = payload.get("reviewer_rebuild_sha256")
    if (
        not isinstance(rebuild_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", rebuild_digest) is None
    ):
        raise RuntimeError("reviewer wheel provenance rebuild attestation is invalid")
    return provenance_path


def _verify_installed_distribution_matches_wheel(
    wheel_path: Path,
    distribution_root: Path,
) -> str:
    aggregate = hashlib.sha256()
    with zipfile.ZipFile(wheel_path) as archive:
        members = sorted(
            member
            for member in archive.namelist()
            if not member.endswith("/") and not member.endswith(".dist-info/RECORD")
        )
        for member in members:
            installed = distribution_root / member
            if not installed.is_file():
                raise RuntimeError(f"installed reviewer wheel member is missing: {member}")
            wheel_digest = hashlib.sha256(archive.read(member)).hexdigest()
            if _sha256(installed) != wheel_digest:
                raise RuntimeError(f"installed reviewer wheel member differs: {member}")
            aggregate.update(member.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(wheel_digest.encode("ascii"))
            aggregate.update(b"\n")
    return aggregate.hexdigest()


def _canonical_raw_directory(raw_manifest: Path) -> Path:
    raw_directory = raw_manifest.resolve(strict=True).parent.parent
    try:
        canonical_manifest = ArtifactReader(raw_directory).result.manifest_path.resolve()
    except ArtifactIntegrityError as error:
        raise RuntimeError("raw manifest is not the canonical signed bundle manifest") from error
    if raw_manifest.resolve() != canonical_manifest:
        raise RuntimeError("raw manifest is not the canonical signed bundle manifest")
    return raw_directory


def _validate_formal_execution_envelope(config: ReviewServiceConfig) -> dict[str, str]:
    reviewer_python = _absolute_executable(config.reviewer_python)
    if _absolute_executable(Path(sys.executable)) != reviewer_python:
        raise RuntimeError("service supervisor is not running from reviewer_python")
    command_python = _absolute_executable(Path(config.command[0]))
    if command_python != reviewer_python:
        raise RuntimeError("review command does not use the frozen reviewer Python")
    expected_prefix = (str(config.command[0]), "-I", "-m")
    if config.command[:3] != expected_prefix or len(config.command) < 4:
        raise RuntimeError("review command must invoke the isolated Stage 5.2 reviewer module")
    module_name = config.command[3]
    if module_name not in _ALLOWED_REVIEW_MODULES:
        raise RuntimeError("review command must invoke the isolated Stage 5.2 reviewer module")
    if not config.log_directory.resolve().is_relative_to(DEFAULT_LOG_ROOT.resolve()):
        raise RuntimeError("formal review logs must use the fixed Stage 5.2 log root")

    raw_directory = _canonical_raw_directory(config.raw_manifest)
    scope_values = _option_values(config.command, "--scope")
    if len(scope_values) != 1:
        raise RuntimeError("formal review command requires one scope")
    if module_name == _PERFORMANCE_REVIEW_MODULE:
        raw_values = _option_values(config.command, "--raw-dir")
        if len(raw_values) != 1 or Path(raw_values[0]).resolve(strict=True) != raw_directory:
            raise RuntimeError("review command raw directory does not match raw manifest")
        component_values = _option_values(config.command, "--component")
        if len(component_values) != 1:
            raise RuntimeError("performance review command requires one component")
        component = component_values[0]
        comparison_values = _option_values(config.command, "--comparison-dir")
        if component in {"hot_path", "job_parallel", "native_kernels"} and not comparison_values:
            raise RuntimeError("performance review command requires a comparison directory")
        for value in comparison_values:
            Path(value).resolve(strict=True)
        prerequisite_values = _option_values(config.command, "--prerequisite")
        if component != "perf_baseline" and not prerequisite_values:
            raise RuntimeError("performance review command requires prerequisite identity")
        for value in prerequisite_values:
            role, separator, raw_path = value.partition("=")
            if not role or not separator or not raw_path:
                raise RuntimeError("formal review prerequisite identity is invalid")
            Path(raw_path).resolve(strict=True)
    else:
        campaign_values = _option_values(config.command, "--campaign-dir")
        if (
            len(campaign_values) != 1
            or Path(campaign_values[0]).resolve(strict=True) != raw_directory
        ):
            raise RuntimeError("campaign review directory does not match raw manifest")
        if _option_values(config.command, "--comparison-dir"):
            raise RuntimeError("campaign review command cannot accept comparison directories")
        prerequisite_values = _option_values(config.command, "--prerequisite-dir")
        if len(prerequisite_values) != 1:
            raise RuntimeError("campaign review command requires one prerequisite directory")
        Path(prerequisite_values[0]).resolve(strict=True)
        component = "benchmark"
    progress_values = _option_values(config.command, "--progress-log")
    if config.progress_log is None or len(progress_values) != 1:
        raise RuntimeError("formal review command requires exactly one progress log")
    if Path(progress_values[0]).resolve() != config.progress_log.resolve():
        raise RuntimeError("review command progress log does not match service config")
    limit_values = _option_values(config.command, "--max-aggregate-rss-gib")
    expected_limit = config.max_aggregate_rss_bytes / 1024**3
    if len(limit_values) != 1 or float(limit_values[0]) != expected_limit:
        raise RuntimeError("review command RSS limit does not match service config")
    receipt_values = _option_values(config.command, _REVIEW_EXECUTION_OPTION)
    expected_receipt = config.log_directory.resolve() / "review_execution.json"
    if len(receipt_values) != 1 or Path(receipt_values[0]).resolve() != expected_receipt:
        raise RuntimeError("review command execution receipt does not match service config")

    manifest = json.loads(config.raw_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("raw manifest must be a JSON object")
    if manifest.get("run_label") != config.run_label:
        raise RuntimeError("raw manifest run label does not match service config")
    if manifest.get("component") != component:
        raise RuntimeError("raw manifest component does not match review command")
    if manifest.get("status") != "complete" or manifest.get("evidence_completeness") != "complete":
        raise RuntimeError("formal reviewer requires complete raw evidence")

    metadata_path = raw_directory / "control" / f"{config.run_label}_run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_snapshot = metadata.get("source_snapshot") if isinstance(metadata, dict) else None
    allowed_payload = (
        source_snapshot.get("allowed_untracked_sha256")
        if isinstance(source_snapshot, dict)
        else None
    )
    allowed_untracked = (
        {
            str(path): str(digest)
            for path, digest in allowed_payload.items()
            if isinstance(path, str) and isinstance(digest, str)
        }
        if isinstance(allowed_payload, dict)
        else {}
    )
    reviewer_revision = _require_clean_repository(
        config.working_directory,
    )
    if (
        config.producer_source_directory is not None
        and reviewer_revision != config.reviewer_revision
    ):
        raise RuntimeError("reviewer working directory revision mismatch")
    producer_source = (
        config.working_directory
        if config.producer_source_directory is None
        else config.producer_source_directory.resolve()
    )
    producer_revision = _require_clean_repository(
        producer_source,
        allowed_untracked_sha256=allowed_untracked,
    )
    if config.producer_source_directory is not None:
        from evrptw.stage052_evidence import (
            stage052_source_snapshot_contract,
            verify_stage052_source_snapshot,
        )

        if (
            not isinstance(source_snapshot, dict)
            or stage052_source_snapshot_contract(
                verify_stage052_source_snapshot(producer_source)
            )
            != stage052_source_snapshot_contract(source_snapshot)
        ):
            raise RuntimeError(
                "producer source directory does not match the sealed raw source snapshot"
            )
    wheel_path = config.wheel_path.resolve(strict=True)
    provenance_path = _verify_wheel_provenance(config, wheel_path)
    install = _reviewer_install_identity(reviewer_python, module_name)
    parsed_url = urlparse(install["direct_url"])
    installed_from = (
        Path(unquote(parsed_url.path)).resolve() if parsed_url.scheme == "file" else None
    )
    if installed_from != wheel_path:
        raise RuntimeError("reviewer Python was not installed from the declared wheel")
    module_path = Path(install["module_path"])
    distribution_root = Path(install["distribution_root"])
    if not module_path.is_relative_to(distribution_root):
        raise RuntimeError("reviewer module is not loaded from its installed distribution")
    installed_distribution_digest = _verify_installed_distribution_matches_wheel(
        wheel_path,
        distribution_root,
    )
    return {
        "producer_repository_revision": producer_revision,
        "reviewer_module_name": module_name,
        "reviewer_module_path": str(module_path),
        "reviewer_distribution_root": str(distribution_root),
        "reviewer_install_url": install["direct_url"],
        "reviewer_installed_distribution_digest": installed_distribution_digest,
        "wheel_provenance_path": str(provenance_path),
        "wheel_provenance_sha256": _sha256(provenance_path),
    }


def _initial_receipt(config: ReviewServiceConfig) -> dict[str, object]:
    return {
        "schema_version": REVIEW_EXECUTION_SCHEMA_VERSION,
        "unit": config.unit,
        "run_label": config.run_label,
        "status": "failed",
        "finalized": False,
        "stop_reason": "service_did_not_finalize",
        "exit_code": None,
        "command": list(config.command),
        "working_directory": str(
            (
                config.working_directory
                if config.producer_source_directory is None
                else config.producer_source_directory
            ).resolve()
        ),
        "reviewer_working_directory": str(config.working_directory.resolve()),
        "python_executable": str(_absolute_executable(config.reviewer_python)),
        "service_execution_path": config.service_execution_path,
        "reviewer_revision": config.reviewer_revision,
        "producer_repository_revision": None,
        "reviewer_module_path": None,
        "reviewer_module_name": None,
        "reviewer_distribution_root": None,
        "reviewer_install_url": None,
        "reviewer_installed_distribution_digest": None,
        "wheel_path": str(config.wheel_path.resolve()),
        "wheel_sha256": None,
        "wheel_provenance_path": str(_wheel_provenance_path(config.wheel_path.resolve())),
        "wheel_provenance_sha256": None,
        "raw_manifest_path": str(config.raw_manifest.resolve()),
        "raw_manifest_sha256_before": None,
        "raw_manifest_sha256_after": None,
        "started_at": _utc_now(),
        "completed_at": None,
        "duration_seconds": None,
        "aggregate_peak_rss_bytes": 0,
        "aggregate_peak_swap_bytes": 0,
        "cgroup_memory_peak_status": "pending",
        "cgroup_memory_peak_error": None,
        "max_aggregate_rss_bytes": config.max_aggregate_rss_bytes,
        "systemd_memory_high_bytes": config.systemd_memory_high_bytes,
        "systemd_memory_max_bytes": config.systemd_memory_max_bytes,
        "systemd_memory_swap_max_bytes": config.systemd_memory_swap_max_bytes,
        "systemd_service_result": None,
        "systemd_exit_code": None,
        "systemd_exit_status": None,
        "service_log_sha256": None,
        "progress_log_path": (
            str(config.progress_log.resolve()) if config.progress_log is not None else None
        ),
        "progress_log_sha256": None,
    }


def _prepare_review_execution(config: ReviewServiceConfig) -> dict[str, object]:
    config.log_directory.parent.mkdir(parents=True, exist_ok=True)
    config.log_directory.mkdir(exist_ok=False)
    receipt = _initial_receipt(config)
    receipt_path = config.log_directory / "review_execution.json"
    _atomic_json(receipt_path, receipt)
    try:
        identity = _validate_formal_execution_envelope(config)
        receipt.update(identity)
        receipt["wheel_sha256"] = _sha256(config.wheel_path)
        receipt["raw_manifest_sha256_before"] = _sha256(config.raw_manifest)
        _atomic_json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        receipt["stop_reason"] = "preflight_failed"
        receipt["preflight_error_type"] = type(error).__name__
        receipt["preflight_error"] = str(error)
        receipt["completed_at"] = _utc_now()
        receipt["finalized"] = True
        _atomic_json(receipt_path, receipt)
        raise


def supervise_review(config: ReviewServiceConfig) -> int:
    """Run one validated reviewer command and persist state before every failure edge."""

    receipt_path = config.log_directory / "review_execution.json"
    if not receipt_path.is_file():
        _prepare_review_execution(config)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise RuntimeError("review execution receipt is invalid")
    _validate_formal_execution_envelope(config)
    raw_before = str(receipt["raw_manifest_sha256_before"])
    started = time.monotonic()
    peak_rss = 0
    peak_swap = 0
    stop_reason = "process_exit"
    service_log = config.log_directory / "service.log"
    exit_code = 1
    process: subprocess.Popen[bytes] | None = None
    previous_handlers = {
        caught: signal.getsignal(caught) for caught in (signal.SIGTERM, signal.SIGINT)
    }

    def interrupt(signum: int, _: object) -> None:
        raise ReviewServiceInterrupted(f"service interrupted by signal {signum}")

    for caught in previous_handlers:
        signal.signal(caught, interrupt)
    try:
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
    except ReviewServiceInterrupted:
        stop_reason = "service_interrupted"
        exit_code = 1
        if process is not None:
            _terminate_process_group(process)
    except BaseException as error:
        stop_reason = "supervisor_failed"
        exit_code = 1
        receipt["supervisor_error_type"] = type(error).__name__
        receipt["supervisor_error"] = str(error)
        if process is not None:
            _terminate_process_group(process)
    finally:
        for caught, previous in previous_handlers.items():
            signal.signal(caught, previous)
        raw_after = _sha256(config.raw_manifest) if config.raw_manifest.is_file() else None
        if raw_after != raw_before:
            stop_reason = "raw_manifest_changed"
            exit_code = 1
        receipt.update(
            {
                "status": (
                    "completed"
                    if exit_code == 0 and stop_reason == "process_exit"
                    else "failed"
                ),
                "stop_reason": stop_reason,
                "exit_code": exit_code,
                "completed_at": _utc_now(),
                "duration_seconds": time.monotonic() - started,
                "aggregate_peak_rss_bytes": peak_rss,
                "aggregate_peak_swap_bytes": peak_swap,
                "raw_manifest_sha256_after": raw_after,
                "service_log_sha256": _sha256(service_log) if service_log.is_file() else None,
                "progress_log_sha256": (
                    _sha256(config.progress_log)
                    if config.progress_log is not None and config.progress_log.is_file()
                    else None
                ),
            }
        )
        _atomic_json(receipt_path, receipt)
    return int(exit_code)


def _systemd_memory_peaks(unit: str) -> tuple[int, int]:
    try:
        result = subprocess.run(
            (
                "systemctl",
                "--user",
                "show",
                unit,
                "--property=MemoryPeak",
                "--property=MemorySwapPeak",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReviewResourceAccountingError(
            f"cannot read systemd memory accounting for {unit}: {error}"
        ) from error
    values: dict[str, int] = {}
    for line in result.stdout.splitlines():
        key, separator, raw_value = line.partition("=")
        if separator and raw_value.isdigit():
            values[key] = int(raw_value)
    if "MemoryPeak" not in values or "MemorySwapPeak" not in values:
        raise ReviewResourceAccountingError(
            f"systemd memory accounting is incomplete for {unit}: {result.stdout!r}"
        )
    return values["MemoryPeak"], values["MemorySwapPeak"]


def _receipt_nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def finalize_review_execution(config: ReviewServiceConfig) -> None:
    """Seal the receipt from ExecStopPost, including hard cgroup/OOM termination."""

    receipt_path = config.log_directory / "review_execution.json"
    if receipt_path.is_file():
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt = payload if isinstance(payload, dict) else _initial_receipt(config)
    else:
        config.log_directory.mkdir(parents=True, exist_ok=True)
        receipt = _initial_receipt(config)
    service_result = os.environ.get("SERVICE_RESULT", "unknown")
    if service_result == "oom-kill":
        receipt["status"] = "failed"
        receipt["stop_reason"] = "systemd_memory_max"
    elif receipt.get("stop_reason") == "service_did_not_finalize":
        receipt["status"] = "failed"
        receipt["stop_reason"] = (
            "service_interrupted" if service_result in {"signal", "timeout"} else "service_failed"
        )
    receipt["systemd_service_result"] = service_result
    receipt["systemd_exit_code"] = os.environ.get("EXIT_CODE")
    receipt["systemd_exit_status"] = os.environ.get("EXIT_STATUS")
    raw_after = _sha256(config.raw_manifest) if config.raw_manifest.is_file() else None
    receipt["raw_manifest_sha256_after"] = raw_after
    raw_before = receipt.get("raw_manifest_sha256_before")
    receipt["raw_manifest_unchanged"] = raw_before is not None and raw_after == raw_before
    if not receipt["raw_manifest_unchanged"]:
        receipt["status"] = "failed"
        receipt["stop_reason"] = "raw_manifest_changed"
    try:
        memory_peak, swap_peak = _systemd_memory_peaks(config.unit)
        receipt["cgroup_memory_peak_status"] = "verified"
        receipt["aggregate_peak_rss_bytes"] = max(
            _receipt_nonnegative_int(receipt.get("aggregate_peak_rss_bytes")), memory_peak
        )
        receipt["aggregate_peak_swap_bytes"] = max(
            _receipt_nonnegative_int(receipt.get("aggregate_peak_swap_bytes")), swap_peak
        )
    except ReviewResourceAccountingError as error:
        receipt["cgroup_memory_peak_status"] = "unavailable"
        receipt["cgroup_memory_peak_error"] = str(error)
        receipt["status"] = "failed"
        if receipt["stop_reason"] != "raw_manifest_changed":
            receipt["stop_reason"] = "resource_accounting_unavailable"
    service_log = config.log_directory / "service.log"
    receipt["service_log_sha256"] = _sha256(service_log) if service_log.is_file() else None
    receipt["progress_log_sha256"] = (
        _sha256(config.progress_log)
        if config.progress_log is not None and config.progress_log.is_file()
        else None
    )
    completed_at = datetime.now(UTC)
    started_at = receipt.get("started_at")
    if isinstance(started_at, str):
        receipt["duration_seconds"] = max(
            0.0,
            (completed_at - datetime.fromisoformat(started_at)).total_seconds(),
        )
    receipt["completed_at"] = completed_at.isoformat()
    receipt["finalized"] = True
    review_manifest_path = (
        config.raw_manifest.resolve().parent.parent / "review" / "review_manifest.json"
    )
    if (
        receipt.get("status") == "completed"
        and service_result == "success"
        and receipt.get("cgroup_memory_peak_status") == "verified"
        and receipt.get("raw_manifest_unchanged") is True
        and review_manifest_path.is_file()
    ):
        receipt["review_manifest_sha256"] = _sha256(review_manifest_path)
    _atomic_json(receipt_path, receipt)
    if receipt.get("review_manifest_sha256") is not None:
        _atomic_json(review_manifest_path.parent / "review_execution.json", receipt)


def launch_review_service(config: ReviewServiceConfig) -> None:
    """Create one transient user service without inheriting the caller's lifetime."""

    _prepare_review_execution(config)
    staging_directory = config.log_directory.with_name(f".{config.log_directory.name}.launch")
    staging_directory.mkdir(exist_ok=False)
    temporary_root = staging_directory / "tmp"
    temporary_root.mkdir()
    config_path = staging_directory / "service_config.json"
    _atomic_json(config_path, config.to_dict())
    finalizer = shlex.join(
        (
            str(_absolute_executable(config.reviewer_python)),
            "-I",
            "-m",
            "evrptw.stage052_review_service",
            "finalize",
            "--config",
            str(config_path.resolve()),
        )
    )
    command = (
        "systemd-run",
        "--user",
        "--collect",
        f"--unit={config.unit}",
        "--property=MemoryAccounting=yes",
        f"--property=MemoryHigh={config.systemd_memory_high_bytes}",
        f"--property=MemoryMax={config.systemd_memory_max_bytes}",
        f"--property=MemorySwapMax={config.systemd_memory_swap_max_bytes}",
        "--property=KillMode=control-group",
        "--property=Restart=no",
        "--property=OOMPolicy=stop",
        f"--property=ExecStopPost={finalizer}",
        f"--property=WorkingDirectory={config.working_directory.resolve()}",
        f"--setenv=PATH={config.service_execution_path}",
        f"--setenv=STAGE052_REVIEW_TMPDIR={temporary_root.resolve()}",
        str(_absolute_executable(config.reviewer_python)),
        "-I",
        "-m",
        "evrptw.stage052_review_service",
        "supervise",
        "--config",
        str(config_path.resolve()),
    )
    try:
        subprocess.run(command, check=True)
    except BaseException as error:
        receipt_path = config.log_directory / "review_execution.json"
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload.update(
                {
                    "status": "failed",
                    "finalized": True,
                    "stop_reason": "service_launch_failed",
                    "launch_error_type": type(error).__name__,
                    "launch_error": str(error),
                    "completed_at": _utc_now(),
                }
            )
            _atomic_json(receipt_path, payload)
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise


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
    launch.add_argument("--reviewer-revision", required=True)
    launch.add_argument("--producer-source-directory", type=Path)
    launch.add_argument("--review-memory-contract", type=Path)
    launch.add_argument("--max-aggregate-rss-gib", type=float, default=5.5)
    launch.add_argument("command", nargs=argparse.REMAINDER)
    supervise = subparsers.add_parser("supervise")
    supervise.add_argument("--config", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--config", type=Path, required=True)
    seal_wheel = subparsers.add_parser("seal-wheel")
    seal_wheel.add_argument("--wheel-path", type=Path, required=True)
    seal_wheel.add_argument("--reviewer-revision", required=True)
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
        log_directory = DEFAULT_LOG_ROOT / arguments.run_label / timestamp
        command = list(_parse_command(arguments.command))
        progress_log = log_directory / "progress.jsonl"
        memory_contract = (
            load_review_memory_contract(arguments.review_memory_contract.resolve())
            if arguments.review_memory_contract is not None
            else None
        )
        maximum_rss_bytes = (
            memory_contract.process_guard_bytes
            if memory_contract is not None
            else int(arguments.max_aggregate_rss_gib * 1024**3)
        )
        command.extend(
            (
                "--progress-log",
                str(progress_log),
                "--max-aggregate-rss-gib",
                str(maximum_rss_bytes / 1024**3),
                _REVIEW_EXECUTION_OPTION,
                str(log_directory / "review_execution.json"),
            )
        )
        if memory_contract is not None:
            command.extend(
                (
                    "--review-workers",
                    str(memory_contract.review_workers),
                    "--review-memory-contract",
                    str(arguments.review_memory_contract.resolve()),
                )
            )
        config = ReviewServiceConfig(
            unit=unit,
            run_label=arguments.run_label,
            command=tuple(command),
            reviewer_python=Path(command[0]),
            reviewer_revision=arguments.reviewer_revision,
            working_directory=arguments.working_directory,
            log_directory=log_directory,
            raw_manifest=arguments.raw_manifest,
            wheel_path=arguments.wheel_path,
            service_execution_path=_service_execution_path(),
            producer_source_directory=arguments.producer_source_directory,
            progress_log=progress_log,
            review_memory_contract=(
                memory_contract.to_dict() if memory_contract is not None else None
            ),
            max_aggregate_rss_bytes=maximum_rss_bytes,
            systemd_memory_high_bytes=(
                memory_contract.memory_high_bytes
                if memory_contract is not None
                else DEFAULT_SYSTEMD_MEMORY_HIGH_BYTES
            ),
            systemd_memory_max_bytes=(
                memory_contract.memory_max_bytes
                if memory_contract is not None
                else DEFAULT_SYSTEMD_MEMORY_MAX_BYTES
            ),
            systemd_memory_swap_max_bytes=(
                memory_contract.memory_swap_max_bytes
                if memory_contract is not None
                else DEFAULT_SYSTEMD_MEMORY_SWAP_MAX_BYTES
            ),
        )
        launch_review_service(config)
        print(json.dumps({"unit": unit, "log_directory": str(log_directory)}))
        return 0
    if arguments.action == "supervise":
        payload = json.loads(arguments.config.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("review service config must be an object")
        return supervise_review(ReviewServiceConfig.from_dict(payload))
    if arguments.action == "finalize":
        payload = json.loads(arguments.config.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("review service config must be an object")
        try:
            finalize_review_execution(ReviewServiceConfig.from_dict(payload))
            return 0
        finally:
            shutil.rmtree(arguments.config.parent, ignore_errors=True)
    if arguments.action == "seal-wheel":
        provenance_path = seal_reviewer_wheel(
            arguments.wheel_path,
            arguments.reviewer_revision,
        )
        print(provenance_path)
        return 0
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
