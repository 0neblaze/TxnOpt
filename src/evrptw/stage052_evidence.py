"""Public Stage 5.2 prerequisite and process-resource evidence seams."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil  # type: ignore[import-untyped]

from evrptw.artifacts import ArtifactIntegrityError, ArtifactReader

STAGE052_REVIEW_SCHEMA_VERSION = "stage05.2-review-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class Stage052PrerequisiteIdentity:
    run_label: str
    component: str
    status: str
    repository_revision: str
    configuration_sha256: str
    raw_manifest_sha256: str
    review_manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def verify_stage052_prerequisite(
    raw_dir: Path,
    *,
    expected_component: str,
    expected_status: str,
    expected_run_label: str | None = None,
) -> Stage052PrerequisiteIdentity:
    """Verify an accepted Stage 5.2 producer and independent review bundle."""

    raw_dir = raw_dir.resolve()
    if expected_run_label is not None and raw_dir.name != expected_run_label:
        raise ArtifactIntegrityError(
            f"prerequisite run label mismatch: expected={expected_run_label} "
            f"observed={raw_dir.name}"
        )
    review_manifest_path = raw_dir / "review" / "review_manifest.json"
    try:
        review = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactIntegrityError(
            f"cannot read prerequisite review manifest: {review_manifest_path}"
        ) from error
    expected_review = {
        "schema_version": STAGE052_REVIEW_SCHEMA_VERSION,
        "run_label": raw_dir.name,
        "component": expected_component,
        "scope": "performance",
        "status": expected_status,
    }
    for field, expected in expected_review.items():
        if review.get(field) != expected:
            raise ArtifactIntegrityError(
                f"prerequisite review {field} mismatch: "
                f"expected={expected} observed={review.get(field)}"
            )
    gates = review.get("gates")
    if (
        not isinstance(gates, dict)
        or not gates
        or any(
            not isinstance(gate, dict) or gate.get("passed") is not True
            for gate in gates.values()
        )
    ):
        raise ArtifactIntegrityError("prerequisite review contains a failed or invalid gate")
    files = review.get("files")
    if not isinstance(files, dict) or set(files) != {
        "review_findings.csv",
        "review_report.md",
    }:
        raise ArtifactIntegrityError("prerequisite review file identity mismatch")
    for name, checksum in files.items():
        path = raw_dir / "review" / str(name)
        if not path.is_file() or _sha256(path) != str(checksum):
            raise ArtifactIntegrityError(f"prerequisite review checksum mismatch: {name}")

    reader = ArtifactReader(raw_dir)
    if reader.manifest.get("evidence_completeness") != "complete":
        raise ArtifactIntegrityError("prerequisite raw evidence is partial")
    metadata_items = [
        item
        for item in reader.manifest.get("artifacts", [])
        if isinstance(item, dict) and item.get("artifact_type") == "manifest_metadata"
    ]
    config_items = [
        item
        for item in reader.manifest.get("artifacts", [])
        if isinstance(item, dict) and item.get("artifact_type") == "config"
    ]
    if len(metadata_items) != 1 or len(config_items) != 1:
        raise ArtifactIntegrityError("prerequisite control artifacts are incomplete")
    metadata = reader.read_json(str(metadata_items[0]["relative_path"]))
    if (
        metadata.get("run_label") != raw_dir.name
        or metadata.get("component") != expected_component
        or metadata.get("scope") != "performance"
        or metadata.get("repository_dirty") is not False
    ):
        raise ArtifactIntegrityError("prerequisite producer identity mismatch")
    config_checksum = str(config_items[0].get("checksum", ""))
    if metadata.get("configuration_sha256") != config_checksum:
        raise ArtifactIntegrityError("prerequisite configuration checksum mismatch")
    revision = str(metadata.get("repository_revision", ""))
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ArtifactIntegrityError("prerequisite repository revision is invalid")
    manifest_path = raw_dir / "control" / f"{raw_dir.name}_manifest.json"
    return Stage052PrerequisiteIdentity(
        run_label=raw_dir.name,
        component=expected_component,
        status=expected_status,
        repository_revision=revision,
        configuration_sha256=config_checksum,
        raw_manifest_sha256=_sha256(manifest_path),
        review_manifest_sha256=_sha256(review_manifest_path),
    )


@dataclass(frozen=True, slots=True)
class RunResourceSummary:
    schema_version: str
    run_wall_seconds: float
    sample_interval_seconds: float
    parent_pid: int
    worker_pids: tuple[int, ...]
    aggregate_peak_rss_bytes: int
    mean_active_cores: float
    peak_active_cores: float
    sample_count: int
    status: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["worker_pids"] = list(self.worker_pids)
        return payload


class ProcessTreeResourceSampler:
    """Sample simultaneous RSS and CPU usage for one process tree."""

    def __init__(self, *, interval_seconds: float = 0.05, parent_pid: int | None = None) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("resource sample interval must be positive")
        self.interval_seconds = interval_seconds
        self.parent_pid = parent_pid or os.getpid()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._peak_rss = 0
        self._active_core_samples: list[float] = []
        self._worker_pids: set[int] = set()
        self._sample_count = 0
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("resource sampler is already started")
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="stage052-resource-sampler")
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> RunResourceSummary:
        if self._thread is None:
            raise RuntimeError("resource sampler was not started")
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 4.0))
        if self._thread.is_alive():
            raise RuntimeError("resource sampler did not stop")
        if self._error is not None:
            raise RuntimeError(f"resource sampling failed: {self._error}") from self._error
        wall = time.perf_counter() - self._started
        return RunResourceSummary(
            schema_version="stage05.2-run-resource-v1",
            run_wall_seconds=wall,
            sample_interval_seconds=self.interval_seconds,
            parent_pid=self.parent_pid,
            worker_pids=tuple(sorted(self._worker_pids)),
            aggregate_peak_rss_bytes=self._peak_rss,
            mean_active_cores=(
                sum(self._active_core_samples) / len(self._active_core_samples)
                if self._active_core_samples
                else 0.0
            ),
            peak_active_cores=max(self._active_core_samples, default=0.0),
            sample_count=self._sample_count,
            status="complete",
        )

    def _run(self) -> None:
        previous_wall: float | None = None
        previous_cpu: float | None = None
        try:
            while not self._stop.is_set():
                processes = self._processes()
                now = time.perf_counter()
                rss = 0
                cpu = 0.0
                for process in processes:
                    try:
                        with process.oneshot():
                            rss += int(process.memory_info().rss)
                            times = process.cpu_times()
                            cpu += float(times.user + times.system)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        continue
                self._peak_rss = max(self._peak_rss, rss)
                self._sample_count += 1
                if previous_wall is not None and previous_cpu is not None and now > previous_wall:
                    self._active_core_samples.append(
                        max(0.0, (cpu - previous_cpu) / (now - previous_wall))
                    )
                previous_wall = now
                previous_cpu = cpu
                self._stop.wait(self.interval_seconds)
        except BaseException as error:
            self._error = error

    def _processes(self) -> list[Any]:
        parent = psutil.Process(self.parent_pid)
        children = parent.children(recursive=True)
        self._worker_pids.update(process.pid for process in children)
        return [parent, *children]
