"""Process-tree resource instrumentation for reproducible experiment envelopes."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

import psutil  # type: ignore[import-untyped]


@dataclass(slots=True)
class _ProcessObservation:
    first_cpu_seconds: float
    last_cpu_seconds: float
    maximum_rss_bytes: int


@dataclass(slots=True)
class ProcessTreeMonitor:
    """Sample one solve's complete process tree, including external roots."""

    additional_root_pids: tuple[int, ...] = ()
    excluded_root_pids: tuple[int, ...] = ()
    sample_interval_seconds: float = 0.05
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _observations: dict[tuple[int, float], _ProcessObservation] = field(
        default_factory=dict,
        init=False,
    )
    _peak_aggregate_rss_bytes: int = field(default=0, init=False)
    _peak_aggregate_pss_bytes: int = field(default=0, init=False)
    _peak_processes: int = field(default=0, init=False)
    _peak_threads: int = field(default=0, init=False)
    _sample_count: int = field(default=0, init=False)
    _baseline_complete: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.sample_interval_seconds <= 0.0:
            raise ValueError("process-tree sample interval must be positive")
        if any(pid <= 0 for pid in self.additional_root_pids):
            raise ValueError("additional process-tree roots must be positive PIDs")
        if any(pid <= 0 for pid in self.excluded_root_pids):
            raise ValueError("excluded process-tree roots must be positive PIDs")
        if set(self.additional_root_pids) & set(self.excluded_root_pids):
            raise ValueError("a process-tree root cannot be both included and excluded")

    def __enter__(self) -> ProcessTreeMonitor:
        if self._thread is not None:
            raise RuntimeError("process-tree monitor is already running")
        self._sample()
        self._baseline_complete = True
        self._thread = threading.Thread(
            target=self._run,
            name="evrptw-process-tree-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join(timeout=max(1.0, self.sample_interval_seconds * 4.0))
        if self._thread.is_alive():
            raise RuntimeError("process-tree monitor did not stop")
        self._sample()

    def statistics(
        self,
        *,
        elapsed_seconds: float,
        compute_thread_limit: int,
    ) -> dict[str, object]:
        if self._thread is None or self._thread.is_alive():
            raise RuntimeError("process-tree monitor statistics require a completed monitor")
        if elapsed_seconds <= 0.0 or compute_thread_limit <= 0:
            raise ValueError("resource-envelope elapsed time and thread limit must be positive")
        cpu_seconds = sum(
            max(0.0, observation.last_cpu_seconds - observation.first_cpu_seconds)
            for observation in self._observations.values()
        )
        return {
            "sample_interval_seconds": self.sample_interval_seconds,
            "sample_count": self._sample_count,
            "observed_processes": len(self._observations),
            "observed_process_ids": sorted(
                {identity[0] for identity in self._observations}
            ),
            "peak_concurrent_processes": self._peak_processes,
            "peak_aggregate_threads": self._peak_threads,
            "peak_aggregate_rss_bytes": self._peak_aggregate_rss_bytes,
            "peak_aggregate_pss_bytes": self._peak_aggregate_pss_bytes,
            "process_tree_cpu_seconds": cpu_seconds,
            "cpu_utilization_percent_of_one_core": (
                100.0 * cpu_seconds / elapsed_seconds
            ),
            "cpu_utilization_percent_of_compute_limit": (
                100.0 * cpu_seconds / (elapsed_seconds * compute_thread_limit)
            ),
            "root_process_id": os.getpid(),
            "additional_root_pids": list(self.additional_root_pids),
            "excluded_root_pids": list(self.excluded_root_pids),
        }

    def _run(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            self._sample()

    def _sample(self) -> None:
        processes: dict[tuple[int, float], psutil.Process] = {}
        excluded: set[tuple[int, float]] = set()
        for pid in self.excluded_root_pids:
            try:
                root = psutil.Process(pid)
                candidates = (root, *root.children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            for process in candidates:
                try:
                    excluded.add((process.pid, process.create_time()))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        for pid in (os.getpid(), *self.additional_root_pids):
            try:
                root = psutil.Process(pid)
                candidates = (root, *root.children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            for process in candidates:
                try:
                    identity = (process.pid, process.create_time())
                    if identity not in excluded:
                        processes[identity] = process
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        aggregate_rss = 0
        aggregate_pss = 0
        aggregate_threads = 0
        for identity, process in processes.items():
            try:
                cpu = process.cpu_times()
                cpu_seconds = float(cpu.user + cpu.system)
                rss_bytes = int(process.memory_info().rss)
                pss_bytes = int(getattr(process.memory_full_info(), "pss", rss_bytes))
                thread_count = process.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            aggregate_rss += rss_bytes
            aggregate_pss += pss_bytes
            aggregate_threads += thread_count
            observation = self._observations.get(identity)
            if observation is None:
                self._observations[identity] = _ProcessObservation(
                    # Processes present in the synchronous entry sample may
                    # have accumulated CPU before this measurement envelope;
                    # subtract that baseline.  A descendant first discovered
                    # later was created during the envelope, so its cumulative
                    # CPU belongs to this solve even if it exits before a
                    # second sample can observe it.
                    first_cpu_seconds=(cpu_seconds if not self._baseline_complete else 0.0),
                    last_cpu_seconds=cpu_seconds,
                    maximum_rss_bytes=rss_bytes,
                )
            else:
                observation.last_cpu_seconds = cpu_seconds
                observation.maximum_rss_bytes = max(
                    observation.maximum_rss_bytes,
                    rss_bytes,
                )
        self._peak_aggregate_rss_bytes = max(
            self._peak_aggregate_rss_bytes,
            aggregate_rss,
        )
        self._peak_aggregate_pss_bytes = max(
            self._peak_aggregate_pss_bytes,
            aggregate_pss,
        )
        self._peak_processes = max(self._peak_processes, len(processes))
        self._peak_threads = max(self._peak_threads, aggregate_threads)
        self._sample_count += 1


__all__ = ("ProcessTreeMonitor",)
