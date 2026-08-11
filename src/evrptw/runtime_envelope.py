"""Process-tree resource instrumentation for reproducible experiment envelopes.

The monitor deliberately treats process identity as ``(pid, create_time)``.
PIDs are reusable, and a process which was already alive when the envelope
started must not contribute CPU accumulated before the envelope.  Conversely,
a descendant created after the monitor started contributes all of its lifetime
CPU, even when it is first observed only after it has already done work.

Optional Linux ``/proc`` and psutil counters are represented as
``"unavailable"`` when they cannot be read.  A missing counter is never
silently converted to zero, because that would make resource gates look more
favourable than the measured execution.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import psutil  # type: ignore[import-untyped]

_UNAVAILABLE = "unavailable"
_MAX_BOUNDED_SAMPLES = 256
_MAX_CPU_STAT_CORES = 256
_MAX_THREAD_OBSERVATIONS = 4096
_PROCESS_ERRORS = (
    psutil.AccessDenied,
    psutil.NoSuchProcess,
    psutil.ZombieProcess,
    NotImplementedError,
    OSError,
    TypeError,
    ValueError,
)
_COUNTER_NAMES = (
    "voluntary_context_switches",
    "involuntary_context_switches",
    "minor_faults",
    "major_faults",
    "read_bytes",
    "write_bytes",
    "schedstat_runtime_ns",
    "schedstat_runqueue_delay_ns",
    "schedstat_timeslices",
    "cpu_migrations",
)
_CPU_STAT_DELTA_FIELDS = ("busy", "user", "system", "iowait", "steal")
_PRESSURE_SOURCES = ("cpu", "memory", "io")
_PRESSURE_STALLS = ("some", "full")
_THREAD_COUNTER_NAMES = (
    "voluntary_context_switches",
    "involuntary_context_switches",
    "minor_faults",
    "major_faults",
    "schedstat_runtime_ns",
    "schedstat_runqueue_delay_ns",
    "schedstat_timeslices",
    "cpu_migrations",
)
_CPUStatSnapshot = dict[int, tuple[int, ...]]
_PressureSnapshot = dict[str, dict[str, float | int]]


@dataclass(frozen=True, slots=True)
class _ProcessSample:
    """One successful process sample.

    RSS and CPU are mandatory.  Every other value is optional and remains
    ``None`` when the host does not expose it.
    """

    user_cpu_seconds: float
    system_cpu_seconds: float
    rss_bytes: int
    pss_bytes: int | None
    thread_count: int | None
    counters: dict[str, int | None]
    affinity: tuple[int, ...] | None


@dataclass(slots=True)
class _ProcessObservation:
    """Cumulative counters for one PID/create-time identity."""

    first_user_cpu_seconds: float
    last_user_cpu_seconds: float
    first_system_cpu_seconds: float
    last_system_cpu_seconds: float
    first_counters: dict[str, int | None]
    last_counters: dict[str, int | None]
    maximum_rss_bytes: int
    maximum_pss_bytes: int | None
    last_rss_bytes: int
    last_pss_bytes: int | None
    last_thread_count: int | None
    last_affinity: tuple[int, ...] | None
    sample_count: int = 1
    cpu_baseline_source: str = "monitor_start"


@dataclass(frozen=True, slots=True)
class _ThreadSample:
    """One Linux task sample used only for thread-level diagnostics."""

    start_time_ticks: int
    user_cpu_seconds: float
    system_cpu_seconds: float
    counters: dict[str, int]
    affinity: tuple[int, ...]


@dataclass(slots=True)
class _ThreadObservation:
    """Cumulative counters for one process/TID/start-time identity."""

    first_user_cpu_seconds: float
    last_user_cpu_seconds: float
    first_system_cpu_seconds: float
    last_system_cpu_seconds: float
    first_counters: dict[str, int]
    last_counters: dict[str, int]
    last_affinity: tuple[int, ...]
    sample_count: int = 1
    cpu_baseline_source: str = "monitor_start"


def _read_proc_stat_faults(pid: int) -> dict[str, int | None]:
    """Read process-only page-fault counters from ``/proc/<pid>/stat``."""

    result: dict[str, int | None] = {
        "minor_faults": None,
        "major_faults": None,
    }
    if os.name != "posix":
        return result
    try:
        payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        # ``comm`` may contain spaces and parentheses.  Splitting after the
        # final closing parenthesis keeps the fixed field offsets intact.
        _, fields = payload.rsplit(")", 1)
        values = fields.split()
        # The first value is field 3 (state); minflt is field 10 and majflt
        # field 12, hence indices 7 and 9 in this suffix.
        if len(values) > 9:
            result["minor_faults"] = int(values[7])
            result["major_faults"] = int(values[9])
    except (OSError, ValueError, UnicodeError):
        return result
    return result


def _read_proc_status_context_switches(pid: int) -> dict[str, int | None]:
    """Read voluntary/non-voluntary context switches if ``/proc`` exposes them."""

    result: dict[str, int | None] = {
        "voluntary_context_switches": None,
        "involuntary_context_switches": None,
    }
    if os.name != "posix":
        return result
    try:
        for line in (
            Path(f"/proc/{pid}/status").read_text(encoding="ascii", errors="strict").splitlines()
        ):
            key, separator, value = line.partition(":")
            if not separator:
                continue
            field = {
                "voluntary_ctxt_switches": "voluntary_context_switches",
                "nonvoluntary_ctxt_switches": "involuntary_context_switches",
            }.get(key)
            if field is not None:
                result[field] = int(value.strip())
    except (OSError, ValueError, UnicodeError):
        return result
    return result


def _read_proc_schedstat(pid: int) -> dict[str, int | None]:
    """Read Linux schedstat runtime, run-queue delay and timeslices."""

    result: dict[str, int | None] = {
        "schedstat_runtime_ns": None,
        "schedstat_runqueue_delay_ns": None,
        "schedstat_timeslices": None,
    }
    if os.name != "posix":
        return result
    try:
        values = Path(f"/proc/{pid}/schedstat").read_text(encoding="ascii", errors="strict").split()
        if len(values) >= 3:
            result["schedstat_runtime_ns"] = int(values[0])
            result["schedstat_runqueue_delay_ns"] = int(values[1])
            result["schedstat_timeslices"] = int(values[2])
    except (OSError, ValueError, UnicodeError):
        return result
    return result


def _read_proc_sched_migrations(pid: int) -> int | None:
    """Read ``se.nr_migrations`` from a Linux per-process sched file."""

    if os.name != "posix":
        return None
    try:
        for line in (
            Path(f"/proc/{pid}/sched").read_text(encoding="ascii", errors="strict").splitlines()
        ):
            key, separator, value = line.partition(":")
            if separator and key.strip() == "se.nr_migrations":
                return int(value.strip())
    except (OSError, ValueError, UnicodeError):
        return None
    return None


def _read_thread_sample(pid: int, tid: int, clock_ticks: int) -> _ThreadSample | None:
    """Read one Linux task without treating a racing exit as zero work."""

    if os.name != "posix" or clock_ticks <= 0:
        return None
    task_root = Path(f"/proc/{pid}/task/{tid}")
    try:
        raw_stat = (task_root / "stat").read_text(encoding="ascii", errors="strict")
        _, suffix = raw_stat.rsplit(")", 1)
        fields = suffix.split()
        if len(fields) <= 19:
            return None
        minor_faults = int(fields[7])
        major_faults = int(fields[9])
        user_ticks = int(fields[11])
        system_ticks = int(fields[12])
        start_time_ticks = int(fields[19])

        schedstat = (task_root / "schedstat").read_text(encoding="ascii", errors="strict").split()
        if len(schedstat) < 3:
            return None
        schedstat_runtime = int(schedstat[0])
        schedstat_runqueue = int(schedstat[1])
        schedstat_timeslices = int(schedstat[2])

        voluntary: int | None = None
        involuntary: int | None = None
        for line in (
            (task_root / "status").read_text(encoding="ascii", errors="strict").splitlines()
        ):
            key, separator, value = line.partition(":")
            if not separator:
                continue
            if key == "voluntary_ctxt_switches":
                voluntary = int(value.strip())
            elif key == "nonvoluntary_ctxt_switches":
                involuntary = int(value.strip())
        if voluntary is None or involuntary is None:
            return None

        migrations: int | None = None
        for line in (task_root / "sched").read_text(encoding="ascii", errors="strict").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == "se.nr_migrations":
                migrations = int(value.strip())
                break
        if migrations is None:
            return None
        affinity = tuple(sorted(os.sched_getaffinity(tid)))
        if not affinity:
            return None
    except (OSError, ValueError, UnicodeError):
        return None
    counters = {
        "voluntary_context_switches": voluntary,
        "involuntary_context_switches": involuntary,
        "minor_faults": minor_faults,
        "major_faults": major_faults,
        "schedstat_runtime_ns": schedstat_runtime,
        "schedstat_runqueue_delay_ns": schedstat_runqueue,
        "schedstat_timeslices": schedstat_timeslices,
        "cpu_migrations": migrations,
    }
    if any(value < 0 for value in counters.values()) or start_time_ticks < 0:
        return None
    return _ThreadSample(
        start_time_ticks=start_time_ticks,
        user_cpu_seconds=user_ticks / clock_ticks,
        system_cpu_seconds=system_ticks / clock_ticks,
        counters=counters,
        affinity=affinity,
    )


def _read_process_threads(
    pid: int,
    clock_ticks: int,
    *,
    tids: tuple[int, ...] | None = None,
) -> tuple[dict[tuple[int, int], _ThreadSample], bool]:
    """Return a bounded Linux task snapshot and an explicit overflow flag."""

    if os.name != "posix" or clock_ticks <= 0:
        return {}, False
    if tids is None:
        tids, overflow = _list_process_thread_ids(pid)
        if overflow:
            return {}, True
    result: dict[tuple[int, int], _ThreadSample] = {}
    for tid in tids:
        sample = _read_thread_sample(pid, tid, clock_ticks)
        if sample is not None:
            result[(tid, sample.start_time_ticks)] = sample
    return result, False


def _list_process_thread_ids(pid: int) -> tuple[tuple[int, ...], bool]:
    """Discover Linux task IDs without reading every expensive task counter."""

    if os.name != "posix":
        return (), False
    try:
        tids = tuple(
            sorted(
                int(entry.name)
                for entry in Path(f"/proc/{pid}/task").iterdir()
                if entry.name.isdigit()
            )
        )
    except (OSError, ValueError):
        return (), False
    if len(tids) > _MAX_THREAD_OBSERVATIONS:
        return (), True
    return tids, False


def _read_boot_time_ticks(clock_ticks: int) -> int | None:
    """Return a conservative Linux boot-time tick marker for task identity.

    Linux task start times use clock ticks since boot.  A strict comparison
    against this marker distinguishes a task created during monitoring from a
    pre-existing task whose first detailed read was delayed.  Tasks in the
    same tick as entry are conservatively treated as pre-existing.
    """

    if os.name != "posix" or clock_ticks <= 0:
        return None
    try:
        uptime_text = Path("/proc/uptime").read_text(
            encoding="ascii",
            errors="strict",
        )
        uptime_seconds = float(uptime_text.split(maxsplit=1)[0])
    except (OSError, ValueError, IndexError, UnicodeError):
        return None
    if not math.isfinite(uptime_seconds) or uptime_seconds < 0.0:
        return None
    return math.floor(uptime_seconds * clock_ticks)


def _read_proc_cpu_stat() -> dict[int, tuple[int, ...]] | None:
    """Read bounded per-CPU jiffy counters from ``/proc/stat``."""

    if os.name != "posix":
        return None
    try:
        lines = Path("/proc/stat").read_text(encoding="ascii", errors="strict").splitlines()
    except (OSError, UnicodeError):
        return None
    counters: dict[int, tuple[int, ...]] = {}
    for line in lines:
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or fields[0] == "cpu":
            continue
        try:
            cpu_id = int(fields[0][3:])
            values = tuple(int(value) for value in fields[1:10])
        except (ValueError, TypeError):
            return None
        if cpu_id < 0 or len(values) < 8:
            return None
        counters[cpu_id] = values
        if len(counters) > _MAX_CPU_STAT_CORES:
            return None
    return counters or None


def _cpu_stat_delta(
    previous: dict[int, tuple[int, ...]],
    current: dict[int, tuple[int, ...]],
) -> dict[int, dict[str, int]] | None:
    """Return monotonic per-CPU busy/user/system/iowait/steal deltas."""

    if set(previous) != set(current):
        return None
    delta: dict[int, dict[str, int]] = {}
    for cpu_id, old_values in previous.items():
        new_values = current[cpu_id]
        if len(old_values) < 8 or len(new_values) < 8:
            return None
        # /proc/stat fields: user, nice, system, idle, iowait, irq,
        # softirq, steal, guest, guest_nice.  ``user`` below includes nice;
        # ``busy`` excludes idle and iowait but includes interrupt/steal time.
        raw = {
            "user": new_values[0] + new_values[1] - old_values[0] - old_values[1],
            "system": new_values[2] - old_values[2],
            "iowait": new_values[4] - old_values[4],
            "steal": new_values[7] - old_values[7],
            "busy": (
                new_values[0]
                + new_values[1]
                + new_values[2]
                + new_values[5]
                + new_values[6]
                + new_values[7]
                - old_values[0]
                - old_values[1]
                - old_values[2]
                - old_values[5]
                - old_values[6]
                - old_values[7]
            ),
        }
        if any(value < 0 for value in raw.values()):
            return None
        delta[cpu_id] = raw
    return delta


def _read_pressure_snapshot(
    source: str,
) -> dict[str, dict[str, float | int]] | None:
    """Read PSI avg10/60/300 and cumulative total for one source."""

    if os.name != "posix":
        return None
    path = Path(f"/proc/pressure/{source}")
    try:
        lines = path.read_text(encoding="ascii", errors="strict").splitlines()
    except (OSError, UnicodeError):
        return None
    result: dict[str, dict[str, float | int]] = {}
    for line in lines:
        fields = line.split()
        if not fields or fields[0] not in _PRESSURE_STALLS:
            continue
        values: dict[str, float | int] = {}
        for token in fields[1:]:
            key, separator, value = token.partition("=")
            if not separator:
                continue
            try:
                values[key] = float(value) if key.startswith("avg") else int(value)
            except ValueError:
                return None
        if not all(key in values for key in ("avg10", "avg60", "avg300", "total")):
            return None
        result[fields[0]] = values
    if not result:
        return None
    for stall in _PRESSURE_STALLS:
        if stall not in result:
            result[stall] = {}
    return result


def _read_process_sample(process: psutil.Process) -> _ProcessSample | None:
    """Collect a process sample without fabricating unavailable metrics."""

    try:
        cpu = process.cpu_times()
        memory = process.memory_info()
        rss_bytes = int(memory.rss)
        if rss_bytes <= 0:
            # A zero-RSS transient is not reliable process-tree evidence.
            return None
        user_cpu_seconds = float(cpu.user)
        system_cpu_seconds = float(cpu.system)
        if not math.isfinite(user_cpu_seconds) or not math.isfinite(system_cpu_seconds):
            return None
    except _PROCESS_ERRORS:
        return None

    try:
        full_memory = process.memory_full_info()
        pss_raw = getattr(full_memory, "pss", None)
        pss_value = None if pss_raw is None else int(pss_raw)
        pss_bytes = None if pss_value is None or pss_value <= 0 else pss_value
    except _PROCESS_ERRORS:
        pss_bytes = None

    try:
        thread_count = int(process.num_threads())
    except _PROCESS_ERRORS:
        thread_count = None

    counters: dict[str, int | None] = {name: None for name in _COUNTER_NAMES}
    try:
        context = process.num_ctx_switches()
        counters["voluntary_context_switches"] = int(context.voluntary)
        counters["involuntary_context_switches"] = int(context.involuntary)
    except _PROCESS_ERRORS:
        pass
    try:
        io = process.io_counters()
        counters["read_bytes"] = int(io.read_bytes)
        counters["write_bytes"] = int(io.write_bytes)
    except _PROCESS_ERRORS:
        pass

    pid = int(process.pid)
    counters.update(_read_proc_stat_faults(pid))
    counters.update(_read_proc_status_context_switches(pid))
    counters.update(_read_proc_schedstat(pid))
    counters["cpu_migrations"] = _read_proc_sched_migrations(pid)

    try:
        affinity = tuple(sorted(int(cpu_id) for cpu_id in process.cpu_affinity()))
    except _PROCESS_ERRORS:
        affinity = None

    return _ProcessSample(
        user_cpu_seconds=user_cpu_seconds,
        system_cpu_seconds=system_cpu_seconds,
        rss_bytes=rss_bytes,
        pss_bytes=pss_bytes,
        thread_count=thread_count,
        counters=counters,
        affinity=affinity,
    )


def _process_identity(process: psutil.Process) -> tuple[int, float] | None:
    """Return a finite PID/create-time identity or no trustworthy identity."""

    try:
        pid = int(process.pid)
        create_time = float(process.create_time())
    except _PROCESS_ERRORS:
        return None
    if pid <= 0 or not math.isfinite(create_time):
        return None
    return pid, create_time


def _counter_delta(first: int | None, last: int | None) -> int | None:
    if first is None or last is None:
        return None
    return max(0, last - first)


def _sum_optional(values: list[int | None]) -> int | str:
    if not values or any(value is None for value in values):
        return _UNAVAILABLE
    return sum(value for value in values if value is not None)


def _quantile(values: list[float], probability: float) -> float | str:
    if not values:
        return _UNAVAILABLE
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


@dataclass(slots=True)
class ProcessTreeMonitor:
    """Sample one solve's complete process tree, including external roots.

    Process identities are PID plus ``create_time``.  This prevents a reused
    PID from inheriting the old process's CPU baseline.  Values for processes
    present in the entry sample start at that sample; processes whose
    ``create_time`` is after the monitor start contribute their full lifetime
    counters when first observed.
    """

    additional_root_pids: tuple[int, ...] = ()
    excluded_root_pids: tuple[int, ...] = ()
    sample_interval_seconds: float = 0.05
    thread_detail_interval_seconds: float = 0.25
    _MAX_BOUNDED_SAMPLES: ClassVar[int] = _MAX_BOUNDED_SAMPLES
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _root_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _registered_additional_root_pids: set[int] = field(default_factory=set, init=False)
    _worker_descendant_classification_epoch: int = field(default=0, init=False)
    _observations: dict[tuple[int, float], _ProcessObservation] = field(
        default_factory=dict,
        init=False,
    )
    _thread_observations: dict[tuple[int, float, int, int], _ThreadObservation] = field(
        default_factory=dict,
        init=False,
    )
    _thread_metrics_available: bool = field(default=True, init=False)
    _thread_observation_overflow: bool = field(default=False, init=False)
    _thread_sample_missed_processes: int = field(default=0, init=False)
    _unresolved_thread_ids: set[tuple[int, float, int]] = field(
        default_factory=set,
        init=False,
    )
    _active_thread_ids: dict[tuple[int, float], set[int]] = field(
        default_factory=dict,
        init=False,
    )
    _known_thread_ids: dict[tuple[int, float], set[int]] = field(
        default_factory=dict,
        init=False,
    )
    _known_thread_start_ticks: dict[tuple[int, float, int], int] = field(
        default_factory=dict,
        init=False,
    )
    _last_thread_detail_monotonic: float | None = field(default=None, init=False)
    _thread_detail_sample_count: int = field(default=0, init=False)
    _thread_discovery_sample_count: int = field(default=0, init=False)
    _thread_affinity_union: set[int] = field(default_factory=set, init=False)
    _thread_affinity_intersection: set[int] | None = field(default=None, init=False)
    _peak_aggregate_rss_bytes: int = field(default=0, init=False)
    _peak_aggregate_pss_bytes: int | None = field(default=0, init=False)
    _peak_worker_descendant_pss_bytes: int = field(default=0, init=False)
    _peak_worker_descendant_pss_sample_index: int | None = field(default=None, init=False)
    _peak_worker_descendant_pss_processes: tuple[tuple[int, float, int], ...] = field(
        default=(),
        init=False,
    )
    _worker_descendant_pss_available: bool = field(default=True, init=False)
    _peak_processes: int = field(default=0, init=False)
    _peak_threads: int = field(default=0, init=False)
    _sample_count: int = field(default=0, init=False)
    _monitor_start_wall_time: float | None = field(default=None, init=False)
    _monitor_start_monotonic: float | None = field(default=None, init=False)
    _monitor_end_monotonic: float | None = field(default=None, init=False)
    _cpu_stat_previous: _CPUStatSnapshot | None = field(default=None, init=False)
    _cpu_stat_deltas: dict[int, dict[str, int]] = field(
        default_factory=dict,
        init=False,
    )
    _cpu_stat_available: bool = field(default=True, init=False)
    _cpu_stat_hz: int | None = field(default=None, init=False)
    _monitor_start_boot_time_ticks: int | None = field(default=None, init=False)
    _pressure_previous: dict[str, _PressureSnapshot | None] = field(
        default_factory=dict,
        init=False,
    )
    _pressure_current: dict[str, _PressureSnapshot | None] = field(
        default_factory=dict,
        init=False,
    )
    _pressure_total_deltas: dict[str, dict[str, int | str]] = field(
        default_factory=dict,
        init=False,
    )
    _pressure_stall_available: dict[str, dict[str, bool]] = field(
        default_factory=lambda: {
            source: {stall: True for stall in _PRESSURE_STALLS} for source in _PRESSURE_SOURCES
        },
        init=False,
    )
    _pss_available: bool = field(default=True, init=False)
    _threads_available: bool = field(default=True, init=False)
    _affinity_available: bool = field(default=True, init=False)
    _affinity_union: set[int] = field(default_factory=set, init=False)
    _affinity_intersection: set[int] | None = field(default=None, init=False)
    _bounded_samples: dict[str, deque[float]] = field(
        default_factory=lambda: {
            name: deque(maxlen=_MAX_BOUNDED_SAMPLES)
            for name in (
                "aggregate_rss_bytes",
                "aggregate_pss_bytes",
                "worker_descendant_pss_bytes",
                "aggregate_user_cpu_seconds",
                "aggregate_system_cpu_seconds",
                "aggregate_processes",
                "aggregate_threads",
            )
        },
        init=False,
    )

    def __post_init__(self) -> None:
        if self.sample_interval_seconds <= 0.0:
            raise ValueError("process-tree sample interval must be positive")
        if self.thread_detail_interval_seconds <= 0.0:
            raise ValueError("thread-detail sample interval must be positive")
        if any(pid <= 0 for pid in self.additional_root_pids):
            raise ValueError("additional process-tree roots must be positive PIDs")
        if any(pid <= 0 for pid in self.excluded_root_pids):
            raise ValueError("excluded process-tree roots must be positive PIDs")
        if set(self.additional_root_pids) & set(self.excluded_root_pids):
            raise ValueError("a process-tree root cannot be both included and excluded")
        self._registered_additional_root_pids.update(self.additional_root_pids)

    def __enter__(self) -> ProcessTreeMonitor:
        if self._thread is not None:
            raise RuntimeError("process-tree monitor is already running")
        self._monitor_start_wall_time = time.time()
        self._monitor_start_monotonic = time.monotonic()
        try:
            self._cpu_stat_hz = int(os.sysconf("SC_CLK_TCK"))
        except (OSError, TypeError, ValueError, AttributeError):
            self._cpu_stat_hz = None
        self._monitor_start_boot_time_ticks = (
            _read_boot_time_ticks(self._cpu_stat_hz) if self._cpu_stat_hz is not None else None
        )
        self._sample(force_thread_details=True)
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
        self._sample(force_thread_details=True)
        self._monitor_end_monotonic = time.monotonic()

    def register_additional_root(self, pid: int) -> None:
        """Classify a newly started child as a shared root before worker launch."""

        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("additional process-tree root must be a positive PID")
        if pid == os.getpid() or pid in self.excluded_root_pids:
            raise ValueError("additional process-tree root conflicts with another root role")
        thread = self._thread
        if thread is None or not thread.is_alive():
            raise RuntimeError("additional process-tree roots require a running monitor")
        try:
            process = psutil.Process(pid)
        except _PROCESS_ERRORS as error:
            raise RuntimeError("additional process-tree root is not observable") from error
        if _process_identity(process) is None:
            raise RuntimeError("additional process-tree root identity is unavailable")
        with self._root_lock:
            if pid in self._registered_additional_root_pids:
                return
            self._registered_additional_root_pids.add(pid)
            self._worker_descendant_classification_epoch += 1
            # Samples recorded before this root existed used a different role
            # partition.  No worker has been submitted yet at the calibration
            # call site, so discard that incompatible descendant peak.
            self._peak_worker_descendant_pss_bytes = 0
            self._peak_worker_descendant_pss_sample_index = None
            self._peak_worker_descendant_pss_processes = ()
            self._worker_descendant_pss_available = True
            self._bounded_samples["worker_descendant_pss_bytes"].clear()

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

        observations = tuple(self._observations.values())
        user_cpu_seconds = sum(
            max(0.0, observation.last_user_cpu_seconds - observation.first_user_cpu_seconds)
            for observation in observations
        )
        system_cpu_seconds = sum(
            max(
                0.0,
                observation.last_system_cpu_seconds - observation.first_system_cpu_seconds,
            )
            for observation in observations
        )
        cpu_seconds = user_cpu_seconds + system_cpu_seconds
        monitor_elapsed = (
            None
            if self._monitor_start_monotonic is None or self._monitor_end_monotonic is None
            else max(0.0, self._monitor_end_monotonic - self._monitor_start_monotonic)
        )
        effective_elapsed = max(elapsed_seconds, monitor_elapsed or 0.0)
        allowed_cpu_seconds = effective_elapsed * compute_thread_limit
        tolerance_seconds = max(1e-6, allowed_cpu_seconds * 1e-6)
        if cpu_seconds > allowed_cpu_seconds + tolerance_seconds:
            raise RuntimeError(
                "process-tree CPU usage exceeds compute-thread limit: "
                f"{cpu_seconds:.9f}s > {allowed_cpu_seconds:.9f}s"
            )
        normalized_cpu_seconds = min(cpu_seconds, allowed_cpu_seconds)
        utilization_of_one_core = 100.0 * cpu_seconds / effective_elapsed
        utilization_of_limit = min(
            100.0,
            100.0 * normalized_cpu_seconds / allowed_cpu_seconds,
        )

        counter_totals = {
            name: _sum_optional(
                [
                    _counter_delta(
                        observation.first_counters.get(name),
                        observation.last_counters.get(name),
                    )
                    for observation in observations
                ]
            )
            for name in _COUNTER_NAMES
        }
        process_metrics: list[dict[str, object]] = []
        for identity, observation in sorted(self._observations.items()):
            deltas = {
                name: _counter_delta(
                    observation.first_counters.get(name),
                    observation.last_counters.get(name),
                )
                for name in _COUNTER_NAMES
            }
            reported_counters = {
                name: value if value is not None else _UNAVAILABLE for name, value in deltas.items()
            }
            process_metrics.append(
                {
                    "pid": identity[0],
                    "create_time": identity[1],
                    "sample_count": observation.sample_count,
                    "user_cpu_seconds": max(
                        0.0,
                        observation.last_user_cpu_seconds - observation.first_user_cpu_seconds,
                    ),
                    "system_cpu_seconds": max(
                        0.0,
                        observation.last_system_cpu_seconds - observation.first_system_cpu_seconds,
                    ),
                    "maximum_rss_bytes": observation.maximum_rss_bytes,
                    "maximum_pss_bytes": (
                        observation.maximum_pss_bytes
                        if observation.maximum_pss_bytes is not None
                        else _UNAVAILABLE
                    ),
                    "cpu_baseline_source": observation.cpu_baseline_source,
                    "counters": reported_counters,
                    "cpu_migrations": reported_counters["cpu_migrations"],
                    "schedstat": {
                        "runtime_ns": deltas["schedstat_runtime_ns"]
                        if deltas["schedstat_runtime_ns"] is not None
                        else _UNAVAILABLE,
                        "runqueue_delay_ns": deltas["schedstat_runqueue_delay_ns"]
                        if deltas["schedstat_runqueue_delay_ns"] is not None
                        else _UNAVAILABLE,
                        "timeslices": deltas["schedstat_timeslices"]
                        if deltas["schedstat_timeslices"] is not None
                        else _UNAVAILABLE,
                    },
                    "last_rss_bytes": observation.last_rss_bytes,
                    "last_pss_bytes": (
                        observation.last_pss_bytes
                        if observation.last_pss_bytes is not None
                        else _UNAVAILABLE
                    ),
                    "last_thread_count": (
                        observation.last_thread_count
                        if observation.last_thread_count is not None
                        else _UNAVAILABLE
                    ),
                    "last_affinity": (
                        list(observation.last_affinity)
                        if observation.last_affinity is not None
                        else _UNAVAILABLE
                    ),
                }
            )

        bounded_samples = {name: list(values) for name, values in self._bounded_samples.items()}
        sample_quantiles = {
            name: {
                "p50": _quantile(list(values), 0.50),
                "p95": _quantile(list(values), 0.95),
                "p99": _quantile(list(values), 0.99),
            }
            for name, values in self._bounded_samples.items()
        }
        pss_peak: int | str = (
            self._peak_aggregate_pss_bytes
            if self._pss_available and self._peak_aggregate_pss_bytes is not None
            else _UNAVAILABLE
        )
        worker_descendant_pss_peak: dict[str, object]
        if (
            self._worker_descendant_pss_available
            and self._peak_worker_descendant_pss_bytes > 0
            and self._peak_worker_descendant_pss_sample_index is not None
            and self._peak_worker_descendant_pss_processes
        ):
            worker_descendant_pss_peak = {
                "status": "available",
                "peak_bytes": self._peak_worker_descendant_pss_bytes,
                "sample_index": self._peak_worker_descendant_pss_sample_index,
                "processes": [
                    {
                        "pid": pid,
                        "create_time": create_time,
                        "pss_bytes": process_pss_bytes,
                    }
                    for pid, create_time, process_pss_bytes in (
                        self._peak_worker_descendant_pss_processes
                    )
                ],
            }
        else:
            worker_descendant_pss_peak = {
                "status": _UNAVAILABLE,
                "peak_bytes": _UNAVAILABLE,
                "sample_index": _UNAVAILABLE,
                "processes": _UNAVAILABLE,
            }
        threads_peak: int | str = self._peak_threads if self._threads_available else _UNAVAILABLE
        affinity_union: list[int] | str = (
            sorted(self._affinity_union) if self._affinity_available else _UNAVAILABLE
        )
        affinity_intersection: list[int] | str = (
            sorted(self._affinity_intersection or set())
            if self._affinity_available
            else _UNAVAILABLE
        )
        affinity_union_count: int | str = (
            len(self._affinity_union) if self._affinity_available else _UNAVAILABLE
        )
        affinity_intersection_count: int | str = (
            len(self._affinity_intersection or set()) if self._affinity_available else _UNAVAILABLE
        )

        voluntary = counter_totals["voluntary_context_switches"]
        involuntary = counter_totals["involuntary_context_switches"]
        migrations = counter_totals["cpu_migrations"]
        context_switches: int | str = (
            int(voluntary) + int(involuntary)
            if isinstance(voluntary, int) and isinstance(involuntary, int)
            else _UNAVAILABLE
        )
        cpu_stat = self._cpu_stat_statistics()
        pressure = self._pressure_statistics()
        thread_statistics = self._thread_statistics()
        thread_counters = thread_statistics["counters"]
        assert isinstance(thread_counters, dict)
        return {
            # Existing fields remain stable for historical readers.
            "sample_interval_seconds": self.sample_interval_seconds,
            "sample_count": self._sample_count,
            "observed_processes": len(self._observations),
            "observed_process_ids": sorted({identity[0] for identity in self._observations}),
            "peak_concurrent_processes": self._peak_processes,
            "peak_aggregate_threads": threads_peak,
            "peak_aggregate_rss_bytes": self._peak_aggregate_rss_bytes,
            "peak_aggregate_pss_bytes": pss_peak,
            "peak_worker_descendant_pss_bytes": worker_descendant_pss_peak[
                "peak_bytes"
            ],
            "worker_descendant_pss_peak": worker_descendant_pss_peak,
            "process_tree_cpu_seconds": cpu_seconds,
            "cpu_utilization_percent_of_one_core": utilization_of_one_core,
            "cpu_utilization_percent_of_compute_limit": utilization_of_limit,
            "root_process_id": os.getpid(),
            "additional_root_pids": sorted(self._registered_additional_root_pids),
            "excluded_root_pids": list(self.excluded_root_pids),
            # Extended trustworthy resource envelope.
            "monitor_elapsed_seconds": monitor_elapsed or effective_elapsed,
            "effective_elapsed_seconds": effective_elapsed,
            "monitor_start_wall_time": (
                self._monitor_start_wall_time
                if self._monitor_start_wall_time is not None
                else _UNAVAILABLE
            ),
            "monitor_start_monotonic": (
                self._monitor_start_monotonic
                if self._monitor_start_monotonic is not None
                else _UNAVAILABLE
            ),
            "monitor_start_boot_time_ticks": (
                self._monitor_start_boot_time_ticks
                if self._monitor_start_boot_time_ticks is not None
                else _UNAVAILABLE
            ),
            "monitor_end_monotonic": (
                self._monitor_end_monotonic
                if self._monitor_end_monotonic is not None
                else _UNAVAILABLE
            ),
            "compute_thread_limit": compute_thread_limit,
            "process_tree_user_cpu_seconds": user_cpu_seconds,
            "process_tree_system_cpu_seconds": system_cpu_seconds,
            "process_tree_cpu_user_seconds": user_cpu_seconds,
            "process_tree_cpu_system_seconds": system_cpu_seconds,
            "cpu_limit_tolerance_seconds": tolerance_seconds,
            "cpu_normalized_within_limit": True,
            "process_tree_voluntary_context_switches": voluntary,
            "process_tree_involuntary_context_switches": involuntary,
            "process_tree_context_switches": context_switches,
            "process_tree_minor_faults": counter_totals["minor_faults"],
            "process_tree_major_faults": counter_totals["major_faults"],
            "process_tree_read_bytes": counter_totals["read_bytes"],
            "process_tree_write_bytes": counter_totals["write_bytes"],
            "process_tree_cpu_migrations": migrations,
            "process_tree_migration_count": migrations,
            "process_tree_schedstat": {
                "runtime_ns": counter_totals["schedstat_runtime_ns"],
                "runqueue_delay_ns": counter_totals["schedstat_runqueue_delay_ns"],
                "timeslices": counter_totals["schedstat_timeslices"],
            },
            "actual_affinity_union": affinity_union,
            "actual_affinity_intersection": affinity_intersection,
            "cpu_affinity_union": affinity_union,
            "cpu_affinity_intersection": affinity_intersection,
            "actual_affinity_union_count": affinity_union_count,
            "actual_affinity_intersection_count": affinity_intersection_count,
            "affinity_status": "available" if self._affinity_available else _UNAVAILABLE,
            "process_metrics": process_metrics,
            "thread_tree_status": thread_statistics["status"],
            "thread_tree": thread_statistics,
            "thread_metrics": thread_statistics["threads"],
            "thread_detail_interval_seconds": self.thread_detail_interval_seconds,
            "thread_detail_sample_count": self._thread_detail_sample_count,
            "thread_discovery_sample_count": self._thread_discovery_sample_count,
            "thread_tree_user_cpu_seconds": thread_statistics["user_cpu_seconds"],
            "thread_tree_system_cpu_seconds": thread_statistics["system_cpu_seconds"],
            "thread_tree_context_switches": (
                int(thread_counters["voluntary_context_switches"])
                + int(thread_counters["involuntary_context_switches"])
                if isinstance(thread_counters["voluntary_context_switches"], int)
                and isinstance(thread_counters["involuntary_context_switches"], int)
                else _UNAVAILABLE
            ),
            "thread_tree_cpu_migrations": thread_counters["cpu_migrations"],
            "thread_tree_minor_faults": thread_counters["minor_faults"],
            "thread_tree_major_faults": thread_counters["major_faults"],
            "thread_tree_schedstat": {
                "runtime_ns": thread_counters["schedstat_runtime_ns"],
                "runqueue_delay_ns": thread_counters["schedstat_runqueue_delay_ns"],
                "timeslices": thread_counters["schedstat_timeslices"],
            },
            "thread_affinity_union": thread_statistics["affinity_union"],
            "thread_affinity_intersection": thread_statistics["affinity_intersection"],
            "bounded_sample_capacity": self._MAX_BOUNDED_SAMPLES,
            "bounded_samples": bounded_samples,
            "sample_quantiles": sample_quantiles,
            "resource_samples": bounded_samples,
            "resource_quantiles": sample_quantiles,
            "cpu_stat": cpu_stat,
            "cpu_stat_available": cpu_stat["status"] == "available",
            "cpu_stat_per_cpu": cpu_stat["per_cpu_delta_seconds"],
            "cpu_stat_aggregate": cpu_stat["aggregate_delta_seconds"],
            "pressure_stall_information": pressure,
            "psi": pressure,
        }

    def _run(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            self._sample()

    def _update_cpu_stat(self) -> None:
        current = _read_proc_cpu_stat()
        if current is None:
            self._cpu_stat_available = False
            return
        if self._cpu_stat_previous is None:
            self._cpu_stat_previous = current
            return
        delta = _cpu_stat_delta(self._cpu_stat_previous, current)
        self._cpu_stat_previous = current
        if delta is None:
            self._cpu_stat_available = False
            return
        for cpu_id, values in delta.items():
            totals = self._cpu_stat_deltas.setdefault(
                cpu_id,
                {field: 0 for field in _CPU_STAT_DELTA_FIELDS},
            )
            for metric in _CPU_STAT_DELTA_FIELDS:
                totals[metric] += values[metric]

    def _update_pressure(self) -> None:
        for source in _PRESSURE_SOURCES:
            current = _read_pressure_snapshot(source)
            self._pressure_current[source] = current
            previous = self._pressure_previous.get(source)
            if current is None:
                for stall in _PRESSURE_STALLS:
                    self._pressure_stall_available[source][stall] = False
                continue
            if previous is None:
                self._pressure_previous[source] = current
                for stall in _PRESSURE_STALLS:
                    fields = current.get(stall, {})
                    if not all(
                        isinstance(fields.get(key), (int, float))
                        for key in ("avg10", "avg60", "avg300", "total")
                    ):
                        self._pressure_stall_available[source][stall] = False
                continue
            for stall in _PRESSURE_STALLS:
                old_fields = previous.get(stall, {})
                new_fields = current.get(stall, {})
                old_total = old_fields.get("total")
                new_total = new_fields.get("total")
                if not isinstance(old_total, int) or not isinstance(new_total, int):
                    self._pressure_stall_available[source][stall] = False
                    continue
                if new_total < old_total:
                    self._pressure_stall_available[source][stall] = False
                    continue
                self._pressure_total_deltas.setdefault(source, {}).setdefault(stall, 0)
                self._pressure_total_deltas[source][stall] = (
                    int(self._pressure_total_deltas[source][stall]) + new_total - old_total
                )
            self._pressure_previous[source] = current

    def _cpu_stat_statistics(self) -> dict[str, object]:
        if (
            not self._cpu_stat_available
            or self._cpu_stat_hz is None
            or self._cpu_stat_previous is None
        ):
            return {
                "status": _UNAVAILABLE,
                "clock_ticks_per_second": _UNAVAILABLE,
                "per_cpu_delta_ticks": _UNAVAILABLE,
                "per_cpu_delta_seconds": _UNAVAILABLE,
                "aggregate_delta_ticks": _UNAVAILABLE,
                "aggregate_delta_seconds": _UNAVAILABLE,
            }
        per_cpu_ticks = {
            str(cpu_id): dict(values) for cpu_id, values in sorted(self._cpu_stat_deltas.items())
        }
        per_cpu_seconds = {
            cpu_id: {field: value / self._cpu_stat_hz for field, value in values.items()}
            for cpu_id, values in per_cpu_ticks.items()
        }
        aggregate_ticks = {
            field: sum(values[field] for values in self._cpu_stat_deltas.values())
            for field in _CPU_STAT_DELTA_FIELDS
        }
        aggregate_seconds = {
            field: value / self._cpu_stat_hz for field, value in aggregate_ticks.items()
        }
        complete = self._thread_sample_missed_processes == 0 and not self._unresolved_thread_ids
        return {
            "status": "available" if complete else "unavailable",
            "clock_ticks_per_second": self._cpu_stat_hz,
            "per_cpu_delta_ticks": per_cpu_ticks,
            "per_cpu_delta_seconds": per_cpu_seconds,
            "aggregate_delta_ticks": aggregate_ticks,
            "aggregate_delta_seconds": aggregate_seconds,
        }

    def _pressure_statistics(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for source in _PRESSURE_SOURCES:
            current = self._pressure_current.get(source)
            source_payload: dict[str, object] = {}
            for stall in _PRESSURE_STALLS:
                fields = {} if current is None else current.get(stall, {})
                snapshot: dict[str, float | int | str] = {
                    key: fields.get(key, _UNAVAILABLE)
                    for key in ("avg10", "avg60", "avg300", "total")
                }
                source_payload[stall] = {
                    "snapshot": snapshot,
                    "total_delta": (
                        self._pressure_total_deltas.get(source, {}).get(
                            stall,
                            _UNAVAILABLE,
                        )
                        if self._pressure_stall_available[source][stall]
                        else _UNAVAILABLE
                    ),
                    "status": (
                        "available"
                        if self._pressure_stall_available[source][stall]
                        else _UNAVAILABLE
                    ),
                }
            source_payload["status"] = (
                "available"
                if any(self._pressure_stall_available[source].values())
                else _UNAVAILABLE
            )
            result[source] = source_payload
        return result

    def _thread_statistics(self) -> dict[str, object]:
        if self._thread_observation_overflow:
            raise RuntimeError(
                "process-tree thread observation capacity exceeded; resource evidence is partial"
            )
        if not self._thread_metrics_available or not self._thread_observations:
            return {
                "status": _UNAVAILABLE,
                "observation_capacity": _MAX_THREAD_OBSERVATIONS,
                "observed_thread_identities": 0,
                "sample_missed_processes": self._thread_sample_missed_processes,
                "unresolved_thread_observation_count": len(self._unresolved_thread_ids),
                "unresolved_thread_ids": [
                    {"pid": pid, "process_create_time": create_time, "tid": tid}
                    for pid, create_time, tid in sorted(self._unresolved_thread_ids)
                ],
                "user_cpu_seconds": _UNAVAILABLE,
                "system_cpu_seconds": _UNAVAILABLE,
                "counters": {name: _UNAVAILABLE for name in _THREAD_COUNTER_NAMES},
                "affinity_union": _UNAVAILABLE,
                "affinity_intersection": _UNAVAILABLE,
                "threads": _UNAVAILABLE,
            }
        rows: list[dict[str, object]] = []
        counter_totals = {name: 0 for name in _THREAD_COUNTER_NAMES}
        user_cpu_seconds = 0.0
        system_cpu_seconds = 0.0
        for identity, observation in sorted(self._thread_observations.items()):
            user_delta = max(
                0.0,
                observation.last_user_cpu_seconds - observation.first_user_cpu_seconds,
            )
            system_delta = max(
                0.0,
                observation.last_system_cpu_seconds - observation.first_system_cpu_seconds,
            )
            counters = {
                name: max(
                    0,
                    observation.last_counters[name] - observation.first_counters[name],
                )
                for name in _THREAD_COUNTER_NAMES
            }
            user_cpu_seconds += user_delta
            system_cpu_seconds += system_delta
            for name, value in counters.items():
                counter_totals[name] += value
            rows.append(
                {
                    "pid": identity[0],
                    "process_create_time": identity[1],
                    "tid": identity[2],
                    "thread_start_time_ticks": identity[3],
                    "sample_count": observation.sample_count,
                    "cpu_baseline_source": observation.cpu_baseline_source,
                    "user_cpu_seconds": user_delta,
                    "system_cpu_seconds": system_delta,
                    "counters": counters,
                    "last_affinity": list(observation.last_affinity),
                }
            )
        return {
            "status": "available",
            "observation_capacity": _MAX_THREAD_OBSERVATIONS,
            "observed_thread_identities": len(rows),
            "sample_missed_processes": self._thread_sample_missed_processes,
            "unresolved_thread_observation_count": len(self._unresolved_thread_ids),
            "unresolved_thread_ids": [
                {"pid": pid, "process_create_time": create_time, "tid": tid}
                for pid, create_time, tid in sorted(self._unresolved_thread_ids)
            ],
            "user_cpu_seconds": user_cpu_seconds,
            "system_cpu_seconds": system_cpu_seconds,
            "counters": counter_totals,
            "affinity_union": sorted(self._thread_affinity_union),
            "affinity_intersection": sorted(self._thread_affinity_intersection or set()),
            "threads": rows,
        }

    def _sample_process_threads(
        self,
        process_identity: tuple[int, float],
        *,
        detailed: bool,
    ) -> None:
        if self._cpu_stat_hz is None:
            self._thread_metrics_available = False
            return
        tids, overflow = _list_process_thread_ids(process_identity[0])
        if overflow:
            self._thread_observation_overflow = True
            return
        if not tids:
            self._thread_sample_missed_processes += 1
            return
        prior_tids = self._active_thread_ids.get(process_identity, set())
        prior_known_tids = self._known_thread_ids.get(process_identity)
        sample_tids = tids if detailed else tuple(tid for tid in tids if tid not in prior_tids)
        if not sample_tids:
            self._active_thread_ids[process_identity] = prior_tids & set(tids)
            self._known_thread_ids[process_identity] = set(tids)
            return
        samples, overflow = _read_process_threads(
            process_identity[0],
            self._cpu_stat_hz,
            tids=sample_tids,
        )
        if overflow:
            self._thread_observation_overflow = True
            return
        successful_tids = {tid for tid, _start_time_ticks in samples}
        self._active_thread_ids[process_identity] = (prior_tids & set(tids)) | successful_tids
        self._known_thread_ids[process_identity] = set(tids)
        for tid in sample_tids:
            unresolved_identity = (process_identity[0], process_identity[1], tid)
            if tid in successful_tids:
                self._unresolved_thread_ids.discard(unresolved_identity)
            else:
                self._unresolved_thread_ids.add(unresolved_identity)
        for (tid, start_time_ticks), sample in samples.items():
            identity = (
                process_identity[0],
                process_identity[1],
                tid,
                start_time_ticks,
            )
            observation = self._thread_observations.get(identity)
            known_start_ticks = self._known_thread_start_ticks.get(
                (process_identity[0], process_identity[1], tid)
            )
            created_during_monitor = (
                (prior_known_tids is not None and tid not in prior_known_tids)
                or (known_start_ticks is not None and known_start_ticks != start_time_ticks)
                or (
                    prior_known_tids is None
                    and self._monitor_start_boot_time_ticks is not None
                    and start_time_ticks > self._monitor_start_boot_time_ticks
                )
            )
            if observation is None:
                if len(self._thread_observations) >= _MAX_THREAD_OBSERVATIONS:
                    self._thread_observation_overflow = True
                    return
                baseline_user = 0.0 if created_during_monitor else sample.user_cpu_seconds
                baseline_system = 0.0 if created_during_monitor else sample.system_cpu_seconds
                baseline_counters = {
                    name: (0 if created_during_monitor else sample.counters[name])
                    for name in _THREAD_COUNTER_NAMES
                }
                self._thread_observations[identity] = _ThreadObservation(
                    first_user_cpu_seconds=baseline_user,
                    last_user_cpu_seconds=sample.user_cpu_seconds,
                    first_system_cpu_seconds=baseline_system,
                    last_system_cpu_seconds=sample.system_cpu_seconds,
                    first_counters=baseline_counters,
                    last_counters=dict(sample.counters),
                    last_affinity=sample.affinity,
                    cpu_baseline_source=(
                        "thread_start" if created_during_monitor else "monitor_start"
                    ),
                )
            else:
                observation.last_user_cpu_seconds = sample.user_cpu_seconds
                observation.last_system_cpu_seconds = sample.system_cpu_seconds
                observation.last_counters = dict(sample.counters)
                observation.last_affinity = sample.affinity
                observation.sample_count += 1
            self._known_thread_start_ticks[(process_identity[0], process_identity[1], tid)] = (
                start_time_ticks
            )
            self._thread_affinity_union.update(sample.affinity)
            if self._thread_affinity_intersection is None:
                self._thread_affinity_intersection = set(sample.affinity)
            else:
                self._thread_affinity_intersection.intersection_update(sample.affinity)

    def _sample(self, *, force_thread_details: bool = False) -> None:
        sample_monotonic = time.monotonic()
        detailed_threads = (
            force_thread_details
            or self._last_thread_detail_monotonic is None
            or sample_monotonic - self._last_thread_detail_monotonic
            >= self.thread_detail_interval_seconds
        )
        if detailed_threads:
            self._last_thread_detail_monotonic = sample_monotonic
            self._thread_detail_sample_count += 1
        else:
            self._thread_discovery_sample_count += 1
        self._update_cpu_stat()
        self._update_pressure()
        with self._root_lock:
            additional_root_pids = tuple(sorted(self._registered_additional_root_pids))
            classification_epoch = self._worker_descendant_classification_epoch
        processes: dict[tuple[int, float], psutil.Process] = {}
        excluded: set[tuple[int, float]] = set()
        for pid in self.excluded_root_pids:
            try:
                root = psutil.Process(pid)
                candidates = (root, *root.children(recursive=True))
            except _PROCESS_ERRORS:
                continue
            for process in candidates:
                identity = _process_identity(process)
                if identity is not None:
                    excluded.add(identity)
        for pid in (os.getpid(), *additional_root_pids):
            try:
                root = psutil.Process(pid)
                candidates = (root, *root.children(recursive=True))
            except _PROCESS_ERRORS:
                continue
            for process in candidates:
                identity = _process_identity(process)
                if identity is not None and identity not in excluded:
                    processes[identity] = process

        aggregate_rss = 0
        aggregate_pss = 0
        aggregate_pss_complete = True
        aggregate_threads = 0
        aggregate_threads_complete = True
        aggregate_user_cpu = 0.0
        aggregate_system_cpu = 0.0
        aggregate_processes = 0
        process_root_pids = {os.getpid(), *additional_root_pids}
        worker_descendant_pss_complete = True
        worker_descendant_pss_processes: list[tuple[int, float, int]] = []
        for identity, process in processes.items():
            sample = _read_process_sample(process)
            if sample is None:
                # A short-lived spawn worker may exit between recursive
                # discovery and the per-process read.  It is no longer part of
                # this simultaneous sample; only a still-identical live
                # process makes the sample incomplete.
                if (
                    identity[0] not in process_root_pids
                    and _process_identity(process) == identity
                ):
                    worker_descendant_pss_complete = False
                continue
            aggregate_processes += 1
            aggregate_rss += sample.rss_bytes
            aggregate_user_cpu += sample.user_cpu_seconds
            aggregate_system_cpu += sample.system_cpu_seconds
            if sample.pss_bytes is None:
                aggregate_pss_complete = False
            else:
                aggregate_pss += sample.pss_bytes
            if identity[0] not in process_root_pids:
                if sample.pss_bytes is None:
                    worker_descendant_pss_complete = False
                else:
                    worker_descendant_pss_processes.append(
                        (identity[0], identity[1], sample.pss_bytes)
                    )
            if sample.thread_count is None:
                aggregate_threads_complete = False
            else:
                aggregate_threads += sample.thread_count
            if sample.affinity is None:
                self._affinity_available = False
            else:
                self._affinity_union.update(sample.affinity)
                if self._affinity_intersection is None:
                    self._affinity_intersection = set(sample.affinity)
                else:
                    self._affinity_intersection.intersection_update(sample.affinity)

            observation = self._observations.get(identity)
            created_during_monitor = (
                self._monitor_start_wall_time is not None
                and identity[1] >= self._monitor_start_wall_time
            )
            if observation is None:
                baseline_source = (
                    "process_create_time" if created_during_monitor else "monitor_start"
                )
                baseline_user = 0.0 if created_during_monitor else sample.user_cpu_seconds
                baseline_system = 0.0 if created_during_monitor else sample.system_cpu_seconds
                baseline_counters = {
                    name: (0 if created_during_monitor else value)
                    for name, value in sample.counters.items()
                }
                self._observations[identity] = _ProcessObservation(
                    first_user_cpu_seconds=baseline_user,
                    last_user_cpu_seconds=sample.user_cpu_seconds,
                    first_system_cpu_seconds=baseline_system,
                    last_system_cpu_seconds=sample.system_cpu_seconds,
                    first_counters=baseline_counters,
                    last_counters=dict(sample.counters),
                    maximum_rss_bytes=sample.rss_bytes,
                    maximum_pss_bytes=sample.pss_bytes,
                    last_rss_bytes=sample.rss_bytes,
                    last_pss_bytes=sample.pss_bytes,
                    last_thread_count=sample.thread_count,
                    last_affinity=sample.affinity,
                    cpu_baseline_source=baseline_source,
                )
            else:
                observation.last_user_cpu_seconds = sample.user_cpu_seconds
                observation.last_system_cpu_seconds = sample.system_cpu_seconds
                observation.last_counters = dict(sample.counters)
                observation.maximum_rss_bytes = max(
                    observation.maximum_rss_bytes,
                    sample.rss_bytes,
                )
                if sample.pss_bytes is not None:
                    observation.maximum_pss_bytes = max(
                        observation.maximum_pss_bytes or 0,
                        sample.pss_bytes,
                    )
                observation.last_rss_bytes = sample.rss_bytes
                observation.last_pss_bytes = sample.pss_bytes
                observation.last_thread_count = sample.thread_count
                observation.last_affinity = sample.affinity
                observation.sample_count += 1
            self._sample_process_threads(identity, detailed=detailed_threads)

        if aggregate_processes == 0:
            aggregate_pss_complete = False
            aggregate_threads_complete = False
            self._affinity_available = False
        self._pss_available = self._pss_available and aggregate_pss_complete
        self._record_worker_descendant_pss_sample(
            worker_descendant_pss_processes,
            complete=worker_descendant_pss_complete,
            classification_epoch=classification_epoch,
        )
        self._threads_available = self._threads_available and aggregate_threads_complete
        if aggregate_pss_complete:
            self._peak_aggregate_pss_bytes = max(
                self._peak_aggregate_pss_bytes or 0,
                aggregate_pss,
            )
        self._peak_aggregate_rss_bytes = max(
            self._peak_aggregate_rss_bytes,
            aggregate_rss,
        )
        self._peak_processes = max(self._peak_processes, aggregate_processes)
        if aggregate_threads_complete:
            self._peak_threads = max(self._peak_threads, aggregate_threads)
        self._bounded_samples["aggregate_rss_bytes"].append(float(aggregate_rss))
        if aggregate_pss_complete:
            self._bounded_samples["aggregate_pss_bytes"].append(float(aggregate_pss))
        self._bounded_samples["aggregate_user_cpu_seconds"].append(aggregate_user_cpu)
        self._bounded_samples["aggregate_system_cpu_seconds"].append(aggregate_system_cpu)
        self._bounded_samples["aggregate_processes"].append(float(aggregate_processes))
        if aggregate_threads_complete:
            self._bounded_samples["aggregate_threads"].append(float(aggregate_threads))
        self._sample_count += 1

    def _record_worker_descendant_pss_sample(
        self,
        processes: list[tuple[int, float, int]],
        *,
        complete: bool,
        classification_epoch: int | None = None,
    ) -> None:
        with self._root_lock:
            if (
                classification_epoch is not None
                and classification_epoch != self._worker_descendant_classification_epoch
            ):
                return
            if not complete:
                self._worker_descendant_pss_available = False
                return
            normalized = tuple(sorted(processes))
            if any(pss_bytes <= 0 for _pid, _create_time, pss_bytes in normalized):
                self._worker_descendant_pss_available = False
                return
            aggregate = sum(pss_bytes for _pid, _create_time, pss_bytes in normalized)
            self._bounded_samples["worker_descendant_pss_bytes"].append(float(aggregate))
            if aggregate > self._peak_worker_descendant_pss_bytes:
                self._peak_worker_descendant_pss_bytes = aggregate
                self._peak_worker_descendant_pss_sample_index = self._sample_count
                self._peak_worker_descendant_pss_processes = normalized


__all__ = ("ProcessTreeMonitor",)
