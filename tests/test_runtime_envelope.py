from __future__ import annotations

import dataclasses
import subprocess
import sys
import threading
import time
from typing import Any, cast

import pytest

import evrptw.runtime_envelope as runtime_envelope
from evrptw.runtime_envelope import ProcessTreeMonitor


def _statistics(
    monitor: ProcessTreeMonitor,
    *,
    elapsed_seconds: float,
    compute_thread_limit: int,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        monitor.statistics(
            elapsed_seconds=elapsed_seconds,
            compute_thread_limit=compute_thread_limit,
        ),
    )


def test_worker_descendant_pss_peak_does_not_sum_sequential_generations() -> None:
    monitor = ProcessTreeMonitor()
    monitor._record_worker_descendant_pss_sample(  # noqa: SLF001
        [(11, 2.0, 80)],
        complete=True,
    )
    monitor._sample_count += 1  # noqa: SLF001
    monitor._record_worker_descendant_pss_sample(  # noqa: SLF001
        [(12, 3.0, 80)],
        complete=True,
    )

    assert monitor._peak_worker_descendant_pss_bytes == 80  # noqa: SLF001
    assert monitor._peak_worker_descendant_pss_processes == (  # noqa: SLF001
        (11, 2.0, 80),
    )


def test_process_tree_monitor_records_complete_local_envelope() -> None:
    with ProcessTreeMonitor(sample_interval_seconds=0.005) as monitor:
        started = time.perf_counter()
        sum(index * index for index in range(100_000))
        elapsed = time.perf_counter() - started

    statistics = _statistics(
        monitor,
        elapsed_seconds=max(elapsed, 1e-6),
        compute_thread_limit=24,
    )

    assert statistics["sample_count"] >= 2
    assert statistics["observed_processes"] >= 1
    assert statistics["peak_concurrent_processes"] >= 1
    assert statistics["peak_aggregate_threads"] >= 1
    assert statistics["peak_aggregate_rss_bytes"] > 0
    assert statistics["peak_aggregate_pss_bytes"] > 0
    assert statistics["process_tree_cpu_seconds"] >= 0.0
    assert statistics["process_tree_user_cpu_seconds"] >= 0.0
    assert statistics["process_tree_system_cpu_seconds"] >= 0.0
    assert statistics["cpu_utilization_percent_of_compute_limit"] <= 100.0
    assert statistics["actual_affinity_union"] != "unavailable"
    assert statistics["actual_affinity_intersection"] != "unavailable"
    assert isinstance(statistics["sample_quantiles"], dict)
    assert len(statistics["bounded_samples"]["aggregate_rss_bytes"]) <= 256
    assert isinstance(statistics["process_tree_schedstat"], dict)
    cpu_stat = statistics["cpu_stat"]
    assert isinstance(cpu_stat, dict)
    assert cpu_stat["status"] in {"available", "unavailable"}
    pressure = statistics["psi"]
    assert isinstance(pressure, dict)
    assert set(pressure) == {"cpu", "memory", "io"}
    assert isinstance(statistics["process_tree_cpu_migrations"], (int, str))
    assert statistics["thread_tree_status"] == "available"
    assert statistics["thread_metrics"]
    assert statistics["thread_tree_schedstat"]["runqueue_delay_ns"] >= 0
    assert statistics["thread_affinity_union"]
    assert statistics["thread_detail_interval_seconds"] == 0.25
    assert statistics["thread_detail_sample_count"] >= 2
    assert statistics["thread_discovery_sample_count"] >= 0


def test_process_tree_monitor_tracks_late_thread_identity_and_scheduler_counters() -> None:
    def work() -> None:
        deadline = time.perf_counter() + 0.05
        total = 0
        while time.perf_counter() < deadline:
            total += 1
        assert total > 0

    with ProcessTreeMonitor(sample_interval_seconds=0.002) as monitor:
        worker = threading.Thread(target=work, name="stage052-thread-telemetry-test")
        worker.start()
        worker.join(timeout=2.0)
        assert not worker.is_alive()

    statistics = _statistics(
        monitor,
        elapsed_seconds=0.06,
        compute_thread_limit=24,
    )
    rows = statistics["thread_metrics"]
    assert isinstance(rows, list)
    assert any(row["cpu_baseline_source"] == "thread_start" for row in rows)
    assert statistics["thread_tree_cpu_migrations"] >= 0
    assert statistics["thread_tree_context_switches"] >= 0


def test_thread_first_read_failure_retries_without_historical_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = ProcessTreeMonitor()
    monitor._cpu_stat_hz = 100
    monitor._monitor_start_boot_time_ticks = 1_000
    monitor._sample_count = 2
    process_identity = (123, 10.0)
    counters = {name: 20 for name in runtime_envelope._THREAD_COUNTER_NAMES}
    later_counters = {name: 23 for name in runtime_envelope._THREAD_COUNTER_NAMES}
    snapshots = [
        ({}, False),
        (
            {
                (7, 900): runtime_envelope._ThreadSample(
                    start_time_ticks=900,
                    user_cpu_seconds=4.0,
                    system_cpu_seconds=1.0,
                    counters=counters,
                    affinity=(0,),
                )
            },
            False,
        ),
        (
            {
                (7, 900): runtime_envelope._ThreadSample(
                    start_time_ticks=900,
                    user_cpu_seconds=4.25,
                    system_cpu_seconds=1.05,
                    counters=later_counters,
                    affinity=(0,),
                )
            },
            False,
        ),
    ]
    monkeypatch.setattr(
        runtime_envelope,
        "_list_process_thread_ids",
        lambda _pid: ((7,), False),
    )
    monkeypatch.setattr(
        runtime_envelope,
        "_read_process_threads",
        lambda _pid, _hz, *, tids: snapshots.pop(0),
    )

    monitor._sample_process_threads(process_identity, detailed=False)
    assert monitor._active_thread_ids[process_identity] == set()
    assert monitor._unresolved_thread_ids == {(123, 10.0, 7)}
    monitor._sample_process_threads(process_identity, detailed=False)
    monitor._sample_process_threads(process_identity, detailed=True)

    thread_tree = monitor._thread_statistics()
    rows = cast(list[dict[str, Any]], thread_tree["threads"])
    assert thread_tree["unresolved_thread_observation_count"] == 0
    assert len(rows) == 1
    assert rows[0]["cpu_baseline_source"] == "monitor_start"
    assert rows[0]["user_cpu_seconds"] == pytest.approx(0.25)
    assert rows[0]["system_cpu_seconds"] == pytest.approx(0.05)
    assert rows[0]["counters"] == {name: 3 for name in runtime_envelope._THREAD_COUNTER_NAMES}


def test_thread_tid_reuse_uses_start_ticks_for_a_new_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = ProcessTreeMonitor()
    monitor._cpu_stat_hz = 100
    monitor._monitor_start_boot_time_ticks = 1_000
    process_identity = (123, 10.0)
    counters = {name: 1 for name in runtime_envelope._THREAD_COUNTER_NAMES}
    snapshots = [
        (
            {
                (7, 900): runtime_envelope._ThreadSample(
                    start_time_ticks=900,
                    user_cpu_seconds=4.0,
                    system_cpu_seconds=1.0,
                    counters=counters,
                    affinity=(0,),
                )
            },
            False,
        ),
        (
            {
                (7, 1_100): runtime_envelope._ThreadSample(
                    start_time_ticks=1_100,
                    user_cpu_seconds=0.2,
                    system_cpu_seconds=0.1,
                    counters=counters,
                    affinity=(1,),
                )
            },
            False,
        ),
    ]
    monkeypatch.setattr(
        runtime_envelope,
        "_list_process_thread_ids",
        lambda _pid: ((7,), False),
    )
    monkeypatch.setattr(
        runtime_envelope,
        "_read_process_threads",
        lambda _pid, _hz, *, tids: snapshots.pop(0),
    )

    monitor._sample_process_threads(process_identity, detailed=True)
    monitor._sample_process_threads(process_identity, detailed=True)

    rows = cast(list[dict[str, Any]], monitor._thread_statistics()["threads"])
    assert len(rows) == 2
    assert rows[0]["cpu_baseline_source"] == "monitor_start"
    assert rows[1]["cpu_baseline_source"] == "thread_start"
    assert rows[1]["user_cpu_seconds"] == pytest.approx(0.2)


def test_process_tree_monitor_rejects_bounded_thread_receipt_overflow() -> None:
    with ProcessTreeMonitor(sample_interval_seconds=0.002) as monitor:
        time.sleep(0.01)
    monitor._thread_observation_overflow = True
    with pytest.raises(RuntimeError, match="thread observation capacity exceeded"):
        monitor.statistics(elapsed_seconds=0.01, compute_thread_limit=24)


def test_process_tree_monitor_rejects_invalid_external_root() -> None:
    try:
        ProcessTreeMonitor(additional_root_pids=(0,))
    except ValueError as error:
        assert "positive PIDs" in str(error)
    else:
        raise AssertionError("invalid external process root was accepted")


def test_process_tree_monitor_registers_late_shared_root_before_worker_samples() -> None:
    shared_root = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.0)"])
    worker: subprocess.Popen[bytes] | None = None
    try:
        with ProcessTreeMonitor(sample_interval_seconds=0.002) as monitor:
            monitor.register_additional_root(shared_root.pid)
            worker = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(1.0)"]
            )
            time.sleep(0.02)
        statistics = _statistics(
            monitor,
            elapsed_seconds=0.02,
            compute_thread_limit=24,
        )
        assert statistics["additional_root_pids"] == [shared_root.pid]
        peak = statistics["worker_descendant_pss_peak"]
        assert peak["status"] == "available"
        assert {row["pid"] for row in peak["processes"]} == {worker.pid}
    finally:
        if worker is not None:
            worker.terminate()
            worker.wait(timeout=5.0)
        shared_root.terminate()
        shared_root.wait(timeout=5.0)


def test_process_tree_monitor_rejects_invalid_thread_detail_interval() -> None:
    with pytest.raises(ValueError, match="thread-detail sample interval"):
        ProcessTreeMonitor(thread_detail_interval_seconds=0.0)


def test_process_tree_monitor_counts_cpu_from_descendant_first_seen_after_entry() -> None:
    with ProcessTreeMonitor(sample_interval_seconds=0.002) as monitor:
        started = time.perf_counter()
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "sum(index * index for index in range(8_000_000))",
            ]
        )
        assert child.wait(timeout=10.0) == 0
        elapsed = time.perf_counter() - started

    statistics = _statistics(
        monitor,
        elapsed_seconds=elapsed,
        compute_thread_limit=24,
    )
    assert statistics["observed_processes"] >= 2
    assert statistics["process_tree_cpu_seconds"] > 0.0
    child_metrics = [
        metric for metric in statistics["process_metrics"] if metric["pid"] == child.pid
    ]
    assert child_metrics


def test_process_tree_monitor_baselines_preexisting_process_at_monitor_start() -> None:
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import sys,time\n"
                "deadline=time.time()+0.15\n"
                "sum_value=0\n"
                "while time.time()<deadline: sum_value+=1\n"
                "print('ready', flush=True)\n"
                "sys.stdin.read(1)\n"
                "deadline=time.time()+0.15\n"
                "while time.time()<deadline: sum_value+=1\n"
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        with ProcessTreeMonitor(
            additional_root_pids=(child.pid,),
            sample_interval_seconds=0.002,
        ) as monitor:
            assert child.stdin is not None
            child.stdin.write("x")
            child.stdin.flush()
            time.sleep(0.10)
        elapsed = 0.10
        statistics = _statistics(
            monitor,
            elapsed_seconds=elapsed,
            compute_thread_limit=24,
        )
        child_metrics = [
            metric for metric in statistics["process_metrics"] if metric["pid"] == child.pid
        ]
        assert child_metrics
        assert child_metrics[0]["cpu_baseline_source"] == "monitor_start"
        assert child_metrics[0]["user_cpu_seconds"] >= 0.0
    finally:
        if child.stdin is not None:
            child.stdin.close()
        child.terminate()
        child.wait(timeout=5.0)


def test_process_tree_monitor_fails_fast_when_cpu_exceeds_limit() -> None:
    with ProcessTreeMonitor(sample_interval_seconds=0.002) as monitor:
        time.sleep(0.02)
    observation = next(iter(monitor._observations.values()))
    observation.last_user_cpu_seconds += 1.0
    with pytest.raises(RuntimeError, match="exceeds compute-thread limit"):
        monitor.statistics(elapsed_seconds=0.001, compute_thread_limit=1)


def test_process_tree_monitor_marks_optional_pss_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runtime_envelope._read_process_sample

    def without_pss(process: object) -> object:
        sample = original(process)  # type: ignore[arg-type]
        if sample is None:
            return None
        return dataclasses.replace(sample, pss_bytes=None)

    monkeypatch.setattr(runtime_envelope, "_read_process_sample", without_pss)
    with ProcessTreeMonitor(sample_interval_seconds=0.002) as monitor:
        time.sleep(0.02)
    statistics = _statistics(
        monitor,
        elapsed_seconds=0.02,
        compute_thread_limit=24,
    )
    assert statistics["peak_aggregate_pss_bytes"] == "unavailable"
    assert statistics["bounded_samples"]["aggregate_pss_bytes"] == []


def test_process_tree_monitor_marks_cpu_stat_and_psi_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_envelope, "_read_proc_cpu_stat", lambda: None)
    monkeypatch.setattr(runtime_envelope, "_read_pressure_snapshot", lambda source: None)
    with ProcessTreeMonitor(sample_interval_seconds=0.002) as monitor:
        time.sleep(0.02)
    statistics = _statistics(
        monitor,
        elapsed_seconds=0.02,
        compute_thread_limit=24,
    )
    assert statistics["cpu_stat"]["status"] == "unavailable"
    for source in ("cpu", "memory", "io"):
        assert statistics["psi"][source]["status"] == "unavailable"


def test_process_tree_monitor_reports_per_cpu_stat_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline: dict[int, tuple[int, ...]] = {
        0: (10, 2, 3, 100, 4, 5, 6, 7, 0),
        1: (20, 0, 1, 100, 0, 0, 0, 0, 0),
    }
    updated: dict[int, tuple[int, ...]] = {
        0: (15, 3, 5, 101, 5, 6, 7, 8, 0),
        1: (21, 0, 2, 100, 0, 0, 0, 0, 0),
    }
    calls = 0

    def fake_cpu_stat() -> dict[int, tuple[int, ...]]:
        nonlocal calls
        calls += 1
        return baseline if calls == 1 else updated

    monkeypatch.setattr(runtime_envelope, "_read_proc_cpu_stat", fake_cpu_stat)
    with ProcessTreeMonitor(sample_interval_seconds=0.002) as monitor:
        time.sleep(0.01)
    statistics = _statistics(
        monitor,
        elapsed_seconds=0.01,
        compute_thread_limit=24,
    )
    cpu_stat = statistics["cpu_stat"]
    assert cpu_stat["status"] == "available"
    assert cpu_stat["per_cpu_delta_ticks"]["0"] == {
        "busy": 11,
        "user": 6,
        "system": 2,
        "iowait": 1,
        "steal": 1,
    }
    assert cpu_stat["aggregate_delta_ticks"]["busy"] == 13


def test_process_tree_monitor_reports_psi_snapshot_and_total_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = {
        "some": {"avg10": 0.1, "avg60": 0.2, "avg300": 0.3, "total": 10},
        "full": {"avg10": 0.0, "avg60": 0.0, "avg300": 0.0, "total": 2},
    }
    updated = {
        "some": {"avg10": 0.4, "avg60": 0.5, "avg300": 0.6, "total": 15},
        "full": {"avg10": 0.1, "avg60": 0.1, "avg300": 0.1, "total": 4},
    }
    calls = {source: 0 for source in ("cpu", "memory", "io")}

    def fake_pressure(source: str) -> dict[str, dict[str, float | int]]:
        calls[source] += 1
        return baseline if calls[source] == 1 else updated

    monkeypatch.setattr(runtime_envelope, "_read_pressure_snapshot", fake_pressure)
    with ProcessTreeMonitor(sample_interval_seconds=0.002) as monitor:
        time.sleep(0.01)
    statistics = _statistics(
        monitor,
        elapsed_seconds=0.01,
        compute_thread_limit=24,
    )
    some = statistics["psi"]["cpu"]["some"]
    assert some["status"] == "available"
    assert some["snapshot"]["avg10"] == 0.4
    assert some["total_delta"] == 5


def test_process_tree_monitor_can_exclude_a_persistent_child_root() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1.0)"])
    try:
        with ProcessTreeMonitor(
            excluded_root_pids=(child.pid,),
            sample_interval_seconds=0.005,
        ) as monitor:
            time.sleep(0.02)
        statistics = _statistics(
            monitor,
            elapsed_seconds=0.02,
            compute_thread_limit=24,
        )
        assert child.pid not in statistics["observed_process_ids"]
        assert statistics["excluded_root_pids"] == [child.pid]
    finally:
        child.terminate()
        child.wait(timeout=5.0)
