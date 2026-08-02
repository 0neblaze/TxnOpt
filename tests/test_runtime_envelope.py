from __future__ import annotations

import time

from evrptw.runtime_envelope import ProcessTreeMonitor


def test_process_tree_monitor_records_complete_local_envelope() -> None:
    with ProcessTreeMonitor(sample_interval_seconds=0.005) as monitor:
        started = time.perf_counter()
        sum(index * index for index in range(100_000))
        elapsed = time.perf_counter() - started

    statistics = monitor.statistics(
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


def test_process_tree_monitor_rejects_invalid_external_root() -> None:
    try:
        ProcessTreeMonitor(additional_root_pids=(0,))
    except ValueError as error:
        assert "positive PIDs" in str(error)
    else:
        raise AssertionError("invalid external process root was accepted")
