from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import evrptw.experiments.stage052_performance_observation as observation_module
from evrptw.experiments.stage052_performance_observation import (
    BlockExecution,
    PerformanceObservationError,
    _atomic_signed_json,
    _local_work_pool_receipt,
    _memory_admission,
    _parallel_diagnostics,
    _replay_identical,
    _require_live_host_identity,
    _resource_summary,
    _signed_axis,
    _worker_descendant_peak_pss_bytes,
    compare_host_scheduler_lifecycles,
)
from evrptw.native_scheduler import NativeHostScheduler
from evrptw.runtime_envelope import ProcessTreeMonitor
from evrptw.stage052_performance import (
    ExecutionTopology,
    HostPerformanceEnvelope,
    RuntimeResourceSummaryV2,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _topology(*, shards: int = 2) -> ExecutionTopology:
    return ExecutionTopology(
        workload_class="100-customer",
        shards=tuple((index,) for index in range(shards)),
        worker_count=shards,
        request_threads=shards,
    )


def _host(*, available: int = 10_000) -> HostPerformanceEnvelope:
    return HostPerformanceEnvelope(
        allowed_cpu_ids=(0, 1, 2, 3),
        physical_core_groups=((0,), (1,), (2,), (3,)),
        memory_total_bytes=available,
        memory_available_bytes=available,
        topology_source="provided",
    )


def test_live_host_identity_allows_dynamic_available_memory_to_decrease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _host()
    observed = replace(
        frozen,
        memory_available_bytes=frozen.memory_available_bytes - 1,
        memory_current_bytes=1,
    )
    monkeypatch.setattr(observation_module, "detect_host_performance", lambda: observed)

    _require_live_host_identity(frozen)


def test_live_host_identity_rejects_stable_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _host()
    observed = replace(frozen, memory_total_bytes=frozen.memory_total_bytes + 1)
    monkeypatch.setattr(observation_module, "detect_host_performance", lambda: observed)

    with pytest.raises(PerformanceObservationError, match="live host identity differs"):
        _require_live_host_identity(frozen)


def test_live_host_identity_rejects_actual_swap_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen = _host()
    observed = replace(
        frozen,
        swap_total_bytes=1,
        swap_used_bytes=1,
        swap_current_bytes=1,
    )
    monkeypatch.setattr(observation_module, "detect_host_performance", lambda: observed)

    with pytest.raises(PerformanceObservationError, match="already uses swap"):
        _require_live_host_identity(frozen)


def test_live_memory_admission_uses_incremental_pss_without_startup_self_race() -> None:
    frozen = _host()
    live = replace(frozen, memory_available_bytes=frozen.memory_available_bytes - 1)

    admission, projected = _memory_admission(
        frozen,
        topology=_topology(shards=4),
        isolated_pss_bytes=1_500,
        scheduler_pss_bytes=500,
        producer_pss_bytes=1_000,
        worker_descendant_pss_bytes=100,
        live_host=live,
        resident_pss_bytes=1_000,
    )

    assert projected == 1_900
    assert admission.required_bytes == 900
    assert admission.available_bytes == 9_999
    assert admission.passed


def test_live_memory_admission_rejects_large_available_memory_drop() -> None:
    frozen = _host()
    live = replace(frozen, memory_available_bytes=2_000)

    admission, projected = _memory_admission(
        frozen,
        topology=_topology(shards=4),
        isolated_pss_bytes=2_500,
        scheduler_pss_bytes=500,
        producer_pss_bytes=1_000,
        worker_descendant_pss_bytes=1_000,
        live_host=live,
        resident_pss_bytes=1_000,
    )

    assert projected == 5_500
    assert admission.required_bytes == 4_500
    assert admission.available_bytes == 2_000
    assert not admission.passed


def test_live_memory_admission_rejects_increased_cgroup_current() -> None:
    frozen = replace(_host(), memory_limit_bytes=10_000, memory_current_bytes=0)
    live = replace(frozen, memory_current_bytes=9_000)

    admission, _projected = _memory_admission(
        frozen,
        topology=_topology(shards=4),
        isolated_pss_bytes=2_500,
        scheduler_pss_bytes=500,
        producer_pss_bytes=1_000,
        worker_descendant_pss_bytes=1_000,
        live_host=live,
        resident_pss_bytes=1_000,
    )

    assert admission.available_bytes == 1_000
    assert not admission.passed


def test_live_memory_admission_never_erases_descendant_demand() -> None:
    frozen = _host(available=1_000)
    live = replace(frozen, memory_available_bytes=150)

    admission, projected = _memory_admission(
        frozen,
        topology=_topology(shards=2),
        isolated_pss_bytes=200,
        scheduler_pss_bytes=0,
        producer_pss_bytes=100,
        worker_descendant_pss_bytes=100,
        live_host=live,
        resident_pss_bytes=400,
    )

    assert projected == 300
    assert admission.required_bytes == 200
    assert not admission.passed


def test_worker_descendant_projection_uses_simultaneous_peak_not_lifetime_pids() -> None:
    statistics: dict[str, object] = {
        "root_process_id": 10,
        "additional_root_pids": [],
        "sample_count": 2,
        "peak_aggregate_pss_bytes": 100,
        "peak_worker_descendant_pss_bytes": 80,
        "worker_descendant_pss_peak": {
            "status": "available",
            "peak_bytes": 80,
            "sample_index": 0,
            "processes": [
                {"pid": 11, "create_time": 2.0, "pss_bytes": 80},
            ],
        },
        "process_metrics": [
            {
                "pid": 10,
                "create_time": 1.0,
                "maximum_pss_bytes": 100,
            },
            {
                "pid": 11,
                "create_time": 2.0,
                "maximum_pss_bytes": 80,
            },
            {
                "pid": 12,
                "create_time": 3.0,
                "maximum_pss_bytes": 80,
            },
        ],
    }

    assert (
        _worker_descendant_peak_pss_bytes(
            statistics,
            scheduler_process_id=None,
        )
        == 80
    )


def test_real_host_scheduler_is_registered_before_worker_classification(
    tmp_path: Path,
) -> None:
    endpoint = tmp_path / "scheduler.sock"
    scheduler = NativeHostScheduler(
        endpoint,
        worker_threads=1,
        request_threads=1,
        cpu_affinity=(min(os.sched_getaffinity(0)),),
        task_receipt_path=tmp_path / "task-receipts.jsonl",
    )
    started = time.perf_counter()
    scheduler_pid: int | None = None
    with ProcessTreeMonitor(sample_interval_seconds=0.002) as monitor:
        try:
            scheduler.start()
            scheduler_pid = scheduler.process_id
            monitor.register_additional_root(scheduler_pid)
            time.sleep(0.02)
        finally:
            scheduler.close()
    elapsed = time.perf_counter() - started
    statistics = monitor.statistics(elapsed_seconds=elapsed, compute_thread_limit=1)

    assert scheduler_pid in statistics["additional_root_pids"]
    assert statistics["worker_descendant_pss_peak"]["status"] == "unavailable"


def _axis_payload(
    *,
    marker: str = "same",
    mode: str = "current_stage052",
) -> dict[str, object]:
    return {
        "status": "completed",
        "mode": mode,
        "objective": {"vehicle_count": 1, "distance": 10.0},
        "routes": [["D0", marker, "D0"]],
        "semantic_trajectory": ["candidate", marker],
        "candidate_work_hash": "a" * 64,
        "route_result_hash": "b" * 64,
        "canonical_semantic_journal": {"sha256": "c" * 64},
        "measurement_evidence": {
            "exact_route_order": ["route-0"],
            "cache_lifecycle": {"lookup": 1, "store": 1},
        },
        "end_to_end_seconds": 2.0,
        "startup_seconds": 0.1,
        "solver_seconds": 1.5,
        "persistence_seconds": 0.4,
    }


def _write_axis_with_persistence(path: Path, payload: dict[str, object]) -> str:
    receipt_path = path.with_suffix(path.suffix + ".persistence")
    payload = {
        **payload,
        "persistence_receipt": {
            "schema_version": "stage05.2-axis-persistence-receipt-v1",
            "path": receipt_path.name,
            "sidecar_path": receipt_path.name + ".sha256",
        },
    }
    digest = _atomic_signed_json(path, payload)
    _atomic_signed_json(
        receipt_path,
        {
            "schema_version": "stage05.2-axis-persistence-receipt-v1",
            "axis_path": path.name,
            "axis_sha256": digest,
            "axis_status": "completed",
            "persistence_seconds": 0.5,
            "end_to_end_seconds": 2.1,
            "primary_artifact_bytes": 10,
            "persistence_receipt_bytes": 5,
            "artifact_bytes": 15,
            "persistence_breakdown": {},
            "timing_scope": "test",
        },
    )
    return digest


def _resource_statistics() -> dict[str, object]:
    return {
        "process_tree_cpu_seconds": 3.0,
        "process_tree_user_cpu_seconds": 2.5,
        "process_tree_system_cpu_seconds": 0.5,
        "process_tree_schedstat": {"runqueue_delay_ns": 10_000_000},
        "thread_tree_status": "available",
        "thread_tree": {
            "sample_missed_processes": 0,
            "unresolved_thread_observation_count": 0,
        },
        "thread_tree_schedstat": {"runqueue_delay_ns": 12_000_000},
        "thread_tree_context_switches": 25,
        "thread_tree_cpu_migrations": 3,
        "thread_tree_minor_faults": 45,
        "thread_tree_major_faults": 0,
        "process_tree_context_switches": 20,
        "process_tree_cpu_migrations": 2,
        "process_tree_minor_faults": 40,
        "process_tree_major_faults": 0,
        "peak_aggregate_rss_bytes": 2_000,
        "peak_aggregate_pss_bytes": 1_500,
    }


def test_atomic_signed_axis_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "axis.json"
    digest = _write_axis_with_persistence(path, _axis_payload())

    payload, inventory = _signed_axis(
        path,
        role="admitted-mode-block",
        storage_root=tmp_path,
    )

    assert payload["status"] == "completed"
    assert payload["producer_pre_receipt_seconds"] == 2.1
    assert "end_to_end_seconds" not in payload
    assert digest == _sha(path.read_bytes())
    assert inventory["sha256"] == digest
    assert inventory["storage_alias"] == "stage052-performance-calibration-run"
    assert inventory["relative_path"] == "axis.json"
    assert inventory["relative_sidecar_path"] == "axis.json.sha256"
    assert inventory["sidecar_sha256"] == _sha(Path(f"{path}.sha256").read_bytes())
    supporting = cast(list[dict[str, object]], inventory["supporting_artifacts"])
    assert {row["role"] for row in supporting} == {
        "axis-persistence-receipt",
        "axis-persistence-receipt-sidecar",
    }
    assert all((tmp_path / str(row["relative_path"])).is_file() for row in supporting)
    with pytest.raises(FileExistsError):
        _atomic_signed_json(path, _axis_payload())


def test_signed_axis_rejects_tampered_payload(tmp_path: Path) -> None:
    path = tmp_path / "axis.json"
    _atomic_signed_json(path, _axis_payload())
    path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    with pytest.raises(PerformanceObservationError, match="SHA-256 mismatch"):
        _signed_axis(path, role="probe", storage_root=tmp_path)


def test_memory_projection_counts_shared_scheduler_once_and_applies_80_percent() -> None:
    topology = _topology(shards=4)
    admitted, projected = _memory_admission(
        _host(),
        topology=topology,
        isolated_pss_bytes=1_500,
        scheduler_pss_bytes=500,
        producer_pss_bytes=500,
        worker_descendant_pss_bytes=500,
    )
    assert projected == 3_000
    assert admitted.passed

    rejected, projected = _memory_admission(
        _host(),
        topology=topology,
        isolated_pss_bytes=3_000,
        scheduler_pss_bytes=500,
        producer_pss_bytes=500,
        worker_descendant_pss_bytes=2_000,
    )
    assert projected == 9_000
    assert not rejected.passed
    assert rejected.reason == "projected PSS exceeds 80% of effective memory"


def test_semantic_replay_rejects_divergent_shards() -> None:
    assert _replay_identical((_axis_payload(), _axis_payload()))["routes"] == [["D0", "same", "D0"]]
    with pytest.raises(PerformanceObservationError, match="semantic replay diverged"):
        _replay_identical((_axis_payload(), _axis_payload(marker="different")))


def test_resource_summary_preserves_cpu_io_tail_and_zero_queue_gate() -> None:
    summary = _resource_summary(
        elapsed_seconds=2.0,
        independent_replay_seconds=0.25,
        statistics_payload=_resource_statistics(),
        payloads=(_axis_payload(), _axis_payload()),
        topology=_topology(),
        cgroup_after={"memory_current_bytes": 900, "memory_peak_bytes": 1_100},
        cgroup_io={"read_bytes": 100, "write_bytes": 200},
        scheduler_statistics={
            "request_queue": {
                "total_wait_seconds": 0.1,
                "peak_pending": 2,
                "queue_full_count": 0,
                "rejected_count": 0,
                "pending": 0,
            },
            "work_queue": {
                "total_wait_seconds": 0.2,
                "peak_pending": 3,
                "queue_full_count": 0,
                "rejected_count": 0,
                "pending": 0,
                "active": 0,
            },
        },
    )

    assert summary.effective_cores == 1.5
    assert summary.cpu_utilization_fraction == 0.75
    assert summary.queue_wait_seconds == pytest.approx(0.3)
    assert summary.pending_tasks_peak == 3
    assert summary.io_write_bytes == 200
    assert summary.worker_p95_seconds == 2.0
    assert summary.replay_seconds == 0.25


def test_resource_summary_rejects_scheduler_queue_overflow() -> None:
    scheduler = {
        "request_queue": {
            "total_wait_seconds": 0.1,
            "peak_pending": 2,
            "queue_full_count": 1,
            "rejected_count": 0,
            "pending": 0,
        },
        "work_queue": {
            "total_wait_seconds": 0.2,
            "peak_pending": 3,
            "queue_full_count": 0,
            "rejected_count": 0,
            "pending": 0,
            "active": 0,
        },
    }
    with pytest.raises(PerformanceObservationError, match="overflow/rejection"):
        _resource_summary(
            elapsed_seconds=2.0,
            independent_replay_seconds=0.25,
            statistics_payload=_resource_statistics(),
            payloads=(_axis_payload(),),
            topology=_topology(),
            cgroup_after={"memory_current_bytes": 900, "memory_peak_bytes": 1_100},
            cgroup_io={"read_bytes": 100, "write_bytes": 200},
            scheduler_statistics=scheduler,
        )


def test_parallel_diagnostics_recompute_speedup_efficiency_and_cpu_work() -> None:
    isolated = BlockExecution(
        elapsed_seconds=4.0,
        independent_replay_seconds=1.0,
        payloads=(_axis_payload(),),
        resource_statistics={},
        resource_summary=RuntimeResourceSummaryV2(
            elapsed_seconds=4.0,
            user_cpu_seconds=3.0,
            system_cpu_seconds=1.0,
            rss_bytes=100,
            pss_bytes=100,
            worker_p95_seconds=4.0,
            worker_max_seconds=4.0,
            worker_min_seconds=4.0,
        ),
        raw_inventory=(),
        scheduler_statistics=None,
        scheduler_process_id=None,
        scheduler_pss_bytes=0,
    )
    concurrent = BlockExecution(
        elapsed_seconds=6.0,
        independent_replay_seconds=1.5,
        payloads=tuple(_axis_payload() for _ in range(4)),
        resource_statistics={},
        resource_summary=RuntimeResourceSummaryV2(
            elapsed_seconds=6.0,
            user_cpu_seconds=12.0,
            system_cpu_seconds=4.0,
            rss_bytes=400,
            pss_bytes=400,
            worker_p95_seconds=5.0,
            worker_max_seconds=6.0,
            worker_min_seconds=4.0,
        ),
        raw_inventory=(),
        scheduler_statistics=None,
        scheduler_process_id=None,
        scheduler_pss_bytes=0,
    )

    diagnostics = _parallel_diagnostics(
        isolated=isolated,
        concurrent=concurrent,
        topology=_topology(shards=4),
    )

    assert diagnostics["speedup"] == pytest.approx(20.0 / 7.5)
    assert diagnostics["parallel_efficiency"] == pytest.approx(20.0 / 7.5 / 4.0)
    assert diagnostics["axes_per_hour"] == pytest.approx(1_920.0)
    assert diagnostics["cpu_seconds_per_axis"] == pytest.approx(4.0)
    assert diagnostics["worker_max_over_min"] == pytest.approx(1.5)


def test_local_native_modes_require_drained_bounded_work_pool_receipts() -> None:
    payload = _axis_payload(mode="per_solve_runtime")
    payload["candidate_transaction_statistics"] = {
        "native_candidate_work_pool": {
            "enabled": True,
            "pending_tasks": 0,
            "active_tasks": 0,
            "peak_pending_tasks": 3,
            "queue_full_count": 0,
            "rejected_count": 0,
            "task_receipt_dropped_count": 0,
            "total_wait_seconds": 0.25,
        }
    }
    assert _local_work_pool_receipt((payload,)) == (0.25, 3, 3, 0, 0)

    with pytest.raises(PerformanceObservationError, match="lacks bounded work-pool"):
        _local_work_pool_receipt((_axis_payload(mode="full_native_alns"),))


def test_scheduler_lifecycle_compares_two_per_wave_starts_with_two_wave_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int]] = []
    queue = {
        "pending": 0,
        "active": 0,
        "queue_full_count": 0,
        "rejected_count": 0,
    }

    def execute(**kwargs: object) -> BlockExecution:
        role = str(kwargs["role"])
        wave_count = cast(int, kwargs.get("wave_count", 1))
        calls.append((role, wave_count))
        elapsed = 3.0 if wave_count == 2 else 2.0
        payload_count = 4 if wave_count == 2 else 2
        return BlockExecution(
            elapsed_seconds=elapsed,
            independent_replay_seconds=0.25,
            payloads=tuple(_axis_payload(mode="host_scheduler") for _ in range(payload_count)),
            resource_statistics={},
            resource_summary=RuntimeResourceSummaryV2(
                elapsed_seconds=elapsed,
                effective_cores=1.0,
                cpu_utilization_fraction=0.5,
                rss_bytes=100,
                pss_bytes=100,
            ),
            raw_inventory=(),
            scheduler_statistics={
                "peak_distinct_client_pids": 2,
                "request_queue": {key: value for key, value in queue.items() if key != "active"},
                "work_queue": queue,
            },
            scheduler_process_id=123,
            scheduler_pss_bytes=100,
        )

    monkeypatch.setattr(
        "evrptw.experiments.stage052_performance_observation.execute_mode_block",
        execute,
    )
    topology = ExecutionTopology(
        workload_class="c5",
        shards=((0,), (1,)),
        worker_count=2,
        scheduler_cpu_ids=(2, 3),
        request_threads=2,
    )
    evidence, blocks = compare_host_scheduler_lifecycles(
        receipt=SimpleNamespace(build_profile="portable-o3"),  # type: ignore[arg-type]
        repeat=0,
        instance_name="c101C5",
        seed=2014,
        benchmark_dir=tmp_path,
        output_root=tmp_path,
        warm_start=(((),), {}),
        topology=topology,
        topology_id="topology",
        archive_root=tmp_path,
    )

    assert calls == [
        ("scheduler-per-wave-1", 1),
        ("scheduler-per-wave-2", 1),
        ("scheduler-mode-block", 2),
    ]
    assert len(blocks) == 3
    assert evidence["per_wave_end_to_end_seconds"] == 4.5
    assert evidence["mode_block_end_to_end_seconds"] == 3.25
    assert evidence["mode_block_faster"] is True
    assert evidence["build_profile"] == "portable-o3"
    assert evidence["topology_id"] == "topology"
    assert evidence["per_wave_process_tree_pss_peak_bytes"] == 100
    assert evidence["mode_block_process_tree_pss_peak_bytes"] == 100
    assert evidence["session_isolation_passed"] is True
    assert evidence["cache_reset_passed"] is True
