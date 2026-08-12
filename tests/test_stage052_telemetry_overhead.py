from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evrptw.experiments.stage052_native_architecture_review import (
    _RESOURCE_TELEMETRY_BASE_TOPOLOGY_FIELDS,
    _RESOURCE_TELEMETRY_DISABLED_TOPOLOGY_FIELDS,
    ReviewRecord,
    _resource_telemetry_topology_error,
)
from evrptw.experiments.stage052_native_architectures import (
    PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION,
    ArchitectureMode,
)
from evrptw.experiments.stage052_telemetry_overhead import (
    TELEMETRY_SAMPLE_SCHEMA_VERSION,
    TelemetryWorkloadSample,
    load_telemetry_overhead_receipt,
    measure_representative_telemetry_overhead,
    measure_telemetry_overhead,
    write_telemetry_overhead_receipt,
)
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.runtime_envelope import (
    DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS,
    PROCESS_TREE_STATISTICS_FIELDS,
)
from evrptw.stage052_performance import (
    ExecutionTopology,
    HostPerformanceEnvelope,
    TelemetryOverheadReceipt,
    execution_topology_id,
)
from evrptw.validation import validate_routes


def _write_signed_json(path: Path, payload: object) -> tuple[str, str]:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = Path(f"{path}.sha256")
    sidecar.write_text(digest + "\n", encoding="ascii")
    return digest, hashlib.sha256(sidecar.read_bytes()).hexdigest()


def test_telemetry_overhead_measurement_is_paired_signed_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    tick = iter(
        (
            0.0,
            1.0,
            1.0,
            2.01,
            2.01,
            3.01,
            3.01,
            4.02,
            4.02,
            5.02,
            5.02,
            6.03,
            6.03,
            7.03,
            7.03,
            8.04,
            8.04,
            9.04,
            9.04,
            10.05,
        )
    )

    def workload(_iterations: int) -> bytes:
        nonlocal calls
        calls += 1
        return b"a" * 64

    class Monitor:
        def __init__(self, *, sample_interval_seconds: float) -> None:
            assert sample_interval_seconds == DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS

        def __enter__(self) -> Monitor:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def statistics(
            self,
            *,
            elapsed_seconds: float,
            compute_thread_limit: int,
        ) -> dict[str, object]:
            assert elapsed_seconds > 0.0
            assert compute_thread_limit > 0
            return {"sample_count": 10}

    monkeypatch.setattr(
        "evrptw.experiments.stage052_telemetry_overhead.ProcessTreeMonitor",
        Monitor,
    )
    monkeypatch.setattr(
        "evrptw.experiments.stage052_telemetry_overhead.time.perf_counter",
        lambda: next(tick),
    )
    receipt = measure_telemetry_overhead(
        iterations=1,
        repeat_count=5,
        minimum_unmonitored_seconds=0.0,
        workload=workload,
    )
    assert calls == 11
    assert receipt.passed
    assert receipt.pair_orders == (
        "off-on",
        "on-off",
        "off-on",
        "on-off",
        "off-on",
    )
    path = tmp_path / "telemetry-overhead.json"
    write_telemetry_overhead_receipt(path, receipt)
    assert load_telemetry_overhead_receipt(path) == receipt
    assert (
        Path(f"{path}.sha256").read_text().strip() == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError):
        write_telemetry_overhead_receipt(path, receipt)


def test_representative_telemetry_gate_covers_complete_fixed_work_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter(float(index) for index in range(28))
    monkeypatch.setattr(
        "evrptw.experiments.stage052_telemetry_overhead.time.perf_counter",
        lambda: next(ticks),
    )
    evidence = {
        "kind": "representative-fixed-work-axis",
        "mode": "current_stage052",
        "axis": "fixed_work",
        "instance": "c101C5",
        "seed": 2014,
        "exact_calls": 20,
        "iterations": 200,
        "batch_size": 128,
        "minimal_validator_replay": True,
        "fingerprints_identical": True,
    }

    def sample(enabled: bool, index: int) -> TelemetryWorkloadSample:
        return TelemetryWorkloadSample(
            b"a" * 64,
            {
                "enabled": enabled,
                "sample_index": index,
                **(
                    {
                        "sample_interval_seconds": (
                            DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS
                        )
                    }
                    if enabled
                    else {}
                ),
            },
            {
                **evidence,
                "semantic_telemetry": True,
                "physical_telemetry": True,
                "persistence": True,
                "independent_replay": True,
                "resource_telemetry": enabled,
            },
        )

    receipt = measure_representative_telemetry_overhead(
        run_sample=sample,
        minimum_unmonitored_seconds=0.0,
    )
    assert receipt.sample_interval_seconds == DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS
    receipt.require_representative_fixed_work()
    assert receipt.passed
    assert receipt.p95_overhead_fraction == pytest.approx(0.0)
    assert receipt.workload_evidence["unmonitored_telemetry_surface"] == {
        "semantic_telemetry": True,
        "physical_telemetry": True,
        "persistence": True,
        "independent_replay": True,
        "resource_telemetry": False,
    }
    assert receipt.workload_evidence["monitored_telemetry_surface"] == {
        "semantic_telemetry": True,
        "physical_telemetry": True,
        "persistence": True,
        "independent_replay": True,
        "resource_telemetry": True,
    }


def test_independent_reviewer_replays_every_raw_telemetry_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evrptw.experiments import stage052_performance_calibration_review as review

    benchmark_dir = Path(__file__).resolve().parents[1] / "data/schneider"
    instance = parse_schneider(benchmark_dir / "c101C5.txt")
    routes = [
        ["D0", "C30", "D0"],
        ["D0", "C12", "D0"],
        ["D0", "C100", "D0"],
        ["D0", "C85", "D0"],
        ["D0", "C64", "D0"],
    ]
    report = validate_routes(instance, routes)
    assert report.feasible
    objective = list(SolutionObjective.from_report(instance, report).key)
    # current_stage052 has no Candidate Control runtime.  Empty rows plus an
    # empty hash are its explicit unavailable receipt, not the hash of [].
    candidate_work_hash = ""
    route_result_hash = ""
    empty_row_receipt = {
        "count": 0,
        "sha256": hashlib.sha256(b"stage05.2-row-evidence-v1\0").hexdigest(),
    }
    semantic_trajectory_receipt = {
        "schema_version": "stage05.2-external-semantic-trajectory-v1",
        "source": "canonical_semantic_journal:operator",
        **empty_row_receipt,
    }
    axis_payload = {
        "objective": objective,
        "routes": routes,
        "candidate_work_hash": candidate_work_hash,
        "route_result_hash": route_result_hash,
        "effective_iterations": 20,
        "termination_reason": "fixed_work_exhausted",
        "accepted_moves": 2,
        "rejected_moves": 18,
        "exact_started_calls": 20,
        "exact_completed_calls": 20,
        "exact_interrupted_calls": 0,
        "semantic_trajectory": semantic_trajectory_receipt,
        "trajectory": empty_row_receipt,
        "stage04_events": empty_row_receipt,
        "candidate_transaction_events": empty_row_receipt,
    }
    fingerprint = hashlib.sha256(
        json.dumps(axis_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected_topology = ExecutionTopology(
        workload_class="c5",
        shards=tuple(tuple(range(index * 4, index * 4 + 4)) for index in range(6)),
        worker_count=6,
        request_threads=6,
        affinity_policy="physical_core_first",
    )
    topology_id = execution_topology_id(expected_topology)
    frozen_host = HostPerformanceEnvelope(
        allowed_cpu_ids=tuple(range(24)),
        memory_total_bytes=16 * 1024**3,
        memory_available_bytes=8 * 1024**3,
        topology_source="provided",
    )
    sample_base: dict[str, object] = {
        "kind": "representative-fixed-work-axis",
        "mode": "current_stage052",
        "axis": "fixed_work",
        "instance": "c101C5",
        "seed": 2014,
        "exact_calls": 20,
        "iterations": 200,
        "batch_size": 128,
        "minimal_validator_replay": True,
        "fingerprints_identical": True,
    }

    def sample(enabled: bool, index: int, elapsed: float) -> dict[str, object]:
        sample_path = tmp_path / f"sample-{index}.json"
        axis_path = tmp_path / f"raw-axis-{index}.json"
        control_topology: dict[str, object] = {
            "shard_processes": 6,
            "threads_per_shard": 4,
            "compute_thread_limit": 24,
            "axis_compute_thread_limit": 4,
            "scheduler_threads": 0,
            "effective_native_search_threads": 4,
            "performance_profile_sha256": "1" * 64,
            "performance_topology_key": f"calibration:current_stage052:c5:{topology_id}",
            "configured_axis_cpu_ids": [0, 1, 2, 3],
            "configured_scheduler_cpu_ids": [],
            "scheduler_request_threads": 6,
            "allow_affinity_overlap": False,
            "shared_native_work_pool": False,
        }
        resource_topology = (
            {
                "sample_interval_seconds": DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS,
                "sample_count": 1,
                "peak_concurrent_processes": 1,
                "peak_aggregate_threads": 1,
                "peak_aggregate_rss_bytes": 1,
                "peak_aggregate_pss_bytes": 1,
                "process_tree_cpu_seconds": 0.1,
            }
            if enabled
            else {"telemetry_status": "disabled"}
        )
        raw_axis_payload = {
            **axis_payload,
            "scope": "performance_calibration",
            "repeat": 0,
            "mode": "current_stage052",
            "axis": "fixed_work",
            "instance": "c101C5",
            "seed": 2014,
            "revision": "2" * 40,
            "wheel_sha256": "3" * 64,
            "native_sha256": "4" * 64,
            "scheduler_sha256": "5" * 64,
            "fixed_work_budget": {
                "axis": "fixed_work",
                "batch_size": 128,
                "exact_calls": 20,
                "iterations": 200,
                "watchdog_seconds": 30.0,
            },
            "topology": {**control_topology, **resource_topology},
        }
        axis_sha256, axis_sidecar_sha256 = _write_signed_json(axis_path, raw_axis_payload)
        persistence_path = tmp_path / f"raw-axis-{index}.persistence.json"
        persistence_sha256, persistence_sidecar_sha256 = _write_signed_json(
            persistence_path,
            {"schema_version": "test-axis-persistence-v1"},
        )
        persistence_sidecar = Path(f"{persistence_path}.sha256")
        axis_inventory = [
            {
                "storage_alias": "stage052-performance-calibration-run",
                "relative_path": axis_path.relative_to(tmp_path).as_posix(),
                "sha256": axis_sha256,
                "relative_sidecar_path": Path(f"{axis_path}.sha256")
                .relative_to(tmp_path)
                .as_posix(),
                "sidecar_sha256": axis_sidecar_sha256,
                "role": (
                    "representative-resource-telemetry-on"
                    if enabled
                    else "representative-resource-telemetry-off"
                ),
                "supporting_artifacts": [
                    {
                        "relative_path": persistence_path.relative_to(tmp_path).as_posix(),
                        "sha256": persistence_sha256,
                        "role": "axis-persistence-receipt",
                    },
                    {
                        "relative_path": persistence_sidecar.relative_to(tmp_path).as_posix(),
                        "sha256": persistence_sidecar_sha256,
                        "role": "axis-persistence-receipt-sidecar",
                    },
                ],
            }
        ]
        workload_evidence = {
            **sample_base,
            "semantic_telemetry": True,
            "physical_telemetry": True,
            "persistence": True,
            "independent_replay": True,
            "resource_telemetry": enabled,
        }
        payload = {
            "schema_version": TELEMETRY_SAMPLE_SCHEMA_VERSION,
            "enabled": enabled,
            "sample_index": index,
            "elapsed_seconds": elapsed,
            "fingerprint": fingerprint,
            "fingerprint_payload": axis_payload,
            "workload_evidence": workload_evidence,
            "raw_axis_inventory": axis_inventory,
        }
        sample_sha256, sample_sidecar_sha256 = _write_signed_json(sample_path, payload)
        return {
            "enabled": enabled,
            "sample_index": index,
            "elapsed_seconds": elapsed,
            "fingerprint": fingerprint,
            "resource_summary": {
                **(
                    {
                        "sample_interval_seconds": (
                            DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS
                        )
                    }
                    if enabled
                    else {}
                ),
                "sample_storage_alias": "stage052-performance-calibration-run",
                "sample_relative_path": sample_path.relative_to(tmp_path).as_posix(),
                "sample_sidecar_relative_path": Path(f"{sample_path}.sha256")
                .relative_to(tmp_path)
                .as_posix(),
                "sample_sha256": sample_sha256,
                "sample_sidecar_sha256": sample_sidecar_sha256,
                "child_elapsed_seconds": elapsed,
                "raw_axis_inventory": axis_inventory,
            },
            "workload_evidence": workload_evidence,
        }

    orders = ("off-on", "on-off", "off-on", "on-off", "off-on")
    unmonitored = (1.0,) * 5
    monitored = (1.01,) * 5
    evidence = {
        **sample_base,
        "semantic_telemetry": True,
        "physical_telemetry": True,
        "persistence": True,
        "independent_replay": True,
        "unmonitored_telemetry_surface": {
            "semantic_telemetry": True,
            "physical_telemetry": True,
            "persistence": True,
            "independent_replay": True,
            "resource_telemetry": False,
        },
        "monitored_telemetry_surface": {
            "semantic_telemetry": True,
            "physical_telemetry": True,
            "persistence": True,
            "independent_replay": True,
            "resource_telemetry": True,
        },
    }
    evidence["warm_sample_evidence"] = (
        sample(False, -2, 1.0),
        sample(True, -1, 1.0),
    )
    evidence["paired_sample_evidence"] = tuple(
        {
            "pair_index": index,
            "order": order,
            "unmonitored": sample(
                False,
                index * 2 if order == "off-on" else index * 2 + 1,
                unmonitored[index],
            ),
            "monitored": sample(
                True,
                index * 2 + 1 if order == "off-on" else index * 2,
                monitored[index],
            ),
        }
        for index, order in enumerate(orders)
    )
    receipt = TelemetryOverheadReceipt(
        unmonitored_seconds=unmonitored,
        monitored_seconds=monitored,
        pair_orders=orders,
        sample_interval_seconds=DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS,
        workload_output_sha256=hashlib.sha256(fingerprint.encode("ascii")).hexdigest(),
        monitored_resource_summaries=tuple(
            {
                "sample_count": 1,
                "sample_interval_seconds": DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS,
            }
            for _ in range(5)
        ),
        workload_evidence=evidence,
    )
    monkeypatch.setattr(
        review,
        "_review_raw_axis",
        lambda *_args, **_kwargs: ({}, "current_stage052", "c5", topology_id),
    )
    monkeypatch.setattr(
        review,
        "generate_mode_topology_candidates",
        lambda *_args, **_kwargs: (expected_topology,),
    )

    def fake_replay_record(
        record: ReviewRecord,
        _benchmark_dir: Path,
        *,
        expected_resource_telemetry: bool,
    ) -> dict[str, object]:
        payload = record.payload
        topology = payload["topology"]
        observed_enabled = topology.get("telemetry_status") != "disabled"
        return {
            "valid": observed_enabled is expected_resource_telemetry,
            "semantics_complete": True,
        }

    monkeypatch.setattr(review, "_replay_record", fake_replay_record)

    replayed = review._replay_telemetry_children(
        receipt,
        run_root=tmp_path,
        benchmark_dir=benchmark_dir,
        build_identity={},
        frozen_host=frozen_host,
    )

    assert replayed == 12

    cadence_mismatch = TelemetryOverheadReceipt(
        unmonitored_seconds=unmonitored,
        monitored_seconds=monitored,
        pair_orders=orders,
        sample_interval_seconds=0.05,
        workload_output_sha256=hashlib.sha256(fingerprint.encode("ascii")).hexdigest(),
        monitored_resource_summaries=tuple(
            {
                "sample_count": 1,
                "sample_interval_seconds": DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS,
            }
            for _ in range(5)
        ),
        workload_evidence=evidence,
    )
    with pytest.raises(
        review.CalibrationReviewError,
        match="telemetry sample interval differs",
    ):
        review._replay_telemetry_children(
            cadence_mismatch,
            run_root=tmp_path,
            benchmark_dir=benchmark_dir,
            build_identity={},
            frozen_host=frozen_host,
        )

    warm_samples = evidence["warm_sample_evidence"]
    assert isinstance(warm_samples, tuple)
    off_row = warm_samples[0]
    assert isinstance(off_row, dict)
    paired_samples = evidence["paired_sample_evidence"]
    assert isinstance(paired_samples, tuple)
    first_pair = paired_samples[0]
    assert isinstance(first_pair, dict)
    duplicate_row = first_pair["unmonitored"]
    assert isinstance(duplicate_row, dict)
    duplicate_resources = duplicate_row["resource_summary"]
    assert isinstance(duplicate_resources, dict)
    duplicate_path = tmp_path / str(duplicate_resources["sample_relative_path"])
    duplicate_payload = json.loads(duplicate_path.read_text(encoding="utf-8"))
    original_inventory = duplicate_payload["raw_axis_inventory"]
    off_resources = off_row["resource_summary"]
    assert isinstance(off_resources, dict)
    duplicate_payload["raw_axis_inventory"] = off_resources["raw_axis_inventory"]
    duplicate_resources["raw_axis_inventory"] = off_resources["raw_axis_inventory"]
    duplicate_sha256, duplicate_sidecar_sha256 = _write_signed_json(
        duplicate_path,
        duplicate_payload,
    )
    duplicate_resources["sample_sha256"] = duplicate_sha256
    duplicate_resources["sample_sidecar_sha256"] = duplicate_sidecar_sha256
    duplicate_receipt = TelemetryOverheadReceipt(
        unmonitored_seconds=unmonitored,
        monitored_seconds=monitored,
        pair_orders=orders,
        sample_interval_seconds=DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS,
        workload_output_sha256=hashlib.sha256(fingerprint.encode("ascii")).hexdigest(),
        monitored_resource_summaries=tuple(
            {
                "sample_count": 1,
                "sample_interval_seconds": DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS,
            }
            for _ in range(5)
        ),
        workload_evidence=evidence,
    )
    with pytest.raises(review.CalibrationReviewError, match="raw axis path is duplicated"):
        review._replay_telemetry_children(
            duplicate_receipt,
            run_root=tmp_path,
            benchmark_dir=benchmark_dir,
            build_identity={},
            frozen_host=frozen_host,
        )
    duplicate_payload["raw_axis_inventory"] = original_inventory
    duplicate_resources["raw_axis_inventory"] = original_inventory
    duplicate_sha256, duplicate_sidecar_sha256 = _write_signed_json(
        duplicate_path,
        duplicate_payload,
    )
    duplicate_resources["sample_sha256"] = duplicate_sha256
    duplicate_resources["sample_sidecar_sha256"] = duplicate_sidecar_sha256

    resources = off_row["resource_summary"]
    assert isinstance(resources, dict)
    off_path = tmp_path / str(resources["sample_relative_path"])
    off_payload = json.loads(off_path.read_text(encoding="utf-8"))
    off_payload["minimal_replay_receipt"] = {"legacy": True}
    sample_sha256, sidecar_sha256 = _write_signed_json(off_path, off_payload)
    resources["sample_sha256"] = sample_sha256
    resources["sample_sidecar_sha256"] = sidecar_sha256
    tampered_receipt = TelemetryOverheadReceipt(
        unmonitored_seconds=unmonitored,
        monitored_seconds=monitored,
        pair_orders=orders,
        sample_interval_seconds=DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS,
        workload_output_sha256=hashlib.sha256(fingerprint.encode("ascii")).hexdigest(),
        monitored_resource_summaries=tuple(
            {
                "sample_count": 1,
                "sample_interval_seconds": DEFAULT_PROCESS_TREE_SAMPLE_INTERVAL_SECONDS,
            }
            for _ in range(5)
        ),
        workload_evidence=evidence,
    )
    with pytest.raises(review.CalibrationReviewError, match="legacy minimal receipt"):
        review._replay_telemetry_children(
            tampered_receipt,
            run_root=tmp_path,
            benchmark_dir=benchmark_dir,
            build_identity={},
            frozen_host=frozen_host,
        )


@pytest.mark.parametrize(
    "residual_field",
    ("process_metrics", "thread_tree", "bounded_samples", "cpu_stat", "psi"),
)
def test_disabled_resource_telemetry_rejects_any_monitor_residue(
    residual_field: str,
) -> None:
    topology = {field: None for field in _RESOURCE_TELEMETRY_DISABLED_TOPOLOGY_FIELDS}
    topology["telemetry_status"] = "disabled"
    assert (
        _resource_telemetry_topology_error(
            topology,
            mode=ArchitectureMode.CURRENT_STAGE052,
            expected_resource_telemetry=False,
        )
        is None
    )
    topology[residual_field] = {}
    assert _resource_telemetry_topology_error(
        topology,
        mode=ArchitectureMode.CURRENT_STAGE052,
        expected_resource_telemetry=False,
    ) is not None


def test_historical_enabled_resource_schema_remains_read_only_compatible() -> None:
    legacy_topology = {
        "sample_count": 1,
        "peak_concurrent_processes": 1,
        "peak_aggregate_threads": 1,
        "peak_aggregate_rss_bytes": 1,
        "peak_aggregate_pss_bytes": 1,
        "process_tree_cpu_seconds": 0.1,
    }
    assert (
        _resource_telemetry_topology_error(
            legacy_topology,
            mode=ArchitectureMode.CURRENT_STAGE052,
            expected_resource_telemetry=True,
            require_complete_schema=False,
        )
        is None
    )


def test_enabled_resource_telemetry_requires_exact_current_schema() -> None:
    topology = {
        field: None
        for field in (
            _RESOURCE_TELEMETRY_BASE_TOPOLOGY_FIELDS
            | PROCESS_TREE_STATISTICS_FIELDS
        )
    }
    assert (
        _resource_telemetry_topology_error(
            topology,
            mode=ArchitectureMode.CURRENT_STAGE052,
            expected_resource_telemetry=True,
        )
        is None
    )
    topology["unknown_process_tree_field"] = 1
    assert _resource_telemetry_topology_error(
        topology,
        mode=ArchitectureMode.CURRENT_STAGE052,
        expected_resource_telemetry=True,
    ) is not None


def test_prior_profile_resource_schema_does_not_require_new_worker_sample_counters() -> None:
    prior_statistics_fields = PROCESS_TREE_STATISTICS_FIELDS - {
        "worker_descendant_pss_complete_sample_count",
        "worker_descendant_pss_incomplete_sample_count",
    }
    topology = {
        field: None
        for field in (_RESOURCE_TELEMETRY_BASE_TOPOLOGY_FIELDS | prior_statistics_fields)
    }

    assert (
        _resource_telemetry_topology_error(
            topology,
            mode=ArchitectureMode.CURRENT_STAGE052,
            expected_resource_telemetry=True,
            comparison_schema=PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION,
        )
        is None
    )
    topology["worker_descendant_pss_complete_sample_count"] = 1
    assert _resource_telemetry_topology_error(
        topology,
        mode=ArchitectureMode.CURRENT_STAGE052,
        expected_resource_telemetry=True,
        comparison_schema=PRIOR_PROFILE_COMPARISON_SCHEMA_VERSION,
    ) is not None


def test_representative_control_rejects_stable_wrong_topology_or_watchdog() -> None:
    from evrptw.experiments import stage052_performance_calibration_review as review

    expected_topology = ExecutionTopology(
        workload_class="c5",
        shards=((0, 1, 2, 3), (4, 5, 6, 7)),
        worker_count=2,
        request_threads=2,
        affinity_policy="physical_core_first",
    )
    topology_id = execution_topology_id(expected_topology)
    payload: dict[str, object] = {
        "scope": "performance_calibration",
        "repeat": 0,
        "mode": "current_stage052",
        "axis": "fixed_work",
        "instance": "c101C5",
        "seed": 2014,
        "revision": "1" * 40,
        "wheel_sha256": "2" * 64,
        "native_sha256": "3" * 64,
        "scheduler_sha256": "4" * 64,
        "fixed_work_budget": {
            "axis": "fixed_work",
            "batch_size": 128,
            "exact_calls": 20,
            "iterations": 200,
            "watchdog_seconds": 30.0,
        },
        "topology": {
            "shard_processes": 2,
            "threads_per_shard": 4,
            "compute_thread_limit": 8,
            "axis_compute_thread_limit": 4,
            "scheduler_threads": 0,
            "effective_native_search_threads": 4,
            "performance_profile_sha256": "5" * 64,
            "performance_topology_key": f"calibration:current_stage052:c5:{topology_id}",
            "configured_axis_cpu_ids": [0, 1, 2, 3],
            "configured_scheduler_cpu_ids": [],
            "scheduler_request_threads": 2,
            "allow_affinity_overlap": False,
            "shared_native_work_pool": False,
        },
    }
    review._representative_axis_control_identity(  # noqa: SLF001
        payload,
        expected_topology=expected_topology,
        topology_id=topology_id,
    )

    free_scheduler = ExecutionTopology(
        workload_class="c5",
        shards=((0, 1, 2, 3), (4, 5, 6, 7)),
        worker_count=2,
        request_threads=2,
        affinity_policy="free_scheduler",
    )
    missing_topology_host = HostPerformanceEnvelope(
        allowed_cpu_ids=tuple(range(8)),
        memory_total_bytes=16 * 1024**3,
        memory_available_bytes=8 * 1024**3,
        topology_source="unavailable",
    )
    selected_without_physical_topology = min(
        review.generate_mode_topology_candidates(
            missing_topology_host,
            mode="current_stage052",
            workload_class="c5",
        ),
        key=lambda item: (
            abs(item.shard_count - 2),
            item.affinity_policy != "physical_core_first",
            execution_topology_id(item),
        ),
    )
    assert selected_without_physical_topology.to_dict() == free_scheduler.to_dict()
    free_scheduler_id = execution_topology_id(free_scheduler)
    free_scheduler_payload = json.loads(json.dumps(payload))
    free_scheduler_payload["topology"].update(
        {
            "threads_per_shard": 4,
            "axis_compute_thread_limit": 8,
            "effective_native_search_threads": 4,
            "performance_topology_key": (
                f"calibration:current_stage052:c5:{free_scheduler_id}"
            ),
            "configured_axis_cpu_ids": list(range(8)),
        }
    )
    review._representative_axis_control_identity(  # noqa: SLF001
        free_scheduler_payload,
        expected_topology=free_scheduler,
        topology_id=free_scheduler_id,
    )

    wrong_watchdog = json.loads(json.dumps(payload))
    wrong_watchdog["fixed_work_budget"]["watchdog_seconds"] = 29.0
    with pytest.raises(review.CalibrationReviewError, match="fixed-work budget schema"):
        review._representative_axis_control_identity(  # noqa: SLF001
            wrong_watchdog,
            expected_topology=expected_topology,
            topology_id=topology_id,
        )

    wrong_topology = json.loads(json.dumps(payload))
    wrong_topology["topology"]["configured_axis_cpu_ids"] = [0]
    with pytest.raises(review.CalibrationReviewError, match="host-derived control"):
        review._representative_axis_control_identity(  # noqa: SLF001
            wrong_topology,
            expected_topology=expected_topology,
            topology_id=topology_id,
        )
