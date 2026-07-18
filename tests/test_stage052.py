from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import evrptw.experiments.stage052_performance as stage052_performance
from evrptw._core import distance_matrix
from evrptw.artifacts import (
    ArtifactBundleWriter,
    ArtifactIntegrityError,
    ArtifactReader,
    ArtifactRunContext,
    ArtifactStorageConfig,
    expand_v2_screening_decision,
)
from evrptw.experiments.stage052_performance import (
    PERFORMANCE_INSTANCES,
    PERFORMANCE_SEEDS,
    _ensure_partial_shard_failure,
    _run_and_persist_v2_shard,
    _run_v2_shard_task,
    _run_v2_tasks,
    _ShardTask,
    axes_for_scope,
    validate_stage052_run_label,
    verify_stage051_prerequisite,
)
from evrptw.experiments.stage052_performance_review import (
    replay_stage052_storage_semantics,
    validate_per_run_scope,
    verify_stage052_review_prerequisite,
)
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.stage052 import (
    AcceleratorDecision,
    ArtifactStorageObservation,
    PerformanceObservation,
    Stage052Component,
    decide_accelerator,
    evaluate_artifact_storage_promotion,
    evaluate_promotion,
    formal_budget_matrix,
    select_worker_count,
)
from evrptw.stage052_evidence import (
    ProcessTreeResourceSampler,
    collect_performance_provenance,
    validate_worker_ownership,
    verify_stage052_prerequisite,
)


def _observation(
    instance: str,
    seed: int,
    seconds: float,
    *,
    customer_count: int = 100,
    semantic_digest: str = "same",
) -> PerformanceObservation:
    return PerformanceObservation(
        instance=instance,
        seed=seed,
        customer_count=customer_count,
        end_to_end_seconds=seconds,
        semantic_digest=semantic_digest,
    )


def test_stage052_components_have_one_strict_order() -> None:
    assert tuple(component.value for component in Stage052Component) == (
        "perf_baseline",
        "hot_path",
        "artifact_streaming",
        "job_parallel",
        "native_kernels",
        "accelerator_pilot",
        "benchmark",
    )


def test_formal_budget_matrix_is_the_declared_2040_runs() -> None:
    matrix = formal_budget_matrix()
    assert matrix.total_runs == 2040
    assert matrix.declared_solver_seconds == 229_200
    assert matrix.budgets_for_customer_count(15) == (30,)
    assert matrix.budgets_for_customer_count(100) == (30, 60, 300)


def test_promotion_requires_semantic_equality_and_15_percent_gain() -> None:
    previous = [
        _observation("c101_21", 2014, 100.0),
        _observation("r101_21", 2014, 100.0),
        _observation("rc101_21", 2014, 100.0),
    ]
    promoted = [
        _observation("c101_21", 2014, 84.0),
        _observation("r101_21", 2014, 84.0),
        _observation("rc101_21", 2014, 84.0),
    ]
    decision = evaluate_promotion(previous, promoted)
    assert decision.passed
    assert decision.aggregate_median_saving >= 0.15

    promoted[1] = _observation("r101_21", 2014, 84.0, semantic_digest="changed")
    decision = evaluate_promotion(previous, promoted)
    assert not decision.passed
    assert "semantic" in decision.detail


def test_promotion_uses_100_customer_pairs_and_keeps_c5_as_control() -> None:
    previous = [
        _observation("c101C5", 2014, 100.0, customer_count=5),
        _observation("c101_21", 2014, 100.0),
        _observation("r101_21", 2014, 100.0),
        _observation("rc101_21", 2014, 100.0),
    ]
    promoted = [
        _observation("c101C5", 2014, 100.0, customer_count=5),
        _observation("c101_21", 2014, 84.0),
        _observation("r101_21", 2014, 84.0),
        _observation("rc101_21", 2014, 84.0),
    ]

    decision = evaluate_promotion(previous, promoted)

    assert decision.passed
    assert decision.aggregate_median_saving == pytest.approx(0.16)

    incomplete = evaluate_promotion(previous[1:-1], promoted[1:-1])
    assert not incomplete.passed
    assert "families" in incomplete.detail


def test_worker_selection_is_fail_fast_and_memory_bounded() -> None:
    assert select_worker_count({1: 100.0, 2: 60.0, 4: 38.0}, {1: 4, 2: 8, 4: 11}) == 4
    assert select_worker_count({1: 100.0, 2: 60.0, 4: 50.0}, {1: 4, 2: 8, 4: 11}) == 2
    with pytest.raises(ValueError, match="NOT_READY"):
        select_worker_count({1: 100.0, 2: 80.0, 4: 30.0}, {1: 4, 2: 8, 4: 11})
    with pytest.raises(ValueError, match="finite"):
        select_worker_count({1: 100.0, 2: math.nan, 4: 30.0}, {1: 4, 2: 8, 4: 11})
    with pytest.raises(ValueError, match="finite"):
        select_worker_count({1: 100.0, 2: 60.0, 4: 30.0}, {1: 4, 2: math.inf, 4: 11})


def _storage_observation(
    *,
    policy: str,
    digest: str = "same",
    persistence_seconds: float = 2.0,
    end_to_end_seconds: float = 10.0,
    peak_rss_bytes: int = 50,
) -> ArtifactStorageObservation:
    return ArtifactStorageObservation(
        instance="c101_21",
        seed=2014,
        axis="fixed_work",
        storage_policy_version=policy,
        semantic_digest=digest,
        artifact_persistence_seconds=persistence_seconds,
        end_to_end_seconds=end_to_end_seconds,
        peak_rss_bytes=peak_rss_bytes,
    )


def test_artifact_storage_promotion_requires_replay_equality() -> None:
    baseline = [_storage_observation(policy="artifact-storage-v1", peak_rss_bytes=100)]
    predecessor = [_storage_observation(policy="artifact-storage-v1")]
    candidate = [_storage_observation(policy="artifact-storage-v2", digest="changed")]

    decision = evaluate_artifact_storage_promotion(baseline, predecessor, candidate)

    assert not decision.replay_equality_passed
    assert "semantic" in decision.replay_detail


def test_artifact_storage_promotion_enforces_persistence_and_half_rss() -> None:
    baseline = [_storage_observation(policy="artifact-storage-v1", peak_rss_bytes=100)]
    predecessor = [_storage_observation(policy="artifact-storage-v1")]
    passing = [_storage_observation(policy="artifact-storage-v2", peak_rss_bytes=50)]
    decision = evaluate_artifact_storage_promotion(baseline, predecessor, passing)
    assert decision.passed

    slow = [
        _storage_observation(
            policy="artifact-storage-v2",
            persistence_seconds=3.01,
            peak_rss_bytes=50,
        )
    ]
    assert not evaluate_artifact_storage_promotion(baseline, predecessor, slow).persistence_passed

    memory_heavy = [_storage_observation(policy="artifact-storage-v2", peak_rss_bytes=51)]
    assert not evaluate_artifact_storage_promotion(baseline, predecessor, memory_heavy).rss_passed


def test_artifact_storage_persistence_gate_uses_run_aggregate() -> None:
    baseline = [
        _storage_observation(policy="artifact-storage-v1", peak_rss_bytes=100),
        replace(
            _storage_observation(policy="artifact-storage-v1", peak_rss_bytes=100),
            instance="r101_21",
        ),
    ]
    predecessor = [
        _storage_observation(policy="artifact-storage-v1"),
        replace(
            _storage_observation(policy="artifact-storage-v1"),
            instance="r101_21",
        ),
    ]
    candidate = [
        _storage_observation(
            policy="artifact-storage-v2",
            persistence_seconds=0.8,
            end_to_end_seconds=1.0,
            peak_rss_bytes=50,
        ),
        replace(
            _storage_observation(
                policy="artifact-storage-v2",
                persistence_seconds=1.0,
                end_to_end_seconds=9.0,
                peak_rss_bytes=50,
            ),
            instance="r101_21",
        ),
    ]

    decision = evaluate_artifact_storage_promotion(baseline, predecessor, candidate)

    assert decision.persistence_passed
    assert "aggregate" in decision.persistence_detail


@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf))
def test_artifact_storage_observation_rejects_non_finite_numbers(
    invalid: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        _storage_observation(
            policy="artifact-storage-v2",
            persistence_seconds=invalid,
        )


def test_accelerator_is_skipped_below_occupancy_threshold() -> None:
    decision = decide_accelerator(median_batch_occupancy=31)
    assert decision is AcceleratorDecision.GPU_NOT_JUSTIFIED


def test_instance_lookup_and_distance_matrix_are_stable() -> None:
    depot = Node("D0", NodeType.DEPOT, 0, 0, 0, 0, 100, 0)
    customer = Node("C1", NodeType.CUSTOMER, 3, 4, 1, 0, 100, 0)
    instance = Instance(
        "toy",
        (depot, customer),
        Vehicle(100, 10, 1, 1, 1),
        distance_backend="python",
    )
    assert instance.by_name is instance.by_name
    assert instance.customers is instance.customers
    assert instance.distance("D0", "C1") == 5.0


def test_stage052_runner_contract_has_canonical_labels_and_axes(
    tmp_path: Path,
) -> None:
    validate_stage052_run_label(
        "stage05.2_perf_baseline_attempt01", Stage052Component.PERF_BASELINE
    )
    with pytest.raises(ValueError, match="canonical"):
        validate_stage052_run_label(
            "stage052_perf_baseline_attempt01", Stage052Component.PERF_BASELINE
        )
    assert tuple(axis.name for axis in axes_for_scope("performance")) == (
        "fixed_work_control",
        "fixed_work",
        "wall_clock_30",
    )
    assert tuple(axis.name for axis in axes_for_scope("formal", customer_count=100)) == (
        "wall_clock_30",
        "wall_clock_60",
        "wall_clock_300",
    )

    publication = tmp_path / "stage051.json"
    publication.write_text(
        '{"run_label":"stage05.1_best_known_attempt06",'
        '"status":"READY_FOR_STAGE05_2",'
        '"comparison_baseline":"stage04_adaptive_weights_attempt15"}',
        encoding="utf-8",
    )
    assert verify_stage051_prerequisite(publication)["status"] == "READY_FOR_STAGE05_2"


def test_stage052_reviewer_rejects_duplicate_or_missing_axes() -> None:
    rows = [
        {
            "instance": instance,
            "seed": str(seed),
            "axis": axis,
            "customer_count": "5" if instance == "c101C5" else "100",
            "validator_passed": "True",
            "failure_status": "",
        }
        for instance in PERFORMANCE_INSTANCES
        for seed in PERFORMANCE_SEEDS
        for axis in ("fixed_work_control", "fixed_work", "wall_clock_30")
    ]
    passed, detail = validate_per_run_scope(
        rows,
        instances=PERFORMANCE_INSTANCES,
        seeds=PERFORMANCE_SEEDS,
        axes=("fixed_work_control", "fixed_work", "wall_clock_30"),
    )
    assert passed, detail
    rows.append(dict(rows[0]))
    passed, detail = validate_per_run_scope(
        rows,
        instances=PERFORMANCE_INSTANCES,
        seeds=PERFORMANCE_SEEDS,
        axes=("fixed_work_control", "fixed_work", "wall_clock_30"),
    )
    assert not passed
    assert "duplicate" in detail


def test_stage052_reviewer_recomputes_customer_count_identity() -> None:
    rows = [
        {
            "instance": instance,
            "seed": str(seed),
            "axis": axis,
            "customer_count": "5" if instance == "c101C5" else "100",
            "validator_passed": "True",
            "failure_status": "",
        }
        for instance in PERFORMANCE_INSTANCES
        for seed in PERFORMANCE_SEEDS
        for axis in ("fixed_work_control", "fixed_work", "wall_clock_30")
    ]
    rows[9]["customer_count"] = "5"

    passed, detail = validate_per_run_scope(
        rows,
        instances=PERFORMANCE_INSTANCES,
        seeds=PERFORMANCE_SEEDS,
        axes=("fixed_work_control", "fixed_work", "wall_clock_30"),
    )

    assert not passed
    assert "customer_count mismatch" in detail


def test_stage052_review_prerequisite_verifies_identity_status_and_files(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "stage05.2_hot_path_attempt03"
    review_dir = raw_dir / "review"
    review_dir.mkdir(parents=True)
    report = review_dir / "review_report.md"
    findings = review_dir / "review_findings.csv"
    report.write_text("accepted\n", encoding="utf-8")
    findings.write_text("gate,passed\nall,True\n", encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (review_dir / "review_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "stage05.2-review-v1",
                "run_label": raw_dir.name,
                "component": "hot_path",
                "scope": "performance",
                "status": "READY_FOR_STAGE052_ARTIFACT_STREAMING",
                "gates": {"all": {"passed": True}},
                "files": {
                    report.name: digest(report),
                    findings.name: digest(findings),
                },
            }
        ),
        encoding="utf-8",
    )

    verify_stage052_review_prerequisite(
        raw_dir,
        expected_component="hot_path",
        expected_status="READY_FOR_STAGE052_ARTIFACT_STREAMING",
    )
    report.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_stage052_review_prerequisite(
            raw_dir,
            expected_component="hot_path",
            expected_status="READY_FOR_STAGE052_ARTIFACT_STREAMING",
        )


def test_stage052_producer_prerequisite_binds_raw_and_review_identity(tmp_path: Path) -> None:
    run_label = "stage05.2_artifact_streaming_attempt04"
    raw_dir = tmp_path / run_label
    config = tmp_path / "stage052.toml"
    config.write_text("[stage05_2]\nschema_version='test'\n", encoding="utf-8")
    config_digest = hashlib.sha256(config.read_bytes()).hexdigest()
    writer = ArtifactBundleWriter(
        raw_dir,
        ArtifactRunContext("stage05.2", "artifact_streaming", run_label),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    writer.write_control(
        metadata={
            "run_label": run_label,
            "component": "artifact_streaming",
            "scope": "performance",
            "repository_dirty": False,
            "repository_revision": "a" * 40,
            "configuration_sha256": config_digest,
        },
        configuration_path=config,
    )
    writer.finalize()
    review_dir = raw_dir / "review"
    review_dir.mkdir()
    report = review_dir / "review_report.md"
    findings = review_dir / "review_findings.csv"
    report.write_text("accepted\n", encoding="utf-8")
    findings.write_text("gate,passed\nall,True\n", encoding="utf-8")
    (review_dir / "review_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "stage05.2-review-v1",
                "run_label": run_label,
                "component": "artifact_streaming",
                "scope": "performance",
                "status": "READY_FOR_STAGE052_JOB_PARALLEL",
                "gates": {"all": {"passed": True}},
                "files": {
                    report.name: hashlib.sha256(report.read_bytes()).hexdigest(),
                    findings.name: hashlib.sha256(findings.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    identity = verify_stage052_prerequisite(
        raw_dir,
        expected_component="artifact_streaming",
        expected_status="READY_FOR_STAGE052_JOB_PARALLEL",
        expected_run_label=run_label,
    )
    assert identity.run_label == run_label
    assert identity.repository_revision == "a" * 40

    report.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        verify_stage052_prerequisite(
            raw_dir,
            expected_component="artifact_streaming",
            expected_status="READY_FOR_STAGE052_JOB_PARALLEL",
        )


def test_process_tree_resource_summary_includes_live_child() -> None:
    sampler = ProcessTreeResourceSampler(
        run_label="stage05.2_job_parallel_attempt99",
        component="job_parallel",
        configured_worker_count=2,
        interval_seconds=0.01,
    )
    sampler.start()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; data=bytearray(8_000_000); time.sleep(.12)"],
    )
    child.wait(timeout=2.0)
    summary = sampler.stop()

    assert child.pid in summary.descendant_pids
    assert summary.aggregate_peak_rss_bytes > 8_000_000
    assert summary.sample_count >= 2
    assert summary.status == "complete"


def test_worker_ownership_requires_actual_sampled_pids() -> None:
    resource = {
        "schema_version": "stage05.2-run-resource-v2",
        "run_label": "stage05.2_job_parallel_attempt99",
        "component": "job_parallel",
        "configured_worker_count": 2,
        "measurement_scope": "task_scheduling_through_parent_control_preparation",
        "status": "complete",
        "run_wall_seconds": 10.0,
        "sample_interval_seconds": 0.05,
        "aggregate_peak_rss_bytes": 1024,
        "mean_active_cores": 1.5,
        "peak_active_cores": 2.0,
        "sample_count": 200,
        "parent_pid": 100,
        "descendant_pids": [201, 202, 301],
    }
    manifests = [
        {
            "run_label": "stage05.2_job_parallel_attempt99",
            "evidence_completeness": "complete",
            "shard_ordinal": 0,
            "worker_identity": "pid-201",
        },
        {
            "run_label": "stage05.2_job_parallel_attempt99",
            "evidence_completeness": "complete",
            "shard_ordinal": 1,
            "worker_identity": "pid-202",
        },
        {
            "run_label": "stage05.2_job_parallel_attempt99",
            "evidence_completeness": "complete",
            "shard_ordinal": 2,
            "worker_identity": "pid-201",
        },
    ]
    passed, _, owners = validate_worker_ownership(
        resource,
        manifests,
        expected_workers=2,
        expected_run_label="stage05.2_job_parallel_attempt99",
        expected_component="job_parallel",
    )
    assert passed
    assert owners == (201, 202)

    fake = [{**manifests[0], "worker_identity": "pid-0"}, *manifests[1:]]
    passed, detail, _ = validate_worker_ownership(
        resource,
        fake,
        expected_workers=2,
        expected_run_label="stage05.2_job_parallel_attempt99",
        expected_component="job_parallel",
    )
    assert not passed
    assert "not observed" in detail

    parent_owned = [{**manifests[0], "worker_identity": "pid-100"}, *manifests[1:]]
    passed, detail, _ = validate_worker_ownership(
        resource,
        parent_owned,
        expected_workers=2,
        expected_run_label="stage05.2_job_parallel_attempt99",
        expected_component="job_parallel",
    )
    assert not passed
    assert "not observed" in detail


def test_parallel_pool_terminates_all_workers_before_recording_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeProcess:
        pid = 999

        def terminate(self) -> None:
            events.append("terminate")

        def join(self, *, timeout: float) -> None:
            events.append(f"join:{timeout}")

        def is_alive(self) -> bool:
            return False

    class FakeFuture:
        def result(self) -> object:
            raise RuntimeError("worker failed")

        def cancel(self) -> None:
            events.append("cancel")

    class FailingExecutor:
        def __init__(self, **_: object) -> None:
            self._processes = {1: FakeProcess()}

        def submit(self, *_: object) -> FakeFuture:
            return FakeFuture()

        def shutdown(self, *, wait: bool, cancel_futures: bool = False) -> None:
            events.append(f"shutdown:{wait}:{cancel_futures}")

    tasks = [SimpleNamespace(instance_name="c101C5", seed=2014)]
    monkeypatch.setattr(stage052_performance, "ProcessPoolExecutor", FailingExecutor)
    monkeypatch.setattr(stage052_performance, "get_context", lambda _: object())
    monkeypatch.setattr(stage052_performance, "as_completed", lambda futures: iter(futures))
    monkeypatch.setattr(
        stage052_performance,
        "_ensure_partial_shard_failure",
        lambda *_: events.append("failure_evidence"),
    )

    with pytest.raises(RuntimeError, match="worker failed"):
        _run_v2_tasks(tasks, worker_count=2)  # type: ignore[arg-type]

    assert events == [
        "cancel",
        "terminate",
        "join:2.0",
        "shutdown:True:True",
        "failure_evidence",
    ]


def test_performance_provenance_records_inputs_without_secret_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = tmp_path / "instance.txt"
    stage02 = tmp_path / "stage02.toml"
    stage04 = tmp_path / "stage04.toml"
    instance.write_text("instance", encoding="utf-8")
    stage02.write_text("stage02", encoding="utf-8")
    stage04.write_text("stage04", encoding="utf-8")
    native_extension = tmp_path / "_core.so"
    native_extension.write_bytes(b"native")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("SECRET_TOKEN", "must-not-be-recorded")

    provenance = collect_performance_provenance(
        instance_paths={"c101C5": instance},
        stage02_config_path=stage02,
        stage04_config_path=stage04,
        max_iterations=1000,
        batch_size=128,
        runtime_environment={
            "python": {"version": "3.13.13", "implementation": "CPython"},
            "system": {"platform": "macOS", "machine": "arm64", "cpu_count": 10},
            "packages": {"numpy": "2.4.1"},
            "native_extension": str(native_extension),
        },
    )

    assert provenance["instance_sha256"] == {
        "c101C5": hashlib.sha256(instance.read_bytes()).hexdigest()
    }
    environment = provenance["environment_variables"]
    assert isinstance(environment, dict)
    assert environment["PYTHONHASHSEED"] == "0"
    assert "SECRET_TOKEN" not in environment
    assert provenance["fallback_allowed"] is False


def test_storage_semantic_replay_is_independent_and_equal_for_v1_v2(
    tmp_path: Path,
) -> None:
    def write(
        policy: str,
        component: str,
        label: str,
        *,
        route_evaluation_status: str = "completed_feasible",
        extra_solution_axis: bool = False,
        extra_unreferenced_route: bool = False,
        fixed_route_customer: str = "C1",
        cache_bytes: int = 396,
    ) -> Path:
        run_dir = tmp_path / label
        writer = ArtifactBundleWriter(
            run_dir,
            ArtifactRunContext("stage05.2", component, label),
            ArtifactStorageConfig(storage_policy_version=policy),
        )
        solution_axes: dict[str, object] = {
            "fixed_work": {
                "routes": [["C1"]],
                "objective_key": [1, 10.0, 0.0, 0],
            }
        }
        if extra_solution_axis:
            solution_axes["forged_axis"] = {
                "routes": [["C1"]],
                "objective_key": [1, 10.0, 0.0, 0],
            }
        route_dictionary = {"fixed-route": (fixed_route_customer,)}
        if extra_unreferenced_route:
            route_dictionary["wall-only-route"] = ("W1",)
        writer.write_instance_seed(
            instance="c101_21",
            seed=2014,
            raw_payload={
                "instance": "c101_21",
                "seed": 2014,
                "axes": {"fixed_work": {"started_calls": 100, "completed_calls": 100}},
            },
            solution_payload={
                "instance": "c101_21",
                "seed": 2014,
                "axes": solution_axes,
            },
            trace_payload={},
            environment_payload={},
            route_dictionary=route_dictionary,
            critical_events=[
                {
                    "event_type": "candidate_state",
                    "benchmark_axis": "fixed_work",
                    "status": "rejected",
                    "accepted": False,
                    "global_best": False,
                    "operation": "candidate_rejected",
                    "cache_key_digest": "abc",
                },
                {
                    "event_type": "route_evaluation",
                    "benchmark_axis": "fixed_work",
                    "status": route_evaluation_status,
                    "kind": "exact_call",
                    "exact_started": True,
                    "exact_completed": True,
                    "evaluation_id": 1,
                    "route_key": "fixed-route",
                },
                {
                    "event_type": "cache_event",
                    "benchmark_axis": "fixed_work",
                    "operation": "store",
                    "cache_key_digest": "abc",
                    "entry_bytes": cache_bytes,
                    "current_bytes": cache_bytes,
                    "current_entries": 1,
                },
            ],
            shard_ordinal=0 if policy == "artifact-storage-v2" else None,
            worker_identity=("worker-0" if policy == "artifact-storage-v2" else None),
        )
        writer.finalize()
        return run_dir

    v1 = write(
        "artifact-storage-v1",
        "hot_path",
        "stage05.2_hot_path_attempt99",
    )
    v2 = write(
        "artifact-storage-v2",
        "artifact_streaming",
        "stage05.2_artifact_streaming_attempt99",
    )

    assert replay_stage052_storage_semantics(v1) == replay_stage052_storage_semantics(v2)
    volatile_cache_bytes = write(
        "artifact-storage-v2",
        "artifact_streaming",
        "stage05.2_artifact_streaming_attempt90",
        cache_bytes=397,
    )
    assert replay_stage052_storage_semantics(v1) == replay_stage052_storage_semantics(
        volatile_cache_bytes
    )
    wall_route_only = write(
        "artifact-storage-v2",
        "artifact_streaming",
        "stage05.2_artifact_streaming_attempt96",
        extra_unreferenced_route=True,
    )
    assert replay_stage052_storage_semantics(v1) == replay_stage052_storage_semantics(
        wall_route_only
    )

    changed_event = write(
        "artifact-storage-v2",
        "artifact_streaming",
        "stage05.2_artifact_streaming_attempt98",
        route_evaluation_status="completed_infeasible",
    )
    assert replay_stage052_storage_semantics(v1) != replay_stage052_storage_semantics(changed_event)

    changed_route = write(
        "artifact-storage-v2",
        "artifact_streaming",
        "stage05.2_artifact_streaming_attempt95",
        fixed_route_customer="C2",
    )
    assert replay_stage052_storage_semantics(v1) != replay_stage052_storage_semantics(changed_route)

    extra_axis = write(
        "artifact-storage-v2",
        "artifact_streaming",
        "stage05.2_artifact_streaming_attempt97",
        extra_solution_axis=True,
    )
    with pytest.raises(ArtifactIntegrityError, match="axis identity mismatch"):
        replay_stage052_storage_semantics(extra_axis)


def test_native_distance_matrix_matches_worked_euclidean_fixture() -> None:
    points = np.asarray(((0.0, 0.0), (3.0, 4.0), (6.0, 8.0)), dtype=np.float64)
    observed = distance_matrix(points)
    assert observed.tolist() == [
        [0.0, 5.0, 10.0],
        [5.0, 0.0, 5.0],
        [10.0, 5.0, 0.0],
    ]


def test_storage_replay_expands_compact_v2_screening_decisions(
    tmp_path: Path,
) -> None:
    def payloads() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
        raw = {
            "instance": "c101_21",
            "seed": 2014,
            "axes": {"fixed_work": {"started_calls": 0, "completed_calls": 0}},
        }
        solution = {
            "instance": "c101_21",
            "seed": 2014,
            "axes": {
                "fixed_work": {
                    "routes": [["C1"]],
                    "objective_key": [1, 10.0, 0.0, 0],
                }
            },
        }
        events = [
            {
                "event_type": "screening_decision",
                "record_type": "screening_decision",
                "benchmark_axis": "fixed_work",
                "route_key": "route:2:C1",
                "lane": "fixed_work:legacy",
                "operator": "relocate",
                "status": "pass",
                "reason": "",
                "decision_id": 1,
                "demand": 1.0,
                "distance_increment_lower_bound": None,
                "distance_lower_bound": 2.0,
                "exact_call_blocked": False,
                "first_failed_check": "",
                "min_time_window_slack": 3.0,
                "negative_cache_hit": False,
                "single_segment_reachable": True,
                "structural_energy_lower_bound": 4.0,
                "checks": [{"check": "capacity", "status": "pass", "value": True}],
            }
        ]
        return raw, solution, events

    v1 = tmp_path / "stage05.2_hot_path_attempt94"
    v1_writer = ArtifactBundleWriter(
        v1,
        ArtifactRunContext("stage05.2", "hot_path", v1.name),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v1"),
    )
    raw, solution, events = payloads()
    v1_writer.write_instance_seed(
        instance="c101_21",
        seed=2014,
        raw_payload=raw,
        solution_payload=solution,
        trace_payload={},
        environment_payload={},
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=events,
    )
    v1_writer.finalize()

    v2 = tmp_path / "stage05.2_artifact_streaming_attempt94"
    v2_writer = ArtifactBundleWriter(
        v2,
        ArtifactRunContext("stage05.2", "artifact_streaming", v2.name),
        ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    raw, solution, events = payloads()
    shard = v2_writer.open_v2_shard(
        instance="c101_21",
        seed=2014,
        shard_ordinal=0,
        worker_identity="worker-0",
    )
    shard.append(
        route_dictionary={"route:2:C1": ("C1",)},
        critical_events=events,
    )
    shard.finalize(
        raw_payload=raw,
        solution_payload=solution,
        trace_payload={},
        environment_payload={},
    )
    v2_writer.finalize()

    assert replay_stage052_storage_semantics(v1) == replay_stage052_storage_semantics(v2)
    v1_trace = ArtifactReader(v1).reconstruct_trace(
        "c101_21/2014/stage05.2_hot_path_attempt94_trace_c101_21_2014.json"
    )
    v2_trace = ArtifactReader(v2).reconstruct_trace(
        "c101_21/2014/stage05.2_artifact_streaming_attempt94_trace_c101_21_2014.json"
    )
    assert v1_trace["screening_decisions"] == v2_trace["screening_decisions"]


def test_definition_encoded_screening_requires_known_untampered_definition() -> None:
    definition = {
        "benchmark_axis": "fixed_work",
        "checks": [
            {
                "check": "capacity",
                "reason": "",
                "status": "pass",
                "value_bool": True,
                "value_float": 1.0,
                "value_text": None,
            }
        ],
        "lane_id": 1,
        "operator_id": 2,
        "route_id": 3,
        "status": "pass",
    }
    definition_json = json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
    definition_id = (
        int.from_bytes(hashlib.sha256(definition_json).digest()[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
    )
    definitions: dict[int, dict[str, object]] = {}
    first = {
        "event_id": 1,
        "definition_id": definition_id,
        "definition_json": definition_json,
        "started_at": 1.25,
        "completed_at": 1.5,
        "iteration": 4,
        "decision_id": 9,
    }

    expanded = expand_v2_screening_decision(first, definitions=definitions)
    assert expanded["embedded_checks"][0]["value"] is True
    assert expanded["timestamp_seconds"] == 1.25
    assert expanded["duration_seconds"] == 0.25
    second = {**first, "event_id": 2, "definition_json": None}
    assert expand_v2_screening_decision(second, definitions=definitions)["route_id"] == 3

    with pytest.raises(ArtifactIntegrityError, match="unknown definition"):
        expand_v2_screening_decision({**second, "definition_id": definition_id + 1}, definitions={})
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        expand_v2_screening_decision({**first, "definition_id": definition_id + 1}, definitions={})


def test_v2_final_flush_is_charged_to_persistence_and_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flush_delay_seconds = 0.03

    class FakeShard:
        def append(self, **_: object) -> int:
            return 1

        def flush(self) -> None:
            time.sleep(flush_delay_seconds)

        def finalize(self, **_: object) -> None:
            return None

        def abort(self, _: BaseException) -> None:
            return None

    class FakeWriter:
        def open_v2_shard(self, **_: object) -> FakeShard:
            return FakeShard()

    trace = SimpleNamespace(
        route_dictionary={},
        reconcile=lambda _result: {"status": "pass"},
        to_index_dict=lambda: {},
    )
    objective = SimpleNamespace(
        key=(1, 1.0, 0.0, 0),
        vehicle_count=1,
        total_distance=1.0,
        total_charging_time=0.0,
        charging_count=0,
    )
    result = SimpleNamespace(
        measurement_trace=trace,
        objective=objective,
        routes=(("D0", "D0"),),
        feasible=True,
        exact_started_calls=1,
        exact_completed_calls=1,
        neighborhood_events=(),
        charging_backend="cpu_batch",
        backend_metrics={"work_batches": 1, "exact_calls": 1},
        screening_statistics={},
        runtime_seconds=0.0,
        effective_iterations=1,
        termination_reason="fixed_work_budget",
    )
    monkeypatch.setattr(
        stage052_performance, "_solve_stage052_axis", lambda *args, **kwargs: result
    )
    monkeypatch.setattr(
        stage052_performance,
        "_iter_stage052_axis_events",
        lambda **kwargs: iter(()),
    )
    monkeypatch.setattr(
        stage052_performance,
        "validate_routes",
        lambda *args, **kwargs: SimpleNamespace(feasible=True),
    )
    monkeypatch.setattr(stage052_performance, "collect_environment", lambda: {})
    monkeypatch.setattr(stage052_performance, "_peak_rss_bytes", lambda: 1)

    axis = stage052_performance.Stage052Axis(
        name="fixed_work",
        termination_mode="fixed_work",
        time_limit_seconds=1.0,
        exact_call_budget=1,
    )
    task = _ShardTask(
        root=tmp_path,
        config_path=tmp_path / "unused.toml",
        run_dir=tmp_path / "results" / "stage05.2_artifact_streaming_attempt94",
        run_label="stage05.2_artifact_streaming_attempt94",
        component="artifact_streaming",
        scope="performance",
        instance_name="c101C5",
        customer_count=5,
        seed=2014,
        shard_ordinal=0,
        worker_count=1,
        storage=ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )

    rows = _run_and_persist_v2_shard(
        task,
        writer=FakeWriter(),  # type: ignore[arg-type]
        config=SimpleNamespace(),  # type: ignore[arg-type]
        stage04=SimpleNamespace(),
        stage02=SimpleNamespace(),
        instance=SimpleNamespace(),  # type: ignore[arg-type]
        axes=(axis,),
        storage=task.storage,
    )

    persistence = float(rows[0]["artifact_persistence_seconds"])
    solver_seconds = float(rows[0]["solver_seconds"])
    assert persistence >= flush_delay_seconds * 0.9
    assert float(rows[0]["end_to_end_seconds"]) == pytest.approx(solver_seconds + persistence)


def test_v2_pre_open_failure_publishes_partial_shard_evidence(
    tmp_path: Path,
) -> None:
    run_label = "stage05.2_artifact_streaming_attempt93"
    task = _ShardTask(
        root=tmp_path,
        config_path=tmp_path / "missing.toml",
        run_dir=tmp_path / "results" / run_label,
        run_label=run_label,
        component="artifact_streaming",
        scope="performance",
        instance_name="c101_21",
        customer_count=100,
        seed=2014,
        shard_ordinal=0,
        worker_count=1,
        storage=ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )

    with pytest.raises(FileNotFoundError):
        _run_v2_shard_task(task)

    prefix = task.run_dir / "c101_21" / "2014" / run_label
    failure = Path(f"{prefix}_failure_c101_21_2014.json")
    manifest = Path(f"{prefix}_shard_manifest_c101_21_2014.json")
    sidecar = Path(f"{prefix}_shard_manifest_c101_21_2014.sha256")
    assert failure.is_file()
    assert sidecar.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["evidence_completeness"] == "partial"


def test_v2_recovery_replaces_invalid_manifest_and_preserves_it(
    tmp_path: Path,
) -> None:
    run_label = "stage05.2_artifact_streaming_attempt92"
    task = _ShardTask(
        root=tmp_path,
        config_path=tmp_path / "missing.toml",
        run_dir=tmp_path / "results" / run_label,
        run_label=run_label,
        component="artifact_streaming",
        scope="performance",
        instance_name="c101_21",
        customer_count=100,
        seed=2014,
        shard_ordinal=0,
        worker_count=1,
        storage=ArtifactStorageConfig(storage_policy_version="artifact-storage-v2"),
    )
    directory = task.run_dir / task.instance_name / str(task.seed)
    directory.mkdir(parents=True)
    manifest = directory / (f"{run_label}_shard_manifest_{task.instance_name}_{task.seed}.json")
    original_manifest = b'{"truncated":true}'
    manifest.write_bytes(original_manifest)
    manifest.with_suffix(".sha256").write_text("wrong\n", encoding="utf-8")

    _ensure_partial_shard_failure(task, "worker exited")

    recovered = json.loads(manifest.read_text(encoding="utf-8"))
    sidecar = manifest.with_suffix(".sha256")
    assert recovered["evidence_completeness"] == "partial"
    assert (
        sidecar.read_text(encoding="utf-8").strip()
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    archived = tuple(
        directory.glob(f"{run_label}_partial_fragment_manifest_{task.instance_name}_*.bin")
    )
    assert len(archived) == 1
    assert archived[0].read_bytes() == original_manifest
