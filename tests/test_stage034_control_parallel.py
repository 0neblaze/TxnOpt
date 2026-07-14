from __future__ import annotations

import time
from pathlib import Path

import pytest

from evrptw.alns import solve_alns
from evrptw.artifacts import ArtifactBundleWriter, ArtifactRunContext, verify_manifest
from evrptw.cache_incremental import CacheIncrementalConfig
from evrptw.candidate_control import (
    CandidateControlConfig,
    CandidateControlRuntime,
    CandidateParallelExecutionError,
    CandidatePlan,
)
from evrptw.exact_deadline import ExactDeadlineConfig
from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES, FORMAL_SEEDS
from evrptw.experiments.stage033_exact_deadline import persist_paired_diagnostic
from evrptw.experiments.stage034_control_parallel import (
    DIAGNOSTIC_AXES,
    STAGE034_SCHEMA_VERSION,
    load_stage034_config,
    run_control_parallel_diagnostic,
)
from evrptw.experiments.stage034_control_parallel_review import evaluate_stage034_gate
from evrptw.measurement import CheapScreeningConfig, MeasurementConfig
from evrptw.models import Instance, Node, NodeType, Vehicle


def _instance() -> Instance:
    return Instance(
        "stage034_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("F1", NodeType.STATION, 5.0, 0.0, 0.0, 0.0, 200.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 8.0, 1.0, 1.0, 0.0, 100.0, 1.0),
            Node("C2", NodeType.CUSTOMER, 8.0, -1.0, 1.0, 0.0, 100.0, 1.0),
            Node("C3", NodeType.CUSTOMER, 2.0, 1.0, 1.0, 0.0, 100.0, 1.0),
        ),
        Vehicle(11.0, 3.0, 1.0, 0.1, 1.0),
    )


def test_candidate_control_is_opt_in_and_requires_cpu_batch() -> None:
    config = CandidateControlConfig(
        proposal_top_k=2,
        max_exact_calls_per_round=4,
        worker_count=1,
    )

    controlled = solve_alns(
        _instance(),
        seed=2014,
        max_iterations=8,
        time_limit_seconds=2.0,
        candidate_control_config=config,
    )

    assert controlled.feasible
    assert controlled.candidate_control_statistics["enabled"] is True
    assert controlled.candidate_control_statistics["proposal_top_k"] == 2
    assert controlled.candidate_control_statistics["worker_count"] == 1
    assert len(controlled.candidate_work_hash) == 64
    assert len(controlled.route_result_hash) == 64

    with pytest.raises(ValueError, match="cpu_batch"):
        solve_alns(
            _instance(),
            seed=2014,
            max_iterations=2,
            time_limit_seconds=1.0,
            backend="cpu_scalar",
            candidate_control_config=config,
        )


def test_candidate_control_ranks_and_budgets_each_complete_round() -> None:
    result = solve_alns(
        _instance(),
        seed=2014,
        max_iterations=8,
        time_limit_seconds=2.0,
        measurement_config=MeasurementConfig(),
        screening_config=CheapScreeningConfig(),
        cache_incremental_config=CacheIncrementalConfig(
            enabled=True,
            max_entries=64,
            max_memory_bytes=1_000_000,
        ),
        candidate_control_config=CandidateControlConfig(
            proposal_top_k=1,
            max_exact_calls_per_round=1,
            worker_count=1,
        ),
    )

    assert result.feasible
    assert result.candidate_control_statistics["candidate_decisions"] > 0
    assert result.candidate_control_statistics["skipped_candidates"] > 0
    assert result.measurement_trace is not None
    assert result.measurement_trace.trace_schema_version == "stage03-trace-v4"
    decisions = [
        event
        for event in result.measurement_trace.events
        if event.get("event_type") == "candidate_control_decision"
    ]
    assert {event["status"] for event in decisions} >= {"selected", "not_selected"}
    per_round: dict[int, int] = {}
    for event in result.measurement_trace.events:
        if event.get("event_type") != "candidate_control_budget":
            continue
        iteration = event.get("iteration")
        if iteration is not None:
            per_round[int(iteration)] = per_round.get(int(iteration), 0) + int(
                event["granted"]
            )
    assert per_round
    assert max(per_round.values()) <= 1


def test_complete_plan_ranking_is_vehicle_first_and_budget_is_atomic() -> None:
    runtime = CandidateControlRuntime(
        CandidateControlConfig(
            proposal_top_k=1,
            max_exact_calls_per_round=2,
            worker_count=1,
        )
    )
    plans = (
        CandidatePlan(0, (("C1",), ("C2",)), 2, 1.0, 1, 0),
        CandidatePlan(1, (("C1", "C2"),), 1, 100.0, 1, 1),
    )

    selected = runtime.select_plans(
        plans,
        lane="all",
        iteration=0,
        operator="test",
    )
    runtime.begin_round(0)

    assert selected == (plans[1],)
    assert runtime.reserve(3, atomic=True, context="complete_candidate") == 0
    assert runtime.round_remaining == 2


def test_controlled_parallelism_preserves_fixed_work_semantics() -> None:
    def run(worker_count: int):
        return solve_alns(
            _instance(),
            seed=2014,
            max_iterations=20,
            time_limit_seconds=10.0,
            measurement_config=MeasurementConfig(),
            screening_config=CheapScreeningConfig(),
            cache_incremental_config=CacheIncrementalConfig(
                enabled=True,
                max_entries=64,
                max_memory_bytes=1_000_000,
            ),
            exact_deadline_config=ExactDeadlineConfig.fixed_exact_calls(
                40,
                watchdog_seconds=10.0,
            ),
            candidate_control_config=CandidateControlConfig(
                proposal_top_k=4,
                max_exact_calls_per_round=4,
                worker_count=worker_count,
            ),
        )

    serial = run(1)
    parallel = run(4)

    assert serial.feasible and parallel.feasible
    assert serial.objective == parallel.objective
    assert serial.customer_sequences == parallel.customer_sequences
    assert serial.candidate_work_hash == parallel.candidate_work_hash
    assert serial.route_result_hash == parallel.route_result_hash
    assert serial.exact_started_calls == parallel.exact_started_calls
    assert serial.effective_iterations == parallel.effective_iterations
    assert parallel.candidate_control_statistics["parallel_batches"] > 0
    assert parallel.measurement_trace is not None
    parallel_events = [
        event
        for event in parallel.measurement_trace.events
        if event.get("event_type") == "parallel_batch"
        and event.get("status") == "parallel_complete"
    ]
    assert parallel_events
    assert all(
        event["merge_order"] == event["submission_order"]
        for event in parallel_events
    )


def test_spawn_worker_failure_and_deadline_do_not_publish_results() -> None:
    runtime = CandidateControlRuntime(
        CandidateControlConfig(
            proposal_top_k=4,
            max_exact_calls_per_round=4,
            worker_count=4,
        )
    )
    sequences = (("C1",), ("C2",), ("C3",), ("C1", "C2"))

    with pytest.raises(CandidateParallelExecutionError, match="deadline"):
        runtime.solve_batch(
            _instance(),
            sequences,
            batch_size=8,
            deadline=time.perf_counter() - 1.0,
            lane="test",
            iteration=0,
            operator="deadline",
        )
    first_result_hash = runtime.route_result_hash

    with pytest.raises(CandidateParallelExecutionError, match="ValueError"):
        runtime.solve_batch(
            _instance(),
            sequences,
            batch_size=0,
            deadline=time.perf_counter() + 10.0,
            lane="test",
            iteration=1,
            operator="worker_failure",
        )

    assert runtime.route_result_hash == first_result_hash
    assert not any(
        event.get("status") == "parallel_complete" for event in runtime.events
    )


def test_stage034_formal_gate_requires_quality_and_performance() -> None:
    rows = [
        {
            "instance": instance,
            "seed": seed,
            "axis": axis,
            "backend": "cpu_batch",
            "valid": True,
            "objective_not_worse": True,
            "started_calls": 100,
            "completed_calls": 100,
            "effective_iterations": 50,
            "unchanged_exact_calls": 0,
            "exact_reconciliation_valid": True,
            "candidate_reconciliation_valid": True,
            "parallel_ordering_valid": True,
            "paired_semantics_valid": True,
        }
        for instance in FORMAL_INSTANCES
        for seed in FORMAL_SEEDS
        for axis in DIAGNOSTIC_AXES
    ]

    ready = evaluate_stage034_gate(rows, scope="formal", prerequisites_valid=True)
    assert ready["status"] == "READY_FOR_STAGE04"
    assert ready["observed_rows"] == 144

    failed = [dict(row) for row in rows]
    for failed_row in failed:
        if (
            failed_row["instance"] == "r101_21"
            and failed_row["axis"] == "serial_wall_clock"
        ):
            failed_row["started_calls"] = 101
    not_ready = evaluate_stage034_gate(
        failed,
        scope="formal",
        prerequisites_valid=True,
    )
    assert not_ready["status"] == "NOT_READY"
    assert not_ready["performance_gate"] is False


def test_stage034_runner_persists_four_current_storage_axes(tmp_path) -> None:
    config = load_stage034_config(Path("configs/stage034_control_parallel.toml"))
    axes = run_control_parallel_diagnostic(
        _instance(),
        seed=2014,
        vehicle_operator_config=None,
        screening_config=config.screening_config,
        cache_incremental_config=config.cache_incremental_config,
        candidate_control_config=config.candidate_control_config,
        exact_call_budget=20,
        wall_clock_seconds=0.1,
        watchdog_seconds=2.0,
        max_iterations=4,
        batch_size=8,
    )
    writer = ArtifactBundleWriter(
        tmp_path / "stage03.4_control_parallel_attempt01",
        ArtifactRunContext(
            "stage03.4",
            "control_parallel",
            "stage03.4_control_parallel_attempt01",
        ),
        config.artifact_storage,
    )
    persist_paired_diagnostic(
        writer,
        instance=_instance(),
        seed=2014,
        pair=axes,
        scope="smoke",
        environment_payload={"peak_rss_bytes": 1},
        diagnostic_axes=DIAGNOSTIC_AXES,
        schema_version=STAGE034_SCHEMA_VERSION,
    )
    bundle = writer.finalize()
    manifest = verify_manifest(bundle.run_dir)

    assert manifest["stage_id"] == "stage03.4"
    assert manifest["component"] == "control_parallel"
    assert manifest["evidence_completeness"] == "complete"
