from __future__ import annotations

import numpy as np
import pytest

from evrptw._core import distance_matrix
from evrptw.experiments.stage052_performance import (
    PERFORMANCE_INSTANCES,
    PERFORMANCE_SEEDS,
    axes_for_scope,
    validate_stage052_run_label,
    verify_stage051_prerequisite,
)
from evrptw.experiments.stage052_performance_review import (
    validate_per_run_scope,
)
from evrptw.models import Instance, Node, NodeType, Vehicle
from evrptw.stage052 import (
    AcceleratorDecision,
    PerformanceObservation,
    Stage052Component,
    decide_accelerator,
    evaluate_promotion,
    formal_budget_matrix,
    select_worker_count,
)


def _observation(
    instance: str,
    seed: int,
    seconds: float,
    *,
    semantic_digest: str = "same",
) -> PerformanceObservation:
    return PerformanceObservation(
        instance=instance,
        seed=seed,
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


def test_worker_selection_is_fail_fast_and_memory_bounded() -> None:
    assert select_worker_count({1: 100.0, 2: 60.0, 4: 38.0}, {1: 4, 2: 8, 4: 11}) == 4
    assert select_worker_count({1: 100.0, 2: 60.0, 4: 50.0}, {1: 4, 2: 8, 4: 11}) == 2
    with pytest.raises(ValueError, match="NOT_READY"):
        select_worker_count({1: 100.0, 2: 80.0, 4: 30.0}, {1: 4, 2: 8, 4: 11})


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
    )
    assert instance.by_name is instance.by_name
    assert instance.customers is instance.customers
    assert instance.distance("D0", "C1") == 5.0


def test_stage052_runner_contract_has_canonical_labels_and_axes(tmp_path) -> None:
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


def test_native_distance_matrix_matches_worked_euclidean_fixture() -> None:
    points = np.asarray(((0.0, 0.0), (3.0, 4.0), (6.0, 8.0)), dtype=np.float64)
    observed = distance_matrix(points)
    assert observed.tolist() == [
        [0.0, 5.0, 10.0],
        [5.0, 0.0, 5.0],
        [10.0, 5.0, 0.0],
    ]
