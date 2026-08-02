from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evrptw.charging import solve_exact_charging
from evrptw.experiments.stage052_native_architecture_review import (
    ReviewRecord,
    render_report,
    review_records,
)
from evrptw.experiments.stage052_native_architectures import (
    MODES,
    ArchitectureAxisTask,
    _run_group,
    build_axis_plan,
    expected_axis_count,
    rotated_modes,
    run_labels_for_scope,
)
from evrptw.native_scheduler import NativeHostScheduler
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.validation import validate_routes


def _plan(scope: str, tmp_path: Path):  # type: ignore[no-untyped-def]
    return build_axis_plan(
        scope,
        attempt=1,
        benchmark_dir=tmp_path / "benchmarks",
        output_root=tmp_path / "results",
        scheduler_socket_path=str(tmp_path / "scheduler.sock"),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        revision="c" * 40,
    )


def test_paired_plan_has_360_axes_and_rotates_all_five_modes(tmp_path: Path) -> None:
    plan = _plan("paired", tmp_path)

    assert len(plan) * len(MODES) == expected_axis_count("paired") == 360
    assert all(set(rotated_modes(task)) == set(MODES) for task in plan)
    first_mode_counts = Counter(rotated_modes(task)[0] for task in plan)
    assert max(first_mode_counts.values()) - min(first_mode_counts.values()) <= 1


def test_pilot_plan_has_180_wall_clock_axes_and_independent_labels(
    tmp_path: Path,
) -> None:
    plan = _plan("pilot", tmp_path)
    labels = run_labels_for_scope("pilot", 1)

    assert len(plan) * len(MODES) == expected_axis_count("pilot") == 180
    assert {task.axis for task in plan} == {"wall_clock_30"}
    assert len(set(labels.values())) == len(MODES)
    assert all("_pilot_attempt01" in label for label in labels.values())


def _review_fixture_records(root: Path) -> tuple[ReviewRecord, ...]:
    instance = parse_schneider(root / "data/schneider/c101C5.txt")
    charging = [solve_exact_charging(instance, (customer.name,)) for customer in instance.customers]
    assert all(result.feasible for result in charging)
    routes = [list(result.route) for result in charging]
    report = validate_routes(instance, routes)
    assert report.feasible
    objective = SolutionObjective.from_report(instance, report)
    records = []
    measurement_evidence: dict[str, Any] = {
        "present": True,
        "exact_route_order": [],
        "cache_lifecycle": [],
        "deadline_boundaries": [],
    }
    measurement_evidence["sha256"] = hashlib.sha256(
        json.dumps(
            measurement_evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    for mode in MODES:
        payload: dict[str, Any] = {
            "schema_version": "stage05.2-native-architecture-comparison-v1",
            "status": "completed",
            "mode": mode.value,
            "repeat": 0,
            "axis": "fixed_work",
            "instance": "c101C5",
            "seed": 2014,
            "routes": routes,
            "objective": list(objective.key),
            "solver_seconds": 1.0,
            "effective_iterations": 50,
            "exact_started_calls": 10,
            "exact_completed_calls": 10,
            "candidate_work_hash": "a" * 64,
            "route_result_hash": "b" * 64,
            "trajectory": [{"iteration": 1, "accepted": False}],
            "operator_statistics": {},
            "stage04_statistics": {},
            "stage04_events": [],
            "candidate_transaction_events": [],
            "measurement_evidence": measurement_evidence,
            "semantic_completeness": {
                "candidate_control": True,
                "stage04": True,
                "measurement_trace": True,
            },
            "fallback_count": 0,
            "native_execution_statistics": {"fallback_count": 0},
            "backend_metrics": {"launch_occupancies": [4]},
            "topology": {
                "rss_bytes": 1024,
                "cpu_utilization_percent_of_one_core": 100.0,
            },
            "cache_memory_bytes": 512,
            "artifact_bytes": 2048,
            "persistence_seconds": 0.01,
            "throughput": {
                "effective_iterations_per_second": 50.0,
                "candidate_transactions_per_second": 10.0,
                "screened_routes_per_second": 10.0,
                "exact_started_per_second": 10.0,
            },
        }
        records.append(ReviewRecord(root / f"{mode.value}.json", payload))
    return tuple(records)


def test_independent_review_replays_routes_and_accepts_equal_fixed_work() -> None:
    root = Path(__file__).resolve().parents[1]
    review = review_records(
        _review_fixture_records(root),
        scope="paired",
        benchmark_dir=root / "data/schneider",
    )

    assert review["axis_replay_passed"] is True
    gates = review["differential_gates"]
    assert isinstance(gates, dict)
    assert all(bool(gates[mode.value]["passed"]) for mode in MODES[2:])
    assert "五模式事实表" in render_report(review)


def test_one_wall_clock_group_runs_all_five_modes_with_one_scheduler(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    endpoint = tmp_path / "scheduler.sock"
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="wall_clock_30",
        instance_name="c101C5",
        seed=2014,
        benchmark_dir=root / "data/schneider",
        output_root=tmp_path,
        run_labels=run_labels_for_scope("paired", 99),
        scheduler_socket_path=str(endpoint),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        revision="c" * 40,
    )

    with NativeHostScheduler(endpoint):
        written = _run_group(task)

    assert len(written) == len(MODES)
    payloads = [json.loads(Path(path).read_bytes()) for path in written]
    assert {payload["mode"] for payload in payloads} == {mode.value for mode in MODES}
    assert all(payload["status"] == "completed" for payload in payloads)


def test_one_fixed_work_group_retains_every_mode_axis(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    endpoint = tmp_path / "scheduler.sock"
    task = ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="fixed_work",
        instance_name="c101C5",
        seed=2014,
        benchmark_dir=root / "data/schneider",
        output_root=tmp_path,
        run_labels=run_labels_for_scope("paired", 98),
        scheduler_socket_path=str(endpoint),
        wheel_sha256="a" * 64,
        native_sha256="b" * 64,
        revision="c" * 40,
    )

    with NativeHostScheduler(endpoint):
        written = _run_group(task)

    payloads = [json.loads(Path(path).read_bytes()) for path in written]
    assert len(payloads) == len(MODES)
    assert {payload["mode"] for payload in payloads} == {mode.value for mode in MODES}
    assert all(payload["status"] == "completed" for payload in payloads)
