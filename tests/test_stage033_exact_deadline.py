from __future__ import annotations

import json
from pathlib import Path

import pytest

from evrptw.alns import solve_alns
from evrptw.artifacts import (
    ArtifactBundleWriter,
    ArtifactReader,
    ArtifactRunContext,
    ArtifactStorageConfig,
    verify_manifest,
)
from evrptw.cache_incremental import CacheIncrementalConfig
from evrptw.cpu_batch import ExactBatchDeadlineExceeded, solve_exact_charging_batch
from evrptw.exact_deadline import ExactDeadlineConfig
from evrptw.experiments.stage033_exact_deadline import (
    load_stage033_config,
    persist_paired_diagnostic,
    run_paired_diagnostic,
    validate_stage033_run_label,
)
from evrptw.experiments.stage033_exact_deadline_review import evaluate_stage033_gate
from evrptw.measurement import CheapScreeningConfig, MeasurementConfig, Stage03Trace
from evrptw.models import Instance, Node, NodeType, Vehicle


def _instance() -> Instance:
    return Instance(
        "stage033_toy",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("F1", NodeType.STATION, 3.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 100.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 100.0, 0.0),
            Node("C3", NodeType.CUSTOMER, 6.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(20.0, 10.0, 1.0, 1.0, 1.0),
    )


def _solve(*, budget: int = 10, backend: str = "cpu_batch"):
    return solve_alns(
        _instance(),
        seed=2014,
        max_iterations=1000,
        time_limit_seconds=10.0,
        operator_profile="stage02_constraint_guided",
        measurement_config=MeasurementConfig(),
        screening_config=CheapScreeningConfig(),
        cache_incremental_config=CacheIncrementalConfig(
            enabled=True,
            max_entries=64,
            max_memory_bytes=1_000_000,
        ),
        backend=backend,
        exact_deadline_config=ExactDeadlineConfig.fixed_exact_calls(
            budget,
            watchdog_seconds=10.0,
        ),
    )


def test_exact_deadline_configuration_is_mode_safe() -> None:
    fixed = ExactDeadlineConfig.fixed_exact_calls(100, watchdog_seconds=120.0)
    wall_clock = ExactDeadlineConfig.wall_clock()

    assert fixed.mode == "exact_call_budget"
    assert fixed.exact_call_budget == 100
    assert fixed.watchdog_seconds == 120.0
    assert wall_clock.mode == "wall_clock"
    assert wall_clock.exact_call_budget is None

    with pytest.raises(ValueError, match="positive"):
        ExactDeadlineConfig.fixed_exact_calls(0, watchdog_seconds=120.0)


def test_stage033_config_requires_cpu_batch() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_stage033_config(root / "configs/stage033_exact_deadline.toml")

    assert config.exact_backend == "cpu_batch"


def test_stage033_rejects_scalar_backend_without_fallback() -> None:
    with pytest.raises(ValueError, match="cpu_batch"):
        _solve(backend="cpu_scalar")


def test_fixed_exact_call_budget_stops_at_cap_and_keeps_complete_incumbent() -> None:
    result = _solve(budget=10)

    assert result.feasible
    assert result.charging_backend == "cpu_batch"
    assert result.exact_started_calls == 10
    assert result.exact_completed_calls == 10
    assert result.exact_interrupted_calls == 0
    assert result.exact_budget_exhaustions == 1
    assert result.termination_reason == "exact_call_budget_exhausted"
    assert result.objective is not None
    assert result.measurement_trace is not None
    assert result.measurement_trace.started_calls == 10
    assert result.measurement_trace.completed_calls == 10
    assert any(
        event.get("event_type") == "exact_budget_boundary"
        for event in result.measurement_trace.events
    )
    rollbacks = [
        event
        for event in result.measurement_trace.events
        if event.get("event_type") == "candidate_cache_rollback"
    ]
    assert rollbacks
    for rollback in rollbacks:
        assert not any(
            event.get("event_type") == "cache_event"
            and event.get("operation") == "store"
            and event.get("lane") == rollback.get("lane")
            and event.get("iteration") == rollback.get("iteration")
            and event.get("operator") == rollback.get("operator")
            for event in result.measurement_trace.events
        )
    assert Stage03Trace.from_dict(result.measurement_trace.to_dict()).to_dict() == (
        result.measurement_trace.to_dict()
    )
    assert {
        record.status for record in result.measurement_trace.route_evaluations
        if record.kind == "exact_call"
    } <= {"completed_feasible", "completed_infeasible"}


def test_cpu_batch_checkpoint_reports_auditable_interruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = 0.0

    def advancing_clock() -> float:
        nonlocal current
        current += 0.001
        return current

    monkeypatch.setattr("evrptw.cpu_batch.time.perf_counter", advancing_clock)

    with pytest.raises(ExactBatchDeadlineExceeded) as captured:
        solve_exact_charging_batch(
            _instance(),
            (("C1", "C2", "C3"), ("C3", "C2", "C1")),
            deadline=0.006,
        )

    error = captured.value
    assert error.started_exact_calls == 2
    assert error.completed_exact_calls + error.interrupted_exact_calls == 2
    assert error.metrics.backend == "cpu_batch"
    assert error.metrics.batch_launches == 1
    assert error.metrics.checkpoint_count > 0
    assert error.metrics.packing_seconds >= 0.0
    assert error.metrics.unpacking_seconds >= 0.0


def test_stage033_paired_diagnostic_uses_only_cpu_batch() -> None:
    pair = run_paired_diagnostic(
        _instance(),
        seed=2014,
        vehicle_operator_config=None,
        screening_config=CheapScreeningConfig(),
        cache_incremental_config=CacheIncrementalConfig(
            enabled=True,
            max_entries=64,
            max_memory_bytes=1_000_000,
        ),
        exact_call_budget=10,
        wall_clock_seconds=1.0,
        watchdog_seconds=10.0,
        max_iterations=3,
        batch_size=8,
    )

    assert set(pair) == {"fixed_exact_calls", "wall_clock"}
    assert all(result.charging_backend == "cpu_batch" for result in pair.values())
    assert pair["fixed_exact_calls"].exact_started_calls <= 10
    assert pair["wall_clock"].exact_deadline_statistics["mode"] == "wall_clock"


def test_stage033_labels_and_scope_gate_are_canonical() -> None:
    validate_stage033_run_label("stage03.3_exact_deadline_attempt01")
    validate_stage033_run_label("stage03.3_exact_deadline_rerun99")
    with pytest.raises(ValueError, match="canonical"):
        validate_stage033_run_label("stage033_exact_deadline_attempt01")

    rows = [
        {
            "instance": instance,
            "seed": seed,
            "axis": axis,
            "backend": "cpu_batch",
            "valid": True,
            "started_calls": 100 if axis == "fixed_exact_calls" else 120,
            "completed_calls": 100 if axis == "fixed_exact_calls" else 120,
            "exact_call_budget": 100 if axis == "fixed_exact_calls" else None,
        }
        for instance in ("c101C5", "r105C5", "rc105C5", "c101_21", "r101_21", "rc101_21")
        for seed in (2014, 2015, 2016)
        for axis in ("fixed_exact_calls", "wall_clock")
    ]
    gate = evaluate_stage033_gate(rows, scope="smoke")

    assert gate["status"] == "READY_FOR_STAGE033_FORMAL"
    rows[0]["backend"] = "cpu_scalar"
    assert evaluate_stage033_gate(rows, scope="smoke")["status"] == "NOT_READY"


def test_stage033_paired_artifact_keeps_axes_and_backend_evidence(tmp_path: Path) -> None:
    run_label = "stage03.3_exact_deadline_attempt01"
    run_dir = tmp_path / "results" / run_label
    writer = ArtifactBundleWriter(
        run_dir,
        ArtifactRunContext("stage03.3", "exact_deadline", run_label),
        ArtifactStorageConfig(),
    )
    pair = run_paired_diagnostic(
        _instance(),
        seed=2014,
        vehicle_operator_config=None,
        screening_config=CheapScreeningConfig(),
        cache_incremental_config=CacheIncrementalConfig(
            enabled=True,
            max_entries=64,
            max_memory_bytes=1_000_000,
        ),
        exact_call_budget=10,
        wall_clock_seconds=1.0,
        watchdog_seconds=10.0,
        max_iterations=3,
        batch_size=8,
    )

    paths = persist_paired_diagnostic(
        writer,
        instance=_instance(),
        seed=2014,
        pair=pair,
        scope="smoke",
        environment_payload={"repository_dirty": False},
    )
    writer.finalize()

    manifest = verify_manifest(run_dir)
    assert manifest["component"] == "exact_deadline"
    reader = ArtifactReader(run_dir)
    solution = reader.read_json(paths["solution"])
    assert set(solution["axes"]) == {"fixed_exact_calls", "wall_clock"}
    events = reader.read_events(paths["events"])
    axes = {
        json.loads(str(row["extras_json"]))["diagnostic_axis"]
        for row in events
        if "diagnostic_axis" in json.loads(str(row["extras_json"]) or "{}")
    }
    assert axes == {"fixed_exact_calls", "wall_clock"}
