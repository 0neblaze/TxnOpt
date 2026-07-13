from __future__ import annotations

import json
from pathlib import Path

import pytest

import evrptw.experiments.stage03_measurement_review as stage03_review
from evrptw.experiments.stage03_measurement import (
    SMOKE_INSTANCES,
    _scope_instances,
    _write_manifest,
)
from evrptw.experiments.stage03_measurement_review import (
    _candidate_state_ok,
    _publish_summaries,
    _verify_manifest,
)
from evrptw.measurement import MeasurementConfig, Stage03Trace


def test_stage03_scope_is_exact_and_unique() -> None:
    assert _scope_instances("smoke") == SMOKE_INSTANCES
    assert len(SMOKE_INSTANCES) == len(set(SMOKE_INSTANCES))
    with pytest.raises(ValueError, match="scope"):
        _scope_instances("formal-ish")


@pytest.mark.parametrize("filename", ("raw.json", "solution.json", "trace.json", "event.jsonl"))
def test_stage03_manifest_rejects_tampered_raw_artifact(
    tmp_path: Path, filename: str
) -> None:
    payload_path = tmp_path / filename
    payload_path.write_text('{"value": 1}\n', encoding="utf-8")
    _write_manifest(tmp_path)

    _verify_manifest(tmp_path)
    payload_path.write_text('{"value": 2}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        _verify_manifest(tmp_path)


def test_stage03_manifest_rejects_tampered_manifest_or_sidecar(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('"1"', '"2"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sidecar hash mismatch"):
        _verify_manifest(tmp_path)

    _write_manifest(tmp_path)
    (tmp_path / "manifest.sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar hash mismatch"):
        _verify_manifest(tmp_path)


def test_stage03_summary_publish_rejects_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_paths = {}
    for key in (
        "review_report",
        "review_findings",
        "stage03_readiness",
        "trace_reconciliation",
        "deadline_report",
        "baseline_comparison",
        "recomputed_per_run",
        "summary_results",
        "review_manifest",
    ):
        path = tmp_path / f"{key}.txt"
        path.write_text("raw\n", encoding="utf-8")
        output_paths[key] = path
    monkeypatch.setattr(stage03_review, "_repository_root", lambda: tmp_path)
    summary_dir = tmp_path / "experiments" / "summaries"
    summary_dir.mkdir(parents=True)
    (summary_dir / "run_review_report.md").write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exist"):
        _publish_summaries(summary_dir, "run", output_paths)


def test_stage03_candidate_auditor_rejects_incomplete_or_vehicle_increasing_acceptance() -> None:
    trace = Stage03Trace(MeasurementConfig())
    trace.register_route(("C1",))
    trace.events.append(
        {
            "event_type": "candidate_state",
            "current_route_keys": ["missing"],
            "candidate_route_keys": [],
            "accepted": True,
            "candidate_feasible": True,
            "candidate_objective_key": [1, 1.0, 0.0, 0],
            "candidate_vehicle_count": 2,
            "current_vehicle_count": 1,
            "status": "accepted",
        }
    )

    assert _candidate_state_ok(trace) is False


def test_stage03_trace_round_trip_preserves_route_dictionary() -> None:
    trace = Stage03Trace(MeasurementConfig())
    trace.record_route_evaluation(
        ("C1", "C2"),
        lane="legacy",
        iteration=0,
        operator="initial_solution",
        kind="exact_call",
        exact_started=True,
        exact_completed=True,
        feasible=True,
        failure_reason="",
    )

    restored = Stage03Trace.from_dict(json.loads(json.dumps(trace.to_dict())))

    assert restored.route_dictionary == trace.route_dictionary
    assert restored.route_evaluations[0].route_key == trace.route_evaluations[0].route_key
