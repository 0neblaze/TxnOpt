from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from txnopt_evidence.cli import main
from txnopt_evidence.codec import read_signed_json
from txnopt_evidence.reviewer import replay_manifest, verify_raw_manifest
from txnopt_evidence.runner import run_config_file

ROOT = Path(__file__).resolve().parents[2]


def _rcpsp_config(
    tmp_path: Path, *, run_label: str = "txnopt_rcpsp_attempt01"
) -> dict[str, object]:
    zero = {"duration": 0, "renewable_demands": [0]}
    work = {"duration": 2, "renewable_demands": [2]}
    return {
        "schema_version": "txnopt-run-config-v1",
        "run_label": run_label,
        "output_root": str(tmp_path / "raw"),
        "run_config": {
            "seed": 2014,
            "workers": 1,
            "execution_mode": "serial",
            "fixed_work": 4,
            "trace_policy": "semantic_and_physical",
            "max_rounds": 1,
        },
        "case": {
            "domain": "rcpsp",
            "oracle_seed": 2014,
            "max_block_size": 2,
            "instance": {
                "name": "tiny-evidence-rcpsp",
                "renewable_capacities": [2],
                "activities": [
                    {"activity_id": 0, "predecessors": [], "modes": [zero]},
                    {"activity_id": 1, "predecessors": [0], "modes": [work]},
                    {"activity_id": 2, "predecessors": [0], "modes": [work]},
                    {"activity_id": 3, "predecessors": [1, 2], "modes": [zero]},
                ],
            },
            "initial_state": {
                "activity_order": [0, 1, 2, 3],
                "mode_vector": [0, 0, 0, 0],
            },
        },
    }


def _evrptw_config(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": "txnopt-run-config-v1",
        "run_label": "txnopt_evrptw_attempt01",
        "output_root": str(tmp_path / "raw"),
        "run_config": {
            "seed": 2014,
            "workers": 1,
            "execution_mode": "serial",
            "fixed_work": 8,
            "trace_policy": "semantic_and_physical",
            "max_rounds": 1,
        },
        "case": {
            "domain": "evrptw",
            "backend": "python",
            "max_candidates": 8,
            "initial_plan": [["C1"], ["C2"]],
            "instance": {
                "name": "tiny-evidence-evrptw",
                "nodes": [
                    {
                        "name": "D",
                        "kind": "d",
                        "x": 0.0,
                        "y": 0.0,
                        "demand": 0.0,
                        "ready_time": 0.0,
                        "due_date": 100.0,
                        "service_time": 0.0,
                    },
                    {
                        "name": "C1",
                        "kind": "c",
                        "x": 1.0,
                        "y": 0.0,
                        "demand": 1.0,
                        "ready_time": 0.0,
                        "due_date": 100.0,
                        "service_time": 0.0,
                    },
                    {
                        "name": "C2",
                        "kind": "c",
                        "x": 2.0,
                        "y": 0.0,
                        "demand": 1.0,
                        "ready_time": 0.0,
                        "due_date": 100.0,
                        "service_time": 0.0,
                    },
                ],
                "vehicle": {
                    "battery_capacity": 100.0,
                    "load_capacity": 10.0,
                    "consumption_rate": 1.0,
                    "inverse_refueling_rate": 0.1,
                    "average_velocity": 1.0,
                },
            },
        },
    }


def _write_config(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.mark.parametrize("factory", [_rcpsp_config, _evrptw_config])
def test_runner_writes_raw_only_and_reviewer_reconstructs_both_domains(
    tmp_path: Path,
    factory: object,
) -> None:
    config_path = tmp_path / "run.json"
    payload = factory(tmp_path)  # type: ignore[operator]
    _write_config(config_path, payload)
    raw = run_config_file(config_path)

    manifest = verify_raw_manifest(raw.manifest_path)
    assert manifest["runner_decision"] is None
    assert manifest["fallback_count"] == 0
    assert manifest["producer_identity"]["binding_status"] == "UNBOUND_TEST_ONLY"
    review = replay_manifest(raw.manifest_path, output_dir=tmp_path / "review")
    assert review["status"] == "PASS"
    assert review["prefix_safety"] == "PASS"
    assert review["case_replay"]["validator_status"] == "PASS"
    assert review["physical_event_count"] == 1
    assert review["readiness_decision"] is None


def test_reviewer_detects_raw_tampering_before_replay(tmp_path: Path) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path))
    raw = run_config_file(config_path)
    events = raw.manifest_path.parent / "events.jsonl"
    events.write_bytes(events.read_bytes() + b"{}\n")

    with pytest.raises(ValueError, match="differs"):
        verify_raw_manifest(raw.manifest_path)


def test_run_and_verify_cli_are_live_without_fallback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path, run_label="txnopt_rcpsp_attempt02"))
    assert main(["run", "--config", str(config_path)]) == 0
    run_output = json.loads(capsys.readouterr().out)
    assert run_output["fallback_count"] == 0

    assert main(["verify", run_output["manifest_path"]]) == 0
    verify_output = json.loads(capsys.readouterr().out)
    assert verify_output["status"] == "verified"
    assert verify_output["fallback_count"] == 0


def test_cli_replay_runs_in_a_fresh_process_and_writes_signed_review(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path))
    raw = run_config_file(config_path)
    review_dir = tmp_path / "independent-review"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "txnopt_evidence.cli",
            "replay",
            str(raw.manifest_path),
            "--output-dir",
            str(review_dir),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    review = read_signed_json(review_dir / "review.json")
    assert review["status"] == "PASS"
    assert "txnopt_evidence.runner" not in completed.stdout


def test_runner_never_overwrites_an_existing_attempt(tmp_path: Path) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path))
    run_config_file(config_path)

    with pytest.raises(FileExistsError, match="already exists"):
        run_config_file(config_path)
