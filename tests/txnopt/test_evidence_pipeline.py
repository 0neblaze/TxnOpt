from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from txnopt_evidence.cli import main
from txnopt_evidence.codec import read_signed_json
from txnopt_evidence.reviewer import (
    _validate_event_stream,
    _validate_physical_trace,
    replay_manifest,
    verify_failure_manifest,
    verify_raw_manifest,
)
from txnopt_evidence.runner import RunExecutionError, run_config_file

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
    assert review["aggregate_refinement_replay"] == "PASS"
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


def test_invalid_config_is_rejected_before_a_run_directory_is_created(tmp_path: Path) -> None:
    payload = _rcpsp_config(tmp_path, run_label="txnopt_invalid_attempt01")
    run_config = payload["run_config"]
    assert isinstance(run_config, dict)
    run_config["workers"] = 0
    config_path = tmp_path / "invalid-run.json"
    _write_config(config_path, payload)

    with pytest.raises(ValueError, match="workers"):
        run_config_file(config_path)

    assert not (tmp_path / "raw" / "txnopt_invalid_attempt01").exists()


def test_reviewer_accepts_one_run_observation_followed_by_audited_t4_events() -> None:
    _validate_physical_trace(
        (
            {
                "event": "run_observation",
                "trace": "txnopt-physical-trace-v1",
                "execution_mode": "serial",
                "workers": 1,
                "started_ns": 10,
                "ended_ns": 20,
                "duration_ns": 10,
                "termination_reason": "runtime_contract_error",
                "observed_cmax_upper_ns": 10,
            },
            {
                "event": "t4_waste_observation",
                "trace": "txnopt-physical-trace-v1",
                "bound_satisfied": True,
                "remaining_budget_before": 10,
                "uncommitted_window": 4,
                "max_requests_per_candidate": 2,
                "post_boundary_capacity_units": 4,
                "observed_discarded_work_units": 3,
                "observed_post_boundary_work_units": 1,
                "discarded_work_bound_units": 8,
                "post_boundary_work_bound_units": 4,
                "cost_basis": "measured_transaction_elapsed_upper_bound_ns",
                "measured_cmax_ns": 10,
                "observed_discarded_cost_upper_ns": 30,
                "discarded_cost_bound_ns": 80,
                "observed_post_boundary_cost_upper_ns": 10,
                "post_boundary_cost_bound_ns": 40,
            },
        )
    )


def test_reviewer_replays_native_prepared_round_observations() -> None:
    run = {
        "event": "run_observation",
        "trace": "txnopt-physical-trace-v1",
        "execution_mode": "barrier",
        "workers": 2,
        "started_ns": 1,
        "ended_ns": 4,
        "duration_ns": 3,
        "termination_reason": "max_rounds",
        "observed_cmax_upper_ns": 3,
    }
    native = {
        "event": "native_round_observation",
        "trace": "txnopt-physical-trace-v1",
        "protocol": "txnopt-native-round-v1",
        "phase": "VALIDATED",
        "phase_trace": ["PREPARED", "RESERVED", "EVALUATING", "VALIDATED"],
        "context_pack_count": 1,
        "round_call_count": 2,
        "worker_count": 2,
        "scheduled_worker_count": 2,
        "execution_policy": "parallel",
        "parallel_route_threshold": 4,
        "started_work": 4,
        "completed_work": 4,
        "interrupted_work": 0,
        "budget_limit": 4,
        "budget_reserved_work": 4,
        "budget_remaining_work": 0,
        "prepared_cache_write_count": 4,
        "prepared_cache_key_checksum": 1,
        "semantic_event_count": 4,
        "task_receipt_count": 2,
        "task_receipts": [
            [0, 0, 0, 2, 1, 2, 3],
            [1, 1, 2, 4, 1, 2, 3],
        ],
        "fallback_count": 0,
        "source_revision": "a" * 40,
        "source_tree": "b" * 40,
    }

    _validate_physical_trace((run, native), semantic_exact_work_started=True)
    native["prepared_cache_write_count"] = 3
    with pytest.raises(ValueError, match="prepared delta"):
        _validate_physical_trace((run, native), semantic_exact_work_started=True)
    native["prepared_cache_write_count"] = 4
    native["task_receipts"][1][2] = 3
    with pytest.raises(ValueError, match="task receipt is malformed"):
        _validate_physical_trace((run, native), semantic_exact_work_started=True)


def test_reviewer_recomputes_t4_bounds_and_rejects_a_claimed_pass() -> None:
    with pytest.raises(ValueError, match="recompute independently"):
        _validate_physical_trace(
            (
                {
                    "event": "run_observation",
                    "trace": "txnopt-physical-trace-v1",
                    "execution_mode": "serial",
                    "workers": 1,
                    "started_ns": 10,
                    "ended_ns": 20,
                    "duration_ns": 10,
                    "termination_reason": "runtime_contract_error",
                    "observed_cmax_upper_ns": 10,
                },
                {
                    "event": "t4_waste_observation",
                    "trace": "txnopt-physical-trace-v1",
                    "bound_satisfied": True,
                    "remaining_budget_before": 4,
                    "uncommitted_window": 1,
                    "max_requests_per_candidate": 1,
                    "post_boundary_capacity_units": 0,
                    "observed_discarded_work_units": 2,
                    "observed_post_boundary_work_units": 0,
                    "discarded_work_bound_units": 1,
                    "post_boundary_work_bound_units": 0,
                    "measured_cmax_ns": 10,
                },
            )
        )


def test_reviewer_requires_cmax_for_live_post_boundary_work() -> None:
    with pytest.raises(ValueError, match="measured run Cmax"):
        _validate_physical_trace(
            (
                {
                    "event": "run_observation",
                    "trace": "txnopt-physical-trace-v1",
                    "execution_mode": "ordered",
                    "workers": 4,
                    "started_ns": 10,
                    "ended_ns": 20,
                    "duration_ns": 10,
                    "termination_reason": "worker_failure",
                },
                {
                    "event": "t4_waste_observation",
                    "trace": "txnopt-physical-trace-v1",
                    "bound_satisfied": True,
                    "remaining_budget_before": 4,
                    "uncommitted_window": 1,
                    "max_requests_per_candidate": 1,
                    "post_boundary_capacity_units": 1,
                    "observed_discarded_work_units": 0,
                    "observed_post_boundary_work_units": 1,
                    "discarded_work_bound_units": 1,
                    "post_boundary_work_bound_units": 1,
                },
            )
        )


def test_reviewer_requires_run_cmax_when_semantic_exact_work_committed() -> None:
    with pytest.raises(ValueError, match="semantic exact work"):
        _validate_physical_trace(
            (
                {
                    "event": "run_observation",
                    "trace": "txnopt-physical-trace-v1",
                    "execution_mode": "serial",
                    "workers": 1,
                    "started_ns": 10,
                    "ended_ns": 20,
                    "duration_ns": 10,
                    "termination_reason": "max_rounds",
                },
            ),
            semantic_exact_work_started=True,
        )


def test_reviewer_rejects_commit_without_started_work_ledger() -> None:
    digest = "0" * 64
    next_digest = "1" * 64
    with pytest.raises(ValueError, match="started-work ledger"):
        _validate_event_stream(
            (
                {"event": "run_open", "state_digest": digest},
                {"event": "candidate_transaction", "txn_id": "a", "phase": "PREPARED"},
                {"event": "candidate_transaction", "txn_id": "a", "phase": "RESERVED"},
                {"event": "candidate_transaction", "txn_id": "a", "phase": "EVALUATING"},
                {"event": "candidate_transaction", "txn_id": "a", "phase": "VALIDATED"},
                {
                    "event": "candidate_transaction",
                    "txn_id": "a",
                    "phase": "COMMITTED",
                    "state_digest": next_digest,
                },
                {"event": "run_terminated", "reason": "max_rounds"},
            ),
            {
                "termination_reason": "max_rounds",
                "state_digest": next_digest,
                "fallback_count": 0,
            },
        )


def test_runner_retains_a_signed_reviewable_bundle_for_fail_fast_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "0" * 64

    def fail_case(
        _case: object,
        *,
        config: object,
        semantic_sink: object,
        physical_sink: object,
    ) -> dict[str, object]:
        del config
        semantic_sink(  # type: ignore[operator]
            (
                {"event": "run_open", "state_digest": digest},
                {"event": "candidate_screening", "admitted_count": 1},
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-00000000",
                    "phase": "PREPARED",
                    "snapshot_digest": digest,
                    "candidate_keys": ["candidate-1"],
                },
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-00000000",
                    "phase": "RESERVED",
                    "snapshot_digest": digest,
                    "candidate_keys": ["candidate-1"],
                },
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-00000000",
                    "phase": "EVALUATING",
                    "snapshot_digest": digest,
                    "candidate_keys": ["candidate-1"],
                },
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-00000000",
                    "phase": "ABORTED",
                    "snapshot_digest": digest,
                    "candidate_keys": ["candidate-1"],
                },
                {"event": "run_terminated", "reason": "runtime_contract_error"},
            )
        )
        physical_sink(  # type: ignore[operator]
            (
                {
                    "event": "run_observation",
                    "trace": "txnopt-physical-trace-v1",
                    "execution_mode": "serial",
                    "workers": 1,
                    "started_ns": 10,
                    "ended_ns": 20,
                    "duration_ns": 10,
                    "termination_reason": "runtime_contract_error",
                    "observed_cmax_upper_ns": 10,
                },
                {
                    "event": "t4_waste_observation",
                    "trace": "txnopt-physical-trace-v1",
                    "bound_satisfied": True,
                    "remaining_budget_before": 4,
                    "uncommitted_window": 1,
                    "max_requests_per_candidate": 1,
                    "post_boundary_capacity_units": 0,
                    "observed_discarded_work_units": 1,
                    "observed_post_boundary_work_units": 0,
                    "discarded_work_bound_units": 1,
                    "post_boundary_work_bound_units": 0,
                    "cost_basis": "measured_transaction_elapsed_upper_bound_ns",
                    "measured_cmax_ns": 10,
                    "observed_discarded_cost_upper_ns": 10,
                    "discarded_cost_bound_ns": 10,
                    "observed_post_boundary_cost_upper_ns": 0,
                    "post_boundary_cost_bound_ns": 0,
                },
            )
        )
        raise RuntimeError("simulated fail-fast producer error")

    monkeypatch.setattr("txnopt_evidence.runner.execute_case", fail_case)
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path, run_label="txnopt_failure_attempt01"))

    with pytest.raises(RunExecutionError) as captured:
        run_config_file(config_path)

    manifest_path = captured.value.manifest_path
    verified = verify_failure_manifest(manifest_path)
    assert verified["runner_decision"] == "FAILED"
    assert read_signed_json(manifest_path.parent / "failure.json")["fallback_count"] == 0


@pytest.mark.parametrize(
    "fault",
    ["empty_semantic", "empty_physical", "non_object_result"],
)
def test_runner_preflight_faults_remain_reviewable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    digest = "0" * 64

    def malformed_case(
        _case: object,
        *,
        config: object,
        semantic_sink: object,
        physical_sink: object,
    ) -> object:
        del config
        if fault == "empty_semantic":
            semantic_sink(())  # type: ignore[operator]
        else:
            semantic_sink(  # type: ignore[operator]
                (
                    {"event": "run_open", "state_digest": digest},
                    {"event": "candidate_screening", "admitted_count": 0},
                    {"event": "round_terminated", "reason": "no_candidates"},
                    {"event": "run_terminated", "reason": "no_candidates"},
                )
            )
        if fault == "empty_physical":
            physical_sink(())  # type: ignore[operator]
        return [] if fault == "non_object_result" else {"status": "unused"}

    monkeypatch.setattr("txnopt_evidence.runner.execute_case", malformed_case)
    config_path = tmp_path / f"{fault}.json"
    _write_config(
        config_path,
        _rcpsp_config(tmp_path, run_label=f"txnopt_{fault}_attempt01"),
    )

    with pytest.raises(RunExecutionError) as caught:
        run_config_file(config_path)

    manifest = verify_failure_manifest(caught.value.manifest_path)
    assert manifest["runner_decision"] == "FAILED"


def test_aggregate_refinement_replay_rejects_visible_intermediate_state() -> None:
    from txnopt_evidence.refinement import (
        RefinementReplayError,
        replay_aggregate_refinement,
    )

    digest = "0" * 64
    with pytest.raises(RefinementReplayError, match="private transaction step"):
        replay_aggregate_refinement(
            (
                {"event": "run_open", "state_digest": digest},
                {"event": "candidate_screening", "admitted_count": 1},
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-00000000",
                    "phase": "PREPARED",
                    "snapshot_digest": digest,
                    "candidate_keys": ["candidate-1"],
                    "state_digest": digest,
                },
                {"event": "run_terminated", "reason": "runtime_contract_error"},
            )
        )


def test_aggregate_refinement_rejects_interleaved_transaction_owners() -> None:
    from txnopt_evidence.refinement import (
        RefinementReplayError,
        replay_aggregate_refinement,
    )

    digest = "0" * 64
    with pytest.raises(RefinementReplayError, match="active transaction"):
        replay_aggregate_refinement(
            (
                {"event": "run_open", "state_digest": digest},
                {"event": "candidate_screening", "admitted_count": 1},
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-a",
                    "phase": "PREPARED",
                    "snapshot_digest": digest,
                    "candidate_keys": ["a"],
                },
                {"event": "candidate_screening", "admitted_count": 1},
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-b",
                    "phase": "PREPARED",
                    "snapshot_digest": digest,
                    "candidate_keys": ["b"],
                },
                {"event": "run_terminated", "reason": "runtime_contract_error"},
            )
        )


def test_aggregate_refinement_rejects_visible_state_on_screening_event() -> None:
    from txnopt_evidence.refinement import (
        RefinementReplayError,
        replay_aggregate_refinement,
    )

    digest = "0" * 64
    with pytest.raises(RefinementReplayError, match="non-transaction event"):
        replay_aggregate_refinement(
            (
                {"event": "run_open", "state_digest": digest},
                {
                    "event": "candidate_screening",
                    "admitted_count": 0,
                    "state_digest": "1" * 64,
                    "cache_generation": 99,
                },
                {"event": "round_terminated", "reason": "no_admissible_candidates"},
                {"event": "run_terminated", "reason": "no_admissible_candidates"},
            )
        )


def test_aggregate_refinement_rejects_round_termination_inside_a_transaction() -> None:
    from txnopt_evidence.refinement import (
        RefinementReplayError,
        replay_aggregate_refinement,
    )

    digest = "0" * 64
    with pytest.raises(RefinementReplayError, match="round termination"):
        replay_aggregate_refinement(
            (
                {"event": "run_open", "state_digest": digest},
                {"event": "candidate_screening", "admitted_count": 1},
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-a",
                    "phase": "PREPARED",
                    "snapshot_digest": digest,
                    "candidate_keys": ["a"],
                },
                {"event": "round_terminated", "reason": "invalid"},
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-a",
                    "phase": "ABORTED",
                    "snapshot_digest": digest,
                    "candidate_keys": ["a"],
                },
                {"event": "run_terminated", "reason": "runtime_contract_error"},
            )
        )


def test_aggregate_refinement_marks_unknown_cache_outcome_not_prefix_safe() -> None:
    from txnopt_evidence.refinement import replay_aggregate_refinement

    digest = "0" * 64
    receipt = replay_aggregate_refinement(
        (
            {"event": "run_open", "state_digest": digest},
            {"event": "candidate_screening", "admitted_count": 1},
            {
                "event": "candidate_transaction",
                "txn_id": "round-a",
                "phase": "PREPARED",
                "snapshot_digest": digest,
                "candidate_keys": ["a"],
            },
            {
                "event": "candidate_transaction",
                "txn_id": "round-a",
                "phase": "INTERRUPTED",
                "snapshot_digest": digest,
                "candidate_keys": ["a"],
                "cache_outcome": "unknown",
            },
            {"event": "run_terminated", "reason": "cache_outcome_unknown"},
        )
    )

    assert receipt.prefix_safety_proven is False


def test_aggregate_refinement_rejects_unknown_cache_outcome_on_abort() -> None:
    from txnopt_evidence.refinement import (
        RefinementReplayError,
        replay_aggregate_refinement,
    )

    digest = "0" * 64
    with pytest.raises(RefinementReplayError, match="only on an interrupted"):
        replay_aggregate_refinement(
            (
                {"event": "run_open", "state_digest": digest},
                {"event": "candidate_screening", "admitted_count": 1},
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-a",
                    "phase": "PREPARED",
                    "snapshot_digest": digest,
                    "candidate_keys": ["a"],
                },
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-a",
                    "phase": "ABORTED",
                    "snapshot_digest": digest,
                    "candidate_keys": ["a"],
                    "cache_outcome": "unknown",
                },
                {"event": "run_terminated", "reason": "runtime_contract_error"},
            )
        )


def test_aggregate_refinement_rejects_unknown_cache_outcome_on_commit() -> None:
    from txnopt_evidence.refinement import (
        RefinementReplayError,
        replay_aggregate_refinement,
    )

    digest = "0" * 64
    with pytest.raises(RefinementReplayError, match="only on an interrupted"):
        replay_aggregate_refinement(
            (
                {"event": "run_open", "state_digest": digest},
                {"event": "candidate_screening", "admitted_count": 1},
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-a",
                    "phase": "PREPARED",
                    "snapshot_digest": digest,
                    "candidate_keys": ["a"],
                },
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-a",
                    "phase": "RESERVED",
                    "snapshot_digest": digest,
                    "candidate_keys": ["a"],
                },
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-a",
                    "phase": "EVALUATING",
                    "snapshot_digest": digest,
                    "candidate_keys": ["a"],
                },
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-a",
                    "phase": "VALIDATED",
                    "snapshot_digest": digest,
                    "candidate_keys": ["a"],
                },
                {
                    "event": "candidate_transaction",
                    "txn_id": "round-a",
                    "phase": "COMMITTED",
                    "snapshot_digest": digest,
                    "candidate_keys": ["a"],
                    "state_digest": "1" * 64,
                    "cache_generation": 1,
                    "started_work": 1,
                    "cache_outcome": "unknown",
                },
                {"event": "run_terminated", "reason": "max_rounds"},
            )
        )
