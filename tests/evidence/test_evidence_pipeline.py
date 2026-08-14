from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from txnopt import _native
from txnopt_evidence.cli import main
from txnopt_evidence.codec import (
    canonical_json_bytes,
    read_signed_json,
    sha256_bytes,
    sha256_file,
    write_signed_json,
)
from txnopt_evidence.identity import ExpectedEvidenceIdentity
from txnopt_evidence.lifecycle import (
    EvidenceLifecycle,
    EvidenceState,
    lifecycle_evidence_sha256,
)
from txnopt_evidence.reviewer import (
    _validate_event_stream,
    _validate_physical_trace,
    replay_legacy_manifest,
    replay_manifest,
    verify_failure_manifest,
    verify_legacy_failure_manifest,
    verify_legacy_raw_manifest,
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


def _expected_identity(config_path: Path, manifest_path: Path) -> ExpectedEvidenceIdentity:
    manifest = read_signed_json(manifest_path)
    return ExpectedEvidenceIdentity.for_unbound_test(
        config_path,
        producer_identity=manifest["producer_identity"],
    )


def _replace_signed_json(path: Path, payload: object) -> None:
    path.unlink()
    path.with_suffix(path.suffix + ".sha256").unlink()
    write_signed_json(path, payload)


def _resign_manifest(path: Path, manifest: dict[str, object]) -> None:
    lifecycle = (
        EvidenceLifecycle.start(
            str(manifest["run_label"]),
            evidence_sha256=str(manifest["input_config_sha256"]),
        )
        .advance(
            EvidenceState.RUNNING,
            evidence_sha256=lifecycle_evidence_sha256(manifest["producer_identity"]),
        )
        .advance(
            EvidenceState.SEALED,
            evidence_sha256=lifecycle_evidence_sha256(manifest["artifacts"]),
        )
    )
    manifest["lifecycle"] = lifecycle.to_payload()
    _replace_signed_json(path, manifest)


def _bind_current_native_build(
    tmp_path: Path,
    payload: dict[str, object],
) -> Path:
    attestation = dict(_native.BUILD_ATTESTATION)
    native_file = _native.__file__
    assert isinstance(native_file, str)
    build_path = tmp_path / "build-manifest.json"
    build_digest = write_signed_json(
        build_path,
        {
            "schema_version": "txnopt-level1-build-manifest-v1",
            "producer": {
                "revision": attestation["source_revision"],
                "git_tree": attestation["source_tree"],
                "source_manifest_sha256": attestation["source_manifest_sha256"],
                "tracked_file_count": attestation["tracked_file_count"],
                "source_dirty": False,
                "development_override": False,
            },
            "artifacts": {
                "wheel": {"sha256": "a" * 64},
                "native_extension": {
                    "sha256": sha256_file(Path(native_file).resolve(strict=True)),
                    "protocol": "txnopt-native-round-v1",
                },
            },
        },
    )
    payload["build_manifest"] = {
        "path": str(build_path),
        "sha256": build_digest,
    }
    return build_path


@pytest.mark.parametrize("factory", [_rcpsp_config, _evrptw_config])
def test_runner_writes_raw_only_and_reviewer_reconstructs_both_domains(
    tmp_path: Path,
    factory: object,
) -> None:
    config_path = tmp_path / "run.json"
    payload = factory(tmp_path)  # type: ignore[operator]
    _write_config(config_path, payload)
    raw = run_config_file(config_path)
    expected = _expected_identity(config_path, raw.manifest_path)

    manifest = verify_raw_manifest(raw.manifest_path, expected_identity=expected)
    assert manifest["schema_version"] == "txnopt-raw-artifact-v3"
    assert manifest["runner_decision"] is None
    assert manifest["fallback_count"] == 0
    assert manifest["producer_identity"]["binding_status"] == "UNBOUND_TEST_ONLY"
    sealed = EvidenceLifecycle.from_payload(manifest["lifecycle"])
    assert sealed.state is EvidenceState.SEALED
    review = replay_manifest(
        raw.manifest_path,
        output_dir=tmp_path / "review",
        expected_identity=expected,
    )
    assert review["schema_version"] == "txnopt-independent-review-v3"
    assert review["expected_identity_sha256"] == sha256_bytes(
        canonical_json_bytes(expected.to_payload(), pretty=True)
    )
    assert review["status"] == "PASS"
    assert review["prefix_safety"] == "PASS"
    assert review["case_replay"]["validator_status"] == "PASS"
    assert review["physical_event_count"] == 1
    assert review["aggregate_refinement_replay"] == "PASS"
    assert review["readiness_decision"] is None
    reviewed = EvidenceLifecycle.from_payload(review["lifecycle"])
    assert reviewed.state is EvidenceState.REVIEWED
    assert reviewed.events[:3] == sealed.events


def test_reviewer_detects_raw_tampering_before_replay(tmp_path: Path) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path))
    raw = run_config_file(config_path)
    expected = _expected_identity(config_path, raw.manifest_path)
    events = raw.manifest_path.parent / "events.jsonl"
    events.write_bytes(events.read_bytes() + b"{}\n")

    with pytest.raises(ValueError, match="differs"):
        verify_raw_manifest(raw.manifest_path, expected_identity=expected)


def test_reviewer_rejects_a_resigned_but_tampered_lifecycle(tmp_path: Path) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path))
    raw = run_config_file(config_path)
    expected = _expected_identity(config_path, raw.manifest_path)
    manifest = read_signed_json(raw.manifest_path)
    lifecycle = manifest["lifecycle"]
    assert isinstance(lifecycle, dict)
    events = lifecycle["events"]
    assert isinstance(events, list) and isinstance(events[1], dict)
    events[1]["evidence_sha256"] = "f" * 64
    raw.manifest_path.unlink()
    raw.manifest_path.with_suffix(".json.sha256").unlink()
    write_signed_json(raw.manifest_path, manifest)

    with pytest.raises(ValueError, match="lifecycle"):
        verify_raw_manifest(raw.manifest_path, expected_identity=expected)


def test_v3_public_verify_rejects_resigned_foreign_producer_identity(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path))
    raw = run_config_file(config_path)
    expected = _expected_identity(config_path, raw.manifest_path)
    manifest = read_signed_json(raw.manifest_path)
    forged = dict(manifest["producer_identity"])
    forged.update(
        {
            "build_manifest_sha256": "f" * 64,
            "wheel_sha256": "e" * 64,
            "installed_native_sha256": "d" * 64,
        }
    )
    manifest["producer_identity"] = forged
    _resign_manifest(raw.manifest_path, manifest)

    with pytest.raises(ValueError, match="producer identity"):
        verify_raw_manifest(raw.manifest_path, expected_identity=expected)


def test_v3_bound_identity_is_derived_from_pre_run_config_and_build(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "run.json"
    payload = _rcpsp_config(tmp_path)
    build_path = _bind_current_native_build(tmp_path, payload)
    _write_config(config_path, payload)
    expected = ExpectedEvidenceIdentity.from_plan_inputs(
        config_path,
        build_manifest_path=build_path,
    )

    raw = run_config_file(config_path)
    manifest = verify_raw_manifest(raw.manifest_path, expected_identity=expected)

    assert manifest["producer_identity"] == dict(expected.producer_identity)
    assert expected.producer_identity["binding_status"] == "BOUND_CLEAN_BUILD"


def test_expected_identity_rejects_detached_retained_config_digest() -> None:
    with pytest.raises(ValueError, match="retained config digest"):
        ExpectedEvidenceIdentity(
            run_label="txnopt_rcpsp_attempt01",
            input_config_sha256="0" * 64,
            config_artifact_sha256="1" * 64,
            domain="rcpsp",
            execution_mode="serial",
            expected_oracle="txnopt_cases.rcpsp.oracle.RCPSPOracle",
            producer_identity={
                "binding_status": "UNBOUND_TEST_ONLY",
                "installed_native_sha256": "2" * 64,
            },
        )


def test_v3_public_verify_rejects_manifest_label_detached_from_retained_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path))
    raw = run_config_file(config_path)
    expected = _expected_identity(config_path, raw.manifest_path)
    manifest = read_signed_json(raw.manifest_path)
    manifest["run_label"] = "txnopt_rcpsp_resigned_attempt99"
    _resign_manifest(raw.manifest_path, manifest)

    with pytest.raises(ValueError, match="run label"):
        verify_raw_manifest(raw.manifest_path, expected_identity=expected)


def test_v3_public_verify_rejects_retained_config_detached_from_input_digest(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path))
    raw = run_config_file(config_path)
    expected = _expected_identity(config_path, raw.manifest_path)
    manifest = read_signed_json(raw.manifest_path)
    retained_path = raw.manifest_path.parent / "config.json"
    retained = read_signed_json(retained_path)
    retained["run_label"] = "txnopt_rcpsp_resigned_attempt99"
    _replace_signed_json(retained_path, retained)
    config_entry = next(
        entry for entry in manifest["artifacts"] if entry["path"] == "config.json"
    )
    config_entry["sha256"] = retained_path.with_suffix(".json.sha256").read_text().split()[0]
    config_entry["bytes"] = retained_path.stat().st_size
    _resign_manifest(raw.manifest_path, manifest)

    with pytest.raises(ValueError, match="config"):
        verify_raw_manifest(raw.manifest_path, expected_identity=expected)


def test_v3_public_verify_rejects_extra_artifact_entry_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path))
    raw = run_config_file(config_path)
    expected = _expected_identity(config_path, raw.manifest_path)
    manifest = read_signed_json(raw.manifest_path)
    manifest["artifacts"][0]["forged"] = "self-resigned"
    _resign_manifest(raw.manifest_path, manifest)

    with pytest.raises(ValueError, match="artifact.*field"):
        verify_raw_manifest(raw.manifest_path, expected_identity=expected)


@pytest.mark.parametrize(
    "mutation",
    ["schema", "domain", "provenance", "physical_ref", "extra"],
)
@pytest.mark.parametrize("entrypoint", ["verify", "replay"])
def test_v3_public_entrypoints_reject_self_resigned_result_identity(
    tmp_path: Path,
    mutation: str,
    entrypoint: str,
) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path))
    raw = run_config_file(config_path)
    expected = _expected_identity(config_path, raw.manifest_path)
    manifest = read_signed_json(raw.manifest_path)
    result_path = raw.manifest_path.parent / "result.json"
    result = read_signed_json(result_path)
    if mutation == "schema":
        result["schema_version"] = "forged-result-v99"
    elif mutation == "domain":
        result["domain"] = "evrptw"
    elif mutation == "provenance":
        result["provenance"] = {**result["provenance"], "oracle": "forged.Oracle"}
    elif mutation == "physical_ref":
        result["physical_artifact_ref"] = "forged-physical-reference"
    else:
        result["forged"] = True
    _replace_signed_json(result_path, result)
    result_entry = next(
        entry for entry in manifest["artifacts"] if entry["path"] == "result.json"
    )
    result_entry["sha256"] = result_path.with_suffix(".json.sha256").read_text().split()[0]
    result_entry["bytes"] = result_path.stat().st_size
    _resign_manifest(raw.manifest_path, manifest)

    with pytest.raises(ValueError, match="result"):
        if entrypoint == "verify":
            verify_raw_manifest(raw.manifest_path, expected_identity=expected)
        else:
            replay_manifest(
                raw.manifest_path,
                output_dir=tmp_path / "review",
                expected_identity=expected,
            )


def test_v1_raw_artifact_remains_replayable_without_a_lifecycle(tmp_path: Path) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path))
    raw = run_config_file(config_path)
    manifest = read_signed_json(raw.manifest_path)
    manifest["schema_version"] = "txnopt-raw-artifact-v1"
    del manifest["lifecycle"]
    raw.manifest_path.unlink()
    raw.manifest_path.with_suffix(".json.sha256").unlink()
    write_signed_json(raw.manifest_path, manifest)

    assert (
        verify_legacy_raw_manifest(raw.manifest_path)["schema_version"]
        == "txnopt-raw-artifact-v1"
    )
    review = replay_legacy_manifest(raw.manifest_path, output_dir=tmp_path / "v1-review")
    assert review["schema_version"] == "txnopt-independent-review-v1"
    assert "lifecycle" not in review


def test_run_and_verify_cli_are_live_without_fallback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path, run_label="txnopt_rcpsp_attempt02"))
    assert main(["run", "--config", str(config_path)]) == 0
    run_output = json.loads(capsys.readouterr().out)
    assert run_output["fallback_count"] == 0
    identity_path = tmp_path / "expected-identity.json"
    raw_manifest_path = Path(run_output["manifest_path"])
    write_signed_json(
        identity_path,
        _expected_identity(config_path, raw_manifest_path).to_payload(),
    )

    assert main(
        [
            "verify",
            run_output["manifest_path"],
            "--expected-identity",
            str(identity_path),
        ]
    ) == 0
    verify_output = json.loads(capsys.readouterr().out)
    assert verify_output["status"] == "verified"
    assert verify_output["fallback_count"] == 0


def test_cli_replay_runs_in_a_fresh_process_and_writes_signed_review(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path))
    raw = run_config_file(config_path)
    identity_path = tmp_path / "expected-identity.json"
    write_signed_json(
        identity_path,
        _expected_identity(config_path, raw.manifest_path).to_payload(),
    )
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
            "--expected-identity",
            str(identity_path),
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


def test_runner_rejects_a_symlinked_config_before_writing_raw(tmp_path: Path) -> None:
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path))
    symlink = tmp_path / "linked-run.json"
    symlink.symlink_to(config_path)

    with pytest.raises(ValueError, match="symlink"):
        run_config_file(symlink)

    assert not (tmp_path / "raw").exists()


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

    expected = ExpectedEvidenceIdentity(
        run_label="txnopt_native_attempt01",
        input_config_sha256="0" * 64,
        config_artifact_sha256="0" * 64,
        domain="evrptw",
        execution_mode="barrier",
        expected_oracle="txnopt_cases.evrptw.native_oracle.NativeEVRPTWOracle",
        producer_identity={
            "binding_status": "BOUND_CLEAN_BUILD",
            "build_manifest_sha256": "0" * 64,
            "source_revision": "a" * 40,
            "source_tree": "b" * 40,
            "source_manifest_sha256": "1" * 64,
            "tracked_file_count": 1,
            "wheel_sha256": "2" * 64,
            "installed_native_sha256": "3" * 64,
            "native_protocol": "txnopt-native-round-v1",
        },
    )
    _validate_physical_trace(
        (run, native),
        semantic_exact_work_started=True,
        expected_identity=expected,
    )
    mismatched_mode = {**run, "execution_mode": "serial"}
    with pytest.raises(ValueError, match="execution mode differs"):
        _validate_physical_trace(
            (mismatched_mode, native),
            semantic_exact_work_started=True,
            expected_identity=expected,
        )
    with pytest.raises(ValueError, match="lacks its native round receipt"):
        _validate_physical_trace(
            (run,),
            semantic_exact_work_started=True,
            expected_identity=expected,
        )
    python_expected = ExpectedEvidenceIdentity(
        run_label="txnopt_python_attempt01",
        input_config_sha256="0" * 64,
        config_artifact_sha256="0" * 64,
        domain="evrptw",
        execution_mode="barrier",
        expected_oracle="txnopt_cases.evrptw.oracle.EVRPTWOracle",
        producer_identity={
            "binding_status": "UNBOUND_TEST_ONLY",
            "installed_native_sha256": "3" * 64,
        },
    )
    with pytest.raises(ValueError, match="non-native evidence"):
        _validate_physical_trace(
            (run, native),
            semantic_exact_work_started=True,
            expected_identity=python_expected,
        )
    native["source_revision"] = "c" * 40
    with pytest.raises(ValueError, match="source identity differs"):
        _validate_physical_trace(
            (run, native),
            semantic_exact_work_started=True,
            expected_identity=expected,
        )
    native["source_revision"] = "a" * 40
    native["prepared_cache_write_count"] = 3
    with pytest.raises(ValueError, match="prepared delta"):
        _validate_physical_trace((run, native), semantic_exact_work_started=True)
    native["prepared_cache_write_count"] = 4
    task_receipts = native["task_receipts"]
    assert isinstance(task_receipts, list) and isinstance(task_receipts[1], list)
    task_receipts[1][2] = 3
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
    verified = verify_failure_manifest(
        manifest_path,
        expected_identity=_expected_identity(config_path, manifest_path),
    )
    assert verified["schema_version"] == "txnopt-failure-artifact-v3"
    assert verified["runner_decision"] == "FAILED"
    assert EvidenceLifecycle.from_payload(verified["lifecycle"]).state is EvidenceState.SEALED
    failure = read_signed_json(manifest_path.parent / "failure.json")
    assert failure["schema_version"] == "txnopt-run-failure-v3"
    assert failure["fallback_count"] == 0

    failure["forged"] = True
    failure_path = manifest_path.parent / "failure.json"
    _replace_signed_json(failure_path, failure)
    manifest = read_signed_json(manifest_path)
    failure_entry = next(
        entry for entry in manifest["artifacts"] if entry["path"] == "failure.json"
    )
    failure_entry["sha256"] = failure_path.with_suffix(".json.sha256").read_text().split()[0]
    failure_entry["bytes"] = failure_path.stat().st_size
    _resign_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="failure receipt field set"):
        verify_failure_manifest(
            manifest_path,
            expected_identity=_expected_identity(config_path, manifest_path),
        )


@pytest.mark.parametrize(
    "physical_evidence",
    [
        "t4_without_receipt",
        "no_physical",
        "zero_work_receipt",
        "undercovered_multiple_streams",
    ],
)
def test_native_failure_bundle_requires_receipt_for_aborted_started_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physical_evidence: str,
) -> None:
    digest = "0" * 64

    def fail_after_native_work(
        _case: object,
        *,
        config: object,
        semantic_sink: object,
        physical_sink: object,
    ) -> dict[str, object]:
        del config
        semantic_stream = (
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
                    "started_work": 1,
                },
                {"event": "run_terminated", "reason": "runtime_contract_error"},
            )
        semantic_sink(semantic_stream)  # type: ignore[operator]
        if physical_evidence == "undercovered_multiple_streams":
            semantic_sink(semantic_stream)  # type: ignore[operator]
        if physical_evidence != "no_physical":
            observations: list[dict[str, object]] = [
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
                }
            ]
            if physical_evidence == "t4_without_receipt":
                observations.append(
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
                    }
                )
            else:
                attestation = _native.BUILD_ATTESTATION
                observed_work = (
                    1 if physical_evidence == "undercovered_multiple_streams" else 0
                )
                observations.append(
                    {
                        "event": "native_round_observation",
                        "trace": "txnopt-physical-trace-v1",
                        "protocol": "txnopt-native-round-v1",
                        "phase": "VALIDATED",
                        "phase_trace": [
                            "PREPARED",
                            "RESERVED",
                            "EVALUATING",
                            "VALIDATED",
                        ],
                        "context_pack_count": 1,
                        "round_call_count": 1,
                        "worker_count": 1,
                        "scheduled_worker_count": 1,
                        "execution_policy": "serial_configured",
                        "parallel_route_threshold": 2,
                        "started_work": observed_work,
                        "completed_work": observed_work,
                        "interrupted_work": 0,
                        "budget_limit": observed_work,
                        "budget_reserved_work": observed_work,
                        "budget_remaining_work": 0,
                        "prepared_cache_write_count": observed_work,
                        "prepared_cache_key_checksum": 0,
                        "semantic_event_count": 4,
                        "task_receipt_count": 0,
                        "task_receipts": [],
                        "fallback_count": 0,
                        "source_revision": attestation["source_revision"],
                        "source_tree": attestation["source_tree"],
                    }
                )
            physical_sink(  # type: ignore[operator]
                tuple(observations)
            )
        raise RuntimeError("simulated native worker failure")

    monkeypatch.setattr("txnopt_evidence.runner.execute_case", fail_after_native_work)
    payload = _evrptw_config(tmp_path)
    case = payload["case"]
    assert isinstance(case, dict)
    case["backend"] = "native"
    build_path = _bind_current_native_build(tmp_path, payload)
    config_path = tmp_path / "native-failure.json"
    _write_config(config_path, payload)
    expected = ExpectedEvidenceIdentity.from_plan_inputs(
        config_path,
        build_manifest_path=build_path,
    )

    with pytest.raises(RunExecutionError) as captured:
        run_config_file(config_path)

    with pytest.raises(ValueError, match="lacks its native round receipt"):
        verify_failure_manifest(
            captured.value.manifest_path,
            expected_identity=expected,
        )


def test_v2_failure_artifact_remains_explicitly_verifiable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_case(
        _case: object,
        *,
        config: object,
        semantic_sink: object,
        physical_sink: object,
    ) -> object:
        del config, semantic_sink, physical_sink
        raise RuntimeError("legacy failure fixture")

    monkeypatch.setattr("txnopt_evidence.runner.execute_case", fail_case)
    config_path = tmp_path / "run.json"
    _write_config(config_path, _rcpsp_config(tmp_path, run_label="txnopt_legacy_attempt01"))
    with pytest.raises(RunExecutionError) as captured:
        run_config_file(config_path)

    manifest_path = captured.value.manifest_path
    manifest = read_signed_json(manifest_path)
    failure_path = manifest_path.parent / "failure.json"
    failure = read_signed_json(failure_path)
    manifest["schema_version"] = "txnopt-failure-artifact-v2"
    failure["schema_version"] = "txnopt-run-failure-v2"
    _replace_signed_json(failure_path, failure)
    failure_entry = next(
        entry for entry in manifest["artifacts"] if entry["path"] == "failure.json"
    )
    failure_entry["sha256"] = failure_path.with_suffix(".json.sha256").read_text().split()[0]
    failure_entry["bytes"] = failure_path.stat().st_size
    _resign_manifest(manifest_path, manifest)

    verified = verify_legacy_failure_manifest(manifest_path)
    assert verified["schema_version"] == "txnopt-failure-artifact-v2"


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

    manifest = verify_failure_manifest(
        caught.value.manifest_path,
        expected_identity=_expected_identity(config_path, caught.value.manifest_path),
    )
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
