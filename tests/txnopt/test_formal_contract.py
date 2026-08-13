from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from txnopt._internal.candidate_txn import TxnPhase
from txnopt._internal.versions import (
    CONTRACT_VERSION,
    NATIVE_ROUND_VERSION,
    PHYSICAL_TRACE_VERSION,
    SEMANTIC_TRACE_VERSION,
)
from txnopt._internal.waste_bounds import WasteBoundViolation, audit_waste, bounded_waste

ROOT = Path(__file__).resolve().parents[2]


def test_formal_model_and_python_state_machine_name_the_same_phases() -> None:
    model = (ROOT / "formal/TxnOpt.tla").read_text(encoding="utf-8")
    for phase in TxnPhase:
        assert f'"{phase.value}"' in model


def test_formal_receipt_binds_checked_in_models_and_generated_translation() -> None:
    receipt = json.loads((ROOT / "formal/model-check-receipt.json").read_text(encoding="utf-8"))
    for relative_path, expected in receipt["input_sha256"].items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected
    translated = (ROOT / "formal/TxnOptOrderedPlusCal.tla").read_text(encoding="utf-8")
    assert "BEGIN TRANSLATION" in translated
    assert receipt["result"] == "PASS"
    assert receipt["scope"]["publication_granularity"] == "atomic_candidate_batch"
    assert receipt["model_check"]["primary"]["distinct_states"] == 22
    assert receipt["model_check"]["pluscal"]["distinct_states"] == 8_134
    assert receipt["proof_package_status"] == "READY_FOR_INDEPENDENT_REVIEW"


def test_first_independent_t3_t4_review_is_signed_and_fail_closed() -> None:
    path = ROOT / "formal/reviews/txnopt_t3_t4_review_attempt01.json"
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    review = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert review["status"] == "NEEDS_WORK"
    assert review["classifications"]["model_check"] == "TLC_SAFETY_PASS_ONLY"
    assert review["classifications"]["t4_runtime_refinement"] == "FAIL_NEEDS_WORK"
    assert review["counterexample"]["started_candidate_count"] > (
        review["counterexample"]["claimed_window_bound_units"]
    )
    assert review["claim_boundary"]["t3_t4_gate_passed"] is False


def test_second_proof_correction_is_signed_but_not_self_approved() -> None:
    path = ROOT / "formal/reviews/txnopt_t3_t4_proof_correction_attempt02.json"
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    correction = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert correction["status"] == "READY_FOR_INDEPENDENT_REVIEW"
    assert correction["claim_boundary"]["independent_review_completed"] is False
    assert correction["claim_boundary"]["t3_t4_gate_passed"] is False


def test_first_generation_protocol_identities_are_exact() -> None:
    assert (
        CONTRACT_VERSION,
        NATIVE_ROUND_VERSION,
        SEMANTIC_TRACE_VERSION,
        PHYSICAL_TRACE_VERSION,
    ) == (
        "txnopt-contract-v1",
        "txnopt-native-round-v1",
        "txnopt-semantic-trace-v1",
        "txnopt-physical-trace-v1",
    )


def test_t4_executable_bound_matches_declared_formula() -> None:
    result = bounded_waste(
        remaining_budget=100,
        uncommitted_window=3,
        max_requests_per_candidate=7,
        max_request_cost=2.5,
        post_boundary_capacity_units=4,
    )
    assert result.discarded_work_units == min(100, 3 * 7)
    assert result.discarded_cost == min(100, 3 * 7) * 2.5
    assert result.post_boundary_work_units == min(4, 3 * 7)
    assert result.post_boundary_cost == min(4, 3 * 7) * 2.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "remaining_budget": -1,
            "uncommitted_window": 1,
            "max_requests_per_candidate": 1,
            "max_request_cost": 1.0,
            "post_boundary_capacity_units": 1,
        },
        {
            "remaining_budget": 1,
            "uncommitted_window": 1,
            "max_requests_per_candidate": 1,
            "max_request_cost": float("inf"),
            "post_boundary_capacity_units": 1,
        },
        {
            "remaining_budget": 1,
            "uncommitted_window": 1,
            "max_requests_per_candidate": 1,
            "max_request_cost": 1.0,
            "post_boundary_capacity_units": -1,
        },
    ],
)
def test_t4_bound_rejects_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        bounded_waste(**kwargs)  # type: ignore[arg-type]


def test_t4_audit_fails_closed_when_observation_exceeds_atomic_window() -> None:
    with pytest.raises(WasteBoundViolation, match="discarded work"):
        audit_waste(
            remaining_budget_before=100,
            uncommitted_window=4,
            max_requests_per_candidate=1,
            post_boundary_capacity_units=4,
            observed_discarded_work_units=5,
            observed_post_boundary_work_units=0,
        )
