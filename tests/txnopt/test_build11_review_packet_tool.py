from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from tools.txnopt_level1_campaign_common import sha256_file
from tools.verify_txnopt_build11_review_packet import (
    EXPECTED_REVIEW_TESTS,
    _junit_counts,
    _verify_bound_inputs,
    _verify_import_independence,
    verify_packet_receipt,
)


def _write_junit(path: Path, *, tests: int) -> None:
    path.write_text(
        f'<testsuites><testsuite tests="{tests}" failures="0" '
        'errors="0" skipped="0" /></testsuites>',
        encoding="utf-8",
    )


def test_bound_input_verifier_rejects_resigned_source_drift(tmp_path: Path) -> None:
    source = tmp_path / "bound.py"
    source.write_text("value = 1\n", encoding="utf-8")
    request = {
        "bound_inputs": [
            {"path": "bound.py", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        ]
    }
    with pytest.raises(ValueError, match="count differs"):
        _verify_bound_inputs(tmp_path, request)


def test_import_independence_rejects_a_reviewer_import(tmp_path: Path) -> None:
    paths = (
        "src/txnopt_evidence/runner.py",
        "src/txnopt_evidence/reviewer.py",
        "tools/run_txnopt_level1_campaign.py",
        "tools/review_txnopt_level1_campaign.py",
    )
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("from __future__ import annotations\n", encoding="utf-8")
    (tmp_path / paths[0]).write_text(
        "import txnopt_evidence.reviewer\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="import independence"):
        _verify_import_independence(tmp_path)


def test_junit_parser_rejects_a_partial_review_suite(tmp_path: Path) -> None:
    junit = tmp_path / "partial.xml"
    _write_junit(junit, tests=EXPECTED_REVIEW_TESTS - 1)
    with pytest.raises(ValueError, match="expected"):
        _junit_counts(junit)


def test_packet_receipt_cannot_claim_an_independent_decision(tmp_path: Path) -> None:
    junit = tmp_path / "pytest-junit.xml"
    stdout = tmp_path / "pytest.stdout.txt"
    stderr = tmp_path / "pytest.stderr.txt"
    _write_junit(junit, tests=EXPECTED_REVIEW_TESTS)
    stdout.write_text("pass\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    receipt = {
        "schema_version": "txnopt-build11-review-packet-verification-v1",
        "run_label": "txnopt_build11_review_packet_verification_attempt01",
        "status": "PACKET_VERIFIED_REVIEW_DECISION_REQUIRED",
        "execution": {
            "junit_path": str(junit),
            "junit_sha256": sha256_file(junit),
            "stdout_path": str(stdout),
            "stdout_sha256": sha256_file(stdout),
            "stderr_path": str(stderr),
            "stderr_sha256": sha256_file(stderr),
            "returncode": 0,
            "timed_out": False,
            "descendant_processes_remaining": [],
            "tests": EXPECTED_REVIEW_TESTS,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        },
        "machine_gates": {"bound_inputs": "PASS_16_OF_16"},
        "claim_boundary": {
            "machine_packet_verification": "PASS",
            "external_independent_review_completed": False,
            "review_decision": None,
            "precloud_gate_unblocked": False,
            "procurement_authorized": False,
            "formal_matrix_authorized": False,
            "level1_ready": False,
            "level2_entry_authorized": False,
        },
    }
    receipt_path = tmp_path / "receipt.json"
    data = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    receipt_path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    receipt_path.with_suffix(".json.sha256").write_text(
        f"{digest}  receipt.json\n", encoding="utf-8"
    )
    verify_packet_receipt(receipt_path)
    claim = cast(dict[str, Any], receipt["claim_boundary"])
    claim["external_independent_review_completed"] = True
    data = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    receipt_path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    receipt_path.with_suffix(".json.sha256").write_text(
        f"{digest}  receipt.json\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="overstates"):
        verify_packet_receipt(receipt_path)


def test_tracked_packet_verification_remains_machine_only() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "formal/reviews/txnopt_build11_review_packet_verification_attempt08.json"
    )
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    manifest = json.loads(path.read_bytes())

    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert manifest["status"] == "MACHINE_VERIFIED_EXTERNAL_DECISION_PENDING"
    assert manifest["review_request"]["bound_input_count"] == 16
    assert manifest["machine_gates"]["lifecycle_refinement_tests"] == (
        "PASS_46_OF_46"
    )
    assert manifest["claim_boundary"]["external_independent_review_completed"] is False
    assert manifest["claim_boundary"]["review_decision"] is None
    assert manifest["claim_boundary"]["precloud_gate_unblocked"] is False
    assert manifest["claim_boundary"]["procurement_authorized"] is False
    assert manifest["claim_boundary"]["level1_ready"] is False
