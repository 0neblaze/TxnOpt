from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.audit_txnopt_level1_fault_gate import (
    EXPECTED_CASES,
    REQUIRED_CATEGORIES,
    _junit_counts,
    verify_fault_gate,
)
from tools.txnopt_level1_campaign_common import sha256_file


def _write_junit(path: Path, *, tests: int, failures: int = 0) -> None:
    path.write_text(
        f'<testsuites><testsuite tests="{tests}" failures="{failures}" '
        'errors="0" skipped="0" /></testsuites>',
        encoding="utf-8",
    )


def test_fault_matrix_has_one_unique_category_per_declared_scenario() -> None:
    assert len(REQUIRED_CATEGORIES) == 24
    assert EXPECTED_CASES == 25


def test_junit_parser_rejects_a_partial_fault_matrix(tmp_path: Path) -> None:
    junit = tmp_path / "partial.xml"
    _write_junit(junit, tests=EXPECTED_CASES - 1)
    with pytest.raises(ValueError, match="expected"):
        _junit_counts(junit)


def test_verifier_rejects_a_tampered_fault_artifact(tmp_path: Path) -> None:
    junit = tmp_path / "pytest-junit.xml"
    stdout = tmp_path / "pytest.stdout.txt"
    stderr = tmp_path / "pytest.stderr.txt"
    _write_junit(junit, tests=EXPECTED_CASES)
    stdout.write_text("pass\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    receipt = {
        "schema_version": "txnopt-level1-local-fault-gate-v1",
        "run_label": "txnopt_level1_fault_gate_attempt01",
        "status": "PASS",
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
            "tests": EXPECTED_CASES,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        },
        "fault_matrix": {"categories": sorted(REQUIRED_CATEGORIES)},
        "gate_results": {
            "completion_order_equivalence": "PASS",
            "deadline_and_worker_failure_prefix_safety": "PASS",
            "budget_cache_and_snapshot_atomicity": "PASS",
            "late_result_rollback": "PASS",
            "unknown_commit_outcome_fail_closed": "PASS",
            "t4_waste_bound": "PASS",
            "fallback_count": 0,
        },
        "claim_boundary": {
            "local_fault_gate": "PASS",
            "cloud_purchase_authorized": False,
            "formal_matrix_started": False,
            "level1_ready": False,
        },
    }
    receipt_path = tmp_path / "receipt.json"
    data = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    receipt_path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    receipt_path.with_suffix(".json.sha256").write_text(
        f"{digest}  receipt.json\n", encoding="utf-8"
    )
    verify_fault_gate(receipt_path)
    stdout.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact differs"):
        verify_fault_gate(receipt_path)
