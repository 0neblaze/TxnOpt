from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from tools.audit_txnopt_level1_static_gate import (
    EXPECTED_GATE_RESULTS,
    EXPECTED_TEST_CASES,
    _active_naming_findings,
    _core_import_cycles,
    _dependency_findings,
    _distribution_identity,
    _junit_counts,
    verify_static_gate,
)
from tools.txnopt_level1_campaign_common import sha256_file


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_junit(path: Path, *, tests: int) -> None:
    path.write_text(
        f'<testsuites><testsuite tests="{tests}" failures="0" '
        'errors="0" skipped="0" /></testsuites>',
        encoding="utf-8",
    )


def test_core_cycle_audit_rejects_a_two_module_cycle(tmp_path: Path) -> None:
    _write(tmp_path / "src/txnopt/a.py", "import txnopt.b\n")
    _write(tmp_path / "src/txnopt/b.py", "import txnopt.a\n")
    assert _core_import_cycles(tmp_path) == [["txnopt.a", "txnopt.b"]]


def test_dependency_audit_rejects_a_reverse_core_import(tmp_path: Path) -> None:
    _write(tmp_path / "src/txnopt/core.py", "import txnopt_evidence\n")
    _write(tmp_path / "src/txnopt_cases/__init__.py", "")
    _write(tmp_path / "src/txnopt_legacy/__init__.py", "")
    _write(tmp_path / "src/txnopt_evidence/runner.py", "")
    _write(tmp_path / "src/txnopt_evidence/reviewer.py", "")
    _write(tmp_path / "tools/run_txnopt_level1_campaign.py", "")
    _write(tmp_path / "tools/review_txnopt_level1_campaign.py", "")
    assert _dependency_findings(tmp_path) == [
        {
            "path": "src/txnopt/core.py",
            "forbidden_import_root": "txnopt_evidence",
        }
    ]


def test_current_active_source_passes_distribution_dependency_and_naming_audit() -> None:
    root = Path(__file__).resolve().parents[2]
    identity = _distribution_identity(root)
    assert identity["console_entrypoint_count"] == 1
    assert identity["root_exports"] == [
        "TxnRuntime",
        "SearchKernel",
        "Oracle",
        "RunConfig",
        "RunResult",
    ]
    assert _core_import_cycles(root) == []
    assert _dependency_findings(root) == []
    assert _active_naming_findings(root) == []


def test_junit_parser_rejects_a_partial_static_suite(tmp_path: Path) -> None:
    junit = tmp_path / "partial.xml"
    _write_junit(junit, tests=EXPECTED_TEST_CASES - 1)
    with pytest.raises(ValueError, match="expected"):
        _junit_counts(junit)


def test_static_receipt_cannot_overstate_level1_readiness(tmp_path: Path) -> None:
    junit = tmp_path / "pytest-junit.xml"
    stdout = tmp_path / "pytest.stdout.txt"
    stderr = tmp_path / "pytest.stderr.txt"
    _write_junit(junit, tests=EXPECTED_TEST_CASES)
    stdout.write_text("pass\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    receipt: dict[str, Any] = {
        "schema_version": "txnopt-level1-static-gate-v1",
        "run_label": "txnopt-level1-static-gate-test",
        "status": "PASS_NOT_LEVEL1_READY",
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
            "tests": EXPECTED_TEST_CASES,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        },
        "source_audit": {
            "core_import_cycles": [],
            "dependency_findings": [],
            "active_naming_findings": [],
        },
        "gate_results": EXPECTED_GATE_RESULTS,
        "claim_boundary": {
            "local_static_gate": "PASS",
            "external_independent_review_completed": False,
            "precloud_gate_unblocked": False,
            "procurement_authorized": False,
            "formal_matrix_authorized": False,
            "formal_matrix_started": False,
            "holdout_opened": False,
            "level1_ready": False,
            "level2_entry_authorized": False,
        },
    }
    receipt_path = tmp_path / "receipt.json"

    def sign() -> None:
        data = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
        receipt_path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        receipt_path.with_suffix(".json.sha256").write_text(
            f"{digest}  receipt.json\n", encoding="utf-8"
        )

    sign()
    verify_static_gate(receipt_path)
    claim = cast(dict[str, Any], receipt["claim_boundary"])
    claim["level1_ready"] = True
    sign()
    with pytest.raises(ValueError, match="overstates"):
        verify_static_gate(receipt_path)
