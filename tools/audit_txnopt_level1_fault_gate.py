"""Run and verify the local Level 1 fault gate against one frozen wheel.

The module is orchestration-only: the selected tests must be byte-identical to
the producer revision, while the implementation under test must come from the
installed wheel bound by the selected campaign plan.
"""

from __future__ import annotations

import argparse
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from tools.txnopt_level1_campaign_common import (
    canonical_json_bytes,
    load_campaign_plan,
    read_signed_object,
    require_clean_repository,
    run_isolated_process,
    sha256_bytes,
    sha256_file,
    verify_runtime_installation,
    write_signed_object,
)

SCHEMA_VERSION = "txnopt-level1-local-fault-gate-v1"
EXPECTED_BUILD_LABEL = "txnopt_level1_build_attempt11"
EXPECTED_PLAN_ATTEMPT = 23


@dataclass(frozen=True, slots=True)
class FaultScenario:
    category: str
    node_id: str
    expected_cases: int = 1


FAULT_SCENARIOS = (
    FaultScenario(
        "completion_order_equivalence",
        "tests/txnopt/test_parallel_runtime.py::"
        "test_all_execution_modes_refine_to_the_same_committed_trace",
    ),
    FaultScenario(
        "complete_budget_reservation",
        "tests/txnopt/test_native_round.py::"
        "test_native_round_reserves_the_complete_budget_before_evaluation",
    ),
    FaultScenario(
        "insufficient_budget",
        "tests/txnopt/test_transaction_properties.py::"
        "test_insufficient_budget_never_starts_or_caches_a_partial_batch",
    ),
    FaultScenario(
        "late_result_deadline",
        "tests/txnopt/test_transaction_properties.py::"
        "test_late_complete_batch_is_rolled_back",
    ),
    FaultScenario(
        "deadline_prefix",
        "tests/txnopt/test_serial_runtime.py::"
        "test_serial_runtime_discards_a_batch_that_crosses_deadline",
    ),
    FaultScenario(
        "worker_failure_prefix",
        "tests/txnopt/test_serial_runtime.py::"
        "test_worker_failure_returns_only_the_last_committed_prefix",
    ),
    FaultScenario(
        "validation_failure",
        "tests/txnopt/test_serial_runtime.py::"
        "test_validation_failure_rolls_back_the_incomplete_round",
    ),
    FaultScenario(
        "cache_write_failure",
        "tests/txnopt/test_serial_runtime.py::"
        "test_cache_write_failure_rolls_back_state_and_cache",
    ),
    FaultScenario(
        "stale_snapshot",
        "tests/txnopt/test_serial_runtime.py::"
        "test_stale_cache_snapshot_interrupts_before_evaluation",
    ),
    FaultScenario(
        "cache_conflict",
        "tests/txnopt/test_transaction_properties.py::"
        "test_cache_conflict_never_publishes_a_stale_transaction",
    ),
    FaultScenario(
        "duplicate_key",
        "tests/txnopt/test_transaction_properties.py::"
        "test_duplicate_candidate_keys_are_evaluated_once",
    ),
    FaultScenario(
        "contract_failure",
        "tests/txnopt/test_serial_runtime.py::"
        "test_contract_failure_after_reservation_rolls_back_audits_and_stays_fail_fast",
    ),
    FaultScenario(
        "rollback_failure",
        "tests/txnopt/test_serial_runtime.py::"
        "test_rollback_callback_failure_does_not_hide_the_primary_contract_error",
    ),
    FaultScenario(
        "invalid_decision",
        "tests/txnopt/test_serial_runtime.py::"
        "test_invalid_kernel_decision_emits_one_aborted_semantic_prefix_before_raising",
    ),
    FaultScenario(
        "unknown_commit_outcome",
        "tests/txnopt/test_serial_runtime.py::"
        "test_unverifiable_cache_outcome_is_neither_committed_nor_aborted",
    ),
    FaultScenario(
        "forged_commit_receipt",
        "tests/txnopt/test_serial_runtime.py::"
        "test_unpublished_cache_cannot_forge_a_valid_commit_receipt",
    ),
    FaultScenario(
        "closed_without_publication",
        "tests/txnopt/test_serial_runtime.py::"
        "test_closed_but_unpublished_cache_is_reported_as_outcome_unknown",
    ),
    FaultScenario(
        "ordered_t4_bound",
        "tests/txnopt/test_parallel_runtime.py::"
        "test_ordered_failure_binds_t4_to_the_full_atomic_transaction_window",
    ),
    FaultScenario(
        "parallel_contract_failure",
        "tests/txnopt/test_parallel_runtime.py::"
        "test_parallel_contract_failure_rolls_back_and_emits_one_audited_trace",
        expected_cases=2,
    ),
    FaultScenario(
        "post_boundary_t4_bound",
        "tests/txnopt/test_parallel_runtime.py::"
        "test_barrier_contract_failure_counts_work_still_running_at_the_boundary",
    ),
    FaultScenario(
        "property_t4_bound",
        "tests/txnopt/test_transaction_properties.py::"
        "test_t4_atomic_window_bound_holds_for_admitted_work",
    ),
    FaultScenario(
        "unknown_outcome_refinement",
        "tests/txnopt/test_evidence_pipeline.py::"
        "test_aggregate_refinement_marks_unknown_cache_outcome_not_prefix_safe",
    ),
    FaultScenario(
        "unknown_outcome_abort_rejected",
        "tests/txnopt/test_evidence_pipeline.py::"
        "test_aggregate_refinement_rejects_unknown_cache_outcome_on_abort",
    ),
    FaultScenario(
        "unknown_outcome_commit_rejected",
        "tests/txnopt/test_evidence_pipeline.py::"
        "test_aggregate_refinement_rejects_unknown_cache_outcome_on_commit",
    ),
)

EXPECTED_CASES = sum(scenario.expected_cases for scenario in FAULT_SCENARIOS)
REQUIRED_CATEGORIES = frozenset(scenario.category for scenario in FAULT_SCENARIOS)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _producer_test_sources(root: Path, *, revision: str) -> list[dict[str, str]]:
    paths = sorted({scenario.node_id.split("::", 1)[0] for scenario in FAULT_SCENARIOS})
    bindings: list[dict[str, str]] = []
    for relative in paths:
        current = (root / relative).read_bytes()
        producer = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if current != producer:
            raise RuntimeError(f"fault test differs from producer revision: {relative}")
        bindings.append({"path": relative, "sha256": sha256_bytes(current)})
    return bindings


def _junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = (root,) if root.tag == "testsuite" else tuple(root.findall("testsuite"))
    if not suites:
        raise ValueError("JUnit receipt contains no test suite")
    fields = ("tests", "failures", "errors", "skipped")
    counts = {
        field: sum(int(suite.attrib.get(field, "0")) for suite in suites)
        for field in fields
    }
    if counts["tests"] != EXPECTED_CASES:
        raise ValueError(
            f"fault gate collected {counts['tests']} tests; expected {EXPECTED_CASES}"
        )
    return counts


def _write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def run_fault_gate(
    *,
    root: Path,
    plan_path: Path,
    python: Path,
    wheel: Path,
    output_dir: Path,
) -> tuple[Path, str]:
    root = root.resolve(strict=True)
    require_clean_repository(root)
    plan = load_campaign_plan(plan_path)
    if plan.payload.get("attempt") != EXPECTED_PLAN_ATTEMPT:
        raise ValueError("fault gate requires Build11 formal plan Attempt23")
    build = read_signed_object(
        plan.build_manifest_path,
        schema_version="txnopt-level1-build-manifest-v1",
    )
    if build.get("run_label") != EXPECTED_BUILD_LABEL:
        raise ValueError("fault gate requires the Build11 producer")
    producer = build.get("producer")
    if not isinstance(producer, dict) or not isinstance(producer.get("revision"), str):
        raise ValueError("Build11 producer identity is malformed")

    output = output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse fault gate directory: {output}")
    output.mkdir(parents=True, exist_ok=False)

    runtime_identity = verify_runtime_installation(plan, python=python, wheel=wheel)
    test_sources = _producer_test_sources(root, revision=str(producer["revision"]))
    junit_path = output / "pytest-junit.xml"
    command = [
        str(python.expanduser().absolute().resolve(strict=True)),
        "-I",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--disable-warnings",
        "--maxfail=1",
        f"--junitxml={junit_path}",
        *(scenario.node_id for scenario in FAULT_SCENARIOS),
    ]
    completed = run_isolated_process(command, cwd=root, timeout_seconds=900.0)
    stdout_path = output / "pytest.stdout.txt"
    stderr_path = output / "pytest.stderr.txt"
    _write_text_exclusive(stdout_path, completed.stdout)
    _write_text_exclusive(stderr_path, completed.stderr)
    if completed.returncode != 0 or completed.timed_out:
        raise RuntimeError("fault gate failed; raw process evidence was retained")
    if completed.descendant_processes_remaining:
        raise RuntimeError("fault gate left live descendant processes")
    counts = _junit_counts(junit_path)
    if counts["failures"] or counts["errors"]:
        raise RuntimeError("fault gate JUnit reports failures or errors")

    tool_relative = "tools/audit_txnopt_level1_fault_gate.py"
    common_relative = "tools/txnopt_level1_campaign_common.py"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "run_label": output.name,
        "status": "PASS",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "producer_identity": {
            "build_manifest_path": str(plan.build_manifest_path),
            "build_manifest_sha256": plan.build_manifest_sha256,
            "source_revision": producer["revision"],
            "source_tree": producer.get("git_tree"),
            "wheel_sha256": runtime_identity["wheel_sha256"],
            "native_sha256": runtime_identity["native_sha256"],
        },
        "orchestration_identity": {
            "revision": _git(root, "rev-parse", "HEAD"),
            "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
            "source_dirty": False,
            "tool_path": tool_relative,
            "tool_sha256": sha256_file(root / tool_relative),
            "common_path": common_relative,
            "common_sha256": sha256_file(root / common_relative),
        },
        "plan_identity": {
            "path": str(plan.manifest_path),
            "sha256": plan.manifest_sha256,
            "attempt": plan.payload["attempt"],
            "config_tree_sha256": plan.payload["config_tree_sha256"],
            "formal_matrix_started": False,
        },
        "runtime_identity": runtime_identity,
        "fault_matrix": {
            "scenario_count": len(FAULT_SCENARIOS),
            "expected_test_case_count": EXPECTED_CASES,
            "categories": sorted(REQUIRED_CATEGORIES),
            "scenarios": [
                {
                    "category": scenario.category,
                    "node_id": scenario.node_id,
                    "expected_cases": scenario.expected_cases,
                }
                for scenario in FAULT_SCENARIOS
            ],
            "producer_test_sources": test_sources,
        },
        "execution": {
            "returncode": completed.returncode,
            "timed_out": completed.timed_out,
            "descendant_cleanup_performed": completed.descendant_cleanup_performed,
            "descendant_processes_remaining": list(
                completed.descendant_processes_remaining
            ),
            "junit_path": str(junit_path),
            "junit_sha256": sha256_file(junit_path),
            "stdout_path": str(stdout_path),
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_path": str(stderr_path),
            "stderr_sha256": sha256_file(stderr_path),
            **counts,
        },
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
            "independent_build11_review_completed": False,
            "cloud_purchase_authorized": False,
            "formal_matrix_started": False,
            "level1_ready": False,
        },
    }
    receipt_path = output / "receipt.json"
    return receipt_path, write_signed_object(receipt_path, receipt)


def verify_fault_gate(receipt_path: Path) -> dict[str, Any]:
    receipt = read_signed_object(receipt_path, schema_version=SCHEMA_VERSION)
    execution = receipt.get("execution")
    matrix = receipt.get("fault_matrix")
    gates = receipt.get("gate_results")
    claim = receipt.get("claim_boundary")
    if not all(isinstance(value, dict) for value in (execution, matrix, gates, claim)):
        raise ValueError("fault gate receipt sections are malformed")
    assert isinstance(execution, dict)
    assert isinstance(matrix, dict)
    assert isinstance(gates, dict)
    assert isinstance(claim, dict)
    for path_field, digest_field in (
        ("junit_path", "junit_sha256"),
        ("stdout_path", "stdout_sha256"),
        ("stderr_path", "stderr_sha256"),
    ):
        artifact = Path(str(execution.get(path_field))).absolute()
        if artifact.is_symlink() or sha256_file(artifact.resolve(strict=True)) != execution.get(
            digest_field
        ):
            raise ValueError(f"fault gate artifact differs: {path_field}")
    counts = _junit_counts(Path(str(execution["junit_path"])))
    if any(counts[field] != execution.get(field) for field in counts):
        raise ValueError("fault gate JUnit counts differ from the receipt")
    categories = matrix.get("categories")
    if not isinstance(categories, list) or set(categories) != REQUIRED_CATEGORIES:
        raise ValueError("fault gate category set differs")
    gate_values = (value for key, value in gates.items() if key != "fallback_count")
    if (
        receipt.get("status") != "PASS"
        or execution.get("returncode") != 0
        or execution.get("timed_out") is not False
        or execution.get("descendant_processes_remaining") != []
        or gates.get("fallback_count") != 0
        or any(value != "PASS" for value in gate_values)
        or claim.get("local_fault_gate") != "PASS"
        or claim.get("cloud_purchase_authorized") is not False
        or claim.get("formal_matrix_started") is not False
        or claim.get("level1_ready") is not False
    ):
        raise ValueError("fault gate receipt does not preserve the claim boundary")
    return cast(dict[str, Any], receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--python", type=Path, required=True)
    run_parser.add_argument("--wheel", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        path, digest = run_fault_gate(
            root=args.root,
            plan_path=args.plan,
            python=args.python,
            wheel=args.wheel,
            output_dir=args.output_dir,
        )
        output = {"status": "PASS", "receipt": str(path), "sha256": digest}
    else:
        receipt = verify_fault_gate(args.receipt)
        output = {
            "status": "PASS",
            "run_label": receipt["run_label"],
            "test_cases": receipt["execution"]["tests"],
        }
    print(canonical_json_bytes(output).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
