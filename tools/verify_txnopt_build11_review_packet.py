"""Machine-verify the Build11 review packet without issuing a review decision.

The command checks the immutable review request, its source bindings, the
sealed formal receipt, runner/reviewer import independence, the exact installed
Build11 wheel, and the request-bound lifecycle/refinement tests.  Its output is
only a packet-verification receipt; an external reviewer must still inspect the
claims and create a separately identified review artifact.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

from tools.txnopt_level1_campaign_common import (
    canonical_json_bytes,
    executable_path,
    load_campaign_plan,
    read_signed_object,
    require_clean_repository,
    run_isolated_process,
    sha256_file,
    verify_runtime_installation,
    verify_sidecar,
    write_signed_object,
)

REQUEST_SCHEMA = "txnopt-formal-independent-review-request-v1"
RECEIPT_SCHEMA = "txnopt-build11-review-packet-verification-v1"
EXPECTED_REQUEST_LABEL = "txnopt_evidence_lifecycle_review_request_attempt07"
EXPECTED_BOUND_INPUTS = 16
EXPECTED_REVIEW_TESTS = 46
REVIEW_TEST_PATHS = (
    "tests/txnopt/test_evidence_lifecycle.py",
    "tests/txnopt/test_evidence_pipeline.py",
    "tests/txnopt/test_formal_contract.py",
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository_file(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or not posix.parts:
        raise ValueError(f"review packet path is not repository-relative: {relative}")
    candidate = root.joinpath(*posix.parts).absolute()
    if candidate.is_symlink():
        raise ValueError(f"review packet input cannot be a symlink: {relative}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError(f"review packet input escapes the repository: {relative}")
    return resolved


def _load_request(root: Path, request_path: Path) -> tuple[dict[str, Any], str]:
    candidate = request_path.expanduser().absolute()
    if candidate.is_symlink():
        raise ValueError("review request cannot be a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("review request must be inside the repository")
    digest = verify_sidecar(resolved)
    request = read_signed_object(resolved, schema_version=REQUEST_SCHEMA)
    required_output = request.get("required_review_output")
    claim = request.get("claim_boundary")
    if (
        request.get("run_label") != EXPECTED_REQUEST_LABEL
        or request.get("status") != "READY_FOR_EXTERNAL_INDEPENDENT_REVIEW"
        or request.get("review_completed") is not False
        or request.get("reviewer_identity") is not None
        or request.get("review_decision") is not None
        or not isinstance(required_output, dict)
        or required_output.get("must_bind_this_request_sha256") is not True
        or required_output.get("must_bind_reviewer_revision_and_source_tree")
        is not True
        or not isinstance(claim, dict)
        or claim.get("independent_review_completed") is not False
        or claim.get("precloud_gate_unblocked") is not False
        or claim.get("procurement_authorized") is not False
        or claim.get("level1_ready") is not False
    ):
        raise ValueError("Build11 review request or claim boundary differs")
    return request, digest


def _verify_bound_inputs(root: Path, request: dict[str, Any]) -> list[dict[str, str]]:
    raw = request.get("bound_inputs")
    if not isinstance(raw, list) or len(raw) != EXPECTED_BOUND_INPUTS:
        raise ValueError("Build11 review request bound-input count differs")
    verified: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("review request bound input is malformed")
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("review request bound input identity is malformed")
        if relative in seen:
            raise ValueError("review request repeats a bound input")
        seen.add(relative)
        actual = sha256_file(_repository_file(root, relative))
        if actual != expected:
            raise ValueError(f"Build11 review input differs: {relative}")
        verified.append({"path": relative, "sha256": actual})
    return verified


def _verify_formal_receipt(root: Path) -> dict[str, Any]:
    path = _repository_file(root, "formal/model-check-receipt.json")
    receipt = json.loads(path.read_bytes())
    if not isinstance(receipt, dict) or receipt.get("result") != "PASS":
        raise ValueError("sealed formal model receipt is not PASS")
    inputs = receipt.get("input_sha256")
    if not isinstance(inputs, dict) or len(inputs) != 10:
        raise ValueError("sealed formal model input inventory differs")
    for relative, expected in inputs.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("formal model input identity is malformed")
        if sha256_file(_repository_file(root, relative)) != expected:
            raise ValueError(f"sealed formal model input differs: {relative}")
    model_check = receipt.get("model_check")
    if (
        receipt.get("proof_package_status") != "READY_FOR_INDEPENDENT_REVIEW"
        or not isinstance(model_check, dict)
        or not isinstance(model_check.get("primary"), dict)
        or model_check["primary"].get("distinct_states") != 22
        or not isinstance(model_check.get("pluscal"), dict)
        or model_check["pluscal"].get("distinct_states") != 8134
        or not isinstance(model_check.get("t3_witness"), dict)
        or model_check["t3_witness"].get("distinct_states") != 5
    ):
        raise ValueError("sealed formal model scope or claim differs")
    return {
        "path": "formal/model-check-receipt.json",
        "sha256": sha256_file(path),
        "input_count": len(inputs),
        "result": "PASS_SEALED_RECEIPT",
        "live_tlc_rerun": "NOT_PERFORMED_BY_PACKET_VERIFIER",
    }


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _verify_import_independence(root: Path) -> dict[str, str]:
    pairs = (
        (
            "src/txnopt_evidence/runner.py",
            "txnopt_evidence.reviewer",
        ),
        (
            "src/txnopt_evidence/reviewer.py",
            "txnopt_evidence.runner",
        ),
        (
            "tools/run_txnopt_level1_campaign.py",
            "tools.review_txnopt_level1_campaign",
        ),
        (
            "tools/review_txnopt_level1_campaign.py",
            "tools.run_txnopt_level1_campaign",
        ),
    )
    for relative, forbidden in pairs:
        imported = _imports(_repository_file(root, relative))
        if forbidden in imported or any(name.startswith(f"{forbidden}.") for name in imported):
            raise ValueError(f"producer/reviewer import independence differs: {relative}")
    return {
        "evidence_runner_reviewer": "PASS",
        "campaign_runner_reviewer": "PASS",
    }


def _junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = (root,) if root.tag == "testsuite" else tuple(root.findall("testsuite"))
    if not suites:
        raise ValueError("review packet JUnit contains no test suite")
    counts = {
        field: sum(int(suite.attrib.get(field, "0")) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }
    if counts["tests"] != EXPECTED_REVIEW_TESTS:
        raise ValueError(
            f"review packet collected {counts['tests']} tests; "
            f"expected {EXPECTED_REVIEW_TESTS}"
        )
    return counts


def _write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def run_packet_verification(
    *,
    root: Path,
    request_path: Path,
    plan_path: Path,
    python: Path,
    wheel: Path,
    output_dir: Path,
) -> tuple[Path, str]:
    root = root.resolve(strict=True)
    require_clean_repository(root)
    request, request_sha256 = _load_request(root, request_path)
    bound_inputs = _verify_bound_inputs(root, request)
    formal_receipt = _verify_formal_receipt(root)
    import_independence = _verify_import_independence(root)
    plan = load_campaign_plan(plan_path)
    producer_identity = request.get("producer_identity")
    if not isinstance(producer_identity, dict):
        raise ValueError("Build11 request producer identity is malformed")
    if (
        plan.build_manifest_sha256 != producer_identity.get("build_manifest_sha256")
        or plan.manifest_sha256
        != "be5a2bbe98bfa6092188ba849eb64f01de8232e10fbb91c8919d90d250c59e1c"
    ):
        raise ValueError("review packet plan does not bind the Build11 request")
    runtime_identity = verify_runtime_installation(plan, python=python, wheel=wheel)
    if (
        runtime_identity.get("source_revision") != producer_identity.get("source_revision")
        or runtime_identity.get("source_tree") != producer_identity.get("source_tree")
        or runtime_identity.get("wheel_sha256") != producer_identity.get("wheel_sha256")
        or runtime_identity.get("native_sha256") != producer_identity.get("native_sha256")
    ):
        raise ValueError("installed runtime differs from the Build11 review request")

    output = output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse review packet directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    junit_path = output / "pytest-junit.xml"
    command = [
        str(executable_path(python)),
        "-I",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--disable-warnings",
        "--maxfail=1",
        f"--junitxml={junit_path}",
        *REVIEW_TEST_PATHS,
    ]
    completed = run_isolated_process(command, cwd=root, timeout_seconds=900.0)
    stdout_path = output / "pytest.stdout.txt"
    stderr_path = output / "pytest.stderr.txt"
    _write_text_exclusive(stdout_path, completed.stdout)
    _write_text_exclusive(stderr_path, completed.stderr)
    if completed.returncode != 0 or completed.timed_out:
        raise RuntimeError("Build11 review packet tests failed; raw evidence was retained")
    if completed.descendant_processes_remaining:
        raise RuntimeError("Build11 review packet left live descendant processes")
    counts = _junit_counts(junit_path)
    if counts["failures"] or counts["errors"]:
        raise RuntimeError("Build11 review packet JUnit reports failures or errors")

    tool_relative = "tools/verify_txnopt_build11_review_packet.py"
    common_relative = "tools/txnopt_level1_campaign_common.py"
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "run_label": output.name,
        "status": "PACKET_VERIFIED_REVIEW_DECISION_REQUIRED",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "request_identity": {
            "path": str(request_path.resolve(strict=True)),
            "sha256": request_sha256,
            "run_label": request["run_label"],
            "bound_input_count": len(bound_inputs),
            "bound_inputs": bound_inputs,
        },
        "producer_identity": producer_identity,
        "orchestration_identity": {
            "revision": _git(root, "rev-parse", "HEAD"),
            "git_tree": _git(root, "rev-parse", "HEAD^{tree}"),
            "source_dirty": False,
            "tool_path": tool_relative,
            "tool_sha256": sha256_file(root / tool_relative),
            "common_path": common_relative,
            "common_sha256": sha256_file(root / common_relative),
        },
        "runtime_identity": runtime_identity,
        "formal_receipt": formal_receipt,
        "import_independence": import_independence,
        "execution": {
            "returncode": completed.returncode,
            "timed_out": completed.timed_out,
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
        "machine_gates": {
            "request_sidecar_and_claim_boundary": "PASS",
            "bound_inputs": "PASS_16_OF_16",
            "installed_build11_wheel": "PASS",
            "sealed_formal_receipt_inputs": "PASS_10_OF_10",
            "producer_reviewer_import_independence": "PASS",
            "lifecycle_refinement_tests": "PASS",
        },
        "reviewer_work_remaining": [
            "independently inspect every review obligation and claim boundary",
            "record critical and major findings separately",
            "bind an external reviewer revision, source tree, and this request SHA-256",
            "issue a new PASS_BUILD11_EVIDENCE_LIFECYCLE_REFINEMENT or NEEDS_WORK artifact",
        ],
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
    receipt_path = output / "receipt.json"
    return receipt_path, write_signed_object(receipt_path, receipt)


def verify_packet_receipt(receipt_path: Path) -> dict[str, Any]:
    receipt = read_signed_object(receipt_path, schema_version=RECEIPT_SCHEMA)
    execution = receipt.get("execution")
    gates = receipt.get("machine_gates")
    claim = receipt.get("claim_boundary")
    if not all(isinstance(value, dict) for value in (execution, gates, claim)):
        raise ValueError("review packet receipt sections are malformed")
    assert isinstance(execution, dict)
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
            raise ValueError(f"review packet artifact differs: {path_field}")
    counts = _junit_counts(Path(str(execution["junit_path"])))
    if any(counts[field] != execution.get(field) for field in counts):
        raise ValueError("review packet JUnit counts differ from receipt")
    if (
        receipt.get("status") != "PACKET_VERIFIED_REVIEW_DECISION_REQUIRED"
        or execution.get("returncode") != 0
        or execution.get("timed_out") is not False
        or execution.get("descendant_processes_remaining") != []
        or any(
            not isinstance(value, str) or not value.startswith("PASS")
            for value in gates.values()
        )
        or claim.get("machine_packet_verification") != "PASS"
        or claim.get("external_independent_review_completed") is not False
        or claim.get("review_decision") is not None
        or claim.get("precloud_gate_unblocked") is not False
        or claim.get("procurement_authorized") is not False
        or claim.get("formal_matrix_authorized") is not False
        or claim.get("level1_ready") is not False
        or claim.get("level2_entry_authorized") is not False
    ):
        raise ValueError("review packet receipt overstates its claim boundary")
    return cast(dict[str, Any], receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--root", type=Path, required=True)
    run_parser.add_argument("--request", type=Path, required=True)
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
        path, digest = run_packet_verification(
            root=args.root,
            request_path=args.request,
            plan_path=args.plan,
            python=args.python,
            wheel=args.wheel,
            output_dir=args.output_dir,
        )
        output = {"status": "PASS", "receipt": str(path), "sha256": digest}
    else:
        receipt = verify_packet_receipt(args.receipt)
        output = {
            "status": "PASS",
            "run_label": receipt["run_label"],
            "review_decision": receipt["claim_boundary"]["review_decision"],
        }
    print(canonical_json_bytes(output).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
