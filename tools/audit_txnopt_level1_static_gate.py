"""Audit the non-cloud Level 1 identity, import, entrypoint, and naming gates.

The implementation under test remains the immutable Build11 wheel and source
tree.  This orchestration-only gate proves that later evidence tooling did not
change that producer, then checks the active package/build/CLI surface and the
protected Stage-era evidence boundary.  It never issues an independent review,
authorizes procurement, or starts the formal matrix.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import tomllib
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from tools.txnopt_level1_campaign_common import (
    canonical_json_bytes,
    executable_path,
    load_campaign_plan,
    read_signed_object,
    require_clean_repository,
    run_isolated_process,
    sha256_bytes,
    sha256_file,
    verify_runtime_installation,
    write_signed_object,
)

SCHEMA_VERSION = "txnopt-level1-static-gate-v1"
EXPECTED_BUILD_LABEL = "txnopt_level1_build_attempt11"
EXPECTED_PLAN_ATTEMPT = 23
EXPECTED_TEST_CASES = 24
ACTIVE_PACKAGES = ("txnopt", "txnopt_cases", "txnopt_evidence", "txnopt_legacy")
ACTIVE_IMPLEMENTATION_PATHS = (
    "pyproject.toml",
    "CMakeLists.txt",
    "src/txnopt",
    "src/txnopt_cases",
    "src/txnopt_evidence",
    "src/txnopt_legacy",
    "cpp/txnopt_core",
    "cpp/txnopt_cases",
)
PROTECTED_HISTORY_PATHS = (
    "experiments/manifests",
    "experiments/migrations",
    "experiments/registries",
    "experiments/summaries",
    "artifacts/index.json",
    "docs/provenance",
    "docs/stage052_change_log.md",
    "docs/stage052_performance_benchmark_workflow.md",
)
STATIC_TEST_PATHS = (
    "tests/txnopt/test_contracts.py",
    "tests/txnopt/test_dependency_direction.py",
    "tests/txnopt/test_distribution_identity.py",
    "tests/txnopt/test_cli.py",
)
EXPECTED_GATE_RESULTS: dict[str, object] = {
    "entrypoint_coverage": 1,
    "core_import_cycles": 0,
    "txnopt_reverse_dependencies": 0,
    "producer_reviewer_mutual_imports": 0,
    "active_evrptw_imports": 0,
    "active_stage05_2_schemas": 0,
    "level1_full_native_fast_paths": 0,
    "historical_path_and_byte_mutations": 0,
    "root_public_export_count": 5,
    "installed_wheel_identity": "PASS",
    "producer_bound_static_tests": "PASS_24_OF_24",
}


def _git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def _git_text(root: Path, *arguments: str) -> str:
    return _git(root, *arguments).stdout.strip()


def _python_files(root: Path, package: str) -> tuple[Path, ...]:
    return tuple(sorted((root / "src" / package).rglob("*.py")))


def _module_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root / "src").with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imported_modules(path: Path, *, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.level == 0:
                imported.add(node.module)
            else:
                package_parts = module.split(".")[:-node.level]
                imported.add(".".join((*package_parts, node.module)))
    return imported


def _import_roots(path: Path, *, module: str) -> set[str]:
    return {name.partition(".")[0] for name in _imported_modules(path, module=module)}


def _strong_components(edges: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indexes[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in edges[node]:
            if target not in indexes:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indexes[target])
        if lowlinks[node] != indexes[node]:
            return
        component: list[str] = []
        while True:
            target = stack.pop()
            on_stack.remove(target)
            component.append(target)
            if target == node:
                break
        components.append(sorted(component))

    for node in sorted(edges):
        if node not in indexes:
            visit(node)
    return components


def _core_import_cycles(root: Path) -> list[list[str]]:
    modules = {
        _module_name(root, path): path for path in _python_files(root, "txnopt")
    }
    edges: dict[str, set[str]] = {module: set() for module in modules}
    for module, path in modules.items():
        for imported in _imported_modules(path, module=module):
            candidates = sorted(
                (
                    target
                    for target in modules
                    if imported == target or imported.startswith(f"{target}.")
                ),
                key=len,
                reverse=True,
            )
            if candidates and candidates[0] != module:
                edges[module].add(candidates[0])
    return [component for component in _strong_components(edges) if len(component) > 1]


def _dependency_findings(root: Path) -> list[dict[str, str]]:
    forbidden = {
        "txnopt": {"evrptw", "txnopt_cases", "txnopt_evidence", "txnopt_legacy"},
        "txnopt_cases": {"evrptw", "txnopt_evidence", "txnopt_legacy"},
        "txnopt_legacy": {"evrptw", "txnopt", "txnopt_cases", "txnopt_evidence"},
    }
    findings: list[dict[str, str]] = []
    for package, roots in forbidden.items():
        for path in _python_files(root, package):
            module = _module_name(root, path)
            for imported in sorted(_import_roots(path, module=module) & roots):
                findings.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "forbidden_import_root": imported,
                    }
                )
    independence_pairs = (
        ("src/txnopt_evidence/runner.py", "txnopt_evidence.reviewer"),
        ("src/txnopt_evidence/reviewer.py", "txnopt_evidence.runner"),
        ("tools/run_txnopt_level1_campaign.py", "tools.review_txnopt_level1_campaign"),
        ("tools/review_txnopt_level1_campaign.py", "tools.run_txnopt_level1_campaign"),
    )
    for relative, forbidden_module in independence_pairs:
        path = root / relative
        imported_modules = _imported_modules(
            path, module=relative.removesuffix(".py").replace("/", ".")
        )
        if forbidden_module in imported_modules:
            findings.append(
                {"path": relative, "forbidden_import_root": forbidden_module}
            )
    return findings


def _literal_all(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                return cast(list[str], value)
    raise ValueError("txnopt root __all__ is absent or non-literal")


def _distribution_identity(root: Path) -> dict[str, Any]:
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    metadata = project.get("project")
    build = project.get("tool", {}).get("scikit-build", {})
    expected_packages = [f"src/{package}" for package in ACTIVE_PACKAGES]
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    root_exports = _literal_all(root / "src/txnopt/__init__.py")
    if (
        not isinstance(metadata, dict)
        or metadata.get("name") != "txnopt"
        or metadata.get("version") != "0.1.0a1"
        or metadata.get("scripts") != {"txnopt": "txnopt_evidence.cli:main"}
        or build.get("wheel", {}).get("packages") != expected_packages
        or "project(txnopt_core LANGUAGES CXX)" not in cmake
        or "pybind11_add_module(\n  _native" not in cmake
        or "install(TARGETS _native DESTINATION txnopt)" not in cmake
        or "pybind11_add_module(\n  _core" in cmake
        or "_native_host_scheduler" in cmake
        or "DESTINATION evrptw" in cmake
        or root_exports
        != ["TxnRuntime", "SearchKernel", "Oracle", "RunConfig", "RunResult"]
    ):
        raise ValueError("active distribution, build, CLI, or root API identity differs")
    return {
        "distribution": "txnopt",
        "version": "0.1.0a1",
        "console_entrypoint_count": 1,
        "console_entrypoint": "txnopt=txnopt_evidence.cli:main",
        "cmake_project": "txnopt_core",
        "native_module": "txnopt._native",
        "root_exports": root_exports,
    }


def _active_naming_findings(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    import_pattern = re.compile(r"(?:from|import)\s+evrptw(?:\.|\s|$)")
    code_roots = (
        *(root / "src" / package for package in ACTIVE_PACKAGES),
        root / "cpp/txnopt_core",
        root / "cpp/txnopt_cases",
    )
    for code_root in code_roots:
        for path in sorted(code_root.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".pyi", ".cpp", ".hpp"}:
                continue
            source = path.read_text(encoding="utf-8")
            if import_pattern.search(source):
                findings.append(
                    {"path": path.relative_to(root).as_posix(), "finding": "import_evrptw"}
                )
            if "stage05.2" in source:
                findings.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "finding": "stage05.2_schema",
                    }
                )
            if "full_native_alns" in source:
                findings.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "finding": "level1_full_native_fast_path",
                    }
                )
    return findings


def _verify_producer_implementation(root: Path, *, revision: str) -> dict[str, Any]:
    diff = _git(
        root,
        "diff",
        "--name-only",
        revision,
        "HEAD",
        "--",
        *ACTIVE_IMPLEMENTATION_PATHS,
    ).stdout.splitlines()
    if diff:
        raise ValueError(f"active implementation differs from Build11: {diff}")
    bindings: list[dict[str, str]] = []
    for relative in STATIC_TEST_PATHS:
        current = (root / relative).read_bytes()
        producer = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if current != producer:
            raise ValueError(f"static gate test differs from Build11: {relative}")
        bindings.append({"path": relative, "sha256": sha256_bytes(current)})
    return {
        "implementation_path_count": len(ACTIVE_IMPLEMENTATION_PATHS),
        "implementation_diff_count": 0,
        "test_sources": bindings,
    }


def _verify_protected_history(root: Path, build: dict[str, Any]) -> dict[str, Any]:
    validation = build.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("Build11 validation inventory is malformed")
    protected = validation.get("historical_protected_paths")
    if not isinstance(protected, dict):
        raise ValueError("Build11 protected-history receipt is malformed")
    base = protected.get("comparison_base_revision")
    if not isinstance(base, str) or len(base) != 40:
        raise ValueError("Build11 protected-history base is malformed")
    changed = _git(
        root,
        "diff",
        "--name-status",
        "--find-renames=100%",
        base,
        "HEAD",
        "--",
        *PROTECTED_HISTORY_PATHS,
    ).stdout.splitlines()
    if changed:
        raise ValueError(f"protected Stage-era paths changed: {changed}")
    return {
        "comparison_base_revision": base,
        "protected_path_count": len(PROTECTED_HISTORY_PATHS),
        "changed_path_count": 0,
    }


def _junit_counts(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = (root,) if root.tag == "testsuite" else tuple(root.findall("testsuite"))
    if not suites:
        raise ValueError("static gate JUnit contains no test suite")
    counts = {
        field: sum(int(suite.attrib.get(field, "0")) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }
    if counts["tests"] != EXPECTED_TEST_CASES:
        raise ValueError(
            f"static gate collected {counts['tests']} tests; expected {EXPECTED_TEST_CASES}"
        )
    return counts


def _write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def run_static_gate(
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
        raise ValueError("static gate requires Build11 formal plan Attempt23")
    build = read_signed_object(
        plan.build_manifest_path,
        schema_version="txnopt-level1-build-manifest-v1",
    )
    producer = build.get("producer")
    if (
        build.get("run_label") != EXPECTED_BUILD_LABEL
        or not isinstance(producer, dict)
        or not isinstance(producer.get("revision"), str)
    ):
        raise ValueError("static gate requires the Build11 producer")

    output = output_dir.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to reuse static gate directory: {output}")

    distribution = _distribution_identity(root)
    cycles = _core_import_cycles(root)
    dependencies = _dependency_findings(root)
    naming = _active_naming_findings(root)
    if cycles or dependencies or naming:
        raise ValueError(
            "Level 1 static source gate failed: "
            f"cycles={cycles}, dependencies={dependencies}, naming={naming}"
        )
    producer_binding = _verify_producer_implementation(
        root, revision=str(producer["revision"])
    )
    protected_history = _verify_protected_history(root, build)
    runtime_identity = verify_runtime_installation(plan, python=python, wheel=wheel)

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
        *STATIC_TEST_PATHS,
    ]
    completed = run_isolated_process(command, cwd=root, timeout_seconds=600.0)
    stdout_path = output / "pytest.stdout.txt"
    stderr_path = output / "pytest.stderr.txt"
    _write_text_exclusive(stdout_path, completed.stdout)
    _write_text_exclusive(stderr_path, completed.stderr)
    if completed.returncode != 0 or completed.timed_out:
        raise RuntimeError("Level 1 static gate tests failed; raw evidence retained")
    if completed.descendant_processes_remaining:
        raise RuntimeError("Level 1 static gate left live descendant processes")
    counts = _junit_counts(junit_path)
    if counts["failures"] or counts["errors"]:
        raise RuntimeError("Level 1 static gate JUnit reports failures or errors")

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "run_label": output.name,
        "status": "PASS_NOT_LEVEL1_READY",
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
            "revision": _git_text(root, "rev-parse", "HEAD"),
            "git_tree": _git_text(root, "rev-parse", "HEAD^{tree}"),
            "source_dirty": False,
            "tool_path": "tools/audit_txnopt_level1_static_gate.py",
            "tool_sha256": sha256_file(root / "tools/audit_txnopt_level1_static_gate.py"),
        },
        "plan_identity": {
            "path": str(plan.manifest_path),
            "sha256": plan.manifest_sha256,
            "attempt": plan.payload["attempt"],
            "config_tree_sha256": plan.payload["config_tree_sha256"],
            "formal_matrix_started": False,
        },
        "runtime_identity": runtime_identity,
        "distribution_identity": distribution,
        "producer_binding": producer_binding,
        "protected_history": protected_history,
        "source_audit": {
            "core_import_cycles": cycles,
            "dependency_findings": dependencies,
            "active_naming_findings": naming,
        },
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
    receipt_path = output / "receipt.json"
    return receipt_path, write_signed_object(receipt_path, receipt)


def verify_static_gate(receipt_path: Path) -> dict[str, Any]:
    receipt = read_signed_object(receipt_path, schema_version=SCHEMA_VERSION)
    execution = receipt.get("execution")
    gates = receipt.get("gate_results")
    source = receipt.get("source_audit")
    claim = receipt.get("claim_boundary")
    if not all(isinstance(value, dict) for value in (execution, gates, source, claim)):
        raise ValueError("static gate receipt sections are malformed")
    assert isinstance(execution, dict)
    assert isinstance(gates, dict)
    assert isinstance(source, dict)
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
            raise ValueError(f"static gate artifact differs: {path_field}")
    counts = _junit_counts(Path(str(execution["junit_path"])))
    if any(counts[field] != execution.get(field) for field in counts):
        raise ValueError("static gate JUnit counts differ from receipt")
    if (
        receipt.get("status") != "PASS_NOT_LEVEL1_READY"
        or execution.get("returncode") != 0
        or execution.get("timed_out") is not False
        or execution.get("descendant_processes_remaining") != []
        or source.get("core_import_cycles") != []
        or source.get("dependency_findings") != []
        or source.get("active_naming_findings") != []
        or gates != EXPECTED_GATE_RESULTS
        or claim.get("local_static_gate") != "PASS"
        or claim.get("external_independent_review_completed") is not False
        or claim.get("precloud_gate_unblocked") is not False
        or claim.get("procurement_authorized") is not False
        or claim.get("formal_matrix_authorized") is not False
        or claim.get("formal_matrix_started") is not False
        or claim.get("holdout_opened") is not False
        or claim.get("level1_ready") is not False
        or claim.get("level2_entry_authorized") is not False
    ):
        raise ValueError("static gate receipt overstates or fails its boundary")
    return cast(dict[str, Any], receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--python", type=Path, required=True)
    run.add_argument("--wheel", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        path, digest = run_static_gate(
            root=args.root,
            plan_path=args.plan,
            python=args.python,
            wheel=args.wheel,
            output_dir=args.output_dir,
        )
        output = {"status": "PASS", "receipt": str(path), "sha256": digest}
    else:
        receipt = verify_static_gate(args.receipt)
        output = {
            "status": "PASS",
            "run_label": receipt["run_label"],
            "level1_ready": receipt["claim_boundary"]["level1_ready"],
        }
    print(canonical_json_bytes(output).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
