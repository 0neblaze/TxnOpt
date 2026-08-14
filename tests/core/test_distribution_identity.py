from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_active_distribution_and_cli_identity_are_txnopt_only() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    assert project["project"]["name"] == "txnopt"
    assert project["project"]["version"] == "0.1.0a1"
    assert project["project"]["scripts"] == {"txnopt": "txnopt_evidence.cli:main"}
    assert project["tool"]["scikit-build"]["wheel"]["packages"] == [
        "src/txnopt",
        "src/txnopt_cases",
        "src/txnopt_evidence",
        "src/txnopt_legacy",
    ]


def test_active_cmake_builds_only_the_new_native_extension() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "project(txnopt_core LANGUAGES CXX)" in cmake
    assert "pybind11_add_module(\n  _native" in cmake
    assert "install(TARGETS _native DESTINATION txnopt)" in cmake
    assert "pybind11_add_module(\n  _core" not in cmake
    assert "_native_host_scheduler" not in cmake
    assert "DESTINATION evrptw" not in cmake


def test_new_active_python_does_not_import_the_frozen_namespace_or_stage_schema() -> None:
    package_roots = (
        ROOT / "src/txnopt",
        ROOT / "src/txnopt_cases",
        ROOT / "src/txnopt_evidence",
        ROOT / "src/txnopt_legacy",
    )
    findings: list[str] = []
    for package_root in package_roots:
        for path in package_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if any(alias.name.partition(".")[0] == "evrptw" for alias in node.names):
                        findings.append(str(path.relative_to(ROOT)))
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module is not None
                    and node.module.partition(".")[0] == "evrptw"
                ):
                    findings.append(str(path.relative_to(ROOT)))
            if "stage05.2" in source:
                findings.append(str(path.relative_to(ROOT)))
    assert findings == []
