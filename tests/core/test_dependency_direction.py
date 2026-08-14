from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.partition(".")[0])
    return roots


def _python_files(package: str) -> list[Path]:
    return sorted((SRC / package).rglob("*.py"))


def test_txnopt_contract_boundary_has_no_case_evidence_or_legacy_import() -> None:
    forbidden = {"evrptw", "txnopt_cases", "txnopt_evidence", "txnopt_legacy"}
    observations = {
        path.relative_to(ROOT).as_posix(): sorted(_import_roots(path) & forbidden)
        for path in _python_files("txnopt")
    }
    assert all(not imports for imports in observations.values()), observations


def test_case_packages_depend_only_inward() -> None:
    forbidden = {"evrptw", "txnopt_evidence", "txnopt_legacy"}
    observations = {
        path.relative_to(ROOT).as_posix(): sorted(_import_roots(path) & forbidden)
        for path in _python_files("txnopt_cases")
    }
    assert all(not imports for imports in observations.values()), observations


def test_legacy_reader_does_not_activate_frozen_solver() -> None:
    forbidden = {"evrptw", "txnopt", "txnopt_cases", "txnopt_evidence"}
    observations = {
        path.relative_to(ROOT).as_posix(): sorted(_import_roots(path) & forbidden)
        for path in _python_files("txnopt_legacy")
    }
    assert all(not imports for imports in observations.values()), observations


def test_runner_and_reviewer_share_only_inward_codecs_not_each_other() -> None:
    runner = SRC / "txnopt_evidence/runner.py"
    reviewer = SRC / "txnopt_evidence/reviewer.py"
    assert "txnopt_evidence.reviewer" not in runner.read_text(encoding="utf-8")
    assert "txnopt_evidence.runner" not in reviewer.read_text(encoding="utf-8")


def test_importing_txnopt_does_not_import_evrptw() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, txnopt; assert 'evrptw' not in sys.modules; "
            "assert txnopt.__all__ == ['TxnRuntime', 'SearchKernel', 'Oracle', "
            "'RunConfig', 'RunResult']",
        ],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
