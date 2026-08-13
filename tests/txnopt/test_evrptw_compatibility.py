from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_new_case_model_import_does_not_activate_legacy_or_numpy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import txnopt_cases.evrptw.models; "
            "assert 'evrptw' not in sys.modules; assert 'numpy' not in sys.modules",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_new_case_model_native_backend_fails_closed_until_explicitly_bound() -> None:
    script = """
from txnopt_cases.evrptw.models import Instance, Node, NodeType, Vehicle
nodes=(
    Node('D', NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
    Node('C', NodeType.CUSTOMER, 3.0, 4.0, 1.0, 0.0, 100.0, 0.0),
)
try:
    Instance('tiny', nodes, Vehicle(100.0, 10.0, 1.0, 1.0, 1.0), 'native')
except RuntimeError as error:
    assert 'not explicitly bound' in str(error)
else:
    raise AssertionError('unbound native backend did not fail')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_legacy_model_and_parser_are_identity_preserving_adapters() -> None:
    script = """
import evrptw.models as legacy_models
import evrptw.parser as legacy_parser
import txnopt_cases.evrptw.models as models
import txnopt_cases.evrptw.parser as parser
assert legacy_models.Instance is models.Instance
assert legacy_models.Node is models.Node
assert legacy_models.NodeType is models.NodeType
assert legacy_models.Vehicle is models.Vehicle
assert legacy_parser.parse_schneider is parser.parse_schneider
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
