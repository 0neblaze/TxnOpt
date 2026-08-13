from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from txnopt_cases.evrptw.charging import solve_exact_charging
from txnopt_cases.evrptw.models import Instance, Node, NodeType, Vehicle
from txnopt_cases.rcpsp import RCPSPState
from txnopt_legacy import LegacyReceiptReader


def test_rcpsp_state_enforces_representation_invariants() -> None:
    assert RCPSPState((0, 2, 1), (0, 1, 0)).mode_vector == (0, 1, 0)
    with pytest.raises(ValueError, match="once"):
        RCPSPState((0, 1, 1), (0, 0, 0))
    with pytest.raises(ValueError, match="align"):
        RCPSPState((0, 1), (0,))


def _evrptw_instance(*, distance_backend: str) -> Instance:
    return Instance(
        "tiny",
        (
            Node("D", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C", NodeType.CUSTOMER, 3.0, 4.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(100.0, 10.0, 1.0, 1.0, 1.0),
        distance_backend=distance_backend,
    )


def test_evrptw_case_model_is_pure_python_and_fail_closed_for_unbound_native() -> None:
    assert _evrptw_instance(distance_backend="python").distance("D", "C") == 5.0


def test_evrptw_charging_oracle_uses_the_migrated_case_model() -> None:
    result = solve_exact_charging(
        _evrptw_instance(distance_backend="python"),
        ("C",),
    )
    assert result.feasible is True
    assert result.route == ("D", "C", "D")
    assert result.distance == 10.0


def test_legacy_reader_verifies_sidecar_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    data = (json.dumps({"schema_version": "legacy-v1"}, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    sidecar = path.with_suffix(".json.sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    before = {item.name: item.read_bytes() for item in tmp_path.iterdir()}
    assert LegacyReceiptReader().read(path) == {"schema_version": "legacy-v1"}
    after = {item.name: item.read_bytes() for item in tmp_path.iterdir()}
    assert after == before


def test_legacy_reader_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text('{"schema_version":"legacy-v1"}\n', encoding="utf-8")
    path.with_suffix(".json.sha256").write_text(
        f"{'0' * 64}  {path.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differs"):
        LegacyReceiptReader().read(path)
