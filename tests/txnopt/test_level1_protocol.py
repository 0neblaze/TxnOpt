from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_level1_case_set_and_holdout_boundary_are_exact() -> None:
    path = ROOT / "experiments/txnopt/level1-protocol.json"
    digest, filename = (
        path.with_suffix(".json.sha256").read_text(encoding="utf-8").strip().split()
    )
    assert filename == path.name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    protocol = json.loads(path.read_bytes())
    domains = protocol["domains"]

    assert len(domains["evrptw"]["pilot"]) == 8
    assert len(domains["evrptw"]["validation"]) == 4
    assert len(domains["rcpsp"]["pilot"]) == 16
    assert len(domains["rcpsp"]["validation"]) == 8
    assert len(protocol["seeds"]) == 10
    assert len(set(protocol["seeds"])) == 10
    assert protocol["formal_axes"] == [
        "serial_1",
        "txnopt_1",
        "txnopt_4",
        "barrier_4",
    ]
    assert protocol["level2_holdout_opened"] is False
    assert protocol["level3_holdout_opened"] is False
