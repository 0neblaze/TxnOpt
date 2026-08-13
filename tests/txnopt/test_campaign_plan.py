from __future__ import annotations

import hashlib
import json
from pathlib import Path

from txnopt_evidence.campaign import materialize_level1_plan


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    evrptw = tmp_path / "tiny_ev.txt"
    evrptw.write_text(
        """StringID Type x y demand ReadyTime DueDate ServiceTime
D0 d 0 0 0 0 100 0
C1 c 1 0 1 0 100 0

Q / 100 /
C / 10 /
r / 1 /
g / 0.1 /
v / 1 /
""",
        encoding="utf-8",
    )
    rcpsp = tmp_path / "tiny_rc.sm"
    rcpsp.write_text(
        """PRECEDENCE RELATIONS:
 1 1 1 2
 2 1 0
REQUESTS/DURATIONS:
 1 1 0 0
 2 1 1 1
RESOURCEAVAILABILITIES:
 R 1
 1
""",
        encoding="utf-8",
    )
    return evrptw, rcpsp


def test_campaign_materializer_binds_every_axis_without_starting_runs(
    tmp_path: Path,
) -> None:
    evrptw, rcpsp = _write_inputs(tmp_path)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema_version": "txnopt-level1-protocol-v1",
                "holdout_opened": False,
                "domains": {
                    "evrptw": {"pilot": ["tinyC5"], "validation": []},
                    "rcpsp": {"pilot": ["tiny_rc"], "validation": []},
                },
                "seeds": [2014],
                "formal_axes": ["serial_1", "txnopt_1", "txnopt_4", "barrier_4"],
                "budgets": ["fixed_work", "fixed_time"],
            }
        ),
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": "txnopt-level1-case-catalog-v1",
                "domains": {
                    "evrptw": {"tinyC5": str(evrptw)},
                    "rcpsp": {"tiny_rc": str(rcpsp)},
                },
            }
        ),
        encoding="utf-8",
    )

    manifest_path = materialize_level1_plan(
        protocol,
        catalog,
        destination=tmp_path / "plan",
        raw_output_root=tmp_path / "raw",
        fixed_work=16,
        fixed_time_seconds=0.5,
        max_rounds=4,
        evrptw_max_candidates=3,
        rcpsp_max_candidates=5,
    )
    manifest = json.loads(manifest_path.read_bytes())
    sidecar_digest, sidecar_name = (
        manifest_path.with_suffix(".json.sha256")
        .read_text(encoding="utf-8")
        .strip()
        .split()
    )

    assert manifest["status"] == "PLANNED_NOT_STARTED"
    assert manifest["config_count"] == 16
    assert manifest["holdout_opened"] is False
    assert manifest["cloud_purchase_authorized"] is False
    assert manifest["max_candidates"] == {"evrptw": 3, "rcpsp": 5}
    assert len(list((tmp_path / "plan/configs").glob("*.json"))) == 16
    assert list((tmp_path / "plan/configs").glob("*tinyc5*"))
    evrptw_config = json.loads(
        next((tmp_path / "plan/configs").glob("*tinyc5*serial_1_fixed_work*.json")).read_bytes()
    )
    rcpsp_config = json.loads(
        next((tmp_path / "plan/configs").glob("*tiny_rc*serial_1_fixed_work*.json")).read_bytes()
    )
    assert evrptw_config["case"]["max_candidates"] == 3
    assert rcpsp_config["case"]["max_candidates"] == 5
    assert sidecar_name == "manifest.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == sidecar_digest
    assert not (tmp_path / "raw").exists()
