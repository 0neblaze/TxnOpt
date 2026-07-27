from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_stage_commit_map_contains_exactly_208_unique_mappings() -> None:
    with (ROOT / "docs/provenance/legacy-stage-commit-map.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 208
    assert len({row["legacy_sha"] for row in rows}) == 208
    assert len({row["public_sha"] for row in rows}) == 208
    assert all(len(row["legacy_sha"]) == 40 for row in rows)
    assert all(len(row["public_sha"]) == 40 for row in rows)


def test_artifact_index_does_not_promote_partial_formal_evidence() -> None:
    payload = json.loads((ROOT / "artifacts/index.json").read_text(encoding="utf-8"))
    entries = {entry["run_label"]: entry for entry in payload["entries"]}

    pilot = entries["stage05.2_benchmark_attempt72"]
    formal = entries["stage05.2_benchmark_attempt73"]
    assert pilot["status"] == "accepted_pilot"
    assert pilot["review_status"] == "READY_FOR_STAGE052_FORMAL_BENCHMARK"
    assert formal["status"] == "partial_unreviewed"
    assert formal["review_status"] is None
    assert formal["source_manifest_status"] == "planned"


def test_commercial_solvers_are_optional_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    mandatory = "\n".join(project["dependencies"]).lower()
    optional = "\n".join(project["optional-dependencies"]["commercial-solvers"]).lower()

    assert all(name not in mandatory for name in ("cplex", "docplex", "gurobipy"))
    assert all(name in optional for name in ("cplex", "docplex", "gurobipy"))


def test_migration_receipt_hashes_match_published_bytes() -> None:
    manifest = json.loads(
        (ROOT / "docs/provenance/migration-manifest.json").read_text(encoding="utf-8")
    )
    source = manifest["source"]
    expected = {
        "docs/provenance/source-file-inventory.tsv": source[
            "source_file_inventory_sha256"
        ],
        "docs/provenance/source-file-disposition.csv": source[
            "source_file_disposition_sha256"
        ],
        "docs/provenance/source-file-disposition-summary.json": source[
            "source_file_disposition_summary_sha256"
        ],
    }

    for relative_path, expected_sha256 in expected.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == (
            expected_sha256
        )
