from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tomllib
from collections import Counter
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


def test_source_disposition_accounts_for_every_target_byte() -> None:
    with (ROOT / "docs/provenance/source-file-disposition.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 878
    assert {row["disposition"] for row in rows} == {
        "declarative_rename",
        "local_external_artifact",
        "migrated_modified_publication",
        "migrated_unchanged",
        "school_specific_exclusion",
    }
    for row in rows:
        if not row["target_path"]:
            assert row["disposition"] in {
                "local_external_artifact",
                "school_specific_exclusion",
            }
            continue
        target = ROOT / row["target_path"]
        assert target.is_file()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == row["target_sha256"]


def test_source_disposition_summary_counts_replay_from_csv() -> None:
    with (ROOT / "docs/provenance/source-file-disposition.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads(
        (ROOT / "docs/provenance/source-file-disposition-summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["source_file_count"] == len(rows)
    assert summary["counts"] == dict(Counter(row["disposition"] for row in rows))
    assert sum(summary["counts"].values()) == summary["source_file_count"]


def test_artifact_index_does_not_promote_partial_formal_evidence() -> None:
    payload = json.loads((ROOT / "artifacts/index.json").read_text(encoding="utf-8"))
    entries = {entry["run_label"]: entry for entry in payload["entries"]}

    pilot = entries["stage05.2_benchmark_attempt72"]
    formal = entries["stage05.2_benchmark_attempt73"]
    assert pilot["status"] == "accepted_pilot"
    assert pilot["review_status"] == "READY_FOR_STAGE052_FORMAL_BENCHMARK"
    assert pilot["source_revision"] == "a5cf00f7580fc2632179495a739a110786ace87d"
    assert formal["status"] == "partial_unreviewed"
    assert formal["review_status"] is None
    assert formal["source_manifest_status"] == "planned"
    assert formal["source_revision"] == "a5cf00f7580fc2632179495a739a110786ace87d"
    assert formal["campaign_manifest_sha256"] == (
        "1b442b6c7f3b4f4982f308db2e00562614f9cd2097d59d8e5fadba459c91b01e"
    )
    assert set(formal["archived_batch_manifest_sha256_by_batch"]) == {
        "batch0001",
        "batch0002",
    }


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


def test_local_runtime_and_large_artifact_paths_are_ignored() -> None:
    paths = (
        "FURP_Showcase.pdf",
        "configs/stage052_campaign_lock.local.json",
        "configs/stage052_campaign_lock.local.sha256",
        "docs/roadmap/evrptw-research-roadmap.local.md",
        "document/literature/README.md",
        "document/literature/example-paper.pdf",
        "scratch/reviewer.sqlite",
        "scratch/reviewer.sqlite3",
    )
    result = subprocess.run(
        ("git", "check-ignore", "-z", "--stdin"),
        cwd=ROOT,
        input="\0".join(paths) + "\0",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert set(result.stdout.rstrip("\0").split("\0")) == set(paths)
