"""Stage 5.1 — Best-known values compilation runner.

Collects published Schneider E-VRPTW best-known solution values, performs
model compatibility assessment, and writes the canonical artifact bundle.

Unlike solver stages, this runner does not invoke ``solve_alns``.  Its
evidence is the BKS reference data itself plus the compatibility assessment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evrptw.artifacts import (
    ArtifactBundleWriter,
    ArtifactRunContext,
    ArtifactStorageConfig,
    verify_manifest,
)
from evrptw.best_known import (
    BEST_KNOWN_VALUES,
    COMPATIBILITY_ASSESSMENT,
    SOURCE_REFERENCES,
    TOTAL_INSTANCES,
)
from evrptw.environment import collect_environment

STAGE051_SCHEMA_VERSION = "stage05.1-best-known-v6"
STAGE051_RUN_LABEL = re.compile(r"^stage05\.1_best_known_(?:attempt|rerun)[0-9]{2}$")

BKS_DATA_FIELDS: tuple[str, ...] = (
    "instance",
    "paper_name",
    "customer_count",
    "class_name",
    "bks_vehicles",
    "bks_distance",
    "bks_charging_time",
    "bks_charging_count",
    "source_ref",
    "source_doi",
    "source_year",
    "source_in_vor_collection",
    "compilation_ref",
    "compilation_doi",
    "source_table",
    "optimal_proven",
    "charging_model",
    "objective_function",
    "distance_metric",
    "model_compatible",
    "compatibility_notes",
)

COMPATIBILITY_FIELDS: tuple[str, ...] = (
    "dimension",
    "our_model",
    "published_model",
    "compatible",
    "notes",
)

_OUR_OBJECTIVE = (
    "lexicographic(vehicle_count, total_distance, "
    "total_charging_time, charging_count)"
)
_OUR_CHARGING = "full_recharge: t = (Q - b) * g"
_OUR_DISTANCE = "unrounded Euclidean (math.hypot)"
_COMPATIBILITY_NOTES = (
    "Objective mismatch: our 4-component lexicographic objective adds "
    "total_charging_time and charging_count not present in published "
    "objectives; HPH uses weighted sum rather than strict lexicographic; "
    "distance metric may differ due to rounding conventions. "
    "No gap computation permitted."
)


@dataclass(frozen=True, slots=True)
class Stage051Config:
    """Configuration for Stage 5.1."""

    benchmark_dir: Path
    stage04_review_manifest: Path
    artifact_storage: ArtifactStorageConfig


def load_stage051_config(path: Path) -> Stage051Config:
    """Parse the Stage 5.1 TOML configuration."""

    with path.open("rb") as handle:
        data = tomllib.load(handle)
    storage_section = data.get("artifact_storage", {})
    return Stage051Config(
        benchmark_dir=Path(data.get("benchmark", {}).get("directory", "data/schneider")),
        stage04_review_manifest=Path(
            data.get("stage05_1", {}).get(
                "stage04_review_manifest",
                "experiments/manifests/stage04_adaptive_weights_artifact_manifest.json",
            )
        ),
        artifact_storage=ArtifactStorageConfig(
            enabled=storage_section.get("enabled", True),
            storage_policy_version=storage_section.get(
                "storage_policy_version", "artifact-storage-v1"
            ),
            event_format=storage_section.get("event_format", "parquet"),
            compression=storage_section.get("compression", "zstd"),
            compression_level=storage_section.get("compression_level", 3),
            critical_evidence=storage_section.get("critical_evidence", "full"),
            diagnostic_evidence=storage_section.get("diagnostic_evidence", "aggregate"),
            per_instance_seed_max_bytes=storage_section.get(
                "per_instance_seed_max_bytes", 2 * 1024 * 1024 * 1024
            ),
            per_run_max_bytes=storage_section.get(
                "per_run_max_bytes", 32 * 1024 * 1024 * 1024
            ),
        ),
    )


def validate_stage051_run_label(run_label: str) -> None:
    """Validate the canonical run label."""

    if not STAGE051_RUN_LABEL.fullmatch(run_label):
        raise ValueError(
            f"non-canonical Stage 5.1 run label: {run_label}; "
            "expected stage05.1_best_known_attemptNN or "
            "stage05.1_best_known_rerunNN"
        )


def validate_stage051_output_dir(repo_root: Path, output_dir: Path, run_label: str) -> Path:
    """Return the canonical run directory or reject nested/non-canonical layouts."""

    resolved = output_dir.resolve()
    if resolved != repo_root.resolve() / "results" / run_label:
        raise ValueError("Stage 5.1 output must be results/<canonical-run-label>")
    return resolved


def run_stage051_best_known(
    *,
    config_path: Path,
    output_dir: Path,
    run_label: str,
) -> dict[str, Path]:
    """Compile BKS data and write the canonical artifact bundle."""

    validate_stage051_run_label(run_label)

    repo_root = _find_repo_root(config_path)
    _require_clean_repository(repo_root)

    config = load_stage051_config(config_path)
    run_dir = validate_stage051_output_dir(repo_root, output_dir, run_label)
    stage04_prerequisite = verify_stage04_prerequisite(
        repo_root, config.stage04_review_manifest
    )
    if run_dir.exists():
        raise RuntimeError(f"output directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    environment = collect_environment()
    git_revision = _git(repo_root, "rev-parse", "HEAD")
    git_dirty = bool(_git(repo_root, "status", "--porcelain"))
    source_hashes = _source_sha256(repo_root)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()

    metadata: dict[str, Any] = {
        "schema_version": STAGE051_SCHEMA_VERSION,
        "run_label": run_label,
        "stage_id": "stage05.1",
        "component": "best_known",
        "total_instances": TOTAL_INSTANCES,
        "environment": environment,
        "git_revision": git_revision,
        "git_dirty": git_dirty,
        "source_sha256": source_hashes,
        "config_sha256": config_hash,
        "stage04_prerequisite": stage04_prerequisite,
        "source_references": {
            abbr: {
                "authors": ref.authors,
                "year": ref.year,
                "title": ref.title,
                "journal": ref.journal,
                "doi": ref.doi,
                "in_vor_collection": ref.in_vor_collection,
            }
            for abbr, ref in SOURCE_REFERENCES.items()
        },
    }

    context = ArtifactRunContext(
        stage_id="stage05.1",
        component="best_known",
        run_label=run_label,
    )
    writer = ArtifactBundleWriter(
        run_dir=run_dir,
        context=context,
        config=config.artifact_storage,
    )
    writer.write_control(metadata=metadata, configuration_path=config_path)

    # --- BKS data CSV ---
    bks_path = run_dir / f"{run_label}_bks_data.csv"
    _write_bks_csv(bks_path)
    writer.record_existing_file(
        bks_path,
        artifact_type="bks_data",
        retention_class="critical",
        storage_format="csv_critical",
        row_count=TOTAL_INSTANCES,
    )

    # --- Compatibility assessment CSV ---
    compat_path = run_dir / f"{run_label}_compatibility_assessment.csv"
    _write_compatibility_csv(compat_path)
    writer.record_existing_file(
        compat_path,
        artifact_type="compatibility_assessment",
        retention_class="critical",
        storage_format="csv_critical",
        row_count=len(COMPATIBILITY_ASSESSMENT.dimensions),
    )

    # --- Summary report ---
    report_path = run_dir / f"{run_label}_summary_report.md"
    _write_summary_report(report_path, run_label, git_revision)
    writer.record_existing_file(
        report_path,
        artifact_type="summary_report",
        retention_class="critical",
        storage_format="markdown",
    )

    result = writer.finalize()
    return {
        "run_dir": result.run_dir,
        "bks_data": bks_path,
        "compatibility_assessment": compat_path,
        "summary_report": report_path,
        "manifest": result.manifest_path,
        "manifest_sidecar": result.manifest_sidecar_path,
    }


# ---------------------------------------------------------------------------
# CSV / Markdown writers
# ---------------------------------------------------------------------------


def _write_bks_csv(path: Path) -> None:
    """Write the BKS data CSV."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BKS_DATA_FIELDS)
        writer.writeheader()
        writer.writerows(canonical_stage051_rows()[0])


def _write_compatibility_csv(path: Path) -> None:
    """Write the compatibility assessment CSV."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPATIBILITY_FIELDS)
        writer.writeheader()
        writer.writerows(canonical_stage051_rows()[1])


def canonical_stage051_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return the complete canonical CSV rows used by both runner and reviewer."""

    bks_rows: list[dict[str, str]] = []
    for rec in BEST_KNOWN_VALUES:
        src = SOURCE_REFERENCES[rec.source_ref]
        comp = SOURCE_REFERENCES[rec.compilation_ref]
        bks_rows.append({
            "instance": rec.instance,
            "paper_name": rec.paper_name,
            "customer_count": str(rec.customer_count),
            "class_name": rec.class_name,
            "bks_vehicles": str(rec.bks_vehicles),
            "bks_distance": str(rec.bks_distance),
            "bks_charging_time": "unknown",
            "bks_charging_count": "unknown",
            "source_ref": rec.source_ref,
            "source_doi": src.doi,
            "source_year": str(src.year),
            "source_in_vor_collection": str(src.in_vor_collection),
            "compilation_ref": rec.compilation_ref,
            "compilation_doi": comp.doi,
            "source_table": rec.source_table,
            "optimal_proven": str(rec.optimal_proven),
            "charging_model": _OUR_CHARGING,
            "objective_function": _OUR_OBJECTIVE,
            "distance_metric": _OUR_DISTANCE,
            "model_compatible": "False",
            "compatibility_notes": _COMPATIBILITY_NOTES,
        })
    compatibility_rows = [
        {
            "dimension": dim.dimension,
            "our_model": dim.our_model,
            "published_model": dim.published_model,
            "compatible": str(dim.compatible),
            "notes": dim.notes,
        }
        for dim in COMPATIBILITY_ASSESSMENT.dimensions
    ]
    return bks_rows, compatibility_rows


def _write_summary_report(path: Path, run_label: str, git_revision: str) -> None:
    """Write the summary report in Markdown."""

    lines: list[str] = []
    lines.append(f"# Stage 5.1 Best-Known Values — {run_label}")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        f"This report compiles best-known solution (BKS) values for all "
        f"{TOTAL_INSTANCES} Schneider et al. (2014) E-VRPTW benchmark "
        f"instances and assesses model compatibility with our lexicographic "
        f"objective."
    )
    lines.append("")
    lines.append("## Data Sources")
    lines.append("")
    for abbr, ref in SOURCE_REFERENCES.items():
        vor = "yes" if ref.in_vor_collection else "no"
        lines.append(
            f"- **{abbr}**: {ref.authors} ({ref.year}), "
            f"\"{ref.title}\", *{ref.journal}*, "
            f"DOI: {ref.doi}, in VOR collection: {vor}"
        )
    lines.append("")
    lines.append("## Model Compatibility Assessment")
    lines.append("")
    lines.append(f"**Overall compatible: {COMPATIBILITY_ASSESSMENT.overall_compatible}**")
    lines.append("")
    lines.append("| Dimension | Our Model | Published Model | Compatible |")
    lines.append("|-----------|----------|-----------------|------------|")
    for dim in COMPATIBILITY_ASSESSMENT.dimensions:
        lines.append(
            f"| {dim.dimension} | {dim.our_model} | {dim.published_model} "
            f"| {'yes' if dim.compatible else 'no'} |"
        )
    lines.append("")
    lines.append(COMPATIBILITY_ASSESSMENT.summary)
    lines.append("")
    lines.append("## BKS Values Summary")
    lines.append("")
    small = [r for r in BEST_KNOWN_VALUES if r.customer_count < 100]
    large = [r for r in BEST_KNOWN_VALUES if r.customer_count == 100]
    lines.append(f"- Small instances (5/10/15 customers): {len(small)}")
    lines.append(f"- Large instances (100 customers): {len(large)}")
    lines.append(f"- Total: {TOTAL_INSTANCES}")
    lines.append("")
    lines.append(
        "All BKS values are marked `model_compatible=False` due to objective "
        "and distance-metric mismatch. No gap is computed."
    )
    lines.append("")
    lines.append(
        "Charging time and charging count are `unknown` for all instances "
        "because published BKS tables report only vehicle count and total "
        "distance."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Git revision: `{git_revision}`")
    lines.append(f"- Schema version: `{STAGE051_SCHEMA_VERSION}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_repo_root(path: Path) -> Path:
    current = path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return path.resolve().parent


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _require_clean_repository(root: Path) -> None:
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("Stage 5.1 runner requires a clean main repository commit")


def verify_stage04_prerequisite(
    root: Path, configured_path: Path
) -> dict[str, str]:
    publication_path = configured_path if configured_path.is_absolute() else root / configured_path
    sidecar_path = publication_path.with_suffix(publication_path.suffix + ".sha256")
    expected_digest = sidecar_path.read_text(encoding="utf-8").split()[0]
    if _sha256(publication_path) != expected_digest:
        raise RuntimeError("Stage 4 publication manifest sidecar mismatch")
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    run_label = publication.get("run_label")
    if (
        publication.get("schema_version") != "stage04-publication-v6"
        or publication.get("review_status") != "READY_FOR_STAGE05"
        or not isinstance(run_label, str)
    ):
        raise RuntimeError("Stage 5.1 requires a published Stage 4 READY_FOR_STAGE05 review")
    raw_dir = root / "results" / run_label
    verify_manifest(raw_dir)
    raw_manifest_path = raw_dir / "control" / f"{run_label}_manifest.json"
    if publication.get("raw_manifest_sha256") != _sha256(raw_manifest_path):
        raise RuntimeError("Stage 4 publication raw manifest hash is invalid")
    files = publication.get("files")
    if (
        not isinstance(files, dict)
        or not files
        or any(not isinstance(record, dict) for record in files.values())
        or any(
            set(record) < {"path", "sha256"}
            or _sha256(root / str(record["path"])) != str(record["sha256"])
            for record in files.values()
        )
    ):
        raise RuntimeError("Stage 4 published review file hashes are invalid")
    review_record = files.get("review_manifest")
    if not isinstance(review_record, dict):
        raise RuntimeError("Stage 4 publication is missing its review manifest")
    review = json.loads(
        (root / str(review_record["path"])).read_text(encoding="utf-8")
    )
    if (
        review.get("schema_version") != "stage04-review-v6"
        or review.get("status") != "READY_FOR_STAGE05"
    ):
        raise RuntimeError("Stage 4 v6 review is not READY_FOR_STAGE05")
    formal_ok, formal_detail = validate_stage04_formal_identity(publication, review)
    if not formal_ok:
        raise RuntimeError(f"Stage 4 Formal identity is invalid: {formal_detail}")
    return {
        "publication_manifest": str(publication_path.relative_to(root)),
        "publication_sha256": expected_digest,
        "run_label": run_label,
        "review_status": "READY_FOR_STAGE05",
    }


def validate_stage04_formal_identity(
    publication: Mapping[str, object], review: Mapping[str, object]
) -> tuple[bool, str]:
    """Require Stage 5.1 to inherit exactly the Stage 4 Formal evidence."""

    failures: list[str] = []
    for name, payload in (("publication", publication), ("review", review)):
        if payload.get("scope") != "formal":
            failures.append(f"{name} scope is not formal")
        if payload.get("observed_axes") != 144:
            failures.append(f"{name} observed_axes is not 144")
    if publication.get("run_label") != review.get("run_label"):
        failures.append("publication/review run_label mismatch")
    return not failures, "; ".join(failures) if failures else "Stage 4 Formal identity is exact"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_sha256(root: Path) -> dict[str, str]:
    files = (
        "src/evrptw/best_known.py",
        "src/evrptw/experiments/stage051_best_known.py",
    )
    result: dict[str, str] = {}
    for relative in files:
        path = root / relative
        if path.exists():
            result[relative] = _sha256(path)
    return result


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile Stage 5.1 best-known values"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/stage051_best_known.toml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    arguments = parser.parse_args()
    outputs = run_stage051_best_known(
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        run_label=arguments.run_label,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
