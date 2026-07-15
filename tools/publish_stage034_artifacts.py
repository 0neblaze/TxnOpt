from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from evrptw.artifacts import ArtifactReader, find_manifest, verify_manifest

from publish_stage033_artifacts import REGISTRY_FIELDS, instance_seed, load_json, sha256

SMOKE_LABEL = "stage03.4_control_parallel_attempt08"
FORMAL_LABEL = "stage03.4_control_parallel_attempt09"
READY_STATUSES = {"READY_FOR_STAGE034_FORMAL", "READY_FOR_STAGE04"}


def _review_directory(root: Path, run_label: str) -> Path | None:
    tracked = root / "experiments/summaries" / f"{run_label}_review"
    local = root / "results" / run_label / "review"
    if tracked.is_dir():
        return tracked
    return local if local.is_dir() else None


def _review_status(path: Path | None) -> str:
    if path is None or not (path / "review_manifest.json").is_file():
        return "not_present"
    return str(load_json(path / "review_manifest.json").get("status", "NOT_READY"))


def _publish_review_summaries(root: Path, run_label: str, expected_status: str) -> None:
    source = root / "results" / run_label / "review"
    target = root / "experiments/summaries" / f"{run_label}_review"
    if _review_status(source) != expected_status:
        raise RuntimeError(f"{run_label} review is not {expected_status}")
    if target.exists():
        for path in source.iterdir():
            if path.is_file() and sha256(path) != sha256(target / path.name):
                raise RuntimeError(f"tracked review differs from accepted raw review: {path.name}")
        return
    shutil.copytree(source, target)


def _base_row(
    run_label: str,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    review_status: str,
) -> dict[str, object]:
    ready = review_status in READY_STATUSES
    return {
        "stage_id": "stage03.4",
        "component": "control_parallel",
        "run_label": run_label,
        "attempt_or_rerun": run_label.rsplit("_", 1)[-1],
        "canonical_run_label": run_label,
        "legacy_run_label": "",
        "scope": metadata.get("scope", "unknown"),
        "instance_scope": json.dumps(metadata.get("instances", []), separators=(",", ":")),
        "seed_scope": json.dumps(metadata.get("seeds", []), separators=(",", ":")),
        "artifact_status": "verified_current" if ready else "retained_unsuccessful",
        "run_status": manifest.get("status", "unknown"),
        "failure_reason": "" if ready else review_status,
        "validator_status": "pass" if ready else "not_verified",
        "raw_review_status": review_status,
        "trusted_review_status": review_status if ready else "not_applicable",
        "raw_to_summary_status": "pass" if ready else "not_published",
        "checksum_source": "artifact_manifest",
        "source_hash": metadata.get("source_sha256", ""),
        "configuration_hash": metadata.get("configuration_sha256", ""),
        "stage00_manifest_hash": metadata.get("stage00_manifest_sha256", ""),
        "repository_revision": metadata.get("repository_revision", ""),
        "repository_dirty": metadata.get("repository_dirty", ""),
        "reference_repositories": json.dumps(
            metadata.get("reference_repositories", {}), sort_keys=True, separators=(",", ":")
        ),
        "comparison_baseline": "stage03.3_exact_deadline_attempt06",
        "supersedes": "",
        "legacy_path_mapping": "physical_current_path",
        "storage_policy_version": manifest.get("storage_policy_version", ""),
        "evidence_completeness": manifest.get("evidence_completeness", ""),
        "policy_compliance": manifest.get("policy_compliance", ""),
    }


def _rows_for_run(root: Path, run_dir: Path) -> list[dict[str, object]]:
    manifest = verify_manifest(run_dir)
    run_label = str(manifest["run_label"])
    if re.fullmatch(r"stage03\.4_control_parallel_(?:attempt|rerun)\d{2}", run_label) is None:
        raise ValueError(f"non-canonical Stage 3.4 run label: {run_label}")
    reader = ArtifactReader(run_dir)
    metadata_ref = next(
        item for item in manifest["artifacts"] if item["artifact_type"] == "manifest_metadata"
    )
    metadata = reader.read_json(str(metadata_ref["relative_path"]))
    review_dir = _review_directory(root, run_label)
    status = _review_status(review_dir)
    base = _base_row(run_label, metadata, manifest, status)
    environment_hashes = {
        instance_seed(str(item["relative_path"])): str(item["checksum"])
        for item in manifest["artifacts"]
        if item["artifact_type"] == "environment"
    }
    instance_hashes: dict[tuple[str, str], str] = {}
    for item in manifest["artifacts"]:
        if item["artifact_type"] != "trace":
            continue
        key = instance_seed(str(item["relative_path"]))
        trace = reader.read_json(str(item["relative_path"]))
        axes = trace.get("axes")
        if isinstance(axes, dict):
            first_axis = next((value for value in axes.values() if isinstance(value, dict)), None)
            if first_axis is not None:
                instance_hashes[key] = str(first_axis.get("instance_hash", ""))
    rows: list[dict[str, object]] = []
    for item in manifest["artifacts"]:
        relative = str(item["relative_path"])
        key = instance_seed(relative)
        rows.append(
            {
                **base,
                "instance": key[0],
                "seed": key[1],
                "artifact_type": item["artifact_type"],
                "canonical_path": f"results/{run_label}/{relative}",
                "legacy_path": "",
                "checksum": item["checksum"],
                "instance_hash": instance_hashes.get(key, ""),
                "environment_hash": environment_hashes.get(key, ""),
                "storage_format": item.get("storage_format", ""),
                "compression": item.get("compression", ""),
                "retention_class": item.get("retention_class", ""),
                "schema_fingerprint": item.get("schema_fingerprint", ""),
                "row_count": item.get("row_count", ""),
                "byte_size": item.get("byte_size", ""),
            }
        )
    manifest_path = find_manifest(run_dir)
    for artifact_type, path in (
        ("manifest", manifest_path),
        ("manifest_checksum", manifest_path.with_suffix(".sha256")),
    ):
        rows.append(
            {
                **base,
                "instance": "__all__",
                "seed": "__all__",
                "artifact_type": artifact_type,
                "canonical_path": str(path.relative_to(root)),
                "legacy_path": "",
                "checksum": sha256(path),
                "instance_hash": "",
                "environment_hash": "",
                "storage_format": "json_control" if artifact_type == "manifest" else "sha256",
                "compression": "none",
                "retention_class": "control",
                "schema_fingerprint": "",
                "row_count": "",
                "byte_size": path.stat().st_size,
            }
        )
    if review_dir is not None:
        review_manifest = load_json(review_dir / "review_manifest.json")
        for path in sorted(review_dir.iterdir()):
            if not path.is_file():
                continue
            expected = review_manifest.get("files", {}).get(path.name)
            if expected is not None and expected != sha256(path):
                raise RuntimeError(f"review checksum mismatch: {path}")
            rows.append(
                {
                    **base,
                    "instance": "__all__",
                    "seed": "__all__",
                    "artifact_type": f"review_{path.stem}",
                    "canonical_path": str(path.relative_to(root)),
                    "legacy_path": "",
                    "checksum": sha256(path),
                    "checksum_source": "review_manifest" if expected else "direct_sha256",
                    "instance_hash": "",
                    "environment_hash": "",
                    "storage_format": path.suffix.lstrip("."),
                    "compression": "none",
                    "retention_class": "review",
                    "schema_fingerprint": "",
                    "row_count": sum(1 for _ in path.open(encoding="utf-8")) - 1
                    if path.suffix == ".csv"
                    else "",
                    "byte_size": path.stat().st_size,
                }
            )
    return rows


def publish(root: Path) -> tuple[Path, Path]:
    _publish_review_summaries(root, SMOKE_LABEL, "READY_FOR_STAGE034_FORMAL")
    _publish_review_summaries(root, FORMAL_LABEL, "READY_FOR_STAGE04")
    run_dirs = sorted((root / "results").glob("stage03.4_control_parallel_attempt??"))
    rows = [row for run_dir in run_dirs for row in _rows_for_run(root, run_dir)]
    registry = root / "experiments/registries/stage03.4_artifact_registry.csv"
    manifest_path = root / "experiments/manifests/stage03.4_control_parallel_artifact_manifest.json"
    with registry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    statuses = {
        run_dir.name: _review_status(_review_directory(root, run_dir.name))
        for run_dir in run_dirs
    }
    with registry.open(newline="", encoding="utf-8") as handle:
        published_rows = list(csv.DictReader(handle))
    rows_payload = json.dumps(
        published_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "stage034-artifact-publication-v1",
                "stage_id": "stage03.4",
                "component": "control_parallel",
                "artifact_registry": str(registry.relative_to(root)),
                "artifact_registry_sha256": sha256(registry),
                "artifact_registry_rows_sha256": hashlib.sha256(rows_payload).hexdigest(),
                "artifact_count": len(rows),
                "canonical_run_labels": [path.name for path in run_dirs],
                "run_review_statuses": statuses,
                "smoke_run_label": SMOKE_LABEL,
                "smoke_review_status": statuses.get(SMOKE_LABEL),
                "formal_run_label": FORMAL_LABEL,
                "formal_review_status": statuses.get(FORMAL_LABEL),
                "comparison_baseline": "stage03.3_exact_deadline_attempt06",
                "raw_artifacts_moved": False,
                "historical_paths_overwritten": False,
                "checks": {
                    "canonical_labels": True,
                    "manifest_and_sidecar_integrity": True,
                    "checksums_recomputed": True,
                    "failed_attempts_retained": True,
                    "smoke_review_ready": statuses.get(SMOKE_LABEL)
                    == "READY_FOR_STAGE034_FORMAL",
                    "formal_review_ready": statuses.get(FORMAL_LABEL) == "READY_FOR_STAGE04",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return registry, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish verified Stage 3.4 registry views")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    for path in publish(arguments.root.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
