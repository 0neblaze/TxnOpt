from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from evrptw.artifacts import ArtifactReader, find_manifest, verify_manifest

REGISTRY_FIELDS = (
    "stage_id", "component", "run_label", "attempt_or_rerun",
    "canonical_run_label", "legacy_run_label", "scope", "instance", "seed",
    "instance_scope", "seed_scope", "artifact_type", "artifact_status",
    "canonical_path", "legacy_path", "run_status", "failure_reason",
    "validator_status", "raw_review_status", "trusted_review_status",
    "raw_to_summary_status", "checksum", "checksum_source", "source_hash",
    "configuration_hash", "instance_hash", "environment_hash",
    "stage00_manifest_hash", "repository_revision", "repository_dirty",
    "reference_repositories", "comparison_baseline", "supersedes",
    "legacy_path_mapping", "storage_policy_version", "storage_format",
    "compression", "retention_class", "evidence_completeness",
    "schema_fingerprint", "row_count", "byte_size", "policy_compliance",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def review_directory(root: Path, run_label: str) -> Path | None:
    tracked = root / "experiments/summaries" / f"{run_label}_review"
    local = root / "results" / run_label / "review"
    if tracked.is_dir():
        return tracked
    return local if local.is_dir() else None


def review_status(path: Path | None) -> str:
    if path is None or not (path / "review_manifest.json").is_file():
        return "not_present"
    return str(load_json(path / "review_manifest.json").get("status", "NOT_READY"))


def instance_seed(relative_path: str) -> tuple[str, str]:
    parts = Path(relative_path).parts
    if len(parts) >= 3 and parts[0] != "control" and parts[1].isdigit():
        return parts[0], parts[1]
    return "__all__", "__all__"


def base_row(
    *,
    run_label: str,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    status: str,
) -> dict[str, object]:
    ready = status in {"READY_FOR_STAGE033_FORMAL", "READY_FOR_STAGE03_4"}
    return {
        "stage_id": "stage03.3",
        "component": "exact_deadline",
        "run_label": run_label,
        "attempt_or_rerun": run_label.rsplit("_", 1)[-1],
        "canonical_run_label": run_label,
        "legacy_run_label": "",
        "scope": metadata.get("scope", "unknown"),
        "instance_scope": json.dumps(metadata.get("instances", []), separators=(",", ":")),
        "seed_scope": json.dumps(metadata.get("seeds", []), separators=(",", ":")),
        "artifact_status": "verified_current" if ready else "retained_unsuccessful",
        "run_status": manifest.get("status", "unknown"),
        "failure_reason": "" if ready else status,
        "validator_status": "pass" if ready else "not_verified",
        "raw_review_status": status,
        "trusted_review_status": status if status == "READY_FOR_STAGE03_4" else "not_applicable",
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
        "comparison_baseline": "stage03.2_cache_incremental_attempt03",
        "supersedes": "",
        "legacy_path_mapping": "physical_current_path",
        "storage_policy_version": manifest.get("storage_policy_version", ""),
        "evidence_completeness": manifest.get("evidence_completeness", ""),
        "policy_compliance": manifest.get("policy_compliance", ""),
    }


def rows_for_run(root: Path, run_dir: Path) -> list[dict[str, object]]:
    manifest = verify_manifest(run_dir)
    run_label = str(manifest["run_label"])
    if re.fullmatch(r"stage03\.3_exact_deadline_(?:attempt|rerun)\d{2}", run_label) is None:
        raise ValueError(f"non-canonical Stage 3.3 run label: {run_label}")
    reader = ArtifactReader(run_dir)
    metadata_ref = next(
        item for item in manifest["artifacts"] if item["artifact_type"] == "manifest_metadata"
    )
    metadata = reader.read_json(metadata_ref["relative_path"])
    review_dir = review_directory(root, run_label)
    status = review_status(review_dir)
    base = base_row(run_label=run_label, metadata=metadata, manifest=manifest, status=status)
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
            fixed = axes.get("fixed_exact_calls")
            if isinstance(fixed, dict):
                instance_hashes[key] = str(fixed.get("instance_hash", ""))
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
    run_dirs = sorted((root / "results").glob("stage03.3_exact_deadline_attempt??"))
    if not run_dirs:
        raise FileNotFoundError("no Stage 3.3 attempt directories found")
    rows = [row for run_dir in run_dirs for row in rows_for_run(root, run_dir)]
    registry = root / "experiments/registries/stage03.3_artifact_registry.csv"
    manifest_path = root / "experiments/manifests/stage03.3_exact_deadline_artifact_manifest.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    statuses = {
        run_dir.name: review_status(review_directory(root, run_dir.name))
        for run_dir in run_dirs
    }
    with registry.open(newline="", encoding="utf-8") as handle:
        published_rows = list(csv.DictReader(handle))
    rows_payload = json.dumps(
        published_rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "stage033-artifact-publication-v1",
                "stage_id": "stage03.3",
                "component": "exact_deadline",
                "artifact_registry": str(registry.relative_to(root)),
                "artifact_registry_sha256": sha256(registry),
                "artifact_registry_rows_sha256": hashlib.sha256(rows_payload).hexdigest(),
                "artifact_count": len(rows),
                "canonical_run_labels": [path.name for path in run_dirs],
                "run_review_statuses": statuses,
                "formal_run_label": "stage03.3_exact_deadline_attempt06",
                "formal_review_status": statuses.get(
                    "stage03.3_exact_deadline_attempt06"
                ),
                "comparison_baseline": "stage03.2_cache_incremental_attempt03",
                "raw_artifacts_moved": False,
                "historical_paths_overwritten": False,
                "checks": {
                    "canonical_labels": True,
                    "manifest_and_sidecar_integrity": True,
                    "checksums_recomputed": True,
                    "failed_attempts_retained": True,
                    "formal_review_ready": statuses.get(
                        "stage03.3_exact_deadline_attempt06"
                    )
                    == "READY_FOR_STAGE03_4",
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
    parser = argparse.ArgumentParser(description="Publish verified Stage 3.3 registry views")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    for path in publish(arguments.root.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
