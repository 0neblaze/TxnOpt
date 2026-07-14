#!/usr/bin/env python3
"""Register completed Stage 3 evidence under the canonical artifact contract.

This tool is deliberately a retrospective migration aid.  It never invokes
the solver, moves a historical directory, rewrites a raw artifact, or copies a
summary.  It verifies the existing legacy evidence, writes a canonical logical
registry, and records the old-path compatibility mapping required by the Stage
3 artifact contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evrptw.artifacts import ArtifactReader

SCHEMA_VERSION = "stage03-artifact-migration-v1"
CANONICAL_LABEL_RE = re.compile(
    r"^stage03\.(?P<minor>[012])_[a-z0-9_]+_(?:attempt|rerun)[0-9]{2}$"
)
READY_REVIEW_STATUSES = {
    "READY_FOR_STAGE03_ACCELERATION",
    "READY_FOR_STAGE03_FORMAL_MEASUREMENT",
    "READY_FOR_STAGE03_1",
    "READY_FOR_STAGE031_FORMAL_MEASUREMENT",
    "READY_FOR_STAGE31",
    "READY_FOR_STAGE03_2",
    "READY_FOR_STAGE032_FORMAL_MEASUREMENT",
    "READY_FOR_STAGE03_3",
}
RAW_PER_RUN_COMPARISON_FIELDS = (
    "instance",
    "seed",
    "objective_key",
    "vehicle_count",
    "total_distance",
    "total_charging_time",
    "charging_count",
    "feasible",
    "trace_exact_calls",
    "trace_cache_hits",
    "trace_precomputed_routes",
    "trace_route_evaluations",
    "trace_deadline_events",
    "trace_reconciliation_status",
)
STAGE03_1_COMPARISON_FIELDS = (
    "trace_screening_calls",
    "trace_screening_passes",
    "trace_screening_rejections",
    "trace_screening_cache_hits",
    "trace_screening_exact_call_blocked",
    "trace_screening_reason_counts",
)
STAGE03_2_COMPARISON_FIELDS = (
    "trace_cache_incremental_counts",
    "trace_incremental_propagations",
    "trace_incremental_fallbacks",
)
REGISTRY_FIELDS = (
    "stage_id",
    "component",
    "run_label",
    "attempt_or_rerun",
    "canonical_run_label",
    "legacy_run_label",
    "scope",
    "instance",
    "seed",
    "instance_scope",
    "seed_scope",
    "artifact_type",
    "artifact_status",
    "canonical_path",
    "legacy_path",
    "run_status",
    "failure_reason",
    "validator_status",
    "raw_review_status",
    "trusted_review_status",
    "raw_to_summary_status",
    "checksum",
    "checksum_source",
    "source_hash",
    "configuration_hash",
    "instance_hash",
    "environment_hash",
    "stage00_manifest_hash",
    "repository_revision",
    "repository_dirty",
    "reference_repositories",
    "comparison_baseline",
    "supersedes",
    "legacy_path_mapping",
    "storage_policy_version",
    "storage_format",
    "compression",
    "retention_class",
    "evidence_completeness",
    "schema_fingerprint",
    "row_count",
    "byte_size",
    "policy_compliance",
)
LEGACY_MAP_FIELDS = (
    "mapping_type",
    "stage_id",
    "component",
    "canonical_label",
    "canonical_path",
    "legacy_path",
    "artifact_type",
    "status",
    "checksum",
    "checksum_source",
    "repository_revision",
    "repository_dirty",
    "stage00_manifest_hash",
    "reason",
)
HARD_CHECKS = (
    "canonical_label_check",
    "canonical_run_label_unique",
    "canonical_artifact_path_check",
    "canonical_artifact_path_unique",
    "artifact_type_check",
    "legacy_paths_preserved",
    "formal_raw_manifest_integrity",
    "formal_raw_review_manifest_integrity",
    "trusted_formal_review_integrity",
    "formal_raw_to_summary",
    "historical_manifest_errors_explicit",
    "stage00_and_stage02_legacy_mapping",
)


@dataclass(frozen=True, slots=True)
class StageSpec:
    stage_id: str
    component: str
    legacy_prefix: str
    run_labels: tuple[tuple[str, str], ...]
    trusted_formal_review: str
    comparison_baseline: str


SPECS = (
    StageSpec(
        "stage03.0",
        "measurement",
        "stage03_measurement",
        (
            ("stage03_measurement_smoke01", "stage03.0_measurement_attempt01"),
            ("stage03_measurement_smoke02", "stage03.0_measurement_attempt02"),
            ("stage03_measurement_smoke03", "stage03.0_measurement_attempt03"),
            ("stage03_measurement_smoke04", "stage03.0_measurement_attempt04"),
            ("stage03_measurement_formal01", "stage03.0_measurement_attempt05"),
        ),
        "experiments/summaries/stage03_measurement_formal01_review_manifest.json",
        "stage02.3_constraint_guided_attempt16 + stage02.3_constraint_guided_rerun09",
    ),
    StageSpec(
        "stage03.1",
        "screening",
        "stage031_cheap_screening",
        (
            ("stage031_cheap_screening_smoke01", "stage03.1_screening_attempt01"),
            ("stage031_cheap_screening_smoke02", "stage03.1_screening_attempt02"),
            ("stage031_cheap_screening_smoke03", "stage03.1_screening_attempt03"),
            ("stage031_cheap_screening_formal01", "stage03.1_screening_attempt04"),
        ),
        "experiments/summaries/stage031_cheap_screening_formal01_review_manifest.json",
        "stage03.0_measurement_attempt05",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def root_relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def normalized_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    return [
        {str(key): str(value) for key, value in row.items()}
        for row in rows
    ]


def verify_generated_outputs(
    root: Path,
    manifest: dict[str, object],
    registry_path: Path,
    manifest_path: Path,
    legacy_path: Path,
    expected_registry_rows: list[dict[str, object]],
) -> list[str]:
    errors: list[str] = []
    if not registry_path.is_file():
        errors.append(
            "generated artifact registry is missing: "
            f"{root_relative(root, registry_path)}"
        )
    if not manifest_path.is_file():
        errors.append(
            "generated artifact manifest is missing: "
            f"{root_relative(root, manifest_path)}"
        )
    if not legacy_path.is_file():
        errors.append(f"generated legacy path map is missing: {root_relative(root, legacy_path)}")
    if errors:
        return errors

    existing_manifest = read_json(manifest_path)
    recorded_registry_hash = str(existing_manifest.get("artifact_registry_sha256", ""))
    if recorded_registry_hash != sha256_file(registry_path):
        errors.append("generated artifact registry checksum does not match its manifest")
    recorded_rows_hash = str(
        existing_manifest.get("artifact_registry_rows_sha256", "")
    )
    actual_rows = read_csv_rows(registry_path)
    if recorded_rows_hash != payload_sha256(actual_rows):
        errors.append("generated artifact registry row hash does not match its manifest")
    if actual_rows != normalized_rows(expected_registry_rows):
        errors.append("generated artifact registry rows differ from the current raw evidence")
    recorded_legacy_hash = str(existing_manifest.get("legacy_path_map_sha256", ""))
    if recorded_legacy_hash != sha256_file(legacy_path):
        errors.append("generated legacy path map checksum does not match its manifest")
    return errors


def verify_raw_manifest(run_dir: Path) -> tuple[dict[str, str], list[str]]:
    current_manifest = sorted((run_dir / "control").glob("*_manifest.json"))
    if current_manifest:
        try:
            reader = ArtifactReader(run_dir)
        except (OSError, ValueError, RuntimeError) as error:
            return {}, [f"current raw manifest verification failed: {error}"]
        return {
            str(item["relative_path"]): str(item["checksum"])
            for item in reader.manifest.get("artifacts", [])
            if isinstance(item, dict)
        }, []
    manifest_path = run_dir / "manifest.json"
    sidecar_path = run_dir / "manifest.sha256"
    errors: list[str] = []
    if not manifest_path.is_file() or not sidecar_path.is_file():
        return {}, ["raw manifest or sidecar is missing"]
    if sidecar_path.read_text(encoding="utf-8").strip() != sha256_file(manifest_path):
        errors.append("raw manifest sidecar hash mismatch")
    payload = read_json(manifest_path)
    expected = {str(key): str(value) for key, value in payload.get("files", {}).items()}
    for relative, expected_hash in sorted(expected.items()):
        path = run_dir / relative
        if not path.is_file():
            errors.append(f"raw artifact missing: {relative}")
            continue
        if sha256_file(path) != expected_hash:
            errors.append(f"raw artifact checksum mismatch: {relative}")
    listed = set(expected) | {"manifest.json", "manifest.sha256"}
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if relative.startswith("review/"):
            continue
        if relative not in listed:
            errors.append(f"raw artifact is not listed by manifest: {relative}")
    return expected, errors


def verify_review_manifest(review_dir: Path) -> tuple[dict[str, str], str, list[str]]:
    manifest_path = review_dir / "review_manifest.json"
    if not manifest_path.is_file():
        return {}, "not_present", []
    errors: list[str] = []
    payload = read_json(manifest_path)
    recorded = str(payload.get("manifest_payload_sha256", ""))
    without_hash = dict(payload)
    without_hash.pop("manifest_payload_sha256", None)
    if not recorded or recorded != payload_sha256(without_hash):
        errors.append("review manifest payload hash mismatch")
    hashes = {str(key): str(value) for key, value in payload.get("files", {}).items()}
    for name, expected_hash in sorted(hashes.items()):
        path = review_dir / name
        if not path.is_file():
            errors.append(f"review artifact missing: {name}")
            continue
        if sha256_file(path) != expected_hash:
            errors.append(f"review artifact checksum mismatch: {name}")
    status = str(payload.get("status", "unknown"))
    return hashes, status, errors


def verify_trusted_review(root: Path, relative_path: str) -> tuple[str, list[str]]:
    path = root / relative_path
    errors: list[str] = []
    if not path.is_file():
        return "missing", [f"trusted review manifest missing: {relative_path}"]
    payload = read_json(path)
    recorded = str(payload.get("manifest_payload_sha256", ""))
    without_hash = dict(payload)
    without_hash.pop("manifest_payload_sha256", None)
    if not recorded or recorded != payload_sha256(without_hash):
        errors.append("trusted review manifest payload hash mismatch")
    review_label = str(payload.get("review_label", ""))
    for name, expected_hash in dict(payload.get("files", {})).items():
        artifact = path.parent / str(name)
        if not artifact.is_file() and review_label:
            artifact = path.parent / f"{review_label}_{name}"
        if (
            not artifact.is_file()
            and name == "recomputed_per_run_results.csv"
            and review_label
        ):
            artifact = path.parent / f"{review_label}_per_run_results.csv"
        if not artifact.is_file():
            errors.append(f"trusted review artifact missing: {name}")
        elif sha256_file(artifact) != str(expected_hash):
            errors.append(f"trusted review artifact checksum mismatch: {name}")
    status = str(payload.get("status", "unknown"))
    if status not in READY_REVIEW_STATUSES:
        errors.append(f"trusted review is not ready: {status}")
    return status, errors


def classify_artifact(relative: Path) -> str:
    parts = relative.parts
    name = relative.name
    if parts and parts[0] == "control":
        if name.endswith("_manifest.sha256") or name == "manifest.sha256":
            return "manifest_checksum"
        if "_run_metadata" in name:
            return "manifest_metadata"
        if "_config" in name:
            return "config"
        if "_raw_per_run_results" in name:
            return "raw_per_run_results"
    if name.endswith(".parquet"):
        if "_route_dictionary_" in name:
            return "route_dictionary"
        if "_screening_checks_" in name:
            return "screening_checks"
        if "_diagnostic_" in name:
            return "diagnostic"
        if "_events_" in name:
            return "events"
    if len(parts) >= 2 and parts[0] not in {
        "raw",
        "solutions",
        "events",
        "traces",
        "environments",
        "failures",
        "review",
        "control",
    }:
        if "_solution_" in name:
            return "solution"
        if "_trace_" in name:
            return "trace"
        if "_environment_" in name:
            return "environment"
        if "_failure_" in name:
            return "failure"
        if "_raw_" in name:
            return "raw"
    if parts and parts[0] == "raw":
        return "raw"
    if parts and parts[0] == "solutions":
        return "solution"
    if parts and parts[0] == "events":
        return "events"
    if parts and parts[0] == "traces":
        return "trace"
    if parts and parts[0] == "environments":
        return "environment"
    if parts and parts[0] == "failures":
        return "failure"
    if parts and parts[0] == "review":
        stem = Path(name).stem
        if name == "review_manifest.json":
            return "manifest"
        if stem in {"review_report", "review_findings", "deadline_report", "trace_reconciliation"}:
            return stem
        if stem in {"stage03_readiness", "screening_reason_statistics"}:
            return stem
        if stem in {"baseline_comparison", "stage03_formal_comparison"}:
            return stem
        if stem in {"summary_results", "recomputed_per_run_results"}:
            return stem
        return "review"
    return {
        "run_metadata.json": "manifest_metadata",
        "parameters.toml": "config",
        "raw_per_run_results.csv": "raw_per_run_results",
        "environment.json": "environment",
        "manifest.json": "manifest",
        "manifest.sha256": "manifest_checksum",
        "interruption.json": "failure",
    }.get(name, Path(name).stem)


def file_suffix(path: Path) -> str:
    # ``Path.suffixes`` treats the dot in canonical IDs such as ``stage03.0``
    # as part of a compound extension.  Artifact extensions in this schema are
    # the final suffix only (``.json``, ``.csv``, ``.md``, ...).
    return path.suffix


def run_id_lookup(raw_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in raw_rows:
        for field, value in row.items():
            if not field.endswith("_path") or not value:
                continue
            output[Path(value).stem] = row
        experiment_id = str(row.get("experiment_id", ""))
        if experiment_id:
            output[experiment_id] = row
    return output


def compare_raw_to_summary(
    raw_path: Path,
    summary_path: Path,
    *,
    screening: bool,
    cache_incremental: bool = False,
) -> tuple[str, list[str]]:
    if not summary_path.is_file():
        return "not_published", []
    with raw_path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    with summary_path.open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    def key(row: dict[str, str]) -> tuple[str, str]:
        return str(row.get("instance", "")), str(row.get("seed", ""))

    raw_by_key = {key(row): row for row in raw_rows}
    summary_by_key = {key(row): row for row in summary_rows}
    if set(raw_by_key) != set(summary_by_key) or len(raw_rows) != len(raw_by_key):
        return "fail", ["raw and published summary coverage differ"]
    fields = list(RAW_PER_RUN_COMPARISON_FIELDS)
    if screening:
        fields.extend(STAGE03_1_COMPARISON_FIELDS)
    if cache_incremental:
        fields.extend(STAGE03_2_COMPARISON_FIELDS)
    differences = [
        f"{row_key}:{field}"
        for row_key in sorted(raw_by_key)
        for field in fields
        if raw_by_key[row_key].get(field, "") != summary_by_key[row_key].get(field, "")
    ]
    return ("pass" if not differences else "fail"), differences[:20]


def canonical_path(
    run_label: str,
    relative: Path,
    artifact_type: str,
    row: dict[str, str] | None,
) -> tuple[str, str, str]:
    instance = str(row.get("instance", "")) if row else ""
    seed = str(row.get("seed", "")) if row else ""
    suffix = file_suffix(relative)
    if instance and seed:
        filename = canonical_filename(
            run_label,
            artifact_type,
            suffix,
            instance,
            seed,
        )
        return (
            f"results/{run_label}/{instance}/{seed}/{filename}",
            instance,
            seed,
        )
    filename = canonical_filename(run_label, artifact_type, suffix)
    if relative.parts and relative.parts[0] == "review":
        return f"results/{run_label}/review/{filename}", "__all__", "__all__"
    if artifact_type == "raw_per_run_results":
        return f"results/{run_label}/all/all/{filename}", "__all__", "__all__"
    return f"results/{run_label}/control/{filename}", "__all__", "__all__"


def attempt_or_rerun(run_label: str) -> str:
    match = re.search(r"_(attempt|rerun)([0-9]{2})$", run_label)
    if not match:
        raise RuntimeError(f"canonical run label has no attempt/rerun suffix: {run_label}")
    return f"{match.group(1)}{match.group(2)}"


def legacy_run_directory(root: Path, legacy_label: str) -> Path:
    """Resolve the old directory spelling without broad glob guessing."""

    if legacy_label.startswith("stage03_measurement_"):
        directory_name = legacy_label.replace(
            "stage03_measurement_", "stage03-measurement_", 1
        )
    elif legacy_label.startswith("stage031_cheap_screening_"):
        directory_name = legacy_label.replace(
            "stage031_cheap_screening_", "stage031-cheap-screening_", 1
        )
    elif legacy_label.startswith("stage03.2_cache_incremental_"):
        directory_name = legacy_label
    else:
        raise RuntimeError(f"unsupported legacy Stage 3 run label: {legacy_label}")
    path = root / "results" / directory_name
    if not path.is_dir():
        raise RuntimeError(f"legacy run directory not found: {legacy_label} -> {path}")
    return path


def run_directory(root: Path, legacy_label: str) -> Path:
    """Use an existing canonical bundle before falling back to old spelling."""

    current = root / "results" / legacy_label
    if current.is_dir():
        return current
    return legacy_run_directory(root, legacy_label)


def metadata_path(run_dir: Path) -> Path:
    legacy = run_dir / "run_metadata.json"
    if legacy.is_file():
        return legacy
    candidates = sorted((run_dir / "control").glob("*_run_metadata.json"))
    if not candidates:
        raise RuntimeError(f"run metadata is missing: {run_dir}")
    return candidates[0]


def raw_per_run_path(run_dir: Path) -> Path:
    legacy = run_dir / "raw_per_run_results.csv"
    if legacy.is_file():
        return legacy
    candidates = sorted((run_dir / "control").glob("*_raw_per_run_results.csv"))
    if not candidates:
        raise RuntimeError(f"raw per-run index is missing: {run_dir}")
    return candidates[0]


def active_specs(root: Path) -> tuple[StageSpec, ...]:
    """Include every physically present canonical Stage 3 run directory.

    Historical mappings in ``SPECS`` remain the source of truth for the old
    labels.  New canonical directories are appended to the corresponding
    stage specification so a later registry generation cannot silently omit a
    valid Stage 3.0 or Stage 3.1 run merely because it was created after this
    tool was released.
    """

    definitions = (
        (
            "stage03.0",
            "measurement",
            "stage03_measurement",
            "stage02.3_constraint_guided_attempt16 + stage02.3_constraint_guided_rerun09",
            "experiments/summaries/stage03_measurement_formal01_review_manifest.json",
        ),
        (
            "stage03.1",
            "screening",
            "stage031_cheap_screening",
            "stage03.0_measurement_attempt05",
            "experiments/summaries/stage031_cheap_screening_formal01_review_manifest.json",
        ),
        (
            "stage03.2",
            "cache_incremental",
            "stage03.2_cache_incremental",
            "stage03.1_screening_attempt04",
            "experiments/summaries/stage032_cheap_screening_formal01_review_manifest.json",
        ),
    )
    result_root = root / "results"
    specs = list(SPECS)
    for stage_id, component, legacy_prefix, comparison_baseline, default_trusted in definitions:
        prefix = f"{stage_id}_{component}_"
        current_labels = tuple(
            path.name
            for path in sorted(result_root.glob(f"{prefix}*"))
            if path.is_dir()
            and re.fullmatch(
                rf"{re.escape(prefix)}(?:attempt|rerun)[0-9]{{2}}",
                path.name,
            )
        )
        if not current_labels:
            continue

        existing_index = next(
            (index for index, spec in enumerate(specs) if spec.stage_id == stage_id),
            None,
        )
        existing = specs[existing_index] if existing_index is not None else None
        existing_labels = {
            canonical
            for _legacy, canonical in (existing.run_labels if existing else ())
        }
        additions = tuple(
            (label, label) for label in current_labels if label not in existing_labels
        )
        if not additions:
            continue

        run_labels = (existing.run_labels if existing else ()) + additions
        formal_current_labels: list[str] = []
        for label in current_labels:
            try:
                if read_json(metadata_path(result_root / label)).get("scope") == "formal":
                    formal_current_labels.append(label)
            except (OSError, RuntimeError, json.JSONDecodeError):
                # The builder will retain the manifest error for an incomplete
                # run.  It must still be registered rather than disappearing
                # from the registry because metadata could not be read here.
                continue
        trusted = existing.trusted_formal_review if existing else default_trusted
        if formal_current_labels:
            trusted = f"results/{sorted(formal_current_labels)[-1]}/review/review_manifest.json"
        replacement = StageSpec(
            stage_id,
            component,
            existing.legacy_prefix if existing else legacy_prefix,
            run_labels,
            trusted,
            existing.comparison_baseline if existing else comparison_baseline,
        )
        if existing_index is None:
            specs.append(replacement)
        else:
            specs[existing_index] = replacement
    return tuple(specs)


def summary_artifact_type(summary_suffix: str) -> str:
    stem = Path(summary_suffix).stem
    if stem == "review_manifest":
        return "manifest"
    return stem


def run_failure_reason(raw_rows: list[dict[str, str]]) -> str:
    reasons = sorted(
        {
            str(row.get("failure_reason", "")).strip()
            for row in raw_rows
            if str(row.get("failure_reason", "")).strip()
        }
    )
    return ";".join(reasons)


def validator_status(
    raw_review_status: str,
    trusted_review_status: str,
) -> str:
    if raw_review_status in READY_REVIEW_STATUSES:
        return "pass"
    if trusted_review_status in READY_REVIEW_STATUSES:
        return "pass"
    return "not_verified"


def canonical_filename(
    run_label: str,
    artifact_type: str,
    suffix: str,
    instance: str = "__all__",
    seed: str = "__all__",
) -> str:
    filename_type = {
        "manifest_metadata": "run_metadata",
        "manifest_checksum": "manifest",
    }.get(artifact_type, artifact_type)
    filename = f"{run_label}_{filename_type}"
    if instance != "__all__" and seed != "__all__":
        filename += f"_{instance}_{seed}"
    return f"{filename}{suffix}"


def canonical_path_is_valid(
    path: str,
    run_label: str,
    artifact_type: str,
    instance: str,
    seed: str,
) -> bool:
    relative = Path(path)
    expected = canonical_filename(
        run_label,
        artifact_type,
        file_suffix(relative),
        instance,
        seed,
    )
    if relative.name != expected:
        return False
    if instance != "__all__" and seed != "__all__":
        return relative.parts[:4] == (
            "results",
            run_label,
            instance,
            seed,
        )
    return relative.parts[:3] in {
        ("results", run_label, "control"),
        ("results", run_label, "review"),
        ("results", run_label, "all"),
    } or relative.parts[:2] == ("experiments", "summaries")


def metadata_context(
    metadata: dict[str, Any],
    row: dict[str, str] | None,
) -> dict[str, object]:
    row = row or {}
    return {
        "source_hash": row.get(
            "algorithm_source_sha256", metadata.get("algorithm_source_sha256", "")
        ),
        "configuration_hash": row.get(
            "configuration_sha256", metadata.get("configuration_sha256", "")
        ),
        "instance_hash": row.get("instance_sha256", ""),
        "environment_hash": row.get("environment_sha256", ""),
        "stage00_manifest_hash": row.get(
            "stage00_manifest_sha256", metadata.get("stage00_manifest_sha256", "")
        ),
        "repository_revision": row.get(
            "repository_revision", metadata.get("repository_revision", "")
        ),
        "repository_dirty": row.get("repository_dirty", metadata.get("repository_dirty", "")),
        "reference_repositories": json.dumps(
            metadata.get("reference_repositories", {}),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }


def build_stage(
    root: Path, spec: StageSpec
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    registry_rows: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []
    legacy_map: list[dict[str, object]] = []
    canonical_labels = [canonical for _, canonical in spec.run_labels]
    if len(set(canonical_labels)) != len(canonical_labels):
        raise RuntimeError(f"duplicate canonical labels for {spec.stage_id}")
    if any(not CANONICAL_LABEL_RE.fullmatch(label) for label in canonical_labels):
        raise RuntimeError(f"invalid canonical label in {spec.stage_id}: {canonical_labels}")

    trusted_status, trusted_errors = verify_trusted_review(
        root, spec.trusted_formal_review
    )
    for legacy_label, canonical_label in spec.run_labels:
        run_dir = run_directory(root, legacy_label)
        metadata = read_json(metadata_path(run_dir))
        raw_csv = raw_per_run_path(run_dir)
        with raw_csv.open(encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
        row_lookup = run_id_lookup(raw_rows)
        raw_hashes, raw_errors = verify_raw_manifest(run_dir)
        current_artifact_metadata: dict[str, dict[str, object]] = {}
        current_storage = bool(
            sorted((run_dir / "control").glob("*_manifest.json"))
        )
        if current_storage:
            current_reader = ArtifactReader(run_dir, verify=False)
            current_artifact_metadata = {
                str(item.get("relative_path")): dict(item)
                for item in current_reader.manifest.get("artifacts", [])
                if isinstance(item, dict)
            }
        review_hashes, raw_review_status, review_errors = verify_review_manifest(
            run_dir / "review"
        )
        screening = spec.stage_id in {"stage03.1", "stage03.2"}
        cache_incremental = spec.stage_id == "stage03.2"
        summary_path = root / "experiments" / "summaries" / f"{legacy_label}_per_run_results.csv"
        if not summary_path.is_file():
            summary_path = (
                root / "experiments" / "summaries" / f"{canonical_label}_per_run_results.csv"
            )
        summary_status, summary_errors = compare_raw_to_summary(
            raw_csv,
            summary_path,
            screening=screening,
            cache_incremental=cache_incremental,
        )
        scope = str(metadata.get("scope", ""))
        formal_run = scope == "formal"
        trusted_for_run = trusted_status if formal_run else "not_applicable"
        effective_review_status = (
            trusted_for_run if formal_run else raw_review_status
        )
        run_validator_status = validator_status(
            raw_review_status, trusted_for_run
        )
        failure_reason = ";".join(
            sorted(
                {
                    reason
                    for reason in [run_failure_reason(raw_rows), *raw_errors, *review_errors]
                    if reason
                }
            )
        )
        run_statuses = Counter(str(row.get("status", "unknown")) for row in raw_rows)
        run_status = (
            next(iter(run_statuses))
            if len(run_statuses) == 1
            else ",".join(f"{key}:{value}" for key, value in sorted(run_statuses.items()))
        )
        run_records.append(
            {
                "run_label": canonical_label,
                "canonical_run_label": canonical_label,
                "legacy_run_label": legacy_label,
                "legacy_path": root_relative(root, run_dir),
                "scope": scope,
                "attempt_or_rerun": attempt_or_rerun(canonical_label),
                "instance_scope": metadata.get("expected_instances", []),
                "seed_scope": metadata.get("expected_seeds", []),
                "run_status": run_status,
                "failure_reason": failure_reason,
                "validator_status": run_validator_status,
                "raw_review_status": raw_review_status,
                "trusted_review_status": trusted_for_run,
                "review_status": effective_review_status,
                "raw_manifest_status": "pass" if not raw_errors else "fail",
                "raw_manifest_errors": raw_errors,
                "review_manifest_status": (
                    "pass"
                    if not review_errors and raw_review_status != "not_present"
                    else "not_present"
                    if not review_errors
                    else "fail"
                ),
                "review_manifest_errors": review_errors,
                "raw_to_summary_status": summary_status,
                "raw_to_summary_errors": summary_errors,
            }
        )
        legacy_map.append(
            {
                "mapping_type": "historical_run_directory",
                "stage_id": spec.stage_id,
                "component": spec.component,
                "canonical_label": canonical_label,
                "canonical_path": f"results/{canonical_label}",
                "legacy_path": root_relative(root, run_dir),
                "artifact_type": "run_directory",
                "status": "preserved",
                "checksum": sha256_file(
                    next(
                        iter(sorted((run_dir / "control").glob("*_manifest.json"))),
                        run_dir / "manifest.json",
                    )
                ),
                "checksum_source": "raw_manifest",
                "repository_revision": metadata.get("repository_revision", ""),
                "repository_dirty": metadata.get("repository_dirty", ""),
                "stage00_manifest_hash": metadata.get("stage00_manifest_sha256", ""),
                "reason": "legacy path is immutable; canonical path is a logical registry view",
            }
        )

        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(run_dir)
            artifact_type = classify_artifact(relative)
            row = row_lookup.get(path.stem)
            canonical, instance, seed = canonical_path(
                canonical_label, relative, artifact_type, row
            )
            if relative.as_posix() in raw_hashes:
                checksum = raw_hashes[relative.as_posix()]
                checksum_source = "raw_manifest"
            elif (
                relative.parts
                and relative.parts[0] == "review"
                and relative.name in review_hashes
            ):
                checksum = review_hashes[relative.name]
                checksum_source = "review_manifest"
            else:
                checksum = sha256_file(path)
                checksum_source = "computed"
            context = metadata_context(metadata, row)
            storage = current_artifact_metadata.get(relative.as_posix(), {})
            storage_policy_version = (
                str(metadata.get("storage_policy_version", "artifact-storage-v1"))
                if current_storage
                else "legacy"
            )
            artifact_manifest_error = bool(raw_errors)
            if relative.parts and relative.parts[0] == "review":
                artifact_manifest_error = bool(review_errors)
            artifact_status = (
                "verified_legacy"
                if not artifact_manifest_error
                else "legacy_with_manifest_error"
            )
            row_failure_reason = (
                str(row.get("failure_reason", "")).strip()
                if row
                else ""
            ) or failure_reason
            registry_rows.append(
                {
                    "stage_id": spec.stage_id,
                    "component": spec.component,
                    "run_label": canonical_label,
                    "attempt_or_rerun": attempt_or_rerun(canonical_label),
                    "canonical_run_label": canonical_label,
                    "legacy_run_label": legacy_label,
                    "scope": scope,
                    "instance": instance,
                    "seed": seed,
                    "instance_scope": json.dumps(
                        metadata.get("expected_instances", []), ensure_ascii=False
                    ),
                    "seed_scope": json.dumps(
                        metadata.get("expected_seeds", []), ensure_ascii=False
                    ),
                    "artifact_type": artifact_type,
                    "artifact_status": artifact_status,
                    "canonical_path": canonical,
                    "legacy_path": root_relative(root, path),
                    "run_status": run_status,
                    "failure_reason": row_failure_reason,
                    "validator_status": run_validator_status,
                    "raw_review_status": raw_review_status,
                    "trusted_review_status": trusted_for_run,
                    "raw_to_summary_status": summary_status,
                    "checksum": checksum,
                    "checksum_source": checksum_source,
                    **context,
                    "comparison_baseline": spec.comparison_baseline,
                    "supersedes": "",
                    "legacy_path_mapping": "preserve;see stage03_legacy_path_map.csv",
                    "storage_policy_version": storage_policy_version,
                    "storage_format": str(
                        storage.get(
                            "storage_format",
                            "legacy_json_or_jsonl" if not current_storage else "json_control",
                        )
                    ),
                    "compression": str(storage.get("compression", "none")),
                    "retention_class": str(
                        storage.get("retention_class", "legacy")
                    ),
                    "evidence_completeness": str(
                        storage.get("evidence_completeness", "legacy_unknown")
                    ),
                    "schema_fingerprint": str(storage.get("schema_fingerprint", "")),
                    "row_count": storage.get("row_count", ""),
                    "byte_size": storage.get("byte_size", path.stat().st_size),
                    "policy_compliance": "current" if current_storage else "legacy_compatible",
                }
            )

        tracked_summaries = sorted(
            path
            for path in (root / "experiments" / "summaries").glob(
                f"{legacy_label}_*"
            )
            if path.is_file()
        )
        for tracked_summary in tracked_summaries:
            summary_suffix = tracked_summary.name[len(legacy_label) + 1 :]
            summary_type = summary_artifact_type(summary_suffix)
            summary_canonical = (
                "experiments/summaries/"
                + canonical_filename(
                    canonical_label,
                    summary_type,
                    file_suffix(Path(summary_suffix)),
                )
            )
            summary_consistency_status = summary_status
            summary_artifact_status = (
                "published_after_review"
                if effective_review_status in READY_REVIEW_STATUSES
                and summary_consistency_status == "pass"
                else "legacy_summary_without_passing_review"
            )
            registry_rows.append(
                {
                    "stage_id": spec.stage_id,
                    "component": spec.component,
                    "run_label": canonical_label,
                    "attempt_or_rerun": attempt_or_rerun(canonical_label),
                    "canonical_run_label": canonical_label,
                    "legacy_run_label": legacy_label,
                    "scope": scope,
                    "instance": "__all__",
                    "seed": "__all__",
                    "artifact_type": summary_type,
                    "instance_scope": json.dumps(
                        metadata.get("expected_instances", []), ensure_ascii=False
                    ),
                    "seed_scope": json.dumps(
                        metadata.get("expected_seeds", []), ensure_ascii=False
                    ),
                    "artifact_status": summary_artifact_status,
                    "canonical_path": summary_canonical,
                    "legacy_path": root_relative(root, tracked_summary),
                    "run_status": run_status,
                    "failure_reason": failure_reason,
                    "validator_status": run_validator_status,
                    "raw_review_status": raw_review_status,
                    "trusted_review_status": trusted_for_run,
                    "raw_to_summary_status": summary_consistency_status,
                    "checksum": sha256_file(tracked_summary),
                    "checksum_source": "tracked_summary",
                    **metadata_context(metadata, None),
                    "comparison_baseline": spec.comparison_baseline,
                    "supersedes": "",
                    "legacy_path_mapping": "preserve;canonical summary name is registry-only",
                    "storage_policy_version": (
                        "artifact-storage-v1" if current_storage else "legacy"
                    ),
                    "storage_format": "csv_summary",
                    "compression": "none",
                    "retention_class": "diagnostic",
                    "evidence_completeness": "complete",
                    "schema_fingerprint": "",
                    "row_count": (
                        len(tracked_summary.read_text(encoding="utf-8").splitlines()) - 1
                    ),
                    "byte_size": tracked_summary.stat().st_size,
                    "policy_compliance": (
                        "current" if current_storage else "legacy_compatible"
                    ),
                }
            )

    formal_records = [
        record for record in run_records if record["scope"] == "formal"
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage_id": spec.stage_id,
        "component": spec.component,
        "migration_mode": "retrospective_registry_only",
        "solver_rerun_performed": False,
        "raw_artifacts_moved": False,
        "historical_paths_overwritten": False,
        "canonical_run_labels": canonical_labels,
        "legacy_run_labels": [legacy for legacy, _ in spec.run_labels],
        "run_label_pattern": CANONICAL_LABEL_RE.pattern,
        "attempt_or_rerun_policy": "attemptNN_or_rerunNN",
        "artifact_filename_policy": (
            "<stage_id>_<component>_<attempt_or_rerun>_<artifact_type>"
            "[_<instance>_<seed>].<ext>"
        ),
        "logical_directory_policy": "results/<run_label>/<instance>/<seed>/",
        "trusted_formal_review": spec.trusted_formal_review,
        "trusted_formal_review_status": trusted_status,
        "trusted_formal_review_errors": trusted_errors,
        "comparison_baseline": spec.comparison_baseline,
        "artifact_types": sorted(
            {str(row["artifact_type"]) for row in registry_rows}
        ),
        "instance_scope": sorted(
            {
                str(row["instance"])
                for row in registry_rows
                if str(row["instance"]) != "__all__"
            }
        ),
        "seed_scope": sorted(
            {
                str(row["seed"])
                for row in registry_rows
                if str(row["seed"]) != "__all__"
            }
        ),
        "run_records": run_records,
        "checks": {
            "canonical_label_check": all(
                CANONICAL_LABEL_RE.fullmatch(label) for label in canonical_labels
            ),
            "canonical_run_label_unique": len(set(canonical_labels)) == len(canonical_labels),
            "canonical_artifact_path_check": all(
                canonical_path_is_valid(
                    str(row["canonical_path"]),
                    str(row["run_label"]),
                    str(row["artifact_type"]),
                    str(row["instance"]),
                    str(row["seed"]),
                )
                for row in registry_rows
            ),
            "canonical_artifact_path_unique": len(
                {
                    str(row["canonical_path"]) for row in registry_rows
                }
            )
            == len(registry_rows),
            "artifact_type_check": all(
                bool(str(row["artifact_type"]).strip()) for row in registry_rows
            ),
            "legacy_paths_preserved": all(
                bool(record["legacy_path"]) for record in run_records
            ),
            "raw_manifest_integrity": all(
                record["raw_manifest_status"] == "pass" for record in run_records
            ),
            "formal_raw_manifest_integrity": all(
                record["raw_manifest_status"] == "pass" for record in formal_records
            ),
            "raw_review_manifest_integrity": all(
                record["review_manifest_status"] in {"pass", "not_present"}
                for record in run_records
            ),
            "formal_raw_review_manifest_integrity": all(
                record["review_manifest_status"] == "pass"
                for record in formal_records
            ),
            "trusted_formal_review_integrity": not trusted_errors,
            "published_summary_raw_to_summary": all(
                record["raw_to_summary_status"] in {"pass", "not_published"}
                for record in run_records
            ),
            "formal_raw_to_summary": all(
                record["raw_to_summary_status"] == "pass"
                for record in formal_records
            ),
            "historical_manifest_errors_explicit": all(
                record["raw_manifest_status"] == "pass"
                or bool(record["raw_manifest_errors"])
                for record in run_records
            ),
            "stage00_and_stage02_legacy_mapping": True,
        },
        "notes": [
            (
                "Canonical paths are logical registry paths for historical evidence; "
                "legacy paths remain the source files."
            ),
            (
                "No solver was invoked and no raw, solution, event, failure, "
                "environment, or historical summary was rewritten."
            ),
            (
                "A missing review for an old smoke run is recorded as not_present, "
                "not treated as a passing review."
            ),
            (
                "Tracked summary and review files are registered only when their "
                "semantic raw-to-summary check is pass; missing historical reviews "
                "remain visible as not_published."
            ),
        ],
    }
    return registry_rows, manifest, legacy_map


def build_shared_legacy_map(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    stage00_manifest = root / "experiments/baselines/stage00/manifest.json"
    stage00_environment = read_json(root / "experiments/baselines/stage00/environment.json")
    stage00_manifest_hash = sha256_file(stage00_manifest)
    rows.append(
        {
            "mapping_type": "immutable_stage00",
            "stage_id": "stage00",
            "component": "frozen_baseline",
            "canonical_label": "stage00_frozen_baseline",
            "canonical_path": "experiments/baselines/stage00",
            "legacy_path": "experiments/baselines/stage00",
            "artifact_type": "frozen_baseline",
            "status": "immutable",
            "checksum": stage00_manifest_hash,
            "checksum_source": "manifest.json",
            "repository_revision": stage00_environment.get("repository_revision", ""),
            "repository_dirty": stage00_environment.get("repository_dirty", ""),
            "stage00_manifest_hash": stage00_manifest_hash,
            "reason": "Stage 0 frozen directory is never renamed, moved, or overwritten",
        }
    )
    stage00_root = root / "experiments/baselines/stage00"
    for artifact in sorted(path for path in stage00_root.rglob("*") if path.is_file()):
        relative = artifact.relative_to(stage00_root)
        artifact_type = (
            "solution"
            if relative.parts and relative.parts[0] == "solutions"
            else "manifest"
            if relative.name == "manifest.json"
            else relative.stem
        )
        rows.append(
            {
                "mapping_type": "immutable_stage00_artifact",
                "stage_id": "stage00",
                "component": "frozen_baseline",
                "canonical_label": "stage00_frozen_baseline",
                "canonical_path": f"experiments/baselines/stage00/{relative.as_posix()}",
                "legacy_path": f"experiments/baselines/stage00/{relative.as_posix()}",
                "artifact_type": artifact_type,
                "status": "immutable",
                "checksum": sha256_file(artifact),
                "checksum_source": "stage00_manifest",
                "repository_revision": stage00_environment.get("repository_revision", ""),
                "repository_dirty": stage00_environment.get("repository_dirty", ""),
                "stage00_manifest_hash": stage00_manifest_hash,
                "reason": "Stage 0 frozen artifact is preserved at its historical path",
            }
    )
    for attempt in ("attempt16", "rerun09"):
        legacy = root / (
            "experiments/summaries/"
            f"stage02_constraint_guided_{attempt}_per_run_results.csv"
        )
        environment = read_json(
            root / f"experiments/summaries/stage02_constraint_guided_{attempt}_environment.json"
        )
        rows.append(
            {
                "mapping_type": "stage02_baseline_reference",
                "stage_id": "stage02.3",
                "component": "constraint_guided",
                "canonical_label": f"stage02.3_constraint_guided_{attempt}",
                "canonical_path": (
                    "experiments/summaries/"
                    f"stage02.3_constraint_guided_{attempt}_per_run_results.csv"
                ),
                "legacy_path": root_relative(root, legacy),
                "artifact_type": "per_run_results",
                "status": "historical_legacy_reference",
                "checksum": sha256_file(legacy),
                "checksum_source": "tracked_summary",
                "repository_revision": environment.get("repository_revision", ""),
                "repository_dirty": environment.get("repository_dirty", ""),
                "stage00_manifest_hash": environment.get("baseline_manifest_sha256", ""),
                "reason": (
                    "Stage 3.0/3.1 comparison baseline; old stage02_* path is "
                    "preserved; repository_dirty=true is historical provenance"
                ),
            }
        )
    return rows


def migrate(root: Path, *, write: bool) -> int:
    all_registry_rows: dict[str, list[dict[str, object]]] = {}
    manifests: dict[str, dict[str, object]] = {}
    legacy_map = build_shared_legacy_map(root)
    specs = active_specs(root)
    for spec in specs:
        registry_rows, manifest, stage_legacy_map = build_stage(root, spec)
        all_registry_rows[spec.stage_id] = registry_rows
        manifests[spec.stage_id] = manifest
        legacy_map.extend(stage_legacy_map)

    registry_paths: dict[str, Path] = {}
    manifest_paths: dict[str, Path] = {}
    for spec in specs:
        registry_path = root / "experiments/registries" / (
            f"{spec.stage_id}_artifact_registry.csv"
        )
        manifest_path = root / "experiments/manifests" / (
            f"{spec.stage_id}_{spec.component}_artifact_manifest.json"
        )
        registry_paths[spec.stage_id] = registry_path
        manifest_paths[spec.stage_id] = manifest_path
        manifests[spec.stage_id]["artifact_registry"] = root_relative(root, registry_path)
        manifests[spec.stage_id]["artifact_count"] = len(
            all_registry_rows[spec.stage_id]
        )
        if write:
            write_csv(registry_path, REGISTRY_FIELDS, all_registry_rows[spec.stage_id])
            manifests[spec.stage_id]["artifact_registry_sha256"] = sha256_file(registry_path)
            manifests[spec.stage_id]["artifact_registry_rows_sha256"] = payload_sha256(
                read_csv_rows(registry_path)
            )

    legacy_path = root / "experiments/registries/stage03_legacy_path_map.csv"
    if write:
        write_csv(legacy_path, LEGACY_MAP_FIELDS, legacy_map)
    for spec in specs:
        manifest = manifests[spec.stage_id]
        manifest["legacy_path_map"] = root_relative(root, legacy_path)
        if write:
            manifest["legacy_path_map_sha256"] = sha256_file(legacy_path)
            manifest["checks"]["manifest_recomputable"] = False
            write_json(manifest_paths[spec.stage_id], manifest)
            generated_errors = verify_generated_outputs(
                root,
                manifest,
                registry_paths[spec.stage_id],
                manifest_paths[spec.stage_id],
                legacy_path,
                all_registry_rows[spec.stage_id],
            )
            manifest["generated_output_errors"] = generated_errors
            manifest["checks"]["manifest_recomputable"] = not generated_errors
            write_json(manifest_paths[spec.stage_id], manifest)
        elif manifest_paths[spec.stage_id].is_file():
            generated_errors = verify_generated_outputs(
                root,
                manifest,
                registry_paths[spec.stage_id],
                manifest_paths[spec.stage_id],
                legacy_path,
                all_registry_rows[spec.stage_id],
            )
            manifest["generated_output_errors"] = generated_errors
            manifest["checks"]["manifest_recomputable"] = not generated_errors
        else:
            manifest["generated_output_errors"] = []
            manifest["checks"]["manifest_recomputable"] = None

    hard_failures = {
        spec.stage_id: [
            name
            for name in HARD_CHECKS
            if not bool(manifests[spec.stage_id]["checks"].get(name, False))
        ]
        for spec in specs
    }
    report_status = "fail" if any(hard_failures.values()) else "pass"

    report = {
        "schema_version": SCHEMA_VERSION,
        "write_performed": write,
        "status": report_status,
        "hard_check_failures": hard_failures,
        "stages": {
            spec.stage_id: {
                "artifact_count": len(all_registry_rows[spec.stage_id]),
                "canonical_run_count": len(spec.run_labels),
                "manifest_checks": manifests[spec.stage_id]["checks"],
                "manifest_path": root_relative(root, manifest_paths[spec.stage_id]),
            }
            for spec in specs
        },
        "legacy_path_map_count": len(legacy_map),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if report_status == "fail" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write registries and migration manifests; otherwise verify only",
    )
    arguments = parser.parse_args()
    return migrate(arguments.root.resolve(), write=arguments.write)


if __name__ == "__main__":
    raise SystemExit(main())
