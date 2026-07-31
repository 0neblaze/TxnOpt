"""Deterministic semantic disposition for the Stage 5.2 migration backlog.

The migration ledger is the reviewed dependency boundary: protected labels can
never be reduced here, the twelve explicitly pending benchmark attempts may be
classified as unreferenced superseded metadata, and the unresolved hot-path
attempt remains fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Final, cast

from evrptw.experiment_lifecycle import (
    MIGRATION_SCHEMA_VERSION,
    LifecycleError,
    _load_signed_json,
    _write_signed_json,
)

SCHEMA_VERSION: Final = "experiment-historical-semantic-adjudication-v1"
REVIEWER_MODULE: Final = (
    "evrptw.experiments.stage052_historical_semantic_review"
)
DEPENDENCY_PROOF_SCHEMA: Final = "stage052-historical-no-dependency-proof-v1"
SUPERSEDED_BENCHMARKS: Final = frozenset(
    {
        "stage05.2_benchmark_attempt33",
        "stage05.2_benchmark_attempt37",
        "stage05.2_benchmark_attempt48",
        "stage05.2_benchmark_attempt51",
        "stage05.2_benchmark_attempt60",
        "stage05.2_benchmark_attempt64",
        "stage05.2_benchmark_attempt71",
        "stage05.2_benchmark_attempt78",
        "stage05.2_benchmark_attempt87",
        "stage05.2_benchmark_attempt90",
        "stage05.2_benchmark_attempt92",
        "stage05.2_benchmark_attempt99",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contains_exact_string(value: object, target: str) -> bool:
    if isinstance(value, str):
        return value == target
    if isinstance(value, list):
        return any(_contains_exact_string(item, target) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_exact_string(key, target)
            or _contains_exact_string(item, target)
            for key, item in value.items()
        )
    return False


def _safe_relative_path(raw_path: object) -> PurePosixPath:
    relative = PurePosixPath(str(raw_path))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise LifecycleError("keeper evidence path is unsafe")
    return relative


def _is_dependency_document(relative_path: PurePosixPath) -> bool:
    parts = relative_path.parts
    if parts == ("wsl_active", "formal_memory_probe_report.json"):
        return True
    if parts == ("wsl_active", "campaign_manifest.json"):
        return True
    if (
        len(parts) == 3
        and parts[:2] == ("wsl_active", "control")
        and parts[2].endswith("_manifest.json")
    ):
        return True
    if parts == ("wsl_active", "review", "review_manifest.json"):
        return True
    return (
        len(parts) == 3
        and parts[0] in {"d_benchmark", "wsl_active"}
        and parts[1].startswith("batch")
        and len(parts[1]) == 9
        and parts[1][5:].isdigit()
        and parts[2] == "batch_manifest.json"
    )


def _dependency_proof_payload(
    *,
    migration_ledger_path: Path,
    content_inventory_path: Path,
    run_label: str,
    archive_root: Path,
) -> dict[str, object]:
    migration = _load_signed_json(migration_ledger_path)
    protected = migration.get("protected_run_labels")
    keeper_evidence = migration.get("protected_keeper_evidence")
    legacy_sources = migration.get("legacy_sources")
    if (
        not isinstance(protected, list)
        or not isinstance(keeper_evidence, dict)
        or set(cast(dict[object, object], keeper_evidence))
        != set(str(item) for item in protected)
        or not isinstance(legacy_sources, list)
    ):
        raise LifecycleError("migration ledger lacks trusted keeper evidence")
    registry_sources = [
        cast(dict[str, object], item)
        for item in legacy_sources
        if isinstance(item, dict)
        and item.get("role") == "read_only_primary_legacy"
        and item.get("root_alias") == "e_archive"
        and item.get("schema_version") == "experiment-retention-registry-v2"
    ]
    if len(registry_sources) != 1:
        raise LifecycleError("migration ledger has no unique trusted registry")
    registry_source = registry_sources[0]
    registry_relative = _safe_relative_path(registry_source.get("relative_path"))
    resolved_archive_root = archive_root.resolve(strict=True)
    registry_path = resolved_archive_root.joinpath(*registry_relative.parts)
    expected_registry_sha256 = str(registry_source.get("registry_sha256", ""))
    if _sha256(registry_path) != expected_registry_sha256:
        raise LifecycleError("trusted retention registry hash differs")
    registry = _load_signed_json(registry_path)
    records = registry.get("records")
    if (
        registry.get("schema_version") != "experiment-retention-registry-v2"
        or not isinstance(records, list)
    ):
        raise LifecycleError("trusted retention registry is invalid")

    references: list[dict[str, str]] = []
    keeper_tree_hashes: dict[str, str] = {}
    document_hashes: dict[str, list[dict[str, str]]] = {}
    for keeper in sorted(str(item) for item in protected):
        raw_evidence = cast(dict[object, object], keeper_evidence).get(keeper)
        if not isinstance(raw_evidence, dict):
            raise LifecycleError("keeper evidence entry is invalid")
        matches = [
            cast(dict[str, object], item)
            for item in records
            if isinstance(item, dict) and item.get("run_label") == keeper
        ]
        if not matches:
            raise LifecycleError("protected keeper is absent from registry")
        identity_fields = (
            "archive_relative_path",
            "archive_root_alias",
            "byte_count",
            "file_count",
            "generation",
            "tree_sha256",
            "verification_status",
        )
        record = matches[0]
        if any(
            any(item.get(field) != record.get(field) for field in identity_fields)
            for item in matches[1:]
        ):
            raise LifecycleError("protected keeper registry records disagree")
        if (
            record.get("archive_root_alias") != "e_archive"
            or record.get("verification_status") != "verified"
            or raw_evidence.get("archive_root_alias") != "e_archive"
            or raw_evidence.get("archive_relative_path")
            != record.get("archive_relative_path")
            or raw_evidence.get("archive_tree_sha256") != record.get("tree_sha256")
        ):
            raise LifecycleError("keeper evidence differs from trusted registry")
        generation_relative = _safe_relative_path(record.get("archive_relative_path"))
        generation_dir = resolved_archive_root.joinpath(*generation_relative.parts)
        try:
            resolved_generation_dir = generation_dir.resolve(strict=True)
            resolved_generation_dir.relative_to(resolved_archive_root)
        except (OSError, ValueError) as error:
            raise LifecycleError(
                "protected keeper generation escapes archive root"
            ) from error
        if not resolved_generation_dir.is_dir() or generation_dir.is_symlink():
            raise LifecycleError("protected keeper generation is invalid")
        raw_documents = raw_evidence.get("dependency_documents")
        if not isinstance(raw_documents, list) or not raw_documents:
            raise LifecycleError("keeper dependency document ledger is empty")
        expected_documents: dict[str, str] = {}
        for item in raw_documents:
            if not isinstance(item, dict):
                raise LifecycleError("keeper dependency document is invalid")
            relative = _safe_relative_path(item.get("relative_path"))
            expected_hash = str(item.get("sha256", ""))
            if (
                not _is_dependency_document(relative)
                or len(expected_hash) != 64
                or relative.as_posix() in expected_documents
            ):
                raise LifecycleError("keeper dependency document binding is invalid")
            expected_documents[relative.as_posix()] = expected_hash
        verified_documents: list[dict[str, str]] = []
        for relative_path in sorted(expected_documents):
            relative = PurePosixPath(relative_path)
            path = resolved_generation_dir.joinpath(*relative.parts)
            try:
                resolved_path = path.resolve(strict=True)
                resolved_path.relative_to(resolved_generation_dir)
            except (OSError, ValueError) as error:
                raise LifecycleError(
                    "keeper dependency document escapes generation"
                ) from error
            if path.is_symlink() or not resolved_path.is_file():
                raise LifecycleError("keeper dependency graph contains unsafe entry")
            observed_hash = _sha256(path)
            if observed_hash != expected_documents[relative_path]:
                raise LifecycleError("keeper dependency document hash differs")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise LifecycleError("keeper dependency document is invalid JSON") from error
            if not isinstance(payload, dict):
                raise LifecycleError("keeper dependency document is not an object")
            verified_documents.append(
                {"relative_path": relative_path, "sha256": observed_hash}
            )
            if _contains_exact_string(payload, run_label):
                references.append(
                    {
                        "keeper_run_label": keeper,
                        "relative_path": relative_path,
                    }
                )
        keeper_tree_hashes[keeper] = str(record["tree_sha256"])
        document_hashes[keeper] = verified_documents
    return {
        "schema_version": DEPENDENCY_PROOF_SCHEMA,
        "run_label": run_label,
        "migration_ledger_sha256": _sha256(migration_ledger_path),
        "content_inventory_sha256": _sha256(content_inventory_path),
        "retention_registry_relative_path": registry_relative.as_posix(),
        "retention_registry_sha256": expected_registry_sha256,
        "protected_run_labels": sorted(str(item) for item in protected),
        "keeper_archive_tree_sha256_by_run": keeper_tree_hashes,
        "keeper_dependency_documents_by_run": document_hashes,
        "reference_hits": references,
        "scan_completed": True,
        "producer_module": REVIEWER_MODULE,
    }


def build_no_dependency_proof(
    *,
    migration_ledger_path: Path,
    content_inventory_path: Path,
    run_label: str,
    archive_root: Path,
    output_path: Path,
) -> Path:
    """Scan the ledger-bound canonical keeper dependency closure."""

    payload = _dependency_proof_payload(
        migration_ledger_path=migration_ledger_path,
        content_inventory_path=content_inventory_path,
        run_label=run_label,
        archive_root=archive_root,
    )
    _write_signed_json(output_path, payload)
    return output_path


def review_historical_stage052(
    *,
    migration_ledger_path: Path,
    content_inventory_path: Path,
    run_label: str,
    dependency_proof_path: Path | None = None,
    archive_root: Path | None = None,
) -> dict[str, object]:
    """Apply only the immutable, label-exact migration rules."""

    migration = _load_signed_json(migration_ledger_path)
    inventory = _load_signed_json(content_inventory_path)
    pending = migration.get("pending_historical_classification")
    protected = migration.get("protected_run_labels")
    files = inventory.get("files")
    if (
        migration.get("schema_version") != MIGRATION_SCHEMA_VERSION
        or not isinstance(pending, list)
        or not isinstance(protected, list)
        or run_label not in pending
        or run_label in protected
        or inventory.get("schema_version") != "experiment-content-inventory-v1"
        or inventory.get("run_label") != run_label
        or not isinstance(files, list)
        or not files
    ):
        raise LifecycleError("Stage 5.2 historical semantic inputs are invalid")
    relative_paths = {
        str(cast(dict[str, object], item).get("relative_path", ""))
        for item in files
        if isinstance(item, dict)
    }
    if len(relative_paths) != len(files) or "" in relative_paths:
        raise LifecycleError("Stage 5.2 historical inventory paths are invalid")
    common = {
        "schema_version": SCHEMA_VERSION,
        "run_label": run_label,
        "migration_ledger_sha256": _sha256(migration_ledger_path),
        "content_inventory_sha256": _sha256(content_inventory_path),
        "reviewer_module_name": REVIEWER_MODULE,
    }
    if run_label in SUPERSEDED_BENCHMARKS:
        if not any(
            "batch_manifest" in path
            or "shard_manifest" in path
            or "failure" in path
            for path in relative_paths
        ):
            raise LifecycleError(
                "superseded benchmark has no campaign control evidence"
            )
        if dependency_proof_path is None:
            raise LifecycleError("superseded benchmark lacks dependency proof")
        if archive_root is None:
            raise LifecycleError("superseded benchmark lacks trusted archive root")
        proof = _load_signed_json(dependency_proof_path)
        expected_proof = _dependency_proof_payload(
            migration_ledger_path=migration_ledger_path,
            content_inventory_path=content_inventory_path,
            run_label=run_label,
            archive_root=archive_root,
        )
        if (
            proof != expected_proof
            or expected_proof.get("reference_hits") != []
        ):
            raise LifecycleError("superseded benchmark dependency proof is invalid")
        return {
            **common,
            "status": "PARTIAL",
            "retention_class": "superseded_metadata",
            "no_dependency_proof": True,
            "dependency_proof_sha256": _sha256(dependency_proof_path),
            "dependency_boundary": {
                "migration_schema_version": MIGRATION_SCHEMA_VERSION,
                "protected_run_labels": sorted(str(item) for item in protected),
                "candidate_set": sorted(SUPERSEDED_BENCHMARKS),
            },
            "failure_identity": {},
        }
    return {
        **common,
        "status": "INVALID",
        "retention_class": "unknown_full",
        "no_dependency_proof": False,
        "failure_identity": {
            "component": "hot_path",
            "invariant_or_check": "stage_specific_root_cause",
            "location": run_label,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migration-ledger", type=Path, required=True)
    parser.add_argument("--content-inventory", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependency-proof", type=Path)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--build-dependency-proof", action="store_true")
    arguments = parser.parse_args()
    if arguments.build_dependency_proof:
        if arguments.archive_root is None:
            raise LifecycleError("dependency proof requires canonical archive root")
        output = build_no_dependency_proof(
            migration_ledger_path=arguments.migration_ledger.resolve(strict=True),
            content_inventory_path=arguments.content_inventory.resolve(strict=True),
            run_label=arguments.run_label,
            archive_root=arguments.archive_root.resolve(strict=True),
            output_path=arguments.output.resolve(),
        )
        print(output)
        return 0
    payload = review_historical_stage052(
        migration_ledger_path=arguments.migration_ledger.resolve(strict=True),
        content_inventory_path=arguments.content_inventory.resolve(strict=True),
        run_label=arguments.run_label,
        dependency_proof_path=(
            arguments.dependency_proof.resolve(strict=True)
            if arguments.dependency_proof is not None
            else None
        ),
        archive_root=(
            arguments.archive_root.resolve(strict=True)
            if arguments.archive_root is not None
            else None
        ),
    )
    output = arguments.output.resolve()
    if output.exists():
        raise LifecycleError("historical semantic output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
