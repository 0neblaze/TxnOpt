"""Deterministic semantic disposition for the Stage 5.2 migration backlog.

The migration ledger is the reviewed dependency boundary: protected labels can
never be reduced here, the twelve explicitly pending benchmark attempts may be
classified as unreferenced superseded metadata, and the historical hot-path
attempt may be reduced only after its accepted-review lineage and accepted
successor are replayed from the canonical archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
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
HOT_PATH_ATTEMPT: Final = "stage05.2_hot_path_attempt04"
HOT_PATH_SUCCESSOR: Final = "stage05.2_hot_path_attempt06"
HOT_PATH_READY_STATUS: Final = "READY_FOR_STAGE052_ARTIFACT_STREAMING"
HOT_PATH_RETRY_FAILURE_GATES: Final = frozenset(
    {"source_snapshot", "prerequisite_performance_baseline"}
)
HOT_PATH_GATE_NAMES: Final = frozenset(
    {
        "exact_scope",
        "optimization_profile",
        "performance_promotion",
        "persistence_attribution",
        "prerequisite_performance_baseline",
        "replay_consistency",
        "runtime_identity",
        "source_snapshot",
        "staging_root_identity",
    }
)
_SHA256: Final = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LifecycleError(f"cannot read historical JSON: {path}") from error
    if not isinstance(payload, dict):
        raise LifecycleError(f"historical JSON is not an object: {path}")
    return cast(dict[str, object], payload)


def _inventory_index(
    inventory: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw_files = inventory.get("files")
    if not isinstance(raw_files, list):
        raise LifecycleError("historical inventory files are invalid")
    output: dict[str, Mapping[str, object]] = {}
    for raw_item in raw_files:
        if not isinstance(raw_item, Mapping):
            raise LifecycleError("historical inventory entry is invalid")
        relative = str(raw_item.get("relative_path", ""))
        if not relative or relative in output:
            raise LifecycleError("historical inventory paths are invalid")
        output[relative] = raw_item
    return output


def _inventory_sha256(
    index: Mapping[str, Mapping[str, object]], relative_path: str
) -> str:
    item = index.get(relative_path)
    digest = "" if item is None else str(item.get("sha256", ""))
    if _SHA256.fullmatch(digest) is None:
        raise LifecycleError(
            f"historical inventory lacks a valid digest for {relative_path}"
        )
    return digest


def _validate_review_files_from_inventory(
    *,
    manifest: Mapping[str, object],
    inventory: Mapping[str, Mapping[str, object]],
    review_root_relative: PurePosixPath,
) -> None:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, Mapping) or not raw_files:
        raise LifecycleError("historical review files are invalid")
    for raw_relative, raw_digest in raw_files.items():
        relative = _safe_relative_path(raw_relative)
        digest = str(raw_digest)
        if _SHA256.fullmatch(digest) is None:
            raise LifecycleError("historical review file digest is invalid")
        inventory_relative = (review_root_relative / relative).as_posix()
        if _inventory_sha256(inventory, inventory_relative) != digest:
            raise LifecycleError("historical review files differ from inventory")


def _failed_gate_names(manifest: Mapping[str, object]) -> frozenset[str]:
    raw_gates = manifest.get("gates")
    if (
        not isinstance(raw_gates, Mapping)
        or set(str(name) for name in raw_gates) != HOT_PATH_GATE_NAMES
    ):
        raise LifecycleError("historical review gates are invalid")
    failed: set[str] = set()
    for raw_name, raw_gate in raw_gates.items():
        if not isinstance(raw_gate, Mapping) or not isinstance(
            raw_gate.get("passed"), bool
        ):
            raise LifecycleError("historical review gate is invalid")
        if raw_gate["passed"] is not True:
            failed.add(str(raw_name))
    return frozenset(failed)


def _validate_raw_manifest_binding(
    *,
    generation_dir: Path,
    inventory: Mapping[str, Mapping[str, object]],
    segment: PurePosixPath,
    run_label: str,
) -> tuple[str, dict[str, object]]:
    relative = segment / "control" / f"{run_label}_manifest.json"
    sidecar_relative = segment / "control" / f"{run_label}_manifest.sha256"
    manifest_path = generation_dir.joinpath(*relative.parts)
    sidecar_path = generation_dir.joinpath(*sidecar_relative.parts)
    digest = _sha256(manifest_path)
    if (
        _inventory_sha256(inventory, relative.as_posix()) != digest
        or sidecar_path.read_text(encoding="utf-8").strip() != digest
        or _inventory_sha256(inventory, sidecar_relative.as_posix())
        != _sha256(sidecar_path)
    ):
        raise LifecycleError("historical raw manifest binding differs")
    payload = _read_json_object(manifest_path)
    if payload.get("run_label") != run_label or payload.get("component") != "hot_path":
        raise LifecycleError("historical raw manifest identity differs")
    return digest, payload


def _trusted_registry(
    *,
    migration: Mapping[str, object],
    archive_root: Path,
) -> tuple[Path, dict[str, object]]:
    raw_sources = migration.get("legacy_sources")
    if not isinstance(raw_sources, list):
        raise LifecycleError("migration ledger lacks legacy sources")
    sources = [
        cast(dict[str, object], item)
        for item in raw_sources
        if isinstance(item, dict)
        and item.get("role") == "read_only_primary_legacy"
        and item.get("root_alias") == "e_archive"
        and item.get("schema_version") == "experiment-retention-registry-v2"
    ]
    if len(sources) != 1:
        raise LifecycleError("migration ledger has no unique trusted registry")
    source = sources[0]
    relative = _safe_relative_path(source.get("relative_path"))
    resolved_root = archive_root.resolve(strict=True)
    registry_path = resolved_root.joinpath(*relative.parts).resolve(strict=True)
    try:
        registry_path.relative_to(resolved_root)
    except ValueError as error:
        raise LifecycleError("trusted retention registry escapes archive root") from error
    if _sha256(registry_path) != source.get("registry_sha256"):
        raise LifecycleError("trusted retention registry hash differs")
    registry = _load_signed_json(registry_path)
    if registry.get("schema_version") != "experiment-retention-registry-v2":
        raise LifecycleError("trusted retention registry is invalid")
    return registry_path, registry


def _trusted_run_record(
    registry: Mapping[str, object], run_label: str
) -> Mapping[str, object]:
    raw_records = registry.get("records")
    if not isinstance(raw_records, list):
        raise LifecycleError("trusted retention registry records are invalid")
    records = [
        cast(dict[str, object], item)
        for item in raw_records
        if isinstance(item, dict) and item.get("run_label") == run_label
    ]
    if not records:
        raise LifecycleError(f"trusted retention record is missing: {run_label}")
    identity_fields = (
        "archive_relative_path",
        "archive_root_alias",
        "byte_count",
        "file_count",
        "generation",
        "tree_sha256",
        "verification_status",
    )
    first = records[0]
    if any(
        any(record.get(field) != first.get(field) for field in identity_fields)
        for record in records[1:]
    ):
        raise LifecycleError("trusted retention records disagree")
    if (
        first.get("archive_root_alias") != "e_archive"
        or first.get("verification_status") != "verified"
        or _SHA256.fullmatch(str(first.get("tree_sha256", ""))) is None
    ):
        raise LifecycleError("trusted retention record is not verified")
    return first


def _generation_dir(archive_root: Path, raw_relative: object) -> Path:
    relative = _safe_relative_path(raw_relative)
    resolved_root = archive_root.resolve(strict=True)
    generation = resolved_root.joinpath(*relative.parts).resolve(strict=True)
    try:
        generation.relative_to(resolved_root)
    except ValueError as error:
        raise LifecycleError("historical generation escapes archive root") from error
    if not generation.is_dir() or generation.is_symlink():
        raise LifecycleError("historical generation is invalid")
    return generation


def _review_manifest(
    *,
    generation_dir: Path,
    inventory: Mapping[str, Mapping[str, object]],
    relative: PurePosixPath,
    run_label: str,
    raw_manifest_sha256: str,
    expected_status: str,
) -> tuple[str, dict[str, object], frozenset[str]]:
    path = generation_dir.joinpath(*relative.parts)
    digest = _sha256(path)
    if _inventory_sha256(inventory, relative.as_posix()) != digest:
        raise LifecycleError("historical review manifest differs from inventory")
    payload = _read_json_object(path)
    if (
        payload.get("schema_version") != "stage05.2-review-v1"
        or payload.get("run_label") != run_label
        or payload.get("component") != "hot_path"
        or payload.get("scope") != "performance"
        or payload.get("status") != expected_status
        or payload.get("raw_manifest_sha256") != raw_manifest_sha256
    ):
        raise LifecycleError("historical review identity differs")
    _validate_review_files_from_inventory(
        manifest=payload,
        inventory=inventory,
        review_root_relative=relative.parent,
    )
    return digest, payload, _failed_gate_names(payload)


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


def _hot_path_supersession_proof(
    *,
    migration: Mapping[str, object],
    inventory: Mapping[str, object],
    archive_root: Path,
) -> dict[str, object]:
    """Replay the immutable accepted lineage and its accepted successor."""

    from evrptw.experiments.lifecycle_historical_review import (
        _inventory_generation,
    )

    registry_path, registry = _trusted_registry(
        migration=migration,
        archive_root=archive_root,
    )
    current_record = _trusted_run_record(registry, HOT_PATH_ATTEMPT)
    current_generation = _generation_dir(
        archive_root, current_record.get("archive_relative_path")
    )
    if (
        inventory.get("archive_root_alias") != "e_archive"
        or Path(str(inventory.get("archive_root_resolved_path", ""))).resolve()
        != archive_root.resolve(strict=True)
        or inventory.get("archive_generation_relative_path")
        != current_record.get("archive_relative_path")
        or inventory.get("legacy_registry_sha256") != _sha256(registry_path)
        or inventory.get("legacy_tree_sha256") != current_record.get("tree_sha256")
        or inventory.get("file_count") != current_record.get("file_count")
        or inventory.get("byte_count") != current_record.get("byte_count")
    ):
        raise LifecycleError("hot-path inventory differs from trusted registry")
    current_index = _inventory_index(inventory)
    raw_candidates = [
        PurePosixPath(relative)
        for relative in current_index
        if relative.endswith(
            f"/control/{HOT_PATH_ATTEMPT}_manifest.json"
        )
    ]
    if len(raw_candidates) != 1 or len(raw_candidates[0].parts) != 3:
        raise LifecycleError("hot-path raw manifest location is ambiguous")
    current_segment = PurePosixPath(raw_candidates[0].parts[0])
    current_raw_sha256, _ = _validate_raw_manifest_binding(
        generation_dir=current_generation,
        inventory=current_index,
        segment=current_segment,
        run_label=HOT_PATH_ATTEMPT,
    )
    review_root = current_segment / "review"
    current_review_relative = review_root / "review_manifest.json"
    (
        current_review_sha256,
        current_review,
        current_failed_gates,
    ) = _review_manifest(
        generation_dir=current_generation,
        inventory=current_index,
        relative=current_review_relative,
        run_label=HOT_PATH_ATTEMPT,
        raw_manifest_sha256=current_raw_sha256,
        expected_status="NOT_READY",
    )
    if current_failed_gates != HOT_PATH_RETRY_FAILURE_GATES:
        raise LifecycleError("hot-path terminal retry failure identity differs")
    current_gates = cast(Mapping[str, Mapping[str, object]], current_review["gates"])
    if (
        current_gates["source_snapshot"].get("detail")
        != "raw source snapshot identity does not match independent live replay"
        or current_gates["prerequisite_performance_baseline"].get("detail")
        != "producer metadata is not bound to the reviewed prerequisite role"
    ):
        raise LifecycleError("hot-path terminal retry failure detail differs")
    raw_lineage = current_review.get("review_manifest_lineage_sha256")
    raw_retry_history = current_review.get("review_retry_history_sha256")
    if (
        not isinstance(raw_lineage, list)
        or len(raw_lineage) != 1
        or _SHA256.fullmatch(str(raw_lineage[0])) is None
        or not isinstance(raw_retry_history, list)
        or not raw_retry_history
        or any(_SHA256.fullmatch(str(item)) is None for item in raw_retry_history)
        or len(set(str(item) for item in raw_retry_history))
        != len(raw_retry_history)
    ):
        raise LifecycleError("hot-path review lineage is invalid")
    retry_hashes = [str(item) for item in raw_retry_history]
    accepted_review_sha256 = str(raw_lineage[0])
    accepted_relative = (
        review_root
        / "history"
        / accepted_review_sha256
        / "review_manifest.json"
    )
    accepted_digest, accepted_review, accepted_failed = _review_manifest(
        generation_dir=current_generation,
        inventory=current_index,
        relative=accepted_relative,
        run_label=HOT_PATH_ATTEMPT,
        raw_manifest_sha256=current_raw_sha256,
        expected_status=HOT_PATH_READY_STATUS,
    )
    if accepted_digest != accepted_review_sha256 or accepted_failed:
        raise LifecycleError("hot-path accepted review lineage is invalid")
    if (
        accepted_review.get("review_manifest_lineage_sha256") != []
        or accepted_review.get("review_retry_history_sha256") != retry_hashes
    ):
        raise LifecycleError("hot-path accepted predecessor lineage is invalid")
    for index, retry_sha256 in enumerate(retry_hashes):
        retry_relative = (
            review_root / "history" / retry_sha256 / "review_manifest.json"
        )
        retry_digest, retry_review, retry_failed = _review_manifest(
            generation_dir=current_generation,
            inventory=current_index,
            relative=retry_relative,
            run_label=HOT_PATH_ATTEMPT,
            raw_manifest_sha256=current_raw_sha256,
            expected_status="NOT_READY",
        )
        if retry_digest != retry_sha256 or not retry_failed:
            raise LifecycleError("hot-path failed-review history is invalid")
        if (
            retry_review.get("review_manifest_lineage_sha256") != []
            or retry_review.get("review_retry_history_sha256", [])
            != retry_hashes[:index]
        ):
            raise LifecycleError("hot-path failed-review lineage is invalid")
    observed_history = {
        PurePosixPath(relative).parts[-2]
        for relative in current_index
        if re.fullmatch(
            rf"{re.escape(review_root.as_posix())}/history/[0-9a-f]{{64}}/review_manifest\.json",
            relative,
        )
    }
    if observed_history != {accepted_review_sha256, *retry_hashes}:
        raise LifecycleError("hot-path review history set differs")
    execution_relative = review_root / "review_execution.json"
    execution_path = current_generation.joinpath(*execution_relative.parts)
    execution_sha256 = _sha256(execution_path)
    if _inventory_sha256(current_index, execution_relative.as_posix()) != execution_sha256:
        raise LifecycleError("hot-path review execution differs from inventory")
    execution = _read_json_object(execution_path)
    if (
        execution.get("schema_version") != "stage05.2-review-execution-v1"
        or execution.get("run_label") != HOT_PATH_ATTEMPT
        or execution.get("status") != "completed"
        or execution.get("exit_code") != 0
        or execution.get("finalized") is not True
        or execution.get("reviewer_module_name")
        != "evrptw.experiments.stage052_performance_review"
        or execution.get("raw_manifest_sha256_before") != current_raw_sha256
        or execution.get("raw_manifest_sha256_after") != current_raw_sha256
        or execution.get("raw_manifest_unchanged") is not True
        or execution.get("review_manifest_sha256") != current_review_sha256
    ):
        raise LifecycleError("hot-path review execution binding differs")

    successor_record = _trusted_run_record(registry, HOT_PATH_SUCCESSOR)
    successor_generation = _generation_dir(
        archive_root, successor_record.get("archive_relative_path")
    )
    successor_inventory, successor_tree_sha256 = _inventory_generation(
        run_label=HOT_PATH_SUCCESSOR,
        generation_dir=successor_generation,
    )
    if (
        successor_inventory.get("file_count") != successor_record.get("file_count")
        or successor_inventory.get("byte_count") != successor_record.get("byte_count")
        or successor_tree_sha256 != successor_record.get("tree_sha256")
    ):
        raise LifecycleError("hot-path successor differs from trusted registry")
    successor_index = _inventory_index(successor_inventory)
    successor_raw_candidates = [
        PurePosixPath(relative)
        for relative in successor_index
        if relative.endswith(
            f"/control/{HOT_PATH_SUCCESSOR}_manifest.json"
        )
    ]
    if len(successor_raw_candidates) != 1 or len(successor_raw_candidates[0].parts) != 3:
        raise LifecycleError("hot-path successor raw manifest location is ambiguous")
    successor_segment = PurePosixPath(successor_raw_candidates[0].parts[0])
    successor_raw_sha256, _ = _validate_raw_manifest_binding(
        generation_dir=successor_generation,
        inventory=successor_index,
        segment=successor_segment,
        run_label=HOT_PATH_SUCCESSOR,
    )
    successor_review_relative = successor_segment / "review" / "review_manifest.json"
    (
        successor_review_sha256,
        _,
        successor_failed,
    ) = _review_manifest(
        generation_dir=successor_generation,
        inventory=successor_index,
        relative=successor_review_relative,
        run_label=HOT_PATH_SUCCESSOR,
        raw_manifest_sha256=successor_raw_sha256,
        expected_status=HOT_PATH_READY_STATUS,
    )
    if successor_failed:
        raise LifecycleError("hot-path successor review is not fully accepted")
    successor_execution_relative = (
        successor_segment / "review" / "review_execution.json"
    )
    successor_execution_path = successor_generation.joinpath(
        *successor_execution_relative.parts
    )
    successor_execution_sha256 = _sha256(successor_execution_path)
    if (
        _inventory_sha256(
            successor_index, successor_execution_relative.as_posix()
        )
        != successor_execution_sha256
    ):
        raise LifecycleError("hot-path successor execution differs from inventory")
    successor_execution = _read_json_object(successor_execution_path)
    if (
        successor_execution.get("schema_version")
        != "stage05.2-review-execution-v1"
        or successor_execution.get("run_label") != HOT_PATH_SUCCESSOR
        or successor_execution.get("status") != "completed"
        or successor_execution.get("exit_code") != 0
        or successor_execution.get("finalized") is not True
        or successor_execution.get("reviewer_module_name")
        != "evrptw.experiments.stage052_performance_review"
        or successor_execution.get("raw_manifest_sha256_before")
        != successor_raw_sha256
        or successor_execution.get("raw_manifest_sha256_after")
        != successor_raw_sha256
        or successor_execution.get("raw_manifest_unchanged") is not True
        or successor_execution.get("review_manifest_sha256")
        != successor_review_sha256
    ):
        raise LifecycleError("hot-path successor execution binding differs")
    return {
        "failure_code": "accepted_review_superseded_after_live_dependency_drift",
        "accepted_review_manifest_sha256": accepted_review_sha256,
        "accepted_raw_manifest_sha256": current_raw_sha256,
        "terminal_review_manifest_sha256": current_review_sha256,
        "terminal_review_execution_sha256": execution_sha256,
        "terminal_failed_gates": sorted(current_failed_gates),
        "failed_review_retry_manifest_sha256": retry_hashes,
        "successor_run_label": HOT_PATH_SUCCESSOR,
        "successor_archive_relative_path": successor_record[
            "archive_relative_path"
        ],
        "successor_archive_tree_sha256": successor_tree_sha256,
        "successor_raw_manifest_sha256": successor_raw_sha256,
        "successor_review_manifest_sha256": successor_review_sha256,
        "successor_review_execution_sha256": successor_execution_sha256,
    }


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
    if run_label == HOT_PATH_ATTEMPT and archive_root is not None:
        proof = _hot_path_supersession_proof(
            migration=migration,
            inventory=inventory,
            archive_root=archive_root,
        )
        return {
            **common,
            "status": "PARTIAL",
            "retention_class": "superseded_accepted_capsule",
            "supersession_proof": proof,
            "failure_identity": {
                "failure_code": proof["failure_code"],
                "component": "hot_path",
                "invariant_or_check": "current_review_live_dependency_binding",
                "location": run_label,
            },
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
    _write_signed_json(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
