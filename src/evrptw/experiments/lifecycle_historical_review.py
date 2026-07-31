"""Read-only historical v1/v2 inventory and fail-closed review adapter.

The adapter verifies one migrated archive generation against the immutable v2
registry and emits a per-file v3 inventory.  It deliberately reports
``INVALID``/``unknown_full`` until a stage-specific semantic reviewer proves a
safe disposition; generating an inventory alone never authorizes deletion.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Final, cast

from evrptw.experiment_lifecycle import (
    HISTORICAL_GATE_SCHEMA_VERSION,
    MIGRATION_SCHEMA_VERSION,
    ExperimentCatalog,
    LifecycleError,
    _load_signed_json,
    _write_signed_json,
    load_lifecycle_migration_ledger,
)

INVENTORY_SCHEMA_VERSION: Final = "experiment-content-inventory-v1"
REVIEWER_MODULE: Final = "evrptw.experiments.lifecycle_historical_review"
SAFE_RETENTION_CLASSES: Final = frozenset(
    {
        "superseded_accepted_capsule",
        "unique_failure_capsule",
        "duplicate_failure_metadata",
        "superseded_metadata",
        "rebuildable",
    }
)


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_snapshot(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_ino,
        stat.st_dev,
    )


def _inventory_file(
    task: tuple[Path, Path],
) -> dict[str, object]:
    generation_dir, path = task
    before = _file_snapshot(path)
    digest = _sha256(path)
    if _file_snapshot(path) != before:
        raise LifecycleError(
            "historical generation changed during inventory: "
            f"{path.relative_to(generation_dir).as_posix()}"
        )
    return {
        "relative_path": path.relative_to(generation_dir).as_posix(),
        "byte_count": before[0],
        "sha256": digest,
        "modified_time_ns": before[1],
    }


def _execution_command_prefix(payload: Mapping[str, object]) -> list[object]:
    command = payload.get("command")
    return command[:3] if isinstance(command, list) else []


def _parse_semantic_command_options(command: tuple[str, ...]) -> dict[str, str]:
    tail = command[3:]
    if len(tail) % 2:
        raise LifecycleError("historical semantic reviewer options are invalid")
    options: dict[str, str] = {}
    allowed = {
        "--migration-ledger",
        "--content-inventory",
        "--run-label",
        "--output",
        "--dependency-proof",
        "--archive-root",
    }
    for index in range(0, len(tail), 2):
        option = tail[index]
        value = tail[index + 1]
        if option not in allowed or option in options or not value:
            raise LifecycleError("historical semantic reviewer options are invalid")
        options[option] = value
    required = allowed - {"--dependency-proof"}
    if set(options) - {"--dependency-proof"} != required:
        raise LifecycleError("historical semantic reviewer options are incomplete")
    return options


def _load_v2_record(
    *, registry_path: Path, run_label: str
) -> Mapping[str, object]:
    registry = _load_signed_json(registry_path)
    records = registry.get("records")
    if (
        registry.get("schema_version") != "experiment-retention-registry-v2"
        or not isinstance(records, list)
    ):
        raise LifecycleError("historical v2 registry is invalid")
    matches = [
        cast(dict[str, object], item)
        for item in records
        if isinstance(item, dict) and item.get("run_label") == run_label
    ]
    if not matches:
        raise LifecycleError("historical run is absent from the v2 registry")
    identity_fields = (
        "archive_relative_path",
        "archive_root_alias",
        "byte_count",
        "file_count",
        "generation",
        "tree_sha256",
        "verification_status",
    )
    first = matches[0]
    if any(
        any(item.get(field) != first.get(field) for field in identity_fields)
        for item in matches[1:]
    ):
        raise LifecycleError("historical v2 segment records disagree")
    if (
        first.get("archive_root_alias") != "e_archive"
        or first.get("verification_status") != "verified"
        or first.get("retention_class") != "unknown_full"
    ):
        raise LifecycleError("historical v2 record is not the expected full archive")
    return first


def _inventory_generation(
    *, run_label: str, generation_dir: Path
) -> tuple[dict[str, object], str]:
    paths: list[Path] = []
    tree_digest = hashlib.sha256()
    for path in sorted(generation_dir.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise LifecycleError("historical generation contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise LifecycleError("historical generation contains an unsupported entry")
        paths.append(path)
    workers = min(32, max(1, os.cpu_count() or 1), max(1, len(paths)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        files = list(
            executor.map(
                _inventory_file,
                ((generation_dir, path) for path in paths),
            )
        )
    for item in files:
        tree_digest.update(
            _canonical_json(
                {
                    "relative_path": item["relative_path"],
                    "byte_count": item["byte_count"],
                    "sha256": item["sha256"],
                }
            )
        )
    inventory = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "run_label": run_label,
        "file_count": len(files),
        "byte_count": sum(cast(int, item["byte_count"]) for item in files),
        "tree_sha256": tree_digest.hexdigest(),
        "scan_passes": 1,
        "hash_backend": "python_thread_pool",
        "hash_workers": workers,
        "files": files,
    }
    return inventory, tree_digest.hexdigest()


def review_historical_generation(
    *,
    repository: Path,
    archive_root: Path,
    migration_ledger_path: Path,
    registry_path: Path,
    output_root: Path,
    run_label: str,
) -> tuple[Path, Path]:
    migration = load_lifecycle_migration_ledger(
        migration_ledger_path,
        repository=repository,
    )
    from evrptw.stage052_campaign import StorageRootLocator
    from evrptw.stage052_campaign_runner import probe_volume_identity

    canonical_locator_path = (
        repository.resolve(strict=True)
        / "configs"
        / "stage052_storage_roots.local.toml"
    ).resolve(strict=True)
    locator = StorageRootLocator.from_toml(canonical_locator_path)
    locator.verify_all(probe_volume_identity, ("e_archive",))
    configured_archive_root = locator.resolve("e_archive").absolute_path.resolve(
        strict=True
    )
    resolved_archive_root = archive_root.resolve(strict=True)
    if resolved_archive_root != configured_archive_root:
        raise LifecycleError("historical archive root is not canonical e_archive")
    legacy_sources = migration.get("legacy_sources")
    if not isinstance(legacy_sources, list):
        raise LifecycleError("migration ledger has no trusted legacy sources")
    registry_sources = [
        cast(dict[str, object], item)
        for item in legacy_sources
        if isinstance(item, dict)
        and item.get("role") == "read_only_primary_legacy"
        and item.get("root_alias") == "e_archive"
        and item.get("schema_version") == "experiment-retention-registry-v2"
    ]
    if len(registry_sources) != 1:
        raise LifecycleError("migration ledger has no unique trusted v2 registry")
    registry_source = registry_sources[0]
    registry_relative = PurePosixPath(str(registry_source.get("relative_path", "")))
    if (
        registry_relative.is_absolute()
        or not registry_relative.parts
        or ".." in registry_relative.parts
    ):
        raise LifecycleError("trusted v2 registry path is unsafe")
    expected_registry_path = configured_archive_root.joinpath(
        *registry_relative.parts
    ).resolve(strict=True)
    try:
        expected_registry_path.relative_to(configured_archive_root)
    except ValueError as error:
        raise LifecycleError("trusted v2 registry escapes e_archive") from error
    resolved_registry_path = registry_path.resolve(strict=True)
    expected_registry_sha256 = str(registry_source.get("registry_sha256", ""))
    if (
        resolved_registry_path != expected_registry_path
        or _sha256(resolved_registry_path) != expected_registry_sha256
    ):
        raise LifecycleError("historical registry is not migration-ledger trusted")
    pending = migration["pending_historical_classification"]
    if run_label not in cast(list[str], pending):
        raise LifecycleError("run is not pending historical classification")
    record = _load_v2_record(
        registry_path=resolved_registry_path, run_label=run_label
    )
    relative = PurePosixPath(str(record["archive_relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise LifecycleError("historical archive path is unsafe")
    generation_dir = resolved_archive_root.joinpath(*relative.parts)
    try:
        resolved_generation_dir = generation_dir.resolve(strict=True)
        resolved_generation_dir.relative_to(resolved_archive_root)
    except (OSError, ValueError) as error:
        raise LifecycleError("historical archive generation escapes root") from error
    if generation_dir.is_symlink() or not resolved_generation_dir.is_dir():
        raise LifecycleError("historical archive generation is missing")
    inventory, tree_sha256 = _inventory_generation(
        run_label=run_label,
        generation_dir=resolved_generation_dir,
    )
    if (
        inventory["file_count"] != record["file_count"]
        or inventory["byte_count"] != record["byte_count"]
        or tree_sha256 != record["tree_sha256"]
    ):
        raise LifecycleError(
            "historical archive differs from the v2 registry: "
            f"expected=(files={record['file_count']},bytes={record['byte_count']},"
            f"tree={record['tree_sha256']}), "
            f"observed=(files={inventory['file_count']},"
            f"bytes={inventory['byte_count']},tree={tree_sha256})"
        )
    inventory.update(
        {
            "archive_root_alias": "e_archive",
            "archive_root_resolved_path": str(resolved_archive_root),
            "archive_generation_relative_path": relative.as_posix(),
            "legacy_registry_sha256": expected_registry_sha256,
            "legacy_tree_sha256": tree_sha256,
        }
    )
    evidence_dir = output_root.resolve() / "evidence" / run_label
    inventory_path = evidence_dir / "content_inventory.json"
    inventory_sha256 = _write_signed_json(inventory_path, inventory)
    review = {
        "schema_version": HISTORICAL_GATE_SCHEMA_VERSION,
        "run_label": run_label,
        "status": "INVALID",
        "retention_class": "unknown_full",
        "migration_schema_version": MIGRATION_SCHEMA_VERSION,
        "migration_ledger_sha256": _sha256(migration_ledger_path),
        "legacy_registry_sha256": expected_registry_sha256,
        "legacy_tree_sha256": tree_sha256,
        "archive_root_alias": "e_archive",
        "archive_root_resolved_path": str(resolved_archive_root),
        "archive_generation_relative_path": relative.as_posix(),
        "content_inventory_sha256": inventory_sha256,
        "reviewer_module_name": REVIEWER_MODULE,
        "failure_identity": {
            "component": "historical_migration",
            "invariant_or_check": "stage_specific_semantic_disposition",
            "location": str(resolved_generation_dir),
        },
    }
    review_path = evidence_dir / "review_manifest.json"
    _write_signed_json(review_path, review)
    return inventory_path, review_path


def apply_semantic_adjudication(
    *,
    catalog_path: Path,
    migration_ledger_path: Path,
    output_root: Path,
    run_label: str,
    semantic_review_path: Path,
) -> Path:
    """Bind a stage-specific signed semantic review to the physical inventory."""

    resolved_output_root = output_root.resolve()
    evidence_dir = resolved_output_root / "evidence" / run_label
    inventory_path = evidence_dir / "content_inventory.json"
    inventory = _load_signed_json(inventory_path)
    semantic = _load_signed_json(semantic_review_path)
    inventory_sha256 = _sha256(inventory_path)
    migration_sha256 = _sha256(migration_ledger_path)
    execution_path = evidence_dir / "semantic_review_execution.json"
    try:
        semantic_relative_path = semantic_review_path.resolve().relative_to(evidence_dir)
    except ValueError as error:
        raise LifecycleError(
            "historical semantic review output is outside its evidence directory"
        ) from error
    execution = _load_signed_json(execution_path)
    execution_binding_path = (
        resolved_output_root / "executions" / f"{run_label}.json"
    )
    execution_binding = _load_signed_json(execution_binding_path)
    retention_class = str(semantic.get("retention_class", ""))
    status = str(semantic.get("status", ""))
    reviewer_module = str(semantic.get("reviewer_module_name", ""))
    catalog = ExperimentCatalog.from_toml(catalog_path)
    module_spec = importlib.util.find_spec(reviewer_module)
    module_path = (
        Path(module_spec.origin).resolve(strict=True)
        if module_spec is not None and module_spec.origin is not None
        else None
    )
    raw_command = execution.get("command")
    if not isinstance(raw_command, list) or not all(
        isinstance(item, str) for item in raw_command
    ):
        raise LifecycleError("historical semantic execution command is invalid")
    command_options = _parse_semantic_command_options(tuple(raw_command))
    expected_command_options = {
        "--migration-ledger": str(migration_ledger_path.resolve(strict=True)),
        "--content-inventory": str(inventory_path.resolve(strict=True)),
        "--run-label": run_label,
        "--output": str(semantic_review_path.resolve(strict=True)),
        "--archive-root": str(execution.get("archive_root_resolved_path", "")),
    }
    if command_options.get("--dependency-proof") is not None:
        dependency_proof = Path(
            command_options["--dependency-proof"]
        ).resolve(strict=True)
        try:
            dependency_proof.relative_to(evidence_dir)
        except ValueError as error:
            raise LifecycleError(
                "historical dependency proof is outside evidence directory"
            ) from error
        expected_command_options["--dependency-proof"] = str(dependency_proof)
    if (
        semantic.get("schema_version")
        != "experiment-historical-semantic-adjudication-v1"
        or semantic.get("run_label") != run_label
        or semantic.get("migration_ledger_sha256") != migration_sha256
        or semantic.get("content_inventory_sha256") != inventory_sha256
        or retention_class not in SAFE_RETENTION_CLASSES
        or status not in {"FAILED_KNOWN", "PARTIAL", "INVALID"}
        or reviewer_module not in catalog.historical_semantic_reviewer_modules
        or execution.get("run_label") != run_label
        or execution.get("status") != "completed"
        or execution.get("exit_code") != 0
        or execution.get("reviewer_module_name") != reviewer_module
        or module_path is None
        or execution.get("reviewer_module_sha256") != _sha256(module_path)
        or execution.get("semantic_review_sha256") != _sha256(semantic_review_path)
        or execution.get("semantic_review_relative_path")
        != semantic_relative_path.as_posix()
        or execution.get("migration_ledger_sha256_before") != migration_sha256
        or execution.get("migration_ledger_sha256_after") != migration_sha256
        or execution.get("content_inventory_sha256_before") != inventory_sha256
        or execution.get("content_inventory_sha256_after") != inventory_sha256
        or execution.get("archive_root_alias") != "e_archive"
        or not str(execution.get("archive_root_resolved_path", ""))
        or not str(execution.get("archive_generation_relative_path", ""))
        or inventory.get("archive_root_alias") != "e_archive"
        or inventory.get("archive_root_resolved_path")
        != execution.get("archive_root_resolved_path")
        or inventory.get("archive_generation_relative_path")
        != execution.get("archive_generation_relative_path")
        or inventory.get("legacy_registry_sha256")
        != execution.get("legacy_registry_sha256")
        or inventory.get("legacy_tree_sha256")
        != execution.get("legacy_tree_sha256")
        or command_options != expected_command_options
        or _execution_command_prefix(execution)
        != [sys.executable, "-m", reviewer_module]
        or execution_binding.get("run_label") != run_label
        or execution_binding.get("review_execution_sha256")
        != _sha256(execution_path)
        or execution_binding.get("semantic_review_sha256")
        != _sha256(semantic_review_path)
        or execution_binding.get("migration_ledger_sha256") != migration_sha256
        or execution_binding.get("content_inventory_sha256") != inventory_sha256
        or execution_binding.get("archive_root_alias") != "e_archive"
        or execution_binding.get("archive_root_resolved_path")
        != execution.get("archive_root_resolved_path")
        or execution_binding.get("archive_generation_relative_path")
        != execution.get("archive_generation_relative_path")
        or execution_binding.get("legacy_registry_sha256")
        != execution.get("legacy_registry_sha256")
        or execution_binding.get("legacy_tree_sha256")
        != execution.get("legacy_tree_sha256")
    ):
        raise LifecycleError("historical semantic adjudication is not controlled")
    failure_identity = semantic.get("failure_identity")
    if retention_class in {
        "unique_failure_capsule",
        "duplicate_failure_metadata",
    }:
        if not isinstance(failure_identity, dict) or not all(
            str(failure_identity.get(field, ""))
            for field in (
                "failure_code",
                "component",
                "invariant_or_check",
                "location",
            )
        ):
            raise LifecycleError("historical failure disposition lacks root-cause identity")
        if (
            retention_class == "duplicate_failure_metadata"
            and not semantic.get("canonical_representative")
        ):
            raise LifecycleError("duplicate historical failure has no representative")
    supersession_proof = semantic.get("supersession_proof")
    if retention_class == "superseded_accepted_capsule":
        required_hashes = (
            "accepted_review_manifest_sha256",
            "accepted_raw_manifest_sha256",
            "terminal_review_manifest_sha256",
            "terminal_review_execution_sha256",
            "successor_archive_tree_sha256",
            "successor_raw_manifest_sha256",
            "successor_review_manifest_sha256",
            "successor_review_execution_sha256",
        )
        if (
            not isinstance(supersession_proof, dict)
            or not str(supersession_proof.get("failure_code", ""))
            or not str(supersession_proof.get("successor_run_label", ""))
            or any(
                not isinstance(supersession_proof.get(field), str)
                or len(str(supersession_proof[field])) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(supersession_proof[field])
                )
                for field in required_hashes
            )
        ):
            raise LifecycleError(
                "historical superseded accepted disposition lacks proof"
            )
    if (
        retention_class == "superseded_metadata"
        and semantic.get("no_dependency_proof") is not True
    ):
        raise LifecycleError("superseded historical evidence lacks dependency proof")
    if retention_class == "rebuildable" and not semantic.get("rebuild_proof_sha256"):
        raise LifecycleError("historical rebuildable disposition lacks rebuild proof")
    review = {
        "schema_version": HISTORICAL_GATE_SCHEMA_VERSION,
        "run_label": run_label,
        "status": status,
        "retention_class": retention_class,
        "migration_schema_version": MIGRATION_SCHEMA_VERSION,
        "migration_ledger_sha256": migration_sha256,
        "content_inventory_sha256": inventory_sha256,
        "reviewer_module_name": REVIEWER_MODULE,
        "semantic_reviewer_module_name": reviewer_module,
        "semantic_review_sha256": _sha256(semantic_review_path),
        "semantic_review_execution_sha256": _sha256(execution_path),
        "archive_root_alias": execution["archive_root_alias"],
        "archive_root_resolved_path": execution["archive_root_resolved_path"],
        "archive_generation_relative_path": execution[
            "archive_generation_relative_path"
        ],
        "legacy_registry_sha256": execution["legacy_registry_sha256"],
        "legacy_tree_sha256": execution["legacy_tree_sha256"],
        "failure_identity": failure_identity or {},
        "canonical_representative": semantic.get("canonical_representative", ""),
        "supersession_proof": supersession_proof or {},
    }
    review_path = evidence_dir / "review_manifest.json"
    _write_signed_json(review_path, review)
    return review_path


def execute_semantic_reviewer(
    *,
    catalog_path: Path,
    migration_ledger_path: Path,
    output_root: Path,
    archive_root: Path,
    run_label: str,
    semantic_review_path: Path,
    command: tuple[str, ...],
) -> Path:
    """Execute the declared stage-specific reviewer and bind immutable inputs."""

    evidence_dir = output_root.resolve() / "evidence" / run_label
    try:
        semantic_relative_path = semantic_review_path.resolve().relative_to(evidence_dir)
    except ValueError as error:
        raise LifecycleError(
            "historical semantic review output is outside its evidence directory"
        ) from error
    inventory_path = evidence_dir / "content_inventory.json"
    inventory = _load_signed_json(inventory_path)
    physical_review_path = evidence_dir / "review_manifest.json"
    physical_review = _load_signed_json(physical_review_path)
    inventory_sha256 = _sha256(inventory_path)
    migration_sha256 = _sha256(migration_ledger_path)
    if len(command) < 3 or Path(command[0]).resolve() != Path(sys.executable).resolve():
        raise LifecycleError("historical semantic reviewer command is not controlled")
    reviewer_module = command[2] if command[1] == "-m" else ""
    catalog = ExperimentCatalog.from_toml(catalog_path)
    if reviewer_module not in catalog.historical_semantic_reviewer_modules:
        raise LifecycleError("historical semantic reviewer module is invalid")
    resolved_archive_root = archive_root.resolve(strict=True)
    if (
        physical_review.get("run_label") != run_label
        or physical_review.get("archive_root_alias") != "e_archive"
        or physical_review.get("archive_root_resolved_path")
        != str(resolved_archive_root)
        or inventory.get("archive_root_alias") != "e_archive"
        or inventory.get("archive_root_resolved_path") != str(resolved_archive_root)
        or inventory.get("archive_generation_relative_path")
        != physical_review.get("archive_generation_relative_path")
        or inventory.get("legacy_registry_sha256")
        != physical_review.get("legacy_registry_sha256")
        or inventory.get("legacy_tree_sha256")
        != physical_review.get("legacy_tree_sha256")
    ):
        raise LifecycleError("historical semantic archive root is not canonical")
    command_options = _parse_semantic_command_options(command)
    expected_options = {
        "--migration-ledger": str(migration_ledger_path.resolve(strict=True)),
        "--content-inventory": str(inventory_path.resolve(strict=True)),
        "--run-label": run_label,
        "--output": str(semantic_review_path.resolve()),
        "--archive-root": str(resolved_archive_root),
    }
    if command_options.get("--dependency-proof") is not None:
        proof_path = Path(command_options["--dependency-proof"]).resolve(strict=True)
        try:
            proof_path.relative_to(evidence_dir)
        except ValueError as error:
            raise LifecycleError(
                "historical dependency proof is outside evidence directory"
            ) from error
        expected_options["--dependency-proof"] = str(proof_path)
    if command_options != expected_options:
        raise LifecycleError("historical semantic reviewer inputs are not canonical")
    module_spec = importlib.util.find_spec(reviewer_module)
    if module_spec is None or module_spec.origin is None:
        raise LifecycleError("historical semantic reviewer cannot be resolved")
    module_path = Path(module_spec.origin).resolve(strict=True)
    module_sha256 = _sha256(module_path)
    execution_path = evidence_dir / "semantic_review_execution.json"
    if semantic_review_path.exists() or execution_path.exists():
        raise LifecycleError("historical semantic reviewer output already exists")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise LifecycleError(
            f"historical semantic reviewer exited with {completed.returncode}"
        )
    try:
        payload = json.loads(semantic_review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LifecycleError("historical semantic reviewer output is invalid") from error
    if not isinstance(payload, dict):
        raise LifecycleError("historical semantic reviewer output is not an object")
    _write_signed_json(semantic_review_path, payload)
    if (
        _sha256(module_path) != module_sha256
        or _sha256(migration_ledger_path) != migration_sha256
        or _sha256(inventory_path) != inventory_sha256
    ):
        raise LifecycleError("historical semantic reviewer inputs changed")
    execution_sha256 = _write_signed_json(
        execution_path,
        {
            "schema_version": "experiment-historical-review-execution-v1",
            "run_label": run_label,
            "status": "completed",
            "exit_code": 0,
            "reviewer_module_name": reviewer_module,
            "reviewer_module_sha256": module_sha256,
            "semantic_review_sha256": _sha256(semantic_review_path),
            "semantic_review_relative_path": semantic_relative_path.as_posix(),
            "migration_ledger_sha256_before": migration_sha256,
            "migration_ledger_sha256_after": migration_sha256,
            "content_inventory_sha256_before": inventory_sha256,
            "content_inventory_sha256_after": inventory_sha256,
            "archive_root_alias": "e_archive",
            "archive_root_resolved_path": str(resolved_archive_root),
            "archive_generation_relative_path": physical_review[
                "archive_generation_relative_path"
            ],
            "legacy_registry_sha256": physical_review[
                "legacy_registry_sha256"
            ],
            "legacy_tree_sha256": physical_review["legacy_tree_sha256"],
            "command": list(command),
        },
    )
    _write_signed_json(
        output_root.resolve() / "executions" / f"{run_label}.json",
        {
            "schema_version": "experiment-historical-review-execution-binding-v1",
            "run_label": run_label,
            "review_execution_sha256": execution_sha256,
            "semantic_review_sha256": _sha256(semantic_review_path),
            "migration_ledger_sha256": migration_sha256,
            "content_inventory_sha256": inventory_sha256,
            "archive_root_alias": "e_archive",
            "archive_root_resolved_path": str(resolved_archive_root),
            "archive_generation_relative_path": physical_review[
                "archive_generation_relative_path"
            ],
            "legacy_registry_sha256": physical_review[
                "legacy_registry_sha256"
            ],
            "legacy_tree_sha256": physical_review["legacy_tree_sha256"],
        },
    )
    return execution_path


def aggregate_historical_gate(
    *,
    repository: Path,
    catalog_path: Path,
    migration_ledger_path: Path,
    output_root: Path,
) -> Path:
    """Publish the gate only when every pending historical run is safe."""

    resolved_output_root = output_root.resolve()
    migration = load_lifecycle_migration_ledger(
        migration_ledger_path,
        repository=repository,
    )
    migration_sha256 = _sha256(migration_ledger_path)
    catalog = ExperimentCatalog.from_toml(catalog_path)
    records: list[dict[str, object]] = []
    unresolved: list[str] = []
    for label in cast(list[str], migration["pending_historical_classification"]):
        evidence_dir = resolved_output_root / "evidence" / label
        review_path = evidence_dir / "review_manifest.json"
        inventory_path = evidence_dir / "content_inventory.json"
        execution_path = evidence_dir / "semantic_review_execution.json"
        execution_binding_path = (
            resolved_output_root / "executions" / f"{label}.json"
        )
        try:
            review = _load_signed_json(review_path)
            inventory = _load_signed_json(inventory_path)
            execution = _load_signed_json(execution_path)
            execution_binding = _load_signed_json(execution_binding_path)
        except LifecycleError:
            unresolved.append(label)
            continue
        retention_class = str(review.get("retention_class", ""))
        semantic_reviewer = str(review.get("semantic_reviewer_module_name", ""))
        module_spec = importlib.util.find_spec(semantic_reviewer)
        module_path = (
            Path(module_spec.origin).resolve(strict=True)
            if module_spec is not None and module_spec.origin is not None
            else None
        )
        semantic_relative = PurePosixPath(
            str(execution.get("semantic_review_relative_path", ""))
        )
        semantic_path = evidence_dir.joinpath(*semantic_relative.parts)
        try:
            semantic = _load_signed_json(semantic_path)
        except LifecycleError:
            unresolved.append(label)
            continue
        raw_command = execution.get("command")
        try:
            if not isinstance(raw_command, list) or not all(
                isinstance(item, str) for item in raw_command
            ):
                raise LifecycleError("historical semantic command is invalid")
            command_options = _parse_semantic_command_options(tuple(raw_command))
            expected_command_options = {
                "--migration-ledger": str(migration_ledger_path.resolve(strict=True)),
                "--content-inventory": str(inventory_path.resolve(strict=True)),
                "--run-label": label,
                "--output": str(semantic_path.resolve(strict=True)),
                "--archive-root": str(review["archive_root_resolved_path"]),
            }
            if command_options.get("--dependency-proof") is not None:
                dependency_proof = Path(
                    command_options["--dependency-proof"]
                ).resolve(strict=True)
                dependency_proof.relative_to(evidence_dir)
                expected_command_options["--dependency-proof"] = str(
                    dependency_proof
                )
        except (KeyError, LifecycleError, OSError, ValueError):
            unresolved.append(label)
            continue
        if (
            review.get("run_label") != label
            or review.get("migration_ledger_sha256") != migration_sha256
            or review.get("content_inventory_sha256") != _sha256(inventory_path)
            or review.get("reviewer_module_name") != REVIEWER_MODULE
            or review.get("semantic_reviewer_module_name")
            not in catalog.historical_semantic_reviewer_modules
            or retention_class not in SAFE_RETENTION_CLASSES
            or review.get("semantic_review_execution_sha256")
            != _sha256(execution_path)
            or execution.get("run_label") != label
            or execution.get("status") != "completed"
            or execution.get("exit_code") != 0
            or execution.get("reviewer_module_name") != semantic_reviewer
            or module_path is None
            or execution.get("reviewer_module_sha256") != _sha256(module_path)
            or execution.get("semantic_review_sha256")
            != review.get("semantic_review_sha256")
            or semantic_relative.is_absolute()
            or not semantic_relative.parts
            or ".." in semantic_relative.parts
            or semantic.get("run_label") != label
            or semantic.get("schema_version")
            != "experiment-historical-semantic-adjudication-v1"
            or semantic.get("reviewer_module_name") != semantic_reviewer
            or semantic.get("migration_ledger_sha256") != migration_sha256
            or semantic.get("content_inventory_sha256")
            != _sha256(inventory_path)
            or _sha256(semantic_path) != review.get("semantic_review_sha256")
            or execution.get("migration_ledger_sha256_before") != migration_sha256
            or execution.get("migration_ledger_sha256_after") != migration_sha256
            or execution.get("content_inventory_sha256_before")
            != _sha256(inventory_path)
            or execution.get("content_inventory_sha256_after")
            != _sha256(inventory_path)
            or execution.get("archive_root_alias") != "e_archive"
            or execution.get("archive_root_resolved_path")
            != review.get("archive_root_resolved_path")
            or execution.get("archive_generation_relative_path")
            != review.get("archive_generation_relative_path")
            or inventory.get("archive_root_alias") != "e_archive"
            or inventory.get("archive_root_resolved_path")
            != review.get("archive_root_resolved_path")
            or inventory.get("archive_generation_relative_path")
            != review.get("archive_generation_relative_path")
            or inventory.get("legacy_registry_sha256")
            != review.get("legacy_registry_sha256")
            or inventory.get("legacy_tree_sha256")
            != review.get("legacy_tree_sha256")
            or execution.get("legacy_registry_sha256")
            != review.get("legacy_registry_sha256")
            or execution.get("legacy_tree_sha256")
            != review.get("legacy_tree_sha256")
            or command_options != expected_command_options
            or _execution_command_prefix(execution)
            != [sys.executable, "-m", semantic_reviewer]
            or execution_binding.get("run_label") != label
            or execution_binding.get("review_execution_sha256")
            != _sha256(execution_path)
            or execution_binding.get("semantic_review_sha256")
            != review.get("semantic_review_sha256")
            or execution_binding.get("migration_ledger_sha256")
            != migration_sha256
            or execution_binding.get("content_inventory_sha256")
            != _sha256(inventory_path)
            or execution_binding.get("archive_root_alias") != "e_archive"
            or execution_binding.get("archive_root_resolved_path")
            != review.get("archive_root_resolved_path")
            or execution_binding.get("archive_generation_relative_path")
            != review.get("archive_generation_relative_path")
            or execution_binding.get("legacy_registry_sha256")
            != review.get("legacy_registry_sha256")
            or execution_binding.get("legacy_tree_sha256")
            != review.get("legacy_tree_sha256")
            or not isinstance(inventory.get("files"), list)
            or not cast(list[object], inventory["files"])
        ):
            unresolved.append(label)
            continue
        records.append(
            {
                "run_label": label,
                "review_status": review["status"],
                "retention_class": retention_class,
                "review_manifest_relative_path": review_path.relative_to(
                    resolved_output_root
                ).as_posix(),
                "review_manifest_sha256": _sha256(review_path),
                "content_inventory_relative_path": inventory_path.relative_to(
                    resolved_output_root
                ).as_posix(),
                "content_inventory_sha256": _sha256(inventory_path),
                "archive_root_alias": review["archive_root_alias"],
                "archive_root_resolved_path": review[
                    "archive_root_resolved_path"
                ],
                "archive_generation_relative_path": review[
                    "archive_generation_relative_path"
                ],
                "legacy_registry_sha256": review["legacy_registry_sha256"],
                "legacy_tree_sha256": review["legacy_tree_sha256"],
            }
        )
    if unresolved:
        raise LifecycleError(
            "historical migration remains BLOCKED_RETENTION: "
            + ", ".join(sorted(unresolved))
        )
    gate_path = resolved_output_root / "gate.json"
    _write_signed_json(
        gate_path,
        {
            "schema_version": HISTORICAL_GATE_SCHEMA_VERSION,
            "migration_ledger_sha256": migration_sha256,
            "status": "complete",
            "records": records,
        },
    )
    return gate_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--migration-ledger", type=Path, required=True)
    parser.add_argument("--legacy-registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-label")
    parser.add_argument("--semantic-review", type=Path)
    parser.add_argument("--semantic-command", nargs=argparse.REMAINDER)
    parser.add_argument("--aggregate-gate", action="store_true")
    arguments = parser.parse_args()
    if arguments.aggregate_gate:
        gate = aggregate_historical_gate(
            repository=arguments.repository.resolve(),
            catalog_path=arguments.catalog.resolve(),
            migration_ledger_path=arguments.migration_ledger.resolve(),
            output_root=arguments.output_root.resolve(),
        )
        print(json.dumps({"gate": str(gate)}))
        return 0
    if not arguments.run_label:
        parser.error("--run-label is required unless --aggregate-gate is used")
    if arguments.semantic_review is not None:
        semantic_command = tuple(arguments.semantic_command or ())
        if semantic_command[:1] == ("--",):
            semantic_command = semantic_command[1:]
        execute_semantic_reviewer(
            catalog_path=arguments.catalog.resolve(),
            migration_ledger_path=arguments.migration_ledger.resolve(),
            output_root=arguments.output_root.resolve(),
            archive_root=arguments.archive_root.resolve(),
            run_label=arguments.run_label,
            semantic_review_path=arguments.semantic_review.resolve(),
            command=semantic_command,
        )
        review = apply_semantic_adjudication(
            catalog_path=arguments.catalog.resolve(),
            migration_ledger_path=arguments.migration_ledger.resolve(),
            output_root=arguments.output_root.resolve(),
            run_label=arguments.run_label,
            semantic_review_path=arguments.semantic_review.resolve(),
        )
        print(json.dumps({"review": str(review)}))
        return 0
    inventory, review = review_historical_generation(
        repository=arguments.repository.resolve(),
        archive_root=arguments.archive_root.resolve(),
        migration_ledger_path=arguments.migration_ledger.resolve(),
        registry_path=arguments.legacy_registry.resolve(),
        output_root=arguments.output_root.resolve(),
        run_label=arguments.run_label,
    )
    print(json.dumps({"inventory": str(inventory), "review": str(review)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
