#!/usr/bin/env python3
"""Atomically publish an accepted Stage 5.2 Formal campaign review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from evrptw.artifacts import signed_sidecar_matches
from evrptw.candidate_transaction import NativeCandidateTransactionConfig
from evrptw.native_kernels import NATIVE_KERNEL_ABI_VERSION

PUBLICATION_SCHEMA: Final = "stage05.2-performance-benchmark-publication-v1"
REVIEW_SCHEMA: Final = "stage05.2-campaign-review-v2"
SUPPORTED_REVIEW_SCHEMAS: Final = frozenset({"stage05.2-campaign-review-v1", REVIEW_SCHEMA})
FORMAL_READY: Final = "READY_FOR_STAGE05_3"
_TARGET_NAMES: Final[dict[str, str]] = {
    "per_run_results": "per_run_results.csv",
    "family_summary": "family_summary.csv",
    "budget_summary": "budget_summary.csv",
    "anytime_summary": "anytime_summary.csv",
    "resource_summary": "resource_summary.csv",
    "persistence_summary": "persistence_summary.csv",
    "performance_gates": "performance_gates.csv",
    "gpu_decision": "gpu_decision.json",
    "failure_analysis": "failure_analysis.csv",
    "review_findings": "review_findings.csv",
    "review_report": "review_report.md",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_fsync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_fsync(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_review_manifest(
    path: Path,
    *,
    expected_scope: str,
    expected_status: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read campaign review manifest: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("campaign review manifest must be an object")
    if payload.get("schema_version") not in SUPPORTED_REVIEW_SCHEMAS:
        raise ValueError("campaign review schema_version is not publishable")
    expected = {
        "component": "benchmark",
        "scope": expected_scope,
        "status": expected_status,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"campaign review {field} is not publishable")
    run_label = payload.get("run_label")
    if not isinstance(run_label, str) or not run_label.startswith("stage05.2_benchmark_"):
        raise ValueError("campaign review run_label is not canonical")
    raw_hash = payload.get("raw_campaign_manifest_sha256")
    if (
        not _is_sha256(raw_hash)
        or not _is_sha256(payload.get("raw_manifest_sha256"))
        or not _is_sha256(payload.get("campaign_prerequisite_review_sha256"))
    ):
        raise ValueError("campaign review raw/campaign manifest hash is invalid")
    selection = payload.get("selection_lock")
    native = payload.get("native_configuration")
    candidate_transaction = payload.get("candidate_transaction_configuration")
    accelerator_decision = payload.get("accelerator_decision")
    is_current_schema = payload.get("schema_version") == REVIEW_SCHEMA
    allowed_producer_workers = {6} if is_current_schema else {2, 4}
    expected_backend = {
        "GPU_NOT_JUSTIFIED": "native_cpu",
        "NATIVE_CPU_RETAINED": "native_cpu",
        "ACCELERATOR_PROMOTED": "cuda",
    }.get(accelerator_decision if isinstance(accelerator_decision, str) else "")
    if (
        not isinstance(selection, Mapping)
        or not isinstance(native, Mapping)
        or not isinstance(candidate_transaction, Mapping)
        or payload.get("native_kernel_config") != native
        or payload.get("candidate_transaction_config") != candidate_transaction
        or dict(candidate_transaction) != NativeCandidateTransactionConfig().to_dict()
        or expected_backend is None
        or payload.get("selected_backend") != expected_backend
        or payload.get("selected_exact_backend") != "cpu_batch"
        or payload.get("selected_workers") not in allowed_producer_workers
        or payload.get("native_profile") != NATIVE_KERNEL_ABI_VERSION
        or selection.get("selected_backend") != payload.get("selected_backend")
        or selection.get("selected_exact_backend") != payload.get("selected_exact_backend")
        or selection.get("selected_workers") != payload.get("selected_workers")
        or selection.get("native_profile") != payload.get("native_profile")
        or selection.get("accelerator_decision") != payload.get("accelerator_decision")
        or selection.get("native_kernel_config") != native
        or selection.get("candidate_transaction_config") != candidate_transaction
        or selection.get("candidate_transaction_config_sha256")
        != hashlib.sha256(
            json.dumps(
                candidate_transaction,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        or selection.get("native_config_sha256")
        != hashlib.sha256(
            json.dumps(native, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        or not _is_sha256(selection.get("accelerator_review_manifest_sha256"))
    ):
        raise ValueError("campaign review execution selection lock is invalid")
    if is_current_schema:
        storage_identity = payload.get("storage_publication_identity")
        storage_batches = (
            storage_identity.get("batches") if isinstance(storage_identity, Mapping) else None
        )
        storage_digest = (
            hashlib.sha256(
                json.dumps(
                    storage_identity,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if isinstance(storage_identity, Mapping)
            else None
        )
        replay_metrics = payload.get("review_shard_metrics")
        if (
            not isinstance(storage_identity, Mapping)
            or storage_identity.get("schema_version") != "stage05.2-storage-publication-identity-v1"
            or storage_identity.get("run_label") != run_label
            or not isinstance(storage_batches, list)
            or not storage_batches
            or any(
                not isinstance(batch, Mapping)
                or set(batch)
                != {
                    "root_alias",
                    "relative_path",
                    "file_count",
                    "byte_count",
                    "tree_sha256",
                }
                or not isinstance(batch.get("root_alias"), str)
                or not isinstance(batch.get("relative_path"), str)
                or not isinstance(batch.get("file_count"), int)
                or batch.get("file_count", 0) <= 0
                or not isinstance(batch.get("byte_count"), int)
                or batch.get("byte_count", 0) <= 0
                or not _is_sha256(batch.get("tree_sha256"))
                for batch in storage_batches
            )
            or payload.get("storage_publication_identity_sha256") != storage_digest
            or not isinstance(replay_metrics, list)
            or not replay_metrics
            or any(
                not isinstance(metric, Mapping)
                or metric.get("replay_backend") != "native_arrow"
                or metric.get("native_fallback_count") != 0
                or metric.get("review_workers") not in {1, 2, 4}
                or not isinstance(metric.get("canonical_merge_ordinal"), int)
                or not isinstance(metric.get("logical_events"), int)
                or metric.get("logical_events", 0) <= 0
                for metric in replay_metrics
            )
        ):
            raise ValueError("campaign review storage/replay observability contract is invalid")
    from evrptw.stage052_evidence import (
        CAMPAIGN_PILOT_GATES,
        verify_stage052_campaign_gate_set,
    )

    if expected_scope == "pilot" and isinstance(payload.get("gates"), Mapping):
        gates = payload["gates"]
        assert isinstance(gates, Mapping)
        provisional = set(gates) == set(CAMPAIGN_PILOT_GATES).difference(
            {"publication_dry_run"}
        ) and all(
            isinstance(gate, Mapping) and gate.get("passed") is True for gate in gates.values()
        )
    else:
        provisional = False
    if not provisional:
        verify_stage052_campaign_gate_set(payload, scope=expected_scope)
    return payload


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verified_sources(
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> dict[str, Path]:
    files = manifest.get("publication_files")
    if not isinstance(files, Mapping) or set(files) != set(_TARGET_NAMES):
        raise ValueError("campaign review publication file set is incomplete")
    review_dir = manifest_path.parent.resolve()
    verified: dict[str, Path] = {}
    for key in _TARGET_NAMES:
        item = files.get(key)
        if not isinstance(item, Mapping):
            raise ValueError(f"publication source is invalid: {key}")
        relative = Path(str(item.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"publication source escapes review directory: {key}")
        source = (review_dir / relative).resolve()
        if review_dir not in source.parents or not source.is_file():
            raise ValueError(f"publication source is missing: {key}")
        checksum = item.get("sha256")
        if not _is_sha256(checksum) or _sha256(source) != checksum:
            raise ValueError(f"publication source checksum mismatch: {key}")
        verified[key] = source
    return verified


def _verify_live_formal_chain(
    *,
    review_manifest: Path,
    review: Mapping[str, object],
    prerequisite_dir: Path,
    repository_root: Path,
    active_results_root: Path | None = None,
) -> None:
    """Re-open the current G02 raw bundle and current accepted G01 prerequisite."""

    from evrptw.artifacts import ArtifactReader
    from evrptw.stage052_campaign import load_campaign_manifest
    from evrptw.stage052_campaign_runner import load_benchmark_execution_lock
    from evrptw.stage052_retention import (
        load_retention_registry,
        path_uses_symlink,
    )

    raw_candidate = review_manifest.parent.parent
    configured_results = (
        active_results_root
        if active_results_root is not None
        else repository_root.resolve() / "results"
    )
    if (
        path_uses_symlink(configured_results)
        or path_uses_symlink(raw_candidate)
        or path_uses_symlink(prerequisite_dir)
        or path_uses_symlink(review_manifest)
    ):
        raise ValueError("publisher raw/prerequisite directories are not canonical live roots")
    raw_dir = raw_candidate.resolve()
    canonical_results = configured_results.resolve()
    prerequisite_dir = prerequisite_dir.resolve()
    if (
        not canonical_results.is_dir()
        or raw_dir.parent != canonical_results
        or prerequisite_dir.parent != canonical_results
    ):
        raise ValueError("publisher raw/prerequisite directories are not canonical live roots")
    retention_registry = (
        repository_root.resolve()
        / "experiments"
        / "registries"
        / "stage05.2_retention_registry.csv"
    )
    if retention_registry.is_file():
        try:
            retained_labels = {
                record.run_label for record in load_retention_registry(retention_registry)
            }
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError("publisher retention registry is invalid") from error
        if raw_dir.name in retained_labels or prerequisite_dir.name in retained_labels:
            raise ValueError("publisher cannot use a run recorded in the retention registry")
    expected_pointer = raw_dir / "review" / "review_manifest.json"
    if review_manifest.resolve() != expected_pointer:
        raise ValueError("publisher requires the current raw review manifest pointer")
    reader = ArtifactReader(raw_dir)
    campaign_path = raw_dir / "campaign_manifest.json"
    campaign = load_campaign_manifest(campaign_path)
    if (
        campaign.run_label != raw_dir.name
        or campaign.scope != "formal"
        or campaign.status != "complete"
        or review.get("run_label") != campaign.run_label
        or review.get("raw_manifest_sha256") != _sha256(reader.result.manifest_path)
        or review.get("raw_campaign_manifest_sha256") != _sha256(campaign_path)
    ):
        raise ValueError("campaign review no longer binds the live Formal raw evidence")
    attribution = raw_dir / "control" / f"{campaign.run_label}_persistence_attribution.json"
    attribution_sidecar = attribution.with_suffix(".sha256")
    if (
        not attribution.is_file()
        or not attribution_sidecar.is_file()
        or not signed_sidecar_matches(attribution, attribution_sidecar)
        or review.get("persistence_attribution_sha256") != _sha256(attribution)
        or review.get("persistence_attribution_sidecar_sha256") != _sha256(attribution_sidecar)
    ):
        raise ValueError("campaign persistence attribution is stale")
    g01_review_dir = prerequisite_dir / "review"
    g01_review = g01_review_dir / "review_manifest.json"
    if g01_review_dir.is_symlink() or g01_review.is_symlink() or g01_review.resolve() != g01_review:
        raise ValueError("publisher raw/prerequisite directories are not canonical live roots")
    if (
        not g01_review.is_file()
        or campaign.prerequisite_review_sha256 != _sha256(g01_review)
        or review.get("campaign_prerequisite_review_sha256") != campaign.prerequisite_review_sha256
    ):
        raise ValueError("Formal campaign no longer binds the current G01 review")
    lock = load_benchmark_execution_lock(
        prerequisite_dir.resolve(),
        expected_scope="pilot",
        expected_status="READY_FOR_STAGE052_FORMAL_BENCHMARK",
    )
    if (
        campaign.selected_backend != lock.selected_backend
        or campaign.selected_exact_backend != lock.selected_exact_backend
        or campaign.selected_workers != lock.selected_workers
        or review.get("selected_backend") != lock.selected_backend
        or review.get("selected_exact_backend") != lock.selected_exact_backend
        or review.get("selected_workers") != lock.selected_workers
    ):
        raise ValueError("Formal campaign execution differs from the current G01 lock")


def _verify_gpu_decision(
    path: Path,
    *,
    review: Mapping[str, object],
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("campaign GPU decision publication is unreadable") from error
    selection = review.get("selection_lock")
    if not isinstance(payload, Mapping) or not isinstance(selection, Mapping):
        raise ValueError("campaign GPU decision publication is invalid")
    expected = {
        "schema_version": "stage05.2-gpu-decision-publication-v1",
        "run_label": review.get("run_label"),
        "decision": review.get("accelerator_decision"),
        "selected_backend": review.get("selected_backend"),
        "selected_exact_backend": review.get("selected_exact_backend"),
        "selected_workers": review.get("selected_workers"),
        "native_profile": review.get("native_profile"),
        "native_config_sha256": selection.get("native_config_sha256"),
        "accelerator_review_manifest_sha256": selection.get("accelerator_review_manifest_sha256"),
        "campaign_prerequisite_review_sha256": review.get("campaign_prerequisite_review_sha256"),
    }
    if any(payload.get(field) != value for field, value in expected.items()):
        raise ValueError("campaign GPU decision/backend/workers/native/F-review lock differs")


def verify_published_stage052_review(path: Path) -> dict[str, object]:
    """Verify one tracked, versioned review and its 11 adjacent summaries."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("published campaign review is unreadable") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "stage05.2-published-campaign-review-v1"
    ):
        raise ValueError("published campaign review schema is invalid")
    publication = payload.get("publication_files")
    files = payload.get("files")
    generation = payload.get("publication_generation")
    if (
        not isinstance(publication, Mapping)
        or set(publication) != set(_TARGET_NAMES)
        or not isinstance(files, Mapping)
        or not _is_sha256(generation)
    ):
        raise ValueError("published campaign review file identity is incomplete")
    observed: dict[str, str] = {}
    for key, filename in _TARGET_NAMES.items():
        item = publication.get(key)
        if not isinstance(item, Mapping) or set(item) != {"relative_path", "sha256"}:
            raise ValueError("published campaign review file entry is invalid")
        relative = Path(str(item["relative_path"]))
        expected_name = f"{payload.get('run_label')}_{generation}_{filename}"
        if relative.parts != (expected_name,):
            raise ValueError("published campaign review key/filename mapping is invalid")
        source = path.parent / relative
        checksum = item.get("sha256")
        if not _is_sha256(checksum) or not source.is_file() or _sha256(source) != checksum:
            raise ValueError("published campaign review checksum mismatch")
        observed[relative.as_posix()] = str(checksum)
    if {str(key): str(value) for key, value in files.items()} != observed:
        raise ValueError("published campaign review files/publication_files differ")
    return payload


def _registry_bytes(
    *,
    run_label: str,
    status: str,
    files: Mapping[str, str],
    review_manifest_sha256: str,
    publication_generation: str,
) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "run_label",
            "artifact_type",
            "relative_path",
            "sha256",
            "status",
            "review_manifest_sha256",
            "publication_generation",
            "trusted_manifest_sha256",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    for relative, checksum in sorted(files.items()):
        writer.writerow(
            {
                "run_label": run_label,
                "artifact_type": Path(relative).stem.removeprefix(f"{run_label}_"),
                "relative_path": relative,
                "sha256": checksum,
                "status": status,
                "review_manifest_sha256": review_manifest_sha256,
                "publication_generation": publication_generation,
                "trusted_manifest_sha256": "",
            }
        )
    return buffer.getvalue().encode("utf-8")


def verify_canonical_registry_against_trusted(
    *,
    registry_path: Path,
    trusted_manifest_path: Path,
) -> None:
    """Rebuild the exact registry rows from the current trusted publication."""

    try:
        trusted = json.loads(trusted_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("trusted publication manifest is unreadable") from error
    if not isinstance(trusted, Mapping):
        raise ValueError("trusted publication manifest is invalid")
    repository = trusted_manifest_path.resolve().parents[2]
    registry_relative = trusted.get("artifact_registry")
    published_review_relative = trusted.get("published_review_manifest")
    files = trusted.get("files")
    expected_trusted_fields = {
        "schema_version",
        "run_label",
        "status",
        "generation_id",
        "raw_manifest_sha256",
        "raw_campaign_manifest_sha256",
        "selected_backend",
        "selected_exact_backend",
        "selected_workers",
        "native_profile",
        "native_config_sha256",
        "accelerator_decision",
        "accelerator_review_manifest_sha256",
        "campaign_prerequisite_review_sha256",
        "source_review_manifest_sha256",
        "published_review_manifest",
        "artifact_registry",
        "files",
    }
    if (
        set(trusted) != expected_trusted_fields
        or trusted.get("schema_version") != PUBLICATION_SCHEMA
        or trusted.get("status") not in {FORMAL_READY, "READY_FOR_STAGE052_FORMAL_BENCHMARK"}
        or not isinstance(trusted.get("run_label"), str)
        or not _is_sha256(trusted.get("generation_id"))
        or not _is_sha256(trusted.get("source_review_manifest_sha256"))
        or not isinstance(registry_relative, str)
        or not isinstance(published_review_relative, str)
        or not isinstance(files, Mapping)
    ):
        raise ValueError("trusted publication identity is invalid")
    normalized_files = {str(path): str(checksum) for path, checksum in files.items()}
    if (
        registry_relative not in normalized_files
        or published_review_relative not in normalized_files
        or any(not _is_sha256(checksum) for checksum in normalized_files.values())
    ):
        raise ValueError("trusted publication file mapping is invalid")
    for relative, checksum in normalized_files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("trusted publication file path is invalid")
        physical = repository / path
        if not physical.is_file() or _sha256(physical) != checksum:
            raise ValueError("trusted publication file checksum mismatch")
    pointed_registry = repository / registry_relative
    if _sha256(registry_path) != _sha256(pointed_registry):
        raise ValueError("canonical registry differs from the trusted registry")
    published_review_path = repository / published_review_relative
    published_review = verify_published_stage052_review(published_review_path)
    published_selection = published_review.get("selection_lock")
    if not isinstance(published_selection, Mapping):
        raise ValueError("published review selection lock is invalid")
    trusted_review_fields = {
        "run_label": published_review.get("run_label"),
        "status": published_review.get("status"),
        "generation_id": published_review.get("publication_generation"),
        "raw_manifest_sha256": published_review.get("raw_manifest_sha256"),
        "raw_campaign_manifest_sha256": published_review.get("raw_campaign_manifest_sha256"),
        "selected_backend": published_review.get("selected_backend"),
        "selected_exact_backend": published_review.get("selected_exact_backend"),
        "selected_workers": published_review.get("selected_workers"),
        "native_profile": published_review.get("native_profile"),
        "native_config_sha256": published_selection.get("native_config_sha256"),
        "accelerator_decision": published_review.get("accelerator_decision"),
        "accelerator_review_manifest_sha256": published_selection.get(
            "accelerator_review_manifest_sha256"
        ),
        "campaign_prerequisite_review_sha256": published_review.get(
            "campaign_prerequisite_review_sha256"
        ),
        "source_review_manifest_sha256": published_review.get("source_review_manifest_sha256"),
    }
    if any(trusted.get(field) != value for field, value in trusted_review_fields.items()) or any(
        not _is_sha256(trusted.get(field))
        for field in (
            "raw_manifest_sha256",
            "raw_campaign_manifest_sha256",
            "native_config_sha256",
            "accelerator_review_manifest_sha256",
            "campaign_prerequisite_review_sha256",
        )
    ):
        raise ValueError("published review provenance does not match the trusted publication")
    with registry_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    expected_fields = [
        "run_label",
        "artifact_type",
        "relative_path",
        "sha256",
        "status",
        "review_manifest_sha256",
        "publication_generation",
        "trusted_manifest_sha256",
    ]
    review_sha256 = normalized_files[published_review_relative]
    expected_rows = [
        {
            "run_label": str(trusted["run_label"]),
            "artifact_type": Path(relative).stem.removeprefix(f"{trusted['run_label']}_"),
            "relative_path": relative,
            "sha256": checksum,
            "status": str(trusted["status"]),
            "review_manifest_sha256": review_sha256,
            "publication_generation": str(trusted["generation_id"]),
            "trusted_manifest_sha256": "",
        }
        for relative, checksum in sorted(normalized_files.items())
        if relative != registry_relative
    ]
    if fieldnames != expected_fields or rows != expected_rows:
        raise ValueError("registry rows do not match the trusted publication")


def _publish_stage052_artifacts(
    *,
    review_manifest: Path,
    repository_root: Path,
    expected_scope: str,
    expected_status: str,
    checkpoint: Callable[[str], None] | None = None,
    prerequisite_dir: Path | None = None,
    verify_live_formal_chain: bool = False,
    active_results_root: Path | None = None,
) -> dict[str, Path]:
    """Publish data first and replace the trusted manifest as the final step.

    ``checkpoint`` is an injectable filesystem-boundary hook used to exercise
    interrupted publication.  No successful status is inferred from copied
    data; consumers must validate the final manifest and its sidecar.
    """

    source_manifest = _load_review_manifest(
        review_manifest,
        expected_scope=expected_scope,
        expected_status=expected_status,
    )
    from evrptw.stage052_evidence import verify_stage052_review_files

    verify_stage052_review_files(review_manifest.parent.parent, source_manifest)
    if verify_live_formal_chain:
        if prerequisite_dir is None:
            raise ValueError("Formal publication requires the accepted G01 raw directory")
        _verify_live_formal_chain(
            review_manifest=review_manifest,
            review=source_manifest,
            prerequisite_dir=prerequisite_dir,
            repository_root=repository_root,
            active_results_root=active_results_root,
        )
    sources = _verified_sources(review_manifest, source_manifest)
    _verify_gpu_decision(sources["gpu_decision"], review=source_manifest)
    source_review_sha256 = _sha256(review_manifest)
    selection_lock = source_manifest["selection_lock"]
    assert isinstance(selection_lock, Mapping)
    run_label = str(source_manifest["run_label"])
    repository = repository_root.resolve()
    experiments = repository / "experiments"
    summaries = experiments / "summaries"
    registries = experiments / "registries"
    manifests = experiments / "manifests"
    for directory in (experiments, summaries, registries, manifests):
        directory.mkdir(parents=True, exist_ok=True)

    generation_seed = hashlib.sha256(review_manifest.read_bytes())
    for key in sorted(sources):
        generation_seed.update(key.encode("utf-8") + b"\0")
        generation_seed.update(_sha256(sources[key]).encode("ascii"))
    generation_id = generation_seed.hexdigest()
    temporary = experiments / f".stage05.2-publication-{generation_id}-{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    trusted_relative = Path(
        "experiments/manifests/stage05.2_performance_benchmark_artifact_manifest.json"
    )
    trusted_destination = repository / trusted_relative
    canonical_registry_relative = Path("experiments/registries/stage05.2_artifact_registry.csv")
    canonical_registry_destination = repository / canonical_registry_relative
    registry_relative = Path("experiments/registries") / (
        f"{run_label}_{generation_id}_artifact_registry.csv"
    )
    registry_destination = repository / registry_relative
    temporary_files: dict[Path, Path] = {}
    try:
        for key, source in sources.items():
            generated_name = f"{run_label}_{generation_id}_{_TARGET_NAMES[key]}"
            relative = Path("experiments/summaries") / generated_name
            destination = repository / relative
            staged = temporary / relative
            _copy_fsync(source, staged)
            publication_files = source_manifest["publication_files"]
            assert isinstance(publication_files, Mapping)
            declared = publication_files[key]
            assert isinstance(declared, Mapping)
            if _sha256(staged) != declared.get("sha256"):
                raise ValueError(f"publication source changed during copy: {key}")
            temporary_files[destination] = staged

        published_file_hashes = {
            relative: _sha256(staged)
            for destination, staged in temporary_files.items()
            for relative in (destination.relative_to(repository).as_posix(),)
        }
        versioned_review_relative = (
            Path("experiments/summaries") / f"{run_label}_{generation_id}_review_manifest.json"
        )
        published_review_files = {
            key: {
                "relative_path": f"{run_label}_{generation_id}_{_TARGET_NAMES[key]}",
                "sha256": _sha256(
                    temporary
                    / "experiments/summaries"
                    / f"{run_label}_{generation_id}_{_TARGET_NAMES[key]}"
                ),
            }
            for key in _TARGET_NAMES
        }
        versioned_review_payload: dict[str, object] = {
            **source_manifest,
            "schema_version": "stage05.2-published-campaign-review-v1",
            "source_review_manifest_sha256": source_review_sha256,
            "publication_files": published_review_files,
            "files": {
                str(item["relative_path"]): str(item["sha256"])
                for item in published_review_files.values()
            },
            "publication_generation": generation_id,
        }
        versioned_review_staged = temporary / versioned_review_relative
        _write_fsync(versioned_review_staged, _json_bytes(versioned_review_payload))
        verify_published_stage052_review(versioned_review_staged)
        versioned_review_destination = repository / versioned_review_relative
        temporary_files[versioned_review_destination] = versioned_review_staged
        published_file_hashes[versioned_review_relative.as_posix()] = _sha256(
            versioned_review_staged
        )

        registry_staged = temporary / registry_relative
        _write_fsync(
            registry_staged,
            _registry_bytes(
                run_label=run_label,
                status=expected_status,
                files=published_file_hashes,
                review_manifest_sha256=_sha256(versioned_review_staged),
                publication_generation=generation_id,
            ),
        )
        temporary_files[registry_destination] = registry_staged
        published_file_hashes[registry_relative.as_posix()] = _sha256(registry_staged)

        trusted_payload: dict[str, object] = {
            "schema_version": PUBLICATION_SCHEMA,
            "run_label": run_label,
            "status": expected_status,
            "generation_id": generation_id,
            "raw_manifest_sha256": source_manifest["raw_manifest_sha256"],
            "raw_campaign_manifest_sha256": source_manifest["raw_campaign_manifest_sha256"],
            "selected_backend": source_manifest["selected_backend"],
            "selected_exact_backend": source_manifest["selected_exact_backend"],
            "selected_workers": source_manifest["selected_workers"],
            "native_profile": source_manifest["native_profile"],
            "native_config_sha256": selection_lock["native_config_sha256"],
            "accelerator_decision": source_manifest["accelerator_decision"],
            "accelerator_review_manifest_sha256": selection_lock[
                "accelerator_review_manifest_sha256"
            ],
            "campaign_prerequisite_review_sha256": source_manifest[
                "campaign_prerequisite_review_sha256"
            ],
            "source_review_manifest_sha256": source_review_sha256,
            "published_review_manifest": versioned_review_relative.as_posix(),
            "artifact_registry": registry_relative.as_posix(),
            "files": dict(sorted(published_file_hashes.items())),
        }
        trusted_staged = temporary / trusted_relative
        _write_fsync(trusted_staged, _json_bytes(trusted_payload))
        canonical_registry_staged = temporary / canonical_registry_relative
        _write_fsync(canonical_registry_staged, registry_staged.read_bytes())

        for destination, staged in temporary_files.items():
            if destination.exists():
                if _sha256(destination) != _sha256(staged):
                    raise FileExistsError(
                        f"immutable Stage 5.2 publication collision: {destination}"
                    )
                staged.unlink()
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, destination)
            _fsync_directory(destination.parent)
        if _sha256(review_manifest) != source_review_sha256:
            raise ValueError("campaign review pointer changed during publication")
        canonical_registry_destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(canonical_registry_staged, canonical_registry_destination)
        _fsync_directory(canonical_registry_destination.parent)
        if checkpoint is not None:
            checkpoint("before_trusted_manifest")

        if checkpoint is not None:
            checkpoint("after_sidecar_before_trusted_manifest")
        final_source_manifest = _load_review_manifest(
            review_manifest,
            expected_scope=expected_scope,
            expected_status=expected_status,
        )
        verify_stage052_review_files(
            review_manifest.parent.parent,
            final_source_manifest,
        )
        if _sha256(review_manifest) != source_review_sha256:
            raise ValueError("campaign review pointer changed before trusted publication")
        final_sources = _verified_sources(review_manifest, final_source_manifest)
        _verify_gpu_decision(
            final_sources["gpu_decision"],
            review=final_source_manifest,
        )
        if verify_live_formal_chain:
            assert prerequisite_dir is not None
            _verify_live_formal_chain(
                review_manifest=review_manifest,
                review=final_source_manifest,
                prerequisite_dir=prerequisite_dir,
                repository_root=repository_root,
                active_results_root=active_results_root,
            )
        os.replace(trusted_staged, trusted_destination)
        _fsync_directory(trusted_destination.parent)
        if checkpoint is not None:
            checkpoint("after_trusted_manifest")

        published = json.loads(trusted_destination.read_text(encoding="utf-8"))
        verify_published_stage052_review(versioned_review_destination)
        verify_canonical_registry_against_trusted(
            registry_path=registry_destination,
            trusted_manifest_path=trusted_destination,
        )
        verify_canonical_registry_against_trusted(
            registry_path=canonical_registry_destination,
            trusted_manifest_path=trusted_destination,
        )
        if any(
            _sha256(repository / relative) != checksum
            for relative, checksum in published["files"].items()
        ):
            raise RuntimeError("published Stage 5.2 generation failed self-verification")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return {
        "artifact_registry": registry_destination,
        "canonical_artifact_registry": canonical_registry_destination,
        "artifact_manifest": trusted_destination,
        "published_review_manifest": versioned_review_destination,
    }


def publish_stage052_artifacts(
    *,
    review_manifest: Path,
    repository_root: Path,
    prerequisite_dir: Path,
    checkpoint: Callable[[str], None] | None = None,
    active_results_root: Path | None = None,
) -> dict[str, Path]:
    """Atomically publish only an accepted G02 Formal review."""

    return _publish_stage052_artifacts(
        review_manifest=review_manifest,
        repository_root=repository_root,
        expected_scope="formal",
        expected_status=FORMAL_READY,
        checkpoint=checkpoint,
        prerequisite_dir=prerequisite_dir,
        verify_live_formal_chain=True,
        active_results_root=active_results_root,
    )


def dry_run_stage052_publication(
    *,
    review_manifest: Path,
    workspace_root: Path,
) -> dict[str, object]:
    """Exercise the complete publisher transaction for an accepted G01 pilot.

    The transaction runs in an isolated temporary repository and is removed
    after self-verification, so a pilot can never leave tracked READY output.
    """

    workspace_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="stage05.2-publication-dry-run-",
        dir=workspace_root,
    ) as temporary:
        outputs = _publish_stage052_artifacts(
            review_manifest=review_manifest,
            repository_root=Path(temporary),
            expected_scope="pilot",
            expected_status="READY_FOR_STAGE052_FORMAL_BENCHMARK",
        )
        manifest = json.loads(outputs["artifact_manifest"].read_text(encoding="utf-8"))
        if manifest.get("status") != "READY_FOR_STAGE052_FORMAL_BENCHMARK":
            raise RuntimeError("pilot publication dry run produced the wrong status")
        return {
            "schema_version": "stage05.2-publication-dry-run-v1",
            "run_label": manifest["run_label"],
            "source_review_manifest_sha256": _sha256(review_manifest),
            "publication_generation": manifest["generation_id"],
            "file_count": len(manifest["files"]),
            "status": "passed",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--prerequisite-dir", type=Path, required=True)
    parser.add_argument(
        "--active-results-root",
        type=Path,
        help=(
            "explicit live evidence root containing both the Formal campaign "
            "and its accepted Pilot prerequisite; defaults to "
            "<repository-root>/results"
        ),
    )
    arguments = parser.parse_args()
    outputs = publish_stage052_artifacts(
        review_manifest=arguments.review_manifest,
        repository_root=arguments.repository_root,
        prerequisite_dir=arguments.prerequisite_dir,
        active_results_root=arguments.active_results_root,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
