"""Fail-fast execution boundary for Stage 5.2 benchmark campaigns.

The campaign contract in :mod:`evrptw.stage052_campaign` owns path-free
planning and state.  This module binds that contract to accepted predecessor
evidence and to the local machine boundaries used by the experiment runner.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import os
import plistlib
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from evrptw.artifacts import (
    ArtifactReader,
    atomic_write_signed_json,
    signed_sidecar_matches,
)
from evrptw.candidate_transaction import NativeCandidateTransactionConfig
from evrptw.native_kernels import NATIVE_KERNEL_ABI_VERSION
from evrptw.stage052 import STAGE052_MAXIMUM_PERSISTENCE_RATIO
from evrptw.stage052_campaign import (
    RUNTIME_LOAD_POLICY,
    ArchiveTransferCompletedError,
    BatchArchiver,
    BatchManifest,
    BatchPlan,
    BenchmarkCampaignConfig,
    BenchmarkPreflightObservation,
    CampaignManifest,
    PilotStorageObservation,
    ProcessCpuCounterSample,
    StorageRootLocator,
    SystemLoadWindow,
    VolumeIdentity,
    directory_byte_count,
    directory_checksum,
    load_campaign_manifest,
    maximum_process_average_cores,
    maximum_process_average_cores_over_windows,
)
from evrptw.stage052_evidence import (
    STAGE052_PREVIOUS_RESOURCE_SCHEMA_VERSION,
    STAGE052_RESOURCE_SCHEMA_VERSION,
    PersistenceInterval,
    RunResourceSummary,
    is_stage052_dedicated_cgroup_path,
    verify_stage052_campaign_gate_set,
    verify_stage052_review_files,
)
from evrptw.stage052_platform import (
    WindowsWslPowerStatus,
    read_windows_wsl_power_status,
    read_wsl_ac_power_online,
)
from evrptw.stage052_resources import (
    FormalResourceRecalibrationEvidence,
    ProducerResourceContract,
)

PER_WORKER_RSS_LIMIT_BYTES = 8 * 1024**3
AGGREGATE_RSS_LIMIT_BYTES = 20 * 1024**3
STAGE052_MINIMUM_FREE_BYTES = 50 * 1024**3
_CAMPAIGN_PUBLICATION_BRIDGE_REVISION = (
    "968b9dd421dd4ae2b8542d43b284b3ade94e760c"
)
_CAMPAIGN_SUCCESSOR_ALLOWED_PATHS = frozenset(
    {
        ".github/workflows/ci.yml",
        ".gitignore",
        "AGENTS.md",
        "configs/experiment_catalog.toml",
        "configs/experiment_storage_governance.toml",
        "configs/stage052_performance.toml",
        "configs/stage052_storage_roots.example.toml",
        "cpp/evrptw_core.cpp",
        "docs/experiment_artifact_storage.md",
        "docs/provenance/README.md",
        "docs/provenance/migration-manifest.json",
        "docs/provenance/source-file-disposition-summary.json",
        "docs/provenance/source-file-disposition.csv",
        "docs/stage052_change_log.md",
        "docs/stage052_performance_benchmark_workflow.md",
        "experiments/migrations/stage052_d_archive_ssd_20260725.json",
        "experiments/migrations/stage052_d_archive_ssd_20260725.json.sha256",
        "experiments/registries/experiment_lifecycle_v3_migration.json",
        "experiments/registries/experiment_lifecycle_v3_migration.json.sha256",
        "pyproject.toml",
        "src/evrptw/_core.pyi",
        "src/evrptw/artifacts.py",
        "src/evrptw/experiment_lifecycle.py",
        "src/evrptw/experiments/lifecycle_historical_review.py",
        "src/evrptw/experiments/lifecycle_legacy_review.py",
        "src/evrptw/experiments/stage00_baseline.py",
        "src/evrptw/experiments/stage01_objective.py",
        "src/evrptw/experiments/stage02_constraint_guided.py",
        "src/evrptw/experiments/stage02_route_quality.py",
        "src/evrptw/experiments/stage02_route_reduction.py",
        "src/evrptw/experiments/stage031_cheap_screening.py",
        "src/evrptw/experiments/stage032_cache_incremental.py",
        "src/evrptw/experiments/stage033_exact_deadline.py",
        "src/evrptw/experiments/stage034_control_parallel.py",
        "src/evrptw/experiments/stage03_measurement.py",
        "src/evrptw/experiments/stage04_weights.py",
        "src/evrptw/experiments/stage051_best_known.py",
        "src/evrptw/experiments/stage052_calibration.py",
        "src/evrptw/experiments/stage052_campaign_review.py",
        "src/evrptw/experiments/stage052_historical_semantic_review.py",
        "src/evrptw/experiments/stage052_performance.py",
        "src/evrptw/experiments/stage052_performance_review.py",
        "src/evrptw/experiments/stage052_replay_benchmark.py",
        "src/evrptw/experiments/stage052_review_benchmark.py",
        "src/evrptw/stage052.py",
        "src/evrptw/stage052_campaign.py",
        "src/evrptw/stage052_campaign_runner.py",
        "src/evrptw/stage052_evidence.py",
        "src/evrptw/stage052_memory.py",
        "src/evrptw/stage052_platform.py",
        "src/evrptw/stage052_replay.py",
        "src/evrptw/stage052_resources.py",
        "src/evrptw/stage052_retention.py",
        "src/evrptw/stage052_review_service.py",
        "src/evrptw/stage052_storage_migration.py",
        "src/evrptw/storage_governance.py",
        "tests/test_artifacts.py",
        "tests/test_experiment_lifecycle.py",
        "tests/test_publication_metadata.py",
        "tests/test_stage052.py",
        "tests/test_stage052_calibration.py",
        "tests/test_stage052_campaign_review.py",
        "tests/test_stage052_campaign_runner.py",
        "tests/test_stage052_campaign.py",
        "tests/test_stage052_platform.py",
        "tests/test_stage052_replay.py",
        "tests/test_stage052_resources.py",
        "tests/test_stage052_retention.py",
        "tests/test_stage052_review_service.py",
        "tests/test_stage052_source_deletion.py",
        "tests/test_stage052_storage_migration.py",
        "tests/test_storage_governance.py",
        "tests/test_artifacts_v3.py",
        "tools/delete_stage052_verified_sources.py",
        "tools/hash_retention_tree_windows.py",
        "tools/migrate_stage052_retention_v2.py",
        "tools/publish_stage052_artifacts.py",
    }
)
_CAMPAIGN_SUCCESSOR_PINNED_PRODUCER_FIXES = {
    "cpp/evrptw_core.cpp": (
        "a7351017c881623e11e3af359f3ccf23a491b56ff38d72ed0e4c867ea48b5537"
    ),
    "src/evrptw/alns.py": "8ac397f127241f441d17d3de5f2034fbc855af7c72a2955dbf3ae52bd66a5a70",
    "src/evrptw/cache_incremental.py": (
        "babe9c2f6315c73a81fbb4c7cbb6d673035cc62c903aa19ee3d430c94e09bdaa"
    ),
    "src/evrptw/candidate_transaction.py": (
        "a8e3ca9f6b6158a40db5e02eb07ca6290a74c1a271a323d84be4611eca486d80"
    ),
    "src/evrptw/experiments/stage03_measurement_review.py": (
        "7299f13a15571ecd6a1617299b8d769baba6a8c3773947304231cdce11438fb8"
    ),
    "src/evrptw/measurement.py": (
        "3e6b5cbfa63a881834d03e14167c0f940b505e5d81400296cd56930bc401917f"
    ),
    "tests/test_alns_wall_clock_only.py": (
        "0bf741d1e4929564882dd8ff48775f0c589e957328aa00c7309176656fd16481"
    ),
    "tests/test_artifacts_v3.py": (
        "3cbb4ef018cab4ba9de8f09810602c7bb7678d235264475e2039a841a7f357d2"
    ),
    "tests/test_candidate_transaction.py": (
        "71581efa9f8e447ac89cb5ae030cc08fdeec362fb28dafcb4cedd228883d8106"
    ),
    "tests/test_stage033_exact_deadline.py": (
        "cd8cbcf415b30c3d29d53aca8fecd22bd3ded62099de621b9cbf8c011d827de1"
    ),
    "tests/test_stage052_streaming_trace.py": (
        "c9c01a14fc1e77dc3f9b475c68cadd03a22734df84d2c281eb7b5be3be2d32b9"
    ),
}

_CALIBRATION_SUCCESSOR_PREDECESSOR = (
    "58c325ac4263b623449d465a803adc5953d32644"
)
_CALIBRATION_SUCCESSOR_RUN_LABEL = "stage05.2_resource_calibration_attempt23"
_CALIBRATION_SUCCESSOR_ALLOWED_PATHS = frozenset(
    {
        ".gitignore",
        "AGENTS.md",
        "docs/provenance/migration-manifest.json",
        "docs/provenance/source-file-disposition.csv",
        "docs/stage052_change_log.md",
        "docs/stage052_performance_benchmark_workflow.md",
        "src/evrptw/experiment_lifecycle.py",
        "src/evrptw/experiments/stage052_campaign_review.py",
        "src/evrptw/experiments/stage052_performance.py",
        "src/evrptw/stage052_campaign.py",
        "src/evrptw/stage052_campaign_runner.py",
        "src/evrptw/stage052_evidence.py",
        "tests/test_experiment_lifecycle.py",
        "tests/test_stage052_campaign.py",
        "tests/test_stage052_campaign_review.py",
        "tests/test_stage052_campaign_runner.py",
        "tools/create_stage052_calibration_successor_attestation.py",
    }
)
_CALIBRATION_REQUIRED_REVIEW_GATES = frozenset(
    {
        "terminal_manifest_replay",
        "artifact_inventory",
        "resource_contract_replay",
        "formal_memory_measurement_binding",
        "cgroup_peak_reset",
        "parent_memory_release",
        "axis_memory_release",
        "batch_memory_release",
        "locked_topology",
    }
)
_CALIBRATION_SCIENTIFIC_PATHS = (
    "configs/stage02_constraint_guided.toml",
    "configs/stage04_weights.toml",
    "configs/stage052_performance.toml",
    "cpp/evrptw_core.cpp",
    "src/evrptw/alns.py",
    "src/evrptw/artifacts.py",
    "src/evrptw/candidate_control.py",
    "src/evrptw/candidate_transaction.py",
    "src/evrptw/charging.py",
    "src/evrptw/exact_deadline.py",
    "src/evrptw/models.py",
    "src/evrptw/native_kernels.py",
    "src/evrptw/objective.py",
    "src/evrptw/validation.py",
)
_CALIBRATION_PINNED_RECOVERY_BLOBS = {
    "src/evrptw/experiments/stage052_performance.py": (
        "def7b941ed4e824f6eaa6b899924ba1ff01332cce41e5734437c88764b55f56d"
    ),
    "src/evrptw/stage052_campaign.py": (
        "5a2ef5e115cf1f9d144b2861cc6f9d40cc7d7a06c33f819eafb8d63ecd91a75c"
    ),
    "src/evrptw/stage052_evidence.py": (
        "7e3a9b126b5076688efbfdd4d48fc960f1c5ff9f79a6c139c73f04c18a45caaf"
    ),
}
_CALIBRATION_ATTESTATION_NODES = frozenset(
    {
        "_CALIBRATION_SUCCESSOR_PREDECESSOR",
        "_CALIBRATION_SUCCESSOR_RUN_LABEL",
        "_CALIBRATION_SUCCESSOR_ALLOWED_PATHS",
        "_CALIBRATION_REQUIRED_REVIEW_GATES",
        "_CALIBRATION_SCIENTIFIC_PATHS",
        "_CALIBRATION_PINNED_RECOVERY_BLOBS",
        "_CALIBRATION_ATTESTATION_NODES",
        "CalibrationSuccessorAttestation",
        "_calibration_successor_blob_hashes",
        "_git_revision_bytes",
        "_calibration_scientific_surface_sha256",
        "_verified_calibration_successor_inputs",
        "create_calibration_successor_attestation",
        "verify_calibration_successor_attestation",
    }
)


@dataclass(frozen=True, slots=True)
class CalibrationSuccessorAttestation:
    """Narrow proof that only the reviewed resource contract crosses revision."""

    predecessor_revision: str
    successor_revision: str
    calibration_run_label: str
    calibration_report_sha256: str
    calibration_review_manifest_sha256: str
    resource_contract_sha256: str
    predecessor_scientific_surface_sha256: str
    successor_scientific_surface_sha256: str
    changed_blob_sha256_by_path: Mapping[str, str]
    inheritance_scope: str = "resource_contract_only"
    formal_batch_geometry_contribution: int = 0

    def __post_init__(self) -> None:
        if (
            self.predecessor_revision != _CALIBRATION_SUCCESSOR_PREDECESSOR
            or re.fullmatch(r"[0-9a-f]{40}", self.successor_revision) is None
            or self.calibration_run_label != _CALIBRATION_SUCCESSOR_RUN_LABEL
            or self.inheritance_scope != "resource_contract_only"
            or self.formal_batch_geometry_contribution != 0
        ):
            raise ValueError("calibration successor identity/scope is invalid")
        digests = (
            self.calibration_report_sha256,
            self.calibration_review_manifest_sha256,
            self.resource_contract_sha256,
            self.predecessor_scientific_surface_sha256,
            self.successor_scientific_surface_sha256,
        )
        changed = dict(self.changed_blob_sha256_by_path)
        if (
            any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in digests)
            or not changed
            or not set(changed).issubset(_CALIBRATION_SUCCESSOR_ALLOWED_PATHS)
            or any(
                re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for digest in changed.values()
            )
        ):
            raise ValueError("calibration successor digest/path inventory is invalid")
        object.__setattr__(self, "changed_blob_sha256_by_path", changed)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "stage05.2-calibration-successor-attestation-v1",
            "predecessor_revision": self.predecessor_revision,
            "successor_revision": self.successor_revision,
            "calibration_run_label": self.calibration_run_label,
            "calibration_report_sha256": self.calibration_report_sha256,
            "calibration_review_manifest_sha256": (
                self.calibration_review_manifest_sha256
            ),
            "resource_contract_sha256": self.resource_contract_sha256,
            "predecessor_scientific_surface_sha256": (
                self.predecessor_scientific_surface_sha256
            ),
            "successor_scientific_surface_sha256": (
                self.successor_scientific_surface_sha256
            ),
            "changed_blob_sha256_by_path": dict(
                sorted(self.changed_blob_sha256_by_path.items())
            ),
            "inheritance_scope": self.inheritance_scope,
            "formal_batch_geometry_contribution": (
                self.formal_batch_geometry_contribution
            ),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> CalibrationSuccessorAttestation:
        fields = {
            "schema_version",
            "predecessor_revision",
            "successor_revision",
            "calibration_run_label",
            "calibration_report_sha256",
            "calibration_review_manifest_sha256",
            "resource_contract_sha256",
            "predecessor_scientific_surface_sha256",
            "successor_scientific_surface_sha256",
            "changed_blob_sha256_by_path",
            "inheritance_scope",
            "formal_batch_geometry_contribution",
        }
        if set(payload) != fields or payload.get("schema_version") != (
            "stage05.2-calibration-successor-attestation-v1"
        ):
            raise ValueError("calibration successor attestation schema differs")
        changed = payload.get("changed_blob_sha256_by_path")
        if not isinstance(changed, Mapping) or any(
            not isinstance(path, str) or not isinstance(digest, str)
            for path, digest in changed.items()
        ):
            raise ValueError("calibration successor changed-blob map is invalid")
        contribution = payload.get("formal_batch_geometry_contribution")
        if isinstance(contribution, bool) or not isinstance(contribution, int):
            raise ValueError("calibration successor geometry contribution is invalid")
        return cls(
            predecessor_revision=str(payload["predecessor_revision"]),
            successor_revision=str(payload["successor_revision"]),
            calibration_run_label=str(payload["calibration_run_label"]),
            calibration_report_sha256=str(payload["calibration_report_sha256"]),
            calibration_review_manifest_sha256=str(
                payload["calibration_review_manifest_sha256"]
            ),
            resource_contract_sha256=str(payload["resource_contract_sha256"]),
            predecessor_scientific_surface_sha256=str(
                payload["predecessor_scientific_surface_sha256"]
            ),
            successor_scientific_surface_sha256=str(
                payload["successor_scientific_surface_sha256"]
            ),
            changed_blob_sha256_by_path={
                str(path): str(digest) for path, digest in changed.items()
            },
            inheritance_scope=str(payload["inheritance_scope"]),
            formal_batch_geometry_contribution=contribution,
        )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def campaign_runtime_contract_sha256(value: Mapping[str, object]) -> str:
    """Hash the explicit hard runtime contract; every other field is telemetry."""

    hard_fields = (
        "schema_version",
        "repository_revision",
        "wheel_filename",
        "wheel_sha256",
        "python_version",
        "python_executable_sha256",
        "native_extension_sha256",
        "dependency_versions",
        "dependency_manifest_sha256",
        "installed_distribution_sha256",
        "installed_editable",
    )
    missing = [field for field in hard_fields if field not in value]
    if missing:
        raise RuntimeError(
            "runtime identity hard contract is incomplete: " + ", ".join(missing)
        )
    return _canonical_sha256({field: value[field] for field in hard_fields})


def campaign_runtime_selection_sha256(
    value: Mapping[str, object],
    *,
    storage_migration: Mapping[str, object] | None = None,
) -> str:
    """Hash stable third-party runtime inputs across an attested G successor."""

    excluded = {
        "dependency_manifest_sha256",
        "installed_distribution_sha256",
        "machine_identity",
        "native_extension",
        "native_extension_sha256",
        "python_executable",
        "repository_revision",
        "source_repository_mount",
        "source_repository_root",
        "wheel_filename",
        "wheel_path",
        "wheel_sha256",
    }
    selection = {key: item for key, item in value.items() if key not in excluded}
    dependencies = selection.get("dependency_versions")
    if isinstance(dependencies, Mapping):
        selection["dependency_versions"] = {
            key: item
            for key, item in dependencies.items()
            if key not in {"evrptw-reproduction", "reproducible-evrptw"}
        }
    return _canonical_sha256(selection)


def campaign_configuration_selection_sha256(
    content: bytes,
    *,
    archive_root_aliases_override: tuple[str, ...] | None = None,
) -> str:
    """Hash scientific inputs while resource tuning remains separately signed."""

    try:
        payload = tomllib.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError("benchmark configuration is not valid UTF-8 TOML") from error
    campaign = payload.get("campaign")
    if isinstance(campaign, dict):
        campaign.pop("resource_calibration_contract", None)
        if archive_root_aliases_override is not None:
            if (
                len(archive_root_aliases_override) != 1
                or not archive_root_aliases_override[0]
            ):
                raise RuntimeError("archive root alias override must contain one alias")
            campaign["archive_root_aliases"] = list(archive_root_aliases_override)
            retention = payload.get("retention")
            if not isinstance(retention, dict):
                raise RuntimeError("benchmark retention configuration is missing")
            retention["archive_root_alias"] = archive_root_aliases_override[0]
    artifacts = payload.get("artifact_storage_v2")
    if isinstance(artifacts, dict):
        artifacts.pop("parquet_row_group_size", None)
        artifacts.pop("parquet_queue_depth", None)
    return _canonical_sha256(payload)


def _resolve_campaign_predecessor_revision(
    repository: Path,
    *,
    predecessor_revision: str,
    current_revision: str,
) -> str:
    """Resolve an immutable legacy revision through the public-history bridge."""

    direct = subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            predecessor_revision,
            current_revision,
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if direct.returncode == 0:
        return predecessor_revision
    bridge_ancestor = subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            _CAMPAIGN_PUBLICATION_BRIDGE_REVISION,
            current_revision,
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if bridge_ancestor.returncode != 0:
        raise RuntimeError(
            "benchmark repository revision is not a descendant of the publication bridge"
        )
    mapping_bytes = subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "show",
            (
                f"{_CAMPAIGN_PUBLICATION_BRIDGE_REVISION}:"
                "docs/provenance/legacy-stage-commit-map.csv"
            ),
        ),
        check=True,
        capture_output=True,
        timeout=10.0,
    ).stdout
    rows = csv.DictReader(io.StringIO(mapping_bytes.decode("utf-8-sig")))
    mapped = {
        str(row.get("legacy_sha", "")): str(row.get("public_sha", ""))
        for row in rows
    }.get(predecessor_revision)
    if mapped is None or re.fullmatch(r"[0-9a-f]{40}", mapped) is None:
        raise RuntimeError(
            "benchmark predecessor revision has no immutable public-history mapping"
        )
    mapped_ancestor = subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            mapped,
            _CAMPAIGN_PUBLICATION_BRIDGE_REVISION,
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if mapped_ancestor.returncode != 0:
        raise RuntimeError(
            "benchmark predecessor public mapping does not reach the publication bridge"
        )
    return _CAMPAIGN_PUBLICATION_BRIDGE_REVISION


def _runtime_selection_with_attested_archive_source(
    runtime: Mapping[str, object],
    storage_migration: Mapping[str, object],
) -> dict[str, object]:
    """Normalize only an attested archive-disk replacement to its source identity."""

    machine = _mapping(runtime.get("machine_identity"), "runtime machine identity")
    observed_disk = _mapping(
        machine.get("d_archive_disk"),
        "runtime D archive disk identity",
    )
    source_disk = _mapping(
        storage_migration.get("source_machine_disk"),
        "storage migration source disk identity",
    )
    destination_disk = _mapping(
        storage_migration.get("destination_machine_disk"),
        "storage migration destination disk identity",
    )
    if observed_disk != destination_disk:
        raise RuntimeError(
            "benchmark runtime archive disk differs from the attested migration destination"
        )
    normalized_machine = dict(machine)
    normalized_machine["d_archive_disk"] = dict(source_disk)
    normalized_runtime = dict(runtime)
    normalized_runtime["machine_identity"] = normalized_machine
    return normalized_runtime


def verify_campaign_successor_revision(
    repository: Path,
    *,
    predecessor_revision: str,
    current_revision: str,
) -> tuple[str, ...]:
    """Allow a newer revision only when every change is confined to G governance."""

    for label, revision in (
        ("predecessor", predecessor_revision),
        ("current", current_revision),
    ):
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise RuntimeError(f"campaign {label} repository revision is invalid")
    resolved = repository.resolve()
    comparison_revision = _resolve_campaign_predecessor_revision(
        resolved,
        predecessor_revision=predecessor_revision,
        current_revision=current_revision,
    )
    result = subprocess.run(
        (
            "git",
            "-C",
            str(resolved),
            "diff",
            "--name-only",
            "-z",
            comparison_revision,
            current_revision,
        ),
        check=True,
        capture_output=True,
        timeout=10.0,
    )
    changed_paths = tuple(
        sorted(
            path.decode("utf-8")
            for path in result.stdout.split(b"\0")
            if path
        )
    )
    forbidden = tuple(
        path
        for path in changed_paths
        if path not in _CAMPAIGN_SUCCESSOR_ALLOWED_PATHS
        and path not in _CAMPAIGN_SUCCESSOR_PINNED_PRODUCER_FIXES
    )
    if not changed_paths:
        raise RuntimeError("campaign successor revision has no recorded changes")
    if forbidden:
        raise RuntimeError(
            "campaign successor revision changes non-G paths: " + ", ".join(forbidden)
        )
    pinned_paths = frozenset(_CAMPAIGN_SUCCESSOR_PINNED_PRODUCER_FIXES)
    pinned_changed = frozenset(changed_paths) & pinned_paths
    for path in sorted(pinned_paths if pinned_changed else ()):
        try:
            content = subprocess.run(
                ("git", "-C", str(resolved), "show", f"{current_revision}:{path}"),
                check=True,
                capture_output=True,
                timeout=10.0,
            ).stdout
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"campaign successor revision lacks pinned producer-fix content: {path}"
            ) from error
        if (
            hashlib.sha256(content).hexdigest()
            != _CAMPAIGN_SUCCESSOR_PINNED_PRODUCER_FIXES[path]
        ):
            raise RuntimeError(
                f"campaign successor revision changes pinned producer-fix content: {path}"
            )
    return changed_paths


def _calibration_successor_blob_hashes(
    repository: Path,
    *,
    predecessor_revision: str,
    successor_revision: str,
) -> dict[str, str]:
    result = subprocess.run(
        (
            "git",
            "-C",
            str(repository.resolve()),
            "diff",
            "--name-only",
            "-z",
            predecessor_revision,
            successor_revision,
        ),
        check=True,
        capture_output=True,
        timeout=10.0,
    )
    paths = tuple(
        sorted(
            path.decode("utf-8")
            for path in result.stdout.split(b"\0")
            if path
        )
    )
    forbidden = set(paths).difference(_CALIBRATION_SUCCESSOR_ALLOWED_PATHS)
    if not paths or forbidden:
        raise RuntimeError(
            "calibration successor changes forbidden scientific/runtime paths: "
            + ", ".join(sorted(forbidden))
        )
    blobs: dict[str, str] = {}
    for path in paths:
        try:
            content = subprocess.run(
                (
                    "git",
                    "-C",
                    str(repository.resolve()),
                    "show",
                    f"{successor_revision}:{path}",
                ),
                check=True,
                capture_output=True,
                timeout=10.0,
            ).stdout
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"calibration successor path is deleted or unreadable: {path}"
            ) from error
        blobs[path] = hashlib.sha256(content).hexdigest()
    if set(_CALIBRATION_PINNED_RECOVERY_BLOBS).difference(blobs):
        raise RuntimeError("calibration successor lacks pinned recovery-only blobs")
    if any(
        blobs[path] != expected
        for path, expected in _CALIBRATION_PINNED_RECOVERY_BLOBS.items()
    ):
        raise RuntimeError(
            "calibration successor changes an audited recovery-only mixed module"
        )
    return blobs


def _git_revision_bytes(repository: Path, revision: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ("git", "-C", str(repository.resolve()), "show", f"{revision}:{path}"),
            check=True,
            capture_output=True,
            timeout=10.0,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"calibration scientific surface path is unreadable: {path}"
        ) from error


def _calibration_scientific_surface_sha256(
    repository: Path,
    revision: str,
) -> str:
    """Hash solver/config/artifact code plus protected producer top-level nodes."""

    digest = hashlib.sha256(b"stage05.2-calibration-scientific-surface-v1\0")
    for path in _CALIBRATION_SCIENTIFIC_PATHS:
        content = _git_revision_bytes(repository, revision, path)
        digest.update(path.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(content).digest())
    runner_path = "src/evrptw/stage052_campaign_runner.py"
    source = _git_revision_bytes(repository, revision, runner_path).decode(
        "utf-8"
    )
    module = ast.parse(source, filename=runner_path)
    protected_nodes: list[ast.stmt] = []
    for node in module.body:
        if isinstance(node, ast.Import) and any(
            alias.name == "ast" for alias in node.names
        ):
            continue
        name = getattr(node, "name", None)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {
                target.id for target in targets if isinstance(target, ast.Name)
            }
            if names.intersection(_CALIBRATION_ATTESTATION_NODES):
                continue
        elif isinstance(name, str) and name in _CALIBRATION_ATTESTATION_NODES:
            continue
        protected_nodes.append(node)
    protected_module = ast.Module(body=protected_nodes, type_ignores=[])
    digest.update(runner_path.encode("utf-8") + b"\0")
    digest.update(ast.dump(protected_module, include_attributes=False).encode("utf-8"))
    return digest.hexdigest()


def _verified_calibration_successor_inputs(
    *,
    calibration_report_path: Path,
    calibration_review_manifest_path: Path,
    resource_contract_path: Path,
) -> tuple[str, str, str]:
    for path, sidecar in (
        (
            calibration_report_path,
            calibration_report_path.with_suffix(".sha256"),
        ),
        (
            calibration_review_manifest_path,
            calibration_review_manifest_path.with_suffix(
                calibration_review_manifest_path.suffix + ".sha256"
            ),
        ),
        (resource_contract_path, resource_contract_path.with_suffix(".sha256")),
    ):
        if not signed_sidecar_matches(path, sidecar):
            raise RuntimeError(f"calibration successor input is not signed: {path}")
    try:
        report = json.loads(calibration_report_path.read_text(encoding="utf-8"))
        review = json.loads(
            calibration_review_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("calibration successor input JSON is invalid") from error
    if (
        not isinstance(report, Mapping)
        or report.get("run_label") != _CALIBRATION_SUCCESSOR_RUN_LABEL
        or not isinstance(review, Mapping)
        or review.get("schema_version")
        != "stage05.2-resource-calibration-review-v1"
        or review.get("run_label") != _CALIBRATION_SUCCESSOR_RUN_LABEL
        or review.get("status") != "ACCEPTED"
        or review.get("calibration_report_sha256")
        != _file_sha256(calibration_report_path)
        or review.get("resource_contract_sha256")
        != _file_sha256(resource_contract_path)
        or not isinstance(review.get("gates"), Mapping)
        or set(cast(Mapping[str, object], review["gates"]))
        != _CALIBRATION_REQUIRED_REVIEW_GATES
        or any(
            not isinstance(gate, Mapping) or gate.get("passed") is not True
            for gate in cast(Mapping[str, object], review["gates"]).values()
        )
    ):
        raise RuntimeError("calibration successor inputs are not independently accepted")
    return (
        _file_sha256(calibration_report_path),
        _file_sha256(calibration_review_manifest_path),
        _file_sha256(resource_contract_path),
    )


def create_calibration_successor_attestation(
    *,
    repository: Path,
    successor_revision: str,
    calibration_report_path: Path,
    calibration_review_manifest_path: Path,
    resource_contract_path: Path,
    output_path: Path,
) -> Path:
    """Create the signed Attempt23 resource-contract-only successor proof."""

    report_sha, review_sha, contract_sha = _verified_calibration_successor_inputs(
        calibration_report_path=calibration_report_path,
        calibration_review_manifest_path=calibration_review_manifest_path,
        resource_contract_path=resource_contract_path,
    )
    blobs = _calibration_successor_blob_hashes(
        repository,
        predecessor_revision=_CALIBRATION_SUCCESSOR_PREDECESSOR,
        successor_revision=successor_revision,
    )
    predecessor_surface = _calibration_scientific_surface_sha256(
        repository, _CALIBRATION_SUCCESSOR_PREDECESSOR
    )
    successor_surface = _calibration_scientific_surface_sha256(
        repository, successor_revision
    )
    if predecessor_surface != successor_surface:
        raise RuntimeError(
            "calibration successor changes solver, artifact, or scientific config surface"
        )
    attestation = CalibrationSuccessorAttestation(
        predecessor_revision=_CALIBRATION_SUCCESSOR_PREDECESSOR,
        successor_revision=successor_revision,
        calibration_run_label=_CALIBRATION_SUCCESSOR_RUN_LABEL,
        calibration_report_sha256=report_sha,
        calibration_review_manifest_sha256=review_sha,
        resource_contract_sha256=contract_sha,
        predecessor_scientific_surface_sha256=predecessor_surface,
        successor_scientific_surface_sha256=successor_surface,
        changed_blob_sha256_by_path=blobs,
    )
    path, _sidecar = atomic_write_signed_json(output_path, attestation.to_dict())
    return path


def verify_calibration_successor_attestation(
    *,
    repository: Path,
    current_revision: str,
    attestation_path: Path,
    calibration_report_path: Path,
    calibration_review_manifest_path: Path,
    resource_contract_path: Path,
) -> CalibrationSuccessorAttestation:
    """Recompute Git and evidence bindings before inheriting Attempt23 limits."""

    if not signed_sidecar_matches(
        attestation_path, attestation_path.with_suffix(".sha256")
    ):
        raise RuntimeError("calibration successor attestation is not signed")
    try:
        payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("calibration successor attestation is invalid") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError("calibration successor attestation must be an object")
    try:
        attestation = CalibrationSuccessorAttestation.from_dict(payload)
    except ValueError as error:
        raise RuntimeError("calibration successor attestation contract differs") from error
    report_sha, review_sha, contract_sha = _verified_calibration_successor_inputs(
        calibration_report_path=calibration_report_path,
        calibration_review_manifest_path=calibration_review_manifest_path,
        resource_contract_path=resource_contract_path,
    )
    observed_blobs = _calibration_successor_blob_hashes(
        repository,
        predecessor_revision=attestation.predecessor_revision,
        successor_revision=current_revision,
    )
    predecessor_surface = _calibration_scientific_surface_sha256(
        repository, attestation.predecessor_revision
    )
    successor_surface = _calibration_scientific_surface_sha256(
        repository, current_revision
    )
    if (
        attestation.successor_revision != current_revision
        or attestation.calibration_report_sha256 != report_sha
        or attestation.calibration_review_manifest_sha256 != review_sha
        or attestation.resource_contract_sha256 != contract_sha
        or dict(attestation.changed_blob_sha256_by_path) != observed_blobs
        or attestation.predecessor_scientific_surface_sha256
        != predecessor_surface
        or attestation.successor_scientific_surface_sha256 != successor_surface
        or predecessor_surface != successor_surface
    ):
        raise RuntimeError("calibration successor attestation replay differs")
    return attestation


def _input_lock_payload(value: Mapping[str, object]) -> dict[str, object]:
    """Remove scope/runtime observations while retaining stable solver inputs."""

    excluded = {
        "instance_sha256",
        "background_load",
        "power_mode",
        "runtime_signature",
        "process_status_counts",
        "load_average",
    }
    return {key: item for key, item in value.items() if key not in excluded}


def _instance_hashes(value: Mapping[str, object]) -> dict[str, str]:
    raw = value.get("instance_sha256")
    if (
        not isinstance(raw, Mapping)
        or not raw
        or any(
            not isinstance(instance, str)
            or not instance
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for instance, digest in raw.items()
        )
    ):
        raise RuntimeError("accepted benchmark instance hashes are invalid")
    return {str(instance): str(digest) for instance, digest in raw.items()}


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RuntimeError(f"accepted benchmark {field} must be an object")
    return {str(key): item for key, item in value.items()}


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"accepted benchmark {field} must be a SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class MachineSnapshot:
    """One power/load sample at the local machine boundary."""

    power_source: str
    low_power_mode_enabled: bool
    load1: float
    unrelated_process_average_cores: float
    sampled_at_seconds: float | None = None
    unrelated_process_cpu_seconds: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.power_source:
            raise ValueError("machine power_source is required")
        if not isinstance(self.low_power_mode_enabled, bool):
            raise ValueError("machine low_power_mode_enabled must be boolean")
        if (
            not math.isfinite(self.load1)
            or self.load1 < 0.0
            or not math.isfinite(self.unrelated_process_average_cores)
            or self.unrelated_process_average_cores < 0.0
        ):
            raise ValueError("machine load sample is invalid")
        if self.sampled_at_seconds is not None and (
            not math.isfinite(self.sampled_at_seconds) or self.sampled_at_seconds < 0.0
        ):
            raise ValueError("machine sample timestamp is invalid")
        if any(
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or isinstance(seconds, bool)
            or not isinstance(seconds, int | float)
            or not math.isfinite(float(seconds))
            or float(seconds) < 0.0
            for pid, seconds in self.unrelated_process_cpu_seconds.items()
        ):
            raise ValueError("machine process CPU counters are invalid")


def _maximum_window_process_average_cores(
    snapshots: Sequence[MachineSnapshot],
    *,
    logical_cpu_count: int | None = None,
) -> float:
    """Return a replayable conservative CPU-time average over the full window."""

    if not snapshots:
        raise ValueError("machine window requires at least one snapshot")
    counter_snapshots = [
        sample
        for sample in snapshots
        if sample.sampled_at_seconds is not None
    ]
    if len(counter_snapshots) >= 2:
        samples = tuple(
            ProcessCpuCounterSample(
                sampled_at_seconds=cast(float, sample.sampled_at_seconds),
                cpu_seconds_by_pid=sample.unrelated_process_cpu_seconds,
            )
            for sample in counter_snapshots
        )
        return maximum_process_average_cores(
            samples,
            logical_cpu_count=logical_cpu_count or (os.cpu_count() or 1),
        )
    # Compatibility for injected/legacy snapshots without cumulative counters:
    # use the full-window sample mean, never an instantaneous maximum.
    return sum(sample.unrelated_process_average_cores for sample in snapshots) / len(
        snapshots
    )


@dataclass(frozen=True, slots=True)
class BatchRuntimeEvidence:
    """Continuous power/load evidence for one campaign batch."""

    sample_count: int
    maximum_load1: float
    maximum_permitted_load1: float
    maximum_unrelated_process_average_cores: float
    power_sources: tuple[str, ...]
    low_power_mode_observed: bool
    passed: bool
    failure_reason: str
    logical_cpu_count: int | None = None
    process_cpu_samples: tuple[ProcessCpuCounterSample, ...] = ()

    @classmethod
    def from_snapshots(
        cls,
        snapshots: tuple[MachineSnapshot, ...],
        *,
        config: BenchmarkCampaignConfig,
        logical_cpu_count: int | None = None,
    ) -> BatchRuntimeEvidence:
        if not snapshots:
            raise ValueError("batch runtime evidence requires at least one sample")
        sources = tuple(sorted({sample.power_source for sample in snapshots}))
        maximum_load1 = max(sample.load1 for sample in snapshots)
        counter_samples = tuple(
            ProcessCpuCounterSample(
                sampled_at_seconds=float(sample.sampled_at_seconds),
                cpu_seconds_by_pid=sample.unrelated_process_cpu_seconds,
            )
            for sample in snapshots
            if sample.sampled_at_seconds is not None
        )
        recorded_logical_cpu_count = logical_cpu_count or (os.cpu_count() or 1)
        if (
            isinstance(recorded_logical_cpu_count, bool)
            or recorded_logical_cpu_count <= 0
        ):
            raise ValueError("batch runtime logical CPU count is invalid")
        maximum_unrelated = (
            maximum_process_average_cores_over_windows(
                counter_samples,
                logical_cpu_count=recorded_logical_cpu_count,
                window_seconds=config.preflight_window_seconds,
            )
            if len(counter_samples) >= 2
            else _maximum_window_process_average_cores(snapshots)
        )
        maximum_permitted_load1 = RUNTIME_LOAD_POLICY.runtime_maximum_load1
        low_power = any(sample.low_power_mode_enabled for sample in snapshots)
        failures: list[str] = []
        if recorded_logical_cpu_count < config.selected_workers:
            failures.append(
                "logical CPU count is below the locked worker count "
                f"({config.selected_workers})"
            )
        return cls(
            sample_count=len(snapshots),
            maximum_load1=maximum_load1,
            maximum_permitted_load1=maximum_permitted_load1,
            maximum_unrelated_process_average_cores=maximum_unrelated,
            power_sources=sources,
            low_power_mode_observed=low_power,
            passed=not failures,
            failure_reason="; ".join(failures),
            logical_cpu_count=recorded_logical_cpu_count,
            process_cpu_samples=counter_samples if len(counter_samples) >= 2 else (),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "stage05.2-batch-runtime-evidence-v1",
            "sample_count": self.sample_count,
            "maximum_load1": self.maximum_load1,
            "maximum_permitted_load1": self.maximum_permitted_load1,
            "maximum_unrelated_process_average_cores": (
                self.maximum_unrelated_process_average_cores
            ),
            "power_sources": list(self.power_sources),
            "low_power_mode_observed": self.low_power_mode_observed,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
            "logical_cpu_count": self.logical_cpu_count,
            "process_cpu_samples": [
                sample.to_dict() for sample in self.process_cpu_samples
            ],
        }


def collect_preflight_observation(
    config: BenchmarkCampaignConfig,
    *,
    snapshot: Callable[[], MachineSnapshot],
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    sample_interval_seconds: float = 1.0,
    require_idle_load: bool = True,
) -> BenchmarkPreflightObservation:
    """Measure exactly two consecutive preflight windows."""

    if not math.isfinite(sample_interval_seconds) or sample_interval_seconds <= 0.0:
        raise ValueError("preflight sample interval must be positive")
    campaign_started = monotonic()
    observed_power: str | None = None
    observed_low_power = False
    windows: list[SystemLoadWindow] = []
    for window_index in range(config.preflight_window_count):
        window_started = window_index * config.preflight_window_seconds
        absolute_deadline = campaign_started + (
            (window_index + 1) * config.preflight_window_seconds
        )
        maximum_load1 = 0.0
        window_snapshots: list[MachineSnapshot] = []
        while True:
            sample = snapshot()
            window_snapshots.append(sample)
            if observed_power is None:
                observed_power = sample.power_source
            observed_low_power = observed_low_power or sample.low_power_mode_enabled
            maximum_load1 = max(maximum_load1, sample.load1)
            remaining = absolute_deadline - monotonic()
            if remaining <= 0.0:
                break
            sleep(min(sample_interval_seconds, remaining))
        replay_samples = tuple(
            ProcessCpuCounterSample(
                sampled_at_seconds=float(sample.sampled_at_seconds),
                cpu_seconds_by_pid=sample.unrelated_process_cpu_seconds,
            )
            for sample in window_snapshots
            if sample.sampled_at_seconds is not None
        )
        windows.append(
            SystemLoadWindow(
                started_at_seconds=window_started,
                duration_seconds=config.preflight_window_seconds,
                maximum_load1=maximum_load1,
                maximum_unrelated_process_average_cores=(
                    _maximum_window_process_average_cores(window_snapshots)
                ),
                logical_cpu_count=(os.cpu_count() or 1) if len(replay_samples) >= 2 else None,
                process_cpu_samples=replay_samples if len(replay_samples) >= 2 else (),
            )
        )
    observation = BenchmarkPreflightObservation(
        power_source=observed_power or "unknown",
        low_power_mode_enabled=observed_low_power,
        windows=tuple(windows),
    )
    config.validate_preflight(observation, require_idle_load=require_idle_load)
    return observation


def _non_negative_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"batch {field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(f"batch {field} must be finite and non-negative")
    return result


def validate_batch_measurements(
    *,
    batch: BatchPlan,
    rows: Sequence[Mapping[str, object]],
    resource_summary: RunResourceSummary,
    runtime_evidence: BatchRuntimeEvidence,
    expected_workers: int,
    producer_resource_contract: ProducerResourceContract | None = None,
    additional_persistence_seconds: float = 0.0,
) -> float:
    """Enforce per-batch geometry, persistence, memory, and fallback gates."""

    expected = {
        (shard.instance, shard.seed, f"wall_clock_{budget}")
        for shard in batch.shards
        for budget in shard.budgets_seconds
    }
    observed: set[tuple[str, int, str]] = set()
    solver_seconds = 0.0
    persistence_seconds = _non_negative_number(
        additional_persistence_seconds,
        "additional_persistence_seconds",
    )
    for row in rows:
        instance = row.get("instance")
        seed = row.get("seed")
        axis = row.get("axis")
        if (
            not isinstance(instance, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(axis, str)
        ):
            raise RuntimeError("batch per-run identity is invalid")
        identity = (instance, seed, axis)
        if identity in observed:
            raise RuntimeError(f"batch per-run identity is duplicate: {identity}")
        observed.add(identity)
        if row.get("backend") != "cpu_batch":
            raise RuntimeError("batch backend is not cpu_batch")
        if row.get("worker_count") != expected_workers:
            raise RuntimeError("batch row worker count differs from selected workers")
        for field_name in ("native_fallbacks", "native_protocol_fallbacks"):
            value = row.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value != 0:
                raise RuntimeError(f"batch {field_name} must be zero")
        solver_seconds += _non_negative_number(row.get("solver_seconds"), "solver_seconds")
        persistence_seconds += _non_negative_number(
            row.get("artifact_persistence_seconds"),
            "artifact_persistence_seconds",
        )
    if observed != expected:
        raise RuntimeError(
            f"batch axis geometry mismatch: expected={sorted(expected)} observed={sorted(observed)}"
        )
    denominator = solver_seconds + persistence_seconds
    if denominator <= 0.0:
        raise RuntimeError("batch solver plus persistence time must be positive")
    persistence_ratio = persistence_seconds / denominator
    if persistence_ratio > STAGE052_MAXIMUM_PERSISTENCE_RATIO:
        raise RuntimeError(
            "batch aggregate persistence ratio exceeds "
            f"{STAGE052_MAXIMUM_PERSISTENCE_RATIO:.0%}: {persistence_ratio:.9f}"
        )
    if not runtime_evidence.passed:
        raise RuntimeError(f"batch runtime capability violation: {runtime_evidence.failure_reason}")
    allowed_resource_schemas = (
        {STAGE052_RESOURCE_SCHEMA_VERSION}
        if producer_resource_contract is not None
        else {
            STAGE052_PREVIOUS_RESOURCE_SCHEMA_VERSION,
            STAGE052_RESOURCE_SCHEMA_VERSION,
        }
    )
    if (
        resource_summary.schema_version not in allowed_resource_schemas
        or resource_summary.configured_worker_count != expected_workers
        or resource_summary.status != "complete"
    ):
        raise RuntimeError("batch resource summary identity is invalid")
    if producer_resource_contract is not None:
        if producer_resource_contract.selected_workers != expected_workers:
            raise RuntimeError("batch producer resource contract worker mismatch")
        aggregate_limit = producer_resource_contract.aggregate_memory_limit_bytes
        per_worker_limit = producer_resource_contract.per_worker_memory_limit_bytes
    else:
        # Historical Stage 5.2 evidence remains readable under its frozen limits.
        aggregate_limit = AGGREGATE_RSS_LIMIT_BYTES
        per_worker_limit = PER_WORKER_RSS_LIMIT_BYTES
    if producer_resource_contract is not None:
        if (
            resource_summary.aggregate_memory_source != "cgroup_v2"
            or not is_stage052_dedicated_cgroup_path(
                resource_summary.cgroup_path
            )
            or resource_summary.cgroup_swap_peak_bytes != 0
        ):
            raise RuntimeError(
                "batch aggregate memory is not bound to a swap-free "
                "isolated cgroup v2 service"
            )
        if resource_summary.aggregate_peak_memory_bytes > aggregate_limit:
            raise RuntimeError("batch cgroup v2 memory exceeds its campaign lock")
    elif resource_summary.aggregate_peak_rss_bytes > aggregate_limit:
        raise RuntimeError(
            "batch process-tree aggregate RSS exceeds its historical campaign lock"
        )
    descendants = set(resource_summary.descendant_pids)
    process_peaks = dict(resource_summary.process_peak_rss_bytes)
    if not descendants or not descendants.issubset(process_peaks):
        raise RuntimeError("batch worker process RSS evidence is incomplete")
    if any(
        process_peaks[pid] > per_worker_limit for pid in descendants
    ):
        raise RuntimeError("batch per-worker RSS exceeds its campaign lock")
    return persistence_ratio


@dataclass(frozen=True, slots=True)
class BenchmarkExecutionLock:
    """Execution identity frozen by accepted F02 or G01 evidence."""

    prerequisite_run_label: str
    raw_manifest_sha256: str
    selected_backend: str
    selected_exact_backend: str
    selected_workers: int
    repository_revision: str
    runtime_identity_sha256: str
    runtime_contract_sha256: str
    runtime_selection_sha256: str
    input_provenance_sha256: str
    configuration_sha256: str
    configuration_selection_sha256: str
    native_config_sha256: str
    native_kernel_config: Mapping[str, object]
    candidate_transaction_config_sha256: str
    candidate_transaction_config: Mapping[str, object]
    instance_sha256: Mapping[str, str]
    staging_root_alias: str | None = None
    archive_root_aliases_exercised: tuple[str, ...] = ()
    producer_resource_contract: Mapping[str, object] | None = None

    @classmethod
    def from_accepted_evidence(
        cls,
        *,
        metadata: Mapping[str, object],
        review_manifest: Mapping[str, object],
        raw_manifest_sha256: str,
        expected_scope: str,
        expected_status: str,
        configuration_selection_sha256: str | None = None,
    ) -> BenchmarkExecutionLock:
        """Bind every performance-affecting field from one accepted review."""

        raw_sha = _sha256(raw_manifest_sha256, "raw manifest digest")
        run_label = metadata.get("run_label")
        if not isinstance(run_label, str) or not run_label:
            raise RuntimeError("accepted benchmark run_label is missing")
        for field_name in ("run_label", "component", "scope"):
            if review_manifest.get(field_name) != metadata.get(field_name):
                raise RuntimeError(
                    f"accepted benchmark review {field_name} identity mismatch"
                )
        if metadata.get("scope") != expected_scope:
            raise RuntimeError("accepted benchmark predecessor scope mismatch")
        if review_manifest.get("status") != expected_status:
            raise RuntimeError("accepted benchmark predecessor review status mismatch")
        if review_manifest.get("raw_manifest_sha256") != raw_sha:
            raise RuntimeError("accepted benchmark review/raw manifest identity mismatch")
        selected_backend = review_manifest.get("selected_backend")
        raw_accelerator_decision = review_manifest.get("accelerator_decision")
        accelerator_decision = (
            raw_accelerator_decision if isinstance(raw_accelerator_decision, str) else ""
        )
        expected_backend = {
            "GPU_NOT_JUSTIFIED": "native_cpu",
            "NATIVE_CPU_RETAINED": "native_cpu",
            "ACCELERATOR_PROMOTED": "cuda",
        }.get(accelerator_decision)
        if (
            expected_backend is None
            or selected_backend != expected_backend
            or metadata.get("backend") != "cpu_batch"
            or metadata.get("execution_backend") != selected_backend
            or review_manifest.get("selected_exact_backend") != "cpu_batch"
        ):
            raise RuntimeError(
                "Stage 5.2 benchmark execution backend does not match the accepted "
                "accelerator decision/cpu_batch exact backend"
            )
        expected_profile = "cuda" if selected_backend == "cuda" else "native"
        if (
            metadata.get("optimization_profile") != expected_profile
            or review_manifest.get("selected_optimization_profile") != expected_profile
        ):
            raise RuntimeError("accepted accelerator optimization profile is inconsistent")
        workers = metadata.get("worker_count")
        if isinstance(workers, bool) or workers not in {2, 4, 5, 6}:
            raise RuntimeError("accepted benchmark worker selection must be 2, 4, 5, or 6")
        if review_manifest.get("selected_workers") != workers:
            raise RuntimeError("accepted benchmark review worker selection mismatch")
        revision = metadata.get("repository_revision")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise RuntimeError("accepted benchmark repository revision is invalid")
        native = _mapping(metadata.get("native_kernel_config"), "native kernel config")
        required_native = {
            "enabled": True,
            "exact_charging": True,
            "screening": True,
            "propagation": True,
            "distance_matrix": True,
            "abi_version": NATIVE_KERNEL_ABI_VERSION,
            "context_policy": "pack_once_per_solve",
            "failure_policy": "fail_fast_no_fallback",
        }
        if native != required_native:
            raise RuntimeError("accepted benchmark native kernel profile is incomplete")
        if review_manifest.get("native_configuration") != native:
            raise RuntimeError("accepted benchmark review native configuration mismatch")
        compatibility_native = review_manifest.get("native_kernel_config")
        if compatibility_native is not None and compatibility_native != native:
            raise RuntimeError("accepted benchmark review native compatibility field mismatch")
        candidate_transaction = _mapping(
            metadata.get("candidate_transaction_config"),
            "candidate transaction config",
        )
        if candidate_transaction != NativeCandidateTransactionConfig().to_dict():
            raise RuntimeError(
                "accepted benchmark candidate transaction profile is incomplete"
            )
        if (
            review_manifest.get("candidate_transaction_configuration")
            != candidate_transaction
        ):
            raise RuntimeError(
                "accepted benchmark review candidate transaction configuration mismatch"
            )
        runtime = _mapping(metadata.get("runtime_identity"), "runtime identity")
        inputs = _mapping(metadata.get("performance_provenance"), "input provenance")
        config_sha = _sha256(metadata.get("configuration_sha256"), "configuration digest")
        if metadata.get("storage_policy_version") != "artifact-storage-v2":
            raise RuntimeError("accepted benchmark storage policy is not artifact-storage-v2")
        if metadata.get("screening_schema_version") != "screening_decisions_v3":
            raise RuntimeError("accepted benchmark physical schema is not screening_decisions_v3")
        staging_root_alias: str | None = None
        archive_root_aliases_exercised: tuple[str, ...] = ()
        raw_producer_contract = metadata.get("producer_resource_contract")
        reviewed_producer_contract = review_manifest.get("producer_resource_contract")
        producer_contract: Mapping[str, object] | None = None
        if raw_producer_contract is not None or reviewed_producer_contract is not None:
            if (
                not isinstance(raw_producer_contract, Mapping)
                or reviewed_producer_contract != raw_producer_contract
            ):
                raise RuntimeError(
                    "accepted benchmark producer resource contract mismatch"
                )
            parsed_producer_contract = ProducerResourceContract.from_dict(
                raw_producer_contract
            )
            if parsed_producer_contract.selected_workers != workers:
                raise RuntimeError(
                    "accepted benchmark producer resource contract worker mismatch"
                )
            producer_contract = parsed_producer_contract.to_dict()
        if expected_scope == "pilot":
            raw_staging_alias = review_manifest.get("staging_root_alias")
            raw_archive_aliases = review_manifest.get("archive_root_aliases_exercised")
            if (
                not isinstance(raw_staging_alias, str)
                or not raw_staging_alias
                or not isinstance(raw_archive_aliases, list)
                or not raw_archive_aliases
                or any(not isinstance(alias, str) or not alias for alias in raw_archive_aliases)
                or len(set(raw_archive_aliases)) != len(raw_archive_aliases)
                or raw_staging_alias in raw_archive_aliases
            ):
                raise RuntimeError(
                    "accepted G01 review storage root aliases are incomplete or invalid"
                )
            staging_root_alias = raw_staging_alias
            archive_root_aliases_exercised = tuple(sorted(raw_archive_aliases))
        return cls(
            prerequisite_run_label=run_label,
            raw_manifest_sha256=raw_sha,
            selected_backend=str(selected_backend),
            selected_exact_backend="cpu_batch",
            selected_workers=workers,
            repository_revision=revision,
            runtime_identity_sha256=_canonical_sha256(runtime),
            runtime_contract_sha256=campaign_runtime_contract_sha256(runtime),
            runtime_selection_sha256=campaign_runtime_selection_sha256(runtime),
            input_provenance_sha256=_canonical_sha256(_input_lock_payload(inputs)),
            configuration_sha256=config_sha,
            configuration_selection_sha256=(
                config_sha
                if configuration_selection_sha256 is None
                else _sha256(
                    configuration_selection_sha256,
                    "configuration selection digest",
                )
            ),
            native_config_sha256=_canonical_sha256(native),
            native_kernel_config=dict(native),
            candidate_transaction_config_sha256=_canonical_sha256(
                candidate_transaction
            ),
            candidate_transaction_config=dict(candidate_transaction),
            instance_sha256=_instance_hashes(inputs),
            staging_root_alias=staging_root_alias,
            archive_root_aliases_exercised=archive_root_aliases_exercised,
            producer_resource_contract=producer_contract,
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "prerequisite_run_label": self.prerequisite_run_label,
            "raw_manifest_sha256": self.raw_manifest_sha256,
            "selected_backend": self.selected_backend,
            "selected_exact_backend": self.selected_exact_backend,
            "selected_workers": self.selected_workers,
            "repository_revision": self.repository_revision,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "runtime_contract_sha256": self.runtime_contract_sha256,
            "runtime_selection_sha256": self.runtime_selection_sha256,
            "input_provenance_sha256": self.input_provenance_sha256,
            "configuration_sha256": self.configuration_sha256,
            "configuration_selection_sha256": self.configuration_selection_sha256,
            "native_config_sha256": self.native_config_sha256,
            "native_kernel_config": dict(self.native_kernel_config),
            "candidate_transaction_config_sha256": (
                self.candidate_transaction_config_sha256
            ),
            "candidate_transaction_config": dict(
                self.candidate_transaction_config
            ),
            "instance_sha256": dict(sorted(self.instance_sha256.items())),
            "staging_root_alias": self.staging_root_alias,
            "archive_root_aliases_exercised": list(self.archive_root_aliases_exercised),
        }
        if self.producer_resource_contract is not None:
            payload["producer_resource_contract"] = dict(
                self.producer_resource_contract
            )
        return payload

    def with_producer_resource_contract(
        self,
        contract: ProducerResourceContract,
        *,
        formal_recalibration: FormalResourceRecalibrationEvidence | None = None,
    ) -> dict[str, object]:
        """Apply a new Pilot calibration without rewriting predecessor evidence."""

        if self.producer_resource_contract is not None:
            predecessor = ProducerResourceContract.from_dict(
                self.producer_resource_contract
            )
            if predecessor != contract and formal_recalibration is None:
                raise RuntimeError(
                    "producer resource contract differs from the accepted Pilot lock"
                )
            if predecessor != contract:
                assert formal_recalibration is not None
                if formal_recalibration.aggregate_memory_source == "cgroup_v2":
                    aggregate_memory_invalid = (
                        formal_recalibration.replacement_aggregate_peak_memory_bytes
                        != contract.selected_aggregate_peak_rss_bytes
                        or contract.aggregate_memory_limit_bytes
                        < contract.selected_aggregate_peak_rss_bytes
                    )
                else:
                    aggregate_memory_invalid = (
                        contract.selected_aggregate_peak_rss_bytes
                        < predecessor.selected_aggregate_peak_rss_bytes
                        or contract.aggregate_memory_limit_bytes
                        < predecessor.aggregate_memory_limit_bytes
                        or formal_recalibration.predecessor_aggregate_peak_rss_bytes
                        != contract.selected_aggregate_peak_rss_bytes
                    )
                topology_invalid = (
                    formal_recalibration.aggregate_memory_source != "cgroup_v2"
                    and (
                        predecessor.row_group_size != contract.row_group_size
                        or predecessor.queue_depth != contract.queue_depth
                    )
                )
                if (
                    predecessor.selected_workers != contract.selected_workers
                    or predecessor.selected_workers != self.selected_workers
                    or topology_invalid
                    or aggregate_memory_invalid
                    or contract.selected_per_worker_peak_rss_bytes
                    < predecessor.selected_per_worker_peak_rss_bytes
                    or contract.per_worker_memory_limit_bytes
                    < predecessor.per_worker_memory_limit_bytes
                    or formal_recalibration.campaign_geometry_contribution != 0
                    or formal_recalibration.replacement_contract_sha256
                    != _canonical_sha256(contract.to_dict())
                    or formal_recalibration.predecessor_per_worker_peak_rss_bytes
                    != contract.selected_per_worker_peak_rss_bytes
                ):
                    raise RuntimeError(
                        "Formal resource-envelope recalibration changed topology, "
                        "reduced a memory floor, or lacks exact zero-geometry evidence"
                    )
                payload = self.to_dict()
                payload["predecessor_producer_resource_contract"] = predecessor.to_dict()
                payload["producer_resource_contract"] = contract.to_dict()
                payload["producer_resource_recalibration"] = (
                    formal_recalibration.to_dict()
                )
                return payload
            return self.to_dict()
        payload = self.to_dict()
        payload["predecessor_selected_workers"] = self.selected_workers
        payload["selected_workers"] = contract.selected_workers
        payload["producer_resource_contract"] = contract.to_dict()
        return payload

    def verify_planned_storage_roots(
        self,
        *,
        staging_root_alias: str,
        planned_archive_root_aliases: tuple[str, ...],
        migrated_archive_root_aliases: tuple[str, ...] = (),
    ) -> None:
        """Reject Formal storage-root drift before any batch is dispatched."""

        if self.staging_root_alias is None or not self.archive_root_aliases_exercised:
            raise RuntimeError(
                "accepted G01 review does not freeze Formal storage root aliases"
            )
        if staging_root_alias != self.staging_root_alias:
            raise RuntimeError(
                "Formal staging root alias differs from the accepted G01 review"
            )
        if (
            not planned_archive_root_aliases
            or any(
                not isinstance(alias, str) or not alias
                for alias in planned_archive_root_aliases
            )
            or len(set(planned_archive_root_aliases)) != len(planned_archive_root_aliases)
            or staging_root_alias in planned_archive_root_aliases
        ):
            raise RuntimeError("Formal planned archive root alias set is invalid")
        migration_coverage = set(migrated_archive_root_aliases)
        if (
            any(not isinstance(alias, str) or not alias for alias in migration_coverage)
            or staging_root_alias in migration_coverage
        ):
            raise RuntimeError("Formal migrated archive root alias set is invalid")
        covered_aliases = set(self.archive_root_aliases_exercised) | migration_coverage
        if not set(planned_archive_root_aliases).issubset(covered_aliases):
            raise RuntimeError(
                "Formal planned archive root aliases exceed accepted G01 drill "
                "coverage or verified migration coverage"
            )

    def verify_current_execution(
        self,
        *,
        selected_backend: str,
        selected_exact_backend: str,
        selected_workers: int,
        repository_revision: str,
        configuration_sha256: str,
        configuration_selection_sha256: str | None = None,
        predecessor_configuration_selection_sha256: str | None = None,
        migrated_archive_roots_verified: bool = False,
        runtime_identity: object,
        input_provenance: object,
        native_kernel_config: object,
        producer_resource_contract: ProducerResourceContract | None = None,
        formal_recalibration: FormalResourceRecalibrationEvidence | None = None,
        repository: Path | None = None,
        storage_migration: Mapping[str, object] | None = None,
    ) -> None:
        """Reject any execution drift from the accepted predecessor lock."""

        if selected_backend != self.selected_backend:
            raise RuntimeError("benchmark execution backend differs from accepted selection")
        if selected_exact_backend != self.selected_exact_backend:
            raise RuntimeError("benchmark exact backend differs from accepted selection")
        expected_workers = (
            producer_resource_contract.selected_workers
            if producer_resource_contract is not None
            else self.selected_workers
        )
        if selected_workers != expected_workers:
            raise RuntimeError("benchmark worker count differs from its frozen selection")
        if self.producer_resource_contract is not None:
            if producer_resource_contract is None:
                raise RuntimeError(
                    "benchmark producer resource contract differs from accepted Pilot"
                )
            self.with_producer_resource_contract(
                producer_resource_contract,
                formal_recalibration=formal_recalibration,
            )
        observed_configuration_selection = (
            configuration_sha256
            if configuration_selection_sha256 is None
            else configuration_selection_sha256
        )
        configuration_matches = (
            observed_configuration_selection == self.configuration_selection_sha256
            or (
                migrated_archive_roots_verified
                and predecessor_configuration_selection_sha256
                == self.configuration_selection_sha256
            )
        )
        if not configuration_matches:
            raise RuntimeError("benchmark configuration differs from accepted selection")
        current_runtime = _mapping(runtime_identity, "runtime identity")
        if repository_revision == self.repository_revision:
            if (
                campaign_runtime_contract_sha256(current_runtime)
                != self.runtime_contract_sha256
            ):
                raise RuntimeError(
                    "benchmark runtime contract differs from accepted selection"
                )
        else:
            if repository is None:
                raise RuntimeError(
                    "benchmark successor revision requires an auditable repository"
                )
            verify_campaign_successor_revision(
                repository,
                predecessor_revision=self.repository_revision,
                current_revision=repository_revision,
            )
            current_selection_sha256 = campaign_runtime_selection_sha256(
                current_runtime
            )
            if (
                current_selection_sha256 != self.runtime_selection_sha256
                and storage_migration is not None
            ):
                current_selection_sha256 = campaign_runtime_selection_sha256(
                    _runtime_selection_with_attested_archive_source(
                        current_runtime,
                        storage_migration,
                    )
                )
            if current_selection_sha256 != self.runtime_selection_sha256:
                raise RuntimeError(
                    "benchmark runtime selection differs from accepted selection"
                )
        current_inputs = _mapping(input_provenance, "input provenance")
        if _canonical_sha256(_input_lock_payload(current_inputs)) != self.input_provenance_sha256:
            raise RuntimeError("benchmark input provenance differs from accepted selection")
        current_instance_hashes = _instance_hashes(current_inputs)
        if any(
            current_instance_hashes.get(instance) != digest
            for instance, digest in self.instance_sha256.items()
        ):
            raise RuntimeError("benchmark instance inputs differ from accepted selection")
        if (
            _canonical_sha256(_mapping(native_kernel_config, "native kernel config"))
            != self.native_config_sha256
        ):
            raise RuntimeError("benchmark native configuration differs from accepted selection")


def load_benchmark_execution_lock(
    prerequisite_dir: Path,
    *,
    expected_scope: str,
    expected_status: str,
) -> BenchmarkExecutionLock:
    """Load the verified producer metadata and its current accepted review."""

    reader = ArtifactReader(prerequisite_dir)
    references = [
        item
        for item in reader.manifest.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("artifact_type") == "manifest_metadata"
    ]
    if len(references) != 1:
        raise RuntimeError("accepted benchmark predecessor lacks one metadata artifact")
    relative_path = references[0].get("relative_path")
    if not isinstance(relative_path, str):
        raise RuntimeError("accepted benchmark metadata path is invalid")
    metadata = reader.read_json(relative_path)
    config_references = [
        item
        for item in reader.manifest.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("artifact_type") == "config"
    ]
    if len(config_references) != 1:
        raise RuntimeError("accepted benchmark predecessor lacks one configuration artifact")
    config_relative_path = config_references[0].get("relative_path")
    if not isinstance(config_relative_path, str):
        raise RuntimeError("accepted benchmark configuration path is invalid")
    config_content = (prerequisite_dir / config_relative_path).read_bytes()
    review_path = prerequisite_dir / "review" / "review_manifest.json"
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("accepted benchmark review manifest is unreadable") from error
    if not isinstance(review, Mapping):
        raise RuntimeError("accepted benchmark review manifest must be an object")
    run_label = review.get("run_label")
    if (
        not isinstance(run_label, str)
        or re.fullmatch(
            r"stage05\.2_[a-z0-9_]+_(?:attempt|rerun)[0-9]{2}",
            run_label,
        )
        is None
    ):
        raise RuntimeError("accepted benchmark review run label is invalid")
    expected_review_schemas = (
        {
            "stage05.2-campaign-review-v1",
            "stage05.2-campaign-review-v2",
        }
        if expected_scope == "pilot"
        else {"stage05.2-review-v1"}
    )
    expected_review_component = (
        "benchmark" if expected_scope == "pilot" else "accelerator_pilot"
    )
    if (
        review.get("schema_version") not in expected_review_schemas
        or review.get("component") != expected_review_component
    ):
        raise RuntimeError("accepted benchmark review schema/component is invalid")
    if expected_scope == "pilot":
        verify_stage052_campaign_gate_set(review, scope="pilot")
    else:
        gates = review.get("gates")
        if (
            not isinstance(gates, Mapping)
            or not gates
            or any(
                not isinstance(gate, Mapping) or gate.get("passed") is not True
                for gate in gates.values()
            )
        ):
            raise RuntimeError("accepted benchmark review contains a failed or invalid gate")
    verified_review_files = verify_stage052_review_files(prerequisite_dir, review)
    if any(
        len(Path(relative).parts) != 3
        or Path(relative).parts[0] != "generations"
        for relative in verified_review_files
    ):
        raise RuntimeError("accepted benchmark review must use an immutable generation")
    if metadata.get("persistence_attribution") == "primary_active_writes_v1":
        attribution_path = (
            prerequisite_dir
            / "control"
            / f"{run_label}_persistence_attribution.json"
        )
        attribution_sidecar = attribution_path.with_suffix(".sha256")
        if (
            not attribution_path.is_file()
            or not attribution_sidecar.is_file()
            or not signed_sidecar_matches(attribution_path, attribution_sidecar)
            or review.get("persistence_attribution_sha256")
            != _file_sha256(attribution_path)
            or review.get("persistence_attribution_sidecar_sha256")
            != _file_sha256(attribution_sidecar)
        ):
            raise RuntimeError("accepted benchmark persistence attribution is stale")
    manifest_path = reader.result.manifest_path
    if expected_scope == "pilot":
        campaign_path = prerequisite_dir / "campaign_manifest.json"
        campaign = load_campaign_manifest(campaign_path)
        if (
            campaign.status != "complete"
            or campaign.scope != "pilot"
            or campaign.run_label != run_label
            or review.get("raw_campaign_manifest_sha256")
            != _file_sha256(campaign_path)
        ):
            raise RuntimeError("accepted G01 campaign manifest binding is stale")
    return BenchmarkExecutionLock.from_accepted_evidence(
        metadata=metadata,
        review_manifest=review,
        raw_manifest_sha256=_file_sha256(manifest_path),
        expected_scope=expected_scope,
        expected_status=expected_status,
        configuration_selection_sha256=campaign_configuration_selection_sha256(
            config_content
        ),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def probe_volume_identity(path: Path) -> VolumeIdentity:
    """Return the mounted volume UUID/filesystem for one local root."""

    if sys.platform == "darwin":
        return _probe_macos_volume_identity(path)
    completed = subprocess.run(
        (
            "findmnt",
            "--json",
            "--target",
            str(path),
            "--output",
            "SOURCE,FSTYPE,UUID",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
        filesystems = payload["filesystems"]
        mount = filesystems[0]
        source = str(mount["source"])
        filesystem = str(mount["fstype"])
        uuid_value = mount.get("uuid")
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"findmnt returned invalid volume metadata for {path}") from error
    if isinstance(uuid_value, str) and uuid_value:
        return VolumeIdentity(device_uuid=uuid_value, filesystem=filesystem)
    drive_match = re.fullmatch(r"([A-Za-z]):\\", source)
    if drive_match is None:
        raise RuntimeError(f"mounted volume identity is incomplete for {path}: {source}")
    return VolumeIdentity(
        device_uuid=_windows_disk_identity(drive_match.group(1)),
        filesystem=filesystem,
    )


def _windows_disk_identity(drive_letter: str) -> str:
    powershell = shutil.which("pwsh.exe") or shutil.which("powershell.exe")
    if powershell is None:
        raise RuntimeError("Windows drive identity requires PowerShell")
    script = (
        f"$partition = Get-Partition -DriveLetter '{drive_letter}'; "
        "$disk = $partition | Get-Disk; "
        f"$volume = Get-Volume -DriveLetter '{drive_letter}'; "
        "[pscustomobject]@{FriendlyName=$disk.FriendlyName;"
        "SerialNumber=$disk.SerialNumber;BusType=[string]$disk.BusType;"
        "FileSystem=[string]$volume.FileSystem} "
        "| ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        (powershell, "-NoLogo", "-NoProfile", "-Command", script),
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(completed.stdout)
        raw_friendly_name = payload["FriendlyName"]
        raw_serial_number = payload["SerialNumber"]
        raw_bus_type = payload["BusType"]
        raw_filesystem = payload["FileSystem"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("PowerShell returned invalid Windows disk identity") from error
    if not all(
        isinstance(value, str) and value.strip()
        for value in (
            raw_friendly_name,
            raw_serial_number,
            raw_bus_type,
            raw_filesystem,
        )
    ):
        raise RuntimeError("Windows archive must resolve to an identified physical disk")
    friendly_name = raw_friendly_name.strip()
    serial_number = raw_serial_number.strip()
    bus_type = raw_bus_type.strip()
    filesystem = raw_filesystem.strip()
    if bus_type.casefold() in {"unknown", "file backed virtual", "virtual"}:
        raise RuntimeError("Windows archive must use an identified physical disk")
    if filesystem.casefold() != "ntfs":
        raise RuntimeError("Windows archive backing filesystem must be NTFS")
    identity = re.sub(
        r"[^a-z0-9]+",
        "-",
        f"{bus_type}-{friendly_name}-{serial_number}".casefold(),
    ).strip("-")
    if not identity:
        raise RuntimeError("Windows physical-disk identity is empty")
    return identity


def _probe_macos_volume_identity(path: Path) -> VolumeIdentity:
    filesystem_result = subprocess.run(
        ("df", "-P", str(path)),
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in filesystem_result.stdout.splitlines() if line.strip()]
    if len(lines) != 2:
        raise RuntimeError(f"df returned invalid mount metadata for {path}")
    device = lines[1].split(maxsplit=1)[0]
    if not device.startswith("/dev/"):
        raise RuntimeError(f"df returned an invalid device for {path}: {device}")
    completed = subprocess.run(
        ("diskutil", "info", "-plist", device),
        check=True,
        capture_output=True,
    )
    try:
        payload = plistlib.loads(completed.stdout)
    except plistlib.InvalidFileException as error:
        raise RuntimeError(f"diskutil returned invalid volume metadata for {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"diskutil returned invalid volume metadata for {path}")
    device_uuid = payload.get("VolumeUUID") or payload.get("DiskUUID")
    filesystem = payload.get("FilesystemName") or payload.get("FilesystemType")
    if not isinstance(device_uuid, str) or not isinstance(filesystem, str):
        raise RuntimeError(f"volume identity is incomplete for {path}")
    return VolumeIdentity(device_uuid=device_uuid, filesystem=filesystem)


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def verify_campaign_root_locations(
    *,
    repository_root: Path,
    locator: StorageRootLocator,
    staging_root_alias: str = "wsl_staging",
    archive_root_aliases: tuple[str, ...] | None = None,
) -> None:
    """Reject local locator drift before creating or writing any campaign root."""

    del repository_root
    selected_archives = (
        tuple(
            alias
            for alias in locator.aliases
            if alias != staging_root_alias and alias.endswith("_archive")
        )
        if archive_root_aliases is None
        else archive_root_aliases
    )
    if not selected_archives or len(set(selected_archives)) != len(selected_archives):
        raise RuntimeError("Stage 5.2 requires unique archive root aliases")
    staging = locator.resolve(staging_root_alias)
    staging_path = staging.absolute_path.resolve()
    if staging.volume.filesystem.casefold() != "ext4" or staging_path.is_relative_to(
        Path("/mnt")
    ):
        raise RuntimeError("Stage 5.2 wsl_staging must be on WSL2 native ext4")
    for alias in selected_archives:
        archive = locator.resolve(alias)
        archive_path = archive.absolute_path.resolve()
        filesystem = archive.volume.filesystem.casefold()
        if filesystem in {"exfat", "vfat", "fat", "fat32"}:
            raise RuntimeError("Stage 5.2 archive cannot use ExFAT or FAT storage")
        if filesystem not in {"9p", "ntfs"}:
            raise RuntimeError("Stage 5.2 archive must resolve to NTFS through WSL")
        if archive.volume == staging.volume or archive_path == staging_path:
            raise RuntimeError("Stage 5.2 staging and archive roots must be distinct")
        if os.name != "nt" and not archive_path.is_relative_to(Path("/mnt")):
            raise RuntimeError("Stage 5.2 WSL archive must resolve below /mnt")


def verify_rolling_campaign_capacity(
    *,
    config: BenchmarkCampaignConfig,
    campaign: CampaignManifest,
    locator: StorageRootLocator,
    batch_id: str,
    phase: str,
    free_space: Callable[[Path], int] = free_bytes,
    volume_probe: Callable[[Path], VolumeIdentity] = probe_volume_identity,
) -> dict[str, object]:
    """Re-probe capacity before dispatch/archive and preserve every reserve."""

    if phase not in {"pre_dispatch", "pre_archive", "post_archive"}:
        raise ValueError("rolling capacity phase is invalid")
    if campaign.run_label != config.run_label or campaign.scope != config.scope:
        raise ValueError("rolling capacity campaign/config identity mismatch")
    batches = {batch.batch_id: batch for batch in campaign.batches}
    current = batches.get(batch_id)
    if current is None:
        raise ValueError("rolling capacity batch is not in the campaign")
    expected_status = {
        "pre_dispatch": "planned",
        "pre_archive": "verified",
        "post_archive": "archived",
    }[phase]
    if current.status != expected_status:
        raise RuntimeError(
            f"rolling capacity {phase} requires {expected_status} batch state"
        )

    aliases = tuple(
        dict.fromkeys((config.staging_root_alias, *config.archive_root_aliases))
    )
    operational_locator = locator.with_observed_volumes(volume_probe, aliases)
    roots = {alias: operational_locator.resolve(alias) for alias in aliases}
    free_by_device: dict[str, int] = {}
    for root in roots.values():
        measured = free_space(root.absolute_path)
        if isinstance(measured, bool) or not isinstance(measured, int) or measured < 0:
            raise RuntimeError("rolling capacity free-byte measurement is invalid")
        device = root.volume.device_uuid
        free_by_device[device] = min(free_by_device.get(device, measured), measured)

    staging_device = roots[config.staging_root_alias].volume.device_uuid
    required_by_device: dict[str, int] = {}
    for alias in config.archive_root_aliases:
        root = roots[alias]
        device = root.volume.device_uuid
        reserve = (
            config.external_safety_reserve_bytes
            + config.external_active_workspace_bytes
            if device == staging_device
            else config.internal_safety_reserve_bytes
        )
        required_by_device[device] = max(required_by_device.get(device, 0), reserve)

    current_index = tuple(batch.batch_id for batch in campaign.batches).index(batch_id)
    future = campaign.batches[current_index + 1 :]
    if any(batch.status != "planned" for batch in future):
        raise RuntimeError("rolling capacity future batch states are not planned")
    projected = campaign.batches[current_index:] if phase == "pre_dispatch" else future
    for batch in projected:
        device = roots[batch.archive_root_alias].volume.device_uuid
        required_by_device[device] = (
            required_by_device.get(device, 0) + batch.estimated_bytes
        )

    if phase == "pre_archive":
        if current.actual_bytes is None:
            raise RuntimeError("verified batch has no actual byte count")
        target_device = roots[current.archive_root_alias].volume.device_uuid
        if target_device != staging_device:
            required_by_device[target_device] = (
                required_by_device.get(target_device, 0) + current.actual_bytes
            )
        staging_required = config.external_safety_reserve_bytes
        if future:
            if target_device == staging_device:
                staging_required = (
                    config.external_safety_reserve_bytes
                    + config.external_active_workspace_bytes
                )
            else:
                staging_required = max(
                    staging_required,
                    config.external_safety_reserve_bytes
                    + config.external_active_workspace_bytes
                    - current.actual_bytes,
                )
        required_by_device[staging_device] = max(
            required_by_device.get(staging_device, 0),
            staging_required,
        )
    else:
        staging_reserve = config.external_safety_reserve_bytes
        if phase == "pre_dispatch" or future:
            staging_reserve += config.external_active_workspace_bytes
        required_by_device[staging_device] = max(
            required_by_device.get(staging_device, 0),
            staging_reserve,
        )

    deficits = {
        device: required - free_by_device.get(device, 0)
        for device, required in required_by_device.items()
        if free_by_device.get(device, 0) < required
    }
    if deficits:
        observation = {
            "schema_version": "stage05.2-rolling-capacity-v1",
            "run_label": campaign.run_label,
            "batch_id": batch_id,
            "phase": phase,
            "free_bytes_by_device": dict(sorted(free_by_device.items())),
            "required_bytes_by_device": dict(sorted(required_by_device.items())),
            "deficits_by_device": dict(sorted(deficits.items())),
            "passed": False,
        }
        raise RollingCampaignCapacityError(
            observation=observation,
            message=(
                f"rolling campaign capacity cannot preserve reserves at {phase}: "
                f"{deficits}"
            ),
        )
    return {
        "schema_version": "stage05.2-rolling-capacity-v1",
        "run_label": campaign.run_label,
        "batch_id": batch_id,
        "phase": phase,
        "free_bytes_by_device": dict(sorted(free_by_device.items())),
        "required_bytes_by_device": dict(sorted(required_by_device.items())),
        "passed": True,
    }


class RollingCampaignCapacityError(RuntimeError):
    """A failed reserve check carrying the complete durable observation."""

    def __init__(self, *, observation: Mapping[str, object], message: str) -> None:
        super().__init__(message)
        self.observation = dict(observation)


def campaign_control_paths(run_dir: Path, run_label: str) -> dict[str, Path]:
    control = run_dir / "control"
    return {
        "campaign_manifest": run_dir / "campaign_manifest.json",
        "campaign_plan": control / f"{run_label}_campaign_plan.json",
        "preflight": control / f"{run_label}_campaign_preflight.json",
        "archive_dry_run": control / f"{run_label}_archive_dry_run.json",
        "publication_dry_run": control / f"{run_label}_publication_dry_run.json",
        "failure_state_drill": control / f"{run_label}_failure_state_drill.json",
        "raw_replay_drill": control / f"{run_label}_raw_replay_drill.json",
        "rolling_capacity": control / f"{run_label}_rolling_capacity.json",
        "persistence_attribution": control
        / f"{run_label}_persistence_attribution.json",
    }


def persist_campaign_manifest(run_dir: Path, manifest: CampaignManifest) -> Path:
    path = campaign_control_paths(run_dir, manifest.run_label)["campaign_manifest"]
    atomic_write_signed_json(path, manifest.to_dict())
    loaded = load_campaign_manifest(path)
    if loaded != manifest:
        raise RuntimeError("atomic campaign manifest read-back mismatch")
    return path


class BatchRuntimeMonitor:
    """Continuously sample batch power/load and expose a pool-abort reason."""

    def __init__(
        self,
        config: BenchmarkCampaignConfig,
        *,
        snapshot: Callable[[], MachineSnapshot],
        interval_seconds: float = 1.0,
    ) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("batch runtime sampling interval must be positive")
        self._config = config
        self._snapshot = snapshot
        self._interval_seconds = interval_seconds
        self._samples: list[MachineSnapshot] = []
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("batch runtime monitor was already started")
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="stage052-batch-runtime-monitor",
            daemon=True,
        )
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self._snapshot()
            except BaseException as error:
                with self._lock:
                    self._error = error
                self._stop.set()
                return
            with self._lock:
                self._samples.append(sample)
            if not BatchRuntimeEvidence.from_snapshots((sample,), config=self._config).passed:
                self._stop.set()
                return
            self._stop.wait(self._interval_seconds)

    def abort_reason(self) -> str | None:
        with self._lock:
            if self._error is not None:
                return f"batch runtime sampling failed: {type(self._error).__name__}: {self._error}"
            samples = tuple(self._samples)
        if not samples:
            return None
        evidence = BatchRuntimeEvidence.from_snapshots(samples, config=self._config)
        return None if evidence.passed else evidence.failure_reason

    def stop(self) -> BatchRuntimeEvidence:
        if self._thread is None:
            raise RuntimeError("batch runtime monitor was not started")
        self._stop.set()
        self._thread.join(timeout=max(5.0, self._interval_seconds * 2.0))
        if self._thread.is_alive():
            raise RuntimeError("batch runtime monitor did not stop")
        with self._lock:
            error = self._error
            samples = tuple(self._samples)
        if error is not None:
            raise RuntimeError(
                f"batch runtime sampling failed: {type(error).__name__}: {error}"
            ) from error
        if not samples:
            raise RuntimeError("batch runtime monitor captured no samples")
        return BatchRuntimeEvidence.from_snapshots(samples, config=self._config)


def load_pilot_storage_observations(
    prerequisite_dir: Path,
    *,
    locator: StorageRootLocator,
) -> tuple[PilotStorageObservation, ...]:
    """Re-verify archived G01 bytes and derive G02 next-fit estimates."""

    manifest = load_campaign_manifest(prerequisite_dir / "campaign_manifest.json")
    run_label = manifest.run_label
    paths = campaign_control_paths(prerequisite_dir, run_label)
    if manifest.scope != "pilot" or manifest.status != "complete":
        raise RuntimeError("G02 requires one complete accepted G01 campaign")
    expected_roots = set(manifest.storage_roots)
    if not expected_roots.issubset(locator.aliases):
        raise RuntimeError("G01/G02 storage root aliases are unavailable")
    for alias in sorted(expected_roots):
        if not locator.resolve(alias).absolute_path.is_dir():
            raise RuntimeError(f"G01/G02 storage root is unavailable: {alias}")
    try:
        plan_payload = json.loads(paths["campaign_plan"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("accepted G01 campaign plan is unreadable") from error
    if not isinstance(plan_payload, Mapping) or (
        plan_payload.get("scope") != "pilot"
        or plan_payload.get("shard_count") != 36
        or plan_payload.get("axis_count") != 36
        or plan_payload.get("checkpoint_count") != 144
    ):
        raise RuntimeError("accepted G01 campaign plan geometry is invalid")
    shards = plan_payload.get("shards")
    if not isinstance(shards, list) or len(shards) != 36:
        raise RuntimeError("accepted G01 campaign plan shard list is invalid")
    actual_by_shard: dict[str, int] = {}
    migrated_batch_root = prerequisite_dir.parent / "d_benchmark"
    segmented_retention = (
        prerequisite_dir.name == "wsl_active" and migrated_batch_root.is_dir()
    )
    for batch in manifest.batches:
        if batch.status != "archived" or batch.shard_actual_bytes_by_id is None:
            raise RuntimeError("accepted G01 batch is not archived with shard bytes")
        logical_parts = Path(batch.logical_path).parts
        if logical_parts != (run_label, batch.batch_id):
            raise RuntimeError("accepted G01 batch logical path is invalid")
        if segmented_retention:
            batch_path = migrated_batch_root / batch.batch_id
        else:
            archive_root = locator.resolve(batch.root_alias)
            batch_path = archive_root.absolute_path.joinpath(*logical_parts)
        if (
            directory_checksum(batch_path) != batch.checksum_sha256
            or directory_byte_count(batch_path) != batch.actual_bytes
        ):
            raise RuntimeError(f"accepted G01 archived batch checksum failed: {batch.batch_id}")
        overlap = set(actual_by_shard).intersection(batch.shard_actual_bytes_by_id)
        if overlap:
            raise RuntimeError(f"accepted G01 shard bytes are duplicate: {sorted(overlap)}")
        actual_by_shard.update(batch.shard_actual_bytes_by_id)
    observations: list[PilotStorageObservation] = []
    for raw in shards:
        if not isinstance(raw, Mapping):
            raise RuntimeError("accepted G01 campaign shard is invalid")
        shard_id = raw.get("shard_id")
        family = raw.get("family")
        customer_count = raw.get("customer_count")
        budgets = raw.get("budgets_seconds")
        if (
            not isinstance(shard_id, str)
            or not isinstance(family, str)
            or isinstance(customer_count, bool)
            or not isinstance(customer_count, int)
            or budgets != [30]
            or shard_id not in actual_by_shard
        ):
            raise RuntimeError("accepted G01 campaign shard storage identity is invalid")
        observations.append(
            PilotStorageObservation(
                family=family,
                customer_count=customer_count,
                budget_seconds=30,
                compressed_bytes=actual_by_shard[shard_id],
            )
        )
    if len(observations) != 36:
        raise RuntimeError("accepted G01 storage observations are incomplete")
    return tuple(observations)


def archive_verified_batch(
    *,
    batch: BatchManifest,
    locator: StorageRootLocator,
) -> BatchManifest:
    """Archive a verified batch; retained for non-campaign dry-run callers."""

    return archive_verified_batch_with_evidence(batch=batch, locator=locator).batch


@dataclass(frozen=True, slots=True)
class ArchivedBatchEvidence:
    """Archived state plus the separately timed final manifest write."""

    batch: BatchManifest
    manifest_sha256: str
    state_write_interval: PersistenceInterval
    manifest_path: Path


class ArchivedBatchStateWriteError(RuntimeError):
    """Archive transfer completed, but its final signed state could not be sealed."""

    def __init__(
        self,
        *,
        batch: BatchManifest,
        destination: Path,
        cause: Exception,
    ) -> None:
        super().__init__(
            "archived batch transfer completed but final state write failed: "
            f"{type(cause).__name__}: {cause}"
        )
        self.batch = batch
        self.destination = destination


def archive_verified_batch_with_evidence(
    *,
    batch: BatchManifest,
    locator: StorageRootLocator,
    _state_writer: Callable[
        [Path, Mapping[str, object]], tuple[Path, Path]
    ]
    | None = None,
) -> ArchivedBatchEvidence:
    state_writer = atomic_write_signed_json if _state_writer is None else _state_writer
    try:
        archived = BatchArchiver(locator).archive(batch)
    except ArchiveTransferCompletedError as error:
        raise ArchivedBatchStateWriteError(
            batch=error.batch,
            destination=error.destination,
            cause=error,
        ) from error
    destination_root = locator.resolve(archived.root_alias)
    destination = destination_root.absolute_path.joinpath(*Path(archived.logical_path).parts)
    started_ns = time.monotonic_ns()
    try:
        manifest_path, _ = state_writer(
            destination / "batch_manifest.json",
            archived.to_dict(),
        )
        completed_ns = time.monotonic_ns()
        if (
            directory_checksum(destination) != archived.checksum_sha256
            or directory_byte_count(destination) != archived.actual_bytes
        ):
            raise RuntimeError("archived batch changed after manifest state update")
    except Exception as error:
        raise ArchivedBatchStateWriteError(
            batch=archived,
            destination=destination,
            cause=error,
        ) from error
    return ArchivedBatchEvidence(
        batch=archived,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        state_write_interval=PersistenceInterval(
            label="archived_batch_manifest_write",
            started_ns=started_ns,
            completed_ns=completed_ns,
        ),
        manifest_path=manifest_path,
    )


class WindowsWslMachineSnapshotSource:
    """Sample Windows power state plus WSL2 load and unrelated CPU usage."""

    def __init__(self) -> None:
        self._native_status: WindowsWslPowerStatus | None = None

    def refresh_native_status(self) -> WindowsWslPowerStatus:
        """Refresh native Windows state outside the measured runtime interval."""

        observed = read_windows_wsl_power_status()
        self._native_status = observed
        return observed

    def verify_native_status_unchanged(self) -> dict[str, object]:
        """Verify stable native invariants and return both boundary observations."""

        expected = self._native_status
        if expected is None:
            raise RuntimeError("Windows native power status was not sampled at preflight")
        observed = read_windows_wsl_power_status()
        expected_invariants = (
            expected.ac_online,
            expected.battery_saver,
            expected.active_power_scheme,
        )
        observed_invariants = (
            observed.ac_online,
            observed.battery_saver,
            observed.active_power_scheme,
        )
        if observed_invariants != expected_invariants:
            raise RuntimeError(
                "Windows native power state changed during the benchmark batch: "
                f"expected={expected!r} observed={observed!r}"
            )
        return {
            "schema_version": "stage05.2-native-power-boundary-v1",
            "before": _windows_power_status_to_dict(expected),
            "after": _windows_power_status_to_dict(observed),
            "stable_invariants": [
                "ac_online",
                "battery_saver",
                "active_power_scheme",
            ],
            "invariants_unchanged": True,
        }

    def __call__(self) -> MachineSnapshot:
        power = self._native_status
        if power is None:
            power = self.refresh_native_status()
        ac_online = read_wsl_ac_power_online()
        return MachineSnapshot(
            power_source="AC Power" if ac_online else "Battery Power",
            low_power_mode_enabled=power.battery_saver,
            load1=float(os.getloadavg()[0]),
            unrelated_process_average_cores=0.0,
            sampled_at_seconds=time.monotonic(),
            unrelated_process_cpu_seconds=_unrelated_user_cpu_seconds(),
        )


def _parse_process_cpu_time(value: str) -> float:
    day_parts = value.split("-", maxsplit=1)
    days = 0
    clock = value
    if len(day_parts) == 2:
        try:
            days = int(day_parts[0])
        except ValueError as error:
            raise RuntimeError(f"invalid process CPU time: {value}") from error
        clock = day_parts[1]
    parts = clock.split(":")
    try:
        if len(parts) == 2:
            hours = 0
            minutes = int(parts[0])
            seconds = float(parts[1])
        elif len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
        else:
            raise ValueError
    except ValueError as error:
        raise RuntimeError(f"invalid process CPU time: {value}") from error
    return days * 86_400.0 + hours * 3_600.0 + minutes * 60.0 + seconds


def _windows_power_status_to_dict(status: WindowsWslPowerStatus) -> dict[str, object]:
    return {
        "ac_online": status.ac_online,
        "battery_saver": status.battery_saver,
        "battery_life_percent": status.battery_life_percent,
        "battery_flag": status.battery_flag,
        "active_power_scheme": status.active_power_scheme,
    }


def _unrelated_user_cpu_seconds() -> dict[int, float]:
    completed = subprocess.run(
        ("ps", "-axo", "pid=,ppid=,uid=,time="),
        check=True,
        capture_output=True,
        text=True,
    )
    current_pid = os.getpid()
    current_uid = os.getuid()
    records: list[tuple[int, int, int, float]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 4:
            raise RuntimeError(f"invalid non-empty ps process row: {line!r}")
        try:
            records.append(
                (
                    int(fields[0]),
                    int(fields[1]),
                    int(fields[2]),
                    _parse_process_cpu_time(fields[3]),
                )
            )
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"cannot parse ps process row: {line!r}") from error
    related = {current_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent_pid, _, _ in records:
            if parent_pid in related and pid not in related:
                related.add(pid)
                changed = True
    return {
        pid: cpu_seconds
        for pid, _, uid, cpu_seconds in records
        if uid == current_uid and pid not in related
    }
