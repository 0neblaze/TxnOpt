"""Fail-closed helpers shared by the Level 1 campaign producer and reviewer."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import platform
import signal
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from txnopt_evidence.identity import ExpectedEvidenceIdentity
from txnopt_evidence.level1_protocol import validate_level1_protocol_v2

AXES: dict[str, tuple[str, int, int]] = {
    "serial_1": ("serial", 1, 0),
    "txnopt_1": ("ordered", 1, 1),
    "txnopt_4": ("ordered", 4, 4),
    "barrier_4": ("barrier", 4, 0),
}
BUDGETS = ("fixed_work", "fixed_time")
DOMAINS = ("evrptw", "rcpsp")
ANALYSIS_STATUS = "PREREGISTERED_BEFORE_FORMAL_MATRIX"
FIXED_WORK_COMPARISON = (
    "exact canonical objective and semantic digest equality across serial_1, "
    "txnopt_1, txnopt_4, and barrier_4"
)
PAIRING_UNIT = ("domain", "case_id", "seed")
PERFORMANCE_MEASURE = {
    "budget_axis": "fixed_work",
    "duration_field": "txnopt-physical-trace-v1.run_observation.duration_ns",
    "four_worker_ratio": "serial_1_duration_ns / txnopt_4_duration_ns",
    "one_worker_overhead": ("max(0, txnopt_1_duration_ns / serial_1_duration_ns - 1)"),
}


@dataclass(frozen=True, slots=True)
class CampaignEntry:
    ordinal: int
    relative_path: str
    config_path: Path
    config_sha256: str
    expected_identity_relative_path: str | None
    expected_identity_path: Path | None
    expected_identity_sha256: str | None
    run_label: str
    raw_output_root: Path
    domain: str
    case_id: str
    seed: int
    axis: str
    budget: str
    workers: int


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    manifest_path: Path
    manifest_sha256: str
    payload: dict[str, Any]
    protocol_path: Path
    protocol_sha256: str
    raw_output_root: Path
    build_manifest_path: Path
    build_manifest_sha256: str
    expected_identity_tree_sha256: str | None
    entries: tuple[CampaignEntry, ...]


@dataclass(frozen=True, slots=True)
class IsolatedProcessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    descendant_cleanup_performed: bool
    descendant_processes_remaining: tuple[int, ...]


def canonical_json_bytes(payload: object, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {"allow_nan": False, "sort_keys": True}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(payload, **options) + ("\n" if pretty else "")).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sidecar(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"signed artifact cannot be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"signed artifact must be a regular file: {path}")
    digest = sha256_file(resolved)
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.is_symlink():
        raise ValueError(f"signed sidecar is missing: {sidecar}")
    if sidecar.read_text(encoding="utf-8").strip().split() != [digest, resolved.name]:
        raise ValueError(f"signed sidecar differs: {resolved}")
    return digest


def read_signed_object(path: Path, *, schema_version: str | None = None) -> dict[str, Any]:
    verify_sidecar(path)
    payload: object = json.loads(path.read_bytes())
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"signed JSON must contain an object: {path}")
    if schema_version is not None and payload.get("schema_version") != schema_version:
        raise ValueError(f"unsupported schema in {path}")
    return payload


def write_signed_object(path: Path, payload: object) -> str:
    data = canonical_json_bytes(payload, pretty=True)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or path.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError(f"refusing to overwrite signed artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
    digest = sha256_bytes(data)
    with sidecar.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {path.name}\n")
        handle.flush()
    return digest


def load_campaign_plan(path: Path) -> CampaignPlan:
    manifest_path = path.resolve(strict=True)
    manifest_sha256 = verify_sidecar(manifest_path)
    plan = _object(json.loads(manifest_path.read_bytes()), "campaign plan")
    schema_version = plan.get("schema_version")
    if (
        schema_version
        not in {"txnopt-level1-campaign-plan-v1", "txnopt-level1-campaign-plan-v2"}
        or plan.get("status") != "PLANNED_NOT_STARTED"
        or plan.get("holdout_opened") is not False
        or plan.get("cloud_purchase_authorized") is not False
    ):
        raise ValueError("campaign plan is not a closed, unstarted Level 1 plan")

    protocol_path = _bound_file(plan, "protocol_path", "protocol_sha256")
    protocol_sha256 = sha256_file(protocol_path)
    protocol = _object(json.loads(protocol_path.read_bytes()), "Level 1 protocol")
    protocol_schema = protocol.get("schema_version")
    if protocol_schema == "txnopt-level1-protocol-v1":
        if (
            protocol.get("holdout_opened") is not False
            or protocol.get("level2_holdout_opened") is not False
            or protocol.get("level3_holdout_opened") is not False
        ):
            raise ValueError("Level 1 protocol or holdout boundary differs")
    elif protocol_schema == "txnopt-level1-protocol-v2":
        validate_level1_protocol_v2(protocol)
    else:
        raise ValueError("Level 1 protocol schema differs")
    verify_sidecar(protocol_path)
    catalog_path = _bound_file(plan, "catalog_path", "catalog_sha256")
    catalog = _object(json.loads(catalog_path.read_bytes()), "case catalog")
    if catalog.get("schema_version") != "txnopt-level1-case-catalog-v1":
        raise ValueError("campaign case catalog schema differs")

    build_manifest_path = _bound_file(plan, "build_manifest_path", "build_manifest_sha256")
    build_manifest_sha256 = verify_sidecar(build_manifest_path)
    build = read_signed_object(
        build_manifest_path,
        schema_version="txnopt-level1-build-manifest-v1",
    )
    producer = _object(build.get("producer"), "build producer")
    artifacts = _object(build.get("artifacts"), "build artifacts")
    native = _object(artifacts.get("native_extension"), "build native extension")
    resource_soak = _object(artifacts.get("resource_soak"), "build resource soak")
    validation = _object(build.get("validation"), "build validation")
    if (
        producer.get("source_dirty") is not False
        or producer.get("development_override") is not False
        or native.get("protocol") != "txnopt-native-round-v1"
        or resource_soak.get("status") != "PASS"
        or resource_soak.get("fallback_count") != 0
    ):
        raise ValueError("campaign build is not a clean TxnOpt native-round producer")
    run_label = build.get("run_label")
    status = build.get("status")
    additional_gates: tuple[str, ...]
    if run_label == "txnopt_level1_build_attempt09":
        formal_package = _object(build.get("formal_package"), "build formal package")
        if (
            status != "BUILD_AND_FORMAL_PACKAGE_COMPLETE_NOT_LEVEL1_READY"
            or formal_package.get("unresolved_critical_formal_findings") != 0
            or formal_package.get("aggregate_refinement_replay") != "PASS"
        ):
            raise ValueError("Build09 formal package is not independently accepted")
        surface_gate = "wheel_surface"
        additional_gates = ("formal_replay",)
    elif run_label == "txnopt_level1_build_attempt10":
        formal_successor = _object(build.get("formal_successor"), "build formal successor")
        if (
            status != "BUILD_COMPLETE_FORMAL_SUCCESSOR_REVIEW_PENDING_NOT_LEVEL1_READY"
            or formal_successor.get("prior_review_binding_status") != "PRIOR_SOURCE_ONLY"
            or formal_successor.get("successor_status") != "REVIEW_PENDING_BUILD10"
            or formal_successor.get("independent_successor_review_completed") is not False
            or formal_successor.get("level1_formal_gate_passed") is not False
        ):
            raise ValueError("Build10 formal-successor boundary differs")
        surface_gate = "wheel_surface_and_record"
        additional_gates = ()
    elif run_label == "txnopt_level1_build_attempt11":
        formal_successor = _object(build.get("formal_successor"), "build formal successor")
        if (
            status
            != "BUILD_COMPLETE_EVIDENCE_LIFECYCLE_REVIEW_PENDING_NOT_LEVEL1_READY"
            or formal_successor.get("prior_review_binding_status") != "PRIOR_SOURCE_ONLY"
            or formal_successor.get("successor_status") != "REVIEW_PENDING_BUILD11"
            or formal_successor.get("independent_successor_review_completed") is not False
            or formal_successor.get("level1_formal_gate_passed") is not False
        ):
            raise ValueError("Build11 formal-successor boundary differs")
        surface_gate = "wheel_surface_and_record"
        additional_gates = ()
    elif run_label == "txnopt_level1_build_attempt14":
        formal_successor = _object(build.get("formal_successor"), "build formal successor")
        if (
            status
            != "BUILD_COMPLETE_ANCHORED_EVIDENCE_REVIEW_PENDING_NOT_LEVEL1_READY"
            or formal_successor.get("prior_review_binding_status") != "PRIOR_SOURCE_ONLY"
            or formal_successor.get("successor_status") != "REVIEW_PENDING_BUILD14"
            or formal_successor.get("independent_successor_review_completed") is not False
            or formal_successor.get("level1_formal_gate_passed") is not False
        ):
            raise ValueError("Build14 formal-successor boundary differs")
        surface_gate = "wheel_surface_and_record"
        additional_gates = ()
    elif run_label == "txnopt_level1_build_attempt16":
        formal_successor = _object(build.get("formal_successor"), "build formal successor")
        if (
            status != "BUILD_COMPLETE_TENCENT_CLOUD_CUTOVER_NOT_LEVEL1_READY"
            or formal_successor.get("prior_review_binding_status") != "PRIOR_SOURCE_ONLY"
            or formal_successor.get("successor_status") != "REVIEW_PENDING_BUILD16"
            or formal_successor.get("independent_successor_review_completed") is not False
            or formal_successor.get("level1_formal_gate_passed") is not False
        ):
            raise ValueError("Build16 Tencent-cloud boundary differs")
        surface_gate = "wheel_surface_and_record"
        additional_gates = (
            "formal_contract_tests",
            "formal_model_receipt",
            "native_source_identity",
            "tencent_interface",
            "toolchain_lock",
        )
    else:
        raise ValueError("campaign build identity is not an approved Level 1 producer")
    if schema_version == "txnopt-level1-campaign-plan-v2":
        if (
            protocol_schema == "txnopt-level1-protocol-v1"
            and run_label != "txnopt_level1_build_attempt14"
        ):
            raise ValueError("campaign plan v2 requires the approved Build14 producer")
        if (
            protocol_schema == "txnopt-level1-protocol-v2"
            and run_label != "txnopt_level1_build_attempt16"
        ):
            raise ValueError("campaign protocol v2 requires the approved Build16 producer")
    for gate in (
        "ruff",
        "strict_mypy",
        "pytest_wheel_installed",
        "property_based_tests",
        surface_gate,
        "legacy_verify",
        "asan_ubsan",
        "tsan",
        *additional_gates,
    ):
        gate_record = _object(validation.get(gate), f"build validation {gate}")
        if gate_record.get("status") != "PASS" or gate_record.get("exit_code") != 0:
            raise ValueError(f"campaign build validation did not pass: {gate}")
    historical_gate = _object(
        validation.get("historical_protected_paths"),
        "historical protected paths",
    )
    if historical_gate.get("status") != "UNCHANGED" or historical_gate.get("exit_code") != 0:
        raise ValueError("campaign build changed a historical protected path")

    raw_output = plan.get("raw_output_root")
    if not isinstance(raw_output, str) or not raw_output:
        raise ValueError("campaign raw output root is missing")
    raw_output_root = Path(raw_output).expanduser().resolve()
    expected_axes = _strings(protocol, "formal_axes")
    if tuple(expected_axes) != tuple(AXES):
        raise ValueError("campaign formal axes differ from the canonical Level 1 axes")
    if tuple(_strings(protocol, "budgets")) != BUDGETS:
        raise ValueError("campaign budget axes differ")
    seeds = _integers(protocol, "seeds")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("campaign seeds are missing or duplicated")
    case_ids = _protocol_case_ids(protocol)
    catalog_sources = _catalog_sources(catalog, case_ids)

    raw_entries = plan.get("entries")
    config_count = plan.get("config_count")
    if (
        not isinstance(raw_entries, list)
        or isinstance(config_count, bool)
        or not isinstance(config_count, int)
        or config_count != len(raw_entries)
    ):
        raise ValueError("campaign config count differs from its entries")
    expected_count = sum(len(values) for values in case_ids.values()) * len(seeds) * 8
    if config_count != expected_count:
        raise ValueError("campaign matrix size differs from the preregistered scope")

    entries: list[CampaignEntry] = []
    identities: set[tuple[str, str, int, str, str]] = set()
    canonical_entries: list[dict[str, str]] = []
    canonical_identity_entries: list[dict[str, str]] = []
    for ordinal, raw_entry in enumerate(raw_entries, 1):
        entry = _object(raw_entry, "campaign entry")
        expected_entry_fields = (
            {"path", "sha256"}
            if schema_version == "txnopt-level1-campaign-plan-v1"
            else {
                "path",
                "sha256",
                "expected_identity_path",
                "expected_identity_sha256",
            }
        )
        if set(entry) != expected_entry_fields:
            raise ValueError("campaign entry field set differs from its plan schema")
        relative_path = entry.get("path")
        expected_sha256 = entry.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
            raise ValueError("campaign entry path or digest is invalid")
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or ".." in pure.parts or pure.parts[:1] != ("configs",):
            raise ValueError("campaign config path escapes its plan")
        config_candidate = manifest_path.parent / Path(*pure.parts)
        if config_candidate.is_symlink():
            raise ValueError("campaign config cannot be a symlink")
        config_path = config_candidate.resolve(strict=True)
        if manifest_path.parent not in config_path.parents:
            raise ValueError("campaign config must be a regular in-plan file")
        if sha256_file(config_path) != expected_sha256:
            raise ValueError(f"campaign config digest differs: {relative_path}")
        config = _object(json.loads(config_path.read_bytes()), "campaign config")
        identity_path: Path | None = None
        identity_relative: str | None = None
        identity_sha256: str | None = None
        if schema_version == "txnopt-level1-campaign-plan-v2":
            identity_relative = entry.get("expected_identity_path")
            identity_sha256 = entry.get("expected_identity_sha256")
            if not isinstance(identity_relative, str) or not isinstance(
                identity_sha256, str
            ):
                raise ValueError("campaign expected identity binding is invalid")
            identity_pure = PurePosixPath(identity_relative)
            if (
                identity_pure.is_absolute()
                or ".." in identity_pure.parts
                or identity_pure.parts[:1] != ("expected-identities",)
            ):
                raise ValueError("campaign expected identity path escapes its plan")
            identity_candidate = manifest_path.parent / Path(*identity_pure.parts)
            if identity_candidate.is_symlink():
                raise ValueError("campaign expected identity cannot be a symlink")
            identity_path = identity_candidate.resolve(strict=True)
            if manifest_path.parent not in identity_path.parents:
                raise ValueError("campaign expected identity must be an in-plan file")
            if verify_sidecar(identity_path) != identity_sha256:
                raise ValueError("campaign expected identity digest differs")
            identity_bytes = identity_path.read_bytes()
            identity_payload = json.loads(identity_bytes)
            if identity_bytes != canonical_json_bytes(identity_payload, pretty=True):
                raise ValueError("campaign expected identity must use canonical bytes")
            observed_identity = ExpectedEvidenceIdentity.from_payload(identity_payload)
            derived_identity = ExpectedEvidenceIdentity.from_plan_inputs(
                config_path,
                build_manifest_path=build_manifest_path,
            )
            if observed_identity != derived_identity:
                raise ValueError("campaign expected identity differs from plan inputs")
            canonical_identity_entries.append(
                {"path": identity_relative, "sha256": identity_sha256}
            )
        campaign_entry = _parse_config_entry(
            ordinal,
            relative_path,
            config_path,
            expected_sha256,
            config,
            plan,
            raw_output_root,
            case_ids,
            catalog_sources,
            seeds,
            expected_identity_path=identity_path,
            expected_identity_relative_path=identity_relative,
            expected_identity_sha256=identity_sha256,
        )
        identity = (
            campaign_entry.domain,
            campaign_entry.case_id,
            campaign_entry.seed,
            campaign_entry.axis,
            campaign_entry.budget,
        )
        if identity in identities:
            raise ValueError(f"duplicate campaign identity: {identity}")
        identities.add(identity)
        entries.append(campaign_entry)
        canonical_entries.append({"path": relative_path, "sha256": expected_sha256})

    expected_identities = {
        (domain, case_id, seed, axis, budget)
        for domain in DOMAINS
        for case_id in case_ids[domain]
        for seed in seeds
        for axis in AXES
        for budget in BUDGETS
    }
    if identities != expected_identities:
        raise ValueError("campaign config identity set differs from the protocol")
    if sha256_bytes(canonical_json_bytes(canonical_entries)) != plan.get("config_tree_sha256"):
        raise ValueError("campaign config tree digest differs")
    identity_tree_sha256: str | None = None
    if schema_version == "txnopt-level1-campaign-plan-v2":
        identity_tree_sha256 = sha256_bytes(
            canonical_json_bytes(canonical_identity_entries)
        )
        if identity_tree_sha256 != plan.get("expected_identity_tree_sha256"):
            raise ValueError("campaign expected identity tree digest differs")
    elif "expected_identity_tree_sha256" in plan:
        raise ValueError("legacy campaign plan cannot bind a v2 identity tree")
    return CampaignPlan(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        payload=plan,
        protocol_path=protocol_path,
        protocol_sha256=protocol_sha256,
        raw_output_root=raw_output_root,
        build_manifest_path=build_manifest_path,
        build_manifest_sha256=build_manifest_sha256,
        expected_identity_tree_sha256=identity_tree_sha256,
        entries=tuple(entries),
    )


def require_prebound_expected_identities(plan: CampaignPlan) -> None:
    if (
        plan.payload.get("schema_version") != "txnopt-level1-campaign-plan-v2"
        or plan.expected_identity_tree_sha256 is None
        or any(
            entry.expected_identity_relative_path is None
            or entry.expected_identity_path is None
            or entry.expected_identity_sha256 is None
            for entry in plan.entries
        )
    ):
        raise ValueError("formal campaign requires a prebound expected identity tree")


def load_prebound_expected_identity(entry: CampaignEntry) -> ExpectedEvidenceIdentity:
    if entry.expected_identity_path is None or entry.expected_identity_sha256 is None:
        raise ValueError("campaign entry lacks its prebound expected identity")
    if verify_sidecar(entry.expected_identity_path) != entry.expected_identity_sha256:
        raise ValueError("campaign expected identity changed after plan validation")
    if sha256_file(entry.config_path) != entry.config_sha256:
        raise ValueError("campaign config changed after plan validation")
    identity = ExpectedEvidenceIdentity.from_payload(
        json.loads(entry.expected_identity_path.read_bytes())
    )
    if (
        identity.run_label != entry.run_label
        or identity.input_config_sha256 != entry.config_sha256
        or identity.domain != entry.domain
    ):
        raise ValueError("campaign expected identity differs from its entry")
    return identity


def load_analysis_protocol(path: Path, *, plan: CampaignPlan) -> tuple[dict[str, Any], str]:
    payload = read_signed_object(path)
    analysis_schema = payload.get("schema_version")
    protocol = _object(json.loads(plan.protocol_path.read_bytes()), "Level 1 protocol")
    protocol_schema = protocol.get("schema_version")
    if (analysis_schema, protocol_schema) not in {
        ("txnopt-level1-analysis-protocol-v1", "txnopt-level1-protocol-v1"),
        ("txnopt-level1-analysis-protocol-v2", "txnopt-level1-protocol-v2"),
    }:
        raise ValueError("analysis protocol version differs from its Level 1 protocol")
    digest = verify_sidecar(path)
    if (
        payload.get("status") != ANALYSIS_STATUS
        or payload.get("campaign_plan_sha256") != plan.manifest_sha256
        or payload.get("level1_protocol_sha256") != plan.protocol_sha256
        or payload.get("holdout_opened") is not False
    ):
        raise ValueError("analysis protocol is not bound to the closed campaign plan")
    performance = _object(payload.get("performance_gates"), "performance gates")
    bootstrap = _object(payload.get("bootstrap"), "bootstrap protocol")
    resources = _object(payload.get("resource_environment"), "resource environment")
    quality = _object(payload.get("fixed_work_quality_gate"), "fixed-work quality gate")
    t4_cmax = _object(payload.get("t4_cmax_gate"), "T4/Cmax gate")
    if (
        quality.get("comparison") != FIXED_WORK_COMPARISON
        or quality.get("maximum_regressions") != 0
        or tuple(payload.get("pairing_unit", ())) != PAIRING_UNIT
        or payload.get("performance_measure") != PERFORMANCE_MEASURE
        or performance
        != {
            "four_worker_geomean_minimum": 1.3,
            "four_worker_ci_lower_minimum_exclusive": 1.05,
            "one_worker_maximum_overhead_fraction": 0.15,
        }
        or bootstrap.get("method") != "paired_percentile_bootstrap_of_log_speedup"
        or bootstrap.get("confidence_level") != 0.95
        or bootstrap.get("quantile_rule") != "nearest_rank"
        or bootstrap.get("resamples") != 20_000
        or bootstrap.get("seed") != 20_260_813
        or t4_cmax
        != {
            "require_positive_observed_cmax_for_every_run_with_semantic_exact_work": True,
            "require_independent_waste_bound_recomputation": True,
        }
    ):
        raise ValueError("analysis protocol differs from the exact preregistration")
    if analysis_schema == "txnopt-level1-analysis-protocol-v1":
        if (
            resources.get("exclusive_linux_required") is not True
            or resources.get("maximum_consecutive_window_days") != 14
            or resources.get("minimum_physical_cores") != 32
            or resources.get("minimum_memory_gib") != 128
        ):
            raise ValueError("analysis resource environment differs from Level 1 v1")
    elif resources != {
        "minimum_physical_cores": 64,
        "minimum_provider_memory_gb": 128,
        "core_count": 64,
        "thread_per_core": 1,
        "region": None,
        "region_required_live_input": True,
        "exclusive_linux_required": True,
        "maximum_consecutive_window_days": 14,
        "predicted_completion_days_max": 10,
        "linux_visible_memory_is_admission_gate": False,
        "attempt27_peak_rss_fraction_max": 0.8,
    }:
        raise ValueError("analysis resource environment differs from Level 1 v2")
    return payload, digest


def validate_authorization(
    payload: dict[str, Any],
    plan: CampaignPlan,
    analysis_sha256: str,
) -> None:
    build = read_signed_object(
        plan.build_manifest_path,
        schema_version="txnopt-level1-build-manifest-v1",
    )
    wheel = _object(
        _object(build.get("artifacts"), "build artifacts").get("wheel"),
        "build wheel",
    )
    attempt = plan.payload.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
        raise PermissionError("campaign plan attempt identity is invalid")
    expected_scope = f"execute_exact_level1_attempt{attempt:02d}_only"
    if (
        payload.get("authorized") is not True
        or payload.get("authorization_scope") != expected_scope
        or payload.get("campaign_plan_sha256") != plan.manifest_sha256
        or payload.get("analysis_protocol_sha256") != analysis_sha256
        or payload.get("config_tree_sha256") != plan.payload.get("config_tree_sha256")
        or payload.get("expected_identity_tree_sha256")
        != plan.expected_identity_tree_sha256
        or payload.get("build_manifest_sha256") != plan.build_manifest_sha256
        or payload.get("wheel_sha256") != wheel.get("sha256")
        or payload.get("raw_output_root") != str(plan.raw_output_root)
        or payload.get("holdout_opened") is not False
        or payload.get("maximum_window_days") != 14
        or payload.get("exclusive_linux") is not True
    ):
        raise PermissionError("procurement authorization does not match the exact campaign")
    protocol = _object(json.loads(plan.protocol_path.read_bytes()), "Level 1 protocol")
    if protocol.get("schema_version") == "txnopt-level1-protocol-v2":
        provider = _object(
            payload.get("tencent_provider_instance"),
            "Tencent provider instance",
        )
        if (
            not isinstance(provider.get("instance_type"), str)
            or not provider["instance_type"]
            or provider.get("physical_cores") != 64
            or isinstance(provider.get("memory_gb"), bool)
            or not isinstance(provider.get("memory_gb"), int)
            or provider["memory_gb"] < 128
            or not isinstance(provider.get("region"), str)
            or not provider["region"]
            or not isinstance(provider.get("zone"), str)
            or not provider["zone"]
            or payload.get("cvm_dry_run_passed") is not True
            or not isinstance(payload.get("cvm_dry_run_receipt_sha256"), str)
            or len(payload["cvm_dry_run_receipt_sha256"]) != 64
        ):
            raise PermissionError(
                "procurement authorization lacks the exact Tencent CVM live evidence"
            )


def executable_path(path: Path) -> Path:
    absolute = path.expanduser().absolute()
    if not absolute.is_file() or not absolute.resolve(strict=True).is_file():
        raise FileNotFoundError(f"Python executable is missing: {absolute}")
    return absolute


def isolated_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "PYTHONSAFEPATH": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def run_isolated_process(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> IsolatedProcessResult:
    """Run one producer/reviewer in its own Linux process group.

    A timeout or a surviving descendant terminates the complete process group.
    Returning with a live descendant is forbidden because it could mutate a
    supposedly sealed raw bundle after the parent command exits.
    """

    if platform.system() != "Linux":
        raise RuntimeError("campaign subprocess isolation requires Linux")
    if timeout_seconds <= 0:
        raise ValueError("campaign subprocess timeout must be positive")
    process = subprocess.Popen(  # noqa: S603 - exact command is caller-owned
        command,
        cwd=cwd,
        env=isolated_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    cleanup_performed = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        cleanup_performed = True
        _terminate_process_group(process.pid)
        stdout, stderr = process.communicate(timeout=5)
    members = _process_group_members(process.pid)
    if members:
        cleanup_performed = True
        _terminate_process_group(process.pid)
        members = _process_group_members(process.pid)
    if members:
        raise RuntimeError(f"campaign process descendants survived cleanup: {members}")
    return IsolatedProcessResult(
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        descendant_cleanup_performed=cleanup_performed,
        descendant_processes_remaining=members,
    )


def campaign_claim_path(plan: CampaignPlan) -> Path:
    return plan.raw_output_root.with_name(plan.raw_output_root.name + ".launch-claim")


def active_campaign_processes(plan: CampaignPlan) -> tuple[dict[str, object], ...]:
    markers = (
        str(plan.manifest_path),
        str(plan.raw_output_root),
        str(plan.manifest_path.parent / "configs"),
    )
    active: list[dict[str, object]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            executable_name = (entry / "exe").resolve(strict=True).name
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if (
            executable_name.startswith("python")
            and any(marker in command for marker in markers)
            and (
                "txnopt_evidence.level1_campaign_runner" in command
                or "txnopt_evidence.cli" in command
            )
        ):
            active.append(
                {
                    "pid": int(entry.name),
                    "executable": executable_name,
                    "command": command,
                }
            )
    return tuple(sorted(active, key=lambda item: int(str(item["pid"]))))


def linux_host_identity(
    *,
    physical_cores: int,
    memory_gib: int,
    usable_core_tokens: int,
    exclusive_linux: bool,
) -> dict[str, object]:
    if platform.system() != "Linux" or not Path("/proc").is_dir():
        raise RuntimeError("formal campaign requires an exclusive Linux host")
    cpu_model = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("model name") and ":" in line
        ),
        None,
    )
    os_release = Path("/etc/os-release")
    return {
        "system": platform.system(),
        "kernel_release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "physical_cores": physical_cores,
        "memory_gib": memory_gib,
        "usable_core_tokens": usable_core_tokens,
        "exclusive_linux_authorized": exclusive_linux,
        "os_release_sha256": sha256_file(os_release) if os_release.is_file() else None,
    }


def physical_core_count() -> int:
    blocks = Path("/proc/cpuinfo").read_text(encoding="utf-8").split("\n\n")
    identities: set[tuple[str, str]] = set()
    for block in blocks:
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
        if "physical id" in fields and "core id" in fields:
            identities.add((fields["physical id"], fields["core id"]))
    if not identities:
        raise RuntimeError("physical core identity is unavailable")
    return len(identities)


def memory_gib() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            kib = int(line.split()[1])
            return kib // (1024 * 1024)
    raise RuntimeError("host memory identity is unavailable")


def _process_group_members(process_group_id: int) -> tuple[int, ...]:
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            remainder = stat[stat.rfind(")") + 2 :].split()
            state = remainder[0]
            group_id = int(remainder[2])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
        if group_id == process_group_id and state != "Z":
            members.append(int(entry.name))
    return tuple(sorted(members))


def _terminate_process_group(process_group_id: int) -> None:
    for termination_signal in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process_group_id, termination_signal)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if not _process_group_members(process_group_id):
                return
            time.sleep(0.02)


def require_clean_repository(root: Path) -> None:
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("campaign orchestration repository must be clean")


def verify_runtime_installation(
    plan: CampaignPlan,
    *,
    python: Path,
    wheel: Path,
) -> dict[str, Any]:
    python_path = executable_path(python)
    wheel_candidate = wheel.expanduser().absolute()
    if wheel_candidate.is_symlink():
        raise ValueError("runtime wheel cannot be a symlink")
    wheel_path = wheel_candidate.resolve(strict=True)
    build = read_signed_object(
        plan.build_manifest_path,
        schema_version="txnopt-level1-build-manifest-v1",
    )
    artifacts = _object(build.get("artifacts"), "build artifacts")
    wheel_record = _object(artifacts.get("wheel"), "build wheel")
    native_record = _object(artifacts.get("native_extension"), "build native")
    if sha256_file(wheel_path) != wheel_record.get("sha256"):
        raise ValueError("runtime wheel differs from the campaign build manifest")
    probe_source = """
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import txnopt
from txnopt import _native
import txnopt_cases
import txnopt_evidence
import txnopt_legacy

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

dependencies = sorted(
    {
        (str(distribution.metadata.get("Name")), distribution.version)
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
)

print(json.dumps({
    "version": importlib.metadata.version("txnopt"),
    "prefix": sys.prefix,
    "package_root": str(Path(txnopt.__file__).resolve().parent.parent),
    "txnopt_path": str(Path(txnopt.__file__).absolute()),
    "cases_path": str(Path(txnopt_cases.__file__).absolute()),
    "evidence_path": str(Path(txnopt_evidence.__file__).absolute()),
    "legacy_path": str(Path(txnopt_legacy.__file__).absolute()),
    "native_path": str(Path(_native.__file__).absolute()),
    "native_sha256": digest(_native.__file__),
    "python_executable": sys.executable,
    "python_version": sys.version,
    "python_implementation": platform.python_implementation(),
    "platform": platform.platform(),
    "kernel_release": platform.release(),
    "dependencies": dependencies,
    "attestation": dict(_native.BUILD_ATTESTATION),
}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(python_path), "-I", "-c", probe_source],
        check=False,
        capture_output=True,
        text=True,
        cwd=plan.manifest_path.parent,
        env=isolated_environment(),
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"runtime identity probe failed: {completed.stderr.strip()}")
    probe = _object(json.loads(completed.stdout), "runtime identity probe")
    producer = _object(build.get("producer"), "build producer")
    attestation = _object(probe.get("attestation"), "native attestation")
    if (
        probe.get("version") != "0.1.0a1"
        or probe.get("native_sha256") != native_record.get("sha256")
        or attestation.get("source_revision") != producer.get("revision")
        or attestation.get("source_tree") != producer.get("git_tree")
        or attestation.get("source_manifest_sha256") != producer.get("source_manifest_sha256")
        or attestation.get("source_dirty") is not False
        or attestation.get("development_override") is not False
    ):
        raise ValueError("installed runtime identity differs from the campaign build")
    package_root = Path(str(probe.get("package_root"))).resolve(strict=True)
    prefix = Path(str(probe.get("prefix"))).resolve(strict=True)
    if not package_root.is_relative_to(prefix):
        raise ValueError("runtime packages are not installed below the selected Python prefix")
    expected_import_paths = {
        "txnopt_path": package_root / "txnopt" / "__init__.py",
        "cases_path": package_root / "txnopt_cases" / "__init__.py",
        "evidence_path": package_root / "txnopt_evidence" / "__init__.py",
        "legacy_path": package_root / "txnopt_legacy" / "__init__.py",
        "native_path": package_root / "txnopt" / Path(str(probe["native_path"])).name,
    }
    for key, expected_path in expected_import_paths.items():
        actual_path = Path(str(probe.get(key))).expanduser().absolute()
        if actual_path.is_symlink() or actual_path.resolve(strict=True) != expected_path:
            raise ValueError(f"runtime import resolved outside the verified wheel: {key}")
    compared = 0
    with zipfile.ZipFile(wheel_path) as archive:
        wheel_entries = [info for info in archive.infolist() if not info.is_dir()]
        if len(wheel_entries) != wheel_record.get("entry_count"):
            raise ValueError("runtime wheel entry count differs from the build manifest")
        _verify_wheel_record(archive, wheel_entries)
        for info in archive.infolist():
            if (
                info.is_dir()
                or info.filename.endswith((".pyc", ".pyo"))
                or info.filename.endswith(".dist-info/RECORD")
            ):
                continue
            installed = package_root / Path(*info.filename.split("/"))
            if not installed.is_file() or installed.is_symlink():
                raise ValueError(f"installed wheel entry is missing: {info.filename}")
            if sha256_bytes(archive.read(info)) != sha256_file(installed):
                raise ValueError(f"installed wheel entry differs: {info.filename}")
            compared += 1
        _verify_installed_record(package_root, archive, wheel_entries)
    if compared <= 0:
        raise ValueError("runtime wheel verification compared no package entries")
    return {
        "python": str(python_path),
        "python_resolved": str(python_path.resolve(strict=True)),
        "python_executable_sha256": sha256_file(python_path.resolve(strict=True)),
        "python_version": str(probe["python_version"]),
        "python_implementation": str(probe["python_implementation"]),
        "platform": str(probe["platform"]),
        "kernel_release": str(probe["kernel_release"]),
        "python_prefix": str(prefix),
        "wheel_path": str(wheel_path),
        "wheel_sha256": str(wheel_record["sha256"]),
        "native_sha256": str(native_record["sha256"]),
        "source_revision": str(producer["revision"]),
        "source_tree": str(producer["git_tree"]),
        "installed_entry_count_verified": compared,
        "wheel_entry_count_verified": len(wheel_entries),
        "dependency_count": len(probe["dependencies"]),
        "dependency_lock_sha256": sha256_bytes(canonical_json_bytes(probe["dependencies"])),
        "status": "PASS",
    }


def _verify_wheel_record(
    archive: zipfile.ZipFile,
    wheel_entries: list[zipfile.ZipInfo],
) -> None:
    record_names = [
        info.filename for info in wheel_entries if info.filename.endswith(".dist-info/RECORD")
    ]
    if len(record_names) != 1:
        raise ValueError("runtime wheel must contain exactly one RECORD")
    rows = list(csv.reader(io.StringIO(archive.read(record_names[0]).decode("utf-8"))))
    recorded = {row[0]: row[1:] for row in rows if len(row) == 3}
    if set(recorded) != {info.filename for info in wheel_entries}:
        raise ValueError("runtime wheel RECORD entry set differs")
    for info in wheel_entries:
        digest_field, size_field = recorded[info.filename]
        if info.filename == record_names[0]:
            if digest_field or size_field:
                raise ValueError("runtime wheel RECORD self-entry must be unhashed")
            continue
        data = archive.read(info)
        expected_digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        )
        if digest_field != f"sha256={expected_digest}" or size_field != str(len(data)):
            raise ValueError(f"runtime wheel RECORD differs: {info.filename}")


def _verify_installed_record(
    package_root: Path,
    archive: zipfile.ZipFile,
    wheel_entries: list[zipfile.ZipInfo],
) -> None:
    record_name = next(
        info.filename for info in wheel_entries if info.filename.endswith(".dist-info/RECORD")
    )
    installed_record = package_root / Path(*record_name.split("/"))
    if not installed_record.is_file() or installed_record.is_symlink():
        raise ValueError("installed wheel RECORD is missing")
    rows = list(csv.reader(installed_record.read_text(encoding="utf-8").splitlines()))
    installed_rows = {row[0]: row[1:] for row in rows if len(row) == 3}
    for info in wheel_entries:
        if info.filename == record_name:
            continue
        fields = installed_rows.get(info.filename)
        data = archive.read(info)
        expected_digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        )
        if fields != [f"sha256={expected_digest}", str(len(data))]:
            raise ValueError(f"installed wheel RECORD differs: {info.filename}")


def read_event_stream(path: Path) -> tuple[dict[str, Any], ...]:
    verify_sidecar(path)
    previous = "0" * 64
    events: list[dict[str, Any]] = []
    for expected_id, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        raw: object = json.loads(line)
        record = _object(raw, "event record")
        payload = _object(record.get("payload"), "event payload")
        unsigned = {
            "event_id": expected_id,
            "previous_event_sha256": previous,
            "payload": payload,
        }
        actual = sha256_bytes(canonical_json_bytes(unsigned))
        if (
            record.get("event_id") != expected_id
            or record.get("previous_event_sha256") != previous
            or record.get("event_sha256") != actual
        ):
            raise ValueError(f"event chain differs at {path}:{expected_id}")
        events.append(payload)
        previous = actual
    if not events:
        raise ValueError(f"event stream is empty: {path}")
    return tuple(events)


def _parse_config_entry(
    ordinal: int,
    relative_path: str,
    config_path: Path,
    config_sha256: str,
    config: dict[str, Any],
    plan: dict[str, Any],
    raw_output_root: Path,
    case_ids: dict[str, tuple[str, ...]],
    catalog_sources: dict[str, dict[str, Path]],
    seeds: tuple[int, ...],
    *,
    expected_identity_path: Path | None,
    expected_identity_relative_path: str | None,
    expected_identity_sha256: str | None,
) -> CampaignEntry:
    if config.get("schema_version") != "txnopt-run-config-v1":
        raise ValueError("campaign config schema differs")
    run_label = config.get("run_label")
    output_root = config.get("output_root")
    build_binding = _object(config.get("build_manifest"), "config build binding")
    run_config = _object(config.get("run_config"), "config run payload")
    case = _object(config.get("case"), "config case")
    if not isinstance(run_label, str):
        raise ValueError("campaign run label is missing")
    if not isinstance(output_root, str) or Path(output_root).resolve() != raw_output_root:
        raise ValueError("campaign config raw output root differs")
    if build_binding.get("path") != str(
        Path(str(plan["build_manifest_path"])).resolve()
    ) or build_binding.get("sha256") != plan.get("build_manifest_sha256"):
        raise ValueError("campaign config build binding differs")
    domain = case.get("domain")
    source_path = case.get("source_instance_path")
    source_sha256 = case.get("source_instance_sha256")
    if (
        domain not in DOMAINS
        or not isinstance(source_path, str)
        or not isinstance(source_sha256, str)
    ):
        raise ValueError("campaign case domain or source path differs")
    source_candidate = Path(source_path).expanduser().absolute()
    if source_candidate.is_symlink():
        raise ValueError("campaign source instance cannot be a symlink")
    source = source_candidate.resolve(strict=True)
    if sha256_file(source) != source_sha256:
        raise ValueError("campaign source instance binding differs")
    case_id = Path(source_path).stem
    if case_id not in case_ids[str(domain)]:
        raise ValueError("campaign case is outside the preregistered scope")
    if source != catalog_sources[str(domain)][case_id]:
        raise ValueError("campaign config source differs from the case catalog")
    seed = run_config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in seeds:
        raise ValueError("campaign seed is outside the preregistered scope")
    if case.get("oracle_seed") != seed:
        raise ValueError("campaign oracle seed differs from the runtime seed")
    mode = run_config.get("execution_mode")
    workers = run_config.get("workers")
    speculation = run_config.get("speculation_window")
    if (
        not isinstance(mode, str)
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or isinstance(speculation, bool)
        or not isinstance(speculation, int)
    ):
        raise ValueError("campaign execution axis fields are malformed")
    matching_axes = [
        name for name, values in AXES.items() if values == (mode, workers, speculation)
    ]
    if len(matching_axes) != 1:
        raise ValueError("campaign execution axis is not canonical")
    fixed_work = run_config.get("fixed_work")
    deadline = run_config.get("deadline_seconds")
    if fixed_work is not None and deadline is None:
        budget = "fixed_work"
        if fixed_work != plan.get("fixed_work"):
            raise ValueError("campaign fixed-work budget differs")
    elif fixed_work is None and deadline is not None:
        budget = "fixed_time"
        if deadline != plan.get("fixed_time_seconds"):
            raise ValueError("campaign fixed-time budget differs")
    else:
        raise ValueError("campaign config must have exactly one budget")
    expected_label = (
        f"txnopt_level1_{domain}_{case_id.lower()}_{seed}_{matching_axes[0]}_"
        f"{budget}_attempt{int(plan['attempt']):02d}"
    )
    if run_label != expected_label:
        raise ValueError("campaign run label differs from its canonical identity")
    if (
        run_config.get("max_rounds") != plan.get("max_rounds")
        or run_config.get("trace_policy") != "semantic_and_physical"
    ):
        raise ValueError("campaign round or trace policy differs")
    return CampaignEntry(
        ordinal=ordinal,
        relative_path=relative_path,
        config_path=config_path,
        config_sha256=config_sha256,
        expected_identity_relative_path=expected_identity_relative_path,
        expected_identity_path=expected_identity_path,
        expected_identity_sha256=expected_identity_sha256,
        run_label=run_label,
        raw_output_root=raw_output_root,
        domain=str(domain),
        case_id=case_id,
        seed=seed,
        axis=matching_axes[0],
        budget=budget,
        workers=workers,
    )


def _bound_file(payload: dict[str, Any], path_key: str, digest_key: str) -> Path:
    raw_path = payload.get(path_key)
    expected = payload.get(digest_key)
    if not isinstance(raw_path, str) or not isinstance(expected, str):
        raise ValueError(f"signed binding is missing: {path_key}")
    candidate = Path(raw_path).expanduser().absolute()
    if candidate.is_symlink():
        raise ValueError(f"signed binding cannot be a symlink: {path_key}")
    path = candidate.resolve(strict=True)
    if sha256_file(path) != expected:
        raise ValueError(f"signed binding differs: {path_key}")
    return path


def _protocol_case_ids(protocol: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    domains = _object(protocol.get("domains"), "protocol domains")
    if set(domains) != set(DOMAINS):
        raise ValueError("protocol domain identity set differs")
    result: dict[str, tuple[str, ...]] = {}
    for domain in DOMAINS:
        scope = _object(domains.get(domain), f"{domain} scope")
        identifiers = (*_strings(scope, "pilot"), *_strings(scope, "validation"))
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{domain} protocol scope is empty or duplicated")
        result[domain] = identifiers
    return result


def _catalog_sources(
    catalog: dict[str, Any],
    case_ids: dict[str, tuple[str, ...]],
) -> dict[str, dict[str, Path]]:
    domains = _object(catalog.get("domains"), "catalog domains")
    if set(domains) != set(DOMAINS):
        raise ValueError("case catalog domain identity set differs")
    result: dict[str, dict[str, Path]] = {}
    for domain in DOMAINS:
        raw_sources = _object(domains.get(domain), f"{domain} catalog sources")
        if set(raw_sources) != set(case_ids[domain]):
            raise ValueError(f"{domain} catalog case identity set differs")
        sources: dict[str, Path] = {}
        for case_id, raw_path in raw_sources.items():
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"{domain} catalog source path is invalid")
            candidate = Path(raw_path).expanduser().absolute()
            if candidate.is_symlink():
                raise ValueError("case catalog source cannot be a symlink")
            sources[case_id] = candidate.resolve(strict=True)
        result[domain] = sources
    return result


def _strings(payload: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = payload.get(key)
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise ValueError(f"{key} must be a string list")
    return tuple(raw)


def _integers(payload: dict[str, Any], key: str) -> tuple[int, ...]:
    raw = payload.get(key)
    if not isinstance(raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw
    ):
        raise ValueError(f"{key} must be an integer list")
    return tuple(raw)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return value


__all__ = [
    "AXES",
    "BUDGETS",
    "DOMAINS",
    "CampaignEntry",
    "CampaignPlan",
    "canonical_json_bytes",
    "load_analysis_protocol",
    "load_campaign_plan",
    "executable_path",
    "isolated_environment",
    "read_event_stream",
    "read_signed_object",
    "sha256_file",
    "require_clean_repository",
    "validate_authorization",
    "verify_sidecar",
    "verify_runtime_installation",
    "write_signed_object",
]
