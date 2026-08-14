"""Offline Tencent deployment bundle for one exact Build16/Attempt26 identity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from txnopt_evidence.codec import (
    canonical_json_bytes,
    read_signed_json,
    sha256_bytes,
    sha256_file,
    verify_sidecar,
    write_exclusive,
    write_signed_json,
)
from txnopt_evidence.toolchain import read_toolchain_lock

_FORBIDDEN_SERIALIZED_FIELDS = (
    "SecretId",
    "SecretKey",
    "SessionToken",
)


@dataclass(frozen=True, slots=True)
class TencentDeploymentInputs:
    build_manifest: Path
    wheel: Path
    source_manifest: Path
    native_attestation: Path
    uv_lock: Path
    toolchain_lock: Path
    plan_manifest: Path


def materialize_tencent_deployment(
    destination: Path,
    *,
    inputs: TencentDeploymentInputs,
) -> Path:
    """Create an account-free, region-free deployment bundle exactly once."""

    identity = _validate_inputs(inputs)
    target = destination.expanduser().absolute()
    target.mkdir(parents=False, exist_ok=False)
    artifacts = target / "artifacts"
    artifacts.mkdir()
    copied: list[dict[str, object]] = []
    for role, source, name, signed in (
        ("build_manifest", inputs.build_manifest, "build16-manifest.json", True),
        ("wheel", inputs.wheel, inputs.wheel.name, False),
        ("source_manifest", inputs.source_manifest, "source-manifest.json", False),
        ("native_attestation", inputs.native_attestation, "native-attestation.json", False),
        ("uv_lock", inputs.uv_lock, "uv.lock", False),
        ("toolchain_lock", inputs.toolchain_lock, "toolchain-lock.json", True),
        ("attempt26_plan", inputs.plan_manifest, "attempt26-plan.json", True),
    ):
        copied.append(_copy_input(artifacts, role=role, source=source, name=name))
        if signed:
            copied.append(
                _copy_input(
                    artifacts,
                    role=f"{role}_sidecar",
                    source=source.with_suffix(source.suffix + ".sha256"),
                    name=f"{name}.sha256",
                )
            )

    generated = {
        "bootstrap-ubuntu-24.04.sh": _bootstrap_script(inputs.wheel.name),
        "cloud-init.yaml": _cloud_init(),
        "runinstances-request-template.json": canonical_json_bytes(
            _runinstances_template(), pretty=True
        ),
        "cos-contract-template.json": canonical_json_bytes(
            _cos_contract_template(), pretty=True
        ),
    }
    for relative_path, data in generated.items():
        path = target / relative_path
        digest = write_exclusive(path, data)
        if relative_path.endswith(".sh"):
            path.chmod(0o755)
        copied.append(
            {
                "role": relative_path,
                "relative_path": relative_path,
                "sha256": digest,
                "size": len(data),
            }
        )
    copied.sort(key=lambda entry: str(entry["relative_path"]))
    tree_digest = sha256_bytes(canonical_json_bytes(copied))
    manifest = target / "bundle-receipt.json"
    write_signed_json(
        manifest,
        {
            "schema_version": "txnopt-tencent-deployment-bundle-v1",
            "status": "BUNDLE_MATERIALIZED_NOT_AUTHORIZED",
            "build": "Build16",
            "attempt": 26,
            "build_manifest_sha256": identity["build_manifest_sha256"],
            "plan_manifest_sha256": identity["plan_manifest_sha256"],
            "config_tree_sha256": identity["config_tree_sha256"],
            "expected_identity_tree_sha256": identity["expected_identity_tree_sha256"],
            "region": None,
            "region_required_live_input": True,
            "credentials_included": False,
            "dry_run_only": True,
            "instance_creation_interface_included": False,
            "cloud_purchase_authorized": False,
            "formal_matrix_started": False,
            "file_tree_sha256": tree_digest,
            "file_count": len(copied),
            "files": copied,
        },
    )
    return manifest


def verify_tencent_deployment(manifest_path: Path) -> dict[str, object]:
    """Verify every byte and the offline/live boundary of a deployment bundle."""

    payload = read_signed_json(manifest_path)
    expected_fields = {
        "schema_version",
        "status",
        "build",
        "attempt",
        "build_manifest_sha256",
        "plan_manifest_sha256",
        "config_tree_sha256",
        "expected_identity_tree_sha256",
        "region",
        "region_required_live_input",
        "credentials_included",
        "dry_run_only",
        "instance_creation_interface_included",
        "cloud_purchase_authorized",
        "formal_matrix_started",
        "file_tree_sha256",
        "file_count",
        "files",
    }
    if set(payload) != expected_fields:
        raise ValueError("deployment bundle receipt field set differs")
    if (
        payload["schema_version"] != "txnopt-tencent-deployment-bundle-v1"
        or payload["status"] != "BUNDLE_MATERIALIZED_NOT_AUTHORIZED"
        or payload["build"] != "Build16"
        or payload["attempt"] != 26
        or payload["region"] is not None
        or payload["region_required_live_input"] is not True
        or payload["credentials_included"] is not False
        or payload["dry_run_only"] is not True
        or payload["instance_creation_interface_included"] is not False
        or payload["cloud_purchase_authorized"] is not False
        or payload["formal_matrix_started"] is not False
    ):
        raise ValueError("deployment bundle boundary differs")
    files = payload["files"]
    if not isinstance(files, list) or payload["file_count"] != len(files):
        raise ValueError("deployment bundle file count differs")
    root = manifest_path.resolve(strict=True).parent
    expected_paths = {manifest_path.name, f"{manifest_path.name}.sha256"}
    normalized: list[dict[str, object]] = []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {
            "role",
            "relative_path",
            "sha256",
            "size",
        }:
            raise ValueError("deployment bundle file entry differs")
        relative = _relative_path(entry["relative_path"])
        candidate = root / relative
        if candidate.is_symlink() or candidate.resolve(strict=True).parent not in {
            root,
            root / "artifacts",
        }:
            raise ValueError("deployment bundle path escapes its root")
        digest = sha256_file(candidate)
        size = candidate.stat().st_size
        if digest != entry["sha256"] or size != entry["size"]:
            raise ValueError(f"deployment bundle digest differs: {relative}")
        expected_paths.add(relative)
        normalized.append(dict(entry))
    if sha256_bytes(canonical_json_bytes(normalized)) != payload["file_tree_sha256"]:
        raise ValueError("deployment bundle tree digest differs")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("deployment bundle path set differs")
    request = json.loads((root / "runinstances-request-template.json").read_bytes())
    if request != _runinstances_template():
        raise ValueError("deployment RunInstances template differs")
    serialized = b"\n".join(path.read_bytes() for path in root.rglob("*") if path.is_file())
    if any(field.encode() in serialized for field in _FORBIDDEN_SERIALIZED_FIELDS):
        raise ValueError("deployment bundle contains credential fields")
    read_toolchain_lock(root / "artifacts" / "toolchain-lock.json")
    return {
        "schema_version": "txnopt-tencent-deployment-verification-v1",
        "status": "BUNDLE_VERIFIED_NOT_AUTHORIZED",
        "build": "Build16",
        "attempt": 26,
        "region": None,
        "credentials_included": False,
        "file_tree_sha256": payload["file_tree_sha256"],
        "file_count": payload["file_count"],
        "cloud_purchase_authorized": False,
        "formal_matrix_started": False,
    }


def _validate_inputs(inputs: TencentDeploymentInputs) -> dict[str, object]:
    build_digest = verify_sidecar(inputs.build_manifest)
    build = read_signed_json(inputs.build_manifest)
    if build.get("run_label") != "txnopt_level1_build_attempt16":
        raise ValueError("deployment requires Build16")
    if build.get("wheel_sha256") != sha256_file(inputs.wheel):
        raise ValueError("Build16 wheel identity differs")
    if build.get("source_manifest_sha256") != sha256_file(inputs.source_manifest):
        raise ValueError("Build16 source manifest identity differs")
    if build.get("native_build_attestation_sha256") != sha256_file(
        inputs.native_attestation
    ):
        raise ValueError("Build16 native attestation identity differs")
    read_toolchain_lock(inputs.toolchain_lock)
    plan_digest = verify_sidecar(inputs.plan_manifest)
    plan = read_signed_json(inputs.plan_manifest)
    if (
        plan.get("schema_version") != "txnopt-level1-campaign-plan-v2"
        or plan.get("protocol_schema_version") != "txnopt-level1-protocol-v2"
        or plan.get("attempt") != 26
        or plan.get("status") != "PLANNED_NOT_STARTED"
        or plan.get("build_manifest_sha256") != build_digest
        or plan.get("region") is not None
        or plan.get("region_required_live_input") is not True
        or plan.get("formal_matrix_started") is not False
        or plan.get("cloud_purchase_authorized") is not False
        or plan.get("holdout_opened") is not False
    ):
        raise ValueError("Attempt26 plan or region boundary differs")
    return {
        "build_manifest_sha256": build_digest,
        "plan_manifest_sha256": plan_digest,
        "config_tree_sha256": _sha_field(plan, "config_tree_sha256"),
        "expected_identity_tree_sha256": _sha_field(
            plan, "expected_identity_tree_sha256"
        ),
    }


def _copy_input(
    destination: Path,
    *,
    role: str,
    source: Path,
    name: str,
) -> dict[str, object]:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"deployment input must be a regular file: {source}")
    data = source.read_bytes()
    relative = f"artifacts/{name}"
    return {
        "role": role,
        "relative_path": relative,
        "sha256": write_exclusive(destination / name, data),
        "size": len(data),
    }


def _runinstances_template() -> dict[str, object]:
    return {
        "Region": None,
        "DryRun": True,
        "Placement": {"Zone": None},
        "InstanceType": None,
        "ImageId": None,
        "VirtualPrivateCloud": {"VpcId": None, "SubnetId": None},
        "SecurityGroupIds": [],
        "CpuTopology": {"CoreCount": 64, "ThreadPerCore": 1},
        "InstanceCount": 1,
        "InternetAccessible": {"PublicIpAssigned": False},
        "live_input_required": True,
        "provider_memory_gb_minimum": 128,
    }


def _cos_contract_template() -> dict[str, object]:
    return {
        "schema_version": "txnopt-tencent-cos-contract-template-v1",
        "bucket": None,
        "region": None,
        "versioning": "Enabled",
        "object_lock_mode": "COMPLIANCE",
        "minimum_retention_days": 365,
        "exact_version_id_required": True,
        "credentials_included": False,
        "live_input_required": True,
    }


def _bootstrap_script(wheel_name: str) -> bytes:
    script = f"""#!/usr/bin/env bash
set -euo pipefail
test "$(uname -s)" = Linux
test -f artifacts/{wheel_name}
test -f artifacts/uv.lock
test -f artifacts/toolchain-lock.json
echo 'TxnOpt offline bundle verified; account and region input remain required.'
"""
    return script.encode()


def _cloud_init() -> bytes:
    return b"""#cloud-config
package_update: false
runcmd:
  - [bash, /opt/txnopt/bootstrap-ubuntu-24.04.sh]
final_message: "TxnOpt offline bootstrap complete; no instance run was authorized."
"""


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("deployment bundle relative path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("deployment bundle relative path is invalid")
    return path.as_posix()


def _sha_field(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Attempt26 {field} is invalid")
    return value


__all__ = [
    "TencentDeploymentInputs",
    "materialize_tencent_deployment",
    "verify_tencent_deployment",
]
