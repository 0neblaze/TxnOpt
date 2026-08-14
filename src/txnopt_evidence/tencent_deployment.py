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
    pyproject: Path
    uv_lock: Path
    uv_wheel: Path
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
        ("pyproject", inputs.pyproject, "pyproject.toml", False),
        ("uv_lock", inputs.uv_lock, "uv.lock", False),
        ("uv_wheel", inputs.uv_wheel, inputs.uv_wheel.name, False),
        ("toolchain_lock", inputs.toolchain_lock, "toolchain-lock.json", True),
        ("attempt26_plan", inputs.plan_manifest, "attempt26-plan.json", True),
    ):
        copied.append(_copy_input(artifacts, role=role, source=source, name=name))
        if signed:
            copied.append(
                _write_rebased_sidecar(
                    artifacts,
                    role=f"{role}_sidecar",
                    source=source,
                    target_name=name,
                )
            )

    generated = {
        "bootstrap-ubuntu-24.04.sh": _bootstrap_script(
            wheel_name=inputs.wheel.name,
            wheel_sha256=sha256_file(inputs.wheel),
            uv_wheel_name=inputs.uv_wheel.name,
            uv_wheel_sha256=sha256_file(inputs.uv_wheel),
        ),
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
    if (
        build.get("schema_version") != "txnopt-level1-build-manifest-v1"
        or build.get("run_label") != "txnopt_level1_build_attempt16"
    ):
        raise ValueError("deployment requires Build16")
    producer = build.get("producer")
    artifacts = build.get("artifacts")
    if not isinstance(producer, dict) or not isinstance(artifacts, dict):
        raise ValueError("Build16 producer or artifact identity is missing")
    wheel = artifacts.get("wheel")
    native = artifacts.get("native_extension")
    if not isinstance(wheel, dict) or not isinstance(native, dict):
        raise ValueError("Build16 wheel or native identity is missing")
    if wheel.get("sha256") != sha256_file(inputs.wheel):
        raise ValueError("Build16 wheel identity differs")
    if producer.get("source_manifest_sha256") != sha256_file(
        inputs.source_manifest
    ):
        raise ValueError("Build16 source manifest identity differs")
    if native.get("attestation_sha256") != sha256_file(
        inputs.native_attestation
    ):
        raise ValueError("Build16 native attestation identity differs")
    toolchain = read_toolchain_lock(inputs.toolchain_lock)
    uv_entries = [
        entry
        for entry in toolchain["tools"]
        if isinstance(entry, dict) and entry.get("name") == "uv"
    ]
    if len(uv_entries) != 1:
        raise ValueError("toolchain lock lacks its unique uv identity")
    uv_entry = uv_entries[0]
    if (
        uv_entry.get("version") != "0.12.4"
        or uv_entry.get("download_sha256") != sha256_file(inputs.uv_wheel)
    ):
        raise ValueError("deployment uv wheel differs from the toolchain lock")
    pyproject = inputs.pyproject.read_text(encoding="utf-8")
    if (
        'requires-python = ">=3.13,<3.14"' not in pyproject
        or 'cos-python-sdk-v5==1.9.44' not in pyproject
        or 'tencentcloud-sdk-python-common==3.1.156' not in pyproject
        or 'tencentcloud-sdk-python-cvm==3.1.156' not in pyproject
    ):
        raise ValueError("deployment pyproject lacks the locked Tencent environment")
    plan_digest = verify_sidecar(inputs.plan_manifest)
    plan = read_signed_json(inputs.plan_manifest)
    protocol_path_value = plan.get("protocol_path")
    if not isinstance(protocol_path_value, str) or not protocol_path_value:
        raise ValueError("Attempt26 protocol path is missing")
    protocol_path = Path(protocol_path_value).expanduser().absolute()
    protocol_digest = verify_sidecar(protocol_path)
    protocol = read_signed_json(protocol_path)
    if (
        plan.get("schema_version") != "txnopt-level1-campaign-plan-v2"
        or protocol.get("schema_version") != "txnopt-level1-protocol-v2"
        or plan.get("protocol_sha256") != protocol_digest
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


def _write_rebased_sidecar(
    destination: Path,
    *,
    role: str,
    source: Path,
    target_name: str,
) -> dict[str, object]:
    digest = verify_sidecar(source)
    name = f"{target_name}.sha256"
    data = f"{digest}  {target_name}\n".encode()
    return {
        "role": role,
        "relative_path": f"artifacts/{name}",
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


def _bootstrap_script(
    *,
    wheel_name: str,
    wheel_sha256: str,
    uv_wheel_name: str,
    uv_wheel_sha256: str,
) -> bytes:
    script = f"""#!/usr/bin/env bash
set -euo pipefail
test "$(uname -s)" = Linux
test "$(uname -m)" = x86_64
. /etc/os-release
test "$ID" = ubuntu
test "$VERSION_ID" = 24.04
BUNDLE_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd -P)"
INSTALL_ROOT="${{TXNOPT_INSTALL_ROOT:-/opt/txnopt/runtime}}"
SYSTEM_PYTHON="${{TXNOPT_SYSTEM_PYTHON:-python3}}"
WHEEL="$BUNDLE_ROOT/artifacts/{wheel_name}"
UV_WHEEL="$BUNDLE_ROOT/artifacts/{uv_wheel_name}"
test -f "$WHEEL"
test -f "$UV_WHEEL"
test -f "$BUNDLE_ROOT/artifacts/pyproject.toml"
test -f "$BUNDLE_ROOT/artifacts/uv.lock"
test -f "$BUNDLE_ROOT/artifacts/toolchain-lock.json"
printf '%s  %s\n' '{wheel_sha256}' "$WHEEL" | sha256sum --check --status
printf '%s  %s\n' '{uv_wheel_sha256}' "$UV_WHEEL" | sha256sum --check --status
test ! -e "$INSTALL_ROOT"
mkdir -p "$INSTALL_ROOT"
"$SYSTEM_PYTHON" -m venv "$INSTALL_ROOT/uv-bootstrap"
"$INSTALL_ROOT/uv-bootstrap/bin/python" -m pip install \
  --disable-pip-version-check --no-index --no-deps "$UV_WHEEL"
UV_BIN="$INSTALL_ROOT/uv-bootstrap/bin/uv"
case "$("$UV_BIN" --version)" in
  "uv 0.12.4"*) ;;
  *) echo 'locked uv 0.12.4 is unavailable' >&2; exit 2 ;;
esac
export UV_PYTHON_INSTALL_DIR="$INSTALL_ROOT/python"
"$UV_BIN" python install 3.13.13
PYTHON_313="$("$UV_BIN" python find 3.13.13)"
test "$("$PYTHON_313" -c 'import platform; print(platform.python_version())')" = 3.13.13
mkdir "$INSTALL_ROOT/project"
cp "$BUNDLE_ROOT/artifacts/pyproject.toml" "$INSTALL_ROOT/project/pyproject.toml"
cp "$BUNDLE_ROOT/artifacts/uv.lock" "$INSTALL_ROOT/project/uv.lock"
export UV_PROJECT_ENVIRONMENT="$INSTALL_ROOT/venv"
cd "$INSTALL_ROOT/project"
"$UV_BIN" sync --frozen --no-install-project --no-dev --extra tencent --python 3.13.13
"$UV_BIN" pip install --python "$INSTALL_ROOT/venv/bin/python" --no-deps "$WHEEL"
"$INSTALL_ROOT/venv/bin/python" - <<'PY'
import txnopt
import txnopt._native
import txnopt_cases
import txnopt_evidence
import txnopt_legacy

assert set(txnopt.__all__) == {{
    "TxnRuntime", "SearchKernel", "Oracle", "RunConfig", "RunResult"
}}
print("TxnOpt Build16 installation verified; account and region remain required.")
PY
"$INSTALL_ROOT/venv/bin/txnopt" --help >/dev/null
"""
    return script.encode()


def _cloud_init() -> bytes:
    return b"""#cloud-config
package_update: true
packages:
  - ca-certificates
  - python3
  - python3-venv
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
