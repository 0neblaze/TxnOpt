from __future__ import annotations

import json
from pathlib import Path

import pytest

from txnopt_evidence.cli import main
from txnopt_evidence.codec import read_signed_json, sha256_file, write_signed_json
from txnopt_evidence.tencent_deployment import (
    TencentDeploymentInputs,
    materialize_tencent_deployment,
    verify_tencent_deployment,
)
from txnopt_evidence.toolchain import ToolchainEntry, write_toolchain_lock

_TOOLS = (
    "uv",
    "cpython",
    "temurin-jdk",
    "tla2tools",
    "tlc",
    "pluscal",
    "jq",
    "numactl",
    "lsof",
    "fuser",
    "ipcs",
    "gcc",
    "cmake",
    "ninja",
    "cos-python-sdk-v5",
    "tencentcloud-sdk-python-common",
    "tencentcloud-sdk-python-cvm",
)


def _inputs(
    tmp_path: Path,
    *,
    region: object = None,
    build_number: int = 16,
    formal_attempt: int = 26,
) -> TencentDeploymentInputs:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    wheel = inputs / "txnopt-0.1.0a1-cp313-cp313-linux_x86_64.whl"
    wheel.write_bytes(b"wheel")
    source_manifest = inputs / "source-manifest.json"
    source_manifest.write_text(
        json.dumps({"source": f"build{build_number}"}) + "\n",
        encoding="utf-8",
    )
    native = inputs / "native-attestation.json"
    native.write_text(
        json.dumps({"source_revision": f"build{build_number}"}) + "\n",
        encoding="utf-8",
    )
    pyproject = inputs / "pyproject.toml"
    pyproject.write_text(
        """[project]
requires-python = ">=3.13,<3.14"
dependencies = []
[project.optional-dependencies]
tencent = [
  "cos-python-sdk-v5==1.9.44",
  "tencentcloud-sdk-python-common==3.1.156",
  "tencentcloud-sdk-python-cvm==3.1.156",
]
""",
        encoding="utf-8",
    )
    uv_lock = inputs / "uv.lock"
    uv_lock.write_text("version = 1\n", encoding="utf-8")
    uv_wheel = inputs / "uv-0.12.4-py3-none-manylinux2014_x86_64.whl"
    uv_wheel.write_bytes(b"locked uv wheel")
    toolchain = inputs / "txnopt-toolchain-lock-v1.json"
    write_toolchain_lock(
        toolchain,
        [
            ToolchainEntry(
                name=name,
                version="0.12.4" if name == "uv" else "locked",
                executable_path=f"/external/{name}",
                sha256="a" * 64,
                validation_exit_code=0,
                source_url="https://example.invalid/tool",
                download_sha256=(
                    sha256_file(uv_wheel) if name == "uv" else "b" * 64
                ),
            )
            for name in _TOOLS
        ],
    )
    build = inputs / f"build{build_number}.json"
    statuses = {
        16: "BUILD_COMPLETE_TENCENT_CLOUD_CUTOVER_NOT_LEVEL1_READY",
        18: "BUILD_COMPLETE_TENCENT_PRE_CLOUD_SUCCESSOR_NOT_LEVEL1_READY",
    }
    write_signed_json(
        build,
        {
            "schema_version": "txnopt-level1-build-manifest-v1",
            "run_label": f"txnopt_level1_build_attempt{build_number}",
            "status": statuses[build_number],
            "formal_successor": {
                "prior_review_binding_status": "PRIOR_SOURCE_ONLY",
                "successor_status": f"REVIEW_PENDING_BUILD{build_number}",
                "independent_successor_review_completed": False,
                "level1_formal_gate_passed": False,
            },
            "producer": {
                "source_manifest_sha256": sha256_file(source_manifest),
            },
            "artifacts": {
                "wheel": {"sha256": sha256_file(wheel)},
                "native_extension": {
                    "attestation_sha256": sha256_file(native),
                },
            },
        },
    )
    protocol = inputs / "level1-protocol-v2.json"
    write_signed_json(
        protocol,
        {
            "schema_version": "txnopt-level1-protocol-v2",
        },
    )
    plan = inputs / f"attempt{formal_attempt}.json"
    write_signed_json(
        plan,
        {
            "schema_version": "txnopt-level1-campaign-plan-v2",
            "attempt": formal_attempt,
            "status": "PLANNED_NOT_STARTED",
            "protocol_path": str(protocol),
            "protocol_sha256": sha256_file(protocol),
            "build_manifest_sha256": sha256_file(build),
            "config_tree_sha256": "c" * 64,
            "expected_identity_tree_sha256": "d" * 64,
            "region": region,
            "region_required_live_input": True,
            "formal_matrix_started": False,
            "cloud_purchase_authorized": False,
            "holdout_opened": False,
        },
    )
    return TencentDeploymentInputs(
        build_manifest=build,
        wheel=wheel,
        source_manifest=source_manifest,
        native_attestation=native,
        pyproject=pyproject,
        uv_lock=uv_lock,
        uv_wheel=uv_wheel,
        toolchain_lock=toolchain,
        plan_manifest=plan,
    )


def test_bundle_materializes_offline_no_secret_no_region_contract(tmp_path: Path) -> None:
    destination = tmp_path / "deployment"

    manifest = materialize_tencent_deployment(
        destination,
        inputs=_inputs(tmp_path),
    )
    receipt = verify_tencent_deployment(manifest)

    assert receipt["status"] == "BUNDLE_VERIFIED_NOT_AUTHORIZED"
    assert receipt["build"] == "Build16"
    assert receipt["attempt"] == 26
    assert receipt["region"] is None
    assert receipt["credentials_included"] is False
    assert (destination / "bootstrap-ubuntu-24.04.sh").is_file()
    assert (destination / "cloud-init.yaml").is_file()
    bootstrap = (destination / "bootstrap-ubuntu-24.04.sh").read_text()
    assert "uv 0.12.4" in bootstrap
    assert "3.13.13" in bootstrap
    assert "sync --frozen --no-install-project --no-dev --extra tencent" in bootstrap
    assert "pip install" in bootstrap
    assert "txnopt._native" in bootstrap
    request = json.loads((destination / "runinstances-request-template.json").read_text())
    assert request["DryRun"] is True
    assert request["CpuTopology"] == {"CoreCount": 64, "ThreadPerCore": 1}
    assert request["Placement"]["Zone"] is None
    assert request["Region"] is None
    serialized = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in destination.rglob("*")
        if path.is_file()
    )
    assert "SecretId" not in serialized
    assert "SecretKey" not in serialized
    assert "SessionToken" not in serialized


def test_bundle_accepts_only_the_closed_build18_attempt28_pair(tmp_path: Path) -> None:
    manifest = materialize_tencent_deployment(
        tmp_path / "build18-deployment",
        inputs=_inputs(tmp_path, build_number=18, formal_attempt=28),
    )

    receipt = verify_tencent_deployment(manifest)

    assert receipt["build"] == "Build18"
    assert receipt["attempt"] == 28
    assert read_signed_json(manifest)["schema_version"] == (
        "txnopt-tencent-deployment-bundle-v2"
    )
    assert (manifest.parent / "artifacts" / "build18-manifest.json").is_file()
    assert (manifest.parent / "artifacts" / "attempt28-plan.json").is_file()
    assert (manifest.parent / "artifacts" / "level1-protocol-v2.json").is_file()

    mismatch_root = tmp_path / "mismatch"
    mismatch_root.mkdir()
    with pytest.raises(ValueError, match="formal attempt"):
        materialize_tencent_deployment(
            mismatch_root / "deployment",
            inputs=_inputs(
                mismatch_root,
                build_number=18,
                formal_attempt=29,
            ),
        )


def test_bundle_rejects_live_region_and_tamper(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="region"):
        materialize_tencent_deployment(
            tmp_path / "region-bound",
            inputs=_inputs(tmp_path, region="ap-guangzhou"),
        )

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    manifest = materialize_tencent_deployment(
        clean_root / "deployment",
        inputs=_inputs(clean_root),
    )
    (manifest.parent / "cloud-init.yaml").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        verify_tencent_deployment(manifest)


def test_bundle_refuses_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "deployment"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        materialize_tencent_deployment(destination, inputs=_inputs(tmp_path))


def test_bundle_and_bundle_verify_are_on_the_unified_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = _inputs(tmp_path)
    destination = tmp_path / "deployment"
    assert main(
        [
            "cloud",
            "tencent",
            "bundle",
            "--destination",
            str(destination),
            "--build-manifest",
            str(inputs.build_manifest),
            "--wheel",
            str(inputs.wheel),
            "--source-manifest",
            str(inputs.source_manifest),
            "--native-attestation",
            str(inputs.native_attestation),
            "--pyproject",
            str(inputs.pyproject),
            "--uv-lock",
            str(inputs.uv_lock),
            "--uv-wheel",
            str(inputs.uv_wheel),
            "--toolchain-lock",
            str(inputs.toolchain_lock),
            "--plan-manifest",
            str(inputs.plan_manifest),
        ]
    ) == 0
    capsys.readouterr()
    assert main(
        [
            "cloud",
            "tencent",
            "bundle-verify",
            "--manifest",
            str(destination / "bundle-receipt.json"),
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "BUNDLE_VERIFIED_NOT_AUTHORIZED"
