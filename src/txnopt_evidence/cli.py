"""Composition-root CLI for TxnOpt runtime, evidence, and legacy verification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from txnopt_evidence.codec import read_signed_json
from txnopt_evidence.identity import ExpectedEvidenceIdentity
from txnopt_evidence.reviewer import (
    verify_legacy_manifest,
    verify_manifest,
)

_VERSION: Final = "0.1.0a1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="txnopt")
    parser.add_argument("--version", action="version", version=f"txnopt {_VERSION}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    verify_identity = verify.add_mutually_exclusive_group()
    verify_identity.add_argument("--expected-identity", type=Path)
    verify_identity.add_argument("--legacy-compatibility", action="store_true")

    plan = commands.add_parser(
        "plan",
        help="materialize the closed Level 1 campaign plan without executing it",
    )
    plan.add_argument("--protocol", type=Path, required=True)
    plan.add_argument("--catalog", type=Path, required=True)
    plan.add_argument("--destination", type=Path, required=True)
    plan.add_argument("--raw-output-root", type=Path, required=True)
    plan.add_argument("--build-manifest", type=Path, required=True)
    plan.add_argument("--fixed-work", type=int, required=True)
    plan.add_argument("--fixed-time-seconds", type=float, required=True)
    plan.add_argument("--max-rounds", type=int, required=True)
    plan.add_argument("--evrptw-max-candidates", type=int, required=True)
    plan.add_argument("--rcpsp-max-candidates", type=int, required=True)
    plan.add_argument("--attempt", type=int, default=1)

    preflight = commands.add_parser(
        "preflight",
        help="validate a closed Level 1 campaign without creating raw output",
    )
    _add_campaign_preflight_arguments(preflight)

    review = commands.add_parser(
        "review",
        help="independently replay and review a completed Level 1 campaign",
    )
    _add_campaign_review_arguments(review)

    cloud = commands.add_parser("cloud")
    cloud_providers = cloud.add_subparsers(dest="cloud_provider", required=True)
    tencent = cloud_providers.add_parser("tencent")
    tencent_commands = tencent.add_subparsers(dest="tencent_command", required=True)
    tencent_assess = tencent_commands.add_parser("assess")
    tencent_assess.add_argument("--protocol", type=Path, required=True)
    tencent_assess.add_argument("--calibration", type=Path, required=True)
    tencent_assess.add_argument("--physical-cores", type=int, default=64)
    tencent_assess.add_argument("--provider-memory-gb", type=int, default=128)
    tencent_assess.add_argument("--scheduler-efficiency", type=float, default=0.8)
    tencent_spec = tencent_commands.add_parser("spec")
    tencent_spec.add_argument("--output", type=Path, required=True)
    tencent_bundle = tencent_commands.add_parser("bundle")
    tencent_bundle.add_argument("--destination", type=Path, required=True)
    tencent_bundle.add_argument("--build-manifest", type=Path, required=True)
    tencent_bundle.add_argument("--wheel", type=Path, required=True)
    tencent_bundle.add_argument("--source-manifest", type=Path, required=True)
    tencent_bundle.add_argument("--native-attestation", type=Path, required=True)
    tencent_bundle.add_argument("--pyproject", type=Path, required=True)
    tencent_bundle.add_argument("--uv-lock", type=Path, required=True)
    tencent_bundle.add_argument("--uv-wheel", type=Path, required=True)
    tencent_bundle.add_argument("--toolchain-lock", type=Path, required=True)
    tencent_bundle.add_argument("--plan-manifest", type=Path, required=True)
    tencent_bundle_verify = tencent_commands.add_parser("bundle-verify")
    tencent_bundle_verify.add_argument("--manifest", type=Path, required=True)
    tencent_doctor = tencent_commands.add_parser("doctor")
    tencent_doctor.add_argument("--provider-receipt", type=Path, required=True)
    tencent_doctor.add_argument("--expected-peak-rss-bytes", type=int)
    tencent_doctor.add_argument("--work-directory", type=Path, default=Path.cwd())
    tencent_doctor.add_argument("--output", type=Path, required=True)
    tencent_dry_run = tencent_commands.add_parser("dry-run")
    tencent_dry_run.add_argument("--region", required=True)
    tencent_dry_run.add_argument("--zone", required=True)
    tencent_dry_run.add_argument("--instance-type", required=True)
    tencent_dry_run.add_argument("--image-id", required=True)
    tencent_dry_run.add_argument("--vpc-id", required=True)
    tencent_dry_run.add_argument("--subnet-id", required=True)
    tencent_dry_run.add_argument("--security-group-id", required=True)
    tencent_dry_run.add_argument("--request-output", type=Path, required=True)
    tencent_dry_run.add_argument("--receipt-output", type=Path, required=True)
    tencent_cos = tencent_commands.add_parser("cos")
    tencent_cos_commands = tencent_cos.add_subparsers(
        dest="tencent_cos_command",
        required=True,
    )
    tencent_cos_mirror = tencent_cos_commands.add_parser("mirror")
    tencent_cos_mirror.add_argument("source", type=Path)
    _add_tencent_cos_arguments(tencent_cos_mirror)
    tencent_cos_mirror.add_argument("--commit-id", required=True)

    tencent_cos_verify = tencent_cos_commands.add_parser("verify")
    _add_tencent_cos_ref_arguments(tencent_cos_verify)

    tencent_cos_restore = tencent_cos_commands.add_parser("restore")
    _add_tencent_cos_ref_arguments(tencent_cos_restore)
    tencent_cos_restore.add_argument("--destination", type=Path, required=True)

    archive = commands.add_parser("archive")
    archive_commands = archive.add_subparsers(dest="archive_command", required=True)
    archive_inventory = archive_commands.add_parser("inventory")
    archive_inventory.add_argument("source", type=Path)

    return parser


def _error(command: str, error: Exception) -> int:
    manifest_path = getattr(error, "manifest_path", None)
    payload = {
        "schema_version": "txnopt-cli-error-v1",
        "command": command,
        "error_type": type(error).__name__,
        "error": _redacted_error_text(error),
        "fallback_used": False,
    }
    if isinstance(manifest_path, Path):
        payload["failure_manifest_path"] = str(manifest_path)
    print(
        json.dumps(payload, sort_keys=True),
        file=sys.stderr,
    )
    return 2


def _redacted_error_text(error: Exception) -> str:
    message = str(error)
    for variable in (
        "TENCENTCLOUD_SECRET_ID",
        "TENCENTCLOUD_SECRET_KEY",
        "TENCENTCLOUD_SESSION_TOKEN",
    ):
        value = os.environ.get(variable)
        if value:
            message = message.replace(value, "[REDACTED]")
    return message


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "run":
            from txnopt_evidence.runner import run_config_file

            raw = run_config_file(arguments.config)
            result: object = {
                "schema_version": "txnopt-run-command-v1",
                "status": "complete",
                "run_label": raw.run_label,
                "manifest_path": str(raw.manifest_path),
                "manifest_sha256": raw.manifest_sha256,
                "fallback_count": 0,
            }
        elif arguments.command == "plan":
            from txnopt_evidence.campaign import materialize_level1_plan

            manifest_path = materialize_level1_plan(
                arguments.protocol,
                arguments.catalog,
                destination=arguments.destination,
                raw_output_root=arguments.raw_output_root,
                build_manifest_path=arguments.build_manifest,
                fixed_work=arguments.fixed_work,
                fixed_time_seconds=arguments.fixed_time_seconds,
                max_rounds=arguments.max_rounds,
                evrptw_max_candidates=arguments.evrptw_max_candidates,
                rcpsp_max_candidates=arguments.rcpsp_max_candidates,
                attempt=arguments.attempt,
            )
            result = {
                "schema_version": "txnopt-level1-plan-command-v1",
                "status": "planned",
                "manifest_path": str(manifest_path),
            }
        elif arguments.command == "preflight":
            from txnopt_evidence.level1_campaign_runner import preflight_campaign

            result = preflight_campaign(
                arguments.plan_manifest,
                arguments.analysis_protocol,
                python=arguments.python,
                wheel=arguments.wheel,
            )
        elif arguments.command == "review":
            from txnopt_evidence.level1_campaign_reviewer import review_campaign

            result = review_campaign(
                arguments.plan_manifest,
                arguments.analysis_protocol,
                execution_receipt_path=arguments.execution_receipt,
                python=arguments.python,
                wheel=arguments.wheel,
                review_root=arguments.review_root,
                review_receipt_path=arguments.review_receipt,
                review_workers=arguments.review_workers,
                per_run_timeout_seconds=arguments.per_run_timeout_seconds,
            )
        elif arguments.command == "verify":
            payload = (
                verify_legacy_manifest(arguments.manifest)
                if arguments.legacy_compatibility
                else verify_manifest(
                    arguments.manifest,
                    expected_identity=_expected_identity(arguments.expected_identity),
                )
            )
            result = {
                "schema_version": "txnopt-verification-v1",
                "status": "verified",
                "manifest_schema_version": payload.get("schema_version", ""),
                "run_label": payload.get("run_label", ""),
                "fallback_count": 0,
            }
        elif arguments.command == "cloud":
            result = _cloud_command(arguments)
        elif arguments.command == "archive":
            result = _archive_command(arguments)
        else:
            raise ValueError("unsupported TxnOpt command")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        return _error(str(arguments.command), error)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _expected_identity(path: Path | None) -> ExpectedEvidenceIdentity:
    if path is None:
        raise ValueError("expected evidence identity is required")
    return ExpectedEvidenceIdentity.from_payload(read_signed_json(path))


def _add_campaign_preflight_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--analysis-protocol", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--wheel", type=Path)


def _add_campaign_review_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--analysis-protocol", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--review-workers", type=int, default=4)
    parser.add_argument("--per-run-timeout-seconds", type=float, default=600.0)


def _add_tencent_cos_ref_arguments(parser: argparse.ArgumentParser) -> None:
    _add_tencent_cos_arguments(parser)
    parser.add_argument("--commit-id", required=True)
    parser.add_argument("--commit-key", required=True)
    parser.add_argument("--commit-version-id", required=True)
    parser.add_argument("--commit-sha256", required=True)
    parser.add_argument("--commit-size", type=int, required=True)
    parser.add_argument("--commit-retain-until", required=True)


def _add_tencent_cos_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cos-bucket", required=True)
    parser.add_argument("--cos-region", required=True)
    parser.add_argument("--cos-prefix", default="txnopt")
    parser.add_argument("--cos-minimum-retention-days", type=int, default=365)


def _archive_command(arguments: argparse.Namespace) -> object:
    from txnopt_evidence.archive import inventory_tree

    if arguments.archive_command == "inventory":
        return inventory_tree(arguments.source).to_payload()
    raise ValueError("unknown archive command")


def _tencent_cos_command(arguments: argparse.Namespace) -> object:
    from txnopt_evidence.tencent_cos import TencentCosArchive, TencentCosCommitRef

    archive = TencentCosArchive.from_environment(
        bucket=arguments.cos_bucket,
        region=arguments.cos_region,
        prefix=arguments.cos_prefix,
        minimum_retention_days=arguments.cos_minimum_retention_days,
    )
    if arguments.tencent_cos_command == "mirror":
        ref = archive.mirror_tree(arguments.source, commit_id=arguments.commit_id)
        return {
            "schema_version": "txnopt-tencent-cos-mirror-result-v1",
            "bucket_contract": _json_dataclass(archive.bucket_receipt),
            "commit_ref": ref.to_payload(),
        }
    ref = TencentCosCommitRef(
        commit_id=arguments.commit_id,
        key=arguments.commit_key,
        version_id=arguments.commit_version_id,
        sha256=arguments.commit_sha256,
        size=arguments.commit_size,
        retain_until=arguments.commit_retain_until,
    )
    if arguments.tencent_cos_command == "verify":
        verification = archive.verify_commit(ref)
        return {
            "schema_version": "txnopt-tencent-cos-verification-v1",
            "commit_ref": ref.to_payload(),
            "object_count": verification.object_count,
            "total_size": verification.total_size,
            "verified": verification.verified,
        }
    if arguments.tencent_cos_command == "restore":
        restore_receipt = archive.restore_commit(ref, destination=arguments.destination)
        return {
            "schema_version": "txnopt-tencent-cos-restore-v1",
            "commit_ref": ref.to_payload(),
            "object_count": restore_receipt.object_count,
            "total_size": restore_receipt.total_size,
            "verified": restore_receipt.verified,
        }
    raise ValueError("unknown Tencent COS command")


def _cloud_command(arguments: argparse.Namespace) -> object:
    from txnopt_evidence.codec import write_signed_json
    from txnopt_evidence.tencent_cloud import (
        TencentCvmSelection,
        assess_tencent_capacity,
        create_tencent_provisioning_spec,
        execute_cvm_dry_run,
        inspect_tencent_host_from_provider_receipt,
        prepare_cvm_dry_run_envelope,
    )

    if arguments.cloud_provider != "tencent":
        raise ValueError("unsupported cloud provider")
    if arguments.tencent_command == "cos":
        return _tencent_cos_command(arguments)
    if arguments.tencent_command == "assess":
        return assess_tencent_capacity(
            read_signed_json(arguments.protocol),
            read_signed_json(arguments.calibration),
            physical_cores=arguments.physical_cores,
            provider_memory_gb=arguments.provider_memory_gb,
            scheduler_efficiency=arguments.scheduler_efficiency,
        )
    if arguments.tencent_command == "spec":
        digest = write_signed_json(
            arguments.output,
            create_tencent_provisioning_spec(),
        )
        return {
            "schema_version": "txnopt-tencent-spec-command-v1",
            "status": "SPEC_MATERIALIZED_NOT_AUTHORIZED",
            "spec_path": str(arguments.output),
            "spec_sha256": digest,
            "cloud_purchase_authorized": False,
        }
    if arguments.tencent_command == "bundle":
        from txnopt_evidence.tencent_deployment import (
            TencentDeploymentInputs,
            materialize_tencent_deployment,
            verify_tencent_deployment,
        )

        manifest = materialize_tencent_deployment(
            arguments.destination,
            inputs=TencentDeploymentInputs(
                build_manifest=arguments.build_manifest,
                wheel=arguments.wheel,
                source_manifest=arguments.source_manifest,
                native_attestation=arguments.native_attestation,
                pyproject=arguments.pyproject,
                uv_lock=arguments.uv_lock,
                uv_wheel=arguments.uv_wheel,
                toolchain_lock=arguments.toolchain_lock,
                plan_manifest=arguments.plan_manifest,
            ),
        )
        return verify_tencent_deployment(manifest)
    if arguments.tencent_command == "bundle-verify":
        from txnopt_evidence.tencent_deployment import verify_tencent_deployment

        return verify_tencent_deployment(arguments.manifest)
    if arguments.tencent_command == "doctor":
        receipt = inspect_tencent_host_from_provider_receipt(
            arguments.provider_receipt,
            expected_peak_rss_bytes=arguments.expected_peak_rss_bytes,
            work_directory=arguments.work_directory,
        )
        digest = write_signed_json(arguments.output, receipt)
        if receipt["capacity_pass"] is not True:
            raise RuntimeError(
                "Tencent host failed its resource contract; "
                f"signed receipt: {arguments.output} ({digest})"
            )
        return receipt
    if arguments.tencent_command == "dry-run":
        selection = TencentCvmSelection(
            region=arguments.region,
            zone=arguments.zone,
            instance_type=arguments.instance_type,
            image_id=arguments.image_id,
            vpc_id=arguments.vpc_id,
            subnet_id=arguments.subnet_id,
            security_group_id=arguments.security_group_id,
        )
        request_digest = write_signed_json(
            arguments.request_output,
            prepare_cvm_dry_run_envelope(selection),
        )
        receipt = execute_cvm_dry_run(selection)
        receipt_digest = write_signed_json(arguments.receipt_output, receipt)
        return {
            "schema_version": "txnopt-tencent-dry-run-command-v1",
            "status": "DRY_RUN_PASS_NO_INSTANCE_CREATED",
            "request_path": str(arguments.request_output),
            "request_sha256": request_digest,
            "receipt_path": str(arguments.receipt_output),
            "receipt_sha256": receipt_digest,
            "submitted": True,
            "dry_run": True,
            "instance_created": False,
            "cloud_purchase_authorized": False,
        }
    raise ValueError("unsupported Tencent cloud command")


def _json_dataclass(value: object) -> object:
    from dataclasses import asdict, is_dataclass

    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("expected a dataclass receipt")
    return asdict(value)


if __name__ == "__main__":
    raise SystemExit(main())
