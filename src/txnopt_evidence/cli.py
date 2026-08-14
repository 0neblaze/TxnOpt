"""Composition-root CLI for TxnOpt runtime, evidence, and legacy verification."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from txnopt_evidence.codec import read_signed_json
from txnopt_evidence.identity import ExpectedEvidenceIdentity
from txnopt_evidence.reviewer import (
    replay_legacy_manifest,
    replay_manifest,
    verify_legacy_manifest,
    verify_manifest,
)
from txnopt_legacy import LegacyReceiptReader

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

    replay = commands.add_parser("replay")
    replay.add_argument("manifest", type=Path)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay_identity = replay.add_mutually_exclusive_group()
    replay_identity.add_argument("--expected-identity", type=Path)
    replay_identity.add_argument("--legacy-compatibility", action="store_true")

    commands.add_parser("env")

    archive = commands.add_parser("archive")
    archive_commands = archive.add_subparsers(dest="archive_command", required=True)
    archive_inventory = archive_commands.add_parser("inventory")
    archive_inventory.add_argument("source", type=Path)

    archive_mirror = archive_commands.add_parser("mirror")
    archive_mirror.add_argument("source", type=Path)
    archive_mirror.add_argument("--store", type=Path, required=True)
    archive_mirror.add_argument("--commit-id", required=True)

    archive_verify = archive_commands.add_parser("verify")
    _add_archive_ref_arguments(archive_verify)

    archive_restore = archive_commands.add_parser("restore")
    _add_archive_ref_arguments(archive_restore)
    archive_restore.add_argument("--destination", type=Path, required=True)

    legacy = commands.add_parser("legacy")
    legacy_commands = legacy.add_subparsers(dest="legacy_command", required=True)
    legacy_verify = legacy_commands.add_parser("verify")
    legacy_verify.add_argument("receipt", type=Path)
    return parser


def _error(command: str, error: Exception) -> int:
    manifest_path = getattr(error, "manifest_path", None)
    payload = {
        "schema_version": "txnopt-cli-error-v1",
        "command": command,
        "error_type": type(error).__name__,
        "error": str(error),
        "fallback_used": False,
    }
    if isinstance(manifest_path, Path):
        payload["failure_manifest_path"] = str(manifest_path)
    print(
        json.dumps(payload, sort_keys=True),
        file=sys.stderr,
    )
    return 2


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
        elif arguments.command == "replay":
            result = (
                replay_legacy_manifest(
                    arguments.manifest,
                    output_dir=arguments.output_dir,
                )
                if arguments.legacy_compatibility
                else replay_manifest(
                    arguments.manifest,
                    output_dir=arguments.output_dir,
                    expected_identity=_expected_identity(arguments.expected_identity),
                )
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
        elif arguments.command == "env":
            result = {
                "schema_version": "txnopt-environment-v1",
                "txnopt_version": _VERSION,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            }
        elif arguments.command == "archive":
            result = _archive_command(arguments)
        else:
            payload = LegacyReceiptReader().read(arguments.receipt)
            result = {
                "schema_version": "txnopt-legacy-verification-v1",
                "status": "verified",
                "legacy_schema_version": payload.get("schema_version", ""),
            }
    except (OSError, RuntimeError, ValueError) as error:
        return _error(str(arguments.command), error)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _expected_identity(path: Path | None) -> ExpectedEvidenceIdentity:
    if path is None:
        raise ValueError("expected evidence identity is required")
    return ExpectedEvidenceIdentity.from_payload(read_signed_json(path))


def _add_archive_ref_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--commit-id", required=True)
    parser.add_argument("--commit-sha256", required=True)
    parser.add_argument("--commit-size", type=int, required=True)


def _archive_command(arguments: argparse.Namespace) -> object:
    from txnopt_evidence.archive import (
        ArchiveCommitRef,
        LocalFilesystemArchiveStore,
        inventory_tree,
        mirror_tree,
        restore_commit,
    )

    if arguments.archive_command == "inventory":
        return inventory_tree(arguments.source).to_payload()
    store = LocalFilesystemArchiveStore(arguments.store)
    if arguments.archive_command == "mirror":
        ref = mirror_tree(arguments.source, store=store, commit_id=arguments.commit_id)
        return {
            "schema_version": "txnopt-archive-mirror-result-v1",
            "commit_ref": _archive_ref_payload(ref),
        }
    ref = ArchiveCommitRef(
        commit_id=arguments.commit_id,
        sha256=arguments.commit_sha256,
        size=arguments.commit_size,
    )
    if arguments.archive_command == "verify":
        verification = store.verify_commit(ref)
        return {
            "schema_version": "txnopt-archive-verification-v1",
            "commit_ref": _archive_ref_payload(ref),
            "object_count": verification.object_count,
            "total_size": verification.total_size,
            "verified": verification.verified,
        }
    if arguments.archive_command == "restore":
        restore_receipt = restore_commit(store, ref, destination=arguments.destination)
        return {
            "schema_version": "txnopt-archive-restore-v1",
            "commit_ref": _archive_ref_payload(ref),
            "object_count": restore_receipt.object_count,
            "total_size": restore_receipt.total_size,
            "verified": restore_receipt.verified,
        }
    raise ValueError("unknown archive command")


def _archive_ref_payload(ref: object) -> dict[str, object]:
    from txnopt_evidence.archive import ArchiveCommitRef

    if not isinstance(ref, ArchiveCommitRef):
        raise TypeError("archive commit reference has the wrong type")
    return {
        "commit_id": ref.commit_id,
        "sha256": ref.sha256,
        "size": ref.size,
    }


if __name__ == "__main__":
    raise SystemExit(main())
