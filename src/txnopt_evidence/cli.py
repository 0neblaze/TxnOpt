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


if __name__ == "__main__":
    raise SystemExit(main())
