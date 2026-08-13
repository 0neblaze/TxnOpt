"""Composition-root CLI for TxnOpt runtime, evidence, and legacy verification."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from txnopt_evidence.reviewer import replay_manifest, verify_raw_manifest
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

    replay = commands.add_parser("replay")
    replay.add_argument("manifest", type=Path)
    replay.add_argument("--output-dir", type=Path, required=True)

    commands.add_parser("env")

    legacy = commands.add_parser("legacy")
    legacy_commands = legacy.add_subparsers(dest="legacy_command", required=True)
    legacy_verify = legacy_commands.add_parser("verify")
    legacy_verify.add_argument("receipt", type=Path)
    return parser


def _error(command: str, error: Exception) -> int:
    print(
        json.dumps(
            {
                "schema_version": "txnopt-cli-error-v1",
                "command": command,
                "error_type": type(error).__name__,
                "error": str(error),
                "fallback_used": False,
            },
            sort_keys=True,
        ),
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
            result = replay_manifest(
                arguments.manifest,
                output_dir=arguments.output_dir,
            )
        elif arguments.command == "verify":
            payload = verify_raw_manifest(arguments.manifest)
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


if __name__ == "__main__":
    raise SystemExit(main())
