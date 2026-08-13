"""Composition-root CLI for TxnOpt runtime, evidence, and legacy verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from txnopt_legacy import LegacyReceiptReader

_VERSION: Final = "0.1.0a1"


def _signed_json(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    data = resolved.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError("signed JSON sidecar is missing")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != digest or fields[1] != resolved.name:
        raise ValueError("signed JSON sidecar differs")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("signed JSON must contain an object")
    return payload


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

    commands.add_parser("env")

    legacy = commands.add_parser("legacy")
    legacy_commands = legacy.add_subparsers(dest="legacy_command", required=True)
    legacy_verify = legacy_commands.add_parser("verify")
    legacy_verify.add_argument("receipt", type=Path)
    return parser


def _unavailable(command: str) -> int:
    print(
        json.dumps(
            {
                "schema_version": "txnopt-cli-error-v1",
                "command": command,
                "error": "level1_runtime_not_implemented",
                "fallback_used": False,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        return _unavailable("run")
    if arguments.command == "replay":
        return _unavailable("replay")
    if arguments.command == "verify":
        payload = _signed_json(arguments.manifest)
        result: object = {
            "schema_version": "txnopt-verification-v1",
            "status": "verified",
            "manifest_schema_version": payload.get("schema_version", ""),
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
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
