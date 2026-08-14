"""Private fresh-process entrypoint for one independent raw-bundle replay."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from txnopt_evidence.codec import read_signed_json
from txnopt_evidence.identity import ExpectedEvidenceIdentity
from txnopt_evidence.reviewer import replay_legacy_manifest, replay_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m txnopt_evidence.review_cli")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--expected-identity", type=Path)
    identity.add_argument("--legacy-compatibility", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        result = (
            replay_legacy_manifest(arguments.manifest, output_dir=arguments.output_dir)
            if arguments.legacy_compatibility
            else replay_manifest(
                arguments.manifest,
                output_dir=arguments.output_dir,
                expected_identity=ExpectedEvidenceIdentity.from_payload(
                    read_signed_json(arguments.expected_identity)
                ),
            )
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "schema_version": "txnopt-review-cli-error-v1",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "fallback_used": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
