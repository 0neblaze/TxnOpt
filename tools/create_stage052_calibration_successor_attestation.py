"""Create the signed Attempt23 resource-contract successor attestation."""

from __future__ import annotations

import argparse
from pathlib import Path

from evrptw.stage052_campaign_runner import (
    create_calibration_successor_attestation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--successor-revision", required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--calibration-review-manifest", type=Path, required=True)
    parser.add_argument("--resource-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = create_calibration_successor_attestation(
        repository=args.repository,
        successor_revision=args.successor_revision,
        calibration_report_path=args.calibration_report,
        calibration_review_manifest_path=args.calibration_review_manifest,
        resource_contract_path=args.resource_contract,
        output_path=args.output,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
