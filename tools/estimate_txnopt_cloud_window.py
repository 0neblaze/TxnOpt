"""Estimate the Level 1 cloud window; never procure or launch a server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from txnopt_evidence.codec import write_signed_json
from txnopt_evidence.runtime_estimate import estimate_cloud_window


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--physical-cores", type=int, required=True)
    parser.add_argument("--scheduler-efficiency", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    protocol = json.loads(arguments.protocol.resolve(strict=True).read_bytes())
    calibration = json.loads(arguments.calibration.resolve(strict=True).read_bytes())
    result = estimate_cloud_window(
        protocol,
        calibration,
        physical_cores=arguments.physical_cores,
        scheduler_efficiency=arguments.scheduler_efficiency,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_signed_json(arguments.output, result)
    print(digest)
    return 0 if result["status"] == "PURCHASE_GATE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
