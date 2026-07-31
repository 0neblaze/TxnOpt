"""Fail-closed independent review boundary for frozen Stage 0--2.2 runners.

Those historical producers predate the lifecycle-v3 replay contract.  This
module deliberately cannot promote their new output: a stage-specific replay
adapter must be implemented before a future rerun can leave
``BLOCKED_RETENTION``.  Keeping this boundary separate from each producer
prevents producer self-attestation from becoming review evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Final

from evrptw.experiment_lifecycle import _write_signed_json

SCHEMA_VERSION: Final = "experiment-legacy-independent-review-v1"


def review_legacy_run(*, manifest_path: Path) -> dict[str, object]:
    """Verify the signed manifest identity, then fail closed without an adapter."""

    data = manifest_path.read_bytes()
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256").read_text(
        encoding="utf-8"
    )
    fields = sidecar.strip().split()
    if (
        len(fields) != 2
        or fields[1] != manifest_path.name
        or fields[0] != hashlib.sha256(data).hexdigest()
    ):
        raise ValueError("legacy manifest sidecar identity differs")
    payload = json.loads(data)
    if not isinstance(payload, dict) or not isinstance(payload.get("run_label"), str):
        raise ValueError("legacy manifest run identity is invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_label": payload["run_label"],
        "status": "INVALID",
        "failure_identity": {
            "component": "lifecycle_legacy_review",
            "invariant_or_check": "stage_specific_replay_adapter",
            "location": manifest_path.name,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    run_dir = arguments.run_dir.resolve()
    manifest_path = arguments.manifest.resolve(strict=True)
    try:
        manifest_path.relative_to(run_dir)
    except ValueError as error:
        parser.error(f"--manifest must be below --run-dir: {error}")
    raw_sha256_before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    review = review_legacy_run(manifest_path=manifest_path)
    review_dir = run_dir / "review"
    review_path = review_dir / "review_manifest.json"
    review_sha256 = _write_signed_json(review_path, review)
    raw_sha256_after = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    _write_signed_json(
        review_dir / "review_execution.json",
        {
            "schema_version": "experiment-review-execution-v1",
            "run_label": review["run_label"],
            "status": "completed",
            "finalized": True,
            "exit_code": 0,
            "reviewer_module_name": "evrptw.experiments.lifecycle_legacy_review",
            "reviewer_installed_distribution_digest": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "raw_manifest_sha256_before": raw_sha256_before,
            "raw_manifest_sha256_after": raw_sha256_after,
            "raw_manifest_unchanged": raw_sha256_before == raw_sha256_after,
            "review_manifest_sha256": review_sha256,
        },
    )
    print(review_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
