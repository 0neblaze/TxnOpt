from __future__ import annotations

import argparse
import json
from pathlib import Path

from evrptw.experiments.stage03_measurement_review import (
    READY_FOR_STAGE032_FORMAL,
    READY_FOR_STAGE033,
    review_run,
)


def review_stage032(
    *,
    run_dir: Path,
    summary_dir: Path | None = None,
    review_label: str | None = None,
) -> dict[str, Path]:
    """Independently replay and audit Stage 3.2 raw evidence."""

    return review_run(
        run_dir=run_dir,
        summary_dir=summary_dir,
        review_label=review_label,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay-audit Stage 3.2 cache/incremental raw evidence"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--review-label")
    arguments = parser.parse_args()
    outputs = review_stage032(
        run_dir=arguments.run_dir,
        summary_dir=arguments.summary_dir,
        review_label=arguments.review_label,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    status = json.loads(outputs["review_manifest"].read_text(encoding="utf-8"))["status"]
    return 0 if status in {READY_FOR_STAGE032_FORMAL, READY_FOR_STAGE033} else 1


if __name__ == "__main__":
    raise SystemExit(main())
