from __future__ import annotations

import argparse
from pathlib import Path

from evrptw.experiments.stage03_measurement import load_config, run_stage03
from evrptw.storage_governance import preflight_cli_attempt


def run_stage032(
    *,
    config_path: Path,
    output_dir: Path,
    scope: str = "smoke",
    run_label: str = "stage03.2_cache_incremental_attempt01",
    summary_dir: Path | None = None,
    smoke_review_dir: Path | None = None,
) -> dict[str, Path]:
    """Run Stage 3.2 raw evidence with the shared immutable evidence writer."""

    config = load_config(config_path)
    if config.cache_incremental_config is None or not config.cache_incremental_config.enabled:
        raise ValueError("Stage 3.2 requires an enabled [cache_incremental] configuration")
    if config.screening_config is None or not config.screening_config.enabled:
        raise ValueError("Stage 3.2 requires an enabled [screening] configuration")
    return run_stage03(
        config_path=config_path,
        output_dir=output_dir,
        scope=scope,
        run_label=run_label,
        summary_dir=summary_dir,
        smoke_review_dir=smoke_review_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage 3.2 cache/incremental raw evidence"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage032_cache_incremental.toml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "formal"), default="smoke")
    parser.add_argument(
        "--run-label",
        default="stage03.2_cache_incremental_attempt01",
    )
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--smoke-review-dir", type=Path)
    arguments = parser.parse_args()
    preflight_cli_attempt(
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        run_label=arguments.run_label,
    )
    outputs = run_stage032(
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        scope=arguments.scope,
        run_label=arguments.run_label,
        summary_dir=arguments.summary_dir,
        smoke_review_dir=arguments.smoke_review_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
