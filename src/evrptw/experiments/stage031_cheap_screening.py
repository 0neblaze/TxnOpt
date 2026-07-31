from __future__ import annotations

import argparse
from pathlib import Path

from evrptw.experiments.stage03_measurement import (
    load_config,
    run_stage03,
)
from evrptw.storage_governance import (
    preflight_cli_attempt,
    seal_cli_attempt,
    seal_failed_cli_attempt,
)


def run_stage031(
    *,
    config_path: Path,
    output_dir: Path,
    scope: str = "smoke",
    run_label: str = "stage03.1_screening_attempt01",
    summary_dir: Path | None = None,
    smoke_review_dir: Path | None = None,
) -> dict[str, Path]:
    """Run Stage 3.1 raw measurement through the shared evidence writer."""

    config = load_config(config_path)
    if config.screening_config is None or not config.screening_config.enabled:
        raise ValueError("Stage 3.1 requires an enabled [screening] configuration")
    return run_stage03(
        config_path=config_path,
        output_dir=output_dir,
        scope=scope,
        run_label=run_label,
        summary_dir=summary_dir,
        smoke_review_dir=smoke_review_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 3.1 cheap-screening evidence")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage031_cheap_screening.toml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--run-label", default="stage03.1_screening_attempt01")
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--smoke-review-dir", type=Path)
    arguments = parser.parse_args()
    preflight_cli_attempt(
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        run_label=arguments.run_label,
    )
    try:
        outputs = run_stage031(
            config_path=arguments.config,
            output_dir=arguments.output_dir,
            scope=arguments.scope,
            run_label=arguments.run_label,
            summary_dir=arguments.summary_dir,
            smoke_review_dir=arguments.smoke_review_dir,
        )
        seal_cli_attempt(
            config_path=arguments.config,
            output_dir=arguments.output_dir,
            run_label=arguments.run_label,
            manifest_path=outputs["manifest"],
        )
    except BaseException as error:
        seal_failed_cli_attempt(
            config_path=arguments.config,
            output_dir=arguments.output_dir,
            run_label=arguments.run_label,
            error=error,
        )
        raise
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
