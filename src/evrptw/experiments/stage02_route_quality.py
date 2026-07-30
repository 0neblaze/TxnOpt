from __future__ import annotations

import argparse
from pathlib import Path

from evrptw.experiments.stage02_route_reduction import run_stage02
from evrptw.storage_governance import preflight_cli_attempt


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 2.2 cross-route quality experiments")
    parser.add_argument("--config", type=Path, default=Path("configs/stage02_route_quality.toml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stage02.2_route_quality_attempt01"),
    )
    parser.add_argument("--summary-dir", type=Path, default=Path("experiments/summaries"))
    parser.add_argument(
        "--run-label",
        default="stage02.2_route_quality_attempt01",
    )
    parser.add_argument(
        "--repeat-of",
        type=Path,
        help="first complete Stage 2.2 output directory or gate report to compare",
    )
    arguments = parser.parse_args()
    preflight_cli_attempt(
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        run_label=arguments.run_label,
    )
    outputs = run_stage02(
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        summary_dir=arguments.summary_dir,
        run_label=arguments.run_label,
        repeat_of=arguments.repeat_of,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
