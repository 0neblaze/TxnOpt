from __future__ import annotations

import json
from pathlib import Path

from evrptw.experiments.gpu_batch_pilot import PilotConfig, run_profile


def test_pilot_profile_writes_gate_and_raw_manifest_without_publishing_summary(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = PilotConfig(
        root=root,
        benchmark_dir=root / "data" / "schneider",
        stage02_config=root / "configs" / "stage02_constraint_guided.toml",
        output_dir=tmp_path / "gpu_batch_pilot_attempt01",
        summary_dir=tmp_path / "summaries",
        instances=("c101C5",),
        seeds=(2014,),
        profiler_time_limit_seconds=2.0,
        profiler_max_iterations=2,
    )

    outputs = run_profile(config)

    gate = json.loads(outputs["profile_gate"].read_text(encoding="utf-8"))
    assert gate["status"] == "STOP_BEFORE_GPU"
    assert outputs["raw_manifest"].is_file()
    assert (config.output_dir / "raw_manifest.sha256").is_file()
    assert (config.output_dir / "profile" / "c101C5_2014.json").is_file()
    assert not config.summary_dir.exists()
