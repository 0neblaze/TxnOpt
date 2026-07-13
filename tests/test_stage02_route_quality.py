from __future__ import annotations

from pathlib import Path

from evrptw.experiments.stage02_route_reduction import (
    ROUTE_QUALITY_ALGORITHM,
    load_config,
)


def test_stage02_route_quality_config_keeps_formal_scope_and_fixed_comparison() -> None:
    config = load_config(Path("configs/stage02_route_quality.toml"))

    assert config.algorithm == ROUTE_QUALITY_ALGORITHM
    assert config.operator_profile.value == "stage02_route_quality"
    assert config.comparison_per_run.as_posix().endswith(
        "experiments/summaries/stage02_attempt02_per_run_results.csv"
    )
    assert config.comparison_label == "stage02_1"
    assert config.comparison_gate == "stage02_1_best_objective"
    assert len(config.instances) == 12
    assert config.seeds == (2014, 2015, 2016)
    assert config.time_limit_seconds == 30.0
    assert config.max_iterations == 1000
    assert config.threads == 1
    assert config.vehicle_operator_config.route_segment_min_length == 2
    assert config.vehicle_operator_config.route_segment_max_length == 5
    assert config.vehicle_operator_config.ejection_chain_max_depth == 3
    assert config.vehicle_operator_config.ejection_chain_beam_width == 16
