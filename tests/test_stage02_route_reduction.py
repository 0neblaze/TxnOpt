from __future__ import annotations

from pathlib import Path

from evrptw.experiments.stage02_route_reduction import (
    ALGORITHM,
    FOCUSED_100_INSTANCES,
    load_config,
)


def test_stage02_config_keeps_formal_scope_and_operator_budgets() -> None:
    config = load_config(Path("configs/stage02_route_reduction.toml"))

    assert config.algorithm == ALGORITHM
    assert config.instances == (
        "c101C5",
        "r105C5",
        "rc105C5",
        "c104C10",
        "r103C10",
        "rc102C10",
        "c106C15",
        "r105C15",
        "rc103C15",
        "c101_21",
        "r101_21",
        "rc101_21",
    )
    assert set(FOCUSED_100_INSTANCES) <= set(config.instances)
    assert config.seeds == (2014, 2015, 2016)
    assert config.time_limit_seconds == 30.0
    assert config.max_iterations == 1000
    assert config.threads == 1
    assert config.vehicle_operator_config.max_route_elimination_attempts == 8
    assert config.vehicle_operator_config.route_elimination_exact_evaluation_budget == 256
    assert config.vehicle_operator_config.route_merge_exact_evaluation_budget == 64
    assert config.vehicle_operator_config.vehicle_repair_exact_evaluation_budget == 256
