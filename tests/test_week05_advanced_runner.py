from __future__ import annotations

import csv
import json
from pathlib import Path

from evrptw.experiments.week02_baseline_comparison import write_schneider_instance
from evrptw.experiments.week05_advanced_benchmark import (
    PER_RUN_FIELDS,
    run_week05_advanced_benchmark,
)
from evrptw.models import Instance, Node, NodeType, Vehicle


def _instance() -> Instance:
    return Instance(
        "runnerC5",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("F1", NodeType.STATION, 4.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 7.0, 1.0, 1.0, 0.0, 100.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(10.0, 2.0, 1.0, 0.1, 1.0),
    )


def test_advanced_runner_writes_traceable_primary_and_stress_outputs(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    write_schneider_instance(_instance(), benchmark / "runnerC5.txt")

    outputs = run_week05_advanced_benchmark(
        benchmark_dir=benchmark,
        output_dir=tmp_path / "results",
        instance_names=("runnerC5",),
        stress_instance_names=("runnerC5",),
        seeds=(2014,),
        alns_iterations=10,
        time_limit_seconds=2.0,
        ga_population_size=4,
        ga_generations=1,
        include_stress=True,
    )

    rows = list(csv.DictReader(outputs["per_run"].open(encoding="utf-8", newline="")))
    assert len(rows) == 8
    assert {row["benchmark_type"] for row in rows} == {"Primary", "Stress"}
    assert {row["algorithm"] for row in rows} == {
        "ALNS_EXACT_CHARGING",
        "BRANCH_PRICE_AND_CUT",
        "GA_VRPTW",
        "OR_TOOLS_VRPTW",
    }
    assert tuple(rows[0]) == PER_RUN_FIELDS
    for row in rows:
        raw_path = Path(row["raw_log_path"])
        solution_path = Path(row["solution_path"])
        assert raw_path.exists()
        assert solution_path.exists()
        record = json.loads(raw_path.read_text(encoding="utf-8"))["record"]
        assert record["raw_log_path"] == str(raw_path.resolve())
    assert outputs["environment"].exists()
    assert outputs["instance_audits"].exists()
    assert outputs["summary"].exists()
    assert outputs["failures"].exists()
