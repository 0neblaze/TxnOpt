from __future__ import annotations

import csv
import json
from pathlib import Path

from evrptw.experiments.week03_evaluation import run_week03_evaluation


def test_week03_runner_writes_required_outputs(tmp_path: Path) -> None:
    outputs = run_week03_evaluation(
        output_dir=tmp_path / "week03",
        summary_dir=tmp_path / "summaries",
        scales=(8,),
        seeds=(2014,),
        battery_capacity=70.0,
        ga_population_size=8,
        ga_generations=2,
        ortools_time_limit_seconds=1,
    )

    per_run_csv = outputs["per_run_csv"]
    summary_csv = outputs["summary_csv"]
    failure_cases_md = outputs["failure_cases_md"]

    assert per_run_csv.exists()
    assert summary_csv.exists()
    assert failure_cases_md.exists()

    rows = list(csv.DictReader(per_run_csv.open(encoding="utf-8", newline="")))
    assert {row["method"] for row in rows} == {
        "OR_TOOLS_VRPTW",
        "OR_TOOLS_VRPTW_CHARGING_REPAIR",
        "GA_VRPTW",
        "GA_VRPTW_CHARGING_REPAIR",
    }
    for field in (
        "feasible",
        "objective_value",
        "runtime_seconds",
        "energy_violations",
        "charging_count",
        "first_infeasible_step",
        "raw_json",
    ):
        assert field in rows[0]

    raw_payload = json.loads(Path(rows[0]["raw_json"]).read_text(encoding="utf-8"))
    assert "metrics" in raw_payload

    summary_rows = list(csv.DictReader(summary_csv.open(encoding="utf-8", newline="")))
    assert len(summary_rows) == 4
    assert "feasible_rate" in summary_rows[0]
    assert (tmp_path / "summaries" / "week03_summary_results.csv").exists()
