from __future__ import annotations

import csv
import json
from pathlib import Path

from evrptw.experiments.week04_extension import run_week04_extension


def test_week04_runner_writes_battery_sweep_outputs(tmp_path: Path) -> None:
    outputs = run_week04_extension(
        output_dir=tmp_path / "week04",
        summary_dir=tmp_path / "summaries",
        scales=(8,),
        seeds=(2014,),
        battery_capacities=(70.0, 100.0),
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
        "OR_TOOLS_VRPTW_ANTICIPATORY_REPAIR",
        "GA_VRPTW",
        "GA_VRPTW_CHARGING_REPAIR",
        "GA_VRPTW_ANTICIPATORY_REPAIR",
    }
    assert {row["battery_capacity"] for row in rows} == {"70.0", "100.0"}
    assert len(rows) == 12

    for field in (
        "battery_capacity",
        "method",
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
    assert "battery_capacity" in raw_payload
    assert "metrics" in raw_payload

    summary_rows = list(csv.DictReader(summary_csv.open(encoding="utf-8", newline="")))
    assert len(summary_rows) == 12
    assert "feasible_rate" in summary_rows[0]
    assert (tmp_path / "summaries" / "week04_summary_results.csv").exists()
