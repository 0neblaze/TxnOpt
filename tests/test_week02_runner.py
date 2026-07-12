from __future__ import annotations

import csv
import json
from pathlib import Path

from evrptw.experiments.week02_baseline_comparison import run_week02_comparison


def test_week02_runner_writes_summary_and_raw_outputs(tmp_path: Path) -> None:
    outputs = run_week02_comparison(
        output_dir=tmp_path,
        scales=(50,),
        seed=2014,
        ga_population_size=10,
        ga_generations=2,
        ortools_time_limit_seconds=1,
    )

    results_csv = outputs["results_csv"]
    convergence_csv = outputs["convergence_csv"]
    environment_json = outputs["environment_json"]

    assert results_csv.exists()
    assert convergence_csv.exists()
    assert environment_json.exists()

    rows = list(csv.DictReader(results_csv.open(encoding="utf-8", newline="")))
    assert {row["method"] for row in rows} == {"OR_TOOLS_VRPTW", "GA_VRPTW"}
    for field in (
        "objective_value",
        "feasibility_status",
        "runtime_seconds",
        "violation_count",
        "raw_json",
    ):
        assert field in rows[0]

    convergence_rows = list(csv.DictReader(convergence_csv.open(encoding="utf-8", newline="")))
    assert len(convergence_rows) == 3
    assert convergence_rows[-1]["generation"] == "2"
    assert "best_objective" in convergence_rows[-1]
    assert "average_objective" in convergence_rows[-1]
    assert "feasible_rate" in convergence_rows[-1]

    raw_payload = json.loads(Path(rows[-1]["raw_json"]).read_text(encoding="utf-8"))
    assert "convergence" in raw_payload
    assert raw_payload["customer_count"] == 50
