from __future__ import annotations

import csv
import json
from pathlib import Path

from evrptw.experiments.week05_consolidation import run_week05_consolidation


def test_week05_runner_writes_consolidation_outputs(tmp_path: Path) -> None:
    outputs = run_week05_consolidation(
        output_dir=tmp_path / "week05",
        summary_dir=tmp_path / "summaries",
        verify_against_week04=False,
        scales=(8,),
        seeds=(2014,),
        ga_population_size=8,
        ga_generations=2,
        ortools_time_limit_seconds=1,
    )

    rows = list(csv.DictReader(outputs["per_run_csv"].open(encoding="utf-8", newline="")))
    assert {row["method"] for row in rows} == {
        "OR_TOOLS_VRPTW",
        "OR_TOOLS_VRPTW_CHARGING_REPAIR",
        "OR_TOOLS_VRPTW_ANTICIPATORY_REPAIR",
        "OR_TOOLS_VRPTW_ANTICIPATORY_SPLIT_REPAIR",
        "GA_VRPTW",
        "GA_VRPTW_CHARGING_REPAIR",
        "GA_VRPTW_ANTICIPATORY_REPAIR",
        "GA_VRPTW_ANTICIPATORY_SPLIT_REPAIR",
    }
    assert len(rows) == 8
    assert outputs["summary_csv"].exists()
    assert outputs["failure_cases_md"].exists()
    assert outputs["environment_json"].exists()
    assert outputs["reproducibility_check_md"].exists()
    assert (tmp_path / "summaries" / "week05_summary_results.csv").exists()

    split_row = next(row for row in rows if row["method"].endswith("ANTICIPATORY_SPLIT_REPAIR"))
    raw_payload = json.loads(Path(split_row["raw_json"]).read_text(encoding="utf-8"))
    assert "split_count" in raw_payload["post_processing"]
    assert "added_vehicle_count" in raw_payload["post_processing"]
    assert "Verification was disabled" in outputs["reproducibility_check_md"].read_text(
        encoding="utf-8"
    )
