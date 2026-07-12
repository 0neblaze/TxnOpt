from __future__ import annotations

import csv
import json
from pathlib import Path

from evrptw.experiments.week05_infrastructure_augmentation import (
    run_week05_infrastructure_augmentation,
)


def test_week05_infrastructure_runner_records_both_variants_and_manifest(tmp_path: Path) -> None:
    outputs = run_week05_infrastructure_augmentation(
        output_dir=tmp_path / "week05-infrastructure",
        summary_dir=tmp_path / "summaries",
        scales=(50,),
        seeds=(2015,),
        ga_population_size=8,
        ga_generations=2,
        ortools_time_limit_seconds=1,
    )

    rows = list(csv.DictReader(outputs["per_run_csv"].open(encoding="utf-8", newline="")))
    assert len(rows) == 4
    assert {row["infrastructure_variant"] for row in rows} == {"original", "augmented"}
    assert {row["method"] for row in rows} == {
        "OR_TOOLS_VRPTW_ANTICIPATORY_SPLIT_REPAIR",
        "GA_VRPTW_ANTICIPATORY_SPLIT_REPAIR",
    }
    assert all("added_station_count" in row for row in rows)

    augmented_rows = [row for row in rows if row["infrastructure_variant"] == "augmented"]
    assert all(row["feasible"] == "True" for row in augmented_rows)
    assert all(row["energy_violations"] == "0" for row in augmented_rows)
    assert all(int(row["added_station_count"]) > 0 for row in augmented_rows)

    manifests = sorted(outputs["manifest_dir"].glob("*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["added_station_count"] == int(augmented_rows[0]["added_station_count"])
    assert outputs["acceptance_md"].exists()
    assert "every augmented run is feasible" in outputs["acceptance_md"].read_text(
        encoding="utf-8"
    )
    assert (tmp_path / "summaries" / "week05_infrastructure_summary_results.csv").exists()
