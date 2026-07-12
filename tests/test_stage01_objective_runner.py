from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from evrptw.experiments.stage00_baseline import load_config, run_stage00
from evrptw.experiments.stage01_objective import (
    STAGE01_PER_RUN_FIELDS,
    build_objective_ranking_report,
    run_stage01_objective,
)
from evrptw.experiments.week02_baseline_comparison import write_schneider_instance
from evrptw.models import Instance, Node, NodeType, Vehicle


def _instance() -> Instance:
    return Instance(
        "runnerC5",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("F1", NodeType.STATION, 2.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 100.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 3.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(10.0, 5.0, 1.0, 0.1, 1.0),
    )


def _config_file(tmp_path: Path) -> Path:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    write_schneider_instance(_instance(), benchmark / "runnerC5.txt")
    path = tmp_path / "stage01.toml"
    path.write_text(
        "\n".join(
            (
                "[stage00]",
                'schema_version = "1"',
                'baseline_id = "stage01-test"',
                'algorithm = "ALNS_EXACT_CHARGING"',
                "",
                "[benchmark]",
                f'directory = "{benchmark}"',
                'instances = ["runnerC5"]',
                "",
                "[run]",
                "seeds = [2014]",
                "time_limit_seconds = 1.0",
                "max_iterations = 5",
                "threads = 1",
                'command = "stage01-test"',
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage01_runner_separates_objective_levels_and_preserves_stage00(tmp_path: Path) -> None:
    config_path = _config_file(tmp_path)
    config = load_config(config_path)
    baseline_dir = tmp_path / "baseline"
    run_stage00(config, config_path, tmp_path / "stage00-results", baseline_dir=baseline_dir)
    manifest_before = _sha256(baseline_dir / "manifest.json")

    outputs = run_stage01_objective(
        config_path=config_path,
        baseline_dir=baseline_dir,
        output_dir=tmp_path / "stage01-results",
        summary_dir=tmp_path / "summaries",
    )

    rows = list(csv.DictReader(outputs["per_run"].open(encoding="utf-8", newline="")))
    assert tuple(rows[0]) == STAGE01_PER_RUN_FIELDS
    assert {row["algorithm"] for row in rows} == {
        "ALNS_EXACT_CHARGING",
        "BRANCH_PRICE_AND_CUT",
    }
    for row in rows:
        assert row["objective_schema"] == "vehicles,distance,charging_time,charging_count"
        assert row["primary_vehicle_count"]
        assert row["secondary_total_distance"]
        assert row["tertiary_total_charging_time"]
        assert row["quaternary_charging_count"]
        assert row["objective_key"]
    assert outputs["summary"].exists()
    assert outputs["failures"].exists()
    assert outputs["ranking_changes"].exists()
    assert outputs["stage00_comparison"].exists()
    assert _sha256(baseline_dir / "manifest.json") == manifest_before


def test_ranking_report_exposes_vehicle_first_rank_reversal(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    solutions = baseline / "solutions"
    solutions.mkdir(parents=True)
    fields = (
        "instance",
        "seed",
        "vehicle_count",
        "total_distance",
        "total_charging_time",
        "solution_path",
    )
    rows = [
        {
            "instance": "rc105C5",
            "seed": "2014",
            "vehicle_count": "3",
            "total_distance": "238.0",
            "total_charging_time": "4.0",
            "solution_path": "solutions/2014.json",
        },
        {
            "instance": "rc105C5",
            "seed": "2015",
            "vehicle_count": "2",
            "total_distance": "241.0",
            "total_charging_time": "5.0",
            "solution_path": "solutions/2015.json",
        },
    ]
    with (baseline / "per_run_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (solutions / "2014.json").write_text(
        json.dumps({"routes": [["D0", "F1", "D0"], ["D0", "D0"], ["D0", "D0"]]}),
        encoding="utf-8",
    )
    (solutions / "2015.json").write_text(
        json.dumps({"routes": [["D0", "F1", "D0"]]}), encoding="utf-8"
    )

    report = build_objective_ranking_report(baseline)
    by_seed = {row["seed"]: row for row in report}

    assert by_seed["2014"]["old_distance_rank"] == 1
    assert by_seed["2014"]["new_lexicographic_rank"] == 2
    assert by_seed["2015"]["old_distance_rank"] == 2
    assert by_seed["2015"]["new_lexicographic_rank"] == 1
    assert {row["reason"] for row in report} == {"vehicle_count_priority"}
