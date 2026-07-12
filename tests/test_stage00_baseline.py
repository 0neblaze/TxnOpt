from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evrptw.experiments.stage00_baseline import (
    COMPARISON_FIELDS,
    Stage00Config,
    compare_results,
    load_config,
    run_stage00,
    verify_results,
)
from evrptw.experiments.week02_baseline_comparison import write_schneider_instance
from evrptw.models import Instance, Node, NodeType, Vehicle


def _instance() -> Instance:
    return Instance(
        "tinyC5",
        (
            Node("D0", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("F1", NodeType.STATION, 2.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 100.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 3.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(10.0, 5.0, 1.0, 0.1, 1.0),
    )


def _config_file(tmp_path: Path) -> Path:
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    write_schneider_instance(_instance(), benchmark_dir / "tinyC5.txt")
    path = tmp_path / "stage00.toml"
    path.write_text(
        "\n".join(
            (
                '[stage00]',
                'schema_version = "1"',
                'baseline_id = "test-stage00"',
                'algorithm = "ALNS_EXACT_CHARGING"',
                '',
                '[benchmark]',
                f'directory = "{benchmark_dir}"',
                'instances = ["tinyC5"]',
                '',
                '[run]',
                'seeds = [2014, 2015]',
                'time_limit_seconds = 1.0',
                'max_iterations = 5',
                'threads = 1',
                'command = "uv run python -m evrptw.experiments.stage00_baseline run"',
                '',
            )
        ),
        encoding="utf-8",
    )
    return path


def test_user_can_load_the_frozen_stage00_configuration(tmp_path: Path) -> None:
    config = load_config(_config_file(tmp_path))

    assert config == Stage00Config(
        schema_version="1",
        baseline_id="test-stage00",
        algorithm="ALNS_EXACT_CHARGING",
        benchmark_dir=tmp_path / "benchmark",
        instances=("tinyC5",),
        seeds=(2014, 2015),
        time_limit_seconds=1.0,
        max_iterations=5,
        threads=1,
        command="uv run python -m evrptw.experiments.stage00_baseline run",
    )


def test_run_verify_and_compare_form_a_reproducible_public_workflow(tmp_path: Path) -> None:
    config_path = _config_file(tmp_path)
    config = load_config(config_path)
    output_dir = tmp_path / "results"
    baseline_dir = tmp_path / "baseline"

    run_stage00(config, config_path, output_dir, baseline_dir=baseline_dir)
    verified = verify_results(config, baseline_dir, require_manifest=True)
    report_path = tmp_path / "comparison.csv"
    rows = compare_results(config, baseline_dir, baseline_dir, report_path)

    assert verified == 2
    assert rows
    assert {row["classification"] for row in rows} == {"unchanged"}
    assert {row["gate_status"] for row in rows} == {"pass"}
    with report_path.open(encoding="utf-8", newline="") as handle:
        assert tuple(next(csv.reader(handle))) == COMPARISON_FIELDS
    assert (baseline_dir / "failure_cases.csv").read_text(encoding="utf-8").count("\n") == 1
    assert (baseline_dir / "manifest.json").exists()
    assert not (baseline_dir / "raw").exists()


def test_run_refuses_to_overwrite_an_existing_target(tmp_path: Path) -> None:
    config_path = _config_file(tmp_path)
    config = load_config(config_path)
    output_dir = tmp_path / "results"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        run_stage00(config, config_path, output_dir)


def test_verify_fails_on_a_tampered_solution(tmp_path: Path) -> None:
    config_path = _config_file(tmp_path)
    config = load_config(config_path)
    baseline_dir = tmp_path / "baseline"
    run_stage00(config, config_path, tmp_path / "results", baseline_dir=baseline_dir)
    solution = next((baseline_dir / "solutions").glob("*.json"))
    payload = json.loads(solution.read_text(encoding="utf-8"))
    payload["routes"] = []
    solution.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="failed unified validator"):
        verify_results(config, baseline_dir)


def test_manifest_detects_a_tampered_frozen_csv(tmp_path: Path) -> None:
    config_path = _config_file(tmp_path)
    config = load_config(config_path)
    baseline_dir = tmp_path / "baseline"
    run_stage00(config, config_path, tmp_path / "results", baseline_dir=baseline_dir)
    per_run = baseline_dir / "per_run_results.csv"
    per_run.write_text(per_run.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest checksum mismatch"):
        verify_results(config, baseline_dir, require_manifest=True)


def test_verify_fails_when_a_seed_record_is_missing(tmp_path: Path) -> None:
    config_path = _config_file(tmp_path)
    config = load_config(config_path)
    output_dir = tmp_path / "results"
    run_stage00(config, config_path, output_dir)
    path = output_dir / "per_run_results.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows[:-1])

    with pytest.raises(ValueError, match="missing run records"):
        verify_results(config, output_dir)


def test_verify_fails_when_a_seed_record_is_duplicated(tmp_path: Path) -> None:
    config_path = _config_file(tmp_path)
    config = load_config(config_path)
    output_dir = tmp_path / "results"
    run_stage00(config, config_path, output_dir)
    path = output_dir / "per_run_results.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows([*rows, rows[0]])

    with pytest.raises(ValueError, match="duplicate run records"):
        verify_results(config, output_dir)


def test_verify_fails_when_a_failure_record_is_deleted(tmp_path: Path) -> None:
    config_path = _config_file(tmp_path)
    config = load_config(config_path)
    output_dir = tmp_path / "results"
    run_stage00(config, config_path, output_dir)
    path = output_dir / "per_run_results.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    rows[0]["feasible"] = "False"
    rows[0]["status"] = "invalid"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="does not retain every failed run"):
        verify_results(config, output_dir)


def test_compare_classifies_improvement_regression_and_unchanged(tmp_path: Path) -> None:
    config_path = _config_file(tmp_path)
    config = load_config(config_path)
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    run_stage00(config, config_path, tmp_path / "baseline-results", baseline_dir=baseline_dir)
    run_stage00(config, config_path, candidate_dir)

    path = candidate_dir / "summary_results.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    for row in rows:
        if row["metric"] == "total_distance":
            row["mean"] = str(float(row["mean"]) - 1.0)
        elif row["metric"] == "vehicle_count":
            row["mean"] = str(float(row["mean"]) + 1.0)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    comparison = compare_results(
        config, baseline_dir, candidate_dir, tmp_path / "comparison.csv"
    )
    by_metric = {
        (row["metric"], row["statistic"]): row["classification"] for row in comparison
    }
    assert by_metric[("total_distance", "mean")] == "improvement"
    assert by_metric[("vehicle_count", "mean")] == "regression"
    assert by_metric[("iterations", "mean")] == "unchanged"
