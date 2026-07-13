import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.stage03_artifact_migration import (
    CANONICAL_LABEL_RE,
    canonical_filename,
    canonical_path_is_valid,
    compare_raw_to_summary,
    legacy_run_directory,
)


def test_canonical_artifact_name_keeps_stage_id_dot_out_of_extension() -> None:
    run_label = "stage03.0_measurement_attempt05"
    filename = canonical_filename(
        run_label,
        "solution",
        ".json",
        "c101C5",
        "2014",
    )

    assert filename == (
        "stage03.0_measurement_attempt05_solution_c101C5_2014.json"
    )
    assert CANONICAL_LABEL_RE.fullmatch(run_label)
    assert canonical_path_is_valid(
        "results/stage03.0_measurement_attempt05/c101C5/2014/" + filename,
        run_label,
        "solution",
        "c101C5",
        "2014",
    )


def test_legacy_directory_mapping_preserves_historical_spelling(tmp_path) -> None:
    measurement = tmp_path / "results" / "stage03-measurement_smoke01"
    screening = tmp_path / "results" / "stage031-cheap-screening_smoke01"
    measurement.mkdir(parents=True)
    screening.mkdir(parents=True)

    assert legacy_run_directory(tmp_path, "stage03_measurement_smoke01") == measurement
    assert (
        legacy_run_directory(tmp_path, "stage031_cheap_screening_smoke01")
        == screening
    )


def test_raw_to_summary_compares_semantics_not_runtime_metadata(tmp_path) -> None:
    fields = [
        "instance",
        "seed",
        "objective_key",
        "vehicle_count",
        "total_distance",
        "total_charging_time",
        "charging_count",
        "feasible",
        "trace_exact_calls",
        "trace_cache_hits",
        "trace_precomputed_routes",
        "trace_route_evaluations",
        "trace_deadline_events",
        "trace_reconciliation_status",
    ]
    row = {
        "instance": "c101C5",
        "seed": "2014",
        "objective_key": "[2, 10.0, 20.0, 1]",
        "vehicle_count": "2",
        "total_distance": "10.0",
        "total_charging_time": "20.0",
        "charging_count": "1",
        "feasible": "True",
        "trace_exact_calls": "3",
        "trace_cache_hits": "4",
        "trace_precomputed_routes": "0",
        "trace_route_evaluations": "7",
        "trace_deadline_events": "0",
        "trace_reconciliation_status": "pass",
    }
    raw_path = tmp_path / "raw.csv"
    summary_path = tmp_path / "summary.csv"
    for path, runtime in ((raw_path, "0.1"), (summary_path, "0.2")):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*fields, "runtime_seconds"])
            writer.writeheader()
            writer.writerow({**row, "runtime_seconds": runtime})

    assert compare_raw_to_summary(raw_path, summary_path, screening=False) == (
        "pass",
        [],
    )


def test_published_stage03_registries_cover_only_completed_substages() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "stage03.0": {
            "stage03.0_measurement_attempt01",
            "stage03.0_measurement_attempt02",
            "stage03.0_measurement_attempt03",
            "stage03.0_measurement_attempt04",
            "stage03.0_measurement_attempt05",
        },
        "stage03.1": {
            "stage03.1_screening_attempt01",
            "stage03.1_screening_attempt02",
            "stage03.1_screening_attempt03",
            "stage03.1_screening_attempt04",
        },
    }

    for stage_id, labels in expected.items():
        manifest_path = next(
            (root / "experiments" / "manifests").glob(f"{stage_id}_*_artifact_manifest.json")
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["solver_rerun_performed"] is False
        assert manifest["checks"]["formal_raw_manifest_integrity"] is True
        assert manifest["checks"]["formal_raw_to_summary"] is True
        assert all("stage03.2" not in value for value in manifest["canonical_run_labels"])

        registry_path = root / manifest["artifact_registry"]
        with registry_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert {row["run_label"] for row in rows} == labels
        assert len({row["canonical_path"] for row in rows}) == len(rows)
