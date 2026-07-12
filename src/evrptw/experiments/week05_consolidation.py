from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from evrptw.baselines.ga_vrptw import solve_ga_vrptw
from evrptw.baselines.ortools_vrptw import solve_vrptw
from evrptw.environment import collect_environment
from evrptw.experiments.week02_baseline_comparison import write_schneider_instance
from evrptw.experiments.week04_extension import (
    PER_RUN_FIELDS,
    SUMMARY_FIELDS,
    _as_route_lists,
    _recommended_vehicle_count,
    _record_repair_result,
    _summary_rows,
    _write_csv,
    _write_failure_cases,
    _write_json,
    generate_week04_instance,
)
from evrptw.repairs import split_routes_after_anticipatory_repair

_REPRODUCIBILITY_FIELDS = (
    "instance",
    "size",
    "battery_capacity",
    "method",
    "seed",
    "feasible",
    "objective_value",
    "vehicle_count",
    "capacity_violations",
    "time_window_violations",
    "energy_violations",
    "coverage_violations",
    "charging_count",
    "charging_time",
    "first_infeasible_step",
    "total_violation_count",
)
_RERUN_METHODS = frozenset(
    {
        "OR_TOOLS_VRPTW",
        "OR_TOOLS_VRPTW_CHARGING_REPAIR",
        "OR_TOOLS_VRPTW_ANTICIPATORY_REPAIR",
        "GA_VRPTW",
        "GA_VRPTW_CHARGING_REPAIR",
        "GA_VRPTW_ANTICIPATORY_REPAIR",
    }
)


def run_week05_consolidation(
    *,
    output_dir: Path,
    summary_dir: Path | None = Path("experiments/summaries"),
    week04_per_run_csv: Path = Path("experiments/summaries/week04_per_run_results.csv"),
    verify_against_week04: bool = True,
    scales: tuple[int, ...] = (50, 100, 200),
    seeds: tuple[int, ...] = (2014, 2015, 2016),
    battery_capacity: float = 60.0,
    ga_population_size: int = 100,
    ga_generations: int = 200,
    crossover_probability: float = 0.8,
    mutation_probability: float = 0.2,
    ortools_time_limit_seconds: int = 30,
    min_reserve_ratio: float = 0.0,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    instance_dir = output_dir / "instances"
    per_run_csv = output_dir / "week05_per_run_results.csv"
    summary_csv = output_dir / "week05_summary_results.csv"
    failure_cases_md = output_dir / "week05_failure_cases.md"
    environment_json = output_dir / "week05_environment.json"

    _write_json(
        environment_json,
        {
            "environment": collect_environment(),
            "experiment": {
                "scales": scales,
                "seeds": seeds,
                "battery_capacity": battery_capacity,
                "ga_population_size": ga_population_size,
                "ga_generations": ga_generations,
                "crossover_probability": crossover_probability,
                "mutation_probability": mutation_probability,
                "ortools_time_limit_seconds": ortools_time_limit_seconds,
                "min_reserve_ratio": min_reserve_ratio,
                "reuses_week04_instances": True,
                "allows_added_vehicles": True,
            },
        },
    )

    rows: list[dict[str, Any]] = []
    for scale in scales:
        for seed in seeds:
            instance = generate_week04_instance(
                scale,
                seed=seed,
                battery_capacity=battery_capacity,
            )
            write_schneider_instance(instance, instance_dir / f"{instance.name}.txt")
            vehicle_count = _recommended_vehicle_count(instance)

            ortools_result = solve_vrptw(
                instance,
                vehicle_count=vehicle_count,
                time_limit_seconds=ortools_time_limit_seconds,
            )
            rows.extend(
                _rows_for_constructor_with_split(
                    instance=instance,
                    seed=seed,
                    battery_capacity=battery_capacity,
                    constructor_method="OR_TOOLS_VRPTW",
                    routes=_as_route_lists(ortools_result["routes"]),
                    constructor_runtime_seconds=float(ortools_result["runtime_seconds"]),
                    raw_dir=raw_dir,
                    payload={"constructor": ortools_result},
                    min_reserve_ratio=min_reserve_ratio,
                )
            )

            ga_result = solve_ga_vrptw(
                instance,
                seed=seed,
                population_size=ga_population_size,
                generations=ga_generations,
                crossover_probability=crossover_probability,
                mutation_probability=mutation_probability,
            )
            rows.extend(
                _rows_for_constructor_with_split(
                    instance=instance,
                    seed=seed,
                    battery_capacity=battery_capacity,
                    constructor_method="GA_VRPTW",
                    routes=[list(route) for route in ga_result.routes],
                    constructor_runtime_seconds=ga_result.runtime_seconds,
                    raw_dir=raw_dir,
                    payload={"constructor": ga_result.to_dict()},
                    min_reserve_ratio=min_reserve_ratio,
                )
            )

    _write_csv(per_run_csv, PER_RUN_FIELDS, rows)
    summary_rows = _summary_rows(rows)
    _write_csv(summary_csv, SUMMARY_FIELDS, summary_rows)
    _write_failure_cases(
        failure_cases_md,
        rows,
        week_label="Week 5",
        introduction=(
            "Failure cases show where anticipatory charging and customer-boundary route "
            "splitting still fail at battery capacity 60."
        ),
    )

    outputs = {
        "per_run_csv": per_run_csv,
        "summary_csv": summary_csv,
        "failure_cases_md": failure_cases_md,
        "environment_json": environment_json,
        "raw_dir": raw_dir,
        "instance_dir": instance_dir,
    }
    if summary_dir is not None:
        summary_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(summary_dir / "week05_per_run_results.csv", PER_RUN_FIELDS, rows)
        _write_csv(summary_dir / "week05_summary_results.csv", SUMMARY_FIELDS, summary_rows)
        outputs["summary_copy_dir"] = summary_dir

    reproducibility_md = output_dir / "week05_reproducibility_check.md"
    _verify_week04_rerun(
        rows,
        baseline_csv=week04_per_run_csv,
        report_path=reproducibility_md,
        enabled=verify_against_week04,
    )
    outputs["reproducibility_check_md"] = reproducibility_md
    return outputs


def _rows_for_constructor_with_split(
    *,
    instance: Any,
    seed: int,
    battery_capacity: float,
    constructor_method: str,
    routes: list[list[str]],
    constructor_runtime_seconds: float,
    raw_dir: Path,
    payload: dict[str, Any],
    min_reserve_ratio: float,
) -> list[dict[str, Any]]:
    from evrptw.experiments.week04_extension import _rows_for_constructor

    rows = _rows_for_constructor(
        instance=instance,
        seed=seed,
        battery_capacity=battery_capacity,
        constructor_method=constructor_method,
        routes=routes,
        constructor_runtime_seconds=constructor_runtime_seconds,
        raw_dir=raw_dir,
        payload=payload,
        min_reserve_ratio=min_reserve_ratio,
    )
    split_repair = split_routes_after_anticipatory_repair(
        instance,
        routes,
        min_reserve_ratio=min_reserve_ratio,
    )
    rows.append(
        _record_repair_result(
            instance=instance,
            seed=seed,
            battery_capacity=battery_capacity,
            method=f"{constructor_method}_ANTICIPATORY_SPLIT_REPAIR",
            repair=split_repair,
            constructor_runtime_seconds=constructor_runtime_seconds,
            raw_dir=raw_dir,
            payload=payload,
        )
    )
    return rows


def _verify_week04_rerun(
    rows: list[dict[str, Any]],
    *,
    baseline_csv: Path,
    report_path: Path,
    enabled: bool,
) -> None:
    if not enabled:
        report_path.write_text(
            "# Week 5 Reproducibility Check\n\nVerification was disabled for this run.\n",
            encoding="utf-8",
        )
        return
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Week 4 per-run baseline is missing: {baseline_csv}")

    previous_rows = list(csv.DictReader(baseline_csv.open(encoding="utf-8", newline="")))
    expected = {
        _row_key(row): row
        for row in previous_rows
        if float(row["battery_capacity"]) == 60.0 and row["method"] in _RERUN_METHODS
    }
    actual = {
        _row_key(row): row
        for row in rows
        if row["method"] in _RERUN_METHODS
    }
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    differences = [
        (key, field, expected[key][field], actual[key][field])
        for key in sorted(set(expected) & set(actual))
        for field in _REPRODUCIBILITY_FIELDS
        if not _values_match(expected[key][field], actual[key][field])
    ]

    lines = [
        "# Week 5 Reproducibility Check",
        "",
        "Compared rerun Week 4 methods at battery capacity 60. Runtime and raw output paths "
        "are excluded.",
        "",
        f"- Expected rows: {len(expected)}",
        f"- Rerun rows: {len(actual)}",
        f"- Missing rows: {len(missing)}",
        f"- Extra rows: {len(extra)}",
        f"- Deterministic field differences: {len(differences)}",
    ]
    if missing or extra or differences:
        lines.extend(["", "## Differences"])
        lines.extend(f"- missing: {key}" for key in missing)
        lines.extend(f"- extra: {key}" for key in extra)
        lines.extend(
            f"- {key} / {field}: expected {expected_value}, got {actual_value}"
            for key, field, expected_value, actual_value in differences[:20]
        )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        raise RuntimeError("Week 5 rerun diverged from the tracked Week 4 deterministic results")

    lines.extend(["", "Result: all checked deterministic fields match the tracked Week 4 results."])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row["instance"]),
        str(row["battery_capacity"]),
        str(row["method"]),
        str(row["seed"]),
        str(row["size"]),
    )


def _values_match(expected: Any, actual: Any) -> bool:
    try:
        return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-9)
    except (TypeError, ValueError):
        return str(expected) == str(actual)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week 5 EVRP-TW consolidation experiments")
    parser.add_argument("--output-dir", type=Path, default=Path("results/week05"))
    parser.add_argument("--summary-dir", type=Path, default=Path("experiments/summaries"))
    parser.add_argument(
        "--week04-per-run-csv",
        type=Path,
        default=Path("experiments/summaries/week04_per_run_results.csv"),
    )
    parser.add_argument("--skip-week04-verification", action="store_true")
    parser.add_argument("--scales", default="50,100,200")
    parser.add_argument("--seeds", default="2014,2015,2016")
    parser.add_argument("--battery-capacity", type=float, default=60.0)
    parser.add_argument("--ga-population-size", type=int, default=100)
    parser.add_argument("--ga-generations", type=int, default=200)
    parser.add_argument("--crossover-probability", type=float, default=0.8)
    parser.add_argument("--mutation-probability", type=float, default=0.2)
    parser.add_argument("--ortools-time-limit-seconds", type=int, default=30)
    parser.add_argument("--min-reserve-ratio", type=float, default=0.0)
    arguments = parser.parse_args()

    outputs = run_week05_consolidation(
        output_dir=arguments.output_dir,
        summary_dir=arguments.summary_dir,
        week04_per_run_csv=arguments.week04_per_run_csv,
        verify_against_week04=not arguments.skip_week04_verification,
        scales=_parse_int_tuple(arguments.scales),
        seeds=_parse_int_tuple(arguments.seeds),
        battery_capacity=arguments.battery_capacity,
        ga_population_size=arguments.ga_population_size,
        ga_generations=arguments.ga_generations,
        crossover_probability=arguments.crossover_probability,
        mutation_probability=arguments.mutation_probability,
        ortools_time_limit_seconds=arguments.ortools_time_limit_seconds,
        min_reserve_ratio=arguments.min_reserve_ratio,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
