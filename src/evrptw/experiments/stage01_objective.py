from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evrptw.alns import ALNSResult, solve_alns
from evrptw.bpc import BPCResult, solve_branch_price_and_cut
from evrptw.environment import collect_environment
from evrptw.experiments.stage00_baseline import Stage00Config, load_config, verify_results
from evrptw.models import Instance
from evrptw.objective import (
    ObjectiveComparison,
    SolutionObjective,
    compare_objectives,
    count_charging_visits,
)
from evrptw.parser import parse_schneider
from evrptw.validation import SolutionReport, validate_routes

OBJECTIVE_SCHEMA = "vehicles,distance,charging_time,charging_count"

STAGE01_PER_RUN_FIELDS = (
    "experiment_id",
    "instance",
    "algorithm",
    "seed",
    "objective_schema",
    "objective_key",
    "primary_vehicle_count",
    "secondary_total_distance",
    "tertiary_total_charging_time",
    "quaternary_charging_count",
    "total_energy",
    "total_charged_energy",
    "status",
    "feasible",
    "proven_optimal",
    "runtime_seconds",
    "iterations",
    "accepted_moves",
    "improving_moves",
    "rejected_moves",
    "charging_subproblem_calls",
    "charging_subproblem_average_seconds",
    "root_search_bound",
    "final_search_bound",
    "search_gap",
    "failure_reason",
    "raw_log_path",
    "solution_path",
)

SUMMARY_FIELDS = (
    "instance",
    "algorithm",
    "runs",
    "feasible_runs",
    "feasibility_rate",
    "best_objective_key",
    "best_vehicle_count",
    "best_total_distance",
    "best_total_charging_time",
    "best_charging_count",
    "mean_vehicle_count",
    "mean_total_distance",
    "mean_total_charging_time",
    "mean_charging_count",
    "mean_runtime_seconds",
)

RANKING_FIELDS = (
    "instance",
    "seed",
    "vehicle_count",
    "total_distance",
    "total_charging_time",
    "charging_count",
    "old_distance_rank",
    "new_lexicographic_rank",
    "rank_delta",
    "old_objective",
    "new_objective_key",
    "ordering_changed",
    "reason",
)

COMPARISON_FIELDS = (
    "instance",
    "baseline_best_objective",
    "candidate_best_objective",
    "classification",
    "baseline_feasibility_rate",
    "candidate_feasibility_rate",
    "gate_status",
    "reason",
)


def run_stage01_objective(
    *,
    config_path: Path,
    baseline_dir: Path,
    output_dir: Path,
    summary_dir: Path,
) -> dict[str, Path]:
    config = load_config(config_path)
    verify_results(config, baseline_dir, require_manifest=True)
    if output_dir.exists():
        raise FileExistsError(f"Stage 1 output directory already exists: {output_dir}")
    tracked_names = (
        "stage01_per_run_results.csv",
        "stage01_summary_results.csv",
        "stage01_failure_cases.csv",
        "stage01_objective_ranking_changes.csv",
        "stage01_stage00_comparison.csv",
    )
    existing = [summary_dir / name for name in tracked_names if (summary_dir / name).exists()]
    if existing:
        raise FileExistsError(f"tracked Stage 1 summaries already exist: {existing}")
    output_dir.mkdir(parents=True)
    raw_dir = output_dir / "raw"
    solution_dir = output_dir / "solutions"
    raw_dir.mkdir()
    solution_dir.mkdir()
    per_run = output_dir / "stage01_per_run_results.csv"
    summary = output_dir / "stage01_summary_results.csv"
    failures = output_dir / "stage01_failure_cases.csv"
    ranking = output_dir / "stage01_objective_ranking_changes.csv"
    comparison = output_dir / "stage01_stage00_comparison.csv"
    environment = output_dir / "stage01_environment.json"

    rows: list[dict[str, Any]] = []
    for name in config.instances:
        instance = parse_schneider(config.benchmark_dir / f"{name}.txt")
        for seed in config.seeds:
            alns_result = solve_alns(
                instance,
                seed=seed,
                max_iterations=config.max_iterations,
                time_limit_seconds=config.time_limit_seconds,
                operator_profile="baseline",
            )
            row = _record_alns(instance, seed, alns_result, output_dir, raw_dir, solution_dir)
            rows.append(row)
            _fail_fast_after_persisting(rows, per_run, failures)
        if len(instance.customers) <= 5:
            bpc_result = solve_branch_price_and_cut(
                instance,
                time_limit_seconds=config.time_limit_seconds,
                max_customers=8,
            )
            row = _record_bpc(instance, bpc_result, output_dir, raw_dir, solution_dir)
            rows.append(row)
            _fail_fast_after_persisting(rows, per_run, failures)
    summary_rows = _summarize(rows)
    ranking_rows = build_objective_ranking_report(baseline_dir, benchmark_dir=config.benchmark_dir)
    comparison_rows = _compare_with_stage00(
        baseline_dir,
        config,
        [row for row in rows if row["algorithm"] == "ALNS_EXACT_CHARGING"],
    )
    _write_csv(per_run, STAGE01_PER_RUN_FIELDS, rows)
    _write_csv(summary, SUMMARY_FIELDS, summary_rows)
    _write_csv(
        failures,
        STAGE01_PER_RUN_FIELDS,
        [row for row in rows if not bool(row["feasible"])],
    )
    _write_csv(ranking, RANKING_FIELDS, ranking_rows)
    _write_csv(comparison, COMPARISON_FIELDS, comparison_rows)
    _write_json(
        environment,
        {
            "captured_at": datetime.now(UTC).isoformat(),
            "objective_schema": OBJECTIVE_SCHEMA,
            "objective_precision_digits": 9,
            "vehicle_increase_policy": "always_reject",
            "configuration_sha256": _sha256(config_path),
            "baseline_manifest_sha256": _sha256(baseline_dir / "manifest.json"),
            "repository_revision": _git_revision(),
            "repository_dirty": _git_dirty(),
            "environment": collect_environment(),
        },
    )
    if any(row["gate_status"] == "fail" for row in comparison_rows):
        raise RuntimeError("Stage 1 failed at least one Stage 0 regression gate")

    summary_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        "per_run": (per_run, summary_dir / per_run.name),
        "summary": (summary, summary_dir / summary.name),
        "failures": (failures, summary_dir / failures.name),
        "ranking_changes": (ranking, summary_dir / ranking.name),
        "stage00_comparison": (comparison, summary_dir / comparison.name),
    }
    for source, destination in copies.values():
        if destination.exists():
            raise FileExistsError(f"tracked Stage 1 summary already exists: {destination}")
        destination.write_bytes(source.read_bytes())
    return {
        **{name: destination for name, (_, destination) in copies.items()},
        "environment": environment,
        "raw_dir": raw_dir,
        "solution_dir": solution_dir,
    }


def build_objective_ranking_report(
    baseline_dir: Path,
    *,
    benchmark_dir: Path,
) -> list[dict[str, Any]]:
    rows = _read_csv(baseline_dir / "per_run_results.csv")
    enriched: list[tuple[dict[str, str], SolutionObjective]] = []
    for row in rows:
        payload = json.loads((baseline_dir / row["solution_path"]).read_text(encoding="utf-8"))
        routes = [list(route) for route in payload["routes"]]
        instance = parse_schneider(benchmark_dir / f"{row['instance']}.txt")
        charging_count = count_charging_visits(
            instance,
            routes,
        )
        enriched.append(
            (
                row,
                SolutionObjective(
                    int(row["vehicle_count"]),
                    float(row["total_distance"]),
                    float(row["total_charging_time"]),
                    charging_count,
                ),
            )
        )

    output: list[dict[str, Any]] = []
    for instance_name in sorted({row["instance"] for row, _ in enriched}):
        group = [
            (row, objective) for row, objective in enriched if row["instance"] == instance_name
        ]
        old_order = sorted(group, key=lambda item: (item[1].total_distance, int(item[0]["seed"])))
        new_order = sorted(group, key=lambda item: (item[1].key, int(item[0]["seed"])))
        old_rank = {row["seed"]: rank for rank, (row, _) in enumerate(old_order, start=1)}
        new_rank = {row["seed"]: rank for rank, (row, _) in enumerate(new_order, start=1)}
        vehicle_counts = {objective.vehicle_count for _, objective in group}
        for row, objective in sorted(group, key=lambda item: int(item[0]["seed"])):
            changed = old_rank[row["seed"]] != new_rank[row["seed"]]
            reason = "unchanged"
            if changed:
                reason = (
                    "vehicle_count_priority"
                    if len(vehicle_counts) > 1
                    else "charging_time_or_count_tiebreak"
                )
            output.append(
                {
                    "instance": instance_name,
                    "seed": row["seed"],
                    "vehicle_count": objective.vehicle_count,
                    "total_distance": objective.total_distance,
                    "total_charging_time": objective.total_charging_time,
                    "charging_count": objective.charging_count,
                    "old_distance_rank": old_rank[row["seed"]],
                    "new_lexicographic_rank": new_rank[row["seed"]],
                    "rank_delta": new_rank[row["seed"]] - old_rank[row["seed"]],
                    "old_objective": objective.total_distance,
                    "new_objective_key": _render_key(objective),
                    "ordering_changed": changed,
                    "reason": reason,
                }
            )
    return output


def _record_alns(
    instance: Instance,
    seed: int,
    result: ALNSResult,
    output_dir: Path,
    raw_dir: Path,
    solution_dir: Path,
) -> dict[str, Any]:
    report = validate_routes(
        instance,
        [list(route) for route in result.routes],
        claimed_objective=result.objective_value,
    )
    objective, validation_failure = _validated_objective(instance, report, result.objective)
    identifier = f"{instance.name}-alns_exact_charging-{seed}"
    row = _base_row(identifier, instance.name, "ALNS_EXACT_CHARGING", seed, objective, report)
    row.update(
        {
            "proven_optimal": False,
            "runtime_seconds": result.runtime_seconds,
            "iterations": result.iterations,
            "accepted_moves": result.accepted_moves,
            "improving_moves": result.improving_moves,
            "rejected_moves": result.rejected_moves,
            "charging_subproblem_calls": result.charging_subproblem_calls,
            "charging_subproblem_average_seconds": (
                result.charging_subproblem_time / result.charging_subproblem_calls
                if result.charging_subproblem_calls
                else 0.0
            ),
            "failure_reason": _join_failures(result.failure_reason, validation_failure),
        }
    )
    _persist(row, result, result.routes, output_dir, raw_dir, solution_dir)
    return row


def _record_bpc(
    instance: Instance,
    result: BPCResult,
    output_dir: Path,
    raw_dir: Path,
    solution_dir: Path,
) -> dict[str, Any]:
    report = validate_routes(
        instance,
        [list(route) for route in result.routes],
        claimed_objective=result.objective_value,
    )
    objective, validation_failure = _validated_objective(instance, report, result.objective)
    identifier = f"{instance.name}-branch_price_and_cut-0"
    row = _base_row(identifier, instance.name, "BRANCH_PRICE_AND_CUT", 0, objective, report)
    row.update(
        {
            "proven_optimal": result.proven_optimal,
            "runtime_seconds": result.runtime_seconds,
            "root_search_bound": result.root_search_bound,
            "final_search_bound": result.final_search_bound,
            "search_gap": result.search_gap,
            "failure_reason": _join_failures(result.failure_reason, validation_failure),
        }
    )
    _persist(row, result, result.routes, output_dir, raw_dir, solution_dir)
    return row


def _base_row(
    identifier: str,
    instance: str,
    algorithm: str,
    seed: int,
    objective: SolutionObjective | None,
    report: SolutionReport,
) -> dict[str, Any]:
    accepted = report.feasible and objective is not None
    row: dict[str, Any] = {field: "" for field in STAGE01_PER_RUN_FIELDS}
    row.update(
        {
            "experiment_id": identifier,
            "instance": instance,
            "algorithm": algorithm,
            "seed": seed,
            "objective_schema": OBJECTIVE_SCHEMA,
            "objective_key": _render_key(objective) if objective is not None else "",
            "primary_vehicle_count": objective.vehicle_count if objective is not None else "",
            "secondary_total_distance": (objective.total_distance if objective is not None else ""),
            "tertiary_total_charging_time": (
                objective.total_charging_time if objective is not None else ""
            ),
            "quaternary_charging_count": (
                objective.charging_count if objective is not None else ""
            ),
            "total_energy": report.total_energy,
            "total_charged_energy": report.total_charged_energy,
            "status": "feasible" if accepted else "invalid",
            "feasible": accepted,
        }
    )
    return row


def _persist(
    row: dict[str, Any],
    result: Any,
    routes: tuple[tuple[str, ...], ...],
    output_dir: Path,
    raw_dir: Path,
    solution_dir: Path,
) -> None:
    identifier = str(row["experiment_id"])
    raw = raw_dir / f"{identifier}.json"
    solution = solution_dir / f"{identifier}.json"
    row["raw_log_path"] = str(raw.relative_to(output_dir))
    row["solution_path"] = str(solution.relative_to(output_dir))
    _write_json(raw, {"record": row, "solver_result": asdict(result)})
    _write_json(solution, {"instance": row["instance"], "routes": routes})


def _validated_objective(
    instance: Instance,
    report: SolutionReport,
    claimed: SolutionObjective | None,
) -> tuple[SolutionObjective | None, str]:
    if not report.feasible:
        violations = (*report.violations, *(v for route in report.routes for v in route.violations))
        return None, "unified validation failed: " + "; ".join(violations)
    objective = SolutionObjective.from_report(instance, report)
    if claimed is None or compare_objectives(objective, claimed) is not ObjectiveComparison.EQUAL:
        return None, "solver objective differs from unified validation"
    return objective, ""


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["instance"]), str(row["algorithm"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (instance, algorithm), group in sorted(groups.items()):
        feasible = [row for row in group if bool(row["feasible"])]
        objectives = [_objective_from_stage01_row(row) for row in feasible]
        best = min(objectives, key=lambda objective: objective.key) if objectives else None
        output.append(
            {
                "instance": instance,
                "algorithm": algorithm,
                "runs": len(group),
                "feasible_runs": len(feasible),
                "feasibility_rate": len(feasible) / len(group),
                "best_objective_key": _render_key(best) if best else "",
                "best_vehicle_count": best.vehicle_count if best else "",
                "best_total_distance": best.total_distance if best else "",
                "best_total_charging_time": best.total_charging_time if best else "",
                "best_charging_count": best.charging_count if best else "",
                "mean_vehicle_count": _mean(objectives, "vehicle_count"),
                "mean_total_distance": _mean(objectives, "total_distance"),
                "mean_total_charging_time": _mean(objectives, "total_charging_time"),
                "mean_charging_count": _mean(objectives, "charging_count"),
                "mean_runtime_seconds": statistics.fmean(
                    float(row["runtime_seconds"]) for row in group
                ),
            }
        )
    return output


def _compare_with_stage00(
    baseline_dir: Path,
    config: Stage00Config,
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_rows = _read_csv(baseline_dir / "per_run_results.csv")
    ranking = build_objective_ranking_report(baseline_dir, benchmark_dir=config.benchmark_dir)
    baseline_objectives = {
        (str(row["instance"]), str(row["seed"])): SolutionObjective(
            int(row["vehicle_count"]),
            float(row["total_distance"]),
            float(row["total_charging_time"]),
            int(row["charging_count"]),
        )
        for row in ranking
    }
    output: list[dict[str, Any]] = []
    for instance in config.instances:
        baseline_group = [row for row in baseline_rows if row["instance"] == instance]
        candidate_group = [row for row in candidate_rows if row["instance"] == instance]
        expected_seeds = {str(seed) for seed in config.seeds}
        baseline_seeds = [row["seed"] for row in baseline_group]
        candidate_seeds = [str(row["seed"]) for row in candidate_group]
        if len(baseline_seeds) != len(expected_seeds) or set(baseline_seeds) != expected_seeds:
            raise RuntimeError(f"Stage 0 baseline coverage mismatch for {instance}")
        if len(candidate_seeds) != len(expected_seeds) or set(candidate_seeds) != expected_seeds:
            raise RuntimeError(f"Stage 1 run coverage mismatch for {instance}")
        baseline_best = min(
            (baseline_objectives[(instance, row["seed"])] for row in baseline_group),
            key=lambda objective: objective.key,
        )
        candidate_objectives = [
            _objective_from_stage01_row(row) for row in candidate_group if bool(row["feasible"])
        ]
        candidate_best = (
            min(candidate_objectives, key=lambda objective: objective.key)
            if candidate_objectives
            else None
        )
        comparison = (
            compare_objectives(candidate_best, baseline_best)
            if candidate_best is not None
            else ObjectiveComparison.WORSE
        )
        baseline_rate = sum(row["feasible"] == "True" for row in baseline_group) / len(
            baseline_group
        )
        candidate_rate = sum(bool(row["feasible"]) for row in candidate_group) / len(
            candidate_group
        )
        gate = "fail" if candidate_rate < baseline_rate else "pass"
        output.append(
            {
                "instance": instance,
                "baseline_best_objective": _render_key(baseline_best),
                "candidate_best_objective": (
                    _render_key(candidate_best) if candidate_best is not None else ""
                ),
                "classification": comparison.value,
                "baseline_feasibility_rate": baseline_rate,
                "candidate_feasibility_rate": candidate_rate,
                "gate_status": gate,
                "reason": (
                    "candidate feasibility rate regressed"
                    if gate == "fail"
                    else "candidate retained Stage 0 structural feasibility"
                ),
            }
        )
    return output


def _objective_from_stage01_row(row: dict[str, Any]) -> SolutionObjective:
    return SolutionObjective(
        int(row["primary_vehicle_count"]),
        float(row["secondary_total_distance"]),
        float(row["tertiary_total_charging_time"]),
        int(row["quaternary_charging_count"]),
    )


def _join_failures(*reasons: str) -> str:
    return "; ".join(reason for reason in reasons if reason)


def _fail_fast_after_persisting(rows: list[dict[str, Any]], per_run: Path, failures: Path) -> None:
    if bool(rows[-1]["feasible"]):
        return
    _write_csv(per_run, STAGE01_PER_RUN_FIELDS, rows)
    _write_csv(
        failures,
        STAGE01_PER_RUN_FIELDS,
        [row for row in rows if not bool(row["feasible"])],
    )
    raise RuntimeError("Stage 1 recorded a failed solver run")


def _mean(objectives: list[SolutionObjective], field: str) -> float | str:
    if not objectives:
        return ""
    return statistics.fmean(float(getattr(objective, field)) for objective in objectives)


def _render_key(objective: SolutionObjective) -> str:
    return json.dumps(objective.key, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 1 lexicographic objective audit")
    parser.add_argument("--config", type=Path, default=Path("configs/stage00_baseline.toml"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("experiments/baselines/stage00"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/stage01"))
    parser.add_argument("--summary-dir", type=Path, default=Path("experiments/summaries"))
    arguments = parser.parse_args()
    outputs = run_stage01_objective(
        config_path=arguments.config,
        baseline_dir=arguments.baseline_dir,
        output_dir=arguments.output_dir,
        summary_dir=arguments.summary_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
