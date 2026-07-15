"""Stage 4 adaptive weights experiment runner.

Runs four axes per (instance, seed) comparing adaptive versus fixed weights
and wall-clock versus fixed-work (exact-call-budget) termination:

  - adaptive_wall_clock:  Stage04Config(adaptive), wall-clock termination
  - adaptive_fixed_work:   Stage04Config(adaptive), exact_call_budget=100
  - fixed_wall_clock:      Stage04Config(fixed_weights=True), wall-clock
  - fixed_fixed_work:      Stage04Config(fixed_weights=True), exact_call_budget=100

Uses the Stage 0 12-instance, 3-seed formal scope, cpu_batch backend, and the
stage02_constraint_guided operator profile.  Raw evidence is written through
the shared :class:`~evrptw.artifacts.ArtifactBundleWriter`; the per-run and
summary CSVs are registered as control artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import resource
import statistics
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evrptw.alns import ALNSResult, solve_alns
from evrptw.artifacts import (
    ArtifactBundleWriter,
    ArtifactRunContext,
    ArtifactStorageConfig,
    aggregate_diagnostic_events,
    build_stage03_critical_events,
)
from evrptw.cache_incremental import CacheIncrementalConfig, canonical_instance_hash
from evrptw.environment import collect_environment
from evrptw.exact_deadline import ExactDeadlineConfig
from evrptw.experiments.stage02_route_reduction import (
    FORMAL_INSTANCES,
    FORMAL_SEEDS,
)
from evrptw.experiments.stage02_route_reduction import (
    load_config as load_stage02_config,
)
from evrptw.experiments.stage03_measurement import (
    SMOKE_INSTANCES,
    _reference_repositories,
)
from evrptw.measurement import CheapScreeningConfig, MeasurementConfig
from evrptw.models import Instance
from evrptw.neighborhoods import VehicleOperatorConfig
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.stage04 import Stage04Config, with_fixed_weights

STAGE04_SCHEMA_VERSION = "stage04-adaptive-weights-v1"
STAGE04_RUN_LABEL = re.compile(
    r"stage04_adaptive_weights_(?:attempt|rerun)[0-9]{2}"
)
DIAGNOSTIC_AXES = (
    "adaptive_wall_clock",
    "adaptive_fixed_work",
    "fixed_wall_clock",
    "fixed_fixed_work",
)

PER_RUN_FIELDS = (
    "instance",
    "seed",
    "axis",
    "weight_mode",
    "termination_mode",
    "vehicle_count",
    "total_distance",
    "total_charging_time",
    "charging_count",
    "iterations",
    "effective_iterations",
    "accepted_moves",
    "improving_moves",
    "rejected_moves",
    "accepted_improving",
    "accepted_equal",
    "accepted_worse",
    "reheat_count",
    "restart_count",
    "intensification_active",
    "acceptance_rate",
    "initial_temperature",
    "maximum_stagnation",
    "feasible",
    "exact_started_calls",
    "exact_completed_calls",
    "termination_reason",
)

SUMMARY_FIELDS = (
    "instance",
    "metric",
    "runs",
    "best",
    "mean",
    "median",
    "worst",
    "standard_deviation",
)

_SUMMARY_METRICS = (
    "vehicle_count",
    "total_distance",
    "total_charging_time",
    "iterations",
    "effective_iterations",
    "accepted_moves",
    "improving_moves",
    "rejected_moves",
)

_SOLUTION_METRICS = {
    "vehicle_count",
    "total_distance",
    "total_charging_time",
}


@dataclass(frozen=True, slots=True)
class Stage04WeightsConfig:
    """Parsed configuration for the Stage 4 adaptive-weights runner."""

    benchmark_dir: Path
    stage02_config: Path
    seeds: tuple[int, ...]
    wall_clock_seconds: float
    max_iterations: int
    exact_call_budget: int
    watchdog_seconds: float
    batch_size: int
    stage04_config: Stage04Config
    screening_config: CheapScreeningConfig
    cache_incremental_config: CacheIncrementalConfig
    artifact_storage: ArtifactStorageConfig


def load_stage04_config(path: Path) -> Stage04WeightsConfig:
    """Load and validate the TOML configuration for the Stage 4 runner."""

    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    try:
        stage = payload["stage04"]
        run = payload["run"]
        exact = payload["exact_deadline"]
        stage04_payload = dict(payload["stage04_config"])
        config = Stage04WeightsConfig(
            benchmark_dir=Path(str(payload["benchmark"]["directory"])),
            stage02_config=Path(str(stage["stage02_config"])),
            seeds=tuple(int(value) for value in run["seeds"]),
            wall_clock_seconds=float(run["time_limit_seconds"]),
            max_iterations=int(run["max_iterations"]),
            exact_call_budget=int(exact["exact_call_budget"]),
            watchdog_seconds=float(exact["watchdog_seconds"]),
            batch_size=int(exact["batch_size"]),
            stage04_config=Stage04Config(**stage04_payload),
            screening_config=CheapScreeningConfig(**dict(payload["screening"])),
            cache_incremental_config=CacheIncrementalConfig(
                **dict(payload["cache_incremental"])
            ),
            artifact_storage=ArtifactStorageConfig(
                **dict(payload["artifact_storage"])
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid Stage 4 configuration: {error}") from error
    if config.seeds != tuple(FORMAL_SEEDS):
        raise ValueError("Stage 4 seeds must match the Stage 0 formal seeds")
    if config.wall_clock_seconds != 30.0 or config.max_iterations != 1000:
        raise ValueError("Stage 4 requires 30 seconds and 1000 iterations")
    if config.exact_call_budget != 100 or config.watchdog_seconds != 120.0:
        raise ValueError("Stage 4 requires 100 calls and a 120-second watchdog")
    if config.batch_size <= 0:
        raise ValueError("Stage 4 batch_size must be positive")
    return config


def validate_stage04_run_label(run_label: str) -> None:
    """Raise ``ValueError`` if *run_label* is not canonical."""

    if STAGE04_RUN_LABEL.fullmatch(run_label) is None:
        raise ValueError(
            "Stage 4 run label must be canonical: "
            "stage04_adaptive_weights_attemptNN or rerunNN"
        )


def _run_axis(
    instance: Instance,
    *,
    seed: int,
    axis: str,
    stage04_config: Stage04Config,
    vehicle_operator_config: VehicleOperatorConfig | None,
    screening_config: CheapScreeningConfig,
    cache_incremental_config: CacheIncrementalConfig,
    wall_clock_seconds: float,
    watchdog_seconds: float,
    exact_call_budget: int,
    max_iterations: int,
    batch_size: int,
) -> ALNSResult:
    """Run one Stage 4 axis and return the :class:`ALNSResult`."""

    if axis.endswith("_fixed_work"):
        exact_deadline_config = ExactDeadlineConfig.fixed_exact_calls(
            exact_call_budget,
            watchdog_seconds=watchdog_seconds,
        )
        time_limit = watchdog_seconds
    else:
        exact_deadline_config = ExactDeadlineConfig.wall_clock()
        time_limit = wall_clock_seconds
    return solve_alns(
        instance,
        seed=seed,
        max_iterations=max_iterations,
        time_limit_seconds=time_limit,
        operator_profile="stage02_constraint_guided",
        vehicle_operator_config=vehicle_operator_config,
        measurement_config=MeasurementConfig(),
        screening_config=screening_config,
        cache_incremental_config=cache_incremental_config,
        backend="cpu_batch",
        batch_size=batch_size,
        exact_deadline_config=exact_deadline_config,
        stage04_config=stage04_config,
    )


def _axis_metadata(axis: str) -> tuple[str, str]:
    """Return ``(weight_mode, termination_mode)`` for one axis name."""

    if axis.startswith("adaptive_"):
        weight_mode = "adaptive"
    elif axis.startswith("fixed_"):
        weight_mode = "fixed"
    else:
        raise ValueError(f"unknown axis: {axis}")
    if axis.endswith("_wall_clock"):
        termination_mode = "wall_clock"
    elif axis.endswith("_fixed_work"):
        termination_mode = "fixed_work"
    else:
        raise ValueError(f"unknown axis: {axis}")
    return weight_mode, termination_mode


def _sum_operator_field(
    result: ALNSResult, field: str
) -> int:
    """Sum a per-operator statistic across all neighborhoods."""

    return sum(
        int(stats.get(field, 0))
        for stats in result.neighborhood_statistics.values()
        if isinstance(stats, Mapping)
    )


def _per_run_row(
    instance_name: str,
    seed: int,
    axis: str,
    result: ALNSResult,
) -> dict[str, Any]:
    """Build one per-run CSV row from a solver result."""

    weight_mode, termination_mode = _axis_metadata(axis)
    objective: SolutionObjective | None = result.objective
    stats = result.stage04_statistics
    return {
        "instance": instance_name,
        "seed": seed,
        "axis": axis,
        "weight_mode": weight_mode,
        "termination_mode": termination_mode,
        "vehicle_count": objective.vehicle_count if objective is not None else "",
        "total_distance": objective.total_distance if objective is not None else "",
        "total_charging_time": (
            objective.total_charging_time if objective is not None else ""
        ),
        "charging_count": objective.charging_count if objective is not None else "",
        "iterations": result.iterations,
        "effective_iterations": result.effective_iterations,
        "accepted_moves": result.accepted_moves,
        "improving_moves": result.improving_moves,
        "rejected_moves": result.rejected_moves,
        "accepted_improving": _sum_operator_field(result, "accepted_improving"),
        "accepted_equal": _sum_operator_field(result, "accepted_equal"),
        "accepted_worse": _sum_operator_field(result, "accepted_worse"),
        "reheat_count": stats.get("reheat_count", 0),
        "restart_count": stats.get("restart_count", 0),
        "intensification_active": stats.get("intensification_active", False),
        "acceptance_rate": stats.get("acceptance_rate", 0.0),
        "initial_temperature": stats.get("initial_temperature", 0.0),
        "maximum_stagnation": stats.get("maximum_stagnation", result.maximum_stagnation),
        "feasible": result.feasible,
        "exact_started_calls": result.exact_started_calls,
        "exact_completed_calls": result.exact_completed_calls,
        "termination_reason": result.termination_reason,
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-run rows into the Stage 0 summary format.

    Each (instance, metric) pair produces one row.  Solution metrics are
    summarised over feasible runs only; search-control metrics over all runs.
    """

    by_instance: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_instance.setdefault(row["instance"], []).append(row)
    summary: list[dict[str, Any]] = []
    for instance, group in sorted(by_instance.items()):
        for metric in _SUMMARY_METRICS:
            if metric in _SOLUTION_METRICS:
                selected = [row for row in group if _bool(row["feasible"])]
            else:
                selected = group
            values = [
                float(row[metric])
                for row in selected
                if row.get(metric) not in ("", None)
            ]
            summary.append(_summarize_metric_values(instance, metric, values))
    return summary


def _summarize_metric_values(
    instance: str, metric: str, values: list[float]
) -> dict[str, Any]:
    if not values:
        return {
            "instance": instance,
            "metric": metric,
            "runs": 0,
            "best": "",
            "mean": "",
            "median": "",
            "worst": "",
            "standard_deviation": "",
        }
    return {
        "instance": instance,
        "metric": metric,
        "runs": len(values),
        "best": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "worst": max(values),
        "standard_deviation": statistics.pstdev(values),
    }


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"expected a boolean value, got {value!r}")
    return normalized == "true"


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _persist_axis_evidence(
    writer: ArtifactBundleWriter,
    *,
    instance: Instance,
    seed: int,
    axes: Mapping[str, ALNSResult],
    scope: str,
    environment_payload: Mapping[str, object],
) -> dict[str, str]:
    """Persist all four axis results for one (instance, seed) pair."""

    if set(axes) != set(DIAGNOSTIC_AXES):
        raise ValueError("axis evidence does not match the declared axes")
    route_dictionary: dict[str, tuple[str, ...]] = {}
    critical_events: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    raw_axes: dict[str, object] = {}
    solution_axes: dict[str, object] = {}
    trace_axes: dict[str, object] = {}
    failures: list[str] = []
    for axis in DIAGNOSTIC_AXES:
        result = axes[axis]
        if result.charging_backend != "cpu_batch":
            raise RuntimeError("Stage 4 evidence may record only cpu_batch")
        trace = result.measurement_trace
        if trace is None:
            raise RuntimeError(f"Stage 4 {axis} result is missing its trace")
        route_dictionary.update(trace.route_dictionary)
        reconciliation = trace.reconcile(result)
        axis_events = build_stage03_critical_events(
            trace,
            result.neighborhood_events,
        )
        for event in axis_events:
            event = _route_reference_event(event, route_dictionary)
            lane = str(event.get("lane", ""))
            critical_events.append(
                {
                    **event,
                    "lane": f"{axis}:{lane}",
                    "diagnostic_axis": axis,
                }
            )
        diagnostic_rows.extend(
            aggregate_diagnostic_events(
                axis_events,
                run_label=writer.context.run_label,
                instance=instance.name,
                seed=seed,
            )
        )
        objective_key = list(result.objective.key) if result.objective is not None else []
        valid = result.feasible and reconciliation["status"] == "pass"
        if not valid:
            failures.append(f"{axis}: solver or trace reconciliation failed")
        raw_axes[axis] = {
            "backend": result.charging_backend,
            "batch_size": result.batch_size,
            "backend_metrics": result.backend_metrics,
            "objective_key": objective_key,
            "runtime_seconds": result.runtime_seconds,
            "iterations": result.iterations,
            "effective_iterations": result.effective_iterations,
            "accepted_moves": result.accepted_moves,
            "improving_moves": result.improving_moves,
            "rejected_moves": result.rejected_moves,
            "started_calls": result.exact_started_calls,
            "completed_calls": result.exact_completed_calls,
            "interrupted_calls": result.exact_interrupted_calls,
            "budget_exhaustions": result.exact_budget_exhaustions,
            "termination_reason": result.termination_reason,
            "trace_reconciliation": reconciliation,
            "valid": valid,
            "stage04_statistics": result.stage04_statistics,
        }
        solution_axes[axis] = {
            "routes": [list(route) for route in result.routes],
            "customer_sequences": [
                list(sequence) for sequence in result.customer_sequences
            ],
            "objective_key": objective_key,
            "feasible": result.feasible,
        }
        trace_payload = trace.to_dict()
        trace_axes[axis] = {
            "trace_schema_version": trace.trace_schema_version,
            "instance_hash": canonical_instance_hash(instance),
            "config": trace_payload["config"],
            "exact_deadline_config": trace_payload.get("exact_deadline_config"),
            "candidate_control_config": trace_payload.get(
                "candidate_control_config"
            ),
            "summary": trace_payload["summary"],
            "result_summary": trace_payload["result_summary"],
        }
    raw_payload = {
        "schema_version": STAGE04_SCHEMA_VERSION,
        "run_label": writer.context.run_label,
        "scope": scope,
        "instance": instance.name,
        "seed": seed,
        "axes": raw_axes,
    }
    failure_payload = (
        {
            "schema_version": STAGE04_SCHEMA_VERSION,
            "instance": instance.name,
            "seed": seed,
            "reasons": failures,
            "evidence_completeness": "complete",
        }
        if failures
        else None
    )
    return writer.write_instance_seed(
        instance=instance.name,
        seed=seed,
        raw_payload=raw_payload,
        solution_payload={
            "schema_version": STAGE04_SCHEMA_VERSION,
            "instance": instance.name,
            "seed": seed,
            "axes": solution_axes,
        },
        trace_payload={
            "trace_schema_version": f"{STAGE04_SCHEMA_VERSION}-trace-index-v1",
            "axes": trace_axes,
        },
        environment_payload=dict(environment_payload),
        route_dictionary=route_dictionary,
        critical_events=critical_events,
        diagnostic_rows=diagnostic_rows,
        failure_payload=failure_payload,
    )


def _route_reference_event(
    event: Mapping[str, object],
    route_dictionary: dict[str, tuple[str, ...]],
) -> dict[str, object]:
    """Replace complete candidate routes with dictionary references."""

    from evrptw.measurement import canonical_route_key

    output = dict(event)
    raw_sequences = output.pop("customer_sequences", None)
    if raw_sequences is None:
        return output
    if not isinstance(raw_sequences, (list, tuple)):
        raise TypeError("candidate customer_sequences must be a sequence")
    route_keys: list[str] = []
    for raw_sequence in raw_sequences:
        if not isinstance(raw_sequence, (list, tuple)) or not all(
            isinstance(node, str) for node in raw_sequence
        ):
            raise TypeError("candidate route must contain node identifiers")
        sequence = tuple(raw_sequence)
        route_key = canonical_route_key(sequence)
        route_dictionary.setdefault(route_key, sequence)
        route_keys.append(route_key)
    output["route_keys"] = route_keys
    return output


def run_stage04_weights(
    *,
    config_path: Path,
    output_dir: Path,
    scope: str,
    run_label: str,
) -> dict[str, Path]:
    """Run the Stage 4 adaptive-weights experiment.

    Parameters
    ----------
    config_path
        Path to ``configs/stage04_weights.toml``.
    output_dir
        Must equal ``results/<run_label>`` relative to the repository root.
    scope
        ``"smoke"`` or ``"formal"``.
    run_label
        Canonical label matching ``stage04_adaptive_weights_(attempt|rerun)NN``.
    """

    root = Path(__file__).resolve().parents[3]
    resolved_config = _resolve(root, config_path)
    resolved_output = _resolve(root, output_dir)
    validate_stage04_run_label(run_label)
    if scope not in {"smoke", "formal"}:
        raise ValueError("Stage 4 scope must be smoke or formal")
    if resolved_output != root / "results" / run_label:
        raise ValueError("Stage 4 output must be results/<canonical-run-label>")
    if resolved_output.exists():
        raise FileExistsError(resolved_output)
    _require_clean_repository(root)
    config = load_stage04_config(resolved_config)
    instances = SMOKE_INSTANCES if scope == "smoke" else FORMAL_INSTANCES
    stage02 = load_stage02_config(_resolve(root, config.stage02_config))
    repository_revision = _git(root, "rev-parse", "HEAD")
    source_sha256 = _source_sha256(root)
    stage00_manifest_sha256 = _sha256(
        root / "experiments/baselines/stage00/manifest.json"
    )
    references = _reference_repositories(root)
    environment = {
        **collect_environment(),
        "repository_revision": repository_revision,
        "repository_dirty": False,
        "exact_backend": "cpu_batch",
        "stage04_schema_version": STAGE04_SCHEMA_VERSION,
        "configuration_sha256": _sha256(resolved_config),
        "source_sha256": source_sha256,
        "stage00_manifest_sha256": stage00_manifest_sha256,
        "reference_repositories": references,
        "scope": scope,
        "diagnostic_axes": list(DIAGNOSTIC_AXES),
        "wall_clock_seconds": config.wall_clock_seconds,
        "max_iterations": config.max_iterations,
        "exact_call_budget": config.exact_call_budget,
        "watchdog_seconds": config.watchdog_seconds,
        "batch_size": config.batch_size,
        "stage04_config": asdict(config.stage04_config),
    }
    writer = ArtifactBundleWriter(
        resolved_output,
        ArtifactRunContext("stage04", "adaptive_weights", run_label),
        config.artifact_storage,
    )
    writer.write_control(
        metadata={
            "schema_version": STAGE04_SCHEMA_VERSION,
            "scope": scope,
            "instances": list(instances),
            "seeds": list(config.seeds),
            "diagnostic_axes": list(DIAGNOSTIC_AXES),
            "repository_revision": repository_revision,
            "repository_dirty": False,
            "source_sha256": source_sha256,
            "stage00_manifest_sha256": stage00_manifest_sha256,
            "reference_repositories": references,
            "configuration_sha256": _sha256(resolved_config),
            "exact_backend": "cpu_batch",
            "wall_clock_seconds": config.wall_clock_seconds,
            "max_iterations": config.max_iterations,
            "exact_call_budget": config.exact_call_budget,
            "watchdog_seconds": config.watchdog_seconds,
            "batch_size": config.batch_size,
            "stage04_config": asdict(config.stage04_config),
            "operator_profile": "stage02_constraint_guided",
        },
        configuration_path=resolved_config,
    )
    per_run_path = resolved_output / "control" / f"{run_label}_per_run_results.csv"
    _write_csv(per_run_path, PER_RUN_FIELDS, [])
    per_run_rows: list[dict[str, Any]] = []
    try:
        for instance_name in instances:
            instance = parse_schneider(
                _resolve(root, config.benchmark_dir) / f"{instance_name}.txt"
            )
            for seed in config.seeds:
                peak_before = _peak_rss_bytes()
                axes: dict[str, ALNSResult] = {}
                for axis in DIAGNOSTIC_AXES:
                    if axis.startswith("fixed_"):
                        stage04_cfg = with_fixed_weights(config.stage04_config)
                    else:
                        stage04_cfg = config.stage04_config
                    axes[axis] = _run_axis(
                        instance,
                        seed=seed,
                        axis=axis,
                        stage04_config=stage04_cfg,
                        vehicle_operator_config=stage02.vehicle_operator_config,
                        screening_config=config.screening_config,
                        cache_incremental_config=config.cache_incremental_config,
                        wall_clock_seconds=config.wall_clock_seconds,
                        watchdog_seconds=config.watchdog_seconds,
                        exact_call_budget=config.exact_call_budget,
                        max_iterations=config.max_iterations,
                        batch_size=config.batch_size,
                    )
                _persist_axis_evidence(
                    writer,
                    instance=instance,
                    seed=seed,
                    axes=axes,
                    scope=scope,
                    environment_payload={
                        **environment,
                        "peak_rss_bytes": max(peak_before, _peak_rss_bytes()),
                    },
                )
                for axis in DIAGNOSTIC_AXES:
                    per_run_rows.append(
                        _per_run_row(instance_name, seed, axis, axes[axis])
                    )
                    _write_csv(per_run_path, PER_RUN_FIELDS, per_run_rows)
    except BaseException:
        writer.finalize(status="partial", evidence_completeness="partial")
        raise
    writer.record_existing_file(
        per_run_path,
        artifact_type="per_run_results",
        retention_class="control",
        storage_format="csv_control",
        row_count=len(per_run_rows),
    )
    summary_path = resolved_output / "control" / f"{run_label}_summary_results.csv"
    summary_rows = _summarize(per_run_rows)
    _write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    writer.record_existing_file(
        summary_path,
        artifact_type="summary_results",
        retention_class="control",
        storage_format="csv_control",
        row_count=len(summary_rows),
    )
    bundle = writer.finalize()
    return {
        "run_dir": bundle.run_dir,
        "per_run_results": per_run_path,
        "summary_results": summary_path,
        "manifest": bundle.manifest_path,
        "manifest_sidecar": bundle.manifest_sidecar_path,
    }


def _resolve(root: Path, path: Path | None) -> Path:
    if path is None:
        raise ValueError("required Stage 4 path is missing")
    return path if path.is_absolute() else root / path


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_clean_repository(root: Path) -> None:
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("Stage 4 runner requires a clean main repository commit")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return value if value > 10_000_000 else value * 1024


def _source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "src/evrptw/alns.py",
        "src/evrptw/stage04.py",
        "src/evrptw/cpu_batch.py",
        "src/evrptw/exact_deadline.py",
        "src/evrptw/measurement.py",
        "src/evrptw/neighborhoods.py",
        "src/evrptw/artifacts.py",
        "src/evrptw/experiments/stage04_weights.py",
    ):
        path = root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage 4 adaptive-weights diagnostics"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage04_weights.toml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--run-label", required=True)
    arguments = parser.parse_args()
    outputs = run_stage04_weights(
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        scope=arguments.scope,
        run_label=arguments.run_label,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
