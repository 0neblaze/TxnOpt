from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evrptw.alns import ALNSResult, solve_alns
from evrptw.environment import collect_environment
from evrptw.experiments.stage00_baseline import load_config as load_stage00_config
from evrptw.experiments.stage02_route_reduction import (
    CONSTRAINT_GUIDED_ALGORITHM,
    FORMAL_INSTANCES,
    FORMAL_SEEDS,
    Stage02Config,
)
from evrptw.experiments.stage02_route_reduction import (
    load_config as load_stage02_config,
)
from evrptw.measurement import MeasurementConfig, Stage03ExecutionError, Stage03Trace
from evrptw.models import Instance
from evrptw.objective import ObjectiveComparison, SolutionObjective, compare_objectives
from evrptw.parser import parse_schneider
from evrptw.validation import validate_routes

SCHEMA_VERSION = "1"
OBJECTIVE_SCHEMA = "vehicles,distance,charging_time,charging_count"
SMOKE_INSTANCES = ("c101C5", "r105C5", "rc105C5", "c101_21", "r101_21", "rc101_21")
SEEDS = FORMAL_SEEDS
RAW_PER_RUN_FIELDS = (
    "schema_version",
    "run_label",
    "experiment_id",
    "scope",
    "instance",
    "seed",
    "algorithm",
    "operator_profile",
    "repository_revision",
    "repository_dirty",
    "algorithm_source_sha256",
    "configuration_sha256",
    "instance_sha256",
    "stage00_manifest_sha256",
    "environment_sha256",
    "reference_vrp_evrp_hub_revision",
    "reference_vrp_evrp_hub_dirty",
    "reference_py_ga_vrptw_revision",
    "reference_py_ga_vrptw_dirty",
    "start_utc",
    "end_utc",
    "time_limit_seconds",
    "max_iterations",
    "threads",
    "objective_schema",
    "objective_key",
    "vehicle_count",
    "total_distance",
    "total_charging_time",
    "charging_count",
    "runtime_seconds",
    "first_feasible_time",
    "best_time",
    "iterations",
    "effective_iterations",
    "accepted_moves",
    "improving_moves",
    "rejected_moves",
    "charging_subproblem_calls",
    "trace_started_calls",
    "trace_completed_calls",
    "trace_exact_calls",
    "trace_cache_hits",
    "trace_precomputed_routes",
    "trace_route_evaluations",
    "trace_deadline_events",
    "trace_reconciliation_status",
    "peak_tracemalloc_bytes",
    "peak_rss_bytes",
    "status",
    "feasible",
    "failure_reason",
    "raw_path",
    "solution_path",
    "trace_path",
    "event_path",
    "environment_path",
)


@dataclass(frozen=True, slots=True)
class Stage03Config:
    schema_version: str
    experiment_id: str
    algorithm: str
    operator_profile: str
    stage00_config: Path
    stage02_config: Path
    baseline_dir: Path
    stage02_attempt16_per_run: Path
    stage02_rerun09_per_run: Path
    benchmark_dir: Path
    seeds: tuple[int, ...]
    time_limit_seconds: float
    max_iterations: int
    threads: int


def load_config(path: Path) -> Stage03Config:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    try:
        stage = payload["stage03"]
        benchmark = payload["benchmark"]
        run = payload["run"]
        config = Stage03Config(
            schema_version=str(stage["schema_version"]),
            experiment_id=str(stage["experiment_id"]),
            algorithm=str(stage["algorithm"]),
            operator_profile=str(stage["operator_profile"]),
            stage00_config=Path(str(stage["stage00_config"])),
            stage02_config=Path(str(stage["stage02_config"])),
            baseline_dir=Path(str(stage["baseline_dir"])),
            stage02_attempt16_per_run=Path(str(stage["stage02_attempt16_per_run"])),
            stage02_rerun09_per_run=Path(str(stage["stage02_rerun09_per_run"])),
            benchmark_dir=Path(str(benchmark["directory"])),
            seeds=tuple(int(value) for value in run["seeds"]),
            time_limit_seconds=float(run["time_limit_seconds"]),
            max_iterations=int(run["max_iterations"]),
            threads=int(run["threads"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid Stage 3.0 configuration: {error}") from error
    _validate_config(config)
    return config


def run_stage03(
    *,
    config_path: Path,
    output_dir: Path,
    scope: str = "smoke",
    run_label: str = "stage03_measurement",
    summary_dir: Path | None = None,
    smoke_review_dir: Path | None = None,
) -> dict[str, Path]:
    """Run only Stage 3.0 measurement; publishing is performed by the auditor."""

    root = _repository_root()
    config_path = _resolve(root, config_path)
    output_dir = _resolve(root, output_dir)
    _assert_results_path(root, output_dir, "output_dir")
    config = load_config(config_path)
    stage02 = load_stage02_config(_resolve(root, config.stage02_config))
    _validate_stage02_protocol(config, stage02)
    instances = _scope_instances(scope)
    if scope == "formal":
        _require_smoke_gate(_resolve(root, smoke_review_dir) if smoke_review_dir else None)
    _validate_run_label(run_label)
    _assert_clean_repository(root)
    if output_dir.exists():
        raise FileExistsError(f"Stage 3.0 output directory already exists: {output_dir}")

    benchmark_dir = _resolve(root, config.benchmark_dir)
    baseline_dir = _resolve(root, config.baseline_dir)
    stage00_config = load_stage00_config(_resolve(root, config.stage00_config))
    if stage00_config.instances != FORMAL_INSTANCES or stage00_config.seeds != FORMAL_SEEDS:
        raise RuntimeError("Stage 0 configuration does not have the canonical 12x3 scope")
    if not (baseline_dir / "manifest.json").is_file():
        raise FileNotFoundError(
            "immutable Stage 0 manifest is missing: "
            f"{baseline_dir / 'manifest.json'}"
        )
    baseline_manifest_sha256 = _sha256(baseline_dir / "manifest.json")
    source_hashes = _source_hashes(root)
    algorithm_source_sha256 = _combined_hash(source_hashes)
    repository_revision = _git_revision(root)
    reference_repositories = _reference_repositories(root)
    base_environment = collect_environment()
    metadata: dict[str, Any] = {
        "schema_version": config.schema_version,
        "experiment_id": config.experiment_id,
        "run_label": run_label,
        "scope": scope,
        "algorithm": config.algorithm,
        "operator_profile": config.operator_profile,
        "expected_instances": list(instances),
        "expected_seeds": list(config.seeds),
        "expected_run_keys": [
            [instance, seed] for instance in instances for seed in config.seeds
        ],
        "time_limit_seconds": config.time_limit_seconds,
        "max_iterations": config.max_iterations,
        "threads": config.threads,
        "repository_revision": repository_revision,
        "repository_dirty": False,
        "algorithm_source_sha256": algorithm_source_sha256,
        "algorithm_source_files": source_hashes,
        "configuration_sha256": _sha256(config_path),
        "stage00_configuration_sha256": _sha256(_resolve(root, config.stage00_config)),
        "stage02_configuration_sha256": _sha256(_resolve(root, config.stage02_config)),
        "stage02_attempt16_per_run_sha256": _sha256(
            _resolve(root, config.stage02_attempt16_per_run)
        ),
        "stage02_rerun09_per_run_sha256": _sha256(
            _resolve(root, config.stage02_rerun09_per_run)
        ),
        "stage00_manifest_sha256": baseline_manifest_sha256,
        "benchmark_directory": str(benchmark_dir),
        "baseline_directory": str(baseline_dir),
        "stage02_attempt16_per_run": str(
            _resolve(root, config.stage02_attempt16_per_run)
        ),
        "stage02_rerun09_per_run": str(_resolve(root, config.stage02_rerun09_per_run)),
        "historical_stage02_baselines": _historical_baseline_provenance(root, config),
        "reference_repositories": reference_repositories,
        "captured_environment": base_environment,
        "summary_dir": str(_resolve(root, summary_dir)) if summary_dir else None,
    }

    output_dir.mkdir(parents=True)
    for name in ("raw", "solutions", "traces", "events", "environments"):
        (output_dir / name).mkdir()
    _write_json(output_dir / "run_metadata.json", metadata)
    shutil.copy2(config_path, output_dir / "parameters.toml")
    raw_per_run_path = output_dir / "raw_per_run_results.csv"
    _write_csv(raw_per_run_path, RAW_PER_RUN_FIELDS, [])

    rows: list[dict[str, Any]] = []
    run_environment_hashes: dict[str, str] = {}
    for instance_name in instances:
        instance_path = benchmark_dir / f"{instance_name}.txt"
        if not instance_path.is_file():
            raise FileNotFoundError(f"benchmark instance is missing: {instance_path}")
        instance_hash = _sha256(instance_path)
        instance = parse_schneider(instance_path)
        for seed in config.seeds:
            run_id = f"{instance_name}-{config.algorithm.lower()}-{seed}"
            start = datetime.now(UTC)
            result: ALNSResult | None = None
            trace: Stage03Trace | None = None
            error: BaseException | None = None
            try:
                result = solve_alns(
                    instance,
                    seed=seed,
                    max_iterations=config.max_iterations,
                    time_limit_seconds=config.time_limit_seconds,
                    operator_profile=config.operator_profile,
                    vehicle_operator_config=stage02.vehicle_operator_config,
                    measurement_config=MeasurementConfig(),
                )
                trace = result.measurement_trace
                if trace is None:
                    raise RuntimeError("Stage 3.0 runner received a result without a trace")
            except BaseException as caught:
                error = caught
                if isinstance(caught, Stage03ExecutionError):
                    trace = caught.trace
                else:
                    trace = Stage03Trace(MeasurementConfig())
                    trace.record_execution_error(caught)
                    trace.finish()
            # Do not enable tracemalloc around the solver: its allocation
            # hooks materially change the fixed wall-clock trajectory on the
            # 100-customer instances.  Peak RSS below remains an OS-level
            # memory record without changing the measured solver path.
            peak_tracemalloc: int | None = None
            peak_rss = _peak_rss_bytes()
            end = datetime.now(UTC)
            if trace is None:
                raise RuntimeError("Stage 3.0 run ended without a trace")
            environment_payload = _run_environment(
                metadata,
                instance_name=instance_name,
                seed=seed,
                instance_hash=instance_hash,
                start=start,
                end=end,
                peak_tracemalloc_bytes=peak_tracemalloc,
                peak_rss_bytes=peak_rss,
            )
            environment_hash = _payload_sha256(environment_payload)
            run_environment_hashes[run_id] = environment_hash
            row = _persist_run(
                output_dir=output_dir,
                run_id=run_id,
                run_label=run_label,
                scope=scope,
                config=config,
                instance=instance,
                instance_hash=instance_hash,
                seed=seed,
                result=result,
                trace=trace,
                error=error,
                start=start,
                end=end,
                repository_revision=repository_revision,
                algorithm_source_sha256=algorithm_source_sha256,
                configuration_sha256=str(metadata["configuration_sha256"]),
                baseline_manifest_sha256=baseline_manifest_sha256,
                environment_payload=environment_payload,
                environment_hash=environment_hash,
                reference_repositories=reference_repositories,
                peak_tracemalloc_bytes=peak_tracemalloc,
                peak_rss_bytes=peak_rss,
            )
            rows.append(row)
            _write_csv(raw_per_run_path, RAW_PER_RUN_FIELDS, rows)
            if error is not None:
                _write_json(output_dir / "environment.json", {
                    **metadata,
                    "run_environment_sha256": run_environment_hashes,
                })
                _write_manifest(output_dir)
                raise error

    _write_json(
        output_dir / "environment.json",
        {**metadata, "run_environment_sha256": run_environment_hashes},
    )
    _write_manifest(output_dir)
    return {
        "output_dir": output_dir,
        "raw_per_run": raw_per_run_path,
        "metadata": output_dir / "run_metadata.json",
        "environment": output_dir / "environment.json",
        "parameters": output_dir / "parameters.toml",
        "manifest": output_dir / "manifest.json",
    }


def _persist_run(
    *,
    output_dir: Path,
    run_id: str,
    run_label: str,
    scope: str,
    config: Stage03Config,
    instance: Instance,
    instance_hash: str,
    seed: int,
    result: ALNSResult | None,
    trace: Stage03Trace,
    error: BaseException | None,
    start: datetime,
    end: datetime,
    repository_revision: str,
    algorithm_source_sha256: str,
    configuration_sha256: str,
    baseline_manifest_sha256: str,
    environment_payload: dict[str, Any],
    environment_hash: str,
    reference_repositories: dict[str, dict[str, object]],
    peak_tracemalloc_bytes: int | None,
    peak_rss_bytes: int | None,
) -> dict[str, Any]:
    raw_path = output_dir / "raw" / f"{run_id}.json"
    solution_path = output_dir / "solutions" / f"{run_id}.json"
    trace_path = output_dir / "traces" / f"{run_id}.json"
    event_path = output_dir / "events" / f"{run_id}.jsonl"
    environment_path = output_dir / "environments" / f"{run_id}.json"
    _write_json(trace_path, trace.to_dict())
    event_lines = [
        {"record_type": "trace_event", "event_index": index, "payload": event}
        for index, event in enumerate(trace.events)
    ]
    if result is not None:
        event_lines.extend(
            {
                "record_type": "neighborhood_event",
                "event_index": index,
                "payload": event,
            }
            for index, event in enumerate(result.neighborhood_events)
        )
    _write_jsonl(event_path, event_lines)
    routes = [list(route) for route in result.routes] if result is not None else []
    objective = result.objective if result is not None else None
    report = validate_routes(instance, routes)
    replay_objective = SolutionObjective.from_report(instance, report) if report.feasible else None
    objective_matches = (
        objective is not None
        and replay_objective is not None
        and compare_objectives(objective, replay_objective) is ObjectiveComparison.EQUAL
    )
    trace_reconciliation = trace.reconcile(result) if result is not None else {
        "status": "not_available",
        "checks": {},
    }
    if error is not None:
        status = "interrupted"
        feasible = False
        failure_reason = f"{type(error).__name__}: {error}"
    elif result is None:
        status = "invalid"
        feasible = False
        failure_reason = "missing solver result"
    elif report.feasible and result.feasible and objective_matches:
        status = "feasible"
        feasible = True
        failure_reason = ""
    else:
        status = "invalid"
        feasible = False
        failure_reason = result.failure_reason or (
            "validator/objective replay mismatch"
            if report.feasible and not objective_matches
            else "solver result is infeasible"
        )
    _write_json(
        solution_path,
        {
            "schema_version": config.schema_version,
            "instance": instance.name,
            "seed": seed,
            "routes": routes,
            "objective_key": list(objective.key) if objective is not None else [],
            "solver_feasible": result.feasible if result is not None else False,
        },
    )
    _write_json(environment_path, environment_payload)
    solver_result = asdict(result) if result is not None else None
    if solver_result is not None:
        solver_result["measurement_trace"] = None
    row: dict[str, Any] = {
        "schema_version": config.schema_version,
        "run_label": run_label,
        "experiment_id": run_id,
        "scope": scope,
        "instance": instance.name,
        "seed": seed,
        "algorithm": config.algorithm,
        "operator_profile": config.operator_profile,
        "repository_revision": repository_revision,
        "repository_dirty": False,
        "algorithm_source_sha256": algorithm_source_sha256,
        "configuration_sha256": configuration_sha256,
        "instance_sha256": instance_hash,
        "stage00_manifest_sha256": baseline_manifest_sha256,
        "environment_sha256": environment_hash,
        "reference_vrp_evrp_hub_revision": reference_repositories[
            "VRP-EVRP-Project-Hub"
        ]["revision"],
        "reference_vrp_evrp_hub_dirty": reference_repositories[
            "VRP-EVRP-Project-Hub"
        ]["dirty"],
        "reference_py_ga_vrptw_revision": reference_repositories["py-ga-VRPTW"][
            "revision"
        ],
        "reference_py_ga_vrptw_dirty": reference_repositories["py-ga-VRPTW"]["dirty"],
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "time_limit_seconds": config.time_limit_seconds,
        "max_iterations": config.max_iterations,
        "threads": config.threads,
        "objective_schema": OBJECTIVE_SCHEMA,
        "objective_key": json.dumps(objective.key if objective is not None else ()),
        "vehicle_count": replay_objective.vehicle_count if replay_objective else "",
        "total_distance": replay_objective.total_distance if replay_objective else "",
        "total_charging_time": replay_objective.total_charging_time if replay_objective else "",
        "charging_count": replay_objective.charging_count if replay_objective else "",
        "runtime_seconds": result.runtime_seconds if result is not None else "",
        "first_feasible_time": result.first_feasible_time if result is not None else "",
        "best_time": result.best_time if result is not None else "",
        "iterations": result.iterations if result is not None else "",
        "effective_iterations": result.effective_iterations if result is not None else "",
        "accepted_moves": result.accepted_moves if result is not None else "",
        "improving_moves": result.improving_moves if result is not None else "",
        "rejected_moves": result.rejected_moves if result is not None else "",
        "charging_subproblem_calls": (
            result.charging_subproblem_calls if result is not None else ""
        ),
        "trace_started_calls": trace.started_calls,
        "trace_completed_calls": trace.completed_calls,
        "trace_exact_calls": trace.exact_calls,
        "trace_cache_hits": trace.cache_hits,
        "trace_precomputed_routes": trace.precomputed_routes,
        "trace_route_evaluations": len(trace.route_evaluations),
        "trace_deadline_events": trace.deadline_events,
        "trace_reconciliation_status": trace_reconciliation["status"],
        "peak_tracemalloc_bytes": peak_tracemalloc_bytes,
        "peak_rss_bytes": peak_rss_bytes if peak_rss_bytes is not None else "",
        "status": status,
        "feasible": feasible,
        "failure_reason": failure_reason,
        "raw_path": str(raw_path.relative_to(output_dir)),
        "solution_path": str(solution_path.relative_to(output_dir)),
        "trace_path": str(trace_path.relative_to(output_dir)),
        "event_path": str(event_path.relative_to(output_dir)),
        "environment_path": str(environment_path.relative_to(output_dir)),
    }
    _write_json(
        raw_path,
        {
            "record": row,
            "solver_result": solver_result,
            "validator_replay": {
                "feasible": report.feasible,
                "objective_key": list(replay_objective.key) if replay_objective else [],
                "violations": [*report.violations],
            },
            "trace_path": row["trace_path"],
            "event_path": row["event_path"],
        },
    )
    return row


def _run_environment(
    metadata: dict[str, Any],
    *,
    instance_name: str,
    seed: int,
    instance_hash: str,
    start: datetime,
    end: datetime,
    peak_tracemalloc_bytes: int | None,
    peak_rss_bytes: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": metadata["schema_version"],
        "run_label": metadata["run_label"],
        "scope": metadata["scope"],
        "instance": instance_name,
        "seed": seed,
        "instance_sha256": instance_hash,
        "repository_revision": metadata["repository_revision"],
        "repository_dirty": metadata["repository_dirty"],
        "algorithm_source_sha256": metadata["algorithm_source_sha256"],
        "configuration_sha256": metadata["configuration_sha256"],
        "stage00_manifest_sha256": metadata["stage00_manifest_sha256"],
        "reference_repositories": metadata["reference_repositories"],
        "captured_environment": metadata["captured_environment"],
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "run_start_utc": start.isoformat(),
        "run_end_utc": end.isoformat(),
        "peak_tracemalloc_bytes": peak_tracemalloc_bytes,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_scope": "process_lifetime",
    }


def _validate_config(config: Stage03Config) -> None:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported Stage 3.0 schema version: {config.schema_version}")
    if config.algorithm != CONSTRAINT_GUIDED_ALGORITHM:
        raise ValueError(f"Stage 3.0 only supports {CONSTRAINT_GUIDED_ALGORITHM}")
    if config.operator_profile != "stage02_constraint_guided":
        raise ValueError("Stage 3.0 requires the stage02_constraint_guided profile")
    if config.seeds != SEEDS:
        raise ValueError(f"Stage 3.0 seeds must be exactly {SEEDS}")
    if config.time_limit_seconds != 30.0 or config.max_iterations != 1000:
        raise ValueError("Stage 3.0 must use 30 seconds and 1000 iterations")
    if config.threads != 1:
        raise ValueError("Stage 3.0 requires exactly one thread")


def _validate_stage02_protocol(config: Stage03Config, stage02: Stage02Config) -> None:
    if (
        stage02.algorithm != config.algorithm
        or stage02.operator_profile.value != config.operator_profile
    ):
        raise ValueError("Stage 3.0 protocol differs from configs/stage02_constraint_guided.toml")
    if stage02.instances != FORMAL_INSTANCES or stage02.seeds != SEEDS:
        raise ValueError("Stage 2.3 configuration does not have the canonical formal scope")
    if stage02.time_limit_seconds != config.time_limit_seconds:
        raise ValueError("Stage 3.0 time limit differs from Stage 2.3")
    if stage02.max_iterations != config.max_iterations or stage02.threads != config.threads:
        raise ValueError("Stage 3.0 iteration/thread protocol differs from Stage 2.3")
    if stage02.vehicle_operator_config.constraint_lane_time_budget_seconds != 0.1:
        raise ValueError("Stage 3.0 requires the fixed 0.1-second constraint-lane slice")
    vehicle_config = stage02.vehicle_operator_config
    if (
        vehicle_config.quality_probe_exact_evaluation_budget != 2
        or vehicle_config.quality_route_segment_probe_exact_evaluation_budget != 4
        or vehicle_config.vehicle_reduction_refinement_exact_evaluation_budget != 512
    ):
        raise ValueError(
            "Stage 3.0 requires the accepted Stage 2.3 probe/refinement budgets "
            "(2, 4, and 512)"
        )


def _scope_instances(scope: str) -> tuple[str, ...]:
    if scope == "smoke":
        return SMOKE_INSTANCES
    if scope == "formal":
        return FORMAL_INSTANCES
    raise ValueError("scope must be either smoke or formal")


def _require_smoke_gate(review_dir: Path | None) -> None:
    if review_dir is None or not review_dir.is_dir():
        raise RuntimeError(
            "formal Stage 3.0 is blocked: provide a smoke review directory with "
            "READY_FOR_STAGE03_FORMAL_MEASUREMENT"
        )
    # Re-run the auditor from the raw run directory.  A status string in a
    # mutable review CSV/JSON is not an authorization token for formal work.
    from evrptw.experiments.stage03_measurement_review import review_run

    outputs = review_run(run_dir=review_dir.parent)
    manifest = outputs["review_manifest"]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("scope") == "smoke" and payload.get("status") == (
        "READY_FOR_STAGE03_FORMAL_MEASUREMENT"
    ):
        return
    raise RuntimeError(
        "formal Stage 3.0 is blocked until smoke replay reports "
        "READY_FOR_STAGE03_FORMAL_MEASUREMENT"
    )


def _assert_clean_repository(root: Path) -> None:
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "Stage 3.0 runs require a clean main repository commit; "
            f"observed git status: {status}"
        )


def _assert_results_path(root: Path, path: Path, label: str) -> None:
    results_root = (root / "results").resolve()
    candidate = path.resolve()
    try:
        relative = candidate.relative_to(results_root)
    except ValueError as error:
        raise ValueError(
            f"Stage 3.0 {label} must be a child of the ignored results/ directory: "
            f"{candidate}"
        ) from error
    if not relative.parts:
        raise ValueError(f"Stage 3.0 {label} must be a new run directory below results/")


def _source_hashes(root: Path) -> dict[str, str]:
    paths = (
        Path("src/evrptw/alns.py"),
        Path("src/evrptw/measurement.py"),
        Path("src/evrptw/neighborhoods.py"),
        Path("src/evrptw/objective.py"),
        Path("src/evrptw/charging.py"),
        Path("src/evrptw/environment.py"),
        Path("src/evrptw/validation.py"),
        Path("src/evrptw/experiments/stage03_measurement.py"),
        Path("src/evrptw/experiments/stage03_measurement_review.py"),
        Path("configs/stage00_baseline.toml"),
        Path("configs/stage02_constraint_guided.toml"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    )
    missing = [path for path in paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 3.0 source files are missing: {missing}")
    return {str(path): _sha256(root / path) for path in paths}


def _reference_repositories(root: Path) -> dict[str, dict[str, object]]:
    paths = {
        "VRP-EVRP-Project-Hub": root / "reference" / "VRP-EVRP-Project-Hub",
        "py-ga-VRPTW": root / "reference" / "py-ga-VRPTW",
    }
    return {name: _reference_state(path) for name, path in paths.items()}


def _historical_baseline_provenance(
    root: Path,
    config: Stage03Config,
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for label, path in (
        ("attempt16", _resolve(root, config.stage02_attempt16_per_run)),
        ("rerun09", _resolve(root, config.stage02_rerun09_per_run)),
    ):
        environment_name = path.name.replace(
            "_per_run_results.csv", "_environment.json"
        )
        environment_path = path.with_name(environment_name)
        if not environment_path.is_file():
            output[label] = {
                "environment_path": str(environment_path),
                "repository_revision": None,
                "repository_dirty": None,
            }
            continue
        payload = json.loads(environment_path.read_text(encoding="utf-8"))
        output[label] = {
            "environment_path": str(environment_path),
            "repository_revision": payload.get("repository_revision"),
            "repository_dirty": payload.get("repository_dirty"),
        }
    return output


def _reference_state(path: Path) -> dict[str, object]:
    if not (path / ".git").exists():
        return {"path": str(path), "revision": None, "dirty": None}
    return {
        "path": str(path),
        "revision": _git_revision(path),
        "dirty": _git_dirty(path),
    }


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _git_dirty(root: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )


def _peak_rss_bytes() -> int | None:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    return int(usage if platform.system() == "Darwin" else usage * 1024)


def _write_manifest(directory: Path) -> None:
    manifest_path = directory / "manifest.json"
    sidecar_path = directory / "manifest.sha256"
    files = {
        str(path.relative_to(directory)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path not in {manifest_path, sidecar_path}
    }
    _write_json(manifest_path, {"schema_version": SCHEMA_VERSION, "files": files})
    sidecar_path.write_text(
        _sha256(manifest_path) + "\n",
        encoding="utf-8",
    )


def _validate_run_label(run_label: str) -> None:
    if not run_label or run_label in {".", ".."} or "/" in run_label or "\\" in run_label:
        raise ValueError("run_label must be a non-empty single path segment")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, path: Path | None) -> Path:
    if path is None:
        raise ValueError("path is required")
    return path if path.is_absolute() else root / path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _combined_hash(values: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path, value in sorted(values.items()):
        digest.update(path.encode())
        digest.update(value.encode())
    return digest.hexdigest()


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 3.0 measurement evidence")
    parser.add_argument("--config", type=Path, default=Path("configs/stage03_measurement.toml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--run-label", default="stage03_measurement")
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--smoke-review-dir", type=Path)
    arguments = parser.parse_args()
    outputs = run_stage03(
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        scope=arguments.scope,
        run_label=arguments.run_label,
        summary_dir=arguments.summary_dir,
        smoke_review_dir=arguments.smoke_review_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
