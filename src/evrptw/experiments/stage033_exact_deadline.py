"""Stage 3.3 paired exact-call-budget and wall-clock diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import subprocess
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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
from evrptw.parser import parse_schneider

STAGE033_SCHEMA_VERSION = "stage033-exact-deadline-v1"
STAGE033_RUN_LABEL = re.compile(
    r"stage03\.3_exact_deadline_(?:attempt|rerun)[0-9]{2}"
)
DIAGNOSTIC_AXES = ("fixed_exact_calls", "wall_clock")


@dataclass(frozen=True, slots=True)
class Stage033Config:
    benchmark_dir: Path
    stage02_config: Path
    stage032_review_manifest: Path
    seeds: tuple[int, ...]
    wall_clock_seconds: float
    max_iterations: int
    threads: int
    exact_backend: str
    exact_call_budget: int
    watchdog_seconds: float
    batch_size: int
    screening_config: CheapScreeningConfig
    cache_incremental_config: CacheIncrementalConfig
    artifact_storage: ArtifactStorageConfig


def load_stage033_config(path: Path) -> Stage033Config:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    try:
        stage = payload["stage03"]
        run = payload["run"]
        exact = payload["exact_deadline"]
        config = Stage033Config(
            benchmark_dir=Path(str(payload["benchmark"]["directory"])),
            stage02_config=Path(str(stage["stage02_config"])),
            stage032_review_manifest=Path(str(stage["stage032_review_manifest"])),
            seeds=tuple(int(value) for value in run["seeds"]),
            wall_clock_seconds=float(run["time_limit_seconds"]),
            max_iterations=int(run["max_iterations"]),
            threads=int(run["threads"]),
            exact_backend=str(exact["exact_backend"]),
            exact_call_budget=int(exact["exact_call_budget"]),
            watchdog_seconds=float(exact["watchdog_seconds"]),
            batch_size=int(exact["batch_size"]),
            screening_config=CheapScreeningConfig(**dict(payload["screening"])),
            cache_incremental_config=CacheIncrementalConfig(
                **dict(payload["cache_incremental"])
            ),
            artifact_storage=ArtifactStorageConfig(**dict(payload["artifact_storage"])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid Stage 3.3 configuration: {error}") from error
    if config.seeds != tuple(FORMAL_SEEDS):
        raise ValueError("Stage 3.3 seeds must match the Stage 0 formal seeds")
    if config.wall_clock_seconds != 30.0 or config.max_iterations != 1000:
        raise ValueError("Stage 3.3 requires 30 seconds and 1000 iterations")
    if config.threads != 1:
        raise ValueError("Stage 3.3 requires one thread")
    if config.exact_backend != "cpu_batch":
        raise ValueError("Stage 3.3 requires exact_backend=cpu_batch")
    if (
        config.exact_call_budget != 100
        or config.watchdog_seconds != 120.0
        or config.batch_size <= 0
    ):
        raise ValueError(
            "Stage 3.3 requires 100 calls, 120-second watchdog, and positive batch size"
        )
    return config


def validate_stage033_run_label(run_label: str) -> None:
    if STAGE033_RUN_LABEL.fullmatch(run_label) is None:
        raise ValueError(
            "Stage 3.3 run label must be canonical: "
            "stage03.3_exact_deadline_attemptNN or rerunNN"
        )


def run_paired_diagnostic(
    instance: Instance,
    *,
    seed: int,
    vehicle_operator_config: VehicleOperatorConfig | None,
    screening_config: CheapScreeningConfig,
    cache_incremental_config: CacheIncrementalConfig,
    exact_call_budget: int,
    wall_clock_seconds: float,
    watchdog_seconds: float,
    max_iterations: int,
    batch_size: int,
) -> Mapping[str, ALNSResult]:
    """Run the two Stage 3.3 axes with identical search configuration."""

    fixed = solve_alns(
        instance,
        seed=seed,
        max_iterations=max_iterations,
        time_limit_seconds=watchdog_seconds,
        operator_profile="stage02_constraint_guided",
        vehicle_operator_config=vehicle_operator_config,
        measurement_config=MeasurementConfig(),
        screening_config=screening_config,
        cache_incremental_config=cache_incremental_config,
        backend="cpu_batch",
        batch_size=batch_size,
        exact_deadline_config=ExactDeadlineConfig.fixed_exact_calls(
            exact_call_budget,
            watchdog_seconds=watchdog_seconds,
        ),
    )
    wall_clock = solve_alns(
        instance,
        seed=seed,
        max_iterations=max_iterations,
        time_limit_seconds=wall_clock_seconds,
        operator_profile="stage02_constraint_guided",
        vehicle_operator_config=vehicle_operator_config,
        measurement_config=MeasurementConfig(),
        screening_config=screening_config,
        cache_incremental_config=cache_incremental_config,
        backend="cpu_batch",
        batch_size=batch_size,
        exact_deadline_config=ExactDeadlineConfig.wall_clock(),
    )
    return {"fixed_exact_calls": fixed, "wall_clock": wall_clock}


def persist_paired_diagnostic(
    writer: ArtifactBundleWriter,
    *,
    instance: Instance,
    seed: int,
    pair: Mapping[str, ALNSResult],
    scope: str,
    environment_payload: Mapping[str, object],
) -> dict[str, str]:
    """Persist one paired diagnostic through the shared artifact writer."""

    if set(pair) != set(DIAGNOSTIC_AXES):
        raise ValueError("Stage 3.3 evidence requires both diagnostic axes")
    route_dictionary: dict[str, tuple[str, ...]] = {}
    critical_events: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    raw_axes: dict[str, object] = {}
    solution_axes: dict[str, object] = {}
    trace_axes: dict[str, object] = {}
    failures: list[str] = []
    for axis in DIAGNOSTIC_AXES:
        result = pair[axis]
        if result.charging_backend != "cpu_batch":
            raise RuntimeError("Stage 3.3 evidence may record only cpu_batch")
        trace = result.measurement_trace
        if trace is None:
            raise RuntimeError(f"Stage 3.3 {axis} result is missing its trace")
        route_dictionary.update(trace.route_dictionary)
        reconciliation = trace.reconcile(result)
        axis_events = build_stage03_critical_events(
            trace,
            result.neighborhood_events,
        )
        for event in axis_events:
            lane = str(event.get("lane", ""))
            critical_events.append(
                {
                    **event,
                    "lane": f"{axis}:{lane}",
                    "diagnostic_axis": axis,
                }
            )
        diagnostic_rows.extend(
            {
                "lane": axis,
                "operator": "cpu_batch",
                "reason": "backend_metric",
                "metric": metric,
                "count": int(value) if isinstance(value, int) else None,
                "sum_value": float(value) if isinstance(value, float) else None,
            }
            for metric, value in result.backend_metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
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
            "exact_deadline_statistics": result.exact_deadline_statistics,
            "trace_reconciliation": reconciliation,
            "valid": valid,
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
            "summary": trace_payload["summary"],
            "result_summary": trace_payload["result_summary"],
        }
    raw_payload = {
        "schema_version": STAGE033_SCHEMA_VERSION,
        "run_label": writer.context.run_label,
        "scope": scope,
        "instance": instance.name,
        "seed": seed,
        "peak_rss_bytes": environment_payload.get("peak_rss_bytes"),
        "axes": raw_axes,
    }
    failure_payload = (
        {
            "schema_version": STAGE033_SCHEMA_VERSION,
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
            "schema_version": STAGE033_SCHEMA_VERSION,
            "instance": instance.name,
            "seed": seed,
            "axes": solution_axes,
        },
        trace_payload={
            "trace_schema_version": "stage033-paired-trace-v1",
            "axes": trace_axes,
        },
        environment_payload=dict(environment_payload),
        route_dictionary=route_dictionary,
        critical_events=critical_events,
        diagnostic_rows=diagnostic_rows,
        failure_payload=failure_payload,
    )


def run_stage033(
    *,
    config_path: Path,
    output_dir: Path,
    scope: str,
    run_label: str,
    smoke_review_dir: Path | None = None,
) -> dict[str, Path]:
    """Generate immutable paired Stage 3.3 evidence."""

    root = Path(__file__).resolve().parents[3]
    resolved_config = _resolve(root, config_path)
    resolved_output = _resolve(root, output_dir)
    validate_stage033_run_label(run_label)
    if scope not in {"smoke", "formal"}:
        raise ValueError("Stage 3.3 scope must be smoke or formal")
    if resolved_output != root / "results" / run_label:
        raise ValueError("Stage 3.3 output must be results/<canonical-run-label>")
    if resolved_output.exists():
        raise FileExistsError(resolved_output)
    _require_clean_repository(root)
    config = load_stage033_config(resolved_config)
    _require_stage032_ready(_resolve(root, config.stage032_review_manifest))
    if scope == "formal":
        _require_smoke_ready(
            _resolve(root, smoke_review_dir) if smoke_review_dir else None,
            benchmark_dir=_resolve(root, config.benchmark_dir),
        )
    instances = SMOKE_INSTANCES if scope == "smoke" else FORMAL_INSTANCES
    stage02 = load_stage02_config(_resolve(root, config.stage02_config))
    repository_revision = _git(root, "rev-parse", "HEAD")
    source_sha256 = _source_sha256(root)
    stage00_manifest_sha256 = _sha256(
        root / "experiments/baselines/stage00/manifest.json"
    )
    reference_repositories = _reference_repositories(root)
    environment = {
        **collect_environment(),
        "repository_revision": repository_revision,
        "repository_dirty": False,
        "exact_backend": "cpu_batch",
        "stage033_schema_version": STAGE033_SCHEMA_VERSION,
        "configuration_sha256": _sha256(resolved_config),
        "source_sha256": source_sha256,
        "stage00_manifest_sha256": stage00_manifest_sha256,
        "reference_repositories": reference_repositories,
        "scope": scope,
    }
    writer = ArtifactBundleWriter(
        resolved_output,
        ArtifactRunContext("stage03.3", "exact_deadline", run_label),
        config.artifact_storage,
    )
    writer.write_control(
        metadata={
            "schema_version": STAGE033_SCHEMA_VERSION,
            "scope": scope,
            "instances": list(instances),
            "seeds": list(config.seeds),
            "diagnostic_axes": list(DIAGNOSTIC_AXES),
            "repository_revision": repository_revision,
            "repository_dirty": False,
            "source_sha256": source_sha256,
            "stage00_manifest_sha256": stage00_manifest_sha256,
            "reference_repositories": reference_repositories,
            "configuration_sha256": _sha256(resolved_config),
            "exact_backend": "cpu_batch",
            "exact_call_budget": config.exact_call_budget,
            "wall_clock_seconds": config.wall_clock_seconds,
            "watchdog_seconds": config.watchdog_seconds,
            "max_iterations": config.max_iterations,
            "threads": config.threads,
        },
        configuration_path=resolved_config,
    )
    try:
        for instance_name in instances:
            instance = parse_schneider(
                _resolve(root, config.benchmark_dir) / f"{instance_name}.txt"
            )
            for seed in config.seeds:
                peak_rss_before = _peak_rss_bytes()
                pair = run_paired_diagnostic(
                    instance,
                    seed=seed,
                    vehicle_operator_config=stage02.vehicle_operator_config,
                    screening_config=config.screening_config,
                    cache_incremental_config=config.cache_incremental_config,
                    exact_call_budget=config.exact_call_budget,
                    wall_clock_seconds=config.wall_clock_seconds,
                    watchdog_seconds=config.watchdog_seconds,
                    max_iterations=config.max_iterations,
                    batch_size=config.batch_size,
                )
                persist_paired_diagnostic(
                    writer,
                    instance=instance,
                    seed=seed,
                    pair=pair,
                    scope=scope,
                    environment_payload={
                        **environment,
                        "peak_rss_bytes": max(peak_rss_before, _peak_rss_bytes()),
                    },
                )
    except BaseException:
        writer.finalize(status="partial", evidence_completeness="partial")
        raise
    bundle = writer.finalize()
    return {
        "run_dir": bundle.run_dir,
        "manifest": bundle.manifest_path,
        "manifest_sidecar": bundle.manifest_sidecar_path,
    }


def _resolve(root: Path, path: Path | None) -> Path:
    if path is None:
        raise ValueError("required Stage 3.3 path is missing")
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
        raise RuntimeError("Stage 3.3 runner requires a clean main repository commit")


def _require_stage032_ready(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "READY_FOR_STAGE03_3":
        raise RuntimeError("Stage 3.2 formal review is not READY_FOR_STAGE03_3")


def _require_smoke_ready(path: Path | None, *, benchmark_dir: Path) -> None:
    if path is None:
        raise RuntimeError("formal Stage 3.3 requires the smoke review directory")
    manifest = path / "review_manifest.json" if path.is_dir() else path
    from evrptw.experiments.stage033_exact_deadline_review import review_stage033

    review_dir = manifest.parent
    outputs = review_stage033(
        run_dir=review_dir.parent,
        scope="smoke",
        benchmark_dir=benchmark_dir,
        output_dir=review_dir,
    )
    payload = json.loads(outputs["review_manifest"].read_text(encoding="utf-8"))
    if payload.get("status") != "READY_FOR_STAGE033_FORMAL":
        raise RuntimeError("Stage 3.3 smoke review is not ready for formal diagnostics")


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
        "src/evrptw/cpu_batch.py",
        "src/evrptw/exact_deadline.py",
        "src/evrptw/measurement.py",
        "src/evrptw/artifacts.py",
        "src/evrptw/experiments/stage033_exact_deadline.py",
        "src/evrptw/experiments/stage033_exact_deadline_review.py",
    ):
        path = root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 3.3 exact-deadline diagnostics")
    parser.add_argument("--config", type=Path, default=Path("configs/stage033_exact_deadline.toml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--smoke-review-dir", type=Path)
    arguments = parser.parse_args()
    outputs = run_stage033(
        config_path=arguments.config,
        output_dir=arguments.output_dir,
        scope=arguments.scope,
        run_label=arguments.run_label,
        smoke_review_dir=arguments.smoke_review_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
