"""Stage 3.4 deterministic candidate-control and parallel diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from evrptw.alns import ALNSResult, solve_alns
from evrptw.artifacts import (
    ArtifactBundleWriter,
    ArtifactRunContext,
    ArtifactStorageConfig,
    verify_manifest,
)
from evrptw.cache_incremental import CacheIncrementalConfig
from evrptw.candidate_control import CandidateControlConfig
from evrptw.environment import collect_environment
from evrptw.exact_deadline import ExactDeadlineConfig
from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES, FORMAL_SEEDS
from evrptw.experiments.stage02_route_reduction import load_config as load_stage02_config
from evrptw.experiments.stage03_measurement import SMOKE_INSTANCES, _reference_repositories
from evrptw.experiments.stage033_exact_deadline import (
    _git,
    _peak_rss_bytes,
    _require_clean_repository,
    _resolve,
    _sha256,
    persist_paired_diagnostic,
)
from evrptw.measurement import CheapScreeningConfig, MeasurementConfig
from evrptw.models import Instance, NodeType
from evrptw.neighborhoods import VehicleOperatorConfig
from evrptw.parser import parse_schneider

STAGE034_SCHEMA_VERSION = "stage034-control-parallel-v1"
STAGE034_RUN_LABEL = re.compile(r"stage03\.4_control_parallel_(?:attempt|rerun)[0-9]{2}")
DIAGNOSTIC_AXES = (
    "serial_fixed_exact_calls",
    "parallel_fixed_exact_calls",
    "serial_wall_clock",
    "parallel_wall_clock",
)


@dataclass(frozen=True, slots=True)
class Stage034Config:
    benchmark_dir: Path
    stage02_config: Path
    stage033_review_manifest: Path
    seeds: tuple[int, ...]
    wall_clock_seconds: float
    max_iterations: int
    exact_call_budget: int
    watchdog_seconds: float
    batch_size: int
    candidate_control_config: CandidateControlConfig
    inherit_stage033_incumbent: bool
    proposal_top_k_grid: tuple[int, ...]
    round_budget_grid: tuple[int, ...]
    selection_order: tuple[str, ...]
    screening_config: CheapScreeningConfig
    cache_incremental_config: CacheIncrementalConfig
    artifact_storage: ArtifactStorageConfig


def load_stage034_config(path: Path) -> Stage034Config:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    try:
        stage = payload["stage03"]
        run = payload["run"]
        exact = payload["exact_deadline"]
        control = dict(payload["candidate_control"])
        grid = payload["candidate_control_grid"]
        worker_counts = tuple(int(value) for value in control.pop("worker_counts"))
        inherit_stage033_incumbent = bool(control.pop("inherit_stage033_incumbent"))
        if worker_counts != (1, 4):
            raise ValueError("worker_counts must be [1, 4]")
        config = Stage034Config(
            benchmark_dir=Path(str(payload["benchmark"]["directory"])),
            stage02_config=Path(str(stage["stage02_config"])),
            stage033_review_manifest=Path(str(stage["stage033_review_manifest"])),
            seeds=tuple(int(value) for value in run["seeds"]),
            wall_clock_seconds=float(run["time_limit_seconds"]),
            max_iterations=int(run["max_iterations"]),
            exact_call_budget=int(exact["exact_call_budget"]),
            watchdog_seconds=float(exact["watchdog_seconds"]),
            batch_size=int(exact["batch_size"]),
            candidate_control_config=CandidateControlConfig(
                worker_count=1,
                **control,
            ),
            inherit_stage033_incumbent=inherit_stage033_incumbent,
            proposal_top_k_grid=tuple(int(value) for value in grid["proposal_top_k"]),
            round_budget_grid=tuple(int(value) for value in grid["max_exact_calls_per_round"]),
            selection_order=tuple(str(value) for value in grid["selection_order"]),
            screening_config=CheapScreeningConfig(**dict(payload["screening"])),
            cache_incremental_config=CacheIncrementalConfig(**dict(payload["cache_incremental"])),
            artifact_storage=ArtifactStorageConfig(**dict(payload["artifact_storage"])),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid Stage 3.4 configuration: {error}") from error
    if config.seeds != tuple(FORMAL_SEEDS):
        raise ValueError("Stage 3.4 seeds must match Stage 0")
    if config.wall_clock_seconds != 30.0 or config.max_iterations != 1000:
        raise ValueError("Stage 3.4 requires 30 seconds and 1000 iterations")
    if config.exact_call_budget != 100 or config.watchdog_seconds != 120.0:
        raise ValueError("Stage 3.4 requires 100 calls and a 120-second watchdog")
    if config.batch_size <= 0:
        raise ValueError("Stage 3.4 batch_size must be positive")
    if not config.inherit_stage033_incumbent:
        raise ValueError(
            "Stage 3.4 formal evidence requires the audited Stage 3.3 incumbent warm start"
        )
    if config.proposal_top_k_grid != (1, 2, 4) or config.round_budget_grid != (1, 2, 4):
        raise ValueError("Stage 3.4 candidate-control grid must be {1,2,4} x {1,2,4}")
    if (
        config.candidate_control_config.proposal_top_k not in config.proposal_top_k_grid
        or config.candidate_control_config.max_exact_calls_per_round not in config.round_budget_grid
    ):
        raise ValueError("selected Stage 3.4 configuration is outside the preregistered grid")
    return config


def validate_stage034_run_label(run_label: str) -> None:
    if STAGE034_RUN_LABEL.fullmatch(run_label) is None:
        raise ValueError(
            "Stage 3.4 run label must be stage03.4_control_parallel_attemptNN or rerunNN"
        )


def run_control_parallel_diagnostic(
    instance: Instance,
    *,
    seed: int,
    vehicle_operator_config: VehicleOperatorConfig | None,
    screening_config: CheapScreeningConfig,
    cache_incremental_config: CacheIncrementalConfig,
    candidate_control_config: CandidateControlConfig,
    exact_call_budget: int,
    wall_clock_seconds: float,
    watchdog_seconds: float,
    max_iterations: int,
    batch_size: int,
    initial_customer_sequences: tuple[tuple[str, ...], ...] | None = None,
    initial_solution_provenance: Mapping[str, object] | None = None,
) -> Mapping[str, ALNSResult]:
    output: dict[str, ALNSResult] = {}
    for worker_label, worker_count in (("serial", 1), ("parallel", 4)):
        control = replace(candidate_control_config, worker_count=worker_count)
        output[f"{worker_label}_fixed_exact_calls"] = solve_alns(
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
            candidate_control_config=control,
            initial_customer_sequences=initial_customer_sequences,
            initial_solution_provenance=initial_solution_provenance,
        )
        output[f"{worker_label}_wall_clock"] = solve_alns(
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
            candidate_control_config=control,
            initial_customer_sequences=initial_customer_sequences,
            initial_solution_provenance=initial_solution_provenance,
        )
    return output


def run_stage034(
    *,
    config_path: Path,
    output_dir: Path,
    scope: str,
    run_label: str,
    smoke_review_dir: Path | None = None,
) -> dict[str, Path]:
    root = Path(__file__).resolve().parents[3]
    resolved_config = _resolve(root, config_path)
    resolved_output = _resolve(root, output_dir)
    validate_stage034_run_label(run_label)
    if scope not in {"smoke", "formal"}:
        raise ValueError("Stage 3.4 scope must be smoke or formal")
    if resolved_output != root / "results" / run_label:
        raise ValueError("Stage 3.4 output must be results/<canonical-run-label>")
    if resolved_output.exists():
        raise FileExistsError(resolved_output)
    _require_clean_repository(root)
    config = load_stage034_config(resolved_config)
    stage033_review_manifest = _resolve(root, config.stage033_review_manifest)
    _require_stage033_ready(stage033_review_manifest)
    if scope == "formal":
        _require_stage034_smoke_ready(
            _resolve(root, smoke_review_dir) if smoke_review_dir else None,
            benchmark_dir=_resolve(root, config.benchmark_dir),
        )
    instances = SMOKE_INSTANCES if scope == "smoke" else FORMAL_INSTANCES
    stage02 = load_stage02_config(_resolve(root, config.stage02_config))
    repository_revision = _git(root, "rev-parse", "HEAD")
    source_sha256 = _source_sha256(root)
    stage00_manifest_sha256 = _sha256(root / "experiments/baselines/stage00/manifest.json")
    references = _reference_repositories(root)
    environment = {
        **collect_environment(),
        "repository_revision": repository_revision,
        "repository_dirty": False,
        "exact_backend": "cpu_batch",
        "stage034_schema_version": STAGE034_SCHEMA_VERSION,
        "configuration_sha256": _sha256(resolved_config),
        "source_sha256": source_sha256,
        "stage00_manifest_sha256": stage00_manifest_sha256,
        "reference_repositories": references,
        "scope": scope,
        "worker_counts": [1, 4],
        "executor_model": "process_spawn",
    }
    writer = ArtifactBundleWriter(
        resolved_output,
        ArtifactRunContext("stage03.4", "control_parallel", run_label),
        config.artifact_storage,
    )
    writer.write_control(
        metadata={
            "schema_version": STAGE034_SCHEMA_VERSION,
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
            "exact_call_budget": config.exact_call_budget,
            "wall_clock_seconds": config.wall_clock_seconds,
            "watchdog_seconds": config.watchdog_seconds,
            "max_iterations": config.max_iterations,
            "worker_counts": [1, 4],
            "candidate_control": asdict(config.candidate_control_config),
            "inherit_stage033_incumbent": config.inherit_stage033_incumbent,
            "candidate_control_grid": {
                "proposal_top_k": list(config.proposal_top_k_grid),
                "max_exact_calls_per_round": list(config.round_budget_grid),
                "selection_order": list(config.selection_order),
            },
        },
        configuration_path=resolved_config,
    )
    try:
        for instance_name in instances:
            instance = parse_schneider(
                _resolve(root, config.benchmark_dir) / f"{instance_name}.txt"
            )
            for seed in config.seeds:
                initial_sequences, initial_provenance = _stage033_initial_solution(
                    stage033_review_manifest,
                    instance,
                    seed,
                )
                peak_before = _peak_rss_bytes()
                axes = run_control_parallel_diagnostic(
                    instance,
                    seed=seed,
                    vehicle_operator_config=stage02.vehicle_operator_config,
                    screening_config=config.screening_config,
                    cache_incremental_config=config.cache_incremental_config,
                    candidate_control_config=config.candidate_control_config,
                    exact_call_budget=config.exact_call_budget,
                    wall_clock_seconds=config.wall_clock_seconds,
                    watchdog_seconds=config.watchdog_seconds,
                    max_iterations=config.max_iterations,
                    batch_size=config.batch_size,
                    initial_customer_sequences=initial_sequences,
                    initial_solution_provenance=initial_provenance,
                )
                persist_paired_diagnostic(
                    writer,
                    instance=instance,
                    seed=seed,
                    pair=axes,
                    scope=scope,
                    environment_payload={
                        **environment,
                        "peak_rss_bytes": max(peak_before, _peak_rss_bytes()),
                    },
                    diagnostic_axes=DIAGNOSTIC_AXES,
                    schema_version=STAGE034_SCHEMA_VERSION,
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


def _require_stage033_ready(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "READY_FOR_STAGE03_4":
        raise RuntimeError("Stage 3.3 review is not READY_FOR_STAGE03_4")


def _stage033_initial_solution(
    review_manifest: Path,
    instance: Instance,
    seed: int,
) -> tuple[tuple[tuple[str, ...], ...], dict[str, object]]:
    review = json.loads(review_manifest.read_text(encoding="utf-8"))
    run_directory = review.get("run_directory")
    run_label = review.get("run_label")
    if not isinstance(run_directory, str) or not isinstance(run_label, str):
        raise RuntimeError("Stage 3.3 review lacks inherited-solution provenance")
    solution_path = (
        Path(run_directory)
        / instance.name
        / str(seed)
        / f"{run_label}_solution_{instance.name}_{seed}.json"
    )
    payload = json.loads(solution_path.read_text(encoding="utf-8"))
    wall_clock = payload.get("axes", {}).get("wall_clock")
    if not isinstance(wall_clock, Mapping):
        raise RuntimeError("Stage 3.3 inherited wall-clock solution is missing")
    routes = wall_clock.get("routes")
    objective_key = wall_clock.get("objective_key")
    if not isinstance(routes, list) or not isinstance(objective_key, list):
        raise RuntimeError("Stage 3.3 inherited solution payload is invalid")
    sequences = tuple(
        tuple(
            name
            for name in route
            if isinstance(name, str)
            and name in instance.by_name
            and instance.by_name[name].kind is NodeType.CUSTOMER
        )
        for route in routes
        if isinstance(route, list)
    )
    return sequences, {
        "source_stage": "stage03.3",
        "source_run_label": run_label,
        "source_axis": "wall_clock",
        "source_instance": instance.name,
        "source_seed": seed,
        "source_solution_sha256": _sha256(solution_path),
        "source_objective_key": objective_key,
    }


def _require_stage034_smoke_ready(
    path: Path | None,
    *,
    benchmark_dir: Path,
) -> None:
    if path is None:
        raise RuntimeError("formal Stage 3.4 requires a smoke review")
    manifest = path / "review_manifest.json" if path.is_dir() else path
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    review_dir = manifest.parent
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        raise RuntimeError("Stage 3.4 smoke review file hashes are missing")
    for name, expected in files.items():
        reviewed_file = review_dir / str(name)
        if not reviewed_file.is_file() or _sha256(reviewed_file) != str(expected):
            raise RuntimeError("Stage 3.4 smoke review file hash mismatch")
    run_directory = payload.get("run_directory")
    if not isinstance(run_directory, str):
        raise RuntimeError("Stage 3.4 smoke raw run directory is missing")
    raw_run_directory = Path(run_directory)
    raw_manifest = verify_manifest(raw_run_directory)
    # Rebuild the decision from raw evidence with the current reviewer. A set
    # of mutually consistent review hashes is not itself a trust anchor.
    from evrptw.experiments.stage034_control_parallel_review import review_stage034

    review_stage034(
        run_dir=raw_run_directory,
        scope="smoke",
        benchmark_dir=benchmark_dir,
        output_dir=review_dir,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "READY_FOR_STAGE034_FORMAL"
        or payload.get("scope") != "smoke"
        or payload.get("scope_complete") is not True
        or payload.get("all_rows_valid") is not True
        or payload.get("quality_gate") is not True
        or payload.get("semantics_gate") is not True
        or payload.get("performance_gate") is not True
        or payload.get("run_label") != raw_manifest.get("run_label")
    ):
        raise RuntimeError("Stage 3.4 smoke review is not ready for formal")


def _source_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "src/evrptw/alns.py",
        "src/evrptw/candidate_control.py",
        "src/evrptw/cpu_batch.py",
        "src/evrptw/exact_deadline.py",
        "src/evrptw/measurement.py",
        "src/evrptw/neighborhoods.py",
        "src/evrptw/artifacts.py",
        "src/evrptw/experiments/stage034_control_parallel.py",
    ):
        path = root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 3.4 control/parallel axes")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage034_control_parallel.toml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--smoke-review-dir", type=Path)
    arguments = parser.parse_args()
    outputs = run_stage034(
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
