"""Small, auditable Apple M5/Metal exact-charging pilot.

This runner is deliberately independent of Stage 3.3. It first records a
CPU-only profiler gate, then replays one fixed candidate-work manifest through
``cpu_scalar``, ``cpu_batch`` and ``metal_batch``. End-to-end ALNS paired runs
are attempted only after replay correctness and a positive Metal replay signal
have been established. Raw evidence remains under ``results/``; this module
never writes a tracked summary before the review command is run.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import hashlib
import json
import math
import pstats
import statistics
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evrptw.alns import ALNSResult, solve_alns
from evrptw.charging import ChargingSubproblemResult
from evrptw.environment import collect_environment
from evrptw.experiments.stage02_route_reduction import (
    FORMAL_SEEDS,
)
from evrptw.experiments.stage02_route_reduction import (
    load_config as load_stage02_config,
)
from evrptw.gpu_batch import (
    ExactChargingBackend,
    metal_backend_info,
    solve_exact_charging_batch,
)
from evrptw.measurement import (
    MeasurementConfig,
    Stage03ExecutionError,
    Stage03Trace,
    canonical_route_key,
)
from evrptw.models import Instance
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.validation import validate_routes

PILOT_SCHEMA_VERSION = "gpu-batch-pilot-v1"
RUN_LABEL = "gpu_batch_pilot_attempt01"
PILOT_INSTANCES = ("c101C5", "c101_21", "r101_21", "rc101_21")
PILOT_SEEDS = tuple(FORMAL_SEEDS)
HUNDRED_CUSTOMER_INSTANCES = frozenset({"c101_21", "r101_21", "rc101_21"})
REPLAY_BATCH_SIZES = (32, 128, 512)
DEFAULT_WARMUPS = 3
DEFAULT_MEASUREMENTS = 5
DEFAULT_FIXED_ITERATIONS = 40
DEFAULT_WATCHDOG_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class PilotConfig:
    root: Path
    benchmark_dir: Path
    stage02_config: Path
    output_dir: Path
    summary_dir: Path
    instances: tuple[str, ...] = PILOT_INSTANCES
    seeds: tuple[int, ...] = PILOT_SEEDS
    replay_batch_sizes: tuple[int, ...] = REPLAY_BATCH_SIZES
    warmups: int = DEFAULT_WARMUPS
    measurements: int = DEFAULT_MEASUREMENTS
    fixed_iterations: int = DEFAULT_FIXED_ITERATIONS
    watchdog_seconds: float = DEFAULT_WATCHDOG_SECONDS
    profiler_time_limit_seconds: float = 30.0
    profiler_max_iterations: int = 1000

    @classmethod
    def defaults(cls, root: Path | None = None) -> PilotConfig:
        resolved_root = root or Path(__file__).resolve().parents[3]
        return cls(
            root=resolved_root,
            benchmark_dir=resolved_root / "data" / "schneider",
            stage02_config=resolved_root / "configs" / "stage02_constraint_guided.toml",
            output_dir=resolved_root / "results" / RUN_LABEL,
            summary_dir=resolved_root / "experiments" / "summaries",
        )


def run_profile(config: PilotConfig) -> dict[str, Path]:
    """Run the CPU-only profiler phase and write the explicit GPU gate."""

    _prepare_output(config.output_dir, require_empty=True)
    stage02 = load_stage02_config(config.stage02_config)
    profile_dir = config.output_dir / "profile"
    rows: list[dict[str, object]] = []
    cprofile_paths: list[str] = []
    for instance_name in config.instances:
        instance = _load_instance(config, instance_name)
        for seed in config.seeds:
            started = time.perf_counter()
            result: ALNSResult | None = None
            error: BaseException | None = None
            profile_path: Path | None = None

            def solve(
                current_instance: Instance = instance,
                current_seed: int = seed,
            ) -> None:
                nonlocal result
                result = solve_alns(
                    current_instance,
                    seed=current_seed,
                    max_iterations=config.profiler_max_iterations,
                    time_limit_seconds=config.profiler_time_limit_seconds,
                    operator_profile="stage02_constraint_guided",
                    vehicle_operator_config=stage02.vehicle_operator_config,
                    measurement_config=MeasurementConfig(),
                    backend=ExactChargingBackend.CPU_SCALAR,
                )

            try:
                if instance_name in HUNDRED_CUSTOMER_INSTANCES and seed == config.seeds[0]:
                    profile_path = profile_dir / f"{instance_name}_{seed}.prof"
                    profiler = cProfile.Profile()
                    profiler.enable()
                    solve()
                    profiler.disable()
                    profile_path.parent.mkdir(parents=True, exist_ok=True)
                    profiler.dump_stats(profile_path)
                    cprofile_paths.append(str(profile_path.relative_to(config.output_dir)))
                    summary_path = profile_dir / f"{instance_name}_{seed}_top30.txt"
                    with summary_path.open("w", encoding="utf-8") as handle, redirect_stdout(
                        handle
                    ):
                        pstats.Stats(profiler).strip_dirs().sort_stats("cumtime").print_stats(30)
                else:
                    solve()
            except BaseException as caught:
                error = caught
            runtime = time.perf_counter() - started
            trace = result.measurement_trace if result is not None else None
            exact_time = _trace_exact_seconds(trace)
            total_time = result.runtime_seconds if result is not None else runtime
            row: dict[str, object] = {
                "instance": instance_name,
                "seed": seed,
                "customer_count": len(instance.customers),
                "backend": ExactChargingBackend.CPU_SCALAR.value,
                "runtime_seconds": total_time,
                "exact_time_seconds": exact_time,
                "exact_time_share": exact_time / total_time if total_time > 0 else math.nan,
                "exact_calls": result.charging_subproblem_calls if result else 0,
                "iterations": result.iterations if result else 0,
                "status": "PASS" if error is None and result is not None else "ERROR",
                "error": "" if error is None else f"{type(error).__name__}: {error}",
                "cprofile_path": (
                    ""
                    if profile_path is None
                    else str(profile_path.relative_to(config.output_dir))
                ),
            }
            rows.append(row)
            _write_json(
                config.output_dir / "profile" / f"{instance_name}_{seed}.json",
                {"schema_version": PILOT_SCHEMA_VERSION, "row": row},
            )

    medians: dict[str, float] = {}
    for instance_name in HUNDRED_CUSTOMER_INSTANCES:
        values = [
            _as_float(row["exact_time_share"])
            for row in rows
            if row["instance"] == instance_name and row["status"] == "PASS"
        ]
        if values:
            medians[instance_name] = statistics.median(values)
    gate_pass = (
        len(medians) == len(HUNDRED_CUSTOMER_INSTANCES)
        and all(value >= 0.5 for value in medians.values())
    )
    gate = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "run_label": RUN_LABEL,
        "status": "PASS_EXACT_TIME_DOMINANT" if gate_pass else "STOP_BEFORE_GPU",
        "criterion": "all three 100-customer instance medians have exact-time share >= 0.50",
        "median_exact_time_share": medians,
        "cprofile_paths": cprofile_paths,
        "rows": rows,
        "environment": collect_environment(),
    }
    gate_path = config.output_dir / "profile_gate.json"
    _write_json(gate_path, gate)
    manifest_path = _finalize_raw_manifest(config.output_dir, status=gate["status"])
    return {"profile_gate": gate_path, "raw_manifest": manifest_path}


def run_replay(config: PilotConfig) -> dict[str, Path]:
    """Create fixed work manifests and run the 3+5 replay benchmark."""

    if (config.output_dir / "replay_results.json").exists() or (
        config.output_dir / "replay_gate.json"
    ).exists():
        raise FileExistsError(
            "replay evidence already exists; choose a new pilot attempt directory"
        )
    gate = _read_json(config.output_dir / "profile_gate.json")
    if gate.get("status") != "PASS_EXACT_TIME_DOMINANT":
        raise RuntimeError("GPU replay is blocked because the profiler gate did not pass")
    _prepare_output(config.output_dir)
    stage02 = load_stage02_config(config.stage02_config)
    all_rows: list[dict[str, object]] = []
    for instance_name in config.instances:
        instance = _load_instance(config, instance_name)
        for seed in config.seeds:
            reference = _run_fixed_work(
                instance,
                seed,
                stage02,
                backend=ExactChargingBackend.CPU_SCALAR,
                batch_size=1,
                config=config,
            )
            if reference["error"]:
                raise RuntimeError(
                    f"CPU reference failed for {instance_name}/{seed}: {reference['error']}"
                )
            result = reference["result"]
            if not isinstance(result, ALNSResult):
                raise RuntimeError("CPU reference did not return an ALNSResult")
            if (
                result.effective_iterations != config.fixed_iterations
                or result.watchdog_triggered
            ):
                raise RuntimeError(
                    f"fixed-work reference did not complete {config.fixed_iterations} "
                    f"iterations for {instance_name}/{seed}"
                )
            work_items = _work_items(result, seed)
            orders = tuple(_order_from_item(item) for item in work_items)
            reference_batch = solve_exact_charging_batch(
                instance,
                orders,
                backend=ExactChargingBackend.CPU_SCALAR,
                batch_size=1,
            )
            for item, route_result in zip(work_items, reference_batch.results, strict=True):
                item["reference_route_result_hash"] = _charging_result_hash(route_result)
            candidate_manifest = {
                "schema_version": PILOT_SCHEMA_VERSION,
                "run_label": RUN_LABEL,
                "instance": instance_name,
                "seed": seed,
                "termination_mode": "fixed_work",
                "max_iterations": config.fixed_iterations,
                "watchdog_seconds": config.watchdog_seconds,
                "effective_iterations": result.effective_iterations,
                "exact_calls": result.charging_subproblem_calls,
                "candidate_work_hash": _candidate_work_hash(work_items, seed),
                "items": work_items,
                "reference": _alns_summary(result, instance, seed),
            }
            manifest_path = (
                config.output_dir
                / "candidate_work"
                / instance_name
                / f"{seed}.json"
            )
            _write_json(manifest_path, candidate_manifest)
            for backend, batch_sizes in (
                (ExactChargingBackend.CPU_SCALAR, (1,)),
                (ExactChargingBackend.CPU_BATCH, config.replay_batch_sizes),
                (ExactChargingBackend.METAL_BATCH, config.replay_batch_sizes),
            ):
                for batch_size in batch_sizes:
                    row = _benchmark_replay(
                        instance,
                        instance_name,
                        seed,
                        work_items,
                        reference_batch.results,
                        backend,
                        batch_size,
                        config,
                    )
                    all_rows.append(row)
                    _write_json(
                        config.output_dir
                        / "replay"
                        / instance_name
                        / str(seed)
                        / f"{backend.value}_{batch_size}.json",
                        row,
                    )
    replay_path = config.output_dir / "replay_results.json"
    _write_json(
        replay_path,
        {
            "schema_version": PILOT_SCHEMA_VERSION,
            "metal_backend_info": metal_backend_info(),
            "rows": all_rows,
        },
    )
    gate_path = config.output_dir / "replay_gate.json"
    replay_gate = _replay_gate(all_rows)
    _write_json(gate_path, replay_gate)
    manifest_path = _finalize_raw_manifest(config.output_dir, status=replay_gate["status"])
    return {"replay_results": replay_path, "replay_gate": gate_path, "raw_manifest": manifest_path}


def run_paired(config: PilotConfig) -> dict[str, Path]:
    """Run fixed-work ALNS pairs only after replay correctness and gain."""

    if (config.output_dir / "paired_results.json").exists():
        raise FileExistsError(
            "paired evidence already exists; choose a new pilot attempt directory"
        )
    replay_gate = _read_json(config.output_dir / "replay_gate.json")
    if replay_gate.get("status") != "PASS_REPLAY_AND_GPU_POSITIVE":
        raise RuntimeError("paired ALNS is blocked because replay did not justify it")
    stage02 = load_stage02_config(config.stage02_config)
    replay_rows = _read_json(config.output_dir / "replay_results.json")["rows"]
    batch_size = _select_metal_batch_size(replay_rows)
    candidate_work_hashes = _load_candidate_work_hashes(config)
    rows: list[dict[str, object]] = []
    for instance_name in config.instances:
        instance = _load_instance(config, instance_name)
        for seed in config.seeds:
            cpu = _run_fixed_work(
                instance,
                seed,
                stage02,
                backend=ExactChargingBackend.CPU_SCALAR,
                batch_size=1,
                config=config,
            )
            gpu = _run_fixed_work(
                instance,
                seed,
                stage02,
                backend=ExactChargingBackend.METAL_BATCH,
                batch_size=batch_size,
                config=config,
            )
            manifest_hash = candidate_work_hashes[(instance_name, seed)]
            row = _paired_row(
                instance_name,
                seed,
                instance,
                cpu,
                gpu,
                batch_size,
                manifest_hash,
                config.fixed_iterations,
            )
            rows.append(row)
            _write_json(
                config.output_dir / "paired" / instance_name / f"{seed}.json",
                row,
            )
    paired_path = config.output_dir / "paired_results.json"
    _write_json(
        paired_path,
        {
            "schema_version": PILOT_SCHEMA_VERSION,
            "backend": ExactChargingBackend.METAL_BATCH.value,
            "batch_size": batch_size,
            "rows": rows,
        },
    )
    manifest_path = _finalize_raw_manifest(config.output_dir, status="PAIRED_COMPLETE")
    return {"paired_results": paired_path, "raw_manifest": manifest_path}


def review_pilot(config: PilotConfig) -> dict[str, Path]:
    """Independently verify raw hashes and publish reviewed pilot summaries."""

    raw_manifest_path = config.output_dir / "raw_manifest.json"
    manifest = _read_json(raw_manifest_path)
    checks: dict[str, bool] = {}
    manifest_sidecar = config.output_dir / "raw_manifest.sha256"
    checks["manifest_sidecar"] = (
        manifest_sidecar.is_file()
        and manifest_sidecar.read_text(encoding="utf-8").strip() == _sha256(raw_manifest_path)
    )
    for relative, reference in dict(manifest.get("files", {})).items():
        path = config.output_dir / str(relative)
        checks[str(relative)] = (
            path.is_file()
            and _sha256(path) == str(dict(reference).get("sha256", ""))
        )
    checks["manifest_present"] = raw_manifest_path.is_file()
    checks["no_stage03_claim"] = (
        manifest.get("historical_stage03_claim") == "none"
        and manifest.get("stage03_3_readiness_claim") is False
    )
    profile_path = config.output_dir / "profile_gate.json"
    profile_gate = _read_json(profile_path) if profile_path.is_file() else {}
    checks["profile_gate_present"] = profile_path.is_file()
    checks["profile_gate_passed"] = (
        profile_gate.get("status") == "PASS_EXACT_TIME_DOMINANT"
    )
    checks["profile_scope_complete"] = _review_profile_scope(config, profile_gate)
    checks["candidate_work_replay_hashes"] = _review_candidate_work_replay_hashes(
        config,
    )
    replay_path = config.output_dir / "replay_results.json"
    if replay_path.is_file():
        replay = _read_json(replay_path)
        replay_rows = replay.get("rows", [])
        checks["replay_rows_correct"] = (
            isinstance(replay_rows, list)
            and bool(replay_rows)
            and all(bool(row.get("correctness", False)) for row in replay_rows)
            and _review_replay_scope(config, replay_rows)
        )
    else:
        checks["replay_rows_correct"] = False
    paired_path = config.output_dir / "paired_results.json"
    paired_rows = _read_json(paired_path).get("rows", []) if paired_path.is_file() else []
    checks["paired_rows_valid_or_not_run"] = (
        not paired_rows or all(bool(row.get("paired_valid", False)) for row in paired_rows)
    )
    checks["paired_scope_complete_or_not_run"] = (
        not paired_rows or _review_paired_scope(config, paired_rows)
    )
    checks["validator_objective_replay"] = (
        _review_replay_validator_objective(config)
        and _review_paired_validator_objective(paired_rows, config)
    )
    go_no_go = _go_no_go(paired_rows)
    checks["go_no_go_recorded"] = bool(go_no_go.get("status"))
    review_status = "READY_FOR_PILOT_REPORT" if all(checks.values()) else "REVIEW_FAILED"
    replay_summary = config.summary_dir / f"{RUN_LABEL}_replay_results.csv"
    paired_summary = config.summary_dir / f"{RUN_LABEL}_paired_results.csv"
    report_body = (
        "# GPU batch pilot independent review\n\n"
        f"- status: `{review_status}`\n"
        "- This is an independent pilot report; it does not claim Stage 3.3 readiness.\n"
        f"- go_no_go: `{go_no_go['status']}`\n"
        f"- checks_passed: {sum(checks.values())}/{len(checks)}\n"
    )
    raw_report_path = config.output_dir / "review_report.md"
    raw_report_path.write_text(report_body, encoding="utf-8")
    report_path = raw_report_path
    if review_status == "READY_FOR_PILOT_REPORT":
        config.summary_dir.mkdir(parents=True, exist_ok=True)
        if replay_path.is_file():
            replay_rows = _read_json(replay_path).get("rows", [])
            _write_csv(replay_summary, replay_rows)
        if paired_rows:
            _write_csv(paired_summary, paired_rows)
        report_path = config.summary_dir / f"{RUN_LABEL}_review_report.md"
        report_path.write_text(report_body, encoding="utf-8")
    review_path = config.output_dir / "review.json"
    _write_json(
        review_path,
        {
            "schema_version": PILOT_SCHEMA_VERSION,
            "status": review_status,
            "checks": checks,
            "go_no_go": go_no_go,
            "summary_paths": {
                "replay": str(replay_summary) if replay_summary.is_file() else "",
                "paired": str(paired_summary) if paired_summary.is_file() else "",
                "report": str(report_path),
            },
        },
    )
    return {"review": review_path, "report": report_path}


def _review_candidate_work_replay_hashes(config: PilotConfig) -> bool:
    replay_path = config.output_dir / "replay_results.json"
    if not replay_path.is_file():
        return False
    replay_rows = _read_json(replay_path).get("rows", [])
    if not isinstance(replay_rows, list):
        return False
    expected_by_key: dict[tuple[str, int], str] = {}
    candidate_paths = sorted((config.output_dir / "candidate_work").glob("*/*.json"))
    expected_keys = {(instance, seed) for instance in config.instances for seed in config.seeds}
    observed_keys: set[tuple[str, int]] = set()
    for path in candidate_paths:
        payload = _read_json(path)
        raw_items = payload.get("items")
        if not isinstance(raw_items, list) or not all(
            isinstance(item, dict) for item in raw_items
        ):
            return False
        items = [dict(item) for item in raw_items]
        if len(items) != _as_int(payload.get("exact_calls", -1)):
            return False
        expected_hash = _candidate_work_hash(items, _as_int(payload.get("seed")))
        if expected_hash != str(payload.get("candidate_work_hash", "")):
            return False
        key = (str(payload.get("instance", "")), _as_int(payload.get("seed")))
        expected_by_key[key] = expected_hash
        observed_keys.add(key)
        instance = _load_instance(config, key[0])
        orders = tuple(_order_from_item(item) for item in items)
        replayed = solve_exact_charging_batch(
            instance,
            orders,
            backend=ExactChargingBackend.CPU_SCALAR,
            batch_size=1,
        ).results
        for item, result in zip(items, replayed, strict=True):
            if str(item.get("reference_route_result_hash", "")) != _charging_result_hash(
                result
            ):
                return False
            if result.feasible:
                report = validate_routes(instance, [list(result.route)])
                if not report.routes[0].feasible:
                    return False
                route_objective = SolutionObjective.from_route(
                    instance,
                    result.route,
                    total_distance=result.distance,
                    total_charging_time=result.charging_time,
                )
                route_report = report.routes[0]
                if (
                    route_objective.total_distance != route_report.distance
                    or route_objective.total_charging_time != route_report.charging_time
                ):
                    return False
    if observed_keys != expected_keys:
        return False
    for raw_row in replay_rows:
        if not isinstance(raw_row, dict):
            return False
        key = (str(raw_row.get("instance", "")), _as_int(raw_row.get("seed")))
        if str(raw_row.get("candidate_work_hash", "")) != expected_by_key.get(key):
            return False
    return True


def _load_candidate_work_hashes(config: PilotConfig) -> dict[tuple[str, int], str]:
    hashes: dict[tuple[str, int], str] = {}
    for instance_name in config.instances:
        for seed in config.seeds:
            path = config.output_dir / "candidate_work" / instance_name / f"{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = _read_json(path)
            key = (instance_name, seed)
            value = str(payload.get("candidate_work_hash", ""))
            if not value:
                raise ValueError(f"candidate-work manifest has no hash: {path}")
            hashes[key] = value
    return hashes


def _review_profile_scope(config: PilotConfig, gate: dict[str, object]) -> bool:
    rows = gate.get("rows")
    if not isinstance(rows, list):
        return False
    expected = {(instance, seed) for instance in config.instances for seed in config.seeds}
    observed = {
        (str(row.get("instance")), _as_int(row.get("seed")))
        for row in rows
        if isinstance(row, dict)
    }
    return observed == expected and all(
        isinstance(row, dict) and row.get("status") == "PASS" for row in rows
    )


def _review_replay_scope(config: PilotConfig, rows: list[object]) -> bool:
    expected = {
        (instance, seed, backend, batch_size)
        for instance in config.instances
        for seed in config.seeds
        for backend, batch_sizes in (
            (ExactChargingBackend.CPU_SCALAR.value, (1,)),
            (ExactChargingBackend.CPU_BATCH.value, config.replay_batch_sizes),
            (ExactChargingBackend.METAL_BATCH.value, config.replay_batch_sizes),
        )
        for batch_size in batch_sizes
    }
    observed = {
        (
            str(row.get("instance")),
            _as_int(row.get("seed")),
            str(row.get("backend")),
            _as_int(row.get("batch_size")),
        )
        for row in rows
        if isinstance(row, dict)
    }
    return observed == expected


def _review_paired_scope(config: PilotConfig, rows: list[object]) -> bool:
    expected = {(instance, seed) for instance in config.instances for seed in config.seeds}
    observed = {
        (str(row.get("instance")), _as_int(row.get("seed")))
        for row in rows
        if isinstance(row, dict)
    }
    return observed == expected


def _review_replay_validator_objective(config: PilotConfig) -> bool:
    candidate_dir = config.output_dir / "candidate_work"
    for instance_name in config.instances:
        for seed in config.seeds:
            path = candidate_dir / instance_name / f"{seed}.json"
            if not path.is_file():
                return False
            payload = _read_json(path)
            items = payload.get("items")
            if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
                return False
            instance = _load_instance(config, instance_name)
            results = solve_exact_charging_batch(
                instance,
                tuple(_order_from_item(dict(item)) for item in items),
                backend=ExactChargingBackend.CPU_SCALAR,
                batch_size=1,
            ).results
            for result in results:
                if not result.feasible:
                    continue
                report = validate_routes(instance, [list(result.route)])
                if not report.routes[0].feasible:
                    return False
                SolutionObjective.from_route(
                    instance,
                    result.route,
                    total_distance=report.routes[0].distance,
                    total_charging_time=report.routes[0].charging_time,
                )
    return True


def _go_no_go(paired_rows: list[object]) -> dict[str, object]:
    valid_rows = [
        row
        for row in paired_rows
        if isinstance(row, dict)
        and bool(row.get("paired_valid", False))
        and row.get("saving_percent") is not None
    ]
    if not valid_rows:
        return {
            "status": "NOT_RUN",
            "criterion": "median saving >= 10% and at least two improved 100-customer families",
            "family_median_saving_percent": {},
            "overall_median_saving_percent": None,
            "improved_100_customer_families": 0,
        }
    family_values: dict[str, list[float]] = {}
    for row in valid_rows:
        instance = str(row.get("instance", ""))
        if instance in HUNDRED_CUSTOMER_INSTANCES:
            family_values.setdefault(instance, []).append(_as_float(row["saving_percent"]))
    family_medians = {
        instance: statistics.median(values) for instance, values in family_values.items()
    }
    overall_values = [value for values in family_values.values() for value in values]
    overall_median = statistics.median(overall_values) if overall_values else None
    improved = sum(value > 0.0 for value in family_medians.values())
    go = (
        overall_median is not None
        and overall_median >= 10.0
        and improved >= 2
        and len(family_medians) == len(HUNDRED_CUSTOMER_INSTANCES)
    )
    return {
        "status": "GO" if go else "NO_GO",
        "criterion": "median saving >= 10% and at least two improved 100-customer families",
        "family_median_saving_percent": family_medians,
        "overall_median_saving_percent": overall_median,
        "improved_100_customer_families": improved,
    }


def _review_paired_validator_objective(
    paired_rows: list[object],
    config: PilotConfig,
) -> bool:
    for raw_row in paired_rows:
        if not isinstance(raw_row, dict):
            return False
        instance_name = str(raw_row.get("instance", ""))
        try:
            instance = _load_instance(config, instance_name)
        except (FileNotFoundError, ValueError):
            return False
        for side_name in ("cpu", "gpu"):
            side = raw_row.get(side_name)
            if not isinstance(side, dict):
                return False
            if not bool(side.get("feasible", False)):
                return False
            raw_routes = side.get("routes")
            if not isinstance(raw_routes, list):
                return False
            routes = [list(route) for route in raw_routes if isinstance(route, list)]
            if len(routes) != len(raw_routes):
                return False
            report = validate_routes(instance, routes)
            if not report.feasible or not bool(side.get("validator_feasible", False)):
                return False
            objective = SolutionObjective.from_report(instance, report)
            if list(objective.key) != list(side.get("objective_key", [])):
                return False
    return True


def _run_fixed_work(
    instance: Instance,
    seed: int,
    stage02: Any,
    *,
    backend: ExactChargingBackend,
    batch_size: int,
    config: PilotConfig,
) -> dict[str, object]:
    started = time.perf_counter()
    result: ALNSResult | None = None
    error: BaseException | None = None
    trace: Stage03Trace | None = None
    try:
        result = solve_alns(
            instance,
            seed=seed,
            max_iterations=config.fixed_iterations,
            time_limit_seconds=config.watchdog_seconds,
            operator_profile="stage02_constraint_guided",
            vehicle_operator_config=stage02.vehicle_operator_config,
            measurement_config=MeasurementConfig(),
            backend=backend,
            batch_size=batch_size,
            termination_mode="fixed_work",
            disable_cache=True,
        )
        trace = result.measurement_trace
    except BaseException as caught:
        error = caught
        if isinstance(caught, Stage03ExecutionError):
            trace = caught.trace
    elapsed = time.perf_counter() - started
    return {
        "backend": backend.value,
        "batch_size": batch_size,
        "result": result,
        "trace": trace,
        "error": "" if error is None else f"{type(error).__name__}: {error}",
        "elapsed_seconds": elapsed,
    }


def _benchmark_replay(
    instance: Instance,
    instance_name: str,
    seed: int,
    work_items: list[dict[str, object]],
    reference_results: tuple[ChargingSubproblemResult, ...],
    backend: ExactChargingBackend,
    batch_size: int,
    config: PilotConfig,
) -> dict[str, object]:
    orders = tuple(_order_from_item(item) for item in work_items)
    expected_hashes = tuple(
        str(item.get("reference_route_result_hash", "")) for item in work_items
    )
    if not all(expected_hashes):
        expected_hashes = tuple(_charging_result_hash(item) for item in reference_results)
    samples: list[dict[str, object]] = []
    correctness = True
    cold_initialization_seconds = 0.0
    for warmup in range(config.warmups + config.measurements):
        started = time.perf_counter()
        try:
            batch = solve_exact_charging_batch(
                instance,
                orders,
                backend=backend,
                batch_size=batch_size,
            )
            elapsed = time.perf_counter() - started
            cold_initialization_seconds = max(
                cold_initialization_seconds,
                batch.metrics.initialization_seconds,
            )
            sample_correct = _result_signatures_equal(batch.results, reference_results)
            sample_correct = sample_correct and tuple(
                _charging_result_hash(item) for item in batch.results
            ) == expected_hashes
            correctness = correctness and sample_correct
            if warmup >= config.warmups:
                samples.append(
                    {
                        "total_seconds": elapsed,
                        "kernel_seconds": batch.metrics.kernel_seconds,
                        "transfer_seconds": batch.metrics.transfer_seconds,
                        "packing_seconds": batch.metrics.packing_seconds,
                        "unpacking_seconds": batch.metrics.unpacking_seconds,
                        "cpu_postprocess_seconds": batch.metrics.cpu_postprocess_seconds,
                        "initialization_seconds": batch.metrics.initialization_seconds,
                        "batch_launches": batch.metrics.batch_launches,
                        "transitions": batch.metrics.transitions,
                        "exact_calls": batch.metrics.exact_calls,
                        "correctness": sample_correct,
                    }
                )
        except BaseException as error:
            correctness = False
            if warmup >= config.warmups:
                samples.append(
                    {
                        "total_seconds": None,
                        "error": f"{type(error).__name__}: {error}",
                        "correctness": False,
                    }
                )
    valid_samples = [sample for sample in samples if sample.get("total_seconds") is not None]
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "instance": instance_name,
        "seed": seed,
        "backend": backend.value,
        "batch_size": batch_size,
        "warmups": config.warmups,
        "measurements": config.measurements,
        "correctness": correctness and len(valid_samples) == config.measurements,
        "candidate_work_hash": _candidate_work_hash(work_items, seed),
        "total_seconds_median": _median(valid_samples, "total_seconds"),
        "kernel_seconds_median": _median(valid_samples, "kernel_seconds"),
        "transfer_seconds_median": _median(valid_samples, "transfer_seconds"),
        "packing_seconds_median": _median(valid_samples, "packing_seconds"),
        "unpacking_seconds_median": _median(valid_samples, "unpacking_seconds"),
        "cpu_postprocess_seconds_median": _median(valid_samples, "cpu_postprocess_seconds"),
        "initialization_seconds": cold_initialization_seconds,
        "batch_launches": max(
            (_as_int(sample.get("batch_launches", 0)) for sample in valid_samples),
            default=0,
        ),
        "transitions": max(
            (_as_int(sample.get("transitions", 0)) for sample in valid_samples),
            default=0,
        ),
        "exact_calls": max(
            (_as_int(sample.get("exact_calls", 0)) for sample in valid_samples),
            default=0,
        ),
        "samples": samples,
    }


def _replay_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    correct = bool(rows) and all(bool(row.get("correctness", False)) for row in rows)
    cpu = {
        (str(row["instance"]), _as_int(row["seed"])): _as_float(
            row["total_seconds_median"]
        )
        for row in rows
        if row.get("backend") == ExactChargingBackend.CPU_SCALAR.value
        and row.get("total_seconds_median") is not None
    }
    metal = [
        row
        for row in rows
        if row.get("backend") == ExactChargingBackend.METAL_BATCH.value
        and row.get("total_seconds_median") is not None
        and (str(row["instance"]), _as_int(row["seed"])) in cpu
    ]
    ratios_by_batch: dict[int, list[float]] = {}
    for row in metal:
        if str(row["instance"]) not in HUNDRED_CUSTOMER_INSTANCES:
            continue
        baseline = cpu[(str(row["instance"]), _as_int(row["seed"]))]
        ratios_by_batch.setdefault(_as_int(row["batch_size"]), []).append(
            _as_float(row["total_seconds_median"]) / baseline
        )
    median_ratio_by_batch = {
        str(batch_size): statistics.median(ratios)
        for batch_size, ratios in ratios_by_batch.items()
    }
    positive = bool(median_ratio_by_batch) and min(median_ratio_by_batch.values()) < 1.0
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "status": "PASS_REPLAY_AND_GPU_POSITIVE" if correct and positive else "STOP_BEFORE_PAIRED",
        "all_replay_correct": correct,
        "gpu_positive_on_100_customer_replay_median": positive,
        "median_total_ratio_by_batch_size": median_ratio_by_batch,
        "criterion": (
            "all replay signatures match and at least one Metal batch size has "
            "a sub-1.0 median total-time ratio on the 100-customer scope"
        ),
    }


def _paired_row(
    instance_name: str,
    seed: int,
    instance: Instance,
    cpu: dict[str, object],
    gpu: dict[str, object],
    batch_size: int,
    manifest_candidate_work_hash: str,
    expected_iterations: int,
) -> dict[str, object]:
    cpu_value = cpu.get("result")
    gpu_value = gpu.get("result")
    cpu_result = cpu_value if isinstance(cpu_value, ALNSResult) else None
    gpu_result = gpu_value if isinstance(gpu_value, ALNSResult) else None
    cpu_summary = (
        _alns_summary(cpu_result, instance, seed) if cpu_result is not None else {}
    )
    gpu_summary = (
        _alns_summary(gpu_result, instance, seed) if gpu_result is not None else {}
    )
    equal_fields = {
        "candidate_work_hash": cpu_summary.get("candidate_work_hash")
        == gpu_summary.get("candidate_work_hash"),
        "candidate_work_manifest": (
            cpu_summary.get("candidate_work_hash") == manifest_candidate_work_hash
            and gpu_summary.get("candidate_work_hash") == manifest_candidate_work_hash
        ),
        "exact_calls": cpu_summary.get("exact_calls") == gpu_summary.get("exact_calls"),
        "route_result_hash": cpu_summary.get("route_result_hash")
        == gpu_summary.get("route_result_hash"),
        "objective_key": cpu_summary.get("objective_key") == gpu_summary.get("objective_key"),
        "effective_iterations": cpu_summary.get("effective_iterations")
        == gpu_summary.get("effective_iterations")
        == expected_iterations,
        "trace_reconciliation": (
            cpu_summary.get("trace_reconciliation_status") == "pass"
            and gpu_summary.get("trace_reconciliation_status") == "pass"
        ),
    }
    valid = (
        not cpu.get("error")
        and not gpu.get("error")
        and bool(cpu_summary.get("validator_feasible"))
        and bool(gpu_summary.get("validator_feasible"))
        and not bool(cpu_summary.get("watchdog_triggered"))
        and not bool(gpu_summary.get("watchdog_triggered"))
        and all(equal_fields.values())
    )
    cpu_total = _alns_total(cpu_result)
    gpu_total = _alns_total(gpu_result)
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "instance": instance_name,
        "seed": seed,
        "backend": ExactChargingBackend.METAL_BATCH.value,
        "batch_size": batch_size,
        "paired_valid": valid,
        "invalid_reasons": [name for name, passed in equal_fields.items() if not passed],
        "equal_fields": equal_fields,
        "cpu": cpu_summary,
        "gpu": gpu_summary,
        "cpu_total_seconds": cpu_total,
        "gpu_total_seconds": gpu_total,
        "cpu_exact_charging_seconds": _trace_exact_seconds(
            cpu_result.measurement_trace if cpu_result is not None else None
        ),
        "gpu_exact_charging_seconds": _trace_exact_seconds(
            gpu_result.measurement_trace if gpu_result is not None else None
        ),
        "saving_percent": (
            (cpu_total - gpu_total) / cpu_total * 100.0
            if valid and cpu_total is not None and gpu_total is not None and cpu_total > 0
            else None
        ),
        "speedup": (
            cpu_total / gpu_total
            if valid and cpu_total is not None and gpu_total is not None and gpu_total > 0
            else None
        ),
    }


def _alns_summary(
    result: ALNSResult | None,
    instance: Instance | None,
    seed: int | None = None,
) -> dict[str, object]:
    if result is None:
        return {}
    validator_feasible = False
    if instance is not None and result.feasible:
        validator_feasible = validate_routes(
            instance,
            [list(route) for route in result.routes],
        ).feasible
    work_hash = ""
    reconciliation_status = "not_available"
    if result.measurement_trace is not None and seed is not None:
        work_hash = _candidate_work_hash(_work_items(result, seed), seed)
        reconciliation_status = str(result.measurement_trace.reconcile(result)["status"])
    return {
        "feasible": result.feasible,
        "validator_feasible": validator_feasible,
        "runtime_seconds": result.runtime_seconds,
        "exact_charging_seconds": _trace_exact_seconds(result.measurement_trace),
        "exact_calls": result.charging_subproblem_calls,
        "effective_iterations": result.effective_iterations,
        "iterations": result.iterations,
        "watchdog_triggered": result.watchdog_triggered,
        "objective_key": list(result.objective.key) if result.objective is not None else [],
        "route_result_hash": _route_result_hash(result),
        "routes": [list(route) for route in result.routes],
        "customer_sequences": [list(route) for route in result.customer_sequences],
        "candidate_work_hash": work_hash,
        "trace_reconciliation_status": reconciliation_status,
        "backend_metrics": result.backend_metrics,
        "failure_reason": result.failure_reason,
    }


def _work_items(result: ALNSResult, seed: int) -> list[dict[str, object]]:
    trace = result.measurement_trace
    if trace is None:
        raise RuntimeError("candidate-work manifest requires a Stage03Trace")
    items: list[dict[str, object]] = []
    for record in trace.route_evaluations:
        if record.kind != "exact_call":
            continue
        sequence = trace.route_dictionary[record.route_key]
        if canonical_route_key(sequence) != record.route_key:
            raise RuntimeError(f"trace route key is not canonical: {record.route_key}")
        items.append(
            {
                "request_order": len(items),
                "trace_evaluation_id": record.evaluation_id,
                "seed": seed,
                "iteration": record.iteration,
                "lane": record.lane,
                "operator": record.operator,
                "route_sequence": list(sequence),
                "canonical_route_key": record.route_key,
                "exact_cache_state": "disabled",
                "cache_status": record.kind,
            }
        )
    if not items:
        raise RuntimeError("candidate-work manifest contains no exact charging calls")
    return items


def _candidate_work_hash(items: list[dict[str, object]], seed: int) -> str:
    canonical = {
        "seed": seed,
        "requests": [
            {
                key: item[key]
                for key in (
                    "request_order",
                    "trace_evaluation_id",
                    "seed",
                    "iteration",
                    "lane",
                    "operator",
                    "route_sequence",
                    "canonical_route_key",
                    "exact_cache_state",
                    "cache_status",
                )
            }
            for item in items
        ],
    }
    return _payload_hash(canonical)


def _result_signatures_equal(
    actual: tuple[ChargingSubproblemResult, ...],
    expected: tuple[ChargingSubproblemResult, ...],
) -> bool:
    return len(actual) == len(expected) and all(
        _charging_result_hash(left) == _charging_result_hash(right)
        for left, right in zip(actual, expected, strict=True)
    )


def _charging_result_hash(result: ChargingSubproblemResult) -> str:
    """Hash semantic route output, excluding diagnostic label counters."""

    return _payload_hash(
        {
            "feasible": result.feasible,
            "route": list(result.route),
            "distance": result.distance,
            "total_energy": result.total_energy,
            "charged_energy": result.charged_energy,
            "charging_time": result.charging_time,
            "failure_reason": result.failure_reason,
        }
    )


def _route_result_hash(result: ALNSResult) -> str:
    return _payload_hash(
        {
            "routes": [list(route) for route in result.routes],
            "customer_sequences": [list(route) for route in result.customer_sequences],
        }
    )


def _alns_total(result: ALNSResult | None) -> float | None:
    if result is None:
        return None
    return result.runtime_seconds


def _trace_exact_seconds(trace: Stage03Trace | None) -> float:
    if trace is None:
        return 0.0
    return sum(
        record.duration_seconds
        for record in trace.route_evaluations
        if record.kind == "exact_call" and record.exact_completed
    )


def _median(rows: list[dict[str, object]], key: str) -> float | None:
    values = [_as_float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def _select_metal_batch_size(rows: list[dict[str, object]]) -> int:
    candidates: dict[int, list[float]] = {}
    for row in rows:
        if (
            row.get("backend") == ExactChargingBackend.METAL_BATCH.value
            and row.get("correctness")
            and row.get("total_seconds_median") is not None
            and str(row.get("instance")) in HUNDRED_CUSTOMER_INSTANCES
        ):
            candidates.setdefault(_as_int(row["batch_size"]), []).append(
                _as_float(row["total_seconds_median"])
            )
    if not candidates:
        raise RuntimeError("no correct Metal replay rows available for paired run")
    return min(candidates, key=lambda size: statistics.median(candidates[size]))


def _load_instance(config: PilotConfig, name: str) -> Instance:
    path = config.benchmark_dir / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return parse_schneider(path)


def _order_from_item(item: dict[str, object]) -> tuple[str, ...]:
    value = item.get("route_sequence")
    if not isinstance(value, list):
        raise ValueError("candidate-work item route_sequence must be a list")
    order = tuple(str(name) for name in value)
    if item.get("canonical_route_key") != canonical_route_key(order):
        raise ValueError("candidate-work item canonical route key does not match its sequence")
    return order


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected numeric value, got {type(value).__name__}")
    return float(value)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected integer value, got {type(value).__name__}")
    return int(value)


def _prepare_output(output_dir: Path, *, require_empty: bool = False) -> None:
    if require_empty and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"pilot output already contains evidence; choose a new attempt directory: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _finalize_raw_manifest(output_dir: Path, *, status: object) -> Path:
    manifest_path = output_dir / "raw_manifest.json"
    sidecar_path = output_dir / "raw_manifest.sha256"
    files: dict[str, dict[str, object]] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path in {manifest_path, sidecar_path}:
            continue
        files[str(path.relative_to(output_dir))] = {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    _write_json(
        manifest_path,
        {
            "schema_version": PILOT_SCHEMA_VERSION,
            "run_label": RUN_LABEL,
            "status": status,
            "historical_stage03_claim": "none",
            "stage03_3_readiness_claim": False,
            "files": files,
        },
    )
    sidecar_path.write_text(_sha256(manifest_path) + "\n", encoding="utf-8")
    return manifest_path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON payload is not an object: {path}")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _payload_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the independent Metal batch pilot")
    parser.add_argument("command", choices=("profile", "replay", "paired", "review"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument(
        "--stage02-config",
        type=Path,
        default=Path("configs/stage02_constraint_guided.toml"),
    )
    arguments = parser.parse_args()
    defaults = PilotConfig.defaults()
    root = defaults.root
    config = PilotConfig(
        root=root,
        benchmark_dir=(arguments.benchmark_dir or defaults.benchmark_dir).resolve(),
        stage02_config=(
            arguments.stage02_config
            if arguments.stage02_config.is_absolute()
            else root / arguments.stage02_config
        ).resolve(),
        output_dir=(arguments.output_dir or defaults.output_dir).resolve(),
        summary_dir=(arguments.summary_dir or defaults.summary_dir).resolve(),
    )
    outputs = {
        "profile": run_profile,
        "replay": run_replay,
        "paired": run_paired,
        "review": review_pilot,
    }[arguments.command](config)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
