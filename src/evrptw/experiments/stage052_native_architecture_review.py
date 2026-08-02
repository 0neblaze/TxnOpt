"""Independent replay and reporting for Stage 5.2 native architectures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES
from evrptw.experiments.stage052_native_architectures import (
    AXIS_NAMES,
    MODES,
    PAIRED_INSTANCES,
    SCHEMA_VERSION,
    SEEDS,
    ArchitectureMode,
    expected_axis_count,
    run_labels_for_scope,
)
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.validation import validate_routes

REVIEW_SCHEMA_VERSION = "stage05.2-native-architecture-review-v1"
HISTORICAL_PILOT_ROOT = Path(
    "/mnt/e/Reproducible-EVRPTW-archive/stage05.2/runs/"
    "stage05.2_benchmark_attempt72/generation-0001/d_benchmark/batch0001"
)


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    path: Path
    payload: Mapping[str, object]

    @property
    def mode(self) -> ArchitectureMode:
        return ArchitectureMode(_string(self.payload, "mode"))

    @property
    def key(self) -> tuple[int, str, str, int]:
        return (
            _integer(self.payload, "repeat"),
            _string(self.payload, "axis"),
            _string(self.payload, "instance"),
            _integer(self.payload, "seed"),
        )


def _string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _sequence(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return value


def _verify_signed_json(path: Path) -> Mapping[str, object]:
    data = path.read_bytes()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = sidecar.read_text(encoding="ascii").strip()
    observed = hashlib.sha256(data).hexdigest()
    if expected != observed:
        raise RuntimeError(f"SHA-256 mismatch: {path}")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise RuntimeError(f"signed JSON root is not an object: {path}")
    return payload


def _expected_keys(scope: str) -> set[tuple[int, str, str, int]]:
    if scope == "paired":
        return {
            (repeat, axis, instance, seed)
            for repeat in range(3)
            for axis in AXIS_NAMES
            for instance in PAIRED_INSTANCES
            for seed in SEEDS
        }
    if scope == "pilot":
        return {
            (0, "wall_clock_30", instance, seed)
            for instance in FORMAL_INSTANCES
            for seed in SEEDS
        }
    raise ValueError("scope must be paired or pilot")


def load_records(
    scope: str,
    *,
    attempt: int,
    results_root: Path,
) -> tuple[ReviewRecord, ...]:
    labels = run_labels_for_scope(scope, attempt)
    records: list[ReviewRecord] = []
    common_identity: tuple[str, str, str] | None = None
    for mode in MODES:
        run_dir = results_root / labels[mode.value]
        manifest = _verify_signed_json(run_dir / "run_manifest.json")
        identity = (
            _string(manifest, "revision"),
            _string(manifest, "wheel_sha256"),
            _string(manifest, "native_sha256"),
        )
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise RuntimeError("comparison modes do not share one commit/wheel/native identity")
        paths = sorted((run_dir / "axes").rglob("*.json"))
        mode_records = tuple(
            ReviewRecord(path, _verify_signed_json(path)) for path in paths
        )
        keys = {record.key for record in mode_records}
        if keys != _expected_keys(scope) or len(mode_records) != len(keys):
            raise RuntimeError(f"axis identity set is incomplete or duplicated for {mode.value}")
        if any(record.mode is not mode for record in mode_records):
            raise RuntimeError(f"axis mode identity mismatch for {mode.value}")
        records.extend(mode_records)
    if len(records) != expected_axis_count(scope):
        raise RuntimeError("comparison record count does not match the fixed protocol")
    return tuple(records)


def _replay_record(record: ReviewRecord, benchmark_dir: Path) -> dict[str, object]:
    payload = record.payload
    if _string(payload, "schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported comparison schema: {record.path}")
    if _string(payload, "status") != "completed":
        return {
            "valid": False,
            "reason": f"axis failed: {payload.get('error_type')}: {payload.get('error')}",
        }
    instance_name = _string(payload, "instance")
    instance = parse_schneider(benchmark_dir / f"{instance_name}.txt")
    raw_routes = _sequence(payload, "routes")
    routes: list[list[str]] = []
    for raw_route in raw_routes:
        if not isinstance(raw_route, list) or not all(
            isinstance(name, str) for name in raw_route
        ):
            raise RuntimeError(f"invalid route schema: {record.path}")
        routes.append(list(raw_route))
    report = validate_routes(instance, routes)
    if not report.feasible:
        return {"valid": False, "reason": "unified validator replay failed"}
    objective = SolutionObjective.from_report(instance, report)
    recorded_objective = _sequence(payload, "objective")
    if list(objective.key) != recorded_objective:
        return {"valid": False, "reason": "objective replay mismatch"}
    native = payload.get("native_execution_statistics")
    fallback_count = 0
    if isinstance(native, dict):
        raw_fallback = native.get("fallback_count", 0)
        if isinstance(raw_fallback, int) and not isinstance(raw_fallback, bool):
            fallback_count = raw_fallback
    if fallback_count != 0:
        return {"valid": False, "reason": "native fallback count is non-zero"}
    return {
        "valid": True,
        "objective": objective.key,
        "routes": tuple(tuple(route) for route in routes),
    }


def _common_prefix(left: list[object], right: list[object]) -> int:
    length = 0
    for left_event, right_event in zip(left, right, strict=False):
        if left_event != right_event:
            break
        length += 1
    return length


def _paired(values: Iterable[float]) -> dict[str, float | int | None]:
    items = tuple(values)
    return {
        "count": len(items),
        "median": statistics.median(items) if items else None,
        "minimum": min(items) if items else None,
        "maximum": max(items) if items else None,
    }


def _family(instance: str) -> str:
    lowered = instance.lower()
    if lowered.startswith("rc"):
        return "RC"
    if lowered.startswith("r"):
        return "R"
    return "C"


def _mode_metrics(records: Iterable[ReviewRecord]) -> dict[str, object]:
    values = tuple(record for record in records if record.payload.get("status") == "completed")
    native_queue: list[float] = []
    occupancy: list[int] = []
    for record in values:
        native = record.payload.get("native_execution_statistics")
        if isinstance(native, dict):
            queue = native.get("queue_wait_seconds")
            if isinstance(queue, int | float) and not isinstance(queue, bool):
                native_queue.append(float(queue))
        backend = record.payload.get("backend_metrics")
        if isinstance(backend, dict):
            launches = backend.get("launch_occupancies")
            if isinstance(launches, list):
                occupancy.extend(
                    item
                    for item in launches
                    if isinstance(item, int) and not isinstance(item, bool) and item > 0
                )
    return {
        "completed_axes": len(values),
        "solver_seconds": _paired(_number(record.payload, "solver_seconds") for record in values),
        "effective_iterations": _paired(
            float(_integer(record.payload, "effective_iterations")) for record in values
        ),
        "exact_started_calls": _paired(
            float(_integer(record.payload, "exact_started_calls")) for record in values
        ),
        "rss_bytes": _paired(
            float(_integer(_mapping(record.payload, "topology"), "rss_bytes"))
            for record in values
        ),
        "cache_memory_bytes": _paired(
            float(_integer(record.payload, "cache_memory_bytes")) for record in values
        ),
        "artifact_bytes": _paired(
            float(_integer(record.payload, "artifact_bytes")) for record in values
        ),
        "persistence_seconds": _paired(
            _number(record.payload, "persistence_seconds") for record in values
        ),
        "queue_wait_seconds": _paired(native_queue),
        "transaction_occupancy": _paired(float(value) for value in occupancy),
    }


def _historical_pilot() -> dict[tuple[str, int], dict[str, object]]:
    if not HISTORICAL_PILOT_ROOT.is_dir():
        return {}
    records: dict[tuple[str, int], dict[str, object]] = {}
    for path in HISTORICAL_PILOT_ROOT.glob(
        "*/*/stage05.2_benchmark_attempt72_raw_*.json"
    ):
        payload = json.loads(path.read_bytes())
        if not isinstance(payload, dict):
            continue
        axes = payload.get("axes")
        if not isinstance(axes, dict) or not isinstance(axes.get("wall_clock_30"), dict):
            continue
        records[(str(payload["instance"]), int(payload["seed"]))] = dict(
            axes["wall_clock_30"]
        )
    return records


def review_records(
    records: tuple[ReviewRecord, ...],
    *,
    scope: str,
    benchmark_dir: Path,
) -> dict[str, object]:
    replay = {record.path: _replay_record(record, benchmark_dir) for record in records}
    by_mode = {mode: tuple(record for record in records if record.mode is mode) for mode in MODES}
    by_identity: dict[tuple[int, str, str, int], dict[ArchitectureMode, ReviewRecord]] = (
        defaultdict(dict)
    )
    for record in records:
        by_identity[record.key][record.mode] = record

    differential: dict[str, dict[str, object]] = {}
    for mode in (
        ArchitectureMode.PER_SOLVE_RUNTIME,
        ArchitectureMode.FULL_NATIVE_ALNS,
        ArchitectureMode.HOST_SCHEDULER,
    ):
        comparisons = []
        for key, modes in by_identity.items():
            if key[1] != "fixed_work":
                continue
            baseline = modes[ArchitectureMode.PYTHON_CANDIDATE_CONTROL]
            candidate = modes[mode]
            baseline_trajectory = _sequence(baseline.payload, "trajectory")
            candidate_trajectory = _sequence(candidate.payload, "trajectory")
            comparisons.append(
                {
                    "key": key,
                    "objective_equal": baseline.payload.get("objective")
                    == candidate.payload.get("objective"),
                    "routes_equal": baseline.payload.get("routes")
                    == candidate.payload.get("routes"),
                    "exact_order_and_counts_equal": (
                        baseline.payload.get("exact_started_calls")
                        == candidate.payload.get("exact_started_calls")
                        and baseline.payload.get("exact_completed_calls")
                        == candidate.payload.get("exact_completed_calls")
                    ),
                    "candidate_work_hash_equal": baseline.payload.get("candidate_work_hash")
                    == candidate.payload.get("candidate_work_hash"),
                    "route_result_hash_equal": baseline.payload.get("route_result_hash")
                    == candidate.payload.get("route_result_hash"),
                    "trajectory_equal": baseline_trajectory == candidate_trajectory,
                    "common_prefix": _common_prefix(
                        baseline_trajectory, candidate_trajectory
                    ),
                }
            )
        differential[mode.value] = {
            "comparison_count": len(comparisons),
            "passed": bool(comparisons)
            and all(
                all(
                    bool(comparison[field])
                    for field in (
                        "objective_equal",
                        "routes_equal",
                        "exact_order_and_counts_equal",
                        "candidate_work_hash_equal",
                        "route_result_hash_equal",
                        "trajectory_equal",
                    )
                )
                for comparison in comparisons
            ),
            "comparisons": comparisons,
        }

    speedups: dict[str, dict[str, object]] = {}
    for mode in MODES:
        if mode is ArchitectureMode.CURRENT_STAGE052:
            continue
        versus_current: list[float] = []
        versus_python: list[float] = []
        family_values: dict[str, list[float]] = defaultdict(list)
        wall_objective_not_worse = True
        for key, modes in by_identity.items():
            candidate = modes[mode]
            if candidate.payload.get("status") != "completed":
                continue
            candidate_seconds = _number(candidate.payload, "solver_seconds")
            current_seconds = _number(
                modes[ArchitectureMode.CURRENT_STAGE052].payload, "solver_seconds"
            )
            python_seconds = _number(
                modes[ArchitectureMode.PYTHON_CANDIDATE_CONTROL].payload,
                "solver_seconds",
            )
            versus_current.append(current_seconds / candidate_seconds - 1.0)
            versus_python.append(python_seconds / candidate_seconds - 1.0)
            if key[1] == "fixed_work" and key[2].endswith("_21"):
                family_values[_family(key[2])].append(
                    current_seconds / candidate_seconds - 1.0
                )
            if key[1] == "wall_clock_30":
                candidate_objective = tuple(_sequence(candidate.payload, "objective"))
                python_objective = tuple(
                    _sequence(
                        modes[ArchitectureMode.PYTHON_CANDIDATE_CONTROL].payload,
                        "objective",
                    )
                )
                wall_objective_not_worse = (
                    wall_objective_not_worse and candidate_objective <= python_objective
                )
        family_medians = {
            family: statistics.median(values)
            for family, values in family_values.items()
            if values
        }
        aggregate_100 = [
            current / _number(by_identity[key][mode].payload, "solver_seconds") - 1.0
            for key in by_identity
            if key[1] == "fixed_work"
            and key[2].endswith("_21")
            and by_identity[key][mode].payload.get("status") == "completed"
            and (
                current := _number(
                    by_identity[key][ArchitectureMode.CURRENT_STAGE052].payload,
                    "solver_seconds",
                )
            )
        ]
        performance_qualified = (
            bool(aggregate_100)
            and statistics.median(aggregate_100) >= 0.15
            and all(value >= -0.03 for value in family_medians.values())
            and wall_objective_not_worse
            and (
                mode not in {
                    ArchitectureMode.PER_SOLVE_RUNTIME,
                    ArchitectureMode.FULL_NATIVE_ALNS,
                    ArchitectureMode.HOST_SCHEDULER,
                }
                or bool(differential[mode.value]["passed"])
            )
        )
        speedups[mode.value] = {
            "versus_current": _paired(versus_current),
            "versus_python_candidate_control": _paired(versus_python),
            "aggregate_100_customer": _paired(aggregate_100),
            "family_median_improvement": family_medians,
            "wall_clock_objective_not_worse": wall_objective_not_worse,
            "performance_qualified": performance_qualified,
        }

    historical = _historical_pilot() if scope == "pilot" else {}
    historical_drift: list[dict[str, object]] = []
    if historical:
        for record in by_mode[ArchitectureMode.CURRENT_STAGE052]:
            historical_key = (
                _string(record.payload, "instance"),
                _integer(record.payload, "seed"),
            )
            prior = historical.get(historical_key)
            if prior is None:
                continue
            prior_seconds = _number(prior, "runtime_seconds")
            historical_drift.append(
                {
                    "instance": historical_key[0],
                    "seed": historical_key[1],
                    "solver_ratio_new_over_historical": _number(
                        record.payload, "solver_seconds"
                    )
                    / prior_seconds,
                    "objective_equal": record.payload.get("objective")
                    == prior.get("objective_key"),
                    "validator_historical": prior.get("validator_passed"),
                }
            )

    axis_replay_passed = all(bool(value["valid"]) for value in replay.values())
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "scope": scope,
        "axis_count": len(records),
        "axis_replay_passed": axis_replay_passed,
        "replay_failures": [
            {"path": str(path), **dict(value)}
            for path, value in replay.items()
            if not bool(value["valid"])
        ],
        "mode_metrics": {
            mode.value: _mode_metrics(mode_records)
            for mode, mode_records in by_mode.items()
        },
        "differential_gates": differential,
        "performance": speedups,
        "historical_attempt72": {
            "available": bool(historical),
            "comparison_count": len(historical_drift),
            "drift": historical_drift,
        },
        "cuda_evaluation_condition_met": _scheduler_occupancy_at_least_32(
            by_mode[ArchitectureMode.HOST_SCHEDULER]
        ),
        "formal_started": False,
        "production_default_changed": False,
        "review_status": "COMPARISON_COMPLETE" if axis_replay_passed else "NOT_READY",
    }


def _scheduler_occupancy_at_least_32(records: Iterable[ReviewRecord]) -> bool:
    occupancy: list[int] = []
    for record in records:
        backend = record.payload.get("backend_metrics")
        if not isinstance(backend, dict):
            continue
        values = backend.get("launch_occupancies")
        if isinstance(values, list):
            occupancy.extend(
                value
                for value in values
                if isinstance(value, int) and not isinstance(value, bool)
            )
    return bool(occupancy) and max(occupancy) >= 32


def render_report(review: Mapping[str, object]) -> str:
    metrics = _mapping(review, "mode_metrics")
    performance = _mapping(review, "performance")
    differential = _mapping(review, "differential_gates")
    historical = _mapping(review, "historical_attempt72")
    lines = [
        "# Stage 5.2 五种原生架构同机对比报告",
        "",
        f"- Review status（审查状态）：`{_string(review, 'review_status')}`",
        f"- Raw replay（原始重放）：`{review.get('axis_replay_passed')}`",
        f"- Axis count（轴数）：`{review.get('axis_count')}`",
        "- Formal：未启动；production default（生产默认值）：未改变。",
        "",
        "## 五模式事实表",
        "",
        "| Mode | Solver median s | Effective iterations median | Exact calls median | "
        "Queue wait median s | Occupancy median | RSS median MiB | Artifact median MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        raw = _mapping(metrics, mode.value)
        lines.append(
            "| "
            + mode.value
            + " | "
            + _median_text(raw, "solver_seconds")
            + " | "
            + _median_text(raw, "effective_iterations")
            + " | "
            + _median_text(raw, "exact_started_calls")
            + " | "
            + _median_text(raw, "queue_wait_seconds")
            + " | "
            + _median_text(raw, "transaction_occupancy")
            + " | "
            + _median_scaled_text(raw, "rss_bytes", 1024**2)
            + " | "
            + _median_scaled_text(raw, "artifact_bytes", 1024**2)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Correctness gates（正确性门控）",
            "",
            "| Mode | Fixed-work differential | Performance qualified | "
            "100-customer median improvement | Wall objective not worse |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        perf = performance.get(mode.value)
        perf_map = perf if isinstance(perf, dict) else {}
        diff = differential.get(mode.value)
        diff_map = diff if isinstance(diff, dict) else {}
        aggregate = perf_map.get("aggregate_100_customer")
        aggregate_map = aggregate if isinstance(aggregate, dict) else {}
        lines.append(
            f"| {mode.value} | {diff_map.get('passed', 'baseline')} | "
            f"{perf_map.get('performance_qualified', 'baseline')} | "
            f"{aggregate_map.get('median', 'n/a')} | "
            f"{perf_map.get('wall_clock_objective_not_worse', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## 实现与运维决策矩阵",
            "",
            "下表是实现边界与故障域的事实/工程判断，不替用户选择生产路线。",
            "",
            "| Mode | Python calls | Build complexity | Failure domain | "
            "Recovery difficulty | Maintenance cost |",
            "|---|---|---|---|---|---|",
            "| current_stage052 | 每批 candidate transaction | 中 | solve-local | 低 | 低 |",
            "| python_candidate_control | 每轮 Python control + worker IPC | 低 | "
            "worker pool | 中 | 中 |",
            "| per_solve_runtime | 每 candidate round 一次 C++ | 中 | "
            "solve-local runtime | 中 | 中 |",
            "| full_native_alns | 每 instance/seed 一次 C++ | 高 | 单个 native solve | 高 | 高 |",
            "| host_scheduler | shard 通过 UDS/shared memory | 最高 | "
            "run-wide scheduler | 最高 | 最高 |",
            "",
            "## 当前实现与历史 Pilot",
            "",
            "本轮 `current_stage052` 是所有速度比值和质量差值的主要分母。"
            "Accepted Pilot `attempt72` 仅用于长期漂移核验，不替代同机实测。",
            "",
            f"- Historical available（历史证据可用）：`{historical.get('available')}`",
            f"- Historical comparisons（历史配对数）："
            f"`{historical.get('comparison_count')}`",
            f"- CUDA condition（CUDA 条件）："
            f"`{review.get('cuda_evaluation_condition_met')}`；本轮未自动运行 CUDA。",
            "",
            "## 边界",
            "",
            "未启动 Formal Rerun16，未 push，未清理或覆盖历史 evidence，未切换默认架构。",
        ]
    )
    return "\n".join(lines) + "\n"


def _median_text(payload: Mapping[str, object], key: str) -> str:
    raw = payload.get(key)
    if not isinstance(raw, dict) or raw.get("median") is None:
        return "n/a"
    return f"{float(raw['median']):.6g}"


def _median_scaled_text(payload: Mapping[str, object], key: str, scale: float) -> str:
    raw = payload.get(key)
    if not isinstance(raw, dict) or raw.get("median") is None:
        return "n/a"
    return f"{float(raw['median']) / scale:.3f}"


def write_review(
    review: Mapping[str, object],
    *,
    output_json: Path,
    output_markdown: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(review, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    output_json.write_text(data, encoding="utf-8")
    output_json.with_suffix(output_json.suffix + ".sha256").write_text(
        hashlib.sha256(data.encode("utf-8")).hexdigest() + "\n",
        encoding="ascii",
    )
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(render_report(review), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("paired", "pilot"))
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/schneider"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    arguments = parser.parse_args(argv)
    records = load_records(
        arguments.scope,
        attempt=arguments.attempt,
        results_root=arguments.results_root,
    )
    review = review_records(
        records,
        scope=arguments.scope,
        benchmark_dir=arguments.benchmark_dir,
    )
    write_review(
        review,
        output_json=arguments.output_json,
        output_markdown=arguments.output_markdown,
    )
    print(json.dumps(review, indent=2, sort_keys=True))
    return 0 if review["review_status"] == "COMPARISON_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "HISTORICAL_PILOT_ROOT",
    "REVIEW_SCHEMA_VERSION",
    "ReviewRecord",
    "load_records",
    "render_report",
    "review_records",
    "write_review",
)
