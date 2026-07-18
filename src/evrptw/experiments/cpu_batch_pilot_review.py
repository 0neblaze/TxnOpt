"""Independent replay review for the CPU batch pilot evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evrptw.measurement import canonical_route_key
from evrptw.models import NodeType
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.repository import repository_root
from evrptw.validation import validate_routes

SCHEMA_VERSION = "cpu-batch-pilot-v1"
RUN_LABEL = "cpu_batch_pilot_attempt01"
PILOT_INSTANCES = ("c101C5", "c101_21", "r101_21", "rc101_21")
PILOT_SEEDS = (2014, 2015, 2016)
HUNDRED_CUSTOMER_INSTANCES = frozenset({"c101_21", "r101_21", "rc101_21"})
FIXED_ITERATIONS = 40
WARMUPS = 3
MEASUREMENTS = 5


@dataclass(frozen=True, slots=True)
class ReviewConfig:
    root: Path
    benchmark_dir: Path
    input_dir: Path
    summary_dir: Path

    @classmethod
    def defaults(cls, root: Path | None = None) -> ReviewConfig:
        resolved = root or repository_root()
        return cls(
            root=resolved,
            benchmark_dir=resolved / "data" / "schneider",
            input_dir=resolved / "results" / RUN_LABEL,
            summary_dir=resolved / "experiments" / "summaries",
        )


def evaluate_paired_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    """Independently apply the complete-scope correctness and performance gate."""

    expected_keys = {
        (instance, seed) for instance in PILOT_INSTANCES for seed in PILOT_SEEDS
    }
    rows_by_key: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        rows_by_key.setdefault(
            (str(row.get("instance")), _as_int(row.get("seed"))),
            [],
        ).append(row)
    scope_complete = set(rows_by_key) == expected_keys and all(
        len(group) == 1 for group in rows_by_key.values()
    )
    all_valid = scope_complete and all(bool(row.get("valid")) for row in rows)
    protocol_complete = scope_complete and all(
        _as_int(row.get("fixed_iterations")) == FIXED_ITERATIONS for row in rows
    )
    family_savings: dict[str, float] = {}
    for instance in sorted(HUNDRED_CUSTOMER_INSTANCES):
        values = [
            _as_float(row["saving_fraction"])
            for row in rows
            if row.get("instance") == instance and bool(row.get("valid"))
        ]
        if values:
            family_savings[instance] = statistics.median(values)
    median_family_saving = (
        statistics.median(family_savings.values())
        if len(family_savings) == len(HUNDRED_CUSTOMER_INSTANCES)
        else float("-inf")
    )
    positive_families = sum(value > 0.0 for value in family_savings.values())
    adopt = (
        all_valid
        and protocol_complete
        and median_family_saving >= 0.10
        and positive_families >= 2
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ADOPT_CPU_BATCH_DEFAULT" if adopt else "KEEP_CPU_SCALAR_DEFAULT",
        "all_pairs_valid": all_valid,
        "protocol_complete": protocol_complete,
        "family_median_saving_fraction": family_savings,
        "median_family_saving_fraction": median_family_saving,
        "positive_families": positive_families,
        "criterion": (
            "all pairs valid; median of three 100-customer family medians >= 0.10; "
            "at least two families improve"
        ),
    }


def run_review(config: ReviewConfig) -> dict[str, Path]:
    """Verify checksums, replay validation/objectives, and publish summaries."""

    _verify_raw_manifest(config.input_dir)
    work = _review_candidate_work(config)
    _review_replay_protocol(config.input_dir)
    raw_rows = _object_rows(_read_json(config.input_dir / "paired_results.json"), "rows")
    reviewed_rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for raw_row in raw_rows:
        instance_name = str(raw_row.get("instance"))
        seed = _as_int(raw_row.get("seed"))
        key = (instance_name, seed)
        if key in seen:
            raise RuntimeError(f"duplicate paired row: {instance_name}/{seed}")
        seen.add(key)
        if key not in work:
            raise RuntimeError(f"paired row has no candidate manifest: {instance_name}/{seed}")
        instance = parse_schneider(config.benchmark_dir / f"{instance_name}.txt")
        scalar = _object(raw_row.get("scalar"), "scalar summary")
        batch = _object(raw_row.get("batch"), "batch summary")
        scalar_valid = _review_backend_summary(instance, scalar, work[key])
        batch_valid = _review_backend_summary(instance, batch, work[key])
        paired_equal = all(
            (
                scalar.get("candidate_work_hash") == batch.get("candidate_work_hash"),
                scalar.get("exact_calls") == batch.get("exact_calls"),
                scalar.get("route_result_hash") == batch.get("route_result_hash"),
                scalar.get("objective_key") == batch.get("objective_key"),
                scalar.get("effective_iterations") == batch.get("effective_iterations"),
            )
        )
        valid = scalar_valid and batch_valid and paired_equal
        if bool(raw_row.get("valid")) != valid:
            raise RuntimeError(f"raw paired validity mismatch: {instance_name}/{seed}")
        scalar_seconds = _as_float(raw_row.get("scalar_runtime_seconds"))
        batch_seconds = _as_float(raw_row.get("batch_runtime_seconds"))
        if not math.isclose(
            scalar_seconds,
            _as_float(scalar.get("runtime_seconds")),
            abs_tol=1e-12,
        ) or not math.isclose(
            batch_seconds,
            _as_float(batch.get("runtime_seconds")),
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"paired/backend runtime mismatch: {instance_name}/{seed}")
        saving = (scalar_seconds - batch_seconds) / scalar_seconds
        speedup = scalar_seconds / batch_seconds
        if not math.isclose(saving, _as_float(raw_row.get("saving_fraction")), abs_tol=1e-12):
            raise RuntimeError(f"raw saving mismatch: {instance_name}/{seed}")
        reviewed_rows.append(
            {
                "instance": instance_name,
                "seed": seed,
                "valid": valid,
                "fixed_iterations": FIXED_ITERATIONS,
                "scalar_runtime_seconds": scalar_seconds,
                "batch_runtime_seconds": batch_seconds,
                "saving_fraction": saving,
                "speedup": speedup,
            }
        )
    decision = evaluate_paired_gate(reviewed_rows)
    if decision != _read_json(config.input_dir / "paired_gate.json"):
        raise RuntimeError("independent review recomputed a different paired gate")
    if not bool(decision["protocol_complete"]):
        raise RuntimeError("incomplete CPU pilot protocol cannot publish summaries")
    summary_path = config.summary_dir / f"{RUN_LABEL}_per_run.csv"
    _write_csv(summary_path, reviewed_rows)
    report_path = config.summary_dir / f"{RUN_LABEL}_review.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# CPU batch pilot independent review\n\n"
        f"- Status: `{decision['status']}`\n"
        f"- All pairs valid: `{decision['all_pairs_valid']}`\n"
        f"- Protocol complete: `{decision['protocol_complete']}`\n"
        f"- Median family saving: "
        f"`{_as_float(decision['median_family_saving_fraction']):.6f}`\n"
        f"- Positive 100-customer families: `{decision['positive_families']}`\n"
        "- Manifest, candidate work, replay protocol, validator, objective, final-route "
        "hash, exact-call count and paired equality: `PASS`\n"
        "- Stage 3.3 claim: `none`\n",
        encoding="utf-8",
    )
    return {"per_run_summary": summary_path, "review_report": report_path}


def _review_candidate_work(
    config: ReviewConfig,
) -> dict[tuple[str, int], tuple[str, int]]:
    reviewed: dict[tuple[str, int], tuple[str, int]] = {}
    for instance in PILOT_INSTANCES:
        parsed = parse_schneider(config.benchmark_dir / f"{instance}.txt")
        for seed in PILOT_SEEDS:
            payload = _read_json(
                config.input_dir / "candidate_work" / f"{instance}_{seed}.json"
            )
            requests = payload.get("requests")
            if not isinstance(requests, list) or not requests:
                raise ValueError(f"invalid candidate requests: {instance}/{seed}")
            if payload.get("instance") != instance or _as_int(payload.get("seed")) != seed:
                raise RuntimeError(f"candidate-work identity mismatch: {instance}/{seed}")
            for request_order, request_value in enumerate(requests):
                request = _object(request_value, "candidate-work request")
                route_value = request.get("route_sequence")
                if not isinstance(route_value, list):
                    raise ValueError("candidate-work route_sequence must be a list")
                route = tuple(str(name) for name in route_value)
                if (
                    _as_int(request.get("request_order")) != request_order
                    or _as_int(request.get("seed")) != seed
                    or request.get("canonical_route_key") != canonical_route_key(route)
                    or any(
                        name not in parsed.by_name
                        or parsed.by_name[name].kind is not NodeType.CUSTOMER
                        for name in route
                    )
                ):
                    raise RuntimeError(
                        f"candidate-work request semantics mismatch: "
                        f"{instance}/{seed}/{request_order}"
                    )
            expected_hash = _payload_hash({"seed": seed, "requests": requests})
            if payload.get("candidate_work_hash") != expected_hash:
                raise RuntimeError(f"candidate-work hash mismatch: {instance}/{seed}")
            reviewed[(instance, seed)] = (expected_hash, len(requests))
    return reviewed


def _review_replay_protocol(input_dir: Path) -> None:
    rows = _object_rows(_read_json(input_dir / "replay_results.json"), "rows")
    expected = {
        (instance, seed, backend)
        for instance in PILOT_INSTANCES
        for seed in PILOT_SEEDS
        for backend in ("cpu_scalar", "cpu_batch")
    }
    actual: set[tuple[str, int, str]] = set()
    savings: list[float] = []
    times: dict[tuple[str, int, str], float] = {}
    for row in rows:
        key = (
            str(row.get("instance")),
            _as_int(row.get("seed")),
            str(row.get("backend")),
        )
        if key in actual:
            raise RuntimeError(f"duplicate replay row: {key}")
        actual.add(key)
        if (
            not bool(row.get("correctness"))
            or _as_int(row.get("warmups")) != WARMUPS
            or _as_int(row.get("measurements")) != MEASUREMENTS
        ):
            raise RuntimeError(f"invalid formal replay row: {key}")
        samples = row.get("total_seconds_samples")
        if not isinstance(samples, list) or len(samples) != MEASUREMENTS:
            raise RuntimeError(f"invalid replay sample count: {key}")
        times[key] = statistics.median(_as_float(value) for value in samples)
    if actual != expected:
        raise RuntimeError("formal replay scope mismatch")
    for instance in HUNDRED_CUSTOMER_INSTANCES:
        for seed in PILOT_SEEDS:
            scalar = times[(instance, seed, "cpu_scalar")]
            batch = times[(instance, seed, "cpu_batch")]
            savings.append((scalar - batch) / scalar)
    gate = _read_json(input_dir / "replay_gate.json")
    if (
        gate.get("status") != "PASS_REPLAY"
        or gate.get("protocol_complete") is not True
        or gate.get("correctness") is not True
        or not math.isclose(
            _as_float(gate.get("median_100_customer_saving_fraction")),
            statistics.median(savings),
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("formal replay gate did not independently reconcile")


def _review_backend_summary(
    instance: Any,
    summary: dict[str, object],
    work: tuple[str, int],
) -> bool:
    routes_value = summary.get("routes")
    customers_value = summary.get("customer_sequences")
    if not isinstance(routes_value, list) or not isinstance(customers_value, list):
        return False
    routes = [[str(node) for node in route] for route in routes_value if isinstance(route, list)]
    if len(routes) != len(routes_value):
        return False
    report = validate_routes(instance, routes)
    if not report.feasible:
        return False
    objective = SolutionObjective.from_report(instance, report)
    objective_key = summary.get("objective_key")
    if not isinstance(objective_key, list) or tuple(objective_key) != objective.key:
        return False
    route_hash = _payload_hash(
        {"routes": routes_value, "customer_sequences": customers_value}
    )
    return all(
        (
            summary.get("candidate_work_hash") == work[0],
            _as_int(summary.get("exact_calls")) == work[1],
            summary.get("route_result_hash") == route_hash,
            _as_int(summary.get("effective_iterations")) == FIXED_ITERATIONS,
            summary.get("watchdog_triggered") is False,
            summary.get("validator_feasible") is True,
            summary.get("trace_reconciliation_status") == "pass",
        )
    )


def _verify_raw_manifest(input_dir: Path) -> None:
    manifest_path = input_dir / "raw_manifest.json"
    sidecar_path = input_dir / "raw_manifest.sha256"
    if sidecar_path.read_text(encoding="utf-8").strip() != _sha256(manifest_path):
        raise RuntimeError("CPU pilot raw manifest sidecar mismatch")
    files = _read_json(manifest_path).get("files")
    if not isinstance(files, dict):
        raise ValueError("CPU pilot raw manifest files must be an object")
    for relative, metadata in files.items():
        if not isinstance(relative, str) or not isinstance(metadata, dict):
            raise ValueError("invalid raw manifest file record")
        path = input_dir / relative
        if not path.is_file() or _sha256(path) != str(metadata.get("sha256")):
            raise RuntimeError(f"CPU pilot raw checksum mismatch: {relative}")


def _object_rows(payload: dict[str, Any], key: str) -> list[dict[str, object]]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{key} must be a list of objects")
    return [dict(row) for row in value]


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _payload_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON payload is not an object: {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected numeric value, got {type(value).__name__}")
    return float(value)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"expected integer value, got {type(value).__name__}")
    return int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Review CPU batch pilot evidence")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--benchmark-dir", type=Path)
    arguments = parser.parse_args()
    defaults = ReviewConfig.defaults()
    config = ReviewConfig(
        root=defaults.root,
        benchmark_dir=(arguments.benchmark_dir or defaults.benchmark_dir).resolve(),
        input_dir=(arguments.input_dir or defaults.input_dir).resolve(),
        summary_dir=(arguments.summary_dir or defaults.summary_dir).resolve(),
    )
    run_review(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
