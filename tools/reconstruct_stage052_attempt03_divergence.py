"""Reconstruct the first Attempt03 per-solve semantic divergence."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"nonfinite_float": "nan"}
        return {
            "nonfinite_float": "positive_inf" if value > 0.0 else "negative_inf"
        }
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def _stable_row(value: object) -> object:
    converted = _json_value(value)
    if not isinstance(converted, dict):
        return converted
    volatile = {
        "timestamp_seconds",
        "started_at",
        "completed_at",
        "duration_seconds",
    }
    return {key: item for key, item in converted.items() if key not in volatile}


def _first_divergence(
    left: list[object],
    right: list[object],
) -> dict[str, object] | None:
    for index, (left_row, right_row) in enumerate(zip(left, right, strict=False)):
        if left_row == right_row:
            continue
        left_fields = left_row if isinstance(left_row, dict) else {}
        right_fields = right_row if isinstance(right_row, dict) else {}
        fields = sorted(
            key
            for key in set(left_fields) | set(right_fields)
            if left_fields.get(key) != right_fields.get(key)
        )
        return {
            "index": index,
            "differing_fields": fields,
            "python": left_row,
            "per_solve": right_row,
        }
    if len(left) != len(right):
        return {
            "index": min(len(left), len(right)),
            "differing_fields": ["stream_length"],
            "python": left[min(len(left), len(right))] if len(left) > len(right) else None,
            "per_solve": right[min(len(left), len(right))] if len(right) > len(left) else None,
        }
    return None


def _event_rows(projection: dict[str, object], family: str) -> list[object]:
    value = projection.get(family)
    if not isinstance(value, list):
        raise RuntimeError(f"diagnostic projection lacks event family {family}")
    return value


def _result_projection(result: Any) -> dict[str, object]:
    trace = result.measurement_trace
    return {
        "objective": list(result.objective.key) if result.objective is not None else None,
        "routes": [list(route) for route in result.routes],
        "initial_routes": [list(route) for route in result.initial_routes],
        "initial_customer_sequences": [
            list(route) for route in result.initial_customer_sequences
        ],
        "exact_started_calls": result.exact_started_calls,
        "exact_completed_calls": result.exact_completed_calls,
        "effective_iterations": result.effective_iterations,
        "candidate_work_hash": result.candidate_work_hash,
        "route_result_hash": result.route_result_hash,
        "candidate_transaction_statistics": _json_value(
            result.candidate_transaction_statistics
        ),
        "candidate_control_statistics": _json_value(
            result.candidate_control_statistics
        ),
        "candidate_transaction_events": [
            _stable_row(row) for row in result.candidate_transaction_events
        ],
        "stage04_events": [_stable_row(row) for row in result.stage04_event_log],
        "trace_events": (
            [_stable_row(row) for row in trace.events] if trace is not None else []
        ),
        "screening_decisions": (
            [_stable_row(asdict(row)) for row in trace.screening_decisions]
            if trace is not None
            else []
        ),
        "exact_route_order": (
            [
                _stable_row(
                    {
                        "evaluation_id": row.evaluation_id,
                        "route_key": row.route_key,
                        "lane": row.lane,
                        "iteration": row.iteration,
                        "operator": row.operator,
                        "kind": row.kind,
                        "exact_started": row.exact_started,
                        "exact_completed": row.exact_completed,
                        "feasible": row.feasible,
                        "failure_reason": row.failure_reason,
                        "cache_key_digest": row.cache_key_digest,
                        "route_change_status": row.route_change_status,
                        "status": row.status,
                    }
                )
                for row in trace.route_evaluations
                if row.exact_started or row.exact_completed
            ]
            if trace is not None
            else []
        ),
        "route_dictionary": (
            {key: list(value) for key, value in sorted(trace.route_dictionary.items())}
            if trace is not None
            else {}
        ),
    }


def reconstruct(
    *,
    benchmark_dir: Path,
    output_path: Path,
    revision: str,
    wheel_sha256: str,
    native_sha256: str,
) -> dict[str, object]:
    architectures: Any = importlib.import_module(
        "evrptw.experiments.stage052_native_architectures"
    )
    task = architectures.ArchitectureAxisTask(
        scope="paired",
        repeat=0,
        axis="fixed_work",
        instance_name="c101_21",
        seed=2014,
        benchmark_dir=benchmark_dir,
        output_root=output_path.parent,
        run_labels={},
        scheduler_socket_path=str(output_path.parent / "unused.sock"),
        wheel_sha256=wheel_sha256,
        native_sha256=native_sha256,
        revision=revision,
    )
    python_result, _, _ = architectures._solve_mode(
        architectures.ArchitectureMode.PYTHON_CANDIDATE_CONTROL,
        task,
    )
    per_solve_result, _, _ = architectures._solve_mode(
        architectures.ArchitectureMode.PER_SOLVE_RUNTIME,
        task,
    )
    python_projection = _result_projection(python_result)
    per_solve_projection = _result_projection(per_solve_result)
    families = (
        "candidate_transaction_events",
        "trace_events",
        "screening_decisions",
        "exact_route_order",
        "stage04_events",
    )
    first_by_family = {
        family: _first_divergence(
            _event_rows(python_projection, family),
            _event_rows(per_solve_projection, family),
        )
        for family in families
    }
    payload: dict[str, object] = {
        "schema_version": "stage05.2-attempt03-divergence-diagnostic-v1",
        "source_attempt": 3,
        "source_revision": revision,
        "source_wheel_sha256": wheel_sha256,
        "source_native_sha256": native_sha256,
        "axis": {
            "instance": task.instance_name,
            "seed": task.seed,
            "repeat": task.repeat,
            "termination": task.axis,
            "exact_started_budget": 100,
            "watchdog_seconds": 120.0,
            "candidate_control_workers": 4,
        },
        "first_divergence_by_family": first_by_family,
        "python_candidate_control": python_projection,
        "per_solve_runtime": per_solve_projection,
    }
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if output_path.exists() or output_path.with_suffix(
        output_path.suffix + ".sha256"
    ).exists():
        raise FileExistsError(f"diagnostic output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded)
    output_path.with_suffix(output_path.suffix + ".sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "\n",
        encoding="ascii",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--native-sha256", required=True)
    arguments = parser.parse_args(argv)
    payload = reconstruct(
        benchmark_dir=arguments.benchmark_dir,
        output_path=arguments.output,
        revision=arguments.revision,
        wheel_sha256=arguments.wheel_sha256,
        native_sha256=arguments.native_sha256,
    )
    print(json.dumps(payload["first_divergence_by_family"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
