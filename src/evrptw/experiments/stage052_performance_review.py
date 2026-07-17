"""Independent raw replay for Stage 5.2 performance evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evrptw.artifacts import ArtifactIntegrityError, ArtifactReader
from evrptw.best_known import BEST_KNOWN_VALUES
from evrptw.experiments.stage052_performance import (
    axes_for_scope,
    validate_stage052_run_label,
)
from evrptw.objective import SolutionObjective
from evrptw.parser import parse_schneider
from evrptw.stage052 import (
    PerformanceObservation,
    Stage052Component,
    decide_accelerator,
    evaluate_promotion,
    select_worker_count,
)
from evrptw.validation import validate_routes

STAGE052_REVIEW_SCHEMA_VERSION = "stage05.2-review-v1"
NOT_READY = "NOT_READY"
_CANONICAL_CUSTOMER_COUNTS = {
    record.instance: record.customer_count for record in BEST_KNOWN_VALUES
}
_NEXT_STATUS = {
    Stage052Component.PERF_BASELINE: "READY_FOR_STAGE052_HOT_PATH",
    Stage052Component.HOT_PATH: "READY_FOR_STAGE052_ARTIFACT_STREAMING",
    Stage052Component.ARTIFACT_STREAMING: "READY_FOR_STAGE052_JOB_PARALLEL",
    Stage052Component.JOB_PARALLEL: "READY_FOR_STAGE052_NATIVE_KERNELS",
    Stage052Component.NATIVE_KERNELS: "READY_FOR_STAGE052_ACCELERATOR_DECISION",
    Stage052Component.ACCELERATOR_PILOT: "READY_FOR_STAGE052_BENCHMARK",
    Stage052Component.BENCHMARK: "READY_FOR_STAGE05_3",
}


def validate_per_run_scope(
    rows: Sequence[Mapping[str, object]],
    *,
    instances: Sequence[str],
    seeds: Sequence[int],
    axes: Sequence[str],
) -> tuple[bool, str]:
    expected = {
        (instance, seed, axis)
        for instance in instances
        for seed in seeds
        for axis in axes
    }
    observed: list[tuple[str, int, str]] = []
    failures: list[str] = []
    for row in rows:
        try:
            instance_name = str(row["instance"])
            identity = (
                instance_name,
                _strict_int(row["seed"], "seed"),
                str(row["axis"]),
            )
            customer_count = _strict_int(row["customer_count"], "customer_count")
        except (KeyError, TypeError, ValueError) as error:
            failures.append(str(error))
            continue
        observed.append(identity)
        expected_customer_count = _CANONICAL_CUSTOMER_COUNTS.get(instance_name)
        if expected_customer_count != customer_count:
            failures.append(
                f"customer_count mismatch for {identity}: "
                f"expected={expected_customer_count} observed={customer_count}"
            )
        if not _strict_bool(row.get("validator_passed")):
            failures.append(f"validator failed for {identity}")
        if str(row.get("failure_status", "")):
            failures.append(f"failure status is present for {identity}")
    if len(set(observed)) != len(observed):
        failures.append("duplicate per-run axis identity")
    observed_set = set(observed)
    if observed_set != expected:
        failures.append(
            f"scope identity mismatch: missing={len(expected - observed_set)} "
            f"extra={len(observed_set - expected)}"
        )
    return not failures, "; ".join(failures) if failures else "exact scope passed"


def review_stage052(
    *,
    raw_dir: Path,
    benchmark_dir: Path,
    component: Stage052Component | str,
    scope: str,
    comparison_dirs: Sequence[Path] = (),
) -> dict[str, Path]:
    selected = Stage052Component(component)
    validate_stage052_run_label(raw_dir.name, selected)
    reader = ArtifactReader(raw_dir)
    manifest = reader.manifest
    if manifest.get("evidence_completeness") != "complete":
        raise ArtifactIntegrityError("partial Stage 5.2 evidence cannot be reviewed")
    per_run_ref = _one_artifact(reader, "per_run_results")
    rows = _read_csv(raw_dir / str(per_run_ref["relative_path"]))
    metadata_ref = _one_artifact(reader, "manifest_metadata")
    metadata = reader.read_json(str(metadata_ref["relative_path"]))
    instances = tuple(str(value) for value in metadata["instances"])
    seeds = tuple(_strict_int(value, "seed") for value in metadata["seeds"])
    if scope == "formal":
        scope_passed, scope_detail = _validate_formal_scope(rows, instances, seeds)
    else:
        customer_count = None if scope == "performance" else 5
        axes = tuple(
            axis.name
            for axis in axes_for_scope(scope, customer_count=customer_count)
        )
        scope_passed, scope_detail = validate_per_run_scope(
            rows, instances=instances, seeds=seeds, axes=axes
        )
    replay_passed, replay_detail = _replay_solutions(
        reader, benchmark_dir=benchmark_dir
    )
    gates: dict[str, dict[str, object]] = {
        "exact_scope": {"passed": scope_passed, "detail": scope_detail},
        "replay_consistency": {"passed": replay_passed, "detail": replay_detail},
        "optimization_profile": _optimization_profile_gate(
            selected, metadata.get("optimization_profile")
        ),
        "persistence_attribution": {
            "passed": metadata.get("persistence_attribution")
            == "critical_event_rows",
            "detail": str(metadata.get("persistence_attribution")),
        },
    }
    gates.update(
        _component_gates(
            selected,
            rows,
            comparison_dirs=comparison_dirs,
        )
    )
    passed = all(bool(gate["passed"]) for gate in gates.values())
    status = _NEXT_STATUS[selected] if passed else NOT_READY
    review_dir = raw_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    findings_path = review_dir / "review_findings.csv"
    with findings_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("gate", "passed", "detail"))
        writer.writeheader()
        for gate, result in gates.items():
            writer.writerow({"gate": gate, **result})
    report_path = review_dir / "review_report.md"
    report_path.write_text(
        "\n".join(
            [
                f"# Stage 5.2 Review — {raw_dir.name}",
                "",
                f"**Status: {status}**",
                "",
                *[
                    f"- {name}: {'PASS' if result['passed'] else 'FAIL'} — "
                    f"{result['detail']}"
                    for name, result in gates.items()
                ],
                "",
            ]
        ),
        encoding="utf-8",
    )
    review_manifest_path = review_dir / "review_manifest.json"
    review_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": STAGE052_REVIEW_SCHEMA_VERSION,
                "run_label": raw_dir.name,
                "component": selected.value,
                "scope": scope,
                "status": status,
                "gates": gates,
                "files": {
                    findings_path.name: _sha256(findings_path),
                    report_path.name: _sha256(report_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "review_report": report_path,
        "review_findings": findings_path,
        "review_manifest": review_manifest_path,
    }


def _validate_formal_scope(
    rows: Sequence[Mapping[str, object]],
    instances: Sequence[str],
    seeds: Sequence[int],
) -> tuple[bool, str]:
    expected = {
        (instance, seed, axis.name)
        for instance in instances
        for seed in seeds
        for axis in axes_for_scope(
            "formal", customer_count=_CANONICAL_CUSTOMER_COUNTS[instance]
        )
    }
    observed = {
        (str(row["instance"]), _strict_int(row["seed"], "seed"), str(row["axis"]))
        for row in rows
    }
    if len(rows) != len(observed):
        return False, "duplicate formal identity"
    count_failures = [
        str(row.get("instance"))
        for row in rows
        if _CANONICAL_CUSTOMER_COUNTS.get(str(row.get("instance")))
        != _strict_int(row.get("customer_count"), "customer_count")
    ]
    if count_failures:
        return False, f"customer_count mismatch for {len(count_failures)} formal axes"
    if observed != expected:
        return False, f"formal scope mismatch: expected={len(expected)} observed={len(observed)}"
    if len(expected) != 2040:
        return False, f"formal contract did not produce 2040 axes: {len(expected)}"
    return True, "exact 2040-run formal identity passed"


def _replay_solutions(
    reader: ArtifactReader, *, benchmark_dir: Path
) -> tuple[bool, str]:
    failures: list[str] = []
    for item in reader.manifest.get("artifacts", []):
        if not isinstance(item, Mapping) or item.get("artifact_type") != "solution":
            continue
        payload = reader.read_json(str(item["relative_path"]))
        instance_name = str(payload["instance"])
        instance = parse_schneider(benchmark_dir / f"{instance_name}.txt")
        axes = payload.get("axes")
        if not isinstance(axes, Mapping):
            failures.append(f"{instance_name}: solution axes missing")
            continue
        for axis, raw_axis in axes.items():
            if not isinstance(raw_axis, Mapping):
                failures.append(f"{instance_name}/{axis}: invalid solution axis")
                continue
            routes = raw_axis.get("routes")
            if not isinstance(routes, list):
                failures.append(f"{instance_name}/{axis}: routes missing")
                continue
            report = validate_routes(instance, [list(map(str, route)) for route in routes])
            if not report.feasible:
                failures.append(f"{instance_name}/{axis}: validator failed")
                continue
            replayed = list(SolutionObjective.from_report(instance, report).key)
            recorded = list(raw_axis.get("objective_key", []))
            if replayed != recorded:
                failures.append(f"{instance_name}/{axis}: objective mismatch")
    return (
        not failures,
        "; ".join(failures[:20]) if failures else "validator/objective replay passed",
    )


def _component_gates(
    component: Stage052Component,
    rows: Sequence[Mapping[str, object]],
    *,
    comparison_dirs: Sequence[Path],
) -> dict[str, dict[str, object]]:
    if component is Stage052Component.PERF_BASELINE:
        passed = _axis_semantics_equal(rows, "fixed_work_control", "fixed_work")
        return {
            "instrumentation_semantics": {
                "passed": passed,
                "detail": "fixed-work instrumentation semantic equality"
                if passed
                else "instrumentation changed fixed-work semantics",
            }
        }
    if component in {
        Stage052Component.HOT_PATH,
        Stage052Component.NATIVE_KERNELS,
    }:
        if len(comparison_dirs) != 1:
            return {
                "performance_promotion": {
                    "passed": False,
                    "detail": "one predecessor required",
                }
            }
        previous = _observations(_load_per_run(comparison_dirs[0]), axis="fixed_work")
        candidate = _observations(rows, axis="fixed_work")
        decision = evaluate_promotion(previous, candidate)
        return {
            "performance_promotion": {
                "passed": decision.passed,
                "detail": decision.detail,
            }
        }
    if component is Stage052Component.ARTIFACT_STREAMING:
        persistence_passed = all(
            _strict_float(row["artifact_persistence_seconds"])
            <= 0.30 * _strict_float(row["end_to_end_seconds"])
            for row in rows
        )
        return {
            "persistence_ratio": {
                "passed": persistence_passed,
                "detail": "persistence <=30%" if persistence_passed else "persistence exceeds 30%",
            }
        }
    if component is Stage052Component.JOB_PARALLEL:
        if len(comparison_dirs) != 2:
            return {
                "worker_selection": {
                    "passed": False,
                    "detail": "1/2/4-worker evidence required",
                }
            }
        all_rows: list[Sequence[Mapping[str, object]]] = [
            *(_load_per_run(path) for path in comparison_dirs),
            list(rows),
        ]
        times: dict[int, float] = {}
        rss: dict[int, float] = {}
        for group in all_rows:
            workers = {_strict_int(row["worker_count"], "worker_count") for row in group}
            if len(workers) != 1:
                return {"worker_selection": {"passed": False, "detail": "mixed worker count"}}
            worker = next(iter(workers))
            times[worker] = sum(
                [_strict_float(row["end_to_end_seconds"]) for row in group]
            )
            rss[worker] = max(
                [_strict_float(row["peak_rss_bytes"]) for row in group]
            ) / 2**30
        try:
            selected = select_worker_count(times, rss)
        except ValueError as error:
            return {"worker_selection": {"passed": False, "detail": str(error)}}
        return {"worker_selection": {"passed": True, "detail": f"selected_workers={selected}"}}
    if component is Stage052Component.ACCELERATOR_PILOT:
        occupancy = sorted(_strict_float(row["median_batch_occupancy"]) for row in rows)
        median = occupancy[len(occupancy) // 2] if occupancy else 0.0
        accelerator_decision = decide_accelerator(median_batch_occupancy=median)
        return {
            "accelerator_decision": {
                "passed": True,
                "detail": accelerator_decision.value,
            }
        }
    return {}


def _optimization_profile_gate(
    component: Stage052Component, observed: object
) -> dict[str, object]:
    expected = (
        "none"
        if component is Stage052Component.PERF_BASELINE
        else "python"
        if component
        in {
            Stage052Component.HOT_PATH,
            Stage052Component.ARTIFACT_STREAMING,
            Stage052Component.JOB_PARALLEL,
        }
        else "native"
    )
    passed = observed == expected
    return {
        "passed": passed,
        "detail": f"expected={expected} observed={observed}",
    }


def _axis_semantics_equal(
    rows: Sequence[Mapping[str, object]], left: str, right: str
) -> bool:
    by_identity = {
        (str(row["instance"]), _strict_int(row["seed"], "seed"), str(row["axis"])): str(
            row["semantic_digest"]
        )
        for row in rows
    }
    identities = {(key[0], key[1]) for key in by_identity if key[2] == left}
    return bool(identities) and all(
        by_identity.get((instance, seed, left))
        == by_identity.get((instance, seed, right))
        for instance, seed in identities
    )


def _observations(
    rows: Sequence[Mapping[str, object]], *, axis: str
) -> list[PerformanceObservation]:
    return [
        PerformanceObservation(
            instance=str(row["instance"]),
            seed=_strict_int(row["seed"], "seed"),
            customer_count=_strict_int(row["customer_count"], "customer_count"),
            end_to_end_seconds=_strict_float(row["end_to_end_seconds"]),
            semantic_digest=str(row["semantic_digest"]),
        )
        for row in rows
        if row.get("axis") == axis
    ]


def _load_per_run(raw_dir: Path) -> list[dict[str, str]]:
    reader = ArtifactReader(raw_dir)
    item = _one_artifact(reader, "per_run_results")
    return _read_csv(raw_dir / str(item["relative_path"]))


def _one_artifact(reader: ArtifactReader, artifact_type: str) -> Mapping[str, Any]:
    matches = [
        item
        for item in reader.manifest.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("artifact_type") == artifact_type
    ]
    if len(matches) != 1:
        raise ArtifactIntegrityError(f"expected one {artifact_type} artifact")
    return matches[0]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strict_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise TypeError(f"{field} must be an integer")


def _strict_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("numeric value cannot be boolean")
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError("numeric value is invalid")


def _strict_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Review Stage 5.2 raw evidence")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/schneider"))
    parser.add_argument(
        "--component",
        choices=tuple(component.value for component in Stage052Component),
        required=True,
    )
    parser.add_argument("--scope", choices=("performance", "pilot", "formal"), required=True)
    parser.add_argument("--comparison-dir", type=Path, action="append", default=[])
    arguments = parser.parse_args()
    outputs = review_stage052(
        raw_dir=arguments.raw_dir,
        benchmark_dir=arguments.benchmark_dir,
        component=arguments.component,
        scope=arguments.scope,
        comparison_dirs=arguments.comparison_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
