"""Independent replay and reporting for Stage 5.2 native architectures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
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

REVIEW_SCHEMA_VERSION = "stage05.2-native-architecture-review-v4"
HISTORICAL_PILOT_ROOT = Path(
    "/mnt/e/Reproducible-EVRPTW-archive/stage05.2/runs/"
    "stage05.2_benchmark_attempt72/generation-0001/d_benchmark"
)
HISTORICAL_PILOT_RUN_LABEL = "stage05.2_benchmark_attempt72"
HISTORICAL_PILOT_REVISION = "a5cf00f7580fc2632179495a739a110786ace87d"
HISTORICAL_PILOT_RAW_MANIFEST_SHA256 = (
    "5aa8b773c39ea3e0a8cd31434756f4219c3ec51f430bdcae1e61107fdd3f38e4"
)
HISTORICAL_PILOT_REVIEW_MANIFEST_SHA256 = (
    "24de9cc93ad99f7d607617e11e8c9bee7fda224a421da1f97dd3af1bd6277727"
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
        expected_run_label = labels[mode.value]
        for record in mode_records:
            payload_identity = (
                _string(record.payload, "revision"),
                _string(record.payload, "wheel_sha256"),
                _string(record.payload, "native_sha256"),
            )
            if payload_identity != identity:
                raise RuntimeError(
                    f"axis identity does not match its run manifest: {record.path}"
                )
            if _string(record.payload, "run_label") != expected_run_label:
                raise RuntimeError(f"axis run label mismatch: {record.path}")
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
    fallback_count = payload.get("fallback_count")
    if isinstance(fallback_count, bool) or not isinstance(fallback_count, int):
        return {"valid": False, "reason": "fallback evidence is missing or malformed"}
    if fallback_count != 0:
        return {"valid": False, "reason": "native fallback count is non-zero"}
    semantic_completeness = payload.get("semantic_completeness")
    semantics_complete = isinstance(semantic_completeness, dict) and all(
        semantic_completeness.get(field) is True
        for field in ("candidate_control", "stage04", "measurement_trace")
    )
    measurement = payload.get("measurement_evidence")
    if not isinstance(measurement, dict):
        return {"valid": False, "reason": "measurement evidence is missing"}
    recorded_measurement_hash = measurement.get("sha256")
    if not isinstance(recorded_measurement_hash, str):
        return {"valid": False, "reason": "measurement evidence hash is missing"}
    hashed_measurement = dict(measurement)
    del hashed_measurement["sha256"]
    recomputed_measurement_hash = hashlib.sha256(
        json.dumps(
            hashed_measurement,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if recorded_measurement_hash != recomputed_measurement_hash:
        return {"valid": False, "reason": "measurement evidence hash mismatch"}
    for field in (
        "operator_statistics",
        "stage04_statistics",
        "stage04_events",
        "candidate_transaction_events",
    ):
        if field not in payload:
            return {"valid": False, "reason": f"{field} evidence is missing"}
    return {
        "valid": True,
        "semantics_complete": semantics_complete,
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


def _vehicle_count(payload: Mapping[str, object]) -> float:
    objective = _sequence(payload, "objective")
    if not objective:
        raise ValueError("objective must contain vehicle count")
    value = objective[0]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("objective vehicle count must be numeric")
    return float(value)


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
        "vehicle_count": _paired(
            _vehicle_count(record.payload) for record in values
        ),
        "effective_iterations_per_second": _paired(
            _number(_mapping(record.payload, "throughput"), "effective_iterations_per_second")
            for record in values
        ),
        "candidate_transactions_per_second": _paired(
            _number(_mapping(record.payload, "throughput"), "candidate_transactions_per_second")
            for record in values
        ),
        "screened_routes_per_second": _paired(
            _number(_mapping(record.payload, "throughput"), "screened_routes_per_second")
            for record in values
        ),
        "exact_started_per_second": _paired(
            _number(_mapping(record.payload, "throughput"), "exact_started_per_second")
            for record in values
        ),
        "cpu_utilization_percent_of_one_core": _paired(
            _number(
                _mapping(record.payload, "topology"),
                "cpu_utilization_percent_of_one_core",
            )
            for record in values
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
        "exact_backend_batch_occupancy": _paired(float(value) for value in occupancy),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        sidecar = path.with_suffix(".sha256")
    expected = sidecar.read_text(encoding="ascii").strip().split()[0]
    actual = _sha256(path)
    if expected != actual:
        raise ValueError(f"checksum mismatch: {path}")
    return actual


def _raw_axis_inventory(records: Iterable[ReviewRecord]) -> dict[str, object]:
    values = tuple(records)

    def inventory(selected: Iterable[ReviewRecord]) -> dict[str, object]:
        entries: list[dict[str, object]] = []
        json_bytes = 0
        sidecar_bytes = 0
        for record in sorted(
            selected,
            key=lambda item: (_string(item.payload, "run_label"), item.key),
        ):
            sidecar = record.path.with_suffix(record.path.suffix + ".sha256")
            raw_data = record.path.read_bytes()
            sidecar_data = sidecar.read_bytes()
            json_bytes += len(raw_data)
            sidecar_bytes += len(sidecar_data)
            entries.append(
                {
                    "logical_axis": [
                        _string(record.payload, "run_label"),
                        *record.key,
                    ],
                    "json_sha256": hashlib.sha256(raw_data).hexdigest(),
                    "sidecar_sha256": hashlib.sha256(sidecar_data).hexdigest(),
                }
            )
        canonical = json.dumps(
            entries,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return {
            "axis_count": len(entries),
            "json_bytes": json_bytes,
            "sidecar_bytes": sidecar_bytes,
            "tree_sha256": hashlib.sha256(
                b"stage05.2-native-axis-inventory-v1\0" + canonical
            ).hexdigest(),
        }

    aggregate = inventory(values)
    aggregate["algorithm"] = (
        "sha256(stage05.2-native-axis-inventory-v1\\0 + canonical JSON of "
        "logical axis, JSON SHA-256, and sidecar SHA-256)"
    )
    aggregate["by_mode"] = {
        mode.value: inventory(record for record in values if record.mode is mode)
        for mode in MODES
    }
    return aggregate


def _historical_pilot(
    root: Path = HISTORICAL_PILOT_ROOT,
) -> tuple[dict[tuple[str, int], dict[str, object]], dict[str, object]]:
    if not root.is_dir():
        return {}, {
            "available": False,
            "identity_verified": False,
            "error": "historical archive is not available",
        }
    active = root.parent / "wsl_active"
    try:
        campaign_path = active / "campaign_manifest.json"
        raw_manifest_path = active / "control" / f"{HISTORICAL_PILOT_RUN_LABEL}_manifest.json"
        run_metadata_path = (
            active / "control" / f"{HISTORICAL_PILOT_RUN_LABEL}_run_metadata.json"
        )
        review_manifest_path = active / "review" / "review_manifest.json"
        review_execution_path = active / "review" / "review_execution.json"
        campaign_sha256 = _verify_sidecar(campaign_path)
        raw_manifest_sha256 = _verify_sidecar(raw_manifest_path)
        if raw_manifest_sha256 != HISTORICAL_PILOT_RAW_MANIFEST_SHA256:
            raise ValueError("historical raw manifest is not the accepted identity")
        review_manifest_sha256 = _sha256(review_manifest_path)
        if review_manifest_sha256 != HISTORICAL_PILOT_REVIEW_MANIFEST_SHA256:
            raise ValueError("historical review manifest is not the accepted identity")

        campaign = json.loads(campaign_path.read_bytes())
        raw_manifest = json.loads(raw_manifest_path.read_bytes())
        run_metadata = json.loads(run_metadata_path.read_bytes())
        review_manifest = json.loads(review_manifest_path.read_bytes())
        review_execution = json.loads(review_execution_path.read_bytes())
        if not all(
            isinstance(value, dict)
            for value in (
                campaign,
                raw_manifest,
                run_metadata,
                review_manifest,
                review_execution,
            )
        ):
            raise ValueError("historical identity documents must be JSON objects")
        if (
            campaign.get("run_label") != HISTORICAL_PILOT_RUN_LABEL
            or campaign.get("scope") != "pilot"
            or campaign.get("status") != "complete"
            or campaign.get("axis_count") != 36
            or campaign.get("shard_count") != 36
            or raw_manifest.get("run_label") != HISTORICAL_PILOT_RUN_LABEL
            or raw_manifest.get("status") != "complete"
            or run_metadata.get("repository_revision") != HISTORICAL_PILOT_REVISION
            or review_manifest.get("run_label") != HISTORICAL_PILOT_RUN_LABEL
            or review_manifest.get("scope") != "pilot"
            or review_manifest.get("status")
            != "READY_FOR_STAGE052_FORMAL_BENCHMARK"
            or review_manifest.get("raw_campaign_manifest_sha256")
            != campaign_sha256
            or review_manifest.get("raw_manifest_sha256") != raw_manifest_sha256
            or review_execution.get("review_manifest_sha256")
            != review_manifest_sha256
            or review_execution.get("raw_manifest_sha256_before")
            != raw_manifest_sha256
            or review_execution.get("raw_manifest_sha256_after")
            != raw_manifest_sha256
            or review_execution.get("producer_repository_revision")
            != HISTORICAL_PILOT_REVISION
        ):
            raise ValueError("historical accepted-Pilot identity fields do not match")
        gates = review_manifest.get("gates")
        if not isinstance(gates, dict) or not gates or any(
            not isinstance(gate, dict) or gate.get("passed") is not True
            for gate in gates.values()
        ):
            raise ValueError("historical accepted review gates do not all pass")

        batches = campaign.get("batches")
        if not isinstance(batches, list) or len(batches) != 3:
            raise ValueError("historical campaign must contain exactly three batches")
        records: dict[tuple[str, int], dict[str, object]] = {}
        for batch in batches:
            if not isinstance(batch, dict):
                raise ValueError("historical batch entry must be an object")
            batch_id = _string(batch, "batch_id")
            batch_root = root / batch_id
            batch_manifest_path = batch_root / "batch_manifest.json"
            _verify_sidecar(batch_manifest_path)
            batch_manifest = json.loads(batch_manifest_path.read_bytes())
            if batch_manifest != batch or batch.get("status") != "archived":
                raise ValueError(f"historical batch identity mismatch: {batch_id}")
            shard_hashes = batch.get("shard_manifest_sha256_by_id")
            if not isinstance(shard_hashes, dict) or len(shard_hashes) != 12:
                raise ValueError(f"historical batch shard map is invalid: {batch_id}")
            shard_paths = tuple(batch_root.glob("*/*/*_shard_manifest_*.json"))
            if len(shard_paths) != 12:
                raise ValueError(f"historical batch must contain 12 shard manifests: {batch_id}")
            for shard_path in shard_paths:
                shard = json.loads(shard_path.read_bytes())
                if not isinstance(shard, dict):
                    raise ValueError("historical shard manifest must be an object")
                ordinal = _integer(shard, "shard_ordinal") + 1
                shard_id = f"shard{ordinal:04d}"
                if shard_hashes.get(shard_id) != _sha256(shard_path):
                    raise ValueError(f"historical shard manifest mismatch: {shard_id}")
                if (
                    shard.get("run_label") != HISTORICAL_PILOT_RUN_LABEL
                    or shard.get("evidence_completeness") != "complete"
                ):
                    raise ValueError(f"historical shard is not complete: {shard_id}")
                artifacts = shard.get("artifacts")
                if not isinstance(artifacts, list):
                    raise ValueError("historical shard artifacts must be an array")
                raw_artifacts = [
                    artifact
                    for artifact in artifacts
                    if isinstance(artifact, dict) and artifact.get("artifact_type") == "raw"
                ]
                if len(raw_artifacts) != 1:
                    raise ValueError(f"historical shard raw identity is ambiguous: {shard_id}")
                raw_artifact = raw_artifacts[0]
                raw_path = batch_root / _string(raw_artifact, "relative_path")
                if raw_artifact.get("checksum") != _sha256(raw_path):
                    raise ValueError(f"historical raw checksum mismatch: {shard_id}")
                payload = json.loads(raw_path.read_bytes())
                if not isinstance(payload, dict):
                    raise ValueError("historical raw payload must be an object")
                axes = payload.get("axes")
                if not isinstance(axes, dict) or not isinstance(
                    axes.get("wall_clock_30"), dict
                ):
                    raise ValueError("historical raw payload lacks wall_clock_30")
                key = (_string(payload, "instance"), _integer(payload, "seed"))
                if key in records:
                    raise ValueError(f"duplicate historical instance/seed: {key}")
                records[key] = dict(axes["wall_clock_30"])
        expected_keys = {(name, seed) for name in FORMAL_INSTANCES for seed in SEEDS}
        if set(records) != expected_keys:
            raise ValueError("historical accepted-Pilot geometry does not match 12x3")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return {}, {
            "available": True,
            "identity_verified": False,
            "error": str(exc),
        }
    return records, {
        "available": True,
        "identity_verified": True,
        "error": None,
        "campaign_manifest_sha256": campaign_sha256,
        "raw_manifest_sha256": raw_manifest_sha256,
        "review_manifest_sha256": review_manifest_sha256,
        "producer_revision": HISTORICAL_PILOT_REVISION,
    }


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
            if (
                baseline.payload.get("status") != "completed"
                or candidate.payload.get("status") != "completed"
            ):
                comparisons.append(
                    {
                        "key": key,
                        "baseline_status": baseline.payload.get("status"),
                        "candidate_status": candidate.payload.get("status"),
                        "objective_equal": False,
                        "routes_equal": False,
                        "exact_order_equal": False,
                        "exact_counts_equal": False,
                        "candidate_work_hash_equal": False,
                        "route_result_hash_equal": False,
                        "trajectory_equal": False,
                        "operator_statistics_equal": False,
                        "stage04_state_equal": False,
                        "candidate_transaction_events_equal": False,
                        "cache_lifecycle_equal": False,
                        "deadline_boundaries_equal": False,
                        "measurement_transaction_hash_equal": False,
                        "common_prefix": 0,
                    }
                )
                continue
            baseline_trajectory = _mapping(baseline.payload, "trajectory")
            candidate_trajectory = _mapping(candidate.payload, "trajectory")
            baseline_measurement = _mapping(baseline.payload, "measurement_evidence")
            candidate_measurement = _mapping(candidate.payload, "measurement_evidence")
            comparisons.append(
                {
                    "key": key,
                    "objective_equal": baseline.payload.get("objective")
                    == candidate.payload.get("objective"),
                    "routes_equal": baseline.payload.get("routes")
                    == candidate.payload.get("routes"),
                    "exact_order_equal": (
                        baseline_measurement.get("exact_route_order")
                        == candidate_measurement.get("exact_route_order")
                    ),
                    "exact_counts_equal": (
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
                    "operator_statistics_equal": baseline.payload.get(
                        "operator_statistics"
                    )
                    == candidate.payload.get("operator_statistics"),
                    "stage04_state_equal": (
                        baseline.payload.get("stage04_statistics")
                        == candidate.payload.get("stage04_statistics")
                        and baseline.payload.get("stage04_events")
                        == candidate.payload.get("stage04_events")
                    ),
                    "candidate_transaction_events_equal": baseline.payload.get(
                        "candidate_transaction_events"
                    )
                    == candidate.payload.get("candidate_transaction_events"),
                    "cache_lifecycle_equal": baseline_measurement.get(
                        "cache_lifecycle"
                    )
                    == candidate_measurement.get("cache_lifecycle"),
                    "deadline_boundaries_equal": baseline_measurement.get(
                        "deadline_boundaries"
                    )
                    == candidate_measurement.get("deadline_boundaries"),
                    "measurement_transaction_hash_equal": baseline_measurement.get(
                        "sha256"
                    )
                    == candidate_measurement.get("sha256"),
                    "common_prefix": _common_prefix(
                        [baseline_trajectory], [candidate_trajectory]
                    )
                    * _integer(baseline_trajectory, "count"),
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
                        "exact_order_equal",
                        "exact_counts_equal",
                        "candidate_work_hash_equal",
                        "route_result_hash_equal",
                        "trajectory_equal",
                        "operator_statistics_equal",
                        "stage04_state_equal",
                        "candidate_transaction_events_equal",
                        "cache_lifecycle_equal",
                        "deadline_boundaries_equal",
                        "measurement_transaction_hash_equal",
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
            current_record = modes[ArchitectureMode.CURRENT_STAGE052]
            python_record = modes[ArchitectureMode.PYTHON_CANDIDATE_CONTROL]
            if any(
                record.payload.get("status") != "completed"
                for record in (candidate, current_record, python_record)
            ):
                continue
            candidate_seconds = _number(candidate.payload, "solver_seconds")
            current_seconds = _number(current_record.payload, "solver_seconds")
            python_seconds = _number(python_record.payload, "solver_seconds")
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
            and by_identity[key][ArchitectureMode.CURRENT_STAGE052].payload.get(
                "status"
            )
            == "completed"
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

    historical, historical_identity = (
        _historical_pilot()
        if scope == "pilot"
        else (
            {},
            {
                "available": False,
                "identity_verified": False,
                "error": "not assessed for paired scope",
            },
        )
    )
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
    semantic_gates_passed = all(
        bool(value.get("semantics_complete")) for value in replay.values()
    )
    architecture_modes = (
        ArchitectureMode.PER_SOLVE_RUNTIME,
        ArchitectureMode.FULL_NATIVE_ALNS,
        ArchitectureMode.HOST_SCHEDULER,
    )
    qualification_passed = axis_replay_passed and semantic_gates_passed and all(
        bool(differential[mode.value]["passed"])
        and bool(speedups[mode.value]["performance_qualified"])
        for mode in architecture_modes
    )
    review_status = (
        "COMPARISON_COMPLETE_QUALIFIED"
        if qualification_passed
        else "COMPARISON_COMPLETE_NOT_QUALIFIED"
        if axis_replay_passed
        else "NOT_READY"
    )
    cuda_evaluation_condition = _scheduler_screening_occupancy(
        by_mode[ArchitectureMode.HOST_SCHEDULER]
    )
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "scope": scope,
        "axis_count": len(records),
        "axis_replay_passed": axis_replay_passed,
        "semantic_gates_passed": semantic_gates_passed,
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
            **historical_identity,
            "comparison_count": len(historical_drift),
            "drift": historical_drift,
        },
        "cuda_evaluation_condition": cuda_evaluation_condition,
        "cuda_evaluation_condition_met": bool(
            cuda_evaluation_condition["condition_met"]
        ),
        "producer_identity": _producer_identity(records),
        "reviewer_provenance": _reviewer_provenance(),
        "raw_axis_inventory": _raw_axis_inventory(records),
        "formal_started": False,
        "production_default_changed": False,
        "qualification_passed": qualification_passed,
        "review_status": review_status,
    }


def _scheduler_screening_occupancy(
    records: Iterable[ReviewRecord],
) -> dict[str, object]:
    occupancy: list[int] = []
    for record in records:
        native = record.payload.get("native_execution_statistics")
        if not isinstance(native, dict):
            continue
        values = native.get("candidate_screening_occupancies")
        if isinstance(values, list):
            occupancy.extend(
                value
                for value in values
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
            )
    if not occupancy:
        return {
            "available": False,
            "condition_met": False,
            "maximum": None,
            "reason": "native candidate-screening occupancy is not recorded",
        }
    maximum = max(occupancy)
    return {
        "available": True,
        "condition_met": maximum >= 32,
        "maximum": maximum,
        "reason": "native candidate-screening occupancy replayed",
    }


def _producer_identity(records: Iterable[ReviewRecord]) -> dict[str, object]:
    values = tuple(records)
    return {
        "repository_revisions": sorted(
            {_string(record.payload, "revision") for record in values}
        ),
        "wheel_sha256": sorted(
            {_string(record.payload, "wheel_sha256") for record in values}
        ),
        "native_sha256": sorted(
            {_string(record.payload, "native_sha256") for record in values}
        ),
        "run_labels": sorted(
            {_string(record.payload, "run_label") for record in values}
        ),
    }


def _reviewer_provenance() -> dict[str, object]:
    source_path = Path(__file__).resolve()
    repository_root = source_path.parents[3]
    revision: str | None = None
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        candidate = completed.stdout.strip()
        if len(candidate) == 40:
            revision = candidate
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "repository_revision": revision,
        "source_path": "src/evrptw/experiments/stage052_native_architecture_review.py",
        "source_sha256": _sha256(source_path),
    }


def render_report(review: Mapping[str, object]) -> str:
    metrics = _mapping(review, "mode_metrics")
    performance = _mapping(review, "performance")
    differential = _mapping(review, "differential_gates")
    historical = _mapping(review, "historical_attempt72")
    cuda = _mapping(review, "cuda_evaluation_condition")
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
        "Queue wait median s | Exact batch occupancy median | RSS median MiB | "
        "Artifact median MiB |",
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
            + _median_text(raw, "exact_backend_batch_occupancy")
            + " | "
            + _median_scaled_text(raw, "rss_bytes", 1024**2)
            + " | "
            + _median_scaled_text(raw, "artifact_bytes", 1024**2)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Throughput / resource envelope（吞吐与资源边界）",
            "",
            "| Mode | Iter/s | Candidate tx/s | Screened routes/s | Exact/s | "
            "CPU % of one core | Cache MiB | Persistence median s | Vehicle median |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        raw = _mapping(metrics, mode.value)
        lines.append(
            f"| {mode.value} | {_median_text(raw, 'effective_iterations_per_second')} | "
            f"{_median_text(raw, 'candidate_transactions_per_second')} | "
            f"{_median_text(raw, 'screened_routes_per_second')} | "
            f"{_median_text(raw, 'exact_started_per_second')} | "
            f"{_median_text(raw, 'cpu_utilization_percent_of_one_core')} | "
            f"{_median_scaled_text(raw, 'cache_memory_bytes', 1024**2)} | "
            f"{_median_text(raw, 'persistence_seconds')} | "
            f"{_median_text(raw, 'vehicle_count')} |"
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
            f"- Historical identity verified（历史身份已核验）："
            f"`{historical.get('identity_verified')}`",
            f"- Historical comparisons（历史配对数）："
            f"`{historical.get('comparison_count')}`",
            f"- CUDA condition（CUDA 条件）："
            f"`{cuda.get('condition_met')}`；{cuda.get('reason')}；"
            "本轮未自动运行 CUDA。",
            "",
            "## Instrumentation limitations（测量边界）",
            "",
            "Exact batch occupancy（精确批量占用度）不是 native candidate-screening "
            "occupancy（原生候选筛选占用度），不用于 CUDA 门槛。"
            "`persistence_seconds` 包含 solver 执行时间，不是独立持久化成本。"
            "Host 轴的 CPU/RSS 仅测量 client shard，不含 scheduler service。",
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
    reviewer = _mapping(review, "reviewer_provenance")
    producer = _mapping(review, "producer_identity")
    manifest = {
        "schema_version": "stage05.2-native-architecture-review-manifest-v1",
        "scope": _string(review, "scope"),
        "status": _string(review, "review_status"),
        "reviewer_provenance": dict(reviewer),
        "producer_identity": dict(producer),
        "files": {
            output_json.name: _sha256(output_json),
            output_json.with_suffix(output_json.suffix + ".sha256").name: _sha256(
                output_json.with_suffix(output_json.suffix + ".sha256")
            ),
            output_markdown.name: _sha256(output_markdown),
        },
    }
    manifest_path = output_json.with_name(output_json.stem + "_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
    if review["review_status"] == "COMPARISON_COMPLETE_QUALIFIED":
        return 0
    if review["review_status"] == "COMPARISON_COMPLETE_NOT_QUALIFIED":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "HISTORICAL_PILOT_ROOT",
    "REVIEW_SCHEMA_VERSION",
    "ReviewRecord",
    "_historical_pilot",
    "_raw_axis_inventory",
    "_scheduler_screening_occupancy",
    "load_records",
    "render_report",
    "review_records",
    "write_review",
)
