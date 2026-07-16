"""Independent review CLI for Stage 4 adaptive-weights evidence.

Re-reads raw Stage 4 artifacts, replays the validator and objective, checks
exact-call consistency, evaluates the six Stage 4 readiness gates, and produces
``review_report.md``, ``review_findings.csv``, ``gate_evaluation.csv``, and
``review_manifest.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evrptw.artifacts import ArtifactReader, verify_manifest
from evrptw.experiments.stage02_route_reduction import FORMAL_INSTANCES, FORMAL_SEEDS
from evrptw.experiments.stage03_measurement import SMOKE_INSTANCES
from evrptw.experiments.stage04_weights import DIAGNOSTIC_AXES, PER_RUN_FIELDS
from evrptw.objective import (
    ObjectiveComparison,
    SolutionObjective,
    compare_objectives,
)
from evrptw.parser import parse_schneider
from evrptw.validation import validate_routes

READY_FOR_STAGE05 = "READY_FOR_STAGE05"
NOT_READY = "NOT_READY"
STAGE04_REVIEW_SCHEMA_VERSION = "stage04-review-v4"
STAGE04_RAW_SCHEMA_VERSION = "stage04-adaptive-weights-v4"

_MIN_CALLS_PER_OPERATOR_DEFAULT = 5

_GATE_OPERATOR_CALL_SUFFICIENCY = "operator_call_sufficiency"
_GATE_SIX_CATEGORY_STATISTICS = "six_category_statistics"
_GATE_ADAPTIVE_BETTER_THAN_FIXED = "adaptive_better_than_fixed"
_GATE_NOT_SINGLE_BEST_SEED = "not_single_best_seed"
_GATE_STD_NOT_INCREASED = "std_not_increased"
_GATE_REPLAY_CONSISTENCY = "replay_consistency"

_GATE_NAMES = (
    _GATE_OPERATOR_CALL_SUFFICIENCY,
    _GATE_SIX_CATEGORY_STATISTICS,
    _GATE_ADAPTIVE_BETTER_THAN_FIXED,
    _GATE_NOT_SINGLE_BEST_SEED,
    _GATE_STD_NOT_INCREASED,
    _GATE_REPLAY_CONSISTENCY,
)

_SIX_CATEGORY_FIELDS = (
    "accepted_improving",
    "accepted_equal",
    "accepted_worse",
    "rejected",
    "new_global_best",
    "vehicle_reduction",
)


def validate_stage04_operator_audit(
    operator_statistics: Mapping[str, object],
    segment_events: Sequence[Mapping[str, object]],
    *,
    min_calls: int,
    segment_length: int | None = None,
    completed_iterations: int | None = None,
) -> tuple[bool, str]:
    """Validate complete six-category statistics and segment update boundaries."""

    if not operator_statistics:
        return False, "adaptive operator statistics are missing"
    failures: list[str] = []
    exact_event_matrix = segment_length is not None or completed_iterations is not None
    if exact_event_matrix and (
        segment_length is None
        or completed_iterations is None
        or segment_length <= 0
        or completed_iterations < 0
    ):
        failures.append("segment length and completed iterations are invalid")
    elif not exact_event_matrix and not segment_events:
        failures.append("adaptive segment events are missing")
    for name, raw_stats in operator_statistics.items():
        if not isinstance(raw_stats, Mapping):
            failures.append(f"{name}: statistics are not an object")
            continue
        missing = [field for field in _SIX_CATEGORY_FIELDS if field not in raw_stats]
        if missing:
            failures.append(f"{name}: missing six-category fields {missing}")
            continue
        role = raw_stats.get("role")
        expected_role = name.partition(":")[0]
        if role not in {"neighborhood", "destroy", "repair"} or role != expected_role:
            failures.append(f"{name}: invalid or mismatched adaptive weight role {role!r}")
        try:
            values = {
                field: _strict_nonnegative_int(raw_stats[field], field)
                for field in _SIX_CATEGORY_FIELDS
            }
            accepted = _strict_nonnegative_int(raw_stats.get("accepted"), "accepted")
            calls = _strict_nonnegative_int(raw_stats.get("calls"), "calls")
        except ValueError as error:
            failures.append(f"{name}: {error}")
            continue
        accepted_sum = (
            values["accepted_improving"]
            + values["accepted_equal"]
            + values["accepted_worse"]
        )
        if accepted != accepted_sum:
            failures.append(f"{name}: accepted={accepted} but category sum={accepted_sum}")
        if calls != accepted + values["rejected"]:
            failures.append(
                f"{name}: calls={calls} but accepted + rejected="
                f"{accepted + values['rejected']}"
            )
        if values["new_global_best"] > values["accepted_improving"]:
            failures.append(f"{name}: new_global_best exceeds accepted_improving")
        if values["vehicle_reduction"] > values["accepted_improving"]:
            failures.append(f"{name}: vehicle_reduction exceeds accepted_improving")
    observed_event_keys: list[tuple[int, str]] = []
    for event in segment_events:
        event_type = event.get("type")
        if event_type not in {"stage04_segment_update", "stage04_segment_skip"}:
            failures.append(f"unexpected segment event type {event_type!r}")
            continue
        operator = event.get("operator")
        if operator not in operator_statistics:
            failures.append(f"segment event references unknown operator {operator!r}")
        elif event.get("role") != str(operator).partition(":")[0]:
            failures.append(f"{operator}: segment event role is missing or invalid")
        try:
            iteration = _strict_nonnegative_int(event.get("iteration"), "iteration")
            calls = _strict_nonnegative_int(event.get("segment_calls"), "segment_calls")
        except ValueError as error:
            failures.append(str(error))
            continue
        if event_type == "stage04_segment_update" and calls < min_calls:
            failures.append(f"{event.get('operator')}: update with {calls} calls below {min_calls}")
        if event_type == "stage04_segment_skip" and calls >= min_calls:
            failures.append(
                f"{event.get('operator')}: skipped with {calls} calls at or above "
                f"{min_calls}"
            )
        if isinstance(operator, str):
            observed_event_keys.append((iteration, operator))
    if (
        exact_event_matrix
        and segment_length is not None
        and completed_iterations is not None
        and segment_length > 0
        and completed_iterations >= 0
    ):
        boundaries = range(segment_length - 1, completed_iterations, segment_length)
        expected_event_keys = {
            (boundary, operator)
            for boundary in boundaries
            for operator in operator_statistics
        }
        observed_event_key_set = set(observed_event_keys)
        if len(observed_event_keys) != len(observed_event_key_set):
            failures.append("duplicate segment boundary/operator event")
        missing_events = expected_event_keys - observed_event_key_set
        extra_events = observed_event_key_set - expected_event_keys
        if missing_events:
            failures.append(
                f"missing {len(missing_events)} segment boundary/operator events"
            )
        if extra_events:
            failures.append(
                f"unexpected {len(extra_events)} segment boundary/operator events"
            )
    detail = "; ".join(failures) if failures else "complete operator and segment audit passed"
    return not failures, detail


def _strict_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field} must be a non-negative integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.isdigit():
        result = int(value)
    else:
        raise ValueError(f"{field} must be a non-negative integer")
    if result < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return result


def validate_stage04_scope_identities(
    rows: Sequence[Mapping[str, object]], *, scope: str
) -> tuple[bool, str]:
    """Require the exact configured instance/seed/axis identity set."""

    if scope not in {"smoke", "formal"}:
        return False, f"unknown scope {scope!r}"
    instances = SMOKE_INSTANCES if scope == "smoke" else FORMAL_INSTANCES
    expected = {
        (instance, int(seed), axis)
        for instance in instances
        for seed in FORMAL_SEEDS
        for axis in DIAGNOSTIC_AXES
    }
    observed = [
        (str(row.get("instance")), _as_int(row.get("seed")), str(row.get("axis")))
        for row in rows
    ]
    details: list[str] = []
    if len(observed) != len(set(observed)):
        details.append("duplicate instance/seed/axis identities")
    missing = expected - set(observed)
    extra = set(observed) - expected
    if missing:
        details.append(f"missing={len(missing)}")
    if extra:
        details.append(f"extra={len(extra)}")
    return not details, "; ".join(details) if details else f"exact {len(expected)}-axis scope"


def validate_stage04_per_run_rows(
    rows: Sequence[Mapping[str, str]], *, scope: str
) -> tuple[bool, str]:
    """Strictly validate the complete per-run CSV schema, types, and identities."""

    failures: list[str] = []
    numeric_int_fields = {
        "seed", "vehicle_count", "charging_count", "iterations",
        "effective_iterations", "accepted_moves", "improving_moves",
        "rejected_moves", "accepted_improving", "accepted_equal",
        "accepted_worse", "reheat_count", "restart_count",
        "maximum_stagnation", "exact_started_calls", "exact_completed_calls",
    }
    numeric_float_fields = {
        "total_distance", "total_charging_time", "acceptance_rate",
        "initial_temperature",
    }
    identities: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if tuple(row) != PER_RUN_FIELDS:
            failures.append(f"row {index}: invalid CSV schema or column order")
        try:
            for field in numeric_int_fields:
                _strict_nonnegative_int(row.get(field), field)
            for field in numeric_float_fields:
                value = row.get(field)
                if value is None or value == "" or not isinstance(value, str):
                    raise ValueError(f"{field} must be numeric")
                parsed = float(value)
                if not math.isfinite(parsed) or parsed < 0:
                    raise ValueError(f"{field} must be a finite non-negative number")
            if row.get("feasible") not in {"True", "False"}:
                raise ValueError("feasible must be True or False")
            if row.get("intensification_active") not in {"True", "False"}:
                raise ValueError("intensification_active must be True or False")
        except ValueError as error:
            failures.append(f"row {index}: invalid value: {error}")
        identities.append({
            "instance": row.get("instance", ""),
            "seed": row.get("seed", ""),
            "axis": row.get("axis", ""),
        })
    identity_ok, identity_detail = validate_stage04_scope_identities(
        identities, scope=scope
    )
    if not identity_ok:
        failures.append(identity_detail)
    return not failures, "; ".join(failures) if failures else identity_detail


def review_stage04(
    *,
    run_dir: Path,
    scope: str,
    benchmark_dir: Path,
    output_dir: Path,
    stage00_baseline_dir: Path,
) -> dict[str, Path]:
    """Review Stage 4 adaptive-weights evidence and produce gate reports.

    Parameters
    ----------
    run_dir
        ``results/<run_label>`` directory containing the Stage 4 artifact bundle.
    scope
        ``"smoke"`` or ``"formal"``.
    benchmark_dir
        Directory containing Schneider instance ``.txt`` files.
    output_dir
        Where to write review output files.
    stage00_baseline_dir
        ``experiments/baselines/stage00`` directory with ``summary_results.csv``.
    """

    if scope not in {"smoke", "formal"}:
        raise ValueError("Stage 4 scope must be smoke or formal")

    instances = SMOKE_INSTANCES if scope == "smoke" else FORMAL_INSTANCES
    expected_axes = len(instances) * len(FORMAL_SEEDS) * len(DIAGNOSTIC_AXES)

    # ------------------------------------------------------------------
    # 1. Verify the artifact-bundle manifest.
    # ------------------------------------------------------------------
    try:
        manifest = verify_manifest(run_dir)
    except Exception as error:
        return _write_failure_report(
            output_dir,
            run_dir,
            scope,
            f"manifest verification failed: {error}",
            expected_axes,
        )

    run_label = str(manifest.get("run_label") or run_dir.name)
    if manifest.get("stage_id") != "stage04" or manifest.get("component") != "adaptive_weights":
        return _write_failure_report(
            output_dir,
            run_dir,
            scope,
            "artifact bundle is not Stage 4 adaptive_weights evidence",
            expected_axes,
            run_label=run_label,
        )
    if manifest.get("status") != "complete" or manifest.get("evidence_completeness") != "complete":
        return _write_failure_report(
            output_dir,
            run_dir,
            scope,
            "partial Stage 4 evidence cannot become ready",
            expected_axes,
            run_label=run_label,
        )
    metadata_path = run_dir / "control" / f"{run_label}_run_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        return _write_failure_report(
            output_dir, run_dir, scope, f"metadata verification failed: {error}",
            expected_axes, run_label=run_label,
        )
    if metadata.get("schema_version") != STAGE04_RAW_SCHEMA_VERSION:
        return _write_failure_report(
            output_dir, run_dir, scope, "Stage 4 v3 metadata is required",
            expected_axes, run_label=run_label,
        )

    # ------------------------------------------------------------------
    # 2. Read per_run_results.csv from the run directory.
    # ------------------------------------------------------------------
    per_run_path = _find_per_run_results(run_dir, manifest, run_label)
    per_run_rows: list[dict[str, str]] = []
    if per_run_path.is_file():
        per_run_rows = _read_csv_rows(per_run_path)
    else:
        return _write_failure_report(
            output_dir,
            run_dir,
            scope,
            f"per_run_results.csv not found at {per_run_path}",
            expected_axes,
            run_label=run_label,
        )
    per_run_ok, per_run_detail = validate_stage04_per_run_rows(
        per_run_rows, scope=scope
    )
    if not per_run_ok:
        return _write_failure_report(
            output_dir,
            run_dir,
            scope,
            f"per_run_results.csv validation failed: {per_run_detail}",
            expected_axes,
            run_label=run_label,
        )

    # ------------------------------------------------------------------
    # 3. Read Stage 0 baseline vehicle-count standard deviations.
    # ------------------------------------------------------------------
    stage00_std = _load_stage00_vehicle_std(
        stage00_baseline_dir / "summary_results.csv"
    )

    # ------------------------------------------------------------------
    # 4. Group raw/solution artifacts by (instance, seed) and replay.
    # ------------------------------------------------------------------
    reader = ArtifactReader(run_dir)
    artifact_groups = _group_artifacts(manifest)

    findings: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []

    for (instance_name, seed), group in sorted(artifact_groups.items()):
        instance = parse_schneider(benchmark_dir / f"{instance_name}.txt")

        raw: dict[str, Any] = {}
        solution: dict[str, Any] = {}
        if "raw" in group:
            try:
                raw = reader.read_json(group["raw"])
            except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
                findings.append({
                    "gate": _GATE_REPLAY_CONSISTENCY,
                    "instance": instance_name,
                    "seed": seed,
                    "axis": "",
                    "finding": f"raw JSON unreadable: {error}",
                    "status": "fail",
                })
        if "solution" in group:
            try:
                solution = reader.read_json(group["solution"])
            except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
                findings.append({
                    "gate": _GATE_REPLAY_CONSISTENCY,
                    "instance": instance_name,
                    "seed": seed,
                    "axis": "",
                    "finding": f"solution JSON unreadable: {error}",
                    "status": "fail",
                })

        raw_axes = raw.get("axes")
        if not isinstance(raw_axes, Mapping):
            raw_axes = {}
        sol_axes = solution.get("axes")
        if not isinstance(sol_axes, Mapping):
            sol_axes = {}

        for axis in DIAGNOSTIC_AXES:
            identity = {
                "instance": instance_name,
                "seed": seed,
                "axis": axis,
            }
            per_run = _find_per_run_row(per_run_rows, instance_name, seed, axis)
            sol_axis_value = sol_axes.get(axis)
            sol_axis: Mapping[str, Any] = (
                sol_axis_value if isinstance(sol_axis_value, Mapping) else {}
            )
            raw_axis_value = raw_axes.get(axis)
            raw_axis: Mapping[str, Any] = (
                raw_axis_value if isinstance(raw_axis_value, Mapping) else {}
            )

            # --- Revalidate routes ---
            routes_data = sol_axis.get("routes")
            routes = _routes(routes_data)
            validation_result = None
            objective: SolutionObjective | None = None
            if routes is not None:
                try:
                    validation_result = validate_routes(instance, routes)
                    if validation_result.feasible:
                        objective = SolutionObjective.from_report(
                            instance, validation_result
                        )
                except Exception as error:
                    findings.append({
                        **identity,
                        "gate": _GATE_REPLAY_CONSISTENCY,
                        "finding": f"validation raised: {error}",
                        "status": "fail",
                    })
            else:
                findings.append({
                    **identity,
                    "gate": _GATE_REPLAY_CONSISTENCY,
                    "finding": "solution routes missing",
                    "status": "fail",
                })

            # --- Compare objectives ---
            stored_key = sol_axis.get("objective_key")
            raw_key = raw_axis.get("objective_key")
            objective_matches = (
                objective is not None
                and isinstance(stored_key, list)
                and tuple(stored_key) == objective.key
                and raw_key == stored_key
            )

            # --- Check exact-call consistency ---
            exact_started = _as_int(raw_axis.get("started_calls"))
            exact_completed = _as_int(raw_axis.get("completed_calls"))
            per_run_started = _as_int(
                per_run.get("exact_started_calls")
            ) if per_run else 0
            per_run_completed = _as_int(
                per_run.get("exact_completed_calls")
            ) if per_run else 0
            exact_consistent = (
                exact_started == per_run_started
                and exact_completed == per_run_completed
            )

            # --- Extract Stage 4 statistics ---
            stage04_stats = raw_axis.get("stage04_statistics")
            if not isinstance(stage04_stats, Mapping):
                stage04_stats = {}
            min_calls = _as_int(
                stage04_stats.get("min_calls_per_operator")
            ) or _MIN_CALLS_PER_OPERATOR_DEFAULT

            neighborhood_stats = raw_axis.get("neighborhood_statistics")
            if not isinstance(neighborhood_stats, Mapping):
                neighborhood_stats = None

            operator_statistics = raw_axis.get("adaptive_operator_statistics")
            segment_events = raw_axis.get("stage04_segment_events")
            operator_audit_ok = False
            operator_audit_detail = "Stage 4 v2 operator audit evidence is missing"
            if (
                axis.startswith("adaptive_")
                and isinstance(operator_statistics, Mapping)
                and isinstance(segment_events, list)
            ):
                typed_events = [event for event in segment_events if isinstance(event, Mapping)]
                if len(typed_events) == len(segment_events):
                    operator_audit_ok, operator_audit_detail = validate_stage04_operator_audit(
                        operator_statistics,
                        typed_events,
                        min_calls=min_calls,
                        segment_length=_as_int(stage04_stats.get("segment_length")),
                        completed_iterations=_as_int(
                            raw_axis.get("effective_iterations")
                        ),
                    )

            # --- Extract per-run data for gate evaluation ---
            accepted_improving = _as_int(
                per_run.get("accepted_improving")
            ) if per_run else 0
            accepted_equal = _as_int(
                per_run.get("accepted_equal")
            ) if per_run else 0
            accepted_worse = _as_int(
                per_run.get("accepted_worse")
            ) if per_run else 0
            rejected_moves = _as_int(
                per_run.get("rejected_moves")
            ) if per_run else 0
            vehicle_count = _as_int_or_none(
                per_run.get("vehicle_count")
            ) if per_run else None
            weight_mode = str(
                per_run.get("weight_mode", "")
            ) if per_run else ""

            replay_row: dict[str, object] = {
                **identity,
                "valid": bool(
                    validation_result is not None and validation_result.feasible
                ),
                "objective_matches": objective_matches,
                "exact_consistent": exact_consistent,
                "recomputed_objective_key": (
                    json.dumps(list(objective.key), separators=(",", ":"))
                    if objective is not None
                    else ""
                ),
                "stored_objective_key": (
                    json.dumps(stored_key, separators=(",", ":"))
                    if isinstance(stored_key, list)
                    else ""
                ),
                "raw_objective_key": (
                    json.dumps(raw_key, separators=(",", ":"))
                    if isinstance(raw_key, list)
                    else ""
                ),
                "exact_started_calls": exact_started,
                "exact_completed_calls": exact_completed,
                "per_run_exact_started_calls": per_run_started,
                "per_run_exact_completed_calls": per_run_completed,
                "feasible": bool(sol_axis.get("feasible", False)),
                "min_calls_per_operator": min_calls,
                "accepted_improving": accepted_improving,
                "accepted_equal": accepted_equal,
                "accepted_worse": accepted_worse,
                "rejected_moves": rejected_moves,
                "vehicle_count": vehicle_count if vehicle_count is not None else "",
                "weight_mode": weight_mode,
                "neighborhood_statistics_present": neighborhood_stats is not None,
                "operator_audit_ok": operator_audit_ok,
                "operator_audit_detail": operator_audit_detail,
            }
            replay_rows.append(replay_row)

            # --- Findings for replay consistency ---
            if not replay_row["valid"]:
                findings.append({
                    **identity,
                    "gate": _GATE_REPLAY_CONSISTENCY,
                    "finding": "solution routes invalid or missing",
                    "status": "fail",
                })
            elif not objective_matches:
                findings.append({
                    **identity,
                    "gate": _GATE_REPLAY_CONSISTENCY,
                    "finding": (
                        "objective mismatch: "
                        f"recomputed={replay_row['recomputed_objective_key']} "
                        f"stored={replay_row['stored_objective_key']}"
                    ),
                    "status": "fail",
                })
            else:
                findings.append({
                    **identity,
                    "gate": _GATE_REPLAY_CONSISTENCY,
                    "finding": "routes valid and objective matches",
                    "status": "pass",
                })

            # --- Findings for exact-call consistency ---
            if not exact_consistent:
                findings.append({
                    **identity,
                    "gate": "exact_call_consistency",
                    "finding": (
                        f"exact-call counts inconsistent: "
                        f"raw={exact_started}/{exact_completed} "
                        f"per_run={per_run_started}/{per_run_completed}"
                    ),
                    "status": "fail",
                })

    # ------------------------------------------------------------------
    # 5. Evaluate the six Stage 4 readiness gates.
    # ------------------------------------------------------------------
    gate_results = _evaluate_gates(
        replay_rows, per_run_rows, stage00_std, scope
    )
    gate_entries = _gate_entries(gate_results)
    findings.extend(_gate_specific_findings(gate_results, replay_rows))

    # ------------------------------------------------------------------
    # 6. Write output files.
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "review_report": output_dir / "review_report.md",
        "review_findings": output_dir / "review_findings.csv",
        "gate_evaluation": output_dir / "gate_evaluation.csv",
        "review_manifest": output_dir / "review_manifest.json",
    }

    # review_findings.csv
    _write_csv(
        paths["review_findings"],
        ["gate", "instance", "seed", "axis", "finding", "status"],
        findings,
    )

    # gate_evaluation.csv
    gate_rows = [
        {
            "gate": name,
            "status": "pass" if passed else "fail",
            "details": details,
        }
        for name, (passed, details) in gate_entries.items()
    ]
    _write_csv(
        paths["gate_evaluation"],
        ["gate", "status", "details"],
        gate_rows,
    )

    # review_report.md
    paths["review_report"].write_text(
        _render_report(
            gate_results,
            run_label=run_label,
            scope=scope,
            total_axes=len(replay_rows),
            expected_axes=expected_axes,
        ),
        encoding="utf-8",
    )

    # review_manifest.json
    file_hashes = {
        path.name: _sha256(path)
        for name, path in paths.items()
        if name != "review_manifest"
    }
    paths["review_manifest"].write_text(
        json.dumps(
            {
                "schema_version": STAGE04_REVIEW_SCHEMA_VERSION,
                "run_label": run_label,
                "run_directory": str(run_dir),
                "scope": scope,
                "status": gate_results["status"],
                "gates": {
                    name: {"passed": passed, "details": details}
                    for name, (passed, details) in gate_entries.items()
                },
                "expected_axes": expected_axes,
                "observed_axes": len(replay_rows),
                "files": file_hashes,
                "reviewer_sha256": _sha256(Path(__file__)),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    return paths


# ======================================================================
# Gate evaluation
# ======================================================================


def _evaluate_gates(
    replay_rows: Sequence[Mapping[str, object]],
    per_run_rows: Sequence[Mapping[str, str]],
    stage00_std: Mapping[str, float],
    scope: str,
) -> dict[str, object]:
    """Evaluate all six Stage 4 readiness gates."""

    gates: dict[str, tuple[bool, str]] = {}

    # Gate 4a: Operator call sufficiency.
    gates[_GATE_OPERATOR_CALL_SUFFICIENCY] = _gate_operator_call_sufficiency(
        replay_rows
    )

    # Gate 4b: Six-category statistics.
    gates[_GATE_SIX_CATEGORY_STATISTICS] = _gate_six_category_statistics(
        replay_rows
    )

    # Gate 4c: Adaptive better than fixed (wall-clock comparison).
    adaptive_wins, win_details = _compute_adaptive_wins(replay_rows)
    gates[_GATE_ADAPTIVE_BETTER_THAN_FIXED] = (
        len(adaptive_wins) >= 3,
        f"adaptive wins {len(adaptive_wins)} (instance, seed) pairs; "
        f"need >= 3. {win_details}",
    )

    # Gate 4d: Not single-best-seed.
    winning_seeds = {seed for _, seed in adaptive_wins}
    gates[_GATE_NOT_SINGLE_BEST_SEED] = (
        len(winning_seeds) >= 2,
        f"adaptive wins on {len(winning_seeds)} unique seeds: "
        f"{sorted(winning_seeds)}; need >= 2.",
    )

    # Gate 4e: Standard deviation not increased.
    gates[_GATE_STD_NOT_INCREASED] = _gate_std_not_increased(
        replay_rows, stage00_std
    )

    # Gate 4f: Replay consistency.
    gates[_GATE_REPLAY_CONSISTENCY] = _gate_replay_consistency(replay_rows, scope)

    all_pass = all(passed for passed, _ in gates.values())
    status = READY_FOR_STAGE05 if all_pass else NOT_READY

    return {
        "schema_version": STAGE04_REVIEW_SCHEMA_VERSION,
        "scope": scope,
        "status": status,
        "gates": gates,
        "adaptive_wins": len(adaptive_wins),
        "winning_seeds": sorted(winning_seeds),
        "observed_axes": len(replay_rows),
    }


def _gate_operator_call_sufficiency(
    replay_rows: Sequence[Mapping[str, object]],
) -> tuple[bool, str]:
    """Check that every axis has sufficient operator calls.

    The raw JSON stores ``stage04_statistics.min_calls_per_operator``.  The
    per-run CSV stores aggregate operator activity (``accepted_improving``,
    ``accepted_equal``, ``accepted_worse``, ``rejected_moves``).  When the raw
    JSON also includes ``neighborhood_statistics``, per-operator call counts are
    checked individually; otherwise the aggregate is used as a necessary
    condition.
    """

    failures: list[str] = []
    for row in replay_rows:
        axis = str(row.get("axis", ""))
        if axis != "adaptive_wall_clock":
            continue
        if not bool(row.get("operator_audit_ok")):
            failures.append(
                f"{row['instance']}/{row['seed']}/{row['axis']}: "
                f"{row.get('operator_audit_detail')}"
            )
    if failures:
        return False, "; ".join(failures)
    return True, "all wall_clock axes passed per-operator segment audit"


def _gate_six_category_statistics(
    replay_rows: Sequence[Mapping[str, object]],
) -> tuple[bool, str]:
    """For adaptive wall_clock axes, check that accepted and rejected categories are non-zero.

    fixed_work axes use a 100-call budget that only produces 2-6 effective
    iterations, too few for meaningful weight evaluation.  Only wall_clock
    axes are evaluated for six-category sufficiency.
    """

    failures: list[str] = []
    for row in replay_rows:
        axis = str(row.get("axis", ""))
        if axis != "adaptive_wall_clock":
            continue
        if not bool(row.get("operator_audit_ok")):
            failures.append(
                f"{row['instance']}/{row['seed']}/{row['axis']}: "
                f"{row.get('operator_audit_detail')}"
            )
    if failures:
        return False, "; ".join(failures)
    return True, "all adaptive_wall_clock axes have complete reconciled six-category statistics"


def _compute_adaptive_wins(
    replay_rows: Sequence[Mapping[str, object]],
) -> tuple[list[tuple[str, int]], str]:
    """Compare adaptive_wall_clock vs fixed_wall_clock per (instance, seed).

    Returns a list of ``(instance, seed)`` pairs where adaptive is strictly
    better, plus a human-readable detail string.
    """

    # Build a lookup: (instance, seed, axis) -> objective key tuple.
    keys: dict[tuple[str, int, str], tuple[int, float, float, int]] = {}
    for row in replay_rows:
        instance = str(row["instance"])
        seed = _as_int(row["seed"])
        axis = str(row["axis"])
        stored = row.get("stored_objective_key")
        if isinstance(stored, str) and stored:
            try:
                key_list = json.loads(stored)
                if isinstance(key_list, list) and len(key_list) == 4:
                    keys[(instance, seed, axis)] = (
                        int(key_list[0]),
                        float(key_list[1]),
                        float(key_list[2]),
                        int(key_list[3]),
                    )
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    wins: list[tuple[str, int]] = []
    for (instance, seed, axis), key in keys.items():
        if axis != "adaptive_wall_clock":
            continue
        fixed_key = keys.get((instance, seed, "fixed_wall_clock"))
        if fixed_key is None:
            continue
        adaptive_obj = SolutionObjective(
            vehicle_count=key[0],
            total_distance=key[1],
            total_charging_time=key[2],
            charging_count=key[3],
        )
        fixed_obj = SolutionObjective(
            vehicle_count=fixed_key[0],
            total_distance=fixed_key[1],
            total_charging_time=fixed_key[2],
            charging_count=fixed_key[3],
        )
        comparison = compare_objectives(adaptive_obj, fixed_obj)
        if comparison == ObjectiveComparison.BETTER:
            wins.append((instance, seed))

    detail = (
        f"won: {wins}" if wins else "no adaptive wins"
    )
    return wins, detail


def _gate_std_not_increased(
    replay_rows: Sequence[Mapping[str, object]],
    stage00_std: Mapping[str, float],
) -> tuple[bool, str]:
    """Compare Stage 4 adaptive_wall_clock vehicle_count std to Stage 0.

    Stage 4 per-instance std must not exceed the Stage 0 baseline std.
    """

    stage4_std = _compute_stage4_vehicle_std(replay_rows)
    failures: list[str] = []
    for instance in sorted(stage4_std):
        s4 = stage4_std[instance]
        s0 = stage00_std.get(instance)
        if s0 is None:
            failures.append(f"{instance}: no Stage 0 baseline std")
            continue
        if s4 > s0 + 1e-9:
            failures.append(
                f"{instance}: Stage 4 std={s4:.6f} > Stage 0 std={s0:.6f}"
            )
    if failures:
        return False, "; ".join(failures)
    return True, "Stage 4 vehicle_count std <= Stage 0 for all instances"


def _gate_replay_consistency(
    replay_rows: Sequence[Mapping[str, object]],
    scope: str,
) -> tuple[bool, str]:
    """All solutions pass validator, objectives match, exact calls consistent."""

    failures: list[str] = []
    scope_ok, scope_detail = validate_stage04_scope_identities(replay_rows, scope=scope)
    if not scope_ok:
        failures.append(scope_detail)
    valid_count = 0
    for row in replay_rows:
        if not bool(row.get("valid")):
            failures.append(
                f"{row['instance']}/{row['seed']}/{row['axis']}: invalid routes"
            )
            continue
        if not bool(row.get("objective_matches")):
            failures.append(
                f"{row['instance']}/{row['seed']}/{row['axis']}: objective mismatch"
            )
            continue
        if not bool(row.get("exact_consistent")):
            failures.append(
                f"{row['instance']}/{row['seed']}/{row['axis']}: "
                f"exact-call inconsistency"
            )
            continue
        valid_count += 1
    if failures:
        return False, f"{valid_count}/{len(replay_rows)} valid; " + "; ".join(
            failures
        )
    return True, f"{scope_detail}; all objectives and exact-call counts match"


def _compute_stage4_vehicle_std(
    replay_rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Compute per-instance population std of vehicle_count for adaptive_wall_clock."""

    by_instance: dict[str, list[int]] = {}
    for row in replay_rows:
        if str(row.get("axis", "")) != "adaptive_wall_clock":
            continue
        vc = row.get("vehicle_count")
        if vc is None or vc == "":
            continue
        parsed = _as_int_or_none(vc)
        if parsed is not None:
            by_instance.setdefault(str(row["instance"]), []).append(parsed)
    return {
        instance: statistics.pstdev(values) if len(values) > 1 else 0.0
        for instance, values in by_instance.items()
    }


def _gate_specific_findings(
    gate_results: Mapping[str, object],
    replay_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Generate per-axis findings for gate failures."""

    findings: list[dict[str, object]] = []
    gates = _gate_entries(gate_results)

    # Operator call sufficiency failures.
    passed, details = gates.get(_GATE_OPERATOR_CALL_SUFFICIENCY, (True, ""))
    if not passed:
        for row in replay_rows:
            min_calls = _as_int(row.get("min_calls_per_operator"))
            if min_calls == 0:
                min_calls = _MIN_CALLS_PER_OPERATOR_DEFAULT
            total = (
                _as_int(row.get("accepted_improving"))
                + _as_int(row.get("accepted_equal"))
                + _as_int(row.get("accepted_worse"))
                + _as_int(row.get("rejected_moves"))
            )
            if total < min_calls:
                findings.append({
                    "gate": _GATE_OPERATOR_CALL_SUFFICIENCY,
                    "instance": row["instance"],
                    "seed": row["seed"],
                    "axis": row["axis"],
                    "finding": f"total_calls={total} < min={min_calls}",
                    "status": "fail",
                })

    # Six-category statistics failures.
    passed, details = gates.get(_GATE_SIX_CATEGORY_STATISTICS, (True, ""))
    if not passed:
        for row in replay_rows:
            if str(row.get("weight_mode", "")) != "adaptive":
                continue
            accepted_total = (
                _as_int(row.get("accepted_improving"))
                + _as_int(row.get("accepted_equal"))
                + _as_int(row.get("accepted_worse"))
            )
            rejected = _as_int(row.get("rejected_moves"))
            if accepted_total == 0 or rejected == 0:
                findings.append({
                    "gate": _GATE_SIX_CATEGORY_STATISTICS,
                    "instance": row["instance"],
                    "seed": row["seed"],
                    "axis": row["axis"],
                    "finding": (
                        f"accepted={accepted_total} rejected={rejected}"
                    ),
                    "status": "fail",
                })

    # Adaptive-better-than-fixed failures are reported at the aggregate level.
    passed, details = gates.get(_GATE_ADAPTIVE_BETTER_THAN_FIXED, (True, ""))
    if not passed:
        findings.append({
            "gate": _GATE_ADAPTIVE_BETTER_THAN_FIXED,
            "instance": "",
            "seed": "",
            "axis": "",
            "finding": details,
            "status": "fail",
        })

    # Not-single-best-seed failures.
    passed, details = gates.get(_GATE_NOT_SINGLE_BEST_SEED, (True, ""))
    if not passed:
        findings.append({
            "gate": _GATE_NOT_SINGLE_BEST_SEED,
            "instance": "",
            "seed": "",
            "axis": "",
            "finding": details,
            "status": "fail",
        })

    # Standard deviation failures.
    passed, details = gates.get(_GATE_STD_NOT_INCREASED, (True, ""))
    if not passed:
        findings.append({
            "gate": _GATE_STD_NOT_INCREASED,
            "instance": "",
            "seed": "",
            "axis": "",
            "finding": details,
            "status": "fail",
        })

    return findings


# ======================================================================
# Artifact helpers
# ======================================================================


def _group_artifacts(
    manifest: Mapping[str, Any],
) -> dict[tuple[str, int], dict[str, str]]:
    """Group raw, solution, trace, environment, and events artifacts.

    Unlike the Stage 3.4 version, missing required artifacts are silently
    skipped so that the review can report them as findings rather than
    aborting.
    """

    groups: dict[tuple[str, int], dict[str, str]] = {}
    for reference in manifest.get("artifacts", []):
        if not isinstance(reference, Mapping):
            continue
        artifact_type = str(reference.get("artifact_type"))
        if artifact_type not in {
            "raw",
            "solution",
            "trace",
            "environment",
            "route_dictionary",
            "events",
        }:
            continue
        if artifact_type == "events":
            subtype = str(reference.get("artifact_subtype") or "")
            if subtype != "critical":
                continue
        relative = Path(str(reference["relative_path"]))
        if len(relative.parts) < 3:
            continue
        instance_name = relative.parts[0]
        try:
            seed = int(relative.parts[1])
        except ValueError:
            continue
        groups.setdefault((instance_name, seed), {})[artifact_type] = (
            relative.as_posix()
        )
    return groups


def _find_per_run_results(
    run_dir: Path,
    manifest: Mapping[str, Any],
    run_label: str,
) -> Path:
    """Locate the per_run_results.csv within the run directory."""

    for reference in manifest.get("artifacts", []):
        if not isinstance(reference, Mapping):
            continue
        if reference.get("artifact_type") == "per_run_results":
            return run_dir / str(reference["relative_path"])
    candidates = sorted((run_dir / "control").glob("*per_run_results.csv"))
    if candidates:
        return candidates[0]
    return run_dir / "control" / f"{run_label}_per_run_results.csv"


def _find_per_run_row(
    per_run_rows: Sequence[Mapping[str, str]],
    instance: str,
    seed: int,
    axis: str,
) -> dict[str, str] | None:
    """Find the per-run CSV row matching (instance, seed, axis)."""

    for row in per_run_rows:
        if (
            str(row.get("instance")) == instance
            and _as_int(row.get("seed")) == seed
            and str(row.get("axis")) == axis
        ):
            return dict(row)
    return None


def _load_stage00_vehicle_std(path: Path) -> dict[str, float]:
    """Read per-instance vehicle_count standard deviations from Stage 0."""

    output: dict[str, float] = {}
    if not path.is_file():
        return output
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("metric") != "vehicle_count":
                continue
            raw_std = row.get("standard_deviation", "")
            if not raw_std:
                continue
            with contextlib.suppress(ValueError):
                output[row["instance"]] = float(raw_std)
    return output


# ======================================================================
# Report rendering
# ======================================================================


def _render_report(
    gate_results: Mapping[str, object],
    *,
    run_label: str,
    scope: str,
    total_axes: int,
    expected_axes: int,
) -> str:
    """Render the Markdown review report."""

    status = gate_results["status"]
    gates = _gate_entries(gate_results)

    lines: list[str] = [
        "# Stage 4 adaptive-weights independent review\n",
        f"- Status: `{status}`",
        f"- Run label: `{run_label}`",
        f"- Scope: `{scope}`",
        f"- Axes reviewed: `{total_axes}/{expected_axes}`",
        "",
        "## Gate results\n",
        "| Gate | Status | Details |",
        "|------|--------|---------|",
    ]
    for name in _GATE_NAMES:
        if name not in gates:
            continue
        passed, details = gates[name]
        status_label = "PASS" if passed else "FAIL"
        lines.append(f"| {name} | {status_label} | {details} |")

    lines.append("")
    lines.append(
        f"- Adaptive wins: `{gate_results.get('adaptive_wins', 0)}`"
    )
    lines.append(
        f"- Winning seeds: `{gate_results.get('winning_seeds', [])}`"
    )
    lines.append("")
    return "\n".join(lines)


def _write_failure_report(
    output_dir: Path,
    run_dir: Path,
    scope: str,
    reason: str,
    expected_axes: int,
    *,
    run_label: str = "",
) -> dict[str, Path]:
    """Write a minimal NOT_READY report when the manifest or per-run CSV is missing."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "review_report": output_dir / "review_report.md",
        "review_findings": output_dir / "review_findings.csv",
        "gate_evaluation": output_dir / "gate_evaluation.csv",
        "review_manifest": output_dir / "review_manifest.json",
    }
    findings = [
        {
            "gate": "manifest",
            "instance": "",
            "seed": "",
            "axis": "",
            "finding": reason,
            "status": "fail",
        }
    ]
    _write_csv(
        paths["review_findings"],
        ["gate", "instance", "seed", "axis", "finding", "status"],
        findings,
    )
    gate_rows = [
        {"gate": name, "status": "fail", "details": reason}
        for name in _GATE_NAMES
    ]
    _write_csv(
        paths["gate_evaluation"],
        ["gate", "status", "details"],
        gate_rows,
    )
    paths["review_report"].write_text(
        "# Stage 4 adaptive-weights independent review\n\n"
        f"- Status: `{NOT_READY}`\n"
        f"- Run label: `{run_label or run_dir.name}`\n"
        f"- Scope: `{scope}`\n"
        f"- Axes reviewed: `0/{expected_axes}`\n\n"
        f"**Fatal error**: {reason}\n",
        encoding="utf-8",
    )
    file_hashes = {
        path.name: _sha256(path)
        for name, path in paths.items()
        if name != "review_manifest"
    }
    paths["review_manifest"].write_text(
        json.dumps(
            {
                "schema_version": STAGE04_REVIEW_SCHEMA_VERSION,
                "run_label": run_label or run_dir.name,
                "run_directory": str(run_dir),
                "scope": scope,
                "status": NOT_READY,
                "gates": {
                    name: {"passed": False, "details": reason}
                    for name in _GATE_NAMES
                },
                "expected_axes": expected_axes,
                "observed_axes": 0,
                "files": file_hashes,
                "reviewer_sha256": _sha256(Path(__file__)),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return paths


# ======================================================================
# Small utilities
# ======================================================================


def _routes(value: object) -> list[list[str]] | None:
    """Convert a JSON routes payload to ``list[list[str]]``.

    Returns ``None`` when *value* is not a list.
    """

    if not isinstance(value, list):
        return None
    return [
        [str(name) for name in route]
        for route in value
        if isinstance(route, (list, tuple))
    ]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dicts."""

    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write rows to a CSV file with a fixed field list."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _as_int(value: object) -> int:
    """Safely convert a value to int, returning 0 for None/empty."""

    try:
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (str, bytes, bytearray, int, float)):
            return int(value)
        raise TypeError(type(value).__name__)
    except (TypeError, ValueError):
        return 0


def _as_int_or_none(value: object) -> int | None:
    """Like ``_as_int`` but returns ``None`` for empty/missing values."""

    if value is None or value == "":
        return None
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _gate_entries(
    gate_results: Mapping[str, object],
) -> dict[str, tuple[bool, str]]:
    """Validate and narrow the internal gate-result structure."""

    raw_gates = gate_results.get("gates")
    if not isinstance(raw_gates, Mapping):
        raise TypeError("gate results are missing the gates mapping")
    gates: dict[str, tuple[bool, str]] = {}
    for name, value in raw_gates.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, tuple)
            or len(value) != 2
            or not isinstance(value[0], bool)
            or not isinstance(value[1], str)
        ):
            raise TypeError("malformed gate result")
        gates[name] = (value[0], value[1])
    return gates


def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ======================================================================
# CLI entry point
# ======================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Review Stage 4 adaptive-weights evidence"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=("smoke", "formal"), required=True
    )
    parser.add_argument(
        "--benchmark-dir", type=Path, default=Path("data/schneider")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage00-baseline-dir",
        type=Path,
        default=Path("experiments/baselines/stage00"),
    )
    arguments = parser.parse_args()
    outputs = review_stage04(
        run_dir=arguments.run_dir.resolve(),
        scope=arguments.scope,
        benchmark_dir=arguments.benchmark_dir.resolve(),
        output_dir=arguments.output_dir.resolve(),
        stage00_baseline_dir=arguments.stage00_baseline_dir.resolve(),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    review_manifest = json.loads(
        outputs["review_manifest"].read_text(encoding="utf-8")
    )
    return 0 if str(review_manifest.get("status", "")).startswith(
        "READY_FOR_"
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
