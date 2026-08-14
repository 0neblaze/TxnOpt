"""Independently replay and aggregate a completed TxnOpt Level 1 campaign."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from txnopt_evidence.level1_campaign_common import (
    AXES,
    CampaignEntry,
    CampaignPlan,
    campaign_claim_path,
    canonical_json_bytes,
    executable_path,
    linux_host_identity,
    load_analysis_protocol,
    load_campaign_plan,
    load_prebound_expected_identity,
    memory_gib,
    physical_core_count,
    read_event_stream,
    read_signed_object,
    require_clean_repository,
    require_prebound_expected_identities,
    run_isolated_process,
    sha256_bytes,
    sha256_file,
    validate_authorization,
    verify_runtime_installation,
    verify_sidecar,
    write_signed_object,
)


def review_campaign(
    plan_path: Path,
    analysis_protocol_path: Path,
    *,
    execution_receipt_path: Path,
    python: Path,
    wheel: Path,
    review_root: Path,
    review_receipt_path: Path,
    review_workers: int,
    per_run_timeout_seconds: float,
) -> dict[str, Any]:
    if review_workers <= 0 or per_run_timeout_seconds <= 0:
        raise ValueError("review workers and per-run timeout must be positive")
    reviewer_identity = _reviewer_identity()
    plan = load_campaign_plan(plan_path)
    require_prebound_expected_identities(plan)
    analysis, analysis_sha256 = load_analysis_protocol(
        analysis_protocol_path,
        plan=plan,
    )
    execution = read_signed_object(
        execution_receipt_path,
        schema_version="txnopt-level1-campaign-execution-v2",
    )
    execution_sha256 = verify_sidecar(execution_receipt_path)
    runtime_identity = verify_runtime_installation(plan, python=python, wheel=wheel)
    runner_identity, producer_runtime_identity = _validate_execution_receipt(
        execution,
        plan,
        analysis_sha256,
        analysis=analysis,
        expected_runtime_identity=runtime_identity,
    )
    if reviewer_identity["common_tool_sha256"] != runner_identity["common_sha256"]:
        raise ValueError("campaign reviewer common contract differs from the runner")
    authorization_path = Path(str(execution["authorization_path"])).resolve(strict=True)
    authorization = read_signed_object(
        authorization_path,
        schema_version="txnopt-level1-procurement-authorization-v1",
    )
    if verify_sidecar(authorization_path) != execution.get("authorization_sha256"):
        raise ValueError("campaign execution authorization digest differs")
    validate_authorization(authorization, plan, analysis_sha256)
    if analysis.get("schema_version") == "txnopt-level1-analysis-protocol-v2":
        execution_host = _object(execution.get("host"), "execution host identity")
        if execution_host.get("provider_instance") != authorization.get(
            "tencent_provider_instance"
        ):
            raise ValueError(
                "campaign execution provider identity differs from its authorization"
            )
    review_destination = review_root.resolve()
    if review_destination.exists() or review_destination.is_symlink():
        raise FileExistsError(f"campaign review root already exists: {review_destination}")
    if review_receipt_path.exists() or review_receipt_path.is_symlink():
        raise FileExistsError(f"campaign review receipt already exists: {review_receipt_path}")
    review_destination.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()
    started_ns = time.monotonic_ns()
    records: list[dict[str, Any] | None] = [None] * len(plan.entries)
    lock = threading.Lock()

    def review_one(entry: CampaignEntry) -> dict[str, Any]:
        return _review_one(
            entry,
            python=executable_path(python),
            review_root=review_destination,
            timeout_seconds=per_run_timeout_seconds,
        )

    with ThreadPoolExecutor(max_workers=min(review_workers, len(plan.entries))) as executor:
        futures: dict[Future[dict[str, Any]], CampaignEntry] = {
            executor.submit(review_one, entry): entry for entry in plan.entries
        }
        for future in as_completed(futures):
            entry = futures[future]
            try:
                record = future.result()
            except Exception as error:  # pragma: no cover - defensive process boundary
                record = {
                    **_entry_identity(entry),
                    "status": "FAILED",
                    "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                    "error": str(error),
                }
            with lock:
                records[entry.ordinal - 1] = record

    finalized = [record for record in records if record is not None]
    failures = [record for record in finalized if record.get("status") != "PASS"]
    metrics: dict[str, Any] | None = None
    if not failures and len(finalized) == len(plan.entries):
        metrics = _aggregate(finalized, analysis, plan)
    gates = _gate_results(metrics, failures)
    all_pass = all(value == "PASS" for value in gates.values())
    reviewed_identity_tree_sha256 = _reviewed_identity_tree_sha256(finalized)
    if reviewed_identity_tree_sha256 != plan.expected_identity_tree_sha256:
        raise ValueError("reviewed expected identity tree differs from the plan")
    receipt = {
        "schema_version": "txnopt-level1-campaign-review-v2",
        "status": "READY_FOR_INTERNAL_LEVEL1_SEAL" if all_pass else "NOT_READY",
        "plan_manifest_path": str(plan.manifest_path),
        "plan_manifest_sha256": plan.manifest_sha256,
        "analysis_protocol_path": str(analysis_protocol_path.resolve(strict=True)),
        "analysis_protocol_sha256": analysis_sha256,
        "expected_identity_tree_sha256": plan.expected_identity_tree_sha256,
        "reviewed_expected_identity_tree_sha256": reviewed_identity_tree_sha256,
        "execution_receipt_path": str(execution_receipt_path.resolve(strict=True)),
        "execution_receipt_sha256": execution_sha256,
        "reviewer_orchestration_identity": reviewer_identity,
        "review_runtime_identity": runtime_identity,
        "reviewed_runner_orchestration_identity": runner_identity,
        "producer_runtime_identity": producer_runtime_identity,
        "started_at_utc": started_at,
        "ended_at_utc": _utc_now(),
        "elapsed_seconds": (time.monotonic_ns() - started_ns) / 1_000_000_000,
        "planned_run_count": len(plan.entries),
        "review_pass_count": len(finalized) - len(failures),
        "review_failure_count": len(failures),
        "failures": failures,
        "metrics": metrics,
        "gates": gates,
        "fallback_count": 0,
        "holdout_opened": False,
        "level1_ready": False,
        "internal_seal_pending": all_pass,
        "level2_entry_authorized": False,
        "push_authorized": False,
        "public_release_authorized": False,
    }
    write_signed_object(review_receipt_path.resolve(), receipt)
    return receipt


def _review_one(
    entry: CampaignEntry,
    *,
    python: Path,
    review_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    load_prebound_expected_identity(entry)
    expected_identity_path = entry.expected_identity_path
    expected_identity_sha256 = entry.expected_identity_sha256
    if expected_identity_path is None or expected_identity_sha256 is None:
        raise ValueError("campaign entry lacks its prebound expected identity")
    raw_bundle = entry.raw_output_root / entry.run_label
    manifest_path = raw_bundle / "manifest.json"
    manifest_sha256 = verify_sidecar(manifest_path)
    raw_manifest = read_signed_object(manifest_path)
    if raw_manifest.get("schema_version") != "txnopt-raw-artifact-v3":
        return {
            **_entry_identity(entry),
            "status": "FAILED",
            "error": "raw artifact schema differs",
        }
    review_dir = review_root / entry.run_label
    started_ns = time.monotonic_ns()
    completed = run_isolated_process(
        [
            str(python),
            "-I",
            "-m",
            "txnopt_evidence.review_cli",
            str(manifest_path),
            "--output-dir",
            str(review_dir),
            "--expected-identity",
            str(expected_identity_path),
        ],
        cwd=raw_bundle,
        timeout_seconds=timeout_seconds,
    )
    base = {
        **_entry_identity(entry),
        "raw_manifest_path": str(manifest_path),
        "raw_manifest_sha256": manifest_sha256,
        "expected_identity_path": str(expected_identity_path),
        "expected_identity_sha256": expected_identity_sha256,
        "review_elapsed_seconds": (time.monotonic_ns() - started_ns) / 1_000_000_000,
        "return_code": completed.returncode,
    }
    if completed.timed_out or completed.descendant_cleanup_performed or completed.returncode != 0:
        return {
            **base,
            "status": "FAILED",
            "stderr_tail": completed.stderr[-4000:],
            "stdout_tail": completed.stdout[-1000:],
            "timed_out": completed.timed_out,
            "descendant_cleanup_performed": completed.descendant_cleanup_performed,
            "descendant_processes_remaining": list(completed.descendant_processes_remaining),
        }
    output = _object(json.loads(completed.stdout), "review command output")
    review_path = review_dir / "review.json"
    review_sha256 = verify_sidecar(review_path)
    review = read_signed_object(review_path)
    expected_review_schema = "txnopt-independent-review-v3"
    result = read_signed_object(raw_bundle / "result.json")
    semantic_events = read_event_stream(raw_bundle / "events.jsonl")
    physical_events = read_event_stream(raw_bundle / "physical.jsonl")
    observation = physical_events[0]
    if (
        review.get("schema_version") != expected_review_schema
        or output.get("status") != "PASS"
        or review.get("status") != "PASS"
        or review.get("raw_manifest_sha256") != manifest_sha256
        or review.get("expected_identity_sha256") != expected_identity_sha256
        or review.get("fallback_count") != 0
        or review.get("prefix_safety") != "PASS"
        or review.get("aggregate_refinement_replay") != "PASS"
        or result.get("fallback_count") != 0
        or observation.get("event") != "run_observation"
        or review.get("physical_event_count") != len(physical_events)
    ):
        return {**base, "status": "FAILED", "error": "independent replay differs"}
    duration_ns = observation.get("duration_ns")
    if isinstance(duration_ns, bool) or not isinstance(duration_ns, int) or duration_ns <= 0:
        return {**base, "status": "FAILED", "error": "physical duration is invalid"}
    semantic_exact_work_started = _semantic_exact_work_started(semantic_events)
    cmax = observation.get("observed_cmax_upper_ns")
    if semantic_exact_work_started and (
        isinstance(cmax, bool) or not isinstance(cmax, int) or cmax <= 0
    ):
        return {**base, "status": "FAILED", "error": "full-scope Cmax is absent"}
    try:
        t4_audit = _audit_t4_waste_events(
            physical_events,
            semantic_events=semantic_events,
            budget=entry.budget,
        )
    except ValueError as error:
        return {**base, "status": "FAILED", "error": str(error)}
    return {
        **base,
        "status": "PASS",
        "review_path": str(review_path),
        "review_sha256": review_sha256,
        "semantic_digest": result.get("semantic_digest"),
        "objective": result.get("objective"),
        "state_digest": result.get("state_digest"),
        "termination_reason": result.get("termination_reason"),
        "duration_ns": duration_ns,
        "semantic_exact_work_started": semantic_exact_work_started,
        "observed_cmax_upper_ns": cmax,
        "prefix_safety_replay": "PASS",
        "aggregate_refinement_replay": "PASS",
        "physical_t4_recomputation": "PASS",
        **t4_audit,
        "descendant_cleanup_performed": completed.descendant_cleanup_performed,
        "descendant_processes_remaining": list(completed.descendant_processes_remaining),
    }


def _semantic_exact_work_started(events: tuple[dict[str, Any], ...]) -> bool:
    return any(
        event.get("event") == "candidate_transaction"
        and isinstance(event.get("started_work"), int)
        and not isinstance(event.get("started_work"), bool)
        and event["started_work"] > 0
        for event in events
    )


def _audit_t4_waste_events(
    physical_events: tuple[dict[str, Any], ...],
    *,
    semantic_events: tuple[dict[str, Any], ...],
    budget: str,
) -> dict[str, int]:
    observation = physical_events[0]
    observed_cmax = observation.get("observed_cmax_upper_ns")
    integer_fields = (
        "remaining_budget_before",
        "uncommitted_window",
        "max_requests_per_candidate",
        "post_boundary_capacity_units",
        "observed_discarded_work_units",
        "observed_post_boundary_work_units",
        "discarded_work_bound_units",
        "post_boundary_work_bound_units",
    )
    native_round_events = tuple(
        event
        for event in physical_events[1:]
        if event.get("event") == "native_round_observation"
    )
    waste_events = tuple(
        event
        for event in physical_events[1:]
        if event.get("event") == "t4_waste_observation"
    )
    if len(native_round_events) + len(waste_events) != len(physical_events) - 1:
        raise ValueError("campaign physical trace contains an unsupported observation")
    for event in waste_events:
        if (
            event.get("event") != "t4_waste_observation"
            or event.get("trace") != "txnopt-physical-trace-v1"
            or event.get("bound_satisfied") is not True
            or any(
                isinstance(event.get(field), bool)
                or not isinstance(event.get(field), int)
                or event[field] < 0
                for field in integer_fields
            )
        ):
            raise ValueError("campaign T4 event is malformed")
        discarded_bound = min(
            event["remaining_budget_before"],
            event["uncommitted_window"] * event["max_requests_per_candidate"],
        )
        post_boundary_bound = min(
            event["post_boundary_capacity_units"],
            event["uncommitted_window"] * event["max_requests_per_candidate"],
        )
        if (
            event["discarded_work_bound_units"] != discarded_bound
            or event["post_boundary_work_bound_units"] != post_boundary_bound
            or event["observed_discarded_work_units"] > discarded_bound
            or event["observed_post_boundary_work_units"] > post_boundary_bound
        ):
            raise ValueError("campaign T4 event exceeds its independently recomputed bound")
        work_was_wasted = (
            event["observed_discarded_work_units"] > 0
            or event["observed_post_boundary_work_units"] > 0
        )
        measured_cmax = event.get("measured_cmax_ns")
        if work_was_wasted and (
            isinstance(observed_cmax, bool)
            or not isinstance(observed_cmax, int)
            or observed_cmax <= 0
            or measured_cmax != observed_cmax
            or event.get("observed_discarded_cost_upper_ns")
            != event["observed_discarded_work_units"] * observed_cmax
            or event.get("discarded_cost_bound_ns") != discarded_bound * observed_cmax
            or event.get("observed_post_boundary_cost_upper_ns")
            != event["observed_post_boundary_work_units"] * observed_cmax
            or event.get("post_boundary_cost_bound_ns") != post_boundary_bound * observed_cmax
        ):
            raise ValueError("campaign T4 event lacks its independently recomputed Cmax cost")
    expected_abort_audits = (
        _expected_fixed_work_abort_audits(semantic_events)
        if budget == "fixed_work"
        else 0
    )
    if len(waste_events) < expected_abort_audits:
        raise ValueError("fixed-work abort lacks its T4 waste observation")
    return {
        "native_round_observation_count": len(native_round_events),
        "t4_waste_event_count": len(waste_events),
        "expected_fixed_work_abort_audit_count": expected_abort_audits,
        "independently_recomputed_t4_event_count": len(waste_events),
    }


def _expected_fixed_work_abort_audits(
    semantic_events: tuple[dict[str, Any], ...],
) -> int:
    """Count terminal ledgers that prove new work was discarded.

    ``started_work`` is a cumulative run ledger.  A reservation denial reports
    the already committed ledger again, so a positive absolute value does not
    by itself prove that the interrupted transaction started work.  Only a
    terminal ABORTED/INTERRUPTED ledger that advances beyond the previous
    terminal ledger requires a corresponding T4 waste observation here.
    """

    previous_terminal_started_work = 0
    expected = 0
    for event in semantic_events:
        if event.get("event") != "candidate_transaction":
            continue
        phase = event.get("phase")
        if phase not in {"COMMITTED", "ABORTED", "INTERRUPTED"}:
            continue
        started_work = event.get("started_work")
        if isinstance(started_work, bool) or not isinstance(started_work, int):
            continue
        if phase in {"ABORTED", "INTERRUPTED"} and started_work > (
            previous_terminal_started_work
        ):
            expected += 1
        previous_terminal_started_work = max(
            previous_terminal_started_work,
            started_work,
        )
    return expected


def _aggregate(
    records: list[dict[str, Any]],
    analysis: dict[str, Any],
    plan: CampaignPlan | None = None,
) -> dict[str, Any]:
    indexed: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            str(record["domain"]),
            str(record["case_id"]),
            int(record["seed"]),
            str(record["budget"]),
            str(record["axis"]),
        )
        if key in indexed:
            raise ValueError(f"duplicate reviewed campaign identity: {key}")
        indexed[key] = record
    if plan is not None:
        expected = {
            (entry.domain, entry.case_id, entry.seed, entry.budget, entry.axis)
            for entry in plan.entries
        }
        if set(indexed) != expected:
            raise ValueError("reviewed campaign identity set differs from the plan")
    bootstrap = _object(analysis.get("bootstrap"), "bootstrap protocol")
    performance_gates = _object(analysis.get("performance_gates"), "performance gates")
    by_domain: dict[str, dict[str, Any]] = {}
    parity_failures: list[dict[str, Any]] = []
    cmax_values: list[int] = []
    semantic_work_count = 0
    for record in records:
        if record["semantic_exact_work_started"]:
            semantic_work_count += 1
            cmax_values.append(int(record["observed_cmax_upper_ns"]))
    expected_run_count = len(plan.entries) if plan is not None else len(records)
    prefix_pass_count = sum(record.get("prefix_safety_replay") == "PASS" for record in records)
    refinement_pass_count = sum(
        record.get("aggregate_refinement_replay") == "PASS" for record in records
    )
    t4_recomputation_pass_count = sum(
        record.get("physical_t4_recomputation") == "PASS" for record in records
    )
    t4_waste_event_count = sum(int(record.get("t4_waste_event_count", 0)) for record in records)
    native_round_observation_count = sum(
        int(record.get("native_round_observation_count", 0)) for record in records
    )
    for domain in ("evrptw", "rcpsp"):
        identities = sorted(
            {
                (str(record["case_id"]), int(record["seed"]))
                for record in records
                if record["domain"] == domain and record["budget"] == "fixed_work"
            }
        )
        speedups: list[float] = []
        overheads: list[float] = []
        for case_id, seed in identities:
            group = {axis: indexed[(domain, case_id, seed, "fixed_work", axis)] for axis in AXES}
            objective_digests = {
                sha256_bytes(canonical_json_bytes(item["objective"])) for item in group.values()
            }
            semantic_digests = {str(item["semantic_digest"]) for item in group.values()}
            if len(objective_digests) != 1 or len(semantic_digests) != 1:
                parity_failures.append({"domain": domain, "case_id": case_id, "seed": seed})
            serial_ns = int(group["serial_1"]["duration_ns"])
            txnopt_one_ns = int(group["txnopt_1"]["duration_ns"])
            txnopt_four_ns = int(group["txnopt_4"]["duration_ns"])
            speedups.append(serial_ns / txnopt_four_ns)
            overheads.append(max(0.0, txnopt_one_ns / serial_ns - 1.0))
        geomean = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
        lower, upper = _bootstrap_interval(
            speedups,
            resamples=int(bootstrap["resamples"]),
            seed=int(bootstrap["seed"]) + (0 if domain == "evrptw" else 1),
            confidence=float(bootstrap["confidence_level"]),
        )
        maximum_overhead = max(overheads)
        by_domain[domain] = {
            "paired_sample_count": len(speedups),
            "four_worker_geomean_speedup": geomean,
            "four_worker_speedup_ci_95": [lower, upper],
            "minimum_four_worker_speedup": min(speedups),
            "maximum_four_worker_speedup": max(speedups),
            "maximum_one_worker_overhead_fraction": maximum_overhead,
            "geomean_gate": (
                "PASS"
                if geomean >= float(performance_gates["four_worker_geomean_minimum"])
                else "FAIL"
            ),
            "ci_lower_gate": (
                "PASS"
                if lower > float(performance_gates["four_worker_ci_lower_minimum_exclusive"])
                else "FAIL"
            ),
            "one_worker_overhead_gate": (
                "PASS"
                if maximum_overhead
                <= float(performance_gates["one_worker_maximum_overhead_fraction"])
                else "FAIL"
            ),
        }
    return {
        "fixed_work": {
            "domains": by_domain,
            "semantic_and_objective_parity_failure_count": len(parity_failures),
            "semantic_and_objective_parity_failures": parity_failures,
            "quality_regression_count": len(parity_failures),
        },
        "cmax": {
            "semantic_exact_work_run_count": semantic_work_count,
            "measured_positive_cmax_run_count": len(cmax_values),
            "minimum_observed_cmax_upper_ns": min(cmax_values) if cmax_values else None,
            "maximum_observed_cmax_upper_ns": max(cmax_values) if cmax_values else None,
        },
        "raw_to_review_traceability": {
            "reviewed_run_count": len(records),
            "expected_run_count": expected_run_count,
            "rate": len(records) / expected_run_count,
        },
        "safety_replay": {
            "prefix_safety_pass_count": prefix_pass_count,
            "aggregate_refinement_pass_count": refinement_pass_count,
            "physical_t4_recomputation_pass_count": t4_recomputation_pass_count,
            "native_round_observation_count": native_round_observation_count,
            "t4_waste_event_count": t4_waste_event_count,
            "build09_fault_and_formal_gate": "PASS",
        },
        "aggregation_scope": {
            "performance_budget_axis": "fixed_work",
            "fixed_time_used_for_performance_aggregation": False,
        },
    }


def _bootstrap_interval(
    speedups: list[float],
    *,
    resamples: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    if not speedups or any(value <= 0 or not math.isfinite(value) for value in speedups):
        raise ValueError("bootstrap requires finite positive paired speedups")
    logs = [math.log(value) for value in speedups]
    rng = random.Random(seed)
    values = sorted(
        math.exp(sum(logs[rng.randrange(len(logs))] for _ in logs) / len(logs))
        for _ in range(resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, math.ceil(alpha * resamples) - 1)
    upper_index = min(resamples - 1, math.ceil((1.0 - alpha) * resamples) - 1)
    return values[lower_index], values[upper_index]


def _gate_results(
    metrics: dict[str, Any] | None,
    failures: list[dict[str, Any]],
) -> dict[str, str]:
    if failures or metrics is None:
        return {
            "all_raw_runs_independently_replayed": "FAIL",
            "fixed_work_semantic_digest_and_objective_parity": "FAIL",
            "deadline_and_fault_prefix_safety": "FAIL",
            "measured_waste_within_t4_bound": "FAIL",
            "full_scope_measured_cmax": "FAIL",
            "one_worker_overhead": "FAIL",
            "two_domain_four_worker_speedup": "FAIL",
            "two_domain_ci_lower_bound": "FAIL",
            "fallback_zero": "FAIL",
        }
    domains = metrics["fixed_work"]["domains"]
    cmax = metrics["cmax"]
    traceability = metrics["raw_to_review_traceability"]
    safety = metrics["safety_replay"]
    expected = traceability["expected_run_count"]
    return {
        "all_raw_runs_independently_replayed": "PASS",
        "fixed_work_semantic_digest_and_objective_parity": (
            "PASS"
            if metrics["fixed_work"]["semantic_and_objective_parity_failure_count"] == 0
            else "FAIL"
        ),
        "deadline_and_fault_prefix_safety": (
            "PASS"
            if safety["prefix_safety_pass_count"] == expected
            and safety["aggregate_refinement_pass_count"] == expected
            and safety["build09_fault_and_formal_gate"] == "PASS"
            else "FAIL"
        ),
        "measured_waste_within_t4_bound": (
            "PASS" if safety["physical_t4_recomputation_pass_count"] == expected else "FAIL"
        ),
        "full_scope_measured_cmax": (
            "PASS"
            if cmax["semantic_exact_work_run_count"] == cmax["measured_positive_cmax_run_count"]
            and cmax["semantic_exact_work_run_count"] > 0
            else "FAIL"
        ),
        "one_worker_overhead": (
            "PASS"
            if all(domain["one_worker_overhead_gate"] == "PASS" for domain in domains.values())
            else "FAIL"
        ),
        "two_domain_four_worker_speedup": (
            "PASS"
            if all(domain["geomean_gate"] == "PASS" for domain in domains.values())
            else "FAIL"
        ),
        "two_domain_ci_lower_bound": (
            "PASS"
            if all(domain["ci_lower_gate"] == "PASS" for domain in domains.values())
            else "FAIL"
        ),
        "fallback_zero": "PASS",
    }


def _validate_execution_receipt(
    execution: dict[str, Any],
    plan: CampaignPlan,
    analysis_sha256: str,
    *,
    analysis: dict[str, Any],
    expected_runtime_identity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    runs = execution.get("runs")
    tool_identity = _object(
        execution.get("orchestration_identity"), "runner orchestration identity"
    )
    runtime_identity = _object(
        execution.get("producer_runtime_identity"), "producer runtime identity"
    )
    host = _object(execution.get("host"), "execution host identity")
    if (
        execution.get("status") != "COMPLETE_RAW_ONLY_NOT_REVIEWED"
        or execution.get("plan_manifest_sha256") != plan.manifest_sha256
        or execution.get("analysis_protocol_sha256") != analysis_sha256
        or execution.get("expected_identity_tree_sha256")
        != plan.expected_identity_tree_sha256
        or execution.get("completed_run_count") != len(plan.entries)
        or execution.get("failed_run_count") != 0
        or execution.get("not_started_run_count") != 0
        or execution.get("independent_review_performed") is not False
        or execution.get("readiness_decision") is not None
        or execution.get("fallback_count") != 0
        or not isinstance(runs, list)
        or len(runs) != len(plan.entries)
        or execution.get("maximum_window_days") != 14
        or execution.get("global_window_satisfied") is not True
    ):
        raise ValueError("campaign execution receipt is not a complete raw-only matrix")
    started_at = _parse_utc(str(execution.get("started_at_utc")))
    ended_at = _parse_utc(str(execution.get("ended_at_utc")))
    deadline_at = _parse_utc(str(execution.get("global_deadline_utc")))
    elapsed_seconds = execution.get("elapsed_seconds")
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or elapsed_seconds < 0
        or ended_at < started_at
        or deadline_at != started_at + timedelta(days=14)
        or ended_at > deadline_at
        or elapsed_seconds > 14 * 86_400
    ):
        raise ValueError("campaign execution exceeds its 14-day signed window")
    for key in (
        "wheel_sha256",
        "native_sha256",
        "source_revision",
        "source_tree",
        "python_executable_sha256",
        "python_version",
        "python_implementation",
        "platform",
        "kernel_release",
        "dependency_count",
        "dependency_lock_sha256",
        "wheel_entry_count_verified",
        "installed_entry_count_verified",
    ):
        if runtime_identity.get(key) != expected_runtime_identity.get(key):
            raise ValueError(f"campaign execution runtime identity differs: {key}")
    analysis_schema = analysis.get("schema_version")
    minimum_cores = 64 if analysis_schema == "txnopt-level1-analysis-protocol-v2" else 32
    if (
        host.get("system") != "Linux"
        or host.get("exclusive_linux_authorized") is not True
        or isinstance(host.get("physical_cores"), bool)
        or not isinstance(host.get("physical_cores"), int)
        or host["physical_cores"] < minimum_cores
        or isinstance(host.get("memory_gib"), bool)
        or not isinstance(host.get("memory_gib"), int)
        or host["memory_gib"] <= 0
        or isinstance(host.get("usable_core_tokens"), bool)
        or not isinstance(host.get("usable_core_tokens"), int)
        or host["usable_core_tokens"] <= 0
        or host["usable_core_tokens"] > host["physical_cores"]
        or not isinstance(host.get("kernel_release"), str)
        or not host["kernel_release"]
    ):
        raise ValueError("campaign execution host does not satisfy the resource contract")
    if analysis_schema == "txnopt-level1-analysis-protocol-v1" and host["memory_gib"] < 128:
        raise ValueError("campaign execution host does not satisfy the v1 memory contract")
    if analysis_schema == "txnopt-level1-analysis-protocol-v2":
        provider = _object(host.get("provider_instance"), "execution Tencent provider instance")
        if (
            provider.get("physical_cores") != 64
            or isinstance(provider.get("memory_gb"), bool)
            or not isinstance(provider.get("memory_gb"), int)
            or provider["memory_gb"] < 128
            or host.get("linux_visible_memory_is_admission_gate") is not False
            or isinstance(host.get("attempt27_peak_rss_bytes"), bool)
            or not isinstance(host.get("attempt27_peak_rss_bytes"), int)
            or host["attempt27_peak_rss_bytes"] < 0
        ):
            raise ValueError("campaign execution Tencent resource evidence differs")
    current_host = linux_host_identity(
        physical_cores=physical_core_count(),
        memory_gib=memory_gib(),
        usable_core_tokens=int(host["usable_core_tokens"]),
        exclusive_linux=True,
    )
    for key, value in current_host.items():
        if host.get(key) != value:
            raise ValueError(
                f"campaign execution host identity differs at independent review: {key}"
            )
    claim_path = Path(str(execution.get("launch_claim_path"))).resolve(strict=True)
    expected_claim_path = campaign_claim_path(plan) / "claim.json"
    if claim_path != expected_claim_path:
        raise ValueError("campaign execution launch claim path differs")
    claim = read_signed_object(
        claim_path,
        schema_version="txnopt-level1-campaign-launch-claim-v2",
    )
    if (
        verify_sidecar(claim_path) != execution.get("launch_claim_sha256")
        or claim.get("status") != "CLAIMED_BEFORE_RAW_WRITE"
        or claim.get("plan_manifest_sha256") != plan.manifest_sha256
        or claim.get("analysis_protocol_sha256") != analysis_sha256
        or claim.get("authorization_sha256") != execution.get("authorization_sha256")
        or claim.get("raw_output_root") != str(plan.raw_output_root)
        or claim.get("expected_identity_tree_sha256")
        != plan.expected_identity_tree_sha256
        or claim.get("tool_identity") != tool_identity
        or claim.get("runtime_identity") != runtime_identity
        or claim.get("host") != host
        or claim.get("active_target_process_count_before_claim") != 0
    ):
        raise ValueError("campaign execution launch claim differs")
    for key in ("authorization_path", "authorization_sha256"):
        value = execution.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"campaign execution authorization binding is missing: {key}")
    for key, expected_length in (
        ("revision", 40),
        ("git_tree", 40),
        ("runner_sha256", 64),
        ("common_sha256", 64),
    ):
        value = tool_identity.get(key)
        if not isinstance(value, str) or len(value) != expected_length:
            raise ValueError(f"campaign execution lacks runner identity: {key}")
    if tool_identity.get("source_dirty") is not False:
        raise ValueError("campaign execution runner identity is dirty")
    _verify_committed_runner_identity(tool_identity)
    for entry, raw in zip(plan.entries, runs, strict=True):
        record = _object(raw, "execution run record")
        manifest_path = entry.raw_output_root / entry.run_label / "manifest.json"
        if (
            record.get("ordinal") != entry.ordinal
            or record.get("run_label") != entry.run_label
            or record.get("config_sha256") != entry.config_sha256
            or record.get("expected_identity_relative_path")
            != entry.expected_identity_relative_path
            or record.get("expected_identity_path")
            != (
                str(entry.expected_identity_path)
                if entry.expected_identity_path is not None
                else None
            )
            or record.get("expected_identity_sha256")
            != entry.expected_identity_sha256
            or record.get("status") != "COMPLETE"
            or record.get("raw_manifest_path") != str(manifest_path)
            or record.get("raw_manifest_sha256") != verify_sidecar(manifest_path)
        ):
            raise ValueError(f"execution receipt differs at run {entry.ordinal}")
    return tool_identity, runtime_identity


def _verify_committed_runner_identity(tool_identity: dict[str, Any]) -> None:
    root = Path(__file__).resolve().parents[2]
    revision = str(tool_identity["revision"])
    committed_tree = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"{revision}^{{tree}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if committed_tree.returncode != 0 or committed_tree.stdout.strip() != tool_identity["git_tree"]:
        raise ValueError("campaign runner revision or tree is unavailable to the reviewer")
    for relative, key in (
        ("src/txnopt_evidence/level1_campaign_runner.py", "runner_sha256"),
        ("src/txnopt_evidence/level1_campaign_common.py", "common_sha256"),
    ):
        blob = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{relative}"],
            check=False,
            capture_output=True,
        )
        if blob.returncode != 0 or sha256_bytes(blob.stdout) != tool_identity[key]:
            raise ValueError(f"campaign runner committed source differs: {relative}")


def _reviewer_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    require_clean_repository(root)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    reviewer_path = Path(__file__).resolve()
    common_path = reviewer_path.parent / "txnopt_level1_campaign_common.py"
    identity = {
        "repository_root": str(root),
        "revision": revision,
        "git_tree": tree,
        "source_dirty": False,
        "reviewer_tool_sha256": sha256_file(reviewer_path),
        "common_tool_sha256": sha256_file(common_path),
        "independent_process_per_run": True,
    }
    for path, key in (
        ("src/txnopt_evidence/level1_campaign_reviewer.py", "reviewer_tool_sha256"),
        ("src/txnopt_evidence/level1_campaign_common.py", "common_tool_sha256"),
    ):
        blob = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{path}"],
            check=False,
            capture_output=True,
        )
        if blob.returncode != 0 or sha256_bytes(blob.stdout) != identity[key]:
            raise RuntimeError(f"campaign reviewer source is not committed: {path}")
    return identity


def _entry_identity(entry: CampaignEntry) -> dict[str, Any]:
    return {
        "ordinal": entry.ordinal,
        "run_label": entry.run_label,
        "expected_identity_relative_path": entry.expected_identity_relative_path,
        "expected_identity_path": (
            str(entry.expected_identity_path)
            if entry.expected_identity_path is not None
            else None
        ),
        "expected_identity_sha256": entry.expected_identity_sha256,
        "domain": entry.domain,
        "case_id": entry.case_id,
        "seed": entry.seed,
        "axis": entry.axis,
        "budget": entry.budget,
    }


def _reviewed_identity_tree_sha256(records: list[dict[str, Any]]) -> str:
    ordered = sorted(records, key=lambda record: int(record["ordinal"]))
    entries: list[dict[str, str]] = []
    for record in ordered:
        path = record.get("expected_identity_relative_path")
        digest = record.get("expected_identity_sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("campaign review record lacks its expected identity binding")
        entries.append({"path": path, "sha256": digest})
    return sha256_bytes(canonical_json_bytes(entries))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("campaign timestamp is not UTC")
    return parsed


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--analysis-protocol", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--review-workers", type=int, default=4)
    parser.add_argument("--per-run-timeout-seconds", type=float, default=600.0)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    result = review_campaign(
        arguments.plan_manifest,
        arguments.analysis_protocol,
        execution_receipt_path=arguments.execution_receipt,
        python=arguments.python,
        wheel=arguments.wheel,
        review_root=arguments.review_root,
        review_receipt_path=arguments.review_receipt,
        review_workers=arguments.review_workers,
        per_run_timeout_seconds=arguments.per_run_timeout_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
