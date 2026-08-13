"""Independent raw replay with no import from the producer implementation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from txnopt_evidence.case_codec import review_case
from txnopt_evidence.codec import (
    read_event_stream,
    read_signed_json,
    resolve_bundle_file,
    sha256_file,
    verify_sidecar,
    write_signed_json,
)
from txnopt_evidence.refinement import replay_aggregate_refinement

_TRANSITIONS = {
    "PREPARED": {"RESERVED", "ABORTED", "INTERRUPTED"},
    "RESERVED": {"EVALUATING", "ABORTED", "INTERRUPTED"},
    "EVALUATING": {"VALIDATED", "ABORTED", "INTERRUPTED"},
    "VALIDATED": {"COMMITTED", "ABORTED", "INTERRUPTED"},
    "COMMITTED": set(),
    "ABORTED": set(),
    "INTERRUPTED": set(),
}
_PHYSICAL_ONLY_FIELDS = {
    "started_ns",
    "ended_ns",
    "duration_ns",
    "worker_id",
    "thread_id",
    "completion_order",
}


def verify_raw_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = read_signed_json(manifest_path)
    if manifest.get("schema_version") != "txnopt-raw-artifact-v1":
        raise ValueError("unsupported raw artifact schema")
    if manifest.get("runner_decision") is not None or manifest.get("fallback_count") != 0:
        raise ValueError("raw producer crossed the decision or fallback boundary")
    _validate_producer_identity(manifest.get("producer_identity"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) not in {3, 4}:
        raise ValueError("raw manifest must bind three or four primary artifacts")
    seen: set[str] = set()
    bundle = manifest_path.resolve(strict=True).parent
    for raw_entry in artifacts:
        if not isinstance(raw_entry, dict):
            raise ValueError("artifact entry must be an object")
        relative_path = raw_entry.get("path")
        expected_digest = raw_entry.get("sha256")
        expected_bytes = raw_entry.get("bytes")
        if (
            not isinstance(relative_path, str)
            or relative_path in seen
            or not isinstance(expected_digest, str)
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
        ):
            raise ValueError("artifact manifest entry is malformed")
        seen.add(relative_path)
        artifact = resolve_bundle_file(bundle, relative_path)
        if artifact.stat().st_size != expected_bytes or sha256_file(artifact) != expected_digest:
            raise ValueError(f"artifact differs from manifest: {relative_path}")
        verify_sidecar(artifact)
    if seen not in (
        {"config.json", "events.jsonl", "result.json"},
        {"config.json", "events.jsonl", "result.json", "physical.jsonl"},
    ):
        raise ValueError("raw manifest artifact identity set differs")
    return manifest


def verify_failure_manifest(manifest_path: Path) -> dict[str, Any]:
    """Verify a fail-fast producer bundle without promoting it to a passing run."""

    manifest = read_signed_json(manifest_path)
    if manifest.get("schema_version") != "txnopt-failure-artifact-v1":
        raise ValueError("unsupported failure artifact schema")
    if manifest.get("runner_decision") != "FAILED" or manifest.get("fallback_count") != 0:
        raise ValueError("failure artifact decision or fallback boundary is invalid")
    _validate_producer_identity(manifest.get("producer_identity"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) < 2:
        raise ValueError("failure manifest does not bind its primary artifacts")
    bundle = manifest_path.resolve(strict=True).parent
    seen: set[str] = set()
    for raw_entry in artifacts:
        if not isinstance(raw_entry, dict):
            raise ValueError("failure artifact entry must be an object")
        relative_path = raw_entry.get("path")
        expected_digest = raw_entry.get("sha256")
        expected_bytes = raw_entry.get("bytes")
        if (
            not isinstance(relative_path, str)
            or relative_path in seen
            or not isinstance(expected_digest, str)
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
        ):
            raise ValueError("failure artifact entry is malformed")
        seen.add(relative_path)
        artifact = resolve_bundle_file(bundle, relative_path)
        if artifact.stat().st_size != expected_bytes or sha256_file(artifact) != expected_digest:
            raise ValueError(f"failure artifact differs from manifest: {relative_path}")
        verify_sidecar(artifact)
    if not {"config.json", "failure.json"}.issubset(seen) or "result.json" in seen:
        raise ValueError("failure artifact identity set is invalid")
    if any(
        path not in {"config.json", "failure.json"}
        and not (
            path in {"events.jsonl", "physical.jsonl"}
            or (
                path.endswith(".jsonl")
                and path.rsplit("-", 1)[0] in {"events", "physical"}
                and path.rsplit("-", 1)[1][:-6].isdigit()
            )
        )
        for path in seen
    ):
        raise ValueError("failure manifest contains an unsupported artifact path")
    failure = read_signed_json(bundle / "failure.json")
    semantic_paths = sorted(path for path in seen if path.startswith("events"))
    physical_paths = sorted(path for path in seen if path.startswith("physical"))
    if (
        failure.get("schema_version") != "txnopt-run-failure-v1"
        or failure.get("run_label") != manifest.get("run_label")
        or failure.get("fallback_count") != 0
        or failure.get("semantic_stream_count") != len(semantic_paths)
        or failure.get("physical_stream_count") != len(physical_paths)
    ):
        raise ValueError("failure receipt does not match its manifest")
    semantic_exact_work_started = False
    for relative_path in semantic_paths:
        events, _digest = read_event_stream(bundle / relative_path)
        if not events:
            raise ValueError("failure semantic stream cannot be empty")
        last_digest = events[0].get("state_digest")
        for event in events:
            if event.get("event") == "candidate_transaction" and event.get("phase") == "COMMITTED":
                last_digest = event.get("state_digest")
        _validate_event_stream(
            events,
            {
                "termination_reason": events[-1].get("reason"),
                "state_digest": last_digest,
                "fallback_count": 0,
            },
        )
        refinement = replay_aggregate_refinement(events)
        if refinement.final_state_digest != last_digest:
            raise ValueError("failure refinement replay differs from the committed prefix")
        semantic_exact_work_started = (
            semantic_exact_work_started or _semantic_exact_work_started(events)
        )
    for relative_path in physical_paths:
        events, _digest = read_event_stream(bundle / relative_path)
        _validate_physical_trace(
            events,
            semantic_exact_work_started=semantic_exact_work_started,
        )
    return manifest


def _validate_producer_identity(raw_identity: object) -> None:
    if not isinstance(raw_identity, dict) or raw_identity.get("binding_status") not in {
        "BOUND_CLEAN_BUILD",
        "UNBOUND_TEST_ONLY",
    }:
        raise ValueError("producer identity is missing or invalid")
    if raw_identity.get("binding_status") != "BOUND_CLEAN_BUILD":
        return
    for key in (
        "build_manifest_sha256",
        "wheel_sha256",
        "installed_native_sha256",
    ):
        value = raw_identity.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"bound producer identity lacks {key}")


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = read_signed_json(manifest_path)
    if payload.get("schema_version") == "txnopt-failure-artifact-v1":
        return verify_failure_manifest(manifest_path)
    return verify_raw_manifest(manifest_path)


def replay_manifest(manifest_path: Path, *, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"review directory already exists: {output_dir}")
    manifest = verify_raw_manifest(manifest_path)
    bundle = manifest_path.resolve(strict=True).parent
    config = read_signed_json(bundle / "config.json")
    result = read_signed_json(bundle / "result.json")
    events, terminal_event_digest = read_event_stream(bundle / "events.jsonl")
    if any(_PHYSICAL_ONLY_FIELDS & set(event) for event in events):
        raise ValueError("physical telemetry leaked into the semantic stream")
    semantic_digest = hashlib.sha256(
        json.dumps(
            events,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if result.get("semantic_digest") != semantic_digest:
        raise ValueError("semantic digest differs from raw event replay")
    _validate_event_stream(events, result)
    refinement = replay_aggregate_refinement(events)
    if not refinement.prefix_safety_proven:
        raise ValueError("raw run does not prove committed-prefix safety")
    if refinement.final_state_digest != result.get("state_digest"):
        raise ValueError("aggregate refinement replay differs from the raw result")
    case = config.get("case")
    if not isinstance(case, dict):
        raise ValueError("raw config case must be an object")
    case_review = review_case(case, result)
    physical_events: tuple[dict[str, Any], ...] = ()
    physical_terminal_digest: str | None = None
    physical_path = bundle / "physical.jsonl"
    if physical_path.is_file():
        physical_events, physical_terminal_digest = read_event_stream(physical_path)
        _validate_physical_trace(
            physical_events,
            semantic_exact_work_started=_semantic_exact_work_started(events),
        )
    has_physical_ref = result.get("physical_artifact_ref") is not None
    if has_physical_ref != bool(physical_events):
        raise ValueError("physical artifact reference differs from raw physical trace")
    event_entry = next(
        entry
        for entry in manifest["artifacts"]
        if isinstance(entry, dict) and entry.get("path") == "events.jsonl"
    )
    if (
        event_entry.get("event_count") != len(events)
        or event_entry.get("terminal_event_sha256") != terminal_event_digest
    ):
        raise ValueError("event count or terminal chain digest differs")
    if physical_events:
        physical_entry = next(
            entry
            for entry in manifest["artifacts"]
            if isinstance(entry, dict) and entry.get("path") == "physical.jsonl"
        )
        if (
            physical_entry.get("event_count") != len(physical_events)
            or physical_entry.get("terminal_event_sha256") != physical_terminal_digest
        ):
            raise ValueError("physical event count or chain digest differs")

    committed = sum(
        event.get("event") == "candidate_transaction" and event.get("phase") == "COMMITTED"
        for event in events
    )
    review = {
        "schema_version": "txnopt-independent-review-v1",
        "run_label": manifest.get("run_label"),
        "raw_manifest_sha256": verify_sidecar(manifest_path),
        "producer_identity": manifest["producer_identity"],
        "status": "PASS",
        "semantic_digest": semantic_digest,
        "event_count": len(events),
        "terminal_event_sha256": terminal_event_digest,
        "committed_transactions": committed,
        "case_replay": case_review,
        "physical_event_count": len(physical_events),
        "physical_terminal_event_sha256": physical_terminal_digest,
        "prefix_safety": "PASS",
        "aggregate_refinement_replay": "PASS",
        "final_cache_generation": refinement.final_cache_generation,
        "committed_candidate_key_digests": list(
            refinement.committed_candidate_key_digests
        ),
        "fallback_count": 0,
        "readiness_decision": None,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_signed_json(output_dir / "review.json", review)
    return review


def _validate_physical_trace(
    events: tuple[dict[str, Any], ...],
    *,
    semantic_exact_work_started: bool = False,
) -> None:
    observation = events[0] if events else {}
    if (
        not events
        or observation.get("event") != "run_observation"
        or observation.get("trace") != "txnopt-physical-trace-v1"
        or any(event.get("event") == "run_observation" for event in events[1:])
    ):
        raise ValueError("physical trace requires exactly one leading run observation")
    for field in ("workers", "started_ns", "ended_ns", "duration_ns"):
        value = observation.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("physical run observation is incomplete")
    if (
        observation["workers"] <= 0
        or observation["ended_ns"] < observation["started_ns"]
        or observation["duration_ns"]
        != observation["ended_ns"] - observation["started_ns"]
        or not isinstance(observation.get("execution_mode"), str)
        or not observation["execution_mode"]
        or not isinstance(observation.get("termination_reason"), str)
        or not observation["termination_reason"]
    ):
        raise ValueError("physical run observation is inconsistent")
    observed_cmax = observation.get("observed_cmax_upper_ns")
    if semantic_exact_work_started and (
        isinstance(observed_cmax, bool)
        or not isinstance(observed_cmax, int)
        or observed_cmax <= 0
    ):
        raise ValueError("physical trace lacks run Cmax for semantic exact work")
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
    native_events = tuple(
        event for event in events[1:] if event.get("event") == "native_round_observation"
    )
    waste_events = tuple(
        event for event in events[1:] if event.get("event") == "t4_waste_observation"
    )
    if len(native_events) + len(waste_events) != len(events) - 1:
        raise ValueError("physical trace contains an unsupported observation")
    _validate_native_round_observations(native_events)
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
            raise ValueError("physical trace contains an invalid T4 waste observation")
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
            raise ValueError("physical trace T4 bounds do not recompute independently")
        cmax = event.get("measured_cmax_ns")
        observed_cmax = observation.get("observed_cmax_upper_ns")
        exact_work_started = (
            event["observed_discarded_work_units"] > 0
            or event["observed_post_boundary_work_units"] > 0
        )
        if exact_work_started and (
            isinstance(observed_cmax, bool)
            or not isinstance(observed_cmax, int)
            or observed_cmax <= 0
            or cmax != observed_cmax
        ):
            raise ValueError("physical trace lacks its measured run Cmax")
        if cmax is not None and (
            isinstance(cmax, bool)
            or not isinstance(cmax, int)
            or cmax <= 0
            or event.get("cost_basis")
            != "measured_transaction_elapsed_upper_bound_ns"
            or event.get("observed_discarded_cost_upper_ns")
            != event["observed_discarded_work_units"] * cmax
            or event.get("discarded_cost_bound_ns")
            != event["discarded_work_bound_units"] * cmax
            or event.get("observed_post_boundary_cost_upper_ns")
            != event["observed_post_boundary_work_units"] * cmax
            or event.get("post_boundary_cost_bound_ns")
            != event["post_boundary_work_bound_units"] * cmax
        ):
            raise ValueError("physical trace contains an invalid measured Cmax binding")


def _validate_native_round_observations(
    events: Sequence[Mapping[str, Any]],
) -> None:
    previous_round_call = 0
    integer_fields = (
        "context_pack_count",
        "round_call_count",
        "worker_count",
        "scheduled_worker_count",
        "parallel_route_threshold",
        "started_work",
        "completed_work",
        "interrupted_work",
        "budget_limit",
        "budget_reserved_work",
        "budget_remaining_work",
        "prepared_cache_write_count",
        "prepared_cache_key_checksum",
        "semantic_event_count",
        "task_receipt_count",
        "fallback_count",
    )
    valid_traces = {
        ("PREPARED", "RESERVED", "EVALUATING", "VALIDATED"),
        ("PREPARED", "RESERVED", "EVALUATING", "INTERRUPTED"),
    }
    for event in events:
        if (
            event.get("trace") != "txnopt-physical-trace-v1"
            or event.get("protocol") != "txnopt-native-round-v1"
            or any(
                isinstance(event.get(field), bool)
                or not isinstance(event.get(field), int)
                or event[field] < 0
                for field in integer_fields
            )
            or event["context_pack_count"] != 1
            or event["fallback_count"] != 0
            or event["worker_count"] <= 0
            or event["scheduled_worker_count"] <= 0
            or event["scheduled_worker_count"] > event["worker_count"]
            or event["parallel_route_threshold"] != event["worker_count"] * 2
            or event["budget_limit"]
            != event["budget_reserved_work"] + event["budget_remaining_work"]
            or event["started_work"] > event["budget_reserved_work"]
            or event["started_work"]
            != event["completed_work"] + event["interrupted_work"]
        ):
            raise ValueError("native round observation violates its numeric ledger")
        phase_trace = event.get("phase_trace")
        if (
            not isinstance(phase_trace, list)
            or tuple(phase_trace) not in valid_traces
            or event.get("phase") != phase_trace[-1]
            or event["semantic_event_count"] != len(phase_trace)
        ):
            raise ValueError("native round observation violates its phase trace")
        phase = event["phase"]
        if phase == "VALIDATED":
            if (
                event["interrupted_work"] != 0
                or event["started_work"] != event["budget_reserved_work"]
                or event["prepared_cache_write_count"] != event["completed_work"]
            ):
                raise ValueError("validated native round lacks an atomic prepared delta")
        elif (
            event["prepared_cache_write_count"] != 0
            or (
                event["interrupted_work"] == 0
                and event["started_work"] == event["budget_reserved_work"]
            )
        ):
            raise ValueError("interrupted native round exposed a prepared cache delta")
        execution_policy = event.get("execution_policy")
        expected_policy = (
            "serial_configured"
            if event["worker_count"] == 1
            else (
                "parallel"
                if event["budget_reserved_work"] >= event["parallel_route_threshold"]
                else "serial_small_batch"
            )
        )
        expected_scheduled = (
            min(event["budget_reserved_work"], event["worker_count"])
            if expected_policy == "parallel"
            else 1
        )
        if (
            execution_policy != expected_policy
            or event["scheduled_worker_count"] != expected_scheduled
        ):
            raise ValueError("native round observation violates worker topology")
        task_receipts = event.get("task_receipts")
        if (
            not isinstance(task_receipts, list)
            or event["task_receipt_count"] != len(task_receipts)
            or len(task_receipts)
            != (event["scheduled_worker_count"] if expected_policy == "parallel" else 0)
        ):
            raise ValueError("native round task-receipt count is inconsistent")
        expected_first = 0
        for sequence, task in enumerate(task_receipts):
            if (
                not isinstance(task, list)
                or len(task) != 7
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in task
                )
                or task[0] != sequence
                or task[1] >= event["worker_count"]
                or task[2] != expected_first
                or task[2] >= task[3]
                or task[3] > event["budget_reserved_work"]
                or task[4] > task[5]
                or task[5] > task[6]
            ):
                raise ValueError("native round task receipt is malformed")
            expected_first = task[3]
        if task_receipts and expected_first != event["budget_reserved_work"]:
            raise ValueError("native round task receipts do not cover reserved work")
        round_call = event["round_call_count"]
        if round_call <= previous_round_call:
            raise ValueError("native round call count is not strictly increasing")
        previous_round_call = round_call
        for field in ("source_revision", "source_tree"):
            digest = event.get(field)
            if (
                not isinstance(digest, str)
                or len(digest) != 40
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("native round source identity is malformed")


def _semantic_exact_work_started(events: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        event.get("event") == "candidate_transaction"
        and event.get("phase") == "COMMITTED"
        and isinstance(event.get("started_work"), int)
        and not isinstance(event.get("started_work"), bool)
        and event["started_work"] > 0
        for event in events
    )


def _validate_event_stream(
    events: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
) -> None:
    if events[0].get("event") != "run_open" or events[-1].get("event") != "run_terminated":
        raise ValueError("semantic stream lacks canonical run boundaries")
    if events[-1].get("reason") != result.get("termination_reason"):
        raise ValueError("termination reason differs from semantic stream")
    phases: dict[str, str] = {}
    committed_state_digest = events[0].get("state_digest")
    for event in events:
        if event.get("event") != "candidate_transaction":
            continue
        txn_id = event.get("txn_id")
        phase = event.get("phase")
        if not isinstance(txn_id, str) or not isinstance(phase, str) or phase not in _TRANSITIONS:
            raise ValueError("candidate transaction event is malformed")
        previous = phases.get(txn_id)
        if previous is None:
            if phase != "PREPARED":
                raise ValueError("candidate transaction did not start PREPARED")
        elif phase not in _TRANSITIONS[previous]:
            raise ValueError(f"illegal candidate transaction transition {previous}->{phase}")
        phases[txn_id] = phase
        if phase == "COMMITTED":
            started_work = event.get("started_work")
            if (
                isinstance(started_work, bool)
                or not isinstance(started_work, int)
                or started_work < 0
            ):
                raise ValueError("committed transaction lacks its started-work ledger")
            committed_state_digest = event.get("state_digest")
    if any(phase not in {"COMMITTED", "ABORTED", "INTERRUPTED"} for phase in phases.values()):
        raise ValueError("semantic stream ended with an incomplete transaction")
    if committed_state_digest != result.get("state_digest"):
        raise ValueError("result state is not the last committed prefix")
    if result.get("fallback_count") != 0:
        raise ValueError("raw result reports fallback")


__all__ = [
    "replay_manifest",
    "verify_failure_manifest",
    "verify_manifest",
    "verify_raw_manifest",
]
