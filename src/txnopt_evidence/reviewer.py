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
    case = config.get("case")
    if not isinstance(case, dict):
        raise ValueError("raw config case must be an object")
    case_review = review_case(case, result)
    physical_events: tuple[dict[str, Any], ...] = ()
    physical_terminal_digest: str | None = None
    physical_path = bundle / "physical.jsonl"
    if physical_path.is_file():
        physical_events, physical_terminal_digest = read_event_stream(physical_path)
        if (
            len(physical_events) != 1
            or physical_events[0].get("event") != "run_observation"
            or physical_events[0].get("trace") != "txnopt-physical-trace-v1"
        ):
            raise ValueError("physical trace does not contain one canonical run observation")
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
        "status": "PASS",
        "semantic_digest": semantic_digest,
        "event_count": len(events),
        "terminal_event_sha256": terminal_event_digest,
        "committed_transactions": committed,
        "case_replay": case_review,
        "physical_event_count": len(physical_events),
        "physical_terminal_event_sha256": physical_terminal_digest,
        "prefix_safety": "PASS",
        "fallback_count": 0,
        "readiness_decision": None,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_signed_json(output_dir / "review.json", review)
    return review


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
            committed_state_digest = event.get("state_digest")
    if any(phase not in {"COMMITTED", "ABORTED", "INTERRUPTED"} for phase in phases.values()):
        raise ValueError("semantic stream ended with an incomplete transaction")
    if committed_state_digest != result.get("state_digest"):
        raise ValueError("result state is not the last committed prefix")
    if result.get("fallback_count") != 0:
        raise ValueError("raw result reports fallback")


__all__ = ["replay_manifest", "verify_raw_manifest"]
