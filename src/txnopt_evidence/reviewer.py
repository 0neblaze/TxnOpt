"""Independent raw replay with no import from the producer implementation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from txnopt_evidence.case_codec import review_case
from txnopt_evidence.codec import (
    canonical_json_bytes,
    read_event_stream,
    read_signed_json,
    resolve_bundle_file,
    sha256_bytes,
    sha256_file,
    verify_sidecar,
    write_signed_json,
)
from txnopt_evidence.identity import ExpectedEvidenceIdentity
from txnopt_evidence.lifecycle import (
    EvidenceLifecycle,
    EvidenceState,
    lifecycle_evidence_sha256,
)
from txnopt_evidence.refinement import replay_aggregate_refinement

_RAW_ARTIFACT_SCHEMAS = {
    "txnopt-raw-artifact-v1",
    "txnopt-raw-artifact-v2",
    "txnopt-raw-artifact-v3",
}
_FAILURE_ARTIFACT_SCHEMAS = {
    "txnopt-failure-artifact-v1",
    "txnopt-failure-artifact-v2",
    "txnopt-failure-artifact-v3",
}

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


def verify_raw_manifest(
    manifest_path: Path,
    *,
    expected_identity: ExpectedEvidenceIdentity,
) -> dict[str, Any]:
    manifest = read_signed_json(manifest_path)
    if manifest.get("schema_version") != "txnopt-raw-artifact-v3":
        raise ValueError("anchored replay requires txnopt-raw-artifact-v3")
    return _verify_raw_manifest_payload(
        manifest_path,
        manifest,
        expected_identity=expected_identity,
    )


def verify_legacy_raw_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read a v1/v2 bundle without promoting it into the anchored formal path."""

    manifest = read_signed_json(manifest_path)
    if manifest.get("schema_version") not in {
        "txnopt-raw-artifact-v1",
        "txnopt-raw-artifact-v2",
    }:
        raise ValueError("legacy compatibility requires a v1 or v2 raw artifact")
    return _verify_raw_manifest_payload(
        manifest_path,
        manifest,
        expected_identity=None,
    )


def _verify_raw_manifest_payload(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    expected_identity: ExpectedEvidenceIdentity | None,
) -> dict[str, Any]:
    if manifest.get("runner_decision") is not None or manifest.get("fallback_count") != 0:
        raise ValueError("raw producer crossed the decision or fallback boundary")
    schema = manifest.get("schema_version")
    if schema == "txnopt-raw-artifact-v3":
        _validate_v3_manifest_fields(manifest, failure=False)
        if expected_identity is None:
            raise ValueError("v3 raw artifact requires an expected evidence identity")
        _validate_expected_identity(manifest_path, manifest, expected_identity)
    else:
        _validate_legacy_producer_identity(manifest.get("producer_identity"))
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
        if schema == "txnopt-raw-artifact-v3":
            _validate_artifact_entry_fields(raw_entry)
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
    if expected_identity is not None:
        config = read_signed_json(bundle / "config.json")
        result = read_signed_json(bundle / "result.json")
        _validate_result_identity(
            result,
            config=config,
            expected_identity=expected_identity,
        )
        if (result.get("physical_artifact_ref") is not None) != (
            "physical.jsonl" in seen
        ):
            raise ValueError("result physical reference differs from the artifact set")
    _validate_sealed_lifecycle(manifest)
    return manifest


def verify_failure_manifest(
    manifest_path: Path,
    *,
    expected_identity: ExpectedEvidenceIdentity,
) -> dict[str, Any]:
    """Verify a fail-fast producer bundle without promoting it to a passing run."""

    manifest = read_signed_json(manifest_path)
    if manifest.get("schema_version") != "txnopt-failure-artifact-v3":
        raise ValueError("anchored replay requires txnopt-failure-artifact-v3")
    return _verify_failure_manifest_payload(
        manifest_path,
        manifest,
        expected_identity=expected_identity,
    )


def verify_legacy_failure_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read a v1/v2 failure bundle without promoting it into formal review."""

    manifest = read_signed_json(manifest_path)
    if manifest.get("schema_version") not in {
        "txnopt-failure-artifact-v1",
        "txnopt-failure-artifact-v2",
    }:
        raise ValueError("legacy compatibility requires a v1 or v2 failure artifact")
    return _verify_failure_manifest_payload(
        manifest_path,
        manifest,
        expected_identity=None,
    )


def _verify_failure_manifest_payload(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    expected_identity: ExpectedEvidenceIdentity | None,
) -> dict[str, Any]:
    if manifest.get("runner_decision") != "FAILED" or manifest.get("fallback_count") != 0:
        raise ValueError("failure artifact decision or fallback boundary is invalid")
    schema = manifest.get("schema_version")
    if schema == "txnopt-failure-artifact-v3":
        _validate_v3_manifest_fields(manifest, failure=True)
        _validate_expected_identity(
            manifest_path,
            manifest,
            _required_identity(expected_identity),
        )
    else:
        _validate_legacy_producer_identity(manifest.get("producer_identity"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) < 2:
        raise ValueError("failure manifest does not bind its primary artifacts")
    bundle = manifest_path.resolve(strict=True).parent
    seen: set[str] = set()
    for raw_entry in artifacts:
        if not isinstance(raw_entry, dict):
            raise ValueError("failure artifact entry must be an object")
        if schema == "txnopt-failure-artifact-v3":
            _validate_artifact_entry_fields(raw_entry)
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
    if schema == "txnopt-failure-artifact-v3":
        _validate_failure_receipt_fields(failure)
    semantic_paths = sorted(path for path in seen if path.startswith("events"))
    physical_paths = sorted(path for path in seen if path.startswith("physical"))
    expected_failure_schema = {
        "txnopt-failure-artifact-v1": "txnopt-run-failure-v1",
        "txnopt-failure-artifact-v2": "txnopt-run-failure-v2",
        "txnopt-failure-artifact-v3": "txnopt-run-failure-v3",
    }[str(schema)]
    if (
        failure.get("schema_version") != expected_failure_schema
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
            expected_identity=expected_identity,
        )
    _validate_sealed_lifecycle(manifest)
    return manifest


def _validate_v3_manifest_fields(
    manifest: Mapping[str, Any],
    *,
    failure: bool,
) -> None:
    expected = {
        "schema_version",
        "run_label",
        "input_config_sha256",
        "contract",
        "semantic_trace",
        "physical_trace",
        "domain",
        "producer_identity",
        "artifacts",
        "lifecycle",
        "runner_decision",
        "fallback_count",
    }
    if failure:
        expected.add("failure_artifact")
    if set(manifest) != expected:
        raise ValueError("v3 artifact manifest field set differs")
    expected_schema = (
        "txnopt-failure-artifact-v3" if failure else "txnopt-raw-artifact-v3"
    )
    if (
        manifest.get("schema_version") != expected_schema
        or manifest.get("contract") != "txnopt-contract-v1"
        or manifest.get("semantic_trace") != "txnopt-semantic-trace-v1"
        or manifest.get("physical_trace") != "txnopt-physical-trace-v1"
    ):
        raise ValueError("v3 artifact protocol identity differs")


def _validate_artifact_entry_fields(entry: Mapping[str, Any]) -> None:
    path = entry.get("path")
    base = {"path", "sha256", "bytes"}
    expected = (
        base | {"event_count", "terminal_event_sha256"}
        if isinstance(path, str) and path.endswith(".jsonl")
        else base
    )
    if set(entry) != expected:
        raise ValueError("artifact entry field set differs")


def _validate_expected_identity(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    expected: ExpectedEvidenceIdentity,
) -> None:
    if manifest.get("schema_version") not in {
        expected.ARTIFACT_SCHEMA,
        "txnopt-failure-artifact-v3",
    }:
        raise ValueError("raw manifest differs from its expected evidence identity")
    if manifest.get("run_label") != expected.run_label:
        raise ValueError("raw manifest run label differs from the expected identity")
    if (
        manifest.get("input_config_sha256") != expected.input_config_sha256
        or manifest.get("domain") != expected.domain
    ):
        raise ValueError("raw manifest config or domain differs from the expected identity")
    if manifest.get("producer_identity") != dict(expected.producer_identity):
        raise ValueError("raw producer identity differs from the expected build")
    bundle = manifest_path.resolve(strict=True).parent
    config_path = bundle / "config.json"
    if (
        verify_sidecar(config_path) != expected.config_artifact_sha256
        or sha256_file(config_path) != expected.input_config_sha256
    ):
        raise ValueError("retained config differs from its expected config identity")
    config = read_signed_json(config_path)
    case = config.get("case")
    run_config = config.get("run_config")
    if (
        config.get("schema_version") != "txnopt-run-config-v1"
        or config.get("run_label") != expected.run_label
        or not isinstance(case, dict)
        or case.get("domain") != expected.domain
        or not isinstance(run_config, dict)
        or run_config.get("execution_mode") != expected.execution_mode
    ):
        raise ValueError("retained config run label, domain, or mode differs")
    config_entry = next(
        (
            entry
            for entry in manifest.get("artifacts", [])
            if isinstance(entry, dict) and entry.get("path") == "config.json"
        ),
        None,
    )
    if (
        not isinstance(config_entry, dict)
        or config_entry.get("sha256") != expected.config_artifact_sha256
    ):
        raise ValueError("config artifact entry differs from the expected identity")


def _validate_result_identity(
    result: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    expected_identity: ExpectedEvidenceIdentity,
) -> None:
    fields = {
        "schema_version",
        "domain",
        "last_committed_state",
        "objective",
        "state_digest",
        "termination_reason",
        "semantic_digest",
        "physical_artifact_ref",
        "provenance",
        "fallback_count",
    }
    if set(result) != fields or result.get("schema_version") != "txnopt-run-result-v1":
        raise ValueError("result schema or field set differs")
    if result.get("domain") != expected_identity.domain or result.get("fallback_count") != 0:
        raise ValueError("result domain or fallback identity differs")
    provenance = result.get("provenance")
    expected_provenance = {
        "contract": "txnopt-contract-v1",
        "runtime": "txnopt.python-reference-v1",
        "execution_mode": expected_identity.execution_mode,
        "oracle": expected_identity.expected_oracle,
    }
    if provenance != expected_provenance:
        raise ValueError("result provenance differs from the expected identity")
    run_config = config.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError("retained run config is missing")
    expected_physical_ref = (
        "txnopt-physical-trace-v1:external"
        if run_config.get("trace_policy") == "semantic_and_physical"
        else None
    )
    if result.get("physical_artifact_ref") != expected_physical_ref:
        raise ValueError("result physical artifact reference differs from the config")
    case = config.get("case")
    if not isinstance(case, dict) or case.get("domain") != result.get("domain"):
        raise ValueError("result domain differs from the retained config")


def _validate_failure_receipt_fields(failure: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "run_label",
        "error_type",
        "error_message",
        "semantic_stream_count",
        "physical_stream_count",
        "emitted_semantic_stream_count",
        "emitted_physical_stream_count",
        "traceback",
        "fallback_count",
    }
    if set(failure) != expected:
        raise ValueError("failure receipt field set differs")


def _validate_legacy_producer_identity(raw_identity: object) -> None:
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


def _validate_sealed_lifecycle(manifest: Mapping[str, Any]) -> None:
    schema = manifest.get("schema_version")
    if schema in {"txnopt-raw-artifact-v1", "txnopt-failure-artifact-v1"}:
        if "lifecycle" in manifest:
            raise ValueError("v1 artifact cannot carry a v2 evidence lifecycle")
        return
    lifecycle = EvidenceLifecycle.from_payload(manifest.get("lifecycle"))
    run_label = manifest.get("run_label")
    input_sha256 = manifest.get("input_config_sha256")
    producer_identity = manifest.get("producer_identity")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(run_label, str)
        or not isinstance(input_sha256, str)
        or not isinstance(producer_identity, dict)
        or not isinstance(artifacts, list)
        or lifecycle.run_label != run_label
        or lifecycle.state is not EvidenceState.SEALED
        or len(lifecycle.events) != 3
        or [event.state for event in lifecycle.events]
        != [EvidenceState.PLANNED, EvidenceState.RUNNING, EvidenceState.SEALED]
        or lifecycle.events[0].evidence_sha256 != input_sha256
        or lifecycle.events[1].evidence_sha256
        != lifecycle_evidence_sha256(producer_identity)
        or lifecycle.events[2].evidence_sha256
        != lifecycle_evidence_sha256(artifacts)
    ):
        raise ValueError("artifact evidence lifecycle differs from its bound content")


def verify_manifest(
    manifest_path: Path,
    *,
    expected_identity: ExpectedEvidenceIdentity,
) -> dict[str, Any]:
    payload = read_signed_json(manifest_path)
    if payload.get("schema_version") in _FAILURE_ARTIFACT_SCHEMAS:
        return verify_failure_manifest(manifest_path, expected_identity=expected_identity)
    return verify_raw_manifest(manifest_path, expected_identity=expected_identity)


def verify_legacy_manifest(manifest_path: Path) -> dict[str, Any]:
    """Verify an explicitly requested v1/v2 raw or failure bundle."""

    payload = read_signed_json(manifest_path)
    if payload.get("schema_version") in {
        "txnopt-failure-artifact-v1",
        "txnopt-failure-artifact-v2",
    }:
        return verify_legacy_failure_manifest(manifest_path)
    return verify_legacy_raw_manifest(manifest_path)


def replay_manifest(
    manifest_path: Path,
    *,
    output_dir: Path,
    expected_identity: ExpectedEvidenceIdentity,
) -> dict[str, Any]:
    return _replay_manifest(
        manifest_path,
        output_dir=output_dir,
        expected_identity=expected_identity,
        legacy=False,
    )


def replay_legacy_manifest(manifest_path: Path, *, output_dir: Path) -> dict[str, Any]:
    """Replay v1/v2 evidence for compatibility, never formal readiness."""

    return _replay_manifest(
        manifest_path,
        output_dir=output_dir,
        expected_identity=None,
        legacy=True,
    )


def _replay_manifest(
    manifest_path: Path,
    *,
    output_dir: Path,
    expected_identity: ExpectedEvidenceIdentity | None,
    legacy: bool,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"review directory already exists: {output_dir}")
    manifest = (
        verify_legacy_raw_manifest(manifest_path)
        if legacy
        else verify_raw_manifest(
            manifest_path,
            expected_identity=_required_identity(expected_identity),
        )
    )
    raw_manifest_sha256 = verify_sidecar(manifest_path)
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
            expected_identity=expected_identity,
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
    review_lifecycle: EvidenceLifecycle | None = None
    if manifest.get("schema_version") in {
        "txnopt-raw-artifact-v2",
        "txnopt-raw-artifact-v3",
    }:
        review_lifecycle = EvidenceLifecycle.from_payload(manifest["lifecycle"]).advance(
            EvidenceState.REVIEWED,
            evidence_sha256=raw_manifest_sha256,
        )
    review = {
        "schema_version": (
            "txnopt-independent-review-v3"
            if manifest.get("schema_version") == "txnopt-raw-artifact-v3"
            else (
                "txnopt-independent-review-v2"
                if review_lifecycle is not None
                else "txnopt-independent-review-v1"
            )
        ),
        "run_label": manifest.get("run_label"),
        "raw_manifest_sha256": raw_manifest_sha256,
        **(
            {
                "expected_identity_sha256": sha256_bytes(
                    canonical_json_bytes(
                        _required_identity(expected_identity).to_payload(),
                        pretty=True,
                    )
                )
            }
            if manifest.get("schema_version") == "txnopt-raw-artifact-v3"
            else {}
        ),
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
        **(
            {"lifecycle": review_lifecycle.to_payload()}
            if review_lifecycle is not None
            else {}
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_signed_json(output_dir / "review.json", review)
    return review


def _required_identity(
    identity: ExpectedEvidenceIdentity | None,
) -> ExpectedEvidenceIdentity:
    if identity is None:  # pragma: no cover - private call contract
        raise ValueError("anchored replay requires an expected evidence identity")
    return identity


def _validate_physical_trace(
    events: tuple[dict[str, Any], ...],
    *,
    semantic_exact_work_started: bool = False,
    expected_identity: ExpectedEvidenceIdentity | None = None,
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
    if (
        expected_identity is not None
        and observation["execution_mode"] != expected_identity.execution_mode
    ):
        raise ValueError("physical execution mode differs from expected evidence identity")
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
    if expected_identity is not None:
        expects_native = (
            expected_identity.expected_oracle
            == "txnopt_cases.evrptw.native_oracle.NativeEVRPTWOracle"
        )
        if expects_native and semantic_exact_work_started and not native_events:
            raise ValueError("native exact work lacks its native round receipt")
        if not expects_native and native_events:
            raise ValueError("non-native evidence contains a native round receipt")
    _validate_native_round_observations(
        native_events,
        expected_identity=expected_identity,
    )
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
    *,
    expected_identity: ExpectedEvidenceIdentity | None = None,
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
        if expected_identity is not None:
            producer = expected_identity.producer_identity
            attestation = producer.get("native_build_attestation")
            expected_revision = producer.get("source_revision")
            expected_tree = producer.get("source_tree")
            if isinstance(attestation, dict):
                expected_revision = attestation.get("source_revision")
                expected_tree = attestation.get("source_tree")
            if (
                expected_revision is not None
                and event.get("source_revision") != expected_revision
            ) or (
                expected_tree is not None and event.get("source_tree") != expected_tree
            ):
                raise ValueError("native round source identity differs from the expected build")


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
    "replay_legacy_manifest",
    "replay_manifest",
    "verify_legacy_failure_manifest",
    "verify_legacy_manifest",
    "verify_failure_manifest",
    "verify_legacy_raw_manifest",
    "verify_manifest",
    "verify_raw_manifest",
]
