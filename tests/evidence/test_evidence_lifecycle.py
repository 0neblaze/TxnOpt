from __future__ import annotations

from copy import deepcopy

import pytest

from txnopt_evidence.lifecycle import (
    EvidenceLifecycle,
    EvidenceState,
    lifecycle_evidence_sha256,
)

RUN_LABEL = "txnopt_lifecycle_attempt01"
CONFIG_SHA = "1" * 64
PRODUCER_SHA = "2" * 64
ARTIFACT_SHA = "3" * 64
REVIEW_SHA = "4" * 64


def _reviewed() -> EvidenceLifecycle:
    return (
        EvidenceLifecycle.start(RUN_LABEL, evidence_sha256=CONFIG_SHA)
        .advance(EvidenceState.RUNNING, evidence_sha256=PRODUCER_SHA)
        .advance(EvidenceState.SEALED, evidence_sha256=ARTIFACT_SHA)
        .advance(EvidenceState.REVIEWED, evidence_sha256=REVIEW_SHA)
    )


def test_lifecycle_replays_the_complete_hash_chain() -> None:
    lifecycle = _reviewed()
    replayed = EvidenceLifecycle.from_payload(lifecycle.to_payload())

    assert replayed == lifecycle
    assert replayed.state is EvidenceState.REVIEWED
    assert [event.ordinal for event in replayed.events] == [0, 1, 2, 3]


def test_lifecycle_rejects_illegal_transition_and_post_close_append() -> None:
    lifecycle = EvidenceLifecycle.start(RUN_LABEL, evidence_sha256=CONFIG_SHA)
    with pytest.raises(ValueError, match="illegal"):
        lifecycle.advance(EvidenceState.SEALED, evidence_sha256=ARTIFACT_SHA)

    closed = (
        _reviewed()
        .advance(EvidenceState.CLASSIFIED, evidence_sha256="5" * 64)
        .advance(EvidenceState.RETAINED, evidence_sha256="6" * 64)
        .advance(EvidenceState.CLOSED, evidence_sha256="7" * 64)
    )
    with pytest.raises(ValueError, match="illegal"):
        closed.advance(EvidenceState.RUNNING, evidence_sha256="8" * 64)


def test_lifecycle_rejects_tampered_state_evidence_and_chain() -> None:
    payload = _reviewed().to_payload()
    for field, replacement in (
        ("state", "CLOSED"),
        ("evidence_sha256", "9" * 64),
        ("previous_event_sha256", "a" * 64),
        ("event_sha256", "b" * 64),
    ):
        tampered = deepcopy(payload)
        if field == "state":
            tampered[field] = replacement
        else:
            tampered["events"][2][field] = replacement  # type: ignore[index]
        with pytest.raises(ValueError):
            EvidenceLifecycle.from_payload(tampered)


def test_blocked_retention_can_only_return_through_classification() -> None:
    blocked = _reviewed().advance(
        EvidenceState.BLOCKED_RETENTION,
        evidence_sha256="5" * 64,
    )
    with pytest.raises(ValueError, match="illegal"):
        blocked.advance(EvidenceState.RETAINED, evidence_sha256="6" * 64)
    assert (
        blocked.advance(EvidenceState.CLASSIFIED, evidence_sha256="7" * 64).state
        is EvidenceState.CLASSIFIED
    )


def test_lifecycle_evidence_digest_uses_canonical_content() -> None:
    left = lifecycle_evidence_sha256({"b": 2, "a": 1})
    right = lifecycle_evidence_sha256({"a": 1, "b": 2})
    assert left == right
