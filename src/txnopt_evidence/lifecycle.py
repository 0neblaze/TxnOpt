"""Immutable, replayable lifecycle for one TxnOpt evidence bundle."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from txnopt_evidence.codec import canonical_json_bytes, sha256_bytes

LIFECYCLE_SCHEMA_VERSION = "txnopt-evidence-lifecycle-v1"

_DIGEST = re.compile(r"[0-9a-f]{64}")
_RUN_LABEL = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}")
_GENESIS = "0" * 64


class EvidenceState(StrEnum):
    """Observable evidence states; raw failure remains a sealed bundle."""

    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    SEALED = "SEALED"
    REVIEWED = "REVIEWED"
    CLASSIFIED = "CLASSIFIED"
    BLOCKED_RETENTION = "BLOCKED_RETENTION"
    RETAINED = "RETAINED"
    CLOSED = "CLOSED"


_TRANSITIONS: Mapping[EvidenceState, frozenset[EvidenceState]] = {
    EvidenceState.PLANNED: frozenset({EvidenceState.RUNNING}),
    EvidenceState.RUNNING: frozenset({EvidenceState.SEALED}),
    EvidenceState.SEALED: frozenset({EvidenceState.REVIEWED}),
    EvidenceState.REVIEWED: frozenset(
        {EvidenceState.CLASSIFIED, EvidenceState.BLOCKED_RETENTION}
    ),
    EvidenceState.BLOCKED_RETENTION: frozenset({EvidenceState.CLASSIFIED}),
    EvidenceState.CLASSIFIED: frozenset({EvidenceState.RETAINED}),
    EvidenceState.RETAINED: frozenset({EvidenceState.CLOSED}),
    EvidenceState.CLOSED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    ordinal: int
    state: EvidenceState
    evidence_sha256: str
    previous_event_sha256: str
    event_sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "state": self.state.value,
            "evidence_sha256": self.evidence_sha256,
            "previous_event_sha256": self.previous_event_sha256,
            "event_sha256": self.event_sha256,
        }


@dataclass(frozen=True, slots=True)
class EvidenceLifecycle:
    """Hash-chained state history with one legal successor per append."""

    run_label: str
    events: tuple[LifecycleEvent, ...]

    @classmethod
    def start(cls, run_label: str, *, evidence_sha256: str) -> Self:
        _validate_run_label(run_label)
        _validate_digest(evidence_sha256, "evidence_sha256")
        event = _event(
            run_label,
            ordinal=0,
            state=EvidenceState.PLANNED,
            evidence_sha256=evidence_sha256,
            previous_event_sha256=_GENESIS,
        )
        return cls(run_label, (event,))

    @property
    def state(self) -> EvidenceState:
        if not self.events:
            raise ValueError("evidence lifecycle cannot be empty")
        return self.events[-1].state

    def advance(self, state: EvidenceState, *, evidence_sha256: str) -> Self:
        _validate_digest(evidence_sha256, "evidence_sha256")
        if state not in _TRANSITIONS[self.state]:
            raise ValueError(f"illegal evidence lifecycle transition {self.state}->{state}")
        prior = self.events[-1]
        event = _event(
            self.run_label,
            ordinal=len(self.events),
            state=state,
            evidence_sha256=evidence_sha256,
            previous_event_sha256=prior.event_sha256,
        )
        return type(self)(self.run_label, (*self.events, event))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "run_label": self.run_label,
            "state": self.state.value,
            "events": [event.to_payload() for event in self.events],
        }

    @classmethod
    def from_payload(cls, payload: object) -> Self:
        if not isinstance(payload, dict):
            raise ValueError("evidence lifecycle must be an object")
        if set(payload) != {"schema_version", "run_label", "state", "events"}:
            raise ValueError("evidence lifecycle field set differs")
        if payload.get("schema_version") != LIFECYCLE_SCHEMA_VERSION:
            raise ValueError("unsupported evidence lifecycle schema")
        run_label = payload.get("run_label")
        raw_events = payload.get("events")
        if not isinstance(run_label, str) or not isinstance(raw_events, list) or not raw_events:
            raise ValueError("evidence lifecycle identity or events are invalid")
        _validate_run_label(run_label)
        events: list[LifecycleEvent] = []
        previous = _GENESIS
        previous_state: EvidenceState | None = None
        for ordinal, raw_event in enumerate(raw_events):
            if not isinstance(raw_event, dict) or set(raw_event) != {
                "ordinal",
                "state",
                "evidence_sha256",
                "previous_event_sha256",
                "event_sha256",
            }:
                raise ValueError("evidence lifecycle event field set differs")
            raw_state = raw_event.get("state")
            if not isinstance(raw_state, str):
                raise ValueError("evidence lifecycle state is invalid")
            try:
                state = EvidenceState(raw_state)
            except ValueError as error:
                raise ValueError("evidence lifecycle state is invalid") from error
            evidence_sha256 = raw_event.get("evidence_sha256")
            claimed_previous = raw_event.get("previous_event_sha256")
            claimed_digest = raw_event.get("event_sha256")
            if raw_event.get("ordinal") != ordinal or claimed_previous != previous:
                raise ValueError("evidence lifecycle ordinal or chain differs")
            if not isinstance(evidence_sha256, str) or not isinstance(claimed_digest, str):
                raise ValueError("evidence lifecycle digest is invalid")
            _validate_digest(evidence_sha256, "evidence_sha256")
            _validate_digest(claimed_digest, "event_sha256")
            if ordinal == 0:
                if state is not EvidenceState.PLANNED:
                    raise ValueError("evidence lifecycle must start PLANNED")
            elif previous_state is None or state not in _TRANSITIONS[previous_state]:
                raise ValueError("evidence lifecycle contains an illegal transition")
            expected = _event(
                run_label,
                ordinal=ordinal,
                state=state,
                evidence_sha256=evidence_sha256,
                previous_event_sha256=previous,
            )
            if claimed_digest != expected.event_sha256:
                raise ValueError("evidence lifecycle event digest differs")
            events.append(expected)
            previous = expected.event_sha256
            previous_state = state
        lifecycle = cls(run_label, tuple(events))
        if payload.get("state") != lifecycle.state.value:
            raise ValueError("evidence lifecycle terminal state differs")
        return lifecycle


def lifecycle_evidence_sha256(payload: object) -> str:
    """Bind a lifecycle transition to canonical immutable evidence bytes."""

    return sha256_bytes(canonical_json_bytes(payload))


def _event(
    run_label: str,
    *,
    ordinal: int,
    state: EvidenceState,
    evidence_sha256: str,
    previous_event_sha256: str,
) -> LifecycleEvent:
    unsigned = {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "run_label": run_label,
        "ordinal": ordinal,
        "state": state.value,
        "evidence_sha256": evidence_sha256,
        "previous_event_sha256": previous_event_sha256,
    }
    return LifecycleEvent(
        ordinal,
        state,
        evidence_sha256,
        previous_event_sha256,
        sha256_bytes(canonical_json_bytes(unsigned)),
    )


def _validate_digest(value: str, name: str) -> None:
    if _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _validate_run_label(run_label: str) -> None:
    if _RUN_LABEL.fullmatch(run_label) is None:
        raise ValueError("evidence lifecycle run label is not canonical")


__all__ = [
    "EvidenceLifecycle",
    "EvidenceState",
    "LIFECYCLE_SCHEMA_VERSION",
    "LifecycleEvent",
    "lifecycle_evidence_sha256",
]
