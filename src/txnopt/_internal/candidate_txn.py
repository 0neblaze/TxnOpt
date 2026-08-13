"""Immutable candidate-transaction state machine for the Python specification."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum

from txnopt._internal.versions import CONTRACT_VERSION

_DIGEST = re.compile(r"[0-9a-f]{64}")
_IDENTITY = re.compile(r"[a-z0-9][a-z0-9._-]*")


class TxnPhase(StrEnum):
    PREPARED = "PREPARED"
    RESERVED = "RESERVED"
    EVALUATING = "EVALUATING"
    VALIDATED = "VALIDATED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"
    INTERRUPTED = "INTERRUPTED"


_TERMINAL_PHASES = frozenset(
    {TxnPhase.COMMITTED, TxnPhase.ABORTED, TxnPhase.INTERRUPTED}
)
_ALLOWED_TRANSITIONS: dict[TxnPhase, frozenset[TxnPhase]] = {
    TxnPhase.PREPARED: frozenset(
        {TxnPhase.RESERVED, TxnPhase.ABORTED, TxnPhase.INTERRUPTED}
    ),
    TxnPhase.RESERVED: frozenset(
        {TxnPhase.EVALUATING, TxnPhase.ABORTED, TxnPhase.INTERRUPTED}
    ),
    TxnPhase.EVALUATING: frozenset(
        {TxnPhase.VALIDATED, TxnPhase.ABORTED, TxnPhase.INTERRUPTED}
    ),
    TxnPhase.VALIDATED: frozenset(
        {TxnPhase.COMMITTED, TxnPhase.ABORTED, TxnPhase.INTERRUPTED}
    ),
    TxnPhase.COMMITTED: frozenset(),
    TxnPhase.ABORTED: frozenset(),
    TxnPhase.INTERRUPTED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class CandidateTxn:
    """One ordered candidate transaction bound to an immutable snapshot."""

    txn_id: str
    snapshot_digest: str
    candidate_keys: tuple[str, ...]
    phase: TxnPhase = TxnPhase.PREPARED
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if _IDENTITY.fullmatch(self.txn_id) is None:
            raise ValueError("txn_id must be canonical")
        if _DIGEST.fullmatch(self.snapshot_digest) is None:
            raise ValueError("snapshot_digest must be a lowercase SHA-256")
        if not self.candidate_keys or any(not key for key in self.candidate_keys):
            raise ValueError("candidate_keys must be non-empty")
        if len(set(self.candidate_keys)) != len(self.candidate_keys):
            raise ValueError("candidate_keys must already be deduplicated")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("candidate transaction contract version is unsupported")

    @property
    def terminal(self) -> bool:
        return self.phase in _TERMINAL_PHASES

    def transition(self, next_phase: TxnPhase) -> CandidateTxn:
        """Return the next immutable state or reject an illegal transition."""

        if next_phase not in _ALLOWED_TRANSITIONS[self.phase]:
            raise ValueError(f"illegal candidate transaction transition: {self.phase}")
        return replace(self, phase=next_phase)
