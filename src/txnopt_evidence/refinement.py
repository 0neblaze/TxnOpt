"""Independent semantic-event replay for the aggregate TxnOpt refinement map."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_DIGEST = re.compile(r"[0-9a-f]{64}")
_TRANSITIONS = {
    "PREPARED": {"RESERVED", "ABORTED", "INTERRUPTED"},
    "RESERVED": {"EVALUATING", "ABORTED", "INTERRUPTED"},
    "EVALUATING": {"VALIDATED", "ABORTED", "INTERRUPTED"},
    "VALIDATED": {"COMMITTED", "ABORTED", "INTERRUPTED"},
    "COMMITTED": set(),
    "ABORTED": set(),
    "INTERRUPTED": set(),
}
_TERMINAL = {"COMMITTED", "ABORTED", "INTERRUPTED"}


class RefinementReplayError(ValueError):
    """A concrete semantic trace does not refine the aggregate model."""


@dataclass(frozen=True, slots=True)
class RefinementReplayReceipt:
    committed_transactions: int
    aborted_transactions: int
    interrupted_transactions: int
    final_state_digest: str
    final_cache_generation: int
    private_stuttering_steps: int
    committed_candidate_key_digests: tuple[str, ...]
    prefix_safety_proven: bool


def replay_aggregate_refinement(
    events: Sequence[Mapping[str, object]],
) -> RefinementReplayReceipt:
    if (
        not events
        or events[0].get("event") != "run_open"
        or events[-1].get("event") != "run_terminated"
    ):
        raise RefinementReplayError("semantic trace lacks canonical run boundaries")
    initial_digest = events[0].get("state_digest")
    if not isinstance(initial_digest, str) or _DIGEST.fullmatch(initial_digest) is None:
        raise RefinementReplayError("run_open state digest is invalid")
    visible_digest = initial_digest
    cache_generation = 0
    phases: dict[str, str] = {}
    transaction_keys: dict[str, tuple[str, ...]] = {}
    active_txn_id: str | None = None
    pending_admitted_count: int | None = None
    committed = 0
    aborted = 0
    interrupted = 0
    stuttering = 0
    committed_key_digests: list[str] = []
    prefix_safety_proven = True
    last_started_work = 0
    for event in events[1:-1]:
        event_name = event.get("event")
        if event_name not in {
            "candidate_screening",
            "candidate_transaction",
            "round_terminated",
        }:
            raise RefinementReplayError("semantic trace contains an unknown event")
        if event_name != "candidate_transaction" and (
            "state_digest" in event or "cache_generation" in event
        ):
            raise RefinementReplayError("non-transaction event changed visible state")
        if event_name == "candidate_screening":
            if active_txn_id is not None:
                raise RefinementReplayError("screening interleaved an active transaction")
            if pending_admitted_count is not None:
                raise RefinementReplayError("screening did not resolve its admitted set")
            admitted_count = event.get("admitted_count")
            if (
                isinstance(admitted_count, bool)
                or not isinstance(admitted_count, int)
                or admitted_count < 0
            ):
                raise RefinementReplayError("screening admission count is invalid")
            pending_admitted_count = admitted_count
            continue
        if event_name == "round_terminated":
            if active_txn_id is not None or pending_admitted_count not in {None, 0}:
                raise RefinementReplayError(
                    "round termination crossed an unresolved transaction boundary"
                )
            pending_admitted_count = None
            continue
        if event_name != "candidate_transaction":
            continue
        txn_id = event.get("txn_id")
        phase = event.get("phase")
        if not isinstance(txn_id, str) or not isinstance(phase, str) or phase not in _TRANSITIONS:
            raise RefinementReplayError("candidate transaction identity or phase is invalid")
        raw_keys = event.get("candidate_keys")
        if (
            not isinstance(raw_keys, (list, tuple))
            or not raw_keys
            or any(not isinstance(key, str) or not key for key in raw_keys)
        ):
            raise RefinementReplayError("transaction candidate keys are invalid")
        keys = tuple(raw_keys)
        previous = phases.get(txn_id)
        if previous is None:
            if (
                active_txn_id is not None
                or phase != "PREPARED"
                or event.get("snapshot_digest") != visible_digest
            ):
                raise RefinementReplayError("transaction is not bound to the visible snapshot")
            if len(set(keys)) != len(keys):
                raise RefinementReplayError("transaction candidate keys are not unique")
            if pending_admitted_count is None or len(keys) != pending_admitted_count:
                raise RefinementReplayError("transaction differs from the admitted candidate set")
            transaction_keys[txn_id] = keys
            active_txn_id = txn_id
            pending_admitted_count = None
        elif phase not in _TRANSITIONS[previous]:
            raise RefinementReplayError(f"illegal aggregate transition {previous}->{phase}")
        elif (
            active_txn_id != txn_id
            or event.get("snapshot_digest") != visible_digest
            or keys != transaction_keys[txn_id]
        ):
            raise RefinementReplayError("active transaction identity or candidate set changed")
        phases[txn_id] = phase
        if "cache_outcome" in event and not (
            phase == "INTERRUPTED" and event.get("cache_outcome") == "unknown"
        ):
            raise RefinementReplayError(
                "cache outcome is allowed only on an interrupted transaction"
            )
        if phase == "COMMITTED":
            next_digest = event.get("state_digest")
            next_generation = event.get("cache_generation")
            started_work = event.get("started_work")
            if (
                not isinstance(next_digest, str)
                or _DIGEST.fullmatch(next_digest) is None
                or isinstance(next_generation, bool)
                or not isinstance(next_generation, int)
                or next_generation != cache_generation + 1
                or isinstance(started_work, bool)
                or not isinstance(started_work, int)
                or started_work < last_started_work
            ):
                raise RefinementReplayError(
                    "commit did not publish state, cache, and work ledger atomically"
                )
            visible_digest = next_digest
            cache_generation = next_generation
            last_started_work = started_work
            committed_key_digests.append(
                _candidate_keys_digest(transaction_keys[txn_id])
            )
            committed += 1
        elif phase == "ABORTED":
            if "state_digest" in event or "cache_generation" in event:
                raise RefinementReplayError("aborted transaction exposed private state or cache")
            aborted += 1
        elif phase == "INTERRUPTED":
            if "state_digest" in event or "cache_generation" in event:
                raise RefinementReplayError(
                    "interrupted transaction exposed private state or cache"
                )
            if event.get("cache_outcome") == "unknown":
                prefix_safety_proven = False
            interrupted += 1
        else:
            if "state_digest" in event or "cache_generation" in event:
                raise RefinementReplayError("private transaction step changed visible state")
            stuttering += 1
        if phase in _TERMINAL:
            active_txn_id = None
    if any(phase not in _TERMINAL for phase in phases.values()):
        raise RefinementReplayError("semantic trace ended with a non-terminal transaction")
    if active_txn_id is not None:
        raise RefinementReplayError("semantic trace ended with an active transaction owner")
    if pending_admitted_count not in {None, 0}:
        raise RefinementReplayError("semantic trace omitted an admitted transaction")
    return RefinementReplayReceipt(
        committed_transactions=committed,
        aborted_transactions=aborted,
        interrupted_transactions=interrupted,
        final_state_digest=visible_digest,
        final_cache_generation=cache_generation,
        private_stuttering_steps=stuttering,
        committed_candidate_key_digests=tuple(committed_key_digests),
        prefix_safety_proven=prefix_safety_proven,
    )


def _candidate_keys_digest(keys: tuple[str, ...]) -> str:
    payload = "\0".join(keys).encode("utf-8")
    return hashlib.sha256(b"txnopt-refinement-candidate-keys-v1\0" + payload).hexdigest()


__all__ = [
    "RefinementReplayError",
    "RefinementReplayReceipt",
    "replay_aggregate_refinement",
]
