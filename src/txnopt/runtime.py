"""Canonical serial TxnOpt runtime used as the Level 1 Python specification."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Final

from txnopt._internal.candidate_txn import CandidateTxn, TxnPhase
from txnopt._internal.random_tape import RandomTape
from txnopt._internal.versions import CONTRACT_VERSION
from txnopt.contracts import Oracle, RunConfig, RunResult, SearchKernel

_DIGEST: Final = re.compile(r"[0-9a-f]{64}")
_TAPE_WORDS_PER_ROUND: Final = 16


class RuntimeContractError(RuntimeError):
    """Raised when a kernel or oracle violates the TxnOpt contract."""


class SerialTxnRuntime[StateT, CandidateT, ObjectiveT]:
    """One-owner serial specification with transactional cache visibility."""

    def __init__(self, *, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._clock_ns = clock_ns

    def run(
        self,
        initial_state: StateT,
        *,
        kernel: SearchKernel[StateT, CandidateT],
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
        config: RunConfig,
    ) -> RunResult[StateT, ObjectiveT]:
        if config.execution_mode != "serial":
            raise RuntimeContractError(
                "SerialTxnRuntime supports only the serial execution mode"
            )
        if oracle.deterministic is not True:
            raise RuntimeContractError("oracle did not declare deterministic semantics")

        oracle.validate(initial_state)
        state = initial_state
        state_digest = self._validated_state_digest(oracle, state)
        objective = oracle.objective(state)
        random_tape = RandomTape(config.seed)
        cache: dict[str, StateT] = {}
        started_work = 0
        semantic_events: list[Mapping[str, object]] = [
            {
                "event": "run_open",
                "contract": CONTRACT_VERSION,
                "seed": config.seed,
                "state_digest": state_digest,
            }
        ]
        deadline_ns = (
            None
            if config.deadline_seconds is None
            else self._clock_ns() + int(config.deadline_seconds * 1_000_000_000)
        )

        for round_id in range(config.max_rounds):
            if deadline_ns is not None and self._clock_ns() >= deadline_ns:
                return self._result(
                    state=state,
                    objective=objective,
                    reason="deadline",
                    events=semantic_events,
                    config=config,
                    oracle=oracle,
                )
            candidates = tuple(
                kernel.propose(
                    state,
                    round_id=round_id,
                    random_tape=random_tape.take(_TAPE_WORDS_PER_ROUND),
                )
            )
            if not candidates:
                return self._result(
                    state=state,
                    objective=objective,
                    reason="kernel_exhausted",
                    events=semantic_events,
                    config=config,
                    oracle=oracle,
                )
            unique_candidates, candidate_keys = self._deduplicate(oracle, candidates)
            admitted_mask = tuple(oracle.screen(unique_candidates))
            if len(admitted_mask) != len(unique_candidates) or any(
                not isinstance(admitted, bool) for admitted in admitted_mask
            ):
                raise RuntimeContractError(
                    "oracle screening decisions must be ordered booleans"
                )
            admitted = tuple(
                candidate
                for candidate, keep in zip(unique_candidates, admitted_mask, strict=True)
                if keep
            )
            admitted_keys = tuple(
                key
                for key, keep in zip(candidate_keys, admitted_mask, strict=True)
                if keep
            )
            if not admitted:
                semantic_events.append(
                    {"event": "round_terminated", "reason": "no_admissible_candidates"}
                )
                return self._result(
                    state=state,
                    objective=objective,
                    reason="no_admissible_candidates",
                    events=semantic_events,
                    config=config,
                    oracle=oracle,
                )

            transaction = CandidateTxn(
                txn_id=f"round-{round_id:08d}",
                snapshot_digest=state_digest,
                candidate_keys=admitted_keys,
            )
            semantic_events.append(self._phase_event(transaction))
            missing = tuple(
                (key, candidate)
                for key, candidate in zip(admitted_keys, admitted, strict=True)
                if key not in cache
            )
            if config.fixed_work is not None and (
                len(missing) > config.fixed_work - started_work
            ):
                transaction = transaction.transition(TxnPhase.INTERRUPTED)
                semantic_events.append(self._phase_event(transaction))
                return self._result(
                    state=state,
                    objective=objective,
                    reason="fixed_work_exhausted",
                    events=semantic_events,
                    config=config,
                    oracle=oracle,
                )

            transaction = transaction.transition(TxnPhase.RESERVED)
            semantic_events.append(self._phase_event(transaction))
            staged: dict[str, StateT] = {}
            if missing:
                transaction = transaction.transition(TxnPhase.EVALUATING)
                semantic_events.append(self._phase_event(transaction))
                started_work += len(missing)
                try:
                    evaluated = tuple(
                        oracle.evaluate_batch(
                            tuple(candidate for _key, candidate in missing),
                            work_budget=len(missing),
                            deadline_ns=deadline_ns,
                        )
                    )
                except TimeoutError:
                    if deadline_ns is None:
                        raise
                    transaction = transaction.transition(TxnPhase.INTERRUPTED)
                    semantic_events.append(self._phase_event(transaction))
                    return self._result(
                        state=state,
                        objective=objective,
                        reason="deadline",
                        events=semantic_events,
                        config=config,
                        oracle=oracle,
                    )
                if len(evaluated) != len(missing):
                    transaction = transaction.transition(TxnPhase.ABORTED)
                    semantic_events.append(self._phase_event(transaction))
                    raise RuntimeContractError(
                        "oracle batch results must preserve admitted miss order"
                    )
                staged = {
                    key: evaluated_state
                    for (key, _candidate), evaluated_state in zip(
                        missing, evaluated, strict=True
                    )
                }
            else:
                transaction = transaction.transition(TxnPhase.EVALUATING)
                semantic_events.append(self._phase_event(transaction))

            if deadline_ns is not None and self._clock_ns() >= deadline_ns:
                transaction = transaction.transition(TxnPhase.INTERRUPTED)
                semantic_events.append(self._phase_event(transaction))
                return self._result(
                    state=state,
                    objective=objective,
                    reason="deadline",
                    events=semantic_events,
                    config=config,
                    oracle=oracle,
                )
            resolved = tuple(
                staged[key] if key in staged else cache[key] for key in admitted_keys
            )
            for evaluated_state in resolved:
                oracle.validate(evaluated_state)
                self._validated_state_digest(oracle, evaluated_state)
            transaction = transaction.transition(TxnPhase.VALIDATED)
            semantic_events.append(self._phase_event(transaction))

            next_state = kernel.decide(
                state,
                admitted,
                resolved,
                round_id=round_id,
            )
            oracle.validate(next_state)
            next_digest = self._validated_state_digest(oracle, next_state)
            allowed_digests = {state_digest}
            allowed_digests.update(
                self._validated_state_digest(oracle, item) for item in resolved
            )
            if next_digest not in allowed_digests:
                transaction = transaction.transition(TxnPhase.ABORTED)
                semantic_events.append(self._phase_event(transaction))
                raise RuntimeContractError(
                    "kernel decision is neither the snapshot nor an evaluated state"
                )
            next_objective = oracle.objective(next_state)
            cache.update(staged)
            state = next_state
            state_digest = next_digest
            objective = next_objective
            transaction = transaction.transition(TxnPhase.COMMITTED)
            semantic_events.append(
                {
                    **self._phase_event(transaction),
                    "state_digest": state_digest,
                    "started_work": started_work,
                    "staged_cache_writes": len(staged),
                }
            )

        return self._result(
            state=state,
            objective=objective,
            reason="max_rounds",
            events=semantic_events,
            config=config,
            oracle=oracle,
        )

    @staticmethod
    def _validated_state_digest(
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
        state: StateT,
    ) -> str:
        digest = oracle.state_digest(state)
        if _DIGEST.fullmatch(digest) is None:
            raise RuntimeContractError("oracle state_digest must be a lowercase SHA-256")
        return digest

    @staticmethod
    def _deduplicate(
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
        candidates: Sequence[CandidateT],
    ) -> tuple[tuple[CandidateT, ...], tuple[str, ...]]:
        unique: list[CandidateT] = []
        keys: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = oracle.stable_key(candidate)
            if not key:
                raise RuntimeContractError("oracle stable_key cannot be empty")
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
            keys.append(key)
        if not unique:
            raise RuntimeContractError("candidate deduplication removed the entire round")
        return tuple(unique), tuple(keys)

    @staticmethod
    def _phase_event(transaction: CandidateTxn) -> Mapping[str, object]:
        return {
            "event": "candidate_transaction",
            "txn_id": transaction.txn_id,
            "phase": transaction.phase.value,
            "snapshot_digest": transaction.snapshot_digest,
            "candidate_keys": transaction.candidate_keys,
        }

    @staticmethod
    def _result(
        *,
        state: StateT,
        objective: ObjectiveT,
        reason: str,
        events: Sequence[Mapping[str, object]],
        config: RunConfig,
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
    ) -> RunResult[StateT, ObjectiveT]:
        final_events = [*events, {"event": "run_terminated", "reason": reason}]
        semantic_digest = hashlib.sha256(
            json.dumps(
                final_events,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        oracle_identity = f"{type(oracle).__module__}.{type(oracle).__qualname__}"
        return RunResult(
            last_committed_state=state,
            objective=objective,
            termination_reason=reason,
            semantic_digest=semantic_digest,
            physical_artifact_ref=None,
            provenance={
                "contract": CONTRACT_VERSION,
                "runtime": "txnopt.serial-reference-v1",
                "execution_mode": config.execution_mode,
                "oracle": oracle_identity,
            },
        )
