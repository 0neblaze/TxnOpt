"""Canonical serial TxnOpt runtime used as the Level 1 Python specification."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Final

from txnopt._internal.budget import BudgetLedger
from txnopt._internal.cache import (
    CacheCommitError,
    InMemoryCacheStore,
    StaleCacheSnapshotError,
)
from txnopt._internal.candidate_txn import CandidateTxn, TxnPhase
from txnopt._internal.random_tape import RandomTape
from txnopt._internal.versions import CONTRACT_VERSION
from txnopt.contracts import Oracle, RunConfig, RunResult, SearchKernel

_DIGEST: Final = re.compile(r"[0-9a-f]{64}")
_TAPE_WORDS_PER_ROUND: Final = 16


class RuntimeContractError(RuntimeError):
    """Raised when a kernel or oracle violates the TxnOpt contract."""


class OracleWorkerError(RuntimeError):
    """Controlled worker failure that refines to the last commit prefix."""


class PythonTxnRuntime[StateT, CandidateT, ObjectiveT]:
    """One-owner Python specification for serial, barrier, and ordered execution."""

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        cache_factory: Callable[[], InMemoryCacheStore[StateT]] = InMemoryCacheStore,
    ) -> None:
        self._clock_ns = clock_ns
        self._cache_factory = cache_factory

    def run(
        self,
        initial_state: StateT,
        *,
        kernel: SearchKernel[StateT, CandidateT],
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
        config: RunConfig,
    ) -> RunResult[StateT, ObjectiveT]:
        if oracle.deterministic is not True:
            raise RuntimeContractError("oracle did not declare deterministic semantics")
        if config.execution_mode != "serial" and oracle.parallel_safe is not True:
            raise RuntimeContractError("oracle did not declare parallel-safe evaluation")

        oracle.validate(initial_state)
        state = initial_state
        state_digest = self._validated_state_digest(oracle, state)
        objective = oracle.objective(state)
        random_tape = RandomTape(config.seed)
        cache = self._cache_factory()
        budget = BudgetLedger(config.fixed_work)
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
            cache_transaction = cache.begin()
            lookup = cache.lookup(admitted_keys)
            if lookup.generation != cache_transaction.expected_generation:
                cache.rollback(cache_transaction)
                transaction = transaction.transition(TxnPhase.INTERRUPTED)
                semantic_events.append(self._phase_event(transaction))
                return self._result(
                    state=state,
                    objective=objective,
                    reason="stale_snapshot",
                    events=semantic_events,
                    config=config,
                    oracle=oracle,
                )
            missing = tuple(
                (index, key, candidate)
                for index, (key, candidate, cached) in enumerate(
                    zip(admitted_keys, admitted, lookup.values, strict=True)
                )
                if cached is None
            )
            reservation = budget.reserve(len(missing))
            if not reservation.granted:
                cache.rollback(cache_transaction)
                transaction = transaction.transition(TxnPhase.INTERRUPTED)
                semantic_events.append(
                    {
                        **self._phase_event(transaction),
                        "requested_work": reservation.requested,
                        "started_work": reservation.started_after,
                        "remaining_work": reservation.remaining_after,
                    }
                )
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
                try:
                    evaluated = self._evaluate_candidates(
                        oracle=oracle,
                        candidates=tuple(
                            candidate for _index, _key, candidate in missing
                        ),
                        budget=budget,
                        config=config,
                        deadline_ns=deadline_ns,
                    )
                except OracleWorkerError:
                    cache.rollback(cache_transaction)
                    transaction = transaction.transition(TxnPhase.ABORTED)
                    semantic_events.append(self._phase_event(transaction))
                    return self._result(
                        state=state,
                        objective=objective,
                        reason="worker_failure",
                        events=semantic_events,
                        config=config,
                        oracle=oracle,
                    )
                except TimeoutError:
                    if deadline_ns is None:
                        raise
                    cache.rollback(cache_transaction)
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
                    cache.rollback(cache_transaction)
                    transaction = transaction.transition(TxnPhase.ABORTED)
                    semantic_events.append(self._phase_event(transaction))
                    raise RuntimeContractError(
                        "oracle batch results must preserve admitted miss order"
                    )
                staged = {
                    key: evaluated_state
                    for (_index, key, _candidate), evaluated_state in zip(
                        missing, evaluated, strict=True
                    )
                }
            else:
                transaction = transaction.transition(TxnPhase.EVALUATING)
                semantic_events.append(self._phase_event(transaction))

            if deadline_ns is not None and self._clock_ns() >= deadline_ns:
                cache.rollback(cache_transaction)
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
            resolved_values: list[StateT | None] = list(lookup.values)
            for index, key, _candidate in missing:
                resolved_values[index] = staged[key]
            if any(item is None for item in resolved_values):
                cache.rollback(cache_transaction)
                raise RuntimeContractError("cache lookup could not resolve admitted order")
            resolved = tuple(item for item in resolved_values if item is not None)
            try:
                for evaluated_state in resolved:
                    oracle.validate(evaluated_state)
                    self._validated_state_digest(oracle, evaluated_state)
            except ValueError:
                cache.rollback(cache_transaction)
                transaction = transaction.transition(TxnPhase.ABORTED)
                semantic_events.append(self._phase_event(transaction))
                return self._result(
                    state=state,
                    objective=objective,
                    reason="validation_failure",
                    events=semantic_events,
                    config=config,
                    oracle=oracle,
                )
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
                cache.rollback(cache_transaction)
                transaction = transaction.transition(TxnPhase.ABORTED)
                semantic_events.append(self._phase_event(transaction))
                raise RuntimeContractError(
                    "kernel decision is neither the snapshot nor an evaluated state"
                )
            next_objective = oracle.objective(next_state)
            cache.stage(cache_transaction, staged)
            try:
                cache_receipt = cache.commit(cache_transaction)
            except StaleCacheSnapshotError:
                transaction = transaction.transition(TxnPhase.INTERRUPTED)
                semantic_events.append(self._phase_event(transaction))
                return self._result(
                    state=state,
                    objective=objective,
                    reason="stale_snapshot",
                    events=semantic_events,
                    config=config,
                    oracle=oracle,
                )
            except CacheCommitError:
                transaction = transaction.transition(TxnPhase.ABORTED)
                semantic_events.append(self._phase_event(transaction))
                return self._result(
                    state=state,
                    objective=objective,
                    reason="cache_write_failure",
                    events=semantic_events,
                    config=config,
                    oracle=oracle,
                )
            state = next_state
            state_digest = next_digest
            objective = next_objective
            transaction = transaction.transition(TxnPhase.COMMITTED)
            semantic_events.append(
                {
                    **self._phase_event(transaction),
                    "state_digest": state_digest,
                    "started_work": budget.started,
                    "staged_cache_writes": cache_receipt.writes,
                    "cache_generation": cache_receipt.generation_after,
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
                "runtime": "txnopt.python-reference-v1",
                "execution_mode": config.execution_mode,
                "oracle": oracle_identity,
            },
        )

    @staticmethod
    def _evaluate_candidates(
        *,
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
        candidates: tuple[CandidateT, ...],
        budget: BudgetLedger,
        config: RunConfig,
        deadline_ns: int | None,
    ) -> tuple[StateT, ...]:
        if config.execution_mode == "serial":
            budget.start(len(candidates))
            return tuple(
                oracle.evaluate_batch(
                    candidates,
                    work_budget=len(candidates),
                    deadline_ns=deadline_ns,
                )
            )
        if config.execution_mode == "barrier":
            return PythonTxnRuntime._evaluate_barrier(
                oracle=oracle,
                candidates=candidates,
                budget=budget,
                workers=config.workers,
                deadline_ns=deadline_ns,
            )
        return PythonTxnRuntime._evaluate_ordered(
            oracle=oracle,
            candidates=candidates,
            budget=budget,
            workers=config.workers,
            speculation_window=config.speculation_window,
            deadline_ns=deadline_ns,
        )

    @staticmethod
    def _evaluate_barrier(
        *,
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
        candidates: tuple[CandidateT, ...],
        budget: BudgetLedger,
        workers: int,
        deadline_ns: int | None,
    ) -> tuple[StateT, ...]:
        chunks = PythonTxnRuntime._contiguous_chunks(candidates, workers)
        executor = ThreadPoolExecutor(
            max_workers=len(chunks),
            thread_name_prefix="txnopt-barrier",
        )
        futures: list[Future[Sequence[StateT]]] = []
        try:
            for chunk in chunks:
                futures.append(
                    executor.submit(
                        oracle.evaluate_batch,
                        chunk,
                        work_budget=len(chunk),
                        deadline_ns=deadline_ns,
                    )
                )
                budget.start(len(chunk))
            ordered: list[StateT] = []
            for chunk, future in zip(chunks, futures, strict=True):
                values = tuple(future.result())
                if len(values) != len(chunk):
                    raise RuntimeContractError(
                        "parallel oracle chunk did not preserve its ordered shape"
                    )
                ordered.extend(values)
            return tuple(ordered)
        except Exception:
            for future in futures:
                future.cancel()
            budget.release_reserved()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _evaluate_ordered(
        *,
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
        candidates: tuple[CandidateT, ...],
        budget: BudgetLedger,
        workers: int,
        speculation_window: int,
        deadline_ns: int | None,
    ) -> tuple[StateT, ...]:
        width = min(workers, speculation_window, len(candidates))
        executor = ThreadPoolExecutor(
            max_workers=width,
            thread_name_prefix="txnopt-ordered",
        )
        pending: dict[int, Future[Sequence[StateT]]] = {}
        next_to_submit = 0

        def fill_window() -> None:
            nonlocal next_to_submit
            while next_to_submit < len(candidates) and len(pending) < width:
                index = next_to_submit
                pending[index] = executor.submit(
                    oracle.evaluate_batch,
                    (candidates[index],),
                    work_budget=1,
                    deadline_ns=deadline_ns,
                )
                budget.start(1)
                next_to_submit += 1

        try:
            fill_window()
            ordered: list[StateT] = []
            for index in range(len(candidates)):
                values = tuple(pending.pop(index).result())
                if len(values) != 1:
                    raise RuntimeContractError(
                        "ordered oracle result did not preserve its scalar shape"
                    )
                ordered.append(values[0])
                fill_window()
            return tuple(ordered)
        except Exception:
            for future in pending.values():
                future.cancel()
            budget.release_reserved()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _contiguous_chunks(
        candidates: tuple[CandidateT, ...],
        workers: int,
    ) -> tuple[tuple[CandidateT, ...], ...]:
        chunk_count = min(workers, len(candidates))
        quotient, remainder = divmod(len(candidates), chunk_count)
        chunks: list[tuple[CandidateT, ...]] = []
        offset = 0
        for index in range(chunk_count):
            size = quotient + (1 if index < remainder else 0)
            chunks.append(candidates[offset : offset + size])
            offset += size
        return tuple(chunks)


class SerialTxnRuntime[StateT, CandidateT, ObjectiveT](
    PythonTxnRuntime[StateT, CandidateT, ObjectiveT]
):
    """Compatibility name for the canonical Python reference implementation."""

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
        return super().run(
            initial_state,
            kernel=kernel,
            oracle=oracle,
            config=config,
        )
