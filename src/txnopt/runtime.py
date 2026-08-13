"""Canonical serial TxnOpt runtime used as the Level 1 Python specification."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Final, Protocol, runtime_checkable

from txnopt._internal.budget import BudgetLedger
from txnopt._internal.cache import (
    CacheCommitError,
    CacheCommitReceipt,
    CacheTxn,
    InMemoryCacheStore,
    StaleCacheSnapshotError,
)
from txnopt._internal.candidate_txn import CandidateTxn, TxnPhase
from txnopt._internal.random_tape import RandomTape
from txnopt._internal.versions import CONTRACT_VERSION
from txnopt._internal.waste_bounds import WasteAudit, audit_waste
from txnopt.contracts import Oracle, RunConfig, RunResult, SearchKernel

_DIGEST: Final = re.compile(r"[0-9a-f]{64}")
_TAPE_WORDS_PER_ROUND: Final = 16
_SCREENING_BATCH_SIZE: Final = 64


class RuntimeContractError(RuntimeError):
    """Raised when a kernel or oracle violates the TxnOpt contract."""


class OracleWorkerError(RuntimeError):
    """Controlled worker failure that refines to the last commit prefix."""


class _EvaluationAbort(RuntimeError):
    """Private carrier for a detected evaluation boundary and its live work."""

    def __init__(
        self,
        cause: Exception,
        *,
        post_boundary_work_units: int,
        post_boundary_capacity_units: int,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.post_boundary_work_units = post_boundary_work_units
        self.post_boundary_capacity_units = post_boundary_capacity_units


class _PublishedCacheReceiptError(RuntimeContractError):
    """The cache commit is visible but its returned receipt violated the contract."""

    def __init__(self, receipt: CacheCommitReceipt, *, status: str) -> None:
        super().__init__("cache commit published with a malformed receipt")
        self.receipt = receipt
        self.status = status


class _UnknownCacheOutcomeError(RuntimeContractError):
    """The cache backend acknowledged commit but publication cannot be proved."""


@runtime_checkable
class _AdmissionLimitedKernel(Protocol):
    @property
    def admission_limit(self) -> int: ...


class PythonTxnRuntime[StateT, CandidateT, ObjectiveT]:
    """One-owner Python specification for serial, barrier, and ordered execution."""

    def __init__(
        self,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        cost_clock_ns: Callable[[], int] = time.perf_counter_ns,
        cache_factory: Callable[[], InMemoryCacheStore[StateT]] = InMemoryCacheStore,
        semantic_event_sink: (Callable[[tuple[Mapping[str, object], ...]], None] | None) = None,
        physical_event_sink: (Callable[[tuple[Mapping[str, object], ...]], None] | None) = None,
    ) -> None:
        self._clock_ns = clock_ns
        self._cost_clock_ns = cost_clock_ns
        self._cache_factory = cache_factory
        self._semantic_event_sink = semantic_event_sink
        self._physical_event_sink = physical_event_sink
        self._physical_started_ns: int | None = None
        self._waste_audits: list[Mapping[str, object]] = []
        self._oracle_physical_observations: list[Mapping[str, object]] = []
        self._observed_cmax_upper_ns = 0

    def run(
        self,
        initial_state: StateT,
        *,
        kernel: SearchKernel[StateT, CandidateT],
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
        config: RunConfig,
    ) -> RunResult[StateT, ObjectiveT]:
        self._physical_started_ns = None
        self._waste_audits = []
        self._oracle_physical_observations = []
        self._observed_cmax_upper_ns = 0
        if config.trace_policy == "semantic_and_physical":
            if self._physical_event_sink is None:
                raise RuntimeContractError(
                    "semantic_and_physical tracing requires a physical event sink"
                )
            self._physical_started_ns = self._clock_ns()
        self._collect_oracle_physical_observations(oracle, retain=False)
        if oracle.deterministic is not True:
            raise RuntimeContractError("oracle did not declare deterministic semantics")
        if config.execution_mode != "serial" and oracle.parallel_safe is not True:
            raise RuntimeContractError("oracle did not declare parallel-safe evaluation")
        if oracle.internal_parallelism:
            configured_workers = getattr(oracle, "worker_count", config.workers)
            if configured_workers != config.workers:
                raise RuntimeContractError(
                    "internally parallel oracle worker count differs from RunConfig"
                )

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
            admission_limit: int | None = None
            if isinstance(kernel, _AdmissionLimitedKernel):
                admission_limit = kernel.admission_limit
                if (
                    isinstance(admission_limit, bool)
                    or not isinstance(admission_limit, int)
                    or admission_limit <= 0
                ):
                    raise RuntimeContractError(
                        "kernel admission_limit must be a positive integer"
                    )
            (
                admitted,
                admitted_keys,
                screened_candidate_count,
                unique_count,
                screened_admissible_count,
                screening_complete,
            ) = self._admit_candidates(
                oracle,
                candidates,
                admission_limit=admission_limit,
            )
            reported_admission_limit = (
                screened_admissible_count
                if admission_limit is None
                else admission_limit
            )
            semantic_events.append(
                {
                    "event": "candidate_screening",
                    "proposed_count": len(candidates),
                    "screened_candidate_count": screened_candidate_count,
                    "unique_count": unique_count,
                    "screened_admissible_count": screened_admissible_count,
                    "admitted_count": len(admitted),
                    "admission_limit": reported_admission_limit,
                    "screening_complete": screening_complete,
                }
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
            try:
                cache_transaction = cache.begin()
            except Exception as error:
                transaction = transaction.transition(TxnPhase.ABORTED)
                semantic_events.append(self._phase_event(transaction))
                self._emit_failed_run(
                    reason=self._exception_reason(error),
                    events=semantic_events,
                    config=config,
                    oracle=oracle,
                )
                raise
            try:
                lookup = cache.lookup(admitted_keys)
            except Exception as error:
                self._rollback_preserving_error(cache, cache_transaction, error)
                transaction = transaction.transition(TxnPhase.ABORTED)
                semantic_events.append(self._phase_event(transaction))
                self._emit_failed_run(
                    reason=self._exception_reason(error),
                    events=semantic_events,
                    config=config,
                    oracle=oracle,
                )
                raise
            if lookup.generation != cache_transaction.expected_generation:
                try:
                    self._rollback_or_raise(cache, cache_transaction)
                except Exception as error:
                    transaction = transaction.transition(TxnPhase.ABORTED)
                    semantic_events.append(self._phase_event(transaction))
                    self._emit_failed_run(
                        reason=self._exception_reason(error),
                        events=semantic_events,
                        config=config,
                        oracle=oracle,
                    )
                    raise
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
            try:
                missing = tuple(
                    (index, key, candidate)
                    for index, (key, candidate, cached) in enumerate(
                        zip(admitted_keys, admitted, lookup.values, strict=True)
                    )
                    if cached is None
                )
                missing_work = tuple(
                    self._validated_work_units(oracle, candidate)
                    for _index, _key, candidate in missing
                )
                remaining_before = budget.remaining
                started_before = budget.started
                reservation = budget.reserve(sum(missing_work))
            except Exception as error:
                self._rollback_preserving_error(cache, cache_transaction, error)
                budget.release_reserved()
                transaction = transaction.transition(TxnPhase.ABORTED)
                semantic_events.append(self._phase_event(transaction))
                self._emit_failed_run(
                    reason=self._exception_reason(error),
                    events=semantic_events,
                    config=config,
                    oracle=oracle,
                )
                raise
            if not reservation.granted:
                try:
                    self._rollback_or_raise(cache, cache_transaction)
                except Exception as error:
                    transaction = transaction.transition(TxnPhase.ABORTED)
                    semantic_events.append(self._phase_event(transaction))
                    self._emit_failed_run(
                        reason=self._exception_reason(error),
                        events=semantic_events,
                        config=config,
                        oracle=oracle,
                    )
                    raise
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
            try:
                staged: dict[str, StateT] = {}
                if missing:
                    transaction = transaction.transition(TxnPhase.EVALUATING)
                    semantic_events.append(self._phase_event(transaction))
                    evaluation_started_ns = self._cost_clock_ns()
                    try:
                        try:
                            evaluated = self._evaluate_candidates(
                                oracle=oracle,
                                candidates=tuple(
                                    candidate for _index, _key, candidate in missing
                                ),
                                work_units=missing_work,
                                budget=budget,
                                config=config,
                                deadline_ns=deadline_ns,
                            )
                        except Exception as evaluation_error:
                            try:
                                self._collect_oracle_physical_observations(
                                    oracle,
                                    retain=config.trace_policy
                                    == "semantic_and_physical",
                                )
                            except Exception as observation_error:
                                evaluation_error.add_note(
                                    "secondary physical observation failure: "
                                    f"{observation_error}"
                                )
                            raise
                        else:
                            self._collect_oracle_physical_observations(
                                oracle,
                                retain=config.trace_policy
                                == "semantic_and_physical",
                            )
                        finally:
                            elapsed_ns = self._cost_clock_ns() - evaluation_started_ns
                            self._observed_cmax_upper_ns = max(
                                self._observed_cmax_upper_ns,
                                1 if elapsed_ns <= 0 else elapsed_ns,
                            )
                    except _EvaluationAbort as abort:
                        cleanup_error = self._rollback_preserving_error(
                            cache,
                            cache_transaction,
                            abort.cause,
                        )
                        if isinstance(abort.cause, TimeoutError) and deadline_ns is None:
                            timeout_error = RuntimeContractError(
                                "oracle raised TimeoutError without a configured deadline"
                            )
                            timeout_error.__cause__ = abort
                            terminal_error: Exception | None = timeout_error
                            phase = TxnPhase.ABORTED
                            reason = self._exception_reason(timeout_error)
                        elif isinstance(abort.cause, OracleWorkerError):
                            terminal_error = None
                            phase = TxnPhase.ABORTED
                            reason = "worker_failure"
                        elif isinstance(abort.cause, TimeoutError):
                            terminal_error = None
                            phase = TxnPhase.INTERRUPTED
                            reason = "deadline"
                        else:
                            terminal_error = abort.cause
                            phase = TxnPhase.ABORTED
                            reason = self._exception_reason(terminal_error)
                        if terminal_error is None and cleanup_error is not None:
                            terminal_error = cleanup_error
                            phase = TxnPhase.ABORTED
                            reason = self._exception_reason(terminal_error)
                        self._record_waste_audit(
                            reason=reason,
                            remaining_budget_before=remaining_before,
                            work_units=missing_work,
                            observed_discarded_work_units=budget.started - started_before,
                            observed_post_boundary_work_units=(
                                abort.post_boundary_work_units
                            ),
                            post_boundary_capacity_units=(
                                abort.post_boundary_capacity_units
                            ),
                        )
                        transaction = transaction.transition(phase)
                        semantic_events.append(self._phase_event(transaction))
                        if terminal_error is not None:
                            self._emit_failed_run(
                                reason=reason,
                                events=semantic_events,
                                config=config,
                                oracle=oracle,
                            )
                            raise terminal_error from abort
                        return self._result(
                            state=state,
                            objective=objective,
                            reason=reason,
                            events=semantic_events,
                            config=config,
                            oracle=oracle,
                        )
                    if len(evaluated) != len(missing):
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
                    self._rollback_or_raise(cache, cache_transaction)
                    self._record_waste_audit(
                        reason="deadline",
                        remaining_budget_before=remaining_before,
                        work_units=missing_work,
                        observed_discarded_work_units=budget.started - started_before,
                        observed_post_boundary_work_units=0,
                        post_boundary_capacity_units=0,
                    )
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
                    raise RuntimeContractError("cache lookup could not resolve admitted order")
                resolved = tuple(item for item in resolved_values if item is not None)
                try:
                    for evaluated_state in resolved:
                        oracle.validate(evaluated_state)
                        self._validated_state_digest(oracle, evaluated_state)
                except ValueError:
                    self._rollback_or_raise(cache, cache_transaction)
                    self._record_waste_audit(
                        reason="validation_failure",
                        remaining_budget_before=remaining_before,
                        work_units=missing_work,
                        observed_discarded_work_units=budget.started - started_before,
                        observed_post_boundary_work_units=0,
                        post_boundary_capacity_units=0,
                    )
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
                    raise RuntimeContractError(
                        "kernel decision is neither the snapshot nor an evaluated state"
                    )
                next_objective = oracle.objective(next_state)
                cache.stage(cache_transaction, staged)
                try:
                    raw_cache_receipt = cache.commit(cache_transaction)
                    cache_receipt = self._validated_cache_commit_receipt(
                        cache=cache,
                        transaction=cache_transaction,
                        raw_receipt=raw_cache_receipt,
                        staged=staged,
                    )
                except _PublishedCacheReceiptError as error:
                    cache_receipt = error.receipt
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
                            "cache_receipt_status": error.status,
                        }
                    )
                    self._emit_failed_run(
                        reason="runtime_contract_error",
                        events=semantic_events,
                        config=config,
                        oracle=oracle,
                    )
                    raise
                except _UnknownCacheOutcomeError as error:
                    budget.release_reserved()
                    self._record_waste_audit(
                        reason="cache_outcome_unknown",
                        remaining_budget_before=remaining_before,
                        work_units=missing_work,
                        observed_discarded_work_units=budget.started - started_before,
                        observed_post_boundary_work_units=0,
                        post_boundary_capacity_units=0,
                    )
                    transaction = transaction.transition(TxnPhase.INTERRUPTED)
                    semantic_events.append(
                        {
                            **self._phase_event(transaction),
                            "cache_outcome": "unknown",
                        }
                    )
                    self._emit_failed_run(
                        reason="cache_outcome_unknown",
                        events=semantic_events,
                        config=config,
                        oracle=oracle,
                    )
                    raise error
                except StaleCacheSnapshotError:
                    self._rollback_or_raise(cache, cache_transaction)
                    self._record_waste_audit(
                        reason="stale_snapshot",
                        remaining_budget_before=remaining_before,
                        work_units=missing_work,
                        observed_discarded_work_units=budget.started - started_before,
                        observed_post_boundary_work_units=0,
                        post_boundary_capacity_units=0,
                    )
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
                    self._rollback_or_raise(cache, cache_transaction)
                    self._record_waste_audit(
                        reason="cache_write_failure",
                        remaining_budget_before=remaining_before,
                        work_units=missing_work,
                        observed_discarded_work_units=budget.started - started_before,
                        observed_post_boundary_work_units=0,
                        post_boundary_capacity_units=0,
                    )
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
            except Exception as error:
                if transaction.terminal:
                    raise
                self._rollback_preserving_error(cache, cache_transaction, error)
                budget.release_reserved()
                reason = self._exception_reason(error)
                self._record_waste_audit(
                    reason=reason,
                    remaining_budget_before=remaining_before,
                    work_units=missing_work,
                    observed_discarded_work_units=budget.started - started_before,
                    observed_post_boundary_work_units=0,
                    post_boundary_capacity_units=0,
                )
                if not transaction.terminal:
                    transaction = transaction.transition(TxnPhase.ABORTED)
                    semantic_events.append(self._phase_event(transaction))
                self._emit_failed_run(
                    reason=reason,
                    events=semantic_events,
                    config=config,
                    oracle=oracle,
                )
                raise
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
    def _exception_reason(error: Exception) -> str:
        return (
            "runtime_contract_error"
            if isinstance(error, (RuntimeContractError, ValueError, TypeError))
            else "runtime_exception"
        )

    @staticmethod
    def _rollback_or_raise(
        cache: InMemoryCacheStore[StateT],
        transaction: CacheTxn[StateT],
    ) -> None:
        try:
            cache.rollback(transaction)
        except Exception as error:
            transaction.staged.clear()
            transaction.closed = True
            raise CacheCommitError("cache rollback failed") from error

    @staticmethod
    def _validated_cache_commit_receipt(
        *,
        cache: InMemoryCacheStore[StateT],
        transaction: CacheTxn[StateT],
        raw_receipt: object,
        staged: Mapping[str, StateT],
    ) -> CacheCommitReceipt:
        expected_before = transaction.expected_generation
        expected_after = expected_before + 1
        raw_receipt_valid = (
            isinstance(raw_receipt, CacheCommitReceipt)
            and raw_receipt.generation_before == expected_before
            and raw_receipt.generation_after == expected_after
            and raw_receipt.writes == len(staged)
        )
        reconstructed = CacheCommitReceipt(
            generation_before=expected_before,
            generation_after=expected_after,
            writes=len(staged),
        )
        try:
            generation, values = cache.snapshot()
        except Exception as error:
            if transaction.closed:
                unknown_error = _UnknownCacheOutcomeError(
                    "cache commit outcome cannot be verified after snapshot failure"
                )
                unknown_error.add_note(f"cache snapshot verification failed: {error}")
                raise unknown_error from error
            raise RuntimeContractError(
                "cache commit returned a malformed receipt with an unknown outcome"
            ) from error
        snapshot_matches = generation == expected_after and all(
            values.get(key) == value for key, value in staged.items()
        )
        if raw_receipt_valid and transaction.closed and snapshot_matches:
            return reconstructed
        if snapshot_matches:
            transaction.closed = True
            transaction.staged.clear()
            raise _PublishedCacheReceiptError(
                reconstructed,
                status="reconstructed_after_contract_failure",
            )
        if transaction.closed:
            raise _UnknownCacheOutcomeError(
                "cache commit acknowledgement differs from the published snapshot"
            )
        raise RuntimeContractError(
            "cache commit returned a malformed receipt with an unverified outcome"
        )

    @classmethod
    def _rollback_preserving_error(
        cls,
        cache: InMemoryCacheStore[StateT],
        transaction: CacheTxn[StateT],
        original: Exception,
    ) -> CacheCommitError | None:
        try:
            cls._rollback_or_raise(cache, transaction)
        except CacheCommitError as cleanup_error:
            original.add_note(f"secondary cleanup failure: {cleanup_error}")
            return cleanup_error
        return None

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
    def _admit_candidates(
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
        candidates: Sequence[CandidateT],
        *,
        admission_limit: int | None,
    ) -> tuple[
        tuple[CandidateT, ...],
        tuple[str, ...],
        int,
        int,
        int,
        bool,
    ]:
        """Return the first canonical admissible prefix without screening its tail.

        ``Oracle.screen`` is required to be partition invariant, so splitting the
        canonical candidate stream into bounded batches cannot change a decision.
        This keeps the admission policy exact while avoiding work on candidates
        that cannot enter the current atomic transaction.
        """

        admitted: list[CandidateT] = []
        admitted_keys: list[str] = []
        seen: set[str] = set()
        position = 0
        screened_candidate_count = 0
        screened_admissible_count = 0
        batch_size = (
            _SCREENING_BATCH_SIZE
            if admission_limit is None
            else min(_SCREENING_BATCH_SIZE, admission_limit)
        )
        while position < len(candidates) and (
            admission_limit is None or len(admitted) < admission_limit
        ):
            batch: list[CandidateT] = []
            batch_keys: list[str] = []
            while position < len(candidates) and len(batch) < batch_size:
                candidate = candidates[position]
                position += 1
                key = oracle.stable_key(candidate)
                if not key:
                    raise RuntimeContractError("oracle stable_key cannot be empty")
                if key in seen:
                    continue
                seen.add(key)
                batch.append(candidate)
                batch_keys.append(key)
            if not batch:
                continue
            decisions = tuple(oracle.screen(batch))
            if len(decisions) != len(batch) or any(
                not isinstance(decision, bool) for decision in decisions
            ):
                raise RuntimeContractError(
                    "oracle screening decisions must be ordered booleans"
                )
            screened_candidate_count += len(batch)
            screened_admissible_count += sum(decisions)
            for candidate, key, keep in zip(
                batch,
                batch_keys,
                decisions,
                strict=True,
            ):
                if not keep:
                    continue
                if admission_limit is not None and len(admitted) >= admission_limit:
                    break
                admitted.append(candidate)
                admitted_keys.append(key)
        if not seen:
            raise RuntimeContractError("candidate deduplication removed the entire round")
        return (
            tuple(admitted),
            tuple(admitted_keys),
            screened_candidate_count,
            len(seen),
            screened_admissible_count,
            position == len(candidates),
        )

    @staticmethod
    def _phase_event(transaction: CandidateTxn) -> Mapping[str, object]:
        return {
            "event": "candidate_transaction",
            "txn_id": transaction.txn_id,
            "phase": transaction.phase.value,
            "snapshot_digest": transaction.snapshot_digest,
            "candidate_keys": transaction.candidate_keys,
        }

    def _result(
        self,
        *,
        state: StateT,
        objective: ObjectiveT,
        reason: str,
        events: Sequence[Mapping[str, object]],
        config: RunConfig,
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
    ) -> RunResult[StateT, ObjectiveT]:
        final_events = self._emit_semantic_trace(events=events, reason=reason)
        semantic_digest = hashlib.sha256(
            json.dumps(
                final_events,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        physical_ref = self._emit_physical_trace(
            reason=reason,
            config=config,
            oracle=oracle,
        )
        oracle_identity = f"{type(oracle).__module__}.{type(oracle).__qualname__}"
        return RunResult(
            last_committed_state=state,
            objective=objective,
            termination_reason=reason,
            semantic_digest=semantic_digest,
            physical_artifact_ref=physical_ref,
            provenance={
                "contract": CONTRACT_VERSION,
                "runtime": "txnopt.python-reference-v1",
                "execution_mode": config.execution_mode,
                "oracle": oracle_identity,
            },
        )

    def _emit_failed_run(
        self,
        *,
        reason: str,
        events: Sequence[Mapping[str, object]],
        config: RunConfig,
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
    ) -> None:
        """Seal trace evidence for a fail-fast contract/runtime exception."""

        self._emit_semantic_trace(events=events, reason=reason)
        self._emit_physical_trace(reason=reason, config=config, oracle=oracle)

    def _emit_semantic_trace(
        self,
        *,
        events: Sequence[Mapping[str, object]],
        reason: str,
    ) -> list[Mapping[str, object]]:
        final_events = [*events, {"event": "run_terminated", "reason": reason}]
        if self._semantic_event_sink is not None:
            detached = tuple(
                json.loads(
                    json.dumps(
                        event,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
                for event in final_events
            )
            self._semantic_event_sink(detached)
        return final_events

    def _collect_oracle_physical_observations(
        self,
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
        *,
        retain: bool,
    ) -> None:
        """Drain an optional adapter-local physical seam without changing Oracle."""

        drain = getattr(oracle, "drain_physical_observations", None)
        if drain is None:
            return
        if not callable(drain):
            raise RuntimeContractError(
                "oracle drain_physical_observations must be callable"
            )
        raw_observations = drain()
        if isinstance(raw_observations, (str, bytes)) or not isinstance(
            raw_observations, Sequence
        ):
            raise RuntimeContractError(
                "oracle physical observations must be an ordered sequence"
            )
        detached: list[Mapping[str, object]] = []
        for raw in raw_observations:
            if not isinstance(raw, Mapping):
                raise RuntimeContractError(
                    "oracle physical observation must be a mapping"
                )
            try:
                event = json.loads(
                    json.dumps(
                        dict(raw),
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
            except (TypeError, ValueError) as error:
                raise RuntimeContractError(
                    "oracle physical observation is not canonical JSON"
                ) from error
            if (
                not isinstance(event, dict)
                or not isinstance(event.get("event"), str)
                or not event["event"]
                or event["event"] in {"run_observation", "t4_waste_observation"}
                or event.get("trace") != "txnopt-physical-trace-v1"
            ):
                raise RuntimeContractError(
                    "oracle physical observation uses a reserved or invalid identity"
                )
            detached.append(event)
        if retain:
            self._oracle_physical_observations.extend(detached)

    def _emit_physical_trace(
        self,
        *,
        reason: str,
        config: RunConfig,
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
    ) -> str | None:
        if config.trace_policy != "semantic_and_physical":
            return None
        if self._physical_started_ns is None or self._physical_event_sink is None:
            raise RuntimeContractError("physical trace owner is not initialized")
        screening_statistics = getattr(oracle, "screening_statistics", {})
        if not isinstance(screening_statistics, Mapping) or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in screening_statistics.items()
        ):
            raise RuntimeContractError(
                "oracle screening_statistics must contain non-negative integers"
            )
        ended_ns = self._clock_ns()
        run_observation: Mapping[str, object] = {
            "event": "run_observation",
            "trace": "txnopt-physical-trace-v1",
            "execution_mode": config.execution_mode,
            "workers": config.workers,
            "started_ns": self._physical_started_ns,
            "ended_ns": ended_ns,
            "duration_ns": ended_ns - self._physical_started_ns,
            "termination_reason": reason,
            "observed_cmax_upper_ns": self._observed_cmax_upper_ns,
            **screening_statistics,
        }
        waste_audits = tuple(self._waste_audit_with_cost(event) for event in self._waste_audits)
        self._physical_event_sink(
            (
                run_observation,
                *self._oracle_physical_observations,
                *waste_audits,
            )
        )
        return "txnopt-physical-trace-v1:external"

    def _waste_audit_with_cost(
        self,
        event: Mapping[str, object],
    ) -> Mapping[str, object]:
        cmax = self._observed_cmax_upper_ns
        discarded = event.get("observed_discarded_work_units")
        discarded_bound = event.get("discarded_work_bound_units")
        post_boundary = event.get("observed_post_boundary_work_units")
        post_boundary_bound = event.get("post_boundary_work_bound_units")
        if (
            cmax <= 0
            or not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (
                    discarded,
                    discarded_bound,
                    post_boundary,
                    post_boundary_bound,
                )
            )
        ):
            return event
        assert isinstance(discarded, int)
        assert isinstance(discarded_bound, int)
        assert isinstance(post_boundary, int)
        assert isinstance(post_boundary_bound, int)
        return {
            **event,
            "cost_basis": "measured_transaction_elapsed_upper_bound_ns",
            "measured_cmax_ns": cmax,
            "observed_discarded_cost_upper_ns": discarded * cmax,
            "discarded_cost_bound_ns": discarded_bound * cmax,
            "observed_post_boundary_cost_upper_ns": post_boundary * cmax,
            "post_boundary_cost_bound_ns": post_boundary_bound * cmax,
        }

    def _record_waste_audit(
        self,
        *,
        reason: str,
        remaining_budget_before: int | None,
        work_units: tuple[int, ...],
        observed_discarded_work_units: int,
        observed_post_boundary_work_units: int,
        post_boundary_capacity_units: int,
    ) -> None:
        """Bind a fixed-work abort observation to the atomic transaction window."""

        if remaining_budget_before is None or not work_units:
            return
        audit: WasteAudit = audit_waste(
            remaining_budget_before=remaining_budget_before,
            uncommitted_window=len(work_units),
            max_requests_per_candidate=max(work_units),
            post_boundary_capacity_units=post_boundary_capacity_units,
            observed_discarded_work_units=observed_discarded_work_units,
            observed_post_boundary_work_units=observed_post_boundary_work_units,
        )
        self._waste_audits.append(
            {
                "event": "t4_waste_observation",
                "trace": "txnopt-physical-trace-v1",
                "reason": reason,
                "cost_basis": "normalized_work_unit; measured_Cmax_required",
                "remaining_budget_before": audit.remaining_budget_before,
                "uncommitted_window": audit.uncommitted_window,
                "max_requests_per_candidate": audit.max_requests_per_candidate,
                "post_boundary_capacity_units": audit.post_boundary_capacity_units,
                "observed_discarded_work_units": (
                    audit.observed_discarded_work_units
                ),
                "observed_post_boundary_work_units": (
                    audit.observed_post_boundary_work_units
                ),
                "discarded_work_bound_units": audit.bound.discarded_work_units,
                "post_boundary_work_bound_units": (
                    audit.bound.post_boundary_work_units
                ),
                "bound_satisfied": True,
            }
        )

    @staticmethod
    def _evaluate_candidates(
        *,
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
        candidates: tuple[CandidateT, ...],
        work_units: tuple[int, ...],
        budget: BudgetLedger,
        config: RunConfig,
        deadline_ns: int | None,
    ) -> tuple[StateT, ...]:
        if config.execution_mode == "serial" or oracle.internal_parallelism:
            work = sum(work_units)
            budget.start(work)
            try:
                return tuple(
                    oracle.evaluate_batch(
                        candidates,
                        work_budget=work,
                        deadline_ns=deadline_ns,
                    )
                )
            except Exception as error:
                budget.release_reserved()
                raise _EvaluationAbort(
                    error,
                    post_boundary_work_units=0,
                    post_boundary_capacity_units=0,
                ) from error
        if config.execution_mode == "barrier":
            return PythonTxnRuntime._evaluate_barrier(
                oracle=oracle,
                candidates=candidates,
                work_units=work_units,
                budget=budget,
                workers=config.workers,
                deadline_ns=deadline_ns,
            )
        return PythonTxnRuntime._evaluate_ordered(
            oracle=oracle,
            candidates=candidates,
            work_units=work_units,
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
        work_units: tuple[int, ...],
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
                first, last = chunk
                candidate_chunk = candidates[first:last]
                chunk_work = sum(work_units[first:last])
                futures.append(
                    executor.submit(
                        oracle.evaluate_batch,
                        candidate_chunk,
                        work_budget=chunk_work,
                        deadline_ns=deadline_ns,
                    )
                )
                budget.start(chunk_work)
            ordered: list[StateT] = []
            for chunk, future in zip(chunks, futures, strict=True):
                first, last = chunk
                values = tuple(future.result())
                if len(values) != last - first:
                    raise RuntimeContractError(
                        "parallel oracle chunk did not preserve its ordered shape"
                    )
                ordered.extend(values)
            return tuple(ordered)
        except Exception as error:
            pending_work = sum(
                sum(work_units[chunks[index][0] : chunks[index][1]])
                for index, future in enumerate(futures)
                if not future.done()
            )
            for future in futures:
                future.cancel()
            budget.release_reserved()
            raise _EvaluationAbort(
                error,
                post_boundary_work_units=pending_work,
                post_boundary_capacity_units=sum(work_units),
            ) from error
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _evaluate_ordered(
        *,
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
        candidates: tuple[CandidateT, ...],
        work_units: tuple[int, ...],
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
                    work_budget=work_units[index],
                    deadline_ns=deadline_ns,
                )
                budget.start(work_units[index])
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
        except Exception as error:
            pending_work = sum(
                work_units[index]
                for index, future in pending.items()
                if not future.done()
            )
            for future in pending.values():
                future.cancel()
            budget.release_reserved()
            raise _EvaluationAbort(
                error,
                post_boundary_work_units=pending_work,
                post_boundary_capacity_units=width * max(work_units),
            ) from error
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _contiguous_chunks(
        candidates: tuple[CandidateT, ...],
        workers: int,
    ) -> tuple[tuple[int, int], ...]:
        chunk_count = min(workers, len(candidates))
        quotient, remainder = divmod(len(candidates), chunk_count)
        chunks: list[tuple[int, int]] = []
        offset = 0
        for index in range(chunk_count):
            size = quotient + (1 if index < remainder else 0)
            chunks.append((offset, offset + size))
            offset += size
        return tuple(chunks)

    @staticmethod
    def _validated_work_units(
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
        candidate: CandidateT,
    ) -> int:
        work = oracle.work_units(candidate)
        if isinstance(work, bool) or not isinstance(work, int) or work <= 0:
            raise RuntimeContractError("oracle work_units must be a positive integer")
        return work


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
            raise RuntimeContractError("SerialTxnRuntime supports only the serial execution mode")
        return super().run(
            initial_state,
            kernel=kernel,
            oracle=oracle,
            config=config,
        )
