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


class RuntimeContractError(RuntimeError):
    """Raised when a kernel or oracle violates the TxnOpt contract."""


class OracleWorkerError(RuntimeError):
    """Controlled worker failure that refines to the last commit prefix."""


class _EvaluationAbort(RuntimeError):
    """Private carrier for a detected evaluation boundary and its live work."""

    def __init__(
        self,
        cause: OracleWorkerError | TimeoutError,
        *,
        post_boundary_work_units: int,
        post_boundary_capacity_units: int,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.post_boundary_work_units = post_boundary_work_units
        self.post_boundary_capacity_units = post_boundary_capacity_units


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
        cache_factory: Callable[[], InMemoryCacheStore[StateT]] = InMemoryCacheStore,
        semantic_event_sink: (Callable[[tuple[Mapping[str, object], ...]], None] | None) = None,
        physical_event_sink: (Callable[[tuple[Mapping[str, object], ...]], None] | None) = None,
    ) -> None:
        self._clock_ns = clock_ns
        self._cache_factory = cache_factory
        self._semantic_event_sink = semantic_event_sink
        self._physical_event_sink = physical_event_sink
        self._physical_started_ns: int | None = None
        self._waste_audits: list[Mapping[str, object]] = []

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
        if config.trace_policy == "semantic_and_physical":
            if self._physical_event_sink is None:
                raise RuntimeContractError(
                    "semantic_and_physical tracing requires a physical event sink"
                )
            self._physical_started_ns = self._clock_ns()
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
            unique_candidates, candidate_keys = self._deduplicate(oracle, candidates)
            admitted_mask = tuple(oracle.screen(unique_candidates))
            if len(admitted_mask) != len(unique_candidates) or any(
                not isinstance(admitted, bool) for admitted in admitted_mask
            ):
                raise RuntimeContractError("oracle screening decisions must be ordered booleans")
            screened = tuple(
                candidate
                for candidate, keep in zip(unique_candidates, admitted_mask, strict=True)
                if keep
            )
            screened_keys = tuple(
                key for key, keep in zip(candidate_keys, admitted_mask, strict=True) if keep
            )
            admission_limit = len(screened)
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
            admitted = screened[:admission_limit]
            admitted_keys = screened_keys[:admission_limit]
            semantic_events.append(
                {
                    "event": "candidate_screening",
                    "proposed_count": len(candidates),
                    "unique_count": len(unique_candidates),
                    "screened_admissible_count": len(screened),
                    "admitted_count": len(admitted),
                    "admission_limit": admission_limit,
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
            missing_work = tuple(
                self._validated_work_units(oracle, candidate) for _index, _key, candidate in missing
            )
            remaining_before = budget.remaining
            started_before = budget.started
            reservation = budget.reserve(sum(missing_work))
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
                        candidates=tuple(candidate for _index, _key, candidate in missing),
                        work_units=missing_work,
                        budget=budget,
                        config=config,
                        deadline_ns=deadline_ns,
                    )
                except _EvaluationAbort as abort:
                    cache.rollback(cache_transaction)
                    if isinstance(abort.cause, OracleWorkerError):
                        phase = TxnPhase.ABORTED
                        reason = "worker_failure"
                    else:
                        if deadline_ns is None:
                            raise abort.cause from abort
                        phase = TxnPhase.INTERRUPTED
                        reason = "deadline"
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
                    return self._result(
                        state=state,
                        objective=objective,
                        reason=reason,
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
                cache.rollback(cache_transaction)
                raise RuntimeContractError("cache lookup could not resolve admitted order")
            resolved = tuple(item for item in resolved_values if item is not None)
            try:
                for evaluated_state in resolved:
                    oracle.validate(evaluated_state)
                    self._validated_state_digest(oracle, evaluated_state)
            except ValueError:
                cache.rollback(cache_transaction)
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
            allowed_digests.update(self._validated_state_digest(oracle, item) for item in resolved)
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
        final_events = [*events, {"event": "run_terminated", "reason": reason}]
        semantic_digest = hashlib.sha256(
            json.dumps(
                final_events,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
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
        physical_ref: str | None = None
        if config.trace_policy == "semantic_and_physical":
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
                **screening_statistics,
            }
            self._physical_event_sink((run_observation, *self._waste_audits))
            physical_ref = "txnopt-physical-trace-v1:external"
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
            except (OracleWorkerError, TimeoutError) as error:
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
        except (OracleWorkerError, TimeoutError) as error:
            pending_work = sum(
                sum(work_units[first:last])
                for (first, last), future in zip(chunks, futures, strict=True)
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
        except (OracleWorkerError, TimeoutError) as error:
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
