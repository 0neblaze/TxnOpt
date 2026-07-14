"""Deterministic Stage 3.4 candidate control and ordered CPU parallelism."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Literal

from evrptw.charging import ChargingSubproblemResult
from evrptw.cpu_batch import (
    BackendMetrics,
    BatchChargingResult,
    ExactBatchDeadlineExceeded,
    ExactChargingBackend,
    solve_exact_charging_batch,
)
from evrptw.models import Instance

CANDIDATE_CONTROL_SCHEMA_VERSION = "stage034-control-parallel-v1"
ExecutorModel = Literal["process_spawn"]


@dataclass(frozen=True, slots=True)
class CandidateControlConfig:
    """Opt-in controls for Stage 3.4 candidate ranking and exact work."""

    enabled: bool = True
    schema_version: str = CANDIDATE_CONTROL_SCHEMA_VERSION
    proposal_top_k: int = 1
    max_exact_calls_per_round: int = 1
    worker_count: int = 1
    executor_model: ExecutorModel = "process_spawn"
    ranking_policy: str = "vehicle_distance_changed_routes_route_key_ordinal"
    merge_policy: str = "submission_order"

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_CONTROL_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Stage 3.4 candidate-control schema {self.schema_version}"
            )
        if self.proposal_top_k <= 0:
            raise ValueError("proposal_top_k must be positive")
        if self.max_exact_calls_per_round <= 0:
            raise ValueError("max_exact_calls_per_round must be positive")
        if self.worker_count not in {1, 4}:
            raise ValueError("Stage 3.4 worker_count must be 1 or 4")
        if self.executor_model != "process_spawn":
            raise ValueError("Stage 3.4 requires executor_model=process_spawn")
        if self.merge_policy != "submission_order":
            raise ValueError("Stage 3.4 requires deterministic submission-order merge")


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    """A complete, rankable candidate without embedded exact results."""

    candidate_id: int
    customer_sequences: tuple[tuple[str, ...], ...]
    vehicle_count: int
    optimistic_total_distance: float
    changed_route_count: int
    proposal_ordinal: int

    @property
    def rank_key(self) -> tuple[object, ...]:
        return (
            self.vehicle_count,
            self.optimistic_total_distance,
            self.changed_route_count,
            self.customer_sequences,
            self.proposal_ordinal,
        )


class CandidateParallelExecutionError(RuntimeError):
    """Parallel exact work failed and must not fall back to another backend."""


@dataclass(slots=True)
class CandidateControlRuntime:
    """Process-local shared budget, evidence and worker-pool owner."""

    config: CandidateControlConfig
    events: list[dict[str, object]] = field(default_factory=list)
    _round_key: tuple[str, int] | None = None
    _round_used: int = 0
    _pool: ProcessPoolExecutor | None = field(default=None, init=False, repr=False)
    _submission_serial: int = 0
    _candidate_work: list[dict[str, object]] = field(default_factory=list)
    _route_results: list[dict[str, object]] = field(default_factory=list)
    _attempted_plans: set[tuple[tuple[str, ...], ...]] = field(default_factory=set)

    def begin_round(self, iteration: int, *, lane: str = "all") -> None:
        key = (lane, iteration)
        if self._round_key == key:
            return
        self.finish_round()
        self._round_key = key
        self._round_used = 0
        self.events.append(
            {
                "event_type": "candidate_control_round",
                "status": "started",
                "lane": lane,
                "iteration": iteration,
                "budget": self.config.max_exact_calls_per_round,
            }
        )

    def finish_round(self) -> None:
        if self._round_key is None:
            return
        self.events.append(
            {
                "event_type": "candidate_control_round",
                "status": "completed",
                "lane": self._round_key[0],
                "iteration": self._round_key[1],
                "budget": self.config.max_exact_calls_per_round,
                "used": self._round_used,
                "remainder": self.round_remaining,
            }
        )
        self._round_key = None
        self._round_used = 0

    @property
    def round_remaining(self) -> int:
        if self._round_key is None:
            return self.config.max_exact_calls_per_round
        return max(0, self.config.max_exact_calls_per_round - self._round_used)

    def reserve(self, requested: int, *, atomic: bool, context: str) -> int:
        if requested <= 0:
            return 0
        # Initialization is controlled by ranking but not by one ALNS-iteration
        # budget; otherwise a multi-route feasible incumbent could not be built.
        if self._round_key is None:
            granted = requested
        elif atomic:
            granted = requested if requested <= self.round_remaining else 0
        else:
            granted = min(requested, self.round_remaining)
        self._round_used += granted
        self.events.append(
            {
                "event_type": "candidate_control_budget",
                "status": "reserved" if granted == requested else "budget_skipped",
                "context": context,
                "requested": requested,
                "granted": granted,
                "remaining": self.round_remaining,
                "iteration": None if self._round_key is None else self._round_key[1],
            }
        )
        return granted

    def select_route_candidates(
        self,
        candidates: Sequence[tuple[int, tuple[str, ...], float]],
        *,
        lane: str,
        iteration: int | None,
        operator: str,
    ) -> tuple[int, ...]:
        ordered = sorted(candidates, key=lambda item: (item[2], item[1], item[0]))
        selected = tuple(item[0] for item in ordered[: self.config.proposal_top_k])
        selected_set = set(selected)
        skipped_payload: list[dict[str, object]] = []
        for rank, (index, sequence, lower_bound) in enumerate(ordered, start=1):
            if index not in selected_set:
                skipped_payload.append(
                    {
                        "index": index,
                        "rank": rank,
                        "sequence": list(sequence),
                        "distance_lower_bound": lower_bound,
                    }
                )
                continue
            self.events.append(
                {
                    "event_type": "candidate_control_decision",
                    "status": "selected",
                    "lane": lane,
                    "iteration": iteration,
                    "operator": operator,
                    "candidate_index": index,
                    "rank": rank,
                    "distance_lower_bound": lower_bound,
                    "customer_sequence": list(sequence),
                }
            )
        if skipped_payload:
            self.events.append(
                {
                    "event_type": "candidate_control_decision_aggregate",
                    "status": "not_selected",
                    "reason": "proposal_top_k",
                    "lane": lane,
                    "iteration": iteration,
                    "operator": operator,
                    "aggregate_count": len(skipped_payload),
                    "first_rank": self.config.proposal_top_k + 1,
                    "last_rank": len(ordered),
                    "candidate_pool_hash": _stable_hash(skipped_payload),
                }
            )
        return selected

    def select_plans(
        self,
        plans: Sequence[CandidatePlan],
        *,
        lane: str,
        iteration: int,
        operator: str,
    ) -> tuple[CandidatePlan, ...]:
        """Select never-attempted complete plans by the formal rank key."""

        ordered = sorted(plans, key=lambda plan: plan.rank_key)
        available = [
            plan
            for plan in ordered
            if plan.customer_sequences not in self._attempted_plans
        ]
        selected = tuple(available[: self.config.proposal_top_k])
        selected_ids = {plan.candidate_id for plan in selected}
        for rank, plan in enumerate(ordered, start=1):
            repeated = plan.customer_sequences in self._attempted_plans
            status = (
                "already_attempted"
                if repeated
                else "selected"
                if plan.candidate_id in selected_ids
                else "not_selected"
            )
            self.events.append(
                {
                    "event_type": "candidate_plan_decision",
                    "status": status,
                    "lane": lane,
                    "iteration": iteration,
                    "operator": operator,
                    "candidate_id": plan.candidate_id,
                    "rank": rank,
                    "vehicle_count": plan.vehicle_count,
                    "optimistic_total_distance": plan.optimistic_total_distance,
                    "changed_route_count": plan.changed_route_count,
                    "customer_sequences": [
                        list(sequence) for sequence in plan.customer_sequences
                    ],
                    "proposal_ordinal": plan.proposal_ordinal,
                }
            )
        self._attempted_plans.update(plan.customer_sequences for plan in selected)
        return selected

    def solve_batch(
        self,
        instance: Instance,
        sequences: tuple[tuple[str, ...], ...],
        *,
        batch_size: int,
        deadline: float,
        lane: str,
        iteration: int | None,
        operator: str,
    ) -> BatchChargingResult:
        if not sequences:
            return BatchChargingResult((), BackendMetrics("cpu_batch", batch_size))
        self._candidate_work.append(
            {
                "lane": lane,
                "iteration": iteration,
                "operator": operator,
                "sequences": [list(sequence) for sequence in sequences],
            }
        )
        if self.config.worker_count == 1 or len(sequences) == 1:
            result = solve_exact_charging_batch(
                instance,
                sequences,
                backend=ExactChargingBackend.CPU_BATCH,
                batch_size=batch_size,
                deadline=deadline,
            )
            self._record_results(sequences, result.results)
            self.events.append(
                {
                    "event_type": "parallel_batch",
                    "status": "serial_complete",
                    "worker_count": 1,
                    "submission_order": [0],
                    "completion_order": [0],
                    "merge_order": [0],
                    "lane": lane,
                    "iteration": iteration,
                    "operator": operator,
                }
            )
            return result

        chunks = _contiguous_chunks(sequences, self.config.worker_count)
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            raise CandidateParallelExecutionError("deadline reached before parallel submission")
        pool = self._ensure_pool()
        future_to_submission: dict[Any, tuple[int, int, int]] = {}
        submission_order: list[int] = []
        chunk_offset = 0
        for chunk in chunks:
            submission_id = self._submission_serial
            self._submission_serial += 1
            submission_order.append(submission_id)
            future = pool.submit(
                _solve_cpu_batch_worker,
                instance,
                chunk,
                batch_size,
                remaining,
            )
            future_to_submission[future] = (
                submission_id,
                chunk_offset,
                len(chunk),
            )
            chunk_offset += len(chunk)
        completed: dict[int, BatchChargingResult] = {}
        completed_spans: dict[int, tuple[int, int]] = {}
        interrupted: list[tuple[int, int, ExactBatchDeadlineExceeded]] = []
        completion_order: list[int] = []
        try:
            # Workers retain the exact solve deadline. The parent receives a
            # bounded reconciliation window so a cooperative deadline result
            # is not misclassified as an infrastructure timeout.
            for future in as_completed(
                future_to_submission,
                timeout=remaining + 0.5,
            ):
                submission_id, offset, length = future_to_submission[future]
                completion_order.append(submission_id)
                try:
                    completed[submission_id] = future.result()
                    completed_spans[submission_id] = (offset, length)
                except ExactBatchDeadlineExceeded as caught_deadline:
                    interrupted.append((submission_id, offset, caught_deadline))
        except BaseException as error:
            for future in future_to_submission:
                future.cancel()
            self.close(cancel_futures=True, wait=False)
            raise CandidateParallelExecutionError(
                f"parallel cpu_batch failed without fallback: {type(error).__name__}: {error}"
            ) from error
        observed_submissions = set(completed) | {
            submission_id for submission_id, _offset, _error in interrupted
        }
        if observed_submissions != set(submission_order):
            self.close(cancel_futures=True, wait=False)
            raise CandidateParallelExecutionError("parallel cpu_batch lost a submitted task")
        metrics = BackendMetrics("cpu_batch", batch_size)
        if interrupted:
            completed_indices: set[int] = set()
            for submission_id, result in completed.items():
                metrics.add(result.metrics)
                offset, length = completed_spans[submission_id]
                completed_indices.update(range(offset, offset + length))
            for _submission_id, offset, deadline_item in interrupted:
                metrics.add(deadline_item.metrics)
                completed_indices.update(
                    offset + index for index in deadline_item.completed_indices
                )
            self.events.append(
                {
                    "event_type": "parallel_batch",
                    "status": "deadline_rollback",
                    "worker_count": self.config.worker_count,
                    "submission_order": submission_order,
                    "completion_order": completion_order,
                    "merge_order": [],
                    "completed_indices": sorted(completed_indices),
                    "lane": lane,
                    "iteration": iteration,
                    "operator": operator,
                }
            )
            raise ExactBatchDeadlineExceeded(
                started_exact_calls=len(sequences),
                completed_exact_calls=len(completed_indices),
                metrics=metrics,
                completed_indices=tuple(sorted(completed_indices)),
            )
        merged_results: list[ChargingSubproblemResult] = []
        for submission_id in submission_order:
            item = completed[submission_id]
            metrics.add(item.metrics)
            merged_results.extend(item.results)
        if len(merged_results) != len(sequences):
            raise CandidateParallelExecutionError("parallel cpu_batch result count mismatch")
        output = BatchChargingResult(tuple(merged_results), metrics)
        self._record_results(sequences, output.results)
        self.events.append(
            {
                "event_type": "parallel_batch",
                "status": "parallel_complete",
                "worker_count": self.config.worker_count,
                "submission_order": submission_order,
                "completion_order": completion_order,
                "merge_order": submission_order,
                "lane": lane,
                "iteration": iteration,
                "operator": operator,
            }
        )
        return output

    def _record_results(
        self,
        sequences: Sequence[tuple[str, ...]],
        results: Sequence[ChargingSubproblemResult],
    ) -> None:
        self._route_results.extend(
            {
                "sequence": list(sequence),
                "result": {
                    "feasible": result.feasible,
                    "route": list(result.route),
                    "distance": result.distance,
                    "total_energy": result.total_energy,
                    "charged_energy": result.charged_energy,
                    "charging_time": result.charging_time,
                    "labels_generated": result.labels_generated,
                    "labels_expanded": result.labels_expanded,
                    "labels_pruned": result.labels_pruned,
                    "failure_reason": result.failure_reason,
                },
            }
            for sequence, result in zip(sequences, results, strict=True)
        )

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.config.worker_count,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return self._pool

    def close(
        self,
        *,
        cancel_futures: bool = False,
        wait: bool = True,
    ) -> None:
        self.finish_round()
        if self._pool is None:
            return
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)
        self._pool = None

    @property
    def candidate_work_hash(self) -> str:
        return _stable_hash(self._candidate_work)

    @property
    def route_result_hash(self) -> str:
        return _stable_hash(self._route_results)

    def statistics(self) -> dict[str, object]:
        decisions = [
            event for event in self.events
            if event.get("event_type")
            in {
                "candidate_control_decision",
                "candidate_control_decision_aggregate",
                "candidate_plan_decision",
            }
        ]
        budgets = [
            event for event in self.events
            if event.get("event_type") == "candidate_control_budget"
        ]
        rounds = [
            event
            for event in self.events
            if event.get("event_type") == "candidate_control_round"
            and event.get("status") == "completed"
        ]
        return {
            "enabled": self.config.enabled,
            "schema_version": self.config.schema_version,
            "proposal_top_k": self.config.proposal_top_k,
            "max_exact_calls_per_round": self.config.max_exact_calls_per_round,
            "worker_count": self.config.worker_count,
            "executor_model": self.config.executor_model,
            "ranking_policy": self.config.ranking_policy,
            "merge_policy": self.config.merge_policy,
            "candidate_decisions": len(decisions),
            "selected_candidates": sum(
                _event_count(event)
                for event in decisions
                if event.get("status") == "selected"
            ),
            "skipped_candidates": sum(
                _event_count(event)
                for event in decisions
                if event.get("status") != "selected"
            ),
            "budget_events": len(budgets),
            "budget_skips": sum(event.get("status") == "budget_skipped" for event in budgets),
            "parallel_batches": sum(
                event.get("status") == "parallel_complete" for event in self.events
            ),
            "completed_rounds": len(rounds),
            "maximum_exact_calls_per_round": max(
                (_integer_event_field(event, "used") for event in rounds),
                default=0,
            ),
            "total_round_remainder": sum(
                _integer_event_field(event, "remainder") for event in rounds
            ),
        }


def _integer_event_field(event: dict[str, object], field_name: str) -> int:
    value = event.get(field_name, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"candidate-control event field {field_name} is not numeric")


def _event_count(event: dict[str, object]) -> int:
    return _integer_event_field(event, "aggregate_count") if "aggregate_count" in event else 1


def _solve_cpu_batch_worker(
    instance: Instance,
    sequences: tuple[tuple[str, ...], ...],
    batch_size: int,
    timeout_seconds: float,
) -> BatchChargingResult:
    deadline = time.perf_counter() + max(0.0, timeout_seconds)
    return solve_exact_charging_batch(
        instance,
        sequences,
        backend=ExactChargingBackend.CPU_BATCH,
        batch_size=batch_size,
        deadline=deadline,
    )


def _contiguous_chunks(
    sequences: tuple[tuple[str, ...], ...],
    workers: int,
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    chunk_count = min(workers, len(sequences))
    chunk_size = math.ceil(len(sequences) / chunk_count)
    return tuple(
        sequences[offset : offset + chunk_size]
        for offset in range(0, len(sequences), chunk_size)
    )


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "+Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
