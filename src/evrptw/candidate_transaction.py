"""Stage 5.2 native candidate transactions.

The module owns the ordered screening/cache/exact transaction.  Callers supply
the existing operator exact budget; no ranking or search parameter is added.
Any native, exact-worker, deadline, or integrity failure rolls back staged
cache writes and propagates to the solver without a fallback path.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import struct
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from evrptw.models import Instance
from evrptw.native_kernels import (
    NATIVE_KERNEL_ABI_VERSION,
    NativeKernelRuntime,
)

CANDIDATE_TRANSACTION_SCHEMA_VERSION = "stage05.2-native-candidate-transaction-v1"

CustomerSequence = tuple[str, ...]
CandidateImplementationMode = Literal[
    "pair_pruning",
    "batched_screening",
    "candidate_transaction",
]
_SCREEN_REASONS = {
    0: "",
    1: "route_structure_prefilter",
    2: "capacity_prefilter",
    3: "forward_time_window_prefilter",
    4: "backward_time_window_prefilter",
    5: "time_window_slack_prefilter",
    6: "single_segment_energy_prefilter",
    7: "structural_energy_prefilter",
    8: "time_window_prefilter",
    9: "energy_prefilter",
}
_SCREEN_REASON_CODES = {reason: code for code, reason in _SCREEN_REASONS.items()}
_NATIVE_SCREEN_STATUSES = {
    0: "screened",
    1: "duplicate",
    2: "negative_cache_hit",
}


class CandidateTransactionDeadlineExceeded(RuntimeError):
    """The atomic candidate transaction crossed its lane deadline."""

    def __init__(self, boundary: str) -> None:
        super().__init__(f"candidate transaction deadline reached at {boundary}")
        self.boundary = boundary


@dataclass(frozen=True, slots=True)
class NativeCandidateTransactionConfig:
    """Explicit opt-in for the Stage 5.2 native transaction path."""

    enabled: bool = True
    implementation_mode: CandidateImplementationMode = "candidate_transaction"
    schema_version: str = CANDIDATE_TRANSACTION_SCHEMA_VERSION
    candidate_order_policy: str = "existing_operator_order"
    exact_budget_policy: str = "existing_operator_exact_evaluation_budget"
    cache_write_policy: str = "staged_atomic_commit"
    failure_policy: str = "fail_fast_no_fallback"

    def __post_init__(self) -> None:
        if not self.enabled:
            raise ValueError("disabled candidate transaction is ambiguous; pass None instead")
        if self.schema_version != CANDIDATE_TRANSACTION_SCHEMA_VERSION:
            raise ValueError(
                f"candidate transaction schema must be "
                f"{CANDIDATE_TRANSACTION_SCHEMA_VERSION!r}"
            )
        if self.implementation_mode not in {
            "pair_pruning",
            "batched_screening",
            "candidate_transaction",
        }:
            raise ValueError("unknown Stage 5.2 candidate implementation mode")
        if self.candidate_order_policy != "existing_operator_order":
            raise ValueError("candidate transaction must preserve existing operator order")
        if self.exact_budget_policy != "existing_operator_exact_evaluation_budget":
            raise ValueError("candidate transaction must use the operator exact budget")
        if self.cache_write_policy != "staged_atomic_commit":
            raise ValueError("candidate transaction cache writes must be atomic")
        if self.failure_policy != "fail_fast_no_fallback":
            raise ValueError("candidate transaction failures must not fall back")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateScreeningDecision:
    """One compact native screening result in original candidate order."""

    candidate_id: int
    sequence: CustomerSequence
    accepted: bool
    reason: str
    native_status: str
    duplicate_of: int | None = None
    distance_lower_bound: float = 0.0


@dataclass(frozen=True, slots=True)
class CandidateScreeningBatch:
    """Validated output of one native batched screening invocation."""

    decisions: tuple[CandidateScreeningDecision, ...]
    counters: Mapping[str, int]
    candidate_pool_hash: str

    def __post_init__(self) -> None:
        if len(self.candidate_pool_hash) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.candidate_pool_hash
        ):
            raise ValueError("native candidate-pool SHA-256 is invalid")
        if tuple(decision.candidate_id for decision in self.decisions) != tuple(
            range(len(self.decisions))
        ):
            raise ValueError("native screening decisions lost candidate order")
        expected = len(self.decisions)
        if self.counters.get("input_candidates") != expected:
            raise ValueError("native screening input count is inconsistent")


@dataclass(frozen=True, slots=True)
class CandidateTransactionRequest:
    """The complete deterministic identity and boundaries of one transaction."""

    candidates: tuple[CustomerSequence, ...]
    lane: str
    operator: str
    iteration: int | None
    exact_budget: int
    deadline: float

    def __post_init__(self) -> None:
        if not self.lane or not self.operator:
            raise ValueError("candidate transaction requires lane and operator")
        if self.exact_budget < 0:
            raise ValueError("candidate transaction exact budget cannot be negative")
        if not math.isfinite(self.deadline):
            raise ValueError("candidate transaction deadline must be finite")


@dataclass(frozen=True, slots=True)
class CandidateTransactionAudit:
    """Compact evidence independently recomputable from ordered inputs/results."""

    lane: str
    operator: str
    iteration: int | None
    input_candidates: int
    screening_rejections: int
    cache_hits: int
    exact_misses: int
    budget_skips: int
    duplicate_candidates: int
    screening_pool_hash: str
    transaction_sha256: str


@dataclass(frozen=True, slots=True)
class CandidateTransactionResult[ResultT]:
    ordered_results: tuple[ResultT, ...]
    audit: CandidateTransactionAudit


@dataclass(slots=True)
class NativeCandidateTransactionRuntime:
    """Solve-local owner of compact transaction evidence."""

    config: NativeCandidateTransactionConfig
    events: list[dict[str, object]] = field(default_factory=list)
    transaction_count: int = 0
    input_candidate_count: int = 0
    screening_occupancies: list[int] = field(default_factory=list)
    fallback_count: int = 0

    def statistics(self) -> dict[str, object]:
        median_occupancy = (
            0.0
            if not self.screening_occupancies
            else statistics.median(self.screening_occupancies)
        )
        return {
            "native_candidate_transactions": self.transaction_count,
            "native_candidate_input_count": self.input_candidate_count,
            "native_screening_occupancies": tuple(self.screening_occupancies),
            "native_screening_median_occupancy": median_occupancy,
            "native_candidate_transaction_fallbacks": self.fallback_count,
        }

    def record(self, audit: CandidateTransactionAudit) -> None:
        self.transaction_count += 1
        self.input_candidate_count += audit.input_candidates
        self.screening_occupancies.append(audit.input_candidates)
        self.events.append(
            {
                "event_type": "native_candidate_transaction",
                "status": "committed",
                **asdict(audit),
            }
        )


def native_screen_candidate_batch(
    instance: Instance,
    candidates: tuple[CustomerSequence, ...],
    *,
    native_runtime: NativeKernelRuntime,
    negative_cache: Mapping[CustomerSequence, str],
    deadline: float,
    incremental: npt.NDArray[np.float64] | None = None,
) -> CandidateScreeningBatch:
    """Pack and execute one fail-fast ABI-v2 native screening transaction."""

    if native_runtime.config.abi_version != NATIVE_KERNEL_ABI_VERSION:
        raise ValueError("native candidate transactions require ABI v2")
    native_runtime.context.assert_matches(instance)
    if time.perf_counter() >= deadline:
        raise CandidateTransactionDeadlineExceeded("before_native_screening")
    context = native_runtime.context
    route_offsets, route_indices = _pack_route_rows(
        candidates,
        context.name_to_index,
    )
    candidate_ids = np.arange(len(candidates), dtype=np.int64)
    options = np.ascontiguousarray(
        [1.0, context.reachability_epsilon, 0.0, 0.0],
        dtype=np.float64,
    )
    incremental_rows = (
        np.zeros((len(candidates), 6), dtype=np.float64)
        if incremental is None
        else _strict_array(
            incremental,
            name="incremental propagation",
            dtype=np.dtype(np.float64),
            shape=(len(candidates), 6),
        )
    )
    negative_sequences = tuple(sorted(negative_cache))
    negative_offsets, negative_indices = _pack_route_rows(
        negative_sequences,
        context.name_to_index,
    )
    try:
        negative_reason_codes = np.ascontiguousarray(
            [_SCREEN_REASON_CODES[negative_cache[sequence]] for sequence in negative_sequences],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("negative cache contains an unknown screening reason") from error

    from evrptw import _core as native_core

    started = time.perf_counter()
    try:
        payload = native_core.screen_route_batch_transaction_v2(
            context.node_kind,
            context.demand,
            context.ready_time,
            context.due_date,
            context.service_time,
            context.distance,
            context.reachable,
            context.vehicle,
            route_offsets,
            route_indices,
            candidate_ids,
            options,
            incremental_rows,
            negative_offsets,
            negative_indices,
            negative_reason_codes,
        )
    finally:
        native_runtime.record_screening(
            time.perf_counter() - started,
            batch_candidates=len(candidates),
        )
    if not isinstance(payload, tuple) or len(payload) != 7:
        raise RuntimeError("native candidate screening returned an invalid tuple")
    returned_ids = _strict_array(
        payload[0],
        name="candidate ids",
        dtype=np.dtype(np.int64),
        shape=(len(candidates),),
    )
    statuses = _strict_array(
        payload[1],
        name="candidate statuses",
        dtype=np.dtype(np.int64),
        shape=(len(candidates),),
    )
    duplicate_of = _strict_array(
        payload[2],
        name="duplicate identities",
        dtype=np.dtype(np.int64),
        shape=(len(candidates),),
    )
    codes = _strict_array(
        payload[3],
        name="screening codes",
        dtype=np.dtype(np.int64),
        shape=(len(candidates), 16),
    )
    metrics = _strict_array(
        payload[4],
        name="screening metrics",
        dtype=np.dtype(np.float64),
        shape=(len(candidates), 15),
    )
    raw_counters = _strict_array(
        payload[5],
        name="screening counters",
        dtype=np.dtype(np.int64),
        shape=(5,),
    )
    digest = payload[6]
    if not isinstance(digest, str):
        raise RuntimeError("native candidate screening SHA-256 must be a string")
    if not np.array_equal(returned_ids, candidate_ids):
        raise RuntimeError("native candidate screening changed candidate ids")
    reason_codes = tuple(int(code) for code in codes[:, 1])
    if any(code not in _SCREEN_REASONS for code in reason_codes):
        raise RuntimeError("native candidate screening returned an unknown reason")
    status_codes = tuple(int(status) for status in statuses)
    if any(status not in _NATIVE_SCREEN_STATUSES for status in status_codes):
        raise RuntimeError("native candidate screening returned an unknown status")
    expected_digest = _native_screening_digest(
        candidate_ids,
        statuses,
        duplicate_of,
        codes,
        route_offsets,
        route_indices,
    )
    if digest != expected_digest:
        raise RuntimeError("native candidate screening SHA-256 mismatch")
    counters = {
        "input_candidates": int(raw_counters[0]),
        "unique_candidates": int(raw_counters[1]),
        "duplicate_candidates": int(raw_counters[2]),
        "negative_cache_hits": int(raw_counters[3]),
        "screened_candidates": int(raw_counters[4]),
    }
    if (
        counters["input_candidates"] != len(candidates)
        or counters["unique_candidates"] + counters["duplicate_candidates"]
        != len(candidates)
        or counters["negative_cache_hits"] + counters["screened_candidates"]
        != counters["unique_candidates"]
    ):
        raise RuntimeError("native candidate screening counters are inconsistent")
    decisions = tuple(
        CandidateScreeningDecision(
            candidate_id=index,
            sequence=sequence,
            accepted=bool(codes[index, 0]),
            reason=_SCREEN_REASONS[reason_codes[index]],
            native_status=_NATIVE_SCREEN_STATUSES[status_codes[index]],
            duplicate_of=(
                None if int(duplicate_of[index]) < 0 else int(duplicate_of[index])
            ),
            distance_lower_bound=float(metrics[index, 3]),
        )
        for index, sequence in enumerate(candidates)
    )
    return CandidateScreeningBatch(decisions, counters, digest)


def execute_candidate_transaction[ResultT](
    request: CandidateTransactionRequest,
    *,
    screen_batch: Callable[[tuple[CustomerSequence, ...]], CandidateScreeningBatch],
    cache_lookup: Callable[[CustomerSequence], ResultT | None],
    exact_batch: Callable[[tuple[CustomerSequence, ...]], Sequence[ResultT]],
    stage_cache_write: Callable[[CustomerSequence, ResultT], None],
    commit_cache_writes: Callable[[], None],
    rollback_cache_writes: Callable[[str], None],
    rejected_result: Callable[[CandidateScreeningDecision], ResultT],
    skipped_result: Callable[[str], ResultT],
    clock: Callable[[], float] = time.perf_counter,
) -> CandidateTransactionResult[ResultT]:
    """Execute the fixed Stage 5.2 transaction order with atomic cache writes."""

    staged = False
    try:
        _check_deadline(request, "before_native_screening", clock)
        screening = screen_batch(request.candidates)
        if len(screening.decisions) != len(request.candidates):
            raise RuntimeError("native screening lost candidate rows")
        for candidate_id, (sequence, decision) in enumerate(
            zip(request.candidates, screening.decisions, strict=True)
        ):
            if decision.candidate_id != candidate_id or decision.sequence != sequence:
                raise RuntimeError("native screening changed candidate identity or order")

        resolved: list[ResultT | None] = [None] * len(request.candidates)
        waiting_indices: dict[CustomerSequence, list[int]] = {}
        exact_sequences: list[CustomerSequence] = []
        cache_hits = 0
        screening_rejections = 0
        budget_skips = 0
        duplicate_candidates = 0
        known_resolution: dict[CustomerSequence, ResultT | None] = {}

        for index, decision in enumerate(screening.decisions):
            if decision.native_status == "duplicate":
                duplicate_candidates += 1
            if not decision.accepted:
                screening_rejections += 1
                resolved[index] = rejected_result(decision)
                continue
            sequence = decision.sequence
            if sequence in waiting_indices:
                waiting_indices[sequence].append(index)
                continue
            if sequence in known_resolution:
                cached_duplicate = known_resolution[sequence]
                if cached_duplicate is None:
                    raise RuntimeError("candidate transaction resolution state is invalid")
                resolved[index] = cached_duplicate
                continue
            cached = cache_lookup(sequence)
            if cached is not None:
                cache_hits += 1
                known_resolution[sequence] = cached
                resolved[index] = cached
                continue
            if len(exact_sequences) >= request.exact_budget:
                budget_skips += 1
                skipped = skipped_result("operator_exact_budget_exhausted")
                known_resolution[sequence] = skipped
                resolved[index] = skipped
                continue
            waiting_indices[sequence] = [index]
            exact_sequences.append(sequence)

        _check_deadline(request, "before_exact_batch", clock)
        exact_results = tuple(exact_batch(tuple(exact_sequences)))
        if len(exact_results) != len(exact_sequences):
            raise RuntimeError("ordered exact batch returned the wrong result count")
        for sequence, result in zip(exact_sequences, exact_results, strict=True):
            for index in waiting_indices[sequence]:
                resolved[index] = result
            known_resolution[sequence] = result
            stage_cache_write(sequence, result)
            staged = True

        _check_deadline(request, "before_atomic_commit", clock)
        if any(result is None for result in resolved):
            raise RuntimeError("candidate transaction lost an ordered result")
        transaction_sha256 = _transaction_digest(
            request,
            screening,
            cache_hits=cache_hits,
            exact_misses=len(exact_sequences),
            budget_skips=budget_skips,
        )
        commit_cache_writes()
        staged = False
        audit = CandidateTransactionAudit(
            lane=request.lane,
            operator=request.operator,
            iteration=request.iteration,
            input_candidates=len(request.candidates),
            screening_rejections=screening_rejections,
            cache_hits=cache_hits,
            exact_misses=len(exact_sequences),
            budget_skips=budget_skips,
            duplicate_candidates=duplicate_candidates,
            screening_pool_hash=screening.candidate_pool_hash,
            transaction_sha256=transaction_sha256,
        )
        return CandidateTransactionResult(
            tuple(result for result in resolved if result is not None),
            audit,
        )
    except BaseException as error:
        rollback_cache_writes(
            f"candidate_transaction_rollback:{type(error).__name__}:{error}"
        )
        staged = False
        raise
    finally:
        if staged:
            rollback_cache_writes("candidate_transaction_rollback:uncommitted_stage")


def _check_deadline(
    request: CandidateTransactionRequest,
    boundary: str,
    clock: Callable[[], float],
) -> None:
    if clock() >= request.deadline:
        raise CandidateTransactionDeadlineExceeded(boundary)


def _transaction_digest(
    request: CandidateTransactionRequest,
    screening: CandidateScreeningBatch,
    *,
    cache_hits: int,
    exact_misses: int,
    budget_skips: int,
) -> str:
    payload = {
        "budget_skips": budget_skips,
        "cache_hits": cache_hits,
        "candidates": request.candidates,
        "exact_budget": request.exact_budget,
        "exact_misses": exact_misses,
        "iteration": request.iteration,
        "lane": request.lane,
        "operator": request.operator,
        "schema_version": CANDIDATE_TRANSACTION_SCHEMA_VERSION,
        "screening_pool_hash": screening.candidate_pool_hash,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _pack_route_rows(
    routes: tuple[CustomerSequence, ...],
    name_to_index: Mapping[str, int],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    offsets = [0]
    indices: list[int] = []
    for route in routes:
        indices.extend(name_to_index.get(name, -1) for name in route)
        offsets.append(len(indices))
    return (
        np.ascontiguousarray(offsets, dtype=np.int64),
        np.ascontiguousarray(indices, dtype=np.int64),
    )


def _strict_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise RuntimeError(f"native {name} must be a NumPy array")
    if value.dtype != dtype or value.shape != shape or not value.flags.c_contiguous:
        raise RuntimeError(
            f"native {name} must be C-contiguous {dtype} with shape {shape}"
        )
    return value


def _native_screening_digest(
    candidate_ids: npt.NDArray[np.int64],
    statuses: npt.NDArray[np.int64],
    duplicate_of: npt.NDArray[np.int64],
    codes: npt.NDArray[np.int64],
    route_offsets: npt.NDArray[np.int64],
    route_indices: npt.NDArray[np.int64],
) -> str:
    payload = bytearray()
    for index in range(len(candidate_ids)):
        begin = int(route_offsets[index])
        end = int(route_offsets[index + 1])
        for value in (
            int(candidate_ids[index]),
            int(statuses[index]),
            int(duplicate_of[index]),
            int(codes[index, 1]),
            end - begin,
            *(int(value) for value in route_indices[begin:end]),
        ):
            payload.extend(struct.pack("<q", value))
    return hashlib.sha256(payload).hexdigest()
