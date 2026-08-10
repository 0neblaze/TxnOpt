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
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt

from evrptw.models import Instance
from evrptw.native_kernels import (
    NATIVE_KERNEL_ABI_VERSION,
    NativeKernelRuntime,
)
from evrptw.stage052_physical_telemetry import (
    validate_native_work_task_receipt_stream,
)

if TYPE_CHECKING:
    from evrptw.neighborhoods import ScreeningResult

CANDIDATE_TRANSACTION_SCHEMA_VERSION = "stage05.2-native-candidate-transaction-v1"
STAGE052_NEGATIVE_SCREENING_RESULT_CACHE_ENTRIES = 65_536
STAGE052_NEGATIVE_SEQUENCE_CACHE_ENTRIES = 65_536

CustomerSequence = tuple[str, ...]
CandidateImplementationMode = Literal[
    "pair_pruning",
    "batched_screening",
    "candidate_transaction",
]
SCREEN_REASON_BY_CODE = {
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
_SCREEN_REASON_CODES = {reason: code for code, reason in SCREEN_REASON_BY_CODE.items()}
_NATIVE_SCREEN_STATUSES = {
    0: "screened",
    1: "duplicate",
    2: "negative_cache_hit",
}
_SCREEN_CHECK_BY_CODE = {
    1: "route_structure",
    2: "capacity_lower_bound",
    3: "forward_time_window",
    4: "backward_time_window",
    5: "time_window_slack",
    6: "shortest_distance_lower_bound",
    7: "single_segment_battery_reachability",
    8: "structural_energy_lower_bound",
}
_SCREEN_CHECK_STATUS_BY_CODE = {0: "fail", 1: "pass", 2: "recorded"}


class BoundedScreeningResultCache[ResultT]:
    """Solve-local LRU for safe screening rejections.

    Eviction changes only whether a later route is safely screened again.  It
    cannot admit an infeasible route, consume exact work, or change candidate
    order.  Stage 5.2 records the bound and every eviction in its compact
    runtime statistics.
    """

    def __init__(self, *, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("screening result cache capacity must be a positive integer")
        self.capacity = capacity
        self._entries: OrderedDict[str, ResultT] = OrderedDict()
        self._peak_entries = 0
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0

    def get(self, key: str) -> ResultT | None:
        value = self._entries.get(key)
        if value is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return value

    def store(self, key: str, value: ResultT) -> tuple[str, ResultT] | None:
        self._stores += 1
        if key in self._entries:
            self._entries[key] = value
            self._entries.move_to_end(key)
            return None
        self._entries[key] = value
        if len(self._entries) <= self.capacity:
            self._peak_entries = max(self._peak_entries, len(self._entries))
            return None
        self._evictions += 1
        evicted = self._entries.popitem(last=False)
        self._peak_entries = max(self._peak_entries, len(self._entries))
        return evicted

    def statistics(self) -> dict[str, object]:
        return {
            "backend": "bounded_lru_safe_rejection",
            "capacity": self.capacity,
            "current_entries": len(self._entries),
            "peak_entries": self._peak_entries,
            "hits": self._hits,
            "misses": self._misses,
            "stores": self._stores,
            "evictions": self._evictions,
        }


@dataclass(slots=True)
class NegativeSequenceCacheBatch:
    """Reversible update journal for the bounded safe-rejection cache."""

    added_sequences: tuple[CustomerSequence, ...]
    previous_entries: OrderedDict[CustomerSequence, str] | None
    evicted_count: int
    rollover: bool
    active: bool = True


type NegativeSequenceCacheSnapshot = tuple[
    tuple[tuple[CustomerSequence, str], ...],
    int,
    int,
    int,
    int,
]


class BoundedNegativeSequenceCache(Mapping[CustomerSequence, str]):
    """Bounded generation cache for native safe-screening rejections.

    When a committed batch would exceed the fixed capacity, the cache starts a
    new generation containing that batch.  Dropped entries are safe to
    recompute and never reach exact work merely because they were evicted.
    """

    def __init__(self, *, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("negative sequence cache capacity must be a positive integer")
        self.capacity = capacity
        self._entries: OrderedDict[CustomerSequence, str] = OrderedDict()
        self._active_batch: NegativeSequenceCacheBatch | None = None
        self._peak_entries = 0
        self._stores = 0
        self._evictions = 0
        self._rollovers = 0

    def __getitem__(self, key: CustomerSequence) -> str:
        return self._entries[key]

    def __iter__(self) -> Iterator[CustomerSequence]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def begin_store_many_atomic(
        self,
        entries: Mapping[CustomerSequence, str],
    ) -> NegativeSequenceCacheBatch:
        if self._active_batch is not None:
            raise RuntimeError("negative sequence cache already has an active batch")
        for sequence, reason in entries.items():
            existing = self._entries.get(sequence)
            if existing is not None and existing != reason:
                raise RuntimeError("candidate negative cache reason changed during commit")
        additions = tuple(sequence for sequence in entries if sequence not in self._entries)
        rollover = len(self._entries) + len(additions) > self.capacity
        previous_entries: OrderedDict[CustomerSequence, str] | None = None
        evicted_count = 0
        if rollover:
            replacement = OrderedDict(entries.items())
            if len(replacement) > self.capacity:
                raise RuntimeError(
                    "one candidate transaction exceeds the negative sequence cache capacity"
                )
            previous_entries = self._entries
            evicted_count = sum(sequence not in replacement for sequence in previous_entries)
            self._entries = replacement
        else:
            for sequence in additions:
                self._entries[sequence] = entries[sequence]
        batch = NegativeSequenceCacheBatch(
            added_sequences=additions,
            previous_entries=previous_entries,
            evicted_count=evicted_count,
            rollover=rollover,
        )
        self._active_batch = batch
        return batch

    def commit_store_batch(self, batch: NegativeSequenceCacheBatch) -> None:
        self._require_active(batch)
        self._stores += len(batch.added_sequences)
        self._evictions += batch.evicted_count
        self._rollovers += int(batch.rollover)
        self._peak_entries = max(self._peak_entries, len(self._entries))
        batch.active = False
        batch.previous_entries = None
        self._active_batch = None

    def rollback_store_batch(self, batch: NegativeSequenceCacheBatch) -> None:
        self._require_active(batch)
        if batch.rollover:
            assert batch.previous_entries is not None
            self._entries = batch.previous_entries
        else:
            for sequence in batch.added_sequences:
                self._entries.pop(sequence, None)
        batch.active = False
        batch.previous_entries = None
        self._active_batch = None

    def statistics(self) -> dict[str, object]:
        return {
            "backend": "bounded_generation_safe_rejection",
            "capacity": self.capacity,
            "current_entries": len(self._entries),
            "peak_entries": self._peak_entries,
            "stores": self._stores,
            "evictions": self._evictions,
            "rollovers": self._rollovers,
        }

    def snapshot_state(
        self,
    ) -> NegativeSequenceCacheSnapshot:
        if self._active_batch is not None:
            raise RuntimeError("cannot snapshot an active negative cache batch")
        return (
            tuple(self._entries.items()),
            self._peak_entries,
            self._stores,
            self._evictions,
            self._rollovers,
        )

    def restore_state(
        self,
        snapshot: NegativeSequenceCacheSnapshot,
    ) -> None:
        if self._active_batch is not None:
            raise RuntimeError("cannot restore an active negative cache batch")
        entries, peak_entries, stores, evictions, rollovers = snapshot
        self._entries = OrderedDict(entries)
        self._peak_entries = peak_entries
        self._stores = stores
        self._evictions = evictions
        self._rollovers = rollovers

    def _require_active(self, batch: NegativeSequenceCacheBatch) -> None:
        if self._active_batch is not batch or not batch.active:
            raise RuntimeError("negative sequence cache batch is no longer active")


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
                f"candidate transaction schema must be {CANDIDATE_TRANSACTION_SCHEMA_VERSION!r}"
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
class CandidateScreeningBatch:
    """Validated structured output of one native batched screening invocation."""

    sequences: tuple[CustomerSequence, ...]
    candidate_ids: npt.NDArray[np.int64]
    statuses: npt.NDArray[np.int64]
    duplicate_of: npt.NDArray[np.int64]
    codes: npt.NDArray[np.int64]
    metrics: npt.NDArray[np.float64]
    route_offsets: npt.NDArray[np.int64]
    route_indices: npt.NDArray[np.int64]
    raw_counters: npt.NDArray[np.int64]
    counters: Mapping[str, int]
    candidate_pool_hash: str

    def __post_init__(self) -> None:
        if len(self.candidate_pool_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.candidate_pool_hash
        ):
            raise ValueError("native candidate-pool SHA-256 is invalid")
        expected = len(self.sequences)
        if not np.array_equal(self.candidate_ids, np.arange(expected, dtype=np.int64)):
            raise ValueError("native screening decisions lost candidate order")
        if self.counters.get("input_candidates") != expected:
            raise ValueError("native screening input count is inconsistent")
        if (
            self.statuses.shape != (expected,)
            or self.duplicate_of.shape != (expected,)
            or self.codes.shape != (expected, 16)
            or self.metrics.shape != (expected, 15)
            or self.route_offsets.shape != (expected + 1,)
            or self.raw_counters.shape != (5,)
        ):
            raise ValueError("native screening structured arrays lost candidate rows")

    def accepted(self, index: int) -> bool:
        return bool(self.codes[index, 0])

    def reason(self, index: int) -> str:
        return SCREEN_REASON_BY_CODE[int(self.codes[index, 1])]

    def native_status(self, index: int) -> str:
        return _NATIVE_SCREEN_STATUSES[int(self.statuses[index])]

    def screening_result(self, index: int) -> ScreeningResult:
        """Materialize one scalar-compatible decision from the typed batch."""

        if not 0 <= index < len(self.sequences):
            raise IndexError("native screening result index is out of range")
        from evrptw.measurement import ScreeningCheckTrace
        from evrptw.neighborhoods import ScreeningResult, screening_check_explanation

        check_count = int(self.codes[index, 7])
        if not 0 <= check_count <= 8:
            raise RuntimeError("native screening check count is invalid")
        checks: list[ScreeningCheckTrace] = []
        for check_ordinal in range(check_count):
            packed_check = int(self.codes[index, 8 + check_ordinal])
            check_code, status_code = divmod(packed_check, 10)
            try:
                check_name = _SCREEN_CHECK_BY_CODE[check_code]
                check_status = _SCREEN_CHECK_STATUS_BY_CODE[status_code]
            except KeyError as error:
                raise RuntimeError("native screening check identity is invalid") from error
            checks.append(
                ScreeningCheckTrace(
                    check=check_name,
                    status=check_status,
                    value=(
                        bool(self.metrics[index, 7 + check_ordinal])
                        if check_code in {1, 7}
                        else float(self.metrics[index, 7 + check_ordinal])
                    ),
                    reason=screening_check_explanation(
                        check_code,
                        check_status,
                        event_count=check_count,
                    ),
                )
            )
        failed_check_code = int(self.codes[index, 2])
        distance_increment = float(self.metrics[index, 4])
        return ScreeningResult(
            accepted=self.accepted(index),
            reason=self.reason(index),
            demand=float(self.metrics[index, 0]),
            optimistic_finish_time=float(self.metrics[index, 1]),
            energy_reachable=bool(self.codes[index, 4]),
            checks=tuple(checks),
            first_failed_check=_SCREEN_CHECK_BY_CODE.get(failed_check_code, ""),
            min_time_window_slack=float(self.metrics[index, 2]),
            distance_lower_bound=float(self.metrics[index, 3]),
            distance_increment_lower_bound=(
                None if math.isnan(distance_increment) else distance_increment
            ),
            single_segment_reachable=bool(self.codes[index, 3]),
            structural_energy_lower_bound=float(self.metrics[index, 5]),
        )

    def integrity_evidence(self) -> dict[str, object]:
        """Return compact byte evidence for independent ABI digest replay."""

        return {
            "candidate_ids_le_hex": self.candidate_ids.astype("<i8", copy=False).tobytes().hex(),
            "statuses_le_hex": self.statuses.astype("<i8", copy=False).tobytes().hex(),
            "duplicate_of_le_hex": self.duplicate_of.astype("<i8", copy=False).tobytes().hex(),
            "codes_le_hex": self.codes.astype("<i8", copy=False).tobytes().hex(),
            "metrics_le_hex": self.metrics.astype("<f8", copy=False).tobytes().hex(),
            "route_offsets_le_hex": self.route_offsets.astype("<i8", copy=False).tobytes().hex(),
            "route_indices_le_hex": self.route_indices.astype("<i8", copy=False).tobytes().hex(),
            "counters_le_hex": self.raw_counters.astype("<i8", copy=False).tobytes().hex(),
        }


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
    candidates: tuple[CustomerSequence, ...]
    exact_budget: int
    screening_integrity_evidence: Mapping[str, object]
    screening_passes: int
    screening_rejections: int
    screening_cache_hits: int
    screening_exact_call_blocked: int
    screening_reason_counts: Mapping[str, int]
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
class NegativeCacheCommit:
    """O(changes) journal for solve-local packed negative-cache updates."""

    added_sequences: tuple[CustomerSequence, ...]
    entry_count_before: int
    index_count_before: int
    stores_before: int = 0
    evictions_before: int = 0
    rollovers_before: int = 0
    peak_entries_before: int = 0
    previous_sequences: set[CustomerSequence] | None = None
    previous_offsets: npt.NDArray[np.int64] | None = None
    previous_indices: npt.NDArray[np.int64] | None = None
    previous_reason_codes: npt.NDArray[np.int64] | None = None
    active: bool = True


@dataclass(frozen=True, slots=True)
class NativeCandidateTransactionProtocolSnapshot:
    event_count: int
    transaction_count: int
    input_candidate_count: int
    screening_occupancy_count: int
    fallback_count: int
    protocol_invocations: int
    protocol_total_seconds: float
    protocol_queue_wait_seconds: float


@dataclass(frozen=True, slots=True)
class NativeNegativeCacheSnapshot:
    initialized: bool
    sequences: frozenset[CustomerSequence]
    offsets: npt.NDArray[np.int64]
    indices: npt.NDArray[np.int64]
    reason_codes: npt.NDArray[np.int64]
    entry_count: int
    index_count: int
    peak_entries: int
    stores: int
    evictions: int
    rollovers: int


@dataclass(slots=True)
class NativeRouteMergeProfilePlan:
    """One reversible native cache lookup and invalidation plan."""

    hit_flags: npt.NDArray[np.int64]
    feasible_flags: npt.NDArray[np.int64]
    objective_metrics: npt.NDArray[np.float64]
    charging_counts: npt.NDArray[np.int64]
    miss_indices: npt.NDArray[np.int64]
    hits: int
    misses: int
    invalidations: int
    active: bool = True


def validate_native_work_pool_task_receipts(
    payload: Mapping[str, object],
    *,
    worker_threads: int,
    expected_receipt_path: Path | None = None,
) -> object:
    """Validate the bounded physical task trace shared by native pools."""

    schema = payload.get("schema_version")
    capacity = payload.get("task_receipt_capacity")
    dropped = payload.get("task_receipt_dropped_count")
    completed = payload.get("completed_tasks")
    receipts = payload.get("task_receipts")
    if schema in {
        "stage05.2-candidate-round-work-pool-v3",
        "stage05.2-full-native-work-pool-v3",
    }:
        if (
            isinstance(dropped, bool)
            or not isinstance(dropped, int)
            or dropped != 0
            or isinstance(completed, bool)
            or not isinstance(completed, int)
            or completed < 0
            or expected_receipt_path is None
        ):
            raise RuntimeError("native streamed work-pool task receipt is invalid")
        return validate_native_work_task_receipt_stream(
            receipts,
            expected_path=expected_receipt_path,
            completed_tasks=completed,
            worker_threads=worker_threads,
        )
    if (
        capacity != 65_536
        or isinstance(dropped, bool)
        or not isinstance(dropped, int)
        or dropped != 0
        or isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 0
        or not isinstance(receipts, list)
        or len(receipts) > capacity
        or completed != len(receipts)
    ):
        raise RuntimeError("native work-pool task receipt envelope is invalid")
    sequences: set[int] = set()
    for row in receipts:
        if not isinstance(row, tuple) or len(row) != 7:
            raise RuntimeError("native work-pool task receipt row is invalid")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in row):
            raise RuntimeError("native work-pool task receipt value is invalid")
        sequence, worker, first, last, submitted, started, finished = row
        if (
            sequence < 0
            or sequence in sequences
            or not 0 <= worker < worker_threads
            or not 0 <= first < last
            or not 0 <= submitted <= started <= finished
        ):
            raise RuntimeError("native work-pool task receipt failed reconciliation")
        sequences.add(sequence)
    return receipts


@dataclass(slots=True)
class NativeCandidateTransactionRuntime:
    """Solve-local owner of compact transaction evidence."""

    config: NativeCandidateTransactionConfig
    persistent_worker_threads: int = 0
    task_receipt_path: Path | None = None
    events: list[dict[str, object]] = field(default_factory=list)
    transaction_count: int = 0
    input_candidate_count: int = 0
    screening_occupancies: list[int] = field(default_factory=list)
    fallback_count: int = 0
    protocol_invocations: int = 0
    protocol_total_seconds: float = 0.0
    protocol_queue_wait_seconds: float = 0.0
    route_merge_pool_invocations: int = 0
    route_merge_pool_completions: int = 0
    route_merge_pool_failures: int = 0
    route_merge_pool_seconds: float = 0.0
    route_merge_pool_input_routes: int = 0
    route_merge_pool_output_candidates: int = 0
    route_merge_pool_pruned_pairs: int = 0
    route_merge_pool_pruned_candidates: int = 0
    route_merge_screening_invocations: int = 0
    route_merge_screening_failures: int = 0
    route_merge_screening_seconds: float = 0.0
    route_merge_screened_candidates: int = 0
    route_merge_screening_rejections: int = 0
    route_merge_profile_cache_hits: int = 0
    route_merge_profile_cache_misses: int = 0
    route_merge_profile_cache_invalidations: int = 0
    _negative_cache_initialized: bool = False
    _negative_cache_sequences: set[CustomerSequence] = field(default_factory=set)
    _negative_cache_offsets: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.empty(8, dtype=np.int64)
    )
    _negative_cache_indices: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.empty(16, dtype=np.int64)
    )
    _negative_cache_reason_codes: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.empty(8, dtype=np.int64)
    )
    _negative_cache_entry_count: int = 0
    _negative_cache_index_count: int = 0
    _negative_cache_peak_entries: int = 0
    _negative_cache_stores: int = 0
    _negative_cache_evictions: int = 0
    _negative_cache_rollovers: int = 0
    _native_work_pool: Any = field(init=False, repr=False, default=None)
    _native_route_merge_profile_cache: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.persistent_worker_threads, bool)
            or not isinstance(self.persistent_worker_threads, int)
            or not 0 <= self.persistent_worker_threads <= 256
        ):
            raise ValueError("persistent worker threads must be in [0, 256]")
        if self.task_receipt_path is not None:
            self.task_receipt_path = self.task_receipt_path.resolve()
            if self.persistent_worker_threads == 0:
                raise ValueError("native task-receipt streaming requires a persistent work pool")
        from evrptw import _core as native_core

        self._native_route_merge_profile_cache = native_core.NativeRouteMergeProfileCacheV1()
        if self.persistent_worker_threads > 0:
            self._native_work_pool = native_core.NativeCandidateRoundRuntimeV2(
                self.persistent_worker_threads,
                str(self.task_receipt_path) if self.task_receipt_path is not None else "",
            )

    def plan_route_merge_profiles(
        self,
        sequences: tuple[CustomerSequence, ...],
        available_flags: tuple[bool, ...],
        name_to_index: Mapping[str, int],
    ) -> NativeRouteMergeProfilePlan:
        """Plan one native profile-cache generation without Python key tables."""

        if not sequences or len(available_flags) != len(sequences):
            raise ValueError("route-merge profile plan inputs are invalid")
        route_offsets, route_indices = _pack_route_rows(sequences, name_to_index)
        if np.any(route_indices < 0):
            raise ValueError("route-merge profile contains an unknown node")
        available = np.ascontiguousarray(available_flags, dtype=np.int64)
        raw = self._native_route_merge_profile_cache.plan(
            route_offsets,
            route_indices,
            available,
        )
        if not isinstance(raw, tuple) or len(raw) != 6:
            self._native_route_merge_profile_cache.rollback()
            raise RuntimeError("native route-merge profile plan is invalid")
        route_count = len(sequences)
        hit_flags = _strict_array(
            raw[0], name="route-merge profile hits", dtype=np.dtype(np.int64), shape=(route_count,)
        )
        feasible_flags = _strict_array(
            raw[1],
            name="route-merge profile feasibility",
            dtype=np.dtype(np.int64),
            shape=(route_count,),
        )
        objective_metrics = _strict_array(
            raw[2],
            name="route-merge profile objectives",
            dtype=np.dtype(np.float64),
            shape=(route_count, 2),
        )
        charging_counts = _strict_array(
            raw[3],
            name="route-merge profile charging counts",
            dtype=np.dtype(np.int64),
            shape=(route_count,),
        )
        miss_indices_value = raw[4]
        counters = _strict_array(
            raw[5],
            name="route-merge profile counters",
            dtype=np.dtype(np.int64),
            shape=(3,),
        )
        if not isinstance(miss_indices_value, np.ndarray) or miss_indices_value.dtype != np.int64:
            self._native_route_merge_profile_cache.rollback()
            raise RuntimeError("native route-merge profile miss indices are invalid")
        miss_indices = _strict_array(
            miss_indices_value,
            name="route-merge profile miss indices",
            dtype=np.dtype(np.int64),
            shape=(int(counters[1]),),
        )
        if (
            np.any((hit_flags != 0) & (hit_flags != 1))
            or np.any((feasible_flags != 0) & (feasible_flags != 1))
            or np.any(~np.isfinite(objective_metrics))
            or np.any(objective_metrics < 0.0)
            or np.any(charging_counts < 0)
            or np.any(counters < 0)
            or int(counters[0]) + int(counters[1]) != route_count
            or int(np.sum(hit_flags)) != int(counters[0])
            or (
                miss_indices.size > 0
                and (
                    int(miss_indices[0]) < 0
                    or int(miss_indices[-1]) >= route_count
                    or np.any(np.diff(miss_indices) <= 0)
                    or np.any(hit_flags[miss_indices] != 0)
                )
            )
        ):
            self._native_route_merge_profile_cache.rollback()
            raise RuntimeError("native route-merge profile plan failed reconciliation")
        return NativeRouteMergeProfilePlan(
            hit_flags=hit_flags,
            feasible_flags=feasible_flags,
            objective_metrics=objective_metrics,
            charging_counts=charging_counts,
            miss_indices=miss_indices,
            hits=int(counters[0]),
            misses=int(counters[1]),
            invalidations=int(counters[2]),
        )

    def commit_route_merge_profiles(
        self,
        plan: NativeRouteMergeProfilePlan,
        profiles: tuple[tuple[bool, float, float, int], ...],
    ) -> None:
        """Commit profiles for exactly the native plan's ordered misses."""

        if not plan.active or len(profiles) != plan.misses:
            raise RuntimeError("route-merge profile commit does not match its active plan")
        feasible = np.ascontiguousarray([int(profile[0]) for profile in profiles], dtype=np.int64)
        metrics = np.ascontiguousarray(
            [(profile[1], profile[2]) for profile in profiles], dtype=np.float64
        ).reshape(plan.misses, 2)
        counts = np.ascontiguousarray([profile[3] for profile in profiles], dtype=np.int64)
        self._native_route_merge_profile_cache.commit(
            plan.miss_indices,
            feasible,
            metrics,
            counts,
        )
        for row, route_index in enumerate(plan.miss_indices):
            index = int(route_index)
            plan.feasible_flags[index] = feasible[row]
            plan.objective_metrics[index] = metrics[row]
            plan.charging_counts[index] = counts[row]
        plan.active = False

    def rollback_route_merge_profiles(self, plan: NativeRouteMergeProfilePlan) -> None:
        """Discard an active native profile-cache plan."""

        if plan.active:
            self._native_route_merge_profile_cache.rollback()
            plan.active = False

    def execute_native_round(self, *arguments: object) -> object:
        """Dispatch through the solve-local persistent pool when configured."""

        if self._native_work_pool is not None:
            entrypoint = cast(Callable[..., object], self._native_work_pool.execute)
        else:
            from evrptw import _core as native_core

            entrypoint = cast(Callable[..., object], native_core.candidate_round_transaction_v2)
        return entrypoint(*arguments)

    def execute_route_merge_candidate_pool(self, *arguments: object) -> object:
        """Build one complete route-merge pool in one native crossing."""

        from evrptw import _core as native_core

        self.route_merge_pool_invocations += 1
        started = time.perf_counter()
        try:
            entrypoint = cast(
                Callable[..., object],
                native_core.route_merge_candidate_pool_v2,
            )
            result = entrypoint(*arguments)
        except BaseException:
            self.route_merge_pool_failures += 1
            raise
        finally:
            self.route_merge_pool_seconds += time.perf_counter() - started
        if not isinstance(result, tuple) or len(result) != 4:
            self.route_merge_pool_failures += 1
            raise RuntimeError("native route-merge candidate pool receipt is invalid")
        offsets, _indices, _metadata, pruning = result
        if (
            not isinstance(offsets, np.ndarray)
            or offsets.ndim != 1
            or offsets.size == 0
            or not isinstance(pruning, np.ndarray)
            or pruning.shape != (2,)
        ):
            self.route_merge_pool_failures += 1
            raise RuntimeError("native route-merge candidate pool arrays are invalid")
        route_offsets = arguments[0] if arguments else None
        if not isinstance(route_offsets, np.ndarray) or route_offsets.ndim != 1:
            self.route_merge_pool_failures += 1
            raise RuntimeError("native route-merge route offsets are invalid")
        self.route_merge_pool_completions += 1
        self.route_merge_pool_input_routes += int(route_offsets.size - 1)
        self.route_merge_pool_output_candidates += int(offsets.size - 1)
        self.route_merge_pool_pruned_pairs += int(pruning[0])
        self.route_merge_pool_pruned_candidates += int(pruning[1])
        return result

    def execute_route_merge_screening(self, *arguments: object) -> object:
        """Screen one route-merge pool natively without changing cache state."""

        self.route_merge_screening_invocations += 1
        started = time.perf_counter()
        try:
            if self._native_work_pool is not None:
                entrypoint = cast(
                    Callable[..., object],
                    self._native_work_pool.execute_screening,
                )
            else:
                from evrptw import _core as native_core

                entrypoint = cast(
                    Callable[..., object],
                    native_core.screen_route_batch_transaction_v2,
                )
            return entrypoint(*arguments)
        except BaseException:
            self.route_merge_screening_failures += 1
            raise
        finally:
            self.route_merge_screening_seconds += time.perf_counter() - started

    def execute_candidate_screening(self, *arguments: object) -> object:
        """Use the solve-local persistent pool for a generic screening batch."""

        if self._native_work_pool is not None:
            entrypoint = cast(
                Callable[..., object],
                self._native_work_pool.execute_screening,
            )
        else:
            from evrptw import _core as native_core

            entrypoint = cast(
                Callable[..., object],
                native_core.screen_route_batch_transaction_v2,
            )
        return entrypoint(*arguments)

    def execute_route_merge_candidate_pool_screened(self, *arguments: object) -> tuple[object, ...]:
        """Build and screen the full route-merge pool in one native crossing."""

        self.route_merge_pool_invocations += 1
        self.route_merge_screening_invocations += 1
        try:
            if self._native_work_pool is not None:
                entrypoint = cast(
                    Callable[..., object],
                    self._native_work_pool.execute_route_merge_pool_screened,
                )
            else:
                from evrptw import _core as native_core

                entrypoint = cast(
                    Callable[..., object],
                    native_core.route_merge_candidate_pool_screened_v3,
                )
            raw = entrypoint(*arguments)
            if not isinstance(raw, tuple) or len(raw) != 7:
                raise RuntimeError("native route-merge pool-screening receipt is invalid")
            (
                offsets,
                _indices,
                _metadata,
                pruning,
                pruned_pair_metadata,
                screening,
                timings,
            ) = raw
            if (
                not isinstance(offsets, np.ndarray)
                or offsets.dtype != np.int64
                or offsets.ndim != 1
                or offsets.size == 0
                or not isinstance(pruning, np.ndarray)
                or pruning.dtype != np.int64
                or pruning.shape != (2,)
                or not isinstance(pruned_pair_metadata, np.ndarray)
                or pruned_pair_metadata.dtype != np.int64
                or pruned_pair_metadata.ndim != 2
                or pruned_pair_metadata.shape != (int(pruning[0]), 4)
                or not isinstance(screening, tuple)
                or len(screening) != 7
                or not isinstance(timings, np.ndarray)
                or timings.dtype != np.float64
                or timings.shape != (2,)
                or not np.all(np.isfinite(timings))
                or np.any(timings < 0.0)
            ):
                raise RuntimeError("native route-merge pool-screening arrays are invalid")
            route_offsets = arguments[8] if len(arguments) > 8 else None
            if (
                not isinstance(route_offsets, np.ndarray)
                or route_offsets.dtype != np.int64
                or route_offsets.ndim != 1
                or route_offsets.size < 3
            ):
                raise RuntimeError("native route-merge route offsets are invalid")
            self.route_merge_pool_completions += 1
            self.route_merge_pool_seconds += float(timings[0])
            self.route_merge_screening_seconds += float(timings[1])
            self.route_merge_pool_input_routes += int(route_offsets.size - 1)
            self.route_merge_pool_output_candidates += int(offsets.size - 1)
            self.route_merge_pool_pruned_pairs += int(pruning[0])
            self.route_merge_pool_pruned_candidates += int(pruning[1])
            return raw
        except BaseException:
            self.route_merge_pool_failures += 1
            self.route_merge_screening_failures += 1
            raise

    def record_route_merge_screening(
        self,
        *,
        candidates: int,
        rejections: int,
    ) -> None:
        if (
            isinstance(candidates, bool)
            or not isinstance(candidates, int)
            or candidates < 0
            or isinstance(rejections, bool)
            or not isinstance(rejections, int)
            or not 0 <= rejections <= candidates
        ):
            raise ValueError("route-merge screening counters are invalid")
        self.route_merge_screened_candidates += candidates
        self.route_merge_screening_rejections += rejections

    def native_work_pool_statistics(self) -> dict[str, object]:
        if self._native_work_pool is None:
            return {"enabled": False}
        raw = self._native_work_pool.statistics()
        if not isinstance(raw, dict):
            raise RuntimeError("native candidate work-pool statistics are invalid")
        common_fields = {
            "schema_version",
            "thread_count",
            "maximum_pending_tasks",
            "pending_tasks",
            "active_tasks",
            "peak_pending_tasks",
            "peak_active_tasks",
            "queue_full_count",
            "rejected_count",
            "completed_tasks",
            "total_wait_seconds",
            "maximum_wait_seconds",
            "total_service_seconds",
            "maximum_service_seconds",
            "wait_histogram",
            "service_histogram",
            "task_receipt_dropped_count",
            "task_receipts",
        }
        schema = raw.get("schema_version")
        expected_fields = (
            common_fields | {"task_receipt_capacity"}
            if schema == "stage05.2-candidate-round-work-pool-v2"
            else common_fields
        )
        if set(raw) != expected_fields or schema not in {
            "stage05.2-candidate-round-work-pool-v2",
            "stage05.2-candidate-round-work-pool-v3",
        }:
            raise RuntimeError("native candidate work-pool schema is invalid")
        if (schema == "stage05.2-candidate-round-work-pool-v3") != (
            self.task_receipt_path is not None
        ):
            raise RuntimeError("native candidate work-pool receipt mode differs")
        if raw.get("thread_count") != self.persistent_worker_threads:
            raise RuntimeError("native candidate work-pool thread count diverged")
        for field_name in (
            "maximum_pending_tasks",
            "pending_tasks",
            "active_tasks",
            "peak_pending_tasks",
            "peak_active_tasks",
            "queue_full_count",
            "rejected_count",
            "completed_tasks",
        ):
            value = raw.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError("native candidate work-pool counter is invalid")
        for field_name in (
            "total_wait_seconds",
            "maximum_wait_seconds",
            "total_service_seconds",
            "maximum_service_seconds",
        ):
            value = raw.get(field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise RuntimeError("native candidate work-pool timing is invalid")
        for field_name in ("wait_histogram", "service_histogram"):
            histogram = raw.get(field_name)
            if (
                not isinstance(histogram, list)
                or len(histogram) != 32
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in histogram
                )
            ):
                raise RuntimeError("native candidate work-pool histogram is invalid")
        if any(
            raw.get(field_name) != 0
            for field_name in (
                "pending_tasks",
                "active_tasks",
                "queue_full_count",
                "rejected_count",
            )
        ):
            raise RuntimeError("native candidate work-pool resource gate failed")
        raw["task_receipts"] = validate_native_work_pool_task_receipts(
            raw,
            worker_threads=self.persistent_worker_threads,
            expected_receipt_path=self.task_receipt_path,
        )
        return {"enabled": True, **raw}

    def snapshot_protocol_state(
        self,
    ) -> NativeCandidateTransactionProtocolSnapshot:
        return NativeCandidateTransactionProtocolSnapshot(
            event_count=len(self.events),
            transaction_count=self.transaction_count,
            input_candidate_count=self.input_candidate_count,
            screening_occupancy_count=len(self.screening_occupancies),
            fallback_count=self.fallback_count,
            protocol_invocations=self.protocol_invocations,
            protocol_total_seconds=self.protocol_total_seconds,
            protocol_queue_wait_seconds=self.protocol_queue_wait_seconds,
        )

    def rollback_protocol_state(
        self,
        snapshot: NativeCandidateTransactionProtocolSnapshot,
    ) -> None:
        if (
            len(self.events) < snapshot.event_count
            or len(self.screening_occupancies) < snapshot.screening_occupancy_count
        ):
            raise RuntimeError("candidate transaction journal cannot roll forward")
        del self.events[snapshot.event_count :]
        del self.screening_occupancies[snapshot.screening_occupancy_count :]
        self.transaction_count = snapshot.transaction_count
        self.input_candidate_count = snapshot.input_candidate_count
        self.fallback_count = snapshot.fallback_count
        self.protocol_invocations = snapshot.protocol_invocations
        self.protocol_total_seconds = snapshot.protocol_total_seconds
        self.protocol_queue_wait_seconds = snapshot.protocol_queue_wait_seconds

    def snapshot_negative_cache_state(self) -> NativeNegativeCacheSnapshot:
        return NativeNegativeCacheSnapshot(
            initialized=self._negative_cache_initialized,
            sequences=frozenset(self._negative_cache_sequences),
            offsets=self._negative_cache_offsets[: self._negative_cache_entry_count + 1].copy(),
            indices=self._negative_cache_indices[: self._negative_cache_index_count].copy(),
            reason_codes=self._negative_cache_reason_codes[
                : self._negative_cache_entry_count
            ].copy(),
            entry_count=self._negative_cache_entry_count,
            index_count=self._negative_cache_index_count,
            peak_entries=self._negative_cache_peak_entries,
            stores=self._negative_cache_stores,
            evictions=self._negative_cache_evictions,
            rollovers=self._negative_cache_rollovers,
        )

    def restore_negative_cache_state(
        self,
        snapshot: NativeNegativeCacheSnapshot,
    ) -> None:
        self._negative_cache_initialized = snapshot.initialized
        self._negative_cache_sequences = set(snapshot.sequences)
        self._negative_cache_offsets = _grow_int64_buffer(
            snapshot.offsets.copy(),
            max(8, snapshot.entry_count + 1),
        )
        self._negative_cache_indices = _grow_int64_buffer(
            snapshot.indices.copy(),
            max(16, snapshot.index_count),
        )
        self._negative_cache_reason_codes = _grow_int64_buffer(
            snapshot.reason_codes.copy(),
            max(8, snapshot.entry_count),
        )
        self._negative_cache_entry_count = snapshot.entry_count
        self._negative_cache_index_count = snapshot.index_count
        self._negative_cache_peak_entries = snapshot.peak_entries
        self._negative_cache_stores = snapshot.stores
        self._negative_cache_evictions = snapshot.evictions
        self._negative_cache_rollovers = snapshot.rollovers

    def statistics(self) -> dict[str, object]:
        median_occupancy = (
            0.0 if not self.screening_occupancies else statistics.median(self.screening_occupancies)
        )
        return {
            "native_candidate_transactions": self.transaction_count,
            "native_candidate_input_count": self.input_candidate_count,
            "native_screening_occupancies": tuple(self.screening_occupancies),
            "native_screening_median_occupancy": median_occupancy,
            "native_candidate_transaction_fallbacks": self.fallback_count,
            "native_worker_protocol_invocations": self.protocol_invocations,
            "native_worker_protocol_total_seconds": self.protocol_total_seconds,
            "native_worker_protocol_queue_wait_seconds": (self.protocol_queue_wait_seconds),
            "negative_screening_sequence_cache": {
                "backend": "bounded_generation_safe_rejection",
                "capacity": STAGE052_NEGATIVE_SEQUENCE_CACHE_ENTRIES,
                "current_entries": self._negative_cache_entry_count,
                "peak_entries": self._negative_cache_peak_entries,
                "stores": self._negative_cache_stores,
                "evictions": self._negative_cache_evictions,
                "rollovers": self._negative_cache_rollovers,
            },
            "native_candidate_work_pool": self.native_work_pool_statistics(),
            "native_route_merge_candidate_pool": {
                "invocations": self.route_merge_pool_invocations,
                "completions": self.route_merge_pool_completions,
                "failures": self.route_merge_pool_failures,
                "seconds": self.route_merge_pool_seconds,
                "input_routes": self.route_merge_pool_input_routes,
                "output_candidates": self.route_merge_pool_output_candidates,
                "pruned_pairs": self.route_merge_pool_pruned_pairs,
                "pruned_candidates": self.route_merge_pool_pruned_candidates,
                "screening_invocations": self.route_merge_screening_invocations,
                "screening_failures": self.route_merge_screening_failures,
                "screening_seconds": self.route_merge_screening_seconds,
                "screened_candidates": self.route_merge_screened_candidates,
                "screening_rejections": self.route_merge_screening_rejections,
                "profile_cache_hits": self.route_merge_profile_cache_hits,
                "profile_cache_misses": self.route_merge_profile_cache_misses,
                "profile_cache_invalidations": (self.route_merge_profile_cache_invalidations),
            },
        }

    def record_route_merge_profile_cache(
        self,
        *,
        hits: int,
        misses: int,
        invalidations: int,
    ) -> None:
        if min(hits, misses, invalidations) < 0:
            raise ValueError("route-merge profile cache counters must be non-negative")
        self.route_merge_profile_cache_hits += hits
        self.route_merge_profile_cache_misses += misses
        self.route_merge_profile_cache_invalidations += invalidations

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

    def record_worker_protocol(
        self,
        audit: CandidateTransactionAudit,
        *,
        worker_protocol: str,
        total_seconds: float,
        queue_wait_seconds: float,
        completion_order: tuple[int, ...],
    ) -> None:
        if (
            not math.isfinite(total_seconds)
            or total_seconds < 0.0
            or not math.isfinite(queue_wait_seconds)
            or queue_wait_seconds < 0.0
        ):
            raise RuntimeError("native worker protocol timing is invalid")
        self.record(audit)
        self.protocol_invocations += 1
        self.protocol_total_seconds += total_seconds
        self.protocol_queue_wait_seconds += queue_wait_seconds
        self.events.append(
            {
                "event_type": "native_worker_protocol",
                "status": "committed",
                "worker_protocol": worker_protocol,
                "lane": audit.lane,
                "operator": audit.operator,
                "iteration": audit.iteration,
                "completion_order": completion_order,
                "total_seconds": total_seconds,
                "queue_wait_seconds": queue_wait_seconds,
                "transaction_sha256": audit.transaction_sha256,
            }
        )

    def packed_negative_cache(
        self,
        negative_cache: Mapping[CustomerSequence, str],
        name_to_index: Mapping[str, int],
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ]:
        """Return the amortized solve-local packed negative-cache state."""

        if not self._negative_cache_initialized:
            self._negative_cache_initialized = True
            self._negative_cache_offsets[0] = 0
            self.commit_negative_cache_entries(
                dict(sorted(negative_cache.items())),
                name_to_index,
            )
        elif len(negative_cache) != self._negative_cache_entry_count:
            raise RuntimeError("negative cache changed outside the candidate transaction runtime")
        return (
            self._negative_cache_offsets[: self._negative_cache_entry_count + 1],
            self._negative_cache_indices[: self._negative_cache_index_count],
            self._negative_cache_reason_codes[: self._negative_cache_entry_count],
        )

    def commit_negative_cache_entries(
        self,
        entries: Mapping[CustomerSequence, str],
        name_to_index: Mapping[str, int],
    ) -> NegativeCacheCommit:
        """Append committed entries without repacking prior cache contents."""

        additions = [
            (sequence, reason)
            for sequence, reason in entries.items()
            if sequence not in self._negative_cache_sequences
        ]
        if not additions:
            if self._negative_cache_entry_count == 0:
                self._negative_cache_offsets[0] = 0
            return NegativeCacheCommit(
                (),
                self._negative_cache_entry_count,
                self._negative_cache_index_count,
                stores_before=self._negative_cache_stores,
                evictions_before=self._negative_cache_evictions,
                rollovers_before=self._negative_cache_rollovers,
                peak_entries_before=self._negative_cache_peak_entries,
            )
        entry_count_before = self._negative_cache_entry_count
        index_count_before = self._negative_cache_index_count
        stores_before = self._negative_cache_stores
        evictions_before = self._negative_cache_evictions
        rollovers_before = self._negative_cache_rollovers
        peak_entries_before = self._negative_cache_peak_entries
        packed_routes = tuple(sequence for sequence, _reason in additions)
        packed_offsets, packed_indices = _pack_route_rows(
            packed_routes,
            name_to_index,
        )
        try:
            reason_codes = [_SCREEN_REASON_CODES[reason] for _sequence, reason in additions]
        except KeyError as error:
            raise ValueError("negative cache contains an unknown screening reason") from error
        required_entries = self._negative_cache_entry_count + len(additions)
        if required_entries > STAGE052_NEGATIVE_SEQUENCE_CACHE_ENTRIES:
            raise RuntimeError("negative cache append exceeds the bounded generation capacity")
        required_indices = self._negative_cache_index_count + len(packed_indices)
        self._negative_cache_offsets = _grow_int64_buffer(
            self._negative_cache_offsets,
            required_entries + 1,
        )
        self._negative_cache_reason_codes = _grow_int64_buffer(
            self._negative_cache_reason_codes,
            required_entries,
        )
        self._negative_cache_indices = _grow_int64_buffer(
            self._negative_cache_indices,
            required_indices,
        )
        base_entry = self._negative_cache_entry_count
        base_index = self._negative_cache_index_count
        self._negative_cache_indices[base_index : base_index + len(packed_indices)] = packed_indices
        self._negative_cache_offsets[base_entry + 1 : required_entries + 1] = (
            packed_offsets[1:] + base_index
        )
        self._negative_cache_reason_codes[base_entry:required_entries] = reason_codes
        self._negative_cache_sequences.update(packed_routes)
        self._negative_cache_entry_count = required_entries
        self._negative_cache_index_count = required_indices
        self._negative_cache_stores += len(additions)
        self._negative_cache_peak_entries = max(
            self._negative_cache_peak_entries,
            required_entries,
        )
        return NegativeCacheCommit(
            packed_routes,
            entry_count_before,
            index_count_before,
            stores_before=stores_before,
            evictions_before=evictions_before,
            rollovers_before=rollovers_before,
            peak_entries_before=peak_entries_before,
        )

    def replace_negative_cache_entries(
        self,
        entries: Mapping[CustomerSequence, str],
        name_to_index: Mapping[str, int],
    ) -> NegativeCacheCommit:
        """Atomically start a bounded safe-rejection cache generation."""

        if len(entries) > STAGE052_NEGATIVE_SEQUENCE_CACHE_ENTRIES:
            raise RuntimeError("negative cache replacement exceeds the bounded generation capacity")
        packed_routes = tuple(entries)
        packed_offsets, packed_indices = _pack_route_rows(
            packed_routes,
            name_to_index,
        )
        try:
            reason_codes = np.asarray(
                [_SCREEN_REASON_CODES[entries[sequence]] for sequence in packed_routes],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError("negative cache contains an unknown screening reason") from error
        previous_sequences = self._negative_cache_sequences
        previous_offsets = self._negative_cache_offsets
        previous_indices = self._negative_cache_indices
        previous_reason_codes = self._negative_cache_reason_codes
        entry_count_before = self._negative_cache_entry_count
        index_count_before = self._negative_cache_index_count
        stores_before = self._negative_cache_stores
        evictions_before = self._negative_cache_evictions
        rollovers_before = self._negative_cache_rollovers
        peak_entries_before = self._negative_cache_peak_entries
        next_sequences = set(packed_routes)
        self._negative_cache_sequences = next_sequences
        self._negative_cache_offsets = np.ascontiguousarray(
            packed_offsets,
            dtype=np.int64,
        )
        self._negative_cache_indices = np.ascontiguousarray(
            packed_indices,
            dtype=np.int64,
        )
        self._negative_cache_reason_codes = reason_codes
        self._negative_cache_entry_count = len(packed_routes)
        self._negative_cache_index_count = len(packed_indices)
        self._negative_cache_stores += sum(
            sequence not in previous_sequences for sequence in next_sequences
        )
        self._negative_cache_evictions += sum(
            sequence not in next_sequences for sequence in previous_sequences
        )
        self._negative_cache_rollovers += 1
        self._negative_cache_peak_entries = max(
            self._negative_cache_peak_entries,
            len(packed_routes),
        )
        return NegativeCacheCommit(
            (),
            entry_count_before,
            index_count_before,
            stores_before=stores_before,
            evictions_before=evictions_before,
            rollovers_before=rollovers_before,
            peak_entries_before=peak_entries_before,
            previous_sequences=previous_sequences,
            previous_offsets=previous_offsets,
            previous_indices=previous_indices,
            previous_reason_codes=previous_reason_codes,
        )

    def commit_negative_cache_batch(self, commit: NegativeCacheCommit) -> None:
        """Finalize a packed negative-cache update."""

        if not commit.active:
            raise RuntimeError("negative cache commit is no longer active")
        commit.previous_sequences = None
        commit.previous_offsets = None
        commit.previous_indices = None
        commit.previous_reason_codes = None
        commit.active = False

    def rollback_negative_cache_batch(self, commit: NegativeCacheCommit) -> None:
        """Undo a packed negative-cache append using its bounded journal."""

        if not commit.active:
            raise RuntimeError("negative cache commit is no longer active")
        if commit.previous_sequences is not None:
            assert commit.previous_offsets is not None
            assert commit.previous_indices is not None
            assert commit.previous_reason_codes is not None
            self._negative_cache_sequences = commit.previous_sequences
            self._negative_cache_offsets = commit.previous_offsets
            self._negative_cache_indices = commit.previous_indices
            self._negative_cache_reason_codes = commit.previous_reason_codes
        else:
            for sequence in commit.added_sequences:
                self._negative_cache_sequences.remove(sequence)
        self._negative_cache_entry_count = commit.entry_count_before
        self._negative_cache_index_count = commit.index_count_before
        self._negative_cache_stores = commit.stores_before
        self._negative_cache_evictions = commit.evictions_before
        self._negative_cache_rollovers = commit.rollovers_before
        self._negative_cache_peak_entries = commit.peak_entries_before
        commit.previous_sequences = None
        commit.previous_offsets = None
        commit.previous_indices = None
        commit.previous_reason_codes = None
        commit.active = False


def native_screen_candidate_batch(
    instance: Instance,
    candidates: tuple[CustomerSequence, ...],
    *,
    native_runtime: NativeKernelRuntime,
    transaction_runtime: NativeCandidateTransactionRuntime,
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
    negative_offsets, negative_indices, negative_reason_codes = (
        transaction_runtime.packed_negative_cache(
            negative_cache,
            context.name_to_index,
        )
    )

    started = time.perf_counter()
    try:
        payload = transaction_runtime.execute_candidate_screening(
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
    return decode_native_candidate_screening_payload(
        payload,
        candidates=candidates,
        candidate_ids=candidate_ids,
        route_offsets=route_offsets,
        route_indices=route_indices,
    )


def decode_native_candidate_screening_payload(
    payload: object,
    *,
    candidates: tuple[CustomerSequence, ...],
    candidate_ids: npt.NDArray[np.int64],
    route_offsets: npt.NDArray[np.int64],
    route_indices: npt.NDArray[np.int64],
) -> CandidateScreeningBatch:
    """Validate and replay screening arrays returned by a native transaction."""

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
    if any(code not in SCREEN_REASON_BY_CODE for code in reason_codes):
        raise RuntimeError("native candidate screening returned an unknown reason")
    status_codes = tuple(int(status) for status in statuses)
    if any(status not in _NATIVE_SCREEN_STATUSES for status in status_codes):
        raise RuntimeError("native candidate screening returned an unknown status")
    expected_digest = _native_screening_digest(
        candidate_ids,
        statuses,
        duplicate_of,
        codes,
        metrics,
        raw_counters,
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
    duplicate_candidates = sum(status == 1 for status in status_codes)
    negative_cache_hits = sum(status == 2 for status in status_codes)
    screened_candidates = sum(status == 0 for status in status_codes)
    if (
        counters["input_candidates"] != len(candidates)
        or counters["unique_candidates"] != len(candidates) - duplicate_candidates
        or counters["duplicate_candidates"] != duplicate_candidates
        or counters["negative_cache_hits"] != negative_cache_hits
        or counters["screened_candidates"] != screened_candidates
    ):
        raise RuntimeError("native candidate screening counters are inconsistent")
    first_by_sequence: dict[CustomerSequence, int] = {}
    for index, status in enumerate(status_codes):
        source = int(duplicate_of[index])
        expected_source = first_by_sequence.get(candidates[index])
        if expected_source is None:
            first_by_sequence[candidates[index]] = index
            if status == 1:
                raise RuntimeError("native first candidate is marked as a duplicate")
        elif status != 1 or source != expected_source:
            raise RuntimeError("native repeated candidate lacks its first duplicate identity")
        if status != 1:
            if source != -1:
                raise RuntimeError("native non-duplicate candidate has a duplicate identity")
            if status == 2 and (bool(codes[index, 0]) or int(codes[index, 1]) == 0):
                raise RuntimeError("native negative-cache hit lacks its safe rejection reason")
            continue
        if (
            source < 0
            or source >= index
            or not np.array_equal(codes[source], codes[index])
            or not np.array_equal(metrics[source], metrics[index], equal_nan=True)
        ):
            raise RuntimeError("native duplicate candidate identity is inconsistent")
    return CandidateScreeningBatch(
        candidates,
        returned_ids,
        statuses,
        duplicate_of,
        codes,
        metrics,
        route_offsets,
        route_indices,
        raw_counters,
        counters,
        digest,
    )


def execute_candidate_transaction[ResultT](
    request: CandidateTransactionRequest,
    *,
    screen_batch: Callable[[tuple[CustomerSequence, ...]], CandidateScreeningBatch],
    cache_lookup: Callable[[CustomerSequence], ResultT | None],
    exact_batch: Callable[[tuple[CustomerSequence, ...]], Sequence[ResultT]],
    stage_cache_write: Callable[[CustomerSequence, ResultT], None],
    commit_cache_writes: Callable[[], None],
    rollback_cache_writes: Callable[[str], None],
    rejected_result: Callable[[CustomerSequence, str, str], ResultT],
    skipped_result: Callable[[str], ResultT],
    clock: Callable[[], float] = time.perf_counter,
) -> CandidateTransactionResult[ResultT]:
    """Execute the fixed Stage 5.2 transaction order with atomic cache writes."""

    staged = False
    try:
        _check_deadline(request, "before_native_screening", clock)
        screening = screen_batch(request.candidates)
        if len(screening.sequences) != len(request.candidates):
            raise RuntimeError("native screening lost candidate rows")
        if screening.sequences != request.candidates:
            raise RuntimeError("native screening changed candidate identity or order")

        resolved: list[ResultT | None] = [None] * len(request.candidates)
        waiting_indices: dict[CustomerSequence, list[int]] = {}
        exact_sequences: list[CustomerSequence] = []
        cache_hits = 0
        screening_passes = 0
        screening_rejections = 0
        screening_cache_hits = 0
        screening_exact_call_blocked = 0
        screening_reason_counts: dict[str, int] = {}
        budget_skips = 0
        duplicate_candidates = 0
        known_resolution: dict[CustomerSequence, ResultT | None] = {}

        for index, sequence in enumerate(screening.sequences):
            native_status = screening.native_status(index)
            reason = screening.reason(index)
            if native_status == "duplicate":
                duplicate_candidates += 1
            if not screening.accepted(index):
                screening_exact_call_blocked += 1
                if native_status == "negative_cache_hit":
                    screening_cache_hits += 1
                else:
                    screening_rejections += 1
                if reason:
                    screening_reason_counts[reason] = screening_reason_counts.get(reason, 0) + 1
                resolved[index] = rejected_result(sequence, reason, native_status)
                continue
            screening_passes += 1
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
        audit = CandidateTransactionAudit(
            lane=request.lane,
            operator=request.operator,
            iteration=request.iteration,
            input_candidates=len(request.candidates),
            candidates=request.candidates,
            exact_budget=request.exact_budget,
            screening_integrity_evidence=screening.integrity_evidence(),
            screening_passes=screening_passes,
            screening_rejections=screening_rejections,
            screening_cache_hits=screening_cache_hits,
            screening_exact_call_blocked=screening_exact_call_blocked,
            screening_reason_counts=dict(sorted(screening_reason_counts.items())),
            cache_hits=cache_hits,
            exact_misses=len(exact_sequences),
            budget_skips=budget_skips,
            duplicate_candidates=duplicate_candidates,
            screening_pool_hash=screening.candidate_pool_hash,
            transaction_sha256=transaction_sha256,
        )
        commit_cache_writes()
        staged = False
        return CandidateTransactionResult(
            tuple(result for result in resolved if result is not None),
            audit,
        )
    except BaseException as error:
        rollback_cache_writes(f"candidate_transaction_rollback:{type(error).__name__}:{error}")
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
        raise RuntimeError(f"native {name} must be C-contiguous {dtype} with shape {shape}")
    return value


def _grow_int64_buffer(
    current: npt.NDArray[np.int64],
    required: int,
) -> npt.NDArray[np.int64]:
    if required <= len(current):
        return current
    capacity = max(required, max(8, len(current) * 2))
    grown = np.empty(capacity, dtype=np.int64)
    grown[: len(current)] = current
    return grown


def _native_screening_digest(
    candidate_ids: npt.NDArray[np.int64],
    statuses: npt.NDArray[np.int64],
    duplicate_of: npt.NDArray[np.int64],
    codes: npt.NDArray[np.int64],
    metrics: npt.NDArray[np.float64],
    counters: npt.NDArray[np.int64],
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
            end - begin,
            *(int(value) for value in route_indices[begin:end]),
            *(int(value) for value in codes[index]),
        ):
            payload.extend(struct.pack("<q", value))
        for value in metrics[index]:
            payload.extend(struct.pack("<d", float(value)))
    for value in counters:
        payload.extend(struct.pack("<q", int(value)))
    return hashlib.sha256(payload).hexdigest()
