from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from txnopt import RunConfig
from txnopt._internal.cache import InMemoryCacheStore, StaleCacheSnapshotError
from txnopt._internal.waste_bounds import audit_waste
from txnopt.runtime import PythonTxnRuntime, SerialTxnRuntime


class _ListKernel:
    def __init__(self, candidates: Sequence[int]) -> None:
        self._candidates = tuple(candidates)

    def propose(
        self,
        snapshot: int,
        *,
        round_id: int,
        random_tape: Sequence[int],
    ) -> Sequence[int]:
        return self._candidates

    def decide(
        self,
        snapshot: int,
        candidates: Sequence[int],
        evaluated_states: Sequence[int],
        *,
        round_id: int,
    ) -> int:
        return evaluated_states[0]


class _RecordingOracle:
    deterministic = True
    parallel_safe = True
    internal_parallelism = False

    def __init__(self) -> None:
        self.batches: list[tuple[int, ...]] = []

    def stable_key(self, candidate: int) -> str:
        return f"property:{candidate}"

    def state_digest(self, state: int) -> str:
        return hashlib.sha256(str(state).encode()).hexdigest()

    def work_units(self, candidate: int) -> int:
        return 1

    def screen(self, candidates: Sequence[int]) -> Sequence[bool]:
        return tuple(True for _candidate in candidates)

    def evaluate_batch(
        self,
        candidates: Sequence[int],
        *,
        work_budget: int,
        deadline_ns: int | None,
    ) -> Sequence[int]:
        self.batches.append(tuple(candidates))
        return tuple(candidates)

    def validate(self, state: int) -> None:
        if state < 0:
            raise ValueError("negative state")

    def objective(self, state: int) -> int:
        return state


@st.composite
def _admitted_work(draw: st.DrawFn) -> tuple[tuple[int, ...], int, int, int, int]:
    work = tuple(
        draw(st.lists(st.integers(1, 8), min_size=1, max_size=32))
    )
    total = sum(work)
    remaining = draw(st.integers(total, total + 50))
    started = draw(st.integers(0, total))
    post_capacity = draw(st.integers(0, total))
    post_observed = draw(st.integers(0, min(post_capacity, total)))
    return work, remaining, started, post_capacity, post_observed


@given(_admitted_work())
@settings(max_examples=200, deadline=None)
def test_t4_atomic_window_bound_holds_for_admitted_work(
    case: tuple[tuple[int, ...], int, int, int, int],
) -> None:
    work, remaining, started, post_capacity, post_observed = case
    receipt = audit_waste(
        remaining_budget_before=remaining,
        uncommitted_window=len(work),
        max_requests_per_candidate=max(work),
        post_boundary_capacity_units=post_capacity,
        observed_discarded_work_units=started,
        observed_post_boundary_work_units=post_observed,
    )
    assert receipt.observed_discarded_work_units <= receipt.bound.discarded_work_units
    assert receipt.observed_post_boundary_work_units <= receipt.bound.post_boundary_work_units


@given(
    st.dictionaries(
        st.text(alphabet="abcde", min_size=1, max_size=8),
        st.integers(),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=100, deadline=None)
def test_cache_conflict_never_publishes_a_stale_transaction(
    current_values: dict[str, int],
) -> None:
    store = InMemoryCacheStore[int]()
    stale = store.begin()
    current = store.begin()
    store.stage(current, current_values)
    store.commit(current)
    store.stage(stale, {"stale": -1})

    with pytest.raises(StaleCacheSnapshotError):
        store.commit(stale)
    assert store.snapshot() == (1, current_values)


@given(st.lists(st.integers(0, 20), min_size=1, max_size=20))
@settings(max_examples=100, deadline=None)
def test_duplicate_candidate_keys_are_evaluated_once(candidates: list[int]) -> None:
    unique = tuple(dict.fromkeys(candidates))
    oracle = _RecordingOracle()
    result = SerialTxnRuntime[int, int, int]().run(
        0,
        kernel=_ListKernel(candidates),
        oracle=oracle,
        config=RunConfig(
            seed=0,
            workers=1,
            execution_mode="serial",
            fixed_work=len(unique),
            max_rounds=1,
        ),
    )

    assert oracle.batches == [unique]
    assert result.last_committed_state == unique[0]


@given(
    st.lists(st.integers(1, 100), min_size=2, max_size=20, unique=True),
    st.data(),
)
@settings(max_examples=100, deadline=None)
def test_insufficient_budget_never_starts_or_caches_a_partial_batch(
    candidates: list[int],
    data: st.DataObject,
) -> None:
    budget = data.draw(st.integers(1, len(candidates) - 1))
    oracle = _RecordingOracle()
    cache = InMemoryCacheStore[int]()
    result = SerialTxnRuntime[int, int, int](cache_factory=lambda: cache).run(
        0,
        kernel=_ListKernel(candidates),
        oracle=oracle,
        config=RunConfig(
            seed=0,
            workers=1,
            execution_mode="serial",
            fixed_work=budget,
            max_rounds=1,
        ),
    )

    assert result.termination_reason == "fixed_work_exhausted"
    assert oracle.batches == []
    assert cache.snapshot() == (0, {})


@given(st.lists(st.integers(1, 100), min_size=1, max_size=20, unique=True))
@settings(max_examples=100, deadline=None)
def test_late_complete_batch_is_rolled_back(candidates: list[int]) -> None:
    ticks = iter((0, 0, 2))
    cache = InMemoryCacheStore[int]()
    oracle = _RecordingOracle()
    result = PythonTxnRuntime[int, int, int](
        clock_ns=lambda: next(ticks),
        cache_factory=lambda: cache,
    ).run(
        0,
        kernel=_ListKernel(candidates),
        oracle=oracle,
        config=RunConfig(
            seed=0,
            workers=1,
            execution_mode="serial",
            deadline_seconds=0.000000001,
            max_rounds=1,
        ),
    )

    assert result.termination_reason == "deadline"
    assert result.last_committed_state == 0
    assert cache.snapshot() == (0, {})
