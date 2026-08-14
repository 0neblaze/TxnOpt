from __future__ import annotations

import pytest

from txnopt._internal.budget import BudgetLedger
from txnopt._internal.cache import (
    CacheCommitError,
    InMemoryCacheStore,
    StaleCacheSnapshotError,
)


def test_budget_ledger_reserves_complete_batches_at_work_start() -> None:
    ledger = BudgetLedger(3)

    first = ledger.reserve(2)

    assert first.granted is True
    assert (first.reserved_before, first.reserved_after, first.remaining_after) == (
        0,
        2,
        1,
    )
    ledger.start(1)
    rejected = ledger.reserve(2)
    assert rejected.granted is False
    assert (rejected.started_before, rejected.started_after) == (1, 1)
    ledger.start(1)
    final = ledger.reserve(1)
    assert final.granted is True
    ledger.start(1)
    assert ledger.started == 3
    assert ledger.remaining == 0


def test_budget_ledger_releases_only_work_that_never_started() -> None:
    ledger = BudgetLedger(4)
    assert ledger.reserve(4).granted is True
    ledger.start(2)

    assert ledger.release_reserved() == 2
    assert ledger.started == 2
    assert ledger.reserved == 0
    assert ledger.remaining == 2


def test_cache_staging_is_invisible_until_atomic_commit() -> None:
    store = InMemoryCacheStore[int]()
    transaction = store.begin()
    store.stage(transaction, {"a": 1})

    assert store.lookup(("a",)).values == (None,)
    receipt = store.commit(transaction)

    assert receipt.writes == 1
    assert receipt.generation_before == 0
    assert receipt.generation_after == 1
    assert store.lookup(("a",)).values == (1,)


def test_cache_rollback_and_publication_failure_leave_no_visible_writes() -> None:
    store = InMemoryCacheStore[int]()
    rolled_back = store.begin()
    store.stage(rolled_back, {"a": 1})
    store.rollback(rolled_back)
    assert store.lookup(("a",)).values == (None,)

    def reject(_entries: object) -> None:
        raise OSError("simulated publication failure")

    failing = InMemoryCacheStore[int](on_commit=reject)
    failed = failing.begin()
    failing.stage(failed, {"a": 1})
    with pytest.raises(CacheCommitError, match="publication failed"):
        failing.commit(failed)
    assert failing.snapshot() == (0, {})


def test_cache_rejects_a_stale_generation() -> None:
    store = InMemoryCacheStore[int]()
    stale = store.begin()
    current = store.begin()
    store.commit(current)
    store.stage(stale, {"a": 1})

    with pytest.raises(StaleCacheSnapshotError, match="generation changed"):
        store.commit(stale)
    assert store.snapshot() == (1, {})
