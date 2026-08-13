"""Solve-local ordered cache transactions for ``txnopt-contract-v1``."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field


class CacheCommitError(RuntimeError):
    """A staged cache transaction could not be published."""


class StaleCacheSnapshotError(CacheCommitError):
    """The cache generation changed after transaction preparation."""


@dataclass(slots=True)
class CacheTxn[ValueT]:
    expected_generation: int
    staged: dict[str, ValueT] = field(default_factory=dict)
    closed: bool = False


@dataclass(frozen=True, slots=True)
class CacheLookup[ValueT]:
    generation: int
    values: tuple[ValueT | None, ...]


@dataclass(frozen=True, slots=True)
class CacheCommitReceipt:
    generation_before: int
    generation_after: int
    writes: int


class InMemoryCacheStore[ValueT]:
    """Reference cache store with one atomic generation per committed round."""

    __slots__ = ("_generation", "_on_commit", "_values")

    def __init__(
        self,
        *,
        on_commit: Callable[[Mapping[str, ValueT]], None] | None = None,
    ) -> None:
        self._generation = 0
        self._values: dict[str, ValueT] = {}
        self._on_commit = on_commit

    @property
    def generation(self) -> int:
        return self._generation

    def begin(self) -> CacheTxn[ValueT]:
        return CacheTxn(expected_generation=self._generation)

    def lookup(self, keys: Sequence[str]) -> CacheLookup[ValueT]:
        return CacheLookup(
            generation=self._generation,
            values=tuple(self._values.get(key) for key in keys),
        )

    def stage(self, transaction: CacheTxn[ValueT], entries: Mapping[str, ValueT]) -> None:
        self._require_open(transaction)
        if any(not key for key in entries):
            raise ValueError("cache keys cannot be empty")
        transaction.staged.update(entries)

    def commit(self, transaction: CacheTxn[ValueT]) -> CacheCommitReceipt:
        self._require_open(transaction)
        if transaction.expected_generation != self._generation:
            transaction.closed = True
            transaction.staged.clear()
            raise StaleCacheSnapshotError("cache generation changed before commit")
        if self._on_commit is not None:
            try:
                self._on_commit(dict(transaction.staged))
            except Exception as error:
                transaction.closed = True
                transaction.staged.clear()
                raise CacheCommitError("cache publication failed") from error
        before = self._generation
        self._values.update(transaction.staged)
        self._generation += 1
        writes = len(transaction.staged)
        transaction.closed = True
        transaction.staged.clear()
        return CacheCommitReceipt(before, self._generation, writes)

    def rollback(self, transaction: CacheTxn[ValueT]) -> None:
        if transaction.closed:
            return
        transaction.staged.clear()
        transaction.closed = True

    def snapshot(self) -> tuple[int, Mapping[str, ValueT]]:
        return self._generation, dict(self._values)

    @staticmethod
    def _require_open(transaction: CacheTxn[ValueT]) -> None:
        if transaction.closed:
            raise CacheCommitError("cache transaction is already closed")
