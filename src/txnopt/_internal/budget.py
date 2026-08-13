"""Solve-local work-at-start accounting for ``txnopt-contract-v1``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BudgetReceipt:
    requested: int
    reserved_before: int
    reserved_after: int
    started_before: int
    started_after: int
    remaining_after: int | None
    granted: bool


class BudgetLedger:
    """Single-owner exact-work ledger; reservations are never silently refunded."""

    __slots__ = ("_limit", "_reserved", "_started")

    def __init__(self, limit: int | None) -> None:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            raise ValueError("budget limit must be a positive integer or None")
        self._limit = limit
        self._reserved = 0
        self._started = 0

    @property
    def started(self) -> int:
        return self._started

    @property
    def remaining(self) -> int | None:
        return (
            None
            if self._limit is None
            else self._limit - self._started - self._reserved
        )

    @property
    def reserved(self) -> int:
        return self._reserved

    def reserve(self, count: int) -> BudgetReceipt:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("budget reservation count must be non-negative")
        reserved_before = self._reserved
        started_before = self._started
        remaining = self.remaining
        granted = remaining is None or count <= remaining
        if granted:
            self._reserved += count
        return BudgetReceipt(
            requested=count,
            reserved_before=reserved_before,
            reserved_after=self._reserved,
            started_before=started_before,
            started_after=self._started,
            remaining_after=self.remaining,
            granted=granted,
        )

    def start(self, count: int) -> None:
        """Move admitted work into the irreversible started counter."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("started work count must be non-negative")
        if count > self._reserved:
            raise ValueError("started work exceeds the active reservation")
        self._reserved -= count
        self._started += count

    def release_reserved(self) -> int:
        """Release work that never crossed the execution boundary."""

        released = self._reserved
        self._reserved = 0
        return released
