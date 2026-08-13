"""Executable T4 bounds for one atomic, not-yet-published transaction.

``uncommitted_window`` is deliberately not the executor's physical submission
width.  A runtime that publishes a whole candidate batch atomically retains all
work in that batch until the single commit, including results that have already
completed.  T4 therefore counts the full work-bearing transaction window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WasteBound:
    discarded_work_units: int
    discarded_cost: float
    post_boundary_work_units: int
    post_boundary_cost: float


@dataclass(frozen=True, slots=True)
class WasteAudit:
    """Observed unit-level waste checked against one declared T4 bound."""

    remaining_budget_before: int
    uncommitted_window: int
    max_requests_per_candidate: int
    post_boundary_capacity_units: int
    observed_discarded_work_units: int
    observed_post_boundary_work_units: int
    bound: WasteBound


class WasteBoundViolation(RuntimeError):
    """Raised when runtime observations exceed their receipt-bound T4 limit."""


def bounded_waste(
    *,
    remaining_budget: int,
    uncommitted_window: int,
    max_requests_per_candidate: int,
    max_request_cost: float,
    post_boundary_capacity_units: int,
) -> WasteBound:
    """Evaluate T4 for one atomic transaction parameter tuple."""

    integer_fields = (
        remaining_budget,
        uncommitted_window,
        max_requests_per_candidate,
        post_boundary_capacity_units,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integer_fields
    ):
        raise ValueError("T4 integer parameters must be non-negative integers")
    if (
        isinstance(max_request_cost, bool)
        or not isinstance(max_request_cost, (int, float))
        or not math.isfinite(float(max_request_cost))
        or float(max_request_cost) < 0.0
    ):
        raise ValueError("max_request_cost must be finite and non-negative")
    window_capacity = uncommitted_window * max_requests_per_candidate
    discarded_units = min(remaining_budget, window_capacity)
    post_boundary_units = min(post_boundary_capacity_units, window_capacity)
    request_cost = float(max_request_cost)
    return WasteBound(
        discarded_work_units=discarded_units,
        discarded_cost=discarded_units * request_cost,
        post_boundary_work_units=post_boundary_units,
        post_boundary_cost=post_boundary_units * request_cost,
    )


def audit_waste(
    *,
    remaining_budget_before: int,
    uncommitted_window: int,
    max_requests_per_candidate: int,
    post_boundary_capacity_units: int,
    observed_discarded_work_units: int,
    observed_post_boundary_work_units: int,
) -> WasteAudit:
    """Fail closed if a fixed-work transaction violates its T4 unit bound.

    Cost is intentionally normalized to one per work unit here.  A physical
    cost claim still needs an independently measured finite ``Cmax``.
    """

    observed = (observed_discarded_work_units, observed_post_boundary_work_units)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in observed
    ):
        raise ValueError("observed T4 work must be non-negative integers")
    bound = bounded_waste(
        remaining_budget=remaining_budget_before,
        uncommitted_window=uncommitted_window,
        max_requests_per_candidate=max_requests_per_candidate,
        max_request_cost=1.0,
        post_boundary_capacity_units=post_boundary_capacity_units,
    )
    if observed_discarded_work_units > bound.discarded_work_units:
        raise WasteBoundViolation("discarded work exceeds the atomic-window T4 bound")
    if observed_post_boundary_work_units > bound.post_boundary_work_units:
        raise WasteBoundViolation("post-boundary work exceeds the declared T4 capacity")
    return WasteAudit(
        remaining_budget_before=remaining_budget_before,
        uncommitted_window=uncommitted_window,
        max_requests_per_candidate=max_requests_per_candidate,
        post_boundary_capacity_units=post_boundary_capacity_units,
        observed_discarded_work_units=observed_discarded_work_units,
        observed_post_boundary_work_units=observed_post_boundary_work_units,
        bound=bound,
    )
