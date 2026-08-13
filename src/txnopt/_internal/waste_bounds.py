"""Executable form of the proposed T4 speculative-waste upper bounds."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WasteBound:
    discarded_work_units: int
    discarded_cost: float
    post_boundary_work_units: int
    post_boundary_cost: float


def bounded_waste(
    *,
    remaining_budget: int,
    speculation_window: int,
    max_requests_per_candidate: int,
    max_request_cost: float,
    parallelism: int,
) -> WasteBound:
    """Evaluate T4 for one declared runtime parameter tuple."""

    integer_fields = (
        remaining_budget,
        speculation_window,
        max_requests_per_candidate,
        parallelism,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in integer_fields
    ):
        raise ValueError("T4 integer parameters must be non-negative integers")
    if parallelism == 0:
        raise ValueError("parallelism must be positive")
    if (
        isinstance(max_request_cost, bool)
        or not isinstance(max_request_cost, (int, float))
        or not math.isfinite(float(max_request_cost))
        or float(max_request_cost) < 0.0
    ):
        raise ValueError("max_request_cost must be finite and non-negative")
    window_capacity = speculation_window * max_requests_per_candidate
    discarded_units = min(remaining_budget, window_capacity)
    post_boundary_units = min(parallelism, window_capacity)
    request_cost = float(max_request_cost)
    return WasteBound(
        discarded_work_units=discarded_units,
        discarded_cost=discarded_units * request_cost,
        post_boundary_work_units=post_boundary_units,
        post_boundary_cost=post_boundary_units * request_cost,
    )
