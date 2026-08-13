"""Minimal domain state shared by future RCPSP kernels and validators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RCPSPState:
    """A precedence-feasible activity order paired with one mode per activity.

    Precedence feasibility is an instance-dependent property and therefore must
    be established by the RCPSP adapter's independent validator. This value
    object enforces only representation-level invariants.
    """

    activity_order: tuple[int, ...]
    mode_vector: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.activity_order:
            raise ValueError("activity_order cannot be empty")
        if len(set(self.activity_order)) != len(self.activity_order):
            raise ValueError("activity_order must contain each activity once")
        if any(
            isinstance(activity, bool) or not isinstance(activity, int) or activity < 0
            for activity in self.activity_order
        ):
            raise ValueError("activity identifiers must be non-negative integers")
        if len(self.mode_vector) != len(self.activity_order):
            raise ValueError("mode_vector must align with activity_order")
        if any(
            isinstance(mode, bool) or not isinstance(mode, int) or mode < 0
            for mode in self.mode_vector
        ):
            raise ValueError("mode identifiers must be non-negative integers")
