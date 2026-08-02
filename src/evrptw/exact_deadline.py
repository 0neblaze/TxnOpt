"""Stage 3.3 exact-call budgets and deadline controls.

The controller is process-local and deliberately shared by every ALNS lane so
an exact-call budget describes the complete solve rather than one evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EXACT_DEADLINE_SCHEMA_VERSION = "stage033-exact-deadline-v1"
ExactDeadlineMode = Literal["wall_clock", "exact_call_budget"]


@dataclass(frozen=True, slots=True)
class ExactDeadlineConfig:
    """Opt-in Stage 3.3 stopping semantics for exact route evaluation."""

    mode: ExactDeadlineMode
    exact_call_budget: int | None = None
    watchdog_seconds: float | None = None
    schema_version: str = EXACT_DEADLINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXACT_DEADLINE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Stage 3.3 exact deadline schema {self.schema_version}"
            )
        if self.mode == "wall_clock":
            if self.exact_call_budget is not None or self.watchdog_seconds is not None:
                raise ValueError(
                    "wall-clock exact deadline cannot define an exact-call budget or watchdog"
                )
            return
        if self.mode != "exact_call_budget":
            raise ValueError(f"unsupported exact deadline mode: {self.mode}")
        if self.exact_call_budget is None or self.exact_call_budget <= 0:
            raise ValueError("exact-call budget must be positive")
        if self.watchdog_seconds is None or self.watchdog_seconds <= 0.0:
            raise ValueError("fixed exact-call watchdog must be positive")

    @classmethod
    def wall_clock(cls) -> ExactDeadlineConfig:
        return cls("wall_clock")

    @classmethod
    def fixed_exact_calls(
        cls,
        exact_call_budget: int,
        *,
        watchdog_seconds: float,
    ) -> ExactDeadlineConfig:
        return cls(
            "exact_call_budget",
            exact_call_budget=exact_call_budget,
            watchdog_seconds=watchdog_seconds,
        )


@dataclass(frozen=True, slots=True)
class ExactCallReservation:
    requested: int
    granted: int

    @property
    def partial(self) -> bool:
        return self.granted < self.requested


@dataclass(slots=True)
class ExactCallController:
    """Count started/completed exact calls and enforce one global cap."""

    config: ExactDeadlineConfig
    started_calls: int = 0
    completed_calls: int = 0
    interrupted_calls: int = 0
    boundary_recorded: bool = False

    @property
    def budget(self) -> int | None:
        return self.config.exact_call_budget

    @property
    def budget_reached(self) -> bool:
        return self.budget is not None and self.started_calls >= self.budget

    @property
    def budget_exhaustions(self) -> int:
        return int(self.budget_reached)

    def reserve(self, requested: int) -> ExactCallReservation:
        if requested <= 0:
            raise ValueError("exact-call reservation must be positive")
        if self.budget is None:
            granted = requested
        else:
            granted = min(requested, max(0, self.budget - self.started_calls))
        self.started_calls += granted
        return ExactCallReservation(requested, granted)

    def complete(self, count: int) -> None:
        if (
            count < 0
            or self.completed_calls + self.interrupted_calls + count
            > self.started_calls
        ):
            raise RuntimeError("invalid completed exact-call count")
        self.completed_calls += count

    def interrupt(self, count: int) -> None:
        if count < 0 or self.completed_calls + self.interrupted_calls + count > self.started_calls:
            raise RuntimeError("invalid interrupted exact-call count")
        self.interrupted_calls += count

    def record_native_work(
        self,
        *,
        started: int,
        completed: int,
        interrupted: int,
    ) -> bool:
        """Record an already-executed native receipt without refund semantics."""

        if (
            started < 0
            or completed < 0
            or interrupted < 0
            or completed + interrupted != started
        ):
            raise RuntimeError("invalid native exact-call receipt")
        within_budget = self.budget is None or self.started_calls + started <= self.budget
        self.started_calls += started
        self.completed_calls += completed
        self.interrupted_calls += interrupted
        return within_budget

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.config.schema_version,
            "mode": self.config.mode,
            "exact_call_budget": self.config.exact_call_budget,
            "watchdog_seconds": self.config.watchdog_seconds,
            "started_calls": self.started_calls,
            "completed_calls": self.completed_calls,
            "interrupted_calls": self.interrupted_calls,
            "budget_exhaustions": self.budget_exhaustions,
        }
