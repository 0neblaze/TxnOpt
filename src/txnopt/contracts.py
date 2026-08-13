"""Stable, domain-independent TxnOpt root contracts.

The root API describes ownership and ordering. Candidate transactions, budget
ledgers, cache transactions, random tapes, and trace records intentionally live
below this seam and are not root exports.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

ExecutionMode = Literal["serial", "barrier", "ordered"]
TracePolicy = Literal["none", "semantic", "semantic_and_physical"]

_SEMANTIC_DIGEST = re.compile(r"[0-9a-f]{64}")
_TERMINATION_REASON = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Immutable runtime policy for exactly one fixed-work or deadline run."""

    seed: int
    workers: int
    execution_mode: ExecutionMode
    fixed_work: int | None = None
    deadline_seconds: float | None = None
    speculation_window: int = 0
    trace_policy: TracePolicy = "semantic"
    max_rounds: int = 1000

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or self.workers <= 0
        ):
            raise ValueError("workers must be a positive integer")
        if self.execution_mode not in {"serial", "barrier", "ordered"}:
            raise ValueError("execution_mode is unsupported")
        if self.execution_mode == "serial" and self.workers != 1:
            raise ValueError("serial execution requires exactly one worker")
        if (
            isinstance(self.speculation_window, bool)
            or not isinstance(self.speculation_window, int)
            or self.speculation_window < 0
        ):
            raise ValueError("speculation_window must be a non-negative integer")
        if self.execution_mode == "serial" and self.speculation_window != 0:
            raise ValueError("serial execution cannot speculate")
        if self.execution_mode == "ordered" and self.speculation_window <= 0:
            raise ValueError("ordered execution requires a positive speculation window")
        has_fixed_work = self.fixed_work is not None
        has_deadline = self.deadline_seconds is not None
        if has_fixed_work == has_deadline:
            raise ValueError("exactly one of fixed_work and deadline_seconds is required")
        if has_fixed_work and (
            isinstance(self.fixed_work, bool)
            or not isinstance(self.fixed_work, int)
            or self.fixed_work <= 0
        ):
            raise ValueError("fixed_work must be a positive integer")
        if has_deadline and (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not math.isfinite(float(self.deadline_seconds))
            or float(self.deadline_seconds) <= 0.0
        ):
            raise ValueError("deadline_seconds must be finite and positive")
        if self.trace_policy not in {"none", "semantic", "semantic_and_physical"}:
            raise ValueError("trace_policy is unsupported")
        if (
            isinstance(self.max_rounds, bool)
            or not isinstance(self.max_rounds, int)
            or self.max_rounds <= 0
        ):
            raise ValueError("max_rounds must be a positive integer")


@dataclass(frozen=True, slots=True)
class RunResult[StateT, ObjectiveT]:
    """The last complete commit prefix returned by a TxnOpt run."""

    last_committed_state: StateT
    objective: ObjectiveT
    termination_reason: str
    semantic_digest: str
    physical_artifact_ref: str | None
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        if _TERMINATION_REASON.fullmatch(self.termination_reason) is None:
            raise ValueError("termination_reason must be canonical")
        if _SEMANTIC_DIGEST.fullmatch(self.semantic_digest) is None:
            raise ValueError("semantic_digest must be a lowercase SHA-256")
        if self.physical_artifact_ref is not None and not self.physical_artifact_ref:
            raise ValueError("physical_artifact_ref cannot be empty")
        frozen_provenance = dict(self.provenance)
        if not frozen_provenance or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in frozen_provenance.items()
        ):
            raise ValueError("provenance requires non-empty string keys and values")
        object.__setattr__(self, "provenance", MappingProxyType(frozen_provenance))


@runtime_checkable
class SearchKernel[StateT, CandidateT](Protocol):
    """Pure proposal and decision logic over immutable runtime snapshots."""

    def propose(
        self,
        snapshot: StateT,
        *,
        round_id: int,
        random_tape: Sequence[int],
    ) -> Sequence[CandidateT]:
        """Return candidates in their canonical deterministic order."""

        ...

    def decide(
        self,
        snapshot: StateT,
        candidates: Sequence[CandidateT],
        evaluated_states: Sequence[StateT],
        *,
        round_id: int,
    ) -> StateT:
        """Return a proposed next state without publishing it."""

        ...


@runtime_checkable
class Oracle[OracleCandidateT, OracleStateT, OracleObjectiveT](Protocol):
    """Domain adapter for safe screening and deterministic expensive work."""

    @property
    def deterministic(self) -> bool:
        """Declare whether repeated evaluation is semantically deterministic."""

        ...

    @property
    def parallel_safe(self) -> bool:
        """Declare whether independent batch calls may overlap safely."""

        ...

    @property
    def internal_parallelism(self) -> bool:
        """Declare that one batch call owns all configured worker parallelism."""

        ...

    def stable_key(self, candidate: OracleCandidateT) -> str:
        """Return the canonical cache and deduplication key."""

        ...

    def work_units(self, candidate: OracleCandidateT) -> int:
        """Return the bounded exact-request count for fixed-work accounting."""

        ...

    def state_digest(self, state: OracleStateT) -> str:
        """Return the canonical semantic digest of a validated state."""

        ...

    def screen(self, candidates: Sequence[OracleCandidateT]) -> Sequence[bool]:
        """Return ordered safe-admission decisions; False is a proof of rejection."""

        ...

    def evaluate_batch(
        self,
        candidates: Sequence[OracleCandidateT],
        *,
        work_budget: int,
        deadline_ns: int | None,
    ) -> Sequence[OracleStateT]:
        """Evaluate an ordered batch without committing state or cache writes."""

        ...

    def validate(self, state: OracleStateT) -> None:
        """Raise if an evaluated result cannot be independently validated."""

        ...

    def objective(self, state: OracleStateT) -> OracleObjectiveT:
        """Construct the domain objective through its single authoritative path."""

        ...


@runtime_checkable
class TxnRuntime[StateT, CandidateT, ObjectiveT](Protocol):
    """Sole live owner of state, budget, cache transactions, and trace commit."""

    def run(
        self,
        initial_state: StateT,
        *,
        kernel: SearchKernel[StateT, CandidateT],
        oracle: Oracle[CandidateT, StateT, ObjectiveT],
        config: RunConfig,
    ) -> RunResult[StateT, ObjectiveT]:
        """Return only the last complete committed prefix."""

        ...
