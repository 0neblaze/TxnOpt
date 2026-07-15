"""Stage 4 adaptive weight and search control configuration.

Stage 4 restructures the ALNS adaptive mechanism: segment-based weight update,
six-category operator statistics, auto-estimated SA temperature, reheating,
stagnation restart, and incumbent intensification.  It inherits the Stage 2.3
dynamic removal-size adaptation as a fixed input and does not re-implement it.

When ``stage04_config`` is ``None`` or ``enabled`` is ``False``, the existing
per-call weight update and linear SA cooling schedule are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evrptw.instance import Instance  # noqa: F401

__all__ = ("Stage04Config",)


_REWARD_FLOOR = 0.0
_REWARD_CEIL = 64.0
_DEFAULT_SEGMENT_LENGTH = 50
_DEFAULT_MIN_CALLS = 5
_DEFAULT_WEIGHT_REACTION = 0.2
_DEFAULT_WEIGHT_FLOOR = 0.05
_DEFAULT_WEIGHT_SMOOTHING = 0.8
_DEFAULT_TEMP_TARGET = 0.5
_DEFAULT_TEMP_SAMPLE = 30
_DEFAULT_TEMP_FALLBACK = 0.05
_DEFAULT_REHEAT_THRESHOLD = 10
_DEFAULT_REHEAT_FACTOR = 0.5
_DEFAULT_MAX_REHEATS = 5
_DEFAULT_RESTART_THRESHOLD = 20
_DEFAULT_MAX_RESTARTS = 3
_DEFAULT_INTENSIFICATION_ITERS = 10
_DEFAULT_INTENSIFICATION_FRACTION = 0.10
_DEFAULT_FIXED_WEIGHT = 1.0

# Differentiated reward tiers.
_DEFAULT_REWARD_REJECTED: float = 0.0
_DEFAULT_REWARD_ACCEPTED_WORSE: float = 0.5
_DEFAULT_REWARD_ACCEPTED_EQUAL: float = 1.0
_DEFAULT_REWARD_DISTANCE: float = 4.0
_DEFAULT_REWARD_VEHICLE: float = 8.0
_DEFAULT_REWARD_BEST_DISTANCE: float = 8.0
_DEFAULT_REWARD_BEST_VEHICLE: float = 16.0


@dataclass(frozen=True, slots=True)
class Stage04Config:
    """Configuration for Stage 4 adaptive weights and search control.

    All parameters are opt-in: when ``enabled`` is ``False`` or the config is
    ``None``, the ALNS solver preserves its existing per-call weight update and
    linear SA cooling schedule.

    Segment-based weight update
    ---------------------------
    Instead of updating operator weights after every single call, rewards are
    accumulated over a learning period (``segment_length`` iterations).  At
    each segment boundary, the weight of every operator that received at least
    ``min_calls_per_operator`` calls is updated in batch.  Operators that did
    not meet the minimum are left unchanged to avoid noise from small samples.

    Six-category statistics
    ----------------------
    ``accepted`` is split into ``accepted_improving``, ``accepted_equal`` and
    ``accepted_worse`` to distinguish SA-accepted moves that degrade the
    incumbent from those that improve it.

    Differentiated rewards
    ----------------------
    Vehicle-count reduction receives a higher reward than distance
    improvement.  A new global best that also reduces vehicles receives the
    highest reward tier.

    Auto temperature estimation
    ---------------------------
    Before the main loop, a small sample of random candidates is evaluated to
    estimate the distance delta distribution.  The initial temperature is
    calibrated so that the theoretical worse-solution acceptance rate equals
    ``temperature_target_acceptance_rate``.

    Reheating and restart
    ---------------------
    When stagnation exceeds ``reheat_stagnation_threshold``, the temperature
    is boosted.  When it exceeds ``restart_stagnation_threshold``, the current
    solution is reset to the global best and the search is intensified around
    it for ``intensification_iterations`` iterations.
    """

    enabled: bool = True

    # Segment-based weight update.
    segment_length: int = _DEFAULT_SEGMENT_LENGTH
    min_calls_per_operator: int = _DEFAULT_MIN_CALLS
    weight_reaction: float = _DEFAULT_WEIGHT_REACTION
    weight_floor: float = _DEFAULT_WEIGHT_FLOOR
    weight_smoothing: float = _DEFAULT_WEIGHT_SMOOTHING

    # Fixed-weight ablation mode.
    fixed_weights: bool = False
    fixed_weight_value: float = _DEFAULT_FIXED_WEIGHT

    # Differentiated reward tiers.
    reward_rejected: float = _DEFAULT_REWARD_REJECTED
    reward_accepted_worse: float = _DEFAULT_REWARD_ACCEPTED_WORSE
    reward_accepted_equal: float = _DEFAULT_REWARD_ACCEPTED_EQUAL
    reward_distance_improvement: float = _DEFAULT_REWARD_DISTANCE
    reward_vehicle_reduction: float = _DEFAULT_REWARD_VEHICLE
    reward_new_global_best: float = _DEFAULT_REWARD_BEST_DISTANCE
    reward_new_global_best_vehicle_reduction: float = _DEFAULT_REWARD_BEST_VEHICLE

    # Auto SA temperature estimation.
    auto_temperature: bool = True
    temperature_target_acceptance_rate: float = _DEFAULT_TEMP_TARGET
    temperature_sample_size: int = _DEFAULT_TEMP_SAMPLE
    temperature_fallback_fraction: float = _DEFAULT_TEMP_FALLBACK

    # Reheating.
    reheat_enabled: bool = True
    reheat_stagnation_threshold: int = _DEFAULT_REHEAT_THRESHOLD
    reheat_factor: float = _DEFAULT_REHEAT_FACTOR
    max_reheats: int = _DEFAULT_MAX_REHEATS

    # Stagnation restart.
    restart_enabled: bool = True
    restart_stagnation_threshold: int = _DEFAULT_RESTART_THRESHOLD
    max_restarts: int = _DEFAULT_MAX_RESTARTS

    # Incumbent intensification.
    intensification_enabled: bool = True
    intensification_iterations: int = _DEFAULT_INTENSIFICATION_ITERS
    intensification_removal_fraction: float = _DEFAULT_INTENSIFICATION_FRACTION

    # Acceptance-rate tracking window for diagnostics.
    acceptance_rate_window: int = 50

    def __post_init__(self) -> None:
        if self.segment_length < 1:
            raise ValueError("segment_length must be positive")
        if self.min_calls_per_operator < 1:
            raise ValueError("min_calls_per_operator must be positive")
        if not 0.0 < self.weight_reaction <= 1.0:
            raise ValueError("weight_reaction must be in (0, 1]")
        if not 0.0 < self.weight_floor < 1.0:
            raise ValueError("weight_floor must be in (0, 1)")
        if not 0.0 <= self.weight_smoothing <= 1.0:
            raise ValueError("weight_smoothing must be in [0, 1]")
        if self.fixed_weight_value <= 0.0:
            raise ValueError("fixed_weight_value must be positive")
        for name, val in (
            ("reward_rejected", self.reward_rejected),
            ("reward_accepted_worse", self.reward_accepted_worse),
            ("reward_accepted_equal", self.reward_accepted_equal),
            ("reward_distance_improvement", self.reward_distance_improvement),
            ("reward_vehicle_reduction", self.reward_vehicle_reduction),
            ("reward_new_global_best", self.reward_new_global_best),
            (
                "reward_new_global_best_vehicle_reduction",
                self.reward_new_global_best_vehicle_reduction,
            ),
        ):
            if not _REWARD_FLOOR <= val <= _REWARD_CEIL:
                raise ValueError(f"{name} must be in [{_REWARD_FLOOR}, {_REWARD_CEIL}]")
        if self.reward_vehicle_reduction <= self.reward_distance_improvement:
            raise ValueError(
                "reward_vehicle_reduction must exceed reward_distance_improvement"
            )
        if self.reward_new_global_best_vehicle_reduction <= self.reward_new_global_best:
            raise ValueError(
                "reward_new_global_best_vehicle_reduction must exceed "
                "reward_new_global_best"
            )
        if not 0.0 < self.temperature_target_acceptance_rate < 1.0:
            raise ValueError(
                "temperature_target_acceptance_rate must be in (0, 1)"
            )
        if self.temperature_sample_size < 5:
            raise ValueError("temperature_sample_size must be at least 5")
        if self.temperature_fallback_fraction <= 0.0:
            raise ValueError("temperature_fallback_fraction must be positive")
        if self.reheat_stagnation_threshold < 1:
            raise ValueError("reheat_stagnation_threshold must be positive")
        if not 0.0 < self.reheat_factor <= 2.0:
            raise ValueError("reheat_factor must be in (0, 2]")
        if self.max_reheats < 0:
            raise ValueError("max_reheats must be non-negative")
        if self.restart_stagnation_threshold <= self.reheat_stagnation_threshold:
            raise ValueError(
                "restart_stagnation_threshold must exceed "
                "reheat_stagnation_threshold"
            )
        if self.max_restarts < 0:
            raise ValueError("max_restarts must be non-negative")
        if self.intensification_iterations < 1:
            raise ValueError("intensification_iterations must be positive")
        if not 0.0 < self.intensification_removal_fraction <= 0.5:
            raise ValueError(
                "intensification_removal_fraction must be in (0, 0.5]"
            )
        if self.acceptance_rate_window < 10:
            raise ValueError("acceptance_rate_window must be at least 10")

    def reward_for(
        self,
        *,
        accepted: bool,
        comparison: str,
        is_global_best: bool,
        vehicle_reduction: bool,
    ) -> float:
        """Compute the differentiated reward for a single iteration.

        Parameters
        ----------
        accepted
            Whether the candidate was accepted by the SA criterion.
        comparison
            One of ``"better"``, ``"equal"``, ``"worse"`` (result of
            ``compare_objectives``).
        is_global_best
            Whether this candidate became a new global best.
        vehicle_reduction
            Whether this candidate reduced the vehicle count.
        """
        if not accepted:
            return self.reward_rejected
        if is_global_best:
            if vehicle_reduction:
                return self.reward_new_global_best_vehicle_reduction
            return self.reward_new_global_best
        if comparison == "better":
            if vehicle_reduction:
                return self.reward_vehicle_reduction
            return self.reward_distance_improvement
        if comparison == "equal":
            return self.reward_accepted_equal
        return self.reward_accepted_worse

    def apply_segment_update(
        self,
        weight: float,
        segment_reward_sum: float,
        segment_calls: int,
    ) -> float:
        """Apply the segment-based weight update.

        The update uses exponential moving average with the configured
        reaction factor, but the reward is the *average* reward per call in
        the segment, not a single call's reward.  A smoothing factor blends
        the old and new weight to reduce oscillation.
        """
        if segment_calls == 0:
            return max(self.weight_floor, weight)
        avg_reward = segment_reward_sum / segment_calls
        new_weight = max(
            self.weight_floor,
            (1.0 - self.weight_reaction) * weight
            + self.weight_reaction * avg_reward,
        )
        return self.weight_smoothing * weight + (1.0 - self.weight_smoothing) * new_weight


def with_fixed_weights(config: Stage04Config | None) -> Stage04Config | None:
    """Return a copy of *config* with fixed-weights mode enabled.

    If *config* is ``None``, returns ``None`` (no Stage 4 at all).
    """
    if config is None:
        return None
    return replace(
        config,
        fixed_weights=True,
        auto_temperature=False,  # keep the legacy heuristic temperature
        reheat_enabled=False,
        restart_enabled=False,
        intensification_enabled=False,
    )
