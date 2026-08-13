"""Version-stable SplitMix64 random tape used by the Python specification."""

from __future__ import annotations

_MASK64 = (1 << 64) - 1
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15


class RandomTape:
    """Deterministic word stream whose algorithm is part of contract v1."""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = seed & _MASK64

    def take(self, count: int) -> tuple[int, ...]:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("random tape count must be a positive integer")
        return tuple(self._next() for _ in range(count))

    def _next(self) -> int:
        self._state = (self._state + _GOLDEN_GAMMA) & _MASK64
        value = self._state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
        return (value ^ (value >> 31)) & _MASK64
