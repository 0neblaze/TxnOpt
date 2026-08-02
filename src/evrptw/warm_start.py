"""Explicit exact-verification protocol for cross-architecture warm starts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

WARM_START_VALIDATION_SCHEMA_VERSION = "stage05.2-warm-start-validation-v1"


@dataclass(frozen=True, slots=True)
class WarmStartValidationConfig:
    """Permit a supplied incumbent without enabling Candidate Control."""

    enabled: bool = True
    schema_version: str = WARM_START_VALIDATION_SCHEMA_VERSION
    verification_policy: str = "unified_exact_pipeline"
    failure_policy: str = "fail_fast_no_fallback"
    fallback_allowed: bool = False

    def __post_init__(self) -> None:
        if not self.enabled:
            raise ValueError("disabled warm-start validation is ambiguous; pass None")
        if self.schema_version != WARM_START_VALIDATION_SCHEMA_VERSION:
            raise ValueError("unsupported warm-start validation schema")
        if self.verification_policy != "unified_exact_pipeline":
            raise ValueError("warm starts require the unified exact pipeline")
        if self.failure_policy != "fail_fast_no_fallback" or self.fallback_allowed:
            raise ValueError("warm-start verification failures must not fall back")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = (
    "WARM_START_VALIDATION_SCHEMA_VERSION",
    "WarmStartValidationConfig",
)
