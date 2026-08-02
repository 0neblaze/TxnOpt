"""Explicit exact-verification protocol for cross-architecture warm starts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

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


def canonical_customer_sequences_sha256(
    customer_sequences: tuple[tuple[str, ...], ...],
) -> str:
    """Hash ordered warm-start customer routes with one stable JSON encoding."""

    encoded = json.dumps(
        [list(route) for route in customer_sequences],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_warm_start_provenance(
    customer_sequences: tuple[tuple[str, ...], ...],
    provenance: Mapping[str, object],
) -> None:
    """Bind a supplied warm start to its source bytes and ordered route identity."""

    source_sha256 = provenance.get("source_solution_sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("warm-start source solution SHA-256 is invalid")
    source_path_value = provenance.get("source_solution_path")
    if not isinstance(source_path_value, str) or not source_path_value:
        raise ValueError("warm-start provenance requires a source solution path")
    source_path = Path(source_path_value)
    if not source_path.is_file():
        raise ValueError("warm-start source solution file is missing")
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("warm-start source solution SHA-256 mismatch")
    expected_sequences_sha256 = canonical_customer_sequences_sha256(customer_sequences)
    declared_sequences_sha256 = provenance.get("source_customer_sequences_sha256")
    if declared_sequences_sha256 != expected_sequences_sha256:
        raise ValueError("warm-start customer-sequence SHA-256 mismatch")


__all__ = (
    "WARM_START_VALIDATION_SCHEMA_VERSION",
    "WarmStartValidationConfig",
    "canonical_customer_sequences_sha256",
    "verify_warm_start_provenance",
)
