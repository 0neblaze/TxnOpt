"""Closed Level 1 protocol contracts shared by planning and review."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PROTOCOL_V2_SCHEMA = "txnopt-level1-protocol-v2"
CANONICAL_AXES = ("serial_1", "txnopt_1", "txnopt_4", "barrier_4")
CANONICAL_BUDGETS = ("fixed_work", "fixed_time")
CANONICAL_SEEDS = tuple(range(2014, 2024))
PROTOCOL_V2_RESOURCE_CONTRACT: dict[str, object] = {
    "minimum_physical_cores": 64,
    "minimum_provider_memory_gb": 128,
    "core_count": 64,
    "thread_per_core": 1,
    "region": None,
    "region_required_live_input": True,
    "exclusive_linux_required": True,
    "maximum_consecutive_window_days": 14,
    "predicted_completion_days_max": 10,
    "linux_visible_memory_is_admission_gate": False,
    "attempt27_peak_rss_fraction_max": 0.8,
}


def validate_level1_protocol_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete Build16/Attempt26 protocol without permissive defaults."""

    if payload.get("schema_version") != PROTOCOL_V2_SCHEMA:
        raise ValueError("Level 1 protocol v2 schema differs")
    if (
        payload.get("status") != "preregistered_tencent_cloud_scope_not_authorized"
        or payload.get("holdout_opened") is not False
        or payload.get("level2_holdout_opened") is not False
        or payload.get("level3_holdout_opened") is not False
        or payload.get("cloud_purchase_authorized") is not False
        or payload.get("formal_matrix_started") is not False
    ):
        raise ValueError("Level 1 protocol v2 boundary differs")
    if (
        tuple(_strings(payload.get("formal_axes"), "formal axes")) != CANONICAL_AXES
        or tuple(_strings(payload.get("budgets"), "budget axes")) != CANONICAL_BUDGETS
        or tuple(_integers(payload.get("seeds"), "seeds")) != CANONICAL_SEEDS
    ):
        raise ValueError("Level 1 protocol v2 matrix axes differ")
    domains = payload.get("domains")
    if not isinstance(domains, dict) or set(domains) != {"evrptw", "rcpsp"}:
        raise ValueError("Level 1 protocol v2 domains differ")
    expected_counts = {"evrptw": 12, "rcpsp": 24}
    for domain, count in expected_counts.items():
        scope = domains.get(domain)
        if not isinstance(scope, dict) or set(scope) != {"pilot", "validation"}:
            raise ValueError(f"Level 1 protocol v2 {domain} scope differs")
        identifiers = (
            *_strings(scope.get("pilot"), f"{domain} pilot"),
            *_strings(scope.get("validation"), f"{domain} validation"),
        )
        if len(identifiers) != count or len(set(identifiers)) != count:
            raise ValueError(f"Level 1 protocol v2 {domain} identity count differs")
    if (
        payload.get("fixed_work") != 1200
        or payload.get("fixed_time_seconds") != 3
        or payload.get("max_rounds") != 10
        or payload.get("max_candidates") != {"evrptw": 64, "rcpsp": 64}
        or payload.get("cloud_window_days") != 14
        or payload.get("predicted_formal_matrix_days_max") != 10
        or payload.get("rerun_margin_days") != 4
    ):
        raise ValueError("Level 1 protocol v2 execution envelope differs")
    resource = payload.get("resource_contract")
    if resource != PROTOCOL_V2_RESOURCE_CONTRACT:
        raise ValueError("Level 1 protocol v2 resource contract differs")
    return dict(PROTOCOL_V2_RESOURCE_CONTRACT)


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Level 1 protocol v2 {name} must be strings")
    return tuple(value)


def _integers(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"Level 1 protocol v2 {name} must be integers")
    return tuple(value)


__all__ = [
    "CANONICAL_AXES",
    "CANONICAL_BUDGETS",
    "CANONICAL_SEEDS",
    "PROTOCOL_V2_RESOURCE_CONTRACT",
    "PROTOCOL_V2_SCHEMA",
    "validate_level1_protocol_v2",
]
