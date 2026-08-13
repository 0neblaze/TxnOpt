from __future__ import annotations

from pathlib import Path

import pytest

from txnopt._internal.candidate_txn import TxnPhase
from txnopt._internal.versions import (
    CONTRACT_VERSION,
    NATIVE_ROUND_VERSION,
    PHYSICAL_TRACE_VERSION,
    SEMANTIC_TRACE_VERSION,
)
from txnopt._internal.waste_bounds import bounded_waste

ROOT = Path(__file__).resolve().parents[2]


def test_formal_model_and_python_state_machine_name_the_same_phases() -> None:
    model = (ROOT / "formal/TxnOpt.tla").read_text(encoding="utf-8")
    for phase in TxnPhase:
        assert f'"{phase.value}"' in model


def test_first_generation_protocol_identities_are_exact() -> None:
    assert (
        CONTRACT_VERSION,
        NATIVE_ROUND_VERSION,
        SEMANTIC_TRACE_VERSION,
        PHYSICAL_TRACE_VERSION,
    ) == (
        "txnopt-contract-v1",
        "txnopt-native-round-v1",
        "txnopt-semantic-trace-v1",
        "txnopt-physical-trace-v1",
    )


def test_t4_executable_bound_matches_declared_formula() -> None:
    result = bounded_waste(
        remaining_budget=100,
        speculation_window=3,
        max_requests_per_candidate=7,
        max_request_cost=2.5,
        parallelism=4,
    )
    assert result.discarded_work_units == min(100, 3 * 7)
    assert result.discarded_cost == min(100, 3 * 7) * 2.5
    assert result.post_boundary_work_units == min(4, 3 * 7)
    assert result.post_boundary_cost == min(4, 3 * 7) * 2.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "remaining_budget": -1,
            "speculation_window": 1,
            "max_requests_per_candidate": 1,
            "max_request_cost": 1.0,
            "parallelism": 1,
        },
        {
            "remaining_budget": 1,
            "speculation_window": 1,
            "max_requests_per_candidate": 1,
            "max_request_cost": float("inf"),
            "parallelism": 1,
        },
        {
            "remaining_budget": 1,
            "speculation_window": 1,
            "max_requests_per_candidate": 1,
            "max_request_cost": 1.0,
            "parallelism": 0,
        },
    ],
)
def test_t4_bound_rejects_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        bounded_waste(**kwargs)  # type: ignore[arg-type]
