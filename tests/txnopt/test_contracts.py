from __future__ import annotations

from types import MappingProxyType

import pytest

import txnopt
from txnopt import RunConfig, RunResult
from txnopt._internal.candidate_txn import CandidateTxn, TxnPhase


def test_root_exports_exactly_five_contracts() -> None:
    assert txnopt.__all__ == [
        "TxnRuntime",
        "SearchKernel",
        "Oracle",
        "RunConfig",
        "RunResult",
    ]
    assert "CandidateTxn" not in vars(txnopt)


@pytest.mark.parametrize(
    "config",
    [
        RunConfig(seed=0, workers=1, execution_mode="serial", fixed_work=1),
        RunConfig(
            seed=1,
            workers=4,
            execution_mode="barrier",
            deadline_seconds=1.0,
        ),
        RunConfig(
            seed=2,
            workers=4,
            execution_mode="ordered",
            fixed_work=100,
            speculation_window=8,
            trace_policy="semantic_and_physical",
        ),
    ],
)
def test_run_config_accepts_each_canonical_execution_mode(config: RunConfig) -> None:
    assert config.seed >= 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": 0, "workers": 1, "execution_mode": "serial"},
        {
            "seed": 0,
            "workers": 1,
            "execution_mode": "serial",
            "fixed_work": 1,
            "deadline_seconds": 1.0,
        },
        {"seed": 0, "workers": 2, "execution_mode": "serial", "fixed_work": 1},
        {
            "seed": 0,
            "workers": 4,
            "execution_mode": "ordered",
            "fixed_work": 1,
            "speculation_window": 0,
        },
        {"seed": 0, "workers": 1, "execution_mode": "barrier", "fixed_work": 0},
        {
            "seed": 0,
            "workers": 1,
            "execution_mode": "barrier",
            "deadline_seconds": float("nan"),
        },
    ],
)
def test_run_config_rejects_ambiguous_or_invalid_runtime_policy(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        RunConfig(**kwargs)  # type: ignore[arg-type]


def test_run_result_freezes_provenance_and_requires_semantic_digest() -> None:
    source = {"contract": "txnopt-contract-v1", "source": "abc123"}
    result = RunResult(
        last_committed_state=(1, 2),
        objective=(3, 4),
        termination_reason="fixed_work_exhausted",
        semantic_digest="a" * 64,
        physical_artifact_ref=None,
        provenance=source,
    )
    source["source"] = "changed"
    assert result.provenance == {
        "contract": "txnopt-contract-v1",
        "source": "abc123",
    }
    assert isinstance(result.provenance, MappingProxyType)
    with pytest.raises(TypeError):
        result.provenance["new"] = "value"  # type: ignore[index]
    with pytest.raises(ValueError, match="semantic_digest"):
        RunResult(
            last_committed_state=(),
            objective=(),
            termination_reason="done",
            semantic_digest="not-a-digest",
            physical_artifact_ref=None,
            provenance={"contract": "txnopt-contract-v1"},
        )


def test_candidate_transaction_accepts_only_the_canonical_state_machine() -> None:
    txn = CandidateTxn(
        txn_id="round-0001",
        snapshot_digest="b" * 64,
        candidate_keys=("c0", "c1"),
    )
    for phase in (
        TxnPhase.RESERVED,
        TxnPhase.EVALUATING,
        TxnPhase.VALIDATED,
        TxnPhase.COMMITTED,
    ):
        txn = txn.transition(phase)
    assert txn.terminal is True
    with pytest.raises(ValueError, match="illegal"):
        txn.transition(TxnPhase.ABORTED)


def test_candidate_transaction_can_abort_or_interrupt_without_commit() -> None:
    prepared = CandidateTxn(
        txn_id="round-0002",
        snapshot_digest="c" * 64,
        candidate_keys=("candidate",),
    )
    assert prepared.transition(TxnPhase.ABORTED).terminal is True
    evaluating = prepared.transition(TxnPhase.RESERVED).transition(TxnPhase.EVALUATING)
    assert evaluating.transition(TxnPhase.INTERRUPTED).terminal is True


def test_candidate_transaction_rejects_duplicate_stable_keys() -> None:
    with pytest.raises(ValueError, match="deduplicated"):
        CandidateTxn(
            txn_id="round-0003",
            snapshot_digest="d" * 64,
            candidate_keys=("same", "same"),
        )
