from __future__ import annotations

import pytest

from txnopt_evidence.runtime_estimate import estimate_cloud_window


def _protocol() -> dict[str, object]:
    return {
        "schema_version": "txnopt-level1-protocol-v1",
        "minimum_physical_cores": 32,
        "predicted_formal_matrix_days_max": 10,
        "cloud_window_days": 14,
        "formal_axes": ["serial_1", "txnopt_1", "txnopt_4", "barrier_4"],
        "budgets": ["fixed_work", "fixed_time"],
        "seeds": [2014, 2015],
        "domains": {
            "evrptw": {"pilot": ["ev1"], "validation": []},
            "rcpsp": {"pilot": ["rc1"], "validation": []},
        },
    }


def _calibration() -> dict[str, object]:
    observations = []
    for domain in ("evrptw", "rcpsp"):
        for axis in ("serial_1", "txnopt_1", "txnopt_4", "barrier_4"):
            for budget in ("fixed_work", "fixed_time"):
                observations.append(
                    {
                        "domain": domain,
                        "axis": axis,
                        "budget": budget,
                        "seconds_per_run_p95": 10.0,
                    }
                )
    return {
        "schema_version": "txnopt-local-runtime-calibration-v1",
        "observations": observations,
    }


def test_cloud_estimator_uses_p95_batches_and_preserves_procurement_boundary() -> None:
    result = estimate_cloud_window(
        _protocol(),
        _calibration(),
        physical_cores=32,
        scheduler_efficiency=0.8,
    )

    assert result["status"] == "PURCHASE_GATE_PASS"
    assert result["usable_cores"] == 25
    assert result["predicted_matrix_seconds"] == 160.0
    assert result["reserved_rerun_days"] == 4
    assert result["cloud_purchase_performed"] is False


def test_cloud_estimator_rejects_an_incomplete_calibration() -> None:
    calibration = _calibration()
    observations = calibration["observations"]
    assert isinstance(observations, list)
    observations.pop()

    with pytest.raises(ValueError, match="missing calibration"):
        estimate_cloud_window(
            _protocol(),
            calibration,
            physical_cores=32,
            scheduler_efficiency=0.8,
        )
