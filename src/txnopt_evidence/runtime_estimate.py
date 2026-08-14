"""Conservative Level 1 cloud-window estimator from local p95 observations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from txnopt_evidence.level1_protocol import validate_level1_protocol_v2

_AXIS_WORKERS = {"serial_1": 1, "txnopt_1": 1, "txnopt_4": 4, "barrier_4": 4}


def estimate_cloud_window(
    protocol: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    physical_cores: int,
    scheduler_efficiency: float,
) -> dict[str, object]:
    if protocol.get("schema_version") not in {
        "txnopt-level1-protocol-v1",
        "txnopt-level1-protocol-v2",
    }:
        raise ValueError("unsupported Level 1 protocol schema")
    if protocol.get("schema_version") == "txnopt-level1-protocol-v2":
        validate_level1_protocol_v2(protocol)
    if calibration.get("schema_version") not in {
        "txnopt-local-runtime-calibration-v1",
        "txnopt-local-runtime-calibration-v2",
    }:
        raise ValueError("unsupported runtime calibration schema")
    if isinstance(physical_cores, bool) or physical_cores <= 0:
        raise ValueError("physical_cores must be positive")
    if not 0.0 < scheduler_efficiency <= 1.0:
        raise ValueError("scheduler_efficiency must be in (0, 1]")
    resource_contract = protocol.get("resource_contract")
    minimum_cores = (
        resource_contract.get("minimum_physical_cores")
        if isinstance(resource_contract, dict)
        else protocol.get("minimum_physical_cores")
    )
    if not isinstance(minimum_cores, int) or physical_cores < minimum_cores:
        raise ValueError("physical core count is below the protocol minimum")
    axes = protocol.get("formal_axes")
    budgets = protocol.get("budgets")
    seeds = protocol.get("seeds")
    domains = protocol.get("domains")
    if axes != list(_AXIS_WORKERS) or budgets != ["fixed_work", "fixed_time"]:
        raise ValueError("protocol axes or budgets differ from Level 1")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("protocol seeds are invalid")
    if not isinstance(domains, dict):
        raise ValueError("protocol domains are invalid")
    observations = calibration.get("observations")
    if not isinstance(observations, list):
        raise ValueError("calibration observations must be a list")
    p95: dict[tuple[str, str, str], float] = {}
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("calibration observation must be an object")
        domain = observation.get("domain")
        axis = observation.get("axis")
        budget = observation.get("budget")
        seconds = observation.get("seconds_per_run_p95")
        if (
            not isinstance(domain, str)
            or not isinstance(axis, str)
            or not isinstance(budget, str)
            or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or float(seconds) <= 0.0
        ):
            raise ValueError("calibration observation is invalid")
        key = (domain, axis, budget)
        if key in p95:
            raise ValueError("calibration observation is duplicated")
        p95[key] = float(seconds)

    usable_cores = max(1, math.floor(physical_cores * scheduler_efficiency))
    groups: list[dict[str, object]] = []
    total_seconds = 0.0
    expected_keys: set[tuple[str, str, str]] = set()
    for domain in ("evrptw", "rcpsp"):
        scope = domains.get(domain)
        if not isinstance(scope, dict):
            raise ValueError(f"{domain} scope is invalid")
        pilot = scope.get("pilot")
        validation = scope.get("validation")
        if not isinstance(pilot, list) or not isinstance(validation, list):
            raise ValueError(f"{domain} scope lists are invalid")
        run_count = (len(pilot) + len(validation)) * len(seeds)
        for axis, workers in _AXIS_WORKERS.items():
            slots = max(1, usable_cores // workers)
            batches = math.ceil(run_count / slots)
            for budget in ("fixed_work", "fixed_time"):
                key = (domain, axis, budget)
                expected_keys.add(key)
                if key not in p95:
                    raise ValueError(f"missing calibration observation: {key}")
                group_seconds = batches * p95[key]
                total_seconds += group_seconds
                groups.append(
                    {
                        "domain": domain,
                        "axis": axis,
                        "budget": budget,
                        "workers_per_run": workers,
                        "parallel_slots": slots,
                        "run_count": run_count,
                        "batch_count": batches,
                        "seconds_per_run_p95": p95[key],
                        "predicted_group_seconds": group_seconds,
                    }
                )
    if set(p95) != expected_keys:
        raise ValueError("calibration includes an unregistered observation")
    predicted_days = total_seconds / 86_400.0
    maximum_days = protocol.get("predicted_formal_matrix_days_max")
    cloud_days = protocol.get("cloud_window_days")
    if not isinstance(maximum_days, int) or not isinstance(cloud_days, int):
        raise ValueError("protocol cloud envelope is invalid")
    purchase_gate = predicted_days <= maximum_days
    return {
        "schema_version": "txnopt-cloud-window-estimate-v1",
        "status": "PURCHASE_GATE_PASS" if purchase_gate else "PURCHASE_GATE_FAIL",
        "physical_cores": physical_cores,
        "scheduler_efficiency": scheduler_efficiency,
        "usable_cores": usable_cores,
        "predicted_matrix_seconds": total_seconds,
        "predicted_matrix_days": predicted_days,
        "maximum_matrix_days": maximum_days,
        "cloud_window_days": cloud_days,
        "reserved_rerun_days": cloud_days - maximum_days,
        "groups": groups,
        "cloud_purchase_performed": False,
    }


__all__ = ["estimate_cloud_window"]
