"""Strict JSON composition boundary for the two Level 1 case adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from txnopt import RunConfig, RunResult
from txnopt.runtime import PythonTxnRuntime
from txnopt_cases.evrptw import (
    EVRPTWOracle,
    EVRPTWPlan,
    EVRPTWSearchKernel,
    EVRPTWSolution,
    Instance,
    Node,
    NodeType,
    SolutionObjective,
    Vehicle,
)
from txnopt_cases.rcpsp import (
    Activity,
    ActivityMode,
    RCPSPInstance,
    RCPSPOracle,
    RCPSPSchedule,
    RCPSPSearchKernel,
    RCPSPState,
    ScheduledActivity,
    validate_schedule,
)

SemanticSink = Callable[[tuple[Mapping[str, object], ...]], None]


def parse_run_config(payload: Mapping[str, Any]) -> RunConfig:
    return RunConfig(
        seed=_integer(payload, "seed"),
        workers=_integer(payload, "workers"),
        execution_mode=_string(payload, "execution_mode"),  # type: ignore[arg-type]
        fixed_work=_optional_integer(payload, "fixed_work"),
        deadline_seconds=_optional_number(payload, "deadline_seconds"),
        speculation_window=_integer(payload, "speculation_window", default=0),
        trace_policy=_string(payload, "trace_policy", default="semantic"),  # type: ignore[arg-type]
        max_rounds=_integer(payload, "max_rounds", default=1000),
    )


def execute_case(
    case: Mapping[str, Any],
    *,
    config: RunConfig,
    semantic_sink: SemanticSink,
    physical_sink: SemanticSink,
) -> dict[str, Any]:
    domain = _string(case, "domain")
    if domain == "evrptw":
        return _execute_evrptw(
            case,
            config=config,
            semantic_sink=semantic_sink,
            physical_sink=physical_sink,
        )
    if domain == "rcpsp":
        return _execute_rcpsp(
            case,
            config=config,
            semantic_sink=semantic_sink,
            physical_sink=physical_sink,
        )
    raise ValueError(f"unsupported TxnOpt case domain: {domain}")


def review_case(case: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    domain = _string(case, "domain")
    state_payload = _mapping(result, "last_committed_state")
    if domain == "evrptw":
        evrptw_instance = _evrptw_instance(_mapping(case, "instance"))
        evrptw_oracle = EVRPTWOracle(evrptw_instance)
        evrptw_state = _evrptw_solution(state_payload)
        evrptw_oracle.validate(evrptw_state)
        objective: object = list(evrptw_oracle.objective(evrptw_state).key)
        state_digest = evrptw_oracle.state_digest(evrptw_state)
    elif domain == "rcpsp":
        rcpsp_instance = _rcpsp_instance(_mapping(case, "instance"))
        rcpsp_oracle = RCPSPOracle(
            rcpsp_instance,
            seed=_integer(case, "oracle_seed", default=0),
        )
        rcpsp_state = _rcpsp_schedule(state_payload)
        report = validate_schedule(rcpsp_instance, rcpsp_state)
        if not report.feasible:
            raise ValueError(f"RCPSP independent replay failed: {report.violations}")
        objective = rcpsp_oracle.objective(rcpsp_state)
        state_digest = rcpsp_oracle.state_digest(rcpsp_state)
    else:
        raise ValueError(f"unsupported TxnOpt case domain: {domain}")
    if result.get("objective") != objective:
        raise ValueError("raw objective differs from independent reconstruction")
    if result.get("state_digest") != state_digest:
        raise ValueError("raw state digest differs from independent reconstruction")
    return {
        "domain": domain,
        "objective": objective,
        "state_digest": state_digest,
        "validator_status": "PASS",
    }


def _execute_evrptw(
    case: Mapping[str, Any],
    *,
    config: RunConfig,
    semantic_sink: SemanticSink,
    physical_sink: SemanticSink,
) -> dict[str, Any]:
    instance = _evrptw_instance(_mapping(case, "instance"))
    backend = _string(case, "backend", default="python")
    if backend == "python":
        oracle: EVRPTWOracle = EVRPTWOracle(instance)
    elif backend == "native":
        from txnopt_cases.evrptw.native_oracle import NativeEVRPTWOracle

        oracle = NativeEVRPTWOracle(instance, worker_count=config.workers)
    else:
        raise ValueError("EVRPTW backend must be python or native")
    initial_plan = EVRPTWPlan(
        tuple(
            tuple(_string_value(item) for item in _as_sequence(route, "EVRPTW initial route"))
            for route in _sequence(case, "initial_plan")
        )
    )
    initial = oracle.solve_initial(initial_plan)
    runtime = PythonTxnRuntime[EVRPTWSolution, EVRPTWPlan, SolutionObjective](
        semantic_event_sink=semantic_sink,
        physical_event_sink=physical_sink,
    )
    result = runtime.run(
        initial,
        kernel=EVRPTWSearchKernel(max_candidates=_integer(case, "max_candidates", default=64)),
        oracle=oracle,
        config=config,
    )
    return _evrptw_result(result, oracle)


def _execute_rcpsp(
    case: Mapping[str, Any],
    *,
    config: RunConfig,
    semantic_sink: SemanticSink,
    physical_sink: SemanticSink,
) -> dict[str, Any]:
    instance = _rcpsp_instance(_mapping(case, "instance"))
    oracle = RCPSPOracle(instance, seed=_integer(case, "oracle_seed", default=config.seed))
    initial_state = _rcpsp_state(_mapping(case, "initial_state"))
    initial = oracle.evaluate_batch((initial_state,), work_budget=1, deadline_ns=None)[0]
    runtime = PythonTxnRuntime[RCPSPSchedule, RCPSPState, int](
        semantic_event_sink=semantic_sink,
        physical_event_sink=physical_sink,
    )
    result = runtime.run(
        initial,
        kernel=RCPSPSearchKernel(
            instance,
            max_block_size=_integer(case, "max_block_size", default=3),
        ),
        oracle=oracle,
        config=config,
    )
    return _rcpsp_result(result, oracle)


def _evrptw_instance(payload: Mapping[str, Any]) -> Instance:
    nodes = tuple(
        Node(
            name=_string(node, "name"),
            kind=NodeType(_string(node, "kind")),
            x=_number(node, "x"),
            y=_number(node, "y"),
            demand=_number(node, "demand"),
            ready_time=_number(node, "ready_time"),
            due_date=_number(node, "due_date"),
            service_time=_number(node, "service_time"),
        )
        for node in (_as_mapping(item, "EVRPTW node") for item in _sequence(payload, "nodes"))
    )
    vehicle_payload = _mapping(payload, "vehicle")
    return Instance(
        name=_string(payload, "name"),
        nodes=nodes,
        vehicle=Vehicle(
            battery_capacity=_number(vehicle_payload, "battery_capacity"),
            load_capacity=_number(vehicle_payload, "load_capacity"),
            consumption_rate=_number(vehicle_payload, "consumption_rate"),
            inverse_refueling_rate=_number(vehicle_payload, "inverse_refueling_rate"),
            average_velocity=_number(vehicle_payload, "average_velocity"),
        ),
        distance_backend="python",
    )


def _evrptw_solution(payload: Mapping[str, Any]) -> EVRPTWSolution:
    plan = EVRPTWPlan(
        tuple(
            tuple(_string_value(item) for item in _as_sequence(route, "EVRPTW customer route"))
            for route in _sequence(payload, "customer_routes")
        )
    )
    objective_payload = payload.get("objective")
    if objective_payload is None:
        objective_value = None
    else:
        objective_fields = _as_sequence(objective_payload, "objective")
        if len(objective_fields) != 4:
            raise ValueError("EVRPTW objective must contain four fields")
        objective_value = SolutionObjective(
            vehicle_count=_integer_value(objective_fields[0]),
            total_distance=_number_value(objective_fields[1]),
            total_charging_time=_number_value(objective_fields[2]),
            charging_count=_integer_value(objective_fields[3]),
        )
    return EVRPTWSolution(
        plan=plan,
        routes=tuple(
            tuple(_string_value(item) for item in _as_sequence(route, "EVRPTW exact route"))
            for route in _sequence(payload, "routes")
        ),
        feasible=_boolean(payload, "feasible"),
        objective_value=objective_value,
        failure_reasons=tuple(
            _string_value(item) for item in _sequence(payload, "failure_reasons")
        ),
    )


def _evrptw_result(
    result: RunResult[EVRPTWSolution, SolutionObjective],
    oracle: EVRPTWOracle,
) -> dict[str, Any]:
    state = result.last_committed_state
    return {
        "schema_version": "txnopt-run-result-v1",
        "domain": "evrptw",
        "last_committed_state": {
            "customer_routes": state.plan.customer_routes,
            "routes": state.routes,
            "feasible": state.feasible,
            "objective": None if state.objective_value is None else state.objective_value.key,
            "failure_reasons": state.failure_reasons,
        },
        "objective": list(result.objective.key),
        "state_digest": oracle.state_digest(state),
        "termination_reason": result.termination_reason,
        "semantic_digest": result.semantic_digest,
        "physical_artifact_ref": result.physical_artifact_ref,
        "provenance": dict(result.provenance),
        "fallback_count": 0,
    }


def _rcpsp_instance(payload: Mapping[str, Any]) -> RCPSPInstance:
    activities = tuple(
        Activity(
            activity_id=_integer(activity, "activity_id"),
            predecessors=tuple(
                _integer_value(item) for item in _sequence(activity, "predecessors")
            ),
            modes=tuple(
                ActivityMode(
                    duration=_integer(mode, "duration"),
                    renewable_demands=tuple(
                        _integer_value(item) for item in _sequence(mode, "renewable_demands")
                    ),
                )
                for mode in (
                    _as_mapping(item, "RCPSP mode") for item in _sequence(activity, "modes")
                )
            ),
        )
        for activity in (
            _as_mapping(item, "RCPSP activity") for item in _sequence(payload, "activities")
        )
    )
    return RCPSPInstance(
        name=_string(payload, "name"),
        activities=activities,
        renewable_capacities=tuple(
            _integer_value(item) for item in _sequence(payload, "renewable_capacities")
        ),
    )


def _rcpsp_state(payload: Mapping[str, Any]) -> RCPSPState:
    return RCPSPState(
        activity_order=tuple(_integer_value(item) for item in _sequence(payload, "activity_order")),
        mode_vector=tuple(_integer_value(item) for item in _sequence(payload, "mode_vector")),
    )


def _rcpsp_schedule(payload: Mapping[str, Any]) -> RCPSPSchedule:
    state = _rcpsp_state(_mapping(payload, "state"))
    return RCPSPSchedule(
        state=state,
        activities=tuple(
            ScheduledActivity(
                activity_id=_integer(item, "activity_id"),
                mode_index=_integer(item, "mode_index"),
                start=_integer(item, "start"),
                end=_integer(item, "end"),
            )
            for item in (
                _as_mapping(value, "scheduled activity")
                for value in _sequence(payload, "activities")
            )
        ),
        makespan=_integer(payload, "makespan"),
    )


def _rcpsp_result(
    result: RunResult[RCPSPSchedule, int],
    oracle: RCPSPOracle,
) -> dict[str, Any]:
    state = result.last_committed_state
    return {
        "schema_version": "txnopt-run-result-v1",
        "domain": "rcpsp",
        "last_committed_state": {
            "state": {
                "activity_order": state.state.activity_order,
                "mode_vector": state.state.mode_vector,
            },
            "activities": [
                {
                    "activity_id": item.activity_id,
                    "mode_index": item.mode_index,
                    "start": item.start,
                    "end": item.end,
                }
                for item in state.activities
            ],
            "makespan": state.makespan,
        },
        "objective": result.objective,
        "state_digest": oracle.state_digest(state),
        "termination_reason": result.termination_reason,
        "semantic_digest": result.semantic_digest,
        "physical_artifact_ref": result.physical_artifact_ref,
        "provenance": dict(result.provenance),
        "fallback_count": 0,
    }


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _as_mapping(payload.get(key), key)


def _as_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def _sequence(payload: Mapping[str, Any], key: str) -> Sequence[Any]:
    return _as_sequence(payload.get(key), key)


def _as_sequence(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _string(payload: Mapping[str, Any], key: str, *, default: str | None = None) -> str:
    value = payload.get(key, default)
    return _string_value(value)


def _string_value(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("expected a non-empty string")
    return value


def _integer(payload: Mapping[str, Any], key: str, *, default: int | None = None) -> int:
    return _integer_value(payload.get(key, default))


def _integer_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected an integer")
    return value


def _optional_integer(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    return None if value is None else _integer_value(value)


def _number(payload: Mapping[str, Any], key: str) -> float:
    return _number_value(payload.get(key))


def _number_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a number")
    return float(value)


def _optional_number(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return None if value is None else _number_value(value)


def _boolean(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


__all__ = ["execute_case", "parse_run_config", "review_case"]
