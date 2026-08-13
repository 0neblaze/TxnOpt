from __future__ import annotations

from txnopt import RunConfig
from txnopt.runtime import PythonTxnRuntime
from txnopt_cases.evrptw import (
    EVRPTWOracle,
    EVRPTWPlan,
    EVRPTWSearchKernel,
    Instance,
    Node,
    NodeType,
    Vehicle,
)


def _instance(*, battery: float = 100.0) -> Instance:
    return Instance(
        "tiny-txnopt-evrptw",
        (
            Node("D", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 1.0, 0.0, 1.0, 0.0, 100.0, 0.0),
            Node("C2", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 100.0, 0.0),
        ),
        Vehicle(battery, 10.0, 1.0, 0.1, 1.0),
        distance_backend="python",
    )


def _config(mode: str) -> RunConfig:
    return RunConfig(
        seed=2014,
        workers=1 if mode == "serial" else 2,
        execution_mode=mode,  # type: ignore[arg-type]
        fixed_work=64,
        speculation_window=2 if mode == "ordered" else 0,
        max_rounds=1,
    )


def test_evrptw_oracle_reconstructs_the_authoritative_objective() -> None:
    oracle = EVRPTWOracle(_instance())
    state = oracle.solve_initial(EVRPTWPlan((("C1",), ("C2",))))

    oracle.validate(state)
    assert state.feasible is True
    assert oracle.objective(state).vehicle_count == 2
    assert len(oracle.state_digest(state)) == 64


def test_evrptw_kernel_reduces_vehicle_count_through_each_runtime_mode() -> None:
    oracle = EVRPTWOracle(_instance())
    initial = oracle.solve_initial(EVRPTWPlan((("C1",), ("C2",))))
    results = {
        mode: PythonTxnRuntime().run(
            initial,
            kernel=EVRPTWSearchKernel(),
            oracle=oracle,
            config=_config(mode),
        )
        for mode in ("serial", "barrier", "ordered")
    }

    assert {result.objective.vehicle_count for result in results.values()} == {1}
    assert {result.semantic_digest for result in results.values()} == {
        results["serial"].semantic_digest
    }


def test_evrptw_exact_infeasibility_is_a_valid_noncommittable_result() -> None:
    oracle = EVRPTWOracle(_instance(battery=1.5))
    plan = EVRPTWPlan((("C1", "C2"),))
    result = oracle.evaluate_batch((plan,), work_budget=1, deadline_ns=None)[0]

    oracle.validate(result)
    assert result.feasible is False
    assert result.routes == ()
    assert result.failure_reasons
