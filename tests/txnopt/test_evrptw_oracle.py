from __future__ import annotations

from txnopt import RunConfig
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
    construct_initial_plan,
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


def test_evrptw_safe_energy_infeasibility_is_filtered_before_exact_work() -> None:
    oracle = EVRPTWOracle(_instance(battery=1.5))
    plan = EVRPTWPlan((("C1", "C2"),))

    assert oracle.screen((plan,)) == (False,)


def test_evrptw_replay_accepts_the_canonical_objective_key() -> None:
    oracle = EVRPTWOracle(_instance())
    original = oracle.solve_initial(EVRPTWPlan((("C1", "C2"),)))
    assert original.objective_value is not None
    canonical = SolutionObjective(*original.objective_value.key)
    replayed = EVRPTWSolution(
        plan=original.plan,
        routes=original.routes,
        feasible=True,
        objective_value=canonical,
    )

    oracle.validate(replayed)
    assert oracle.objective(replayed).key == original.objective_value.key


def test_evrptw_initial_constructor_returns_an_exact_feasible_plan() -> None:
    instance = _instance()
    plan = construct_initial_plan(instance)
    state = EVRPTWOracle(instance).solve_initial(plan)

    assert state.feasible is True
    assert {name for route in plan.customer_routes for name in route} == {"C1", "C2"}


def test_evrptw_safe_screen_cache_reuses_unchanged_routes() -> None:
    oracle = EVRPTWOracle(_instance())
    plan = EVRPTWPlan((("C1",), ("C2",)))

    assert oracle.screen((plan,)) == (True,)
    after_first = dict(oracle.screening_statistics)
    assert oracle.screen((plan,)) == (True,)
    after_second = dict(oracle.screening_statistics)

    assert after_first == {
        "route_screen_cache_hits": 0,
        "route_screen_cache_misses": 2,
        "route_screen_cache_size": 2,
        "route_screen_cache_capacity": 65_536,
        "route_screen_cache_evictions": 0,
    }
    assert after_second["route_screen_cache_hits"] == 2
    assert after_second["route_screen_cache_misses"] == 2


def test_evrptw_safe_screen_cache_is_bounded_and_evicts_lru_routes() -> None:
    oracle = EVRPTWOracle(_instance(), route_screen_cache_capacity=2)

    assert oracle.screen((EVRPTWPlan((("C1",), ("C2",))),)) == (True,)
    assert oracle.screen((EVRPTWPlan((("C1", "C2"),)),)) == (True,)

    assert dict(oracle.screening_statistics) == {
        "route_screen_cache_hits": 0,
        "route_screen_cache_misses": 3,
        "route_screen_cache_size": 2,
        "route_screen_cache_capacity": 2,
        "route_screen_cache_evictions": 1,
    }
