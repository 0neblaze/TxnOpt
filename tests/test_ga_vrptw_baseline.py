from __future__ import annotations

from evrptw.baselines.ga_vrptw import decode_chromosome, solve_ga_vrptw
from evrptw.experiments.week02_baseline_comparison import generate_week02_instance


def test_ga_baseline_is_reproducible_with_same_seed() -> None:
    instance = generate_week02_instance(12, seed=2014)

    first = solve_ga_vrptw(instance, seed=7, population_size=12, generations=4)
    second = solve_ga_vrptw(instance, seed=7, population_size=12, generations=4)

    assert first.routes == second.routes
    assert first.objective_value == second.objective_value
    assert first.feasible == second.feasible


def test_decode_visits_each_customer_once() -> None:
    instance = generate_week02_instance(10, seed=2014)
    chromosome = tuple(customer.name for customer in instance.customers)

    routes = decode_chromosome(instance, chromosome)
    visited = [node for route in routes for node in route if node.startswith("C")]

    visited_ids = sorted(int(node.removeprefix("C")) for node in visited)
    assert visited_ids == list(range(1, 11))


def test_ga_convergence_log_contains_required_fields() -> None:
    instance = generate_week02_instance(10, seed=2014)
    result = solve_ga_vrptw(instance, seed=2014, population_size=10, generations=3)
    payload = result.to_dict()
    objective_value = payload["objective_value"]
    runtime_seconds = payload["runtime_seconds"]

    assert isinstance(objective_value, (int, float))
    assert isinstance(runtime_seconds, (int, float))
    assert objective_value > 0
    assert runtime_seconds >= 0
    assert len(result.convergence) == 4
    assert result.convergence[-1].generation == 3
    assert result.convergence[-1].best_objective > 0
    assert result.convergence[-1].average_objective > 0
    assert 0 <= result.convergence[-1].feasible_rate <= 1


def test_ga_rejects_invalid_wall_clock_limit() -> None:
    instance = generate_week02_instance(10, seed=2014)

    try:
        solve_ga_vrptw(instance, time_limit_seconds=0.0)
    except ValueError as error:
        assert "time_limit_seconds" in str(error)
    else:
        raise AssertionError("expected an invalid time limit to fail fast")
