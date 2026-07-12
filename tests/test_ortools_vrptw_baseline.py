from __future__ import annotations

import json
from pathlib import Path

from evrptw.baselines.ortools_vrptw import render_summary, solve_vrptw
from evrptw.parser import parse_schneider
from evrptw.validation import validate_routes

INSTANCE = Path("src/evrptw/instances/week01_vrptw_toy.txt")


def test_week01_toy_instance_parses() -> None:
    instance = parse_schneider(INSTANCE)

    assert instance.name == "week01_vrptw_toy"
    assert len(instance.customers) == 8
    assert not instance.stations


def test_ortools_baseline_returns_feasible_solution() -> None:
    instance = parse_schneider(INSTANCE)
    result = solve_vrptw(instance, vehicle_count=2, time_limit_seconds=5)

    assert result["solver_status"] in {"ROUTING_SUCCESS", "ROUTING_OPTIMAL"}
    assert result["solver_objective"] is not None
    assert result["feasible"] is True
    assert result["runtime_seconds"] >= 0
    assert len(result["routes"]) == 2

    visited_customers = [
        node
        for route in result["routes"]
        for node in route
        if node.startswith("C")
    ]
    assert sorted(visited_customers) == [f"C{index}" for index in range(1, 9)]

    validation = validate_routes(instance, result["routes"])
    assert validation.feasible
    assert validation.total_distance == result["validation_total_distance"]


def test_summary_and_json_payload_include_required_fields(tmp_path: Path) -> None:
    instance = parse_schneider(INSTANCE)
    result = solve_vrptw(instance, vehicle_count=2, time_limit_seconds=5)

    rendered = render_summary(result)
    assert "Objective value:" in rendered
    assert "Feasibility status: feasible" in rendered
    assert "Runtime:" in rendered
    assert "Vehicle 0:" in rendered

    output = tmp_path / "result.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    payload = json.loads(output.read_text(encoding="utf-8"))
    for field in (
        "instance_name",
        "customer_count",
        "vehicle_count",
        "solver_status",
        "solver_objective",
        "validation_total_distance",
        "feasible",
        "runtime_seconds",
        "routes",
        "violations",
    ):
        assert field in payload

