from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # type: ignore[import-untyped]

from evrptw.models import Instance
from evrptw.parser import parse_schneider
from evrptw.validation import validate_routes

_STATUS_NAMES = {
    0: "ROUTING_NOT_SOLVED",
    1: "ROUTING_SUCCESS",
    2: "ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED",
    3: "ROUTING_FAIL",
    4: "ROUTING_FAIL_TIMEOUT",
    5: "ROUTING_INVALID",
    6: "ROUTING_INFEASIBLE",
    7: "ROUTING_OPTIMAL",
}

_FIRST_SOLUTION_STRATEGIES = {
    "PATH_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    "PARALLEL_CHEAPEST_INSERTION": (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    ),
}


@dataclass(frozen=True, slots=True)
class RouteOutput:
    vehicle_id: int
    route: list[str]
    arrivals: list[int]
    loads_after_service: list[int]
    distance: float
    finish_time: float
    feasible: bool
    violations: list[str]


def _scaled_distance_matrix(instance: Instance) -> list[list[int]]:
    nodes = list(instance.nodes)
    return [
        [int(round(origin.distance_to(destination))) for destination in nodes]
        for origin in nodes
    ]


def _vrptw_nodes(instance: Instance) -> list[Any]:
    return [instance.depot, *instance.customers]


def _build_manager_and_model(
    instance: Instance,
    vehicle_count: int,
) -> tuple[
    pywrapcp.RoutingIndexManager,
    pywrapcp.RoutingModel,
    list[list[int]],
    list[Any],
]:
    nodes = _vrptw_nodes(instance)
    depot_index = nodes.index(instance.depot)
    manager = pywrapcp.RoutingIndexManager(len(nodes), vehicle_count, depot_index)
    routing = pywrapcp.RoutingModel(manager)
    distances = [
        [int(round(origin.distance_to(destination))) for destination in nodes]
        for origin in nodes
    ]

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = cast(int, manager.IndexToNode(from_index))
        to_node = cast(int, manager.IndexToNode(to_index))
        return distances[from_node][to_node]

    distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

    def demand_callback(from_index: int) -> int:
        node = nodes[cast(int, manager.IndexToNode(from_index))]
        return int(round(node.demand))

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [int(round(instance.vehicle.load_capacity))] * vehicle_count,
        True,
        "Capacity",
    )

    def time_callback(from_index: int, to_index: int) -> int:
        from_id = cast(int, manager.IndexToNode(from_index))
        to_id = cast(int, manager.IndexToNode(to_index))
        from_node = nodes[from_id]
        travel_time = distances[from_id][to_id] / instance.vehicle.average_velocity
        return int(round(travel_time + from_node.service_time))

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    horizon = int(round(max(node.due_date for node in nodes)))
    routing.AddDimension(
        time_callback_index,
        horizon,
        horizon,
        False,
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    for node_index, node in enumerate(nodes):
        routing_index = manager.NodeToIndex(node_index)
        time_dimension.CumulVar(routing_index).SetRange(
            int(round(node.ready_time)),
            int(round(node.due_date)),
        )
    for vehicle_id in range(vehicle_count):
        start_index = routing.Start(vehicle_id)
        end_index = routing.End(vehicle_id)
        depot = instance.depot
        time_dimension.CumulVar(start_index).SetRange(
            int(round(depot.ready_time)),
            int(round(depot.due_date)),
        )
        time_dimension.CumulVar(end_index).SetRange(
            int(round(depot.ready_time)),
            int(round(depot.due_date)),
        )

    return manager, routing, distances, nodes


def _extract_routes(
    nodes: list[Any],
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    solution: pywrapcp.Assignment,
    vehicle_count: int,
) -> list[list[str]]:
    routes: list[list[str]] = []
    for vehicle_id in range(vehicle_count):
        index = routing.Start(vehicle_id)
        route: list[str] = []
        while not routing.IsEnd(index):
            route.append(nodes[manager.IndexToNode(index)].name)
            index = solution.Value(routing.NextVar(index))
        route.append(nodes[manager.IndexToNode(index)].name)
        if len(route) > 2:
            routes.append(route)
    return routes


def _route_outputs(
    instance: Instance,
    routes: list[list[str]],
) -> list[RouteOutput]:
    report = validate_routes(instance, routes)
    by_name = instance.by_name
    outputs: list[RouteOutput] = []
    for vehicle_id, route_report in enumerate(report.routes):
        time_value = max(0.0, instance.depot.ready_time)
        cumulative_load = 0.0
        arrivals = [int(round(time_value))]
        loads = [0]
        route = list(route_report.route)
        for origin_name, destination_name in zip(route, route[1:], strict=False):
            origin = by_name[origin_name]
            destination = by_name[destination_name]
            time_value += round(origin.distance_to(destination)) / instance.vehicle.average_velocity
            time_value = max(time_value, destination.ready_time)
            arrivals.append(int(round(time_value)))
            cumulative_load += destination.demand
            loads.append(int(round(cumulative_load)))
            time_value += destination.service_time
        outputs.append(
            RouteOutput(
                vehicle_id=vehicle_id,
                route=route,
                arrivals=arrivals,
                loads_after_service=loads,
                distance=route_report.distance,
                finish_time=route_report.finish_time,
                feasible=route_report.feasible,
                violations=list(route_report.violations),
            )
        )
    return outputs


def solve_vrptw(
    instance: Instance,
    *,
    vehicle_count: int = 2,
    time_limit_seconds: int = 5,
    first_solution_strategy: str = "PATH_CHEAPEST_ARC",
) -> dict[str, Any]:
    if first_solution_strategy not in _FIRST_SOLUTION_STRATEGIES:
        choices = ", ".join(sorted(_FIRST_SOLUTION_STRATEGIES))
        raise ValueError(f"unknown first solution strategy; choose one of: {choices}")

    manager, routing, _distances, nodes = _build_manager_and_model(instance, vehicle_count)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = _FIRST_SOLUTION_STRATEGIES[
        first_solution_strategy
    ]
    search_parameters.time_limit.FromSeconds(time_limit_seconds)

    started_at = time.perf_counter()
    solution = routing.SolveWithParameters(search_parameters)
    runtime_seconds = time.perf_counter() - started_at
    status = routing.status()
    status_name = _STATUS_NAMES.get(status, f"UNKNOWN_STATUS_{status}")

    if solution is None:
        return {
            "instance_name": instance.name,
            "customer_count": len(instance.customers),
            "vehicle_count": vehicle_count,
            "solver_status": status_name,
            "solver_objective": None,
            "validation_total_distance": None,
            "feasible": False,
            "runtime_seconds": runtime_seconds,
            "routes": [],
            "route_reports": [],
            "violations": ["OR-Tools did not return a solution"],
            "time_limit_seconds": time_limit_seconds,
            "first_solution_strategy": first_solution_strategy,
        }

    routes = _extract_routes(nodes, manager, routing, solution, vehicle_count)
    validation_report = validate_routes(instance, routes)
    route_outputs = _route_outputs(instance, routes)
    route_violations = [
        violation
        for route_output in route_outputs
        for violation in route_output.violations
    ]

    return {
        "instance_name": instance.name,
        "customer_count": len(instance.customers),
        "vehicle_count": vehicle_count,
        "solver_status": status_name,
        "solver_objective": solution.ObjectiveValue(),
        "validation_total_distance": validation_report.total_distance,
        "feasible": validation_report.feasible,
        "runtime_seconds": runtime_seconds,
        "routes": routes,
        "route_reports": [asdict(route_output) for route_output in route_outputs],
        "violations": [*validation_report.violations, *route_violations],
        "time_limit_seconds": time_limit_seconds,
        "first_solution_strategy": first_solution_strategy,
    }


def _format_route(route_report: dict[str, Any]) -> str:
    route = route_report["route"]
    arrivals = route_report["arrivals"]
    loads = route_report["loads_after_service"]
    stops = [
        f"{node}(t={arrival}, load={load})"
        for node, arrival, load in zip(route, arrivals, loads, strict=True)
    ]
    return " -> ".join(stops)


def render_summary(result: dict[str, Any]) -> str:
    lines = [
        f"Instance: {result['instance_name']} "
        f"({result['customer_count']} customers, {result['vehicle_count']} vehicles)",
        f"Solver status: {result['solver_status']}",
        f"Objective value: {result['solver_objective']}",
        f"Validated total distance: {result['validation_total_distance']}",
        f"Feasibility status: {'feasible' if result['feasible'] else 'infeasible'}",
        f"Runtime: {result['runtime_seconds']:.6f}s",
    ]
    for route_report in result["route_reports"]:
        lines.append(
            f"Vehicle {route_report['vehicle_id']}: "
            f"{_format_route(route_report)} "
            f"| distance={route_report['distance']:.3f} "
            f"| feasible={route_report['feasible']}"
        )
    if result["violations"]:
        lines.append("Violations:")
        lines.extend(f"- {violation}" for violation in result["violations"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Week 1 OR-Tools VRPTW baseline")
    parser.add_argument("--instance", required=True, type=Path, help="Schneider-style instance")
    parser.add_argument("--output", type=Path, help="write machine-readable JSON result")
    parser.add_argument("--vehicle-count", type=int, default=2)
    parser.add_argument("--time-limit-seconds", type=int, default=5)
    parser.add_argument(
        "--first-solution-strategy",
        choices=sorted(_FIRST_SOLUTION_STRATEGIES),
        default="PATH_CHEAPEST_ARC",
    )
    arguments = parser.parse_args()

    instance = parse_schneider(arguments.instance)
    result = solve_vrptw(
        instance,
        vehicle_count=arguments.vehicle_count,
        time_limit_seconds=arguments.time_limit_seconds,
        first_solution_strategy=arguments.first_solution_strategy,
    )
    print(render_summary(result))

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0 if result["feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
