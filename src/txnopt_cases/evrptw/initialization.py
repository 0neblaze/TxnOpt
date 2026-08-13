"""Deterministic, case-local construction of an exact-feasible EVRPTW plan."""

from __future__ import annotations

from typing import Any, cast

from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # type: ignore[import-untyped]

from txnopt_cases.evrptw.charging import solve_exact_charging
from txnopt_cases.evrptw.models import Instance
from txnopt_cases.evrptw.oracle import EVRPTWOracle, EVRPTWPlan


def construct_initial_plan(instance: Instance) -> EVRPTWPlan:
    """Build one vehicle-first VRPTW plan, then split until charging-feasible."""

    if not instance.customers:
        raise ValueError("EVRPTW initial construction requires customers")
    sequences = _vrptw_customer_sequences(instance)
    charging_feasible = tuple(
        part
        for sequence in sequences
        for part in _split_until_charging_feasible(instance, sequence)
    )
    plan = EVRPTWPlan(charging_feasible)
    EVRPTWOracle(instance).solve_initial(plan)
    return plan


def _vrptw_customer_sequences(instance: Instance) -> tuple[tuple[str, ...], ...]:
    nodes = (instance.depot, *instance.customers)
    vehicle_count = len(instance.customers)
    manager = pywrapcp.RoutingIndexManager(len(nodes), vehicle_count, 0)
    routing = pywrapcp.RoutingModel(manager)
    distances = tuple(
        tuple(int(round(instance.distance(origin.name, destination.name))) for destination in nodes)
        for origin in nodes
    )

    def distance_callback(from_index: int, to_index: int) -> int:
        origin = cast(int, manager.IndexToNode(from_index))
        destination = cast(int, manager.IndexToNode(to_index))
        return distances[origin][destination]

    distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)
    maximum_arc = max(max(row) for row in distances)
    route_arc_upper_bound = (len(instance.customers) + vehicle_count) * maximum_arc
    routing.SetFixedCostOfAllVehicles(route_arc_upper_bound + 1)

    def demand_callback(from_index: int) -> int:
        node = nodes[cast(int, manager.IndexToNode(from_index))]
        return int(round(node.demand))

    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(demand_callback),
        0,
        [int(round(instance.vehicle.load_capacity))] * vehicle_count,
        True,
        "Capacity",
    )

    def time_callback(from_index: int, to_index: int) -> int:
        origin_index = cast(int, manager.IndexToNode(from_index))
        destination_index = cast(int, manager.IndexToNode(to_index))
        origin = nodes[origin_index]
        travel = distances[origin_index][destination_index] / instance.vehicle.average_velocity
        return int(round(travel + origin.service_time))

    horizon = int(round(max(node.due_date for node in nodes)))
    routing.AddDimension(
        routing.RegisterTransitCallback(time_callback),
        horizon,
        horizon,
        False,
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    for node_index, node in enumerate(nodes):
        time_dimension.CumulVar(manager.NodeToIndex(node_index)).SetRange(
            int(round(node.ready_time)),
            int(round(node.due_date)),
        )
    for vehicle_id in range(vehicle_count):
        for route_index in (routing.Start(vehicle_id), routing.End(vehicle_id)):
            time_dimension.CumulVar(route_index).SetRange(
                int(round(instance.depot.ready_time)),
                int(round(instance.depot.due_date)),
            )

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    parameters.time_limit.FromSeconds(2)
    solution: Any = routing.SolveWithParameters(parameters)
    if solution is None:
        raise RuntimeError("deterministic EVRPTW initial construction found no VRPTW plan")

    sequences: list[tuple[str, ...]] = []
    for vehicle_id in range(vehicle_count):
        index = routing.Start(vehicle_id)
        customer_sequence: list[str] = []
        while not routing.IsEnd(index):
            node_index = cast(int, manager.IndexToNode(index))
            if node_index != 0:
                customer_sequence.append(nodes[node_index].name)
            index = solution.Value(routing.NextVar(index))
        if customer_sequence:
            sequences.append(tuple(customer_sequence))
    if {name for route in sequences for name in route} != {
        customer.name for customer in instance.customers
    }:
        raise RuntimeError("EVRPTW constructor did not cover every customer exactly once")
    return tuple(sequences)


def _split_until_charging_feasible(
    instance: Instance,
    sequence: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    if solve_exact_charging(instance, sequence).feasible:
        return (sequence,)
    if len(sequence) <= 1:
        raise RuntimeError("VRPTW constructor produced an exact-infeasible singleton")
    midpoint = len(sequence) // 2
    return (
        *_split_until_charging_feasible(instance, sequence[:midpoint]),
        *_split_until_charging_feasible(instance, sequence[midpoint:]),
    )


__all__ = ["construct_initial_plan"]
