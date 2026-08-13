"""EVRPTW model, parser, objective, validator, kernels, and Oracle boundary."""

from txnopt_cases.evrptw.charging import (
    ChargingLabel,
    ChargingSubproblemResult,
    solve_exact_charging,
)
from txnopt_cases.evrptw.initialization import construct_initial_plan
from txnopt_cases.evrptw.kernel import EVRPTWSearchKernel
from txnopt_cases.evrptw.models import Instance, Node, NodeType, Vehicle
from txnopt_cases.evrptw.objective import (
    ObjectiveComparison,
    SolutionObjective,
    accept_annealing_move,
    compare_objectives,
)
from txnopt_cases.evrptw.oracle import EVRPTWOracle, EVRPTWPlan, EVRPTWSolution
from txnopt_cases.evrptw.parser import parse_schneider
from txnopt_cases.evrptw.validation import RouteReport, SolutionReport, validate_routes

ADAPTER_STATUS = "active_python_and_native_round_v1"

__all__ = [
    "ADAPTER_STATUS",
    "ChargingLabel",
    "ChargingSubproblemResult",
    "EVRPTWOracle",
    "EVRPTWPlan",
    "EVRPTWSearchKernel",
    "construct_initial_plan",
    "EVRPTWSolution",
    "Instance",
    "Node",
    "NodeType",
    "ObjectiveComparison",
    "RouteReport",
    "SolutionObjective",
    "SolutionReport",
    "Vehicle",
    "accept_annealing_move",
    "compare_objectives",
    "parse_schneider",
    "solve_exact_charging",
    "validate_routes",
]
