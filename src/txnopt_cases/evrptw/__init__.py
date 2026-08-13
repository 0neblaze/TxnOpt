"""EVRPTW case boundary for TxnOpt.

The active adapter is intentionally not wired in this first structural slice.
Until objective/validator differential gates pass, importing this namespace
must not activate the frozen ``evrptw`` solver or native ABI.
"""

from txnopt_cases.evrptw.charging import (
    ChargingLabel,
    ChargingSubproblemResult,
    solve_exact_charging,
)
from txnopt_cases.evrptw.kernel import EVRPTWSearchKernel
from txnopt_cases.evrptw.models import Instance, Node, NodeType, Vehicle
from txnopt_cases.evrptw.native_oracle import NativeEVRPTWOracle
from txnopt_cases.evrptw.objective import (
    ObjectiveComparison,
    SolutionObjective,
    accept_annealing_move,
    compare_objectives,
)
from txnopt_cases.evrptw.oracle import EVRPTWOracle, EVRPTWPlan, EVRPTWSolution
from txnopt_cases.evrptw.parser import parse_schneider
from txnopt_cases.evrptw.validation import RouteReport, SolutionReport, validate_routes

ADAPTER_STATUS = "model_parser_objective_validator_charging_migrated"

__all__ = [
    "ADAPTER_STATUS",
    "ChargingLabel",
    "ChargingSubproblemResult",
    "EVRPTWOracle",
    "EVRPTWPlan",
    "EVRPTWSearchKernel",
    "EVRPTWSolution",
    "Instance",
    "Node",
    "NodeType",
    "NativeEVRPTWOracle",
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
