"""EVRPTW case boundary for TxnOpt.

The active adapter is intentionally not wired in this first structural slice.
Until objective/validator differential gates pass, importing this namespace
must not activate the frozen ``evrptw`` solver or native ABI.
"""

from txnopt_cases.evrptw.objective import (
    ObjectiveComparison,
    SolutionObjective,
    accept_annealing_move,
    compare_objectives,
)
from txnopt_cases.evrptw.validation import RouteReport, SolutionReport, validate_routes

ADAPTER_STATUS = "objective_and_validator_migrated"

__all__ = [
    "ADAPTER_STATUS",
    "ObjectiveComparison",
    "RouteReport",
    "SolutionObjective",
    "SolutionReport",
    "accept_annealing_move",
    "compare_objectives",
    "validate_routes",
]
