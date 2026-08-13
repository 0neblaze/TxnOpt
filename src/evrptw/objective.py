"""Compatibility import for the active TxnOpt EVRPTW objective policy."""

from txnopt_cases.evrptw.objective import (
    OBJECTIVE_PRECISION_DIGITS,
    OBJECTIVE_SCHEMA_VERSION,
    ObjectiveComparison,
    SolutionObjective,
    accept_annealing_move,
    canonical_objective_component,
    compare_objectives,
    count_charging_visits,
    objective_key_from_exact_numeric,
)

__all__ = [
    "OBJECTIVE_PRECISION_DIGITS",
    "OBJECTIVE_SCHEMA_VERSION",
    "ObjectiveComparison",
    "SolutionObjective",
    "accept_annealing_move",
    "canonical_objective_component",
    "compare_objectives",
    "count_charging_visits",
    "objective_key_from_exact_numeric",
]
