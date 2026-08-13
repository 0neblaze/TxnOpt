"""Compatibility import for the active TxnOpt EVRPTW validator."""

from txnopt_cases.evrptw.validation import RouteReport, SolutionReport, validate_routes

__all__ = ["RouteReport", "SolutionReport", "validate_routes"]
