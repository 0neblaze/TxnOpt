from __future__ import annotations

import pytest

from txnopt_cases.evrptw import Instance, Node, NodeType, Vehicle
from txnopt_cases.evrptw.validation import RouteReportCache, validate_routes


def _instance() -> Instance:
    return Instance(
        "validation-cache",
        (
            Node("D", NodeType.DEPOT, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("F", NodeType.STATION, 1.0, 0.0, 0.0, 0.0, 100.0, 0.0),
            Node("C1", NodeType.CUSTOMER, 2.0, 0.0, 1.0, 0.0, 100.0, 0.5),
            Node("C2", NodeType.CUSTOMER, 3.0, 0.0, 1.0, 0.0, 100.0, 0.5),
        ),
        Vehicle(10.0, 10.0, 1.0, 0.1, 1.0),
        distance_backend="python",
    )


def test_cached_and_uncached_route_validation_are_identical() -> None:
    instance = _instance()
    cases = (
        (("D", "C1", "D"), ("D", "F", "C2", "F", "D")),
        (("D", "C1", "C2", "D"),),
        (("D", "UNKNOWN", "D"),),
        (("C1", "D"), ("D", "C2", "D")),
        (("D", "C1", "D"), ("D", "C1", "C2", "D")),
    )
    cache = RouteReportCache(8)

    for routes in cases:
        assert validate_routes(instance, routes, route_report_cache=cache) == (
            validate_routes(instance, routes)
        )


def test_route_report_cache_is_bounded_and_evicts_least_recently_used() -> None:
    instance = _instance()
    cache = RouteReportCache(2)
    first = ("D", "C1", "D")
    second = ("D", "C2", "D")
    third = ("D", "F", "C1", "F", "D")

    validate_routes(instance, (first, second), route_report_cache=cache)
    validate_routes(instance, (first,), route_report_cache=cache)
    validate_routes(instance, (third,), route_report_cache=cache)
    before_revisit = dict(cache.statistics)
    validate_routes(instance, (second,), route_report_cache=cache)

    assert before_revisit == {
        "route_validation_cache_hits": 1,
        "route_validation_cache_misses": 3,
        "route_validation_cache_size": 2,
        "route_validation_cache_capacity": 2,
        "route_validation_cache_evictions": 1,
    }
    assert cache.statistics["route_validation_cache_misses"] == 4
    assert cache.statistics["route_validation_cache_evictions"] == 2


def test_cached_route_reports_do_not_cache_solution_level_customer_coverage() -> None:
    instance = _instance()
    cache = RouteReportCache(4)
    complete = (("D", "C1", "D"), ("D", "C2", "D"))
    duplicated = (("D", "C1", "D"), ("D", "C1", "C2", "D"))

    complete_report = validate_routes(instance, complete, route_report_cache=cache)
    duplicated_report = validate_routes(instance, duplicated, route_report_cache=cache)

    assert complete_report.feasible is True
    assert duplicated_report.feasible is False
    assert "customers visited more than once: C1" in duplicated_report.violations


def test_route_report_cache_cannot_cross_instance_contexts() -> None:
    first = _instance()
    second = _instance()
    cache = RouteReportCache(4)

    validate_routes(first, (("D", "C1", "D"),), route_report_cache=cache)
    with pytest.raises(ValueError, match="cannot cross instance contexts"):
        validate_routes(second, (("D", "C1", "D"),), route_report_cache=cache)
