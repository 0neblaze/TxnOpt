from __future__ import annotations

from txnopt_cases.evrptw.neighborhoods import (
    canonical_neighborhood_plans,
    merge_plans,
    relocate_plans,
    swap_plans,
)


def test_case_neighborhood_seam_is_canonical_and_domain_local() -> None:
    routes = (("C1",), ("C2", "C3"))
    combined = canonical_neighborhood_plans(routes)

    assert combined == tuple(
        dict.fromkeys((*relocate_plans(routes), *swap_plans(routes), *merge_plans(routes)))
    )
    assert len(combined) == len(set(combined))
    assert any(len(plan.customer_routes) == 1 for plan in combined)
