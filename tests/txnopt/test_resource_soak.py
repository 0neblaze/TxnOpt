from __future__ import annotations

import pytest

from txnopt_evidence.resource_soak import run_native_resource_soak


@pytest.mark.formal_environment
def test_native_resource_soak_keeps_threads_fds_and_rss_bounded() -> None:
    result = run_native_resource_soak(
        rounds=16,
        workers=2,
        warmup_rounds=2,
        maximum_rss_growth_kib=65_536,
    )

    assert result["status"] == "PASS"
    assert result["native_round_call_count"] == 18
    assert result["fallback_count"] == 0
