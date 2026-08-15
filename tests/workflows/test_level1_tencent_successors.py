from __future__ import annotations

import pytest

from txnopt_evidence.level1_tencent_successors import (
    calibration_peak_rss_from_payload,
    require_calibration_attempt,
    require_formal_attempt,
    successor_from_build_manifest,
    successor_from_formal_attempt,
)


def _build(number: int) -> dict[str, object]:
    statuses = {
        16: "BUILD_COMPLETE_TENCENT_CLOUD_CUTOVER_NOT_LEVEL1_READY",
        18: "BUILD_COMPLETE_TENCENT_PRE_CLOUD_SUCCESSOR_NOT_LEVEL1_READY",
        19: "BUILD_COMPLETE_TENCENT_PRECLOUD_REVIEW_SUCCESSOR_NOT_LEVEL1_READY",
        20: "BUILD_COMPLETE_TENCENT_PRECLOUD_REVIEW_SUCCESSOR_NOT_LEVEL1_READY",
        21: "BUILD_COMPLETE_TENCENT_PRECLOUD_REVIEW_SUCCESSOR_NOT_LEVEL1_READY",
        23: "BUILD_COMPLETE_TENCENT_PRECLOUD_REVIEW_SUCCESSOR_NOT_LEVEL1_READY",
    }
    return {
        "schema_version": "txnopt-level1-build-manifest-v1",
        "run_label": f"txnopt_level1_build_attempt{number}",
        "status": statuses[number],
        "formal_successor": {
            "prior_review_binding_status": "PRIOR_SOURCE_ONLY",
            "successor_status": f"REVIEW_PENDING_BUILD{number}",
            "independent_successor_review_completed": False,
            "level1_formal_gate_passed": False,
        },
    }


@pytest.mark.parametrize(
    ("build_number", "formal_attempt", "calibration_attempt"),
    [
        (16, 26, 27),
        (18, 28, 29),
        (19, 30, 31),
        (20, 32, 33),
        (21, 34, 35),
        (23, 38, 39),
    ],
)
def test_registry_closes_each_build_to_one_formal_and_calibration_attempt(
    build_number: int,
    formal_attempt: int,
    calibration_attempt: int,
) -> None:
    successor = successor_from_build_manifest(_build(build_number))

    assert require_formal_attempt(successor, formal_attempt) == successor
    assert require_calibration_attempt(successor, calibration_attempt) == successor
    with pytest.raises(ValueError, match="formal attempt"):
        require_formal_attempt(successor, formal_attempt + 1)
    with pytest.raises(ValueError, match="calibration attempt"):
        require_calibration_attempt(successor, calibration_attempt + 1)


def test_registry_rejects_unapproved_build17_candidate() -> None:
    with pytest.raises(ValueError, match="approved Tencent pre-cloud successor"):
        successor_from_build_manifest(
            {
                "schema_version": "txnopt-level1-build-manifest-v1",
                "run_label": "txnopt_level1_build_attempt17",
                "status": "BUILD_COMPLETE_TENCENT_PRE_CLOUD_SUCCESSOR_NOT_LEVEL1_READY",
                "formal_successor": {
                    "prior_review_binding_status": "PRIOR_SOURCE_ONLY",
                    "successor_status": "REVIEW_PENDING_BUILD17",
                    "independent_successor_review_completed": False,
                    "level1_formal_gate_passed": False,
                },
            }
        )


def test_registry_rejects_failed_build22_candidate() -> None:
    with pytest.raises(ValueError, match="approved Tencent pre-cloud successor"):
        successor_from_build_manifest(
            {
                "schema_version": "txnopt-level1-build-manifest-v1",
                "run_label": "txnopt_level1_build_attempt22",
                "status": (
                    "BUILD_COMPLETE_TENCENT_PRECLOUD_REVIEW_SUCCESSOR_NOT_LEVEL1_READY"
                ),
                "formal_successor": {
                    "prior_review_binding_status": "PRIOR_SOURCE_ONLY",
                    "successor_status": "REVIEW_PENDING_BUILD22",
                    "independent_successor_review_completed": False,
                    "level1_formal_gate_passed": False,
                },
            }
        )


def test_registry_rejects_a_loosened_build18_review_boundary() -> None:
    build = _build(18)
    formal = dict(build["formal_successor"])  # type: ignore[arg-type]
    formal["independent_successor_review_completed"] = True
    build["formal_successor"] = formal

    with pytest.raises(ValueError, match="formal-successor boundary"):
        successor_from_build_manifest(build)


def test_calibration_rss_authorization_is_attempt_specific() -> None:
    build16 = successor_from_formal_attempt(26)
    build18 = successor_from_formal_attempt(28)

    assert calibration_peak_rss_from_payload(
        {"attempt27_peak_rss_bytes": 123},
        build16,
    ) == 123
    assert calibration_peak_rss_from_payload(
        {"calibration_attempt": 29, "calibration_peak_rss_bytes": 456},
        build18,
    ) == 456
    with pytest.raises(ValueError, match="calibration identity"):
        calibration_peak_rss_from_payload(
            {"attempt27_peak_rss_bytes": 456},
            build18,
        )
