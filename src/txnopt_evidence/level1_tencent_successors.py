"""Closed identities for Tencent pre-cloud Level 1 successor evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TencentLevel1Successor:
    build_number: int
    formal_attempt: int
    calibration_attempt: int
    build_status: str

    @property
    def build_name(self) -> str:
        return f"Build{self.build_number}"

    @property
    def run_label(self) -> str:
        return f"txnopt_level1_build_attempt{self.build_number}"

    @property
    def pending_review_status(self) -> str:
        return f"REVIEW_PENDING_BUILD{self.build_number}"


_SUCCESSORS = {
    successor.run_label: successor
    for successor in (
        TencentLevel1Successor(
            build_number=16,
            formal_attempt=26,
            calibration_attempt=27,
            build_status="BUILD_COMPLETE_TENCENT_CLOUD_CUTOVER_NOT_LEVEL1_READY",
        ),
        TencentLevel1Successor(
            build_number=18,
            formal_attempt=28,
            calibration_attempt=29,
            build_status=(
                "BUILD_COMPLETE_TENCENT_PRE_CLOUD_SUCCESSOR_NOT_LEVEL1_READY"
            ),
        ),
        TencentLevel1Successor(
            build_number=19,
            formal_attempt=30,
            calibration_attempt=31,
            build_status=(
                "BUILD_COMPLETE_TENCENT_PRECLOUD_REVIEW_SUCCESSOR_NOT_LEVEL1_READY"
            ),
        ),
    )
}


def successor_from_build_manifest(
    manifest: Mapping[str, Any],
) -> TencentLevel1Successor:
    """Return the one approved closed successor identity or fail closed."""

    if manifest.get("schema_version") != "txnopt-level1-build-manifest-v1":
        raise ValueError("unsupported TxnOpt build manifest schema")
    run_label = manifest.get("run_label")
    successor = _SUCCESSORS.get(run_label) if isinstance(run_label, str) else None
    if successor is None:
        raise ValueError("build is not an approved Tencent pre-cloud successor")
    formal = manifest.get("formal_successor")
    if not isinstance(formal, dict):
        raise ValueError(f"{successor.build_name} formal-successor boundary differs")
    if (
        manifest.get("status") != successor.build_status
        or formal.get("prior_review_binding_status") != "PRIOR_SOURCE_ONLY"
        or formal.get("successor_status") != successor.pending_review_status
        or formal.get("independent_successor_review_completed") is not False
        or formal.get("level1_formal_gate_passed") is not False
    ):
        raise ValueError(f"{successor.build_name} formal-successor boundary differs")
    return successor


def require_formal_attempt(
    successor: TencentLevel1Successor,
    attempt: int,
) -> TencentLevel1Successor:
    if attempt != successor.formal_attempt:
        raise ValueError(
            f"{successor.build_name} formal attempt must be "
            f"Attempt{successor.formal_attempt}"
        )
    return successor


def require_calibration_attempt(
    successor: TencentLevel1Successor,
    attempt: int,
) -> TencentLevel1Successor:
    if attempt != successor.calibration_attempt:
        raise ValueError(
            f"{successor.build_name} calibration attempt must be "
            f"Attempt{successor.calibration_attempt}"
        )
    return successor


def successor_from_formal_attempt(attempt: int) -> TencentLevel1Successor:
    matches = [
        successor
        for successor in _SUCCESSORS.values()
        if successor.formal_attempt == attempt
    ]
    if len(matches) != 1:
        raise ValueError("formal attempt is not an approved Tencent successor")
    return matches[0]


def calibration_peak_rss_from_payload(
    payload: Mapping[str, Any],
    successor: TencentLevel1Successor,
) -> int:
    if successor.build_number == 16:
        value = payload.get("attempt27_peak_rss_bytes")
    else:
        if payload.get("calibration_attempt") != successor.calibration_attempt:
            raise ValueError("Tencent authorization calibration identity differs")
        value = payload.get("calibration_peak_rss_bytes")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Tencent authorization calibration peak RSS is invalid")
    return value


__all__ = [
    "TencentLevel1Successor",
    "calibration_peak_rss_from_payload",
    "require_calibration_attempt",
    "require_formal_attempt",
    "successor_from_build_manifest",
    "successor_from_formal_attempt",
]
