"""Auditable conditional Metal pilot protocol for Stage 5.2.

This module deliberately does not contain a pretend GPU implementation.  The
production adapter invokes an explicit Metal helper and rejects fallback.  A
missing helper produces partial/NOT_READY evidence; tests can inject an
executor while exercising the same validation and promotion gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from evrptw.stage052 import (
    AcceleratorDecision,
    PerformanceObservation,
    decide_accelerator,
)

METAL_OCCUPANCY_THRESHOLD = 32.0
METAL_PILOT_SCHEMA_VERSION = "stage05.2-metal-pilot-v1"
METAL_HELPER_SCHEMA_VERSION = "stage05.2-metal-helper-v1"


class MetalPilotStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class MetalRuntimeUnavailable(RuntimeError):
    """Raised when no explicit Metal runtime can execute the pilot."""


class MetalPilotExecutor(Protocol):
    @property
    def runtime_identity(self) -> Mapping[str, object]: ...

    def execute(
        self, native_observations: Sequence[PerformanceObservation]
    ) -> Sequence[PerformanceObservation]: ...


@dataclass(frozen=True, slots=True)
class InjectedMetalPilotExecutor:
    """Test/embedding seam that still passes through the public audit gates."""

    callback: Callable[[Sequence[PerformanceObservation]], Sequence[PerformanceObservation]]
    identity: Mapping[str, object] | None = None

    def __init__(
        self,
        callback: Callable[[Sequence[PerformanceObservation]], Sequence[PerformanceObservation]],
        *,
        runtime_identity: Mapping[str, object] | None = None,
    ) -> None:
        object.__setattr__(self, "callback", callback)
        object.__setattr__(self, "identity", runtime_identity)

    @property
    def runtime_identity(self) -> Mapping[str, object]:
        return self.identity or {
            "runtime": "injected-metal-executor",
            "production_evidence_eligible": False,
        }

    def execute(
        self, native_observations: Sequence[PerformanceObservation]
    ) -> Sequence[PerformanceObservation]:
        return self.callback(native_observations)


@dataclass(frozen=True, slots=True)
class SubprocessMetalPilotExecutor:
    """Run a separately supplied Metal helper using a strict JSON protocol."""

    helper_path: Path
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("Metal helper timeout must be finite and positive")

    @property
    def runtime_identity(self) -> Mapping[str, object]:
        path = self.helper_path.resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise MetalRuntimeUnavailable(f"Metal helper is unavailable or not executable: {path}")
        return {
            "runtime": "subprocess-metal-helper",
            "helper_path": str(path),
            "helper_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "protocol_schema_version": METAL_HELPER_SCHEMA_VERSION,
            "production_evidence_eligible": True,
        }

    def execute(
        self, native_observations: Sequence[PerformanceObservation]
    ) -> Sequence[PerformanceObservation]:
        identity = self.runtime_identity
        request = {
            "schema_version": METAL_HELPER_SCHEMA_VERSION,
            "execution_backend": "metal",
            "fallback_allowed": False,
            "rows": [_observation_to_dict(item) for item in native_observations],
        }
        completed = subprocess.run(
            [str(identity["helper_path"])],
            input=json.dumps(request, sort_keys=True),
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Metal helper failed without fallback: "
                f"exit={completed.returncode} stderr={completed.stderr.strip()}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise ValueError("Metal helper returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("Metal helper response must be an object")
        if payload.get("schema_version") != METAL_HELPER_SCHEMA_VERSION:
            raise ValueError("Metal helper schema version mismatch")
        if payload.get("execution_backend") != "metal":
            raise ValueError("Metal helper did not report the metal execution backend")
        if payload.get("fallback_used") is not False:
            raise ValueError("Metal helper fallback is forbidden")
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("Metal helper rows must be a list")
        return tuple(_observation_from_dict(item) for item in rows)


@dataclass(frozen=True, slots=True)
class MetalPilotResult:
    status: MetalPilotStatus
    median_batch_occupancy: float
    decision: AcceleratorDecision | None
    selected_backend: str | None
    native_observations: tuple[PerformanceObservation, ...]
    metal_observations: tuple[PerformanceObservation, ...]
    runtime_identity: Mapping[str, object] | None
    semantic_equality_passed: bool
    aggregate_median_saving: float | None
    family_median_savings: Mapping[str, float]
    fallback_used: bool
    failure_code: str | None = None
    failure_detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": METAL_PILOT_SCHEMA_VERSION,
            "status": self.status.value,
            "threshold": METAL_OCCUPANCY_THRESHOLD,
            "median_batch_occupancy": self.median_batch_occupancy,
            "decision": self.decision.value if self.decision is not None else None,
            "selected_backend": self.selected_backend,
            "selected_exact_backend": (
                "cpu_batch" if self.status is MetalPilotStatus.COMPLETE else None
            ),
            "native_rows": [_observation_to_dict(item) for item in self.native_observations],
            "metal_rows": [_observation_to_dict(item) for item in self.metal_observations],
            "runtime_identity": dict(self.runtime_identity) if self.runtime_identity else None,
            "semantic_equality_passed": self.semantic_equality_passed,
            "aggregate_median_saving": self.aggregate_median_saving,
            "family_median_savings": dict(self.family_median_savings),
            "minimum_aggregate_saving": 0.15,
            "maximum_family_regression": 0.03,
            "fallback_used": self.fallback_used,
            "failure_code": self.failure_code,
            "failure_detail": self.failure_detail,
        }


def campaign_execution_adapter_gate(result: MetalPilotResult) -> tuple[bool, str]:
    """Block Metal promotion until the campaign has an audited shard adapter."""

    if result.status is not MetalPilotStatus.COMPLETE or result.decision is None:
        return False, "campaign execution selection is unavailable"
    if result.decision is AcceleratorDecision.ACCELERATOR_PROMOTED:
        return (
            False,
            "Metal pilot passed promotion thresholds, but Stage 5.2 has no audited "
            "Metal campaign shard adapter; F02 remains NOT_READY",
        )
    return True, "selected native CPU campaign execution is implemented"


def run_conditional_metal_pilot(
    *,
    median_batch_occupancy: float,
    native_observations: Sequence[PerformanceObservation],
    executor: MetalPilotExecutor | None = None,
) -> MetalPilotResult:
    """Execute the conditional branch without implicit CPU fallback."""

    if not math.isfinite(median_batch_occupancy) or median_batch_occupancy < 0.0:
        raise ValueError("median_batch_occupancy must be finite and non-negative")
    native = tuple(native_observations)
    if median_batch_occupancy < METAL_OCCUPANCY_THRESHOLD:
        if native:
            raise ValueError("Metal performance rows are forbidden below occupancy 32")
        return MetalPilotResult(
            status=MetalPilotStatus.COMPLETE,
            median_batch_occupancy=median_batch_occupancy,
            decision=AcceleratorDecision.GPU_NOT_JUSTIFIED,
            selected_backend="native_cpu",
            native_observations=(),
            metal_observations=(),
            runtime_identity=None,
            semantic_equality_passed=True,
            aggregate_median_saving=None,
            family_median_savings={},
            fallback_used=False,
        )
    _validate_native_scope(native)
    if executor is None:
        return _runtime_unavailable_result(
            median_batch_occupancy,
            native,
            "no explicit Metal helper/executor was configured",
        )
    try:
        runtime_identity = dict(executor.runtime_identity)
    except MetalRuntimeUnavailable as error:
        return _runtime_unavailable_result(median_batch_occupancy, native, str(error))
    if runtime_identity.get("runtime") in {None, ""}:
        raise ValueError("Metal runtime identity is incomplete")
    metal = tuple(executor.execute(native))
    _validate_native_scope(metal)
    decision = decide_accelerator(
        median_batch_occupancy=median_batch_occupancy,
        native_cpu=native,
        accelerator=metal,
    )
    promotion = _promotion_details(native, metal)
    selected_backend = (
        "metal" if decision is AcceleratorDecision.ACCELERATOR_PROMOTED else "native_cpu"
    )
    return MetalPilotResult(
        status=MetalPilotStatus.COMPLETE,
        median_batch_occupancy=median_batch_occupancy,
        decision=decision,
        selected_backend=selected_backend,
        native_observations=native,
        metal_observations=metal,
        runtime_identity=runtime_identity,
        semantic_equality_passed=promotion[0],
        aggregate_median_saving=promotion[1],
        family_median_savings=promotion[2],
        fallback_used=False,
    )


def audit_metal_pilot_result(
    payload: Mapping[str, object],
    *,
    expected_native: Sequence[PerformanceObservation],
) -> MetalPilotResult:
    """Independently replay a serialized pilot decision and reject tampering."""

    expected_keys = {
        "schema_version",
        "status",
        "threshold",
        "median_batch_occupancy",
        "decision",
        "selected_backend",
        "selected_exact_backend",
        "native_rows",
        "metal_rows",
        "runtime_identity",
        "semantic_equality_passed",
        "aggregate_median_saving",
        "family_median_savings",
        "minimum_aggregate_saving",
        "maximum_family_regression",
        "fallback_used",
        "failure_code",
        "failure_detail",
    }
    if set(payload) != expected_keys:
        raise ValueError("Metal pilot result schema mismatch")
    if (
        payload.get("schema_version") != METAL_PILOT_SCHEMA_VERSION
        or payload.get("threshold") != METAL_OCCUPANCY_THRESHOLD
        or payload.get("minimum_aggregate_saving") != 0.15
        or payload.get("maximum_family_regression") != 0.03
        or payload.get("fallback_used") is not False
    ):
        raise ValueError("Metal pilot constants or fallback state are invalid")
    median = _finite_float(payload["median_batch_occupancy"], "median_batch_occupancy")
    native_rows = payload.get("native_rows")
    metal_rows = payload.get("metal_rows")
    if not isinstance(native_rows, list) or not isinstance(metal_rows, list):
        raise ValueError("Metal pilot observation rows must be lists")
    native = tuple(_observation_from_dict(item) for item in native_rows)
    metal = tuple(_observation_from_dict(item) for item in metal_rows)
    if tuple(_observation_to_dict(item) for item in native) != tuple(
        _observation_to_dict(item) for item in expected_native
    ):
        raise ValueError("Metal pilot native rows do not match accepted E evidence")
    status = MetalPilotStatus(str(payload["status"]))
    if status is MetalPilotStatus.PARTIAL:
        if (
            median < METAL_OCCUPANCY_THRESHOLD
            or metal
            or payload.get("decision") is not None
            or payload.get("selected_backend") is not None
            or payload.get("selected_exact_backend") is not None
            or payload.get("failure_code") != "METAL_RUNTIME_UNAVAILABLE"
        ):
            raise ValueError("partial Metal pilot state is invalid")
        return MetalPilotResult(
            status=status,
            median_batch_occupancy=median,
            decision=None,
            selected_backend=None,
            native_observations=native,
            metal_observations=(),
            runtime_identity=None,
            semantic_equality_passed=False,
            aggregate_median_saving=None,
            family_median_savings={},
            fallback_used=False,
            failure_code="METAL_RUNTIME_UNAVAILABLE",
            failure_detail=str(payload.get("failure_detail")),
        )
    if median < METAL_OCCUPANCY_THRESHOLD:
        expected_decision = AcceleratorDecision.GPU_NOT_JUSTIFIED
        expected_backend = "native_cpu"
        semantic = True
        aggregate: float | None = None
        families: Mapping[str, float] = {}
    else:
        expected_decision = decide_accelerator(
            median_batch_occupancy=median,
            native_cpu=native,
            accelerator=metal,
        )
        expected_backend = (
            "metal"
            if expected_decision is AcceleratorDecision.ACCELERATOR_PROMOTED
            else "native_cpu"
        )
        semantic, aggregate, families = _promotion_details(native, metal)
    if payload.get("decision") != expected_decision.value:
        raise ValueError("Metal pilot decision does not replay")
    if payload.get("selected_backend") != expected_backend:
        raise ValueError("Metal pilot selected backend does not replay")
    if payload.get("selected_exact_backend") != "cpu_batch":
        raise ValueError("Metal pilot exact backend is invalid")
    if payload.get("semantic_equality_passed") is not semantic:
        raise ValueError("Metal pilot semantic equality gate does not replay")
    observed_aggregate = payload.get("aggregate_median_saving")
    if aggregate is None:
        if observed_aggregate is not None:
            raise ValueError("decision-only pilot cannot report aggregate saving")
    elif not math.isclose(
        _finite_float(observed_aggregate, "aggregate_median_saving"),
        aggregate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Metal pilot aggregate saving does not replay")
    observed_families = payload.get("family_median_savings")
    if not isinstance(observed_families, Mapping) or {
        str(key): float(value) for key, value in observed_families.items()
    } != dict(families):
        raise ValueError("Metal pilot family savings do not replay")
    runtime_identity = payload.get("runtime_identity")
    if median >= METAL_OCCUPANCY_THRESHOLD and not isinstance(runtime_identity, Mapping):
        raise ValueError("complete Metal pilot runtime identity is missing")
    return MetalPilotResult(
        status=status,
        median_batch_occupancy=median,
        decision=expected_decision,
        selected_backend=expected_backend,
        native_observations=native,
        metal_observations=metal,
        runtime_identity=(
            dict(runtime_identity) if isinstance(runtime_identity, Mapping) else None
        ),
        semantic_equality_passed=semantic,
        aggregate_median_saving=aggregate,
        family_median_savings=families,
        fallback_used=False,
    )


def _runtime_unavailable_result(
    median_batch_occupancy: float,
    native: tuple[PerformanceObservation, ...],
    detail: str,
) -> MetalPilotResult:
    return MetalPilotResult(
        status=MetalPilotStatus.PARTIAL,
        median_batch_occupancy=median_batch_occupancy,
        decision=None,
        selected_backend=None,
        native_observations=native,
        metal_observations=(),
        runtime_identity=None,
        semantic_equality_passed=False,
        aggregate_median_saving=None,
        family_median_savings={},
        fallback_used=False,
        failure_code="METAL_RUNTIME_UNAVAILABLE",
        failure_detail=detail,
    )


def _validate_native_scope(native: Sequence[PerformanceObservation]) -> None:
    expected = {
        (instance, seed)
        for instance in ("c101_21", "r101_21", "rc101_21")
        for seed in (2014, 2015, 2016)
    }
    observed = {item.identity for item in native}
    if len(observed) != len(native):
        raise ValueError("duplicate native Metal-pilot observation")
    if observed != expected or any(
        item.customer_count != 100
        or not math.isfinite(item.end_to_end_seconds)
        or item.end_to_end_seconds <= 0.0
        for item in native
    ):
        raise ValueError("Metal pilot requires the exact nine C/R/RC 100-customer pairs")


def _promotion_details(
    native: Sequence[PerformanceObservation],
    metal: Sequence[PerformanceObservation],
) -> tuple[bool, float, Mapping[str, float]]:
    from evrptw.stage052 import evaluate_promotion

    decision = evaluate_promotion(native, metal)
    semantic = len(native) == len(metal) and {
        item.identity: item.semantic_digest for item in native
    } == {item.identity: item.semantic_digest for item in metal}
    return semantic, decision.aggregate_median_saving, decision.family_median_savings


def _observation_to_dict(item: PerformanceObservation) -> dict[str, object]:
    return {
        "instance": item.instance,
        "seed": item.seed,
        "customer_count": item.customer_count,
        "end_to_end_seconds": item.end_to_end_seconds,
        "semantic_digest": item.semantic_digest,
    }


def _observation_from_dict(payload: object) -> PerformanceObservation:
    if not isinstance(payload, dict):
        raise ValueError("Metal observation must be an object")
    expected = {
        "instance",
        "seed",
        "customer_count",
        "end_to_end_seconds",
        "semantic_digest",
    }
    if set(payload) != expected:
        raise ValueError("Metal observation schema mismatch")
    return PerformanceObservation(
        instance=str(payload["instance"]),
        seed=int(payload["seed"]),
        customer_count=int(payload["customer_count"]),
        end_to_end_seconds=_finite_float(payload["end_to_end_seconds"], "end_to_end_seconds"),
        semantic_digest=str(payload["semantic_digest"]),
    )


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result
