"""Stage 5.2 performance-governance contracts.

This module is the single public source for component ordering, paired
performance promotion, job-worker selection, accelerator decisions, and the
formal benchmark budget.  Experiment runners and independent reviewers must
call these functions instead of duplicating threshold logic.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class Stage052Component(StrEnum):
    PERF_BASELINE = "perf_baseline"
    HOT_PATH = "hot_path"
    ARTIFACT_STREAMING = "artifact_streaming"
    JOB_PARALLEL = "job_parallel"
    NATIVE_KERNELS = "native_kernels"
    ACCELERATOR_PILOT = "accelerator_pilot"
    BENCHMARK = "benchmark"


class AcceleratorDecision(StrEnum):
    GPU_NOT_JUSTIFIED = "GPU_NOT_JUSTIFIED"
    ACCELERATOR_PROMOTED = "ACCELERATOR_PROMOTED"
    NATIVE_CPU_RETAINED = "NATIVE_CPU_RETAINED"


@dataclass(frozen=True, slots=True)
class PerformanceObservation:
    instance: str
    seed: int
    customer_count: int
    end_to_end_seconds: float
    semantic_digest: str

    def __post_init__(self) -> None:
        if self.customer_count not in {5, 10, 15, 100}:
            raise ValueError("customer_count must be 5, 10, 15, or 100")
        if self.end_to_end_seconds <= 0.0:
            raise ValueError("end_to_end_seconds must be positive")
        if not self.semantic_digest:
            raise ValueError("semantic_digest is required")

    @property
    def identity(self) -> tuple[str, int]:
        return self.instance, self.seed


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    passed: bool
    aggregate_median_saving: float
    family_median_savings: Mapping[str, float]
    detail: str


@dataclass(frozen=True, slots=True)
class ArtifactStorageObservation:
    instance: str
    seed: int
    axis: str
    storage_policy_version: str
    semantic_digest: str
    artifact_persistence_seconds: float
    end_to_end_seconds: float
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        if self.storage_policy_version not in {
            "artifact-storage-v1",
            "artifact-storage-v2",
        }:
            raise ValueError("unsupported artifact storage policy")
        if not self.semantic_digest:
            raise ValueError("semantic_digest is required")
        if not math.isfinite(self.artifact_persistence_seconds):
            raise ValueError("artifact_persistence_seconds must be finite")
        if not math.isfinite(self.end_to_end_seconds):
            raise ValueError("end_to_end_seconds must be finite")
        if self.artifact_persistence_seconds < 0.0:
            raise ValueError("artifact_persistence_seconds cannot be negative")
        if self.end_to_end_seconds <= 0.0:
            raise ValueError("end_to_end_seconds must be positive")
        if self.peak_rss_bytes <= 0:
            raise ValueError("peak_rss_bytes must be positive")

    @property
    def identity(self) -> tuple[str, int, str]:
        return self.instance, self.seed, self.axis


@dataclass(frozen=True, slots=True)
class ArtifactStoragePromotionDecision:
    passed: bool
    replay_equality_passed: bool
    persistence_passed: bool
    rss_passed: bool
    replay_detail: str
    persistence_detail: str
    rss_detail: str


@dataclass(frozen=True, slots=True)
class FormalBudgetMatrix:
    small_instances: int = 36
    large_instances: int = 56
    seeds: int = 10
    small_budgets: tuple[int, ...] = (30,)
    large_budgets: tuple[int, ...] = (30, 60, 300)

    @property
    def total_runs(self) -> int:
        return self.seeds * (
            self.small_instances * len(self.small_budgets)
            + self.large_instances * len(self.large_budgets)
        )

    @property
    def declared_solver_seconds(self) -> int:
        return self.seeds * (
            self.small_instances * sum(self.small_budgets)
            + self.large_instances * sum(self.large_budgets)
        )

    def budgets_for_customer_count(self, customer_count: int) -> tuple[int, ...]:
        if customer_count in {5, 10, 15}:
            return self.small_budgets
        if customer_count == 100:
            return self.large_budgets
        raise ValueError(f"unsupported Stage 5.2 customer count: {customer_count}")


def formal_budget_matrix() -> FormalBudgetMatrix:
    return FormalBudgetMatrix()


def evaluate_promotion(
    previous: Sequence[PerformanceObservation],
    candidate: Sequence[PerformanceObservation],
    *,
    minimum_saving: float = 0.15,
    maximum_family_regression: float = 0.03,
) -> PromotionDecision:
    """Evaluate the Stage 5.2 fixed-work paired promotion gate."""

    previous_by_key = _unique_observations(previous, "previous")
    candidate_by_key = _unique_observations(candidate, "candidate")
    if previous_by_key.keys() != candidate_by_key.keys():
        return PromotionDecision(False, 0.0, {}, "paired scope identity mismatch")
    if not previous_by_key:
        return PromotionDecision(False, 0.0, {}, "paired evidence is empty")

    semantic_failures = [
        key
        for key in previous_by_key
        if previous_by_key[key].semantic_digest
        != candidate_by_key[key].semantic_digest
    ]
    if semantic_failures:
        return PromotionDecision(
            False,
            0.0,
            {},
            f"semantic digest mismatch for {len(semantic_failures)} pair(s)",
        )

    savings: list[float] = []
    by_family: dict[str, list[float]] = {}
    for key, baseline in previous_by_key.items():
        promoted = candidate_by_key[key]
        if baseline.customer_count != promoted.customer_count:
            return PromotionDecision(False, 0.0, {}, "paired customer-count mismatch")
        if baseline.customer_count != 100:
            continue
        saving = 1.0 - promoted.end_to_end_seconds / baseline.end_to_end_seconds
        savings.append(saving)
        by_family.setdefault(_instance_family(baseline.instance), []).append(saving)
    if not savings:
        return PromotionDecision(False, 0.0, {}, "100-customer paired evidence is empty")
    aggregate = statistics.median(savings)
    family_medians = {
        family: statistics.median(values) for family, values in sorted(by_family.items())
    }
    required_families = {"C", "R", "RC"}
    if set(family_medians) != required_families:
        return PromotionDecision(
            False,
            aggregate,
            family_medians,
            "100-customer families must be exactly C, R, and RC",
        )
    family_regressions = {
        family: value
        for family, value in family_medians.items()
        if value < -maximum_family_regression
    }
    passed = aggregate + 1e-12 >= minimum_saving and not family_regressions
    detail = (
        "promotion gate passed"
        if passed
        else (
            f"family regression exceeds {maximum_family_regression:.0%}: "
            f"{sorted(family_regressions)}"
            if family_regressions
            else f"aggregate median saving {aggregate:.6f} is below {minimum_saving:.6f}"
        )
    )
    return PromotionDecision(passed, aggregate, family_medians, detail)


def evaluate_artifact_storage_promotion(
    baseline: Sequence[ArtifactStorageObservation],
    predecessor: Sequence[ArtifactStorageObservation],
    candidate: Sequence[ArtifactStorageObservation],
    *,
    maximum_persistence_ratio: float = 0.30,
    maximum_baseline_rss_fraction: float = 0.50,
) -> ArtifactStoragePromotionDecision:
    """Evaluate artifact-storage-v2 replay, persistence, and memory gates."""

    if (
        not math.isfinite(maximum_persistence_ratio)
        or not 0.0 < maximum_persistence_ratio < 1.0
    ):
        raise ValueError("maximum_persistence_ratio must be finite and between 0 and 1")
    if (
        not math.isfinite(maximum_baseline_rss_fraction)
        or not 0.0 < maximum_baseline_rss_fraction <= 1.0
    ):
        raise ValueError(
            "maximum_baseline_rss_fraction must be finite and between 0 and 1"
        )

    baseline_by_key = _unique_storage_observations(baseline, "baseline")
    predecessor_by_key = _unique_storage_observations(predecessor, "predecessor")
    candidate_by_key = _unique_storage_observations(candidate, "candidate")
    identities_match = (
        bool(candidate_by_key)
        and baseline_by_key.keys() == predecessor_by_key.keys()
        and predecessor_by_key.keys() == candidate_by_key.keys()
    )
    policies_match = (
        all(
            item.storage_policy_version == "artifact-storage-v1"
            for item in (*baseline_by_key.values(), *predecessor_by_key.values())
        )
        and all(
            item.storage_policy_version == "artifact-storage-v2"
            for item in candidate_by_key.values()
        )
    )
    fixed_work_keys = [
        key for key in candidate_by_key if key[2].startswith("fixed_work")
    ]
    semantic_match = (
        identities_match
        and bool(fixed_work_keys)
        and all(
            predecessor_by_key[key].semantic_digest
            == candidate_by_key[key].semantic_digest
            for key in fixed_work_keys
        )
    )
    replay_passed = identities_match and policies_match and semantic_match
    if not identities_match:
        replay_detail = "v1/v2 scope identity mismatch"
    elif not policies_match:
        replay_detail = "expected v1 baseline/predecessor and v2 candidate"
    elif not semantic_match:
        replay_detail = "v1/v2 semantic digest mismatch"
    else:
        replay_detail = "v1/v2 semantic replay equality passed"

    persistence_seconds = sum(
        item.artifact_persistence_seconds for item in candidate_by_key.values()
    )
    end_to_end_seconds = sum(
        item.end_to_end_seconds for item in candidate_by_key.values()
    )
    observed_persistence_ratio = (
        persistence_seconds / end_to_end_seconds if end_to_end_seconds else math.inf
    )
    persistence_passed = (
        bool(candidate_by_key)
        and observed_persistence_ratio <= maximum_persistence_ratio
    )
    persistence_detail = (
        f"aggregate persistence ratio {observed_persistence_ratio:.6f} "
        f"<= {maximum_persistence_ratio:.0%}"
        if persistence_passed
        else (
            f"aggregate persistence ratio {observed_persistence_ratio:.6f} "
            f"exceeds {maximum_persistence_ratio:.0%}"
        )
    )

    rss_passed = bool(baseline_by_key) and bool(candidate_by_key)
    baseline_peak = max(
        (item.peak_rss_bytes for item in baseline_by_key.values()), default=0
    )
    candidate_peak = max(
        (item.peak_rss_bytes for item in candidate_by_key.values()), default=0
    )
    rss_limit = baseline_peak * maximum_baseline_rss_fraction
    rss_passed = rss_passed and candidate_peak <= rss_limit
    rss_detail = (
        f"candidate peak RSS {candidate_peak} <= baseline limit {rss_limit:.0f}"
        if rss_passed
        else f"candidate peak RSS {candidate_peak} exceeds baseline limit {rss_limit:.0f}"
    )
    return ArtifactStoragePromotionDecision(
        replay_passed and persistence_passed and rss_passed,
        replay_passed,
        persistence_passed,
        rss_passed,
        replay_detail,
        persistence_detail,
        rss_detail,
    )


def select_worker_count(
    end_to_end_seconds: Mapping[int, float],
    aggregate_rss_gib: Mapping[int, float],
) -> int:
    """Select 2 or 4 shard workers under the documented speed/RSS gates."""

    if set(end_to_end_seconds) != {1, 2, 4} or set(aggregate_rss_gib) != {1, 2, 4}:
        raise ValueError("worker evidence must contain exactly 1, 2, and 4 workers")
    if any(value <= 0.0 for value in end_to_end_seconds.values()):
        raise ValueError("worker end-to-end times must be positive")
    baseline = end_to_end_seconds[1]
    two_passes = baseline / end_to_end_seconds[2] >= 1.5 and aggregate_rss_gib[2] <= 12.0
    if not two_passes:
        raise ValueError("NOT_READY: two workers failed the 1.5x/12 GiB gate")
    four_passes = baseline / end_to_end_seconds[4] >= 2.5 and aggregate_rss_gib[4] <= 12.0
    return 4 if four_passes else 2


def decide_accelerator(
    *,
    median_batch_occupancy: float,
    native_cpu: Sequence[PerformanceObservation] = (),
    accelerator: Sequence[PerformanceObservation] = (),
) -> AcceleratorDecision:
    if median_batch_occupancy < 32.0:
        if native_cpu or accelerator:
            raise ValueError("accelerator evidence is forbidden below occupancy 32")
        return AcceleratorDecision.GPU_NOT_JUSTIFIED
    decision = evaluate_promotion(native_cpu, accelerator)
    return (
        AcceleratorDecision.ACCELERATOR_PROMOTED
        if decision.passed
        else AcceleratorDecision.NATIVE_CPU_RETAINED
    )


def _unique_observations(
    observations: Sequence[PerformanceObservation], label: str
) -> dict[tuple[str, int], PerformanceObservation]:
    result: dict[tuple[str, int], PerformanceObservation] = {}
    for observation in observations:
        if observation.identity in result:
            raise ValueError(f"duplicate {label} observation: {observation.identity}")
        result[observation.identity] = observation
    return result


def _unique_storage_observations(
    observations: Sequence[ArtifactStorageObservation], label: str
) -> dict[tuple[str, int, str], ArtifactStorageObservation]:
    result: dict[tuple[str, int, str], ArtifactStorageObservation] = {}
    for observation in observations:
        if observation.identity in result:
            raise ValueError(
                f"duplicate {label} storage observation: {observation.identity}"
            )
        result[observation.identity] = observation
    return result


def _instance_family(instance: str) -> str:
    normalized = instance.lower()
    if normalized.startswith("rc"):
        return "RC"
    if normalized.startswith("r"):
        return "R"
    if normalized.startswith("c"):
        return "C"
    raise ValueError(f"unknown Schneider instance family: {instance}")
