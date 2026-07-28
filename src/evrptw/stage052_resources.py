"""Capability-based resource contracts for Stage 5.2 campaigns.

Publication identity is deliberately restricted to the hard scientific and
execution contract. Host hardware and power observations remain visible as
telemetry but cannot invalidate an otherwise capable machine.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

from evrptw.stage052_platform import durable_replace, sync_directory

_SHA256_LENGTH: Final = 64
_MEMORY_FRACTION: Final = 0.75
_MINIMUM_PRODUCER_SPEEDUP: Final = 0.15
_THROUGHPUT_TIE_FRACTION: Final = 0.05
_MINIMUM_PERSISTENCE_IMPROVEMENT: Final = 0.10
_PRODUCER_WORKERS: Final = frozenset({4, 5, 6})
_ROW_GROUP_SIZES: Final = frozenset({65_536, 262_144})
_QUEUE_DEPTHS: Final = frozenset({1, 2})


def _is_sha256(value: str) -> bool:
    return len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def verify_filesystem_capabilities(root: Path) -> None:
    """Exercise the fsync and same-directory atomic-replace operations we require."""

    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise RuntimeError(f"filesystem capability root is not a directory: {resolved}")
    try:
        with tempfile.TemporaryDirectory(
            prefix=".stage052-capability-",
            dir=resolved,
        ) as temporary:
            probe = Path(temporary)
            source = probe / "source"
            destination = probe / "destination"
            with source.open("wb") as handle:
                handle.write(b"stage05.2-filesystem-capability\n")
                handle.flush()
                os.fsync(handle.fileno())
            destination.write_bytes(b"replace-me\n")
            durable_replace(source, destination)
            sync_directory(probe)
            if destination.read_bytes() != b"stage05.2-filesystem-capability\n":
                raise RuntimeError("filesystem atomic replace changed probe content")
    except OSError as error:
        raise RuntimeError(
            f"filesystem lacks required fsync/atomic-replace capability: {resolved}"
        ) from error


@dataclass(frozen=True, slots=True)
class CapabilityRequirement:
    """Hard requirements frozen into a Pilot/Formal campaign lock."""

    workers: int
    minimum_memory_bytes: int
    minimum_free_space_bytes: int
    backend: str
    python_abi: str
    native_extension_sha256: str

    def __post_init__(self) -> None:
        _positive_int(self.workers, "workers")
        _positive_int(self.minimum_memory_bytes, "minimum_memory_bytes")
        _positive_int(self.minimum_free_space_bytes, "minimum_free_space_bytes")
        if not self.backend or not self.python_abi:
            raise ValueError("backend and python_abi are required")
        if not _is_sha256(self.native_extension_sha256):
            raise ValueError("native_extension_sha256 is invalid")


@dataclass(frozen=True, slots=True)
class HostCapabilities:
    """Observed machine capabilities used by the hard preflight gate."""

    logical_cpu_count: int
    available_memory_bytes: int
    free_space_bytes: int
    available_backends: frozenset[str]
    python_abi: str
    native_extension_sha256: str
    filesystem_fsync: bool
    filesystem_atomic_replace: bool

    def __post_init__(self) -> None:
        _positive_int(self.logical_cpu_count, "logical_cpu_count")
        _positive_int(self.available_memory_bytes, "available_memory_bytes")
        _positive_int(self.free_space_bytes, "free_space_bytes")
        if not self.available_backends or any(not item for item in self.available_backends):
            raise ValueError("available_backends must be non-empty")
        if not self.python_abi or not _is_sha256(self.native_extension_sha256):
            raise ValueError("host Python/native extension identity is invalid")


def validate_capabilities(
    capabilities: HostCapabilities,
    requirement: CapabilityRequirement,
) -> None:
    """Fail fast when the machine cannot execute the locked campaign."""

    failures: list[str] = []
    if capabilities.logical_cpu_count < requirement.workers:
        failures.append(
            f"logical CPU count {capabilities.logical_cpu_count} is below {requirement.workers}"
        )
    if capabilities.available_memory_bytes < requirement.minimum_memory_bytes:
        failures.append("available memory is below the campaign requirement")
    if capabilities.free_space_bytes < requirement.minimum_free_space_bytes:
        failures.append("free space is below the campaign requirement")
    if requirement.backend not in capabilities.available_backends:
        failures.append(f"backend {requirement.backend!r} is unavailable")
    if capabilities.python_abi != requirement.python_abi:
        failures.append("Python ABI does not match the campaign lock")
    if capabilities.native_extension_sha256 != requirement.native_extension_sha256:
        failures.append("native extension hash does not match the campaign lock")
    if not capabilities.filesystem_fsync:
        failures.append("filesystem does not provide required fsync support")
    if not capabilities.filesystem_atomic_replace:
        failures.append("filesystem does not provide required atomic replace support")
    if failures:
        raise RuntimeError("; ".join(failures))


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Hard publication identity plus non-blocking, separately hashed telemetry."""

    hard_contract: Mapping[str, object]
    telemetry: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.hard_contract:
            raise ValueError("hard_contract must not be empty")
        object.__setattr__(self, "hard_contract", MappingProxyType(dict(self.hard_contract)))
        object.__setattr__(self, "telemetry", MappingProxyType(dict(self.telemetry)))
        # Reject values that would produce platform-dependent or invalid JSON.
        json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @property
    def publication_sha256(self) -> str:
        encoded = json.dumps(
            dict(self.hard_contract),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def telemetry_sha256(self) -> str:
        encoded = json.dumps(
            dict(self.telemetry),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "stage05.2-runtime-identity-v2",
            "hard_contract": dict(self.hard_contract),
            "publication_sha256": self.publication_sha256,
            "telemetry": dict(self.telemetry),
            "telemetry_sha256": self.telemetry_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProducerBenchmark:
    """One real-shard producer calibration observation."""

    workers: int
    throughput: float
    aggregate_peak_rss_bytes: int
    semantic_digest: str
    swap_peak_bytes: int
    fallback_count: int
    resource_limit_exceeded: bool

    def __post_init__(self) -> None:
        if self.workers not in _PRODUCER_WORKERS:
            raise ValueError("producer workers must be one of 4, 5, or 6")
        if not math.isfinite(self.throughput) or self.throughput <= 0.0:
            raise ValueError("producer throughput must be finite and positive")
        _positive_int(self.aggregate_peak_rss_bytes, "aggregate_peak_rss_bytes")
        if not _is_sha256(self.semantic_digest):
            raise ValueError("producer semantic_digest is invalid")
        if self.swap_peak_bytes < 0 or self.fallback_count < 0:
            raise ValueError("producer swap/fallback counters cannot be negative")


@dataclass(frozen=True, slots=True)
class ProducerSelection:
    """Deterministic producer selection frozen into the campaign lock."""

    selected_workers: int
    semantic_digest: str
    candidate_workers: tuple[int, ...]
    rejected_reasons: Mapping[int, str]

    def __post_init__(self) -> None:
        if self.selected_workers not in self.candidate_workers:
            raise ValueError("selected_workers is not a calibrated candidate")
        object.__setattr__(
            self,
            "rejected_reasons",
            MappingProxyType(dict(self.rejected_reasons)),
        )


@dataclass(frozen=True, slots=True)
class ProducerResourceContract:
    """Calibration-derived producer limits frozen into Pilot and Formal."""

    selected_workers: int
    available_memory_bytes: int
    selected_aggregate_peak_rss_bytes: int
    selected_per_worker_peak_rss_bytes: int
    aggregate_memory_limit_bytes: int
    per_worker_memory_limit_bytes: int
    semantic_digest: str
    calibration_digest: str
    row_group_size: int
    queue_depth: int

    def __post_init__(self) -> None:
        if self.selected_workers not in _PRODUCER_WORKERS:
            raise ValueError("selected_workers must be one of 4, 5, or 6")
        for field_name in (
            "available_memory_bytes",
            "selected_aggregate_peak_rss_bytes",
            "selected_per_worker_peak_rss_bytes",
            "aggregate_memory_limit_bytes",
            "per_worker_memory_limit_bytes",
        ):
            _positive_int(getattr(self, field_name), field_name)
        if not _is_sha256(self.semantic_digest) or not _is_sha256(
            self.calibration_digest
        ):
            raise ValueError("producer semantic/calibration digest is invalid")
        if self.row_group_size not in _ROW_GROUP_SIZES:
            raise ValueError("row_group_size must be 65,536 or 262,144")
        if self.queue_depth not in _QUEUE_DEPTHS:
            raise ValueError("queue_depth must be 1 or 2")
        if self.aggregate_memory_limit_bytes < self.selected_aggregate_peak_rss_bytes:
            raise ValueError("aggregate memory limit is below the calibrated peak")
        if self.per_worker_memory_limit_bytes < self.selected_per_worker_peak_rss_bytes:
            raise ValueError("per-worker memory limit is below the calibrated peak")
        if self.aggregate_memory_limit_bytes > math.floor(
            self.available_memory_bytes * _MEMORY_FRACTION
        ):
            raise ValueError("producer aggregate limit exceeds the 75% capability envelope")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": "stage05.2-producer-resource-contract-v1",
            "selected_workers": self.selected_workers,
            "available_memory_bytes": self.available_memory_bytes,
            "selected_aggregate_peak_rss_bytes": self.selected_aggregate_peak_rss_bytes,
            "selected_per_worker_peak_rss_bytes": self.selected_per_worker_peak_rss_bytes,
            "aggregate_memory_limit_bytes": self.aggregate_memory_limit_bytes,
            "per_worker_memory_limit_bytes": self.per_worker_memory_limit_bytes,
            "semantic_digest": self.semantic_digest,
            "calibration_digest": self.calibration_digest,
            "row_group_size": self.row_group_size,
            "queue_depth": self.queue_depth,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ProducerResourceContract:
        expected = {
            "schema_version",
            "selected_workers",
            "available_memory_bytes",
            "selected_aggregate_peak_rss_bytes",
            "selected_per_worker_peak_rss_bytes",
            "aggregate_memory_limit_bytes",
            "per_worker_memory_limit_bytes",
            "semantic_digest",
            "calibration_digest",
            "row_group_size",
            "queue_depth",
        }
        if set(payload) != expected:
            raise ValueError("producer resource contract fields do not match the schema")
        if payload.get("schema_version") != "stage05.2-producer-resource-contract-v1":
            raise ValueError("producer resource contract schema is unsupported")

        def integer(field_name: str) -> int:
            value = payload.get(field_name)
            _positive_int(value, field_name)  # type: ignore[arg-type]
            return value  # type: ignore[return-value]

        semantic_digest = payload.get("semantic_digest")
        calibration_digest = payload.get("calibration_digest")
        if not isinstance(semantic_digest, str) or not isinstance(
            calibration_digest, str
        ):
            raise ValueError("producer resource contract digests must be strings")
        return cls(
            selected_workers=integer("selected_workers"),
            available_memory_bytes=integer("available_memory_bytes"),
            selected_aggregate_peak_rss_bytes=integer(
                "selected_aggregate_peak_rss_bytes"
            ),
            selected_per_worker_peak_rss_bytes=integer(
                "selected_per_worker_peak_rss_bytes"
            ),
            aggregate_memory_limit_bytes=integer("aggregate_memory_limit_bytes"),
            per_worker_memory_limit_bytes=integer("per_worker_memory_limit_bytes"),
            semantic_digest=semantic_digest,
            calibration_digest=calibration_digest,
            row_group_size=integer("row_group_size"),
            queue_depth=integer("queue_depth"),
        )


def derive_producer_resource_contract(
    *,
    results: Sequence[ProducerBenchmark],
    selection: ProducerSelection,
    selected_per_worker_peak_rss_bytes: int,
    available_memory_bytes: int,
    row_group_size: int,
    queue_depth: int,
) -> ProducerResourceContract:
    """Freeze selected producer peaks with 20% headroom under the 75% ceiling."""

    _positive_int(selected_per_worker_peak_rss_bytes, "selected_per_worker_peak_rss_bytes")
    _positive_int(available_memory_bytes, "available_memory_bytes")
    by_workers = {item.workers: item for item in results}
    if set(by_workers) != _PRODUCER_WORKERS or len(results) != len(_PRODUCER_WORKERS):
        raise ValueError("producer resource derivation requires exactly 4/5/6 results")
    selected = by_workers.get(selection.selected_workers)
    if selected is None or selected.semantic_digest != selection.semantic_digest:
        raise ValueError("producer selection does not match the calibration observations")
    aggregate_limit = math.ceil(selected.aggregate_peak_rss_bytes * 1.20)
    per_worker_limit = math.ceil(selected_per_worker_peak_rss_bytes * 1.20)
    if aggregate_limit > math.floor(available_memory_bytes * _MEMORY_FRACTION):
        raise RuntimeError(
            "calibration-derived producer memory requirement exceeds 75% of available memory"
        )
    calibration_payload = {
        "available_memory_bytes": available_memory_bytes,
        "results": [
            {
                "workers": item.workers,
                "throughput": item.throughput,
                "aggregate_peak_rss_bytes": item.aggregate_peak_rss_bytes,
                "semantic_digest": item.semantic_digest,
                "swap_peak_bytes": item.swap_peak_bytes,
                "fallback_count": item.fallback_count,
                "resource_limit_exceeded": item.resource_limit_exceeded,
            }
            for item in sorted(results, key=lambda item: item.workers)
        ],
        "selected_workers": selection.selected_workers,
        "selected_per_worker_peak_rss_bytes": selected_per_worker_peak_rss_bytes,
        "row_group_size": row_group_size,
        "queue_depth": queue_depth,
    }
    calibration_digest = hashlib.sha256(
        json.dumps(
            calibration_payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return ProducerResourceContract(
        selected_workers=selection.selected_workers,
        available_memory_bytes=available_memory_bytes,
        selected_aggregate_peak_rss_bytes=selected.aggregate_peak_rss_bytes,
        selected_per_worker_peak_rss_bytes=selected_per_worker_peak_rss_bytes,
        aggregate_memory_limit_bytes=aggregate_limit,
        per_worker_memory_limit_bytes=per_worker_limit,
        semantic_digest=selection.semantic_digest,
        calibration_digest=calibration_digest,
        row_group_size=row_group_size,
        queue_depth=queue_depth,
    )


def load_producer_resource_contract(path: Path) -> ProducerResourceContract:
    """Load one immutable calibration contract and verify its SHA-256 sidecar."""

    sidecar = path.with_suffix(".sha256")
    try:
        raw = path.read_bytes()
        declared = sidecar.read_text(encoding="utf-8").strip()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"producer calibration contract is unreadable: {path}") from error
    observed = hashlib.sha256(raw).hexdigest()
    if declared != observed:
        raise RuntimeError("producer calibration contract checksum mismatch")
    if not isinstance(payload, Mapping):
        raise RuntimeError("producer calibration contract must be a JSON object")
    try:
        return ProducerResourceContract.from_dict(payload)
    except ValueError as error:
        raise RuntimeError("producer calibration contract is invalid") from error


def select_producer_configuration(
    results: Sequence[ProducerBenchmark],
    *,
    available_memory_bytes: int,
) -> ProducerSelection:
    """Apply the locked 4/5/6-worker calibration rules."""

    _positive_int(available_memory_bytes, "available_memory_bytes")
    by_workers = {result.workers: result for result in results}
    if set(by_workers) != _PRODUCER_WORKERS or len(results) != len(_PRODUCER_WORKERS):
        raise ValueError("producer calibration requires exactly one 4/5/6-worker result")
    baseline = by_workers[4]
    rejected: dict[int, str] = {}
    eligible: list[ProducerBenchmark] = []
    memory_budget = available_memory_bytes * _MEMORY_FRACTION
    for workers in sorted(by_workers):
        result = by_workers[workers]
        reasons: list[str] = []
        if result.aggregate_peak_rss_bytes > memory_budget or result.resource_limit_exceeded:
            reasons.append("memory budget or resource limit exceeded")
        if result.swap_peak_bytes > 0:
            reasons.append("swap pressure observed")
        if result.fallback_count > 0:
            reasons.append("fallback observed")
        if result.semantic_digest != baseline.semantic_digest:
            reasons.append("semantic digest differs from 4-worker baseline")
        if (
            workers > 4
            and result.throughput
            < baseline.throughput * (1.0 + _MINIMUM_PRODUCER_SPEEDUP)
        ):
            reasons.append("15% throughput improvement gate not met")
        if reasons:
            rejected[workers] = "; ".join(reasons)
        else:
            eligible.append(result)
    if not eligible:
        raise RuntimeError("no producer calibration candidate satisfies the resource contract")
    fastest = max(result.throughput for result in eligible)
    tied = [
        result
        for result in eligible
        if result.throughput >= fastest * (1.0 - _THROUGHPUT_TIE_FRACTION)
    ]
    selected = min(tied, key=lambda result: result.workers)
    return ProducerSelection(
        selected_workers=selected.workers,
        semantic_digest=baseline.semantic_digest,
        candidate_workers=tuple(sorted(by_workers)),
        rejected_reasons=rejected,
    )


@dataclass(frozen=True, slots=True)
class ParquetBenchmark:
    """One bounded persistence configuration observation."""

    row_group_size: int
    queue_depth: int
    persistence_seconds: float
    aggregate_peak_rss_bytes: int
    semantic_digest: str

    def __post_init__(self) -> None:
        if self.row_group_size not in _ROW_GROUP_SIZES:
            raise ValueError("row_group_size must be 65,536 or 262,144")
        if self.queue_depth not in _QUEUE_DEPTHS:
            raise ValueError("queue_depth must be 1 or 2")
        if not math.isfinite(self.persistence_seconds) or self.persistence_seconds <= 0.0:
            raise ValueError("persistence_seconds must be finite and positive")
        _positive_int(self.aggregate_peak_rss_bytes, "aggregate_peak_rss_bytes")
        if not _is_sha256(self.semantic_digest):
            raise ValueError("persistence semantic_digest is invalid")


def select_parquet_configuration(
    results: Sequence[ParquetBenchmark],
    *,
    available_memory_bytes: int,
) -> ParquetBenchmark:
    """Select a changed Parquet setting only after a >=10% critical-path win."""

    _positive_int(available_memory_bytes, "available_memory_bytes")
    identities = {(item.row_group_size, item.queue_depth) for item in results}
    if len(identities) != len(results):
        raise ValueError("persistence calibration configurations must be unique")
    baseline = next(
        (
            result
            for result in results
            if result.row_group_size == 65_536 and result.queue_depth == 1
        ),
        None,
    )
    if baseline is None:
        raise ValueError("persistence calibration requires the 65,536/1 baseline")
    memory_budget = available_memory_bytes * _MEMORY_FRACTION
    eligible = [
        result
        for result in results
        if result.semantic_digest == baseline.semantic_digest
        and result.aggregate_peak_rss_bytes <= memory_budget
        and result.persistence_seconds
        <= baseline.persistence_seconds * (1.0 - _MINIMUM_PERSISTENCE_IMPROVEMENT)
    ]
    if not eligible:
        return baseline
    return min(
        eligible,
        key=lambda result: (
            result.persistence_seconds,
            result.row_group_size,
            result.queue_depth,
        ),
    )


@dataclass(frozen=True, slots=True)
class ReviewMemoryContract:
    """Pilot-derived reviewer process-tree and cgroup limits."""

    parent_baseline_rss_bytes: int
    per_child_p99_rss_bytes: int
    review_workers: int
    available_memory_bytes: int
    memory_high_bytes: int
    process_guard_bytes: int
    memory_max_bytes: int
    memory_swap_max_bytes: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "parent_baseline_rss_bytes",
            "per_child_p99_rss_bytes",
            "review_workers",
            "available_memory_bytes",
            "memory_high_bytes",
            "process_guard_bytes",
            "memory_max_bytes",
        ):
            _positive_int(getattr(self, field_name), field_name)
        if self.review_workers not in {1, 2, 4}:
            raise ValueError("review_workers must be one of 1, 2, or 4")
        if self.memory_swap_max_bytes != 0:
            raise ValueError("reviewer swap must be disabled")
        if not (
            self.memory_high_bytes
            < self.process_guard_bytes
            < self.memory_max_bytes
            <= int(self.available_memory_bytes * _MEMORY_FRACTION)
        ):
            raise ValueError("review memory limits exceed the 75% capability envelope")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": "stage05.2-review-memory-contract-v1",
            "parent_baseline_rss_bytes": self.parent_baseline_rss_bytes,
            "per_child_p99_rss_bytes": self.per_child_p99_rss_bytes,
            "review_workers": self.review_workers,
            "available_memory_bytes": self.available_memory_bytes,
            "memory_high_bytes": self.memory_high_bytes,
            "process_guard_bytes": self.process_guard_bytes,
            "memory_max_bytes": self.memory_max_bytes,
            "memory_swap_max_bytes": self.memory_swap_max_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ReviewMemoryContract:
        expected = {
            "schema_version",
            "parent_baseline_rss_bytes",
            "per_child_p99_rss_bytes",
            "review_workers",
            "available_memory_bytes",
            "memory_high_bytes",
            "process_guard_bytes",
            "memory_max_bytes",
            "memory_swap_max_bytes",
        }
        if set(payload) != expected:
            raise ValueError("review memory contract fields do not match the schema")
        if payload.get("schema_version") != "stage05.2-review-memory-contract-v1":
            raise ValueError("review memory contract schema is unsupported")

        def integer(field_name: str, *, allow_zero: bool = False) -> int:
            value = payload.get(field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or (value == 0 and not allow_zero)
            ):
                raise ValueError(f"{field_name} is invalid")
            return value

        return cls(
            parent_baseline_rss_bytes=integer("parent_baseline_rss_bytes"),
            per_child_p99_rss_bytes=integer("per_child_p99_rss_bytes"),
            review_workers=integer("review_workers"),
            available_memory_bytes=integer("available_memory_bytes"),
            memory_high_bytes=integer("memory_high_bytes"),
            process_guard_bytes=integer("process_guard_bytes"),
            memory_max_bytes=integer("memory_max_bytes"),
            memory_swap_max_bytes=integer(
                "memory_swap_max_bytes",
                allow_zero=True,
            ),
        )


def derive_review_memory_contract(
    *,
    parent_baseline_rss_bytes: int,
    per_child_p99_rss_bytes: int,
    review_workers: int,
    available_memory_bytes: int,
) -> ReviewMemoryContract:
    """Derive frozen limits with 20% headroom and a hard 75% host ceiling."""

    for field_name, value in (
        ("parent_baseline_rss_bytes", parent_baseline_rss_bytes),
        ("per_child_p99_rss_bytes", per_child_p99_rss_bytes),
        ("available_memory_bytes", available_memory_bytes),
    ):
        _positive_int(value, field_name)
    if review_workers not in {1, 2, 4}:
        raise ValueError("review_workers must be one of 1, 2, or 4")
    measured = parent_baseline_rss_bytes + per_child_p99_rss_bytes * review_workers
    memory_max = math.ceil(measured * 1.20)
    capability_ceiling = math.floor(available_memory_bytes * _MEMORY_FRACTION)
    if memory_max > capability_ceiling:
        raise RuntimeError(
            "Pilot-derived reviewer memory requirement exceeds 75% of available memory"
        )
    return ReviewMemoryContract(
        parent_baseline_rss_bytes=parent_baseline_rss_bytes,
        per_child_p99_rss_bytes=per_child_p99_rss_bytes,
        review_workers=review_workers,
        available_memory_bytes=available_memory_bytes,
        memory_high_bytes=math.floor(memory_max * 0.85),
        process_guard_bytes=math.floor(memory_max * 0.95),
        memory_max_bytes=memory_max,
    )


def load_review_memory_contract(path: Path) -> ReviewMemoryContract:
    """Load one signed Pilot-derived reviewer memory contract."""

    sidecar = path.with_suffix(".sha256")
    try:
        raw = path.read_bytes()
        declared = sidecar.read_text(encoding="utf-8").strip()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"review memory contract is unreadable: {path}") from error
    if hashlib.sha256(raw).hexdigest() != declared:
        raise RuntimeError("review memory contract checksum mismatch")
    if not isinstance(payload, Mapping):
        raise RuntimeError("review memory contract must be a JSON object")
    try:
        return ReviewMemoryContract.from_dict(payload)
    except ValueError as error:
        raise RuntimeError("review memory contract is invalid") from error
