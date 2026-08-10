"""Host-aware performance contracts for the Stage 5.2 native architecture."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform as _platform
import re
import statistics
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final, cast

from evrptw.stage052_atomic import publish_no_replace

PERFORMANCE_PROFILE_SCHEMA_VERSION: Final = "stage05.2-performance-profile-v1"
RUNTIME_RESOURCE_SCHEMA_VERSION: Final = "stage05.2-runtime-resource-v2"
TELEMETRY_OVERHEAD_SCHEMA_VERSION: Final = "stage05.2-telemetry-overhead-v2"
_BUILD_NAMES: Final = frozenset({"portable-o3", "portable-lto", "host-native-lto"})
STAGE052_PERFORMANCE_MODES: Final = (
    "current_stage052",
    "python_candidate_control",
    "per_solve_runtime",
    "full_native_alns",
    "host_scheduler",
)
STAGE052_WORKLOAD_CLASSES: Final = ("c5", "100-customer")
_MODE_NAMES: Final = frozenset(STAGE052_PERFORMANCE_MODES)
_TOPOLOGY_POLICIES: Final = frozenset(
    {"uniform", "physical_core_first", "free_scheduler", "scheduler_partition"}
)
_GIT_SHA1_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_MAX_NATIVE_REQUEST_THREADS: Final = 64


def require_clean_repository_root(path: Path) -> Path:
    """Resolve and verify the explicit clean Git root used by calibration CLIs."""

    root = path.resolve(strict=True)
    try:
        top_level = subprocess.run(
            ("git", "rev-parse", "--show-toplevel"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Stage 5.2 repository root is not a readable Git worktree") from error
    if not top_level or Path(top_level).resolve(strict=True) != root:
        raise RuntimeError("Stage 5.2 repository root must name the Git worktree root")
    if status:
        raise RuntimeError("Stage 5.2 calibration requires one clean Git worktree")
    return root


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, name: str) -> int:
    if not _plain_int(value) or cast(int, value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return cast(int, value)


def _nonnegative_int(value: object, name: str) -> int:
    if not _plain_int(value) or cast(int, value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return cast(int, value)


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _cpu_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be an array")
    result = tuple(cast(int, item) for item in value)
    if any(not _plain_int(item) or item < 0 for item in result):
        raise ValueError(f"{name} must contain non-negative integer CPU IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate CPU IDs")
    return tuple(sorted(result))


def _ordered_cpu_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be an array")
    result = tuple(cast(int, item) for item in value)
    if any(not _plain_int(item) or item < 0 for item in result):
        raise ValueError(f"{name} must contain non-negative integer CPU IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate CPU IDs")
    return result


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be an array of strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} must contain non-empty strings")
    return tuple(sorted(set(cast(Sequence[str], value))))


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    raw = cast(Mapping[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{name} keys must be strings")
    result = {cast(str, key): item for key, item in raw.items()}
    thawed = _thaw_json(result)
    if not isinstance(thawed, dict):  # pragma: no cover - result is a dict
        raise AssertionError("JSON thaw lost its object shape")
    json.dumps(thawed, allow_nan=False, sort_keys=True)
    return cast(dict[str, object], thawed)


def _freeze_json(value: object, name: str) -> object:
    """Return a recursively immutable copy of one canonical-JSON value."""

    if value is None or isinstance(value, str | bool):
        return value
    if _plain_int(value):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        raw = cast(Mapping[object, object], value)
        if any(not isinstance(key, str) for key in raw):
            raise ValueError(f"{name} keys must be strings")
        return MappingProxyType(
            {cast(str, key): _freeze_json(item, f"{name}.{key}") for key, item in raw.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item, f"{name}[{index}]") for index, item in enumerate(value))
    raise ValueError(f"{name} must contain only canonical JSON values")


def _freeze_mapping(value: object, name: str) -> Mapping[str, object]:
    frozen = _freeze_json(_mapping(value, name), name)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by _mapping
        raise AssertionError("frozen mapping lost its object shape")
    return cast(Mapping[str, object], frozen)


def _thaw_json(value: object) -> object:
    """Return a detached mutable JSON representation of a frozen value."""

    if isinstance(value, Mapping):
        raw = cast(Mapping[object, object], value)
        return {cast(str, key): _thaw_json(item) for key, item in raw.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw_json(item) for item in value]
    return value


def _exact_fields(payload: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(
            f"{name} fields mismatch "
            f"(missing={sorted(expected - set(payload))}, extra={sorted(set(payload) - expected)})"
        )


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            _thaw_json(payload), allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("payload is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class HostPerformanceEnvelope:
    platform_name: str = "linux"
    architecture: str = "x86_64"
    allowed_cpu_ids: tuple[int, ...] = ()
    physical_core_groups: tuple[tuple[int, ...], ...] = ()
    memory_total_bytes: int = 1
    memory_available_bytes: int = 1
    memory_limit_bytes: int | None = None
    memory_current_bytes: int | None = None
    swap_total_bytes: int = 0
    swap_used_bytes: int = 0
    swap_current_bytes: int | None = None
    swap_limit_bytes: int | None = None
    cpu_features: tuple[str, ...] = ()
    compiler: str = ""
    topology_source: str = "unknown"
    telemetry: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.platform_name, str) or not isinstance(self.architecture, str):
            raise ValueError("platform_name and architecture must be strings")
        platform_name = self.platform_name.casefold().strip()
        architecture = self.architecture.casefold().strip()
        if platform_name != "linux":
            raise ValueError("performance profiles require Linux/WSL")
        if architecture not in {"x86_64", "amd64"}:
            raise ValueError("performance profiles require x86-64")
        allowed = _cpu_tuple(self.allowed_cpu_ids, "allowed_cpu_ids")
        if not allowed:
            raise ValueError("allowed_cpu_ids must not be empty")
        if self.topology_source not in {"sysfs", "provided", "unavailable", "unknown"}:
            raise ValueError("unsupported topology_source")
        groups: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for index, group in enumerate(self.physical_core_groups):
            normalized = _cpu_tuple(group, f"physical_core_groups[{index}]")
            if not normalized:
                raise ValueError("physical core groups must not be empty")
            if any(cpu not in allowed for cpu in normalized):
                raise ValueError("physical core group contains an unavailable CPU")
            if seen.intersection(normalized):
                raise ValueError("physical core groups overlap")
            seen.update(normalized)
            groups.append(normalized)
        object.__setattr__(self, "platform_name", platform_name)
        object.__setattr__(self, "architecture", "x86_64")
        object.__setattr__(self, "allowed_cpu_ids", allowed)
        object.__setattr__(self, "physical_core_groups", tuple(sorted(groups)))
        object.__setattr__(self, "cpu_features", _string_tuple(self.cpu_features, "cpu_features"))
        for name, value in (
            ("memory_total_bytes", self.memory_total_bytes),
            ("memory_available_bytes", self.memory_available_bytes),
            ("swap_total_bytes", self.swap_total_bytes),
            ("swap_used_bytes", self.swap_used_bytes),
        ):
            _nonnegative_int(value, name)
        if self.memory_total_bytes <= 0 or self.memory_available_bytes <= 0:
            raise ValueError("memory_total_bytes and memory_available_bytes must be positive")
        if self.memory_available_bytes > self.memory_total_bytes:
            raise ValueError("memory_available_bytes cannot exceed memory_total_bytes")
        if self.swap_used_bytes > self.swap_total_bytes:
            raise ValueError("swap_used_bytes cannot exceed swap_total_bytes")
        if self.swap_current_bytes is not None:
            _nonnegative_int(self.swap_current_bytes, "swap_current_bytes")
        for field_name, field_value in (
            ("memory_limit_bytes", self.memory_limit_bytes),
            ("memory_current_bytes", self.memory_current_bytes),
            ("swap_limit_bytes", self.swap_limit_bytes),
        ):
            if field_value is not None:
                _nonnegative_int(field_value, field_name)
        if (
            self.memory_limit_bytes is not None
            and self.memory_current_bytes is not None
            and self.memory_current_bytes > self.memory_limit_bytes
        ):
            raise ValueError("memory_current_bytes cannot exceed memory_limit_bytes")
        if self.swap_limit_bytes is not None and self.swap_used_bytes > self.swap_limit_bytes:
            raise ValueError("swap_used_bytes cannot exceed swap_limit_bytes")
        if (
            self.swap_limit_bytes is not None
            and self.swap_current_bytes is not None
            and self.swap_current_bytes > self.swap_limit_bytes
        ):
            raise ValueError("swap_current_bytes cannot exceed swap_limit_bytes")
        object.__setattr__(self, "telemetry", _freeze_mapping(self.telemetry, "telemetry"))

    @property
    def logical_cpu_count(self) -> int:
        return len(self.allowed_cpu_ids)

    @property
    def logical_cpu_ids(self) -> tuple[int, ...]:
        return self.allowed_cpu_ids

    @property
    def allowed_cpus(self) -> tuple[int, ...]:
        return self.allowed_cpu_ids

    @property
    def physical_core_count(self) -> int | None:
        return len(self.physical_core_groups) or None

    @property
    def effective_memory_limit_bytes(self) -> int:
        if self.memory_limit_bytes is None:
            return self.memory_available_bytes
        if self.memory_current_bytes is None:
            return min(self.memory_available_bytes, self.memory_limit_bytes)
        return min(
            self.memory_available_bytes,
            max(0, self.memory_limit_bytes - self.memory_current_bytes),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PERFORMANCE_PROFILE_SCHEMA_VERSION,
            "platform_name": self.platform_name,
            "architecture": self.architecture,
            "allowed_cpu_ids": list(self.allowed_cpu_ids),
            "physical_core_groups": [list(group) for group in self.physical_core_groups],
            "memory_total_bytes": self.memory_total_bytes,
            "memory_available_bytes": self.memory_available_bytes,
            "memory_limit_bytes": self.memory_limit_bytes,
            "memory_current_bytes": self.memory_current_bytes,
            "swap_total_bytes": self.swap_total_bytes,
            "swap_used_bytes": self.swap_used_bytes,
            "swap_current_bytes": self.swap_current_bytes,
            "swap_limit_bytes": self.swap_limit_bytes,
            "cpu_features": list(self.cpu_features),
            "compiler": self.compiler,
            "topology_source": self.topology_source,
            "telemetry": _thaw_json(self.telemetry),
        }

    def identity_dict(self) -> dict[str, object]:
        """Return stable, behavior-affecting host profile identity only."""

        return {
            "schema_version": PERFORMANCE_PROFILE_SCHEMA_VERSION,
            "platform_name": self.platform_name,
            "architecture": self.architecture,
            "allowed_cpu_ids": list(self.allowed_cpu_ids),
            "physical_core_groups": [list(group) for group in self.physical_core_groups],
            "memory_total_bytes": self.memory_total_bytes,
            "memory_limit_bytes": self.memory_limit_bytes,
            "swap_limit_bytes": self.swap_limit_bytes,
            "cpu_features": list(self.cpu_features),
            "compiler": self.compiler,
            "topology_source": self.topology_source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HostPerformanceEnvelope:
        expected = {
            "schema_version",
            "platform_name",
            "architecture",
            "allowed_cpu_ids",
            "physical_core_groups",
            "memory_total_bytes",
            "memory_available_bytes",
            "memory_limit_bytes",
            "memory_current_bytes",
            "swap_total_bytes",
            "swap_used_bytes",
            "swap_limit_bytes",
            "cpu_features",
            "compiler",
            "swap_current_bytes",
            "topology_source",
            "telemetry",
        }
        _exact_fields(payload, expected, "host")
        if payload["schema_version"] != PERFORMANCE_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported HostPerformanceEnvelope schema")
        groups = payload["physical_core_groups"]
        if not isinstance(groups, list):
            raise ValueError("physical_core_groups must be an array")
        for field_name in ("platform_name", "architecture", "compiler", "topology_source"):
            if not isinstance(payload[field_name], str):
                raise ValueError(f"{field_name} must be a string")

        def optional_int(field_name: str) -> int | None:
            value = payload[field_name]
            return None if value is None else _nonnegative_int(value, field_name)

        platform_name = cast(str, payload["platform_name"])
        architecture = cast(str, payload["architecture"])
        compiler = cast(str, payload["compiler"])
        topology_source = cast(str, payload["topology_source"])
        return cls(
            platform_name=platform_name,
            architecture=architecture,
            allowed_cpu_ids=_cpu_tuple(payload["allowed_cpu_ids"], "allowed_cpu_ids"),
            physical_core_groups=tuple(
                _cpu_tuple(item, f"physical_core_groups[{index}]")
                for index, item in enumerate(groups)
            ),
            memory_total_bytes=_positive_int(payload["memory_total_bytes"], "memory_total_bytes"),
            memory_available_bytes=_positive_int(
                payload["memory_available_bytes"], "memory_available_bytes"
            ),
            memory_limit_bytes=optional_int("memory_limit_bytes"),
            memory_current_bytes=optional_int("memory_current_bytes"),
            swap_total_bytes=_nonnegative_int(payload["swap_total_bytes"], "swap_total_bytes"),
            swap_used_bytes=_nonnegative_int(payload["swap_used_bytes"], "swap_used_bytes"),
            swap_current_bytes=optional_int("swap_current_bytes"),
            swap_limit_bytes=optional_int("swap_limit_bytes"),
            cpu_features=_string_tuple(payload["cpu_features"], "cpu_features"),
            compiler=compiler,
            topology_source=topology_source,
            telemetry=_mapping(payload["telemetry"], "telemetry"),
        )


@dataclass(frozen=True, slots=True)
class ExecutionTopology:
    workload_class: str = "general"
    shards: tuple[tuple[int, ...], ...] = ()
    worker_count: int = 0
    scheduler_cpu_ids: tuple[int, ...] = ()
    request_threads: int = 0
    affinity_policy: str = "uniform"
    allow_affinity_overlap: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.workload_class, str) or not self.workload_class:
            raise ValueError("workload_class must be non-empty")
        if (
            not isinstance(self.affinity_policy, str)
            or self.affinity_policy not in _TOPOLOGY_POLICIES
        ):
            raise ValueError("unsupported affinity_policy")
        if not isinstance(self.allow_affinity_overlap, bool):
            raise ValueError("allow_affinity_overlap must be boolean")
        shards: list[tuple[int, ...]] = []
        seen: set[int] = set()
        for index, shard in enumerate(self.shards):
            cpus = _ordered_cpu_tuple(shard, f"shards[{index}]")
            if not cpus:
                raise ValueError("shards must not be empty")
            if seen.intersection(cpus):
                raise ValueError("shards overlap; CPU partitions must be disjoint")
            seen.update(cpus)
            shards.append(cpus)
        if not shards:
            raise ValueError("at least one shard is required")
        worker_count = self.worker_count or len(shards)
        if worker_count != len(shards):
            raise ValueError("worker_count must equal shard count")
        scheduler = _cpu_tuple(self.scheduler_cpu_ids, "scheduler_cpu_ids")
        if not self.allow_affinity_overlap and seen.intersection(scheduler):
            raise ValueError("scheduler CPUs overlap compute shards")
        request_threads = self.request_threads or max(1, worker_count)
        _positive_int(request_threads, "request_threads")
        if request_threads > len(seen | set(scheduler)):
            raise ValueError("request_threads exceeds CPU budget")
        object.__setattr__(self, "shards", tuple(shards))
        object.__setattr__(self, "worker_count", worker_count)
        object.__setattr__(self, "scheduler_cpu_ids", scheduler)
        object.__setattr__(self, "request_threads", request_threads)

    @property
    def shard_count(self) -> int:
        return len(self.shards)

    @property
    def compute_cpu_ids(self) -> tuple[int, ...]:
        return tuple(sorted(cpu for shard in self.shards for cpu in shard))

    @property
    def cpu_ids(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.compute_cpu_ids).union(self.scheduler_cpu_ids)))

    @property
    def scheduler_cpus(self) -> tuple[int, ...]:
        return self.scheduler_cpu_ids

    @property
    def shard_cpu_ids(self) -> tuple[tuple[int, ...], ...]:
        return self.shards

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PERFORMANCE_PROFILE_SCHEMA_VERSION,
            "workload_class": self.workload_class,
            "shards": [list(shard) for shard in self.shards],
            "worker_count": self.worker_count,
            "scheduler_cpu_ids": list(self.scheduler_cpu_ids),
            "request_threads": self.request_threads,
            "affinity_policy": self.affinity_policy,
            "allow_affinity_overlap": self.allow_affinity_overlap,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExecutionTopology:
        expected = {
            "schema_version",
            "workload_class",
            "shards",
            "worker_count",
            "scheduler_cpu_ids",
            "request_threads",
            "affinity_policy",
            "allow_affinity_overlap",
        }
        _exact_fields(payload, expected, "topology")
        if payload["schema_version"] != PERFORMANCE_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported ExecutionTopology schema")
        shards = payload["shards"]
        if not isinstance(shards, list):
            raise ValueError("shards must be an array")
        if not isinstance(payload["workload_class"], str):
            raise ValueError("workload_class must be a string")
        if not isinstance(payload["affinity_policy"], str):
            raise ValueError("affinity_policy must be a string")
        if not isinstance(payload["allow_affinity_overlap"], bool):
            raise ValueError("allow_affinity_overlap must be boolean")
        return cls(
            workload_class=payload["workload_class"],
            shards=tuple(
                _ordered_cpu_tuple(item, f"shards[{index}]") for index, item in enumerate(shards)
            ),
            worker_count=_positive_int(payload["worker_count"], "worker_count"),
            scheduler_cpu_ids=_cpu_tuple(payload["scheduler_cpu_ids"], "scheduler_cpu_ids"),
            request_threads=_positive_int(payload["request_threads"], "request_threads"),
            affinity_policy=payload["affinity_policy"],
            allow_affinity_overlap=payload["allow_affinity_overlap"],
        )


def partition_cpus(cpu_ids: Sequence[int], shard_count: int) -> tuple[tuple[int, ...], ...]:
    cpus = _ordered_cpu_tuple(tuple(cpu_ids), "cpu_ids")
    if not cpus or shard_count <= 0 or shard_count > len(cpus):
        raise ValueError("invalid CPU partition count")
    base, remainder = divmod(len(cpus), shard_count)
    result: list[tuple[int, ...]] = []
    cursor = 0
    for index in range(shard_count):
        size = base + (1 if index < remainder else 0)
        result.append(cpus[cursor : cursor + size])
        cursor += size
    return tuple(result)


def generate_execution_topologies(
    allowed_cpu_ids: Sequence[int],
    *,
    workload_class: str = "general",
    scheduler_cpu_ids: Sequence[int] = (),
    allow_affinity_overlap: bool = False,
    affinity_policy: str = "uniform",
    cpu_order: Sequence[int] | None = None,
    physical_core_groups: Sequence[Sequence[int]] = (),
) -> tuple[ExecutionTopology, ...]:
    cpus = _cpu_tuple(tuple(allowed_cpu_ids), "allowed_cpu_ids")
    scheduler = _cpu_tuple(tuple(scheduler_cpu_ids), "scheduler_cpu_ids")
    if any(cpu not in cpus for cpu in scheduler):
        raise ValueError("scheduler_cpu_ids must be a subset of allowed_cpu_ids")
    compute_set = (
        cpus if allow_affinity_overlap else tuple(cpu for cpu in cpus if cpu not in scheduler)
    )
    if not compute_set:
        raise ValueError("scheduler partition leaves no compute CPUs")
    if cpu_order is not None:
        # Keep the caller's physical-core-first/SMT-last order; sorting here
        # would silently erase the topology policy while still passing set
        # equality checks.
        ordered = _ordered_cpu_tuple(tuple(cpu_order), "cpu_order")
        if set(ordered) != set(compute_set):
            raise ValueError("cpu_order must be a permutation of compute CPUs")
        compute = ordered
    elif affinity_policy == "physical_core_first" and physical_core_groups:
        groups = [
            _cpu_tuple(group, f"physical_core_groups[{index}]")
            for index, group in enumerate(physical_core_groups)
        ]
        if any(set(group) - set(cpus) for group in groups):
            raise ValueError("physical_core_groups must be a subset of allowed CPUs")
        # Scheduler-partition candidates remove scheduler CPUs from the
        # compute side while retaining each physical core's sibling order.
        groups = [tuple(cpu for cpu in group if cpu in compute_set) for group in groups]
        groups = [group for group in groups if group]
        if len({cpu for group in groups for cpu in group}) != sum(len(group) for group in groups):
            raise ValueError("physical_core_groups overlap")
        ordered_list: list[int] = []
        for offset in range(max((len(group) for group in groups), default=0)):
            ordered_list.extend(group[offset] for group in groups if offset < len(group))
        ordered_list.extend(cpu for cpu in compute_set if cpu not in ordered_list)
        compute = tuple(ordered_list)
    else:
        compute = compute_set
    shard_counts = tuple(
        dict.fromkeys(
            (len(compute) + target_size - 1) // target_size for target_size in (1, 2, 3, 4)
        )
    )
    return tuple(
        ExecutionTopology(
            workload_class=workload_class,
            shards=partition_cpus(compute, count),
            worker_count=count,
            scheduler_cpu_ids=scheduler,
            request_threads=max(1, min(_MAX_NATIVE_REQUEST_THREADS, len(cpus), count)),
            affinity_policy=affinity_policy,
            allow_affinity_overlap=allow_affinity_overlap,
        )
        for count in shard_counts
    )


generate_topology_candidates = generate_execution_topologies


def physical_core_first_cpu_order(
    envelope: HostPerformanceEnvelope,
) -> tuple[int, ...]:
    """Return an attested physical-core-first, SMT-later CPU permutation."""

    if not envelope.physical_core_groups:
        return envelope.allowed_cpu_ids
    ordered: list[int] = []
    maximum_siblings = max(len(group) for group in envelope.physical_core_groups)
    for sibling_index in range(maximum_siblings):
        ordered.extend(
            group[sibling_index]
            for group in envelope.physical_core_groups
            if sibling_index < len(group)
        )
    ordered.extend(cpu for cpu in envelope.allowed_cpu_ids if cpu not in ordered)
    if set(ordered) != set(envelope.allowed_cpu_ids):
        raise ValueError("physical topology does not cover the allowed CPU set")
    return tuple(ordered)


def execution_topology_id(topology: ExecutionTopology) -> str:
    payload = topology.to_dict()
    shape = "-".join(str(len(shard)) for shard in topology.shards)
    digest = _canonical_sha256(payload)[:12]
    return f"{topology.affinity_policy}-{topology.shard_count}shards-{shape}threads-{digest}"


def performance_topology_key(mode: str, workload_class: str) -> str:
    """Return the canonical key for one frozen Stage 5.2 topology."""

    if mode not in _MODE_NAMES:
        raise ValueError(f"unsupported Stage 5.2 mode {mode!r}")
    if workload_class not in STAGE052_WORKLOAD_CLASSES:
        raise ValueError(f"unsupported Stage 5.2 workload class {workload_class!r}")
    return f"{mode}:{workload_class}"


def generate_mode_topology_candidates(
    envelope: HostPerformanceEnvelope,
    *,
    mode: str,
    workload_class: str,
) -> tuple[ExecutionTopology, ...]:
    """Generate the complete bounded topology search space for one mode block."""

    if mode not in _MODE_NAMES:
        raise ValueError(f"unsupported Stage 5.2 mode {mode!r}")
    if workload_class not in STAGE052_WORKLOAD_CLASSES:
        raise ValueError("workload_class must be c5 or 100-customer")
    cpus = envelope.allowed_cpu_ids
    physical_order = physical_core_first_cpu_order(envelope)
    candidates: list[ExecutionTopology] = []
    if mode != "host_scheduler":
        candidates.extend(
            generate_execution_topologies(
                cpus,
                workload_class=workload_class,
                affinity_policy="free_scheduler",
            )
        )
        pinned_policy = "physical_core_first" if envelope.physical_core_groups else "uniform"
        candidates.extend(
            generate_execution_topologies(
                cpus,
                workload_class=workload_class,
                affinity_policy=pinned_policy,
                cpu_order=physical_order,
            )
        )
    else:
        candidates.extend(
            generate_execution_topologies(
                cpus,
                workload_class=workload_class,
                scheduler_cpu_ids=cpus,
                allow_affinity_overlap=True,
                affinity_policy="free_scheduler",
            )
        )
        for shared in generate_execution_topologies(
            cpus,
            workload_class=workload_class,
        ):
            client_cpu_count = shared.shard_count
            if client_cpu_count >= len(cpus):
                continue
            scheduler_cpus = physical_order[: len(cpus) - client_cpu_count]
            client_cpus = physical_order[len(cpus) - client_cpu_count :]
            candidates.append(
                ExecutionTopology(
                    workload_class=workload_class,
                    shards=tuple((cpu,) for cpu in client_cpus),
                    worker_count=client_cpu_count,
                    scheduler_cpu_ids=scheduler_cpus,
                    request_threads=min(_MAX_NATIVE_REQUEST_THREADS, client_cpu_count),
                    affinity_policy="scheduler_partition",
                    allow_affinity_overlap=False,
                )
            )
    unique: dict[str, ExecutionTopology] = {}
    for topology in candidates:
        identifier = execution_topology_id(topology)
        if identifier in unique:
            raise ValueError("topology identifier collision")
        unique[identifier] = topology
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True, slots=True)
class MemoryAdmission:
    passed: bool
    available_bytes: int
    required_bytes: int
    headroom_bytes: int
    swap_total_bytes: int
    swap_used_bytes: int
    reason: str
    swap_current_bytes: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("available_bytes", self.available_bytes),
            ("required_bytes", self.required_bytes),
            ("headroom_bytes", self.headroom_bytes),
            ("swap_total_bytes", self.swap_total_bytes),
            ("swap_used_bytes", self.swap_used_bytes),
            ("swap_current_bytes", self.swap_current_bytes),
        ):
            _nonnegative_int(value, name)
        if self.swap_used_bytes > self.swap_total_bytes:
            raise ValueError("swap_used_bytes cannot exceed swap_total_bytes")

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "available_bytes": self.available_bytes,
            "required_bytes": self.required_bytes,
            "headroom_bytes": self.headroom_bytes,
            "swap_total_bytes": self.swap_total_bytes,
            "swap_used_bytes": self.swap_used_bytes,
            "swap_current_bytes": self.swap_current_bytes,
            "reason": self.reason,
        }


def check_memory_admission(
    envelope: HostPerformanceEnvelope,
    required_bytes: int,
    *,
    headroom_fraction: float = 0.20,
    require_zero_swap: bool = True,
) -> MemoryAdmission:
    required = _positive_int(required_bytes, "required_bytes")
    if (
        isinstance(headroom_fraction, bool)
        or not isinstance(headroom_fraction, int | float)
        or not math.isfinite(float(headroom_fraction))
        or not 0.0 <= float(headroom_fraction) < 1.0
    ):
        raise ValueError("headroom_fraction must be finite and in [0, 1)")
    available = envelope.effective_memory_limit_bytes
    headroom = math.ceil(required * float(headroom_fraction))
    needed = required + headroom
    swap_failure = require_zero_swap and (
        envelope.swap_used_bytes > 0
        or (envelope.swap_current_bytes is not None and envelope.swap_current_bytes > 0)
    )
    passed = available >= needed and not swap_failure
    reason = (
        "swap is present while zero-swap admission is required"
        if swap_failure
        else (
            f"available memory {available} is below required {needed}"
            if available < needed
            else "memory and swap admission passed"
        )
    )
    return MemoryAdmission(
        passed,
        available,
        required,
        headroom,
        envelope.swap_total_bytes,
        envelope.swap_used_bytes,
        reason,
        0 if envelope.swap_current_bytes is None else envelope.swap_current_bytes,
    )


def require_memory_admission(
    envelope: HostPerformanceEnvelope,
    required_bytes: int,
    *,
    headroom_fraction: float = 0.20,
    require_zero_swap: bool = True,
) -> MemoryAdmission:
    result = check_memory_admission(
        envelope,
        required_bytes,
        headroom_fraction=headroom_fraction,
        require_zero_swap=require_zero_swap,
    )
    if not result.passed:
        raise RuntimeError(result.reason)
    return result


admit_memory = check_memory_admission
validate_memory_admission = require_memory_admission


@dataclass(frozen=True, slots=True)
class BuildArtifactIdentity:
    """Attestation identity for one concrete wheel/native/scheduler build."""

    git_revision: str
    git_tree: str
    source_manifest_sha256: str
    wheel_sha256: str
    native_sha256: str
    scheduler_sha256: str
    compiler_version: str
    flags: tuple[str, ...]
    cpu_feature_mask: tuple[str, ...]
    schema_version: str = PERFORMANCE_PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PERFORMANCE_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported BuildArtifactIdentity schema")
        for name, value in (
            ("git_revision", self.git_revision),
            ("git_tree", self.git_tree),
        ):
            if not isinstance(value, str) or _GIT_SHA1_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase 40-character Git SHA-1")
        if not isinstance(self.compiler_version, str) or not self.compiler_version.strip():
            raise ValueError("compiler_version must be non-empty")
        for name, value in (
            ("source_manifest_sha256", self.source_manifest_sha256),
            ("wheel_sha256", self.wheel_sha256),
            ("native_sha256", self.native_sha256),
            ("scheduler_sha256", self.scheduler_sha256),
        ):
            _sha256(value, name)
        if not isinstance(self.flags, (tuple, list)):
            raise ValueError("artifact flags must be an array")
        if any(not isinstance(flag, str) or not flag for flag in self.flags):
            raise ValueError("artifact flags must contain non-empty strings")
        object.__setattr__(self, "flags", tuple(self.flags))
        object.__setattr__(
            self,
            "cpu_feature_mask",
            _string_tuple(self.cpu_feature_mask, "cpu_feature_mask"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "git_revision": self.git_revision,
            "git_tree": self.git_tree,
            "source_manifest_sha256": self.source_manifest_sha256,
            "wheel_sha256": self.wheel_sha256,
            "native_sha256": self.native_sha256,
            "scheduler_sha256": self.scheduler_sha256,
            "compiler_version": self.compiler_version,
            "flags": list(self.flags),
            "cpu_feature_mask": list(self.cpu_feature_mask),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BuildArtifactIdentity:
        expected = {
            "schema_version",
            "git_revision",
            "git_tree",
            "source_manifest_sha256",
            "wheel_sha256",
            "native_sha256",
            "scheduler_sha256",
            "compiler_version",
            "flags",
            "cpu_feature_mask",
        }
        _exact_fields(payload, expected, "build_artifact_identity")
        for name in (
            "schema_version",
            "git_revision",
            "git_tree",
            "source_manifest_sha256",
            "wheel_sha256",
            "native_sha256",
            "scheduler_sha256",
            "compiler_version",
        ):
            if not isinstance(payload[name], str):
                raise ValueError(f"{name} must be a string")
        if not isinstance(payload["flags"], list):
            raise ValueError("artifact flags must be an array")
        if any(not isinstance(item, str) for item in payload["flags"]):
            raise ValueError("artifact flags must contain strings")
        return cls(
            git_revision=cast(str, payload["git_revision"]),
            git_tree=cast(str, payload["git_tree"]),
            source_manifest_sha256=cast(str, payload["source_manifest_sha256"]),
            wheel_sha256=cast(str, payload["wheel_sha256"]),
            native_sha256=cast(str, payload["native_sha256"]),
            scheduler_sha256=cast(str, payload["scheduler_sha256"]),
            compiler_version=cast(str, payload["compiler_version"]),
            flags=tuple(cast(list[str], payload["flags"])),
            cpu_feature_mask=_string_tuple(payload["cpu_feature_mask"], "cpu_feature_mask"),
            schema_version=cast(str, payload["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class BuildCandidate:
    name: str
    flags: tuple[str, ...]
    portable: bool
    lto: bool
    host_native: bool
    compiler: str = ""
    cpu_features: tuple[str, ...] = ()
    artifact_identity: BuildArtifactIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name not in _BUILD_NAMES:
            raise ValueError(f"unsupported build profile {self.name!r}")
        if not isinstance(self.flags, (tuple, list)):
            raise ValueError("build flags must be an array")
        if any(not isinstance(flag, str) or not flag for flag in self.flags):
            raise ValueError("build flags must contain non-empty strings")
        if any("fast-math" in flag or "ffast-math" in flag for flag in self.flags):
            raise ValueError("fast-math is forbidden")
        if self.name == "host-native-lto" and not self.host_native:
            raise ValueError("host-native-lto must be host_native")
        if self.name != "host-native-lto" and self.host_native:
            raise ValueError("only host-native-lto may be host_native")
        if self.name.endswith("lto") and not self.lto:
            raise ValueError("LTO candidate must set lto=True")
        if self.name == "portable-o3" and self.lto:
            raise ValueError("portable-o3 must set lto=False")
        object.__setattr__(self, "flags", tuple(self.flags))
        object.__setattr__(self, "cpu_features", _string_tuple(self.cpu_features, "cpu_features"))
        if self.artifact_identity is not None:
            if not isinstance(self.artifact_identity, BuildArtifactIdentity):
                raise ValueError("artifact_identity must be BuildArtifactIdentity or None")
            if self.artifact_identity.flags != self.flags:
                raise ValueError("artifact identity flags do not match build candidate")
            if self.artifact_identity.cpu_feature_mask != self.cpu_features:
                raise ValueError("artifact CPU feature mask does not match candidate")

    @property
    def attested(self) -> bool:
        return self.artifact_identity is not None

    def with_identity(self, identity: BuildArtifactIdentity) -> BuildCandidate:
        return replace(self, artifact_identity=identity)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PERFORMANCE_PROFILE_SCHEMA_VERSION,
            "name": self.name,
            "flags": list(self.flags),
            "portable": self.portable,
            "lto": self.lto,
            "host_native": self.host_native,
            "compiler": self.compiler,
            "cpu_features": list(self.cpu_features),
            "artifact_identity": (
                None if self.artifact_identity is None else self.artifact_identity.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BuildCandidate:
        expected = {
            "schema_version",
            "name",
            "flags",
            "portable",
            "lto",
            "host_native",
            "compiler",
            "cpu_features",
            "artifact_identity",
        }
        _exact_fields(payload, expected, "build_candidate")
        if payload["schema_version"] != PERFORMANCE_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported BuildCandidate schema")
        if not isinstance(payload["name"], str) or not isinstance(payload["flags"], list):
            raise ValueError("build candidate name/flags are invalid")
        if any(not isinstance(payload[name], bool) for name in ("portable", "lto", "host_native")):
            raise ValueError("build candidate booleans are invalid")
        if not isinstance(payload["compiler"], str):
            raise ValueError("build candidate compiler must be a string")
        if any(not isinstance(item, str) for item in payload["flags"]):
            raise ValueError("build flags must be strings")
        artifact_payload = payload["artifact_identity"]
        return cls(
            name=payload["name"],
            flags=tuple(cast(list[str], payload["flags"])),
            portable=cast(bool, payload["portable"]),
            lto=cast(bool, payload["lto"]),
            host_native=cast(bool, payload["host_native"]),
            compiler=payload["compiler"],
            cpu_features=_string_tuple(payload["cpu_features"], "cpu_features"),
            artifact_identity=(
                None
                if artifact_payload is None
                else BuildArtifactIdentity.from_dict(
                    _mapping(artifact_payload, "artifact_identity")
                )
            ),
        )


def build_profile_candidates(
    envelope: HostPerformanceEnvelope | None = None,
    *,
    compiler: str | None = None,
    cpu_features: Sequence[str] = (),
    host_native_supported: bool | None = None,
) -> tuple[BuildCandidate, ...]:
    observed_compiler = compiler or (envelope.compiler if envelope is not None else "")
    features = tuple(cpu_features) or (envelope.cpu_features if envelope is not None else ())
    if host_native_supported is None:
        machine = envelope.architecture if envelope is not None else _platform.machine()
        host_native_supported = (
            sys.platform == "linux"
            and machine.casefold() in {"x86_64", "amd64"}
            and any(
                token in observed_compiler.casefold() for token in ("gcc", "g++", "clang", "llvm")
            )
        )
    result = [
        BuildCandidate("portable-o3", ("-O3",), True, False, False, observed_compiler, features),
        BuildCandidate(
            "portable-lto", ("-O3", "-flto"), True, True, False, observed_compiler, features
        ),
    ]
    if host_native_supported:
        result.append(
            BuildCandidate(
                "host-native-lto",
                ("-O3", "-flto", "-march=native"),
                False,
                True,
                True,
                observed_compiler,
                features,
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CalibrationMeasurement:
    build_profile: str
    elapsed_seconds: float
    peak_memory_bytes: int
    semantic_digest: str
    confidence_interval: tuple[float, float] | None = None
    topology_id: str = "default"

    def __post_init__(self) -> None:
        if self.build_profile not in _BUILD_NAMES:
            raise ValueError("unsupported calibration build_profile")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be finite and positive")
        _positive_int(self.peak_memory_bytes, "peak_memory_bytes")
        _sha256(self.semantic_digest, "semantic_digest")
        if self.confidence_interval is not None:
            low, high = self.confidence_interval
            if not math.isfinite(low) or not math.isfinite(high) or low <= 0 or high < low:
                raise ValueError("confidence_interval is invalid")
        if not self.topology_id:
            raise ValueError("topology_id must not be empty")


CalibrationObservation = CalibrationMeasurement


def _normalize_measurements(
    measurements: Mapping[str, Sequence[CalibrationMeasurement]] | Sequence[CalibrationMeasurement],
) -> dict[str, tuple[CalibrationMeasurement, ...]]:
    if isinstance(measurements, Mapping):
        result: dict[str, tuple[CalibrationMeasurement, ...]] = {}
        for name, values in measurements.items():
            rows = tuple(values)
            if not name or not rows or any(item.build_profile != name for item in rows):
                raise ValueError(f"invalid calibration rows for {name!r}")
            result[name] = rows
        return result
    result_list: dict[str, list[CalibrationMeasurement]] = {}
    for item in measurements:
        result_list.setdefault(item.build_profile, []).append(item)
    return {name: tuple(rows) for name, rows in result_list.items()}


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


@dataclass(frozen=True, slots=True)
class TelemetryOverheadReceipt:
    """Alternating paired A/B evidence for the complete telemetry surface."""

    unmonitored_seconds: tuple[float, ...]
    monitored_seconds: tuple[float, ...]
    pair_orders: tuple[str, ...]
    sample_interval_seconds: float
    workload_output_sha256: str
    monitored_resource_summaries: tuple[Mapping[str, object], ...]
    workload_evidence: Mapping[str, object]
    maximum_overhead_fraction: float = 0.02
    schema_version: str = TELEMETRY_OVERHEAD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TELEMETRY_OVERHEAD_SCHEMA_VERSION:
            raise ValueError("unsupported telemetry overhead schema")
        count = len(self.unmonitored_seconds)
        if (
            count < 5
            or count % 2 == 0
            or len(self.monitored_seconds) != count
            or len(self.pair_orders) != count
            or len(self.monitored_resource_summaries) != count
        ):
            raise ValueError(
                "telemetry overhead evidence requires an odd paired sample of at least 5"
            )
        for name, values in (
            ("unmonitored_seconds", self.unmonitored_seconds),
            ("monitored_seconds", self.monitored_seconds),
        ):
            if any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in values
            ):
                raise ValueError(f"{name} must contain finite positive timings")
        expected_orders = tuple("off-on" if index % 2 == 0 else "on-off" for index in range(count))
        if self.pair_orders != expected_orders:
            raise ValueError("telemetry A/B order must alternate off-on and on-off")
        if (
            isinstance(self.sample_interval_seconds, bool)
            or not math.isfinite(self.sample_interval_seconds)
            or self.sample_interval_seconds <= 0.0
        ):
            raise ValueError("sample_interval_seconds must be finite and positive")
        _sha256(self.workload_output_sha256, "workload_output_sha256")
        summaries: list[Mapping[str, object]] = []
        for index, summary in enumerate(self.monitored_resource_summaries):
            mapping = _mapping(summary, f"monitored_resource_summaries[{index}]")
            _canonical_sha256(mapping)
            summaries.append(_freeze_mapping(mapping, f"monitored_resource_summaries[{index}]"))
        object.__setattr__(self, "monitored_resource_summaries", tuple(summaries))
        evidence = _freeze_mapping(self.workload_evidence, "workload_evidence")
        _canonical_sha256(evidence)
        object.__setattr__(self, "workload_evidence", evidence)
        if (
            isinstance(self.maximum_overhead_fraction, bool)
            or not math.isfinite(self.maximum_overhead_fraction)
            or not 0.0 <= self.maximum_overhead_fraction < 1.0
        ):
            raise ValueError("maximum_overhead_fraction must be finite and in [0, 1)")

    @property
    def paired_overhead_fractions(self) -> tuple[float, ...]:
        return tuple(
            monitored / unmonitored - 1.0
            for unmonitored, monitored in zip(
                self.unmonitored_seconds,
                self.monitored_seconds,
                strict=True,
            )
        )

    @property
    def median_overhead_fraction(self) -> float:
        return _median(self.paired_overhead_fractions)

    @property
    def p95_overhead_fraction(self) -> float:
        ordered = sorted(self.paired_overhead_fractions)
        index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        return ordered[index]

    @property
    def passed(self) -> bool:
        # Timings are binary floats; treat a value that differs from the
        # configured boundary only by representation error as exactly on it.
        tolerance = max(1.0e-12, abs(self.maximum_overhead_fraction) * 1.0e-12)
        return (
            self.median_overhead_fraction <= self.maximum_overhead_fraction + tolerance
            and self.p95_overhead_fraction <= self.maximum_overhead_fraction + tolerance
        )

    def require_passed(self) -> None:
        if not self.passed:
            raise ValueError(
                f"telemetry median or p95 overhead exceeds {self.maximum_overhead_fraction:.2%}"
            )

    def require_representative_fixed_work(self) -> None:
        """Reject synthetic probes at the calibration publication boundary."""

        required = {
            "kind": "representative-fixed-work-axis",
            "mode": "current_stage052",
            "axis": "fixed_work",
            "semantic_telemetry": True,
            "physical_telemetry": True,
            "persistence": True,
            "independent_replay": True,
            "fingerprints_identical": True,
        }
        if any(self.workload_evidence.get(key) != value for key, value in required.items()):
            raise ValueError("telemetry overhead evidence is not a representative fixed-work axis")
        surface_fields = {
            "semantic_telemetry",
            "physical_telemetry",
            "persistence",
            "independent_replay",
        }
        unmonitored_surface = self.workload_evidence.get("unmonitored_telemetry_surface")
        monitored_surface = self.workload_evidence.get("monitored_telemetry_surface")
        if (
            not isinstance(unmonitored_surface, Mapping)
            or set(unmonitored_surface) != surface_fields
            or any(unmonitored_surface[field] is not False for field in surface_fields)
            or not isinstance(monitored_surface, Mapping)
            or set(monitored_surface) != surface_fields
            or any(monitored_surface[field] is not True for field in surface_fields)
        ):
            raise ValueError("telemetry overhead on/off surface contract is invalid")
        exact_calls = self.workload_evidence.get("exact_calls")
        instance = self.workload_evidence.get("instance")
        if (
            isinstance(exact_calls, bool)
            or not isinstance(exact_calls, int)
            or exact_calls <= 0
            or not isinstance(instance, str)
            or not instance
        ):
            raise ValueError("representative telemetry workload identity is incomplete")
        warm_samples = self.workload_evidence.get("warm_sample_evidence")
        paired_samples = self.workload_evidence.get("paired_sample_evidence")
        if (
            not isinstance(warm_samples, tuple)
            or len(warm_samples) != 2
            or not isinstance(paired_samples, tuple)
            or len(paired_samples) != len(self.pair_orders)
        ):
            raise ValueError("representative telemetry raw sample evidence is incomplete")
        expected_fingerprint: str | None = None
        for expected_enabled, expected_index, sample in (
            (False, -2, warm_samples[0]),
            (True, -1, warm_samples[1]),
        ):
            if (
                not isinstance(sample, Mapping)
                or sample.get("enabled") is not expected_enabled
                or sample.get("sample_index") != expected_index
                or not isinstance(sample.get("resource_summary"), Mapping)
                or not isinstance(sample.get("workload_evidence"), Mapping)
            ):
                raise ValueError("representative telemetry warm sample evidence is invalid")
            fingerprint = sample.get("fingerprint")
            if not isinstance(fingerprint, str) or _SHA256_RE.fullmatch(fingerprint) is None:
                raise ValueError("representative telemetry sample fingerprint is invalid")
            if expected_fingerprint is None:
                expected_fingerprint = fingerprint
            elif fingerprint != expected_fingerprint:
                raise ValueError("representative telemetry warm sample fingerprint diverged")
        for index, pair in enumerate(paired_samples):
            if (
                not isinstance(pair, Mapping)
                or set(pair) != {"pair_index", "order", "unmonitored", "monitored"}
                or pair.get("pair_index") != index
                or pair.get("order") != self.pair_orders[index]
            ):
                raise ValueError("representative telemetry paired sample evidence is invalid")
            for enabled, name, seconds in (
                (False, "unmonitored", self.unmonitored_seconds[index]),
                (True, "monitored", self.monitored_seconds[index]),
            ):
                sample = pair.get(name)
                if (
                    not isinstance(sample, Mapping)
                    or sample.get("enabled") is not enabled
                    or sample.get("elapsed_seconds") != seconds
                    or not isinstance(sample.get("resource_summary"), Mapping)
                    or not isinstance(sample.get("workload_evidence"), Mapping)
                ):
                    raise ValueError("representative telemetry paired sample evidence is invalid")
                fingerprint = sample.get("fingerprint")
                if fingerprint != expected_fingerprint:
                    raise ValueError("representative telemetry paired sample fingerprint diverged")
        if (
            expected_fingerprint is None
            or hashlib.sha256(expected_fingerprint.encode("ascii")).hexdigest()
            != self.workload_output_sha256
        ):
            raise ValueError("representative telemetry workload digest is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "unmonitored_seconds": list(self.unmonitored_seconds),
            "monitored_seconds": list(self.monitored_seconds),
            "pair_orders": list(self.pair_orders),
            "sample_interval_seconds": self.sample_interval_seconds,
            "workload_output_sha256": self.workload_output_sha256,
            "monitored_resource_summaries": [
                _thaw_json(summary) for summary in self.monitored_resource_summaries
            ],
            "workload_evidence": _thaw_json(self.workload_evidence),
            "maximum_overhead_fraction": self.maximum_overhead_fraction,
            "paired_overhead_fractions": list(self.paired_overhead_fractions),
            "median_overhead_fraction": self.median_overhead_fraction,
            "p95_overhead_fraction": self.p95_overhead_fraction,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TelemetryOverheadReceipt:
        expected = {
            "schema_version",
            "unmonitored_seconds",
            "monitored_seconds",
            "pair_orders",
            "sample_interval_seconds",
            "workload_output_sha256",
            "monitored_resource_summaries",
            "workload_evidence",
            "maximum_overhead_fraction",
            "paired_overhead_fractions",
            "median_overhead_fraction",
            "p95_overhead_fraction",
            "passed",
        }
        _exact_fields(payload, expected, "telemetry_overhead")
        unmonitored = payload["unmonitored_seconds"]
        monitored = payload["monitored_seconds"]
        orders = payload["pair_orders"]
        summaries = payload["monitored_resource_summaries"]
        if (
            not isinstance(unmonitored, list)
            or not isinstance(monitored, list)
            or any(
                isinstance(item, bool) or not isinstance(item, int | float)
                for item in (*unmonitored, *monitored)
            )
            or not isinstance(orders, list)
            or any(not isinstance(item, str) for item in orders)
            or not isinstance(summaries, list)
        ):
            raise ValueError("telemetry overhead arrays are invalid")
        sample_interval = _finite_nonnegative(
            payload["sample_interval_seconds"],
            "sample_interval_seconds",
        )
        if sample_interval <= 0.0:
            raise ValueError("sample_interval_seconds must be positive")
        receipt = cls(
            schema_version=cast(str, payload["schema_version"]),
            unmonitored_seconds=tuple(float(item) for item in unmonitored),
            monitored_seconds=tuple(float(item) for item in monitored),
            pair_orders=tuple(cast(list[str], orders)),
            sample_interval_seconds=sample_interval,
            workload_output_sha256=_sha256(
                payload["workload_output_sha256"],
                "workload_output_sha256",
            ),
            monitored_resource_summaries=tuple(
                _mapping(item, f"monitored_resource_summaries[{index}]")
                for index, item in enumerate(summaries)
            ),
            workload_evidence=_mapping(payload["workload_evidence"], "workload_evidence"),
            maximum_overhead_fraction=_finite_nonnegative(
                payload["maximum_overhead_fraction"],
                "maximum_overhead_fraction",
            ),
        )
        derived = receipt.to_dict()
        for field_name in (
            "paired_overhead_fractions",
            "median_overhead_fraction",
            "p95_overhead_fraction",
            "passed",
        ):
            if payload[field_name] != derived[field_name]:
                raise ValueError(f"telemetry overhead derived field diverged: {field_name}")
        return receipt


def select_build_candidate(
    candidates: Sequence[BuildCandidate],
    measurements: Mapping[str, Sequence[CalibrationMeasurement]] | Sequence[CalibrationMeasurement],
    *,
    relative_tie_fraction: float = 0.03,
) -> BuildCandidate:
    if not candidates:
        raise ValueError("build candidate set is empty")
    if (
        isinstance(relative_tie_fraction, bool)
        or not isinstance(relative_tie_fraction, int | float)
        or not math.isfinite(float(relative_tie_fraction))
        or not 0 <= float(relative_tie_fraction) < 1
    ):
        raise ValueError("relative_tie_fraction must be finite and in [0, 1)")
    by_name = {candidate.name: candidate for candidate in candidates}
    if len(by_name) != len(candidates):
        raise ValueError("build candidate names must be unique")
    observed = _normalize_measurements(measurements)
    valid: dict[str, tuple[float, int, tuple[float, float] | None]] = {}
    reference: str | None = None
    for name in by_name:
        rows = observed.get(name)
        if not rows:
            continue
        digests = {item.semantic_digest for item in rows}
        if len(digests) != 1:
            continue
        digest = next(iter(digests))
        if reference is None:
            reference = digest
        if digest != reference:
            continue
        intervals = [item.confidence_interval for item in rows if item.confidence_interval]
        ci = (
            (min(item[0] for item in intervals), max(item[1] for item in intervals))
            if intervals
            else None
        )
        valid[name] = (
            _median([item.elapsed_seconds for item in rows]),
            max(item.peak_memory_bytes for item in rows),
            ci,
        )
    if not valid:
        raise ValueError("no semantically valid build candidate")
    fastest = min(valid, key=lambda name: (valid[name][0], name))
    fastest_elapsed = valid[fastest][0]
    fastest_ci = valid[fastest][2]

    def overlaps(first: tuple[float, float] | None, second: tuple[float, float] | None) -> bool:
        return (
            first is not None
            and second is not None
            and first[0] <= second[1]
            and second[0] <= first[1]
        )

    eligible = [
        name
        for name, (elapsed, _, ci) in valid.items()
        if elapsed <= fastest_elapsed * (1 + float(relative_tie_fraction))
        or overlaps(ci, fastest_ci)
    ]
    return by_name[
        min(
            eligible,
            key=lambda name: (
                not by_name[name].portable,
                by_name[name].host_native,
                valid[name][1],
                valid[name][0],
                name,
            ),
        )
    ]


select_build_profile = select_build_candidate
BuildProfile = BuildCandidate


@dataclass(frozen=True, slots=True)
class RuntimeResourceSummaryV2:
    elapsed_seconds: float
    effective_cores: float = 0.0
    cpu_utilization_fraction: float = 0.0
    user_cpu_seconds: float = 0.0
    system_cpu_seconds: float = 0.0
    run_queue_wait_seconds: float = 0.0
    context_switches: int = 0
    cpu_migrations: int = 0
    minor_faults: int = 0
    major_faults: int = 0
    rss_bytes: int = 0
    pss_bytes: int = 0
    cgroup_memory_current_bytes: int = 0
    cgroup_memory_peak_bytes: int = 0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    swap_in_bytes: int = 0
    swap_out_bytes: int = 0
    queue_wait_seconds: float = 0.0
    queue_depth_peak: int = 0
    pending_tasks_peak: int = 0
    queue_full_count: int = 0
    rejected_count: int = 0
    worker_p95_seconds: float = 0.0
    worker_max_seconds: float = 0.0
    worker_min_seconds: float = 0.0
    build_seconds: float = 0.0
    startup_seconds: float = 0.0
    solver_seconds: float = 0.0
    persistence_seconds: float = 0.0
    replay_seconds: float = 0.0
    p50_end_to_end_seconds: float = 0.0
    p95_end_to_end_seconds: float = 0.0
    p99_end_to_end_seconds: float = 0.0
    max_end_to_end_seconds: float = 0.0
    schema_version: str = RUNTIME_RESOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_RESOURCE_SCHEMA_VERSION:
            raise ValueError("unsupported RuntimeResourceSummaryV2 schema")
        float_fields = (
            "elapsed_seconds",
            "effective_cores",
            "cpu_utilization_fraction",
            "user_cpu_seconds",
            "system_cpu_seconds",
            "run_queue_wait_seconds",
            "queue_wait_seconds",
            "worker_p95_seconds",
            "worker_max_seconds",
            "worker_min_seconds",
            "build_seconds",
            "startup_seconds",
            "solver_seconds",
            "persistence_seconds",
            "replay_seconds",
            "p50_end_to_end_seconds",
            "p95_end_to_end_seconds",
            "p99_end_to_end_seconds",
            "max_end_to_end_seconds",
        )
        for name in float_fields:
            _finite_nonnegative(getattr(self, name), name)
        integer_fields = (
            "context_switches",
            "cpu_migrations",
            "minor_faults",
            "major_faults",
            "rss_bytes",
            "pss_bytes",
            "cgroup_memory_current_bytes",
            "cgroup_memory_peak_bytes",
            "io_read_bytes",
            "io_write_bytes",
            "swap_in_bytes",
            "swap_out_bytes",
            "queue_depth_peak",
            "pending_tasks_peak",
            "queue_full_count",
            "rejected_count",
        )
        for name in integer_fields:
            _nonnegative_int(getattr(self, name), name)
        if self.worker_p95_seconds < self.worker_min_seconds:
            raise ValueError("worker_p95_seconds cannot be below worker_min_seconds")
        if self.worker_max_seconds < self.worker_p95_seconds:
            raise ValueError("worker_max_seconds cannot be below worker_p95_seconds")

    @property
    def cpu_utilization_percent(self) -> float:
        return self.cpu_utilization_fraction * 100.0

    @property
    def effective_cpu_seconds(self) -> float:
        return self.effective_cores * self.elapsed_seconds

    def to_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RuntimeResourceSummaryV2:
        _exact_fields(payload, set(cls.__dataclass_fields__), "runtime_resource_summary")
        return cls(**cast(Any, dict(payload)))


def _read(path: Path, reader: Callable[[Path], str] | None) -> str:
    return reader(path) if reader is not None else path.read_text(encoding="utf-8")


def _parse_meminfo(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z_]+):\s+([0-9]+)\s*(kB)?\s*$", line)
        if match is None:
            continue
        value = int(match.group(2))
        result[match.group(1)] = value * 1024 if match.group(3) else value
    return result


def _parse_cgroup_limit(text: str) -> int | None:
    value = text.strip()
    if value == "max":
        return None
    if not value.isdigit():
        raise ValueError("invalid cgroup memory limit")
    return int(value)


def _discover_topology(
    cpus: Sequence[int],
    root: Path,
    reader: Callable[[Path], str] | None,
) -> tuple[tuple[tuple[int, ...], ...], str]:
    groups: dict[tuple[int, int], list[int]] = {}
    for cpu in cpus:
        base = root / "devices" / "system" / "cpu" / f"cpu{cpu}" / "topology"
        try:
            core = int(_read(base / "core_id", reader).strip())
            package = int(_read(base / "physical_package_id", reader).strip())
        except (OSError, ValueError):
            return (), "unavailable"
        groups.setdefault((package, core), []).append(cpu)
    return tuple(tuple(sorted(group)) for _, group in sorted(groups.items())), "sysfs"


def _discover_features(root: Path, reader: Callable[[Path], str] | None) -> tuple[str, ...]:
    try:
        text = _read(root / "cpuinfo", reader)
    except OSError:
        return ()
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().casefold() in {"flags", "features"}:
            return _string_tuple(value.split(), "cpu_features")
    return ()


def _discover_diagnostic_telemetry(
    *,
    sysfs_root: Path,
    cpus: Sequence[int],
    reader: Callable[[Path], str] | None,
) -> dict[str, object]:
    power_sources: list[dict[str, object]] = []
    power_root = sysfs_root / "class" / "power_supply"
    try:
        power_entries = tuple(sorted(power_root.iterdir()))
    except OSError:
        power_entries = ()
    for entry in power_entries:
        try:
            supply_type = _read(entry / "type", reader).strip()
            online = int(_read(entry / "online", reader).strip())
        except (OSError, ValueError):
            continue
        if supply_type.casefold() not in {"mains", "usb", "usb_c", "wireless"}:
            continue
        power_sources.append(
            {
                "name": entry.name,
                "type": supply_type,
                "online": online == 1,
            }
        )
    power_state: object
    if power_sources:
        power_state = {
            "status": (
                "online" if any(bool(item["online"]) for item in power_sources) else "offline"
            ),
            "sources": power_sources,
        }
    else:
        power_state = "unavailable"

    try:
        platform_profile = _read(
            sysfs_root / "firmware" / "acpi" / "platform_profile",
            reader,
        ).strip()
    except OSError:
        platform_profile = ""

    frequencies_khz: list[int] = []
    for cpu in cpus:
        base = sysfs_root / "devices" / "system" / "cpu" / f"cpu{cpu}" / "cpufreq"
        value: int | None = None
        for name in ("scaling_cur_freq", "cpuinfo_cur_freq"):
            try:
                candidate = int(_read(base / name, reader).strip())
            except (OSError, ValueError):
                continue
            if candidate > 0:
                value = candidate
                break
        if value is not None:
            frequencies_khz.append(value)
    frequency: object = "unavailable"
    if frequencies_khz:
        frequency = {
            "unit": "kHz",
            "sample_count": len(frequencies_khz),
            "minimum": min(frequencies_khz),
            "median": statistics.median(frequencies_khz),
            "maximum": max(frequencies_khz),
        }

    temperatures_millidegree_c: list[dict[str, object]] = []
    thermal_root = sysfs_root / "class" / "thermal"
    try:
        thermal_entries = tuple(sorted(thermal_root.glob("thermal_zone*")))
    except OSError:
        thermal_entries = ()
    for entry in thermal_entries:
        try:
            raw_temperature = int(_read(entry / "temp", reader).strip())
            sensor_type = _read(entry / "type", reader).strip()
        except (OSError, ValueError):
            continue
        if not -100_000 <= raw_temperature <= 250_000:
            continue
        temperatures_millidegree_c.append(
            {
                "name": entry.name,
                "type": sensor_type or "unknown",
                "millidegree_celsius": raw_temperature,
            }
        )
    temperature: object = (
        {
            "unit": "millidegree_celsius",
            "sensors": temperatures_millidegree_c,
        }
        if temperatures_millidegree_c
        else "unavailable"
    )
    return {
        "power_state": power_state,
        "power_plan": platform_profile or "unavailable",
        "frequency": frequency,
        "temperature": temperature,
    }


def detect_host_performance(
    *,
    allowed_cpu_ids: Sequence[int] | None = None,
    platform_name: str | None = None,
    architecture: str | None = None,
    proc_root: Path = Path("/proc"),
    sysfs_root: Path = Path("/sys"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    reader: Callable[[Path], str] | None = None,
) -> HostPerformanceEnvelope:
    observed_platform = (platform_name or sys.platform).casefold()
    observed_architecture = (architecture or _platform.machine()).casefold()
    if observed_platform != "linux":
        raise RuntimeError("performance detection supports Linux/WSL only")
    if observed_architecture not in {"x86_64", "amd64"}:
        raise RuntimeError("performance detection supports x86-64 only")
    if allowed_cpu_ids is None:
        try:
            cpus = tuple(sorted(os.sched_getaffinity(0)))
        except (AttributeError, OSError) as error:
            raise RuntimeError("cannot determine allowed CPUs") from error
    else:
        cpus = _cpu_tuple(tuple(allowed_cpu_ids), "allowed_cpu_ids")
    if not cpus:
        raise RuntimeError("allowed CPU set is empty")
    meminfo = _parse_meminfo(_read(proc_root / "meminfo", reader))
    memory_total = meminfo.get("MemTotal", 0)
    memory_available = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
    if memory_total <= 0 or memory_available <= 0:
        raise RuntimeError("MemTotal/MemAvailable are unavailable")
    swap_total = meminfo.get("SwapTotal", 0)
    swap_used = max(0, swap_total - meminfo.get("SwapFree", swap_total))

    def cgroup_directories() -> tuple[Path, ...]:
        try:
            membership = _read(proc_root / "self" / "cgroup", reader)
        except OSError:
            membership = ""
        mount_hierarchy_root = Path("/")
        try:
            mountinfo = _read(proc_root / "self" / "mountinfo", reader)
        except OSError:
            mountinfo = ""
        for line in mountinfo.splitlines():
            fields = line.split()
            if "-" not in fields:
                continue
            separator_index = fields.index("-")
            if (
                separator_index + 1 < len(fields)
                and fields[separator_index + 1] == "cgroup2"
                and len(fields) > 4
                and Path(fields[4]) == cgroup_root
            ):
                candidate_root = Path(fields[3])
                if candidate_root.is_absolute() and ".." not in candidate_root.parts:
                    mount_hierarchy_root = candidate_root
                break
        relative: Path | None = None
        for line in membership.splitlines():
            hierarchy, separator, raw_path = line.partition("::")
            if separator and hierarchy == "0":
                candidate = Path(raw_path.strip())
                if candidate.is_absolute() and ".." not in candidate.parts:
                    try:
                        relative = candidate.relative_to(mount_hierarchy_root)
                    except ValueError:
                        relative = None
                break
        leaf = cgroup_root if relative is None else cgroup_root / relative
        try:
            resolved_root = cgroup_root.resolve(strict=False)
            resolved_leaf = leaf.resolve(strict=False)
            resolved_leaf.relative_to(resolved_root)
        except (OSError, ValueError):
            return (cgroup_root,)
        directories: list[Path] = []
        current = leaf
        while True:
            directories.append(current)
            if current == cgroup_root:
                break
            current = current.parent
        return tuple(directories)

    def optional_limit(path: Path) -> int | None:
        try:
            return _parse_cgroup_limit(_read(path, reader))
        except (OSError, ValueError):
            return None

    def optional_current(path: Path, name: str) -> int | None:
        try:
            value = int(_read(path, reader).strip())
            return _nonnegative_int(value, name)
        except (OSError, ValueError):
            return None

    cgroup_directories_observed = cgroup_directories()

    def effective_cgroup_pair(
        limit_name: str,
        current_name: str,
        current_label: str,
    ) -> tuple[int | None, int | None]:
        finite: list[tuple[int, int, int | None]] = []
        observed_current: list[int] = []
        for directory in cgroup_directories_observed:
            limit = optional_limit(directory / limit_name)
            current = optional_current(directory / current_name, current_label)
            if current is not None:
                observed_current.append(current)
            if limit is not None:
                remaining = limit if current is None else max(0, limit - current)
                finite.append((remaining, limit, current))
        if finite:
            _remaining, limit, current = min(finite, key=lambda row: (row[0], row[1]))
            return limit, current
        return None, (observed_current[0] if observed_current else None)

    # cgroup v2 limits are hierarchical.  Select the ancestor that leaves the
    # least effective headroom, while keeping current/swap probes independent.
    memory_limit, memory_current = effective_cgroup_pair(
        "memory.max", "memory.current", "memory_current_bytes"
    )
    swap_limit, swap_current = effective_cgroup_pair(
        "memory.swap.max", "memory.swap.current", "swap_current_bytes"
    )
    groups, topology_source = _discover_topology(cpus, sysfs_root, reader)
    telemetry: dict[str, object] = {
        "kernel_release": _platform.release(),
        "cgroup_v2_path": str(cgroup_directories_observed[0]),
        "cgroup_v2_ancestor_count": len(cgroup_directories_observed),
        **_discover_diagnostic_telemetry(
            sysfs_root=sysfs_root,
            cpus=cpus,
            reader=reader,
        ),
    }
    return HostPerformanceEnvelope(
        allowed_cpu_ids=cpus,
        physical_core_groups=groups,
        memory_total_bytes=memory_total,
        memory_available_bytes=memory_available,
        memory_limit_bytes=memory_limit,
        memory_current_bytes=memory_current,
        swap_total_bytes=swap_total,
        swap_used_bytes=swap_used,
        swap_current_bytes=swap_current,
        swap_limit_bytes=swap_limit,
        cpu_features=_discover_features(proc_root, reader),
        compiler=_platform.python_compiler(),
        topology_source=topology_source,
        telemetry=telemetry,
    )


@dataclass(frozen=True, slots=True)
class FrozenRuntimeBinding:
    """Validated, typed runtime view of one frozen performance profile."""

    profile_sha256: str
    observed_host: HostPerformanceEnvelope
    selected_build: BuildCandidate
    topologies: Mapping[str, ExecutionTopology]
    memory_admission: Mapping[str, MemoryAdmission]
    telemetry_overhead: TelemetryOverheadReceipt
    signed_input_inventory: Mapping[str, object]
    scheduler_lifecycle_by_workload: Mapping[str, str]
    scheduler_lifecycle_evidence: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        _sha256(self.profile_sha256, "profile_sha256")
        if not isinstance(self.observed_host, HostPerformanceEnvelope):
            raise ValueError("observed_host must be HostPerformanceEnvelope")
        if not isinstance(self.selected_build, BuildCandidate) or not self.selected_build.attested:
            raise ValueError("selected_build must be an attested BuildCandidate")
        topologies = dict(self.topologies)
        admissions = dict(self.memory_admission)
        if not topologies or set(admissions) != set(topologies):
            raise ValueError("runtime topology and memory-admission matrices differ")
        if any(not isinstance(value, ExecutionTopology) for value in topologies.values()):
            raise ValueError("runtime topologies must contain ExecutionTopology values")
        if any(
            not isinstance(value, MemoryAdmission) or not value.passed
            for value in admissions.values()
        ):
            raise ValueError("runtime memory admission must pass for every topology")
        lifecycle = dict(self.scheduler_lifecycle_by_workload)
        evidence = dict(self.scheduler_lifecycle_evidence)
        if set(lifecycle) != set(STAGE052_WORKLOAD_CLASSES) or set(evidence) != set(
            STAGE052_WORKLOAD_CLASSES
        ):
            raise ValueError("scheduler lifecycle matrix is incomplete")
        normalized_evidence: dict[str, Mapping[str, object]] = {}
        for workload_class, value in evidence.items():
            normalized_evidence[workload_class] = _freeze_mapping(
                value, f"scheduler_lifecycle_evidence.{workload_class}"
            )
        inventory = _freeze_mapping(self.signed_input_inventory, "signed_input_inventory")
        object.__setattr__(self, "topologies", MappingProxyType(topologies))
        object.__setattr__(self, "memory_admission", MappingProxyType(admissions))
        object.__setattr__(
            self,
            "scheduler_lifecycle_by_workload",
            MappingProxyType(lifecycle),
        )
        object.__setattr__(
            self,
            "scheduler_lifecycle_evidence",
            MappingProxyType(normalized_evidence),
        )
        object.__setattr__(
            self,
            "signed_input_inventory",
            inventory,
        )

    @property
    def allowed_cpu_ids(self) -> tuple[int, ...]:
        return self.observed_host.allowed_cpu_ids

    @property
    def max_executor_workers(self) -> int:
        return max(topology.shard_count for topology in self.topologies.values())

    @property
    def memory_capacity_bytes(self) -> int:
        return min(
            self.observed_host.memory_total_bytes,
            self.observed_host.memory_limit_bytes or self.observed_host.memory_total_bytes,
        )

    def topology_for(self, mode: str, workload_class: str) -> ExecutionTopology:
        return self.topologies[performance_topology_key(mode, workload_class)]

    def scheduler_lifecycle_for(self, workload_class: str) -> str:
        if workload_class not in STAGE052_WORKLOAD_CLASSES:
            raise ValueError(f"unsupported Stage 5.2 workload class {workload_class!r}")
        return self.scheduler_lifecycle_by_workload[workload_class]

    def host_receipt(self) -> dict[str, object]:
        return {
            "observed_host": self.observed_host.to_dict(),
            "memory_admission": {
                key: self.memory_admission[key].to_dict() for key in sorted(self.memory_admission)
            },
        }


@dataclass(frozen=True, slots=True)
class FrozenPerformanceProfile:
    host: HostPerformanceEnvelope
    selected_build: BuildCandidate
    topologies: Mapping[str, ExecutionTopology]
    calibration: Mapping[str, object]
    selection_reason: str
    schema_version: str = PERFORMANCE_PROFILE_SCHEMA_VERSION
    canonical_sha256: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.schema_version != PERFORMANCE_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported FrozenPerformanceProfile schema")
        if not isinstance(self.selection_reason, str) or not self.selection_reason:
            raise ValueError("selection_reason must be non-empty")
        if not self.selected_build.attested:
            raise ValueError("selected_build requires a complete artifact identity")
        topologies = dict(self.topologies)
        if not topologies:
            raise ValueError("profile must contain at least one topology")
        for name, topology in topologies.items():
            if not isinstance(name, str) or not name:
                raise ValueError("topology keys must be non-empty")
            if not isinstance(topology, ExecutionTopology):
                raise ValueError("topologies must contain ExecutionTopology values")
            if not set(topology.cpu_ids).issubset(self.host.allowed_cpu_ids):
                raise ValueError("topology uses CPUs outside host envelope")
        calibration = _freeze_mapping(self.calibration, "calibration")
        object.__setattr__(self, "topologies", MappingProxyType(topologies))
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "canonical_sha256", _canonical_sha256(self._identity_payload()))

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "host": self.host.identity_dict(),
            "selected_build": self.selected_build.to_dict(),
            "topologies": {
                name: self.topologies[name].to_dict() for name in sorted(self.topologies)
            },
            "calibration": _thaw_json(self.calibration),
            "selection_reason": self.selection_reason,
        }

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "host": self.host.to_dict(),
            "selected_build": self.selected_build.to_dict(),
            "topologies": {
                name: self.topologies[name].to_dict() for name in sorted(self.topologies)
            },
            "calibration": _thaw_json(self.calibration),
            "selection_reason": self.selection_reason,
        }

    @property
    def profile_hash(self) -> str:
        return self.canonical_sha256

    @property
    def canonical_hash(self) -> str:
        return self.canonical_sha256

    @property
    def selected_build_profile(self) -> BuildCandidate:
        return self.selected_build

    def bind_runtime(
        self,
        build_receipt: Mapping[str, object],
        *,
        observed_host: HostPerformanceEnvelope | None = None,
    ) -> FrozenRuntimeBinding:
        """Validate live host/build/calibration state and return one runtime binding."""

        return bind_frozen_performance_profile(
            self,
            build_receipt,
            observed_host=observed_host,
        )

    def to_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["canonical_sha256"] = self.canonical_sha256
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True)

    def save(self, path: Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"performance profile target exists: {destination}")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(self.to_json())
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            publish_no_replace(temporary, destination)
            directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FrozenPerformanceProfile:
        expected = {
            "schema_version",
            "host",
            "selected_build",
            "topologies",
            "calibration",
            "selection_reason",
            "canonical_sha256",
        }
        _exact_fields(payload, expected, "frozen_profile")
        if payload["schema_version"] != PERFORMANCE_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported FrozenPerformanceProfile schema")
        topology_payload = _mapping(payload["topologies"], "topologies")
        if not isinstance(payload["selection_reason"], str):
            raise ValueError("selection_reason must be a string")
        declared = _sha256(payload["canonical_sha256"], "canonical_sha256")
        profile = cls(
            host=HostPerformanceEnvelope.from_dict(_mapping(payload["host"], "host")),
            selected_build=BuildCandidate.from_dict(
                _mapping(payload["selected_build"], "selected_build")
            ),
            topologies={
                name: ExecutionTopology.from_dict(_mapping(value, f"topologies.{name}"))
                for name, value in topology_payload.items()
            },
            calibration=_mapping(payload["calibration"], "calibration"),
            selection_reason=payload["selection_reason"],
        )
        if profile.canonical_sha256 != declared:
            raise ValueError("FrozenPerformanceProfile canonical hash mismatch")
        return profile

    @classmethod
    def load(cls, source: Path | str | Mapping[str, object]) -> FrozenPerformanceProfile:
        if isinstance(source, Mapping):
            payload = source
        else:
            try:
                payload = json.loads(Path(source).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("cannot load frozen performance profile") from error
        return cls.from_dict(_mapping(payload, "frozen_profile"))


def _validate_signed_input_inventory(inventory: Mapping[str, object]) -> None:
    required = {"wheel_receipts", "fixed_work_observations", "telemetry_overhead"}
    if not required.issubset(inventory) or set(inventory) - (required | {"host_envelope"}):
        raise ValueError("signed calibration input inventory fields are invalid")

    seen: set[tuple[str, str]] = set()

    def validate_entry(value: object, name: str) -> None:
        entry = _mapping(value, name)
        if set(entry) != {"storage_alias", "relative_path", "sha256"}:
            raise ValueError(f"{name} fields are invalid")
        alias = entry["storage_alias"]
        relative = entry["relative_path"]
        if not isinstance(alias, str) or not alias:
            raise ValueError(f"{name}.storage_alias must be non-empty")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError(f"{name}.relative_path must be a portable relative path")
        portable = PurePosixPath(relative)
        if portable.is_absolute() or ".." in portable.parts:
            raise ValueError(f"{name}.relative_path must be a portable relative path")
        _sha256(entry["sha256"], f"{name}.sha256")
        identity = (alias, relative)
        if identity in seen:
            raise ValueError("signed calibration input inventory contains duplicate entries")
        seen.add(identity)

    for field_name in ("wheel_receipts", "fixed_work_observations"):
        rows = inventory[field_name]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"signed calibration input {field_name} must be non-empty")
        for index, value in enumerate(rows):
            validate_entry(value, f"signed_input_inventory.{field_name}[{index}]")
    validate_entry(
        inventory["telemetry_overhead"],
        "signed_input_inventory.telemetry_overhead",
    )
    if "host_envelope" in inventory:
        validate_entry(
            inventory["host_envelope"],
            "signed_input_inventory.host_envelope",
        )


def _validate_scheduler_lifecycle(
    raw_lifecycle: object,
    raw_evidence: object,
) -> tuple[dict[str, str], dict[str, Mapping[str, object]]]:
    lifecycle_mapping = _mapping(raw_lifecycle, "host_scheduler_lifecycle_by_workload")
    evidence_mapping = _mapping(raw_evidence, "host_scheduler_lifecycle_evidence")
    if set(lifecycle_mapping) != set(STAGE052_WORKLOAD_CLASSES) or set(evidence_mapping) != set(
        STAGE052_WORKLOAD_CLASSES
    ):
        raise ValueError("host-scheduler lifecycle evidence is incomplete")
    lifecycle: dict[str, str] = {}
    evidence: dict[str, Mapping[str, object]] = {}
    safety_fields = (
        "session_isolation_passed",
        "cache_reset_passed",
        "rss_stability_passed",
        "semantic_replay_passed",
    )
    for workload_class in STAGE052_WORKLOAD_CLASSES:
        choice = lifecycle_mapping[workload_class]
        if choice not in {"per-wave", "mode-block"}:
            raise ValueError("host-scheduler lifecycle choice is invalid")
        receipt = _mapping(
            evidence_mapping[workload_class],
            f"host_scheduler_lifecycle_evidence.{workload_class}",
        )
        per_wave_seconds = receipt.get("per_wave_end_to_end_seconds_median")
        mode_block_seconds = receipt.get("mode_block_end_to_end_seconds_median")
        sample_count = receipt.get("sample_count")
        if (
            receipt.get("wave_count") != 2
            or not _plain_int(sample_count)
            or cast(int, sample_count) <= 0
            or isinstance(per_wave_seconds, bool)
            or not isinstance(per_wave_seconds, int | float)
            or isinstance(mode_block_seconds, bool)
            or not isinstance(mode_block_seconds, int | float)
            or not math.isfinite(float(per_wave_seconds))
            or not math.isfinite(float(mode_block_seconds))
            or float(per_wave_seconds) <= 0.0
            or float(mode_block_seconds) <= 0.0
        ):
            raise ValueError("host-scheduler lifecycle timing evidence is invalid")
        measured_faster = float(mode_block_seconds) < float(per_wave_seconds)
        if receipt.get("mode_block_faster") is not measured_faster:
            raise ValueError("host-scheduler lifecycle speed decision is inconsistent")
        safety_passed = all(receipt.get(field) is True for field in safety_fields)
        qualified = safety_passed and measured_faster
        if receipt.get("qualified") is not qualified:
            raise ValueError("host-scheduler lifecycle qualification is inconsistent")
        expected_choice = "mode-block" if qualified else "per-wave"
        if choice != expected_choice:
            raise ValueError("host-scheduler lifecycle choice does not match its evidence")
        lifecycle[workload_class] = choice
        evidence[workload_class] = receipt
    return lifecycle, evidence


def bind_frozen_performance_profile(
    profile: FrozenPerformanceProfile,
    build_receipt: Mapping[str, object],
    *,
    observed_host: HostPerformanceEnvelope | None = None,
) -> FrozenRuntimeBinding:
    """Validate all live gates hidden behind the performance-profile interface."""

    observed = detect_host_performance() if observed_host is None else observed_host
    frozen_host = profile.host
    for field_name in ("platform_name", "architecture"):
        if getattr(observed, field_name) != getattr(frozen_host, field_name):
            raise RuntimeError(f"live host {field_name} differs from the frozen profile")
    host_identity = (
        ("allowed CPU set", observed.allowed_cpu_ids, frozen_host.allowed_cpu_ids),
        (
            "physical-core topology",
            observed.physical_core_groups,
            frozen_host.physical_core_groups,
        ),
        ("CPU feature mask", observed.cpu_features, frozen_host.cpu_features),
        ("host memory", observed.memory_total_bytes, frozen_host.memory_total_bytes),
        ("cgroup memory limit", observed.memory_limit_bytes, frozen_host.memory_limit_bytes),
        ("cgroup swap limit", observed.swap_limit_bytes, frozen_host.swap_limit_bytes),
    )
    for name, live_value, frozen_value in host_identity:
        if live_value != frozen_value:
            raise RuntimeError(f"{name} differs from the frozen performance profile")

    identity = profile.selected_build.artifact_identity
    if identity is None:
        raise RuntimeError("frozen performance profile has no build identity")
    expected_build = {
        "build_git_revision": identity.git_revision,
        "build_git_tree": identity.git_tree,
        "build_source_manifest_sha256": identity.source_manifest_sha256,
        "wheel_sha256": identity.wheel_sha256,
        "native_sha256": identity.native_sha256,
        "scheduler_sha256": identity.scheduler_sha256,
        "build_compiler_version": identity.compiler_version,
        "build_performance_profile": profile.selected_build.name,
        "build_compiler_id": profile.selected_build.compiler,
        "build_interprocedural_optimization": profile.selected_build.lto,
        "build_host_native": profile.selected_build.host_native,
    }
    for field_name, expected_value in expected_build.items():
        if build_receipt.get(field_name) != expected_value:
            raise RuntimeError(
                f"installed wheel identity differs from frozen profile: {field_name}"
            )
    if identity.flags != profile.selected_build.flags:
        raise RuntimeError("frozen build flags do not reconcile")
    if identity.cpu_feature_mask != profile.selected_build.cpu_features:
        raise RuntimeError("frozen build CPU feature mask does not reconcile")

    expected_topology_keys = {
        performance_topology_key(mode, workload_class)
        for mode in STAGE052_PERFORMANCE_MODES
        for workload_class in STAGE052_WORKLOAD_CLASSES
    }
    if set(profile.topologies) != expected_topology_keys:
        raise RuntimeError("frozen performance profile topology matrix is incomplete")
    calibration = profile.calibration
    raw_requirements = calibration.get("memory_required_bytes_by_topology")
    if not isinstance(raw_requirements, Mapping) or set(raw_requirements) != expected_topology_keys:
        raise RuntimeError("profile topology memory-admission matrix is incomplete")
    try:
        telemetry_overhead = TelemetryOverheadReceipt.from_dict(
            _mapping(calibration.get("telemetry_overhead"), "telemetry_overhead")
        )
        telemetry_overhead.require_passed()
        telemetry_overhead.require_representative_fixed_work()
        signed_input_inventory = _mapping(
            calibration.get("signed_input_inventory"),
            "signed_input_inventory",
        )
        _validate_signed_input_inventory(signed_input_inventory)
        lifecycle, lifecycle_evidence = _validate_scheduler_lifecycle(
            calibration.get("host_scheduler_lifecycle_by_workload"),
            calibration.get("host_scheduler_lifecycle_evidence"),
        )
    except ValueError as error:
        raise RuntimeError("frozen performance profile calibration gate failed") from error

    allowed_cpu_ids = set(observed.allowed_cpu_ids)
    admissions: dict[str, MemoryAdmission] = {}
    for key in sorted(expected_topology_keys):
        topology = profile.topologies[key]
        mode_name, workload_class = key.split(":", 1)
        if topology.workload_class != workload_class:
            raise RuntimeError("profile topology workload identity is inconsistent")
        if set(topology.cpu_ids) != allowed_cpu_ids:
            raise RuntimeError("profile topology does not use the complete CPU budget")
        if mode_name == "host_scheduler":
            if not topology.scheduler_cpu_ids:
                raise RuntimeError("host scheduler topology has no compute CPU allocation")
        elif topology.scheduler_cpu_ids:
            raise RuntimeError("non-scheduler topology allocates scheduler CPUs")
        required = raw_requirements[key]
        if not _plain_int(required) or cast(int, required) <= 0:
            raise RuntimeError("profile topology memory requirement is invalid")
        admissions[key] = require_memory_admission(
            observed,
            cast(int, required),
            headroom_fraction=0.20,
            require_zero_swap=True,
        )
    return FrozenRuntimeBinding(
        profile_sha256=profile.canonical_sha256,
        observed_host=observed,
        selected_build=profile.selected_build,
        topologies=profile.topologies,
        memory_admission=admissions,
        telemetry_overhead=telemetry_overhead,
        signed_input_inventory=signed_input_inventory,
        scheduler_lifecycle_by_workload=lifecycle,
        scheduler_lifecycle_evidence=lifecycle_evidence,
    )


def freeze_performance_profile(
    host: HostPerformanceEnvelope,
    selected_build: BuildCandidate,
    topologies: Mapping[str, ExecutionTopology],
    *,
    calibration: Mapping[str, object] | None = None,
    selection_reason: str = "deterministic calibration selection",
) -> FrozenPerformanceProfile:
    return FrozenPerformanceProfile(
        host=host,
        selected_build=selected_build,
        topologies=topologies,
        calibration={} if calibration is None else calibration,
        selection_reason=selection_reason,
    )


load_frozen_profile = FrozenPerformanceProfile.load


__all__ = (
    "PERFORMANCE_PROFILE_SCHEMA_VERSION",
    "RUNTIME_RESOURCE_SCHEMA_VERSION",
    "STAGE052_PERFORMANCE_MODES",
    "STAGE052_WORKLOAD_CLASSES",
    "TELEMETRY_OVERHEAD_SCHEMA_VERSION",
    "BuildArtifactIdentity",
    "BuildCandidate",
    "BuildProfile",
    "CalibrationMeasurement",
    "CalibrationObservation",
    "ExecutionTopology",
    "FrozenPerformanceProfile",
    "FrozenRuntimeBinding",
    "HostPerformanceEnvelope",
    "MemoryAdmission",
    "RuntimeResourceSummaryV2",
    "TelemetryOverheadReceipt",
    "build_profile_candidates",
    "bind_frozen_performance_profile",
    "admit_memory",
    "check_memory_admission",
    "detect_host_performance",
    "freeze_performance_profile",
    "generate_execution_topologies",
    "generate_mode_topology_candidates",
    "generate_topology_candidates",
    "load_frozen_profile",
    "partition_cpus",
    "performance_topology_key",
    "require_clean_repository_root",
    "require_memory_admission",
    "select_build_candidate",
    "select_build_profile",
    "validate_memory_admission",
)
