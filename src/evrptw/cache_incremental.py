from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Literal

from evrptw.charging import ChargingSubproblemResult
from evrptw.measurement import canonical_route_key
from evrptw.models import Instance, NodeType
from evrptw.objective import OBJECTIVE_SCHEMA_VERSION

CACHE_INCREMENTAL_SCHEMA_VERSION = "stage032-cache-incremental-v1"
_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class CacheIncrementalConfig:
    """Opt-in Stage 3.2 cache and propagation controls.

    The cache is process-local and owned by one ``solve_alns`` call.  A cache
    entry is valid only for the complete key recorded by :class:`RouteCacheKey`.
    """

    enabled: bool = False
    schema_version: str = CACHE_INCREMENTAL_SCHEMA_VERSION
    max_entries: int = 4096
    max_memory_bytes: int = 64 * 1024 * 1024
    eviction_policy: str = "lru"
    shared_across_lanes: bool = True
    incremental_relocate: bool = True
    incremental_swap: bool = True
    station_reachability_bitset: bool = True
    charging_configuration_version: str = "exact-charging-v1"
    objective_schema_version: str = OBJECTIVE_SCHEMA_VERSION
    instance_hash: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != CACHE_INCREMENTAL_SCHEMA_VERSION:
            raise ValueError(
                "unsupported Stage 3.2 cache schema "
                f"{self.schema_version}; expected {CACHE_INCREMENTAL_SCHEMA_VERSION}"
            )
        if self.max_entries <= 0:
            raise ValueError("cache max_entries must be positive")
        if self.max_memory_bytes <= 0:
            raise ValueError("cache max_memory_bytes must be positive")
        if self.eviction_policy != "lru":
            raise ValueError("Stage 3.2 currently supports only the lru eviction policy")
        if not self.charging_configuration_version:
            raise ValueError("charging_configuration_version must not be empty")
        if self.objective_schema_version != OBJECTIVE_SCHEMA_VERSION:
            raise ValueError(
                "cache objective schema must match the unified objective schema: "
                f"{OBJECTIVE_SCHEMA_VERSION}"
            )


@dataclass(frozen=True, slots=True)
class RouteCacheKey:
    instance_hash: str
    customer_sequence: tuple[str, ...]
    charging_configuration_version: str
    objective_schema_version: str

    @property
    def route_key(self) -> str:
        return canonical_route_key(self.customer_sequence)

    @property
    def digest(self) -> str:
        payload = {
            "instance_hash": self.instance_hash,
            "customer_sequence": self.customer_sequence,
            "charging_configuration_version": self.charging_configuration_version,
            "objective_schema_version": self.objective_schema_version,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CacheLookup:
    key: RouteCacheKey
    hit: bool
    result: ChargingSubproblemResult | None
    entry_bytes: int
    current_entries: int
    current_bytes: int


@dataclass(frozen=True, slots=True)
class CacheStore:
    key: RouteCacheKey
    stored: bool
    entry_bytes: int
    evicted: tuple[RouteCacheKey, ...]
    reason: str
    current_entries: int
    current_bytes: int


@dataclass(slots=True)
class RouteCacheStatistics:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    oversize_not_cached: int = 0
    entries_current: int = 0
    entries_peak: int = 0
    bytes_current: int = 0
    bytes_peak: int = 0
    unique_keys_seen: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def estimate_cache_entry_bytes(result: ChargingSubproblemResult) -> int:
    """Estimate memory using a stable serialized payload plus object overhead."""

    payload = json.dumps(asdict(result), sort_keys=True, separators=(",", ":"))
    return 128 + len(payload.encode("utf-8"))


class RouteEvaluationCache:
    """A bounded, deterministic route-result cache shared by evaluator lanes."""

    def __init__(self, instance: Instance, config: CacheIncrementalConfig) -> None:
        if not config.enabled:
            raise ValueError("RouteEvaluationCache requires an enabled configuration")
        self.instance = instance
        self.config = config
        self.instance_hash = config.instance_hash or canonical_instance_hash(instance)
        self._entries: OrderedDict[RouteCacheKey, tuple[ChargingSubproblemResult, int]] = (
            OrderedDict()
        )
        self._seen_keys: set[RouteCacheKey] = set()
        self.statistics = RouteCacheStatistics()

    def make_key(self, sequence: tuple[str, ...] | list[str]) -> RouteCacheKey:
        return RouteCacheKey(
            self.instance_hash,
            tuple(sequence),
            self.config.charging_configuration_version,
            self.config.objective_schema_version,
        )

    def lookup(self, sequence: tuple[str, ...] | list[str]) -> CacheLookup:
        key = self.make_key(sequence)
        self.statistics.lookups += 1
        self._seen_keys.add(key)
        self.statistics.unique_keys_seen = len(self._seen_keys)
        item = self._entries.get(key)
        if item is None:
            self.statistics.misses += 1
            return CacheLookup(
                key,
                False,
                None,
                0,
                self.statistics.entries_current,
                self.statistics.bytes_current,
            )
        result, entry_bytes = item
        self._entries.move_to_end(key)
        self.statistics.hits += 1
        return CacheLookup(
            key,
            True,
            result,
            entry_bytes,
            self.statistics.entries_current,
            self.statistics.bytes_current,
        )

    def contains(self, sequence: tuple[str, ...] | list[str]) -> bool:
        return self.make_key(sequence) in self._entries

    def store(
        self,
        sequence: tuple[str, ...] | list[str],
        result: ChargingSubproblemResult,
    ) -> CacheStore:
        key = self.make_key(sequence)
        entry_bytes = estimate_cache_entry_bytes(result)
        if entry_bytes > self.config.max_memory_bytes:
            self.statistics.oversize_not_cached += 1
            return CacheStore(
                key,
                False,
                entry_bytes,
                (),
                "entry_exceeds_memory_cap",
                self.statistics.entries_current,
                self.statistics.bytes_current,
            )

        previous = self._entries.pop(key, None)
        if previous is not None:
            self.statistics.bytes_current -= previous[1]
            self.statistics.entries_current -= 1
        evicted: list[RouteCacheKey] = []
        while (
            self._entries
            and (
                self.statistics.entries_current >= self.config.max_entries
                or self.statistics.bytes_current + entry_bytes > self.config.max_memory_bytes
            )
        ):
            old_key, (_old_result, old_bytes) = self._entries.popitem(last=False)
            self.statistics.bytes_current -= old_bytes
            self.statistics.entries_current -= 1
            self.statistics.evictions += 1
            evicted.append(old_key)
        self._entries[key] = (result, entry_bytes)
        self.statistics.stores += 1
        self.statistics.entries_current += 1
        self.statistics.bytes_current += entry_bytes
        self.statistics.entries_peak = max(
            self.statistics.entries_peak, self.statistics.entries_current
        )
        self.statistics.bytes_peak = max(self.statistics.bytes_peak, self.statistics.bytes_current)
        return CacheStore(
            key,
            True,
            entry_bytes,
            tuple(evicted),
            "stored",
            self.statistics.entries_current,
            self.statistics.bytes_current,
        )

    def statistics_dict(self) -> dict[str, int | str]:
        return {
            **self.statistics.to_dict(),
            "eviction_policy": self.config.eviction_policy,
            "max_entries": self.config.max_entries,
            "max_memory_bytes": self.config.max_memory_bytes,
            "instance_hash": self.instance_hash,
        }


def canonical_instance_hash(instance: Instance) -> str:
    payload = {
        "name": instance.name,
        "vehicle": asdict(instance.vehicle),
        "nodes": [asdict(node) for node in instance.nodes],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class RoutePropagationSnapshot:
    sequence: tuple[str, ...]
    chain: tuple[str, ...]
    edge_distances: tuple[float, ...]
    earliest_arrivals: tuple[float, ...]
    latest_departures: tuple[float, ...]
    latest_customer_arrivals: tuple[tuple[str, float], ...]
    total_distance: float
    min_time_window_slack: float
    finish_time: float


@dataclass(frozen=True, slots=True)
class IncrementalPropagationResult:
    status: Literal["incremental", "fallback"]
    reason: str
    base_sequence: tuple[str, ...]
    candidate_sequence: tuple[str, ...]
    distance_lower_bound: float
    min_time_window_slack: float
    finish_time: float
    forward_feasible: bool
    backward_feasible: bool
    first_failed_check: str
    reused_prefix_edges: int
    reused_suffix_edges: int
    recomputed_forward_edges: int
    recomputed_backward_edges: int

    @property
    def accepted(self) -> bool:
        return self.forward_feasible and self.backward_feasible


def build_route_propagation_snapshot(
    instance: Instance,
    sequence: tuple[str, ...] | list[str],
    *,
    epsilon: float = _EPSILON,
) -> RoutePropagationSnapshot:
    values = tuple(sequence)
    by_name = instance.by_name
    chain = (instance.depot.name, *values, instance.depot.name)
    edge_distances = tuple(
        by_name[left].distance_to(by_name[right])
        for left, right in zip(chain, chain[1:], strict=False)
    )
    current_time = max(0.0, instance.depot.ready_time)
    earliest: list[float] = [current_time]
    for index, (_left, right) in enumerate(zip(chain, chain[1:], strict=False), start=1):
        destination = by_name[right]
        current_time += edge_distances[index - 1] / instance.vehicle.average_velocity
        current_time = max(current_time, destination.ready_time)
        earliest.append(current_time)
        if destination.kind is NodeType.CUSTOMER:
            current_time += destination.service_time
    latest_departures: list[float] = [0.0] * len(chain)
    latest_departure = instance.depot.due_date
    latest_departures[-1] = latest_departure
    latest_customer_arrivals: dict[str, float] = {}
    for index in range(len(chain) - 2, -1, -1):
        destination = by_name[chain[index + 1]]
        if destination.kind is NodeType.CUSTOMER:
            latest_arrival = min(destination.due_date, latest_departure - destination.service_time)
            latest_customer_arrivals[destination.name] = latest_arrival
        else:
            latest_arrival = min(destination.due_date, latest_departure)
        latest_departure = (
            latest_arrival - edge_distances[index] / instance.vehicle.average_velocity
        )
        latest_departures[index] = latest_departure
    min_slack = float("inf")
    latest_by_name = latest_customer_arrivals
    for index, name in enumerate(chain):
        node = by_name[name]
        if node.kind is NodeType.CUSTOMER:
            slack = latest_by_name[name] - earliest[index]
            min_slack = min(min_slack, slack)
    if not values:
        min_slack = 0.0
    return RoutePropagationSnapshot(
        values,
        chain,
        edge_distances,
        tuple(earliest),
        tuple(latest_departures),
        tuple(sorted(latest_customer_arrivals.items())),
        sum(edge_distances),
        min_slack,
        current_time,
    )


def incremental_route_propagation(
    instance: Instance,
    base: RoutePropagationSnapshot,
    candidate_sequence: tuple[str, ...] | list[str],
    *,
    epsilon: float = _EPSILON,
) -> IncrementalPropagationResult:
    candidate = tuple(candidate_sequence)
    if base.sequence == candidate:
        return IncrementalPropagationResult(
            "incremental",
            "unchanged_route",
            base.sequence,
            candidate,
            base.total_distance,
            base.min_time_window_slack,
            base.finish_time,
            True,
            True,
            "",
            max(0, len(base.chain) - 1),
            0,
            0,
            0,
        )
    by_name = instance.by_name
    base_chain = base.chain
    candidate_chain = (instance.depot.name, *candidate, instance.depot.name)
    if any(name not in by_name for name in candidate) or len(set(candidate)) != len(candidate):
        return IncrementalPropagationResult(
            "fallback",
            "route_structure_requires_full_propagation",
            base.sequence,
            candidate,
            0.0,
            0.0,
            0.0,
            False,
            False,
            "route_structure",
            0,
            0,
            0,
            0,
        )
    prefix_nodes = 0
    for left, right in zip(base_chain, candidate_chain, strict=False):
        if left != right:
            break
        prefix_nodes += 1
    suffix_nodes = 0
    while (
        suffix_nodes < len(base_chain) - prefix_nodes
        and suffix_nodes < len(candidate_chain) - prefix_nodes
        and base_chain[-1 - suffix_nodes] == candidate_chain[-1 - suffix_nodes]
    ):
        suffix_nodes += 1
    candidate_suffix_start = len(candidate_chain) - suffix_nodes
    prefix_edges = max(0, prefix_nodes - 1)
    suffix_edges = max(0, suffix_nodes - 1)
    candidate_edges = len(candidate_chain) - 1
    middle_start = max(0, prefix_nodes - 1)
    middle_end = max(middle_start, candidate_suffix_start - 1)
    distance = sum(
        by_name[left].distance_to(by_name[right])
        for left, right in zip(
            candidate_chain[middle_start : middle_end + 1],
            candidate_chain[middle_start + 1 : middle_end + 2],
            strict=False,
        )
    )
    if prefix_edges:
        distance += sum(base.edge_distances[:prefix_edges])
    if suffix_edges:
        distance += sum(base.edge_distances[-suffix_edges:])

    earliest = [0.0] * len(candidate_chain)
    earliest[0] = max(0.0, instance.depot.ready_time)
    forward_start = max(0, prefix_nodes - 1)
    if prefix_nodes > 0 and prefix_nodes <= len(base.earliest_arrivals):
        earliest[:prefix_nodes] = base.earliest_arrivals[:prefix_nodes]
    current_time = earliest[forward_start]
    if (
        forward_start < len(candidate_chain)
        and by_name[candidate_chain[forward_start]].kind is NodeType.CUSTOMER
    ):
        current_time += by_name[candidate_chain[forward_start]].service_time
    forward_feasible = True
    for index in range(forward_start, len(candidate_chain) - 1):
        destination = by_name[candidate_chain[index + 1]]
        current_time += (
            by_name[candidate_chain[index]].distance_to(destination)
            / instance.vehicle.average_velocity
        )
        current_time = max(current_time, destination.ready_time)
        earliest[index + 1] = current_time
        if current_time > destination.due_date + epsilon:
            forward_feasible = False
        if destination.kind is NodeType.CUSTOMER:
            current_time += destination.service_time

    latest_departures = [0.0] * len(candidate_chain)
    latest_departures[-1] = instance.depot.due_date
    backward_start = candidate_suffix_start
    if suffix_nodes > 0 and candidate_suffix_start < len(base.latest_departures):
        latest_departures[candidate_suffix_start:] = base.latest_departures[
            len(base.latest_departures) - suffix_nodes :
        ]
    latest_departure = latest_departures[backward_start]
    backward_feasible = True
    for index in range(backward_start - 1, -1, -1):
        origin = by_name[candidate_chain[index]]
        destination = by_name[candidate_chain[index + 1]]
        if destination.kind is NodeType.CUSTOMER:
            latest_arrival = min(destination.due_date, latest_departure - destination.service_time)
        else:
            latest_arrival = min(destination.due_date, latest_departure)
        latest_departure = (
            latest_arrival - origin.distance_to(destination) / instance.vehicle.average_velocity
        )
        latest_departures[index] = latest_departure
        if (
            origin.kind is NodeType.CUSTOMER
            and latest_departure < origin.ready_time - epsilon
        ):
            backward_feasible = False

    latest_customer_arrivals: dict[str, float] = {}
    for index in range(len(candidate_chain) - 1):
        destination = by_name[candidate_chain[index + 1]]
        if destination.kind is NodeType.CUSTOMER:
            latest_customer_arrivals[destination.name] = min(
                destination.due_date,
                latest_departures[index + 1] - destination.service_time,
            )
    min_slack = min(
        (
            latest_customer_arrivals[name] - earliest[index]
            for index, name in enumerate(candidate_chain)
            if by_name[name].kind is NodeType.CUSTOMER
        ),
        default=0.0,
    )
    first_failed = ""
    reason = "incremental_propagation"
    if not forward_feasible:
        first_failed = "forward_time_window"
        reason = "forward_time_window_prefilter"
    elif not backward_feasible:
        first_failed = "backward_time_window"
        reason = "backward_time_window_prefilter"
    elif min_slack < -epsilon:
        first_failed = "time_window_slack"
        reason = "time_window_slack_prefilter"
    return IncrementalPropagationResult(
        "incremental",
        reason,
        base.sequence,
        candidate,
        distance,
        min_slack,
        current_time,
        forward_feasible,
        backward_feasible,
        first_failed,
        prefix_edges,
        suffix_edges,
        max(0, candidate_edges - prefix_edges),
        max(0, candidate_edges - suffix_edges),
    )


class StationReachabilityIndex:
    """Precompute optimistic station reachability as integer bitsets."""

    def __init__(self, instance: Instance, *, epsilon: float = _EPSILON) -> None:
        self.instance = instance
        self.epsilon = epsilon
        self.safe_nodes = (instance.depot, *instance.stations)
        self._safe_index = {node.name: index for index, node in enumerate(self.safe_nodes)}
        self._reachable_masks = self._build_masks()
        self.queries = 0

    def _build_masks(self) -> dict[str, int]:
        masks: dict[str, int] = {
            node.name: 1 << index for index, node in enumerate(self.safe_nodes)
        }
        for origin in self.safe_nodes:
            frontier = [origin]
            visited = {origin.name}
            while frontier:
                current = frontier.pop()
                for station in self.instance.stations:
                    if station.name in visited:
                        continue
                    if current.distance_to(station) * self.instance.vehicle.consumption_rate <= (
                        self.instance.vehicle.battery_capacity + self.epsilon
                    ):
                        visited.add(station.name)
                        masks[origin.name] |= 1 << self._safe_index[station.name]
                        frontier.append(station)
        return masks

    def bitset_for(self, origin_name: str) -> int:
        if origin_name not in self._reachable_masks:
            raise ValueError(f"reachability origin is not a depot/station: {origin_name}")
        return self._reachable_masks[origin_name]

    def can_reach(self, origin_name: str, destination_name: str) -> bool:
        self.queries += 1
        by_name = self.instance.by_name
        origin = by_name[origin_name]
        destination = by_name[destination_name]
        if origin_name not in self._safe_index:
            return origin.distance_to(destination) * self.instance.vehicle.consumption_rate <= (
                self.instance.vehicle.battery_capacity + self.epsilon
            )
        if origin.distance_to(destination) * self.instance.vehicle.consumption_rate <= (
            self.instance.vehicle.battery_capacity + self.epsilon
        ):
            return True
        mask = self.bitset_for(origin_name)
        return any(
            mask & (1 << index)
            and node.distance_to(destination) * self.instance.vehicle.consumption_rate
            <= self.instance.vehicle.battery_capacity + self.epsilon
            for index, node in enumerate(self.safe_nodes)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "safe_nodes": [node.name for node in self.safe_nodes],
            "bitsets": {name: mask for name, mask in self._reachable_masks.items()},
            "queries": self.queries,
        }
