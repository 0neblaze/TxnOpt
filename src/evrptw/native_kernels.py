"""Opt-in Stage 5.2 native-kernel configuration.

The configuration object is deliberately separate from the historical
``cpu_batch`` backend.  Passing ``None`` at solver seams preserves every
pre-Stage-5.2 Python path; a configured runtime is all-or-nothing and must not
silently fall back after a native schema, deadline, or execution failure.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from evrptw.models import Instance, NodeType

NATIVE_KERNEL_ABI_VERSION = "stage05.2-native-kernels-v2"
LEGACY_NATIVE_KERNEL_ABI_VERSION = "stage05.2-native-kernels-v1"
SUPPORTED_NATIVE_KERNEL_ABI_VERSIONS = frozenset(
    {LEGACY_NATIVE_KERNEL_ABI_VERSION, NATIVE_KERNEL_ABI_VERSION}
)


@dataclass(frozen=True, slots=True)
class NativeKernelConfig:
    """Explicit, auditable selection of the Stage 5.2 native CPU kernels."""

    enabled: bool = True
    exact_charging: bool = True
    screening: bool = True
    propagation: bool = True
    distance_matrix: bool = True
    abi_version: str = NATIVE_KERNEL_ABI_VERSION
    context_policy: str = "pack_once_per_solve"
    failure_policy: str = "fail_fast_no_fallback"

    def __post_init__(self) -> None:
        if not self.enabled:
            raise ValueError("disabled native configuration is ambiguous; pass None instead")
        if self.abi_version not in SUPPORTED_NATIVE_KERNEL_ABI_VERSIONS:
            raise ValueError(
                "native kernel ABI must be one of "
                f"{sorted(SUPPORTED_NATIVE_KERNEL_ABI_VERSIONS)!r}"
            )
        if not all(
            (
                self.exact_charging,
                self.screening,
                self.propagation,
                self.distance_matrix,
            )
        ):
            raise ValueError("Stage 5.2 evidence requires the complete native kernel set")
        if self.context_policy != "pack_once_per_solve":
            raise ValueError("native context must be packed once per solve")
        if self.failure_policy != "fail_fast_no_fallback":
            raise ValueError("native failures must not fall back")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


Float64Array = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class NativeInstanceContext:
    """C-contiguous, per-instance numeric inputs shared by every ALNS lane."""

    _source_instance: Instance = field(repr=False, compare=False)
    instance_name: str
    node_names: tuple[str, ...]
    name_to_index: Mapping[str, int]
    node_kind: Int64Array
    demand: Float64Array
    ready_time: Float64Array
    due_date: Float64Array
    service_time: Float64Array
    distance: Float64Array
    reachable: npt.NDArray[np.uint8]
    legacy_reachable: npt.NDArray[np.uint8]
    vehicle: Float64Array
    reachability_epsilon: float
    packing_seconds: float

    def assert_matches(self, instance: Instance) -> None:
        names = tuple(node.name for node in instance.nodes)
        if (
            self._source_instance is not instance
            or self.instance_name != instance.name
            or self.node_names != names
        ):
            raise ValueError("native instance context does not match the requested instance")


@dataclass(slots=True)
class NativeKernelRuntime:
    """One solve-local native runtime; the packed instance context is never rebuilt."""

    config: NativeKernelConfig
    context: NativeInstanceContext
    _packing_claimed: bool = False
    screening_invocations: int = 0
    propagation_invocations: int = 0
    screening_batch_invocations: int = 0
    screening_batch_candidates: int = 0
    screening_batch_occupancies: list[int] = field(default_factory=list)
    screening_seconds: float = 0.0
    propagation_seconds: float = 0.0
    fallback_count: int = 0

    @classmethod
    def build(
        cls,
        instance: Instance,
        config: NativeKernelConfig,
    ) -> NativeKernelRuntime:
        return cls(config=config, context=pack_native_instance(instance))

    def claim_context_packing_seconds(self) -> float:
        if self._packing_claimed:
            return 0.0
        self._packing_claimed = True
        return self.context.packing_seconds

    def record_screening(
        self,
        elapsed_seconds: float,
        *,
        batch_candidates: int | None = None,
    ) -> None:
        self.screening_invocations += 1
        self.screening_seconds += elapsed_seconds
        if batch_candidates is not None:
            if batch_candidates < 0:
                raise ValueError("native screening batch occupancy cannot be negative")
            self.screening_batch_invocations += 1
            self.screening_batch_candidates += batch_candidates
            self.screening_batch_occupancies.append(batch_candidates)

    def record_propagation(self, elapsed_seconds: float) -> None:
        self.propagation_invocations += 1
        self.propagation_seconds += elapsed_seconds

    def statistics(self) -> dict[str, object]:
        return {
            "native_screening_invocations": self.screening_invocations,
            "native_propagation_invocations": self.propagation_invocations,
            "native_screening_batch_invocations": self.screening_batch_invocations,
            "native_screening_batch_candidates": self.screening_batch_candidates,
            "native_screening_batch_occupancies": tuple(
                self.screening_batch_occupancies
            ),
            "native_screening_seconds": self.screening_seconds,
            "native_propagation_seconds": self.propagation_seconds,
            "native_protocol_fallbacks": self.fallback_count,
        }


def pack_native_instance(instance: Instance) -> NativeInstanceContext:
    """Pack immutable numeric solver context once, before any exact invocation."""

    started = time.perf_counter()
    names = tuple(node.name for node in instance.nodes)
    index = {name: position for position, name in enumerate(names)}
    kind_codes = {
        NodeType.DEPOT: 0,
        NodeType.CUSTOMER: 1,
        NodeType.STATION: 2,
    }
    node_kind = np.ascontiguousarray(
        [kind_codes[node.kind] for node in instance.nodes],
        dtype=np.int64,
    )
    demand = np.ascontiguousarray(
        [node.demand for node in instance.nodes],
        dtype=np.float64,
    )
    ready_time = np.ascontiguousarray(
        [node.ready_time for node in instance.nodes],
        dtype=np.float64,
    )
    due_date = np.ascontiguousarray(
        [node.due_date for node in instance.nodes],
        dtype=np.float64,
    )
    service_time = np.ascontiguousarray(
        [node.service_time for node in instance.nodes],
        dtype=np.float64,
    )
    distance = np.ascontiguousarray(
        [
            [instance.distance(origin.name, destination.name) for destination in instance.nodes]
            for origin in instance.nodes
        ],
        dtype=np.float64,
    )
    reachability_epsilon = 1e-9
    reachable = _optimistic_reachability_matrix(
        node_kind,
        distance,
        battery_capacity=instance.vehicle.battery_capacity,
        consumption_rate=instance.vehicle.consumption_rate,
        epsilon=reachability_epsilon,
        include_depot_as_intermediate=True,
    )
    legacy_reachable = _optimistic_reachability_matrix(
        node_kind,
        distance,
        battery_capacity=instance.vehicle.battery_capacity,
        consumption_rate=instance.vehicle.consumption_rate,
        epsilon=reachability_epsilon,
        include_depot_as_intermediate=False,
    )
    vehicle = np.ascontiguousarray(
        [
            instance.vehicle.battery_capacity,
            instance.vehicle.load_capacity,
            instance.vehicle.consumption_rate,
            instance.vehicle.inverse_refueling_rate,
            instance.vehicle.average_velocity,
        ],
        dtype=np.float64,
    )
    for array in (
        node_kind,
        demand,
        ready_time,
        due_date,
        service_time,
        distance,
        reachable,
        legacy_reachable,
        vehicle,
    ):
        array.flags.writeable = False
    return NativeInstanceContext(
        _source_instance=instance,
        instance_name=instance.name,
        node_names=names,
        name_to_index=MappingProxyType(index),
        node_kind=node_kind,
        demand=demand,
        ready_time=ready_time,
        due_date=due_date,
        service_time=service_time,
        distance=distance,
        reachable=reachable,
        legacy_reachable=legacy_reachable,
        vehicle=vehicle,
        reachability_epsilon=reachability_epsilon,
        packing_seconds=time.perf_counter() - started,
    )


def _optimistic_reachability_matrix(
    node_kind: Int64Array,
    distance: Float64Array,
    *,
    battery_capacity: float,
    consumption_rate: float,
    epsilon: float,
    include_depot_as_intermediate: bool,
) -> npt.NDArray[np.uint8]:
    """Materialise the Stage 3.1 safe recharge-frontier relation once."""

    node_count = int(node_kind.shape[0])
    safe = tuple(
        int(index)
        for index in np.flatnonzero(
            node_kind != 1 if include_depot_as_intermediate else node_kind == 2
        )
    )
    direct = distance * consumption_rate <= battery_capacity + epsilon
    if not safe:
        output = np.asarray(direct, dtype=np.uint8)
        np.fill_diagonal(output, 1)
        return np.ascontiguousarray(output)
    safe_closure = np.asarray(direct[np.ix_(safe, safe)], dtype=np.bool_)
    np.fill_diagonal(safe_closure, True)
    for pivot in range(len(safe)):
        safe_closure |= safe_closure[:, pivot, None] & safe_closure[None, pivot, :]

    output = np.zeros((node_count, node_count), dtype=np.uint8)
    for origin in range(node_count):
        reachable_safe = np.asarray(direct[origin, safe], dtype=np.bool_)
        if reachable_safe.any():
            reachable_safe = reachable_safe @ safe_closure
        via_safe = np.zeros(node_count, dtype=np.bool_)
        if reachable_safe.any():
            via_safe = np.any(
                reachable_safe[:, None] & direct[np.asarray(safe), :],
                axis=0,
            )
        output[origin] = np.asarray(direct[origin] | via_safe, dtype=np.uint8)
        output[origin, origin] = 1
    return np.ascontiguousarray(output)
