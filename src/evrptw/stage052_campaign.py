"""Public Stage 5.2 Formal benchmark campaign contracts.

The campaign layer deliberately plans immutable ``(instance, seed)`` shards
without starting solver work.  It is therefore safe for runner preflight and
for the independent reviewer to use the same public geometry while still
recomputing all identities from canonical Stage 5.1 data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, Protocol, cast

from evrptw.artifacts import atomic_write_signed_json, signed_sidecar_matches
from evrptw.best_known import BEST_KNOWN_VALUES
from evrptw.objective import ObjectiveComparison, SolutionObjective, compare_objectives
from evrptw.stage052 import STAGE052_MAXIMUM_PERSISTENCE_RATIO
from evrptw.stage052_evidence import BatchPersistenceEnvelope, PersistenceInterval
from evrptw.stage052_resources import ProducerResourceContract

GIB: Final = 1024**3
FORMAL_SEEDS: Final = tuple(range(2014, 2024))
PILOT_SEEDS: Final = (2014, 2015, 2016)
PILOT_INSTANCE_NAMES: Final = (
    "c101C5",
    "r105C5",
    "rc105C5",
    "c104C10",
    "r103C10",
    "rc102C10",
    "c106C15",
    "r105C15",
    "rc103C15",
    "c101_21",
    "r101_21",
    "rc101_21",
)
SMALL_BUDGETS: Final = (30,)
LARGE_BUDGETS: Final = (30, 60, 300)
CHECKPOINT_SECONDS: Final = (1, 5, 10, 30, 60, 120, 300)
type ObjectiveKey = tuple[int, float, float, int]
_CANONICAL_CUSTOMER_COUNTS: Final = {
    record.instance: record.customer_count for record in BEST_KNOWN_VALUES
}
_PILOT_CUSTOMER_COUNTS: Final = {
    instance: _CANONICAL_CUSTOMER_COUNTS[instance] for instance in PILOT_INSTANCE_NAMES
}
_CAMPAIGN_GEOMETRY: Final = {
    "pilot": (36, 36, 1_080, 144),
    "formal": (920, 2_040, 229_200, 10_400),
}


@dataclass(frozen=True, slots=True)
class RuntimeLoadPolicy:
    """Frozen host-capacity policy for Stage 5.2 campaign execution."""

    logical_cpu_count: int
    preflight_maximum_load1: float
    runtime_maximum_load1: float
    maximum_unrelated_process_average_cores: float


RUNTIME_LOAD_POLICY: Final = RuntimeLoadPolicy(
    logical_cpu_count=24,
    preflight_maximum_load1=4.0,
    runtime_maximum_load1=32.0,
    maximum_unrelated_process_average_cores=4.0,
)


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _object_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    raw = cast(Mapping[object, object], value)
    if any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{field} keys must be strings")
    return {str(key): item for key, item in raw.items()}


def _exact_fields(payload: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{field} fields do not match its schema")


def _required_str(payload: Mapping[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _optional_str(payload: Mapping[str, object], field: str) -> str | None:
    value = payload.get(field)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field} must be a string or null")
    return value


def _required_int(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if not _is_plain_int(value):
        raise ValueError(f"{field} must be an integer")
    return cast(int, value)


def _optional_int(payload: Mapping[str, object], field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if not _is_plain_int(value):
        raise ValueError(f"{field} must be an integer or null")
    return cast(int, value)


def _required_number(payload: Mapping[str, object], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _optional_number(payload: Mapping[str, object], field: str) -> float | None:
    if payload.get(field) is None:
        return None
    return _required_number(payload, field)


def _string_tuple(payload: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be an array of strings")
    return tuple(value)


def _optional_string_mapping(payload: Mapping[str, object], field: str) -> Mapping[str, str] | None:
    value = payload.get(field)
    if value is None:
        return None
    result = _object_mapping(value, field)
    if any(not isinstance(item, str) for item in result.values()):
        raise ValueError(f"{field} values must be strings")
    return {key: cast(str, item) for key, item in result.items()}


def _optional_int_mapping(payload: Mapping[str, object], field: str) -> Mapping[str, int] | None:
    value = payload.get(field)
    if value is None:
        return None
    result = _object_mapping(value, field)
    if any(not _is_plain_int(item) for item in result.values()):
        raise ValueError(f"{field} values must be integers")
    return {key: cast(int, item) for key, item in result.items()}


def _objective_from_key(value: ObjectiveKey) -> SolutionObjective:
    if (
        len(value) != 4
        or not _is_plain_int(value[0])
        or value[0] <= 0
        or not math.isfinite(value[1])
        or value[1] < 0.0
        or not math.isfinite(value[2])
        or value[2] < 0.0
        or not _is_plain_int(value[3])
        or value[3] < 0
    ):
        raise ValueError("objective_key must be a valid four-component objective")
    objective = SolutionObjective(
        vehicle_count=value[0],
        total_distance=value[1],
        total_charging_time=value[2],
        charging_count=value[3],
    )
    if objective.key != value:
        raise ValueError("objective_key must use the canonical objective precision")
    return objective


def _objective_key_from_parts(
    *,
    vehicle_count: int,
    total_distance: float,
    total_charging_time: float,
    charging_count: int,
) -> ObjectiveKey:
    objective = SolutionObjective(
        vehicle_count=vehicle_count,
        total_distance=total_distance,
        total_charging_time=total_charging_time,
        charging_count=charging_count,
    )
    if objective.vehicle_count <= 0:
        raise ValueError("objective_key vehicle_count must be positive")
    return objective.key


@dataclass(frozen=True, slots=True)
class AcceptedGlobalBest:
    """One fully completed and accepted global-best event."""

    completed_at_seconds: float
    iteration: int
    objective_key: ObjectiveKey

    def __post_init__(self) -> None:
        if not math.isfinite(self.completed_at_seconds) or self.completed_at_seconds < 0.0:
            raise ValueError("completed_at_seconds must be finite and non-negative")
        if not _is_plain_int(self.iteration) or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer")
        _objective_from_key(self.objective_key)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AcceptedGlobalBest:
        _exact_fields(
            payload,
            {"completed_at_seconds", "iteration", "objective_key"},
            "accepted global best",
        )
        objective = payload.get("objective_key")
        if not isinstance(objective, list) or len(objective) != 4:
            raise ValueError("objective_key must contain four values")
        objective_payload = {
            "vehicles": objective[0],
            "distance": objective[1],
            "charging_time": objective[2],
            "charging_count": objective[3],
        }
        return cls(
            completed_at_seconds=_required_number(payload, "completed_at_seconds"),
            iteration=_required_int(payload, "iteration"),
            objective_key=_objective_key_from_parts(
                vehicle_count=_required_int(objective_payload, "vehicles"),
                total_distance=_required_number(objective_payload, "distance"),
                total_charging_time=_required_number(objective_payload, "charging_time"),
                charging_count=_required_int(objective_payload, "charging_count"),
            ),
        )


@dataclass(frozen=True, slots=True)
class AnytimeCheckpoint:
    """Objective visible at a declared wall-clock checkpoint."""

    instance: str
    seed: int
    axis_budget_seconds: int
    checkpoint_seconds: int
    objective_key: ObjectiveKey
    source: str
    incumbent_completed_at_seconds: float
    incumbent_iteration: int | None

    def __post_init__(self) -> None:
        if self.instance not in _CANONICAL_CUSTOMER_COUNTS:
            raise ValueError("checkpoint instance is not canonical")
        if not _is_plain_int(self.seed) or self.seed not in FORMAL_SEEDS:
            raise ValueError("checkpoint seed must be in the Formal seed scope")
        if (
            not _is_plain_int(self.axis_budget_seconds)
            or self.axis_budget_seconds not in LARGE_BUDGETS
        ):
            raise ValueError("checkpoint axis budget must be 30, 60, or 300 seconds")
        if _CANONICAL_CUSTOMER_COUNTS[self.instance] < 100 and self.axis_budget_seconds != 30:
            raise ValueError("small-instance checkpoint axis must use 30 seconds")
        if (
            not _is_plain_int(self.checkpoint_seconds)
            or self.checkpoint_seconds not in CHECKPOINT_SECONDS
            or self.checkpoint_seconds > self.axis_budget_seconds
        ):
            raise ValueError("checkpoint is outside its axis budget")
        if self.source not in {
            "verified_initial_incumbent",
            "accepted_global_best",
            "final_incumbent_carry_forward",
        }:
            raise ValueError("unsupported checkpoint source")
        if not 0.0 <= self.incumbent_completed_at_seconds <= self.checkpoint_seconds:
            raise ValueError("checkpoint incumbent was not complete at the boundary")
        _objective_from_key(self.objective_key)

    @classmethod
    def for_axis(
        cls,
        *,
        instance: str,
        seed: int,
        axis_budget_seconds: int,
        initial_objective_key: ObjectiveKey,
        accepted_global_bests: tuple[AcceptedGlobalBest, ...],
        max_iterations_completed_at_seconds: float | None = None,
        final_objective_key: ObjectiveKey | None = None,
    ) -> tuple[AnytimeCheckpoint, ...]:
        """Materialize checkpoints from complete accepted incumbent events."""

        previous_objective = _objective_from_key(initial_objective_key)
        previous_time = -1.0
        previous_iteration = -1
        for event in accepted_global_bests:
            if event.completed_at_seconds <= previous_time:
                raise ValueError(
                    "accepted global-best events require strictly increasing completion time"
                )
            if event.iteration < previous_iteration:
                raise ValueError("accepted global-best events require non-decreasing iteration")
            if event.completed_at_seconds > axis_budget_seconds:
                raise ValueError("accepted global-best event exceeds the axis budget")
            event_objective = _objective_from_key(event.objective_key)
            if (
                compare_objectives(event_objective, previous_objective)
                is not ObjectiveComparison.BETTER
            ):
                raise ValueError(
                    "accepted global-best history requires strict objective improvement"
                )
            previous_time = event.completed_at_seconds
            previous_iteration = event.iteration
            previous_objective = event_objective

        if max_iterations_completed_at_seconds is None:
            if final_objective_key is not None:
                raise ValueError("final_objective_key requires max-iteration completion")
        else:
            if _CANONICAL_CUSTOMER_COUNTS[instance] == 100:
                raise ValueError("max-iteration carry-forward is only for small instances")
            if (
                not math.isfinite(max_iterations_completed_at_seconds)
                or not 0.0 <= max_iterations_completed_at_seconds <= axis_budget_seconds
            ):
                raise ValueError("max-iteration completion is outside the axis budget")
            if final_objective_key is None:
                raise ValueError("max-iteration completion requires a final incumbent")
            if accepted_global_bests and (
                accepted_global_bests[-1].completed_at_seconds > max_iterations_completed_at_seconds
            ):
                raise ValueError("accepted global best occurs after max-iteration completion")
            final_objective = _objective_from_key(final_objective_key)
            expected_final = (
                accepted_global_bests[-1].objective_key
                if accepted_global_bests
                else initial_objective_key
            )
            if (
                compare_objectives(
                    final_objective,
                    _objective_from_key(expected_final),
                )
                is not ObjectiveComparison.EQUAL
            ):
                raise ValueError("final incumbent does not match verified incumbent history")

        checkpoints: list[AnytimeCheckpoint] = []
        for checkpoint in CHECKPOINT_SECONDS:
            if checkpoint > axis_budget_seconds:
                continue
            visible = tuple(
                event for event in accepted_global_bests if event.completed_at_seconds <= checkpoint
            )
            if visible:
                selected_objective = visible[-1].objective_key
                selected_time = visible[-1].completed_at_seconds
                selected_iteration: int | None = visible[-1].iteration
                source = "accepted_global_best"
            else:
                selected_objective = initial_objective_key
                selected_time = 0.0
                selected_iteration = None
                source = "verified_initial_incumbent"
            if (
                max_iterations_completed_at_seconds is not None
                and max_iterations_completed_at_seconds <= checkpoint
            ):
                assert final_objective_key is not None
                selected_objective = final_objective_key
                selected_time = visible[-1].completed_at_seconds if visible else 0.0
                selected_iteration = visible[-1].iteration if visible else None
                source = "final_incumbent_carry_forward"
            checkpoints.append(
                cls(
                    instance=instance,
                    seed=seed,
                    axis_budget_seconds=axis_budget_seconds,
                    checkpoint_seconds=checkpoint,
                    objective_key=selected_objective,
                    source=source,
                    incumbent_completed_at_seconds=selected_time,
                    incumbent_iteration=selected_iteration,
                )
            )
        return tuple(checkpoints)

    def to_dict(self) -> dict[str, object]:
        return {
            "instance": self.instance,
            "seed": self.seed,
            "axis_budget_seconds": self.axis_budget_seconds,
            "checkpoint_seconds": self.checkpoint_seconds,
            "objective_key": list(self.objective_key),
            "source": self.source,
            "incumbent_completed_at_seconds": self.incumbent_completed_at_seconds,
            "incumbent_iteration": self.incumbent_iteration,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AnytimeCheckpoint:
        _exact_fields(
            payload,
            {
                "instance",
                "seed",
                "axis_budget_seconds",
                "checkpoint_seconds",
                "objective_key",
                "source",
                "incumbent_completed_at_seconds",
                "incumbent_iteration",
            },
            "anytime checkpoint",
        )
        objective = payload.get("objective_key")
        if isinstance(objective, list) and len(objective) == 4:
            objective_payload = {
                "vehicles": objective[0],
                "distance": objective[1],
                "charging_time": objective[2],
                "charging_count": objective[3],
            }
        elif isinstance(objective, Mapping):
            objective_payload = {
                "vehicles": objective.get("vehicle_count"),
                "distance": objective.get("total_distance"),
                "charging_time": objective.get("total_charging_time"),
                "charging_count": objective.get("charging_count"),
            }
        else:
            raise ValueError("objective_key must contain four values")
        return cls(
            instance=_required_str(payload, "instance"),
            seed=_required_int(payload, "seed"),
            axis_budget_seconds=_required_int(payload, "axis_budget_seconds"),
            checkpoint_seconds=_required_int(payload, "checkpoint_seconds"),
            objective_key=_objective_key_from_parts(
                vehicle_count=_required_int(objective_payload, "vehicles"),
                total_distance=_required_number(objective_payload, "distance"),
                total_charging_time=_required_number(objective_payload, "charging_time"),
                charging_count=_required_int(objective_payload, "charging_count"),
            ),
            source=_required_str(payload, "source"),
            incumbent_completed_at_seconds=_required_number(
                payload, "incumbent_completed_at_seconds"
            ),
            incumbent_iteration=_optional_int(payload, "incumbent_iteration"),
        )


@dataclass(frozen=True, slots=True)
class VolumeIdentity:
    """Stable volume identity recorded without a machine-specific path."""

    device_uuid: str
    filesystem: str

    def __post_init__(self) -> None:
        if not self.device_uuid or not self.filesystem:
            raise ValueError("device_uuid and filesystem are required")

    def to_dict(self) -> dict[str, str]:
        return {
            "device_uuid": self.device_uuid,
            "filesystem": self.filesystem,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> VolumeIdentity:
        _exact_fields(payload, {"device_uuid", "filesystem"}, "volume identity")
        return cls(
            device_uuid=_required_str(payload, "device_uuid"),
            filesystem=_required_str(payload, "filesystem"),
        )


@dataclass(frozen=True, slots=True)
class StorageRoot:
    """One local-only alias-to-path binding."""

    alias: str
    absolute_path: Path
    volume: VolumeIdentity

    def __post_init__(self) -> None:
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.alias) is None:
            raise ValueError(f"invalid storage root alias: {self.alias}")
        if not self.absolute_path.is_absolute():
            raise ValueError(f"storage root path must be absolute: {self.alias}")

    def tracked_identity(self) -> dict[str, str]:
        """Return the path-free identity allowed in tracked artifacts."""

        return self.volume.to_dict()


class StorageRootLocator:
    """Resolve ignored local root aliases and verify their volume identities."""

    __slots__ = ("_roots",)

    def __init__(self, roots: Mapping[str, StorageRoot]) -> None:
        if not roots:
            raise ValueError("at least one storage root is required")
        copied = dict(roots)
        if any(alias != root.alias for alias, root in copied.items()):
            raise ValueError("storage root mapping key and alias disagree")
        self._roots: Mapping[str, StorageRoot] = MappingProxyType(copied)

    @classmethod
    def from_toml(cls, path: Path) -> StorageRootLocator:
        """Read a local-only root locator configuration."""

        try:
            with path.open("rb") as handle:
                payload = tomllib.load(handle)
            roots_payload = payload["roots"]
        except (OSError, KeyError, tomllib.TOMLDecodeError) as error:
            raise ValueError(f"cannot read storage root locator: {path}") from error
        if not isinstance(roots_payload, dict) or not roots_payload:
            raise ValueError("storage root locator must contain a non-empty roots table")
        roots: dict[str, StorageRoot] = {}
        for alias, item in roots_payload.items():
            if not isinstance(alias, str) or not isinstance(item, dict):
                raise ValueError("storage root entries must be TOML tables")
            try:
                absolute_path = Path(str(item["absolute_path"]))
                volume = VolumeIdentity(
                    device_uuid=str(item["device_uuid"]),
                    filesystem=str(item["filesystem"]),
                )
            except KeyError as error:
                raise ValueError(f"storage root is incomplete: {alias}") from error
            roots[alias] = StorageRoot(alias, absolute_path, volume)
        return cls(roots)

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(sorted(self._roots))

    def resolve(self, alias: str) -> StorageRoot:
        try:
            return self._roots[alias]
        except KeyError as error:
            raise KeyError(f"unknown storage root alias: {alias}") from error

    def tracked_payload(self, aliases: tuple[str, ...] | None = None) -> dict[str, object]:
        selected = self.aliases if aliases is None else aliases
        if len(set(selected)) != len(selected):
            raise ValueError("tracked storage root aliases must be unique")
        return {
            "schema_version": "stage05.2-storage-roots-v1",
            "roots": {alias: self.resolve(alias).tracked_identity() for alias in selected},
        }

    def verify_all(
        self,
        probe: Callable[[Path], VolumeIdentity],
        aliases: tuple[str, ...] | None = None,
    ) -> None:
        """Fail if a configured path is now mounted on another volume."""

        selected = self.aliases if aliases is None else aliases
        for alias in selected:
            root = self.resolve(alias)
            observed = probe(root.absolute_path)
            if observed != root.volume:
                raise RuntimeError(
                    "storage volume identity mismatch for "
                    f"{alias}: expected={root.volume.to_dict()} "
                    f"observed={observed.to_dict()}"
                )

    def with_observed_volumes(
        self,
        probe: Callable[[Path], VolumeIdentity],
        aliases: tuple[str, ...] | None = None,
    ) -> StorageRootLocator:
        """Return an operational locator refreshed from non-publication telemetry."""

        selected = set(self.aliases if aliases is None else aliases)
        if not selected or not selected.issubset(self._roots):
            raise ValueError("observed storage root aliases are invalid")
        refreshed = dict(self._roots)
        for alias in selected:
            root = self.resolve(alias)
            refreshed[alias] = replace(root, volume=probe(root.absolute_path))
        return StorageRootLocator(refreshed)


def _instance_family(instance: str) -> str:
    normalized = instance.lower()
    if normalized.startswith("rc"):
        return "RC"
    if normalized.startswith("r"):
        return "R"
    if normalized.startswith("c"):
        return "C"
    raise ValueError(f"unknown Schneider instance family: {instance}")


@dataclass(frozen=True, slots=True)
class CampaignInstance:
    """Canonical identity needed to plan one benchmark instance."""

    instance: str
    customer_count: int
    family: str

    def __post_init__(self) -> None:
        if not self.instance:
            raise ValueError("instance is required")
        if self.customer_count not in {5, 10, 15, 100}:
            raise ValueError("customer_count must be 5, 10, 15, or 100")
        if self.family not in {"C", "R", "RC"}:
            raise ValueError("family must be C, R, or RC")
        if self.family != _instance_family(self.instance):
            raise ValueError("instance and family disagree")


def _canonical_instances() -> tuple[CampaignInstance, ...]:
    return tuple(
        CampaignInstance(
            instance=record.instance,
            customer_count=record.customer_count,
            family=_instance_family(record.instance),
        )
        for record in BEST_KNOWN_VALUES
    )


def _pilot_instances() -> tuple[CampaignInstance, ...]:
    canonical = {item.instance: item for item in _canonical_instances()}
    return tuple(canonical[instance] for instance in PILOT_INSTANCE_NAMES)


@dataclass(frozen=True, slots=True)
class PilotStorageObservation:
    """Maximum-compressed-byte input observed by the accepted G01 pilot."""

    family: str
    customer_count: int
    budget_seconds: int
    compressed_bytes: int

    def __post_init__(self) -> None:
        if self.family not in {"C", "R", "RC"}:
            raise ValueError("pilot family must be C, R, or RC")
        if self.customer_count not in {5, 10, 15, 100}:
            raise ValueError("pilot customer_count must be 5, 10, 15, or 100")
        if not _is_plain_int(self.budget_seconds) or self.budget_seconds <= 0:
            raise ValueError("pilot budget_seconds must be positive")
        if not _is_plain_int(self.compressed_bytes) or self.compressed_bytes <= 0:
            raise ValueError("pilot compressed_bytes must be positive")


@dataclass(frozen=True, slots=True)
class ShardPlan:
    """One indivisible ``(instance, seed)`` Pilot or Formal campaign shard."""

    shard_id: str
    instance: str
    seed: int
    customer_count: int
    family: str
    budgets_seconds: tuple[int, ...]
    checkpoint_seconds: tuple[int, ...]
    estimated_bytes: int
    max_iterations: int | None = None
    scope: str = "formal"

    def __post_init__(self) -> None:
        match = re.fullmatch(r"shard([0-9]{4})", self.shard_id)
        if match is None or not 1 <= int(match.group(1)) <= 920:
            raise ValueError("shard_id must be canonical shard0001 through shard0920")
        if self.scope not in _CAMPAIGN_GEOMETRY:
            raise ValueError("shard scope must be pilot or formal")
        expected_count = _CANONICAL_CUSTOMER_COUNTS.get(self.instance)
        if expected_count != self.customer_count:
            raise ValueError("shard instance/customer_count identity mismatch")
        allowed_seeds = PILOT_SEEDS if self.scope == "pilot" else FORMAL_SEEDS
        if not _is_plain_int(self.seed) or self.seed not in allowed_seeds:
            raise ValueError(f"shard seed is outside the {self.scope} scope")
        if self.scope == "pilot" and self.instance not in _PILOT_CUSTOMER_COUNTS:
            raise ValueError("pilot shard instance is outside the Stage 0 scope")
        if self.family != _instance_family(self.instance):
            raise ValueError("shard instance/family identity mismatch")
        if self.checkpoint_seconds != CHECKPOINT_SECONDS:
            raise ValueError("shard checkpoint schedule is not canonical")
        if not _is_plain_int(self.estimated_bytes) or self.estimated_bytes <= 0:
            raise ValueError("shard estimated_bytes must be positive")
        if self.customer_count < 100:
            if self.budgets_seconds != SMALL_BUDGETS:
                raise ValueError("small shard budget must be exactly 30 seconds")
            if self.max_iterations != 1_000:
                raise ValueError("small shard requires 1000 iterations")
        else:
            expected_budgets = SMALL_BUDGETS if self.scope == "pilot" else LARGE_BUDGETS
            if self.budgets_seconds != expected_budgets:
                raise ValueError(
                    "large shard budgets must be 30 seconds for pilot or "
                    "30/60/300 seconds for formal"
                )
            if self.max_iterations is not None:
                raise ValueError("large shard must be wall-clock only")

    @property
    def axis_count(self) -> int:
        return len(self.budgets_seconds)

    @property
    def declared_solver_seconds(self) -> int:
        return sum(self.budgets_seconds)

    @property
    def checkpoint_count(self) -> int:
        return sum(
            1
            for budget in self.budgets_seconds
            for checkpoint in self.checkpoint_seconds
            if checkpoint <= budget
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "instance": self.instance,
            "seed": self.seed,
            "customer_count": self.customer_count,
            "family": self.family,
            "budgets_seconds": list(self.budgets_seconds),
            "checkpoint_seconds": list(self.checkpoint_seconds),
            "estimated_bytes": self.estimated_bytes,
            "max_iterations": self.max_iterations,
            "scope": self.scope,
        }


@dataclass(frozen=True, slots=True)
class BatchPlan:
    """A sequential next-fit group of indivisible shards."""

    batch_id: str
    shards: tuple[ShardPlan, ...]
    estimated_bytes: int

    def __post_init__(self) -> None:
        if re.fullmatch(r"batch[0-9]{4}", self.batch_id) is None:
            raise ValueError("batch_id must be canonical")
        if not self.shards:
            raise ValueError("batch must contain at least one shard")
        if self.estimated_bytes != sum(shard.estimated_bytes for shard in self.shards):
            raise ValueError("batch estimated_bytes does not match its shards")

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "shard_ids": [shard.shard_id for shard in self.shards],
            "estimated_bytes": self.estimated_bytes,
        }


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    """Complete immutable preflight plan for one Pilot or Formal attempt."""

    run_label: str
    shards: tuple[ShardPlan, ...]
    batches: tuple[BatchPlan, ...] = ()
    scope: str = "formal"

    def __post_init__(self) -> None:
        if self.scope not in _CAMPAIGN_GEOMETRY:
            raise ValueError("campaign plan scope must be pilot or formal")
        customer_counts = (
            _PILOT_CUSTOMER_COUNTS if self.scope == "pilot" else _CANONICAL_CUSTOMER_COUNTS
        )
        seeds = PILOT_SEEDS if self.scope == "pilot" else FORMAL_SEEDS
        expected_identities = tuple(
            sorted(
                (customer_count, instance, seed)
                for instance, customer_count in customer_counts.items()
                for seed in seeds
            )
        )
        observed_identities = tuple(
            (shard.customer_count, shard.instance, shard.seed) for shard in self.shards
        )
        expected_shard_ids = tuple(
            f"shard{index:04d}" for index in range(1, len(expected_identities) + 1)
        )
        if (
            observed_identities != expected_identities
            or tuple(shard.shard_id for shard in self.shards) != expected_shard_ids
            or any(shard.scope != self.scope for shard in self.shards)
        ):
            raise ValueError(
                f"campaign plan does not contain the exact ordered {self.scope} shards"
            )
        if self.batches:
            flattened = tuple(shard for batch in self.batches for shard in batch.shards)
            if flattened != self.shards:
                raise ValueError("campaign batches do not preserve the ordered shards")
            expected_batch_ids = tuple(
                f"batch{index:04d}" for index in range(1, len(self.batches) + 1)
            )
            if tuple(batch.batch_id for batch in self.batches) != expected_batch_ids:
                raise ValueError("campaign batch IDs are not contiguous")

    @property
    def axis_count(self) -> int:
        return sum(shard.axis_count for shard in self.shards)

    @property
    def declared_solver_seconds(self) -> int:
        return sum(shard.declared_solver_seconds for shard in self.shards)

    @property
    def checkpoint_count(self) -> int:
        return sum(shard.checkpoint_count for shard in self.shards)

    @property
    def estimated_bytes(self) -> int:
        return sum(shard.estimated_bytes for shard in self.shards)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "stage05.2-campaign-plan-v1",
            "run_label": self.run_label,
            "scope": self.scope,
            "shard_count": len(self.shards),
            "axis_count": self.axis_count,
            "declared_solver_seconds": self.declared_solver_seconds,
            "checkpoint_count": self.checkpoint_count,
            "estimated_bytes": self.estimated_bytes,
            "shards": [shard.to_dict() for shard in self.shards],
            "batches": [batch.to_dict() for batch in self.batches],
        }


@dataclass(frozen=True, slots=True)
class BatchArchiveAssignment:
    """The archive root selected for one planned batch."""

    batch_id: str
    root_alias: str
    estimated_bytes: int

    def __post_init__(self) -> None:
        if re.fullmatch(r"batch[0-9]{4}", self.batch_id) is None:
            raise ValueError("archive assignment batch_id must be canonical")
        if re.fullmatch(r"[a-z][a-z0-9_]*", self.root_alias) is None:
            raise ValueError("archive assignment root_alias is invalid")
        if not _is_plain_int(self.estimated_bytes) or self.estimated_bytes <= 0:
            raise ValueError("archive assignment estimated_bytes must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "root_alias": self.root_alias,
            "estimated_bytes": self.estimated_bytes,
        }


@dataclass(frozen=True, slots=True)
class CampaignCapacityPlan:
    """Preflight allocation after all per-volume reserves are deducted."""

    assignments: tuple[BatchArchiveAssignment, ...]
    usable_bytes_by_device: Mapping[str, int]
    remaining_bytes_by_device: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.assignments or len(
            {assignment.batch_id for assignment in self.assignments}
        ) != len(self.assignments):
            raise ValueError("capacity assignments must be non-empty and unique")
        for values in (self.usable_bytes_by_device, self.remaining_bytes_by_device):
            if not values or any(
                not device or not _is_plain_int(value) or value < 0
                for device, value in values.items()
            ):
                raise ValueError("capacity device-byte mapping is invalid")
        if set(self.usable_bytes_by_device) != set(self.remaining_bytes_by_device):
            raise ValueError("capacity usable/remaining device sets disagree")
        if any(
            self.remaining_bytes_by_device[device] > usable
            for device, usable in self.usable_bytes_by_device.items()
        ):
            raise ValueError("capacity remaining bytes exceed usable bytes")
        object.__setattr__(
            self,
            "usable_bytes_by_device",
            MappingProxyType(dict(self.usable_bytes_by_device)),
        )
        object.__setattr__(
            self,
            "remaining_bytes_by_device",
            MappingProxyType(dict(self.remaining_bytes_by_device)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "assignments": [assignment.to_dict() for assignment in self.assignments],
            "usable_bytes_by_device": dict(sorted(self.usable_bytes_by_device.items())),
            "remaining_bytes_by_device": dict(sorted(self.remaining_bytes_by_device.items())),
        }


@dataclass(frozen=True, slots=True)
class ProcessCpuCounterSample:
    """One replayable cumulative CPU-time snapshot for unrelated processes."""

    sampled_at_seconds: float
    cpu_seconds_by_pid: Mapping[int, float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.sampled_at_seconds) or self.sampled_at_seconds < 0.0:
            raise ValueError("process CPU sample timestamp is invalid")
        if any(
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or isinstance(seconds, bool)
            or not isinstance(seconds, int | float)
            or not math.isfinite(float(seconds))
            or float(seconds) < 0.0
            for pid, seconds in self.cpu_seconds_by_pid.items()
        ):
            raise ValueError("process CPU sample counters are invalid")
        object.__setattr__(
            self,
            "cpu_seconds_by_pid",
            MappingProxyType(
                {int(pid): float(seconds) for pid, seconds in self.cpu_seconds_by_pid.items()}
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sampled_at_seconds": self.sampled_at_seconds,
            "cpu_seconds_by_pid": {
                str(pid): seconds for pid, seconds in sorted(self.cpu_seconds_by_pid.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ProcessCpuCounterSample:
        _exact_fields(
            payload,
            {"sampled_at_seconds", "cpu_seconds_by_pid"},
            "process CPU sample",
        )
        raw_counters = _object_mapping(
            payload.get("cpu_seconds_by_pid"),
            "cpu_seconds_by_pid",
        )
        counters: dict[int, float] = {}
        for raw_pid, raw_seconds in raw_counters.items():
            if re.fullmatch(r"[1-9][0-9]*", raw_pid) is None:
                raise ValueError("process CPU sample PID is invalid")
            if isinstance(raw_seconds, bool) or not isinstance(raw_seconds, int | float):
                raise ValueError("process CPU sample value is invalid")
            counters[int(raw_pid)] = float(raw_seconds)
        return cls(
            sampled_at_seconds=_required_number(payload, "sampled_at_seconds"),
            cpu_seconds_by_pid=counters,
        )


def maximum_process_average_cores(
    samples: Sequence[ProcessCpuCounterSample],
    *,
    logical_cpu_count: int,
) -> float:
    """Replay a conservative full-window per-process CPU average.

    A PID first observed after the baseline starts at zero CPU time.  When a PID
    disappears, the unknown interval up to the first missing sample is charged
    at the machine's full logical-CPU capacity so an exited process cannot hide
    work between samples.
    """

    if len(samples) < 2:
        raise ValueError("process CPU replay requires at least two samples")
    if isinstance(logical_cpu_count, bool) or logical_cpu_count <= 0:
        raise ValueError("logical_cpu_count must be positive")
    timestamps = tuple(sample.sampled_at_seconds for sample in samples)
    if any(
        right <= left
        for left, right in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise ValueError("process CPU samples must be strictly ordered")
    duration = timestamps[-1] - timestamps[0]
    first_counters = samples[0].cpu_seconds_by_pid
    all_pids = set().union(*(sample.cpu_seconds_by_pid for sample in samples))
    maximum = 0.0
    for pid in all_pids:
        baseline = float(first_counters.get(pid, 0.0))
        last_value: float | None = None
        last_seen_at: float | None = None
        disappearance_upper_bound = 0.0
        disappeared = False
        for sample in samples:
            raw_value = sample.cpu_seconds_by_pid.get(pid)
            if raw_value is None:
                if last_seen_at is not None and not disappeared:
                    disappearance_upper_bound = logical_cpu_count * (
                        sample.sampled_at_seconds - last_seen_at
                    )
                    disappeared = True
                continue
            value = float(raw_value)
            if disappeared:
                raise ValueError("process PID reappeared after disappearing in one window")
            if last_value is not None and value < last_value:
                raise ValueError("process CPU counter decreased during load window")
            last_value = value
            last_seen_at = sample.sampled_at_seconds
        if last_value is None:
            continue
        delta = max(0.0, last_value - baseline) + disappearance_upper_bound
        maximum = max(maximum, delta / duration)
    return maximum


def maximum_process_average_cores_over_windows(
    samples: Sequence[ProcessCpuCounterSample],
    *,
    logical_cpu_count: int,
    window_seconds: float,
) -> float:
    """Return the maximum conservative average over complete rolling windows.

    Runtime monitoring must not interpret the disappearance upper bound from a
    partial startup window as measured CPU consumption. Every eligible window
    spans at least ``window_seconds`` and is chosen as tightly as the sampled
    timestamps permit.
    """

    if len(samples) < 2:
        raise ValueError("process CPU replay requires at least two samples")
    if not math.isfinite(window_seconds) or window_seconds <= 0.0:
        raise ValueError("process CPU window_seconds must be positive")
    timestamps = tuple(sample.sampled_at_seconds for sample in samples)
    if any(
        right <= left
        for left, right in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise ValueError("process CPU samples must be strictly ordered")
    maximum = 0.0
    start_index = 0
    for end_index in range(1, len(samples)):
        while (
            start_index + 1 < end_index
            and timestamps[end_index] - timestamps[start_index + 1] >= window_seconds
        ):
            start_index += 1
        if timestamps[end_index] - timestamps[start_index] < window_seconds:
            continue
        maximum = max(
            maximum,
            maximum_process_average_cores(
                samples[start_index : end_index + 1],
                logical_cpu_count=logical_cpu_count,
            ),
        )
    return maximum


@dataclass(frozen=True, slots=True)
class SystemLoadWindow:
    """One measured preflight/runtime load window."""

    started_at_seconds: float
    duration_seconds: float
    maximum_load1: float
    maximum_unrelated_process_average_cores: float
    logical_cpu_count: int | None = None
    process_cpu_samples: tuple[ProcessCpuCounterSample, ...] = ()

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.started_at_seconds)
            or self.started_at_seconds < 0.0
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds <= 0.0
            or not math.isfinite(self.maximum_load1)
            or self.maximum_load1 < 0.0
            or not math.isfinite(self.maximum_unrelated_process_average_cores)
            or self.maximum_unrelated_process_average_cores < 0.0
        ):
            raise ValueError("system load window contains an invalid measurement")
        if self.process_cpu_samples:
            if (
                isinstance(self.logical_cpu_count, bool)
                or not isinstance(self.logical_cpu_count, int)
                or self.logical_cpu_count <= 0
            ):
                raise ValueError("system load window logical CPU count is invalid")
            replayed = maximum_process_average_cores(
                self.process_cpu_samples,
                logical_cpu_count=self.logical_cpu_count,
            )
            if not math.isclose(
                replayed,
                self.maximum_unrelated_process_average_cores,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("system load window CPU aggregate does not replay")
        elif self.logical_cpu_count is not None:
            raise ValueError("logical CPU count requires replayable process samples")

    def to_dict(self) -> dict[str, object]:
        return {
            "started_at_seconds": self.started_at_seconds,
            "duration_seconds": self.duration_seconds,
            "maximum_load1": self.maximum_load1,
            "maximum_unrelated_process_average_cores": (
                self.maximum_unrelated_process_average_cores
            ),
            "logical_cpu_count": self.logical_cpu_count,
            "process_cpu_samples": [sample.to_dict() for sample in self.process_cpu_samples],
        }


@dataclass(frozen=True, slots=True)
class BenchmarkPreflightObservation:
    """Power and two-window load evidence sampled before a campaign batch."""

    power_source: str
    low_power_mode_enabled: bool
    windows: tuple[SystemLoadWindow, ...]

    def __post_init__(self) -> None:
        if not self.power_source:
            raise ValueError("power_source is required")
        if not isinstance(self.low_power_mode_enabled, bool):
            raise ValueError("low_power_mode_enabled must be boolean")


def _is_sha256(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_logical_path(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("manifest logical_path must be a relative POSIX path")


@dataclass(frozen=True, slots=True)
class BatchManifest:
    """Path-free state and integrity record for one campaign batch."""

    run_label: str
    batch_id: str
    status: str
    root_alias: str
    archive_root_alias: str
    logical_path: str
    volume: VolumeIdentity
    shard_ids: tuple[str, ...]
    estimated_bytes: int
    checksum_sha256: str | None = None
    actual_bytes: int | None = None
    row_count: int | None = None
    physical_schema: str | None = None
    resource_summary_sha256: str | None = None
    persistence_attribution_sha256: str | None = None
    control_persistence_seconds: float | None = None
    persistence_ratio: float | None = None
    shard_manifest_sha256_by_id: Mapping[str, str] | None = None
    shard_actual_bytes_by_id: Mapping[str, int] | None = None
    transfer_mode: str | None = None
    archive_transfer_seconds: float | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"stage05\.2_benchmark_(?:attempt|rerun)[0-9]{2}", self.run_label) is None:
            raise ValueError("batch manifest run_label is not canonical")
        if re.fullmatch(r"batch[0-9]{4}", self.batch_id) is None:
            raise ValueError("batch_id must be canonical")
        if self.status not in {"planned", "verified", "archived", "failed"}:
            raise ValueError("unsupported batch manifest status")
        if (
            re.fullmatch(r"[a-z][a-z0-9_]*", self.root_alias) is None
            or re.fullmatch(r"[a-z][a-z0-9_]*", self.archive_root_alias) is None
        ):
            raise ValueError("batch root aliases are required")
        _validate_logical_path(self.logical_path)
        if not self.shard_ids or len(set(self.shard_ids)) != len(self.shard_ids):
            raise ValueError("batch shard IDs must be non-empty and unique")
        if any(re.fullmatch(r"shard[0-9]{4}", shard_id) is None for shard_id in self.shard_ids):
            raise ValueError("batch shard IDs must be canonical")
        if self.estimated_bytes <= 0:
            raise ValueError("batch estimated_bytes must be positive")
        evidence = (
            self.checksum_sha256,
            self.actual_bytes,
            self.row_count,
            self.physical_schema,
            self.resource_summary_sha256,
            self.persistence_attribution_sha256,
            self.control_persistence_seconds,
            self.persistence_ratio,
            self.shard_manifest_sha256_by_id,
            self.shard_actual_bytes_by_id,
        )
        if self.status == "planned" and any(value is not None for value in evidence):
            raise ValueError("planned batch cannot claim verified evidence")
        if self.status in {"verified", "archived"}:
            if (
                not _is_sha256(self.checksum_sha256)
                or isinstance(self.actual_bytes, bool)
                or not isinstance(self.actual_bytes, int)
                or self.actual_bytes <= 0
                or isinstance(self.row_count, bool)
                or not isinstance(self.row_count, int)
                or self.row_count < 0
                or not self.physical_schema
                or not _is_sha256(self.resource_summary_sha256)
                or not _is_sha256(self.persistence_attribution_sha256)
                or self.control_persistence_seconds is None
                or not math.isfinite(self.control_persistence_seconds)
                or self.control_persistence_seconds < 0.0
                or self.persistence_ratio is None
                or not math.isfinite(self.persistence_ratio)
                or not 0.0
                <= self.persistence_ratio
                <= STAGE052_MAXIMUM_PERSISTENCE_RATIO
                or not isinstance(self.shard_manifest_sha256_by_id, Mapping)
                or set(self.shard_manifest_sha256_by_id) != set(self.shard_ids)
                or any(not _is_sha256(value) for value in self.shard_manifest_sha256_by_id.values())
                or not isinstance(self.shard_actual_bytes_by_id, Mapping)
                or set(self.shard_actual_bytes_by_id) != set(self.shard_ids)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in self.shard_actual_bytes_by_id.values()
                )
            ):
                raise ValueError("verified batch evidence is incomplete")
            if self.actual_bytes > 32 * GIB:
                raise ValueError("batch actual bytes exceed 32 GiB hard cap")
            if any(value > 2 * GIB for value in self.shard_actual_bytes_by_id.values()):
                raise ValueError("shard actual bytes exceed 2 GiB hard cap")
            if sum(self.shard_actual_bytes_by_id.values()) > self.actual_bytes:
                raise ValueError("shard actual bytes exceed batch actual bytes")
            object.__setattr__(
                self,
                "shard_manifest_sha256_by_id",
                MappingProxyType(dict(self.shard_manifest_sha256_by_id)),
            )
            object.__setattr__(
                self,
                "shard_actual_bytes_by_id",
                MappingProxyType(dict(self.shard_actual_bytes_by_id)),
            )
        else:
            if isinstance(self.shard_manifest_sha256_by_id, Mapping):
                object.__setattr__(
                    self,
                    "shard_manifest_sha256_by_id",
                    MappingProxyType(dict(self.shard_manifest_sha256_by_id)),
                )
            if isinstance(self.shard_actual_bytes_by_id, Mapping):
                object.__setattr__(
                    self,
                    "shard_actual_bytes_by_id",
                    MappingProxyType(dict(self.shard_actual_bytes_by_id)),
                )
        if self.status == "archived" and self.transfer_mode not in {
            "same_volume_atomic_rename",
            "cross_volume_verified_copy",
        }:
            raise ValueError("archived batch transfer mode is invalid")
        if self.status == "archived" and (
            self.archive_transfer_seconds is None
            or not math.isfinite(self.archive_transfer_seconds)
            or self.archive_transfer_seconds < 0.0
        ):
            raise ValueError("archived batch transfer time is invalid")
        if self.status in {"planned", "verified"} and (
            self.transfer_mode is not None or self.archive_transfer_seconds is not None
        ):
            raise ValueError("unarchived batch cannot claim archive transfer evidence")
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("failed batch requires a failure reason")
        if self.status != "failed" and self.failure_reason is not None:
            raise ValueError("non-failed batch cannot carry a failure reason")

    @classmethod
    def planned(
        cls,
        *,
        run_label: str,
        plan: BatchPlan,
        staging_root: StorageRoot,
        archive_root_alias: str,
    ) -> BatchManifest:
        return cls(
            run_label=run_label,
            batch_id=plan.batch_id,
            status="planned",
            root_alias=staging_root.alias,
            archive_root_alias=archive_root_alias,
            logical_path=f"{run_label}/{plan.batch_id}",
            volume=staging_root.volume,
            shard_ids=tuple(shard.shard_id for shard in plan.shards),
            estimated_bytes=plan.estimated_bytes,
        )

    def mark_verified(
        self,
        *,
        checksum_sha256: str,
        actual_bytes: int,
        row_count: int,
        physical_schema: str,
        resource_summary_sha256: str,
        persistence_attribution_sha256: str,
        control_persistence_seconds: float,
        persistence_ratio: float,
        shard_manifest_sha256_by_id: Mapping[str, str],
        shard_actual_bytes_by_id: Mapping[str, int],
    ) -> BatchManifest:
        if self.status != "planned":
            raise RuntimeError("only a planned batch can become verified")
        if actual_bytes > 32 * GIB:
            raise ValueError("batch actual bytes exceed 32 GiB hard cap")
        if any(value > 2 * GIB for value in shard_actual_bytes_by_id.values()):
            raise ValueError("shard actual bytes exceed 2 GiB hard cap")
        return replace(
            self,
            status="verified",
            checksum_sha256=checksum_sha256,
            actual_bytes=actual_bytes,
            row_count=row_count,
            physical_schema=physical_schema,
            resource_summary_sha256=resource_summary_sha256,
            persistence_attribution_sha256=persistence_attribution_sha256,
            control_persistence_seconds=control_persistence_seconds,
            persistence_ratio=persistence_ratio,
            shard_manifest_sha256_by_id=shard_manifest_sha256_by_id,
            shard_actual_bytes_by_id=shard_actual_bytes_by_id,
        )

    def mark_archived(
        self,
        *,
        root_alias: str,
        volume: VolumeIdentity,
        transfer_mode: str,
        archive_transfer_seconds: float,
    ) -> BatchManifest:
        if self.status != "verified":
            raise RuntimeError("only a verified batch can become archived")
        if root_alias != self.archive_root_alias:
            raise ValueError("archive root does not match the planned assignment")
        return replace(
            self,
            status="archived",
            root_alias=root_alias,
            volume=volume,
            transfer_mode=transfer_mode,
            archive_transfer_seconds=archive_transfer_seconds,
        )

    def mark_failed(self, reason: str) -> BatchManifest:
        if self.status == "archived":
            raise RuntimeError("an archived batch cannot be rewritten as failed")
        if not reason:
            raise ValueError("batch failure reason is required")
        return replace(self, status="failed", failure_reason=reason)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "stage05.2-batch-manifest-v1",
            "run_label": self.run_label,
            "batch_id": self.batch_id,
            "status": self.status,
            "root_alias": self.root_alias,
            "archive_root_alias": self.archive_root_alias,
            "logical_path": self.logical_path,
            "volume_identity": self.volume.to_dict(),
            "shard_ids": list(self.shard_ids),
            "estimated_bytes": self.estimated_bytes,
            "checksum_sha256": self.checksum_sha256,
            "actual_bytes": self.actual_bytes,
            "row_count": self.row_count,
            "physical_schema": self.physical_schema,
            "resource_summary_sha256": self.resource_summary_sha256,
            "persistence_attribution_sha256": self.persistence_attribution_sha256,
            "control_persistence_seconds": self.control_persistence_seconds,
            "persistence_ratio": self.persistence_ratio,
            "shard_manifest_sha256_by_id": (
                None
                if self.shard_manifest_sha256_by_id is None
                else dict(sorted(self.shard_manifest_sha256_by_id.items()))
            ),
            "shard_actual_bytes_by_id": (
                None
                if self.shard_actual_bytes_by_id is None
                else dict(sorted(self.shard_actual_bytes_by_id.items()))
            ),
            "transfer_mode": self.transfer_mode,
            "archive_transfer_seconds": self.archive_transfer_seconds,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BatchManifest:
        """Parse one strict, path-free batch manifest payload."""

        expected_fields = {
            "schema_version",
            "run_label",
            "batch_id",
            "status",
            "root_alias",
            "archive_root_alias",
            "logical_path",
            "volume_identity",
            "shard_ids",
            "estimated_bytes",
            "checksum_sha256",
            "actual_bytes",
            "row_count",
            "physical_schema",
            "resource_summary_sha256",
            "persistence_attribution_sha256",
            "control_persistence_seconds",
            "persistence_ratio",
            "shard_manifest_sha256_by_id",
            "shard_actual_bytes_by_id",
            "transfer_mode",
            "archive_transfer_seconds",
            "failure_reason",
        }
        _exact_fields(payload, expected_fields, "batch manifest")
        if payload.get("schema_version") != "stage05.2-batch-manifest-v1":
            raise ValueError("batch manifest schema_version is unsupported")
        return cls(
            run_label=_required_str(payload, "run_label"),
            batch_id=_required_str(payload, "batch_id"),
            status=_required_str(payload, "status"),
            root_alias=_required_str(payload, "root_alias"),
            archive_root_alias=_required_str(payload, "archive_root_alias"),
            logical_path=_required_str(payload, "logical_path"),
            volume=VolumeIdentity.from_dict(
                _object_mapping(payload.get("volume_identity"), "volume_identity")
            ),
            shard_ids=_string_tuple(payload, "shard_ids"),
            estimated_bytes=_required_int(payload, "estimated_bytes"),
            checksum_sha256=_optional_str(payload, "checksum_sha256"),
            actual_bytes=_optional_int(payload, "actual_bytes"),
            row_count=_optional_int(payload, "row_count"),
            physical_schema=_optional_str(payload, "physical_schema"),
            resource_summary_sha256=_optional_str(payload, "resource_summary_sha256"),
            persistence_attribution_sha256=_optional_str(
                payload, "persistence_attribution_sha256"
            ),
            control_persistence_seconds=_optional_number(
                payload, "control_persistence_seconds"
            ),
            persistence_ratio=_optional_number(payload, "persistence_ratio"),
            shard_manifest_sha256_by_id=_optional_string_mapping(
                payload, "shard_manifest_sha256_by_id"
            ),
            shard_actual_bytes_by_id=_optional_int_mapping(payload, "shard_actual_bytes_by_id"),
            transfer_mode=_optional_str(payload, "transfer_mode"),
            archive_transfer_seconds=_optional_number(payload, "archive_transfer_seconds"),
            failure_reason=_optional_str(payload, "failure_reason"),
        )


@dataclass(frozen=True, slots=True)
class CampaignManifest:
    """Top-level state machine for one immutable campaign identity."""

    run_label: str
    status: str
    scope: str
    configuration_sha256: str
    prerequisite_review_sha256: str
    selected_backend: str
    selected_exact_backend: str
    selected_workers: int
    native_profile: str
    storage_policy_version: str
    screening_schema_version: str
    storage_roots: Mapping[str, VolumeIdentity]
    shard_count: int
    axis_count: int
    declared_solver_seconds: int
    checkpoint_count: int
    batches: tuple[BatchManifest, ...]
    producer_resource_contract: ProducerResourceContract | None = None
    batch_persistence_envelope_sha256_by_id: Mapping[str, str] = field(
        default_factory=dict
    )
    artifacts: tuple[Mapping[str, object], ...] = ()
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"stage05\.2_benchmark_(?:attempt|rerun)[0-9]{2}", self.run_label) is None:
            raise ValueError("campaign manifest run_label is not canonical")
        if self.status not in {"planned", "complete", "failed"}:
            raise ValueError("unsupported campaign manifest status")
        if self.scope not in _CAMPAIGN_GEOMETRY:
            raise ValueError("campaign manifest scope must be pilot or formal")
        if not _is_sha256(self.configuration_sha256) or not _is_sha256(
            self.prerequisite_review_sha256
        ):
            raise ValueError("campaign configuration/prerequisite hashes are invalid")
        observed_geometry = (
            self.shard_count,
            self.axis_count,
            self.declared_solver_seconds,
            self.checkpoint_count,
        )
        if observed_geometry != _CAMPAIGN_GEOMETRY[self.scope]:
            raise ValueError(f"campaign manifest geometry is not {self.scope}-complete")
        if (
            self.selected_workers not in {2, 4, 5, 6}
            or self.selected_backend not in {"native_cpu", "cuda"}
            or self.selected_exact_backend != "cpu_batch"
            or not self.native_profile
            or self.storage_policy_version != "artifact-storage-v2"
            or self.screening_schema_version != "screening_decisions_v3"
        ):
            raise ValueError("campaign execution contract is invalid")
        if (
            self.producer_resource_contract is not None
            and self.producer_resource_contract.selected_workers != self.selected_workers
        ):
            raise ValueError("campaign producer resource contract worker mismatch")
        if not self.storage_roots or any(
            re.fullmatch(r"[a-z][a-z0-9_]*", alias) is None
            or not isinstance(volume, VolumeIdentity)
            for alias, volume in self.storage_roots.items()
        ):
            raise ValueError("campaign storage root identities are invalid")
        object.__setattr__(
            self,
            "storage_roots",
            MappingProxyType(dict(self.storage_roots)),
        )
        if not self.batches or len({batch.batch_id for batch in self.batches}) != len(self.batches):
            raise ValueError("campaign batches must be non-empty and unique")
        if any(batch.run_label != self.run_label for batch in self.batches):
            raise ValueError("campaign/batch run_label mismatch")
        envelope_hashes = dict(self.batch_persistence_envelope_sha256_by_id)
        batch_ids = {batch.batch_id for batch in self.batches}
        if (
            not set(envelope_hashes).issubset(batch_ids)
            or any(not _is_sha256(digest) for digest in envelope_hashes.values())
        ):
            raise ValueError("campaign batch persistence envelope hashes are invalid")
        object.__setattr__(
            self,
            "batch_persistence_envelope_sha256_by_id",
            MappingProxyType(envelope_hashes),
        )
        artifact_rows = tuple(MappingProxyType(dict(item)) for item in self.artifacts)
        artifact_paths = [str(item.get("relative_path", "")) for item in artifact_rows]
        if artifact_rows and (
            len(set(artifact_paths)) != len(artifact_paths)
            or any(
                not path
                or PurePosixPath(path).is_absolute()
                or ".." in PurePosixPath(path).parts
                or not _is_sha256(str(item.get("checksum", "")))
                or isinstance(item.get("byte_size"), bool)
                or not isinstance(item.get("byte_size"), int)
                or cast(int, item["byte_size"]) < 0
                for path, item in zip(artifact_paths, artifact_rows, strict=True)
            )
        ):
            raise ValueError("campaign artifact inventory is invalid")
        object.__setattr__(self, "artifacts", artifact_rows)
        if any(
            batch.root_alias not in self.storage_roots
            or batch.archive_root_alias not in self.storage_roots
            or batch.volume != self.storage_roots[batch.root_alias]
            for batch in self.batches
        ):
            raise ValueError("campaign/batch storage identity mismatch")
        expected_batch_ids = tuple(f"batch{index:04d}" for index in range(1, len(self.batches) + 1))
        if tuple(batch.batch_id for batch in self.batches) != expected_batch_ids:
            raise ValueError("campaign batch IDs must be contiguous")
        shard_ids = tuple(shard_id for batch in self.batches for shard_id in batch.shard_ids)
        expected_shard_ids = tuple(f"shard{index:04d}" for index in range(1, self.shard_count + 1))
        if shard_ids != expected_shard_ids:
            raise ValueError("campaign batch manifests must cover the exact shard IDs")
        if self.status == "complete" and any(batch.status != "archived" for batch in self.batches):
            raise ValueError("complete campaign contains a non-archived batch")
        if self.status == "complete" and set(envelope_hashes) != batch_ids:
            raise ValueError("complete campaign does not bind every batch persistence envelope")
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("failed campaign requires a failure reason")
        if self.status != "failed" and self.failure_reason is not None:
            raise ValueError("non-failed campaign cannot carry a failure reason")

    @classmethod
    def planned(
        cls,
        *,
        config: BenchmarkCampaignConfig,
        plan: CampaignPlan,
        capacity: CampaignCapacityPlan,
        locator: StorageRootLocator,
        configuration_sha256: str,
        prerequisite_review_sha256: str,
    ) -> CampaignManifest:
        assignments = {assignment.batch_id: assignment for assignment in capacity.assignments}
        if plan.run_label != config.run_label or plan.scope != config.scope:
            raise ValueError("campaign plan/config run_label or scope mismatch")
        if set(assignments) != {batch.batch_id for batch in plan.batches}:
            raise ValueError("capacity assignments do not cover the campaign batches")
        planned_bytes = {batch.batch_id: batch.estimated_bytes for batch in plan.batches}
        if any(
            assignment.root_alias not in config.archive_root_aliases
            or assignment.estimated_bytes != planned_bytes[assignment.batch_id]
            for assignment in assignments.values()
        ):
            raise ValueError("capacity assignment identity does not match the plan")
        staging = locator.resolve(config.staging_root_alias)
        manifests = tuple(
            BatchManifest.planned(
                run_label=config.run_label,
                plan=batch,
                staging_root=staging,
                archive_root_alias=assignments[batch.batch_id].root_alias,
            )
            for batch in plan.batches
        )
        used_aliases = tuple(
            dict.fromkeys(
                (
                    config.staging_root_alias,
                    *config.archive_root_aliases,
                )
            )
        )
        roots = MappingProxyType({alias: locator.resolve(alias).volume for alias in used_aliases})
        return cls(
            run_label=config.run_label,
            status="planned",
            scope=config.scope,
            configuration_sha256=configuration_sha256,
            prerequisite_review_sha256=prerequisite_review_sha256,
            selected_backend=config.selected_backend,
            selected_exact_backend=config.selected_exact_backend,
            selected_workers=config.selected_workers,
            native_profile=config.native_profile,
            storage_policy_version=config.storage_policy_version,
            screening_schema_version=config.screening_schema_version,
            storage_roots=roots,
            shard_count=len(plan.shards),
            axis_count=plan.axis_count,
            declared_solver_seconds=plan.declared_solver_seconds,
            checkpoint_count=plan.checkpoint_count,
            batches=manifests,
            producer_resource_contract=config.producer_resource_contract,
        )

    def with_batch(self, batch: BatchManifest) -> CampaignManifest:
        if self.status != "planned":
            raise RuntimeError("only a planned campaign can replace batch state")
        matches = [
            index for index, item in enumerate(self.batches) if item.batch_id == batch.batch_id
        ]
        if len(matches) != 1:
            raise ValueError("replacement batch is not in the campaign")
        existing = self.batches[matches[0]]
        if existing.run_label != batch.run_label or existing.shard_ids != batch.shard_ids:
            raise ValueError("replacement batch identity mismatch")
        updated = list(self.batches)
        updated[matches[0]] = batch
        return replace(self, batches=tuple(updated))

    def with_batch_persistence_envelope(
        self,
        batch_id: str,
        sha256: str,
    ) -> CampaignManifest:
        if self.status != "planned":
            raise RuntimeError("only a planned campaign can bind batch persistence evidence")
        if batch_id not in {batch.batch_id for batch in self.batches} or not _is_sha256(sha256):
            raise ValueError("batch persistence envelope binding is invalid")
        updated = dict(self.batch_persistence_envelope_sha256_by_id)
        if batch_id in updated and updated[batch_id] != sha256:
            raise RuntimeError("batch persistence envelope binding is immutable")
        updated[batch_id] = sha256
        return replace(self, batch_persistence_envelope_sha256_by_id=updated)

    def mark_complete(self) -> CampaignManifest:
        if self.status != "planned":
            raise RuntimeError("only a planned campaign can become complete")
        if any(batch.status != "archived" for batch in self.batches):
            raise RuntimeError("all batches must be archived before campaign completion")
        return replace(self, status="complete")

    def with_artifacts(
        self, artifacts: Sequence[Mapping[str, object]]
    ) -> CampaignManifest:
        if self.status != "complete" or not artifacts:
            raise RuntimeError("only a complete campaign can bind final artifacts")
        if self.artifacts:
            raise RuntimeError("campaign artifact inventory is immutable")
        return replace(self, artifacts=tuple(dict(item) for item in artifacts))

    def mark_failed(self, reason: str) -> CampaignManifest:
        if self.status == "complete":
            raise RuntimeError("a complete campaign cannot become failed")
        if not reason:
            raise ValueError("campaign failure reason is required")
        return replace(self, status="failed", failure_reason=reason)

    def to_dict(self) -> dict[str, object]:
        schema_version = (
            "stage05.2-campaign-manifest-v3"
            if self.artifacts
            else (
                "stage05.2-campaign-manifest-v2"
                if self.producer_resource_contract is not None
                else "stage05.2-campaign-manifest-v1"
            )
        )
        payload: dict[str, object] = {
            "schema_version": schema_version,
            "run_label": self.run_label,
            "status": self.status,
            "scope": self.scope,
            "configuration_sha256": self.configuration_sha256,
            "prerequisite_review_sha256": self.prerequisite_review_sha256,
            "selected_backend": self.selected_backend,
            "selected_exact_backend": self.selected_exact_backend,
            "selected_workers": self.selected_workers,
            "native_profile": self.native_profile,
            "storage_policy_version": self.storage_policy_version,
            "screening_schema_version": self.screening_schema_version,
            "storage_roots": {
                alias: volume.to_dict() for alias, volume in sorted(self.storage_roots.items())
            },
            "shard_count": self.shard_count,
            "axis_count": self.axis_count,
            "declared_solver_seconds": self.declared_solver_seconds,
            "checkpoint_count": self.checkpoint_count,
            "archive_transfer_seconds": sum(
                batch.archive_transfer_seconds or 0.0 for batch in self.batches
            ),
            "batches": [batch.to_dict() for batch in self.batches],
            "batch_persistence_envelope_sha256_by_id": dict(
                sorted(self.batch_persistence_envelope_sha256_by_id.items())
            ),
            "failure_reason": self.failure_reason,
        }
        if self.producer_resource_contract is not None:
            payload["producer_resource_contract"] = (
                self.producer_resource_contract.to_dict()
            )
        if self.artifacts:
            payload["evidence_completeness"] = "complete"
            payload["artifacts"] = [dict(item) for item in self.artifacts]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CampaignManifest:
        """Parse a strict campaign manifest and re-run all geometry gates."""

        expected_fields = {
            "schema_version",
            "run_label",
            "status",
            "scope",
            "configuration_sha256",
            "prerequisite_review_sha256",
            "selected_backend",
            "selected_exact_backend",
            "selected_workers",
            "native_profile",
            "storage_policy_version",
            "screening_schema_version",
            "storage_roots",
            "shard_count",
            "axis_count",
            "declared_solver_seconds",
            "checkpoint_count",
            "archive_transfer_seconds",
            "batches",
            "batch_persistence_envelope_sha256_by_id",
            "failure_reason",
        }
        schema_version = payload.get("schema_version")
        if schema_version in {
            "stage05.2-campaign-manifest-v2",
            "stage05.2-campaign-manifest-v3",
        }:
            expected_fields.add("producer_resource_contract")
        if schema_version == "stage05.2-campaign-manifest-v3":
            expected_fields.update({"evidence_completeness", "artifacts"})
        _exact_fields(payload, expected_fields, "campaign manifest")
        if schema_version not in {
            "stage05.2-campaign-manifest-v1",
            "stage05.2-campaign-manifest-v2",
            "stage05.2-campaign-manifest-v3",
        }:
            raise ValueError("campaign manifest schema_version is unsupported")
        if schema_version == "stage05.2-campaign-manifest-v3" and not isinstance(
            payload.get("artifacts"), list
        ):
            raise ValueError("campaign artifact inventory must be an array")
        roots_payload = _object_mapping(payload.get("storage_roots"), "storage_roots")
        storage_roots = {
            alias: VolumeIdentity.from_dict(_object_mapping(value, f"storage_roots.{alias}"))
            for alias, value in roots_payload.items()
        }
        batches_payload = payload.get("batches")
        if not isinstance(batches_payload, list):
            raise ValueError("batches must be an array")
        batches = tuple(
            BatchManifest.from_dict(_object_mapping(item, f"batches[{index}]"))
            for index, item in enumerate(batches_payload)
        )
        result = cls(
            run_label=_required_str(payload, "run_label"),
            status=_required_str(payload, "status"),
            scope=_required_str(payload, "scope"),
            configuration_sha256=_required_str(payload, "configuration_sha256"),
            prerequisite_review_sha256=_required_str(payload, "prerequisite_review_sha256"),
            selected_backend=_required_str(payload, "selected_backend"),
            selected_exact_backend=_required_str(payload, "selected_exact_backend"),
            selected_workers=_required_int(payload, "selected_workers"),
            native_profile=_required_str(payload, "native_profile"),
            storage_policy_version=_required_str(payload, "storage_policy_version"),
            screening_schema_version=_required_str(payload, "screening_schema_version"),
            storage_roots=storage_roots,
            shard_count=_required_int(payload, "shard_count"),
            axis_count=_required_int(payload, "axis_count"),
            declared_solver_seconds=_required_int(payload, "declared_solver_seconds"),
            checkpoint_count=_required_int(payload, "checkpoint_count"),
            batches=batches,
            producer_resource_contract=(
                ProducerResourceContract.from_dict(
                    _object_mapping(
                        payload.get("producer_resource_contract"),
                        "producer_resource_contract",
                    )
                )
                if schema_version in {
                    "stage05.2-campaign-manifest-v2",
                    "stage05.2-campaign-manifest-v3",
                }
                else None
            ),
            batch_persistence_envelope_sha256_by_id=_optional_string_mapping(
                payload,
                "batch_persistence_envelope_sha256_by_id",
            )
            or {},
            artifacts=(
                tuple(
                    _object_mapping(item, f"artifacts[{index}]")
                    for index, item in enumerate(
                        cast(list[object], payload.get("artifacts", []))
                    )
                )
                if schema_version == "stage05.2-campaign-manifest-v3"
                and isinstance(payload.get("artifacts"), list)
                else ()
            ),
            failure_reason=_optional_str(payload, "failure_reason"),
        )
        if (
            schema_version == "stage05.2-campaign-manifest-v3"
            and payload.get("evidence_completeness") != "complete"
        ):
            raise ValueError("campaign lifecycle evidence completeness is invalid")
        declared_transfer_seconds = _required_number(payload, "archive_transfer_seconds")
        observed_transfer_seconds = sum(
            batch.archive_transfer_seconds or 0.0 for batch in result.batches
        )
        if not math.isclose(
            declared_transfer_seconds,
            observed_transfer_seconds,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("campaign archive transfer time does not reconcile")
        return result


class ManifestIntegrityError(RuntimeError):
    """A campaign manifest or its SHA-256 sidecar failed verification."""


def _load_verified_manifest_object(path: Path) -> Mapping[str, object]:
    sidecar_path = path.with_suffix(".sha256")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ManifestIntegrityError(f"cannot read manifest envelope: {path}") from error
    if not signed_sidecar_matches(path, sidecar_path):
        raise ManifestIntegrityError("manifest checksum mismatch")
    try:
        decoded = json.loads(raw)
        return _object_mapping(decoded, "manifest")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ManifestIntegrityError("manifest JSON is invalid") from error


def load_batch_manifest(path: Path) -> BatchManifest:
    """Verify and parse a batch manifest plus adjacent ``.sha256`` sidecar."""

    try:
        return BatchManifest.from_dict(_load_verified_manifest_object(path))
    except ValueError as error:
        raise ManifestIntegrityError("batch manifest contract is invalid") from error


def load_campaign_manifest(path: Path) -> CampaignManifest:
    """Verify and parse a campaign manifest plus adjacent ``.sha256`` sidecar."""

    try:
        return CampaignManifest.from_dict(_load_verified_manifest_object(path))
    except ValueError as error:
        raise ManifestIntegrityError("campaign manifest contract is invalid") from error


_RECOVERY_CAUSES: Final = frozenset({"unexpected_host_loss", "operator_stop"})


def _canonical_object_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_signed_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CampaignIdentity:
    """Immutable scientific and runtime identity shared by every recovery epoch."""

    run_label: str
    repository_revision: str
    repository_tree: str
    wheel_sha256: str
    runtime_identity_sha256: str
    configuration_sha256: str
    prerequisite_review_sha256: str
    producer_resource_contract_sha256: str
    campaign_plan_sha256: str
    lifecycle_plan_sha256: str
    start_permit_sha256: str

    def __post_init__(self) -> None:
        if re.fullmatch(
            r"stage05\.2_benchmark_(?:attempt|rerun)[0-9]{2}", self.run_label
        ) is None:
            raise ValueError("campaign recovery run_label is not canonical")
        if re.fullmatch(r"[0-9a-f]{40}", self.repository_revision) is None:
            raise ValueError("campaign recovery repository revision is invalid")
        if re.fullmatch(r"[0-9a-f]{40}", self.repository_tree) is None:
            raise ValueError("campaign recovery repository tree is invalid")
        digests = (
            self.wheel_sha256,
            self.runtime_identity_sha256,
            self.configuration_sha256,
            self.prerequisite_review_sha256,
            self.producer_resource_contract_sha256,
            self.campaign_plan_sha256,
            self.lifecycle_plan_sha256,
            self.start_permit_sha256,
        )
        if any(not _is_sha256(value) for value in digests):
            raise ValueError("campaign recovery identity digest is invalid")

    @property
    def identity_sha256(self) -> str:
        return _canonical_object_sha256(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "stage05.2-campaign-identity-v1",
            "run_label": self.run_label,
            "repository_revision": self.repository_revision,
            "repository_tree": self.repository_tree,
            "wheel_sha256": self.wheel_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "configuration_sha256": self.configuration_sha256,
            "prerequisite_review_sha256": self.prerequisite_review_sha256,
            "producer_resource_contract_sha256": (
                self.producer_resource_contract_sha256
            ),
            "campaign_plan_sha256": self.campaign_plan_sha256,
            "lifecycle_plan_sha256": self.lifecycle_plan_sha256,
            "start_permit_sha256": self.start_permit_sha256,
        }
        if include_digest:
            payload["campaign_identity_sha256"] = self.identity_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CampaignIdentity:
        expected = {
            "schema_version",
            "run_label",
            "repository_revision",
            "repository_tree",
            "wheel_sha256",
            "runtime_identity_sha256",
            "configuration_sha256",
            "prerequisite_review_sha256",
            "producer_resource_contract_sha256",
            "campaign_plan_sha256",
            "lifecycle_plan_sha256",
            "start_permit_sha256",
            "campaign_identity_sha256",
        }
        _exact_fields(payload, expected, "campaign identity")
        if payload.get("schema_version") != "stage05.2-campaign-identity-v1":
            raise ValueError("campaign identity schema is unsupported")
        identity = cls(
            run_label=_required_str(payload, "run_label"),
            repository_revision=_required_str(payload, "repository_revision"),
            repository_tree=_required_str(payload, "repository_tree"),
            wheel_sha256=_required_str(payload, "wheel_sha256"),
            runtime_identity_sha256=_required_str(payload, "runtime_identity_sha256"),
            configuration_sha256=_required_str(payload, "configuration_sha256"),
            prerequisite_review_sha256=_required_str(
                payload, "prerequisite_review_sha256"
            ),
            producer_resource_contract_sha256=_required_str(
                payload, "producer_resource_contract_sha256"
            ),
            campaign_plan_sha256=_required_str(payload, "campaign_plan_sha256"),
            lifecycle_plan_sha256=_required_str(payload, "lifecycle_plan_sha256"),
            start_permit_sha256=_required_str(payload, "start_permit_sha256"),
        )
        if payload.get("campaign_identity_sha256") != identity.identity_sha256:
            raise ValueError("campaign identity self-digest differs")
        return identity


@dataclass(frozen=True, slots=True)
class ResumePermit:
    """One-use authorization to reopen an existing RUNNING campaign."""

    run_label: str
    epoch: int
    cause: str
    campaign_identity_sha256: str
    lifecycle_plan_sha256: str
    start_permit_sha256: str
    previous_campaign_manifest_sha256: str
    previous_epoch_sha256: str
    stop_evidence_sha256: str
    completed_batch_manifest_sha256_by_id: Mapping[str, str]
    interrupted_batch_id: str | None
    nonce: str

    def __post_init__(self) -> None:
        if re.fullmatch(
            r"stage05\.2_benchmark_(?:attempt|rerun)[0-9]{2}", self.run_label
        ) is None:
            raise ValueError("resume permit run_label is not canonical")
        if not _is_plain_int(self.epoch) or self.epoch < 2:
            raise ValueError("resume permit epoch must be at least two")
        if self.cause not in _RECOVERY_CAUSES:
            raise ValueError("resume permit cause is not recoverable")
        digest_values = (
            self.campaign_identity_sha256,
            self.lifecycle_plan_sha256,
            self.start_permit_sha256,
            self.previous_campaign_manifest_sha256,
            self.stop_evidence_sha256,
        )
        if any(not _is_sha256(value) for value in digest_values):
            raise ValueError("resume permit digest is invalid")
        if self.previous_epoch_sha256 and not _is_sha256(
            self.previous_epoch_sha256
        ):
            raise ValueError("resume permit previous epoch digest is invalid")
        completed = dict(self.completed_batch_manifest_sha256_by_id)
        if any(
            re.fullmatch(r"batch[0-9]{4}", batch_id) is None
            or not _is_sha256(digest)
            for batch_id, digest in completed.items()
        ):
            raise ValueError("resume permit completed batch identities are invalid")
        object.__setattr__(
            self,
            "completed_batch_manifest_sha256_by_id",
            MappingProxyType(completed),
        )
        if self.interrupted_batch_id is not None and re.fullmatch(
            r"batch[0-9]{4}", self.interrupted_batch_id
        ) is None:
            raise ValueError("resume permit interrupted batch identity is invalid")
        if re.fullmatch(r"[0-9a-f]{32}", self.nonce) is None:
            raise ValueError("resume permit nonce is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "stage05.2-resume-permit-v1",
            "run_label": self.run_label,
            "epoch": self.epoch,
            "cause": self.cause,
            "campaign_identity_sha256": self.campaign_identity_sha256,
            "lifecycle_plan_sha256": self.lifecycle_plan_sha256,
            "start_permit_sha256": self.start_permit_sha256,
            "previous_campaign_manifest_sha256": (
                self.previous_campaign_manifest_sha256
            ),
            "previous_epoch_sha256": self.previous_epoch_sha256,
            "stop_evidence_sha256": self.stop_evidence_sha256,
            "completed_batch_manifest_sha256_by_id": dict(
                sorted(self.completed_batch_manifest_sha256_by_id.items())
            ),
            "interrupted_batch_id": self.interrupted_batch_id,
            "nonce": self.nonce,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ResumePermit:
        expected = {
            "schema_version",
            "run_label",
            "epoch",
            "cause",
            "campaign_identity_sha256",
            "lifecycle_plan_sha256",
            "start_permit_sha256",
            "previous_campaign_manifest_sha256",
            "previous_epoch_sha256",
            "stop_evidence_sha256",
            "completed_batch_manifest_sha256_by_id",
            "interrupted_batch_id",
            "nonce",
        }
        _exact_fields(payload, expected, "resume permit")
        if payload.get("schema_version") != "stage05.2-resume-permit-v1":
            raise ValueError("resume permit schema is unsupported")
        return cls(
            run_label=_required_str(payload, "run_label"),
            epoch=_required_int(payload, "epoch"),
            cause=_required_str(payload, "cause"),
            campaign_identity_sha256=_required_str(
                payload, "campaign_identity_sha256"
            ),
            lifecycle_plan_sha256=_required_str(payload, "lifecycle_plan_sha256"),
            start_permit_sha256=_required_str(payload, "start_permit_sha256"),
            previous_campaign_manifest_sha256=_required_str(
                payload, "previous_campaign_manifest_sha256"
            ),
            previous_epoch_sha256=_required_str(payload, "previous_epoch_sha256"),
            stop_evidence_sha256=_required_str(payload, "stop_evidence_sha256"),
            completed_batch_manifest_sha256_by_id=(
                _optional_string_mapping(
                    payload, "completed_batch_manifest_sha256_by_id"
                )
                or {}
            ),
            interrupted_batch_id=_optional_str(payload, "interrupted_batch_id"),
            nonce=_required_str(payload, "nonce"),
        )


@dataclass(frozen=True, slots=True)
class CampaignExecutionSession:
    """Restart-safe state returned through the campaign recovery interface."""

    identity: CampaignIdentity
    campaign: CampaignManifest
    epoch: int
    resumed: bool
    completed_batch_ids: tuple[str, ...]
    pending_batch_ids: tuple[str, ...]
    epoch_manifest_path: Path


class CampaignRecoveryError(RuntimeError):
    """The campaign cannot be recovered without weakening evidence identity."""


class CampaignRecoveryController:
    """Deep module for fresh/open recovery and batch-boundary reconciliation."""

    def __init__(self, *, run_dir: Path, locator: StorageRootLocator) -> None:
        self.run_dir = run_dir
        self.locator = locator

    @property
    def identity_path(self) -> Path:
        return self.run_dir / "control" / "campaign_identity.json"

    @property
    def recovery_root(self) -> Path:
        return self.run_dir / "control" / "recovery"

    def open(
        self,
        *,
        expected_identity: CampaignIdentity,
        fresh_campaign: CampaignManifest,
        resume_permit_path: Path | None,
    ) -> CampaignExecutionSession:
        if resume_permit_path is None:
            return self._open_fresh(expected_identity, fresh_campaign)
        return self._open_resumed(
            expected_identity,
            fresh_campaign,
            resume_permit_path,
        )

    def record_epoch_result(
        self,
        campaign: CampaignManifest,
        *,
        status: str,
    ) -> Path:
        """Seal the current epoch result before the top-level artifact manifest."""

        if status != "batch_execution_completed":
            raise ValueError("unsupported campaign recovery epoch result status")
        epoch_paths = sorted(
            (self.recovery_root / "epochs").glob(
                "epoch[0-9][0-9][0-9][0-9].json"
            )
        )
        if not epoch_paths:
            raise CampaignRecoveryError("campaign recovery epoch is missing")
        return _write_recovery_epoch_result(
            run_dir=self.run_dir,
            campaign=campaign,
            epoch=len(epoch_paths),
            status=status,
            cause=None,
        )

    def record_start_evidence(
        self,
        *,
        lifecycle_plan: Mapping[str, object],
        start_permit_path: Path,
        start_observation_path: Path,
        maintenance_audit_path: Path,
    ) -> None:
        """Persist and verify the immutable controls bound into the identity."""

        try:
            identity = CampaignIdentity.from_dict(
                _load_verified_manifest_object(self.identity_path)
            )
            permit_raw = start_permit_path.read_bytes()
            permit_text = permit_raw.decode("utf-8")
            permit_payload = json.loads(permit_text)
            observation_raw = start_observation_path.read_bytes()
            maintenance_raw = maintenance_audit_path.read_bytes()
        except (
            ManifestIntegrityError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            raise CampaignRecoveryError("campaign start evidence is invalid") from error
        if not isinstance(permit_payload, Mapping):
            raise CampaignRecoveryError("campaign start permit must be a JSON object")
        if _canonical_object_sha256(lifecycle_plan) != identity.lifecycle_plan_sha256:
            raise CampaignRecoveryError("campaign lifecycle plan identity differs")
        if hashlib.sha256(permit_raw).hexdigest() != identity.start_permit_sha256:
            raise CampaignRecoveryError("campaign start permit identity differs")
        observation_sha256 = hashlib.sha256(observation_raw).hexdigest()
        maintenance_sha256 = hashlib.sha256(maintenance_raw).hexdigest()
        if (
            permit_payload.get("run_label") != identity.run_label
            or permit_payload.get("stage_plan_sha256")
            != identity.lifecycle_plan_sha256
            or permit_payload.get("observation_sha256") != observation_sha256
            or permit_payload.get("maintenance_audit_sha256")
            != maintenance_sha256
        ):
            raise CampaignRecoveryError("campaign start permit bindings differ")
        controls: dict[Path, Mapping[str, object]] = {
            self.recovery_root / "start" / "lifecycle_plan.json": {
                "schema_version": "stage05.2-lifecycle-plan-binding-v1",
                "run_label": identity.run_label,
                "lifecycle_plan_sha256": identity.lifecycle_plan_sha256,
                "lifecycle_plan": dict(lifecycle_plan),
            },
            self.recovery_root / "start" / "start_permit.json": {
                "schema_version": "stage05.2-start-permit-binding-v1",
                "run_label": identity.run_label,
                "start_permit_sha256": identity.start_permit_sha256,
                "start_permit_raw_utf8": permit_text,
            },
            self.recovery_root / "start" / "start_observation.json": {
                "schema_version": "stage05.2-start-observation-binding-v1",
                "run_label": identity.run_label,
                "start_observation_sha256": observation_sha256,
                "start_observation_raw_utf8": observation_raw.decode("utf-8"),
            },
            self.recovery_root / "start" / "maintenance_audit.json": {
                "schema_version": "stage05.2-maintenance-audit-binding-v1",
                "run_label": identity.run_label,
                "maintenance_audit_sha256": maintenance_sha256,
                "maintenance_audit_raw_utf8": maintenance_raw.decode("utf-8"),
            },
        }
        for path, payload in controls.items():
            if path.exists() or path.with_suffix(".sha256").exists():
                if _load_verified_manifest_object(path) != payload:
                    raise CampaignRecoveryError(
                        f"campaign immutable start control differs: {path.name}"
                    )
            else:
                atomic_write_signed_json(path, payload)

    def _open_fresh(
        self,
        identity: CampaignIdentity,
        campaign: CampaignManifest,
    ) -> CampaignExecutionSession:
        if self.run_dir.name != identity.run_label:
            raise CampaignRecoveryError("campaign output run label differs")
        if self.run_dir.exists():
            raise CampaignRecoveryError(
                "existing campaign output requires a signed resume permit"
            )
        if identity.run_label != campaign.run_label:
            raise CampaignRecoveryError("fresh campaign identity differs")
        self.run_dir.mkdir(parents=True)
        atomic_write_signed_json(self.identity_path, identity.to_dict())
        epoch_path, _ = atomic_write_signed_json(
            self.recovery_root / "epochs" / "epoch0001.json",
            {
                "schema_version": "stage05.2-recovery-epoch-v1",
                "run_label": identity.run_label,
                "epoch": 1,
                "status": "started",
                "cause": "initial_execution",
                "campaign_identity_sha256": identity.identity_sha256,
                "previous_epoch_sha256": "",
                "resume_permit_sha256": "",
                "completed_batch_manifest_sha256_by_id": {},
                "interrupted_batch_id": None,
                "interrupted_batch_action": None,
            },
        )
        return CampaignExecutionSession(
            identity=identity,
            campaign=campaign,
            epoch=1,
            resumed=False,
            completed_batch_ids=(),
            pending_batch_ids=tuple(batch.batch_id for batch in campaign.batches),
            epoch_manifest_path=epoch_path,
        )

    def _open_resumed(
        self,
        expected_identity: CampaignIdentity,
        fresh_campaign: CampaignManifest,
        resume_permit_path: Path,
    ) -> CampaignExecutionSession:
        if self.run_dir.name != expected_identity.run_label:
            raise CampaignRecoveryError("campaign output run label differs")
        if not self.run_dir.is_dir():
            raise CampaignRecoveryError("resume campaign output is missing")
        try:
            permit_parent = resume_permit_path.resolve(strict=True).parent
        except OSError as error:
            raise CampaignRecoveryError("resume permit is missing") from error
        if permit_parent != (self.recovery_root / "permits").resolve():
            raise CampaignRecoveryError("resume permit is outside the campaign")
        try:
            identity = CampaignIdentity.from_dict(
                _load_verified_manifest_object(self.identity_path)
            )
            permit = ResumePermit.from_dict(
                _load_verified_manifest_object(resume_permit_path)
            )
            campaign_path = self.run_dir / "campaign_manifest.json"
            campaign = load_campaign_manifest(campaign_path)
            permit_campaign_path = (
                self.recovery_root
                / "campaign-manifests"
                / f"epoch{permit.epoch:04d}-pre.json"
            )
            permit_campaign = load_campaign_manifest(permit_campaign_path)
        except (ManifestIntegrityError, ValueError) as error:
            raise CampaignRecoveryError(
                "campaign recovery control evidence is invalid"
            ) from error
        if identity != expected_identity or identity.run_label != fresh_campaign.run_label:
            raise CampaignRecoveryError("campaign frozen identity drifted")
        if campaign.status != "planned":
            raise CampaignRecoveryError("only a RUNNING planned campaign can resume")
        if (
            permit.run_label != identity.run_label
            or permit.campaign_identity_sha256 != identity.identity_sha256
            or permit.lifecycle_plan_sha256 != identity.lifecycle_plan_sha256
            or permit.start_permit_sha256 != identity.start_permit_sha256
            or permit.previous_campaign_manifest_sha256
            != _sha256_file(permit_campaign_path)
        ):
            raise CampaignRecoveryError("resume permit does not bind current campaign")
        expected_campaign = replace(
            fresh_campaign,
            batches=permit_campaign.batches,
            batch_persistence_envelope_sha256_by_id=(
                permit_campaign.batch_persistence_envelope_sha256_by_id
            ),
        )
        if permit_campaign != expected_campaign:
            raise CampaignRecoveryError("campaign plan or execution contract drifted")
        consumed_path = self.recovery_root / "consumed" / f"{permit.nonce}.json"
        epoch_paths = sorted((self.recovery_root / "epochs").glob("epoch[0-9][0-9][0-9][0-9].json"))
        if len(epoch_paths) + 1 != permit.epoch:
            raise CampaignRecoveryError("resume permit epoch is not the next epoch")
        previous_epoch_sha256 = _sha256_file(epoch_paths[-1]) if epoch_paths else ""
        if permit.previous_epoch_sha256 != previous_epoch_sha256:
            raise CampaignRecoveryError("resume permit recovery chain differs")
        completed = self._verify_completed_batches(permit_campaign, permit)
        campaign, interrupted, interrupted_action = self._prepare_interrupted_batch(
            campaign, permit
        )
        expected_after_recovery = permit_campaign
        if interrupted_action == "archive_completed":
            recovered_batch = next(
                batch
                for batch in campaign.batches
                if batch.batch_id == permit.interrupted_batch_id
            )
            expected_after_recovery = expected_after_recovery.with_batch(
                recovered_batch
            )
            recovered_envelope = (
                campaign.batch_persistence_envelope_sha256_by_id.get(
                    recovered_batch.batch_id
                )
            )
            if recovered_envelope is None:
                raise CampaignRecoveryError(
                    "recovered archived batch lacks persistence evidence"
                )
            expected_after_recovery = (
                expected_after_recovery.with_batch_persistence_envelope(
                    recovered_batch.batch_id, recovered_envelope
                )
            )
        if campaign != expected_after_recovery:
            raise CampaignRecoveryError("campaign recovery progress differs")
        for batch in campaign.batches:
            if batch.status == "archived" and batch.batch_id not in completed:
                manifest_path = self.locator.resolve(batch.root_alias).absolute_path.joinpath(
                    *PurePosixPath(batch.logical_path).parts,
                    "batch_manifest.json",
                )
                completed[batch.batch_id] = _sha256_file(manifest_path)
        consumption_payload = {
            "schema_version": "stage05.2-resume-permit-consumption-v1",
            "run_label": identity.run_label,
            "epoch": permit.epoch,
            "resume_permit_sha256": _sha256_file(resume_permit_path),
            "nonce": permit.nonce,
            "status": "consumed",
        }
        if consumed_path.exists() or consumed_path.with_suffix(".sha256").exists():
            if _load_verified_manifest_object(consumed_path) != consumption_payload:
                raise CampaignRecoveryError("resume permit consumption differs")
        else:
            atomic_write_signed_json(consumed_path, consumption_payload)
        epoch_path, _ = atomic_write_signed_json(
            self.recovery_root / "epochs" / f"epoch{permit.epoch:04d}.json",
            {
                "schema_version": "stage05.2-recovery-epoch-v1",
                "run_label": identity.run_label,
                "epoch": permit.epoch,
                "status": "started",
                "cause": permit.cause,
                "campaign_identity_sha256": identity.identity_sha256,
                "previous_epoch_sha256": previous_epoch_sha256,
                "resume_permit_sha256": _sha256_file(resume_permit_path),
                "completed_batch_manifest_sha256_by_id": dict(sorted(completed.items())),
                "interrupted_batch_id": interrupted,
                "interrupted_batch_action": interrupted_action,
            },
        )
        pending = tuple(
            batch.batch_id for batch in campaign.batches if batch.status != "archived"
        )
        return CampaignExecutionSession(
            identity=identity,
            campaign=campaign,
            epoch=permit.epoch,
            resumed=True,
            completed_batch_ids=tuple(sorted(completed)),
            pending_batch_ids=pending,
            epoch_manifest_path=epoch_path,
        )

    def _verify_completed_batches(
        self,
        campaign: CampaignManifest,
        permit: ResumePermit,
    ) -> dict[str, str]:
        observed: dict[str, str] = {}
        for batch in campaign.batches:
            if batch.status != "archived":
                continue
            root = self.locator.resolve(batch.root_alias)
            batch_dir = root.absolute_path.joinpath(
                *PurePosixPath(batch.logical_path).parts
            )
            manifest_path = batch_dir / "batch_manifest.json"
            loaded = load_batch_manifest(manifest_path)
            envelope_path = batch_dir / "batch_persistence_envelope.json"
            expected_envelope = campaign.batch_persistence_envelope_sha256_by_id.get(
                batch.batch_id
            )
            if (
                loaded != batch
                or directory_checksum(batch_dir) != batch.checksum_sha256
                or directory_byte_count(batch_dir) != batch.actual_bytes
                or expected_envelope is None
                or not signed_sidecar_matches(
                    envelope_path, envelope_path.with_suffix(".sha256")
                )
                or _sha256_file(envelope_path) != expected_envelope
            ):
                raise CampaignRecoveryError(
                    f"completed batch changed before resume: {batch.batch_id}"
                )
            observed[batch.batch_id] = _sha256_file(manifest_path)
        if observed != dict(permit.completed_batch_manifest_sha256_by_id):
            raise CampaignRecoveryError("resume permit completed batch set differs")
        return observed

    def _prepare_interrupted_batch(
        self,
        campaign: CampaignManifest,
        permit: ResumePermit,
    ) -> tuple[CampaignManifest, str | None, str | None]:
        non_archived = tuple(
            batch for batch in campaign.batches if batch.status != "archived"
        )
        if any(batch.status == "failed" for batch in non_archived):
            raise CampaignRecoveryError("a failed batch cannot resume")
        existing: list[tuple[BatchManifest, Path]] = []
        for batch in non_archived:
            source = self.locator.resolve(batch.root_alias).absolute_path.joinpath(
                *PurePosixPath(batch.logical_path).parts
            )
            destination = self.locator.resolve(
                batch.archive_root_alias
            ).absolute_path.joinpath(*PurePosixPath(batch.logical_path).parts)
            if source.exists() or destination.exists():
                existing.append((batch, source if source.exists() else destination))
        if len(existing) > 1:
            raise CampaignRecoveryError("multiple interrupted batches are present")
        interrupted_id = existing[0][0].batch_id if existing else None
        if interrupted_id is None and permit.interrupted_batch_id is not None:
            prior = next(
                batch
                for batch in campaign.batches
                if batch.batch_id == permit.interrupted_batch_id
            )
            if prior.status == "archived":
                return campaign, prior.batch_id, "archive_completed"
            capsule = (
                self.locator.resolve(prior.archive_root_alias).absolute_path
                / ".campaign-recovery"
                / prior.run_label
                / f"epoch{permit.epoch:04d}"
                / prior.batch_id
            )
            if capsule.exists():
                self._compact_interrupted_batch(
                    path=self.locator.resolve(prior.root_alias).absolute_path.joinpath(
                        *PurePosixPath(prior.logical_path).parts
                    ),
                    batch=prior,
                    permit=permit,
                )
                return campaign, prior.batch_id, "discarded_and_recomputed"
        if permit.interrupted_batch_id != interrupted_id:
            raise CampaignRecoveryError("resume permit interrupted batch differs")
        if not existing:
            return campaign, None, None
        batch, path = existing[0]
        manifest_path = path / "batch_manifest.json"
        verified = None
        if manifest_path.is_file() and manifest_path.with_suffix(".sha256").is_file():
            candidate = load_batch_manifest(manifest_path)
            if candidate.status == "verified":
                verified = candidate
            elif candidate.status == "archived":
                verified = replace(
                    candidate,
                    status="verified",
                    root_alias="wsl_staging",
                    volume=campaign.storage_roots["wsl_staging"],
                    transfer_mode=None,
                    archive_transfer_seconds=None,
                )
        if batch.status == "verified" or verified is not None:
            expected = batch if batch.status == "verified" else verified
            assert expected is not None
            if (
                expected.run_label != batch.run_label
                or expected.batch_id != batch.batch_id
                or expected.shard_ids != batch.shard_ids
            ):
                raise CampaignRecoveryError("verified recovery batch identity differs")
            campaign = self._finish_verified_batch(campaign, expected)
            return campaign, batch.batch_id, "archive_completed"
        self._compact_interrupted_batch(path=path, batch=batch, permit=permit)
        return campaign, batch.batch_id, "discarded_and_recomputed"

    def _finish_verified_batch(
        self,
        campaign: CampaignManifest,
        verified: BatchManifest,
    ) -> CampaignManifest:
        source = self.locator.resolve(verified.root_alias).absolute_path.joinpath(
            *PurePosixPath(verified.logical_path).parts
        )
        destination_root = self.locator.resolve(verified.archive_root_alias)
        destination = destination_root.absolute_path.joinpath(
            *PurePosixPath(verified.logical_path).parts
        )
        incoming = destination.with_name(f"{destination.name}.incoming")
        archived_write_interval: PersistenceInterval | None = None
        if incoming.exists():
            if (
                not source.is_dir()
                or directory_checksum(incoming) != verified.checksum_sha256
                or directory_byte_count(incoming) != verified.actual_bytes
            ):
                raise CampaignRecoveryError("archive incoming recovery state differs")
            shutil.rmtree(incoming)
        if destination.is_dir():
            destination_manifest = load_batch_manifest(
                destination / "batch_manifest.json"
            )
            if destination_manifest.status == "archived":
                archived = destination_manifest
                if (
                    archived.run_label != verified.run_label
                    or archived.batch_id != verified.batch_id
                    or archived.shard_ids != verified.shard_ids
                    or archived.checksum_sha256 != verified.checksum_sha256
                    or archived.actual_bytes != verified.actual_bytes
                    or archived.root_alias != destination_root.alias
                    or archived.volume != destination_root.volume
                    or directory_checksum(destination) != verified.checksum_sha256
                    or directory_byte_count(destination) != verified.actual_bytes
                ):
                    raise CampaignRecoveryError(
                        "archived destination payload differs from verified batch"
                    )
            elif destination_manifest == verified:
                started_ns = time.monotonic_ns()
                archived = verified.mark_archived(
                    root_alias=destination_root.alias,
                    volume=destination_root.volume,
                    transfer_mode="cross_volume_verified_copy",
                    archive_transfer_seconds=0.0,
                )
                atomic_write_signed_json(
                    destination / "batch_manifest.json", archived.to_dict()
                )
                archived_write_interval = PersistenceInterval(
                    label="archived_batch_manifest_write",
                    started_ns=started_ns,
                    completed_ns=time.monotonic_ns(),
                )
            else:
                raise CampaignRecoveryError("archive destination recovery state differs")
            if source.exists():
                if (
                    directory_checksum(source) != verified.checksum_sha256
                    or directory_byte_count(source) != verified.actual_bytes
                ):
                    raise CampaignRecoveryError("archive source/destination forked")
                shutil.rmtree(source)
        else:
            started_ns = time.monotonic_ns()
            archived = BatchArchiver(self.locator).archive(verified)
            atomic_write_signed_json(destination / "batch_manifest.json", archived.to_dict())
            archived_write_interval = PersistenceInterval(
                label="archived_batch_manifest_write",
                started_ns=started_ns,
                completed_ns=time.monotonic_ns(),
            )
        receipt_path = destination / "batch_verified_receipt.json"
        if not signed_sidecar_matches(
            receipt_path, receipt_path.with_suffix(".sha256")
        ):
            raise CampaignRecoveryError("verified batch recovery receipt is missing")
        receipt = _load_verified_manifest_object(receipt_path)
        if (
            receipt.get("schema_version")
            != "stage05.2-batch-verified-receipt-v1"
            or receipt.get("run_label") != verified.run_label
            or receipt.get("batch_id") != verified.batch_id
            or receipt.get("verified_manifest_sha256")
            != _canonical_signed_json_sha256(verified.to_dict())
        ):
            raise CampaignRecoveryError("verified batch recovery receipt differs")
        envelope_path = destination / "batch_persistence_envelope.json"
        if signed_sidecar_matches(
            envelope_path, envelope_path.with_suffix(".sha256")
        ):
            try:
                existing_envelope = BatchPersistenceEnvelope.from_dict(
                    _load_verified_manifest_object(envelope_path)
                )
            except (ManifestIntegrityError, ValueError) as error:
                raise CampaignRecoveryError(
                    "archived batch persistence envelope is invalid"
                ) from error
            if (
                existing_envelope.run_label != archived.run_label
                or existing_envelope.batch_id != archived.batch_id
                or existing_envelope.archived_manifest_sha256
                != _sha256_file(destination / "batch_manifest.json")
            ):
                raise CampaignRecoveryError(
                    "archived batch persistence envelope differs"
                )
            campaign = campaign.with_batch(archived)
            campaign = campaign.with_batch_persistence_envelope(
                archived.batch_id, _sha256_file(envelope_path)
            )
            atomic_write_signed_json(
                self.run_dir / "campaign_manifest.json", campaign.to_dict()
            )
            return campaign
        raw_interval = receipt.get("verified_manifest_write_interval")
        if not isinstance(raw_interval, Mapping):
            raise CampaignRecoveryError("verified batch recovery interval is invalid")
        if archived_write_interval is None:
            now_ns = time.monotonic_ns()
            archived_write_interval = PersistenceInterval(
                label="archived_batch_manifest_write",
                started_ns=now_ns,
                completed_ns=now_ns,
            )
        envelope = BatchPersistenceEnvelope(
            run_label=verified.run_label,
            batch_id=verified.batch_id,
            base_attribution_sha256=str(receipt["base_attribution_sha256"]),
            verified_manifest_sha256=str(receipt["verified_manifest_sha256"]),
            archived_manifest_sha256=_sha256_file(
                destination / "batch_manifest.json"
            ),
            solver_seconds=_required_number(receipt, "solver_seconds"),
            base_persistence_seconds=_required_number(
                receipt, "base_persistence_seconds"
            ),
            state_intervals=(
                PersistenceInterval.from_dict(raw_interval),
                archived_write_interval,
            ),
        )
        envelope_path, _ = atomic_write_signed_json(
            destination / "batch_persistence_envelope.json",
            envelope.to_dict(),
        )
        campaign = campaign.with_batch(archived)
        campaign = campaign.with_batch_persistence_envelope(
            archived.batch_id, _sha256_file(envelope_path)
        )
        atomic_write_signed_json(
            self.run_dir / "campaign_manifest.json", campaign.to_dict()
        )
        return campaign

    def _compact_interrupted_batch(
        self,
        *,
        path: Path,
        batch: BatchManifest,
        permit: ResumePermit,
    ) -> None:
        archive = self.locator.resolve(batch.archive_root_alias).absolute_path
        capsule = (
            archive
            / ".campaign-recovery"
            / batch.run_label
            / f"epoch{permit.epoch:04d}"
            / batch.batch_id
        )
        capsule.mkdir(parents=True, exist_ok=True)
        inventory_path = capsule / "interrupted_inventory.json"
        receipt_path = capsule / "deletion_receipt.json"
        inventory_payload: dict[str, object] | None = None
        if path.is_dir():
            inventory: list[dict[str, object]] = []
            inventory_bytes = 0
            for item in _tree_files(path):
                stat = item.stat()
                inventory_bytes += stat.st_size
                inventory.append(
                    {
                        "relative_path": item.relative_to(path).as_posix(),
                        "byte_size": stat.st_size,
                        "sha256": _sha256_file(item),
                    }
                )
            inventory_payload = {
                "schema_version": "stage05.2-interrupted-batch-inventory-v1",
                "run_label": batch.run_label,
                "batch_id": batch.batch_id,
                "epoch": permit.epoch,
                "cause": permit.cause,
                "campaign_identity_sha256": permit.campaign_identity_sha256,
                "file_count": len(inventory),
                "byte_count": inventory_bytes,
                "files": inventory,
            }
        if inventory_path.exists() or inventory_path.with_suffix(".sha256").exists():
            if inventory_payload is not None and (
                _load_verified_manifest_object(inventory_path) != inventory_payload
            ):
                raise CampaignRecoveryError(
                    "interrupted batch changed after inventory was signed"
                )
            stored_inventory = _load_verified_manifest_object(inventory_path)
            if (
                stored_inventory.get("run_label") != batch.run_label
                or stored_inventory.get("batch_id") != batch.batch_id
                or stored_inventory.get("epoch") != permit.epoch
                or stored_inventory.get("campaign_identity_sha256")
                != permit.campaign_identity_sha256
            ):
                raise CampaignRecoveryError("interrupted batch inventory differs")
        elif inventory_payload is None:
            raise CampaignRecoveryError(
                "interrupted batch vanished before a signed inventory was durable"
            )
        else:
            inventory_path, _ = atomic_write_signed_json(
                inventory_path, inventory_payload
            )
        if path.is_dir():
            control_source = path / "control"
            control_capsule = capsule / "control"
            if control_source.is_dir():
                if control_capsule.exists():
                    shutil.rmtree(control_capsule)
                shutil.copytree(control_source, control_capsule)
            if receipt_path.exists() or receipt_path.with_suffix(".sha256").exists():
                raise CampaignRecoveryError(
                    "interrupted batch source exists after deletion receipt"
                )
            shutil.rmtree(path)
        if path.exists():
            raise CampaignRecoveryError("interrupted batch cleanup did not finish")
        receipt_payload = {
                "schema_version": "stage05.2-interrupted-batch-deletion-v1",
                "run_label": batch.run_label,
                "batch_id": batch.batch_id,
                "epoch": permit.epoch,
                "source_absent": True,
                "inventory_sha256": _sha256_file(inventory_path),
                "retention_class": "duplicate_failure_metadata",
            }
        if receipt_path.exists() or receipt_path.with_suffix(".sha256").exists():
            if _load_verified_manifest_object(receipt_path) != receipt_payload:
                raise CampaignRecoveryError(
                    "interrupted batch deletion receipt differs"
                )
        else:
            atomic_write_signed_json(receipt_path, receipt_payload)


def _write_recovery_epoch_result(
    *,
    run_dir: Path,
    campaign: CampaignManifest,
    epoch: int,
    status: str,
    cause: str | None,
) -> Path:
    if status not in {"interrupted", "batch_execution_completed"}:
        raise ValueError("campaign recovery epoch result status is invalid")
    campaign_path = run_dir / "campaign_manifest.json"
    completed: dict[str, str] = {}
    for batch in campaign.batches:
        if batch.status != "archived":
            continue
        # The payload checksum is immutable in the archived campaign state;
        # the caller separately replays the live manifest before resume.
        completed[batch.batch_id] = batch.checksum_sha256 or ""
    result_path = (
        run_dir
        / "control"
        / "recovery"
        / "results"
        / f"epoch{epoch:04d}.json"
    )
    payload: dict[str, object] = {
        "schema_version": "stage05.2-recovery-epoch-result-v1",
        "run_label": campaign.run_label,
        "epoch": epoch,
        "status": status,
        "cause": cause,
        "campaign_manifest_sha256": _sha256_file(campaign_path),
        "completed_batch_payload_sha256_by_id": dict(sorted(completed.items())),
        "pending_batch_ids": [
            batch.batch_id for batch in campaign.batches if batch.status != "archived"
        ],
    }
    if result_path.exists():
        existing = _load_verified_manifest_object(result_path)
        if existing != payload:
            raise CampaignRecoveryError("campaign recovery epoch result differs")
        return result_path
    path, _ = atomic_write_signed_json(result_path, payload)
    return path


def authorize_resume(
    *,
    run_dir: Path,
    identity: CampaignIdentity,
    locator: StorageRootLocator,
    cause: str,
    stop_evidence_path: Path,
    lifecycle_state: str,
    writer_active: bool,
) -> tuple[ResumePermit, Path]:
    """Issue one signed permit after fail-closed lifecycle and identity checks."""

    if cause not in _RECOVERY_CAUSES:
        raise CampaignRecoveryError("resume cause is not recoverable")
    if lifecycle_state != "RUNNING":
        raise CampaignRecoveryError("resume requires a RUNNING lifecycle")
    if writer_active:
        raise CampaignRecoveryError("resume requires no active campaign writer")
    if not run_dir.is_dir():
        raise CampaignRecoveryError("resume campaign directory is missing")
    if run_dir.name != identity.run_label:
        raise CampaignRecoveryError("resume output run label differs")
    identity_path = run_dir / "control" / "campaign_identity.json"
    persisted = CampaignIdentity.from_dict(_load_verified_manifest_object(identity_path))
    if persisted != identity:
        raise CampaignRecoveryError("resume frozen identity differs")
    try:
        stop_evidence_path.resolve(strict=True).relative_to(
            run_dir / "control" / "recovery"
        )
    except (OSError, ValueError) as error:
        raise CampaignRecoveryError(
            "resume cause evidence is outside the campaign recovery root"
        ) from error
    _verify_recovery_cause_evidence(
        run_dir=run_dir,
        identity=identity,
        cause=cause,
        evidence_path=stop_evidence_path,
    )
    terminal_candidates = (
        run_dir / "control" / f"{identity.run_label}_terminal_manifest.json",
        run_dir / "failure_manifest.json",
        run_dir / "failure_summary.json",
    )
    if any(path.exists() for path in terminal_candidates):
        raise CampaignRecoveryError("terminal campaign evidence forbids resume")
    campaign_path = run_dir / "campaign_manifest.json"
    campaign = load_campaign_manifest(campaign_path)
    if campaign.status != "planned":
        raise CampaignRecoveryError("only a planned campaign can be resumed")
    epoch_paths = sorted(
        (run_dir / "control" / "recovery" / "epochs").glob(
            "epoch[0-9][0-9][0-9][0-9].json"
        )
    )
    next_epoch = len(epoch_paths) + 1
    outstanding = tuple(
        (run_dir / "control" / "recovery" / "permits").glob(
            f"epoch{next_epoch:04d}-*.json"
        )
    )
    if outstanding:
        raise CampaignRecoveryError("a resume permit already exists for the next epoch")
    previous_epoch_sha256 = _sha256_file(epoch_paths[-1]) if epoch_paths else ""
    completed: dict[str, str] = {}
    interrupted: str | None = None
    for batch in campaign.batches:
        if batch.status == "archived":
            manifest_path = locator.resolve(batch.root_alias).absolute_path.joinpath(
                *PurePosixPath(batch.logical_path).parts,
                "batch_manifest.json",
            )
            completed[batch.batch_id] = _sha256_file(manifest_path)
        else:
            source_candidate = locator.resolve(
                batch.root_alias
            ).absolute_path.joinpath(
                *PurePosixPath(batch.logical_path).parts
            )
            archive_candidate = locator.resolve(
                batch.archive_root_alias
            ).absolute_path.joinpath(*PurePosixPath(batch.logical_path).parts)
            if source_candidate.exists() or archive_candidate.exists():
                if interrupted is not None:
                    raise CampaignRecoveryError("multiple interrupted batches are present")
                interrupted = batch.batch_id
    previous_result_path = (
        run_dir
        / "control"
        / "recovery"
        / "results"
        / f"epoch{len(epoch_paths):04d}.json"
    )
    if previous_result_path.exists():
        previous_result = _load_verified_manifest_object(previous_result_path)
        if (
            previous_result.get("run_label") != identity.run_label
            or previous_result.get("epoch") != len(epoch_paths)
            or previous_result.get("status")
            not in {"interrupted", "batch_execution_completed"}
        ):
            raise CampaignRecoveryError("previous recovery epoch result differs")
    else:
        _write_recovery_epoch_result(
            run_dir=run_dir,
            campaign=campaign,
            epoch=len(epoch_paths),
            status="interrupted",
            cause=cause,
        )
    snapshot_path, _ = atomic_write_signed_json(
        run_dir
        / "control"
        / "recovery"
        / "campaign-manifests"
        / f"epoch{next_epoch:04d}-pre.json",
        campaign.to_dict(),
    )
    if _sha256_file(snapshot_path) != _sha256_file(campaign_path):
        raise CampaignRecoveryError("recovery campaign snapshot differs")
    permit = ResumePermit(
        run_label=identity.run_label,
        epoch=next_epoch,
        cause=cause,
        campaign_identity_sha256=identity.identity_sha256,
        lifecycle_plan_sha256=identity.lifecycle_plan_sha256,
        start_permit_sha256=identity.start_permit_sha256,
        previous_campaign_manifest_sha256=_sha256_file(campaign_path),
        previous_epoch_sha256=previous_epoch_sha256,
        stop_evidence_sha256=_sha256_file(stop_evidence_path),
        completed_batch_manifest_sha256_by_id=completed,
        interrupted_batch_id=interrupted,
        nonce=os.urandom(16).hex(),
    )
    permit_path, _ = atomic_write_signed_json(
        run_dir
        / "control"
        / "recovery"
        / "permits"
        / f"epoch{permit.epoch:04d}-{permit.nonce}.json",
        permit.to_dict(),
    )
    return permit, permit_path


def _verify_recovery_cause_evidence(
    *,
    run_dir: Path,
    identity: CampaignIdentity,
    cause: str,
    evidence_path: Path,
) -> None:
    """Verify that a recoverable interruption has an auditable host receipt."""

    try:
        evidence = _load_verified_manifest_object(evidence_path)
    except ManifestIntegrityError as error:
        raise CampaignRecoveryError("resume cause evidence is not signed") from error
    common = {
        "schema_version",
        "run_label",
        "cause",
        "campaign_identity_sha256",
        "writer_absent",
        "service_state",
    }
    if cause == "unexpected_host_loss":
        _exact_fields(evidence, common, "unexpected host-loss receipt")
        valid_schema = "stage05.2-unexpected-host-loss-receipt-v1"
    else:
        _exact_fields(
            evidence,
            common | {"stop_intent_relative_path", "stop_intent_sha256"},
            "operator-stop receipt",
        )
        valid_schema = "stage05.2-operator-stop-receipt-v1"
    if (
        evidence.get("schema_version") != valid_schema
        or evidence.get("run_label") != identity.run_label
        or evidence.get("cause") != cause
        or evidence.get("campaign_identity_sha256") != identity.identity_sha256
        or evidence.get("writer_absent") is not True
        or evidence.get("service_state") not in {"inactive", "failed", "not_found"}
    ):
        raise CampaignRecoveryError("resume cause evidence identity/state differs")
    if cause != "operator_stop":
        return
    relative = PurePosixPath(
        _required_str(evidence, "stop_intent_relative_path")
    )
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignRecoveryError("operator stop-intent path is invalid")
    intent_path = run_dir.joinpath(*relative.parts)
    try:
        intent = _load_verified_manifest_object(intent_path)
    except ManifestIntegrityError as error:
        raise CampaignRecoveryError("operator stop intent is not signed") from error
    _exact_fields(
        intent,
        {
            "schema_version",
            "run_label",
            "campaign_identity_sha256",
            "requested_action",
            "status",
        },
        "operator stop intent",
    )
    if (
        evidence.get("stop_intent_sha256") != _sha256_file(intent_path)
        or intent.get("schema_version") != "stage05.2-stop-intent-v1"
        or intent.get("run_label") != identity.run_label
        or intent.get("campaign_identity_sha256") != identity.identity_sha256
        or intent.get("requested_action") != "stop_for_recovery"
        or intent.get("status") != "requested_before_service_stop"
    ):
        raise CampaignRecoveryError("operator stop-intent binding differs")


class ArchiveTransferError(RuntimeError):
    """A batch archive transaction did not reach verified completion."""


class ArchiveTransferCompletedError(ArchiveTransferError):
    """The payload reached its final root, but a trailing durability step failed."""

    def __init__(
        self,
        *,
        batch: BatchManifest,
        destination: Path,
        cause: Exception,
    ) -> None:
        super().__init__(
            "batch archive payload reached its final destination before a trailing "
            f"durability failure: {type(cause).__name__}: {cause}"
        )
        self.batch = batch
        self.destination = destination


def _tree_files(path: Path) -> tuple[Path, ...]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"archive payload is not a directory: {path}")
    files: list[Path] = []
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"archive payload cannot contain symlinks: {item}")
        if item.is_file():
            files.append(item)
    return tuple(sorted(files, key=lambda item: item.relative_to(path).as_posix()))


_BATCH_ENVELOPE_FILENAMES: Final = frozenset(
    {
        "batch_manifest.json",
        "batch_manifest.sha256",
        "batch_verified_receipt.json",
        "batch_verified_receipt.sha256",
        "batch_persistence_envelope.json",
        "batch_persistence_envelope.sha256",
    }
)


def _batch_payload_files(path: Path) -> tuple[Path, ...]:
    """Return payload files, excluding only the top-level batch envelope."""

    return tuple(
        file_path
        for file_path in _tree_files(path)
        if not (file_path.parent == path and file_path.name in _BATCH_ENVELOPE_FILENAMES)
    )


def directory_checksum(path: Path) -> str:
    """Hash batch payload names and bytes without its self-referential envelope.

    The top-level ``batch_manifest.json`` and adjacent SHA-256 sidecar are not
    payload.  Nested shard/control manifests remain covered by this digest.
    """

    digest = hashlib.sha256(b"stage05.2-directory-v1\0")
    for file_path in _batch_payload_files(path):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = file_path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with file_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def directory_byte_count(path: Path) -> int:
    """Count batch payload bytes with the same envelope rule as its digest."""

    return sum(file_path.stat().st_size for file_path in _batch_payload_files(path))


def directory_file_count(path: Path) -> int:
    """Count files covered by the stable batch tree digest."""

    return len(_batch_payload_files(path))


class ArchiveIO(Protocol):
    """Injectable filesystem boundary for small tests and real transfers."""

    def is_directory(self, path: Path) -> bool: ...

    def exists(self, path: Path) -> bool: ...

    def make_directory(self, path: Path) -> None: ...

    def copy_tree(self, source: Path, destination: Path) -> None: ...

    def replace(self, source: Path, destination: Path) -> None: ...

    def fsync_tree(self, path: Path) -> None: ...

    def fsync_directory(self, path: Path) -> None: ...

    def checksum_tree(self, path: Path) -> str: ...

    def byte_count(self, path: Path) -> int: ...

    def remove_tree(self, path: Path) -> None: ...


class FileSystemArchiveIO:
    """Default archive operations; no cleanup occurs on a failed transfer."""

    def is_directory(self, path: Path) -> bool:
        return path.is_dir()

    def exists(self, path: Path) -> bool:
        return path.exists()

    def make_directory(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def copy_tree(self, source: Path, destination: Path) -> None:
        shutil.copytree(source, destination)

    def replace(self, source: Path, destination: Path) -> None:
        os.replace(source, destination)

    def fsync_tree(self, path: Path) -> None:
        files = _tree_files(path)
        for file_path in files:
            with file_path.open("rb") as handle:
                os.fsync(handle.fileno())
        directories = [item for item in path.rglob("*") if item.is_dir()]
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            self.fsync_directory(directory)
        self.fsync_directory(path)

    def fsync_directory(self, path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def checksum_tree(self, path: Path) -> str:
        return directory_checksum(path)

    def byte_count(self, path: Path) -> int:
        return directory_byte_count(path)

    def remove_tree(self, path: Path) -> None:
        shutil.rmtree(path)


class BatchArchiver:
    """Archive a verified batch without hiding partial transfer evidence."""

    __slots__ = ("_clock", "_io", "_locator")

    def __init__(
        self,
        locator: StorageRootLocator,
        *,
        io: ArchiveIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._locator = locator
        self._io = FileSystemArchiveIO() if io is None else io
        self._clock = clock

    def archive(self, batch: BatchManifest) -> BatchManifest:
        if batch.status != "verified":
            raise ArchiveTransferError("only a verified batch can be archived")
        source_root = self._locator.resolve(batch.root_alias)
        destination_root = self._locator.resolve(batch.archive_root_alias)
        if source_root.volume != batch.volume:
            raise ArchiveTransferError("batch/source volume identity mismatch")
        source = source_root.absolute_path.joinpath(*PurePosixPath(batch.logical_path).parts)
        destination = destination_root.absolute_path.joinpath(
            *PurePosixPath(batch.logical_path).parts
        )
        incoming = destination.with_name(f"{destination.name}.incoming")
        try:
            self._verify_payload(source, batch, label="source")
            if self._io.exists(destination):
                raise ArchiveTransferError("archive destination already exists")
            self._io.make_directory(destination.parent)
            if source_root.volume.device_uuid == destination_root.volume.device_uuid:
                started_at = self._clock()
                return self._same_volume_archive(
                    batch,
                    source,
                    destination,
                    destination_root,
                    started_at,
                )
            if self._io.exists(incoming):
                raise ArchiveTransferError("archive incoming path already exists")
            started_at = self._clock()
            return self._cross_volume_archive(
                batch,
                source,
                destination,
                incoming,
                destination_root,
                started_at,
            )
        except ArchiveTransferError:
            raise
        except (OSError, ValueError) as error:
            raise ArchiveTransferError(
                f"batch archive transaction failed: {type(error).__name__}: {error}"
            ) from error

    def _verify_payload(
        self,
        path: Path,
        batch: BatchManifest,
        *,
        label: str,
    ) -> None:
        if not self._io.is_directory(path):
            raise ArchiveTransferError(f"{label} batch directory is missing")
        if self._io.checksum_tree(path) != batch.checksum_sha256:
            raise ArchiveTransferError(f"{label} checksum mismatch")
        if self._io.byte_count(path) != batch.actual_bytes:
            raise ArchiveTransferError(f"{label} byte count mismatch")

    def _same_volume_archive(
        self,
        batch: BatchManifest,
        source: Path,
        destination: Path,
        destination_root: StorageRoot,
        started_at: float,
    ) -> BatchManifest:
        self._io.replace(source, destination)
        try:
            self._io.fsync_directory(destination.parent)
            if source.parent != destination.parent:
                self._io.fsync_directory(source.parent)
            self._verify_payload(destination, batch, label="archived")
        except (OSError, ValueError) as error:
            self._raise_completed_transfer_if_recoverable(
                batch=batch,
                source=source,
                destination=destination,
                incoming=None,
                destination_root=destination_root,
                transfer_mode="same_volume_atomic_rename",
                started_at=started_at,
                cause=error,
            )
            raise
        return batch.mark_archived(
            root_alias=destination_root.alias,
            volume=destination_root.volume,
            transfer_mode="same_volume_atomic_rename",
            archive_transfer_seconds=self._elapsed_since(started_at),
        )

    def _cross_volume_archive(
        self,
        batch: BatchManifest,
        source: Path,
        destination: Path,
        incoming: Path,
        destination_root: StorageRoot,
        started_at: float,
    ) -> BatchManifest:
        self._io.copy_tree(source, incoming)
        try:
            self._io.fsync_tree(incoming)
            self._io.fsync_directory(incoming.parent)
            self._verify_payload(incoming, batch, label="incoming")
            self._io.replace(incoming, destination)
            self._io.fsync_directory(destination.parent)
            self._verify_payload(destination, batch, label="archived")
            self._io.remove_tree(source)
            self._io.fsync_directory(source.parent)
        except (OSError, ValueError) as error:
            self._raise_completed_transfer_if_recoverable(
                batch=batch,
                source=source,
                destination=destination,
                incoming=incoming,
                destination_root=destination_root,
                transfer_mode="cross_volume_verified_copy",
                started_at=started_at,
                cause=error,
            )
            raise
        return batch.mark_archived(
            root_alias=destination_root.alias,
            volume=destination_root.volume,
            transfer_mode="cross_volume_verified_copy",
            archive_transfer_seconds=self._elapsed_since(started_at),
        )

    def _raise_completed_transfer_if_recoverable(
        self,
        *,
        batch: BatchManifest,
        source: Path,
        destination: Path,
        incoming: Path | None,
        destination_root: StorageRoot,
        transfer_mode: str,
        started_at: float,
        cause: Exception,
    ) -> None:
        if (
            self._io.exists(source)
            or not self._io.is_directory(destination)
            or (incoming is not None and self._io.exists(incoming))
        ):
            return
        try:
            self._verify_payload(destination, batch, label="archived recovery")
            archived = batch.mark_archived(
                root_alias=destination_root.alias,
                volume=destination_root.volume,
                transfer_mode=transfer_mode,
                archive_transfer_seconds=self._elapsed_since(started_at),
            )
        except ArchiveTransferError:
            return
        raise ArchiveTransferCompletedError(
            batch=archived,
            destination=destination,
            cause=cause,
        ) from cause

    def _elapsed_since(self, started_at: float) -> float:
        elapsed = self._clock() - started_at
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ArchiveTransferError("archive transfer clock is invalid")
        return elapsed


@dataclass(frozen=True, slots=True)
class BenchmarkCampaignConfig:
    """Fail-fast configuration for a Stage 5.2 Pilot or Formal campaign."""

    run_label: str
    staging_root_alias: str
    archive_root_aliases: tuple[str, ...]
    selected_backend: str
    selected_exact_backend: str
    selected_workers: int
    native_profile: str
    instances: tuple[CampaignInstance, ...]
    scope: str
    seeds: tuple[int, ...] = FORMAL_SEEDS
    storage_policy_version: str = "artifact-storage-v2"
    screening_schema_version: str = "screening_decisions_v3"
    batch_target_bytes: int = 24 * GIB
    batch_hard_cap_bytes: int = 32 * GIB
    shard_hard_cap_bytes: int = 2 * GIB
    external_safety_reserve_bytes: int = 50 * GIB
    external_active_workspace_bytes: int = 32 * GIB
    internal_safety_reserve_bytes: int = 0
    producer_resource_contract: ProducerResourceContract | None = None
    required_power_source: str | None = None
    preflight_window_count: int = 2
    preflight_window_seconds: float = 30.0
    maximum_load1: float = RUNTIME_LOAD_POLICY.preflight_maximum_load1
    maximum_unrelated_process_average_cores: float = (
        RUNTIME_LOAD_POLICY.maximum_unrelated_process_average_cores
    )

    def __post_init__(self) -> None:
        if re.fullmatch(r"stage05\.2_benchmark_(?:attempt|rerun)[0-9]{2}", self.run_label) is None:
            raise ValueError("campaign run_label is not canonical")
        if self.scope not in _CAMPAIGN_GEOMETRY:
            raise ValueError("campaign scope must be pilot or formal")
        if not self.staging_root_alias or not self.archive_root_aliases:
            raise ValueError("staging and archive root aliases are required")
        if len(set(self.archive_root_aliases)) != len(self.archive_root_aliases):
            raise ValueError("archive root aliases must be unique")
        if (
            self.selected_backend not in {"native_cpu", "cuda"}
            or self.selected_exact_backend != "cpu_batch"
            or not self.native_profile
        ):
            raise ValueError(
                "Stage 5.2 campaign requires selected native_cpu/cuda execution "
                "and cpu_batch exact backend"
            )
        if self.selected_workers not in {2, 4, 5, 6}:
            raise ValueError("selected_workers must be a reviewed 2/4/5/6-worker selection")
        if (
            self.producer_resource_contract is not None
            and self.producer_resource_contract.selected_workers != self.selected_workers
        ):
            raise ValueError("producer resource contract worker selection mismatch")
        if (
            self.storage_policy_version != "artifact-storage-v2"
            or self.screening_schema_version != "screening_decisions_v3"
        ):
            raise ValueError("Stage 5.2 campaign requires storage v2/schema v3")
        expected_instances = _pilot_instances() if self.scope == "pilot" else _canonical_instances()
        canonical = {
            (item.instance, item.customer_count, item.family) for item in expected_instances
        }
        observed = {(item.instance, item.customer_count, item.family) for item in self.instances}
        if len(self.instances) != len(expected_instances) or observed != canonical:
            raise ValueError(f"{self.scope} campaign instance scope is not canonical")
        expected_seeds = PILOT_SEEDS if self.scope == "pilot" else FORMAL_SEEDS
        if self.seeds != expected_seeds:
            raise ValueError(f"{self.scope} campaign seed scope is not canonical")
        if (
            self.batch_target_bytes != 24 * GIB
            or self.batch_hard_cap_bytes != 32 * GIB
            or self.shard_hard_cap_bytes != 2 * GIB
            or self.external_safety_reserve_bytes != 50 * GIB
            or self.external_active_workspace_bytes != 32 * GIB
            or self.internal_safety_reserve_bytes != 0
        ):
            raise ValueError("Stage 5.2 storage byte limits are fixed")
        if not (
            0 < self.shard_hard_cap_bytes <= self.batch_target_bytes <= self.batch_hard_cap_bytes
        ):
            raise ValueError("shard/target/hard-cap byte limits are inconsistent")
        if (
            self.external_safety_reserve_bytes <= 0
            or self.external_active_workspace_bytes <= 0
            or self.internal_safety_reserve_bytes < 0
        ):
            raise ValueError(
                "external capacity reserves must be positive and the internal reserve "
                "must be non-negative"
            )
        if (
            self.required_power_source is not None
            or self.preflight_window_count != 2
            or self.preflight_window_seconds != 30.0
            or self.maximum_load1 != RUNTIME_LOAD_POLICY.preflight_maximum_load1
            or self.maximum_unrelated_process_average_cores
            != RUNTIME_LOAD_POLICY.maximum_unrelated_process_average_cores
        ):
            raise ValueError("Stage 5.2 telemetry sampling protocol is fixed")

    def to_dict(self) -> dict[str, object]:
        """Return the complete path-free configuration used for hashing."""

        return {
            "schema_version": (
                "stage05.2-benchmark-campaign-config-v2"
                if self.producer_resource_contract is not None
                else "stage05.2-benchmark-campaign-config-v1"
            ),
            "run_label": self.run_label,
            "scope": self.scope,
            "staging_root_alias": self.staging_root_alias,
            "archive_root_aliases": list(self.archive_root_aliases),
            "selected_backend": self.selected_backend,
            "selected_exact_backend": self.selected_exact_backend,
            "selected_workers": self.selected_workers,
            "native_profile": self.native_profile,
            "storage_policy_version": self.storage_policy_version,
            "screening_schema_version": self.screening_schema_version,
            "instances": [
                {
                    "instance": item.instance,
                    "customer_count": item.customer_count,
                    "family": item.family,
                }
                for item in sorted(
                    self.instances,
                    key=lambda item: (item.customer_count, item.instance),
                )
            ],
            "seeds": list(self.seeds),
            "batch_target_bytes": self.batch_target_bytes,
            "batch_hard_cap_bytes": self.batch_hard_cap_bytes,
            "shard_hard_cap_bytes": self.shard_hard_cap_bytes,
            "external_safety_reserve_bytes": self.external_safety_reserve_bytes,
            "external_active_workspace_bytes": self.external_active_workspace_bytes,
            "internal_safety_reserve_bytes": self.internal_safety_reserve_bytes,
            "producer_resource_contract": (
                self.producer_resource_contract.to_dict()
                if self.producer_resource_contract is not None
                else None
            ),
            "power_load_preflight": {
                "required_power_source": self.required_power_source,
                "low_power_mode_enabled": False,
                "window_count": self.preflight_window_count,
                "window_seconds": self.preflight_window_seconds,
                "maximum_load1": self.maximum_load1,
                "maximum_unrelated_process_average_cores": (
                    self.maximum_unrelated_process_average_cores
                ),
            },
        }

    @classmethod
    def formal(
        cls,
        *,
        run_label: str,
        staging_root_alias: str,
        archive_root_aliases: tuple[str, ...],
        selected_backend: str,
        selected_exact_backend: str,
        selected_workers: int,
        native_profile: str,
        batch_target_bytes: int = 24 * GIB,
        batch_hard_cap_bytes: int = 32 * GIB,
        shard_hard_cap_bytes: int = 2 * GIB,
        external_safety_reserve_bytes: int = 50 * GIB,
        external_active_workspace_bytes: int = 32 * GIB,
        internal_safety_reserve_bytes: int = 0,
        producer_resource_contract: ProducerResourceContract | None = None,
    ) -> BenchmarkCampaignConfig:
        """Build the canonical 92 x 10 configuration."""

        return cls(
            run_label=run_label,
            staging_root_alias=staging_root_alias,
            archive_root_aliases=archive_root_aliases,
            selected_backend=selected_backend,
            selected_exact_backend=selected_exact_backend,
            selected_workers=selected_workers,
            native_profile=native_profile,
            instances=_canonical_instances(),
            scope="formal",
            batch_target_bytes=batch_target_bytes,
            batch_hard_cap_bytes=batch_hard_cap_bytes,
            shard_hard_cap_bytes=shard_hard_cap_bytes,
            external_safety_reserve_bytes=external_safety_reserve_bytes,
            external_active_workspace_bytes=external_active_workspace_bytes,
            internal_safety_reserve_bytes=internal_safety_reserve_bytes,
            producer_resource_contract=producer_resource_contract,
        )

    @classmethod
    def pilot(
        cls,
        *,
        run_label: str,
        staging_root_alias: str,
        archive_root_aliases: tuple[str, ...],
        selected_backend: str,
        selected_exact_backend: str,
        selected_workers: int,
        native_profile: str,
        batch_target_bytes: int = 24 * GIB,
        batch_hard_cap_bytes: int = 32 * GIB,
        shard_hard_cap_bytes: int = 2 * GIB,
        external_safety_reserve_bytes: int = 50 * GIB,
        external_active_workspace_bytes: int = 32 * GIB,
        internal_safety_reserve_bytes: int = 0,
        producer_resource_contract: ProducerResourceContract | None = None,
    ) -> BenchmarkCampaignConfig:
        """Build the fixed Stage 0 12-instance x three-seed G01 scope."""

        return cls(
            run_label=run_label,
            staging_root_alias=staging_root_alias,
            archive_root_aliases=archive_root_aliases,
            selected_backend=selected_backend,
            selected_exact_backend=selected_exact_backend,
            selected_workers=selected_workers,
            native_profile=native_profile,
            instances=_pilot_instances(),
            scope="pilot",
            seeds=PILOT_SEEDS,
            batch_target_bytes=batch_target_bytes,
            batch_hard_cap_bytes=batch_hard_cap_bytes,
            shard_hard_cap_bytes=shard_hard_cap_bytes,
            external_safety_reserve_bytes=external_safety_reserve_bytes,
            external_active_workspace_bytes=external_active_workspace_bytes,
            internal_safety_reserve_bytes=internal_safety_reserve_bytes,
            producer_resource_contract=producer_resource_contract,
        )

    def build_plan(
        self,
        observations: tuple[PilotStorageObservation, ...] = (),
    ) -> CampaignPlan:
        """Estimate and order every Pilot or Formal campaign shard.

        G01 may plan conservatively before it has produced storage observations;
        each pilot shard then reserves the fixed 2 GiB hard cap.  G02 requires
        the complete accepted G01 observations and applies the 1.5x / 13x rules.
        """

        maxima: dict[tuple[str, int], int] = {}
        for observation in observations:
            if observation.budget_seconds != 30:
                continue
            key = observation.family, observation.customer_count
            maxima[key] = max(maxima.get(key, 0), observation.compressed_bytes)
        required = {
            (family, customer_count)
            for family in ("C", "R", "RC")
            for customer_count in (5, 10, 15, 100)
        }
        missing = sorted(required - maxima.keys())
        if self.scope == "formal" and missing:
            raise ValueError(f"pilot storage estimates are incomplete: {missing}")
        if self.scope == "pilot" and maxima and missing:
            raise ValueError(f"pilot storage estimates are incomplete: {missing}")

        identities = sorted(
            (
                instance.customer_count,
                instance.instance,
                seed,
                instance.family,
            )
            for instance in self.instances
            for seed in self.seeds
        )
        shards: list[ShardPlan] = []
        for ordinal, (customer_count, instance, seed, family) in enumerate(identities, start=1):
            pilot_maximum = maxima.get((family, customer_count))
            budgets: tuple[int, ...]
            if pilot_maximum is None:
                estimated_bytes = self.shard_hard_cap_bytes
            elif customer_count < 100 or self.scope == "pilot":
                estimated_bytes = (pilot_maximum * 3 + 1) // 2
            else:
                estimated_bytes = (pilot_maximum * 39 + 1) // 2
            budgets = (
                SMALL_BUDGETS if customer_count < 100 or self.scope == "pilot" else LARGE_BUDGETS
            )
            if estimated_bytes > self.shard_hard_cap_bytes:
                raise ValueError(f"shard estimate exceeds 2 GiB hard cap: {instance}/{seed}")
            shards.append(
                ShardPlan(
                    shard_id=f"shard{ordinal:04d}",
                    instance=instance,
                    seed=seed,
                    customer_count=customer_count,
                    family=family,
                    budgets_seconds=budgets,
                    checkpoint_seconds=CHECKPOINT_SECONDS,
                    estimated_bytes=estimated_bytes,
                    max_iterations=1000 if customer_count < 100 else None,
                    scope=self.scope,
                )
            )
        batches: list[BatchPlan] = []
        current: list[ShardPlan] = []
        current_bytes = 0
        for shard in shards:
            if current and current_bytes + shard.estimated_bytes > self.batch_target_bytes:
                batches.append(
                    BatchPlan(
                        batch_id=f"batch{len(batches) + 1:04d}",
                        shards=tuple(current),
                        estimated_bytes=current_bytes,
                    )
                )
                current = []
                current_bytes = 0
            current.append(shard)
            current_bytes += shard.estimated_bytes
            if current_bytes > self.batch_hard_cap_bytes:
                raise ValueError("planned batch exceeds 32 GiB hard cap")
        if current:
            batches.append(
                BatchPlan(
                    batch_id=f"batch{len(batches) + 1:04d}",
                    shards=tuple(current),
                    estimated_bytes=current_bytes,
                )
            )
        plan = CampaignPlan(
            self.run_label,
            tuple(shards),
            tuple(batches),
            scope=self.scope,
        )
        observed_geometry = (
            len(plan.shards),
            plan.axis_count,
            plan.declared_solver_seconds,
            plan.checkpoint_count,
        )
        if observed_geometry != _CAMPAIGN_GEOMETRY[self.scope]:
            raise ValueError(f"{self.scope} campaign geometry does not match the contract")
        return plan

    def validate_preflight(
        self,
        observation: BenchmarkPreflightObservation,
        *,
        require_idle_load: bool = True,
    ) -> None:
        """Validate the telemetry sampling protocol without gating host identity."""

        del require_idle_load
        if len(observation.windows) != self.preflight_window_count:
            raise RuntimeError("Stage 5.2 campaign requires two load windows")
        previous_end: float | None = None
        for window in observation.windows:
            if window.duration_seconds != self.preflight_window_seconds:
                raise RuntimeError("each Stage 5.2 load window must be exactly 30 seconds")
            if previous_end is not None and not math.isclose(
                window.started_at_seconds,
                previous_end,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError("Stage 5.2 load windows must be consecutive")
            previous_end = window.started_at_seconds + window.duration_seconds

    def plan_archive_roots(
        self,
        plan: CampaignPlan,
        locator: StorageRootLocator,
        *,
        free_bytes_by_alias: Mapping[str, int],
    ) -> CampaignCapacityPlan:
        """Reserve workspace/safety capacity and assign every complete batch.

        Capacity is counted once per device UUID even when the ignored locator
        exposes multiple logical aliases on the same volume.
        """

        if plan.run_label != self.run_label or plan.scope != self.scope:
            raise ValueError("campaign plan/config run_label or scope mismatch")
        required_aliases = (self.staging_root_alias, *self.archive_root_aliases)
        roots = {alias: locator.resolve(alias) for alias in required_aliases}
        free_values: dict[str, int] = {}
        for alias in required_aliases:
            value = free_bytes_by_alias.get(alias)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid free-byte measurement for root alias: {alias}")
            device = roots[alias].volume.device_uuid
            free_values[device] = min(free_values.get(device, value), value)

        staging = roots[self.staging_root_alias]
        staging_device = staging.volume.device_uuid
        external_floor = self.external_safety_reserve_bytes + self.external_active_workspace_bytes
        if free_values[staging_device] < external_floor:
            raise RuntimeError(
                "staging capacity cannot preserve the 50 GiB safety "
                "reserve and 32 GiB active-batch workspace"
            )

        representative_alias: dict[str, str] = {}
        usable: dict[str, int] = {}
        for alias in self.archive_root_aliases:
            root = roots[alias]
            device = root.volume.device_uuid
            if device in representative_alias:
                continue
            representative_alias[device] = alias
            if device == staging_device:
                reserve = external_floor
            else:
                if not root.volume.filesystem.strip():
                    raise RuntimeError(
                        f"archive root lacks filesystem capability telemetry: {alias}"
                    )
                reserve = self.internal_safety_reserve_bytes
            usable[device] = max(0, free_values[device] - reserve)

        if sum(usable.values()) < plan.estimated_bytes:
            raise RuntimeError(
                "campaign archive projection exceeds capacity after required reserves"
            )

        remaining = dict(usable)
        assignments: list[BatchArchiveAssignment] = []
        ordered_devices = tuple(representative_alias)
        for batch in plan.batches:
            selected_device = next(
                (
                    device
                    for device in ordered_devices
                    if remaining[device] >= batch.estimated_bytes
                ),
                None,
            )
            if selected_device is None:
                raise RuntimeError("campaign archive projection cannot place an indivisible batch")
            remaining[selected_device] -= batch.estimated_bytes
            assignments.append(
                BatchArchiveAssignment(
                    batch_id=batch.batch_id,
                    root_alias=representative_alias[selected_device],
                    estimated_bytes=batch.estimated_bytes,
                )
            )
        return CampaignCapacityPlan(
            assignments=tuple(assignments),
            usable_bytes_by_device=MappingProxyType(dict(usable)),
            remaining_bytes_by_device=MappingProxyType(remaining),
        )
