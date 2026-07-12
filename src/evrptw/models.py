from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import hypot


class NodeType(StrEnum):
    DEPOT = "d"
    STATION = "f"
    CUSTOMER = "c"


@dataclass(frozen=True, slots=True)
class Node:
    name: str
    kind: NodeType
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float

    def distance_to(self, other: Node) -> float:
        return hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True, slots=True)
class Vehicle:
    battery_capacity: float
    load_capacity: float
    consumption_rate: float
    inverse_refueling_rate: float
    average_velocity: float


@dataclass(frozen=True, slots=True)
class Instance:
    name: str
    nodes: tuple[Node, ...]
    vehicle: Vehicle

    def __post_init__(self) -> None:
        depots = [node for node in self.nodes if node.kind is NodeType.DEPOT]
        if len(depots) != 1:
            raise ValueError(f"expected exactly one depot, found {len(depots)}")
        if len({node.name for node in self.nodes}) != len(self.nodes):
            raise ValueError("node names must be unique")
        if self.vehicle.average_velocity <= 0:
            raise ValueError("average velocity must be positive")

    @property
    def depot(self) -> Node:
        return next(node for node in self.nodes if node.kind is NodeType.DEPOT)

    @property
    def customers(self) -> tuple[Node, ...]:
        return tuple(node for node in self.nodes if node.kind is NodeType.CUSTOMER)

    @property
    def stations(self) -> tuple[Node, ...]:
        return tuple(node for node in self.nodes if node.kind is NodeType.STATION)

    @property
    def by_name(self) -> dict[str, Node]:
        return {node.name: node for node in self.nodes}
