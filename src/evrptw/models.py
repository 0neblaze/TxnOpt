from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import hypot

import numpy as np

from evrptw._core import distance_matrix as native_distance_matrix


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
    _depot: Node = field(init=False, repr=False, compare=False)
    _customers: tuple[Node, ...] = field(init=False, repr=False, compare=False)
    _stations: tuple[Node, ...] = field(init=False, repr=False, compare=False)
    _by_name: dict[str, Node] = field(init=False, repr=False, compare=False)
    _node_index: dict[str, int] = field(init=False, repr=False, compare=False)
    _distance_matrix: tuple[tuple[float, ...], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        depots = [node for node in self.nodes if node.kind is NodeType.DEPOT]
        if len(depots) != 1:
            raise ValueError(f"expected exactly one depot, found {len(depots)}")
        if len({node.name for node in self.nodes}) != len(self.nodes):
            raise ValueError("node names must be unique")
        if self.vehicle.average_velocity <= 0:
            raise ValueError("average velocity must be positive")
        object.__setattr__(self, "_depot", depots[0])
        object.__setattr__(
            self,
            "_customers",
            tuple(node for node in self.nodes if node.kind is NodeType.CUSTOMER),
        )
        object.__setattr__(
            self,
            "_stations",
            tuple(node for node in self.nodes if node.kind is NodeType.STATION),
        )
        object.__setattr__(self, "_by_name", {node.name: node for node in self.nodes})
        object.__setattr__(
            self,
            "_node_index",
            {node.name: index for index, node in enumerate(self.nodes)},
        )
        matrix = native_distance_matrix(
            np.asarray([(node.x, node.y) for node in self.nodes], dtype=np.float64)
        )
        object.__setattr__(
            self,
            "_distance_matrix",
            tuple(tuple(float(value) for value in row) for row in matrix),
        )

    @property
    def depot(self) -> Node:
        return self._depot

    @property
    def customers(self) -> tuple[Node, ...]:
        return self._customers

    @property
    def stations(self) -> tuple[Node, ...]:
        return self._stations

    @property
    def by_name(self) -> dict[str, Node]:
        return self._by_name

    def distance(self, origin: str, destination: str) -> float:
        """Return one precomputed unrounded Euclidean distance."""

        try:
            return self._distance_matrix[
                self._node_index[origin]
            ][self._node_index[destination]]
        except KeyError as error:
            raise KeyError(f"unknown node in distance lookup: {error.args[0]}") from error
