"""Compatibility adapter from the frozen EVRPTW ABI to TxnOpt case models."""

from __future__ import annotations

from collections.abc import Sequence

from txnopt_cases.evrptw.models import (
    Instance,
    Node,
    NodeType,
    Vehicle,
    configure_native_distance_builder,
)


def _legacy_native_distance_builder(
    coordinates: Sequence[tuple[float, float]],
) -> Sequence[Sequence[float]]:
    import numpy as np

    from evrptw._core import distance_matrix as native_distance_matrix

    matrix = native_distance_matrix(np.asarray(coordinates, dtype=np.float64))
    return tuple(tuple(float(value) for value in row) for row in matrix)


configure_native_distance_builder(_legacy_native_distance_builder)

__all__ = ["Instance", "Node", "NodeType", "Vehicle"]
