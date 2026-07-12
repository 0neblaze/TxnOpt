from __future__ import annotations

import re
from pathlib import Path

from evrptw.models import Instance, Node, NodeType, Vehicle

_PROPERTY_PATTERN = re.compile(r"^\s*([QCrgv])\b.*?/\s*([-+0-9.eE]+)\s*/")


def parse_schneider(path: str | Path) -> Instance:
    source = Path(path)
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError(f"empty instance file: {source}")

    index = next((i for i, line in enumerate(lines) if line.strip()), None)
    if index is None or "StringID" not in lines[index]:
        raise ValueError("missing Schneider instance header")

    nodes: list[Node] = []
    index += 1
    while index < len(lines) and lines[index].strip():
        fields = lines[index].split()
        if len(fields) != 8:
            raise ValueError(f"invalid node row at line {index + 1}: {lines[index]}")
        name, raw_kind, *values = fields
        try:
            kind = NodeType(raw_kind.lower())
            x, y, demand, ready, due, service = map(float, values)
        except ValueError as error:
            raise ValueError(f"invalid node row at line {index + 1}: {lines[index]}") from error
        nodes.append(Node(name, kind, x, y, demand, ready, due, service))
        index += 1

    properties: dict[str, float] = {}
    for line_number, line in enumerate(lines[index:], start=index + 1):
        if not line.strip():
            continue
        match = _PROPERTY_PATTERN.match(line)
        if match is None:
            raise ValueError(f"invalid property row at line {line_number}: {line}")
        properties[match.group(1)] = float(match.group(2))

    missing = {"Q", "C", "r", "g", "v"} - properties.keys()
    if missing:
        raise ValueError(f"missing instance properties: {', '.join(sorted(missing))}")

    vehicle = Vehicle(
        battery_capacity=properties["Q"],
        load_capacity=properties["C"],
        consumption_rate=properties["r"],
        inverse_refueling_rate=properties["g"],
        average_velocity=properties["v"],
    )
    return Instance(source.stem, tuple(nodes), vehicle)
