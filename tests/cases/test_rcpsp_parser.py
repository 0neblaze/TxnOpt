from __future__ import annotations

from pathlib import Path

import pytest

from txnopt_cases.rcpsp import (
    parse_psplib_sm,
    precedence_feasible_initial_state,
)


def _write_sm(path: Path, *, unknown_successor: bool = False) -> None:
    final_successors = "1 99" if unknown_successor else "0"
    path.write_text(
        f"""************************************************************************
PRECEDENCE RELATIONS:
jobnr. #modes #successors successors
 1 1 2 2 3
 2 1 1 4
 3 1 1 4
 4 1 {final_successors}
************************************************************************
REQUESTS/DURATIONS:
jobnr. mode duration R 1
 1 1 0 0
 2 1 2 2
 3 1 3 1
 4 1 0 0
************************************************************************
RESOURCEAVAILABILITIES:
 R 1
 2
************************************************************************
""",
        encoding="utf-8",
    )


def test_psplib_sm_parser_builds_predecessors_and_initial_state(tmp_path: Path) -> None:
    source = tmp_path / "j1201_1.sm"
    _write_sm(source)

    instance = parse_psplib_sm(source)
    initial = precedence_feasible_initial_state(instance)

    assert instance.name == "j1201_1"
    assert instance.renewable_capacities == (2,)
    assert instance.by_id[4].predecessors == (2, 3)
    assert initial.activity_order == (1, 2, 3, 4)
    assert initial.mode_vector == (0, 0, 0, 0)


def test_psplib_sm_parser_rejects_unknown_successor(tmp_path: Path) -> None:
    source = tmp_path / "invalid.sm"
    _write_sm(source, unknown_successor=True)

    with pytest.raises(ValueError, match="successor"):
        parse_psplib_sm(source)
