from __future__ import annotations

from pathlib import Path

import pytest

from txnopt_evidence.workspace import default_cache_root, default_state_root


def test_default_workspace_roots_are_outside_the_repository_and_tree_scoped() -> None:
    source_tree = "a" * 40
    home = Path("/home/tester")

    assert default_state_root(source_tree, environ={}, home=home) == (
        home / ".local" / "state" / "txnopt" / source_tree
    )
    assert default_cache_root(source_tree, environ={}, home=home) == (
        home / ".cache" / "txnopt" / source_tree
    )


def test_workspace_roots_honor_absolute_xdg_locations() -> None:
    source_tree = "b" * 64
    environ = {
        "XDG_STATE_HOME": "/var/lib/user-state",
        "XDG_CACHE_HOME": "/var/cache/user-cache",
    }

    assert default_state_root(source_tree, environ=environ) == Path(
        "/var/lib/user-state/txnopt", source_tree
    )
    assert default_cache_root(source_tree, environ=environ) == Path(
        "/var/cache/user-cache/txnopt", source_tree
    )


@pytest.mark.parametrize("source_tree", ["", "A" * 40, "a" * 39, "z" * 40])
def test_workspace_roots_reject_noncanonical_source_identity(source_tree: str) -> None:
    with pytest.raises(ValueError):
        default_state_root(source_tree, environ={}, home=Path("/tmp/home"))


def test_workspace_roots_reject_relative_xdg_location() -> None:
    with pytest.raises(ValueError):
        default_cache_root(
            "c" * 40,
            environ={"XDG_CACHE_HOME": "relative/cache"},
            home=Path("/tmp/home"),
        )
