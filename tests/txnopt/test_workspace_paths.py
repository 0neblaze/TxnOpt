from __future__ import annotations

from pathlib import Path

import pytest

from txnopt_evidence.workspace import (
    default_cache_root,
    default_state_root,
    resolve_run_output_root,
)


def test_default_workspace_roots_are_outside_the_repository_and_tree_scoped(
    tmp_path: Path,
) -> None:
    source_tree = "a" * 40
    home = tmp_path / "home"

    assert default_state_root(source_tree, environ={}, home=home) == (
        home / ".local" / "state" / "txnopt" / source_tree
    )
    assert default_cache_root(source_tree, environ={}, home=home) == (
        home / ".cache" / "txnopt" / source_tree
    )


def test_workspace_roots_honor_absolute_xdg_locations(tmp_path: Path) -> None:
    source_tree = "b" * 64
    environ = {
        "XDG_STATE_HOME": str(tmp_path / "user-state"),
        "XDG_CACHE_HOME": str(tmp_path / "user-cache"),
    }

    assert default_state_root(source_tree, environ=environ) == Path(
        tmp_path / "user-state" / "txnopt", source_tree
    )
    assert default_cache_root(source_tree, environ=environ) == Path(
        tmp_path / "user-cache" / "txnopt", source_tree
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


def test_run_output_root_defaults_to_tree_scoped_state(tmp_path: Path) -> None:
    source_tree = "d" * 40
    home = tmp_path / "home"

    assert resolve_run_output_root(
        None,
        producer_identity={"source_tree": source_tree},
        environ={},
        home=home,
    ) == home / ".local" / "state" / "txnopt" / source_tree / "runs"


def test_run_output_root_preserves_an_explicit_external_root(tmp_path: Path) -> None:
    output_root = tmp_path / "raw"
    assert resolve_run_output_root(
        str(output_root),
        producer_identity={
            "binding_status": "UNBOUND_TEST_ONLY",
            "source_tree": "e" * 40,
        },
    ) == output_root


def test_bound_run_requires_explicit_governed_output_root(tmp_path: Path) -> None:
    identity = {
        "binding_status": "BOUND_CLEAN_BUILD",
        "source_tree": "f" * 40,
    }

    with pytest.raises(ValueError, match="explicit governed"):
        resolve_run_output_root(None, producer_identity=identity)
    with pytest.raises(ValueError, match="absolute"):
        resolve_run_output_root("relative/raw", producer_identity=identity)

    worktree = tmp_path / "worktree"
    (worktree / ".git").mkdir(parents=True)
    with pytest.raises(ValueError, match="Git worktree"):
        resolve_run_output_root(
            str(worktree / "raw"),
            producer_identity=identity,
        )


def test_bound_run_rejects_rebuildable_state_as_governed_output(tmp_path: Path) -> None:
    source_tree = "1" * 40
    home = tmp_path / "home"
    state_raw = home / ".local" / "state" / "txnopt" / source_tree / "formal"

    with pytest.raises(ValueError, match="rebuildable"):
        resolve_run_output_root(
            str(state_raw),
            producer_identity={
                "binding_status": "BOUND_CLEAN_BUILD",
                "source_tree": source_tree,
            },
            environ={},
            home=home,
        )


def test_bound_run_rejects_symlink_alias_to_rebuildable_state(tmp_path: Path) -> None:
    source_tree = "2" * 40
    home = tmp_path / "home"
    state = home / ".local" / "state" / "txnopt" / source_tree
    state.mkdir(parents=True)
    alias = tmp_path / "state-alias"
    alias.symlink_to(state, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        resolve_run_output_root(
            str(alias / "formal"),
            producer_identity={
                "binding_status": "BOUND_CLEAN_BUILD",
                "source_tree": source_tree,
            },
            environ={},
            home=home,
        )
