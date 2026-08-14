"""Tree-scoped defaults for rebuildable TxnOpt state and cache data."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def default_state_root(
    source_tree: str,
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return the XDG state root for one exact source tree without creating it."""

    return _default_root(
        source_tree,
        environment_key="XDG_STATE_HOME",
        home_suffix=(".local", "state"),
        environ=environ,
        home=home,
    )


def default_cache_root(
    source_tree: str,
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return the XDG cache root for one exact source tree without creating it."""

    return _default_root(
        source_tree,
        environment_key="XDG_CACHE_HOME",
        home_suffix=(".cache",),
        environ=environ,
        home=home,
    )


def _default_root(
    source_tree: str,
    *,
    environment_key: str,
    home_suffix: tuple[str, ...],
    environ: Mapping[str, str] | None,
    home: Path | None,
) -> Path:
    _validate_source_tree(source_tree)
    environment = os.environ if environ is None else environ
    configured = environment.get(environment_key)
    if configured:
        base = Path(configured).expanduser()
        if not base.is_absolute():
            raise ValueError(f"{environment_key} must be an absolute path")
    else:
        base = (Path.home() if home is None else home).joinpath(*home_suffix)
        if not base.is_absolute():
            raise ValueError("TxnOpt home directory must be absolute")
    return base / "txnopt" / source_tree


def _validate_source_tree(value: str) -> None:
    if len(value) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("source tree must be a lowercase Git object digest")


__all__ = ["default_cache_root", "default_state_root"]
