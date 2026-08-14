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


def resolve_run_output_root(
    configured: object,
    *,
    producer_identity: Mapping[str, object],
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve an explicit raw root or the tree-scoped local state default."""

    source_tree = _producer_source_tree(producer_identity)
    bound = producer_identity.get("binding_status") == "BOUND_CLEAN_BUILD"
    if configured is not None:
        if not isinstance(configured, str) or not configured:
            raise ValueError("output_root must be a non-empty path string when provided")
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            raise ValueError("output_root must be an explicit absolute path")
        if bound:
            return require_governed_external_root(
                candidate,
                source_tree=source_tree,
                environ=environ,
                home=home,
            )
        return candidate.absolute()
    if bound:
        raise ValueError("bound evidence requires an explicit governed output_root")
    return default_state_root(
        source_tree,
        environ=environ,
        home=home,
    ) / "runs"


def require_governed_external_root(
    path: Path,
    *,
    source_tree: str | None = None,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Reject repository-local or rebuildable roots for governed evidence."""

    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise ValueError("governed root must be an explicit absolute path")
    absolute = candidate.absolute()
    _reject_symlink_ancestry(absolute)
    for ancestor in (absolute, *absolute.parents):
        marker = ancestor / ".git"
        if marker.exists() or marker.is_symlink():
            raise ValueError("governed root cannot be inside a Git worktree")
    placeholder_tree = source_tree if source_tree is not None else "0" * 40
    state_base = default_state_root(
        placeholder_tree,
        environ=environ,
        home=home,
    ).parent
    cache_base = default_cache_root(
        placeholder_tree,
        environ=environ,
        home=home,
    ).parent
    if _contains(state_base, absolute) or _contains(cache_base, absolute):
        raise ValueError("governed root cannot be inside rebuildable state or cache")
    return absolute


def _producer_source_tree(producer_identity: Mapping[str, object]) -> str:
    source_tree = producer_identity.get("source_tree")
    if not isinstance(source_tree, str):
        attestation = producer_identity.get("native_build_attestation")
        if isinstance(attestation, Mapping):
            source_tree = attestation.get("source_tree")
    if not isinstance(source_tree, str):
        raise ValueError("producer identity does not provide a source tree")
    _validate_source_tree(source_tree)
    return source_tree


def _contains(parent: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_symlink_ancestry(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError("governed root ancestry cannot contain a symlink")


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


__all__ = [
    "default_cache_root",
    "default_state_root",
    "require_governed_external_root",
    "resolve_run_output_root",
]
