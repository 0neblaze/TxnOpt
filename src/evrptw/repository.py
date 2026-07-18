"""Fail-fast discovery of the active EVRP-TW repository checkout."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def repository_root(start: Path | None = None) -> Path:
    """Return the active project checkout for source and wheel installations.

    A wheel's ``__file__`` lives below ``site-packages`` and cannot identify the
    checkout whose configs, data, and Git provenance are being used.  Prefer an
    explicit environment binding, then the supplied path, then the current
    working directory.  Every candidate must be both a Git worktree and this
    project; ambiguity fails instead of silently selecting an unrelated repo.
    """

    explicit = os.environ.get("EVRPTW_REPOSITORY_ROOT")
    candidates = [
        *([Path(explicit).expanduser()] if explicit else []),
        *([start] if start is not None else []),
        Path.cwd(),
        Path(__file__).resolve(),
    ]
    failures: list[str] = []
    seen: set[Path] = set()
    for candidate in candidates:
        base = candidate.resolve()
        if base.is_file():
            base = base.parent
        if base in seen:
            continue
        seen.add(base)
        result = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            failures.append(f"{base}: not a Git worktree")
            continue
        root = Path(result.stdout.strip()).resolve()
        if not (root / "pyproject.toml").is_file() or not (
            root / "configs" / "stage052_performance.toml"
        ).is_file():
            failures.append(f"{root}: not the EVRP-TW project checkout")
            continue
        return root
    detail = "; ".join(failures) or "no repository candidates were available"
    raise RuntimeError(
        "cannot locate the active EVRP-TW repository; run from the checkout or "
        f"set EVRPTW_REPOSITORY_ROOT ({detail})"
    )
