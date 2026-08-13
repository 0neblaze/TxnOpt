"""Ports that keep raw production separate from independent review."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from txnopt import RunConfig


@dataclass(frozen=True, slots=True)
class RawArtifactRef:
    run_label: str
    manifest_path: Path
    manifest_sha256: str


@runtime_checkable
class RunnerPort(Protocol):
    """Producer port: create raw artifacts, never a readiness decision."""

    def run(
        self,
        *,
        run_label: str,
        output_dir: Path,
        config: RunConfig,
    ) -> RawArtifactRef: ...


@runtime_checkable
class ReviewerPort(Protocol):
    """Reviewer port: reconstruct only from a raw manifest in a fresh process."""

    def review(
        self,
        *,
        raw: RawArtifactRef,
        output_dir: Path,
    ) -> Mapping[str, object]: ...
