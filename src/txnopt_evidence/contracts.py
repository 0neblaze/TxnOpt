"""Ports that keep raw production separate from independent review."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from txnopt_evidence.identity import ExpectedEvidenceIdentity


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
        config_path: Path,
    ) -> RawArtifactRef: ...


@runtime_checkable
class ReviewerPort(Protocol):
    """Reviewer port: reconstruct only from a raw manifest in a fresh process."""

    def review(
        self,
        *,
        manifest_path: Path,
        output_dir: Path,
        expected_identity: ExpectedEvidenceIdentity,
    ) -> Mapping[str, object]: ...
