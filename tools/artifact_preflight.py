#!/usr/bin/env python3
"""Read-only preflight for current and legacy experiment bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evrptw.artifacts import (
    CURRENT_STORAGE_FORMAT,
    ArtifactReader,
    canonical_run_label_is_valid,
)


def preflight_run(run_dir: Path) -> dict[str, Any]:
    reader = ArtifactReader(run_dir)
    if reader.storage_format != CURRENT_STORAGE_FORMAT:
        return {
            "run_dir": str(run_dir),
            "status": "legacy_compatible",
            "storage_format": reader.storage_format,
            "policy_compliance": "legacy_compatible",
        }
    label = str(reader.manifest.get("run_label", run_dir.name))
    if not canonical_run_label_is_valid(label):
        raise ValueError(f"non-canonical current run label: {label}")
    artifacts = reader.manifest.get("artifacts", [])
    required_fields = {
        "storage_policy_version",
        "storage_format",
        "compression",
        "retention_class",
        "evidence_completeness",
        "checksum",
        "byte_size",
    }
    missing = [
        str(item.get("relative_path", ""))
        for item in artifacts
        if not required_fields.issubset(item)
    ]
    if missing:
        raise ValueError(f"current artifacts lack storage metadata: {missing}")
    return {
        "run_dir": str(run_dir),
        "run_label": label,
        "status": str(reader.manifest.get("status", "unknown")),
        "storage_format": CURRENT_STORAGE_FORMAT,
        "policy_compliance": "current",
        "evidence_completeness": reader.manifest.get("evidence_completeness"),
        "artifact_count": len(artifacts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight experiment artifact bundles")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    root = arguments.run_dir or arguments.results_root
    run_dirs = (
        [root]
        if arguments.run_dir
        else sorted(path for path in root.iterdir() if path.is_dir())
    )
    reports: list[dict[str, Any]] = []
    try:
        for run_dir in run_dirs:
            reports.append(preflight_run(run_dir))
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "fail", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "pass", "runs": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
