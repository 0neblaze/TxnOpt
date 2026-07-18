"""Freeze the non-editable wheel runtime used by current Stage 5.2 evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from evrptw.stage052_evidence import create_stage052_runtime_identity


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/stage052_runtime_identity.local.json"),
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("runtime identity may be frozen only from a clean commit")
    revision = _git(root, "rev-parse", "HEAD")
    output = arguments.output if arguments.output.is_absolute() else root / arguments.output
    identity = create_stage052_runtime_identity(
        output_path=output,
        wheel_path=arguments.wheel,
        repository_revision=revision,
    )
    print(json.dumps(identity, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
