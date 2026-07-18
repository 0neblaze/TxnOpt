from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PACKAGES = (
    "cplex",
    "docplex",
    "gurobipy",
    "highspy",
    "jupyterlab",
    "matplotlib",
    "networkx",
    "numba",
    "numpy",
    "ortools",
    "orjson",
    "pandas",
    "psutil",
    "pyarrow",
    "pybind11",
    "scipy",
    "seaborn",
)


def _command_version(command: str, *arguments: str) -> str | None:
    executable = shutil.which(command)
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, *arguments], capture_output=True, text=True, check=False
    )
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return output[0] if output else "available"


def collect_environment() -> dict[str, Any]:
    try:
        native_extension = importlib.import_module("evrptw._core").__file__
    except ImportError as error:
        native_extension = f"unavailable: {error}"

    metadata: dict[str, Any] = {
        "captured_at": datetime.now(UTC).isoformat(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "gpu_used": False,
        },
        "tools": {
            "cmake": _command_version("cmake", "--version"),
            "ninja": _command_version("ninja", "--version"),
            "clang": _command_version("clang", "--version"),
            "git": _command_version("git", "--version"),
            "uv": _command_version("uv", "--version"),
        },
        "packages": {},
        "native_extension": native_extension,
    }
    for package in _PACKAGES:
        try:
            metadata["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata["packages"][package] = None

    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the EVRP-TW research environment")
    parser.add_argument("--output", type=Path, help="also write the report to this JSON file")
    arguments = parser.parse_args()
    report = collect_environment()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
