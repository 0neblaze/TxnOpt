# Source Code

The `evrptw` Python package and its native C++ extension live in this directory.

From the repository root:

```bash
uv sync --all-groups
uv run evrptw-env
```

The package currently provides:

- a parser for Schneider E-VRPTW benchmark files;
- route feasibility checks for capacity, battery, and time windows;
- deterministic two-opt/VNS scaffolding;
- a C++20 extension for route-distance and two-opt calculations;
- reproducible environment and run metadata.

Large benchmark datasets and generated run outputs are intentionally excluded
from Git. See [`docs/environment.md`](../docs/environment.md) for setup details.
