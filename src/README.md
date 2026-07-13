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

Stage 2.2 adds the `stage02_route_quality` profile on top of the Stage 2.1
route-reduction operators. Its cross-route deep module provides `relocate`,
`swap`, `two_opt_star`, `route_segment_destroy`, and bounded `ejection_chain`
proposals. The profile is the current `solve_alns()` default; historical
profiles remain explicit and reproducible.

The formal Stage 2.2 experiment is run with:

```bash
uv run python -m evrptw.experiments.stage02_route_quality \
  --config configs/stage02_route_quality.toml
```

Raw outputs are written under ignored `results/`; tracked summaries are written
under `experiments/summaries/`. The fixed comparison baseline is the successful
Stage 2.1 per-run summary at
`experiments/summaries/stage02_attempt02_per_run_results.csv`.
