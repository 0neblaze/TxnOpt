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

Stage 2.3 adds the `stage02_constraint_guided` profile on top of the accepted
Stage 2.2 route-quality trajectory. Its deep module provides deterministic
constraint-guided removals, dynamic removal tiers, and an observable constraint
lane. The profile is the current `solve_alns()` default; `baseline`,
`stage02_route_reduction`, and `stage02_route_quality` remain explicit and
reproducible historical profiles.

The Stage 2.2 cross-route operators remain available through the same deep
module: `relocate`, `swap`, `two_opt_star`, `route_segment_destroy`, and bounded
`ejection_chain`.

The formal Stage 2.3 experiment is run with:

```bash
uv run python -m evrptw.experiments.stage02_constraint_guided \
  --config configs/stage02_constraint_guided.toml \
  --output-dir results/stage02-constraint-guided_attempt16 \
  --run-label stage02_constraint_guided_attempt16

uv run python -m evrptw.experiments.stage02_constraint_guided_review \
  --run-dir results/stage02-constraint-guided_attempt16 \
  --comparison-dir results/stage02-quality_attempt02 \
  --review-label stage02_constraint_guided_attempt16

uv run python -m evrptw.experiments.stage02_constraint_guided \
  --config configs/stage02_constraint_guided.toml \
  --output-dir results/stage02-constraint-guided-rerun09 \
  --run-label stage02_constraint_guided_rerun09 \
  --repeat-of results/stage02-constraint-guided_attempt16

uv run python -m evrptw.experiments.stage02_constraint_guided_review \
  --run-dir results/stage02-constraint-guided-rerun09 \
  --comparison-dir results/stage02-quality_attempt02 \
  --review-label stage02_constraint_guided_rerun09
```

Raw outputs are written under ignored `results/`; tracked summaries and review
artifacts are written under `experiments/summaries/`. The fixed Stage 2.3
comparison baseline is the successful Stage 2.2 per-run summary at
`experiments/summaries/stage02_quality_attempt02_per_run_results.csv`.
