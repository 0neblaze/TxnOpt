# Stage 0 reproduction audit

This directory preserves the second full 36-run execution used to audit the frozen baseline.

- `reproduction_per_run_results.csv`: second-run per-seed metrics.
- `reproduction_summary_results.csv`: summary recomputed from the second-run records.
- `reproduction_environment.json`: second-run environment and source identity.
- `comparison_report.csv`: automatic comparison against the frozen Stage 0 baseline.

The full second-run raw logs and solutions remain under the Git-ignored
`results/stage00-reproduction/` directory. The tracked audit is evidence for the repeatability
finding documented in `docs/stage00_baseline.md`; it is not a replacement baseline.
