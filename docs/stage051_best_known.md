# Stage 5.1 — Best-Known Values

## Purpose

Compile formally published Schneider E-VRPTW best-known solution (BKS)
values for all 92 benchmark instances, assess model compatibility with our
lexicographic objective, and publish the reference data as a canonical
artifact bundle.

## Data Sources

| Abbreviation | Citation | VOR in collection |
|---|---|---|
| SSG | Schneider, Stenger, Goeke (2014), *Transportation Science* 48(4), 500–520, DOI: 10.1287/trsc.2013.0490 | yes |
| GS | Goeke, Schneider (2015), *EJOR* 245(1), 81–99, DOI: 10.1016/j.ejor.2015.01.049 | no |
| HPH | Hiermann, Puchinger, Ropke, Hartl (2016), *EJOR* 252(3), 995–1018, DOI: 10.1016/j.ejor.2016.01.038 | yes |
| KC | Keskin, Catay (2016), *TRC* 65, 111–127, DOI: 10.1016/j.trc.2016.01.013 | yes |

Small instances (36): BKS values extracted from Schneider et al. (2014)
Table 5 (page 511). CPLEX optimal or best upper bound within 7200 seconds.
For RC204-15, VNS/TS found a better solution (384.86) than the CPLEX upper
bound (407.45); the BKS is the VNS/TS value.

Large instances (56): BKS values extracted from Keskin and Catay (2016)
Table 2 (page 122), which cites the original source (SSG, GS, or HPH) for
each value. These are the most up-to-date full-recharge BKS values as of
2016.

## Model Compatibility Assessment

| Dimension | Our Model | Published Model | Compatible |
|---|---|---|---|
| charging_model | full_recharge: t = (Q - b) * g | full_recharge (SSG, GS, HPH) | yes |
| objective_function | lexicographic(vehicle_count, total_distance, total_charging_time, charging_count) | SSG/GS: lexicographic(vehicle_count, total_distance); HPH: weighted_sum | no |
| distance_metric | unrounded Euclidean (math.hypot) | Euclidean, Solomon convention (possibly rounded) | no |
| vehicle_parameters | Schneider benchmark (unchanged) | Schneider benchmark (unchanged) | yes |
| time_windows | Schneider-rewritten Solomon (unchanged) | Schneider-rewritten Solomon (unchanged) | yes |

**Overall: NOT compatible.** The objective function and distance metric
differ. Per roadmap policy, no gap is computed. BKS values are listed as
reference only.

Charging time and charging count are `unknown` for all instances because
published BKS tables report only vehicle count and total distance.

## Artifact Layout

```
results/stage05.1_best_known_attemptNN/
  control/
    stage05.1_best_known_attemptNN_run_metadata.json
    stage05.1_best_known_attemptNN_config.toml
    stage05.1_best_known_attemptNN_manifest.json
    stage05.1_best_known_attemptNN_manifest.sha256
  stage05.1_best_known_attemptNN_bks_data.csv
  stage05.1_best_known_attemptNN_compatibility_assessment.csv
  stage05.1_best_known_attemptNN_summary_report.md
  review/
```

## Review Gates

| Gate | Description |
|---|---|
| instance_coverage | All 92 instances present in BKS CSV |
| bks_values_present | All instances have bks_vehicles and bks_distance (not `unknown`) |
| no_gap_computation | No gap columns in BKS CSV |
| compatibility_assessment_correct | Compatibility CSV matches canonical assessment |
| replay_consistency | CSV BKS values match canonical data from `evrptw.best_known` |

Review status `READY_FOR_STAGE05_2` requires all five gates to pass.

The v5 reviewer compares every canonical BKS and compatibility field,
including source/compilation references, DOI, charging fields, and
`model_compatible`; it requires the canonical single-level
`results/<run_label>/` layout, a clean revision, and the published Stage 4 v5
prerequisite. Duplicate, missing, additional, or malformed rows fail the
review immediately. Earlier evidence remains historical and cannot be promoted
by the v5 reviewer.

## Accepted Evidence

`stage05.1_best_known_attempt02` is preserved v2 evidence. Its raw bundle uses
the canonical single-level layout and contains exactly 92 unique instance rows,
but it is superseded by the v3 Stage 4 prerequisite and independent-conversion
replay requirements. A new attempt must pass those gates before
`READY_FOR_STAGE05_2` is republished. `stage05.1_best_known_attempt01` remains
superseded v1 evidence under its historical nested layout.

The v3 evidence `stage05.1_best_known_attempt03` is preserved but superseded
because its Stage 4 v3 prerequisite was withdrawn. v4 attempt04 is also
preserved but superseded after the Stage 4 v4 prerequisite was withdrawn.
`stage05.1_best_known_attempt05` uses the published v5 Stage 4 formal review
(`stage04_adaptive_weights_attempt13`) and passes all five independent replay
gates for exactly 92 unique instances. It is the current
`READY_FOR_STAGE05_2` evidence; no gap field or model-inconsistent gap is
published.
