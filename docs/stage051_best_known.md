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

The v2 reviewer compares every BKS and compatibility field, requires the
canonical `results/<run_label>/` layout, verifies the clean revision and the
published Stage 4 v2 prerequisite, and rejects duplicate, missing, additional,
or malformed rows. Stage 5.1 v1 evidence remains historical and cannot be
promoted by the v2 reviewer.
