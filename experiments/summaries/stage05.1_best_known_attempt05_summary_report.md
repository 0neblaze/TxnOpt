# Stage 5.1 Best-Known Values — stage05.1_best_known_attempt05

## Overview

This report compiles best-known solution (BKS) values for all 92 Schneider et al. (2014) E-VRPTW benchmark instances and assesses model compatibility with our lexicographic objective.

## Data Sources

- **SSG**: Schneider, Stenger, Goeke (2014), "The Electric Vehicle-Routing Problem with Time Windows and Recharging Stations", *Transportation Science*, DOI: 10.1287/trsc.2013.0490, in VOR collection: yes
- **GS**: Goeke, Schneider (2015), "Routing a mixed fleet of electric and conventional vehicles", *European Journal of Operational Research*, DOI: 10.1016/j.ejor.2015.01.049, in VOR collection: no
- **HPH**: Hiermann, Puchinger, Ropke, Hartl (2016), "The Electric Fleet Size and Mix Vehicle Routing Problem with Time Windows and Recharging Stations", *European Journal of Operational Research*, DOI: 10.1016/j.ejor.2016.01.038, in VOR collection: yes
- **KC**: Keskin, Catay (2016), "Partial Recharge Strategies for the Electric Vehicle Routing Problem with Time Windows", *Transportation Research Part C*, DOI: 10.1016/j.trc.2016.01.013, in VOR collection: yes

## Model Compatibility Assessment

**Overall compatible: False**

| Dimension | Our Model | Published Model | Compatible |
|-----------|----------|-----------------|------------|
| charging_model | full_recharge: t = (Q - b) * g | full_recharge (SSG, GS, HPH) | yes |
| objective_function | lexicographic(vehicle_count, total_distance, total_charging_time, charging_count) | SSG/GS: lexicographic(vehicle_count, total_distance); HPH: weighted_sum(fixed_cost * vehicle_count + total_distance) | no |
| distance_metric | unrounded Euclidean (math.hypot) | Euclidean, Solomon convention (possibly rounded) | no |
| vehicle_parameters | Schneider benchmark instances (unchanged) | Schneider benchmark instances (unchanged) | yes |
| time_windows | Schneider-rewritten Solomon time windows (unchanged) | Schneider-rewritten Solomon time windows (unchanged) | yes |

Model is NOT fully compatible.  The objective function and distance metric differ.  Per roadmap policy, no gap is computed.  BKS values are listed as reference only.

## BKS Values Summary

- Small instances (5/10/15 customers): 36
- Large instances (100 customers): 56
- Total: 92

All BKS values are marked `model_compatible=False` due to objective and distance-metric mismatch. No gap is computed.

Charging time and charging count are `unknown` for all instances because published BKS tables report only vehicle count and total distance.

## Provenance

- Git revision: `3921aa48a8d524e73a0c56ad41243301c6d314cf`
- Schema version: `stage05.1-best-known-v5`
