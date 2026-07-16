# Stage 4 adaptive-weights independent review

- Status: `READY_FOR_STAGE05`
- Run label: `stage04_adaptive_weights_attempt05`
- Scope: `formal`
- Axes reviewed: `144/144`

## Gate results

| Gate | Status | Details |
|------|--------|---------|
| operator_call_sufficiency | PASS | all wall_clock axes passed per-operator segment audit |
| six_category_statistics | PASS | all adaptive_wall_clock axes have complete reconciled six-category statistics |
| adaptive_better_than_fixed | PASS | adaptive wins 4 (instance, seed) pairs; need >= 3. won: [('c101_21', 2014), ('c101_21', 2016), ('r105C15', 2016), ('rc101_21', 2015)] |
| not_single_best_seed | PASS | adaptive wins on 3 unique seeds: [2014, 2015, 2016]; need >= 2. |
| std_not_increased | PASS | Stage 4 vehicle_count std <= Stage 0 for all instances |
| replay_consistency | PASS | exact 144-axis scope; all objectives and exact-call counts match |

- Adaptive wins: `4`
- Winning seeds: `[2014, 2015, 2016]`
