# Experiment evidence index

This directory contains reviewable evidence, not a flat collection of
successful results.

## Status vocabulary

- `accepted`: independent raw replay passed every mandatory gate.
- `superseded`: historically valid evidence replaced by a stricter protocol.
- `failed` or `NOT_READY`: retained failure evidence; never a performance claim.
- `partial`: interrupted or incomplete evidence.
- `roadmap`: planned work with no implementation claim.

## Accepted chain

| Stage | Accepted evidence |
|---|---|
| Stage 0 | `experiments/baselines/stage00/` |
| Stage 2.3 | `stage02_constraint_guided_attempt16` and `rerun09` |
| Stage 3.4 | `stage03.4_control_parallel_attempt11`, `READY_FOR_STAGE04` |
| Stage 4 | `stage04_adaptive_weights_attempt15`, `READY_FOR_STAGE05` |
| Stage 5.1 | `stage05.1_best_known_attempt06`, `READY_FOR_STAGE05_2` |
| Stage 5.2 Pilot | `stage05.2_benchmark_attempt72`, `READY_FOR_STAGE052_FORMAL_BENCHMARK` |

Stage 5.2 Formal `attempt73` is incomplete and unreviewed. Its archived and
staging bytes are external to Git and indexed in `artifacts/index.json`.

## Directory roles

- `baselines/`: immutable frozen comparison material.
- `summaries/`: tracked review products and historical failure/supersession
  records.
- `registries/`: artifact identities, paths, checksums, and legacy mappings.
- `manifests/`: accepted stage-level publication manifests.
- `migrations/`: signed storage-migration attestations.

Raw evidence belongs in ignored `results/` or a verified external archive. A
summary is publishable only after its independent reviewer has re-read the raw
artifacts and reproduced the validator, objective, counters, and hashes.
