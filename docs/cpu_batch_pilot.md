# CPU batch exact-charging pilot

`cpu_batch_pilot_attempt01` is an independent CPU performance experiment. It
does not claim Stage 3.3 readiness and does not modify Stage 3.0--3.2 evidence.

The implementation exposes two exact-charging backends:

- `cpu_scalar`: the historical scalar reference;
- `cpu_batch`: ordered multi-route label expansion on the CPU.

CPU batching preserves screening order, cache hits and misses, bounded LRU
stores and evictions, exact-call counts, route results, objective tuples and
fixed-work iterations. Only contiguous exact cache misses are combined.

Run the phases independently:

```bash
uv run python -m evrptw.experiments.cpu_batch_pilot replay
uv run python -m evrptw.experiments.cpu_batch_pilot paired
uv run python -m evrptw.experiments.cpu_batch_pilot_review
```

Or run both evidence-generation phases with:

```bash
uv run python -m evrptw.experiments.cpu_batch_pilot all
```

The separate review command must still be run afterwards; the evidence runner
cannot publish summaries.

Raw evidence is written below `results/cpu_batch_pilot_attempt01/`. The review
verifies the raw manifest and SHA-256 sidecar before writing summaries. The
accepted `attempt01` completed all 12 paired runs with identical candidate
work, exact-call counts, route-result hashes, objectives, iterations and
validator status. The family median end-to-end savings were 21.97% for
`c101_21`, 11.54% for `r101_21`, and 24.40% for `rc101_21`; all three improved
and their median was 21.97%. C5 ranged from 8.66% slower to 4.54% faster, with
a -0.12% median, and remains a reported control. The predeclared gate passed
after the separate reviewer independently replayed the manifest, candidate
work, formal replay protocol, validator, objective, final-route hash and exact
call counts. Therefore `solve_alns()` now
defaults to `cpu_batch`; historical Stage 0--3.2 runners remain explicitly
pinned to `cpu_scalar`.
