# Apple M5/Metal GPU batch acceleration pilot

This is an independent performance pilot. It does not implement or claim
Stage 3.3 readiness and does not modify the historical Stage 3.0--3.2 raw
evidence.

## Scope and gate

The fixed scope is `c101C5`, `c101_21`, `r101_21`, and `rc101_21`, with seeds
`2014`, `2015`, and `2016`. The profiler phase runs the current
`stage02_constraint_guided` CPU-only solver and stores `cProfile` output for
one run of each 100-customer instance. GPU replay is blocked unless the exact
charging median time share is at least 50% for all three 100-customer families.

## Backends

The explicit exact-charging backends are:

- `cpu_scalar`: the existing scalar label-setting reference.
- `cpu_batch`: the same label manager with ordered transition batches.
- `metal_batch`: the Objective-C++/Metal bridge computes only transition
  arithmetic. The CPU retains the priority queue, label dominance and path
  reconstruction.

Metal capability errors, precision disagreements, and label-buffer overflow
raise immediately. There is no silent CPU fallback. The default
`solve_alns()` path remains `cpu_scalar`.

## Reproducible workflow

From the repository root:

```text
uv run python -m evrptw.experiments.gpu_batch_pilot profile
uv run python -m evrptw.experiments.gpu_batch_pilot replay
uv run python -m evrptw.experiments.gpu_batch_pilot paired
uv run python -m evrptw.experiments.gpu_batch_pilot review
```

The profiler and replay phases write raw evidence under
`results/gpu_batch_pilot_attempt01/`, with a raw manifest and SHA-256 sidecar.
Replay uses three warm-ups and five
measured repetitions, and sweeps batch sizes `32`, `128`, and `512` for the
batch backends. The fixed-work ALNS pair uses 40 iterations, disables
screening and cache, and treats 120 seconds only as a watchdog. Every result
records total time, exact-charging time, kernel time, transfer time, packing and
unpacking time, launches, transitions, exact calls, work hash, objective and
validator status.

Only `review` may create the tracked pilot summaries in
`experiments/summaries/`. A paired run is valid only when candidate-work hash,
exact-call count, route-result hash, objective tuple and effective iterations
match exactly. The default go/no-go criterion is at least 10% median end-to-end
saving across the three 100-customer families and real improvement for at
least two families; C5 is reported separately because fixed launch and
transfer overhead may dominate it.
