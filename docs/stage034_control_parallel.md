# Stage 3.4 Candidate Control and Controlled Parallel

Stage 3.4 adds an opt-in candidate-control layer and a deterministic
four-process exact-evaluation path. It does not change the Stage 0--3.3 path
when `candidate_control_config=None`.

## Public configuration

`CandidateControlConfig` records `proposal_top_k`,
`max_exact_calls_per_round`, `worker_count`, `executor_model=process_spawn`,
the ranking policy, and the merge policy. The preregistered Smoke grid is
`{1,2,4} x {1,2,4}`. The checked-in selected configuration must remain inside
that grid; changing it requires a new attempt rather than overwriting evidence.

The rank key is:

1. vehicle count;
2. optimistic total-distance lower bound;
3. changed-route count;
4. canonical route key;
5. original proposal ordinal.

Safe screening rejection, cache hit, unchanged route, and a supplied
precomputed route do not consume the round exact-call budget. A group of cache
misses required by one complete candidate is reserved atomically. A skipped
candidate is recorded as candidate-control skip and is never represented as an
exact-infeasible result or written to a negative cache.

## Parallel transaction

The four-worker variant creates one reusable macOS-compatible `spawn` process
pool for the solve lifetime. Work is split into deterministic contiguous
chunks. Completion order is evidence only; the parent merges in submission
order. An initializer, worker, timeout, or result-count failure raises
immediately, cancels pending work, discards the candidate transaction, and does
not fall back to serial execution or `cpu_scalar`.

`Stage03Trace` v4 records candidate decisions, budget reservation and
remainder, worker/submission/completion/merge ordering, candidate-work hash,
route-result hash, backend metrics, and per-round statistics. Complete route
sequences remain in the route dictionary rather than being duplicated in event
rows.

## Evidence protocol

The runner produces these four axes for every instance/seed:

- `serial_fixed_exact_calls`;
- `parallel_fixed_exact_calls`;
- `serial_wall_clock`;
- `parallel_wall_clock`.

Smoke uses six instances and three seeds (72 axes). Formal uses the Stage 0
twelve instances and three seeds (144 axes). Both use 1000 maximum iterations,
100 fixed-work started calls, a 120-second fixed-work watchdog, 30 wall-clock
seconds, `cpu_batch`, and `artifact-storage-v1`.

The independent reviewer reads the manifest first and then recomputes raw
solutions, objectives, event counts, route references, candidate budgets,
parallel ordering, exact-call reconciliation, screening/cache/backend/deadline
statistics, and fixed serial/parallel trajectory hashes. It compares every
wall-clock objective with the accepted Stage 3.3 wall-clock instance/seed.

The reviewer writes `READY_FOR_STAGE04` only if all correctness, provenance,
objective, deterministic-equivalence, and performance gates pass. Until an
accepted Formal review exists, Stage 3.4 and Stage 3 performance remain
incomplete.

## Current evidence status

`stage03.4_control_parallel_attempt05` completed the full 72-axis Smoke scope
but independently reviewed as `NOT_READY`. Attempts 06--09 remain preserved as
partial, failed, or intermediate warm-start evidence. The corrected
transactional warm-start protocol passed as Smoke
`stage03.4_control_parallel_attempt10` (72/72 axes,
`READY_FOR_STAGE034_FORMAL`) and Formal
`stage03.4_control_parallel_attempt11` (144/144 axes,
`READY_FOR_STAGE04`). The published registry and manifest retain every earlier
attempt without overwriting historical bytes.
