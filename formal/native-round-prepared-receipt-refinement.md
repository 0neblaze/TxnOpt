# Native round prepared-receipt refinement note

Status: `REVIEW_PENDING_BUILD10`

This additive note does not modify or supersede the signed Attempt04/Attempt05
refinement package. A successor independent review must bind this file together
with the Build10 source, runtime, native adapter, and evidence reviewer before
it can be used as Level 1 evidence.

The `txnopt-native-round-v1` module records only a private prepared transaction:

```text
PREPARED -> RESERVED -> EVALUATING -> VALIDATED / INTERRUPTED / ABORTED
```

Its receipt binds the complete-work reservation, ordered worker receipts,
prepared cache-write count and key checksum, phase trace, and source identity.
The EVRPTW adapter drains each receipt exactly once into a
`txnopt-physical-trace-v1` `native_round_observation`. The independent evidence
reviewer rechecks the phase, budget, task-order, prepared-cache, worker-policy,
and source-identity invariants.

This native state is a stuttering implementation step with respect to the
aggregate formal state. It cannot publish the Python incumbent, semantic
digest, solve-wide budget, or cache. Only `TxnRuntime` may validate the returned
candidate state and perform the single aggregate Python commit. An interrupted
or aborted native receipt therefore has zero prepared cache writes, while a
validated receipt remains private until that Python commit succeeds.
