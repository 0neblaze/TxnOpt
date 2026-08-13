# TxnOpt experiments

This directory tracks only new protocol definitions, manifests, registries,
and independently reviewed summaries. Raw run output belongs under ignored
`results/<run_label>/` or a governed external root.

Level 1 holdout access is forbidden. Every failed attempt keeps its raw
evidence under a fresh TxnOpt run label; no historical Stage label is reused.
Build11 and later raw/failure bundles use `txnopt-raw-artifact-v2` or
`txnopt-failure-artifact-v2` with the hash-chained
`txnopt-evidence-lifecycle-v1` contract. The producer seals
`PLANNED -> RUNNING -> SEALED`; the independent replay appends `REVIEWED`.
Build10 v1 bundles remain read-only and replayable without retroactive edits.

The exact Level 1 case set, seeds, axes, and cloud envelope are bound by
`level1-protocol.json` and its sidecar. `INDEX.md` is the current status view;
it must never infer readiness from runner completion.

Benchmark files are not redistributed. Copy `case-catalog.example.json` to the
ignored `case-catalog.local.json`, replace every placeholder with an absolute
Schneider or PSPLIB path, and materialize (but do not run) the matrix with:

```bash
python tools/materialize_txnopt_level1.py \
  --protocol experiments/txnopt/level1-protocol.json \
  --catalog experiments/txnopt/case-catalog.local.json \
  --destination /external/txnopt-level1-plan-attempt01 \
  --raw-output-root /external/txnopt-level1-raw \
  --fixed-work CALIBRATED_WORK --fixed-time-seconds CALIBRATED_SECONDS \
  --max-rounds CALIBRATED_ROUNDS
```

`CALIBRATED_*` values are frozen only after the local pilot. The materializer
requires an exact catalog, hashes every source instance and generated config,
refuses overwrite, leaves holdouts closed, and does not execute or procure
anything. `tools/estimate_txnopt_cloud_window.py` converts the complete local
p95 calibration into a signed 32/64-core time forecast. Server rental remains
blocked unless that forecast is at most ten days.

The prior formal identities Attempts18 and 21 remain immutable and unexecuted.
The current post-calibration Build11 identity is external Attempt23, with its
own preregistered analysis file and a fresh absent raw root. Structural
preflight is read-only:

Build11 freezes the active solver source as
`manifests/txnopt_level1_build_attempt11.json`. Build10 and all earlier
calibrations, plans, and pre-cloud gates remain immutable but cannot authorize
Build11. Build11 lifecycle/formal refinement is explicitly pending independent
review; Attempt22 calibration and Attempt23 plan use fresh identities.

```bash
python -m tools.run_txnopt_level1_campaign preflight \
  --plan-manifest /home/oneblaze/txnopt-plans/level1-formal-plan-attempt23/manifest.json \
  --analysis-protocol /home/oneblaze/txnopt-plans/level1-formal-plan-attempt23/analysis-protocol.json
```

Attempt23 preflight status is `PASS_NOT_AUTHORIZED_TO_EXECUTE`.
The `run` subcommand
also requires the exact plan-bound Build11 wheel, an isolated Python installation, host
resource checks, and a separately signed
`txnopt-level1-procurement-authorization-v1` receipt. That receipt must bind the
exact plan, analysis protocol, config tree, Build11 manifest, wheel, raw root,
exclusive-Linux contract, and 14-day maximum window. Before the first raw
write, the runner creates an immutable atomic launch claim and records that no
target process is active. Each run executes in its own process group; timeout
cleanup terminates and verifies the whole group rather than leaving a writer
behind. It writes raw bundles and one raw-only execution receipt; it cannot
review or declare readiness. After a complete run,
`python -m tools.review_txnopt_level1_campaign` independently
replays every raw manifest into a new review root and recomputes the registered
performance, confidence-interval, parity, prefix-safety, fallback, and Cmax
gates. Failed attempts retain their raw root and require a new attempt label.
The representative Build11 fault gate is retained as Attempt02. The current
bounded-exhaustive local gate is
`manifests/txnopt_level1_fault_gate_attempt03.json`; its first orchestration
failure remains separately retained. The current Build11 decision receipt is
`manifests/txnopt_level1_precloud_gate_attempt05.json`. It is explicitly
`BLOCKED_BUILD11_INDEPENDENT_REVIEW_PENDING`, not an authorization receipt and
not a Level 1 readiness claim. Pre-cloud Attempts01-04 remain immutable prior
results.
