# TxnOpt experiments

This directory tracks only new protocol definitions, manifests, registries,
and independently reviewed summaries. Raw run output belongs under ignored
`results/<run_label>/` or a governed external root.

Level 1 holdout access is forbidden. Every failed attempt keeps its raw
evidence under a fresh TxnOpt run label; no historical Stage label is reused.

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

The active formal identity is external Attempt18, bound to Build09 and a fresh
raw root. Its analysis is preregistered in
`level1-analysis-protocol-v1.json`. Structural preflight is read-only:

```bash
python -m tools.run_txnopt_level1_campaign preflight \
  --plan-manifest /home/oneblaze/txnopt-plans/level1-formal-plan-attempt18/manifest.json \
  --analysis-protocol experiments/txnopt/level1-analysis-protocol-v1.json
```

The preflight status is `PASS_NOT_AUTHORIZED_TO_EXECUTE`. The `run` subcommand
also requires an exact Build09 wheel, an isolated Python installation, host
resource checks, and a separately signed
`txnopt-level1-procurement-authorization-v1` receipt. That receipt must bind the
exact plan, analysis protocol, config tree, Build09 manifest, wheel, raw root,
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
