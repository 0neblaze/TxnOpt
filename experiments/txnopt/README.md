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
