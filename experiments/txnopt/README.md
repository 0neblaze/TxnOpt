# TxnOpt experiments

This directory tracks only new protocol definitions, manifests, registries,
and independently reviewed summaries. Raw run output belongs under ignored
`results/<run_label>/` or a governed external root.

Level 1 holdout access is forbidden. Every failed attempt keeps its raw
evidence under a fresh TxnOpt run label; no historical Stage label is reused.

The exact Level 1 case set, seeds, axes, and cloud envelope are bound by
`level1-protocol.json` and its sidecar. `INDEX.md` is the current status view;
it must never infer readiness from runner completion.
