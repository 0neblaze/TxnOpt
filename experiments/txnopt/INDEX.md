# TxnOpt experiment index

| Protocol | Status | Scope | Raw evidence | Independent review |
| --- | --- | --- | --- | --- |
| `txnopt-level1-protocol-v1` | build complete, experiment not started | 12 EVRPTW + 24 RCPSP, 10 seeds | build manifest only | pending |
| Level 2 | gated | unopened | none | none |
| Level 3 | gated | unopened | none | none |

`level1-protocol.json` and its SHA-256 sidecar are the immutable case-set
identity for new Level 1 runs. Creating the protocol does not authorize cloud
purchase, benchmark launch, holdout access, or a readiness claim.

The current internal build identity is recorded in
`manifests/txnopt_level1_build_attempt01.json`. Its status is deliberately
`BUILD_COMPLETE_NOT_LEVEL1_READY`: build success is not experiment readiness.
