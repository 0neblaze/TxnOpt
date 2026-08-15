# TxnOpt experiments

This directory tracks only new protocol definitions, manifests, registries,
and independently reviewed summaries. Raw run output belongs under ignored
`results/<run_label>/` or a governed external root.

The retained Build19 Tencent chain uses `level1-protocol-v2.json`, external
formal plan Attempt30, representative local calibration Attempt31, and
Pre-cloud Attempt09. Independent Review13 rejected its final account-ready
claim with three major findings. Build20, external formal plan Attempt32,
representative calibration Attempt33, deployment Attempt03, and producer-side
Pre-cloud10 are retained after Independent Review14 found one remaining CVM
credential-acquisition exception leak. Build21, external formal plan Attempt34,
representative calibration Attempt35, deployment Attempt04, and Pre-cloud11 are
also retained as rejected evidence: Independent Review15 found that redacted
SDK errors still retained credential material through Python `__context__`.
Build22 is retained as a failed build because its first offline invocation
bound an empty UV cache. Build23 is the clean successor: its wheel surface,
368 installed-wheel tests, five fixed-seed property suites, both sanitizer
smokes, 100,000-round soak, credential-context adversarial tests, and live TLC
all pass. External formal plan Attempt38 now binds 2,880 configs and 2,880
pre-run expected identities, passes structural preflight, and keeps its raw
root and launch claim absent. The remaining successor sequence is
representative calibration Attempt39, deployment Attempt05, Pre-cloud12, and
Independent Review16. Attempt38 requires at least 64
physical cores, `CoreCount=64`, `ThreadPerCore=1`, and provider-reported memory
of at least 128 GB; Linux-visible memory is observational rather than a 128-GiB
admission threshold. Region has no default and remains a mandatory live input.
Attempt35 retains 96 raw bundles and 96 independent reviews, while the Attempt34
formal raw root and launch claim remain absent. The Build21 deployment Attempt04
is offline-only and records the expected local 12-physical-core doctor failure.
Only a zero-finding Review16 may accept the Build23 successor; none of these
artifacts authorizes cloud access or Level 1 readiness.

Level 1 holdout access is forbidden. Every failed attempt keeps its raw
evidence under a fresh TxnOpt run label; no historical Stage label is reused.
Build14 raw/failure bundles use `txnopt-raw-artifact-v3` or
`txnopt-failure-artifact-v3` with the hash-chained
`txnopt-evidence-lifecycle-v1` contract. The producer seals
`PLANNED -> RUNNING -> SEALED`; the independent replay appends `REVIEWED`.
Build10 v1 and Build11 v2 bundles remain read-only and replayable only through
explicit compatibility entry points without retroactive edits.
Attempt09 independently rejected Build11's self-contained v2 evidence boundary:
its sidecars proved internal consistency but did not anchor producer, config,
run-label, and result identity to a pre-run trust root. The successor schema is
`txnopt-raw-artifact-v3`/`txnopt-failure-artifact-v3`; formal replay requires a
separate `txnopt-expected-evidence-identity-v1` derived from the frozen plan,
exact config bytes, and clean build manifest. V1/v2 replay remains an explicit
compatibility operation and cannot enter a formal readiness decision.

The exact active Level 1 case set, seeds, axes, and Tencent resource envelope
are bound by `level1-protocol-v2.json` and its sidecar. The unversioned
`level1-protocol.json` is the immutable protocol-v1 predecessor and must not be
used for a new plan. `INDEX.md` is the status view; it must never infer
readiness from runner completion.

Benchmark files are not redistributed. Copy `case-catalog.example.json` to the
ignored `case-catalog.local.json`, replace every placeholder with an absolute
Schneider or PSPLIB path, and materialize a fresh successor plan without
running it through the installed unified CLI:

```bash
txnopt plan \
  --protocol experiments/txnopt/level1-protocol-v2.json \
  --catalog experiments/txnopt/case-catalog.local.json \
  --destination /governed/txnopt-plans/level1-formal-plan-attemptN \
  --raw-output-root /governed/txnopt-results/level1-formal-buildN-attemptN \
  --build-manifest experiments/txnopt/manifests/txnopt_level1_build_attemptN.json \
  --fixed-work 1200 --fixed-time-seconds 3 --max-rounds 10 \
  --evrptw-max-candidates 64 --rcpsp-max-candidates 64 --attempt N
```

`N` must be the source registry's next Build/formal-attempt pair; it is not a
free label. The materializer requires an exact catalog, hashes every source
instance and generated config, refuses overwrite, leaves holdouts closed, and
does not execute or procure anything. Structural preflight and the 64-core
capacity estimate use the same installed command surface:

```bash
txnopt preflight \
  --plan-manifest /governed/txnopt-plans/level1-formal-plan-attemptN/manifest.json \
  --analysis-protocol /governed/txnopt-plans/level1-formal-plan-attemptN/analysis-protocol.json \
  --python /governed/txnopt-builds/buildN/venv/bin/python \
  --wheel /governed/txnopt-builds/buildN/wheel/txnopt-0.1.0a1-cp313-cp313-linux_x86_64.whl

txnopt cloud tencent assess \
  --protocol experiments/txnopt/level1-protocol-v2.json \
  --calibration /governed/txnopt-calibration/level1-local-calibration-attemptN.json \
  --physical-cores 64 --provider-memory-gb 128
```

Server rental remains blocked unless the estimate is at most ten days. The
live host doctor additionally requires the signed Tencent DryRun receipt; SKU
resources are never accepted from self-reported CLI numbers.

The prior formal identities Attempts18, 21, 23, 24, 28, and 30 remain immutable
and unexecuted. Attempt32 is the retained Build20 protocol-v2 plan with 2,880
prebound expected identities, its own preregistered analysis file, and a fresh
absent raw root. Structural preflight is read-only.

Build14/Attempt24, Build18/Attempt28, and Build19/Attempt30 remain historical,
byte-preserved predecessor chains. Build20/Attempt32 is also retained and
unexecuted after failed Review14. Build21/Attempt34 remains a rejected formal
predecessor; its structural preflight passes while its raw root and launch
claim remain absent. Build23/Attempt38 is the current clean formal identity;
Attempt38 has been materialized and independently preflighted, while its raw
root and launch claim remain absent.
All four predecessor
preflights are `PASS_NOT_AUTHORIZED_TO_EXECUTE`;
holdouts and cloud purchase remain closed, and no launch claim or raw root
exists. New execution rejects protocol-v1 plans and any build/attempt pair not
present in the closed successor registry.
Build14-bound local calibration Attempt25 separately retains 96/96 anchored v3
raw bundles and 96/96 independent reviews. Its representative fixed-work
geometric-mean speedups are 1.37x for EVRPTW and 2.18x for RCPSP, with maximum
one-worker overheads of 5.69% and 1.86%; all fixed-work semantic and objective
groups match and fallback remains zero. Independent calibration audit reports
zero findings, but this is not the 2,880-run Level 1 matrix. Pre-cloud Attempt07
therefore records `READY_FOR_SEPARATE_PROCUREMENT_AUTHORIZATION_NOT_AUTHORIZED`:
it permits only a later explicit authorization decision and does not itself
authorize procurement, cloud use, formal execution, holdout access, or Level 1
readiness.
The first v3 successor build is retained as failed Build12 evidence after an
adversarial reviewer exposed a vacuous empty native-round receipt check. The
corrected implementation must use a later build attempt and may not overwrite
the Build12 external directory or failure receipt.
The later formal successor must also use campaign plan v2: expected identities
are materialized and tree-bound before raw execution, then consumed unchanged
by the runner and independent reviewer. Legacy plan v1 remains readable only
for historical inspection and cannot enter formal preflight, execution, or
review.
The campaign `run` boundary also requires the exact plan-bound successor wheel,
an isolated Python installation, host
resource checks, and a separately signed
`txnopt-level1-procurement-authorization-v1` receipt. That receipt must bind the
exact plan, analysis protocol, config and identity trees, producer manifest,
wheel, raw root,
exclusive-Linux contract, and 14-day maximum window. Before the first raw
write, the runner creates an immutable atomic launch claim and records that no
target process is active. Each run executes in its own process group; timeout
cleanup terminates and verifies the whole group rather than leaving a writer
behind. It writes raw bundles and one raw-only execution receipt; it cannot
review or declare readiness. After a complete run, the installed command
`txnopt review --plan-manifest <plan> --analysis-protocol <analysis> \
  --execution-receipt <execution> --python <python> --wheel <wheel> \
  --review-root <review-root> --review-receipt <review-receipt>` independently
replays every raw manifest into a new review root and recomputes the registered
performance, confidence-interval, parity, prefix-safety, fallback, and Cmax
gates. Failed attempts retain their raw root and require a new attempt label.
The representative Build11 fault gate is retained as Attempt02. The current
bounded-exhaustive local gate is
`manifests/txnopt_level1_fault_gate_attempt03.json`; its first orchestration
failure remains separately retained. The historical Build14 decision receipt is
`manifests/txnopt_level1_precloud_gate_attempt07.json`. It is explicitly ready
only for a separate procurement-authorization decision, not an authorization
receipt and not a Level 1 readiness claim. Pre-cloud Attempts01-06 remain
immutable prior results.
