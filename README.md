# TxnOpt

> [!IMPORTANT]
> **Archived historical snapshot — 2026-08-16.** This repository is retained
> read-only for provenance, research handoff, and audit. It is no longer under
> active development, and its installation and execution instructions are not
> maintained compatibility guarantees. The internal alpha `0.1.0a1` was never
> released to PyPI, Zenodo, or as a public software release.

TxnOpt was developed as an auditable ordered-transaction runtime for
state-dependent optimization. Recovery branches and tags on the existing Git
remote preserve history; they are not release claims.

The implemented Level 1 candidate provides:

- one state owner, `TxnRuntime`, with serial, deterministic barrier, and ordered
  transaction execution;
- pure `SearchKernel` proposal/decision logic and domain-owned `Oracle`
  validation/objective logic;
- fixed-work and deadline budgets, atomic cache publication, rollback, and no
  hidden fallback;
- one `txnopt-native-round-v1` Python/native call per EVRPTW round, with one
  packed instance context per solve;
- independent EVRPTW and RCPSP adapters;
- separate semantic and physical trace artifacts, raw-only production, and
  independent-process replay;
- bounded TLA+/PlusCal models and T1--T4 proof obligations.

The public roadmap is
[`docs/roadmap/txnopt-level1-to-level3-roadmap.md`](docs/roadmap/txnopt-level1-to-level3-roadmap.md).
It is retained as the historical plan. TxnOpt did not reach the Level 1 Ready
gate, and Level 2 and Level 3 were not completed.

## Final archived status

[Independent Review18](formal/reviews/txnopt_tencent_precloud_review_attempt18.json)
returned `PASS` with zero critical, major, or minor findings for reviewed
revision `2125477ed75f2a55e60bf368d599ca6fd984018a`. Its terminal status is:

```text
READY_FOR_TENCENT_ACCOUNT_INPUT_NOT_AUTHORIZED
```

That result means the account-independent pre-cloud work was complete and ready
to receive separately authorized Tencent account inputs. It does **not** mean
that Level 1 completed or that live cloud execution occurred. The following
were never completed or authorized:

- Tencent account or credential input and live SKU selection;
- an online `RunInstances DryRun=true` request, cloud purchase, or instance
  creation;
- live COS mirroring, exact-version restore, or retention validation;
- the 2,880-run formal matrix or access to the final holdout;
- a public alpha, PyPI/Zenodo release, DOI archive, or paper submission;
- deletion or rewriting of the governed E-drive evidence archive.

Review18 remains an immutable receipt for its exact reviewed revision and tree.
The later research-handoff, WSL-invocation, and archival README commits are
documentation-only successors; they are not a new independent review or a new
evidence-ready source identity.

## What this snapshot preserves

- the generic runtime, EVRPTW and RCPSP adapters, native kernels, and formal
  models under `src/`, `cpp/`, and `formal/`;
- protocol, manifest, review, and lifecycle material under
  `experiments/txnopt/` and `formal/reviews/`;
- the frozen pre-TxnOpt boundary at `stage052-legacy-freeze-v1` and the
  pre-refactor recovery point at `txnopt-pre-refactor-v1`;
- the
  [COR research handoff](docs/research/txnopt-cor-research-handoff-2026-08-16.md),
  whose research judgments remain explicitly marked `UNVERIFIED`.

Large raw experiment evidence is retained in governed external storage and is
not part of this GitHub repository. Repository-local `results/` is ignored and
was not uploaded during archival publication. This snapshot therefore preserves
the tracked source and audit trail, not a complete raw experiment bundle.

## Public API

The root package exports exactly five interfaces:

```python
from txnopt import Oracle, RunConfig, RunResult, SearchKernel, TxnRuntime
```

Candidate transactions, budget/cache ledgers, random tapes, native receipts,
and trace records are versioned internal contracts.

The formal EVRPTW objective remains lexicographic:

```text
(vehicle_count, total_distance, total_charging_time, charging_count)
```

Its only active implementation is `txnopt_cases.evrptw.objective`.

## Archived installation and verification reference

At the time of archival, the project used Python 3.13 and
[uv](https://docs.astral.sh/uv/). The commands below record the last maintained
workflow; they are not a promise of compatibility with future toolchains.

```bash
uv sync --frozen --all-groups
uv run ruff check src/txnopt src/txnopt_cases src/txnopt_evidence src/txnopt_legacy
uv run mypy
uv run pytest -q
```

The archived internal CLI surface is:

```text
txnopt run --config RUN.json
txnopt verify RAW/manifest.json \
  --expected-identity PLAN/expected-identities/RUN.json
txnopt plan --protocol PROTOCOL.json --catalog CASE-CATALOG.json \
  --destination PLAN_DIR --raw-output-root RAW_ROOT \
  --build-manifest BUILD.json --fixed-work WORK \
  --fixed-time-seconds SECONDS --max-rounds ROUNDS \
  --evrptw-max-candidates N --rcpsp-max-candidates N [--attempt ATTEMPT]
txnopt preflight --plan-manifest PLAN/manifest.json \
  --analysis-protocol PLAN/analysis-protocol.json [--python PYTHON --wheel WHEEL]
txnopt review --plan-manifest PLAN/manifest.json \
  --analysis-protocol PLAN/analysis-protocol.json \
  --execution-receipt EXECUTION.json --python PYTHON --wheel WHEEL \
  --review-root REVIEW_ROOT --review-receipt REVIEW.json \
  [--review-workers N --per-run-timeout-seconds SECONDS]
txnopt cloud tencent assess --protocol PROTOCOL.json \
  --calibration CALIBRATION.json --physical-cores 64 --provider-memory-gb 128
txnopt cloud tencent spec --output SPEC.json
txnopt cloud tencent doctor --provider-receipt DRY-RUN-RECEIPT.json \
  --expected-peak-rss-bytes BYTES --output HOST-RECEIPT.json
txnopt cloud tencent dry-run --region REGION --zone ZONE \
  --instance-type SKU --image-id IMAGE --vpc-id VPC --subnet-id SUBNET \
  --security-group-id SECURITY-GROUP \
  --request-output REQUEST.json --receipt-output RECEIPT.json
txnopt cloud tencent bundle --destination DEPLOYMENT \
  --build-manifest BUILD.json --wheel TXNOPT.whl \
  --source-manifest SOURCE.json --native-attestation NATIVE.json \
  --uv-lock uv.lock --toolchain-lock TOOLCHAIN.json \
  --plan-manifest FORMAL-PLAN.json
txnopt cloud tencent bundle-verify --manifest DEPLOYMENT/bundle-receipt.json
txnopt archive inventory SOURCE
txnopt cloud tencent cos mirror SOURCE --cos-bucket BUCKET-APPID \
  --cos-region REGION --cos-prefix PREFIX --commit-id ATTEMPT
txnopt cloud tencent cos verify --cos-bucket BUCKET-APPID --cos-region REGION \
  --cos-prefix PREFIX --commit-id ATTEMPT --commit-key KEY \
  --commit-version-id VERSION --commit-sha256 SHA256 --commit-size BYTES \
  --commit-retain-until TIMESTAMP
txnopt cloud tencent cos restore --cos-bucket BUCKET-APPID --cos-region REGION \
  --cos-prefix PREFIX --commit-id ATTEMPT --commit-key KEY \
  --commit-version-id VERSION --commit-sha256 SHA256 --commit-size BYTES \
  --commit-retain-until TIMESTAMP --destination DESTINATION
```

Runner output contains raw artifacts only. Readiness decisions must be made by
a separate review/gate process after raw replay.

Rebuildable local state is tree-scoped outside the repository:

```text
~/.local/state/txnopt/<source-tree>/
~/.cache/txnopt/<source-tree>/
```

When a local run config omits `output_root`, raw output defaults to the `runs/`
directory below that tree-scoped state root. Formal evidence always requires an
explicit governed output root. There is no generic `ArchiveStore`, local archive
adapter, or S3 compatibility layer. `txnopt_evidence.tencent_cos` is a direct
Tencent COS integration and `txnopt_evidence.archive` only inventories local
source bytes. COS objects and commit markers bind the exact returned `VersionId`;
same-name keys are never treated as immutable identities by themselves. The
bucket must have versioning plus default COMPLIANCE Object Lock for at least 365
days, and every exact version is downloaded to recompute its SHA-256 before it is
accepted. Install the optional client with `txnopt[tencent]`. Credentials are read
only from `TENCENTCLOUD_SECRET_ID`, `TENCENTCLOUD_SECRET_KEY`, optional
`TENCENTCLOUD_SESSION_TOKEN`, or a CVM CAM role selected by
`TENCENTCLOUD_USE_CVM_ROLE=1`; they are not CLI arguments and must not be written
to Git. Third-party COS/CVM SDK exception bodies are discarded at the adapter
boundary; only the local operation name and exception type may enter CLI output.
Native Windows source inventory and atomic restore remain unimplemented.

The archived Tencent capacity contract requires **64 physical cores**, not 64
vCPU, plus a provider memory specification of at least 128 GB. A qualifying
formal host would also have needed `CoreCount=64`, `ThreadPerCore=1`, a matching
Linux topology probe, and calibration peak RSS below 80% of Linux-visible
memory. Offline assessment alone cannot establish those conditions.

The final retained pre-cloud chain is Build24, formal plan Attempt40,
representative calibration Attempt41, Deployment06, Pre-cloud13, and
Independent Review18. Attempt40 contains 2,880 planned configurations but has
no formal raw root and was never launched. Earlier rejected and failed builds,
plans, calibrations, deployments, pre-cloud gates, and Reviews13--17 remain
preserved as append-only evidence; Review18 did not rewrite them. Intermediate
validation candidates are not approved formal producers.

## Archived evidence and formal verification

Protocol definitions and indexes are preserved under `experiments/txnopt/`.
Formal raw output was required to use a governed external root.
Repository-local `results/` is legacy or rebuildable state, not formal evidence.

The bounded formal receipt is re-run with:

```bash
python -m txnopt_evidence.formal_verify \
  --java /path/to/java \
  --tla2tools /path/to/tla2tools.jar
```

Model-check success is not presented as a complete mathematical proof or an
independent review.

Level 1 planning, preflight, raw execution, independent review, calibration,
cloud-window estimation, and native resource-soak implementations are preserved
in `txnopt_evidence`. Plan materialization, fail-closed preflight, and
independent review remain exposed through the unified CLI. The formal matrix was
never authorized or started. Historical Build11 gate executors remain
recoverable from the frozen Git tags and are not shipped in the archived wheel.

## Frozen EVRPTW history

The pre-TxnOpt implementation is preserved at tag
`stage052-legacy-freeze-v1`, commit
`3b0cf371759f3465c7264b85894d090004f3cf43`. Historical Stage labels, schemas,
paths, manifests, hashes, statuses, and ABI exports are not renamed or
recomputed. The active `txnopt` wheel contains no `evrptw` package and no old
Stage CLI wrappers; historical reproduction uses the frozen tag/wheel and the
read-only `txnopt_legacy` reader.

The pre-refactor TxnOpt recovery baseline is additionally preserved by the
annotated tag `txnopt-pre-refactor-v1` at commit
`5901a339ae0fa0fe490a67d2ce9d995a530d110b` and an externally verified full Git
bundle. The active branch and both recovery tags are present on the existing
remote, and a fresh clone has passed tag resolution, Git object verification,
protected-hash checks, wheel installation, and the active test suite. This
recovery publication does not rename the GitHub repository or authorize a
public release.

Schneider and PSPLIB benchmark data are not redistributed. Third-party data,
papers, and solvers remain under their own licenses; see
[THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).

## License and citation

Original code and documentation are licensed under Apache-2.0. This read-only
GitHub snapshot is not a software release or DOI archive; the internal alpha
`0.1.0a1` was never published. See [CITATION.cff](CITATION.cff) for the preserved
software citation metadata.
