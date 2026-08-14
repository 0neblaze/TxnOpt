# TxnOpt

TxnOpt is an auditable ordered-transaction runtime for state-dependent
optimization. The active internal alpha is `0.1.0a1`; it has not been released
to PyPI, Zenodo, or as a public software release. Recovery branches and tags on
the existing Git remote are not a release claim.

The Level 1 implementation provides:

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
Level 2 and Level 3 remain mandatory gated stages; Level 1 alone is not a
publication result.

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

## Install and verify

Python 3.13 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --frozen --all-groups
uv run ruff check src/txnopt src/txnopt_cases src/txnopt_evidence src/txnopt_legacy
uv run mypy
uv run pytest -q
```

The internal CLI surface is:

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
to Git. Native Windows source inventory and atomic restore remain unimplemented.

The Tencent capacity profile means **64 physical cores**, not 64 vCPU. Offline
assessment cannot prove a cloud instance satisfies that requirement. Before a
formal run, both the Tencent instance metadata/API and the Linux topology probe
must show 64 physical cores and one thread per core. The Tencent product/API
memory specification must be at least 128 GB. Linux `MemTotal` is recorded but
may be slightly below the product value because of platform reservations; that
alone is not a failure. The calibration receipt bound to the selected producer
must report peak RSS below 80% of actual visible memory. The closed historical
pairings are Build16/Attempt26/Attempt27 and Build18/Attempt28/Attempt29; the
current successor is Build19/Attempt30/Attempt31. Intermediate validation
candidates are not approved formal producers. Capacity assessment, account configuration,
procurement authorization, build portability, and formal execution remain
separate gates.

## Evidence and formal verification

New protocol definitions and indexes live under `experiments/txnopt/`; new raw
output must use a governed external root. Repository-local `results/` is legacy
or rebuildable state, not the default destination and not formal evidence.

The bounded formal receipt is re-run with:

```bash
python -m txnopt_evidence.formal_verify \
  --java /path/to/java \
  --tla2tools /path/to/tla2tools.jar
```

Model-check success is not presented as a complete mathematical proof or an
independent review.

Level 1 plan, preflight, raw execution, independent review, calibration,
cloud-window estimation, and native resource-soak implementations live in
`txnopt_evidence`. Plan materialization, fail-closed preflight, and independent
review are exposed through the unified CLI. Formal execution still requires a
separate signed authorization and is not implied by any of those commands.
Historical Build11 gate executors remain recoverable from the frozen Git tags
and are not shipped in the active wheel.

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

Original code and documentation are licensed under Apache-2.0. The internal
alpha is not an archival release and has no DOI. See [CITATION.cff](CITATION.cff)
for the current software citation metadata.
