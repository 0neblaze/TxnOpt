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
txnopt replay RAW/manifest.json --output-dir REVIEW \
  --expected-identity PLAN/expected-identities/RUN.json
txnopt env
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
txnopt archive inventory SOURCE
txnopt archive mirror SOURCE --store ARCHIVE --commit-id ATTEMPT
txnopt archive verify --store ARCHIVE --commit-id ATTEMPT \
  --commit-sha256 SHA256 --commit-size BYTES
txnopt archive restore --store ARCHIVE --commit-id ATTEMPT \
  --commit-sha256 SHA256 --commit-size BYTES --destination DESTINATION
txnopt archive mirror SOURCE --s3-bucket BUCKET --s3-prefix PREFIX \
  --s3-region REGION [--s3-endpoint-url https://S3-ENDPOINT] \
  --commit-id ATTEMPT
txnopt legacy verify LEGACY_RECEIPT.json
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
explicit governed output root. The internal `ArchiveStore` port has a WSL/Linux
`LocalFilesystemArchiveStore` and an optional `S3ArchiveStore`. The S3 adapter
uses boto3's standard credential chain, requires bucket versioning plus a
minimum COMPLIANCE Object Lock rule, conditionally creates every object, and
downloads each object to recompute SHA-256 before treating it as verified.
Install the optional client with `txnopt[s3]`; credentials must not be passed on
the command line or written to Git. A provider is not admitted for governed
evidence until its live bucket passes the same contract tests. Native Windows
archive portability remains unimplemented.

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
