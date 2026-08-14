# TxnOpt

TxnOpt is an auditable ordered-transaction runtime for state-dependent
optimization. The active internal alpha is `0.1.0a1`; it is not published to
PyPI, Zenodo, or a remote Git repository.

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
uv run pytest tests/txnopt -q
```

The internal CLI surface is:

```text
txnopt run --config RUN.json
txnopt verify RAW/manifest.json
txnopt replay RAW/manifest.json --output-dir REVIEW
txnopt env
txnopt archive inventory SOURCE
txnopt archive mirror SOURCE --store ARCHIVE --commit-id ATTEMPT
txnopt archive verify --store ARCHIVE --commit-id ATTEMPT \
  --commit-sha256 SHA256 --commit-size BYTES
txnopt archive restore --store ARCHIVE --commit-id ATTEMPT \
  --commit-sha256 SHA256 --commit-size BYTES --destination DESTINATION
txnopt legacy verify LEGACY_RECEIPT.json
```

Runner output contains raw artifacts only. Readiness decisions must be made by
a separate review/gate process after raw replay.

Rebuildable local state is tree-scoped outside the repository:

```text
~/.local/state/txnopt/<source-tree>/
~/.cache/txnopt/<source-tree>/
```

Formal evidence always requires an explicit governed output root. The internal
`ArchiveStore` port currently has one production adapter,
`LocalFilesystemArchiveStore`; no cloud-provider adapter is claimed before a
provider is selected and separately authorized.

## Evidence and formal verification

New protocol definitions and indexes live under `experiments/txnopt/`; new raw
output must use a governed external root. Repository-local `results/` is legacy
or rebuildable state, not the default destination and not formal evidence.

The bounded formal receipt is re-run with:

```bash
python tools/verify_txnopt_formal.py \
  --java /path/to/java \
  --tla2tools /path/to/tla2tools.jar
```

Model-check success is not presented as a complete mathematical proof or an
independent review.

The Build11 evidence-lifecycle review packet can be machine-checked with
`tools/verify_txnopt_build11_review_packet.py`. This verifies the request,
bound source and artifact identities, installed wheel, sealed formal receipt,
import independence, and request-scoped tests; it deliberately emits no
independent review decision.

Level 1 matrix materialization, cloud-window estimation, and the native
resource soak are still transitional tools pending the remote recovery gate.
They write signed receipts, never start a cloud server, and fail closed on
incomplete catalogs, calibrations, or existing output paths. Their next active
implementation belongs in `txnopt_evidence`; the frozen tool bytes referenced
by prior attempts are not rewritten.

## Frozen EVRPTW history

The pre-TxnOpt implementation is preserved at tag
`stage052-legacy-freeze-v1`, commit
`3b0cf371759f3465c7264b85894d090004f3cf43`. Historical Stage labels, schemas,
paths, manifests, hashes, statuses, and ABI exports are not renamed or
recomputed. The active `txnopt` wheel contains no `evrptw` package and no old
Stage CLI wrappers; historical reproduction uses the frozen tag/wheel and the
read-only `txnopt_legacy` reader.

The pre-refactor TxnOpt recovery baseline is additionally preserved by the
local annotated tag `txnopt-pre-refactor-v1` at commit
`5901a339ae0fa0fe490a67d2ce9d995a530d110b` and an externally verified full Git
bundle. Pushing this branch and both recovery tags remains separately
authorized work; tracked legacy deletion is blocked until a fresh remote clone
passes recovery checks.

Schneider and PSPLIB benchmark data are not redistributed. Third-party data,
papers, and solvers remain under their own licenses; see
[THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).

## License and citation

Original code and documentation are licensed under Apache-2.0. The internal
alpha is not an archival release and has no DOI. See [CITATION.cff](CITATION.cff)
for the current software citation metadata.
