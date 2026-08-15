# Contributing

Contributions are welcome when they preserve the repository's scientific and
evidence contracts.

## Before changing code

1. Read `AGENTS.md` and the active TxnOpt protocol. Use a frozen Stage protocol
   only when performing an explicitly read-only historical replay.
2. Do not edit frozen baselines, accepted raw evidence, or historical hashes.
3. Use a new canonical run label for every failed, repeated, or superseding run.
4. Keep formal output under an explicit governed external root. Local scratch
   output belongs under the tree-scoped XDG state root, never repository-local
   `results/`.
5. Add tests at a public behavior boundary before changing formal behavior.

## Local checks

```bash
uv sync --all-groups
uv run pytest -m "not external_data"
uv run ruff check .
uv run mypy
git diff --check
```

The active TxnOpt test suite is self-contained and must run with zero skipped
tests in the locked build environment. Historical compatibility checks use
their frozen fixtures and may not turn external Stage data into an active
runtime dependency.

## Pull requests

Explain the research or maintenance question, the formal behavior affected,
the evidence or tests used, and whether any artifact schema or provenance
identity changes. Never claim readiness from producer output alone; cite the
independent review result.
