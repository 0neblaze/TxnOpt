# Contributing

Contributions are welcome when they preserve the repository's scientific and
evidence contracts.

## Before changing code

1. Read `AGENTS.md` and the relevant stage protocol.
2. Do not edit frozen baselines, accepted raw evidence, or historical hashes.
3. Use a new canonical run label for every failed, repeated, or superseding run.
4. Keep large generated output under ignored `results/`.
5. Add tests at a public behavior boundary before changing formal behavior.

## Local checks

```bash
uv sync --all-groups
uv run pytest -m "not external_data"
uv run ruff check .
uv run mypy
git diff --check
```

Run the complete test suite only after supplying the external Schneider data
and historical comparison input documented in the relevant test or stage.

## Pull requests

Explain the research or maintenance question, the formal behavior affected,
the evidence or tests used, and whether any artifact schema or provenance
identity changes. Never claim readiness from producer output alone; cite the
independent review result.
