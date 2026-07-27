# External artifact index

Large experiment evidence is not stored in Git. `index.json` is the lightweight
public catalog for accepted and incomplete Stage 5.2 campaigns.

The local, ignored `configs/stage052_storage_roots.local.toml` resolves
`wsl_staging` and `d_archive` aliases. Public records use aliases and logical
paths only; machine-specific absolute paths are not committed.

An entry marked `accepted_pilot` passed independent review. An entry marked
`partial_unreviewed` is preserved for audit but must not be used as a formal
result. `release_url` remains `null` until an immutable external bundle is
published and verified.
