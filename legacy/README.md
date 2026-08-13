# Frozen legacy boundary

The immutable implementation boundary is tag `stage052-legacy-freeze-v1` at
commit `3b0cf371759f3465c7264b85894d090004f3cf43`.

Historical manifests, schemas, run labels, paths, and ABI exports remain in
their original tracked or governed external locations. This directory records
new TxnOpt-facing freeze metadata only; it never copies, rewrites, or promotes
legacy evidence.

The complete former root policy is preserved byte-for-byte at
`governance/AGENTS.stage052.md` with SHA-256
`4ceb354c6448282f621a08091a2f2e97882b54c515c1194043c0f505e04b4b7d`.

The freeze is intentionally not named `correctness`: the historical source
disposition receipt describes an older snapshot and remains immutable.

The Stage-era tests remain at their historical `tests/test_*.py` paths because
some frozen manifests bind those names. They are not part of active TxnOpt
test discovery and require the frozen tag and legacy wheel. The repository-root
`pytest` command runs `tests/txnopt`; this boundary removes no historical test
bytes and does not restore an active `evrptw.*` compatibility package.
