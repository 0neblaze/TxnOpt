# Reproducible-EVRPTW

> **TxnOpt transition:** the frozen EVRPTW implementation remains available at
> tag `stage052-legacy-freeze-v1`. Active in-place restructuring now follows
> [`docs/roadmap/txnopt-level1-to-level3-roadmap.md`](docs/roadmap/txnopt-level1-to-level3-roadmap.md).
> Level 1 is in progress; no TxnOpt performance, formal-completeness, package
> release, or publication claim is currently made.

Reproducible research code and evidence for the Electric Vehicle Routing
Problem with Time Windows and Recharging Stations (EVRP-TW).

The repository develops an ALNS-based matheuristic with exact full-recharge
route evaluation, deterministic candidate control, auditable deadline
semantics, bounded artifact streaming, and independent evidence replay.

## Research status

| Stage | Capability | Current evidence |
|---|---|---|
| 0 | Frozen ALNS and exact-charging baseline | Accepted and immutable |
| 1 | Lexicographic objective | Accepted |
| 2.1--2.3 | Route reduction, cross-route quality, constraint guidance | Accepted |
| 3.0--3.4 | Measurement, screening, cache/incremental evaluation, deadline and parallel control | `READY_FOR_STAGE04` |
| 4 | Adaptive weights and search control | `READY_FOR_STAGE05` |
| 5.1 | Best-known-value collection and model-compatibility audit | `READY_FOR_STAGE05_2` |
| 5.2 Pilot | Performance and artifact-streaming protocol | Attempt 72: `READY_FOR_STAGE052_FORMAL_BENCHMARK` |
| 5.2 Formal | 92-instance, ten-seed benchmark | Attempt 73 is incomplete and unreviewed |
| 6--8 | BPC expansion, solution schema, partial/nonlinear charging | Roadmap only |

Repository engineering and extensive audit evidence do **not** by themselves
establish a Q1/Q2 paper contribution. In particular, no Stage 5.2 formal
performance claim is made until a complete run passes independent review.

## Formal objective

All solution comparisons use the lexicographic tuple

```text
(vehicle_count, total_distance, total_charging_time, charging_count)
```

Vehicle count has absolute priority. A candidate that increases vehicle count
is rejected, including during simulated annealing. Objective construction and
comparison are centralized in `evrptw.objective`.

## Installation

Python 3.13 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --all-groups
uv run pytest -m "not external_data"
uv run ruff check .
uv run mypy
```

Commercial solver packages are optional and are not required by the default
installation or continuous integration:

```bash
uv sync --all-groups --extra commercial-solvers
```

CPLEX and Gurobi remain subject to their own licenses.

## Benchmark data

Schneider benchmark instances are not redistributed in this repository. Place
the verified files under `data/schneider/`; formal runs validate the expected
instance identities and hashes before execution. See
[Third-party data and software](THIRD_PARTY_DATA.md).

Tests marked `external_data` require the benchmark data or preserved historical
comparison inputs. The unmarked test suite is self-contained and is the suite
run by public CI.

## Reproducing the staged evidence

The high-level sequence is:

1. install the locked environment;
2. provide the external benchmark data;
3. run or review the stage-specific CLI with its tracked TOML configuration;
4. verify the raw manifest and checksum before trusting any summary;
5. compare only against the baseline named by that stage's protocol.

Start with:

- [Research roadmap](docs/roadmap/README.md)
- [Literature and citation index](docs/literature.md)
- [Experiment evidence index](experiments/INDEX.md)
- [Artifact index](artifacts/README.md)
- [Migration and provenance](docs/provenance/README.md)
- [Stage 5.2 workflow](docs/stage052_performance_benchmark_workflow.md)
- [ALNS and exact-charging methodology](docs/methodology/alns-exact-charging.md)

Generated raw artifacts belong below ignored `results/<run_label>/`. Curated
summaries, registries, manifests, and review products are tracked only after
the corresponding independent replay gate succeeds.

## Artifact availability

Large raw evidence is intentionally not stored in Git. The public artifact
index records logical identity, status, byte count, checksum, source revision,
and release state. Accepted release bundles will be deposited in an external
archive such as Zenodo or OSF and linked by DOI and SHA-256.

Attempt 72 is an accepted Stage 5.2 Pilot. Attempt 73 remains
`partial_unreviewed`; its source campaign manifest still records `planned`.
This distinction is deliberate and must not be rewritten as success.

## Development

Contributions must preserve the formal objective, immutable evidence rules,
fail-fast behavior, and raw-to-summary replay semantics described in
[AGENTS.md](AGENTS.md). See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a
change.

## Citation

Use [CITATION.cff](CITATION.cff). A versioned archival DOI will be added when a
paper-facing release bundle is accepted.

## License

Code and original documentation are licensed under Apache-2.0. Benchmark data,
published papers, commercial solver packages, and third-party repositories are
excluded; see [NOTICE](NOTICE) and [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).
