# FURP Project Repository

> **Faculty Undergraduate Research Practice (FURP)**
> Undergraduate Research Group · Faculty of Science and Engineering · University of Nottingham Ningbo China

This is your project home for the FURP programme. **Fork this template**, rename your repo, fill in the content each week, and share it with us (or make it public) so we can follow your progress and review your weekly work.

---

## Getting started (do this in Week 1)

1. **Fork / use this template** to create your own repository.
2. **Rename your repo** following the naming convention:
   ```
   FURP-2025/YourName-ProjectTag
   # e.g. furp-2025/Jason-ROSBootcamp
   ```
3. **Give us access:** either make the repo **public**, or **share it** with the research group accounts (ask your project lead for the usernames to add as collaborators).
4. **Fill in this README** — replace the placeholders in the *Project Info* section below.
5. **Start your weekly log** by copying the template in
   [`docs/00_weekly.md`](docs/00_weekly.md) to a numbered weekly file, beginning
   with [`docs/01_weekly.md`](docs/01_weekly.md).

---

## Project Info

| Field | Your entry |
|---|---|
| Student name(s) | Yiyang Guo |
| Project title | Replication and Extension of the Electric Vehicle-Routing Problem with Time Windows and Recharging Stations |
| Project tag | EVRP-TW |
| Track | Research |
| Supervising faculty | To be confirmed |
| Project lead | To be confirmed |
| Team or individual | Individual |
| Cited paper being replicated | Michael Schneider, Andreas Stenger, and Dominik Goeke (2014), [The Electric Vehicle-Routing Problem with Time Windows and Recharging Stations](https://doi.org/10.1287/trsc.2013.0490), *Transportation Science*, 48(4), 500-520. DOI: `10.1287/trsc.2013.0490` |

**One-line summary:** This project reproduces the core modelling and computational workflow for Schneider, Stenger, and Goeke's E-VRPTW study, then evaluates practical extensions around route feasibility, charging-station insertion, and reproducible open-source baselines.

### Replication paper resources

- [Publisher page and DOI](https://doi.org/10.1287/trsc.2013.0490)
- [Public E-VRPTW benchmark instances](https://doi.org/10.17632/h3mrm5dhxw.1)
- [Selection rationale and candidate comparison](docs/reference_selection.md)
- [BibTeX entry](docs/references.bib)
- [Local environment and solver setup](docs/environment.md)

---

### Week 5 low-battery experiments

Run the Week 5 low-battery consolidation experiment with:

```bash
uv run python -m evrptw.experiments.week05_consolidation \
  --output-dir results/week05 \
  --summary-dir experiments/summaries
```

It reruns the Week 4 `60`-unit battery cases, checks their deterministic
metrics against the tracked Week 4 per-run table, and compares baseline routes,
late charging repair, anticipatory charging repair, and anticipatory repair
plus route splitting. It is retained as the original low-battery structural
infeasibility control.

Run the infrastructure-augmentation experiment with:

```bash
uv run python -m evrptw.experiments.week05_infrastructure_augmentation \
  --output-dir results/week05_infrastructure \
  --summary-dir experiments/summaries
```

It preserves the customers, seeds, battery capacity, and vehicle parameters,
then adds deterministic midpoint charging stations only for customers that
cannot complete a safe-node-to-customer-to-safe-node energy cycle. Raw artifacts
are Git-ignored; the reviewable summaries and documentation are:

- [Week 5 checkpoint](docs/week05_project_checkpoint.md)
- [Week 5 technical report](docs/week05_consolidation_and_route_splitting.md)
- [Week 5 progress log](docs/05_weekly.md)
- [Week 5 summary results](experiments/summaries/week05_summary_results.csv)
- [Week 5 infrastructure summary](experiments/summaries/week05_infrastructure_summary_results.csv)

### Week 5 advanced method benchmark

The current primary method is an ALNS-based matheuristic with an exact
full-recharge charging subproblem. A small-scale Branch-Price-and-Cut solver
with bidirectional labeling provides proven-optimal references for instances
with at most eight customers. OR-Tools remains a transparent classical VRPTW
baseline, and the GA is retained only as a weak baseline.

The Schneider benchmark files are local, Git-ignored research data under
`data/schneider/`. Run the reproducible Primary and battery Stress benchmarks
with:

```bash
uv sync --all-groups
uv run python -m evrptw.experiments.week05_advanced_benchmark \
  --benchmark-dir data/schneider \
  --output-dir results/week05_advanced \
  --instances c101C5,r105C5,rc105C5,c104C10,r103C10,rc102C10,c106C15,r105C15,rc103C15,c101_21,r101_21,rc101_21 \
  --stress-instances c101C5,r105C5,rc105C5 \
  --seeds 2014,2015,2016 \
  --alns-iterations 1000 \
  --time-limit-seconds 30 \
  --ga-population-size 60 \
  --ga-generations 80
```

The full methodology, limitations, Week 5 second-edition report, and reviewable
results are:

- [ALNS, exact charging, and BPC methodology](docs/week05_alns_bpc_methodology.md)
- [Week 5 second-edition report](docs/05_weekly_v2_alns_bpc_benchmark_rebuild.md)
- [Schneider 92-instance audit](experiments/summaries/schneider_instance_catalog.csv)
- [Advanced per-run results](experiments/summaries/week05_advanced_per_run_results.csv)
- [Advanced summary results](experiments/summaries/week05_advanced_summary_results.csv)
- [Advanced failure records](experiments/summaries/week05_advanced_failure_cases.csv)

### Stage 0 frozen ALNS baseline

Stage 0 freezes the current `ALNS_EXACT_CHARGING` method on 12 representative
Schneider instances, seeds `2014/2015/2016`, a 30-second limit, and one thread.
The workflow records validated solution metrics, exact-charging activity,
structured constraint violations, source and instance hashes, environment
metadata, immutable checksums, and an automatic regression comparison report.

```bash
uv run python -m evrptw.experiments.stage00_baseline run \
  --config configs/stage00_baseline.toml \
  --output-dir results/stage00 \
  --baseline-dir experiments/baselines/stage00
```

Full raw logs remain under the ignored `results/` directory. The curated,
checksum-protected baseline is tracked under `experiments/baselines/stage00/`.
See [the Stage 0 baseline protocol](docs/stage00_baseline.md) for verification
and candidate-comparison commands.

### Stage 1 lexicographic objective

The formal objective is now vehicle-first:
`(vehicle count, total distance, total charging time, charging count)`. ALNS
hard-rejects moves that add a vehicle, and the small exact BPC reference uses
the same objective for its incumbent and optimality claim. Stage 0 remains an
immutable historical baseline.

See [the Stage 1 objective protocol](docs/stage01_lexicographic_objective.md)
for the comparison interface, acceptance policy, experiment schema, and
reproduction command.

- [Stage 1 per-run results](experiments/summaries/stage01_per_run_results.csv)
- [Stage 1 summary](experiments/summaries/stage01_summary_results.csv)
- [Old-vs-new objective ranking](experiments/summaries/stage01_objective_ranking_changes.csv)
- [Stage 0 comparison](experiments/summaries/stage01_stage00_comparison.csv)

### Stage 2.1 route-reduction operators

Stage 2.1 adds `route elimination destroy`, `vehicle-count-aware repair`, and
`route merge` behind the `stage02_route_reduction` operator profile. The formal
experiment keeps Stage 0's 12 instances, three seeds, 30-second limit, 1000
maximum iterations, and one thread. Every run is validated independently and
records raw JSON, solution routes, operator events, failure reasons, environment
metadata, and comparison gates.

See [the Stage 2.1 route-reduction protocol](docs/stage02_route_reduction.md)
for the implementation invariants, fixed experiment scope, failure loop, and
acceptance gates.

```bash
uv run pytest
uv run ruff check .
uv run mypy
uv run python -m evrptw.experiments.stage02_route_reduction \
  --config configs/stage02_route_reduction.toml

# 第二次独立完整复跑
uv run python -m evrptw.experiments.stage02_route_reduction \
  --config configs/stage02_route_reduction.toml \
  --output-dir results/stage02-rerun01 \
  --run-label stage02_rerun01 \
  --repeat-of results/stage02
```

Raw Stage 2.1 evidence remains under ignored `results/`; each retry uses a new
run label and output directory. The second run writes an `independent_complete_rerun`
gate after comparing the first run's complete gate report. Curated summaries are
written under `experiments/summaries/` only after the corresponding run completes.

The completed formal evidence is recorded in the [successful gate report](experiments/summaries/stage02_attempt02_gate_report.csv),
the [independent rerun gate report](experiments/summaries/stage02_rerun01_gate_report.csv),
and the [failure-round record](experiments/summaries/stage02_gate_report.csv).

### Stage 2.2 cross-route quality

Stage 2.2 adds `relocate`, `swap`, `two_opt_star`, `route_segment_destroy`, and
bounded `ejection_chain` proposals behind the `stage02_route_quality` profile.
The profile keeps the fixed Stage 0 scope and compares against the accepted
Stage 2.1 per-run results. Its protocol, failure rounds, and independent rerun
are documented in [the Stage 2.2 protocol](docs/stage02_route_quality.md).

```bash
uv run python -m evrptw.experiments.stage02_route_quality \
  --config configs/stage02_route_quality.toml
```

The accepted Stage 2.2 comparison baseline for the next stage is
`experiments/summaries/stage02_quality_attempt02_per_run_results.csv`.

### Stage 2.3 constraint-guided search and Stage 3 readiness review

Stage 2.3 is the current `solve_alns()` default profile. It adds deterministic
constraint-guided removals (`station_pressure`, `time_window_conflict`,
`worst_energy_detour`, and `shaw_related`) and dynamic removal tiers. The
formal protocol keeps the 12-instance, three-seed, 30-second, 1000-iteration,
single-thread scope and does not implement Stage 3 acceleration.

```bash
uv run pytest
uv run ruff check .
uv run mypy
uv run python -m evrptw.experiments.stage02_constraint_guided \
  --config configs/stage02_constraint_guided.toml \
  --output-dir results/stage02-constraint-guided_attempt16 \
  --run-label stage02_constraint_guided_attempt16

uv run python -m evrptw.experiments.stage02_constraint_guided_review \
  --run-dir results/stage02-constraint-guided_attempt16 \
  --comparison-dir results/stage02-quality_attempt02 \
  --review-label stage02_constraint_guided_attempt16

uv run python -m evrptw.experiments.stage02_constraint_guided \
  --config configs/stage02_constraint_guided.toml \
  --output-dir results/stage02-constraint-guided-rerun09 \
  --run-label stage02_constraint_guided_rerun09 \
  --repeat-of results/stage02-constraint-guided_attempt16

uv run python -m evrptw.experiments.stage02_constraint_guided_review \
  --run-dir results/stage02-constraint-guided-rerun09 \
  --comparison-dir results/stage02-quality_attempt02 \
  --review-label stage02_constraint_guided_rerun09
```

The completed Stage 2.3 evidence is recorded in the [attempt16 gate report](experiments/summaries/stage02_constraint_guided_attempt16_gate_report.csv),
the [independent rerun gate report](experiments/summaries/stage02_constraint_guided_rerun09_gate_report.csv),
the [attempt16 readiness review](experiments/summaries/stage02_constraint_guided_attempt16_stage03_readiness.csv),
and the [READY_FOR_STAGE03 rerun review](experiments/summaries/stage02_constraint_guided_rerun09_review_report.md).
The readiness-only Stage 2.2 instrumentation supplement is tracked separately at
`experiments/summaries/stage02_quality_attempt02_readiness_metrics.csv`; the formal
comparison remains the fixed Stage 2.2 per-run table above.
The first review remains explicitly `PENDING_INDEPENDENT_RERUN`; the independent rerun
review is `READY_FOR_STAGE03`. Attempts 08, 09, and 10 retain their objective or
operator-coverage failures, and all earlier failed or superseded rounds remain preserved
under distinct attempt and rerun labels.

---

## Repository structure

This structure is **mandatory** — please keep it intact.

```
/docs
 ├── 00_weekly.md         ← weekly-log template; keep unchanged
 ├── 01_weekly.md         ← Week 1 progress, challenges, and next steps
 └── meeting_notes/       ← key takeaways from all team meetings
/src                      ← your code / experiments / materials
FURP_Showcase.pdf         ← your poster / presentation PDF, in the repo root
```

- **`docs/00_weekly.md`** — the reusable weekly-log template.
- **`docs/NN_weekly.md`** — one numbered progress log per week, starting with
  `01_weekly.md`. The latest weekly file is the first thing reviewers check.
- **`docs/meeting_notes/`** — one file per meeting with key takeaways and action items.
- **`src/`** — all your code, scripts, notebooks, and experiment materials.
- **`FURP_Showcase.pdf`** — your final poster, placed in the **repo root** with this exact filename.

---

## The three rules for your certificate

To earn your FURP certificate, **all three** must be satisfied:

1. **Attend > 50%** of programme activities (weekly meetings, workshops, scheduled sessions — online or in person).
2. **Submit a poster** — place it as `FURP_Showcase.pdf` in this repo root.
3. **Present at the Poster Showcase** — in person (strongly preferred), or send a stand-in if you truly cannot attend.

> Miss any one of the three, and the certificate is not awarded this round.

**Research Track — minimum for certification:** successful replication of a cited paper with at least **10% innovation** (reproduce the work *and* add something new).

---

## Weekly cadence

Every week, you should:

- ✅ Create or update the current numbered weekly file in `docs/`
- ✅ Log meeting notes in [`docs/meeting_notes/`](docs/meeting_notes/)
- ✅ Attend the weekly meeting (online or in person)

Consistent weekly engagement is the backbone of a successful FURP project — and it feeds directly into your attendance (Rule 1).

---

## Leave & withdrawal

Any **leave of absence** or **withdrawal** must be notified to us **by email** — a verbal or chat message is not sufficient.

- **Leave:** email *before* the session where possible, state the date(s) and reason. Note that leave still counts against the >50% attendance rule.
- **Withdrawal:** email us to formally withdraw so we can free your project slot and update records.
- **Switching tracks:** email the project lead with the subject *"Project Transfer Request"* and CC your supervising faculty member.

> No email = no record. Always put leave and withdrawal in writing.

---

## Quick checklist

- [x] Forked the template and renamed the repo (`FURP-2026-Yiyang-GUO-EVRP-TW`)
- [ ] Made the repo public **or** shared it with the research group
- [x] Filled in the *Project Info* table above
- [x] Created `docs/01_weekly.md` from the weekly template
- [ ] Created my first file in `docs/meeting_notes/`
- [ ] (By Showcase) Added `FURP_Showcase.pdf` to the repo root

---

*Bridging the gap between classroom knowledge and cutting-edge research.*
