# Stage 5.2 五种执行架构对比：Attempt 03

## 结论边界

本轮完成同机五模式比较，但没有任何新架构取得 production qualification
（生产资格）。Paired review 为 `NOT_READY`；Pilot review 为
`COMPARISON_COMPLETE_NOT_QUALIFIED`。没有启动 Formal Rerun16，没有切换默认
架构，没有运行 CUDA，没有 push，也没有删除、覆盖或复用失败 evidence。

当前 `current_stage052` 是本轮速度和质量比较的主要分母。accepted Pilot
`stage05.2_benchmark_attempt72` 只用于长期漂移核验。

## 身份与范围

| Item | Value |
|---|---|
| Producer commit | `6443099498dc12ec9e9753201ebbac3875e23af9` |
| Reviewer commit | `c23fc2fc1d209b554857f0f8b0533502a8346e1c` |
| Reviewer source SHA-256 | `c63c461e2709b2da11249ce054e225afeafeff25d54e99b229970d2415b67eef` |
| Wheel SHA-256 | `2767e3ee362f2455c6c0fd07633e66371a31216b9e1c1086aed0e88e4ad38076` |
| Native binary SHA-256 | `625e7bfab1417183421779a2a01b3c3f945b7784e78066aca6ddcf713d68f019` |
| Paired scope | 4 instances × 3 seeds × 3 repeats × 2 axes × 5 modes = 360 axes |
| Pilot scope | 12 instances × 3 seeds × 5 modes × wall-clock = 180 axes |
| Resource envelope | 6 shard processes × 4 threads; 24 compute-thread cap |
| Host scheduler observation | 24 configured C++ workers; 49 observed service threads |

Wheel receipt simultaneously verified its file SHA-256, installed `direct_url.json`,
site-packages source, native binary SHA-256, and the Git revision compiled into the
native module. All attempt03 axes record the same identities. The final review also
binds every raw axis JSON and its SHA-256 sidecar through deterministic tree hashes:
paired `754f684e...4522ab` for 360 axes and Pilot `e38632cc...dc524` for 180 axes,
with per-mode subtrees retained in the review JSON.

## 实现状态

| Mode | Implemented execution boundary | Semantic status | Main failure domain |
|---|---|---|---|
| `current_stage052` | Existing native kernels and candidate transaction path | Historical path retained; accepted-result equivalence is not established; 72/72 paired and 36/36 Pilot axes completed | Solve-local |
| `python_candidate_control` | Historical Python Candidate Control worker path | Semantic baseline; all axes completed | Python worker pool |
| `per_solve_runtime` | One contiguous SoA Python→C++ call per candidate round; staged cache/control commit | Protocol implemented, but fixed-work differential failed and one wall-clock transaction reached its deadline before dispatch | Solve-local native transaction |
| `full_native_alns` | One Python→C++ call per instance/seed | Prototype only: simplified route mutations; Stage 2.3 lanes/operators, refinement, cache lifecycle, Candidate Control and Stage 4 semantics are incomplete | Single native solve |
| `host_scheduler` | C++ UDS accept loop, POSIX shared memory and 24-worker request queue | Native control plane implemented; inherits incomplete full-native semantics, uses a pickle result blob, and Python-aware solve work remains GIL-serialized | Run-wide service |

`native_execution_config=None` keeps the historical path and guard. Explicit protocols
fail without fallback on worker/service loss, partial IPC, deadline, hash mismatch and
cache commit failure. Output shared memory now uses an ACK and server-owned unlink.

## Paired Attempt 03

### Replay and failure facts

| Mode | Completed | Failed | Fixed-work differential |
|---|---:|---:|---|
| `current_stage052` | 72 | 0 | Primary denominator |
| `python_candidate_control` | 72 | 0 | Semantic baseline |
| `per_solve_runtime` | 71 | 1 | Failed |
| `full_native_alns` | 45 | 27 | Failed |
| `host_scheduler` | 45 | 27 | Failed |

The single per-solve failure was `r101_21/2014`, repeat 2, `wall_clock_30`:
the candidate round reached the deadline before dispatch and failed without fallback.
Every full/host fixed-work failure was a 100-customer axis whose initial route split
could not fit the atomic 100-started-call budget (27 axes per mode).

Among the 36 fixed-work comparisons, per-solve matched Python Candidate Control on
objective and routes in only 9 cases. It matched none of the complete candidate/
acceptance trajectory, operator statistics, Stage 4 state, or cache lifecycle streams.
Deadline-boundary evidence matched 36/36. Therefore the explicit worker protocol is
implemented, but semantic equivalence is not established.

### Paired performance facts

| Mode | Median solver s | Median effective iterations | Median exact calls | Median RSS MiB | 100-customer median vs current |
|---|---:|---:|---:|---:|---:|
| `current_stage052` | 0.961 | 158.5 | 100 | 508.5 | denominator |
| `python_candidate_control` | 31.434 | 139 | 55 | 430.9 | -98.0% |
| `per_solve_runtime` | 26.189 | 139 | 55 | 471.7 | -96.7% |
| `full_native_alns` | 2.112 | 1000 | 43,740 | 412.2 | not measurable; fixed-work axes failed |
| `host_scheduler` | 2.193 | 1000 | 43,740 | 373.7 | not measurable; fixed-work axes failed |

Per-solve family medians relative to current were C -95.8%, R -96.5%, and RC
-98.5%, all far below the required no-worse-than -3% family gate. Its median across
all completed paired axes was 5.95% slower than current and approximately equal to
Python Candidate Control, but that aggregate does not override the 100-customer
fixed-work failure.

The very high full/host throughput is not a valid speedup result: those modes execute
different, incomplete search semantics and produced much worse objectives.

## Pilot Attempt 03

Pilot raw replay passed all 180 axes. Overall qualification still failed because the
full/host semantics are incomplete and none of the three new architectures passed the
quality/differential gates.

| Mode | Median solver s | Median effective iterations | Median exact calls | Vehicle median | Quality vs current (better/equal/worse) |
|---|---:|---:|---:|---:|---:|
| `current_stage052` | 8.978 | 1000 | 749 | 3.5 | denominator |
| `python_candidate_control` | 5.913 | 1000 | 88 | 3.5 | 2 / 21 / 13 |
| `per_solve_runtime` | 5.609 | 1000 | 91 | 3.5 | 1 / 14 / 21 |
| `full_native_alns` | 0.043 | 1000 | 3,683 | 4 | 0 / 5 / 31 |
| `host_scheduler` | 0.050 | 1000 | 3,683 | 4 | 0 / 5 / 31 |

Relative to Python Candidate Control, per-solve was better/equal/worse on
9/13/14 Pilot objectives. It therefore fails the required wall-clock objective
non-regression condition even though its all-instance median solver time is lower.

## Current implementation and accepted Pilot drift

The historical identity chain was verified before drift calculation: campaign and raw
manifest sidecars, all three batch manifests, 36 shard-manifest hashes, each selected
raw-artifact hash, accepted review status/gates, producer revision, and the tracked
accepted raw/review manifest identities all matched. The accepted raw manifest is
`5aa8b773...f38e4`; the accepted review manifest is `24de9cc9...27727`.

All 36 accepted-attempt72 historical validator flags were true. The new current-mode
objective matched attempt72 on 30/36 instance/seed pairs. The six changes were
`c101_21/2015`, all three `r101_21` seeds, and `rc101_21/2014,2015`.

For the nine 100-customer pairs, the median new/historical solver-time ratio was
1.0026 (range 0.9768–1.0056), so the 100-customer timing environment was nearly flat.
For the 27 small-instance pairs (C5/C10/C15), the median ratio was 1.3228
(range 0.9536–1.8637); small-instance
runtime drift is material and must not be described as an architecture speedup.

Attempt72 uses a different semantic-digest/event-stream schema. This review verifies
historical validator/objective/timing fields but does not claim byte-identical core
transaction replay across those schemas.

## Instrumentation limitations

The v3 `persistence_seconds` field starts before solve execution and therefore includes
solver time; it is not a valid standalone persistence-cost measurement. Artifact bytes
and retained directory sizes remain valid. Host per-axis CPU/RSS fields measure the
client shard, not the separate scheduler process; the manifest records 49 observed
scheduler service threads, but the 24-compute-thread envelope is not independently
closed by per-process CPU/RSS accounting. These limitations block resource-efficiency
promotion claims.

The CUDA threshold is **not met on replayable evidence**. Attempt03 does not record
native candidate-screening occupancy, so exact-backend `launch_occupancies` cannot be
used as a substitute. The earlier `True` result based on exact-route batch size was
invalidated; no CUDA evaluation was run.

## Verification

- Full pytest: `1127 passed in 208.59s`.
- Ruff: all checks passed.
- Strict mypy: success across 80 source files.
- Publication metadata and native-architecture directed suite: `16 passed`.
- `git diff --check`: passed before publication commit.

## Persistence and capacity

| Evidence generation | Retained size | Status |
|---|---:|---|
| attempt01 | 16 GiB | Immutable v1 evidence-design failure; complete rows were duplicated into JSON and reviewer memory was unbounded |
| attempt02 | 9.2 MiB | Immutable v2 failure evidence; non-finite diagnostic encoding and failed-axis review were incomplete |
| attempt03 | 12 MiB | Current comparison evidence |

After all retained generations, the WSL filesystem had approximately 581 GiB free.
No evidence was cleaned. Attempt03 semantic streams store counts and SHA-256 digests;
they eliminate duplicated rows but are not a substitute for a future canonical
Parquet/raw-event replay if one of these architectures is selected.

## Decision matrix

| Criterion | Current | Python CC | Per-solve C++ | Full native | Host scheduler |
|---|---|---|---|---|---|
| Historical execution path retained | Yes; result equivalence not established | Reference only | No | No | No |
| Same-wheel Pilot replay | Pass | Pass | Pass | Pass | Pass |
| Fixed-work semantic differential | Baseline | Baseline | Fail | Fail | Fail |
| 100-customer ≥15% paired gain | Baseline | Fail | Fail | Not measurable | Not measurable |
| Wall-clock objective non-regression | Baseline | Not a promotion candidate | Fail | Fail | Fail |
| Build/operational complexity | Medium | Low/medium | Medium | High | Highest |
| Failure/recovery scope | Solve-local | Worker pool | Solve-local | Native solve | Run-wide service |
| Production eligibility after this round | Not requalified; historical default unchanged | No promotion | No | No | No |

This table does not select the final production route. If a new architecture is to be
continued, it needs a new implementation stage and new labels. Production calibration,
successor attestation, runtime freeze, canonical Pilot and Formal remain explicitly
deferred until after a user selection.

## Evidence pointers

- `experiments/manifests/stage052_native_architecture_comparison_attempt03_artifact_manifest.json`
- `experiments/summaries/stage052_native_architectures_paired_attempt03_review.json`
- `experiments/summaries/stage052_native_architectures_paired_attempt03_report.md`
- `experiments/summaries/stage052_native_architectures_paired_attempt03_review_manifest.json`
- `experiments/summaries/stage052_native_architectures_pilot_attempt03_review.json`
- `experiments/summaries/stage052_native_architectures_pilot_attempt03_report.md`
- `experiments/summaries/stage052_native_architectures_pilot_attempt03_review_manifest.json`
- `results/stage05.2_native_architecture_<mode>_<scope>_attempt03/`
