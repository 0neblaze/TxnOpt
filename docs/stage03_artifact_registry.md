# Stage 3 artifact registry（阶段 3 产物登记）

本文档把外部路线图的 canonical naming（规范命名）落到 Stage 3.0、Stage 3.1
历史 evidence（证据）、Stage 3.2 cache/incremental evidence（缓存/增量证据）
和 Stage 3.3 exact-deadline evidence（精确截止证据）上。它只整理 registry（登记表）、
manifest（校验清单）和 legacy mapping（历史路径映射），不重新运行 solver
（求解器），不移动或复制 `results/` 中的 raw evidence（原始证据）。

## 适用范围

当前维护以下子阶段：

| Canonical stage ID | Component | 已登记的历史 run |
| --- | --- | --- |
| `stage03.0` | `measurement` | 4 个 smoke 历史轮次和 1 个 formal 历史轮次 |
| `stage03.1` | `screening` | 3 个 smoke 历史轮次和 1 个 formal 历史轮次 |
| `stage03.2` | `cache_incremental` | 新 runner 的 smoke/formal evidence，review 通过后登记为 verified |
| `stage03.3` | `exact_deadline` | attempt01--06；05 为 accepted smoke，06 为 accepted formal |

`stage03.3` 已由独立 formal review 发布 `READY_FOR_STAGE03_4`；`stage03.4`
仍是 planned（计划中）阶段，不创建伪造结果或 readiness（就绪）结论。

## Canonical naming and logical layout（规范命名与逻辑目录）

阶段产物使用：

```text
<stage_id>_<component>_<attempt_or_rerun>_<artifact_type>[_<instance>_<seed>].<ext>
```

运行目录使用：

```text
results/<stage_id>_<component>_<attempt_or_rerun>/<instance>/<seed>/
```

本次登记的 canonical run label（规范运行标签）为：

| Legacy run label | Canonical run label |
| --- | --- |
| `stage03_measurement_smoke01` | `stage03.0_measurement_attempt01` |
| `stage03_measurement_smoke02` | `stage03.0_measurement_attempt02` |
| `stage03_measurement_smoke03` | `stage03.0_measurement_attempt03` |
| `stage03_measurement_smoke04` | `stage03.0_measurement_attempt04` |
| `stage03_measurement_formal01` | `stage03.0_measurement_attempt05` |
| `stage031_cheap_screening_smoke01` | `stage03.1_screening_attempt01` |
| `stage031_cheap_screening_smoke02` | `stage03.1_screening_attempt02` |
| `stage031_cheap_screening_smoke03` | `stage03.1_screening_attempt03` |
| `stage031_cheap_screening_formal01` | `stage03.1_screening_attempt04` |

这里的 canonical path（规范路径）是 registry 中的 logical view（逻辑视图）。
例如旧的
`results/stage03-measurement_formal01/solutions/...json` 会登记为类似：

```text
results/stage03.0_measurement_attempt05/c101C5/2014/
stage03.0_measurement_attempt05_solution_c101C5_2014.json
```

实际旧文件仍是唯一 source of truth（事实来源）；不产生第二份 raw、solution、
events 或 trace 文件。

## Registry、manifest 与兼容映射

工具会生成并维护：

- `experiments/registries/stage03.0_artifact_registry.csv`
- `experiments/registries/stage03.1_artifact_registry.csv`
- `experiments/registries/stage03.2_artifact_registry.csv`
- `experiments/registries/stage03.3_artifact_registry.csv`
- `experiments/registries/stage03_legacy_path_map.csv`
- `experiments/manifests/stage03.0_measurement_artifact_manifest.json`
- `experiments/manifests/stage03.1_screening_artifact_manifest.json`
- `experiments/manifests/stage03.2_cache_incremental_artifact_manifest.json`
- `experiments/manifests/stage03.3_exact_deadline_artifact_manifest.json`

Registry 至少登记每个旧目录中的 raw、solution、events、trace、environment、
config、failure、manifest 和 review 文件，以及已发布的 tracked summaries。每行
包含：

- `stage_id`、`component`、canonical `run_label`、`attempt_or_rerun` 和
  `artifact_type`；
- instance/seed、scope、canonical path 和 immutable legacy path；
- `status`、`failure_reason`、`validator_status`、raw review status、trusted
  formal review status 和 `raw_to_summary_status`；
- artifact checksum、checksum source、source/config/instance/environment hash；
- Stage 0 manifest hash、主仓库 revision/dirty 状态、两个 reference repository
  的 revision/dirty 状态、comparison baseline 和 supersedes/mapping 字段。
- `storage_policy_version`、`storage_format`、`compression`、`retention_class`、
  `evidence_completeness`、`schema_fingerprint`、`row_count`、`byte_size` 和
  `policy_compliance`。

历史文件固定标记为 `legacy_json_or_jsonl`、`legacy`、`legacy_unknown`、
`legacy_compatible`；新文件固定标记为 `parquet_or_json_control`、
`critical/diagnostic/control` 和 `current`。Parquet 的 schema fingerprint 与
row count 必须由实际文件重算，不能复制 runner summary。

`stage03_legacy_path_map.csv` 特别保留：

- `experiments/baselines/stage00` 的 `stage00_frozen_baseline` 映射；
- `stage02_constraint_guided_attempt16` 与 `stage02_constraint_guided_rerun09`
  的 Stage 2.3 baseline reference（基线引用）；
- 每个 Stage 3 历史 run directory 到 canonical run label 的映射。

Stage 0 冻结目录、Stage 2.3 历史结果、Stage 3 raw 目录和旧 tracked summary
均不改名、不覆盖、不移动。历史 `repository_dirty=true` 保持原样。

## Preflight and migration（预检与迁移）

只读预检/独立复核：

```bash
uv run python tools/stage03_artifact_migration.py
```

首次生成 registry 和 manifest：

```bash
uv run python tools/stage03_artifact_migration.py --write
```

该工具只读取并验证现有 raw manifest、sidecar、review manifest、tracked summary
和旧路径；不会调用 Stage 3 runner。`--write` 只写新的 tracked registry、manifest
和 mapping 文件，不写入 `results/`。

正式实验或进入下一阶段前，必须检查：

1. canonical label 使用唯一的 `attemptNN`/`rerunNN` 后缀；
2. artifact type 明确，instance/seed scope 完整；
3. raw manifest 和 `manifest.sha256` sidecar 可验证；
4. tracked registry 与 manifest 的 checksum、row hash 可重算；
5. Stage 0、Stage 2.3 和旧 Stage 3 路径映射存在；
6. raw runner CSV 与 tracked summary 的语义字段 raw-to-summary 一致。

raw-to-summary 是 semantic comparison（语义比较），不是未经审计的 byte copy（字节
复制）。Stage 3.0 比较 objective、feasibility、trace counters 和 reconciliation
字段；Stage 3.1 另外比较 screening counters/reason statistics。历史 review
缺失记为 `not_present`，没有 summary 的轮次记为 `not_published`，不被提升为通过。

新 run 的物理路径为 `results/<run_label>/<instance>/<seed>/`，registry 中的
canonical path 只是 logical view（逻辑视图），不表示复制或移动。独立 reviewer
必须先验证 manifest/sidecar、Parquet schema、row count、checksum、critical event
completeness、validator/objective replay、trace reconciliation、deadline 和
raw-to-summary；通过前不得写 tracked summary 或把 registry 行标为 published。

## 历史异常与失败保留

旧 smoke 轮次如果缺少 manifest、sidecar、review 或发生中断，registry 仍登记已有
文件，并显式记录 `legacy_with_manifest_error`、`not_present`、`not_published` 或
failure reason。工具不会补造 checksum、solution 或审计结论；原始异常必须继续可见。

Stage 3.0 的 formal raw manifest、formal raw-to-summary 和 trusted formal review
使用独立字段记录，不能因为早期 smoke 历史轮次不完整而被混成同一个结论。Stage
3.1 formal 的 comparison baseline 是 canonical
`stage03.0_measurement_attempt05`；exact-call reduction（精确调用减少）仍只是
measurement evidence（测量证据），不是 Stage 3.3 fixed-work/wall-clock
acceleration（固定工作量/墙钟加速）结论。

Stage 3.2 的 comparison baseline 是 Stage 3.1 formal per-run evidence。reviewer
还必须重算 route cache key、cache hit/miss/store/eviction、feasible/infeasible
cache 生命周期、changed/unchanged route、station reachability bitset，以及
incremental propagation 与 full propagation 的一致性。Stage 3.2 formal review
状态必须为 `READY_FOR_STAGE03_3` 才能进入下一阶段；这不是 Stage 3.3 加速结论。
