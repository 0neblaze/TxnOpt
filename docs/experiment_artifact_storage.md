# 实验产物存储规则 v2

本规则适用于所有新的 Stage 0–8 实验运行。它只改变 artifact persistence（产物持久化），不改变 ALNS、`evrptw.objective`、统一 validator（验证器）、vehicle-first acceptance（车辆数优先接受规则）或任何 Stage 3.3/3.4 算法。

## 强制策略

新 runner 必须在 TOML 中提供 `[artifact_storage]`，并通过 `evrptw.artifacts.ArtifactBundleWriter` 写入产物。当前固定策略为：

| 字段 | 固定值 |
| --- | --- |
| `storage_policy_version` | `artifact-storage-v1` |
| `event_format` | `parquet` |
| `compression` | `zstd` |
| `compression_level` | `3` |
| `critical_evidence` | `full` |
| `diagnostic_evidence` | `aggregate` |
| `per_instance_seed_max_bytes` | `2 GiB` |
| `per_run_max_bytes` | `32 GiB` |

缺失、禁用、legacy format（旧格式）或不支持的参数必须在 runner 启动前失败。runner 不得直接重复实现 JSON、事件、checksum（校验和）或 manifest（清单）写入逻辑。
仓库中仍保留的非 canonical 旧 Stage 0–2 调用只用于读取/复现历史兼容流程；它们的旧配置
不含 `[artifact_storage]`，不得作为新实验入口。仓库提供的新配置已经包含该区块，并会
拒绝非 canonical output directory 或 run label。

## 物理布局

新 run 使用以下结构；`run_label` 必须符合 `stageNN[_minor]_component_attemptNN` 或 `rerunNN` 规范：

```text
results/<run_label>/
  control/
    <canonical>_run_metadata.json
    <canonical>_config.toml
    <canonical>_manifest.json
    <canonical>_manifest.sha256
  <instance>/<seed>/
    <canonical>_raw_<instance>_<seed>.json
    <canonical>_solution_<instance>_<seed>.json
    <canonical>_trace_<instance>_<seed>.json
    <canonical>_events_<instance>_<seed>.parquet
    <canonical>_route_dictionary_<instance>_<seed>.parquet
    <canonical>_diagnostic_<instance>_<seed>.parquet
    <canonical>_environment_<instance>_<seed>.json
    <canonical>_failure_<instance>_<seed>.json
  review/
```

`failure` 不适用时可以不生成，但 manifest 必须记录 `not_applicable` 语义。所有新 raw evidence（原始证据）只写入 Git-ignored `results/`；tracked summary（受 Git 跟踪的汇总）只能由独立 review/preflight 在 raw replay（原始证据回放）通过后发布。

Screening 的规范检查表另存为同一 instance/seed 下的
`<canonical>_screening_checks_<instance>_<seed>.parquet`，并由 trace index 引用。
Manifest 的 `artifact_status.failure` 固定为 `present` 或 `not_applicable`，避免用
缺失文件推断失败状态。一个 cache lookup 及其紧随的 hit/miss 结果在物理
`events.parquet` 中只保存一条 `lookup_result` critical event；旧 JSON/JSONL 事件仍按
原始双事件语义读取。

## 证据分层

必须逐条保存的 critical evidence（关键证据）包括 exact call 的 started/completed、feasible/infeasible、screening decision、failure、cache lookup/store/hit/eviction/oversize、incremental propagation/fallback、deadline、execution error、accepted candidate、global-best candidate、vehicle-count 变化和所有 exact route evaluation。

普通成功候选、普通 rejected candidate、重复 route timing、operator call 总量和按 `run/lane/iteration/operator/reason` 的诊断数据进入 `diagnostic.parquet` 聚合。聚合不得删除或改变 critical event，也不得影响 validator、objective、exact-call ordering 或 failure replay。route sequence 只在 `route_dictionary.parquet` 保存一次，事件使用整数 `route_id` 和稳定的全局递增 `event_id`。

Parquet 使用明确 Arrow schema（Arrow 模式）、Zstandard level 3 和 dictionary encoding（字典编码）。`events.parquet` 的 route、lane、operator 均保存为整数 ID；lane/operator 字典写入 trace index，route 字典写入独立 route dictionary。`trace.json` 是 trace index（轨迹索引），只保存 counters、配置、字典和 Parquet 引用及 schema fingerprint，不重复嵌入完整事件集合。

## 超限和中断

达到 instance/seed 或 run byte budget 时，writer 必须关闭当前 Parquet writer，保留已经完成的 raw、solution、event、environment 和 failure evidence，写入 `evidence_completeness=partial`，更新 manifest 与 sidecar，然后立即抛出错误。不得静默截断、覆盖或删除部分证据；partial、timeout、failure 和 manifest error 均禁止发布 summary。

## 历史兼容

已有 Stage 0 frozen baseline 和 Stage 3.0–3.2 historical raw evidence 不做物理迁移，不压缩、不移动、不重写。它们继续通过 legacy JSON/JSONL reader 读取，并在 registry 中标记为：

```text
storage_format = legacy_json_or_jsonl
retention_class = legacy
policy_compliance = legacy_compatible
```

旧路径由 `stage03_legacy_path_map.csv` 映射到 canonical logical path（规范逻辑路径）；logical mapping 不代表复制或移动。历史 `repository_dirty=true`、失败状态、manifest error 和未发布状态不得被改写成 current evidence。

## 进入正式实验前的 preflight

每个新 run 必须在正式实验前检查 canonical label、attempt/rerun 唯一性、artifact type、control 配置、manifest 与 `.sha256` sidecar、source/config/instance/environment/reference provenance、Parquet schema fingerprint、row count、byte size 和 raw-to-summary consistency。独立 reviewer 必须先验证 manifest，再 replay raw solution、events、trace index、route dictionary、validator 和 objective。

Stage 3 的新 runner 为 `stage03.0_measurement`、`stage03.1_screening`、`stage03.2_cache_incremental`。Stage 3.0–3.2 的历史路径仍是 compatibility mapping；Stage 3.3–3.4 不得因存储格式重构而产生 readiness 或 acceleration claim（加速声明）。
