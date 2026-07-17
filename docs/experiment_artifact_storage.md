# 实验产物存储规则：v1 现状与 v2 迁移契约

本规则适用于 Stage 0–8 的实验产物。它只约束 artifact persistence（产物持久化），不改变 ALNS、`evrptw.objective`、统一 validator（验证器）、vehicle-first acceptance（车辆数优先接受规则）或 exact charging（精确充电子问题）的算法语义。

## 当前可用状态

仓库当前实现的是 `artifact-storage-v1`，定义在 `src/evrptw/artifacts.py`。Stage 5.2 必须实现并独立验收 `artifact-storage-v2`，通过后才能用于 Stage 5.2 pipeline pilot（流程试运行）、正式 benchmark 及后续阶段。v2 实现完成前，runner 不得接受 v2 配置，也不得把本文的迁移契约写成已具备的运行能力。

当前 v1 固定策略为：

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

所有新 runner 必须提供启用的 `[artifact_storage]`，并通过共享 writer/reader 写入或读取产物。缺失、禁用、legacy format（旧格式）或不支持的参数必须在启动前失败。runner 不得重复实现事件、checksum（校验和）或 manifest（清单）逻辑。

## 共同物理布局

`run_label` 必须符合 `stageNN[_minor]_component_attemptNN` 或 `rerunNN`：

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
    <canonical>_screening_checks_<instance>_<seed>.parquet
    <canonical>_diagnostic_<instance>_<seed>.parquet
    <canonical>_environment_<instance>_<seed>.json
    <canonical>_failure_<instance>_<seed>.json
    <canonical>_shard_manifest_<instance>_<seed>.json
    <canonical>_shard_manifest_<instance>_<seed>.sha256
  review/
```

v1 不要求 shard manifest（分片清单）；最后两项由 v2 新增。`failure` 不适用时可以不生成，但 manifest 必须记录 `artifact_status.failure=not_applicable`。所有 raw evidence（原始证据）只写入 Git-ignored `results/`；tracked summary（受 Git 跟踪的汇总）只能由独立 reviewer 在 raw replay（原始证据回放）通过后发布。

## Critical evidence 与索引

必须逐条保存的 critical evidence（关键证据）包括 exact call 的 started/completed、feasible/infeasible、screening decision、failure、cache lookup/store/hit/eviction/oversize、incremental propagation/fallback、deadline、execution error、accepted candidate、global-best candidate、vehicle-count 变化和所有 exact route evaluation。

普通成功候选、普通 rejected candidate、重复 route timing、operator call 总量和按 `run/lane/iteration/operator/reason` 的诊断数据可以进入 `diagnostic.parquet` 聚合。聚合不得删除或改变 critical event，也不得影响 validator、objective、exact-call ordering 或 failure replay。route sequence 只在 `route_dictionary.parquet` 保存一次；事件使用整数 route/lane/operator ID。

Parquet 使用明确 Arrow schema（Arrow 模式）、Zstandard level 3 和 dictionary encoding（字典编码）。lane/operator 字典写入 trace index，route 字典写入独立 route dictionary；v2 streaming 不得改变这些编码和索引语义。

一个 cache lookup 及其紧随的 hit/miss 结果在物理事件文件中只保存一条 `lookup_result`。旧 JSON/JSONL 事件仍按其原始双事件语义读取。`trace.json` 是 trace index（轨迹索引），只保存 counters、配置、字典、Parquet 引用及 schema fingerprint，不重复嵌入完整事件集合。

## `artifact-storage-v2` 实施契约

### Streaming writer

- Parquet row group 固定为 65,536 rows；单 writer 最多同时缓存 2 个 row groups；
- event、route dictionary、screening checks 和 diagnostic stream 分别增量 flush，不得先在 Python list 中累积全量 run；
- `ArtifactStorageConfig(storage_policy_version="artifact-storage-v2")` 只在 v2 writer/reader、配置校验和测试全部落地后开放；
- writer 生命周期必须覆盖 `open shard → append → flush → finalize/abort`，异常路径也要关闭 writer 并生成 partial manifest；
- reader 同时支持 v1 bundle、v2 shard bundle 和 immutable legacy bundle。

### Shard ownership

- job parallelism（任务级并行）的工作单元是独立 `(instance, seed)` shard；包含固定四 axis 的性能实验把同一 pair 的四 axis 写入同一 shard；benchmark 的不同时间预算作为 shard 内的明确 axis/budget 字段；
- worker 只能写自己拥有的 shard，不得写 run-level control 文件或其他 worker 的目录；
- parent 只能在所有 worker 状态确定后写 run metadata、control manifest 和 sidecar，不得把全部 event rows 读回内存合并；
- 每个 shard 独立记录 schema fingerprint、row count、byte size、checksum、evidence completeness、worker identity、axis identity 和 provenance；
- manifest 以 canonical `(instance, seed)` key 排序，不依赖 worker completion order（完成顺序）。

### Deterministic identity

v2 event identity 由 canonical shard ordinal（规范分片序号）和 shard-local event ID（分片内事件编号）确定。reviewer 使用该复合身份和 manifest 顺序重放，不能依赖进程完成顺序、文件系统遍历顺序或 Parquet 字节完全一致。重新写出的 Parquet 只要求 schema 和声明语义一致，不要求 byte equality（字节相等）。

### 中断、超限和失败

达到 shard 或 run byte budget 时，writer 必须 flush/close 当前 row group，保留已完成的 raw、solution、event、environment 和 failure evidence，将 shard 与 run 标记为 `evidence_completeness=partial`，写入 manifest/sidecar 后立即抛出错误。不得静默截断、覆盖、删除 partial shard，或使用 serial fallback（串行兜底）。partial、timeout、failure 和 manifest error 均禁止发布 summary。

## v2 替换门槛

`stage05.2_artifact_streaming_attemptNN` 必须同时满足：

1. 相同 fixed-work 输入下，v1/v2 的 validator、objective、critical-event、exact-call、candidate/cache 和 failure semantics 完全一致；
2. v1 历史和现有 bundle 继续通过同一 reader/reviewer；
3. artifact persistence time 不超过 end-to-end time 的 30%；
4. peak RSS 不超过 Stage 5.2 v1 baseline 的 50%；
5. partial/timeout/worker failure 都产生可校验 shard manifest 并 fail fast；
6. independent reviewer 从 raw shard 重算所有汇总，不信任 runner 自报计数。

未通过时继续使用 v1 做诊断，不得进入 Stage 5.2 pipeline pilot 或正式 benchmark，也不得把未验收 v2 写成默认策略。

## 历史兼容

Stage 0 frozen baseline、Stage 2/3 历史证据和已发布 v1 bundle 不做物理迁移、不压缩、不移动、不重写。它们继续通过 legacy/v1 reader 读取，并在 registry 中保留其真实 `storage_format`、`retention_class`、`policy_compliance`、dirty、failure 和 publication 状态。logical mapping（逻辑映射）不代表复制或移动。

## 正式运行前 preflight

每个新 run 必须检查 canonical label、attempt/rerun 唯一性、artifact type、control 配置、manifest 与 sidecar、source/config/instance/environment/reference provenance、Parquet schema fingerprint、row count、byte size、shard identity 和 raw-to-summary consistency。独立 reviewer 必须先验证 run/shard manifest，再 replay raw solution、events、trace index、route dictionary、validator 和 objective。

Stage 5.2 以后，报告必须分列 solver time、artifact persistence time 和 end-to-end time；CPU 使用率、芯片功耗、kernel time 或压缩比不能单独构成加速结论。
