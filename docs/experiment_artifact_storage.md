# 实验产物存储规则：v2 policy、v3 physical schema 与历史兼容

本规则适用于 Stage 0–8 的实验产物。它只约束 artifact persistence（产物持久化），不改变 ALNS、`evrptw.objective`、统一 validator（验证器）、vehicle-first acceptance（车辆数优先接受规则）或 exact charging（精确充电子问题）的算法语义。

## 当前可用状态

仓库保留 `artifact-storage-v2` 作为 storage policy（存储策略），定义在
`src/evrptw/artifacts.py`；当前物理证据使用 `screening_decisions_v3`，并继续读取
v1、旧 v2 与 legacy evidence。Stage 5.2 只维护一套当前代码，A--G 是顺序 gate，
不是独立版本。具体 attempt 的历史状态、失败原因和 prerequisite 关系记录在 raw
manifest、retention registry（保留登记表）和 change log，不硬编码进长期政策。

所有 gate 必须绑定经过审查的直接 predecessor、clean commit（干净提交）和 frozen
wheel runtime（冻结 wheel 运行时）；actual selected workers/backend 只能由当前 raw
review 决定。`attemptNN`/`rerunNN` 是唯一运行身份，不是代码版本。

v1/v2 的固定策略为：

| 字段 | 固定值 |
| --- | --- |
| `storage_policy_version` | `artifact-storage-v1` 或 `artifact-storage-v2` |
| `screening_schema_version` | 新 evidence 固定 `screening_decisions_v3`；旧 v2 保持可读 |
| `event_format` | `parquet` |
| `compression` | `zstd` |
| `compression_level` | v1 为 `3`；accepted v2 为 `1` |
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

v1 不要求 shard manifest（分片清单）；最后两项由 v2 新增。Stage 5.2 Pilot/Formal
在顶层 canonical run 下增加 `batch0001` 等内部 batch 目录，并通过 signed campaign/batch
manifest（签名活动/批次清单）登记 logical path、root alias、volume identity、byte
count 与 checksum；batch 不得伪装成新 attempt。`failure` 不适用时可以不生成，但
manifest 必须记录 `artifact_status.failure=not_applicable`。所有 raw evidence（原始
证据）先写入 Git-ignored active staging root；运行封存后由 retention interface
校验完整 tree SHA-256 和字节数，并移入
`d_archive/stage05.2/history/<run_label>/`。tracked summary（受 Git 跟踪的汇总）只能
由独立 reviewer 在 raw replay（原始证据回放）通过后发布。

## Critical evidence 与索引

必须逐条保存的 critical evidence（关键证据）包括 exact call 的 started/completed、feasible/infeasible、screening decision、failure、cache lookup/store/hit/eviction/oversize、incremental propagation/fallback、deadline、execution error、accepted candidate、global-best candidate、vehicle-count 变化和所有 exact route evaluation。

普通成功候选、普通 rejected candidate、重复 route timing、operator call 总量和按 `run/lane/iteration/operator/reason` 的诊断数据可以进入 `diagnostic.parquet` 聚合。聚合不得删除或改变 critical event，也不得影响 validator、objective、exact-call ordering 或 failure replay。route sequence 只在 `route_dictionary.parquet` 保存一次；事件使用整数 route/lane/operator ID。

Parquet 使用明确 Arrow schema（Arrow 模式）和 dictionary encoding（字典编码）；
v1 使用 Zstandard level 3，accepted v2 使用 level 1。lane/operator 字典写入
trace index，route 字典写入独立 route dictionary；v2 streaming 不得改变这些
编码和索引语义。

一个 cache lookup 及其紧随的 hit/miss 结果在物理事件文件中只保存一条 `lookup_result`。旧 JSON/JSONL 事件仍按其原始双事件语义读取。`trace.json` 是 trace index（轨迹索引），只保存 counters、配置、字典、Parquet 引用及 schema fingerprint，不重复嵌入完整事件集合。

## `artifact-storage-v2` 实施契约

### Streaming writer

- Parquet row group 固定为 65,536 rows；单 writer 最多同时缓存 2 个 row groups；
- event、route dictionary、screening definitions、screening occurrences 和 diagnostic stream 分别使用 typed column buffers（类型化列缓冲）增量 flush，不得使用 dict-per-row 热点或先在 Python list 中累积全量 run；
- `ArtifactStorageConfig(storage_policy_version="artifact-storage-v2")` 只在 v2 writer/reader、配置校验和测试全部落地后开放；
- writer 生命周期必须覆盖 `open shard → append → flush → finalize/abort`，异常路径也要关闭 writer 并生成 partial manifest；
- reader 同时支持 v1 bundle、旧 v2 shard、v3 physical schema 和 immutable legacy bundle，并以 streaming k-way merge（流式多路归并）恢复历史排序。

### Shard ownership

- job parallelism（任务级并行）的工作单元是独立 `(instance, seed)` shard；包含固定四 axis 的性能实验把同一 pair 的四 axis 写入同一 shard；benchmark 的不同时间预算作为 shard 内的明确 axis/budget 字段；
- worker 只能写自己拥有的 shard，不得写 run-level control 文件或其他 worker 的目录；
- parent 只能在所有 worker 状态确定后写 run metadata、control manifest 和 sidecar，不得把全部 event rows 读回内存合并；
- 每个 shard 独立记录 schema fingerprint、row count、byte size、checksum、evidence completeness、worker identity、axis identity 和 provenance；
- manifest 以 canonical `(instance, seed)` key 排序，不依赖 worker completion order（完成顺序）。

### Deterministic identity

v2 event identity 由 canonical shard ordinal（规范分片序号）和 shard-local event ID（分片内事件编号）确定。reviewer 使用该复合身份和 manifest 顺序重放，不能依赖进程完成顺序、文件系统遍历顺序或 Parquet 字节完全一致。重新写出的 Parquet 只要求 schema 和声明语义一致，不要求 byte equality（字节相等）。

### 中断、超限和失败

达到 shard 或 run byte budget 时，writer 必须 flush/close 当前 row group，封存已完成的
raw、solution、event、environment 和 failure evidence，将 shard 与 run 标记为
`evidence_completeness=partial`，写入 manifest/sidecar 后立即抛出错误。不得静默截断、
覆盖或使用 serial fallback（串行兜底）。partial、timeout、failure 和 manifest error
均禁止发布科学 summary；其 sealed directory 经 retention audit 后移入外部归档，
不在工作区无限累积。

## v2 替换门槛

`stage05.2_artifact_streaming_attemptNN` 必须同时满足：

1. 相同 fixed-work 输入下，v1/v2 的 validator、objective、critical-event、exact-call、candidate/cache 和 failure semantics 完全一致；
2. v1 历史和现有 bundle 继续通过同一 reader/reviewer；
3. artifact persistence time 不超过 end-to-end time 的 30%；
4. peak RSS 不超过 Stage 5.2 v1 baseline 的 50%；
5. partial/timeout/worker failure 都产生可校验 shard manifest 并 fail fast；
6. independent reviewer 从 raw shard 重算所有汇总，不信任 runner 自报计数。

当前 C 运行必须完成声明的 performance scope 与 fixed-work equality，证明 expanded
logical semantics（展开逻辑语义）完全一致，并满足 30% persistence 门槛。任何旧
physical schema、remediation source 或失败 predecessor 的关系只由 manifest、
retention registry 和 change log 保存。后续 gate 若破坏这些门槛必须标记
`NOT_READY`，不得退回隐式 v1 fallback 或降低阈值。

## 历史兼容

Stage 0 frozen baseline、Stage 2/3 历史证据和已发布 v1 bundle 不做物理迁移、不压缩、不移动、不重写。它们继续通过 legacy/v1 reader 读取，并在 registry 中保留其真实 `storage_format`、`retention_class`、`policy_compliance`、dirty、failure 和 publication 状态。logical mapping（逻辑映射）不代表复制或移动。

## Stage 5.2 retention policy

- `Stage052RetentionPolicy` 固定 `workspace_full_evidence=active_only`，complete 与 failed
  run 均执行 `archive`。
- audit inventory 记录 run label、component、状态、completeness、source commit、
  prerequisite identities、文件数、总字节数和 tree SHA-256，并使用独立 sidecar 签名。
- archive 必须显式绑定 inventory SHA-256，并由 ignored storage-root locator 验证 alias
  与 volume identity。同 volume 使用校验后的原子移动；跨 volume 先复制到目标卷隐藏
  临时目录，完整复验后在目标卷原子落位，最后才清理源。源目录在 audit 后发生变化、
  目标内容不同或迁移后复验失败时均 fail fast，且不得删除源数据。
- active/unsealed run 默认不能进入 inventory；仅改造前历史迁移可显式 override，并同时
  绑定预期目录数和总字节数。registry 更新必须按 run label 原子合并，禁止覆盖历史行。
- 轻量 `stage05.2_retention_registry.csv` 只记录 archive alias 和相对路径，不记录本机
  绝对路径。详细实现变更追加到 `docs/stage052_change_log.md`。
- prerequisite/review 以 run label 调用 `resolve_retained_run`，由 registry 和本地
  storage-root locator 解析 archive alias；返回现有 runner/reviewer 前再次核对文件数、
  字节数和 tree SHA-256。调用方不得自行拼接或在 tracked 文件中保存绝对归档路径。
- 归档目录只能作为只读 comparison/prerequisite/replay 输入；reviewer 不得把新的 review
  generation 写回已登记的归档 tree，否则会破坏 registry checksum。需发布新 generation
  的 raw 必须留在 active root，发布并封存后再归档。

## 正式运行前 preflight

每个新 run 必须检查 canonical label、attempt/rerun 唯一性、artifact type、control 配置、manifest 与 sidecar、source/config/instance/environment/reference provenance、Parquet schema fingerprint、row count、byte size、shard identity 和 raw-to-summary consistency。独立 reviewer 必须先验证 run/shard manifest，再 replay raw solution、events、trace index、route dictionary、validator 和 objective。

Stage 5.2 D 以后，worker-owned shard 的 `worker_identity` 必须是实际执行进程 PID，
并且可在同一 run 的 50 ms process-tree resource samples 中找到；多 worker 运行禁止
把 parent PID 当作 shard owner。resource summary、raw/solution/trace 和 per-run CSV
必须拥有完全相同的 canonical scope，任何缺失、重复、partial 或 fallback 标记均
fail fast（快速失败）。
reviewer 还必须验证 canonical `(instance, seed) → shard_ordinal` 映射、每个 shard
manifest 与专属 sidecar 的双向 SHA-256 绑定，以及 trace index 中相同的复合 event
identity。性能 axis 必须按配置顺序串行且互不重叠，finalization 只能在最后一个
axis 完成后开始。

Stage 5.2 re-review（重新审查）不得原地覆盖受 manifest 保护的 report/findings。
新 report 与 findings 先写入 `review/generations/<content-sha256>/` 并完成 fsync，
最后只原子替换 `review_manifest.json` 这一可信指针；旧 accepted review 的 manifest、
report 和 findings 按旧 manifest SHA-256 归档到 `review/history/`。发布中断时旧
manifest 及其引用文件仍保持可验证，orphan generation（孤立审查代次）不参与门槛。

Stage 5.2 以后，报告必须分列 solver time、artifact persistence time 和 end-to-end time；CPU 使用率、芯片功耗、kernel time 或压缩比不能单独构成加速结论。
