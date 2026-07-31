# 实验产物存储规则：lifecycle v3、retention v3 与历史兼容

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

## Lifecycle v3 强制入口

所有顶层实验必须先登记在 `configs/experiment_catalog.toml`，并通过
`experiment-lifecycle` 完成 `PLANNED -> PERMITTED -> RUNNING -> SEALED -> REVIEWED ->
CLASSIFIED -> RETAINED/COMPACTED -> CLOSED`。上一项顶层实验未 `CLOSED`、存在
`BLOCKED_RETENTION`、catalog/prerequisite/source/config/lock/environment identity
不一致，或资源计划超过登记上限时，程序在创建 run directory 前拒绝启动。

retention class（保留类别）只能由独立 reviewer 的受控状态、failure code、完整失败
身份和签名 root-cause adjudication 通过规则引擎产生。未知根因固定为 `unknown_full` 并
阻塞后续实验。v1/v2 registry 只读兼容；
`experiments/registries/experiment_lifecycle_v3_migration.json` 固定其身份，v3 是唯一
允许新增治理事实的 schema。

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
`e_archive/stage05.2/history/<run_label>/`。tracked summary（受 Git 跟踪的汇总）只能
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
3. artifact persistence time 不超过 end-to-end time 的 36%；
4. peak RSS 不超过 Stage 5.2 v1 baseline 的 50%；
5. partial/timeout/worker failure 都产生可校验 shard manifest 并 fail fast；
6. independent reviewer 从 raw shard 重算所有汇总，不信任 runner 自报计数。

当前 C 运行必须完成声明的 performance scope 与 fixed-work equality，证明 expanded
logical semantics（展开逻辑语义）完全一致，并满足 36% persistence 门槛。任何旧
physical schema、remediation source 或失败 predecessor 的关系只由 manifest、
retention registry 和 change log 保存。后续 gate 若破坏这些门槛必须标记
`NOT_READY`，不得退回隐式 v1 fallback 或降低阈值。

## 历史兼容

Stage 0 frozen baseline、Stage 2/3 历史证据和已发布 v1 bundle 不做物理迁移、不压缩、不移动、不重写。它们继续通过 legacy/v1 reader 读取，并在 registry 中保留其真实 `storage_format`、`retention_class`、`policy_compliance`、dirty、failure 和 publication 状态。logical mapping（逻辑映射）不代表复制或移动。

## Stage 5.2 retention policy

- Retention policy v2（留存政策第二版）仅用于读取历史事实，分为 `accepted_full`、
  `unique_failure_full`、`duplicate_failure_reduced`、`rebuildable` 和
  fail-closed（关闭式失败）的 `unknown_full`。accepted/current-chain 与独特根因失败保留
  完整 raw；重复根因只有在签名 adjudication record（裁定记录）绑定稳定
  `root_cause_id`、完整 canonical representative（规范代表）和证据引用后才可减量。
- reduced generation（减量代次）是 audit-only（仅审计），保留 control manifests、
  review generations、日志、checksums、failure evidence 和触发失败的代表 shard；它不得
  作为 scientific comparison（科学比较）或 prerequisite（先决证据）。原 manifest 不
  改写，projection manifest 同时记录所有保留和省略文件的原始 SHA-256。
- audit inventory 记录 run label、component、状态、completeness、source commit、
  prerequisite identities、文件数、总字节数和 tree SHA-256，并使用独立 sidecar 签名。
- archive 必须显式绑定 inventory SHA-256，并由 ignored storage-root locator 验证 alias
  与 volume identity。新归档使用 `e_archive`；`d_archive` 只解析 legacy evidence
  （历史证据）并承担 host/VHDX capacity gate（主机容量门槛）。跨卷先复制到目标卷隐藏
  incoming generation，完整复验后原子发布并登记。源目录在 audit 后变化、目标碰撞、
  校验失败或 registry 写入失败时均保留；已发布但未登记的同一代次可在内容完全一致时
  安全重试。
- cross-role storage migration attestation（跨角色存储迁移证明）必须与详细签名 dry run
  位于同一 governance generation，逐项绑定 source/destination alias、相对路径、logical
  ID、file count、byte count、tree SHA-256、两端 volume identity 与全量 retained
  projection 上限。Stage 5.2 campaign reviewer 必须通过同一通用 verifier 消费 v2
  attestation；仅有一份合法但不含相同 mappings 的 dry run 不构成迁移证明。
- active/unsealed run 默认不能进入 inventory；仅改造前历史迁移可显式 override，并同时
  绑定预期目录数和总字节数。registry 更新必须按 run label 原子合并，禁止覆盖历史行。
- v2 registry identity（注册表身份）为
  `(run_label, segment_id, generation)`；resolver（解析器）优先选择最新 verified v2
  generation。v2 registry 与 full-replay receipts 存放在 `e_archive` 自身的签名
  `.storage-governance` 状态中；既有 tracked
  `stage05.2_retention_registry.csv` 仅作为不可变 v1 fallback（回退），不是并行的 v2
  truth source（事实源）。registry 只记录 archive alias 和相对路径，不记录本机绝对路径。
- 新写 retention v3 分为 `published_full`、`current_accepted_full`、
  `superseded_accepted_capsule`、`unique_failure_capsule`、
  `duplicate_failure_metadata`、`superseded_metadata`、`rebuildable` 和
  `unknown_full`。原位精简严格执行 PREPARED/APPLYING/COMMITTED，输入只能是 controller
  生成的签名 keep/delete 清单及精确计划 SHA-256；原始 manifest 不修改，删除身份追加到
  ledger。close 最多一次内容哈希通读和一次删除遍历，未校准 native backend、隐式
  fallback、重复哈希或全 E 盘扫描均使 performance gate 失败。
- full-retention receipt 必须逐项等于增量 content inventory 的 tree SHA-256、文件数和
  字节数；controller 在 E 盘治理状态中保存该 inventory identity，供未来 supersession
  transaction（取代事务）复用。新 `current_accepted_full` 关闭时，同一 experiment 的旧
  current 通过追加式 supersession projection（取代投影）获得
  `superseded_accepted_capsule` 的有效分类；旧 `CLOSED` record 本身不重开、不改写。
  中断后从 PREPARED/APPLYING checkpoint 重入，不能留下两个有效 current。
- PERMITTED -> RUNNING 会重新绑定同一 immutable plan。worker/thread/process、source、
  config、lock、environment、prerequisite、I/O backend 任一漂移均拒绝；已经 RUNNING 的
  幂等重试也执行同一检查。compaction apply 对 inventory 后新增文件、symlink、缺失文件、
  stat 漂移和内容漂移全部 fail closed。
- `accepted_full` 与 `unique_failure_full` 在登记前必须生成签名 independent replay
  receipt（独立重放回执），绑定 archive tree SHA-256、文件数、字节数、verifier
  identity，以及 validator、objective 和 raw-review replay 三项通过状态。resolver
  每次返回 full-retention generation 前重新验证该回执；只有调用回调而没有可复验回执
  不构成完整 replay。
- prerequisite/review 以 run label 调用 `resolve_retained_run`，由 registry 和本地
  storage-root locator 解析 archive alias；返回现有 runner/reviewer 前再次核对文件数、
  字节数和 tree SHA-256。调用方不得自行拼接或在 tracked 文件中保存绝对归档路径。
- 归档目录只能作为只读 comparison/prerequisite/replay 输入；reviewer 不得把新的 review
  generation 写回已登记的归档 tree，否则会破坏 registry checksum。需发布新 generation
  的 raw 必须留在 active root，发布并封存后再归档。
- 历史 `stage052-retention-v2-20260731` 的人工确认边界保持为不可变事实。未来处置不复用
  该确认，也不要求第二次人工确认；显式 lifecycle close、controller 签名计划和匹配
  SHA-256 共同构成执行授权。same-attempt rolling-batch handoff（同一 attempt 滚动批次
  移交）仍受其顶层 lifecycle identity 约束。

### Stage 5.2 historical migration status

`stage052-retention-v2-20260731` 已于 2026-07-31 完成：307 个 run、356 个 segment、
712,267,368,027 bytes 全部发布到绑定的 `e_archive`，两卷 migration attestation、
307/307 resolver ledger replay 与所有签名 sidecar 均通过。精确删除 manifest 的
SHA-256 为
`0b75becf3478b2183728da082ffeddb016a56ff7f829e8394420127fb7b248c8`。
用户在查看完整 source/target/bytes/tree-SHA 清单和单介质风险后再次字面确认；执行器在
删除前重新完整复验全部源，随后删除 356/356 个清单路径并复核全部不存在。删除执行回执
SHA-256 为
`6192df5608288b9a0692afd512965469e2cc9aed3e5fea43d33d67ad15af3b36`。

本次迁移后 E 盘是这些 raw evidence 的唯一长期介质副本，不得把 content verification
表述为 backup。维护 allowlist 的归档后 dry run 为零候选；Ubuntu VHDX 在 TRIM 后通过
离线 `Optimize-VHD -Mode Full` 从 456,645,410,816 bytes 压缩至约
336,704,045,056 bytes。未来处置必须使用新的签名 lifecycle transaction，不复用本次
人工确认。

上述完成状态只表示 v2 物理迁移完成，不表示 v3 历史语义裁定完成。migration ledger
中的 13 个 pending run 必须各自产生一次 content inventory（内容清单）和独立语义
review；仅生成 inventory 会得到 `INVALID`/`unknown_full`，不会生成可通过的
`historical-migration/gate.json`，并继续阻塞 Stage 5.2 Calibration。完整 gate 必须绑定
每个 review、inventory 及 migration ledger 的精确 SHA-256。

Producer 正常完成后必须调用共享 `seal_cli_attempt` 停止 writer lease 并进入
`SEALED`。Stage 5.2 benchmark 使用 `stage05.2-campaign-manifest-v3` 作为顶层 sealed
manifest；其 `artifacts` 复用写入时已生成的 checksum。资源 calibration 等多 child
manifest producer 使用 terminal manifest 聚合这些签名 identity，只读取小型
manifest/report，并在排他 writer lease 内对当前 run 做一次最终内容验证；child 状态、
checksum、完整文件集或 metadata 有任何漂移都会拒绝。损坏 producer manifest 不会被
重新解释为可信 child，而是进入显式 `untrusted_failure_capsule`。内部 lifecycle state 使用
`.json.sha256`，外部 artifact/reviewer 证据同时支持仓库既有 `.sha256` 约定，两类验证
逻辑保持隔离。

历史语义处置只允许 catalog 登记的
`stage052_historical_semantic_review`。12 个 benchmark 候选只有从 migration ledger
绑定的 v2 retention registry 解析 canonical protected keeper generation，并对账本中
SHA-bound campaign/control/review/batch manifest closure 重新完成零引用扫描时，才可生成
`no_dependency_proof`；调用方不能替换 keeper 路径或提交手写 proof。
完整 semantic command、content inventory、execution receipt、binding 和 aggregate gate
还必须携带 physical historical review 所绑定的同一个 canonical `e_archive` root 与
generation identity；临时 mirror 不能替代该根。`hot_path_attempt04` 只有在重放同一
raw manifest、完整 accepted-review lineage（已接受审查谱系）与 failed-review retry set
（失败审查重试集合）、terminal review execution（最终审查执行）及其精确失败门，并对
v2 registry 绑定的后继 `hot_path_attempt06` 完成全树、accepted review 和 finalized
execution 复验后，才可分类为 `superseded_accepted_capsule`；任一绑定不成立仍保持
`unknown_full`。physical review 自身必须通过仓库 local storage-root locator 解析并验证
live `e_archive` volume identity，且只能读取 signed migration ledger 声明的唯一 v2
registry relative path/SHA；registry SHA 与 generation tree SHA 继续贯穿 inventory、
execution、binding、final review 和 gate。aggregate gate 还会复核 reviewer module/hash、
physical inventory 的 tree digest 必须复用 storage-governance v2 的 pretty canonical JSON
序列化，并以单遍 bounded thread pool（有界线程池）记录每文件 SHA-256；compact JSON
产生的不同摘要不构成 archive drift（归档漂移）。
完整命令、输入前后 hash、签名语义输出和 E state-root execution binding，手工拼接
review JSON 不能替代受控执行。gate consumer 还会重新验证 live locator root，并重放
signed review、inventory、semantic output、execution receipt 与 execution binding 的完整
交叉绑定，并重新执行 retention-class-specific gates：semantic status/class 必须与
review/gate 一致，failure identity/canonical representative、no-dependency proof 或
rebuild proof 必须按类别成立。因此手写 gate bundle 或把 `unknown_full` 重标成安全类别
都不能作为 Stage 5.2 prerequisite。

## Stage 0--8 capacity stop gate

每个新 attempt/rerun 在创建 run directory（运行目录）或启动 worker 前必须提交可重放
experiment plan，包含预计 archive bytes、最大 active workspace，以及 shard/run/batch
hard caps。`preflight_run` 对 E/D/WSL 重新探测同一套 volume identity（卷身份）并采用
动态门槛：E 至少保留 `planned_archive + 0 GiB`，D 至少保留
`projected_WSL_growth + 0 GiB`，WSL ext4 至少保留
`active_workspace + 50 GiB`；Stage 5.2 的 active workspace 不得小于 32 GiB。缺失计划、
身份漂移、空间不足或未核销 permit（许可）均在写入前 fail fast，并持久化完整 capacity
observation（容量观测）。

## Rebuildable asset maintenance

cache、venv、build 和 temporary spool 只在精确 allowlist（允许清单）中接受审计。
maintenance audit（维护审计）必须独立扫描 keeper references（保留者引用），检查活动锁、
保留期、tree SHA-256 和 manifest 引用；venv/build 还必须验证签名 isolated rebuild
proof（隔离重建证明）、sealed wheel/lockfile/Python/native identity 与 smoke test。任何
输入缺失或结果不一致都保留资产。实际清理必须重新计算同一清单，并绑定前一步签名
dry-run receipt（试运行回执）；清单、身份、引用或字节数漂移时拒绝执行并保留资产。
执行开始、失败或完成均生成独立签名回执。活动 Git repository、sealed source snapshot、
registry、manifest、review、checksum、active/unsealed run 永不进入自动清理。
Stage start 与 retention 完成后的 production hook（生产钩子）必须扫描 policy TOML
声明的精确相对 allowlist；不存在的路径如实跳过，存在的 cache/venv/build/spool 必须进入
签名 audit decision。未通过 keeper/rebuild proof 的资产只会以 retained reason（保留
原因）登记，不得把“无法验证”当作空审计或删除许可。真正 apply 还必须绑定完全相同的
dry run、controller 生成的精确计划 SHA-256 和显式 lifecycle close；不另设第二次人工
确认。

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
