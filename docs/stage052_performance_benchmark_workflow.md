# Stage 5.2 性能治理与分层 Benchmark 工作流

## 目的与入口

Stage 5.2 先修复已测得的工程瓶颈，再扩大实验规模。它不是直接“多跑 benchmark”，也不以 CPU 占用率或功耗代替端到端性能证据。

唯一入口是：

- `stage05.1_best_known_attempt06` 已通过独立 replay，状态为 `READY_FOR_STAGE05_2`；
- 上游 Formal identity 为 `stage04_adaptive_weights_attempt15`，144 axes 完整；
- main repository（主仓库）使用 clean commit，Stage 0 frozen baseline 与历史 raw evidence 未改写；
- 本文件、`AGENTS.md`、路线图和 `docs/experiment_artifact_storage.md` 的规则一致。

## 单一版本与 Canonical gates

Stage 5.2 只维护一套当前代码。下表 A--G 是同一实现内部必须依次通过的 gate
（门槛），不是七个软件版本，也不得复制成七套长期维护的实现。

| 顺序 | Component | Canonical label | 目标 |
| --- | --- | --- | --- |
| A | `perf_baseline` | `stage05.2_perf_baseline_attemptNN` | 冻结可复验性能基线 |
| B | `hot_path` | `stage05.2_hot_path_attemptNN` | 消除 Python 重复工作 |
| C | `artifact_streaming` | `stage05.2_artifact_streaming_attemptNN` | 实现 v2 流式分片证据 |
| D | `job_parallel` | `stage05.2_job_parallel_attemptNN` | 选择任务级 worker 数 |
| E | `native_kernels` | `stage05.2_native_kernels_attemptNN` | 迁移 profiling 确认的 CPU 热点 |
| F | `accelerator_pilot` | `stage05.2_accelerator_pilot_attemptNN` | 条件式 GPU/Metal/MPS 决策 |
| G | `benchmark` | `stage05.2_benchmark_attemptNN` | 执行 pilot 和分层正式实验 |

`attemptNN`/`rerunNN` 只表示唯一运行身份，不表示代码版本。失败或中断后使用新的
运行身份，原 shard 不得拼入新运行；但 sealed raw evidence（已封存原始证据）不在
仓库工作区无限累积，而是在 checksum（校验和）验证后移入配置的外部归档。
F 可以以 `GPU_NOT_JUSTIFIED` 通过；A--E 和 G 不得跳过。

current chain（当前证据链）由 signed manifest（签名清单）、prerequisite identity
（先决身份）和 `experiments/registries/stage05.2_retention_registry.csv` 共同确定。
政策文档不得硬编码某次 attempt 为永久 current。producer/storage/native/config
语义变化仍从受影响的最早 gate 重跑；reviewer-only 修复可以复用同一 raw，但每个
review generation 继续保持内容寻址和不可变。

归档后的 run 以 run label 交给 `resolve_retained_run`；该接口根据 registry 中的
archive alias 和本机 ignored storage-root locator 定位目录，重新验证文件数、字节数与
tree SHA-256 后，才把普通 `Path` 交给现有 prerequisite verifier 或 reviewer。仍标记为
active 的目录必须先确认 producer/reviewer 已停止并完成稳定 audit，不能边写边迁移。
registry 以 run label 增量合并，已存在 identity 只有完全一致时才幂等；任何 checksum、
bytes、component 或 prerequisite 冲突立即失败，不能用新一轮 archive 覆盖历史行。

## 统一测量合同

性能 gate 的固定 scope 为 `c101C5`、`c101_21`、`r101_21`、`rc101_21` × seeds `2014/2015/2016`。同一 comparison pair（对照对）必须固定：

- commit、配置、实例 hash、seed、warm start 和 operator surface；
- fixed-work exact-call/iteration budget、deadline semantics 和 cache limits；
- worker affinity（若可用）、环境变量、并发后台负载记录和电源模式；
- validator、objective、candidate ordering 和 failure policy。

每个 run 至少输出：

- `solver_seconds`、`artifact_persistence_seconds`、`end_to_end_seconds`；
- parsing/initialisation/screening/cache/exact/operator/review/manifest 分阶段耗时；
- started/completed exact calls、effective iterations、batch launches、route count per launch 和 median batch occupancy；
- 每 operator calls、exact work、accepted/rejected、累计时间和 p50/p95 cost；
- active cores、worker count、peak/aggregate RSS、写入 rows/bytes、compression time；
- objective tuple、validator result、candidate/cache/event reconciliation 和 failure status。

fixed-work 是性能因果结论的主轴；wall-clock 用于判断实际吞吐和 anytime result（任意时刻结果）。两者均报告，不能互相替代。

## 严格性能门槛

对 B、E 和任何拟替换正式 backend 的实现，使用相对于“声明的直接前一 accepted configuration”的 paired comparison（配对比较）：

1. fixed-work objective、validator、started/completed exact-call ordering、candidate decision、cache lifecycle 和 critical events 完全一致；
2. 全部 100-customer 配对的 aggregate paired median end-to-end time（总体配对中位端到端时间）至少降低 15%；
3. C、R、RC 任一 family 的 family median 不得回退超过 3%；
4. 逐实例异常值、最差 pair 和置信区间完整报告，不能只给总中位数；
5. 未通过即 `NOT_READY`，不得改变 scope、减少 seed、降低门槛或用 wall-clock objective 差异掩盖固定工作量退化。

## A. Performance baseline

以单 worker、当前 `cpu_batch`、`artifact-storage-v1` 运行固定性能 scope。除统一测量 instrumentation（测量插桩）外不改变算法。baseline review 必须证明 instrumentation on/off 的 fixed-work 语义一致，并形成后续各 gate 使用的 immutable comparison table（不可变对照表）。

## B. Python hot-path reduction

按 profiling 排序实施，每项单独形成 paired evidence：

1. 缓存 `Instance` name lookup、depot/customers/stations 和 distance matrix；
2. 把 eager `setdefault`/等价路径改成只在 cache miss 时构建 propagation snapshot；
3. ejection chain 只对 changed routes 执行 screening/exact evaluation；
4. 仅缓存可证明安全、key 完整、生命周期可重放的 screening 结果；
5. 为 operator 记录并执行明确的 time/exact-call budget，预算耗尽作为可见事件。

B 的最终组合必须通过严格性能门槛，单项收益和组合收益都要保留，避免把无效改动藏在总体结果中。

## C. Artifact storage v2

按 `docs/experiment_artifact_storage.md` 实现：65,536-row row groups、最多缓存两个 row groups、worker-owned `(instance, seed)` shards、shard manifest/checksum、parent-only control finalisation、v1/v2/legacy reader compatibility（兼容性）。

除语义 gate 外，C 还必须满足：

- artifact persistence ≤ end-to-end 的 36%；
- peak RSS ≤ A 中 v1 baseline 的 50%；
- timeout、byte-budget、writer error 和 worker failure 均保留 partial shard 并 fail fast；
- reviewer 不读取全量 events 到一个 Python list，不依赖 worker completion order。

当前 C 实现保持 v2 policy，并使用 `screening_decisions_v3`
definitions/occurrences 分表、typed buffers、bounded streaming merge 和跨
v1/旧v2/v3/legacy reader。任何 C 运行必须通过 36 axes、fixed-work semantic
equality 和 36% persistence gate；旧 physical schema 或失败 remediation（补救）
关系只记录在 manifest、retention registry 和 change log 中。

## D. Job-level parallelism

并行独立 `(instance, seed)` shard，依次评估 1、2、4 workers。禁止复用“单候选内部四进程 exact-call”作为正式方案，也禁止 worker error 后转串行。

选择规则：

- 2 workers：相对 1 worker speedup ≥ 1.5×，aggregate RSS ≤ 12 GiB；
- 4 workers：仅在 speedup ≥ 2.5× 且 aggregate RSS ≤ 12 GiB 时选择；
- 4 workers 未过而 2 workers 通过：Formal 固定 2 workers；
- 2 workers 未过：D 为 `NOT_READY`，先定位调度、写盘、内存或 oversubscription（过度订阅）根因。

`RunResourceSummary` v2 的 `run_wall_seconds` 精确定义为从 task scheduling
（任务调度）开始，经过全部 shard completion（分片完成）和 parent adoption，直到
per-run control preparation（逐运行控制工件准备）完成；preflight 和 summary 自身的
manifest finalisation 不混入该并行阶段 speedup。它同时记录 parent/descendant PID、
真实 shard owner PID、50 ms process-tree samples、aggregate RSS 和 active cores。
reviewer 对 1/2/4-worker 三个 bundle 分别要求 exact 36-axis raw/solution/trace replay，
并独立核对当前 accepted C identity、instance/config/native-extension checksum、Python/package/
machine identity、affinity、non-secret performance environment variables、background
load 和 power mode。

每次 D selection 必须使用三个新的唯一运行身份分别执行 1/2/4 workers。reviewer
从 cumulative per-PID CPU samples、真实 worker ownership、完整 shard identity 与
36-axis replay 重算 speedup/RSS 后选择 worker count，不得沿用或硬编码历史结论。

## E. Native CPU kernels

只迁移 profiling 已证明占主导的路径：screening、propagation snapshot、distance lookup、exact label expansion/dominance/heap。数据边界使用 contiguous integer/float arrays；C++ 核心计算可释放 GIL，但进入 Python callback、异常构造或对象访问前必须重新持有 GIL。

每个 kernel 先做 frozen fixture、small brute-force（小规模暴力枚举）和 fixed-work
differential test（差分测试），再进入完整性能 scope。E 的组合实现只绑定当前
accepted D selection，并重新执行 Python/native equality、15%/3% performance、36%
persistence、每 worker 4,357,382,144-byte RSS、process-tree 12-GiB RSS 和
zero-fallback gate。旧 E 失败原因进入 change log，不进入长期政策正文。

## F. Conditional accelerator pilot

只有 E 后 median route batch occupancy ≥ 32 才启动。否则直接发布带证据的 `GPU_NOT_JUSTIFIED`。pilot 必须记录 packing、host-to-device、kernel、device-to-host、synchronisation 和 total end-to-end time。

GPU/Metal/MPS promotion 同时要求：fixed-work 语义一致；相对 E selected native CPU，全部 100-customer 配对的总体端到端中位时间至少降低 15%；C、R、RC 任一 family 的 family median 回退不超过 3%。任一条件失败即保留 native CPU。正式 runner 不得设置隐式 CPU fallback。

F 只接受当前 E raw/review identity，并独立重算 occupancy；median <32 才允许
decision-only `GPU_NOT_JUSTIFIED`，median >=32 则必须执行 registered helper 并
通过语义相等与 15%/3% 门槛。缺失 helper、CPU fallback 或缺少经过审查的 G
campaign adapter 均为 `NOT_READY`。

## G. Pipeline pilot 与正式分层预算

### Pipeline pilot

Pilot 使用固定 Stage 0 representative set（代表集）执行 `12 instances × 3 seeds`
与一个 30 秒 axis。它必须完整走过当前 F gate 实际选择的 backend/worker、真实 failure
recovery、resource sampling、bounded raw replay、1/5/10/30-second anytime、所有
archive root 和 interrupted publication dry run。所有 36 bundles 完整且独立 review
报告 `READY_FOR_STAGE052_FORMAL_BENCHMARK` 后才开放 Formal。

Campaign active writes 与归档目标只通过 ignored root locator（忽略的根目录定位器）
中的 `wsl_staging` 和 `d_archive` aliases 解析。绝对路径不得写入 tracked artifact。
next-fit partitioning 的 target/hard cap 为 24/32 GiB，单 shard hard cap 2 GiB，并
持续满足 locator 声明的容量 reserve。

producer、retention、performance reviewer 与 campaign reviewer 必须调用同一个
cross-platform `probe_volume_identity`：WSL 用 `findmnt`，DrvFS 额外绑定 Windows
NVMe identity，macOS 才使用 `diskutil`。reviewer 不得复制或硬编码单一平台 probe。
rolling-capacity replay 必须从重建且与 campaign identity 一致的 canonical
`BenchmarkCampaignConfig` 读取 reserve：WSL active/future workspace 为
`50 + 32 = 82 GiB`，final WSL safety 为 `50 GiB`，D archive internal safety 为
`50 GiB`。reviewer 不得另设 magic constants（魔法常量）或降低 producer 门槛。

### Formal budget matrix

| Scope | Seeds | 30 s | 60 s | 300 s |
| --- | ---: | ---: | ---: | ---: |
| 36 个 5/10/15-customer 实例 | 10 | 必做 | 不运行 | 不运行 |
| 56 个 100-customer 实例 | 10 | 必做 | 必做 | 必做 |

该矩阵共 2,040 个 solver runs（求解运行），声明的 solver time budget 总计 229,200 秒，即 63 小时 40 分钟；该数值不含启动、写盘、review 和失败重跑开销。正式排期必须再用 D 阶段选定的 worker 数和实测 persistence overhead（持久化开销）估算 wall time，不能把理论时间预算当作实际完工时间。

anytime checkpoints 为预算范围内的 `1/5/10/30/60/120/300 s`。small instances 在 30 秒或 iteration limit 后结束，不为矩阵对称性浪费 60/300 秒。随机 seeds 在 Formal 前冻结并写入 config/manifest；不得按结果更换。

## Retention 与变更日志

- `python -m evrptw.stage052_retention audit` 只读枚举 Stage 5.2 运行，记录状态、
  completeness、source commit、prerequisite run labels、文件数、字节数和完整 tree
  SHA-256，并生成带 sidecar 的 inventory。
- `python -m evrptw.stage052_retention archive` 必须显式绑定 inventory SHA-256。
  同盘目标使用原子迁移；跨盘先写目标卷隐藏临时目录，复验后在目标卷原子落位，再
  清理 source。source 漂移、目标冲突或校验失败均保留 source 并 fail fast；中断留下
  的临时副本可在源完整时安全重建并重试。
- 完整 raw 保存在 `d_archive/stage05.2/history/<run_label>/`。仓库只跟踪
  `experiments/registries/stage05.2_retention_registry.csv`、最终科学汇总和
  `docs/stage052_change_log.md`；registry 不记录本机绝对路径。
- change log 按时间追加原因、修改范围、行为与证据影响、验证结果、失效运行和新运行
  identity。后续改进直接进入当前 Stage 5.2 实现，不复制新版本目录或模块。

## Review 与发布

independent reviewer 必须从 raw shards 重算：exact scope identity、validator/objective、event/cache/exact-call、deadline/budget、worker/shard completeness、resource limits、严格性能 gate、anytime 汇总和模型兼容性。BKS compatibility 仍为 `False`，因此不创建 gap 列。

`deadline_boundary` 按 lane（搜索通道）生效：同一 lane 在 boundary 后不得再启动
exact work、写 cache 或接受 candidate；同时所有 exact completion 和 accepted
candidate 仍不得越过 axis 的总 wall-clock budget。Stage 2.3 明确保留的最后 0.1 秒
constraint-lane slice 不得被 legacy/quality lane 的提前 boundary 错误截断。

已进入 retention registry 的归档目录是 immutable review input（不可变审查输入），可
作为 comparison/prerequisite 读取，但不能再作为写入 review generation 的 `raw_dir`。
需要产生新 review generation 时，目标 raw 必须仍位于 active root；封存完成后再归档。

### Reviewer 内存与后台运行合同

storage semantic replay（存储语义重放）必须在读取 canonical record（规范记录）时
直接更新每个 axis 的 SHA-256；禁止构建完整的 per-axis event list（逐轴事件列表）。
多个 raw bundle 严格按输入顺序处理，每个 bundle 使用一个新的 `spawn` 子进程，父
进程只接收小型 digest map（摘要映射）。只有需要 exact equality 的 fixed-work axis
digest 不一致时才执行字段级重放；wall-clock axis 的预期 trajectory 差异只写一条
aggregate digest row（聚合摘要行），不得展开为数十 GiB 的逐事件差异。字段级重放使用
ext4 上的临时 SQLite spool；spool 每条 canonical record 只保存一行压缩
payload 和 digest，比较时才展开字段，禁止按每个 field 写一行造成磁盘与 cgroup
page-cache 放大。spool 只保存 comparison bundle（对照证据）；candidate bundle（候选
证据）边读边按主键 point lookup（点查询）和比较，不得同时保存两套完整 payload。
left-only 记录由每个 axis 的最终 ordinal tail（序号尾部）确定，禁止用大规模 DELETE
制造 SQLite 脏页。字段差异先流式写入每个 axis 的有序临时 fragment（片段），再按
identity 顺序拼接；SQLite、fragment、最终 CSV 和发布副本必须周期性 `fsync` 并使用
`POSIX_FADV_DONTNEED` 释放已落盘 page cache。SQLite 建库必须按记录窗口提交并释放
脏页；fragment 使用跨全部 axis 的全局字节窗口，left-only tail 复用相同窗口。raw
manifest hashing 和 Parquet/JSONL iterator（迭代器）在文件生命周期结束时也必须释放
source page cache（源页缓存）；禁止携带 BLOB 的 temp sort。
`semantic_mismatches.csv` 必须直接流式写临时文件，并通过流式 hash/copy 发布，禁止
在 `StringIO` 或 `bytes` 中累积完整 mismatch 输出。成功或失败后都删除临时数据。

Windows/WSL2 formal reviewer 固定通过
`python -m evrptw.stage052_review_service launch` 启动 transient
`systemd --user` service。service 固定使用 `MemoryHigh=5G`、`MemoryMax=6G`、
`MemorySwapMax=2G`、`KillMode=control-group`、`Restart=no` 和 `OOMPolicy=stop`；
reviewer 自身在 aggregate RSS 达到 5.5 GiB 时先行失败。每次运行在
`/home/oneblaze/stage052-review-logs/<run-label>/<UTC timestamp>/` 保留
`progress.jsonl`、`service.log` 和 `review_execution.json`。这些是 operational
evidence（运行证据），不写入 immutable raw manifest，也不改变 review gate。
launcher 必须在启动前解析并冻结 `nvidia-smi`、`powershell.exe` 和 `wsl.exe`
所在目录，将完整 service `PATH` 显式传入 systemd 并记录到
`review_execution.json`；不得依赖 Codex 或交互式 shell 偶然继承的 Windows
interop `PATH`。缺少任一正式运行工具时必须在 service 启动前 fail fast。
producer runtime identity 必须由 raw-bound（原始证据绑定）的 producer venv
独立重放 wheel、Python、native extension（原生扩展）、dependency（依赖）和
machine identity（机器身份）；不得用新的 reviewer wheel 冒充 producer wheel。
review-only `.wslconfig` memory cap（仅审查内存上限）作为 execution receipt
中的 operational evidence 单独审计，不得改写历史 producer identity，也不得掩盖
CPU、GPU、Windows、WSL、mount 或 NVMe identity 的变化。
若已发布的 review 因 reviewer/runtime 缺陷为 `NOT_READY`，后续显式 service
运行必须先把旧 manifest 和全部 review files 原样归档到
`review/history/<manifest-sha256>/`，再把 hash 追加到新 manifest 的
`review_retry_history_sha256`。失败 review 不得伪装成 accepted lineage（已接受谱系），
但也不得阻止修复后的新 generation；每条 retry archive 都必须在 prerequisite
校验时重新验证 manifest、raw binding 和文件哈希。
service 还必须配置 `ExecStopPost` 外部收尾器；即使 `MemoryMax` 直接终止 reviewer
与 supervisor，收尾器仍从 systemd 的 `SERVICE_RESULT/EXIT_CODE/EXIT_STATUS` 封存
失败回执，并重新计算 raw manifest、service/progress log hash 以及读取 cgroup
`MemoryPeak/MemorySwapPeak`。回执只有 `finalized=true` 才是最终状态。
任一 cgroup peak 无法读取时，回执必须标记
`resource_accounting_unavailable` 并失败，禁止以 `0` 静默代替。

固定操作入口为：

- `... stage052_review_service status --unit <unit>`：查看状态；
- `... stage052_review_service follow --unit <unit>`：跟踪 journal；
- `... stage052_review_service stop --unit <unit>`：停止完整 control group；
- `... stage052_review_service receipt --log-dir <dir>`：读取最终执行回执。

service 只允许 `evrptw.experiments.stage052_performance_review` 与
`evrptw.experiments.stage052_campaign_review` 两个 isolated module（隔离模块），并分别
验证 raw、comparison/prerequisite、scope 和 run label。两者都必须使用外部 progress
log 与 5.5-GiB 内部 process-tree guard。正式启动必须传入 reviewer wheel 对应的完整
`--reviewer-revision`、frozen
non-editable wheel、clean working directory、raw manifest 和完整 reviewer command。
reviewer command 必须包含该 reviewer 类型要求的 raw、comparison/prerequisite 与
scope identity，并固定通过该 wheel 所在 venv 的 Python 使用 `-I -m` 启动 allowlisted
module；
launcher 会核对 installed distribution 的 `direct_url.json`、module path、wheel
逐文件内容、wheel 中全部 tracked Python/native/build input 与 clean revision 的绑定、
以及 no-cache clean rebuild（无缓存干净重建）的逐 member 一致性；由
clean build source 生成的 `.whl.reviewer-provenance.json`、wheel
path/hash、raw run label 以及 clean producer snapshot，然后自动追加外部 progress log
和 5.5-GiB 内部上限。Codex 只启动和
轮询 service，不持有 reviewer 生命周期。超限、worker failure 或 service interruption
不得自动重试；只有显式的新 review generation 才能再次运行。
`--max-aggregate-rss-gib` 在 formal launch 中固定为 5.5，不得放宽。launcher 只接受
`ArtifactReader` 解析出的 canonical signed raw manifest。科学 reviewer 写出的 READY 在
`ExecStopPost` 完成前只是 provisional（暂定）；只有成功 receipt 已绑定当前
review-manifest SHA-256、raw manifest 未变化且 cgroup peaks 可用时，后续 prerequisite
verifier 才能消费该 READY。

必须发布：

- `experiments/registries/stage05.2_artifact_registry.csv`；
- `experiments/registries/stage05.2_retention_registry.csv`；
- `experiments/manifests/stage05.2_performance_benchmark_artifact_manifest.json`；
- per-run、per-family、per-budget、anytime、resource 和 persistence summaries；
- performance gate、GPU decision、failure analysis 和 review report。

只有所有 required components 与 Formal bundles 通过时，reviewer 才能报告 `READY_FOR_STAGE05_3`。

## Stage 5.3 及以后

- Stage 5.3 固定使用 Stage 5.2 selected backend、worker count 和 artifact-storage-v2；fixed-work 为消融主轴，wall-clock 为实际收益轴；exact-charging-removal 是唯一不调用 exact backend 的 ablation。
- Stage 6 pricing 可采用自己的算法后端，但证据继承 streaming/shard/job-parallel contract；使用 ALNS incumbent 或 route evaluation 时继承 selected ALNS backend。
- Stage 7 的 solution conversion 和 validator replay 保持 ordered backend semantics（有序后端语义）。
- Stage 8 每种新 charging model 必须重新通过 fixed-work replacement、严格 15%/3% 性能和 v2 evidence gate 才能进入 Formal。
- 后续出现新的 scalar hotspot 时，返回本流程的 profiling → native CPU → conditional accelerator gate；不得仅凭低 CPU 占用或低功耗直接选择 GPU。
