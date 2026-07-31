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
（先决身份）和 `e_archive/.storage-governance` 中最新 verified v2 registry
generation（已验证注册表代次）共同确定。tracked
`experiments/registries/stage05.2_retention_registry.csv` 仅为不可变 v1 compatibility
fallback（兼容回退），不是共同的 current-chain truth source（当前证据链事实源）。
政策文档不得硬编码某次 attempt 为永久 current。producer/storage/native/config
语义变化仍从受影响的最早 gate 重跑；reviewer-only 修复可以复用同一 raw，但每个
review generation 继续保持内容寻址和不可变。

Formal atomic publisher（正式原子发布器）必须同时重新打开当前 Formal raw 与其
accepted Pilot prerequisite（已接受先决 Pilot）。默认 live evidence root（活动证据
根目录）仍为 `<repository>/results`；使用独立 ext4 active root 时必须通过
`--active-results-root` 显式传入，并要求 Formal 与 Pilot 都是该根目录的直接子目录。
publisher 的 resolved path identity（解析后路径身份）拒绝 symlink（符号链接）绕过；
操作流程不得用 bind-mount path masquerading（绑定挂载路径伪装）规避显式参数，也
不得把 retention registry 已登记的 immutable archive（不可变归档）作为新的
publication source（发布源）。

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

1. fixed-work objective、validator、started/completed exact-call ordering、candidate
   decision、solution、acceptance/global-best 与 deadline critical events 完全一致；
   safe deduplication/batching（安全去重／批处理）可改变 screening/negative-cache
   diagnostic lookup count，但每个实现的 exact/cache lifecycle、transaction hash 和
   zero-fallback 必须由独立 raw replay 分别通过；
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

这里的 worker count 是同时运行的最大并发数，不是整个 batch 生命周期内 PID 的总数。
每个 shard 必须由新的 `spawn` worker process 执行，executor 固定
`max_tasks_per_child=1`，不得跨 shard 复用 Python/native allocator state。batch
metadata 必须记录 `worker_process_lifecycle=one_shard_per_spawned_process`；
每个 worker 在 shard 计时开始前还必须执行仅内存的 Arrow/Zstandard warmup，
并记录 `worker_runtime_warmup=in_memory_arrow_zstd1`。warmup 不得写 raw
artifact，也不得替代、删除、聚合或跳过任何审计事件。
`candidate_state.timestamp_seconds` 包含已单独归因的 interleaved persistence；
deadline replay 必须以 lane-local deadline boundary 为权威，不得再把该 wall
timestamp 直接与 declared solver seconds 比较。boundary 后 acceptance、
exact completion crossing 和 boundary 后 cache store 仍必须 fail fast。
Native counter replay 必须显式对账 deadline interruption：batch 已声明 launch
后，deadline checkpoint 可在 native invocation 前终止，因此
`batch_launches - native_invocations` 只能小于等于已重放的 interrupted call
数量；同时必须满足 `started = completed + interrupted`，且所有 native/protocol
fallback counter 为零。
Single-pass replay 同时接受显式 `deadline_boundary` event，以及
`route_evaluation.status=interrupted_deadline` 作为权威 deadline evidence。
后者必须绑定当前 lane，并对其后的 exact work、cache store 和 acceptance
应用相同的 fail-fast 规则。
reviewer 要求所有 shard owner 都属于 50-ms process-tree samples，并至少观察到冻结的
并发 worker 数。worker recycle（工作进程回收）不得改变 shard 顺序、solver、backend、
objective、validator、event schema 或失败语义。
Producer 按冻结的 worker 数把 shards 切成连续 waves（波次）；每一 wave 的
spawn pool 必须在下一 wave 创建前完整 shutdown。每个 worker 仍只处理一个 shard，
因此每个 shard 保持唯一 PID，同时避免旧 worker 退出与新 worker 初始化重叠而把
campaign 自身负载重复计入 `load1`。

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
四步 native ablation 的 promotion timing 必须全部来自同一个
`in_memory_measurement_trace_no_stream_sink_v1` instrumentation envelope；正式
fixed-work axis 的 asynchronous streaming timing 只用于 campaign/resource/
persistence evidence，不得只替换其中一个 ablation mode 的计时。每个 v4 ablation
axis 必须显式记录该 envelope，reviewer 对缺失或不同值 fail fast。
E 相对 D 的 performance promotion（性能晋级）必须在 24-axis core semantic
replay 已通过后，把 paired fixed-work timing（配对固定工作量计时）绑定到该
replayed core digest；不得继续使用包含 transaction/screening/cache diagnostics
（事务／筛选／缓存诊断）的 full-storage digest 作为跨实现语义身份。full-storage
digest 仍用于同一实现的存储回放与 aggregate diagnostic mismatch（聚合诊断差异）。

## F. Conditional accelerator pilot

只有 E 后 median native candidate screening occupancy（原生候选筛选中位占用度）
≥ 32 才启动。该值必须从 fixed-work raw 的 candidate transaction statistics
独立重算；exact backend launch occupancy 不能代替。否则直接发布带证据的
`GPU_NOT_JUSTIFIED`。pilot 必须记录 packing、host-to-device、kernel、
device-to-host、synchronisation 和 total end-to-end time。

CUDA promotion 同时要求：fixed-work 语义一致；相对 E selected native CPU，
全部 100-customer 配对的总体端到端中位时间至少降低 15%；C、R、RC 任一
family 的 family median 回退不超过 3%。完整但未达 promotion gate 的 pilot
发布 `NATIVE_CPU_RETAINED`。正式 runner 不得设置隐式 CPU fallback。

F 只接受当前 E raw/review identity，并独立重算 occupancy；median <32 才允许
decision-only `GPU_NOT_JUSTIFIED`，median >=32 则必须执行 registered helper 并
通过语义相等与 15%/3% 门槛。缺失 helper、CPU fallback 或缺少经过审查的 G
campaign adapter 均为 `NOT_READY`。
E 保持 accepted D selection 的 CPU worker width（当前为 4）；F 的 metadata 与
campaign adapter 固定使用 6 workers，为后续 G 的用户锁定配置提供同一执行合同。
这两个 worker count 属于相邻阶段的不同角色，不得要求数值相等。

## G. Pipeline pilot 与正式分层预算

### Pipeline pilot

Pilot 使用固定 Stage 0 representative set（代表集）执行 `12 instances × 3 seeds`
与一个 30 秒 axis。它必须完整走过当前 F gate 实际选择的 backend/worker、真实 failure
recovery、resource sampling、bounded raw replay、1/5/10/30-second anytime、所有
archive root 和 interrupted publication dry run。所有 36 bundles 完整且独立 review
报告 `READY_FOR_STAGE052_FORMAL_BENCHMARK` 后才开放 Formal。

Campaign active writes 与归档目标只通过 ignored root locator（忽略的根目录定位器）
中的 `wsl_staging`、legacy `d_archive` 与长期 `e_archive` aliases 解析。绝对路径不得
写入 tracked artifact。
next-fit partitioning 的 target/hard cap 为 24/32 GiB，单 shard hard cap 2 GiB，并
持续满足 locator 声明的容量 reserve。

campaign 启动前与 per-batch handoff preflight（批次交接预检）都保留两段连续
30 秒窗口。AC/battery、low-power mode、`load1`、unrelated process、CPU/GPU
型号、系统版本、温度以及磁盘/设备信息全部作为 non-blocking telemetry（非阻断遥测）
保留，异常退出也必须写入 partial raw evidence（部分原始证据）。硬 gate 只验证：
逻辑 CPU 数不少于已校准 workers、可用内存与空间满足冻结合同、backend/Python ABI/
native extension 与 source/wheel/config/input/schema hashes 一致，以及文件系统支持
所需 fsync 和 atomic transfer（原子传输）。

若 G 自身的 campaign runner/reviewer 出现缺陷，可以消费已接受的完整 Attempt72
Pilot evidence（其科学选择继续追溯到已接受 F evidence），但必须同时满足：legacy
revision 通过不可变 provenance map 解析到公开历史，当前 revision 是固定 publication
bridge 的 Git descendant（后继）；bridge 后全部 changed paths 均落在显式 G runner/artifact-persistence
adapter/reviewer/test/documentation allowlist；
Python、dependency、source、native extension、scientific configuration、instance、
backend 与 exact backend 的稳定选择 hash 完全一致。worker/RSS/Parquet tuning 由
新的 signed resource contract 单独冻结，machine/device 仅为 telemetry。producer 与
independent reviewer 各自执行该检查。solver、objective、科学配置或 allowlist 外
source path 变化都 fail fast，并要求新的 prerequisite。

已接受 D/F evidence 的历史 resource gate 不追溯重写。当前 G Benchmark 校准在
Attempt73 的只读大型 shard 与固定高内存 scope 上比较 4/5/6 workers，并按用户要求
增加 8-worker stress probe（压力探针）。6 workers 相对 4 workers 达到既定吞吐门槛，
不使用 swap/fallback 且语义摘要一致；8 workers 仅为排除性测试，吞吐显著低于 6，
因此 replacement Pilot/Formal 固定 6 workers。resource gate 使用实际 per-worker
RSS 与隔离 cgroup v2 的 `memory.current`/`memory.peak`、机器可用能力和明确的
operating headroom（运行余量）；process-tree RSS 总和因会重复计算 spawned workers
共享映射，只保留为 compatibility telemetry（兼容遥测），不再作为 aggregate hard
gate（聚合硬门槛），也不再把可用内存的任意固定百分比作为 publication gate
（发布门槛）。最终 worker 数、per-worker/cgroup limits、Parquet row group
16,384/65,536/262,144 与 queue depth 1/2
写入 signed resource contract（签名资源合同），Formal 原样继承。通过独立
Formal memory probe 的低内存 row-group/queue-depth pair 必须作为显式
locked configuration（锁定配置）传入后续 full resource calibration；full
calibration 不得忽略该 pair 后按 persistence-only timing（仅持久化耗时）自动重选。
Independent
reviewer 单独校准 1/2/4 workers，并由 parent baseline 与 per-child p99 RSS 推导
`MemoryHigh`、内部 guard 和 `MemoryMax`；swap 固定为 0，资源上限必须容纳已选并发
且不得触发持续 throttling（限流）。

若完整 Formal 在零 readiness geometry（就绪几何）贡献的 partial batch 中仅因已签名
aggregate/per-worker hard limit fail fast，replacement Formal 可以使用新的
signed calibration report 重新冻结 resource envelope（资源封套）。v2 report
仅用于读取 Attempt99 历史；新 v3 report 必须绑定 rerun02/batch0008，并从隔离
systemd service 的 cgroup v2 实际内存重测 aggregate peak。失败 predecessor 的
process-tree RSS 总和只作为不可变 failure provenance，不可作为物理内存 floor。
memory
capacity/peaks/limits 与 calibration/semantic digests 可以重测；worker 拓扑及
independent scientific execution selection lock 不变。row-group/queue-depth 是
physical persistence topology（物理持久化拓扑），只能由新 signed Formal memory
probe 的 exact pair 替换，并由 full calibration 显式锁定，不能成为运行时 fallback。
report 必须绑定失败 run/batch/resource summary、clean calibration revision、
exact unique `{4,5,6}` worker identity set 的 cross-worker semantic equality、
exact replacement-contract digest、精确 20% headroom、零
swap/fallback/resource-limit failure 与
`campaign_geometry_contribution=0`；producer 和
independent reviewer 必须通过同一 loader 重算。Formal memory probe 必须在独立
`systemd --user` service 中运行，保留其自身可验证的 semantic digest、cgroup path、
`memory.peak` 与 `memory.swap.peak`，并且 cgroup aggregate/per-worker peak 不得超过
replacement contract 的对应 selected peak。`/init.scope`、共享 cgroup、缺失
memory controller 文件或任何 aggregate RSS fallback 均 fail fast。任何缺失
report、checksum 不一致、worker identity
缺失或重复、拓扑变化、memory floor 降低或非零 geometry 都继续拒绝，失败 label
不续跑、不导入 shard。

Formal calibration 与 benchmark producer 的 interpreter 必须以
`PYTHONMALLOC=malloc` 启动。allocator identity（分配器身份）写入 signed producer
resource contract，并在 benchmark admission 重放；缺少该字段的 legacy contract 只能
解析为 `default`，不能通过新的 Formal calibration gate。async Parquet writer 每完成
8 个 batches，就在仍持有 measured writer turn 时丢弃刚持久化 batch 的引用并执行 libc
`malloc_trim`。每个 axis 固定记录 release ordinal、allocator、trim 可用性/结果及前后
RSS。independent calibration reviewer 从 `submitted_batches` 重算 release 数量与 exact
ordinal sequence，拒绝缺失或伪造的 release telemetry。该协议只约束可回收 Python
pages；6 workers、16,384/1、swap 0、fallback 0 与 20% headroom 不变。

sealed source snapshot 不能只做 Git clone：Schneider benchmark 是 ignored input，
必须在启动前以 snapshot-local ordinary files（快照内普通文件）实体化，并核对 exact
probe instance 非空。Formal memory probe 在创建 output root 和提交 worker work
之前执行该门槛；缺失输入不得进入内存测量。若 transient unit 已使用某 label 启动，
即使此门槛立即失败，该 label 仍作为不可变 failure evidence 保留并由新 label 替代。
所有 calibration CLI 必须显式传入 clean ext4 checkout 的
`--repository-root`；service working directory 不再承担隐式 source binding。
缺少该参数必须在 argparse 阶段 fail fast，避免从 `/home/oneblaze` 等非 worktree
目录启动后才消费 evidence label。相对 `--config` 必须基于该显式 repository root
解析，禁止基于 service cwd 解析。

producer 的 screening-definition identity state（筛选定义身份状态）固定使用
`native_bounded_digest`（原生有界摘要）backend，保存完整 SHA-256 collision token
（碰撞令牌）而不复制 JSON payload。hard limit 固定为 2,097,152 unique definitions；
`register_many` 必须在任何写入前完成批内去重、已有身份碰撞检查和容量检查，使碰撞或
overflow（越界）整批 fail fast 且不留下前缀写入。producer 禁止 SQLite spill、
scratch directory 和 Python fallback。可丢弃的 native definition-key memo（原生定义键
备忘缓存）与 collision state 分离，固定为 8,192 entries，即一个 live screening
transaction（活动筛选事务）；FIFO eviction（先进先出
淘汰）只触发 full-SHA-256 identity 的安全重算，不删除全 shard collision proof。
route-ID resolution（路线身份解析）以及 route-evaluation/cache-event sparse extras
JSON（路线评估／缓存事件稀疏附加字段）属于可确定性重算的 Python memo，统一固定为
8,192 entries 和 FIFO safe recomputation。typed negative-evidence route guard
同样固定为 8,192 个完整 `(token, signature)`，其淘汰只允许重新计算相同完整签名；
deferred occurrence 与 native collision check 仍必须携带和比较完整 signature，禁止
降为 marker-only identity。淘汰只允许重复 canonical route-key parsing
或 canonical JSON construction；不得删除 disk-backed route identity、完整 digest/payload
collision state 或任何 Parquet row。signed batch metadata 必须记录 v6 store contract；
independent reviewer 必须确认整条
campaign 只有一个相同合同，并证明每个
signed shard descriptor 的 definition row count 均不超过当次 raw 中签入的 hard
limit。review/read 的 payload-retaining compatibility store（载荷保留兼容存储）与
exact-route identity store 保持各自独立的 bounded SQLite 路径。route dictionary
identity 与 exact unique-route identity 各自最多保留一个 65,536-row Parquet group
的 hot state（热状态），超过后将完整 digest/payload 原子写入 shard-local SQLite；
这不是 producer definition fallback，也不得削弱 duplicate/collision proof。只有同时
通过 native bound、零 fallback、36% persistence gate 与 shard cleanup gate，才允许
新的 Formal。

Candidate transaction（候选事务）的 safe-rejection caches（安全拒绝缓存）同样不得
无界增长。scalar screening result（标量筛选结果）使用 65,536-entry solve-local LRU；
Python sequence mapping 与 packed native ABI state 使用 65,536-entry generation
cache（分代缓存）。容量将溢出时，当前 candidate transaction 必须原子替换两侧分代；
worker、deadline、integrity 或 commit 失败必须恢复旧分代。淘汰只允许导致后续 safe
screening 重算，不得增加 exact call、改变 candidate order 或削弱 collision proof。
bounded LRU 的 negative-hit marker 必须由 collision-free canonical route key 与完整
normalized screening result 稳定派生：等价重算保持 marker。该 marker 仅是 lookup
accelerator；route-key consistency guard 还必须比较完整 compact binary typed result
signature（长度前缀字符串、显式 type tag、原始 IEEE-754 bits），使 63-bit 截断或碰撞
无法隐藏任一证据字段变化并 fail fast，同时避免恢复 object-heavy evidence retention。
deferred native occurrence identity 也必须使用 `(marker, complete signature)` 复合值，
不得在 route-key guard 淘汰后退化为 marker-only cache hit。
不得使用会随对象生命周期变化的地址作为持久 marker。
raw result/transaction statistics 必须签入 capacity、current/peak、stores、evictions
与 rollovers，independent reviewer 对缺失、无界或内部不一致证据 fail fast。

producer、retention、performance reviewer 与 campaign reviewer 必须调用同一个
cross-platform `probe_volume_identity`：WSL 用 `findmnt`，DrvFS 额外绑定 Windows
physical-disk identity（物理磁盘身份），macOS 才使用 `diskutil`。reviewer 不得复制或
硬编码单一平台 probe。
rolling-capacity replay 必须从重建且与 campaign identity 一致的 canonical
`BenchmarkCampaignConfig` 读取 reserve：WSL active/future workspace 为
`50 + 32 = 82 GiB`，final WSL safety 为 `50 GiB`，E archive safety 为
`200 GiB`，D host/VHDX safety 为 `200 GiB`。D archive 的既有 v1 证据保持只读。
reviewer 不得另设 magic constants（魔法常量）或降低 producer 门槛。

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
- legacy `python -m evrptw.stage052_retention` 仅用于 v1 audit/read/resolve；其 archive
  命令不得向 `e_archive` 发布。E 盘新 generation 必须走 experiment retention v2：
  跨盘先写目标卷隐藏 incoming generation，复验、独立 replay 并登记后，historical
  D/WSL source 仍保持原位，直至生成精确删除清单并取得字面 `确认`。
- future benchmark attempt 的 verified batch 可按预先配置执行 same-attempt rolling
  handoff：目标卷完整校验、原子发布后清理该 batch 的 staging source，以维持 32 GiB
  active-workspace cap。该路径不授权清理本轮历史迁移源。
- 新完整 raw generation 保存在
  `e_archive/stage05.2/generations/<run_label>/<generation>/`；既有
  `d_archive/stage05.2/history/<run_label>/` 由 v1 resolver 保持只读可解析。reduced
  duplicate 是 audit-only，不能作为 comparison/prerequisite。registry 不记录本机
  绝对路径。
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
进程只接收小型 digest map（摘要映射）。storage-only replacement（仅存储替换）对
fixed-work canonical digest 要求 exact equality；native candidate transaction
这种会新增 transaction/screening observability 的跨实现比较，改用只包含
solution/objective、candidate-state order、ordered exact-route result 和 deadline
boundary 的 core semantic digest，并由独立 ABI/cache gate 复核被排除的诊断字段。
wall-clock 与跨实现 full-storage digest 的预期差异只写 aggregate digest row（聚合摘要
行），不得展开为数十 GiB 的逐事件差异。确需字段级重放时使用
ext4 上的临时 SQLite spool；spool 每条 canonical record 只保存一行压缩
payload 和 digest，比较时才展开字段，禁止按每个 field 写一行造成磁盘与 cgroup
page-cache 放大。spool 只保存 comparison bundle（对照证据）；candidate bundle（候选
证据）边读边按主键 point lookup（点查询）和比较，不得同时保存两套完整 payload。
left-only 记录由每个 axis 的最终 ordinal tail（序号尾部）确定，禁止用大规模 DELETE
制造 SQLite 脏页。字段差异先流式写入每个 axis 的有序临时 fragment（片段），再按
identity 顺序拼接；SQLite、fragment、最终 CSV 和发布副本必须周期性 `fsync`。SQLite
建库必须按记录窗口提交并释放脏页；fragment 使用跨全部 axis 的全局字节窗口，
left-only tail 复用相同窗口。raw manifest hashing 和 Parquet/JSONL iterator
（迭代器）在文件生命周期结束时也必须处理 source page cache（源页缓存）；禁止携带
BLOB 的 temp sort。原生 Linux 使用 `POSIX_FADV_DONTNEED` 执行 advisory release；
Windows/WSL2
不得调用它，因为当前正式主机已复现该调用后的错误 page-cache read。WSL reviewer
依靠逐 bundle/逐 shard 新进程退出释放进程 RSS，并继续执行 `fsync`、SQLite
`shrink_memory` 与 scratch cleanup；禁止为了模拟 page-cache release 而降低审计范围。
`semantic_mismatches.csv` 必须直接流式写临时文件，并通过流式 hash/copy 发布，禁止
在 `StringIO` 或 `bytes` 中累积完整 mismatch 输出。成功或失败后都删除临时数据。
Benchmark campaign reviewer 对 campaign 内每个 shard 同样使用一个全新的、严格串行的
`spawn` 子进程；不得用长寿命 pool 复用 allocator state。每个 child 必须重读 signed
batch/shard manifest，并在一次 logical event pass 中同时完成 persistence ledger、
cache/exact/deadline transaction 和 global-best/checkpoint replay。父进程只接收有界
per-axis summary、checkpoint、计数与 resource telemetry，不得接收 raw event、Arrow
table、route dictionary 或 screening definition。child failure 不得回退到 parent
重放；scratch 必须在 child 退出前验证清理，parent/child RSS、PID、event count、
single-pass count 和 cleanup state 必须写入外部 progress log。child summary 必须是
JSON-safe（可安全 JSON 序列化）的有界对象并携带 run/batch/shard/instance/seed 身份；
parent 必须逐字段核对。失败路径同样必须写 child PID/peak RSS、parent RSS、耗时和
cleanup state，残留 scratch 在 fail fast 前清除并记录，不得静默重试。

若 sealed raw（已密封原始证据）归档后发生物理 D 盘更换，retrospective review
（追溯复审）只能通过 signed storage migration attestation（签名存储迁移证明）
接受该变化。证明必须同时绑定旧/新 volume identity（卷身份）、旧/新物理磁盘身份、
campaign 与标准 raw manifest SHA-256，以及每个归档 batch 的目录 checksum 和字节数。
仅允许证明中精确声明的归档存储映射发生变化。producer runtime 的 hard contract
（硬合同）、source revision、tracked/allowed-untracked file hashes（文件哈希）、
read-only state（只读状态）、ext4 capability（ext4 能力）、solver、backend 和科学语义
仍须完全一致；mount source、ext4 UUID、source target path 与物理磁盘身份仅作为
telemetry（遥测）及 attested storage mapping（经证明的存储映射），不得进入
publication identity（发布身份）。reviewer source 与 producer source snapshot
必须作为两个独立输入验证，不得用新 reviewer checkout 冒充历史 producer snapshot。
通过迁移证明完成的 campaign review 必须在 review manifest 顶层发布 attestation
及 sidecar SHA-256；后续 Formal producer 只有在显式提供相同证明且 SHA-256 与
accepted Pilot review 完全一致时，才可将该 Pilot 作为 current-chain prerequisite。
若 producer defect 要求在物理迁移后重跑 successor Pilot，该 Pilot 仍以 accepted F
accelerator pilot 作为科学 prerequisite，并额外显式提供
`--storage-migration-evidence-dir` 指向迁移证明声明的历史 G campaign。系统必须重新
验证该 campaign 的全部归档 batch、accepted Pilot review 和 finalized successful
receipt 后，才可仅规范化冻结 selection lock 中的 D 盘身份；不得用历史 G review
替代 F prerequisite，也不得只信任 attestation schema。
successor campaign 的 independent reviewer 必须接收并执行同一
`--storage-migration-evidence-dir` 合同；它先重验历史 migration campaign，再审计
当前 successor raw。禁止把旧 attestation 的 run label 或 batch set 与当前 campaign
强行比较，也禁止仅加载签名 JSON 后跳过历史 batch/review/receipt 复验。

Windows/WSL2 formal reviewer 固定通过
`python -m evrptw.stage052_review_service launch` 启动 transient
`systemd --user` service。review concurrency（审查并发）、`MemoryHigh`、
`MemoryMax`、`MemorySwapMax=0` 与内部 aggregate RSS guard（聚合常驻内存防线）
由 Pilot 的 signed review calibration contract（签名审查校准合同）冻结；合同必须
记录 parent baseline、per-child p99 RSS、候选 1/2/4 workers 的等价性与吞吐，并受
当前可用内存约束。Formal 不得动态改 workers，也不得在 native child 失败时回退到
Python。service 仍固定使用 `KillMode=control-group`、`Restart=no` 和
`OOMPolicy=stop`。每次运行在
`$XDG_STATE_HOME/reproducible-evrptw/stage052-review-logs/<run-label>/<UTC timestamp>/`
（未设置 `XDG_STATE_HOME` 时使用 `~/.local/state`）保留
`progress.jsonl`、`service.log` 和 `review_execution.json`。这些是 operational
evidence（运行证据），不写入 immutable raw manifest，也不改变 review gate。
launcher 必须在启动前解析并冻结 `nvidia-smi`、`powershell.exe` 和 `wsl.exe`
所在目录，将完整 service `PATH` 显式传入 systemd 并记录到
`review_execution.json`；不得依赖 Codex 或交互式 shell 偶然继承的 Windows
interop `PATH`。缺少任一正式运行工具时必须在 service 启动前 fail fast。
producer runtime identity 必须由 raw-bound（原始证据绑定）的 producer venv
独立重放 wheel、Python、native extension（原生扩展）、dependency（依赖）和
machine identity（机器身份）；不得用新的 reviewer wheel 冒充 producer wheel。
同 revision 的 campaign runtime contract（实验运行时合同）只排除
`machine_identity`、mount telemetry（挂载遥测）和本地 absolute paths（绝对路径）；
wheel、Python、native extension、dependency、ABI 与 source revision 的 hash 仍是
hard gate。WSL `memory_bytes`、设备枚举和 snapshot 根目录变化只能进入 telemetry，
不得改变 selection verdict（选择结论）。
review-only `.wslconfig` memory cap（仅审查内存上限）作为 execution receipt
中的 operational evidence 单独审计，不得改写历史 producer identity。CPU、GPU、
Windows、WSL、mount、NVMe 与实时内存变化必须完整记录，但不改变 selection
verdict（选择结论）。
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
和 signed review contract 的内部 process-tree guard。Codex 只启动和
轮询 service，不持有 reviewer 生命周期。超限、worker failure 或 service interruption
不得自动重试；只有显式的新 review generation 才能再次运行。
reviewer 或 pilot publication dry-run 运行时导入的 tracked `tools` modules 必须随
reviewer wheel 一起安装，并由 source attestation（源证明）和 clean rebuild 逐文件
绑定；`python -I` service 不得从 working checkout 临时导入未密封工具。
`source_snapshot` 是 Pilot/Formal 共用的 mandatory review gate（强制审查门槛），
provisional Pilot publication 也必须在 exact gate set 中保留并通过该项。该 hard
contract 锁定 revision、tracked file count、允许的 untracked file hashes、read-only
状态与 ext4 filesystem；mount source、UUID 和 snapshot absolute path 作为 telemetry
保留，不参与 readiness identity。
current-chain performance reviews 继续要求含 `semantic_mismatches.csv` 的三文件
generation；Benchmark campaign prerequisite 使用其完整 content-addressed publication
surface（内容寻址发布面），不得被通用 verifier 错套 performance-only 文件名。
Benchmark review manifest 还必须在顶层发布 `selected_optimization_profile`，并与
`selection_lock` 中的冻结值一致；下一次 campaign loader 不从嵌套字段静默补值。
`--max-aggregate-rss-gib` 在 formal launch 中必须由 signed review contract 的
`process_guard_bytes` 精确推导，不得人工放宽。launcher 只接受
`ArtifactReader` 解析出的 canonical signed raw manifest。科学 reviewer 写出的 READY 在
`ExecStopPost` 完成前只是 provisional（暂定）；只有成功 receipt 已绑定当前
review-manifest SHA-256、raw manifest 未变化且 cgroup peaks 可用时，后续 prerequisite
verifier 才能消费该 READY。

必须发布：

- `experiments/registries/stage05.2_artifact_registry.csv`；
- `e_archive/.storage-governance` 中最新 verified v2 retention registry generation；
- 既有 `experiments/registries/stage05.2_retention_registry.csv` 保持不可变，仅供 v1
  compatibility fallback，不由 v2 publication（发布）改写；
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
