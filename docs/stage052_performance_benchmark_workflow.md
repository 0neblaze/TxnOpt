# Stage 5.2 性能治理与分层 Benchmark 工作流

## 目的与入口

Stage 5.2 先修复已测得的工程瓶颈，再扩大实验规模。它不是直接“多跑 benchmark”，也不以 CPU 占用率或功耗代替端到端性能证据。

唯一入口是：

- `stage05.1_best_known_attempt06` 已通过独立 replay，状态为 `READY_FOR_STAGE05_2`；
- 上游 Formal identity 为 `stage04_adaptive_weights_attempt15`，144 axes 完整；
- main repository（主仓库）使用 clean commit，Stage 0 frozen baseline 与历史 raw evidence 未改写；
- 本文件、`AGENTS.md`、路线图和 `docs/experiment_artifact_storage.md` 的规则一致。

## Canonical components

| 顺序 | Component | Canonical label | 目标 |
| --- | --- | --- | --- |
| A | `perf_baseline` | `stage05.2_perf_baseline_attemptNN` | 冻结可复验性能基线 |
| B | `hot_path` | `stage05.2_hot_path_attemptNN` | 消除 Python 重复工作 |
| C | `artifact_streaming` | `stage05.2_artifact_streaming_attemptNN` | 实现 v2 流式分片证据 |
| D | `job_parallel` | `stage05.2_job_parallel_attemptNN` | 选择任务级 worker 数 |
| E | `native_kernels` | `stage05.2_native_kernels_attemptNN` | 迁移 profiling 确认的 CPU 热点 |
| F | `accelerator_pilot` | `stage05.2_accelerator_pilot_attemptNN` | 条件式 GPU/Metal/MPS 决策 |
| G | `benchmark` | `stage05.2_benchmark_attemptNN` | 执行 pilot 和分层正式实验 |

任何失败或被中断的运行使用新 `attemptNN`/`rerunNN`，原 shard、manifest 和 failure evidence 不得覆盖。F 可以以 `GPU_NOT_JUSTIFIED` 通过；A–E 和 G 不得跳过。

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

- artifact persistence ≤ end-to-end 的 30%；
- peak RSS ≤ A 中 v1 baseline 的 50%；
- timeout、byte-budget、writer error 和 worker failure 均保留 partial shard 并 fail fast；
- reviewer 不读取全量 events 到一个 Python list，不依赖 worker completion order。

当前 accepted evidence（已验收证据）为
`stage05.2_artifact_streaming_attempt04`，其独立审查状态为
`READY_FOR_STAGE052_JOB_PARALLEL`。固定 4 instances × 3 seeds 共 36 axes
全部通过 exact scope、validator/objective replay 和 Python optimization
profile；其中 24 个 fixed-work axes 通过 v1/v2 semantic equality。最终
row-group flush、Zstandard level 1 压缩、磁盘写入与 control finalisation
均计入后，aggregate persistence ratio 为 24.8546%；peak RSS 为
2,565,537,792 bytes，低于 A 阶段 4,357,382,144-byte 上限。
attempt01--03 保持为失败或 partial evidence，D 只能从 attempt04 进入。

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
并独立核对 C04 identity、instance/config/native-extension checksum、Python/package/
machine identity、affinity、non-secret performance environment variables、background
load 和 power mode。

`stage05.2_job_parallel_attempt01`--`attempt03` 的数值门槛虽然通过，但 shard
manifest 记录的是按 ordinal 推算的 synthetic PID（合成进程标识），无法证明真实
worker ownership，因此永久保留为不可晋级证据。修正后的正式序列从 attempt04
开始并已通过独立 review：

- `stage05.2_job_parallel_attempt04`：1 worker，`run_wall_seconds=406.5132`，
  aggregate RSS 为 2.2379 GiB；
- `stage05.2_job_parallel_attempt05`：2 workers，`run_wall_seconds=223.4358`，
  aggregate RSS 为 4.0124 GiB，相对 attempt04 speedup 为 1.819x；
- `stage05.2_job_parallel_attempt06`：4 workers，`run_wall_seconds=130.6914`，
  aggregate RSS 为 5.4178 GiB，相对 attempt04 speedup 为 3.110x。

selection review 选择 `selected_workers=4`，状态为
`READY_FOR_STAGE052_NATIVE_KERNELS`。Component E 必须绑定 attempt06 的 selection
identity；此处不构成 Component E 或 F 已完成的声明。

## E. Native CPU kernels

只迁移 profiling 已证明占主导的路径：screening、propagation snapshot、distance lookup、exact label expansion/dominance/heap。数据边界使用 contiguous integer/float arrays；C++ 核心计算可释放 GIL，但进入 Python callback、异常构造或对象访问前必须重新持有 GIL。

每个 kernel 先做 frozen fixture、small brute-force（小规模暴力枚举）和 fixed-work differential test（差分测试），再进入完整性能 scope。E 的组合实现必须相对 D 的 accepted configuration 再通过严格 15%/3% 门槛。

## F. Conditional accelerator pilot

只有 E 后 median route batch occupancy ≥ 32 才启动。否则直接发布带证据的 `GPU_NOT_JUSTIFIED`。pilot 必须记录 packing、host-to-device、kernel、device-to-host、synchronisation 和 total end-to-end time。

GPU/Metal/MPS promotion 同时要求：fixed-work 语义一致；相对 E selected native CPU，全部 100-customer 配对的总体端到端中位时间至少降低 15%；C、R、RC 任一 family 的 family median 回退不超过 3%。任一条件失败即保留 native CPU。正式 runner 不得设置隐式 CPU fallback。

## G. Pipeline pilot 与正式分层预算

### Pipeline pilot

先用固定 Stage 0 representative set（代表集）执行 `12 instances × 3 seeds`，完整走过 selected backend、selected worker count、artifact-storage-v2、failure handling、independent replay、summary generation 和 registry/manifest publication dry run（发布演练）。所有 36 bundles 完整且 review 通过后才开放 Formal。

### Formal budget matrix

| Scope | Seeds | 30 s | 60 s | 300 s |
| --- | ---: | ---: | ---: | ---: |
| 36 个 5/10/15-customer 实例 | 10 | 必做 | 不运行 | 不运行 |
| 56 个 100-customer 实例 | 10 | 必做 | 必做 | 必做 |

该矩阵共 2,040 个 solver runs（求解运行），声明的 solver time budget 总计 229,200 秒，即 63 小时 40 分钟；该数值不含启动、写盘、review 和失败重跑开销。正式排期必须再用 D 阶段选定的 worker 数和实测 persistence overhead（持久化开销）估算 wall time，不能把理论时间预算当作实际完工时间。

anytime checkpoints 为预算范围内的 `1/5/10/30/60/120/300 s`。small instances 在 30 秒或 iteration limit 后结束，不为矩阵对称性浪费 60/300 秒。随机 seeds 在 Formal 前冻结并写入 config/manifest；不得按结果更换。

## Review 与发布

independent reviewer 必须从 raw shards 重算：exact scope identity、validator/objective、event/cache/exact-call、deadline/budget、worker/shard completeness、resource limits、严格性能 gate、anytime 汇总和模型兼容性。BKS compatibility 仍为 `False`，因此不创建 gap 列。

必须发布：

- `experiments/registries/stage05.2_artifact_registry.csv`；
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
