# Stage 5.2 持续迭代变更日志

Stage 5.2 只维护一套 current implementation（当前实现）。A--G 是该实现内部的顺序
gate（门槛），`attemptNN`/`rerunNN` 是实验运行身份，不是代码版本。此文件按时间追加，
不得为整理历史而改写旧条目。

每条记录至少包含：原因、修改范围、行为变化、证据影响、失效或迁移的运行身份、验证
结果及后续运行要求。大型 raw evidence（原始证据）的物理位置由
`experiments/registries/stage05.2_retention_registry.csv` 记录。

## 2026-07-23：E11 真实剖析与 bounded async persistence pipeline

- 失败证据：`stage05.2_native_kernels_attempt11` 在 clean commit `cffa5a5` 上完成 36 个
  axis 且资源记录不再包含零 RSS 进程，但 producer persistence ratio 仍为
  0.381931566（solver 201.266761702 秒，persistence 124.371550767 秒），因此不进入
  明知必败的耗时独立 review，也不得成为 E prerequisite。该完整 raw 保持不可变。
- 证伪与剖析：exact-route identity 内存化虽在 225,000-entry microbenchmark 中达到约
  2.89x，却不是 formal workload（正式负载）的主导成本。E11 的大型 shard 共含
  7,346,949 个 screening occurrences，其中 6,143,240 个是 negative-cache hit；真实
  `c101_21/2014` cProfile 显示 screening callback preparation 与同步 shard append 是
  persistence critical path（持久化关键路径）。
- 修改：native/benchmark component 使用一个 shard-local bounded async persistence
  pipeline（分片本地有界异步持久化流水线）。每个 axis 只有一个 non-daemon FIFO writer
  thread，队列硬上限为一个 65,536-row callback batch；无 fallback。producer 可与后台
  Parquet 编码/I/O 重叠，但 `finish`、semantic digest（语义摘要）和 finalize 前必须完整
  drain。后台异常在下一次 submit/drain 立即抛出；abort 先停止并回收线程再封存 partial
  shard。非 native 历史路径保持同步。
- 可观测性：每个 trace axis 记录 mode、queue hard bound、submitted/completed batch、
  peak queued batches、producer/writer wall nanoseconds、writer thread CPU nanoseconds、两者
  wall activity union、solver-boundary concurrent critical path diagnostic（并发关键路径诊断量）、
  producer wait，以及逐 batch 的 row count + logical-event SHA-256 ledger。正式 30% gate 仍使用
  drained solver boundary 上 producer/writer wall interval 的并集，重叠只计一次；producer wall
  与 writer thread CPU 的最大值只用于解释 GIL scheduling，不参与 readiness。后台
  hashing/Parquet I/O 不得静默消失或伪造为零。E reviewer 与 G campaign
  reviewer 都从 logical event stream 独立重算 ledger，并交叉核对 timing evidence；缺失、
  partial completion、over-bound、digest mismatch 或异步错误均 fail fast。pipeline metadata
  是物理存储证据，不参与 D/E fixed-work algorithm semantics（算法语义）比较。
- GIL-aware cooperative turn（感知 GIL 的协作轮次）保证已提交批次在下一 callback 前完成，
  writer 可与 callback 之间的 solver core work（求解器核心工作）重叠，但不会因 Python
  线程争用把 descheduled wall time（被调度暂停的墙钟时间）伪装为写入成本。每批 writer
  thread CPU 只作为诊断，并按同一批 wall interval 上限裁剪粗粒度 CPU clock tick；正式归因
  始终使用 wall activity union。async callback batch 被 producer 与两个 reviewer 同时硬限制为
  `1..65,536` rows。
- negative-cache 快速路径：每个 route 仍逐次比较完整 evidence tail 以检测漂移，但重复
  cache hit 复用已验证的 `PrecomputedScreeningDefinition`，不重复展开 screening checks。
- 验证：一百万个重复 negative-cache callback 从 1.601136860 秒降至 0.907102130 秒；
  初版带真实求解、Parquet writer 和 finalization 的单 shard cProfile 从
  10.448471206 秒降至 6.593397739 秒，但该初版未把后台 writer 活动并入 gate attribution，
  因而只用于定位、不能作为 E 通过依据。修正后的计时明确覆盖 producer preparation、batch
  hashing 和 Parquet append；正式结论仍只接受新 clean commit 的 E12 raw 与 systemd
  independent review（独立审查）。

## 2026-07-23：E10 persistence 与 resource identity 根因修复

- 失败证据：`stage05.2_native_kernels_attempt09` 在创建任何 shard 前因冻结 source
  缺少 benchmark data（基准数据）而 fail fast；空 run directory 保留且该 label 不复用。
  `stage05.2_native_kernels_attempt10` 完成 36/36 validator replay（验证器重放），独立审查
  仍发布 `NOT_READY`：aggregate persistence ratio 为 0.381727992，且 resource contract
  （资源合同）发现 transient exited process（瞬时退出进程）的零 RSS 记录。E09/E10 均
  保持不可变。
- persistence 根因：E10 共持久化 9,139,744 个事件，其中 286,085 个 exact route
  evaluation（精确路径评估）原本对三类唯一身份逐次执行 SQLite SELECT/INSERT；各 shard
  的实际集合远低于既有有界内存预算。现在 exact-route identity store（精确路径身份存储）
  在内存中保存完整 SHA-256 digest 与 payload collision proof（载荷碰撞证明），超过明确
  262,144-entry 上限后才原子迁移到 shard-local SQLite。spill（溢写）后的去重、碰撞和
  namespace（命名空间）计数语义不变。
- screening 存储：producer 已把 canonical definition（规范定义）写入 Parquet，因此
  collision store（碰撞存储）只保留完整 32-byte SHA-256 token，不再保留第二份 JSON；
  reviewer/read path 仍保留可解析 payload，超过 producer 的 1,200,000-entry 硬上限才
  spill。八项 screening-check 上限在 typed fast path（类型化快速路径）与兼容路径中均
  fail fast。
- resource 根因：`psutil` 可能在子进程退出后、成为 zombie（僵尸进程）前短暂返回零化
  memory record（内存记录）。sampler 现在只在 RSS 为正且 CPU times 同一次 oneshot
  采样成功后登记 process identity（进程身份）；半采样和零 RSS 进程均不形成虚假的
  measured worker（已测工作进程）。
- 验证：新增内存路径、强制 spill 去重、producer digest-only、零 RSS transient process
  回归测试。225,000 个 exact-route identities 的 ext4 microbenchmark（微基准）从
  1.382240497 秒降至 0.478620427 秒（约 2.89x）；Stage 5.2 artifact/core 133 项和
  governance 184 项测试通过，Ruff 与 strict mypy 通过。正式 E 门槛仍须由新 clean
  commit 的 E11 producer 与独立 raw replay 决定，微基准不构成通过证据。

## 2026-07-23：E native runtime 与 v3 screening 热路径修复

- 失败证据：`stage05.2_native_kernels_attempt08` 保留为 `NOT_READY`；其 reviewer 发现
  producer/reviewer 的 native extension（原生扩展）虽来自同一 sealed wheel（密封
  wheel）且 SHA-256 相同，却因 venv 绝对路径不同而被拒绝；同时 5,831,819 条
  screening decision（筛选决策）仍经过逐事件 dict normalization（字典规范化），使
  aggregate persistence ratio（聚合持久化占比）达到 0.496440834。
- 根因修复：native profile 改为比较 reviewer 当前扩展的内容哈希与 producer 签名哈希，
  不再要求机器本地安装路径相等；v3 live trace bridge（实时轨迹桥）复用预计算 typed
  screening definition（类型化筛选定义），直接向 artifact deep module（产物深模块）
  提交 occurrence tuple（出现记录元组），同时保留 v2 compatibility path（兼容路径）、
  negative-cache evidence drift（负缓存证据漂移）检查和完整 persistence timing（持久化
  计时）。
- 验证：新增跨 venv 同哈希 native extension、预计算定义复用和真实 Parquet round-trip
  （Parquet 往返）回归测试。100,000 条 synthetic screening（合成筛选）探针测得
  0.390909233 秒 charged persistence（计入持久化时间），用于在新 attempt 正式重跑前
  验证热路径量级；正式门槛仍只由独立 raw replay（原始重放）决定。

## 2026-07-23：v1 物理 screening schema 独立识别修复

- 原因：C16 的独立审查正确验证了 24 个 fixed-work axis（固定工作量轴）的 v1/v3
  canonical equality（规范等价），但 reviewer 只把独立
  `screening_decisions_v1` subtype（子类型）识别为 v1。实际 artifact-storage-v1 将
  screening decision（筛选决策）保存在普通 critical event stream（关键事件流）中，
  另存 `screening_checks`；因此不存在该 subtype，C16 被错误判为物理 schema 无效。
- 修改：v1 识别现在要求 manifest 明确声明 `screening_decisions_v1`、同时存在
  `critical` 与 `screening_checks` 事件工件、至少一个 trace index（轨迹索引），且每个
  trace 均不存在 v2 compact 或 v3 definitions/occurrences 引用。任何混合或缺失布局仍
  fail fast。
- 证据影响：B06、C16 raw 与 C16 的首个 `NOT_READY` review generation 保持不可变；
  这是 reviewer-only 修复，可在新 clean reviewer revision 下复用 C16 raw 产生追加式
  review generation，无需重跑 solver。
- 验证：新增 v1 embedded layout（内嵌布局）正例和伪装 v2 reference（引用）反例；修复
  前正例按预期失败，修复后 3 个 screening-schema 定向测试通过，并在真实 B06/C16
  manifest 上分别重放为 v1/v3。

## 2026-07-23：跨 revision 冻结 producer replay 修复

- 原因：A10 通过新 reviewer receipt 后，B04 的 producer prerequisite binding 仍指向旧
  A10 review hash，因此 B04 在现行合同下正确降为 `NOT_READY`。随后 B05 preflight 又
  暴露 current-chain verifier 错用下游 current worktree 重放上游 producer runtime；当
  两者 revision 不同时必然失败。
- 修改：current-chain verifier 从已绑定当前 review manifest 的 systemd receipt 读取上游
  producer `working_directory`，要求 receipt 中的 producer revision 与 raw identity 完全
  一致，并在该冻结 source root 中重放 wheel、Python、native extension、dependency 与
  machine identity。下游 current worktree 只继续提供当前 storage-root locator。
- 证据影响：B04 的 `NOT_READY` 与失败 review 保留；B05 在 output directory 创建前失败，
  没有 raw shard，后续实际 B 重跑必须使用新 label。A10 raw 未改写，其现行 review receipt
  可作为跨 revision prerequisite。
- 验证：新增跨 revision producer root 与 receipt revision mismatch 测试；修复前 2/2 按
  预期失败，修复后 2/2 通过。Ruff 与 strict mypy 通过；完整套件将在新 ext4 clean
  worktree 安装后复跑。

## 2026-07-23：C 前独立审查加固

- 原因：从固定点 `7f0944e` 的双重独立 code review 发现 canonical raw、wheel source、
  fresh-process field replay、systemd receipt、retention registry preflight 和追加式 review
  lineage 存在可导致无效正式证据的路径。
- 修改：formal reviewer 固定 5.5-GiB 内部上限，绑定 canonical signed manifest，校验 wheel
  与 clean revision 的 tracked Python source；field-level mismatch 的 comparison/candidate
  replay 分别使用 fresh spawned process；`ExecStopPost` 把成功 receipt 与当前 review hash
  绑定后 READY 才可消费。retention 在移动 source 前持锁预检 registry，current status 不再
  被历史 NOT_READY 覆盖；accepted/retry lineage 支持追加且接受合法三文件 generation。
- 证据影响：A10/B04 的既有 raw 不改写，但 B04 必须用新 reviewer 重新审查并生成 receipt
  binding；C–G 只能从本修复后的 clean commit 启动。

## 2026-07-23：单一版本与外部证据归档

- 基准源码 revision：`1faf761`（本次修改保持未提交状态）。
- 原因：仓库工作区累计 166 个 Stage 5.2 运行目录、41,638,830,980 bytes；A--G 与
  `attempt/rerun` 被误解为多套长期版本，失败和 superseded raw 在工作区无限增长。
- 修改范围：新增 `evrptw.stage052_retention`；更新 Stage 5.2 配置、治理规则、工作流、
  artifact storage 文档和主路线图。
- 行为变化：
  - Stage 5.2 代码只保留一套当前实现，A--G 继续保持原顺序和全部验收门槛；
  - run label 继续唯一且不得覆盖，但 sealed run 通过签名 inventory 与完整 tree SHA-256
    校验后迁移到 `d_archive/stage05.2/history/<run_label>/`；
  - 同 volume 使用原子移动；跨 volume 使用隐藏临时目录复制、完整复验、目标卷原子落位，
    再清理 source。source 漂移、目标冲突或复验失败均 fail fast；
  - registry 原子合并历史行，相同 run identity 幂等，checksum/bytes 等身份冲突立即失败；
    read-merge-replace 由跨进程 lock 串行化，避免并发 archive 丢失历史行；
  - active/unsealed 默认拒绝；历史迁移 override 必须同时声明预期目录数和总字节数；
  - performance runner/reviewer 与 Formal campaign reviewer 可用 run label 经 registry
    与 storage-root locator 解析并复验归档 prerequisite/comparison；归档 tree 保持只读，
    新 review generation 必须在 active raw 上完成后再归档；
  - current chain 从 manifest、prerequisite identity 和 retention registry 解析，不再写死
    C05/D07 等 attempt 编号。
- 证据影响：不改变 solver、objective、validator、fixed-work、performance gate 或
  independent review 语义；只改变 Stage 5.2 raw 的工作区保留位置。Stage 0--5.1 冻结
  证据不在本次迁移范围内。
- 迁移范围：改造前 `results/stage05.2_*` 全部作为 historical evidence（历史证据）迁移；
  每个目录的状态、completeness、source commit、prerequisite、文件数、字节数和 SHA-256
  进入轻量 registry。
- 迁移与 audit identity（审计身份）：源 inventory SHA-256 为
  `537d848642d394811c1518d85d2a33479d8439e962efb0e46a8b99fcd003a044`；归档后
  inventory SHA-256 为
  `0a9dc19c2de43e35c45d2f387a26ae7f815978c36ad48439b15c65cb8418c32d`。166 个
  目录、41,638,830,980 bytes 的逐目录文件数、字节数与 tree SHA-256 全部一致；仓库
  `results/` 中旧 Stage 5.2 大型目录计数归零，归档位置目录计数为 166。
- 迁移 preflight：归档开始前检查 Stage 5.2 producer/reviewer 进程，无写入进程；随后
  重新生成 inventory 并确认恰为 166 个目录、41,638,830,980 bytes 后才执行迁移。
  registry 中无法从旧目录恢复语义状态的 `unknown` 行仅表示历史字节保存，不单独构成
  current-chain 或 accepted prerequisite；后续使用仍须通过 manifest/reviewer gate。
- 验证：retention 单元测试 25/25 通过；完整 pytest 681/681 通过；Ruff 通过；strict
  mypy 对全部既有 `src` 模块（69 个）及新增 retention 模块分别通过；
  `git diff --check` 通过。
- 后续要求：新 Stage 5.2 producer/reviewer 只能短期使用 active staging root；封存后必须
  audit/archive。任何新的算法或证据语义修改直接更新当前实现，并在此追加新条目。

## 2026-07-23：C--G reviewer 执行封套与差异输出约束

- 基准源码 revision：`7f0944e` 为 accepted B04 producer；最终实现 revision 在验证后记录。
- 原因：B04 的 wall-clock trajectory（墙钟轨迹）预期不同，却被展开为逐字段 mismatch；
  单个 `semantic_mismatches.csv` 达 13,312,141,268 bytes。另一个阻塞是 bounded
  `systemd --user` service 只允许 performance reviewer，G Pilot/Formal campaign
  reviewer 无法进入同一 cgroup、receipt 与内部 RSS 合同。
- 行为变化：只有 fixed-work axes 执行字段级差异；wall-clock axes 每个 identity 只保留
  aggregate digest row。review service 明确允许 performance 与 campaign 两个隔离模块，
  分别验证 raw/prerequisite 参数；campaign reviewer 使用相同外部 progress log 和
  5.5-GiB process-tree guard。
- 证据影响：producer/solver 语义不变；B04 使用新 reviewer generation 复审后才锁定为
  C prerequisite。旧 review generation 与巨大 mismatch 文件保持不可变并归档。
