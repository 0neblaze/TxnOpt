# Stage 5.2 持续迭代变更日志

Stage 5.2 只维护一套 current implementation（当前实现）。A--G 是该实现内部的顺序
gate（门槛），`attemptNN`/`rerunNN` 是实验运行身份，不是代码版本。此文件按时间追加，
不得为整理历史而改写旧条目。

每条记录至少包含：原因、修改范围、行为变化、证据影响、失效或迁移的运行身份、验证
结果及后续运行要求。大型 raw evidence（原始证据）的当前物理位置由 `e_archive`
中的签名 v2 registry generation（注册表代次）记录；tracked
`experiments/registries/stage05.2_retention_registry.csv` 保持为不可变 v1
compatibility fallback（兼容回退）。

## 2026-07-26：G56 24-thread saturation runtime guard 修复

- Formal `stage05.2_benchmark_attempt56` 从密封 revision
  `8267475d4048adac88b05df604aa1cfdc4423c3a` 启动，声明范围保持 920 shards、
  2,040 axes、229,200 solver seconds 与 10,400 checkpoints。batch0001 在
  campaign 自身进入正常高并发阶段后由 runtime guard 终止；该 label 永久保留，
  不续跑且不导入其中已产生的 shard。
- 失败 runtime evidence 记录 `maximum_load1=21.60009765625`、
  `maximum_unrelated_process_average_cores=0.0`、24 logical CPUs、AC power、
  low-power mode 关闭；systemd 记录 aggregate peak RSS 仅 97.4 MiB、swap 为零。
  因此首因是旧 `20.0` 上限低于冻结四-worker producer 在 24-thread machine 上的
  正常满载与短时运行/I/O 排队，不是外部 CPU 竞争、内存不足、供电或机器身份漂移。
- Operator 明确要求提高负载上限并最大化匹配本机性能。preflight 的两个
  `load1 <= 4.0` 空闲窗口保持不变；runtime fail-fast ceiling 提高到 `32.0`，
  即允许 24 logical CPUs 满载和最多八个任务的短时排队。它不增加 worker 或 solver
  线程，只避免把正常饱和误判成 runaway。排除 campaign PID tree 后的 unrelated
  process 一整核 gate、20-GiB producer RSS、36% persistence、四 workers、
  solver/backend/objective/validator 与 5.5-GiB reviewer RSS 合同均保持不变。
- 回归测试使用 Attempt56 的真实 `21.60009765625` 样本：旧合同确定失败，新合同
  必须通过；`32.1` 仍必须 fail fast。producer 与 independent reviewer 都从同一个
  current contract 重建 `32.0`，禁止 producer 自报更高阈值。合同变更后必须以新
  label 重跑 Pilot 并通过 independent review，再以新的最低未占用 Formal label
  从零运行。
- 新合同下的 Pilot `stage05.2_benchmark_attempt57` 正常完成并归档 batch0001 与
  batch0002；两批 runtime peaks 分别为 `0.6875` 与 `8.56982421875`，外部无关进程
  均为 `0.0` 核。batch0003 启动前却因 `Stage 5.2 preflight load1 exceeds 4.0`
  失败。worker 已全部退出，数分钟后 load1 自然降至 `1.30`，证明被拒绝的是
  batch0002 留在 Linux 1-minute load average 中的历史，而非并发外部负载。
- campaign-start 两段 `load1 <= 4.0` 门槛保持不变。per-batch handoff preflight
  仍保留两段 30 秒原始采样并校验 AC、low-power mode、24 logical CPUs 与排除
  campaign PID tree 后的一整核 unrelated-process gate，但不再用上一批的衰减
  system load average 拒绝下一批；新批次启动后立即受 `32.0` runtime guard 约束。
  回归测试要求 Attempt57 的 `8.56982421875` handoff 通过，同时 campaign 首次启动
  的 `4.1` 仍失败。Attempt57 永久保留，不续跑、不导入 shard。

## 2026-07-26：G48 24-thread runtime load ceiling 修复

- Formal `stage05.2_benchmark_attempt48` 使用 fresh-worker waves 与已放宽的
  20-GiB producer process-tree memory gate；batch0001 的 395/395 shards 完成并
  通过 `persistence_ratio=0.2552491582829223`，约 9.64 GB raw 经
  copy→verify→atomic publish 归档到 D 盘。batch0002 运行到 runtime guard 触发后
  以 partial evidence finalized，不续跑、不导入任何已完成 shard。
- batch0002 runtime evidence 记录 `maximum_load1=11.791015625`、
  `maximum_unrelated_process_average_cores=0.0`、AC power 稳定且 low-power mode
  关闭；resource evidence 的 `mean_active_cores=4.107891249524004`、
  `peak_active_cores=26.76909314835371`、aggregate peak RSS
  `8,490,471,424` bytes。失败不是内存不足或外部进程竞争，而是旧的 runtime
  `load1 <= 8.0` ceiling 未匹配冻结机器的 24 logical CPUs 与高基数 Parquet
  persistence 工作负载。
- Operator 明确授权提高负载上限以匹配本机性能。campaign preflight 继续要求两个
  连续 30 秒窗口 `load1 <= 4.0`；batch runtime total-load ceiling 固定为 `20.0`，
  允许使用约 83% 的 24 logical CPUs，并为 host/archive I/O 保留四个逻辑处理器。
  unrelated user CPU 的完整 30 秒窗口一整核 gate、AC power、low-power mode、
  20-GiB producer RSS、36% persistence、四 worker、solver/backend/objective/
  validator 与 5.5-GiB reviewer memory contract 均不放宽。
- Producer runtime evidence 显式记录 `20.0`，independent reviewer 从 shared
  current contract 独立重建该值并比较实际 maximum；边界回归要求 19.9 通过、
  20.1 fail fast。修复后必须使用新 label 重跑 Pilot 并通过 independent review，
  再以新的最低未占用 Formal label 从零执行 920 shards。

## 2026-07-26：G46 fresh-worker wave overlap 的 load1 根因修复

- Formal `stage05.2_benchmark_attempt46` 通过完整 preflight，并在 batch0001
  完成 396/396 shards 后以 `load1 exceeded 8.0` fail fast。失败 runtime evidence
  完整记录 146 次采样：`maximum_load1=8.1640625`，排除 campaign descendants 后的
  `maximum_unrelated_process_average_cores=0.8031696417825888`，AC power 稳定且
  low-power mode 始终关闭。Attempt46 不续跑、不导入任何 shard。
- 根因是 `max_tasks_per_child=1` 虽保证每 shard 唯一 PID，但一个长寿命
  `ProcessPoolExecutor` 会在旧 worker 退出期间初始化 replacement worker；快速小
  shards 连续完成时，retiring/warmup 生命周期短暂重叠，使 campaign 自身总
  `load1` 超过固定的 `4.0 + selected_workers = 8.0`，并非外部主机负载。
- Producer 现在把有序 shards 分成最多四个一组的连续 waves。每一 wave 保持冻结的
  四 worker 并发、`spawn`、`max_tasks_per_child=1` 和 in-memory Arrow/Zstd warmup，
  但必须完整 shutdown 后才创建下一 wave。total-load、unrelated-process、AC power
  和 low-power gates 均不放宽；solver、objective、validator、backend、事件和
  persistence 合同不变。由于这是 producer 调度修复，必须先用新标签重跑 Pilot 并
  通过独立 review，才能启动新的 Formal。

## 2026-07-26：Formal attempt37 producer 逐 shard worker 回收

- `stage05.2_benchmark_attempt37` 完成并跨卷复验归档 batch0001--batch0006，
  共 636/920 shards；batch0007 完成 raw shard 计算后由既有 12-GiB
  process-tree aggregate RSS gate 拒绝。失败 batch 的实测峰值为
  `15,841,632,256` bytes，四个长期 worker 的各自峰值为
  `5,079,650,304`、`4,651,958,272`、`4,480,434,176` 和
  `4,106,354,688` bytes。Attempt37 永久保留为 failed evidence，不续跑、不导入
  已完成 shard。随后 operator 明确授权放宽当前 G campaign 的 producer 内存门槛；
  这不追溯改变 Attempt37 的失败判定。
- 根因是 producer 的 `spawn` `ProcessPoolExecutor` 在整个 batch 内复用四个 worker；
  每个进程连续处理约十二个高基数 shard，Python/native allocator 高水位跨 shard
  保留。resource sampler 的 parent 仅约 76 MiB，短命 native child 绝大多数约
  4 MiB，排除了 parent accumulation、退出进程重复计数和 orphan overlap。
- parallel shard executor 现在固定 `max_tasks_per_child=1`：并发上限仍是冻结的四
  workers，但每个 `(instance, seed)` shard 使用新的 `spawn` worker，进程退出后回收
  Python/native allocator。batch metadata 新增
  `worker_process_lifecycle=one_shard_per_spawned_process`，independent reviewer
  并在每个新 worker 的计时区间开始前执行
  `worker_runtime_warmup=in_memory_arrow_zstd1`，消除 fresh-spawn 引入的
  PyArrow/Zstandard 一次性初始化成本；warmup 仅写入内存 buffer，不生成、
  删除或聚合任何审计事件。
- Pilot Attempt42 首次 independent replay 暴露 reviewer 将包含 interleaved
  persistence 的 `candidate_state.timestamp_seconds` 直接与 declared solver
  seconds 比较，重复并错误地拒绝了 deadline boundary 前的合法事务。修复后
  lane-local deadline boundary 仍是权威边界；exact completion、cache store
  和 boundary 后 acceptance 的原有拒绝规则保持不变。
- 同一 Pilot 的下一代 review 又暴露 deadline interrupt 可在 batch 已声明
  launch、但 native kernel invocation 尚未发生时终止。counter reconciliation
  现在要求 `started = completed + interrupted`，并只允许
  `batch_launches - native_invocations` 落在已证明的 interrupted 数量内；
  fallback 仍必须为零，未证明的 counter 缺口仍 fail fast。
- Combined single-pass accumulator 还必须把
  `route_evaluation.status=interrupted_deadline` 识别为该 lane/axis 的 deadline
  boundary evidence；不能只依赖可为空的附加 `deadline_boundary` 字段。该映射
  同时阻止同 lane 后续 exact work、cache store 或 candidate acceptance。
  必须核对该合同；worker ownership 接受多于 configured concurrency 的真实、已采样
  owner PID，但少于四个仍 fail fast。
- 当前 G Benchmark Pilot/Formal 的 producer/reviewer batch resource gate 调整为
  per-worker `8 GiB`、process-tree aggregate `20 GiB`；WSL producer memory cap 从
  `16 GB` 调整为 `24 GB`，给 Windows host 保留约 `8 GB`。已接受 D/F selection 的
  12-GiB scientific gate、36% persistence gate、四 worker 并发、solver/backend 和
  independent reviewer 的 5.5-GiB internal guard/6-GiB systemd `MemoryMax` 均不变。
- 定向反馈测试在修复前因缺失 `max_tasks_per_child` 稳定失败；修复后同时证明每个
  shard 获得唯一 PID、异常 worker 仍无 fallback、旧 ownership evidence 仍兼容。
  Stage 5.2 定向测试为 360 passed；完整测试为 768 passed，Ruff、strict mypy 和
  `git diff --check` 全部通过。
  由于这是 producer 缺陷，后续必须使用新 clean revision 先跑新的 Benchmark Pilot，
  通过独立 review 后再以新的最低未占用 Formal label 执行完整 920-shard campaign。

## 2026-07-25：G26 campaign reviewer 单遍重放与逐 shard 内存隔离

- `stage05.2_benchmark_attempt26` producer 已完整生成 36/36 Pilot axes，aggregate
  persistence ratio 为 `0.33339944391668525`，但首次 independent review 在
  batch0003 只完成 26 个 shard 后达到固定 5.5-GiB aggregate RSS guard，发布
  `NOT_READY`。其余 geometry、persistence、resource、runtime 和 publication failure
  是 replay 未完成后的级联结果，raw producer evidence 未被判定为科学失败。
- 根因是 campaign reviewer 在同一长寿命进程内对每个 shard 的 logical event stream
  分别执行 persistence ledger、transaction/deadline 和 global-best 三次完整重建。
  `ArtifactReader.iter_events` 每次都重建 route dictionary 与 screening definition
  disk-backed stores；PyArrow、SQLite 与 Python allocator 高水位跨 shard 保留，违反
  reviewer 文档已经要求的 fresh spawned bundle isolation。
- reviewer 现在用一个 incremental accumulator 在一次 logical pass 内同时完成三类
  审计。每个 shard 严格由新的 `spawn` child process 重读 signed batch/shard manifest、
  在独立 ext4 scratch 中重放，并只向 parent 返回 per-axis rows、checkpoints、计数和
  resource telemetry。child 异常直接 fail fast，无 parent replay 或进程复用 fallback。
- progress log 对每个 shard 记录 parent/child PID、RSS、PyArrow allocation、事件数、
  单遍计数、耗时与 scratch cleanup；5.5-GiB internal guard、systemd MemoryHigh/Max、
  raw schema、objective、validator、event surface 和 publication schema 均未改变。
- 真实 G26 只读回归完整重放 36/36 shards 与 16,295,563 条 logical events，使用 36 个
  唯一 child PID，耗时 376.728890604 秒；parent peak RSS 为 89,194,496 bytes，每个
  child 结束时 PyArrow allocation 为 0，scratch 全部清理。旧失败点之后的
  100-customer shards 也完整通过，未修改 G26 raw 或首次 `NOT_READY` generation。
- 完整测试暴露出 WSL 上 `POSIX_FADV_DONTNEED` 对刚完成原子替换的小型 signed JSON
  产生错误 page-cache replay；同一读取路径曾返回全零页或 systemd journal block，
  而落盘后的最终 SHA-256 仍正确。离线 `e2fsck -f` 验证 ext4 元数据完整且 0 bad
  blocks。平台合同现在在 WSL 明确禁用该 advisory cache-drop 优化，原生 Linux 行为
  保持不变；该优化不计入进程 RSS，也不改变 raw/review 语义。修复后两个原故障测试
  连续五轮通过，完整测试为 755 passed，Ruff、strict mypy 与 `git diff --check`
  全部通过。

## 2026-07-24：G15 producer 通过 36% 门槛但独立 review 拒绝不完整 route identity

- clean revision `9834400` 上的 `stage05.2_accelerator_pilot_attempt14` 已由独立
  transient service replay 通过，状态为 `READY_FOR_STAGE052_BENCHMARK`。raw/review
  manifest SHA-256 分别为
  `cad31e49851ee5953431045e1fcf601097ae6c1515619d7c875133ae0fc48c4b`
  和 `94b5a22df0c983903dff487598af02935f9a7da15c5ce30e151f83513cec5d16`。
  同 revision 的 F13 在 raw 创建前因测试工作副本遗留 `.pytest_cache` 被 source
  snapshot guard 拒绝；正式 producer 改用从未执行测试的新只读 ext4 clone。
- `stage05.2_benchmark_attempt15` producer 完成 36/36 pilot shards，三个 batch
  persistence ratio 为 `0.34517927869972653`、`0.29652732269638515` 和
  `0.3381115581407812`，campaign aggregate 为 `0.33377378597941765`，全部低于
  36% hard gate。独立 campaign reviewer 正确返回 `NOT_READY`，未开放 Formal。
- 最早 review failure 是首个 global-best event 的 route replay。producer 的
  `candidate_route_keys` 按设计只表示 customer sequence；reviewer 错把该序列当作已经
  包含 depot/station 的完整路线交给 validator，导致首个客户被报告为 unvisited。
  其余 geometry/resource/storage 失败是 reviewer 在首 shard 中止后的级联结果，不得
  单独标为真实 gate failure。
- 当前实现保留原 `candidate_route_keys` 语义，并新增
  `candidate_full_route_keys`，从同一个 feasible `_EvaluatedSolution` 的 exact-charging
  results 记录完整 depot/station routes。该字段作为 typed event 的 bounded
  `extras_json` list 持久化，不改变现有 Parquet schema；完整 route keys 同样进入
  route dictionary。reviewer 现在验证完整路线与 objective，并要求其 customer
  projection 与原 customer sequences 完全相等。缺失、不可行、objective mismatch
  或 projection mismatch 均 fail fast。G15 保留 `NOT_READY`，后续须从新 clean
  revision 重跑 F/G。

## 2026-07-24：F12 通过、G13/G14 环境失败并稳定 Windows edition identity

- clean revision `c17e0dc` 上的 `stage05.2_accelerator_pilot_attempt12` 已由独立
  transient service replay 通过，状态为 `READY_FOR_STAGE052_BENCHMARK`。raw/review
  manifest SHA-256 分别为
  `e6927e66fd544aa3e4f5f21381c61f9b54c47a08f7a8179b12858037f816cebc`
  和 `53fa45cc8b6f8dc0c44eb93f24c7c2fa07e39ef531bf235ee29d62d421e508fc`；
  raw before/after 相同，cgroup peak、冻结 wheel/native identity 均通过。
- `stage05.2_benchmark_attempt13` 在 batch0001 运行时因 unrelated process 平均占用
  达到一个完整 CPU core 被环境 guard 中止；该 identity 保留为 startup/runtime
  failure，不归类为 36% persistence 失败。后续运行在 measured interval 内不再启动
  额外诊断 shell。
- `stage05.2_benchmark_attempt14` 在创建测量 batch 前 fail fast：同一 Windows edition
  的 CIM `Caption` 在英文 `Microsoft Windows 11 Pro for Workstations` 与中文
  `Microsoft Windows 11 专业工作站版` 之间变化，逐字 machine-identity 比较因此
  拒绝启动。CPU/GPU、Windows Version/BuildNumber、WSL、mount UUID 和 NVMe identity
  均未变化；G14 不进入性能判定或独立 review。
- 当前实现不再把本地化 `Caption` 写入新 producer machine identity，改用语言无关的
  数值 CIM `OperatingSystemSKU`，并继续精确绑定 `Version`、`BuildNumber` 与
  `TotalVisibleMemorySize`。缺失、布尔值、非整数或非正 SKU 均 fail fast；历史 evidence
  的中英文 Caption reviewer compatibility 不变。后续必须从新 clean revision 重跑
  F，并用新 label 执行 G pilot/Formal 完整链路；36% 门槛及归因公式不变。

## 2026-07-24：F11 通过、G11/G12 保留失败并复用 negative-result identity

- clean revision `9d72dd9` 上的 `stage05.2_accelerator_pilot_attempt11` 已由独立
  transient service replay 通过，状态为 `READY_FOR_STAGE052_BENCHMARK`。raw/review
  manifest SHA-256 分别为
  `c9bf0a675e8bd8213c9d08620eadfc55af110da1ed3620c87078a2b89318da85`
  和 `be52182cacb8d53b0ccb16df7052171e6c164a5e12a3183f7865240332b06967`；
  cgroup、冻结 wheel/native identity 与 raw before/after 均通过。
- `stage05.2_benchmark_attempt11` 的 batch0001 ratio 为
  `0.32809368818248325`，随后 batch0002 因主机 `load1` 超过 formal guard `4.0`
  fail fast。该环境失败 identity 保留，不归类为 persistence 失败，也不复用 label。
- `stage05.2_benchmark_attempt12` 的 batch0001/0002 ratio 分别为
  `0.31947982337183994` 和 `0.3026241963675267`；batch0003 以
  `0.383628973309216` 被 36% hard gate 拒绝。失败 batch 的 solver 为
  `198.560791860` 秒，shard persistence 为 `123.576862186` 秒，control
  persistence 为 `0.007260724` 秒。G12 保持不可变，不进入独立 campaign review
  或 Formal。
- 剖析确认 100-customer screening occurrence 中约 85%--91% 是 negative-cache hit。
  ALNS 对同一路由返回同一个仍存活的 frozen `ScreeningResult`，但旧路径每次重新构造
  固定 cache-hit check，并让 writer 对同一 12-field evidence 重做完整 typed
  signature。当前实现复用唯一 immutable check tuple，并把 `id(result)` 作为仅限
  producer 进程内的 positive integer token。typed sink 用有界 262,144-entry
  `route_key -> token` map 验证身份稳定；变化立即 fail fast。manual/legacy rows 仍走
  完整字段、typed signature 与 collision check。
- 该 token 不写入 Parquet、definition identity、batch ledger、semantic digest 或
  review products；首次 occurrence 仍从完整 evidence 生成 canonical definition，
  reviewer 仍从 raw rows 独立重放全部语义。代表性单 shard probes 在 24 个 batch 下
  将 `r101_21/2014` ratio 从约 `0.3987` 降至 `0.3052`；`c101_21/2014` 和
  `rc101_21/2014` 分别为约 `0.2855`、`0.3482`。probe 不是晋级证据；后续必须在新
  clean revision 上重新执行 F 与完整 G producer/reviewer chain。

## 2026-07-24：G pilot attempt10 保留失败并修复稀疏 definition buffer rotation

- clean revision `fe35cd9` 上的 `stage05.2_accelerator_pilot_attempt10` 已由独立
  transient service replay 通过，状态为 `READY_FOR_STAGE052_BENCHMARK`。raw/review
  manifest SHA-256 分别为
  `7be1a39234804a6dd914113ea31addd240b0da178f550fc0b8261e43e27368ac`
  和 `69c624a47bc9e087c7448448a36c9b20375af62aae400b13ad9426fdf6911333`；
  review receipt 的 raw before/after 相同、cgroup peak 已验证且 exit code 为 0。
- `stage05.2_benchmark_attempt10` 的 batch0001/0002 分别以
  `0.335693353885035` 和 `0.3025066369792433` 通过并归档；batch0003 以
  `0.38743692510495514` 被 36% producer hard gate 拒绝。失败 batch 的 solver 为
  `195.487977799` 秒，shard persistence 为 `123.633580845` 秒，control
  persistence 为 `0.009622798` 秒。G10 保持不可变，不进入独立 campaign review
  或 Formal。
- 根因分析发现 v3 live transaction 每次先把稀疏 screening definition 写入一个
  partial Parquet row group，再在 occurrence/event 两个高流量 sink 之间执行
  two-buffer FIFO rotation；第三个 definition sink 会反复提前 flush occurrence 或
  event，产生大量不足 65,536 行的 row group。当前实现将已经通过 collision store
  的 definition transaction 立即写入，不让它进入高流量 bounded working set；
  两个受限非空缓冲稳定留给 occurrence 与 event。完整 definition/occurrence/event
  行、顺序、typed schema、SHA-256 identity、36% 归因公式及最大两个非空缓冲的资源
  上限均不改变。
- 新回归以三个相邻 mixed transactions 验证稀疏 definition 不再轮换两个高流量
  buffer，并继续验证 occurrence row-group 上限和 simultaneous-buffer 上限。ext4
  完整测试为 734 passed；Ruff、Mypy 与 `git diff --check` 通过。后续 F/G 必须使用
  新 clean revision 和新 label；G10 不得重用。

## 2026-07-23：G pilot attempt09 保留失败并引入 exact-key native cache

- clean revision `7a8c0ed` 上的 `stage05.2_accelerator_pilot_attempt09` 已由独立
  service replay 通过，decision 为 `GPU_NOT_JUSTIFIED`，状态为
  `READY_FOR_STAGE052_BENCHMARK`；raw/review manifest SHA-256 分别为
  `e8f0184f210dac173bfc4ad1603e17fca964b0d6469d72039bfde468d59168c9`
  和 `a14e464c2db920e28046ed593b65702a149548c120af211b222e1aa3c09cae0c`。
- `stage05.2_benchmark_attempt09` 的 batch0001/0002 分别以
  `0.33753112465519763` 和 `0.3103083675151225` 通过；batch0003 以
  `0.3927620670953848` 被 36% hard gate 拒绝。失败 batch 的 solver 为
  `194.144702373` 秒，shard persistence 为 `125.565650103` 秒，control
  persistence 为 `0.007326300` 秒。G09 保持不可变，未进入独立 campaign review
  或 Formal。
- profile 将剩余 screening producer 热点定位为每条记录重复构造并哈希 Python
  occurrence tuple。当前实现使用 shard-local native capsule 持有有界 262,144-entry
  exact-key cache。独立代码审查在正式运行前发现 Python equality 会把 `True/1.0`
  及 `0.0/-0.0` 错误合并；该版本没有生成 F/G identity。修正实现以 canonical typed
  signature 区分 bool/numeric union 和 IEEE-754 signed zero，并对所有 hash bucket
  继续执行精确字段比较；negative-evidence drift 使用同一 typed signature。
- Native cache 默认预留但硬限制为 262,144 entries，FIFO node pointer 在 rehash 后
  保持有效；测试专用的小容量仍不得超过该硬上限。仅 cache miss 保存一份 owning
  tuple 并继续生成 canonical sorted JSON、完整 SHA-256 与 typed definition row。
  新回归覆盖 forced string-hash collision、bool/float、signed zero、FIFO eviction、
  typed negative-evidence drift 和非法 capsule。四 worker contention probes 仍受主机
  调度噪声影响，观察到约 `0.367--0.386`；这些 probe 不替代后续不可变 F/G
  producer/reviewer evidence。

## 2026-07-23：G pilot attempt08 保留失败并去除 route-key 重复解析

- clean revision `24c6b3d` 上的 `stage05.2_accelerator_pilot_attempt08` 已由独立
  service replay 通过，decision 为 `GPU_NOT_JUSTIFIED`，状态为
  `READY_FOR_STAGE052_BENCHMARK`；raw/review manifest SHA-256 分别为
  `4f25aa93ae639936fd46d66dba49c39620313b4533ce645942d59bbb390557d8`
  和 `e8a9a3703745f70d9465b666789e0f5585f5952c236d925441eaa4000530d916`。
- `stage05.2_benchmark_attempt08` 的 batch0001/0002 分别以
  `0.3285104105625469` 和 `0.30459863419342853` 通过；batch0003 以
  `0.3774904401730277` 被 36% hard gate（硬门槛）拒绝。失败 batch 的 solver 为
  `198.709423171` 秒，shard persistence 为 `120.490075623` 秒，control persistence
  为 `0.007523868` 秒。G08 保持不可变，未进入独立 campaign review 或 Formal。
- native screening transaction 与 deferred sparse event packer 现在在首次观察 route
  时返回已经严格验证的 customer sequence；writer 不再把同一个 canonical route key
  在 Python 中第二次 split/length-check。普通非 negative-cache occurrence key 改用
  单一 flat exact tuple，避免为约五十万条 screening record 额外分配嵌套 evidence
  tuple；negative-cache token、完整 evidence drift 检查、definition SHA-256、事件顺序
  与 Parquet schema 均不变。
- campaign reviewer 不再硬编码过期的 0.05 秒 writer switch interval，而是验证
  producer 公开的同一协议常量；这修复了 producer 记录 0.5 秒但 reviewer 必然拒绝的
  contract drift（契约漂移）。后续仍须以新 F/G identity 运行完整 producer/reviewer
  链，诊断 probe 不构成通过证据。

## 2026-07-23：G pilot attempt07 保留失败并消除 definition 双重物化

- clean revision `01a9f87` 上的 `stage05.2_accelerator_pilot_attempt07` 已由独立
  service replay 通过，decision 为 `GPU_NOT_JUSTIFIED`，状态为
  `READY_FOR_STAGE052_BENCHMARK`；raw/review manifest SHA-256 分别为
  `297a725628a7f77bf7fc706e3756c7a5e1582de132a9a5e1ff6dd365a92bac4d`
  和 `e816fd9383ab22f200e099ced60147e380f67ec69f71d00664f57c81039746b7`。
  receipt 的 raw before/after 相同、cgroup peak 已验证、exit code 为 0。
- `stage05.2_benchmark_attempt07` 的 batch0001/0002 分别以
  `0.35311510035046867` 和 `0.3137778929854619` 通过并归档；batch0003 以
  `0.3915422769640815` 被 36% gate 拒绝。失败 batch 的 solver 为
  `195.835908036` 秒，shard persistence 为 `126.012511109` 秒，control
  persistence 为 `0.007809516` 秒。G07 保持不可变，未进入独立 campaign review
  或 Formal。
- profile 证明 transaction packer 虽已移除逐 miss Python control flow，但每个新
  definition 仍先在 C++ 构造 compact-check tuple 和 typed check dict，再调用 Python
  重新构造第二份 canonical payload/check dict。当前实现直接从已经验证的 typed fields
  构造唯一 canonical payload，用该对象同时生成 sorted JSON/SHA-256 和 Parquet row；
  不再创建中间 cache-key payload。`r101_21/2014` profile 中 native transaction
  packing 从约 `5.42` 秒降至 `3.55` 秒，writer append 从约 `8.19` 秒降至
  `6.19` 秒。
- high-cardinality definition Parquet 与 events/occurrences 一样关闭 dictionary 和
  statistics；Zstandard level 1、65,536-row group、typed nested checks、row order、
  semantic digest 与完整 replay 均不变。四进程 100-customer 争用探针仍显示机器抖动，
  因而这些数字不是通过证据；后续必须以新 F/G identity 重跑完整链。

## 2026-07-23：G pilot attempt06 保留失败并原生封装 screening transaction

- clean revision `4be5425` 上的 `stage05.2_accelerator_pilot_attempt06` 已完成独立
  replay，decision 为 `GPU_NOT_JUSTIFIED`，状态为
  `READY_FOR_STAGE052_BENCHMARK`。`stage05.2_benchmark_attempt06` 的 batch0001
  与 batch0002 分别以 `0.3468719322` 和 `0.3168614722` 通过 36% gate 并归档；
  batch0003 以 `0.37722766418417175` 被正确拒绝。该 batch 的 solver 为
  `199.819570929` 秒，persistence 为 `121.035353796` 秒，其中 shard persistence
  为 `121.027650956` 秒、control persistence 为 `0.007702840` 秒。attempt06
  保持不可变，未送独立 campaign review，Formal 未启动。
- batch0003 的九个 100-customer shard 各有约 60.9 万至 107.4 万条 screening
  occurrences，以及约 4.7 万至 12.5 万条首次 definition。此前 native prepass
  只批量发现 occurrence cache miss，随后仍逐 miss 回到 Python 重建 evidence tail、
  compact checks、lane/operator/route identity、definition payload 和 occurrence
  binding。新的 native screening transaction packer（原生筛选事务打包器）在同一
  FIFO batch 内一次完成这些 miss 事务，并返回 schema-ordered occurrence columns、
  typed definition rows 和需要进入 Python collision store 的 canonical payload。
- 新路径仍对 negative-cache evidence drift、八项 check 上限、stable ID、canonical
  sorted JSON、SHA-256 definition identity、route collision registration 和 unknown
  event fallback 执行 fail fast；没有删除、聚合或重排事件，也没有改变 36% 归因公式。
  duplicate occurrence 在同一批次只生成一个 definition，回归测试逐字段比较 native
  row 与 canonical Python payload/identity。
- 同一 `r101_21/2014` 无 profiler 对照中，旧冻结 `4be5425` runtime 的 persistence
  为 `16.149927407` 秒，新 transaction packer 为 `14.422620998` 秒；c101 profile
  中 writer CPU 由约 `7.53` 秒降至 `6.22` 秒。两者只是修复诊断，不是晋级证据；
  后续必须以新 clean revision 重跑 F，再用全新 G label 完成 producer gate 和独立
  campaign replay。
- 首次 F07 preflight（预检）在创建 raw 目录前重放 E15 runtime，发现 Windows CIM
  `Caption` 从中文 `Microsoft Windows 11 专业工作站版` 变为英文
  `Microsoft Windows 11 Pro for Workstations`；Version、BuildNumber、内存、CPU、
  GPU、WSL、kernel、mount 和 NVMe identity 全部相同。这是 PowerShell locale
  presentation drift（区域语言呈现漂移），不是机器变化。runtime comparison 只把这
  两个精确 edition alias 规范成同一 canonical value；其他 Caption、Version 或硬件
  变化仍 fail fast。由于预检没有创建 output directory，F07 label 尚未消耗。

## 2026-07-23：G pilot attempt05 保留失败并批量列式封装 sparse events

- clean revision `90c36ff` 上的 `stage05.2_accelerator_pilot_attempt05` 已由独立
  reviewer 完整复核，decision 为 `GPU_NOT_JUSTIFIED`，状态为
  `READY_FOR_STAGE052_BENCHMARK`。随后 `stage05.2_benchmark_attempt05` 在首个
  12-shard batch 被 36% producer gate 正确拒绝：solver 为 `23.187877539` 秒，
  persistence 为 `14.957757702` 秒，ratio 为 `0.392122391`。失败 evidence 保留，
  Formal 未启动。
- attempt05 已消除 route-evaluation 的 producer/writer 重复物化，但 writer 仍逐条
  建立约九万条 route-evaluation/cache-event sparse rows，并逐列执行 36 次 Python
  append。当前实现增加 native deferred sparse-event column packer（原生延迟稀疏事件
  列封装器）：同一 mixed batch 内的 route evaluation 和 cache event 直接生成完整
  `EVENTS_SCHEMA` 列，普通事件只保留原位置 placeholder 并继续走通用 normalizer。
  writer 在原位置回填普通事件后一次性提交列，因此 event ID、FIFO 顺序、route/lane/
  operator identity、extras presence、semantic digest 与 Parquet schema 均不变。
- native packer 只缓存 canonical extras JSON 和已经解析的稳定 ID；每批观察到的 route
  identity 仍回到 Python collision store 做完整 SHA-256 注册，未知 tuple、未知 marker
  和普通 mapping 均不被静默解释。新增 real-writer mixed-order 回归覆盖
  `route_evaluation -> operator_call -> cache_event` 的 event ID、lane、evaluation 和
  lookup-result 重放。
- `c104C10/2014` 三次单 shard 诊断 persistence 为 2.226010、2.246415 和
  2.265286 秒，对应 ratio 为 29.4264%、29.6320% 和 29.8214%；相较 attempt05 的
  同类 shard 约 2.68--3.04 秒已有明确余量。该诊断仍不是晋级证据；下一步必须用新
  F/G label、冻结 runtime 和独立 raw replay 验证完整 pilot。

## 2026-07-23：G pilot attempt04 保留失败并移除 route-evaluation 重复物化

- 当前 revision 重新生成并独立复核
  `stage05.2_accelerator_pilot_attempt04`；其 decision-only 证据仍为
  `GPU_NOT_JUSTIFIED`，全部 gate 通过并报告 `READY_FOR_STAGE052_BENCHMARK`。
  `stage05.2_benchmark_attempt04` 的首个 12-shard batch 随后被 36% producer gate
  正确拒绝：solver 为 `23.781636617` 秒，persistence 为 `16.071832465` 秒，
  ratio 为 `0.403273111`。失败 raw evidence 保留，Formal 未启动。
- attribution 显示 c104C10 的 route-evaluation（路径评估）高基数记录仍在 producer
  逐条建立 dictionary、writer 再逐条解析为相同的 sparse schema row（稀疏模式行）。
  streaming sink 现在提交 immutable flat deferred route-evaluation（不可变扁平延迟
  路径评估）；writer 直接构造 schema-ordered tuple，并有界缓存只含 benchmark axis、
  deadline 和 label counters 的 canonical extras JSON。事件字段、event ID、route ID、
  FIFO batch ledger、semantic digest 和 Parquet schema 均不变。
- exact unique-route identity（精确唯一路径身份）在 producer 持有 shard turn 时先行
  注册，使尚未到 row-group flush 的最后一批记录也能参与 solver reconciliation；
  writer 不重复注册。同一个 65,537-record 边界测试继续证明有界内存、精确计数和 ext4
  scratch 位置。
- `c104C10/2014` 单 shard 诊断从 attempt04 的 3.358311 秒 persistence 降至
  2.598002 秒，诊断 ratio 为 31.8991%。该诊断不是晋级证据；后续必须用新 G label
  重新执行完整 36-bundle pilot，并由独立 reviewer 确认。

## 2026-07-23：F/G 前置持久化热路径修复

- `stage05.2_benchmark_attempt03` 的 36% producer gate 失败后，继续从最早受影响的
  screening/neighborhood persistence（筛选/邻域持久化）热路径修复；没有启动 Formal，
  也没有复用 attempt03 label。
- v3 screening callback 现在提交 immutable flat deferred rows（不可变扁平延迟行），
  writer 通过 native typed-column prepass（原生类型列预扫描）批量绑定 occurrence；
  negative-cache evidence（负缓存证据）按 route identity 有界复用，并在证据漂移时
  fail fast。definition payload、完整 checks、SHA-256 collision check 和 occurrence
  顺序均保留。
- 内存中的 neighborhood block（邻域事件块）使用 native column packer（原生列打包器）
  直接生成 36 列 schema；包含普通事件的 mixed batch（混合批次）仍回退到原顺序
  normalizer，避免通过重排换取性能。未知字段也回退到通用路径，不会静默丢失。
- producer callback 的单调时钟纳秒仍逐次测量，但 cooperative shard turn 下的
  non-overlapping intervals（非重叠区间）按最多 4,096 次 callback 有界汇总后写入
  union meter；提交 batch、释放 writer turn、finish 或 close 前强制 flush。该修改只
  去除每条记录重复获取计量锁的成本，不改变累计纳秒或 36% 公式。
- writer 使用 0.5 秒 thread switch interval（线程切换间隔），producer callback 与
  writer 采用 cooperative turns（协作式执行权）；显式 finish/drain 中允许的重叠仍由
  union meter 只计一次。reviewer 从同一常量验证该物理协议。
- differential、negative-cache drift、FIFO ledger、real-writer round-trip、mypy 和 ruff
  回归均通过。代表性单 shard 诊断仍有机器抖动，不能替代正式 batch aggregate gate；
  后续 F/G 必须使用新 label，由 producer 和 independent reviewer 共同执行 36% 硬门。

## 2026-07-23：G pilot attempt03 拒绝错误重叠计量并确认剩余编码瓶颈

- `stage05.2_benchmark_attempt03` 在完成首个 12-shard batch 后由 producer gate
  拒绝：solver 为 `21.586026866` 秒，persistence 为 `17.875972692` 秒，
  batch ratio 为 `0.452992066`，超过当前 36% 门槛。该失败证据保留，label 不复用，
  Formal 未启动。
- attempt03 暴露 `fdc7227` overlap 改动后的 attribution bug：producer callback
  仍调用只适用于 non-overlapping interval（非重叠区间）的 `record_serialized()`，
  同时 writer 已允许并发，因此 producer/writer 同时活跃的墙钟区间会重复计数。
  producer callback 现在与 writer 一样通过同一个 activity meter 的 `enter/exit`
  记录；union 只计算一次重叠区间，producer/writer 各自 wall time 仍完整保留。
- 普通 critical events 不再先积累 row tuples 后二次转置，而是直接进入 typed column
  buffers；`EVENTS_SCHEMA` 的高基数 event ID、timestamp 和 extras 不再启用 Parquet
  dictionary/statistics，Zstandard level 1、65,536-row group、完整事件、FIFO 顺序、
  semantic digest 和 queue bound 均不变。
- 独立诊断仍证明 G 未达到晋级条件：`c101_21/2014` 的 30-second axis 产生
  1,063,145 条 screening decisions，正确计量后的 solver 为 `13.213825385` 秒、
  persistence 为 `17.934538759` 秒。不得据此启动新的 G pilot/Formal；下一次正式
  attempt 必须先证明 100-customer typed encoding 能把代表性 persistence ratio
  降到 36% 以下，且不能删除事件、重排 batch 或改写归因公式。

## 2026-07-23：G pilot attempt01 暴露 async producer/writer 串行化

- `stage05.2_benchmark_attempt01` 完成 12 个 pilot shard 后由 producer gate 拒绝：
  batch persistence ratio 为 `0.421537045`，超过修订后的 36% 门槛。该失败证据保留，
  label 不复用。
- attribution 显示 bounded async writer（有界异步写入器）仍在每次 solver callback
  开始前等待上一批写完，使 producer preparation 与 Parquet writer 无法重叠。以
  `c104C10/2014` 为例，194,918 个事件产生 3.284804 秒 solver-interleaved
  persistence，其中 producer active 为 1.914229 秒、writer CPU 为 2.026477 秒；
  两者原本可在 queue bound（队列上限）内安全重叠。
- async pipeline 现在只串行化 producer callback 本身；单 writer 仍按 FIFO 写入，
  queue 仍严格最多保留一个 waiting batch，失败仍 fail fast，finish/close 仍完整 drain。
  已提交 batch 的 event mapping 和 prepared screening definition 均为 owned/immutable，
  因此 writer 活跃时 producer 可准备下一批而不会共享可变 shard state。新增阻塞 writer
  回归测试证明下一批 producer preparation 不等待 active writer，且最终顺序保持不变。

## 2026-07-23：E native screening differential 采用诊断专用精度

- E 的独立 raw replay 显示 Python/native fixed-work 流的 event identity（事件身份）、
  route、status、reason、cache 行为和逐项 check 结论完全一致。唯一差异是 166 个事件中
  220 个派生诊断浮点值；最大绝对误差为 `5.684341886080802e-14`，最大相对误差为
  `9.9785060296947e-16`。
- storage semantic replay 现在只把 `screening_decision.min_time_window_slack`、
  `screening_decision.distance_lower_bound` 和数值型
  `screening_decision.checks[].value` 诊断量规范到小数点后十位。所有离散决策字段和所有
  非 screening 事件仍要求 exact equality（精确相等）；超出该 machine-roundoff
  envelope（机器舍入误差包络）的诊断变化仍会使 native differential gate 失败。

## 2026-07-23：E persistence gate 调整为 36% 并接受 attempt15

- 用户明确将 Stage 5.2 persistence-to-end-to-end ratio（持久化占端到端比例）硬门槛
  从 30% 调整为 36%。该调整发生在 `stage05.2_native_kernels_attempt15` raw evidence
  生成之后，必须作为显式 protocol revision（协议修订）保留，不能回写成原始 30% 规则
  下通过；此前按 30% 判定失败或未送审的 E13/E14 结论保持历史原貌。
- 唯一执行常量为 `STAGE052_MAXIMUM_PERSISTENCE_RATIO = 0.36`。producer、performance
  reviewer、campaign producer/reviewer、batch/campaign manifest 校验和 remediation
  默认值均引用该常量。新 persistence attribution 与 batch envelope 使用 v2 schema
  并声明 36%；v1 schema 仍按其原始 30% 字段只读校验，promotion（晋级）则使用当前
  36% 规则。
- E15 在 clean commit `8f148d9` 上完成 36/36 axis，solver 为 215.867958 秒，
  persistence 为 113.799960 秒，ratio 为 0.345195736；因此它低于 36% 新门槛，可进入
  独立 raw replay 并作为 E 候选通过证据。最终 E readiness 仍以重审生成的签名 review
  products（审查产物）为准。
- 首次重审在 `c101C5/2014/fixed_work_control` 暴露 reviewer-only ledger
  reconstruction bug（仅审查器的账本重构缺陷）：prepared v3 producer 直接把空
  `ScreeningDecision.reason` 字符串写入 batch token，而 expanded physical row（展开后的
  物理行）按 schema 表示为 null；两个 reviewer 错误地继续把 null 哈希为 null。逐字段
  重算证明把该 screening null 恢复为空字符串后，首批 27,194-row SHA-256 与 raw ledger
  完全一致。performance 与 campaign reviewer 现仅在 screening token 上恢复这一有损表示，
  producer/raw 均不修改；空 reason 回归测试覆盖实际 v3 round-trip。
- 第二次重审完整通过 36% persistence、validator/objective、fixed-work differential、
  resource、source snapshot 与 frozen producer runtime identity（冻结生产者运行时身份）
  等门，但旧 `native_producer_contract` 又把 E15 历史 producer 的 package/native
  identity 直接和含新审查修复的 reviewer wheel 比较，产生唯一的 `NOT_READY`。该重复比较
  与 raw-bound producer venv 独立重放合同冲突。performance provenance 现在只校验捕获值
  的内部一致性及其与已冻结 producer runtime identity 的 Python、dependency、native SHA-256
  和 CPU-count 绑定；完整 wheel、Python、native、dependency 与 machine identity 仍由独立
  frozen producer runtime gate 重放，新 reviewer wheel 不再冒充 producer。失败 review
  generation 和 service receipt 保留，修复后必须按 retry-history 协议产生新 review。

## 2026-07-23：E13--E14 mixed-batch columnar screening persistence

- 失败证据：`stage05.2_native_kernels_attempt12` 的冻结 source snapshot（源码快照）漏带
  ignored benchmark data，在创建 shard 前失败，空目录保留且 label 不复用。
  `stage05.2_native_kernels_attempt13` 在 clean commit `75cf816` 上完成 36/36 axis，但
  producer persistence ratio 为 0.347341265（solver 216.880358280 秒，persistence
  115.422492442 秒），未达到 30% gate，因此不提交独立 readiness review，raw evidence
  保持不可变。`stage05.2_native_kernels_attempt14` 在 clean commit `ea1733f` 上也完成
  36/36 axis，但把 canonical definition preparation 移入 producer callback 后削弱了与
  writer 的有效重叠，ratio 回退到 0.382957313（solver 203.324121904 秒，persistence
  126.189745317 秒）；该完整失败 evidence 同样保留且不进入独立 readiness review。
- 根因：大型 wall-clock shards 中 screening occurrences 占绝大多数；producer 已复用
  definition，却仍在 writer 内为每条 occurrence 构造 row tuple，随后再次转置成 Parquet
  columns。把 prepared screening 与其他事件按连续段拆开会破坏 batch 粒度，故未采用。
- 修改：每个 route 的 negative-cache evidence、definition 与最后一个 lane/operator prepared
  context 合并到一个有界 entry；其他 context 回退到既有 262,144-entry 全局有界 cache，避免
  route-local 字典绕过内存硬上限。`PreparedScreeningDefinition` 只绑定 typed tail 与顶层上下文；
  canonical payload、route/lane/operator identity、SHA-256 和 typed definition row 仍在 writer
  首次 cache miss 时构造，使 CPU 工作留在可与 solver core overlap 的线程。writer 保持原始
  mixed batch 与 FIFO ledger 不变，在同一遍扫描中把全部 screening occurrence 直接累积为
  schema-ordered columns，并继续按原 event ID 顺序写入。row-group、cache、transaction 和 batch
  上限均保持 65,536 或既有更严格边界；Prepared 不携带可与顶层身份交叉拼接的 pending row。
- 验证：artifact v3 与 streaming trace 定向测试 97 项通过；Ruff 与 `git diff --check` 通过。
  `c101_21/2014` 单 shard probe 的 wall-clock persistence union 为约 9.47 秒。正式门槛仍只接受
  后续新 clean commit 上完整 36-axis producer evidence 与独立 raw replay。

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

## 2026-07-23：G pilot attempt16 lane-local deadline replay 修复

- `stage05.2_benchmark_attempt16` producer 完成 36/36 pilot shards，三批 persistence
  ratio 分别为 `0.32856322689393336`、`0.2994853687778845` 和
  `0.33177058212924965`，campaign aggregate 为 `0.32904761287664286`，均未超过
  36% 门槛；raw manifest SHA-256 为
  `65c809ecc63b4a28d9f3fffc9548ce74605d07f18b2585403328e4ab8441bacf`。
- 首次 independent campaign review 发布 `NOT_READY`。最早失败为
  `batch0003/c101_21/2015` 的 “exact work started after deadline”；其余 geometry、
  persistence、storage、resource 和 publication failures 是该 shard 未进入 aggregate
  后的级联结果。
- 原始事件证明首个 boundary 位于 `29.919126434993814` 秒并属于
  `wall_clock_30:legacy`；其后 56 条 exact-start/cache-store 事件全部属于预留的
  `wall_clock_30:constraint_lane`，均在 30 秒总预算内。同一 lane 在自身 boundary 后
  的违规数为零。producer raw 因而没有 deadline violation。
- 根因是 campaign reviewer 用一个 axis-global `deadline_seen` Boolean 截断所有 lane，
  与 ALNS 的 `legacy_deadline = overall_deadline - 0.1s` 和 constraint-lane reservation
  冲突。reviewer 现在分别记录 lane-local boundary；同 lane 后续 exact/cache/accept
  仍是 hard failure，任何 exact completion 或 accepted candidate 超过 axis 总预算也
  仍是 hard failure。
- 回归测试先在旧状态机上稳定失败，再在修复后通过；修复版对原始 1,305,412 条逻辑
  事件完整重放通过，核对 1,365 次 started/completed exact calls、148 次 accepted
  candidates、6 次 global best 和 wall-clock deadline identity。producer、solver、
  objective、36% persistence gate 和 raw bytes 均未修改；首次 NOT_READY review
  generation 保留在 immutable history 后，以新 reviewer wheel 显式复审同一 sealed raw。
- lane-local 修复后的第二次 review 证明 shard replay、36/36 geometry、36% persistence、
  resource、runtime 和 power/load gates 全部通过，但暴露独立的 storage-root probe
  defect：campaign reviewer 无条件调用 macOS `diskutil`，所以 WSL2 上
  `wsl_staging` 与 `d_archive` 均无法探测，rolling-capacity drill 和 publication
  dry-run 随之级联失败。producer、retention 和 performance reviewer 已使用共享
  cross-platform `probe_volume_identity`；campaign reviewer 现在复用同一实现，WSL
  ext4 UUID 与 D: NVMe/9p identity 均重新核对。旧 reviewer generation 和 finalized
  receipt 继续保留，不改写 raw。
- cross-platform probe 修复后的第三次 review 已令 `storage_roots` 及此前通过的全部
  scientific gates 通过，但在 `pilot_campaign_drills` 暴露 reviewer 自身的
  rolling-capacity arithmetic drift：producer 的 canonical config 固定
  `external_safety=50 GiB`、`external_active_workspace=32 GiB`、
  `internal_safety=50 GiB`，而旧 reviewer 错误硬编码为 52/20 GiB。九份已签名
  observation 与 producer 逻辑一致：active/future 阶段 WSL staging reserve 为
  82 GiB，final safety reserve 为 50 GiB，D archive internal reserve 为 50 GiB。
  reviewer 现在从重建的 `BenchmarkCampaignConfig` 独立计算同一公式并验证
  campaign/config/alias identity，不降低任何容量门槛；第三个 `NOT_READY`
  generation 继续进入 immutable review history，sealed raw 不变。
- 第四次 service execution 完成全部 raw replay 后首次到达 pilot publication dry-run，
  但 isolated `python -I` runtime 报告 `No module named 'tools'`；前三个
  `NOT_READY` generation 因前置 gate 未全过而从未执行该路径。该 execution 以
  finalized failed receipt 保留，raw manifest 仍为
  `65c809ecc63b4a28d9f3fffc9548ce74605d07f18b2585403328e4ab8441bacf`，现有 review
  pointer/history 均未变化。根因是 wheel 只打包 `src/evrptw`，却在 READY pilot
  路径动态导入 tracked `tools.publish_stage052_artifacts`。reviewer wheel 现在同时
  打包 tracked `tools` package，seal source attestation 与 no-cache rebuild 对这些
  modules 逐文件验证；禁止从 checkout 手工复制或注入未密封工具。
- sealed-tools 修复后的下一次 review 已运行到 publisher，但 provisional manifest
  被拒绝为 `campaign review mandatory gate set is incomplete`。实际 gate diff 只有
  一项：reviewer 已独立验证并发布 `source_snapshot`，而
  `CAMPAIGN_PILOT_GATES`/`CAMPAIGN_FORMAL_GATES` 的 exact-set contract 漏登记该硬
  gate。现将 `source_snapshot` 加入 Pilot/Formal common mandatory gate set；这会
  收紧 publisher/prerequisite 验证，不删除 gate、不降低门槛。该次 `NOT_READY`
  generation 与成功 finalized receipt 均保留，raw manifest 不变。
- G16 18/18 gates 最终通过并发布
  `READY_FOR_STAGE052_FORMAL_BENCHMARK`；review SHA-256 为
  `7d87b8f7df84cadd04e73e5cc36a01c10b0cbf64496be70da175913091bbef73`，
  raw SHA-256 仍为
  `65c809ecc63b4a28d9f3fffc9548ce74605d07f18b2585403328e4ab8441bacf`。
  Formal lock preflight 随后暴露 generic current-chain verifier 错把 performance review
  专属的 `semantic_mismatches.csv` 强制用于 Benchmark campaign review；campaign
  已由 `verify_stage052_review_files` 验证完整 11-file content-addressed generation，
  不生成该 performance comparison 文件。verifier 现在按 component 区分：
  performance 仍严格要求三文件 generation，Benchmark 要求已验证的完整 campaign
  publication surface；不减少任何 campaign artifact 或 gate。
- G17 `stage05.2_benchmark_attempt17` 在创建 raw 目录和启动 solver 前被 Formal
  predecessor loader fail-fast 拒绝：G16 review 的嵌套 `selection_lock` 已记录
  `selected_optimization_profile=native`，但顶层 review manifest 漏发该冻结字段。
  attempt17 不复用。根因修复是在 campaign reviewer 的顶层 publication surface
  同步发布该字段；loader 继续要求顶层与 metadata 精确一致，不从嵌套值兜底。
- `stage05.2_benchmark_attempt20` producer 完成 36 shards，但独立 replay 在
  `r101_21/2016` 发现一次 wall-clock exact transaction 于 29.999819 秒开始、
  30.000254 秒返回；producer 已丢弃 candidate/cache，却错误保留
  `exact_completed=True`。review 正确发布 `NOT_READY`。修复将所有在 lane deadline
  时或之后返回的单次/批量 exact transaction 原子记为 interrupted，并清除 completed
  counters、route identity 与 candidate/cache 影响；不增加 deadline 容差。

## 2026-07-24：G21 Pilot 通过，G22 Formal 自身负载误判修复

- `stage05.2_benchmark_attempt21` 已完成 36/36 Pilot axes，independent campaign
  review 的 18/18 gates 全部通过，状态为
  `READY_FOR_STAGE052_FORMAL_BENCHMARK`。campaign aggregate persistence ratio 为
  `0.335996595`；review receipt finalized、raw before/after hash 相同、cgroup peak
  可用且 service exit code 为 0。
- `stage05.2_benchmark_attempt22` Formal preflight 两个连续窗口的最大 `load1` 分别
  为 `0.76416015625` 和 `0.5302734375`，unrelated process average cores 均为 0；
  batch0001 启动四个已选择 worker 后，runtime guard 却以 `load1 exceeded 4.0`
  中止并封存 partial evidence。该 label 已消耗，不得续跑或导入 shard。
- 根因是 runtime guard 把 campaign 自己的四个 worker 也计入固定的 idle-host
  `load1 <= 4.0` 阈值；同一监视器其实已经按 PID tree 排除 campaign descendants，
  单独测量 unrelated user CPU。修复保留 preflight 的 4.0 hard gate，batch 运行中
  使用可重放的 `4.0 + selected_workers` 总负载上限；unrelated process 完整窗口一整核、
  AC power 和 low-power mode 仍分别 fail fast。
- runtime monitor evidence 现在显式记录 `maximum_permitted_load1`；power/load
  artifact 只记录实际采样，reviewer 从冻结 worker 数独立重建
  `4.0 + selected_workers` 并比较实际 maximum，避免信任 producer 自报阈值。
- 为避免 G-only 修复误触发 F 重跑，execution lock 允许 F revision 的 Git
  descendant，但逐路径 diff 必须完全落在显式 G runner/reviewer/test/documentation
  allowlist；producer 与 independent reviewer 各自重放该检查。Python、依赖、机器、
  source mount、native extension、配置、实例、backend 和 worker 仍由稳定 runtime
  selection hash 冻结，solver/objective/native/其他 source drift 直接 fail fast。
  后续只从新的 G Pilot identity 继续，不重跑 C--F。

## 2026-07-24：G25 Formal SQLite 跨线程 spill 根因修复

- `stage05.2_benchmark_attempt25` 在 batch0001 运行约三小时后失败；首个真实错误为
  `c103_21/2015` 的 SQLite thread-affinity（线程亲和性）拒绝。此时 route identity
  累计首次超过 262,144-entry 内存上限并进入 disk spill；连接由 producer thread
  创建，单一 async artifact writer thread 在 shard-turn lock 保护下接管写入，但旧
  connection 仍启用 SQLite 默认 same-thread 检查。
- 修复只允许已经由 shard-turn lock 严格串行化的 producer/writer ownership handoff；
  route identity 与 screening definition 两个 spill connection 使用
  `check_same_thread=False`，不允许并发 SQL、不改变 row identity、schema、objective、
  solver、native kernel、backend 或 worker selection。
- 新回归测试把 route/unique identity 与 screening definition 的内存阈值压到 1，
  强制 writer thread spill，再由 producer thread 读取、去重和关闭，证明跨线程串行
  handoff 及 scratch cleanup。G25 及其 6.2-GB partial evidence 保持 failed，不复用；
  后续从新的 G Pilot 标签重试，不重跑 A--F。

## 2026-07-25：G26 reviewer 内存隔离与归档盘迁移证明

- G26 首次 campaign review 在 batch0003 累积到 5.5 GiB reviewer RSS 后失败。根因是
  同一长寿命 reviewer process 对每个 shard 三次重放 critical events，并跨 shard
  保留 Python、PyArrow 与 SQLite allocator 高水位。campaign reviewer 现改为每 shard
  一个全新、严格串行的 `spawn` child，并以一次 logical event pass 同时完成
  persistence、transaction/deadline 与 global-best/checkpoint 审计。
- child 重新验证 signed batch/shard manifest，只返回有界 JSON-safe summary；parent
  验证完整 run/batch/shard/instance/seed 身份、single-pass count、PID、RSS 和 scratch
  cleanup。异常 child 不重试、不回退到 parent replay。真实 G26 只读 harness 已完成
  36/36 shards、16,295,563 events、36 个不同 child PID，parent peak RSS 为
  89,194,496 bytes。
- G26 sealed raw 随 D: 从 Samsung SSD 990 EVO Plus 1TB 迁移到 ZHITAI TiPlus7100s
  2TB；raw 中旧物理卷身份不可改写。新增 signed storage migration attestation，
  绑定旧/新物理磁盘和 volume identity、campaign/standard raw manifest SHA-256，
  以及 batch0001--0003 的目录 checksum 与 byte count。reviewer 仅在全部内容重验
  一致时允许 `d_archive_disk` 这一项变化；其他 machine/runtime/source identity
  仍严格匹配。
- retrospective review 新增显式 `--producer-source-dir` 与
  `--storage-migration`。前者验证 G26 的只读 producer source snapshot，当前 reviewer
  source 继续由密封 wheel 独立证明，避免用 reviewer revision 冒充 producer revision。
- bounded review service 同步增加独立 `--producer-source-directory` execution-envelope
  输入；receipt 的 `producer_repository_revision`/`working_directory` 绑定历史 producer
  snapshot，新增 `reviewer_working_directory` 绑定当前 reviewer snapshot。两者分别验证
  clean/read-only source 和 wheel provenance，禁止把 reviewer revision 重复填入
  producer 字段。
- successful campaign review 在顶层 manifest 发布 migration attestation 及 sidecar
  SHA-256。Formal producer 必须显式提供同一 `--storage-migration`，并在 current-chain
  prerequisite replay 前核对该 SHA-256；未绑定或替换的证明均 fail fast。用于
  非-Benchmark prerequisite 的证明默认拒绝，唯一例外是下文经过历史 G campaign
  完整复验的 successor Benchmark Pilot。
- 一次 WSL lifecycle interruption 在 315.5 MiB cgroup peak 时终止 reviewer；失败
  receipt 正确记录 `service_interrupted`，但旧 exception path 把 review manifest 的
  standard raw SHA-256 留空，使下一代无法归档该 `NOT_READY` pointer。失败路径现在
  独立重读标准 raw manifest；历史兼容只接受 `NOT_READY`、同一 campaign SHA、完整
  generation 文件、明确 failed `campaign_replay` 且空 raw hash 的已知形态，归档后
  继续 append，不改写失败代次，也不允许其成为 prerequisite。
- Formal attempt27 在 raw 创建前因 transient service 漏传 Windows/WSL executable
  PATH 而失败；attempt28 补齐 PATH 后又在 raw 创建前被 successor allowlist 拒绝。
  allowlist 现仅新增本轮实际变更的 reviewer、review service、migration attestation、
  platform guard 及对应测试/文档路径；solver、objective、validator、native producer
  algorithm 与配置仍不在允许范围。attempt27/28 的 service logs 保留，均不复用。
- Formal attempt29 在 raw 创建前由 runtime selection lock 拒绝。根因是 prerequisite
  verification 已验证 signed D archive migration，但 successor runtime selection
  comparison 未接收同一 migration；同时新建环境解析了不同的间接依赖版本。修复后
  仅当当前 D archive disk 精确等于 attestation destination 时，比较前将该单一字段
  规范化为 attestation source，其余 runtime 字段继续冻结；Formal 环境复用 G26 的
  精确 dependency set，仅替换当前 sealed wheel。attempt29 不复用。
- Formal attempt30 在 raw 创建前再次由同一 lock 拒绝。逐字段比较确认依赖完全一致，
  唯一剩余差异是 WSL 报告的总内存相差一个 4 KiB page。campaign selection 现与既有
  producer replay/storage migration 合同一致，仅排除易波动的
  `machine_identity.memory_bytes`；CPU、GPU、OS、kernel、mount、disk（经签名迁移
  规范化）、dependency、Python/native identity 仍全部冻结。attempt30 不复用。
- Formal attempt31 已通过 selection lock，但 Windows-side keepalive 参数引用失效，
  WSL lifecycle 在 raw 创建前结束；unit 随 system shutdown 正常收束且未创建 campaign
  目录。随后以独立 stop sentinel 的持久 keeper 验证连续心跳，attempt31 不复用。
- Formal attempt32 通过完整 preflight 并进入 batch0001，但在 worker 启动后 4.69 秒
  被 runtime guard 拒绝：Codex 同时执行的递归 WSL `find/sort` 进度检查成为 unrelated
  full-core process。守卫未降低，partial evidence 已密封；后续 producer 运行期间禁止
  启动 WSL 监控进程，只允许 Windows UNC 轻量只读 manifest 检查。attempt32 不续跑。

## 2026-07-25：G33 跨 batch 短窗口 PID 消失假阳性修复

- Formal `stage05.2_benchmark_attempt33` 的 batch0001 完成 405/405 shards，并以
  `cross_volume_verified_copy` 将 12,351,248,493 bytes 原子归档到新 D 盘；归档目录
  checksum 为 `98e6a2ea5ee985a4864a8cfccd552f59f944e2c6479cb0aa2e3f1b7f588dc492`。
  batch0002 启动约 2.1 秒后，runtime guard 报告 unrelated process 平均占满一核并
  fail fast；attempt33 保持 failed，不续跑、不导入已完成 shard。
- 失败与 lifecycle keeper 的固定 30 秒边界重合。keeper 的 `sleep` 子进程几乎不消耗
  CPU，但旧实现对刚消失 PID 的未知尾段按“全部逻辑 CPU × 一个采样间隔”计入上界。
  在尚不足一个完整观察窗的 2.1 秒新 batch 中，该上界必然超过 1 core，因而把空闲
  短生命周期进程误判为干扰。abort 报告中的 PID 14797 由 resource summary 证明是
  campaign 自身 worker，只是 kill 后未在旧 4 秒等待内完成回收，并非 offending PID。
- runtime unrelated-process gate 现使用连续完整 30 秒 rolling windows；不足 30 秒时
  AC power、low-power mode 与 `load1 <= 4.0 + selected_workers` 仍即时 fail fast，
  只有需要“平均一整核”语义的 CPU gate 等待首个完整窗口。之后每个新采样都重放最近
  的完整窗口，持续一整核的外部负载仍在 30 秒时被拒绝，门槛未提高。
- independent reviewer 对同一 runtime samples 使用相同的窗口切分并独立重算最大值；
  raw schema 与 power/load publication surface 不变。失败 batch 现在也持久化已取得的
  runtime evidence，避免只留下摘要错误而无法重放触发窗口。

## 2026-07-25：迁移后 successor Pilot 双重证据绑定

- G33 暴露 producer defect（生产器缺陷）后，协议要求先以新标签重跑 Pilot，再启动
  Formal。该 Pilot 的科学 prerequisite（前置条件）仍是 accepted F17 accelerator
  pilot，不能把 G26 benchmark review 冒充成 F prerequisite；但 F17 的 selection
  lock 又冻结了更换前的 D 盘身份。
- benchmark Pilot 现在可同时显式提供 `--storage-migration` 和
  `--storage-migration-evidence-dir`。后者必须精确指向 attestation 声明的历史 G
  campaign，系统会重验 signed campaign manifest、campaign/raw manifest SHA-256、
  每个归档 batch 的 checksum/byte count、accepted Pilot review，以及 finalized
  successful review receipt。只有整条链成立，迁移证明才可用于 F prerequisite 的
  machine identity normalization（机器身份规范化）。
- 该例外仅适用于 `benchmark/pilot` 接受 `accelerator_pilot` prerequisite 的场景。
  Formal 仍要求其 Benchmark Pilot prerequisite 自身在 review manifest 中绑定同一
  migration SHA-256；缺失、替换、失败或未最终化的历史 review 均 fail fast。
- `stage05.2_benchmark_attempt36` producer 完成 36/36 shards 后，首代 reviewer 因仍把
  G26 migration attestation 与当前 successor campaign 强行做 run-label/batch-set
  绑定而给出级联 `NOT_READY`。reviewer 现在接受同一显式
  `--storage-migration-evidence-dir`，重验历史 G26 campaign 后再把 migration payload
  用于 F prerequisite 与 successor runtime selection replay；不得把旧 attestation
  直接绑定到 Attempt36。Attempt36 sealed raw 不变，只追加新 review generation。
- 第二代复审证明 prerequisite 已恢复，但 batch provenance replay 仍未把已验证的
  migration payload 传入 runtime-selection hash，因而把新 D 盘误报为 F17 selection
  drift，并再次级联为 0 shards。`campaign_runtime_selection_sha256` 现提供同一显式
  migration normalization 路径；只允许 destination disk 精确替换为 attested source
  disk，memory 仍按既有规则排除，其余 runtime 字段全部参与 hash。Attempt36 继续只
  追加 review generation。

## 2026-07-26：Attempt51 候选事务截止边界修复

- `stage05.2_benchmark_attempt51` 在完成并归档 batch0001--0005 后，于 batch0006
  fail fast。campaign manifest 保持 `failed`，已完成的 557/920 shards 与全部 partial
  evidence 不续跑、不导入后续 run。失败不是内存或负载：systemd process-tree peak
  为 14.6 GiB，准确异常为 `accepted global-best event exceeds the axis budget`。
- 真实事件重放定位到 `r106_21/2016` 的 `wall_clock_60`：legacy candidate 在
  59.082 秒完成 route/cache 工作，随后 constraint lane 在 60.010 秒记录
  `deadline_boundary`，旧控制流仍于 60.010851 秒提交该 legacy candidate 为
  accepted global-best。该 10.85 ms 越界证明根因是 candidate transaction
  commit（候选事务提交）缺少最后截止检查，不是浮点误差、日志延迟或 reviewer
  阈值过严。
- legacy、quality-shadow 与 constraint-lane 现在都在 acceptance decision 之后、
  incumbent/global-best/statistics mutation 之前执行严格的 commit deadline 检查。
  到达或超过 deadline 时记录 `before_candidate_commit`，将事务作为 time-limit
  rejection 保留，禁止更新 current/global-best；reviewer 的严格预算检查保持不变。
- 新的确定性回归测试让 acceptance decision 恰好跨过 deadline：修复前观察到
  `accepted_moves == 1` 并失败，修复后要求 0 次接受、1 次拒绝和
  `wall_clock_deadline`。按 producer-defect 治理，Attempt51 永久保留；修复完成后
  必须先以最低未占用标签跑新 Pilot 并独立复审，再以另一新标签从头执行 Formal。
- Attempt52 因启动命令误用 prerequisite role `accelerator_pilot` 而在 raw 创建前
  失败；Attempt53 使用正确的 `accelerator_decision` 后，又由 successor revision
  gate 在 raw 创建前拒绝这两个非 G 路径。两者均不复用。治理锁现不把 `alns.py`
  泛化加入 G allowlist，而是同时要求 solver 与回归测试两个路径完整出现，并将
  current Git blob 的 SHA-256 固定为本次已审计内容；缺失任一路径或未来任何字节漂移
  都 fail fast。该窄例外只用于执行协议本身要求的“producer defect 修复后新 Pilot”。

## 2026-07-26 至 2026-07-27：Attempt58、Formal 主机合同与 SQLite 事务修复

- `stage05.2_benchmark_attempt54` 在候选事务截止修复 revision 上完成 36/36 Pilot
  shards 并通过 independent review，但随后真实 Formal 暴露主机负载合同缺陷，因此
  该 Pilot 作为完整、不可变的中间 evidence 保留，不再作为 current Formal
  prerequisite。Attempt55 因 producer source-root identity 不匹配在 raw 创建前
  fail fast，不复用。
- `stage05.2_benchmark_attempt56` Formal 在 batch0001 运行时被
  `load1 exceeded 20.0` 拒绝。失败 evidence 显示
  `maximum_load1=21.60009765625`、unrelated process average cores 为 0、AC power、
  low-power mode 关闭、24 个 logical CPUs，systemd peak 仅 97.4 MiB。根因是
  runtime ceiling 低于这台 24-thread 主机的正常饱和 load，而非外部干扰或内存不足。
  campaign-start idle gate 保持 4.0；四 worker 启动后的 runtime guard 提高为 32.0，
  继续即时检查 AC、low-power、24 logical CPUs 与 unrelated-process 完整窗口，
  不改变 frozen `native_cpu + cpu_batch + 4 workers`。
- `stage05.2_benchmark_attempt57` 使用新的 runtime ceiling 完成并归档 batch0001--0002，
  但 batch0003 handoff preflight 把前一批自身造成、仍在指数衰减的 Linux 1-minute
  `load1=8.56982421875` 再次当作 campaign-start idle load 而拒绝。batch0001 runtime
  maximum 为 0.6875，batch0002 runtime maximum 为 8.56982421875，两个 batch 的
  unrelated process average cores 均为 0。修复后只有 campaign-start preflight
  执行两个窗口的 `load1 <= 4.0` idle gate；batch handoff 仍执行两个完整窗口并严格
  检查 AC、low-power、24 logical CPUs 与 unrelated-process one-core gate，但不再用
  campaign 自己上一批的全局 load history 拒绝下一批。batch 启动后继续执行 32.0
  runtime guard。
- 当前 Pilot `stage05.2_benchmark_attempt58` 在 revision
  `1012288283cba69c127c28b509eb4ecb34221baa` 完成 36/36 shards、36 axes、
  1,080 declared solver seconds 与 144 checkpoints。producer raw manifest
  SHA-256 为
  `de4e224735c07cdfdc262db63f34816cd934de0dcb1fefa22604e425b2a9f9dc`；
  independent review 逐 shard spawn 完成 36/36 replay，18/18 mandatory gates
  全部通过，aggregate peak RSS 为 529,690,624 bytes，raw before/after hash
  相同，finalized receipt 成功，状态为
  `READY_FOR_STAGE052_FORMAL_BENCHMARK`。review manifest SHA-256 为
  `f15e043b66780b39214d91be99cf7a175b6a78d02b8429661138fd531f54ace6`。
- Attempt59 的 Formal service 命令手工转录 prerequisite config SHA-256 时漏掉两个
  hex characters，campaign lock verifier 在 raw 创建前正确 fail fast；该 label
  不复用。后续 Formal lock 由
  `verify_stage052_evidence_input`、`upsert_stage052_campaign_lock` 与
  `verify_stage052_campaign_lock` 直接从 accepted Attempt58 evidence 程序化生成，
  禁止再手工转录 hash。
- `stage05.2_benchmark_attempt60` 完成并原子归档 batch0001--0007，在 batch0008
  完成 16/50 shards 后因 screening-definition SQLite scratch 报
  `database disk image is malformed` fail fast。该运行没有续跑或导入已完成 shard，
  全部 partial/failed evidence 原位保留。外部 cgroup 记录
  `MemoryPeak=21,495,746,560 bytes`、`MemoryHigh=20 GiB`、`MemoryMax=22 GiB`，
  `memory.events` 的 `max=0`、`oom=0`、`oom_kill=0`；内核日志没有本次运行期间的
  ext4/NVMe/I/O error，active ext4 仍有 745 GiB 与 94% inode 空余。因此失败不是
  hard memory kill、OOM、磁盘满或已观测的文件系统 I/O 故障。
- 根因审计发现 `_BoundedScreeningDefinitionStore` 使用 `journal_mode=OFF`、
  `synchronous=OFF`，首次 spill 后没有显式 commit，`register_many` 也没有事务
  rollback。故障注入红灯测试证明旧实现的第二条 INSERT 失败后，第一条已残留在
  definition store；另一条测试证明旧合同实际返回 `journal_mode=off`。修复后 scratch
  固定使用 `journal_mode=MEMORY`、`synchronous=NORMAL`，首次 spill 和每个
  `register_many` 都由 SQLite transaction context 执行，成功 commit、异常 rollback。
  两条红灯均转绿；`test_artifacts_v3` 66/66、Stage 5.2 定向测试 374/374、完整 pytest
  785/785、全仓 Ruff、strict mypy（70 source files）与 `git diff --check` 全部通过。
  250,000-row 高基数 spill 回归完成 `integrity_check=ok`，吞吐约 186,000 rows/s，
  peak RSS 91.7 MiB，退出后 scratch 清理。该 producer 修复必须先以新 Pilot 通过
  36% persistence 与 independent review，再以新 Formal label 重跑完整 scope。
- `stage05.2_benchmark_attempt61` 在 revision
  `823d7c860eb6473db46de6f6dff255e6b4ed5698` 完成 36/36 Pilot shards；三批
  persistence ratio 分别为 35.22%、31.23% 与 31.56%；signed batch resource
  evidence 的最大 process-tree aggregate peak 为 3,666,194,432 bytes（3.41 GiB），
  没有 warning/error。但 signed screening-definition descriptor 显示
  单 shard 最大仅 143,326 rows，低于当时 1,200,000-entry producer memory bound，
  因此该 Pilot 没有真实执行刚修复的 SQLite spill path，不能作为 Formal
  prerequisite，完整 raw 继续保留且不追加成功 review。
- 为关闭该验证盲区，producer bound 固定为 131,072 unique definitions；Attempt61
  的原始 row counts 证明相同 Pilot scope 会有三个 shard 越过该界限。spilled store
  关闭前现在强制执行 `PRAGMA integrity_check`，并以嵌套 `finally` 保证 connection、
  temporary directory 与内存索引在成功和异常路径都清理。campaign reviewer 从每个
  signed shard descriptor 读取 definition row count，并从 signed batch metadata
  读取 producer 当时的完整 store contract，要求所有 batch 精确绑定 131,072-entry
  bound 且至少一个 shard 超过该值；不得用未来 reviewer 自身常量倒推历史 producer。
  reviewer 同时拒绝 shard 目录中的任何 producer scratch 残留，包括空目录；否则现有
  `batch_shard_artifact_replay` mandatory gate 为 `NOT_READY`。
  `register_many` 的未 spill 内存路径也不再边验证边写入；它先完成整批去重与 collision
  预检，再一次性更新内存索引，确保后项失败不会留下前项。
  该变更不改变 solver、objective、validator、event/raw schema、四 worker 或 36%
  persistence gate；必须用下一最低未占用标签重新跑 Pilot。
- `stage05.2_benchmark_attempt62` 在 revision
  `9e168218d2e0a821bfb31beadc8bf594ba89910d` 完成 36/36 Pilot shards，
  三批 persistence ratio 分别为 35.36%、31.06% 与 31.79%，producer cgroup
  peak 为 3,671,543,808 bytes 且零 swap。36 个 screening-definition artifacts
  共含 1,184,415 rows，其中三个 shard 分别为 143,326、140,183 与 135,892 rows，
  真实越过 signed 131,072-entry bound；producer scratch 成功清理。因此 SQLite
  spill 与完整性合同已被真实 Pilot 执行。
- Attempt62 的第一代 independent review 保持 immutable `NOT_READY`：review service
  正常 finalized、raw before/after SHA-256 均为
  `aa717c59507267c40a097b4d287958da32f7639536ce795ca013be823232476a`，
  aggregate peak RSS 为 565,665,792 bytes；但在 batch0003 的
  `r101_21/2014` 发现 raw 声明 `wall_clock_deadline`、event stream 却没有
  `deadline_boundary`。同一缺口在 Attempt61 可重复，Attempt58 则有 12 条边界事件，
  故这是 producer 终止证据缺口，不是 reviewer 内存或 single-pass 解码错误。
- 根因是 `_solve_alns` 在最终 unified validation 后才按 `solve_completed_at`
  判定 `wall_clock_deadline`，而旧实现只在 exact/candidate 内部跨线时记录 boundary。
  当最后一次完整 transaction 在 deadline 前结束、收尾 validation 跨线时，raw
  termination 与 trace 失配。新增确定性 streaming regression 让时钟仅在最终
  validation 跨线：修复前稳定得到 `wall_clock_deadline` 加零 boundary；修复后每个
  deadline termination 都追加一条独立 `solver_finalization` lane 的
  `solver_termination` boundary。它不冒充提前 0.1 秒结束的 legacy/quality lane，
  也不替代已有 lane-local boundary。真实 Stage 5.2 shard writer 的集成回归完成
  stream append、flush/finalize、Parquet seal 与 `ArtifactReader.iter_events()` 回读，
  并要求 raw termination 与读回的 terminal boundary 一致。solver 决策、objective、
  validator、预算和 acceptance 语义均不改变。Attempt62 raw 与失败 review
  generation 永久保留；该 producer 修复必须使用下一最低未占用 Pilot label 重新验证。

## 2026-07-27：G64 unrelated-process 负载合同适配 24-thread 主机

- 修复后的 Pilot `stage05.2_benchmark_attempt63` 完成 36/36 shards，独立 campaign
  review 通过全部 18 个 mandatory gates，并报告
  `READY_FOR_STAGE052_FORMAL_BENCHMARK`。随后 Formal
  `stage05.2_benchmark_attempt64` 按完整 920-shard / 2,040-axis scope 从零启动。
  batch0001 的 402/402 shards 完成，`persistence_ratio=0.26139573609082606`，
  9,542,031,075 bytes raw 经 copy→verify→atomic publish 归档到 `d_archive`。
- batch0002 运行时，Codex 错误执行了递归 `/home/oneblaze/.local` 文件查找。失败
  runtime evidence 的 908 个可重放样本记录 PID 727009 连续消耗约 28 CPU seconds，
  `maximum_unrelated_process_average_cores=1.5603668609234687`；其时间范围与该
  24.2-second 查找严格重合。旧合同只要任一 campaign 外进程在完整 30 秒窗口达到
  1.0 core 即终止，因此 guard 以
  `unrelated process averaged one full core` fail fast。Attempt64 batch0002
  保留 8 个 complete 与 37 个 partial shard envelopes 及失败 scratch，不续跑、
  不导入或复用其中 shard；batch0001 归档也保持不可变。
- Operator 已明确要求提高负载上限并最大化匹配这台 24-logical-CPU 主机。current
  `RuntimeLoadPolicy` 现在把 unrelated-process ceiling 固定为 4.0 cores：低于
  4.0 的单进程后台维护/监控负载允许，达到 4.0 立即 fail fast。该值仍按每个进程的
  完整 rolling 30-second average 计算，并继续排除 campaign PID tree；24-thread
  identity、campaign-start `load1 <= 4.0`、runtime `load1 <= 32.0`、AC power、
  low-power mode、四 workers、solver/backend/objective/validator、36% persistence、
  producer memory 和 reviewer 5.5-GiB memory contracts 均不改变。raw/review schema
  与 publication surface 保持兼容。
- 最小反馈回路直接重放30秒单核样本：旧实现稳定返回 abort，修复后稳定通过；
  新的30秒四核边界样本仍被 runtime evidence 与 live monitor 拒绝。Stage 5.2
  campaign runner/reviewer 定向测试 126 项通过。合同变化后必须使用下一最低未占用
  label 从零跑新 Pilot 并通过独立 review，再以新的最低未占用 Formal label 重跑；
  producer 运行期间不再执行递归 WSL 文件扫描。

## 2026-07-27：G65--G68 sealed launch envelope 与 successor allowlist 修复

- `stage05.2_benchmark_attempt65`、`attempt66` 与 `attempt67` 均在 raw directory
  创建前由 fail-fast launch checks 拒绝：Attempt65 的 transient service PATH
  缺少 Windows PowerShell；Attempt66 把 benchmark Pilot prerequisite role 误写为
  `accelerator_pilot` 而非规范的 `accelerator_decision`；Attempt67 的 service PATH
  缺少 WSL GPU bridge `/usr/lib/wsl/lib`，使 F17 frozen runtime replay 找不到
  `nvidia-smi`。三者没有 producer raw，但 label 均不复用。
- 启动环境现先在相同 PATH 下完整执行 F17 frozen runtime identity script，而不是
  分别探测单个命令。该 probe 已同时验证 Python 3.13.13、24 logical CPUs、RTX 4070
  identity、driver 610.62、CUDA capability 8.9、PowerShell 与 ZHITAI D archive
  disk identity。Attempt68 随后在 raw 创建前暴露独立的 successor allowlist 缺口。
- G64 负载合同的边界回归直接位于 `tests/test_stage052_campaign.py`，但旧 G-only
  successor allowlist 只列出 campaign runner/reviewer 测试文件，因而把合法的
  configuration-contract test 误报为 non-G change。allowlist 仅新增这一条精确测试
  路径；`src/evrptw/objective.py` 等非G路径仍由现有负向回归拒绝。修复前的最小
  successor fixture 稳定失败，修复后必须返回 runner 与 config-test 两条精确路径。
  Attempt68 不复用；新 sealed revision 必须从下一最低未占用 Pilot label 启动。

## 2026-07-28：G69 native Arrow replay 与自校准资源合同

- Attempt73 保持 immutable interrupted evidence（不可变中断证据）：batch0001/0002
  只作为只读 benchmark/differential corpus（基准/差分语料），对新 campaign
  geometry、readiness count 与 shard inheritance 的贡献均为 0。新 Pilot/Formal
  必须使用新 label 从零执行。
- 新增深模块 `evrptw.stage052_replay`。`python_reference` 仅用于差分与旧格式兼容；
  Formal 强制 `native_arrow`，直接消费 Arrow 列缓冲并在 C++ 状态机审计 event/order、
  cache/exact/deadline/candidate transaction、vehicle-first acceptance 与 persistence
  ledger，任何 native 失败均禁止回退。Attempt72 全部 36 shards、16,974,466 events
  逐字段相等；native 单 child 为 742,995 events/s，相对本机 Python reference
  7.0966x，相对旧 41,966 events/s 基线 17.7x。
- reviewer scheduler 在 Attempt72 全 36 shards 上实测选择四 workers：
  699,474 events/s，约为旧串行基线 16.67x；1/2/4-worker semantic digest 完全一致。
  Pilot-derived memory contract 为 `MemoryHigh=2,292,604,354`、
  process guard `2,562,322,514`、`MemoryMax=2,697,181,594` bytes，swap 为 0。
  Review/campaign schema 升至 v2，逐 shard 记录 elapsed、events/s、child peak RSS、
  canonical merge ordinal、in-flight bound 和 native fallback count。
- producer 改为 4/5/6-worker real-shard calibration；选择规则固定为 75% available
  memory、相对四 workers至少 15% 提升、5% tie 选择更少 workers、zero swap/fallback
  与 identical semantic digest。Attempt73 batch0001/0002 的 sealed long-shard
  process-tree/per-worker RSS 作为保守内存下限，并按 worker 数线性投影，避免短时
  fixed-work calibration 低估 Formal 峰值。Parquet 仍为 Zstandard level 1 和既有科学 schema；
  新 writer 支持校准的 65,536/262,144 row group 与 queue depth 1/2。Attempt73 两个
  sealed c101_21 event files 的实际 writer-path benchmark 中，262,144/2 从
  1.605 秒降至 1.085 秒（约 32.4%），峰值约 593 MB，满足 10% adoption gate。
- 用户要求的 8-worker exploratory probe 使用同一六-shard fixed-work scope：
  `51.658 exact calls/s`，低于同轮 6 workers 的 `95.230`（慢约 45.8%），也低于
  4 workers 的 `60.939`；semantic digest 相同且 swap/fallback 为 0，但 sealed
  long-shard 线性投影 aggregate RSS 为 `18,444,517,376` bytes。因此 8 workers
  只保留为零 campaign-geometry 的只读 probe evidence，Pilot 仍选择 6 workers。
  Parquet 的 row-group/queue-depth 最终值由 signed producer resource contract
  覆盖静态 TOML 默认值并进入 campaign lock，避免计时噪声要求修改 source revision。
- producer selection 的 75% 包络以稳定 WSL/cgroup memory capacity（内存容量）
  计算，不再使用受 IDE、page cache 与刚结束测试影响的瞬时 `available` 值；校准报告
  同时保留当时的 observed available memory。Pilot/Formal 启动前仍以实际可用内存
  对冻结的 aggregate limit 执行 hard preflight，因此容量合同不会掩盖运行时竞争。
- runtime identity 分离 hard contract 与 non-blocking telemetry。AC/battery、固定
  24 logical CPUs、load/temperature、磁盘型号/序列与 device UUID 不再决定 readiness
  或 publication identity；硬 gate 只要求所选 workers 所需 CPU、内存/空间、backend、
  Python ABI、native extension、source/wheel/config/input/schema hashes 以及 fsync/
  atomic-transfer capability。长期 storage publication identity 只包含 alias、
  relative path、file count、byte count 与 tree SHA-256。

## 2026-07-28：G70 Attempt72 continuity contract

- 新 Pilot 的唯一 prerequisite（前置证据）改为已接受且完整独立审查的
  `stage05.2_benchmark_attempt72`，不再回跳到更早的 accelerator decision
  `stage05.2_accelerator_pilot_attempt17`。Attempt72 继续冻结 backend、exact backend、
  native kernel、科学配置、instance hashes 与输入 provenance；新的 signed producer
  resource contract 只替换 worker/RSS/Parquet tuning（资源与持久化调优）。
- successor gate 现在从不可变 public-history bridge
  `968b9dd421dd4ae2b8542d43b284b3ade94e760c` 读取 legacy-to-public provenance map，
  将 Attempt72 的 legacy revision
  `a5cf00f7580fc2632179495a739a110786ace87d` 解析为公开历史
  `8b1494e0aa6864a88e37ed700cf686d1165a5320`，再只审计 bridge 之后的显式 G/replay/
  resource/test/documentation allowlist。未映射 legacy revision、无法到达 bridge
  或 allowlist 外变化均 fail fast。
- configuration selection digest（配置选择摘要）排除由 signed resource contract
  单独冻结的 `resource_calibration_contract`、Parquet row group 与 queue depth；
  compression、scientific axes、solver/config inputs 等任意变化仍会拒绝。旧 raw
  configuration SHA-256 与新 raw configuration SHA-256 均原样保留，不改写证据。

## 2026-07-28：G71 successor runtime dependency lock

- `stage05.2_benchmark_attempt74` 在 raw directory 创建前 fail fast，退出原因为旧
  runtime selection hash 要求整个 native extension 与 Attempt72 字节相同。新增
  `Stage052ReplayState` 后 extension SHA 必然变化，因此该旧比较无法表达“同一 solver
  ABI + 新 native replay adapter”。Attempt74 无 shard、无 campaign geometry，
  label 不复用。
- 新 successor runtime selection 保持 Attempt72 的完整 third-party dependency
  versions（第三方依赖版本）与 Python 3.13 identity；仅排除由当前 sealed wheel
  单独冻结的 native-extension/install-distribution bytes，以及公开仓库改名造成的
  self distribution name `evrptw-reproduction` → `reproducible-evrptw`。任一第三方
  依赖版本变化仍 fail fast。当前 extension SHA、wheel SHA、native ABI/config 与
  installed distribution SHA 继续写入新 runtime identity，并由新 raw/reviewer
  独立验证；Attempt72 全 shard differential equality 与 signed resource semantic
  digest 负责证明新增 replay surface 未改变 solver/event 语义。

## 2026-07-28：G72 accepted-Pilot reviewer continuity

- `stage05.2_benchmark_attempt75` producer 完成全部 36 shards 与 3 batches；
  15,099,532 raw event rows（最终值仍由 accepted review 独立重算）全部归档，三批
  persistence ratio 为 `0.3011348793`、`0.2923761033`、`0.3223115685`，
  systemd producer peak 为 5.3 GiB、swap 为 0。该 raw campaign 保持 immutable。
- 第一代独立 native review generation
  `d53db7125c853cde37da96243273cedc0bcf950805b3e9a5b77ace13a18e30f7`
  返回 `NOT_READY`，因为 campaign reviewer 的 Pilot prerequisite adapter 仍按旧
  F02 accelerator component 与四 workers 验证，未消费 producer 已正确使用的
  Attempt72 accepted-Pilot lock + 6-worker resource contract。它在 prerequisite
  gate 后 fail closed，因此显示 0/36 replay；不是 raw shard 语义失败。
- reviewer 现在独立加载 Attempt72 的 accepted campaign lock，重新应用 signed
  producer resource contract，并以 configuration selection digest 校验新 config
  artifact。旧 accelerator evidence 仍走显式 historical adapter；新 Pilot 不允许
  implicit fallback。失败 generation 与 finalized receipt 保留，修复后的 reviewer
  必须以新 sealed revision 写入 immutable retry history 后重放同一 Attempt75 raw。
- 第二代 generation
  `0711a7fde9603c5c6a577fd3f8559a8af64f1155090b2b7af9233e63d0eb71f2`
  成功以 `native_arrow`/4 workers 重放 36/36 shards、15,099,532 events，0 fallback，
  19 个非 publication gates 中 17 个通过。它仍为 `NOT_READY`：reviewer 重建 plan
  时漏传 signed producer resource contract；同时最大真实 shard 的
  `screening_definition_rows=131043`，比不可降低的 131,072 spill boundary 少 29。
  plan replay 已补齐 resource contract；Attempt75 不被提升，新 Pilot 使用下一 label
  从零运行，只有真实 shard 越过 131,072 才可通过 spill gate。
- `stage05.2_benchmark_attempt76` 随后从零完成 36/36 shards；真实
  `rc101_21/2015` shard 写入 131,106 个 unique screening definitions，超过固定
  131,072 boundary 34 条并触发 transactional SQLite spill。独立 native review
  generation `62ff2886851c214aaffef66fd3aa370fe054f97052b385d0a31b51a646a973c0`
  重放 14,978,339 events 后通过 19/20 gates，仅 publication dry run 因新 selection
  lock 漏带 Attempt72 内部冻结的原 accelerator-review SHA 而失败。该 hash 与
  Attempt72 review SHA 是两条不同的谱系边：修复后同时保留两者，再以新 sealed
  reviewer generation 重放同一 immutable Attempt76 raw。

## 2026-07-28：G73 accepted Pilot 与 Formal launch

- Attempt76 的最终 independent review generation
  `4bc54ffae0ddfb97ddaf9a47e0a9d64b6de62e306c1a172631bf253c17d86b64`
  使用 `native_arrow`、4 reviewer workers 重放全部 36 shards、14,978,339 events，
  `native_fallback_count=0`，20/20 mandatory gates 全部通过并发布
  `READY_FOR_STAGE052_FORMAL_BENCHMARK`。finalized review receipt 的 raw manifest
  before/after SHA-256 均为
  `e347a435039427ed4b70c0abedc399cdbfbdb4778cd65a0e13629b187d604668`；
  accepted review manifest SHA-256 为
  `68af4272706e6e7913ec0680cb7120534a92bdb34fdd3c5bfd032b9c21e96e58`。
- Formal 使用与 Pilot 相同的 producer revision、wheel、第三方依赖、6-worker
  resource contract 和科学配置，但通过新的 read-only source snapshot 把 Attempt76
  写入 signed campaign lock；Attempt76 使用过的 producer snapshot 保持不变。
- `stage05.2_benchmark_attempt77` 在 raw directory 创建前 fail fast：transient
  systemd service 的 `PATH` 漏带 Windows interop 目录，冻结 runtime replay
  子进程无法调用 `powershell.exe`/`wsl.exe`。该标签不复用；没有 shard、axis 或
  campaign geometry。
- `stage05.2_benchmark_attempt78` 随后以修正后的完整 service execution `PATH` 启动。
  Formal geometry 已在落盘前确认为 920 shards、2,040 axes、229,200 declared solver
  seconds 和 10,400 checkpoints；producer 固定 6 workers，
  `MemoryHigh=13,833,388,032`、`MemoryMax=16,600,065,639`、
  `MemorySwapMax=0` bytes。运行中状态不构成 readiness，只有 sealed raw 的后续
  independent native review 全部通过后才能发布 `READY_FOR_STAGE05_3`。

## 2026-07-28：G74 Attempt78 失败保留与 native bounded producer store

- Attempt78 完成 batch0001/0002 后在 batch0003 的 `c203_21/2016` shard fail fast。
  `PRAGMA integrity_check` 返回非 `ok`，触发
  `ArtifactIntegrityError: screening definition scratch failed SQLite integrity_check`。
  该服务未发生 OOM、swap、ext4/NVMe I/O error 或 fallback；systemd process-tree
  peak 为 14,444,707,840 bytes。Attempt78 作为 immutable failed evidence 保留，
  不续跑、不导入其已完成 batches，也不进入下一 Formal geometry。
- 对失败 shard 的 sealed raw 重建得到 505,237 个 definitions、3,255,454 个
  screening occurrences、64 个定义 transaction；全部 definition ID/full SHA-256
  与 transaction 插入计数一致。单进程、producer/writer 跨线程、6-process 重放和
  60-store 压力重放均未复现 corruption（损坏）。低层 SQLite page detail 因失败路径
  已按旧合同清理 scratch 而不可恢复，因此不把未证实的存储设备或线程假设写成根因。
- 根因修复针对 producer 的系统性 failure surface（失败面）：producer 本来只需要
  full SHA-256 collision token，却把超过 131,072 的状态转入 disposable SQLite。
  新 `native_bounded_digest` C++ store 使用 2,097,152-entry hard limit，批量注册在
  写入前原子完成去重、碰撞和容量验证；overflow/collision 直接失败，无 SQLite、
  scratch 或 Python fallback。review/read payload store 与 exact-route store 的
  既有 bounded compatibility path 保持独立。
- 失败 c203 occurrence stream 以新 backend 重放
  `transactions=64, occurrences=3,255,454, definitions=505,237`，插入计数完全一致，
  `sqlite_connection=0`、`scratch_directory=0`。历史最大同协议 shard
  `c201_21/2014` 的 1,131,700 identities 用时 26.460 s、RSS delta 310,140,928
  bytes；旧 SQLite 路径为 54.119 s、208,474,112 bytes，即新路径快 51.1%，增量
  内存约 101.7 MB。6-worker 同规模并发重放 6,790,200 identities 全部通过，
  aggregate peak RSS 2,233,892,864 bytes，swap 为 0。
- 4/5/6 calibration 后按用户要求补测 8 workers：8-worker throughput 为
  51.658 exact calls/s，比 6 workers 的 95.230 低 45.8%，也比 4 workers 低
  15.2%；semantic digest 相同且无 swap/fallback。因此下一 Pilot/Formal 仍固定
  6 producer workers，但资源合同改用实测峰值与明确 operating headroom，不再用
  任意 75% 百分比丢弃可用内存。
- 当前 source/protocol 已不同于 Attempt76 的 accepted SQLite-store revision，
  所以不能直接重启 Formal。必须密封新 revision/wheel/snapshots，以新未占用 label
  从零运行 replacement Pilot，通过 native independent review 后，再用另一个新
  label 从零运行 Formal。只有 Formal finalized receipt、raw before/after hash
  一致和全部 mandatory gates 通过后才能发布 `READY_FOR_STAGE05_3`。

## 2026-07-28：G75 replacement Pilot、8-worker 复测与 Formal runtime contract

- 新 revision `615acfc7aa0b63a1d04f56882ceb71053cf4871e` 的 producer calibration
  在相同 sealed corpus 上选择 6 workers：相对 4 workers 吞吐提高 83.5%，
  semantic digest、swap=0 与 fallback=0 保持一致。按用户要求补做的 8-worker
  exploratory probe 为 110.33 units/s，低于 6 workers 的 122.77 units/s 约
  10.1%，因此 Pilot/Formal 固定 6，不硬编码 8。
- `stage05.2_benchmark_attempt79` 因错误注入 `PYTHONHASHSEED=0` 在 raw 创建前失败；
  Attempt80 完成 batch0001 后因没有 Windows-side WSL keepalive 被 WSL 自动关闭；
  Attempt81 因 transient service 的 `StandardError` 路径错误在 exec 前失败。三者
  均不复用、不导入 shards。`stage05.2_benchmark_attempt82` 随后从零完成 36/36
  shards、14,313,033 events 和 3/3 archived batches，aggregate persistence ratio
  为 30.0718%。independent `native_arrow` review 以 4 workers、0 fallback 重放
  36/36 shards，20/20 mandatory gates 通过并发布
  `READY_FOR_STAGE052_FORMAL_BENCHMARK`；raw before/after SHA-256 均为
  `14642ef490d66432fe4727710e00f8d32e8c37732318e03fc728a9b8b50214e7`，
  accepted review SHA-256 为
  `304ee9968461dd5ba62cb5d519e03cc49d8167c6450a9c35ef70e670e025ad5e`。
- reviewer calibration 的 1/2/4-worker throughput 分别为
  253,025 / 423,428 / 580,346 events/s；4-worker `native_arrow` 达到历史
  41,966 events/s 基线的 13.83 倍并冻结进 signed review contract。合同使用
  `MemoryHigh=2,145,696,153`、`MemoryMax=2,524,348,416`、
  `MemorySwapMax=0` 和内部 process guard 2,398,130,995 bytes，不允许 Python
  fallback。
- `stage05.2_benchmark_attempt83` 使用 16,600,065,639-byte producer RSS 合同值
  作为外层 cgroup `MemoryMax`；6 个 workers 的 RSS 尚在预算内时，page cache 已使
  `memory.events.max=986`，造成非科学性的回收/节流。该运行在 batch0001 提交前
  主动停止并保留 5,706,485,367 bytes partial evidence；不续跑、不导入。
  后续 Formal 的外层 cgroup 使用完整实测 WSL memory 25,196,933,120 bytes，
  内部 6-worker aggregate/per-worker RSS 合同保持不变。
- Attempt84 的 CLI 多传了不合法的 `--prerequisite` 值，Attempt85 的 service PATH
  漏掉 Windows interop，Attempt86 则暴露同 revision runtime 比较仍错误地把 WSL
  `memory_bytes` 的 4--12 KiB 波动视为 selection drift。三次均在 raw directory
  创建前 fail fast，标签不复用。新增显式 hard-field allowlist 的
  `runtime_contract_sha256`：revision、wheel、Python、native extension、dependency、
  ABI 及其 hashes 继续锁定，其他字段均为 telemetry；producer、每 batch reviewer
  与跨 batch runtime-provenance gate 使用同一合同算法，hard hash drift 继续失败，
  memory/mount/path/temperature 等 telemetry 变化不改变 verdict。`source_snapshot`
  同样只锁定 revision、tracked/untracked hashes、read-only 和 ext4 capability；
  mount source/UUID/absolute path 仅记录，不再进入 readiness identity。

## 2026-07-29：G76 Attempt90 内存根因、独立后台宿主与 Formal 高水位合同

- `stage05.2_benchmark_attempt90` 完成并归档 batch0001/0002 后，在 batch0003
  完成 45/45 shards 但因 6-worker process-tree aggregate RSS 超过旧合同而
  fail fast。campaign/batch 状态保持 `failed`，batch0003 保留为 `partial`
  artifact bundle，不续跑、不复用。签名 artifact manifest 绑定的 resource
  summary SHA-256 为
  `a813c5cd8cbe82bed7570fe86fc95ae6b2c5885291d75dd3c9b3e473e917cac5`；
  实测 aggregate peak 为 18,449,874,944 bytes，单 child 最高为
  3,326,586,880 bytes，无 OOM、swap 或 fallback。
- producer 的资源采样器现在在运行波次内暴露 aggregate/per-process RSS hard
  violation；调度器每 0.5 秒轮询，首次违规即取消未完成 futures、终止整个
  process pool、为全部相关 shards 保留 partial failure evidence，且无 retry/
  fallback。旧做法直到批次结束才比较峰值，可能让越界任务继续数十分钟。
- 完整 30/60/300-second Formal memory probes 使用 `c203_21` 高内存波次且
  `campaign_geometry_contribution=0`。6-worker attempt02 的 aggregate peak 为
  16,939,073,536 bytes；按用户要求执行的 8-worker attempt03 峰值为
  21,343,768,576 bytes，占 WSL 容量约 84.7%，swap/fallback/limit violation
  均为 0。8-worker 仅是 exploratory evidence；replacement Pilot/Formal
  仍固定 6 workers。calibration report 以 `user_locked` 明确记录这一覆盖决定；
  6-worker 只要未通过原有 semantic/speed/resource gates 就直接失败，不会降级到
  4/5。probe 无论成功或被 guard 中止，都会先写 signed v2 attempt report；失败
  报告记录原异常与零 campaign geometry 后再以非零状态退出。
- 新 loader 只接受 `failed Formal campaign manifest → failed batch manifest →
  partial artifact-storage-v2 manifest → complete resource summary` 的完整签名与
  SHA-256 绑定链，并同时验证 run/batch、worker、row-group 和 queue-depth
  identity。Attempt90 只提供资源 high-water floor，不贡献新 campaign geometry。
  新 6-worker contract 取 fresh c203 probe 与 Attempt90 sealed peaks 的逐项最大值；
  20% operating headroom 对应 aggregate limit 22,139,849,933 bytes、per-worker
  limit 3,991,904,256 bytes，仍低于当前 WSL 25,196,941,312-byte capability。
- Windows Scheduled Task `Reproducible-EVRPTW-Stage052-WSL-Host` 通过原子
  launch nonce、launcher mutex、Linux `flock`、独立 controller 与落盘 receipt
  托管所有长任务。ChatGPT/Codex desktop 不在计算进程父链中；客户端退出或被
  清除后台不会终止已启动的 WSL 计算。客户端在线时只读监控日志、资源、manifest
  与 receipt；失败后保留证据并由人工审查根因，再以新 label 启动，不自动修改代码
  或静默重试。

## 2026-07-29：G77 external active root 发布与 split campaign retention

- 原因：新 Pilot/Formal 的权威 active writer 位于独立 ext4
  `/home/oneblaze/stage052-active`，而 atomic publisher 仍把 live raw 和
  prerequisite 写死为 `<repository>/results`；同时 batched campaign 已把每个
  verified batch 原子归档到 signed logical path
  `d_archive/<run_label>/batchNNNN`，旧 retention 文档却仍把完整 run 描述为一个
  `stage05.2/history/<run_label>` 目录。继续沿用旧假设会阻断合法发布，或诱导人为
  搬动 batch、改写 immutable campaign manifest。
- 修改范围：publisher 新增显式 `--active-results-root`，要求 Formal 与 accepted
  Pilot 都是该 resolved root 的直接子目录并拒绝 symlink path masquerading；
  `stage052_retention audit` 新增可重复 `--run-label`，可在共享 active root 中精确
  选择 immutable attempt。campaign metadata/review tree 仍归档到
  `stage05.2/history/<run_label>`，已签名 batch 保持原 logical path，不复制、不
  hardlink、不物理合并。
- retention archive 和 locator resolver 现在通过同一深模块复验 top-level
  campaign manifest sidecar 及外部 batch：batch manifest/sidecar、persistence
  envelope/sidecar、alias、relative path、file/byte count、tree SHA-256 和
  `.incoming` absence。accepted campaign 使用 review 中的
  `storage_publication_identity`；Attempt73 这类 interrupted/unreviewed campaign
  只复验 signed campaign manifest 中已经标为 `archived` 的 batch，active partial
  batch 继续由 metadata retention tree 覆盖且不贡献 replacement Formal geometry。
- 科学证据影响：不修改正在运行的 sealed producer/reviewer snapshot，不改变
  6-worker、native Arrow replay、920-shard geometry、objective、event audit 或任何
  gate。当前 Formal `stage05.2_benchmark_attempt92` 仍在运行，不能据此声明
  `READY_FOR_STAGE05_3`。
- 当前验证：修改文件已通过 Python syntax compilation 和 `git diff --check`。
  targeted/full pytest、Ruff、strict mypy、publication dry-run、实际 archive 与
  retained resolver re-audit 明确延后到 Formal producer/reviewer 性能证据结束后，
  以避免争用 CPU/磁盘污染 wall-clock evidence；完成结果必须另行追加记录。

## 2026-07-29：G78 publication/retention path 与 review-state hardening

- 原因：两轴只读 review 发现，retention 仅按
  `storage_publication_identity` 字段是否存在区分 accepted review，会把正常
  `NOT_READY` campaign 错当成 READY 并阻断失败证据归档，也会把缺失该字段的畸形
  READY review 降级成 unreviewed；publisher 仍可能跟随 Pilot `review/` 或
  `review_manifest.json` symlink，retention 也只拒绝末级 batch symlink、未拒绝
  signed `<run_label>` 父目录 symlink。
- 修改范围：campaign retention 改为先按 review status/scope 分类；Pilot/Formal
  READY 必须完整验证 storage identity、mandatory gates 和 finalized execution
  receipt，`NOT_READY`/中断/未审查证据则只按 signed campaign manifest 复验已经
  archived 的 batches。external storage root 到 batch 的每个路径组件均禁止
  symlink；其 nonzero file count 会重新计算，signed tree digest 对每个 relative
  file path/size 编码并绑定该 count，accepted review 还要求显式 count 相等。
  publisher 对 active root、raw、prerequisite 和 review 的每个路径组件拒绝
  symlink，同时读取 tracked retention registry，拒绝任何已登记 immutable run
  再次作为 live publication source。
- 回归覆盖：新增 READY 缺 storage identity、NOT_READY 仍含 storage identity、
  external batch 父目录 symlink、Pilot review directory/file symlink，以及
  retention-registry publication source 的负向测试。
- 科学证据影响：修复只作用于后续 publication/retention boundary，不修改
  `fed049d` sealed producer/reviewer runtime、Formal92 geometry、6-worker 计算或
  native Arrow replay。当前仅完成 Python syntax compilation 和
  `git diff --check`；pytest、Ruff、strict mypy 与实际 archive 仍须在 Formal
  性能链结束后执行并另行追加结果。

## 2026-07-29：Attempt92 supersession 与 ABI-v2 candidate transaction

- Attempt92 已停止且不会续跑。新增 signed supersession receipt，绑定原 campaign
  manifest、6 个外部 archived batches、未完成 batch0007 所在活动树
  `130c96702ce5456f5c9e2fc7170ffbfec92714f03f31767d19f72bfb27c9f8b7`、
  Windows host run `20260728T200039Z-460`、launch nonce
  `fc094d4828bf457681bda14091f98e70`、exit code 143 与进程不存在证明。receipt
  SHA-256 为
  `bb3c8ee92bfd3d34190bdabf9b1abec8b5f9970451bdb60ba6529995622696aa`。
  retention inventory SHA-256 为
  `1d1faabf2dd8d300e1256e0440d9d35929127cc75e9c85ea423d75ad8a308db8`。
- 用户确认后执行“先归档后清理”。metadata/activity tree 已完成 checksum-verified
  archive，活动目录不存在；6 个外部 batch 保持原 signed logical path，不移动、
  不复制、不导入。retention registry 记录 `superseded/partial/archived/verified`。
- 新 native ABI 为 `stage05.2-native-kernels-v2`。新增 ragged batch screening
  接口与 `NativeCandidateTransactionConfig` deep module，固定顺序为 candidate
  generation → pair pruning → native batch screening → original-order cache lookup
  → ordered `cpu_batch` → staged writes → deadline/budget validation → atomic commit
  或 rollback。任何 native/worker/deadline/integrity 错误均 fail fast，fallback
  计数必须为零。
- fixed-work E evidence 新增四步消融：`current_native`、`pair_pruning`、
  `batched_screening`、`candidate_transaction`。每个 shard 持久化原候选顺序记录及
  可重算 SHA-256、objective/validator、exact/cache/deadline counters 与 routes；
  reviewer 独立重算并应用 aggregate ≥15%、C/R/RC family regression ≤3% gate。
- accelerator decision 改为独立重算 native candidate screening occupancy；
  不再使用 exact backend launch occupancy。31/32 边界保持硬门槛，CUDA 仅在
  median ≥32 时执行；低于门槛发布 `GPU_NOT_JUSTIFIED`。
- 当前定向验证为 270 passed；native rebuild、完整 pytest、Ruff、strict mypy、
  Pilot/Formal 与 publication 结果将在各自完成后继续 append-only 追加。此条目
  不构成 `READY_FOR_STAGE05_3`。
- 实现提交前验证：editable native wheel 重建成功；完整 pytest 为 873 passed；
  Ruff 全仓通过；项目 strict mypy 为 71 个 source files 通过，三个本次修改的
  Stage 5.2 tools 另行 strict mypy 通过；`git diff --check` 通过。

## 2026-07-29：Candidate transaction 最终审查加固

- exact route cache、Python negative cache 与 solve-local native packed
  negative cache 现在共用延迟 finalize 的 O(changes) rollback journals。任一 sibling
  store、native commit 或完整性步骤失败都会撤销本事务的 exact/negative insertions
  与 LRU eviction/statistics；不会保留部分 cache writes。
- pair-capacity aggregate event 新增左右路线原始序列。reviewer 从两条路线、原始
  route indices 和 `len(left)+len(right)+2` 独立重算 skipped count、canonical pair
  identity 与 SHA-256，不再只检查 digest 格式。
- batched-screening ablation 现在持久化 ordered candidates、ABI-v2 structured array
  bytes 与 counters；reviewer 独立重算 screening SHA-256。deadline replay 改为
  lane-local，process-wide exact-budget boundary 仍保持全局终止语义。
- accelerator Pilot metadata gate 明确要求 6 workers、`cpu_batch`、选定 native
  backend、ABI-v2 native config 与完全相等的
  `NativeCandidateTransactionConfig`；review manifest 只复制已验证的原始配置。
- editable native wheel 再次重建成功；focused suite 为 242 passed，完整 pytest 为
  885 passed；Ruff 全仓、70 个 source files 的 strict mypy 与
  `git diff --check` 全部通过。该结果仅完成实现验证，不构成性能 promotion、
  Pilot/Formal readiness 或 `READY_FOR_STAGE05_3`。

## 2026-07-29：E16 streaming audit materialization 失败与有界审计流

- `stage05.2_native_kernels_attempt16` 在新 frozen commit `cb41aa1`、non-editable
  wheel 与 read-only source snapshot 上启动后 fail fast。首批 fixed-work trace 已
  externalized 到 Parquet，但 `_native_ablation_record` 仍尝试迭代完整
  `trace.events`，触发 `events were externalized during solve and cannot be
  materialized in memory`。该 attempt 为 partial evidence，不续跑、不导入、不复用
  label。
- 根因修复是在现有 streaming sink 内增加 fixed-work 专用 semantic audit stream：
  只保留 candidate/cache/boundary/native transaction 关键事件、ordered route
  evaluations 与 pair-pruning aggregates；每个 family 硬限制 65,536 rows，越界
  直接失败。完整主事件流继续只写 Parquet，不重新物化、不添加 Python fallback。
- `_native_ablation_record` 在 streaming 模式读取上述有界审计流；非 streaming 的
  三个 ablation mode 保持历史内存路径。focused streaming/native suite 为
  198 passed，Ruff、strict mypy 与 `git diff --check` 通过。修复必须形成新 commit、
  wheel、read-only snapshot 与 attempt17，E16 不参与 promotion。

## 2026-07-29：E17 启动失败、E18 trace reconciliation 拒绝与聚合对账修复

- `stage05.2_native_kernels_attempt17` 在创建 shard 前失败。Windows hidden host
  launcher 使用 `Start-Process -ArgumentList` 时拆分了 `bash -lc` command string，
  导致相对 `configs/stage052_performance.toml` 从错误工作目录读取并报告缺少
  `candidate_transaction`。外部启动日志保留；该 label 不复用。后续 launcher 改用
  `ProcessStartInfo.ArgumentList` 逐参数传递并把 config 固定为绝对路径。
- `stage05.2_native_kernels_attempt18` 在 clean commit `51128e9` 的 non-editable
  producer wheel 和 read-only source snapshot 上完成 12/12 performance shards，
  producer failure count 为零。独立 `systemd --user` reviewer 随后在
  `c101_21/2014` fail fast：三个主轴的 trace reconciliation 都缺少 candidate
  transaction 批量路径产生的 4 条 screening/incremental propagation 计数，因此该
  attempt 保留为 rejected evidence，不进入 promotion。
- 根因是 native batch 已正确增加 solver screening/cache/propagation statistics，
  但 trace 仍只统计逐条 `ScreeningDecision` 与旧 incremental event；紧凑 transaction
  event 未参与同一 producer reconciliation。修复新增 compact screening aggregate
  counters，不逐候选构造 Python screening object；同时为批量路径记录既有结构化
  incremental propagation event。transaction audit 现在区分 pass、safe rejection、
  negative-cache hit 和 exact-call-blocked，并携带 reason counts。
- independent reviewer 从 ABI-v2 statuses/codes/counters bytes 重新计算上述 aggregate
  fields；任何计数或 reason mismatch 都是完整性错误。fixed-work streaming 与
  non-streaming trace 共用同一 aggregate reconciliation 语义。
- campaign successor gate 同步固定本次 producer/reconciliation surface：
  `alns.py`、`candidate_transaction.py`、`measurement.py` 及对应 regression tests
  均绑定精确 SHA-256；测试覆盖从旧 producer revision 到包含整组修复的新 revision，
  防止后续 Pilot/Formal 在 source preflight 把合法修复误判为 non-G drift。
- 修复后的 focused candidate/streaming/native review suite 为 74 passed；完整 pytest
  为 887 passed；Ruff、70 个 source files 的 strict mypy 与 `git diff --check`
  通过。E18 仍为不可变失败证据；下一次性能运行必须使用新 commit、wheel、snapshot
  和 attempt label。

## 2026-07-29：E19 reviewer WSL 生命周期中断与信号来源修复

- `stage05.2_native_kernels_attempt19` 使用 clean commit `59ba963`、non-editable
  producer wheel `d6ba433d4d56375feb0e7e6c19674e01e3d0aa5d8c62e8c7e9af4fd3e399587c`
  和 869 个 tracked files 的 read-only ext4 source snapshot 完成 12/12
  performance shards。producer manifest 为 `complete`，36 个主结果轴均无 failure，
  manifest sidecar hash 通过；该 raw attempt 不进入 promotion。
- 独立 reviewer wheel
  `be74d460657aba585a26db160e11cebd86107f95afd932caeac6f0ad29c4b027`
  通过 transient `systemd --user` service 启动。服务在完成三个 C5 shard、进入首个
  100-customer shard 时因无存活 Windows WSL client 而在约 30 秒的 VM idle boundary
  收到外部 `SIGTERM`。finalized receipt 记录 reviewer exit 1、raw manifest 前后
  SHA-256 相等、systemd cgroup memory peak 1,239,379,968 bytes、swap 0；因此不是
  5.5 GiB process guard 或 6 GiB cgroup 上限触发。
- 根因同时暴露了信号可观测性错误：`ReviewProcessMemoryGuard` 原先把任何
  `SIGTERM` 都报告为 `ReviewMemoryLimitExceeded`。现在外部终止固定使用
  `SIGTERM` 并报告 `ReviewServiceInterrupted` 及 signal number；memory sampler
  使用独立 `SIGUSR1` 报告 `ReviewMemoryLimitExceeded`。即使 progress evidence
  写入失败，内部越界信号也在 `finally` 发出。sampler 自身的非阈值异常使用独立
  `SIGUSR2` 传回主线程并报告 `ReviewMemoryGuardFailed`，不允许 daemon thread
  静默退出。回归测试覆盖三条信号、周期／越界写入失败和 handler restoration。
- 后续 reviewer 必须由隐藏 Windows keepalive WSL client 持续绑定到 transient unit
  生命周期；仅在 unit 终止后退出。E19 review execution 保留为 immutable failed
  receipt，下一次 producer/reviewer 使用新 commit、wheel、read-only snapshot 和
  attempt20 label，不重用 E19。

## 2026-07-29：E20 独立复核完成、producer root 绑定与批内负缓存计数修复

- `stage05.2_native_kernels_attempt20` 使用 clean commit `497f50c`、non-editable
  producer wheel
  `6eda8244dde86d8bf80b91664c8dbcc58653381674ed66f5cd64027729957797`
  与 869 个 tracked files 的 read-only ext4 source snapshot 完成 12/12 shards、
  36/36 axes，producer failure count 为零。Windows keepalive client 持续绑定
  WSL 生命周期后，独立 transient `systemd --user` reviewer 完整退出，finalized
  receipt 记录 exit 0、raw manifest 前后 SHA-256 相等、cgroup memory peak
  5,369,495,552 bytes、swap 0 和原子 review publication。
- 独立复核状态为 `NOT_READY`。scope、job-parallel selection、optimization
  profile、persistence attribution/ratio、prerequisite、replay consistency、
  resources 与 staging identity gates 通过；source snapshot、runtime identity
  与 native execution gates 失败。E20 为不可变 rejected evidence，不进入
  promotion，不重写 review。
- 前两个失败共用同一 reviewer root-selection 根因：service receipt 已区分
  producer `working_directory` 与 `reviewer_working_directory`，但 source/runtime
  gates 仍调用 reviewer checkout 的 `find_repository_root()`，因而把活动目录的
  `.ruff_cache/.gitignore` 当成 sealed producer snapshot 污染，并从错误根目录
  校验 producer runtime。receipt 现在新增显式 `producer_source_directory`；
  reviewer 只从该绝对、存在且为目录的 receipt binding 复核 source snapshot 与
  frozen runtime，旧 receipt 仅为兼容读取 `working_directory`。
- native execution 失败来自旧计数公式
  `screening_calls - screening_cache_hits`：它把一个 native batch 当成多个
  scalar invocations，并遗漏 batch 内 negative-cache hit。修复后 reviewer 从每个
  committed transaction 的 ABI-v2 statuses/codes/counters bytes、candidate order
  与两个 SHA-256 独立重算 batch candidates、cache hits、invocations 和
  occupancies，再用
  `scalar_non_cache_candidates + native_batch_invocations` 对账。E20 的失败轴
  `c101_21/2014/fixed_work` 因此独立重算为 2,346 次，和 raw native counter
  一致。
- 为使上述复核不信任 producer summary，candidate list、screening integrity
  bytes、native counters 与 reason counts 现在随 critical Parquet event 的
  `extras_json` 持久化；既有 event schema 不变。缺字段、hash mismatch、aggregate
  mismatch、负数 scalar count 或 undeclared batch evidence 均 fail fast，无
  Python/serial/CUDA fallback。
- ABI-v2 raw counters 还必须逐行等于 statuses 中的 unique、duplicate、
  negative-cache-hit 与 screened 数量；duplicate candidate 必须指向更早且完全相同
  的首次 route/codes/metrics row；reviewer 也反向维护 first-by-route identity，
  后续相同 canonical route 不得伪装成新的 screened row。非 duplicate 的
  `duplicate_of` 必须为 -1，negative cache hit 必须携带安全拒绝 reason。producer
  deep module 和 reviewer 共用这组 fail-fast 语义，防止内部代数恒等但逐行
  自相矛盾的 native evidence 被接受。下一次运行必须使用新 commit、wheel、
  read-only snapshot 和 attempt21 label。

## 2026-07-29：E21/E22 启动前拒绝、E23 batch replay 复核修复

- `stage05.2_native_kernels_attempt21` 在创建 shard 前由 canonical output-path
  preflight 拒绝：launcher 把 staging root 而非完整 run-label directory 传给
  `--output-dir`。`stage05.2_native_kernels_attempt22` 同样在创建 shard 前拒绝：
  launcher 未传显式 `stage05.2_job_parallel_attempt21` worker-selection
  prerequisite。两个 label 均已消费，分别保留 `control/launch_failure.json`；
  E22 另保留 producer log。没有 shard、raw artifact 或签名 manifest 被复用。
- `stage05.2_native_kernels_attempt23` 使用 clean commit `e2841d6`、producer wheel
  SHA-256
  `919151faa8a6caffd33463a697ebe476596cb8f4cd0f8a4330313aa1fe5cefbb`
  和只读 ext4 producer source snapshot 完成 12/12 shards、36/36 axes，failure
  count 为零。独立 reviewer wheel SHA-256 为
  `41dbf4361873c9fc8e029eec94bf6b1743578af2bb19ce3e1a0ce66ed202d2a2`；
  transient `systemd --user` review 完整结束，finalized receipt 记录 exit 0、
  raw manifest 未变化、cgroup memory peak 5,369,589,760 bytes、swap 0。
- E23 独立复核为 `NOT_READY`。scope、job-parallel selection、optimization
  profile、persistence attribution/ratio、prerequisite、replay consistency、
  resource、runtime/source snapshot 和 staging gates 全部通过；唯一失败 gate
  为 `native_execution`，detail 为 `native screening batch evidence is missing`。
  该结果保持不可变，不重写、不作为 promotion evidence。
- 根因是 reviewer 的两项 replay 语义错误，而非 E23 raw evidence 缺失。第一，
  C5 控制轴合法声明零 batch 为 `0/0/[]`，reviewer 却把非 `None` 的零值误判成
  必须存在 transaction event。第二，Stage 5.2 持久化流水线按协议把 lane 改写为
  `benchmark_axis:original_lane`，transaction SHA-256 则绑定改写前的 original
  lane；reviewer 未先验证并移除 axis prefix，因而会误拒绝真实 batch hash。
- 修复后，零 batch 仅在 recorded counters 精确为 `0/0/[]` 且不存在 transaction
  event 时接受；存在任意 event 仍作为 undeclared evidence 失败。对 transaction，
  reviewer 要求持久化 lane 具有精确 axis prefix，移除一次 prefix 后再重算原始
  transaction SHA-256；original lane 只允许 producer 可生成的 `legacy`、
  `quality_shadow`、`constraint_lane` 或 `initialization`，空 lane、重复 axis prefix
  与其他 lane 均 fail fast。使用修复后的 checkout 对 E23 已签名 Parquet 做只读
  诊断，12 个 shard 的 36 个轴全部通过 ABI-v2 bytes、hash、occupancy 与 native
  invocation counter 对账；此诊断不改变 E23 的 immutable `NOT_READY` status。
- 新回归测试覆盖零 batch 与 persisted lane normalization。editable native wheel
  重建成功；focused replay/publication suites 通过；完整 pytest 为 897 passed；
  Ruff、70 个 source files 的 strict mypy 与 `git diff --check` 全部通过。修复经
  两路独立只读代码复核后形成新 commit；后续性能证据必须使用新 wheel、只读
  source snapshot 和 attempt24 label。

## 2026-07-29：E24 producer 完成、review launch 参数面拒绝

- `stage05.2_native_kernels_attempt24` 使用 clean commit `d925645`、producer wheel
  SHA-256
  `01d294e2e816e81663f96c3ea1972482e37c5ed48bec75305216ce1a573128b1`
  和只读 ext4 source snapshot 完成 12/12 shards、36/36 axes；36 个 validator
  全部通过，failure status 为零。Windows Scheduled Task host run 为
  `20260729T092055Z-420`，launch nonce 为
  `356e168f2f54474c80b39b66d8523d4b`；producer manifest SHA-256 为
  `72d91eb6e294b4160ec6b2cf41feb41612ffcd4a4cc4c4803b77dd20197b29a6`。
- 独立 review service 在进入 raw replay 前 fail fast。launcher 错误地向 performance
  reviewer 传入 review-calibration contract，service 因而追加该 CLI 不支持的
  `--review-workers 4` 与 `--review-memory-contract`，argparse exit code 为 2。
  finalized execution receipt 记录 status `failed`、raw manifest 前后 SHA-256
  相同、cgroup peak 78,864,384 bytes、swap 0；没有 review generation 或
  review manifest 被写入。
- E24 按失败即消费 label 的规则保留，不在同一 label 重试，也不进入 promotion。
  根因属于 launcher 参数面，不改变 producer/reviewer 科学代码。下一次 review
  service 只使用 performance reviewer 已验证的 5.5-GiB process guard 参数面；
  新证据必须从新 commit、wheel、read-only snapshot 和 attempt25 label 生成。

## 2026-07-29：E25 producer 完成、Windows keepalive quoting 失败

- `stage05.2_native_kernels_attempt25` 使用 clean commit `a3cf2ea`、producer wheel
  SHA-256
  `b51bced9692b86e133b1ff6c343011cca3c71fdd5bc9828a00825095f829053a`
  和新只读 ext4 source snapshot 完成 12/12 shards、36/36 axes。Windows
  Scheduled Task host run 为 `20260729T093220Z-527`，launch nonce 为
  `95b9a4e0043045348d99fe36517b5903`；producer manifest SHA-256 为
  `f07df74128a982040b4ede3183e20fe5b97bb0c761a4344e3a0372099056f0a6`。
- reviewer 使用已验证的 performance 参数面启动，但 Windows keepalive 又通过
  `Start-Process -ArgumentList` 传递含空格的 `bash -lc` command string，Windows
  将其拆分，keepalive 未持续绑定 WSL。reviewer 在约 27 秒后收到外部 `SIGTERM`；
  finalized receipt 记录 `ReviewServiceInterrupted`、exit 1、raw manifest unchanged、
  aggregate peak RSS 1,414,012,928 bytes、swap 0，且没有 review manifest。
- E25 按失败即消费 label 保留，不在同一 label 重试。下一次 keepalive 必须通过
  `.NET ProcessStartInfo.ArgumentList.Add()` 逐参数传递，并在 review 启动后独立
  验证 Windows keepalive PID 存活及 WSL command line 完整。新证据使用新 commit、
  wheel、read-only snapshot 和 attempt26 label。

## 2026-07-29：E26 完整独立复核与 native ablation 根因修复

- `stage05.2_native_kernels_attempt26` 使用 clean commit `e20777b`、producer wheel
  SHA-256
  `bee0bcb6d06b5939f65c980695f7a3a6479d502b47ac801208540f89c67f1c4b`
  和只读 ext4 source snapshot 完成 12/12 shards、36/36 axes。Windows Scheduled
  Task host run 为 `20260729T094159Z-466`，launch nonce 为
  `096923d976304e66987309f7ad6022e5`；producer manifest SHA-256 为
  `0816a9c81d48363a310e34615f8a764f5c268ab08fd6e8a6a4e48796ccb8c877`。
- 独立 reviewer wheel SHA-256 为
  `c8be024383d0648589972d2ed8a3c2ebdfc24398274a221248042fe7e386eb86`。
  Windows keepalive 使用 `ProcessStartInfo.ArgumentList.Add()` 后持续绑定 WSL；
  transient `systemd --user` reviewer 完成 attempt26 与
  `stage05.2_job_parallel_attempt21` 的全部 raw replay，并以 exit 0 原子发布 review。
  finalized receipt 记录 raw manifest 前后不变、cgroup peak 5,369,487,360 bytes、
  swap 0。
- E26 独立复核为 `NOT_READY`，且唯一失败 gate 为 `native_ablation`；其余 exact
  scope、worker selection、optimization profile、persistence、prerequisite、
  validator/objective replay、resource、runtime/source snapshot 与 staging root
  gates 全部通过。该 review generation 保持不可变，不重写、不进入 accelerator
  decision。
- 第一项 reviewer 根因是把 canonical JSON object 的字典 key 顺序误当成 ablation
  实验顺序。producer raw 实际包含全部四模式；规范化 JSON 按 key 排序，科学顺序则
  已由固定 `current_native -> pair_pruning -> batched_screening ->
  candidate_transaction` 循环控制。修复后 gate 验证精确 mode identity set，并继续
  按固定顺序逐项重放 schema、objective、candidate/exact order 与 audit bytes。
- 只读诊断随后暴露两项真实 producer 语义错误。其一，native route-merge pool 在
  进入 ABI v2 前由 Python 去掉重复候选位置，改变 exact-call order 和 100-call
  budget boundary；现在仅 Stage 5.2 transaction 保留全部原位置并由 native batch
  执行批内 dedup，Stage 3.4 historical candidate-control path 保持不变。其二，
  candidate 比 base propagation snapshot 更长时，Python 与 C++ 的 backward
  suffix copy 错把 candidate suffix index 与 base length 比较，导致可行路线被错误
  标记为 `backward_time_window_prefilter`；两端已同步按 suffix length 复制并新增
  full-propagation differential test。
- native ablation axis 升级为 v3，显式记录 batch invocation/candidate counters；
  reviewer 允许可观察的 `0/0` zero-work batch，同时对非零批次从 raw integrity
  bytes 重新计算 invocation、candidate count、occupancy、median 与 transaction
  hash。修复后单个 `r101_21/2014` scratch shard 的四模式 objective、candidate
  state order 与 exact route order 哈希完全相同；batched/full 均为一次
  45-candidate batch，fallback 为零。该 scratch 仅作修复前验证，不消费 canonical
  attempt label；新 evidence 必须使用新 commit、wheel、只读 snapshot 和 attempt27。

## 2026-07-29：E27 promotion timing 封套失败与 v4 修复

- `stage05.2_native_kernels_attempt27` 使用 clean commit `50ad1b4`、producer
  wheel SHA-256
  `0e203ed30f0c4061c77bf5cd38d9cfce6abb02ea3192a2c990a2427429c3cf28`
  和只读 ext4 source snapshot 完成 12/12 shards、36/36 axes；producer manifest
  SHA-256 为
  `673587999b8089564a8d7738a9d39aea31787265cc49238350005a30c1697a73`。
  Windows Scheduled Task host run 为 `20260729T104556Z-277`，launch nonce 为
  `754761285aa94085a6bfd39d6ceeee16`。
- 独立 reviewer wheel SHA-256 为
  `a0d9066c469f95c1301e25afd3b819385d8d92641f6e03c796cf13cbc56bc112`。
  transient `systemd --user` reviewer 完成两层 attempt27/attempt21 raw replay，
  finalized receipt 记录 exit 0、raw manifest 前后 SHA-256 相等、cgroup peak
  5,369,712,640 bytes、swap 0，并原子发布 generation
  `58a3f5030331094997d203421fe820e30099e91d7be9030dfaf3185d75ea344b`。
- E27 独立复核为 `NOT_READY`，唯一失败 gate 为 `native_ablation`：100-customer
  aggregate paired median saving 为 12.5481%，低于 15%；C/R/RC family savings
  分别为 8.2216%、12.5481%、19.5177%，没有 family regression。其余 scope、
  source/runtime、validator/objective、worker selection、persistence、resource、
  staging 与 replay gates 全部通过。E27 保持不可变，不进入 accelerator decision。
- 根因是四步消融使用了不同 instrumentation envelope。完整
  `candidate_transaction` 计时取自正式 fixed-work axis，带 asynchronous trace
  streaming；`current_native`、`pair_pruning` 与 `batched_screening` 则由单独的
  in-memory trace 路径计时。runner 虽扣除了实际持久化时间，但没有也不能可靠扣除
  serialization 与 queue management，因此只给完整事务增加了额外成本。C5 控制的
  完整事务相对 current native 慢 27%--58%，与该单边插桩偏差一致。
- native ablation axis 升级为 v4。四个模式现在严格按
  `current_native -> pair_pruning -> batched_screening -> candidate_transaction`
  在相同 `in_memory_measurement_trace_no_stream_sink_v1` 封套中独立运行；正式
  fixed-work streaming axis 继续保留为 campaign/resource/persistence evidence，
  但不再替换其中一个 promotion timing。每个 ablation row 显式声明 timing
  envelope，reviewer 对旧 schema、缺失或混用 envelope fail fast。E27 不回写；
  修复必须使用新 commit、wheel、只读 snapshot 和新 attempt label。

## 2026-07-29：E28 sealed snapshot 本地配置遗漏

- `stage05.2_native_kernels_attempt28` 由 audited Windows Scheduled Task host run
  `20260729T113702Z-417` 启动，launch nonce 为
  `f01039fbde7a421ab50934ec13086cee`，controller SHA-256 为
  `8e07e2a335b34a6ffe68cfbd83c3de2143f3aca533c2c104d1dd13eded664499`。
  它使用 clean commit `64ff84d`、producer wheel SHA-256
  `926c0768cdcdbb41aeb0b4a2a20f7dcecffc2f9bec99e73b3498b954cfa16354`
  与只读 ext4 source snapshot。
- producer 在创建 output directory 或 shard 前 fail fast，Windows Scheduled Task
  与 Linux host controller 均记录 exit code 1。直接原因为 sealed snapshot 缺少
  ignored but required（被忽略但必需）的
  `configs/stage052_storage_roots.local.toml`；runner 因而拒绝解析 storage root
  locator。attempt28 只保留 producer log、controller、host/launch evidence，
  不存在 raw shard、manifest 或 review。
- 根因是手工 snapshot packaging 只复制了 Schneider data 与新 runtime identity，
  没有复制当前 checkout 中完整的七项本地 control files：campaign lock 及 sidecar、
  resource calibration 及 sidecar、review calibration 及 sidecar、storage roots。
  下一 snapshot 必须在冻结为只读前逐项复制并验证这些七项，再单独生成当前
  runtime identity；attempt28 label 不复用。

## 2026-07-29：E29 runtime identity source root 不一致

- `stage05.2_native_kernels_attempt29` 由 audited Windows Scheduled Task host run
  `20260729T114757Z-4908` 启动，launch nonce 为
  `0fab48569c0b4de0b1da1bc2ec4c4daf`，current controller SHA-256 为
  `c75f2e804a38e79d40aa00d46e99a1cb75c14d3ccac8a7d893a64e8b64ee11d2`。
  它使用 clean commit
  `f095e92fd34a8c9e2c55597e697215cf0457e8a6`、producer wheel SHA-256
  `c7f239d81546533c4c7fd4ac52acb36724da4770413cdf323f82987d8ccf44d2`
  与已验证包含八项本地控制文件的只读 ext4 source snapshot。
- producer 在创建 shard 前 fail fast，Windows Scheduled Task 与 Linux host
  controller 均记录 exit code 1。runner 创建了空 output directory，但没有 raw
  shard、manifest 或 review。直接原因为 runtime identity 的
  `source_repository_root` 记录活动 checkout，而 runner 要求它等于 sealed
  snapshot root。
- 根因是 runtime identity 虽写入 snapshot，却在活动 checkout 作为 current
  working directory（当前工作目录）生成。下一 snapshot 必须保持可写直到
  runtime identity 在该 snapshot root 内生成并复核
  `source_repository_root`，随后才可冻结为只读；attempt29 label 不复用。

## 2026-07-29：E30 pre-dispatch deadline counter 复核修复

- `stage05.2_native_kernels_attempt30` 在 clean commit
  `061dc121dcc38a9eda61512ee37d71b2a3c620aa` 上完成 12/12 shards 与
  36/36 axes；raw manifest SHA-256 为
  `04f1a439c9599306759b27ff8e87f702cb2453af5be1fbd8c6c70ff1d9380f17`。
  独立 reviewer service exit 0、raw manifest unchanged、swap 0，review manifest
  SHA-256 为
  `ed4e452357a861295a67a1db11197c724bfec3325012fbe87ec7458302298091`，
  但状态为 `NOT_READY`，唯一失败 gate 为 `native_execution`。
- 失败轴 `c101_21/2015/wall_clock_30` 的 raw counters 为
  `batch_launches=1445`、`native_invocations=1444`、`started=1450`、
  `completed=1449`、`interrupted=1`。raw route event 证明最后一次 exact call
  在 native kernel invocation 前的 deadline checkpoint 中断：
  `exact_started=true`、`exact_completed=false`、`labels_generated=0`，随后有
  `before_candidate_commit` deadline boundary；不存在 fallback。
- 根因是 reviewer 实现仍错误要求
  `native_invocations == batch_launches == work_batches`，没有落实工作流已声明的
  pre-dispatch deadline 语义。修复后的 gate 要求
  `work_batches == batch_launches`、
  `0 <= batch_launches - native_invocations <= interrupted_calls`、
  `completed + interrupted == exact_calls`、occupancy sum 等于 exact calls，
  并继续要求 native/protocol fallback 为零。新增回归测试同时拒绝没有
  interrupted evidence 的 invocation gap；修复后的审计函数已对 E30 全部
  36 axes 重算通过。E30 review 不回写，修复使用新 commit/wheel/attempt label。

## 2026-07-29：E31 fixed-work core semantic differential

- `stage05.2_native_kernels_attempt31` 在 clean commit
  `6b98bfdfe750563d742aa8cd29c6dac1df20c1f4` 上完成 12/12 shards 与
  36/36 axes；raw manifest SHA-256 为
  `e7aef963d27b58c8280931bd5cdc1cd8b117532740da1c500d0d070711c9dcad`。
  独立 reviewer service exit 0、raw manifest unchanged、swap 0，review manifest
  SHA-256 为
  `a859c11239bd6b4892e25b03bc19738e1cdfa424c1c4f5647a82ab714a0d2fef`，
  但状态为 `NOT_READY`，唯一失败 gate 为
  `native_fixed_work_differential`。
- E31 的四步 ablation raw replay 实际通过：100-customer aggregate paired
  median saving 为 `0.182139`，C/R/RC family 分别为
  `0.182139`、`0.234702`、`0.140098`；四个 mode 的 objective、
  candidate-state order、ordered exact-route work、validator、cache/deadline 和
  zero-fallback gate 全部通过。
- 失败根因是跨实现 differential 复用了 storage-only full canonical event
  equality。candidate transaction 合法新增 transaction events、batched-screening
  statistics 并减少重复 diagnostic cache-hit，因此 full event stream 必然不同；
  reviewer 还把该预期差异展开成 2.7-GB `semantic_mismatches.csv`。这不是 search
  semantic regression。
- 修复新增 streaming core semantic digest，只包含 solution/objective、
  candidate-state order、ordered exact-route results 与 deadline boundaries；
  transaction/screening/cache diagnostics 仍由每个 ablation mode 的独立 raw
  replay 审计。该 digest 已对 accepted D predecessor attempt21 与 E31 的
  24/24 fixed-work axes 实测完全相等。native full-storage mismatch publication
  仅保留 aggregate rows，不再展开字段级差异。E31 review 不回写，修复使用新
  commit/wheel/attempt label。

## 2026-07-29：E32 promotion semantic identity 修复

- `stage05.2_native_kernels_attempt32` 在 clean commit
  `82e06dbde45f20c97abd8123e6f940456a6f19ef` 上完成 12/12 shards 与
  36/36 axes；raw manifest SHA-256 为
  `210cef9f947243c374613202efbad269e45d4791dc515d03df905720ffa3be69`。
  第一个 reviewer generation 因 Windows 侧等待期间没有活跃 WSL client，在约
  32 秒收到 WSL lifecycle 的 `SIGTERM 15`；其 finalized receipt 证明 raw
  manifest unchanged、peak RSS 1,431,629,824 bytes、swap 0。该失败 generation
  保留，不改写 raw。
- 第二个独立 reviewer generation 通过持续 WSL client keeper 完整退出 0；
  aggregate peak RSS 2,394,062,848 bytes、swap 0，review manifest SHA-256 为
  `fcb6fe3daebf5889693ebfb90ecebaacb3d1a090079b39211ccf7676e92cf2c5`。
  native ablation 通过，aggregate paired median saving 为 `0.181622`，
  C/R/RC 分别为 `0.181622`、`0.211868`、`0.162964`；24-axis solution、
  candidate-state、ordered exact-route 与 deadline core semantics 全部相等。
- 唯一失败 gate 为 `performance_promotion`。其旧调用仍直接读取 per-run
  full-storage `semantic_digest`，因此把 9 个 100-customer pair 的预期
  transaction/screening/cache diagnostic 差异误报成 search semantic mismatch；
  同一 reviewer 内更严格的 native core differential 与四步 ablation 已独立证明
  search semantics 相等。这不是性能或解语义回退。
- performance observations 现可显式绑定 streaming core replay digest，并要求
  per-run fixed-work identity 与 replay map 精确一致；Hot Path 历史路径继续使用
  原 full-storage digest。对 E32 timing rows 使用已通过的 core identity 重算，
  performance promotion aggregate 为 `0.713761`，C/R/RC 分别为
  `0.783280`、`0.692386`、`0.713761`。E32 review 不回写，修复使用新
  commit/wheel/attempt label。

## 2026-07-29：E33 accepted 与 F18 worker contract contradiction

- `stage05.2_native_kernels_attempt33` 在 clean commit
  `2d9a5f202064a305d588de2d56bead1253ee2b43` 上完成 12/12 shards、36/36 axes；
  raw manifest SHA-256 为
  `ee08673f4bcbd0a616f9bda1f81f0a8eb0c14108df38aa5491d3ac745d4956ff`。
  独立 reviewer exit 0、raw unchanged、aggregate peak RSS
  2,402,344,960 bytes、swap 0；review manifest SHA-256 为
  `6df1b8e98238699cc4692cf3f9be91627a41e9bae716603ff8e6f5e193434a18`，
  status 为 `READY_FOR_STAGE052_ACCELERATOR_DECISION`。
- E33 performance promotion aggregate 为 `0.709855`，C/R/RC 分别为
  `0.773610`、`0.707182`、`0.655874`；四步 native ablation aggregate 为
  `0.225312`，C/R/RC 分别为 `0.209141`、`0.263753`、`0.290641`。全部
  semantics、native execution、resource、persistence 与 zero-fallback gate 通过。
- E33 的 9 个 100-customer fixed-work candidate screening occupancy 为
  `4, 4, 4, 19, 25.5, 35, 41, 45, 63`，独立中位数为 `25.5 < 32`；
  F 必须走 decision-only `GPU_NOT_JUSTIFIED`，不得启动 CUDA helper。
- `stage05.2_accelerator_pilot_attempt18` 在 raw 创建前由入口拒绝：
  `Stage 5.2 worker_count is invalid for the selected component`。根因是 F 入口只允许
  6 workers，但后续又错误要求 F worker count 等于 accepted E 的 4；当前 reviewer
  与 G adapter 均明确要求 F 为 6。修复保留 E 的 accepted D width（4），同时固定
  F/G adapter width 为 6，并只验证 E predecessor width 属于合法的 1/2/4 selection。
  F18 日志与 stale host receipt 保留，修复使用新 commit 与 attempt label。

## 2026-07-29：F19 reviewer identity transition 修复

- `stage05.2_accelerator_pilot_attempt19` 在 clean commit
  `85dc49acd65c707ac11a196e5fad5a506fb82211` 上按固定 6-worker adapter 完成
  decision-only evidence；producer 判定为 `GPU_NOT_JUSTIFIED`，独立重算的 9 个
  occupancy 输入中位数为 `25.5 < 32`，无 GPU rows、无 CUDA service、无
  fallback。raw manifest SHA-256 为
  `2bed3fed687b2c225107611c9b05913ba9c7f1724efd435635d4a7dfa176dd2b`。
- 首次独立 review 完整退出但返回 `NOT_READY`，review manifest SHA-256 为
  `fc7546775800c6fd592dd11da67f63bf52d530af0e7e503c61aac95432562710`。
  decision schema、occupancy、source snapshot、staging root 与 worker selection
  均通过；失败仅为 E-to-F worker identity transition 与 runtime machine telemetry。
- 根因一是 reviewer 仍要求 F worker count 等于 accepted E 的 4，与已经明确的
  E=4、F/G=6 adapter contract 冲突。reviewer 现复用 producer 的
  `_validate_accelerator_worker_transition`，要求合法的 E width 和固定 F=6，而不把
  两者错误视为相等。
- 根因二是 reviewer 对完整 runtime identity 做 byte-for-byte comparison，而 WSL2
  dynamic memory（动态内存）的 `memory_bytes` 在相邻调用之间可变化 4096 bytes。
  wheel、Python、native extension、dependency、source mount 与其余 machine
  identity 仍严格相等；仅使用既有的 WSL review-memory normalization 排除该动态
  telemetry，CPU/GPU/OS 等硬件漂移仍会 fail fast。F19 evidence 不回写，修复使用
  新 commit、wheel 和 attempt label。

## 2026-07-29：F20 accepted 与 G Pilot93 prerequisite role 修复

- `stage05.2_accelerator_pilot_attempt20` 在 clean commit
  `c0ded60b1a5dfc6e18852e8ca8c93a0bb14be745` 上完成 decision-only evidence；
  raw manifest SHA-256 为
  `7a2c06966eafa1fe711a534faa5c4c5fd2bd61328c81d2f01db143f9c89bd09e`。
  独立 reviewer 完整退出并通过全部 gate，review manifest SHA-256 为
  `5633f04744cbfda5e0472441d008fec5de9a1b6f94652a13865864843cdcc24e`，
  status 为 `READY_FOR_STAGE052_BENCHMARK`。最终 accelerator decision 为
  `GPU_NOT_JUSTIFIED`，occupancy median `25.5 < 32`，Stage 5.2 保留 native CPU。
- `stage05.2_benchmark_attempt93` 在任何 shard 启动前 fail fast，错误为
  `KeyError: 'accepted_pilot'`；host exit 1、producer/controller log 与可能生成的
  partial startup evidence 保留，label 不复用。
- 根因是 benchmark campaign implementation 将 Pilot 错误映射到未定义的
  `accepted_pilot` role，并要求 Pilot predecessor 已经是
  `READY_FOR_STAGE052_FORMAL_BENCHMARK`；这把 Formal 的 predecessor contract
  错套到了 Pilot。修复后的唯一映射为：Pilot 使用
  `accelerator_selection` / performance / `READY_FOR_STAGE052_BENCHMARK`，Formal
  使用 `campaign_pilot` / pilot / `READY_FOR_STAGE052_FORMAL_BENCHMARK`。
  新 Pilot 使用新 commit、wheel、snapshot 和 label。

## 2026-07-29：G Pilot94 reviewer resource-lock 修复

- `stage05.2_benchmark_attempt94` 在 clean commit
  `12d027fd222738acde923ec07ae695682b8ffd9b` 上完成 36/36 shards、36/36
  axes、1,080 declared solver seconds 与 144 checkpoints；三个 batch 均已
  checksum-verified 后归档至签名的 D archive root，Windows Scheduled Task 与
  WSL controller 均 exit 0。
- 首次独立 review 完整退出但返回 `NOT_READY`，review manifest SHA-256 为
  `dfc2ed01b3d5d0291da6c62c29cd2a42ad11dee466a98a184694021c41b65c7a`。
  reviewer 在 batch0001 的第一个 provenance check 处中止，因此其余 35 个
  shards 与 batch0002/0003 未进入 replay；三个不可变 archive batch 均仍物理存在，
  每个 519 files。
- 根因是 accelerator prerequisite adapter 已验证 F20 的 scientific/runtime
  selection lock，但构造 Pilot selection lock 时漏加 campaign 已签入的
  `producer_resource_contract`。同一遗漏同时使 `accepted_prerequisite` 失败，并
  使 batch metadata 与 campaign resource contract 比较失败。reviewer 现只从
  campaign 的已验证 typed contract 补入该字段，并要求 contract worker count 与
  campaign selected workers 精确相等；backend、native、runtime、input 与 source
  identity 检查保持不变。
- Pilot94 raw 与失败 review 不回写、不晋级；修复使用新 commit、producer/reviewer
  wheels、sealed source snapshot 与新 Pilot label。

## 2026-07-30：G Pilot95 reviewer host-liveness interruption

- `stage05.2_benchmark_attempt95` 在 clean commit
  `031822a9cf598ee0cddcb4ad6aa708056d41c0b8` 上完成 36/36 shards、36/36
  axes、1,080 declared solver seconds 与 144 checkpoints；三个 batch 均归档，
  Windows Scheduled Task、WSL controller 与 completion receipt 均 exit 0。
- 独立 reviewer 已通过 accelerator prerequisite、batch0001/0002 provenance 与
  24/36 native Arrow shard replays；已完成的每个 replay 均为
  `logical_pass_count=1`、`scratch_cleaned=true`、`native_fallback_count=0`。
  process-tree peak RSS 为 1,320,587,264 bytes、swap 0，未触发 2,562,322,514-byte
  guard。
- reviewer 在 24/36 后被外部 WSL host lifecycle 中断；raw manifest 前后
  SHA-256 相等，未生成或发布 review manifest。根因不是 reviewer timeout 或
  scientific gate，而是 launcher 返回后未启动 Windows-side WSL keeper，导致没有
  长期 Windows WSL client 时 transient user service 随 WSL idle shutdown 停止。
- Pilot95 raw、partial progress 与未最终化的 interrupted service receipt 原样保留，
  不晋级、不重试同一 label。后续 reviewer launch 必须同时启动 wave-owned Windows
  `wsl.exe` keeper，并由 keeper 轮询该 exact unit 至终态；修复从新 Pilot label 建链。

## 2026-07-30：G Pilot96 exact-completion timestamp 修复

- `stage05.2_benchmark_attempt96` 在 clean commit
  `f4991e1fce8a061ea99630d273a6549d6542f898` 上完成 36/36 shards、36/36
  axes、1,080 declared solver seconds 与 144 checkpoints；三个 batch 均归档，
  Windows Scheduled Task、WSL controller 与 completion receipt 均 exit 0。
- 独立 reviewer 的 Windows-side WSL keeper 保持到 exact transient unit 终态，证明
  Pilot95 host-liveness 根因已修复。service exit 0、receipt finalized、raw unchanged、
  cgroup peak RSS 1,634,975,744 bytes、swap 0；review status 为 `NOT_READY`，
  review manifest SHA-256 为
  `05c45a3727b832380938ff64ddc719c07ab5b0f3479f05c95bb61e053fba178d`。
- reviewer 在 batch0003 shard0031（`r101_21/2014`）发现唯一根失败：
  `exact completion crosses deadline on wall_clock_30`。该 exact call 于
  `29.999507759` 秒启动，raw 错记 `completed_at=30.000085435` 秒；其余 gate
  失败均由 replay 在 24/36 后 fail fast 引起。
- 根因是 single-route exact path 已在 backend 返回后捕获 `exact_completed_at` 并据此
  正确判断事务是否在 deadline 前完成，但成功 trace 随后再次读取时钟，把 cache/
  controller bookkeeping 后的时间错误写成 exact completion time。修复后的 trace
  使用已捕获的 backend-return timestamp；后续 pre-commit deadline check、candidate
  rollback、cache rollback 与 deadline boundary 保持不变，不增加容差、不放宽 gate。
- Pilot96 raw 与失败 review 不回写、不晋级；修复使用新 commit、wheels、sealed
  snapshot 与新 Pilot label。

## 2026-07-30：G Pilot97 accepted

- `stage05.2_benchmark_attempt97` 在 clean commit
  `71f09a67898159dd8411c0c6e641d1e83196b379` 上完成 36/36 shards、36/36
  axes、1,080 declared solver seconds 与 144 checkpoints；三个 batch 均已
  checksum-verified 后归档至签名的 D archive root，Windows Scheduled Task、
  WSL controller 与 completion receipt 均 exit 0。raw manifest SHA-256 为
  `c826b22d55447431ef7734fa907a699cda39f0b33e2137bd69d145c46f1e68c7`。
- 独立 reviewer 由受限 transient `systemd --user` service 执行，并由
  Windows-side WSL keeper 保持宿主存活至 exact unit 终态。service exit 0、
  receipt finalized、raw manifest 前后 SHA-256 相同；cgroup aggregate peak
  RSS 为 1,662,361,600 bytes、swap 0，低于 2,562,322,514-byte guard。
- reviewer 使用 4 workers 和 native Arrow replay 重放 36/36 shards、
  15,096,880 logical events；maximum in-flight shards 为 4，
  `native_fallback_count=0`。candidate transaction、objective、validator、
  exact/cache/deadline、source snapshot、producer resource、persistence、
  archive identity 与 publication dry-run 等 20/20 gates 全部通过。
- review manifest SHA-256 为
  `29c0756c53bee52c29ef193bcfee952238ec04d9a243a3228fbfa764c0fc232f`，
  status 为 `READY_FOR_STAGE052_FORMAL_BENCHMARK`。该 Pilot 是后续新 Formal
  campaign 的唯一 campaign prerequisite；它本身不构成
  `READY_FOR_STAGE05_3`。

## 2026-07-30：G Formal98 prerequisite lock 与 Formal99 memory floor

- `stage05.2_benchmark_attempt98` 在任何 shard 启动前 fail fast：sealed local
  campaign lock 仍只绑定 Pilot 的 accelerator predecessor，没有绑定新接受的
  `stage05.2_benchmark_attempt97`。Attempt98 startup evidence、host exit 1 与
  controller log 原样保留；新 snapshot 使用 `upsert_stage052_campaign_lock`
  验证并加入 Pilot97 的 exact raw/review/config/runtime identity。
- `stage05.2_benchmark_attempt99` 在 clean commit
  `a2eac7f24849c2c6d717015ec6df99c0308fd401` 上通过 Formal preflight，并生成精确
  920 shards、2,040 axes、229,200 declared solver seconds、10,400 checkpoints、
  12 batches、6 workers 的签名计划。前六批共 637 shards 完成并归档；`batch0007`
  的 53/53 shard manifests 亦已落盘，但在整批 commit 前由 runtime guard fail
  fast，因此 Attempt99 不贡献 readiness geometry。
- 唯一根失败是 worker PID 63832 的 RSS 达
  `3,993,497,600` bytes，超过冻结的 `3,991,904,256`-byte per-worker hard
  limit `1,593,344` bytes（约 0.04%）。同批 process-tree aggregate peak 为
  `20,060,610,560` bytes，仍低于原 `22,184,042,496`-byte aggregate limit；
  swap/fallback 未被用作继续执行手段。abort 后 PID 63833 未在 terminate/kill
  窗口内退出，进程池整体中止；campaign、batch0007 partial artifact manifest、
  resource summary、PID peaks、host exit 1 与六个不可变 archive batches 全部保留。
- 根因是 resource calibration loader 仅接受旧的 aggregate-RSS Formal failure
  作为零 geometry memory floor，无法消费同样签名且完整绑定的 per-process RSS
  hard-limit failure。loader 现在只新增该 fail-fast memory reason 到显式白名单；
  campaign/batch/artifact/resource/PID/SHA-256、6-worker、row-group、queue-depth
  与 partial-evidence 绑定保持严格，普通 worker failure 继续被拒绝。
- 两位 `attemptNN` 空间已到 `attempt99`；修复后的新 Formal 使用下一未占用的合法
  `stage05.2_benchmark_rerun01`，以新 commit、wheels、sealed source snapshot 和
  使用 Attempt99 batch0007 签名 memory floor 重新校准的 6-worker contract 从零
  执行。Attempt99 不续跑、不导入 shard。

## 2026-07-30：G Formal rerun01 resource recalibration

- `stage05.2_resource_calibration_attempt05` 由 transient user service 启动后，
  Windows-side WSL keeper 在 service active 状态可见前提前退出；WSL idle
  shutdown 随后终止校准。该 label 的 partial 目录仅含未完成的 measurement
  目录、不含 calibration report 或 producer resource contract，原样保留且不复用。
- `stage05.2_resource_calibration_attempt06` 改由 Windows-side `wsl.exe` 直接
  托管完整 non-editable producer process，在 clean commit
  `2f97f9f6d205b2827719dc8e6027f1bcfa1a6fcc` 上完成。签名 report SHA-256 为
  `8f421fb032dd7e979921a613a504220c1c824cc2b9158282dc636e760e923576`，
  contract SHA-256 为
  `23b5e2ada31f816f8ae2b0aa4be23ca1d2795ffaa861e5b6bf5df2296480fb0f`；
  calibration digest 为
  `086f3bb8175f8ae51c8a3d8ef75f20415d988244a6cd106600409917b9aa5ede`。
- 新 contract 保持用户锁定的 6 workers、row group `262144`、queue depth `2`，
  并精确绑定 Attempt99 `batch0007` resource summary
  `cdaa627f53d14ce9a34d0054eb22cbaf4d388c80147980d428842c358599a7e5`。
  memory floor 为 aggregate `20,060,610,560` bytes、per-worker
  `3,993,497,600` bytes；20% headroom 后的 hard limits 分别为
  `24,072,732,672` 与 `4,792,197,120` bytes，未改变 Formal work geometry，
  未使用 worker downgrade、swap 或 fallback。
- `stage05.2_benchmark_rerun01` 将从零使用该 content-addressed contract、
  新 commit/wheels 与新 sealed source snapshot；Attempt99 的 637 archived
  shards 和 batch0007 partial shards 均不导入。

## 2026-07-30：G Formal rerun01 resource-lock protocol

- `stage05.2_benchmark_rerun01` 在任何 output directory 或 shard 创建前 fail
  fast；sealed runtime/source preflight 已通过，但 accepted Pilot97 selection
  lock 仍冻结旧 producer resource contract，因此拒绝 Attempt99-derived contract。
  Windows host/controller exit 1、launch nonce 与 pre-dispatch log 原样保留，
  rerun01 label 不复用。
- 根因是 resource calibration 已能安全消费 Attempt99 的 per-process RSS failure，
  但 producer/reviewer 的 campaign selection-lock protocol 尚无显式表达
  replacement Formal memory-only recalibration 的字段。直接替换 contract 会失去
  Pilot lock 的审计意义，因此没有绕过或放宽 equality gate。
- 新 public loader 验证 signed v2 calibration report/sidecar、clean calibration
  revision、Attempt99/batch0007/resource-summary identity、零 readiness geometry、
  固定 6-worker/row-group/queue-depth topology、fresh/producer 两组各自精确且唯一的
  `{4,5,6}` worker identity set、跨 worker semantic equality、完整 20% headroom，
  以及 report 与 exact contract equality。独立 Formal memory probe 使用其自身的
  可验证 semantic digest，aggregate/per-worker peak 均不得超过 replacement
  contract 的对应 selected peak。`BenchmarkExecutionLock` 仅在该 evidence 存在时
  允许 memory floor 单调增加，并在 effective selection lock 中同时保留 predecessor
  contract、replacement contract 与 recalibration receipt。
- producer 和 independent Formal reviewer 通过同一 loader 与 execution-lock
  method 重算；source snapshot allowlist 仅新增 report 与 sidecar 两个固定本地
  文件。缺失/篡改 report、非零 geometry、拓扑变化、降低 memory floor 或 checksum
  不一致全部 fail fast。修复后的 replacement 使用新 commit、wheels、sealed
  snapshot 与 `stage05.2_benchmark_rerun02` 从零执行。

## 2026-07-30：G Formal rerun02 aggregate-memory accounting 根因修复

- `stage05.2_benchmark_rerun02` 在 clean commit
  `c45d748aa0a59808d777f88989a168a1325900b4` 上完成并归档 batch0001--0007，
  共 678/920 shards；batch0008 在 `r205_21` wave 中完成 12 shards 后触发
  aggregate guard。raw 显示 process-tree RSS 总和
  `24,099,033,088` bytes，内部门槛 `24,072,732,672` bytes；Windows host/
  controller exit 1，batch0008 partial evidence 与前七个不可变 archive batches
  全部保留，attempt99、rerun01、rerun02 均不续跑、不导入 shard。
- 同一失败 wave 的独立 transient systemd probe 显示 cgroup physical peak 仅约
  2.5 GiB、swap 0，而内部进程 RSS 总和约 24.1 GiB。根因是六个 spawned workers
  的共享映射被逐进程 RSS 累加六次；它不是物理内存耗尽、page-cache 累积或
  row-group/queue-depth 不足。
- resource evidence 升级为 v4：aggregate hard gate 只读 dedicated cgroup v2
  `memory.current`，签入 `memory.peak`、`memory.swap.peak` 与 exact cgroup path；
  process-tree aggregate RSS 继续保留为兼容遥测，per-worker RSS 继续独立硬限制。
  Formal 有 aggregate limit 时若处于 `/init.scope`、共享 user service、缺少 memory
  controller 文件或无法验证 service cgroup，必须在工作开始前 fail fast，不提供
  RSS fallback。
- Formal memory probe 改为失败时也封存完整 resource summary；worker abort 先
  shutdown/cancel，再 terminate，并给 SIGKILL 后的操作系统 reap 10 秒有界窗口，
  survivor 仍作为显式失败。新 calibration report v3 精确绑定
  `rerun02/batch0008` resource summary
  `ec5e8eb43562b983ac0b3d733da446f45fa59407b278dff93323ce4d6e6e6253`；
  predecessor RSS 总和仅作失败 provenance，replacement aggregate floor 必须来自
  完整 R205 cgroup measurement，per-worker floor 与固定 6-worker topology 保持。

## 2026-07-30：G Formal memory attempt07/08 与低内存 row-group 候选

- `stage05.2_formal_memory_probe_attempt07` 在 clean commit
  `d108be9c1d9916a3684fdda857e0b50782730ea3` 上完成 exact `r205_21`、
  seeds 2014--2019、30/60/300-second axes 与固定 6 workers。签名 v3 report
  记录 dedicated cgroup physical peak `24,123,187,200` bytes、cgroup swap
  peak 0、per-worker RSS peak `4,946,546,688` bytes；20% headroom 需要
  `28,947,824,640` bytes，超过 host capacity `25,196,929,024` bytes，因此
  attempt07 是完整诊断证据但不满足 Formal resource gate。
- host-wide `psutil` swap delta 只保留为 telemetry；Formal 硬门槛只使用专用
  cgroup v2 `memory.swap.peak == 0`。独立回归测试覆盖“host 有 swap 变化但
  workload cgroup swap 为 0”的情况；fallback 和 resource-limit 仍 fail fast。
- `stage05.2_formal_memory_probe_attempt08` 尝试 16,384-row diagnostic A/B，
  但 d108be9 CLI 的显式允许集合仍只有 65,536/262,144，因参数校验以 exit 2
  fail fast。该 label 与 journal 原样保留且不复用，不产生 readiness geometry。
- artifact storage、resource contract、calibration candidate matrix 与 CLI
  现在显式支持 16,384-row low-memory candidate。live trace buffer 使用签名
  row-group selection；这不是静默 fallback，也不改变 candidate order、
  objective、exact/cache/deadline 或 evidence schema。下一次内存探针必须使用
  新 clean commit/wheels/sealed source 与新 label。

## 2026-07-30：G Formal memory attempt09 keeper 失败

- 低内存根因修复在 clean commit
  `3ca5297f65ad97066413fae61e0585238b866fae` 上通过完整 pytest 936、
  Ruff、strict mypy、diff-check 与双轴独立 code review；producer/reviewer
  wheel SHA-256 均为
  `08227d49b0613d76e33f361bd72774457fb8e57c75a418a3edb255fb13ee8e71`。
- `stage05.2_formal_memory_probe_attempt09` 使用 16,384-row、queue depth 1、
  exact `r205_21` seeds 2014--2019 与固定 6 workers 启动，但 Windows-side
  `wsl.exe` keeper 的参数转义错误使 keeper 立即退出。WSL 随后停止 dedicated
  transient unit；journal 记录约 3 分 14 秒 CPU、2.3 GiB service memory peak
  与 swap 0，但探针未完成，也未封存 signed v3 report。
- attempt09 是不可变 partial failure evidence，不续跑、不复用 label，也不参与
  resource calibration 或 readiness。替代探针必须先验证 keeper 能跨过一个
  transient unit 生命周期，再使用新 clean commit/wheels/sealed source 与新
  label 从零运行。

## 2026-07-30：G Formal memory attempt10 与 bounded safe-rejection 根因修复

- `stage05.2_formal_memory_probe_attempt10` 使用 clean commit
  `6e9112a44025d1f41803260165898356fd3a5577`、exact `r205_21` seeds
  2014--2019、固定 6 workers、16,384-row、queue depth 1，并由经过 lifecycle
  smoke 的 Windows-side keeper 持有 dedicated transient unit。signed v3 failure
  report SHA-256 为
  `182e9aae3e01ffccbb2c1decf472cbac2fe43446db3f8d2799026c877d1e4852`；
  exit 1、cgroup swap peak 0、clean provenance。约 34 分 44 秒后 direct cgroup
  guard 观察到 `memory.current=24,124,645,376` bytes 超过
  `24,123,195,392`-byte limit；resource summary peak 为
  `24,133,230,592` bytes，20% headroom 为 `28,959,876,711` bytes，高于 host
  capacity `25,196,937,216` bytes。
- attempt10 比 65,536-row attempt07 的 `24,123,187,200`-byte peak 反而高
  `10,043,392` bytes（0.0416%），因此 row-group 16,384 方向被实证否定。
  attempt10 原样保留且不复用；resource calibration 与 Formal rerun03 继续阻塞。
  terminal journal 的约 5.5 GiB 峰值不覆盖 signed report：独立 single/
  multiprocess cgroup harness 已证明 direct `memory.current`/`memory.peak`
  采样与 live systemd 值一致，Stage 5.2 resource gate 继续使用 dedicated-cgroup
  direct sampling。
- 对 rerun02 最大 screening shard（1,226,847 definitions）的只读重放分离了三种
  working set。完整 1,250,000-entry SHA-256 identity store 的 RSS 增量约
  102 MiB，因此降低 2,097,152 collision capacity 既会破坏 exact proof，也不足以
  修复多 GiB 峰值。相反，442,044-entry scalar safe-rejection result mapping
  增量约 656 MiB，262,144-entry native definition memo 增量约 469 MiB；同一
  raw shard 另有 442,044 unique rejected routes，证明可丢弃 memo/cache 才是主要
  可控增长源。
- 当前修复保留完整 2,097,152-entry full-SHA-256 collision state；将 scalar
  safe-rejection result 设为可审计 65,536-entry LRU，将 Python/native packed
  safe-rejection sequence state 设为 65,536-entry atomic generation cache，并将
  producer definition-key memo 设为 65,536-entry FIFO safe-recompute cache。
  Python/native generation rollover 共用 candidate transaction commit/rollback；
  淘汰只触发 safe screening 重算，不改变 candidate order、objective、exact budget、
  exact/cache/deadline semantics 或 collision proof。
- raw screening/candidate-transaction statistics 记录 capacity、current/peak、
  stores、evictions 与 rollovers；campaign reviewer 拒绝无界、缺失或内部不一致的
  cache evidence。signed screening-definition contract 升级为 v3，分别绑定完整
  collision capacity、memo capacity、FIFO safe-recompute policy 与
  `identity_collision_proof=full_sha256`。在新 clean commit、完整质量门槛、
  双轴 code review、新 wheels/venvs/runtime/sealed source 和下一未占用 memory
  probe label 完成前，状态仍为 `NOT_READY`。

## 2026-07-30：G Formal memory attempt11 evidence-token 根因修复

- bounded safe-rejection 修复在 clean commit
  `a2a16335a478c6f0bf63e806712b0ef936c72ee7` 上通过完整 pytest 943、
  Ruff、strict mypy、diff-check 与双轴 code review；producer/reviewer wheel
  SHA-256 均为
  `91f4a5fb67dc3e8136bec2bbcaa8b10bbb58221ce1714b888400d94ca8f9bd46`。
  `stage05.2_formal_memory_probe_attempt11` 使用 exact `r205_21` seeds
  2014--2019、固定 6 workers、16,384-row、queue depth 1，并由 Windows-side
  `wsl.exe` keeper 持有 dedicated transient unit。
- attempt11 在约 92 秒后 fail fast；signed v3 failure report SHA-256 为
  `abc23985028e827f37a65a9b0ee9f035c63b5a767eef5455247d891606f8cf46`，
  exit 1、clean `a2a1633` provenance、cgroup physical peak
  `7,178,526,720` bytes、cgroup swap peak 0，最高 worker RSS
  `1,328,115,712` bytes。该 peak 是失败前 partial scope 的诊断值，不是完整
  Formal memory acceptance measurement；attempt11 原样保留且不复用。
- seed 2017 的原始 partial failure 首先记录
  `negative screening cache returned inconsistent evidence identity for one route`；
  其他 worker 的 `no default __reduce__ due to non-trivial __cinit__` 是并行失败
  传播/清理时的次生序列化错误。根因是 bounded LRU 淘汰后，同一路线的等价安全
  拒绝被重新计算为新的 Python object，而 streamed evidence 仍使用 `id(result)`
  作为持久身份 marker。
- Stage 5.2 bounded LRU 路径现在用 collision-free canonical route key 与完整
  normalized screening result 的 SHA-256 派生稳定正整数 lookup marker。等价安全
  重算保持同一 marker；stream sink 同时保存无 hash 截断的完整 compact binary typed
  result signature（字符串长度前缀、bool/None type tag、IEEE-754 原始 64-bit float），
  并对 marker 与完整 bytes 比较，因此 63-bit marker 截断或碰撞不能隐藏任一筛选
  证据字段变化，也不会恢复 object-heavy evidence retention。相同的
  `(marker, complete signature)` 复合值继续进入 deferred native occurrence identity；
  即使 route-key guard 已 FIFO 淘汰，旧 marker-only occurrence 也不能绕过完整签名。
  未启用 bounded LRU 的历史路径保持原有 object-identity 行为。回归测试覆盖等价
  重算、相同 marker 下的证据变化、guard-eviction 后的 native occurrence collision
  路径和实际 LRU evict/recompute/hit 序列。新修复必须重新完成
  clean commit、完整质量门槛、双轴 code review、wheels/venvs/runtime/sealed
  source，并使用下一未占用 probe label；resource calibration 与 Formal rerun03
  继续阻塞，状态为 `NOT_READY`。

## 2026-07-30：G Formal memory attempt12 sealed-input preflight

- attempt11 的稳定 evidence identity 修复在 clean commit
  `20302404a301e6a5aa219a7d70e59bc988c3979b` 上通过完整 pytest 946、
  Ruff、strict mypy、diff-check 与双轴 code review。producer/reviewer wheel
  SHA-256 均为
  `6600f4c1d17f490db1c9c92598d7f66f060796cc804720cce34ef2b704216dd5`，
  reviewer wheel provenance sidecar 绑定同一完整 revision。
- `stage05.2_formal_memory_probe_attempt12` 使用该 clean runtime、固定 6 workers、
  16,384-row、queue depth 1 和正确的 Windows-side `wsl.exe` keeper 启动；dedicated
  transient unit 与 sealed working directory 均已建立。但新 sealed checkout 只做了
  Git clone，没有实体化 Git-ignored Schneider input，因缺少
  `data/schneider/r205_21.txt` 在约 1.09 秒内 fail fast。signed v3 failure report
  SHA-256 为
  `01bb9ed520b0028d8cf053185db26c7a242f2b5c9159b133fbe0fc49b5f107d3`；
  exit 1、clean `2030240` provenance、cgroup swap peak 0。该运行没有完成任何
  memory acceptance geometry；attempt12 原样保留且不续跑、不复用。
- 根因是 active checkout 的 `data/schneider` 为 ignored symlink，而历史 sealed
  sources 使用 snapshot-local ordinary file copies；普通 clone 无法携带该输入。
  Formal memory probe 现在在创建 output root 和提交 worker work 前要求 exact
  `r205_21.txt` 存在且非空。回归测试证明缺失输入不会调用 runner，也不会创建
  output root。下一 sealed source 必须实体化并核验 Schneider input，使用新的 clean
  commit、wheels/venvs/runtime 与未占用 probe label；resource calibration 和 Formal
  rerun03 继续阻塞，状态仍为 `NOT_READY`。

## 2026-07-31：G Formal memory attempt13 与 retained hot-state 根因修复

- `stage05.2_formal_memory_probe_attempt13` 使用 clean commit
  `e3a4edd50e3800b24455712ef2a3c330570c2a69`、exact `r205_21` seeds
  2014--2019、固定 6 workers、16,384-row、queue depth 1，在 dedicated transient
  unit 中完成 6/6 shards。service exit 0，signed v3 report SHA-256 为
  `7433086ffc1becc86b3402bfe70c33511c19f912d2dc2ff12f9a5115dc698957`；
  aggregate cgroup peak `23,965,954,048` bytes，最高 per-worker RSS
  `4,734,369,792` bytes，cgroup swap peak `8,192` bytes。20% headroom 需要
  `28,759,144,858` bytes，超过 host capacity `25,196,933,120` bytes
  `3,562,211,738` bytes，因此 attempt13 是不可变 complete diagnostic evidence，
  但不满足 resource gate；resource calibration 与 Formal rerun03 继续阻塞。
- 独立 disposable cgroup harness 同时读到 direct cgroup peak
  `1,232,916,480` bytes，而 systemd terminal 仅报告 `512.0K`。因此 WSL user
  manager 的 terminal `Memory peak` 会严重低报，不能覆盖 signed report 的 direct
  dedicated-cgroup sampling；attempt13 的非零 swap 同样按原始证据保留，不人工
  改写为通过。
- TDD memory-composition seam 对实际 native store 分项测量：完整
  2,097,152-capacity definition identity store 在 1,226,847-entry Formal 上界附近
  只增加约 96 MiB RSS，因此继续保留完整 full-SHA-256 collision proof。相反，
  65,536-entry native definition memo 的 synthetic persistent delta 约 329 MiB；
  route dictionary/unique-route identity 的两个 262,144-entry hot stores 合计约
  360 MiB。主因是可重算/可精确 spill 的 retained hot state，不是 identity capacity。
- 修复将 native/Python definition-key memo 绑定到一个 8,192-row live screening
  transaction，并将 signed contract 升级为
  `stage05.2-screening-definition-store-v4`。route dictionary identity 与 exact
  unique-route identity 的内存阈值各降至一个 65,536-row Parquet group；超过阈值
  继续使用既有 shard-local SQLite 保存完整 digest/payload，duplicate/collision
  proof 不变。新增 native cache size/capacity 只读 introspection，使测试能核对真实
  C++ capsule bound，而不是只信 Python 常量；campaign successor 的 pinned
  producer-fix SHA-256 同步更新，旧 native/test 内容不能绕过 exact pin。
- 同一 disposable synthetic seam 的修复后 persistent delta：definition memo 约
  192 MiB，route/unique identity hot state 合计约 7.8 MiB；估算每 worker 合计减少
  约 523 MiB，六 worker 约 3.14 GiB。该数值只用于决定下一 probe 是否值得运行，
  不是 readiness evidence；必须在新 clean commit、完整质量门槛、双轴 code review、
  新 wheels/venvs/runtime/sealed source 与新未占用 Formal memory probe 上复核。
  在该 probe 通过 signed v3、exit 0、exact scope、cgroup swap 0、per-worker RSS、
  clean provenance 和 20% headroom 前，状态仍为 `NOT_READY`。

## 2026-07-31：G Formal memory attempt14 与可重算 sparse memo 限界

- `stage05.2_formal_memory_probe_attempt14` 使用 clean commit
  `c4f63af51d35dd581d6e53902909a11d521a5ed1`、exact `r205_21` seeds
  2014--2019、固定 6 workers、16,384-row、queue depth 1，在 dedicated transient
  unit 中完成 wall-clock 30/60/300 三轴 scope。service exit 0，signed v3 report
  SHA-256 为
  `4752e0712ce7704479e9c166a1971a1fb628d0bd9514fb39438d089f1edc1bb3`；
  aggregate cgroup physical peak `21,280,231,424` bytes，最高 per-worker RSS
  `4,138,356,736` bytes，cgroup swap peak 0，fallback count 0。20% headroom
  需要 `25,536,277,709` bytes，超过 host capacity `25,196,941,312` bytes
  `339,336,397` bytes。因此 attempt14 是不可变 complete diagnostic evidence，
  但 resource calibration 与 Formal rerun03 仍被阻塞，状态为 `NOT_READY`。
- attempt14 的 worker 同时完成说明剩余差额不是由未完成 scope 估算。通过只读 raw
  复核，`cache_event` 与 `route_evaluation` 的可选 extras 组合分别达到约
  85,000 与 52,000 个，而 exact route LRU 仅约 4,096 entries、约 2.2 MiB；
  因此 solver exact cache 不是主要 retained state。另一个 disposable full-identity
  seam 在 900,000 definitions 与 419,000 routes 附近只增加约 106 MiB RSS，
  也否定了直接缩小 2,097,152 definition identity capacity 的方向。
- 使用 attempt14 seed 2018 的真实 extras 进行独立 RSS A/B：将可重算 sparse-extras
  memo 从 65,536 限到 8,192，单进程减少约 40.7 MiB；对 route-ID resolution memo
  做同样限界，再减少约 28.4 MiB。合计约 69 MiB/worker，高于当前 gate 差额所需的
  约 47 MiB/worker。该 A/B 只证明新 probe 值得运行，不作为 readiness evidence。
- route-ID resolution、route-evaluation extras 与 cache-event extras 都是纯
  deterministic recomputation memo：淘汰后重新解析 canonical route key 或重建
  canonical JSON，输出 row 必须逐字段相同。三者统一绑定一个 8,192-row live
  transaction 与 FIFO safe-recompute policy；signed contract 升级为
  `stage05.2-screening-definition-store-v5`。完整 definition full-SHA-256 state、
  disk-backed route identity、完整 payload/collision proof 和 Parquet rows 均不删除。
  回归测试强制 memo 淘汰后 route-evaluation/cache-event row 除 event ID 外完全相同。
- 新实现必须通过 clean commit、完整质量门槛、双轴 code review、新
  wheels/venvs/runtime/sealed source，并使用下一未占用 Formal memory probe label。
  只有该 probe 的 signed v3、exit 0、exact scope、cgroup swap 0、per-worker RSS、
  clean provenance 与 20% headroom 全部通过，才允许生成 resource contract。

## 2026-07-31：G Formal memory attempt15 与 typed negative-evidence guard 根因

- `stage05.2_formal_memory_probe_attempt15` 使用 clean commit
  `5902043c0cb8c998368061706c47426cb88d1aa0`、producer/reviewer wheel
  SHA-256
  `beea1a720d2dfd8c07c92cba5de97929095bbe441a5e10c6263b5bcd2fab99e5`、
  exact `r205_21` seeds 2014--2019、固定 6 workers、16,384-row、queue depth 1，
  在 Windows-side `wsl.exe` keeper 持有的 dedicated transient unit 中完成全部
  wall-clock 30/60/300 轴。signed v3 report SHA-256 为
  `c9d8c7258bbd6d9c5268ba626e484a16d4ca5a71a57b7ef4e6dab24db2c8a592`；
  service exit 0、aggregate cgroup peak `21,451,608,064` bytes、最高
  per-worker RSS `4,182,781,952` bytes、cgroup swap peak 0、fallback count 0。
  20% headroom 需要 `25,741,929,677` bytes，超过 host capacity
  `25,196,937,216` bytes `544,992,461` bytes。因此 attempt15 是不可变 complete
  diagnostic evidence，但仍为 `NOT_READY`；resource calibration 与 Formal
  rerun03 不得启动。
- attempt15 与 attempt14 的真实峰值差异证明 8,192-entry route-ID/sparse-extras
  memo 不是峰值主导项。只读重放 screening definitions 与 route dictionary 后，
  每个 seed 有 `214,188`--`282,330` 条 unique negative-cache route identity；
  当前 typed trace guard 会达到原 262,144-entry 上限，并为每条路线同时保留完整
  route key、positive token 与 complete binary signature。
- 使用 attempt15 seed 2015 的 `282,330` 个真实 negative route keys 做独立 RSS
  A/B，保持相同完整 signature 形状：262,144-entry guard 增加
  `153,911,296` bytes，而 8,192-entry guard 不再形成可测的额外 persistent delta。
  约 154 MiB/worker、六 worker 约 924 MiB 的差额超过 attempt15 所需的约
  545 MiB，因此该根因修复值得进入下一 probe；该 A/B 仍不是 readiness evidence。
- typed negative-evidence route guard 仅验证同一路线在仍驻留时的
  `(token, complete signature)` 一致性。既有设计已经把同一完整 pair 放入 deferred
  occurrence，native occurrence cache 即使在 route-guard 淘汰后仍比较完整 signature；
  因此 FIFO eviction 只导致 safe recomputation，不删除 definition、occurrence、
  full-SHA-256 collision state 或 Parquet row。guard 固定为一个 8,192-row live
  transaction，signed contract 升级为
  `stage05.2-screening-definition-store-v6` 并显式绑定 limit/policy。红测先证明
  三个 unique routes 不会按 2-entry test bound 淘汰，修复后验证淘汰与首次路线重现
  均不改变 evidence row。
- 新修复仍必须形成新的 clean commit、完整质量门槛、双轴 code review、新
  wheels/venvs/runtime/sealed source，并使用下一未占用 Formal memory probe label；
  attempt15 不续跑、不复用、不导入。

## 2026-07-31：G Formal memory attempt16 host-working-directory failure

- `stage05.2_formal_memory_probe_attempt16` 绑定 clean commit
  `3ee20fa9ec71bd84f4a0962bb09f6812f8ad7a20` 和 wheel SHA-256
  `8ee0dfd106f127cd70d1be6790765e7bf24efeda3377c0afdf512cd42662c477`，
  但 Windows-side keeper 构造的 transient `systemd --user` invocation 漏掉了
  working directory。三次 host invocation 均在约 0.3 秒内由
  `repository_root()` fail fast：当前目录 `/home/oneblaze` 和 installed wheel
  均不是 Git worktree；service status 1，memory peak 不超过 760 KiB、swap peak 0。
  solver、worker 与 output root 均未创建，因此没有 signed v3 report 或 readiness
  geometry。journal 是该 label 的不可变 host failure evidence；attempt16 不续跑、
  不复用、不导入。
- 根因不是 candidate transaction，而是 calibration CLI 把 repository identity
  隐式依赖于 process cwd。红测先证明 CLI 不接受显式 repository root；修复后
  `--repository-root` 成为所有 probe/calibration mode 的必需参数，并作为 `root`
  显式传入 deep runner。后续 systemd command 不再依赖 WorkingDirectory 才能定位
  source；缺少 binding 会在 argparse 阶段、启动 evidence 前失败。
- 该 host-orchestration 修复必须再次形成 clean commit、完整质量门槛、双轴
  code review、新 wheels/venvs/runtime/sealed source，并使用下一未占用 label。
- 首次双轴复核拦截了一个尚未消费新 label 的残留 cwd 依赖：相对 `--config`
  仍由 `Path.resolve()` 基于 process cwd 解析。新增红测使用错误 cwd、显式
  repository root 与相对 config，随后统一把相对 config 解析为
  `repository_root / config`；绝对 config 保持原 identity。该 finding 修复并重审
  之前不得启动下一 probe。

## 2026-07-31：G Formal memory attempt17 sealed-source packaging failure

- `stage05.2_formal_memory_probe_attempt17` 使用 clean commit
  `37ad9ff9a81a688e818c8d137299461583be7546`、wheel SHA-256
  `832435c2a036048c67c52be5cc8aff6dd303ff42dac25495df5af8fcb666342d`
  和显式 `--repository-root` 启动，但 host packaging 把 sealed source 做成
  `git archive`，同时错误地把 active checkout 作为 repository root。active checkout
  的 `data/schneider/r205_21.txt` 是指向 D: 的 symlink；producer 在约 0.4 秒内由
  snapshot-local ordinary-file gate fail fast。service status 1、memory peak
  256 KiB、swap peak 0；solver、worker 与 output root 均未创建，没有 signed v3
  report 或 readiness geometry。journal 与错误 sealed tree 作为不可变 failure
  evidence 保留；attempt17 不续跑、不复用、不导入。
- replacement sealed source 必须由 clean revision 通过
  `git clone --local --no-hardlinks` 创建，随后把 exact `r205_21` 实体化为 snapshot
  内 ordinary file。启动前必须独立验证 `.git` 存在、HEAD 精确、status clean、
  input 非 symlink 且 SHA-256 精确；缺一项不得启动下一 label。

## 2026-07-31：G Formal memory attempt18 通过与 rerun02 floor 兼容修复

- `stage05.2_formal_memory_probe_attempt18` 使用 clean commit
  `fac294b89e107826b021f61634026bdf2654f7b0`、producer/reviewer wheel
  SHA-256
  `519e3b0b6bae1bbfef22c94e07fa8806624deb05272f26b9382d184f011b152e`，
  exact `r205_21` seeds 2014--2019、固定 6 workers、16,384-row、queue depth 1，
  在 Windows-side keeper 持有的 dedicated transient user service 中完成全部
  wall-clock 30/60/300 轴。signed v3 report SHA-256 为
  `7390cf82f7638ee4609e4b897d76456b32e562a16abfa18099c40bf0760efbc6`；
  service exit 0、aggregate cgroup peak `18,879,209,472` bytes、最高
  per-worker RSS `3,699,142,656` bytes、cgroup swap peak 0、fallback count 0。
  20% headroom 为 `22,655,051,367` bytes，低于 host capacity
  `25,196,933,120` bytes，余量 `2,541,881,753` bytes；因此该 probe 通过
  resource-calibration prerequisite，但本身仍不贡献 Formal campaign geometry。
- resource calibration 的预检随后正确拒绝消费新 label：现有 failed-Formal floor
  loader 只认可历史的 campaign-lock 与 per-process RSS failure marker，未认可
  rerun02 batch0008 已签名的 aggregate RSS hard-limit marker。该 evidence chain
  的 resource summary SHA-256 仍精确为
  `ec5e8eb43562b983ac0b3d733da446f45fa59407b278dff93323ce4d6e6e6253`；
  兼容修复仅扩展 fail-fast memory reason allowlist，不放宽 signed campaign、
  batch、artifact、run metadata、resource summary、worker/row-group/queue-depth
  或 checksum binding。新增回归测试覆盖 rerun02 的精确 failure-reason 形状；
  非内存 worker failure 仍必须 fail fast。

## 2026-07-31：G Resource calibration attempt07 physical-lock failure

- aggregate RSS floor 兼容修复在 clean commit
  `a3c9c97d5f96694953df6f3135def0bf3839cbb5` 通过 952 项完整测试、Ruff、
  71 source files strict mypy、diff-check 与双轴 code review；新
  producer/reviewer wheel SHA-256 为
  `d8087c67db5ca1ce2b1fc9813c00108f09cc1dcd52ffff6b474404794b63d3a8`。
  `stage05.2_resource_calibration_attempt07` 使用该 clean runtime、固定 6 workers
  和 rerun02/batch0008 exact resource-summary SHA-256
  `ec5e8eb43562b983ac0b3d733da446f45fa59407b278dff93323ce4d6e6e6253`
  启动于 Windows-side keeper 持有的 dedicated transient user service。
- full calibration 虽然测量了 16,384/1 low-memory pair，却忽略 CLI physical
  configuration，按独立 Parquet persistence timing 自动选择 262,144/2，并把该 pair
  用于最终 `r205_21` Formal memory measurement。约 6 分 49 秒后 cgroup 达到
  `MemoryMax=24,123,191,296` bytes，systemd 以 `oom-kill` 终止 unit；
  memory swap peak 0。output root 保留 258 个 partial files，但未生成
  calibration report、signed contract 或 readiness geometry。attempt07 是不可变
  host/resource failure evidence，不续跑、不导入、不复用。
- 根因是 formal probe 与 full calibration 之间缺少 exact physical-configuration
  binding。修复保留完整 6-combination measurement，但允许调用方显式锁定一个已测
  `(row_group_size, queue_depth)`；locked pair 必须存在、与 65,536/1 baseline
  semantic digest 相同且不超过 host memory capability，否则 fail fast。
  CLI 只有在 row-group 与 queue-depth 同时显式提供时才形成该 lock；缺一项拒绝，
  两项均省略时才保留历史 performance selection。signed v3 report 记录
  `parquet_policy=user_locked`，Formal memory measurement 与最终 contract 必须复用
  同一 exact pair。replacement calibration 必须使用新 clean commit/runtime/sealed
  source 与下一未占用 label。
## 2026-08-01：Resource calibration attempt21 reviewer predecessor-topology 修复

- `stage05.2_resource_calibration_attempt21` 在 clean commit
  `aad33e2158b5e42309e0594a6caa450ce44223a0` 上完成冻结的 6-worker、16,384-row、
  queue-depth-1、`PYTHONMALLOC=malloc` 校准。producer terminal inventory 正确重放
  direct `artifact-storage-v2` shard manifests 并进入 `SEALED`；systemd service exit 0、
  service lifetime peak 显示为 `16.7G`、swap peak 0。signed v3 report 记录 scoped
  aggregate cgroup peak `18,160,549,888` bytes，合同 aggregate limit 为
  `21,792,659,866` bytes，均低于 host capacity `25,196,937,216` bytes。
- 独立 reviewer 随即 fail fast：它错误地要求 immutable predecessor
  `stage05.2_benchmark_rerun02/batch0008` 的物理 topology
  (6 workers, 262,144-row groups, queue depth 2) 等于新合同冻结的
  (6 workers, 16,384-row groups, queue depth 1)。producer 和既有 calibration
  tests 已明确允许两者不同：前者只提供 signed failure provenance 与 per-worker RSS
  floor，后者才是 replacement calibration 的执行 topology。该要求使冻结协议不可验收，
  属于 reviewer gate contradiction，不是 raw evidence 漂移。
- 修复后 v3 reviewer 分别绑定 predecessor 的 exact immutable topology 与 contract 的
  exact locked topology；仍要求 predecessor run/batch/resource-summary SHA-256、6-worker、
  per-worker RSS floor、零 geometry contribution，以及 contract selection 全部精确一致。
  v2 Attempt99 兼容路径仍要求 floor 与其历史 contract topology 相同。新增回归测试覆盖
  “不同但各自精确”的合法 v3 形状和错误 predecessor topology 的 fail-fast 拒绝。
- Attempt21 进入 `CLASSIFIED/current_accepted_full` 后，归档前交叉检查又发现 lifecycle
  content inventory 与 storage-v2 retention receipt 使用不同的 canonical tree encoding：
  两者文件集、逐文件 size/SHA-256 和总 bytes 完全相同，但 tree SHA-256 必然不同。
  原 binding writer 把 storage tree 写进 lifecycle `archive_tree_sha256`，使首次 accepted
  lifecycle-v3 full-retention 永远无法通过 `mark_retained`。修复不改写任何既有 storage
  registry identity；binding receipt 现在同时保存 `storage_tree_sha256`，并从签名 content
  inventory 独立重算、逐文件重放后写入 lifecycle `archive_tree_sha256`。回归测试要求两种
  identity 各自精确、明确不同，且 inventory SHA-256 也进入 binding receipt。
  end-to-end close-path 复核同时修正 lifecycle controller 的下游比较：archive
  layout 与 close-time content replay 只绑定 lifecycle tree；storage registry 与
  retention replay 只绑定 receipt 中单独的 `storage_tree_sha256`。缺少任一身份或
  交叉使用都会 fail fast，不再依赖两种编码偶然相等。
- Attempt21 关闭后的 Formal 入口预检发现，v2 resolver 虽能重放迁移 generation，
  benchmark consumer 仍从目录 basename 和已删除的 D: batch 路径推导 Pilot 身份。
  修复把签名 review/campaign 内的 canonical run label 作为身份来源，并直接从已验证
  generation 的 `wsl_active` 读取 control/review、从 `d_benchmark` 读取 Pilot batch；
  campaign 创建后新增的治理 aliases 允许作为 locator superset。真实 Pilot97 重放验证
  原 raw/review SHA、execution lock、36 个 storage observations 和 339,561,305 observed
  bytes 全部保持不变，不创建 D: compatibility symlink，也不恢复已删除源数据。
- Formal Rerun03 首次启动在 output/permit/lifecycle record 创建前又暴露同类遗留断言：
  performance staging 预检仍要求 locator aliases 精确等于
  `d_archive + wsl_staging`，而已验证的当前 locator 还必须包含 lifecycle 使用的
  `d_host + e_archive`。修复改为必要 aliases 的 subset 验证，额外治理 aliases
  仍由各自的下游 root/volume gate 独立验证；回归测试直接使用四 alias 生产形状。
  该 pre-admission failure 没有 output、permit 或 lifecycle record，`rerun03` 标签未被消费。
- 新 sealed producer 补入 Attempt21 已归档且 sidecar 验证的 v3 recalibration report 后，
  Formal 继续在同一 pre-admission 边界暴露 producer effective-lock 的旧逻辑：它仍要求
  Pilot predecessor 的 262,144/2 topology 与 replacement contract 的 16,384/1 相同，
  与 v3 loader/reviewer 已验证的“predecessor 和 replacement 各自精确绑定”协议冲突。
  effective-lock 现在仅对 `cgroup_v2` v3 recalibration 接受该已签名 topology replacement；
  历史 process-tree v2 仍要求 topology 不变，workers、memory floors、20% headroom、
  replacement contract SHA 和 zero geometry 所有门禁保持。回归测试使用 Attempt21 的真实拓扑差异。
- Formal 随后进入 configuration selection gate，只发现 Pilot97 的 `d_archive`
  与当前 storage-governance-v2 的 `e_archive` 两处物理存储别名差异，其余 TOML 字节一致。
  consumer 现在必须用 `is_retained_path_from_locator` 重新验证 prerequisite 确实位于
  当前 E retention root，才可仅将 archive alias 规范化回 predecessor 值复核旧
  configuration selection hash，并将 E alias 加入 planned-root migration coverage。未验证的
  alias drift、staging drift 和任何 scientific/resource 字段变化仍 fail fast；新测试要求
  无 retained-migration proof 时相同 D→E 差异必须被拒绝。
- successor-revision 重放继续显示 Pilot97 之后的 Stage 0--8 lifecycle/storage-governance
  接线路径尚未进入 G-governance allowlist。当前提交只追加精确的 lifecycle catalog/
  policy/registry、storage governance deep module、各历史 runner 的共享 preflight 入口、
  CI/tests 和签名迁移/删除工具；`objective.py`、ALNS 搜索、neighborhoods 与其他
  scientific core 仍不在 allowlist。回归测试允许 `storage_governance.py` successor，
  并继续明确拒绝 `objective.py` drift。
- `c31ee32` sealed producer 完整通过离线 pre-admission replay 后，Formal 仍在
  output/permit/lifecycle 创建前被 lifecycle clean-tree gate 拒绝：v3 recalibration
  report 及 sidecar 已由 source-snapshot allowlist 和签名 loader 明确要求，却未列入
  `.gitignore`，因此同一 producer 一方面必须携带二者，另一方面又必然被
  `git status --untracked-files=all` 判 dirty。修复只把这两个精确 local evidence path
  加入 ignore；source snapshot 仍遍历并哈希其实际字节，formal loader 仍验证 sidecar
  与 replacement contract binding，其他未登记文件继续 fail fast。回归测试用
  `git check-ignore` 固定这两个路径的 lifecycle cleanliness 语义。本次失败没有
  output、permit 或 lifecycle record，`rerun03` 标签仍未消费。
- `f1d577b` producer 随后完整通过 920-shard/2040-axis pre-admission 与 lifecycle plan
  重放，但 E 盘 free bytes 比签名 Formal archive estimate 少 `13,240,344,460` bytes。
  为避免手写删除清单或篡改 v2 registry，lifecycle controller 新增 historical
  compaction transaction：只从 complete aggregate gate 的唯一 record、canonical E
  generation、签名 physical inventory 和 review 生成 plan；PREPARED 不导入 run record、
  不删除文件。apply 重新验证 gate/plan/inventory/review、legacy pretty-canonical tree、
  完整文件集、mtime/size/SHA-256 与 writer absence，随后才导入 CLASSIFIED、复用现有
  APPLYING/COMMITTED 单遍删除引擎并写 CLOSED receipt。仅
  `superseded_metadata`/`superseded_accepted_capsule` 可用；回归测试覆盖 active-writer
  拒绝、精确 keep/delete、幂等重入和最终 lifecycle audit。

## 2026-07-31：Stage 0--8 storage governance v2 与 E 盘归档入口

- 原因：Stage 5.2 active/history/benchmark raw 的现有投影已超过 600 GiB，旧的
  “所有失败永久完整保留 + D 盘 50 GiB reserve”合同不能继续作为未来 Stage 的容量
  边界。E 盘已作为新的单份长期 archive（归档）介质接入，但 SHA-256 内容一致性不等于
  backup（备份）；删除已验证源之后不存在介质故障回滚。
- 新增 `evrptw.storage_governance` deep module（深模块）：
  `preflight_run` 在 run directory/worker 创建前验证 experiment plan、三卷 identity、
  动态容量和唯一 run label，并通过带锁 ledger 预留容量；失败 observation 带 SHA-256
  sidecar 持久化，permit 未经审计不自动释放。
- 动态 stop gate 固定为 E `planned archive + 200 GiB`、D
  `projected WSL growth + 200 GiB`、WSL ext4
  `active workspace + 50 GiB`；Stage 5.2 active workspace 至少 32 GiB。benchmark
  producer 已在 output directory 创建前取得 permit，并把 plan/observation identity
  写入 control metadata。Stage 0--5.1 及 Stage 5.2 非 benchmark producer CLI 也在
  调用各自 runner 前经过同一 preflight；后续 Stage 6--8 producer 必须复用该入口。
  benchmark 每次 batch dispatch/archive 前重新使用同一 permit identity 复测三卷容量。
- retention v2 将 evidence 分为 `accepted_full`、`unique_failure_full`、
  `duplicate_failure_reduced`、`rebuildable` 与 `unknown_full`。duplicate reduction
  必须加载并复验签名 adjudication、匹配已完整保存的 canonical root-cause
  representative，并生成 immutable projection manifest；reduced generation 明确为
  audit-only。v2 resolver 优先选择最新 verified generation，并回退读取既有 Stage 5.2
  v1 registry。accepted/unique full generation 还必须登记签名 independent replay
  receipt，绑定 archive tree、verifier identity、validator/objective/raw-review replay；
  resolver 每次重新验证该回执。
- archive role 从硬编码 `d_archive` 改为 `e_archive`。共享 volume probe 允许绑定
  model/serial/BusType 的 NTFS USB physical disk，同时拒绝 FAT/ExFAT、虚拟/未知设备、
  序列号或盘符漂移。`d_archive` 保留为 legacy resolver 和 WSL VHDX host-capacity role。
- rebuildable maintenance 默认 dry run，只允许精确 allowlist、无 active lock、超过
  retention period、keeper 零引用且 tree identity 一致的 cache/spool；venv/build
  还必须具有签名 isolated rebuild + hash/smoke-test proof。缺少任何证据均 fail closed。
  实际 apply 必须绑定内容完全一致的签名 dry-run receipt；开始、失败和完成状态分别保留
  签名回执，避免清理清单在审计后漂移。
- 本条只发布代码、配置和只读迁移入口；尚未复制或删除 Stage 5.2 raw，也未 compact
  VHDX。任何物理迁移后的源删除仍需逐目录报告 source/target/bytes/hash/可释放空间和
  单份介质风险，并等待用户字面确认 `确认`。

## 2026-07-31：Stage 5.2 E 盘迁移、源删除与 VHDX compact 完成

- `stage052-retention-v2-20260731` 最终发现并迁移 307 个 run、356 个 segment，
  总 source payload 为 712,267,368,027 bytes。全部 generation 在
  `E:\Reproducible-EVRPTW-archive` 原子发布并登记；WSL→E 的 133 个 mapping 与
  D→E 的 223 个 mapping 均完成独立 source/destination tree SHA-256 replay。
  307/307 个 run 通过 v2 resolver ledger replay，migration receipt 状态为
  `verified_not_deleted` 后才生成删除候选清单。
- 为消除 WSL/DrvFS I/O amplification（输入输出放大），归档复制使用
  `robocopy /MT:32 /J` 或 32 路 stream copy；E 目标和 D 源的最终 attestation 改为
  Windows-native 单进程/单遍 32 路 SHA-256。正式双原生复验实测 D 峰值约
  1.87 GiB/s、E 峰值约 963 MiB/s，校验语义、逐文件账本和 canonical tree identity
  均未降低。
- 签名删除候选 manifest 精确列出 356 个互不重复、无嵌套的源目录，SHA-256 为
  `0b75becf3478b2183728da082ffeddb016a56ff7f829e8394420127fb7b248c8`。
  用户在看到精确路径、663.35 GiB 总量、全部校验结果和“E 为单份介质、删除后无介质
  回滚”风险后再次回复 `确认`。删除执行器先重新读取并验证全部 712,267,368,027 bytes，
  再删除 356/356 个路径；重启 Ubuntu 后独立检查仍为 0 个存在。执行收据 SHA-256 为
  `6192df5608288b9a0692afd512965469e2cc9aed3e5fea43d33d67ad15af3b36`。
- 精确 rebuildable allowlist 的归档后 maintenance dry run 为 0 candidates / 0 bytes；
  producer logs、storage-governance ledger、manifest、registry 与 review 未进入清理。
  ext4 `fstrim` 报告 746,023,903,232 bytes 可回收；Ubuntu 停止后
  `Optimize-VHD -Mode Full` 把 `D:\WSL\Ubuntu\ext4.vhdx` 从
  456,645,410,816 bytes 压缩至 336,704,045,056 bytes。重启后 Ubuntu 正常，
  最终空闲空间为 D 1252.32 GiB、E 267.40 GiB、WSL ext4 约 644.24 GiB，全部超过
  storage stop gate。

## 2026-07-31：全仓库 experiment lifecycle v3 落地

- 新增 `evrptw.experiment_lifecycle` deep module（深模块）与统一
  `experiment-lifecycle` CLI，强制顶层实验依次完成 PLANNED、PERMITTED、RUNNING、
  SEALED、REVIEWED、CLASSIFIED、RETAINED/COMPACTED、CLOSED；未关闭或
  `BLOCKED_RETENTION` 记录会在新 run directory 创建前阻塞下一实验。
- `configs/experiment_catalog.toml` 登记现有 13 个 Stage 0--5.2 producer；CI 静态审计
  runner 入口、共享 lifecycle preflight 和 v1/v2 migration ledger。新写治理事实只使用
  v3；既有 v2 E registry 与 tracked v1 registry 保持只读兼容。
- retention v3 的分类由 reviewer status、受控 failure code、完整 failure identity 和
  签名 root-cause adjudication 决定。未知根因固定为 `unknown_full` 并阻塞；原位精简只
  接受 controller 签名 keep/delete 计划和精确 SHA-256，并记录释放字节、最终 tree
  SHA-256、哈希/扫描/删除遍历次数和 native I/O backend。
- 容量合同更新为 E archive reserve 0、D host reserve 0、WSL safety 50 GiB、Stage 5.2
  active floor 32 GiB。`e_archive` 是唯一新归档角色；`d_archive` 只读解析历史，`d_host`
  只承担 VHDX host-capacity measurement（宿主容量测量）。
- 后续双轴审查加固了事实边界：reviewer status 与 classification 只从签名 manifest 推导；
  unknown failure 必须经签名 adjudication 才能解锁；full-retention receipt 必须与增量
  content inventory 的 tree/file/byte identity 完全一致。新 current 的 close transaction
  自动胶囊化旧 current，并保留可重入 supersession checkpoint。
- permit reconciliation 改为 ledger-first、receipt-second 的可恢复提交；CLOSED receipt
  先于最终 record 写入且必须逐字段匹配。运行时会再次绑定 source/config/environment 和
  worker/thread/process 参数；Stage 3.4、Stage 4 与 Stage 5.2 的真实并发度已显式进入
  immutable plan。CI 使用 AST call inspection，不再以字符串出现作为接入证明。
- compaction 不再在 close 阶段重扫或重哈希 E 盘：只消费 producer 增量生成的签名
  inventory，apply 单次遍历当前 run 并拒绝任何新增文件、symlink、缺失或 stat 漂移；
  keep/delete ledger 仅追加，重复执行与 APPLYING 中断均可安全恢复。
- 双轴审查后的 v3 加固将 lifecycle transition 改为 append-first event + per-run CAS，
  producer 持有共享 writer lease，seal/compaction 取得互斥租约；compaction apply 会重新
  校验每个文件的内容 SHA-256，因此同大小、同 mtime 篡改也会失败。旧 current 的原始
  `CLOSED` record 不再重开，effective retention class 只写入追加式 supersession receipt。
- retention 与 close 不再接受任意路径上的自报 JSON：它们分别绑定 E 盘 canonical
  v3 retention receipt、实时 v2 registry/replay identity，以及 WSL capacity ledger 中的
  原始 permit/reconciliation。独立 review 必须另有签名 `review_execution.json`，绑定
  reviewer module、raw manifest 前后哈希和 review manifest。
- migration ledger 中的 13 个历史 pending run 新增独立硬门。只读 historical reviewer
  可以单遍生成逐文件 inventory 并复核 v2 tree/file/byte identity，但在没有 stage-specific
  semantic disposition（阶段专用语义裁定）时固定输出 `INVALID`/`unknown_full`；因此当前
  不会伪造 `historical-migration/gate.json`，Calibration Attempt09 仍被程序化阻塞。
- 第三轮 fail-closed 加固把所有 catalog producer 的正常完成路径接到共享
  `seal_cli_attempt`；CI 同时检查 preflight 与 seal 调用。Stage 5.2 calibration 通过已签名
  child manifest 聚合 terminal inventory，不重新读取大型 raw 内容；benchmark 的最终
  campaign manifest 升级为 v3，直接绑定 primary manifest、persistence attribution 和
  全部增量 artifact identity，可作为 canonical sealed manifest。
- lifecycle 对内部 state 继续强制 `.json.sha256` 双文件格式；producer/reviewer 外部证据
  单独兼容项目既有 `.sha256` sidecar，并要求同时存在的 sidecar 都与同一 payload 一致。
  compaction plan 改为按 SHA-256 寻址，supersession status 通过追加 receipt 投影 effective
  retention class；full-retention close 会对真实 archive generation 做一次精确内容复验。
- 第六轮可靠性加固把 terminal build、failure capsule、seal 与 close 放入同一 run 的排他
  writer lease。producer 在首次 `mkdir` 前失败时，只能从已签名 lifecycle plan 创建精确
  output directory；无 RUNNING record 的 pre-admission 错误不会被二次 seal 异常覆盖。
  reviewer 完成后对 sealed raw 做一次内容校验，并在 execution receipt/state binding 写入后
  再核对 metadata snapshot；current record 首次写入中断可从 append-first event 恢复。
- historical semantic reviewer 已列入中央 catalog allowlist。benchmark 历史候选必须从
  migration-ledger-bound v2 registry 解析 canonical protected keeper generation，重新验证
  账本中 SHA-bound campaign/control/review/batch manifest closure 的签名零引用证明；调用
  方 keeper 路径和手写 proof 均被拒绝。完整 semantic command、inventory、execution、
  binding 与 aggregate gate 必须一致绑定 physical review 的 canonical `e_archive` root
  和 generation identity，临时 mirror 不能替代。physical review 又必须通过仓库 local
  storage-root locator 验证 live volume identity，并使用 signed migration ledger 唯一声明
  的 v2 registry path/SHA；registry SHA 与 generation tree SHA 贯穿后续全部签名证据。
  aggregate gate 逐项复核执行命令、module SHA-256、migration/inventory 前后 identity、
  签名 semantic output 和 E state-root binding。后续根因审计确认
  `stage05.2_hot_path_attempt04` 的 raw manifest 在全部 review generation 中不变，历史
  READY review 被当前 manifest 明确列入 accepted lineage，后续终态 NOT_READY 只命中
  `source_snapshot` 与 `prerequisite_performance_baseline` 两个 live/current-chain gate；
  `stage05.2_hot_path_attempt06` 又以独立 finalized review 接替并通过同一 hot-path 状态。
  reviewer 因此新增精确 supersession replay：它从 ledger-bound v2 registry 复验 attempt04
  的完整 lineage/retry/files/execution 绑定，并重新通读 attempt06 全树、raw/review/execution
  identity。只有这些证据全部成立才输出 `PARTIAL`/`superseded_accepted_capsule`；否则仍为
  `INVALID`/`unknown_full`。真实 historical gate 与后续 Calibration 仍须在该实现提交后
  另行受控执行，不因代码路径存在而自动解锁。
- historical gate consumer 现在再次验证 live locator `e_archive`、migration-ledger v2
  registry SHA，并交叉重放 gate/review/inventory/semantic execution/binding 的 root、
  generation、registry、tree 与 module/command/hash identity；它还重放 semantic
  status/retention class、failure identity/representative、no-dependency/rebuild proof 等
  class-specific gates，手写 gate 或把 `unknown_full` 重标为安全类别均被拒绝。
- 首次真实 `hot_path_attempt04` physical review 暴露了 reviewer 与 v2 governance 的
  canonical JSON（规范 JSON）不一致：前者使用 compact serialization，后者及 Windows
  migration verifier 使用 indent=2、sort_keys 与末尾换行，因此 file/byte 完全相同仍会
  得到不同 tree SHA-256。这是 reviewer 的确定性算法缺陷，不是归档内容漂移。修复后
  inventory 复用 v2 序列化合同，并以最多 32 workers 的单遍 thread pool 计算逐文件摘要、
  前后 stat snapshot 和 tree identity；不匹配异常同时报告 expected/observed
  file/byte/tree 三元组。回归测试直接将 historical inventory 与
  `storage_governance.compute_tree_identity` 对账，防止再次分叉。

## 2026-07-31：Calibration lifecycle 与 cgroup scoped-peak 修复

- `stage05.2_resource_calibration_attempt09` 在普通 WSL 会话中启动，正式内存门禁因
  不属于独立 `systemd --user` service cgroup 而 fail fast。独立 failure-capsule
  reviewer 重算其 218 项 manifest inventory，并把精确错误裁定为
  `FAILED_KNOWN`；lifecycle 以 `unique_failure_capsule` 保留全部 224 个终态文件、
  删除 0 文件并完成 `CLOSED`。
- 隔离 service 启动后的 `stage05.2_resource_calibration_attempt11` 完成 16384/1、
  6-worker Formal 测量，但 contract derivation 因读取 service-lifetime
  `memory.peak` 而错误地把先前 worker/Parquet 阶段高水位算入 scoped R205 峰值，
  最终触发 20% headroom 门禁。独立 `stage05.2_formal_memory_probe_attempt18` 已在
  相同物理参数下记录 R205 cgroup peak `18,879,209,472` bytes，证明该失败不是
  放宽 headroom 的理由。
- `CgroupV2MemorySource.reset_peaks()` 现在在 Formal sampler 启动前把独立 service
  的 `memory.peak` 与 `memory.swap.peak` 重置为当前 scoped baseline，并验证 swap
  counter 一致性。Calibration 同时写入签名
  `formal_memory_cgroup_peak_reset.json`；后续尝试继续固定 6 workers、16384/1、
  swap 0、20% headroom 和无 fallback，不复用或覆盖上述失败标签。
- Formal sampler 一旦返回就先签名写入 `formal_memory_measurement.json`，再执行
  contract/headroom 门禁；因此即使门禁随后 fail fast，也会保留本次 cgroup 峰值、
  per-worker 峰值与 swap 证据。成功的 calibration report 还必须绑定该文件 SHA-256。
- `stage05.2_resource_calibration_attempt13` 在 pre-admission 阶段、创建 output/permit/
  lifecycle record 之前被 historical gate replay 拒绝。根因是 consumer 把历史 semantic
  reviewer 的 Python 绝对路径错误地要求等于当前 producer `sys.executable`，与独立冻结
  producer/reviewer runtime 的协议冲突。修复后仍验证签名 command 的现存绝对 Python、
  `-m`、module、完整 options 及全部下游 hash，只移除跨运行时相等这一错误约束。
- Attempt14 随后在同一 pre-admission 边界暴露第二个 relocation 缺陷：历史命令绑定
  active checkout 的 migration-ledger 绝对路径，而封存 producer 以相同字节的 sealed
  checkout 路径重放。consumer 现在要求历史路径继续存在，且历史与当前 ledger 都匹配
  同一个签名 SHA-256；仅该 ledger 路径允许 relocation，inventory/output/archive/module
  与其余 options 仍保持 path-exact。Attempt14 同样没有 output、permit 或 lifecycle record。
- `stage05.2_resource_calibration_attempt16` 在 durable Windows Scheduled Task 持有的独立
  systemd service 中完成 6-worker、16384/1 R205 Formal 测量；其 scoped cgroup peak 为
  `22,823,022,592` bytes，swap 0、fallback 0，但加 20% headroom 后超过
  `25,196,941,312` bytes 可用容量。独立 reviewer 复核 300 个 raw artifacts（共
  `710,413,605` bytes）并报告 `FAILED_KNOWN`；该新根因以
  `stage052-calibration-workers6-formal-memory-headroom-v1` 全量保留、删除 0 文件并 CLOSED。
- 六个 Formal child 的 lifetime peak RSS 为约 3.16--4.35 GiB；runner 原先只执行 Python
  GC，没有让 PyArrow 的 mimalloc pool 在相邻 30/60/300 秒 axis 之间归还空闲页。现在每个
  axis 终态在 artifact flush 后显式执行 `MemoryPool.release_unused()`，并记录 pool backend、
  release 前后 Arrow live bytes 和 process RSS。release 异常或 live allocation 增长会
  fail fast；6 workers、16384/1、swap 0、无 fallback 与 20% headroom 门禁均未改变。
- `stage05.2_resource_calibration_attempt17` 在相同 6-worker、16384/1、swap-free cgroup
  协议下复测；显式 Arrow release 把 aggregate peak 从 Attempt16 的
  `22,823,022,592` 降至 `22,311,612,416` bytes，但仍未满足 20% headroom。独立
  failure reviewer 复核 300 个 raw artifacts（`717,316,302` bytes）并报告
  `FAILED_KNOWN`；它与 Attempt16 的 failure identity 相同，按
  `duplicate_failure_metadata` 签名分类、compaction 与 close 后为 `CLOSED`。本次运行
  完成 `288,281` 次 exact starts，而旧通过探针 Attempt18 为 `263,816` 次，说明当前
  wall-clock workload 增加约 9.27%，旧 `18.88 GiB` 峰值不能直接外推。
- 后续 producer 每个 axis 都把 GC、Arrow pool release、libc `malloc_trim` 可用性/结果及
  release 前后 RSS 写入不可变 trace；Calibration parent 在 Formal cgroup peak reset
  之前执行相同释放并签名写入 `formal_memory_parent_release.json`。成功 reviewer 必须
  重放该文件、calibration report 与 terminal inventory 的 SHA binding，并核验六个
  R205 seed 的全部 18 个 axis memory-release record；未持久化的内存日志不能成为资源
  契约依据。
- calibration 的成功路径新增独立 terminal-manifest replay：reviewer 逐文件复核 byte、mtime
  和 SHA-256，重放签名 v3 report、formal memory measurement、cgroup peak reset 与外部
  producer resource contract，并只在固定 6/16384/1 拓扑及全部 binding 一致时输出
  `ACCEPTED`。producer 自报 complete 不再足以进入 lifecycle classification。
- `stage05.2_resource_calibration_attempt18` 在上述 parent/axis release 协议下仍记录
  cgroup peak `23,090,085,888` bytes（容量 `25,196,937,216` bytes）、swap 0、fallback 0，
  因 20% headroom 失败。独立 reviewer 复核 302 个 artifacts（`740,559,325` bytes）并
  报告 `FAILED_KNOWN`；failure identity 继续归并到 Attempt16，按
  `duplicate_failure_metadata` compaction 后 CLOSED。18 个 axis 的 Arrow live allocation
  仅约 14.6--15.7 MiB，而 300 秒 axis 结束回收后的 worker RSS 仍约 2.89--3.98 GiB；
  证据因此把剩余问题定位到 CPython mimalloc 长轴内的 retained/fragmented pages，而非
  PyArrow live buffers，也不构成放宽资源门禁的依据。
- 后续 Formal calibration/benchmark 把 `PYTHONMALLOC=malloc` 纳入签名 producer resource
  contract。async writer 每完成 8 个 Parquet batches，就在持有 writer turn 时丢弃刚完成
  batch 引用并执行 libc `malloc_trim`；trace 固定记录 release ordinal、allocator、trim
  结果与前后 RSS。成功 reviewer 从 `submitted_batches` 独立重算每个 axis 的 release
  次数和 ordinal 序列，并要求 parent 与 18 个 axis 的 allocator 均为 `malloc`。该变更
  只控制 300 秒 axis 内的可回收 Python page，6 workers、16384/1、swap 0、fallback 0 与
  20% headroom 均保持不变；必须用新 label 重测，不能重标 Attempt18。
- `stage05.2_resource_calibration_attempt19` 在 pre-admission 阶段因 Scheduled Task 启动的
  systemd user service 缺少 Windows PowerShell/WSL 路径而拒绝 canonical E: volume
  identity；它没有 output、permit 或 lifecycle record，但其 systemd journal 保留失败，
  label 不复用。后续 service 明确绑定最小 Linux、WSL、PowerShell PATH，并在分配新 label
  前用无实验写入的 systemd diagnostic 重放 historical gate 与 runtime identity。
- `stage05.2_resource_calibration_attempt20` 首次在固定 6/16384/1、swap 0、20% headroom
  协议下完成全部计算并通过资源门：cgroup aggregate peak `18,141,900,800` bytes，
  per-worker formal peak `3,345,002,496` bytes，contract aggregate limit
  `21,770,280,960` bytes，capacity `25,196,937,216` bytes，fallback 0；allocator 为
  `malloc`。但 success terminal sealing 随后拒绝全部 direct v2 shard artifacts 为
  unlisted，因此该 attempt 仍是不可接受的 sealing failure，不能启动 Formal。
- 根因是通用 CLI terminal builder 只发现 `**/control/*_manifest.json`，而 Calibration
  的 `artifact-storage-v2` manifests 直接位于 `instance/seed/`；早先 headroom failures
  只走自动收纳残留的 failure manifest 路径，未触发 success closure。builder 现在独立
  发现 direct shard manifests，验证签名、schema、run/instance/seed、shard/worker 身份、
  completeness 与 exact path prefix，并从 worker/formal root 重算每个 artifact 的 byte/
  SHA-256。新回归测试复现真实 Formal directory shape；Attempt20 不重封或重标，修复必须
  以新提交和新 calibration label 重跑。
- `stage05.2_benchmark_rerun03` 通过全部 admission/capacity gates 后，由可见 Windows
  Scheduled Task console 被人工关闭而收到 `0xC000013A`；systemd 同时记录外部 stop，
  producer 无算法、容量、内存或 swap failure。该 label 按 partial failure inventory
  封存，独立 reviewer 复核 2,639 个 artifacts（281,046,376 bytes）且 raw manifest
  前后 SHA-256 不变，再以签名 adjudication 归类为唯一 external-control interruption；
  同 label、partial shard 均禁止复用。
- 本次中断同时暴露 rolling-capacity preflight 会覆写最初 lifecycle-bound permit 的根因：
  `PERMITTED` record 保存首次 permit SHA-256，而 batch pre-dispatch 又以同一 run label
  写入新 observation 和 permit，导致最终 reconciliation/close 必然漂移。修复后首次
  permit 文件不可变；滚动检查只追加 observation 并单调缩小 ledger reservation。
  回归测试验证连续 preflight 后 permit bytes/SHA-256 不变、最新 E reserve 生效、ledger
  收缩且 reconciliation 仍绑定首次 permit。后续 Formal 必须用新 commit、新 sealed
  runtime 和新 label，并通过隐藏且与可见 console 解耦的启动控制面从零运行。

## 2026-08-01：Formal Rerun05 candidate-cache commit 与 worker error 根因修复

- `stage05.2_benchmark_rerun05` 在隐藏 Scheduled Task 与独立 user service 中从零启动；
  batch0001--batch0004 完成并复验归档。batch0005 的 51 个预分配 shard manifests 中仅
  36 个达到 `evidence_completeness=complete`，其余 15 个为 partial；按文件存在数量监控
  会错误报告进度，因此 complete/partial 必须读取 manifest 语义字段。该 label 已
  `SEALED`，不得续跑、覆盖或向后续 Formal 导入任何 shard。
- systemd journal 保留的首个真实异常是 shared `RouteEvaluationCache` 在 candidate commit
  收到已存在 key：`atomic candidate cache batch contains a non-miss key`。随后 bounded
  `BoundedNegativeSequenceCache` 尚未建立 batch，异常路径却断言它必须是 `dict`，以
  `AssertionError` 掩盖首因；`Stage03ExecutionError` 又携带 async persistence trace，
  ProcessPool 序列化时最终把两层错误掩盖为 `TypeError: cannot pickle
  '_queue.SimpleQueue' object`。全部 51 个 task 输入经 `pickle` 与 `ForkingPickler`
  单独验证可序列化，排除了 task payload 与 Windows console 作为首因。
- route-cache atomic commit 现在对迟到的已存在 key 重算除 `runtime_seconds` 外的完整
  deterministic result payload。exact equality 时不重写 LRU 或 statistics，而写入包含
  pending/existing SHA-256 的 `equivalent_existing` reconciliation event；任一字段不同则
  报告两份摘要并 fail fast。current compact cache event 扩展为 20 字段，Python writer
  与 native sparse packer 继续读取旧 18 字段；campaign 与 performance reviewers 对
  已存在 key 和摘要相等性进行独立审计。
- bounded negative-cache rollback 只在 batch 已建立时回滚；若失败发生在此前且没有
  dictionary insertion，则不执行错误类型断言。worker 在先写入 partial-shard evidence
  后，把任意原异常转换为只含字符串的 `Stage052ShardExecutionError`，保留原始类型与
  消息并安全通过 `spawn` ProcessPool；trace、executor 与 `SimpleQueue` 不再跨进程边界。
- 回归测试覆盖 equivalent late commit、semantic conflict、bounded rollback、20/18 字段
  native round-trip、reviewer 拒绝错误摘要，以及真实 `spawn` pool 中携带
  `SimpleQueue` 的异常。完整验证为 1,065 passed，Ruff、strict mypy（77 source files）
  与 `git diff --check` 通过。Rerun05 仍须按独立 failure review、adjudication、签名
  retention plan、compaction 与 lifecycle close 完成闭环；修复后只能以新 clean revision
  和 `stage05.2_benchmark_rerun06` 从零运行。

## 2026-08-02：Formal Rerun15 cgroup page-cache 根因修复

- `stage05.2_benchmark_rerun15` 在 clean revision
  `07c24e6a8e94bef6bcd9260d4bc4354d02161948` 上完成并归档 batch0001--batch0006；
  batch0007 以正确的 fail-fast 路径报告 cgroup v2 memory hard limit exceeded：首次观测
  `21,793,177,600` bytes，contract limit `21,792,659,866` bytes，最终 cgroup peak
  `21,812,838,400` bytes、process-tree RSS peak `18,590,945,280` bytes、swap 0。
  独立 failure reviewer 复核 504 个 artifacts（`3,337,645,926` bytes）且 raw manifest
  前后哈希不变；签名 adjudication 将其归类为唯一
  `stage052-formal-runtime-memory-guard-underestimate-v1`，随后以
  `unique_failure_capsule` 完成 compaction、permit reconciliation 与 lifecycle
  `CLOSED`。该 label 与 partial shards 均不得复用。
- cgroup 与进程树峰值相差约 3.22 GB，而未归档 batch0007 内容约 3.34 GB。代码审计确认
  大型 neighborhood JSONL spool 和普通 JSON evidence 在 ext4 写入后保留 clean/dirty
  file-backed pages；`posix_file_cache_drop_is_safe` 又把 WSL2 原生 ext4 与 DrvFS/9p
  一并禁用，短 calibration scope 未覆盖长批次累计的 cgroup page cache。该证据不支持
  抬高门禁或减少 Formal geometry。
- 修复后 mountinfo 采用最长挂载点匹配：WSL2 native ext4 允许
  `POSIX_FADV_DONTNEED`，`/mnt/e` 等 9p 路径仍禁止。大型 neighborhood spool 每
  64 MiB 执行 durable flush、`fsync` 和 page-cache release，最终 drain/close 再释放；
  canonical merge 读取 spool 时也每 64 MiB 释放已读 clean pages，避免读路径重新建立
  同等峰值；JSON artifacts 与签名 JSON 在 durable write 后同样释放 clean pages。所有
  I/O 时间仍计入 persistence，不吞掉错误，也不改变事件、objective、validator 或
  artifact bytes。
- Formal recalibration failure loader 现在同时接受历史 v3 process-tree RSS 与 v4 dedicated
  cgroup evidence；v4 必须验证 exact cgroup path、swap 0、sealed resource summary，并把
  `aggregate_peak_memory_bytes` 绑定为 predecessor high-water mark。Rerun15 batch0007 的
  实际读取重放得到 aggregate `21,812,838,400`、per-worker `3,662,557,184`、拓扑
  6/16384/1、resource SHA-256
  `468b368ab395f211dc7fb26e31b15c9b2ca990ab75cacab33a4f896e275898d8`。
  修复必须先以新 clean revision 完成 zero-geometry resource calibration，再以全新 Formal
  label 从零运行；Rerun15 仅作为失败根因证据，不能提供 readiness geometry。

## 2026-08-02：full-retention 单段 archive supersession 路径修复

- `stage05.2_resource_calibration_attempt22` 的独立 reviewer 完整重放 302 个 artifacts、
  18 个 axis memory releases 与 98 个 batch memory releases 并报告 `ACCEPTED`；full
  retention（完整保留）归档与 lifecycle `CLOSED` 已完成。关闭事务随后在自动取代前任
  Attempt21 时 fail fast：既有 retention receipt 合法绑定
  `generation-0001/wsl_active`，但 supersession controller 仅接受路径末端本身为
  `generation-NNNN`，导致协议层错误地拒绝 retention 层允许的 canonical single-segment
  archive（规范单段归档）。Attempt21、Attempt22 的原始回执与证据均未改写。
- supersession archive validator 与 compaction planner 现在共同接受两种精确形状：
  `generation-NNNN` 根目录，或其下名称匹配 `[a-z][a-z0-9_]*` 的唯一逻辑 segment leaf。
  run label、generation、signed inventory、archive tree、writer lease、计划 SHA 与
  append-only receipts 仍逐项校验；任意更深嵌套、非规范 leaf、内容漂移或活动 writer
  继续 fail fast。cross-volume（跨卷）归档会把 plan 的 `mtime_ns` 重新绑定到已经通过
  byte/SHA-256 复验的目标文件系统现场值；为恢复修复前已签名的 PREPARED plan，只允许
  source nanoseconds 与 9p/NTFS 目标时间戳位于同一 UTC 秒，byte count 与 SHA-256 仍须
  精确相等，因此不会把实际内容漂移当成时间精度差异。
- 回归测试新增真实 `generation-0001/wsl_active` predecessor，要求新 current accepted
  close 在同一事务中生成 supersession receipt、删除仅由计划声明的大型 raw 文件、保留
  predecessor 原 `CLOSED` record，并使 lifecycle audit 通过。该 controller 修复改变
  source revision，因此后续 Formal 必须在新 clean commit 上重新执行 resource
  calibration，并继续使用全新、未创建的 Formal label。

## 2026-08-02：同身份按批次恢复协议

- 新增 `CampaignRecoveryController` deep module（深模块），统一首次执行和恢复打开路径。
  首次运行写入不可变 `campaign_identity.json` 和 epoch0001；后续只能使用 lifecycle 在
  `RUNNING`、writer 已退出且无终态清单时签发的一次性 `ResumePermit`。恢复原因限于
  `unexpected_host_loss` 和有 signed pre-stop intent 的 `operator_stop`，不新增
  `PAUSED` 状态。
- 已归档 batch 在恢复时重新验证 manifest/sidecar、tree hash、bytes 与 persistence
  envelope，原 SHA-256 保持不变；verified/archive transaction 中断幂等完成。未完成
  batch 形成签名 inventory、control-only interruption capsule 和 deletion receipt 后整批
  重算，旧 shard 不得进入最终 geometry。run label、Git tree/revision、wheel/runtime、
  config、prerequisite、resource contract、plan 或 start permit 漂移均 fail closed。
- runner 从所有已验证 archived batches 重建 per-run rows、signed per-batch checkpoints、
  rolling-capacity journal 与 persistence attribution，不依赖前一进程内存。campaign
  reviewer 新增 recovery gate，独立重放 identity/epoch/permit/consumption/capsule 链和
  batch execution epoch，并继续从 raw evidence 重算 geometry、validator/objective 与
  checkpoint 汇总。
- Calibration Attempt23 的跨 revision 继承改为签名 successor attestation：精确绑定
  `58c325a`、新 commit、全部 changed paths/blob hashes、Attempt23 accepted report/review
  和 resource contract。该证明仅允许 `resource_contract_only` 继承，Formal shard 不能
  跨 revision 复用；scientific、solver、objective、native-kernel 或 shard-schema 路径
  变化直接拒绝。
- 本变更只实现和验证协议；未启动 Formal Rerun16，未启用 Scheduled Task，未删除历史
  evidence，也不声明 `READY_FOR_STAGE05_3`。

## 2026-08-04：原生执行 v2 的 owned request 与 initial-state 边界

- 三种实验架构继续使用 `stage052-native-execution-v2`，但 host binary kernel
  protocol（主机二进制内核协议）因新增完整搜索请求和初始状态操作升级到版本 3，magic
  同步升级；旧版本仅可读取 attempt03 的冻结证据，不再生产新结果。协议输入固定为 21
  个 typed SoA arrays（类型化结构分离数组），包含完整 problem、warm start、Candidate
  Control、Stage 4、operator、budget 和 deadline 配置。
- 新增纯 C++ `RequestV2`、`InitialStateV2` 和 `initialize_state()`。本地 full-native
  入口拥有请求副本后释放 GIL 执行初始 exact charging；host 入口把同一完整请求经 UDS
  和 POSIX shared memory 提交给 scheduler。初始路线、objective、exact-call 计费和
  deadline 与 Python warm-start 路径对齐，不再允许替代拆分算法。
- 初始状态输出包含 path、status/reason、metrics、label counters、batch counters、正式
  objective、accounting、request hash 与 independently recomputable state hash（可独立复算
  的状态哈希）。客户端在 ACK 前校验完整 typed schema、路径结构、客户投影、计数、目标、
  request/state identity 和 deadline；partial IPC、offset corruption、budget failure 或超时
  均 fail fast 且不提交共享内存状态。
- full-native 总事务新增 initial-state ownership receipt（初始状态所有权回执），把
  local/host ownership、operation count、request hash、state hash 和 receipt hash 纳入总
  transaction digest（事务摘要）。性能 telemetry 只用于与已哈希回执交叉核对，不能单独
  证明 host 真正拥有该操作。
- 本条只闭合纯原生搜索迁移的输入和初始状态边界；三种架构 capability bits（能力位）仍为
  0。必须继续迁移完整三 lane 搜索状态、Candidate Control、cache、Stage 4 与 terminal
  projection，并通过 12/12 fixed-work 语义门控后，才允许创建 attempt04 标签或运行性能
  实验。本条未启动 Formal、CUDA，未切换默认架构，未清理或复用 attempt01--03。
- 验证使用同一新 wheel 分两个不重叠的 mode waves：非 host 回归
  `278 passed, 12 skipped`，host scheduler/UDS/shared-memory 回归 `43 passed`；公开投影、
  ownership 伪造与 reviewer tamper（审查器篡改）聚焦回归 `8 passed`，architecture gate
  tests `37 passed`，Ruff、strict mypy（83 个 source files）和 `git diff --check` 通过。
  聚焦回归所用 wheel SHA-256 为
  `670ad226eef691330b6c91d5877c980763e6af2a5fd14160fc56e1eb4f8ee53f`，native extension
  为 `0ebd279e8cfc485f82ef0d5e334e59a8c4872273e1a0838a98fd70c21949e7bc`，scheduler 为
  `0aa1e822f98f55a1e8440199e6fa234b8c38ac4b55527b54e6673a10755c9b31`。独立 follow-up
  review 要求公开 `ALNSResult` 持久化 ownership receipt；修复后 raw architecture
  reviewer 独立重算 receipt hash 并核对 mode、operation count 与 telemetry，避免底层
  回执在实验投影中丢失；对所有计数字段显式拒绝 Python `bool` 与 JSON boolean，新增
  门禁对应的 review schema 升级为 v7。

## 2026-08-04：纯 C++ 初始四 lane ownership 与可复算 receipt v3

- `InitialStateV2` 之后新增纯 C++ `LaneStateV2` 与
  `InitialFourLaneStateV2`。它们分别拥有 constraint、legacy、quality-shadow 和
  global-best 的初始路线、exact 结果、正式 objective、accounting、两条 Python
  `random.Random` 兼容 RNG 的初始 seed、iteration 0 以及 request/initial-state
  identity。此对象只是四 lane 的初始快照，不是 live search state（实时搜索状态）；
  当前搜索迭代仍由既有 Python-backed mirrors（Python 支撑镜像）承载，后续必须继续
  迁移 candidate transaction、lane apply、Stage 4 和 terminal projection。
- owned-request 路径从同一个 `InitialStateV2` 构造四份独立 lane ownership，并在进入
  搜索前严格核对现有四组镜像的 C-contiguous dtype、维度、精确 shape 和逐值相等。
  测试专用一次性 fault injection 覆盖错误维度和错误 dtype；注入使用新建 tuple，避免
  在共享 Python tuple 上原位替换造成 CPython `SystemError`。任何镜像 drift（漂移）均
  fail fast，且 capability bits 仍全部为 0。
- initial-state ownership receipt 升级为 v3：除 host ownership、operation count、
  request/state hash 外，新增初始四-lane state hash 和 17-column typed projection。
  Python decoder 与 raw reviewer v8 分别独立重算 lane/state/receipt SHA-256；持久化投影
  使用 JSON-safe 规范字段。reviewer 对 boolean count、越界 int64、非有限或溢出 float、
  route/path/shape/objective/accounting/RNG/iteration 不一致均把该轴判 invalid，不允许异常
  数值使整个 review CLI 崩溃。两端还从投影独立重算 initial-state hash，并把 seed、
  node-kind、batch-size 与调用输入/独立解析的 benchmark 绑定，要求所有 customer 恰好
  覆盖一次；自洽重签的虚假 initial-state identity 同样被拒绝。
- 本切片提交前验证使用 wheel SHA-256
  `12a1241728effdc29ab5bfe01748b8bf6aa10f1aad5720283c033d14696ae80f`；已安装 native
  extension 为
  `77c7575d03054e965e91827805a8f0bd64bf39a2e05cd0f169432f607c3c4c1f`，host scheduler
  仍为
  `0aa1e822f98f55a1e8440199e6fa234b8c38ac4b55527b54e6673a10755c9b31`。互斥 mode waves
  的 non-host full-native 回归为 `124 passed, 211 deselected`，host UDS/shared-memory
  回归为 `43 passed, 292 deselected`；架构和一调用语义回归 `51 passed`，receipt、
  ownership 与故障注入聚焦回归 `21 passed`。Ruff、strict mypy 和 `git diff --check`
  通过，独立代码审查未发现新的 C++ 正确性、UB、哈希歧义或 copy/move 生命周期问题。
- 本条不创建 attempt04、不启动 Paired/Pilot/Formal/CUDA、不切换默认架构，也不改写或
  复用 attempt01--03。下一步必须把 candidate-round 唯一 ownership 和原子提交迁入
  C++，再依次迁移三 lane、Stage 4、terminal envelope 与整次调用 GIL release；只有三种
  新架构各自通过 12/12 fixed-work 全字段语义门控，才允许开始五模式重测。

## 2026-08-04：candidate-round staged state 的纯 C++ ownership

- `NativeSearchEngineV2` 新增纯 C++ `CandidateRoundState` 与
  `CandidateExactBatchState`，在一个候选事务完成时拥有原始 plan/route SoA、规范
  feasible order、objective matrix、exact route completion rows、逐路线 exact payload
  和 transaction SHA-256。deferred composite commit（延迟复合提交）不再保存
  `py::tuple pending_candidate_exact_payload_`；pending state 在 commit/rollback 后显式
  reset，避免失效 Python tuple 继续持有事务内存。
- `prepare_first_feasible_candidate()`、sequential repair（顺序修复）和 constraint probe
  （约束探针）改为只消费上述 C++ staged state。返回给 Python 的 transaction tuple
  仍用于 ABI/证据投影，但不再能改变候选选择、路线、exact payload 或 objective。
  排名和可行计划规范排序也直接调用纯 C++ `native_candidate_plan::rank()` 与
  `order_feasible()`，不再经内部 Python tuple wrapper 重演。
- 新增 strict mirror validation（严格镜像校验）：输入 SoA、objective、exact completion
  order、feasible order 和 transaction identity 必须与 C++ staged state 一致。比较采用
  typed-buffer byte equality（类型化缓冲区逐字节相等），因此能正确处理 canonical NaN
  sentinel，同时仍区分 `-0.0` 和不同 NaN payload。一次性 mirror-tamper fault injection
  （镜像篡改故障注入）证明不一致会 fail fast，并回滚 route cache、negative cache、
  attempted plans、候选轮预算 reservation（预留）、causal journal、solution 和三条 lane；
  started exact accounting（已启动精确调用计费）按既有失败语义保留，同一操作随后可正常
  重试。
- 独立复审后继续收紧此边界：`CandidateRoundState` 现在拥有事务哈希覆盖的全部 typed
  字段，并可从自身规范状态独立重算 SHA-256；返回镜像校验不再只比较同源字符串。故障
  注入移动到 deferred state 已发布、实际返回 tuple 已构造之后，校验失败通过正式 pending
  rollback 路径撤销事务。发布顺序也改为先完成可能抛错的 causal journal，再原子公开
  pending members，消除 active flag 设置前留下半发布状态的异常窗口。搜索、温度估计、
  quality/refinement 与事件投影不再从返回 tuple 读取状态、可行顺序或 objective，而从
  C++-owned round state 读取；tuple 只保留 ABI 和证据用途。返回前逐字段校验 tuple 的
  selected/status/objective/resolution/exact/completion/counters/cache/negative/budget/feasible
  投影及事务哈希，故障注入专门篡改非 objective counter（计数器）以覆盖证据字段漂移；
  未求解 exact sentinel（精确结果哨兵）的 reason、path、metrics 与 label counters 也必须
  保持规范零值。
- 本切片的原始 1-thread/4-thread transaction、deadline、duplicate plan、cache、budget、
  commit failure、objective ordering 与 canonical journal 聚焦回归为 `19 passed`；扩大到
  所有 `native_search_engine`/`full_native` 路径的 non-host 回归为
  `149 passed, 187 deselected`。本条仍未迁移 route/negative cache 与 budget 的 Python
  receipt adapters，也未迁移 live lane apply、Stage 4 controller 或 terminal projection；
  capability bits 继续全部为 0，不创建 attempt04，不启动 Paired/Pilot/Formal/CUDA。

## 2026-08-04：live lane apply 的 C++ 权威状态

- `NativeSearchEngineV2` 新增四份有界的 C++ `LaneStateV2` live state，分别拥有 legacy、
  quality-shadow、constraint/current 和 global-best 的 routes、exact payload 与正式
  objective。lane swap 同时交换 C++ ownership；candidate acceptance、global-best 更新、
  Stage 4 restart 和 global-search rollback 先更新或恢复 C++ state，再重建现有 Python
  mirrors。`solution_state()`、`lane_solution_state()` 与 `best_solution_payload()` 改由 C++
  state 生成独立数组，Python mirrors 只保留尚未迁完的内部 ABI 兼容用途。
- 所有通用 candidate-round 与 constraint-probe 特殊路径均保存同构 C++ candidate lane；
  legacy deferred candidate、refinement replacement、暂存/恢复和丢弃路径同步转移或清空
  ownership。acceptance、vehicle-first comparison 与 incumbent identity 改读 C++ state，
  不再从 candidate/current objective mirrors 做决策。
- owned-request 的每份 live lane 在读取边界独立验证 route offsets、customer exact-once
  coverage、exact path/customer order、depot/station 结构、feasibility、metrics、labels 与
  objective 重算。新增 live-lane mirror tamper fault：在 apply 前篡改 Python current
  objective，必须 fail fast、从 C++ state 恢复全部 lane mirrors、保持候选未消费并允许同一
  apply 正常重试。该故障与上一条 candidate-round 故障的聚焦集合当前为 `20 passed`。
- 本条是 live-lane ownership checkpoint，不代表 whole-search 已脱离 Python objects；
  operators、cache adapters、terminal projection 和外层 whole-call GIL release 仍需继续
  迁移并通过完整 differential/fault gates。production capability bits 保持 0，不创建
  attempt04，不启动 Paired/Pilot/Formal/CUDA。
- 独立规格/标准审查随后发现并阻断了四个异常路径缺口。global-search snapshot 现完整保存
  四条 lane 的 Python mirrors、constraint/full-operator Stage 4 weights/rewards/calls/totals、
  reheat/restart/intensification state；故障注入移动到 Stage 4 boundary 和 restart 已发生之后，
  rollback 后逐 lane、best、两组 Stage 4 状态保持一致并允许同 iteration 重试。
- legacy candidate apply 不再在比较前破坏性移动所有权：原 legacy owner 保留到验证、发布和
  Stage 4 计数全部成功，mirror fault 后仍可从原入口重试。所有 last-to-legacy ownership
  transfer 在 `std::move(optional)` 后显式 reset source，维持 `ready == has_value`；candidate
  在 objective comparison 前用初始化时固化的纯 C++ `ProblemV2` 执行完整 `validate_live()`，
  accepted current/best publication 另有强异常安全快照。针对 restart-envelope rollback、
  current candidate mirror retry 与 legacy owner retry 的聚焦回归为 `3 passed`，扩展相关集合
  为 `13 passed`。`ProblemV2` 的 O(n²) 静态数组只在 initialize gate 完整验证一次，后续仍
  逐次执行完整 lane/customer/path/objective 验证，避免把不变距离矩阵验证变成搜索主成本；
  真实四实例三 seed fixed-work 门控复跑为 `12 passed`、耗时 `90.98s`。Ruff、83-file
  strict mypy 与 `git diff --check` 通过。
- clean checkpoint `e938e42cfca48123ab32d2f56d192438d891b8bf` 的 exact wheel
  SHA-256 为 `2b04216697974e5c8f58f27cad8a1663d9544ad6d9d6e7e21bb0ce7483e8d0d8`，
  extension 为 `af8ba4ead1c79fa4a932573577450f12a589468a7c853cf7689031b05ff798c1`，
  scheduler 保持
  `0aa1e822f98f55a1e8440199e6fa234b8c38ac4b55527b54e6673a10755c9b31`。
  exact-wheel 扩大回归（含新增 legacy/current retry）为
  `151 passed, 187 deselected`；两份独立复审逐项关闭全部六个原 blocker，接受该提交为
  live-lane ownership checkpoint。内存侧 non-blocker 是 `live_problem_` 当前额外持有一份
  O(n²) immutable problem copy，后续资源测量必须单列 RSS/PSS 并评估共享 ownership。

## 2026-08-04：搜索决策停止读取持久 lane mirrors

- Candidate Control 的 route-count eligibility、ranking incumbent identity，Stage 4 auto-
  temperature 初始路线/目标、legacy/quality/constraint operators 的 incumbent routes/exact、
  refinement candidate、global-search previous state，以及 terminal event 的 current/best
  objective 和 route count 全部改读 C++ `LaneStateV2`。持久 `current_/legacy_/quality_/best_`
  Python arrays/tuples 现在只用于初始化 ABI、fault injection、独立 mirror validation、异常
  rollback 和兼容投影，不再参与搜索选择、接受、Stage 4 或终止判断。
- 仍要求 py-array ABI 的 route-merge/changed-route/constraint-removal helper，由 C++ lane 临时
  生成输入 projection；其决策源不再是持久 mirror。下一步必须把这些 helper 抽成 span/vector
  typed return，并将 exact/negative cache、repair、AcceptanceOutcome、Stage4 boundary 和
  terminal stream 一并改为 owned structs，才能在外层一次性释放 GIL。
- exact-wheel 相关 operator/round/Stage4 回归为 `103 passed`；四实例三 seed fixed-work
  全字段配对门控为 `12 passed`、耗时 `92.14s`。本切片仍保持 capability bits 全 0，且不创建
  attempt04、不启动 Paired/Pilot/Formal/CUDA。
- clean checkpoint `125072223f679def5ef225097adf98e8b89000b8` 的 exact wheel
  SHA-256 为 `55ae4f32fb2c75fbeed57d6302cc09c39e77298ae95143649be74900aca57327`，
  extension 为 `7c9a26e8c5b89891db95bc356918a583039a26d6ff28b81ab582f758d7cf3b28`，
  scheduler 保持
  `0aa1e822f98f55a1e8440199e6fa234b8c38ac4b55527b54e6673a10755c9b31`。
  exact-wheel 扩大回归为 `151 passed, 187 deselected`、耗时 `434.66s`；独立规格复审与
  标准复审均接受该提交，确认所有搜索决策已停止读取持久 lane mirror，未发现新的
  生命周期、异常安全或语义 blocker。复审同时保留非阻断债务：旧 helper 的临时 py-array
  projection、逐次 O(n) lane validation 和缺少防止未来重新读取 mirror 的静态守卫，均需在
  后续 typed helper/GIL ownership 切片继续处理。

## 2026-08-04：candidate-plan cache/exact 的 typed C++ 事务

- `NativeSearchEngineV2.evaluate_plans()` 的 route/negative cache lookup、negative store、
  exact dispatch、exact result journal 与 exact-cache store 改为连续 `RouteBatchViewV2`
  和 owned C++ payload；Python ABI 只在公开入口适配一次数组。cache store wrapper 在发布
  active batch 前完成 Python object、tuple、buffer 与 raw-pointer 获取，发布后只做平凡复制；
  exact payload 扩展失败和 cache conflict 均在 rollback guard 内，禁止活动批事务逃逸。
- local pool 与 host scheduler 共用 `validate_exact_batch_output()`：在任何 exact-call completion
  accounting、path indexing、journal、cache store、telemetry 或 ACK 前，验证 typed descriptor、
  shape、CSR offset、status/reason、metrics、label/batch counters、node kind、depot/station path 和
  fixed customer order。新增 `exact_path_offset_oob` production fault，证明自哈希但损坏的 host
  输出在 ACK 前被拒绝、server-owned shared memory 被回收，且同一 scheduler 可继续下一次求解；
  仍无 serial/Python fallback。
- 原 `cache_ownership_receipt` 改为诚实的 lifetime
  `cache_execution_coverage_receipt`。它只记录 typed negative lookup、exact-cache lookup、exact
  dispatch 与 exact-cache store 是否曾执行；第 5 位 `typed terminal-state projection complete`
  明确保留为 0。该回执不是 transaction-bound ownership evidence，production capability bits
  继续全部为 0。
- clean code checkpoint `e25590dfea1cb4c5e6f5aa23eec85c122faf048f` 的 wheel SHA-256 为
  `3f17343bf8855159cc4c755104f471cab62855bf7fd6a84c6a818dca9b9c0938`，extension 为
  `5b70be1fd29265d16c492c2c5937cf9c192e03fc353fe7187be3c2b426684b75`，scheduler 为
  `a1cd6e81caa49cdd7d162e44ee1274b7645dd26fd0f6b8760ae004b8d5e80617`。安装后的 extension
  自报 revision 与 checkpoint 完全一致。
- clean-wheel 正式验证：cache/exact/deadline/commit/host-corruption 聚焦集合
  `11 passed, 329 deselected`；四实例三 seed fixed-work 全字段差分 `12 passed`、耗时
  `91.61s`；完整 `tests/test_native_execution.py` 为 `328 passed, 12 skipped`、耗时
  `783.91s`。Ruff、83-file strict mypy 与 `git diff --check` 通过。独立 Spec 与 Standards
  两轴复审均 ACCEPT，确认 ACK 前验证、active-batch 异常安全和回执误报三个 blocker 全部关闭。
  本条仍不是 full-native/host-scheduler 完工证明，不创建 attempt04，不启动
  Paired/Pilot/Formal/CUDA，不切换默认架构。

## 2026-08-04：原生 acceptance outcome 成为唯一搜索控制源

- 新增命名的 C++ `AcceptanceOutcomeV2`，统一表示 accepted、improved-global-best 和
  vehicle-reduction。legacy、quality-shadow、constraint、Stage 4 stagnation/global-best 与
  fixed-work exhaustion 的内部决策全部读取 typed outcome，不再解析 Python tuple 槽位；
  公开 `apply_last_candidate()` / `apply_legacy_candidate()` tuple ABI 和规范事件保持不变。
- `AcceptanceOutcomeProjectionV2` 在任何 owned acceptance commit 前预分配 tuple 和 0/1 Python
  对象；commit 后 `finish()` 仅做已有引用赋值，避免 Python allocation failure 发生在 live-lane
  publication、candidate owner 清除或 Stage 4 accounting 之后。quality/constraint/legacy 的
  typed optional outcome 在每轮入口 reset，仅在实际 apply 成功后发布，保留原 retry/rollback、
  RNG consumption 和 operator statistics 顺序。
- clean code checkpoint `41c2aa29c5e3fb24561b87584e74dbe173b54c86` 的 wheel SHA-256 为
  `c1363ccb0de5bcf3b7e5df1c6794e501d3043fbfaee1afb60ec7e6de312ec646`，extension 为
  `4ad569fc759e7e338c4596273a069e8a0171e6ec93e4daed7298970ea4cbad9e`，scheduler 保持
  `a1cd6e81caa49cdd7d162e44ee1274b7645dd26fd0f6b8760ae004b8d5e80617`；安装后的 extension
  自报 revision 与 checkpoint 完全一致。
- clean-wheel 验证：acceptance/three-lane/constraint/Stage 4 聚焦集合
  `15 passed, 325 deselected`；四实例三 seed fixed-work 全字段差分 `12 passed`、耗时
  `91.27s`；完整 `tests/test_native_execution.py` 为 `328 passed, 12 skipped`、耗时
  `790.29s`。Ruff、83-file strict mypy 与 `git diff --check` 通过；独立 Spec 与 Standards
  两轴复审均 ACCEPT。production capability bits 继续全部为 0，不创建 attempt04，不启动
  Paired/Pilot/Formal/CUDA，不切换默认架构。

## 2026-08-04：three-lane termination state 完成 typed ownership

- 新增 C++ `ThreeLaneTerminationStateV2`，统一持有 reason、exact budget、started、
  completed、interrupted 和 completed iterations。bootstrap、follow-up、outer deadline、
  budget boundary 与 fixed-work exhaustion 的搜索控制全部读取 typed state；reason 3
  exhaustion 只修改 typed reason，再重算单轮与总 search-stream canonical hash。Python
  termination array 仅在最终 ABI/证据边界生成，不再反馈 three-lane 搜索决策。
- `ThreeLaneTerminationProjectionV2` 在搜索开始前分配 NumPy array 并缓存 raw pointer；
  `finish()` 只执行六个 `int64_t` 写入与 move。lifetime coverage receipt 的第 5 位仅在
  projection 返回后置 1，关闭了“typed state 已发布但 buffer request 失败”的异常窗口；
  未执行 three-lane terminal projection 的普通 candidate transaction 仍诚实保持第 5 位为 0。
- clean code checkpoint `fbe2daf611a0d934fb09afaf67f6e4db0eb7af38` 的 wheel
  SHA-256 为 `201830b775f8261c6c3ea7a86bfd88e8e42f39c551a18e3778441676ff0802c4`，
  extension 为 `06d6e033f8956ac147f6a7d41b7a1baa6c02c004c9e455a710c9e03664c1041f`，
  scheduler 保持 `a1cd6e81caa49cdd7d162e44ee1274b7645dd26fd0f6b8760ae004b8d5e80617`；
  安装后的 extension 自报 revision 与 checkpoint 完全一致。
- clean-wheel 验证：聚焦 terminal/cache/deadline/causal 集合 `6 passed`；扩大
  three-lane/full-native 集合 `121 passed, 219 deselected`、耗时 `429.49s`；四实例三 seed
  fixed-work 全字段差分 `12 passed`、耗时 `93.09s`；完整
  `tests/test_native_execution.py` 为 `328 passed, 12 skipped`、耗时 `791.08s`。
  Ruff、83-file strict mypy 与 `git diff --check` 通过。独立 Spec 审查 ACCEPT；Standards
  首轮发现 projection 后置 buffer-request blocker，修复后复审 ACCEPT。production capability
  bits 继续全部为 0，不创建 attempt04，不启动 Paired/Pilot/Formal/CUDA，不切换默认架构。

## 2026-08-04：Stage 4 boundary 完成 typed staged commit

- 新增 C++ `Stage04BoundaryStateV2`，在 owned boundary 中统一计算 segment status、四个
  constraint operator 的 old/new weights、boundary calls/rewards、reheat、restart、
  intensification 和 stagnation 状态。`Stage04BoundaryProjectionV2` 在任何状态修改前分配完整
  tuple/arrays 并缓存 raw pointers；commit 后只复制固定大小基础类型并 move payload。
- reheat/restart/intensification 改为 staged next-state。restart 需要的 best-lane routes、exact
  payload 和 objective mirrors 全部在提交前构造；发布阶段只移动 C++/pybind owned objects 并
  更新标量。新增 projection failure injection 证明失败后 weights、segment accumulators 和
  last-finished iteration 均不变，同一 iteration 可按原输入成功重试。
- clean code checkpoint `dfb38a82240792f9fad06f4a4642bf043a21a287` 的 wheel
  SHA-256 为 `f9b548b8d1b0d287a3b7d568fa214a995fade4608fd4395f54a5a9ac6cc3579f`，
  extension 为 `512c3ef52def97e4636a70a028f2f961d68c5b7e8c49c275a9722857a4b69839`，
  scheduler 保持 `a1cd6e81caa49cdd7d162e44ee1274b7645dd26fd0f6b8760ae004b8d5e80617`；
  安装后的 extension 自报 revision 与 checkpoint 完全一致。
- clean-wheel 验证：Stage 4/three-lane 扩大集合 `71 passed, 270 deselected`；四实例三 seed
  fixed-work 全字段差分 `12 passed`、耗时 `93.62s`；完整
  `tests/test_native_execution.py` 为 `329 passed, 12 skipped`、耗时 `798.47s`。
  Ruff、83-file strict mypy 与 `git diff --check` 通过；独立 Spec 与 Standards 双审查均
  ACCEPT。production capability bits 继续全部为 0，不创建 attempt04，不启动
  Paired/Pilot/Formal/CUDA，不切换默认架构。

## 2026-08-04：constraint iteration outcome 完成 typed ownership

- 新增 C++ `ConstraintIterationOutcomeV2`，统一持有 operation、probe seed、candidate
  feasible、accepted、improved-global-best、vehicle-reduction 与仅供内部校验的 iteration
  identity。`ConstraintIterationProjectionV2` 保持公开 `(selection, probe, int64[6])` ABI，
  但 projection tuple/array/raw pointer 与 selection/probe slots 均在搜索状态提交前准备；
  commit 后只写六个 `int64_t` 并 move payload。
- zero-removal 路径仍消费一次真实 RNG seed 以保持 Python `random.Random` 调用序列，但公开
  probe seed 继续按历史语义投影为 0。正常路径保存真实 seed。constraint-search、global-search
  和 three-lane 事件、stagnation 与 feasibility 控制全部改读 typed outcome，不再解析 outcome
  NumPy array。getter 同时校验 outcome iteration 与 `last_completed_constraint_iteration_`，外层
  rollback 后的同迭代或跨迭代 stale outcome 均 fail fast。
- clean code checkpoint `857852ae4be952ce28ae9c1dfeb78c4de21d1d7e` 的 wheel
  SHA-256 为 `e260ffda84e2de559f33eb262231cb33cea02d3dccb8124b18498c29b85c4a83`，
  extension 为 `ae7edaf4f31e0fa487f151a72ff2b2be799236557a333e06eaa3fd0c070619f5`，
  scheduler 保持 `a1cd6e81caa49cdd7d162e44ee1274b7645dd26fd0f6b8760ae004b8d5e80617`；
  安装后的 extension 自报 revision 与 checkpoint 完全一致。
- clean-wheel 验证：constraint/global/three-lane 扩大集合
  `84 passed, 257 deselected`；四实例三 seed fixed-work 全字段差分 `12 passed`、耗时
  `92.38s`；完整 `tests/test_native_execution.py` 为 `329 passed, 12 skipped`、耗时
  `792.96s`。Ruff、83-file strict mypy 与 `git diff --check` 通过；独立 Spec 与 Standards
  双审查均 ACCEPT，并按 Standards 建议补上同 iteration rollback 身份门。production
  capability bits 继续全部为 0，不创建 attempt04，不启动 Paired/Pilot/Formal/CUDA，
  不切换默认架构。

## 2026-08-04：dynamic removal typed ownership 与 canonical telemetry 边界

- 新增 C++ `DynamicRemovalSelectionV2`，统一持有 tier、requested/lower/upper count、
  stagnation、trigger、reset flag 与仅供内部验证的 iteration identity。Stage 2.3 的
  large/medium 优先级、periodic exploration 晋级、`1..n-1` clamp、zero-removal 和
  global-best-reset 记录语义保持不变；公开 `dynamic_removal_selection_v2()` 仍返回历史
  `int64[7]` ABI。constraint iteration、constraint search 与 global search 的预算选择和事件
  投影全部改读 typed state，不再从返回 NumPy selection array 反向读取搜索控制值。
- `ConstraintIterationProjectionV2` 现在预分配 selection/outcome arrays；成功提交时同时发布
  带相同 iteration identity 的 selection 与 outcome。新增只读状态接口证明 Python 修改返回
  selection 不会影响原生状态，错误 iteration fail fast。global-search snapshot 同时保存并恢复
  selection、constraint outcome、acceptance outcome 与 completed-iteration identity；envelope
  fault 后失败轮 selection 不可见，并可按原输入重试。公开绑定已同步 `_core.pyi`。
- 首次真实四实例三 seed fixed-work 门控诚实得到 `3 passed, 9 failed`：所有失败轴的 objective、
  routes、candidate-work hash、route-result hash、exact started/completed 和非遥测 operator
  statistics 均已相同，首个分叉只出现在 route-merge 的
  `pair_prefilter_rejected_aggregate` / `prefilter_rejected_aggregate`。这两类事件描述 safe
  prefilter/pair pruning 避免的实现专属工作量，不是 candidate transaction、预算、cache 或
  acceptance 决策；原生与 Python 合法地具有不同 aggregate reason/count/hash。
- canonical candidate trajectory 因此只排除上述两类预筛聚合遥测，并在过滤后连续分配
  semantic ordinal，防止遥测数量改变后续 candidate ID。普通 `prefilter_rejected`、candidate
  pool、Candidate Control、exact result、budget、acceptance 与 Stage 4 事件仍参与严格比较。
  两类 aggregate 继续保留在 raw neighborhood evidence；pair-pruning 还保留专门的 native-
  ablation audit stream。回归证明有无两类遥测时 surviving semantic event 与 candidate ID
  完全相同。修复后的原失败单轴通过；提交前与 exact-wheel 的完整真实门控分别均为
  `12 passed`，正式复验耗时 `421.46s`。
- clean code checkpoint `c7d3e19e1304e743dc5787e39ef02c69be58fea2` 的 wheel SHA-256 为
  `30f7478d9bbed68661f2dd0b825c5f53d537036fe629a72117630edeaa054175`，extension 为
  `6178055cff5eac7888530533fea8142250124509c49a31fab8ea7631b290ae02`，scheduler 保持
  `a1cd6e81caa49cdd7d162e44ee1274b7645dd26fd0f6b8760ae004b8d5e80617`。
- clean-wheel 验证：dynamic-selection/constraint/global/semantic-trajectory 聚焦集合
  `25 passed, 317 deselected`；完整 `tests/test_native_execution.py` 为
  `330 passed, 12 skipped`、耗时 `796.10s`；四实例三 seed 真实 fixed-work 为
  `12 passed`、耗时 `421.46s`。Ruff、83-file strict mypy 与 `git diff --check` 通过；独立
  Spec 与 Standards 双审查均 ACCEPT。production capability bits 继续全部为 0；本条只封存
  per-solve 与 reviewer semantic projection 检查点，不声称 full-native/host-scheduler 已完成，
  不创建 attempt04，不启动 Paired/Pilot/Formal/CUDA，不切换默认架构。

## 2026-08-04：constraint removal 输出完成 typed ownership

- 新增 C++ `ConstraintRemovalResultV2`，自有保存 partial-route CSR、removed customers、
  ranking score 三列、三字段 metadata 与 iteration identity。原公开
  `constraint_removal_v2()` 继续返回历史七数组 ABI；`project_constraint_removal_v2()` 每次
  复制到新的 NumPy arrays，Python 修改公开返回值不能污染搜索引擎保存的 owned state。
  `constraint_removal_state(expected_iteration)` 提供只读验证投影并拒绝 stale iteration。
- constraint probe 的三个成功出口、zero-removal constraint iteration、正常 constraint
  iteration、constraint search 与 global search 均改为发布或读取 typed removal state，而不再
  从返回 tuple 的 metadata、scores、removed/partial arrays 反向读取搜索控制。初始化会清除
  上轮状态；constraint-iteration 异常恢复 previous optional，global-search snapshot 同时保存
  并恢复非空 removal state。global rollback 的 `noexcept` guard 使用 move restore，避免
  optional<vector> 深复制分配失败导致 `std::terminate`。
- 测试过程保留两项非通过记录。第一次把“非空 removal 快照恢复”直接塞进旧 global rollback
  测试后，预先 probe 造成后续 exact/cache 命中，旧断言所要求的 started-call 增量不再出现；
  该组合测试诚实失败，随后恢复旧测试原样，并增加独立 non-empty snapshot regression，二者
  同时通过。另一次使用过宽的 `-k full_native` 选择器误收集 154 个慢用例，184 秒外层
  watchdog 超时后留下 pytest PID 96206；按精确 PID 正常终止并确认退出，该次不计入通过
  证据。之后用精确 constraint/global 集合复验为 `28 passed, 315 deselected`。
- clean code checkpoint `497398905337a3c4c75b22fd37c9f9bf98b75d39` 的 wheel SHA-256
  为 `6dbeeb7c70f15ae38fd1f25014195efb7357a580c6b48603fbf5bc95d817104a`，extension
  为 `378212c1f01ce0f77e08c174017c7da94432a990d2da108fb1c39d95313e72df`，安装态
  scheduler 保持
  `a1cd6e81caa49cdd7d162e44ee1274b7645dd26fd0f6b8760ae004b8d5e80617`。
- clean-wheel 完整 `tests/test_native_execution.py` 为 `331 passed, 12 skipped`、耗时
  `791.84s`；四实例三 seed 真实 per-solve fixed-work 全字段差分为 `12 passed`、耗时
  `421.74s`。Ruff、83-file strict mypy 与 `git diff --check` 通过。独立 Spec、Standards 与
  code review 三路均 ACCEPT；Standards 首轮发现 `noexcept` 深复制 blocker，move restore 与
  非空快照回归完成后复审接受。
- 本条只完成 constraint-removal **输出端** ownership。`constraint_removal_owned_v2()` 仍接收
  `py::handle`/NumPy 输入，repair 仍消费 Python projection，因此尚不是 GIL-safe input core，
  也不证明 full-native/host-scheduler 完工。下一顺序仍是 typed repair、typed helper pools、
  typed evaluate-plans/probe/outer search inputs，最后才能释放 whole-call GIL。production
  capability bits 继续全部为 0；不创建性能 attempt04，不启动 Paired/Pilot/Formal/CUDA，
  不切换默认架构。

## 2026-08-04：candidate-control repair 输出完成 typed ownership

- 新增 C++ `RepairResultV2`，自有保存 repaired-route CSR、七字段 counters 与 iteration
  identity。校验覆盖 offsets 首尾/严格单调、indices 长度、status 0/1/2、非负 counters、
  `calls == passes + rejections`、成功 pending=0/非空路线，以及失败严格 `[0], []`。
  公开 `candidate_control_repair_v2()` 保持历史三数组 ABI、dtype、字段顺序和失败 partial-work
  counters；`constraint_repair_state(expected_iteration)` 返回独立复制并拒绝 stale iteration。
- legacy route elimination、vehicle-count-aware、standard repair/refinement、quality route-segment
  与 constraint probe 五个 C++ consumers 全部改读 typed offsets/indices/counters。Python tuple 只在
  公开返回或仍要求数组的 `evaluate_plans` seam 投影。legacy elimination 的 identity hash 从
  array encoder 改为同 shape 的 vector encoder，ndim→shape→count→values 字节编码与路线顺序保持
  一致；constraint/global semantic stream 的成功 repair 路线也改读 owned state。
- repair 只有在 post-repair deadline、exact transaction 和 envelope gate 正常完成后才发布；
  safe repair 成功但 exact candidate infeasible 仍保留有效 repair state。repair status 1/2、
  no-removal 和 zero-removal 均不发布。constraint-iteration catch 与 global snapshot 保存并 move-
  restore optional repair，initialize 清空状态；非空 iteration 99 快照在 iteration 0 global fault
  后逐数组恢复。
- Standards 首轮复审发现同 iteration stale-state blocker：同一 iteration 先成功、再正常
  repair-failure/no-repair 时，旧成功 state 可能被 `run_constraint_search` 误用于本轮 canonical
  routes。新增 `invalidate_constraint_repair_for_iteration_noexcept()`，只在三个无成功 repair 的
  正常出口失效相同 iteration，不删除其他 iteration 的证据，异常仍由快照恢复。真实回归在同一
  engine/iteration 先成功并 apply，再产生 status 1 `[1,0,1,8,8,0,1]`，旧 state 随后必须
  unavailable；Standards 修复后复审 ACCEPT。
- TDD/验证过程保留其余非通过记录：缺少 state accessor 的初始 red test；首次编译发现三处已删
  `repair_counters` 仅用于返回 envelope 的残留引用；一项测试曾误把 safe-repair 成功后的 exact-
  infeasible 事件当作 repair failure；新 failure test 首次插入位置错误，使原 solution assertions
  落入新函数并触发 `NameError`。这些问题分别通过 ABI-only counter projection、正确的 exact-
  infeasible publish 断言、独立 capacity=2/route-change-limit=1 status-1 fixture 和恢复原测试作用
  域解决，失败结果不计入通过证据。
- clean code checkpoint `3cbf2dba16880b938b9d505daf7f42636165c40c` 的 wheel SHA-256
  为 `ebb4a8dda097d061f2b78a0024303237aeb925bf9da5187fce6d03b5ad74a414`，extension
  为 `c71c77a9f1fd42954c29f138f4082096153767c63260436887a529a3d012ceac`，scheduler
  保持 `a1cd6e81caa49cdd7d162e44ee1274b7645dd26fd0f6b8760ae004b8d5e80617`。
  clean-wheel 聚焦 repair/constraint/global/legacy/three-lane 集合为
  `36 passed, 308 deselected`；完整 `tests/test_native_execution.py` 为
  `332 passed, 12 skipped`、耗时 `792.45s`；四实例三 seed 真实 per-solve fixed-work
  全字段差分为 `12 passed`、耗时 `422.70s`。Ruff、83-file strict mypy 与
  `git diff --check` 通过；独立 Spec、Standards 与 code review 最终均 ACCEPT。
- 本条仍只是 repair **输出端** ownership：`candidate_control_repair_owned_v2()` 继续接收十二个
  `py::handle` 数组并读取 NumPy storage，projection 同样需要 GIL；不能释放 whole-call GIL，
  也不证明 full-native/host-scheduler 完工。下一步继续 typed helper pools 与 typed
  evaluate-plans/probe/outer inputs。production capability bits 继续全部为 0；不创建性能
  attempt04，不启动 Paired/Pilot/Formal/CUDA，不切换默认架构。

## 2026-08-04：insertion plan pool 完成 typed output ownership

- 新增 C++ `InsertionPlanPoolV2`，自有保存 plan→route 与 route→customer 两层 CSR、以及
  `[target_route, insertion_position]` metadata。纯 `insertion_candidate_plans_owned_v2()`
  只接收连续 `std::span` 与基础配置；公开 `insertion_candidate_plans_v2()` 仍返回历史四数组
  ABI，其中 metadata 保持连续 `int64[N,2]`、target 外层/position 内层的确定性枚举顺序，
  `allow_new_route=true` 的 singleton 路线仍位于最后。
- `legacy_standard_probe` 的 sequential multi-customer 与 single-removed 两个内部 consumer 改为
  直接传递 C++ offsets/indices 和初始化时固化的 `ProblemV2::demand`；sequential 排序读取 owned
  metadata，不再从 Python projection 反向读取。只有仍接受 NumPy 的 `evaluate_plans` seam 和公开
  返回边界执行一次独立 projection。capacity compensated sum、epsilon、new-route 条件、方案顺序、
  exact budget 与 staged commit 语义均未改变。
- 输入 CSR 在任何 pointer range 构造前完整验证 route offset 上界。Standards 首轮审查同时指出
  result validator 若在未验证全部 plan offset 前读取 metadata，畸形中间 offset 可能越界；修复为
  route CSR、plan CSR、metadata 三阶段校验，并在 `int64_t` cast 前验证所有 vector sizes，使用
  `metadata.size() % 2` 和 `/ 2` 避免 `plan_count * 2` 溢出。复审确认该 P1 关闭；独立 Spec 与
  Standards 最终均 ACCEPT。
- clean code checkpoint `94afd978d476df699b1b21146fd77bdba5c6494b` 的 wheel SHA-256 为
  `9b083528466beec2a674955c3379e6503ace7c27f048fae15864825f475a5e32`，extension 为
  `ada9521be77fd52d8b06c12a36e7ce62e619cc8a164f9727a0d8c643107e6015`，scheduler 保持
  `a1cd6e81caa49cdd7d162e44ee1274b7645dd26fd0f6b8760ae004b8d5e80617`；安装后的 extension
  自报 revision 与 checkpoint 完全一致。
- clean-wheel 聚焦公开枚举与两类 standard repair 路径为 `6 passed, 338 deselected`；扩大
  非真实数据 full-native 集合为 `86 passed, 258 deselected`；完整
  `tests/test_native_execution.py` 为 `332 passed, 12 skipped`、耗时 `797.17s`；四实例三 seed
  真实 per-solve fixed-work 全字段差分为 `12 passed`、耗时 `424.25s`。Ruff、83-file strict
  mypy 与 `git diff --check` 通过。
- 本条只完成 insertion helper 的 owned output/pure-span computation；公开 adapter、projector 与
  `evaluate_plans` 仍使用 pybind/NumPy，`legacy_standard_probe` 整体没有释放 GIL，不能报告为
  whole-call GIL-safe 或性能收益。production capability bits 继续全部为 0；不创建 attempt04，
  不启动 Paired/Pilot/Formal/CUDA，不切换默认架构。下一 helper slice 为
  `route_merge_candidate_pool_v2`。

## 2026-08-04：route-merge candidate pool 完成 typed ownership

- 新增 C++ `RouteMergeCandidatePoolV2`，自有保存 candidate CSR、五列
  `[pair_left, pair_right, source, target, position]` metadata、pair-pruning 两计数和仅供内部校验的
  input route count。`route_merge_candidate_pool_owned_v2()` 接收连续 spans，公开
  `route_merge_candidate_pool_v2()` 仍返回历史四数组 ABI：`int64[K+1]` offsets、`int64[M]`
  indices、连续 `int64[K,5]` metadata 与 `int64[2]` pruning；全剪枝空池保持
  `[0], [], shape=(0,5)`。
- pair 排序键、两种 source/target 方向、position `0..target.size()`、exact sequence duplicate
  identity、`preserve_duplicates` 和严格 `combined_demand > capacity + epsilon` 的整 pair 剪枝
  顺序完全保持。pruned-candidate 仍按 `len(left)+len(right)+2` 计费；新增的 bounds/overflow
  gate 只让非法或不可表示输入 fail fast。
- `legacy_route_merge_probe` 现在从 legacy lane 和 frozen `ProblemV2` 构造 route metrics、demand、
  capacity，并直接读取 owned candidate offsets/indices/metadata 完成并行 screening、journal、plan
  重建与 source-candidate identity；公开 pool projection 在任何 screening/state mutation 前一次性
  分配，既保留旧返回 envelope，也不让 Python array 反向控制搜索。released-GIL screening 段只读
  生命周期由 state mutex 与局部 `pool_state` 覆盖的 owned buffers。
- 输入 route CSR 在构造 pointer range 前验证 terminal、严格非空、单调和每段上界；输出 validator
  分层验证 CSR、metadata shape、pair/source-target identity、route bounds、position 和非负 pruning。
  Standards 首轮复审发现一个继承自旧公开 helper 的 P2：NaN/Inf/负 demand 会进入 pair sorting
  key，破坏 `stable_sort` 的 strict weak ordering。现已在每个 demand 进入补偿求和前要求 finite 且
  non-negative，并用 NaN、+Inf、-Inf、-1.0 四类公开 ABI 回归关闭；复审最终 ACCEPT。
- 测试补强覆盖输出 dtype/C-contiguity、duplicate on/off、pair-pruning on/off、全剪枝空池、剪枝
  计数、畸形非单调 offset 和四类非法 demand。验证过程有一次命令组合误把 C++ 源文件直接传给
  Python Ruff，Ruff 按 Python 语法解析后失败；该次仅是验证命令误用，不计入通过证据。随后标准
  `ruff check .`、83-file strict mypy 与 `git diff --check` 均通过。
- clean code checkpoint `7a0d797dbb6603d29b15ed2ed3492c429ff6b817` 的 wheel SHA-256 为
  `640bf4b4ad8601d013ad25059e7b3ec8dda6a5d4de75ca4530fb56991f96205b`，extension 为
  `3e2638583dd8199b530d0e0fa105a709770135f2197817b3a734a1febc8dae86`，scheduler 保持
  `a1cd6e81caa49cdd7d162e44ee1274b7645dd26fd0f6b8760ae004b8d5e80617`；安装后的 extension
  自报 revision 与 checkpoint 完全一致。
- clean-wheel route-merge/follow-up 聚焦集合为 `59 passed, 285 deselected`；扩大非真实数据
  full-native 集合为 `86 passed, 258 deselected`；完整 `tests/test_native_execution.py` 为
  `332 passed, 12 skipped`、耗时 `802.42s`；四实例三 seed 真实 per-solve fixed-work 全字段差分
  为 `12 passed`、耗时 `420.39s`。独立 Spec 与 Standards 最终均 ACCEPT。
- 本条仍只完成 route-merge pool 的 pure-span computation/owned output。公开 adapter/projector、
  probe envelope 和其余 screening input mirrors 仍依赖 pybind/NumPy；不能宣称 whole-call GIL-safe
  或性能收益。production capability bits 继续全部为 0；不创建 attempt04，不启动
  Paired/Pilot/Formal/CUDA，不切换默认架构。下一 slice 为 changed-candidate pool 与 plan assembly。

## 2026-08-04：changed-candidate pool 与 plan assembly 完成 typed ownership

- 新增 C++ `ChangedCandidatePoolV1`，自有保存 ordered changed-route pairs、两条 changed route 的
  CSR、removed-customer CSR、operation identity 与 input route count；新增 `CandidatePlanPoolV1`
  自有保存 plan→route 与 route→customer 两层 CSR 及 routes-per-plan identity。公开
  `changed_candidate_pool_v1()` 继续返回历史五数组 ABI，公开
  `assemble_changed_candidate_plans_v1()` 继续返回历史三数组 ABI；dtype、C-contiguity、shape、
  candidate ordinal 与 route order 均保持不变。
- relocate、swap、two-opt-star 三个 generator 的循环、left/right 规范顺序、target insertion/cut
  顺序和 removed 数量语义保持为 1/2/0。`changed_candidate_plan_selection_v1` 在一次 checked input
  adapter 后直接组合两个 owned cores，再仅为现有 screening/ranking/公开 return seam 投影一次；
  selection eligible/rank/top-k 和 evidence digest 仍使用相同数组字节与顺序。
- `quality_changed_probe` 直接从 quality lane vectors 构造 owned changed pool，quality first-seen
  plan dedup、identity 字节编码、plan packing、source ordinal 与 Stage 4/acceptance 流全部读取 owned
  buffers；公开 pool projection 在状态修改前预分配，异常与返回 ABI 顺序保持。新增 append 和 plan
  assembly 累积长度的 `int64_t` overflow gates，局部分配/追加失败不发布部分状态。
- 输入 current/change CSR 在任何 pointer-range 构造前验证 terminal、上界、严格单调和非空；真正
  zero-candidate pool 仍规范保存 `[0]` changed/removed offsets 与 `[0],[0],[]` plan sentinel。测试覆盖
  三 operation Python 顺序、dtype/contiguity、empty pool、畸形非单调 offsets、全空 current/change、
  仅一条 current route 为空和仅一条 changed route 为空。
- Standards 首轮发现非空 plan 允许零长度 route 时，空 vector `data()+0` 可能产生未定义指针算术；
  current/change inputs 与 `CandidatePlanPoolV1` output 已统一要求实际 route 严格非空。复审又发现
  removed count 在验证 offsets 前做有符号减法可能使损坏 typed state 触发 signed overflow；validator
  现分两阶段先验证非负/单调/terminal，再计算差值并核验 relocate/swap/two-opt-star 为 1/2/0。
  两项 P2 均经独立 Spec 与 Standards 复审关闭，最终无 P0–P3 finding。
- clean code checkpoint `a3d5209e90bc5d29238c0c65ff98e1a222c61373` 的 wheel SHA-256 为
  `15116d0ee42d697e95e64f8f11a009681497dc0b3dcf8bfc152acc1aaceb4fa5`，extension 为
  `beff6271fdd458c1a269885e022677e17e1a182044a11b8a9751879ec4418ca3`，scheduler 保持
  `a1cd6e81caa49cdd7d162e44ee1274b7645dd26fd0f6b8760ae004b8d5e80617`；安装后的 extension
  自报 revision 与 checkpoint 完全一致。
- clean-wheel changed/assemble/selection/quality 聚焦复核为 `9 passed, 336 deselected`；扩大
  非真实数据 full-native 为 `86 passed, 259 deselected`；完整 `tests/test_native_execution.py`
  为 `333 passed, 12 skipped`、耗时 `793.85s`；四实例三 seed 真实 per-solve fixed-work 全字段
  差分为 `12 passed`、耗时 `420.51s`。Ruff、83-file strict mypy 与 `git diff --check` 通过。
- 本条仍只完成 changed/assembled pool computation 与 output ownership。screening、ranking、
  public adapters、quality return envelope 和 outer probe 仍存在 pybind/NumPy seam，不能宣称
  whole-call GIL-safe 或性能收益。production capability bits 继续全部为 0；不创建 attempt04，
  不启动 Paired/Pilot/Formal/CUDA，不切换默认架构。下一 slice 为 typed screen batch。

## 2026-08-05：screen batch 完成 typed input/output boundary

- 新增 C++ `ScreenBatchInputV2` 与 `ScreenBatchResultV2`。筛选核心现只接收连续 spans，并自有保存
  candidate IDs、status、duplicate source、连续 `codes[N,16]`、`metrics[N,15]`、五项 counters 与
  transaction digest；Python/NumPy 分配被移到单一 checked adapter 与 projector。公开
  `screen_route_batch_transaction_v2()` 仍严格返回历史七项 ABI，dtype、C-contiguity、shape、候选顺序、
  duplicate first-source、negative-cache 优先级、counters 与 digest 字节序均保持不变。
- `candidate_round_transaction_impl` 与 `changed_candidate_plan_selection_v1` 直接读取 owned screening
  state，仅在原公开 envelope 需要时投影一次。candidate-round resource receipt 仍在筛选前进入 phase 1；
  输入、筛选、projection 或后续事务失败均不退款、不提交且 fallback 为零。二维 distance/reachable 与
  incremental 现在在 flatten 前核验精确 shape，拒绝元素数相同但维度错误的别名输入。
- canonical route key 改用安全 subspan；零候选、零 route indices 与空 negative-cache route 不再执行
  `nullptr + 0` 指针算术。输出 validator 独立核验 aligned vector sizes、乘法溢出、status、唯一
  candidate identity、duplicate 只能引用更早候选、五项 counter 恒等式及 64 位小写十六进制 digest。
- 筛选 worker 的异常由各线程捕获、触发 atomic cancel、汇合全部 worker 后在调用线程重新抛出。
  Standards 首轮发现 P1：逐个构造 `std::thread` 时，第 N 个线程创建失败会在栈展开销毁先前仍
  joinable 的线程并触发 `std::terminate`。实现已改为作用域内 `std::vector<std::jthread>`；所有
  mutex、exception、cancel、scheduler context 和 typed buffers 均在线程组前构造，创建循环失败先
  cancel，随后 RAII 自动 join。新增 one-shot fault injection 在第二次 launch 前故障，回归证明只抛
  `NativeCandidateRoundFailure`，receipt 保持 phase 1、exact started/completed/interrupted 全零、fallback
  为零，transaction/runtime 均未提交。Standards 复审最终 ACCEPT。
- Linux host-scheduler endpoint、required flag 与 telemetry collector 显式复制到每个筛选子线程的
  `NativeSchedulerThreadContext`，关闭 thread-local context 丢失后误走本地 kernel 的 hidden fallback
  风险。collector 本身使用 mutex 汇总；成功结果仍按 submission/candidate order 合并。
- clean code checkpoint `32251c6fdc8bd0ec598ba4ac33e17c16bfc5709b` 的 wheel SHA-256 为
  `915f1d432b135b62ee470e36b4feba00daba1d34d14f78142d9b35aaf0271016`，extension 为
  `d2dc7083e88df9a7205f492750c303a35d6bca28b70aa85790542db030188a0d`，scheduler 为
  `6bf66e7bfcfc4cb2c6ee8dae8c39fd6c54280771612944886780e34962307e30`；安装后的 extension 自报
  revision 与 checkpoint 完全一致。
- clean-wheel 聚焦空批次、shape alias、公开语义与线程 launch failure 为 `4 passed`；candidate-round/
  changed-plan 聚焦集合为 `13 passed, 333 deselected`；扩大非真实数据 full-native 集合为
  `106 passed, 240 deselected`；完整 `tests/test_native_execution.py` 为 `334 passed, 12 skipped`、耗时
  `818.02s`；四实例三 seed 真实 per-solve fixed-work 全字段差分为 `12 passed`、耗时 `418.34s`。
  Ruff、83-file strict mypy 与 `git diff --check` 通过；独立 Spec 与 Standards 最终均 ACCEPT。
- 第一次真实 12 轴验证在 Codex tool connection 被用户中断后继续运行，但其 deleted stdout descriptor
  未留下 pytest 摘要；仅凭进程消失不计为通过证据。随后在相同 revision/wheel/lease 下完整重跑并取得
  上述 `12 passed` 与退出码 0。本条不创建或复用 campaign label，也不修改 raw evidence。
- 本条完成 screen-batch owned computation、worker failure containment 与 Python boundary projection；
  ranking/prepare/decide/order、exact dispatch、outer candidate-round envelope 和完整 search engine 仍有
  pybind/NumPy seam，不能宣称 whole-call GIL-safe 或性能收益。production capability bits 继续全部为 0；
  不创建 attempt04，不启动 Paired/Pilot/Formal/CUDA，不切换默认架构。下一 slice 为 typed
  ranking/prepare/decide/order。

## 2026-08-05：candidate ranking/prepare/decide/order 完成 typed ownership

- 新增独立的 owned adapter/projector seam：candidate-plan rank、prepare、decide 与 feasible-order
  核心均只接收 typed spans/vectors，并自有保存输出；公开 Python ABI 继续返回历史 dtype、shape、
  C-contiguity 与 tuple 顺序。`changed_candidate_plan_selection_v1` 现以 `CandidatePlanPoolV1`、
  `ScreenBatchResultV2`、vector eligibility/lower bounds/attempted state 和 owned ranking 为唯一搜索
  决策源，NumPy 只在 evidence/return boundary 投影。
- `NativeSearchEngineV2::evaluate_plans` 的 canonical expected customers、screen lower bounds、
  ranked/selected plans、exact-loop plan identity、integer/float objectives 与 feasible order 全程由
  C++ vectors 驱动；`CandidateRoundState` 直接从这些 vectors 构造。Python arrays 仅在 evidence、
  mirror validation 与最终 return 前一次性投影，历史 evidence append 顺序、NaN 初始化字节、
  transaction schema、tamper injection 和 Python tuple ABI 保持不变。
- 公开 adapter 与 full-native engine 共用唯一 canonical objective order helper；只规范化 feasible
  objective rows，浮点分量继续使用 `1e-9` half-away-from-zero 语义。新增亚纳秒 raw objective
  差异但 canonical tie 时由 route lexical order 决胜的回归，避免把实现专属浮点噪声变成搜索分叉。
- rank/prepare/decide/order output validators 独立重算 CPython compensated distance、changed-route
  count、完整排序/selected prefix、customer coverage、first-seen unique-route CSR、vehicle cap 与
  feasible adjacent order。CSR terminal 超过 `INT64_MAX`、实际空 plan/route row、未知 current/candidate
  node、单 feasible plan comparator bypass、空 pool/空 feasible set 与非 feasible NaN 均有显式回归。
- `ScreenBatchPythonProblemV2` 在 adapter 边界持有 problem arrays 强引用，changed selection 直接把
  owned plan CSR 交给 typed screen core；GIL 释放期间 spans 与本地 vectors 生命周期覆盖完整同步调用，
  正常和异常展开均先恢复 GIL。晚期 projection 分配仍处于完整 transaction try/catch 内，cache、budget、
  attempted-plan journal 和 negative-cache 的 rollback/fail-fast/zero-fallback 边界不变。
- clean code checkpoint `bb67eccb0542a2e09f4a22718dda46b70dda00a6` 的 wheel SHA-256 为
  `4ba751c35654c209770e42b447c6f0035d99844ea50a815e5a776c4eebec0de1`，extension 为
  `7a5fbf6856a8d175f147aa96bf76b3a003f1409c559423ef1cf157ddbf491bd0`，scheduler 保持
  `6bf66e7bfcfc4cb2c6ee8dae8c39fd6c54280771612944886780e34962307e30`；安装后的 extension 自报
  revision 与 checkpoint 完全一致。
- WIP wheel 的直接 candidate/full-native transaction 聚焦集合为 `30 passed, 317 deselected`，扩大
  非真实数据 full-native 集合为 `106 passed, 241 deselected`，完整文件为 `335 passed, 12 skipped`、
  耗时 `782.13s`。clean-checkpoint wheel 的完整 `tests/test_native_execution.py` 再次为
  `335 passed, 12 skipped`、耗时 `771.82s`；四实例三 seed 真实 per-solve fixed-work 全字段差分为
  `12 passed`、耗时 `409.47s`。Ruff、83-file strict mypy 与 `git diff --check` 通过。
- 独立 Spec 复审确认上轮唯一 P2 已关闭：changed selection 与 full-native evaluate-plans 的内部消费者
  不再以 Python arrays 为状态源；独立 Standards 复审确认 shared canonical ordering、validators、
  empty spans、GIL/lifetime、rollback、evidence bytes、ABI 与 mirror tamper 均无 P1/P2 finding。两项
  最终均 ACCEPT。
- 本条只完成 candidate plan 的 prepare/screen/decide/rank/order typed control boundary；exact dispatch、
  outer candidate-round envelope 与完整 full-native search engine 仍有 pybind/NumPy seam，不能宣称
  whole-call GIL-safe 或性能收益。production capability bits 继续全部为 0；不创建 attempt04，不启动
  Paired/Pilot/Formal/CUDA，不切换默认架构。下一 slice 为 typed exact dispatch 与 candidate-round
  transaction envelope。

## 2026-08-05：typed exact candidate-round 边界与 per-solve 真实门控完成

- solver code checkpoint 为 `9aca455d9e37e125ce151123c98772e1573e192e`。冻结 wheel 位于
  `build/stage052-wheels/9aca455/reproducible_evrptw-0.1.0-cp313-cp313-linux_x86_64.whl`，SHA-256 为
  `3b9a629792144dc4990a12e2ad0c28ec16de8a2864aa1928d0fc83640cb84693`；wheel 内嵌 build revision
  与 code checkpoint 完全一致。本条后续只提交证据文档和对应 provenance hash，不改变该 solver
  checkpoint，也不生成冒充新 solver identity 的 wheel。
- candidate round 的 typed exact 输出现在包含 canonical feasible order、实际 completion order、exact
  result、cache journal、deadline/budget 状态和三类 transaction hash；initial/lane state hash 显式绑定
  completion order。standalone、four-lane、full-native 与 host projection 共用同一验证边界，Python 只验证
  journal 并原子提交；worker/IPC/deadline/hash/cache commit 失败继续整体 rollback、fail fast、fallback=0。
- 最终冻结 wheel 上的完整 pytest 为 `1507 passed, 12 skipped`，耗时 `990.99s`；JUnit 为
  `/tmp/stage052-9aca455-full-pytest.xml`。全量 Ruff 通过；83-file strict mypy 为 `Success: no issues
  found in 83 source files`；publication metadata 为 `8 passed`；`git diff --check` 通过。
- 真实 fixed-work 门控使用 `c101C5`、`c101_21`、`r101_21`、`rc101_21` × seeds
  2014/2015/2016，以 Python Candidate Control 对比 `per_solve_runtime`，结果为 `12 passed`，耗时
  `425.41s`，JUnit 为 `/tmp/stage052-9aca455-real-12.xml`。每轴核验 validator、objective、routes、
  candidate-work/route-result hash、exact started/completed、规范语义轨迹、operator statistics 与 Stage 4
  statistics/log/history；因此 `per_solve_runtime` 的本轮真实 12/12 语义门控完成。
- 兼容回归明确保留历史 generic exact/screen 空路线语义；candidate-round 在 receipt phase 1 显式拒绝
  空路线，并保持 exact started/completed/interrupted 全零。Stage 3 screening decision/aggregate trace 的
  历史记录条件恢复。local/host empty-route differential、ACK/RAII、worker launch failure、hash mismatch 与
  publication provenance 均纳入最终完整测试。
- publication provenance 共 878 行，其中 67 项 `migrated_modified_publication`、765 项
  `migrated_unchanged`，最终 mismatch count 为 0。独立 Spec 与 Standards 对最终修复 diff 均给出
  ACCEPT，无 P1/P2 finding。
- 本条只完成 `per_solve_runtime` 真实 12/12 门控。`full_native_alns` 与 `host_scheduler` 各自的真实
  12/12 仍未完成，三种新架构的总语义门控仍为关闭；不创建 attempt04，不启动 360-axis Paired、
  180-axis Pilot、Formal 或 CUDA，不切换默认架构。下一 phase 继续补齐 full-native/host 的完整搜索执行面
  和真实差分门控。
