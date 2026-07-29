# Stage 5.2 持续迭代变更日志

Stage 5.2 只维护一套 current implementation（当前实现）。A--G 是该实现内部的顺序
gate（门槛），`attemptNN`/`rerunNN` 是实验运行身份，不是代码版本。此文件按时间追加，
不得为整理历史而改写旧条目。

每条记录至少包含：原因、修改范围、行为变化、证据影响、失效或迁移的运行身份、验证
结果及后续运行要求。大型 raw evidence（原始证据）的物理位置由
`experiments/registries/stage05.2_retention_registry.csv` 记录。

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
