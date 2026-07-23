# Stage 5.2 持续迭代变更日志

Stage 5.2 只维护一套 current implementation（当前实现）。A--G 是该实现内部的顺序
gate（门槛），`attemptNN`/`rerunNN` 是实验运行身份，不是代码版本。此文件按时间追加，
不得为整理历史而改写旧条目。

每条记录至少包含：原因、修改范围、行为变化、证据影响、失效或迁移的运行身份、验证
结果及后续运行要求。大型 raw evidence（原始证据）的物理位置由
`experiments/registries/stage05.2_retention_registry.csv` 记录。

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
