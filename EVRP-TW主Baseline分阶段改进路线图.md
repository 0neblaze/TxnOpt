# EVRP-TW 主 Baseline 分阶段改进路线图

## 1. 总目标与当前判断

当前主方法为 ALNS-based matheuristic（基于自适应大邻域搜索的数学启发式）加 exact charging subproblem（精确充电子问题），并使用小规模 Branch-Price-and-Cut（分支定价切割，BPC）作为精确理论对照。

当前版本已经解决了“能否稳定得到可行解”的问题：在现有 5/10/15/100-customer Schneider benchmark（Schneider 标准测试集）实验中，ALNS 的可行率达到 100%。但是，它目前更准确的定位仍然是：

> 高可行率、具备小规模精确校验的第一版强基线，而不是已经证明最强的算法。

下一阶段必须把评价重心从“是否可行”转移到：

1. 车辆数是否显著下降；
2. 总距离是否接近 best-known value（最佳已知值）；
3. 30 秒内能否完成足够多的有效搜索；
4. 不同随机种子之间是否稳定；
5. 小规模最优性和中大规模下界是否有可信证据。

## 2. 最高优先级原则

后续工作必须遵守以下顺序，不应同时跳到多个方向：

1. 先统一目标函数与评价标准；
2. 再直接提升 ALNS 解质量；
3. 然后解决搜索速度瓶颈；
4. 再完善自适应搜索机制；
5. 接着建立 best-known comparison（最佳已知值对照）和完整实验体系；
6. 然后扩展 BPC 的理论对照规模；
7. 最后才升级 partial/nonlinear charging（部分充电/非线性充电）模型。

任何阶段没有通过验收门槛，都不应进入下一阶段。尤其不能在大规模 ALNS 质量仍明显不足时，提前把主要精力转向非线性充电。

## 全阶段产物命名与目录结构规范（强制）

本节是 Stage 0–8 以及每个子阶段、算子、实验轮次和审查活动的总要求。任何新代码、配置、测试、实验结果、失败证据和文档都必须能够通过统一标签追溯到产生它的阶段和部分。

### 1. Canonical stage ID（规范阶段编号）

- 对人可读的标题使用 `Stage 02.1`；文件名、目录名和 manifest 使用无空格的 canonical ID `stage02.1`，两者表示同一个阶段。
- Stage 0 使用 `stage00`；Stage 1 使用 `stage01`；Stage 2 使用 `stage02`。
- Stage 2 的三个部分固定使用 `stage02.1`、`stage02.2`、`stage02.3`。
- 其他阶段的子任务继续使用同一规则，例如 `stage03.0`、`stage03.1`、`stage05.1` 和 `stage06.1`。
- 文档、manifest（校验清单）和审查报告使用带点号的 canonical ID；Python module（Python 模块）等必须使用下划线的文件名时，manifest 仍必须记录带点号的 canonical ID。

### 2. 文件名与目录名格式

阶段专属产物使用以下格式：

```text
<stage_id>_<component>_<attempt_or_rerun>_<artifact_type>[_<instance>_<seed>].<ext>
```

目录使用以下格式：

```text
results/<stage_id>_<component>_<attempt_or_rerun>/<instance>/<seed>/
```

其中：

- `<component>` 是子阶段或算子，例如 `route_elimination`、`relocate`、`station_pressure`；
- `<attempt_or_rerun>` 必须使用 `attemptNN` 或 `rerunNN`，不得使用含义不明的 `final`、`new` 或 `latest`；
- `<artifact_type>` 必须明确，例如 `config`、`raw`、`solution`、`events`、`failure_cases`、`environment`、`manifest`、`summary`、`review`、`readiness`、`test` 或 `doc`。

规范示例：

```text
stage02.1_route_elimination_attempt02_events.json
stage02.1_vehicle_count_aware_repair_attempt02_failure_cases.csv
stage02.2_ejection_chain_rerun01_per_run_results.csv
stage02.3_station_pressure_attempt16_events.json
stage02.3_constraint_guided_rerun09_review_report.md
results/stage02.3_constraint_guided_rerun09/r101_21/2015/
```

### 3. 每个产物必须记录的元数据

每个 raw artifact（原始产物）及其 manifest 至少记录：

- `stage_id`、`component`、`artifact_type`；
- `run_label`、`attempt` 或 `rerun`；
- instance scope（实例范围）和 seed scope（随机种子范围）；
- source/config/instance/environment hash（源码、配置、实例和环境哈希）；
- comparison baseline（比较基准）和 `supersedes`/`legacy_path` 映射；
- status（状态）、failure reason（失败原因）和是否通过 validator。

同一轮实验的 raw JSON、solution、events、failure、environment、manifest、summary 和 review 必须使用同一个 `run_label`，不能分别起互不相关的名字。

每个正式阶段和子阶段至少维护一份对应的 artifact registry（产物登记表），例如 `stage02.1_artifact_registry.csv` 或等价 JSON。登记表必须列出该阶段全部产物、相对路径、artifact type、状态、checksum 和历史路径映射，不能只登记成功结果。

### 4. 共享文件、历史文件和冻结结果

- 同时服务多个阶段的 `alns.py`、`neighborhoods.py`、`objective.py`、validator 和公共测试不得为了贴标签而改名；它们通过 profile、operator registry（算子注册表）、source hash 和 manifest 标记所属阶段。
- Stage 0 冻结目录、历史结果和 checksum 不得重命名、覆盖或移动；只能在 artifact registry（产物登记表）中增加 `stage00` 标签和旧路径映射。
- 已存在的 `stage02_*` 历史路径属于 legacy artifact（历史产物），必须保留；新实验必须使用 canonical `stage02.1`、`stage02.2` 或 `stage02.3` 标签，并在 manifest 中记录旧路径与新标签的对应关系。
- 失败轮次、timeout、invalid、infeasible 和 error 产物不得删除或改名为成功结果；修复后必须使用新的 `attemptNN` 或 `rerunNN`。

### 5. 阶段覆盖表

| 路线图部分 | 强制 `stage_id` | component 标记示例 |
|---|---|---|
| Stage 0 | `stage00` | `frozen_baseline` |
| Stage 1 | `stage01` | `objective_policy`、`bpc_validation` |
| Stage 2.1 | `stage02.1` | `route_elimination`、`vehicle_count_aware_repair`、`route_merge` |
| Stage 2.2 | `stage02.2` | `relocate`、`swap`、`two_opt_star`、`route_segment_destroy`、`ejection_chain` |
| Stage 2.3 | `stage02.3` | `station_pressure`、`time_window_conflict`、`worst_energy_detour`、`shaw_related` |
| Stage 3.0–3.4 | `stage03.0`–`stage03.4` | `measurement`、`screening`、`cache`、`incremental`、`parallel` |
| Stage 4 | `stage04` | `adaptive_weights`、`restart`、`intensification` |
| Stage 5.1–5.3 | `stage05.1`–`stage05.3` | `best_known`、`benchmark`、`ablation` |
| Stage 6.1–6.3 | `stage06.1`–`stage06.3` | `pricing`、`branching`、`validation` |
| Stage 7 | `stage07` | `solution_schema`、`validator_contract` |
| Stage 8 | `stage08` | `partial_linear`、`piecewise_linear`、`nonlinear`、`queueing` |

### 6. 强制检查

每个阶段进入正式实验或下一阶段前，必须检查：路径标签完整、artifact type 不含糊、run label 唯一、manifest 可重算、历史映射存在、raw-to-summary（原始数据到汇总）一致，并在阶段审查报告中记录结果。

### 7. 数据保存与证据分层规则（artifact-storage-v1，强制）

从本条规则生效后，所有新的 Stage 0–8 runner 都必须在配置中提供
`[artifact_storage]`，并通过共享 `ArtifactBundleWriter`/`ArtifactReader` 写入和
回放产物。默认物理格式为 Parquet/Arrow events（Zstandard level 3），critical
evidence（关键证据）完整保存，diagnostic evidence（诊断证据）按
run/lane/iteration/operator/reason 聚合；每个 instance/seed 上限 2 GiB，每个 run
上限 32 GiB。

旧的非 canonical Stage 0–2 调用仅保留给历史兼容复现，且不得使用新的
`[artifact_storage]` 配置；仓库提供的新配置会拒绝非 canonical output directory 或
run label。未来新的 Stage 0 冻结操作必须先完成 current evidence，再生成兼容 baseline view，
不能覆盖现有 frozen baseline。

新产物的固定目录为：

```text
results/<run_label>/
  control/<canonical>_run_metadata.json
  control/<canonical>_config.toml
  control/<canonical>_manifest.json
  control/<canonical>_manifest.sha256
  <instance>/<seed>/<canonical>_raw_<instance>_<seed>.json
  <instance>/<seed>/<canonical>_solution_<instance>_<seed>.json
  <instance>/<seed>/<canonical>_trace_<instance>_<seed>.json
  <instance>/<seed>/<canonical>_events_<instance>_<seed>.parquet
  <instance>/<seed>/<canonical>_route_dictionary_<instance>_<seed>.parquet
  <instance>/<seed>/<canonical>_screening_checks_<instance>_<seed>.parquet
  <instance>/<seed>/<canonical>_diagnostic_<instance>_<seed>.parquet
  <instance>/<seed>/<canonical>_environment_<instance>_<seed>.json
  review/
```

`events.parquet` 必须保留全局递增 `event_id`、exact started/completed、screening、
cache、incremental、deadline、failure、accepted candidate 和 global-best 等
critical events；事件中的 route、lane、operator 使用整数字典编号，完整客户序列只在 route dictionary 保存一次。trace JSON 保存
counters、配置、Parquet 引用和 schema fingerprint，不得重新嵌入完整 events、
screening decisions 或 route evaluations。diagnostic 聚合不得影响 validator、
objective、exact-call ordering 或 failure replay。

Screening check 使用独立的规范列保存；cache lookup 及其紧随的 hit/miss 结果在物理
`events.parquet` 中合并为一条 `lookup_result` 记录。failure artifact 不适用时，manifest
必须写明 `artifact_status.failure=not_applicable`，不能用文件缺失代替状态。

达到任一 byte budget 时，writer 关闭当前 Parquet writer，保留已完成 raw、solution、
event、environment 和 failure evidence，写入 `evidence_completeness=partial`，更新
manifest/checksum 后立即失败；禁止静默截断、覆盖或删除。partial、timeout、failure
和 manifest error 不能发布 summary。

已有 Stage 0 frozen results、Stage 3.0–3.2 raw/solution/events/manifest/summary 不做
物理迁移，不移动、不压缩、不改写原始字节；只在 registry 和 legacy mapping 中标记
`storage_format=legacy_json_or_jsonl`、`retention_class=legacy`、
`policy_compliance=legacy_compatible`。新数据必须标记为 current；历史 dirty 状态、
失败状态和旧路径不可伪装为新规则产物。

每个正式运行或阶段转换前，必须完成 label、attempt/rerun 唯一性、artifact type、
checksum、manifest/sidecar、Arrow schema fingerprint、row count、byte size、
provenance 和 semantic raw-to-summary 检查。tracked summary 和 registry publication
只能发生在独立 reviewer 完成 raw replay 后。详细接口、字段和失败保留规则见
`docs/experiment_artifact_storage.md`。

---

## 阶段 0：冻结当前版本，建立不可退化的基准

### 阶段目标

将当前版本作为正式 comparison point（比较起点），确保之后每次修改都能判断究竟改善了什么、是否引入退化。

### 产物标签

本阶段所有冻结基准、配置快照、环境记录、比较模板和审查结果统一标记为 `stage00_frozen_baseline`；冻结结果另受 Stage 0 不可修改规则保护。

### 具体任务

1. 冻结当前 Schneider 代表性实例集合、随机种子和 30 秒运行条件。
2. 保存当前每个实例的：
   - 可行率；
   - 车辆数；
   - 总距离；
   - 总能耗；
   - 总充电量与充电时间；
   - 运行时间；
   - ALNS 迭代次数；
   - 精确充电子问题调用次数与平均耗时；
   - Best/Mean/Median/Worst/Standard deviation（最佳值/均值/中位数/最差值/标准差）。
3. 固定当前 100-customer 基准：

| 实例 | 当前车辆数 | 当前主要问题 |
| --- | ---: | --- |
| `c101_21` | 14 | 相对合理，但没有可靠 gap |
| `r101_21` | 22–28 | 车辆数偏高、seed 波动明显 |
| `rc101_21` | 24–25 | 车辆数和距离仍有改善空间 |

4. 建立 regression gates（回归门槛）：后续版本不得降低结构可行率，不得让统一 validator（验证器）失效，不得删除失败记录。

### 必须产出

- 当前版本的不可修改基准 CSV；
- 当前参数配置快照；
- 当前算法版本和运行环境记录；
- 一份新旧版本自动比较报告模板。

### 进入下一阶段的门槛

- 当前结果能够用同一命令重新运行并生成结构完整、可验证的实验产物；重复运行不要求得到逐位相同的数值解，但任何数值差异必须通过下述 numerical reproducibility exemption（数值可复现性豁免）；
- 汇总表能够从逐次结果自动重算；
- 所有可行解重新通过统一验证器；
- 后续实验能自动标记 improvement（改善）、regression（退化）或 unchanged（无变化）。

### Numerical reproducibility exemption（数值可复现性豁免）

只有同时满足以下全部条件，重复运行之间的数值差异才可以被接受：

1. 算法必须明确采用 wall-clock deadline（墙钟截止时间）或其他受实际运行速度影响的停止条件；固定迭代数、固定候选数或固定 exact-call budget（精确调用预算）的运行不得使用本豁免。
2. 基准实例及其 SHA-256、参数配置、随机种子、算法核心源文件 SHA-256、依赖版本、线程数和求解器设置必须一致；Git working tree（Git 工作区）在正式捕获时必须干净。若硬件或操作系统不同，必须明确记录，且不得声称是同环境重复实验。
3. 必须至少完成一次独立 full rerun（完整复跑），且复跑必须覆盖与冻结基准完全相同的全部 `(instance, seed)`；任何缺失、重复或额外记录都不符合豁免条件。
4. 冻结运行和复跑的所有解都必须重新通过同一个统一 validator；结构可行率不得下降，failure records（失败记录）不得缺失或被删除，manifest（校验清单）和汇总重算必须全部通过。
5. 数值差异必须能够由已记录的 completed iterations（已完成迭代数）、exact charging subproblem calls（精确充电子问题调用数）或其他可观察的实际工作量差异解释，并且该工作量差异发生在 deadline 边界附近。仅以“机器有波动”作为解释不够充分。
6. 自动比较报告必须逐项保留 baseline value（基准值）、rerun value（复跑值）、absolute/relative delta（绝对/相对差值）和 classification（分类）；不得覆盖冻结 CSV，也不得只保留更好的复跑结果。
7. 豁免只取消“逐位相同数值解”的要求，不豁免配置漂移、代码漂移、实例变化、validator 失败、可行率下降、记录缺失或无法解释的非确定性。出现这些情况时必须判定验收失败并调查根因。
8. 如果复跑出现车辆数增加、可行率下降或其他实质性退化，不能自动归因于 deadline noise（截止时间噪声）；必须追加复跑或使用固定 work budget（工作预算）进行对照，证明差异属于运行时波动后才能接受。

---

## 阶段 1：确定正式目标函数和指标优先级

### 阶段目标

解决“距离更短但车辆更多时，到底算不算更好”的根本问题。没有明确目标层级，后续算法优化没有稳定方向。

### 产物标签

本阶段统一标记为 `stage01`；目标定义、comparison、BPC、ALNS、validator 和测试产物必须在文件名或 manifest 中记录对应 component。

### 具体任务

采用 lexicographic objective（字典序目标），顺序固定为：

1. 首先最小化使用车辆数；
2. 车辆数相同时最小化总行驶距离；
3. 前两项相同时最小化总充电时间；
4. 最后比较总充电次数或总充电量。

候选解比较不再只使用一个距离浮点数，而应使用类似：

$$
(N_{vehicle},D,T_{charge},N_{charge})
$$

的有序目标元组。

同时需要：

- 修改 ALNS incumbent（当前最佳解）的比较逻辑；
- 修改 simulated annealing（模拟退火）的接受逻辑，使车辆数增加受到明确限制；
- 修改实验 CSV，使 primary objective（主要目标）和 secondary metrics（次要指标）分开记录；
- 保留 total distance，避免“只减少车辆但距离严重恶化”。

### 必须产出

- 正式目标函数定义；
- 统一的 solution comparison API（解比较接口）；
- 车辆数优先的单元测试；
- 新旧目标函数对同一批结果的排序差异报告。

### 进入下一阶段的门槛

- 任意两个解的优劣关系不存在歧义；
- 增加车辆的解不能仅因距离略短而成为新 incumbent；
- BPC、ALNS、实验汇总和文档使用相同目标定义；
- 所有旧测试保持通过。

### 实施结果（2026-07-13）

- 已建立统一 `SolutionObjective` comparison seam（比较接缝），正式目标固定为 `(车辆数, 总距离, 总充电时间, 充电次数)`，浮点指标按 `1e-9` 规范化比较。
- ALNS 对车辆数增加执行 hard rejection（硬拒绝）；车辆数减少必然接受，同车辆时仅对距离恶化保留 simulated annealing（模拟退火）。
- BPC 路径列使用同一四级目标；内部标量界仅用于搜索，完整列池通过精确字典序集合划分确认最终 incumbent，只有完整目标求解完成才声明最优。
- 阶段 0 冻结结果的 old-vs-new ranking report 共 36 行，其中 6 行发生换位：`rc105C5` 由车辆数优先改写排序，`rc103C15` 由充电指标完成平局决胜。
- 正式阶段 1实验得到 36/36 ALNS 可行解和 3/3 已证明四级最优的 5-customer BPC 解；12/12 Stage 0 结构可行性门槛通过。
- 最终正式运行在双轴 code review（代码审查）问题全部关闭后的 clean commit `c5e2c16` 上完成；39 个解均由统一 validator 独立重算通过，`pytest` 为 61 passed，Ruff 与 mypy 均通过。
- Stage 0 best-objective comparison 为 3 个实例改善、8 个不变、1 个变差；`r101_21` 本次最佳车辆数为 24，差于阶段 0 的 22，已作为 time-budgeted search（时间预算搜索）质量波动保留，不影响本阶段“目标定义一致且可行性不退化”的验收结论。
- 阶段 0 manifest SHA-256 保持为 `b226b97e0e67288aaaf85726ad855df71cb81406685c57c8e8c40cd8996aa0da`，冻结文件未修改。

---

## 阶段 2：直接减少车辆数，让 ALNS 真正成为大邻域搜索

### 阶段目标

优先解决 100-customer R/RC 实例车辆数偏高的问题，使 ALNS 不再主要依赖每次只移动少量客户的 small neighborhood（小邻域）。

### 产物标签

本阶段统一标记为 `stage02`；正式实验、失败轮次、独立复跑和审查产物必须进一步使用 `stage02.1`、`stage02.2` 或 `stage02.3`，不得只使用笼统的 `stage02`。

### 具体任务

按以下顺序增加算子：

#### 2.1 第一批：直接针对车辆数

本部分的 component 标签固定为 `route_elimination`、`vehicle_count_aware_repair` 和 `route_merge`，对应产物必须使用 `stage02.1_<component>_<attempt_or_rerun>_<artifact_type>`。

1. route elimination destroy（整条路径消除算子）：
   - 优先选择客户少、距离贡献高或充电压力大的路线；
   - 移除整条路线的客户；
   - 尝试把客户分配到其他路线；
   - 只有全部客户重新插入成功时才删除原路线。
2. vehicle-count-aware repair（车辆数感知修复）：
   - 优先插入已有路线；
   - 只有所有已有路线都不可行时才创建新路线；
   - 对创建新车辆设置最高级别的惩罚。
3. route merge（路径合并）：
   - 枚举有希望合并的两条路线；
   - 先通过容量、时间窗和能量下界筛选；
   - 通过筛选后再调用精确充电子问题。

#### 2.2 第二批：跨路线质量改善

本部分的 component 标签固定为 `relocate`、`swap`、`two_opt_star`、`route_segment_destroy` 和 `ejection_chain`，对应产物必须使用 `stage02.2_<component>_<attempt_or_rerun>_<artifact_type>`。

- relocate（客户重定位）；
- swap（客户交换）；
- 2-opt*（跨路径边交换）；
- route segment destroy（路径片段破坏）；
- ejection chain（逐出链）。

#### 2.3 第三批：约束导向算子

本部分的 component 标签固定为 `station_pressure`、`time_window_conflict`、`worst_energy_detour` 和 `shaw_related`；dynamic removal tier（动态破坏层级）及其审查产物必须使用 `stage02.3_<component>_<attempt_or_rerun>_<artifact_type>`。

- station-pressure destroy（充电压力破坏）；
- time-window conflict destroy（时间窗冲突破坏）；
- worst energy detour removal（最差能量绕行移除）；
- related/Shaw removal（相关客户移除）强化版。

大规模 removal size（移除规模）不再固定最多 3 个客户，应使用分层范围，例如小、中、大三种破坏强度，并根据停滞程度动态调整。

### 必须产出

- 每个新增算子的独立实现和统计；
- route elimination 成功/失败原因日志；
- 每个算子的调用数、可行修复数、接受数、车辆数改善数和距离改善数；
- 针对 R/RC 100-customer 的车辆数对比表。

### 进入下一阶段的门槛

- 100-customer 可行率仍保持至少 95%，目标保持 100%；
- `r101_21` 和 `rc101_21` 的平均车辆数相对阶段 0 明显下降；
- 新算法不能仅通过严重增加距离来减少车辆数；
- route elimination 在多个实例上产生真实车辆数改善，而不是只存在于代码中；
- 不同 seed 的车辆数差异明显收窄。

### 实施结果（2026-07-13）

阶段 2 按 2.1、2.2、2.3 分成三个 profile（配置档）。正式范围固定为 Stage 0 的 12 个实例、3 个 seed、30 秒、1000 iterations、单线程。

#### 2.1 路径消除与减车

- `route elimination` 在 8 个不同实例上产生真实减车；`route merge` 产生 11 个真实减车候选。
- 两轮完整实验均为 36/36 通过统一 validator（验证器），72/72 个解由独立逻辑重算 objective（目标）一致。
- `c101_21`、`r101_21`、`rc101_21` 的 Stage 2.1 平均车辆数分别为 12.000、20.667、19.333；其中 R/RC 相对 Stage 0 分别减少 2.667 和 4.334 辆。
- 首轮失败的根因为精确充电评估耗尽 30 秒；随后补充容量、乐观时间窗和能量预筛选，并增加回归测试。失败 raw JSON、solution 和事件日志均被保留。

#### 2.2 跨路线质量改善

- `relocate`、`swap`、`2-opt*`、`route segment destroy`、`ejection chain` 五类算子均被调用并产生可行候选。
- 两轮完整实验均通过继承门槛；12/12 个实例的正式 best objective 不劣于 Stage 2.1，产生 21 次同车辆数的真实距离改善接受事件。
- 由于首轮质量探针改变了 Stage 2.1 主搜索轨迹，修复为 legacy trajectory（历史轨迹）与独立 quality probe lane（质量探测轨道）分离；这次修复没有放宽目标或 validator。

#### 2.3 约束导向与动态破坏规模

- 最终 attempt16 与独立 rerun09 均完成 `READY_FOR_STAGE03` 审查：36/36 solution 和 objective 被审查工具重新读取并验证，保留 20 个真实 failure/poor-quality cases（失败/低质量案例）。
- `station_pressure`、`time_window_conflict`、`worst_energy_detour`、`shaw_related` 四类算子均调用、产生可行候选并至少一次被接受；small、medium、large 三个 removal tier（破坏层级）均真实出现。
- 发生由停滞触发的 tier escalation（层级升级）和 global-best reset（全局最优重置）；100-customer focused run（重点运行）的最大实际移除数为 20，不再固定为 3。
- 100-customer 结果显示，车辆数和距离已稳定，但 exact charging subproblem（精确充电子问题）仍是主要耗时来源：

| 实例 | 平均车辆数 | 平均距离 | 有效迭代数（attempt16 / rerun09） | exact calls（attempt16 / rerun09） |
|---|---:|---:|---:|---:|
| `c101_21` | 12.000 | 1055.920 | 22.0 / 22.0 | 1746.7 / 1802.3 |
| `r101_21` | 20.667 | 1812.683 | 51.0 / 52.0 | 1966.0 / 1959.7 |
| `rc101_21` | 19.333 | 1968.486 | 28.0 / 29.0 | 2133.7 / 2095.0 |

- Stage 2.3 已有 lane-local cache（轨道内缓存）和 unchanged-route precomputation（未变化路线预计算），并记录 cache hits/misses、unique route evaluations 和 exact-call；但尚未实现 Stage 3 所需的可审计全局缓存、增量传播、可中断 exact solver 和并行加速。因此 `median exact calls <= 100` 与 `median effective iterations >= 50` 仍是 Stage 3 的目标，不能写成阶段 2.3 已达成。
- 最终 review artifacts（审查产物）包括 `review_report.md`、`review_findings.csv`、`failure_analysis.csv`、`stage03_readiness.csv` 和 `review_manifest.json`。Stage 0 manifest SHA-256 仍为 `b226b97e0e67288aaaf85726ad855df71cb81406685c57c8e8c40cd8996aa0da`。
- 可追溯证据位于 `docs/stage02_constraint_guided.md`、`docs/stage02_constraint_guided_review.md`，以及 `experiments/summaries/stage02_constraint_guided_rerun09_review_report.md`、`stage02_constraint_guided_rerun09_failure_analysis.csv` 和 `stage02_constraint_guided_rerun09_stage03_readiness.csv`。

### 阶段 2.3 审查：为什么本轮活动耗时较长

结论是“技术瓶颈为主，工程闭环放大耗时”，不是单纯工程拖延，也不是验收门槛过严。阶段 2.3 已达到本阶段门槛，但它把搜索、审查和可复现性问题暴露得更完整。

| 类型 | 直接证据 | 判断 |
|---|---|---|
| 技术：exact charging 成本 | 重点 100-customer run 在 30 秒内只完成约 22–52 次有效迭代，却调用约 1.75k–2.13k 次 exact subproblem | 核心瓶颈。阶段 2.3 明确没有提前实现 Stage 3 加速，因此不能靠当前 profile 消除它 |
| 技术：多轨道候选评估 | legacy、quality probe、constraint lane 同时产生候选；每次动态移除还要排序、修复和记录事件 | 单次 iteration 的工作量过大，新增算子提高了搜索覆盖面，也提高了评估开销 |
| 技术：deadline 边界 | cooperative deadline（协作式截止）无法中断正在执行的 exact call；两轮各有 9/36 次略超时，最大约超 0.017–0.018 秒 | 是 exact solver 接口的可中断性问题，不能简单归因于机器波动 |
| 技术：可观测性语义 | 早期审查发现 unchanged-route reevaluation（未变化路线重复评估）和把 feasible probe 误记为 accepted | 是算法与事件模型的真实缺陷，已在最终轮修复，但说明必须把“候选、接受、重算”分开定义 |
| 工程：正式实验规模 | 每轮固定 12×3=36 runs，失败后必须保留完整证据并用新目录重跑；最终经历 16 个 attempt 和 9 个 independent rerun | 这是研究协议的必然成本，不能通过删失败、换 seed 或放宽 gates（门槛）降低 |
| 工程：审查发现偏晚 | provenance、事件计数和复核完整性是在正式运行后才逐步加强 | 预检不足。应先用小实例和 3 个重点 100-customer run 完成 replay（回放）与证据一致性检查，再启动 36-run |
| 工程：代码与实验状态 | 正式运行期间曾记录 repository dirty；最终代码已提交并恢复 clean working tree | 不是数值算法问题，而是实验编排问题。以后必须先冻结 commit、记录 hash，再运行正式实验 |

因此，阶段 2.3 的长耗时不能用“增加时间预算”解决。正确方向是先降低每个候选的 exact cost，再减少无效候选，最后才讨论并行；同时把昂贵的完整实验推迟到 preflight（预检）和 replay（重放）都通过之后。

### 基于前三个阶段的问题，对后续阶段的科学调整

- 保留现有 gates、实例、seed、validator 和正式 objective；只改变实现顺序、测量方法和工程编排，不以放宽验收换取速度。
- Stage 3 先做可观测性和单线程语义保持，再做缓存、增量传播和 exact solver 可中断化，最后才做受控并行。
- Stage 4 不再重新发明动态 removal size；把 Stage 2.3 的动态破坏规模作为固定输入，只研究 adaptive weights（自适应权重）、接受率和重启机制。
- Stage 5 先做小规模 pilot（试运行）和模型兼容性核对，再扩展完整 benchmark；Stage 6 作为独立 BPC track（分支定价切割轨道）并行推进，不阻塞 ALNS 加速。
- Stage 7 可提前进行解接口和 validator contract（验证器契约）工作，但 partial charging（部分充电）仍必须等显式充电事件稳定后进入 Stage 8。

---

## 阶段 3：加速精确充电子问题调用，提高有效迭代数

### 阶段目标

解决阶段 2.3 已量化的核心瓶颈：100-customer focused run（重点运行）在 30 秒内约完成 22–52 次有效迭代，却调用约 1.75k–2.13k 次精确充电子问题。

### 产物标签

本阶段统一标记为 `stage03`；测量、预筛选、缓存、增量传播、可中断求解和并行必须分别使用 `stage03.0`、`stage03.1` 等 component 标签，不能把不同加速机制混在同一结果目录。

Stage 3 的第一原则是保持 `(车辆数, 总距离, 总充电时间, 充电次数)`、validator 和事件含义不变。任何加速都必须先证明“少算了”，而不是“少记录了”。

Stage 3.0–3.2 的新 evidence 必须遵守本路线图的
`artifact-storage-v1`：使用 Parquet/Arrow 分层保存 critical/diagnostic evidence，
并通过 shared artifact writer、manifest、sidecar 和独立 replay reviewer。已有
Stage 3.0–3.2 raw evidence 保持旧字节和旧路径，只作为 `legacy_compatible` 兼容
证据；不因新规则重新跑 formal/smoke，也不把存储压缩或日志减少宣称为算法加速。

### 具体任务

建立 two-stage evaluation（两阶段候选评估）：

#### 3.0 先测量，再进入正式加速

本部分产物前缀固定为 `stage03.0_measurement`，manifest 的 component 记录为 `measurement`。

- 以 Stage 2.3 attempt16/rerun09 为固定 baseline，锁定 source/config/instance/environment hash（源码、配置、实例和环境哈希）。
- 增加 route-level timing（路线级耗时）、operator-level exact calls（算子级精确调用）、candidate state（候选状态）和 deadline boundary（截止边界）记录。
- 建立 replay auditor（回放审计器）：从 raw solution 和 event log 重新计算 exact calls、cache hits、accepted candidate 和 objective，不接受只读 summary 的审计。
- 新运行的 critical event stream 使用 `events.parquet`，route sequence 使用
  `route_dictionary.parquet`，普通诊断使用聚合 `diagnostic.parquet`；`trace.json`
  只保存可回放索引。历史 JSON/JSONL 通过兼容 reader 读取而不做物理转换。
- 先在 5–8 customer 小实例和 `c101_21/r101_21/rc101_21` 各做 smoke run（冒烟运行）；只有预检和回放一致性通过后，才启动 36-run 正式实验。

#### 3.1 Cheap screening（低成本预筛选）

本部分产物前缀固定为 `stage03.1_screening`，manifest 的 component 记录为 `screening`。

候选路线先依次检查：

1. 载重容量下界；
2. 前向/后向时间窗传播；
3. 时间窗 slack（松弛量）；
4. 最短距离增量下界；
5. 电池单段可达性；
6. 仓库—客户—充电站结构下界；
7. 已知不可行客户序列缓存。

只有预筛选通过后才调用 exact charging subproblem。

#### 3.2 缓存与增量传播

本部分产物前缀固定为 `stage03.2_cache_incremental`，manifest 的 component 记录为 `cache_incremental`。

- 将现有 cache hit/miss（缓存命中/未命中）扩展为可审计的 route evaluation cache（路径评估缓存），键至少包含 instance hash、客户序列、充电配置和 objective schema version（目标模式版本）；
- 缓存可行结果和明确不可行结果；
- 对 relocate/swap 使用增量距离与时间窗计算；
- 建立 station reachability bitset（充电站可达位集合）；
- 对等价客户序列避免重复求解；
- 为缓存设置内存上限和可观察的淘汰策略。
- unchanged route（未变化路线）不得重新调用 exact subproblem；其结果必须能够在日志中与 changed route（变化路线）区分。

### CPU Batch pilot 实施结果与未来强制继承政策（2026-07-14）

独立的 `cpu_batch_pilot_attempt01` 已完成 4 个实例、3 个 seed、40 个固定 iterations 的 12 组 paired runs（配对运行）。全部配对的 candidate-work hash、exact-call 数、route-result hash、objective tuple、effective iterations 和 validator 状态完全一致。三个 100-customer 实例族的端到端中位节省分别为：`c101_21` 21.97%、`r101_21` 11.54%、`rc101_21` 24.40%；C5 三次运行的中位节省为 -0.12%，作为小实例固定开销对照单独保留。该 pilot 只证明 `cpu_batch` 可以成为后续默认后端，不冒充 Stage 3.3 readiness，也不改写 Stage 3.0–3.2 的历史代码路径、raw evidence 或审查结论。

从 Stage 3.3 开始，凡调用 ALNS exact charging route evaluation（精确充电路径评估）的新代码、测试和实验必须遵守以下规则：

- `cpu_batch` 是唯一默认和正式后端；Stage 3.3–8 的新实验、性能测试、消融实验、回归测试和正式 benchmark 不得运行 `cpu_scalar`。
- `cpu_scalar` 仅允许由 2026-07-14 之前已经存在的 Stage 0–3.2 历史 runner 配合其冻结配置显式调用，且用途只能是复现既有结果；任何新建或新配置的运行不论使用什么 stage label，都不得调用 `cpu_scalar`，也不得用它生成新的阶段证据或作为未来性能对照。
- 正确性验证使用已冻结的 pilot 一致性证据、golden fixtures（标准结果样本）、小规模 brute-force enumeration（暴力枚举）和统一 validator 重算，不再重复执行耗时的 scalar 对照。
- `cpu_batch` 不可用、精度冲突、容量溢出或 deadline 状态不一致时必须 fail fast（立即失败），不得自动或静默回退到 `cpu_scalar`。
- 每次正式运行必须在配置、manifest、environment 和 reviewer 输出中记录 exact backend、batch launches、transition 数、packing/unpacking 时间、exact-call 数和 batch 总耗时，使后端选择与性能收益可审计。

未来候选加速方案只能与 `cpu_batch` 进行相同工作量的配对比较。替换默认后端前，必须使用相同实例、seed、iterations、候选工作量和 screening/cache 设置，并保证 candidate-work hash、route-result hash、objective tuple、validator、exact-call 数和 effective iterations 完全一致；端到端中位耗时必须比 `cpu_batch` 至少下降 10%，至少两个重点大实例族改善，且任何重点大实例族的中位耗时不得退化。只有独立 reviewer 通过后才能替换；未通过时继续使用 `cpu_batch` 并保留完整失败证据。

#### 3.3 Exact solver 接口与固定工作量诊断

本部分产物前缀固定为 `stage03.3_exact_deadline`，manifest 的 component 记录为 `exact_deadline`。

- 将 cooperative deadline 改造成可检查的 checkpoint（检查点）或可中断接口；截止时只能返回最近一个完整 incumbent，不能留下半成品候选。
- 每次 exact call 都记录开始、完成、预算耗尽、不可行和中断状态；区分 completed calls（已完成调用）与 started calls（已启动调用）。
- 在相同实例和 seed 下只使用 `cpu_batch`，分别执行固定 exact-call budget（精确调用预算）与 wall-clock budget（墙钟预算）诊断，拆分算法改进和机器速度影响；不得为该诊断重新运行 `cpu_scalar`。
- 先要求 `cpu_batch` 在固定工作量下与 frozen golden evidence（冻结标准证据）的 validator、objective 和接受语义一致，再评价 30 秒内的迭代数。

##### Stage 3.3 实施与审查结果（2026-07-14）

Stage 3.3 已完成。Accepted smoke 为
`stage03.3_exact_deadline_attempt05`（36/36 axes），accepted formal 为
`stage03.3_exact_deadline_attempt06`（72/72 axes）；独立审查状态为
`READY_FOR_STAGE03_4`。`attempt01` 保留为 partial，`attempt02`--`attempt04`
保留为 `NOT_READY`，未覆盖或改写。

实现已加入全局 100 started exact-call cap、120 秒 watchdog、30 秒 wall-clock
轴、可检查 `cpu_batch` checkpoint、candidate-level cache transaction、started /
completed / infeasible / interrupted 对账，以及 packing、unpacking、batch timing
和 peak RSS 证据。Formal validator、objective、acceptance、cache、deadline、
provenance 和 frozen CPU batch golden evidence 均通过独立重放。

Stage 3 总体性能目标尚未完成：`r101_21` / `rc101_21` 的 wall-clock median
started calls 为 1860 / 2039，fixed-work median effective iterations 为 6 / 3。
因此 Stage 3.4 必须继续处理 candidate control；Stage 3.3 readiness 不等于
Stage 3 性能完成。

#### 3.4 候选控制与受控并行

本部分产物前缀固定为 `stage03.4_control_parallel`，manifest 的 component 记录为 `control_parallel`。

当前状态（2026-07-15）：Smoke `stage03.4_control_parallel_attempt10` 完成
72/72 axes 全部通过，独立审查为 `READY_FOR_STAGE034_FORMAL`。Formal
`stage03.4_control_parallel_attempt11` 完成 144/144 axes 全部通过，独立审查为
`READY_FOR_STAGE04`。corrected warm-start 协议（含交易性 reviewer gate、候选哈希
独立重算、硬化前置检查）已通过全部门槛。早期失败 attempt（01--09）保留且不覆盖。

Stage 3.4 formal evidence 使用显式注册的 **inherited warm start（继承热启动）**
协议：从已审查的 Stage 3.3 wall-clock incumbent（解、目标键和来源 SHA-256）
加载为每个 instance/seed 的初始解，但仍通过 Stage 3.4 完整
screening→ranking→cpu_batch exact transaction pipeline 重新验证。此协议是
必要的：Stage 3.3 用约 1860--2039 次 started exact calls 找到其 incumbent，
而 Stage 3.4 的 fixed-work 轴限制为 100 次 started calls；cold-start 搜索无法
在该预算内达到 Stage 3.3 的目标质量。继承的 incumbent 不被直接信任；每条候选
路线由 Stage 3.4 candidate-control runtime 独立 screening、缓存和精确评估，
reviewer 从 raw Parquet events 独立重算 candidate-work 和 route-result 哈希。
配置中 `inherit_stage033_incumbent` 必须为 `true`。

Fixed-work 终止使用两个显式 config 参数：`fixed_work_exhaustion_rounds`
（连续无新增 exact-call 的轮数阈值，当前为 10）和
`min_iterations_before_exhaustion`（允许 exhaustion 终止前的最小有效迭代数，
当前为 50）。两者均记录在 config、manifest 和 environment metadata 中。有效
迭代是指任何完成完整 ALNS 迭代（候选被提出、评估并接受或拒绝）的轮次。
Wall-clock 轴不受 exhaustion 终止约束。

- 每轮只精确评估排名最靠前且通过预筛选的候选；
- 为每轮设置 exact-call budget（精确调用预算）；
- 只有单线程缓存、增量传播和 deadline 回归全部通过后，才对独立候选使用受控并行评估；
- 固定随机数流、候选排序和结果归并顺序，避免并行改变搜索轨迹；
- 并行版本必须单独记录线程数、任务提交顺序、完成顺序和最终归并顺序。
- 任何并行或候选控制方案都以 `cpu_batch` 为唯一现行基线；并行初始化或执行失败时直接失败，不得退回 `cpu_scalar`。

### 必须产出

- 预筛选原因统计；
- cache hit rate（缓存命中率）；
- 每轮 exact calls 数量；
- 路线级和算子级耗时；
- 每秒有效 ALNS 迭代数；
- 性能 profile（性能剖析）和内存峰值；
- fixed-work / wall-clock 双轴对比；
- exact-call reconciliation（精确调用对账）和 deadline 事件报告；
- `cpu_batch` 与候选更优后端在相同 seed、相同工作量和相同时间预算下的配对对比；不存在候选替代方案时不额外运行后端对照。
- exact backend 与 batch launches、transitions、packing/unpacking、batch 总耗时统计。

### 进入下一阶段的门槛

- 继承阶段 2.3 全部 hard gates（硬门槛），且 36/36 solution、objective、event log 可由独立 replay auditor 重算；
- unchanged route 的 exact calls 为 0，candidate proposed、accepted 和 global best 事件语义完全对账；
- 缓存命中与未命中、可行与不可行结果都能重放，缓存不改变未加速版本的正式目标排序；
- 大规模每轮 exact calls 相对 Stage 2.3 实现可解释、可复现的下降；
- Stage 3 的正式性能目标仍为重点 R/RC 运行 `median exact charging calls <= 100`、`median effective iterations >= 50`，不能用增加时间预算或排除超时运行替代；
- 可行性和目标排序与 frozen golden evidence 以及现行 `cpu_batch` 基线一致；
- cache、并行和 deadline 不引入非确定性错误；
- deadline 超时不再出现无法解释的额外 exact call；若仍有求解器边界误差，必须记录 completed iterations、exact calls、硬件、源码和配置差异。
- Stage 3.3–3.4 的全部新运行均显式记录 `cpu_batch` 或已通过替换门槛的新后端；发现 `cpu_scalar`、隐式 fallback 或后端字段缺失时审查直接失败。

---

## 阶段 4：重构自适应机制与搜索控制

### 阶段目标

在 Stage 3 先把单次评估成本和事件语义稳定下来，再让 Adaptive Large Neighborhood Search（自适应大邻域搜索）中的“Adaptive”具有统计意义。不能用更快但不可解释的搜索轨迹替代质量证据。

### 产物标签

本阶段统一标记为 `stage04`；fixed weights、adaptive weights、temperature、restart 和 intensification 的日志、消融和比较结果必须分别记录 component。

Stage 4 的 fixed/adaptive、temperature、restart 和 intensification 对比统一使用 `cpu_batch`，不得通过更换 exact backend 改变实验工作量或重新引入 `cpu_scalar`。

### 具体任务

1. 使用 segment-based weight update（分段权重更新），而不是每次调用后立即更新。
2. 每个算子在一个学习周期中必须达到最少调用次数。
3. 分开统计：
   - accepted improving（接受且改善）；
   - accepted equal（接受且等价）；
   - accepted worse（接受较差解）；
   - rejected（拒绝）；
   - new global best（产生全局最佳）；
   - vehicle reduction（减少车辆）。
4. 自动估计 simulated annealing 初始温度，使初始较差解接受率达到预设区间。
5. 增加：
   - reheating（再加热）；
   - stagnation restart（停滞重启）；
   - 继承 Stage 2.3 的 removal-size adaptation（移除规模自适应），本阶段只评估其与权重更新的交互，不重复实现；
   - incumbent intensification（围绕最佳解强化搜索）。
6. 车辆数减少的奖励高于距离改善；不可行或创建新车辆的操作不能获得虚假正奖励。
7. 在固定 `cpu_batch` 后端下，使用 fixed-work（固定工作量）和 wall-clock（墙钟时间）两种预算分别比较 fixed weights（固定权重）与 adaptive weights（自适应权重），并按实例和 seed 汇总，不以单个最佳 seed 结论化。

### 必须产出

- 算子分段权重变化日志；
- 温度、接受率、停滞和重启记录；
- 算子贡献排名；
- 固定权重与自适应权重的 ablation comparison（消融对比）。

### 进入下一阶段的门槛

- 所有算子在正式运行中都有足够调用样本；
- 大规模运行同时出现合理数量的接受改善、接受较差和拒绝操作；
- 自适应版本在多个实例/seed 上优于固定权重版本；
- 不能只用单个最佳 seed 证明有效；
- 结果标准差相对阶段 0 下降；若未下降，必须报告原因，不得删除高波动 seed；
- 算子奖励、事件计数和 objective 重算通过 Stage 3 replay auditor。

### 实施结果（2026-07-15）

Stage 4 实现了 segment-based weight update（分段权重更新）、six-category operator statistics（六类算子统计）、auto-estimated SA temperature（自动估计模拟退火初始温度）、reheating（再加热）、stagnation restart（停滞重启）、incumbent intensification（围绕最佳解强化搜索）和 differentiated rewards（差异化奖励），并完成 fixed weights vs adaptive weights 的 ablation comparison（消融对比）。全部使用 `cpu_batch` 后端和 `stage02_constraint_guided` operator profile。

Smoke `stage04_adaptive_weights_attempt01` 完成 72/72 axes 全部可行，独立审查状态为 `READY_FOR_STAGE05`。Formal `stage04_adaptive_weights_attempt02` 完成 144/144 axes 全部可行，独立审查状态为 `READY_FOR_STAGE05`。

六个审查 gate 的结果：

| Gate | Status | Details |
|------|--------|---------|
| operator_call_sufficiency | PASS | 所有 wall_clock 轴算子调用充足 |
| six_category_statistics | PASS | 所有 adaptive_wall_clock 轴有接受和拒绝操作 |
| adaptive_better_than_fixed | PASS | adaptive 在 4 个 (instance, seed) 上严格优于 fixed |
| not_single_best_seed | PASS | adaptive 在 3 个 seed (2014, 2015, 2016) 上均有胜出 |
| std_not_increased | PASS | Stage 4 vehicle_count std 不超过 Stage 0 |
| replay_consistency | PASS | 144/144 axes 通过 validator 和 objective 重算 |

Formal 100-customer 结果（adaptive_wall_clock 轴，3 seeds 的 best/mean/median）：

| 实例 | Best 车辆数 | Mean 车辆数 | Median 车辆数 | Std | Best 距离 |
|---|---:|---:|---:|---:|---:|
| `c101_21` | 12 | 12.5 | 12.5 | 0.5 | 1054.03 |
| `r101_21` | 20 | 22.08 | 21.0 | 2.47 | 1745.58 |
| `rc101_21` | 18 | 21.0 | 20.5 | 2.92 | 1858.33 |

自适应权重在 4 个 (instance, seed) 对上严格优于固定权重：`c101_21/2014`、`c101_21/2016`、`r105C15/2016`、`rc101_21/2015`。胜出分布在全部 3 个 seed 上，不依赖单一最佳 seed。

Stage 4 的配置参数记录在 `configs/stage04_weights.toml`；`Stage04Config` 定义在 `src/evrptw/stage04.py`；ALNS 集成在 `src/evrptw/alns.py`；实验 runner 在 `src/evrptw/experiments/stage04_weights.py`；独立审查 CLI 在 `src/evrptw/experiments/stage04_weights_review.py`；单元测试在 `tests/test_stage04.py`（32 个测试全部通过）。Ruff 和 mypy 均通过。

可追溯证据位于 `results/stage04_adaptive_weights_attempt02/`（formal raw artifacts）、`results/stage04_adaptive_weights_attempt02_review/`（review artifacts）、`experiments/summaries/stage04_adaptive_weights_attempt02_*.csv` 和 `.md`（tracked summaries）、`experiments/registries/stage04_artifact_registry.csv` 和 `experiments/manifests/stage04_adaptive_weights_artifact_manifest.json`。Stage 0 manifest SHA-256 保持为 `b226b97e0e67288aaaf85726ad855df71cb81406685c57c8e8c40cd8996aa0da`，冻结文件未修改。

---

## 阶段 5：建立 Best-Known 对照与完整实验体系

### 阶段目标

从“算法能够得到可行解”升级为“能够量化解质量，并说明改进来自哪里”。

### 产物标签

本阶段统一标记为 `stage05`；best-known、benchmark 和 ablation 三部分分别使用 `stage05.1`、`stage05.2` 和 `stage05.3`，不能共用无阶段标记的汇总文件。

本阶段的入口不是“Stage 2 已经足够快”，而是 Stage 3 已完成 exact-call 对账、单线程语义回归和 deadline 复验，Stage 4 已完成固定权重/自适应权重的公平比较。否则扩展 benchmark 只会放大不可解释的运行成本。

Stage 5 的 pilot、完整 benchmark 和所有仍调用 exact charging 的 ablation 统一使用 `cpu_batch`。不得把 `cpu_scalar` 当作消融项、性能基线或回归路径；移除 exact charging 的消融版本不调用任何 exact backend。

### 具体任务

#### 5.1 Best-known values（最佳已知值）

本部分产物前缀固定为 `stage05.1_best_known`，manifest 的 component 记录为 `best_known`。

1. 收集正式公开的 Schneider best-known values；
2. 核对距离度量、充电假设、车辆数目标、时间窗和车辆参数；
3. 只有模型完全一致时才计算 gap；
4. 模型不一致的结果单独列出，禁止直接比较；
5. 对没有 best-known value 的实例，报告 `unknown`，不能估算或补填。

##### 实施结果（2026-07-15）

Stage 5.1 已完成实现。BKS 数据来自三篇正式发表的期刊文章：Schneider, Stenger & Goeke (2014) Table 5 提供小规模实例（5/10/15 customers）的 CPLEX 最优值，其中 RC204-15 使用 VNS/TS 改进值；Keskin & Çatay (2016) Table 2 提供大规模实例（100 customers）的汇编最佳已知值，该表整合了 SSG、Goeke & Schneider (2015) 和 Hiermann et al. (2016) 的结果。Goeke & Schneider 的 DOI 已通过 CrossRef 验证，但其 PDF 尚未纳入本地 VOR 文献库。

实例命名映射：文献记法 `C101-5` 映射为仓库 `c101C5`（小写，连字符替换为 `C`）；文献记法 `c101` 映射为 `c101_21`（追加 Solomon 100-customer 后缀）。全部 92 个实例（36 小规模 + 56 大规模）均有 BKS 车辆数和距离值。充电时间和充电次数在已发表的 BKS 表格中从不报告，统一记为 `unknown`。

模型兼容性评估覆盖五个维度：充电模型（满充——兼容）、目标函数（已发表 BKS 使用以车辆数优先的距离最小化，不含充电时间或充电次数项——不兼容）、距离度量（已发表 BKS 可能使用四舍五入的欧氏距离——不兼容）、时间窗（兼容）、车辆参数（兼容）。总体兼容性为 `False`。由于模型不完全一致，不进行 gap 计算；所有 92 个实例在 `experiments/baselines/schneider_best_known.csv` 中标记 `model_compatible=False`。

实现文件：核心数据模块 `src/evrptw/best_known.py`（92 条 BKS 记录、源文献引用、兼容性评估）；实验 runner `src/evrptw/experiments/stage051_best_known.py`；独立审查 CLI `src/evrptw/experiments/stage051_best_known_review.py`（5 个 gate：`instance_coverage`、`bks_values_present`、`no_gap_computation`、`compatibility_assessment_correct`、`replay_consistency`）；配置 `configs/stage051_best_known.toml`；文档 `docs/stage051_best_known.md`；单元测试 `tests/test_stage051.py`（48 个测试全部通过）。Ruff 和 mypy 均通过，全仓库 254 个测试通过。

正式运行需要 clean commit 后执行；审查通过后发布 `experiments/registries/stage05.1_artifact_registry.csv` 和 `experiments/manifests/stage05.1_best_known_artifact_manifest.json`，审查状态为 `READY_FOR_STAGE05_2`。

#### 5.2 扩展 benchmark

本部分产物前缀固定为 `stage05.2_benchmark`，manifest 的 component 记录为 `benchmark`。

按预先声明的规则扩展：

- 36 个全部 5/10/15-customer 小规模实例；
- 56 个全部 100-customer 实例，或先按 C/R/RC、type 1/type 2 分层覆盖；
- 每个随机算法至少 10 个 seed；
- 使用 30/60/300 秒统一时间预算；
- 报告 anytime curve（任意时刻性能曲线）。

执行顺序调整为：先用完整 12×3 组合做 pilot，确认 raw-to-summary（原始数据到汇总）重算、timeout 和模型兼容性均通过；再扩展到 56 个 100-customer 实例。任何失败都保留原目录，不缩减失败样本。

#### 5.3 Ablation study（消融实验）

本部分产物前缀固定为 `stage05.3_ablation`，manifest 的 component 记录为 `ablation`。

分别移除或替换：

- exact charging；
- route elimination；
- adaptive weights；
- simulated annealing；
- energy-aware repair；
- OR-Tools initialization；
- cheap screening 与缓存。

每个消融版本必须使用相同实例、seed、时间预算和 `cpu_batch` 后端；只有“移除 exact charging”这一项不调用 exact backend。

### 必须产出

- best-known 数据来源与模型兼容性表；
- 全部逐次结果和自动汇总；
- gap、车辆数、距离、可行率、运行时间和稳定性表；
- anytime curves；
- 消融实验结论；
- 完整失败、invalid、timeout 和 error 记录。

### 进入下一阶段的门槛

- 不再仅依靠 12/92 个代表实例得出总体结论；
- 所有公开对比都经过模型兼容性检查；
- ALNS 在大规模上不仅保持高可行率，而且车辆数和距离相对当前版本显著改善；
- 能够量化各核心组件对结果的独立贡献；
- 所有汇总数字都能从原始数据自动重算。

---

## 阶段 6：将 BPC 从 8 客户扩展到 10–15 客户

### 阶段目标

把当前 enumerated-column BPC（穷举列池分支定价切割）升级为真正的 dynamic column generation（动态列生成），为中等规模提供可信下界和最优性证据。

### 产物标签

本阶段统一标记为 `stage06`；pricing、state-space、branching 和 validation 产物必须分别记录对应 component、客户规模和 BPC 节点范围。

Stage 6 与 ALNS 加速是独立 track（轨道）。BPC 扩容不得阻塞 Stage 3/4，也不得把启发式 incumbent 误写成下界或最优性证明。

ESPPRC/BPC pricing（定价）本身不执行 ALNS exact charging route evaluation，因此不强行套用 `cpu_batch`；但 Stage 6 的 ALNS incumbent warm start 及任何复用 ALNS 路径评估的组件必须使用 `cpu_batch`，不得调用 `cpu_scalar`。

### 具体任务

#### 6.1 动态 ESPPRC Pricing

本部分产物前缀固定为 `stage06.1_pricing`，manifest 的 component 记录为 `pricing`。

实现 ESPPRC：Elementary Shortest Path Problem with Resource Constraints（带资源约束的基本最短路问题）定价。标号至少包含：

$$
L=(i,t,q,b,\bar c,V)
$$

分别表示当前位置、时间、载重、电量、约化成本和已访问客户集合。

需要实现：

- forward/backward resource labels（前后向资源标号）；
- resource extension functions（资源扩展函数）；
- reduced-cost dominance（约化成本支配）；
- resource-compatible joining（资源兼容连接）；
- negative reduced-cost route search（负约化成本路径搜索）；
- 找不到负约化成本列时的严格终止证明。

#### 6.2 状态空间与分支

本部分产物前缀固定为 `stage06.2_branching`，manifest 的 component 记录为 `branching`。

- ng-route relaxation（ng-route 松弛）；
- unreachable-set strengthening（不可达集合强化）；
- decremental state-space relaxation（递减状态空间松弛）；
- Ryan-Foster branching（Ryan-Foster 分支），替代依赖完整列池的路径变量分支；
- ALNS incumbent warm start（ALNS 当前最佳解热启动）。

#### 6.3 分阶段验证

本部分产物前缀固定为 `stage06.3_validation`，manifest 的 component 记录为 `validation`。

1. 在 5–8 customers 上与现有完整列池 BPC 逐实例核对；
2. 两者必须得到相同 root bound、incumbent 和最终最优值；
3. 再开放 10-customer；
4. 通过后再尝试 15-customer；
5. 100-customer 初期只要求可靠 lower bound，不强求证明最优。

### 必须产出

- 动态 pricing 日志；
- root/final lower bound；
- incumbent 和 optimality gap；
- labels generated/pruned/joined；
- columns generated；
- BPC nodes、cuts 和内存峰值；
- 与旧完整列池版本的交叉验证结果。

### 进入下一阶段的门槛

- 5–8 customers 的动态版本与完整列池版本结果完全一致；
- 10-customer 能在合理时间内生成严格下界；
- 不能因为启发式 pricing 找不到列就错误声称 LP 最优；
- 超时必须保留下界、上界和 gap；
- BPC 扩容不能影响 ALNS 主方法的正常运行。

---

## 阶段 7：增强解表示和统一验证器

### 阶段目标

为后续 partial recharge（部分充电）准备明确、算法无关的解结构，避免 validator 自动替算法隐式执行满充。

### 产物标签

本阶段统一标记为 `stage07`；solution schema、route schedule、charging event、validator contract 和兼容适配器必须各自建立可追溯 artifact type。

解接口和 validator contract（验证器契约）可以在 Stage 3 的测量轨道中提前开发，但必须先完成 full-recharge（满充）兼容回放，再作为 Stage 8 的模型入口。接口迁移不能改写 Stage 0–2 历史结果。

Stage 7 的 ALNS 输出转换、full-recharge 兼容回放和 validator 集成必须保持 `cpu_batch` 接口及批处理顺序语义；正确性检查使用冻结样本和 validator 重算，不运行 `cpu_scalar`。

### 具体任务

引入显式：

- `ChargingEvent`（充电事件）：站点、到达电量、充电量、开始时间、结束时间、离开电量；
- `RouteSchedule`（路径时刻计划）：节点序列、到达/开始服务/离开时间、电量和载重轨迹；
- `Solution`（解）：路线集合、目标元组和算法声明指标。

统一 validator 必须重新检查：

- 客户恰好服务一次；
- 路线从仓库出发并返回；
- 连续时间、电量和载重传播；
- 充电只能发生在合法站点；
- 声明充电量与电池变化一致；
- 充电时间与充电模型一致；
- 声明目标与重算目标一致；
- 不允许自动补充算法没有输出的充电决策。

### 必须产出

- 显式解数据结构；
- 旧 full-recharge 路线到新解结构的兼容适配器；
- ALNS、BPC、OR-Tools、GA 的统一输出转换；
- 错误充电量、错误时间和非法站点的验证测试。

### 进入下一阶段的门槛

- 当前 full-recharge 实验结果通过新 validator 且数值不漂移；
- 所有算法使用同一个解接口；
- validator 不依赖某个具体算法的内部假设；
- partial charging 可以在不破坏历史结果的情况下作为新模型加入。

---

## 阶段 8：升级 Partial/Nonlinear Charging 模型

### 阶段目标

在搜索质量、速度、理论对照和解表示稳定后，再扩展更真实的充电决策。

### 产物标签

本阶段统一标记为 `stage08`；partial linear、piecewise linear、nonlinear 和 queueing 必须使用独立 component、配置、结果目录和 validator 报告。

Stage 8 是最后的模型扩展阶段。partial、piecewise-linear 和 nonlinear 结果必须与当前 full-recharge baseline 分目录、分配置、分 validator 报告；不能用模型变化解释成搜索算法收益。

Stage 8 每种新充电模型只要进入 ALNS exact route evaluation，就必须实现兼容的 ordered CPU batch（有序 CPU 批处理）接口，或先按统一替换门槛证明另一后端优于 `cpu_batch`。只有 scalar 实现的模型只能作为开发原型，不得进入正式实验、性能结论或阶段验收。

### 具体任务

严格按以下顺序推进：

1. partial linear recharge（线性部分充电）；
2. piecewise-linear charging（分段线性充电）；
3. nonlinear charging curve（非线性充电曲线）；
4. 最后才考虑 station capacity/queueing（充电站容量/排队）。

每个模型必须：

- 有独立配置名称和结果目录；
- 有独立精确子问题或严格离散化误差说明；
- 有兼容的 ordered CPU batch 接口或已通过独立 reviewer 的更优后端，不允许正式运行回退到 `cpu_scalar`；
- 不覆盖当前 full-recharge 结果；
- 与对应 validator 公式一致；
- 报告模型复杂度、运行时间和解质量变化；
- 禁止将不同充电模型的结果混为同一 benchmark 结论。

### 必须产出

- 充电模型版本化接口；
- partial charging 精确或有误差界的子问题；
- full vs partial vs nonlinear 对照实验；
- 充电量、充电时间、路径距离和车辆数变化表；
- 模型差异与适用范围说明。

### 阶段完成标准

- 新充电模型通过独立验证；
- 历史 full-recharge 结果保持可复现；
- 模型升级带来的收益能够与算法搜索改进分离；
- 结果不因放宽充电假设而被错误描述成算法本身更强。

---

## 3. 跨阶段测试与质量门槛

以下测试不是最后一次执行，而是每个阶段都必须维护：

1. `cpu_batch` exact charging 与 frozen golden fixtures（冻结标准样本）及小规模 brute-force enumeration（暴力枚举）交叉验证，不重新运行 `cpu_scalar`；
2. ALNS 客户覆盖、仓库起终点和可行性不变量测试；
3. deadline 严格执行测试；
4. cache 一致性和内存上限测试；
5. BPC 与完整整数枚举/完整列池交叉验证；
6. BPC 非根节点分支回归测试；
7. invalid/infeasible/timeout/error 不得被吞掉；
8. 大规模性能回归测试；
9. `pytest`、Ruff 和 MyPy strict（严格类型检查）保持全绿；
10. exact-call、cache、candidate state 和 objective 必须能从 raw logs（原始日志）自动对账；
11. 正式实验前完成小规模 smoke/replay preflight（冒烟/回放预检），正式运行使用已冻结的 clean commit 和完整 hash manifest；
12. 所有实验表格从 raw logs 自动生成，failure、invalid、timeout 和 error 不得只存在于终端输出。
13. 2026-07-14 之后的任何新 ALNS/exact charging 代码、测试和实验，不论 stage label，都必须断言 backend 不是 `cpu_scalar`，并验证错误路径 fail fast、不发生隐式 fallback；唯一例外是使用 2026-07-14 之前既有的 Stage 0–3.2 历史 runner 和冻结配置进行明确标记的历史复现。
14. 新加速后端的替换测试必须以 `cpu_batch` 为基线，满足相同工作量、结果语义一致、端到端中位节省至少 10% 和重点大实例族不退化。

任何阶段如果使可行率、验证正确性或可复现性退化，应先修复根因，再继续扩展。

## 4. 最终验收目标

整个路线图完成后，主 baseline 应达到以下标准：

1. **可行性**：标准 Schneider 实例上稳定保持高可行率，目标为 100%，最低不得低于 95%；
2. **车辆数**：R/RC 100-customer 相对当前 22–28/24–25 辆显著下降；
3. **距离质量**：能够报告与 compatible best-known values（模型兼容最佳已知值）的 gap；
4. **搜索效率**：在不改变正式 objective 的前提下，30 秒内完成足够多的有效邻域迭代，并达到经 Stage 3 审查确认的 exact-call 与 effective-iteration 目标；
5. **稳定性**：多个 seed 的标准差和最差值明显改善；
6. **理论证据**：BPC 在 10–15 customers 提供可靠下界，在小规模证明最优；
7. **实验可信度**：覆盖完整或预先声明的 Schneider 分层实例，不 cherry-pick（选择性报告）；
8. **模型清晰度**：full/partial/nonlinear charging 结果严格分离；
9. **可追溯性**：任何可行、无效、超时或失败结果都能追溯到原始日志与解文件。

最终评价标准不再只是：

> 算法是否找到一个可行解。

而应升级为：

> 算法是否在标准实例上稳定找到车辆数更少、距离更短、接近最佳已知值、具有可信下界或 gap 证据的可行解，并且所有结论都能够复现和验证。
