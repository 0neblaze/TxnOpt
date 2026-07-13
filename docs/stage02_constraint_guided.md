# 阶段 2.3：约束导向算子与动态破坏规模

## 1. 阶段边界

Stage 2.3 在已通过的 Stage 2.2 `stage02_route_quality` 轨迹上增加
constraint-guided search（约束导向搜索）。`solve_alns()` 的默认 profile（配置档）为
`stage02_constraint_guided`，但 `baseline`、`stage02_route_reduction` 和
`stage02_route_quality` 仍可显式调用，以保持 Stage 0、Stage 1、Stage 2.1 和 Stage 2.2
的历史复现能力。

本阶段不实现 Stage 3 的 cache acceleration（缓存加速）、parallel evaluation（并行评估）或
exact charging acceleration（精确充电加速）。Stage 3 的性能数值是进入审查时登记的目标，
不是 Stage 2.3 的已完成结果。

正式 objective（目标）固定为：

```text
(vehicle_count, total_distance, total_charging_time, charging_count)
```

车辆数拥有绝对优先级；所有构造、比较和充电次数统计继续由 `evrptw.objective` 完成。

## 2. Deep module（深模块）接口

`src/evrptw/neighborhoods.py` 对调用者隐藏排序、预筛选、repair（回填）和事件细节，并提供：

- `ConstraintRemovalOperator.station_pressure`：综合路线充电时间、充电能量、充电站访问和
  局部充电绕行；
- `ConstraintRemovalOperator.time_window_conflict`：重放 exact route path（精确路线）并按
  time-window slack（时间窗松弛）排序；
- `ConstraintRemovalOperator.worst_energy_detour`：比较实际充电路线局部距离和直接客户连接；
- `ConstraintRemovalOperator.shaw_related`：以 seeded anchor（有种子控制的锚点）综合空间、
  时间窗、需求和乐观电量可达性；
- `select_dynamic_removal_size(...)`：根据停滞长度、层级和周期探索确定移除规模；
- `propose_constraint_removal(...)`：产生带 deterministic tie-break（确定性平局决策）的
  `RemovalProposal`；
- `repair_constraint_removal(...)`：只回填已有路线，不创建新路线。

候选仍经过共享的 capacity（容量）、optimistic time-window（乐观时间窗）和 optimistic
energy reachability（乐观电量可达性）筛选。发生变化的路线才进入 exact charging
subproblem（精确充电子问题）。所有 probe、接受、拒绝、不可行、预算耗尽和新增车辆事件都
保留在 operator event log（算子事件日志）中。

## 3. Dynamic removal size（动态破坏规模）

只有 Stage 2.3 启用动态破坏规模；早期 profile 的破坏行为不改变。固定配置位于
`configs/stage02_constraint_guided.toml`：

| tier（层级） | 客户数比例 |
| --- | ---: |
| `small` | 5%--10% |
| `medium` | 10%--20% |
| `large` | 20%--35% |

中等停滞阈值为 4 iterations（迭代），大型停滞阈值为 8 iterations，exploration period
（探索周期）为 3 iterations。requested count（请求数量）和 actual count（实际数量）始终
限制在 `1` 至 `customer_count - 1`；100-customer 实例不再固定截断为 3 个客户。

在正式 Stage 2.3 配置中，quality shadow（质量探针）对两路线算子保留 2 次 exact route
evaluation（精确路线评估），route-segment probe（路线片段探针）保留 4 次；两项都显式写入
TOML。约束 lane 另有显式 0.1 秒 slice（时间片）；主/质量 legacy lane 使用其余 29.9 秒，
总实验预算仍为 30 秒。该配置是在完整失败轮次中验证后固定的：预算 4 曾造成 objective 回退，
预算 1 又无法满足两路线算子的 feasible-candidate gate，预算 2 加 route-segment 4 才同时满足
质量和时间门槛。

`rerun05` 暴露了固定 30 秒边界下 `rc101_21/2015` 在已减车但距离较差的中间解上停止的
问题。根因不是 validator 或 objective 不一致，而是低车队解的 legacy trajectory（历史轨迹）
在时间预算耗尽前缺少确定性距离强化。随后新增 `vehicle_reduction_refinement`：仅在减车后
进入低车队区间时触发，使用 `worst + regret-2` 的确定性既有路线回填，TOML 中固定
`vehicle_reduction_refinement_exact_evaluation_budget = 512`，不创建新路线，并将调用、可行性、
接受和 exact evaluation 记录到事件及 operator statistics（算子统计）中。该修复增加了
attempt13/rerun06 两轮完整结果；随后又修正了 cooperative deadline（协作式截止时间）边界、
预计算路线的 exact evaluation 计数、constraint timeout event（约束超时事件）保留以及局部
候选的完整 objective 重算，形成 attempt15/rerun08 证据；随后修复了 quality/ejection 对未变化
路线的重复 exact evaluation（精确评估）以及 feasible probe 被误记为 accepted 的事件语义，形成
最终 attempt16/rerun09 证据。所有更早结果仍保留，rerun05
的失败证据不被覆盖。

选择器根据最近一次 global-best improvement（全局最佳改善）计算停滞长度。达到阈值时提升层级；
周期探索只能在满足停滞条件后再提升一层；全局最佳改善会记录 reset（重置）并将停滞重新计数。
每次选择记录 tier、请求数、实际数、停滞长度、触发原因和 reset 标记。

## 4. Constraint lane（约束导向搜索轨道）

Stage 2.2 的 legacy trajectory（历史主搜索轨迹）保持独立运行。四个新算子先进行
round-robin warm-up（轮询预热），之后按统计权重在 constraint lane 中调用；该 lane 使用独立
seed-controlled random stream（种子控制的随机流），不会静默改变历史主轨迹。

只有通过统一 lexicographic objective 的候选才能更新 global best；任何接受的候选都不能增加
车辆数。`ALNSResult` 和 `NeighborhoodEvent` 额外记录 cache hits/misses（缓存命中/未命中）、
unique route evaluations（唯一路线评估）、effective iterations（有效迭代）、removal tier
counts、maximum stagnation，以及四个约束算子的调用、可行、接受、拒绝和失败原因统计。

## 5. 固定实验协议

正式对照固定为已通过的 Stage 2.2 结果：
`experiments/summaries/stage02_quality_attempt02_per_run_results.csv`。

实验范围固定为 Stage 0 的 12 个实例、seeds `2014/2015/2016`、30 秒、最多 1000 iterations
和单线程，并单独检查 `c101_21`、`r101_21`、`rc101_21` 的 100-customer 结果。

```bash
uv run pytest
uv run ruff check .
uv run mypy

uv run python -m evrptw.experiments.stage02_constraint_guided \
  --config configs/stage02_constraint_guided.toml \
  --output-dir results/stage02-constraint-guided_attempt16 \
  --run-label stage02_constraint_guided_attempt16

uv run python -m evrptw.experiments.stage02_constraint_guided_review \
  --run-dir results/stage02-constraint-guided_attempt16 \
  --comparison-dir results/stage02-quality_attempt02 \
  --review-label stage02_constraint_guided_attempt16
```

独立完整复跑必须使用全新目录和 label（标签）：

```bash
uv run python -m evrptw.experiments.stage02_constraint_guided \
  --config configs/stage02_constraint_guided.toml \
  --output-dir results/stage02-constraint-guided-rerun09 \
  --run-label stage02_constraint_guided_rerun09 \
  --repeat-of results/stage02-constraint-guided_attempt16

uv run python -m evrptw.experiments.stage02_constraint_guided_review \
  --run-dir results/stage02-constraint-guided-rerun09 \
  --comparison-dir results/stage02-quality_attempt02 \
  --review-label stage02_constraint_guided_rerun09
```

raw artifacts（原始产物）位于 Git-ignored `results/`；tracked summaries（受版本控制的汇总）
位于 `experiments/summaries/`。失败轮次使用新的目录和 label，且 attempt01 至 attempt13
以及 rerun01 至 rerun08 的证据均保留；最终验收证据为 attempt16/rerun09。

## 6. 失败闭环

第一轮 `stage02_constraint_guided_attempt01` 的失败包括
`stage02_2_best_objective` 和动态移除数量范围。根因是约束 lane 的回填和精确评估开销在
固定 30 秒预算中干扰了历史质量轨迹；修复为独立、可观测的 bounded constraint lane，并
保留统一 objective 比较。

第二轮 `attempt02` 通过；第三轮 `attempt03` 暴露了 `medium` tier 在边界条件下未出现，
因此补充了确定性的停滞层级边界测试和回归修复。第四轮原先通过 gate，但双轴代码审查发现
约束回填是恢复原位置的 no-op（空操作），该轮不作为最终算法证据。第五轮改为真实 exact
repair（精确回填）后暴露了预算侵占主轨迹的问题，`stage02_2_best_objective` 失败；因此
将每次约束 probe 的正式 exact budget 固定为 4，并在第六轮重新完成全部实验；随后根据双轴
复核对事件重放、哈希重算和 no-op 候选的意见，在第七轮重新完成全部实验。所有失败和
被 supersede（替代）的轮次都保留，没有通过更换实例、seed、validator、objective 或删除
记录来处理。第八轮因 `rc101_21/2015` 的同车辆数距离退化而失败；第九轮用预算 1 恢复
objective 但暴露四类质量算子 candidate gate 失败；第十轮用预算 2 后只剩
`route_segment_destroy` candidate gate，最终第十一轮增加显式 route-segment probe budget
并通过全部 gates。第十二轮在新 lane 隔离实现下暴露 `rc101_21/2015` 的时间边界目标
退化；第十三轮加入受 512 次 exact evaluation budget 约束的低车队 refinement 后通过，
第六次独立复跑也通过全部 gates。第十四轮针对双轴审查发现的 deadline、预计算计数和
超时事件丢失问题增加回归修复；第十五轮继续修正了局部候选的完整 objective、cache-hit
exact evaluation 计数、约束 lane 的 unchanged route 重评估和中断产物保留；第十六轮再修正
quality/ejection 的 unchanged route 重评估和 feasible probe 的 accepted 标记；attempt16 与
rerun09 在这些修复后再次通过全部 gates。

## 7. 正式验收结果（2026-07-13）

`stage02_constraint_guided_attempt16` 和独立完整复跑
`stage02_constraint_guided_rerun09` 均通过全部 hard gates（硬性门槛）：

| Gate / 指标 | 结果 |
| --- | --- |
| 完整唯一 runs | 两轮均 36/36 |
| unified validator | 两轮均 36/36 |
| Stage 2.2 objective comparison | 12/12 instances 均 equal |
| `r101_21` vehicle mean | 20.667；Stage 0 为 23.333 |
| `rc101_21` vehicle mean | 19.333；Stage 0 为 23.667 |
| R/RC mean distance | 1812.683 / 1968.486，均低于 Stage 1 × 1.10 |
| R/RC vehicle standard deviation | 0.471 / 0.471，不高于 Stage 0 |
| 四个约束算子 | 均调用、产生 feasible candidate、产生 accepted candidate |
| quality/ejection exact-event reconciliation | 两轮质量算子事件均记录 348 次 exact evaluation；未变化路线使用 precomputed route results |
| dynamic removal tiers | `small`、`medium`、`large` 均真实出现 |
| 100-customer actual removal > 3 | 通过；集中审计事件最大实际移除数为 20 |
| 停滞升级 / global-best reset | 通过；attempt16 / rerun09 分别记录 17892 / 17896 次升级，均记录 144 次 reset |
| constraint-level diagnosis | 每轮 20 个真实案例，覆盖 time-window、operator failure、computational limit |
| Stage 0 manifest | SHA-256 保持 `b226b97e0e67288aaaf85726ad855df71cb81406685c57c8e8c40cd8996aa0da` |
| independent complete rerun | pass；36/36 keys，配置一致 |

独立审查对两轮的 raw solution 和 event log 重新读取，并重新执行 validator 和 objective
重算，同时独立重算 runner hard gates；首轮审查状态为
`PENDING_INDEPENDENT_RERUN`，独立复跑审查状态为 `READY_FOR_STAGE03`。审查还重新验证了
首轮的 raw evidence、事件、formal configuration、provenance、time-budget exemption、哈希和
manifest，而不是信任首轮 gate CSV；review CLI source hash 也写入 review provenance。由于历史 Stage 2.2
summary 没有记录新 instrumentation 字段，另生成了与固定
objective 逐行一致的 `stage02_quality_attempt02_readiness_metrics.csv` supplement；审查
不会将它当作正式 objective baseline。审查产物见
[Stage 2.3 review protocol](stage02_constraint_guided_review.md)。

## 8. Numerical reproducibility exemption（数值可复现性豁免记录）

正式配置仍是 30 秒总预算；exact charging solver（精确充电子问题求解器）当前为协作式
截止时间，单次不可中断的 exact call（精确调用）可能在截止点后完成。attempt16 的 36 个
run 中 9 个 runtime 略高于 30 秒（最大 30.018086 秒），rerun09 为同样 9 个（最大
30.017146 秒）。这不是提高预算或静默忽略超时：每个 run 的 completed iterations、exact
charging calls、runtime、failure events、硬件、源码 hash 和 TOML hash 均记录在对应的
`per_run_results.csv`、`*_environment.json` 和 raw manifest 中。两轮的 objective、validator
结果和 hard gates 均通过独立重算；因此该豁免只记录 cooperative cutoff 的数值边界，不改变
正式 objective、实例、seed、validator 或 acceptance gates。

attempt16/rerun09 的共同 provenance（来源）为：`arm64`、Python `3.13.13`，algorithm source
SHA-256 `b744edb9f3aee4d52078281494d839d69c3540f1ee5735b618fc36fb365ad433`，configuration
SHA-256 `a206249ce19ec810f6ef4b04cfa9dd6d841bd8da4db9f05764450615b685fff5`；review CLI source
SHA-256 `dd99343d40f27d0a469adf7f721250b6ecab070d998a365bd661c54056e5ac4a`。

## 9. Stage 3 进入目标（仅登记，不宣称已完成）

阶段 3 审查登记以下目标：100-customer R/RC 的 median exact charging calls（中位精确充电
调用次数）不超过 100，median effective iterations（中位有效迭代）至少 50，validator
feasibility 保持 100%，且加速不造成 objective regression（目标退化）。缓存、并行和精确充电
加速必须在 Stage 3 单独实现、实验和验收。
