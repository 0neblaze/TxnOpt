# 阶段 1：车辆数优先的 Lexicographic Objective

## 1. 正式目标

阶段 1 起，所有可行解使用同一个 lexicographic objective（字典序目标）：

\[
(N_{vehicle}, D, T_{charge}, N_{charge}).
\]

依次最小化车辆数、总行驶距离、总充电时间和充电站访问次数。原始浮点指标完整记录；
比较时把距离和充电时间规范化到 `1e-9`，从而得到明确、可传递的排序。`objective_value`
仅作为历史 total-distance compatibility alias（总距离兼容别名），不再代表正式目标。

唯一 comparison seam（比较接缝）是 `evrptw.objective` 中的 `SolutionObjective` 和
`compare_objectives()`。ALNS、BPC、实验 runner 和测试不得自行复制 tuple 比较规则。

## 2. ALNS 接受与 Incumbent

- 候选车辆数减少：必然接受；
- 候选车辆数增加：始终拒绝，不允许通过距离缩短补偿；
- 车辆数相同且字典序改善：接受；
- 车辆数、距离相同但充电指标恶化：拒绝；
- 车辆数相同但距离恶化：保留 simulated annealing（模拟退火）概率接受。

全局 incumbent（当前最优解）始终使用完整四级目标。阶段 1 不修改 destroy/repair
operators（破坏/修复算子）的邻域结构；车辆感知修复和路径消除属于阶段 2。

## 3. BPC 目标与证明语义

每个 route column（路径列）携带 `(1, distance, charging_time, charging_count)` 贡献。
内部 LP/branch search（线性规划/分支搜索）使用从完整列池距离上界推导的车辆优先搜索
系数，而不是任意常数。内部 scalar search bounds（标量搜索界）仅用于搜索、剪枝和队列
排序，不是正式 objective 或 lexicographic gap（字典序差距）。

搜索正常完成后，完整列池通过 exact lexicographic set partitioning（精确字典序集合划分）
选择四级最优列组合；只有该步骤完成且搜索未超时，结果才能标记 `proven_optimal=True`。
超时结果可以保留已找到的可行 incumbent，但不得声明完整四级目标已证明最优。

## 4. 实验字段与命令

阶段 1 per-run CSV 明确分离：

- `primary_vehicle_count`；
- `secondary_total_distance`；
- `tertiary_total_charging_time`；
- `quaternary_charging_count`；
- `objective_key`。

正式运行复用阶段 0 的 12 个实例、seeds `2014/2015/2016`、30 秒、1000 次最大迭代和
单线程设置；5-customer 实例额外运行 BPC。

```bash
uv run python -m evrptw.experiments.stage01_objective \
  --config configs/stage00_baseline.toml \
  --baseline-dir experiments/baselines/stage00 \
  --output-dir results/stage01 \
  --summary-dir experiments/summaries
```

完整 raw logs（原始日志）和 solutions（解）保存在 ignored `results/stage01/`；审阅用
per-run、summary、failure、Stage 0 comparison 和 old-vs-new ranking CSV 保存在
`experiments/summaries/`。runner 在运行前重新验证阶段 0 manifest，并在结束时拒绝任何
可行率退化、缺失 seed 或 validator 失败。

## 5. 冻结结果排序审计

在阶段 0 的 36 条冻结结果上，新旧排序有 6 条记录换位：

- `rc105C5`：距离最短的 3 车解不再优于距离略长的 2 车解；
- `rc103C15`：车辆数与距离相同时，由充电时间和充电次数完成 tie-break（平局决胜）。

该报告只读取阶段 0 冻结 CSV 和 solutions，不修改其 manifest 或任何冻结文件。
