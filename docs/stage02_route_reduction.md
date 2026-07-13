# 阶段 2.1：Route Reduction Operators

## 1. 范围与不变量

阶段 2.1 在阶段 1 的 lexicographic objective（字典序目标）上增加三个邻域算子：

- `route elimination destroy`：移除一条完整路线，并尝试把其全部客户回填到其余已有路线；
- `vehicle-count-aware repair`：先只检查已有路线，只有全部已有路线都无法容纳时才允许新增路线；
- `route merge`：合并两条路线，先做容量、乐观 time-window（时间窗）和 optimistic energy
  reachability（乐观电量可达性）筛选，再调用 exact charging subproblem（精确充电子问题）。

正式目标仍是：

```text
(vehicle_count, total_distance, total_charging_time, charging_count)
```

所有目标构造、比较和充电站访问次数统计都通过 `evrptw.objective`。候选增加车辆数时，
即使距离更短，也不能被 simulated annealing（模拟退火）接受。

## 2. 实现边界

`src/evrptw/neighborhoods.py` 是 deep module（深模块），隐藏路线筛选、插入顺序、精确评估
预算和失败原因。`solve_alns()` 默认使用 `stage02_route_reduction` profile（配置档），并
保留 `baseline` profile 供 Stage 0/Stage 1 历史流程复现。Stage 0 冻结目录、Stage 1 历史
结果和 `reference/` 参考仓库不修改。

每次新算子调用都会记录：调用、prefilter（预筛选）通过/拒绝、精确路线评估、候选可行性、
接受/拒绝、车辆减少、距离改善、新路线数量和失败原因。正式实验还保存每次运行的 raw JSON、
solution、事件日志、failure cases、environment metadata（环境元数据）和 manifest。

## 3. 固定实验协议

正式配置为 `configs/stage02_route_reduction.toml`：

- Stage 0 的 12 个实例；
- seeds `2014/2015/2016`；
- 每次运行 30 秒、最多 1000 次迭代、单线程；
- 独立审计 `c101_21`、`r101_21`、`rc101_21` 三个 100-customer 实例。

运行命令：

```bash
uv run pytest
uv run ruff check .
uv run mypy
uv run python -m evrptw.experiments.stage02_route_reduction \
  --config configs/stage02_route_reduction.toml

# 第二次独立完整复跑；路径和 run label 必须是新的
uv run python -m evrptw.experiments.stage02_route_reduction \
  --config configs/stage02_route_reduction.toml \
  --output-dir results/stage02-rerun01 \
  --run-label stage02_rerun01 \
  --repeat-of results/stage02
```

每一轮必须使用新的 `--run-label` 和 raw output directory。runner 在开始时重新验证 Stage 0
manifest，并在正式结果写入后重新验证 validator、Stage 1 比较、100-customer 车辆数/距离/seed
稳定性、路线消除多实例成功、路线合并候选和 Stage 0 不变性。
第一次运行不覆盖已有结果；第二次独立运行使用新的 output directory 和 run label，并传入
`--repeat-of` 指向第一次 output directory。第二次 runner 会生成 repeatability CSV，并把第一轮和
第二轮的 gate、36 个 `(instance, seed)` keys、配置一致性汇总为
`independent_complete_rerun` gate。

## 4. Acceptance gates（验收门槛）

阶段 2.1 只有在以下条件全部满足后才完成：

1. 36/36 runs 通过统一 validator；
2. 每个实例的 Stage 2.1 最佳正式目标不劣于 Stage 1；
3. `r101_21` 和 `rc101_21` 平均车辆数至少比 Stage 0 低 1 辆；
4. 100-customer R/RC 平均距离不超过 Stage 1 的 110%；
5. route elimination 在至少两个不同实例产生真实减车；
6. route merge 至少产生一个真实减车候选；
7. R/RC 车辆数标准差不高于 Stage 0；
8. Stage 0 manifest、冻结结果和 checksum 不变；
9. 相同配置完成两次独立完整复跑，且两次均通过全部 hard gates（硬性门槛）。

结果不理想不能通过换实例、换 seed、删除失败记录、放宽 validator、修改正式目标或排除不可行
结果来处理。若时间截止导致数值变化，必须保留 completed iterations、exact charging calls、
硬件、源码和配置差异，并按项目既有 numerical reproducibility exemption（数值可复现性豁免）
记录。

## 5. 失败闭环

每轮失败都保留完整证据，并在下一轮前定位具体 gate 和根因。修复后需要新增 regression test
（回归测试），在不可覆盖的新目录重新运行 36 次正式实验，再重新检查全部 gates。阶段文档只在
正式实验和两次独立复跑全部通过后更新为完成状态。
