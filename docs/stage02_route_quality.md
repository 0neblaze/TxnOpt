# 阶段 2.2：跨路线质量改善

## 1. 范围与不变量

Stage 2.2 在 Stage 2.1 的 lexicographic objective（字典序目标）和三个
route-reduction operator（路线减少算子）之上增加五个跨路线算子：

- `relocate`（客户重定位）；
- `swap`（客户交换）；
- `two_opt_star`（跨路线尾段交换）；
- `route_segment_destroy`（连续路径片段破坏与已有路线回填）；
- `ejection_chain`（逐出链，最大深度 3、beam width 16）。

正式目标仍为 `(vehicle_count, total_distance, total_charging_time,
charging_count)`，并继续统一使用 `evrptw.objective`。新增算子不创建路线；车辆数
优先级和统一 validator（验证器）规则不变。

## 2. 实现与记录

`src/evrptw/neighborhoods.py` 是 deep module（深模块）。五个 proposal interface
均先调用共享的容量、乐观 time-window（时间窗）和 optimistic energy reachability
（乐观电量可达性）prefilter（预筛选），然后只对发生变化的路线调用 exact charging
subproblem（精确充电子问题）。事件记录包含 affected routes、candidate routes、
prefilter、exact calls、chain depth、segment length、feasibility、acceptance、
distance improvement 和 failure reason。

正式配置为 `configs/stage02_route_quality.toml`，固定使用 Stage 0 的 12 个实例、
三个 seed、30 秒、1000 iterations（迭代）和单线程。objective comparison（目标比较）
固定对照已通过的 Stage 2.1 结果：
`experiments/summaries/stage02_attempt02_per_run_results.csv`。

## 3. Acceptance gates（验收门槛）

本阶段继承 Stage 2.1 的全部 hard gates（硬性门槛），并增加：

1. 36/36 runs 通过统一 validator；
2. 五个新增算子均至少被调用，并各自产生至少一个可行 candidate；
3. 至少一个新增跨路线算子产生真实接受的同车辆数 distance improvement；
4. 每个实例的 Stage 2.2 最佳正式 objective 不劣于 Stage 2.1；
5. R/RC 100-customer 的车辆均值、距离上限和 seed 稳定性继续通过；
6. Stage 0 manifest、冻结结果和 checksum 不变；
7. 第二次独立完整复跑通过 `independent_complete_rerun`。

失败轮次保留新的 `results/` raw JSON、solution、事件日志、failure cases、environment
和 manifest，不覆盖既有证据。只有第一轮和独立复跑全部 hard gates 通过后，才在本节记录
正式完成结果。

## 4. 运行命令

```bash
uv run pytest
uv run ruff check .
uv run mypy
uv run python -m evrptw.experiments.stage02_route_quality \
  --config configs/stage02_route_quality.toml

uv run python -m evrptw.experiments.stage02_route_quality \
  --config configs/stage02_route_quality.toml \
  --output-dir results/stage02-quality-rerun01 \
  --run-label stage02_quality_rerun01 \
  --repeat-of results/stage02-quality_attempt01
```
