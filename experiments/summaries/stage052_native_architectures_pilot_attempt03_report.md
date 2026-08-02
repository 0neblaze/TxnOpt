# Stage 5.2 五种原生架构同机对比报告

- Review status（审查状态）：`COMPARISON_COMPLETE_NOT_QUALIFIED`
- Raw replay（原始重放）：`True`
- Axis count（轴数）：`180`
- Formal：未启动；production default（生产默认值）：未改变。

## 五模式事实表

| Mode | Solver median s | Effective iterations median | Exact calls median | Queue wait median s | Exact batch occupancy median | RSS median MiB | Artifact median MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_stage052 | 8.97831 | 1000 | 749 | n/a | 1 | 230.756 | 0.018 |
| python_candidate_control | 5.91253 | 1000 | 88 | n/a | 1 | 238.000 | 0.016 |
| per_solve_runtime | 5.60867 | 1000 | 91 | 0 | 1 | 195.234 | 0.019 |
| full_native_alns | 0.0429362 | 1000 | 3683 | 0.00176984 | 4 | 158.100 | 0.011 |
| host_scheduler | 0.0504111 | 1000 | 3683 | 0.00681646 | 4 | 155.604 | 0.011 |

## Throughput / resource envelope（吞吐与资源边界）

| Mode | Iter/s | Candidate tx/s | Screened routes/s | Exact/s | CPU % of one core | Cache MiB | Persistence median s | Vehicle median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_stage052 | 113.241 | 0 | 17059.5 | 46.5928 | 101.185 | 0.000 | 16.0956 | 3.5 |
| python_candidate_control | 169.755 | 0 | 752.3 | 20.6105 | 84.5486 | 0.000 | 6.50773 | 3.5 |
| per_solve_runtime | 178.359 | 2.38075 | 834.019 | 19.8252 | 85.7079 | 0.000 | 6.20123 | 3.5 |
| full_native_alns | 23302.1 | 0 | 0 | 72460.5 | 101.408 | 0.000 | 0.0782276 | 4 |
| host_scheduler | 19857 | 0 | 0 | 62299.5 | 0.988589 | 0.000 | 0.065515 | 4 |

## Correctness gates（正确性门控）

| Mode | Fixed-work differential | Performance qualified | 100-customer median improvement | Wall objective not worse |
|---|---:|---:|---:|---:|
| current_stage052 | baseline | baseline | n/a | n/a |
| python_candidate_control | baseline | False | None | True |
| per_solve_runtime | False | False | None | False |
| full_native_alns | False | False | None | False |
| host_scheduler | False | False | None | False |

## 实现与运维决策矩阵

下表是实现边界与故障域的事实/工程判断，不替用户选择生产路线。

| Mode | Python calls | Build complexity | Failure domain | Recovery difficulty | Maintenance cost |
|---|---|---|---|---|---|
| current_stage052 | 每批 candidate transaction | 中 | solve-local | 低 | 低 |
| python_candidate_control | 每轮 Python control + worker IPC | 低 | worker pool | 中 | 中 |
| per_solve_runtime | 每 candidate round 一次 C++ | 中 | solve-local runtime | 中 | 中 |
| full_native_alns | 每 instance/seed 一次 C++ | 高 | 单个 native solve | 高 | 高 |
| host_scheduler | shard 通过 UDS/shared memory | 最高 | run-wide scheduler | 最高 | 最高 |

## 当前实现与历史 Pilot

本轮 `current_stage052` 是所有速度比值和质量差值的主要分母。Accepted Pilot `attempt72` 仅用于长期漂移核验，不替代同机实测。

- Historical available（历史证据可用）：`True`
- Historical identity verified（历史身份已核验）：`True`
- Historical comparisons（历史配对数）：`36`
- CUDA condition（CUDA 条件）：`False`；native candidate-screening occupancy is not recorded；本轮未自动运行 CUDA。

## Instrumentation limitations（测量边界）

Exact batch occupancy（精确批量占用度）不是 native candidate-screening occupancy（原生候选筛选占用度），不用于 CUDA 门槛。`persistence_seconds` 包含 solver 执行时间，不是独立持久化成本。Host 轴的 CPU/RSS 仅测量 client shard，不含 scheduler service。

## 边界

未启动 Formal Rerun16，未 push，未清理或覆盖历史 evidence，未切换默认架构。
