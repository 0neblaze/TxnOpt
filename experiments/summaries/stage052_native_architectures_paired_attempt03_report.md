# Stage 5.2 五种原生架构同机对比报告

- Review status（审查状态）：`NOT_READY`
- Raw replay（原始重放）：`False`
- Axis count（轴数）：`360`
- Formal：未启动；production default（生产默认值）：未改变。

## 五模式事实表

| Mode | Solver median s | Effective iterations median | Exact calls median | Queue wait median s | Exact batch occupancy median | RSS median MiB | Artifact median MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| current_stage052 | 0.960764 | 158.5 | 100 | n/a | 1 | 508.543 | 0.017 |
| python_candidate_control | 31.4343 | 139 | 55 | n/a | 1 | 430.889 | 0.017 |
| per_solve_runtime | 26.1893 | 139 | 55 | 0 | 1 | 471.680 | 0.019 |
| full_native_alns | 2.11244 | 1000 | 43740 | 0.00348676 | 45 | 412.168 | 0.013 |
| host_scheduler | 2.19344 | 1000 | 43740 | 0.00849072 | 45 | 373.711 | 0.013 |

## Throughput / resource envelope（吞吐与资源边界）

| Mode | Iter/s | Candidate tx/s | Screened routes/s | Exact/s | CPU % of one core | Cache MiB | Persistence median s | Vehicle median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_stage052 | 5.27034 | 0.0664701 | 15467.4 | 55.8751 | 100.089 | 0.000 | 1.69784 | 15.5 |
| python_candidate_control | 4.53503 | 0 | 1118.14 | 2.34841 | 93.0228 | 0.000 | 33.8673 | 16 |
| per_solve_runtime | 4.52071 | 1.09336 | 912.706 | 2.40649 | 92.4507 | 0.000 | 29.6316 | 12 |
| full_native_alns | 473.386 | 0 | 0 | 20601.7 | 100.06 | 0.000 | 2.15436 | 45 |
| host_scheduler | 455.904 | 0 | 0 | 19598.9 | 0.717459 | 0.000 | 2.21227 | 45 |

## Correctness gates（正确性门控）

| Mode | Fixed-work differential | Performance qualified | 100-customer median improvement | Wall objective not worse |
|---|---:|---:|---:|---:|
| current_stage052 | baseline | baseline | n/a | n/a |
| python_candidate_control | baseline | False | -0.9802039406799221 | True |
| per_solve_runtime | False | False | -0.9668438202376723 | False |
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

- Historical available（历史证据可用）：`False`
- Historical identity verified（历史身份已核验）：`False`
- Historical comparisons（历史配对数）：`0`
- CUDA condition（CUDA 条件）：`False`；native candidate-screening occupancy is not recorded；本轮未自动运行 CUDA。

## Instrumentation limitations（测量边界）

Exact batch occupancy（精确批量占用度）不是 native candidate-screening occupancy（原生候选筛选占用度），不用于 CUDA 门槛。`persistence_seconds` 包含 solver 执行时间，不是独立持久化成本。Host 轴的 CPU/RSS 仅测量 client shard，不含 scheduler service。

## 边界

未启动 Formal Rerun16，未 push，未清理或覆盖历史 evidence，未切换默认架构。
