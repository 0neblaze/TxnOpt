# Stage 3.0：精确充电调用的可审计测量与回放

Stage 3.0 只回答一个问题：在当前 `stage02_constraint_guided` 主轨迹中，时间究竟消耗在什么 exact charging（精确充电）调用上。它不改变求解器行为，也不宣称加速。

## 固定协议

| 项目 | 规定 |
| --- | --- |
| Objective（目标） | `(vehicle_count, total_distance, total_charging_time, charging_count)`，由 `evrptw.objective` 统一构造与比较 |
| Profile（算子配置） | `stage02_constraint_guided` |
| Time limit（时间上限） | 30 秒总预算；约束 lane 的既有 0.1 秒切片保持不变 |
| Iterations（迭代上限） | 1000 |
| Threads（线程） | 1 |
| Seeds（随机种子） | `2014, 2015, 2016` |
| Smoke scope（冒烟范围） | `c101C5`, `r105C5`, `rc105C5`, `c101_21`, `r101_21`, `rc101_21` |
| Formal scope（正式范围） | Stage 0 的 12 个 instance × 3 个 seed = 36 runs |
| Raw output（原始输出） | `results/` 下新的 run directory；失败证据不得覆盖 |
| Tracked summary（跟踪摘要） | 仅由 replay auditor（回放审计器）通过全部门槛后写入 `experiments/summaries/` |

历史 `stage02_constraint_guided_attempt16` 与 `stage02_constraint_guided_rerun09` 只作为 baseline provenance（基线溯源）。其中 `repository_dirty=true` 是历史事实，不能被改写成 clean。

## 数据字典

每个 run 至少包含以下 raw 文件：

- `solutions/<run>.json`：最终 route sequence（路线序列）和求解器给出的 objective key；审计器会重新运行统一 validator。
- `traces/<run>.json`：`Stage03Trace`。`route_dictionary` 以 canonical route key（规范路线键）保存客户序列；`route_evaluations` 保存 `exact_call`、`cache_hit`、`precomputed_route`、lane、iteration、operator、timing、feasibility、failure reason、label 统计，以及 `exact_started`/`exact_completed`。
- `events/<run>.jsonl`：trace event（trace 事件）与既有 `NeighborhoodEvent`（邻域事件）的逐行记录。
- `environments/<run>.json`：source/config/instance/environment hash、Python/package/native extension、hardware、peak memory、Stage 0 manifest hash，以及 `VRP-EVRP-Project-Hub` 和 `py-ga-VRPTW` 的 revision/dirty 状态。
- `raw/<run>.json`：原始 record、solver result、validator replay 线索和其他 raw 文件相对路径。
- `manifest.json` 与旁置的 `manifest.sha256`：raw 文件清单、SHA-256 和清单完整性锚点；auditor 会先验证二者。

`Stage03Trace` 的关键聚合字段如下：

| 字段 | 含义 |
| --- | --- |
| `started_calls` | exact solver 已开始的调用数 |
| `completed_calls` | exact solver 返回结果的调用数；可能小于 `started_calls` |
| `exact_calls` | `exact_call` route records 数 |
| `cache_hits` | evaluator lane 内命中既有 route cache 的次数；Stage 3.0 只测量，不新增 cache 策略 |
| `precomputed_routes` | 使用调用方已提供 route result 的次数 |
| `deadline_boundary` | deadline 前检查、exact call 完成后越过 lane boundary 等边界事件 |
| `candidate_state` | current/candidate route keys、objective keys、vehicle counts、accepted、global-best |

所有 summary CSV 都必须由 raw solution、raw trace、event log 和 manifest 重算。runner 写出的 `raw_per_run_results.csv` 不是审计结论。

内存记录使用 OS-level `peak_rss_bytes`（进程生命周期内的峰值常驻内存），不在 exact charging 求解路径启用 `tracemalloc`；后者会改变 100-customer fixed wall-clock trajectory（固定 wall-clock 轨迹）。因此 `peak_tracemalloc_bytes` 在 Stage 3.0 raw schema 中明确为空，而不是把高开销内存追踪伪装成 solver measurement（求解器测量）。

## 运行命令

运行命令必须从 clean main-repository commit（干净主仓库提交）开始；runner 会主动拒绝 dirty worktree（脏工作树）。

```bash
uv run python -m evrptw.experiments.stage03_measurement \
  --config configs/stage03_measurement.toml \
  --scope smoke \
  --run-label stage03_measurement_smoke01 \
  --output-dir results/stage03-measurement_smoke01

uv run python -m evrptw.experiments.stage03_measurement_review \
  --run-dir results/stage03-measurement_smoke01 \
  --summary-dir experiments/summaries \
  --review-label stage03_measurement_smoke01
```

只有 smoke review 输出 `READY_FOR_STAGE03_FORMAL_MEASUREMENT` 后，才允许启动正式 36-run：

```bash
uv run python -m evrptw.experiments.stage03_measurement \
  --config configs/stage03_measurement.toml \
  --scope formal \
  --run-label stage03_measurement_formal01 \
  --output-dir results/stage03-measurement_formal01 \
  --smoke-review-dir results/stage03-measurement_smoke01/review
```

正式 run 完成后再次使用 `stage03_measurement_review`。正式审计通过才报告已具备进入 Stage 3.1 的证据；不会把 Stage 3.0 报告写成 acceleration result（加速结果）。

## 保留规则与门槛

- 输出目录和 run label 已存在时直接失败；raw output 强制位于 ignored `results/`，tracked summaries 强制位于 `experiments/summaries/`；历史 Stage 2.3 目录与摘要不覆盖。
- 求解异常、deadline 中断和未完成 exact call 先写入 partial trace、event log、environment 和 raw record，再重新抛错；不可用半成品候选冒充 accepted。
- replay auditor 先验证 manifest SHA-256，再读取任何 raw 结论。篡改 solution、trace、event 或 manifest 文件都会 fail fast。
- smoke 必须覆盖 18 个唯一 run key；所有 solution 通过 validator，objective replay 一致，trace 与 `ALNSResult` 对账一致；C5 objective key 必须与 Stage 2.3 attempt16 对应 baseline 一致，100-customer runs 必须可行且 vehicle count 不得变差。
- 其他 wall-clock 差异统一标为 `time-budget variation`。fixed-work/wall-clock 双轴诊断留给 Stage 3.3。
- 36-run formal replay 通过后，才允许开始 Stage 3.1 的 cheap screening（廉价预筛）；Stage 3.0 本身不实现 3.1–3.4。
