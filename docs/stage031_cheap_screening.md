# 阶段 3.1：Cheap Screening（低成本预筛选）

Stage 3.1 只在已审计的 Stage 3.0 formal evidence（正式证据）上增加
safe cheap screening（安全低成本预筛选）。它的唯一求解器目的，是在进入
exact charging（精确充电）前安全地排除不可能可行的 customer sequence（客户
序列）；它不改变 lexicographic objective（字典序目标）、统一 validator
（验证器）、vehicle-first acceptance（车辆数优先接受）或 Stage 0--3.0 的默认轨迹。

## 固定协议

| 项目 | 规定 |
| --- | --- |
| Profile | `stage02_constraint_guided` |
| Time limit | 30 秒总预算；既有 0.1 秒 constraint lane slice 保持不变 |
| Iterations | 1000 |
| Threads | 1 |
| Seeds | `2014, 2015, 2016` |
| Smoke | 6 个 instance × 3 个 seed = 18 runs |
| Formal | Stage 0 的 12 个 instance × 3 个 seed = 36 runs |
| Raw output | 新的 ignored `results/<run-label>/` |
| Tracked summary | replay audit 通过后才写入 `experiments/summaries/` |

正式 Stage 3.1 启动前必须同时满足：

1. Stage 3.1 smoke review 报告 `READY_FOR_STAGE031_FORMAL_MEASUREMENT`；
2. Stage 3.0 formal raw manifest、sidecar 和 trusted review manifest（受信审计清单）
   完整且未篡改；
3. 主仓库处于 clean commit；reference repositories（参考仓库）保持只读。

## Screening data dictionary（预筛选数据字典）

`CheapScreeningConfig` 的关键字段：

- `enabled`：是否启用 Stage 3.1；未启用时保持历史求解器路径；
- `negative_sequence_cache`：是否启用 process-local known-infeasible sequence cache；
- `epsilon`：安全数值比较容差；
- `schema_version`：当前为 `stage031-screening-v1`。

每个 screening decision（预筛选决策）记录：

- canonical `route_key`、lane、iteration、operator；
- `status`：`pass`、`rejected` 或 `negative_cache_hit`；
- `checks`：route structure、capacity、forward/backward time-window、slack、
  shortest-distance lower bound、single-segment battery reachability、structural
  energy lower bound 的 status/value/reason；
- `first_failed_check`、failure reason、demand、minimum slack；
- distance lower bound 和 optional distance increment；
- `negative_cache_hit`、`exact_call_blocked`、开始/结束/耗时。

其中 shortest-distance lower bound 只能作为安全记录和排序指标，不能造成拒绝。
只有 `pass` 才允许访问既有 exact route cache 或 exact charging；`rejected` 和
`negative_cache_hit` 均不增加 `exact_call`、`cache_hit` 或 `started_calls`。

Stage 3.1 trace 仍使用 Stage 3.0 的 route dictionary（路线字典）去重完整客户
序列。新增 counters 必须与 `ALNSResult.screening_statistics` 对账，聚合 CSV 必须
从 raw trace、raw solution、event log 和 manifest 重算。

## 运行命令

```bash
uv run python -m evrptw.experiments.stage031_cheap_screening \
  --config configs/stage031_cheap_screening.toml \
  --scope smoke \
  --run-label stage031_cheap_screening_smoke01 \
  --output-dir results/stage031-cheap-screening_smoke01

uv run python -m evrptw.experiments.stage031_cheap_screening_review \
  --run-dir results/stage031-cheap-screening_smoke01 \
  --summary-dir experiments/summaries \
  --review-label stage031_cheap_screening_smoke01
```

Smoke replay 通过后运行 formal：

```bash
uv run python -m evrptw.experiments.stage031_cheap_screening \
  --config configs/stage031_cheap_screening.toml \
  --scope formal \
  --run-label stage031_cheap_screening_formal01 \
  --output-dir results/stage031-cheap-screening_formal01 \
  --smoke-review-dir results/stage031-cheap-screening_smoke01/review

uv run python -m evrptw.experiments.stage031_cheap_screening_review \
  --run-dir results/stage031-cheap-screening_formal01 \
  --summary-dir experiments/summaries \
  --review-label stage031_cheap_screening_formal01
```

## Failure retention and review gates

- Existing output directory or run label 直接失败；历史 baseline、Stage 3.0
  evidence、reference/ 和外部路线图不覆盖。
- 异常、deadline overrun、未完成 exact call 和 partial trace 先保留 raw
  solution/trace/event/environment/manifest，再暴露错误；半成品候选不得成为
  accepted candidate。
- Auditor 先验 manifest，再 replay validator/objective、screening decision、
  exact/cache ordering、event log、candidate state、deadline 和 provenance。
  任意 raw solution、trace、event、manifest 或 hash 被篡改都必须 fail fast。
- `screening_reason_statistics.csv` 和 `stage03_formal_comparison.csv` 只由
  raw replay 生成。C5 objective key 与 Stage 3.0 formal evidence 不得劣化；
  100-customer runs 必须可行且 vehicle count 不得增加；exact-call reduction
  可以报告，但不能宣称 Stage 3.3 fixed-work/wall-clock acceleration。
- formal review 通过后必须输出 `READY_FOR_STAGE03_2`，这才是进入 Stage 3.2
  的唯一门槛。
