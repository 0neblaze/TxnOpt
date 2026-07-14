# Stage 3.3：Exact solver interface 与 fixed-work / wall-clock 诊断

Stage 3.3 在 Stage 3.2 已审计的 screening、bounded LRU route cache、station
reachability bitset 和 incremental propagation 之上增加可检查的 exact charging
checkpoint。它不增加候选并行，也不以 `cpu_scalar` 重新做性能对照。

## 固定协议

| 项目 | 值 |
| --- | --- |
| Component | `exact_deadline` |
| Backend | `cpu_batch` |
| Fixed-work | 全局最多 100 个 started exact calls |
| Fixed-work watchdog | 120 秒 |
| Wall-clock | 30 秒 |
| Iterations | 1000 |
| Threads | 1 |
| Smoke | 6 instances × 3 seeds × 2 axes |
| Formal | 12 instances × 3 seeds × 2 axes |

预算与 deadline 都在候选提交边界执行 transactional semantics（事务语义）。若
exact batch 只完成了一部分，已启动与已完成工作仍进入 trace，但候选、cache store、
acceptance 和 global best 更新全部不提交。求解器最终返回最近一个完整 incumbent。

## 运行与审查

```bash
uv run python -m evrptw.experiments.stage033_exact_deadline \
  --config configs/stage033_exact_deadline.toml \
  --scope smoke \
  --run-label stage03.3_exact_deadline_attempt01 \
  --output-dir results/stage03.3_exact_deadline_attempt01

uv run python -m evrptw.experiments.stage033_exact_deadline_review \
  --run-dir results/stage03.3_exact_deadline_attempt01 \
  --scope smoke \
  --benchmark-dir data/schneider \
  --output-dir results/stage03.3_exact_deadline_attempt01/review
```

Smoke review 必须报告 `READY_FOR_STAGE033_FORMAL`。Formal runner 另外接收
`--smoke-review-dir`，并在 36 个 instance/seed pair 的两个轴全部完成后交给同一
reviewer。只有 reviewer 从 raw solution、Parquet events 和 manifest 独立重算并报告
`READY_FOR_STAGE03_4`，才算 Stage 3.3 完成。

Stage 3 的 `median exact charging calls <= 100` 与
`median effective iterations >= 50` 仍需在 formal report 中如实展示；Stage 3.3
不通过改时间预算或排除运行来宣称这些目标已经达到。
