# 阶段 0：冻结当前 ALNS Baseline

## 1. 目的与边界

阶段 0 将当前 `ALNS_EXACT_CHARGING` 固定为后续改进的 comparison point（比较起点）。
本阶段不修改 ALNS、精确充电子问题、目标函数或算子，也不引入 best-known values
（最佳已知值）。GA、OR-Tools、stress benchmark（压力测试）和 BPC 不属于本次冻结运行。

固定实验包含 C/R/RC 三类、5/10/15/100 customers（客户）共 12 个 Schneider 实例；
每个实例使用 seeds（随机种子）`2014/2015/2016`，每次最多 1000 次迭代、30 秒、
单线程。唯一参数来源是 `configs/stage00_baseline.toml`。

## 2. 数据记录规范

逐次记录包括：可行性、车辆数、距离、能耗、充电量、充电时间、运行时间、迭代数、
接受/改善/拒绝次数、exact charging subproblem（精确充电子问题）调用数和平均耗时，
以及 coverage/capacity/time-window/energy/objective（覆盖/容量/时间窗/能量/目标值）
等结构化违约计数。每个 solution（解）必须重新通过统一 `validate_routes()`。

环境记录包括 Git revision（版本）、dirty state（工作区状态）、核心算法源文件 SHA-256、
实例 SHA-256、配置 SHA-256、参考仓库版本、Python/操作系统/CPU/内存和依赖版本。
`summary_results.csv` 从逐次记录自动计算 Best/Mean/Median/Worst/Population standard
deviation（最佳值/均值/中位数/最差值/总体标准差）。即使无失败，
`failure_cases.csv` 仍必须存在并保留表头。

完整 raw JSON（原始记录）保存在 Git-ignored（Git 忽略）的 `results/stage00/`；可审阅的
CSV、环境、配置快照、经过验证的 solution JSON 和 checksum manifest（校验清单）保存在
`experiments/baselines/stage00/`。正式基准目录不允许原地覆盖，任何文件变化都会被
manifest 检测。

## 3. 正式生成与验证

在干净工作区运行：

```bash
uv run python -m evrptw.experiments.stage00_baseline run \
  --config configs/stage00_baseline.toml \
  --output-dir results/stage00 \
  --baseline-dir experiments/baselines/stage00
```

重新验证冻结产物：

```bash
uv run python -m evrptw.experiments.stage00_baseline verify \
  --config configs/stage00_baseline.toml \
  --results-dir experiments/baselines/stage00 \
  --require-manifest
```

验证会拒绝缺失或重复的 `(instance, seed)`、被篡改的文件、不可重算的汇总、被删除的
失败记录、validator（验证器）不通过的解，以及 CSV 与重算指标之间的差异。

## 4. 后续版本比较

候选版本先使用同一配置运行到新的目录（不传 `--baseline-dir`），再执行：

```bash
uv run python -m evrptw.experiments.stage00_baseline compare \
  --config configs/stage00_baseline.toml \
  --baseline-dir experiments/baselines/stage00 \
  --candidate-dir results/stage00-candidate \
  --report results/stage00-candidate/comparison.csv
```

报告逐指标标记 `improvement`、`regression` 或 `unchanged`。可行率下降、validator 失败、
缺少实例/seed 或删除失败记录会触发 regression gate（回归门槛）失败；其余指标只分别
分类，不提前定义跨指标总排名。正式 lexicographic objective（字典序目标）留到阶段 1。

## 5. 冻结时的 100-customer 检查点

路线图要求复核下列当前车辆数范围：

| Instance | Stage 0 vehicle count |
| --- | ---: |
| `c101_21` | 14 |
| `r101_21` | 22–28 |
| `rc101_21` | 24–25 |

最终数值以冻结 CSV 和 manifest 为唯一事实来源。
