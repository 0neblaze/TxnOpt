# 阶段 2.3 独立审查协议

## 1. 审查原则

审查 CLI 为只读审查入口：

```bash
uv run python -m evrptw.experiments.stage02_constraint_guided_review \
  --run-dir results/stage02-constraint-guided_attempt16 \
  --comparison-dir results/stage02-quality_attempt02 \
  --review-label stage02_constraint_guided_attempt16
```

审查工具不信任 solver summary（求解器汇总）作为唯一证据，而是重新读取 raw JSON、solution
JSON、operator event CSV、failure event CSV、environment metadata（环境元数据）和 run
manifest（运行清单）。它会重放 operator event CSV 的约束算子调用、可行候选、接受状态、
车辆变化、tier 和动态数量边界，并重新计算当前源码、配置、实例以及 Stage 0 冻结目录的
hash（哈希），而不是只检查文件是否存在；`runner_hard_gates` 也从 raw evidence 独立重算
目标、R/RC 门槛和质量算子门槛，不把 gate CSV 的 status 当作结论。比较目录缺失时，工具回退到固定的 tracked Stage 2.2 summary：
`experiments/summaries/stage02_quality_attempt02_per_run_results.csv`；这不会改变正式对照。
由于该历史 summary 未记录 Stage 2.2 的 cache/effective-iteration/tier 字段，审查使用同一
Stage 2.2 profile、同一 scope 和同一时间预算生成的 instrumented metric supplement：
`experiments/summaries/stage02_quality_attempt02_readiness_metrics.csv`。该 supplement 的
36 个 objective 已与固定正式对照逐行核对一致，不替换正式对照。

当 `repeatability.csv` 指向首轮完整运行时，独立复跑还会重新读取首轮的 parameters TOML、
environment metadata 和 operator events，分别重验 formal configuration、provenance
（包括 review CLI source hash）以及 cooperative time-budget exemption（协作式时间预算豁免），
不会只依赖首轮的 review manifest。

## 2. 必检项目

审查必须验证：

1. 36 个 `(instance, seed)` 键完整且唯一；
2. 每一份 raw solution 通过统一 validator；
3. validator 重新计算的 objective 与 solver objective 一致；
4. raw JSON、solution、operator event、failure event 和 environment/manifest 产物完整；
5. source/config/instance/environment hash（源码、配置、实例、环境哈希）一致；
6. review manifest 额外记录 `pyproject.toml`、`uv.lock`、native extension hash、review CLI
   source hash、package versions 和 Python metadata，避免只记录 native extension 路径；
7. Stage 0 manifest、冻结结果和 checksum 保持不变；
8. 失败和低质量候选具有真实的 constraint-level diagnosis（约束级诊断）；
9. Stage 2.2 与 Stage 2.3 的 readiness row（就绪记录）覆盖全部 36 个 runs。

每次审查在对应 run directory（运行目录）生成：

- `review_report.md`：人类可读的审查结论；
- `review_findings.csv`：逐项 gate 结果；
- `failure_analysis.csv`：真实失败/低质量候选及根因、下一步修复动作；
- `stage03_readiness.csv`：36 行双阶段对照记录；
- `review_manifest.json`：产物哈希、环境、版本和审查状态。

## 3. Failure analysis（失败分析）规则

`failure_analysis.csv` 不人工补造，也不删除不理想候选。每行至少包含 instance、seed、
operator、objective、feasibility、首次失败位置、约束触发、根因分类和下一步动作。根因分类
区分 `modeling`、`operator_logic`、`parameter` 和 `computational_limit`；当前成功轮真实记录
了 `time_window`、`operator_failure` 和 `computational_limit` 三类触发，共保留 20 个案例；
更早轮次出现的 `exact_infeasible` 记录仍保留在对应的失败轮次产物中。

## 4. Stage 3 readiness（就绪审查）字段

`stage03_readiness.csv` 对每个 `(instance, seed)` 同时记录：

- Stage 2.2 与 Stage 2.3 objective、feasibility、iterations 和 runtime；
- exact charging calls、cache hits/misses 和 unique route evaluations；
- removal tier counts；
- failure events、accepted moves；
- 四个约束算子的 acceptance/failure statistics；
- objective comparison 与 readiness status。

其中 Stage 2.2 的 instrumentation 字段必须来自上述 supplement；若回退为
`not_recorded_in_stage02_2_summary`，审查将失败。

Stage 3 的 acceleration targets（加速目标）只作为该文件和审查报告中的入口目标：100-customer
R/RC 的 median exact charging calls ≤ 100、median effective iterations ≥ 50、validator
feasibility 100%，且 objective 不退化。Stage 2.3 不把这些目标伪装成已完成的加速结果。

## 5. 审查结论

最终成功正式轮的审查产物：

- `results/stage02-constraint-guided_attempt16/`；
- `results/stage02-constraint-guided-rerun09/`。

复跑 `review_findings.csv` 的 18 个项目均为 `pass`；首轮仅将独立复跑相关的两个 finding
标记为 `pending`，其余项目均为 `pass`。两轮都重新验证 36/36 solution 和 objective，均保留
20 个真实 failure/poor-quality cases；最终状态以独立复跑的 `READY_FOR_STAGE03` 为准。对应 tracked summaries 位于
`experiments/summaries/stage02_constraint_guided_attempt16_*_review*`、
`stage02_constraint_guided_rerun09_*_review*`、`*_failure_analysis.csv` 和
`*_stage03_readiness.csv`。
