# TxnOpt 发表缺口审计

审计日期：2026-08-14  
状态：NOT_READY（未就绪）

## 1. 中心研究问题

TxnOpt 的正面论文不能只主张“使用事务化 speculative parallelism（推测并行）”。可
检验的中心问题应是：在状态依赖、确定性的组合搜索中，能否通过预先固定的事务发布
语义，在保持 objective（目标）、最终状态与 semantic digest（语义摘要）等价的同时，
获得跨 EVRPTW 与 RCPSP 的可重复 wall-clock improvement（实际耗时改进），并把失败
与未提交工作限制在已审查的 T4 bound（T4 界）内？

这个问题同时需要方法主张、形式边界、实现和外部实验；任何单项通过都不能代替其余
项目。

## 2. 当前已具备与仍缺失的证据

| 维度 | 当前边界 | 发表缺口 |
|---|---|---|
| 核心语义 | 五个公开接口、事务状态机、独立 runner/reviewer（运行器/审查器）和 expected identity（预期身份）已有测试与冻结审查 | 新重构 source identity（源码身份）必须重新构建并复核，不能沿用 Build14 结论 |
| T3/T4 | 条件性 meta-theorem（元定理）、bounded model checking（有界模型检查）、单位工作界与 Cmax（最大单元成本）审查已形成 | 不能把有界 TLC 收据表述成一般领域 liveness（活性）证明；新执行仍需逐条 T4/Cmax 证据 |
| 代表性本地性能 | Attempt25 的 EVRPTW/RCPSP 本地校准通过其代表性门 | 样本小、不是正式矩阵，不提供预注册 95% CI（置信区间）或 Level 1 正式结论 |
| 正式矩阵 | Attempt24 的 2,880 个 config（配置）与 expected identity 已冻结 | raw root 未启动；没有独立授权、正式执行、全范围 Cmax、CI 或最终 report（报告） |
| 外部方法比较 | 已定位双层 CEVRP 方法和独立 benchmark lanes（基准通道） | 还没有同模型、同预算、同硬件的外部 baseline（外部基线）与 route-level replay（路线级重放） |
| 新颖性 | T3 的 scheduler-relative boundary（相对调度器边界）与 T4 的可测工作界提供了可证伪结构 | 仍需系统文献审计、明确最接近工作差异，并避免把工程封装本身当新颖性 |
| 可复现性 | Build14/Attempt25/Attempt07、生命周期、sidecar（哈希旁文件）与独立 review 已冻结 | Build portability（构建可移植性）未验证；需要第二主机 fresh build/restore（全新构建/恢复）与公开最小复现实验 |

## 3. 正面主张的最低门槛

在下列项目全部通过前，不应把 TxnOpt 表述为 Level 1 完成或论文结果：

1. 新 successor build（后继构建）通过 fresh-wheel tests（全新轮子测试）、Ruff、strict
   mypy（严格静态类型检查）、ASan/TSan（地址/线程消毒器）和形式契约；
2. 独立 reviewer 对 runner 无 import（导入）依赖，并重算 semantic parity（语义一致
   性）、objective、identity、lifecycle（生命周期）、T4 与 Cmax；
3. Attempt24 或其明确 successor 的完整预注册矩阵在单独授权后运行，且两个领域的
   geomean（几何平均）和 95% CI 下界同时满足锁定阈值；
4. 1-worker overhead（单工作线程开销）、fallback count（回退次数）、fault prefix
   safety（故障前缀安全）和 measured waste（实测浪费）均满足协议；
5. 至少一个与 TxnOpt 同模型、同目标、同预算的外部 baseline 被独立复跑；
6. 在干净第二主机完成 build/restore/review，并发布最小的 source、config、raw、review
   和 report identity chain（身份链）。

任何 critical/major finding（严重/主要问题）、性能阈值失败或新颖性审计失败都必须保留
工程成果，但暂停正面 Q1 方法论文路线，转向优化、边界或负结果。

## 4. 期刊口径

Transportation Science 与 INFORMS Journal on Computing 的 editorial statement
（编辑声明）都要求超越实现质量的研究贡献。对 TxnOpt 而言，代码整洁、可重放、云端
可运行或速度更快本身都不足够；必须把可一般化的机制、适用域、反例和统计证据写成
一致的论证。

- [Transportation Science editorial statement](https://pubsonline.informs.org/page/trsc/editorial-statement)
- [INFORMS Journal on Computing editorial statement](https://pubsonline.informs.org/page/ijoc/editorial-statement)

## 5. 当前结论

Build14、Attempt25 与 Attempt07 支持继续进行独立采购授权评估，但不授权采购，也不
等于正式矩阵开始。Attempt24 仍未开始；Level 1、publication（发表）、public release
（公开发布）与 Level 2 正面入口均保持关闭。本文是研究决策边界，不重写任何冻结
manifest（清单）或审查收据。
