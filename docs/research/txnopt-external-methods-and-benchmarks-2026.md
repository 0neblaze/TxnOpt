# TxnOpt 外部方法与基准边界

审计日期：2026-08-14  
状态：研究背景，非 TxnOpt evidence（证据）

## 1. 目的

本文把旧 Windows clone（克隆）中仍有价值的算法和 benchmark（基准）判断改写为
TxnOpt 术语。它只定义外部比较与领域接入边界，不证明新颖性、最优性、BKS（最佳
已知解）改进或 Level 1 readiness（一级就绪）。原始迁移判定与来源核验见
[Windows salvage source audit](txnopt-windows-salvage-source-audit.md)。

## 2. 双层 CEVRP 方法能提供什么

三篇 CEVRP（容量约束电动车路径问题）工作共同采用 upper-level routing（上层路径
搜索）与 fixed-route charging problem（固定路线充电子问题）的双层分解：

- Jia、Mei 与 Zhang 的 bilevel ACO（双层蚁群优化）把路径搜索与充电计划评价分开；
- confidence-based ACO（基于置信度的蚁群优化）研究候选筛选与编码方式，以减少昂贵
  的下层评价；
- Feng 等人的 bilevel hybrid genetic algorithm（双层混合遗传算法）组合上层混合
  搜索与下层筛选/枚举。

这些工作可作为 TxnOpt 的外部方法参照：昂贵的确定性下层评价、候选的私有并行
完成、以及在唯一发布边界前进行筛选，都是可比较的工程结构。但它们不能直接充当
TxnOpt 的 safe-screening theorem（安全筛选定理）或 transactional publication
claim（事务发布主张），因为其模型、状态与证明义务不同。

## 3. 模型差异

CEVRP 文献与当前 TxnOpt EVRPTW adapter（适配器）至少有以下差异：

- CEVRP 没有当前 EVRPTW 的 customer time windows（客户时间窗）；
- 文献的目标函数不能替代 TxnOpt 当前四分量 lexicographic objective（字典序目标）；
- 论文自报的 7/8/11 个 BKS 更新尚未通过全文结果表与独立 route-level replay
  （路线级重放）核验；
- 因而不得把这些数值写入 Schneider registry（登记表），也不得改写成 TxnOpt
  speedup（加速比）或 semantic digest（语义摘要）结果。

## 4. 独立 benchmark lane（基准通道）

| 数据/仓库 | 可核验边界 | TxnOpt 接入要求 |
|---|---|---|
| WCCI-2020 CEVRP 17-instance protocol（17 实例协议） | 独立的 CEVRP 评估协议 | 新 domain descriptor（领域描述）、parser（解析器）、objective、validator（验证器）、manifest（清单）和许可记录 |
| Mavrovouniotis E-CVRP | 24 个约 21--1000 customers（客户）的实例；有电池/载重、重复充电站与 load-dependent energy（载重相关能耗） | 不复用 Schneider parser；`OPTIMAL_VALUE` 不能未经来源分级就称 proven optimum（已证明最优） |
| 2E-EVRP-Instances | 两级电动车路径数据，模型并非 Schneider EVRPTW | 单独 adapter、数据许可和 objective；第三方归档不进入 Apache-2.0 源码树 |
| E-VRP-HC | 带 heterogeneous charging（异构充电）的独立模型 | 单独 schema（模式）、validator 与 evidence identity（证据身份） |
| HEVRP-NL | heterogeneous EVRP with nonlinear charging（带非线性充电的异构 EVRP） | 单独非线性充电语义、数值容差与 replay contract（重放契约） |

任何新 benchmark lane 都必须先固定数据版本、逐文件 SHA-256、许可、案例身份和目标
定义，再进入 `txnopt_cases`。动态下载、隐式模型转换和把论文 heuristic BKS（启发式
最佳已知解）当作 certified optimum（经证明最优）均 fail closed（失败即停止）。

## 5. 对当前 TxnOpt 的实际含义

当前最有价值的迁移结论不是“直接加入更多实例”，而是保持三个 seam（接缝）：

1. runtime（运行时）只处理事务、预算、cache（缓存）与发布；
2. 每个领域 adapter 独占 parser、candidate identity（候选身份）、objective 与 validator；
3. evidence workflow（证据工作流）按独立协议固定 build、case、config（配置）、raw 与
   reviewer identity（审查者身份）。

这使外部方法能够作为新 adapter 或 baseline（基线）接入，而不会污染已冻结的
Build14、Attempt24、Attempt25 或 Attempt07。

## 6. 直接来源

- Jia et al., BACO：[DOI](https://doi.org/10.1109/TCYB.2021.3069942)。
- Jia et al., confidence-based ACO：[DOI](https://doi.org/10.1109/TEVC.2022.3144142)。
- Feng et al., BHGA：[DOI](https://doi.org/10.1109/CEC60901.2024.10611987)。
- WCCI-2020 CEVRP：[technical report](https://mavrovouniotis.github.io/Papers/TR-EVRP-Competition.pdf)。
- Mavrovouniotis E-CVRP：[CEC-2020 paper](https://mavrovouniotis.github.io/Papers/CEC20.pdf)、[fixed repository snapshot](https://github.com/Mavrovouniotis/e-cvrp_benchmark_instances/tree/a3f59afb6ddc3999961060ca7035a08fa1f7d59c)。
- 2E-EVRP：[data article](https://doi.org/10.1016/j.dib.2025.111470)、[Zenodo v2](https://zenodo.org/records/14844216)。
- E-VRP-HC：[DOI](https://doi.org/10.1016/j.cor.2025.107374)。
- HEVRP-NL：[DOI](https://doi.org/10.1016/j.trc.2024.104932)、[GERAD report](https://www.gerad.ca/en/papers/G-2024-01)。
