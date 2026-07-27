# EVRPTW 发表前沿与 Q1/Q2 门槛调研（2023--2026）

调研日期：2026-07-27

## 1. 调研边界与分区口径

本文只回答外部文献与期刊门槛问题，不评价本仓库当前代码、实验是否已经达到这些门槛。

“一区/二区”存在多种互不等价的口径，包括 JCR quartile（JCR 分区）、SJR quartile（SJR 分区）和中国科学院期刊分区。本文能公开复核的分区统一采用 **2024 SJR** 或期刊官网公开的 **2024 JCR**；不能把它直接等同于中国科学院分区。投稿前应按学校当年的正式考核口径再次查询。

## 2. 外部结论

### 2.1 领域已经相当拥挤

单纯实现一个可运行的 EVRPTW solver（电动车时间窗路径求解器）、增加若干 destroy/repair operators（破坏/修复算子）、引入 adaptive weights（自适应权重），或者把已有候选评价并行化，通常不足以构成 Q1/Q2 论文的核心贡献。

直接证据是 Voigt 对 2006--2023 年文献的系统研究：它分析了 **211 篇**采用 ALNS 的 VRP 论文，统一识别出 **57 种 removal operators（移除算子）和 42 种 insertion operators（插入算子）**；sequence-based removal（基于序列的移除）和具有 foresight（前瞻性）的 regret insertion（后悔值插入）已经是明确的高性能设计方向。因此，“我们也实现了 ALNS 并加入若干常见算子”不是稀缺贡献。[EJOR 论文原文](https://doi.org/10.1016/j.ejor.2024.05.033)

### 2.2 近年的强论文通常同时占据至少两个贡献维度

常见组合如下：

1. **新问题机制 + 新算法**：如 time-dependent travel and discharge（时变行驶与放电）、heterogeneous nonlinear charging（异质非线性充电）、capacitated charging stations（有容量限制的充电站）。
2. **新算法机制 + 可证明/可复核性质**：如新的 bidirectional labeling（双向标号）、dominance rules（支配规则）、branching（分支策略）、safe preprocessing（安全预处理）。
3. **新算法 + 强计算证据**：完整公开 benchmark（基准）、exact comparison（精确算法对照）、多规模实例、ablation（消融）、统计推断和新 BKS/optimal solutions（最佳已知解/最优解）。
4. **软件/数据贡献 + 长期复现价值**：不是只上传代码，而是稳定 API、文档、测试、许可、固定版本快照和可重跑结果。

### 2.3 “工程量大”与“学术新颖性强”是两件不同的事

高质量日志、manifest（清单）、hash（哈希）、独立 replay（重放）、deadline safety（截止期安全性）和并行一致性会显著增强可信度，但审稿人仍会追问：

- 新的科学问题是什么？
- 与最近最强方法相比，新方法为什么有效？
- 贡献能否迁移到当前仓库以外的问题或 solver？
- 是获得更好的解、更快的等价解，还是带来新的建模/管理洞见？
- 性能提升是否在相同 objective（目标）、相同工作量和相同硬件约束下成立？

这些问题必须由论文的研究设计回答，不能由工程证据数量代替。

## 3. 2023--2026 年代表性前沿

| 年份 | 论文与 venue（期刊） | 核心创新 | 数据规模与对照 | 对投稿门槛的启示 |
|---|---|---|---|---|
| 2023 | Bruglieri, Paolucci & Pisacane, *Computers & Operations Research*, [DOI](https://doi.org/10.1016/j.cor.2023.106261)，[作者机构公开稿](https://re.public.polimi.it/bitstream/11311/1260037/1/Bruglieri_Paolucci_Pisacane_CAOR2023_Matheuristic_EVRP.pdf) | 首次将 speed（速度）和 load（载重）共同进入 energy consumption rate（能耗率）；提出 cloneless MILP（无充电站克隆的混合整数模型）与 Random Kernel Search matheuristic（随机核搜索数学启发式） | Schneider/Solomon 派生实例；12 个 25-customer 与 30 个 100-customer 实例；MILP vs matheuristic；每个随机实例 **15 次运行**，用 90% confidence（置信度）和 0.1% margin of error（误差界）确定样本量，并使用 Friedman test（Friedman 检验） | 强论文把现实能耗模型、数学模型、算法、统计设计放在同一贡献链中 |
| 2023 | Rastani & Çatay, *Annals of Operations Research*, [DOI](https://doi.org/10.1007/s10479-021-04320-9)，[机构条目](https://research.sabanciuniv.edu/id/eprint/47761/) | Load-dependent EVRPTW（载重依赖 EVRPTW）；两个数学模型；把 optimal repair（最优修复）嵌入 LNS matheuristic | 小实例由 commercial solver（商业求解器）验证，大实例用于 LNS；进一步研究忽略载重对 fleet size（车队规模）与路线可行性的影响 | 算法结果之外还要回答模型因素带来的决策影响 |
| 2024 | Lera-Romero, Miranda-Bront & Soulignac, *European Journal of Operational Research*, [DOI/原文页](https://doi.org/10.1016/j.ejor.2023.06.037)，[公开预印本](https://optimization-online.org/wp-content/uploads/2020/09/8026.pdf) | 同时建模 time-dependent travel time（时变行驶时间）和 speed-dependent battery consumption（速度依赖电耗）；统一可变等待/充电时间；新的 BCP、piecewise-linear resource labels（分段线性资源标签）、partial dominance（部分支配）与预处理 | 扩展 Desaulniers/Schneider benchmarks；最多 100 customers；报告 **13 个新的最优解**；分析忽略时变速度导致的不可行性，其中某些场景最多 40% 来自超出电池容量 | Q1 方法论文通常既扩展模型，又对 exact algorithm（精确算法）本身作出实质推进 |
| 2024 | Klein & Schiffer, *INFORMS Journal on Computing*, [论文](https://doi.org/10.1287/ijoc.2023.0104)，[开发仓库](https://github.com/tumBAIS/RoutingBlocks)，[期刊固定快照](https://github.com/INFORMSJoC/2023.0104) | RoutingBlocks：面向带 intermediate stops（中间停靠点）VRP 的模块化 Python/C++ 算法框架；提供 ALNS、local search（局部搜索）、move cache（移动缓存）和 EVRPTW partial-recharge 示例 | 软件、数据、结果、文档、许可证和固定 SHA 快照公开；期刊仓库单独赋 DOI | 软件论文的竞争基准不是“代码可用”，而是“可复用、可维护、可验证且具有社区价值” |
| 2024 | Zhou et al., *Journal of Cleaner Production*, [DOI/原文页](https://doi.org/10.1016/j.jclepro.2023.140188) | Time-dependent EV routing and scheduling；MILP + VNS with partial model（带部分模型的变邻域搜索），允许在拥堵时段于节点或弧上调度等待 | 大实例达到 **200 customers**；56 个相关 EVRPTW benchmarks 中报告 **11 个新 BKS**；另有 case study（案例研究）与 sensitivity analysis（敏感性分析） | 仅在 100-customer 小范围上复现实验已不是规模优势；实际洞见可以显著提高 transportation/sustainability venue 的适配度 |
| 2025 | Voigt, *European Journal of Operational Research*, [DOI/原文](https://doi.org/10.1016/j.ejor.2024.05.033) | 对 211 篇 ALNS-VRP 论文做 network meta-analysis（网络元分析）；形成算子分类、排名和未来实验指南 | 57 种移除、42 种插入算子；sequence removal 与 regret-style insertion 排名靠前 | 新 ALNS 论文必须用消融和统计证据证明 problem-specific operators（问题特定算子）的独立价值 |
| 2025 | Nafstad, Desaulniers & Stålhane, *Transportation Science*, [论文与开放原文](https://doi.org/10.1287/trsc.2024.0725) | E-VRPTW with heterogeneous technologies and nonlinear charging（异质技术与非线性充电）；把完整 time--SoC trade-off（时间--电量权衡）嵌入 exact pricing（精确定价）；双向标号优于单向标号 | 新 benchmarks 为 25/50/100 customers、21 stations、三种充电技术；对照既有 E-VRPTW-L 与 E-VRP-NL exact methods；最多 100 customers 可在一小时内求解，E-VRP-NL 对照中新增 24 个可证最优实例 | “exact charging（精确充电）”前沿已经达到异质、非线性、双向标号与完整 BCP；线性/full-recharge route evaluator 本身难成为建模创新 |
| 2025 | Bruglieri et al., *International Transactions in Operational Research*, [DOI/原文页](https://doi.org/10.1111/itor.70104) | EVRPTW with capacitated recharging stations；partial recharge（部分充电）；MILP-based ALNS；同时处理 linear 与 piecewise-linear charging | 与 state-of-the-art exact algorithm（最先进精确算法）在公开 benchmark 上比较 | 充电站资源冲突与部分充电已经进入近期 matheuristic 的直接对照范围 |
| 2026 | Rastani et al., *European Journal of Operational Research*, [DOI](https://doi.org/10.1016/j.ejor.2026.01.020)，[机构摘要](https://research.sabanciuniv.edu/id/eprint/53686/) | 同时处理 road gradients（道路坡度）、load dynamics（载重动态）和 regenerative braking（再生制动）；四个数学模型 | 基于 benchmarks 生成带 elevation（海拔）的新数据集，比较三类地形并分析可行性与路线变化 | 现实能耗研究正从固定距离能耗继续推进到坡度、载重和能量回收 |
| 2026 | Bacci, Gentile & Pizzari, *Optimization Letters*, [开放原文](https://doi.org/10.1007/s11590-026-02319-4) | 对 nonlinear charging（非线性充电）提出 exact perspective-cut linearization（精确透视割线性化），并预处理 dominated charging paths（被支配充电路径） | 与既有 piecewise-linear/path formulations（分段线性/路径模型）比较 | 即使投更聚焦的 optimization venue，仍需要清晰的数学方法创新，不只是实现优化 |

## 4. 近期工作的共同实验标准

以下不是所有期刊的明文“最低条款”，而是代表性论文体现出的事实标准。

### 4.1 Benchmark coverage（基准覆盖）

- 小实例用于 exact validation（精确验证）：MILP/CPLEX/Gurobi 或 exact algorithm 应能给出 optimum/bound（最优值/界）。
- 大实例用于 heuristic performance（启发式性能）：不能只报告一个家族或少量方便实例。
- Schneider/Solomon 衍生的 5/10/15/25/50/100-customer 实例仍常用，但 2024 年论文已经做到 200 customers；更大规模并非自动创新，却是算法可扩展性的必要证据之一。[Zhou et al.](https://doi.org/10.1016/j.jclepro.2023.140188)
- 若 objective 或 charging model 与原 benchmark 不同，必须明确阻止不兼容的 gap（差距）比较，或建立经过验证的新 baseline；不能把不同 objective 的数值直接当作进步。

### 4.2 Baselines（对照）

一篇以求解算法为主要贡献的论文通常至少需要三类对照：

1. 原始/经典 published baseline（发表基线）；
2. 近期 state of the art（最先进方法）或公开实现；
3. 自身方法的 controlled ablations（受控消融），例如：
   - adaptive vs fixed weights；
   - new operator on/off；
   - cache/screening/batching/parallel 各自 on/off；
   - serial vs parallel 在相同 candidate order（候选顺序）和相同 work budget（工作量预算）下比较。

RoutingBlocks 已公开高性能 Python/C++ ALNS 组件与 EVRPTW partial-recharge 示例，因此它至少应被讨论；若问题定义兼容，应作为实现/效率参照之一。[论文](https://doi.org/10.1287/ijoc.2023.0104)；[代码](https://github.com/tumBAIS/RoutingBlocks)

### 4.3 Randomized evaluation（随机算法评价）

- 三个 seed（随机种子）适合 regression/evidence gate（回归/证据门禁），但通常不足以稳定估计 stochastic metaheuristic（随机元启发式）的均值、方差和尾部表现。
- 代表性 COR 论文根据 90% confidence 和 0.1% margin of error 计算样本量，最终每实例运行 15 次，并使用 Friedman non-parametric test。[公开论文，第 6 节](https://re.public.polimi.it/bitstream/11311/1260037/1/Bruglieri_Paolucci_Pisacane_CAOR2023_Matheuristic_EVRP.pdf)
- 实际论文应预先写明 sampling unit（抽样单位）是 instance、instance-seed 还是 family；报告 median/mean、dispersion（离散程度）、confidence interval（置信区间）、effect size（效应量）和 paired test（配对检验），而不是只报“赢了多少次”。
- 多算法/多参数比较需要 multiple-comparison control（多重比较控制）或整体检验；不要从大量 seed/instance 中事后挑最好结果。

### 4.4 Performance protocol（性能协议）

- Wall-clock（墙钟时间）比较必须记录 CPU、线程、内存、编译器、solver 版本。
- 并行算法应同时报告：
  - fixed-work（固定工作量）质量；
  - fixed-time（固定时间）质量；
  - speedup（加速比）与 parallel efficiency（并行效率）；
  - 1/2/4/8 worker scaling（工作进程扩展），如果硬件允许；
  - deterministic equivalence（确定性等价）或并行引入的随机差异。
- 不能把更强硬件、更多 exact calls（精确调用）、提前写入 cache（缓存）或不同 deadline semantics（截止语义）产生的优势归因于算法。
- “四 worker 更快”只是结果；可发表的贡献需要进一步解释可迁移的 batching/scheduling/transaction design（批处理/调度/事务设计），最好给出复杂度、正确性不变量或足够一般的算法框架。

### 4.5 Reproducibility（可复现性）

IJOC 明确要求 software/data archive（软件/数据归档），其 Software Tools area（软件工具方向）还会分别审查软件质量/可维护性与论文新颖性/社区价值。[IJOC editorial statement](https://pubsonline.informs.org/page/ijoc/editorial-statement)；[submission guidelines](https://pubsonline.informs.org/page/ijoc/submission-guidelines)

一个有竞争力的复现包至少应包含：

- 固定 commit/tag 与开源许可证；
- 可从原始实例一键生成表格/图的命令；
- 环境 lockfile/container（锁文件/容器）；
- 原始逐运行结果，而不只是汇总 CSV；
- validator（验证器）与 objective recomputation（目标重算）；
- seed、time/work budget、硬件、线程和 solver metadata（元数据）；
- 失败/timeout（超时）运行的明确处理；
- artifact checksums（产物校验和）；
- README 中说明预期运行时间和最小复现实验。

RoutingBlocks 的 IJOC 固定仓库包含 `data/`、`results/`、`src/`、文档、许可证、CITATION 和基于明确 SHA 的期刊快照，是直接可参照的公开样例。[期刊代码快照](https://github.com/INFORMSJoC/2023.0104)

## 5. 候选期刊与实际 fit（适配度）

### 5.1 分区提醒

2024 SJR 中，下列主要候选均为 Q1：Transportation Science、Transportation Research Part E、European Journal of Operational Research、Computers & Operations Research、INFORMS Journal on Computing。公开 SJR 表中相应数值分别约为 2.324、2.513、2.239、1.605、1.439；分区会随年份和学科类别变化。[Transportation/OR 的 2024 SJR 表](https://kniznica.umb.sk/app/cmsFile.php?ID=22055&disposition=i)；[Computing/OR 的 2024 SJR 表](https://kniznica.umb.sk/app/cmsFile.php?ID=21759&disposition=i)；[EJOR SCImago 页面](https://www.scimagojr.com/journalsearch.php?q=22489&tip=sid)

公开的 2024 JCR 口径可能不同，例如 Optimization Letters 为 Q2；这再次说明投稿时必须写清采用哪套分区。[Springer 期刊页](https://link.springer.com/journal/11590)；[公开 JCR 汇总](https://journalsimpactfactors.com/journal.php?id=13121)

### 5.2 期刊梯度

| 目标 | Journal scope（期刊范围）与合适稿件 | 对当前主题的实际门槛 |
|---|---|---|
| Stretch Q1 | **Transportation Science**：旗舰 transportation analysis，Logistics & Routing area 要求 exciting new modeling/methodology 与新的 scientific knowledge；鼓励 data-driven/real-time routing。[官方 scope](https://pubsonline.informs.org/page/trsc/editorial-statement) | 需要清晰的新问题或可迁移算法创新；只有 ALNS 调参、缓存和本机并行加速很容易在 desk review（编辑初审）阶段被判 contribution insufficient（贡献不足） |
| Stretch Q1 | **EJOR**：高质量原创 OR methodology 或 innovative application（创新应用）。[官方 scope](https://www.sciencedirect.com/journal/european-journal-of-operational-research) | 需要方法论深度、强 state-of-the-art 对比、新 optimum/BKS 或有解释力的决策洞见；近期同主题已有 time-dependent BCP、ALNS operator meta-analysis、坡度/载重/再生制动 |
| Stretch Q1 | **Transportation Research Part C / E**：Part C 强调 emerging technology 对交通系统的影响与开放大数据；Part E 强调 logistics 并接受广泛 OR/AI 方法。[Part C scope](https://www.sciencedirect.com/journal/transportation-research-part-c-emerging-technologies)；[Part E scope](https://www.sciencedirect.com/journal/transportation-research-part-e-logistics-and-transportation-review) | 仅 benchmark algorithm paper（基准算法论文）往往不够；需要现实交通/物流机制、案例或有推广价值的 managerial insight（管理洞见） |
| Strong Q1 | **Computers & Operations Research**：OR theory/practice 与 advanced computational methodology（先进计算方法）的结合，明确覆盖 transportation/logistics。[官方 scope](https://www.sciencedirect.com/journal/computers-and-operations-research) | 对“deadline-safe exact evaluator + batching/cache/screening + rigorous benchmark”这类计算方法论文最自然，但必须把工程机制提升为一般算法贡献，并做全面消融与统计 |
| Strong Q1 / software route | **INFORMS Journal on Computing**：要求显著 computing contribution（计算贡献）；Software Tools area 要求软件长期有用、开放许可，并分别审查软件与论文。[官方 scope](https://pubsonline.informs.org/page/ijoc/editorial-statement) | 若主张 solver/framework 贡献，需达到 RoutingBlocks 一类的模块化、文档、稳定 API、可维护性和公开 archive；若只是单项目实验代码，fit 很弱 |
| Broad OR candidate | **Annals of Operations Research / International Transactions in Operational Research**：近年均直接发表 EVRPTW LNS/matheuristic；ITOR 强调理论与应用、学术与实践之间的桥梁。[ITOR 官方 scope](https://onlinelibrary.wiley.com/page/journal/14753995/homepage/productinformation.html) | 比 Transportation Science 更适合“新变体 + 扎实 matheuristic”，但仍需要独立的新问题设定、完整模型、强对照和洞见 |
| Focused Q2 example | **Optimization Letters**：覆盖 optimization theory、algorithms、computational studies 与 applications；近期已发表 nonlinear charging 的 perspective cuts。[期刊 scope](https://link.springer.com/journal/11590)；[2026 EVRP 论文](https://doi.org/10.1007/s11590-026-02319-4) | 适合边界清晰、篇幅集中、具有数学/算法新意的 exact charging 或 preprocessing 结果；不适合把庞大的阶段性工程记录整体塞入一篇文章 |

## 6. 三条可行的论文路线

以下路线是外部文献导出的研究设计建议，不代表仓库目前已经具备相应贡献。

### Route A：计算方法论文（最贴近现有技术主题）

可能的中心问题：

> 如何在严格 deadline/work-budget semantics（截止期/工作量预算语义）下，将 exact charging route evaluation（精确充电路线评价）安全地批处理、缓存并行化，同时保持候选事务原子性、目标一致性和可复现性？

必须补齐：

1. 将 evaluator 抽象成独立、可复用的问题与 API，而不是 ALNS 内部优化细节。
2. 给出 correctness invariants（正确性不变量）：started/completed/interrupted calls、cache commit、deadline boundary、candidate atomicity。
3. 与 naive scalar（朴素标量）、现有 exact route charging/scheduling 方法及兼容的 RoutingBlocks path 比较。
4. 固定工作量和固定时间双轴；多 worker scaling；独立重复；置信区间与 paired tests。
5. 在完整兼容 benchmark 上证明：
   - 等价 objective/feasibility；
   - 统计显著且有实际幅度的 wall-clock 或 effective-search improvement；
   - 最终 solution quality 不回退。
6. 至少讨论如何扩展到 partial/nonlinear/heterogeneous charging；若只支持 full linear charging，要把限制写清。

较自然 venue：COR；若形成通用开放软件与长期维护路线，可考虑 IJOC。

### Route B：EVRPTW 新模型 + matheuristic/exact hybrid

选择一个没有被当前近期文献直接覆盖、且有现实意义的机制，例如：

- charging station capacity + queue uncertainty（充电站容量与排队不确定性）；
- heterogeneous nonlinear charging + load/gradient（异质非线性充电与载重/坡度）；
- battery degradation（电池退化）；
- dynamic requests + robust/stochastic energy（动态请求与鲁棒/随机能耗）。

必须补齐：

1. 完整数学定义与 objective compatibility（目标兼容性）。
2. 小实例 exact verification。
3. 新 benchmark 或可审计的公开 benchmark extension。
4. 与最近同变体算法对比，而不只是与 Schneider 2014 比。
5. 参数/机制 sensitivity analysis 和 managerial insights。

较自然 venue：AOR/ITOR/COR；若模型与洞见非常强，可挑战 EJOR/TR-C/TR-E/Transportation Science。

### Route C：开放软件/复现基础设施论文

中心贡献必须是社区工具，而不是“公开本项目代码”：

- 稳定且文档化的 EVRPTW research framework；
- 可插拔 charging oracle、objective、validator、ALNS operators；
- benchmark registry 与一键复现实验；
- Python API + 性能关键 native backend；
- 第三方可扩展示例、测试、版本化数据/结果与长期维护计划。

必须直接说明相对于 RoutingBlocks 的非重复价值，例如：

- deadline-transaction semantics；
- formal evidence/replay；
- heterogeneous exact charging backends；
- benchmark provenance/audit；
- cross-platform reproducibility。

较自然 venue：IJOC Software Tools；但这是与成熟公开包正面竞争的高门槛路线。

## 7. 可量化的投稿 readiness checklist（就绪检查表）

这不是期刊官方评分，而是用于避免“感觉差不多”的内部 gate。

### Q2/广义强 OR venue 的最低建议

- [ ] 1 个一句话可说清、没有被近期论文直接覆盖的 central research question（核心研究问题）
- [ ] 至少 1 个模型或算法层面的 primary contribution（主要贡献），而非纯工程重构
- [ ] 小实例 exact validation；大实例覆盖主要 benchmark families
- [ ] 至少 2 个外部 baselines + 完整自身 ablation matrix
- [ ] 随机算法使用经解释的重复次数；至少给出 CI、effect size、paired non-parametric test
- [ ] objective/model 完全兼容；不兼容时不计算误导性 gap
- [ ] 可重跑 artifact package、固定版本、许可证、原始逐运行结果
- [ ] 限制、失败与 negative results（负结果）如实报告

八项中若缺失 central question 或 primary contribution，其余工程质量再高也不构成投稿就绪。若其余项目缺两项以上，通常仍处于研究原型/实验基础设施阶段。

### Q1 强方法/交通 venue 的附加建议

- [ ] 对近 3 年最强同类方法有逐项对照，而非只引经典文献
- [ ] 至少一个可证性质、一般算法机制、新 optimum/BKS，或无法被简单实现替代的建模洞见
- [ ] 多规模、多 family、完整范围的结果，不依赖少数 seed/实例
- [ ] 有 practical/managerial interpretation（实践/管理解释），或方法可明显迁移到其他问题
- [ ] 所有 headline claims（主结论）都有统计与复现实验证据

Q1 并不是“Q2 清单多跑一些实例”。最主要的额外距离通常是 **贡献锐度与一般性**，其次才是实验规模。

## 8. 论文写作上的必要收敛

最终稿不应按 Stage 0--5 的开发时间线组织。更合适的论文结构是：

1. 一个研究问题；
2. 一个明确的问题定义；
3. 2--3 个可验证 contributions；
4. 方法与正确性/复杂度；
5. 实验问题与预注册式协议；
6. 结果、消融、统计与管理/算法洞见；
7. 限制与下一步。

阶段 registry、manifest、review gate 和原始 trace 应作为 reproducibility backbone（复现骨架）放入 supplement/artifact（补充材料/产物），而不是取代论文的 scientific narrative（科学叙事）。

## 9. 最终判断框架

只依据外部领域门槛，可以得出：

- 一个严谨、可复现的 EVRPTW ALNS 工程 **具备成为论文实验平台的价值**；
- 但“完成了许多 solver stages、算子、cache、parallel 和 evidence gates”本身不能证明达到 Q1/Q2；
- 最短发表路径通常是从现有工作中提炼 **一个一般化的算法问题**，补强近期外部 baseline、统计实验和兼容 benchmark；
- 如果没有新的中心研究问题，距离 Q1/Q2 不是“再跑 20% 实验”，而是缺少论文的核心贡献；
- 如果已有可迁移的 deadline-safe/batched exact-evaluation 方法，则距离更可能集中在：理论化、外部对照、全量 benchmark、统计推断和公开复现包。

对具体仓库的“已经完成多少、还差多少”，应把本清单与当前实现、formal results（正式结果）、失败证据和模型边界逐项映射后再评分。
