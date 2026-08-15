# TxnOpt → Computers & Operations Research 研究交接文档

日期：2026-08-16  
文档状态：`ACTIVE_HANDOFF`  
版本：`txnopt-cor-handoff-v1.0`  
目标期刊：*Computers & Operations Research*（COR）  
当前项目阶段：腾讯云账号输入前，正式 Level 1 矩阵尚未授权、尚未启动

## 1. 交接目的

本文把 TxnOpt 当前工程状态、论文定位、transport contribution（交通运输领域贡献）
判断、实验缺口、云端执行口径和下一阶段动作固定下来，使新的研究者、审查者或 Codex
任务无需依赖此前对话也能继续工作。

本文不是云采购授权、正式矩阵授权、holdout（最终保留集）授权、投稿授权或公开发布
授权。任何新的代码、协议、实验或论文主张仍须服从仓库根目录 `AGENTS.md`、活动路线图
和 append-only evidence（追加式证据）边界。

## 2. 已锁定决策

1. 首选投稿方向为 *Computers & Operations Research*，不再把
   *Transportation Science* 作为默认首投目标。
2. 论文的 primary contribution（主要贡献）是 general computational OR
   methodology（通用计算运筹方法），而不是新的交通模型或新的 EVRPTW 元启发式算法。
3. transport contribution 只作为 secondary contribution（次要贡献）主张；在未补齐
   运输专属证据前，只能称 transportation application（交通运输应用）。
4. 当前 Level 1–3 原计划全部完成但不增加运输专属研究设计时，COR 审稿人接受一项真实
   次要 transport contribution 的主观校准概率约为 **40%**，合理区间为
   **30%–50%**。
5. 如果补齐本文第 8 节的 transportation evidence package（交通运输证据包），该概率
   可提高到约 **70%**，合理区间为 **65%–80%**。
6. 上述概率是 research judgment（研究判断），不是期刊录用率、历史频率或统计置信区间。
7. 完成更多运行只增强 evidence strength（证据强度）；不能自动创造 domain novelty
   （领域新颖性）。

## 3. 当前权威状态

### 3.1 Git 与独立审查

- 权威工作区：`/home/oneblaze/work/TxnOpt`
- 分支：`codex/txnopt-level1`
- 本文创建前 HEAD：`b14b46ea1d75eeb829946ae611110660f040c68d`
- 本文创建前工作树：clean，且与 `origin/codex/txnopt-level1` 对齐
- 当前生产链：Build24 → formal Attempt40 → local calibration Attempt41 →
  Deployment06 → Pre-cloud Attempt13 → Independent Review Attempt18
- [Independent Review Attempt18](../../formal/reviews/txnopt_tencent_precloud_review_attempt18.json)
  结果：`PASS`
- Attempt18 状态：`READY_FOR_TENCENT_ACCOUNT_INPUT_NOT_AUTHORIZED`
- finding counts（发现计数）：critical=0，major=0，minor=0

### 3.2 明确未发生的动作

- 未提供腾讯云账号或凭据；
- 未选择 region（地域）、zone（可用区）或 SKU（实例规格）；
- 未完成在线 `DryRun=true`；
- 未授权云采购或实例创建；
- 未启动 Attempt40 正式矩阵；
- 未打开 holdout；
- 未达到 Level 1 Ready；
- 未授权公开发布、投稿或 E 盘删除。

### 3.3 当前关键证据

- [Build24 manifest](../../experiments/txnopt/manifests/txnopt_level1_build_attempt24.json)
- [Attempt40 formal plan manifest](../../experiments/txnopt/manifests/txnopt_level1_formal_plan_attempt40.json)
- [Attempt41 local calibration manifest](../../experiments/txnopt/manifests/txnopt_level1_local_calibration_attempt41.json)
- [Deployment06 manifest](../../experiments/txnopt/manifests/txnopt_tencent_deployment_build24_attempt06.json)
- [Pre-cloud Attempt13 manifest](../../experiments/txnopt/manifests/txnopt_level1_precloud_gate_attempt13.json)
- [活动路线图](../roadmap/txnopt-level1-to-level3-roadmap.md)
- [发表缺口审计](txnopt-publication-gap-audit-2026.md)
- [外部方法与基准边界](txnopt-external-methods-and-benchmarks-2026.md)

注意：部分较早的研究审计仍使用 Build14/Attempt24 等旧名称。其发表缺口判断仍可作为
研究输入，但任何 current-state（当前状态）主张必须以 Build24/Attempt40/Attempt41/
Pre-cloud13/Review18 链为准。

## 4. 论文的推荐定位

### 4.1 一句话定位

TxnOpt 是一个面向 state-dependent deterministic combinatorial search
（状态依赖的确定性组合搜索）的 auditable transactional parallel runtime
（可审计事务并行运行时）：它在保持 serial decision sequence（串行决策序列）、最终
状态、objective（目标）和 semantic digest（语义摘要）一致的同时，提高实际运行速度，
并对 deadline（截止时间）、worker failure（工作线程故障）和 discarded work（废弃工作）
提供可重放的边界证据。

### 4.2 推荐贡献结构

1. **方法贡献**：预先固定的 transactional publication semantics（事务发布语义），
   适用于状态依赖、不能任意重排接受决策的组合搜索。
2. **形式与安全贡献**：scheduler-relative refinement（相对调度器精化）、failure/deadline
   prefix safety（故障/截止时间前缀安全）和 measured waste bound（实测浪费界）。
3. **计算贡献**：跨 EVRPTW 与 RCPSP 的预注册、多执行轴、fixed-work（固定工作量）/
   fixed-time（固定时间）、独立 runner/reviewer（运行器/审查器）证据。
4. **次要领域贡献**：确定性、故障安全的并行搜索在受决策时间限制的电动车路径规划中
   何时能改善路线决策，以及何时会被候选成本不均衡或无效工作限制。

### 4.3 禁止或暂缓的主张

在没有新增证据前，不得声称：

- 提出了新的 EVRPTW 模型；
- 提出了新的 EVRPTW 元启发式算法；
- 解决了实时电动车路径规划；
- 证明了普适 transportation speedup（交通运输加速规律）；
- TLC bounded model checking（有界模型检查）等于一般 liveness proof（活性证明）；
- 本地代表性校准等于 64 物理核心的真实云端扩展性证明；
- 代码整洁、可重放或可在云端部署本身构成科学新颖性。

## 5. 为什么 COR 是合理目标

COR 官方范围把 Transportation（交通）、Logistics（物流）、Manufacturing（制造）和
Supply Chain Management（供应链管理）列为 OR application areas（运筹应用领域），
并强调 advanced computational methodologies（先进计算方法）。完整研究论文需要体现
constructive algorithmic complexity（具有实质内容的算法复杂性）和 extensive numerical
experiments（广泛数值实验）。

官方来源：

- [Computers & Operations Research — Aims & Scope](https://www.sciencedirect.com/journal/computers-and-operations-research)
- [COR — Guide for Authors](https://www.sciencedirect.com/journal/computers-and-operations-research/publish/guide-for-authors)

因此，COR 允许以下结构：

> 通用 OR 计算机制为主要贡献，EVRPTW 与 RCPSP 作为具有不同状态结构和验证语义的
> 两个应用领域，证明该机制不是单领域特例。

这意味着 transport contribution 不是 COR 论文成立的必要条件，但如果要把它列入正式
贡献列表，就必须通过第 8 节的领域证据门。

## 6. Transportation Science 边界

*Transportation Science* 要求论文产生与运输系统规划、设计、运营、经济或社会问题
相关的新知识。其 Logistics & Routing（物流与路径）方向欢迎推进路径问题模型、方法或
科学认识的工作。

官方来源：

- [Transportation Science 2025/2026 Editorial Statement](https://pubsonline.informs.org/doi/10.1287/trsc.2025.ed.v60.n1)
- [Transportation Science Areas — Logistics & Routing](https://pubsonline.informs.org/page/trsc/editorial-statement)
- [Transportation Science Submission Guidelines](https://pubsonline.informs.org/page/trsc/submission-guidelines)

TxnOpt 当前更像“在交通领域验证的通用计算贡献”，而不是“由交通问题驱动并产生新交通
知识的贡献”。因此不应使用 Transportation Science 的审稿标准反向扭曲 COR 稿件。

## 7. Research Question handoff（研究问题交接）

### 7.1 主研究问题

在状态依赖、确定性的组合搜索中，预先固定的事务发布语义能否在保持 objective、最终
状态和 semantic digest 等价的同时，获得跨 EVRPTW 与 RCPSP 的可复现 wall-clock
improvement（实际耗时改进），并把 deadline、failure 和未提交工作限制在预注册的可测
边界内？

### 7.2 运输子研究问题

在具有充电约束、时间窗和严格决策时间预算的 EVRPTW 搜索中，deterministic
transactional parallelism（确定性事务并行）能否在保持路线可行性、目标语义和复现性
的同时，提高固定时间内的路线质量，并把故障导致的废弃计算限制在可测边界内？

### 7.3 子问题

1. 相同 fixed-work 下，serial、barrier 和 ordered transaction execution 是否产生完全
   相同的 objective、final state 和 semantic digest？
2. 相同 fixed-time 下，TxnOpt 是否完成更多有效工作并改善 EVRPTW 路线质量？
3. 收益是否随 customer count、time-window tightness、charging-station density、battery
   tightness 或昂贵候选评价占比发生系统变化？
4. deadline 和 worker failure 下，最后提交的路线是否保持可行，废弃工作是否满足 T4/
   Cmax 边界？
5. 与至少一个 same-model/same-objective/same-budget/same-hardware 外部 EVRPTW baseline
   相比，TxnOpt 的速度、固定时间质量和复现性是否具有实质增量？

### 7.4 Scope（范围）

**In scope（范围内）**：

- generic transactional runtime；
- EVRPTW 与 RCPSP 两个 adapter（适配器）；
- deterministic serial/barrier/ordered execution；
- fixed-work 与 fixed-time；
- objective/semantic parity、prefix safety、T4、Cmax、fallback=0；
- EVRPTW 路线质量、time-to-best（达到最佳解的时间）和 transport heterogeneity
  （交通实例异质性）分析；
- 同条件外部基线和独立重放。

**Out of scope（范围外）**：

- 声称一般异步搜索都可确定化；
- 新交通需求模型、行为模型或政策模型；
- 未实现的在线动态请求；
- 未验证的真实实时运营系统；
- 以不同模型的 CEVRP 结果冒充 EVRPTW 基线；
- 将 heuristic BKS（启发式最佳已知解）表述为 certified optimum（经证明最优）。

### 7.5 初步假设

在昂贵确定性候选评价占比较高、可并行评估但接受决策必须保持串行语义的实例上，TxnOpt
可在不改变固定工作量语义的前提下提高吞吐；在固定时间预算下，这可能转化为更好的
EVRPTW 路线质量。收益预计受候选成本方差、可行候选比例和 transport constraint
tightness（运输约束紧度）调节。

该假设尚未通过正式云矩阵或运输专属 successor protocol（后继协议）验证。

## 8. Transportation evidence package

### 8.1 必须增加的实验

1. **同条件外部基线**
   - 模型、目标、实例、随机预算、wall-clock budget（实际时间预算）和硬件一致；
   - 至少一个可独立运行、许可清楚、来源可核验的 EVRPTW baseline；
   - 现有 CEVRP 双层方法因时间窗和目标不同，不能直接充当该基线。

2. **固定时间路线质量**
   - 至少报告 1、3、10 分钟或经预注册选择的多个决策预算；
   - 不得只报告 speedup；必须把更多有效工作映射到路线质量。

3. **运输专属异质性分析**
   - customer count；
   - time-window tightness；
   - charging-station density；
   - battery/charging constraint tightness；
   - expensive evaluation share；
   - 候选成本分布或方差。

4. **故障与截止时间的路线级证据**
   - 最后 committed route（已提交路线）可行；
   - vehicle count 不回退；
   - objective tuple 可独立重建；
   - discarded work 与 T4/Cmax 一致；
   - 失败后不得把未验证路线作为结果发布。

### 8.2 必报指标

- vehicle count；
- total distance；
- total charging time；
- charging count；
- feasible-solution rate；
- time-to-best；
- fixed-time objective gap 或相对基线改善；
- fixed-work semantic/objective parity；
- wall-clock speedup；
- one-worker overhead；
- T4 waste 与 measured Cmax；
- fallback count；
- peak RSS（峰值常驻内存）；
- 按实例特征分层的 effect size（效应量）与 uncertainty interval（不确定性区间）。

### 8.3 Transport contribution 通过门

只有以下项目全部成立，论文才可把 transport contribution 列为正式次要贡献：

1. 外部基线比较满足同模型、同目标、同预算、同硬件；
2. fixed-work 下无路线质量、可行性或语义回退；
3. 至少一个预注册 fixed-time 预算出现具有实际意义的路线质量或 time-to-best 改善，且
   不确定性分析支持该结论；
4. 至少识别并复核一条运输实例特征与收益边界之间的非平凡规律；
5. fault/deadline 结果保持最后完整可行路线并满足废弃工作界；
6. 独立 reviewer 从 raw evidence 重建全部路线级指标；
7. 论文将结论限制在已测试的 EVRPTW 模型、实例、预算和硬件范围内。

若其中任一项失败，论文仍可保留 COR 的 general OR computing contribution，但应把
EVRPTW 描述为 evaluation domain（评估领域），而不是 transport contribution。

## 9. 贡献成立概率与决策规则

| 研究状态 | COR 次要 transport contribution | 强到可作为论文主要交通贡献 | Transportation Science 定位 |
|---|---:|---:|---:|
| 当前云前状态 | 15%–25% | <10% | <10% |
| 完成当前 Level 1–3，不补运输专属设计 | 30%–50%，中心 40% | 15%–25% | 10%–20% |
| 完成 Level 1–3，并通过第 8 节证据包 | 65%–80%，中心 70% | 40%–60% | 30%–50% |

这些区间只用于研究资源分配。它们不得写入论文、manifest、统计结果或投稿信。

**决策规则**：

- 若 transport gate 全部通过：在 COR 中列为第 4 项次要贡献；
- 若仅速度和复现性通过：称为 EVRPTW application evidence（EVRPTW 应用证据）；
- 若 fixed-time 质量无改善但安全/复现性强：把 transport 结果写成 boundary finding
  （边界发现）或 negative result（负结果）；
- 若通用机制、新颖性或外部基线失败：暂停 Q1/Q2 正面方法论文，保留软件和负结果路线。

## 10. Level 1–3 与论文关系

### Level 1

- 完成 Attempt40 successor formal execution（后继正式执行）及独立 review；
- 证明语义、故障、安全、T4/Cmax 和基本性能门；
- 不是论文完成，也不能独自建立 transport contribution。

### Level 2

- 独立 30 EVRPTW + 60 RCPSP、20 seeds、1/2/4/8 worker 轴和探索性 16-worker 轴；
- major ablations（主要消融）、独立 rerun（重跑）、preprint draft（预印本草稿）；
- 应在 Level 2 protocol 冻结前加入第 8 节运输证据包；
- Level 2 后进行一次 COR journal-fit review（期刊适配审查），但按仓库政策不得跳过
  Level 3 或提前投稿。

### Level 3

- 60 Homberger-1000 EVRPTW + 120 PSPLIB J120；
- 30 seeds、五个 controls（对照）、1/2/4/8/16 workers；
- fixed-work、fixed-time、fault injection（故障注入）和 scaling（扩展性）；
- one-time final holdout、完整 DOI archive（DOI 归档）和 submission package（投稿包）；
- Level 3 完成后才能依据活动路线图进入最终投稿判断。

## 11. 云端资源与时间口径

### 11.1 主机合同

- 腾讯云；
- 至少 64 physical CPU cores（物理核心）；
- `CoreCount=64`；
- `ThreadPerCore=1`；
- provider-reported memory（厂商规格内存）至少 128 GB；
- Linux 可见内存只记录，不因系统保留略低于 128 GiB 自动拒绝；
- Attempt41/对应 successor calibration 的 peak RSS 必须低于目标主机可见内存 80%；
- region、zone、SKU、image、VPC、subnet、security group 必须由用户显式提供；
- 当前阶段只允许 `DryRun=true`，实例创建和正式矩阵需要新的明确授权。

### 11.2 存储与网络建议

- 系统盘：50 GB；若在云端完整重建、保留 sanitizer/JDK/TLC/测试缓存，建议 80 GB；
- 数据盘：20 GB；需要保留多轮正式 raw/review 或双本地 staging 时选 50 GB；
- 不复制仓库内约 100 GiB 历史 `results/`；
- 不需要固定公网 IP；优先同地域 COS + VPC 私网；bootstrap（引导安装）如需公网出站，
  使用 NAT 或受控跳板；
- 用户已说明网络速率价格相同，因此不以带宽档位作为主要优化变量。

### 11.3 Level 1 时间

- Attempt41 外推的 2,880-run raw matrix 点估计：约 1,922 秒，即约 32.0 分钟；
- 该值不是 live-cloud proof（真实云证明）；
- 第一次正式 raw run 应保护 60–90 分钟连续无中断窗口；
- 含启动、doctor、部署/fresh build、正式 raw、独立 review、COS、核验和应急余量，整次
  云端会话建议 4–8 小时；
- 第一次购买/保留 12–24 小时即可，不建议预购多天；
- SSH 断开但后台进程继续不算中断；VM 停止、进程死亡或 evidence root 被污染算失败；
- 同一正式 attempt 不得暂停后续跑，失败后必须使用新 attempt label 和新 raw root。

### 11.4 Level 2–3 粗略规划

- Level 1–2 合计预计约 3–5 个实际 server-on days（服务器开启日），分为 2–3 个运行块；
- 运行块之间可离线分析 3–10 天，无需持续租用实例；
- Level 3 仅按 cell count（运行单元数）计算的乐观下限约为 Level 1 的 93.75 倍，但
  Homberger-1000、16-worker 轴、fault/scaling 和确认性重跑会显著增加耗时；
- Level 3 规划区间约 7–21 个额外 64-physical-core machine-days（机器日）；
- Level 1–3 总规划区间约 10–26 个 server-on days；
- 不应一次性购买 10–26 天。先购买 1 天取得真实云 p95、NUMA、磁盘和编译数据，再重新
  估计 Level 2/3。

## 12. 下一执行者的最短正确路径

### 阶段 A：账号输入前

1. 阅读本文、`AGENTS.md`、活动路线图、Attempt18 和 Attempt40/41 manifests；
2. 确认工作树 clean、远端分支一致、Attempt40 raw root 和 launch claim 均不存在；
3. 不修改 Attempt40、Build24、Attempt41、Pre-cloud13、Review18 或历史 sidecar；
4. 完成 COR closest-work audit（最接近工作审计）和外部 EVRPTW baseline 选择；
5. 为 Level 2 起草 successor preregistration，加入第 8 节运输证据包；
6. baseline 未完成 same-model audit 前，不把任何 CEVRP 方法写成直接对照。

### 阶段 B：腾讯账号到位后，但正式运行前

1. 收集显式 region/zone/SKU/image/VPC/subnet/security group；
2. 在线调用仅限 `DryRun=true`；
3. 交叉确认腾讯 API 的 64 physical cores/128 GB 与 Linux topology；
4. 跑 `txnopt cloud tencent doctor` 并保留 receipt；
5. 核验 COS versioning、COMPLIANCE Object Lock、至少 365 天保留和精确 VersionId；
6. 提交精确 plan/build/root/window 的正式执行授权请求；
7. 未获得单独授权前，不创建实例、不运行矩阵、不打开 holdout。

### 阶段 C：Level 1 正式执行

1. 原子 claim attempt；
2. runner 只写 raw；
3. 同一 attempt 连续完成，不暂停续跑；
4. 独立进程 reviewer 重放每个 raw manifest；
5. 聚合 95% CI、objective/semantic parity、overhead、T4、Cmax 和 fallback；
6. 任一失败保留 immutable failure（不可变失败证据），使用新 successor attempt 修复；
7. Level 1 PASS 后再进入 Level 2 transport protocol freeze。

### 阶段 D：论文路线

1. Level 2 前完成系统文献检索和 novelty matrix（新颖性矩阵）；
2. 冻结主张、基线、指标、实例特征和统计分析；
3. Level 2 后做一次 COR quick journal-fit review，不提前投稿；
4. Level 3 后生成 manuscript v1；
5. 依次完成 citation/data integrity、五视角 peer review、revision、re-review 和 final
   integrity；
6. 投稿仍需用户单独明确授权。

## 13. 论文草稿建议结构

1. **Introduction**：状态依赖组合搜索中的并行、安全和复现性冲突；
2. **Related Work**：ordered/speculative search、deterministic parallelism、transactional
   publication、EVRPTW/RCPSP computing；
3. **Problem and Contracts**：五接口、状态机、预算、trace、identity；
4. **TxnOpt Runtime**：proposal/evaluation/publication seam；
5. **Formal Obligations**：T1–T4、适用范围、反例和非主张；
6. **Experimental Protocol**：预注册、实例、轴、预算、统计、独立 reviewer；
7. **Cross-domain Results**：EVRPTW 与 RCPSP 的 parity、安全、性能和扩展性；
8. **Transportation Analysis**：固定时间路线质量、外部基线、异质性和故障决策；
9. **Limitations**：调度器相对性、硬件依赖、模型范围、无一般 liveness 证明；
10. **Reproducibility and Artifact**：source/build/raw/review/report identity chain；
11. **Conclusion**：通用计算贡献为主，运输结论严格限定。

## 14. 主要风险登记

| 风险 | 当前状态 | 响应 |
|---|---|---|
| 通用 speculative/ordered search 已有先行工作 | 已知高风险 | 把贡献限定为 auditable runtime/refinement/failure/waste boundary |
| 无同条件外部基线 | 未关闭 | Level 2 前完成 baseline audit 和独立运行 |
| transport contribution 只是应用 | 未关闭 | 执行第 8 节，未通过则降级措辞 |
| Level 1 正式性能门失败 | 未知 | 保留负结果，修复后新 attempt，不覆盖 |
| 64 核云端与本地 p95 不一致 | 未知 | 第一云日重新估计，不外推本地结果冒充 live evidence |
| Build portability | 云上未验证 | 云实例 fresh build 和独立来源检查 |
| 论文新颖性不足 | 未关闭 | systematic closest-work audit，允许转软件/边界/负结果路线 |
| 运输固定时间质量无改善 | 未知 | 不强行包装；保留可靠性/复现性边界或负结果 |
| 研究范围失控 | 持续风险 | 不同时加入多个新交通模型；先完成一个严格 EVRPTW lane |

## 15. 完成交接的验收条件

下一执行者在开始任何新实验前，应能明确回答：

1. COR 的主要贡献是什么，transport contribution 为什么只是次要主张？
2. 当前 transport contribution 的 40% 判断依赖哪些条件？
3. 哪七项门决定它能否提高到约 70%？
4. 当前权威 build/plan/calibration/deployment/precloud/review 身份是什么？
5. 哪些证据和标签绝对不可修改或复用？
6. 为什么 Attempt40 不能为运输专属设计而重写？
7. 腾讯账号到位后，哪些步骤仍需要单独授权？
8. 正式 attempt 中断后为什么必须使用新 label/root？
9. 哪些结果支持 COR，哪些结果才支持真正的 transport contribution？
10. 何时应停止正面论文路线并转向软件、边界或负结果？

若任一问题不能回答，应先回读本文件和所列权威材料，不得直接启动云端矩阵。

## 16. Material Passport（材料护照）

- Origin Skill：`academic-research-suite`
- Origin Mode：`academic-pipeline-handoff`
- Origin Date：`2026-08-16`
- Verification Status：`UNVERIFIED`
- Version Label：`txnopt-cor-handoff-v1.0`
- Upstream Dependencies：Build24、Attempt40、Attempt41、Deployment06、Pre-cloud13、
  Independent Review18、活动路线图、发表缺口审计
- Repro Lock：`null`
- Experiment Intake Declaration：`experiments_declared`

`UNVERIFIED` 的含义：仓库身份、Review18 和云前状态已在本文创建时重新读取；但期刊适配、
40%/70% 概率和未来实验设计属于研究判断，尚未经过独立学术 reviewer 或正式文献完整性
门。任何论文草稿消费本文前，仍须执行新的 literature search、claim verification 和
pre-review integrity gate。

