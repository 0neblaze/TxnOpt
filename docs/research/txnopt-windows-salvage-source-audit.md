# TxnOpt Windows salvage source audit

审计日期：2026-08-14  
审计对象：旧 Windows clone `D:\UserData\Documents\GitHub\TxnOpt` 中的 dirty research additions（未提交研究增量）、`docs/literature.md` 与 `docs/references.bib`。  
权限边界：本审计只读旧 clone、论文/出版社页面和云厂商官方文档；本次只新增本文件，不提交、不推送、不删除 clone、WSL 文件或 E 盘数据。

## 结论

1. 四份算法/benchmark（基准）/发表缺口文档有可迁移的研究判断，但当前仍以旧 EVRPTW/Stage 路径叙述，不能原样进入 TxnOpt 活树。可迁移的是模型差异、外部基准的隔离原则、发表就绪缺口和 provenance（来源追踪）要求；所有 `Stage 5.x`、`src/evrptw`、旧 CLI、旧 roadmap 名称和“当前已完成”表述必须改写或删除。
2. 三篇新增论文的题名、作者、载体、卷期页码和 DOI 与 IEEE publisher metadata（出版社元数据）一致，BibTeX 条目可迁移。论文中“更新 7/8/11 个 BKS（最佳已知解）”等实验结果属于全文结果断言；本次没有把它们升级为 TxnOpt 证据，迁移前仍需以论文全文和独立 route-level replay（路线级重放）核对。
3. 云报告的“vCPU 不等于物理核”“库存/配额/价格需控制台实时确认”“不能用月价外推按量成本”边界可迁移；旧的月租推荐和 Stage 5.2 资源结论不能迁移。当前官方百度 BCC 页面显示 `bcc.c7.c64m128`，而 dirty 报告写成 `bcc.c7.c64m256`，该行必须修正；`ca3` 页面仍显示 64/128 与 128/256 规格。
4. TxnOpt 当前应保留的本地估计边界是：约 32 physical cores（物理核）、约 4,092 秒（约 1.14 小时），且仅为 provisional local evidence（暂定本地证据），不是云报价、不是 SLA，也不是正式矩阵结果。Build14 wheel（轮子）的跨主机 Build portability（构建可移植性）仍是 **unverified（未验证）**；不得把“能在当前主机安装”写成可移植性通过。
5. 本阶段不采购、不启动 Attempt24 formal matrix（正式矩阵）、不发布、不推送、不删除 E 盘。任何云端适配器和 E 盘退役仍受计划中独立授权与恢复演练门槛约束。

## 1. 旧 Windows 文档的迁移判定

下表把“事实可迁移”与“文件现状”分开。文档本身不是 TxnOpt evidence（证据），只可作为研究背景和待审查来源。

| dirty 文档 | 可迁移事实 | 必须改写或删除的内容 | 推荐处置 |
|---|---|---|---|
| `bilevel-cevrp-algorithms-assessment.md` | 三篇 CEVRP 论文共同采用上层 routing（路径）+ fixed-route charging problem（固定路线充电子问题）的双层分解；便宜候选筛选可减少昂贵下层评价；启发式 charging plan 可作为 incumbent/upper bound（当前最好解/上界）；WCCI-2020 17-instance protocol（17 实例协议）应作为 out-of-family benchmark（分布外基准）。论文/协议来源见[WCCI technical report](https://mavrovouniotis.github.io/Papers/TR-EVRP-Competition.pdf)及三篇 DOI。 | `Stage 5.1/5.2`、`src/evrptw/charging.py`、旧 `atomic decision` 路径和“与当前 solver 已兼容”的强断言；不能把论文 BKS 写入 Schneider/TxnOpt best-known registry。 | 迁入 `docs/research/` 的 TxnOpt 术语版本；保留“独立 benchmark lane（基准通道）”和未决全文核验项。 |
| `mavrovouniotis-e-cvrp-benchmark-assessment.md` | Mavrovouniotis E-CVRP 是 24 个、约 21--1000 customers（客户）的独立 E-CVRP；模型有 battery/cargo constraints（电池/载重约束）、可重复充电站和 load-dependent energy（载重相关能耗），目标不是 TxnOpt 的四元字典序目标。来源为[CEC-2020 论文](https://mavrovouniotis.github.io/Papers/CEC20.pdf)和[实例仓库固定快照](https://github.com/Mavrovouniotis/e-cvrp_benchmark_instances/tree/a3f59afb6ddc3999961060ca7035a08fa1f7d59c)。 | 旧 `parse_schneider()`、`src/evrptw` 路径、将 `OPTIMAL_VALUE` 统一称为 proven optimum（已证明最优）、把 24 个文件加入当前 formal scope。 | 作为独立 `txnopt_cases` benchmark lane 候选；必须另建 parser、objective、validator、manifest 和许可记录。 |
| `evrptw-publication-gap-audit-2026.md` | 中心研究问题、可证伪 claims（主张）、同模型外部 baseline（外部基线）、统计推断、公开 raw artifact（原始产物）、独立外部复现仍是论文级门槛；工程质量不能代替方法新颖性。期刊口径可引用[Transportation Science editorial statement](https://pubsonline.informs.org/page/trsc/editorial-statement)和[IJOC editorial statement](https://pubsonline.informs.org/page/ijoc/editorial-statement)。 | 所有 Stage 0--8 状态、`evrptw-research-roadmap.local.md`、旧 Stage readiness（就绪）判断和旧实现路径；不能把“假设 Stage 全部完成”的反事实当成当前结果。 | 重写为 TxnOpt 的研究问题/证据缺口审计；保留 pending（待验证）的新颖性与外部复现实验缺口。 |
| `three-evrp-repositories-assessment.md` | `2E-EVRP-Instances`、`E-VRP-HC`、`HEVRP-NL` 都不是 Schneider formal scope 的直接替代；三者需要独立模型、parser、validator、objective、manifest 和许可边界。2E 数据论文/版本见[DOI](https://doi.org/10.1016/j.dib.2025.111470)、[PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11985070/)和[Zenodo v2](https://zenodo.org/records/14844216)；E-VRP-HC 见[DOI](https://doi.org/10.1016/j.cor.2025.107374)；HEVRP-NL 见[DOI](https://doi.org/10.1016/j.trc.2024.104932)及[GERAD report](https://www.gerad.ca/en/papers/G-2024-01)。 | `src/evrptw`、Stage 目录、把第三方 RAR 原始字节纳入 Apache-2.0、把论文 heuristic BKS 当 certified optimum（经证明最优）。 | 作为外部依赖与隔离 benchmark 说明；未经许可和完整配置核验不得拷贝 RAR 或接入 formal evidence。 |

## 2. 三篇论文与 BibTeX 核验

Dirty addition 的三个 key 与下列出版社身份一致。DOI 页面是 publisher primary source（出版社一手来源）；IEEE Xplore 页面对第一篇和第三篇还提供了直接记录。作者、题名、载体、卷期和页码与当前 dirty BibTeX 相符。

| key | 出版者身份核验 | BibTeX 判定 | 可迁移的研究事实 | 尚未闭合的证据 |
|---|---|---|---|---|
| `Jia2022BilevelACO` | Jia, Ya-Hui; Mei, Yi; Zhang, Mengjie. *A Bilevel Ant Colony Optimization Algorithm for Capacitated Electric Vehicle Routing Problem*. IEEE T-Cybernetics 52(10), 10855--10868. [DOI](https://doi.org/10.1109/TCYB.2021.3069942), [IEEE record](https://ieeexplore.ieee.org/document/9409782) | **PASS（元数据）**。DOI 年份为 2021 是在线/登记年份，期刊卷期为 2022；dirty `year = 2022` 与正式卷期一致。 | BACO 的 routing/charging 双层分解，以及固定路线充电子问题边界。 | “17 个实例更新 7 个 BKS”需以正式全文的结果表核对；不能把它当 TxnOpt 结果。 |
| `Jia2022ConfidenceACO` | Jia, Ya-Hui; Mei, Yi; Zhang, Mengjie. *Confidence-Based Ant Colony Optimization for Capacitated Electric Vehicle Routing Problem With Comparison of Different Encoding Schemes*. IEEE T-Evolutionary Computation 26(6), 1394--1408. [DOI](https://doi.org/10.1109/TEVC.2022.3144142) | **PASS（元数据）**。题名、大小写、卷期、页码和 DOI 一致。 | confidence-based candidate selection（候选筛选）、两种 encoding（编码）和 simple enumeration（简单枚举）可作为方法背景；不能改写成 TxnOpt safe-screening theorem（安全筛选定理）。 | “更新 8 个 BKS”及 30-run statistics（30 次统计）必须以全文表格/原始实验协议核对。 |
| `Feng2024BilevelHGA` | Feng, Chang-Tao; Jia, Ya-Hui; Yang, Qiang; Chen, Wei-Neng; Jiang, Huaiguang. *A Bilevel Hybrid Genetic Algorithm for Capacitated Electric Vehicle Routing Problem*. 2024 IEEE CEC, pp. 1--8. [DOI](https://doi.org/10.1109/CEC60901.2024.10611987), [IEEE record](https://ieeexplore.ieee.org/abstract/document/10611987) | **PASS（元数据）**。会议题名、五位作者、页码和 DOI 一致。 | 上层 hybrid genetic search（混合遗传搜索）、screening/enumeration（筛选/枚举）管线可作为外部方法参照。 | “更新 11 个 BKS”需以正式论文结果核对；不能写成对 Schneider 或 TxnOpt 的改进。 |

共同模型边界应写清：这些论文是 CEVRP，不含当前 TxnOpt EVRPTW 的 customer time windows（客户时间窗）和四分量 lexicographic objective（字典序目标）。WCCI-2020 技术报告的实例/评估协议见[TR-EVRP-Competition.pdf](https://mavrovouniotis.github.io/Papers/TR-EVRP-Competition.pdf)；另一个 load-dependent benchmark（载重相关基准）见[CEC20.pdf](https://mavrovouniotis.github.io/Papers/CEC20.pdf)。因此三篇论文只能作为独立外部方法和 benchmark lane 依据，不能直接生成 TxnOpt BKS、speedup（加速比）或 semantic digest（语义摘要）。

## 3. 云报告的官方规格复核

以下只保留可由官方规格页支持的“候选规格存在”事实；inventory（库存）、quota（配额）、region（地域）和实时价格必须在控制台/API 再确认。规格页不是采购承诺。

| 厂商 | 官方页面当前可见的候选 | 对 dirty cloud report 的判定 |
|---|---|---|
| 阿里云 ECS g8a | `ecs.g8a.16xlarge` = 64 vCPU/256 GiB；`ecs.g8a.32xlarge` = 128 vCPU/512 GiB；默认超线程。见[官方 g 系列规格](https://help.aliyun.com/zh/ecs/user-guide/general-purpose-instance-families)和[CPU options](https://help.aliyun.com/en/ecs/user-guide/cpu-options-of-general-purpose-instance-families)。 | **可迁移，但只写 vCPU**；不得写成 64/128 physical cores。 |
| 腾讯云 CVM SA5 | `SA5.16XLARGE256` = 64 vCPU/256 GB；`SA5.32XLARGE576` = 128 vCPU/576 GB。见[官方 SA5 规格表](https://cloud.tencent.com/document/product/213/101709)。 | **可迁移**；价格和可售地域仍需控制台确认。 |
| 华为云 ECS C7/C7e | C7/C7e 页面可见 64 vCPU/256 GiB 与 128 vCPU/512 GiB 级别规格。见[官方 x86 规格清单](https://support.huaweicloud.com/intl/zh-cn/productdesc-ecs/ecs_01_0014.html)。 | **可迁移**；vCPU 与 physical core 必须分开记录。裸金属 physical 规格另作候选，不可当按量方案。 |
| 火山引擎 ECS g4ie/g4ale | dirty report 引用的[官方实例规格清单](https://www.volcengine.com/docs/6396/68526?lang=zh)与[官方计费项](https://www.volcengine.com/docs/6396/69812?lang=zh)可作为规格/计费入口。 | **保留为待控制台复核候选**；页面的可售地域、规格拓扑和库存不能由静态文档承诺。 |
| 百度智能云 BCC | 当前官方页显示 `bcc.c7.c64m128`（64/128）、`bcc.ca3.c64m128`（64/128）、`bcc.ca3.c128m256`（128/256）和 `bcc.c6.c128m256`。见[BCC 官方规格页](https://cloud.baidu.com/doc/BCC/s/wjwvynogv)。 | **需改写**：dirty report 的 `bcc.c7.c64m256` 未在当前官方 c7 表中出现；不能保留该规格。 |

可迁移的共同计费边界：静态规格页没有“指定地域 + 可用区 + 镜像 + 磁盘 + 网络”的固定总价；公网 IP/带宽、云盘、镜像和快照等常是独立计费项；抢占/竞价受库存和价格回收影响。采购前必须保存控制台或官方 API 返回的时间戳、地域、规格、计价粒度和配额信息。可用[阿里云计费/购买向导](https://help.aliyun.com/zh/ecs/user-guide/create-an-instance-by-using-the-wizard/)、[腾讯计费模式](https://cloud.tencent.com/document/product/213/2180)、[华为 ECS 计费](https://support.huaweicloud.com/intl/zh-cn/productdesc-ecs/ecs_01_0065.html)、[火山引擎计费项](https://www.volcengine.com/docs/6396/69812?lang=zh)和[BCC 计费](https://cloud.baidu.com/doc/BCC/s/lkb7dburb)核对。不能把“月租”数字或折扣写成 TxnOpt 预算，也不能从月价除以 30 推导 7/28 天成本。

## 4. TxnOpt 当前必须保留的边界

- `experiments/txnopt/INDEX.md` 记录 Build14-bound Attempt25 的 32-physical-core estimate（约 4,092 秒，约 1.14 小时）；原文同时标明它是 provisional local evidence，并明确 Attempt07 不授权采购、正式执行或发布。该估计是本地规划输入，不是任何云厂商报价，也不是 64/128 vCPU 的等价证明。见[TxnOpt evidence index](../../experiments/txnopt/INDEX.md)。
- Build14、Attempt25、Attempt07、Attempt24 的身份、raw/review bytes（原始/审查字节）和 sidecar（哈希旁文件）必须保持不变。当前运行说明要求 exact plan-bound Build14 wheel、隔离 Python、host resource checks（主机资源检查）和单独授权；见[TxnOpt experiment README](../../experiments/txnopt/README.md)。
- Build portability（构建可移植性）应记录为 **unverified**：现有材料证明的是特定 Build14 wheel 与绑定主机/计划的使用条件，没有跨主机重建、ABI（应用二进制接口）检查或第二主机 restore（恢复）收据。除非形成新的 clean build identity（干净构建身份）和独立复核，不得写成“云端可运行”或“跨平台通过”。
- 32 physical cores 是 campaign estimate 的资源假设；云实例页面所说的 64/128 vCPU 不能直接替换它。正式云端迁移必须重新冻结 CPU 型号、NUMA、超线程、镜像、编译器、wheel/native hash（轮子/原生哈希）和 reviewer identity（审查者身份）。

## 5. 未决事项与下一步门槛

1. 先把四份文档改写为 TxnOpt 术语版本，再由独立 reviewer（独立审查者）检查旧 Stage 路径、旧 ABI 和当前状态断言是否全部移除；本审计不代替该 review。
2. 取得三篇论文正式全文后，逐一核对 7/8/11 BKS 更新数、17-instance scope（17 实例范围）、运行次数、比较表和数据来源；未核对前只保留为“论文自报，pending”。
3. 修正云报告的百度 c7 规格，并把所有 provider recommendation（供应商推荐）降级为 provider-neutral candidate（供应商中立候选）；不保留月租结论、月价折算或采购顺序。
4. 为 Build portability 建立新的跨主机 build/restore evidence（构建/恢复证据）后，才能讨论云适配器；不能用本审计或静态规格页授权采购。
5. E 盘归档继续只读。没有双独立副本、317,936 文件全量哈希复核、干净主机恢复演练和字面删除确认前，不得删除 `E:\Reproducible-EVRPTW-archive`。

## 6. 来源索引（均为直接来源）

- Jia et al. BACO：[DOI](https://doi.org/10.1109/TCYB.2021.3069942)、[IEEE Xplore](https://ieeexplore.ieee.org/document/9409782)。
- Jia et al. confidence-based ACO：[DOI](https://doi.org/10.1109/TEVC.2022.3144142)。
- Feng et al. BHGA：[DOI](https://doi.org/10.1109/CEC60901.2024.10611987)、[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10611987)。
- WCCI-2020 CEVRP protocol：[technical report](https://mavrovouniotis.github.io/Papers/TR-EVRP-Competition.pdf)；载重相关 E-CVRP：[CEC20 paper](https://mavrovouniotis.github.io/Papers/CEC20.pdf)。
- Alibaba：[g-series specs](https://help.aliyun.com/zh/ecs/user-guide/general-purpose-instance-families)、[CPU options](https://help.aliyun.com/en/ecs/user-guide/cpu-options-of-general-purpose-instance-families)。
- Tencent：[SA5 specs](https://cloud.tencent.com/document/product/213/101709)、[billing](https://cloud.tencent.com/document/product/213/2180)。
- Huawei：[x86 ECS specs](https://support.huaweicloud.com/intl/zh-cn/productdesc-ecs/ecs_01_0014.html)、[billing](https://support.huaweicloud.com/intl/zh-cn/productdesc-ecs/ecs_01_0065.html)。
- Volcengine：[instance list](https://www.volcengine.com/docs/6396/68526?lang=zh)、[billing items](https://www.volcengine.com/docs/6396/69812?lang=zh)。
- Baidu：[BCC specs](https://cloud.baidu.com/doc/BCC/s/wjwvynogv)、[billing](https://cloud.baidu.com/doc/BCC/s/lkb7dburb)。
