# TxnOpt 云资源边界

审计日期：2026-08-14  
状态：provider-neutral（供应商中立），未采购

## 1. 当前可使用的规划事实

- 本地代表性校准所用资源假设约为 32 physical cores（物理核）。
- 当前 provisional estimate（暂定估计）约 4,092.47 秒，即约 1.14 小时。
- 该数字来自 Build14-bound Attempt25（绑定 Build14 的 Attempt25）本地证据，只能用于
  采购前容量估计；它不是云端运行时间、SLA（服务等级协议）或报价。
- Build portability（构建可移植性）仍为 **unverified（未验证）**。当前 wheel（轮子）
  在绑定环境可用，不代表能在另一 CPU、镜像、glibc、Python 或编译器组合上运行。
- Attempt24 formal matrix（正式矩阵）未启动；本文件不授权采购或执行。

## 2. vCPU 与物理核

云厂商规格中的 vCPU（虚拟处理器）通常是逻辑处理器，不能直接替换 32 physical
cores。采购前必须记录 CPU 型号、SMT/超线程、NUMA（非统一内存访问）拓扑、实际
physical-core count（物理核数）、内存、磁盘、镜像和可用区。只满足“64 vCPU”不等于
满足 TxnOpt 的 32 物理核资源门。

## 3. 可核验的候选规格入口

以下仅说明官方规格页存在相应 vCPU/内存级别，不代表库存、地域、价格或适配性：

| 厂商 | 官方静态规格中可见的例子 | 限制 |
|---|---|---|
| 阿里云 ECS g8a | 64 vCPU/256 GiB、128 vCPU/512 GiB | 默认超线程；需控制台复核 CPU 与库存 |
| 腾讯云 CVM SA5 | 64 vCPU/256 GB、128 vCPU/576 GB | 可售地域与实时价格需控制台确认 |
| 华为云 ECS C7/C7e | 64 vCPU/256 GiB、128 vCPU/512 GiB 级别 | vCPU 不能写成物理核 |
| 火山引擎 ECS | 官方实例清单提供 64/128 vCPU 级别入口 | 需重新确认具体规格、地域与拓扑 |
| 百度智能云 BCC | 当前页含 `bcc.c7.c64m128`、`bcc.ca3.c64m128`、`bcc.ca3.c128m256`、`bcc.c6.c128m256` | 旧报告的 `bcc.c7.c64m256` 未在当前 c7 表中出现，已排除 |

官方入口与复核日期记录在
[Windows salvage source audit](txnopt-windows-salvage-source-audit.md)。静态网页不能代替
采购时的带时间戳 API/控制台回执。

## 4. 采购前必须重新验证

1. 在目标镜像上从新 source identity（源码身份）重建 wheel/native extension（原生
   扩展），保存编译器、依赖、CPU feature（CPU 特性）和完整哈希；
2. 运行 doctor/preflight（环境诊断/预检），验证 Linux、32 物理核、至少 128 GiB
   内存、独占使用和 process-group cleanup（进程组清理）；
3. 先运行小型 portability smoke（可移植性冒烟测试）与独立 replay（重放），不能直接
   启动 2,880-run matrix（2,880 次矩阵）；
4. 保存地域、可用区、实例类型、CPU 型号、计费粒度、磁盘、网络、quota（配额）、
   inventory（库存）和含税价格；
5. 抢占式/竞价实例若可能被回收，必须有独立 checkpoint/immutable commit（检查点/
   不可变提交）策略，且不能把回收导致的缺失样本当成功证据。

## 5. 明确排除

- 不保留旧月租数字、月价除以 30 的短租推导或厂商排序；
- 不把 vCPU 写成 physical core；
- 不把约 1.14 小时写成云端承诺；
- 不假装已有任何 cloud ArchiveStore adapter（云归档适配器）；
- 不采购、不启动正式矩阵、不删除 E 盘档案。

云厂商确定后，必须另开 implementation slice（实施切片），对其 object store adapter
（对象存储适配器）运行与本地适配器相同的契约、故障、完整恢复和不可覆盖测试。
