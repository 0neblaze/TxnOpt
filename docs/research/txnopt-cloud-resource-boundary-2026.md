# TxnOpt 云资源边界

审计日期：2026-08-15
状态：腾讯云已选定；64 个物理核心/腾讯商品规格 128 GB 离线容量门；未采购、未接账号

## 1. 当前可使用的规划事实

- 本地代表性校准的原容量模型使用 32 physical cores（物理核）、80% scheduler
  efficiency（调度效率）和 25 个可用 core tokens（核心令牌）。
- 对同一 Attempt25 p95 observation（第 95 百分位观测）按 64 个物理核心、80% 效率
  重新计算，得到 51 个可用核心令牌和约 2,051.06 秒，即约 34.18 分钟/0.57 小时。
- 该数字来自 Build14-bound Attempt25（绑定 Build14 的 Attempt25）本地证据，只能用于
  采购前容量估计；它不是云端运行时间、SLA（服务等级协议）或报价。
- Build portability（构建可移植性）仍为 **unverified（未验证）**。当前 wheel（轮子）
  在绑定环境可用，不代表能在另一 CPU、镜像、glibc、Python 或编译器组合上运行。
- Attempt24 formal matrix（正式矩阵）未启动；本文件不授权采购或执行。

## 2. vCPU 与物理核

腾讯云把 vCPU（虚拟处理器）定义为 hyper-thread（超线程／逻辑线程）；一个物理核心
可对应两个 vCPU。因此本轮门槛是 **64 physical cores**，不是 64 vCPU。采购前必须
同时保存 CVM API 的 `CoreCount >= 64`、`ThreadPerCore`、CPU 型号、NUMA（非统一内存
访问）拓扑，以及实例内 Linux 实测的唯一 `(physical id, core id)` 数量。只满足
“64 vCPU/128 GB”不能通过资源门。

## 3. 腾讯云实例准入

腾讯云公开实例表给出的数量是 vCPU，不是物理核心。例如 C6 的 64 vCPU 规格同时配
256 GB 内存；这不能单独证明存在 64 个物理核心。CVM CPU topology API（CPU 拓扑
接口）提供 `CoreCount` 与 `ThreadPerCore`，采购候选必须以该回执和实例内实测为准。

如果普通 CVM 不能提供 `CoreCount >= 64` 的实例，则候选应转为 CBM（黑石物理服务器）
或腾讯明确给出物理核心拓扑的专用实例。腾讯商品/API 规格内存必须至少 128 GB；
Linux `MemTotal` 因平台保留略低于 128 GiB 不单独构成失败。真正的内存性能门是
Attempt27 峰值 RSS 低于目标主机实际可见内存的 80%。公开规格、库存、地域和价格
都不能替代购买时的带时间戳 API/控制台回执。

官方入口：

- CVM 实例规格：<https://cloud.tencent.com/document/product/213/11518>
- CPU topology API：<https://cloud.tencent.com/document/api/213/15753>
- hyper-threading 配置：<https://cloud.tencent.com/document/product/213/103798>
- CBM 实例规格：<https://cloud.tencent.com/document/product/386/63404>

## 4. 采购前必须重新验证

1. 在目标镜像上从新 source identity（源码身份）重建 wheel/native extension（原生
   扩展），保存编译器、依赖、CPU feature（CPU 特性）和完整哈希；
2. 运行 doctor/preflight（环境诊断/预检），验证 Linux、64 物理核、腾讯商品/API
   至少 128 GB 内存、Attempt27 RSS 余量、独占使用和 process-group cleanup
   （进程组清理）；
3. 先运行小型 portability smoke（可移植性冒烟测试）与独立 replay（重放），不能直接
   启动 2,880-run matrix（2,880 次矩阵）；
4. 保存地域、可用区、实例类型、CPU 型号、计费粒度、磁盘、网络、quota（配额）、
   inventory（库存）和含税价格；
5. 抢占式/竞价实例若可能被回收，必须有独立 checkpoint/immutable commit（检查点/
   不可变提交）策略，且不能把回收导致的缺失样本当成功证据。

## 5. 明确排除

- 不保留旧月租数字、月价除以 30 的短租推导或厂商排序；
- 不把 vCPU 写成 physical core；
- 不把约 0.57 小时写成云端承诺；
- 不保留通用 `ArchiveStore` 或把 S3 兼容语义冒充腾讯 COS 语义；
- 不采购、不启动正式矩阵、不删除 E 盘档案。

当前已实现腾讯 COS 的离线接口和 fake-client contract tests（伪客户端契约测试），但
没有账号时无法验证 bucket versioning、COMPLIANCE Object Lock、白名单、实际
`VersionId`、网络、权限、带宽或恢复。真实账号接入后必须新开 live validation slice
（真实验证切片）；通过前不得签发云迁移或 Level 1 readiness（Level 1 就绪）结论。
