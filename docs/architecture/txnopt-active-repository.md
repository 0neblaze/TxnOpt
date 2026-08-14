# TxnOpt 活仓库架构与归档边界

## 1. 目标

TxnOpt 在现有 Git history（Git 历史）中原地演进。活分支只承载事务运行时、
两个领域适配器、独立 evidence workflow（证据工作流）和只读 legacy reader
（历史读取器）。历史 Stage 实现由冻结 tag、Git bundle（Git 归档包）和受治理
外部档案恢复，不再作为新 wheel 的运行路径。

GitHub 仓库已经原地改名为 `0neblaze/TxnOpt`，并完成分支、冻结标签和全新 clone
（克隆）恢复验证。当前阶段仍不包含 formal matrix（正式矩阵）或 E 盘删除；云账号、
计费与 live bucket（真实存储桶）尚未配置，不能把本地实现测试表述成已完成云迁移。

## 2. 活模块与依赖方向

```text
txnopt_cases ───────► txnopt
txnopt_evidence ────► txnopt + txnopt_cases
txnopt_evidence historical replay ───► txnopt_legacy
txnopt ─────────────► standard library only
```

- `txnopt` 根包严格导出 `TxnRuntime`、`SearchKernel`、`Oracle`、`RunConfig`、
  `RunResult` 五个接口。
- `txnopt_cases.evrptw` 与 `txnopt_cases.rcpsp` 是领域 Adapter（适配器），不得把
  领域字段泄漏回通用运行时。
- `txnopt_evidence.runner` 只生成 raw evidence（原始证据）。
- `txnopt_evidence.reviewer` 在独立进程中重算；runner 与 reviewer 禁止互相导入。
- `txnopt_legacy` 只读，不能成为新执行的 fallback（回退路径）。

## 3. 默认状态位置

可重建状态不写入 Git 工作树：

```text
state: ~/.local/state/txnopt/<source-tree>/
cache: ~/.cache/txnopt/<source-tree>/
```

普通 `txnopt run` 配置省略 `output_root` 时，runner 会实际写入上述 state 下的
`runs/`；正式 evidence（证据）仍必须使用显式、受治理的外部 root，不允许从默认
cache 推断 formal readiness（正式就绪状态）。`source-tree` 是当前 source identity
（源码身份）绑定的 Git tree digest（Git 树摘要）；更换源码必须进入新目录。

## 4. 腾讯 COS 归档边界

活树不再包含 provider-neutral `ArchiveStore`（供应商中立归档接口）、本地归档
适配器或 S3 兼容层。当前只有两个深模块：

- `txnopt_evidence.archive`：在 WSL/Linux 上逐级拒绝符号链接和非普通文件，生成
  source inventory（源清单）并在上传前重新核验同一文件；
- `txnopt_evidence.tencent_cos`：直接实现腾讯 COS bucket contract（存储桶契约）、
  upload、multipart、commit、verify 与 restore。

COS 开启 versioning（版本控制）后，同一 key（键）可以存在多个版本。因此 key 和
SHA-256 不能单独构成对象身份。`TencentCosObjectRef` 与 `TencentCosCommitRef` 都必须
绑定 COS 返回的精确 `VersionId`；所有 HEAD、GET、retention 和恢复请求都使用该版本。
新 wire schema（线格式）为 `txnopt-tencent-cos-commit-v1`，它只引用现有 evidence
字节，不升级或重签旧 evidence schema。

不变量：

1. bucket versioning 必须为 `Enabled`；
2. bucket 必须启用默认 `COMPLIANCE` Object Lock，保留期至少 365 天；
3. 每次上传必须返回非空 `VersionId`，否则不签发引用；
4. 每个对象使用 `STANDARD` 存储层和 COS server-side encryption（服务端加密）；
5. HEAD metadata、精确版本 retention、实际 GET 字节、大小和 SHA-256 必须全部一致；
6. commit 在所有 blob 上传并验证后最后写入，commit key 同时包含 commit id 和摘要；
7. mirror 前后 source inventory 必须相同；
8. restore 写入临时目录并使用 Linux 原子 no-replace 发布，已有目标拒绝覆盖；
9. AccessKey/SecretKey 不进入 CLI、Git、manifest 或日志，只从进程环境读取；
10. 未经 live bucket（真实存储桶）契约检查、全量 verify 和恢复演练，不得把接口测试
    表述成云迁移完成。

统一命令：

```text
txnopt cloud tencent assess --protocol <protocol> --calibration <calibration> \
  --physical-cores 64 --provider-memory-gb 128
txnopt cloud tencent spec --output <spec>
txnopt cloud tencent doctor --instance-type <sku> --provider-physical-cores 64 \
  --provider-memory-gb 128 --expected-peak-rss-bytes <bytes> --output <receipt>
txnopt cloud tencent bundle --destination <deployment> --build-manifest <build> \
  --wheel <wheel> --source-manifest <source> --native-attestation <native> \
  --uv-lock <uv.lock> --toolchain-lock <toolchain> --plan-manifest <formal-plan>
txnopt cloud tencent bundle-verify --manifest <deployment>/bundle-receipt.json
txnopt archive inventory <source>
txnopt cloud tencent cos mirror <source> --cos-bucket <bucket-appid> \
  --cos-region <region> \
  --cos-prefix <prefix> --commit-id <id>
txnopt cloud tencent cos verify --cos-bucket <bucket-appid> --cos-region <region> \
  --cos-prefix <prefix> --commit-id <id> --commit-key <key> \
  --commit-version-id <version> --commit-sha256 <sha256> --commit-size <bytes> \
  --commit-retain-until <timestamp>
```

`txnopt[tencent]` 安装腾讯官方 Python SDK。凭据环境变量仅为
`TENCENTCLOUD_SECRET_ID`、`TENCENTCLOUD_SECRET_KEY` 和可选
`TENCENTCLOUD_SESSION_TOKEN`。对象锁是否已由腾讯云开放、bucket 是否可用、
Build portability（构建可移植性）以及正式矩阵执行都必须等真实账号提供后单独验证。

云主机门要求 **至少 64 个 physical cores（物理核心）**，并要求腾讯商品/API
明确给出至少 **128 GB** 内存。64 vCPU 不是 64 个物理核心的替代证据；腾讯 API
必须返回 `CoreCount=64`、`ThreadPerCore=1`，Linux `/proc/cpuinfo` 拓扑也必须达到
64 个物理核心。Linux 可见 `MemTotal` 只记录、不因平台保留略低于 128 GiB 而失败；
真正的内存门是与所选 producer（生产器）绑定的校准峰值 RSS 低于实际可见内存的
80%。Build16 只允许 Attempt26/27；Build18 只允许 Attempt28/29。Build17
validation candidate（验证候选）未闭合正式计划入口，不能作为正式 producer。

## 5. 不可变恢复边界

重构前基线为 commit `5901a339ae0fa0fe490a67d2ce9d995a530d110b`、tree
`8bd04d72f95d8e94d00750ddee899369acdc1400`，由 annotated tag
（带说明标签）`txnopt-pre-refactor-v1` 和完整 Git bundle 保护。当前 TxnOpt
分支与 `txnopt-pre-refactor-v1`、`stage052-legacy-freeze-v1` 已推送到远端，
并已通过全新 clone（克隆）、tag 解析、`git fsck` 与受保护哈希复核；后续活树
删除不改变这两个恢复对象。

以下对象保持逐字节不变：

- Build14；
- Attempt24 plan（计划）及预绑定 expected identities（预期身份）；
- Attempt25 raw/review 与校准 manifest；
- Attempt07 pre-cloud gate（云前门）；
- `experiments/txnopt`、`formal`、`legacy` 中已经封存的 manifests、reviews 与
  sidecars（哈希旁文件）；
- `stage052-legacy-freeze-v1` 指向的历史标签和历史绝对路径。

活动测试按 `tests/core`、`tests/cases`、`tests/evidence`、`tests/workflows`、
`tests/formal`、`tests/legacy` 分类。历史收据若绑定已经从活树删除的路径，验证器
从冻结 tag 读取原字节，不修改收据或把旧工具重新装入 wheel。`tools/` 仅保留
构建安装包之前必须执行的 `native_build_attestation.py`。

## 6. E 盘与未来云归档

`E:\Reproducible-EVRPTW-archive` 当前仍是受保护档案，不是 cache。云端迁移必须先
完成两个独立可读副本、全量路径/大小/SHA-256 复核、代表性恢复演练和两名独立
reviewer（审查者）收据。只有收到字面确认
`确认删除 E:\Reproducible-EVRPTW-archive` 后，才允许精确删除该目录；不得格式化
E 盘，也不得触碰 `$RECYCLE.BIN` 或 `System Volume Information`。
