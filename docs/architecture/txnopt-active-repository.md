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

## 4. ArchiveStore 接缝

`ArchiveStore` 是 `txnopt_evidence` 内部 provider-neutral port（供应商中立端口），
不从 `txnopt` 根包导出。`LocalFilesystemArchiveStore` 使用 WSL/Linux 的 POSIX
no-follow（禁止跟随链接）与原子 no-replace（禁止替换）原语，当前不宣称原生
Windows portability（可移植性）。`S3ArchiveStore` 是注入 S3 client（客户端）的
云端 Adapter（适配器），可由 AWS S3 或通过 live contract（真实契约）验证的
S3-compatible provider（S3 兼容供应商）使用；它不把任何供应商类型泄漏到 port。

接口：

```text
put_blob(source, expected_sha256, expected_size)
open(ref)
head(ref)
publish_commit(commit, expected_absent=True)
verify_commit(ref)
```

不变量：

1. blob 按 SHA-256 content address（内容地址）保存，写入过程中不可见；
2. 大小或摘要不符时不发布；
3. 已存在的同摘要 blob 必须重新验证；
4. `txnopt-archive-commit-v1` 只引用现有字节、路径、大小与摘要，不升级、重签或
   解释旧 evidence schema（证据模式）；
5. commit marker（提交标记）最后发布，默认同名冲突即失败；
6. verify 必须重算 commit 和所有引用 blob；
7. restore 使用临时目录，目标已存在时拒绝覆盖，完成前不暴露最终目录；
8. 路径穿越、符号链接、截断写入和摘要漂移均 fail closed（失败即停止）。
9. mirror（镜像）前后源树摘要必须一致；崩溃遗留的不可见 staging（暂存）文件由
   下一次持有独占 writer lock（写入锁）的适配器实例清理；
10. CLI 的 archive store 必须是 Git 工作树外的显式绝对路径，且不得与镜像源重叠。
11. inventory/mirror（清单/镜像）逐级拒绝源目录符号链接和非普通文件；`open` 在同一
    文件句柄上先核验大小与 SHA-256，FIFO（命名管道）等对象不得阻塞读取；
12. verify/restore（验证/恢复）以只读模式打开 store，不创建 root、锁或 staging；恢复
    目标不得位于 store 内部。
13. S3 bucket 必须启用 versioning（版本控制）、COMPLIANCE Object Lock（合规对象锁）
    和不少于声明下限的默认保留期；仅有可重算 sidecar（哈希旁文件）不构成信任根。
14. S3 blob 与 commit marker 均使用 `If-None-Match: *` 条件创建；同名 key（键）不得
    被普通 PUT 覆盖。大文件只在 conditional multipart completion（条件分段完成）后
    可见，失败的未完成分段不属于 evidence。
15. S3 `open`/`head`/`verify_commit` 下载实际对象并重算字节数与 SHA-256；不能仅信任
    ETag 或用户 metadata（元数据）。commit marker 仍在所有 blob 后最后发布。

统一命令：

```text
txnopt archive inventory <source>
txnopt archive mirror <source> --store <root> --commit-id <id>
txnopt archive verify --store <root> --commit-id <id> \
  --commit-sha256 <sha256> --commit-size <bytes>
txnopt archive restore --store <root> --commit-id <id> \
  --commit-sha256 <sha256> --commit-size <bytes> --destination <path>
txnopt archive mirror <source> --s3-bucket <bucket> --s3-prefix <prefix> \
  --s3-region <region> [--s3-endpoint-url https://<endpoint>] --commit-id <id>
```

S3 凭据只从 boto3 standard credential provider chain（标准凭据提供链）读取；CLI 不
接受 access key（访问密钥）参数。首次云端副本必须使用可即时读取的 storage class
（存储层）完成全量 verify 与恢复演练；若后续转入 Deep Archive（深度归档），由 bucket
lifecycle（存储桶生命周期）在验证后执行，不能在对象仍不可读时签发迁移 PASS。

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
