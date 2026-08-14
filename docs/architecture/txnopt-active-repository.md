# TxnOpt 活仓库架构与归档边界

## 1. 目标

TxnOpt 在现有 Git history（Git 历史）中原地演进。活分支只承载事务运行时、
两个领域适配器、独立 evidence workflow（证据工作流）和只读 legacy reader
（历史读取器）。历史 Stage 实现由冻结 tag、Git bundle（Git 归档包）和受治理
外部档案恢复，不再作为新 wheel 的运行路径。

当前阶段不包含云采购、formal matrix（正式矩阵）、GitHub 仓库改名或 E 盘删除。
这些动作仍分别需要显式授权。

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

正式 evidence（证据）必须使用显式、受治理的外部 root，不允许从默认 cache
推断 formal readiness（正式就绪状态）。`source-tree` 是当前 source identity
（源码身份）绑定的 Git tree digest（Git 树摘要）；更换源码必须进入新目录。

## 4. ArchiveStore 接缝

`ArchiveStore` 是 `txnopt_evidence` 内部 provider-neutral port（供应商中立端口），
不从 `txnopt` 根包导出。当前唯一生产实现是
`LocalFilesystemArchiveStore`；云厂商确定后，新的 object store adapter
（对象存储适配器）必须复用相同契约测试。

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

统一命令：

```text
txnopt archive inventory <source>
txnopt archive mirror <source> --store <root> --commit-id <id>
txnopt archive verify --store <root> --commit-id <id> \
  --commit-sha256 <sha256> --commit-size <bytes>
txnopt archive restore --store <root> --commit-id <id> \
  --commit-sha256 <sha256> --commit-size <bytes> --destination <path>
```

## 5. 不可变恢复边界

重构前基线为 commit `5901a339ae0fa0fe490a67d2ce9d995a530d110b`、tree
`8bd04d72f95d8e94d00750ddee899369acdc1400`，由本地 annotated tag
（带说明标签）`txnopt-pre-refactor-v1` 和完整 Git bundle 保护。

以下对象保持逐字节不变：

- Build14；
- Attempt24 plan（计划）及预绑定 expected identities（预期身份）；
- Attempt25 raw/review 与校准 manifest；
- Attempt07 pre-cloud gate（云前门）；
- `experiments/txnopt`、`formal`、`legacy` 中已经封存的 manifests、reviews 与
  sidecars（哈希旁文件）；
- `stage052-legacy-freeze-v1` 指向的历史标签和历史绝对路径。

旧 tracked files（受版本控制文件）只有在当前 TxnOpt 分支与两个冻结标签完成
远端推送，并从全新 clone（克隆）验证恢复后，才允许从活 HEAD 删除。

## 6. E 盘与未来云归档

`E:\Reproducible-EVRPTW-archive` 当前仍是受保护档案，不是 cache。云端迁移必须先
完成两个独立可读副本、全量路径/大小/SHA-256 复核、代表性恢复演练和两名独立
reviewer（审查者）收据。只有收到字面确认
`确认删除 E:\Reproducible-EVRPTW-archive` 后，才允许精确删除该目录；不得格式化
E 盘，也不得触碰 `$RECYCLE.BIN` 或 `System Volume Information`。
