# Stage 3.2：可审计缓存与增量传播

Stage 3.2 的实现入口是 `stage032_cache_incremental`。它是 opt-in
cache（缓存）与 incremental propagation（增量传播）层；`solve_alns` 未传入启用的
`CacheIncrementalConfig` 时，保持 Stage 0–3.1 的求解路径和结果语义。

## 固定协议

- profile：`stage02_constraint_guided`；30 秒、1000 iterations（迭代）、1 thread。
- smoke：`c101C5`、`r105C5`、`rc105C5`、`c101_21`、`r101_21`、`rc101_21` ×
  seeds `2014/2015/2016`，共 18 runs。
- formal：Stage 0 的 12 instances × 3 seeds，共 36 runs。formal 启动前必须验证
  Stage 3.2 smoke review，以及 Stage 3.1 formal 的 raw manifest、sidecar 和 trusted
  review manifest；后者必须为 `READY_FOR_STAGE03_2`。
- route cache（路线缓存）只在单次 `solve_alns` 内存在。默认是 4096 entries、64 MiB、
  LRU；Stage 3.2 evaluator 不再保留第二份无界 exact-result 字典。实际配置、instance
  hash、objective schema version 和 charging configuration version 都写入
  trace/result/environment evidence。
- cache key 由 `instance_hash`、ordered canonical customer sequence、charging
  configuration version/hash 和 `OBJECTIVE_SCHEMA_VERSION` 组成。feasible 与明确
  infeasible exact result 都可以缓存；screening rejection 不冒充 exact result。
- station reachability 使用 depot/station safe-node bitset（位集），并为 customer
  origin 保存可到达 safe-node closure；reviewer 通过独立 optimistic frontier 重算。
  relocate/swap 使用增量 distance、forward/backward time-window propagation；无法安全
  复用时返回并记录 `fallback`，不静默降级。
- 不实现 Stage 3.3 的 interruptible exact solver、fixed-work/wall-clock diagnostic，
  也不实现 Stage 3.4 parallel evaluation。

## 运行和复核

实现提交并保持主仓库 clean 后运行：

```text
uv run python -m evrptw.experiments.stage032_cache_incremental \
  --config configs/stage032_cache_incremental.toml \
  --output-dir results/stage03.2_cache_incremental_attempt01 \
  --scope smoke

uv run python -m evrptw.experiments.stage032_cache_incremental_review \
  --run-dir results/stage03.2_cache_incremental_attempt01
```

正式测量使用新的 `attemptNN` 或 `rerunNN` label，不能复用旧目录：

```text
results/stage03.2_cache_incremental_attemptNN/
  control/         # metadata, config, manifest and manifest sidecar
  <instance>/<seed>/
    *_raw_*.json
    *_solution_*.json
    *_trace_*.json
    *_events_*.parquet
    *_route_dictionary_*.parquet
    *_screening_checks_*.parquet
    *_diagnostic_*.parquet
    *_environment_*.json
    *_failure_*.json  # only when applicable
  review/          # generated only by independent replay reviewer
```

runner 只写 ignored `results/`。每个 run 同时记录 source/config/instance/environment
hash、Stage 0 manifest hash、主仓库 revision/dirty、两个 reference repository 的
revision/dirty、硬件和峰值 RSS。失败、partial trace 和 deadline boundary 都保留
`failures/` evidence，summary 不覆盖历史文件。

reviewer 的结论只来自 raw solution、raw trace、event log、environment 和 manifest。
它会重算 validator/objective，检查 screening→cache→exact 顺序、cache digest 和
eviction 生命周期、changed/unchanged route 语义、增量传播与 full propagation、
station bitset、deadline、candidate vehicle-first acceptance 以及 `ALNSResult` 对账。
summary 只有在全部 raw replay gate 通过后才可发布。

## 产物与命名

canonical label 必须为：

```text
stage03.2_cache_incremental_attemptNN
stage03.2_cache_incremental_rerunNN
```

canonical logical path 为 `results/<run_label>/<instance>/<seed>/`，文件名采用
`<stage_id>_<component>_<attempt_or_rerun>_<artifact_type>[_<instance>_<seed>].ext`。
`experiments/registries/stage03.2_artifact_registry.csv` 登记每一项 raw、solution、
events、trace、environment、failure、manifest、review 和 tracked summary；旧路径
只通过 `stage03_legacy_path_map.csv` 兼容映射保留，不移动、不覆盖。

## 进入 Stage 3.3 的门槛

只有 formal 36-run 的 independent review 完成且 review manifest 状态为
`READY_FOR_STAGE03_3` 才能进入 Stage 3.3。必须同时满足：

1. coverage 完整且唯一，所有 solution validator/objective replay 通过；
2. trace exact/cache/precomputed、cache lifecycle、incremental、operator、candidate、
   deadline 与 `ALNSResult` 全部对账；
3. C5 objective key 不劣化，100-customer runs 100% feasible 且 vehicle count 不增加；
4. raw manifest、hash、source/config/reference provenance 和 Stage 3.1 formal 对比通过；
5. 报告中的 exact-call reduction 只能标为 Stage 3.2 cache evidence，不能宣称 Stage
   3.3 acceleration（加速）。
# Stage 3.2 storage note

Stage 3.2 新运行必须使用 `stage03.2_cache_incremental_attemptNN` 或 `rerunNN`，并
通过共享 `ArtifactBundleWriter` 保存 bounded route cache、station reachability、
incremental propagation、fallback、eviction、deadline 和失败证据。Cache lookup 的
每个结果只写一条 critical event；route sequence 只在 route dictionary 保存一次。

历史 Stage 3.2 raw evidence 不物理迁移。独立 reviewer 通过 `ArtifactReader` 的
Parquet/legacy 双路径，重算 manifest、schema、checksum、cache lifecycle、route
change status、station bitset、full propagation 对账、validator/objective 和
raw-to-summary。只有 review 状态为 `READY_FOR_STAGE03_3` 才能进入 Stage 3.3；该
状态不代表已经完成 Stage 3.3 加速诊断。
