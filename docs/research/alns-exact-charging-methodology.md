# ALNS、精确充电子问题与 Branch-Price-and-Cut 方法说明

## 1. 问题边界

本实现求解带时间窗的电动车路径问题 EVRP-TW。距离为未取整的二维欧氏距离
`EUC_2D`，行驶时间为距离除以平均速度，行驶能耗为距离乘单位距离能耗率。
车辆在仓库满电出发，只能在 Schneider 充电站进行 full recharge（满充），充电时间与
补充电量线性相关。当前版本不声称支持 partial recharge（部分充电）、nonlinear
charging（非线性充电）、充电站容量或排队约束。

阶段 0 的历史目标为最小化总行驶距离。阶段 1 起，正式统一目标改为
lexicographic objective（字典序目标）
`(车辆数, 总距离, 总充电时间, 充电次数)`；车辆数具有绝对优先级。历史
`objective_value` 仅保留为总距离兼容字段，不再代表正式目标。

## 2. 主问题与充电子问题分解

ALNS-based matheuristic（基于自适应大邻域搜索的数学启发式）维护客户路径序列

\[
S=(s_1,\ldots,s_m),\qquad s_k=(i_1,\ldots,i_{n_k}).
\]

主问题决定客户分配、车辆路径数和每条路径的客户访问顺序。固定一条客户序列后，
`solve_exact_charging(instance, customer_order)` 决定充电站插入序列。它返回完整路径、
可行性、距离、能耗、充电量、充电时间、标号数、剪枝数和失败原因。任何不可行子问题
都会使对应 ALNS 候选不可接受，不通过 penalty（惩罚）伪装成可行解。

### 2.1 精确充电标号

一个标号为

\[
L=(p,i,t,b,d,e,q,h,P),
\]

其中 `p` 是已服务客户数，`i` 是当前位置，`t` 是时间，`b` 是剩余电量，`d` 是距离，
`e` 是总能耗，`q` 是总充电量，`h` 是充电时间，`P` 是完整节点序列。扩展只允许到
下一个固定客户、任一合法充电站，或在客户全部服务后返回仓库。

行驶扩展为

\[
b'=b-rd_{ij},\qquad t'=\max\{a_j,t+d_{ij}/v\}.
\]

客户节点再增加服务时间；充电站节点执行

\[
q'=Q-b',\qquad h'=gq',\qquad b'\leftarrow Q.
\]

若电量为负或到达超过 due date（最迟服务时间），标号立即删除。同一 `(p,i)` 状态下，
若标号 A 的时间和距离不大于 B、电量不小于 B，且至少一项严格更优，则 A 支配 B。
因此算法枚举 full-recharge linear model（满充线性模型）下全部未支配的站点插入结构，
不是 greedy charging（贪心充电）。

## 3. ALNS 设计

ALNS 使用 `random`、`worst`、`related` 三个 destroy operators（破坏算子），以及
`greedy`、`regret2`、`energy` 三个 repair operators（修复算子）。所有插入候选均调用
精确充电子问题。接受准则为 simulated annealing（模拟退火）；算子权重根据接受、改善和
产生全局最佳解的奖励自适应更新。候选车辆数减少时必然接受，车辆数增加时始终拒绝；
车辆数相同时才对距离恶化使用模拟退火。运行日志记录调用数、接受数、改善数、最终权重、接受/
拒绝次数、首次可行时间、最佳解时间以及精确子问题调用与标号统计。

## 4. Branch-Price-and-Cut 与双向标号

小规模精确对照使用 set-partitioning master problem（集合划分主问题）：

\[
\min \sum_{r\in\Omega} c_r x_r,
\qquad
\sum_{r\in\Omega} a_{ir}x_r=1\quad\forall i,
\qquad x_r\in\{0,1\}.
\]

并加入容量导出的 fleet lower-bound cut（车队下界切割）：

\[
\sum_r x_r\ge
\left\lceil\frac{\sum_i q_i}{C}\right\rceil.
\]

`generate_columns_bidirectionally()` 分别生成 forward labels（前向标号）和 backward
labels（后向标号），在客户集合不相交且容量可行时连接，再调用精确充电子问题验证完整
route column（路径列）。根节点从 singleton columns（单客户列）开始，通过 LP duals
（线性规划对偶变量）加入负约化成本列。为保证当前小规模版本的精确性，进入分支树后使用
双向标号穷举得到的完整列池；分支变量为路径列变量。每个节点都保留容量下界切割。
每列同时记录车辆、距离、充电时间和充电次数贡献；完整列池最终使用精确字典序集合划分
确认四级 incumbent。内部 scalar search bounds 只用于搜索，不能解释为正式目标差距。

该实现最多支持 8 个客户，超过限制会 fail fast。它是真实的小规模列生成、分支和切割
实现，但不是适用于 100-customer 的 production-grade BPC（生产级分支定价切割）。
大规模 Ryan-Foster branching（Ryan-Foster 分支）、动态双向 ESPPRC pricing（带资源
约束的基本最短路动态定价）尚未实现，不能把当前结果外推到中大规模。

## 5. 统一可行性验证

`validate_routes()` 对所有方法统一检查：仓库起终点、仓库不得位于路径内部、客户恰好
访问一次、载重、连续时间传播、时间窗、电量非负、站点合法性、满充量、电池上限、线性
充电时间、行驶能耗和声明目标值。验证器独立重算距离、能耗、充电量和充电时间。未通过
的解标记为 `invalid`，不进入可行解目标值统计。

## 6. Benchmark 与电池界

正式目录包含 Schneider 的 92 个实例：12 个 5-customer、12 个 10-customer、12 个
15-customer 和 56 个 100-customer 实例。完整审计见
`experiments/summaries/schneider_instance_catalog.csv`。

对客户 `i`，令 `R` 为仓库与充电站集合，定义

\[
B_{lb}=\max_i\{\min_{u\in R}e_{ui},\min_{v\in R}e_{iv}\},
\]

\[
B_{struct}=\max_i\min_{u,v\in R}(e_{ui}+e_{iv}).
\]

`B_lb` 是不可避免的单段能耗下界；`B_struct` 是每个客户能够位于两个可充电节点之间的
客户级必要结构下界，不是完整可行解的充分条件。正式实例使用原始 `B_exp=Q`；Stress
实例使用 `B_exp=1.05 B_struct`，并先重新通过结构审计。压力系数为
`B_exp/B_struct`。

Primary 预先选择 12 个实例：`c101C5/r105C5/rc105C5`、
`c104C10/r103C10/rc102C10`、`c106C15/r105C15/rc103C15` 和
`c101_21/r101_21/rc101_21`，同时覆盖 5/10/15/100 customers 以及 clustered（聚类）、
random（随机）和 random-clustered（随机聚类）三类。BPC 只对 5-customer 子集提供
可证明最优对照，其他规模明确记录为 `not_applicable`。该代表性子集不等同于完整 92
实例统计，选择规则是在每个规模固定覆盖 C/R/RC 各一个实例，而非按算法结果筛选。

## 7. 可复现命令

原 Week 5 runner 属于学校项目边界，已从 publication branch（发表分支）移除。
其科学方法进入 Stage 0 frozen baseline（冻结基线），公开复现入口为：

```bash
uv sync --all-groups
uv run python -m evrptw.experiments.stage00_baseline run \
  --config configs/stage00_baseline.toml \
  --output-dir results/stage00 \
  --baseline-dir experiments/baselines/stage00
```

Stage 0 的冻结结果位于 `experiments/baselines/stage00/`；新运行的 raw、solution
和环境记录写入 ignored（被 Git 忽略）的 `results/stage00/`。
