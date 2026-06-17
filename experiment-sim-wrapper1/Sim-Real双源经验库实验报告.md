# Sim-Real 双源经验库 MVP 实验报告

## 1. 实验目标

本次 MVP 实验验证以下链路是否已经工程可行：

```text
pseudo-real result.json
→ real memory 导入
→ simulation memory 合成/对齐
→ sim-real pair
→ gap 计算
→ gap-aware / risk-aware retrieval
→ sandbox calibration
→ candidate plan scoring
→ Markdown / JSON 报告
```

本实验不是最终完整论文实验。它的目标是证明双源经验库的核心模块已经可运行、可解释、可回归。

## 2. 实验配置

主配置文件：

```text
configs/mvp_dual_source_gap_critic_u3_u4_v1.json
```

U4-3 专项配置文件：

```text
configs/mvp_dual_source_gap_critic_u4_3_v1.json
```

运行命令：

```bash
python -B tools/run_mvp_dual_source_pipeline.py \
  --config configs/mvp_dual_source_gap_critic_u3_u4_v1.json
```

U4-3 专项运行命令：

```bash
python -B tools/run_mvp_dual_source_pipeline.py \
  --config configs/mvp_dual_source_gap_critic_u4_3_v1.json
```

回归命令：

```bash
python -B tools/smoke_mvp_dual_source_pipeline.py
python -B tools/smoke_mvp_dual_source_u4_3.py
```

输出目录：

```text
results/mvp_dual_source_gap_critic_u3_u4_v1/
analysis/mvp_dual_source_report.md
```

## 3. 输入数据

当前 MVP 使用 8 条 pseudo-real 输入：

| 来源 | 条件 | 说明 |
| --- | --- | --- |
| `results/demo_u3_4_direct_memory_video/result.json` | U3-4 | 夹爪/滑移相关异常 |
| `results/demo_u4_2_direct_memory_video/result.json` | U4-2 | 放置/运输相关异常 |
| `results/demo_u4_2_hier_memory_video/result.json` | U4-2 | 层级记忆方法输出 |
| `results/u5_all5_direct_smoke_v1/.../U5-1/.../result.json` | U5-1 | 路径/安全类异常 |
| `results/u5_all5_direct_smoke_v1/.../U5-2/.../result.json` | U5-2 | 路径/安全类异常 |
| `results/u5_all5_direct_smoke_v1/.../U5-3/.../result.json` | U5-3 | 路径/安全类异常 |
| `results/u5_all5_direct_smoke_v1/.../U5-4/.../result.json` | U5-4 | 路径/安全类异常 |
| `results/u5_all5_direct_smoke_v1/.../U5-5/.../result.json` | U5-5 | 路径/安全类异常 |

U4-3 已使用现有 batch 数据补齐。输入来自：

```text
results/u4_3condition_4method_10trials_rolling_v2/U4/U4-3/
```

该目录包含 4 个方法各 10 个 trial，共 40 条 `result.json`。

## 4. 生成产物

| 文件 | 内容 |
| --- | --- |
| `real_memory_snapshot.json` | 8 条 real/pseudo-real memory |
| `sim_real_memory_snapshot.json` | 8 条 real + 8 条 synthetic sim memory |
| `gap_report.json` | 8 个 sim-real pair 和 gap |
| `candidate_scoring_report.json` | U3-4 / U4-2 检索、评分、calibration |
| `analysis/mvp_dual_source_report.md` | 面向阅读的汇总报告 |

实际运行结果：

```text
real_entry_count = 8
sim_entry_count = 8
pair_count = 8
```

U4-3 专项运行结果：

```text
real_entry_count = 40
sim_entry_count = 40
pair_count = 40
```

## 5. Pair 与 Gap 结果

本次共生成 8 个 pair：

```text
U3-4: 1 pair
U4-2: 2 pairs
U5-1: 1 pair
U5-2: 1 pair
U5-3: 1 pair
U5-4: 1 pair
U5-5: 1 pair
```

关键 gap 类型：

| 条件 | outcome gap | 说明 |
| --- | --- | --- |
| U3-4 | `sim_success_real_fail` | 仿真成功但 pseudo-real 失败，候选应降权 |
| U4-2 | `matched_success` | 有真实成功先验，但仍存在 pose/contact gap |
| U4-2 | `sim_success_real_fail` | 同条件下也有失败经验，可作为风险证据 |
| U5-1 到 U5-5 | 多数为 `sim_success_real_fail` | 路径/安全类异常存在明显仿真-真实差异 |

U3-4 的关键证据：

```text
gap_score = 0.8655
outcome_gap.type = sim_success_real_fail
contact_mismatch = true
pose_estimation_gap ≈ 0.0524
uncertainty = 0.47
```

U4-2 的关键证据：

```text
matched_success gap_score = 0.5427
sim_success_real_fail gap_score = 0.8853
contact_mismatch = true
```

解释：

1. U3-4 是典型“仿真看起来可行，但真实/伪真实失败”的风险案例。
2. U4-2 同时有成功和失败证据，因此不能简单认为某个恢复计划一定可靠，需要按检索证据和 gap 分解评分。
3. U5 系列说明双源经验库对路径/安全异常也能记录 sim-real mismatch，但本次 MVP 没有对 U5 做候选计划评分。

U4-3 专项 gap：

```text
pair_count = 40
matched_success = 20
sim_success_real_fail = 20
contact_mismatch = 0
```

U4-3 的结论是：同一候选恢复族既有大量成功证据，也有大量失败证据；仅按普通 top-k 检索会偏向成功样本，因此需要显式风险检索分支。

## 6. Candidate Scoring 结果

候选计划：

```text
move-grasp
gripper-action state=1
vertical-grasp
```

### 6.1 U3-4

评分结果：

```text
candidate_score = 0.4148
decision = rewrite_recommended
failure_overlap_risk = 0.4286
gap_uncertainty = 0.3643
critic_risk = 0.0
calibration_applied = true
calibration_confidence = 0.7077
```

检索证据：

| source | validation | success | score | gap_uncertainty | role |
| --- | --- | --- | ---: | ---: | --- |
| simulation | simulation_validated | true | 1.77 | 0.0 | synthetic_sim_prior |
| real | real_executed | false | 1.327 | 0.85 | sim_real_gap_memory |

结论：

```text
U3-4 中候选计划与历史失败动作存在重叠，并且存在 sim_success_real_fail gap。
因此系统建议重写，而不是直接信任仿真成功经验。
```

这正是双源经验库区别于 sim-only memory 的核心收益：仿真成功不能自动等于真实可执行。

### 6.2 U4-2

评分结果：

```text
candidate_score = 0.7216
decision = prefer
failure_overlap_risk = 0.0
gap_uncertainty = 0.15
critic_risk = 0.0
calibration_applied = true
calibration_confidence = 0.6843
```

检索证据：

| source | validation | success | score | gap_uncertainty | role |
| --- | --- | --- | ---: | ---: | --- |
| real | real_executed | true | 1.822 | 0.35 | sim_real_gap_memory |
| simulation | simulation_validated | true | 1.775 | 0.0 | synthetic_sim_prior |
| simulation | simulation_validated | true | 1.75 | 0.0 | synthetic_sim_prior |
| real | real_executed | false | 1.307 | 0.85 | sim_real_gap_memory |

结论：

```text
U4-2 有真实成功先验，且候选计划与失败动作重叠较低。
系统倾向采用候选计划，但仍保留 sandbox calibration。
```

### 6.3 U4-3

候选计划：

```text
detect-object
create-grasp
move-pregrasp
move-grasp
gripper-action state=1
vertical-grasp
move-pregrasp
```

评分结果：

```text
candidate_score = 0.5292
decision = allow
failure_overlap_risk = 0.8571
gap_uncertainty = 0.7286
critic_risk = 0.0
support_evidence_count = 6
risk_evidence_count = 6
```

解释：

```text
U4-3 的成功经验能支持该候选计划，但失败经验与候选动作高度重叠。
因此系统不应直接 prefer，而是给出 allow：可以继续进入沙盒推演或 critic 检查，但不应无条件执行。
```

## 7. Sandbox Calibration 结果

U3-4 校准：

```text
object_pose_bias = [0.006993, 0.04, -0.016422]
slip_risk_bias = 1.0
contact_success_bias = -1.0
calibration_confidence = 0.7077
source_gap_count = 1
```

U4-2 校准：

```text
object_pose_bias = [0.04, 0.011142, -0.013343]
slip_risk_bias = 0.5222
contact_success_bias = -1.0
calibration_confidence = 0.6843
source_gap_count = 2
```

解释：

1. 系统已经能从 gap memory 中提取 pose/contact/outcome 证据。
2. calibration 被明确写入报告和 experience 字段。
3. 当前校准是规则加权平均，不是最终真实系统辨识模型。

## 8. Smoke 回归结果

已通过：

```bash
python -B tools/smoke_mvp_dual_source_pipeline.py
python -B tools/smoke_mvp_dual_source_u4_3.py
python -B tools/smoke_dual_source_scoring.py
python -B tools/smoke_real_memory_batch_import.py
```

smoke 覆盖：

1. MVP pipeline 可在 `/tmp` 输出完整产物。
2. real memory 至少导入 5 条。
3. sim-real pair 至少生成 5 条。
4. gap-aware / risk-aware 检索能影响候选评分。
5. scoring report 中包含 U3-4、U4-2 和 U4-3 专项结果。
6. sandbox calibration 至少在一个条件上被应用。
7. U4-3 风险分支能拉入失败样本，并把候选评分从过度乐观降为审慎 `allow`。

## 9. 与基线方法的关系

当前 MVP 离线流水线已经为后续完整对照实验准备了以下方法名：

```text
direct_llm_weak
sim_only_weak
sim_memory_weak
hierarchical_memory_weak
dual_source_gap_memory
dual_source_gap_critic
```

但本次报告尚未给出完整 success rate 对比，因为 `dual_source_gap_memory` 和 `dual_source_gap_critic` 还没有作为在线 batch method 完整执行 5-10 trials。

当前能够证明的是：

1. 双源 memory 能改变候选策略评分。
2. 仿真成功但真实失败的 gap 能降低候选信任。
3. 真实成功先验能提升候选信任。
4. gap 能生成沙盒校准参数。
5. 所有证据能被 JSON 和 Markdown 解释。

## 10. 可行性检查

通过：

1. schema 可行：旧数据兼容，新字段完整。
2. 数据接入可行：现有 result.json 能导入为 real episode。
3. pair/gap 可行：主 MVP 生成 8 个 pair，U4-3 专项生成 40 个 pair。
4. 检索可行：query 能返回 sim/real 混合证据。
5. 评分可行：U3-4、U4-2、U4-3 得到不同决策。
6. 沙盒校准可行：calibration 参数能生成并进入报告。

需要补强：

1. 真机数据缺口：当前是 pseudo-real，不是真实机器人在线闭环。
2. 在线双源方法缺口：`dual_source_gap_memory` / `dual_source_gap_critic` 还未作为 batch method 完整执行。
3. 批量统计缺口：需要完整对比 success rate、unsafe count、repeated failure。
4. critic 深度不足：当前规则 critic 可解释，但覆盖范围有限。
5. 视觉记忆不足：关键帧尚未进入 embedding 检索。

## 11. 下一轮实验计划

1. 将 `dual_source_gap_memory` 接入在线 batch 方法。
2. 将 `dual_source_gap_critic` 接入在线 batch 方法，使用 `candidate_score.decision` 触发 rewrite。
3. 每个条件至少跑 5 trials，优先条件：

```text
U3-4
U4-2
U4-3
U5-1
U5-2
U5-3
U5-4
U5-5
```

4. 新增分析脚本，输出：

```text
recovery_success_rate
unsafe_motion_count
repeated_failure_rate
sim_real_prediction_error
critic_rewrite_count
critic_rejection_count
memory_usefulness
```

5. 形成论文实验表：

| 方法 | 是否用 sim memory | 是否用 real memory | 是否用 gap | 是否用 critic | 是否用 calibration |
| --- | --- | --- | --- | --- | --- |
| direct_llm_weak | 否 | 否 | 否 | 否 | 否 |
| sim_only_weak | 是 | 否 | 否 | 否 | 否 |
| sim_memory_weak | 是 | 否 | 否 | 否 | 否 |
| hierarchical_memory_weak | 是 | 否 | 否 | 部分 | 否 |
| dual_source_gap_memory | 是 | 是 | 是 | 否 | 是 |
| dual_source_gap_critic | 是 | 是 | 是 | 是 | 是 |
