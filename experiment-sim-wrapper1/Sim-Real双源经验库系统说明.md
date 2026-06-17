# Sim-Real 双源经验库系统说明

## 1. 系统定位

当前实现的目标不是单独增加一个机器人记忆列表，而是在现有 `experiment-sim-wrapper1` 的异常恢复实验框架上，形成一个可继续开发的 **Sim-Real 双源异常经验库与执行前沙盒推演系统**。

系统核心主张：

```text
机器人异常恢复不应只依赖仿真成功经验，也不应把真实失败简单作为噪声丢弃。
系统应同时保存仿真经验、真实/伪真实经验、二者差异、critic 风险与沙盒校准信息，
并在恢复动作执行前完成检索、评分、推演和风险过滤。
```

本版本已经把该主张落到代码链路中：

1. 仿真经验继续由 `run_experiment_v4.py` 和 `experiment_method_runner.py` 写入。
2. 真实/伪真实经验通过 `real_memory/ingest.py` 导入为 `MemoryV3Entry(source="real")`。
3. 同一异常条件下的 sim/real 经验通过 `memory/dual_source.py` 配对。
4. sim-real gap 被结构化保存到经验条目中。
5. gap 和 critic 被用于检索排序、候选策略评分和沙盒校准。
6. `tools/run_mvp_dual_source_pipeline.py` 将导入、配对、gap、calibration、retrieval、candidate scoring 串成可复现 MVP 流水线。

## 2. 代码模块地图

| 模块 | 作用 | 当前状态 |
| --- | --- | --- |
| `memory/v3.py` | MemoryV3 主 schema、序列化、检索、旧 JSON 兼容 | 已扩展双源字段 |
| `memory/gating.py` | 经验写入门控，判断是否值得长期保存 | 已接入实验写入 |
| `memory/critic.py` | 结构化 critic_result，统一规则 critic 和 LLM critic 输出 | 已接入失败/虚拟验证经验 |
| `memory/dual_source.py` | sim-real 经验配对与 gap 计算 | 已实现 pair/gap |
| `memory/calibration.py` | 从 gap 经验估计沙盒校准参数 | 已接入 virtual scene |
| `memory/scoring.py` | gap-aware、risk-aware 检索调整与候选计划评分 | 已接入 query 和 runner |
| `real_memory/schema.py` | 真实 episode 最小结构 | 已实现 |
| `real_memory/ingest.py` | 手工/自动标注 JSON 与现有 result.json 导入 | 已实现 |
| `tools/import_real_episode.py` | real/pseudo-real 批量导入 CLI | 已实现 |
| `tools/build_sim_real_pairs.py` | 对 memory snapshot 生成 pair/gap | 已实现 |
| `tools/run_mvp_dual_source_pipeline.py` | MVP 离线双源流水线 | 已实现 |
| `tools/smoke_mvp_dual_source_pipeline.py` | MVP 快速回归 | 已实现 |

## 3. MemoryV3 双源字段

`MemoryV3Entry` 在不破坏旧数据的前提下新增了以下字段。旧 JSON 读取时会自动填默认值。

| 字段 | 作用 | 主要消费者 |
| --- | --- | --- |
| `memory_gate` | 写入门控结果，保存 `write_score`、decision、reason | 长期库写入、后续压缩 |
| `critic_result` | 结构化 critic 输出，包含风险分数、规则 flags、rewrite feedback | 候选评分、计划重写 |
| `sim_real_pair` | 当前条目与另一来源经验的配对关系 | gap 报告、证据追踪 |
| `sim_real_gap` | 仿真与真实/伪真实之间的差异签名 | calibration、检索排序、候选评分 |
| `sandbox_calibration` | 从 gap 推导的沙盒校准信息 | `sim_wrapper.build_virtual_scene()` |
| `real_episode_ref` | 真实 episode 来源、raw id、设备/操作者/数据路径引用 | 真实数据审计 |
| `memory_tags` | memory type、role、summary level、keyframe role 等标签 | 层级记忆与检索 |

`build_retrieval_key()` 已把以下信息纳入检索键：

```text
gap_type
gap_uncertainty_bucket
critic_status
critic_flags
memory_type
memory_role
real_validated
pair_status
```

这样后续可以区分：

1. 普通仿真成功经验。
2. 真实成功先验。
3. 真实失败风险案例。
4. sim-real gap memory。
5. critic warning/reject 案例。

## 4. 经验写入门控

门控实现位于 `memory/gating.py`，入口为：

```python
compute_memory_gate(metrics, recovery_success, task_success, validation_status, sim_real_gap)
```

当前版本先记录门控，不强制丢弃经验，原因是 MVP 阶段需要保留足够样本做对照分析。门控评分重点考虑：

1. 异常/失败是否明显。
2. 恢复是否成功或处在成功失败边界。
3. 是否存在高 sim-real gap。
4. 是否存在 critic warning 或 reject。
5. 是否具备真实执行价值。

接入点：

```text
run_experiment_v4.py::_build_experience_entry()
```

可行性判断：

1. 对已有 batch 实验低风险，因为默认只写字段不改变行为。
2. 后续可以增加配置项，让低价值 episode 只进 short-term buffer，不进长期库。
3. 真实数据量变大后，门控会成为控制库规模和重复样本的关键模块。

## 5. 结构化 critic

`memory/critic.py` 将规则 critic 和 LLM critic 统一成 `critic_result`。

核心字段：

```text
overall_status
critic_risk_score
rule_flags
llm_flags
feedback_for_rewrite
evidence
```

当前已经支持将以下风险写入经验：

1. 虚拟验证失败。
2. 抓取后无接触。
3. 提升阶段接触丢失。
4. 规则 critic 和 LLM critic 的合并风险。

接入点：

```text
experiment_method_runner.py::save_experience()
memory/scoring.py::critic_risk_score()
```

可行性判断：

1. 第一版规则 critic 足够支撑论文中的安全评估闭环。
2. 不需要一开始训练 RoboCritics 类模型。
3. 后续可将视觉轨迹 critic、接触稳定性 critic、关节速度/限位 critic 逐步接入同一 schema。

## 6. Sim-Real 配对与 Gap 表示

`memory/dual_source.py` 实现两层逻辑：

1. `pair_sim_real_experiences(sim_entries, real_entries)`：找到同 condition/scenario/plan signature 的 sim-real pair。
2. `compute_sim_real_gap(sim_entry, real_entry)`：计算差异签名。

当前 gap 结构包括：

```text
pose_gap.object_pose_error
pose_gap.z_after_recovery_gap
contact_gap.contact_mismatch
outcome_gap.type
outcome_gap.outcome_gap_score
perception_gap.pose_estimation_gap
actuation_gap.gripper_closure_gap
uncertainty
evidence.gap_components
```

重要 gap 类型：

| gap 类型 | 含义 | 候选评分影响 |
| --- | --- | --- |
| `matched_success` | 仿真和真实都成功，但可能存在姿态/接触差异 | 可作为真实成功先验，但保留校准 |
| `sim_success_real_fail` | 仿真成功但真实失败 | 强烈降低相似候选的信任分 |
| `sim_fail_real_success` | 仿真失败但真实成功 | 提示沙盒过保守，需要单独分析 |
| `matched_failure` | 两边都失败 | 可作为失败风险先验 |

可行性判断：

1. 当前 pseudo-real 数据已经能生成稳定 pair/gap。
2. gap 的 pose/contact/outcome 三类证据足以支撑 MVP。
3. 真机阶段需要补充更可靠的时间同步、力/触觉、相机关键帧引用。

## 7. 沙盒校准

`memory/calibration.py` 根据检索到的 gap memory 估计沙盒校准：

```text
object_pose_bias
gripper_delay_bias
slip_risk_bias
contact_success_bias
perception_noise_bias
calibration_confidence
source_gap_ids
```

`sim_wrapper.build_virtual_scene()` 已支持传入 `sandbox_calibration`，并将校准影响到：

1. apple 初始位置。
2. 感知位置与校准位置。
3. 抓取 attach 距离。
4. reconstruction artifacts。

`run_experiment_v4.py` 在执行 LLM 恢复沙盒推演前，会基于最近检索的经验生成 calibration，并写入：

```text
metrics["sandbox_calibration"]
experience.sandbox_calibration
reconstruction_artifacts.sandbox_calibration
```

可行性判断：

1. 当前校准是规则加权平均，适合 MVP 和论文消融。
2. 校准只影响沙盒和评分，不直接修改 LLM 生成计划，因此行为风险可控。
3. 后续真机数据足够后，可以把 calibration 从规则模型升级为 learned residual model。

## 8. 双源检索与候选评分

`MemoryV3Library.query()` 新增参数：

```python
gap_aware=True
risk_aware=True
```

默认不打开时保持旧行为。打开后，检索分数会结合：

1. `gap_uncertainty`：gap 高且不确定时降权。
2. `critic_risk`：critic warning/reject 时降权。
3. `real_validation_bonus`：真实验证或真实执行经验加权。
4. `risk_penalty`：仿真成功但真实失败的案例降权。
5. `trust_bonus`：matched real success 增信。

候选计划评分入口：

```python
score_candidate_plan(candidate_steps, retrieved_experiences)
```

输出字段：

```text
candidate_score
decision
support_score
real_success_prior
failure_overlap_risk
gap_uncertainty
critic_risk
candidate_actions
evidence
```

决策含义：

| decision | 含义 |
| --- | --- |
| `prefer` | 可优先采用，真实成功先验或低风险证据较强 |
| `allow` | 可执行但证据不足，需要沙盒/critic 继续确认 |
| `rewrite_recommended` | 建议 LLM 重写，常见于失败重叠、高 gap 或 critic 风险 |
| `reject_recommended` | 高风险，不建议执行 |

## 9. MVP 流水线

入口：

```bash
python -B tools/run_mvp_dual_source_pipeline.py \
  --config configs/mvp_dual_source_gap_critic_u3_u4_v1.json
```

配置文件：

```text
configs/mvp_dual_source_gap_critic_u3_u4_v1.json
```

输出：

```text
results/mvp_dual_source_gap_critic_u3_u4_v1/real_memory_snapshot.json
results/mvp_dual_source_gap_critic_u3_u4_v1/sim_real_memory_snapshot.json
results/mvp_dual_source_gap_critic_u3_u4_v1/gap_report.json
results/mvp_dual_source_gap_critic_u3_u4_v1/candidate_scoring_report.json
analysis/mvp_dual_source_report.md
```

快速回归：

```bash
python -B tools/smoke_mvp_dual_source_pipeline.py
```

当前 MVP 实测结果：

```text
real_entry_count = 8
sim_entry_count = 8
pair_count = 8
conditions = U3-4, U4-2
```

U4-3 专项配置已经补齐：

```bash
python -B tools/run_mvp_dual_source_pipeline.py \
  --config configs/mvp_dual_source_gap_critic_u4_3_v1.json
```

U4-3 输出：

```text
results/mvp_dual_source_gap_critic_u4_3_v1/
analysis/mvp_dual_source_u4_3_report.md
```

U4-3 专项实测结果：

```text
real_entry_count = 40
sim_entry_count = 40
pair_count = 40
matched_success = 20
sim_success_real_fail = 20
candidate_score = 0.5292
decision = allow
```

该结果说明 U4-3 同时存在强成功先验和强失败风险。只看 top-k 支持证据时会过度乐观，因此 pipeline 已加入 `support_retrieved` 与 `risk_retrieved` 双分支证据：支持分支保留成功经验，风险分支显式拉入失败、高 gap 或 critic 风险经验。

## 10. 当前可行性结论

已经可行：

1. 旧 memory JSON 兼容读取。
2. 新双源字段能完整序列化和反序列化。
3. pseudo-real result.json 能导入为 real memory。
4. sim-real pair 和 gap 能稳定生成。
5. gap 能影响检索排序、候选评分和沙盒校准。
6. MVP 离线流水线可复现，并有 smoke 回归。

尚未完成：

1. 还没有真实机器人在线数据采集 adapter。
2. 还没有完整 5-10 trials 的 `dual_source_gap_memory` / `dual_source_gap_critic` 在线对照 batch。
3. 当前 critic 是规则化 MVP，不是训练得到的轨迹级模型。
4. 视觉关键帧目前作为引用字段存在，尚未接入 embedding 检索。
5. calibration 是规则加权平均，不是系统辨识或学习模型。

结论：

```text
当前实现已经足以作为后续双源经验库开发基线。
它证明了 schema、导入、配对、gap、校准、检索、评分和 MVP 报告链路可跑通。
下一阶段重点不应继续扩 schema，而应补充真实/伪真实数据规模和在线实验对照。
```

## 11. 后续开发优先级

P0：补齐真实/伪真实实验数据

1. U4-3 已用现有 v2 batch 的 40 条 `result.json` 跑通离线双源链路。
2. 下一步补齐 `U3-4`、`U4-2`、`U5-1` 到 `U5-5` 的更平衡 5-10 trials 对照。
3. 每条保存 `result.json`、`experience_after.json`、关键帧引用。
4. 用 `tools/import_real_episode.py` 统一导入。

P1：把双源方法接入在线 batch

1. 新增或扩展 method：`dual_source_gap_memory`。
2. 新增或扩展 method：`dual_source_gap_critic`。
3. 在 `experiment_method_runner.py` 中让 candidate_score 决策影响重写或执行选择。
4. 输出 `candidate_score_history`、`sandbox_calibration`、`critic_result`。

P2：实验分析脚本

1. success rate。
2. unsafe motion count。
3. repeated failure rate。
4. sim-real prediction error。
5. critic rejection / rewrite count。
6. memory usefulness。

P3：真实机器人 adapter

1. 定义 real episode 标注格式。
2. 接入相机关键帧、末端位姿、夹爪状态、接触估计。
3. 真机执行后自动回写 `real_episode_ref` 和 `sim_real_gap`。

P4：论文材料增强

1. 画系统架构图。
2. 整理 schema 表。
3. 整理方法对比表。
4. 给出 ablation：无 gap、无 critic、无 calibration、sim-only、real-only。
