# Learning From Failure 对 Sim-Real 双源经验库的启发

## 1. 阅读对象

本文档记录对以下论文的分析：

1. `Learning From Failure.pdf`

重点关注该论文对当前研究方向 **Sim-Real 双源经验库与异常恢复沙盒推演** 的启发，尤其是失败经验库、失败预警、风险感知动作选择、失败经验检索和仿真伪真机阶段的经验复用。

## 2. 论文的核心内容

该论文提出 **Failure Episodic Memory Alert，FEMA**。它的核心问题是：机器人强化学习早期会产生大量短时程、低回报、提前终止的失败轨迹，例如碰撞、跌倒、翻滚等。传统经验回放通常把这些失败 transition 当作低价值样本，甚至会被大量失败轨迹拖慢学习。

论文的关键观点是：

```text
失败轨迹不是无用噪声。
失败轨迹包含“如何走向危险状态”的时空结构。
```

FEMA 的基本流程是：

```text
失败 episode 发生
→ 保存终止前 K 步状态-动作序列
→ 计算失败片段的 Monte Carlo return
→ 学习 state-action 风险表征
→ 当前决策时检索相似失败经验
→ 对候选动作进行风险评分
→ 避免再次进入相似危险状态
```

它包含两个模块：

1. **Failure Episodic Memory Construction**
   - 收集提前终止的失败轨迹；
   - 保存失败前一段窗口；
   - 用状态-动作嵌入表示危险模式；
   - 用低回报作为风险监督信号。

2. **Risk-aware Action Selection**
   - 当前状态下采样多个候选动作；
   - 检索相似失败事件；
   - 对候选动作打风险分；
   - 选择低风险、高长期价值的动作。

这篇论文的目标虽然是提高 RL 训练效率，但其思想非常适合当前课题中的异常恢复经验库。

## 3. 对当前研究的总体启发

当前研究已经有成功经验和失败经验分区，也有 failed memory blocker / rewrite 的原型。FEMA 可以为这部分提供更清晰的研究支撑：

```text
失败经验不只是记录失败原因，
还应作为恢复动作选择时的风险先验。
```

这对当前研究尤其重要，因为异常恢复动作常发生在高风险状态之后。如果只检索成功经验，系统容易重复尝试在真机或伪真机中已经失败过的动作。失败经验库的价值在于：

1. 识别危险状态。
2. 识别危险动作。
3. 提醒 LLM 或策略不要重复失败路径。
4. 在沙盒推演前降低高风险候选动作的排序。
5. 在没有真机时，用仿真失败积累初始风险知识。

因此，当前研究可以把 FEMA 的思想转化为：

```text
Failure Memory Branch
  → 失败事件保存
  → 相似失败检索
  → 候选恢复动作风险惩罚
  → 失败经验回写
```

## 4. 对 failed_memory 的启发

当前经验库中已有：

```text
failed_memory
failure_taxonomy
failure_reason
failed memory blocker
```

结合 FEMA，失败经验不应只保存自然语言失败原因，还应保存失败前的状态-动作结构。

建议失败经验最小字段为：

```text
failed_experience {
  failure_event_id,
  source: sim / pseudo_real / real,
  condition_id,
  scenario_id,
  task_stage,
  failure_type,
  failure_root_cause,
  pre_failure_state_window,
  failed_action_sequence,
  terminal_state,
  failure_return,
  risk_embedding,
  recovery_block_hint
}
```

其中 `pre_failure_state_window` 是关键。它对应 FEMA 中保存终止前 K 步轨迹的思想。对于当前系统，可以先保存：

```text
before_anomaly
anomaly_detected
before_recovery
after_failed_recovery
```

这些切片已经和现有 `key_slices` / `keyframes` 结构兼容。

## 5. 对候选恢复动作排序的启发

当前研究文档中已有候选动作评分：

```text
candidate_score =
  sim_success_score
  + real_success_prior
  - failure_memory_risk
  - sim_real_gap_uncertainty
  - safety_risk
```

FEMA 可以补强 `failure_memory_risk` 的具体计算逻辑。建议改为：

```text
failure_memory_risk =
  c1 * similar_failure_count
  + c2 * failure_similarity
  + c3 * failed_action_overlap
  + c4 * terminal_risk_score
```

其中：

1. `similar_failure_count`：相似条件下失败经验数量。
2. `failure_similarity`：当前状态与失败状态的相似度。
3. `failed_action_overlap`：候选动作是否与历史失败动作相似。
4. `terminal_risk_score`：失败是否导致严重终止，例如碰撞、掉落、夹爪卡死。

这样失败经验就不是简单的 blocker，而是可以参与候选动作连续评分。

## 6. 对仿真伪真机阶段的启发

当前没有真机，可以把 `experiment-sim-wrapper` 中的主仿真或扰动仿真当作 pseudo-real。FEMA 对这个阶段非常有价值，因为失败经验可以先从仿真中大量生成。

建议分成两层：

```text
simulation_failure_memory:
  来自普通仿真批量异常注入

pseudo_real_failure_memory:
  来自 sim_wrapper / shadow validation / 更严格扰动仿真
```

前者用于覆盖大量失败模式，后者用于模拟“更接近真实执行”的高置信失败。

对应 `validation_status` 可以设计为：

```text
simulation_only
simulation_validated
pseudo_real_failed
pseudo_real_validated
real_validated
```

在没有真机时，`pseudo_real_failed` 可以作为 `real_failed` 的替代层，用于后续论文中的 staged validation：

```text
阶段一：仿真异常经验
阶段二：shadow simulation 伪真机验证
阶段三：未来真机验证
```

## 7. 对 Top-O 关键失败选择的启发

FEMA 中有一个重要设计：不是所有检索到的失败经验都参与风险评分，而是选择最关键的 Top-O 失败事件。如果使用所有失败经验，系统可能变得过于保守。

这对当前研究也非常重要。异常恢复不能因为历史上失败过就完全不动，否则会导致恢复策略过度保守。

建议当前系统使用：

```text
top_critical_failures =
  retrieve_similar_failures(...)
  → sort by failure_severity + similarity
  → keep top O
```

然后只用这些关键失败经验影响候选动作评分。

这样可以避免两个问题：

1. 失败经验过多导致所有动作都被惩罚。
2. 低相似失败经验误伤当前可行恢复动作。

## 8. 对经验库检索的启发

FEMA 使用 state embedding 和 state-action embedding 进行相似失败检索。当前研究可以对应为混合检索：

```text
failure_retrieval_key {
  condition_id,
  scenario_id,
  task_stage,
  failure_type,
  contact_pattern,
  plan_signature,
  pre_failure_state_signature,
  failed_action_signature
}
```

其中 `condition_id` 仍然应作为硬过滤，避免跨条件错误召回。然后再用 `plan_signature`、`contact_pattern` 和状态相似度排序。

这与现有 `memory/v3.py` 中的：

```text
retrieval_key
anomaly_state
failure_taxonomy
plan_signature
contact_pattern
```

高度兼容。

## 9. 对现有 experiment-sim-wrapper 的对应关系

当前 `experiment-sim-wrapper` 已经有若干 FEMA 式基础：

1. `failed_memory`：失败经验分区。
2. `failure_taxonomy`：失败类型与诊断。
3. `retrieval_key`：检索键。
4. `plan_signature`：动作序列签名。
5. `key_slices` / `keyframes`：失败前后片段。
6. `sim_wrapper`：执行前虚拟验证。
7. `validation_status`：区分 simulation_only / simulation_validated / failed。

因此，FEMA 不需要从零实现，而是可以作为现有系统的理论升级：

```text
failed memory blocker
→ risk-aware failure memory
```

也就是说，失败经验不只是阻止完全相同的错误计划，而是对所有候选恢复动作提供风险评分。

## 10. 对实验设计的启发

可以设计如下消融实验：

```text
baseline: 无经验库
success_memory_only: 只用成功经验
failure_blocker: 使用失败经验硬阻断
risk_aware_failure_memory: 使用失败经验风险评分
dual_memory: 成功经验 + 风险感知失败经验
```

评价指标：

```text
recovery_success_rate
repeated_failure_rate
unsafe_action_count
invalid_plan_count
average_recovery_time
task_completion_rate
```

其中最关键的指标是：

```text
repeated_failure_rate
```

因为 FEMA 的核心价值就是减少机器人重复进入相似失败状态。

## 11. 与已有启发文档的关系

这篇论文和前面几篇材料形成互补：

1. RISE 强调 imagined rollout 和 value 评分。
2. Robotic Sim-to-Real Transfer 强调感知差异和执行差异。
3. RialTo 强调 real-to-sim scene twin。
4. Worth Remembering 强调什么经验值得写入。
5. Learning From Failure 强调失败经验如何在后续决策中产生风险预警。

组合后，当前研究可以形成：

```text
异常发生
→ 写入门控判断是否值得保存
→ 成功经验提供候选动作
→ 失败经验提供风险惩罚
→ 沙盒推演验证
→ 真机或伪真机执行
→ 回写成功/失败经验
```

## 12. 不能直接照搬的部分

该论文有些内容不适合直接照搬：

1. 它主要面向 RL 训练，不是 LLM 规划或异常恢复。
2. 它的状态-动作嵌入训练需要大量交互数据。
3. 它的 MuJoCo benchmark 多为 locomotion，不是 manipulation。
4. 它的风险头和候选动作采样机制不一定适合当前基于技能序列的恢复系统。

当前研究应吸收其 failure-centric episodic memory 思想，而不是完整复现其 RL 算法。

## 13. 可写入当前研究方案的关键表述

后续整合到总方案时，可以使用如下表述：

```text
Learning From Failure 表明，失败轨迹并非无价值噪声，而是包含机器人如何进入危险状态的时空结构。其 Failure Episodic Memory Alert 机制通过保存提前终止前的失败片段，并在后续决策中检索相似失败经验，对候选动作进行风险惩罚，从而减少重复失败。

对本研究而言，这一思想可以转化为异常恢复中的 failure memory branch：成功经验用于生成候选恢复动作，失败经验用于识别高风险动作和状态，二者共同参与沙盒推演前的候选动作排序。
```

## 14. 阶段性结论

该论文对当前研究的核心启发可以概括为：

```text
失败经验不只是要保存，
更要在下一次恢复决策中主动提醒系统不要重蹈覆辙。
```

具体而言，当前研究可以吸收以下内容：

1. 建立 failure-centric memory branch。
2. 保存失败前 K 步关键状态-动作窗口。
3. 用相似失败检索为候选恢复动作提供风险评分。
4. 使用 Top-O 关键失败，避免系统过度保守。
5. 在没有真机时，用仿真和 shadow simulation 先积累 pseudo-real failure memory。

因此，该论文非常适合作为当前研究中 **失败经验库、失败预警、风险感知候选动作排序和仿真伪真机阶段失败经验积累** 的支撑材料。
