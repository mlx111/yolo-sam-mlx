# Plan in Sandbox 对 Sim-Real 双源经验库的启发

## 1. 阅读对象

本文档记录对以下论文的分析：

1. `Plan in Sandbox1.pdf`

重点关注该论文对当前研究方向 **Sim-Real 双源经验库与异常恢复沙盒推演** 的启发，尤其是在没有真机或真机数据不足时，如何利用 physics-grounded sandbox 生成、抽象和复用经验。

## 2. 论文的核心内容

该论文提出 **SAGE，Sandbox-Abstracted Grounded Experience**。其目标是解决 embodied navigation 中真实数据稀缺和 Sim-to-Real 迁移困难的问题。

SAGE 的核心思想是：

```text
不必追求完全照片级真实仿真，
而是在物理约束下的语义抽象沙盒中学习可迁移经验。
```

整体流程包括三阶段：

1. **Genesis**
   - 在 physics-grounded sandbox 中合成任务；
   - 生成可执行轨迹；
   - 从轨迹中抽取经验规则。

2. **Evolution**
   - 通过检索经验规则增强训练样本；
   - 使用 RL 或策略优化让 agent 内化这些经验。

3. **Navigation**
   - 将沙盒中学到的抽象经验迁移到开放真实环境；
   - 使用 Frontier / Memory Node 作为中间决策目标，由几何规划器执行。

论文中经验规则采用类似：

```text
IF answering/searching [Task]
AND observing [Visual Cues]
THEN prioritize this path.
```

这种经验不是低层控制轨迹，而是从沙盒交互中抽象出的可复用策略规则。

## 3. 对当前研究的总体启发

这篇论文对当前研究非常重要，尤其是在当前没有真机的情况下。它提供了一个很有用的论证：

```text
沙盒经验不一定必须来自真实机器人，
只要沙盒具有物理约束和任务相关语义，
其中生成的经验也可以成为后续真实或伪真实执行的先验。
```

当前研究可以将这一思想改写为：

```text
没有真机时，
先用 experiment-sim-wrapper 作为 physics-grounded sandbox，
批量生成异常恢复经验，
再将其中高价值经验写入双源经验库的 simulation / pseudo-real 分区。
```

这能支撑当前阶段的研究路线：

```text
仿真异常注入
→ 沙盒恢复推演
→ 生成经验
→ 经验检索
→ shadow validation
→ 伪真机验证
```

## 4. 对“仿真作为伪真机”的启发

SAGE 不要求沙盒完全复刻真实世界，而是强调：

```text
physics-grounded semantic abstraction
```

这对当前研究很关键。因为你现在没有真机，不必把研究卡在真实机器人数据采集上，而可以明确提出一个阶段性替代方案：

```text
Simulation-as-Pseudo-Real
```

具体做法是：

1. 使用主 MuJoCo 仿真作为“执行环境”。
2. 使用 `sim_wrapper` 的 shadow simulation 作为执行前验证环境。
3. 通过扰动、噪声、异常注入、感知偏差模拟真实不确定性。
4. 将 shadow validation 通过的经验标记为 `simulation_validated`。
5. 将高风险失败经验标记为 `pseudo_real_failed` 或 `failed_memory`。

这种设计可以形成：

```text
simulation branch:
  大量生成异常经验

pseudo-real branch:
  用更严格的 shadow simulation 验证经验

future real branch:
  后续真机实验再替换或校准 pseudo-real branch
```

这正好对应当前没有真机的限制。

## 5. 对沙盒经验表示的启发

SAGE 的经验规则是可检索、可注入 prompt 的 IF-THEN 形式。当前研究也可以把异常恢复经验抽象成类似模板：

```text
IF encountering [anomaly_state]
AND observing [physical_evidence]
THEN prioritize [recovery_strategy]
```

例如：

```text
IF encountering object_slip
AND observing contact_after_close_only but no lift
THEN prioritize regrasp_with_lower_approach_and_lift_verify.
```

或者：

```text
IF encountering pose_shift
AND observing perceived_position differs from last known position
THEN prioritize relocalize_before_regrasp.
```

这比单纯保存动作序列更适合 LLM 使用，因为它保留了触发条件、观测证据和恢复策略之间的关系。

## 6. 对经验库字段的启发

当前经验库已经有：

```text
retrieval_key
text_summary
skill_sequence
key_slices
validation_status
```

结合 SAGE，建议新增或强化：

```text
sandbox_experience_rule {
  if_task_or_anomaly,
  if_observation,
  then_strategy,
  supporting_evidence,
  source_rollout,
  validation_status,
  confidence
}
```

对应当前系统可以映射为：

```text
if_task_or_anomaly:
  condition_id + failure_type + task_stage

if_observation:
  contact_pattern + perception_snapshot + anomaly_state

then_strategy:
  skill_sequence / abstract_skill

supporting_evidence:
  recovery_success_criteria + keyframes + execution_feedback
```

这样，经验库不仅能检索“相似动作”，还能检索“相似条件下为什么应该这么做”。

## 7. 对经验生成流程的启发

SAGE 的 Genesis 阶段不是被动等待数据，而是在沙盒中自动生成任务和经验。当前研究也可以采用主动经验生成：

```text
for condition_id in U1-U5:
  inject anomaly
  generate candidate recovery plans
  run sandbox validation
  save success/failure experience
  abstract reusable IF-THEN rule
```

这非常适合当前的 `experiment-sim-wrapper`，因为已有：

1. U1-U5 异常 benchmark。
2. batch runner。
3. rolling memory。
4. sim_wrapper validation。
5. experience_write / experience_read。
6. keyframes 和 retrieval_key。

因此，当前系统已经具备 SAGE 式 Genesis 的雏形。

## 8. 对抽象技能的启发

SAGE 从沙盒轨迹中抽象经验规则。当前仓库中也已有 `docs/E_SKILL_ABSTRACTION.md`，计划从成功经验中挖掘高频动作 n-gram 形成抽象技能。

这两者可以合并为：

```text
sandbox rollout
→ successful skill_sequence
→ high-frequency recovery pattern
→ abstract_skill
→ LLM planning action space
```

例如：

```text
detect-object
→ create-grasp
→ move-pregrasp
→ move-grasp
→ gripper-close
→ vertical-lift
```

可以抽象为：

```text
regrasp-and-lift-v1
```

这样可以降低 LLM 规划复杂度，也能让经验库从“轨迹日志”升级为“技能知识”。

## 9. 对沙盒推演评分的启发

SAGE 使用经验增强训练样本，鼓励 policy 内化已知经验，同时仍保留探索能力。当前研究可以把它转化为候选动作排序：

```text
candidate_score =
  sandbox_success_score
  + retrieved_experience_prior
  + abstract_skill_confidence
  - failed_memory_risk
  - safety_risk
  - pseudo_real_gap_uncertainty
```

其中 `retrieved_experience_prior` 对应 SAGE 的经验增强，`pseudo_real_gap_uncertainty` 对应当前没有真机时的仿真可信度惩罚。

## 10. 对当前阶段实验设计的启发

当前没有真机时，可以把实验设计成三阶段：

### 10.1 Stage 1: Sandbox Experience Generation

```text
在 MuJoCo 中批量注入 U1-U5 异常
→ 执行多种恢复策略
→ 保存成功和失败经验
```

### 10.2 Stage 2: Pseudo-Real Shadow Validation

```text
从经验库中检索候选策略
→ 在 sim_wrapper shadow scene 中验证
→ 通过则标记 simulation_validated
→ 失败则写入 failed_memory
```

### 10.3 Stage 3: Memory-Augmented Recovery

```text
新异常发生
→ 检索 sandbox experience
→ 检索 failed memory
→ 候选动作排序
→ shadow validation
→ 执行并回写
```

这可以在没有真机的情况下形成完整实验闭环。

## 11. 与现有 experiment-sim-wrapper 的对应关系

SAGE 的三个阶段可以直接对应到现有代码：

```text
Genesis:
  run_experiment_batch.py
  anomaly_conditions.py
  anomaly_injectors.py

Evolution:
  rolling_memory
  experience_read / experience_write
  failed_memory blocker
  abstract skill mining

Navigation / Execution:
  run_experiment_v4.py
  sim_wrapper.py
  recovery_steps.py
```

这说明你当前系统不是单纯的仿真实验，而已经具备：

```text
沙盒经验生成
→ 经验复用
→ 虚拟验证
→ 回写更新
```

的结构。

## 12. 与前面几篇启发文档的关系

这篇论文与前面材料的关系如下：

1. Worth Remembering 说明哪些经验值得保存。
2. Learning From Failure 说明失败经验如何用于风险预警。
3. Plan in Sandbox 说明没有真机时，如何用物理约束沙盒生成和抽象经验。
4. RISE 说明如何用 imagined rollout 评价动作后果。
5. RialTo 说明未来如何把真实场景同步到仿真。

因此，当前阶段可以先形成：

```text
Sandbox-first dual-source memory prototype
```

即先用沙盒和 shadow simulation 代替真机，建立经验库闭环，再在未来接入真实机器人。

## 13. 不能直接照搬的部分

该论文主要面向导航任务，而不是机械臂操作。因此不适合直接照搬：

1. Frontier / Memory Node 的具体导航形式。
2. A* 轨迹生成流程。
3. 视觉问答式导航任务格式。
4. 面向导航的 reward shaping。

当前研究应吸收的是其 **physics-grounded sandbox + abstracted experience** 思想，而不是其导航任务实现。

## 14. 可写入当前研究方案的关键表述

后续整合到总方案时，可以使用如下表述：

```text
Plan in Sandbox 提出的 SAGE 框架表明，机器人经验不一定只能来自真实世界，也可以在具有物理约束的语义抽象沙盒中生成、筛选和抽象。该工作强调沙盒不必追求完全照片级真实，而应提供足够的物理可执行性和任务相关语义，使其中生成的经验能够迁移到真实或伪真实执行环境。

对本研究而言，这一思想为当前无真机阶段提供了直接支撑：可以先将 MuJoCo 与 sim_wrapper 构成 physics-grounded sandbox / pseudo-real validation layer，通过异常注入和 shadow validation 批量生成成功与失败恢复经验，再将高价值经验写入 Sim-Real 双源经验库。
```

## 15. 阶段性结论

该论文对当前研究的核心启发可以概括为：

```text
没有真机并不意味着不能构建经验闭环；
只要沙盒具备物理约束和任务相关抽象，就可以先生成可复用的伪真实经验。
```

具体而言，当前研究可以吸收以下内容：

1. 将 `experiment-sim-wrapper` 明确定位为 physics-grounded sandbox。
2. 将 `sim_wrapper` 明确定位为 pseudo-real shadow validation layer。
3. 从沙盒轨迹中抽象 IF-THEN 异常恢复经验规则。
4. 用 rolling memory 和 batch runner 构建自动经验生成流程。
5. 在未来接入真机后，用真实执行结果校准当前 pseudo-real memory。

因此，该论文非常适合作为当前研究中 **无真机阶段、仿真伪真机验证、沙盒经验生成和抽象恢复经验规则** 的支撑材料。
