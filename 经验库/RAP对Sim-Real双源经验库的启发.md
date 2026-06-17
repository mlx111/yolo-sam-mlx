# RAP 对 Sim-Real 双源经验库的启发

## 1. 阅读对象

本文档记录对以下论文的分析：

1. `RAPv1.pdf`

重点关注该论文对当前研究方向 **Sim-Real 双源经验库与异常恢复沙盒推演** 的启发，尤其是检索增强规划、上下文记忆、retrieval key、经验窗口截取和 LLM 异常恢复规划。

## 2. 论文的核心内容

RAP 提出 **Retrieval-Augmented Planning**，目标是让 LLM agent 在当前决策时动态利用过去经验，而不是只靠当前 prompt 和模型参数。

RAP 的整体结构包括四个模块：

```text
Memory
Reasoner
Retriever
Executor
```

其中：

1. **Memory**
   - 保存过去成功任务的日志；
   - 每条日志包含任务信息、总体计划、动作-观测轨迹。

2. **Reasoner**
   - 根据当前任务生成总体计划；
   - 根据当前上下文生成 action plan；
   - 生成 retrieval key。

3. **Retriever**
   - 根据任务、总体计划和 retrieval key 检索相似经验；
   - 对文本和图像环境分别计算相似度；
   - 只截取与当前最相关的轨迹窗口，而不是传入完整历史。

4. **Executor**
   - 将检索到的经验作为上下文；
   - 结合当前状态生成下一步动作。

该论文的核心思想是：

```text
经验不是静态存档，
而是在每一步决策时动态检索、局部截取、作为上下文参与规划。
```

## 3. 对当前研究的总体启发

当前研究中的经验库不能只是“执行后保存、下次整体召回”。异常恢复是一个多步过程，不同阶段需要检索不同类型的经验。

RAP 对当前研究的启发是：

```text
异常恢复规划应采用 retrieval-augmented planning，
根据当前异常阶段、动作意图和观测状态动态检索经验。
```

这与当前系统中的 `retrieval_key`、`plan_signature`、`anomaly_state`、`text_summary` 高度一致。

## 4. 对 Memory 结构的启发

RAP 的 memory 保存：

```text
task information
overall plan
action-observation trajectory
```

当前研究可以对应保存：

```text
experience_log {
  condition_id,
  scenario_id,
  task_stage,
  recovery_goal,
  overall_recovery_plan,
  skill_sequence,
  observation_sequence,
  execution_feedback,
  outcome
}
```

这说明经验库不应只保存最终结果，而应保存“计划-动作-观测”的对应关系。

## 5. 对 retrieval key 的启发

RAP 的一个关键机制是 Reasoner 根据当前 action plan 动态生成 retrieval key。例如当前计划是“寻找某物”，retrieval key 就会围绕搜索目标生成。

当前异常恢复也可以类似：

```text
如果当前阶段是异常识别：
  retrieval_key 关注 anomaly_type + perception evidence

如果当前阶段是候选恢复生成：
  retrieval_key 关注 failure_type + task_stage + object_state

如果当前阶段是动作执行前验证：
  retrieval_key 关注 plan_signature + contact_pattern + critic flags
```

因此，当前系统可以把 retrieval key 分成多种：

```text
retrieval_key {
  anomaly_key,
  recovery_action_key,
  failure_risk_key,
  critic_key,
  visual_key
}
```

这样可以避免所有阶段都用同一个检索条件。

## 6. 对经验窗口截取的启发

RAP 不把完整轨迹全部塞给 LLM，而是围绕最相似动作或观测截取局部窗口。这对当前研究非常重要。

异常恢复经验通常包含很多步骤：

```text
detect-object
→ create-grasp
→ move-pregrasp
→ move-grasp
→ gripper-close
→ vertical-lift
→ place
```

但当前阶段可能只需要其中一小段。例如，当前问题是抓取前位姿不准，则应召回：

```text
move-pregrasp
→ move-grasp
→ gripper-close
```

而不是整条任务轨迹。

建议经验库增加：

```text
retrieval_window {
  center_step,
  window_before,
  window_after,
  local_observation,
  local_outcome
}
```

这和已有 `key_slices` 很匹配。

## 7. 对 LLM 异常恢复规划的启发

RAP 的 Executor 不是复制历史动作，而是把相关经验作为 in-context examples，用于生成当前动作。

当前研究中也应避免：

```text
检索到成功经验 → 直接复制 skill_sequence
```

更合理的是：

```text
检索相似经验
→ 提取触发条件、动作片段、结果证据
→ 让 LLM 根据当前状态改写恢复计划
```

例如 prompt 中可以写：

```text
相似经验显示：
在 condition_id=U3-4 中，目标位置偏移后直接 move-grasp 失败；
成功恢复经验先执行 relocalize，再 move-pregrasp，再 close-gripper。
当前状态与该经验相似，但目标高度不同，请生成适配当前状态的恢复计划。
```

## 8. 对现有 experiment-sim-wrapper 的对应关系

当前系统已经具备 RAP 式组件：

```text
Memory:
  memory/v3.py

Reasoner:
  llm_handler.py

Retriever:
  MemoryV3Library.query()

Executor:
  run_experiment_v4.py + recovery_steps.py
```

现有检索已经考虑：

```text
scenario_id
condition_id
available_actions
retrieval_key
anomaly_state
task_stage
text_summary
visual_index
```

这说明系统已经具备 RAP 的基本骨架。后续升级重点不是“有没有检索”，而是：

1. 动态生成 retrieval key。
2. 只截取局部经验窗口。
3. 区分成功经验、失败经验和 critic 经验的检索目标。
4. 在 prompt 中说明经验适用边界。

## 9. 对多模态检索的启发

RAP 在多模态任务中使用视觉观测做检索。当前系统已有 `memory/visual_retrieval.py`，支持 CLIP + FAISS 的视觉经验检索。

这可以支撑：

```text
keyframe retrieval
visual_context retrieval
condition scene similarity
failure image similarity
```

在没有真机阶段，关键帧检索仍然有意义，因为不同异常的视觉表现不同，例如：

1. 物体偏移。
2. 抓取后未抬起。
3. 夹爪闭合但无接触。
4. 放置后目标不在 plate 上。

视觉检索可以作为结构化检索的补充。

## 10. 对实验设计的启发

可以设置如下对比：

```text
No memory:
  不使用经验

Static memory prompt:
  固定拼接 top-k 历史经验

RAP-style retrieval:
  根据当前 condition / stage / action 动态检索

RAP-style retrieval + local window:
  只传入最相关动作窗口
```

评价指标：

```text
recovery_success_rate
invalid_plan_count
retrieval_precision
prompt_token_cost
repeated_failure_rate
task_completion_rate
```

RAP 特别适合支撑 `prompt_token_cost` 和 `retrieval_precision`，因为它强调只给 LLM 当前最相关的经验。

## 11. 与已有启发文档的关系

RAP 与前面几篇材料形成如下关系：

1. Worth Remembering 解决“哪些经验值得写入”。
2. RAP 解决“当前步骤该检索哪些经验”。
3. Learning From Failure 解决“失败经验如何影响动作风险”。
4. RoboCritics 解决“计划执行前如何检查安全性”。

组合后，当前研究可以写成：

```text
surprise-gated memory writing
→ retrieval-augmented recovery planning
→ failure-risk-aware candidate ranking
→ motion-level critic validation
→ sandbox / pseudo-real execution
```

## 12. 不能直接照搬的部分

RAP 主要面向通用 LLM agent，包括文本环境和多模态 benchmark，不是专门针对机器人异常恢复。因此不适合直接照搬：

1. ALFWorld / WebShop 的任务结构。
2. 只保存成功经验的 memory 假设。
3. 简单文本相似度作为主要检索依据。
4. 对物理状态和接触状态建模不足。

当前研究应吸收其 retrieval-augmented planning 框架，同时强化物理状态、失败风险和 Sim-Real 差异检索。

## 13. 可写入当前研究方案的关键表述

后续整合到总方案时，可以使用如下表述：

```text
RAP 表明，历史经验应在规划过程中被动态检索和局部使用，而不是作为静态日志整体拼接进 prompt。其 Memory-Reasoner-Retriever-Executor 框架说明，agent 可以根据当前任务、总体计划和局部动作意图生成 retrieval key，并检索最相关的经验窗口辅助下一步决策。

对本研究而言，这一思想可转化为异常恢复中的 retrieval-augmented recovery planning：系统根据当前异常状态、恢复阶段、候选动作和 critic 反馈动态检索成功经验、失败经验和 Sim-Real 差异经验，并将局部经验窗口提供给 LLM 生成可执行恢复计划。
```

## 14. 阶段性结论

该论文对当前研究的核心启发可以概括为：

```text
经验库的价值不在于保存很多历史，
而在于在当前规划步骤检索到刚好相关的经验片段。
```

具体而言，当前研究可以吸收以下内容：

1. 将异常恢复规划设计为 retrieval-augmented planning。
2. 动态生成不同阶段的 retrieval key。
3. 围绕最相似动作或观测截取局部经验窗口。
4. 将多模态关键帧检索作为结构化检索的补充。
5. 避免 LLM 机械复制历史经验，而是让其根据当前状态改写计划。

因此，该论文非常适合作为当前研究中 **经验检索、LLM 异常恢复规划、局部经验窗口和多模态记忆检索** 的支撑材料。
