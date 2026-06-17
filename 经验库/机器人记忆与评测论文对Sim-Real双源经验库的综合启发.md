# 机器人记忆与评测论文对 Sim-Real 双源经验库的综合启发

## 1. 阅读对象

本文档综合分析以下四篇机器人记忆和记忆评测相关论文：

1. `RoboMemory.pdf`
2. `RoboMME.pdf`
3. `RoboMemArena.pdf`
4. `Episodic_Memory_Verbalization_Using_Hierarchical_Representations_of_Life-Long_Robot_Experience.pdf`

重点关注它们对当前研究方向 **Sim-Real 双源经验库与异常恢复沙盒推演** 的启发，尤其是多类型记忆架构、关键帧记忆、长期经验层级化、记忆依赖评测、经验检索粒度和异常恢复经验库字段设计。

## 2. 四篇论文的核心内容

这组论文共同说明一个趋势：

```text
机器人经验库不能只是 episode 日志或 top-k 文本检索，
而应成为可更新、可检索、可评测、可解释的多层记忆系统。
```

### 2.1 RoboMemory

RoboMemory 提出一个脑启发的多记忆 agent 框架，将机器人记忆分为：

```text
Temporal Memory
Spatial Memory
Semantic Memory
Episodic Memory
```

其中：

1. **Temporal Memory** 保存当前任务内的动作和观测历史。
2. **Spatial Memory** 使用动态空间知识图谱记录物体、位置和空间关系。
3. **Semantic Memory** 保存跨任务、较稳定的环境知识和事实。
4. **Episodic Memory** 保存任务级执行轨迹、反馈和历史尝试。

RoboMemory 还强调两点：

1. 多个记忆模块应并行更新和检索，避免长时任务中记忆系统拖慢执行。
2. 规划器应结合 critic 闭环，根据最新观测和记忆动态重规划。

### 2.2 RoboMME

RoboMME 是面向机器人通用操作策略的记忆评测 benchmark。它将机器人记忆需求分为四类：

```text
Temporal Memory
Spatial Memory
Object Memory
Procedural Memory
```

对应的任务包括：

1. **Counting**：测试事件计数和顺序记忆。
2. **Permanence**：测试遮挡和环境变化下的空间位置记忆。
3. **Reference**：测试短暂指示对象后的身份保持。
4. **Imitation**：测试对历史演示轨迹和过程的复现。

RoboMME 还比较了三类记忆表示：

```text
symbolic memory
perceptual memory
recurrent memory
```

论文的重要结论是：没有一种记忆表示在所有任务上都最优。符号记忆适合计数、短期推理和可解释规划；视觉/感知记忆对时间敏感、运动轨迹和模仿类任务更重要。

### 2.3 RoboMemArena

RoboMemArena 提出大规模长时程机器人记忆 benchmark，并设计 PrediMem。其关键思想包括：

```text
recent buffer
+ keyframe buffer
+ high-level planner
+ low-level VLA executor
+ predictive coding for keyframe sensitivity
```

其中 recent buffer 保存最近若干帧，keyframe buffer 保存长期任务中决策关键的历史帧。系统在执行中判断当前帧是否应写入关键帧记忆，以避免重要状态转移离开短期窗口后丢失。

该论文对当前研究特别重要的是：

1. 记忆写入不能只按固定频率保存，应围绕关键帧和状态转移保存。
2. long-horizon 任务中，很多子任务是否正确依赖历史，而不是当前观测。
3. benchmark 应统计 memory-dependent subtask ratio，即当前动作是否必须依赖历史才能正确决策。

### 2.4 Episodic Memory Verbalization

该论文关注机器人长期经验的语言化和问答。它将连续经验流组织成层级 history tree：

```text
L0 Raw Experiences
L1 Scene Graphs
L2 Events
L3 Goals
L4+ Higher-Level Summaries
```

查询时，LLM 不直接读取完整历史，而是：

```text
collapsed history tree
→ expand relevant nodes
→ search local subtree
→ optionally call VLM on raw image
→ answer / summarize
```

这对当前经验库的启发是：长期经验应支持从摘要到原始证据的逐层展开，而不是把所有 episode 展平成文本后一次性塞给 LLM。

## 3. 对当前研究的总体启发

当前 Sim-Real 双源经验库应从“经验条目集合”升级为“多类型机器人记忆系统”：

```text
short-term execution memory
+ spatial relation memory
+ episodic recovery memory
+ semantic failure memory
+ visual keyframe memory
+ sim-real gap memory
```

对于异常恢复任务，单条经验至少要回答五类问题：

1. 当前异常之前发生了什么？
2. 物体和机器人当前处于什么空间关系？
3. 历史上相似异常如何恢复？
4. 哪些失败经验提示当前恢复动作有风险？
5. 这条经验是否通过沙盒、critic 或 pseudo-real 验证？

这意味着经验库的核心不是“保存更多历史”，而是让系统在异常发生时能检索到正确类型、正确粒度、正确可信度的经验。

## 4. 对经验库记忆类型的启发

结合四篇论文，当前经验库可以扩展为如下记忆层：

```text
memory_system {
  temporal_memory,
  spatial_memory,
  episodic_memory,
  semantic_memory,
  perceptual_keyframe_memory,
  sim_real_gap_memory
}
```

### 4.1 temporal_memory

用于记录当前 episode 内的动作-观测-反馈序列：

```text
temporal_memory {
  recent_steps,
  action_history,
  observation_history,
  gripper_state_history,
  object_state_history,
  stage_transition_history
}
```

它主要回答：

```text
刚刚做过什么？
异常是在哪个阶段发生的？
是否已经尝试过某个恢复动作？
```

### 4.2 spatial_memory

用于保存任务相关空间关系，而不是完整点云：

```text
spatial_memory {
  object_relations,
  support_relations,
  containment_relations,
  distance_bins,
  occlusion_state,
  robot_object_relation
}
```

例如：

```text
target_object on table
target_object near plate
gripper above target_object
obstacle between gripper and target_object
```

这比只保存位置坐标更适合 LLM 规划，也能为 failure taxonomy 和 critic 提供解释。

### 4.3 episodic_memory

用于保存具体任务尝试和恢复轨迹：

```text
episodic_memory {
  condition_id,
  scenario_id,
  anomaly_state,
  recovery_goal,
  skill_sequence,
  observation_sequence,
  key_slices,
  execution_feedback,
  outcome
}
```

它主要回答：

```text
以前遇到类似异常时，完整恢复过程是什么？
哪些动作有效，哪些动作导致失败？
```

### 4.4 semantic_memory

用于保存跨 episode 稳定成立的规则、经验和失败模式：

```text
semantic_memory {
  failure_rule,
  recovery_rule,
  object_affordance,
  action_precondition,
  sim_real_warning,
  critic_rule
}
```

例如：

```text
如果 close-gripper 后 object_height 未变化，通常说明 grasp_miss。
如果 object lateral offset 较大，直接 move-grasp 容易失败，应先 relocalize。
```

这类记忆适合写入 LLM prompt，作为异常恢复的高层原则。

### 4.5 perceptual_keyframe_memory

用于保存视觉关键帧和局部状态转移：

```text
perceptual_keyframe_memory {
  keyframe_id,
  image_path,
  depth_path,
  stage,
  event_type,
  visual_embedding,
  object_boxes,
  gripper_object_contact,
  before_after_state
}
```

关键帧不应只按固定时间间隔保存，而应围绕以下事件保存：

1. 异常首次出现。
2. grasp 前后。
3. object pose 明显变化。
4. contact pattern 变化。
5. critic 判定风险最高的时刻。
6. 恢复成功或失败的证据帧。

### 4.6 sim_real_gap_memory

用于记录仿真经验的迁移风险：

```text
sim_real_gap_memory {
  sim_condition,
  pseudo_real_condition,
  mismatch_type,
  transferable_part,
  non_transferable_part,
  uncertainty_score,
  validation_evidence
}
```

在没有真机阶段，这一层可以先由 `sandbox validation + critic + perturbation test` 近似构建。

## 5. 对经验写入机制的启发

结合 RoboMemArena 和 Worth Remembering 的思想，经验写入应从“episode 结束后全量保存”改成：

```text
surprise-gated writing
+ keyframe-gated writing
+ critic-gated promotion
```

建议写入触发条件包括：

```text
memory_write_trigger {
  anomaly_detected,
  recovery_attempt_started,
  recovery_attempt_finished,
  object_state_transition,
  contact_state_transition,
  critic_warning_or_reject,
  sim_real_uncertainty_high,
  unexpected_success,
  repeated_failure
}
```

对于每次触发，系统不一定保存完整轨迹，而是保存：

```text
event_summary
+ local_window
+ keyframe
+ structured_state
+ validation_evidence
```

这样可以避免经验库变成低价值视频和日志堆积。

## 6. 对经验检索机制的启发

RoboMemory、RoboMemArena 和 Episodic Memory Verbalization 都说明：长期经验检索不能只依赖单一 top-k 文本相似度。

当前研究可以采用多通道检索：

```text
retrieval_channels {
  temporal_stage_retrieval,
  spatial_relation_retrieval,
  failure_pattern_retrieval,
  recovery_episode_retrieval,
  visual_keyframe_retrieval,
  sim_real_gap_retrieval,
  critic_case_retrieval
}
```

不同阶段使用不同检索目标：

```text
异常识别阶段:
  anomaly_state + visual_keyframe + failure_pattern

候选恢复生成阶段:
  recovery_goal + task_stage + successful_episode

候选动作排序阶段:
  failed_memory + critic_case + sim_real_gap

沙盒验证后重写阶段:
  critic_feedback + failed_plan_rewrite + local_window
```

这与 RAP 文档中的 retrieval-augmented planning 可以合并，形成：

```text
stage-aware retrieval-augmented recovery planning
```

## 7. 对长期经验层级化的启发

Episodic Memory Verbalization 说明，长期经验必须支持层级展开：

```text
month / batch summary
→ condition summary
→ episode summary
→ event window
→ keyframe / raw state
```

当前经验库可以采用：

```text
experience_tree {
  root_summary,
  scenario_nodes,
  condition_nodes,
  episode_nodes,
  event_nodes,
  keyframe_nodes
}
```

查询时，LLM 不直接读取所有 episode，而是先看摘要，再按相关性展开：

```text
query anomaly="grasp_miss at move-grasp"
→ expand scenario U3
→ expand condition U3-4
→ search event grasp_miss
→ retrieve local window around close-gripper
→ inspect keyframes and critic result
```

这种方式可以降低 prompt token 成本，也更容易解释“为什么检索到这条经验”。

## 8. 对现有 experiment-sim-wrapper 的对应关系

当前系统已有若干可以承接上述设计的字段和模块：

```text
Memory:
  memory/v3.py

Visual retrieval:
  memory/visual_retrieval.py

Reasoner:
  llm_handler.py

Executor:
  run_experiment_v4.py
  recovery_steps.py

Validation:
  sim_wrapper
  validation_evidence
  critic_result
```

已有字段中，以下内容可以直接对应多记忆系统：

```text
scenario_id
condition_id
task_stage
anomaly_state
retrieval_key
plan_signature
key_slices
visual_index
execution_feedback
failure_taxonomy
validation_evidence
critic_result
```

后续升级重点不是重新设计一套系统，而是在现有 `MemoryV3` 上增加记忆类型、关键帧索引和层级摘要。

## 9. 建议增加的经验库字段

建议在经验条目中增加如下字段：

```text
memory_tags {
  memory_type: temporal / spatial / episodic / semantic / perceptual / sim_real_gap,
  memory_scope: current_episode / condition / scenario / global,
  memory_role: success_prior / failure_risk / critic_case / visual_evidence / transfer_warning
}
```

```text
keyframe_index {
  anomaly_keyframe,
  pre_action_keyframe,
  contact_keyframe,
  critic_evidence_keyframe,
  outcome_keyframe,
  visual_embedding_id
}
```

```text
spatial_state {
  object_relations,
  robot_object_relation,
  support_surface,
  occlusion_state,
  pose_change_summary
}
```

```text
memory_dependency {
  requires_temporal_history,
  requires_spatial_history,
  requires_object_identity,
  requires_procedural_trace,
  dependency_reason
}
```

```text
promotion_status {
  raw_log,
  sandbox_validated,
  critic_validated,
  pseudo_real_validated,
  high_confidence
}
```

其中 `memory_dependency` 可以借鉴 RoboMME / RoboMemArena，用于标记某个恢复决策是否真正依赖历史。

## 10. 对实验设计的启发

可以围绕“记忆是否真的有用”设置对比：

```text
No memory:
  只用当前观测和 LLM 规划

Flat episodic memory:
  检索完整相似 episode

Stage-aware memory:
  根据异常阶段检索局部经验窗口

Multi-memory:
  同时使用 temporal / spatial / episodic / semantic / visual keyframe memory

Multi-memory + critic:
  多记忆检索后再经过沙盒和 critic 验证
```

评价指标可以包括：

```text
recovery_success_rate
task_completion_rate
invalid_plan_count
repeated_failure_rate
retrieval_precision
prompt_token_cost
memory_dependent_success_rate
critic_rejection_rate
sim_real_uncertainty_reduction
```

其中最能体现这组论文启发的是：

```text
memory_dependent_success_rate
retrieval_precision
prompt_token_cost
```

也可以仿照 RoboMemArena 增加：

```text
memory_dependent_step_ratio =
  需要历史信息才能正确决策的恢复步骤数 / 总恢复步骤数
```

该指标能说明当前任务是否真的考验经验库，而不是普通反应式控制就能完成。

## 11. 与已有启发文档的关系

这组记忆论文与前面几篇材料的关系如下：

1. Worth Remembering 解决“哪些经验值得写入”。
2. Learning From Failure 解决“失败经验如何作为风险先验”。
3. Plan in Sandbox 解决“如何用沙盒生成和验证经验”。
4. RoboCritics 解决“执行轨迹如何用 critic 检查安全性”。
5. RAP 解决“规划时如何动态检索局部经验”。
6. 机器人记忆与评测论文解决“经验库本身应如何分层、分类和评测”。

组合后，当前系统可以描述为：

```text
surprise-gated / keyframe-gated memory writing
→ multi-type memory storage
→ stage-aware retrieval
→ LLM recovery planning
→ sandbox rollout
→ motion-level critic
→ pseudo-real validation
→ high-confidence memory promotion
```

## 12. 不能直接照搬的部分

这组论文不适合直接照搬的部分包括：

1. RoboMemory 的室内导航和 VLM-VLA 框架，与当前 MuJoCo 异常恢复任务不完全一致。
2. RoboMME 的 benchmark 任务是通用操作评测，不是 Sim-Real 异常恢复。
3. RoboMemArena 的 PrediMem 需要训练 VLA 和 predictive coding head，当前阶段不一定具备训练资源。
4. Episodic Memory Verbalization 面向问答和总结，不直接解决恢复动作执行。

当前研究应吸收的是：

```text
多类型记忆架构
关键帧记忆写入
层级经验摘要
记忆依赖评测
符号记忆与感知记忆互补
```

而不是完整复现它们的模型训练流程。

## 13. 可写入当前研究方案的关键表述

后续整合到总方案时，可以使用如下表述：

```text
近期机器人记忆研究表明，长时程机器人任务中的经验不能仅以扁平 episode 日志保存。RoboMemory 将 embodied memory 分为 temporal、spatial、semantic 和 episodic memory，说明机器人经验库应同时保存当前动作历史、空间关系、任务轨迹和跨任务经验规则。RoboMME 和 RoboMemArena 进一步表明，不同机器人任务依赖不同类型的记忆，且许多长时程子任务的正确决策无法由当前观测单独确定，需要历史关键帧、对象身份、空间位置和过程轨迹共同支撑。Episodic Memory Verbalization 则说明，长期经验应组织为可逐层展开的层级结构，以便 LLM 在低 token 成本下检索到相关原始证据。

因此，本研究将 Sim-Real 双源经验库设计为多类型、层级化、关键帧增强的机器人记忆系统。系统在异常发生、状态转移、critic 警告和恢复结果出现时触发经验写入，并将成功经验、失败经验、空间关系、视觉关键帧和 Sim-Real 迁移风险分别索引。在异常恢复规划阶段，系统根据当前恢复阶段动态检索对应记忆，为 LLM 提供局部经验窗口和可解释证据，再通过沙盒推演和 motion-level critic 完成验证与经验晋升。
```

## 14. 阶段性结论

这组论文对当前研究的核心启发可以概括为：

```text
经验库不是日志仓库，而是机器人长期记忆系统；
评估经验库也不能只看最终成功率，还要看历史依赖步骤是否被正确恢复。
```

具体而言，当前研究可以吸收以下内容：

1. 将经验库拆分为 temporal、spatial、episodic、semantic、perceptual keyframe 和 sim-real gap memory。
2. 用关键帧和状态转移触发经验写入，避免全量日志污染。
3. 用层级摘要支持长期经验的低成本检索和证据展开。
4. 区分符号记忆和感知记忆：前者支撑 LLM 规划，后者支撑视觉状态和运动过程判断。
5. 增加 memory-dependent recovery step 指标，证明异常恢复确实受益于经验库。
6. 将多记忆检索与 RAP、RoboCritics、Plan in Sandbox 结合，形成完整的检索-规划-沙盒-critic-回写闭环。

因此，这组论文非常适合作为当前研究中 **多类型经验库架构、关键帧记忆、长期经验层级化、记忆检索评测和机器人异常恢复经验系统设计** 的支撑材料。
