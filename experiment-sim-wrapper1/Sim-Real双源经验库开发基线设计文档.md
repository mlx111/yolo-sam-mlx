# Sim-Real 双源经验库开发基线设计文档

## 1. 文档目标

本文档作为后续开发 **Sim-Real 双源经验库与异常恢复沙盒推演系统** 的基线文档。它综合当前文件夹中的研究方案、缺口分析、实现方案、各论文启发文档，以及 `experiment-sim-wrapper` 现有代码结构，目标是明确：

1. 这个系统到底要解决什么问题。
2. 当前已有代码能复用什么。
3. 论文和已有文档中的思想如何落到工程模块。
4. 第一版应该实现哪些字段、流程和实验。
5. 哪些内容可行，哪些内容当前不应过早实现。
6. 后续开发按什么阶段完成。

本文档后续应作为代码开发、实验设计、论文/开题表述和模块验收的共同依据。

## 2. 最终研究定位

本研究不应表述为“构建一个机器人经验库”。单纯经验库、长期记忆、失败恢复、检索增强规划、Real-to-Sim-to-Real 和沙盒推演都已有相关工作。

更准确的定位是：

```text
面向机器人异常恢复的 Sim-Real 双源异常经验库与执行前沙盒推演方法。
```

该定位包含三个不可拆开的核心点：

1. **双源经验**：经验同时来自仿真和真实机器人，而不是单一仿真 memory。
2. **差异经验化**：经验库显式保存仿真预测与真实执行之间的差异，而不是只保存成功/失败。
3. **执行前沙盒推演**：真机异常发生后，不直接执行恢复动作，而是检索经验、校准沙盒、推演候选动作、通过 critic 检查，再执行。

最终系统主张可以写成：

```text
本文提出一种面向机器人异常恢复的 Sim-Real 双源异常经验库。该经验库不仅保存仿真和真机中的异常处理经验，还显式建模二者之间的差异，并在真机恢复动作执行前，通过校准后的仿真沙盒对候选恢复策略进行推演和安全评估，从而提升真实环境下的异常恢复成功率并降低真机试错风险。
```

## 3. 已有材料的核心结论

### 3.1 当前已有总方案

当前几份总方案已经形成一致判断：

1. 当前 `experiment-sim-wrapper` 已具备仿真异常恢复实验框架。
2. 现有系统可作为最终双源经验库中的 **Simulation Memory Branch**。
3. 最终缺口不是“再做一个 memory”，而是新增：
   - 真机经验接入。
   - Sim-Real 差异签名。
   - 仿真经验与真机经验配对。
   - 真机经验校准沙盒。
   - 真机执行后回写闭环。
4. 创新点应集中在：
   - Sim-Real 双源异常经验表示。
   - 基于差异校正的执行前沙盒推演。
   - 经验检索、沙盒推演、真机执行、差异回写的闭环更新。

### 3.2 Worth Remembering

该论文支撑 **经验写入门控**。

核心启发：

```text
不是所有轨迹都值得写入长期经验库。
```

本系统应采用工程化 `memory_gate`：

```text
write_score =
  w_anomaly * anomaly_score
  + w_failure * failure_score
  + w_gap * sim_real_gap_score
  + w_utility * recovery_utility_score
```

优先写入：

1. 异常突发事件。
2. 恢复成功/失败分界点。
3. 高 Sim-Real 差异事件。
4. critic warning/reject 事件。
5. 高价值成功恢复片段。

不应长期写入：

1. 大量重复中间帧。
2. 无异常、无恢复、无状态转移的普通轨迹。
3. 与恢复无关的背景片段。

### 3.3 Learning From Failure

该论文支撑 **失败经验风险分支**。

核心启发：

```text
失败经验不是噪声，而是风险先验。
```

当前 `failed_memory blocker` 应升级为：

```text
risk-aware failure memory
```

失败风险评分建议为：

```text
failure_memory_risk =
  c1 * similar_failure_count
  + c2 * failure_similarity
  + c3 * failed_action_overlap
  + c4 * terminal_risk_score
```

失败经验不应直接提供“下次该怎么做”的答案，而应保存失败事实、失败动作签名和物理证据，作为候选动作排序和 LLM 重写的约束。

### 3.4 Reconciling Reality Through Simulation

该论文支撑 **真实异常状态同步到仿真沙盒**。

核心启发：

```text
真实环境不仅可以被感知，还可以被构造成可推演的仿真场景。
```

当前系统第一版不需要完整 3D 数字孪生，但应保留以下字段和接口：

```text
scene_twin {
  reconstruction_source,
  object_pose_set,
  collision_mesh_quality,
  physical_params,
  layout_context,
  calibration_status
}
```

以及：

```text
robot_state_sync {
  end_effector_pose,
  gripper_state,
  joint_positions,
  joint_velocities,
  contact_estimate,
  execution_timestep
}
```

### 3.5 Plan in Sandbox

该论文支撑 **无真机阶段的伪真机闭环**。

核心启发：

```text
在真机数据不足时，可以先用 physics-grounded sandbox 生成、验证和抽象经验。
```

当前 `experiment-sim-wrapper` 可以明确定位为：

```text
physics-grounded sandbox / pseudo-real validation layer
```

阶段性闭环可以是：

```text
Sandbox Experience Generation
→ Pseudo-Real Shadow Validation
→ Memory-Augmented Recovery
```

### 3.6 RoboCritics

该论文支撑 **轨迹级安全 critic**。

核心启发：

```text
沙盒推演不能只问“能不能成功”，还必须问“这个成功是否安全、稳定、可解释”。
```

需要把 `safety_risk` 拆成结构化 critic：

```text
critic_result {
  collision_risk,
  joint_speed_risk,
  joint_limit_risk,
  end_effector_pose_risk,
  pinch_point_risk,
  workspace_usage_risk,
  object_contact_risk,
  suggested_fix
}
```

第一版可先实现规则 critic，不需要训练模型。

### 3.7 RAP

该论文支撑 **按阶段动态检索经验**。

核心启发：

```text
经验不是整体拼接进 prompt，而是在每一步规划时动态检索最相关的局部经验窗口。
```

当前系统应采用：

```text
stage-aware retrieval-augmented recovery planning
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

### 3.8 RoboMemory / RoboMME / RoboMemArena / Episodic Memory Verbalization

这组论文支撑 **多类型、层级化、关键帧增强的机器人记忆系统**。

经验库不应只是扁平 episode 列表，而应至少区分：

```text
temporal_memory
spatial_memory
episodic_memory
semantic_memory
perceptual_keyframe_memory
sim_real_gap_memory
```

同时需要支持：

1. recent buffer + keyframe buffer。
2. 层级摘要，从 scenario 到 condition 到 episode 到 event window。
3. memory-dependent recovery step 指标。
4. 符号记忆和视觉关键帧互补。

### 3.9 Robotic Sim-to-Real Transfer

该论文支撑 **Sim-Real 差异签名细分**。

核心启发：

```text
Sim-Real Gap 应拆成 perception gap、actuation gap、outcome gap 和 scene reconstruction gap。
```

尤其要记录：

```text
perception_gap {
  detection_recall_drop,
  pose_estimation_noise,
  temporal_pose_jitter,
  target_lost_duration,
  background_interference
}
```

```text
actuation_gap {
  grasp_pose_gap,
  servo_final_error,
  alignment_time_gap,
  control_delay,
  gripper_closure_gap,
  sim_penetration_real_jamming
}
```

以及：

```text
outcome_gap:
  sim_success_real_fail / sim_fail_real_success
```

这可以避免把“仿真器伪失败”误当作真实风险，也可以避免把“仿真成功”误当作真机可靠。

### 3.10 RISE

RISE 对当前系统的启发主要是 **真机 episode 数据应分两层保存**：

第一层是原始 episode 数据：

```text
raw_real_episode {
  observations,
  actions,
  robot_state,
  camera_images,
  timestamps,
  task_info
}
```

第二层是异常恢复经验元数据：

```text
real_experience_metadata {
  anomaly_type,
  recovery_action,
  outcome,
  value_score,
  sim_real_gap,
  validation_evidence
}
```

短期内不应复现 RISE 的完整世界模型和在线 RL，只吸收真机数据组织、value-style 评分和真实失败经验校准沙盒评分。

## 4. 现有 `experiment-sim-wrapper` 能力盘点

当前代码已经不是一个简单仿真实验，而是具备较完整的仿真侧异常经验库原型。

### 4.1 异常条件体系

文件：

```text
../experiment-sim-wrapper/ur5e/anomaly_conditions.py
../experiment-sim-wrapper/ur5e/anomaly_injectors.py
```

当前已定义 U1-U5 共 25 类异常：

| 场景 | 覆盖问题 | 代表条件 |
| --- | --- | --- |
| U1 | 感知/目标确认异常 | 目标混淆、遮挡、位姿过期、姿态误估、边界混淆 |
| U2 | 抓取几何异常 | 抓取横向偏移、高度偏移、姿态偏差、预抓过近/过远 |
| U3 | 夹爪/夹持保持异常 | 夹爪未闭合、部分闭合、过早闭合、滑落、渐发滑移 |
| U4 | 运输/放置异常 | 运输掉落、运输位移、放置位置错误、过早释放、放置姿态错误 |
| U5 | 路径/策略异常 | 路径阻挡、接近碰撞、桌面碰撞、动作顺序错误、重复无进展 |

这些条件足够支撑第一阶段实验，不需要继续扩异常类型。

### 4.2 技能注册与执行

文件：

```text
../experiment-sim-wrapper/ur5e/skills/registry.py
../experiment-sim-wrapper/ur5e/skills/recovery_steps.py
../experiment-sim-wrapper/ur5e/skills/u1.py
../experiment-sim-wrapper/ur5e/skills/u2.py
../experiment-sim-wrapper/ur5e/skills/u3.py
../experiment-sim-wrapper/ur5e/skills/u4.py
../experiment-sim-wrapper/ur5e/skills/u5.py
```

系统已经有场景特化技能集合：

```text
U1: confirm-target, resegment-object, estimate-object-pose, validate-perception
U2: adjust-grasp-pose, adjust-pregrasp, verify-grasp
U3: check-gripper-state, set-gripper-force, recover-slip
U4: detect-place-target, create-place-pose, move-to-place-prepose, place-object, release-object, verify-placement
U5: retreat, move-safe-waypoint, replan-path, avoid-obstacle, validate-progress, switch-strategy
```

这说明后续双源经验库不需要先解决“恢复动作如何执行”，而是重点解决“如何选择、验证和回写恢复动作”。

### 4.3 MemoryV3+

文件：

```text
../experiment-sim-wrapper/memory/v3.py
```

已有核心能力：

1. `MemoryV3Entry` 已包含较完整 episode 字段。
2. 支持 `simulation_memory`、`validated_memory`、`real_memory`、`failed_memory` 分区推断。
3. 支持 `validation_status`：
   - `simulation_only`
   - `simulation_validated`
   - `real_executed`
   - `real_validated`
   - `failed`
4. 支持 `keyframes`、`key_slices`、`retrieval_key`、`anomaly_state`、`failure_taxonomy`。
5. 支持 STM/LTM 分层和自动 consolidation。
6. query 已经包含结构化评分、视觉 boost、diversity、critic prefilter。

当前最重要的判断：

```text
MemoryV3+ 不需要推倒重写，应在其上扩展 dual-source 字段。
```

### 4.4 视觉关键帧检索

文件：

```text
../experiment-sim-wrapper/memory/visual_retrieval.py
```

已有能力：

1. CLIP 提取 keyframe embedding。
2. FAISS 做视觉相似检索。
3. 按 experience_id 聚合相似度。
4. 可作为 `MemoryV3Library.query()` 的视觉加权项。

第一版双源经验库可以直接复用该机制，后续把真机关键帧写入同一视觉索引。

### 4.5 失败聚类与规则 critic

文件：

```text
../experiment-sim-wrapper/memory/failure_cluster.py
../experiment-sim-wrapper/experiment_method_runner.py
```

已有能力：

1. 失败经验可按语义 embedding 聚类。
2. `deterministic_rule_critic()` 已能产生规则化失败诊断。
3. 已有 failed plan blocker。
4. 已有 failed plan rewrite。
5. 已能统计重复失败、动作签名重叠、unsafe plan count。

这可以作为 `failure_memory_risk` 和 `critic_result` 第一版基础。

### 4.6 沙盒虚拟验证

文件：

```text
../experiment-sim-wrapper/sim_wrapper.py
../experiment-sim-wrapper/run_experiment_v4.py
```

已有能力：

1. `SimWrapper.capture_state()` / `restore_state()`。
2. `build_virtual_scene()` 根据感知位置构建虚拟场景。
3. `plan_recovery_in_virtual()` 和 `_execute_steps_in_virtual()` 执行虚拟验证。
4. `run_experiment_v4.py` 中 `_condition_a_recovery_with_llm()` 已实现：

```text
LLM plan
→ virtual scene validation
→ success: migrate execution
→ fail: no execute / fallback
```

这正是执行前沙盒推演的最小原型。

### 4.7 经验构造与回写

文件：

```text
../experiment-sim-wrapper/run_experiment_v4.py
```

`_build_experience_entry()` 已经写入：

```text
anomaly
scene
task
perception
reconstruction_artifacts
recovery_plan
execution_feedback
key_slices
keyframes
anomaly_state
retrieval_key
failure_taxonomy
validation_status
validation_evidence
```

这说明第一版开发重点不是补齐仿真侧记录，而是新增真机侧字段、配对字段和 gap 计算。

### 4.8 批量实验与 rolling memory

文件：

```text
../experiment-sim-wrapper/run_experiment_batch.py
../experiment-sim-wrapper/experiment_method_runner.py
../experiment-sim-wrapper/configs/*.json
```

已有方法：

```text
direct_llm_weak
direct_memory
sim_only_weak
sim_memory_weak
hierarchical_memory_weak
hierarchical_no_failed
```

已有 rolling memory 模式：

```text
rolling_memory_scope: cell / scenario / global
experience_save_mode: success_only / all / none
```

后续实验可以在这个框架上新增双源方法，而不需要新写批处理系统。

## 5. 系统总体架构

最终系统应由七个分支组成：

```text
1. Simulation Memory Branch
2. Real Memory Branch
3. Sim-Real Pair Branch
4. Sim-Real Gap Branch
5. Sandbox Calibration Branch
6. Critic and Risk Branch
7. Closed-loop Update Branch
```

总体流程：

```text
仿真批量异常注入
→ 生成 simulation_memory / failed_memory
→ 伪真机 shadow validation
→ 晋升 validated_memory

真机异常发生
→ 采集 raw_real_episode
→ 切片并写入 real_memory
→ 检索相似 simulation_memory / real_memory / failed_memory
→ 建立或更新 sim_real_pair
→ 计算 sim_real_gap
→ 校准 sandbox
→ 推演候选恢复动作
→ motion-level critic 检查
→ 选择候选动作并真机执行
→ 回写 real_memory、gap、pair、confidence
```

## 6. 核心数据模型

### 6.1 保留现有 MemoryV3Entry

继续使用 `MemoryV3Entry` 作为统一经验条目，不新建完全独立 schema。

原因：

1. 现有 runner、batch、analysis 已依赖该结构。
2. 字段已覆盖 episode 级经验。
3. 已支持 source、domain、validation、partition、keyframes、retrieval_key。
4. 可通过新增 dataclass 字段保持向后兼容。

### 6.2 新增顶层字段

建议在 `MemoryV3Entry` 中新增以下字段：

```text
memory_gate: MemoryGateInfo
sim_real_gap: SimRealGapInfo
sim_real_pair: SimRealPairInfo
sandbox_calibration: SandboxCalibrationInfo
critic_result: CriticResultInfo
real_episode_ref: RealEpisodeRef
memory_tags: dict
```

### 6.3 memory_gate

```text
memory_gate {
  anomaly_score: float
  failure_score: float
  sim_real_gap_score: float
  recovery_utility_score: float
  surprise_score: float
  write_score: float
  write_decision: raw_log / stm_only / ltm_candidate / high_value
  trigger_events: list[str]
}
```

第一版工程实现：

1. `anomaly_score`：有异常检测则 1，否则 0。
2. `failure_score`：恢复失败或 task 失败则 1，否则 0。
3. `sim_real_gap_score`：没有真机时为 0；有配对后由 gap 计算。
4. `recovery_utility_score`：成功且有清晰技能序列为 0.7-1.0。
5. `surprise_score`：先用规则近似，不上 V-JEPA。

### 6.4 sim_real_gap

```text
sim_real_gap {
  gap_id: string
  gap_score: float
  pose_gap: {
    object_pose_error: float
    ee_pose_error: float
    placement_error: float
  }
  contact_gap: {
    sim_contact_state: string
    real_contact_state: string
    contact_mismatch: bool
    slip_mismatch: bool
  }
  outcome_gap: {
    sim_success: bool
    real_success: bool
    type: sim_success_real_fail / sim_fail_real_success / matched_success / matched_failure
  }
  perception_gap: {
    pose_estimation_noise: float
    temporal_pose_jitter: float
    target_lost_duration: float
    detection_confidence_gap: float
  }
  actuation_gap: {
    control_delay: float
    gripper_closure_gap: float
    servo_final_error: float
    grasp_pose_gap: float
  }
  scene_reconstruction_gap: {
    object_pose_sync_error: float
    collision_shape_gap: float
    layout_gap: float
    physics_param_gap: float
  }
  uncertainty: float
  evidence: dict
}
```

第一版最小字段：

```text
pose_gap.object_pose_error
contact_gap.contact_mismatch
outcome_gap.type
actuation_gap.gripper_closure_gap
uncertainty
```

### 6.5 sim_real_pair

```text
sim_real_pair {
  pair_id: string
  sim_experience_id: string
  real_experience_id: string
  paired_by: condition_id / plan_signature / state_similarity / manual
  pair_score: float
  gap_score: float
  validation_status: paired / calibrated / invalidated
  created_at: string
  updated_at: string
}
```

配对原则：

1. `condition_id` 必须优先一致。
2. `scenario_id` 必须一致。
3. `available_actions` 必须兼容。
4. `plan_signature`、`contact_pattern`、`task_stage` 用于排序。
5. 第一版允许人工指定 pair，避免自动配对误差影响整体闭环。

### 6.6 sandbox_calibration

```text
sandbox_calibration {
  calibration_id: string
  source_gap_ids: list[str]
  object_pose_bias: list[float]
  gripper_delay_bias: float
  slip_risk_bias: float
  contact_success_bias: float
  perception_noise_bias: list[float]
  applied_to_candidate: bool
  calibration_confidence: float
}
```

第一版只做三类校准：

```text
object_pose_bias
gripper_delay_bias
slip_risk_bias
```

### 6.7 critic_result

```text
critic_result {
  overall_status: pass / warning / reject
  critic_risk_score: float
  collision: {
    status: pass / warning / reject
    collision_count: int
    min_distance: float
    involved_objects: list[str]
  }
  joint: {
    status: pass / warning / reject
    max_joint_delta: float
    joint_limit_violation: bool
  }
  gripper_contact: {
    status: pass / warning / reject
    contact_pattern: string
    pinch_distance: float
    object_drop_risk: float
  }
  end_effector_pose: {
    status: pass / warning / reject
    pose_error: float
    risky_segment: string
  }
  rule_flags: list[dict]
  feedback_for_rewrite: string
}
```

第一版可直接从 `deterministic_rule_critic()` 和 `metrics` 组装。

### 6.8 real_episode_ref

原始真机数据不应全部塞入经验 JSON。建议保存引用：

```text
real_episode_ref {
  raw_episode_id: string
  hdf5_path: string
  video_dir: string
  keyframe_dir: string
  robot_log_path: string
  time_range: [start, end]
  sensor_modalities: list[str]
}
```

经验库保存结构化索引和摘要，原始数据另存为 HDF5/目录。

### 6.9 memory_tags

```text
memory_tags {
  memory_type: temporal / spatial / episodic / semantic / perceptual / sim_real_gap
  memory_scope: current_episode / condition / scenario / global
  memory_role: success_prior / failure_risk / critic_case / visual_evidence / transfer_warning
  memory_dependency: {
    requires_temporal_history: bool
    requires_spatial_history: bool
    requires_object_identity: bool
    requires_procedural_trace: bool
    dependency_reason: string
  }
}
```

该字段用于后续分析“哪些恢复步骤确实依赖经验库”。

## 7. 经验分区设计

继续使用现有 `memory_partition`，但语义扩展如下：

| 分区 | 含义 | 写入来源 |
| --- | --- | --- |
| `simulation_memory` | 仿真恢复经验，尚未通过严格验证 | 普通仿真执行 |
| `validated_memory` | 通过 sim_wrapper / pseudo-real 验证的仿真经验 | 沙盒验证成功 |
| `failed_memory` | 失败经验与失败诊断 | 仿真、伪真机、真机失败 |
| `real_memory` | 真实机器人执行经验 | 真机 episode |
| `gap_memory` | Sim-Real 差异经验 | sim-real pair 计算 |
| `critic_memory` | critic 发现的风险案例 | 沙盒或真机 critic |

实现上可以先不新增分区名，只在 `memory_tags.memory_role` 中表达 `gap_memory` / `critic_memory`，避免破坏现有 `infer_memory_partition()`。

## 8. 检索与候选动作评分

### 8.1 检索阶段

第一层硬过滤：

```text
scenario_id match
condition_id match when available
available_actions compatible
```

第二层结构化排序：

```text
validation_status_score
result_success
retrieval_key_similarity
anomaly_state_similarity
task_stage_match
text_summary_similarity
visual_similarity
memory_tier_boost
```

第三层双源增强：

```text
real_success_prior
failure_memory_risk
sim_real_gap_uncertainty
critic_case_similarity
```

### 8.2 候选动作来源

候选恢复动作来自三类：

1. LLM 根据当前上下文生成。
2. 成功经验中的 `skill_sequence`。
3. 规则化恢复模板或技能抽象。

第一版不需要做复杂采样器。可以先保留当前 LLM 单计划，新增“评分解释”；第二版再扩展为多候选。

### 8.3 候选动作评分函数

最终评分：

```text
candidate_score =
  w1 * sim_success_score
  + w2 * real_success_prior
  + w3 * retrieved_success_prior
  - w4 * failure_memory_risk
  - w5 * sim_real_gap_uncertainty
  - w6 * critic_risk_score
  - w7 * recovery_time_cost
```

第一版可用：

```text
candidate_score =
  1.0 * virtual_validation_success
  + 0.5 * real_success_prior
  + 0.3 * retrieved_success_prior
  - 0.5 * failed_action_overlap
  - 0.5 * sim_real_gap_uncertainty
  - 0.5 * critic_risk_score
```

没有真机时：

```text
real_success_prior = 0
sim_real_gap_uncertainty = pseudo_real_gap_uncertainty
```

## 9. 沙盒校准机制

### 9.1 当前已有基础

当前 `SimWrapper.build_virtual_scene()` 已能根据感知位置创建虚拟场景。它是沙盒校准的入口。

### 9.2 第一版校准输入

```text
current_anomaly_state
retrieved_real_memory
retrieved_gap_memory
sim_real_pair
```

### 9.3 第一版校准输出

```text
calibrated_state {
  calibrated_object_pose
  calibrated_gripper_delay
  slip_risk_bias
  calibration_confidence
}
```

### 9.4 第一版校准规则

1. 如果同类真机经验显示目标位置平均偏差为 `[dx, dy, dz]`，则虚拟场景初始化时加入 `object_pose_bias`。
2. 如果同类真机经验显示夹爪闭合不足，则提高 `gripper_closure_gap` 和 `slip_risk_bias`。
3. 如果同类经验常出现 `sim_success_real_fail`，则提高 `sim_real_gap_uncertainty`，即使沙盒成功也降低候选评分。
4. 如果同类经验常出现 `sim_fail_real_success`，则标记 `sim_model_artifact`，避免过度保守。

### 9.5 不建议第一版实现的内容

第一版不建议实现：

1. 残差动力学模型。
2. 在线 RL fine-tuning。
3. 完整 3D scene reconstruction。
4. V-JEPA surprise embedding。
5. predictive coding keyframe head。

这些内容可作为后续论文拓展，不应阻塞最小闭环。

## 10. 现有代码改造清单

### 10.1 `memory/v3.py`

新增 dataclass：

```text
MemoryGateInfo
SimRealGapInfo
SimRealPairInfo
SandboxCalibrationInfo
CriticResultInfo
RealEpisodeRef
```

修改：

1. `MemoryV3Entry` 增加上述字段。
2. `entry_from_dict()` 保持向后兼容。
3. `build_retrieval_key()` 增加：
   - gap_type
   - critic_flags
   - memory_role
   - real_validated flag
4. `validation_status_score()` 保留现有排序。
5. query 中增加可选 `gap_aware=True` 和 `risk_aware=True`。

### 10.2 `run_experiment_v4.py`

修改：

1. `_build_experience_entry()` 写入：
   - `memory_gate`
   - `critic_result`
   - `sandbox_calibration`
2. `_condition_a_recovery_with_llm()` 记录：
   - candidate_score
   - virtual critic
   - calibration_used
3. `_query_recovery_experiences()` 支持 stage-aware retrieval 参数。
4. `save_experience()` 支持写入门控：
   - `none`
   - `stm_only`
   - `ltm_candidate`

### 10.3 `experiment_method_runner.py`

修改：

1. `deterministic_rule_critic()` 输出结构化 `critic_result`。
2. failed memory blocker 输出 `failure_memory_risk`。
3. 新增方法：

```text
sim_real_gap_memory_weak
dual_source_memory_weak
dual_source_gap_critic
```

4. metrics 增加：

```text
candidate_score
failure_memory_risk
sim_real_gap_uncertainty
critic_risk_score
real_success_prior
memory_dependent_step
```

### 10.4 `sim_wrapper.py`

修改：

1. `build_virtual_scene()` 接受 `sandbox_calibration`：

```text
object_pose_bias
gripper_delay_bias
slip_risk_bias
```

2. 虚拟执行返回更完整 trace：

```text
trajectory_trace
contact_trace
min_distance_trace
joint_delta_trace
```

3. 新增：

```text
score_virtual_rollout()
```

用于汇总 virtual success、critic、risk。

### 10.5 新增 `memory/dual_source.py`

职责：

```text
pair_sim_real_experiences()
compute_sim_real_gap()
update_gap_memory()
estimate_real_success_prior()
estimate_gap_uncertainty()
```

第一版可以只做规则计算。

### 10.6 新增 `memory/gating.py`

职责：

```text
compute_memory_gate()
should_write_experience()
explain_write_decision()
```

### 10.7 新增 `memory/critic.py`

职责：

```text
build_critic_result_from_metrics()
score_critic_risk()
build_feedback_for_rewrite()
```

第一版可从 `experiment_method_runner.py` 中迁出 deterministic critic。

### 10.8 新增 `real_memory/`

建议新增目录：

```text
../experiment-sim-wrapper/real_memory/
  schema.py
  ingest.py
  pair.py
  gap.py
```

职责：

1. 读取真机 episode 元数据。
2. 生成 `MemoryV3Entry(source="real")`。
3. 与仿真经验配对。
4. 计算 gap。

如果短期没有真机，可先用 pseudo-real JSON 走同一接口。

## 11. 可行性检查

### 11.1 已具备条件

当前已经具备：

1. MuJoCo UR5e 异常恢复环境。
2. U1-U5 25 类异常 benchmark。
3. 结构化经验库 `memory_v3_plus`。
4. 成功/失败经验分区。
5. rolling memory。
6. sim_wrapper 虚拟验证。
7. keyframes 和视觉检索。
8. failed memory blocker / rewrite。
9. deterministic rule critic。
10. 批量实验和多方法对比。

因此，开发双源经验库不是从零开始，主要是扩展。

### 11.2 最大缺口

最大缺口仍然是：

```text
真实机器人经验数据。
```

没有真机数据时，只能完成：

```text
sandbox-first / pseudo-real dual-source prototype
```

这仍然有价值，但论文表述必须诚实：

1. 阶段一是仿真侧 + shadow validation。
2. 阶段二接入少量真机。
3. 阶段三才是完整 real validated 双源闭环。

### 11.3 第一版可行路线

第一版完全可行，因为它只需要在现有代码上新增：

1. schema 字段。
2. 写入门控。
3. 规则 gap 计算。
4. 规则 sandbox calibration。
5. 结构化 critic_result。
6. 批量实验指标。

这些不需要新训练模型，也不需要大量真机。

### 11.4 风险与控制

| 风险 | 影响 | 控制方式 |
| --- | --- | --- |
| 真机数据不足 | 无法证明 real branch | 先做 pseudo-real，后续采集 2 类异常 × 5-10 条 |
| schema 过大 | 开发拖慢 | 先实现最小字段，其余放 metadata |
| 自动配对误配 | gap 计算污染 | 第一版支持人工 pair / condition 硬过滤 |
| 沙盒校准过度 | 候选策略过保守 | 校准只影响评分，不直接覆盖执行 |
| critic 误拒绝 | 成功动作被拒 | 第一版分 pass/warning/reject，warning 不强制拦截 |
| LLM 计划不稳定 | 实验噪声大 | 使用现有 weak baseline + rolling memory 多 trial |
| 经验库保存过多 | 检索噪声 | 上 memory_gate 和 STM/LTM consolidation |

## 12. 最小可实现版本

### 12.1 MVP 目标

MVP 要证明：

```text
双源/伪双源经验 + gap/risk/critic 信息
能比单纯仿真经验更可靠地选择异常恢复动作。
```

### 12.2 MVP 输入

1. 现有仿真异常经验。
2. shadow validation 结果。
3. 少量手工构造 pseudo-real 或真实经验。
4. U3 / U4 中 2-3 个条件。

推荐先选：

```text
U3-4 early_lift_slip
U4-3 wrong_placement_position
U5-2 approach_collision_neighbor
```

如果需要更小：

```text
U3-4 + U4-3
```

### 12.3 MVP 方法对比

```text
No memory:
  direct_llm_weak

Sim only:
  sim_memory_weak

Sim + failed:
  hierarchical_memory_weak

Sim + pseudo-real gap:
  dual_source_gap_memory

Sim + pseudo-real gap + critic:
  dual_source_gap_critic
```

### 12.4 MVP 指标

```text
recovery_success_rate
task_success_rate
virtual_validation_success_rate
invalid_plan_count
unsafe_motion_count
failed_plan_block_rate
rewrite_success_rate
critic_rejection_rate
sim_real_prediction_error
gap_uncertainty_reduction
memory_dependent_success_rate
retrieval_precision
prompt_token_cost
```

第一版核心指标：

```text
recovery_success_rate
unsafe_motion_count
repeated_failure_rate
sim_real_prediction_error
```

## 13. 完成计划

### Phase 0：冻结基线与数据检查

目标：

```text
确认现有 experiment-sim-wrapper 能稳定运行，并锁定第一版实验条件。
```

任务：

1. 固定基线 commit / 目录状态。
2. 选择 MVP 条件：建议 `U3-4`、`U4-3`。
3. 运行已有 smoke 配置，确认 `direct_llm_weak`、`sim_only_weak`、`sim_memory_weak`、`hierarchical_memory_weak` 正常。
4. 保存一份 clean memory snapshot。
5. 确认 `result.json`、`experience_after.json`、`keyframes` 都能正常生成。

产物：

```text
baseline_run_report.md
memory_empty_or_seed.json
first_mvp_config.json
```

验收：

1. 每个方法至少 1-3 次 smoke 能跑通。
2. `experience_after.json` 中有 `validation_status`、`keyframes`、`retrieval_key`。

### Phase 1：扩展 MemoryV3 schema

目标：

```text
在不破坏旧数据的情况下加入双源经验字段。
```

任务：

1. 在 `memory/v3.py` 新增：
   - `MemoryGateInfo`
   - `SimRealGapInfo`
   - `SimRealPairInfo`
   - `SandboxCalibrationInfo`
   - `CriticResultInfo`
   - `RealEpisodeRef`
2. `MemoryV3Entry` 增加对应字段。
3. 确保旧 JSON 读取不报错。
4. 更新 `entry_to_dict()` / `entry_from_dict()`。
5. 写一个小型 schema smoke tool。

产物：

```text
memory/v3.py updated
tools/smoke_memory_dual_source_schema.py
```

验收：

1. 旧 `memory_v3_plus` JSON 能 load/save。
2. 新字段默认值完整。
3. 现有 batch 不受影响。

### Phase 2：经验写入门控

目标：

```text
避免所有 episode 无差别写入长期经验库。
```

任务：

1. 新增 `memory/gating.py`。
2. 实现 `compute_memory_gate(metrics, entry)`。
3. 在 `_build_experience_entry()` 写入 `memory_gate`。
4. 在 `save_experience()` 支持 gate decision。
5. 保留配置开关，默认先不强制丢弃，先只记录门控结果。

产物：

```text
memory/gating.py
memory_gate fields in experience JSON
```

验收：

1. 成功、失败、虚拟验证失败三种情况 gate 分数不同。
2. gate 解释可读。
3. 不影响当前 rolling memory 实验。

### Phase 3：结构化 critic_result

目标：

```text
把已有 deterministic_rule_critic 升级成可存储、可评分、可重写反馈的 critic_result。
```

任务：

1. 新增 `memory/critic.py`。
2. 从 `experiment_method_runner.py` 迁移或封装 `deterministic_rule_critic()`。
3. 生成统一 `critic_result`：
   - `overall_status`
   - `critic_risk_score`
   - `rule_flags`
   - `feedback_for_rewrite`
4. 在失败经验和虚拟验证失败经验中写入。
5. 在 candidate score 中预留 `critic_risk_score`。

产物：

```text
memory/critic.py
critic_result in experience JSON
critic metrics in result.json
```

验收：

1. 虚拟验证失败时有 `virtual_validation_failed` flag。
2. 抓取无接触时有 `no_contact_detected` flag。
3. critic_result 能进入 failure_taxonomy 或独立顶层字段。

### Phase 4：Sim-Real pair 与 gap 计算

目标：

```text
实现双源经验库区别于普通经验库的核心字段。
```

任务：

1. 新增 `memory/dual_source.py`。
2. 实现经验配对：
   - condition_id match
   - scenario_id match
   - plan_signature similarity
   - contact_pattern match
3. 实现 gap 计算：
   - pose_gap
   - contact_gap
   - outcome_gap
   - gripper_closure_gap
   - uncertainty
4. 支持 pseudo-real pair：
   - simulation result vs shadow validation result
   - simulation result vs perturbed simulation result
5. 将 `sim_real_pair`、`sim_real_gap` 写入 entry 或独立 gap entry。

产物：

```text
memory/dual_source.py
tools/build_sim_real_pairs.py
gap_report.json
```

验收：

1. 同 condition 下能生成 pair。
2. `sim_success_real_fail` 和 `matched_success` 能被正确标记。
3. gap_score 可用于排序。

### Phase 5：沙盒校准

目标：

```text
让真机/伪真机经验影响沙盒初始化和候选评分。
```

任务：

1. `sim_wrapper.build_virtual_scene()` 支持 calibration 参数。
2. 实现：
   - `object_pose_bias`
   - `gripper_delay_bias`
   - `slip_risk_bias`
3. 在 `_condition_a_recovery_with_llm()` 中读取 gap memory 并构造 calibration。
4. 校准结果写入 `sandbox_calibration`。
5. 第一版校准只影响：
   - 虚拟场景初始物体 pose。
   - candidate score。
   - critic 风险，不强制改 LLM 计划。

产物：

```text
sim_wrapper.py updated
sandbox_calibration in result.json / experience JSON
```

验收：

1. 有 gap 时 calibration 被应用。
2. 无 gap 时行为与原系统一致。
3. calibration 有可解释日志。

### Phase 6：双源检索与候选评分

目标：

```text
让 real/pseudo-real success prior、failure risk、gap uncertainty 和 critic risk 共同影响候选策略。
```

任务：

1. 扩展 `MemoryV3Library.query()` 或新增 wrapper：
   - `query_success_memory`
   - `query_failure_memory`
   - `query_gap_memory`
   - `query_critic_cases`
2. 实现 `estimate_real_success_prior()`。
3. 实现 `estimate_gap_uncertainty()`。
4. 实现 `score_candidate_plan()`。
5. 在 result 中保存 scoring breakdown。

产物：

```text
candidate_score breakdown
dual_source retrieval records
```

验收：

1. 仿真成功但 pseudo-real 失败的方案分数下降。
2. 历史失败动作重叠高的方案分数下降。
3. critic warning/reject 能影响排序或执行决策。

### Phase 7：真机经验接入

目标：

```text
把真实执行数据以最小成本接入 real_memory。
```

任务：

1. 新增 `real_memory/schema.py`。
2. 新增 `real_memory/ingest.py`，读取手工/自动标注 JSON。
3. 真机经验最小字段：
   - condition_id
   - scenario_id
   - task_stage
   - observed_pos
   - executed_recovery_steps
   - recovery_success
   - failure_reason
   - keyframes
   - real_episode_ref
4. 生成 `MemoryV3Entry(source="real", validation_status="real_executed")`。
5. 与 simulation memory 建 pair。

产物：

```text
real_memory/ingest.py
real_memory_example.json
real_memory_snapshot.json
```

验收：

1. 至少 5 条 real/pseudo-real entry 可导入。
2. 导入后 query 能检索到 real_memory。
3. pair/gap 可计算。

### Phase 8：MVP 实验

目标：

```text
证明双源 gap + critic 比单纯仿真经验更可靠。
```

任务：

1. 新增 batch config：

```text
configs/mvp_dual_source_gap_critic_u3_u4_v1.json
```

2. 方法：

```text
direct_llm_weak
sim_only_weak
sim_memory_weak
hierarchical_memory_weak
dual_source_gap_memory
dual_source_gap_critic
```

3. 条件：

```text
U3-4
U4-3
```

4. 每组至少 5-10 trials。
5. 输出分析：
   - success rate
   - unsafe count
   - repeated failure
   - gap prediction error
   - critic rejection
   - memory usefulness

产物：

```text
results/mvp_dual_source_gap_critic_u3_u4_v1/
analysis/mvp_dual_source_report.md
```

验收：

1. 双源方法不低于 hierarchical baseline。
2. 在存在 pseudo-real gap 的案例中，双源方法能降低错误执行或重复失败。
3. 结果能解释“为什么某个候选被降权”。

### Phase 9：文档与论文材料固化

目标：

```text
把工程实现转成论文/开题可用材料。
```

任务：

1. 更新系统架构图。
2. 整理 schema 表。
3. 整理方法对比表。
4. 整理可行性与消融结果。
5. 将所有关键字段截图/JSON 示例写入附录。

产物：

```text
Sim-Real双源经验库系统说明.md
Sim-Real双源经验库实验报告.md
论文方法章节草稿.md
```

## 14. 开发优先级

最高优先级：

1. schema 扩展但保持兼容。
2. critic_result 结构化。
3. memory_gate 记录。
4. sim_real_pair / sim_real_gap 规则计算。
5. pseudo-real gap 实验。

中优先级：

1. sandbox calibration。
2. dual-source candidate scoring。
3. real_memory ingest。
4. gap-aware retrieval。

低优先级：

1. 自动技能抽象。
2. V-JEPA surprise。
3. predictive coding keyframe selection。
4. 残差动力学。
5. 在线训练或 RL fine-tuning。

## 15. 最终结论

基于现有文档和代码，建立 Sim-Real 双源经验库是可行的，但必须避免范围过大。

当前最稳妥的路线是：

```text
不要重写 experiment-sim-wrapper；
以 MemoryV3+ 和 sim_wrapper 为基础，
新增 real_memory、sim_real_pair、sim_real_gap、sandbox_calibration、critic_result 和 memory_gate，
先用 pseudo-real/少量真机经验证明双源差异经验能改善异常恢复决策。
```

第一版系统的核心交付应是：

```text
一个可运行的异常恢复闭环：

异常检测
→ 多源经验检索
→ 失败风险和 gap 风险评分
→ 校准沙盒推演
→ motion-level critic
→ 执行/拒绝/重写
→ 经验与差异回写
```

这样既能充分吸收当前文件夹中所有论文启发，也能贴合 `experiment-sim-wrapper` 的现有工程基础，并为后续真实机器人接入留下清晰接口。
