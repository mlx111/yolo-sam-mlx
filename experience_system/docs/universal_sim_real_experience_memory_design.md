# 通用 Sim-Real 双源经验库生成方案

## 1. 目标

本文档给出一个通用经验库设计，使现有机械臂、移动双臂机器人、仿真后端和真机都能使用同一套经验库。

核心目标：

```text
不同机器人负责把自己的执行记录归一化为通用 ExperienceEntry。
经验库核心只负责存储、检索、门控、critic、sim-real 配对、gap 计算和沙盒评分。
```

也就是说，经验库不应该写死为 `R1ProMemory` 或 `UR5Memory`。R1Pro、UR5e、Isaac、MuJoCo、真机都通过 adapter 接入统一 schema。

## 2. 依据

### 2.1 本地实现依据

`../ur5e_mujoco` 已经实现了一套 UR5e 异常恢复经验库，关键模块包括：

```text
memory/v3.py
memory/gating.py
memory/scoring.py
memory/critic.py
memory/dual_source.py
memory/calibration.py
memory/visual_retrieval.py
memory/failure_cluster.py
real_memory/schema.py
real_memory/ingest.py
real_memory/collector.py
```

这些模块已经证明以下结构可行：

1. `MemoryV3Entry` 作为统一经验条目。
2. `source=simulation/real` 共用 schema。
3. `memory_gate` 记录经验写入价值。
4. `critic_result` 记录规则或 LLM critic 风险。
5. `sim_real_pair` 和 `sim_real_gap` 记录仿真与真实差异。
6. `sandbox_calibration` 用 gap memory 修正沙盒。
7. `real_episode_ref` 用外部引用保存真机原始数据。

当前 `galaxea_mujoco` 已经有 R1Pro 侧的最小版本：

```text
memory/v1.py
memory/gating.py
source/run_r1pro_task_chain.py
results/memory/r1pro_memory_v1.json
```

但这还是 R1Pro 局部版本。下一步应把它升级为通用核心。

### 2.2 论文和网络资料依据

近期机器人记忆和评测工作给出几个共同结论：

1. RoboMemory 强调机器人长期记忆应拆成 spatial、temporal、episodic、semantic 多模块，并在闭环规划中配合 critic 使用。
2. RoboMME 将机器人记忆评测拆成 temporal、spatial、object、procedural memory，说明单一文本记忆不足以覆盖长时程操作。
3. RoboMemArena 强调 recent buffer、keyframe buffer 和 memory-dependent subtask，说明长期任务不能只保存完整视频或完整日志。
4. Episodic Memory Verbalization 使用层级 episodic tree，从 raw data 到 event/goal/summary 分层组织经验，避免长期记忆检索成本失控。
5. Worth Remembering 提出 surprise-gated memory，说明长期记忆必须选择性写入。
6. Sim-to-real long-horizon pick-and-place 工作强调 perception 和 actuation discrepancy 是真实迁移中的主要 gap，应被显式记录。

这些结论支持本文档的核心设计：

```text
通用经验库 = 多类型记忆 + 选择性写入 + 结构化检索 + sim-real gap + sandbox calibration
```

## 3. 总体架构

推荐架构：

```text
experience_core/
  schema.py
  library.py
  gating.py
  retrieval.py
  scoring.py
  critic.py
  dual_source.py
  calibration.py
  visual_index.py
  failure_cluster.py

experience_adapters/
  r1pro_mujoco.py
  ur5e_wrapper.py
  isaac_robot.py
  real_episode.py

task_runners/
  r1pro_task_chain_runner.py
  ur5e_experiment_runner.py
  real_robot_episode_collector.py
```

数据流：

```text
robot/backend runner
-> adapter.normalize(...)
-> ExperienceEntry
-> MemoryGate
-> MemoryLibrary
-> Retrieval + Critic + Scoring
-> Sandbox calibration
-> Candidate plan selection
-> Execution
-> Write back result/gap
```

## 4. 核心原则

### 4.1 经验库核心不依赖机器人

核心库不能 import：

```text
R1Pro 技能
UR5e 环境
MuJoCo model
Isaac API
ROS topic
具体夹爪驱动
```

核心库只能处理通用数据：

```text
ExperienceEntry
RobotState
ObjectState
SkillTrace
SensorSummary
SimRealGap
CriticResult
```

### 4.2 机器人差异由 adapter 处理

每个机器人 adapter 负责把自己的日志转成统一字段。

例子：

```text
R1Pro adapter:
  base_pose
  torso_posture
  left_tcp_pose
  right_tcp_pose
  dual_arm_state
  attach_mode

UR5e adapter:
  arm_qpos
  ee_pose
  gripper_width
  contact_after_close
  contact_after_lift

real adapter:
  controller_log
  camera_refs
  force_torque
  gripper_driver_state
  raw_episode_ref
```

adapter 的输出必须是同一个 `ExperienceEntry`。

### 4.3 原始数据外部引用，经验库保存摘要

经验库 JSON 不应塞完整高频轨迹、完整视频或完整点云。通用做法：

```text
ExperienceEntry:
  保存关键摘要、关键帧路径、指标、状态、索引字段

raw data:
  保存到 HDF5 / rosbag / video dir / log jsonl / npy
  由 raw_refs 或 real_episode_ref 引用
```

这样经验库可在线检索，同时保留离线复核能力。

## 5. 通用 ExperienceEntry

建议的核心 schema：

```text
ExperienceEntry {
  schema_version
  experience_id
  created_at
  updated_at

  source
  domain
  backend
  robot
  embodiment

  scenario
  condition
  task
  anomaly

  skill_sequence
  action_trace
  observation_trace

  state_before
  state_after
  sensor_summary
  spatial_state
  object_state

  result
  execution_feedback
  key_slices
  keyframes

  retrieval_key
  memory_tags
  memory_gate
  memory_partition

  critic_result
  failure_taxonomy

  sim_real_pair
  sim_real_gap
  sandbox_calibration

  raw_refs
}
```

其中 `source` 和 `backend` 分开：

```text
source:
  simulation / real / pseudo_real

backend:
  mujoco / isaac / real_robot / replay
```

例如：

```json
{
  "source": "simulation",
  "backend": "mujoco",
  "robot": {
    "robot_id": "r1pro_mujoco_001",
    "robot_type": "mobile_dual_arm"
  }
}
```

## 6. 通用 Robot 字段

不要把机器人状态压成一个固定机械臂格式。建议用可扩展结构：

```text
robot {
  robot_id
  robot_type
  embodiment_tags
  kinematic_groups
  end_effectors
  mobile_base
  torso
  grippers
}
```

例子：

```json
{
  "robot_id": "r1pro_mujoco_001",
  "robot_type": "mobile_dual_arm",
  "embodiment_tags": ["mobile_base", "torso", "dual_arm", "parallel_gripper"],
  "kinematic_groups": {
    "left_arm": {"joint_names": ["..."]},
    "right_arm": {"joint_names": ["..."]}
  },
  "end_effectors": {
    "left": {"pose": [0.1, 0.2, 0.9, 1, 0, 0, 0]},
    "right": {"pose": [0.1, -0.2, 0.9, 1, 0, 0, 0]}
  },
  "mobile_base": {"pose": [-0.25, 0.0, 0.0]},
  "torso": {"posture": [0.0, 0.35, -0.25, 0.0]},
  "grippers": {
    "left": {"state": "closed", "width": 0.025},
    "right": {"state": "closed", "width": 0.025}
  }
}
```

UR5e 可以省略 `mobile_base` 和 `torso`。

## 7. 通用 SkillTrace

技能记录不要依赖具体类名，但要保留原始技能名。

```text
SkillTraceItem {
  name
  primitive_type
  phase
  inputs
  outputs
  success
  error
  duration
  safety_flags
}
```

例子：

```json
{
  "name": "dual_arm_pregrasp",
  "primitive_type": "reach",
  "phase": "pregrasp",
  "success": true,
  "error": 0.00024,
  "outputs": {
    "left_tcp_error": 0.00024,
    "right_tcp_error": 0.00024
  }
}
```

这样检索时既能按抽象类型查，也能按具体技能名查。

## 8. 记忆类型

每条经验可以同时带多个 memory tag：

```text
memory_tags {
  memory_type
  memory_scope
  memory_role
  confidence
}
```

建议的 `memory_type`：

```text
temporal
spatial
episodic
semantic
procedural
perceptual_keyframe
sim_real_gap
failure
critic_case
```

建议的 `memory_role`：

```text
success_prior
failure_case
risk_warning
gap_memory
visual_evidence
calibration_evidence
recovery_template
```

一条完整 episode 通常是 `episodic`；由多条 episode 总结出的规则才是 `semantic` 或 `procedural`。

## 9. 写入门控

通用经验库应使用 deterministic gate 作为第一版，不要一开始训练复杂模型。

门控输入：

```text
anomaly_score
failure_score
sim_real_gap_score
recovery_utility_score
surprise_score
critic_risk_score
```

门控输出：

```text
raw_log
stm_only
ltm_candidate
high_value
```

触发条件：

```text
anomaly_detected
recovery_started
recovery_finished
task_failed
unexpected_success
contact_state_changed
object_state_changed
critic_warning
sim_real_gap_high
repeated_failure
```

这对应 `experience_system/memory/gating.py` 的现有做法，也符合 surprise-gated memory 的研究方向。

## 10. 检索

检索不能只用文本 embedding。推荐四层检索。

### 10.1 硬过滤

```text
scenario_id
task_stage
robot_type
embodiment_tags
source
memory_partition
```

例如，双臂搬运异常不应优先召回单臂抓取经验。

### 10.2 结构化相似度

```text
condition_id
object_class
failure_type
plan_signature
contact_pattern
place_site_state
attach_mode
control_mode
```

### 10.3 时空和视觉相似度

```text
object_pose_distance
end_effector_pose_distance
spatial_relation_match
keyframe_similarity
occlusion_state_match
```

视觉索引可以后接 CLIP/FAISS，但第一版可以只保留接口。

### 10.4 gap/risk 调整

```text
real_validation_bonus
critic_risk_penalty
gap_uncertainty_penalty
sim_success_real_fail_penalty
failure_overlap_penalty
```

这对应 `experience_system/memory/scoring.py` 的现有设计。

## 11. Critic

critic 输出要通用，不要绑定具体机器人。

```text
CriticResult {
  overall_status
  critic_risk_score
  rule_flags
  feedback_for_rewrite
  evidence
}
```

通用 rule flags：

```text
collision_risk
joint_limit_risk
workspace_unreachable
gripper_contact_missing
object_not_lifted
object_slip
place_zone_miss
perception_inconsistent
plan_missing_required_phase
sim_real_gap_high
```

机器人 adapter 可以补充机器人专属 flags：

```text
R1Pro:
  dual_arm_height_mismatch
  base_pose_incompatible
  torso_height_incompatible

UR5e:
  pinch_too_wide
  contact_lost_during_lift
```

## 12. Sim-Real 配对

通用配对函数：

```text
pair_score(sim_entry, real_entry)
```

基础配对信号：

```text
scenario_id match
condition_id match
robot_type compatible
task_stage match
object_class match
plan_signature match
contact_pattern match
```

配对输出：

```text
sim_real_pair {
  pair_id
  sim_experience_id
  real_experience_id
  paired_by
  pair_score
  validation_status
}
```

同一真实经验只选一个最佳 sim pair，避免一条真机数据被重复计入 gap。

## 13. Sim-Real Gap

通用 gap 结构：

```text
sim_real_gap {
  gap_id
  gap_score
  uncertainty
  outcome_gap
  pose_gap
  contact_gap
  perception_gap
  actuation_gap
  robot_state_gap
  timing_gap
  scene_reconstruction_gap
  evidence
}
```

关键 gap 类型：

```text
matched_success
matched_failure
sim_success_real_fail
sim_fail_real_success
```

不同机器人都能填写这些字段，只是证据来源不同。

R1Pro 的 gap 重点：

```text
base_pose_gap
torso_posture_gap
left_tcp_pose_gap
right_tcp_pose_gap
dual_arm_height_gap
object_pose_gap
place_pose_gap
attach/contact_gap
```

UR5e 的 gap 重点：

```text
ee_pose_gap
arm_qpos_gap
gripper_width_gap
contact_after_close_gap
contact_after_lift_gap
object_z_gap
```

## 14. 沙盒校准

沙盒校准不应该直接修改经验库条目，而是由检索到的 gap memories 生成临时校准：

```text
sandbox_calibration {
  calibration_id
  source_gap_ids
  object_pose_bias
  perception_noise_bias
  actuation_delay_bias
  contact_success_bias
  slip_risk_bias
  calibration_confidence
}
```

第一版使用规则加权平均：

```text
high pair_score
+ low uncertainty
+ high gap_score
=> higher calibration weight
```

后续真机数据足够后，再训练 residual model。

## 15. Adapter 接口

建议定义一个很薄的 adapter 协议：

```python
class ExperienceAdapter:
    robot_type: str
    backend: str

    def normalize_episode(self, raw_episode: dict) -> ExperienceEntry:
        ...

    def build_retrieval_key(self, entry: ExperienceEntry) -> dict:
        ...

    def extract_pairing_features(self, entry: ExperienceEntry) -> dict:
        ...

    def compute_robot_specific_gap(self, sim_entry: ExperienceEntry, real_entry: ExperienceEntry) -> dict:
        ...
```

### 15.1 R1Pro adapter

输入：

```text
TaskChainResult from source/run_r1pro_task_chain.py
```

输出：

```text
ExperienceEntry
```

重点映射：

```text
scenario_id = G3/G4/G2/grasp
robot_type = mobile_dual_arm
backend = mujoco
embodiment_tags = mobile_base, torso, dual_arm, gripper
skill_sequence = task result skill_trace
object_state = target object start/final pose
spatial_state = selected place site / occupancy state
execution_feedback = object lift / final place error / selected recovery branch
```

### 15.2 UR5e wrapper adapter

输入：

```text
ur5e_mujoco MemoryV3Entry or result.json
```

输出：

```text
ExperienceEntry
```

重点映射：

```text
robot_type = fixed_single_arm
backend = mujoco
skill_sequence = executed_recovery_steps
object_state = apple/pear/plate states
execution_feedback = contact_after_close / contact_after_lift / apple_z_after_recovery
```

### 15.3 Real robot adapter

输入：

```text
episode.json
sensor_summary.json
robot_log.jsonl
keyframes/
video/
hdf5/rosbag path
```

输出：

```text
ExperienceEntry(source="real")
```

必须保留：

```text
raw_refs
real_episode_ref
sensor_modalities
time_range
operator/device metadata
```

## 16. 生成流程

通用经验库生成流程：

```text
1. runner 执行任务或读取真机 episode
2. adapter 归一化为 ExperienceEntry
3. gate 计算写入价值
4. critic 计算风险
5. library 写入 raw/stm/ltm 分区
6. visual/keyframe index 可选更新
7. 如果有 sim/real 双源，执行 pair
8. 计算 sim_real_gap
9. 更新 retrieval_key 和 memory_tags
10. 保存 memory snapshot
```

## 17. 推荐目录结构

可以在当前仓库或上一级单独建通用包。更推荐先放在当前仓库验证，再抽到上一级公共库。

```text
galaxea_mujoco/
  experience_core/
    __init__.py
    schema.py
    library.py
    gating.py
    retrieval.py
    scoring.py
    critic.py
    dual_source.py
    calibration.py

  experience_adapters/
    __init__.py
    r1pro_mujoco.py
    wrapper1_ur5e.py
    real_episode.py

  source/
    run_r1pro_task_chain.py
    run_experience_memory_batch.py
    import_real_episode.py
    build_sim_real_pairs.py
```

等接口稳定后，可以把 `experience_core/` 移到上一级，供 `ur5e_mujoco` 和 `galaxea_mujoco` 共同 import。

## 18. 与当前代码的关系

当前已有：

```text
galaxea_mujoco/memory/v1.py
galaxea_mujoco/memory/gating.py
galaxea_mujoco/source/run_r1pro_task_chain.py
```

建议下一步不要继续把 `memory/v1.py` 扩成 R1Pro 专用，而是改为：

```text
experience_core/schema.py
experience_core/library.py
experience_adapters/r1pro_mujoco.py
```

`memory/v1.py` 可以暂时作为兼容层，后面逐步迁移。

`ur5e_mujoco` 的 `MemoryV3Entry` 不需要立刻改动。可以先写 `wrapper1_ur5e.py` adapter，把旧条目读进通用 `ExperienceEntry`。

## 19. MVP 实现顺序

### Phase 1：通用 schema 和 library

实现：

```text
experience_core/schema.py
experience_core/library.py
```

验收：

```text
R1Pro G3/G4 四条经验能写入通用库
```

### Phase 2：R1Pro adapter

实现：

```text
experience_adapters/r1pro_mujoco.py
```

验收：

```text
source/run_r1pro_task_chain.py 输出 TaskChainResult
adapter 转 ExperienceEntry
```

### Phase 3：wrapper1 UR5e adapter

实现：

```text
experience_adapters/wrapper1_ur5e.py
```

验收：

```text
能读取 ur5e_mujoco 的 MemoryV3+ JSON
能转换成通用 ExperienceEntry
```

### Phase 4：通用 gate / critic / retrieval

实现：

```text
experience_core/gating.py
experience_core/critic.py
experience_core/retrieval.py
experience_core/scoring.py
```

验收：

```text
R1Pro 和 UR5e 经验可用同一 query 接口检索
```

### Phase 5：sim-real pair/gap

实现：

```text
experience_core/dual_source.py
experience_core/calibration.py
```

验收：

```text
simulation entry + pseudo_real/real entry 可配对
生成 sim_real_pair 和 sim_real_gap
```

### Phase 6：批量生成和报告

实现：

```text
source/run_experience_memory_batch.py
source/build_sim_real_pairs.py
```

验收：

```text
批量跑 G3/G4 或导入 UR5e 结果
输出 memory snapshot 和 summary report
```

## 20. 结论

经验库可以，也应该做成通用形式。正确边界是：

```text
Experience core:
  管 schema、存储、检索、门控、critic、pair、gap、calibration

Robot adapters:
  管不同机器人和后端如何把原始 episode 转成统一 ExperienceEntry

Task runners:
  管任务怎么执行和日志怎么产生
```

这样一套库可以同时接：

```text
UR5e MuJoCo wrapper
R1Pro MuJoCo
Isaac Sim robot
real robot episode
```

同时仍然保留论文需要的核心能力：

```text
多类型机器人记忆
选择性写入
长时程 episode 层级化
sim-real 双源经验
gap-aware sandbox calibration
critic/risk-aware recovery planning
```

下一步应优先实现 `experience_core/schema.py` 和 `experience_adapters/r1pro_mujoco.py`，把当前 R1Pro 经验从专用 `memory/v1.py` 迁移到通用 `ExperienceEntry`。

## 参考资料

1. RoboMemory: A Brain-inspired Multi-memory Agentic Framework for Lifelong Learning in Physical Embodied Systems, arXiv:2508.01415.
2. RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies, arXiv:2603.04639.
3. RoboMemArena: A Comprehensive and Challenging Robotic Memory Benchmark, arXiv:2605.10921.
4. Episodic Memory Verbalization using Hierarchical Representations of Life-Long Robot Experience, arXiv:2409.17702.
5. Worth Remembering: Surprise-Gated Robot Episodic Memory, arXiv:2606.03787.
6. Robotic Sim-to-Real Transfer for Long-Horizon Pick-and-Place Tasks in the Robotic Sim2Real Competition, arXiv:2503.11012.
7. `../ur5e_mujoco/docs/双源经验库设计与使用说明.md`.
8. `../ur5e_mujoco/Sim-Real双源经验库系统说明.md`.
9. `../经验库/机器人记忆与评测论文对Sim-Real双源经验库的综合启发.md`.
10. `../经验库/双源经验库与沙盒推演研究方案.md`.
