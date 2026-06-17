> 状态说明：本文是早期实现方案，部分目标已经在当前 `experience_system`
> 中以不同结构实现，部分真机相关内容仍是未来工作。
> 当前实现证据以 `experience_system/docs/` 和
> `galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/` 下报告为准。
> 不要把本文中的“真机下发/真机执行成功”表述当作已验证结果。

# Sim-Real 双源经验库与沙盒推演系统实现方案

## 1. 实现目标

本实现方案面向长时程机器人操作任务中的异常恢复，目标是构建一个可运行、可扩展、可实验验证的 Sim-Real 双源经验库系统。系统需要同时保存仿真运行异常处理经验和真机运行异常处理经验，并在真机遇到异常时，先检索经验、修正仿真沙盒、推演候选恢复动作，再选择安全可靠的动作下发到真机执行。

系统的核心能力包括：

1. 从仿真和真机运行过程中采集异常经验。
2. 对经验进行筛选，只保存高价值异常、失败和高 Sim-Real 差异片段。
3. 对异常经验进行结构化存储和向量化索引。
4. 在真机异常发生时进行混合检索，召回相似的仿真经验和真机经验。
5. 根据真机经验估计 Sim-Real 差异，修正仿真沙盒。
6. 在沙盒中推演候选恢复动作，并通过 critic 评分筛选动作。
7. 将真机执行结果回写经验库，形成闭环更新。

本系统不应一开始追求通用具身智能记忆，而应优先实现一个最小可验证闭环：

```text
异常检测
→ 经验写入/检索
→ Sim-Real 差异估计
→ 沙盒推演
→ critic 评分
→ 真机执行
→ 结果回写
```

## 2. 论文与网络资料对实现的启发

### 2.1 Worth Remembering: 经验写入门控

`Worth Remembering1.pdf` 提出机器人长期记忆不能无差别保存所有观测，而应通过 Bayesian Surprise 等机制筛选高价值事件。网络检索确认该论文使用 V-JEPA-2 latent space 中的 surprise-gated episodic memory 来选择值得记忆的经验。

对本系统的启发是：经验库不能保存所有仿真和真机轨迹，否则会带来存储压力和检索噪声。应先实现工程化的写入门控：

```text
write_score =
  w_anomaly * anomaly_score
  + w_failure * failure_score
  + w_gap * sim_real_gap_score
  + w_utility * recovery_utility_score
```

当 `write_score` 超过阈值时，将该片段写入长期经验库。

### 2.2 Learning From Failure: 失败经验显式保存

`Learning From Failure.pdf` 的 FEMA 思想强调失败轨迹不是噪声，而是避免机器人重复进入危险状态的重要经验。该论文报告 FEMA 在 MuJoCo 任务中提升了样本效率，并能用于真实双足机器人任务。

对本系统的启发是：经验库需要单独维护失败经验索引。真机异常发生时，系统不仅要检索成功恢复经验，也要检索相似失败经验，用于降低候选动作评分或触发安全中断。

### 2.3 RialTo: Real-to-Sim-to-Real 沙盒对齐

`Reconciling Reality through Simulation3.pdf` 和 RialTo 官网都表明，真实场景可以通过少量真实数据构建数字孪生仿真环境，再在仿真中进行 RL fine-tuning 或策略验证，最后迁回真机。

对本系统的启发是：沙盒不能是固定仿真环境，而应根据当前真机异常状态进行 Real-to-Sim 初始化。第一阶段可以不做完整 3D 重建，而是把关键状态同步到 MuJoCo：

1. 机器人关节状态。
2. 末端执行器位姿。
3. 目标物体 6D 位姿。
4. 障碍物位姿。
5. 夹爪开合状态。
6. 估计的物体质量、摩擦和接触参数。

### 2.4 Plan in Sandbox: 先推演再执行

`Plan in Sandbox1.pdf` 的 SAGE 框架强调在物理约束的抽象沙盒中学习和推演经验，再迁移到真实环境。虽然该论文主要面向导航任务，但其思想适合迁移到异常恢复：机器人在真机执行前先在内部物理环境中推演动作。

对本系统的启发是：沙盒推演不一定一开始追求照片级真实，而应优先保证关键物理约束正确，例如碰撞、抓取稳定性、运动学可达性和任务成功条件。

### 2.5 RoboMemory / RoboMME / RoboMemArena: 经验类型划分

这些机器人记忆论文说明，记忆不是单一文本日志，而应包含时间、空间、对象和程序性经验。RoboMME 进一步将机器人操作记忆拆成 temporal、spatial、object、procedural 等类型。

对本系统的启发是：经验库需要至少保存四类检索键：

1. 时间键：异常发生阶段、动作序列、前后状态。
2. 空间键：物体位姿、机器人末端位姿、接触关系。
3. 对象键：物体类别、尺寸、材质、摩擦、易滑程度。
4. 程序键：恢复动作模板和轨迹片段。

### 2.6 RAP / EEAgent: 检索增强规划与反思

`RAPv1.pdf` 和 `Evolvable Embodied Agent for Robotic.pdf` 支撑“根据当前上下文检索过去经验，并用经验辅助规划”的思路。它们更偏 LLM/VLM agent，但可借鉴其经验结构：保存任务、计划、动作、观察、结果和反思。

对本系统的启发是：每条经验除数值轨迹外，还应保存一段简短的自然语言总结，方便调试、人工分析和后续 LLM 辅助规划。

### 2.7 RoboCritics: 候选动作安全检查

`RoboCritics1.pdf` 说明机器人程序或轨迹需要经过 critic 检查，包括碰撞、关节速度、末端姿态和执行安全。该思想可直接放入沙盒推演阶段。

对本系统的启发是：候选恢复动作不能只看是否完成任务，也要经过安全评分：

1. 是否碰撞。
2. 是否关节越界。
3. 是否速度或加速度过大。
4. 是否夹爪姿态危险。
5. 是否增加物体滑落风险。

### 2.8 长时程 Pick-and-Place Sim-to-Real 论文

`Robotic Sim-to-Real Transfer for Long-Horizon Pick-and-Place Tasks in.pdf` 表明长时程 pick-and-place 中的 Sim-to-Real Gap 可以具体拆成感知误差和执行误差，例如运动模糊、位姿估计误差、抓取姿态不合适和非线性执行误差。

对本系统的启发是：第一阶段实验应优先选择 pick-and-place 异常恢复，而不是开放世界大任务。异常类型可以先限制为：

1. 抓取失败。
2. 抓取不稳或物体滑落。
3. 放置偏移。
4. 堆叠失败。

## 3. 系统总体架构

系统建议分为八个模块：

```text
┌──────────────────────────────┐
│ 1. 数据采集与运行监控模块       │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 2. 异常检测与经验切片模块       │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 3. 经验写入门控模块             │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 4. 双源经验库与索引模块         │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 5. 混合经验检索模块             │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 6. Sim-Real 差异估计模块        │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 7. 沙盒推演与 critic 评分模块   │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ 8. 真机执行与经验回写模块       │
└──────────────────────────────┘
```

第一阶段不建议引入太多复杂学习模块。建议先做规则化、可解释、可调试的系统，再逐步替换为学习模型。

## 4. 数据采集与运行监控模块

### 4.1 采集对象

仿真和真机都需要统一采集以下数据：

```text
timestamp
source: sim / real
episode_id
task_id
task_stage
robot_qpos
robot_qvel
ee_pose
gripper_width
object_pose
object_velocity
rgb_image_path
depth_image_path
pointcloud_path
force_torque
action_command
controller_status
success_flag
anomaly_flag
anomaly_type
```

其中第一阶段必须采集：

1. `robot_qpos`
2. `ee_pose`
3. `gripper_width`
4. `object_pose`
5. `action_command`
6. `success_flag`
7. `anomaly_type`

RGB-D、点云和力/力矩可以作为增强字段，后续逐步接入。

### 4.2 轨迹缓存

运行时维护一个短期环形缓存：

```text
TrajectoryBuffer {
  max_seconds: 10-30
  frames: [Frame]
}
```

当异常发生时，从缓存中截取异常前后窗口：

```text
experience_window =
  frames[t_anomaly - T_pre : t_anomaly + T_post]
```

建议参数：

```text
T_pre = 3-5 秒
T_post = 3-10 秒
```

## 5. 异常检测与经验切片模块

第一阶段建议使用规则检测，而不是一开始训练异常检测模型。

### 5.1 抓取失败检测

判断条件：

```text
gripper_closed == true
AND object_lift_height < height_threshold
AND distance(object, gripper) > distance_threshold
```

输出：

```text
anomaly_type = grasp_miss
```

### 5.2 物体滑落检测

判断条件：

```text
object_was_grasped == true
AND object_height_decreases_fast == true
AND gripper_closed == true
```

输出：

```text
anomaly_type = object_slip
```

### 5.3 放置偏移检测

判断条件：

```text
place_action_finished == true
AND norm(object_pose - target_pose) > pose_threshold
```

输出：

```text
anomaly_type = place_pose_shift
```

### 5.4 碰撞检测

仿真中可直接读取接触信息；真机中可通过力/力矩、关节电流或控制器报警检测。

输出：

```text
anomaly_type = collision
```

## 6. 经验写入门控模块

### 6.1 写入评分

每个异常片段计算写入评分：

```text
write_score =
  0.35 * anomaly_score
  + 0.25 * failure_score
  + 0.25 * sim_real_gap_score
  + 0.15 * recovery_utility_score
```

各项含义：

1. `anomaly_score`：异常严重程度。
2. `failure_score`：是否导致任务失败或恢复失败。
3. `sim_real_gap_score`：仿真预测与真实执行差异。
4. `recovery_utility_score`：该片段是否包含可复用恢复动作。

第一阶段可以用规则打分：

```text
anomaly_score:
  collision = 1.0
  object_slip = 0.9
  grasp_miss = 0.8
  place_pose_shift = 0.7

failure_score:
  task_failed = 1.0
  recovery_failed = 0.8
  partial_success = 0.4
  success = 0.1

sim_real_gap_score:
  normalize(pose_gap + contact_gap + outcome_gap)

recovery_utility_score:
  successful_recovery = 1.0
  failed_but_informative = 0.7
  no_recovery_attempt = 0.3
```

### 6.2 写入策略

```text
if write_score >= 0.6:
    write_to_long_term_memory()
else:
    keep_in_short_term_cache()
```

对于真机失败经验，即使 `write_score` 较低，也建议强制写入，因为真机失败数据稀缺且价值高。

## 7. 双源经验库设计

### 7.1 存储分层

建议使用三层存储：

1. **文件存储**：保存图像、点云、轨迹 numpy 文件、仿真 replay。
2. **SQLite 元数据表**：保存结构化字段、指标、路径、标签。
3. **向量索引**：保存状态向量、视觉向量、轨迹向量和文本总结向量。

第一阶段可以使用：

```text
SQLite + Chroma
```

理由：

1. SQLite 便于保存结构化经验和 JSON 字段。
2. Chroma 支持 embedding、document、metadata 和 metadata filtering。
3. 后续数据规模变大后，可替换或并行接入 FAISS。FAISS 官方定位是高效 dense vector similarity search。

### 7.2 目录结构

建议在项目中创建：

```text
experience_system/
  configs/
    db.yaml
    sandbox.yaml
    anomaly_rules.yaml
  data/
    experience.sqlite
    chroma/
    raw/
      sim/
      real/
    trajectories/
    images/
    pointclouds/
    rollouts/
  src/
    collector.py
    anomaly_detector.py
    memory_gate.py
    experience_store.py
    retriever.py
    gap_estimator.py
    sandbox_runner.py
    critic.py
    recovery_policy.py
    executor.py
    update_loop.py
  scripts/
    collect_sim_experience.py
    collect_real_experience.py
    build_index.py
    run_recovery_demo.py
    evaluate_ablation.py
  notebooks/
    inspect_experience.ipynb
    analyze_gap.ipynb
```

## 8. 数据库 Schema

### 8.1 experiences 表

```sql
CREATE TABLE experiences (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  task_type TEXT NOT NULL,
  task_stage TEXT,
  anomaly_type TEXT NOT NULL,
  outcome TEXT NOT NULL,
  created_at TEXT NOT NULL,
  episode_id TEXT,
  robot_model TEXT,
  sim_env TEXT,
  object_id TEXT,
  object_category TEXT,
  summary TEXT,
  write_score REAL,
  trajectory_path TEXT,
  image_dir TEXT,
  pointcloud_path TEXT,
  metadata_json TEXT
);
```

### 8.2 states 表

```sql
CREATE TABLE states (
  id TEXT PRIMARY KEY,
  experience_id TEXT NOT NULL,
  frame_index INTEGER NOT NULL,
  timestamp REAL NOT NULL,
  robot_qpos_json TEXT,
  robot_qvel_json TEXT,
  ee_pose_json TEXT,
  gripper_width REAL,
  object_pose_json TEXT,
  force_torque_json TEXT,
  FOREIGN KEY(experience_id) REFERENCES experiences(id)
);
```

### 8.3 actions 表

```sql
CREATE TABLE actions (
  id TEXT PRIMARY KEY,
  experience_id TEXT NOT NULL,
  frame_index INTEGER NOT NULL,
  action_type TEXT,
  action_json TEXT,
  controller_status TEXT,
  FOREIGN KEY(experience_id) REFERENCES experiences(id)
);
```

### 8.4 recovery_results 表

```sql
CREATE TABLE recovery_results (
  id TEXT PRIMARY KEY,
  experience_id TEXT NOT NULL,
  recovery_action_id TEXT,
  sandbox_success INTEGER,
  real_success INTEGER,
  pose_error REAL,
  collision_count INTEGER,
  recovery_time REAL,
  safety_score REAL,
  notes TEXT,
  FOREIGN KEY(experience_id) REFERENCES experiences(id)
);
```

### 8.5 sim_real_gaps 表

```sql
CREATE TABLE sim_real_gaps (
  id TEXT PRIMARY KEY,
  real_experience_id TEXT NOT NULL,
  sim_experience_id TEXT,
  anomaly_type TEXT NOT NULL,
  pose_gap REAL,
  contact_gap REAL,
  outcome_gap REAL,
  execution_delay_gap REAL,
  estimated_friction_gap REAL,
  estimated_mass_gap REAL,
  gap_json TEXT
);
```

## 9. 向量索引设计

建议建立四个 collection：

```text
experience_state
experience_trajectory
experience_visual
experience_text
```

### 9.1 state embedding

第一阶段可以不用深度模型，直接拼接归一化状态：

```text
state_vector = [
  ee_position,
  ee_orientation,
  gripper_width,
  object_position,
  object_orientation,
  target_position,
  relative_ee_object_pose,
  task_stage_one_hot,
  anomaly_type_one_hot
]
```

### 9.2 trajectory embedding

第一阶段可以提取统计特征：

```text
trajectory_vector = [
  delta_ee_pose,
  delta_object_pose,
  max_force,
  mean_force,
  min_gripper_width,
  object_height_change,
  trajectory_duration,
  final_pose_error
]
```

后续可替换为 GRU/Transformer 编码器。

### 9.3 visual embedding

第一阶段可以先不用视觉 embedding，只保存关键帧路径。后续接入 CLIP、DINOv2、V-JEPA-2 或项目已有视觉模型。

### 9.4 text embedding

每条经验生成一句可读总结：

```text
在 pick-and-place 的 grasp 阶段，夹爪闭合后目标物体未被抬起，恢复动作为重新对准并降低抓取高度，真机执行成功。
```

文本总结用于人工检索、调试和后续 LLM 辅助规划。

## 10. 混合检索流程

真机异常发生后，输入当前异常状态：

```text
Query {
  task_type
  task_stage
  anomaly_type
  current_state
  object_info
  recent_trajectory
}
```

检索流程：

```text
1. metadata filter:
   source in [sim, real]
   task_type == current_task
   anomaly_type == current_anomaly

2. state similarity search:
   retrieve top_k_state

3. trajectory similarity search:
   retrieve top_k_traj

4. failure memory search:
   retrieve similar failed cases

5. sim-real gap search:
   retrieve similar gap signatures

6. rerank:
   score = state_sim + traj_sim + outcome_weight - failure_risk - gap_uncertainty
```

推荐重排公式：

```text
retrieval_score =
  0.30 * state_similarity
  + 0.25 * trajectory_similarity
  + 0.20 * task_stage_match
  + 0.15 * object_similarity
  + 0.10 * source_weight
  - 0.20 * failure_risk
  - 0.15 * sim_real_uncertainty
```

其中：

```text
source_weight:
  real_success = 1.0
  sim_success = 0.7
  real_failure = 0.6
  sim_failure = 0.4
```

## 11. Sim-Real 差异估计模块

### 11.1 差异定义

```text
sim_real_gap =
  b_pose * pose_gap
  + b_contact * contact_gap
  + b_outcome * outcome_gap
  + b_delay * execution_delay_gap
```

第一阶段建议：

```text
b_pose = 0.35
b_contact = 0.30
b_outcome = 0.25
b_delay = 0.10
```

### 11.2 pose_gap

```text
pose_gap =
  norm(real_object_position - sim_object_position)
  + lambda_rot * rotation_distance(real_object_quat, sim_object_quat)
```

### 11.3 contact_gap

```text
contact_gap =
  1.0 if sim_grasp_stable and real_object_slip
  0.7 if sim_no_collision and real_collision
  0.0 if contact_outcome_consistent
```

### 11.4 outcome_gap

```text
outcome_gap =
  1.0 if sandbox_success != real_success
  0.0 otherwise
```

### 11.5 沙盒参数修正

根据检索到的 gap signature 修正仿真参数：

```text
friction_real_est = friction_sim + delta_friction
mass_real_est = mass_sim + delta_mass
pose_real_est = pose_observed + delta_pose_bias
gripper_delay_est = measured_delay
```

第一阶段可以只修正三类：

1. 物体初始位姿。
2. 摩擦系数。
3. 夹爪闭合阈值或延迟。

## 12. 沙盒推演模块

### 12.1 仿真平台

建议第一阶段优先使用 MuJoCo，原因：

1. 本地已有 MuJoCo 相关实验基础。
2. MuJoCo Python API 支持从 XML 加载模型和直接操作 `MjModel`、`MjData`。
3. 适合快速做 pick-and-place、接触、碰撞和轨迹 replay。

如果后续需要更强视觉真实感和复杂场景重建，再考虑 Isaac Sim 或其他平台。

### 12.2 沙盒初始化

```text
SandboxState {
  robot_qpos = real_robot_qpos
  robot_qvel = real_robot_qvel
  object_pose = observed_object_pose + pose_bias
  target_pose = task_target_pose
  friction = estimated_friction
  mass = estimated_mass
  gripper_state = real_gripper_state
}
```

### 12.3 候选恢复动作

第一阶段建议实现规则模板：

#### 重新对准抓取

```text
regrasp_align:
  1. open_gripper
  2. move_above_object(offset_z=0.08)
  3. align_gripper_to_object_axis
  4. descend_to_grasp_pose
  5. close_gripper
  6. lift_and_check
```

#### 抬升后重新放置

```text
lift_and_replace:
  1. lift_object(offset_z=0.05)
  2. move_to_target_above
  3. descend_slowly
  4. open_gripper
  5. check_pose_error
```

#### 后退避障

```text
retreat_and_replan:
  1. stop_current_motion
  2. move_back_along_approach_vector
  3. raise_ee
  4. call_motion_planner
```

#### 放置姿态修正

```text
place_pose_correction:
  1. compute_object_target_error
  2. grasp_or_push_object
  3. apply_small_pose_correction
  4. recheck_target_pose
```

### 12.4 推演次数

每个候选动作在沙盒中至少推演多次，以覆盖不确定性：

```text
n_rollouts_per_action = 5-20
```

扰动参数：

```text
object_pose_noise
friction_noise
mass_noise
control_delay_noise
perception_noise
```

## 13. Critic 评分模块

每条候选动作经过 critic 评分：

```text
sandbox_score =
  0.35 * predicted_success
  - 0.20 * collision_risk
  - 0.15 * pose_error
  - 0.10 * recovery_time
  - 0.10 * joint_limit_risk
  - 0.10 * sim_real_uncertainty
```

### 13.1 predicted_success

```text
predicted_success = successful_rollouts / total_rollouts
```

### 13.2 collision_risk

```text
collision_risk = collision_rollouts / total_rollouts
```

### 13.3 joint_limit_risk

```text
joint_limit_risk =
  count(qpos near limit or qvel too high) / total_steps
```

### 13.4 sim_real_uncertainty

来自历史相似经验的 gap：

```text
sim_real_uncertainty = mean(sim_real_gap of retrieved similar cases)
```

### 13.5 动作选择

```text
best_action = argmax(sandbox_score)

if best_score < safety_threshold:
    stop_and_request_manual_intervention()
else:
    execute_on_real_robot(best_action)
```

建议：

```text
safety_threshold = 0.45-0.60
```

## 14. 真机执行与结果回写

真机执行后记录：

```text
real_success
real_pose_error
real_collision
real_recovery_time
real_force_torque
real_object_slip
```

然后与沙盒预测比较：

```text
gap = compute_sim_real_gap(sandbox_result, real_result)
```

回写内容：

1. 新增一条真机经验。
2. 更新相似仿真经验的 `sim_real_gap`。
3. 更新候选恢复动作的成功率统计。
4. 如果真机失败，写入失败经验索引。
5. 如果真机成功，写入成功恢复模板库。

## 15. 最小可运行版本

### 15.1 MVP 功能范围

第一阶段只做：

```text
任务：pick-and-place
异常：grasp_miss, object_slip, place_pose_shift
平台：MuJoCo + 真机日志接口
经验库：SQLite + Chroma
恢复动作：规则模板
沙盒推演：MuJoCo rollout
评分：规则 critic
```

### 15.2 MVP 输入

```text
current_state.json
recent_trajectory.npy
object_pose.json
task_config.json
```

### 15.3 MVP 输出

```text
selected_recovery_action.json
sandbox_report.json
experience_update.json
```

### 15.4 MVP 验证标准

系统应能完成：

1. 自动记录仿真异常经验。
2. 手动或自动导入真机异常经验。
3. 真机异常状态输入后，召回相似经验。
4. 根据经验生成候选恢复动作。
5. 在 MuJoCo 中推演候选动作。
6. 输出带评分的恢复动作。
7. 将执行结果回写经验库。

## 16. 实验设计

### 16.1 对比方法

```text
Baseline A: 固定恢复策略
Baseline B: 仅仿真经验检索
Baseline C: 仅真机经验检索
Ours-1: 仿真 + 真机经验检索
Ours-2: 仿真 + 真机经验检索 + Sim-Real 差异修正 + 沙盒推演
```

### 16.2 评价指标

```text
recovery_success_rate
task_completion_rate
average_recovery_time
real_robot_trial_count
collision_count
object_drop_count
final_pose_error
sim_real_prediction_consistency
```

其中最关键的是：

1. 异常恢复成功率。
2. 真机试错次数。
3. 沙盒预测与真机结果一致性。

### 16.3 消融实验

```text
w/o real memory
w/o sim memory
w/o failure memory
w/o sim-real gap correction
w/o sandbox critic
w/o write gate
```

### 16.4 预期结果

如果方案有效，应该观察到：

1. 双源经验库优于单一仿真经验库。
2. 加入 Sim-Real 差异修正后，沙盒预测更接近真机结果。
3. 加入失败经验后，重复进入危险状态的次数下降。
4. 加入 critic 后，碰撞和关节越界风险下降。
5. 随着真机经验增加，恢复成功率逐步提升。

## 17. 实现阶段计划

### 阶段 1：离线经验库原型

目标：

1. 定义经验数据结构。
2. 建立 SQLite 数据库。
3. 建立 Chroma 向量索引。
4. 从仿真日志中导入异常经验。
5. 实现基础检索和可视化检查。

产出：

```text
experience.sqlite
build_index.py
retrieve_demo.py
```

### 阶段 2：仿真异常采集

目标：

1. 在 MuJoCo 中批量生成 pick-and-place 异常。
2. 自动标注异常类型。
3. 自动计算写入评分。
4. 保存成功和失败恢复轨迹。

产出：

```text
collect_sim_experience.py
sim_experience_dataset/
```

### 阶段 3：沙盒推演

目标：

1. 将当前异常状态同步到 MuJoCo。
2. 生成候选恢复动作。
3. 批量 rollout。
4. 输出 sandbox_score。

产出：

```text
sandbox_runner.py
critic.py
sandbox_report.json
```

### 阶段 4：真机经验接入

目标：

1. 导入真机日志。
2. 统一真机和仿真经验格式。
3. 计算 Sim-Real Gap。
4. 将真机经验纳入检索和修正。

产出：

```text
collect_real_experience.py
gap_estimator.py
```

### 阶段 5：闭环实验

目标：

1. 真机异常触发检索。
2. 沙盒推演恢复动作。
3. 真机执行。
4. 执行结果回写。
5. 完成对比实验和消融实验。

产出：

```text
run_recovery_demo.py
evaluate_ablation.py
experiment_results/
```

## 18. 推荐代码接口

### 18.1 ExperienceStore

```python
class ExperienceStore:
    def add_experience(self, exp: dict) -> str:
        ...

    def get_experience(self, exp_id: str) -> dict:
        ...

    def update_gap(self, exp_id: str, gap: dict) -> None:
        ...

    def query_by_metadata(self, filters: dict) -> list[dict]:
        ...
```

### 18.2 ExperienceRetriever

```python
class ExperienceRetriever:
    def retrieve(self, query: dict, top_k: int = 10) -> list[dict]:
        ...

    def rerank(self, query: dict, candidates: list[dict]) -> list[dict]:
        ...
```

### 18.3 SandboxRunner

```python
class SandboxRunner:
    def reset_from_real_state(self, state: dict, gap_hint: dict) -> None:
        ...

    def rollout(self, recovery_action: dict, n: int = 10) -> dict:
        ...
```

### 18.4 Critic

```python
class Critic:
    def score(self, rollout_results: list[dict], retrieved_cases: list[dict]) -> dict:
        ...
```

### 18.5 RecoveryLoop

```python
class RecoveryLoop:
    def handle_anomaly(self, current_state: dict) -> dict:
        retrieved = self.retriever.retrieve(current_state)
        gap_hint = self.gap_estimator.estimate(current_state, retrieved)
        actions = self.policy.generate_candidates(current_state, retrieved)
        reports = []
        for action in actions:
            self.sandbox.reset_from_real_state(current_state, gap_hint)
            rollout = self.sandbox.rollout(action)
            score = self.critic.score(rollout, retrieved)
            reports.append((action, score))
        best_action = select_best_safe_action(reports)
        return best_action
```

## 19. 风险与收敛建议

### 19.1 不建议一开始做的内容

1. 不建议一开始做完整开放世界记忆系统。
2. 不建议一开始训练复杂 VLA 或世界模型。
3. 不建议同时覆盖导航、操作、问答和语言反思。
4. 不建议把所有传感器数据都纳入第一版。
5. 不建议把核心贡献放在数据库工程本身。

### 19.2 建议优先做的内容

1. 先做 pick-and-place 异常恢复闭环。
2. 先做规则异常检测和规则恢复动作。
3. 先证明双源经验比单源经验有效。
4. 先证明 Sim-Real 差异修正能提升沙盒预测一致性。
5. 先用 critic 降低危险动作执行概率。

### 19.3 论文表达建议

论文中应避免把贡献写成“做了一个经验库”。更好的表达是：

```text
We propose a Sim-Real dual-source anomaly experience library
for robot recovery, where simulated and real-world failure/recovery
episodes are jointly indexed, retrieved, and used to calibrate a
pre-execution sandbox for safer real-world recovery.
```

中文可表述为：

```text
本文提出一种面向机器人异常恢复的 Sim-Real 双源异常经验库。
该经验库不仅保存仿真和真机中的异常处理经验，还显式建模二者之间的差异，
并在真机恢复动作执行前，通过校准后的仿真沙盒对候选恢复策略进行推演和安全评估。
```

## 20. 参考资料

1. Worth Remembering: Surprise-Gated Robot Episodic Memory, arXiv: https://arxiv.org/abs/2606.03787
2. Learning From Failures: Efficient Reinforcement Learning Control with Episodic Memory, arXiv: https://arxiv.org/abs/2603.07110
3. Reconciling Reality Through Simulation: A Real-to-Sim-to-Real Approach for Robust Manipulation, project page: https://real-to-sim-to-real.github.io/RialTo/
4. Reconciling Reality Through Simulation, arXiv: https://arxiv.org/abs/2403.03949
5. Plan in Sandbox, Navigate in Open Worlds, arXiv: https://arxiv.org/abs/2605.10118
6. MuJoCo Python documentation: https://mujoco.readthedocs.io/en/stable/python.html
7. MuJoCo API reference: https://mujoco.readthedocs.io/en/stable/APIreference.html
8. Chroma metadata filtering documentation: https://docs.trychroma.com/docs/querying-collections/metadata-filtering
9. Chroma collections documentation: https://docs.trychroma.com/docs/collections/manage-collections
10. FAISS documentation: https://faiss.ai/
11. SQLite JSON functions documentation: https://www.sqlite.org/json1.html
