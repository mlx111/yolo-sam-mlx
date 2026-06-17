# Universal Experience Memory Current Implementation

## 1. Current Position

当前 `galaxea_mujoco` 中的经验库已经从 R1Pro 专用日志升级为通用 Sim-Real 双源经验库框架。

它的定位是：

```text
不同机器人/后端通过 adapter 生成统一 ExperienceEntry。
experience_core 只负责 schema、存储、门控、检索、sim-real pair/gap、critic、calibration 和 scoring。
```

也就是说，经验库核心不直接依赖 R1Pro 技能、UR5e 环境、MuJoCo model、Isaac API 或真机驱动。机器人差异都在 adapter 和 runner 里处理。

当前核心链路已经实现：

```text
task runner / real episode / wrapper1 memory
-> adapter
-> ExperienceEntry
-> ExperienceLibrary
-> retrieval
-> sim-real pair/gap
-> critic
-> calibration
-> scoring
```

## 2. Directory Layout

当前实现分三层：

```text
experience_core/
  schema.py          # 通用 ExperienceEntry 和子结构
  library.py         # JSON 存储、加载、query
  gating.py          # 通用写入门控
  quality.py         # entry/episode 质量校验
  failure_taxonomy.py # 标准 failure type 映射
  retrieval.py       # 结构化检索
  dual_source.py     # sim-real / sim-pseudo_real 配对和 gap
  critic.py          # 规则 critic
  calibration.py     # gap -> sandbox calibration
  scoring.py         # 检索结果和候选计划评分

experience_adapters/
  r1pro_mujoco.py    # R1Pro task result -> ExperienceEntry
  wrapper1_ur5e.py   # experiment-sim-wrapper1 MemoryV3+ -> ExperienceEntry
  real_episode.py    # real/pseudo-real episode JSON/目录 -> ExperienceEntry

source/
  run_r1pro_task_chain.py
  import_wrapper1_ur5e_memory.py
  import_real_episode.py
  build_universal_sim_real_pairs.py
  apply_universal_critic.py
  query_universal_experience.py
  build_universal_calibration.py
  build_policy_risk_calibration.py
  consolidate_universal_experience.py
  summarize_universal_experience.py
  validate_real_episode.py
  compare_policy_baseline.py
  run_r1pro_memory_policy_smoke.py
  run_universal_memory_pipeline.py
```

旧的 R1Pro 专用库已从 `experience_system` 当前主线中移除：

```text
experience_system/memory/v1.py
experience_system/memory/gating.py
```

当前通用经验库主线统一使用 `experience_core/*` 和
`experience_adapters/*`。如果需要查看更早的 R1Pro 专用实现，应到
`galaxea_mujoco` 历史代码或旧提交中追溯，不再把它作为
`experience_system` 的运行入口。

## 3. Core Schema

核心条目是 `ExperienceEntry`，定义在 `experience_core/schema.py`。

主要字段：

```text
schema_version
experience_id
created_at / updated_at

source              # simulation / real / pseudo_real
domain
backend             # mujoco / real_robot / replay ...
validation_status
memory_partition    # simulation_memory / validated_memory / real_memory / failed_memory

robot               # RobotState
embodiment
scenario
condition
task
anomaly

skill_sequence      # list[SkillTraceItem]
action_trace
observation_trace

state_before
state_after
sensor_summary      # SensorSummary
spatial_state
object_state        # ObjectState

result
execution_feedback
key_slices
keyframes

retrieval_key
memory_tags
memory_gate

critic_result
failure_taxonomy

sim_real_pair
sim_real_gap
sandbox_calibration

real_episode_ref
raw_refs
metadata
```

关键子结构：

```text
RobotState:
  robot_id, robot_type, embodiment_tags, backend,
  kinematic_groups, end_effectors, mobile_base, torso, grippers, joints

ObjectState:
  objects, target_object, object_class, spatial_relations, support_relations, occupancy

SkillTraceItem:
  name, primitive_type, phase, inputs, outputs, success, error, duration, safety_flags, message, raw

SensorSummary:
  joint_positions, joint_velocities, end_effector_pose, gripper_state,
  contact_state, force_torque, timestamps, sensor_modalities, raw_refs

MemoryGate:
  anomaly_score, failure_score, sim_real_gap_score, recovery_utility_score,
  surprise_score, write_score, write_decision, trigger_events

CriticResult:
  overall_status, critic_risk_score, rule_flags, feedback_for_rewrite, evidence

SimRealGap:
  gap_id, gap_score, uncertainty,
  outcome_gap, pose_gap, contact_gap, perception_gap, actuation_gap,
  robot_state_gap, timing_gap, scene_reconstruction_gap

SandboxCalibration:
  calibration_id, source_gap_ids, object_pose_bias, perception_noise_bias,
  actuation_delay_bias, contact_success_bias, slip_risk_bias,
  calibration_confidence, details
```

`retrieval_key` 自动从条目里生成，包含：

```text
source
backend
robot_id
robot_type
embodiment_tags
scenario_id
condition_id
task_stage
task_name
object_class
target_object
plan_signature
memory_type
memory_role
failure_type
critic_status
pair_status
gap_type
real_episode_ref
```

## 4. Memory Partition

`memory_partition` 自动推断：

```text
real_memory:
  source == real
  or validation_status in {real_executed, real_validated}

failed_memory:
  result.success == false

validated_memory:
  validation_status in {simulation_validated, sandbox_validated}

simulation_memory:
  其他 simulation entry
```

注意：`pseudo_real` 成功条目目前通常会因为 `validation_status=pseudo_real_executed` 进入非 failed 的普通分区，失败条目进入 `failed_memory`。如果后续需要把所有 `pseudo_real` 单独分区，应扩展 `infer_memory_partition()`。

## 5. Data Sources

### 5.1 R1Pro MuJoCo

入口：

```text
source/run_r1pro_task_chain.py
```

支持：

```text
G3 clean
G3 place_occupied
G4 clean
G4 place_occupied
```

R1Pro runner 执行任务链，得到 `TaskChainResult`，再由 `R1ProMujocoAdapter` 转成 `ExperienceEntry`。

R1Pro entry 的特点：

```text
source = simulation
backend = mujoco
robot_type = mobile_dual_arm
validation_status = simulation_validated
domain = r1pro_anomaly_recovery
```

它会记录：

```text
skill_sequence
object_start / object_final
selected_place_site
object_lift
place_occupied
attach_mode
model_path
```

运行示例：

```bash
python source/run_r1pro_task_chain.py \
  --scenario G4 \
  --condition place_occupied \
  --universal-experience-lib results/memory/universal_experience_v1.json
```

### 5.2 experiment-sim-wrapper1 UR5e

入口：

```text
source/import_wrapper1_ur5e_memory.py
```

作用：

```text
读取 ../experiment-sim-wrapper1 的 MemoryV3+ JSON
转换成通用 ExperienceEntry
导入同一个 universal experience library
```

UR5e wrapper entry 的特点：

```text
robot_type = fixed_single_arm
backend = mujoco
source 从原始 entry 继承：simulation / real / pseudo_real
```

运行示例：

```bash
python source/import_wrapper1_ur5e_memory.py \
  --input ../experiment-sim-wrapper1/acknowledge/sim_memory_weak.json \
  --universal-experience-lib results/memory/universal_experience_v1.json \
  --limit 3 \
  --report results/memory/wrapper1_ur5e_import_report.json
```

### 5.3 Real / Pseudo-Real Episode

入口：

```text
source/import_real_episode.py
experience_adapters/real_episode.py
```

支持三种输入：

```text
--input episode.json
--episode-dir episode_dir/
--batch-dir episode_root/
```

episode 目录最小格式：

```text
episode_dir/
  episode.json | real_episode.json | result.json
```

可选文件：

```text
experience_after.json
sensor_summary.json | sensors.json
keyframes/*.jpg|png
frames/*.jpg|png
episode.hdf5 | episode.h5
robot_log.jsonl | robot_log.json
video/ | videos/ | rgb/ | camera/
```

导入时会自动：

```text
合并 experience_after.json
合并 sensor_summary.json / sensors.json
从 keyframes/ 或 frames/ 生成 keyframes
把 hdf5、日志、视频目录写入 raw_refs
生成 real_episode_ref
归一化常见字段别名
```

字段别名包括：

```text
skill_sequence / executed_recovery_steps / recovery_steps
scenario_id / scene_id
condition_id / anomaly_id
task_stage / stage
object_class / target_class
target_object / target_name
state_before / before_state
state_after / after_state
sensor_summary.ee_pose -> end_effector_pose
sensor_summary.contacts -> contact_state
```

运行示例：

```bash
python source/import_real_episode.py \
  --batch-dir /path/to/episode_root \
  --source pseudo_real \
  --backend real_robot \
  --universal-experience-lib results/memory/universal_experience_v1.json \
  --report results/memory/real_episode_import_report.json
```

详细格式见：

```text
docs/real_episode_import_format.md
```

## 6. Storage

通用库由 `ExperienceLibrary` 管理：

```python
from experience_core import ExperienceLibrary

lib = ExperienceLibrary.load("results/memory/universal_experience_v1.json")
lib.add(entry)
lib.save("results/memory/universal_experience_v1.json")
```

JSON 文件结构：

```json
{
  "schema_version": "universal_experience_library_v1",
  "updated_at": "...",
  "entry_count": 4,
  "entries": []
}
```

`ExperienceLibrary.add()` 按 `experience_id` upsert。相同 ID 会覆盖旧 entry，并更新 `updated_at`。

## 7. Gating

通用写入门控在：

```text
experience_core/gating.py
```

输入：

```text
metrics
recovery_success
task_success
validation_status
sim_real_gap
```

输出：

```text
MemoryGate {
  anomaly_score
  failure_score
  sim_real_gap_score
  recovery_utility_score
  surprise_score
  write_score
  write_decision
  trigger_events
}
```

`write_decision` 分四档：

```text
high_value      write_score >= 0.75
ltm_candidate   write_score >= 0.50
stm_only        write_score >= 0.20
raw_log         write_score < 0.20
```

当前权重：

```text
anomaly:          0.22
failure:          0.23
sim_real_gap:     0.22
recovery_utility: 0.15
surprise:         0.08
validation:       0.10
```

目前 gate 只记录写入价值，不强制丢弃经验。也就是说，entry 仍会写入库，后续可以根据 `write_decision` 做清理、压缩或分层存储。

## 8. Retrieval

结构化检索在：

```text
experience_core/retrieval.py
```

老接口：

```python
lib.query(
    scenario_id="G4",
    condition_id="place_occupied",
    robot_type="mobile_dual_arm",
)
```

新接口：

```python
from experience_core import RetrievalQuery

matches = lib.query_structured(RetrievalQuery(
    scenario_id="G4",
    condition_id="place_occupied",
    robot_type="mobile_dual_arm",
    critic_status="block",
    gap_type="sim_success_real_fail",
    risk_aware=True,
))
```

支持字段：

```text
scenario_id
condition_id
robot_type
backend
task_stage
source
memory_role
memory_type
memory_partition
failure_type
critic_status
gap_type
object_class
target_object
plan_signature
skill_sequence
include_failed
risk_aware
top_k
```

硬过滤字段：

```text
source
memory_role
memory_type
memory_partition
failure_type
critic_status
gap_type
object_class
target_object
```

打分字段：

```text
scenario_id
condition_id
robot_type
backend
task_stage
plan overlap
real/pseudo_real source bonus
risk adjustment
```

CLI 示例：

```bash
python source/query_universal_experience.py \
  --input results/memory/universal_experience_with_pseudo_real_paired_critic.json \
  --scenario-id G4 \
  --critic-status block \
  --gap-type sim_success_real_fail \
  --top-k 5 \
  --report results/memory/query_g4_block_gap_report.json
```

## 9. Sim-Real Pair And Gap

配对和 gap 在：

```text
experience_core/dual_source.py
source/build_universal_sim_real_pairs.py
```

配对条件：

```text
sim_entry.source != real
real_entry.source in {real, pseudo_real}
scenario_id 必须相同
```

`pair_score` 加权：

```text
scenario baseline: 0.25
condition_id match: 0.25
robot_type match: 0.15
task_stage match: 0.10
plan_signature match: 0.15
contact_pattern match: 0.05
object_class match: 0.05
```

默认 `min_pair_score = 0.55`。

gap 类型：

```text
matched_success
matched_failure
sim_success_real_fail
sim_fail_real_success
```

gap 还会计算：

```text
object_pose_error
contact_mismatch
gap_score
uncertainty
```

运行示例：

```bash
python source/build_universal_sim_real_pairs.py \
  --input results/memory/universal_experience_v1.json \
  --output results/memory/universal_experience_paired_v1.json \
  --report results/memory/universal_pair_report.json
```

配对后会写回：

```text
entry.sim_real_pair
entry.sim_real_gap
entry.memory_tags.memory_role = sim_real_gap_memory
entry.retrieval_key
```

## 10. Critic

规则 critic 在：

```text
experience_core/critic.py
source/apply_universal_critic.py
```

critic 作用：

```text
把经验条目转换成可用于评分和检索的风险信号
```

输出：

```text
critic_result.overall_status     # pass / warn / block / unknown
critic_result.critic_risk_score
critic_result.rule_flags
critic_result.feedback_for_rewrite
```

当前规则：

```text
object_not_lifted
place_xy_error_high
place_z_error_high
place_zone_miss
gripper_contact_missing
no_contact_detected
contact_lost_during_lift
dual_arm_height_mismatch
collision_risk
joint_limit_risk
sim_real_gap_high
sim_success_real_fail
gripper_failure_reason
task_failure_reason
```

`sim_success_real_fail`、`collision_risk`、`joint_limit_risk` 会进入 `block`。

运行示例：

```bash
python source/apply_universal_critic.py \
  --input results/memory/universal_experience_paired_v1.json \
  --output results/memory/universal_experience_paired_critic_v1.json \
  --report results/memory/universal_critic_report.json
```

带有 `sim_success_real_fail` 的 G4 测试结果：

```text
critic_status = block
critic_risk_score = 0.85
rule_flags = [sim_real_gap_high, sim_success_real_fail]
```

## 11. Failure Taxonomy

标准失败类型在：

```text
experience_core/failure_taxonomy.py
```

当前标准枚举：

```text
grasp_miss
grasp_slip
object_not_lifted
place_occupied
place_error
transport_collision
dual_arm_mismatch
perception_miss
actuation_limit
sim_success_real_fail
unknown_failure
```

标准化入口：

```text
normalize_failure_type(value)
infer_standard_failure_type(entry)
standardize_failure_taxonomy(entry)
```

写入位置：

```text
failure_taxonomy.failure_type           # 标准类型，用于 retrieval/filter
failure_taxonomy.standard_failure_type  # 标准类型，显式字段
failure_taxonomy.raw_failure_type       # 原始自由文本，保留排查
```

标准化触发点：

```text
R1ProMujocoAdapter.normalize_episode()
Wrapper1UR5eAdapter.normalize_entry()
RealEpisodeAdapter.normalize_episode()
apply_pair_and_gap()
apply_critic()
```

代表性映射：

```text
contact_lost_during_lift -> grasp_slip
gripper_failure_reason -> grasp_miss
object_not_lifted -> object_not_lifted
place_xy_error_high/place_z_error_high/place_zone_miss -> place_error
collision_risk -> transport_collision
joint_limit_risk -> actuation_limit
dual_arm_height_mismatch -> dual_arm_mismatch
sim_success_real_fail -> sim_success_real_fail
```

当前验证结果：

```text
wrapper1 自由文本 "恢复阶段未满足判定条件：u3_gripper_recovered_and_lifted。" -> grasp_slip
G4 sim-real gap -> sim_success_real_fail

standard_failure_type_distribution:
  grasp_slip = 3
  sim_success_real_fail = 2
```

## 12. Calibration

沙盒校准在：

```text
experience_core/calibration.py
source/build_universal_calibration.py
```

它把 paired sim-real gap 聚合成 sandbox 偏置。

聚合 key：

```text
robot_type / scenario_id / condition_id / object_class
```

输出：

```text
sandbox_calibration.object_pose_bias
sandbox_calibration.perception_noise_bias
sandbox_calibration.contact_success_bias
sandbox_calibration.slip_risk_bias
sandbox_calibration.calibration_confidence
```

规则含义：

```text
pose gap 高:
  object_pose_bias / perception_noise_bias 增大

contact mismatch:
  contact_success_bias 降低
  slip_risk_bias 增大

sim_success_real_fail:
  contact_success_bias 降低
  slip_risk_bias 增大

matched_success:
  contact_success_bias 小幅提高
```

运行示例：

```bash
python source/build_universal_calibration.py \
  --input results/memory/universal_experience_paired_critic_v1.json \
  --output results/memory/universal_experience_calibrated_v1.json \
  --report results/memory/universal_calibration_report.json
```

当前 smoke 中 G4 的校准结果：

```text
group = mobile_dual_arm / G4 / place_occupied / large_object
source_gap_ids = ['gap_5ce7747b5980']
object_pose_bias = [0.019974, 0.02, -0.014937]
perception_noise_bias = [0.03197, 0.03197, 0.03197]
contact_success_bias = -0.7
slip_risk_bias = 0.7
calibration_confidence = 0.17
```

## 13. Scoring

候选计划评分在：

```text
experience_core/scoring.py
```

主要函数：

```text
entry_risk_adjustment(entry)
estimate_real_success_prior(entries)
score_candidate_plan(candidate_steps, retrieved_experiences, query_context=None)
```

风险来源：

```text
failure_penalty
sim_real_failure_penalty
gap_score_penalty
gap_uncertainty_penalty
critic_penalty
```

候选评分现在会计算 `risk_transfer_weight`，用于控制失败/gap 风险从历史经验传播到当前候选的强度。它同时考虑：

```text
action_lcs_ratio      # 顺序动作重合
action_set_overlap    # 动作集合重合
scenario_match
condition_match
task_stage_match
object_class_match
```

其中 sim-real gap 在 `condition_id` 不一致时会明显衰减，避免 G4/place_occupied 的失败经验把 G4/clean 全部压成 rewrite；同 condition 的 gap 仍然强传播。

policy 风险传播校准在：

```text
experience_core/policy_calibration.py
source/build_policy_risk_calibration.py
```

它会从当前 universal library 中统计失败/gap 经验，生成 `policy_risk_calibration_v1.json`。当前校准内容包括：

```text
default_weights:
  gap_condition_mismatch_scale
  failure_condition_mismatch_scale
  gap_task_stage_mismatch_scale
  gap_object_class_mismatch_scale
  min_nonzero_risk_transfer

groups:
  robot_type / scenario_id / condition_id / task_stage / object_class
  entry_count / failure_rate / sim_success_real_fail_rate / evidence_confidence
  weights
  evidence_ids
```

policy smoke 可通过 `--policy-calibration` 读取该文件。未传入时使用 scoring 内置默认权重。

信任来源：

```text
real_success_bonus
paired_bonus
```

候选计划输出：

```text
candidate_score
decision        # accept / review / reject / rewrite
support_score
risk_score
failure_overlap_risk
gap_uncertainty
critic_risk
real_success_prior
evidence
```

当前 `sim_success_real_fail` 测试中，G4 候选策略被压低：

```text
decision = reject
candidate_score = 0.2392
critic_risk = 0.3857
```

这说明经验库已经能把“仿真成功但真实/伪真实失败”的经验转成后续决策风险。

## 14. Recommended Pipeline

第一版推荐离线流水线：

```bash
# 1. 生成 R1Pro simulation experience
python source/run_r1pro_task_chain.py \
  --scenario G4 \
  --condition place_occupied \
  --universal-experience-lib results/memory/universal_experience_v1.json

# 2. 导入 wrapper1 UR5e memory
python source/import_wrapper1_ur5e_memory.py \
  --input ../experiment-sim-wrapper1/acknowledge/sim_memory_weak.json \
  --universal-experience-lib results/memory/universal_experience_v1.json \
  --limit 3

# 3. 导入 real / pseudo-real episodes
python source/import_real_episode.py \
  --batch-dir /path/to/episode_root \
  --source pseudo_real \
  --backend real_robot \
  --universal-experience-lib results/memory/universal_experience_v1.json

# 4. 构建 sim-real pair/gap
python source/build_universal_sim_real_pairs.py \
  --input results/memory/universal_experience_v1.json \
  --output results/memory/universal_experience_paired_v1.json \
  --report results/memory/universal_pair_report.json

# 5. 应用 critic
python source/apply_universal_critic.py \
  --input results/memory/universal_experience_paired_v1.json \
  --output results/memory/universal_experience_paired_critic_v1.json \
  --report results/memory/universal_critic_report.json

# 6. 构建 calibration
python source/build_universal_calibration.py \
  --input results/memory/universal_experience_paired_critic_v1.json \
  --output results/memory/universal_experience_calibrated_v1.json \
  --report results/memory/universal_calibration_report.json

# 7. 构建 policy risk-transfer calibration
python source/build_policy_risk_calibration.py \
  --input results/memory/universal_experience_calibrated_v1.json \
  --output results/memory/policy_risk_calibration_v1.json

# 8. 总结经验库内容和质量
python source/summarize_universal_experience.py \
  --input results/memory/universal_experience_calibrated_v1.json \
  --report results/memory/universal_experience_summary_report.json \
  --top-k 10

# 9. 检索风险经验
python source/query_universal_experience.py \
  --input results/memory/universal_experience_calibrated_v1.json \
  --scenario-id G4 \
  --critic-status block \
  --gap-type sim_success_real_fail \
  --top-k 5

# 10. 用经验库做 R1Pro 执行前策略选择 smoke
python source/run_r1pro_memory_policy_smoke.py \
  --scenario G4 \
  --condition place_occupied \
  --universal-experience-lib results/memory/universal_experience_calibrated_v1.json \
  --save results/memory/r1pro_policy_smoke_g4_place_occupied_report.json \
  --execute-on review
```

## 15. Batch Pipeline CLI

现在已有配置驱动的批处理流水线：

```text
source/run_universal_memory_pipeline.py
```

它把以下步骤串成一条流程：

```text
R1Pro task-chain simulation
wrapper1 memory import
real/pseudo-real episode import
sim-real pair/gap
critic
sandbox calibration
policy risk-transfer calibration
consolidation
summary
policy smoke
policy baseline-vs-memory comparison
```

示例配置：

```text
configs/universal_memory_pipeline_smoke.json
```

运行：

```bash
python source/run_universal_memory_pipeline.py \
  --config configs/universal_memory_pipeline_smoke.json
```

主要输出：

```text
results/memory/universal_pipeline_smoke/universal_experience_library.json
results/memory/universal_pipeline_smoke/pipeline_report.json
results/memory/universal_pipeline_smoke/summary_report.json
results/memory/universal_pipeline_smoke/policy_risk_calibration.json
results/memory/universal_pipeline_smoke/consolidation_report.json
results/memory/universal_pipeline_smoke/policy_baseline_vs_memory_report.json
```

配置项：

```text
output_dir
base_library
r1pro_tasks
wrapper1_imports
real_episode_imports
min_pair_score
critic_thresholds
summary_top_k
policy_smoke
policy_comparison
```

`--stop-after` 可用于只跑到某个阶段：

```text
build
pair
critic
calibration
policy_calibration
summary
policy_smoke
```

质量检查：

```bash
python source/run_universal_memory_pipeline.py \
  --config configs/universal_memory_pipeline_smoke.json \
  --strict-quality
```

`--strict-quality` 会在 final library 存在必填字段缺失、空技能名、重复 experience_id 等 error 时退出非零。`--check-refs` 会额外检查 keyframe/log/video/HDF5 路径是否存在；真实数据导入时建议开启，纯仿真 smoke 可关闭。

当前 smoke 配置只包含 R1Pro 仿真成功样本，不导入 real/pseudo-real episode，因此：

```text
entry_count = 4
success_rate = 1.0
pair_count = 0
policy_calibration_group_count = 0
```

policy smoke 结果：

```text
G3/clean:
  best_candidate = g3_place_first
  decision = accept
  executed = true

G4/clean:
  best_candidate = g4_safer_transport
  decision = accept
  executed = true

G4/place_occupied:
  best_candidate = g4_safer_transport
  decision = accept
  executed = false   # smoke config 中该项 execute=false
```

policy comparison 结果：

```text
comparison_count = 3
changed_count = 3

G3/clean:
  baseline_candidate = g3_default
  memory_selected_candidate = g3_place_first
  candidate_changed = true

G4/clean:
  baseline_candidate = g4_default
  memory_selected_candidate = g4_safer_transport
  candidate_changed = true

G4/place_occupied:
  baseline_candidate = g4_default
  memory_selected_candidate = g4_safer_transport
  candidate_changed = true
```

注意：当前 smoke library 只包含成功仿真样本，没有 real/pseudo-real gap，因此 comparison 中 `risk_delta=0` 是正常结果。导入真实/伪真实失败经验后，该报告会显示 memory policy 的风险变化和 rewrite/reject 差异。

## 16. Quality Validation

质量规则在：

```text
experience_core/quality.py
source/validate_real_episode.py
```

当前检查：

```text
scenario_id
condition_id
robot_type
backend
skill_sequence
result.success / recovery_success
object_class
empty_skill_name
duplicate_experience_id
keyframe/log/video/HDF5 ref existence   # 可选
```

真实/伪真实 episode 模板：

```text
docs/real_episode_template.json
```

导入前校验：

```bash
python source/validate_real_episode.py \
  --input docs/real_episode_template.json \
  --source real \
  --backend real_robot \
  --strict
```

带引用路径检查：

```bash
python source/validate_real_episode.py \
  --episode-dir /path/to/episode_dir \
  --source real \
  --backend real_robot \
  --check-refs \
  --strict
```

当前验证结果：

```text
real_episode_template.json:
  error_count = 0
  warning_count = 0
  passed = true

pipeline strict-quality smoke:
  entry_count = 4
  error_count = 0
  warning_count = 0
  passed = true
```

## 17. Consolidation

经验合并在：

```text
experience_core/consolidation.py
source/consolidate_universal_experience.py
```

合并 key：

```text
robot_type
scenario_id
condition_id
object_class
plan_signature
result.success
failure_type
source
```

合并策略：

```text
低风险成功仿真样本 -> 可合并，代表条目 metadata.support_count 增加
real / pseudo_real -> 保留
失败样本 -> 保留
sim_real_gap_memory -> 保留
critic block -> 保留
有 failure_taxonomy -> 保留
```

运行：

```bash
python source/consolidate_universal_experience.py \
  --input results/memory/universal_pipeline_smoke/universal_experience_library.json \
  --output results/memory/universal_pipeline_smoke/universal_experience_library_consolidated.json \
  --report results/memory/universal_pipeline_smoke/consolidation_report_manual.json
```

验证结果：

```text
unique smoke library:
  input_count = 4
  output_count = 4
  removed_count = 0
  merged_group_count = 0

duplicated smoke library test:
  input_count = 8
  output_count = 4
  removed_count = 4
  merged_group_count = 4
  support_count = 2 for each merged group
```

pipeline 现在默认在 policy calibration 后、summary 前执行 consolidation。可在 config 中设置：

```json
{
  "consolidate": true
}
```

## 18. How To Use In Code

### 18.1 Load Library

```python
from experience_core import ExperienceLibrary

lib = ExperienceLibrary.load("results/memory/universal_experience_calibrated_v1.json")
```

### 18.2 Query Similar Experience

```python
from experience_core import RetrievalQuery

matches = lib.query_structured(RetrievalQuery(
    scenario_id="G4",
    condition_id="place_occupied",
    robot_type="mobile_dual_arm",
    risk_aware=True,
    top_k=5,
))
```

### 18.3 Score Candidate Plan

```python
from experience_core import matches_to_tuples, score_candidate_plan

candidate_steps = [
    "dual_arm_pregrasp",
    "dual_arm_approach",
    "dual_gripper_close",
    "dual_arm_synchronized_lift",
    "segmented_transport",
    "dual_arm_place",
]

score = score_candidate_plan(candidate_steps, matches_to_tuples(matches))
```

### 18.4 Read Calibration

```python
entry = matches[0].entry
calibration = entry.sandbox_calibration

print(calibration.object_pose_bias)
print(calibration.contact_success_bias)
print(calibration.slip_risk_bias)
```

## 19. Current Verified Outputs

当前已经验证过的结果包括：

```text
R1Pro G3/G4 clean/place_occupied simulation entries 可写入 universal library
wrapper1 UR5e MemoryV3+ 可导入同一个 universal library
real/pseudo-real episode 单文件、单目录、批量目录可导入
sim-real / sim-pseudo_real pair 可构建
sim_success_real_fail 可计算 gap_score=1.0
critic 可将 sim_success_real_fail 标为 block
failure taxonomy 可将自由文本/critic rule/gap 归一到标准 failure type
retrieval 可按 critic_status/gap_type/failure_type 精确检索
calibration 可从 G4 gap 生成 contact/slip/pose bias
scoring 会因为 gap 和 critic 风险压低候选策略分数
summary CLI 可统计 entry/source/robot/scenario/gap/critic/calibration/gate 分布
consolidation 可合并重复低风险成功仿真经验，并保留 real/failure/gap 证据
R1Pro policy smoke 可在执行前检索经验、评分多个候选策略，并按 accept/review/rewrite/reject 选择是否执行
```

代表性结果：

```text
G4 sim_success_real_fail:
  critic_status = block
  critic_risk_score = 0.85
  rule_flags = [sim_real_gap_high, sim_success_real_fail]
  standard_failure_type = sim_success_real_fail

Wrapper1 U3 grasp failures:
  raw_failure_type = 恢复阶段未满足判定条件：u3_gripper_recovered_and_lifted。
  standard_failure_type = grasp_slip

G4 calibration:
  object_pose_bias = [0.019974, 0.02, -0.014937]
  contact_success_bias = -0.7
  slip_risk_bias = 0.7
  calibration_confidence = 0.17

Candidate scoring:
  decision = reject
  candidate_score = 0.2392
```

当前 R1Pro policy smoke 结果：

```text
G4/place_occupied:
  retrieved = [G4 clean success, G4 pseudo_real gap failure, G4 sim gap success]
  candidate_count = 3
  best_candidate = g4_default
  decision = rewrite
  candidate_score = 0.341
  risk_score = 0.9175
  executed = false

G3/clean:
  retrieved = [G3 clean success, G3 place_occupied success]
  candidate_count = 3
  best_candidate = g3_place_first
  selected_candidate = g3_place_first
  decision = accept
  candidate_score = 0.875
  risk_score = 0.0
  executed = true

G4/clean:
  candidate_count = 3
  best_candidate = g4_avoid_occupied_primary
  selected_candidate = g4_avoid_occupied_primary
  decision = accept
  candidate_score = 0.7175
  risk_score = 0.1623
  executed = true
```

当前 summary smoke：

```text
entry_count = 8
success_rate = 0.5
source_distribution = {simulation: 7, pseudo_real: 1}
robot_type_distribution = {mobile_dual_arm: 5, fixed_single_arm: 3}
critic_status_distribution = {pass: 3, warn: 3, block: 2}
gap_type_distribution = {sim_success_real_fail: 2}
pair_count = 1
gap_count = 1
calibration_count = 1
duplicate_experience_ids = {}
entries_with_missing_required_fields = 0
```

## 20. What This Enables

当前经验库已经可以支持：

```text
1. 同一个 JSON 里混合 R1Pro、UR5e、real/pseudo-real 经验
2. 按机器人类型、场景、异常、失败类型、critic 状态、gap 类型检索
3. 找出仿真成功但真实失败的危险经验
4. 用 sim-real gap 修正 sandbox 参数
5. 用真实/伪真实经验影响候选恢复策略排序
6. 在 R1Pro policy smoke 中对多个候选策略做风险排序，并只执行通过策略阈值的可执行候选
7. 保留 keyframes、HDF5、视频、日志等原始数据引用

## STM/LTM Memory Lifecycle

当前通用经验库已经加入轻量级 STM/LTM 分层，不新增 schema 顶层字段，而是复用：

```text
entry.memory_tags.memory_tier
entry.metadata.memory_lifecycle
```

这样旧的 `ExperienceEntry` JSON 仍可直接加载。

核心实现：

```text
experience_core/lifecycle.py
source/consolidate_memory_lifecycle.py
```

检索时：

```text
experience_core/retrieval.py
```

会对 `memory_tier == "ltm"` 的经验增加小的稳定性分数，默认 `ltm_weight = 0.05`。如果没有关闭 `update_retrieval_stats`，进入 top-k 的经验会递增：

```text
metadata.memory_lifecycle.retrieval_count
metadata.memory_lifecycle.last_retrieved_at
```

生命周期 consolidation 的规则：

```text
STM -> LTM:
- real / pseudo_real 来源，默认晋升
- failure 或有明确 failure_taxonomy，默认晋升
- validated success，默认晋升
- memory_gate.write_score >= min_write_score，默认 0.65
- 成功经验 retrieval_count >= min_retrieval_count，默认 3

STM eviction:
- 只在 STM 数量超过 stm_capacity 时发生
- 优先淘汰低 retrieval_count、低 write_score、较旧的低价值 STM
- failure / failure_taxonomy 经验被保护，不优先淘汰
```

单独运行：

```bash
python source/consolidate_memory_lifecycle.py \
  --input results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --output results/memory/universal_pipeline_calibration_v1/universal_experience_library_lifecycle.json \
  --report results/memory/universal_pipeline_calibration_v1/memory_lifecycle_report.json
```

pipeline 配置：

```json
"memory_lifecycle": {
  "enabled": true,
  "stm_capacity": 30,
  "min_retrieval_count": 3,
  "min_write_score": 0.65,
  "promote_real": true,
  "promote_failures": true,
  "promote_validated_success": true,
  "evict_batch_size": 5
}
```

注意：当前 calibration 集很小，并且包含 pseudo-real 与 validated simulation，所以按默认策略大部分甚至全部经验会进入 LTM。这是为了防止真实/伪真实失败经验被短期容量淘汰。后续如果要模拟 rolling-memory 压力，可以关掉 `promote_validated_success` 或降低 `stm_capacity` 做压力测试。

### Retrieval Count Persistence

结构化检索默认会更新进入 top-k 的经验：

```text
metadata.memory_lifecycle.retrieval_count
metadata.memory_lifecycle.last_retrieved_at
```

单独查询时如果需要写回：

```bash
python source/query_universal_experience.py \
  --input results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --scenario-id G3 \
  --condition-id clean \
  --save-updated results/memory/universal_pipeline_calibration_v1/universal_experience_library.json
```

policy smoke / baseline comparison 也支持写回检索计数：

```bash
python source/run_r1pro_memory_policy_smoke.py ... --save-updated-library
python source/compare_policy_baseline.py ... --save-updated-library
```

pipeline 内部的 `policy_smoke` 和 `policy_comparison` 步骤会在检索后保存经验库，因此在线策略评估会留下使用痕迹。

### Failure Taxonomy Cleanup

成功经验不再因为 critic 的泛化规则名被写成普通 `unknown_failure`。现在规则是：

```text
success + no actionable failure -> failure_taxonomy = {}
failure 或明确 failure_type -> 保留标准 failure_taxonomy
sim_success_real_fail -> 即使当前 entry success，也作为 sim-real gap 风险保留
```

这样可以避免成功经验在 write policy、lifecycle promotion、risk transfer 中被误当成失败经验。

### Lifecycle Pressure Test

为了验证 STM/LTM 不是只在小库里“全部晋升”，新增压力测试脚本：

```text
source/run_memory_lifecycle_pressure_test.py
configs/memory_lifecycle_pressure_test_v1.json
```

运行：

```bash
python source/run_memory_lifecycle_pressure_test.py \
  --config configs/memory_lifecycle_pressure_test_v1.json
```

测试会从当前 calibration library 生成合成压力库：

```text
protected_failure_or_gap: 保留原失败 / sim-real gap 风险经验
frequently_retrieved_success: 人工设置 retrieval_count 较高的成功经验
low_value_duplicate_success: 大量低 write_score、低 retrieval_count 的重复成功经验
```

当前结果：

```text
before_count = 44
after_count = 18
removed_count = 26
stm_count = 10
ltm_count = 8

protected_retention_rate = 1.0
low_value_eviction_rate = 0.7222
hot_success_promotion_rate = 1.0

evicted_by_class = {low_value_duplicate_success: 26}
promoted_by_class = {
  protected_failure_or_gap: 6,
  frequently_retrieved_success: 2
}
```

结论：在 `stm_capacity = 10` 的压力下，lifecycle consolidation 会优先淘汰低价值重复成功经验，同时保留失败/Sim-Real gap 风险经验，并把高检索成功经验晋升到 LTM。
```

它还不是完整在线闭环系统。当前主要是离线经验生成、导入、分析和评分框架。

## 21. R1Pro Multi-Candidate Policy Smoke

`source/run_r1pro_memory_policy_smoke.py` 已从单候选 smoke 升级为多候选 policy smoke。

当前流程：

```text
scenario/condition
-> candidates_for_scenario()
-> 每个候选分别 retrieval + sim-real gap memory retrieval
-> score_candidate_plan()
-> 按 candidate_score 高、risk_score 低排序
-> 只从 accept/review 且 executable=true 的候选中选择执行
```

当前候选：

```text
G3:
  g3_default              executable=true
  g3_place_first          executable=true
  g3_cautious_place       executable=true

G4:
  g4_default              executable=true
  g4_safer_transport      executable=true
  g4_avoid_occupied_primary executable=true
```

`run_r1pro_task_chain.py` 现在支持 `--candidate-id`，policy smoke 选中候选后会把候选 ID 传入 runner。默认链保持不变；非默认候选只改变步骤顺序或插入已有技能，不放宽技能阈值。

报告字段：

```text
best_candidate       # 全部候选里分数最高/风险最低的候选
selected_candidate   # 通过 execute_on 且 executable=true 的执行候选
candidates           # 每个候选的 retrieval、matches、candidate_score、risk_score、evidence
executed             # 是否实际调用 run_task_chain()
execution_skip_reason
```

已验证：

```text
G4/place_occupied:
  best_candidate_id = g4_default
  selected_candidate_id = ""
  decision = rewrite
  candidate_score = 0.341
  risk_score = 0.9175
  executed = false

G3/clean:
  best_candidate_id = g3_place_first
  selected_candidate_id = g3_place_first
  decision = accept
  candidate_score = 0.875
  risk_score = 0.0
  executed = true

G4/clean with current calibrated library:
  best_candidate_id = g4_avoid_occupied_primary
  selected_candidate_id = g4_avoid_occupied_primary
  decision = accept
  candidate_score = 0.716
  risk_score = 0.1698
  executed = true
```

注意：G4/place_occupied 仍然被判为 rewrite，因为同 condition 的 `sim_success_real_fail` gap 风险权重保持 1.0。G4 clean 已不再被该 gap 压死：使用 `policy_risk_calibration_v1.json` 后，place_occupied gap 对 G4 clean 的 `risk_transfer_weight` 约为 0.1851。

## 22. Baseline Vs Memory Policy Comparison

策略对比工具：

```text
source/compare_policy_baseline.py
```

它比较：

```text
baseline: 固定默认候选
  G3 -> g3_default
  G4 -> g4_default

memory policy:
  retrieval -> candidate scoring -> selected candidate
```

输出字段：

```text
baseline_candidate
memory_selected_candidate
candidate_changed
score_delta
risk_delta
memory_decision
baseline_executed / memory_executed
baseline_success / memory_success
skill_sequence_diff
baseline_risk_evidence / memory_risk_evidence
candidate_ranking
```

快速对比，不执行任务链：

```bash
python source/compare_policy_baseline.py \
  --scenario G3 \
  --scenario G4 \
  --condition clean \
  --condition place_occupied \
  --universal-experience-lib results/memory/universal_experience_with_pseudo_real_paired_critic_calibrated.json \
  --policy-calibration results/memory/policy_risk_calibration_v1.json \
  --save results/memory/policy_baseline_vs_memory_report.json
```

带执行验证：

```bash
python source/compare_policy_baseline.py \
  --scenario G3 \
  --condition clean \
  --universal-experience-lib results/memory/universal_experience_with_pseudo_real_paired_critic_calibrated.json \
  --policy-calibration results/memory/policy_risk_calibration_v1.json \
  --execute \
  --save results/memory/policy_baseline_vs_memory_g3_clean_exec_report.json
```

当前 calibrated library 快速对比结果：

```text
comparison_count = 4
changed_count = 3

G3/clean:
  g3_default -> g3_place_first

G3/place_occupied:
  g3_default -> g3_place_first

G4/clean:
  g4_default -> g4_avoid_occupied_primary

G4/place_occupied:
  g4_default -> g4_default
```

G3/clean 执行版验证：

```text
baseline_executed = true
baseline_success = true
memory_executed = true
memory_success = true
first_difference_index = 2
```

## 23. Write-Time Gate

写入时 gate 已接入 universal core：

```text
experience_core/write_policy.py
ExperienceLibrary.add_with_policy()
```

当前策略：

```text
real / pseudo_real failure -> write
real / pseudo_real success -> write
sim_real_gap_memory -> write
critic block -> write
failure_taxonomy 非空 -> write
ordinary failure -> write
missing required fields + strict_quality -> reject
duplicate low-risk simulation success -> merge into existing support_count
low-value low-risk simulation success -> skip
ordinary accepted entry -> write
```

已接入入口：

```text
source/run_universal_memory_pipeline.py
source/import_real_episode.py
source/import_wrapper1_ur5e_memory.py
source/run_r1pro_task_chain.py
source/run_r1pro_memory_policy_smoke.py
```

pipeline 会输出：

```text
results/memory/universal_pipeline_smoke/write_policy_report.json
```

当前 smoke 结果：

```text
decision_counts = {write: 4}
reason_counts = {accepted: 4}
entry_count = 4
success_rate = 1.0
quality passed = true
```

重复低风险成功经验策略级检查：

```text
first candidate -> write
second same-key candidate -> merge
stored entry support_count = 2
```

## 24. Visual Keyframe Retrieval

视觉 keyframe 检索已接入 universal core：

```text
experience_core/visual_retrieval.py
source/build_visual_keyframe_index.py
source/query_visual_keyframes.py
```

实现方式：

```text
entry.keyframes[*].image_path
-> CLIP image embedding
-> FAISS IndexFlatIP
-> visual_index.faiss + visual_mapping.json
-> query image returns similar experience_id
-> RetrievalQuery.visual_scores 融合进结构化检索分数
```

当前 smoke：

```text
indexed_entry_count = 2
indexed_image_count = 2
faiss_size = 2
same-image query similarity = 1.0
structured visual_score is included in retrieval explanation
```

详细说明见：

```text
docs/visual_keyframe_retrieval.md
```

注意：R1Pro MuJoCo task runner 已可通过 `--keyframe-dir` 生成仿真 keyframes，并可进入视觉索引。当前 R1Pro pseudo-real/real episodes 仍缺真实图片 keyframe，真实视觉检索效果需要后续采集真实/伪真实图像路径。

## 25. Remaining Gaps

仍需解决的问题：

```text
1. policy risk calibration 已可从经验库生成，但当前证据量很小，需要更多真实/伪真实 episode 校准 condition/task_stage/action 的权重。
2. visual keyframe retrieval 已接入 universal library，R1Pro 仿真 keyframe 已可生成；pseudo-real/real episode 还缺实际 keyframe 图像数据。
3. real robot collector 还没有直接从 ROS/控制器日志自动生成 episode。
4. physical MuJoCo actuator/contact branch 还没有稳定地产生 R1Pro gap 数据。
5. STM/LTM 分层还没有迁入 universal core；当前只有低风险成功经验 consolidation。
6. calibration 现在是规则聚合，不是学习模型或系统辨识。
7. write-time gate 已接入主要入口，但还需要在更多批量数据上观察 skip/merge/reject 分布，并决定是否默认替代所有 legacy 写入路径。
8. legacy `experience_system/memory` 已清理；后续只需检查
   `galaxea_mujoco` 侧是否还有独立 task-chain 兼容需求。
```

下一步最合理的工程任务：

```text
1. 接入真实/伪真实 episode 数据规模，至少覆盖 G3/G4 clean/place_occupied/slip/grasp_miss。
2. 用更多 episode 重新生成 `policy_risk_calibration_v1.json`，校准 `risk_transfer_weight`，尤其是同 scenario 不同 condition、同动作不同物体状态的风险传播。
3. 增强 batch pipeline：加入更多真实/伪真实导入配置、失败条件矩阵和自动对比报告。
4. 再考虑 R1Pro keyframe 数据采集和 STM/LTM 分层。
```
