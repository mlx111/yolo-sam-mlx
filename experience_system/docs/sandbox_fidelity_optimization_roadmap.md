# Sandbox Fidelity Optimization Roadmap

本文档记录当前 `experience_system` 中 sandbox 从“最小可用推演层”继续升级为
“更接近数字孪生验证层”的优化计划。目标不是立即宣称 full digital twin，而是把
可实现、可评测、可写入论文证据的增强项拆成阶段。

## 1. 当前基线

当前已经实现：

- 候选计划可以通过 `tools/candidate_sandbox.py` 调用 `run_task_chain(...)` 执行 shadow rollout。
- rollout 结果会经 `R1ProMujocoAdapter` 转成 `ExperienceEntry`。
- `experience_core/critic.py` 会给出 rule-based critic，包括 lift、place、contact、motion-level risk。
- `experience_core/calibration.py` 已支持从 sim-real/pseudo-real gap 生成 `SandboxCalibration`。
- 当前 calibration 已能影响 sandbox：`object_pose_bias` 会修改目标物体初始位置，contact/slip gap 会进入风险惩罚。

当前不能声称：

- 不能说 sandbox 是 full digital twin。
- 不能说 critic 是 learned critic。
- 不能说 sensor-derived calibration 已经改善真实机器人执行。
- 不能说真实机器人实验已经证明 recovery success 提升。

当前安全表述：

```text
The implementation performs gap-calibrated MuJoCo sandbox rollout and
rule-based motion-level critic scoring before selecting anomaly-recovery
candidates.
```

## 2. 论文与资料对优化方向的要求

### 2.1 Plan in Sandbox / SAGE

本地文档 `经验库/Plan in Sandbox对Sim-Real双源经验库的启发.md` 和
`经验库/Sim-Real双源经验库沙盒推演实现方案.md` 都强调：

```text
candidate plan -> sandbox rollout -> critic -> rewrite/accept
```

这要求 sandbox 不只是一个固定仿真，而应能根据当前异常状态初始化，并在执行前推演恢复动作。

### 2.2 RialTo / Real-to-Sim-to-Real

RialTo 强调从少量真实数据构建 digital twin simulation environments，并用 real-to-sim-to-real 管线增强真实策略鲁棒性。
对当前系统的直接启发是：sandbox 应优先实现“当前状态同步”和“gap 驱动参数校准”，而不是只跑固定 clean/place_occupied 初始场景。

参考：<https://arxiv.org/abs/2403.03949>

### 2.3 RoboCritics

RoboCritics 强调 motion-level execution trace critic，并检查 joint speed、collision、unsafe end-effector pose 等风险。
当前系统已经实现了一部分 motion-level critic，但还需要更完整的轨迹记录、逐阶段风险定位和反馈给 LLM rewrite 的结构化消息。

参考：<https://arxiv.org/abs/2603.06842>

### 2.4 MuJoCo rollout / batched physics

MuJoCo 生态支持把 rollout 用于 planning、system identification 和 trajectory optimization。近期 MuJoCoUni 进一步强调 stateful batched runtime、reset-time domain randomization、sensor forward query、Jacobian query 等能力。
对当前系统的启发是：后续 sandbox 不应只单候选单次执行，而应支持多参数、多扰动、多候选的批量 rollout 评估。

参考：<https://arxiv.org/abs/2605.24922>

## 3. 目标架构

升级后的 sandbox 应形成以下闭环：

```text
real/sim episode state
-> sandbox state initializer
-> gap-calibrated model/runtime parameters
-> candidate plan rollout
-> uncertainty/domain-randomization sweep
-> trajectory + sensor + contact critic
-> validated/rejected/rewrite-required robot plan
-> execution result writeback
```

其中经验库仍然负责：

- 检索相似成功/失败/gap 经验。
- 生成临时 `SandboxCalibration`。
- 保存 sandbox evidence。
- 用 critic 和 sandbox score 影响候选排序。

MuJoCo runner 负责：

- 状态初始化。
- 参数扰动。
- 动力学 rollout。
- 轨迹、接触、传感器读数记录。

## 4. P0: 当前异常状态初始化

### 目标

让 sandbox 从“固定 scenario/condition 初始状态”升级为“从 episode 或当前观测恢复初始状态”。

### 需要新增

建议新增：

```text
experience_core/sandbox_state.py
experience_adapters/r1pro_sandbox_state.py
tools/build_sandbox_state_from_episode.py
```

核心结构：

```text
SandboxInitialState {
  scenario_id
  condition_id
  robot_qpos
  robot_qvel
  ee_pose
  gripper_width
  object_poses
  obstacle_poses
  contact_state
  timestamp
  source_episode_id
  confidence
}
```

### 接入点

修改：

```text
tools/candidate_sandbox.py
source/run_r1pro_task_chain.py
```

新增参数：

```text
--sandbox-initial-state path/to/state.json
```

`run_task_chain(...)` 内部在加载 MuJoCo model 后，将 qpos、目标物体、障碍物、夹爪状态写入 `MjData`。

### 输出指标

```text
state_initialized_from_episode: true/false
initialized_robot_qpos_count
initialized_object_pose_count
initial_state_confidence
state_init_missing_fields
state_init_pose_error_proxy
```

### 论文价值

可以把 claim 从“fixed scenario rollout”提升到：

```text
The sandbox can be initialized from structured episode state before candidate
rollout.
```

## 5. P1: 不确定性 sweep / domain randomization

### 目标

单次 rollout 容易过拟合一个初始状态。应对每个候选执行一组轻量扰动，估计稳定性。

### 扰动维度

第一版只做低成本参数：

```text
object_pose_noise_xyz
object_yaw_noise
friction_scale
mass_scale
grasp_offset_noise
actuation_delay_steps
controller_gain_scale
perception_noise_xyz
```

### 需要新增

建议新增：

```text
experience_core/sandbox_uncertainty.py
tools/run_candidate_sandbox_sweep.py
```

输出：

```text
candidate_id
num_rollouts
success_rate
critic_block_rate
critic_warn_rate
risk_score_mean
risk_score_p95
score_mean
score_p10
worst_case_failure_reason
robust_accept
```

### 融合分数

当前 `sandbox_score` 是单次值。升级为：

```text
robust_sandbox_score =
  0.50 * score_mean
  + 0.25 * score_p10
  + 0.25 * success_rate
  - 0.30 * critic_block_rate
  - 0.15 * critic_warn_rate
```

### 论文价值

可以证明系统不是只看一次成功，而是评估候选在小扰动下是否稳定。

安全表述：

```text
We estimate candidate robustness by sweeping lightweight state and parameter
perturbations in sandbox rollout.
```

## 6. P2: 接触、滑移和腕部力反馈 critic

### 目标

当前 critic 对接触和力反馈还偏结构化字段判断。真机阶段需要把 wrist force、contact duration、slip proxy 明确纳入 sandbox critic。

### 新增指标

```text
contact_after_close
contact_during_lift_ratio
contact_lost_step
object_slip_distance
object_tilt_delta
wrist_force_peak
wrist_force_impulse
wrist_torque_peak
force_discontinuity_score
grasp_stability_score
```

### 接入点

修改：

```text
source/run_r1pro_task_chain.py
experience_core/critic.py
experience_adapters/r1pro_mujoco.py
```

MuJoCo 中第一版可用 contact count、object relative motion、site/body wrench proxy 代替真实六维力传感器。
真机 episode 导入后，用真实 `wrist_force_observation` 覆盖或对齐这些 proxy。

### 输出 critic flags

```text
contact_missing_after_close
contact_lost_during_transport
slip_risk_high
wrist_force_peak_high
wrist_force_discontinuity
grasp_stability_low
```

### 论文价值

这能支撑“安全推演”而不是只看任务成功。

## 7. P3: gap 到物理参数的校准

### 目标

当前 calibration 主要是 object pose bias 和风险惩罚。下一步应把 sim-real gap 映射到 MuJoCo 参数 sweep 或运行时参数。

### 可校准参数

```text
object_pose_bias
perception_noise_bias
friction_scale
mass_scale
contact_solref_delta
contact_solimp_delta
actuation_delay_steps
controller_gain_scale
gripper_closure_bias
```

### 第一版规则

```text
if sim_success_real_fail and contact_mismatch:
  decrease contact_success_bias
  increase slip_risk_bias
  sweep lower friction_scale

if real object pose error high:
  increase perception_noise_bias
  expand object_pose_noise_xyz

if real execution lag high:
  increase actuation_delay_steps
  reduce controller_gain_scale
```

### 需要新增

```text
experience_core/sandbox_parameterization.py
tools/build_sandbox_parameter_sweep_from_gaps.py
```

输出：

```text
sandbox_parameter_profile.json
```

包含：

```text
profile_id
source_gap_ids
parameter_ranges
confidence
expected_failure_modes
```

### 论文价值

可以把当前“gap as penalty”升级成“gap informs sandbox parameterization”。

安全表述：

```text
Gap memories are converted into sandbox initialization and parameter-sweep
profiles.
```

## 8. P4: 轨迹级执行 trace 与可回放 evidence

### 目标

当前 report 有 skill trace 和 critic flags，但还不够像一个可审计的 sandbox evidence。需要保存更细的 trajectory trace。

### 新增 trace

```text
time
qpos
qvel
ee_pose_left
ee_pose_right
object_pose
object_velocity
contact_count
contact_pairs
gripper_width
control_command
critic_probe_values
```

### 文件结构

```text
results/memory/.../sandbox_traces/
  candidate_id/
    rollout_000_trace.jsonl
    rollout_000_summary.json
    rollout_000_keyframes/
```

### 报告字段

```text
trace_path
summary_path
keyframe_dir
max_joint_speed_step
first_collision_step
first_contact_loss_step
first_place_error_step
critic_timeline
```

### 论文价值

能支持 appendix 中的 per-candidate evidence，而不是只给最终分数。

## 9. P5: LLM rewrite 与 sandbox 反复验证

### 目标

当前 LLM 可以生成 recovery plan，但 sandbox 还没有完整形成：

```text
LLM plan -> sandbox reject/warn -> structured feedback -> LLM rewrite -> sandbox recheck
```

### 需要新增

```text
tools/run_recovery_plan_sandbox_loop.py
```

循环：

```text
1. render planner_input
2. LLM generate recovery_plan
3. validate skill schema
4. sandbox rollout
5. critic feedback
6. if warn/block: feed feedback to LLM
7. max_rewrite_rounds
8. export validated_robot_plan.json
```

### 输出

```text
initial_plan
rewrite_rounds
critic_feedback_history
validated_plan
final_sandbox_status
final_robot_command_plan
```

### 论文价值

这最贴近 Plan in Sandbox 的主线：不是只排序候选，而是沙盒驱动计划修正。

## 10. P6: 真机 episode 对齐和回写

### 目标

真机执行后，把真实执行结果和 sandbox prediction 对齐，形成新的 sim-real gap。

### 对齐字段

```text
sandbox_prediction_id
real_episode_id
selected_candidate_id
predicted_success
real_success
predicted_contact_timeline
real_contact_timeline
predicted_force_proxy
real_wrist_force
predicted_object_final_pose
real_object_final_pose
```

### 输出

```text
sandbox_real_alignment_report.json
```

指标：

```text
prediction_accuracy
success_prediction_error
pose_prediction_error
contact_timeline_iou
force_peak_error
new_gap_count
calibration_update_count
```

### 论文价值

有真实数据后，可以逐步把 real-format support 推进到 real-robot evidence。

## 11. P7: 批量并行 rollout

### 目标

P1 sweep 会增加 rollout 数量。后续需要提高吞吐。

### 当前状态

第一阶段已实现：`run_candidate_sandbox_sweep.py` 支持 subprocess worker 并行。

当前实现：

```text
--parallel-workers N
--determinism-check
per-rollout subprocess worker
per-rollout worker success/error metadata
serial fallback when --parallel-workers 1
```

新增指标：

```text
rollouts_per_minute
parallel_worker_count
failed_worker_count
mean_rollout_time_s
determinism_check
determinism_check_pass
```

已验证：

```text
G3 clean / g3_default / 1 rollout / serial:
  parallel_worker_count = 1
  failed_worker_count = 0

G3 clean / g3_default / 1 rollout / --parallel-workers 2:
  parallel_worker_count = 2
  failed_worker_count = 0

G3 clean / g3_default / --parallel-workers 2 / --determinism-check:
  determinism_check_pass = true
```

### 后续可选实现

第一阶段不引入新依赖，已完成。第二阶段再考虑：

```text
mujoco.rollout
MuJoCoUni-style persistent pool
MJX / GPU backend
```

当前限制：

```text
Current parallelism is process-level and robust, but each rollout pays Python
startup/import/model-load overhead. It improves throughput when multiple
candidates/perturbations are available, but a persistent MuJoCo worker pool
would be faster for large sweeps.
```

## 12. 推荐实施顺序

### P0-A: state initializer

状态：基础版已实现。没有它，sandbox 仍然是固定场景验证。

交付：

```text
sandbox_initial_state schema
build_sandbox_state_from_episode.py
run_task_chain(..., sandbox_initial_state=...)
state_init_report.json
```

当前实现：

```text
experience_core/sandbox_state.py
tools/build_sandbox_state_from_episode.py
tools/run_candidate_sandbox_rollout.py --sandbox-initial-state
source/run_r1pro_task_chain.py --sandbox-initial-state
```

已验证：

```text
G3 clean: state initialized from exp_50d6a96f91ed, target_cube pose applied.
G4 place_occupied: state initialized from exp_9c3954664691, large_object_body pose applied.
```

当前限制：

```text
Existing simulation entries contain target object pose but usually lack
robot_qpos, robot_qvel, gripper_state, contact_state, and obstacle pose fields.
The state builder records these as missing_fields and keeps confidence at 0.5.
```

### P0-B: sandbox sweep

状态：基础版已实现。它能让 sandbox 从单次成功变成 robust score。

交付：

```text
run_candidate_sandbox_sweep.py
sandbox_sweep_report.json
robust_sandbox_score
```

当前实现：

```text
experience_core/sandbox_uncertainty.py
tools/run_candidate_sandbox_sweep.py
```

已验证：

```text
G3 clean / g3_default / 2 rollouts:
  robust_sandbox_score = 1.0
  success_rate = 1.0
  critic_block_rate = 0.0

G3 clean / all candidates / 2 rollouts:
  rollout_count = 6
  selected_candidate_id = g3_place_first
```

当前扰动：

```text
object_pose_noise_xyz
perception_noise_xyz
```

当前限制：

```text
The first sweep implementation perturbs sandbox initial-state object poses only.
Friction, mass, contact solver, actuation delay, and controller gain sweeps are
left for the later gap-to-parameter-profile stage.
```

### P1: contact/force critic

状态：基础版已实现。它直接加强安全性 claim。

交付：

```text
contact_stability_metrics
wrist_force_proxy_metrics
critic flags
```

当前实现：

```text
source/run_r1pro_task_chain.py:
  contact_after_close
  contact_during_lift_ratio
  contact_lost_step
  object_slip_distance
  object_lift_slip_distance
  object_vertical_lift
  wrist_force_proxy
  grasp_stability_score

experience_adapters/r1pro_mujoco.py:
  maps contact_stability into execution_feedback and sensor_summary

experience_core/critic.py:
  contact_missing_after_close
  contact_during_lift_low
  contact_lost_during_transport
  slip_risk_high
  wrist_force_proxy_high
  grasp_stability_low
```

已验证：

```text
G3 clean:
  contact_after_close = true
  contact_during_lift_ratio = 1.0
  object_lift_slip_distance = 0.017005
  grasp_stability_score = 0.887969

G4 place_occupied:
  contact_after_close = true
  contact_during_lift_ratio = 1.0
  object_lift_slip_distance = 0.000006
  grasp_stability_score = 0.939966
```

当前限制：

```text
These metrics are MuJoCo proxies. The current simulator uses attachment-style
grasping, so wrist_force_proxy is derived from contact-count changes rather than
real six-axis wrist force. Real wrist-force logs can later populate the same
fields.
```

### P2: parameter profile from gaps

状态：基础版已实现。把 gap memory 从 risk penalty 升级到 sandbox parameterization。

交付：

```text
sandbox_parameter_profile.json
gap_to_parameter_report.json
```

当前实现：

```text
experience_core/sandbox_parameter_profile.py
tools/build_sandbox_parameter_profile_from_gaps.py
tools/run_candidate_sandbox_sweep.py --sandbox-parameter-profile
```

当前 profile 字段：

```text
object_pose_noise_xyz
object_yaw_noise_deg
friction_scale
mass_scale
actuation_delay_steps
controller_gain_scale
gripper_closure_bias
contact_success_bias
slip_risk_bias
expected_failure_modes
```

已验证：

```text
G4 place_occupied profile:
  profile_id = sandbox_profile_6c12cad88d84
  object_pose_noise_xyz = [-0.08, 0.08]
  friction_scale = [0.35, 1.05]
  actuation_delay_steps = [0, 12]
  expected_failure_modes includes sim_success_real_fail and slip_or_grasp_instability

G4 place_occupied / g4_avoid_occupied_primary / 2 rollout sweep:
  command object_pose_noise = 0.01
  effective_object_pose_noise = 0.08 from profile
  rollout_count = 2
```

当前限制：

```text
The current integration applies profile-derived object_pose_noise_xyz to sweep
sampling and applies friction_scale, mass_scale, controller_gain_scale,
contact_solref_time_scale, contact_solimp_margin_scale, and gripper_closure_bias
inside MuJoCo rollout. It also applies actuation_delay_steps to MuJoCo
physics-step control commands. Friction scaling modifies the target object's geom
friction at runtime. Mass scaling modifies the target body's mass and inertia.
Controller-gain scaling modifies position-actuator stiffness, damping, and force
ranges. Contact solver scaling modifies the target geom's contact recovery time
and solimp margin with conservative bounds. Gripper_closure_bias is passed into
the close-gripper skill to emulate weak or over-strong closure. The actuation-delay
wrapper delays `data.ctrl` seen by `mujoco.mj_step`; direct-qpos benchmark moves
are not delayed.
```

### P3: LLM rewrite loop

状态：基础版已实现。它把 LLM recovery plan、sandbox rollout、critic feedback、
rewrite 和 validated robot plan 输出串成闭环。

交付：

```text
run_recovery_plan_sandbox_loop.py
validated_robot_plan.json
rewrite_loop_report.json
```

当前实现：

```text
tools/run_recovery_plan_sandbox_loop.py
experience_core/recovery_plan.py
experience_core/skill_semantics.py
tools/candidate_sandbox.py
source/run_r1pro_task_chain.py --trace-dir
source/run_r1pro_task_chain.py run_task_plan_chain(...)
```

闭环：

```text
stage planner input
-> LLM/mock LLM structured recovery plan
-> schema normalization and allowed-skill validation
-> skill precondition/effect semantic validation before sandbox
-> executable candidate mapping or direct general plan execution
-> MuJoCo sandbox rollout
-> critic feedback with trajectory trace paths and compact trace-derived hints
-> rewrite if review/reject/block/warn
-> validated_robot_plan_v1
-> dry-run executor validation
```

已验证：

```text
G3 clean / g3_default / dry-run LLM:
  final_sandbox_status = accept
  attempt_count = 1
  validated_robot_plan = /tmp/validated_robot_plan_loop_g3.json

G4 place_occupied / g4_fast_transport / dry-run LLM:
  round 0 = review
  critic flag = joint_speed_risk on segmented_transport
  round 1 rewrite = g4_default
  final_sandbox_status = accept
  validated_robot_plan = /tmp/validated_robot_plan_loop_g4.json

G3 clean / g3_default / --use-general-plan-executor:
  final_sandbox_status = accept
  sandbox general_plan_executor = true
  executed LLM step graph directly

G4 place_occupied / g4_default + safe_transport_pose / --use-general-plan-executor:
  sandbox general_plan_executor = true
  executed step graph directly with trace output
  final_sandbox_status = review due to critic warning

G3 clean / g3_default / --use-general-plan-executor / semantic validation:
  plan_semantic_validation.schema_version = recovery_plan_semantic_validation_v2
  validator = skill_precondition_effect_graph
  plan_semantic_validation.status = pass
  final_sandbox_status = accept
  trace_feedback.trace_sample_count = 277
  trace_feedback_text includes joint-limit-margin feedback

G3 clean / g3_default / mock place_before_grasp:
  plan_semantic_validation.status = fail
  failure code = missing_precondition
  sandbox_skipped = true
  final_sandbox_status = reject
  failure_reason = LLM plan failed pre-sandbox semantic validation

G4 place_occupied / g4_fast_transport / semantic validation:
  plan_semantic_validation.status = pass
  final_sandbox_status = review
  interpretation = skill graph says executable, sandbox critic still flags risk
```

当前限制：

```text
The smoke tests use --dry-run-llm for deterministic offline validation. Removing
that flag calls the configured experience-system LLM provider from
experience_system/.env. The first general executor covers the current G3/G4
allowed skill set with built-in scenario defaults and context passing. It does
not yet infer arbitrary missing parameters for new skill APIs outside the
current R1Pro benchmark skill graph. Semantic validation is now driven by a
skill precondition/effect graph rather than fixed G3/G4 order rules. It blocks
missing required facts before MuJoCo rollout and emits warnings for missing
optional facts or duplicate steps, but it is not a full symbolic planner and
depends on each real robot skill declaring accurate requires/effects metadata.
```

### P3-C: LLM multi-candidate plan search

状态：基础版已实现。它把单计划 rewrite loop 扩展为候选计划集合搜索：
LLM/mock LLM 生成多个 recovery plans，每个计划先做语义校验，合法计划并行进入
MuJoCo sandbox，再由 critic/sandbox score 排序并导出 best validated robot plan。

交付：

```text
tools/run_llm_plan_candidate_search.py
llm_plan_candidate_search_*_report.json
llm_plan_candidate_search_*_validated_plan.json
```

闭环：

```text
stage planner input
-> generate N candidate recovery plans
-> normalize schema and allowed skills
-> skill precondition/effect semantic validation
-> parallel sandbox rollout with general plan executor
-> critic/status/score ranking
-> best validated_robot_plan_v1
-> dry-run executor validation
```

已验证：

```text
G3 clean / dry-run LLM / 3 plans / 3 workers:
  sandboxed_plan_count = 3
  failed_worker_count = 0
  final_sandbox_status = accept

G4 place_occupied / dry-run LLM / 4 plans / 4 workers:
  sandboxed_plan_count = 4
  failed_worker_count = 0
  final_sandbox_status = accept
  rollouts_per_minute = 7.8824
  report = results/memory/universal_pipeline_calibration_v1/llm_plan_candidate_search_g4_place_occupied_report.json
  validated_plan = results/memory/universal_pipeline_calibration_v1/llm_plan_candidate_search_g4_place_occupied_validated_plan.json

G4 search behavior:
  3 plans accepted by sandbox critic
  1 plan reviewed due to contact_lost_during_transport

G3 clean / real LLM / 2 plans / 2 workers:
  dry_run_llm = false
  provider = doubao
  sandboxed_plan_count = 2
  failed_worker_count = 0
  final_sandbox_status = accept
  plan 0 adds verify_grasp before lift
  plan 1 adds a second detect_place_occupancy before place
  report = results/memory/universal_pipeline_calibration_v1/llm_plan_candidate_search_g3_clean_real_llm_report.json
  validated_plan = results/memory/universal_pipeline_calibration_v1/llm_plan_candidate_search_g3_clean_real_llm_validated_plan.json
```

当前限制：

```text
Dry-run runs enumerate existing executable candidate step graphs as deterministic
stand-ins. A real Doubao run has also been verified for G3 clean with two
distinct generated plans. This is candidate search over the current executable
R1Pro skill graph, not arbitrary unseen robot skills.
```

### P3-B: LLM-assisted skill semantics induction

状态：最小闭环已实现。目标不是“盲目信任 LLM 理解未知技能”，而是让 LLM 根据技能代码、
docstring 和可选文档生成 `requires/effects/consumes/risks` 草案，再由确定性校验器检查后
人工或 sandbox evidence 确认。

交付：

```text
tools/induce_skill_semantics_llm.py
tools/validate_skill_semantics_candidate.py
experience_core/skill_semantics.py
```

流程：

```text
skill code/doc
-> LLM or dry-run candidate induction
-> skill_semantics_candidate_v1
-> deterministic schema/fact validation
-> normalized candidate / registry_fragment
-> human review or sandbox evidence before registry write
```

示例命令：

```bash
python -B ../experience_system/tools/induce_skill_semantics_llm.py \
  --skill-name capture_head_rgbd \
  --skill-file skills/base/head_camera_skill.py \
  --class-name R1ProHeadCameraSkill \
  --dry-run \
  --save /tmp/capture_head_rgbd_skill_semantics_candidate.json

python -B ../experience_system/tools/validate_skill_semantics_candidate.py \
  --candidate /tmp/capture_head_rgbd_skill_semantics_candidate.json \
  --save-report /tmp/capture_head_rgbd_skill_semantics_validation.json \
  --save-normalized /tmp/capture_head_rgbd_skill_semantics_normalized.json
```

已验证：

```text
capture_head_rgbd dry-run candidate:
  effects = rgbd_observation_available, scene_observed
  validation status = warn
  reason = introduces new fact names

read_wrist_force dry-run candidate:
  effects = wrist_force_observed
  validation status = warn
  reason = introduces new fact names
```

当前限制：

```text
LLM-generated skill semantics are candidates only. They must not be written to
the trusted registry solely because the JSON schema passes. For real robot use,
new requires/effects should be confirmed by sandbox smoke, dry-run executor
checks, real skill documentation, or operator review.
```

### P4: trajectory trace evidence

状态：基础版已实现。它把 sandbox report 从最终分数扩展为可审计 rollout evidence。

交付：

```text
trace.jsonl
summary.json
trajectory_trace report field
```

当前实现：

```text
source/run_r1pro_task_chain.py:
  TrajectoryTraceRecorder
  run_task_chain(..., trace_dir=...)
  --trace-dir

tools/candidate_sandbox.py:
  evaluate_candidate_in_sandbox(..., trace_dir=...)

tools/run_candidate_sandbox_sweep.py:
  --trace-dir
```

当前 trace 字段：

```text
time
skill
qpos
qvel
ctrl
ee_pose
object_pose
contact_count
contact_pairs
joint_limit_margin_min
```

已验证：

```text
G4 place_occupied / g4_avoid_occupied_primary / 1 rollout:
  trace_path = /tmp/sandbox_traces_smoke/g4_avoid_occupied_primary/rollout_000/trace.jsonl
  summary_path = /tmp/sandbox_traces_smoke/g4_avoid_occupied_primary/rollout_000/summary.json
  sample_count = 1159
  step_count_observed = 11330
  max_contact_count = 19
  object_path_length = 0.543333
```

当前限制：

```text
The trace is sampled through skill step callbacks. Direct-qpos benchmark moves
produce valid state/control samples, but MuJoCo time only advances when the skill
uses mj_step. Physics-path rollout with actuation delay records advancing MuJoCo
time.
```

## 13. 最小下一步

建议下一步直接实现：

```text
P7 parallel rollout or general LLM-plan executor
```

原因：

- 不依赖真机。
- P0-P5 已经提供 state init、robust sweep、contact critic、parameter profile、
  trajectory trace 和 LLM rewrite/recheck。
- 并行 rollout 可以降低 sweep 和 rewrite loop 的运行时间。
- general LLM-plan executor 可以让任意 validated step graph 直接进入 sandbox，
  不再只映射到已知 G3/G4 candidate。

第一版命令：

```bash
PYTHONPATH=experience_system python -B \
  experience_system/tools/build_sandbox_state_from_episode.py \
  --input galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --scenario G4 \
  --condition place_occupied \
  --save galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/sandbox_initial_state_g4_place_occupied.json
```

然后：

```bash
PYTHONPATH=experience_system conda run -n mujoco1 python -B \
  experience_system/tools/run_candidate_sandbox_rollout.py \
  --scenario G4 \
  --condition place_occupied \
  --sandbox-initial-state galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/sandbox_initial_state_g4_place_occupied.json \
  --universal-experience-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --save galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/sandbox_rollout_g4_place_occupied_state_init.json
```

## 14. Paper Claim Boundaries

完成 P0-A 后可以写：

```text
The sandbox rollout can be initialized from structured episode state.
```

完成 P0-B 后可以写：

```text
The system evaluates candidate robustness under lightweight sandbox
perturbation sweeps.
```

完成 P1 后可以写：

```text
The sandbox critic evaluates contact stability and force-proxy risks in
addition to task success.
```

完成 P2 后可以写：

```text
Sim-real gap memories inform sandbox parameter profiles used during rollout.
```

完成 P6 且有真机数据后才可以写：

```text
Real execution outcomes are aligned with sandbox predictions and used to update
sim-real gap memory.
```

仍然不能写，除非有对应证据：

```text
The sandbox is a full digital twin.
The system proves real-robot success-rate improvement.
The critic is learned.
The simulator is physically identical to the real robot.
```
