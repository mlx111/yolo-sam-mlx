# MuJoCo Anomaly Recovery Experiment Notes

## 2026-05-24: Direct LLM Baseline Prompt Calibration

### Goal

Tune the no-memory baseline so it is not trivially zero-success and not as strong as the previous engineered prompt. The target is a usable baseline that can sometimes recover, while leaving room for experience memory to improve recovery.

### Prompt Variants Tested

#### Pure weak prompt

Fields:

- anomaly type
- condition hint
- memory context if present
- available skills
- JSON output constraints

Observed earlier behavior:

- Baseline often omitted `vertical-grasp`.
- `recovery_success` was near zero in the earlier all-weak no-fallback runs.

#### Full history prompt

Fields added:

- `已执行的技能及状态：{task_list}`
- `动作历史摘要：{history_text}`

Removed from the old strong prompt:

- explicit `vertical-grasp` recovery target
- forced gripper-open-before-grasp rule
- `grasp_miss` ordering hint
- anti-air-close constraint

Result on `grasp_miss/direct_llm_weak`, 20 trials:

```text
recovery_success: 20/20 = 100%
task_success:     20/20 = 100%
```

Conclusion: full history prompt is too strong for `grasp_miss`.

#### Current baseline prompt: task-list only

Current main-method prompt profile is `history`, but it now includes only `task_list`, not `history_text`.

Template:

```text
你是mujoco仿真中的机械臂控制助手。当前任务是重新抓取物体{target}。
检测到异常类型：{anomaly_type}。
已执行的技能及状态：{task_list}
{condition_hint}
{exp_context}
{blocker_context}

可用技能：
camera-image, detect-object(target_class), create-cloud, create-grasp,
move-pregrasp, move-grasp, vertical-grasp, gripper-action(state=0/1)

请输出一个恢复方案。

输出要求：
1. JSON 数组格式，每项含 action 和 parameters
2. detect-object 参数为 target_class
3. gripper-action 参数为 state (0=张开, 1=闭合)
4. 不要 Markdown，只输出 JSON
```

Code references:

- `llm_handler.py`, `prompt_profile == "history"`
- `experiment_method_runner.py`, main methods mapped to `prompt_profile = "history"`

### 20-Trial Baseline Check

Command:

```bash
conda run -n mujoco1 python run_experiment_batch.py \
  --anomaly grasp_miss \
  --methods direct_llm_weak \
  --trials-per-method 20 \
  --seed-start 24300 \
  --experience-read results/memory_snapshots/memory_empty.json \
  --output-dir results/direct_llm_weak_tasklist_grasp_miss_20trials_v1 \
  --noise-scale 0.03 \
  --no-viewer
```

Result directory:

```text
results/direct_llm_weak_tasklist_grasp_miss_20trials_v1
```

Results:

```text
recovery_success: 15/20 = 75%
task_success:     13/20 = 65%
```

### Interpretation

The task-list-only prompt is a better baseline than both extremes:

```text
pure weak prompt:      near-zero recovery in earlier no-fallback runs
full history prompt:   100% recovery and 100% task success on grasp_miss
task-list-only prompt: 75% recovery and 65% task success on grasp_miss
```

This baseline is usable for the next round because it is neither fully broken nor saturated.

### Next Check

Run the same 20-trial direct baseline check on `object_displaced`. That anomaly was harder under the earlier weak prompt, so it should reveal whether task-list-only prompt is still usable across harder recovery cases.

## 2026-05-24: Task-List Prompt on Other Three Methods

### Setup

Prompt profile:

```text
task-list-only history profile
```

Anomaly:

```text
grasp_miss
```

Methods:

```text
sim_only_weak
sim_memory_weak
hierarchical_memory_weak
```

Trials:

```text
3 per method
```

Memory:

```text
results/memory_snapshots/memory_5anomaly_with_collision_failed_v10.json
results/memory_indexes/text_memory_v10_collision_failed
```

Result directory:

```text
results/tasklist_prompt_other3_grasp_miss_3trials_v1
```

### Results

```text
sim_only_weak:
  recovery_success: 2/3 = 66.7%
  task_success:     1/3 = 33.3%
  retrieved:        0

sim_memory_weak:
  recovery_success: 3/3 = 100%
  task_success:     3/3 = 100%
  retrieved:        5 positive memories each trial

hierarchical_memory_weak:
  recovery_success: 3/3 = 100%
  task_success:     3/3 = 100%
  retrieved:        5 positive + 3 failed memories each trial
```

### Interpretation

The task-list-only prompt does not saturate `sim_only_weak`, while both memory methods are stable on this smoke test. This is a useful preliminary signal that the current prompt leaves room for memory context to improve recovery.

## 2026-05-24: Rolling Memory From Empty Library

### Setup

Config:

```text
configs/main_2anomaly_4method_5trials_v7_rolling_memory_empty.json
```

Result directory:

```text
results/main_2anomaly_4method_5trials_v7_rolling_memory_empty
```

Design:

```text
2 anomalies × 4 methods × 5 trials = 40 runs
initial memory: results/memory_snapshots/memory_empty.json
rolling_memory: true
rolling_memory_scope: cell
```

Methods:

```text
direct_llm_weak: no sim, no memory
sim_only_weak: sim wrapper, no memory
sim_memory_weak: sim wrapper, success-only rolling memory
hierarchical_memory_weak: sim wrapper, success + failure rolling memory
```

### Results

#### grasp_miss

```text
direct_llm_weak:
  recovery_success: 3/5
  task_success:     2/5
  retrieved_count:  [0, 0, 0, 0, 0]

sim_only_weak:
  recovery_success: 3/5
  task_success:     2/5
  retrieved_count:  [0, 0, 0, 0, 0]

sim_memory_weak:
  recovery_success: 4/5
  task_success:     0/5
  retrieved_count:  [0, 0, 1, 2, 3]
  positive_count:   [0, 0, 1, 2, 3]
  failed_count:     [0, 0, 0, 0, 0]

hierarchical_memory_weak:
  recovery_success: 5/5
  task_success:     5/5
  retrieved_count:  [0, 1, 2, 3, 4]
  positive_count:   [0, 1, 2, 3, 4]
  failed_count:     [0, 0, 0, 0, 0]
```

#### object_displaced

```text
direct_llm_weak:
  recovery_success: 1/5
  task_success:     0/5
  retrieved_count:  [0, 0, 0, 0, 0]

sim_only_weak:
  recovery_success: 3/5
  task_success:     1/5
  retrieved_count:  [0, 0, 0, 0, 0]

sim_memory_weak:
  recovery_success: 2/5
  task_success:     0/5
  retrieved_count:  [0, 0, 0, 0, 1]
  positive_count:   [0, 0, 0, 0, 1]
  failed_count:     [0, 0, 0, 0, 0]

hierarchical_memory_weak:
  recovery_success: 4/5
  task_success:     0/5
  retrieved_count:  [0, 1, 2, 3, 4]
  positive_count:   [0, 0, 1, 2, 3]
  failed_count:     [0, 1, 1, 1, 1]
```

### Interpretation

Rolling memory is working:

```text
grasp_miss/sim_memory_weak retrieved_count:        0, 0, 1, 2, 3
grasp_miss/hierarchical_memory_weak retrieved_count: 0, 1, 2, 3, 4
object_displaced/hierarchical_memory_weak retrieved_count: 0, 1, 2, 3, 4
```

The key difference between success-only and hierarchical memory appears clearly on `object_displaced`:

```text
sim_memory_weak:
  first three trials failed, so no experience was saved and retrieval stayed 0.

hierarchical_memory_weak:
  trial_000 failed but was saved as failed memory.
  trial_001 retrieved that failed memory and recovery improved from failure to success.
```

Current limitation:

```text
object_displaced/hierarchical_memory_weak improves recovery_success to 4/5,
but task_success remains 0/5.
```

This suggests failed memory helps the recovery phase, but the full task completion criterion for `object_displaced` still needs further debugging or a stronger downstream execution criterion.

## 2026-05-26: Skill Completion Plan for 25 UR5e Anomaly Conditions

### Current Exposed Recovery Skills

The current LLM recovery planner can only call the following actions:

```text
camera-image
detect-object(target_class)
create-cloud
create-grasp
move-pregrasp
move-grasp
vertical-grasp
gripper-action(state=0/1)
```

These skills are enough for coarse legacy anomalies such as `grasp_miss`,
`object_displaced`, and `slip`, where many failures can be recovered by
redetecting the object, rebuilding a grasp pose, regrasping, and lifting.
They are not sufficient to represent all 25 condition-level UR5e anomalies as
skill-based recovery. Several current recoveries are still handled by rule
patches or hard-coded placement/collision logic rather than explicit callable
skills.

### Required New Skills

#### Perception Repair Skills for U1

```text
confirm-target(target_class)
resegment-object(target_class)
estimate-object-pose(target_class)
validate-perception(target_class)
```

Purpose:

- `confirm-target` supports U1-1 similar-object confusion by verifying that the target is the apple rather than a distractor object.
- `resegment-object` supports U1-2 partial occlusion and U1-5 boundary confusion by explicitly repairing mask/segmentation state.
- `estimate-object-pose` supports U1-3 stale pose and U1-4 orientation error by updating both target position and orientation.
- `validate-perception` records whether the corrected perception state is usable before grasp synthesis.

Implementation notes:

- These skills should update `metrics["perceived_position"]`, perception recovery records, and target identity/orientation fields.
- They should not encode a fixed U1 recovery sequence in the global prompt.
- Memory methods should learn which perception repair skill to use from retrieved experiences.

#### Grasp Geometry Skills for U2 and U3

```text
adjust-grasp-pose(dx, dy, dz, yaw_deg)
adjust-pregrasp(clearance 或 height_offset)
verify-grasp()
retry-grasp()
```

Purpose:

- `adjust-grasp-pose` supports U2-1 lateral offset, U2-2 height offset, and U2-3 orientation offset.
- `adjust-pregrasp` supports U2-4 pregrasp too close and U2-5 pregrasp too far.
- `verify-grasp` makes grasp validation an explicit skill instead of relying only on final z-change or attach gate.
- `retry-grasp` can be a macro skill that expands into open, pregrasp, grasp, close, and verify.

Implementation notes:

- `create-grasp` already has partial parameter support, but the skill name is too generic for experience retrieval.
- These skills should produce `SkillResult` records so failed and successful geometry adjustments can enter memory.

#### Gripper and Retention Skills for U3 and U4

```text
check-gripper-state()
set-gripper-force(force 或 mode)
recover-slip()
```

Purpose:

- `check-gripper-state` supports U3-1 not closed and U3-2 partial close.
- `set-gripper-force` gives the planner a way to represent stronger or safer retention rather than only open/close.
- `recover-slip` supports U3-4, U3-5, and U4-1 by representing a high-level slip recovery pattern.

Implementation notes:

- MuJoCo control may approximate force/mode through existing gripper control values if real force control is unavailable.
- The skill result should include gripper command, contact state, tracked body, and object height.

#### Placement Skills for U4

```text
move-to-place-prepose()
place-object()
release-object()
correct-placement-position()
correct-placement-orientation()
verify-placement()
```

Purpose:

- `move-to-place-prepose`, `place-object`, and `release-object` expose the Step 9 placement logic as skills.
- `correct-placement-position` supports U4-3 wrong placement position.
- `correct-placement-orientation` supports U4-5 wrong placement orientation.
- `verify-placement` checks on-plate, gripper-open, home, and orientation criteria.

Implementation notes:

- Current U4-3 to U4-5 recovery is largely deterministic code in placement handling.
- To support a skill-based paper claim, placement recovery should be moved behind callable step handlers and recorded as skill results.

#### Collision, Path, and Strategy Skills for U5

```text
retreat()
move-safe-waypoint()
replan-path(strategy)
avoid-obstacle()
validate-progress()
switch-strategy(strategy)
```

Purpose:

- `retreat` creates a safe recovery state after collision or failed approach.
- `move-safe-waypoint` supports U5-1 blocked straight path.
- `replan-path` and `avoid-obstacle` support U5-1, U5-2, and U5-3.
- `validate-progress` detects repeated no-progress retry behavior.
- `switch-strategy` supports U5-5 by making strategy changes explicit rather than rule-patched.

Implementation notes:

- These skills can initially use deterministic waypoint offsets in MuJoCo.
- The important requirement is to expose them as planner-callable actions with recorded success/failure, not to implement a full motion planner immediately.

### Implementation Order

Phase 1: U1 perception repair.

```text
confirm-target
resegment-object
estimate-object-pose
validate-perception
```

Goal: make U1 recovery representable as skill execution without strengthening the global prompt.

Phase 2: U2/U3 grasp and gripper repair.

```text
adjust-grasp-pose
adjust-pregrasp
verify-grasp
check-gripper-state
set-gripper-force
retry-grasp
recover-slip
```

Goal: replace generic regrasp-only behavior with explicit geometry and retention skills.

Phase 3: U5 path and strategy recovery.

```text
retreat
move-safe-waypoint
replan-path
avoid-obstacle
validate-progress
switch-strategy
```

Goal: move blocked-path, collision, and no-progress recovery away from hidden plan patches.

Phase 4: U4 placement recovery.

```text
move-to-place-prepose
place-object
release-object
correct-placement-position
correct-placement-orientation
verify-placement
```

Goal: expose placement recovery as callable skills rather than deterministic post-recovery code.

### Prompt Policy

The global weak prompt should not be changed into a condition-specific rule list.
For example, it should not say "for U1 always call detect-object first" or
"for perception anomaly use this fixed sequence." That would make direct LLM too
strong and reduce the measurable effect of the experience library.

The preferred design is:

```text
Weak prompt:
  exposes the available skill set and current task state.

Experience memory:
  provides reusable successful or failed skill sequences.

Skill executor:
  gives each skill concrete MuJoCo behavior and records SkillResult evidence.
```

This keeps the comparison fair: direct LLM sees the same skills, while memory
methods gain advantage by retrieving which skill combinations worked for similar
conditions.
