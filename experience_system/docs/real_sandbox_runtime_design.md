# Real Sandbox Runtime Design

本文档定义真机实验阶段的 sandbox 边界。结论很明确：

```text
Fixed G3/G4 task-chain runners are simulation benchmark tools.
Real robot experiments must use dynamic scene construction and validated_robot_plan execution.
```

## 1. 为什么不能直接用固定 G3/G4 runner

当前 `galaxea_mujoco/source/run_r1pro_task_chain.py` 中的 `run_g3()` 和
`run_g4()` 是为了仿真 benchmark 写的固定任务链：

```text
detect -> pregrasp -> grasp -> lift -> occupancy -> place -> verify
```

它适合：

- 生成仿真 episode。
- 做 sandbox rollout smoke。
- 做 memory policy ablation。
- 做 critic/sweep 的可重复基准。

它不适合真机：

- 现场物体位置不是固定 XML 中的位置。
- 障碍物、桌面、目标区域来自实时 RGB-D/LiDAR/标定。
- 异常处理计划应该由 LLM/planner 和经验库共同生成，不应写死在 `run_g3/run_g4`。
- 真机控制要由机器人自己的 skill executor 解释结构化计划，而不是调用 MuJoCo benchmark 函数。

## 2. 真机阶段目标流程

真机阶段应使用以下流程：

```text
real sensors
-> RealSceneObservation
-> RuntimeSandboxScene
-> runtime MuJoCo XML / MjSpec
-> planner generates candidate plans
-> sandbox validates candidates
-> validated_robot_plan
-> real robot skill executor
-> real episode writeback
```

经验库仍然负责：

- 检索成功/失败/gap 经验。
- 构建 planner input。
- 调 LLM 生成或改写异常处理计划。
- 调 sandbox 验证候选计划。
- 输出 `validated_robot_plan`。
- 写回执行结果。

机器人/仿真模块负责：

- 传感器采集。
- 动态场景观测。
- MuJoCo runtime scene 构建。
- 具体技能执行。
- 真机控制安全中断。

## 3. 动态场景格式

当前已固定现场观测 schema：

```text
templates/field_runtime_scene_observation_template.json
docs/runtime_scene_observation_schema_zh.md
```

目标转换链路是：

```text
field_runtime_scene_observation_v1
-> runtime_sandbox_scene_v1
-> runtime_scene.xml
-> MuJoCo sandbox rollout
```

输入观测应能表达：

```text
scene_id
coordinate_frame
robot_state
table
target_object
objects[]
obstacles[]
place_zones[]
sensor_refs
metadata
```

每个 object/obstacle 至少包含：

```text
name
pose: [x, y, z]
size
geom_type
mass
friction
freejoint
```

后续工具建议：

```bash
PYTHONPATH=experience_system python -B \
  experience_system/tools/build_runtime_sandbox_scene.py \
  --observation results/field_atomic_real_experiment/scene_observation.json \
  --base-model galaxea_mujoco/r1pro_g3_sorting_scene.xml \
  --save-scene results/field_atomic_real_experiment/runtime_scene.xml \
  --save-report results/field_atomic_real_experiment/runtime_scene_report.json
```

注意：具体“观测 JSON -> runtime XML”的映射规则暂不写死，需要等现场 RGB-D、
LiDAR、物体检测和坐标系外参的实际输出格式确定后再实现。当前阶段只固定 schema、
字段含义、工具入口和报告格式，避免提前实现出与现场数据不一致的转换器。

## 4. 计划执行边界

真机执行的输入不是 G3/G4 hard-coded chain，而是：

```text
validated_robot_plan_v1
```

结构：

```text
plan_id
scenario
condition
selected_candidate_id
goal
steps[]
constraints[]
risk_notes[]
validation
```

每个 step：

```text
action
parameters
stage
reason
```

真机侧应实现一个 `SkillExecutor`：

```python
class SkillExecutor:
    def can_execute(self, action: str) -> bool: ...
    def execute(self, action: str, parameters: dict) -> dict: ...
```

经验库只保证：

- plan schema 合法。
- action 在 skill registry 中存在。
- plan 已经过 sandbox/critic 验证。

经验库不保证：

- 具体控制器如何发命令。
- 每个机器人技能如何实现。
- 真机硬件安全策略。

## 5. 固定 XML 与动态 XML 的关系

固定 XML 保留：

```text
r1pro_g3_sorting_scene.xml
r1pro_dual_arm_scene.xml
```

用途：

- benchmark。
- unit/smoke test。
- 可重复 ablation。
- fallback sandbox template。

动态 XML 新增：

```text
runtime_sandbox_scene_v1 -> build_runtime_sandbox_scene.py -> runtime_scene.xml
```

用途：

- 真机现场对象/障碍物位置不同。
- pseudo-real episode 需要不同初始布局。
- sandbox sweep 需要动态扰动场景。

## 6. 下一步代码任务

已完成：

```text
RuntimeSandboxScene schema
runtime XML generation
runtime scene observation template
validated_robot_plan dry-run executor boundary
```

还需要：

```text
1. run_task_chain 支持 --model-path runtime_scene.xml。
2. 动态 scene 中的 object 名称映射到 planner target_object。
3. skill registry / executor interface。
4. validated_robot_plan real-executor smoke，不直接调用 G3/G4 runner。
5. 从真实 RGB-D/LiDAR 观测生成 runtime_sandbox_scene_v1。
```

基础版已实现：

```text
experience_core/robot_plan_executor.py
tools/run_validated_robot_plan_dry_run.py
```

它只验证 plan dispatch 边界，不移动真机。

## 7. Paper Claim Boundary

当前可以写：

```text
The implementation separates fixed benchmark scenes from runtime sandbox scene
construction and provides a structured format for generating MuJoCo scenes from
observed object and obstacle poses.
```

不能写：

```text
The real robot already uses fully dynamic scene reconstruction.
The runtime scene builder reconstructs arbitrary meshes from RGB-D.
The fixed G3/G4 task chains represent the final real-robot execution policy.
```
