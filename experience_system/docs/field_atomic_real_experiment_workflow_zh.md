# Field Atomic 现场实验流程

本文档给出现场实验时 `field_atomic` 原子技能层的推荐使用流程。

目标不是让大模型直接控制真机乱试，而是：

```text
现场观测
-> LLM 生成底层原子动作和参数
-> MuJoCo sandbox 先验证
-> critic / 安全检查
-> 通过后再交给真机执行器
-> 成功和失败都写回经验库
```

## 1. 前提

现场需要准备：

```text
1. 真机当前 RGB-D 图像。
2. 真机当前 LiDAR 或障碍物信息。
3. 当前机器人状态，例如底盘位姿、躯干关节、左右臂关节、夹爪状态。
4. 目标物体位置或由感知模块估计出的物体位姿。
5. 放置区或操作区位置。
6. 经验库文件路径。
```

当前系统已经有 MuJoCo 侧的 `field_atomic` 执行和写回能力，但真机 SDK 执行器还需要现场根据机器人接口补。

## 2. 推荐目录

建议现场实验生成文件放在：

```text
results/field_atomic_real_experiment/
```

示例：

```text
results/field_atomic_real_experiment/
  scene_observation.json
  runtime_scene.xml
  field_atomic_plan.json
  sandbox_report.json
  validated_robot_plan.json
  real_execution_report.json
  updated_experience_library.json
```

## 3. 第一步：采集现场观测

现场观测至少需要整理成结构化 JSON。

示例：

```json
{
  "scene_id": "real_scene_001",
  "timestamp": "2026-xx-xxTxx:xx:xx",
  "robot_state": {
    "base_pose": [0.0, 0.0, 0.0],
    "torso_qpos": [0.0, 0.0, 0.0, 0.0],
    "left_arm_qpos": [],
    "right_arm_qpos": [],
    "left_gripper_state": "open",
    "right_gripper_state": "open"
  },
  "objects": [
    {
      "name": "target_object",
      "class": "sphere",
      "position": [0.32, 0.12, 0.82],
      "size": [0.04, 0.04, 0.04]
    }
  ],
  "obstacles": [],
  "place_zones": [
    {
      "name": "place_zone_1",
      "position": [0.45, -0.12, 0.82],
      "size": [0.12, 0.12, 0.02]
    }
  ],
  "sensor_refs": {
    "rgb_path": "camera/rgb_0001.png",
    "depth_path": "camera/depth_0001.npy",
    "lidar_path": "lidar/scan_0001.json"
  }
}
```

## 4. 第二步：生成 runtime sandbox scene

现场物体位置不是固定的，所以不能只用固定 G3/G4 XML。

应把 `scene_observation.json` 转成：

```text
runtime_scene.xml
```

已有工具：

```text
experience_system/tools/build_runtime_sandbox_scene.py
```

示例：

```bash
python ../experience_system/tools/build_runtime_sandbox_scene.py \
  --scene results/field_atomic_real_experiment/scene_observation.json \
  --output results/field_atomic_real_experiment/runtime_scene.xml
```

注意：现场阶段需要根据真实感知输出适配 `scene_observation.json` 格式。

## 5. 第三步：让 LLM 生成 field atomic plan

使用：

```text
source/run_field_atomic_llm_plan.py
```

dry-run 示例：

```bash
MUJOCO_GL=osmesa conda run -n mujoco1 python -B source/run_field_atomic_llm_plan.py \
  --goal "move to a safer reach posture and inspect the target object" \
  --dry-run-llm \
  --model-path results/field_atomic_real_experiment/runtime_scene.xml \
  --universal-experience-lib results/field_atomic_real_experiment/updated_experience_library.json \
  --writeback-library-output results/field_atomic_real_experiment/updated_experience_library.json \
  --scenario-id real_scene_001 \
  --condition-id field_atomic_recovery \
  --save-plan results/field_atomic_real_experiment/field_atomic_plan.json \
  --save-report results/field_atomic_real_experiment/sandbox_report.json
```

真实 LLM 示例：

```bash
MUJOCO_GL=osmesa conda run -n mujoco1 python -B source/run_field_atomic_llm_plan.py \
  --goal "move to a safer reach posture and inspect the target object" \
  --model-path results/field_atomic_real_experiment/runtime_scene.xml \
  --universal-experience-lib results/field_atomic_real_experiment/updated_experience_library.json \
  --writeback-library-output results/field_atomic_real_experiment/updated_experience_library.json \
  --scenario-id real_scene_001 \
  --condition-id field_atomic_recovery \
  --save-plan results/field_atomic_real_experiment/field_atomic_plan.json \
  --save-report results/field_atomic_real_experiment/sandbox_report.json
```

真实 LLM 需要先配置：

```text
experience_system/.env
```

必须有：

```text
EXPERIENCE_LLM_API_KEY
EXPERIENCE_LLM_BASE_URL
EXPERIENCE_LLM_MODEL
```

## 6. 第四步：检查 sandbox 结果

需要检查：

```text
sandbox_report.json
```

关键字段：

```text
success_count
failure_count
step_reports
writeback.write_count
```

如果出现失败：

```text
不要直接发给真机。
失败经验仍然写回经验库。
下一轮 LLM 会看到这些 field_atomic_failure。
```

如果全部成功，也不能直接等同于真机安全，只能说明：

```text
该原子动作序列在当前 MuJoCo runtime scene 中通过了基础执行检查。
```

## 7. 第五步：生成给真机的计划

真机执行器应该读取：

```text
field_atomic_plan.json
```

其中每一步都是：

```json
{
  "action": "base_move_to_pose",
  "parameters": {
    "base_x": 0.08,
    "base_y": -0.03,
    "base_yaw": 0.0
  }
}
```

真机执行器需要做：

```text
1. 检查 action 是否属于 field_atomic 白名单。
2. 检查参数是否在机器人安全范围内。
3. 转换到真实机器人 SDK 命令。
4. 执行前支持人工确认或急停。
5. 执行后记录结果。
```

当前还没有实现真实机器人 SDK executor，所以这一步需要现场补。

## 8. 第六步：真机执行后写回经验库

真机执行后必须写回：

```text
成功经验
失败经验
异常原因
执行参数
传感器证据引用
```

建议写回为：

```text
source = real
memory_type = field_atomic_experience
memory_role = field_atomic_success 或 field_atomic_failure
```

这样后续 LLM 生成参数时，可以同时参考：

```text
MuJoCo field_atomic 经验
真机 field_atomic 经验
```

## 9. 现场推荐限制

为避免无限搜索，现场建议：

```text
一次最多生成 3 个候选计划。
每个候选最多 sandbox 一次。
最多允许 1 轮 rewrite。
全部失败则输出 no_safe_plan。
```

如果使用当前单计划工具，则建议：

```text
每次只生成一个 field_atomic_plan。
失败后人工确认是否再跑下一轮。
```

## 10. 失败如何发挥作用

失败经验不是 blocker，而是参数先验。

例如某次失败记录：

```text
action = base_move_to_pose
base_x = 0.25
base_y = 0.15
status = failed
reason = actuator_tracking_error
```

下一次 LLM 应该从经验中得到：

```text
类似状态下不要一次移动这么大。
优先生成更小的 base_x/base_y。
必要时先调整 torso，再移动手臂。
```

这就是经验库对大模型的实际作用。

## 11. 当前能做和不能做

当前能做：

```text
1. 让 LLM 生成 field_atomic_plan。
2. 在 MuJoCo 中执行 field_atomic_plan。
3. 成功和失败都写回经验库。
4. 下一轮计划生成时读取 field_atomic 历史经验。
```

当前不能直接声称：

```text
1. 已经完成真机闭环。
2. MuJoCo 结果等价于真机结果。
3. LLM 生成的参数可以不经安全检查直接发给真机。
```

## 12. 最小现场闭环

最小可执行流程：

```text
1. 拍照和读取 LiDAR。
2. 生成 scene_observation.json。
3. 生成 runtime_scene.xml。
4. LLM 生成 field_atomic_plan.json。
5. MuJoCo 执行并写回经验库。
6. 人工检查 field_atomic_plan.json 和 sandbox_report.json。
7. 若通过，再交给真机执行器。
8. 真机执行结果再次写回经验库。
```

这个闭环完成后，经验库就能从空库逐步累积：

```text
空经验库
-> sandbox 成功/失败经验
-> 真机成功/失败经验
-> 后续 LLM 参数生成参考这些经验
```
