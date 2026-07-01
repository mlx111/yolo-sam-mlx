# Field Atomic 现场原子技能说明

本文档说明 `skills/field_atomic/` 的作用和使用方式。

## 1. 为什么需要这一层

现有技能里有很多任务语义名称，例如：

```text
move_to_pregrasp
approach_object
place_object
recover_from_joint_limit
```

这些名字适合仿真 benchmark 和论文流程，但现场真机实验时不一定合适。现场更接近：

```text
左臂移动到某个位置
右臂移动到某个位置
底盘移动到某个位姿
躯干移动到某个姿态
夹爪打开或闭合
读取相机
读取激光雷达
```

因此新增 `field_atomic` 层，让大模型直接生成底层动作和参数，而不是生成固定任务技能名称。

## 2. 当前动作列表

```text
left_arm_move_to_position
right_arm_move_to_position
left_gripper_set
right_gripper_set
torso_move_to_posture
base_move
head_camera_capture
```

这些动作目前放在：

```text
galaxea_mujoco/skills/field_atomic/
```

它们不会替换现有：

```text
skills/primitives/
skills/composites/
```

## 3. 设计原则

核心原则：

```text
动作名固定少量。
动作参数由 LLM 生成。
执行结果成功和失败都写回经验库。
后续 LLM 再生成类似动作时，可以参考历史成功/失败参数。
```

这比固定写死 `pregrasp_distance` 或 `grasp_offset_z` 更接近现场实验，因为现场物体位置、机器人初始姿态和障碍物状态都可能变化。

## 4. 动作参数格式

### 4.1 左臂移动到指定位置

```json
{
  "action": "left_arm_move_to_position",
  "parameters": {
    "target_x": 0.32,
    "target_y": 0.12,
    "target_z": 0.90,
    "steps": 1200,
    "settle_steps": 400,
    "max_joint_step": 0.004,
    "fail_threshold": 0.03,
    "direct_qpos": false
  }
}
```

右臂同理：

```text
right_arm_move_to_position
```

### 4.2 夹爪开合

闭合：

```json
{
  "action": "left_gripper_set",
  "parameters": {
    "state": 1,
    "direct_qpos": false
  }
}
```

打开：

```json
{
  "action": "left_gripper_set",
  "parameters": {
    "state": 0,
    "direct_qpos": false
  }
}
```

右夹爪使用：

```text
right_gripper_set
```

### 4.3 躯干移动到指定姿态

```json
{
  "action": "torso_move_to_posture",
  "parameters": {
    "target_qpos": [0.0, 0.05, 0.0, 0.12],
    "steps": 500,
    "settle_steps": 120,
    "max_joint_step": 0.004,
    "direct_qpos": false
  }
}
```

### 4.4 底盘移动到指定位姿

```json
{
  "action": "base_move",
  "parameters": {
    "x": 0.08,
    "y": -0.03,
    "yaw": 0.0,
    "frame": "world",
    "steps": 400,
    "settle_steps": 80,
    "max_joint_step": 0.004,
    "direct_qpos": false
  }
}
```

说明：

```text
frame = world  -> 绝对运动，x/y/yaw 直接表示世界坐标目标位姿
frame = base   -> 相对运动，x/y 表示机器人当前 base 坐标系下的位移，yaw 表示相对当前朝向的转角
```

### 4.5 获取头部相机图像

```json
{
  "action": "head_camera_capture",
  "parameters": {
    "width": 320,
    "height": 240,
    "include_depth": true
  }
}
```

注意：这里默认是现场外接 RGB-D 相机，不依赖机器人自带头部相机是否有深度。

## 5. LLM 生成方式

已新增脚本：

```text
galaxea_mujoco/source/run_field_atomic_llm_plan.py
```

示例 dry-run：

```bash
MUJOCO_GL=osmesa conda run -n mujoco1 python -B source/run_field_atomic_llm_plan.py \
  --goal "align robot and inspect scene" \
  --dry-run-llm \
  --model-path r1pro_g3_sorting_scene.xml \
  --writeback-library-output /tmp/field_atomic_llm_library.json \
  --save-plan /tmp/field_atomic_llm_plan.json \
  --save-report /tmp/field_atomic_llm_report.json
```

使用真实 LLM 时去掉：

```text
--dry-run-llm
```

并确保：

```text
experience_system/.env
```

中已经配置：

```text
EXPERIENCE_LLM_API_KEY
EXPERIENCE_LLM_BASE_URL
EXPERIENCE_LLM_MODEL
```

## 6. 输出计划格式

LLM 输出给执行器的是最简执行 JSON：

```text
field_atomic_execution_steps_v1
```

结构示例：

```json
{
  "steps": [
    {
      "action": "head_camera_rgbd_save",
      "parameters": {}
    },
    {
      "action": "head_camera_grounded_sam2_pose",
      "parameters": {
        "target_class": "apple"
      }
    }
  ]
}
```

`goal`、`reason`、`constraints`、`risk_notes`、`evidence_ids`、`confidence` 等信息只进入 report 或经验库 metadata，不写入执行 JSON。

## 7. 经验库写回

每个原子动作执行后都会生成经验条目。

成功：

```text
memory_type = field_atomic_experience
memory_role = field_atomic_success
```

失败：

```text
memory_type = field_atomic_experience
memory_role = field_atomic_failure
```

写回内容包括：

```text
execution_feedback.field_atomic_action
execution_feedback.field_atomic_parameters
execution_feedback.field_atomic_result
```

这样后续 LLM 再生成原子动作时，可以从经验库里看到：

```text
哪些参数成功
哪些参数失败
失败时对应哪个动作
```

## 8. 与现有经验库的关系

`field_atomic` 不是另一个经验库，而是共同经验库中的一种经验类型。

区别是：

```text
普通 sandbox recovery 经验：记录任务级恢复计划和 critic 结果。
field_atomic 经验：记录底层动作参数和单步执行成败。
```

两者可以同时存在。后续现场实验中，大模型可以先用任务级经验理解异常，再用 field atomic 经验选择更合适的底层动作参数。

## 9. 当前边界

当前 `field_atomic` 已支持 MuJoCo 仿真执行和经验写回。

手臂原子动作 `left_arm_move_to_position` / `right_arm_move_to_position`
支持位置和可选末端朝向：

```json
{
  "target_x": 0.32,
  "target_y": 0.12,
  "target_z": 0.90,
  "control_frame": "grasp_tool",
  "target_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "orientation_weight": 0.35,
  "orientation_threshold": 0.15
}
```

`control_frame` 目前支持：

```text
grasp_tool
hand_tcp
```

默认推荐使用 `grasp_tool`，也就是抓取规划和抓取执行对外统一暴露的工具坐标系；只有在需要兼容底层手腕参考点时才使用 `hand_tcp`。

`target_quat_wxyz` 是当前对外支持的末端朝向参数。不需要控制朝向时可以省略。

当前仿真模型中，`grasp_tool` 与 `hand_tcp` 先按相同几何位置定义，目的是先把接口和 IK frame 选择打通；如果后续拿到真机标定值，再把 `grasp_tool` 相对 `hand_tcp` 的固定偏移补进模型即可。

还没有完成：

```text
真实机器人 SDK 执行器
真实相机图像保存路径规范
真实 LiDAR 原始点云格式
真机安全急停与速度限制接入
现场坐标系标定
```

因此当前可以说：

```text
系统支持 field-style atomic action planning, MuJoCo execution, and experience writeback.
```

不能说：

```text
field_atomic 已经完成真实机器人闭环验证。
```

## 10. 推荐现场用法

现场阶段建议流程：

```text
1. 获取相机 / LiDAR / 当前机器人状态。
2. 构造 planner_input。
3. LLM 生成 field_atomic_plan_v1。
4. MuJoCo sandbox 先执行并检查风险。
5. 只把通过验证的动作序列交给真机执行器。
6. 真机执行结果写回同一个经验库。
```

如果 sandbox 或真机执行失败，也必须写回经验库。失败经验不是垃圾数据，而是后续 LLM 避免错误参数的重要反馈。
