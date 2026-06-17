# 现场原子技能层

这个目录用于现场实验阶段的参数化原子技能，不替换现有 `skills/primitives` 和 `skills/composites`。

设计目标：

```text
LLM 不再生成 move_to_pregrasp / approach_object 这类任务语义技能。
LLM 只生成少量底层动作和参数。
动作执行成功或失败后，结果可以作为经验写回经验库。
```

## 动作列表

```text
left_arm_move_to_position
right_arm_move_to_position
left_gripper_set
right_gripper_set
torso_move_to_posture
base_move_to_pose
head_camera_capture
base_lidar_scan
```

## 参数示例

左臂移动：

```json
{
  "action": "left_arm_move_to_position",
  "parameters": {
    "target_x": 0.32,
    "target_y": 0.12,
    "target_z": 0.90,
    "target_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "orientation_weight": 0.35,
    "orientation_threshold": 0.15,
    "steps": 1200,
    "settle_steps": 400,
    "max_joint_step": 0.004,
    "fail_threshold": 0.03,
    "direct_qpos": false
  }
}
```

`target_quat_wxyz` 是可选的末端朝向参数。
如果不传，手臂只控制 TCP 位置；如果传入，底层会做位置 + 姿态 IK。

底盘移动：

```json
{
  "action": "base_move_to_pose",
  "parameters": {
    "base_x": 0.08,
    "base_y": -0.03,
    "base_yaw": 0.0,
    "steps": 400,
    "settle_steps": 80,
    "max_joint_step": 0.004,
    "direct_qpos": false
  }
}
```

躯干移动：

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

夹爪：

```json
{
  "action": "left_gripper_set",
  "parameters": {
    "state": 1,
    "direct_qpos": true
  }
}
```

相机和激光雷达：

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

```json
{
  "action": "base_lidar_scan",
  "parameters": {
    "ray_count": 181,
    "horizontal_fov_deg": 360.0,
    "max_range": 5.0
  }
}
```

## 经验库反馈

每次动作执行后都可以生成 `ExperienceEntry`：

```text
memory_type = field_atomic_experience
memory_role = field_atomic_success 或 field_atomic_failure
execution_feedback.field_atomic_parameters = LLM 传入的参数
execution_feedback.field_atomic_result = 执行结果
```

这样后续 LLM 再生成类似底层动作时，可以从经验库里看到哪些参数成功、哪些参数失败。

## 边界

这层只解决“现场风格底层动作”的表达和执行入口，不负责：

```text
任务语义分解
碰撞全局规划
真机安全急停
真实驱动 SDK 调用
```

真机阶段需要把 `FieldAtomicSkillExecutor` 的 MuJoCo backend 替换成真实机器人 backend，但动作名和参数 schema 可以保持一致。
