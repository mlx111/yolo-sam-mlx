# 真机执行文件格式与技能参数说明

本文档定义经验库/规划侧输出给真机执行器的文件格式，并梳理当前技能白名单、每个技能的作用，以及机器人侧需要解析的参数字段。

目标不是让真机直接解析完整的 `validated_robot_plan_v1`，而是输出一个更精简、更稳定、更适合机器人侧解析的执行文件。

## 1. 推荐给真机的文件

推荐输出：

```text
robot_execution_steps_v1
```

建议文件名：

```text
robot_execution_steps.json
```

推荐结构：

```json
{
  "schema_version": "robot_execution_steps_v1",
  "plan_id": "recovery_plan_xxx",
  "steps": [
    {
      "index": 0,
      "action": "detect_multiple_objects",
      "parameters": {},
      "stage": "sandbox_rewrite",
      "reason": "observe scene before selecting target"
    }
  ]
}
```

说明：

- `schema_version`：执行文件版本，便于机器人侧解析器升级。
- `plan_id`：计划唯一标识，用于日志对账、执行结果回传、经验库写回。
- `steps`：机器人侧实际需要执行的动作列表。
- `index`：步骤编号，便于报错定位。
- `action`：技能名，机器人侧需要映射到真实 SDK 或 ROS 接口。
- `parameters`：执行参数。
- `stage`：生成该步骤时所属阶段，主要用于调试，可不参与实际执行。
- `reason`：生成原因，主要用于人工查看，可不参与实际执行。

如果机器人侧想进一步简化解析，最低可接受结构为：

```json
{
  "schema_version": "robot_execution_steps_v1",
  "plan_id": "recovery_plan_xxx",
  "steps": [
    {
      "action": "move_to_pregrasp",
      "parameters": {}
    }
  ]
}
```

## 2. 机器人侧执行建议

机器人侧解析 `robot_execution_steps.json` 时，建议执行以下流程：

```text
1. 检查 schema_version 是否支持
2. 顺序读取 steps
3. 检查 action 是否属于白名单
4. 检查 parameters 是否齐全、是否越界
5. 将 action + parameters 映射为真机 SDK/ROS 调用
6. 每步执行后记录 step_report
7. 整体生成 real_execution_report.json
```

遇到以下情况应拒绝执行：

- 未知 `action`
- 缺少必填参数
- 参数超出真机安全范围
- 需要人工确认但未确认
- 急停或现场安全策略触发

## 3. 当前技能白名单

依据当前 `experience_system/experience_core/robot_plan_executor.py` 的默认注册表，经验库侧当前允许的技能包括：

```text
approach_object
adjust_torso_for_reach
base_move_to_place
base_move_to_region
base_move
choose_alternate_place
detect_multiple_objects
detect_place_occupancy
dual_arm_approach
dual_arm_level_object
dual_arm_place
dual_arm_pregrasp
dual_arm_synchronized_lift
dual_gripper_close
dual_gripper_release
head_camera_capture
left_arm_move_to_position
left_gripper_close
left_gripper_set
left_vertical_lift
move_to_pregrasp
open_gripper_release
place_object
recover_from_joint_limit
reposition_base_for_reach
retry_lift_after_grasp_check
retry_pregrasp_with_safer_offset
right_arm_move_to_position
right_gripper_set
safe_transport_pose
segmented_transport
select_correct_object
slow_cartesian_approach
torso_move_to_posture
torso_set_height
verify_grasp
verify_place_zone
```

其中可以分成两类：

### 3.1 现场原子技能

这类最适合直接对接真机：

```text
base_move
torso_move_to_posture
left_arm_move_to_position
right_arm_move_to_position
left_gripper_set
right_gripper_set
head_camera_capture
```

### 3.2 任务语义技能

这类更像“技能级命令”，机器人侧需要自己实现更高层封装：

```text
detect_multiple_objects
select_correct_object
move_to_pregrasp
approach_object
left_gripper_close
left_vertical_lift
detect_place_occupancy
choose_alternate_place
place_object
open_gripper_release
verify_grasp
verify_place_zone
reposition_base_for_reach
adjust_torso_for_reach
retry_pregrasp_with_safer_offset
slow_cartesian_approach
recover_from_joint_limit
retry_lift_after_grasp_check
base_move_to_region
torso_set_height
safe_transport_pose
base_move_to_place
dual_arm_pregrasp
dual_arm_approach
dual_gripper_close
dual_arm_synchronized_lift
dual_arm_level_object
segmented_transport
dual_arm_place
dual_gripper_release
```

如果机器人侧第一版只想做稳定对接，建议优先实现原子技能，不要一开始就全接。

## 4. 每个技能的作用和参数

下面先按技能说明“作用”和“机器人侧至少应支持的参数”。

### 4.1 detect_multiple_objects

作用：

```text
检测当前场景中的候选目标物，为后续选择目标物和抓取做准备。
```

当前代码中常见参数：

```json
{
  "object_bodies": [],
  "object_geoms": [],
  "object_sites": [],
  "name_prefix": "",
  "exclude_prefix": [],
  "workspace_bounds": [],
  "movable_only": true
}
```

建议：

- 如果真机侧已有成熟感知模块，可允许 `parameters` 为空，由机器人侧直接返回检测结果。
- 如果需要用文件驱动感知，也可以把目标类别、工作空间过滤条件写在 `parameters` 里。

### 4.2 select_correct_object

作用：

```text
从候选物体中选出当前任务的目标物。
```

当前代码中常见参数：

```json
{
  "objects": [],
  "target_label": "",
  "target_name": "",
  "target_position": [0.0, 0.0, 0.0],
  "require_unique": false
}
```

### 4.3 move_to_pregrasp

作用：

```text
将单臂移动到目标物附近的预抓取位姿。
```

当前代码常见参数来源包括抓取工具、目标物、偏移和运动控制项。建议机器人侧至少支持：

```json
{
  "side": "left",
  "object_body": "target_object",
  "pregrasp_distance": 0.06,
  "grasp_offset_x": 0.0,
  "grasp_offset_y": 0.0,
  "grasp_offset_z": 0.0,
  "steps": 900,
  "settle_steps": 220,
  "segment_count": 8,
  "max_joint_step": 0.004,
  "fail_threshold": 0.04,
  "velocity_limit": 0.45,
  "force_scale": 0.65,
  "direct_qpos": false
}
```

### 4.4 approach_object

作用：

```text
从预抓取位姿向目标物做最后一段接近。
```

建议支持：

```json
{
  "side": "left",
  "object_body": "target_object",
  "approach_dx": 0.0,
  "approach_dy": 0.0,
  "approach_dz": -1.0,
  "steps": 1200,
  "settle_steps": 260,
  "segment_count": 10,
  "velocity_limit": 0.25,
  "force_scale": 0.45,
  "fail_threshold": 0.04,
  "direct_qpos": false
}
```

### 4.5 left_gripper_close

作用：

```text
关闭左夹爪执行抓取。
```

建议支持：

```json
{
  "object_body": "target_object",
  "gripper_steps": 240,
  "direct_qpos": false,
  "closure_bias": 0.0,
  "attach_on_close": false
}
```

说明：

- 真机侧通常不需要 `attach_on_close` 这种仿真参数，但可以保留并忽略。

### 4.6 left_vertical_lift

作用：

```text
抓住目标物后做竖直抬升。
```

建议支持：

```json
{
  "side": "left",
  "lift_dx": 0.0,
  "lift_dy": 0.0,
  "lift_dz": 0.12,
  "lift_height": 0.12,
  "steps": 1500,
  "settle_steps": 3000,
  "max_joint_step": 0.006,
  "fail_threshold": 0.02,
  "orientation_threshold": 1.0,
  "orientation_weight": 0.02,
  "force_scale": 1.0,
  "direct_qpos": false
}
```

### 4.7 detect_place_occupancy

作用：

```text
检测默认放置区域是否被占用。
```

建议支持：

```json
{
  "place_site": "place_zone_site",
  "candidate_bodies": [],
  "exclude_bodies": [],
  "occupancy_radius": 0.12,
  "z_tolerance": 0.12
}
```

### 4.8 choose_alternate_place

作用：

```text
在默认放置区不可用时，选择一个可用的替代放置区。
```

建议支持：

```json
{
  "place_sites": [],
  "place_site": "place_zone_site",
  "alternate_place_site": "alternate_place_zone_site",
  "candidate_bodies": [],
  "exclude_bodies": [],
  "occupancy_radius": 0.12,
  "z_tolerance": 0.12
}
```

### 4.9 place_object

作用：

```text
将已经抓住的目标物移动到放置区并完成放置动作。
```

建议支持：

```json
{
  "side": "left",
  "place_offset_x": 0.0,
  "place_offset_y": 0.0,
  "place_offset_z": 0.0,
  "steps": 1200,
  "settle_steps": 260,
  "max_joint_step": 0.006,
  "fail_threshold": 0.03,
  "orientation_threshold": 1.0,
  "orientation_weight": 0.02,
  "force_scale": 1.0,
  "direct_qpos": false
}
```

### 4.10 open_gripper_release

作用：

```text
在放置动作完成后，打开夹爪释放物体。
```

建议支持：

```json
{
  "gripper_steps": 360,
  "settle_steps": 240,
  "direct_qpos": false,
  "detach_on_release": true
}
```

### 4.11 verify_grasp

作用：

```text
检查目标物是否还在夹爪中、抓取是否有效。
```

建议支持：

```json
{
  "side": "left",
  "expected_object_body": "target_object",
  "expected_label": "",
  "max_grasp_distance": 0.08
}
```

### 4.12 verify_place_zone

作用：

```text
验证目标物是否已经落在放置区中。
```

建议支持：

```json
{
  "place_site": "place_zone_site",
  "candidate_bodies": [],
  "occupancy_radius": 0.12,
  "z_tolerance": 0.12
}
```

### 4.13 base_move

作用：

```text
将底盘移动到显式给定的二维位姿，可同时支持世界系绝对运动和 base 系相对运动。
```

建议支持：

```json
{
  "x": 0.08,
  "y": -0.03,
  "yaw": 0.0,
  "frame": "world",
  "steps": 400,
  "settle_steps": 80,
  "max_joint_step": 0.004,
  "fail_threshold": 0.05,
  "direct_qpos": false
}
```

约定：

- `frame = world`：`x/y/yaw` 表示世界坐标系下的绝对目标位姿
- `frame = base`：`x/y` 表示机器人当前 base 坐标系下的相对位移，`yaw` 表示相对当前朝向的转角
- `yaw` 按右手法则

兼容说明：

- 面向真机的执行文件建议统一使用 `base_move + x/y/yaw/frame`

### 4.14 torso_move_to_posture

作用：

```text
将躯干移动到指定姿态。
```

建议支持：

```json
{
  "target_qpos": [0.0, 0.02, 0.0, 0.02],
  "steps": 500,
  "settle_steps": 120,
  "max_joint_step": 0.004,
  "fail_threshold": 0.05,
  "closed_loop_gain": 1.0,
  "direct_qpos": false,
  "lock_posture": true
}
```

也可以兼容另一种写法：

```json
{
  "torso_joint1": 0.0,
  "torso_joint2": 0.02,
  "torso_joint3": 0.0,
  "torso_joint4": 0.02
}
```

### 4.15 left_arm_move_to_position

作用：

```text
将左臂末端移动到指定三维目标位置。
```

建议支持：

```json
{
  "target_x": 0.32,
  "target_y": 0.12,
  "target_z": 0.90,
  "target_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "steps": 1200,
  "settle_steps": 400,
  "stabilize": true,
  "lock_posture": true,
  "max_joint_step": 0.004,
  "fail_threshold": 0.03,
  "orientation_threshold": 0.15,
  "orientation_weight": 0.35,
  "control_frame": "grasp_tool",
  "pose_segment_count": 6,
  "pose_posture_gain": 0.02,
  "direct_qpos": false
}
```

### 4.16 right_arm_move_to_position

作用：

```text
将右臂末端移动到指定三维目标位置。
```

参数格式与 `left_arm_move_to_position` 相同。

### 4.17 left_gripper_set

作用：

```text
显式设置左夹爪开合状态。
```

建议支持：

```json
{
  "state": 1,
  "direct_qpos": false
}
```

约定：

- `state = 1` 表示闭合
- `state = 0` 表示打开

### 4.18 right_gripper_set

作用：

```text
显式设置右夹爪开合状态。
```

参数格式与 `left_gripper_set` 相同。

### 4.19 head_camera_capture

作用：

```text
采集头部 RGB/RGB-D 图像。
```

建议支持：

```json
{
  "width": 320,
  "height": 240,
  "include_depth": true
}
```

## 5. 其它技能说明

下面这些技能当前已经在白名单里，但更适合先作为“规划侧语义动作”，不建议第一版真机立即全接：

```text
base_move_to_region
torso_set_height
safe_transport_pose
reposition_base_for_reach
adjust_torso_for_reach
retry_pregrasp_with_safer_offset
slow_cartesian_approach
recover_from_joint_limit
retry_lift_after_grasp_check
base_move_to_place
dual_arm_pregrasp
dual_arm_approach
dual_gripper_close
dual_arm_synchronized_lift
dual_arm_level_object
segmented_transport
dual_arm_place
dual_gripper_release
```

原因：

- 这些技能通常会组合调用多个底层技能。
- 它们依赖较强的场景语义、规划语义或双臂协调逻辑。
- 真机侧如果没有完全对应的高层控制器，直接解析会比较脆弱。

建议第一版策略：

```text
优先接原子技能
-> 稳定完成单步解析和执行
-> 再逐步接入高层语义技能
```

## 6. 建议的机器人回传文件

真机执行后建议输出：

```text
real_execution_report.json
```

建议结构：

```json
{
  "schema_version": "real_robot_execution_report_v1",
  "plan_id": "recovery_plan_xxx",
  "executor_name": "robot_sdk_executor",
  "started_at": "2026-01-01T10:00:00Z",
  "finished_at": "2026-01-01T10:00:10Z",
  "success": true,
  "status": "executed",
  "step_reports": [
    {
      "index": 0,
      "action": "head_camera_capture",
      "parameters": {
        "width": 320,
        "height": 240,
        "include_depth": true
      },
      "success": true,
      "status": "ok",
      "message": "",
      "final_error": 0.0,
      "risk_flags": [],
      "raw": {}
    }
  ]
}
```

这样便于后续：

```text
real_execution_report.json
-> 整理成 real episode
-> validate_real_episode.py
-> import_real_episode.py
-> 写回经验库
```

## 7. 第一版落地建议

如果现在要尽快和真机侧对接，建议第一版只强制支持下面 8 个动作：

```text
base_move
torso_move_to_posture
left_arm_move_to_position
right_arm_move_to_position
left_gripper_set
right_gripper_set
head_camera_capture
```

然后把下面这些作为第二阶段：

```text
detect_multiple_objects
select_correct_object
move_to_pregrasp
approach_object
left_gripper_close
left_vertical_lift
detect_place_occupancy
choose_alternate_place
place_object
open_gripper_release
verify_grasp
verify_place_zone
```

原因很简单：

- 第一阶段参数结构更稳定
- 更容易映射到机器人 SDK
- 更容易做安全约束
- 更容易做失败回传和复现
