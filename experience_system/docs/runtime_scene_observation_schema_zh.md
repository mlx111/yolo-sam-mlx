# Runtime Scene 现场观测 JSON 规范

本文档固定现场 RGB-D / LiDAR / 机器人状态需要整理成的 JSON 格式。

目标是把现场观测转换成：

```text
field_runtime_scene_observation_v1
-> runtime_sandbox_scene_v1
-> runtime_scene.xml
-> MuJoCo sandbox rollout
```

模板文件：

```text
experience_system/templates/field_runtime_scene_observation_template.json
```

## 1. 为什么需要这个格式

现场物体位置不是固定的，不能直接使用固定 G3/G4 XML。

必须先把真实观测整理为结构化 JSON，再生成 MuJoCo runtime XML。

这样可以保证：

```text
1. LLM 看到的是当前现场状态。
2. sandbox 验证的是当前现场布局。
3. 真机执行前能检查障碍物、目标物体和放置区。
4. 后续经验库能记录“这次是在什么现场状态下成功或失败”。
```

## 2. 顶层字段

必须包含：

```text
schema_version
scene_id
timestamp
coordinate_frame
robot_state
table
objects
obstacles
place_zones
sensor_refs
calibration
runtime_scene
metadata
```

## 3. coordinate_frame

用于说明坐标系。

```json
{
  "world_frame": "mujoco_world",
  "robot_base_frame": "r1pro_base_link",
  "camera_frame": "external_head_rgbd_camera",
  "lidar_frame": "base_lidar_site",
  "units": "meter_radian"
}
```

要求：

```text
所有 object / obstacle / place_zone 的 pose 必须转换到 world_frame。
如果现场只拿到相机坐标系，需要先用外参转换。
```

## 4. robot_state

记录当前机器人状态。

```json
{
  "base_pose_xyyaw": [0.0, 0.0, 0.0],
  "torso_qpos": [0.0, 0.0, 0.0, 0.0],
  "left_arm_qpos": [],
  "right_arm_qpos": [],
  "left_gripper_state": "open",
  "right_gripper_state": "open"
}
```

现场如果暂时拿不到完整关节状态，可以先留空，但要在 `metadata.notes` 中说明。

## 5. objects

目标物体和其他可操作物体。

```json
{
  "name": "target_object",
  "class": "unknown_object",
  "pose": [0.12, 0.08, 0.79],
  "size": [0.025, 0.025, 0.025],
  "geom_type": "box",
  "mass": 0.05,
  "confidence": 0.8,
  "source": "rgbd_detection",
  "freejoint": true
}
```

字段说明：

```text
name: runtime XML 中的 body/site 名称。
class: 感知类别，可以是 sphere/cube/bottle/unknown_object。
pose: world_frame 下的位置，单位米。
size: MuJoCo geom size。
geom_type: box / sphere / cylinder。
confidence: 感知置信度。
source: rgbd_detection / manual / lidar_cluster。
freejoint: 是否允许物体在仿真中被抓取移动。
```

## 6. obstacles

障碍物格式与 objects 类似。

区别是：

```text
obstacles 默认不是任务目标。
它们用于 sandbox 碰撞/路径/可达性检查。
```

## 7. place_zones

放置区域。

```json
{
  "name": "primary_place_zone",
  "pose": [0.02, 0.24, 0.805],
  "size": [0.075, 0.055, 0.008],
  "confidence": 0.8,
  "source": "operator_or_perception"
}
```

现场如果放置区由人工指定，也应写入 `source=operator`。

## 8. sensor_refs

只保存路径引用，不直接把大图像写进 JSON。

```json
{
  "rgb_path": "camera/rgb_0001.png",
  "depth_path": "camera/depth_0001.npy",
  "lidar_path": "lidar/scan_0001.json",
  "detection_path": "perception/detections_0001.json",
  "calibration_path": "calibration/extrinsics.json"
}
```

## 9. calibration

用于记录外参和不确定性。

```json
{
  "camera_to_world": [],
  "lidar_to_world": [],
  "pose_uncertainty_xyz": [0.02, 0.02, 0.02],
  "yaw_uncertainty_rad": 0.1
}
```

如果现场外参还没标定，应保留空数组，并在 `metadata.notes` 说明。

## 10. 转换为 runtime_sandbox_scene_v1

`runtime_sandbox_scene_v1` 当前使用字段：

```text
scene_id
robot_model_include
timestep
table_pose
table_size
target_object
objects
obstacles
place_zones
metadata
```

转换规则：

```text
table.pose -> table_pose
table.size -> table_size
runtime_scene.robot_model_include -> robot_model_include
runtime_scene.timestep -> timestep
runtime_scene.target_object -> target_object
objects -> objects
obstacles -> obstacles
place_zones -> place_zones
sensor_refs/calibration/robot_state -> metadata
```

## 11. 现场最小可用要求

最小必须有：

```text
scene_id
target object pose
target object size
table pose / size
robot_model_include
```

建议同时有：

```text
RGB 图像路径
深度图路径
LiDAR 路径
机器人 base/torso 状态
障碍物列表
放置区列表
```

## 12. 当前边界

可以声称：

```text
系统定义了现场观测到 runtime sandbox scene 的统一 JSON 格式。
```

不能声称：

```text
真实 RGB-D/LiDAR 已经自动稳定生成高精度 MuJoCo 场景。
```
