# R1Pro 真机 Episode 采集规范

这份文档规定一次 R1Pro 真机实验需要保存哪些数据，才能导入共同的 universal experience library。

## 目标

每个真机 episode 至少要回答四个问题：

1. 执行了什么任务和条件？
2. 实际执行了哪些技能/动作，哪些成功，哪些失败？
3. 哪些传感器证据支持这个结果？
4. 是否可以直接校验并导入经验库？

一个 episode 对应一个目录，目录中包含 `episode.json` 和传感器/日志文件夹。

## 目录结构

推荐结构：

```text
r1pro_real_episode_xxx/
  episode.json
  rgb/
  depth/
  lidar/
  force/
  keyframes/
  video/
  logs/
```

新建一次实验目录：

```bash
PYTHONPATH=experience_system python -B \
  experience_system/tools/create_real_episode_template.py \
  --output-dir /tmp/r1pro_real_episode_demo \
  --episode-id r1pro_real_demo_001 \
  --scenario G1 \
  --condition clean \
  --task-name grasp_place_demo
```

## 必填字段

`episode.json` 中必须填写：

- `episode_id`：全局唯一实验编号。
- `scenario_id`：任务类型，例如 `G1`、`G3`、`G4`。
- `condition_id`：条件/异常，例如 `clean`、`place_occupied`、`grasp_miss`。
- `robot_type`：填写 `r1pro`。
- `object_class`：目标物体类别。
- `skill_sequence`：实际执行的技能序列。
- `result`：实验结果，至少包含 `success`、`recovery_success`、`task_success`。

建议填写：

- `robot_id`：真机编号。
- `operator`：实验人员或运行程序。
- `task_name`：任务名称。
- `target_object`：目标物体实例名。
- `failure_reason`：失败时填写简短原因。

## 传感器证据

### RGB-D 相机

外接头部 RGB-D 相机写入 `visual_observation`。

必填字段：

- `camera_name`：填写 `external_head_rgbd_camera`。
- `rgb_path`：RGB 图像路径，例如 `rgb/000000.png`。
- `depth_path`：深度图路径，例如 `depth/000000.npy`。

建议字段：

- `rgb_resolution`：`[width, height]`。
- `depth_resolution`：`[width, height]`。
- `timestamp`：采集时间或机器人时钟。
- `intrinsics`：相机内参，如果可用。
- `extrinsics`：相机到机器人 base/head frame 的外参，如果可用。

RGB 关键帧建议保存为普通图片。深度图可以是 `.npy`、`.png` 或其他项目可读格式，但路径必须和 `episode.json` 一致。

### 底盘激光雷达

底盘 360 度雷达写入 `lidar_observation`。

必填字段：

- `site_name`：填写 `base_lidar_site`。
- `scan_path`：雷达扫描文件路径，例如 `lidar/scan_000000.json`。

建议字段：

- `ray_count`：扫描点数或 beam 数。
- `horizontal_fov_deg`：通常为 `360.0`。
- `min_range`：近距离盲区，通常为 `0.1`。
- `max_range`：记录时使用的最大量程。
- `nearest_obstacle_distance`：episode 中最近障碍距离。
- `timestamp`：扫描时间或机器人时钟。

扫描文件可以保存原始 ranges、角度元数据，或 ROS 消息转出的 JSON。经验库当前只要求路径和摘要字段有效，原始格式后续可以扩展。

### 腕部/末端受力

末端外力证据写入 `wrist_force_observation`。

必填字段：

- `left.log_path` 或 `right.log_path`：左/右腕力日志路径。
- `peak_force_norm`：episode 中最大力模长。

建议字段：

- `left.force_norm`、`right.force_norm`：左/右侧最终或峰值力。
- `left.torque_norm`、`right.torque_norm`：左/右侧最终或峰值力矩。
- `samples`：紧凑时间序列，如果可用。
- `threshold_exceeded`：是否超过安全或接触阈值。

如果真机没有直接腕部六维力传感器，可以在 `wrist_force_observation.source` 中注明估计来源，例如 `joint_torque_estimate` 或 `controller_contact_estimate`。

## Keyframes

`keyframes` 用于视觉检索和论文证据。推荐阶段：

- `before_grasp`
- `after_grasp`
- `before_place`
- `after_place`
- `failure_state`

每个 keyframe 建议包含：

- `stage`
- `image_path`
- `description`
- `used_for_retrieval`

示例：

```json
{
  "stage": "after_grasp",
  "image_path": "keyframes/after_grasp.png",
  "description": "object lifted by left gripper",
  "used_for_retrieval": true
}
```

## 技能序列

`skill_sequence` 记录真实执行过的动作，不只记录计划动作。技能名尽量和仿真侧保持一致。

示例：

```json
[
  {"name": "detect_target", "success": true},
  {"name": "move_to_pregrasp", "success": true},
  {"name": "grasp_target", "success": true},
  {"name": "place_target", "success": false, "message": "place zone occupied"}
]
```

失败技能也要保留在序列中，并设置 `success: false`。这样失败 episode 才能进入 Top-O 风险记忆和后续检索。

## 校验

导入前先校验：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B galaxea_mujoco/source/validate_real_episode.py \
  --episode-dir /path/to/r1pro_real_episode_xxx \
  --check-refs
```

期望结果：

```json
{
  "error_count": 0,
  "passed": true
}
```

warning 通常表示引用的图像、深度图、雷达扫描、腕力日志、视频或 keyframe 路径不存在。如果该 episode 要作为论文证据，建议先清掉 warning。

## 导入

导入共同经验库：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B galaxea_mujoco/source/import_real_episode.py \
  --episode-dir /path/to/r1pro_real_episode_xxx \
  --source real \
  --universal-experience-lib galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --report galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/import_real_episode_report.json
```

导入后重新生成 real-format evidence pack：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B galaxea_mujoco/source/build_real_format_evidence_pack.py \
  --input galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/universal_experience_library.json \
  --save galaxea_mujoco/results/memory/universal_pipeline_calibration_v1/real_format_evidence_pack.json
```

重点检查这些字段：

- `real_format_entry_count`
- `sensor_evidence_entry_count`
- `rgbd_evidence_entry_count`
- `lidar_evidence_entry_count`
- `wrist_force_evidence_entry_count`
- `sensor_modality_distribution`

## 离场前检查

离开机器人现场前确认：

- `episode.json` 存在并能通过校验。
- `result.success` 和 `result.task_success` 反映真实结果。
- `skill_sequence` 包含失败步骤。
- RGB 图像路径存在。
- 使用 RGB-D 时，深度图路径存在。
- 雷达扫描路径存在。
- 腕力日志路径存在，或写明不可用原因。
- 至少有一张 keyframe 图像。
- `validate_real_episode.py --check-refs` 的 `error_count=0`。

