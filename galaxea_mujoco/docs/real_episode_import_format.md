# Real Episode Import Format

`source/import_real_episode.py` 现在支持三种输入：

1. 单个 JSON：`--input episode.json`
2. 单个 episode 目录：`--episode-dir /path/to/episode_dir`
3. 批量目录：`--batch-dir /path/to/root`

## 目录结构

最小目录格式：

```text
episode_dir/
  episode.json | real_episode.json | result.json
```

可选补充文件：

```text
episode_dir/
  experience_after.json
  sensor_summary.json | sensors.json
  keyframes/*.jpg|png
  frames/*.jpg|png
  episode.hdf5 | episode.h5
  robot_log.jsonl | robot_log.json
  video/ | videos/ | rgb/ | camera/
```

导入时会自动：

- 读取 `episode.json / real_episode.json / result.json`
- 合并 `experience_after.json`
- 合并 `sensor_summary.json / sensors.json`
- 从 `keyframes/` 或 `frames/` 生成关键帧
- 把 HDF5、视频目录、日志目录写入 `raw_refs`
- 生成 `real_episode_ref`

## 最小 JSON 字段

最小可用字段：

```json
{
  "episode_id": "real_ep_001",
  "scenario_id": "G3",
  "condition_id": "place_occupied",
  "task_stage": "place",
  "observed_pos": [0.1, 0.2, 0.3],
  "executed_recovery_steps": [
    {"action": "detect-place"},
    {"action": "choose-alternate-place"},
    {"action": "place-object", "success": true}
  ],
  "recovery_success": true,
  "failure_reason": ""
}
```

## 推荐字段

推荐再提供这些字段：

```json
{
  "source": "real",
  "backend": "real_robot",
  "robot_type": "mobile_single_arm",
  "robot_id": "r1pro_real_001",
  "task_name": "g3_sorting",
  "result": {
    "success": true,
    "recovery_success": true,
    "task_success": true,
    "failure_reason": ""
  },
  "object_state": {
    "target_object": "target_cube",
    "object_class": "cube",
    "objects": {
      "target_cube": {
        "observed_position": [0.1, 0.2, 0.3]
      }
    }
  },
  "sensor_summary": {
    "gripper_state": {"left": "open", "right": "open"},
    "contact_state": {"left": false, "right": false}
  },
  "real_episode_ref": {
    "raw_episode_id": "real_ep_001",
    "hdf5_path": "/data/real_ep_001.h5",
    "video_dir": "/data/real_ep_001/video",
    "robot_log_path": "/data/real_ep_001/robot_log.jsonl"
  },
  "keyframes": [
    {"stage": "before", "image_path": "keyframes/before_recovery.jpg"},
    {"stage": "after", "image_path": "keyframes/after_recovery.jpg"}
  ]
}
```

完整模板见：

```text
docs/real_episode_template.json
```

## 字段别名

adapter 也接受这些别名：

- `skill_sequence` 或 `executed_recovery_steps` 或 `recovery_steps`
- `scenario_id` 或 `scene_id`
- `condition_id` 或 `anomaly_id`
- `task_stage` 或 `stage`
- `object_class` 或 `target_class`
- `target_object` 或 `target_name`
- `state_before` 或 `before_state`
- `state_after` 或 `after_state`
- `sensor_summary.ee_pose -> end_effector_pose`
- `sensor_summary.contacts -> contact_state`

## 导入命令

导入前建议先校验：

```bash
python source/validate_real_episode.py \
  --input docs/real_episode_template.json \
  --source real \
  --backend real_robot \
  --strict
```

如果要检查 keyframes、HDF5、视频、日志路径是否存在，加：

```bash
python source/validate_real_episode.py \
  --episode-dir /path/to/episode_dir \
  --source real \
  --backend real_robot \
  --check-refs \
  --strict
```

单目录：

```bash
python source/import_real_episode.py \
  --episode-dir /path/to/episode_dir \
  --source real \
  --backend real_robot \
  --universal-experience-lib results/memory/universal_experience_v1.json
```

批量目录：

```bash
python source/import_real_episode.py \
  --batch-dir /path/to/episode_root \
  --source pseudo_real \
  --backend real_robot \
  --universal-experience-lib results/memory/universal_experience_v1.json
```
