# 当前 Galaxea Field Atomic 技能清单

本文档记录当前实验使用的 Galaxea 原子技能名称和参数。大模型生成计划时只应输出：

```json
{
  "steps": [
    {
      "action": "技能名",
      "parameters": {}
    }
  ]
}
```

所有距离单位均为米。坐标相关参数默认在 `torso_link4` 坐标系下表达。

## 允许的 target_class

当前实验允许以下物体名称：

```text
apple
red sphere
red cube
red box
green box
```

`target_class` 是技能之间连接物体位置的关键字段。识别技能会把该物体的位置写入运行时 tmp 目录，后续同名 `target_class` 的技能会读取这个位置。

## 技能列表

### move_base_relative

作用：底盘相对移动，用于目标超出手臂可达范围时调整机器人位置。

参数：

```json
{
  "x": 0.0,
  "y": 0.0
}
```

范围：

- `x`: `[-0.5, 0.5]`
- `y`: `[-0.5, 0.5]`

### set_torso_posture

作用：设置躯干高度档位。

参数：

```json
{
  "level": "mid"
}
```

可选值：

```text
mid
high
```

当前档位与真机 `真机代码/skill.py` 对齐：

- `high`: `[0.0, 0.0, 0.0, 0.0]`
- `mid`: `[0.87, -1.35, -0.48, 0.0]`

注意：当前 MuJoCo 模型的前三个 torso 关节方向与真机命令约定相反，所以仿真内部会把真机 `mid` 映射为 `[-0.87, 1.35, 0.48, 0.0]` 执行。对大模型和实验 JSON 来说仍然只需要输出 `"level": "mid"`。

腰部移动成功后，本轮运行目录下的旧物体位置和旧轨迹会被清理；后续必须重新执行视觉识别和轨迹规划。

仿真中 `head_camera_rgbd_save` 会根据当前腰部档位自动切换 `head_top_work_camera` 的局部相机角度：

- `high`: 使用原始相机角度，适合直立腰部下识别桌面物体。
- `mid`: 使用真机相机角度，适合中位腰部下识别桌面物体。

实验 JSON 不需要传相机位姿参数，只需要先执行 `set_torso_posture` 并在腰部移动后重新执行 `head_camera_rgbd_save` / `head_camera_grounded_sam2_pose`。

注意：躯干高度变化后，相机位姿随之变化，后续应重新执行视觉识别。

### head_camera_rgbd_save

作用：保存头部相机 RGB-D 图像，供后续视觉定位使用。

参数：

```json
{}
```

### head_camera_grounded_sam2_pose

作用：识别目标物体，并保存该物体的位置 JSON。

参数：

```json
{
  "target_class": "apple"
}
```

范围：

- `target_class`: 见“允许的 target_class”

### plan_cartesian_trajectory

作用：生成到预抓取点的轨迹规划。该技能只规划轨迹，不负责最终抓取。

参数：

```json
{
  "side": "left",
  "target_class": "apple",
  "pregrasp_offset_x": 0.0,
  "pregrasp_offset_y": 0.0,
  "pregrasp_offset_z": 0.08,
  "mode": "side_then_in",
  "side_offset_x": -0.06,
  "side_offset_y": 0.0,
  "clearance_z": 0.08,
  "topdown_mode": "palm_down"
}
```

范围：

- `side`: `left | right`
- `target_class`: 见“允许的 target_class”
- `pregrasp_offset_x`: `[-0.2, 0.2]`
- `pregrasp_offset_y`: `[-0.2, 0.2]`
- `pregrasp_offset_z`: `[-0.05, 0.2]`
- `mode`: `straight | top_then_down | side_then_in`
- `side_offset_x`: `[-0.1, 0.1]`
- `side_offset_y`: `[-0.1, 0.1]`
- `clearance_z`: `[0.0, 0.15]`
- `topdown_mode`: `palm_down | vertical_down | forward_parallel | current`

轨迹模式：

- `straight`: 直接到预抓取点，路径最短，只适合中间无障碍情况。
- `top_then_down`: 先抬高 `clearance_z`，再平移到目标上方，最后下降。
- `side_then_in`: 先到 `pregrasp + [side_offset_x, side_offset_y, 0]` 的入口点，再进入预抓取点。

### move_to_pregrasp

作用：执行移动到预抓取点。若前面用了 `plan_cartesian_trajectory`，这里应重复相同的关键参数。

参数：

```json
{
  "side": "left",
  "target_class": "apple",
  "pregrasp_offset_x": 0.0,
  "pregrasp_offset_y": 0.0,
  "pregrasp_offset_z": 0.08,
  "topdown_mode": "palm_down"
}
```

范围：

- `side`: `left | right`
- `target_class`: 见“允许的 target_class”
- `pregrasp_offset_x`: `[-0.2, 0.2]`
- `pregrasp_offset_y`: `[-0.2, 0.2]`
- `pregrasp_offset_z`: `[-0.05, 0.2]`
- `topdown_mode`: `palm_down | vertical_down | forward_parallel | current`

### approach_object

作用：从预抓取点进入最终抓取点。

参数：

```json
{
  "side": "left",
  "target_class": "apple",
  "visual_grasp_offset_z": 0.007,
  "topdown_mode": "palm_down"
}
```

范围：

- `side`: `left | right`
- `target_class`: 见“允许的 target_class”
- `visual_grasp_offset_z`: `[-0.05, 0.08]`
- `topdown_mode`: `palm_down | vertical_down | forward_parallel | current`

### close_gripper

作用：闭合夹爪。

参数：

```json
{
  "side": "left"
}
```

范围：

- `side`: `left | right`

### lift

作用：抓取后沿 torso 正 z 方向提升物体。

参数：

```json
{
  "side": "left",
  "target_class": "apple",
  "lift_height": 0.1
}
```

范围：

- `side`: `left | right`
- `target_class`: 见“允许的 target_class”
- `lift_height`: `[0.02, 0.25]`

### transport_to_detected_target

作用：夹住物体后，移动到已经识别出的目标物体位置附近，并允许小范围 xy 放置偏移。

参数：

```json
{
  "side": "left",
  "target_class": "red box",
  "place_offset_x": 0.0,
  "place_offset_y": 0.0
}
```

范围：

- `side`: `left | right`
- `target_class`: 见“允许的 target_class”
- `place_offset_x`: `[-0.02, 0.02]`
- `place_offset_y`: `[-0.02, 0.02]`

限制：

- 必须先执行 `head_camera_grounded_sam2_pose` 获取同名 `target_class` 的位置。
- 不接受 `place_offset_z`。
- z 高度保持当前 TCP 高度。

### lower_held_object

作用：夹住物体后，沿 torso 负 z 方向下降，用于放置前靠近目标表面。

参数：

```json
{
  "side": "left",
  "lower_distance": 0.04
}
```

范围：

- `side`: `left | right`
- `lower_distance`: `[0.0, 0.08]`

限制：

- 执行过程中保持夹爪闭合。
- 该技能只负责 TCP 下降，不负责判断放置是否稳定。

### open_gripper

作用：打开夹爪，用于释放物体或失败恢复。

参数：

```json
{
  "side": "left"
}
```

范围：

- `side`: `left | right`

## 朝向参数

只有抓取前和抓取阶段需要让大模型控制 `topdown_mode`：

```text
plan_cartesian_trajectory
move_to_pregrasp
approach_object
```

抓住物体后的技能不应重新控制夹爪朝向：

```text
lift
transport_to_detected_target
lower_held_object
close_gripper
open_gripper
```

`topdown_mode` 可选值：

- `palm_down`: 默认桌面抓取姿态。
- `vertical_down`: 竖直向下抓取。
- `forward_parallel`: 夹爪面平行于地面并向前。
- `current`: 保持当前 TCP 朝向，主要用于调试或特殊情况。

## 推荐技能序列

### 抓取并提升 apple

```json
{
  "steps": [
    {"action": "head_camera_rgbd_save", "parameters": {}},
    {"action": "head_camera_grounded_sam2_pose", "parameters": {"target_class": "apple"}},
    {"action": "plan_cartesian_trajectory", "parameters": {"side": "left", "target_class": "apple", "mode": "side_then_in", "pregrasp_offset_x": 0.0, "pregrasp_offset_y": 0.0, "pregrasp_offset_z": 0.08, "side_offset_x": -0.06, "side_offset_y": 0.0, "topdown_mode": "palm_down"}},
    {"action": "move_to_pregrasp", "parameters": {"side": "left", "target_class": "apple", "pregrasp_offset_x": 0.0, "pregrasp_offset_y": 0.0, "pregrasp_offset_z": 0.08, "topdown_mode": "palm_down"}},
    {"action": "approach_object", "parameters": {"side": "left", "target_class": "apple", "visual_grasp_offset_z": 0.007, "topdown_mode": "palm_down"}},
    {"action": "close_gripper", "parameters": {"side": "left"}},
    {"action": "lift", "parameters": {"side": "left", "target_class": "apple", "lift_height": 0.1}}
  ]
}
```

### 把 apple 放到 red box 上

```json
{
  "steps": [
    {"action": "head_camera_rgbd_save", "parameters": {}},
    {"action": "head_camera_grounded_sam2_pose", "parameters": {"target_class": "red box"}},
    {"action": "transport_to_detected_target", "parameters": {"side": "left", "target_class": "red box", "place_offset_x": 0.0, "place_offset_y": 0.0}},
    {"action": "lower_held_object", "parameters": {"side": "left", "lower_distance": 0.04}},
    {"action": "open_gripper", "parameters": {"side": "left"}}
  ]
}
```
