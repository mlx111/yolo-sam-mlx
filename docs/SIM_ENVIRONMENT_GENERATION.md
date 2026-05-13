# 仿真环境生成流程说明

本文档说明当前项目里两条生成 MuJoCo 运行场景的流程：

- 真实转仿真：从真实相机 RGB-D 图片生成仿真环境。
- 仿真套仿真：从已有 MuJoCo 虚拟环境采集到的 RGB-D 图片再生成一个新的仿真环境。

两条流程最终都会生成同一类运行文件，后续 grasp、chat、异常处理仍然读取生成后的运行场景。

## 共同输出

主要输出文件如下：

```bash
manipulator_grasp/assets/scenes/apple_pear_runtime.xml
manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml
manipulator_grasp/assets/fruit/stl/apple.stl
manipulator_grasp/assets/fruit/stl/pear.stl
runtime_pose_calibration.json
runtime_assets/left_view_refined_pose.json
```

其中 grasp 服务实际使用的重点文件是：

```bash
manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml
```

如果重新生成了场景，建议重启 grasp/chat 服务，让服务重新加载最新 XML。

## 真实转仿真

真实转仿真入口是：

```bash
python zhenghe2_buquan.py
```

如果只生成指定物体，可以传入 JSON 列表：

```bash
python zhenghe2_buquan.py '["apple", "pear"]'
```

当前 `zhenghe2_buquan.py` 的实现会在场景生成后直接启动 `grasp_fastapi_completion_v4` 服务。脚本里保留了 `--start-server` 参数，但现在主流程已经会启动服务，因此通常不需要额外加这个参数。

### 输入要求

真实图片和深度图仍然使用项目原有命名，主要放在 `inputs/` 下，例如：

```bash
inputs/cleft001.png
inputs/cright001.png
inputs/dleft001.png
inputs/dright001.png
```

真实转仿真使用真实相机对应的标定和点云流程，不使用 MuJoCo 相机内参。

### 内部流程

`zhenghe2_buquan.py` 当前主要流程是：

1. 调用 `cv_proc.gen_mask(objects, recognition_mode="real")`，用模型识别真实图片里的物体并生成 mask。
2. 确保背景点云存在，必要时通过 `pointcloud_v2.point(...)` 生成背景点云。
3. 调用 `runtime_scene_original.build_original_runtime_scene_inputs(...)`，从真实 RGB-D、mask 和相机标定计算物体位置、相机位姿和校准数据。
4. 写出 `runtime_pose_calibration.json`。
5. 调用 `new_runtime.apple_pear_scene.resolve_meshes(...)`，用 buquan 流程生成或复用物体 STL。
6. 调用 `object_pose_runtime.estimate_runtime_object_quats(...)` 估计物体姿态。
7. 调用 `camera_facing_local_axis.align_object_quats_to_camera(...)` 对齐物体朝向。
8. 调用 `dong2.generate_scene(...)` 生成基础 XML。
9. 运行 `left_view_pose_refiner.py` 和 `apply_refined_pose_to_scene.py`，生成 refined XML。
10. 启动 `grasp_fastapi_completion_v4`。

### 适用场景

当输入图片来自真实相机时，使用这条流程。它依赖真实相机标定和真实点云转换逻辑。如果把虚拟相机图片直接放进这条流程，物体位置容易产生明显偏差，因为虚拟相机内参与真实相机不同。

## 仿真套仿真

仿真套仿真入口是：

```bash
python build_runtime_scene_from_sim_camera.py \
  --xml manipulator_grasp/assets/scenes/apple_pear_runtime_refined111_no_gripper.xml
```

常用参数：

```bash
# 指定物体
python build_runtime_scene_from_sim_camera.py \
  --xml manipulator_grasp/assets/scenes/apple_pear_runtime_refined111_no_gripper.xml \
  --objects apple pear

# 只生成基础场景，不执行 refined pose
python build_runtime_scene_from_sim_camera.py \
  --xml manipulator_grasp/assets/scenes/apple_pear_runtime_refined111_no_gripper.xml \
  --no-refine

# 生成后直接启动 grasp 服务
python build_runtime_scene_from_sim_camera.py \
  --xml manipulator_grasp/assets/scenes/apple_pear_runtime_refined111_no_gripper.xml \
  --start-server
```

### 输入要求

虚拟环境采集到的图片、深度和 mask 放在 `inputs/` 下。当前脚本默认读取：

```bash
inputs/cleft001.png
inputs/cright001.png
inputs/dleft001.npy
inputs/dright001.npy
inputs/left_mask_apple.png
inputs/right_mask_apple.png
inputs/left_mask_pear.png
inputs/right_mask_pear.png
inputs/left_mask_roboticarm.png
inputs/right_mask_roboticarm.png
```

深度优先使用 `.npy`。如果没有 `.npy`，脚本会尝试读取 `dleft001.png` / `dright001.png`，但该 PNG 必须是以毫米为单位的 uint16 深度图。普通可视化深度图通常只有 0-255，不能作为真实深度使用。

`--xml` 指向的 MuJoCo XML 必须就是生成这些虚拟图片时使用的源场景，至少要保证 `cam1`、`cam2` 的位置、姿态和 `fovy` 与图片采集时一致。

### 内部流程

`build_runtime_scene_from_sim_camera.py` 当前主要流程是：

1. 从 `--xml` 加载 MuJoCo 模型，读取 `cam1`、`cam2` 的相机位姿和 `fovy`。
2. 根据 MuJoCo 相机参数计算虚拟相机内参，不使用真实相机内参。
3. 根据 `inputs/` 中的 RGB、深度、mask 反投影出物体点云。
4. 用 mask 内的点云估计每个物体的三维位置。
5. 写出 `runtime_pose_calibration.json`。
6. 写出 `runtime_assets/sim_camera_intrinsics.json`，供 buquan 使用虚拟相机内参。
7. 调用 `resolve_meshes(..., camera_model="sim", sim_intrinsics_json=...)`，用模型识别和 buquan 流程生成 STL。
8. 估计物体姿态，并根据相机方向修正苹果、梨的朝向。
9. 调用 `dong2.generate_scene(...)` 生成基础 XML。
10. 调用 support height 修正，按照最终姿态和 STL 顶点把物体底部贴到支撑平面上。
11. 默认执行 refined pose，生成 `apple_pear_runtime_refined.xml`。

这条流程的关键点是：虚拟图片必须使用虚拟相机内参反投影，不能继续套用真实相机参数。否则会出现物体位置和原虚拟场景差很多的问题。

### 仿真套仿真的额外输出

除共同输出外，仿真套仿真还会生成：

```bash
runtime_assets/sim_camera_intrinsics.json
outputs/raw_left_apple.npy
outputs/raw_right_apple.npy
outputs/raw_left_pear.npy
outputs/raw_right_pear.npy
outputs/raw_left_roboticarm_world.npy
outputs/raw_right_roboticarm_world.npy
```

这些文件主要用于检查虚拟相机内参、物体点云和机械臂点云是否正确。

## 两条流程怎么选

如果图片来自真实相机，使用：

```bash
python zhenghe2_buquan.py
```

如果图片来自 MuJoCo 虚拟相机，使用：

```bash
python build_runtime_scene_from_sim_camera.py \
  --xml manipulator_grasp/assets/scenes/apple_pear_runtime_refined111_no_gripper.xml
```

不要混用两套相机模型：

- 真实转仿真使用真实相机标定和 `pointcloud_v2` 相关逻辑。
- 仿真套仿真使用 MuJoCo XML 里的相机位姿和 `fovy` 计算虚拟内参。

两条流程生成后的最终 XML 路径一致，所以后续 grasp、chat、异常处理的入口不需要换。

## 生成后如何检查

可以先检查最终 XML 里物体位置和姿态：

```bash
python - <<'PY'
import xml.etree.ElementTree as ET

p = "manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml"
root = ET.parse(p).getroot()
for name in ["apple0", "pear0"]:
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        print(name, "missing")
    else:
        print(name, "pos=", body.get("pos"), "quat=", body.get("quat"))
PY
```

如果 grasp/chat 服务已经在运行，重新生成 XML 后建议重启服务：

```bash
python grasp_fastapi_completion_v4.py
```

## 常见问题

### 仿真套仿真生成的位置明显不对

优先检查：

- `--xml` 是否就是生成 `inputs/` 图片的同一个 MuJoCo 场景。
- `cam1`、`cam2` 的位置、姿态、`fovy` 是否和采图时一致。
- 深度是否是 metric depth。`.npy` 最稳妥，0-255 的深度预览图不能用于反投影。
- mask 是否覆盖正确物体，特别是 `left_mask_apple.png`、`left_mask_pear.png`、右相机对应 mask。

### 物体识别不到

真实转仿真和仿真套仿真都仍然依赖模型识别，不直接读取 MuJoCo 物体真值。识别增强和调试输出见：

```bash
SIM_OBJECT_RECOGNITION_ENHANCEMENT.md
```

仿真套仿真中，buquan 会读取：

```bash
runtime_assets/sim_camera_intrinsics.json
```

因此如果识别框、mask 或补全结果不对，需要同时检查虚拟内参文件和识别调试图。

### 梨或苹果倒下

当前流程已经加入两类修正：

- `camera_facing_local_axis.py` 中设置物体局部 top 轴，让梨、苹果以正确方向面对相机。
- `runtime_support_height.py` 根据最终姿态和 STL 顶点修正 body 的 z，使物体底部落在支撑平面上。

如果重新生成 STL 或换了源图片后仍然倒下，先重新运行对应生成命令，再检查最终 XML 中 `pear0`、`apple0` 的 `quat` 和 `pos`。

### 服务没有使用最新场景

生成 XML 后，已经运行中的服务可能仍然持有旧环境。重启：

```bash
python grasp_fastapi_completion_v4.py
```

`build_runtime_scene_from_sim_camera.py` 也可以直接加 `--start-server`，生成完后启动服务。
