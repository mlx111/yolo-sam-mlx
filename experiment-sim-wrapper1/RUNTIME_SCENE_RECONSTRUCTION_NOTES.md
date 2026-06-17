# Runtime Scene Reconstruction Notes

本文档记录根目录 `zhenghe2_buquan.py` 的虚拟环境生成方式，以及当前
`experiment-sim-wrapper_1` 中 `sim_wrapper` 路径与它的差异。

## 结论

根目录的虚拟环境不是只靠一个目标位置 `recovery_pos` 临时摆放物体生成的。
它是一条完整的 runtime scene reconstruction 链路：

1. 真实 RGB-D 输入。
2. 目标分割 mask。
3. 左右相机点云。
4. 机械臂参考坐标。
5. 目标物体位置估计。
6. 相机位姿估计。
7. 物体姿态估计。
8. 点云补全/mesh 选择。
9. MuJoCo XML 生成。
10. 左视角 mask 对齐 refine。

因此，不能把根目录的虚拟环境生成逻辑等同于
`experiment-sim-wrapper_1/sim_wrapper.py` 当前的影子仿真逻辑。

## 根目录实现链路

入口文件：

```text
zhenghe2_buquan.py
```

主入口：

```python
build_runtime_scene(objects=None, scene_out=None, start_server=False, experience_lib_path=None)
```

核心步骤如下。

### 1. 真实图像分割

```python
gen_mask(objects, recognition_mode="real")
```

该步骤来自 `Grounded-SAM-2/cv_proc.py`，用于根据真实输入图像生成目标 mask。
后续点云估计会读取类似下面的文件：

```text
inputs/left_mask_apple.png
inputs/right_mask_apple.png
inputs/left_mask_pear.png
inputs/right_mask_pear.png
```

### 2. 背景点云生成

```python
_ensure_runtime_background_point_clouds()
```

该函数对左右相机分别调用：

```python
point(flag, rx, ry, rz, None)
```

输出背景点云：

```text
outputs/left1_background.ply
outputs/right1_background.ply
```

### 3. runtime scene 输入构建

```python
scene_inputs = build_original_runtime_scene_inputs(objects=objects)
```

实现文件：

```text
runtime_scene_original.py
```

该函数内部做两件关键事情：

```python
object_positions_raw = pos(object_names)
scene_camera_poses = estimate_runtime_camera_poses()
```

`pos(object_names)` 用左右相机估计 `roboticarm + objects` 的点云中心，然后以
`roboticarm` 为参考，将物体位置转换到机械臂相关坐标系，并对左右相机结果取平均。

`estimate_runtime_camera_poses()` 用左右相机看到的 `roboticarm` 位置反推 MuJoCo
场景中的 `cam1` 和 `cam2` 位姿。

### 4. 点云位置估计

实现文件：

```text
pointcloud_v2.py
```

核心函数：

```python
point(flag, rx, ry, rz, objects=None)
pos(objects)
estimate_runtime_camera_poses()
```

`point(...)` 读取：

```text
inputs/cleft001.png
inputs/dleft001.png
inputs/cright001.png
inputs/dright001.png
```

如果指定目标，则读取对应 mask：

```text
inputs/{left|right}_mask_{object}.png
```

然后根据相机内参、深度图、mask 和相机姿态生成局部或全局点云，并返回目标中心。

`pos(objects)` 的处理逻辑是：

1. 将 `roboticarm` 加入目标列表。
2. 左右相机分别估计 `roboticarm` 和每个物体位置。
3. 对每个物体减去同视角下的 `roboticarm` 位置。
4. 左右相机结果取平均。
5. 返回 `{object_name: [x, y, z]}`。

### 5. mesh 补全与选择

在 `zhenghe2_buquan.py` 中：

```python
resolve_meshes(mesh_source="buquan", camera="left", objects=target_objects)
```

实现文件：

```text
new_runtime/apple_pear_scene.py
```

该逻辑会优先复用 `buquan` 生成的 watertight/low-poly STL，并把可用 mesh 安装到：

```text
manipulator_grasp/assets/fruit/stl/
```

同时读取 runner report，用于决定 mesh 旋转恢复等信息。

### 6. 物体姿态与相机朝向对齐

在 `zhenghe2_buquan.py` 中：

```python
object_quats = estimate_runtime_object_quats(...)
object_quats, camera_facing_debug = align_object_quats_to_camera(...)
```

该部分根据 runtime calibration、点云/mesh 信息估计 apple/pear 的姿态，并对齐到
相机可见方向。

### 7. MuJoCo XML 生成

在 `zhenghe2_buquan.py` 中：

```python
out_path = generate_scene(
    result,
    camera_poses=camera_poses,
    object_quats=object_quats,
    mesh_quats=mesh_quats,
    scene_out=scene_out,
)
```

实现文件：

```text
dong2.py
```

`generate_scene(...)` 做的事情：

1. 读取模板 `manipulator_grasp/assets/scenes/scene2.xml`。
2. 根据 `camera_poses` 更新 `cam1/cam2` 等相机位姿。
3. 根据 `result` 添加 apple/pear 等物体 body。
4. 根据 `object_quats` 和 `mesh_quats` 设置物体姿态。
5. 编译 MuJoCo spec 检查合法性。
6. 写出 XML。

默认输出：

```text
manipulator_grasp/assets/scenes/apple_pear_runtime.xml
```

### 8. refine 后处理

在 `zhenghe2_buquan.py` 中：

```python
_build_refined_scene_xml()
```

内部运行：

```text
left_view_pose_refiner.py
apply_refined_pose_to_scene.py
```

最终输出：

```text
manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml
runtime_assets/left_view_refined_pose.json
```

这个 refined XML 才是根目录运行时最终使用的虚拟场景。

## 当前 experiment-sim-wrapper_1 的差异

当前 `experiment-sim-wrapper_1/sim_wrapper.py` 的逻辑更简单：

1. 固定读取 `apple_pear_runtime_refined.xml`。
2. 创建 MuJoCo model/data。
3. 根据传入的 `perceived_pos` 或 `recovery_pos` 移动 `apple0`。
4. 在这个影子场景中验证 LLM 给出的恢复动作。

它没有执行：

1. `gen_mask(...)`
2. `pointcloud_v2.pos(...)`
3. `estimate_runtime_camera_poses(...)`
4. `resolve_meshes(...)`
5. `generate_scene(...)`
6. `left_view_pose_refiner.py`
7. `apply_refined_pose_to_scene.py`

因此，当前 `sim_wrapper` 并不是根目录的 runtime scene reconstruction。

## 为什么会出现 no_recovery_target

原始 `experiment-sim-wrapper/run_experiment_v4.py` 中：

```python
_get_recovery_position()
```

如果没有 `metrics["perceived_position"]`，会直接回退到真值 apple 位置。

当前 `experiment-sim-wrapper_1` 为了避免隐藏真值增强 baseline，关闭了默认真值回退。
因此当没有显式感知结果时：

```text
recovery_pos = None
```

随后 `sim_wrapper` 由于缺少目标位置，无法构建恢复验证场景，于是得到：

```text
executed_plan_source = no_recovery_target
```

这不是因为 XML 里没有相机。两个场景 XML 都包含 `cam`、`cam1`、`cam2`、
`ee_camera`。问题是当前 `sim_wrapper` 没有使用这些相机重新执行根目录那套
runtime scene reconstruction。

## 后续改造方向

如果要让 `experiment-sim-wrapper_1` 使用根目录原本的虚拟环境生成方式，应当做
模块化接入，而不是在 runner 里恢复隐藏真值。

建议方案：

1. 把 `zhenghe2_buquan.py::build_runtime_scene(...)` 抽成稳定 API。
2. 增加参数控制，允许只生成 XML，不强制启动 `grasp_fastapi_completion_v4` server。
3. 让实验 runner 在需要 sim-wrapper 前调用 runtime scene reconstruction。
4. `sim_wrapper` 改为加载新生成的 refined XML，而不是只加载固定旧 XML。
5. 实验记录中保存 reconstruction artifacts，包括 object positions、camera poses、scene XML、refined pose JSON。
6. baseline 中不能隐藏调用真值位置；如果需要定位，必须通过显式技能或显式重建流程产生。

## 与异常恢复实验的关系

异常恢复实验需要区分两类能力：

1. 异常处理策略能力：LLM/经验库选择哪些技能、如何排序。
2. 场景重建能力：从真实/仿真 RGB-D 和 mask 生成可用于验证的虚拟环境。

当前 U3 实验失败在 sim-wrapper 路径上，主要不是恢复策略本身的问题，而是
`sim_wrapper` 没有接入 runtime scene reconstruction，且默认真值回退已被关闭。

后续论文实验如果要比较：

```text
direct_llm_weak
sim_only_weak
sim_memory_weak
hierarchical_memory_weak
```

则 sim 相关方法需要先明确虚拟环境生成来源：

1. 使用根目录 runtime scene reconstruction。
2. 或者明确声明使用显式感知技能产生的 `perceived_position` 构建影子场景。

不能混用“隐藏真值目标位置”和“经验库提升异常恢复能力”，否则会污染 baseline。

## 当前临时实验口径

完整 runtime scene reconstruction 管线较慢，短期会影响 U3 多条件、多方法、多 trial
实验。因此当前 `experiment-sim-wrapper_1` 先采用旧版快速口径：

```text
sim_wrapper 默认允许 ground-truth recovery position fallback
direct 默认不允许 ground-truth recovery position fallback
```

也就是说，当 `condition == "sim_wrapper"` 且异常后没有 `perceived_position` 时，
`run_experiment_v4.py::_get_recovery_position()` 会直接读取当前 MuJoCo 主场景中的
`apple0` 真值位置，并用它生成影子仿真环境。

该口径的目的只是先保证 sim 相关方法可运行：

```text
sim_only_weak
sim_memory_weak
hierarchical_memory_weak
```

实验结果中会记录：

```text
recovery_position_source = ground_truth_fallback
```

后续替换为正式重建管线时，应将该字段变为：

```text
recovery_position_source = perception
```

或记录更细的重建来源，例如：

```text
runtime_scene_reconstruction
```

论文写作时需要明确区分这两个阶段：

1. 临时快速实验：sim-wrapper 用真值目标位置生成虚拟恢复环境。
2. 正式实验版本：sim-wrapper 使用感知/重建管线生成虚拟环境。

## 当前 sim-wrapper 与经验库的关系

当前实验中，经验库只作为 LLM 决策上下文使用：

```text
sim_memory_weak
hierarchical_memory_weak
```

会按照 `condition_id` 检索经验，并把成功经验/失败经验写入 LLM prompt。

随后 `sim_wrapper` 只验证 LLM 在该 prompt 下生成的方案。也就是说：

```text
经验库 -> prompt 上下文 -> LLM 输出恢复方案 -> sim_wrapper 虚拟验证 -> 迁移执行
```

不会把经验库中的成功 `skill_sequence` 直接作为额外候选方案送入虚拟环境验证。
这样可以保证经验库的作用体现为对 LLM 规划的影响，而不是绕过 LLM 直接执行历史方案。

该机制不影响：

```text
direct_llm_weak
sim_only_weak
```

因为它们的 `memory_policy == none`。

因此可以区分：

1. 没有经验时：sim-wrapper 验证 LLM 直接生成的方案。
2. 有成功经验时：sim-wrapper 验证“带成功经验上下文”的 LLM 方案。
3. hierarchical 有失败经验时：失败经验作为 prompt 反例和 blocker 信息使用，不作为正向执行候选。
