# zhenghe2_buquan 当前物体位姿获取流程说明

本文档描述的是当前仓库里**实际生效**的 `zhenghe2_buquan.py` 流程，不包含之前已经试过但没有继续保留的方案。

## 1. 最终结论

当前这条链路里：

- 相机位姿：使用 `pointcloud_v2` 的原始获取方式。
- 物体位置 `pos`：使用 `pointcloud_v2.pos()` 的原始获取方式。
- 物体姿态 `quat`：先由左相机点云 + 左视图 mask 得到基础姿态，再做一层面向相机的后处理。
- 最终用于场景的 XML：`manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml`

如果只看“当前实际最终产物”，应该看：

- `manipulator_grasp/assets/scenes/apple_pear_runtime.xml`
- `manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml`
- `runtime_assets/left_view_refined_pose.json`

其中真正作为最终结果使用的是 `apple_pear_runtime_refined.xml`。

## 2. `zhenghe2_buquan.py` 当前主流程

入口文件：`zhenghe2_buquan.py`

当前主流程顺序如下：

1. `gen_mask(objects)`
   - 生成苹果和梨的分割 mask。

2. `_ensure_runtime_background_point_clouds()`
   - 如果缺少背景点云，则补生成：
   - `outputs/left1_background.ply`
   - `outputs/right1_background.ply`

3. `build_original_runtime_scene_inputs(objects=objects)`
   - 获取当前运行时使用的相机位姿、物体位置、校准信息。

4. 写出 runtime calibration JSON
   - 路径：`runtime_pose_calibration.json`

5. `resolve_meshes(mesh_source="buquan", camera="left")`
   - 跑/复用 `buquan` 补全流程。
   - 安装当前运行时使用的 STL。
   - 这一步当前使用的是**反旋转后的 STL**。

6. `estimate_runtime_object_quats(camera="left", ...)`
   - 先根据左相机原始点云和左视图 mask 估计基础物体姿态。

7. `align_object_quats_to_camera(...)`
   - 对上一步得到的基础 `quat` 再做一层当前项目里的面向相机修正。

8. `generate_scene(...)`
   - 生成中间场景 XML：
   - `manipulator_grasp/assets/scenes/apple_pear_runtime.xml`

9. `_build_refined_scene_xml()`
   - 先跑 `left_view_pose_refiner.py`
   - 再跑 `apply_refined_pose_to_scene.py`
   - 输出最终 refined XML：
   - `manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml`

## 3. 相机位姿是怎么获取的

相机位姿当前不是用新的右侧背景世界系那套实验链，而是恢复成了原始获取方式。

代码入口：`runtime_scene_original.py`

核心逻辑：

- `scene_camera_poses = estimate_runtime_camera_poses()`
- 这个函数来自 `pointcloud_v2.py`

也就是说，XML 中的：

- `cam1`
- `cam2`

当前都来自 `pointcloud_v2.estimate_runtime_camera_poses()`。

同时，`runtime_scene_original.py` 还会额外调用：

- `convert_raw_rotation_to_mujoco(...)`
- `rotation_matrix_from_euler_xyz_deg(...)`

把这套相机姿态整理进 calibration，供后面的姿态估计使用。

当前 calibration 里和相机相关的关键信息包括：

- `translation_mj`
- `quat_wxyz`
- `rotation_matrix_mj_from_cam`
- `rotation_matrix_world_from_cam`
- `point_transform_matrix`
- `rotation_transform_matrix`

## 4. 物体位置 `pos` 是怎么获取的

物体位置当前也不是新的重建位置方案，而是恢复成了原始获取方式。

代码入口：`runtime_scene_original.py`

核心逻辑：

- `object_positions_raw = pos(object_names)`
- 这个 `pos(...)` 来自 `pointcloud_v2.py`

当前 calibration 明确写着：

- `relative_position_source = pointcloud_v2.pos averaged left/right local centers`

也就是说，当前苹果和梨在 XML 里的位置，本质上来自：

- `pointcloud_v2.pos()`
- 它使用左右视角局部点云中心的原始方法得到物体相对位置

因此，当前流程里：

- 相机位姿是旧方法
- 物体位置也是旧方法

这两部分目前保持一致，没有再使用之前实验过的新世界系替换它们。

## 5. 物体基础 `quat` 是怎么获取的

代码入口：`object_pose_runtime.py`

当前基础姿态估计只支持左相机：

- `estimate_runtime_object_quats(camera="left", ...)`

流程如下。

### 5.1 读取左相机原始点云

对每个物体，读取：

- `outputs/raw_left_apple.npy`
- `outputs/raw_left_pear.npy`

这一步使用的是左相机的原始点云。

### 5.2 读取当前安装到场景里的 STL

对每个物体，读取：

- `manipulator_grasp/assets/fruit/stl/apple.stl`
- `manipulator_grasp/assets/fruit/stl/pear.stl`

注意：这里读取的是**当前场景真正使用的 STL**。目前这两个 STL 已经是 `buquan` 流程安装后的版本，并且走的是反旋转后的链路。

### 5.3 分别对观测点云和 mesh 做主轴分析

`object_pose_runtime.py` 会对两类数据分别构造主轴坐标系：

- 左相机下观测到的原始点云主轴
- 当前 STL mesh 的主轴

代码里对应：

- `_principal_frame(points, object_name, camera_observed=True/False)`
- `_base_rotation_left_camera(...)`

基础思想是：

- 先在观测点云里找出物体当前主轴方向
- 再在 mesh 自身坐标系里找出主轴方向
- 再把 mesh 的主轴对齐到观测点云主轴

于是得到一个基础旋转：

- `base_rotation_left_camera`

它表示的是：

- 当前 STL 在左相机坐标系下应该怎样旋转，才能和左视图观测更一致

### 5.4 用左视图 mask 在若干候选姿态里选一个基础结果

在基础旋转上，还会试若干离散候选修正。

当前候选集合：

- `apple`
  - `identity`
  - `roll_180`
- `pear`
  - `identity`
  - `z_pos_90`
  - `z_neg_90`
  - `z_180`
  - `y_180`
  - 以及一条历史保留的 `v0_legacy` 分支

代码里会：

- 用候选姿态把 mesh 投影到左图
- 与真实左视图 mask 比较
- 选出当前得分最合适的基础姿态

也就是说，`estimate_runtime_object_quats(...)` 输出的并不是完全手工指定的姿态，而是：

- 左相机点云主轴对齐
- 再加左图 mask 候选比较

得到的**基础 world quat**。

## 6. 当前最终 `quat` 还会再经过什么后处理

代码入口：`camera_facing_local_axis.py`

`estimate_runtime_object_quats(...)` 给出的只是基础姿态。当前实际写入 XML 的姿态，还会经过：

- `align_object_quats_to_camera(...)`

这一层后处理。

### 6.1 当前统一的局部轴定义

当前项目里这层后处理使用的是 STL 局部坐标系定义：

- 正面轴：局部 `-Z`
- 主转轴：局部 `+Y`

对应代码常量：

- `FRONT_AXIS_LOCAL_BY_OBJECT`
- `HINGE_AXIS_LOCAL_BY_OBJECT`

当前苹果和梨都采用：

- `front = local -Z`
- `hinge = local +Y`

### 6.2 当前梨的处理方式

梨当前使用的是：

- 自己算出来的 camera-facing 旋转角

也就是：

- 先根据当前 `quat`、物体位置、左相机位置
- 计算“局部 `-Z` 还差多少才能朝向相机”
- 再绕局部 `+Y` 对应到世界中的轴去旋转

### 6.3 当前苹果的处理方式

苹果当前不是完全按自己单独计算角度，而是使用了项目内的特殊规则：

1. 苹果先复用梨算出来的旋转角
2. 然后苹果再单独施加一个“顶部朝上”硬约束

当前苹果额外使用的顶部轴定义是：

- 局部 `-Y` 视为顶部轴

也就是说，苹果最终姿态当前满足的是：

- 正面仍按局部 `-Z`
- 主旋转仍按局部 `+Y`
- 但最终结果再强制让局部 `-Y` 尽量对齐世界 `+Z`

这也是为什么当前苹果和梨虽然都走 `align_object_quats_to_camera(...)`，但苹果有一层额外约束，梨没有。

## 7. refined 阶段做了什么

refined 阶段由两步组成：

1. `left_view_pose_refiner.py`
2. `apply_refined_pose_to_scene.py`

### 7.1 `left_view_pose_refiner.py`

它会：

- 重新读取原始 scene inputs
- 重新估计基础 `quat`
- 再走一遍 `align_object_quats_to_camera(...)`
- 跑 `solve_left_view_object_quats(...)`
- 然后再次经过 `align_object_quats_to_camera(...)`
- 最后把结果写到：
  - `runtime_assets/left_view_refined_pose.json`

当前这个 JSON 里保存了：

- `camera_poses`
- `initial`
- `refined`
- `camera_facing_local_axis`
- `position_refined = false`

这里要注意：

- 当前 refined 阶段并**不会修改物体位置**
- 当前 refined 阶段关注的是姿态和调试输出

### 7.2 `apply_refined_pose_to_scene.py`

这一步会把 `left_view_refined_pose.json` 里的结果写回到：

- `manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml`

所以当前真正最终使用的场景文件是：

- `apple_pear_runtime_refined.xml`

## 8. 当前 STL 是怎么来的

当前 `zhenghe2_buquan.py` 在生成场景前会调用：

- `resolve_meshes(mesh_source="buquan", camera="left")`

代码入口：`new_runtime/apple_pear_scene.py`

当前这条链会：

- 调 `buquan` 的点云补全和 watertight STL 生成流程
- 优先使用已经带反旋转结果的 runner 输出
- 最终安装到：
  - `manipulator_grasp/assets/fruit/stl/apple.stl`
  - `manipulator_grasp/assets/fruit/stl/pear.stl`

也就是说，当前场景里真正使用的 mesh：

- 已经不是补全过程中 PCA 对齐到 `+Z` 的姿态
- 而是**做过反旋转恢复后的 STL**

## 9. 能不能直接用 `zhenghe2_buquan.py` 获取目标环境

可以，但要准确理解“目标环境”指的是什么。

### 9.1 如果“目标环境”指的是当前 MuJoCo 场景文件和其运行时资源

那么答案是：**可以直接用 `zhenghe2_buquan.py` 获取。**

直接运行：

```bash
python zhenghe2_buquan.py
```

当前它会直接完成：

- mask 生成
- 背景点云补齐
- runtime calibration 写出
- `buquan` mesh 补全与 STL 安装
- 运行时场景 XML 生成
- refined pose JSON 生成
- 最终 refined XML 生成

关键输出包括：

- `runtime_pose_calibration.json`
- `runtime_assets/left_view_refined_pose.json`
- `manipulator_grasp/assets/scenes/apple_pear_runtime.xml`
- `manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml`
- `manipulator_grasp/assets/fruit/stl/apple.stl`
- `manipulator_grasp/assets/fruit/stl/pear.stl`

因此，如果你的意思是：

- “直接得到当前目标场景对应的 XML 和 mesh 环境”

那么答案就是：**能**。

### 9.2 如果“目标环境”指的是直接启动服务/直接进入运行状态

那要分两种情况：

- 默认运行 `python zhenghe2_buquan.py`
  - 会生成环境文件，但不会自动启动服务。
- 如果使用：

```bash
python zhenghe2_buquan.py --start-server
```

  - 则会在生成完场景之后调用 `start()` 启动服务。

所以更精确地说：

- `zhenghe2_buquan.py` 可以直接生成当前目标环境所需的 scene XML 和 mesh 资源。
- 如果还要顺便启动服务，则需要加 `--start-server`。

## 10. 当前实际生效的位姿链路总结

一句话总结当前版本：

- 相机 `pose`：`pointcloud_v2.estimate_runtime_camera_poses()`
- 物体 `pos`：`pointcloud_v2.pos()`
- 基础物体 `quat`：左相机原始点云主轴 + 当前 STL 主轴对齐 + 左视图 mask 候选筛选
- 最终物体 `quat`：再经过 `camera_facing_local_axis.py` 的项目内局部轴后处理
- 最终环境文件：`apple_pear_runtime_refined.xml`

如果后面还要继续调整苹果或梨朝向，当前最直接应该改的文件是：

- `object_pose_runtime.py`
- `camera_facing_local_axis.py`
- `left_view_pose_refiner.py`
