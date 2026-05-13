# Pose Alignment Runtime Notes

## 1. 文档目的

这份文档总结当前仓库里和以下问题相关的完整历史与现状：

- 苹果、梨在 MuJoCo 场景中的位置与姿态
- 左右相机在 MuJoCo 场景中的位置与姿态
- `buquan` 补全 STL 与 `zhenghe2_buquan.py` 一键生成场景的串联关系
- 以后如果需要继续调位姿，应该改哪些文件、哪些常量

当前主入口是：

- [zhenghe2_buquan.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/zhenghe2_buquan.py)

当前主输出场景是：

- [apple_pear_runtime.xml](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/manipulator_grasp/assets/scenes/apple_pear_runtime.xml)


## 2. 当前一键链路

`zhenghe2_buquan.py` 现在会按下面的顺序运行：

1. `gen_mask(objects)`  
   生成苹果、梨、机械臂等对象的 mask。

2. `build_runtime_pose_calibration()`  
   通过 [calibrate_runtime_pose_from_clouds.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/calibrate_runtime_pose_from_clouds.py) 构造运行时标定结果，内部会：
   - 用 `right1_background.ply` 定义 MuJoCo 世界坐标系
   - 计算左右相机旋转矩阵与平移
   - 对 `cam1` 额外执行一次“机械臂端点 + 苹果/梨中心”的受限版 2D 对齐
   - 调用 `pointcloud_v2.pos()` 生成苹果和梨的相对位置

3. 将标定结果写到：
   - [runtime_pose_calibration.json](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/runtime_pose_calibration.json)

4. `resolve_meshes(mesh_source="buquan", camera="left")`  
   通过 `buquan` 生成补全 STL，并安装到：
   - `manipulator_grasp/assets/fruit/stl/apple.stl`
   - `manipulator_grasp/assets/fruit/stl/pear.stl`

5. `camera_poses_for_scene(calibration)`  
   从标定结果中提取 MuJoCo scene 直接可用的：
   - `cam1.pos`
   - `cam1.quat`
   - `cam2.pos`
   - `cam2.quat`

6. `estimate_runtime_object_quats(camera="left")`  
   动态计算苹果和梨的姿态四元数。

7. `dong2.generate_scene(...)`  
   用动态位置、动态相机位姿、动态物体姿态写出 XML。


## 3. 历史问题与修复过程

### 3.1 早期问题：场景还在用固定 STL

最开始运行时场景仍然引用旧的静态水果 STL，没有真正使用 `buquan` 补全后的网格。

后来的修复是：

- 在 `new_runtime/apple_pear_scene.py` 中把 `mesh_source="buquan"` 接通
- 将补全后的 STL 覆盖安装到标准目录：
  - `manipulator_grasp/assets/fruit/stl/apple.stl`
  - `manipulator_grasp/assets/fruit/stl/pear.stl`

这样 scene 不用改 mesh 路径，就能自动吃到补全后的水果网格。


### 3.2 位置严重错误：物体坐标只有 `10^-4`

早期 `pointcloud_v2.point()` 的返回值量级明显错误，例如：

- apple: `0.000573...`
- pear: `0.000410...`

后来定位到根因是：

- `generate_point_cloud()` 返回的坐标已经是米
- `point()` 末尾又额外除了一次 `1000`

修复后，`point()` 去掉了重复的 `/1000`，位置恢复正常量级，例如：

- apple: `0.573377...`
- pear: `0.410301...`

相关文件：

- [pointcloud_v2.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/pointcloud_v2.py)


### 3.3 相机位置错误：相机掉到地下

最初动态相机位置是直接取：

- `camera_pos = -point(..., "roboticarm")`

这样 `z` 也被取反了，导致 MuJoCo 中相机掉到地面以下。

后来的修复是：

- `x = -roboticarm_x`
- `y = -roboticarm_y`
- `z = +roboticarm_z`

也就是相机位置和物体位置的 `z` 处理方式不同。

当前实现位置：

- [pointcloud_v2.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/pointcloud_v2.py) 中的 `estimate_runtime_camera_poses()`


### 3.3.1 机械臂基座参考点后来又做了一次修正

一开始无论是旧方法还是新脚本，机械臂位置都默认取点云包围盒中心：

- `x = (x_min + x_max) / 2`
- `y = (y_min + y_max) / 2`
- `z = (z_min + z_max) / 2`

后来确认这不符合真实基座原点定义。  
现在统一改成：

- `x = (x_min + x_max) / 2`
- `y = (y_min + y_max) / 2`
- `z = z_min`

也就是：

- `x/y` 仍然取中心
- `z` 取机械臂基座点云的最底端

这个定义同时影响：

- 旧方法中的 `pos()`
- 新离线标定脚本中的机械臂参考点

所以当前凡是提到“机械臂基座位置”，都应理解为：

- `x/y` 中心
- `z` 最小值


### 3.4 相机旋转方向不对

相机位置修好后，渲染视角仍然不对。原因不是 `pointcloud_v2.py` 的原始相机姿态完全错，而是：

- `pointcloud_v2.py` 中的 `left/right` 角度来自 RANSAC / 点云处理
- 这些角度对应的是“原始相机姿态”
- 但 MuJoCo 相机坐标系和这套坐标系不一致

因此后来做了两层处理：

1. 保留 `pointcloud_v2.py` 中的原始相机欧拉角：
   - `left = [-17.08, -40.16, 96.59]`
   - `right = [-97.480497, -1.078239, -0.070489]`

2. 使用参考场景反推固定补偿矩阵，保存到：
   - [camera_pose_calibration.json](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/camera_pose_calibration.json)

后来这条链路又继续演化成当前的运行时方案：

1. 先用右侧全局点云定义 MuJoCo 世界系
2. 用原始 left/right 姿态得到左右相机初始旋转
3. 对 `cam1` 再做一次受限版 2D 对齐
4. 将对齐后的 `cam1` 写入：
   - [runtime_pose_calibration.json](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/runtime_pose_calibration.json)

历史工具：

- [camera_pose_mujoco.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/camera_pose_mujoco.py)

仍然保留在仓库里，但当前主运行时链路中真正生效的是：

- [calibrate_runtime_pose_from_clouds.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/calibrate_runtime_pose_from_clouds.py)


### 3.5 物体姿态错误：不能直接用 `buquan` 的对齐结果

最初曾尝试直接利用 `buquan` 的 `rotation_to_z_3x3` 来恢复苹果和梨的姿态，但最终确认这条路线不对。

原因是：

- `buquan` 会把点云先居中
- 再把选中的轴对齐到 `+Z`
- 这个旋转的目的是方便 `z-copy fill`
- 它不是“物体在真实场景中的最终姿态”

因此后来结论是：

- `buquan` 只负责 STL 补全
- 物体姿态必须回到原始点云来估计


### 3.6 物体姿态改为基于左相机原始点云

为了解决 `buquan` 坐标已被改写的问题，后来在 [pointcloud_v2.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/pointcloud_v2.py) 中新增了原始相机点云导出：

- `outputs/raw_left_apple.npy`
- `outputs/raw_left_pear.npy`

这些文件保存的是：

- 反投影后的左相机坐标系点云
- 尚未乘上 `camera_to_tcp_pose`
- 也未被 `buquan` 居中或对齐

后来 `object_pose_runtime.py` 就改成只读取这些原始左相机点云来估计物体姿态。


### 3.7 苹果基本正确，但梨偏了 90 度

在只使用原始左相机点云 PCA 之后：

- 苹果姿态已经比较接近真实图
- 梨仍然明显不对

最终定位到根因是：

- 梨的观测主轴本身没有大错
- 但梨 STL 本体的局部坐标轴和观测主轴不一致
- 相当于“STL 本体局部前向”与我们当前姿态估计默认的前向存在一个固定错位

最开始的修复是：

1. 读取 `raw_left_pear.npy`
2. 计算梨在左相机中的观测 PCA 主轴
3. 读取 `pear.stl`
4. 计算梨 STL 本体的 PCA 主轴
5. 先用 `observed_frame @ mesh_frame.T` 对齐 STL 主轴与观测主轴
6. 再对梨叠加一个局部 `+90°` 修正

但后来发现这仍然不稳，因为：

- 左相机局部点云只能给出几何主轴
- 梨 STL 本体坐标和观测主轴之间仍可能存在多个离散等价候选
- 单个固定 `+90°` 不能覆盖真实图片中的所有情况

所以当前版本已经升级为：

1. 左相机局部点云给出基础姿态 `R_base_left`
2. 左相机到 MuJoCo 的旋转矩阵来自：
   - `runtime_pose_calibration.json`
3. 在 `R_base_left` 上枚举少量离散候选：
   - `identity`
   - `±90°`
   - `180°`
4. 把每个候选都变到 MuJoCo 世界系
5. 再投影回左相机图像
6. 和真实左侧 mask 做评分
7. 选择分数最高的候选作为最终物体姿态

相关实现文件：

- [object_pose_runtime.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/object_pose_runtime.py)


## 4. 当前关键文件与职责

### 4.1 `zhenghe2_buquan.py`

主入口，一键生成场景。

负责：

- 调 `gen_mask`
- 调 `build_runtime_pose_calibration`
- 落盘 `runtime_pose_calibration.json`
- 调 `resolve_meshes`
- 调 `camera_poses_for_scene`
- 调 `estimate_runtime_object_quats`
- 调 `dong2.generate_scene`


### 4.2 `pointcloud_v2.py`

负责视觉侧的位置与原始点云输出。

当前和位姿最相关的部分有：

- `CAMERA_EULER_DEG`
- `point()`
- `pos()`
- `estimate_runtime_camera_poses()`

其中：

- `pos()` 负责苹果和梨的位置
- `estimate_runtime_camera_poses()` 负责 MuJoCo 中 `cam1/cam2` 的动态位置
- `point()` 在当前版本中还会保存：
  - `outputs/raw_left_apple.npy`
  - `outputs/raw_left_pear.npy`


### 4.3 `camera_pose_mujoco.py`

负责把“原始相机姿态”转换成 MuJoCo 相机姿态。

当前逻辑是：

- 优先读取 `camera_pose_calibration.json`
- 使用参考 scene 推导出的固定补偿矩阵
- 输出 MuJoCo 可直接写入 XML 的 `quat_wxyz`


### 4.4 `object_pose_runtime.py`

负责苹果和梨的动态姿态。

当前逻辑分两类：

- `apple`
  - 左相机原始点云 PCA
  - STL 本体 PCA
  - 候选局部修正较少
  - 当前通过左 mask 评分在少量离散候选中选择最佳姿态

- `pear`
  - 左相机原始点云 PCA
  - 梨 STL 本体 PCA
  - 枚举 `identity / ±90° / 180°` 等局部修正候选
  - 再结合：
    - 左相机真实 mask
    - 左相机到 MuJoCo 的旋转矩阵
    选出最终姿态

也就是说，当前 `object_pose_runtime.py` 已经不再是“固定一个手写修正矩阵”，而是“候选姿态 + 图像评分”的选择器。

同时要注意：

- `object_pose_runtime.py` 当前直接读取 [runtime_pose_calibration.json](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/runtime_pose_calibration.json) 里的左相机旋转矩阵和平移
- 所以物体姿态好坏已经直接受 `cam1` 对齐结果影响
- 当前不应再把“相机误差”和“物体误差”当成完全独立的问题


### 4.5 `dong2.py`

负责把所有动态结果真正写进 MuJoCo scene。

当前支持：

- `camera_poses`
- `object_quats`

如果 `object_quats` 不传，则会回退到旧的固定姿态。


## 5. 当前已经生效的关键数值

### 5.1 相机原始欧拉角

位置：

- [pointcloud_v2.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/pointcloud_v2.py)

当前值：

```python
CAMERA_EULER_DEG = {
    "left": [-17.08, -40.16, 96.59],
    "right": [-97.480497, -1.078239, -0.070489],
}
```


### 5.2 相机动态位置规则

位置：

- `estimate_runtime_camera_poses()` in `pointcloud_v2.py`

当前规则：

```python
cam_x = -roboticarm_x
cam_y = -roboticarm_y
cam_z = +roboticarm_z
```


### 5.3 当前物体姿态选择的关键信息

位置：

- [object_pose_runtime.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/object_pose_runtime.py)

当前核心输入：

- 左图：
  - `inputs/cleft001.png`
- 左 mask：
  - `inputs/left_mask_apple.png`
  - `inputs/left_mask_pear.png`
- 左原始局部点云：
  - `outputs/raw_left_apple.npy`
  - `outputs/raw_left_pear.npy`
- 当前运行时标定：
  - `runtime_pose_calibration.json`

当前候选局部修正集合：

```python
OBJECT_LOCAL_CANDIDATES = {
    "apple": [
        ("identity", ...),
        ("roll_180", ...),
    ],
    "pear": [
        ("identity", ...),
        ("z_pos_90", ...),
        ("z_neg_90", ...),
        ("z_180", ...),
        ("y_180", ...),
    ],
}
```

含义：

- 苹果只在少量候选中消除绕主轴歧义
- 梨允许在多个离散候选中选择
- 最终哪一个生效，不是写死的，而是由左 mask 评分选出来的

调试输出会写到：

- `outputs/object_pose_debug/`

包括：

- 每个候选的投影 mask
- 每个候选的得分
- 最终选中的候选对应的四元数


## 6. 如果以后要继续改位姿，应该改哪里

### 6.1 改相机位置

改这里：

- `estimate_runtime_camera_poses()` in [pointcloud_v2.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/pointcloud_v2.py)

适用场景：

- 相机高度不对
- 相机左右方向对
- 但位置整体偏了

优先修改：

- `cam_x / cam_y / cam_z` 的符号规则
- `roboticarm` 点云中心的定义


### 6.2 改相机旋转

改这里：

- [camera_pose_calibration.json](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/camera_pose_calibration.json)
- [camera_pose_mujoco.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/camera_pose_mujoco.py)

适用场景：

- 视角方向不对
- 相机位置已经对了
- 但 MuJoCo 看到的方向与真实图像不一致

优先修改：

- `refinement_euler_xyz_deg`
- 或重新用参考 scene 生成固定补偿矩阵


### 6.3 改苹果姿态

改这里：

- `_principal_frame(...)` in [object_pose_runtime.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/object_pose_runtime.py)

适用场景：

- 苹果倾斜方向不对
- 苹果沿主轴滚转方向不稳定

优先修改：

- 苹果 `major` 轴正负选择规则
- 是否需要给 `apple` 增加局部模板修正


### 6.4 改梨姿态

改这里：

- `OBJECT_LOCAL_CORRECTIONS["pear"]`
- `_estimate_object_rotation_in_left_camera(...)`
- `_principal_frame(...)`

适用场景：

- 梨仍然像是差一个 `90°`
- 梨的尖端/粗端方向反了
- 梨的长轴方向对了，但网格本体朝向还不对

当前最值得优先尝试的改法：

1. 看 `outputs/object_pose_debug/pear_candidates.json`
2. 检查当前被选中的候选是不是符合左图观感
3. 如果候选集合不够，再往 `OBJECT_LOCAL_CANDIDATES["pear"]` 里加新的离散修正
4. 如果候选都有问题，再改 `_principal_frame()` 里对 `pear` 的主轴符号规则

当前不再推荐的做法是：

- 直接盲改一个固定 `+90°` 或 `-90°`

因为当前系统已经支持按真实左图自动在多个候选中选分数最高的一组。


## 7. 当前推荐的调姿态顺序

以后继续调参时，推荐按下面顺序，不要反过来：

1. 先确认 STL 是否来自 `buquan`
2. 再确认物体位置 `pos` 是否正常
3. 再确认相机位置
4. 再确认相机旋转
5. 最后才调苹果和梨的局部姿态

原因是：

- 相机旋转错了，看起来像物体姿态错
- 物体位置错了，也容易误判为姿态错
- 只有在相机与位置都基本对齐后，物体 `quat` 的调整才有意义


## 8. 当前常用命令

### 8.1 一键生成完整运行时场景

```bash
cd /home/mlx/mujoco/YOLO_World-SAM-GraspNet
CONDA_NO_PLUGINS=true conda run -n mujoco1 python zhenghe2_buquan.py
```


### 8.2 只查看当前动态物体姿态

```bash
cd /home/mlx/mujoco/YOLO_World-SAM-GraspNet
CONDA_NO_PLUGINS=true conda run -n mujoco1 python -c "from object_pose_runtime import estimate_runtime_object_quats; print(estimate_runtime_object_quats())"
```


### 8.3 只查看当前动态相机位姿

```bash
cd /home/mlx/mujoco/YOLO_World-SAM-GraspNet
CONDA_NO_PLUGINS=true conda run -n mujoco1 python -c "from pointcloud_v2 import estimate_runtime_camera_poses; print(estimate_runtime_camera_poses())"
```


## 9. 当前状态总结

截至目前，系统状态是：

- `buquan` STL 补全：已接通
- 苹果、梨位置：已修复到正常量级
- 相机位置：已改为动态传入，且 `z` 不再取反
- 相机旋转：已通过参考场景补偿接通
- 苹果姿态：当前基本可用
- 梨姿态：已从“纯原始点云 PCA”升级到“观测轴 + STL 本体轴 + 局部 90° 修正”

如果后续你再看到梨方向还有问题，最优先要看的不是整条链路，而是：

- [object_pose_runtime.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/object_pose_runtime.py) 里的 `OBJECT_LOCAL_CORRECTIONS["pear"]`

这通常会是下一轮调参最直接的入口。


## 10. 新增的离线“全局+局部融合标定”工具

为了在**不修改当前运行时主链路**的前提下，额外得到：

- 左右相机坐标系到 MuJoCo 世界系的旋转矩阵
- 左右相机相对机械臂基座中心的位置
- 苹果、梨相对机械臂的更稳定位置

现在新增了一个独立脚本：

- [calibrate_runtime_pose_from_clouds.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/calibrate_runtime_pose_from_clouds.py)

这个脚本是**离线工具**，不会替代 `zhenghe2_buquan.py` 当前的一键运行逻辑。


### 10.1 这个新脚本解决什么问题

当前运行时主链路的问题是：

- 相机位置、物体位置、物体姿态，很多逻辑还混在一条链里
- `pointcloud_v2.py` 的主目标是“让抓取链路跑通”，不是“严格做统一外参标定”

因此新增这个脚本的目标是：

1. 用全局点云定义更清楚的 MuJoCo 世界坐标系
2. 用局部点云补足机械臂中心和物体中心的精细位置
3. 输出一份统一的离线标定结果 JSON

也就是：

- 全局点云：负责定坐标系、定相机旋转的大方向
- 局部点云：负责定机械臂与物体中心的位置


### 10.2 当前使用的输入

这个脚本当前会使用：

- 全局点云
  - `outputs/right1_background.ply`
  - `outputs/left1_background.ply`

- 局部原始点云
  - `outputs/raw_left_apple.npy`
  - `outputs/raw_left_pear.npy`
  - `outputs/raw_right_apple.npy`
  - `outputs/raw_right_pear.npy`

- 机械臂局部点云
  - 如果没有缓存文件，会直接临时从：
    - `inputs/cleft001.png`
    - `inputs/dleft001.png`
    - `inputs/left_mask_roboticarm.png`
    - `inputs/cright001.png`
    - `inputs/dright001.png`
    - `inputs/right_mask_roboticarm.png`
    重新生成机械臂局部相机系点云


### 10.3 新脚本如何定义世界坐标系

当前世界坐标系定义为：

- `Z` 轴：固定为机械臂竖直方向，也就是 MuJoCo 的 `[0, 0, 1]`
- `X` 轴：来自 `right1_background.ply` 中主参考平面的法向量，在水平面上的投影，并选择朝向物体一侧为正方向
- `Y = Z × X`

当前脚本里，世界系主要由右侧全局点云定义，因为：

- 右侧全局点云里的大平面更稳定
- 左侧全局点云结构更碎，更适合作为辅助，而不是主锚点


### 10.4 新脚本如何求相机位姿

#### 右相机

右相机是世界系锚点。

脚本会：

1. 从 `right1_background.ply` 提取主平面法向量
2. 构造当前世界系的 `X/Y/Z`
3. 得到：
   - `rotation_matrix_mj_from_cam`

右相机平移则通过机械臂基座中心约束得到：

- 先在右相机局部点云中找到机械臂中心
- 再令机械臂中心在世界系中作为原点
- 反推出右相机在世界系中的平移


#### 左相机

左相机不单独重新定义世界系，而是：

1. 先读取当前已有的左/右相机运行时旋转
2. 用右相机新世界系结果，反推出一个“旧世界 -> 新世界”的旋转
3. 把这个旋转同步作用到左相机上

这样做的原因是：

- 右侧全局点云更适合作为坐标系锚点
- 左侧全局点云质量较差，不适合单独重新建世界系
- 但左/右相机之间原来的相对关系仍然是有价值的


### 10.5 新脚本如何求机械臂与物体的相对位置

脚本当前采用的是：

- 相机旋转矩阵：来自新世界系构造
- 相对位置真值：直接复用旧 `pointcloud_v2.pos()` 的结果

这里要特别注意：

- 新脚本**不再**自己重新融合苹果/梨相对机械臂的位置
- 因为用户确认后，旧 `pos()` 更符合真实左相机图像中的相对关系
- 但旧 `pos()` 中的机械臂基座参考点已经同步修正为：
  - `x/y` 中心
  - `z = z_min`

计算流程是：

1. 通过新方法构造 MuJoCo 世界坐标系和左右相机旋转矩阵
2. 通过局部机械臂点云求相机平移，其中机械臂参考点定义为：
   - `x/y` 中心
   - `z = z_min`
3. 苹果和梨的相对位置不再由新脚本自行融合
4. 直接调用 `pointcloud_v2.pos(["apple", "pear"])`
5. 把这组结果写入：
   - `object_positions`
   - `relative_positions`

当前输出的相对位置就是：

- `apple_minus_arm`
- `pear_minus_arm`

当前它们本质上就是：

- 旧 `pointcloud_v2.pos()` 的输出
- 但其中机械臂基座参考点已经换成底端定义


### 10.6 新脚本的输出文件

当前默认输出：

- [runtime_pose_calibration.json](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/runtime_pose_calibration.json)

结构主要包括：

- `world_axes`
  - `x_axis`
  - `y_axis`
  - `z_axis`
  - `plane_normal_raw`
  - `plane_center_camera`
  - `plane_inlier_ratio`

- `camera_poses`
  - `left.rotation_matrix_mj_from_cam`
  - `left.translation_mj`
  - `left.roboticarm_center_camera`
  - `right.rotation_matrix_mj_from_cam`
  - `right.translation_mj`
  - `right.roboticarm_center_camera`

- `cam1_alignment`
  - 第一版 `cam1` 对齐器的优化结果
  - 包括：
    - 平移增量
    - 旋转增量
    - 投影点
    - 左图中的目标点

- `robot_positions`
  - `arm_base_world`

- `object_positions`
  - `apple_world`
  - `pear_world`

- `relative_positions`
  - `apple_minus_arm`
  - `pear_minus_arm`

- `relative_position_source`
  - 明确说明当前相对位置来自：
    - `pointcloud_v2.pos()`
    - 且机械臂基座参考点定义为 `x/y` 中心加 `z_min`


### 10.6.1 `cam1` 第一版对齐调试文件

当前 `calibrate_runtime_pose_from_clouds.py` 已经集成了一个“受限版 `cam1` 对齐器”，它只使用：

- 左图中的机械臂 mask
- 苹果 mask
- 梨 mask

当前第一版 `cam1` 对齐器的具体做法是：

1. 从 `left_mask_roboticarm.png` 提取：
   - 质心
   - 主轴
   - 主轴两端端点
2. 从 `left_mask_apple.png` 和 `left_mask_pear.png` 提取中心点
3. 构造 4 对 3D-2D 锚点：
   - 机械臂基座 `[0, 0, 0]`
   - 机械臂上方锚点 `[0, 0, 0.75]`
   - 苹果位置 `apple_world`
   - 梨位置 `pear_world`
4. 分别计算两类相机初值：
   - `rigid`
   - `pnp`
5. 先比较这两类初值的二维投影分数
6. 选更稳的一组，再对 `tx, ty, tz, rx, ry, rz` 做小范围细化
7. 如果当前主 scene 中已经存在苹果/梨姿态，则额外把水果 STL 投影成左图 mask，把 IoU 作为 `cam1` 的二次细化项

对应调试输出在：

- [outputs/cam1_alignment_debug/cam1_pose_alignment.json](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/cam1_alignment_debug/cam1_pose_alignment.json)
- [outputs/cam1_alignment_debug/cam1_alignment_overlay.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/cam1_alignment_debug/cam1_alignment_overlay.png)

这些输出的用途是：

- 看当前 `cam1` 优化器是否收敛
- 看投影点和真实左图中的目标点还差多少
- 判断当前偏差主要来自：
  - 相机位置
  - 相机旋转
  - 还是锚点定义不够好

当前 `cam1_pose_alignment.json` 里重点关注这些字段：

- `start_pose_source`
- `start_pose_score`
- `rigid_rotation_matrix` / `rigid_translation`
- `pnp_rotation_matrix` / `pnp_translation`
- `delta_translation`
- `delta_euler_xyz_deg`

含义是：

- `start_pose_source = rigid|pnp`
  表示最终细化是从哪一个初值开始的
- `rigid`
  表示旧的 3D-3D 刚体对齐初值
- `pnp`
  表示新的 3D-2D `solvePnP` 初值
- `used_scene_object_quats`
  表示这一轮 `cam1` 细化是否进一步使用了主 scene 里的苹果/梨姿态作为 mask IoU 约束

要注意：

- 这一版 `cam1` 对齐器还只是基础设施，不应直接当成最终真值
- 它目前主要用于提供：
  - 调整量
  - 调试图
  - 下一阶段继续增强评分函数的起点


### 10.7 如何运行新脚本

```bash
cd /home/mlx/mujoco/YOLO_World-SAM-GraspNet
CONDA_NO_PLUGINS=true conda run -n mujoco1 python calibrate_runtime_pose_from_clouds.py
```

如果想指定输出文件：

```bash
cd /home/mlx/mujoco/YOLO_World-SAM-GraspNet
CONDA_NO_PLUGINS=true conda run -n mujoco1 python calibrate_runtime_pose_from_clouds.py \
  --output /home/mlx/mujoco/YOLO_World-SAM-GraspNet/runtime_pose_calibration.json
```


### 10.8 这个离线工具和当前主链路的关系

这条新增离线工具链路现在已经正式接入运行时主链路：

- `zhenghe2_buquan.py` 会在每次生成 scene 前先运行它
- 标定结果会落盘到：
  - [runtime_pose_calibration.json](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/runtime_pose_calibration.json)
- 然后从这份标定结果中提取：
  - 相机位置
  - 相机四元数
  - 苹果/梨位置

当前职责分工是：

1. 运行时主链路  
   直接消费标定结果生成 MuJoCo scene。

2. 离线融合标定链路  
   负责提供：
   - 世界坐标系
   - 左右相机旋转矩阵
   - 左右相机相对机械臂的位置
   - 苹果/梨相对机械臂的位置（当前复用旧 `pos()` 真值）

当前仍然没有直接接回的部分主要是：

- `camera_pose_mujoco.py`

也就是说，运行时 scene 已经直接用新标定结果，但旧的相机姿态工具模块仍然保留在仓库中作为历史/辅助工具。
