# Real2Sim Camera Intrinsics and Right-Camera Grasping Issues

## 1. 这份文档要解决什么

这份文档把两个经常被混在一起的问题放到一处说明：

- 如何让 MuJoCo 仿真相机的内参与真实相机尽量一致
- 为什么右侧相机更容易出现“机械臂能到附近，但抓不准”的问题

本仓库里当前已经存在几条相关链路：

- [main_yoloWorld_sam.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/main_yoloWorld_sam.py)
- [camera_pose_mujoco.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/camera_pose_mujoco.py)
- [calibrate_runtime_pose_from_clouds.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/calibrate_runtime_pose_from_clouds.py)
- [pointcloud_v2.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/pointcloud_v2.py)
- [POSE_ALIGNMENT_RUNTIME.md](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/POSE_ALIGNMENT_RUNTIME.md)
- [QUAT_RUNTIME_FLOW.md](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/QUAT_RUNTIME_FLOW.md)

其中：

- `main_yoloWorld_sam.py` 当前仍然是 `fovy` 简化相机路线
- `pointcloud_v2.py` 和 `calibrate_runtime_pose_from_clouds.py` 则显式使用真实标定式的 `fx, fy, cx, cy`

这意味着：如果你只改了 MuJoCo 的相机位姿，但没有把内参和投影模型统一起来，右侧相机下的抓取误差会被进一步放大。

## 2. 真实相机内参如何映射到 MuJoCo

### 2.1 真实相机内参的标准形式

真实相机通常用下面的 pinhole 模型表示：

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

其中：

- `fx, fy` 是以像素为单位的焦距
- `cx, cy` 是主点
- `width, height` 是图像分辨率

在本仓库中，这组参数已经被直接写进点云和投影逻辑里：

- [pointcloud_v2.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/pointcloud_v2.py)
- [calibrate_runtime_pose_from_clouds.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/calibrate_runtime_pose_from_clouds.py)

例如当前左相机使用的是：

```text
fx = 1129.8136
fy = 1128.6075
cx = 961.0022
cy = 546.8298
```

右相机使用的是：

```text
fx = 1126.8856
fy = 1126.4037
cx = 954.9412
cy = 536.3848
```

### 2.2 MuJoCo 里能对齐什么

MuJoCo 相机本质上是理想针孔模型，因此可以对齐的核心量是：

- 相机外参
- 图像分辨率
- 视场角或等价焦距
- 主点位置的近似

如果你的 MuJoCo 版本支持相机内参字段，可以把真实标定值更直接地写进去。  
如果当前只使用 `fovy`，那就只能做垂直视场角意义上的近似匹配。

### 2.3 如果只用 `fovy`，怎么从真实内参近似

如果仿真相机使用的是 `fovy`，那么一般按纵向焦距来近似：

```text
fy ≈ height / (2 * tan(fovy / 2))
fovy ≈ 2 * atan(height / (2 * fy))
```

这也是 [main_yoloWorld_sam.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/main_yoloWorld_sam.py) 里当前采用的思路：

```text
fovy = pi / 4
focal = height / (2 * tan(fovy / 2))
cx = width / 2
cy = height / 2
```

这种方式的优点是简单，缺点是：

- 只能精确拟合一个垂直视场角
- 默认 `fx = fy`
- 默认主点在图像中心
- 无法完整表达真实相机的非对称主点和轻微像素尺度差异

### 2.4 更推荐的做法

如果目标是 sim2real 抓取稳定，建议按下面顺序做：

1. 先从真实相机标定中保留 `fx, fy, cx, cy`
2. 在 MuJoCo 里把分辨率设成同样的 `width x height`
3. 再把相机外参对齐到真实安装位姿
4. 如果 MuJoCo 版本允许，优先用显式内参字段而不是只靠 `fovy`
5. 如果真实相机有明显畸变，先在真实图像上做去畸变，再和仿真图对齐

### 2.5 你需要记住的边界

即使内参对齐，MuJoCo 也仍然不会完全复现这些真实相机效应：

- 镜头畸变
- rolling shutter
- 传感器噪声
- 自动曝光和白平衡漂移
- 镜头微小装配偏差

所以这里的目标应该定义为：

- 投影关系一致
- 像素尺度一致
- 主点尽量一致
- 位姿反投影尽量一致

而不是“每一张图像都逐像素完全相同”。

## 3. 为什么右侧相机更容易抓偏

你前面提到的现象是：

- 机械臂上方相机拍摄的场景，抓取位姿更准确
- 右前方/右侧相机拍摄的场景，机械臂能到附近，但抓不准

这是典型的“视角更斜，误差被放大”的问题。

### 3.1 俯视相机更符合当前抓取假设

很多当前代码和抓取策略隐含的默认前提是：

- 物体主要从上方接近
- 抓取方向接近竖直方向
- 物体中心点可以近似作为抓取参考点

在这种前提下，上方相机天然更稳，因为：

- 物体轮廓更完整
- 遮挡更少
- 深度反投影更稳定
- 像素误差对世界坐标的放大较小

### 3.2 右侧相机的透视误差更大

右侧相机通常是斜视角，问题会被明显放大：

- 同样的像素误差会对应更大的空间误差
- 检测框中心不再等于真实几何中心
- SAM mask 的质心更容易偏向可见边缘
- 深度边界更容易被拉花

对于梨这种曲面物体，这种偏差尤其明显。

### 3.3 遮挡更严重

右侧相机更容易看到：

- 机械臂本体
- 夹爪
- 末端执行器阴影
- 目标和机械臂的重叠区域

这会导致：

- YOLO-World 检测框抖动
- SAM mask 不稳定
- 深度图局部污染
- 反投影点云偏移

上方相机通常更少被机械臂自遮挡，所以更容易得到稳定的目标中心和局部点云。

### 3.4 外参误差在右侧视角下更敏感

如果相机外参只差一点点：

- 俯视图里可能只是小偏差
- 右侧斜视图里就会变成明显的抓取偏移

这也是为什么同一套位姿链路，在上方相机下看起来“差不多能用”，到了右侧相机下就会暴露问题。

### 3.5 抓取姿态生成本身也更偏向俯视场景

你当前的抓取链路里有一条关键逻辑：

```text
T2 = T_wo * SE3(-0.1, 0, 0)
```

这等价于默认：

- `T_wo` 的局部 `x` 轴就是接近方向
- 预抓取位姿是沿局部 `x` 轴后退 10 cm

这个假设在俯视场景里往往还能凑合，但在右侧斜视场景里很容易失效，因为：

- 物体局部坐标系和真实可抓方向不一致
- mesh 原点和局部轴可能有固定偏置
- 代码把“抓取位姿”误当成了“物体位姿”来理解

### 3.6 梨比苹果更容易暴露这个问题

梨的几何特征通常比苹果更不对称、更依赖朝向：

- 细长
- 局部曲率变化大
- 正面/侧面差异明显

所以只要相机视角偏斜、姿态估计有一点偏差，梨就更容易抓偏。

## 4. 推荐的修正顺序

如果你要把右侧相机也调到可用，建议按下面顺序处理：

1. **先对齐内参**
   - 保证 `fx, fy, cx, cy`、分辨率和真实相机一致

2. **再对齐外参**
   - 确认 MuJoCo 相机位置和朝向和真实安装位姿一致

3. **检查深度与 RGB 对齐**
   - 不能只看 RGB 中的目标中心，必须检查深度反投影是否落在真实物体上

4. **检查 mask 质量**
   - 右侧相机更容易被遮挡，必须确认 SAM mask 没把背景或夹爪算进去

5. **重新定义抓取接近方向**
   - 不要固定依赖 `T_wo` 局部 `x` 轴
   - 对右侧相机单独验证接近方向是否合理

6. **补偿 TCP**
   - 确认夹爪工具中心点和末端法兰中心不重合时，已经做了几何补偿

## 5. 你当前仓库里最相关的文件

- [main_yoloWorld_sam.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/main_yoloWorld_sam.py)
  - 当前使用简化 `fovy` 推相机内参
  - 也是抓取位姿 `T_wo` 的来源

- [camera_pose_mujoco.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/camera_pose_mujoco.py)
  - 负责相机姿态从原始坐标系到 MuJoCo 坐标系的转换

- [calibrate_runtime_pose_from_clouds.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/calibrate_runtime_pose_from_clouds.py)
  - 显式保存并使用真实风格的 `fx, fy, cx, cy`

- [pointcloud_v2.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/pointcloud_v2.py)
  - 反投影和点云生成的核心，直接依赖真实内参

- [POSE_ALIGNMENT_RUNTIME.md](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/POSE_ALIGNMENT_RUNTIME.md)
  - 解释相机位姿和物体位姿的整体历史链路

- [QUAT_RUNTIME_FLOW.md](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/QUAT_RUNTIME_FLOW.md)
  - 解释梨子的最终 quat 为什么不是单纯的中间场景结果

## 6. 简短结论

可以把 MuJoCo 的相机内参调到和真实相机非常接近，但要注意：

- 这只能保证投影几何尽量一致
- 不能消除真实镜头畸变和噪声
- 右侧相机比俯视相机更容易放大外参误差、遮挡误差和抓取方向误差

所以如果右侧相机下抓取失败，不应该只怀疑机械臂控制，而要优先检查：

- 内参是否一致
- 外参是否一致
- 深度是否对齐
- 抓取局部坐标系是否和物体模型一致
