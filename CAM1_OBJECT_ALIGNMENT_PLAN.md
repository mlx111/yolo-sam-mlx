# CAM1 And Object Alignment Plan

## 1. 文档目的

这份文档专门描述下一阶段的修正方案：

1. 先把 `cam1` 对齐到真实左相机图像 [cleft001.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/inputs/cleft001.png)
2. 再在对齐后的 `cam1` 下重新修正苹果和梨的物体位姿

这份文档不是当前运行时链路的概述，而是一个**后续实现方案文档**。  
它的目的只有一个：

- 把“左相机位姿误差”和“物体位姿误差”彻底拆开
- 给后续实现者一份足够详细、可以直接照着实现的说明

相关现有文档：

- 运行时整体链路说明：
  [POSE_ALIGNMENT_RUNTIME.md](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/POSE_ALIGNMENT_RUNTIME.md)


## 2. 当前问题的真实拆解

### 2.1 目前不是只有梨的姿态错

对比下面两张图：

- 当前仿真截图：
  [left.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/left.png)
- 真实左相机图：
  [cleft001.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/inputs/cleft001.png)

可以明确看出，当前误差至少有两层：

1. `cam1` 的位姿没有对齐真实左相机
2. 梨的物体姿态也不对

### 2.2 为什么必须先修 `cam1`

如果 `cam1` 本身不对，会出现这些问题：

- 机械臂在画面里的位置和透视关系不对
- 传送带上表面的可见面积不对
- 物体整体出现在错误的图像区域
- 梨看起来“不对”，但这部分“不对”可能是相机错造成的

因此如果继续直接调 `pear` 的 `quat`，会发生：

- 相机误差和物体误差混在一起
- 梨的评分函数会被错误视角污染
- 最终出现“数学上换了四元数，但视觉上没区别”的情况

### 2.3 当前结论

后续必须分成两个阶段：

1. **阶段一：左相机对齐**
2. **阶段二：在新 `cam1` 下重新估计苹果/梨位姿**

这两个阶段不能颠倒。


## 3. 当前已有输入与真值来源

### 3.1 左相机真实图像

当前左相机视觉真值是：

- [cleft001.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/inputs/cleft001.png)

它是后续所有相机与物体姿态对齐的主要视觉标准。

### 3.2 左相机真实 mask

当前可用左侧物体 mask：

- [left_mask_apple.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/inputs/left_mask_apple.png)
- [left_mask_pear.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/inputs/left_mask_pear.png)

这些 mask 当前已经被 `object_pose_runtime.py` 用于候选姿态评分。

### 3.3 左相机局部原始点云

当前可用左侧局部点云：

- [raw_left_apple.npy](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/raw_left_apple.npy)
- [raw_left_pear.npy](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/raw_left_pear.npy)

这类点云保存的是：

- 左相机原始坐标系下的局部点云
- 尚未乘相机到 MuJoCo 的旋转
- 也未被 `buquan` 居中和对齐

### 3.4 当前 scene 与当前标定

当前运行时 scene：

- [apple_pear_runtime.xml](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/manipulator_grasp/assets/scenes/apple_pear_runtime.xml)

当前运行时标定结果：

- [runtime_pose_calibration.json](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/runtime_pose_calibration.json)

其中已经包含：

- `cam1` 的 `rotation_matrix_mj_from_cam`
- `cam1` 的 `translation_mj`
- 苹果和梨的位置

### 3.5 当前问题定位用截图

当前用来说明问题的仿真图：

- [left.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/left.png)

它对应的是当前 scene 在左视角下的渲染结果。


## 4. 阶段一：左相机位姿对齐

这一阶段只解决一件事：

- 让 `cam1` 的构图、透视、机械臂位置，以及苹果/梨在画面中的布局尽可能接近 [cleft001.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/inputs/cleft001.png)

当前阶段一已经收缩为只关心：

- 机械臂
- 苹果
- 梨

不再把传送带、按钮盒、挡板当作硬约束，因为这些环境几何当前并不在运行时 scene 中。

### 4.1 阶段一的输入

固定输入如下：

- 真实左图：
  [cleft001.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/inputs/cleft001.png)
- 当前仿真图：
  [left.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/left.png)
- 当前 scene：
  [apple_pear_runtime.xml](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/manipulator_grasp/assets/scenes/apple_pear_runtime.xml)
- 当前标定：
  [runtime_pose_calibration.json](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/runtime_pose_calibration.json)

### 4.2 阶段一不关注什么

这一阶段**不把梨局部姿态当主目标**。

也就是说：

- 苹果和梨的 `quat` 不作为第一阶段主优化对象
- 物体只是辅助判断画面构图是否接近
- 不能因为梨当前姿态不完美，就把 `cam1` 往错误方向调

### 4.3 阶段一要对齐的视觉要素

必须对齐这些视觉元素：

1. 机械臂底座在画面中的位置
2. 机械臂主立柱的垂直方向和透视角度
3. 苹果在画面中的位置
4. 梨在画面中的位置
5. 苹果与梨的相对左右关系
6. 苹果/梨相对机械臂的整体布局区域

### 4.4 阶段一要优化的参数

阶段一只优化 `cam1`：

- 平移：
  - `tx`
  - `ty`
  - `tz`
- 旋转：
  - `rx`
  - `ry`
  - `rz`

或者等价表示为：

- `cam1.pos`
- `cam1.quat`

但实现时推荐内部仍然保留“增量欧拉角”表示，便于调试。

### 4.5 阶段一的粗调策略

先粗调，再细调。

#### 粗调起点

以 [runtime_pose_calibration.json](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/runtime_pose_calibration.json) 里的当前 `cam1` 为初值。

#### 粗调搜索范围

推荐范围：

- 平移：
  - `x/y/z` 各自 `±0.2m`
- 旋转：
  - `x/y/z` 各自 `±20°`

#### 粗调目标

粗调只要达到：

- 机械臂大致站到真实图中的位置
- 传送带透视方向接近
- 整体构图不再明显偏近景或偏俯视

### 4.6 阶段一的细调策略

在粗调结果附近继续做更小范围优化。

推荐范围：

- 平移：
  - `±0.05m`
- 旋转：
  - `±5°`

细调目标：

- 机械臂立柱方向、底座位置与真实图更贴近
- 传送带上表面可见区域更接近
- 物体整体落点和真实图更一致

### 4.7 阶段一的评分函数

这一阶段的评分函数必须以“相机构图”为主，不以水果局部姿态为主。

当前第一版实现采用了“端点 + 中心”的四点锚定策略。

二维观测量：

1. `left_mask_roboticarm.png` 沿主轴方向的基座端点
2. `left_mask_roboticarm.png` 沿主轴方向的顶部端点
3. `left_mask_apple.png` 的中心点
4. `left_mask_pear.png` 的中心点

对应三维锚点：

1. 机械臂基座点 `[0, 0, 0]`
2. 机械臂竖直方向上的上方点 `[0, 0, 0.75]`
3. 苹果世界位置 `apple_world`
4. 梨世界位置 `pear_world`

当前实现分两步：

1. `solvePnP` 先求一个 `cam1` 位姿初值
2. 再在该初值附近对 `tx, ty, tz, rx, ry, rz` 做小范围优化

为了避免 `solvePnP` 给出退化解，当前实现还同时保留了一组旧的刚体对齐初值：

- `pnp`
- `rigid`

两组初值都会先算分，再选投影有效且分数更低的一组作为真正的优化起点。

当前评分组成：

1. 机械臂基座端点偏差
2. 机械臂顶部端点偏差
3. 机械臂主轴方向一致性
4. 苹果中心偏差
5. 梨中心偏差
6. 苹果与梨的水平间距偏差
7. 小的位姿正则项

### 4.8 阶段一应输出的调试产物

实现时必须输出：

- `cam1_pose_alignment.json`
- `outputs/cam1_alignment_debug/`

建议包含：

- 每个候选 `cam1` 的参数
- 每个候选的评分
- 每个候选的渲染截图或投影图
- 最终选中的 `cam1 pos + quat`

当前已经实际输出：

- [cam1_pose_alignment.json](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/cam1_alignment_debug/cam1_pose_alignment.json)
- [cam1_alignment_overlay.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/cam1_alignment_debug/cam1_alignment_overlay.png)

其中 `cam1_pose_alignment.json` 额外记录：

- `start_pose_source`
- `start_pose_score`
- `rigid_rotation_matrix` / `rigid_translation`
- `pnp_rotation_matrix` / `pnp_translation`
- `delta_translation`
- `delta_euler_xyz_deg`
- 最终投影点与目标点

### 4.9 阶段一验收标准

视为通过的标准：

- [left.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/left.png) 与 [cleft001.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/inputs/cleft001.png) 构图接近
- 机械臂不再明显偏近景/偏中景
- 传送带面积、角度、透视关系接近
- 物体整体分布区接近


## 5. 阶段二：在对齐后的左相机下重做物体姿态

这一阶段只在新的 `cam1` 下进行。

### 5.1 阶段二输入

固定输入：

- 对齐后的 `cam1` 位姿
- 左真实图：
  [cleft001.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/inputs/cleft001.png)
- 左物体 mask：
  - [left_mask_apple.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/inputs/left_mask_apple.png)
  - [left_mask_pear.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/inputs/left_mask_pear.png)
- 左局部原始点云：
  - [raw_left_apple.npy](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/raw_left_apple.npy)
  - [raw_left_pear.npy](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/raw_left_pear.npy)
- 当前场景使用的 STL：
  - `apple.stl`
  - `pear.stl`

### 5.2 苹果姿态策略

苹果采取保守策略：

1. 左局部点云做 PCA
2. 得到苹果基础姿态
3. 只枚举少量绕主轴候选
4. 选一个不退化的结果即可

苹果不是当前主要问题，不要求第一版做到唯一真值。

### 5.3 梨姿态策略

梨采取强化策略：

1. 左局部点云做 PCA，得到观测主轴
2. 梨 STL 做 PCA，得到 mesh canonical frame
3. 先求基础对齐：

```text
R_base_left = observed_frame_left @ mesh_frame_local.T
```

4. 在基础姿态上枚举离散候选：
   - `identity`
   - `+90°`
   - `-90°`
   - `180°`
   - 必要时增加绕中轴的 `±90°/180°`

5. 对每个候选都做：

```text
R_candidate_world = R_mj_from_cam_left @ R_candidate_left
```

6. 再投影回左图，与真实左 mask 比较
7. 选得分最高的候选作为最终梨姿态

### 5.4 阶段二的评分函数

阶段二的评分函数不能再只看 mask IoU。

必须包含：

1. 投影 mask 与真实 mask 的 IoU
2. 长轴方向一致性
3. 梨尖端/粗端方向一致性
4. 梨相对苹果、机械臂底座的左右关系一致性
5. 中心位置偏差

推荐权重优先级：

1. 长轴方向
2. 尖端/粗端方向
3. mask IoU
4. 中心位置
5. 与其它物体相对关系

### 5.5 阶段二调试产物

实现时必须输出：

- `outputs/object_pose_debug_v2/`
- 每个候选姿态对应的：
  - 投影图
  - 得分 JSON
  - 最终四元数

### 5.6 阶段二验收标准

视为通过：

- 苹果姿态不比当前更差
- 梨不再出现当前“视觉上像差 90°，但评分却认为没问题”的情况
- `apple_pear_runtime.xml` 里实际写入了新的 `quat`


## 6. 当前代码接入点

后续实现时主要涉及这些模块：

### 6.1 `calibrate_runtime_pose_from_clouds.py`

用于第一阶段：

- 从当前标定结果出发
- 对齐 `cam1`
- 输出新 `cam1` 位姿

### 6.2 `object_pose_runtime.py`

用于第二阶段：

- 读取左局部点云
- 读取左 mask
- 在新 `cam1` 下重算苹果/梨姿态

### 6.3 `zhenghe2_buquan.py`

负责串联阶段一与阶段二：

1. 先生成或更新 `cam1`
2. 再计算物体姿态
3. 最后写 scene

### 6.4 `dong2.py`

不需要改变职责，只消费：

- `camera_poses`
- `object_quats`


## 7. 推荐实施顺序

### 7.1 先做的事

1. 保持当前 STL 链路不动
2. 保持当前物体位置链路不动
3. 先单独实现 `cam1` 对齐输出
4. 再基于新 `cam1` 重跑物体姿态选择

### 7.2 不要先做的事

不要先继续往 `pear` 的固定 `+90°/-90°` 上打补丁。  
因为当前主要误差源之一还是 `cam1` 本身。


## 8. 为什么这份文档必须独立存在

当前 [POSE_ALIGNMENT_RUNTIME.md](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/POSE_ALIGNMENT_RUNTIME.md) 主要记录的是：

- 当前运行时链路
- 历史问题
- 现有实现结果

但这份新文档关注的是：

- 下一阶段如何系统性修正 `cam1`
- 以及在新 `cam1` 下如何重修物体姿态

这两者不应该继续混在同一份文档里，否则后续实现者很容易把：

- 当前已生效逻辑
- 下一阶段计划逻辑

混为一谈。


## 9. 当前最终结论

当前最正确的推进方向不是继续单独调梨，而是：

1. 先让 `cam1` 对齐 [cleft001.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/inputs/cleft001.png)
2. 再在这个新的 `cam1` 下重新修苹果和梨的位姿

只有这样，后面修出来的梨姿态才不会继续被错误相机视角污染。


## 10. 当前已落地的第一版实现

这份文档对应的第一版实现已经接入到了标定脚本中，但当前还只是一个**受限版 `cam1` 对齐器**。

### 10.1 当前实现位置

当前第一阶段相机对齐逻辑在：

- [calibrate_runtime_pose_from_clouds.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/calibrate_runtime_pose_from_clouds.py)

当前实现内容是：

- 只使用左图中三类对象：
  - 机械臂
  - 苹果
  - 梨
- 不使用传送带、按钮盒、挡板等环境几何
- 从机械臂 mask 中提取：
  - 质心
  - 主轴
  - 基座端点
  - 顶部端点
- 先计算两类初值：
  - `rigid`
  - `pnp`
- 再选择更稳的一类初值，对 `cam1` 做小范围微调
- 如果当前主 scene 中已经存在苹果/梨的姿态：
  - 还会把苹果/梨 STL 投影成左图 mask
  - 以这些 mask 的 IoU 作为第二阶段的小范围相机细化约束

### 10.2 当前第一版使用的观测量

当前第一版 `cam1` 对齐器使用：

- `left_mask_roboticarm.png` 的：
  - 中心点
  - 主轴方向
  - 主轴两端端点
- `left_mask_apple.png` 的中心点
- `left_mask_pear.png` 的中心点

配合 MuJoCo 世界中的 3D 锚点：

- 机械臂底端附近点
- 机械臂竖直方向上的上方锚点
- 苹果位置
- 梨位置

### 10.3 当前调试输出

当前实现会输出：

- [outputs/cam1_alignment_debug/cam1_pose_alignment.json](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/cam1_alignment_debug/cam1_pose_alignment.json)
- [outputs/cam1_alignment_debug/cam1_alignment_overlay.png](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/cam1_alignment_debug/cam1_alignment_overlay.png)

这些文件的作用：

- `cam1_pose_alignment.json`
  - 记录本轮优化是否成功
  - 记录平移增量
  - 记录旋转增量
  - 记录投影点和真实目标点

- `cam1_alignment_overlay.png`
  - 把当前投影到真实左图上的 3D 锚点画出来
  - 便于人工直观看到 `cam1` 目前还差多少

### 10.4 当前第一版实现的局限

当前第一版已经能输出新的 `cam1` 候选位姿，但还没有达到最终可用状态。

当前已知局限：

1. 仍然只用了少量 2D 锚点，约束还不够强
2. 机械臂端点来自 mask 主轴投影，不一定与真实三维端点完全对应
3. `solvePnP` 在个别输入下仍可能给出退化解，所以必须保留 `rigid` 回退
4. 还没有加入环境结构约束
5. 当前评分函数仍偏粗，对苹果和梨的纵向位置敏感度不够
6. 即使加入苹果/梨投影 mask 的 IoU，当前结果也说明主要瓶颈仍然是相机 3D 锚点本身的定义，而不是缺少简单的轮廓重叠项

### 10.5 当前阶段的正确理解

因此，现在这套实现应该理解为：

- `cam1` 对齐的第一版基础设施已经接通
- 可以开始输出和分析 `cam1` 调整量
- 但还不能把它视为最终完成的 `cam1` 真值

也就是说：

- 它已经是一个可运行的起点
- 但后续还需要继续增强评分函数和约束
