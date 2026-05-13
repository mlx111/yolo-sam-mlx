# 机械臂抓取失败原因分析：`cam1` 侧视与上方相机差异

## 1. 问题现象

你现在遇到的现象是：

- 运行 `main_yoloWorld_sam_pdf_cam.py`、`main_yoloWorld_sam.py` 等流程后，机械臂能够识别到目标，也能生成抓取位姿；
- 但真正执行抓取时，抓取成功率不高，经常出现夹空、偏抓、抓后滑落，或者末端姿态看起来对但实际没有夹稳；
- 当使用 `xml` 里的 `cam1` 作为观测相机时，抓取效果一般；
- 当换成机械臂上方的相机时，抓取成功率明显提升。

这个现象通常不是单一原因造成的，而是“相机视角、点云重建、抓取筛选、坐标系转换、执行动作”共同叠加后的结果。

---

## 2. 仓库里的实际链路

从当前仓库代码看，流程大致是：

1. 用 RGB-D 图像和 mask 生成点云；
2. 把点云喂给 GraspNet / AnyGrasp 类抓取网络；
3. 对网络输出进行碰撞过滤和角度筛选；
4. 选出一个 best grasp；
5. 将这个 grasp 从相机系转换到机械臂基座系；
6. 执行预抓取、下探、闭合、抬升。

对应关键位置如下：

- [main_yoloWorld_sam_pdf_cam.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/main_yoloWorld_sam_pdf_cam.py)
- [main_yoloWorld_sam.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/main_yoloWorld_sam.py)
- [calibrate_runtime_pose_from_clouds.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/calibrate_runtime_pose_from_clouds.py)
- [manipulator_grasp/assets/scenes/scene2.xml](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/manipulator_grasp/assets/scenes/scene2.xml)

这个链路本身没有明显断裂，但它对“相机视角是否近似俯视”和“相机系是否和世界竖直方向一致”非常敏感。

---

## 3. 为什么 `cam` 比 `cam1` 更容易成功

在 `scene2.xml` 里，两个相机的安装方式差异很大：

- `cam` 位于 `pos="0.05 0 1.2"`，朝向更接近俯视；
- `cam1` 位于 `pos="1.2 -0.94 0.27"`，是明显的侧视角。

参考：

- [scene2.xml](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/manipulator_grasp/assets/scenes/scene2.xml#L53)

这种差异会直接影响抓取效果，原因有几个。

### 3.1 俯视视角更接近“抓取先验”

很多抓取网络，包括当前这条链路里的筛选逻辑，默认倾向于寻找接近桌面法向的抓取，也就是“从上往下抓”或者“近似竖直下抓”。

当相机本身接近俯视时：

- 物体的顶部轮廓更完整；
- 深度图中的遮挡更少；
- 物体中心和姿态更容易估计；
- 抓取候选更容易和真实世界里的“下抓”一致。

当相机是侧视时：

- 物体高度方向被压缩到图像平面里；
- 遮挡更严重；
- 物体边缘、夹爪路径、物体高度这些信息更难稳定恢复；
- 网络输出的抓取方向更容易和真实可执行方向不一致。

### 3.2 侧视相机对深度误差更敏感

侧视时，深度噪声、边缘缺失、遮挡空洞会被放大成明显的 3D 位置误差。  
对抓取来说，这会导致：

- 抓取点偏离物体真实中心；
- 接近方向偏离；
- 预抓取路径穿过桌面、物体边缘或机械臂本体；
- 夹爪闭合时位置不在最佳夹持区域。

### 3.3 侧视角更容易破坏“垂直抓取”筛选

当前代码里有一个非常关键的经验假设：

- 把 `[0, 0, 1]` 当成“竖直方向”；
- 只保留接近该方向的抓取。

这个假设在俯视相机下近似成立，因为相机坐标和世界竖直方向更容易对齐；  
但在侧视相机下就不成立，因为此时相机的 `z` 轴不再等价于世界的“向上”方向。

也就是说，系统在侧视相机下可能筛掉了本来可执行的 grasp，或者保留下来的 grasp 虽然分数高，但方向并不适合真实抓取。

---

## 4. 代码里的核心问题

### 4.1 点云是在相机系里构建的

在 `get_and_process_data()` 里，代码通过相机内参和深度图生成点云：

- `CameraInfo(...)`
- `create_point_cloud_from_depth_image(...)`

这一步得到的点云本质上仍然在**相机坐标系**中。

相关代码位置：

- [main_yoloWorld_sam_pdf_cam.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/main_yoloWorld_sam_pdf_cam.py#L120)
- [main_yoloWorld_sam.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/main_yoloWorld_sam.py#L120)

这本身没错，但它意味着后续所有“竖直、桌面法向、抓取朝向”的判断都必须非常小心地做坐标系变换。

### 4.2 抓取筛选直接把 `[0,0,1]` 当成世界竖直

在两个 main 文件中，都有类似逻辑：

- `vertical = np.array([0, 0, 1])`
- 取 `grasp.rotation_matrix[:, 0]` 作为接近方向
- 用这个方向和 `vertical` 做夹角筛选

参考：

- [main_yoloWorld_sam_pdf_cam.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/main_yoloWorld_sam_pdf_cam.py#L328)
- [main_yoloWorld_sam.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/main_yoloWorld_sam.py#L327)

这里的问题是：

- `grasp.rotation_matrix[:, 0]` 是抓取在**当前坐标系**下的方向；
- `vertical` 却被写死成 `[0,0,1]`；
- 这个写法只有在“当前坐标系已经接近世界系竖直方向”时才合理。

对于上方相机，这个近似还能凑合用。  
对于 `cam1` 这种侧视相机，`[0,0,1]` 很可能只是“相机深度方向”，并不是“世界中的竖直方向”。

这会直接导致：

- 筛选逻辑不稳定；
- 得分最高的 grasp 可能方向上并不是最适合执行的；
- 机械臂在真实执行时会出现偏抓或抓空。

### 4.3 抓取执行仍然使用 `cam1` 作为世界变换基准

在执行阶段，代码直接取 `cam1` 的位姿：

- `cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam1")`
- 再从 `data.cam_xpos` 和 `data.cam_xmat` 构造 `T_wc`

参考：

- [main_yoloWorld_sam_pdf_cam.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/main_yoloWorld_sam_pdf_cam.py#L450)
- [main_yoloWorld_sam.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/main_yoloWorld_sam.py#L443)

然后再用：

- `T_wo = T_wc * T_co`

把抓取从相机系转到世界系。

问题在于：

- 这个过程假定 `cam1` 的外参已经足够准确；
- 也假定 `cam1` 的坐标轴和抓取网络输出的坐标轴是一致可解释的；
- 但实际上侧视相机对外参误差更敏感。

所以一旦 `cam1` 的位姿有一点偏差，最后末端抓取点和抓取姿态就会被整体放大误差。

### 4.4 标定链路本身就是围绕 `cam1` 做修正的

`calibrate_runtime_pose_from_clouds.py` 里已经写得很明确：系统会针对相机位姿、机械臂中心、背景平面做一套运行时标定，并且 `cam1` 还会走额外对齐逻辑。

参考：

- [calibrate_runtime_pose_from_clouds.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/calibrate_runtime_pose_from_clouds.py#L167)
- [calibrate_runtime_pose_from_clouds.py](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/calibrate_runtime_pose_from_clouds.py#L173)

这说明系统本身已经意识到：

- 相机外参不是简单“写死就能一直用”的；
- 尤其是 `cam1` 这种侧视相机，需要额外修正才能稳定工作。

---

## 5. 更深层的原因

### 5.1 侧视相机的几何信息不利于抓取

对于桌面抓取来说，最理想的观测通常是：

- 物体轮廓清晰；
- 遮挡少；
- 物体高度和朝向能稳定恢复；
- 抓取方向和可达路径容易规划。

上方相机满足这些条件得更好。  
侧视相机则会引入更多几何歧义：

- 物体前后关系更混乱；
- 表面法向估计更不稳定；
- 物体与桌面、机械臂之间的层次关系更难看清；
- 同一物体在图像里更容易被“压扁”。

### 5.2 模型的先验可能更偏向俯视抓取

当前筛选逻辑不是纯网络输出，而是叠加了“接近竖直方向”的规则。  
这类先验很容易天然偏向俯视场景。

如果训练数据、验证样本、场景经验大多来自上方或近俯视视角，那么模型也会更“习惯”这种视角。  
换成侧视后，网络和后处理都可能一起退化。

### 5.3 预抓取路径本身也可能更难走

抓取失败不一定发生在“抓取点”本身，有时失败发生在：

- 机械臂移动到预抓取位姿时发生轻微碰撞；
- 下探时路径经过桌面边缘；
- 夹爪闭合时物体已经有偏移；
- 抬升时因为接触点不佳而滑脱。

侧视相机下，这些问题会更加明显，因为抓取目标的深度和夹取空间更难估算准确。

---

## 6. 为什么这不是单纯“识别不准”

很多时候会直觉地认为：抓不稳就是检测框不准、mask 不准、或者网络没学好。  
但从当前代码看，问题更像是**几何链路不一致**，而不是纯粹识别错误。

原因是：

1. 即使检测到了目标；
2. 即使网络给出了 grasp；
3. 即使碰撞过滤和排序也正常；
4. 最终仍可能因为坐标系解释错误而抓偏。

也就是说，视觉识别只是前半段，真正决定成功率的，是“把视觉结果变成机械臂可执行动作”的几何链路。

---

## 7. 结论

综合来看，`cam1` 抓取效果不如上方相机，最可能的原因是：

1. `cam1` 是侧视相机，天然更容易带来遮挡、深度噪声和姿态歧义；
2. 当前抓取筛选把 `[0,0,1]` 当成竖直方向，但这对侧视相机并不成立；
3. 点云和 grasp 大多仍在相机系中处理，世界系和相机系之间的转换不够严格；
4. 执行阶段仍直接依赖 `cam1` 的外参，侧视相机下外参误差会被进一步放大；
5. 当前系统本身就属于“对 `cam1` 很敏感”的几何链路，换成上方相机后等于减少了坐标系失配带来的误差。

一句话概括：

> `cam1` 能看见目标，但它看到的是一个对抓取不友好的几何投影；上方相机能更接近抓取任务所需要的“真实竖直与真实可达性”，因此抓取成功率更高。

---

## 8. 建议的验证方式

如果你要继续排查，建议按下面顺序验证：

1. 分别统计 `cam1` 和上方相机下的抓取成功率、碰撞率、夹空率。
2. 把 `approach_dir` 先转换到世界系，再做竖直抓取筛选，看 `cam1` 的成功率是否明显提升。
3. 可视化同一个 grasp 在相机系、世界系下的方向差异，检查是否存在坐标轴误读。
4. 对比两个相机下点云的完整性，尤其是物体顶部、侧边和夹持区域。
5. 检查 `cam1` 的 XML 外参和运行时标定结果是否完全一致。

---

## 9. 建议的修复方向

如果后面要改代码，优先级建议如下：

1. 不要直接把 `[0,0,1]` 当成世界竖直，先把 grasp 方向变换到统一世界系。
2. 把“相机坐标系下的抓取排序”改成“世界坐标系下的抓取排序”。
3. 为 `cam1` 单独做严格外参验证，避免 XML、运行时标定、执行坐标系三者不一致。
4. 如果任务本质是桌面抓取，优先使用更接近俯视的相机作为主抓取视角。
5. 如果必须用侧视相机，则需要补充更完整的 3D 几何约束，而不是只靠当前的垂直方向筛选。
