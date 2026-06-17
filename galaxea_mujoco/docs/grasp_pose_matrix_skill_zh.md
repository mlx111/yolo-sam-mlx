# 抓取位姿矩阵生成技能说明

## 目标

新增的 `skills/base/grasp_pose_skill.py` 用来生成类似根目录机械臂代码中的抓取位姿矩阵：

```text
object pose / grasp proposal
-> T_grasp
-> T_pregrasp
-> grasp_position / pregrasp_position
-> executable_parameters
```

它目前只负责生成位姿和参数，不直接强制机械臂执行任意末端朝向。这样不会破坏当前已经稳定的 R1Pro 抓取、闭合、竖直提升流程。

## 为什么先这样做

根目录机械臂流程中，GraspNet/AnyGrasp 会输出 `translation + rotation_matrix`，然后构造 `T_wo`，再沿抓取坐标系退一段得到 `T_pregrasp`。R1Pro 现在的任意姿态 IK 还不稳定，因此第一版先把矩阵作为“计划层输出”和“现场调参接口”：

```text
T_grasp/T_pregrasp 用于描述抓取意图
grasp_position/pregrasp_position 用于调用当前稳定执行器
topdown_mode/control_frame 用于保持当前可靠抓取约束
```

## 当前输出

运行脚本：

```bash
conda run -n mujoco1 python source/run_grasp_pose_skill_smoke.py \
  --model r1pro_g3_sorting_scene.xml \
  --object-body target_cube \
  --side left \
  --grasp-mode topdown \
  --pregrasp-distance 0.06
```

输出字段：

- `grasp_matrix_4x4`：抓取目标齐次矩阵。
- `pregrasp_matrix_4x4`：预抓取齐次矩阵。
- `grasp_position`：当前稳定执行器可用的抓取点。
- `pregrasp_position`：当前稳定执行器可用的预抓取点。
- `approach_axis_world`：接近物体方向。
- `retreat_axis_world`：从抓取点退到预抓取点的方向。
- `grasp_width`：根据物体尺寸估计的夹爪开度。
- `executable_parameters`：可以交给现有抓取执行器的参数。

## 现场使用方式

现场 RGB-D/GraspNet 接入后，可以把真机感知得到的物体位置、尺寸、抓取旋转矩阵接入同一个接口。后续如果 R1Pro 的姿态 IK 调通，再把 `grasp_matrix_4x4` 中的旋转部分传给低层控制。

当前建议：

```text
先用矩阵生成 grasp/pregrasp 位置
再用现有稳定 IK 多段插值移动
再闭合夹爪
再保持抓取姿态做竖直提升
```

不要在现场第一版直接启用任意四元数姿态控制。

## 抓取工具坐标系

为了接近根目录机械臂中“工具坐标系”的用法，MuJoCo 模型中已经显式增加：

```text
left_grasp_tool
right_grasp_tool
```

第一版 `grasp_tool` 放在原来夹爪中心位置，表示两指之间的抓取中心。这样做不会改变现有抓取几何，只是把“抓取专用工具坐标系”固定下来。

从现在开始，抓取规划和执行的公开接口统一使用：

```text
control_frame=grasp_tool
```

`pinch` 已从 MuJoCo 模型中移除，避免和 `grasp_tool` 混用。`hand_tcp` 仍作为底层 IK/URDF 内部 frame 保留，不作为抓取规划接口使用。

后续 GraspNet/AnyGrasp 输出的抓取矩阵应理解为：

```text
grasp_matrix_4x4 = grasp_tool 在世界坐标系下的目标位姿
pregrasp_matrix_4x4 = grasp_tool 的预抓取位姿
```

现有 MuJoCo site servo 和抓取执行器已支持：

```text
control_frame=grasp_tool
```

如果姿态控制还不稳定，可以继续只使用矩阵中的位置；如果后续调通 6D 控制，再让 `grasp_tool` 对齐矩阵旋转。
