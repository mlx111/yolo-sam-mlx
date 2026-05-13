# Runtime Quat 获取说明

这份文档说明当前工程里苹果和梨子的 `quat` 是怎么得到的，以及运行哪个流程会落到哪个 XML。

## 1. 最终看哪个 XML

动态流程跑完后，最终应查看：

- `manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml`

不要只看中间文件：

- `manipulator_grasp/assets/scenes/apple_pear_runtime.xml`

原因是主流程 `zhenghe2_buquan.py` 在生成中间场景后，还会继续运行：

1. `left_view_pose_refiner.py`
2. `apply_refined_pose_to_scene.py`

最终姿态会写入 `apple_pear_runtime_refined.xml`。

## 2. 当前苹果和梨子的 quat 来源

### 苹果

苹果当前直接使用左视角 refine 的结果，不额外叠加 two-rotation 修正。

来源链：

1. `left_view_pose_refiner.py`
2. `runtime_assets/left_view_refined_pose.json`
3. `apply_refined_pose_to_scene.py` 直接把 apple 的 refined quat 写入最终 XML

当前最终 apple quat 例子：

```text
0.819000653 0.381991836 0.325066738 -0.278660696
```

### 梨子

梨子当前不直接使用 `left_view_refined_pose.json` 里的 refined quat。

梨子现在走的是一条单独的正向公式：

```text
pear_final_quat = pear_object_quat(v0_legacy) ⊗ pear_mesh_quat(two_rot)
```

也就是说，最终梨子姿态来自：

1. `object_pose_runtime.py` 中的 `pear_strategy="v0_legacy"`
2. `dynamic_two_rotation_scene_builder.py` 中的 `mesh_quat`
3. 两者在 `apply_refined_pose_to_scene.py` 里相乘

当前最终 pear quat 例子：

```text
-0.61757 0.707178 0.165614 -0.301792
```

注意：四元数整体乘以 `-1` 仍表示同一个旋转，所以：

```text
-0.61757 0.707178 0.165614 -0.301792
```

和

```text
0.617569925 -0.707178171 -0.165613852 0.301792106
```

是同一个姿态。

## 3. 梨子的正向公式

### 3.1 object_quat

梨子的 `object_quat` 现在由 `object_pose_runtime.py` 正向计算，不再依赖历史结果回放。

位置：

- `object_pose_runtime.py`

当前可选策略：

- `pear_strategy="best"`
- `pear_strategy="v0_legacy"`

当前主流程已经切到：

```text
pear_strategy="v0_legacy"
```

`v0_legacy` 的含义是：

```text
pear_object_quat(v0_legacy)
= pear 的 z_180 候选
  再绕物体局部 Z 轴补一个固定角度
```

固定角度是：

```text
-13.144693696620259°
```

因此它等价于：

```text
z_180 + local_z(-13.144693696620259°)
```

当前这一步得到的 pear object quat 为：

```text
[0.8506043269, -0.1749986964, 0.4024299037, 0.2896513557]
```

### 3.2 mesh_quat

梨子的 `mesh_quat` 来自 two-rotation 链：

```text
R_mesh = inverse(R1 @ R2)
```

其中：

- `R1`：补全过程里点云对齐到 `z` 轴的旋转
- `R2`：左相机旋转转换后的矩阵

代码入口：

- `dynamic_two_rotation_scene_builder.py`

当前实际使用：

```text
order = r1_r2
mesh_quat = inverse(R1 @ R2)
```

### 3.3 final_quat

场景里真正写入的是：

```text
final_quat = object_quat ⊗ mesh_quat
```

这个组合方式和 `dong2.py` 中的旋转组合一致。

## 4. 主流程里哪些脚本会产出这些 quat

### 4.1 主入口

主入口：

- `zhenghe2_buquan.py`

运行后会经历：

1. 构建运行时标定
2. 估计物体位置
3. 生成中间场景 `apple_pear_runtime.xml`
4. 运行左视角 refine
5. 运行 `apply_refined_pose_to_scene.py`
6. 产出最终场景 `apple_pear_runtime_refined.xml`

### 4.2 关键脚本

#### `object_pose_runtime.py`

作用：

- 生成 apple/pear 的 `object_quat`

当前和本问题直接相关的是：

- pear 支持 `pear_strategy="v0_legacy"`

#### `dynamic_two_rotation_scene_builder.py`

作用：

- 根据 `R1` 和 `R2` 生成 `mesh_quat`

当前梨子目标姿态依赖：

```text
order = r1_r2
pear_strategy = v0_legacy
```

#### `apply_refined_pose_to_scene.py`

作用：

- 将最终姿态写入 `apple_pear_runtime_refined.xml`

当前行为：

- apple：直接使用 `left_view_refined_pose.json` 里的 refined quat
- pear：覆盖为 `object_quat(v0_legacy) ⊗ mesh_quat(two_rot)`

## 5. 现在如何获取最终 quat

### 方式 1：跑完整动态流程

运行：

```bash
python zhenghe2_buquan.py
```

最终查看：

```text
manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml
```

### 方式 2：只重算 pear 的 two-rot 结果

运行：

```bash
python dynamic_two_rotation_scene_builder.py --camera left --objects pear --order r1_r2 --pear-strategy v0_legacy --scene-out manipulator_grasp/assets/scenes/pear_two_rot_v0_check.xml
```

查看：

```text
manipulator_grasp/assets/scenes/pear_two_rot_v0_check.xml
```

### 方式 3：从 refined 流程覆盖最终 XML

运行：

```bash
python apply_refined_pose_to_scene.py
```

查看：

```text
manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml
```

## 6. 当前接入点

已经接入 `pear_strategy="v0_legacy"` 的位置：

- `object_pose_runtime.py`
- `apply_refined_pose_to_scene.py`
- `left_view_pose_refiner.py`
- `zhenghe2_buquan.py`
- `dynamic_two_rotation_scene_builder.py`
- `march11_pose_replay.py`
- `left_view_orientation_replay.py`
- `legacy_camera_left_view_replay.py`

## 7. 排查要点

如果梨子的最终 quat 不对，优先检查这几项：

1. `runtime_assets/reports/` 下是否存在 pear 的 selected runner report
2. 对应 pipeline report 是否能读到 `rotation_to_z_3x3`
3. 最终查看的是不是 `apple_pear_runtime_refined.xml`
4. `pear_strategy` 是否仍然是 `v0_legacy`
5. 是否误把 `left_view_refined_pose.json` 里的 pear refined quat 当成最终结果

## 8. 一句话总结

当前工程里：

- apple 最终 quat 来自 left-view refined
- pear 最终 quat 来自 `v0_legacy object_quat + two-rotation mesh_quat`
- 最终统一看 `apple_pear_runtime_refined.xml`
