# 虚拟场景物体识别增强说明

## 背景

当前抓取服务的 `/detect-object` 仍然保留视觉模型识别链路，不使用 MuJoCo 的真值 segmentation。输入图像来自 `/camera-image` 保存的 `color_img1.jpg`，分割结果写入 `inputs/mask1.png`，后续 `create-cloud -> create-grasp` 都依赖这个 mask。

这次问题表现为：虚拟场景图像中肉眼能看到苹果和梨，但 `/detect-object` 返回未识别到目标，`inputs/mask1.png` 是空 mask。

直接原因是旧的 `Grounded-SAM-2/cv_proc_sim.py` 在虚拟渲染图上只用单一文本提示和单个候选框策略，GroundingDINO 容易把目标框到地面反光、坐标轴或错误物体附近，SAM2 再基于错误框输出空 mask 或错误 mask。

## 增强思路

增强目标是：仍然使用视觉模型识别物体，但让模型在虚拟渲染图上更稳定。

核心变化有三点：

1. 多提示词识别

   不再只使用 `pear.` 或 `apple.`，而是给每类物体增加多个语义提示。例如：

   - `pear`
   - `yellow pear`
   - `green pear`
   - `fruit pear`
   - `apple`
   - `red apple`
   - `fruit apple`

   这样可以避免模型因为虚拟渲染材质、颜色或形状与真实图像分布不同而漏检。

2. 多候选框和多模型融合

   `cv_proc_sim.py` 现在会收集多个候选框：

   - GroundingDINO 候选
   - YOLO-World 候选，如果当前环境可用

   YOLO-World 不可用时会自动跳过，不影响 GroundingDINO 路径。

3. SAM mask 质量评分

   旧逻辑基本是选一个框后直接交给 SAM2。新逻辑会把多个候选框都交给 SAM2，并打开 `multimask_output=True`，再对所有 mask 评分。

   评分会考虑：

   - mask 是否为空
   - mask 面积是否合理
   - mask 是否覆盖目标颜色前景
   - mask 与候选框的填充比例
   - mask 是否过细长，避免选到红绿坐标轴
   - mask 是否过大，避免选到桌面或反光区域

   最终选择综合分数最高的 mask。

## 目标名规范化

`grasp_fastapi_completion_v4.py` 的 `/detect-object` 入口增加了目标名规范化，避免 chat 或人工输入中文时直接传给英文模型。

当前映射：

```text
苹果 -> apple
梨 / 梨子 -> pear
碗 -> bowl
```

因此下面两种调用都会走同一个识别目标：

```bash
curl -X POST "http://localhost:8080/detect-object?target_class=pear"
curl -X POST "http://localhost:8080/detect-object?target_class=梨"
```

## 改动文件

主要改动在：

```text
Grounded-SAM-2/cv_proc_sim.py
grasp_fastapi_completion_v4.py
```

`Grounded-SAM-2/cv_proc_sim.py` 负责虚拟图像的视觉模型识别和分割。

`grasp_fastapi_completion_v4.py` 负责 API 层目标名规范化，并调用：

```python
segment_image_ground("color_img1.jpg", normalized_target_class)
```

## 输出与调试文件

每次识别会生成：

```text
inputs/mask1.png
outputs/sim_groundingdino_mask1.png.jpg
outputs/sim_groundingdino_selected_mask1.png.jpg
```

含义：

```text
inputs/mask1.png
```

最终给点云和抓取使用的二值 mask。

```text
outputs/sim_groundingdino_mask1.png.jpg
```

候选框 debug 图，用于查看 GroundingDINO/YOLO-World 提出了哪些候选。

```text
outputs/sim_groundingdino_selected_mask1.png.jpg
```

最终 mask overlay 图，用于确认 mask 是否覆盖了正确物体。

如果识别失败，优先检查这三个文件。

## 生成虚拟环境

识别增强只解决 `/detect-object` 的视觉识别问题。使用它之前，需要先通过虚拟图像生成当前抓取服务要加载的 MuJoCo 场景。

当前虚拟图像默认来自：

```text
inputs/cleft001.png
inputs/cright001.png
inputs/dleft001.png 或 inputs/dleft001.npy
inputs/dright001.png 或 inputs/dright001.npy
inputs/left_mask_apple.png
inputs/right_mask_apple.png
inputs/left_mask_pear.png
inputs/right_mask_pear.png
```

生成虚拟环境使用：

```bash
python build_runtime_scene_from_sim_camera.py \
  --xml manipulator_grasp/assets/scenes/apple_pear_runtime_refined111_no_gripper.xml
```

这条命令会：

1. 从源 XML 读取 MuJoCo 相机参数。
2. 使用 `inputs/` 中的虚拟 RGB、深度和 mask 反投影点云。
3. 估计 apple/pear 的世界坐标。
4. 使用 `buquan` 生成并安装 `apple.stl`、`pear.stl`。
5. 写入运行时场景：

```text
manipulator_grasp/assets/scenes/apple_pear_runtime.xml
manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml
```

抓取服务实际加载的是：

```text
manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml
```

如果不想执行 refined pose，可以运行：

```bash
python build_runtime_scene_from_sim_camera.py \
  --xml manipulator_grasp/assets/scenes/apple_pear_runtime_refined111_no_gripper.xml \
  --no-refine
```

但正常抓取建议使用默认命令，也就是保留 refined pose。

生成完成后，重要输出包括：

```text
runtime_pose_calibration.json
runtime_assets/left_view_refined_pose.json
runtime_assets/sim_camera_intrinsics.json
manipulator_grasp/assets/fruit/stl/apple.stl
manipulator_grasp/assets/fruit/stl/pear.stl
manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml
```

注意：如果 `grasp_fastapi_completion_v4.py` 已经启动，重新生成 XML 后当前服务不会自动重载 MuJoCo model。需要重启服务，才能使用新场景。

## 验证方式

先重启服务，使新代码生效：

```bash
python grasp_fastapi_completion_v4.py
```

然后执行：

```bash
curl -X POST "http://localhost:8080/camera-image"
curl -X POST "http://localhost:8080/detect-object?target_class=pear"
```

检查返回是否为：

```json
{"status": "success"}
```

然后检查 mask 是否非空：

```bash
python - <<'PY'
import cv2
import numpy as np

mask = cv2.imread("inputs/mask1.png", cv2.IMREAD_GRAYSCALE)
print("mask_nonzero:", int(np.count_nonzero(mask)))
PY
```

苹果同理：

```bash
curl -X POST "http://localhost:8080/camera-image"
curl -X POST "http://localhost:8080/detect-object?target_class=apple"
```

这次在当前 `color_img1.jpg` 上的验证结果是：

```text
pear mask_nonzero = 3061
apple mask_nonzero = 2656
中文“梨” mask_nonzero = 3061
```

## 识别失败时的排查顺序

1. 看 `color_img1.jpg`

   确认目标物体是否真的在主相机 `cam1` 视野内。

2. 看 `inputs/mask1.png`

   如果非零像素为 0，说明识别/分割失败。

3. 看 `outputs/sim_groundingdino_mask1.png.jpg`

   如果候选框没有框到目标，说明检测模型或 prompt 需要调整。

4. 看 `outputs/sim_groundingdino_selected_mask1.png.jpg`

   如果候选框正确但 mask 错，说明 SAM2 mask 评分或前景先验需要调整。

5. 看服务控制台日志

   新逻辑会打印：

   ```text
   candidate_count
   grounding_dino_count
   yolo_world_count
   box_score
   mask_score
   area
   foreground_iou
   foreground_coverage
   bbox_fill
   ```

   这些信息可以判断是检测候选不够，还是 mask 评分选错。

## 可调参数

在 `Grounded-SAM-2/cv_proc_sim.py` 中可以优先调这些参数：

```python
MIN_MASK_AREA = 64
MAX_CANDIDATES_TO_SCORE = 16
```

如果目标很小但被过滤掉，可以降低 `MIN_MASK_AREA`。

如果候选很多但正确目标排在后面，可以提高 `MAX_CANDIDATES_TO_SCORE`。

如果颜色前景筛选不适合新物体，需要调整 `_foreground_prior(...)` 中针对不同类别的 HSV 范围。

## 重要边界

这套增强仍然是视觉模型识别，不使用 MuJoCo segmentation、geom id 或 body id。

它适合当前“虚拟图像输入，但希望保持模型识别流程”的需求。真实图片流程仍然可以继续走原来的 `recognition_mode="real"` 分支。
