# 图片到 STL 一体化流程说明

## 1. 目标
本流程用于把 `buquan/inputs` 中的图像与 mask，经过点云生成、去离群、Z 轴复制填充，最终转换为封闭 STL。

## 2. 核心脚本
- `buquan/pipeline_zcopy_to_stl_runner.py`
作用：一键串联整个流程（推荐只运行这个）。
- `buquan/pipeline_inputs_to_zcopy_robust.py`
作用：从图片生成点云并执行增强去离群 + 复制填充。
- `buquan/zcopy_ply_to_stl_watertight.py`
作用：把复制后的点云转成 watertight STL。

## 3. 输入文件
- 图像与 mask 目录：`buquan/inputs/`
- 标定文件：`buquan/inputs/双ob(1)/双ob/标定.pdf`
- 可选外参：`buquan/calib/refined_stereo_extrinsics.json`

默认按如下文件名读取：
- 深度图：`dleft001.png`
- mask 模板：`left_mask_{obj}.png`（例如 `left_mask_pear.png`）

## 4. 一键运行
```bash
python buquan/pipeline_zcopy_to_stl_runner.py \
  --object pear \
  --input-root buquan/inputs \
  --calib-pdf buquan/inputs/双ob\(1\)/双ob/标定.pdf \
  --output-dir buquan/outputs_pipeline_zcopy_to_stl_final
```

## 5. 输出文件
在 `--output-dir` 下会生成：
- `{object}_zcopy_filled_aligned.ply`：复制填充后的点云
- `{object}_watertight_voxel.stl`：最终 STL
- `{object}_runner_report.json`：串联执行报告
- `{object}_pipeline_report.json`：点云阶段报告
- `{object}_watertight_voxel_report.json`：网格阶段报告

## 6. 复制填充参数（最常调）
在 runner 上直接传这些参数即可：
- `--copy-times`：复制层数（越大越厚）
- `--copy-step-mm`：层间距 mm（越大延展越快）
- `--copy-direction`：`neg | pos | both`
- `--copy-source`：`cleaned_raw | processed | raw`（建议 `cleaned_raw`）
- `--copy-voxel-mm`：复制后体素去重，越小细节更多

示例（双向填充）：
```bash
python buquan/pipeline_zcopy_to_stl_runner.py \
  --object pear \
  --input-root buquan/inputs \
  --calib-pdf buquan/inputs/双ob\(1\)/双ob/标定.pdf \
  --copy-direction both \
  --copy-times 10 \
  --copy-step-mm 2.5 \
  --copy-source cleaned_raw \
  --output-dir buquan/outputs_pipeline_zcopy_to_stl_final
```

## 7. 去离群参数（抑制后侧尖刺）
- `--cluster-radius-mm`：最大连通簇半径
- `--tail-bin-mm`：主轴尾部裁剪分箱
- `--tail-min-support`：每箱最小支持点
- `--tail-max-remove-ratio`：尾部最多裁掉比例

建议起点：
- `--cluster-radius-mm 6.0`
- `--tail-min-support 12`
- `--tail-max-remove-ratio 0.05`

## 8. STL 网格参数
- `--mesh-voxel-mm`
- `--close-iters`
- `--open-iters`
- `--smooth-iters`
- `--smooth-lambda`
- `--target-faces`
- `--fallback convex_hull|none`

说明：
- 默认 `fallback=convex_hull`，优先保证输出是封闭实体。
- 若你更想保留细节，可尝试 `--fallback none`，但可能不封闭。

## 9. 常见问题
- 问题：后侧出现尖刺。
处理：优先加强去离群（第 7 节），并确认 `--copy-source cleaned_raw`。
- 问题：模型太薄。
处理：增大 `--copy-times` 或增大 `--copy-step-mm`。
- 问题：模型过于圆滑。
处理：减小 `--smooth-iters`，减小 `--copy-voxel-mm`，或 `--fallback none` 进行对比。

## 10. 依赖
需可用的 Python 包：
- `open3d`
- `numpy`
- `scipy`
- `opencv-python`
- `trimesh`

