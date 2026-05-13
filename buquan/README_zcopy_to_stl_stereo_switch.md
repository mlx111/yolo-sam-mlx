# 图片到 STL 一体化流程（Stereo Switch 复制版）

## 1. 说明
本文件对应新增复制版脚本，不改动原有代码与原有 README。

新增脚本：
- `pipeline_zcopy_to_stl_runner_stereo_switch.py`
- `pipeline_inputs_to_zcopy_robust_stereo_switch.py`
- `pipeline_inputs_to_zcopy_robust_stereo_switch_pear_ycanon.py`
- `zcopy_ply_to_stl_watertight_stereo_switch.py`

## 2. 新能力
支持通过 `--camera left|right` 选择使用左/右相机输入：
- `left`：读取 `--left-depth-rel` + `--left-mask-tpl`
- `right`：读取 `--right-depth-rel` + `--right-mask-tpl`

输出坐标系跟随所选相机。

对 `pear`，runner 默认会在 Stage1 自动将点云尖端规范到 `+Y`（可关闭）。

## 3. 一键运行（左相机）
```bash
python pipeline_zcopy_to_stl_runner_stereo_switch.py \
  --object pear \
  --camera left \
  --input-root inputs \
  --calib-pdf inputs/双ob\(1\)/双ob/标定.pdf \
  --output-dir outputs_pipeline_zcopy_to_stl_stereo_switch
```

## 4. 一键运行（右相机）
```bash
python pipeline_zcopy_to_stl_runner_stereo_switch.py \
  --object pear \
  --camera right \
  --input-root inputs \
  --calib-pdf inputs/双ob\(1\)/双ob/标定.pdf \
  --right-depth-rel dright001.png \
  --right-mask-tpl right_mask_{obj}.png \
  --output-dir outputs_pipeline_zcopy_to_stl_stereo_switch
```

## 5. 关键参数（新增）
- `--camera {left,right}`
- `--right-depth-rel`（默认 `dright001.png`）
- `--right-mask-tpl`（默认 `right_mask_{obj}.png`）
- `--pear-y-canonical {on,off}`（默认 `on`，仅 `pear` 生效）
- `--pear-tip-percentile`（默认 `0.10`）
- `--max-split-faces`（默认 `300000`，避免大网格 `split` 卡死）

其余参数与原 runner 保持一致。

## 6. 报告新增字段
在 `*_pipeline_report.json` 中新增：
- `inputs.camera_selected`
- `inputs.selected_depth`
- `inputs.selected_mask`
- `calibration.selected_intrinsics_name`
- `calibration.selected_intrinsics`
