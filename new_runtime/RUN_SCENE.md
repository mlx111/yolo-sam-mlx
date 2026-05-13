# Generate Apple-Pear Scene With `buquan`

This flow does two things before writing the runtime scene XML:

- generate fresh apple and pear STL files from `buquan`
- install those STL files into `manipulator_grasp/assets/fruit/stl/` using the standard names `apple.stl` and `pear.stl`

The generated scene still uses the `ur5e` robot from `scene2.xml`.

## 1. Enter the project

```bash
cd /home/mlx/mujoco/YOLO_World-SAM-GraspNet
```

## 2. Check the environment

If `conda` plugin loading is unstable on this machine, keep `CONDA_NO_PLUGINS=true`.

```bash
CONDA_NO_PLUGINS=true conda run -n mujoco1 python -V
```

## 3. Generate the scene with fresh completed STL files

This command:

- computes object positions from `pointcloud_v2.pos(['apple', 'pear'])`
- runs `buquan` with `left` camera first
- falls back to `right` camera only if `left` fails
- copies the selected completed STL files to:
  - `manipulator_grasp/assets/fruit/stl/apple.stl`
  - `manipulator_grasp/assets/fruit/stl/pear.stl`
- writes the runtime XML

```bash
CONDA_NO_PLUGINS=true conda run -n mujoco1 python -m new_runtime.build_apple_pear_scene \
  --mesh-source buquan \
  --camera left \
  --validate-load
```

## 4. Output files

Runtime scene XML:

```text
manipulator_grasp/assets/scenes/apple_pear_runtime.xml
```

Installed STL files used by the scene:

```text
manipulator_grasp/assets/fruit/stl/apple.stl
manipulator_grasp/assets/fruit/stl/pear.stl
```

Reports from the selected `buquan` runs:

```text
runtime_assets/reports/apple_selected_runner_report.json
runtime_assets/reports/pear_selected_runner_report.json
```

Per-camera intermediate outputs:

```text
runtime_assets/reports/apple_left/
runtime_assets/reports/apple_right/
runtime_assets/reports/pear_left/
runtime_assets/reports/pear_right/
```

## 5. Optional: override positions manually

If you do not want the automatic `pointcloud_v2.pos(...)` positions, pass explicit coordinates:

```bash
CONDA_NO_PLUGINS=true conda run -n mujoco1 python -m new_runtime.build_apple_pear_scene \
  --mesh-source buquan \
  --camera left \
  --apple-pos 0.01,0.02,0.03 \
  --pear-pos 0.04,0.05,0.06 \
  --validate-load
```

## 6. Optional: skip `buquan` and use current standard STL files

```bash
CONDA_NO_PLUGINS=true conda run -n mujoco1 python -m new_runtime.build_apple_pear_scene \
  --mesh-source fixed \
  --validate-load
```
