# Deployment Guide / 部署指南

This repository is a robotics workspace, not a single packaged app. For a second
machine, the most practical path is to recreate the environment, copy the runtime
assets, verify the main grasp service, and only then enable optional subprojects.

这个仓库不是单一应用，而是一个机器人工作区。迁移到另一台机器时，
最稳妥的做法是先重建环境、同步运行资产、验证主抓取服务，再决定是否启用
可选子项目。

1. Recreate the Python and system environment.
2. Copy the runtime assets and model weights.
3. Verify the main grasp service.
4. Only then decide whether to enable the optional subprojects.

## Recommended target / 推荐目标

- Linux
- NVIDIA GPU with working drivers
- Python 3.10
- MuJoCo-compatible OpenGL stack

建议目标机器满足以下条件：

- Linux
- 可正常工作的 NVIDIA GPU 驱动
- Python 3.10
- 支持 MuJoCo 渲染的 OpenGL 环境

## What must be present / 必备内容

For the main runtime path, the new machine needs:

- This repository
- `graspnet-baseline/`
- `Grounded-SAM-2/`
- `anygrasp_sdk/`
- `manipulator_grasp/`
- `runtime_pose_calibration.json`
- `camera_pose_calibration.json`
- `yolov8s-world.pth`
- `sam_b.pt`
- MuJoCo scene assets under `manipulator_grasp/assets/`

If you plan to use the chat / recovery flow, also install the LLM SDK
dependencies used by `doubao.py` and `exception_handling_agent.py`.

如果你还要使用聊天调度和异常恢复链路，还要额外准备 `doubao.py`
和 `exception_handling_agent.py` 依赖的 LLM SDK。

## Environment setup / 环境搭建

Create a clean environment first.

先创建一个干净环境。

```bash
conda create -n grasp-runtime python=3.10 -y
conda activate grasp-runtime
python -m pip install --upgrade pip
```

Install a PyTorch build that matches the target machine's CUDA driver.
After that, install the packages used by the main grasp stack:

先安装和目标机器 CUDA 驱动匹配的 PyTorch，然后再安装主抓取链路依赖：

```bash
pip install -r requirements.txt
```

If you need the simulation benchmark stack or training code, install those
subprojects separately. They are not required to bring up the grasp service.

如果你还需要仿真基准或训练代码，再单独安装对应子项目即可。
这些不是主抓取服务启动所必需的。

## Deployment order / 部署顺序

1. Copy the repository and assets to the new machine.
2. Run the environment check:

```bash
python tools/check_deploy_env.py
```

3. Rebuild calibration if the source machine state is not reused:

```bash
python calibrate_runtime_pose_from_clouds.py
```

4. Rebuild the runtime scene if the XML or meshes changed:

```bash
python build_runtime_scene_from_sim_camera.py \
  --xml manipulator_grasp/assets/scenes/apple_pear_runtime_refined111_no_gripper.xml
```

5. Start the grasp service:

```bash
python grasp_fastapi_completion_v4.py
```

The service listens on `0.0.0.0:8080` and uses a single worker because it keeps
global runtime state in memory.

服务监听 `0.0.0.0:8080`，并且只能单 worker 运行，因为它依赖进程内全局状态。

## Verification checklist / 验证清单

- `python tools/check_deploy_env.py` reports no missing required files or modules.
- `python grasp_fastapi_completion_v4.py` starts without import errors.
- `/camera-image` returns a valid RGB/depth capture.
- `/detect-object` can segment the target class.
- `/create-cloud` and `/create-grasp` complete successfully.
- `runtime_pose_calibration.json` and the scene XML paths match the new machine.

检查项：

- `python tools/check_deploy_env.py` 没有报缺失文件或模块。
- `python grasp_fastapi_completion_v4.py` 启动时没有导入错误。
- `/camera-image` 能正常返回 RGB/depth 图像。
- `/detect-object` 能正确分割目标类别。
- `/create-cloud` 和 `/create-grasp` 能顺利完成。
- `runtime_pose_calibration.json` 和场景 XML 路径在新机器上是正确的。

## Practical notes / 实用说明

- A unified Docker image does not exist in the repo yet.
- `Grounded-SAM-2` has its own Docker support, but the full workspace still
  benefits from a native conda install first.
- The LLM-driven recovery path is optional. If you only need deterministic
  grasping and scene reconstruction, you can leave that part disabled.

- 仓库里还没有统一的 Docker 镜像。
- `Grounded-SAM-2` 自带 Docker 支持，但整套工作区先用本地 conda 环境
  更容易把主链路跑通。
- LLM 驱动的异常恢复链路是可选的。如果你只需要确定性的抓取和场景重建，
  可以先不启用它。
