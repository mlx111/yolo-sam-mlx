#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

try:
    import mujoco.viewer as mj_viewer
except Exception:  # pragma: no cover
    mj_viewer = None


ARM_ACTUATORS = ["j1_pos", "j2_pos", "j3_pos", "j4_pos", "j5_pos", "j6_pos"]
GRIPPER_ACTUATORS = ["fj1_pos", "fj2_pos"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_scene() -> Path:
    return repo_root() / "manipulator_grasp" / "assets" / "scenes" / "scene4_fr5v6.xml"


def clip_ctrl(model: mujoco.MjModel, ctrl: np.ndarray) -> np.ndarray:
    out = ctrl.copy()
    for i in range(model.nu):
        low, high = model.actuator_ctrlrange[i]
        if low < high:
            out[i] = np.clip(out[i], low, high)
    return out


def lookup_actuators(model: mujoco.MjModel) -> dict[str, int]:
    names = ARM_ACTUATORS + GRIPPER_ACTUATORS
    mapping: dict[str, int] = {}
    for name in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise ValueError(f"Actuator not found: {name}")
        mapping[name] = aid
    return mapping


def build_ctrl(
    model: mujoco.MjModel,
    aid: dict[str, int],
    q: np.ndarray,
    grip: float,
) -> np.ndarray:
    ctrl = np.zeros(model.nu, dtype=float)
    for i, name in enumerate(ARM_ACTUATORS):
        ctrl[aid[name]] = float(q[i])
    ctrl[aid["fj1_pos"]] = float(grip)
    ctrl[aid["fj2_pos"]] = float(grip)
    return clip_ctrl(model, ctrl)


def smooth_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer,
    target_ctrl: np.ndarray,
    duration_s: float,
    realtime: bool,
) -> None:
    dt = float(model.opt.timestep)
    steps = max(1, int(round(duration_s / dt)))
    start_ctrl = data.ctrl.copy()
    target_ctrl = clip_ctrl(model, target_ctrl)

    for k in range(steps):
        alpha = (k + 1) / steps
        s = 0.5 - 0.5 * np.cos(np.pi * alpha)
        data.ctrl[:] = (1.0 - s) * start_ctrl + s * target_ctrl

        tic = time.perf_counter()
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if realtime:
            delay = dt - (time.perf_counter() - tic)
            if delay > 0:
                time.sleep(delay)


def set_actuated_joint_qpos_from_ctrl(model: mujoco.MjModel, data: mujoco.MjData, ctrl: np.ndarray) -> None:
    # 只写入执行器直接驱动的滑动/转动关节，避免影响自由关节物体（如苹果/梨）。
    for aid in range(model.nu):
        jnt_id = int(model.actuator_trnid[aid, 0])
        if jnt_id < 0:
            continue
        jnt_type = int(model.jnt_type[jnt_id])
        if jnt_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            qadr = int(model.jnt_qposadr[jnt_id])
            data.qpos[qadr] = ctrl[aid]


def run(scene_path: Path, repeats: int, use_viewer: bool, realtime: bool) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    aid = lookup_actuators(model)

    print("Loaded scene:", scene_path)
    print("Arm actuators:", ARM_ACTUATORS)
    print("Gripper actuators:", GRIPPER_ACTUATORS)

    # 先恢复场景默认初始态（保留 free-joint 物体的初始位置），再单独设置机器人 home。
    mujoco.mj_resetData(model, data)
    home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key >= 0:
        home_ctrl = model.key_ctrl[home_key].copy()
    else:
        home_ctrl = np.zeros(model.nu, dtype=float)
    home_ctrl = clip_ctrl(model, home_ctrl)
    data.ctrl[:] = home_ctrl
    set_actuated_joint_qpos_from_ctrl(model, data, home_ctrl)
    mujoco.mj_forward(model, data)

    q_home = home_ctrl[[aid[n] for n in ARM_ACTUATORS]].copy()

    # 如果开合方向与你模型相反，交换下面两个值即可。
    grip_open = 0.038
    grip_close = 0.004

    q_pre_grasp = np.array([-1.20, -1.75, 1.95, -1.65, -1.40, 0.15], dtype=float)
    q_lift = np.array([-1.20, -1.45, 1.55, -1.65, -1.40, 0.15], dtype=float)
    q_place = np.array([-2.10, -1.55, 1.70, -1.55, -1.90, 0.35], dtype=float)

    sequence = [
        ("home", build_ctrl(model, aid, q_home, grip_open), 1.0),
        ("move_to_pre_grasp", build_ctrl(model, aid, q_pre_grasp, grip_open), 2.0),
        ("close_gripper", build_ctrl(model, aid, q_pre_grasp, grip_close), 1.2),
        ("lift", build_ctrl(model, aid, q_lift, grip_close), 1.5),
        ("move_to_place", build_ctrl(model, aid, q_place, grip_close), 2.0),
        ("open_gripper", build_ctrl(model, aid, q_place, grip_open), 1.2),
        ("back_home", build_ctrl(model, aid, q_home, grip_open), 2.0),
    ]

    def replay(viewer_handle) -> None:
        for _ in range(repeats):
            for label, target_ctrl, duration_s in sequence:
                print(f"[segment] {label:>18s}  duration={duration_s:.2f}s")
                smooth_step(model, data, viewer_handle, target_ctrl, duration_s, realtime)

    if use_viewer:
        if mj_viewer is None:
            raise RuntimeError("mujoco.viewer is unavailable. Use --no-viewer.")
        with mj_viewer.launch_passive(model, data) as viewer:
            replay(viewer)
    else:
        replay(None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control demo for scene4_fr5v6.xml (no 2f85)")
    parser.add_argument("--scene", type=str, default=str(default_scene()), help="Path to scene XML")
    parser.add_argument("--repeats", type=int, default=1, help="Replay count")
    parser.add_argument("--no-viewer", action="store_true", help="Run in headless mode")
    parser.add_argument("--no-realtime", action="store_true", help="Run faster than real-time")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = Path(args.scene).expanduser().resolve()
    if not scene.exists():
        raise FileNotFoundError(f"Scene file not found: {scene}")
    run(scene, repeats=max(1, args.repeats), use_viewer=not args.no_viewer, realtime=not args.no_realtime)


if __name__ == "__main__":
    main()
