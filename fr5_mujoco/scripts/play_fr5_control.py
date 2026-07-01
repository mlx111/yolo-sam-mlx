from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fr5_mujoco import FR5MuJoCoController


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a MuJoCo viewer and run a simple FR5 control demo.")
    parser.add_argument(
        "--scene",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "scene.xml"),
        help="Path to FR5 MuJoCo scene XML.",
    )
    parser.add_argument("--seconds", type=float, default=12.0, help="Demo duration.")
    args = parser.parse_args()

    ctrl = FR5MuJoCoController.from_scene(args.scene)
    ctrl.reset_home()

    q_home = ctrl.qpos
    q_target = q_home + np.array([0.35, -0.25, 0.35, -0.2, 0.25, 0.2])

    with mujoco.viewer.launch_passive(ctrl.model, ctrl.data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.45]
        viewer.cam.distance = 1.6
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -25

        start = time.time()
        while viewer.is_running() and time.time() - start < args.seconds:
            t = time.time() - start
            phase = 0.5 - 0.5 * np.cos(2.0 * np.pi * min(t / args.seconds, 1.0))
            q_des = (1.0 - phase) * q_home + phase * q_target
            gripper = 0.04 if t < args.seconds * 0.5 else 0.0

            step_start = time.time()
            ctrl.step_joint_target(q_des, gripper_opening_m=gripper)
            viewer.sync()

            sleep_time = ctrl.model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
