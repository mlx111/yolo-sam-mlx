from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import mujoco.viewer

from skills.base.gripper_skill import load_skill


DEFAULT_MODEL = Path(__file__).with_name("model.xml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a MuJoCo viewer for the current robot model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the MuJoCo XML model. Defaults to ./model.xml",
    )
    parser.add_argument(
        "--gripper-open",
        type=float,
        default=0.0,
        help="Initial gripper command. 0.0 is open, 0.025 is closed.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run continuous dynamics simulation instead of holding a static preview pose.",
    )
    parser.add_argument(
        "--animate-gripper",
        action="store_true",
        help="Continuously cycle the gripper between open and closed.",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=3.0,
        help="Open-close cycle period in seconds when --animate-gripper is set.",
    )
    args = parser.parse_args()

    model_path = args.model.resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    gripper_skill = load_skill()
    gripper_skill.apply(model, data, args.gripper_open)

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 1.0)
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -18

        step_count = 0
        while viewer.is_running():
            if args.animate_gripper:
                t = step_count * model.opt.timestep
                phase = 0.5 * (1.0 + math.sin(2.0 * math.pi * t / args.period))
                command = 0.025 * phase
                gripper_skill.apply(model, data, command, direct_qpos=True)
                step_count += 1

            if args.simulate:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
