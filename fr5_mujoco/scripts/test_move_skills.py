from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fr5_mujoco import FR5MotionRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FR5 movement skills without opening a viewer.")
    parser.add_argument(
        "--scene",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "scene.xml"),
        help="Path to FR5 MuJoCo scene XML.",
    )
    args = parser.parse_args()

    runtime = FR5MotionRuntime.from_scene(args.scene)
    runtime.reset_home()

    q_target = runtime.qpos + np.array([0.12, -0.08, 0.1, -0.06, 0.08, 0.05])
    joint_result = runtime.move_joints(q_target, duration=1.0)

    tcp_target = runtime.tcp_pos + np.array([0.0, 0.0, 0.03])
    cart_result = runtime.move_cartesian(tcp_target, duration=1.0)

    home_result = runtime.go_home(duration=1.0)

    for result in (joint_result, cart_result, home_result):
        print(result.to_dict())


if __name__ == "__main__":
    main()
