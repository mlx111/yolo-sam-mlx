from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fr5_mujoco import FR5MotionRuntime
from skills.field_atomic.atomic_executor import FR5FieldAtomicSkillExecutor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-class", default="apple")
    parser.add_argument("--device", default=None)
    parser.add_argument("--strict", action="store_true", help="exit non-zero when the target is not detected")
    args = parser.parse_args()

    runtime = FR5MotionRuntime.from_scene()
    runtime.reset_home()
    executor = FR5FieldAtomicSkillExecutor(runtime)
    steps = [
        {
            "action": "camera_rgbd_save",
            "parameters": {
                "camera_name": "ee_camera",
                "width": 640,
                "height": 480,
                "output_dir": str(Path(__file__).resolve().parents[1] / "output" / "camera_rgbd_pose"),
                "prefix": "pose",
            },
        },
        {
            "action": "detect_object_pose",
            "parameters": {
                "target_class": args.target_class,
                "device": args.device,
                "output_dir": str(Path(__file__).resolve().parents[1] / "output" / "grounded_sam2_pose"),
            },
        },
    ]
    results = executor.execute_plan(steps)
    for result in results:
        print(result)

    if results[0].success and results[1].success:
        grasp = executor.execute("create_fixed_vertical_grasp", {"target_class": args.target_class})
        print(grasp)
        if not grasp.success:
            raise SystemExit(1)
        return

    if args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
