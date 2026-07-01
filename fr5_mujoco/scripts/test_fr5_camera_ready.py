from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fr5_mujoco import FR5MotionRuntime
from skills.field_atomic.atomic_executor import FR5FieldAtomicSkillExecutor


def main() -> None:
    runtime = FR5MotionRuntime.from_scene()
    runtime.reset_home()
    executor = FR5FieldAtomicSkillExecutor(runtime)
    results = executor.execute_plan(
        [
            {"action": "go_camera_ready", "parameters": {"duration": 1.5}},
            {
                "action": "camera_rgbd_save",
                "parameters": {
                    "camera_name": "ee_camera",
                    "width": 320,
                    "height": 240,
                    "output_dir": str(Path(__file__).resolve().parents[1] / "output" / "camera_ready"),
                    "prefix": "camera_ready",
                },
            },
        ]
    )
    for result in results:
        print(result)
    if any(not result.success for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
