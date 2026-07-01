from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fr5_mujoco import FR5MotionRuntime
from skills.field_atomic.atomic_executor import FR5FieldAtomicSkillExecutor


def main() -> None:
    runtime = FR5MotionRuntime.from_scene()
    runtime.reset_home()
    executor = FR5FieldAtomicSkillExecutor(runtime)
    result = executor.execute(
        "camera_rgbd_save",
        {
            "camera_name": "ee_camera",
            "width": 320,
            "height": 240,
            "output_dir": str(Path(__file__).resolve().parents[1] / "output" / "camera_rgbd_test"),
            "prefix": "smoke",
        },
    )
    print(result)
    if not result.success:
        raise SystemExit(1)

    camera = runtime.metrics["last_camera_rgbd"]
    for key in ("rgb_path", "depth_npy_path", "depth_png_path"):
        path = Path(camera[key])
        if not path.exists():
            raise SystemExit(f"missing output: {path}")
    depth = np.load(camera["depth_npy_path"])
    if depth.shape != (240, 320):
        raise SystemExit(f"unexpected depth shape: {depth.shape}")


if __name__ == "__main__":
    main()
