from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrate_runtime_pose_from_clouds import (
    DEFAULT_JSON_PATH,
    build_runtime_pose_calibration,
    camera_poses_for_scene,
    object_positions_for_scene,
)
from dong2 import generate_scene
from object_pose_runtime import estimate_runtime_object_quats

from new_runtime.left_view_orientation_solver import solve_left_view_object_quats


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SCENE_OUT = ROOT_DIR / "manipulator_grasp" / "assets" / "scenes" / "left_view_orientation_replay.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit apple and pear orientation to the left camera masks while keeping camera pose and positions fixed."
    )
    parser.add_argument("--scene-out", default=str(DEFAULT_SCENE_OUT))
    parser.add_argument("--calibration-out", default=str(DEFAULT_JSON_PATH))
    parser.add_argument("--start-server", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_out = Path(args.scene_out).resolve()
    calibration_out = Path(args.calibration_out).resolve()

    print("[INFO] left-view replay: building runtime calibration", flush=True)
    calibration = build_runtime_pose_calibration()
    calibration_out.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] left-view replay: calibration -> {calibration_out}", flush=True)

    object_positions = object_positions_for_scene(calibration)
    camera_poses = camera_poses_for_scene(calibration)

    print("[INFO] left-view replay: estimating legacy initial object quaternions", flush=True)
    initial_quats = estimate_runtime_object_quats(
        camera="left",
        calibration_path=calibration_out,
        pear_strategy="v0_legacy",
    )
    print({"initial_object_quats": initial_quats})

    print("[INFO] left-view replay: optimizing object orientations against left masks", flush=True)
    object_quats = solve_left_view_object_quats(
        initial_object_quats=initial_quats,
        object_positions=object_positions,
        camera_poses=camera_poses,
    )
    print({"optimized_object_quats": object_quats})

    print("[INFO] left-view replay: exporting scene", flush=True)
    out_path = generate_scene(
        object_positions,
        camera_poses=camera_poses,
        object_quats=object_quats,
        scene_out=str(scene_out),
    )
    print(f"[INFO] left-view replay: scene -> {Path(out_path).resolve()}", flush=True)

    if args.start_server:
        print("[INFO] left-view replay: starting legacy grasp server", flush=True)
        from grasp_fastapi import start

        start()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
