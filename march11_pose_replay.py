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


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SCENE_OUT = ROOT_DIR / "manipulator_grasp" / "assets" / "scenes" / "object_pose_dynamic_test_replay.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the March 11 style runtime flow: calibration -> legacy object pose -> scene export."
    )
    parser.add_argument(
        "--scene-out",
        default=str(DEFAULT_SCENE_OUT),
        help="Output XML path for the replay scene.",
    )
    parser.add_argument(
        "--calibration-out",
        default=str(DEFAULT_JSON_PATH),
        help="Path to write runtime calibration JSON.",
    )
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start the legacy grasp server after generating the scene.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_out = Path(args.scene_out).resolve()
    calibration_out = Path(args.calibration_out).resolve()

    print("[INFO] march11 replay: building runtime calibration", flush=True)
    calibration = build_runtime_pose_calibration()
    calibration_out.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] march11 replay: calibration -> {calibration_out}", flush=True)

    result = object_positions_for_scene(calibration)
    camera_poses = camera_poses_for_scene(calibration)

    print("[INFO] march11 replay: estimating object quaternions with legacy object_pose_runtime.py", flush=True)
    object_quats = estimate_runtime_object_quats(
        camera="left",
        calibration_path=calibration_out,
        pear_strategy="v0_legacy",
    )

    print({"object_positions": result})
    print({"camera_poses": camera_poses})
    print({"object_quats": object_quats})

    print("[INFO] march11 replay: exporting scene", flush=True)
    out_path = generate_scene(
        result,
        camera_poses=camera_poses,
        object_quats=object_quats,
        scene_out=str(scene_out),
    )
    print(f"[INFO] march11 replay: scene -> {Path(out_path).resolve()}", flush=True)

    if args.start_server:
        print("[INFO] march11 replay: starting legacy grasp server", flush=True)
        from grasp_fastapi import start

        start()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
