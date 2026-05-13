from __future__ import annotations

import argparse
from pathlib import Path

from dong2 import generate_scene
from object_pose_runtime import estimate_runtime_object_quats
from pointcloud_v2 import estimate_runtime_camera_poses, pos as estimate_positions

from new_runtime.left_view_orientation_solver import solve_left_view_object_quats


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SCENE_OUT = ROOT_DIR / "manipulator_grasp" / "assets" / "scenes" / "legacy_camera_left_view_replay.xml"
DEFAULT_OBJECTS = ["apple", "pear"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay left-view orientation fitting using the legacy camera pose chain from pointcloud_v2."
    )
    parser.add_argument("--scene-out", default=str(DEFAULT_SCENE_OUT))
    parser.add_argument("--start-server", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scene_out = Path(args.scene_out).resolve()

    print("[INFO] legacy-camera replay: estimating positions via pointcloud_v2.pos", flush=True)
    object_positions = estimate_positions(list(DEFAULT_OBJECTS))
    print({"object_positions": object_positions})

    print("[INFO] legacy-camera replay: estimating camera poses via pointcloud_v2.estimate_runtime_camera_poses", flush=True)
    camera_poses = estimate_runtime_camera_poses()
    print({"camera_poses": camera_poses})

    print("[INFO] legacy-camera replay: estimating legacy initial object quaternions", flush=True)
    initial_quats = estimate_runtime_object_quats(camera="left", pear_strategy="v0_legacy")
    print({"initial_object_quats": initial_quats})

    print("[INFO] legacy-camera replay: optimizing object orientations against left masks", flush=True)
    object_quats = solve_left_view_object_quats(
        initial_object_quats=initial_quats,
        object_positions=object_positions,
        camera_poses=camera_poses,
    )
    print({"optimized_object_quats": object_quats})

    print("[INFO] legacy-camera replay: exporting scene", flush=True)
    out_path = generate_scene(
        object_positions,
        camera_poses=camera_poses,
        object_quats=object_quats,
        scene_out=str(scene_out),
    )
    print(f"[INFO] legacy-camera replay: scene -> {Path(out_path).resolve()}", flush=True)

    if args.start_server:
        print("[INFO] legacy-camera replay: starting legacy grasp server", flush=True)
        from grasp_fastapi import start

        start()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
