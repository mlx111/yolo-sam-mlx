#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrate_runtime_pose_from_clouds import build_runtime_pose_calibration, object_positions_for_scene
from dong2 import generate_scene


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT_DIR / "runtime_assets" / "two_rot_variants_results.json"
DEFAULT_SCENE_OUT = ROOT_DIR / "manipulator_grasp" / "assets" / "scenes" / "apple_pear_runtime_two_rot_replayed.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a historical two-rotation variant from runtime_assets/two_rot_variants_results.json."
    )
    parser.add_argument("--variant", default="v0_baseline_r1r2_inv")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--scene-out", default=str(DEFAULT_SCENE_OUT))
    return parser.parse_args()


def _load_variant(results_path: Path, variant_name: str) -> dict:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    for item in payload:
        if str(item.get("variant", "")).strip() == variant_name:
            return item
    available = [str(item.get("variant", "")) for item in payload]
    raise ValueError(f"Variant not found: {variant_name}. Available: {available}")


def main() -> None:
    args = parse_args()
    results_path = Path(args.results).expanduser().resolve()
    if not results_path.exists():
        raise FileNotFoundError(f"Missing variant results JSON: {results_path}")

    variant = _load_variant(results_path, str(args.variant))
    details = variant.get("details", {})
    object_quats = {
        object_name: pose["final_quat_wxyz"]
        for object_name, pose in details.items()
        if isinstance(pose, dict) and "final_quat_wxyz" in pose
    }
    if not object_quats:
        raise ValueError(f"No final quaternions found in variant: {args.variant}")

    calibration = build_runtime_pose_calibration()
    positions = object_positions_for_scene(calibration)
    scene_positions = {name: positions[name] for name in object_quats}

    scene_out = Path(args.scene_out).expanduser().resolve()
    scene_path = generate_scene(
        scene_positions,
        camera_poses=None,
        object_quats=object_quats,
        mesh_quats=None,
        scene_out=scene_out,
    )

    print(
        json.dumps(
            {
                "variant": str(args.variant),
                "results": str(results_path),
                "scene_out": str(scene_path),
                "replayed_object_quats_wxyz": object_quats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
