#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation

from calibrate_runtime_pose_from_clouds import build_runtime_pose_calibration, object_positions_for_scene
from camera_pose_mujoco import convert_raw_rotation_to_mujoco, rotation_matrix_from_euler_xyz_deg
from dong2 import generate_scene
from object_pose_runtime import estimate_runtime_object_quats
from pointcloud_v2 import CAMERA_EULER_DEG


ROOT_DIR = Path(__file__).resolve().parent
RUNTIME_REPORT_DIR = ROOT_DIR / "runtime_assets" / "reports"
DEFAULT_SCENE_OUT = ROOT_DIR / "manipulator_grasp" / "assets" / "scenes" / "apple_pear_runtime_two_rot.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build runtime scene with mesh quats computed from two dynamic rotations."
    )
    parser.add_argument("--camera", choices=["left", "right"], default="left")
    parser.add_argument("--objects", default="apple,pear", help="Comma separated object names.")
    parser.add_argument("--order", choices=["r2_r1", "r1_r2"], default="r1_r2")
    parser.add_argument("--scene-out", default=str(DEFAULT_SCENE_OUT))
    parser.add_argument(
        "--use-object-quats",
        choices=["on", "off"],
        default="on",
        help="Use estimate_runtime_object_quats as object_quats when generating scene.",
    )
    parser.add_argument(
        "--pear-strategy",
        choices=["best", "v0_legacy"],
        default="v0_legacy",
        help="Pear object orientation strategy when --use-object-quats=on.",
    )
    return parser.parse_args()


def _parse_objects(text: str) -> list[str]:
    values = []
    for token in str(text).split(","):
        name = token.strip().lower()
        if not name:
            continue
        if name not in {"apple", "pear"}:
            raise ValueError(f"Unsupported object: {name}")
        if name not in values:
            values.append(name)
    if not values:
        raise ValueError("No valid objects.")
    return values


def _as_rot3x3(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"Rotation must be 3x3, got {arr.shape}.")
    return arr


def _matrix_to_quat_wxyz(rotation_matrix: np.ndarray) -> list[float]:
    quat_xyzw = Rotation.from_matrix(_as_rot3x3(rotation_matrix)).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def _selected_runner_report_path(object_name: str) -> Path:
    return RUNTIME_REPORT_DIR / f"{object_name}_selected_runner_report.json"


def _resolve_pipeline_report_from_runner_report(runner_report: Path) -> Path:
    payload = json.loads(runner_report.read_text(encoding="utf-8"))
    stage1 = payload.get("stage1", {})
    zcopy_report_raw = str(stage1.get("zcopy_report", "")).strip()

    candidates = []
    if zcopy_report_raw:
        zcopy_report = Path(zcopy_report_raw).expanduser()
        if zcopy_report.is_absolute():
            candidates.append(zcopy_report)
            candidates.append(runner_report.parent / zcopy_report.name)
        else:
            candidates.append((runner_report.parent / zcopy_report).resolve())

    object_name = str(payload.get("object", "")).strip().lower()
    if object_name:
        candidates.append(runner_report.parent / f"{object_name}_pipeline_report.json")
        candidates.append(
            ROOT_DIR
            / "buquan"
            / "outputs_pipeline_zcopy_to_stl_stereo_switch_combined_test"
            / f"{object_name}_pipeline_report.json"
        )
        for side in ("left", "right"):
            candidates.append(
                ROOT_DIR
                / "buquan"
                / "outputs_eval_lr_20260308"
                / f"{object_name}_{side}"
                / f"{object_name}_pipeline_report.json"
            )
        candidates.append(
            ROOT_DIR / "buquan" / "outputs_pipeline_zcopy_to_stl_final" / f"{object_name}_pipeline_report.json"
        )

    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Unable to resolve pipeline report from runner report: {runner_report}")


def _load_rotation_to_z(object_name: str) -> np.ndarray:
    runner_report = _selected_runner_report_path(object_name)
    if not runner_report.exists():
        raise FileNotFoundError(
            f"Missing selected runner report for {object_name}: {runner_report}. "
            "Run build_runtime_stl.py first."
        )
    pipeline_report = _resolve_pipeline_report_from_runner_report(runner_report)
    payload = json.loads(pipeline_report.read_text(encoding="utf-8"))
    geometry = payload.get("geometry", {})
    rotation_to_z = _as_rot3x3(geometry.get("rotation_to_z_3x3", []))
    return rotation_to_z


def _camera_rotation_for_position(camera: str) -> np.ndarray:
    rx, ry, rz = CAMERA_EULER_DEG[camera]
    pose = convert_raw_rotation_to_mujoco(rotation_matrix_from_euler_xyz_deg(rx, ry, rz), camera)
    return _as_rot3x3(pose.rotation_matrix)


def _mesh_quat_from_two_rotations(r1_align_to_z: np.ndarray, r2_for_position: np.ndarray, order: str) -> list[float]:
    if order == "r2_r1":
        r_total = r2_for_position @ r1_align_to_z
    elif order == "r1_r2":
        r_total = r1_align_to_z @ r2_for_position
    else:
        raise ValueError(f"Unsupported order: {order}")
    # Mesh in runtime is already rotated by r_total in reconstruction chain; use inverse to recover object frame.
    r_mesh = r_total.T
    return _matrix_to_quat_wxyz(r_mesh)


def build_mesh_quats(objects: list[str], camera: str, order: str) -> Dict[str, list[float]]:
    r2 = _camera_rotation_for_position(camera)
    out: Dict[str, list[float]] = {}
    for name in objects:
        r1 = _load_rotation_to_z(name)
        out[name] = _mesh_quat_from_two_rotations(r1, r2, order)
    return out


def main() -> None:
    args = parse_args()
    objects = _parse_objects(args.objects)

    calibration = build_runtime_pose_calibration()
    positions = object_positions_for_scene(calibration)
    scene_positions = {name: positions[name] for name in objects}

    mesh_quats = build_mesh_quats(objects=objects, camera=str(args.camera), order=str(args.order))
    object_quats = (
        estimate_runtime_object_quats(camera=str(args.camera), pear_strategy=str(args.pear_strategy))
        if args.use_object_quats == "on"
        else None
    )
    if object_quats is not None:
        object_quats = {name: object_quats[name] for name in objects}

    scene_out = Path(args.scene_out).resolve()
    scene_path = generate_scene(
        scene_positions,
        camera_poses=None,
        object_quats=object_quats,
        mesh_quats=mesh_quats,
        scene_out=scene_out,
    )

    payload = {
        "scene_out": str(scene_path),
        "camera": str(args.camera),
        "order": str(args.order),
        "objects": objects,
        "mesh_quats_wxyz": mesh_quats,
        "object_quats_wxyz": object_quats,
        "pear_strategy": str(args.pear_strategy),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
