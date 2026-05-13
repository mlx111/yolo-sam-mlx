#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from runtime_scene_original import build_original_runtime_scene_inputs
from new_runtime.left_view_orientation_solver import (
    _load_mask,
    _load_mesh,
    _matrix_to_quat_wxyz,
    _quat_wxyz_to_matrix,
    _render_candidate_mask,
    _score_rendered_mask,
    solve_left_view_object_quats,
)
from object_pose_runtime import estimate_runtime_object_quats
from camera_facing_local_axis import align_object_quats_to_camera


ROOT_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = ROOT_DIR / "runtime_assets"
CALIBRATION_OUT = RUNTIME_DIR / "left_view_refiner_runtime_pose_calibration.json"
POSE_OUT = RUNTIME_DIR / "left_view_refined_pose.json"
DEBUG_DIR = RUNTIME_DIR / "left_view_refiner_debug"
SUPPORTED_OBJECTS = ("apple", "pear")


def _ensure_dirs() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def _save_overlay(object_name: str, tag: str, rendered: np.ndarray, gt_mask: np.ndarray) -> None:
    rendered_u8 = (rendered > 0).astype(np.uint8) * 255
    gt_u8 = (gt_mask > 0).astype(np.uint8) * 255
    overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    overlay[..., 1] = gt_u8
    overlay[..., 2] = rendered_u8
    cv2.imwrite(str(DEBUG_DIR / f"{object_name}_{tag}_rendered.png"), rendered_u8)
    cv2.imwrite(str(DEBUG_DIR / f"{object_name}_{tag}_overlay.png"), overlay)


def _camera_pose_components(camera_poses: dict[str, dict[str, list[float]]]) -> tuple[np.ndarray, np.ndarray]:
    pose = camera_poses["cam1"]
    return _quat_wxyz_to_matrix(pose["quat"]), np.asarray(pose["pos"], dtype=float)


def _evaluate_object_pose(
    object_name: str,
    quat_wxyz: list[float],
    pos_world: list[float],
    camera_poses: dict[str, dict[str, list[float]]],
    tag: str | None = None,
) -> dict[str, float | list[float]]:
    mesh = _load_mesh(object_name)
    gt_mask = _load_mask(object_name)
    camera_rotation, camera_translation = _camera_pose_components(camera_poses)
    rendered = _render_candidate_mask(
        mesh=mesh,
        rotation_world=_quat_wxyz_to_matrix(quat_wxyz),
        translation_world=np.asarray(pos_world, dtype=float),
        camera_rotation_world=camera_rotation,
        camera_translation_world=camera_translation,
    )
    score, iou, center_error_px = _score_rendered_mask(rendered, gt_mask)
    if tag is not None:
        _save_overlay(object_name, tag, rendered, gt_mask)
    return {
        "quat_wxyz": [float(v) for v in quat_wxyz],
        "pos": [float(v) for v in pos_world],
        "score": float(score),
        "iou": float(iou),
        "center_error_px": float(center_error_px),
    }


def main() -> None:
    _ensure_dirs()

    scene_inputs = build_original_runtime_scene_inputs(objects=SUPPORTED_OBJECTS)
    calibration = scene_inputs["calibration"]
    CALIBRATION_OUT.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")

    object_positions = scene_inputs["object_positions"]
    camera_poses = scene_inputs["camera_poses"]
    initial_quats = estimate_runtime_object_quats(
        camera="left",
        calibration_path=CALIBRATION_OUT,
        pear_strategy="v0_legacy",
    )
    initial_quats, initial_camera_facing_debug = align_object_quats_to_camera(
        initial_quats,
        object_positions,
        camera_poses,
        camera_name="cam1",
        objects=SUPPORTED_OBJECTS,
    )

    before = {
        object_name: _evaluate_object_pose(
            object_name=object_name,
            quat_wxyz=initial_quats[object_name],
            pos_world=object_positions[object_name],
            camera_poses=camera_poses,
            tag="before",
        )
        for object_name in SUPPORTED_OBJECTS
    }

    refined_quats = solve_left_view_object_quats(
        initial_object_quats=initial_quats,
        object_positions=object_positions,
        camera_poses=camera_poses,
    )
    refined_quats, refined_camera_facing_debug = align_object_quats_to_camera(
        refined_quats,
        object_positions,
        camera_poses,
        camera_name="cam1",
        objects=SUPPORTED_OBJECTS,
    )
    refined_positions = {name: [float(v) for v in object_positions[name]] for name in SUPPORTED_OBJECTS}

    after = {
        object_name: _evaluate_object_pose(
            object_name=object_name,
            quat_wxyz=refined_quats[object_name],
            pos_world=refined_positions[object_name],
            camera_poses=camera_poses,
            tag="after",
        )
        for object_name in SUPPORTED_OBJECTS
    }

    payload = {
        "calibration_path": str(CALIBRATION_OUT),
        "camera_poses": camera_poses,
        "position_refined": False,
        "camera_facing_local_axis": {
            "initial": initial_camera_facing_debug,
            "refined": refined_camera_facing_debug,
        },
        "initial": before,
        "refined": after,
    }
    POSE_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(POSE_OUT), "refined": after}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
