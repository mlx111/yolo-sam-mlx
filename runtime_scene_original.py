from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from camera_pose_mujoco import (
    DEFAULT_CALIBRATION_PATH,
    build_calibration_from_reference_scene,
    convert_raw_rotation_to_mujoco,
    rotation_matrix_from_euler_xyz_deg,
    save_camera_pose_calibration,
)
from pointcloud_v2 import CAMERA_EULER_DEG, estimate_runtime_camera_poses, pos


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OBJECTS = ("apple", "pear")
CAMERA_NAME_MAP = {"left": "cam1", "right": "cam2"}
DEFAULT_REFERENCE_SCENE_PATH = (
    ROOT_DIR / "manipulator_grasp" / "assets" / "scenes" / "apple_pear_runtime_refined111.xml"
)


class OriginalRuntimeSceneError(RuntimeError):
    pass


def _float_list(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def _ensure_camera_pose_calibration() -> None:
    if DEFAULT_CALIBRATION_PATH.exists():
        return
    if not DEFAULT_REFERENCE_SCENE_PATH.exists():
        return

    calibration = build_calibration_from_reference_scene(
        reference_scene_path=DEFAULT_REFERENCE_SCENE_PATH,
        raw_euler_deg_by_camera=CAMERA_EULER_DEG,
    )
    save_camera_pose_calibration(calibration, DEFAULT_CALIBRATION_PATH)


def build_original_runtime_scene_inputs(objects: Iterable[str] = DEFAULT_OBJECTS) -> dict:
    _ensure_camera_pose_calibration()
    object_names = [str(name) for name in objects]
    object_positions_raw = pos(object_names)
    scene_camera_poses = estimate_runtime_camera_poses()

    camera_poses = {}
    for flag, scene_name in CAMERA_NAME_MAP.items():
        scene_pose = scene_camera_poses.get(scene_name)
        if not isinstance(scene_pose, dict):
            raise OriginalRuntimeSceneError(f'Missing scene camera pose for {scene_name}')
        rx, ry, rz = CAMERA_EULER_DEG[flag]
        mujoco_pose = convert_raw_rotation_to_mujoco(
            rotation_matrix_from_euler_xyz_deg(rx, ry, rz),
            flag,
        )
        camera_poses[flag] = {
            'translation_mj': _float_list(scene_pose['pos']),
            'quat_wxyz': _float_list(scene_pose['quat']),
            'rotation_matrix_mj_from_cam': [[float(v) for v in row] for row in mujoco_pose.rotation_matrix],
            'rotation_matrix_world_from_cam': [[float(v) for v in row] for row in mujoco_pose.rotation_matrix],
            'point_transform_matrix': [[float(v) for v in row] for row in mujoco_pose.point_transform_matrix],
            'rotation_transform_matrix': [[float(v) for v in row] for row in mujoco_pose.rotation_transform_matrix],
            'fixed_rotation_matrix': (
                None
                if mujoco_pose.fixed_rotation_matrix is None
                else [[float(v) for v in row] for row in mujoco_pose.fixed_rotation_matrix]
            ),
            'refinement_euler_xyz_deg': _float_list(mujoco_pose.refinement_euler_xyz_deg),
            'selected_candidate': str(mujoco_pose.selected_candidate),
        }

    object_positions = {
        f'{name}_world': _float_list(object_positions_raw[name])
        for name in object_names
        if name in object_positions_raw
    }

    calibration = {
        'world_axes': {
            'world_frame_source': 'pointcloud_v2_original_camera_mapping',
        },
        'camera_poses': camera_poses,
        'robot_positions': {
            'arm_base_world': [0.0, 0.0, 0.0],
        },
        'object_positions': object_positions,
        'relative_positions': {
            f'{name}_minus_arm': values
            for name, values in object_positions.items()
        },
        'relative_position_source': 'pointcloud_v2.pos averaged left/right local centers',
    }
    return {
        'calibration': calibration,
        'camera_poses': {
            scene_name: {
                'pos': _float_list(scene_camera_poses[scene_name]['pos']),
                'quat': _float_list(scene_camera_poses[scene_name]['quat']),
            }
            for scene_name in CAMERA_NAME_MAP.values()
            if scene_name in scene_camera_poses
        },
        'object_positions': {
            name: _float_list(object_positions_raw[name])
            for name in object_names
            if name in object_positions_raw
        },
    }


def main() -> None:
    payload = build_original_runtime_scene_inputs()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
