from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation


SUPPORTED_OBJECTS = ("apple", "pear")
FRONT_AXIS_LOCAL_BY_OBJECT = {
    "apple": np.array([0.0, 0.0, -1.0], dtype=float),
    "pear": np.array([0.0, 0.0, -1.0], dtype=float),
}
HINGE_AXIS_LOCAL_BY_OBJECT = {
    "apple": np.array([0.0, 1.0, 0.0], dtype=float),
    "pear": np.array([0.0, 1.0, 0.0], dtype=float),
}
TOP_AXIS_LOCAL_BY_OBJECT = {
    "apple": np.array([0.0, -1.0, 0.0], dtype=float),
    "pear": np.array([0.0, -1.0, 0.0], dtype=float),
}
DELTA_SOURCE_BY_OBJECT = {
    "apple": "pear",
    "pear": "pear",
}


class CameraFacingQuatError(RuntimeError):
    pass


def _quat_wxyz_to_matrix(quat_wxyz: list[float]) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=float)
    if quat.shape != (4,):
        raise CameraFacingQuatError(f"Expected quaternion shape (4,), got {quat.shape}")
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _matrix_to_quat_wxyz(rotation_matrix: np.ndarray) -> list[float]:
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise CameraFacingQuatError("Cannot normalize near-zero vector.")
    return vector / norm


def _signed_angle_about_axis(a: np.ndarray, b: np.ndarray, axis: np.ndarray) -> float:
    a_n = _normalize(a)
    b_n = _normalize(b)
    axis_n = _normalize(axis)
    sin_term = float(axis_n @ np.cross(a_n, b_n))
    cos_term = float(np.clip(a_n @ b_n, -1.0, 1.0))
    return float(math.atan2(sin_term, cos_term))


def _axes_for_object(object_name: str) -> tuple[np.ndarray, np.ndarray]:
    if object_name not in FRONT_AXIS_LOCAL_BY_OBJECT or object_name not in HINGE_AXIS_LOCAL_BY_OBJECT:
        raise CameraFacingQuatError(f"Unsupported object for camera-facing alignment: {object_name}")
    return FRONT_AXIS_LOCAL_BY_OBJECT[object_name], HINGE_AXIS_LOCAL_BY_OBJECT[object_name]


def compute_camera_facing_delta(
    object_name: str,
    quat_wxyz: list[float],
    object_pos_world: list[float],
    camera_pos_world: list[float],
) -> tuple[float, dict[str, float | str | bool | None]]:
    rotation_world = _quat_wxyz_to_matrix(quat_wxyz)
    object_pos = np.asarray(object_pos_world, dtype=float)
    camera_pos = np.asarray(camera_pos_world, dtype=float)
    front_local, hinge_local = _axes_for_object(object_name)

    front_world = rotation_world @ front_local
    hinge_world = _normalize(rotation_world @ hinge_local)
    to_camera_world = camera_pos - object_pos

    front_projected = front_world - hinge_world * float(front_world @ hinge_world)
    camera_projected = to_camera_world - hinge_world * float(to_camera_world @ hinge_world)

    if float(np.linalg.norm(front_projected)) <= 1e-9 or float(np.linalg.norm(camera_projected)) <= 1e-9:
        return 0.0, {
            "front_axis_local": "-Z",
            "hinge_axis_local": "+Y",
            "delta_computed_deg": 0.0,
            "residual_deg": None,
            "applied": False,
        }

    delta_rad = _signed_angle_about_axis(front_projected, camera_projected, hinge_world)
    return delta_rad, {
        "front_axis_local": "-Z",
        "hinge_axis_local": "+Y",
        "delta_computed_deg": float(np.degrees(delta_rad)),
        "residual_deg": None,
        "applied": True,
    }


def apply_top_axis_upright_constraint(
    object_name: str,
    quat_wxyz: list[float],
    object_pos_world: list[float],
    camera_pos_world: list[float],
) -> tuple[list[float], dict[str, float | str | bool | None]]:
    if object_name not in TOP_AXIS_LOCAL_BY_OBJECT:
        return [float(v) for v in quat_wxyz], {
            "top_axis_local": None,
            "top_upright_delta_deg": 0.0,
            "top_residual_deg": None,
            "top_constraint_applied": False,
        }

    rotation_world_before = _quat_wxyz_to_matrix(quat_wxyz)
    front_local, _ = _axes_for_object(object_name)
    top_local = TOP_AXIS_LOCAL_BY_OBJECT[object_name]
    object_pos = np.asarray(object_pos_world, dtype=float)
    camera_pos = np.asarray(camera_pos_world, dtype=float)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)

    to_camera_world = camera_pos - object_pos
    front_target = to_camera_world - world_up * float(to_camera_world @ world_up)
    if float(np.linalg.norm(front_target)) <= 1e-9:
        return [float(v) for v in quat_wxyz], {
            "top_axis_local": "-Y",
            "top_upright_delta_deg": 0.0,
            "top_residual_deg": None,
            "top_constraint_applied": False,
        }
    front_target = _normalize(front_target)

    y_world = -world_up
    z_world = -front_target
    x_world = _normalize(np.cross(y_world, z_world))
    z_world = _normalize(np.cross(x_world, y_world))
    corrected_rotation_world = np.column_stack([x_world, y_world, z_world])
    if float(np.linalg.det(corrected_rotation_world)) < 0.0:
        corrected_rotation_world[:, 0] = -corrected_rotation_world[:, 0]

    top_world = corrected_rotation_world @ top_local
    front_world = corrected_rotation_world @ front_local
    top_residual_deg = float(np.degrees(np.arccos(np.clip(_normalize(top_world) @ world_up, -1.0, 1.0))))
    front_horizontal = front_world - world_up * float(front_world @ world_up)
    front_residual_deg = float(np.degrees(np.arccos(np.clip(_normalize(front_horizontal) @ front_target, -1.0, 1.0))))

    relative_rotation = corrected_rotation_world @ rotation_world_before.T
    delta_deg = float(np.degrees(Rotation.from_matrix(relative_rotation).magnitude()))

    return _matrix_to_quat_wxyz(corrected_rotation_world), {
        "top_axis_local": "-Y",
        "top_upright_delta_deg": delta_deg,
        "top_residual_deg": top_residual_deg,
        "top_front_horizontal_residual_deg": front_residual_deg,
        "top_constraint_applied": True,
    }


def apply_camera_facing_delta(
    object_name: str,
    quat_wxyz: list[float],
    delta_rad: float,
    object_pos_world: list[float],
    camera_pos_world: list[float],
) -> tuple[list[float], dict[str, float | str | bool | None]]:
    rotation_world = _quat_wxyz_to_matrix(quat_wxyz)
    object_pos = np.asarray(object_pos_world, dtype=float)
    camera_pos = np.asarray(camera_pos_world, dtype=float)
    front_local, hinge_local = _axes_for_object(object_name)

    hinge_world = _normalize(rotation_world @ hinge_local)
    correction_world = Rotation.from_rotvec(hinge_world * delta_rad).as_matrix()
    corrected_rotation_world = correction_world @ rotation_world

    corrected_front_world = corrected_rotation_world @ front_local
    corrected_hinge_world = _normalize(corrected_rotation_world @ hinge_local)
    to_camera_world = camera_pos - object_pos
    corrected_front_projected = corrected_front_world - corrected_hinge_world * float(corrected_front_world @ corrected_hinge_world)
    corrected_camera_projected = to_camera_world - corrected_hinge_world * float(to_camera_world @ corrected_hinge_world)

    residual_deg: float | None
    if float(np.linalg.norm(corrected_front_projected)) <= 1e-9 or float(np.linalg.norm(corrected_camera_projected)) <= 1e-9:
        residual_deg = None
    else:
        residual_cos = float(np.clip(_normalize(corrected_front_projected) @ _normalize(corrected_camera_projected), -1.0, 1.0))
        residual_deg = float(np.degrees(np.arccos(residual_cos)))

    debug = {
        "front_axis_local": "-Z",
        "hinge_axis_local": "+Y",
        "delta_used_deg": float(np.degrees(delta_rad)),
        "residual_deg": residual_deg,
        "applied": True,
    }
    return _matrix_to_quat_wxyz(corrected_rotation_world), debug


def align_object_quats_to_camera(
    object_quats: dict[str, list[float]],
    object_positions: dict[str, list[float]],
    camera_poses: dict[str, dict[str, list[float]]],
    *,
    camera_name: str = "cam1",
    objects: Iterable[str] | None = None,
) -> tuple[dict[str, list[float]], dict[str, dict[str, float | str | bool | None]]]:
    if camera_name not in camera_poses:
        raise CameraFacingQuatError(f"Missing camera pose for {camera_name}")
    target_objects = tuple(objects) if objects is not None else tuple(object_quats.keys())
    camera_pos = camera_poses[camera_name]["pos"]

    computed_delta_by_object: dict[str, float] = {}
    computed_debug_by_object: dict[str, dict[str, float | str | bool | None]] = {}
    for object_name in target_objects:
        if object_name not in object_quats:
            raise CameraFacingQuatError(f"Missing quaternion for {object_name}")
        if object_name not in object_positions:
            raise CameraFacingQuatError(f"Missing position for {object_name}")
        delta_rad, debug = compute_camera_facing_delta(
            object_name,
            object_quats[object_name],
            object_positions[object_name],
            camera_pos,
        )
        computed_delta_by_object[object_name] = delta_rad
        computed_debug_by_object[object_name] = debug

    aligned: dict[str, list[float]] = {}
    debug: dict[str, dict[str, float | str | bool | None]] = {}
    for object_name in target_objects:
        delta_source_object = DELTA_SOURCE_BY_OBJECT.get(object_name, object_name)
        if delta_source_object not in computed_delta_by_object:
            raise CameraFacingQuatError(
                f"Missing delta source {delta_source_object} for {object_name}. Available: {sorted(computed_delta_by_object)}"
            )
        corrected_quat, applied_debug = apply_camera_facing_delta(
            object_name,
            object_quats[object_name],
            computed_delta_by_object[delta_source_object],
            object_positions[object_name],
            camera_pos,
        )
        corrected_quat, top_debug = apply_top_axis_upright_constraint(
            object_name,
            corrected_quat,
            object_positions[object_name],
            camera_pos,
        )
        aligned[object_name] = corrected_quat
        debug[object_name] = {
            **computed_debug_by_object[object_name],
            **applied_debug,
            **top_debug,
            "delta_source_object": delta_source_object,
        }
    return aligned, debug
