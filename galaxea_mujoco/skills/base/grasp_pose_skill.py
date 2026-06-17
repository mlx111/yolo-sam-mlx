from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import mujoco
import numpy as np


GraspMode = Literal["topdown", "side_x", "side_y", "current_rotation"]


@dataclass(frozen=True)
class GraspPoseResult:
    success: bool
    object_body: str
    grasp_mode: str
    object_position: list[float]
    object_half_extents: list[float]
    grasp_matrix_4x4: list[list[float]]
    pregrasp_matrix_4x4: list[list[float]]
    grasp_position: list[float]
    pregrasp_position: list[float]
    approach_axis_world: list[float]
    retreat_axis_world: list[float]
    grasp_width: float
    confidence: float
    executable_parameters: dict[str, float | str | list[float]] = field(default_factory=dict)
    message: str = ""


class R1ProGraspPoseSkill:
    """Generate grasp/pregrasp SE(3) poses without changing robot control state."""

    def compute_from_object(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        object_body: str,
        side: Literal["left", "right"] = "left",
        grasp_mode: GraspMode = "topdown",
        pregrasp_distance: float = 0.06,
        grasp_offset: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        grasp_width_margin: float = 0.012,
        min_grasp_width: float = 0.018,
        max_grasp_width: float = 0.085,
    ) -> GraspPoseResult:
        mujoco.mj_forward(model, data)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        if body_id < 0:
            return _failed(object_body, grasp_mode, f"MuJoCo body not found: {object_body}")

        object_position = data.xpos[body_id].copy()
        object_xmat = data.xmat[body_id].reshape(3, 3).copy()
        half_extents = _object_half_extents(model, body_id)
        offset = np.zeros(3, dtype=np.float64) if grasp_offset is None else np.asarray(grasp_offset, dtype=np.float64).reshape(3)
        grasp_position = object_position + offset
        grasp_rotation, approach_axis_world, topdown_mode = _rotation_for_mode(
            object_xmat=object_xmat,
            side=side,
            grasp_mode=grasp_mode,
        )
        approach_axis_world = _normalize(approach_axis_world)
        retreat_axis_world = -approach_axis_world
        pregrasp_position = grasp_position + retreat_axis_world * float(pregrasp_distance)

        grasp_matrix = _make_transform(grasp_rotation, grasp_position)
        pregrasp_matrix = _make_transform(grasp_rotation, pregrasp_position)
        grasp_width = _estimate_grasp_width(half_extents, grasp_mode, grasp_width_margin, min_grasp_width, max_grasp_width)

        return GraspPoseResult(
            success=True,
            object_body=object_body,
            grasp_mode=grasp_mode,
            object_position=_round_list(object_position),
            object_half_extents=_round_list(half_extents),
            grasp_matrix_4x4=_round_matrix(grasp_matrix),
            pregrasp_matrix_4x4=_round_matrix(pregrasp_matrix),
            grasp_position=_round_list(grasp_position),
            pregrasp_position=_round_list(pregrasp_position),
            approach_axis_world=_round_list(approach_axis_world),
            retreat_axis_world=_round_list(retreat_axis_world),
            grasp_width=float(round(grasp_width, 6)),
            confidence=0.75 if grasp_mode == "topdown" else 0.55,
            executable_parameters={
                "side": side,
                "object_body": object_body,
                "grasp_offset_x": float(round(offset[0], 6)),
                "grasp_offset_y": float(round(offset[1], 6)),
                "grasp_offset_z": float(round(offset[2], 6)),
                "pregrasp_distance": float(round(pregrasp_distance, 6)),
                "topdown_mode": topdown_mode,
                "control_frame": "grasp_tool",
            },
            message="grasp pose generated; execute positions with the stable grasp executor before enabling full pose tracking",
        )


def load_skill() -> R1ProGraspPoseSkill:
    return R1ProGraspPoseSkill()


def _failed(object_body: str, grasp_mode: str, message: str) -> GraspPoseResult:
    zero_transform = np.eye(4, dtype=np.float64)
    return GraspPoseResult(
        success=False,
        object_body=object_body,
        grasp_mode=grasp_mode,
        object_position=[0.0, 0.0, 0.0],
        object_half_extents=[0.0, 0.0, 0.0],
        grasp_matrix_4x4=_round_matrix(zero_transform),
        pregrasp_matrix_4x4=_round_matrix(zero_transform),
        grasp_position=[0.0, 0.0, 0.0],
        pregrasp_position=[0.0, 0.0, 0.0],
        approach_axis_world=[0.0, 0.0, -1.0],
        retreat_axis_world=[0.0, 0.0, 1.0],
        grasp_width=0.0,
        confidence=0.0,
        message=message,
    )


def _object_half_extents(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != int(body_id):
            continue
        geom_type = model.geom_type[geom_id]
        size = np.asarray(model.geom_size[geom_id][:3], dtype=np.float64)
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return np.array([size[0], size[0], size[0]], dtype=np.float64)
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return size.copy()
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return np.array([size[0], size[0], size[1]], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def _rotation_for_mode(
    *,
    object_xmat: np.ndarray,
    side: Literal["left", "right"],
    grasp_mode: GraspMode,
) -> tuple[np.ndarray, np.ndarray, str]:
    if grasp_mode == "current_rotation":
        rotation = _project_to_rotation_matrix(object_xmat)
        approach_axis = -rotation[:, 2]
        return rotation, approach_axis, "current"
    if grasp_mode == "side_x":
        approach_axis = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        y_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return _basis_from_approach_and_y(approach_axis, y_axis), approach_axis, "x_forward"
    if grasp_mode == "side_y":
        approach_axis = np.array([0.0, -1.0 if side == "left" else 1.0, 0.0], dtype=np.float64)
        y_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return _basis_from_approach_and_y(approach_axis, y_axis), approach_axis, "x_side"

    # Match the stable R1Pro top-down convention in continuous_grasp_executor:
    # y axis points up, z axis points laterally, and approach is world -Z.
    y_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_axis = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    x_axis = _normalize(np.cross(y_axis, z_axis))
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return rotation, np.array([0.0, 0.0, -1.0], dtype=np.float64), "palm_down"


def _basis_from_approach_and_y(approach_axis: np.ndarray, y_axis_hint: np.ndarray) -> np.ndarray:
    z_axis = -_normalize(approach_axis)
    y_axis = _normalize(y_axis_hint - np.dot(y_axis_hint, z_axis) * z_axis)
    x_axis = _normalize(np.cross(y_axis, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def _estimate_grasp_width(
    half_extents: np.ndarray,
    grasp_mode: GraspMode,
    margin: float,
    min_width: float,
    max_width: float,
) -> float:
    if grasp_mode == "side_x":
        width = 2.0 * float(half_extents[1]) + float(margin)
    elif grasp_mode == "side_y":
        width = 2.0 * float(half_extents[0]) + float(margin)
    else:
        width = 2.0 * min(float(half_extents[0]), float(half_extents[1])) + float(margin)
    return float(np.clip(width, min_width, max_width))


def _make_transform(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _project_to_rotation_matrix(rotation)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform


def _project_to_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("cannot normalize near-zero vector")
    return arr / norm


def _round_list(values: np.ndarray) -> list[float]:
    return [float(round(v, 6)) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


def _round_matrix(values: np.ndarray) -> list[list[float]]:
    arr = np.asarray(values, dtype=np.float64)
    return [[float(round(item, 6)) for item in row] for row in arr.tolist()]
