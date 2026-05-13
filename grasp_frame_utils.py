from __future__ import annotations

import numpy as np
import spatialmath as sm
import mujoco


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def camera_rotation_world_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str = "cam1",
    *,
    align_rx_pi: bool = True,
) -> np.ndarray:
    """Return the world-from-camera rotation matrix used by the existing grasp execution path."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera not found in MuJoCo model: {camera_name}")

    t_wc_raw = data.cam_xpos[cam_id].copy()
    r_wc_raw = data.cam_xmat[cam_id].reshape(3, 3).copy()
    t_wc_raw_se3 = sm.SE3.Rt(sm.SO3(r_wc_raw), t_wc_raw)
    if align_rx_pi:
        t_wc = t_wc_raw_se3 * sm.SE3.Rx(np.pi)
    else:
        t_wc = t_wc_raw_se3
    return np.asarray(t_wc.R, dtype=float)


def filter_grasps_by_world_vertical(
    grasps,
    camera_rotation_world_from_cam: np.ndarray,
    world_vertical: np.ndarray | None = None,
    *,
    angle_threshold_deg: float = 30.0,
    keep_top_k: int = 20,
):
    """Filter grasp candidates by comparing the grasp approach axis in world coordinates."""
    world_vertical = np.asarray(world_vertical if world_vertical is not None else [0.0, 0.0, 1.0], dtype=float)
    world_vertical = world_vertical / max(float(np.linalg.norm(world_vertical)), 1e-12)
    camera_rotation_world_from_cam = np.asarray(camera_rotation_world_from_cam, dtype=float)

    all_grasps = list(grasps)
    if len(all_grasps) == 0:
        return []

    angle_threshold = np.deg2rad(angle_threshold_deg)
    filtered = []
    for grasp in all_grasps:
        approach_cam = np.asarray(grasp.rotation_matrix[:, 0], dtype=float)
        approach_world = camera_rotation_world_from_cam @ approach_cam
        norm = float(np.linalg.norm(approach_world))
        if norm <= 1e-12:
            continue
        approach_world = approach_world / norm
        cos_angle = float(np.dot(approach_world, world_vertical))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(grasp)

    if len(filtered) == 0:
        filtered = all_grasps

    filtered.sort(key=lambda g: g.score, reverse=True)
    return filtered[:keep_top_k]


def grasp_approach_tilt_deg(
    grasp,
    camera_rotation_world_from_cam: np.ndarray,
    world_axis: np.ndarray | None = None,
) -> float:
    """Return the unsigned tilt angle of the grasp approach axis away from a world axis."""
    world_axis = _normalize(world_axis if world_axis is not None else [0.0, 0.0, 1.0])
    camera_rotation_world_from_cam = np.asarray(camera_rotation_world_from_cam, dtype=float)

    approach_cam = np.asarray(grasp.rotation_matrix[:, 0], dtype=float)
    approach_world = camera_rotation_world_from_cam @ approach_cam
    norm = float(np.linalg.norm(approach_world))
    if norm <= 1e-12:
        return 0.0
    approach_world = approach_world / norm

    cos_tilt = float(abs(np.dot(approach_world, world_axis)))
    cos_tilt = float(np.clip(cos_tilt, 0.0, 1.0))
    return float(np.degrees(np.arccos(cos_tilt)))


def filter_grasps_by_world_tilt(
    grasps,
    camera_rotation_world_from_cam: np.ndarray,
    world_axis: np.ndarray | None = None,
    *,
    min_tilt_deg: float = 55.0,
    keep_top_k: int = 20,
):
    """Keep grasps whose approach axis stays away from a world axis by at least min_tilt_deg."""
    all_grasps = list(grasps)
    if len(all_grasps) == 0:
        return []

    filtered = []
    for grasp in all_grasps:
        tilt_deg = grasp_approach_tilt_deg(grasp, camera_rotation_world_from_cam, world_axis=world_axis)
        if tilt_deg >= min_tilt_deg:
            filtered.append((tilt_deg, float(grasp.score), grasp))

    if len(filtered) == 0:
        return []

    filtered.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [grasp for _, _, grasp in filtered[:keep_top_k]]
