from __future__ import annotations

import os
import sys

import cv2
import mujoco
import numpy as np
import open3d as o3d
import spatialmath as sm
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "Grounded-SAM-2"))
sys.path.append(os.path.join(ROOT_DIR, "graspnet-baseline", "models"))
sys.path.append(os.path.join(ROOT_DIR, "graspnet-baseline", "dataset"))
sys.path.append(os.path.join(ROOT_DIR, "graspnet-baseline", "utils"))
sys.path.append(os.path.join(ROOT_DIR, "manipulator_grasp"))
sys.path.append(os.path.join(ROOT_DIR, "anygrasp_sdk", "grasp_detection"))

import main_yoloWorld_sam as base_main
from cv_proc import segment_image_ground
from graspnetAPI import Grasp
from manipulator_grasp.arm.motion_planning import *  # noqa: F403
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv
from grasp_frame_utils import (
    camera_rotation_world_from_mujoco,
    filter_grasps_by_world_vertical,
    grasp_approach_tilt_deg,
)
from pointcloud_completion_utils import (
    CompletionConfig,
    complete_point_cloud,
    make_point_cloud,
    point_cloud_bounds,
)

COMPLETION_CONFIG = CompletionConfig()
WORLD_VERTICAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Static translation correction applied in code before execution.
# `frame="world"` means offset in world XYZ.
# `frame="grasp"` means offset in the grasp local XYZ.
# Unit: meters.
DEFAULT_GRASP_TRANSLATION_CORRECTION = {
    "frame": "world",
    "offset_m": [0.0, 0.020, 0.0],
}

GRASP_TRANSLATION_CORRECTIONS_BY_CLASS = {
    # "apple": {"frame": "world", "offset_m": [0.005, -0.003, 0.0]},
    # "pear": {"frame": "grasp", "offset_m": [0.0, 0.0, -0.004]},
}

# Candidate mappings from the GraspNet frame into the robot TCP frame.
# Each candidate also declares the TCP local axis that should retreat away
# from the object for a top-down pregrasp.
TCP_FRAME_CANDIDATES = [
    {
        "name": "tcp_+x_from_grasp_x",
        "R_grasp_to_tcp": np.eye(3, dtype=np.float64),
        "retreat_axis_tcp": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    },
    {
        "name": "tcp_-x_from_grasp_x",
        "R_grasp_to_tcp": np.diag([-1.0, 1.0, -1.0]).astype(np.float64),
        "retreat_axis_tcp": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    },
    {
        "name": "tcp_+z_from_grasp_x",
        "R_grasp_to_tcp": np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "retreat_axis_tcp": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    },
    {
        "name": "tcp_-z_from_grasp_x",
        "R_grasp_to_tcp": np.array(
            [
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "retreat_axis_tcp": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    },
]


def _show_side_by_side(raw_points: np.ndarray, completed_points: np.ndarray) -> None:
    raw_cloud = make_point_cloud(raw_points)
    completed_cloud = make_point_cloud(completed_points)

    raw_bounds = point_cloud_bounds(raw_points)
    completed_bounds = point_cloud_bounds(completed_points)
    print("[INFO] raw bbox extent (m):", np.round(raw_bounds["extent"], 4).tolist())
    print("[INFO] completed bbox extent (m):", np.round(completed_bounds["extent"], 4).tolist())

    raw_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
    completed_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
    raw_offset = np.array([-0.18, 0.0, 0.0], dtype=np.float64)
    completed_offset = np.array([0.18, 0.0, 0.0], dtype=np.float64)
    raw_cloud.translate(raw_offset)
    completed_cloud.translate(completed_offset)
    raw_frame.translate(raw_offset)
    completed_frame.translate(completed_offset)
    o3d.visualization.draw_geometries([raw_cloud, raw_frame, completed_cloud, completed_frame])


def _build_completed_end_points(points: np.ndarray) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    points_tensor = torch.from_numpy(points[np.newaxis].astype(np.float32)).to(device)
    colors = np.zeros((len(points), 3), dtype=np.float32)
    return {
        "point_clouds": points_tensor,
        "cloud_colors": colors,
    }


def _build_open3d_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype=np.float64))
    if colors is None:
        colors = np.zeros((len(points), 3), dtype=np.float64)
    cloud.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors, dtype=np.float64))
    return cloud


def _single_grasp_group(grasp):
    gg = base_main.GraspGroup()
    gg.add(grasp)
    return gg


def _clone_grasp(grasp: Grasp) -> Grasp:
    return Grasp(np.array(grasp.grasp_array, dtype=np.float64, copy=True))


def _run_trajectory(env, robot, action, planner, time_seconds: float):
    time_array = [0.0, time_seconds]
    planner_array = [planner]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break


def _run_cartesian_move(env, robot, T_start: sm.SE3, T_end: sm.SE3, time_seconds: float, action):
    position_parameter = LinePositionParameter(T_start.t, T_end.t)
    attitude_parameter = OneAttitudeParameter(sm.SO3(T_start.R), sm.SO3(T_end.R))
    cartesian_parameter = CartesianParameter(position_parameter, attitude_parameter)
    velocity_parameter = QuinticVelocityParameter(time_seconds)
    trajectory_parameter = TrajectoryParameter(cartesian_parameter, velocity_parameter)
    planner = TrajectoryPlanner(trajectory_parameter)
    _run_trajectory(env, robot, action, planner, time_seconds)


def _project_to_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    u, _, vh = np.linalg.svd(matrix)
    rot = u @ vh
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vh
    return rot


def _wrapped_joint_delta(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    q_from = np.asarray(q_from, dtype=np.float64)
    q_to = np.asarray(q_to, dtype=np.float64)
    return np.arctan2(np.sin(q_to - q_from), np.cos(q_to - q_from))


def _camera_pose_world_from_env(env: UR5GraspEnv) -> sm.SE3:
    data = env.mj_data
    model = env.mj_model
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam1")
    if cam_id < 0:
        raise ValueError("Camera cam1 not found in MuJoCo model.")
    t_wc_raw = data.cam_xpos[cam_id].copy()
    R_wc_raw = data.cam_xmat[cam_id].reshape(3, 3).copy()
    T_wc_raw = sm.SE3.Rt(sm.SO3(R_wc_raw), t_wc_raw)
    return T_wc_raw * sm.SE3.Rx(np.pi)


def _extract_top_surface_info(
    completed_points: np.ndarray,
    T_wc: sm.SE3,
    *,
    top_band_m: float = 0.01,
    top_band_fallback_quantile: float = 0.92,
) -> dict | None:
    if len(completed_points) == 0:
        return None

    completed_points = np.asarray(completed_points, dtype=np.float64)
    world_points = (np.asarray(T_wc.R, dtype=np.float64) @ completed_points.T).T + np.asarray(T_wc.t, dtype=np.float64).reshape(1, 3)
    z_vals = world_points[:, 2]
    z_max = float(np.max(z_vals))
    mask = z_vals >= (z_max - float(top_band_m))
    if int(np.count_nonzero(mask)) < 64:
        z_thresh = float(np.quantile(z_vals, top_band_fallback_quantile))
        mask = z_vals >= z_thresh
    top_points = world_points[mask]
    if len(top_points) == 0:
        return None

    top_xy_center = np.median(top_points[:, :2], axis=0)
    top_xy_distances = np.linalg.norm(top_points[:, :2] - top_xy_center.reshape(1, 2), axis=1)
    top_xy_radius = float(np.quantile(top_xy_distances, 0.8))
    top_xy_inner_radius = float(np.quantile(top_xy_distances, 0.5))
    return {
        "world_points": world_points,
        "top_points": top_points,
        "z_max": z_max,
        "z_min_band": float(np.min(top_points[:, 2])),
        "top_xy_center": top_xy_center,
        "top_xy_radius": top_xy_radius,
        "top_xy_inner_radius": top_xy_inner_radius,
        "count": int(len(top_points)),
    }


def _score_grasp_for_top_surface(
    grasp,
    T_wc: sm.SE3,
    top_surface_info: dict | None,
) -> dict:
    if top_surface_info is None:
        return {
            "world_top_hit": False,
            "world_top_distance_m": float("inf"),
            "world_top_z_gap_m": float("inf"),
            "world_top_center_offset_m": float("inf"),
            "world_top_inner_hit": False,
        }

    translation = np.asarray(grasp.translation, dtype=np.float64)
    world_translation = np.asarray(T_wc.R @ translation + T_wc.t, dtype=np.float64)
    top_points = np.asarray(top_surface_info["top_points"], dtype=np.float64)
    dists = np.linalg.norm(top_points - world_translation.reshape(1, 3), axis=1)
    top_surface_distance_m = float(np.min(dists))
    top_z_gap_m = float(top_surface_info["z_max"] - world_translation[2])
    xy_center_distance_m = float(np.linalg.norm(world_translation[:2] - top_surface_info["top_xy_center"]))
    top_band_hit = bool(world_translation[2] >= top_surface_info["z_min_band"])
    top_inner_hit = bool(xy_center_distance_m <= top_surface_info["top_xy_inner_radius"])
    return {
        "world_top_hit": top_band_hit,
        "world_top_distance_m": top_surface_distance_m,
        "world_top_z_gap_m": top_z_gap_m,
        "world_top_center_offset_m": xy_center_distance_m,
        "world_top_inner_hit": top_inner_hit,
    }


def _select_surface_preferred_grasp(
    grasps,
    T_wc: sm.SE3,
    top_surface_info: dict | None,
) -> tuple[object | None, dict]:
    candidates = []
    for grasp in grasps:
        surface_stats = _score_grasp_for_top_surface(grasp, T_wc, top_surface_info)
        candidates.append(
            {
                "grasp": grasp,
                "score": float(grasp.score),
                **surface_stats,
            }
        )

    if len(candidates) == 0:
        return None, {"selection_source": "empty"}

    preferred = [
        item
        for item in candidates
        if item["world_top_hit"] and item["world_top_distance_m"] <= 0.02
    ]

    if len(preferred) > 0:
        preferred.sort(
            key=lambda item: (
                not item["world_top_inner_hit"],
                item["world_top_distance_m"],
                item["world_top_z_gap_m"],
                item["world_top_center_offset_m"],
                -item["score"],
            )
        )
        best = preferred[0]
        selection_source = "world-top preferred"
    else:
        candidates.sort(
            key=lambda item: (
                not item["world_top_hit"],
                item["world_top_distance_m"],
                item["world_top_center_offset_m"],
                -item["score"],
            )
        )
        best = candidates[0]
        selection_source = "fallback"

    print("[INFO] selected grasp source:", selection_source)
    return best["grasp"], {
        "selection_source": selection_source,
        "world_top_distance_m": best["world_top_distance_m"],
        "world_top_z_gap_m": best["world_top_z_gap_m"],
        "world_top_center_offset_m": best["world_top_center_offset_m"],
        "world_top_hit": best["world_top_hit"],
        "world_top_inner_hit": best["world_top_inner_hit"],
    }


def _normalize_vector(vec: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-9:
        return vec / norm
    if fallback is None:
        raise ValueError("Cannot normalize near-zero vector without a fallback.")
    fallback = np.asarray(fallback, dtype=np.float64).reshape(3)
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm <= 1e-9:
        raise ValueError("Fallback vector is also near zero.")
    return fallback / fallback_norm


def _stabilize_horizontal_axis(axis_xy: np.ndarray) -> np.ndarray:
    axis = _normalize_vector(np.array([axis_xy[0], axis_xy[1], 0.0], dtype=np.float64), fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    prefer_x = abs(float(axis[0])) >= abs(float(axis[1]))
    if prefer_x:
        if axis[0] < 0.0:
            axis = -axis
    else:
        if axis[1] < 0.0:
            axis = -axis
    return axis


def _transform_points_to_world(points: np.ndarray, T_wc: sm.SE3) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return (np.asarray(T_wc.R, dtype=np.float64) @ points.T).T + np.asarray(T_wc.t, dtype=np.float64).reshape(1, 3)


def _transform_pose_world_to_camera(
    translation_world: np.ndarray,
    rotation_world: np.ndarray,
    T_wc: sm.SE3,
) -> tuple[np.ndarray, np.ndarray]:
    R_wc = np.asarray(T_wc.R, dtype=np.float64)
    t_wc = np.asarray(T_wc.t, dtype=np.float64)
    R_cw = R_wc.T
    translation_cam = R_cw @ (np.asarray(translation_world, dtype=np.float64) - t_wc)
    rotation_cam = _project_to_rotation_matrix(R_cw @ np.asarray(rotation_world, dtype=np.float64))
    return translation_cam, rotation_cam


def _grasp_translation_world(grasp: Grasp, T_wc: sm.SE3) -> np.ndarray:
    translation_cam = np.asarray(grasp.translation, dtype=np.float64)
    return np.asarray(T_wc.R @ translation_cam + T_wc.t, dtype=np.float64)


def _apply_translation_offset_to_grasp(
    grasp: Grasp,
    T_wc: sm.SE3,
    offset_m: np.ndarray,
    *,
    frame: str,
) -> Grasp:
    corrected = _clone_grasp(grasp)
    offset_m = np.asarray(offset_m, dtype=np.float64).reshape(3)
    if frame == "world":
        offset_cam = np.asarray(T_wc.R, dtype=np.float64).T @ offset_m
    elif frame == "grasp":
        grasp_rotation = _project_to_rotation_matrix(np.asarray(corrected.rotation_matrix, dtype=np.float64))
        offset_cam = grasp_rotation @ offset_m
    else:
        raise ValueError(f"Unsupported correction frame: {frame}")
    corrected.translation = np.asarray(corrected.translation, dtype=np.float64) + offset_cam
    return corrected


def _static_translation_correction_for_class(target_class: str) -> tuple[str, np.ndarray]:
    config = GRASP_TRANSLATION_CORRECTIONS_BY_CLASS.get(target_class, DEFAULT_GRASP_TRANSLATION_CORRECTION)
    frame = str(config.get("frame", "world")).strip().lower()
    if frame not in {"world", "grasp"}:
        raise ValueError(f"Invalid correction frame for {target_class}: {frame}")
    offset_m = np.asarray(config.get("offset_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    return frame, offset_m


def _apply_static_grasp_translation_correction(
    grasp: Grasp,
    T_wc: sm.SE3,
    *,
    target_class: str,
) -> Grasp:
    corrected = _clone_grasp(grasp)
    frame, offset_m = _static_translation_correction_for_class(target_class)
    if np.linalg.norm(offset_m) > 0.0:
        corrected = _apply_translation_offset_to_grasp(
            corrected,
            T_wc,
            offset_m,
            frame=frame,
        )
        print(
            "[INFO] applied static grasp correction:",
            {
                "target_class": target_class,
                "frame": frame,
                "offset_mm": np.round(offset_m * 1000.0, 3).tolist(),
                "translation_camera_m": np.round(np.asarray(corrected.translation, dtype=np.float64), 4).tolist(),
                "translation_world_m": np.round(_grasp_translation_world(corrected, T_wc), 4).tolist(),
            },
        )
    return corrected


def _build_analytic_top_grasp(
    completed_points: np.ndarray,
    T_wc: sm.SE3,
    top_surface_info: dict | None,
):
    if top_surface_info is None or len(completed_points) == 0:
        return None, {"reason": "top surface unavailable"}

    world_points = np.asarray(top_surface_info["world_points"], dtype=np.float64)
    top_points = np.asarray(top_surface_info["top_points"], dtype=np.float64)
    if len(world_points) < 64 or len(top_points) < 32:
        return None, {"reason": "insufficient completed points"}

    z_vals = world_points[:, 2]
    z_min = float(np.min(z_vals))
    z_max = float(np.max(z_vals))
    obj_height = max(z_max - z_min, 1e-4)

    top_xy_center = np.asarray(top_surface_info["top_xy_center"], dtype=np.float64)
    xy_offsets = top_points[:, :2] - top_xy_center.reshape(1, 2)
    if len(top_points) >= 3:
        cov = np.cov(xy_offsets.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        minor_xy = eigvecs[:, 0]
        major_xy = eigvecs[:, 1]
    else:
        minor_xy = np.array([1.0, 0.0], dtype=np.float64)
        major_xy = np.array([0.0, 1.0], dtype=np.float64)

    closing_axis_world = _stabilize_horizontal_axis(minor_xy)
    finger_axis_world = _stabilize_horizontal_axis(major_xy)
    if abs(float(np.dot(closing_axis_world[:2], finger_axis_world[:2]))) > 0.95:
        finger_axis_world = np.array([-closing_axis_world[1], closing_axis_world[0], 0.0], dtype=np.float64)
    finger_axis_world = _normalize_vector(finger_axis_world)

    approach_axis_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    closing_axis_world = _normalize_vector(closing_axis_world - np.dot(closing_axis_world, approach_axis_world) * approach_axis_world, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    finger_axis_world = _normalize_vector(np.cross(approach_axis_world, closing_axis_world), fallback=finger_axis_world)
    rotation_world = np.column_stack((approach_axis_world, closing_axis_world, finger_axis_world))
    rotation_world = _project_to_rotation_matrix(rotation_world)

    world_xy_offsets = world_points[:, :2] - top_xy_center.reshape(1, 2)
    width_samples = world_xy_offsets @ closing_axis_world[:2]
    finger_samples = world_xy_offsets @ finger_axis_world[:2]
    estimated_width = float(np.quantile(width_samples, 0.95) - np.quantile(width_samples, 0.05))
    finger_span = float(np.quantile(finger_samples, 0.95) - np.quantile(finger_samples, 0.05))
    grasp_width = float(np.clip(estimated_width + 0.01, 0.03, 0.095))

    core_radius = max(float(top_surface_info["top_xy_inner_radius"]), 0.012)
    xy_dist = np.linalg.norm(world_points[:, :2] - top_xy_center.reshape(1, 2), axis=1)
    core_points = world_points[xy_dist <= core_radius]
    if len(core_points) < 32:
        core_points = world_points

    descend = float(np.clip(obj_height * 0.35, 0.02, 0.05))
    target_z = float(np.clip(z_max - descend, z_min + 0.015, z_max - 0.005))
    if len(core_points) >= 32:
        target_z = float(np.clip(np.quantile(core_points[:, 2], 0.65), z_min + 0.015, z_max - 0.005))

    translation_world = np.array([top_xy_center[0], top_xy_center[1], target_z], dtype=np.float64)
    translation_cam, rotation_cam = _transform_pose_world_to_camera(translation_world, rotation_world, T_wc)

    grasp = Grasp(
        1.0,
        grasp_width,
        0.02,
        0.03,
        rotation_cam,
        translation_cam,
        -1,
    )
    return grasp, {
        "world_translation": translation_world,
        "world_rotation": rotation_world,
        "object_height_m": obj_height,
        "estimated_width_m": estimated_width,
        "finger_span_m": finger_span,
        "grasp_width_m": grasp_width,
        "top_xy_center": top_xy_center,
        "core_radius_m": core_radius,
        "z_min": z_min,
        "z_max": z_max,
        "target_z": target_z,
    }


def _analytic_grasp_has_valid_tcp_mapping(env: UR5GraspEnv, T_wc: sm.SE3, grasp) -> bool:
    grasp_translation = np.asarray(grasp.translation, dtype=np.float64)
    grasp_rotation = _project_to_rotation_matrix(np.asarray(grasp.rotation_matrix, dtype=np.float64))
    q_pre = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
    grasp_width = float(grasp.width)
    approach_distance = float(np.clip(grasp_width + 0.05, 0.08, 0.15))
    mapping = _select_tcp_mapping(
        env.robot,
        T_wc,
        grasp_translation,
        grasp_rotation,
        q_pre,
        approach_distance,
    )
    return mapping is not None


def _build_pose_from_grasp(
    T_wc: sm.SE3,
    grasp_translation: np.ndarray,
    grasp_rotation: np.ndarray,
    mapping: dict,
) -> tuple[sm.SE3, np.ndarray]:
    tcp_rotation = _project_to_rotation_matrix(grasp_rotation @ mapping["R_grasp_to_tcp"])
    T_co = sm.SE3.Trans(grasp_translation) * sm.SE3(sm.SO3(tcp_rotation, check=False))
    return T_wc * T_co, tcp_rotation


def _evaluate_tcp_mapping(
    robot,
    T_wc: sm.SE3,
    grasp_translation: np.ndarray,
    grasp_rotation: np.ndarray,
    mapping: dict,
    q_seed: np.ndarray,
    approach_distance: float,
) -> dict:
    original_q = np.asarray(robot.get_joint(), dtype=np.float64)
    T_grasp, tcp_rotation = _build_pose_from_grasp(T_wc, grasp_translation, grasp_rotation, mapping)
    retreat_axis_tcp = np.asarray(mapping["retreat_axis_tcp"], dtype=np.float64)
    retreat_axis_world = np.asarray(T_grasp.R @ retreat_axis_tcp, dtype=np.float64)
    retreat_axis_world /= max(float(np.linalg.norm(retreat_axis_world)), 1e-12)
    retreat_alignment = float(np.dot(retreat_axis_world, WORLD_VERTICAL))
    retreat_angle_deg = float(np.degrees(np.arccos(np.clip(retreat_alignment, -1.0, 1.0))))

    offset = retreat_axis_tcp * float(approach_distance)
    T_pregrasp = T_grasp * sm.SE3(offset[0], offset[1], offset[2])
    pregrasp_above = bool(float(T_pregrasp.t[2]) > float(T_grasp.t[2]))

    ik_success = False
    q_pregrasp = None
    q_grasp = None
    joint_cost = float("inf")
    wrist_cost = float("inf")
    try:
        robot.set_joint(np.asarray(q_seed, dtype=np.float64))
        q_pregrasp_candidate = robot.ikine(T_pregrasp)
        if q_pregrasp_candidate is not None and len(q_pregrasp_candidate) == 6:
            q_pregrasp_candidate = np.asarray(q_pregrasp_candidate, dtype=np.float64)
            robot.set_joint(q_pregrasp_candidate)
            q_grasp_candidate = robot.ikine(T_grasp)
            if q_grasp_candidate is not None and len(q_grasp_candidate) == 6:
                q_grasp_candidate = np.asarray(q_grasp_candidate, dtype=np.float64)
                ik_success = True
                q_pregrasp = q_pregrasp_candidate
                q_grasp = q_grasp_candidate
                delta_pre = _wrapped_joint_delta(q_seed, q_pregrasp)
                delta_grasp = _wrapped_joint_delta(q_pregrasp, q_grasp)
                joint_cost = float(np.linalg.norm(delta_pre) + np.linalg.norm(delta_grasp))
                wrist_cost = float(np.sum(np.abs(delta_pre[3:])) + np.sum(np.abs(delta_grasp[3:])))
    finally:
        robot.set_joint(original_q)

    return {
        "name": mapping["name"],
        "R_grasp_to_tcp": mapping["R_grasp_to_tcp"],
        "tcp_rotation": tcp_rotation,
        "retreat_axis_tcp": retreat_axis_tcp,
        "retreat_axis_world": retreat_axis_world,
        "retreat_alignment": retreat_alignment,
        "retreat_angle_deg": retreat_angle_deg,
        "T_grasp": T_grasp,
        "T_pregrasp": T_pregrasp,
        "pregrasp_above": pregrasp_above,
        "ik_success": ik_success,
        "q_pregrasp": q_pregrasp,
        "q_grasp": q_grasp,
        "joint_cost": joint_cost,
        "wrist_cost": wrist_cost,
    }


def _select_tcp_mapping(
    robot,
    T_wc: sm.SE3,
    grasp_translation: np.ndarray,
    grasp_rotation: np.ndarray,
    q_seed: np.ndarray,
    approach_distance: float,
) -> dict | None:
    evaluations = [
        _evaluate_tcp_mapping(
            robot,
            T_wc,
            grasp_translation,
            grasp_rotation,
            mapping,
            q_seed,
            approach_distance,
        )
        for mapping in TCP_FRAME_CANDIDATES
    ]

    for item in evaluations:
        print(
            "[INFO] tcp candidate:",
            item["name"],
            "pregrasp_above:",
            item["pregrasp_above"],
            "retreat_angle_deg:",
            round(item["retreat_angle_deg"], 3),
            "ik_success:",
            item["ik_success"],
            "joint_cost:",
            None if not np.isfinite(item["joint_cost"]) else round(item["joint_cost"], 4),
            "wrist_cost:",
            None if not np.isfinite(item["wrist_cost"]) else round(item["wrist_cost"], 4),
        )

    valid = [item for item in evaluations if item["pregrasp_above"] and item["ik_success"]]
    if len(valid) == 0:
        print("[Warning] No TCP mapping candidate satisfies top-down pregrasp and IK constraints.")
        return None

    valid.sort(
        key=lambda item: (
            item["retreat_angle_deg"],
            item["joint_cost"],
            item["wrist_cost"],
        )
    )
    best = valid[0]
    print("[INFO] selected tcp mapping:", best["name"])
    return best


def execute_grasp_completion(env, gg):
    if gg is None or len(gg) == 0:
        print("[Warning] Empty grasp group, skipping execution.")
        return

    robot = env.robot
    action = np.zeros(7)
    T_wc = _camera_pose_world_from_env(env)

    grasp_translation = gg.translations[0]
    grasp_rotation = _project_to_rotation_matrix(np.asarray(gg.rotation_matrices[0], dtype=np.float64))

    q0 = robot.get_joint()
    q_pre = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
    grasp_width = float(gg.width[0]) if hasattr(gg, "width") and len(gg.width) > 0 else 0.08
    approach_distance = float(np.clip(grasp_width + 0.05, 0.08, 0.15))
    print("approach_distance:", approach_distance)

    best_mapping = _select_tcp_mapping(
        robot,
        T_wc,
        grasp_translation,
        grasp_rotation,
        q_pre,
        approach_distance,
    )
    if best_mapping is None:
        return

    T_wo = best_mapping["T_grasp"]
    T_pregrasp = best_mapping["T_pregrasp"]
    tcp_rotation = best_mapping["tcp_rotation"]
    print("T_wo:", T_wo)
    print("gg.translation:", np.asarray(grasp_translation, dtype=float))
    print("gg.rotation_matrix:\n", grasp_rotation)
    print("tcp_rotation_from_grasp:\n", tcp_rotation)
    print("tcp_x_world:", np.asarray(T_wo.R[:, 0], dtype=float))
    print("tcp_y_world:", np.asarray(T_wo.R[:, 1], dtype=float))
    print("tcp_z_world:", np.asarray(T_wo.R[:, 2], dtype=float))
    print("retreat_axis_world:", np.asarray(best_mapping["retreat_axis_world"], dtype=float))
    print("T_pregrasp(T2):", T_pregrasp)
    print("T_grasp(T3):", T_wo)

    parameter0 = JointParameter(q0, q_pre)
    velocity_parameter0 = QuinticVelocityParameter(1.0)
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner1 = TrajectoryPlanner(trajectory_parameter0)
    _run_trajectory(env, robot, action, planner1, 1.0)

    T_start = robot.get_cartesian()
    _run_cartesian_move(env, robot, T_start, T_pregrasp, 1.0, action)
    _run_cartesian_move(env, robot, T_pregrasp, T_wo, 1.0, action)

    for _ in range(60):
        env.step(action)

    for _ in range(1000):
        action[-1] += 0.2
        action[-1] = np.min([action[-1], 255])
        env.step(action)

    T_lift = sm.SE3.Trans(0.0, 0.0, 0.3) * T_wo
    _run_cartesian_move(env, robot, T_wo, T_lift, 1.0, action)

    T_place = sm.SE3.Trans(0.3, 0.3, T_lift.t[2]) * sm.SE3(sm.SO3(T_lift.R))
    _run_cartesian_move(env, robot, T_lift, T_place, 1.0, action)

    T_lower = sm.SE3.Trans(0.0, 0.0, -0.1) * T_place
    _run_cartesian_move(env, robot, T_place, T_lower, 1.0, action)

    for _ in range(1000):
        action[-1] -= 0.2
        action[-1] = np.max([action[-1], 0])
        env.step(action)


def _generate_topdown_grasp_candidates(
    env: UR5GraspEnv,
    end_points: dict,
    cloud: o3d.geometry.PointCloud,
    *,
    angle_threshold_deg: float = 10.0,
    keep_top_k: int = 20,
    visual: bool = False,
):
    net = base_main.GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    net.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load("./logs/log_rs/checkpoint-rs.tar")
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = base_main.pred_decode(end_points)
    gg = base_main.GraspGroup(grasp_preds[0].detach().cpu().numpy())

    collision_thresh = 0.005
    if collision_thresh > 0:
        mfcdetector = base_main.ModelFreeCollisionDetector(
            np.asarray(cloud.points),
            voxel_size=0.005,
        )
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        gg = gg[~collision_mask]

    gg.nms().sort_by_score()
    all_grasps = list(gg)
    if len(all_grasps) == 0:
        return base_main.GraspGroup()

    camera_rotation_world_from_cam = camera_rotation_world_from_mujoco(
        env.mj_model,
        env.mj_data,
        camera_name="cam1",
        align_rx_pi=True,
    )
    top_grasps = filter_grasps_by_world_vertical(
        all_grasps,
        camera_rotation_world_from_cam,
        world_vertical=WORLD_VERTICAL,
        angle_threshold_deg=angle_threshold_deg,
        keep_top_k=keep_top_k,
    )

    if len(top_grasps) == 0:
        scored = []
        for grasp in all_grasps:
            tilt_deg = grasp_approach_tilt_deg(
                grasp,
                camera_rotation_world_from_cam,
                world_axis=WORLD_VERTICAL,
            )
            scored.append((tilt_deg, float(grasp.score), grasp))
        scored.sort(key=lambda item: (item[0], item[1]))
        if len(scored) == 0:
            print("[Warning] No grasps available after world tilt fallback.")
            return []
        print(
            "[WARN] No exact top-down grasp found; using closest world-vertical grasp.",
            "best_tilt_deg:",
            round(float(scored[0][0]), 3),
        )
        return [grasp for _, _, grasp in scored[:keep_top_k]]

    print(
        "[INFO] top-down grasp candidates:",
        len(top_grasps),
    )
    if visual:
        gg_preview = base_main.GraspGroup()
        for grasp in top_grasps:
            gg_preview.add(grasp)
        grippers = gg_preview.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
    return top_grasps


def run_completion_grasp_loop(env: UR5GraspEnv, *, iterations: int = 4) -> None:
    for _ in range(iterations):
        for _ in range(500):
            env.step()

        base_main.pasue_1(env)
        imgs = env.render(1)
        color_img = imgs["img"]
        depth_img = imgs["depth"]

        color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("color_img_path.jpg", color_img)
        cv2.imwrite("color_img_depth.jpg", depth_img)

        target_class = input("\n===============\nEnter class name: ").strip()
        if not target_class:
            print("[Warning] Empty class name, skipping this round.")
            continue

        mask_img_path = segment_image_ground("color_img_path.jpg", target_class)
        if mask_img_path is None or np.count_nonzero(mask_img_path) == 0:
            print("[Warning] Empty mask, skipping this round.")
            continue

        _, raw_cloud_o3d = base_main.get_and_process_data(color_img, depth_img, mask_img_path)
        raw_points = np.asarray(raw_cloud_o3d.points, dtype=np.float64)
        raw_colors = np.asarray(raw_cloud_o3d.colors, dtype=np.float64) if raw_cloud_o3d.has_colors() else None

        try:
            completion = complete_point_cloud(raw_points, raw_colors, COMPLETION_CONFIG)
        except Exception as exc:  # noqa: BLE001
            print(f"[Warning] Point cloud completion failed, falling back to raw cloud: {exc}")
            completed_points = raw_points
            completed_colors = raw_colors
            completion_report = None
            completion_result = None
        else:
            completed_points = completion.completed_points
            completed_colors = completion.completed_colors
            completion_report = completion.report
            completion_result = completion

        print("raw_points:", len(raw_points))
        print("completed_points:", len(completed_points))
        if completion_report is not None:
            print("[INFO] completion counts:", completion_report["counts"])
            print("[INFO] completion geometry:", completion_report["geometry"])

        _show_side_by_side(raw_points, completed_points)

        completed_cloud_o3d = _build_open3d_cloud(completed_points, completed_colors)
        completed_end_points = _build_completed_end_points(completed_points)
        T_wc = _camera_pose_world_from_env(env)
        top_surface_info = _extract_top_surface_info(completed_points, T_wc)
        if top_surface_info is not None:
            print("[INFO] world top points:", top_surface_info["count"])
            print("[INFO] world top z_max:", round(float(top_surface_info["z_max"]), 4))
            print("[INFO] world top band z_min:", round(float(top_surface_info["z_min_band"]), 4))
            print("[INFO] world top xy center:", np.round(top_surface_info["top_xy_center"], 4).tolist())
        else:
            print("[INFO] world top points: unavailable")

        selected_grasp = None
        selection_info = None

        analytic_grasp, analytic_info = _build_analytic_top_grasp(completed_points, T_wc, top_surface_info)
        if analytic_grasp is not None:
            print(
                "[INFO] analytic top grasp:",
                {
                    "world_translation": np.round(analytic_info["world_translation"], 4).tolist(),
                    "grasp_width_m": round(float(analytic_info["grasp_width_m"]), 4),
                    "target_z": round(float(analytic_info["target_z"]), 4),
                    "object_height_m": round(float(analytic_info["object_height_m"]), 4),
                    "core_radius_m": round(float(analytic_info["core_radius_m"]), 4),
                },
            )
            if _analytic_grasp_has_valid_tcp_mapping(env, T_wc, analytic_grasp):
                selected_grasp = analytic_grasp
                selection_info = {
                    "selection_source": "analytic world-top grasp",
                    "grasp_width_m": float(analytic_info["grasp_width_m"]),
                    "target_z": float(analytic_info["target_z"]),
                }
            else:
                print("[WARN] Analytic world-top grasp is not executable; falling back to GraspNet candidates.")
        else:
            print("[WARN] Analytic world-top grasp unavailable:", analytic_info["reason"])

        if selected_grasp is None:
            grasp_candidates = _generate_topdown_grasp_candidates(
                env,
                completed_end_points,
                completed_cloud_o3d,
                angle_threshold_deg=10.0,
                keep_top_k=20,
                visual=False,
            )
            if grasp_candidates is None or len(grasp_candidates) == 0:
                print("[Warning] Empty grasp result, skipping this round.")
                continue

            for idx, grasp in enumerate(grasp_candidates[:10]):
                stats = _score_grasp_for_top_surface(grasp, T_wc, top_surface_info)
                print(
                    "[INFO] grasp candidate:",
                    idx,
                    "score:",
                    round(float(grasp.score), 4),
                    "world_top_hit:",
                    stats["world_top_hit"],
                    "world_top_inner_hit:",
                    stats["world_top_inner_hit"],
                    "world_top_distance_m:",
                    None if not np.isfinite(stats["world_top_distance_m"]) else round(stats["world_top_distance_m"], 4),
                    "world_top_z_gap_m:",
                    None if not np.isfinite(stats["world_top_z_gap_m"]) else round(stats["world_top_z_gap_m"], 4),
                    "world_top_center_offset_m:",
                    None if not np.isfinite(stats["world_top_center_offset_m"]) else round(stats["world_top_center_offset_m"], 4),
                )

            selected_grasp, selection_info = _select_surface_preferred_grasp(
                grasp_candidates,
                T_wc,
                top_surface_info,
            )
            if selected_grasp is None:
                print("[Warning] No grasp selected after top-surface ranking.")
                continue

        print("[INFO] selected grasp info:", selection_info)

        selected_grasp = _apply_static_grasp_translation_correction(
            selected_grasp,
            T_wc,
            target_class=target_class,
        )

        gg = _single_grasp_group(selected_grasp)
        print("gg:", gg)

        execute_grasp_completion(env, gg)


if __name__ == "__main__":
    env = UR5GraspEnv()
    try:
        env.reset()
        run_completion_grasp_loop(env, iterations=4)
    finally:
        env.close()
