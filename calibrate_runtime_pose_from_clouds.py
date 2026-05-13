from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from camera_pose_mujoco import convert_raw_rotation_to_mujoco, rotation_matrix_from_euler_xyz_deg
from pointcloud_v2 import CAMERA_EULER_DEG, PointCloudGenerator, pos


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
DEFAULT_JSON_PATH = ROOT_DIR / "runtime_pose_calibration.json"
WORLD_Z_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)
LEFT_INTRINSICS = {
    "fx": 1129.8136,
    "fy": 1128.6075,
    "cx": 961.0022,
    "cy": 546.8298,
    "width": 1920,
    "height": 1080,
}
CAM1_DEBUG_DIR = OUTPUT_DIR / "cam1_alignment_debug"
DEFAULT_SCENE_XML = ROOT_DIR / "manipulator_grasp" / "assets" / "scenes" / "apple_pear_runtime.xml"
CAM1_SILHOUETTE_MAX_FACES = 4000
CAM1_ENABLE_SILHOUETTE = os.environ.get("CAM1_ENABLE_SILHOUETTE", "0").strip() == "1"


RIGHT_WORKSPACE_CROP_PLY = OUTPUT_DIR / "right1_background_workspace_crop.ply"
RIGHT_BOARD_MAX_ABS_Z = 0.2
GROUND_MIN_ABS_Z = 0.85
PLANE_SEGMENT_DISTANCE_M = 0.01
PLANE_SEGMENT_MAX_PLANES = 8
PLANE_SEGMENT_MIN_POINTS = 20000


@dataclass
class CameraPoseResult:
    rotation_matrix_mj_from_cam: list[list[float]]
    rotation_matrix_world_from_cam: list[list[float]]
    translation_mj: list[float]
    roboticarm_center_camera: list[float]
    quat_wxyz: list[float]


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vec / norm


def _load_point_cloud(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 32:
        raise ValueError(f"Invalid point cloud: {path}")
    return points


def _load_raw_points(flag: str, object_name: str) -> np.ndarray:
    path = OUTPUT_DIR / f"raw_{flag}_{object_name}.npy"
    if path.exists():
        points = np.load(path)
        if points.ndim == 2 and points.shape[1] == 3 and len(points) >= 8:
            return np.asarray(points, dtype=float)
    return _generate_raw_points(flag, object_name)


def _generator_for(flag: str, rx: float, ry: float, rz: float) -> PointCloudGenerator:
    camera_to_tcp_pose = [0, 0, 0, rx / 180.0 * np.pi, ry / 180.0 * np.pi, rz / 180.0 * np.pi]
    tcp_pose = [0, 0, 0, 0, 0, 0]
    if flag == "left":
        return PointCloudGenerator(
            fx=1129.8136,
            fy=1128.6075,
            cx=961.0022,
            cy=546.8298,
            tcp_pose=tcp_pose,
            camera_to_tcp_pose=camera_to_tcp_pose,
            visualize=False,
            save_point_cloud=False,
        )
    if flag == "right":
        return PointCloudGenerator(
            fx=1126.8856,
            fy=1126.4037,
            cx=954.9412,
            cy=536.3848,
            tcp_pose=tcp_pose,
            camera_to_tcp_pose=camera_to_tcp_pose,
            visualize=False,
            save_point_cloud=False,
        )
    raise ValueError(f"Unsupported camera flag: {flag}")


def _generate_raw_points(flag: str, object_name: str) -> np.ndarray:
    rx, ry, rz = CAMERA_EULER_DEG[flag]
    generator = _generator_for(flag, rx, ry, rz)
    color_img = PointCloudGenerator.read_image_safely(f"inputs/c{flag}001.png", is_depth=False)
    depth_img = PointCloudGenerator.read_image_safely(f"inputs/d{flag}001.png", is_depth=True)
    mask_path = f"inputs/{flag}_mask_{object_name}.png"
    response = generator.generate_point_cloud(
        color_image_ori=color_img,
        depth_image_ori=depth_img,
        mask_path=mask_path,
        use_mask_auto=True,
        downsample_scale=1.0,
        objects=object_name,
        flag=flag,
        type1="normal",
    )
    if response.get("state") != "success" or response.get("point_cloud_cam") is None:
        raise ValueError(f"Unable to generate raw point cloud for {flag}:{object_name}")
    points = np.asarray(response["point_cloud_cam"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ValueError(f"Invalid generated raw point cloud for {flag}:{object_name}")
    return points


def _segment_planes(points: np.ndarray) -> list[dict[str, Any]]:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    remaining = cloud
    planes: list[dict[str, Any]] = []

    for index in range(PLANE_SEGMENT_MAX_PLANES):
        if len(remaining.points) < PLANE_SEGMENT_MIN_POINTS:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=PLANE_SEGMENT_DISTANCE_M,
            ransac_n=3,
            num_iterations=3000,
        )
        if len(inliers) < PLANE_SEGMENT_MIN_POINTS:
            break

        inlier_points = np.asarray(remaining.points)[np.asarray(inliers, dtype=int)]
        normal = _normalize(np.asarray(plane_model[:3], dtype=float))
        bbox_min = np.min(inlier_points, axis=0)
        bbox_max = np.max(inlier_points, axis=0)
        extents = bbox_max - bbox_min
        planes.append({
            "index": int(index),
            "count": int(len(inliers)),
            "normal": normal,
            "center": inlier_points.mean(axis=0),
            "d": float(plane_model[3]),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "extents": extents,
        })
        remaining = remaining.select_by_index(inliers, invert=True)
    return planes


def _dominant_plane_normal(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    planes = _segment_planes(points)
    if not planes:
        raise ValueError("Failed to segment any plane from background point cloud.")

    candidates = [plane for plane in planes if abs(float(plane["normal"][2])) >= GROUND_MIN_ABS_Z]
    if not candidates:
        raise ValueError("Failed to find a near-horizontal ground plane in right background.")

    ground = max(candidates, key=lambda item: item["count"])
    normal = np.asarray(ground["normal"], dtype=float)
    if normal[2] < 0.0:
        normal = -normal

    metadata = {
        "plane_normal_raw": normal.tolist(),
        "plane_center_camera": np.asarray(ground["center"], dtype=float).tolist(),
        "plane_inlier_ratio": float(ground["count"] / len(points)),
        "ground_plane_index": int(ground["index"]),
        "ground_plane_count": int(ground["count"]),
        "segmented_plane_count": int(len(planes)),
    }
    return normal, metadata


def _board_x_axis_from_planes(points: np.ndarray, world_z_axis: np.ndarray, right_relative_object_dirs: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    planes = _segment_planes(points)
    if not planes:
        raise ValueError("Failed to segment planes from right workspace crop.")

    candidates = []
    for plane in planes:
        normal = np.asarray(plane["normal"], dtype=float)
        projected = normal - float(normal @ world_z_axis) * world_z_axis
        projected_norm = float(np.linalg.norm(projected))
        if projected_norm <= 1e-9:
            continue
        if abs(float(normal @ world_z_axis)) > RIGHT_BOARD_MAX_ABS_Z:
            continue
        candidates.append((plane, projected / projected_norm))

    if not candidates:
        raise ValueError("Failed to find a vertical board plane in right workspace crop.")

    board_plane, x_axis = max(candidates, key=lambda item: item[0]["count"])
    avg_dir = np.mean(np.stack(right_relative_object_dirs, axis=0), axis=0)
    avg_dir = avg_dir - float(avg_dir @ world_z_axis) * world_z_axis
    avg_dir = _normalize(avg_dir)
    if float(x_axis @ avg_dir) < 0.0:
        x_axis = -x_axis

    metadata = {
        "board_plane_index": int(board_plane["index"]),
        "board_plane_count": int(board_plane["count"]),
        "board_plane_normal_raw": np.asarray(board_plane["normal"], dtype=float).tolist(),
        "board_plane_center_camera": np.asarray(board_plane["center"], dtype=float).tolist(),
        "board_plane_bbox_min": np.asarray(board_plane["bbox_min"], dtype=float).tolist(),
        "board_plane_bbox_max": np.asarray(board_plane["bbox_max"], dtype=float).tolist(),
        "board_plane_extents": np.asarray(board_plane["extents"], dtype=float).tolist(),
    }
    return x_axis, metadata


def _relative_center(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float).mean(axis=0)


def _roboticarm_reference_point(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    return np.array(
        [
            (x_min + x_max) / 2.0,
            (y_min + y_max) / 2.0,
            z_min,
        ],
        dtype=float,
    )


def _choose_world_x_axis(plane_normal: np.ndarray, right_relative_object_dirs: list[np.ndarray]) -> np.ndarray:
    horizontal = plane_normal.copy()
    horizontal[2] = 0.0
    horizontal = _normalize(horizontal)
    avg_dir = np.mean(np.stack(right_relative_object_dirs, axis=0), axis=0)
    avg_dir[2] = 0.0
    if float(horizontal @ avg_dir) < 0.0:
        horizontal = -horizontal
    return horizontal


def _current_runtime_rotation(flag: str) -> np.ndarray:
    rx, ry, rz = CAMERA_EULER_DEG[flag]
    pose = convert_raw_rotation_to_mujoco(rotation_matrix_from_euler_xyz_deg(rx, ry, rz), flag)
    return np.asarray(pose.rotation_matrix, dtype=float)


def _camera_pose_from_arm_center(rotation_mj_from_cam: np.ndarray, arm_center_camera: np.ndarray) -> CameraPoseResult:
    translation = -rotation_mj_from_cam @ arm_center_camera
    quat_xyzw = Rotation.from_matrix(rotation_mj_from_cam).as_quat()
    rotation_list = rotation_mj_from_cam.tolist()
    return CameraPoseResult(
        rotation_matrix_mj_from_cam=rotation_list,
        rotation_matrix_world_from_cam=rotation_list,
        translation_mj=translation.tolist(),
        roboticarm_center_camera=arm_center_camera.tolist(),
        quat_wxyz=[
            float(quat_xyzw[3]),
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
        ],
    )


def _load_mask_center_and_axis(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {path}")
    ys, xs = np.where(mask > 0)
    if len(xs) < 8:
        raise ValueError(f"Mask has too few pixels: {path}")
    center = np.array([xs.mean(), ys.mean()], dtype=float)
    coords = np.column_stack([xs, ys]).astype(float)
    coords -= coords.mean(axis=0, keepdims=True)
    eigvals, eigvecs = np.linalg.eigh(np.cov(coords.T))
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    return center, axis


def _load_mask_line_features(path: Path) -> dict[str, np.ndarray]:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {path}")
    ys, xs = np.where(mask > 0)
    if len(xs) < 8:
        raise ValueError(f"Mask has too few pixels: {path}")

    center = np.array([xs.mean(), ys.mean()], dtype=float)
    coords = np.column_stack([xs, ys]).astype(float)
    top_y = int(ys.min())
    bottom_y = int(ys.max())
    band = 3
    top_band = xs[ys <= top_y + band]
    bottom_band = xs[ys >= bottom_y - band]
    if len(top_band) == 0 or len(bottom_band) == 0:
        raise ValueError(f"Mask has invalid top/bottom bands: {path}")
    top_uv = np.array([float(top_band.mean()), float(top_y)], dtype=float)
    base_uv = np.array([float(bottom_band.mean()), float(bottom_y)], dtype=float)
    axis = top_uv - base_uv
    axis = axis / max(np.linalg.norm(axis), 1e-12)

    return {
        "center_uv": center,
        "axis_uv": axis,
        "base_uv": base_uv,
        "top_uv": top_uv,
    }


def _project_world_points_to_left_image(points_world: np.ndarray, rotation_mj_from_cam: np.ndarray, translation_mj: np.ndarray) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=float)
    rotation = np.asarray(rotation_mj_from_cam, dtype=float)
    translation = np.asarray(translation_mj, dtype=float)
    camera_points = (rotation.T @ (points_world - translation).T).T
    result = np.full((len(points_world), 2), np.nan, dtype=float)
    valid = camera_points[:, 2] > 1e-6
    if not np.any(valid):
        return result
    pts = camera_points[valid]
    result[valid] = np.column_stack([
        LEFT_INTRINSICS["fx"] * pts[:, 0] / pts[:, 2] + LEFT_INTRINSICS["cx"],
        LEFT_INTRINSICS["fy"] * pts[:, 1] / pts[:, 2] + LEFT_INTRINSICS["cy"],
    ])
    return result


def _rigid_transform_from_correspondences(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.shape != target.shape or source.shape[0] < 3 or source.shape[1] != 3:
        raise ValueError("Need at least 3 corresponding 3D points with matching shapes.")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    h = source_zero.T @ target_zero
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = target_center - r @ source_center
    return r, t


def _solve_pnp_camera_pose(
    anchors_world: np.ndarray,
    anchors_image: np.ndarray,
    initial_rotation_mj_from_cam: np.ndarray,
    initial_translation_mj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(
        [
            [LEFT_INTRINSICS["fx"], 0.0, LEFT_INTRINSICS["cx"]],
            [0.0, LEFT_INTRINSICS["fy"], LEFT_INTRINSICS["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    dist_coeffs = np.zeros((4, 1), dtype=float)
    object_points = np.asarray(anchors_world, dtype=np.float64)
    image_points = np.asarray(anchors_image, dtype=np.float64)

    initial_rotation = np.asarray(initial_rotation_mj_from_cam, dtype=float)
    initial_translation = np.asarray(initial_translation_mj, dtype=float)
    initial_rvec, _ = cv2.Rodrigues(initial_rotation.T)
    initial_tvec = (-initial_rotation.T @ initial_translation.reshape(3, 1)).astype(np.float64)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        rvec=initial_rvec.astype(np.float64),
        tvec=initial_tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            raise ValueError("solvePnP failed for cam1 alignment.")

    rotation_cam_from_world, _ = cv2.Rodrigues(rvec)
    rotation_mj_from_cam = rotation_cam_from_world.T
    translation_mj = (-rotation_mj_from_cam @ tvec.reshape(3)).astype(float)
    return rotation_mj_from_cam, translation_mj


def _load_scene_object_quats(scene_path: Path = DEFAULT_SCENE_XML) -> dict[str, np.ndarray]:
    if not scene_path.exists():
        return {}
    tree = ET.parse(scene_path)
    root = tree.getroot()
    quats: dict[str, np.ndarray] = {}
    for object_name in ("apple", "pear"):
        body = root.find(f".//body[@name='{object_name}0']")
        if body is None:
            continue
        quat_attr = body.get("quat")
        if not quat_attr:
            continue
        values = np.array([float(v) for v in quat_attr.split()], dtype=float)
        if values.shape != (4,):
            continue
        quat_xyzw = np.array([values[1], values[2], values[3], values[0]], dtype=float)
        quats[object_name] = Rotation.from_quat(quat_xyzw).as_matrix()
    return quats


def _load_mesh_geometry(object_name: str) -> tuple[np.ndarray, np.ndarray]:
    mesh_path = ROOT_DIR / "manipulator_grasp" / "assets" / "fruit" / "stl" / f"{object_name}.stl"
    mesh = trimesh.load(mesh_path, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Invalid mesh faces for {object_name}: {mesh_path}")
    if len(faces) > CAM1_SILHOUETTE_MAX_FACES:
        stride = int(np.ceil(len(faces) / CAM1_SILHOUETTE_MAX_FACES))
        faces = faces[::stride]
        print(
            f"[INFO] cam1 silhouette mesh decimated: {object_name} faces {len(np.asarray(mesh.faces, dtype=int))} -> {len(faces)}",
            flush=True,
        )
    return vertices, faces


def _render_mesh_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    rotation_world: np.ndarray,
    translation_world: np.ndarray,
    rotation_mj_from_cam: np.ndarray,
    translation_mj: np.ndarray,
) -> np.ndarray:
    transformed = (rotation_world @ vertices.T).T + translation_world
    camera_points = (rotation_mj_from_cam.T @ (transformed - translation_mj).T).T
    mask = np.zeros((LEFT_INTRINSICS["height"], LEFT_INTRINSICS["width"]), dtype=np.uint8)
    for face in faces:
        tri = camera_points[face]
        if np.any(tri[:, 2] <= 1e-6):
            continue
        proj = np.column_stack(
            [
                LEFT_INTRINSICS["fx"] * tri[:, 0] / tri[:, 2] + LEFT_INTRINSICS["cx"],
                LEFT_INTRINSICS["fy"] * tri[:, 1] / tri[:, 2] + LEFT_INTRINSICS["cy"],
            ]
        )
        if np.all(
            (proj[:, 0] < 0)
            | (proj[:, 0] >= LEFT_INTRINSICS["width"])
            | (proj[:, 1] < 0)
            | (proj[:, 1] >= LEFT_INTRINSICS["height"])
        ):
            continue
        cv2.fillConvexPoly(mask, np.round(proj).astype(np.int32), 1)
    return mask


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def _refine_cam1_pose(
    base_rotation_mj_from_cam: np.ndarray,
    base_translation_mj: np.ndarray,
    arm_center_camera: np.ndarray,
    object_positions: dict[str, list[float]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    arm_features = _load_mask_line_features(ROOT_DIR / "inputs" / "left_mask_roboticarm.png")
    apple_center_2d, _ = _load_mask_center_and_axis(ROOT_DIR / "inputs" / "left_mask_apple.png")
    pear_center_2d, _ = _load_mask_center_and_axis(ROOT_DIR / "inputs" / "left_mask_pear.png")

    anchors_world = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.35],
            [0.0, 0.0, 0.75],
            object_positions["apple_world"],
            object_positions["pear_world"],
        ],
        dtype=float,
    )
    anchors_image = np.array(
        [
            arm_features["base_uv"],
            arm_features["center_uv"],
            arm_features["top_uv"],
            apple_center_2d,
            pear_center_2d,
        ],
        dtype=float,
    )
    object_mask_gt = {
        "apple": (cv2.imread(str(ROOT_DIR / "inputs" / "left_mask_apple.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8),
        "pear": (cv2.imread(str(ROOT_DIR / "inputs" / "left_mask_pear.png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8),
    }
    scene_object_quats = _load_scene_object_quats()
    mesh_geometry = {}
    for object_name in ("apple", "pear"):
        if object_name in scene_object_quats:
            mesh_geometry[object_name] = _load_mesh_geometry(object_name)

    source_points = np.array(
        [
            arm_center_camera,
            _relative_center(_load_raw_points("left", "apple")),
            _relative_center(_load_raw_points("left", "pear")),
        ],
        dtype=float,
    )
    target_points = np.array(
        [
            [0.0, 0.0, 0.0],
            object_positions["apple_world"],
            object_positions["pear_world"],
        ],
        dtype=float,
    )
    rigid_rotation, rigid_translation = _rigid_transform_from_correspondences(source_points, target_points)

    def pose_score(rotation: np.ndarray, translation: np.ndarray) -> float:
        projected = _project_world_points_to_left_image(anchors_world, rotation, translation)
        if np.isnan(projected).any():
            return 1e9
        arm_base_uv, arm_mid_uv, arm_top_uv, apple_uv, pear_uv = projected
        arm_vec = arm_top_uv - arm_base_uv
        arm_norm = np.linalg.norm(arm_vec)
        if arm_norm <= 1e-9:
            return 1e9
        arm_dir = arm_vec / arm_norm
        anchor_error = (
            1.8 * np.linalg.norm(arm_base_uv - arm_features["base_uv"])
            + 1.2 * np.linalg.norm(arm_mid_uv - arm_features["center_uv"])
            + 1.5 * np.linalg.norm(arm_top_uv - arm_features["top_uv"])
            + np.linalg.norm(apple_uv - apple_center_2d)
            + np.linalg.norm(pear_uv - pear_center_2d)
        )
        axis_error = 250.0 * (1.0 - abs(float(np.dot(arm_dir, arm_features["axis_uv"]))))
        spacing_error = abs((apple_uv[0] - pear_uv[0]) - (apple_center_2d[0] - pear_center_2d[0]))
        return float(anchor_error + axis_error + 0.4 * spacing_error)

    candidate_starts: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("rigid", rigid_rotation, rigid_translation),
    ]
    try:
        pnp_rotation, pnp_translation = _solve_pnp_camera_pose(
            anchors_world=anchors_world,
            anchors_image=anchors_image,
            initial_rotation_mj_from_cam=base_rotation_mj_from_cam,
            initial_translation_mj=base_translation_mj,
        )
        candidate_starts.append(("pnp", pnp_rotation, pnp_translation))
    except Exception:
        pnp_rotation = None
        pnp_translation = None

    start_name = "rigid"
    best_start_score = 1e18
    for name, rotation_candidate, translation_candidate in candidate_starts:
        score = pose_score(rotation_candidate, translation_candidate)
        if score < best_start_score:
            best_start_score = score
            start_name = name
            base_rotation_mj_from_cam = rotation_candidate
            base_translation_mj = translation_candidate

    def objective(delta: np.ndarray) -> float:
        delta_t = delta[:3]
        delta_r = Rotation.from_euler("xyz", delta[3:], degrees=True).as_matrix()
        rotation = delta_r @ base_rotation_mj_from_cam
        translation = base_translation_mj + delta_t
        pose_error = pose_score(rotation, translation)
        regularization = 5.0 * np.linalg.norm(delta_t) + 2.0 * np.linalg.norm(delta[3:])
        return float(pose_error + regularization)

    print("[INFO] cam1 alignment: starting pose optimize (Powell)", flush=True)
    result = minimize(
        objective,
        x0=np.zeros(6, dtype=float),
        method="Powell",
        options={"maxiter": 120, "xtol": 1e-3, "ftol": 1e-3},
    )
    print(f"[INFO] cam1 alignment: pose optimize done score={float(result.fun):.4f}", flush=True)

    best_delta = result.x
    best_rotation = Rotation.from_euler("xyz", best_delta[3:], degrees=True).as_matrix() @ base_rotation_mj_from_cam
    best_translation = base_translation_mj + best_delta[:3]

    if CAM1_ENABLE_SILHOUETTE and scene_object_quats and mesh_geometry:
        def silhouette_objective(delta: np.ndarray) -> float:
            delta_t = delta[:3]
            delta_r = Rotation.from_euler("xyz", delta[3:], degrees=True).as_matrix()
            rotation = delta_r @ best_rotation
            translation = best_translation + delta_t
            score = objective(np.concatenate([delta_t, delta[3:]]))
            for object_name in ("apple", "pear"):
                if object_name not in scene_object_quats:
                    continue
                vertices, faces = mesh_geometry[object_name]
                rendered = _render_mesh_mask(
                    vertices,
                    faces,
                    scene_object_quats[object_name],
                    np.asarray(object_positions[f"{object_name}_world"], dtype=float),
                    rotation,
                    translation,
                )
                iou = _mask_iou(rendered, object_mask_gt[object_name])
                score += 180.0 * (1.0 - iou)
            return float(score)

        print("[INFO] cam1 alignment: starting silhouette optimize (Powell)", flush=True)
        silhouette_result = minimize(
            silhouette_objective,
            x0=np.zeros(6, dtype=float),
            method="Powell",
            options={"maxiter": 40, "xtol": 1e-3, "ftol": 1e-3},
        )
        print(f"[INFO] cam1 alignment: silhouette optimize done score={float(silhouette_result.fun):.4f}", flush=True)
        if silhouette_result.fun < result.fun:
            best_delta = np.asarray(silhouette_result.x, dtype=float)
            best_rotation = Rotation.from_euler("xyz", best_delta[3:], degrees=True).as_matrix() @ best_rotation
            best_translation = best_translation + best_delta[:3]
            result = silhouette_result
    elif scene_object_quats and mesh_geometry:
        print(
            "[INFO] cam1 alignment: skip silhouette optimize (set CAM1_ENABLE_SILHOUETTE=1 to enable).",
            flush=True,
        )

    projected = _project_world_points_to_left_image(anchors_world, best_rotation, best_translation)
    debug = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "score": float(result.fun),
        "delta_translation": best_delta[:3].tolist(),
        "delta_euler_xyz_deg": best_delta[3:].tolist(),
        "start_pose_source": start_name,
        "start_pose_score": float(best_start_score),
        "rigid_rotation_matrix": rigid_rotation.tolist(),
        "rigid_translation": rigid_translation.tolist(),
        "pnp_rotation_matrix": None if pnp_rotation is None else pnp_rotation.tolist(),
        "pnp_translation": None if pnp_translation is None else pnp_translation.tolist(),
        "used_scene_object_quats": sorted(scene_object_quats.keys()),
        "projected_points": {
            "arm_base_uv": projected[0].tolist(),
            "arm_center_uv": projected[1].tolist(),
            "arm_top_uv": projected[2].tolist(),
            "apple_uv": projected[3].tolist(),
            "pear_uv": projected[4].tolist(),
        },
        "target_points": {
            "arm_center_uv": arm_features["center_uv"].tolist(),
            "arm_axis_uv": arm_features["axis_uv"].tolist(),
            "arm_base_uv": arm_features["base_uv"].tolist(),
            "arm_top_uv": arm_features["top_uv"].tolist(),
            "apple_center_uv": apple_center_2d.tolist(),
            "pear_center_uv": pear_center_2d.tolist(),
        },
    }

    CAM1_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(ROOT_DIR / "inputs" / "cleft001.png"))
    if image is not None:
        labels = ["arm_base", "arm_center", "arm_top", "apple", "pear"]
        colors = [(255, 0, 0), (0, 200, 255), (255, 255, 0), (0, 255, 0), (0, 0, 255)]
        for idx, uv in enumerate(projected):
            if np.any(np.isnan(uv)):
                continue
            pt = tuple(np.round(uv).astype(int))
            cv2.circle(image, pt, 10, colors[idx], -1)
            cv2.putText(image, labels[idx], (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], 2)
        for label, color in [("arm_base_uv", (255, 128, 0)), ("arm_center_uv", (128, 200, 255)), ("arm_top_uv", (128, 255, 0)), ("apple_center_uv", (0, 255, 255)), ("pear_center_uv", (255, 0, 255))]:
            uv = np.asarray(debug["target_points"][label], dtype=float)
            pt = tuple(np.round(uv).astype(int))
            cv2.circle(image, pt, 8, color, 2)
            cv2.putText(image, f"gt_{label}", (pt[0] + 8, pt[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(str(CAM1_DEBUG_DIR / "cam1_alignment_overlay.png"), image)
    (CAM1_DEBUG_DIR / "cam1_pose_alignment.json").write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
    return best_rotation, best_translation, debug


def _fused_world_center(
    camera_results: dict[str, CameraPoseResult],
    object_centers_camera: dict[str, np.ndarray],
) -> list[float]:
    candidates = []
    for flag, center_camera in object_centers_camera.items():
        pose = camera_results[flag]
        rotation = np.asarray(pose.rotation_matrix_mj_from_cam, dtype=float)
        translation = np.asarray(pose.translation_mj, dtype=float)
        candidates.append(rotation @ center_camera + translation)
    return np.mean(np.stack(candidates, axis=0), axis=0).tolist()


def build_runtime_pose_calibration() -> dict[str, Any]:
    right_background = _load_point_cloud(OUTPUT_DIR / "right1_background.ply")
    left_background = _load_point_cloud(OUTPUT_DIR / "left1_background.ply")
    right_workspace = _load_point_cloud(RIGHT_WORKSPACE_CROP_PLY) if RIGHT_WORKSPACE_CROP_PLY.exists() else right_background
    world_z_axis, plane_meta = _dominant_plane_normal(right_workspace)

    arm_centers_camera = {
        "right": _roboticarm_reference_point(_generate_raw_points("right", "roboticarm")),
        "left": _roboticarm_reference_point(_generate_raw_points("left", "roboticarm")),
    }
    apple_centers_camera = {
        "right": _relative_center(_load_raw_points("right", "apple")),
        "left": _relative_center(_load_raw_points("left", "apple")),
    }
    pear_centers_camera = {
        "right": _relative_center(_load_raw_points("right", "pear")),
        "left": _relative_center(_load_raw_points("left", "pear")),
    }
    right_relative_object_dirs = [
        apple_centers_camera["right"] - arm_centers_camera["right"],
        pear_centers_camera["right"] - arm_centers_camera["right"],
    ]

    world_x_in_right_cam, board_meta = _board_x_axis_from_planes(
        right_workspace,
        world_z_axis,
        right_relative_object_dirs,
    )
    world_y_in_right_cam = _normalize(np.cross(world_z_axis, world_x_in_right_cam))
    world_x_in_right_cam = _normalize(np.cross(world_y_in_right_cam, world_z_axis))
    world_basis_in_right_cam = np.column_stack([world_x_in_right_cam, world_y_in_right_cam, world_z_axis])
    rotation_mj_from_cam_right = world_basis_in_right_cam.T

    current_right = _current_runtime_rotation("right")
    current_left = _current_runtime_rotation("left")
    delta_world = rotation_mj_from_cam_right @ current_right.T
    rotation_mj_from_cam_left = delta_world @ current_left

    camera_results = {
        "right": _camera_pose_from_arm_center(rotation_mj_from_cam_right, arm_centers_camera["right"]),
        "left": _camera_pose_from_arm_center(rotation_mj_from_cam_left, arm_centers_camera["left"]),
    }

    apple_world = _fused_world_center(camera_results, apple_centers_camera)
    pear_world = _fused_world_center(camera_results, pear_centers_camera)
    refined_left_rotation, refined_left_translation, cam1_alignment = _refine_cam1_pose(
        rotation_mj_from_cam_left,
        np.asarray(camera_results["left"].translation_mj, dtype=float),
        arm_centers_camera["left"],
        {
            "apple_world": apple_world,
            "pear_world": pear_world,
        },
    )
    refined_left_arm_center_camera = -(refined_left_rotation.T @ refined_left_translation)
    camera_results["left"] = _camera_pose_from_arm_center(refined_left_rotation, refined_left_arm_center_camera)

    apple_world = _fused_world_center(camera_results, apple_centers_camera)
    pear_world = _fused_world_center(camera_results, pear_centers_camera)

    calibration = {
        "world_axes": {
            "x_axis": world_x_in_right_cam.tolist(),
            "y_axis": world_y_in_right_cam.tolist(),
            "z_axis": world_z_axis.tolist(),
            **plane_meta,
            **board_meta,
            "world_frame_source": "right_workspace_ground_plus_board",
            "left_background_point_count": int(len(left_background)),
            "right_background_point_count": int(len(right_background)),
            "right_workspace_point_count": int(len(right_workspace)),
        },
        "camera_poses": {
            flag: asdict(result)
            for flag, result in camera_results.items()
        },
        "cam1_alignment": cam1_alignment,
        "robot_positions": {
            "arm_base_world": [0.0, 0.0, 0.0],
        },
        "object_positions": {
            "apple_world": apple_world,
            "pear_world": pear_world,
        },
        "relative_positions": {
            "apple_minus_arm": apple_world,
            "pear_minus_arm": pear_world,
        },
        "relative_position_source": "fused raw point-cloud centers under unified camera-to-world rotations",
    }
    return calibration


def camera_poses_for_scene(calibration: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    camera_map = {
        "left": "cam1",
        "right": "cam2",
    }
    scene_poses: dict[str, dict[str, list[float]]] = {}
    for flag, scene_name in camera_map.items():
        pose = calibration["camera_poses"][flag]
        scene_poses[scene_name] = {
            "pos": [float(v) for v in pose["translation_mj"]],
            "quat": [float(v) for v in pose["quat_wxyz"]],
        }
    return scene_poses


def object_positions_for_scene(calibration: dict[str, Any]) -> dict[str, list[float]]:
    object_positions = calibration["object_positions"]
    return {
        "apple": [float(v) for v in object_positions["apple_world"]],
        "pear": [float(v) for v in object_positions["pear_world"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate runtime camera and object poses from global/local clouds.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="Path to write the calibration JSON.",
    )
    args = parser.parse_args()

    calibration = build_runtime_pose_calibration()
    args.output.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(calibration["world_axes"], ensure_ascii=False, indent=2))
    print(json.dumps(calibration["camera_poses"], ensure_ascii=False, indent=2))
    print(json.dumps(calibration["relative_positions"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
