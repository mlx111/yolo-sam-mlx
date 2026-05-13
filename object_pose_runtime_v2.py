from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from calibrate_runtime_pose_from_clouds import DEFAULT_JSON_PATH


ROOT_DIR = Path(__file__).resolve().parent
SUPPORTED_OBJECTS = ("apple", "pear")
LEFT_INTRINSICS = {
    "fx": 1129.8136,
    "fy": 1128.6075,
    "cx": 961.0022,
    "cy": 546.8298,
    "width": 1920,
    "height": 1080,
}
OBJECT_LOCAL_CANDIDATES = {
    "apple": [
        ("identity", np.eye(3)),
        ("roll_180", Rotation.from_euler("z", 180.0, degrees=True).as_matrix()),
    ],
    "pear": [
        ("identity", np.eye(3)),
        ("z_pos_90", Rotation.from_euler("z", 90.0, degrees=True).as_matrix()),
        ("z_neg_90", Rotation.from_euler("z", -90.0, degrees=True).as_matrix()),
        ("z_180", Rotation.from_euler("z", 180.0, degrees=True).as_matrix()),
        ("y_180", Rotation.from_euler("y", 180.0, degrees=True).as_matrix()),
    ],
}
REFERENCE_MESH_BY_OBJECT = {
    "apple": ROOT_DIR / "manipulator_grasp" / "assets" / "fruit" / "stl" / "apple1.stl",
    "pear": ROOT_DIR / "manipulator_grasp" / "assets" / "fruit" / "stl" / "pear1.stl",
}
RUNTIME_MESH_BY_OBJECT = {
    "apple": ROOT_DIR / "manipulator_grasp" / "assets" / "fruit" / "stl" / "apple.stl",
    "pear": ROOT_DIR / "manipulator_grasp" / "assets" / "fruit" / "stl" / "pear.stl",
}
DEBUG_DIR = ROOT_DIR / "outputs" / "object_pose_debug_v2"


class ObjectPoseError(RuntimeError):
    pass


def _quat_wxyz(rotation_matrix: np.ndarray) -> list[float]:
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def _raw_left_cloud_path(object_name: str) -> Path:
    return ROOT_DIR / "outputs" / f"raw_left_{object_name}.npy"


def _left_mask_path(object_name: str) -> Path:
    return ROOT_DIR / "inputs" / f"left_mask_{object_name}.png"


def _reference_mesh_path(object_name: str) -> Path:
    path = REFERENCE_MESH_BY_OBJECT[object_name]
    if not path.exists():
        raise ObjectPoseError(f"Missing reference mesh for pose estimation: {path}")
    return path


def _runtime_mesh_path(object_name: str) -> Path:
    path = RUNTIME_MESH_BY_OBJECT[object_name]
    if not path.exists():
        raise ObjectPoseError(f"Missing runtime mesh for pose estimation: {path}")
    return path


def _load_runtime_calibration(calibration_path: str | Path = DEFAULT_JSON_PATH) -> dict:
    path = Path(calibration_path)
    if not path.exists():
        raise ObjectPoseError(f"Missing runtime calibration: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_raw_left_cloud(object_name: str) -> np.ndarray:
    path = _raw_left_cloud_path(object_name)
    if not path.exists():
        raise ObjectPoseError(f"Missing raw left point cloud: {path}")
    points = np.load(path)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ObjectPoseError(f"Invalid raw left point cloud: {path}")
    return np.asarray(points, dtype=float)


def _load_left_mask(object_name: str) -> np.ndarray:
    path = _left_mask_path(object_name)
    if not path.exists():
        raise ObjectPoseError(f"Missing left mask: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ObjectPoseError(f"Failed to load left mask: {path}")
    return (mask > 0).astype(np.uint8)


def _load_mesh(path: Path, label: str) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 8:
        raise ObjectPoseError(f"Invalid {label} mesh vertices: {path}")
    return mesh


def _principal_frame(points: np.ndarray, object_name: str, *, camera_observed: bool) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    major = eigvecs[:, 0]
    middle = eigvecs[:, 1]
    minor = eigvecs[:, 2]

    if camera_observed:
        if minor[2] > 0:
            minor = -minor
    else:
        if minor[2] < 0:
            minor = -minor

    if object_name == "apple":
        if major[0] < 0:
            major = -major
    elif object_name == "pear":
        projection = centered @ major
        if abs(float(projection.min())) > abs(float(projection.max())):
            major = -major
    else:
        raise ObjectPoseError(f"Unsupported object for pose estimation: {object_name}")

    y_axis = np.cross(minor, major)
    if np.linalg.norm(y_axis) < 1e-9:
        y_axis = middle
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-12)

    x_axis = np.cross(y_axis, minor)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / max(np.linalg.norm(z_axis), 1e-12)

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(rotation) < 0:
        rotation[:, 1] = -rotation[:, 1]
    return rotation


def _base_rotation_left_camera(object_name: str, points: np.ndarray) -> np.ndarray:
    observed_frame = _principal_frame(points, object_name, camera_observed=True)
    reference_vertices = np.asarray(
        _load_mesh(_reference_mesh_path(object_name), "reference").vertices, dtype=float
    )
    reference_frame = _principal_frame(reference_vertices, object_name, camera_observed=False)
    return observed_frame @ reference_frame.T


def _render_candidate_mask(
    mesh: trimesh.Trimesh,
    rotation_world: np.ndarray,
    translation_world: np.ndarray,
    calibration: dict,
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    transformed = (rotation_world @ vertices.T).T + translation_world

    rotation = np.asarray(calibration["camera_poses"]["left"]["rotation_matrix_mj_from_cam"], dtype=float)
    translation = np.asarray(calibration["camera_poses"]["left"]["translation_mj"], dtype=float)
    camera_points = (rotation.T @ (transformed - translation).T).T

    fx = LEFT_INTRINSICS["fx"]
    fy = LEFT_INTRINSICS["fy"]
    cx = LEFT_INTRINSICS["cx"]
    cy = LEFT_INTRINSICS["cy"]
    width = LEFT_INTRINSICS["width"]
    height = LEFT_INTRINSICS["height"]
    mask = np.zeros((height, width), dtype=np.uint8)

    for face in faces:
        tri = camera_points[face]
        if np.any(tri[:, 2] <= 1e-6):
            continue
        proj = np.column_stack([
            fx * tri[:, 0] / tri[:, 2] + cx,
            fy * tri[:, 1] / tri[:, 2] + cy,
        ])
        if np.all((proj[:, 0] < 0) | (proj[:, 0] >= width) | (proj[:, 1] < 0) | (proj[:, 1] >= height)):
            continue
        cv2.fillConvexPoly(mask, np.round(proj).astype(np.int32), 1)
    return mask


def _mask_orientation(mask: np.ndarray) -> float | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 8:
        return None
    coords = np.column_stack([xs, ys]).astype(float)
    coords -= coords.mean(axis=0, keepdims=True)
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    return float(np.arctan2(major[1], major[0]))


def _score_candidate(candidate_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = np.logical_and(candidate_mask > 0, gt_mask > 0).sum()
    union = np.logical_or(candidate_mask > 0, gt_mask > 0).sum()
    if union == 0:
        return -1e9
    iou = inter / union

    cand_pts = np.argwhere(candidate_mask > 0)
    gt_pts = np.argwhere(gt_mask > 0)
    if len(cand_pts) == 0 or len(gt_pts) == 0:
        return -1e9
    center_penalty = np.linalg.norm(cand_pts.mean(axis=0) - gt_pts.mean(axis=0)) / max(gt_mask.shape)

    cand_angle = _mask_orientation(candidate_mask)
    gt_angle = _mask_orientation(gt_mask)
    angle_penalty = 0.0
    if cand_angle is not None and gt_angle is not None:
        diff = abs(cand_angle - gt_angle)
        diff = min(diff, abs(np.pi - diff), abs(2 * np.pi - diff))
        angle_penalty = diff / np.pi

    return float(iou - 0.25 * center_penalty - 0.15 * angle_penalty)


def _choose_best_rotation_world(
    object_name: str,
    base_rotation_left_camera: np.ndarray,
    calibration: dict,
) -> np.ndarray:
    gt_mask = _load_left_mask(object_name)
    runtime_mesh = _load_mesh(_runtime_mesh_path(object_name), "runtime")
    translation_world = np.asarray(calibration["object_positions"][f"{object_name}_world"], dtype=float)
    camera_rotation_world = np.asarray(calibration["camera_poses"]["left"]["rotation_matrix_mj_from_cam"], dtype=float)

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    candidate_logs = []
    best_score = -1e9
    best_rotation_world = camera_rotation_world @ base_rotation_left_camera

    print(f"[INFO] object pose v2: reference mesh -> {_reference_mesh_path(object_name)}", flush=True)
    print(f"[INFO] object pose v2: runtime mesh -> {_runtime_mesh_path(object_name)}", flush=True)

    for candidate_name, correction in OBJECT_LOCAL_CANDIDATES[object_name]:
        rotation_left_camera = base_rotation_left_camera @ correction
        rotation_world = camera_rotation_world @ rotation_left_camera
        candidate_mask = _render_candidate_mask(runtime_mesh, rotation_world, translation_world, calibration)
        score = _score_candidate(candidate_mask, gt_mask)
        candidate_logs.append(
            {
                "candidate": candidate_name,
                "score": score,
                "quat_wxyz": _quat_wxyz(rotation_world),
            }
        )
        cv2.imwrite(str(DEBUG_DIR / f"{object_name}_{candidate_name}.png"), candidate_mask * 255)
        if score > best_score:
            best_score = score
            best_rotation_world = rotation_world

    (DEBUG_DIR / f"{object_name}_candidates.json").write_text(
        json.dumps(
            {
                "object_name": object_name,
                "selected_score": best_score,
                "candidates": candidate_logs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return best_rotation_world


def estimate_runtime_object_quats(camera: str = "left", calibration_path: str | Path = DEFAULT_JSON_PATH) -> dict[str, list[float]]:
    if camera != "left":
        raise ObjectPoseError("Current runtime object pose estimation only supports the left camera.")

    calibration = _load_runtime_calibration(calibration_path)
    object_quats: dict[str, list[float]] = {}
    for object_name in SUPPORTED_OBJECTS:
        raw_points = _load_raw_left_cloud(object_name)
        base_rotation_left_camera = _base_rotation_left_camera(object_name, raw_points)
        rotation_world = _choose_best_rotation_world(object_name, base_rotation_left_camera, calibration)
        object_quats[object_name] = _quat_wxyz(rotation_world)
    return object_quats
