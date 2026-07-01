from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

TOOL_DIR = Path(__file__).resolve().parent
UR5E_ROOT = TOOL_DIR.parent
REPO_ROOT = UR5E_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skills.backends.grounded_sam2_backend import GroundedSAM2Segmenter
from skills.backends.yolo_seg_pointcloud_backend import PointCloudGenerator
from ur5e_mujoco.runtime_scene import generate_scene


DEFAULT_COLOR_IMAGE = UR5E_ROOT / "inputs" / "cleft001.png"
DEFAULT_DEPTH_IMAGE = UR5E_ROOT / "inputs" / "dleft001.png"
DEFAULT_OUTPUT_DIR = UR5E_ROOT / "scene"
DEFAULT_SCENE_OUT = DEFAULT_OUTPUT_DIR / "scene.xml"
DEFAULT_REPORT_OUT = DEFAULT_OUTPUT_DIR / "report.json"
DEFAULT_TEMPLATE = UR5E_ROOT / "assets" / "scenes" / "scene2.xml"
DEFAULT_CAMERA = "cam1"
DEFAULT_OBJECTS = ("apple",)
DEFAULT_GROUNDED_SAM2_ROOT = REPO_ROOT / "Grounded-SAM-2"
MJ_CAMERA_FROM_CV = np.diag([1.0, -1.0, -1.0])
FRUIT_STL_DIR = UR5E_ROOT / "assets" / "fruit" / "stl"
OBJECT_LOCAL_CANDIDATES = {
    "apple": (
        ("identity", np.eye(3)),
        ("roll_180", Rotation.from_euler("z", 180.0, degrees=True).as_matrix()),
    ),
    "pear": (
        ("identity", np.eye(3)),
        ("z_pos_90", Rotation.from_euler("z", 90.0, degrees=True).as_matrix()),
        ("z_neg_90", Rotation.from_euler("z", -90.0, degrees=True).as_matrix()),
        ("z_180", Rotation.from_euler("z", 180.0, degrees=True).as_matrix()),
        ("y_180", Rotation.from_euler("y", 180.0, degrees=True).as_matrix()),
    ),
}
FRONT_AXIS_LOCAL_BY_OBJECT = {
    "apple": np.array([0.0, 0.0, -1.0], dtype=float),
    "pear": np.array([0.0, 0.0, -1.0], dtype=float),
}
TOP_AXIS_LOCAL_BY_OBJECT = {
    "apple": np.array([0.0, -1.0, 0.0], dtype=float),
    "pear": np.array([0.0, -1.0, 0.0], dtype=float),
}

UR5E_LEFT_COLOR_INTRINSICS = {
    "fx": 1129.8136,
    "fy": 1128.6075,
    "cx": 961.0022,
    "cy": 546.8298,
    "width": 1920,
    "height": 1080,
    "distortion_model": "radial-tangential",
    "distortion": {
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "k5": 0.0,
        "k6": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    },
}
UR5E_LEFT_DEPTH_INTRINSICS = {
    "fx": 504.7930,
    "fy": 504.7819,
    "cx": 324.9653,
    "cy": 337.3727,
    "width": 640,
    "height": 576,
    "distortion_model": "radial-tangential",
    "distortion": {
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "k5": 0.0,
        "k6": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    },
}
UR5E_LEFT_ALIGNED_DEPTH_INTRINSICS = {
    "fx": UR5E_LEFT_COLOR_INTRINSICS["fx"],
    "fy": UR5E_LEFT_COLOR_INTRINSICS["fy"],
    "cx": UR5E_LEFT_COLOR_INTRINSICS["cx"],
    "cy": UR5E_LEFT_COLOR_INTRINSICS["cy"],
}
UR5E_LEFT_DEPTH_NATIVE_INTRINSICS = {
    "fx": UR5E_LEFT_DEPTH_INTRINSICS["fx"],
    "fy": UR5E_LEFT_DEPTH_INTRINSICS["fy"],
    "cx": UR5E_LEFT_DEPTH_INTRINSICS["cx"],
    "cy": UR5E_LEFT_DEPTH_INTRINSICS["cy"],
}
def _resolve_path(raw_path: str | Path, *, prefer_ur5e_root: bool = True) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    ur5e_relative = UR5E_ROOT / path
    if prefer_ur5e_root and ur5e_relative.exists():
        return ur5e_relative.resolve()
    if path.exists():
        return path.resolve()
    repo_relative = REPO_ROOT / path
    if repo_relative.exists() or str(path).startswith("ur5e_mujoco/"):
        return repo_relative.resolve()
    return ur5e_relative.resolve() if prefer_ur5e_root else repo_relative.resolve()


def _read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read color image: {path}")
    return image


def _read_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth image: {path}")
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth.squeeze(axis=2)
    if depth.ndim == 3:
        raise ValueError(f"Depth image must be single-channel metric depth, got shape={depth.shape}: {path}")
    if np.issubdtype(depth.dtype, np.integer) and int(np.max(depth)) <= 255:
        raise ValueError(
            f"Depth image looks like a normalized preview, not metric depth: {path}, max={int(np.max(depth))}. "
            "Use uint16 millimeter PNG or metric .npy depth."
        )
    return depth


def _shape_hw(image: np.ndarray) -> list[int]:
    return [int(image.shape[0]), int(image.shape[1])]


def _intrinsics_for_depth(depth_img: np.ndarray, color_img: np.ndarray) -> tuple[dict[str, float], str]:
    if depth_img.shape[:2] == color_img.shape[:2]:
        return dict(UR5E_LEFT_ALIGNED_DEPTH_INTRINSICS), "aligned_depth_uses_color_intrinsics"
    expected_native = (int(UR5E_LEFT_DEPTH_INTRINSICS["height"]), int(UR5E_LEFT_DEPTH_INTRINSICS["width"]))
    if depth_img.shape[:2] == expected_native:
        return dict(UR5E_LEFT_DEPTH_NATIVE_INTRINSICS), "resize_mask_to_depth"
    raise ValueError(
        "Unsupported depth shape for UR5e left camera: "
        f"depth={depth_img.shape[:2]}, color={color_img.shape[:2]}, expected native depth={expected_native}."
    )


def _resize_mask_to_depth(mask: np.ndarray, depth_img: np.ndarray) -> np.ndarray:
    if mask.shape[:2] == depth_img.shape[:2]:
        return mask
    return cv2.resize(mask.astype(np.uint8), (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)


def _blank_color_like_depth(depth_img: np.ndarray) -> np.ndarray:
    return np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)


def _scene_camera_transform(template_path: Path, camera_name: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not template_path.exists():
        raise FileNotFoundError(f"Scene template not found: {template_path}")
    model = mujoco.MjModel.from_xml_path(str(template_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera not found in scene template: {camera_name}")
    rotation_mj_from_cam = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    rotation_world_from_cv = rotation_mj_from_cam @ MJ_CAMERA_FROM_CV
    translation_world = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
    debug = {
        "template_path": str(template_path.resolve()),
        "camera_name": camera_name,
        "camera_xpos_world": translation_world.tolist(),
        "rotation_world_from_cv": rotation_world_from_cv.tolist(),
        "fovy_deg": float(model.cam_fovy[cam_id]),
    }
    return rotation_world_from_cv, translation_world, debug


def _points_camera_to_world(points_camera_m: np.ndarray, rotation_world_from_cv: np.ndarray, translation_world: np.ndarray) -> np.ndarray:
    return (rotation_world_from_cv @ points_camera_m.T).T + translation_world


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize near-zero vector.")
    return vec / norm


def _matrix_to_quat_wxyz(rotation_matrix: np.ndarray) -> list[float]:
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def _mesh_path(object_name: str) -> Path:
    path = FRUIT_STL_DIR / f"{object_name}.stl"
    if not path.exists():
        raise FileNotFoundError(f"Missing mesh for pose estimation: {path}")
    return path


def _load_mesh(object_name: str) -> trimesh.Trimesh:
    mesh = trimesh.load(_mesh_path(object_name), force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 8:
        raise ValueError(f"Invalid mesh vertices for pose estimation: {object_name}")
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

    y_axis = np.cross(minor, major)
    if np.linalg.norm(y_axis) < 1e-9:
        y_axis = middle
    y_axis = _normalize(y_axis)

    x_axis = _normalize(np.cross(y_axis, minor))
    z_axis = _normalize(np.cross(x_axis, y_axis))

    rotation = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(rotation) < 0:
        rotation[:, 1] = -rotation[:, 1]
    return rotation


def _base_rotation_camera_from_cloud_and_mesh(object_name: str, points_camera_m: np.ndarray) -> np.ndarray:
    observed_frame = _principal_frame(points_camera_m, object_name, camera_observed=True)
    mesh_frame = _principal_frame(np.asarray(_load_mesh(object_name).vertices, dtype=float), object_name, camera_observed=False)
    return observed_frame @ mesh_frame.T


def _project_world_to_image(
    points_world: np.ndarray,
    rotation_world_from_cv: np.ndarray,
    translation_world: np.ndarray,
    camera_intrinsics: dict[str, float],
) -> np.ndarray:
    points_camera = (rotation_world_from_cv.T @ (points_world - translation_world).T).T
    z = points_camera[:, 2]
    fx = float(camera_intrinsics["fx"])
    fy = float(camera_intrinsics["fy"])
    cx = float(camera_intrinsics["cx"])
    cy = float(camera_intrinsics["cy"])
    uv = np.empty((len(points_camera), 2), dtype=float)
    uv[:, 0] = fx * points_camera[:, 0] / z + cx
    uv[:, 1] = fy * points_camera[:, 1] / z + cy
    return uv


def _render_candidate_mask(
    mesh: trimesh.Trimesh,
    rotation_world: np.ndarray,
    translation_world: np.ndarray,
    rotation_world_from_cv: np.ndarray,
    camera_translation_world: np.ndarray,
    camera_intrinsics: dict[str, float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    transformed = (rotation_world @ vertices.T).T + translation_world
    points_camera = (rotation_world_from_cv.T @ (transformed - camera_translation_world).T).T
    mask = np.zeros(image_shape, dtype=np.uint8)
    h, w = image_shape
    fx = float(camera_intrinsics["fx"])
    fy = float(camera_intrinsics["fy"])
    cx = float(camera_intrinsics["cx"])
    cy = float(camera_intrinsics["cy"])

    for face in faces:
        tri = points_camera[face]
        if np.any(tri[:, 2] <= 1e-6):
            continue
        proj = np.column_stack([
            fx * tri[:, 0] / tri[:, 2] + cx,
            fy * tri[:, 1] / tri[:, 2] + cy,
        ])
        if np.all((proj[:, 0] < 0) | (proj[:, 0] >= w) | (proj[:, 1] < 0) | (proj[:, 1] >= h)):
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


def _score_candidate_mask(candidate_mask: np.ndarray, target_mask: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidate = candidate_mask > 0
    target = target_mask > 0
    intersection = int(np.count_nonzero(candidate & target))
    union = int(np.count_nonzero(candidate | target))
    iou = 0.0 if union == 0 else float(intersection / union)
    candidate_angle = _mask_orientation(candidate_mask)
    target_angle = _mask_orientation(target_mask.astype(np.uint8))
    if candidate_angle is None or target_angle is None:
        angle_error_deg = 180.0
    else:
        delta = abs(float(candidate_angle - target_angle))
        delta = min(delta, np.pi - delta % np.pi)
        angle_error_deg = float(np.degrees(delta))
    score = iou - 0.2 * (angle_error_deg / 180.0)
    return score, {"iou": iou, "angle_error_deg": angle_error_deg}


def _apply_top_axis_upright_constraint(
    object_name: str,
    quat_wxyz: list[float],
    object_pos_world: list[float],
    camera_pos_world: list[float],
) -> tuple[list[float], dict[str, Any]]:
    if object_name not in TOP_AXIS_LOCAL_BY_OBJECT or object_name not in FRONT_AXIS_LOCAL_BY_OBJECT:
        return quat_wxyz, {"top_constraint_applied": False}
    rotation_before = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()
    object_pos = np.asarray(object_pos_world, dtype=float)
    camera_pos = np.asarray(camera_pos_world, dtype=float)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    to_camera_world = camera_pos - object_pos
    front_target = to_camera_world - world_up * float(to_camera_world @ world_up)
    if float(np.linalg.norm(front_target)) <= 1e-9:
        return quat_wxyz, {"top_constraint_applied": False, "reason": "camera_above_object"}
    front_target = _normalize(front_target)

    y_world = -world_up
    z_world = -front_target
    x_world = _normalize(np.cross(y_world, z_world))
    z_world = _normalize(np.cross(x_world, y_world))
    corrected = np.column_stack([x_world, y_world, z_world])
    if float(np.linalg.det(corrected)) < 0.0:
        corrected[:, 0] = -corrected[:, 0]

    relative_rotation = corrected @ rotation_before.T
    return _matrix_to_quat_wxyz(corrected), {
        "top_constraint_applied": True,
        "top_upright_delta_deg": float(np.degrees(Rotation.from_matrix(relative_rotation).magnitude())),
        "method": "camera_facing_top_upright_constraint",
    }


def _estimate_object_quat_root_style(
    *,
    object_name: str,
    points_camera_m: np.ndarray,
    object_pos_world: list[float],
    target_mask: np.ndarray,
    rotation_world_from_cv: np.ndarray,
    camera_translation_world: np.ndarray,
    camera_intrinsics: dict[str, float],
    image_shape: tuple[int, int],
    output_dir: Path,
) -> tuple[list[float], dict[str, Any]]:
    if object_name not in OBJECT_LOCAL_CANDIDATES:
        return [1.0, 0.0, 0.0, 0.0], {"status": "fallback_identity", "reason": "unsupported_object"}
    if points_camera_m.ndim != 2 or points_camera_m.shape[1] != 3 or len(points_camera_m) < 8:
        return [1.0, 0.0, 0.0, 0.0], {"status": "fallback_identity", "reason": "insufficient_camera_points"}

    base_rotation_camera = _base_rotation_camera_from_cloud_and_mesh(object_name, points_camera_m)
    mesh = _load_mesh(object_name)
    translation_world = np.asarray(object_pos_world, dtype=float)
    candidate_dir = output_dir / "orientation"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    candidate_logs: list[dict[str, Any]] = []
    best_score = -1e9
    best_name = "base"
    best_rotation_world = rotation_world_from_cv @ base_rotation_camera
    for candidate_name, correction in OBJECT_LOCAL_CANDIDATES[object_name]:
        rotation_camera = base_rotation_camera @ correction
        rotation_world = rotation_world_from_cv @ rotation_camera
        candidate_mask = _render_candidate_mask(
            mesh,
            rotation_world,
            translation_world,
            rotation_world_from_cv,
            camera_translation_world,
            camera_intrinsics,
            image_shape,
        )
        score, score_debug = _score_candidate_mask(candidate_mask, target_mask)
        cv2.imwrite(str(candidate_dir / f"{object_name}_{candidate_name}.png"), candidate_mask * 255)
        log = {
            "candidate": candidate_name,
            "score": score,
            "quat_wxyz": _matrix_to_quat_wxyz(rotation_world),
            **score_debug,
        }
        candidate_logs.append(log)
        if score > best_score:
            best_score = score
            best_name = candidate_name
            best_rotation_world = rotation_world

    quat = _matrix_to_quat_wxyz(best_rotation_world)
    top_quat, top_debug = _apply_top_axis_upright_constraint(
        object_name,
        quat,
        object_pos_world,
        camera_translation_world.tolist(),
    )
    return top_quat, {
        "status": "success",
        "method": "root_style_3d_pca_mesh_frame_mask_candidate_camera_facing",
        "base_quat_wxyz": _matrix_to_quat_wxyz(best_rotation_world),
        "selected_candidate": best_name,
        "selected_score": best_score,
        "candidates": candidate_logs,
        "camera_facing": top_debug,
    }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        if value.ndim >= 2:
            return {"array_shape": list(value.shape), "array_dtype": str(value.dtype)}
        return value.tolist()
    if hasattr(value, "item") and value.__class__.__module__.startswith("numpy"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _detect_object_position(
    *,
    segmenter: GroundedSAM2Segmenter,
    color_image_path: Path,
    color_img: np.ndarray,
    depth_img: np.ndarray,
    object_name: str,
    camera_intrinsics: dict[str, float],
    rotation_world_from_cv: np.ndarray,
    translation_world: np.ndarray,
    output_dir: Path,
    downsample_scale: float,
    alignment_mode: str,
) -> tuple[list[float], list[float], dict[str, Any]]:
    mask_path = output_dir / "masks" / f"{color_image_path.stem}_{object_name}_mask.png"
    annotated_path = output_dir / "annotated" / f"{color_image_path.stem}_{object_name}_annotated.png"
    pcd_path = output_dir / "point_clouds" / f"{color_image_path.stem}_{object_name}_camera.npy"

    seg_result = segmenter.segment_image(
        image_path=str(color_image_path),
        target_class=object_name,
        output_mask_path=str(mask_path),
        output_annotated_path=str(annotated_path),
    )
    if seg_result is None:
        raise RuntimeError(f"Grounded-SAM2 did not detect object: {object_name}")

    mask_for_pcd = _resize_mask_to_depth(seg_result.mask, depth_img)
    color_for_pcd = color_img if color_img.shape[:2] == depth_img.shape[:2] else _blank_color_like_depth(depth_img)

    pcd_generator = PointCloudGenerator(
        camera_intrinsics=camera_intrinsics,
        visualize=False,
        save_point_cloud=True,
        save_path=str(pcd_path),
        denoise=True,
        denoise_neighbors=10,
        denoise_std_ratio=5.0,
        use_dbscan=False,
    )
    pcd_result = pcd_generator.generate_point_cloud(
        color_image_aligned=color_for_pcd,
        depth_image_aligned=depth_img,
        mask=mask_for_pcd,
        downsample_scale=downsample_scale,
        target_coordinate_system="camera",
    )
    if pcd_result.get("state") != "success":
        raise RuntimeError(f"Point cloud generation failed for {object_name}: {pcd_result.get('info')}")

    points_camera_m = np.asarray(pcd_result["point_cloud"], dtype=np.float64) / 1000.0
    points_world = _points_camera_to_world(points_camera_m, rotation_world_from_cv, translation_world)
    if points_world.ndim != 2 or points_world.shape[1] != 3 or len(points_world) == 0:
        raise RuntimeError(f"Invalid world point cloud for {object_name}.")
    world_pcd_path = output_dir / "point_clouds" / f"{color_image_path.stem}_{object_name}_world.npy"
    np.save(world_pcd_path, points_world)
    mins = np.min(points_world, axis=0)
    maxs = np.max(points_world, axis=0)
    center_world = (mins + maxs) / 2.0
    position = [
        float(center_world[0]),
        float(center_world[1]),
        max(0.0, float(center_world[2])),
    ]
    object_quat, orientation_report = _estimate_object_quat_root_style(
        object_name=object_name,
        points_camera_m=points_camera_m,
        object_pos_world=position,
        target_mask=seg_result.mask,
        rotation_world_from_cv=rotation_world_from_cv,
        camera_translation_world=translation_world,
        camera_intrinsics=camera_intrinsics,
        image_shape=tuple(color_img.shape[:2]),
        output_dir=output_dir,
    )
    return position, object_quat, {
        "mask_path": str(mask_path.resolve()),
        "annotated_path": str(annotated_path.resolve()),
        "point_cloud_path": str(pcd_path.resolve()),
        "world_point_cloud_path": str(world_pcd_path.resolve()),
        "candidate": _jsonable(seg_result.candidate),
        "candidate_count": len(seg_result.candidates),
        "rgb_mask_shape": _shape_hw(seg_result.mask),
        "pcd_mask_shape": _shape_hw(mask_for_pcd),
        "color_shape": _shape_hw(color_img),
        "depth_shape": _shape_hw(depth_img),
        "alignment_mode": alignment_mode,
        "point_cloud_info": _jsonable({k: v for k, v in pcd_result.items() if k != "point_cloud"}),
        "world_bounds_m": {
            "min": mins.tolist(),
            "max": maxs.tolist(),
            "center": center_world.tolist(),
        },
        "object_quat_wxyz": object_quat,
        "orientation": orientation_report,
    }


def build_runtime_scene_from_real_camera(
    *,
    color_image_path: Path,
    depth_image_path: Path,
    objects: list[str],
    scene_out: Path,
    report_out: Path,
    template_path: Path = DEFAULT_TEMPLATE,
    camera_name: str = DEFAULT_CAMERA,
    camera_intrinsics: dict[str, float] | None = None,
    downsample_scale: float = 1.0,
    grounded_sam2_root: Path | None = None,
) -> dict[str, Any]:
    color_img = _read_color(color_image_path)
    depth_img = _read_depth(depth_image_path)

    inferred_intrinsics, alignment_mode = _intrinsics_for_depth(depth_img, color_img)
    camera_intrinsics = camera_intrinsics or inferred_intrinsics
    rotation_world_from_cv, translation_world, camera_pose_debug = _scene_camera_transform(template_path, camera_name)
    output_dir = report_out.parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    grounded_sam2_root = grounded_sam2_root or DEFAULT_GROUNDED_SAM2_ROOT

    segmenter = GroundedSAM2Segmenter(
        grounded_sam2_root=str(grounded_sam2_root),
        box_threshold=0.2,
        text_threshold=0.2,
        device=None,
        multimask_output=True,
    )

    object_positions: dict[str, list[float]] = {}
    object_quats: dict[str, list[float]] = {}
    object_reports: dict[str, Any] = {}
    for object_name in objects:
        position, object_quat, object_report = _detect_object_position(
            segmenter=segmenter,
            color_image_path=color_image_path,
            color_img=color_img,
            depth_img=depth_img,
            object_name=object_name,
            camera_intrinsics=camera_intrinsics,
            rotation_world_from_cv=rotation_world_from_cv,
            translation_world=translation_world,
            output_dir=output_dir,
            downsample_scale=downsample_scale,
            alignment_mode=alignment_mode,
        )
        object_positions[object_name] = position
        object_quats[object_name] = object_quat
        object_reports[object_name] = object_report

    scene_path = generate_scene(object_positions, scene_out=scene_out, template=template_path, object_quats=object_quats)
    report = {
        "status": "success",
        "color_image_path": str(color_image_path.resolve()),
        "depth_image_path": str(depth_image_path.resolve()),
        "scene_out": str(Path(scene_path).resolve()),
        "scene_template": str(template_path.resolve()),
        "objects": objects,
        "object_positions_m": object_positions,
        "object_quats_wxyz": object_quats,
        "color_shape": _shape_hw(color_img),
        "depth_shape": _shape_hw(depth_img),
        "alignment_mode": alignment_mode,
        "color_intrinsics": UR5E_LEFT_COLOR_INTRINSICS,
        "depth_intrinsics": UR5E_LEFT_DEPTH_INTRINSICS,
        "camera_intrinsics": camera_intrinsics,
        "scene_camera": camera_pose_debug,
        "grounded_sam2_root": str(Path(grounded_sam2_root).resolve()),
        "target_coordinate_system": "mujoco_world_from_scene_camera",
        "downsample_scale": float(downsample_scale),
        "object_reports": object_reports,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a MuJoCo runtime scene from one real RGB-D camera frame.")
    parser.add_argument("--color", default=str(DEFAULT_COLOR_IMAGE), help="Real camera BGR/RGB image path.")
    parser.add_argument("--depth", default=str(DEFAULT_DEPTH_IMAGE), help="Real camera metric depth path: uint16 PNG or NPY.")
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS), help="Object classes to detect.")
    parser.add_argument("--scene-out", default=str(DEFAULT_SCENE_OUT), help="Output MuJoCo XML path.")
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_OUT), help="Output JSON report path.")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="UR5e MuJoCo template scene, default scene2.xml.")
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help="Camera name in template scene used for RGB-D projection.")
    parser.add_argument("--downsample-scale", type=float, default=1.0)
    parser.add_argument("--grounded-sam2-root", default=str(DEFAULT_GROUNDED_SAM2_ROOT))
    args = parser.parse_args()

    objects = [str(item).strip().lower().rstrip(".") for item in args.objects if str(item).strip()]
    result = build_runtime_scene_from_real_camera(
        color_image_path=_resolve_path(args.color),
        depth_image_path=_resolve_path(args.depth),
        objects=objects,
        scene_out=_resolve_path(args.scene_out),
        report_out=_resolve_path(args.report_out),
        template_path=_resolve_path(args.template),
        camera_name=str(args.camera),
        downsample_scale=float(args.downsample_scale),
        grounded_sam2_root=_resolve_path(args.grounded_sam2_root, prefer_ur5e_root=False),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
