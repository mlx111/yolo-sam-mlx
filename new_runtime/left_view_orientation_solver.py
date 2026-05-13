from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from itertools import product

import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


ROOT_DIR = Path(__file__).resolve().parents[1]
LEFT_INTRINSICS = {
    "fx": 1129.8136,
    "fy": 1128.6075,
    "cx": 961.0022,
    "cy": 546.8298,
    "width": 1920,
    "height": 1080,
}
SUPPORTED_OBJECTS = ("apple", "pear")
RUNTIME_MESH_BY_OBJECT = {
    "apple": ROOT_DIR / "manipulator_grasp" / "assets" / "fruit" / "stl" / "apple.stl",
    "pear": ROOT_DIR / "manipulator_grasp" / "assets" / "fruit" / "stl" / "pear.stl",
}
LEFT_MASK_BY_OBJECT = {
    "apple": ROOT_DIR / "inputs" / "left_mask_apple.png",
    "pear": ROOT_DIR / "inputs" / "left_mask_pear.png",
}
DEBUG_DIR = ROOT_DIR / "outputs" / "left_view_orientation_debug"
LOCAL_CANDIDATES_DEG = {
    "apple": [
        ("identity", (0.0, 0.0, 0.0)),
        ("yaw_pos_90", (0.0, 0.0, 90.0)),
        ("yaw_neg_90", (0.0, 0.0, -90.0)),
        ("yaw_180", (0.0, 0.0, 180.0)),
        ("roll_180", (180.0, 0.0, 0.0)),
    ],
    "pear": [
        ("identity", (0.0, 0.0, 0.0)),
        ("yaw_pos_90", (0.0, 0.0, 90.0)),
        ("yaw_neg_90", (0.0, 0.0, -90.0)),
        ("yaw_180", (0.0, 0.0, 180.0)),
        ("pitch_180", (0.0, 180.0, 0.0)),
        ("roll_180", (180.0, 0.0, 0.0)),
    ],
}
COARSE_EULER_GRID_DEG = (0.0, 90.0, 180.0, 270.0)
LOCAL_REFINE_TOP_K = 4
LEFT_VIEW_RENDER_MAX_FACES = 5000


class LeftViewOrientationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OrientationFitResult:
    quat_wxyz: list[float]
    score: float
    iou: float
    center_error_px: float


def _quat_wxyz_to_matrix(quat_wxyz: list[float]) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=float)
    if quat.shape != (4,):
        raise LeftViewOrientationError(f"Expected quat shape (4,), got {quat.shape}")
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _matrix_to_quat_wxyz(rotation_matrix: np.ndarray) -> list[float]:
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def _load_mask(object_name: str) -> np.ndarray:
    path = LEFT_MASK_BY_OBJECT[object_name]
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise LeftViewOrientationError(f"Failed to load left mask: {path}")
    binary = (mask > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        raise LeftViewOrientationError(f"Left mask is empty: {path}")
    return binary



def _load_mesh(object_name: str) -> trimesh.Trimesh:
    path = RUNTIME_MESH_BY_OBJECT[object_name]
    if not path.exists():
        raise LeftViewOrientationError(f"Missing runtime mesh: {path}")
    mesh = trimesh.load(path, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 8:
        raise LeftViewOrientationError(f"Invalid runtime mesh vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 8:
        raise LeftViewOrientationError(f"Invalid runtime mesh faces: {path}")
    if len(faces) > LEFT_VIEW_RENDER_MAX_FACES:
        stride = int(np.ceil(len(faces) / LEFT_VIEW_RENDER_MAX_FACES))
        reduced_faces = faces[::stride]
        reduced = trimesh.Trimesh(vertices=vertices, faces=reduced_faces, process=False)
        print(
            f"[INFO] left-view orientation mesh decimated: {object_name} faces {len(faces)} -> {len(reduced_faces)}",
            flush=True,
        )
        return reduced
    return mesh


def _render_candidate_mask(
    mesh: trimesh.Trimesh,
    rotation_world: np.ndarray,
    translation_world: np.ndarray,
    camera_rotation_world: np.ndarray,
    camera_translation_world: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    transformed = (rotation_world @ vertices.T).T + translation_world
    camera_points = (camera_rotation_world.T @ (transformed - camera_translation_world).T).T

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


def _mask_center(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([np.nan, np.nan], dtype=float)
    return np.array([xs.mean(), ys.mean()], dtype=float)


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


def _recenter_mask(mask: np.ndarray, target_center: np.ndarray) -> np.ndarray:
    center = _mask_center(mask)
    if np.isnan(center).any() or np.isnan(target_center).any():
        return np.zeros_like(mask, dtype=np.uint8)
    shift_x = float(target_center[0] - center[0])
    shift_y = float(target_center[1] - center[1])
    transform = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=float)
    shifted = cv2.warpAffine(
        (mask > 0).astype(np.uint8),
        transform,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (shifted > 0).astype(np.uint8)


def _orientation_alignment_metrics(rendered: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float, float]:
    rendered_center = _mask_center(rendered)
    gt_center = _mask_center(gt_mask)
    if np.isnan(rendered_center).any() or np.isnan(gt_center).any():
        return 0.0, 180.0, 1e6
    center_error_px = float(np.linalg.norm(rendered_center - gt_center))
    recentered = _recenter_mask(rendered, gt_center)
    recentered_iou = _mask_iou(recentered, gt_mask)
    rendered_angle = _mask_orientation(recentered)
    gt_angle = _mask_orientation(gt_mask)
    angle_error_deg = 180.0
    if rendered_angle is not None and gt_angle is not None:
        diff = abs(rendered_angle - gt_angle)
        diff = min(diff, abs(np.pi - diff), abs(2 * np.pi - diff))
        angle_error_deg = float(np.degrees(diff))
    return float(recentered_iou), angle_error_deg, center_error_px


def _score_rendered_mask(rendered: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float, float]:
    recentered_iou, angle_error_deg, center_error_px = _orientation_alignment_metrics(rendered, gt_mask)
    angle_penalty = angle_error_deg / 180.0
    score = float((1.0 - recentered_iou) + 0.2 * angle_penalty)
    return score, recentered_iou, center_error_px


def _write_debug_images(
    object_name: str,
    rendered: np.ndarray,
    gt_mask: np.ndarray,
    candidate_name: str,
) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    rendered_u8 = (rendered > 0).astype(np.uint8) * 255
    gt_u8 = (gt_mask > 0).astype(np.uint8) * 255
    overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    overlay[..., 1] = gt_u8
    overlay[..., 2] = rendered_u8
    cv2.imwrite(str(DEBUG_DIR / f"{object_name}_{candidate_name}_rendered.png"), rendered_u8)
    cv2.imwrite(str(DEBUG_DIR / f"{object_name}_{candidate_name}_overlay.png"), overlay)


def _fit_single_object(
    object_name: str,
    initial_quat_wxyz: list[float],
    translation_world: list[float],
    camera_rotation_world: np.ndarray,
    camera_translation_world: np.ndarray,
) -> OrientationFitResult:
    mesh = _load_mesh(object_name)
    gt_mask = _load_mask(object_name)
    translation = np.asarray(translation_world, dtype=float)
    initial_rotation = _quat_wxyz_to_matrix(initial_quat_wxyz)

    candidate_logs: list[dict] = []
    best_name = "initial"
    best_rotation = initial_rotation
    best_score = float("inf")
    best_iou = 0.0
    best_center_error = 1e6

    seen_candidates: set[tuple[float, ...]] = set()

    def evaluate_candidate(candidate_name: str, rotation_world: np.ndarray, euler_deg: tuple[float, float, float] | None) -> None:
        nonlocal best_name, best_rotation, best_score, best_iou, best_center_error
        key = tuple(np.round(rotation_world.reshape(-1), 6).tolist())
        if key in seen_candidates:
            return
        seen_candidates.add(key)
        rendered = _render_candidate_mask(
            mesh,
            rotation_world,
            translation,
            camera_rotation_world,
            camera_translation_world,
        )
        score, iou, center_error = _score_rendered_mask(rendered, gt_mask)
        candidate_logs.append(
            {
                "candidate": candidate_name,
                "euler_xyz_deg": None if euler_deg is None else list(euler_deg),
                "score": score,
                "iou": iou,
                "center_error_px": center_error,
                "quat_wxyz": _matrix_to_quat_wxyz(rotation_world),
            }
        )
        _write_debug_images(object_name, rendered, gt_mask, candidate_name)
        if score < best_score:
            best_name = candidate_name
            best_rotation = rotation_world
            best_score = score
            best_iou = iou
            best_center_error = center_error

    for candidate_name, euler_deg in LOCAL_CANDIDATES_DEG[object_name]:
        correction = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
        rotation_world = initial_rotation @ correction
        evaluate_candidate(candidate_name, rotation_world, euler_deg)

    for rx_deg, ry_deg, rz_deg in product(COARSE_EULER_GRID_DEG, repeat=3):
        euler_deg = (rx_deg, ry_deg, rz_deg)
        correction = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
        rotation_world = initial_rotation @ correction
        candidate_name = f"coarse_{int(rx_deg)}_{int(ry_deg)}_{int(rz_deg)}"
        evaluate_candidate(candidate_name, rotation_world, euler_deg)

    sorted_candidates = sorted(candidate_logs, key=lambda item: item["score"])
    refine_seeds = sorted_candidates[:LOCAL_REFINE_TOP_K]

    def objective_for_seed(seed_rotation: np.ndarray, delta_euler_deg: np.ndarray) -> float:
        rotation_world = seed_rotation @ Rotation.from_euler("xyz", delta_euler_deg, degrees=True).as_matrix()
        rendered = _render_candidate_mask(
            mesh,
            rotation_world,
            translation,
            camera_rotation_world,
            camera_translation_world,
        )
        score, _, _ = _score_rendered_mask(rendered, gt_mask)
        regularization = 0.0008 * float(np.linalg.norm(delta_euler_deg))
        return float(score + regularization)

    refine_logs: list[dict] = []
    best_refined_rotation = best_rotation
    best_refined_score = best_score
    best_refined_iou = best_iou
    best_refined_center_error = best_center_error
    best_refined_result = None
    best_refined_seed = best_name

    for seed in refine_seeds:
        seed_rotation = _quat_wxyz_to_matrix(seed["quat_wxyz"])
        result = minimize(
            lambda delta: objective_for_seed(seed_rotation, delta),
            x0=np.zeros(3, dtype=float),
            method="Powell",
            options={"maxiter": 80, "xtol": 1e-2, "ftol": 1e-3},
        )
        refined_rotation = seed_rotation @ Rotation.from_euler("xyz", result.x, degrees=True).as_matrix()
        refined_rendered = _render_candidate_mask(
            mesh,
            refined_rotation,
            translation,
            camera_rotation_world,
            camera_translation_world,
        )
        refined_score, refined_iou, refined_center_error = _score_rendered_mask(refined_rendered, gt_mask)
        refine_logs.append(
            {
                "seed_candidate": seed["candidate"],
                "success": bool(result.success),
                "message": str(result.message),
                "delta_euler_xyz_deg": [float(v) for v in result.x],
                "score": refined_score,
                "iou": refined_iou,
                "center_error_px": refined_center_error,
                "quat_wxyz": _matrix_to_quat_wxyz(refined_rotation),
            }
        )
        _write_debug_images(object_name, refined_rendered, gt_mask, f"refined_{seed['candidate']}")
        if refined_score < best_refined_score:
            best_refined_rotation = refined_rotation
            best_refined_score = refined_score
            best_refined_iou = refined_iou
            best_refined_center_error = refined_center_error
            best_refined_result = result
            best_refined_seed = seed["candidate"]

    final_rotation = best_refined_rotation
    final_rendered = _render_candidate_mask(
        mesh,
        final_rotation,
        translation,
        camera_rotation_world,
        camera_translation_world,
    )
    final_score, final_iou, final_center_error = _score_rendered_mask(final_rendered, gt_mask)
    _write_debug_images(object_name, final_rendered, gt_mask, "final")

    report = {
        "object_name": object_name,
        "initial_quat_wxyz": [float(v) for v in initial_quat_wxyz],
        "best_discrete_candidate": best_name,
        "discrete_best_score": best_score,
        "discrete_best_iou": best_iou,
        "discrete_best_center_error_px": best_center_error,
        "refine_seed_count": len(refine_seeds),
        "best_refine_seed": best_refined_seed,
        "local_refine_success": bool(best_refined_result.success) if best_refined_result is not None else False,
        "local_refine_message": str(best_refined_result.message) if best_refined_result is not None else "no improvement over discrete search",
        "local_refine_delta_euler_xyz_deg": [float(v) for v in best_refined_result.x] if best_refined_result is not None else [0.0, 0.0, 0.0],
        "final_quat_wxyz": _matrix_to_quat_wxyz(final_rotation),
        "final_score": final_score,
        "final_iou": final_iou,
        "final_center_error_px": final_center_error,
        "candidates": candidate_logs,
        "refine_runs": refine_logs,
    }
    (DEBUG_DIR / f"{object_name}_fit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return OrientationFitResult(
        quat_wxyz=report["final_quat_wxyz"],
        score=final_score,
        iou=final_iou,
        center_error_px=final_center_error,
    )


def solve_left_view_object_quats(
    initial_object_quats: dict[str, list[float]],
    object_positions: dict[str, list[float]],
    camera_poses: dict[str, dict[str, list[float]]],
) -> dict[str, list[float]]:
    missing = [name for name in SUPPORTED_OBJECTS if name not in initial_object_quats or name not in object_positions]
    if missing:
        raise LeftViewOrientationError(f"Missing object inputs for: {', '.join(missing)}")
    if "cam1" not in camera_poses:
        raise LeftViewOrientationError("camera_poses must contain cam1")

    camera_rotation = _quat_wxyz_to_matrix(camera_poses["cam1"]["quat"])
    camera_translation = np.asarray(camera_poses["cam1"]["pos"], dtype=float)

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float | list[float]]] = {}
    solved_quats: dict[str, list[float]] = {}
    for object_name in SUPPORTED_OBJECTS:
        fit = _fit_single_object(
            object_name=object_name,
            initial_quat_wxyz=initial_object_quats[object_name],
            translation_world=object_positions[object_name],
            camera_rotation_world=camera_rotation,
            camera_translation_world=camera_translation,
        )
        solved_quats[object_name] = fit.quat_wxyz
        summary[object_name] = {
            "quat_wxyz": fit.quat_wxyz,
            "score": fit.score,
            "iou": fit.iou,
            "center_error_px": fit.center_error_px,
        }

    (DEBUG_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return solved_quats
