"""UR5e runtime helpers for fixed vertical grasping.

This module intentionally does not call GraspNet.  It uses the existing
Grounded-SAM2 perception path and point-cloud completion to choose a top-surface
grasp point, then keeps the gripper orientation fixed vertically downward.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import spatialmath as sm

from pointcloud_completion_utils import CompletionConfig, complete_point_cloud


DOWN_GRASP_ROTATION = sm.SE3.Rz(-np.pi / 2).R @ sm.SE3.Ry(np.pi / 2).R
WORLD_VERTICAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)
DEFAULT_TOP_BAND_M = 0.01
DEFAULT_TOP_QUANTILE = 0.92
DEFAULT_GRASP_DESCEND_M = 0.025
DEFAULT_PREGRASP_HEIGHT_M = 0.127


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


@dataclass
class FixedVerticalGraspContext:
    target_class: str
    target_pos: np.ndarray
    pregrasp_pos: np.ndarray
    raw_points_world: np.ndarray
    completed_points_world: np.ndarray
    top_points_world: np.ndarray
    report: dict[str, Any]


def _project_to_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    u, _, vh = np.linalg.svd(matrix)
    rot = u @ vh
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vh
    return rot


def _normalize_vector(vec: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        return vec / norm
    if fallback is None:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    fallback = np.asarray(fallback, dtype=np.float64)
    return fallback / max(float(np.linalg.norm(fallback)), 1e-12)


def _stabilize_horizontal_axis(axis_xy: np.ndarray) -> np.ndarray:
    axis_xy = np.asarray(axis_xy, dtype=np.float64).reshape(2)
    axis = np.array([axis_xy[0], axis_xy[1], 0.0], dtype=np.float64)
    axis = _normalize_vector(axis, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    if axis[0] < 0.0 or (abs(axis[0]) < 1e-9 and axis[1] < 0.0):
        axis *= -1.0
    return axis


def _wrapped_joint_delta(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    q_from = np.asarray(q_from, dtype=np.float64)
    q_to = np.asarray(q_to, dtype=np.float64)
    return np.arctan2(np.sin(q_to - q_from), np.cos(q_to - q_from))


def build_fixed_vertical_grasp_rotation(points_world: np.ndarray | None = None) -> np.ndarray:
    """Build the same analytic top-grasp frame used by the original demo.

    The frame is semantic grasp-frame orientation, not the robot TCP frame.
    `select_fixed_vertical_tcp_pose` maps it into the actual robot TCP frame.
    """
    closing_axis_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if points_world is not None:
        points = np.asarray(points_world, dtype=np.float64)
        if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] >= 3:
            xy_center = np.median(points[:, :2], axis=0)
            xy_offsets = points[:, :2] - xy_center.reshape(1, 2)
            try:
                cov = np.cov(xy_offsets.T)
                _, eigvecs = np.linalg.eigh(cov)
                closing_axis_world = _stabilize_horizontal_axis(eigvecs[:, 0])
            except np.linalg.LinAlgError:
                closing_axis_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    approach_axis_world = WORLD_VERTICAL.copy()
    closing_axis_world = _normalize_vector(
        closing_axis_world - np.dot(closing_axis_world, approach_axis_world) * approach_axis_world,
        fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    finger_axis_world = _normalize_vector(
        np.cross(approach_axis_world, closing_axis_world),
        fallback=np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )
    return _project_to_rotation_matrix(np.column_stack((approach_axis_world, closing_axis_world, finger_axis_world)))


def _build_pose_from_grasp(
    grasp_translation: np.ndarray,
    grasp_rotation: np.ndarray,
    mapping: dict[str, Any],
) -> tuple[sm.SE3, np.ndarray]:
    tcp_rotation = _project_to_rotation_matrix(np.asarray(grasp_rotation, dtype=np.float64) @ mapping["R_grasp_to_tcp"])
    return sm.SE3.Trans(np.asarray(grasp_translation, dtype=np.float64)) * sm.SE3(sm.SO3(tcp_rotation, check=False)), tcp_rotation


def _evaluate_tcp_mapping(
    robot: Any,
    grasp_translation: np.ndarray,
    grasp_rotation: np.ndarray,
    mapping: dict[str, Any],
    q_seed: np.ndarray,
    approach_distance: float,
) -> dict[str, Any]:
    original_q = np.asarray(robot.get_joint(), dtype=np.float64)
    T_grasp, tcp_rotation = _build_pose_from_grasp(grasp_translation, grasp_rotation, mapping)
    retreat_axis_tcp = np.asarray(mapping["retreat_axis_tcp"], dtype=np.float64)
    retreat_axis_world = np.asarray(T_grasp.R @ retreat_axis_tcp, dtype=np.float64)
    retreat_axis_world = _normalize_vector(retreat_axis_world, fallback=WORLD_VERTICAL)
    retreat_alignment = float(np.dot(retreat_axis_world, WORLD_VERTICAL))
    retreat_angle_deg = float(np.degrees(np.arccos(np.clip(retreat_alignment, -1.0, 1.0))))

    offset = retreat_axis_tcp * float(approach_distance)
    T_pregrasp = T_grasp * sm.SE3(float(offset[0]), float(offset[1]), float(offset[2]))
    pregrasp_above = bool(float(T_pregrasp.t[2]) > float(T_grasp.t[2]))

    ik_success = False
    q_pregrasp = None
    q_grasp = None
    joint_cost = float("inf")
    wrist_cost = float("inf")
    try:
        robot.set_joint(np.asarray(q_seed, dtype=np.float64))
        q_pre = robot.ikine(T_pregrasp)
        if q_pre is not None and len(q_pre) == 6:
            q_pre = np.asarray(q_pre, dtype=np.float64)
            robot.set_joint(q_pre)
            q_grasp = robot.ikine(T_grasp)
            if q_grasp is not None and len(q_grasp) == 6:
                q_grasp = np.asarray(q_grasp, dtype=np.float64)
                ik_success = True
                q_pregrasp = q_pre
                delta_pre = _wrapped_joint_delta(q_seed, q_pregrasp)
                delta_grasp = _wrapped_joint_delta(q_pregrasp, q_grasp)
                joint_cost = float(np.linalg.norm(delta_pre) + np.linalg.norm(delta_grasp))
                wrist_cost = float(np.sum(np.abs(delta_pre[3:])) + np.sum(np.abs(delta_grasp[3:])))
    finally:
        robot.set_joint(original_q)

    return {
        "name": mapping["name"],
        "R_grasp_to_tcp": np.asarray(mapping["R_grasp_to_tcp"], dtype=np.float64),
        "retreat_axis_tcp": retreat_axis_tcp,
        "retreat_axis_world": retreat_axis_world,
        "retreat_alignment": retreat_alignment,
        "retreat_angle_deg": retreat_angle_deg,
        "tcp_rotation": tcp_rotation,
        "T_grasp": T_grasp,
        "T_pregrasp": T_pregrasp,
        "pregrasp_above": pregrasp_above,
        "ik_success": ik_success,
        "q_pregrasp": q_pregrasp,
        "q_grasp": q_grasp,
        "joint_cost": joint_cost,
        "wrist_cost": wrist_cost,
    }


def select_fixed_vertical_tcp_pose(
    robot: Any,
    grasp_pos: np.ndarray,
    *,
    grasp_rotation: np.ndarray | None = None,
    top_points_world: np.ndarray | None = None,
    q_seed: np.ndarray | None = None,
    approach_distance: float = DEFAULT_PREGRASP_HEIGHT_M,
    require_ik: bool = True,
) -> dict[str, Any]:
    grasp_pos = np.asarray(grasp_pos, dtype=np.float64).reshape(3)
    if grasp_rotation is None:
        grasp_rotation = build_fixed_vertical_grasp_rotation(top_points_world)
    else:
        grasp_rotation = _project_to_rotation_matrix(grasp_rotation)
    if q_seed is None:
        q_seed = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0], dtype=np.float64)

    evaluations = [
        _evaluate_tcp_mapping(
            robot,
            grasp_pos,
            grasp_rotation,
            mapping,
            q_seed,
            float(approach_distance),
        )
        for mapping in TCP_FRAME_CANDIDATES
    ]
    valid = [item for item in evaluations if item["pregrasp_above"] and item["ik_success"]]
    if not valid and not require_ik:
        valid = [item for item in evaluations if item["pregrasp_above"]]
    if not valid:
        details = [
            {
                "name": item["name"],
                "pregrasp_above": item["pregrasp_above"],
                "ik_success": item["ik_success"],
                "retreat_angle_deg": item["retreat_angle_deg"],
            }
            for item in evaluations
        ]
        raise RuntimeError(f"no_valid_fixed_vertical_tcp_mapping: {details}")

    valid.sort(key=lambda item: (item["retreat_angle_deg"], item["joint_cost"], item["wrist_cost"]))
    best = valid[0]
    return {
        **best,
        "grasp_frame_rotation": grasp_rotation,
        "evaluations": evaluations,
    }


def install_fixed_vertical_tcp_pose(
    experiment: Any,
    grasp_pos: np.ndarray,
    *,
    top_points_world: np.ndarray | None = None,
    pregrasp_height: float = DEFAULT_PREGRASP_HEIGHT_M,
    require_ik: bool = True,
) -> dict[str, Any]:
    mapping = select_fixed_vertical_tcp_pose(
        experiment.robot,
        grasp_pos,
        top_points_world=top_points_world,
        approach_distance=float(pregrasp_height),
        require_ik=require_ik,
    )
    experiment.T_wo = mapping["T_grasp"]
    experiment.T_pregrasp = mapping["T_pregrasp"]
    experiment._fixed_vertical_tcp_mapping = mapping
    return mapping


def tcp_mapping_report(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        "tcp_mapping": str(mapping.get("name", "")),
        "tcp_grasp_pos": np.asarray(mapping["T_grasp"].t, dtype=np.float64).tolist(),
        "tcp_pregrasp_pos": np.asarray(mapping["T_pregrasp"].t, dtype=np.float64).tolist(),
        "semantic_grasp_rotation": np.asarray(mapping["grasp_frame_rotation"], dtype=np.float64).tolist(),
        "tcp_rotation": np.asarray(mapping["tcp_rotation"], dtype=np.float64).tolist(),
        "retreat_axis_tcp": np.asarray(mapping["retreat_axis_tcp"], dtype=np.float64).tolist(),
        "retreat_axis_world": np.asarray(mapping["retreat_axis_world"], dtype=np.float64).tolist(),
        "retreat_angle_deg": float(mapping["retreat_angle_deg"]),
        "q_pregrasp": None if mapping.get("q_pregrasp") is None else np.asarray(mapping["q_pregrasp"], dtype=np.float64).tolist(),
        "q_grasp": None if mapping.get("q_grasp") is None else np.asarray(mapping["q_grasp"], dtype=np.float64).tolist(),
    }


def _metrics(experiment: Any) -> dict[str, Any]:
    metrics = getattr(experiment, "metrics", None)
    if not isinstance(metrics, dict):
        experiment.metrics = {}
    return experiment.metrics


def _completion_config(params: dict[str, Any] | None = None) -> CompletionConfig:
    params = params if isinstance(params, dict) else {}
    cfg = CompletionConfig()
    overrides = params.get("completion_config")
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    return cfg


def save_camera_rgbd(experiment: Any, work_dir: str | Path | None = None) -> dict[str, Any]:
    work_dir = work_dir or Path("/tmp") / "wrapper1_camera_rgbd"
    rgbd = experiment.perception.render_rgbd(work_dir=work_dir)
    metrics = _metrics(experiment)
    camera_context = {
        "camera_name": str(rgbd.get("camera_name") or ""),
        "work_dir": str(rgbd.get("work_dir") or work_dir),
        "rgb_path": str(rgbd.get("rgb_path") or ""),
        "depth_path": str(rgbd.get("depth_path") or ""),
        "camera_config": rgbd.get("camera_config") or {},
    }
    metrics["last_camera_rgbd"] = camera_context
    return camera_context


def detect_object(experiment: Any, target_class: str = "apple", work_dir: str | Path | None = None) -> Any:
    work_dir = work_dir or Path("/tmp") / f"wrapper1_detect_{target_class}"
    metrics = _metrics(experiment)
    rgbd = metrics.get("last_camera_rgbd") if isinstance(metrics.get("last_camera_rgbd"), dict) else None
    if not (rgbd and rgbd.get("rgb_path") and rgbd.get("depth_path")):
        raise RuntimeError("camera_rgbd_missing_run_camera_rgbd_save_first")
    scene = experiment.perception.detect_from_rgbd(
        target_class=target_class,
        rgb_path=rgbd["rgb_path"],
        depth_path=rgbd["depth_path"],
        work_dir=work_dir,
        return_cloud=True,
        camera_config=rgbd.get("camera_config") if isinstance(rgbd.get("camera_config"), dict) else None,
    )
    rgbd_source = "cached_camera_rgbd"
    metrics["last_perception_scene"] = {
        "target_class": target_class,
        "detection_ok": bool(scene.detection_ok),
        "confidence": float(scene.confidence),
        "mask_nonzero": int(scene.mask_nonzero),
        "rgb_path": scene.rgb_path,
        "mask_path": scene.mask_path,
        "depth_path": scene.depth_path,
        "rgbd_source": rgbd_source,
    }
    if scene.detection_ok and scene.apple_pos is not None:
        perceived = np.asarray(scene.apple_pos, dtype=np.float64)
        if target_class == "apple" and hasattr(experiment, "apple_initial_pos"):
            max_tabletop_z = float(np.asarray(experiment.apple_initial_pos, dtype=np.float64)[2] + 0.12)
            secured_in_gripper = False
            if hasattr(experiment, "_current_skill_observation"):
                try:
                    obs = experiment._current_skill_observation()
                    contact = obs.get("contact", {}) if isinstance(obs, dict) else {}
                    secured_in_gripper = bool(
                        contact.get("left_contact")
                        or contact.get("right_contact")
                        or float(obs.get("pinch_distance", 1.0)) < 0.08
                    )
                except Exception:
                    secured_in_gripper = False
            if float(perceived[2]) > max_tabletop_z and not secured_in_gripper:
                metrics["last_perception_scene"]["detection_ok"] = False
                metrics["last_perception_scene"]["rejected_reason"] = "implausible_tabletop_object_z"
                metrics["last_perception_scene"]["rejected_position"] = perceived.tolist()
                metrics["last_perception_scene"]["max_tabletop_z"] = max_tabletop_z
                metrics["perceived_position"] = None
                metrics["observed_pos"] = None
                _clear_context(experiment)
                scene.detection_ok = False
                scene.apple_pos = None
                scene.apple_quat = None
                scene.raw_points_world = None
                scene.raw_points_camera = None
                return scene
        metrics.setdefault("detected_objects", {})[target_class] = perceived.tolist()
        if target_class in {"plate", "bowl", "container", "target"}:
            metrics["transport_target_class"] = target_class
            metrics["transport_target_pos"] = perceived.tolist()
            metrics["place_target_class"] = target_class
            metrics["place_target_pos"] = perceived.tolist()
        else:
            metrics["perceived_position"] = perceived.tolist()
            metrics["observed_pos"] = perceived.tolist()
            metrics["object_observed_pos"] = perceived.tolist()
        experiment._last_perceived_scene = scene
        experiment._last_perceived_scene_target = target_class
        if target_class not in {"plate", "bowl", "container", "target"}:
            _clear_context(experiment)
    return scene


def build_fixed_vertical_grasp_source(
    experiment: Any,
    target_class: str = "apple",
    work_dir: str | Path | None = None,
    params: dict[str, Any] | None = None,
) -> FixedVerticalGraspContext:
    scene = getattr(experiment, "_last_perceived_scene", None)
    scene_target = getattr(experiment, "_last_perceived_scene_target", None)
    if scene is None or scene_target != target_class:
        raise RuntimeError("object_detection_unavailable_run_detect_object_pose_first")
    last_scene = _metrics(experiment).get("last_perception_scene", {})
    metrics_rejected = isinstance(last_scene, dict) and not bool(last_scene.get("detection_ok", True))
    if metrics_rejected or not scene.detection_ok or scene.raw_points_world is None or len(scene.raw_points_world) < 8:
        raise RuntimeError("perception_points_unavailable")

    raw_points = np.asarray(scene.raw_points_world, dtype=np.float64)
    try:
        completion = complete_point_cloud(raw_points, None, _completion_config(params))
        completed_points = np.asarray(completion.completed_points, dtype=np.float64)
        completion_report = completion.report
        completion_status = "completed"
    except Exception as exc:  # noqa: BLE001
        completed_points = raw_points
        completion_report = {"error": str(exc), "fallback": "raw_points"}
        completion_status = "fallback_raw"

    context = _build_fixed_vertical_grasp_context(
        target_class=target_class,
        raw_points_world=raw_points,
        completed_points_world=completed_points,
        params=params,
        completion_report=completion_report,
        completion_status=completion_status,
        scene=scene,
    )
    _install_context(experiment, context)
    return context


def detect_object_and_create_fixed_vertical_grasp(
    experiment: Any,
    params: dict[str, Any] | None = None,
    default_pregrasp_height: float = 0.127,
) -> FixedVerticalGraspContext:
    """Compatibility wrapper for older callers; public skills should use split steps."""
    params = params if isinstance(params, dict) else {}
    target_class = str(params.get("target_class") or "apple").lower()
    detect_object(experiment, target_class=target_class, work_dir=params.get("work_dir"))
    build_fixed_vertical_grasp_source(experiment, target_class=target_class, params=params)
    return create_fixed_vertical_grasp(experiment, params, default_pregrasp_height)


def create_fixed_vertical_grasp(
    experiment: Any,
    params: dict[str, Any] | None = None,
    default_pregrasp_height: float = 0.127,
) -> FixedVerticalGraspContext:
    params = params if isinstance(params, dict) else {}
    target_class = str(params.get("target_class") or "apple").lower()
    context = getattr(experiment, "_fixed_vertical_grasp_context", None)
    if not isinstance(context, FixedVerticalGraspContext) or context.target_class != target_class:
        raise RuntimeError("grasp_context_unavailable_run_create_fixed_vertical_grasp_after_detect_object_pose")

    pregrasp_height = float(params.get("pregrasp_height", default_pregrasp_height))
    mapping = install_fixed_vertical_tcp_pose(
        experiment,
        context.target_pos,
        top_points_world=context.top_points_world,
        pregrasp_height=pregrasp_height,
    )
    context.pregrasp_pos = np.asarray(mapping["T_pregrasp"].t, dtype=np.float64)
    context.report.update(tcp_mapping_report(mapping))
    context.report["semantic_grasp_pos"] = context.target_pos.tolist()
    _install_context(experiment, context)
    return context


def _build_fixed_vertical_grasp_context(
    *,
    target_class: str,
    raw_points_world: np.ndarray,
    completed_points_world: np.ndarray,
    params: dict[str, Any] | None,
    completion_report: dict[str, Any],
    completion_status: str,
    scene: Any,
) -> FixedVerticalGraspContext:
    params = params if isinstance(params, dict) else {}
    top_band_m = float(params.get("top_band_m", DEFAULT_TOP_BAND_M))
    fallback_quantile = float(params.get("top_fallback_quantile", DEFAULT_TOP_QUANTILE))
    descend_m = float(params.get("grasp_descend_m", DEFAULT_GRASP_DESCEND_M))
    pregrasp_height = float(params.get("pregrasp_height", 0.127))

    completed = np.asarray(completed_points_world, dtype=np.float64)
    z_vals = completed[:, 2]
    z_max = float(np.max(z_vals))
    top_mask = z_vals >= z_max - top_band_m
    if int(np.count_nonzero(top_mask)) < 16:
        z_threshold = float(np.quantile(z_vals, fallback_quantile))
        top_mask = z_vals >= z_threshold
    top_points = completed[top_mask]
    if len(top_points) < 8:
        top_points = completed

    xy_center = np.median(top_points[:, :2], axis=0)
    z_min = float(np.min(completed[:, 2]))
    target_z = float(np.clip(z_max - descend_m, z_min + 0.01, z_max))
    target_pos = np.array([xy_center[0], xy_center[1], target_z], dtype=np.float64)
    pregrasp_pos = target_pos + np.array([0.0, 0.0, pregrasp_height], dtype=np.float64)

    report = {
        "target_class": target_class,
        "selection_source": "fixed_vertical_top_surface",
        "orientation_policy": "fixed_downward_no_graspnet",
        "completion_status": completion_status,
        "raw_points": int(len(raw_points_world)),
        "completed_points": int(len(completed_points_world)),
        "top_points": int(len(top_points)),
        "z_min": z_min,
        "z_max": z_max,
        "target_pos": target_pos.tolist(),
        "pregrasp_pos": pregrasp_pos.tolist(),
        "top_xy_center": xy_center.tolist(),
        "top_band_m": top_band_m,
        "grasp_descend_m": descend_m,
        "perception": {
            "confidence": float(getattr(scene, "confidence", 0.0)),
            "mask_nonzero": int(getattr(scene, "mask_nonzero", 0)),
            "rgb_path": getattr(scene, "rgb_path", None),
            "mask_path": getattr(scene, "mask_path", None),
        },
        "completion_report": completion_report,
    }
    return FixedVerticalGraspContext(
        target_class=target_class,
        target_pos=target_pos,
        pregrasp_pos=pregrasp_pos,
        raw_points_world=raw_points_world,
        completed_points_world=completed_points_world,
        top_points_world=top_points,
        report=report,
    )


def _install_context(experiment: Any, context: FixedVerticalGraspContext) -> None:
    experiment._fixed_vertical_grasp_context = context
    experiment._fixed_vertical_grasp_valid = True
    metrics = _metrics(experiment)
    metrics["fixed_vertical_grasp"] = context.report
    metrics["grasp_target_pos"] = context.target_pos.tolist()
    metrics["semantic_grasp_pos"] = context.target_pos.tolist()
    metrics["tcp_grasp_pos"] = context.report.get("tcp_grasp_pos")
    metrics["tcp_pregrasp_pos"] = context.report.get("tcp_pregrasp_pos")
    metrics["grasp_orientation_policy"] = "fixed_downward_no_graspnet"


def _clear_context(experiment: Any) -> None:
    experiment._fixed_vertical_grasp_context = None
    experiment.T_wo = None
    experiment.T_pregrasp = None
    experiment._fixed_vertical_tcp_mapping = None
    experiment._fixed_vertical_grasp_valid = False
