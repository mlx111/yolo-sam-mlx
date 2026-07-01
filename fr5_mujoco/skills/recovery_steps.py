from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any

import mujoco
import numpy as np


Q_HOME = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, -0.7853], dtype=float)
Q_CAMERA_READY = np.array([0.17417, -1.18453, -0.48640, -2.49164, 1.88242, -0.23681], dtype=float)
DEFAULT_PREGRASP_HEIGHT = 0.08
DEFAULT_FIXED_VERTICAL_YAW = -0.393


@dataclass(frozen=True)
class Pose:
    t: np.ndarray
    quat: np.ndarray

    @property
    def R(self) -> np.ndarray:
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, self.quat)
        return mat.reshape(3, 3)


def _params(params: dict | None) -> dict:
    return params if isinstance(params, dict) else {}


def _require_float(params: dict[str, Any], key: str, skill: str) -> float:
    if key not in params:
        raise ValueError(f"{skill} requires parameter {key}")
    try:
        return float(params[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{skill} parameter {key} must be a float") from exc


def _require_offset(params: dict[str, Any], skill: str) -> np.ndarray:
    return np.array(
        [
            _require_float(params, "dx", skill),
            _require_float(params, "dy", skill),
            _require_float(params, "dz", skill),
        ],
        dtype=float,
    )


def _record(experiment: Any, skill: str, success: bool = True, reason: str = "ok", **extra: Any) -> None:
    if hasattr(experiment, "_record_basic_skill"):
        experiment._record_basic_skill(skill, success, reason, **extra)


def _tcp_pose(experiment: Any) -> Pose:
    return Pose(np.asarray(experiment.tcp_pos, dtype=float).reshape(3), np.asarray(experiment.tcp_quat, dtype=float).reshape(4))


def _quat_from_rotation(rotation: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=float).reshape(9))
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def _fixed_vertical_tcp_quat(yaw: float = DEFAULT_FIXED_VERTICAL_YAW) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    # TCP local Z points downward in world; local X/Y stay horizontal.
    rotation = np.array(
        [
            [c, s, 0.0],
            [s, -c, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    return _quat_from_rotation(rotation)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _fr5_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_grounded_sam2_segmenter():
    backend_path = _repo_root() / "skills" / "backends" / "grounded_sam2_backend.py"
    if not backend_path.exists():
        raise FileNotFoundError(f"GroundedSAM2 backend not found: {backend_path}")
    spec = importlib.util.spec_from_file_location("_fr5_grounded_sam2_backend", backend_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load GroundedSAM2 backend: {backend_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GroundedSAM2Segmenter


def _detected_object(experiment: Any, target_class: str) -> dict[str, Any] | None:
    metrics = getattr(experiment, "metrics", {}) or {}
    detected = metrics.get("detected_objects") if isinstance(metrics, dict) else None
    if not isinstance(detected, dict):
        return None
    target_key = str(target_class).strip().lower()
    for key, value in detected.items():
        if str(key).strip().lower() == target_key and isinstance(value, dict):
            return value
    return None


def _target_pos_from_params_or_detection(experiment: Any, params: dict[str, Any], skill: str) -> tuple[np.ndarray, str | None]:
    if "target_pos" in params:
        return np.asarray(params["target_pos"], dtype=float).reshape(3), None
    target_class = str(params.get("target_class") or params.get("target") or "apple")
    detected = _detected_object(experiment, target_class)
    if detected is None:
        raise RuntimeError(f"{skill} requires target_pos or a detected object for target_class={target_class!r}")
    if "position" not in detected:
        raise RuntimeError(f"detected object {target_class!r} does not contain position")
    return np.asarray(detected["position"], dtype=float).reshape(3), target_class


def camera_image(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    if not hasattr(experiment, "render_camera_rgbd"):
        raise RuntimeError("camera_rgbd_save requires experiment.render_camera_rgbd")
    result = experiment.render_camera_rgbd(
        str(params.get("camera_name", "ee_camera")),
        width=int(params.get("width", 640)),
        height=int(params.get("height", 480)),
        output_dir=params.get("output_dir") or _fr5_root() / "output" / "camera_rgbd",
        prefix=str(params.get("prefix", "fr5_ee_camera")),
    )
    _record(experiment, "camera_rgbd_save", True, "rgbd_saved", **result)


def detect_object(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    target_class = str(params.get("target_class", "")).strip()
    if not target_class:
        raise ValueError("detect_object_pose requires parameter target_class")
    metrics = getattr(experiment, "metrics", {}) or {}
    camera = params.get("camera") or metrics.get("last_camera_rgbd")
    if not isinstance(camera, dict):
        raise RuntimeError("detect_object_pose requires camera_rgbd_save to run first")

    image_path = str(params.get("image_path") or camera["rgb_path"])
    output_dir = Path(str(params.get("output_dir") or _fr5_root() / "output" / "grounded_sam2"))
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_target = target_class.replace(" ", "_")
    mask_path = str(params.get("output_mask_path") or output_dir / f"{Path(image_path).stem}_{safe_target}_mask.png")
    annotated_path = str(params.get("output_annotated_path") or output_dir / f"{Path(image_path).stem}_{safe_target}_annotated.png")

    segmenter_cls = _load_grounded_sam2_segmenter()
    segmenter = segmenter_cls(
        grounded_sam2_root=str(params.get("grounded_sam2_root", "Grounded-SAM-2")),
        sam2_checkpoint=params.get("sam2_checkpoint"),
        sam2_model_config=str(params.get("sam2_model_config", "configs/sam2.1/sam2.1_hiera_l.yaml")),
        grounding_dino_config=params.get("grounding_dino_config"),
        grounding_dino_checkpoint=params.get("grounding_dino_checkpoint"),
        bert_path=params.get("bert_path"),
        box_threshold=float(params.get("box_threshold", 0.2)),
        text_threshold=float(params.get("text_threshold", 0.2)),
        device=params.get("device"),
        multimask_output=bool(params.get("multimask_output", True)),
    )
    seg_result = segmenter.segment_image(image_path=image_path, target_class=target_class, output_mask_path=mask_path, output_annotated_path=annotated_path)
    if seg_result is None:
        _record(experiment, "detect_object_pose", False, "target_not_detected", target_class=target_class, image_path=image_path)
        return

    depth_path = str(params.get("depth_npy_path") or camera["depth_npy_path"])
    depth_m = np.load(depth_path)
    if not hasattr(experiment, "point_cloud_from_camera_mask"):
        raise RuntimeError("detect_object_pose requires experiment.point_cloud_from_camera_mask")
    pcd = experiment.point_cloud_from_camera_mask(
        seg_result.mask,
        depth_m,
        str(params.get("camera_name") or camera.get("camera_name", "ee_camera")),
        intrinsics=camera.get("intrinsics"),
        max_points=int(params.get("max_points", 20000)),
        max_depth_m=float(params.get("max_depth_m", 10.0)),
    )
    success = pcd.get("state") == "success"
    observed_pos = [float(pcd["x"]), float(pcd["y"]), float(pcd["z"])] if success else None
    detected_record = {
        "target_class": target_class,
        "position": observed_pos,
        "mask_path": seg_result.mask_path,
        "annotated_path": seg_result.annotated_path,
        "candidate": seg_result.candidate,
        "candidate_count": len(seg_result.candidates),
        "point_cloud_info": {key: value for key, value in pcd.items() if key != "point_cloud"},
        "camera": camera,
    }
    metrics.setdefault("detected_objects", {})[target_class] = detected_record
    metrics["last_perception_scene"] = detected_record
    metrics["last_detected_object"] = detected_record
    if observed_pos is not None:
        metrics["observed_pos"] = observed_pos
        metrics["perceived_position"] = observed_pos
        metrics["object_observed_pos"] = observed_pos
    _record(
        experiment,
        "detect_object_pose",
        success,
        "object_pose_detected" if success else str(pcd.get("info", "point_cloud_failed")),
        target_class=target_class,
        observed_pos=observed_pos,
        mask_path=seg_result.mask_path,
        annotated_path=seg_result.annotated_path,
        point_cloud_info={key: value for key, value in pcd.items() if key != "point_cloud"},
    )


def create_grasp(experiment: Any, params: dict | None = None, default_pregrasp_height: float = DEFAULT_PREGRASP_HEIGHT) -> None:
    params = _params(params)
    target_pos, detected_target = _target_pos_from_params_or_detection(experiment, params, "create_fixed_vertical_grasp")
    if "quat" in params:
        quat = np.asarray(params["quat"], dtype=float).reshape(4)
        orientation_source = "explicit_quat"
    else:
        yaw = float(params.get("yaw", DEFAULT_FIXED_VERTICAL_YAW))
        quat = _fixed_vertical_tcp_quat(yaw)
        orientation_source = "fixed_vertical_down"
    experiment.T_wo = Pose(target_pos, quat)
    height = float(params.get("pregrasp_height", default_pregrasp_height))
    experiment.T_pregrasp = Pose(target_pos + np.array([0.0, 0.0, height], dtype=float), quat)
    metrics = getattr(experiment, "metrics", {}) or {}
    metrics["grasp_target_pos"] = target_pos.tolist()
    metrics["tcp_grasp_pos"] = experiment.T_wo.t.tolist()
    metrics["tcp_pregrasp_pos"] = experiment.T_pregrasp.t.tolist()
    _record(
        experiment,
        "create_fixed_vertical_grasp",
        True,
        "fixed_vertical_grasp_created",
        target_class=detected_target,
        target_pos=target_pos.tolist(),
        pregrasp_height=height,
        orientation_source=orientation_source,
        yaw=float(yaw) if "quat" not in params else None,
        quat=quat.tolist(),
    )


def move_pregrasp(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    offset = _require_offset(params, "move_to_pregrasp")
    duration = float(params.get("duration", 1.0))
    if not hasattr(experiment, "T_wo") or experiment.T_wo is None:
        raise RuntimeError("move_to_pregrasp requires T_wo to be set first")
    experiment.T_pregrasp = Pose(np.asarray(experiment.T_wo.t, dtype=float) + offset, np.asarray(experiment.T_wo.quat, dtype=float))
    result = experiment._move_cartesian(experiment.T_pregrasp, duration)
    if hasattr(experiment, "_record_skill_result"):
        experiment._record_skill_result(result)


def move_grasp(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    offset = _require_offset(params, "approach_object")
    duration = float(params.get("duration", 1.0))
    if not hasattr(experiment, "T_wo") or experiment.T_wo is None:
        raise RuntimeError("approach_object requires T_wo to be set first")
    experiment.T_wo = Pose(np.asarray(experiment.T_wo.t, dtype=float) + offset, np.asarray(experiment.T_wo.quat, dtype=float))
    result = experiment._move_cartesian(experiment.T_wo, duration)
    if hasattr(experiment, "_record_skill_result"):
        experiment._record_skill_result(result)
    experiment._step_n(int(params.get("settle_steps", 500)))
    _record(
        experiment,
        "approach_object",
        bool(getattr(result, "success", True)),
        "approach_executed" if bool(getattr(result, "success", True)) else str(getattr(result, "reason", "failed")),
        tcp_target_pos=experiment.T_wo.t.tolist(),
    )


def gripper_action(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    state = int(params.get("state", 0))
    if state not in {0, 1}:
        raise ValueError(f"gripper state must be 0 or 1, got {state!r}")
    if state == 0:
        result = experiment._gripper_open(float(params.get("duration", 0.5)))
        if hasattr(experiment, "_record_skill_result"):
            experiment._record_skill_result(result)
        _record(experiment, "open_gripper", True, "gripper_opened")
        return
    result = experiment._gripper_close(float(params.get("duration", 0.5)))
    if hasattr(experiment, "_record_skill_result"):
        experiment._record_skill_result(result)
    contact = experiment._contact_summary() if hasattr(experiment, "_contact_summary") else {}
    has_contact = bool(contact.get("left_contact") or contact.get("right_contact"))
    _record(
        experiment,
        "close_gripper",
        True,
        "physical_contact_detected" if has_contact else "closed_without_contact",
        contact=contact,
    )


def vertical_grasp(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    lift_height = _require_float(params, "lift_height", "lift")
    duration = float(params.get("duration", 1.0))
    start = experiment.T_wo if getattr(experiment, "T_wo", None) is not None else _tcp_pose(experiment)
    target = Pose(np.asarray(start.t, dtype=float) + np.array([0.0, 0.0, lift_height], dtype=float), np.asarray(start.quat, dtype=float))
    result = experiment._move_cartesian(target, duration)
    experiment.T_wo = target
    if hasattr(experiment, "_record_skill_result"):
        experiment._record_skill_result(result)
    _record(
        experiment,
        "lift",
        bool(getattr(result, "success", True)),
        "tcp_lifted" if bool(getattr(result, "success", True)) else str(getattr(result, "reason", "failed")),
        lift_height=lift_height,
        tcp_target_pos=target.t.tolist(),
    )


def execute_init(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    result = experiment._move_joints(np.asarray(params.get("q", Q_HOME), dtype=float), float(params.get("duration", 1.0)))
    if hasattr(experiment, "_record_skill_result"):
        experiment._record_skill_result(result)


def execute_camera_ready(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    q_target = np.asarray(params.get("q", Q_CAMERA_READY), dtype=float)
    result = experiment._move_joints(q_target, float(params.get("duration", 1.5)))
    if hasattr(experiment, "_record_skill_result"):
        experiment._record_skill_result(result)
    metrics = getattr(experiment, "metrics", {}) or {}
    metrics["camera_ready_q"] = q_target.reshape(6).tolist()
    _record(
        experiment,
        "go_camera_ready",
        bool(getattr(result, "success", True)),
        "camera_ready_reached" if bool(getattr(result, "success", True)) else str(getattr(result, "reason", "failed")),
        target_joint=q_target.reshape(6).tolist(),
    )


def move_lifted_object_to(experiment: Any, params: dict | None = None) -> None:
    params = _params(params)
    target_name = str(params.get("target") or params.get("target_class") or "").strip()
    if target_name:
        target_pos, detected_target = _target_pos_from_params_or_detection(experiment, {"target_class": target_name}, "move_lifted_object_to")
    elif "target_pos" in params:
        target_pos, detected_target = _target_pos_from_params_or_detection(experiment, params, "move_lifted_object_to")
    else:
        raise ValueError("move_lifted_object_to requires target, target_class, or target_pos")
    place_height = float(params.get("height", params.get("z", target_pos[2] + 0.08)))
    tcp_target = np.array([target_pos[0], target_pos[1], place_height], dtype=float)
    placement_offset_xy = np.zeros(2, dtype=float)
    compensate_held_object = bool(params.get("compensate_held_object", True))
    if compensate_held_object and hasattr(experiment, "model") and hasattr(experiment, "data"):
        object_geom = str(params.get("object_geom", "apple0"))
        geom_id = mujoco.mj_name2id(experiment.model, mujoco.mjtObj.mjOBJ_GEOM, object_geom)
        if geom_id >= 0:
            object_pos = np.asarray(experiment.data.geom_xpos[geom_id], dtype=float).reshape(3)
            tcp_pos = np.asarray(getattr(experiment, "tcp_pos", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
            placement_offset_xy = object_pos[:2] - tcp_pos[:2]
            tcp_target[:2] = target_pos[:2] - placement_offset_xy
    quat = np.asarray(params.get("quat", getattr(experiment, "tcp_quat", [1.0, 0.0, 0.0, 0.0])), dtype=float).reshape(4)
    target_pose = Pose(tcp_target, quat)
    duration = float(params.get("duration", 1.0))
    result = experiment._move_cartesian(target_pose, duration)
    experiment.T_wo = target_pose
    if hasattr(experiment, "_record_skill_result"):
        experiment._record_skill_result(result)
    _record(
        experiment,
        "move_lifted_object_to",
        bool(getattr(result, "success", True)),
        "moved_to_place_pose" if bool(getattr(result, "success", True)) else str(getattr(result, "reason", "failed")),
        target=detected_target,
        target_pos=target_pos.tolist(),
        tcp_target_pos=tcp_target.tolist(),
        placement_offset_xy=placement_offset_xy.tolist(),
    )
