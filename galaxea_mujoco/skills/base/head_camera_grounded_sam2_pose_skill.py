from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

import cv2
import mujoco
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from runtime_perception.warnings_filter import suppress_grounded_sam2_warnings

suppress_grounded_sam2_warnings()


MJ_CAMERA_FROM_CV = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


@dataclass(frozen=True)
class HeadCameraGroundedSAM2PoseResult:
    success: bool
    target_class: str
    object_body: str | None
    rgb_path: str
    depth_path: str
    metadata_path: str
    mask_path: str
    annotated_path: str
    report_path: str
    position_output_path: str
    point_cloud_path: str | None
    bbox_xyxy: list[float] | None
    mask_pixel_count: int
    valid_depth_count: int
    valid_depth_ratio: float
    center_camera_cv: list[float] | None
    center_world: list[float] | None
    median_world: list[float] | None
    bounds_world_min: list[float] | None
    bounds_world_max: list[float] | None
    reference_frame: str
    center_reference: list[float] | None
    median_reference: list[float] | None
    bounds_reference_min: list[float] | None
    bounds_reference_max: list[float] | None
    depth_stats_m: dict[str, float | None]
    candidate: dict[str, Any] | None
    candidate_count: int
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class R1ProHeadCameraGroundedSAM2PoseSkill:
    """Segment an object in saved head-camera RGB-D and estimate its world position."""

    def __init__(
        self,
        output_dir: str | Path = "output/head_camera_grounded_sam2_pose",
        grounded_sam2_root: str = "Grounded-SAM-2",
    ):
        self.output_dir = Path(output_dir)
        self.grounded_sam2_root = str(grounded_sam2_root)

    def detect_object_position(
        self,
        *,
        rgb_path: str | Path,
        depth_path: str | Path,
        metadata_path: str | Path,
        target_class: str,
        output_dir: str | Path | None = None,
        prefix: str | None = None,
        grounded_sam2_root: str | None = None,
        sam2_checkpoint: str | None = None,
        sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        grounding_dino_config: str | None = None,
        grounding_dino_checkpoint: str | None = None,
        bert_path: str | None = None,
        box_threshold: float = 0.2,
        text_threshold: float = 0.2,
        device: str | None = None,
        multimask_output: bool = True,
        min_valid_depth_points: int = 32,
        max_depth_m: float = 5.0,
        save_point_cloud: bool = True,
        position_output_path: str | Path | None = None,
        model: Any | None = None,
        data: Any | None = None,
        reference_frame: str = "torso_link4",
        object_body: str | None = None,
    ) -> HeadCameraGroundedSAM2PoseResult:
        rgb_path = Path(rgb_path)
        depth_path = Path(depth_path)
        metadata_path = Path(metadata_path)
        target_slug = _target_slug(target_class)
        runtime_tmp_dir = None
        if output_dir is None and position_output_path is None:
            runtime_tmp_dir = _runtime_tmp_dir_from_paths(rgb_path, depth_path, metadata_path)
        out_dir = _object_position_dir(target_slug, runtime_tmp_dir) if output_dir is None else Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = prefix or target_slug

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        depth = np.load(depth_path).astype(np.float32)

        mask_path = out_dir / f"{prefix}_mask.png"
        annotated_path = out_dir / f"{prefix}_annotated.png"
        report_path = out_dir / f"{prefix}_pose_report.json"
        position_output_path = Path(position_output_path) if position_output_path is not None else out_dir / "object_position.json"
        point_cloud_path = out_dir / f"{prefix}_points_world.npy" if save_point_cloud else None

        segmenter_cls = _load_root_grounded_sam2_segmenter()
        segmenter = segmenter_cls(
            grounded_sam2_root=grounded_sam2_root or self.grounded_sam2_root,
            sam2_checkpoint=sam2_checkpoint,
            sam2_model_config=sam2_model_config,
            grounding_dino_config=grounding_dino_config,
            grounding_dino_checkpoint=grounding_dino_checkpoint,
            bert_path=bert_path,
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            device=device,
            multimask_output=bool(multimask_output),
        )
        seg_result = segmenter.segment_image(
            image_path=str(rgb_path),
            target_class=str(target_class),
            output_mask_path=str(mask_path),
            output_annotated_path=str(annotated_path),
        )
        if seg_result is None:
            result = HeadCameraGroundedSAM2PoseResult(
                success=False,
                target_class=str(target_class),
                object_body=str(object_body) if object_body is not None else None,
                rgb_path=str(rgb_path),
                depth_path=str(depth_path),
                metadata_path=str(metadata_path),
                mask_path=str(mask_path),
                annotated_path=str(annotated_path),
                report_path=str(report_path),
                position_output_path=str(position_output_path),
                point_cloud_path=str(point_cloud_path) if point_cloud_path is not None else None,
                bbox_xyxy=None,
                mask_pixel_count=0,
                valid_depth_count=0,
                valid_depth_ratio=0.0,
                center_camera_cv=None,
                center_world=None,
                median_world=None,
                bounds_world_min=None,
                bounds_world_max=None,
                reference_frame=str(reference_frame),
                center_reference=None,
                median_reference=None,
                bounds_reference_min=None,
                bounds_reference_max=None,
                depth_stats_m={"min": None, "max": None, "median": None},
                candidate=None,
                candidate_count=0,
                message=f"target not detected by Grounded-SAM2: {target_class}",
            )
            position_output_path.parent.mkdir(parents=True, exist_ok=True)
            position_output_path.write_text(json.dumps(_compact_position(result.to_dict()), ensure_ascii=False, indent=2), encoding="utf-8")
            report_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            return result

        mask = np.asarray(seg_result.mask) > 0
        points_camera_cv, valid_depth_values = _backproject_masked_depth(
            depth,
            mask,
            metadata["intrinsics"],
            max_depth_m=float(max_depth_m),
        )
        points_world = _camera_cv_points_to_world(points_camera_cv, metadata)
        if point_cloud_path is not None and len(points_world) > 0:
            np.save(point_cloud_path, points_world.astype(np.float32))

        points_reference = _world_points_to_reference_frame(model, data, points_world, str(reference_frame))
        success = int(len(points_world)) >= int(min_valid_depth_points)
        center_camera_cv = _rounded(np.mean(points_camera_cv, axis=0)) if len(points_camera_cv) else None
        center_world = _rounded(np.mean(points_world, axis=0)) if len(points_world) else None
        median_world = _rounded(np.median(points_world, axis=0)) if len(points_world) else None
        bounds_world_min = _rounded(np.min(points_world, axis=0)) if len(points_world) else None
        bounds_world_max = _rounded(np.max(points_world, axis=0)) if len(points_world) else None
        center_reference = _rounded(np.mean(points_reference, axis=0)) if len(points_reference) else None
        median_reference = _rounded(np.median(points_reference, axis=0)) if len(points_reference) else None
        bounds_reference_min = _rounded(np.min(points_reference, axis=0)) if len(points_reference) else None
        bounds_reference_max = _rounded(np.max(points_reference, axis=0)) if len(points_reference) else None

        mask_pixel_count = int(np.count_nonzero(mask))
        result = HeadCameraGroundedSAM2PoseResult(
            success=success,
            target_class=str(target_class),
            object_body=str(object_body) if object_body is not None else None,
            rgb_path=str(rgb_path),
            depth_path=str(depth_path),
            metadata_path=str(metadata_path),
            mask_path=str(mask_path),
            annotated_path=str(annotated_path),
            report_path=str(report_path),
            position_output_path=str(position_output_path),
            point_cloud_path=str(point_cloud_path) if point_cloud_path is not None else None,
            bbox_xyxy=_candidate_bbox(seg_result.candidate),
            mask_pixel_count=mask_pixel_count,
            valid_depth_count=int(len(points_world)),
            valid_depth_ratio=float(len(points_world) / max(mask_pixel_count, 1)),
            center_camera_cv=center_camera_cv,
            center_world=center_world,
            median_world=median_world,
            bounds_world_min=bounds_world_min,
            bounds_world_max=bounds_world_max,
            reference_frame=str(reference_frame),
            center_reference=center_reference,
            median_reference=median_reference,
            bounds_reference_min=bounds_reference_min,
            bounds_reference_max=bounds_reference_max,
            depth_stats_m=_depth_stats(valid_depth_values),
            candidate=_jsonable(seg_result.candidate),
            candidate_count=int(len(seg_result.candidates)),
            message="object position estimated from head camera RGB-D" if success else "too few valid depth points in mask",
        )
        position_output_path.parent.mkdir(parents=True, exist_ok=True)
        position_output_path.write_text(json.dumps(_compact_position(result.to_dict()), ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def execute_recovery_action(self, _model: Any, _data: Any, params: dict[str, Any]) -> HeadCameraGroundedSAM2PoseResult:
        target_class = str(params.get("target_class", "apple"))
        runtime_tmp_dir = params.get("_runtime_tmp_dir")
        return self.detect_object_position(
            rgb_path=params.get("rgb_path", _default_capture_path("rgb", runtime_tmp_dir)),
            depth_path=params.get("depth_path", _default_capture_path("depth", runtime_tmp_dir)),
            metadata_path=params.get("metadata_path", _default_capture_path("metadata", runtime_tmp_dir)),
            target_class=target_class,
            output_dir=params.get("output_dir", _object_position_dir(_target_slug(target_class), runtime_tmp_dir) if runtime_tmp_dir else None),
            prefix=params.get("prefix"),
            grounded_sam2_root=params.get("grounded_sam2_root"),
            sam2_checkpoint=params.get("sam2_checkpoint"),
            sam2_model_config=str(params.get("sam2_model_config", "configs/sam2.1/sam2.1_hiera_l.yaml")),
            grounding_dino_config=params.get("grounding_dino_config"),
            grounding_dino_checkpoint=params.get("grounding_dino_checkpoint"),
            bert_path=params.get("bert_path"),
            box_threshold=float(params.get("box_threshold", 0.2)),
            text_threshold=float(params.get("text_threshold", 0.2)),
            device=params.get("device"),
            multimask_output=bool(params.get("multimask_output", True)),
            min_valid_depth_points=int(params.get("min_valid_depth_points", 16)),
            max_depth_m=float(params.get("max_depth_m", 5.0)),
            save_point_cloud=bool(params.get("save_point_cloud", True)),
            position_output_path=params.get("position_output_path"),
            model=_model,
            data=_data,
            reference_frame=str(params.get("reference_frame", "torso_link4")),
            object_body=params.get("object_body"),
        )


def _load_root_grounded_sam2_segmenter():
    repo_root = Path(__file__).resolve().parents[3]
    backend_path = repo_root / "skills" / "backends" / "grounded_sam2_backend.py"
    if not backend_path.exists():
        raise FileNotFoundError(f"Grounded-SAM2 backend not found: {backend_path}")
    spec = importlib.util.spec_from_file_location("_root_grounded_sam2_backend", backend_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Grounded-SAM2 backend: {backend_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.GroundedSAM2Segmenter


def _target_slug(target_class: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(target_class).strip().lower()).strip("_")
    return slug or "object"


def _object_position_dir(target_slug: str, runtime_tmp_dir: Any | None = None) -> Path:
    if runtime_tmp_dir:
        return Path(str(runtime_tmp_dir)) / "object_positions" / target_slug
    return Path("output/object_positions") / target_slug


def _runtime_tmp_dir_from_paths(*paths: str | Path) -> Path | None:
    for path in paths:
        parts = Path(path).parts
        if "tmp" in parts:
            index = parts.index("tmp")
            if index + 1 < len(parts) and parts[index + 1].startswith("field_atomic_run_"):
                return Path(*parts[: index + 2])
    return None


def _default_capture_path(kind: str, runtime_tmp_dir: Any | None = None) -> str:
    names = {
        "rgb": ("head_top_rgb.png", "head_top_rgb.png"),
        "depth": ("head_top_depth.npy", "head_top_depth.npy"),
        "metadata": ("head_top_metadata.json", "head_top_metadata.json"),
    }
    filename, fallback = names[kind]
    if runtime_tmp_dir:
        return str(Path(str(runtime_tmp_dir)) / "head_camera_rgbd" / filename)
    preferred = Path("output/left_head_camera_rgbd") / filename
    if preferred.exists():
        return str(preferred)
    return str(Path("output/head_camera_rgbd") / fallback)


def _backproject_masked_depth(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict[str, float | int],
    *,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if depth_m.shape != mask.shape:
        raise ValueError(f"depth/mask shape mismatch: depth={depth_m.shape}, mask={mask.shape}")
    valid = mask & np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= max_depth_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.float64)
    v, u = np.where(valid)
    z = depth_m[valid].astype(np.float64)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    return np.column_stack([x, y, z]), z


def _camera_cv_points_to_world(points_camera_cv: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    if len(points_camera_cv) == 0:
        return np.empty((0, 3), dtype=np.float64)
    camera_position = np.asarray(metadata["camera_position"], dtype=np.float64).reshape(3)
    camera_xmat = np.asarray(metadata["camera_xmat"], dtype=np.float64).reshape(3, 3)
    rotation_world_from_cv = camera_xmat @ MJ_CAMERA_FROM_CV
    return (rotation_world_from_cv @ points_camera_cv.T).T + camera_position.reshape(1, 3)


def _world_points_to_reference_frame(model: Any | None, data: Any | None, points_world: np.ndarray, frame_name: str) -> np.ndarray:
    if len(points_world) == 0:
        return np.empty((0, 3), dtype=np.float64)
    if model is None or data is None:
        return np.empty((0, 3), dtype=np.float64)
    name = str(frame_name or "world").strip()
    if not name or name == "world":
        return np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id >= 0:
        frame_pos = data.xpos[body_id].copy()
        frame_xmat = data.xmat[body_id].reshape(3, 3).copy()
    else:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            return np.empty((0, 3), dtype=np.float64)
        frame_pos = data.site_xpos[site_id].copy()
        frame_xmat = data.site_xmat[site_id].reshape(3, 3).copy()
    return (frame_xmat.T @ (np.asarray(points_world, dtype=np.float64).reshape(-1, 3) - frame_pos.reshape(1, 3)).T).T


def _candidate_bbox(candidate: dict[str, Any] | None) -> list[float] | None:
    if not candidate or candidate.get("xyxy") is None:
        return None
    return [float(v) for v in candidate["xyxy"]]


def _depth_stats(depth: np.ndarray) -> dict[str, float | None]:
    if depth.size == 0:
        return {"min": None, "max": None, "median": None}
    return {
        "min": float(np.min(depth)),
        "max": float(np.max(depth)),
        "median": float(np.median(depth)),
    }


def _rounded(values: np.ndarray, digits: int = 9) -> list[float]:
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _compact_position(payload: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "success": bool(payload.get("success", False)),
        "target_class": payload.get("target_class"),
        "reference_frame": payload.get("reference_frame"),
        "center_reference": payload.get("center_reference"),
        "median_reference": payload.get("median_reference"),
        "center_world": payload.get("center_world"),
        "median_world": payload.get("median_world"),
        "center_camera_cv": payload.get("center_camera_cv"),
        "bbox_xyxy": payload.get("bbox_xyxy"),
    }
    if payload.get("object_body") is not None:
        compact["object_body"] = payload.get("object_body")
    return compact


def load_skill(
    output_dir: str | Path = "output/head_camera_grounded_sam2_pose",
    grounded_sam2_root: str = "Grounded-SAM-2",
) -> R1ProHeadCameraGroundedSAM2PoseSkill:
    return R1ProHeadCameraGroundedSAM2PoseSkill(output_dir=output_dir, grounded_sam2_root=grounded_sam2_root)
