from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from runtime_perception.warnings_filter import suppress_grounded_sam2_warnings

suppress_grounded_sam2_warnings()

from runtime_perception.backends.grounded_sam2_backend import GroundedSAM2Segmenter
from runtime_perception.backends.pointcloud_backend import (
    DEFAULT_CAMERA_EXTRINSICS,
    DEFAULT_CAMERA_INTRINSICS,
    PointCloudGenerator,
)
from runtime_perception.buquan_completion import build_buquan_completed_stl
from dong2 import generate_scene


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_COLOR_IMAGE = ROOT_DIR / "inputs" / "color_2026-06-24T02_33_20.png"
DEFAULT_DEPTH_IMAGE = ROOT_DIR / "inputs" / "depth_2026-06-24T02_33_20.png"
DEFAULT_SCENE_OUT = ROOT_DIR / "scence" / "initial_runtime_scene.xml"
DEFAULT_REPORT_OUT = ROOT_DIR / "scence" / "initial_runtime_scene_report.json"
DEFAULT_OBSERVATION_OUT = ROOT_DIR / "scence" / "initial_runtime_observation.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "runtime_scenes" / "real_camera_runtime"
DEFAULT_GROUNDED_SAM2_ROOT = ROOT_DIR.parent / "Grounded-SAM-2"
DEFAULT_OBJECTS = ("apple", "red box")
DEFAULT_BUQUAN_DEPTH_BAND_TOLERANCE_MM = 40.0
MESH_ROTATION_MODE = "world_x180_current_x180"


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    root_relative = (ROOT_DIR / path).resolve()
    if root_relative.exists():
        return root_relative
    return root_relative


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
    return depth


def _jsonable(value: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        if value.ndim >= 2:
            return {"array_shape": list(value.shape), "array_dtype": str(value.dtype)}
        return value.tolist()
    if hasattr(value, "item") and value.__class__.__module__.startswith("numpy"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _matrix_to_quat_wxyz(rotation_matrix: np.ndarray) -> list[float]:
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def _camera_to_target_mesh_quat(target_coordinate_system: str) -> list[float]:
    """Rotate restored camera-frame STL geometry into the generated scene frame."""
    local_x180 = np.diag([1.0, -1.0, -1.0])
    world_x180 = np.diag([1.0, -1.0, -1.0])
    if target_coordinate_system == "camera":
        return _matrix_to_quat_wxyz(world_x180 @ local_x180)

    cam_to_tcp = PointCloudGenerator.pose_to_matrix(DEFAULT_CAMERA_EXTRINSICS["camera_to_tcp_pose"])
    rotation = cam_to_tcp[:3, :3]
    if target_coordinate_system in {"base", "world"}:
        tcp_to_base = PointCloudGenerator.pose_to_matrix(DEFAULT_CAMERA_EXTRINSICS["tcp_pose"])
        rotation = tcp_to_base[:3, :3] @ rotation
    if target_coordinate_system == "world":
        base_to_world = PointCloudGenerator.pose_to_matrix(DEFAULT_CAMERA_EXTRINSICS["base_to_world_pose"])
        rotation = base_to_world[:3, :3] @ rotation
    rotation = world_x180 @ (rotation @ local_x180)
    return _matrix_to_quat_wxyz(rotation)


def _object_body_name(object_name: str, index: int = 0) -> str:
    normalized = str(object_name).strip().lower().rstrip(".").replace(" ", "_")
    if normalized.endswith("box"):
        return f"{normalized}_{index}"
    return f"{normalized}{index}"


def _mesh_report_path(report: dict[str, Any]) -> str:
    if str(report.get("status") or "") != "success":
        return ""
    return str(report.get("installed_mesh") or report.get("restored_stl") or "")


def _build_initial_observation(
    *,
    scene_id: str,
    color_image_path: Path,
    depth_image_path: Path,
    objects: list[str],
    object_positions: dict[str, list[float]],
    settled_object_positions: dict[str, list[float]] | None,
    completed_mesh_reports: dict[str, Any],
    mesh_quats: dict[str, list[float]],
    target_coordinate_system: str,
    scene_out: Path,
    report_out: Path,
) -> dict[str, Any]:
    observation_objects: list[dict[str, Any]] = []
    for object_name in objects:
        normalized = object_name.replace(" ", "_")
        mesh_report = completed_mesh_reports.get(object_name, {})
        settled_pose = settled_object_positions.get(object_name) if settled_object_positions else None
        observation_objects.append({
            "name": _object_body_name(object_name),
            "class": object_name,
            "pose": object_positions.get(object_name, [0.0, 0.0, 0.0]),
            "settled_pose": settled_pose,
            "size": [0.03, 0.03, 0.03],
            "geom_type": "mesh" if _mesh_report_path(mesh_report) else "box",
            "mass": 0.05,
            "confidence": None,
            "source": "rgbd_grounded_sam2_buquan",
            "freejoint": True,
            "mesh_path": _mesh_report_path(mesh_report),
            "mesh_quat_wxyz": mesh_quats.get(normalized),
            "mesh_rotation_mode": MESH_ROTATION_MODE,
        })
    return {
        "schema_version": "field_runtime_scene_observation_v1",
        "scene_id": scene_id,
        "coordinate_frame": {
            "world_frame": target_coordinate_system,
            "units": "meter_radian",
            "notes": "Task-start observation generated from real RGB-D before anomaly replay.",
        },
        "robot_state": {},
        "objects": observation_objects,
        "obstacles": [],
        "place_zones": [],
        "sensor_refs": {
            "rgb_path": str(color_image_path.resolve()),
            "depth_path": str(depth_image_path.resolve()),
            "runtime_scene_path": str(scene_out.resolve()),
            "runtime_scene_report_path": str(report_out.resolve()),
        },
        "calibration": {
            "camera_intrinsics": DEFAULT_CAMERA_INTRINSICS,
            "camera_extrinsics": DEFAULT_CAMERA_EXTRINSICS,
        },
        "runtime_scene": {
            "scene_out": str(scene_out.resolve()),
            "target_object": observation_objects[0]["name"] if observation_objects else "",
            "generator": "build_runtime_scene_from_real_camera",
        },
        "metadata": {
            "table_policy": "fixed_scene2_template",
            "mesh_rotation_mode": MESH_ROTATION_MODE,
            "task_start_runtime_scene": True,
            "settled_pose_source": "mujoco_settle_after_scene_generation" if settled_object_positions else "",
        },
    }


def _settled_object_positions(scene_path: Path, objects: list[str], settle_steps: int) -> dict[str, list[float]]:
    if int(settle_steps) <= 0:
        return {}
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(int(settle_steps)):
        mujoco.mj_step(model, data)

    positions: dict[str, list[float]] = {}
    for object_name in objects:
        body_name = _object_body_name(object_name)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        positions[object_name] = np.round(data.xpos[body_id].copy(), 9).astype(float).tolist()
    return positions


def _largest_mask_component(mask: np.ndarray) -> np.ndarray:
    mask01 = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if count <= 1:
        return mask01
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    return (labels == largest_label).astype(np.uint8)


def _save_buquan_clean_mask(
    *,
    raw_mask: np.ndarray,
    depth_img: np.ndarray,
    output_path: Path,
    depth_band_tolerance_mm: float = DEFAULT_BUQUAN_DEPTH_BAND_TOLERANCE_MM,
) -> dict[str, Any]:
    mask01 = (raw_mask > 0).astype(np.uint8)
    if mask01.shape[:2] != depth_img.shape[:2]:
        mask01 = cv2.resize(mask01, (depth_img.shape[1], depth_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    depth = np.asarray(depth_img, dtype=np.float32)
    valid = (mask01 > 0) & np.isfinite(depth) & (depth > 0)
    raw_pixels = int(np.count_nonzero(mask01))
    if not np.any(valid):
        clean = _largest_mask_component(mask01)
        reason = "no_valid_depth_fallback_largest_component"
        median_depth = None
    else:
        median_depth = float(np.median(depth[valid]))
        tolerance = float(depth_band_tolerance_mm)
        if median_depth < 20.0:
            tolerance = tolerance / 1000.0
        depth_refined = mask01 & (np.abs(depth - median_depth) <= tolerance).astype(np.uint8)
        min_pixels = max(16, int(raw_pixels * 0.25))
        if int(np.count_nonzero(depth_refined)) < min_pixels:
            depth_refined = mask01
            reason = "depth_band_too_small_fallback_raw"
        else:
            reason = "depth_band_largest_component_morphology"
        kernel = np.ones((3, 3), dtype=np.uint8)
        clean = cv2.morphologyEx(depth_refined.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        if int(np.count_nonzero(clean)) < min_pixels:
            clean = depth_refined.astype(np.uint8)
        clean = _largest_mask_component(clean)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), clean.astype(np.uint8) * 255)
    return {
        "raw_mask_pixels": raw_pixels,
        "clean_mask_pixels": int(np.count_nonzero(clean)),
        "median_depth": median_depth,
        "depth_band_tolerance_mm": float(depth_band_tolerance_mm),
        "reason": reason,
    }


def _detect_object_position(
    *,
    segmenter: GroundedSAM2Segmenter,
    color_image_path: Path,
    color_img: np.ndarray,
    depth_img: np.ndarray,
    object_name: str,
    output_dir: Path,
    target_coordinate_system: str,
    downsample_scale: float,
) -> tuple[list[float], dict[str, Any]]:
    mask_path = output_dir / "masks" / f"{color_image_path.stem}_{object_name}_mask.png"
    clean_mask_path = output_dir / "masks" / f"{color_image_path.stem}_{object_name}_buquan_clean_mask.png"
    annotated_path = output_dir / "annotated" / f"{color_image_path.stem}_{object_name}_annotated.png"
    pcd_path = output_dir / "point_clouds" / f"{color_image_path.stem}_{object_name}_{target_coordinate_system}.npy"

    seg_result = segmenter.segment_image(
        image_path=str(color_image_path),
        target_class=object_name,
        output_mask_path=str(mask_path),
        output_annotated_path=str(annotated_path),
    )
    if seg_result is None:
        raise RuntimeError(f"Grounded-SAM2 did not detect object: {object_name}")
    clean_mask_stats = _save_buquan_clean_mask(
        raw_mask=seg_result.mask,
        depth_img=depth_img,
        output_path=clean_mask_path,
    )

    pcd_generator = PointCloudGenerator(
        camera_intrinsics=DEFAULT_CAMERA_INTRINSICS,
        camera_extrinsics=DEFAULT_CAMERA_EXTRINSICS,
        save_point_cloud=True,
        save_path=str(pcd_path),
        denoise=True,
        denoise_neighbors=10,
        denoise_std_ratio=5.0,
        use_dbscan=False,
    )
    pcd_result = pcd_generator.generate_point_cloud(
        color_image_aligned=color_img,
        depth_image_aligned=depth_img,
        mask=seg_result.mask,
        downsample_scale=downsample_scale,
        target_coordinate_system=target_coordinate_system,
    )
    if pcd_result.get("state") != "success":
        raise RuntimeError(f"Point cloud generation failed for {object_name}: {pcd_result.get('info')}")

    position = [
        float(pcd_result["x"]) / 1000.0,
        float(pcd_result["y"]) / 1000.0,
        float(pcd_result["z"]) / 1000.0,
    ]
    return position, {
        "mask_path": str(mask_path.resolve()),
        "buquan_clean_mask_path": str(clean_mask_path.resolve()),
        "buquan_clean_mask_stats": clean_mask_stats,
        "annotated_path": str(annotated_path.resolve()),
        "point_cloud_path": str(pcd_path.resolve()),
        "point_cloud_ply_path": str(pcd_path.with_suffix(".ply").resolve()),
        "candidate": _jsonable(seg_result.candidate),
        "candidate_count": len(seg_result.candidates),
        "point_cloud_info": _jsonable(pcd_result),
    }


def build_runtime_scene_from_real_camera(
    *,
    color_image_path: Path,
    depth_image_path: Path,
    objects: list[str],
    scene_out: Path = DEFAULT_SCENE_OUT,
    report_out: Path = DEFAULT_REPORT_OUT,
    observation_out: Path = DEFAULT_OBSERVATION_OUT,
    grounded_sam2_root: Path = DEFAULT_GROUNDED_SAM2_ROOT,
    target_coordinate_system: str = "world",
    downsample_scale: float = 1.0,
    mesh_files: dict[str, str | Path] | None = None,
    settle_steps: int = 1500,
) -> dict[str, Any]:
    color_img = _read_color(color_image_path)
    depth_img = _read_depth(depth_image_path)
    if color_img.shape[:2] != depth_img.shape[:2]:
        raise ValueError(f"Color/depth shape mismatch: color={color_img.shape[:2]}, depth={depth_img.shape[:2]}")

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_intrinsics_path = output_dir / "buquan_sim_intrinsics.json"
    sim_intrinsics_path.write_text(
        json.dumps(
            {
                "left": {
                    **DEFAULT_CAMERA_INTRINSICS,
                    "width": int(color_img.shape[1]),
                    "height": int(color_img.shape[0]),
                },
                "right": {
                    **DEFAULT_CAMERA_INTRINSICS,
                    "width": int(color_img.shape[1]),
                    "height": int(color_img.shape[0]),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    segmenter = GroundedSAM2Segmenter(
        grounded_sam2_root=str(grounded_sam2_root),
        box_threshold=0.2,
        text_threshold=0.2,
        device=None,
        multimask_output=True,
    )

    object_positions: dict[str, list[float]] = {}
    object_reports: dict[str, Any] = {}
    completed_mesh_reports: dict[str, Any] = {}
    mesh_quats: dict[str, list[float]] = {}
    for object_name in objects:
        position, object_report = _detect_object_position(
            segmenter=segmenter,
            color_image_path=color_image_path,
            color_img=color_img,
            depth_img=depth_img,
            object_name=object_name,
            output_dir=output_dir,
            target_coordinate_system=target_coordinate_system,
            downsample_scale=downsample_scale,
        )
        object_positions[object_name] = position
        object_reports[object_name] = object_report

        completed_mesh_reports[object_name] = build_buquan_completed_stl(
            object_name=object_name,
            color_image_path=color_image_path,
            depth_image_path=depth_image_path,
            mask_path=object_report["buquan_clean_mask_path"],
            camera="left",
        )
        mesh_quats[object_name.replace(" ", "_")] = _camera_to_target_mesh_quat(target_coordinate_system)

    scene_path = generate_scene(object_positions, scene_out=scene_out, mesh_quats=mesh_quats)
    settled_positions = _settled_object_positions(Path(scene_path), objects, int(settle_steps))
    observation = _build_initial_observation(
        scene_id=Path(scene_out).stem,
        color_image_path=color_image_path,
        depth_image_path=depth_image_path,
        objects=objects,
        object_positions=object_positions,
        settled_object_positions=settled_positions,
        completed_mesh_reports=completed_mesh_reports,
        mesh_quats=mesh_quats,
        target_coordinate_system=target_coordinate_system,
        scene_out=Path(scene_path),
        report_out=report_out,
    )
    report = {
        "status": "success",
        "color_image_path": str(color_image_path.resolve()),
        "depth_image_path": str(depth_image_path.resolve()),
        "scene_out": str(scene_path.resolve()),
        "observation_out": str(Path(observation_out).resolve()),
        "objects": objects,
        "object_positions_m": object_positions,
        "settle_steps": int(settle_steps),
        "settled_object_positions_m": settled_positions,
        "camera_intrinsics": DEFAULT_CAMERA_INTRINSICS,
        "camera_extrinsics": DEFAULT_CAMERA_EXTRINSICS,
        "grounded_sam2_root": str(Path(grounded_sam2_root).resolve()),
        "target_coordinate_system": target_coordinate_system,
        "downsample_scale": float(downsample_scale),
        "mesh_rotation_mode": MESH_ROTATION_MODE,
        "mesh_quats": mesh_quats,
        "completed_mesh_reports": completed_mesh_reports,
        "object_reports": object_reports,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    observation_out.parent.mkdir(parents=True, exist_ok=True)
    observation_out.write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Galaxea MuJoCo scene from one real RGB-D camera frame.")
    parser.add_argument("--color", default=str(DEFAULT_COLOR_IMAGE), help="Real camera color image path.")
    parser.add_argument("--depth", default=str(DEFAULT_DEPTH_IMAGE), help="Real camera metric depth path: uint16 PNG or NPY.")
    parser.add_argument("--objects", nargs="+", default=list(DEFAULT_OBJECTS), help="Object classes to detect.")
    parser.add_argument("--scene-out", default=str(DEFAULT_SCENE_OUT), help="Output MuJoCo XML path.")
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_OUT), help="Output JSON report path.")
    parser.add_argument("--observation-out", default=str(DEFAULT_OBSERVATION_OUT), help="Output task-start field observation JSON path.")
    parser.add_argument("--grounded-sam2-root", default=str(DEFAULT_GROUNDED_SAM2_ROOT), help="External or internal Grounded-SAM-2 root.")
    parser.add_argument("--coordinate-system", default="world", choices=["camera", "base", "world"])
    parser.add_argument("--downsample-scale", type=float, default=1.0)
    parser.add_argument("--settle-steps", type=int, default=1500, help="MuJoCo steps used to record settled object poses after scene generation.")
    parser.add_argument("--print-json", action="store_true", help="Print the full JSON report to stdout.")
    args = parser.parse_args()

    objects = [str(item).strip().lower().rstrip(".") for item in args.objects if str(item).strip()]
    result = build_runtime_scene_from_real_camera(
        color_image_path=_resolve_path(args.color),
        depth_image_path=_resolve_path(args.depth),
        objects=objects,
        scene_out=_resolve_path(args.scene_out),
        report_out=_resolve_path(args.report_out),
        observation_out=_resolve_path(args.observation_out),
        grounded_sam2_root=_resolve_path(args.grounded_sam2_root),
        target_coordinate_system=args.coordinate_system,
        downsample_scale=float(args.downsample_scale),
        settle_steps=int(args.settle_steps),
    )
    if args.print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"status: {result['status']}")
    print(f"coordinate_system: {result['target_coordinate_system']}")
    print("object_positions_m:")
    for object_name, position in result["object_positions_m"].items():
        print(f"  {object_name}: x={position[0]:.6f}, y={position[1]:.6f}, z={position[2]:.6f}")
    print(f"scene_out: {result['scene_out']}")
    print(f"report_out: {_resolve_path(args.report_out)}")


if __name__ == "__main__":
    main()
