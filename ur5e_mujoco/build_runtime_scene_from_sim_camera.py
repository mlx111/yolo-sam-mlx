from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experience_system.memory.v3 import MemoryV3Entry, MemoryV3Library, ReconstructionArtifacts, SceneInfo, TaskInfo
from ur5e_mujoco.runtime_scene import adjust_scene_support_heights, generate_scene


INPUTS_DIR = ROOT_DIR / "inputs"
OUTPUTS_DIR = ROOT_DIR / "outputs"
RUNTIME_DIR = ROOT_DIR / "runtime_assets"
DEFAULT_JSON_PATH = RUNTIME_DIR / "runtime_pose_calibration.json"
DEFAULT_XML = ROOT_DIR / "assets" / "scenes" / "scene2.xml"
DEFAULT_RUNTIME_SCENE_DIR = ROOT_DIR / "scene"
DEFAULT_SCENE_OUT = DEFAULT_RUNTIME_SCENE_DIR / "sim_camera_scene.xml"
DEFAULT_SCENE_REFINED_OUT = DEFAULT_RUNTIME_SCENE_DIR / "sim_camera_scene_refined.xml"
DEFAULT_REFINED_POSE_JSON = RUNTIME_DIR / "left_view_refined_pose.json"
DEFAULT_EXPERIENCE_LIB = ROOT_DIR / "results" / "experience_library.json"
DEFAULT_OBJECTS = ["apple", "pear"]
CAMERA_BY_SIDE = {"left": "cam1", "right": "cam2"}
BODY_BY_OBJECT = {"apple": "apple0", "pear": "pear0"}
MESH_BY_OBJECT = {
    "apple": ROOT_DIR / "assets" / "fruit" / "stl" / "apple.stl",
    "pear": ROOT_DIR / "assets" / "fruit" / "stl" / "pear.stl",
}
MJ_CAMERA_FROM_CV = np.diag([1.0, -1.0, -1.0])
DEPTH_TRUNC_M = 6.0
MASK_ERODE_PIXELS = 1
POSITION_TRIM_QUANTILES = (0.02, 0.98)
SUPPORT_CLEARANCE_M = 0.001


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (ROOT_DIR / path).resolve()


def _matrix_to_quat_wxyz(rotation_matrix: np.ndarray) -> list[float]:
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def _camera_intrinsics_from_fovy(width: int, height: int, fovy_deg: float) -> dict[str, float | int]:
    fy = height / (2.0 * np.tan(np.deg2rad(float(fovy_deg)) / 2.0))
    return {
        "fx": float(fy),
        "fy": float(fy),
        "cx": float((width) / 2.0),
        "cy": float((height) / 2.0),
        "width": int(width),
        "height": int(height),
    }


def _load_rgb(side: str) -> np.ndarray:
    path = INPUTS_DIR / f"c{side}001.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")
    return image


def _load_depth_meters(side: str) -> np.ndarray:
    npy_path = INPUTS_DIR / f"d{side}001.npy"
    if npy_path.exists():
        depth = np.load(npy_path)
        depth = np.asarray(depth, dtype=np.float64)
        if depth.ndim != 2:
            raise ValueError(f"Depth npy must be HxW: {npy_path}, got shape={depth.shape}")
        return depth

    png_path = INPUTS_DIR / f"d{side}001.png"
    depth_png = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if depth_png is None:
        raise FileNotFoundError(f"Failed to read depth image: {png_path}")
    if depth_png.ndim == 3:
        raise ValueError(f"Depth PNG must be single-channel metric depth, not color preview: {png_path}")
    if not np.issubdtype(depth_png.dtype, np.integer):
        raise ValueError(f"Depth PNG must use an integer dtype: {png_path}, got {depth_png.dtype}")
    if int(np.max(depth_png)) <= 255:
        raise ValueError(
            f"Depth PNG looks like a normalized preview, not metric depth: {png_path}, max={int(np.max(depth_png))}. "
            f"Use d{side}001.npy or a uint16 millimeter PNG."
        )
    return np.asarray(depth_png, dtype=np.float64) / 1000.0


def _load_mask(side: str, object_name: str, target_shape: tuple[int, int]) -> np.ndarray:
    path = INPUTS_DIR / f"{side}_mask_{object_name}.png"
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if mask.shape[:2] != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(str(path), mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
    binary = mask > 0
    if int(np.count_nonzero(binary)) == 0:
        raise ValueError(f"Mask is empty: {path}")
    return binary


def _erode_mask(mask: np.ndarray, pixels: int = MASK_ERODE_PIXELS) -> np.ndarray:
    if pixels <= 0:
        return mask
    kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
    return eroded if int(np.count_nonzero(eroded)) >= 8 else mask


def _backproject_masked_depth(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict[str, float | int],
) -> np.ndarray:
    valid = mask & np.isfinite(depth_m) & (depth_m > 0) & (depth_m <= DEPTH_TRUNC_M)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)
    v, u = np.where(valid)
    z = depth_m[valid].astype(np.float64)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    return np.column_stack([x, y, z])


def _load_model_and_camera_data(xml_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing XML scene: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _camera_config(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str, width: int, height: int) -> dict[str, object]:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera not found: {camera_name}")
    rotation_mj_from_cam = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    rotation_world_from_cv = rotation_mj_from_cam @ MJ_CAMERA_FROM_CV
    translation = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
    intrinsics = _camera_intrinsics_from_fovy(width, height, float(model.cam_fovy[cam_id]))
    return {
        "name": camera_name,
        "translation_mj": translation,
        "quat_wxyz_mj": _matrix_to_quat_wxyz(rotation_mj_from_cam),
        "rotation_matrix_mj_from_cam": rotation_world_from_cv,
        "rotation_matrix_world_from_cam": rotation_world_from_cv,
        "intrinsics": intrinsics,
        "fovy_deg": float(model.cam_fovy[cam_id]),
    }


def _cv_points_to_world(points_cv: np.ndarray, rotation_world_from_cv: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (rotation_world_from_cv @ points_cv.T).T + translation


def _support_plane_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        return float(data.geom_xpos[floor_id][2])
    return 0.0


def _support_adjust_scene(scene_path: str | Path, mesh_files: dict[str, Path], support_z: float) -> dict[str, dict[str, object]]:
    return adjust_scene_support_heights(
        scene_path=scene_path,
        mesh_paths=mesh_files,
        body_by_object=BODY_BY_OBJECT,
        support_z=support_z,
        clearance=SUPPORT_CLEARANCE_M,
    )


def _trimmed_xy_mean(points_world: np.ndarray) -> np.ndarray:
    if points_world.ndim != 2 or points_world.shape[1] != 3 or len(points_world) < 8:
        raise ValueError("Invalid point cloud for center estimation.")
    lo_q, hi_q = POSITION_TRIM_QUANTILES
    lo = np.quantile(points_world, lo_q, axis=0)
    hi = np.quantile(points_world, hi_q, axis=0)
    keep = np.all((points_world >= lo) & (points_world <= hi), axis=1)
    trimmed = points_world[keep]
    if len(trimmed) < 8:
        trimmed = points_world
    return np.asarray(trimmed[:, :2].mean(axis=0), dtype=np.float64)


def _robust_object_position(point_sets: list[np.ndarray], support_z: float) -> list[float]:
    if not point_sets:
        raise ValueError("No point clouds available for object position estimation.")
    xy_estimates = [_trimmed_xy_mean(points) for points in point_sets if len(points) >= 8]
    if not xy_estimates:
        raise ValueError("No valid point clouds available for object position estimation.")
    xy = np.mean(np.vstack(xy_estimates), axis=0)
    return [float(xy[0]), float(xy[1]), max(0.0, float(support_z))]


def _write_sim_intrinsics(camera_configs: dict[str, dict[str, object]]) -> Path:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        side: {
            key: float(value) if key not in {"width", "height"} else int(value)
            for key, value in dict(config["intrinsics"]).items()
        }
        for side, config in camera_configs.items()
    }
    path = RUNTIME_DIR / "sim_camera_intrinsics.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_sim_point_clouds(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    objects: list[str],
) -> tuple[dict[str, list[float]], dict[str, dict[str, object]], dict[str, dict[str, list[float]]]]:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    object_world_points: dict[str, list[np.ndarray]] = {name: [] for name in objects}
    camera_configs: dict[str, dict[str, object]] = {}
    support_z = _support_plane_z(model, data)

    for side, camera_name in CAMERA_BY_SIDE.items():
        rgb = _load_rgb(side)
        depth = _load_depth_meters(side)
        if depth.shape[:2] != rgb.shape[:2]:
            raise ValueError(f"RGB/depth shape mismatch for {side}: rgb={rgb.shape[:2]}, depth={depth.shape[:2]}")
        height, width = depth.shape[:2]
        camera_config = _camera_config(model, data, camera_name, width, height)
        camera_configs[side] = camera_config
        intrinsics = camera_config["intrinsics"]
        rotation_world_from_cv = np.asarray(camera_config["rotation_matrix_world_from_cam"], dtype=np.float64)
        translation = np.asarray(camera_config["translation_mj"], dtype=np.float64)

        for object_name in ["roboticarm", *objects]:
            mask = _erode_mask(_load_mask(side, object_name, depth.shape[:2]))
            points_cv = _backproject_masked_depth(depth, mask, intrinsics)
            if len(points_cv) < 8:
                raise ValueError(f"Too few valid depth points for {side}:{object_name}")
            points_world = _cv_points_to_world(points_cv, rotation_world_from_cv, translation)

            if object_name in objects:
                np.save(OUTPUTS_DIR / f"raw_{side}_{object_name}.npy", points_cv)
                np.save(OUTPUTS_DIR / f"raw_{side}_{object_name}_world.npy", points_world)
                object_world_points[object_name].append(points_world)
            else:
                np.save(OUTPUTS_DIR / f"raw_{side}_{object_name}_world.npy", points_world)

    object_positions = {
        object_name: _robust_object_position(point_sets, support_z)
        for object_name, point_sets in object_world_points.items()
        if point_sets
    }
    scene_camera_poses = {
        config["name"]: {
            "pos": [float(v) for v in np.asarray(config["translation_mj"], dtype=np.float64)],
            "quat": [float(v) for v in config["quat_wxyz_mj"]],
        }
        for config in camera_configs.values()
    }
    return object_positions, camera_configs, scene_camera_poses


def _calibration_payload(
    object_positions: dict[str, list[float]],
    camera_configs: dict[str, dict[str, object]],
) -> dict[str, object]:
    camera_poses = {}
    for side, config in camera_configs.items():
        intrinsics = config["intrinsics"]
        camera_poses[side] = {
            "translation_mj": [float(v) for v in np.asarray(config["translation_mj"], dtype=np.float64)],
            "quat_wxyz": [float(v) for v in config["quat_wxyz_mj"]],
            "rotation_matrix_mj_from_cam": [[float(v) for v in row] for row in np.asarray(config["rotation_matrix_mj_from_cam"])],
            "rotation_matrix_world_from_cam": [[float(v) for v in row] for row in np.asarray(config["rotation_matrix_world_from_cam"])],
            "intrinsics": {key: float(value) if key not in {"width", "height"} else int(value) for key, value in intrinsics.items()},
            "fovy_deg": float(config["fovy_deg"]),
            "camera_model": "mujoco_opengl_converted_to_cv_pinhole",
        }
    return {
        "world_axes": {"world_frame_source": "mujoco_sim_camera"},
        "camera_poses": camera_poses,
        "robot_positions": {"arm_base_world": [0.0, 0.0, 0.0]},
        "object_positions": {f"{name}_world": [float(v) for v in pos] for name, pos in object_positions.items()},
        "relative_positions": {f"{name}_minus_arm": [float(v) for v in pos] for name, pos in object_positions.items()},
        "relative_position_source": "sim camera metric depth and masks, robust trimmed xy mean, support-plane z",
    }


def _patch_left_intrinsics(intrinsics: dict[str, float | int]) -> dict[str, float | int]:
    return {
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "cx": float(intrinsics["cx"]),
        "cy": float(intrinsics["cy"]),
        "width": int(intrinsics["width"]),
        "height": int(intrinsics["height"]),
    }


def _build_refined_pose_json(
    *,
    object_positions: dict[str, list[float]],
    scene_camera_poses: dict[str, dict[str, list[float]]],
    initial_quats: dict[str, list[float]],
    objects: list[str],
) -> str:
    raise RuntimeError(
        "Refined pose postprocess is not internalized for UR5e yet. "
        "Run without --refine to generate the self-contained UR5e runtime scene."
    )


def _run_apply_refined_pose() -> None:
    raise RuntimeError(
        "Refined pose postprocess is not internalized for UR5e yet. "
        "Run without --refine to generate the self-contained UR5e runtime scene."
    )


def _resolve_local_meshes(objects: list[str]) -> dict[str, Path]:
    mesh_files: dict[str, Path] = {}
    for object_name in objects:
        normalized = str(object_name).strip().lower()
        mesh_path = MESH_BY_OBJECT.get(normalized)
        if mesh_path is None:
            continue
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing UR5e local mesh for {normalized}: {mesh_path}")
        mesh_files[normalized] = mesh_path
    return mesh_files


def build_runtime_scene_from_sim_camera(
    *,
    xml_path: Path,
    objects: list[str],
    scene_out: Path,
    refine: bool,
    start_server: bool,
    experience_lib_path: Path | None = None,
) -> dict[str, object]:
    model, data = _load_model_and_camera_data(xml_path)
    support_z = _support_plane_z(model, data)
    object_positions, camera_configs, scene_camera_poses = _write_sim_point_clouds(
        model=model,
        data=data,
        objects=objects,
    )
    calibration = _calibration_payload(object_positions, camera_configs)
    DEFAULT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_JSON_PATH.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")

    _patch_left_intrinsics(camera_configs["left"]["intrinsics"])
    sim_intrinsics_path = _write_sim_intrinsics(camera_configs)

    mesh_files = _resolve_local_meshes(objects)
    mesh_quats: dict[str, list[float]] = {}
    print({"runtime_meshes": {name: str(path) for name, path in mesh_files.items()}})

    initial_quats = {object_name: [1.0, 0.0, 0.0, 0.0] for object_name in objects}
    camera_facing_debug = {
        object_name: {
            "status": "skipped",
            "reason": "UR5e internal sim-camera builder uses fixed mesh orientation by default.",
        }
        for object_name in objects
    }

    base_scene_path = generate_scene(
        object_positions,
        camera_poses=scene_camera_poses,
        object_quats=initial_quats,
        mesh_quats=mesh_quats,
        scene_out=str(scene_out),
    )
    support_adjustments = {
        "base": _support_adjust_scene(base_scene_path, mesh_files, support_z),
    }

    refined_pose_json = None
    refined_scene_path = None
    if refine:
        refined_pose_json = _build_refined_pose_json(
            object_positions=object_positions,
            scene_camera_poses=scene_camera_poses,
            initial_quats=initial_quats,
            objects=objects,
        )
        _run_apply_refined_pose()
        refined_scene_path = str(DEFAULT_SCENE_REFINED_OUT.resolve())
        support_adjustments["refined"] = _support_adjust_scene(DEFAULT_SCENE_REFINED_OUT, mesh_files, support_z)

    if start_server:
        from grasp_fastapi_completion_v4 import start as start_v4_server

        start_v4_server()

    result = {
        "status": "success",
        "xml_source": str(xml_path.resolve()),
        "scene_out": str(Path(base_scene_path).resolve()),
        "scene_out_refined": refined_scene_path,
        "refined_pose_json": refined_pose_json,
        "runtime_pose_calibration_path": str(DEFAULT_JSON_PATH.resolve()),
        "object_positions": object_positions,
        "object_quats": initial_quats,
        "camera_poses": scene_camera_poses,
        "camera_facing_local_axis": camera_facing_debug,
        "mesh_quats": mesh_quats,
        "support_height_adjustments": support_adjustments,
        "left_intrinsics": camera_configs["left"]["intrinsics"],
        "right_intrinsics": camera_configs["right"]["intrinsics"],
        "sim_intrinsics_json": str(sim_intrinsics_path.resolve()),
    }
    if experience_lib_path is not None:
        lib = MemoryV3Library.load(experience_lib_path)
        entry = MemoryV3Entry(
            episode_type="scene_reconstruction",
            source="simulation",
            scene=SceneInfo(
                objects=objects or [],
                camera_view="sim_camera",
                scene_name=Path(scene_out).stem,
            ),
            task=TaskInfo(
                name="scene_reconstruction",
                stage="reconstruction",
                scene_name=Path(scene_out).stem,
            ),
            reconstruction_artifacts=ReconstructionArtifacts(
                reconstruction_type="sim_camera",
                object_positions=object_positions,
                object_quats=initial_quats,
                camera_poses=scene_camera_poses,
                scene_out=str(Path(base_scene_path).resolve()),
                scene_out_refined=refined_scene_path or "",
                runtime_pose_calibration_path=str(DEFAULT_JSON_PATH.resolve()),
                sim_intrinsics_json=str(sim_intrinsics_path.resolve()),
                mesh_quats=mesh_quats,
                support_height_adjustments=support_adjustments,
                reconstruction_signature=f"sim_camera:{xml_path.stem}:{','.join(objects)}",
            ),
            summary=f"基于仿真相机完成场景重建，objects={','.join(objects)}。",
        )
        # Override auto-derived status — this is a reconstruction record, not a recovery trial
        entry.status = "success"
        entry.result.success = True
        entry.result.task_success = True
        lib.upsert(entry)
        experience_lib_path.parent.mkdir(parents=True, exist_ok=True)
        lib.save(experience_lib_path)
        result["experience_entry_id"] = entry.experience_id
        result["experience_lib_path"] = str(experience_lib_path.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build runtime scene from simulator RGB/depth/masks using simulator camera parameters.")
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="Source MuJoCo scene XML that produced the images.")
    parser.add_argument("--objects", nargs="+", default=DEFAULT_OBJECTS)
    parser.add_argument("--scene-out", default=str(DEFAULT_SCENE_OUT))
    parser.add_argument("--refine", action="store_true", help="Run legacy refined-pose postprocess if its local assets are available.")
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--experience-lib", default=str(DEFAULT_EXPERIENCE_LIB), help="经验库 JSON 保存路径")
    args = parser.parse_args()

    objects = [str(obj).strip() for obj in args.objects if str(obj).strip()]
    result = build_runtime_scene_from_sim_camera(
        xml_path=_resolve_path(args.xml),
        objects=objects,
        scene_out=_resolve_path(args.scene_out),
        refine=bool(args.refine),
        start_server=args.start_server,
        experience_lib_path=_resolve_path(args.experience_lib) if args.experience_lib else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
